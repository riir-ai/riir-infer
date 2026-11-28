//! Issue 1253 T5.5 (plan 639) — the GGUF-embedded MTP draft block's
//! METAL DECODE ECONOMICS CELL: the draft block on the Metal serving
//! posture over the real Bonsai-2.5 `-mtp` twin, batched verify per
//! the Issue-665 sequential-speculative path.
//!
//! SEAM CORRECTION (recorded in plan 639 before this cell ran — the
//! claim commit's seam-map said "one chunk pass of the prefill
//! mechanics"; the substrate refutes that shape for GPU verify):
//! - Issue 652's chunkwise-parallel Delta rule (the batched-verify G1)
//!   FAILED on GPU — the chunk path's GDN layers stay per-token
//!   sequential anyway, so a "prefill-chunk verify" buys nothing.
//! - The shipped micro-batch rung is `forward_speculative_verify`
//!   (Issue 665): K queued decode passes with ONE sync, K+1
//!   position-aligned logits (`SPEC_MAX_K = 8` == the Apple card's
//!   chain ceiling), plus `checkpoint_speculative` /
//!   `rollback_speculative` (one batched read / one batched write;
//!   KV rows beyond the cursor are stale-but-unread — the positional
//!   law the seam-map cited). This example CONSUMES that surface and
//!   adds no forward path of its own.
//!
//! ROUND SHAPE (the keyed walk — `draft_chain` proposes the echo slot
//! as chain[0]; the preset's min-prob gate applies uniformly including
//! step 0, so a starved chain means a plain decode round):
//! 1. free row: forward the last landed token at its slot — its argmax
//!    verdicts chain[0]; its final hidden seeds the next round when the
//!    round ends plain (norm_x stays primal — the Plan-602-B3 law — so
//!    the GPU seed needs NO inverse rotation, unlike the CPU harness).
//! 2. `checkpoint_speculative` + `forward_speculative_verify(chain)`:
//!    out[0] == the free row's logits (pool-rotation witness, asserted),
//!    out[i] verdicts chain[i] (i = 1..len), out[len] is the bonus row
//!    (lands free on a full accept).
//! 3. reject at r: `rollback_speculative`, re-forward the accepted
//!    prefix chain[0..r] at their slots (the reference consumer's state
//!    discipline; the correction lands unforwarded), the last
//!    re-forward's final hidden seeds the next round.
//!    full accept: the bonus lands; `read_hidden_state` (the batch's
//!    last row's pre-norm x) + the final RMSNorm re-derives the seed —
//!    no extra forward.
//!
//! The CPU draft step (T5.2) runs ONCE per round and is counted in
//! the wall (the plan's own law; no GPU draft block in this cell).
//!
//! Losslessness: greedy specdec lands exactly the plain greedy stream —
//! the G1 arm walks both from the same reset+prefill state and asserts
//! byte equality (the qwen38_specdec_t1 G1-T1 pattern on this lane),
//! and refuses to report until at least one rollback was exercised.
//! The ECONOMICS run is QUIET-WINDOW-ONLY per plan 639:
//! MTP_ECON_PROVENANCE (bench_preflight.sh's PROVENANCE line) is
//! REQUIRED for a bar verdict — MTP_ECON_SMOKE=1 is the
//! correctness-smoke escape (no verdict printed).
//!
//! ```text
//! cargo run --release -p riir-infer-dist \
//!   --features mtp_draft_block,bonsai2_hadamard,gpu,speculative_decode \
//!   --example mtp_block_economics_metal
//!   # env: MTP_GGUF, MTP_CHAT_TOKENS, MTP_ECON_PROMPT (256),
//!   #      MTP_ECON_GEN (256), MTP_ECON_G1_STEPS (48),
//!   #      MTP_ECON_ARM (all|g1|ordinary|nvidia|apple), MTP_ECON_SMOKE,
//!   #      MTP_ECON_PROVENANCE
//! ```

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use riir_infer_core::deltanet::ternary_weights::QwenDeltaNetTernaryWeights;
use riir_infer_core::gguf_loader::{
    GgufFile, load_mtp_draft_block, load_qwen_deltanet_ternary_from_gguf,
};
use riir_infer_core::types::rmsnorm_with_gamma_eps;
use riir_infer_gpu::ternary_deltanet_gpu_forward::SpeculativeCheckpoint;
use riir_infer_gpu::{CubeCLContext, TernaryDeltanetGpuForward};
use riir_infer_dist::argmax_first;
use riir_infer_dist::spec_mtp_block::{MtpDraftBlockDrafter, MtpDraftPreset};

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Per-arm accounting. `syncs` counts GPU readback events conservatively
/// (forward_token = 1, with_final_hidden = 2, the batch verify's single
/// batched read = 1, checkpoint = 1, hidden read = 1; the rollback write
/// is non-blocking and counts 0).
#[derive(Default)]
struct RoundStats {
    rounds: usize,
    plain_rounds: usize,
    full_accept_rounds: usize,
    rejected_rounds: usize,
    rollbacks: usize,
    bonus_tokens: usize,
    drafts_proposed: usize,
    drafts_accepted: usize,
    /// Per-slot (chain index) hits / verdicts — the serving-posture
    /// bar (a) re-read (slot 0 = the echo slot). A slot is verdicted
    /// only when every earlier slot hit, so deep slots vanish when the
    /// chain collapses — the counts ARE the depth story.
    slot_hits: [usize; 8],
    slot_verdicts: [usize; 8],
    forwards: usize,
    syncs: usize,
    draft_steps: usize,
    landed: usize,
}

impl RoundStats {
    fn accept_rate(&self) -> f64 {
        if self.drafts_proposed == 0 {
            0.0
        } else {
            self.drafts_accepted as f64 / self.drafts_proposed as f64
        }
    }
}

/// One verify round on `fwd` at the keyed shape. `h_seed` = the primal
/// post-final-norm hidden at slot `pos - 1` (the arm-5 hand-off);
/// `last` = the token landed at slot `pos - 1`. Returns
/// `(seed_hidden, next_pos, landed_tokens)` — the seed for the next
/// round (the primal hidden at the round's last LANDED slot minus
/// one), the position the next round's free row runs at (= the last
/// landed slot), and every token the round landed IN SLOT ORDER (the
/// free row's argmax first — a full accept lands feed0 + the accepted
/// prefix + the bonus; a reject lands feed0 + the prefix + the
/// correction; a plain round lands feed0 alone).
struct RoundCtx<'a> {
    fwd: &'a mut TernaryDeltanetGpuForward,
    weights: &'a QwenDeltaNetTernaryWeights,
    posture: &'a RoundPosture,
    drafter: &'a mut MtpDraftBlockDrafter,
}

fn verify_round(
    ctx: &mut RoundCtx<'_>,
    h_seed: &[f32],
    last: u32,
    pos: usize,
    stats: &mut RoundStats,
) -> (Vec<f32>, usize, Vec<u32>) {
    let RoundCtx {
        fwd,
        weights,
        posture,
        drafter,
    } = ctx;
    let n = posture.n_embd;
    let eps = posture.eps;
    // 1. The free row: verdicts chain[0]; its final hidden is the seed
    //    when the round ends plain.
    fwd.set_pos(pos);
    fwd.set_input_token(weights, last as usize);
    let (lg0, h_free) = fwd.forward_token_with_final_hidden();
    stats.forwards += 1;
    stats.syncs += 2;
    let feed0 = argmax_first(&lg0);

    // 2. The chain (the CPU draft step — counted in the wall).
    drafter.seed(h_seed, last, pos).expect("seed (metal)");
    stats.draft_steps += 1;
    let mut chain_buf = Vec::with_capacity(drafter.preset().k);
    drafter.draft_chain_into(&mut chain_buf);
    if let Some(e) = drafter.take_error() {
        panic!("draft chain failed at the serving posture: {e}");
    }
    let next_free = pos + 1;
    if chain_buf.is_empty() {
        // Plain decode round.
        stats.rounds += 1;
        stats.plain_rounds += 1;
        stats.landed += 1;
        return (h_free, next_free, vec![feed0]);
    }

    let chain: Vec<usize> = chain_buf.iter().map(|(t, _)| *t as usize).collect();
    let k = chain.len();
    assert!(
        (1..=8).contains(&k),
        "chain length {k} outside the SPEC_MAX_K window"
    );

    // 3. Checkpoint + the batched verify (K queued rows, one sync).
    let cp: SpeculativeCheckpoint = fwd.checkpoint_speculative();
    stats.syncs += 1;
    let outs = fwd.forward_speculative_verify(weights, &chain);
    stats.forwards += k;
    stats.syncs += 1;
    assert_eq!(
        outs.len(),
        k + 1,
        "verify returns K+1 position-aligned logits"
    );
    // The pool-rotation witness: out[0] is the free row's logits.
    assert_eq!(outs[0], lg0, "verify out[0] != the free row's logits");

    // 4. The walk: verdict_m = argmax(outs[m]) vs chain[m].
    let mut accept_len = 0usize;
    let mut correction: Option<u32> = None;
    for m in 0..k {
        let g = argmax_first(&outs[m]);
        stats.slot_verdicts[m] += 1;
        stats.drafts_proposed += 1;
        if g == chain_buf[m].0 {
            stats.slot_hits[m] += 1;
            stats.drafts_accepted += 1;
            accept_len += 1;
        } else {
            correction = Some(g);
            break;
        }
    }

    if accept_len == k {
        // Full accept: the bonus row lands free (argmax(outs[k]) is the
        // token for slot next_free + k).
        let bonus = argmax_first(&outs[k]);
        stats.rounds += 1;
        stats.full_accept_rounds += 1;
        stats.bonus_tokens += 1;
        let mut landed: Vec<u32> = chain_buf.iter().map(|(t, _)| *t).collect();
        landed.push(bonus);
        stats.landed += landed.len();
        // Seed = the batch's last row's post-final-norm primal hidden
        // (slot next_free + k - 1), re-derived from the pre-norm x —
        // no extra forward. The next free row runs at next_free + k,
        // feeding the bonus.
        let mut h = fwd.read_hidden_state();
        stats.syncs += 1;
        rmsnorm_with_gamma_eps(&mut h[..n], &weights.final_norm, eps);
        return (h, next_free + k, landed);
    }

    let r = accept_len;
    let correction = correction.expect("reject recorded no correction");
    // Reject at r: rollback + repair the accepted prefix. Slot map:
    // feed0 landed at next_free (slot pos+1), chain[m] at slot
    // next_free+m for m in 0..r, and the correction lands at slot
    // next_free + r — where the next free row runs, feeding the
    // correction.
    fwd.rollback_speculative(&cp);
    stats.rollbacks += 1;
    for (m, &tok) in chain.iter().enumerate().take(r) {
        fwd.set_pos(next_free + m);
        fwd.set_input_token(weights, tok);
        if m + 1 == r {
            // The last repair row (slot next_free + r - 1) seeds the
            // next round (the seed contract wants h@(P-1) with
            // P = next_free + r).
            let (_, h) = fwd.forward_token_with_final_hidden();
            stats.syncs += 2;
            stats.rounds += 1;
            stats.rejected_rounds += 1;
            let mut landed: Vec<u32> = chain_buf.iter().take(r).map(|(t, _)| *t).collect();
            landed.push(correction);
            stats.landed += landed.len();
            return (h, next_free + r, landed);
        }
        fwd.forward_token();
        stats.forwards += 1;
        stats.syncs += 1;
    }
    // r == 0: the correction IS feed0 (argmax(outs[0]) = the free
    // logits) — the same shape as the plain round; the free row's
    // hidden seeds.
    stats.rounds += 1;
    stats.rejected_rounds += 1;
    stats.landed += 1;
    (h_free, next_free, vec![correction])
}

struct ArmOut {
    tok_s: f64,
    stats: RoundStats,
    /// The landed token stream in order (the G1 compare rides it).
    stream: Vec<u32>,
}

/// The scalars the verify loop reads every round — one bundled
/// posture instead of an argument list that grows with each seam.
struct RoundPosture {
    n_embd: usize,
    eps: f64,
    block_cap: usize,
}

/// The specdec arm: prefill once, then verify rounds to `gen_cap`
/// landed tokens. The stream starts with the prefill argmax, like the
/// plain arm. `preset` is applied before the run.
fn run_spec_arm(
    fwd: &mut TernaryDeltanetGpuForward,
    weights: &Arc<QwenDeltaNetTernaryWeights>,
    drafter: &mut MtpDraftBlockDrafter,
    preset: MtpDraftPreset,
    posture: &RoundPosture,
    prompt: &[usize],
    gen_cap: usize,
) -> ArmOut {
    let n = posture.n_embd;
    let block_cap = posture.block_cap;
    let eps = posture.eps;
    drafter.reset();
    drafter.set_preset(preset);
    fwd.reset_state();
    let logits = fwd.prefill(prompt);
    let pending = argmax_first(&logits);
    let mut stream = vec![pending];
    let mut stats = RoundStats::default();
    // The prefill's final hidden seeds round 0: the seed contract is
    // (h@(P-1), tok@P) with P = prompt.len() — h@(prompt-1) is the
    // prefill's own final hidden (read pre-norm x + the final norm) and
    // tok@prompt = `pending`.
    let mut h_seed = fwd.read_hidden_state();
    stats.syncs += 1;
    rmsnorm_with_gamma_eps(&mut h_seed[..n], &weights.final_norm, eps);
    let mut last = pending;
    let mut pos = prompt.len();
    let t0 = Instant::now();
    while stats.landed < gen_cap {
        let mut ctx = RoundCtx {
            fwd,
            weights,
            posture,
            drafter,
        };
        let (h, next_pos, landed) = verify_round(&mut ctx, &h_seed, last, pos, &mut stats);
        h_seed = h;
        last = *landed.last().expect("a round lands at least one token");
        pos = next_pos;
        stream.extend_from_slice(&landed);
        if pos + 8 >= block_cap {
            eprintln!("arm: position bound {block_cap} reached at {pos}");
            break;
        }
    }
    let wall = t0.elapsed();
    ArmOut {
        tok_s: stats.landed as f64 / wall.as_secs_f64(),
        stats,
        stream,
    }
}

/// The plain greedy arm: prefill once, then decode steps.
fn run_ordinary_arm(
    fwd: &mut TernaryDeltanetGpuForward,
    weights: &Arc<QwenDeltaNetTernaryWeights>,
    prompt: &[usize],
    gen_cap: usize,
) -> ArmOut {
    fwd.reset_state();
    let logits = fwd.prefill(prompt);
    let mut next = argmax_first(&logits);
    let mut stream = vec![next];
    let mut stats = RoundStats::default();
    let t0 = Instant::now();
    for _ in 0..gen_cap {
        fwd.set_input_token(weights, next as usize);
        let l = fwd.forward_token();
        stats.forwards += 1;
        stats.syncs += 1;
        stats.landed += 1;
        next = argmax_first(&l);
        stream.push(next);
    }
    let wall = t0.elapsed();
    ArmOut {
        tok_s: stats.landed as f64 / wall.as_secs_f64(),
        stats,
        stream,
    }
}

fn print_arm(name: &str, out: &ArmOut, gen_cap: usize) {
    let s = &out.stats;
    println!(
        "ARM {name}: {} landed (of {gen_cap} asked) = {:.2} tok/s — rounds {} \
         (plain {}, full-accept {}, rejected {}, rollbacks {}) | slots verdicted {} \
         hits {} (rate {:.3}) | bonus {} | forwards {} syncs {} draft-steps {}",
        s.landed,
        out.tok_s,
        s.rounds,
        s.plain_rounds,
        s.full_accept_rounds,
        s.rejected_rounds,
        s.rollbacks,
        s.drafts_proposed,
        s.drafts_accepted,
        s.accept_rate(),
        s.bonus_tokens,
        s.forwards,
        s.syncs,
        s.draft_steps,
    );
    let slots: Vec<String> = (0..8)
        .filter(|&m| s.slot_verdicts[m] > 0)
        .map(|m| {
            format!(
                "s{m} {}/{}={:.3}",
                s.slot_hits[m],
                s.slot_verdicts[m],
                s.slot_hits[m] as f64 / s.slot_verdicts[m] as f64
            )
        })
        .collect();
    if !slots.is_empty() {
        println!(
            "ARM {name} per-slot agreement (the bar (a) serving-posture re-read): {}",
            slots.join(" | ")
        );
    }
}

fn main() {
    let smoke = std::env::var("MTP_ECON_SMOKE").ok().as_deref() == Some("1");
    let arm_sel = std::env::var("MTP_ECON_ARM").unwrap_or_else(|_| "all".into());
    let prompt_len = if smoke {
        64
    } else {
        env_usize("MTP_ECON_PROMPT", 256)
    };
    let gen_cap = if smoke { 48 } else { env_usize("MTP_ECON_GEN", 256) };
    let g1_steps = if smoke { 24 } else { env_usize("MTP_ECON_G1_STEPS", 48) };

    let twin = PathBuf::from(
        std::env::var("MTP_GGUF")
            .unwrap_or_else(|_| "../riir-train/data/Ternary-Bonsai-2.5-27B-PQ2_0-mtp.gguf".into()),
    );
    let Some(twin) = twin.canonicalize().ok() else {
        eprintln!("SKIP: the 2.5 -mtp GGUF is not on this box (MTP_GGUF names it)");
        std::process::exit(2);
    };
    let tokens_path = PathBuf::from(
        std::env::var("MTP_CHAT_TOKENS")
            .unwrap_or_else(|_| "../riir-train/data/bonsai_mtp/tokens.chat.tokens".into()),
    );
    let Ok(tokens_bytes) = std::fs::read(&tokens_path) else {
        eprintln!(
            "SKIP: the chat token file is absent at {} — regenerate with: cargo run \
             --release -p riir-infer-dist --example mtp_chat_tokens",
            tokens_path.display()
        );
        std::process::exit(2);
    };
    assert!(
        tokens_bytes.as_chunks::<4>().1.is_empty(),
        "the chat token file is u32-LE"
    );
    let tokens: Vec<usize> = tokens_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b) as usize)
        .collect();
    let prompt_len = prompt_len.min(tokens.len());
    let prompt = &tokens[..prompt_len];
    eprintln!(
        "chat transcript: {} tokens from {} (smoke={smoke}, prompt {prompt_len}, gen {gen_cap})",
        tokens.len(),
        tokens_path.display()
    );

    let gguf = GgufFile::open(&twin).expect("open the real twin");
    let (config, weights) =
        load_qwen_deltanet_ternary_from_gguf(&gguf, "").expect("main stack loads");
    let weights = Arc::new(weights);
    let block = Arc::new(load_mtp_draft_block(&gguf, &config).expect("draft block admits"));

    // Position caps (the Issue-929 seam; the Issue-746 append-only KV
    // tripwire needs pos + k <= block_size): the decode horizon + the
    // chain + margin. The drafter's own cache rides the same cap.
    let cap = prompt_len + gen_cap + 8 + 64;
    let mut capped = config.clone();
    if capped.block_size > cap {
        capped.block_size = cap;
    }

    let ctx = CubeCLContext::new().expect("GPU init (Metal posture)");
    let mut fwd = TernaryDeltanetGpuForward::new(&ctx, &capped, &weights);
    eprintln!(
        "gpu forward ready: n_layer {} block_size {} (capped {})",
        config.n_layer, config.block_size, capped.block_size
    );

    let mut drafter = MtpDraftBlockDrafter::new(
        block,
        weights.clone(),
        capped.clone(),
        MtpDraftPreset::nvidia_draft3(),
    )
    .expect("seat the draft block");
    let posture = RoundPosture {
        n_embd: config.n_embd,
        eps: config.rms_norm_eps,
        block_cap: capped.block_size,
    };

    // ---------------- G1: the stream-identity gate ----------------
    if arm_sel == "all" || arm_sel == "g1" || smoke {
        let plain = run_ordinary_arm(&mut fwd, &weights, prompt, g1_steps);
        let spec = run_spec_arm(
            &mut fwd,
            &weights,
            &mut drafter,
            MtpDraftPreset::nvidia_draft3(),
            &posture,
            prompt,
            g1_steps,
        );
        let m = plain.stream.len().min(spec.stream.len());
        let mismatch = (0..m).find(|&i| plain.stream[i] != spec.stream[i]);
        assert!(
            mismatch.is_none(),
            "G1 STREAM IDENTITY FAILED at {mismatch:?}: plain {:?} vs specdec {:?}",
            mismatch.map(|i| &plain.stream[i..(i + 4).min(m)]),
            mismatch.map(|i| &spec.stream[i..(i + 4).min(m)]),
        );
        assert!(
            spec.stats.rollbacks > 0,
            "G1 exercised no rollback in {g1_steps} steps — the repair path is unproven \
             (raise MTP_ECON_G1_STEPS or check the acceptance wiring)"
        );
        println!(
            "G1 stream identity: {m} tokens byte-identical (rollbacks exercised: {}) — PASS",
            spec.stats.rollbacks
        );
        if arm_sel == "g1" {
            return;
        }
    }

    if !smoke && (arm_sel == "all" || arm_sel == "ordinary" || arm_sel == "nvidia" || arm_sel == "apple")
    {
        // The quiet-window law: an economics run must quote the
        // preflight PROVENANCE (the caller pipes it in).
        let provenance = std::env::var("MTP_ECON_PROVENANCE").unwrap_or_default();
        if provenance.is_empty() {
            eprintln!(
                "REFUSE: no MTP_ECON_PROVENANCE — quote bench_preflight.sh's PROVENANCE \
                 line (the quiet-window law binds the economics cell; MTP_ECON_SMOKE=1 is \
                 the correctness-smoke escape)"
            );
            std::process::exit(2);
        }
        println!("PROVENANCE: {provenance}");
    }

    // ---------------- the arms ----------------
    let mut ordinary = None;
    if arm_sel == "all" || arm_sel == "ordinary" {
        let out = run_ordinary_arm(&mut fwd, &weights, prompt, gen_cap);
        print_arm("ordinary", &out, gen_cap);
        ordinary = Some(out);
    }
    let mut nvidia = None;
    if arm_sel == "all" || arm_sel == "nvidia" {
        let out = run_spec_arm(
            &mut fwd,
            &weights,
            &mut drafter,
            MtpDraftPreset::nvidia_draft3(),
            &posture,
            prompt,
            gen_cap,
        );
        print_arm("draft(nvidia3)", &out, gen_cap);
        nvidia = Some(out);
    }
    if arm_sel == "all" || arm_sel == "apple" {
        let out = run_spec_arm(
            &mut fwd,
            &weights,
            &mut drafter,
            MtpDraftPreset::apple(8).expect("the K=8 preset"),
            &posture,
            prompt,
            gen_cap,
        );
        print_arm("draft(apple8)", &out, gen_cap);
    }

    // ---------------- the bar ----------------
    if let (Some(ord), Some(nv)) = (ordinary.as_ref(), nvidia.as_ref()) {
        let ratio = nv.tok_s / ord.tok_s;
        println!(
            "=== T5.5 bar (b): net tok/s draft(nvidia3) / ordinary = {ratio:.4} \
             (bar >= 1.05) — acceptance {:.3}, bonus {}, rollbacks {} ===",
            nv.stats.accept_rate(),
            nv.stats.bonus_tokens,
            nv.stats.rollbacks,
        );
        if ratio >= 1.05 {
            println!(
                "BAR (b): PASS ({ratio:.4} >= 1.05) — the GOAT gate's economics leg clears"
            );
        } else {
            println!(
                "BAR (b): MISS ({ratio:.4} < 1.05) — recorded NEGATIVE; no retune without \
                 a new pre-registration"
            );
        }
    }
    println!(
        "posture: smoke={smoke} — figures quote PROVENANCE above; the box state rides every number"
    );
}
