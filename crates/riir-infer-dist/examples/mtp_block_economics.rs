//! Issue 1253 T5.4 (plan 639) — the GGUF-embedded MTP draft block's
//! ECONOMICS CELL over the real Bonsai-2.5 `-mtp` twin: the acceptance
//! rates + E[m] per verify round (leg A, load-valid) and the decode
//! tok/s arms (leg B, QUIET-WINDOW-ONLY per the plan's law).
//!
//! Leg A folds the chain verification into the T5.4-R leg-A window roll
//! (the bar (a)-R shape that passed 129/251 = 0.5139): the roll's main
//! forwards already produce every slot's target greedy, so one roll
//! scores a full K=8 no-floor draft chain per position. Both card
//! postures derive EXACTLY from the recorded chains by mass truncation —
//! the min-prob gate is "the first step below the floor ends the chain,
//! its prediction NOT proposed", a deterministic walk over the recorded
//! per-step masses (the gate applies uniformly INCLUDING step 0).
//!
//! Conditioning note (why per-slot rates beyond the echo are honest): in
//! real specdec, draft slot j is only reached when slots 0..j-1 were
//! accepted, and the accepted prefix IS the chain's own conditioning
//! prefix — so the conditional rate P(c_j == g_j | c_i == g_i ∀i<j) is
//! the serving posture's rate, not a proxy.
//!
//! Leg B is the free-running greedy decode loop (draft influences the
//! path) with SEQUENTIAL verify — the CPU reference posture, the row
//! comparable to Bench 924's 5.35-vs-7.69 trained-head prior. At this
//! posture every landed token costs exactly one main forward (no bonus
//! term — that belongs to batched verify, T5.5's Metal cell), so leg B
//! measures the overhead class + the realized acceptance, never a
//! promotion claim.
//!
//! ```text
//! cargo run --release -p riir-infer-dist \
//!   --features mtp_draft_block,deltanet_ternary_inference,bonsai2_hadamard \
//!   --example mtp_block_economics
//!   # env: MTP_GGUF, MTP_CHAT_TOKENS, MTP_ECON_WINDOWS (256),
//!   #      MTP_ECON_LEG (A|B|AB), MTP_ECON_PROMPT (256), MTP_ECON_GEN (256),
//!   #      MTP_CHAT_START_WINDOW (0), MTP_ECON_SMOKE (tiny caps)
//! ```
//!
//! A latency number without its box state is not a measurement — quote
//! `bench_preflight.sh`'s PROVENANCE line (or the uptime load class)
//! beside every leg-B figure.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use riir_infer_core::deltanet::forward::{HybridCache, HybridForwardScratch, effective_rotary_dim};
use riir_infer_core::deltanet::rotation::rotate_inverse_inplace;
use riir_infer_core::deltanet::ternary_forward::forward_qwen_deltanet_ternary;
use riir_infer_core::gguf_loader::{
    GgufFile, load_mtp_draft_block, load_qwen_deltanet_ternary_from_gguf,
};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_dist::spec_mtp_block::{MtpDraftBlockDrafter, MtpDraftPreset};

const WIN: usize = 64;
const STRIDE: usize = 56;
/// One chain per position at the Apple card's ceiling; both card
/// postures derive from the recorded chains by mass truncation.
const CHAIN_K: usize = 8;
/// The NVIDIA card's chain floor (the derivation threshold over the
/// recorded per-step masses).
const NVIDIA_MIN_PROB: f32 = 0.10;
/// KV position cap (the Issue-929 seam — leg A writes ≤ 64, leg B ≤
/// prompt+gen+K; 8192 covers both with margin; the attention forwards
/// index positionally, so values are bit-identical to the unbounded
/// cache — the T5.4-R bounded-cache proof carries over).
const CACHE_CAP: usize = 8192;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// The arm-5 pairing's main-side argmax rule
/// (`katgpt_core::float_order::cmp_for_max` + `max_by` semantics: NaN
/// below every real, equal reals LAST-wins) inlined — the dist crate
/// carries no katgpt-core dep and the pairing must not drift from the
/// bar (a)-R evidence.
fn argmax_last_ties(row: &[f32]) -> usize {
    let key = |a: f32| if a.is_nan() { f32::NEG_INFINITY } else { a };
    let mut best = 0usize;
    for (i, &l) in row.iter().enumerate() {
        if key(l).total_cmp(&key(row[best])) != std::cmp::Ordering::Less {
            best = i;
        }
    }
    best
}

struct LegAStats {
    /// slot 0 — the block's own read of the echo slot (c_0 vs the main's
    /// greedy at the same position; diagnostic, never part of E[m]).
    slot0_agree: usize,
    slot0_total: usize,
    /// slots 1..CHAIN_K-1 — the DRAFTED slots; per-slot agreement among
    /// samples where the slot was proposed AND scored.
    slot_agree: [usize; CHAIN_K],
    slot_total: [usize; CHAIN_K],
    /// per-slot agreement CONDITIONED on every earlier drafted slot
    /// agreeing (the serving posture's reachability).
    cond_agree: [usize; CHAIN_K],
    cond_total: [usize; CHAIN_K],
    /// Σ accepted drafts over full-chain positions, and the position
    /// count (the sample-mean E[m] divides these; per posture).
    sum_m_apple: u64,
    sum_m_nvidia: u64,
    full_chain_positions: usize,
    /// E[m] from the conditional rates (valid for the Apple posture —
    /// no truncation; the NVIDIA sample mean is the honest read, the
    /// shared-rate form would ignore the floor's reachability).
    em_apple: f64,
    windows: usize,
}

impl LegAStats {
    fn new() -> Self {
        Self {
            slot0_agree: 0,
            slot0_total: 0,
            slot_agree: [0; CHAIN_K],
            slot_total: [0; CHAIN_K],
            cond_agree: [0; CHAIN_K],
            cond_total: [0; CHAIN_K],
            sum_m_apple: 0,
            sum_m_nvidia: 0,
            full_chain_positions: 0,
            em_apple: 0.0,
            windows: 0,
        }
    }
}

/// The card postures derived from one recorded chain. Returns
/// (apple_m, nvidia_m): leading agreed counts among the PROPOSED drafted
/// slots (slot 0 is the echo — always landed free, never scored as a
/// draft). `nvidia` truncates at the first step whose mass < 0.10 (that
/// step's prediction NOT proposed); `apple` keeps the whole chain.
/// `agrees[j]` = (c_j == the target greedy at slot 0+j); an out-of-window
/// slot is `None` — a chain touching one is not a full-chain sample.
fn derive_posture_ms(chain: &[(u32, f32)], agrees: &[Option<bool>]) -> (Option<u64>, Option<u64>) {
    let mut nvidia_cut = chain.len();
    for (s, (_, mass)) in chain.iter().enumerate() {
        if *mass < NVIDIA_MIN_PROB {
            nvidia_cut = s;
            break;
        }
    }
    let count = |proposed: usize| -> Option<u64> {
        // proposed = the number of chain steps whose predictions are
        // proposed (steps 0..proposed-1). Drafted slots are 1..proposed.
        if proposed < 2 {
            return Some(0); // echo only — no drafted slot reached
        }
        let mut m = 0u64;
        for j in 1..proposed {
            match agrees.get(j).and_then(|a| *a) {
                Some(true) => m += 1,
                Some(false) => break,
                None => return None, // out-of-window — not a full-chain sample
            }
        }
        Some(m)
    };
    (count(chain.len()), count(nvidia_cut))
}

fn main() {
    let smoke = std::env::var("MTP_ECON_SMOKE").ok().as_deref() == Some("1");
    let debug = std::env::var("MTP_ECON_DEBUG").ok().as_deref() == Some("1");
    let leg: String = std::env::var("MTP_ECON_LEG").unwrap_or_else(|_| "AB".into());
    let max_windows = if smoke {
        4
    } else {
        env_usize("MTP_ECON_WINDOWS", 256)
    };
    let start_window = env_usize("MTP_CHAT_START_WINDOW", 0);
    let prompt_len = if smoke {
        64
    } else {
        env_usize("MTP_ECON_PROMPT", 256)
    };
    let gen_cap = if smoke {
        32
    } else {
        env_usize("MTP_ECON_GEN", 256)
    };

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
    eprintln!(
        "chat transcript: {} tokens from {} (smoke={smoke})",
        tokens.len(),
        tokens_path.display()
    );

    let gguf = GgufFile::open(&twin).expect("open the real twin");
    let (config, weights) =
        load_qwen_deltanet_ternary_from_gguf(&gguf, "").expect("main stack loads");
    let weights = Arc::new(weights);
    let block = Arc::new(load_mtp_draft_block(&gguf, &config).expect("draft block admits"));
    let rotary_dim = effective_rotary_dim(&config);
    let rope = RopeFreqTable::new(config.rope_theta, rotary_dim);
    let n = config.n_embd;
    let v = config.vocab_size;
    let rot = weights
        .rotation
        .as_ref()
        .expect("the league twin is folded");
    let signs = rot.signs_for_width(n).map(|s| s.to_vec());
    let block_size = rot.block_size;

    // Bounded-capacity caches (the Issue-929 seam; the T5.4-R posture).
    let mut cache_cfg = config.clone();
    if cache_cfg.block_size > CACHE_CAP {
        cache_cfg.block_size = CACHE_CAP;
    }
    let mut cache =
        HybridCache::with_layer_types_bounded(&cache_cfg, &weights.layer_types, CACHE_CAP);
    let mut scratch = HybridForwardScratch::new(&config);
    let mut x = vec![0.0f32; v.max(n)];

    let preset = MtpDraftPreset::apple(CHAIN_K).expect("the K=8 chain preset");
    let mut drafter = MtpDraftBlockDrafter::new(block, weights.clone(), cache_cfg.clone(), preset)
        .expect("seat the draft block");

    // One primal-hidden slot the roll reuses (the arm-5 hand-off, zero
    // per-position allocation beyond the one capture buffer). Leg A owns
    // `have_h_prev` (position 0 has no predecessor); leg B re-arms the
    // seed from the prefill's own final state.
    let mut h_prev: Vec<f32> = vec![0.0; n];

    // ---------------- Leg A: the chain-acceptance cell ----------------
    let mut stats = LegAStats::new();
    if leg.contains('A') {
        let mut have_h_prev: bool;
        // Determinism witness: window 0, position 1's chain, replayed at
        // the end (the bar (a)-R bit-identity class).
        let mut det_chain: Option<Vec<(u32, f32)>> = None;
        let mut windows = 0usize;
        let mut start = start_window * STRIDE;
        while start + WIN <= tokens.len() && windows < max_windows {
            let slice = &tokens[start..start + WIN];
            cache.reset();
            drafter.reset();
            have_h_prev = false;
            // Target greedy per position: greedy[p] = the content of slot
            // p+1 (the roll's own argmax at p).
            let mut greedy = [0usize; WIN];
            // Per-position chain record (scored AFTER the roll — the
            // target greedy for slots ahead of the cursor does not exist
            // until the roll reaches them).
            let mut chains: Vec<Vec<(u32, f32)>> = vec![Vec::new(); WIN];
            for (pos, &tok) in slice.iter().enumerate() {
                forward_qwen_deltanet_ternary(
                    &mut x,
                    &weights,
                    &mut cache,
                    tok,
                    pos,
                    &config,
                    &mut scratch,
                    &rope,
                );
                greedy[pos] = argmax_last_ties(&x[..v]);
                // The chain at pos consumes the PREVIOUS position's
                // hidden (the arm-5 hand-off: h_prev primal + the token
                // AT pos, predicting slot pos+1).
                if have_h_prev {
                    drafter
                        .seed(&h_prev, tok as u32, pos)
                        .expect("seed (leg A)");
                    chains[pos] = drafter.draft_chain();
                }
                h_prev.copy_from_slice(&scratch.hidden_copy[..n]);
                if let Some(s) = &signs {
                    rotate_inverse_inplace(&mut h_prev[..n], Some(s), block_size);
                }
                have_h_prev = true;
            }
            // Fold the window into the stats — scoring happens HERE, after
            // the roll, when every slot's target greedy is on the table.
            // Chain element j at pos predicts slot pos+1+j; the CONTENT of
            // slot S is the roll's argmax at S-1 (greedy[p] predicts slot
            // p+1) — scoreable while pos+j ≤ WIN-1.
            for pos in 1..WIN {
                let chain = &chains[pos];
                if chain.is_empty() {
                    continue;
                }
                let ag: Vec<Option<bool>> = chain
                    .iter()
                    .enumerate()
                    .map(|(j, (tok_c, _))| {
                        (pos + j < WIN).then(|| *tok_c as usize == greedy[pos + j])
                    })
                    .collect();
                if let Some(a0) = ag.first().and_then(|a| *a) {
                    stats.slot0_total += 1;
                    if a0 {
                        stats.slot0_agree += 1;
                    }
                }
                let mut prefix_ok = true;
                for j in 1..chain.len() {
                    match ag.get(j).and_then(|a| *a) {
                        Some(hit) => {
                            stats.slot_total[j] += 1;
                            if hit {
                                stats.slot_agree[j] += 1;
                            }
                            if prefix_ok {
                                stats.cond_total[j] += 1;
                                if hit {
                                    stats.cond_agree[j] += 1;
                                }
                            }
                            if !hit {
                                prefix_ok = false;
                            }
                        }
                        None => prefix_ok = false,
                    }
                }
                // Full-chain m samples: every drafted slot must be
                // scoreable in-window → the last chain element (j =
                // CHAIN_K-1) needs pos + CHAIN_K - 1 ≤ WIN - 1.
                if pos + CHAIN_K - 1 < WIN
                    && let (Some(ma), Some(mn)) = derive_posture_ms(chain, &ag)
                {
                    stats.sum_m_apple += ma;
                    stats.sum_m_nvidia += mn;
                    stats.full_chain_positions += 1;
                }
                if windows == 0 && pos == 1 {
                    det_chain = Some(chain.clone());
                }
                if debug && windows == 0 && pos <= 4 {
                    eprintln!(
                        "[dbg] pos {pos}: chain {:?} | greedy targets pos..pos+len: {:?}",
                        chain
                            .iter()
                            .map(|(t, m)| (t, *m >= NVIDIA_MIN_PROB))
                            .collect::<Vec<_>>(),
                        greedy[pos..(pos + chain.len()).min(WIN)].to_vec(),
                    );
                }
            }
            windows += 1;
            start += STRIDE;
            // Checkpoint: cumulative counts ride the log (the long-runs
            // law's cheap half — a resumed run's harvest reads them here;
            // the T5.4-R precedent). Windows are independent, so a
            // MTP_CHAT_START_WINDOW resume is an exact concatenation.
            if windows.is_multiple_of(16) {
                let n1 = stats.cond_total[1];
                let r1 = if n1 > 0 {
                    stats.cond_agree[1] as f64 / n1 as f64
                } else {
                    0.0
                };
                eprintln!(
                    "leg A: window {}/{} (start_window {start_window}) — slot0 {}/{} \
                     slot1-cond {}/{} = {:.4} | Σm apple {} / full-chain {}",
                    start_window + windows,
                    start_window + max_windows,
                    stats.slot0_agree,
                    stats.slot0_total,
                    stats.cond_agree[1],
                    n1,
                    r1,
                    stats.sum_m_apple,
                    stats.full_chain_positions
                );
            }
        }
        stats.windows = windows;

        // Determinism witness: replay window 0, position 1's chain.
        if let Some(c0) = &det_chain {
            let s0 = start_window * STRIDE;
            let slice = &tokens[s0..s0 + WIN];
            cache.reset();
            drafter.reset();
            have_h_prev = false;
            let mut replay = None;
            for (pos, &tok) in slice.iter().enumerate() {
                forward_qwen_deltanet_ternary(
                    &mut x,
                    &weights,
                    &mut cache,
                    tok,
                    pos,
                    &config,
                    &mut scratch,
                    &rope,
                );
                if have_h_prev {
                    drafter.seed(&h_prev, tok as u32, pos).expect("replay seed");
                    if pos == 1 {
                        replay = Some(drafter.draft_chain());
                    } else {
                        drafter.draft_chain();
                    }
                }
                h_prev.copy_from_slice(&scratch.hidden_copy[..n]);
                if let Some(s) = &signs {
                    rotate_inverse_inplace(&mut h_prev[..n], Some(s), block_size);
                }
                have_h_prev = true;
            }
            assert_eq!(
                replay.as_deref(),
                Some(c0.as_slice()),
                "leg A determinism: window-0 pos-1 chain drifted across the pass"
            );
            eprintln!("leg A determinism: window-0 pos-1 chain bit-identical across the pass");
        }

        // E[m] from the conditional rates — valid for the Apple posture
        // (no truncation): E[m] = Σ_{j=1..K-1} Π_{i=1..j} a_i.
        let mut em = 0.0f64;
        let mut p = 1.0f64;
        for j in 1..CHAIN_K {
            let rate = if stats.cond_total[j] > 0 {
                stats.cond_agree[j] as f64 / stats.cond_total[j] as f64
            } else {
                0.0
            };
            p *= rate;
            em += p;
        }
        stats.em_apple = em;
    }

    // ---------------- Leg B: the decode tok/s cell ----------------
    // QUIET-WINDOW-ONLY (the plan's law): the caller arms it explicitly
    // on a quiet box; the printed lines must carry the load class.
    if leg.contains('B') {
        let prompt = prompt_len.min(tokens.len());

        // ---- arm 1: ordinary greedy decode (the incumbent row) ----
        cache.reset();
        let mut last = tokens[prompt - 1];
        for (pos, &tok) in tokens[..prompt].iter().enumerate() {
            forward_qwen_deltanet_ternary(
                &mut x,
                &weights,
                &mut cache,
                tok,
                pos,
                &config,
                &mut scratch,
                &rope,
            );
            if pos == prompt - 1 {
                last = argmax_last_ties(&x[..v]);
            }
        }
        let mut landed = 0usize;
        let t0 = Instant::now();
        for pos in prompt..prompt + gen_cap {
            forward_qwen_deltanet_ternary(
                &mut x,
                &weights,
                &mut cache,
                last,
                pos,
                &config,
                &mut scratch,
                &rope,
            );
            last = argmax_last_ties(&x[..v]);
            landed += 1;
        }
        let ordinary_wall = t0.elapsed();
        println!(
            "LEG B ordinary: {landed} tokens in {:.2?} = {:.2} tok/s [PROVENANCE: \
             quote bench_preflight or the uptime load class]",
            ordinary_wall,
            landed as f64 / ordinary_wall.as_secs_f64()
        );

        // ---- arm 2: repeat-last drafter, K=3 (the machinery floor) ----
        // Drafts the just-landed token into every slot; acceptance =
        // P(the model's greedy repeats the token). Sequential verify
        // lands exactly one token per forward — the pure overhead class.
        cache.reset();
        let mut last = tokens[prompt - 1];
        for (pos, &tok) in tokens[..prompt].iter().enumerate() {
            forward_qwen_deltanet_ternary(
                &mut x,
                &weights,
                &mut cache,
                tok,
                pos,
                &config,
                &mut scratch,
                &rope,
            );
            if pos == prompt - 1 {
                last = argmax_last_ties(&x[..v]);
            }
        }
        let mut landed = 0usize;
        let mut verifies = 0usize;
        let mut accepts = 0usize;
        let t0 = Instant::now();
        let mut pos = prompt;
        while landed < gen_cap {
            // The free token.
            forward_qwen_deltanet_ternary(
                &mut x,
                &weights,
                &mut cache,
                last,
                pos,
                &config,
                &mut scratch,
                &rope,
            );
            let mut feed = argmax_last_ties(&x[..v]);
            landed += 1;
            pos += 1;
            // Verify 2 repeats of the just-landed token (K-1 drafted slots).
            for _ in 0..2 {
                if landed >= gen_cap {
                    break;
                }
                forward_qwen_deltanet_ternary(
                    &mut x,
                    &weights,
                    &mut cache,
                    feed,
                    pos,
                    &config,
                    &mut scratch,
                    &rope,
                );
                let g2 = argmax_last_ties(&x[..v]);
                if g2 == feed {
                    accepts += 1;
                }
                feed = g2;
                landed += 1;
                pos += 1;
                verifies += 1;
            }
            last = feed;
        }
        let repeat_wall = t0.elapsed();
        println!(
            "LEG B repeat-last: {landed} tokens in {:.2?} = {:.2} tok/s — repeat-accepts \
             {accepts}/{verifies} (the degenerate-drafter floor)",
            repeat_wall,
            landed as f64 / repeat_wall.as_secs_f64()
        );

        // ---- arm 3: the draft block (NVIDIA card: K=3, min-prob 0.10) ----
        cache.reset();
        drafter.reset();
        drafter.set_preset(MtpDraftPreset::nvidia_draft3());
        let mut last = tokens[prompt - 1];
        for (pos, &tok) in tokens[..prompt].iter().enumerate() {
            forward_qwen_deltanet_ternary(
                &mut x,
                &weights,
                &mut cache,
                tok,
                pos,
                &config,
                &mut scratch,
                &rope,
            );
            if pos == prompt - 1 {
                last = argmax_last_ties(&x[..v]);
            }
        }
        // h_final = the hidden at the LAST LANDED position (prompt-1),
        // primal — the seed the first decode round consumes.
        let mut h_final = scratch.hidden_copy[..n].to_vec();
        if let Some(s) = &signs {
            rotate_inverse_inplace(&mut h_final[..n], Some(s), block_size);
        }
        let mut landed = 0usize;
        let mut rounds = 0usize;
        let mut drafts_proposed = 0usize;
        let mut drafts_accepted = 0usize;
        let mut full_accept_rounds = 0usize;
        let t0 = Instant::now();
        let mut pos = prompt;
        while landed < gen_cap {
            // The round's free token: forward the last landed token at
            // its position; h_final is hidden@(pos-1) primal (the seed).
            forward_qwen_deltanet_ternary(
                &mut x,
                &weights,
                &mut cache,
                last,
                pos,
                &config,
                &mut scratch,
                &rope,
            );
            let mut feed = argmax_last_ties(&x[..v]);
            landed += 1;
            pos += 1;
            // The chain: seeded from (h_final = hidden@pos-1, the token
            // AT pos-1, predicting from position pos-1) — the arm-5
            // wiring; chain[0] re-reads the echo slot (pos-1+1), the
            // drafted slots are pos.. forward.
            drafter
                .seed(&h_final, last as u32, pos - 1)
                .expect("seed (leg B)");
            let chain = drafter.draft_chain();
            let drafts: Vec<u32> = chain.iter().skip(1).map(|(t, _)| *t).collect();
            let mut proposed_this_round = drafts.len();
            let mut accepted_this_round = 0usize;
            for &d in &drafts {
                if landed >= gen_cap {
                    proposed_this_round -= 1;
                    break;
                }
                // Verify d (drafts[i] = chain[i+1] predicts slot pos+i+1):
                // forward the token `feed` (at slot pos-1) AT position
                // pos-1? — no: `feed` is the model's pick for slot pos,
                // so this forward runs AT pos and produces the greedy for
                // slot pos+1, the slot d occupies. On accept, d becomes
                // the feed for the next verify forward; on mismatch, the
                // model's own pick lands (the correction).
                forward_qwen_deltanet_ternary(
                    &mut x,
                    &weights,
                    &mut cache,
                    feed,
                    pos,
                    &config,
                    &mut scratch,
                    &rope,
                );
                let g2 = argmax_last_ties(&x[..v]);
                let hit = g2 == d as usize;
                if hit {
                    drafts_accepted += 1;
                    accepted_this_round += 1;
                    feed = d as usize;
                } else {
                    feed = g2; // the correction lands
                }
                landed += 1;
                pos += 1;
                if !hit {
                    break;
                }
            }
            if accepted_this_round == proposed_this_round && proposed_this_round > 0 {
                full_accept_rounds += 1;
            }
            drafts_proposed += proposed_this_round;
            rounds += 1;
            last = feed;
            // The next round's seed = the hidden at the LAST forwarded
            // position (pos-1), primal.
            h_final.clear();
            h_final.extend_from_slice(&scratch.hidden_copy[..n]);
            if let Some(s) = &signs {
                rotate_inverse_inplace(&mut h_final[..n], Some(s), block_size);
            }
        }
        let draft_wall = t0.elapsed();
        println!(
            "LEG B draft(nvidia3): {landed} tokens in {:.2?} = {:.2} tok/s — drafts \
             proposed {drafts_proposed}, accepted {drafts_accepted} (rate {:.3}), \
             full-accept rounds {full_accept_rounds}/{rounds}",
            draft_wall,
            landed as f64 / draft_wall.as_secs_f64(),
            drafts_accepted as f64 / drafts_proposed.max(1) as f64
        );

        // Restore the leg-A posture (harmless; the example ends).
        drafter.set_preset(MtpDraftPreset::apple(CHAIN_K).expect("the K=8 preset"));
    }

    // ---------------- The verdict block ----------------
    if leg.contains('A') && stats.windows > 0 {
        println!(
            "=== LEG A — chain acceptance (windows {}, start_window {}) ===",
            stats.windows, start_window
        );
        println!(
            "slot 0 (the block's own echo-slot read): {}/{} = {:.4}",
            stats.slot0_agree,
            stats.slot0_total,
            stats.slot0_agree as f64 / stats.slot0_total.max(1) as f64
        );
        for j in 1..CHAIN_K {
            if stats.slot_total[j] > 0 {
                println!(
                    "slot {j}: agree {}/{} = {:.4} | conditional {}/{} = {:.4}",
                    stats.slot_agree[j],
                    stats.slot_total[j],
                    stats.slot_agree[j] as f64 / stats.slot_total[j] as f64,
                    stats.cond_agree[j],
                    stats.cond_total[j],
                    stats.cond_agree[j] as f64 / stats.cond_total[j].max(1) as f64
                );
            }
        }
        let mean_a = stats.sum_m_apple as f64 / stats.full_chain_positions.max(1) as f64;
        let mean_n = stats.sum_m_nvidia as f64 / stats.full_chain_positions.max(1) as f64;
        println!(
            "E[m] apple8: sample-mean {mean_a:.3} over {} full-chain positions (of \
             {} windows) | from conditional rates {:.3}",
            stats.full_chain_positions, stats.windows, stats.em_apple
        );
        println!(
            "E[m] nvidia3 (mass ≥ {NVIDIA_MIN_PROB}): sample-mean {mean_n:.3} (the honest \
             read — the shared-rate form would ignore the floor's reachability)"
        );
        // Tokens per verify round at the batched-verify ceiling: the echo
        // + E[m] (the batched +1 bonus is T5.5's own accounting).
        println!(
            "tokens/round ceiling (batched verify, echo + E[m]): apple8 {:.3} | \
             nvidia3 {:.3}",
            1.0 + mean_a,
            1.0 + mean_n
        );
    }
}
