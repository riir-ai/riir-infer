//! Issue 1253 T5.2 (plan 639) — the MTP draft-block CPU-reference gates
//! (synthetic, pure-CPU; the real teacher-forced cell rides `MTP_GGUF` at
//! `--ignored` release, the T5.2 falsifier's posture).
//!
//! Pins the reference step's wiring + determinism + basis contract on the
//! synthetic folded twin (the T5.1 fixture's 15-tensor draft block + the
//! kit's folded ternary main stack):
//! 1. END-TO-END runs — main-stack forward → hidden → primal recovery →
//!    draft step → logits: finite, right width, deterministic (bit-identical
//!    twice).
//! 2. THE BASIS CONTRACT — feeding the rotated head-basis copy (the
//!    hidden variant's shipped form) produces DIFFERENT logits from the
//!    primal hand-off on a folded fixture: the step consumes primal and
//!    the recovery step the caller owes is load-bearing, not cosmetic.
//! 3. THE COMPOSE ORDER — enorm embeds FIRST, hnorm(h) SECOND (the concat
//!    order): a probe block whose eh_proj one-hot selects one coordinate
//!    per half reads the halves in the pinned order (with the zeroed
//!    block adding nothing).
//! 4. The REAL teacher-forced arm (SKIP loud without MTP_GGUF, --ignored):
//!    draft-vs-main greedy agreement on the real twin — the T5.2
//!    falsifier's cell, never a default gate ride.

#[path = "common/mtp_draft_fixture.rs"]
mod fixture;

use riir_infer_core::deltanet::forward::{HybridCache, HybridForwardScratch, effective_rotary_dim};
use riir_infer_core::deltanet::rotation::rotate_inverse_inplace;
use riir_infer_core::deltanet::ternary_forward::forward_qwen_deltanet_ternary;
use riir_infer_core::deltanet::ternary_weights::QwenDeltaNetTernaryWeights;
use riir_infer_core::gguf_loader::{
    GgufFile, load_mtp_draft_block, load_qwen_deltanet_ternary_from_gguf, mtp_twin_census,
};
use riir_infer_core::mtp_draft::{MtpDraftStepScratch, mtp_draft_step};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::transformer::KVCache;
use riir_infer_core::types::{Config, rmsnorm_with_gamma_eps};

use std::io::Write as _;

fn write_tmp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("{name}_{}.gguf", std::process::id()));
    let mut f = std::fs::File::create(&path).expect("create temp gguf");
    f.write_all(bytes).expect("write temp gguf");
    path
}

fn rope_table(config: &Config) -> RopeFreqTable {
    let rotary_dim = effective_rotary_dim(config);
    RopeFreqTable::new(config.rope_theta, rotary_dim)
}

/// Land `toks` on the synthetic main stack; return (cache, scratch, the
/// primal hidden of the LAST position).
fn land_main_stack(
    config: &Config,
    weights: &QwenDeltaNetTernaryWeights,
    toks: &[usize],
) -> (HybridCache, HybridForwardScratch, Vec<f32>) {
    let n = config.n_embd;
    let v = config.vocab_size;
    let rope = rope_table(config);
    let mut cache = HybridCache::with_layer_types(config, &weights.layer_types);
    let mut scratch = HybridForwardScratch::new(config);
    let mut x = vec![0.0f32; v.max(n)];
    for (pos, tok) in toks.iter().enumerate() {
        forward_qwen_deltanet_ternary(
            &mut x,
            weights,
            &mut cache,
            *tok,
            pos,
            config,
            &mut scratch,
            &rope,
        );
    }
    // The hidden variant's shipped form is the ROTATED head-basis copy —
    // recover the primal h the draft step consumes (the Issue-1111
    // recipe; the module basis contract).
    let mut h = scratch.hidden_copy[..n].to_vec();
    if let Some(rot) = &weights.rotation {
        let signs = rot.signs_for_width(n).map(|s| s.to_vec());
        if let Some(s) = &signs {
            rotate_inverse_inplace(&mut h[..n], Some(s), rot.block_size);
        }
    }
    (cache, scratch, h)
}

fn load_twin(
    unique: &str,
) -> (
    std::path::PathBuf,
    Config,
    QwenDeltaNetTernaryWeights,
    riir_infer_core::gguf_loader::MtpDraftBlock,
) {
    let path = write_tmp(unique, &fixture::declared_twin_bytes());
    let gguf = GgufFile::open(&path).expect("open twin");
    let census = mtp_twin_census(&gguf);
    assert!(census.draft_admittable());
    let (config, weights): (Config, QwenDeltaNetTernaryWeights) =
        load_qwen_deltanet_ternary_from_gguf(&gguf, "").expect("main stack loads");
    let block = load_mtp_draft_block(&gguf, &config).expect("draft block admits");
    (path, config, weights, block)
}

/// Arm 1 — the end-to-end synthetic walk: finite logits, right width,
/// deterministic (bit-identical twice over the same inputs).
#[test]
fn draft_step_end_to_end_deterministic() {
    let (path, config, weights, block) = load_twin("riir_1253_t52_det");
    assert!(
        weights.rotation.is_some(),
        "the synthetic fixture is folded (the basis arms need it)"
    );
    let rope = rope_table(&config);
    let n = config.n_embd;
    let v = config.vocab_size;

    let (_cache, _scratch, h) = land_main_stack(&config, &weights, &[7, 11, 23]);

    let mut ds = MtpDraftStepScratch::new(&config);
    let mut logits_a = vec![0.0f32; v];
    let mut logits_b = vec![0.0f32; v];
    let mut h_a = h.clone();
    mtp_draft_step(
        &block,
        &weights,
        &mut KVCache::new(&config),
        3,
        &mut h_a,
        23,
        &config,
        &rope,
        &mut ds,
        &mut logits_a,
    )
    .expect("draft step a");
    let mut h_b = h.clone();
    mtp_draft_step(
        &block,
        &weights,
        &mut KVCache::new(&config),
        3,
        &mut h_b,
        23,
        &config,
        &rope,
        &mut ds,
        &mut logits_b,
    )
    .expect("draft step b");

    assert_eq!(logits_a.len(), v);
    assert!(
        logits_a.iter().all(|l| l.is_finite()),
        "draft logits finite"
    );
    assert_eq!(
        logits_a, logits_b,
        "the step is deterministic (bit-identical)"
    );
    assert_eq!(h_a[..n], h_b[..n], "h_draft deterministic too");

    let pick = logits_a
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(**a, **b))
        .map_or(0, |(i, _)| i);
    assert!(pick < v, "the greedy draft pick is well-defined");
    let _ = std::fs::remove_file(&path);
}

/// Arm 2 — the basis contract: the primal hand-off and the raw rotated
/// copy give DIFFERENT draft logits on a folded fixture.
#[test]
fn draft_step_consumes_primal_basis() {
    let (path, config, weights, block) = load_twin("riir_1253_t52_basis");
    let rope = rope_table(&config);
    let n = config.n_embd;
    let v = config.vocab_size;

    let (_cache, scratch, primal) = land_main_stack(&config, &weights, &[5, 9]);
    let rotated = scratch.hidden_copy[..n].to_vec();
    assert_ne!(primal[..n], rotated[..n], "the fixture folds non-trivially");

    let mut ds = MtpDraftStepScratch::new(&config);
    let mut lg_primal = vec![0.0f32; v];
    let mut lg_rotated = vec![0.0f32; v];
    let mut hp = primal.clone();
    let mut hr = rotated;
    mtp_draft_step(
        &block,
        &weights,
        &mut KVCache::new(&config),
        2,
        &mut hp,
        9,
        &config,
        &rope,
        &mut ds,
        &mut lg_primal,
    )
    .expect("primal step");
    mtp_draft_step(
        &block,
        &weights,
        &mut KVCache::new(&config),
        2,
        &mut hr,
        9,
        &config,
        &rope,
        &mut ds,
        &mut lg_rotated,
    )
    .expect("rotated step");
    assert_ne!(
        lg_primal, lg_rotated,
        "the basis the step consumes is observable — primal ≠ rotated"
    );
    let _ = std::fs::remove_file(&path);
}

/// Arm 3 — the compose order: a probe block whose eh_proj one-hot selects
/// concat[0] (the enorm half) and concat[n] (the hnorm half) reads the
/// halves enorm-first / hnorm-second; the zeroed attention/FFN adds
/// nothing.
#[test]
fn draft_step_compose_order_enorm_first() {
    let (path, config, weights, mut block) = load_twin("riir_1253_t52_probe");
    let rope = rope_table(&config);
    let n = config.n_embd;
    let v = config.vocab_size;
    let eps = config.rms_norm_eps;

    // Probe block: attention + FFN projections zero; eh_proj rows 0/1
    // select concat[0] (enorm half) / concat[n] (hnorm half).
    for field in [
        &mut block.attn_q,
        &mut block.attn_k,
        &mut block.attn_v,
        &mut block.attn_output,
        &mut block.ffn_gate,
        &mut block.ffn_up,
        &mut block.ffn_down,
    ] {
        field.fill(0.0);
    }
    block.eh_proj.fill(0.0);
    let cols = 2 * n;
    block.eh_proj[0] = 1.0; // row 0 ← concat[0]
    block.eh_proj[cols + n] = 1.0; // row 1 ← concat[n]
    block.enorm.fill(0.0);
    block.enorm[0] = 3.0;
    block.hnorm.fill(1.0);

    // The primal embed of token 0 (the same recovery the step performs).
    let mut emb = vec![0.0f32; n];
    weights.wte.dequant_row_into(0, &mut emb);
    if let Some(rot) = &weights.rotation {
        let signs = rot.signs_for_width(n).map(|s| s.to_vec());
        if let Some(s) = &signs {
            rotate_inverse_inplace(&mut emb[..n], Some(s), rot.block_size);
        }
    }
    // Expected halves via the SAME norm fn (the probe pins the ORDER and
    // the wiring, never a second norm implementation):
    let mut e_normed = emb.clone();
    rmsnorm_with_gamma_eps(&mut e_normed[..n], &block.enorm, eps);
    let h_const = 2.0f32;
    let mut h = vec![h_const; n];
    let mut h_normed = h.clone();
    rmsnorm_with_gamma_eps(&mut h_normed[..n], &block.hnorm, eps);
    let expect0 = e_normed[0];
    let expect1 = h_normed[0];

    let mut ds = MtpDraftStepScratch::new(&config);
    let mut lg = vec![0.0f32; v];
    mtp_draft_step(
        &block,
        &weights,
        &mut KVCache::new(&config),
        0,
        &mut h,
        0,
        &config,
        &rope,
        &mut ds,
        &mut lg,
    )
    .expect("probe step");

    let tol = 1e-4;
    assert!(
        (h[0] - expect0).abs() < tol,
        "row 0 reads the ENORM half first: {} vs {expect0}",
        h[0]
    );
    assert!(
        (h[1] - expect1).abs() < tol,
        "row 1 reads the HNORM half second: {} vs {expect1}",
        h[1]
    );
    assert!(
        h[2..n].iter().all(|v| v.abs() < tol),
        "the zeroed block adds nothing"
    );
    let _ = std::fs::remove_file(&path);
}

/// Arm 3b — the block's OWN attn_norm is APPLIED pre-qkv (the
/// katgpt-rs-88 source-read class: loaded + pinned but never read)./// Construction: eh_proj one-hot onto concat[0] → h = [e0, 0…];
/// attn_norm all-ones (pure RMS — NOT identity); attn_v row 0 one-hot →
/// v[0] = normed_h[0]; attn_q zero except head 0's FIRST gate coordinate
/// driven to ±hard so sigmoid(gate) = 1 (sign computed from the expected
/// norm); k zero (t_n = 1 → softmax weight = 1 regardless); wo row 0
/// one-hot. Then h[0] = inp_sa[0] + normed_h[0] — WITHOUT the fix the
/// projection reads the RAW h and h[0] = 2·e0 instead.
#[test]
fn draft_step_applies_block_attn_norm() {
    let (path, config, weights, mut block) = load_twin("riir_1253_t52_anorm");
    let rope = rope_table(&config);
    let n = config.n_embd;
    let v = config.vocab_size;
    let hd = config.head_dim;
    let eps = config.rms_norm_eps;

    // Zero the block; eh_proj row 0 selects concat[0] (the enorm half).
    for field in [
        &mut block.attn_q,
        &mut block.attn_k,
        &mut block.attn_v,
        &mut block.attn_output,
        &mut block.ffn_gate,
        &mut block.ffn_up,
        &mut block.ffn_down,
    ] {
        field.fill(0.0);
    }
    block.eh_proj.fill(0.0);
    block.eh_proj[0] = 1.0;
    block.enorm.fill(0.0);
    block.enorm[0] = 1.0;
    block.hnorm.fill(1.0);

    // The compose output e0 (the same primitives the step runs).
    let mut emb = vec![0.0f32; n];
    weights.wte.dequant_row_into(0, &mut emb);
    if let Some(rot) = &weights.rotation {
        let signs = rot.signs_for_width(n).map(|s| s.to_vec());
        if let Some(s) = &signs {
            rotate_inverse_inplace(&mut emb[..n], Some(s), rot.block_size);
        }
    }
    let mut e_normed = emb.clone();
    rmsnorm_with_gamma_eps(&mut e_normed[..n], &block.enorm, eps);
    let e0 = e_normed[0];
    let h_in = vec![e0, 0.0f32]; // [e0, 0, 0, …]

    // With attn_norm all-ones: normed_h = rmsnorm([e0, 0…]) → normed_h[0]
    // = e0/rms = sqrt(n)·sign(e0). Compute via the SAME norm fn.
    block.attn_norm.fill(1.0);
    let mut normed = h_in.clone();
    normed.resize(n, 0.0);
    rmsnorm_with_gamma_eps(&mut normed[..n], &block.attn_norm, eps);
    let normed0 = normed[0];
    assert!(
        normed0.abs() > 1.0,
        "the probe needs a non-trivial norm scale"
    );

    // v[0] ← normed_h[0]; wo row 0 ← attn_out[0].
    block.attn_v[0] = 1.0; // row 0, col 0 (row-major [kv_dim × n])
    block.attn_output[0] = 1.0; // row 0, col 0 ([n × q_dim])
    // Head 0's first gate coordinate: attn_q row = hd (q‖gate interleave).
    // Drive the gate hard in the SIGN of normed0 so sigmoid → 1 exactly.
    let gate_sign = if normed0 > 0.0 { 1.0f32 } else { -1.0 };
    block.attn_q[hd * n] = 1000.0 * gate_sign;

    let mut h = vec![0.0f32; n];
    h[0] = e0;
    let mut ds = MtpDraftStepScratch::new(&config);
    let mut lg = vec![0.0f32; v];
    mtp_draft_step(
        &block,
        &weights,
        &mut KVCache::new(&config),
        0,
        &mut h,
        0,
        &config,
        &rope,
        &mut ds,
        &mut lg,
    )
    .expect("probe step");

    // sigmoid(1000·|normed0|) = 1.0 in f32 (|normed0| ≈ sqrt(n) = 32) →
    // h[0] = inp_sa[0] + wo(attn_out[0]·1) = e0 + normed0.
    let expect = e0 + normed0;
    let tol = 1e-4;
    assert!(h[0].is_finite());
    assert!(h[1..n].iter().all(|v| v.abs() < tol));
    assert!(
        (h[0] - expect).abs() < tol,
        "attn_norm must be applied pre-qkv: h[0] = {} vs expected e0 + normed0 = {expect} \
         (the raw-projection bug reads 2·e0 = {})",
        h[0],
        2.0 * e0
    );
    let _ = std::fs::remove_file(&path);
}

/// Arm 4 — the REAL teacher-forced cell (SKIP loud without MTP_GGUF;
/// `--ignored` at release, the twin on disk). The T5.2 falsifier: the
/// draft step teacher-forces the MAIN stack's own greedy continuation;
/// its top-1 agreement is plan 639 bar (a)'s input (≥ 0.4353, the 924
/// trained-head chat per-slot rate, or the block intake is wrong).
#[test]
#[ignore]
fn draft_step_real_teacher_forced_agreement() {
    let path = std::path::PathBuf::from(
        std::env::var("MTP_GGUF")
            .unwrap_or_else(|_| "../riir-train/data/Ternary-Bonsai-2.5-27B-PQ2_0-mtp.gguf".into()),
    );
    let Some(path) = path.canonicalize().ok() else {
        eprintln!("SKIP: the 2.5 -mtp GGUF is not on this box (MTP_GGUF names it)");
        return;
    };
    let gguf = GgufFile::open(&path).expect("open the real twin");
    let (config, weights) =
        load_qwen_deltanet_ternary_from_gguf(&gguf, "").expect("main stack loads");
    let block = load_mtp_draft_block(&gguf, &config).expect("draft block admits");
    let rope = rope_table(&config);
    let n = config.n_embd;
    let v = config.vocab_size;

    let seed_tokens: Vec<usize> = (0..24).map(|i| (i * 7919 + 13) % v).collect();
    let mut cache = HybridCache::with_layer_types(&config, &weights.layer_types);
    let mut scratch = HybridForwardScratch::new(&config);
    let mut x = vec![0.0f32; v.max(n)];
    let rot = weights
        .rotation
        .as_ref()
        .expect("the league twin is folded");
    let signs = rot.signs_for_width(n).map(|s| s.to_vec());

    let mut draft_cache = KVCache::new(&config);
    let mut ds = MtpDraftStepScratch::new(&config);
    let mut lg = vec![0.0f32; v];
    let mut agree = 0usize;
    let mut total = 0usize;
    let mut h_prev: Option<Vec<f32>> = None;

    for (pos, &tok) in seed_tokens.iter().enumerate() {
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
        let main_pick = x[..v]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(**a, **b))
            .map_or(0, |(i, _)| i);
        if let Some(mut h) = h_prev.take() {
            if let Some(s) = &signs {
                rotate_inverse_inplace(&mut h[..n], Some(s), rot.block_size);
            }
            mtp_draft_step(
                &block,
                &weights,
                &mut draft_cache,
                pos,
                &mut h,
                tok,
                &config,
                &rope,
                &mut ds,
                &mut lg,
            )
            .expect("real draft step");
            let draft_pick = lg[..v]
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(**a, **b))
                .map_or(0, |(i, _)| i);
            if draft_pick == main_pick {
                agree += 1;
            }
            total += 1;
        }
        h_prev = Some(scratch.hidden_copy[..n].to_vec());
    }
    let rate = agree as f64 / total.max(1) as f64;
    println!(
        "draft-vs-main greedy top-1 agreement: {agree}/{total} = {rate:.4} (bar (a) >= 0.4353, \
         the 924 trained-head chat per-slot rate)"
    );
    assert!(total > 0, "the walk produced no draft positions");
}

/// Arm 5 — the T5.4-R re-open cell (plan 639, pre-registered 2026-11-27,
/// `zed-f021648e@amo`): the chat-transcript teacher-forced agreement, the
/// 924 prior's own measurement shape. SKIP loud without the token file
/// (the regeneration command is named) or the twin (`MTP_GGUF`). The
/// numerics are byte-identical to arm 4 — only the SEED DISTRIBUTION and
/// the sample size change (the two confounds the bar-(a) NEGATIVE named).
///
/// Leg A (the bar leg, the 924 @63 shape): sliding 64-token windows,
/// stride 56, capped at 256 windows; per window FRESH main + draft
/// caches, the arm-4 teacher-forced roll runs the WHOLE window (a draft
/// step at every pos >= 1 warms the block's own KV — a lone step at 63
/// would attend over an empty cache and undercount the block), and ONLY
/// the pos-63 comparison is recorded. BAR (a)-R >= 0.4353 — Bench 917's
/// pairing (head vs THE MODEL'S OWN argmax on the same row, never vs
/// gold).
///
/// Leg B (context row, never the bar): one sequential teacher-forced
/// walk over the first min(file, 4096) tokens (the serving ctx), the
/// arm-4 loop shape, agreement at every position >= 1.
#[test]
#[ignore]
fn draft_step_chat_transcript_teacher_forced_agreement() {
    let twin = std::path::PathBuf::from(
        std::env::var("MTP_GGUF")
            .unwrap_or_else(|_| "../riir-train/data/Ternary-Bonsai-2.5-27B-PQ2_0-mtp.gguf".into()),
    );
    let Some(twin) = twin.canonicalize().ok() else {
        eprintln!("SKIP: the 2.5 -mtp GGUF is not on this box (MTP_GGUF names it)");
        return;
    };
    let tokens_path = std::path::PathBuf::from(
        std::env::var("MTP_CHAT_TOKENS")
            .unwrap_or_else(|_| "../riir-train/data/bonsai_mtp/tokens.chat.tokens".into()),
    );
    let Ok(tokens_bytes) = std::fs::read(&tokens_path) else {
        eprintln!(
            "SKIP: the chat token file is absent at {} — regenerate with: cargo run \
             --release -p riir-infer-dist --example mtp_chat_tokens",
            tokens_path.display()
        );
        return;
    };
    assert_eq!(tokens_bytes.len() % 4, 0, "the chat token file is u32-LE");
    let tokens: Vec<usize> = tokens_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
        .collect();
    eprintln!(
        "chat transcript: {} tokens from {}",
        tokens.len(),
        tokens_path.display()
    );

    let gguf = GgufFile::open(&twin).expect("open the real twin");
    let (config, weights) =
        load_qwen_deltanet_ternary_from_gguf(&gguf, "").expect("main stack loads");
    let block = load_mtp_draft_block(&gguf, &config).expect("draft block admits");
    let rope = rope_table(&config);
    let n = config.n_embd;
    let v = config.vocab_size;
    let rot = weights
        .rotation
        .as_ref()
        .expect("the league twin is folded");
    let signs = rot.signs_for_width(n).map(|s| s.to_vec());
    let argmax = |row: &[f32]| -> usize {
        row.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(**a, **b))
            .map_or(0, |(i, _)| i)
    };

    // Bounded-capacity caches (the Issue-929 seam): the unbounded
    // constructor allocates full-block_size KV per layer (128k-class
    // context on this config — run 1's kill class: ~43.6 GB RSS,
    // commit-pressure death at window ~192). max_positions must exceed
    // every position written: leg A ≤ 63, leg B ≤ 4095; 8192 covers both
    // with margin. The attention forwards index positionally (pos*kvd, no
    // growth, no ring), so the values are bit-identical to the unbounded
    // cache. The MAIN forward keeps the original config (geometry) and its
    // own scratch shape; the DRAFT step + its cache + scratch share the
    // capped clone so the kernel's slot-stride argument and the cache
    // agree.
    let mut cache_cfg = config.clone();
    if cache_cfg.block_size > 8192 {
        cache_cfg.block_size = 8192;
    }

    // ---- Leg A: the 924 @63 shape (sliding windows, roll-through) ----
    const WIN: usize = 64;
    const STRIDE: usize = 56;
    const MAX_WINDOWS: usize = 256;
    // Resume mechanics (run 2, disclosed in plan 639): the windows are
    // independent (fresh caches per window), so a resumed slice is a
    // statistically exact concatenation with run 1's completed windows —
    // never a re-measure of a different design. Default 0 = the registered
    // run byte-identical.
    let start_window: usize = std::env::var("MTP_CHAT_START_WINDOW")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let mut agree_a = 0usize;
    let mut total_a = 0usize;
    let mut windows = 0usize;
    let mut start = start_window * STRIDE;
    // One allocation, reset per window (the Issue-838 H1 seam — the
    // per-call reset for many independent sequences against one cache;
    // identical values to fresh caches, no GB-class memset churn ×250).
    let mut cache = HybridCache::with_layer_types_bounded(&cache_cfg, &weights.layer_types, 8192);
    let mut scratch = HybridForwardScratch::new(&config);
    let mut x = vec![0.0f32; v.max(n)];
    let mut draft_cache = KVCache::new(&cache_cfg);
    let mut ds = MtpDraftStepScratch::new(&cache_cfg);
    let mut lg = vec![0.0f32; v];
    while start + WIN <= tokens.len() && windows < MAX_WINDOWS {
        let slice = &tokens[start..start + WIN];
        cache.reset();
        draft_cache.reset();
        let mut h_prev: Option<Vec<f32>> = None;
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
            let last = pos == WIN - 1;
            if let Some(mut h) = h_prev.take() {
                if let Some(s) = &signs {
                    rotate_inverse_inplace(&mut h[..n], Some(s), rot.block_size);
                }
                mtp_draft_step(
                    &block,
                    &weights,
                    &mut draft_cache,
                    pos,
                    &mut h,
                    tok,
                    &cache_cfg,
                    &rope,
                    &mut ds,
                    &mut lg,
                )
                .expect("real draft step (leg A)");
                if last {
                    if argmax(&lg[..v]) == argmax(&x[..v]) {
                        agree_a += 1;
                    }
                    total_a += 1;
                }
            }
            h_prev = Some(scratch.hidden_copy[..n].to_vec());
        }
        windows += 1;
        start += STRIDE;
        if windows.is_multiple_of(32) {
            eprintln!(
                "leg A: window {}/{} — agree {}/{} (start_window {start_window})",
                start_window + windows,
                start_window + MAX_WINDOWS,
                agree_a,
                total_a
            );
        }
    }
    let rate_a = agree_a as f64 / total_a.max(1) as f64;

    // ---- Leg B: the serving-ctx sequential walk (context row) ----
    // Walk cap is a smoke lever (default 4096 = the registered walk; a
    // smoke sets 64 to exercise the bounded cache cheaply).
    let walk_cap: usize = std::env::var("MTP_CHAT_WALK_CAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);
    let walk_len = tokens.len().min(walk_cap);
    cache.reset();
    draft_cache.reset();
    let mut agree_b = 0usize;
    let mut total_b = 0usize;
    let mut h_prev: Option<Vec<f32>> = None;
    for (pos, &tok) in tokens[..walk_len].iter().enumerate() {
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
        let main_pick = argmax(&x[..v]);
        if let Some(mut h) = h_prev.take() {
            if let Some(s) = &signs {
                rotate_inverse_inplace(&mut h[..n], Some(s), rot.block_size);
            }
            mtp_draft_step(
                &block,
                &weights,
                &mut draft_cache,
                pos,
                &mut h,
                tok,
                &cache_cfg,
                &rope,
                &mut ds,
                &mut lg,
            )
            .expect("real draft step (leg B)");
            if argmax(&lg[..v]) == main_pick {
                agree_b += 1;
            }
            total_b += 1;
        }
        h_prev = Some(scratch.hidden_copy[..n].to_vec());
    }
    let rate_b = agree_b as f64 / total_b.max(1) as f64;

    println!(
        "LEG A (@63, {windows} windows): {agree_a}/{total_a} = {rate_a:.4} — BAR (a)-R >= 0.4353 {}",
        if rate_a >= 0.4353 { "PASS" } else { "MISS" }
    );
    println!(
        "LEG B (ctx walk, first {walk_len} tokens): {agree_b}/{total_b} = {rate_b:.4} (context \
         row, never the bar)"
    );
    assert!(total_a > 0, "leg A produced no windows");
    assert!(total_b > 0, "leg B produced no positions");
}
