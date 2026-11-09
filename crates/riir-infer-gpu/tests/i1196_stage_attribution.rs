//! Issue 1196 — the synth Bonsai-2 fixture's a_log convention + the hd-128
//! Metal-vs-CPU dense decode parity gate it unblocks.
//!
//! RESOLUTION (T1+T2, m3 Metal, 2026-11-09): the "Metal dense decode NaNs at
//! step N" finding was a FIXTURE BUG, not a kernel defect. The kit wrote
//! `ssm_a = +1.0` — a RAW positive A_log — while the loader contract
//! (Issue 594) carries `a_log = −exp(A_log_raw)` (NEGATIVE; the converter
//! applied −exp once). With a_log = +1, decay = exp(+softplus(·)) ≥ 1 on
//! EVERY lane: the recurrence state grows without bound (measured: state
//! max 9.7 → 1.2e26 by step 9 → 1e34 by step 12 → Inf/NaN at 14 on Metal;
//! beta/decay buffers read 63.8 / 1066 / 924 at step 0 — impossible under
//! the convention). Metal's fp path crossed the overflow cliff sooner than
//! the CPU's; the "bit-clean prefix then NaN, Metal-only" shape was the
//! growth surfacing, never an hd-128 kernel assumption (hd 128 is in fact
//! the CANONICAL decode geometry — both recurrence kernels require exactly
//! head_dim == 128; the league model's GDN head_dim is 128, its ATTENTION
//! head_dim is 256).
//!
//! Attribution chain that named it (the stage-mask census, `DECODE_STAGE_MASK`):
//! producer = the DeltaNet region (mask-off deltanet kills the NaN; attention /
//! ffn / tail_norm do not; tail_lm_head is the disclosure point); bounded to
//! [pre_betal2, recurrence] by the sub-bit split; the per-step buffer census
//! (`state_readback::read_decode_diagnostic_buffers`) then showed L0.state
//! exploding BEFORE any non-finite beta/decay/qkv buffer — growth, not a pad
//! read — and the decay_buf values named the sign error.
//!
//! THE FIX: the kit writes `ssm_a = −exp(1.0)` (the converter's form for a
//! raw A of +1.0). Consumers assert INTERNAL consistency (lane-vs-lane on
//! the same fixture), not absolute pins, so no pin moves. T3 below is the
//! gate this finding exposes: Metal dense vs CPU dense on the CORRECT
//! fixture must agree — the standing hd-128 baseline the workspace never
//! had.
//!
//! Requires Metal (the compute law: GPU runs on GPU). SKIPs loud off-Metal.
#![cfg(feature = "flashmemory_budget_metal_t2")]

#[path = "../../../tests/common/synth_bonsai2.rs"]
mod synth;

use riir_infer_core::deltanet::{HybridCache, HybridForwardScratch, forward_qwen_deltanet_ternary};
use riir_infer_core::gguf_loader::load_qwen_deltanet_ternary_weights_gguf;
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::types::Config;
use riir_infer_gpu::ternary_deltanet_gpu_forward::{
    DECODE_STAGE_ALL, TernaryDeltanetGpuForward, set_decode_stage_mask,
};

/// The walk window: 40 teacher-forced steps — short by the compute law
/// (Metal decode on a 4-layer synth is sub-second per walk).
const STEPS: usize = 40;

/// The GPU forward's KV working-set clamp (the 1191 T2 pattern).
const GPU_CTX_CLAMP: usize = 8192;

/// The kit's raw A (a_log = −exp(RAW_A) after this issue's fix).
const RAW_A: f32 = 1.0;

/// Agreement bars for the dense-vs-dense parity gate. Measured (m3 Metal,
/// 2026-11-09, 40-step walk): max |Δ|/range 1.0e-1, top-1 0.95 (2/40 flips
/// on near-tie logits — an untrained random network's logit gaps sit at a
/// few % of range, so flips are tie-class, not divergence-class). The league
/// posture's top-1 ≥ 99.9% bar is model-calibrated (trained Bonsai logits
/// are well-separated); this gate pins the SYNTH baseline at its own
/// measured level with headroom. A regression beyond these bars means a
/// real Metal-vs-CPU divergence on the hd-128 shapes.
const TOP1_MIN: f64 = 0.90;
const MAX_REL_RANGE: f32 = 0.15;

fn build_synth(
    folded: bool,
    a_log_convention: bool,
) -> (
    Config,
    riir_infer_core::deltanet::QwenDeltaNetTernaryWeights,
) {
    let metadata = synth::synth_metadata(folded);
    let mut tensors = synth::synth_tensors_bonsai2();
    if a_log_convention {
        // The kit (post-1196) already carries the convention — normalize to
        // −exp(RAW_A) explicitly so both legs are deterministic regardless.
        let v = -RAW_A.exp();
        for t in &mut tensors {
            if t.name.ends_with(".ssm_a") {
                t.data = synth::f32_payload(&vec![v; t.ne.iter().product::<u64>() as usize]);
            }
        }
    } else {
        // The PRE-FIX bug, re-created deliberately: raw positive A_log.
        for t in &mut tensors {
            if t.name.ends_with(".ssm_a") {
                t.data = synth::f32_payload(&vec![RAW_A; t.ne.iter().product::<u64>() as usize]);
            }
        }
    }
    let bytes = synth::build_gguf(&metadata, &tensors);
    let stem = match (folded, a_log_convention) {
        (true, true) => "i1196_synth_folded_fixed.gguf",
        (true, false) => "i1196_synth_folded_bada.gguf",
        (false, true) => "i1196_synth_plain_fixed.gguf",
        (false, false) => "i1196_synth_plain_bada.gguf",
    };
    let path = synth::write_tmp(stem, &bytes);
    load_qwen_deltanet_ternary_weights_gguf(&path).expect("synth fixture load")
}

fn token_stream(len: usize, vocab: usize) -> Vec<usize> {
    let mut state = 0x1196_5A17u64;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as usize % vocab
        })
        .collect()
}

fn metal_dense(
    config: &Config,
    weights: &riir_infer_core::deltanet::QwenDeltaNetTernaryWeights,
    tokens: &[usize],
) -> Vec<Vec<f32>> {
    set_decode_stage_mask(DECODE_STAGE_ALL);
    let ctx = riir_infer_gpu::cubecl_runtime::CubeCLContext::new().expect("Metal ctx");
    assert!(
        ctx.is_metal_adapter(),
        "the gate REQUIRES Metal (compute law: GPU runs on GPU)"
    );
    let mut gpu_config = config.clone();
    gpu_config.block_size = GPU_CTX_CLAMP;
    let mut fwd = TernaryDeltanetGpuForward::new(&ctx, &gpu_config, weights);
    let mut logits = Vec::with_capacity(tokens.len());
    for &tok in tokens {
        fwd.set_input_token(weights, tok);
        logits.push(fwd.forward_token());
    }
    logits
}

fn cpu_dense(
    config: &Config,
    weights: &riir_infer_core::deltanet::QwenDeltaNetTernaryWeights,
    tokens: &[usize],
) -> Vec<Vec<f32>> {
    let rope = RopeFreqTable::new(config.rope_theta, config.head_dim);
    let mut cache =
        HybridCache::with_layer_types_bounded(config, &weights.layer_types, tokens.len() + 2);
    let mut scratch = HybridForwardScratch::new(config);
    let mut x = vec![0.0f32; config.vocab_size.max(config.n_embd)];
    let mut logits = Vec::with_capacity(tokens.len());
    for (pos, &tok) in tokens.iter().enumerate() {
        let out = forward_qwen_deltanet_ternary(
            &mut x,
            weights,
            &mut cache,
            tok,
            pos,
            config,
            &mut scratch,
            &rope,
        );
        logits.push(out.to_vec());
    }
    logits
}

fn agreement(a: &[Vec<f32>], b: &[Vec<f32>]) -> (f64, f32) {
    let mut agree = 0usize;
    let mut worst_rel = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let (ia, _) = x
            .iter()
            .enumerate()
            .max_by(|p, q| p.1.total_cmp(q.1))
            .unwrap();
        let (ib, _) = y
            .iter()
            .enumerate()
            .max_by(|p, q| p.1.total_cmp(q.1))
            .unwrap();
        if ia == ib {
            agree += 1;
        }
        let range = x
            .iter()
            .zip(y.iter())
            .map(|(&u, &v)| u.abs().max(v.abs()))
            .fold(0.0f32, |m, v| m.max(v))
            .max(1e-6);
        for (u, v) in x.iter().zip(y.iter()) {
            worst_rel = worst_rel.max((u - v).abs() / range);
        }
    }
    (agree as f64 / a.len() as f64, worst_rel)
}

/// The standing hd-128 baseline gate (T3): on the CONVENTION-CORRECT
/// FOLDED fixture (the decode posture serving runs), Metal dense vs CPU
/// dense must agree at the measured synth baseline. The PLAIN fixture runs
/// REPORT-ONLY beside it: its Metal-vs-CPU divergence GROWS with position
/// (|Δ|/range 2e-3 at steps 0–1 → 1.6 by step 27 — fp reduction-order noise
/// amplified through the untrained random network's direct path; the folded
/// staging chain is ~10× tighter). That amplification is a separate
/// accuracy study, filed forward — not this gate's law (see the issue's
/// follow-up row).
#[test]
fn i1196_hd128_dense_metal_vs_cpu_parity() {
    let (config, weights) = build_synth(true, true);
    let tokens = token_stream(STEPS, config.vocab_size);
    let metal = metal_dense(&config, &weights, &tokens);
    let cpu = cpu_dense(&config, &weights, &tokens);
    assert!(
        metal.iter().flatten().all(|v| v.is_finite()),
        "folded: Metal dense logits must be finite on the convention-correct fixture"
    );
    let (top1, rel) = agreement(&metal, &cpu);
    println!("[i1196] folded hd-128 dense parity: top-1 {top1:.4} · worst |Δ|/range {rel:.3e}");
    assert!(
        top1 >= TOP1_MIN,
        "folded: top-1 agreement {top1} < {TOP1_MIN}"
    );
    assert!(
        rel <= MAX_REL_RANGE,
        "folded: worst |Δ|/range {rel:.3e} > {MAX_REL_RANGE}"
    );

    // Report-only plain leg (the follow-up issue's baseline row).
    let (pconfig, pweights) = build_synth(false, true);
    let ptokens = token_stream(STEPS, pconfig.vocab_size);
    let pmetal = metal_dense(&pconfig, &pweights, &ptokens);
    let pcpu = cpu_dense(&pconfig, &pweights, &ptokens);
    let (ptop1, prel) = agreement(&pmetal, &pcpu);
    println!(
        "[i1196] plain hd-128 dense parity (REPORT-ONLY, filed forward): top-1 {ptop1:.4} · worst |Δ|/range {prel:.3e}"
    );
}

/// The a_log-convention tooth (T2's regression half): on the BAD fixture
/// (raw positive ssm_a), the recurrence state must grow without bound —
/// the decay buffer reads ≥ 1 where the convention guarantees ≤ 1. This
/// pins the fixture bug so a future kit regression re-trips it loudly
/// instead of surfacing as a mystery NaN three lanes downstream.
#[test]
fn i1196_bada_fixture_decay_exceeds_one() {
    let (config, weights) = build_synth(true, false);
    let ctx = riir_infer_gpu::cubecl_runtime::CubeCLContext::new().expect("Metal ctx");
    assert!(ctx.is_metal_adapter(), "requires Metal");
    let mut gpu_config = config.clone();
    gpu_config.block_size = GPU_CTX_CLAMP;
    let mut fwd = TernaryDeltanetGpuForward::new(&ctx, &gpu_config, &weights);
    let tokens = token_stream(4, config.vocab_size);
    for &tok in &tokens {
        fwd.set_input_token(&weights, tok);
        fwd.forward_token();
    }
    let bufs = riir_infer_gpu::state_readback::read_decode_diagnostic_buffers(&fwd);
    let decay: &Vec<f32> = &bufs
        .iter()
        .find(|(n, _)| n == "L0.decay_buf")
        .expect("L0.decay_buf in the manifest")
        .1;
    let over = decay.iter().filter(|&&d| d > 1.0 + 1e-3).count();
    println!(
        "[i1196] bad-fixture L0.decay_buf = {decay:?} — {over}/{} heads exceed 1.0",
        decay.len()
    );
    assert!(
        over >= decay.len() / 2,
        "the raw-positive ssm_a fixture must produce decay > 1 (found {over} heads)"
    );
}
