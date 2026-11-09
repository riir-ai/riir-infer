//! Issue 1197 — WHY the plain synth fixture's Metal-vs-CPU dense decode
//! divergence grows with position while the folded one stays bounded
//! (plain: |Δ|/range 2.5e-3 at steps 0–1 → 1.57 by step 27; folded: ≤ 1.0e-1
//! throughout — the 1196 gate's report-only rows).
//!
//! Hypotheses:
//! - H1 (DeltaNet state map): the delta-rule update S' = decay·S + β·k(v −
//!   kᵀS) amplifies perturbations whenever β·‖k_head‖² > 1 + decay for some
//!   head. The synth's UNTRAINED weight scales make this a coin flip per
//!   fixture: folded and plain carry DIFFERENT effective networks (the fold
//!   rotates the projections), so one can be stable while the other is not.
//! - H2 (attention KV): the cache grows by one row per step; accumulated
//!   cache noise scales with position.
//!
//! Probes (readback-based, the 1196 seam — diagnostic only):
//! 1. Per-step divergence curve (plain) + the per-head amplification
//!    factor `decay + β·‖k‖²` computed from the census values (beta_buf +
//!    qkv_expanded's K region). H1 is CONFIRMED if some head's factor sits
//!    above 1 sustained.
//! 2. Zero-attention-output surgery on the plain fixture: the attention
//!    block contributes nothing (W_o = 0), so position-dependent KV noise
//!    is eliminated from the residual stream — if the growth persists, H2
//!    is dead.
//!
//! Requires Metal (the compute law). SKIPs loud off-Metal.
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

const STEPS: usize = 40;
const GPU_CTX_CLAMP: usize = 8192;

/// The plain fixture with convention-correct a_log, optionally with the
/// attention layer's output projection zeroed (the surgery leg: the block
/// contributes nothing, so position-dependent KV noise cannot reach the
/// residual stream).
fn build_plain(
    zero_attention_out: bool,
) -> (
    Config,
    riir_infer_core::deltanet::QwenDeltaNetTernaryWeights,
) {
    let metadata = synth::synth_metadata(false);
    let mut tensors = synth::synth_tensors_bonsai2();
    if zero_attention_out {
        for t in &mut tensors {
            // The kit's attention-block tensors are named `blk.{last}.attn_*`;
            // the output projection is the `attn_output.weight` row set.
            // Zeroing W_o makes attn_out = 0 for every position, so the
            // attention layer (and its growing KV cache) cannot influence
            // the residual stream — every GDN layer is untouched.
            if t.name.contains("attn_output") {
                t.data = synth::f32_payload(&vec![0.0f32; t.ne.iter().product::<u64>() as usize]);
            }
        }
    }
    let bytes = synth::build_gguf(&metadata, &tensors);
    let stem = if zero_attention_out {
        "i1197_plain_zeroattn.gguf"
    } else {
        "i1197_plain.gguf"
    };
    let path = synth::write_tmp(stem, &bytes);
    load_qwen_deltanet_ternary_weights_gguf(&path).expect("plain fixture load")
}

fn token_stream(len: usize, vocab: usize) -> Vec<usize> {
    let mut state = 0x1197_0DDu64;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as usize % vocab
        })
        .collect()
}

/// Metal dense walk; `census` prints L0's per-step state max and the
/// per-head amplification factor `decay + β·‖k‖²` (all terms from the
/// census: beta_buf, decay_buf, qkv_expanded's K region).
fn metal_walk(
    config: &Config,
    weights: &riir_infer_core::deltanet::QwenDeltaNetTernaryWeights,
    tokens: &[usize],
    census: bool,
    label: &str,
) -> Vec<Vec<f32>> {
    set_decode_stage_mask(DECODE_STAGE_ALL);
    let ctx = riir_infer_gpu::cubecl_runtime::CubeCLContext::new().expect("Metal ctx");
    assert!(ctx.is_metal_adapter(), "requires Metal");
    let mut gpu_config = config.clone();
    gpu_config.block_size = GPU_CTX_CLAMP;
    let mut fwd = TernaryDeltanetGpuForward::new(&ctx, &gpu_config, weights);
    let mut logits = Vec::with_capacity(tokens.len());
    for (step, &tok) in tokens.iter().enumerate() {
        fwd.set_input_token(weights, tok);
        logits.push(fwd.forward_token());
        if census && (step % 4 == 0 || step == tokens.len() - 1) {
            let bufs = riir_infer_gpu::state_readback::read_decode_diagnostic_buffers(&fwd);
            let beta = &bufs
                .iter()
                .find(|(n, _)| n == "L0.beta_buf")
                .expect("L0.beta_buf")
                .1;
            let decay = &bufs
                .iter()
                .find(|(n, _)| n == "L0.decay_buf")
                .expect("L0.decay_buf")
                .1;
            let kv = &bufs
                .iter()
                .find(|(n, _)| n == "L0.qkv_expanded")
                .expect("L0.qkv_expanded")
                .1;
            let state = &bufs
                .iter()
                .find(|(n, _)| n == "L0.state")
                .expect("L0.state")
                .1;
            let smax = state.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            // Expanded layout: [Q(n_v·hd) | K(n_v·hd) | V(n_v·hd)].
            let n_v = beta.len();
            let hd = kv.len() / (3 * n_v);
            let k_base = n_v * hd;
            let amps: Vec<String> = (0..n_v)
                .map(|h| {
                    let knorm2: f32 = kv[k_base + h * hd..k_base + (h + 1) * hd]
                        .iter()
                        .map(|v| v * v)
                        .sum();
                    format!("{:.3}", decay[h] + beta[h] * knorm2)
                })
                .collect();
            println!(
                "[i1197] {label} step {step}: state max {smax:.3e} · amplification decay+β‖k‖² = {amps:?}"
            );
        }
    }
    logits
}

fn cpu_walk(
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

fn divergence_curve(metal: &[Vec<f32>], cpu: &[Vec<f32>]) -> (Vec<f32>, usize) {
    let mut flips = 0usize;
    let curve: Vec<f32> = metal
        .iter()
        .zip(cpu.iter())
        .map(|(u_row, v_row)| {
            let (ia, _) = u_row
                .iter()
                .enumerate()
                .max_by(|p, q| p.1.total_cmp(q.1))
                .unwrap();
            let (ib, _) = v_row
                .iter()
                .enumerate()
                .max_by(|p, q| p.1.total_cmp(q.1))
                .unwrap();
            if ia != ib {
                flips += 1;
            }
            let range = u_row
                .iter()
                .zip(v_row.iter())
                .map(|(&u, &v)| u.abs().max(v.abs()))
                .fold(0.0f32, |m, v| m.max(v))
                .max(1e-6);
            u_row
                .iter()
                .zip(v_row.iter())
                .map(|(&u, &v)| (u - v).abs() / range)
                .fold(0.0f32, |m, r| m.max(r))
        })
        .collect();
    (curve, flips)
}

#[test]
fn i1197_amplifier_attribution() {
    // ── Leg 1 + H1 measurement: plain walk with the census on. ──
    let (config, weights) = build_plain(false);
    let tokens = token_stream(STEPS, config.vocab_size);
    let metal = metal_walk(&config, &weights, &tokens, true, "plain");

    let cpu = cpu_walk(&config, &weights, &tokens);
    let (curve, flips) = divergence_curve(&metal, &cpu);
    println!("[i1197] plain divergence curve (|Δ|/range): {curve:?}");
    println!("[i1197] plain top-1 flips: {flips}/{}", curve.len());

    // ── Leg 2 (H2): zero the attention output projection — the KV cache
    //    can no longer reach the residual stream. Growth that persists is
    //    the DeltaNet state map's. ──
    let (zconfig, zweights) = build_plain(true);
    let ztokens = token_stream(STEPS, zconfig.vocab_size);
    let zmetal = metal_walk(&zconfig, &zweights, &ztokens, false, "zeroattn");
    let zcpu = cpu_walk(&zconfig, &zweights, &ztokens);
    let (zcurve, zflips) = divergence_curve(&zmetal, &zcpu);
    println!("[i1197] ZERO-ATTN-OUT divergence curve: {zcurve:?}");
    println!("[i1197] zeroattn top-1 flips: {zflips}/{}", zcurve.len());
}
