//! Issue 1208 T4 — the llama-lane (qwen2) forward-level G1 battery for
//! the deletion-mask application, synthetic-weights (no model file): the
//! tiny-config `forward_llama` walk digest must be IDENTICAL between
//!
//! - the apply-at-load posture (`apply_mask_at_load_llama`), and
//! - the independent test-side bit walk (not the shipped
//!   `apply_mask_f32`), and
//! - an all-keep identity mask ≡ the pristine base.
//!
//! The T4 real-model cell (`i1208_t4_league_cell`) replays the same
//! three-way at 7B scale; this battery is its cheap falsifier — a
//! section-walk or digest-convention defect fires here first.
#![cfg(feature = "deletion_mask")]

use katgpt_transformer::MultiLayerKVCache;
use riir_infer_core::adapter::f32_apply::{apply_mask_at_load_llama, gate_tensor_name};
use riir_infer_core::adapter::mask_codec::{MaskManifest, MaskSection, decode};
use riir_infer_core::llama_layer::{LlamaLayerWeights, LlamaTransformerWeights};
use riir_infer_core::transformer::{ForwardContext, forward_llama};
use riir_infer_core::types::Config;

fn tiny_config() -> Config {
    Config {
        n_layer: 2,
        n_head: 2,
        n_kv_head: 1,
        head_dim: 4,
        n_embd: 8,
        mlp_hidden: 16,
        vocab_size: 8,
        block_size: 64,
        ..Config::default()
    }
}

/// Deterministic synthetic weights (seeded LCG; every matrix filled
/// row-major — the layout law the mask's bit order rides).
fn synthetic_weights(config: &Config) -> LlamaTransformerWeights {
    let mut state = 0x2545F4914F6CDD1Du64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as f32 / (u32::MAX as f32) - 0.5
    };
    let vec_of = |rows: usize, cols: usize, next: &mut dyn FnMut() -> f32| {
        (0..rows * cols).map(|_| next()).collect::<Vec<f32>>()
    };
    LlamaTransformerWeights {
        wte: vec_of(config.vocab_size, config.n_embd, &mut next),
        lm_head: vec_of(config.vocab_size, config.n_embd, &mut next),
        final_norm: vec_of(config.n_embd, 1, &mut next),
        layers: (0..config.n_layer)
            .map(|_| LlamaLayerWeights {
                attn_wq: vec_of(config.n_head * config.head_dim, config.n_embd, &mut next),
                attn_wk: vec_of(config.n_kv_head * config.head_dim, config.n_embd, &mut next),
                attn_wv: vec_of(config.n_kv_head * config.head_dim, config.n_embd, &mut next),
                attn_wo: vec_of(config.n_embd, config.n_head * config.head_dim, &mut next),
                gate_proj: vec_of(config.mlp_hidden, config.n_embd, &mut next),
                up_proj: vec_of(config.mlp_hidden, config.n_embd, &mut next),
                down_proj: vec_of(config.n_embd, config.mlp_hidden, &mut next),
                input_norm: vec_of(config.n_embd, 1, &mut next),
                post_attn_norm: vec_of(config.n_embd, 1, &mut next),
            })
            .collect(),
    }
}

fn fnv(bytes: impl Iterator<Item = u8>) -> u64 {
    bytes.fold(0xcbf29ce484222325u64, |h, b| {
        (h ^ b as u64).wrapping_mul(0x100000001b3)
    })
}

fn walk_digest(config: &Config, weights: &LlamaTransformerWeights, steps: usize) -> u64 {
    let mut cache = MultiLayerKVCache::new(config);
    let mut ctx = ForwardContext::new(config);
    let mut digests = Vec::new();
    for step in 0..steps {
        let logits = forward_llama(
            &mut ctx,
            weights,
            &mut cache,
            step % config.vocab_size,
            step,
            config,
        );
        digests.extend(
            logits[..config.vocab_size]
                .iter()
                .map(|v| v.to_bits().to_le_bytes()),
        );
    }
    fnv(digests.into_iter().flatten())
}

/// A mask over every layer's gate projection: delete every 9th weight
/// (the G2-cell convention — well above noise, visible in the digest).
fn build_mask(config: &Config, base: [u8; 32], keep_all: bool) -> (MaskManifest, Vec<u8>) {
    let count = config.mlp_hidden * config.n_embd;
    let sections: Vec<MaskSection> = (0..config.n_layer)
        .map(|i| MaskSection::new(gate_tensor_name(i), (i * count) as u64, count as u64).unwrap())
        .collect();
    let manifest = MaskManifest::new(base, sections).unwrap();
    let mut bits = vec![0xFFu8; (config.n_layer * count).div_ceil(8)];
    if !keep_all {
        for i in 0..config.n_layer {
            for w in (0..count).step_by(9) {
                let p = i * count + w;
                bits[p / 8] &= !(1 << (p % 8));
            }
        }
    }
    (manifest, bits)
}

/// The independent reference: zero straight off the decoded bit stream,
/// deliberately not through `apply_mask_f32`.
fn zero_reference(weights: &mut LlamaTransformerWeights, decoded_bits: &[u8], count: usize) {
    for (i, layer) in weights.layers.iter_mut().enumerate() {
        for w in 0..count {
            let p = i * count + w;
            if decoded_bits[p / 8] >> (p % 8) & 1 == 0 {
                layer.gate_proj[w] = 0.0;
            }
        }
    }
}

#[test]
fn llama_lane_three_way_walk_parity() {
    let config = tiny_config();
    let base = *blake3::hash(b"i1208-llama-lane-base").as_bytes();
    let pristine = synthetic_weights(&config);
    let count = config.mlp_hidden * config.n_embd;

    let base_digest = walk_digest(&config, &pristine, 6);

    let (manifest, bits) = build_mask(&config, base, false);
    let artifact = manifest.pack(&bits, None).unwrap();
    let decoded = decode(&artifact, Some(&base), None).unwrap();

    // posture A: apply-at-load
    let mut applied = pristine.clone();
    let reports = apply_mask_at_load_llama(&mut applied, &decoded).unwrap();
    assert_eq!(reports.len(), config.n_layer, "every section must bind");
    let applied_zeroed: usize = reports.iter().map(|r| r.weights_zeroed).sum();
    assert_eq!(applied_zeroed, config.n_layer * count.div_ceil(9));
    let applied_digest = walk_digest(&config, &applied, 6);

    // posture B: the independent reference
    let mut reference = pristine.clone();
    zero_reference(&mut reference, &decoded.bits, count);
    let reference_digest = walk_digest(&config, &reference, 6);

    assert_eq!(
        applied_digest, reference_digest,
        "apply-at-load != independent zeroing — a bit-order/offset defect"
    );
    assert_ne!(
        applied_digest, base_digest,
        "the mask did not move the walk — the deleted weights never reached the forward"
    );

    // the identity pin: all-keep ≡ base
    let (identity_manifest, identity_bits) = build_mask(&config, base, true);
    let identity_artifact = identity_manifest.pack(&identity_bits, None).unwrap();
    let identity_decoded = decode(&identity_artifact, Some(&base), None).unwrap();
    let mut identity_applied = pristine.clone();
    apply_mask_at_load_llama(&mut identity_applied, &identity_decoded).unwrap();
    assert_eq!(
        walk_digest(&config, &identity_applied, 6),
        base_digest,
        "the all-keep mask moved the walk — application corrupts kept weights"
    );
}

#[test]
fn llama_lane_length_refusal() {
    let config = tiny_config();
    let base = *blake3::hash(b"i1208-llama-lane-base").as_bytes();
    let mut weights = synthetic_weights(&config);
    let count = config.mlp_hidden * config.n_embd;
    // a section one bit LONGER than the tensor must refuse loudly
    let sections = vec![MaskSection::new(gate_tensor_name(0), 0, count as u64 + 1).unwrap()];
    let manifest = MaskManifest::new(base, sections).unwrap();
    let bits = vec![0xFFu8; (count + 1).div_ceil(8)];
    let artifact = manifest.pack(&bits, None).unwrap();
    let decoded = decode(&artifact, Some(&base), None).unwrap();
    assert!(apply_mask_at_load_llama(&mut weights, &decoded).is_err());
}

#[test]
fn llama_lane_unknown_sections_skip_silently() {
    let config = tiny_config();
    let base = *blake3::hash(b"i1208-llama-lane-base").as_bytes();
    let mut weights = synthetic_weights(&config);
    // a section naming a layer beyond the model skips (bind-where-bound)
    // only when it is NOT parseable as a blk.N gate name; a parseable
    // out-of-range layer refuses
    let beyond = MaskSection::new(gate_tensor_name(99), 0, 8).unwrap();
    let manifest = MaskManifest::new(base, vec![beyond]).unwrap();
    let bits = vec![0xFFu8; 1];
    let artifact = manifest.pack(&bits, None).unwrap();
    let decoded = decode(&artifact, Some(&base), None).unwrap();
    assert!(
        apply_mask_at_load_llama(&mut weights, &decoded).is_err(),
        "an out-of-range LAYER index must refuse loudly (the deltanet law)"
    );
    // a non-gate name is silent
    let non_gate = MaskSection::new("encoder.some_tensor", 0, 8).unwrap();
    let manifest = MaskManifest::new(base, vec![non_gate]).unwrap();
    let artifact = manifest.pack(&bits, None).unwrap();
    let decoded = decode(&artifact, Some(&base), None).unwrap();
    let reports = apply_mask_at_load_llama(&mut weights, &decoded).unwrap();
    assert!(reports.is_empty(), "non-gate sections must skip silently");
}
