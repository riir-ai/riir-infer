//! Issue 1208 T2 — the dense-application G1 battery on the synthetic
//! hybrid (the plan622 kit: fast, deterministic, box-portable).
//!
//! Pins, on the same weights run three ways:
//!
//! 1. **G1 three-way digest parity** — apply-at-load (the weight copy
//!    diverges) == in-forward shadow overlay (base pristine, overlay
//!    dispatches) == the INDEPENDENT test-side zeroing reference. All
//!    three must produce bit-identical decode logits.
//! 2. **The forward-level identity pin** — an all-keep mask (applied at
//!    load AND mounted as an overlay) is byte-identical to running the
//!    base weights: the empty mask ≡ base, now proven THROUGH the
//!    forward, not just the codec.
//! 3. **Base pristine** — after an overlay run, the rig's base gate
//!    weights are unchanged (the shadow-copy contract; hot-swap is
//!    pointer-swap).
//! 4. **Determinism** — the same mask mounted twice → the same digest.
//! 5. **Report accuracy** — `apply_mask_at_load`'s report counts the
//!    exact zeroed weights; silent sections (unknown tensors) skip;
//!    length disagreements refuse.

#![cfg(all(feature = "deletion_mask", not(target_arch = "wasm32")))]

#[path = "common/plan622_synth.rs"]
mod synth;

use riir_infer_core::adapter::f32_apply::{GateMaskOverlay, apply_mask_at_load};
use riir_infer_core::adapter::mask_codec::{MaskManifest, MaskSection, decode};
use riir_infer_core::deltanet::forward::{
    HybridCache, HybridForwardScratch, forward_qwen_deltanet,
};
use riir_infer_core::deltanet::weights::QwenDeltaNetWeights;
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::types::Config;

use synth::{hybrid_config, synth_hybrid_weights};

/// A deterministic decode run: feed `tokens` one at a time from pos 0,
/// digest every output logits vector.
fn decode_digest(
    cfg: &Config,
    weights: &QwenDeltaNetWeights,
    tokens: &[usize],
    overlay: Option<GateMaskOverlay>,
) -> [u8; 32] {
    let mut cache = HybridCache::with_layer_types(cfg, &weights.layer_types);
    let mut scratch = HybridForwardScratch::new(cfg);
    #[cfg(feature = "deletion_mask")]
    {
        scratch.gate_mask_overlay = overlay;
    }
    #[cfg(not(feature = "deletion_mask"))]
    {
        let _ = overlay;
    }
    let rope_freq = RopeFreqTable::new(cfg.rope_theta, cfg.rope_dimension_count);
    let mut hasher = blake3::Hasher::new();
    let mut x = vec![0.0f32; cfg.vocab_size.max(cfg.n_embd) * 2];
    for (step, &tok) in tokens.iter().enumerate() {
        let out = forward_qwen_deltanet(
            &mut x,
            weights,
            &mut cache,
            tok,
            step,
            cfg,
            &mut scratch,
            &rope_freq,
        );
        hasher.update(bytemuck::cast_slice::<f32, u8>(&out[..cfg.vocab_size]));
    }
    *hasher.finalize().as_bytes()
}

/// The mask manifest over every layer's gate projection: deterministic
/// deletion pattern (every 5th weight of layer 0, every 7th of layer 1,
/// every 11th of layer 2; keep-all elsewhere in those tensors).
fn deletion_mask(cfg: &Config) -> (MaskManifest, Vec<u8>) {
    let base_digest = *blake3::hash(b"i1208-t2-synth-base").as_bytes();
    let sections: Vec<MaskSection> = (0..cfg.n_layer)
        .map(|i| {
            let weights = cfg.mlp_hidden * cfg.n_embd;
            MaskSection::new(
                format!("blk.{i}.ffn_gate.weight"),
                (i * weights) as u64,
                weights as u64,
            )
            .unwrap()
        })
        .collect();
    let manifest = MaskManifest::new(base_digest, sections).unwrap();
    let total_bits = cfg.n_layer * cfg.mlp_hidden * cfg.n_embd;
    let mut bits = vec![0xFFu8; total_bits.div_ceil(8)];
    for i in 0..cfg.n_layer {
        let stride = [5usize, 7, 11][i];
        let base = i * cfg.mlp_hidden * cfg.n_embd;
        let count = cfg.mlp_hidden * cfg.n_embd;
        for w in (0..count).step_by(stride) {
            let p = base + w;
            bits[p / 8] &= !(1 << (p % 8));
        }
    }
    (manifest, bits)
}

#[test]
fn g1_three_way_parity_and_identity() {
    let cfg = hybrid_config();
    let tokens: Vec<usize> = (0..12).map(|i| (i * 7 + 3) % cfg.vocab_size).collect();

    // the pristine base digest
    let base_weights = synth_hybrid_weights(&cfg, 0xC0FFEE);
    let base_digest = decode_digest(&cfg, &base_weights, &tokens, None);

    // the mask + the three application paths
    let (manifest, bits) = deletion_mask(&cfg);
    let artifact = manifest.pack(&bits, None).unwrap();
    let decoded = decode(&artifact, None, None).unwrap();

    // Path A: apply-at-load (the weight copy diverges)
    let mut loaded = synth_hybrid_weights(&cfg, 0xC0FFEE);
    let reports = apply_mask_at_load(&mut loaded, &decoded).unwrap();
    assert_eq!(reports.len(), cfg.n_layer, "every layer's section applied");
    let digest_a = decode_digest(&cfg, &loaded, &tokens, None);

    // Path B: in-forward shadow overlay (base pristine)
    let pristine = synth_hybrid_weights(&cfg, 0xC0FFEE);
    let overlay = GateMaskOverlay::mount(&pristine, &decoded).unwrap();
    let digest_b = decode_digest(&cfg, &pristine, &tokens, Some(overlay));

    // Path C: the INDEPENDENT test-side zeroing reference — zero straight
    // from the test's own deletion pattern, never through the adapter.
    let mut reference = synth_hybrid_weights(&cfg, 0xC0FFEE);
    for i in 0..cfg.n_layer {
        let stride = [5usize, 7, 11][i];
        let data = reference.layers[i].gate_proj.dense_data_mut();
        for w in (0..data.len()).step_by(stride) {
            data[w] = 0.0;
        }
    }
    let digest_c = decode_digest(&cfg, &reference, &tokens, None);

    assert_eq!(digest_a, digest_c, "apply-at-load == zeroed reference");
    assert_eq!(digest_b, digest_c, "in-forward overlay == zeroed reference");
    assert_ne!(digest_a, base_digest, "the mask must move the logits");

    // identity pin: all-keep mask through BOTH postures == base digest
    let keep_bits = vec![0xFFu8; bits.len()];
    let keep_manifest = MaskManifest::new(
        *blake3::hash(b"i1208-t2-synth-base").as_bytes(),
        (0..cfg.n_layer)
            .map(|i| {
                let weights = cfg.mlp_hidden * cfg.n_embd;
                MaskSection::new(
                    format!("blk.{i}.ffn_gate.weight"),
                    (i * weights) as u64,
                    weights as u64,
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap();
    let keep_artifact = keep_manifest.pack(&keep_bits, None).unwrap();
    let keep_decoded = decode(&keep_artifact, None, None).unwrap();

    let mut kept_load = synth_hybrid_weights(&cfg, 0xC0FFEE);
    apply_mask_at_load(&mut kept_load, &keep_decoded).unwrap();
    let kept_digest = decode_digest(&cfg, &kept_load, &tokens, None);
    assert_eq!(kept_digest, base_digest, "all-keep at load == base");

    let keep_overlay = GateMaskOverlay::mount(&pristine, &keep_decoded).unwrap();
    let kept_overlay_digest = decode_digest(&cfg, &pristine, &tokens, Some(keep_overlay));
    assert_eq!(kept_overlay_digest, base_digest, "all-keep overlay == base");
}

#[test]
fn overlay_leaves_base_pristine() {
    let cfg = hybrid_config();
    let weights = synth_hybrid_weights(&cfg, 0xABCD);
    let before: Vec<Vec<f32>> = weights
        .layers
        .iter()
        .map(|l| l.gate_proj.dense_data().to_vec())
        .collect();

    let (manifest, bits) = deletion_mask(&cfg);
    let artifact = manifest.pack(&bits, None).unwrap();
    let decoded = decode(&artifact, None, None).unwrap();
    let overlay = GateMaskOverlay::mount(&weights, &decoded).unwrap();

    let tokens: Vec<usize> = (0..6).map(|i| (i * 5 + 1) % cfg.vocab_size).collect();
    let _ = decode_digest(&cfg, &weights, &tokens, Some(overlay));

    for (i, layer) in weights.layers.iter().enumerate() {
        assert_eq!(
            layer.gate_proj.dense_data(),
            before[i].as_slice(),
            "layer {i} base gate weights must stay pristine under the overlay"
        );
    }
}

#[test]
fn overlay_determinism_and_report_counts() {
    let cfg = hybrid_config();
    let weights = synth_hybrid_weights(&cfg, 0xBEEF);
    let (manifest, bits) = deletion_mask(&cfg);
    let artifact = manifest.pack(&bits, None).unwrap();
    let decoded = decode(&artifact, None, None).unwrap();

    let tokens: Vec<usize> = (0..8).map(|i| (i * 3 + 2) % cfg.vocab_size).collect();
    let o1 = GateMaskOverlay::mount(&weights, &decoded).unwrap();
    let o2 = GateMaskOverlay::mount(&weights, &decoded).unwrap();
    assert_eq!(
        decode_digest(&cfg, &weights, &tokens, Some(o1)),
        decode_digest(&cfg, &weights, &tokens, Some(o2)),
        "same mask mounted twice -> same digest"
    );

    // report counts: layer 0's stride-5 pattern zeroes floor(count/5) weights
    let mut loaded = synth_hybrid_weights(&cfg, 0xBEEF);
    let reports = apply_mask_at_load(&mut loaded, &decoded).unwrap();
    let count = cfg.mlp_hidden * cfg.n_embd;
    let strided = |stride: usize| count.div_ceil(stride);
    assert_eq!(reports[0].weights_zeroed, strided(5));
    assert_eq!(reports[1].weights_zeroed, strided(7));
    assert_eq!(reports[2].weights_zeroed, strided(11));
    assert_eq!(reports[0].weights_total, count);
    // and the zeroing matches the mask bytes directly
    let mut expect_zeroed = 0usize;
    let base = 0usize;
    for w in (0..count).step_by(5) {
        let p = base + w;
        if (bits[p / 8] >> (p % 8)) & 1 == 0 {
            expect_zeroed += 1;
        }
    }
    assert_eq!(reports[0].weights_zeroed, expect_zeroed);
}

#[test]
fn length_mismatch_and_unknown_tensor_routing() {
    let cfg = hybrid_config();
    let weights = synth_hybrid_weights(&cfg, 0x1234);

    // a section naming an unknown tensor is SILENT (bind-where-bound)
    let base_digest_input = *blake3::hash(b"routing").as_bytes();
    let weights_count = cfg.mlp_hidden * cfg.n_embd;
    let manifest = MaskManifest::new(
        base_digest_input,
        vec![
            MaskSection::new("blk.0.ffn_gate.weight", 0, weights_count as u64).unwrap(),
            // a non-layer tensor name: the mask is silent where it does not bind
            MaskSection::new(
                "output_norm.weight",
                weights_count as u64,
                weights_count as u64,
            )
            .unwrap(),
        ],
    )
    .unwrap();
    let mut bits = vec![0xFFu8; (2 * weights_count).div_ceil(8)];
    bits[0] &= 0xFE; // delete weight 0 of layer 0
    let artifact = manifest.pack(&bits, None).unwrap();
    let decoded = decode(&artifact, None, None).unwrap();

    let mut loaded = synth_hybrid_weights(&cfg, 0x1234);
    let reports = apply_mask_at_load(&mut loaded, &decoded).unwrap();
    assert_eq!(reports.len(), 1, "only the bound section applied");
    assert_eq!(reports[0].tensor_name, "blk.0.ffn_gate.weight");
    assert_eq!(reports[0].weights_zeroed, 1);

    // overlay: same silent-skip law
    let overlay = GateMaskOverlay::mount(&weights, &decoded).unwrap();
    assert!(overlay.arms(0));
    assert!(!overlay.arms(1));
    assert!(!overlay.arms(2));

    // a section whose bit_len disagrees with the tensor refuses loudly
    let bad = MaskManifest::new(
        base_digest_input,
        vec![MaskSection::new("blk.0.ffn_gate.weight", 0, (weights_count + 7) as u64).unwrap()],
    )
    .unwrap();
    let bad_bits = vec![0xFFu8; (weights_count + 7).div_ceil(8)];
    let bad_artifact = bad.pack(&bad_bits, None).unwrap();
    let bad_decoded = decode(&bad_artifact, None, None).unwrap();
    let mut target = synth_hybrid_weights(&cfg, 0x1234);
    assert!(apply_mask_at_load(&mut target, &bad_decoded).is_err());
    assert!(GateMaskOverlay::mount(&weights, &bad_decoded).is_err());

    // a section naming a layer beyond the model refuses loudly (NOT silent —
    // only unknown TENSOR NAMES skip; an out-of-range layer index is a
    // mask-authoring bug)
    let deep = MaskManifest::new(
        base_digest_input,
        vec![MaskSection::new("blk.9.ffn_gate.weight", 0, weights_count as u64).unwrap()],
    )
    .unwrap();
    let deep_bits = vec![0xFFu8; weights_count.div_ceil(8)];
    let deep_artifact = deep.pack(&deep_bits, None).unwrap();
    let deep_decoded = decode(&deep_artifact, None, None).unwrap();
    let mut target = synth_hybrid_weights(&cfg, 0x1234);
    assert!(apply_mask_at_load(&mut target, &deep_decoded).is_err());
    assert!(GateMaskOverlay::mount(&weights, &deep_decoded).is_err());
}
