//! Issue 1188 T0 fixture — the family-taxonomy gates.
//!
//! The classifier is the census's whole authority: a misfiled tensor name
//! would ablate (or skip) the wrong family silently. These gates pin the
//! grammar against the committed census fixture
//! (`tests/fixtures/i1188_family_census_bonsai2_27b_pq2.txt`, the REAL
//! Bonsai-2-27B-PQ2_0 header walk) — name list and family assignment must
//! match it exactly, unknown names must miss (never guess), and the
//! storage-class partition must stay load-bearing (every per-layer family
//! ablatable in v1).

use riir_infer_core::family_census::{classify, Family, StorageClass};

/// Every tensor name recorded in the committed census fixture, verbatim,
/// with the family the census printed. A header change on the target GGUF
/// (a new/renamed tensor) reds here and forces a taxonomy decision before
/// any ablation trusts the classification.
const FIXTURE_NAMES: &[(&str, Family)] = &[
    ("token_embd.weight", Family::TokenEmbd),
    ("output_norm.weight", Family::Norms),
    ("output.weight", Family::Output),
    ("blk.0.attn_qkv.weight", Family::GdnQkv),
    ("blk.0.attn_gate.weight", Family::AttnGate),
    ("blk.0.ssm_conv1d.weight", Family::GdnConv),
    ("blk.0.ssm_dt.bias", Family::GdnDt),
    ("blk.0.ssm_a", Family::GdnA),
    ("blk.0.ssm_beta.weight", Family::GdnBeta),
    ("blk.0.ssm_alpha.weight", Family::GdnAlpha),
    ("blk.0.ssm_norm.weight", Family::Norms),
    ("blk.0.ssm_out.weight", Family::GdnOut),
    ("blk.0.attn_norm.weight", Family::Norms),
    ("blk.0.post_attention_norm.weight", Family::Norms),
    ("blk.0.ffn_gate.weight", Family::FfnGate),
    ("blk.0.ffn_up.weight", Family::FfnUp),
    ("blk.0.ffn_down.weight", Family::FfnDown),
    // A full-attention layer (every 4th on the qwen35 hybrid interval).
    ("blk.3.attn_q.weight", Family::AttnQ),
    ("blk.3.attn_k.weight", Family::AttnK),
    ("blk.3.attn_v.weight", Family::AttnV),
    ("blk.3.attn_output.weight", Family::AttnOut),
    ("blk.3.attn_q_norm.weight", Family::Norms),
    ("blk.3.attn_k_norm.weight", Family::Norms),
    ("blk.3.attn_norm.weight", Family::Norms),
    ("blk.3.post_attention_norm.weight", Family::Norms),
    ("blk.3.ffn_gate.weight", Family::FfnGate),
    ("blk.3.ffn_up.weight", Family::FfnUp),
    ("blk.3.ffn_down.weight", Family::FfnDown),
];

#[test]
fn fixture_names_classify_exactly_as_the_census_recorded() {
    for (name, want) in FIXTURE_NAMES {
        let got = classify(name).unwrap_or_else(|| panic!("{name} must classify"));
        assert_eq!(got.family, *want, "{name}");
    }
}

#[test]
fn every_layer_index_classifies_the_same_way() {
    // The grammar is index-independent — spot-check the highest recorded
    // layer (63 on the 64-layer Bonsai-2) against the same suffixes.
    for suffix_family in [
        ("attn_qkv.weight", Family::GdnQkv),
        ("ssm_alpha.weight", Family::GdnAlpha),
        ("ffn_down.weight", Family::FfnDown),
        ("attn_norm.weight", Family::Norms),
    ] {
        let name = format!("blk.63.{}", suffix_family.0);
        let got = classify(&name).unwrap_or_else(|| panic!("{name} must classify"));
        assert_eq!(got.family, suffix_family.1, "{name}");
        assert_eq!(got.layer, Some(63));
    }
}

#[test]
fn unknown_names_miss_and_never_guess() {
    for name in [
        "blk.0.unknown_thing.weight",
        "blk.x.attn_q.weight",
        "blk.0",
        "nextn.blk.0.attn_q.weight",
        "attn_q.weight", // a per-layer suffix WITHOUT the blk.N prefix is foreign
    ] {
        assert!(classify(name).is_none(), "{name} must MISS (never guess)");
    }
}

/// The v1 ablation surface: every family that `apply_family_ablation`
/// prices must be Proj/GateProj/Vector — and the two globals the v1
/// refuses must be exactly TokenEmbd + Output (the bare-ternary fields).
#[test]
fn v1_ablation_surface_partition_holds() {
    let per_layer: Vec<Family> = Family::ALL
        .iter()
        .copied()
        .filter(|f| !matches!(f, Family::TokenEmbd | Family::Output))
        .collect();
    assert_eq!(per_layer.len(), 16, "16 per-layer families in v1");
    for f in per_layer {
        assert!(
            matches!(
                f.storage_class(),
                StorageClass::Proj | StorageClass::GateProj | StorageClass::Vector
            ),
            "{} must carry an ablatable storage class",
            f.slug()
        );
    }
    assert_eq!(Family::TokenEmbd.storage_class(), StorageClass::Proj);
    assert_eq!(Family::Output.storage_class(), StorageClass::Proj);
}
