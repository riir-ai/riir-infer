//! Issue 1221 T5 — the dory lane's arch gates.
//!
//! Two halves:
//!
//! 1. **The negative guard (the T5 law):** a Dory GGUF handed to a
//!    non-dory lane REFUSES LOUD by arch name. Every lane's loader
//!    gates `general.architecture` before touching tensors
//!    (`expected <arch> architecture, got '<file arch>'`) — this gate
//!    pins it so a new lane (or a refactor) cannot silently accept a
//!    dory file, and cannot LOSE the arch-naming property.
//!
//! 2. **The family_census dory arch row (T5):** `census_of` dispatches
//!    on the file's own `general.architecture` — a dory GGUF censuses
//!    through the dory grammar with ZERO unclassified tensors over the
//!    full T0 stem set, while the same file under the arch-blind
//!    fallback would misfile the colliding `attn_gate.weight`.
//!
//! Synthesis rides the shared test builder (`common/synth_bonsai2.rs`);
//! the dory file needs ONLY the arch metadata for both halves — the
//! arch gate fires before any tensor read, and the census grammar is
//! exercised via the lib-side `classify_in_arch` battery (in
//! `family_census.rs`) plus the end-to-end `census_of` dispatch here.

#![cfg(feature = "dory")]

#[path = "common/synth_bonsai2.rs"]
mod synth;

use riir_infer_core::family_census::{Family, census_of};
use riir_infer_core::gguf_loader::{
    GgufFile, load_gemma2_weights_gguf, load_llama_weights_gguf, load_qwen_deltanet_weights_gguf,
    load_qwen2_weights_gguf,
};
use synth::{Val, build_gguf, write_tmp};

/// A minimal dory GGUF: the arch metadata ONLY (the loaders' arch gates
/// fire before any tensor read; the census dispatch reads the same key).
fn dory_gguf() -> Vec<u8> {
    build_gguf(
        &[(
            "general.architecture".to_string(),
            Val::Str("dory".to_string()),
        )],
        &[],
    )
}

/// Every non-dory loader refuses a dory file LOUD and BY ARCH NAME.
/// (`expect_err` is unusable here — the Ok types carry no Debug.)
#[test]
fn a_dory_gguf_refuses_loud_by_arch_name_on_every_foreign_lane() {
    let path = write_tmp("i1221_t5_dory_refusal", &dory_gguf());
    macro_rules! check_refusal {
        ($what:literal, $call:expr) => {
            match $call {
                Ok(_) => panic!("{} must refuse a dory file", $what),
                Err(e) => assert!(
                    e.to_string().contains("dory"),
                    "{} refusal must NAME the arch: {}",
                    $what,
                    e
                ),
            }
        };
    }
    check_refusal!("llama", load_llama_weights_gguf(&path));
    check_refusal!("gemma2", load_gemma2_weights_gguf(&path));
    check_refusal!("qwen2", load_qwen2_weights_gguf(&path));
    check_refusal!("qwen35", load_qwen_deltanet_weights_gguf(&path));
    let _ = std::fs::remove_file(&path);
}

/// The census dispatch: a dory GGUF censuses through the dory grammar —
/// the colliding `attn_gate.weight` lands in AttnGateProj (the
/// arch-blind fallback would misfile it as the qwen35 DeltaNet z gate),
/// and the census rollup names the dory families.
#[test]
fn census_of_dispatches_dory_to_the_dory_grammar() {
    // One attention pair + one ffn pair + globals — the full stem set.
    let stems_attn = [
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.attn_gate.weight",
        "blk.0.attn_qk_scale.weight",
        "blk.0.attn_gate_scale.weight",
        "blk.0.residual_alpha.weight",
    ];
    let stems_ffn = [
        "blk.1.ffn_gate.weight",
        "blk.1.ffn_up.weight",
        "blk.1.ffn_down.weight",
        "blk.1.ffn_gate_scale.weight",
        "blk.1.ffn_up_scale.weight",
        "blk.1.residual_alpha.weight",
    ];
    let mut names: Vec<&str> = Vec::new();
    names.extend_from_slice(&stems_attn);
    names.extend_from_slice(&stems_ffn);
    names.push("token_embd.weight");
    names.push("output.weight");
    names.push("output_scale.weight");
    let bytes = build_gguf(
        &[(
            "general.architecture".to_string(),
            Val::Str("dory".to_string()),
        )],
        // Zero data — the census reads NAMES only; every spec needs a
        // nonempty shape/type but the payload can be one element.
        &names
            .iter()
            .map(|name| synth::TensorSpec {
                name: (*name).to_string(),
                ne: vec![1],
                ggml_type: 30, // BF16
                data: vec![0u8; 2],
            })
            .collect::<Vec<_>>(),
    );
    let path = write_tmp("i1221_t5_dory_census", &bytes);
    let file = GgufFile::open(&path).expect("synth dory gguf opens");
    let census = census_of(&file);
    assert_eq!(
        census.unclassified, 0,
        "every dory stem must classify: unclassified = {census:?}"
    );
    let rollup = |slug: &str| {
        census
            .rollups
            .iter()
            .find(|r| r.family.slug() == slug)
            .unwrap_or_else(|| panic!("family {slug} missing from the rollup"))
    };
    assert_eq!(rollup("attn_gate_proj").tensors, 1, "the colliding gate");
    assert_eq!(rollup("residual_alpha").tensors, 2, "one per block");
    assert_eq!(rollup("output_scale").tensors, 1, "the global exit scale");
    assert_eq!(rollup("attn_qk_scale").tensors, 1);
    assert_eq!(rollup("ffn_up_scale").tensors, 1);
    // And the qwen35 DeltaNet z-gate family must NOT fire for dory.
    assert!(
        census.rollups.iter().all(|r| r.family != Family::AttnGate),
        "a dory census must never produce the qwen35 AttnGate family"
    );
    let _ = std::fs::remove_file(&path);
}
