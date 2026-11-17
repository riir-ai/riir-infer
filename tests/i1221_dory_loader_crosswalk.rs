//! Issue 1221 T1 — the loader-vs-census crosswalk on the REAL pin.
//!
//! The T0 tripwire (`i1221_dory_header_tripwire.rs`) asserts the FILE's
//! census. THIS gate asserts the LOADER reads that file into the census
//! values — the join between the two files a loader defect would hide
//! (the len-derived-binding lesson: the defect lives in the JOIN).
//!
//! Skip-LOUD without the pin (the T0 posture): `I1221_DORY_GGUF` env,
//! else `../riir-train/data/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf`.

#![cfg(feature = "dory")]

use riir_infer_core::dory::{DoryBlockKind, DoryDims, load_dory_weights};

fn pin_path() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("I1221_DORY_GGUF") {
        let p = std::path::PathBuf::from(p);
        return if p.exists() { Some(p) } else { None };
    }
    let p =
        std::path::PathBuf::from("../riir-train/data/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf");
    if p.exists() { Some(p) } else { None }
}

#[test]
fn loader_reads_the_pinned_census() {
    let Some(path) = pin_path() else {
        panic!(
            "SKIP LOUD: the Dory pin is absent (I1221_DORY_GGUF or \
             ../riir-train/data/…Q5_K_M.gguf); restore via the riir-train \
             data_backup lane — a loader change without this gate is \
             unverified, never green"
        );
    };
    let w = load_dory_weights(&path).expect("the pin loads (the arch is dory)");

    // The header-driven partition (never hardcoded — asserted to MATCH
    // the census values, which is the format-drift tripwire's job).
    assert_eq!(w.meta.num_loops, 4, "num_loops");
    assert_eq!(w.meta.input_layers, 16, "input_layers");
    assert_eq!(w.meta.recurrent_layers, 40, "recurrent_layers");
    assert_eq!(w.meta.output_layers, 16, "output_layers");
    assert_eq!(w.meta.kv_cache_mode, "per_loop", "kv_cache_mode");
    assert_eq!(w.blocks.len(), 72, "physical blocks");
    assert_eq!(w.sliding_window, 1024, "sliding_window");

    // Weight-ties NONE (T0): the recurrent range's repeated stems must
    // diverge in the LOADED f32 too — the loader never dedups.
    let s = w.meta.recurrent_start();
    let b1 = &w.blocks[s];
    let b2 = &w.blocks[s + 1];
    assert_ne!(
        b1.residual_alpha[0].to_bits(),
        b2.residual_alpha[0].to_bits(),
        "recurrent blocks 16/17 alphas are byte-identical — the loader deduped"
    );

    // Parity partition: even attn / odd ffn, and the shape pins.
    let d = DoryDims::census();
    for (i, b) in w.blocks.iter().enumerate() {
        let want = if i % 2 == 0 {
            DoryBlockKind::Attn
        } else {
            DoryBlockKind::Ffn
        };
        assert_eq!(b.kind, want, "block {i} kind");
        match b.kind {
            DoryBlockKind::Attn => {
                assert_eq!(b.attn_q.len(), d.n_head * d.head_dim * d.n_embd);
                assert_eq!(b.attn_k.len(), d.n_kv_head * d.head_dim * d.n_embd);
                assert_eq!(b.attn_output.len(), d.n_embd * d.n_head * d.head_dim);
                assert_eq!(b.attn_gate_scale.len(), d.n_head * d.head_dim);
                assert!(b.ffn_gate.is_empty(), "attn block carries no ffn weights");
            }
            DoryBlockKind::Ffn => {
                assert_eq!(b.ffn_gate.len(), d.mlp_hidden * d.n_embd);
                assert_eq!(b.ffn_down.len(), d.n_embd * d.mlp_hidden);
                assert!(b.attn_q.is_empty(), "ffn block carries no attn weights");
            }
        }
    }

    // Vocab tensors transposed to [vocab][n_embd] row-major.
    assert_eq!(w.wte.len(), d.vocab * d.n_embd, "wte rows");
    assert_eq!(w.lm_head.len(), d.vocab * d.n_embd, "lm_head rows");
    assert_eq!(w.output_scale.len(), d.vocab, "output_scale (vocab-wide)");

    // The dual rope bases (T0: 1e7 / 1e4).
    assert!((w.rope_freq_base - 1e7).abs() < 1.0, "freq_base");
    assert!((w.rope_freq_base_2 - 1e4).abs() < 1.0, "freq_base_2");

    // The rope profiles are 2/1 alternating runs (T0 fixture: 72 items,
    // values ∈ {1, 2}) — the loader's `rope_primary` bit must be true on
    // the 2-entries and false on the 1-entries, and BOTH classes exist.
    let primaries: Vec<bool> = w.blocks.iter().map(|b| b.rope_primary).collect();
    let any_secondary = primaries.iter().any(|p| !p);
    let any_primary = primaries.iter().any(|p| *p);
    assert!(any_primary && any_secondary, "both rope bases are selected");
    assert_eq!(w.blocks[0].rope_profile, 2, "census block-0 profile");
    assert_eq!(w.blocks[1].rope_profile, 1, "census block-1 profile");
    assert!(w.blocks[0].rope_primary);
    assert!(!w.blocks[1].rope_primary);
}

#[test]
fn loader_is_deterministic_two_passes_bit_identical() {
    let Some(path) = pin_path() else {
        panic!("SKIP LOUD: the Dory pin is absent (see the other gate)");
    };
    let a = load_dory_weights(&path).expect("pass 1");
    let b = load_dory_weights(&path).expect("pass 2");
    // The T1 determinism law starts at the LOADER: same file ⇒
    // bit-identical f32 (same dequant path, no platform float drift).
    assert_eq!(a.blocks.len(), b.blocks.len());
    for (i, (x, y)) in a.blocks.iter().zip(b.blocks.iter()).enumerate() {
        assert_eq!(
            x.residual_alpha[0].to_bits(),
            y.residual_alpha[0].to_bits(),
            "block {i} alpha"
        );
        let (xa, ya) = match x.kind {
            DoryBlockKind::Attn => (&x.attn_q, &y.attn_q),
            DoryBlockKind::Ffn => (&x.ffn_gate, &y.ffn_gate),
        };
        assert_eq!(xa.len(), ya.len(), "block {i} width");
        // Bit-compare a stride sample (a full 18M-element compare per
        // block per pass is the probe's job; the stride pins the law).
        for j in (0..xa.len()).step_by(9973) {
            assert_eq!(xa[j].to_bits(), ya[j].to_bits(), "block {i} [{j}]");
        }
    }
    assert_eq!(a.wte.len(), b.wte.len());
    for j in (0..a.wte.len()).step_by(104_729) {
        assert_eq!(a.wte[j].to_bits(), b.wte[j].to_bits(), "wte [{j}]");
    }
}
