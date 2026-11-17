//! Issue 1221 T1 — the LOADER's file-layout law on a synthetic GGUF.
//!
//! The real-pin crosswalk (`i1221_dory_loader_crosswalk.rs`) asserts the
//! loader's output against the census but cannot prove the VOCAB-TENSOR
//! INDEXING (any fixed layout is self-consistent under a two-pass
//! bit-identity read). THIS gate builds a GGUF whose token_embd/output
//! bytes encode their own coordinates, so a scrambled indexing — the
//! T1a defect (the loader treated ne[0]=n_embd as the SLOW dim and
//! transposed through the wrong stride) — diffs HERE, loudly, with no
//! pin required.
//!
//! The law being pinned (`gguf_loader.rs`): ne[0] is the innermost/
//! fastest dim = in_features for 2-D weights, so the BF16 dequant flat
//! IS `[vocab][n_embd]` row-major for the vocab tensors and
//! `[out][in]` for the projections — the loader stores them verbatim.

#![cfg(feature = "dory")]

#[path = "common/synth_bonsai2.rs"]
mod synth;
#[path = "common/synth_dory.rs"]
mod synth_dory;

use riir_infer_core::dory::{DoryBlockKind, DoryForwardError, load_dory_weights};
use riir_infer_core::dory::{DoryForward, DoryKvCache};
use synth::write_tmp;
use synth_dory::{BLOCKS, HEAD_DIM, MLP, N_EMBD, N_HEAD, N_KV, VOCAB, coord, synth_gguf};

/// THE layout law: the loaded wte/lm_head rows are the FILE's rows —
/// element (v, e) at file offset v*n_embd+e, verbatim. The T1a scramble
/// (indexing through the wrong stride) fails this on the first row.
#[test]
fn vocab_tensors_are_the_file_rows_verbatim() {
    let bytes = synth_gguf("dory", true);
    let path = write_tmp("i1221_dory_synth", &bytes);
    let w = load_dory_weights(&path).expect("synth dory loads");
    let _ = std::fs::remove_file(&path);

    let n_embd = N_EMBD as usize;
    for v in 0..VOCAB as usize {
        for e in 0..n_embd {
            let want = coord(v * n_embd + e);
            assert_eq!(
                w.wte[v * n_embd + e],
                want,
                "wte[{v}][{e}] is not file element {idx}",
                idx = v * n_embd + e
            );
            let want_oh = ((v * n_embd + e + 1) as f32) * 32.0;
            assert_eq!(
                w.lm_head[v * n_embd + e],
                want_oh,
                "lm_head[{v}][{e}] is not file element {idx}",
                idx = v * n_embd + e
            );
        }
    }
}

/// The header-derived dims the forward reads (never hardcoded — the
/// loader parses `dory.attention.*` and the shape lengths).
#[test]
fn dims_and_meta_echo_the_header() {
    let bytes = synth_gguf("dory", true);
    let path = write_tmp("i1221_dory_synth_meta", &bytes);
    let w = load_dory_weights(&path).expect("synth dory loads");
    let _ = std::fs::remove_file(&path);

    let d = &w.dims;
    assert_eq!(d.vocab, VOCAB as usize);
    assert_eq!(d.n_embd, N_EMBD as usize);
    assert_eq!(d.n_head, N_HEAD as usize);
    assert_eq!(d.n_kv_head, N_KV as usize);
    assert_eq!(d.head_dim, HEAD_DIM as usize);
    assert_eq!(d.mlp_hidden, MLP as usize);
    assert_eq!(d.n_blocks, BLOCKS as usize);
    assert_eq!(d.sliding_window, 3);
    assert_eq!(d.norm_eps, 1e-6);

    assert_eq!(w.meta.num_loops, 2);
    assert_eq!(w.meta.recurrent_start(), 2);
    assert_eq!(w.meta.output_start(), 4);
    assert_eq!(w.blocks.len(), BLOCKS as usize);
    for (i, b) in w.blocks.iter().enumerate() {
        let want = if i % 2 == 0 {
            DoryBlockKind::Attn
        } else {
            DoryBlockKind::Ffn
        };
        assert_eq!(b.kind, want, "block {i} kind");
    }
    // The profile mix: block 0 full (freq_base), blocks 2/4 SWA
    // (freq_base_2) — the SWA-window gate has a windowed block to bite
    // on.
    assert!(w.blocks[0].rope_primary);
    assert!(!w.blocks[2].rope_primary);
    assert!(!w.blocks[4].rope_primary);
}

/// A non-dory arch refuses loud and named (the T5 posture's seed).
#[test]
fn foreign_arch_refuses_loud() {
    let bytes = synth_gguf("llama", true);
    let path = write_tmp("i1221_dory_synth_llama", &bytes);
    let err = load_dory_weights(&path).expect_err("llama arch must refuse");
    let _ = std::fs::remove_file(&path);
    assert!(
        err.to_string().contains("llama"),
        "the refusal must name the arch: {err}"
    );
}

/// A missing load-bearing metadata key refuses loud and named.
#[test]
fn missing_meta_refuses_loud() {
    let bytes = synth_gguf("dory", false);
    let path = write_tmp("i1221_dory_synth_noloops", &bytes);
    let err = load_dory_weights(&path).expect_err("missing num_loops must refuse");
    let _ = std::fs::remove_file(&path);
    assert!(
        err.to_string().contains("dory.num_loops"),
        "the refusal must name the key: {err}"
    );
}

/// The forward end-to-end on the synthetic file: the k=1 identity
/// holds on LOADED weights (not just in-test synthetics) and the OOB
/// token refusal rides the same error type.
#[test]
fn forward_runs_on_the_loaded_synth() {
    let bytes = synth_gguf("dory", true);
    let path = write_tmp("i1221_dory_synth_fwd", &bytes);
    let w = load_dory_weights(&path).expect("synth dory loads");
    let _ = std::fs::remove_file(&path);

    let f = DoryForward::new(&w);
    let toks = [1u32, 5, 3, 9];
    let mut cache = DoryKvCache::new(&w, 1);
    let prefill = f.prefill(&toks, &mut cache).expect("prefill");

    let mut cache2 = DoryKvCache::new(&w, 1);
    let mut incremental = Vec::new();
    for &t in &toks {
        incremental.push(f.decode_step(t, &mut cache2).expect("decode"));
    }
    for (i, (a, b)) in prefill.iter().zip(incremental.iter()).enumerate() {
        for (j, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "loaded-weights identity: logits[{i}][{j}]"
            );
        }
    }

    let err = f
        .decode_step(VOCAB as u32, &mut cache)
        .expect_err("oob must refuse");
    assert!(matches!(err, DoryForwardError::TokenOutOfRange { .. }));
}
