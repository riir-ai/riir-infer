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

use riir_infer_core::dory::{DoryBlockKind, DoryForwardError, load_dory_weights};
use riir_infer_core::dory::{DoryForward, DoryKvCache};
use synth::{TensorSpec, Val, bf16_payload, build_gguf, write_tmp};

const N_EMBD: u64 = 16;
const VOCAB: u64 = 11;
const N_HEAD: u64 = 4;
const N_KV: u64 = 2;
const HEAD_DIM: u64 = 4;
const MLP: u64 = 8;
const BLOCKS: u64 = 6; // input pair, recurrent pair, output pair

/// The exact coordinate value the fixture bakes into flat element
/// `idx`: `(idx + 1) * 0.03125` — a bf16-exact multiple of 2^-5 that
/// stays distinct per element (flat size 256 max → values ≤ 8.0).
fn coord(idx: usize) -> f32 {
    ((idx + 1) as f32) * 0.03125
}

fn synth_gguf(arch: &str, with_num_loops: bool) -> Vec<u8> {
    let mut md: Vec<(String, Val)> = vec![
        (
            "general.architecture".to_string(),
            Val::Str(arch.to_string()),
        ),
        ("dory.block_count".to_string(), Val::U64(BLOCKS)),
        ("dory.embedding_length".to_string(), Val::U64(N_EMBD)),
        ("dory.feed_forward_length".to_string(), Val::U64(MLP)),
        ("dory.attention.head_count".to_string(), Val::U64(N_HEAD)),
        ("dory.attention.head_count_kv".to_string(), Val::U64(N_KV)),
        ("dory.attention.key_length".to_string(), Val::U64(HEAD_DIM)),
        (
            "dory.attention.value_length".to_string(),
            Val::U64(HEAD_DIM),
        ),
        ("dory.attention.sliding_window".to_string(), Val::U64(3)),
        ("dory.rope.dimension_count".to_string(), Val::U64(HEAD_DIM)),
        ("dory.rope.freq_base".to_string(), Val::F64(1e4)),
        ("dory.rope.freq_base_2".to_string(), Val::F64(1e2)),
        ("dory.input_layer_count".to_string(), Val::U64(2)),
        ("dory.recurrent_layer_count".to_string(), Val::U64(2)),
        ("dory.output_layer_count".to_string(), Val::U64(2)),
        (
            "dory.recurrent_kv_cache_mode".to_string(),
            Val::Str("per_loop".to_string()),
        ),
        (
            "dory.rope.profile".to_string(),
            // Block 0 full attention (profile 2 = freq_base), blocks
            // 2/4 SWA (profile 1 = freq_base_2) — the SWA gate needs a
            // windowed attention block to bite.
            Val::ArrU64(vec![2, 1, 1, 1, 1, 1]),
        ),
    ];
    if with_num_loops {
        md.push(("dory.num_loops".to_string(), Val::U64(2)));
    }

    let mut tensors: Vec<TensorSpec> = Vec::new();

    // token_embd: ne = [n_embd, vocab] — flat element (v, e) lives at
    // v * n_embd + e (ne[0] fastest). The baked values ARE the coords.
    let mut te = vec![0.0f32; (N_EMBD * VOCAB) as usize];
    for v in 0..VOCAB as usize {
        for e in 0..N_EMBD as usize {
            te[v * N_EMBD as usize + e] = coord(v * N_EMBD as usize + e);
        }
    }
    tensors.push(TensorSpec {
        name: "token_embd.weight".into(),
        ne: vec![N_EMBD, VOCAB],
        ggml_type: 30,
        data: bf16_payload(&te),
    });
    // Untied head, same layout — coordinates ×32 (bf16-exact, disjoint
    // from wte's ×1/32 range).
    let oh: Vec<f32> = (0..(N_EMBD * VOCAB) as usize)
        .map(|i| ((i + 1) as f32) * 32.0)
        .collect();
    tensors.push(TensorSpec {
        name: "output.weight".into(),
        ne: vec![N_EMBD, VOCAB],
        ggml_type: 30,
        data: bf16_payload(&oh),
    });
    // output_scale: one coord vector over the vocab.
    let os: Vec<f32> = (0..VOCAB as usize).map(|i| 2.0 + coord(i)).collect();
    tensors.push(TensorSpec {
        name: "output_scale.weight".into(),
        ne: vec![VOCAB],
        ggml_type: 30,
        data: bf16_payload(&os),
    });

    // Per-block tensors: attention on even, FFN on odd — every matrix
    // bakes its own flat coordinates, every vector bakes coord(i).
    for b in 0..BLOCKS as usize {
        let alpha: Vec<f32> = (0..N_EMBD as usize).map(|i| coord(b * 100 + i)).collect();
        tensors.push(TensorSpec {
            name: format!("blk.{b}.residual_alpha.weight"),
            ne: vec![N_EMBD],
            ggml_type: 30,
            data: bf16_payload(&alpha),
        });
        if b % 2 == 0 {
            let q_dim = N_HEAD * HEAD_DIM;
            let kv_dim = N_KV * HEAD_DIM;
            // (inner = in_features, outer = out_features) — ne[0] is the
            // innermost dim (the layout law this gate pins).
            for (stem, inner, outer) in [
                ("attn_q", N_EMBD, q_dim),
                ("attn_k", N_EMBD, kv_dim),
                ("attn_v", N_EMBD, kv_dim),
                ("attn_gate", N_EMBD, q_dim),
                ("attn_output", q_dim, N_EMBD),
            ] {
                let n = (inner * outer) as usize;
                let data: Vec<f32> = (0..n).map(|i| coord(b * 100 + i) + 1.0).collect();
                tensors.push(TensorSpec {
                    name: format!("blk.{b}.{stem}.weight"),
                    ne: vec![inner, outer],
                    ggml_type: 30,
                    data: bf16_payload(&data),
                });
            }
            for (stem, n) in [("attn_gate_scale", q_dim), ("attn_qk_scale", q_dim)] {
                let data: Vec<f32> = (0..n as usize).map(|i| coord(b * 100 + i) + 2.0).collect();
                tensors.push(TensorSpec {
                    name: format!("blk.{b}.{stem}.weight"),
                    ne: vec![n],
                    ggml_type: 30,
                    data: bf16_payload(&data),
                });
            }
        } else {
            for (stem, inner, outer) in [
                ("ffn_gate", N_EMBD, MLP),
                ("ffn_up", N_EMBD, MLP),
                ("ffn_down", MLP, N_EMBD),
            ] {
                let n = (inner * outer) as usize;
                let data: Vec<f32> = (0..n).map(|i| coord(b * 100 + i) + 1.0).collect();
                tensors.push(TensorSpec {
                    name: format!("blk.{b}.{stem}.weight"),
                    ne: vec![inner, outer],
                    ggml_type: 30,
                    data: bf16_payload(&data),
                });
            }
            for (stem, n) in [("ffn_gate_scale", MLP), ("ffn_up_scale", MLP)] {
                let data: Vec<f32> = (0..n as usize).map(|i| coord(b * 100 + i) + 2.0).collect();
                tensors.push(TensorSpec {
                    name: format!("blk.{b}.{stem}.weight"),
                    ne: vec![n],
                    ggml_type: 30,
                    data: bf16_payload(&data),
                });
            }
        }
    }

    build_gguf(&md, &tensors)
}

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
