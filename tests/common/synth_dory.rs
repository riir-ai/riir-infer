//! The synthetic dory GGUF fixture kit — extracted from
//! `tests/i1221_dory_loader_synth.rs` at the i1221 T3(a) landing so the
//! spectra gates build the EXACT same contract files instead of a second
//! hand-copy of the writer.
//!
//! Include via `#[path = "common/synth_dory.rs"] mod synth_dory;` — cargo
//! does not auto-discover `tests/` subdirectories as targets. Everything
//! is `pub` and `#![allow(dead_code)]` by design: a fixture library
//! over-serves its include sites.
//!
//! Shape (the loader test's defaults): 6 physical blocks = input pair +
//! recurrent pair + output pair, block 0 full attention, blocks 2/4 SWA,
//! `dory.num_loops` = 2 when armed. Every tensor bakes its own flat
//! coordinates (the layout law's fixture form).

#![allow(dead_code)]

use super::synth::{TensorSpec, Val, bf16_payload, build_gguf};

pub const N_EMBD: u64 = 16;
pub const VOCAB: u64 = 11;
pub const N_HEAD: u64 = 4;
pub const N_KV: u64 = 2;
pub const HEAD_DIM: u64 = 4;
pub const MLP: u64 = 8;
pub const BLOCKS: u64 = 6; // input pair, recurrent pair, output pair

/// The exact coordinate value the fixture bakes into flat element
/// `idx`: `(idx + 1) * 0.03125` — a bf16-exact multiple of 2^-5 that
/// stays distinct per element (flat size 256 max → values ≤ 8.0).
pub fn coord(idx: usize) -> f32 {
    ((idx + 1) as f32) * 0.03125
}

/// The full synthetic dory GGUF (arch-name parameterized — the
/// foreign-arch refusal arms pass `"llama"`).
pub fn synth_gguf(arch: &str, with_num_loops: bool) -> Vec<u8> {
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
