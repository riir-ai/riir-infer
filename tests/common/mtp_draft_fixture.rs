//! The MTP draft-block fixture kit (Issue 1253 T5.1/T5.2) — the
//! real-NAME, real-SHAPE 15-tensor draft block over the synthetic config,
//! shared by `mtp_draft_block_gates.rs` (T5.1 admission) and
//! `mtp_draft_step_gates.rs` (T5.2 reference step).
//!
//! Include via `#[path = "common/mtp_draft_fixture.rs"] mod fixture;`.
//! Real names + geometry pinned from the actual 2.5 MTP twin header
//! (15 tensors at blk.64, read 2026-11-27 from
//! `Ternary-Bonsai-2.5-27B-PQ2_0-mtp.gguf`); dims ride the synthetic
//! config (synth_bonsai2.rs). Everything `pub` + `#![allow(dead_code)]`
//! by design (a fixture library over-serves its include sites).
#![allow(dead_code)]

#[path = "synth_bonsai2.rs"]
pub mod synth;

/// Deterministic Q4_0 payload with an f16-EXACT scale: d = 0.25, element j
/// of block b carries nibble n = (2j + b + seed) % 16 (lo nibble = element
/// j, hi = element j+16 per the loader), so the dequant round-trip is
/// EXACT (d·(n−8) is a quarter-integer — f16-exact) and the gates pin
/// recovered values, not just shapes.
pub fn q4_0_payload(n_elements: u64, seed: u64) -> Vec<u8> {
    assert!(
        n_elements.is_multiple_of(32),
        "Q4_0 payloads are 32-aligned"
    );
    let n_blocks = (n_elements / 32) as usize;
    let mut out = Vec::with_capacity(n_blocks * 18);
    let d = half::f16::from_f32(0.25);
    for b in 0..n_blocks {
        out.extend_from_slice(&d.to_bits().to_le_bytes());
        let mut qs = [0u8; 16];
        for (j, q) in qs.iter_mut().enumerate() {
            let lo = ((2 * j + b + seed as usize) % 16) as u8;
            let hi = ((2 * j + 1 + b + seed as usize) % 16) as u8;
            *q = lo | (hi << 4);
        }
        out.extend_from_slice(&qs);
    }
    out
}

/// The value `q4_0_payload` encodes (element e of block b) — mirrors the
/// encoder's nibble layout: block position p<16 rides the lo nibble of
/// byte p, p≥16 the hi nibble of byte p−16.
pub fn q4_0_value(e: usize, b: usize, seed: u64) -> f32 {
    let pos = e % 32;
    let j = pos & 15;
    let n = if pos < 16 {
        (2 * j + b + seed as usize) % 16
    } else {
        (2 * j + 1 + b + seed as usize) % 16
    };
    0.25 * (n as f32 - 8.0)
}

/// One draft block at `blk.{N_LAYER}` with the REAL tensor names and the
/// synthetic config's shapes (q = 2·q_dim gated-attention concat, Issue
/// 594; per-head q/k norms [head_dim]; the three optional nextn tensors
/// absent — the real 2.5 twin's observed shape).
pub fn mtp_draft_block_tensors() -> Vec<synth::TensorSpec> {
    let i = synth::N_LAYER;
    let n = synth::N_EMBD;
    let q_dim = synth::N_HEAD * synth::HEAD_DIM;
    let kv_dim = synth::N_KV * synth::HEAD_DIM;
    let mlp = synth::FFN;
    let head_dim = synth::HEAD_DIM;
    let mut seed = 1u64;
    let mut proj = |name: String, ne0: u64, ne1: u64| {
        seed += 1;
        synth::TensorSpec {
            name,
            ne: vec![ne0, ne1],
            ggml_type: 2, // Q4_0
            data: q4_0_payload(ne0 * ne1, seed),
        }
    };
    let norm = |name: String, len: u64, fill: f32| synth::TensorSpec {
        name,
        ne: vec![len],
        ggml_type: 0, // F32
        data: synth::f32_payload(&vec![fill; len as usize]),
    };
    let mut out = vec![
        proj(format!("blk.{i}.attn_q.weight"), n, 2 * q_dim),
        proj(format!("blk.{i}.attn_k.weight"), n, kv_dim),
        proj(format!("blk.{i}.attn_v.weight"), n, kv_dim),
        proj(format!("blk.{i}.attn_output.weight"), q_dim, n),
        norm(format!("blk.{i}.attn_q_norm.weight"), head_dim, 1.0),
        norm(format!("blk.{i}.attn_k_norm.weight"), head_dim, 1.0),
        norm(format!("blk.{i}.attn_norm.weight"), n, 1.0),
        norm(format!("blk.{i}.post_attention_norm.weight"), n, 1.0),
        proj(format!("blk.{i}.ffn_gate.weight"), n, mlp),
        proj(format!("blk.{i}.ffn_up.weight"), n, mlp),
        proj(format!("blk.{i}.ffn_down.weight"), mlp, n),
        proj(format!("blk.{i}.nextn.eh_proj.weight"), 2 * n, n),
        norm(format!("blk.{i}.nextn.enorm.weight"), n, 1.0),
        norm(format!("blk.{i}.nextn.hnorm.weight"), n, 1.0),
        norm(format!("blk.{i}.nextn.shared_head_norm.weight"), n, 1.0),
    ];
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// The fixture's seed per projection name — `mtp_draft_block_tensors`
/// assigns seeds in construction order (attn_q first → 2, attn_k → 3, …);
/// the value round-trip gates read them back through this map.
pub fn proj_seed(name_suffix: &str) -> u64 {
    match name_suffix {
        "attn_q" => 2,
        "attn_k" => 3,
        "attn_v" => 4,
        "attn_output" => 5,
        "ffn_gate" => 6,
        "ffn_up" => 7,
        "ffn_down" => 8,
        "eh_proj" => 9,
        other => panic!("no fixture seed for projection '{other}'"),
    }
}

/// The declared twin bytes: the synthetic main stack + the draft block +
/// `block_count`/`nextn_predict_layers` covering it.
pub fn declared_twin_bytes() -> Vec<u8> {
    let mut meta = synth::synth_metadata(true);
    meta.push((
        "qwen35.block_count".into(),
        synth::Val::U64(synth::N_LAYER + 1),
    ));
    meta.push(("qwen35.nextn_predict_layers".into(), synth::Val::U64(1)));
    let mut all = synth::synth_tensors_bonsai2();
    all.extend(mtp_draft_block_tensors());
    synth::build_gguf(&meta, &all)
}
