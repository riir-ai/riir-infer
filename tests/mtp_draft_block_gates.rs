//! Issue 1253 T5.1 (plan 639) — the typed MTP draft-block admission gates
//! (synthetic, pure-CPU; the real-file arm rides the `MTP_GGUF` env when
//! the 2.5 `-mtp` twin is on the box).
//!
//! Pins the admission contract `load_mtp_draft_block` introduces beside the
//! T2 census gate:
//! 1. DECLARED twin admits — all 15 real-shape tensors parse into
//!    `MtpDraftBlock` with the shapes pinned against the config, the
//!    Q4_0 projections dequant EXACTLY at the fixture's f16-exact scale,
//!    and the three optional nextn tensors read `None` (the real twin's
//!    observed shape).
//! 2. TORN twin refuses loud — one missing required tensor names itself
//!    (a file declaring layers but carrying a partial block).
//! 3. CLEAN file refuses loud — a plain file drafts nothing (never a
//!    silent empty drafter).
//! 4. SHAPE mismatch refuses loud — a wrong projection width names the
//!    expected [in × out] pair.
//! 5. The census predicate `draft_admittable` reads false on
//!    clean/undeclared and true only on the declared twin (the admitted
//!    class the disclosure names).
//! 6. The UNDECLARED class reads uncovered (the main-stack refuse is the
//!    T2 gate's, re-asserted there; the admission never bypasses it).
//!
//! Real names + geometry are pinned from the actual 2.5 MTP twin header
//! (15 tensors at blk.64, read 2026-11-27 from
//! `Ternary-Bonsai-2.5-27B-PQ2_0-mtp.gguf`): the 11 attention/FFN tensors
//! under the STANDARD main-layer names + 4 `nextn.*` tensors; Q4_0
//! projections (ggml type 2) dequant through the loader's existing arm.

#[path = "common/mtp_draft_fixture.rs"]
mod fixture;

// The fixture owns the synth kit module (one TensorSpec type identity);
// the spellings below ride it.
use fixture::synth;
use fixture::{mtp_draft_block_tensors, proj_seed, q4_0_value};
use riir_infer_core::gguf_loader::{
    GgufFile, load_mtp_draft_block, mtp_twin_census, qwen35_deltanet_config_from_gguf_metadata,
};

use std::io::Write as _;

/// The metadata-only config walk (one geometry path — the loader's own
/// parser; no ternary load in a pure-CPU gate battery).
fn config_of(gguf: &GgufFile) -> riir_infer_core::types::Config {
    qwen35_deltanet_config_from_gguf_metadata(gguf)
        .map(|(c, _)| c)
        .expect("config parses")
}

fn write_tmp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("{name}_{}.gguf", std::process::id()));
    let mut f = std::fs::File::create(&path).expect("create temp gguf");
    f.write_all(bytes).expect("write temp gguf");
    path
}

/// Arm 1 — the declared twin ADMITS: 15 tensors parse, shapes pin against
/// the config, Q4_0 dequant round-trips EXACTLY at the fixture scale, the
/// three optional nextn tensors read None, and the admitted-class census
/// predicate is true.
#[test]
fn mtp_draft_block_declared_twin_admits() {
    let bytes = fixture::declared_twin_bytes();
    assert_eq!(
        mtp_draft_block_tensors().len(),
        15,
        "the real twin's 15-tensor class"
    );
    let path = write_tmp("riir_1253_t51_declared", &bytes);

    let gguf = GgufFile::open(&path).expect("open declared twin");
    let census = mtp_twin_census(&gguf);
    assert!(census.draft_admittable(), "declared twin admits");
    assert_eq!(census.trailing_tensors, 15);

    let config = config_of(&gguf);
    let block = load_mtp_draft_block(&gguf, &config).expect("draft block admits");
    assert_eq!(block.block_index, synth::N_LAYER as usize);

    let n = synth::N_EMBD as usize;
    let q_dim = (synth::N_HEAD * synth::HEAD_DIM) as usize;
    let kv_dim = (synth::N_KV * synth::HEAD_DIM) as usize;
    let mlp = synth::FFN as usize;
    assert_eq!(
        block.attn_q.len(),
        n * 2 * q_dim,
        "attn_q [n_embd × 2·q_dim]"
    );
    assert_eq!(block.attn_k.len(), n * kv_dim);
    assert_eq!(block.attn_v.len(), n * kv_dim);
    assert_eq!(block.attn_output.len(), q_dim * n);
    assert_eq!(block.attn_q_norm.len(), synth::HEAD_DIM as usize);
    assert_eq!(block.attn_k_norm.len(), synth::HEAD_DIM as usize);
    assert_eq!(block.attn_norm.len(), n);
    assert_eq!(block.post_attention_norm.len(), n);
    assert_eq!(block.ffn_gate.len(), n * mlp);
    assert_eq!(block.ffn_up.len(), n * mlp);
    assert_eq!(block.ffn_down.len(), mlp * n);
    assert_eq!(block.eh_proj.len(), 2 * n * n);
    assert_eq!(block.enorm.len(), n);
    assert_eq!(block.hnorm.len(), n);
    assert!(block.shared_head_norm.is_some(), "the real twin carries it");
    assert!(block.shared_head_head.is_none(), "the real twin omits it");
    assert!(block.embed_tokens.is_none(), "the real twin omits it");

    // The Q4_0 round-trip is EXACT at the fixture scale (d = 0.25,
    // quarter-integer values): spot-check attn_q across block boundaries.
    let seed = proj_seed("attn_q");
    for row in [0usize, 1, q_dim, 2 * q_dim - 1] {
        for col in [0usize, 15, 16, 31, n - 1] {
            let e = row * n + col;
            let b = e / 32;
            let expect = q4_0_value(e, b, seed);
            assert_eq!(
                block.attn_q[e], expect,
                "attn_q[{row},{col}] Q4_0 round-trip"
            );
        }
    }
    let _ = std::fs::remove_file(&path);
}

/// Arm 2 — the TORN twin refuses loud: dropping one required tensor
/// (attn_k) names the missing tensor in the refusal.
#[test]
fn mtp_draft_block_torn_twin_refuses_loud() {
    let tensors: Vec<synth::TensorSpec> = mtp_draft_block_tensors()
        .into_iter()
        .filter(|t| !t.name.ends_with("blk.4.attn_k.weight"))
        .collect();
    let mut meta = synth::synth_metadata(true);
    meta.push((
        "qwen35.block_count".into(),
        synth::Val::U64(synth::N_LAYER + 1),
    ));
    meta.push(("qwen35.nextn_predict_layers".into(), synth::Val::U64(1)));
    let mut all = synth::synth_tensors_bonsai2();
    all.extend(tensors);
    let bytes = synth::build_gguf(&meta, &all);
    let path = write_tmp("riir_1253_t51_torn", &bytes);

    let gguf = GgufFile::open(&path).expect("open torn twin");
    let config = config_of(&gguf);
    let err = match load_mtp_draft_block(&gguf, &config) {
        Ok(_) => panic!("torn twin must refuse"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("blk.4.attn_k.weight"),
        "refusal must name the missing tensor, got: {msg}"
    );
    assert!(
        msg.contains("torn-twin"),
        "refusal must name the torn-twin class, got: {msg}"
    );
    let _ = std::fs::remove_file(&path);
}

/// Arm 3 — the CLEAN file refuses loud: a plain file drafts nothing.
#[test]
fn mtp_draft_block_clean_file_refuses() {
    let mut meta = synth::synth_metadata(true);
    meta.push(("qwen35.block_count".into(), synth::Val::U64(synth::N_LAYER)));
    let bytes = synth::build_gguf(&meta, &synth::synth_tensors_bonsai2());
    let path = write_tmp("riir_1253_t51_clean", &bytes);

    let gguf = GgufFile::open(&path).expect("open clean file");
    let census = mtp_twin_census(&gguf);
    assert!(!census.draft_admittable(), "clean file does not admit");
    let config = config_of(&gguf);
    let err = match load_mtp_draft_block(&gguf, &config) {
        Ok(_) => panic!("clean file must refuse draft admission"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("no admissible draft block"),
        "refusal must name the no-block class, got: {msg}"
    );
    let _ = std::fs::remove_file(&path);
}

/// Arm 4 — a SHAPE mismatch refuses loud naming the expected pair: eh_proj
/// with a half-width input (the compose contract is 2·n_embd).
#[test]
fn mtp_draft_block_shape_mismatch_refuses() {
    let tensors: Vec<synth::TensorSpec> = mtp_draft_block_tensors()
        .into_iter()
        .map(|mut t| {
            if t.name.ends_with("nextn.eh_proj.weight") {
                t.ne = vec![synth::N_EMBD, synth::N_EMBD]; // wrong: half the input
            }
            t
        })
        .collect();
    let mut meta = synth::synth_metadata(true);
    meta.push((
        "qwen35.block_count".into(),
        synth::Val::U64(synth::N_LAYER + 1),
    ));
    meta.push(("qwen35.nextn_predict_layers".into(), synth::Val::U64(1)));
    let mut all = synth::synth_tensors_bonsai2();
    all.extend(tensors);
    let bytes = synth::build_gguf(&meta, &all);
    let path = write_tmp("riir_1253_t51_shape", &bytes);

    let gguf = GgufFile::open(&path).expect("open shape-mismatch twin");
    let config = config_of(&gguf);
    let err = match load_mtp_draft_block(&gguf, &config) {
        Ok(_) => panic!("shape mismatch must refuse"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("eh_proj") && msg.contains("[2048 × 1024]"),
        "refusal must name the tensor + the expected [in × out] pair, got: {msg}"
    );
    let _ = std::fs::remove_file(&path);
}

/// Arm 5 — the UNDECLARED class reads uncovered (the draft admission
/// refuses too; the main-stack refuse is the T2 gate's own row).
#[test]
fn mtp_draft_block_undeclared_admits_nothing() {
    let mut meta = synth::synth_metadata(true);
    meta.push((
        "qwen35.block_count".into(),
        synth::Val::U64(synth::N_LAYER + 1),
    ));
    // NO nextn_predict_layers — the undeclared class.
    let mut all = synth::synth_tensors_bonsai2();
    all.extend(mtp_draft_block_tensors());
    let bytes = synth::build_gguf(&meta, &all);
    let path = write_tmp("riir_1253_t51_undeclared", &bytes);

    let gguf = GgufFile::open(&path).expect("open undeclared twin");
    let census = mtp_twin_census(&gguf);
    assert!(
        !census.draft_admittable(),
        "undeclared twins admit nothing — the T2 declare-or-refuse law holds \
         upstream (gate: mtp_twin_load_gates, ternary posture)"
    );
    assert!(
        !census.covered(),
        "the undeclared class reads uncovered (the main-stack load refuses there)"
    );
    let _ = std::fs::remove_file(&path);
}

/// Arm 6 — the REAL-FILE arm (skip loud without the twin): over the actual
/// 2.5 `-mtp` file the admission pins the observed 15-tensor class —
/// n_embd 5120, attn_q [5120 × 12288], eh_proj [10240 × 5120], per-head
/// norms [256], shared_head_norm present, shared_head_head/embed_tokens
/// absent. Header + the 15-tensor reads only.
#[test]
fn mtp_draft_block_real_file_admits() {
    let path = std::path::PathBuf::from(std::env::var("MTP_GGUF").unwrap_or_else(|_| {
        "../riir-train/data/Ternary-Bonsai-2.5-27B-PQ2_0-mtp.gguf".to_string()
    }));
    let Some(path) = path.canonicalize().ok() else {
        eprintln!("SKIP: the 2.5 -mtp GGUF is not on this box (MTP_GGUF names it)");
        return;
    };
    let gguf = GgufFile::open(&path).expect("open the real 2.5 -mtp file");
    let census = mtp_twin_census(&gguf);
    assert_eq!(census.n_main, 64);
    assert!(census.draft_admittable());

    let config = config_of(&gguf);
    let block = load_mtp_draft_block(&gguf, &config).expect("the real twin admits");
    assert_eq!(block.block_index, 64);
    assert_eq!(block.attn_q.len(), 5120 * 12288);
    assert_eq!(block.attn_k.len(), 5120 * 1024);
    assert_eq!(block.attn_output.len(), 6144 * 5120);
    assert_eq!(block.attn_q_norm.len(), 256);
    assert_eq!(block.attn_k_norm.len(), 256);
    assert_eq!(block.ffn_gate.len(), 5120 * 17408);
    assert_eq!(block.eh_proj.len(), 10240 * 5120);
    assert!(block.shared_head_norm.is_some());
    assert!(block.shared_head_head.is_none());
    assert!(block.embed_tokens.is_none());
    // Q4_0 values are finite + bounded (|v| ≤ 8·d_max per group scale).
    assert!(
        block.attn_q.iter().all(|v| v.is_finite()),
        "attn_q dequant is finite"
    );
}
