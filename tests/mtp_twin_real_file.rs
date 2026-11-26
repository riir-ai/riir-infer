//! Issue 1253 T2 — the REAL-FILE MTP-twin census gate over
//! `Ternary-Bonsai-2.5-27B-PQ2_0-mtp.gguf` (PrismML, Apache-2.0; 7,492,338,944
//! bytes, sha256 `ededc9fc0a8a6a8dd30914d107b2d2ce7674d789f1d05a1a481f240609aa6c23`).
//!
//! Header-census assertions run whenever the file is present (`MTP_GGUF`
//! overrides the path); the full main-stack load runs at `--release` under
//! `--ignored` (minutes + ~2 GiB host RAM — a proof cell, not a gate ride).

#[path = "common/synth_bonsai2.rs"]
mod synth;

use riir_infer_core::gguf_loader::{GgufFile, load_qwen_deltanet_ternary_from_gguf};

fn gguf_path() -> Option<std::path::PathBuf> {
    let path = std::path::PathBuf::from(std::env::var("MTP_GGUF").unwrap_or_else(|_| {
        "../riir-train/data/Ternary-Bonsai-2.5-27B-PQ2_0-mtp.gguf".to_string()
    }));
    path.canonicalize().ok()
}

/// The census half: cheap (header + mmap'd tensor table only). Pins the
/// declared shape the loader must honor — 64 main + 1 declared MTP block.
#[test]
fn real_mtp_twin_header_census() {
    let Some(path) = gguf_path() else {
        eprintln!("SKIP: the 2.5 -mtp GGUF is not on this box (MTP_GGUF names it)");
        return;
    };
    let gguf = GgufFile::open(&path).expect("open the real 2.5 -mtp file");
    let census = riir_infer_core::gguf_loader::mtp_twin_census(&gguf);
    assert_eq!(census.declared_nextn, 1, "the twin declares nextn=1");
    assert_eq!(census.n_main, 64, "block_count 65 − nextn 1");
    assert_eq!(census.trailing_tensors, 15, "blk.64: the full draft block");
    assert_eq!(census.highest_block, Some(64));
    assert_eq!(
        census.nextn_names, 4,
        "eh_proj/enorm/hnorm/shared_head_norm"
    );
    assert!(census.covered(), "declared nextn=1 covers blk.64");
    let _ = synth::N_LAYER; // the synth mod is the kit include; keep the path live
}

/// The load half (PROOF CELL): the full main-stack load SUCCEEDS over the
/// twin with the disclosure — 64 main layers, invariants hold. Release +
/// --ignored (the bench note records the outcome).
#[test]
#[ignore]
fn real_mtp_twin_full_load_proof() {
    let Some(path) = gguf_path() else {
        eprintln!("SKIP: the 2.5 -mtp GGUF is not on this box (MTP_GGUF names it)");
        return;
    };
    let gguf = GgufFile::open(&path).expect("open the real 2.5 -mtp file");
    let (config, weights) =
        load_qwen_deltanet_ternary_from_gguf(&gguf, "").expect("the declared twin loads");
    assert_eq!(
        config.n_layer, 64,
        "the draft block never enters the main stack"
    );
    assert_eq!(weights.layers.len(), 64);
    assert!(weights.invariants_hold(), "G1 bit-plane invariant holds");
    assert!(weights.rotation.is_some(), "the folded contract parses");
}
