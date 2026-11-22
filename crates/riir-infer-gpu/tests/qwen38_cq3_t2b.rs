#![cfg(feature = "cq3_calibrate")]
//! Issue 1186 T2b — the capture-seam fixture battery (plan 629; the ours
//! half's hermetic cells, no GPU: sinks, budget law, site mapping, ledger).
//!
//! The pins:
//! - **t2b-sink round-trip** — append → finish → re-read: header magic +
//!   format + d_in, row count recoverable from file length, values
//!   bit-identical (the f32 wire is raw LE, no rounding).
//! - **t2b-budget** — the budget is a PIN: the row that overruns fails
//!   loud naming the counts (the matched-budget law; never a silent
//!   widening).
//! - **t2b-shape** — a rows-slice not a multiple of d_in panics loud.
//! - **t2b-sites** — the site map is closed: every league matrix suffix
//!   maps to a site, non-matrix suffixes map to None, and the per-layer
//!   file names the driver derives match the session's sink names.
//! - **t2b-lazy** — an armed-but-quiet session writes no files (the sink
//!   opens on first append; the byte-identical unarmed posture).

use riir_infer_gpu::qwen38_cq3_capture::{
    CAPTURE_SITES, CQ3CAP_FORMAT_V1, CQ3CAP_MAGIC, CaptureSession, CaptureSink,
};

fn tmpdir(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!(
        "i1186_t2b_{tag}_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    std::fs::create_dir_all(&d).expect("mkdir");
    d
}

#[test]
fn sink_round_trip_is_bit_exact() {
    let dir = tmpdir("sink");
    let d_in = 8usize;
    let mut sink = CaptureSink::new("blk000_mlp_gateup_in".into(), d_in, &dir, 16);
    let rows: Vec<f32> = (0..2 * d_in).map(|i| (i as f32 + 0.5) * 0.25).collect();
    sink.append(&rows).expect("append 1");
    let rows2: Vec<f32> = (0..d_in).map(|i| -0.125 * i as f32).collect();
    sink.append(&rows2).expect("append 2");
    assert_eq!(sink.rows_written(), 3);
    sink.finish().expect("finish");

    let bytes = std::fs::read(dir.join("blk000_mlp_gateup_in.cq3cap")).expect("read");
    assert_eq!(&bytes[..7], &CQ3CAP_MAGIC[..], "magic");
    let fmt = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    assert_eq!(fmt, CQ3CAP_FORMAT_V1);
    let file_d_in = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    assert_eq!(file_d_in, d_in);
    let vals: &[f32] = bytemuck::cast_slice(&bytes[16..]);
    assert_eq!(vals.len(), 3 * d_in);
    assert!(
        vals.iter()
            .zip(rows.iter().chain(rows2.iter()))
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "round trip is bit-identical"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn budget_overrun_fails_loud() {
    let dir = tmpdir("budget");
    let d_in = 4usize;
    let mut sink = CaptureSink::new("blk001_mlp_down_in".into(), d_in, &dir, 3);
    sink.append(&[0.0; 8]).expect("2 rows fit");
    let err = sink
        .append(&[0.0; 12]) // 2 + 3 > 3
        .expect_err("overrun refuses");
    assert!(err.contains("budget overrun"), "err names the law: {err}");
    assert!(
        err.contains("(2+3 > 3 rows)"),
        "err carries the counts: {err}"
    );
    // The refused append wrote nothing: still 2 rows on disk.
    assert_eq!(sink.rows_written(), 2);
    sink.finish().expect("finish");
    let bytes = std::fs::read(dir.join("blk001_mlp_down_in.cq3cap")).expect("read");
    let vals: &[f32] = bytemuck::cast_slice(&bytes[16..]);
    assert_eq!(vals.len(), 2 * d_in, "the refused rows never landed");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[should_panic(expected = "not a multiple of d_in")]
fn ragged_rows_panic_loud() {
    let dir = tmpdir("ragged");
    let mut sink = CaptureSink::new("blk002_attn_qkv_in".into(), 8, &dir, 8);
    let _ = sink.append(&[0.0f32; 7]); // 7 % 8 != 0
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn site_map_is_closed_over_the_league_set() {
    // Every league matrix suffix maps; the file name the driver derives
    // matches the session's sink name for a layer.
    let league_attn = [
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_output",
        "ffn_gate",
        "ffn_up",
        "ffn_down",
    ];
    let league_gdn = [
        "attn_qkv",
        "attn_gate",
        "ssm_alpha",
        "ssm_beta",
        "ssm_out",
        "ffn_gate",
        "ffn_up",
        "ffn_down",
    ];
    for s in league_attn.iter().chain(league_gdn.iter()) {
        let name = format!("blk.7.{s}.weight");
        let site = riir_infer_gpu::qwen38_cq3_capture::site_for_matrix(&name);
        assert!(site.is_some(), "{name} has a site");
        assert!(
            CAPTURE_SITES.contains(&site.unwrap()),
            "{site:?} is a declared site"
        );
    }
    for s in [
        "attn_norm",
        "post_attention_norm",
        "attn_q_norm",
        "ssm_norm",
        "ssm_a",
    ] {
        let name = format!("blk.7.{s}.weight");
        assert!(
            riir_infer_gpu::qwen38_cq3_capture::site_for_matrix(&name).is_none(),
            "{name} is not a capture-fed matrix"
        );
    }
    // Session sink naming round-trip: layer index + site → the driver's
    // file name format.
    let dir = tmpdir("sites");
    let dims: Vec<(&'static str, usize)> = CAPTURE_SITES.iter().map(|&s| (s, 4usize)).collect();
    let mut session = CaptureSession::with_config(dir.clone(), 4, 2, &dims);
    let rows = vec![1.0f32; 4];
    session
        .tap_rows(1, "mlp_down_in", 1, 4, &rows)
        .expect("tap");
    session
        .finish(&riir_infer_gpu::qwen38_cq3_capture::CaptureLedgerMeta {
            corpus_blake3: "aa".into(),
            corpus_bytes: 1,
            tokens_captured: 1,
            chunk: 1,
            state_per_chunk: "fresh",
            window: "head",
        })
        .expect("finish");
    assert!(
        dir.join("blk001_mlp_down_in.cq3cap").exists(),
        "the (layer, site) file name"
    );
    assert!(dir.join("ledger.json").exists(), "the ledger lands");
    let ledger = std::fs::read_to_string(dir.join("ledger.json")).expect("ledger read");
    assert!(
        ledger.contains("\"window\": \"head\""),
        "the ledger carries the window identity (the calibration-integrity law)"
    );
    assert!(
        !dir.join("blk000_mlp_down_in.cq3cap").exists(),
        "untouched sinks write nothing (lazy)"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// The calibration-integrity law (issue 1186, 2026-11-22): two corpora
/// sharing their head (v2 = v1 + a tail continuation) collapse to ONE
/// arm under a head-window budget — the window selection must slice
/// DISTINCT token ranges, and the ledger must carry which window ran.
/// This is the class that produced byte-identical b1/b2 artifacts
/// (blake3 64322cd5 twice) and the one-byte-different opponent
/// imatrices; caught from artifacts post-hoc, pinned here pre-T3.
#[test]
fn same_prefix_corpora_distinct_windows_slice_differently() {
    use riir_infer_gpu::qwen38_cq3_capture::window_slice;
    // A 64-token corpus; v2 shares the first 32 tokens (the head) and
    // continues differently (the tail).
    let total = 64usize;
    let budget = 32usize;
    let (h0, h1) = window_slice(total, budget, "head").expect("head");
    let (t0, t1) = window_slice(total, budget, "tail").expect("tail");
    assert_eq!((h0, h1), (0, 32), "head = the first budget tokens");
    assert_eq!((t0, t1), (32, 64), "tail = the last budget tokens");
    assert_ne!(
        (h0, h1),
        (t0, t1),
        "prefix-sharing corpora MUST slice differently under distinct windows"
    );
    // The head windows of v1 and v2 are IDENTICAL token ranges (the
    // collapse v1.0's defect); the tail window reaches v2's continuation.
    // Refusals: unknown spelling, over-budget window.
    assert!(window_slice(total, budget, "middle").is_err());
    assert!(window_slice(total, total + 1, "tail").is_err());
    assert!(
        window_slice(total, total + 1, "head").is_ok(),
        "head clamps by the corpus check upstream; the slice is [0, budget) and the driver's length check refuses short corpora"
    );
}
