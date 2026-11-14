//! Issue 1181 T1 — the per-token host ENCODE-share demote gate, measured at
//! the post-1165-flip resident posture (27B folded, Metal, census posture).
//!
//! The watch lane's claim: per-token host command-encoding cost is the next
//! decode lever (a Metal indirect-command-buffer replay arm, the cubecl
//! #1993 shape). The pre-registered demote bar (issue 1181 T1): **DEMOTE
//! unless measured encode ≥ 1.0 ms/token.** This cell produces that number:
//!
//! - INCUMBENT decomposition (the Issue-1168 P1 shape re-read at the census
//!   posture): per token, `set_input_token + forward_dispatch_only` is the
//!   host ENQUEUE FRONT (pure dispatch encode — the queue was just drained
//!   by the previous token's read), `read_logits` is the GPU-serialized
//!   wait. P1 measured front 2.455 ms / read 15.328 ms at the pre-census
//!   posture.
//! - RESIDENT: per-call `resident_enqueue()` durations across a long walk.
//!   Deep-queue calls return after pure command encode; the queue-full tail
//!   and the periodic `memory_cleanup()` cadence read as spikes (the p90 /
//!   max columns). The encode estimate is the p10 of the distribution,
//!   cross-checked against the incumbent front median (the same
//!   `forward_dispatch_only` path, one model load).
//! - decode wall µs/token for both lanes — the comparison axis.
//!
//! Verdict discipline (the issue-1129 convention, same as
//! `plan626_resident_g2_ab`): the numbers are THE RECORD, printed and never
//! asserted; the bar comparison is printed as a verdict line and the issue
//! adjudicates. Box state comes from the runner's bench_preflight
//! PROVENANCE line (the cell's own Linux-only load gate is inherited from
//! the G2 cell and does not fire on macOS).

#![cfg(all(
    feature = "cubecl_runtime",
    feature = "ternary_gemv",
    feature = "ternary_gemm_batched",
    feature = "ternary_deltanet_chunked_prefill",
    feature = "ternary_attention_batched_prefill",
    feature = "deltanet_recurrence_rowpar",
    feature = "params_handle_cache",
    target_os = "macos",
    not(debug_assertions)
))]

use std::time::Instant;

use riir_infer_core::deltanet::ternary_weights::QwenDeltaNetTernaryWeights;
use riir_infer_gpu::TernaryDeltanetGpuForward;
use riir_infer_gpu::cubecl_runtime::CubeCLContext;

/// Measured tokens per incumbent walk (the P1-shape decomposition).
const WALK_INC: usize = 32;
/// Measured tokens per resident walk (12 bursts of 8 — the distribution).
const WALK_RES: usize = 96;
/// The harvest cadence (the resident ring depth; matches the serving lane).
const K: usize = 8;
/// Round pairs (warm-up excluded), ABBA-interleaved in one process.
const PAIRS: usize = 3;
/// The pre-registered demote bar (issue 1181 T1): µs/token.
const DEMOTE_BAR_US: f64 = 1000.0;

fn pct(sorted: &[f64], p: f64) -> f64 {
    let idx = (((sorted.len() as f64 - 1.0) * p).round() as usize).min(sorted.len() - 1);
    sorted[idx]
}

fn median(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn stats(label: &str, mut v: Vec<f64>) {
    let n = v.len();
    v.sort_by(|a, b| a.total_cmp(b));
    eprintln!(
        "[i1181] {label}: n={n} min={:.1} p10={:.1} p50={:.1} p90={:.1} max={:.1} µs",
        v[0],
        pct(&v, 0.10),
        pct(&v, 0.50),
        pct(&v, 0.90),
        v[n - 1],
    );
}

/// Host argmax last-index (the harness rule) — the incumbent arm's pick.
fn host_argmax_last(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .expect("non-empty logits")
        .0
}

/// The incumbent arm with the P1 decomposition: front (dispatch encode) vs
/// read (GPU wait). Returns (fronts_µs, reads_µs, wall_ms, tokens_picked).
fn walk_incumbent_instrumented(
    fwd: &mut TernaryDeltanetGpuForward,
    weights: &QwenDeltaNetTernaryWeights,
    seed: usize,
) -> (Vec<f64>, Vec<f64>, f64, Vec<u32>) {
    fwd.reset_state();
    let mut token = seed;
    let mut fronts = Vec::with_capacity(WALK_INC);
    let mut reads = Vec::with_capacity(WALK_INC);
    let mut picks = Vec::with_capacity(WALK_INC);
    let t0 = Instant::now();
    for _ in 0..WALK_INC {
        let f0 = Instant::now();
        fwd.set_input_token(weights, token);
        fwd.forward_dispatch_only();
        fronts.push(f0.elapsed().as_secs_f64() * 1e6);
        let r0 = Instant::now();
        let logits = fwd.read_logits();
        reads.push(r0.elapsed().as_secs_f64() * 1e6);
        token = host_argmax_last(&logits);
        picks.push(token as u32);
    }
    let wall = t0.elapsed().as_secs_f64() * 1000.0;
    (fronts, reads, wall, picks)
}

/// The resident arm with per-call enqueue timing. Returns
/// (enqueues_µs, harvests_µs, wall_ms, tokens_picked).
fn walk_resident_instrumented(
    fwd: &mut TernaryDeltanetGpuForward,
    seed: usize,
) -> (Vec<f64>, Vec<f64>, f64, Vec<u32>) {
    fwd.reset_state();
    fwd.resident_begin(seed, K);
    let before = TernaryDeltanetGpuForward::resident_launch_count();
    let mut enqueues = Vec::with_capacity(WALK_RES);
    let mut harvests = Vec::with_capacity(WALK_RES / K + 1);
    let mut picks = Vec::with_capacity(WALK_RES);
    let t0 = Instant::now();
    let mut pending = 0;
    for _ in 0..WALK_RES {
        let e0 = Instant::now();
        fwd.resident_enqueue();
        enqueues.push(e0.elapsed().as_secs_f64() * 1e6);
        pending += 1;
        if pending == K {
            let h0 = Instant::now();
            let ids = fwd.resident_harvest(K);
            harvests.push(h0.elapsed().as_secs_f64() * 1e6);
            picks.extend_from_slice(&ids);
            pending = 0;
        }
    }
    if pending > 0 {
        let h0 = Instant::now();
        let ids = fwd.resident_harvest(pending);
        harvests.push(h0.elapsed().as_secs_f64() * 1e6);
        picks.extend_from_slice(&ids);
    }
    let wall = t0.elapsed().as_secs_f64() * 1000.0;
    let delta = TernaryDeltanetGpuForward::resident_launch_count() - before;
    assert_eq!(delta, WALK_RES as u64, "resident reach: {delta} != {WALK_RES}");
    (enqueues, harvests, wall, picks)
}

#[test]
#[ignore = "league-model GPU lane — issue 1181 T1 demote gate; run on a GPU-exclusive box"]
fn issue1181_encode_share_probe() {
    // The in-process subset of the bench_preflight posture (the G2 cell's
    // gate, inherited verbatim; Linux-only, passes silently on macOS — box
    // state is the runner's PROVENANCE line).
    let load = std::fs::read_to_string("/proc/loadavg").ok().and_then(|s| {
        s.split_whitespace()
            .next()
            .and_then(|l| l.parse::<f64>().ok())
    });
    if let Some(l1) = load {
        assert!(
            l1 < 6.0,
            "T1 REFUSED: 1-min load {l1:.2} ≥ 6.0 — rerun on a quieter box"
        );
    }

    let Ok(ctx) = CubeCLContext::new() else {
        eprintln!("[skip] no GPU (CubeCLContext::new failed)");
        return;
    };
    // The bench_763 census posture (the plan-626 G2 lesson: without the
    // uniform-cache override the cell reads ~2x and the bar is
    // unadjudicable).
    cubecl::wgpu::set_uniform_cache_override(Some(true));

    let model_path = std::env::var("BONSAI_GGUF")
        .unwrap_or_else(|_| "../../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf".to_string());
    let model_path = std::path::PathBuf::from(model_path);
    eprintln!("[i1181] loading {} …", model_path.display());
    let t0 = Instant::now();
    let (mut config, weights) =
        riir_infer_core::gguf_loader::load_qwen_deltanet_ternary_weights_gguf(&model_path)
            .map_err(|e| format!("load: {e}"))
            .unwrap();
    if config.block_size > 4096 {
        config.block_size = 4096;
    }
    let vocab = config.vocab_size;
    let bos = config.bos_token;
    let mut fwd = TernaryDeltanetGpuForward::new(&ctx, &config, &weights);
    eprintln!(
        "[i1181] forward ready ({:.0}s) · vocab={vocab} · n_layer={} · folded={}",
        t0.elapsed().as_secs_f32(),
        config.n_layer,
        fwd.is_folded(),
    );
    let seed = bos;

    // Warm-up (excluded): one arm each, drives the lazy compile paths.
    let _ = walk_incumbent_instrumented(&mut fwd, &weights, seed);
    let _ = walk_resident_instrumented(&mut fwd, seed);

    // Walk identity cross-check: the two lanes must agree token-exactly on
    // the overlap (the plan-626 G1 law) — a cheap reach sanity for this
    // cell's numbers (one walk's picks, first WALK_INC tokens).
    let (_, _, _, inc_picks) = walk_incumbent_instrumented(&mut fwd, &weights, seed);
    let (_, _, _, res_picks) = walk_resident_instrumented(&mut fwd, seed);
    assert_eq!(
        inc_picks, res_picks[..WALK_INC],
        "lane walks diverged — the timing cells would be measuring different work"
    );

    let mut inc_front_all: Vec<f64> = Vec::new();
    let mut inc_read_all: Vec<f64> = Vec::new();
    let mut res_eq_all: Vec<f64> = Vec::new();
    let mut res_hv_all: Vec<f64> = Vec::new();
    let mut inc_wall: Vec<f64> = Vec::with_capacity(PAIRS);
    let mut res_wall: Vec<f64> = Vec::with_capacity(PAIRS);
    let mut host_busy_share: Vec<f64> = Vec::with_capacity(PAIRS);

    for pair in 0..PAIRS {
        // ABBA — incumbent, resident, resident, incumbent.
        let (f1, r1, w1, _) = walk_incumbent_instrumented(&mut fwd, &weights, seed);
        let (e1, h1, w2, _) = walk_resident_instrumented(&mut fwd, seed);
        let (e2, h2, w3, _) = walk_resident_instrumented(&mut fwd, seed);
        let (f2, r2, w4, _) = walk_incumbent_instrumented(&mut fwd, &weights, seed);
        inc_front_all.extend(f1.iter().chain(f2.iter()));
        inc_read_all.extend(r1.iter().chain(r2.iter()));
        res_eq_all.extend(e1.iter().chain(e2.iter()));
        res_hv_all.extend(h1.iter().chain(h2.iter()));
        inc_wall.push((w1 + w4) / 2.0);
        res_wall.push((w2 + w3) / 2.0);
        // Host busy share of the resident arm: every µs the host spent
        // inside enqueue + harvest over the pair's wall.
        let busy: f64 = e1.iter().chain(e2.iter()).sum::<f64>()
            + h1.iter().chain(h2.iter()).sum::<f64>();
        host_busy_share.push(busy / (w2 + w3) / 1000.0);
        eprintln!(
            "[i1181] pair {pair}: incumbent {:.2} ms/tok · resident {:.2} ms/tok · host busy share {:.1}%",
            (w1 + w4) / 2.0 / WALK_INC as f64,
            (w2 + w3) / 2.0 / WALK_RES as f64,
            host_busy_share[pair] * 100.0,
        );
    }

    // ── THE RECORD ──
    stats("incumbent front (dispatch encode)", inc_front_all.clone());
    stats("incumbent read (GPU wait)", inc_read_all.clone());
    stats("resident enqueue (per call)", res_eq_all.clone());
    stats("resident harvest (per burst)", res_hv_all.clone());
    let inc_ms = median(&mut inc_wall) / WALK_INC as f64;
    let res_ms = median(&mut res_wall) / WALK_RES as f64;
    let share = median(&mut host_busy_share);
    res_eq_all.sort_by(|a, b| a.total_cmp(b));
    let encode_p10 = pct(&res_eq_all, 0.10);
    let encode_min = res_eq_all[0];
    let inc_front = {
        let mut v = inc_front_all.clone();
        median(&mut v)
    };
    eprintln!("═════════════════════════════════════════════════════════");
    eprintln!(
        "[i1181] decode wall: incumbent {:.2} ms/tok · resident {:.2} ms/tok (ratio {:.3}x)",
        inc_ms,
        res_ms,
        res_ms / inc_ms
    );
    eprintln!(
        "[i1181] ENCODE per token: resident p10 = {:.1} µs (min {:.1} µs) · incumbent front median = {:.1} µs",
        encode_p10, encode_min, inc_front
    );
    eprintln!(
        "[i1181] resident host busy share (enqueue+harvest over wall): {:.1}%",
        share * 100.0
    );
    // The pre-registered verdict (printed, never asserted — issue 1129
    // convention). The bar reads on the resident p10 (deep-queue pure
    // encode); the incumbent front median is the cross-check. A figure
    // above the bar keeps the watch lane OPEN (T2+ become candidates);
    // below it DEMOTES the lane (T2–T4 die).
    let verdict = if encode_p10 >= DEMOTE_BAR_US {
        "LANE OPEN — encode ≥ 1.0 ms/token bar MET"
    } else {
        "DEMOTE — encode < 1.0 ms/token bar (the replay arm has no room)"
    };
    eprintln!("[i1181] VERDICT: {verdict}");
}
