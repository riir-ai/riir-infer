//! i1226 F2 — the producer-writes-f16 PRICE CELL (the gate after G1).
//!
//! The issue-1225 frame posture (the promoted serving set, chunk × frames
//! passes, ABBA-interleaved pairs, per-arm bit anchors) with the incumbent
//! moved to where the gate order puts it: **arm B = the armed f16-B twin
//! with its per-launch `cast_f32_f16_zc` (Bench 1225's armed posture)**,
//! **arm C = B + the producer shadow** (`set_f16b_producer(true)` — the
//! rotate writes the f16 pairs, the twin skips the cast encoder). Both
//! latches are read per call, so the arms alternate inside one process.
//!
//! What the cell asserts (a failure here voids the price, it is not a
//! perf verdict):
//! - **engagement** — C skips cast encoders (`cast_skipped_count` > 0) and
//!   runs fewer of them than B; B skips none; no shadow fell back on a
//!   misaligned offset (`shadow_misaligned_count` unchanged).
//! - **bit identity, end to end** — every C pass's logits are
//!   byte-identical to B's (G1 proved shadow == cast elementwise; a frame
//!   that differs is a stale-shadow / lifetime bug, never noise).
//! - **anchors** — each arm is deterministic across its passes, and the
//!   f32 lane (arm A, route off) differs from B (the twin is live).
//!
//! What it REPORTS (the price; the GOAT/verdict steps read it): the saved
//! cast launches per pass and per layer, the per-pair B−C Δ (median, range,
//! sign consistency), the Δ against the Bench-1223 per-launch prediction
//! (0.15–0.19 ms × saved launches), and the one-line f32-lane delta (arm A
//! runs once per pair AFTER the B/C block — informational, NOT interleaved,
//! so it is never the verdict's Δ).
//!
//! ```sh
//! CARGO_TARGET_DIR=/tmp/i1226f2 cargo test -p riir-infer-gpu --release \
//!   --features prefill_f16b,production,cubecl_runtime,multi_host,riir-infer-core/bonsai2_hadamard \
//!   --test i1226_f2_price_cell -- --ignored --nocapture --test-threads=1
//! # I1226P_PAIRS (default 5) · I1226P_CHUNK (512) · I1226P_FRAMES (8 = pp4096; 64 = pp32k)
//! # I1226P_TAIL (2) — tokens past chunk × frames (a ragged final frame)
//! # I1226P_IDENTITY_ONLY=1 — the delegated verdict's condition (b): engagement +
//! #   C == B identity at another shape (e.g. chunk 128 = the dist default, tail 37),
//! #   no timed pairs
//! ```

#![cfg(all(
    feature = "prefill_f16b",
    feature = "metal_tensor_gemm",
    feature = "cubecl_runtime",
    feature = "ternary_gemv",
    feature = "multi_host",
    feature = "ternary_gemm_batched",
    target_os = "macos"
))]

#[path = "support/f16b_frame.rs"]
mod f16b_frame;

use std::path::PathBuf;

use f16b_frame::{FOLDED_MODEL, SEED_TEXT, median, run_pass};
use riir_infer_core::gguf_loader::{GgufFile, load_qwen_deltanet_ternary_weights_gguf};
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_gpu::TernaryDeltanetGpuForward;
use riir_infer_gpu::cubecl_runtime::CubeCLContext;
use riir_infer_gpu::gemm_ternary_metal_tensor::set_metal_tensor_relaxed_precision;
use riir_infer_gpu::prefill_f16b::{
    cast_encoder_count, cast_skipped_count, set_f16b_placebo, set_f16b_producer, set_f16b_route,
    shadow_misaligned_count,
};
use riir_infer_gpu::{set_prefill_attn_metal_tensor, set_prefill_use_metal_tensor_zerocopy};

/// Bench 1223's measured per-launch cast tax band (ms) — the prediction the
/// measured Δ is set beside.
const CAST_TAX_MS: (f64, f64) = (0.15, 0.19);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arm {
    /// f32 lane (route off) — informational.
    A,
    /// armed twin, cast per launch — the incumbent.
    B,
    /// armed twin + producer shadow — the candidate.
    C,
}

fn set_arm(arm: Arm) {
    set_f16b_route(arm != Arm::A);
    set_f16b_producer(arm == Arm::C);
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[test]
#[ignore = "requires the Metal GPU + the Bonsai-2 PQ2_0 model; run explicitly with --release"]
fn f2_price_cell() {
    let model = std::env::var_os("BONSAI2_G1_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(FOLDED_MODEL));
    assert!(
        model.exists(),
        "the Bonsai-2 league pin is the cell's premise — not found at {}",
        model.display()
    );
    let pairs = env_usize("I1226P_PAIRS", 5);
    assert!(pairs >= 3, "need at least 3 pairs for a median");
    let chunk = env_usize("I1226P_CHUNK", 512);
    let frames = env_usize("I1226P_FRAMES", 8);
    let tail = env_usize("I1226P_TAIL", 2);
    let identity_only = std::env::var("I1226P_IDENTITY_ONLY").as_deref() == Ok("1");

    // The serving posture (relaxed — the promoted dist default), identical
    // for every arm; the ONLY deltas are the route and producer latches.
    set_prefill_use_metal_tensor_zerocopy(true);
    set_prefill_attn_metal_tensor(true);
    set_metal_tensor_relaxed_precision(true);
    set_f16b_placebo(false);
    // The twin is a construction-time posture: route pinned ON before
    // `new`, the per-pass latches decide who serves.
    set_f16b_route(true);

    let tok = {
        let gguf = GgufFile::open(&model).expect("open gguf for tokenizer");
        BpeTokenizer::from_gguf(&gguf).expect("bpe tokenizer from gguf")
    };
    let (mut config, weights) =
        load_qwen_deltanet_ternary_weights_gguf(&model).expect("folded model load");
    let n_layer = config.n_layer;
    let toks: Vec<usize> = tok
        .encode(SEED_TEXT)
        .iter()
        .cycle()
        .take(chunk * frames + tail)
        .copied()
        .collect();
    config.block_size = toks.len() + 8;

    let ctx = CubeCLContext::new().expect("GPU init");
    let mut fwd = TernaryDeltanetGpuForward::new(&ctx, &config, &weights);
    println!(
        "# i1226 F2 price cell · relaxed · {pairs} ABBA pairs (B=twin+cast, C=twin+shadow) · \
         chunk {chunk} × frames {frames} = {} tok/pass · {n_layer} layers",
        toks.len()
    );

    let mut pass = |arm: Arm| -> (f64, String) {
        set_arm(arm);
        fwd.reset_state();
        run_pass(&mut fwd, &toks, chunk, n_layer)
    };

    // 2 untimed compile passes per arm (kernel JIT + cache warm).
    for arm in [Arm::A, Arm::B, Arm::C] {
        for _ in 0..2 {
            let _ = pass(arm);
        }
    }

    // ── Engagement: one counted pass per twin arm ─────────────────────
    let counted = |arm: Arm, pass: &mut dyn FnMut(Arm) -> (f64, String)| {
        let (e0, s0, m0) = (
            cast_encoder_count(),
            cast_skipped_count(),
            shadow_misaligned_count(),
        );
        let _ = pass(arm);
        (
            cast_encoder_count() - e0,
            cast_skipped_count() - s0,
            shadow_misaligned_count() - m0,
        )
    };
    let (b_enc, b_skip, b_mis) = counted(Arm::B, &mut pass);
    let (c_enc, c_skip, c_mis) = counted(Arm::C, &mut pass);
    let per_layer = |n: u64| n as f64 / (frames * n_layer) as f64;
    println!(
        "# engagement per pass: B cast encoders {b_enc} (skipped {b_skip}) · \
         C cast encoders {c_enc} (skipped {c_skip}) · misaligned B {b_mis} C {c_mis}\n\
         # saved cast launches: {} per pass = {:.2} per layer per frame",
        b_enc.saturating_sub(c_enc),
        per_layer(b_enc.saturating_sub(c_enc))
    );
    assert!(
        b_enc > 0,
        "arm B ran no cast encoders — the twin never engaged"
    );
    assert_eq!(
        b_skip, 0,
        "arm B skipped casts with the producer OFF — latch leak"
    );
    assert!(
        c_skip > 0,
        "arm C skipped no casts — the producer shadow never bound"
    );
    assert!(
        c_enc < b_enc,
        "arm C ran {c_enc} cast encoders, not fewer than B's {b_enc}"
    );
    assert_eq!(
        b_mis + c_mis,
        0,
        "a shadow fell back on a misaligned offset — the price would be mixed"
    );
    let saved = b_enc - c_enc;

    if identity_only {
        // Condition (b): the shadow's end-to-end identity at a shape the
        // price cell did not time — two passes per twin arm, every digest
        // equal across AND between arms.
        let digests: Vec<(Arm, String)> = [Arm::B, Arm::C, Arm::C, Arm::B]
            .into_iter()
            .map(|arm| (arm, pass(arm).1))
            .collect();
        set_arm(Arm::B);
        assert!(
            digests.windows(2).all(|w| w[0].1 == w[1].1),
            "C != B (or an arm is non-deterministic) at chunk {chunk} × frames {frames} + tail \
             {tail}: {digests:?}"
        );
        println!(
            "\nF2 IDENTITY CELL (chunk {chunk} × frames {frames} + tail {tail} = {} tok): \
             C == B byte-identical × {} passes; {saved} casts saved/pass; misaligned 0 — PASS",
            toks.len(),
            digests.len()
        );
        return;
    }

    // ── The timed pairs ───────────────────────────────────────────────
    let (mut b_walls, mut c_walls, mut a_walls) = (Vec::new(), Vec::new(), Vec::new());
    let (mut b_dig, mut c_dig, mut a_dig) = (Vec::new(), Vec::new(), Vec::new());
    let mut deltas: Vec<f64> = Vec::with_capacity(pairs);
    for i in 0..pairs {
        let order = if i % 2 == 0 {
            [Arm::B, Arm::C, Arm::C, Arm::B]
        } else {
            [Arm::C, Arm::B, Arm::B, Arm::C]
        };
        let (mut b_sum, mut c_sum) = (0.0, 0.0);
        for arm in order {
            let (wall, digest) = pass(arm);
            match arm {
                Arm::B => {
                    b_sum += wall;
                    b_walls.push(wall);
                    b_dig.push(digest);
                }
                Arm::C => {
                    c_sum += wall;
                    c_walls.push(wall);
                    c_dig.push(digest);
                }
                Arm::A => unreachable!(),
            }
        }
        let (b_mean, c_mean) = (b_sum / 2.0, c_sum / 2.0);
        let delta = b_mean - c_mean;
        deltas.push(delta);
        // The informational f32 line — after the block, never in the Δ.
        let (a_wall, a_digest) = pass(Arm::A);
        a_walls.push(a_wall);
        a_dig.push(a_digest);
        println!(
            "  pair {i}: B mean {b_mean:.2} ms · C mean {c_mean:.2} ms · Δ {delta:.2} ms ({:.2}%) · A {a_wall:.2} ms",
            delta / b_mean * 100.0
        );
    }
    set_arm(Arm::B);

    // ── Anchors + end-to-end bit identity ─────────────────────────────
    for (name, d) in [("A", &a_dig), ("B", &b_dig), ("C", &c_dig)] {
        assert!(
            d.windows(2).all(|w| w[0] == w[1]),
            "arm {name} bit anchor BROKEN — the passes disagree: {d:?}"
        );
    }
    assert_eq!(
        b_dig[0], c_dig[0],
        "arm C's logits differ from arm B's — the shadow is NOT bit-identical to the cast \
         end to end (stale shadow / pool lifetime bug; G1 holds per kernel)"
    );
    assert_ne!(
        a_dig[0], b_dig[0],
        "arm B reproduced the f32 lane's bits — the twin never engaged"
    );
    println!(
        "# bit anchors: A × {}, B × {}, C × {} — HELD; C == B byte-identical (e2e shadow identity), A ≠ B",
        a_dig.len(),
        b_dig.len(),
        c_dig.len()
    );

    // ── The price ─────────────────────────────────────────────────────
    let (b_med, c_med, a_med) = (median(b_walls), median(c_walls), median(a_walls));
    let d_med = median(deltas.clone());
    let d_lo = deltas.iter().copied().fold(f64::INFINITY, f64::min);
    let d_hi = deltas.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let (p_lo, p_hi) = (saved as f64 * CAST_TAX_MS.0, saved as f64 * CAST_TAX_MS.1);
    let sign_consistent = deltas.iter().all(|&d| d > 0.0);
    println!(
        "\nB median {b_med:.2} ms · C median {c_med:.2} ms · Δ median {d_med:.2} ms \
         (range [{d_lo:.2}, {d_hi:.2}]) = {:.2}% of the armed-twin frame · \
         sign-consistent {sign_consistent}\n\
         prediction (Bench 1223 tax × {saved} saved launches): {p_lo:.2}–{p_hi:.2} ms \
         · measured / prediction-mid {:.2}\n\
         f32 lane (informational, not interleaved): A median {a_med:.2} ms → C is {:.2}% under A",
        d_med / b_med * 100.0,
        d_med / ((p_lo + p_hi) / 2.0),
        (a_med - c_med) / a_med * 100.0
    );
    let verdict = if sign_consistent && d_med > 0.0 {
        "GAIN — sign-consistent on every pair"
    } else if d_med > 0.0 {
        "UNDECIDED — positive median but a pair straddles zero; re-run alone"
    } else {
        "NO GAIN — record the negative, never promote"
    };
    println!("\nF2 PRICE CELL VERDICT (pp{}): {verdict}", chunk * frames);
}
