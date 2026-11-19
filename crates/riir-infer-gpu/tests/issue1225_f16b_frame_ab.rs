//! Issue 1225 T3 — the paired frame-census A/B on the f16-B serving twin.
//!
//! The Bench-937 frame harness posture (the rung-C decider machinery): the
//! serving default — tensor zerocopy + attn-tensor + relaxed matmul2d (the
//! promoted dist posture), frame chunk 512 × 8 frames = pp4096 passes (the
//! 937 Δ-comparability shape). Arm A (route OFF) is the incumbent f32-B
//! serving path; arm B (route ON) is the f16-B twin. ABBA-interleaved pairs
//! (the order alternates per pair so linear drift cancels in the per-pair
//! Δ), a same-arm consecutive-pair read for the order-bias probe, and
//! bit-identity anchors per arm (each arm is deterministic; its passes must
//! agree byte-for-byte — a wobbly anchor voids the cell).
//!
//! The T1 decider ([`.benchmarks/1224_i1225_f16b_retention_walk.md`]) is
//! this gate's quality half and has PASSED; this cell is the ≥1.5% e2e NET
//! half. NET is direct here: the twin's per-launch cast fallback is INSIDE
//! arm B's frames — win and tax are one number.
//!
//! ```sh
//! CARGO_TARGET_DIR=/tmp/i1225f16b cargo test -p riir-infer-gpu --release \
//!   --features prefill_f16b,metal_tensor_gemm,cubecl_runtime,ternary_gemv,multi_host,ternary_gemm_batched,ternary_gemm_simdgroup,ternary_deltanet_chunked_prefill,ternary_attention_batched_prefill,riir-infer-core/bonsai2_hadamard \
//!   --test issue1225_f16b_frame_ab -- --ignored --nocapture --test-threads=1
//! # I1225AB_PAIRS (default 5) · I1225AB_CHUNK (512) · I1225AB_FRAMES (8)
//! # I1225_POSTURE=strict re-runs the reproduction arm (default relaxed = serving)
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
use riir_infer_gpu::prefill_f16b::{f16b_route_armed, set_f16b_placebo, set_f16b_route};
use riir_infer_gpu::{set_prefill_attn_metal_tensor, set_prefill_use_metal_tensor_zerocopy};

/// The e2e NET bar (the issue's T3 gate; the frame reference is Bench 937's
/// 500.86 ms — printed beside the verdict).
const NET_BAR_FRAC: f64 = 0.015;

#[test]
#[ignore = "requires the Metal GPU + the Bonsai-2 PQ2_0 model; run explicitly with --release"]
fn f16b_frame_ab() {
    let model = std::env::var_os("BONSAI2_G1_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(FOLDED_MODEL));
    assert!(
        model.exists(),
        "the Bonsai-2 league pin is the cell's premise — not found at {}",
        model.display()
    );
    let posture = std::env::var("I1225_POSTURE").unwrap_or_else(|_| "relaxed".to_string());
    assert!(
        matches!(posture.as_str(), "relaxed" | "strict"),
        "I1225_POSTURE must be relaxed|strict, got {posture}"
    );
    let strict = posture == "strict";
    let pairs: usize = std::env::var("I1225AB_PAIRS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    assert!(pairs >= 3, "need at least 3 pairs for a median");
    let chunk: usize = std::env::var("I1225AB_CHUNK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    let frames: usize = std::env::var("I1225AB_FRAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);

    // The serving posture (the promoted arms) — identical for both arms;
    // the ONLY delta is the twin route.
    set_prefill_use_metal_tensor_zerocopy(true);
    set_prefill_attn_metal_tensor(true);
    set_metal_tensor_relaxed_precision(!strict);
    set_f16b_placebo(false);
    // Route pinned ON at construction (the twin is a construction-time
    // posture); the per-pass flag flips decide who serves.
    set_f16b_route(true);

    let tok = {
        let gguf = GgufFile::open(&model).expect("open gguf for tokenizer");
        BpeTokenizer::from_gguf(&gguf).expect("bpe tokenizer from gguf")
    };
    let (mut config, weights) =
        load_qwen_deltanet_ternary_weights_gguf(&model).expect("folded model load");
    let n_layer = config.n_layer as usize;
    let toks: Vec<usize> = tok
        .encode(SEED_TEXT)
        .iter()
        .cycle()
        .take(chunk * frames + 2)
        .copied()
        .collect();
    config.block_size = toks.len() + 8;

    let ctx = CubeCLContext::new().expect("GPU init");
    let mut fwd = TernaryDeltanetGpuForward::new(&ctx, &config, &weights);
    println!(
        "# issue1225 f16-B frame A/B · posture {posture} · {pairs} ABBA pairs · \
         chunk {chunk} × frames {frames} = {} tok/pass · NET bar {:.1}%",
        toks.len(),
        NET_BAR_FRAC * 100.0
    );

    // 2 untimed compile passes per arm (kernel JIT + cache warm — the
    // census's WARMUP_PASSES lesson).
    for &route in &[false, true] {
        set_f16b_route(route);
        fwd.reset_state();
        let _ = run_pass(&mut fwd, &toks, chunk, n_layer);
    }

    let mut a_walls: Vec<f64> = Vec::new();
    let mut b_walls: Vec<f64> = Vec::new();
    let mut a_digests: Vec<String> = Vec::new();
    let mut b_digests: Vec<String> = Vec::new();
    let mut a_first_walls: Vec<f64> = Vec::new();
    let mut a_second_walls: Vec<f64> = Vec::new();
    let mut deltas: Vec<f64> = Vec::with_capacity(pairs);

    for i in 0..pairs {
        // ABBA with the pair order alternating: even pairs A B B A, odd
        // pairs B A A B — linear drift cancels in the per-pair Δ either way.
        let order = if i % 2 == 0 {
            [false, true, true, false]
        } else {
            [true, false, false, true]
        };
        let mut walls = [0.0f64; 4];
        for (slot, &route) in order.iter().enumerate() {
            set_f16b_route(route);
            fwd.reset_state();
            let (wall, digest) = run_pass(&mut fwd, &toks, chunk, n_layer);
            walls[slot] = wall;
            if route {
                b_digests.push(digest);
            } else {
                a_digests.push(digest);
            }
        }
        // The pair's A mean and B mean: slots where order == false are A.
        let a_in_pair: Vec<f64> = order
            .iter()
            .zip(walls.iter())
            .filter(|(r, _)| !**r)
            .map(|(_, w)| *w)
            .collect();
        let b_in_pair: Vec<f64> = order
            .iter()
            .zip(walls.iter())
            .filter(|(r, _)| **r)
            .map(|(_, w)| *w)
            .collect();
        let a_mean = a_in_pair.iter().sum::<f64>() / a_in_pair.len() as f64;
        let b_mean = b_in_pair.iter().sum::<f64>() / b_in_pair.len() as f64;
        // The order-bias probe: the FIRST A wall of the pair vs the second.
        let a_seq: Vec<f64> = order
            .iter()
            .enumerate()
            .filter(|(_, r)| !**r)
            .map(|(slot, _)| walls[slot])
            .collect();
        if a_seq.len() == 2 {
            a_first_walls.push(a_seq[0]);
            a_second_walls.push(a_seq[1]);
        }
        let delta = a_mean - b_mean;
        deltas.push(delta);
        a_walls.extend_from_slice(&a_in_pair);
        b_walls.extend_from_slice(&b_in_pair);
        println!(
            "  pair {i}: A mean {a_mean:.2} ms · B mean {b_mean:.2} ms · Δ {delta:.2} ms ({:.2}%)",
            delta / a_mean * 100.0
        );
    }
    set_f16b_route(f16b_route_armed()); // restore the armed posture

    // Bit anchors: every A pass byte-identical, every B pass byte-identical,
    // and A != B (the lever is numerically live — a twin that reproduced the
    // incumbent bits would be a route bug, not a pass).
    assert!(
        a_digests.windows(2).all(|w| w[0] == w[1]),
        "arm A (incumbent) bit anchor BROKEN — the passes disagree: {:?}",
        a_digests
    );
    assert!(
        b_digests.windows(2).all(|w| w[0] == w[1]),
        "arm B (twin) bit anchor BROKEN — the passes disagree: {:?}",
        b_digests
    );
    assert!(
        a_digests[0] != b_digests[0],
        "arm B reproduced arm A's logits bits — the twin never engaged (route bug, not a pass)"
    );
    println!(
        "# bit anchors: A {} × {}, B {} × {} — HELD (A≠B as expected)",
        "identical",
        a_digests.len(),
        "identical",
        b_digests.len()
    );

    // Order-bias probe: A-first vs A-second mean walls (a load-drift read).
    let a_first = a_first_walls.iter().sum::<f64>() / a_first_walls.len() as f64;
    let a_second = a_second_walls.iter().sum::<f64>() / a_second_walls.len() as f64;
    println!(
        "# order-bias probe: A-first mean {a_first:.2} ms vs A-second mean {a_second:.2} ms \
         (drift {:+.2}%, |drift| must be ≪ the effect)",
        (a_second - a_first) / a_first * 100.0
    );

    let a_med = median(a_walls.clone());
    let b_med = median(b_walls.clone());
    let d_lo = deltas
        .iter()
        .cloned()
        .min_by(|a, b| a.total_cmp(b))
        .expect("pairs");
    let d_hi = deltas
        .iter()
        .cloned()
        .max_by(|a, b| a.total_cmp(b))
        .expect("pairs");
    let d_med = median(deltas.clone());
    let share = d_med / a_med * 100.0;
    println!(
        "\nA median {a_med:.2} ms · B median {b_med:.2} ms · Δ median {d_med:.2} ms \
         (range [{d_lo:.2}, {d_hi:.2}] ms) = {share:.2}% of the frame \
         (bar ≥ {:.1}% = {:.2} ms of the {:.2} ms frame reference)",
        NET_BAR_FRAC * 100.0,
        500.86 * NET_BAR_FRAC,
        500.86
    );
    // Sign consistency (the rung-C evidence standard): every pair's Δ
    // positive — no pair straddles zero.
    let sign_consistent = deltas.iter().all(|&d| d > 0.0);
    if d_med >= 500.86 * NET_BAR_FRAC && sign_consistent {
        println!(
            "\nFRAME A/B VERDICT {posture}: OVER — sign-consistent NET evidence on all \
             {pairs} pairs; with T1 PASS (Bench 1224, no family demotes) the GOAT gate's \
             G1+G2 halves hold — T4 (G3 decode-by-construction + G4 alloc + promotion) is cleared to adjudicate"
        );
    } else if d_med >= 500.86 * NET_BAR_FRAC {
        panic!(
            "Δ median {d_med:.2} ms clears the bar but the pairs are NOT sign-consistent \
             (a pair straddles zero) — re-run alone before adjudicating"
        );
    } else {
        panic!(
            "Δ median {d_med:.2} ms < the {:.2} ms bar — the twin's e2e NET is UNDER at \
             {posture}; record the negative, never promote",
            500.86 * NET_BAR_FRAC
        );
    }
}
