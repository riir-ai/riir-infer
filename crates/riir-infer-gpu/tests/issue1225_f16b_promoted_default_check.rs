//! Issue 1225 T4 — the executed promotion check (the delegated-owner
//! verdict's condition 2, 2026-11-16): the promoted default really takes
//! the twin path, and the `=0` demote really restores the incumbent bytes.
//!
//! Two invocations of THIS test (the env is read once per process):
//!
//! ```sh
//! D=.benchmarks/1226_i1225_t4_flip
//! # armed (unset = ON at the dist stage posture): twin engaged, artifact written
//! I1225_CHECK_OUT=$D/digest.txt cargo test -p riir-infer-gpu --release \
//!   --features prefill_f16b,production,cubecl_runtime,multi_host,riir-infer-core/bonsai2_hadamard \
//!   --test issue1225_f16b_promoted_default_check -- --ignored --nocapture
//! # demote (=0): twin never dispatches, incumbent bytes match the artifact
//! I1225_CHECK_OUT=$D/digest.txt RIIR_PREFILL_F16B=0 cargo test ... (same)
//! ```
//!
//! Each process runs TWO passes: the incumbent (route pinned off — the A
//! anchor) and the env-shaped route (the stage line's own expression,
//! executed). The artifact carries the incumbent digest; the demote
//! process compares its route-off digest against it byte-for-byte. The
//! armed process additionally asserts the twin's dispatch counter > 0
//! (default really engaged) and its route digest differs from the
//! incumbent's (the lever is numerically live).

#![cfg(all(
    feature = "prefill_f16b",
    feature = "metal_tensor_gemm",
    feature = "cubecl_runtime",
    feature = "ternary_gemv",
    feature = "multi_host",
    feature = "ternary_gemm_batched",
    target_os = "macos"
))]

use std::path::PathBuf;

use riir_infer_core::gguf_loader::{GgufFile, load_qwen_deltanet_ternary_weights_gguf};
use riir_infer_core::multi_host::LayerRange;
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_gpu::TernaryDeltanetGpuForward;
use riir_infer_gpu::cubecl_runtime::CubeCLContext;
use riir_infer_gpu::gemm_ternary_metal_tensor::set_metal_tensor_relaxed_precision;
use riir_infer_gpu::prefill_f16b::{set_f16b_placebo, set_f16b_route, set_f16b_tap};
use riir_infer_gpu::{set_prefill_attn_metal_tensor, set_prefill_use_metal_tensor_zerocopy};

const FOLDED_MODEL: &str = "../../../riir-train/data/Ternary-Bonsai-2.5-27B-PQ2_0.gguf";

/// The same seed text + pass shape as the frame A/B (the incumbent bytes
/// are the A anchor's bytes for this corpus/posture).
const SEED_TEXT: &str = "The lighthouse keeper had kept the same routine for eleven years. Each \
    evening, an hour before sunset, he climbed the one hundred and twelve steps \
    to the lamp room, wiped the salt from the great lens, and checked the oil. \
    The island was small enough to walk around in twenty minutes, and in winter \
    the supply boat came only twice a month, so he had learned to read the \
    weather the way other people read the newspaper. A grey band low on the \
    horizon meant rain by morning. A sudden stillness in the gulls meant wind.\n\n";

fn logits_digest(
    fwd: &mut TernaryDeltanetGpuForward,
    toks: &[usize],
    chunk: usize,
    n_layer: usize,
) -> String {
    let prompt_len = toks.len();
    fwd.reset_state();
    let mut last = Vec::new();
    for (fi, frame) in toks.chunks(chunk).enumerate() {
        let tok_base = fi * chunk;
        let is_final = tok_base + frame.len() == prompt_len;
        let _ = fwd.prefill_range_chunk(
            frame,
            None,
            tok_base,
            prompt_len,
            LayerRange::new(0, n_layer as u32),
            is_final,
        );
        last = fwd.read_logits();
    }
    let bits = last
        .iter()
        .flat_map(|f| f.to_bits().to_le_bytes())
        .collect::<Vec<u8>>();
    let h = bits.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
        (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3)
    });
    format!("fnv {h:016x}")
}

#[test]
#[ignore = "requires the Metal GPU + the Bonsai-2 PQ2_0 model; run explicitly with --release, twice (armed + RIIR_PREFILL_F16B=0)"]
fn f16b_promoted_default_check() {
    let check_out = std::env::var("I1225_CHECK_OUT").expect(
        "I1225_CHECK_OUT=<path> names the digest artifact (caller-provided — \
         never a fixed temp path)",
    );
    let model = std::env::var_os("BONSAI2_G1_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(FOLDED_MODEL));
    assert!(model.exists(), "model not found at {}", model.display());

    // THE STAGE LINE ITSELF, executed (stage_gpu's promoted expression —
    // if the flip is wrong, this reads the wrong default).
    let stage_line = std::env::var("RIIR_PREFILL_F16B").as_deref() != Ok("0");

    set_prefill_use_metal_tensor_zerocopy(true);
    set_prefill_attn_metal_tensor(true);
    set_metal_tensor_relaxed_precision(true);
    set_f16b_tap(false);
    set_f16b_placebo(false);
    // THE ORDERING LAW: the twin is a construction-time posture — the
    // env-shaped route pins BEFORE `new` (stage_gpu's arm line runs before
    // the forwarder constructs, by the same law).
    set_f16b_route(stage_line);

    let tok = {
        let gguf = GgufFile::open(&model).expect("open gguf for tokenizer");
        BpeTokenizer::from_gguf(&gguf).expect("bpe tokenizer from gguf")
    };
    let (mut config, weights) =
        load_qwen_deltanet_ternary_weights_gguf(&model).expect("folded model load");
    let n_layer = config.n_layer as usize;
    let chunk = 512usize;
    let frames = 2usize; // a bounded pass — the check is correctness, not perf
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

    // Pass 1: the incumbent anchor (route pinned OFF — the A arm; the twin
    // exists but the route never consults it).
    set_f16b_route(false);
    let (inc_before, _) = fwd.f16b_route_dispatch_counts();
    let incumbent = logits_digest(&mut fwd, &toks, chunk, n_layer);
    let (inc_after, _) = fwd.f16b_route_dispatch_counts();

    // Pass 2: the env-shaped route (the stage line's answer).
    set_f16b_route(stage_line);
    let (_, twin0) = fwd.f16b_route_dispatch_counts();
    let routed = logits_digest(&mut fwd, &toks, chunk, n_layer);
    let (_, twin1) = fwd.f16b_route_dispatch_counts();
    let twin_dispatches = twin1 - twin0;
    let _ = inc_before;

    println!("# stage line: RIIR_PREFILL_F16B unset/other=ON, =0=OFF → route armed = {stage_line}");
    println!(
        "# incumbent (A anchor): {incumbent} · incumbent dispatches {} (pin-off pass)",
        inc_after - inc_before
    );
    println!("# env-shaped route: {routed} · twin dispatches {twin_dispatches}");

    if stage_line {
        // The ARMED process: the twin really took the route, and the lever
        // is numerically live (the rounded input changes the bytes).
        assert!(
            twin_dispatches > 0,
            "the promoted default did NOT engage the twin (0 dispatches) — the \
             flip is broken or the route fell through"
        );
        assert!(
            routed != incumbent,
            "the twin route reproduced the incumbent's logits bytes — the twin \
             never engaged numerically"
        );
        // Write the incumbent anchor for the demote process to match.
        // NOTE: the `-p` test binary CWDs at the package root, so a relative
        // I1225_CHECK_OUT resolves there — create the parent first.
        if let Some(parent) = std::path::Path::new(&check_out).parent() {
            std::fs::create_dir_all(parent).expect("create the artifact's parent dir");
        }
        std::fs::write(&check_out, format!("{incumbent}\n")).expect("write the anchor artifact");
        println!("# anchor artifact written: {check_out}");
    } else {
        // The DEMOTE process: the twin never dispatched, and the route-off
        // bytes equal the anchor artifact (the demote restores the
        // pre-promotion incumbent byte-identically).
        assert_eq!(
            twin_dispatches, 0,
            "RIIR_PREFILL_F16B=0 did NOT demote — the twin dispatched {twin_dispatches}×"
        );
        let anchor = std::fs::read_to_string(&check_out)
            .expect("the anchor artifact (run the armed invocation first)");
        let anchor = anchor.trim();
        assert_eq!(
            routed, anchor,
            "the demoted route's logits {routed} != the pre-promotion incumbent anchor \
             {anchor} — the demote does not restore the incumbent bytes"
        );
        println!(
            "# DEMOTE byte-identity: {routed} == anchor ✓ the =0 literal restores the incumbent"
        );
    }
}
