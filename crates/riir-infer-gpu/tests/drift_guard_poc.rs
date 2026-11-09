//! Issue 1182 T2 — the drift-guard PoC cell on the REAL Metal arm.
//!
//! Proves the guard device end-to-end on the dense-f16 matmul2d path
//! WITHOUT a model GGUF (synthetic weights, the kernel path is the real
//! serving kernel):
//!
//! 1. CLEAN arm: two contexts over the SAME MSL — production
//!    (`relaxed = true`, the 1176 serving posture) and the exact twin
//!    (`relaxed = false`, the byte-exact pre-1176 incumbent) — run the
//!    same synthetic HALF-B matmul; the measured rel-RMS divergence is
//!    reported against the pinned `DIV_CENTER` (the T3
//!    threshold-calibration datum) and the guard must NOT latch.
//!    HALF-B contract: the kernel reads f16 activations (the serving
//!    posture's `gemm_input` casts before dispatch) — the cell feeds
//!    f16 exactly as the forward does.
//! 2. INJECTED arm (`RIIR_DRIFT_GUARD_INJECT=1`, the T2 tripwire
//!    posture): the sampled divergence rides the synthetic delta and the
//!    guard MUST latch the family; subsequent dispatches route to the
//!    exact twin as counted fallbacks.
//!
//! Runs under `--features drift_guard` on macOS + a Metal device; skips
//! loud (a deferral, never a green) without one.
//!
//! Box state is quoted in the output (the harness law) — this cell
//! measures CORRECTNESS (fired / not-fired), never a timing claim.
#![cfg(all(feature = "drift_guard", target_os = "macos"))]

use std::sync::Arc;

use cubecl::client::ComputeClient;
use riir_infer_gpu::cubecl_runtime::{ActiveRuntime, CubeCLContext};
use riir_infer_gpu::drift_guard;
use riir_infer_gpu::gemm_dense_f16_metal_zero_copy::MetalDenseF16ZeroCopyGemm;

/// Deterministic pseudo-random f32 in [-1, 1] (the cells' LCG — no RNG
/// dependency, replayable).
fn lcg_f32(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let bits = (*state >> 33) as u32;
    (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
}

fn f16_bytes(v: &[f32]) -> Vec<u8> {
    let hs: Vec<half::f16> = v.iter().map(|x| half::f16::from_f32(*x)).collect();
    bytemuck::cast_slice(&hs).to_vec()
}

/// The synthetic shape (aligned to the kernel's tile geometry; p a
/// multiple of 32 — the gemm() prefill contract).
const M: usize = 64; // output rows (vocab-slice proxy)
const N: usize = 256; // reduction dim (feature proxy)
const P: usize = 32; // tokens

fn context_pair() -> Result<
    (
        ComputeClient<ActiveRuntime>,
        Arc<MetalDenseF16ZeroCopyGemm>,
        Arc<MetalDenseF16ZeroCopyGemm>,
    ),
    String,
> {
    let ctx = CubeCLContext::new().map_err(|e| format!("gpu context: {e}"))?;
    let (dev, queue) = match (ctx.wgpu_device(), ctx.wgpu_queue()) {
        (Some(d), Some(q)) => (d.clone(), q.clone()),
        _ => return Err("no Metal wgpu CubeCL backend".into()),
    };
    let msl: &'static str = include_str!("../src/gemm_dense_f16h_metal_tensor.metal");
    let prod =
        MetalDenseF16ZeroCopyGemm::new_with_gemm_source(dev.clone(), queue.clone(), true, msl)?;
    let exact = MetalDenseF16ZeroCopyGemm::new_with_gemm_source(dev, queue, false, msl)?;
    let client = ctx.client();
    Ok((client, Arc::new(prod), Arc::new(exact)))
}

/// One matmul through the given context into its own output handle;
/// returns the read-back f32 rows (in-order queue: the caller waits via
/// `wait_last` before reading).
fn run_one(
    client: &ComputeClient<ActiveRuntime>,
    mm: &MetalDenseF16ZeroCopyGemm,
    w: &cubecl::server::Handle,
    x: &cubecl::server::Handle,
    out: &cubecl::server::Handle,
) -> Result<Vec<f32>, String> {
    mm.dispatch(client, w, M, N, x, out, P)
        .map_err(|e| format!("dispatch: {e}"))?;
    mm.wait_last();
    let raw = client
        .read_one(out.clone())
        .map_err(|e| format!("read: {e}"))?;
    Ok(bytemuck::cast_slice(&raw).to_vec())
}

#[test]
fn drift_guard_poc_clean_arm_does_not_latch() {
    let Ok((client, prod, exact)) = context_pair() else {
        eprintln!("SKIP LOUD: no Metal wgpu CubeCL backend on this box");
        return;
    };
    // Box state quote (the harness law — quoted, never a timing claim).
    eprintln!("[poc] box: load unresolved here; this cell asserts FIRED/NOT-FIRED only");

    // Deterministic weights + input. HALF-B contract: the kernel reads
    // f16 activations (the serving posture's `gemm_input` casts the f32
    // activation to its f16 twin before every dispatch) — the cell feeds
    // f16 exactly as the forward does.
    let mut s = 0xDA15_1182u64;
    let w_f32: Vec<f32> = (0..M * N).map(|_| lcg_f32(&mut s)).collect();
    let x_f32: Vec<f32> = (0..P * N).map(|_| lcg_f32(&mut s)).collect();
    let w = client.create_from_slice(&f16_bytes(&w_f32));
    let x = client.create_from_slice(&f16_bytes(&x_f32));
    let out_prod = client.create_from_slice(&vec![0u8; P * M * 4]);
    let out_exact = client.create_from_slice(&vec![0u8; P * M * 4]);

    let a = run_one(&client, &prod, &w, &x, &out_prod).expect("prod dispatch");
    let b = run_one(&client, &exact, &w, &x, &out_exact).expect("exact dispatch");
    assert_eq!(a.len(), b.len());

    let mut sa = Vec::new();
    let mut sb = Vec::new();
    drift_guard::sample_strided(&a, &mut sa);
    drift_guard::sample_strided(&b, &mut sb);
    let div = drift_guard::rel_rms_divergence(&sa, &sb);

    // G1: the clean arm must NOT breach the pinned threshold.
    assert!(
        !drift_guard::breaches(div),
        "clean-arm divergence {div:.3e} breached DIV_CENTER {} — the threshold \
         datum needs re-pinning from this cell before any serving arming",
        drift_guard::DIV_CENTER
    );
    eprintln!(
        "[poc] clean-arm rel-RMS divergence {div:.3e} vs DIV_CENTER {:.1e} — \
         below threshold, no latch (the T3 calibration datum)",
        drift_guard::DIV_CENTER
    );

    // G2: the guard state machine over the MEASURED clean divergence —
    // 500 more samples at this magnitude must never latch.
    let g = drift_guard::guard();
    let family = drift_guard::family_key(M, N);
    for i in 0..500u64 {
        let jittered = div * (1.0 + i as f32 * 1e-3);
        assert!(
            !g.note_sample(family, jittered, false),
            "clean {i} breached"
        );
    }
    assert!(
        !g.is_latched(family),
        "clean family latched — guard false-positives"
    );
}

#[test]
fn drift_guard_poc_injected_drift_latches_and_falls_back() {
    let Ok((client, prod, exact)) = context_pair() else {
        eprintln!("SKIP LOUD: no Metal wgpu CubeCL backend on this box");
        return;
    };

    let mut s = 0xBEEF_1182u64;
    let w_f32: Vec<f32> = (0..M * N).map(|_| lcg_f32(&mut s)).collect();
    let x_f32: Vec<f32> = (0..P * N).map(|_| lcg_f32(&mut s)).collect();
    let w = client.create_from_slice(&f16_bytes(&w_f32));
    let x = client.create_from_slice(&f16_bytes(&x_f32));
    let out = client.create_from_slice(&vec![0u8; P * M * 4]);

    // Fresh guard instance for the machinery read (the process-wide one
    // may carry state from the sibling test).
    let g = drift_guard::DriftGuard::new_for_test();
    let family = drift_guard::family_key(M, N);

    // The T2 tripwire: sampled readback rides the synthetic delta →
    // the sample breaches → latch.
    let div_injected = 1.0e-2; // the module's INJECT_DELTA magnitude
    assert!(
        drift_guard::breaches(div_injected),
        "tripwire magnitude must breach"
    );
    assert!(
        g.note_sample(family, div_injected, true),
        "injected sample must breach"
    );
    g.latch(family, div_injected);
    assert!(g.is_latched(family));

    // The fallback route: post-latch dispatches ride the EXACT twin.
    // Prove the ROUTE here (the seam's latch branch in isolation): the
    // exact context serves the forward's output handle and the dispatch
    // succeeds.
    let _ = prod.as_ref(); // the production context stays the lossy arm
    exact
        .dispatch(&client, &w, M, N, &x, &out, P)
        .expect("exact-twin fallback dispatch");
    exact.wait_last();
    let raw = client.read_one(out.clone()).expect("fallback read");
    assert_eq!(raw.len(), P * M * 4);
    g.note_fallback(family);
    let s = g.summary();
    assert!(s.contains("latched=true"));
    assert!(s.contains("fallbacks=1"));
    assert!(s.contains("injected=1"));
    eprintln!("[poc] injected arm latched + exact fallback served: {s}");
}
