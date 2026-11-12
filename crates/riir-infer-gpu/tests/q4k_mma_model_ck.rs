//! Plan 614 Phase-3 session-3 — the MODEL-LEVEL C_K cell (the verdict's
//! park bar, not a per-kernel microbench).
//!
//! The REAL verify chunk's GEMM-side cost: 192 back-to-back
//! `dispatch_mma_q4k` calls at the production ffn shape [17408×5120]
//! (one per layer, the Bench-1162 protocol), wall-clocked AS A CHUNK —
//! the in-order shared queue streams the dispatches without host syncs,
//! one sync at the end (the serial decode's own shape: the host never
//! round-trips between layers). The staged matmul2d kernel runs the same
//! chunk as the paired arm — the ABBA interleaved protocol (the
//! Bench-957 preflight law).
//!
//! C_K = chunk_ms / 118.6 (the recorded full-model serial token,
//! packed Q4_K posture, Bench 1162's 8.43 tok/s short-ctx row). The
//! GEMM-side-only conservatism direction is stated, not hidden: the real
//! chunk ADDs attention/GDN/drafter/rollback, so every C_K here is a
//! floor and the "PARK" verdicts only strengthen.
//!
//! PRE-REGISTERED BANDS (verdict session 59dafa39-886a-4b77-8df0-ba49aa7bd58c,
//! the gate `L_eff/(C_K+d) ≥ 2.31`, S = 2.31–2.32, d = 0.135, best
//! measured L_eff = 2.83):
//! - C_K ≤ 0.95  → PASS (needed L_eff ≤ 2.50, the session-2 bar)
//! - 0.95 < C_K ≤ 1.09 → MARGINAL (break-even; continue ONLY alongside a
//!   concurrent acceptance/drafter lever, else park)
//! - C_K > 1.09  → AUTOMATIC PARK
//!
//! TIMING-SENSITIVE: run on an IDLE BMO window only (the verdict's box
//! law). This cell is `#[ignore]`d everywhere else.
//!
//! Run:
//! ```text
//! cargo test --release -p riir-infer-gpu --features qwen38_wgpu,metal_tensor_gemm \
//!   --test q4k_mma_model_ck q4k_mma_model_ck_cell -- --ignored --nocapture
//! ```

#![cfg(all(
    feature = "qwen38_wgpu",
    feature = "metal_tensor_gemm",
    target_os = "macos"
))]

use riir_infer_gpu::cubecl_runtime::CubeCLContext;
use riir_infer_gpu::gemm_dense_f16_metal_zero_copy::MetalDenseF16ZeroCopyGemm;
use riir_infer_gpu::gemv_q4k_cubecl::Q4KHandle;

/// The recorded serial token (Bench 1162: 8.43 tok/s short-ctx, packed
/// Q4_K posture, 32 steps, GPU-quiet provenance).
const SERIAL_TOKEN_MS: f64 = 118.6;

/// The verify chunk's layer count at the production geometry.
const CHUNK_LAYERS: usize = 192;

#[test]
#[ignore]
fn q4k_mma_model_ck_cell() {
    use bytemuck::Zeroable;
    use riir_infer_core::quant::q4k::{BlockQ4K, QK_K, quantize_row_q4_k};

    let (m, n) = (17_408usize, 5_120usize);
    let p = 9usize; // the verify shape (the bar pins at p=9)
    let mut seed: u64 = 0x0614_d5f8_5e55_1000;
    let wf: Vec<f32> = (0..m * n).map(|_| lcg(&mut seed) * 0.05).collect();

    let bpr = n / QK_K;
    let mut blocks = vec![BlockQ4K::zeroed(); m * bpr];
    let mut row = vec![0.0f32; n];
    for r in 0..m {
        row.copy_from_slice(&wf[r * n..(r + 1) * n]);
        quantize_row_q4_k(&row, &mut blocks[r * bpr..(r + 1) * bpr]);
    }

    let ctx = CubeCLContext::new().expect("gpu context");
    let client = ctx.client();
    let mm = MetalDenseF16ZeroCopyGemm::new(
        ctx.wgpu_device().expect("wgpu device").clone(),
        ctx.wgpu_queue().expect("wgpu queue").clone(),
    )
    .expect("dense zero-copy context");

    let q4 = Q4KHandle::from_blocks(&client, &blocks, m, n);
    let x: Vec<f32> = (0..p * n).map(|_| lcg(&mut seed)).collect();
    let x_h = client.create_from_slice(bytemuck::cast_slice(&x));
    let out_h = client.create_from_slice(&vec![0u8; p * m * 4]);
    pollster::block_on(client.sync()).expect("uploads sync");

    let dispatch = |arm: u8| -> Result<(), String> {
        match arm {
            0 => mm.dispatch_q4k(&client, &q4.weight_q4k, &q4.d_dmin, m, n, &x_h, &out_h, p),
            _ => mm.dispatch_mma_q4k(&client, &q4.weight_q4k, &q4.d_dmin, m, n, &x_h, &out_h, p),
        }
    };

    // Warm both arms + the output cache (the dedicated-buffer first-use
    // cure) before any timing.
    for arm in [0u8, 1] {
        for _ in 0..3 {
            dispatch(arm).expect("warm");
        }
    }
    mm.wait_last();
    pollster::block_on(client.sync()).expect("warm sync");

    // The chunk: 192 dispatches streamed, ONE sync at the end. ABBA ×2
    // interleaved (staged arm = A, mma arm = B).
    let chunk = |arm: u8| -> f64 {
        pollster::block_on(client.sync()).expect("pre-chunk");
        let t0 = std::time::Instant::now();
        for layer in 0..CHUNK_LAYERS {
            // The chunk's activations change per layer in reality; the
            // timing contract only needs the same GEMM stream (the weights
            // dominate; x is per-tick, not per-layer, at the verify shape).
            let _ = layer;
            dispatch(arm).expect("chunk dispatch");
        }
        mm.wait_last();
        pollster::block_on(client.sync()).expect("post-chunk");
        t0.elapsed().as_secs_f64() * 1000.0
    };

    let mut t_mma = Vec::new();
    let mut t_staged = Vec::new();
    for _ in 0..2 {
        for (arm, ts) in [(0u8, &mut t_staged), (1u8, &mut t_mma)] {
            ts.push(chunk(arm));
        }
    }
    t_mma.sort_by(|a, a2| a.partial_cmp(a2).unwrap());
    t_staged.sort_by(|a, a2| a.partial_cmp(a2).unwrap());
    let chunk_mma = t_mma[t_mma.len() / 2];
    let chunk_staged = t_staged[t_staged.len() / 2];

    let ck = chunk_mma / SERIAL_TOKEN_MS;
    let ck_staged = chunk_staged / SERIAL_TOKEN_MS;
    println!("model-level C_K cell (the verdict's park bar):");
    println!(
        "  chunk [17408×5120] p={p} × {CHUNK_LAYERS} layers: mma {chunk_mma:.1} ms | staged {chunk_staged:.1} ms"
    );
    println!(
        "  C_K(mma) = {chunk_mma:.1} / {SERIAL_TOKEN_MS} = {ck:.3}   C_K(staged) = {ck_staged:.3}"
    );
    let band = if ck <= 0.95 {
        "PASS (C_K ≤ 0.95 — needed L_eff ≤ 2.50; promotion work continues)"
    } else if ck <= 1.09 {
        "MARGINAL (0.95 < C_K ≤ 1.09 — continue ONLY alongside a concurrent acceptance/drafter lever, else park)"
    } else {
        "AUTOMATIC PARK (C_K > 1.09)"
    };
    println!("  VERDICT: {band}");
    println!(
        "  honesty: GEMM-side floor only — attention/GDN/drafter/rollback ADD to the real chunk"
    );
}

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32) / ((1u64 << 31) as f32) - 1.0
}
