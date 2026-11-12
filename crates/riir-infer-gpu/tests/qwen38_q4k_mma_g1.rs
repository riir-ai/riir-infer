//! Plan 614 Phase-3 session-2 G1 — the simdgroup-mma Q4_K kernel unit gate.
//!
//! One kernel, one gate: `gemm_q4k_mma` (the dequant → threadgroup f32
//! tile → simdgroup-mma skeleton, ONE barrier per K-tile with A+B
//! double-buffered) vs a CPU reference over the SAME f16-ROUNDED dequanted
//! weights — the tile values are `d * sc * q - dmin * mn` in f32 rounded to
//! half once (the `dequantize_row_q4_k` expression order), so the band
//! isolates mma summation-order drift exactly as the staged tensor G1
//! (`qwen38_q4k_tensor_g1.rs`) does. The 7 shape classes, the dequant lens,
//! and the tolerance law are REUSED VERBATIM from that rig.
//!
//! Shape laws: `n` a multiple of 256 (the Q4_K block); `m`, `p` ≥ 1
//! (guarded tiles — the mma store path handles tile-ragged m and p).
//!
//! Tolerance: relative to the per-element accumulation MAGNITUDE `Σ|w·x|`
//! (2e-4, cancellation-free).

#![cfg(all(
    feature = "qwen38_wgpu",
    feature = "metal_tensor_gemm",
    target_os = "macos"
))]

use cubecl::client::ComputeClient;
use riir_infer_gpu::cubecl_runtime::{ActiveRuntime, CubeCLContext};
use riir_infer_gpu::gemm_dense_f16_metal_zero_copy::MetalDenseF16ZeroCopyGemm;
use riir_infer_gpu::gemv_q4k_cubecl::Q4KHandle;

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(14_426_950_408_889_407);
    ((*seed >> 33) as f32 / (u64::MAX >> 33) as f32) * 2.0 - 1.0
}

/// Dequantize one row's Q4_K blocks to f32 (the reference order), then round
/// each weight to f16 bits — the CPU reference's weights.
fn dequant_row_f16_bits(blocks: &[riir_infer_core::quant::q4k::BlockQ4K]) -> Vec<u16> {
    use riir_infer_core::quant::q4k::dequantize_row_q4_k;
    let n = blocks.len() * 256;
    let mut w = vec![0.0f32; n];
    dequantize_row_q4_k(blocks, &mut w);
    w.iter()
        .map(|&v| half::f16::from_f32(v).to_bits())
        .collect()
}

/// CPU GEMM over the f16-rounded weights (f32 accumulate) + the magnitudes.
fn cpu_gemm_f16(w16: &[u16], x: &[f32], m: usize, n: usize, p: usize) -> (Vec<f32>, Vec<f32>) {
    let (mut out, mut mag) = (vec![0.0f32; p * m], vec![0.0f32; p * m]);
    for t in 0..p {
        for r in 0..m {
            let (mut dot, mut acc) = (0.0f32, 0.0f32);
            for c in 0..n {
                let term = f32::from(half::f16::from_bits(w16[r * n + c])) * x[t * n + c];
                dot += term;
                acc += term.abs();
            }
            out[t * m + r] = dot;
            mag[t * m + r] = acc;
        }
    }
    (out, mag)
}

fn run_case(
    mm: &MetalDenseF16ZeroCopyGemm,
    client: &ComputeClient<ActiveRuntime>,
    m: usize,
    n: usize,
    p: usize,
) {
    use bytemuck::Zeroable;
    use riir_infer_core::quant::q4k::{BlockQ4K, QK_K, quantize_row_q4_k};

    let mut seed: u64 =
        0x6144_d100_0000_0000 ^ (m as u64) ^ ((n as u64) << 21) ^ ((p as u64) << 42);
    let wf: Vec<f32> = (0..m * n).map(|_| lcg(&mut seed) * 0.05).collect();
    let x: Vec<f32> = (0..p * n).map(|_| lcg(&mut seed)).collect();

    let bpr = n / QK_K;
    let mut blocks = vec![BlockQ4K::zeroed(); m * bpr];
    let mut row = vec![0.0f32; n];
    for r in 0..m {
        row.copy_from_slice(&wf[r * n..(r + 1) * n]);
        quantize_row_q4_k(&row, &mut blocks[r * bpr..(r + 1) * bpr]);
    }
    let w16 = dequant_row_f16_bits(&blocks);
    let (expect, mag) = cpu_gemm_f16(&w16, &x, m, n, p);

    let q4 = Q4KHandle::from_blocks(client, &blocks, m, n);
    let x_h = client.create_from_slice(bytemuck::cast_slice(&x));
    let out_h = client.create_from_slice(&vec![0u8; p * m * 4]);
    // Host uploads are DEFERRED writes (the issue-1146 attention-gate
    // hazard): settle the fixture before the raw commit.
    pollster::block_on(client.sync()).expect("sync after uploads");

    mm.dispatch_mma_q4k(client, &q4.weight_q4k, &q4.d_dmin, m, n, &x_h, &out_h, p)
        .unwrap_or_else(|e| panic!("mma dispatch {m}x{n} p{p}: {e}"));
    mm.wait_last();
    let got_bytes = client.read_one(out_h).expect("read");
    let got: &[f32] = bytemuck::cast_slice(&got_bytes);

    let mut max_rel = 0.0f64;
    for (i, (&g, (&e, &mg))) in got.iter().zip(expect.iter().zip(mag.iter())).enumerate() {
        let rel = ((g - e).abs() / mg.max(1e-9)) as f64;
        if rel > max_rel {
            max_rel = rel;
        }
        assert!(
            rel < 2e-4,
            "mma case {m}x{n} p{p}: element {i} got {g} want {e} (rel-to-mag {rel:.3e})"
        );
    }
    println!("mma case {m}x{n} p{p}: max_rel-to-mag {max_rel:.3e}");
}

/// The dequant lens: dispatch with the IDENTITY batch (x[t] = one-hot at t,
/// p = n) — y[t][r] is then exactly the dequanted weight W[r][t]
/// (f16-rounded), so any dequant/indexing/fragment-layout defect is visible
/// ELEMENT-WISE, not just as a dot-product residual. Run when the G1 dot
/// cases red and the class is not obvious.
#[test]
#[ignore]
fn q4k_mma_dequant_lens() {
    use bytemuck::Zeroable;
    use riir_infer_core::quant::q4k::{BlockQ4K, QK_K, dequantize_row_q4_k, quantize_row_q4_k};

    let (m, n) = (64usize, 512usize);
    let p = n; // identity batch
    let mut seed: u64 = 0x0614_deca_1000_0000;
    let wf: Vec<f32> = (0..m * n).map(|_| lcg(&mut seed) * 0.05).collect();

    let bpr = n / QK_K;
    let mut blocks = vec![BlockQ4K::zeroed(); m * bpr];
    let mut row = vec![0.0f32; n];
    for r in 0..m {
        row.copy_from_slice(&wf[r * n..(r + 1) * n]);
        quantize_row_q4_k(&row, &mut blocks[r * bpr..(r + 1) * bpr]);
    }
    let mut w_ref = vec![0.0f32; m * n];
    for r in 0..m {
        dequantize_row_q4_k(
            &blocks[r * bpr..(r + 1) * bpr],
            &mut w_ref[r * n..(r + 1) * n],
        );
    }

    let ctx = CubeCLContext::new().expect("gpu context");
    let client = ctx.client();
    let mm = MetalDenseF16ZeroCopyGemm::new(
        ctx.wgpu_device().expect("wgpu device").clone(),
        ctx.wgpu_queue().expect("wgpu queue").clone(),
    )
    .expect("dense zero-copy context");

    let q4 = Q4KHandle::from_blocks(&client, &blocks, m, n);
    let x: Vec<f32> = (0..p * n)
        .map(|i| if i / n == i % n { 1.0 } else { 0.0 })
        .collect();
    let x_h = client.create_from_slice(bytemuck::cast_slice(&x));
    let out_h = client.create_from_slice(&vec![0u8; p * m * 4]);
    pollster::block_on(client.sync()).expect("sync");

    mm.dispatch_mma_q4k(&client, &q4.weight_q4k, &q4.d_dmin, m, n, &x_h, &out_h, p)
        .expect("lens dispatch");
    mm.wait_last();
    let got_bytes = client.read_one(out_h).expect("read");
    let got: &[f32] = bytemuck::cast_slice(&got_bytes);

    // y[t * m + r] ≈ f16(dequant(W[r][t])).
    let mut bad = 0usize;
    let mut first = None;
    for r in 0..m {
        for t in 0..n {
            let got_w = got[t * m + r];
            let want = f32::from(half::f16::from_f32(w_ref[r * n + t]));
            if (got_w - want).abs() > (1e-4f32).max(want.abs() * 1e-3) {
                bad += 1;
                if first.is_none() {
                    first = Some((r, t, got_w, want));
                }
            }
        }
    }
    println!(
        "q4k mma dequant lens {m}x{n}: {bad} mismatched elements of {total}",
        total = m * n
    );
    if let Some((r, t, g, w)) = first {
        println!(
            "  first mismatch: r={r} t={t} (block {}) got {g:.6} want {w:.6}",
            t / 256
        );
        panic!("dequant lens red");
    }
}

#[test]
fn q4k_mma_g1() {
    let ctx = CubeCLContext::new().expect("gpu context");
    let client = ctx.client();
    let mm = MetalDenseF16ZeroCopyGemm::new(
        ctx.wgpu_device().expect("wgpu device").clone(),
        ctx.wgpu_queue().expect("wgpu queue").clone(),
    )
    .expect("dense zero-copy context");

    // The staged tensor G1's 7 shape classes, VERBATIM: production column
    // count n=5120 (20 Q4_K blocks/row) at the p classes 1 (decode),
    // 9 (the plan-614 verify shape), 32/64 (the prefill class); small-n
    // edges; the m-ragged-alone cell.
    run_case(&mm, &client, 64, 5120, 1);
    run_case(&mm, &client, 128, 5120, 9);
    run_case(&mm, &client, 512, 5120, 32);
    run_case(&mm, &client, 64, 5120, 64);
    run_case(&mm, &client, 128, 256, 9);
    run_case(&mm, &client, 64, 512, 9);
    run_case(&mm, &client, 48, 256, 64);
}

/// Timing probe — the mma kernel beside the staged tensor kernel at the
/// production ffn shape [17408×5120] (the session-1 design's probe-cell
/// law: call it beside the old one so Bench 1162/1164 rows stay
/// comparable). TIMING-SENSITIVE: run on an idle box only (the amo
/// correctness runs above are timing-insensitive; this cell is not).
/// The model-level C_K measurement (the verdict's park bar) is the
/// REAL verify chunk on BMO, not this per-kernel microbench.
///
/// Run:
/// ```text
/// cargo test --release -p riir-infer-gpu --features qwen38_wgpu,metal_tensor_gemm \
///   --test qwen38_q4k_mma_g1 q4k_mma_rate_probe -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn q4k_mma_rate_probe() {
    use bytemuck::Zeroable;
    use riir_infer_core::quant::q4k::{BlockQ4K, QK_K, quantize_row_q4_k};

    let (m, n) = (17_408usize, 5_120usize);
    let mut seed: u64 = 0x0614_d5f8_0000_0000;
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
    pollster::block_on(client.sync()).expect("sync after uploads");

    let q4k_wbytes = (m * n * 152 / 256) as f64;
    println!(
        "mma rate probe @{m}x{n}: q4k weights {:.1} MB",
        q4k_wbytes / 1e6
    );
    println!(
        "{:>4} | {:>10} {:>10} {:>9} {:>8}",
        "p", "mma-ms", "q4k-mm ms", "TF/s", "GB/s"
    );

    const N: usize = 20;
    for p in [9usize, 32, 64, 128, 512] {
        let x: Vec<f32> = (0..p * n).map(|_| lcg(&mut seed)).collect();
        let x_h = client.create_from_slice(bytemuck::cast_slice(&x));
        let out_h = client.create_from_slice(&vec![0u8; p * m * 4]);

        // ABBA ×2 vs the staged tensor kernel — the interleaved paired
        // protocol (the Bench-957 preflight law: single-arm sequential
        // reads ride load-climb contamination).
        let dispatch = |arm: u8| -> Result<(), String> {
            match arm {
                0 => mm.dispatch_q4k(&client, &q4.weight_q4k, &q4.d_dmin, m, n, &x_h, &out_h, p),
                _ => {
                    mm.dispatch_mma_q4k(&client, &q4.weight_q4k, &q4.d_dmin, m, n, &x_h, &out_h, p)
                }
            }
        };
        for arm in [0u8, 1] {
            for _ in 0..3 {
                dispatch(arm).expect("warm");
            }
        }
        mm.wait_last();
        pollster::block_on(client.sync()).expect("warm sync");

        // Loud-zero insurance: one readback vs the CPU dot at row 0.
        {
            let w16 = dequant_row_f16_bits(&blocks[..bpr]);
            let mut want = 0.0f32;
            let mut acc = 0.0f32;
            for c in 0..n {
                let term = f32::from(half::f16::from_bits(w16[c])) * x[c];
                want += term;
                acc += term.abs();
            }
            mm.dispatch_mma_q4k(&client, &q4.weight_q4k, &q4.d_dmin, m, n, &x_h, &out_h, p)
                .expect("verify dispatch");
            mm.wait_last();
            let got_bytes = client.read_one(out_h.clone()).expect("read probe");
            let got: &[f32] = bytemuck::cast_slice(&got_bytes);
            let rel = ((got[0] - want).abs() / acc.max(1e-9)) as f64;
            assert!(
                rel < 2e-4,
                "mma probe p{p} loud-zero: row0 got {} want {want} (rel-to-mag {rel:.3e})",
                got[0]
            );
        }

        let mut t_mma = Vec::new();
        let mut t_q4k = Vec::new();
        for _ in 0..2 {
            for (arm, ts) in [(0u8, &mut t_q4k), (1u8, &mut t_mma)] {
                pollster::block_on(client.sync()).expect("pre");
                let t0 = std::time::Instant::now();
                for _ in 0..N {
                    dispatch(arm).expect("timed");
                }
                mm.wait_last();
                pollster::block_on(client.sync()).expect("post");
                ts.push(t0.elapsed().as_secs_f64() * 1000.0 / N as f64);
            }
        }
        t_mma.sort_by(|a, b| a.partial_cmp(b).unwrap());
        t_q4k.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let t = t_mma[t_mma.len() / 2];
        let t_old = t_q4k[t_q4k.len() / 2];
        let gflop = 2.0 * m as f64 * n as f64 * p as f64 / 1e9;
        println!(
            "{p:>4} | {t:>10.3} {t_old:>10.3} {:>9.2} {:>8.0}",
            gflop / (t / 1000.0),
            q4k_wbytes / (t / 1000.0) / 1e9,
        );
    }
}
