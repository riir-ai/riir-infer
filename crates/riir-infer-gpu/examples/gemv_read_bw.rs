//! Issue 1177 item 2 — the MEASURED read-bandwidth ceiling at the decode
//! GEMV access class. The league roofline figures (~810-819 GB/s) were
//! ASSUMED (Apple's published M5 Ultra number); this probe times the
//! engaged decode kernel itself (`gemv_dense_f16`, one simdgroup per row,
//! coalesced f16 reads, one simd_sum) over a bandwidth-saturating shape
//! and prints the achieved GB/s — the roofline denominator becomes a
//! measurement, never an assumption.
//!
//! Probe shape: m=32768 rows × n=32768 f16 columns = 2 GiB read per
//! dispatch (large enough to amortize launch cost, small enough to stay
//! resident beside the host). Timing is the decode-class pattern: commit +
//! `wait_last` per dispatch (each decode token waits for its GEMV), median
//! of 10 after 3 warm-ups. Commit-batched pipelining is deliberately NOT
//! measured — decode never overlaps GEMVs on the wire.
//!
//! Run: cargo run --release -p riir-infer-gpu --features metal_tensor_gemm,cubecl_runtime \
//!        --example gemv_read_bw
#![cfg(all(
    target_os = "macos",
    feature = "metal_tensor_gemm",
    feature = "cubecl_runtime"
))]

use metal::Device;
fn main() {
    let device = Device::system_default().expect("no Metal device");
    let m: usize = 32 * 1024;
    let n: usize = 32 * 1024;
    let w_bytes = m * n * 2;
    println!(
        "[probe] M5 device: {} | W = {} MiB (m={m} rows × n={n} f16 cols)",
        device.name(),
        w_bytes / (1024 * 1024)
    );

    // The shared CubeCL context gives the wgpu device/queue pair the
    // zero-copy context binds its raw dispatches to (the constructor's
    // documented contract).
    let ctx = riir_infer_gpu::cubecl_runtime::CubeCLContext::new()
        .unwrap_or_else(|e| panic!("cubecl context: {e}"));
    let (dev, queue) = match (ctx.wgpu_device(), ctx.wgpu_queue()) {
        (Some(d), Some(q)) => (d.clone(), q.clone()),
        _ => panic!("no Metal wgpu backend"),
    };
    let gemm =
        riir_infer_gpu::gemm_dense_f16_metal_zero_copy::MetalDenseF16ZeroCopyGemm::new(dev, queue)
            .unwrap_or_else(|e| panic!("dense context: {e}"));
    println!(
        "[probe] context armed (relaxed_precision={})",
        gemm.relaxed_precision()
    );

    let client = ctx.client();

    // Host-side W: zeroed f16 bytes (the probe measures READ bandwidth —
    // the values are irrelevant, the pattern is not).
    let w_host = vec![0u8; w_bytes];
    let x = vec![0.0f32; n];
    let out = vec![0u8; m * 4];
    let w_h = client.create_from_slice(bytemuck::cast_slice(&w_host));
    let x_h = client.create_from_slice(bytemuck::cast_slice(&x));
    let o_h = client.create_from_slice(&out);

    for _ in 0..3 {
        gemm.dispatch_gemv(&client, &w_h, m, n, &x_h, &o_h)
            .expect("dispatch");
        gemm.wait_last();
    }
    let mut rates = Vec::new();
    for _ in 0..10 {
        let t = std::time::Instant::now();
        gemm.dispatch_gemv(&client, &w_h, m, n, &x_h, &o_h)
            .expect("dispatch");
        gemm.wait_last();
        let dt = t.elapsed().as_secs_f64();
        rates.push(w_bytes as f64 / dt / 1e9);
    }
    rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "[probe] decode-class read BW: median {:.1} GB/s (min {:.1}, max {:.1}, 10 dispatches of {:.2} GiB)",
        rates[5],
        rates[0],
        rates[9],
        w_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );
}
