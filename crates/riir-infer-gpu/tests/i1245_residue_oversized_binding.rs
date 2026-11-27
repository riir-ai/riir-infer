//! Issue 1245 residue — the oversized-binding trap is CLOSED across the
//! remaining len()-deriving GEMV families.
//!
//! The Bench-1256 root cause: a kernel-visible slice length is the binding's
//! BACKING slice size (the launcher registers it in the metadata; the
//! `BufferArg::from_raw_parts` length does not constrain it). Any kernel that
//! derived `n`/`m` from `input.len()`/`output.len()` read the whole backing
//! allocation whenever a caller bound an offset view into a bigger buffer
//! (the chunk tail's `tmp_b`/`logits_b` pool scratch shape).
//!
//! These cells construct exactly that binding shape — a handle whose backing
//! allocation is 2× the bound window (`parent.offset_start(0)`) — for one
//! kernel per params-channel shape and assert the EXACT result comes back:
//!
//! - 2-param channel `[n, m]`: `gemv_plane_f16_f32` (f16 plane),
//!   `gemv_q4k_plane` (q4k plane), `gemv_geglu_plane_f32` (geglu plane).
//! - 3-param channel `[n, m_a, m_total]`: `gemv_plane_f32_pair` (the pair).
//!
//! The remaining kernels in each family share the launcher-side params
//! pattern (arity- and order-checked by the compiler, value-checked by the
//! per-family in-file CPU-reference unit tests); these cells prove the
//! channel itself overrides the backing size.
//!
//! Run:
//! ```sh
//! cargo test --release -p riir-infer-gpu --features qwen38_wgpu \
//!   --test i1245_residue_oversized_binding -- --nocapture
//! ```
#![cfg(all(test, feature = "qwen38_wgpu"))]

use bytemuck::Zeroable;
use cubecl::client::ComputeClient;
use cubecl::features::Plane;
use cubecl::server::Handle;
use riir_infer_gpu::cubecl_runtime::{ActiveRuntime, CubeCLContext};

fn plane_ops(client: &ComputeClient<ActiveRuntime>) -> bool {
    client.features().plane.contains(Plane::Ops)
}

/// Oversized-backing variant of an exact input: a parent allocation of
/// `2 * n` f32 with the payload at offset 0, bound as the offset view.
/// The view's window is `n` f32; its BACKING is the parent's `2 * n` — the
/// exact shape that poisoned the len()-deriving kernels.
fn oversized_input(client: &ComputeClient<ActiveRuntime>, payload: &[f32]) -> Handle {
    let mut parent = payload.to_vec();
    parent.resize(payload.len() * 2, f32::NAN);
    let parent_h = client.create_from_slice(bytemuck::cast_slice(&parent));
    parent_h.offset_start(0)
}

fn oversize_output(client: &ComputeClient<ActiveRuntime>, m_elems: usize) -> Handle {
    let parent = client.empty(m_elems * 2 * core::mem::size_of::<f32>());
    parent.offset_start(0)
}

fn read_f32(client: &ComputeClient<ActiveRuntime>, h: &Handle) -> Vec<f32> {
    let bytes = client.read_one(h.clone()).expect("readback");
    bytemuck::cast_slice(&bytes).to_vec()
}

/// A 4×4 identity plane GEMV whose input handle's backing is 2× the window.
/// The len()-deriving kernel read n = 8 (backing) → row stride 8 → garbage;
/// the params kernel reads n = 4 from params → the identity column.
#[test]
fn f16_plane_kernel_reads_params_not_backing_size() {
    let ctx = CubeCLContext::new().expect("cubecl context");
    let client = ctx.client();
    if !plane_ops(&client) {
        eprintln!("Skipping: device does not support plane ops");
        return;
    }

    let m = 4usize;
    let n = 4usize;
    let weight_f32: Vec<f32> = vec![
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 1.0,
    ];
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

    let w_f16: Vec<half::f16> = weight_f32.iter().map(|&x| half::f16::from_f32(x)).collect();
    let weight_handle = client.create_from_slice(bytemuck::cast_slice(&w_f16));
    let input_handle = oversized_input(&client, &input);
    let output_handle = client.empty(m * core::mem::size_of::<f32>());

    unsafe {
        riir_infer_gpu::gemv_f16_cubecl::GemvF16CubeCL::launch_plane::<ActiveRuntime>(
            &client,
            weight_handle,
            input_handle,
            output_handle.clone(),
            m,
            n,
        );
    }

    let out = read_f32(&client, &output_handle);
    for (i, &got) in out.iter().enumerate() {
        let want = input[i];
        assert!(
            (got - want).abs() < 1e-5,
            "row {i}: got {got}, want {want} — the kernel bounded n from the \
             BACKING size, not the params buffer"
        );
    }
}

/// The q4k plane kernel at a packed-block shape whose input handle's backing
/// is 2× the window: the len()-deriving kernel doubled `q4k_stride` (n from
/// the backing) and read past every row; the params kernel is exact vs the
/// canonical CPU reference (`riir_infer_core::quant::q4k::gemv_q4_k`).
#[test]
fn q4k_plane_kernel_reads_params_not_backing_size() {
    use riir_infer_core::quant::q4k::{gemv_q4_k, quantize_row_q4_k, BlockQ4K, QK_K};
    use riir_infer_gpu::gemv_q4k_cubecl::{GemvQ4KCubeCL, Q4KHandle};

    let ctx = CubeCLContext::new().expect("cubecl context");
    let client = ctx.client();
    if !plane_ops(&client) {
        eprintln!("Skipping: device does not support plane ops");
        return;
    }

    let m = 4usize;
    let n = 256usize; // one q4k block per row
    let weight: Vec<f32> = (0..m * n)
        .map(|i| ((i as f32 * 0.618).sin()) * 0.5)
        .collect();

    let mut all_blocks = Vec::new();
    for row in 0..m {
        let start = all_blocks.len();
        all_blocks.resize(start + n.div_ceil(QK_K), BlockQ4K::zeroed());
        quantize_row_q4_k(&weight[row * n..(row + 1) * n], &mut all_blocks[start..]);
    }
    let q4k_handle = Q4KHandle::from_blocks(&client, &all_blocks, m, n);

    let input: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.37).sin()) * 2.0).collect();
    let input_handle = oversized_input(&client, &input);
    let output_handle = client.empty(m * core::mem::size_of::<f32>());

    unsafe {
        GemvQ4KCubeCL::launch::<ActiveRuntime>(
            &client,
            &q4k_handle,
            input_handle,
            output_handle.clone(),
        );
    }

    let out = read_f32(&client, &output_handle);
    let want = gemv_q4_k(&all_blocks, &input, m, n);
    for (i, (&got, &exp)) in out.iter().zip(want.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 5e-2 * exp.abs().max(1.0),
            "row {i}: got {got}, want {exp} — q4k_stride bounded from the \
             BACKING size, not the params buffer"
        );
    }
}

/// The geglu plane kernel whose input handle's backing is 2× the window.
#[test]
fn geglu_plane_kernel_reads_params_not_backing_size() {
    use riir_infer_gpu::gemv_geglu_cubecl::GemvGegluCubeCL;

    let ctx = CubeCLContext::new().expect("cubecl context");
    let client = ctx.client();
    if !plane_ops(&client) {
        eprintln!("Skipping: device does not support plane ops");
        return;
    }

    let m = 4usize;
    let n = 8usize;
    let gate: Vec<f32> = (0..m * n)
        .map(|i| ((i as f32 * 0.618).sin()) * 0.5)
        .collect();
    let up: Vec<f32> = (0..m * n)
        .map(|i| ((i as f32 * 0.37).sin()) * 0.5)
        .collect();
    let input: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.111).cos()) * 1.5).collect();

    let gate_handle = client.create_from_slice(bytemuck::cast_slice(&gate));
    let up_handle = client.create_from_slice(bytemuck::cast_slice(&up));
    let input_handle = oversized_input(&client, &input);
    let output_handle = client.empty(m * core::mem::size_of::<f32>());

    unsafe {
        GemvGegluCubeCL::launch::<ActiveRuntime>(
            &client,
            gate_handle,
            up_handle,
            input_handle,
            output_handle.clone(),
            m,
            n,
        );
    }

    // CPU mirror: row = gelu_tanh(gate_row · x) * (up_row · x) — the same
    // gelu_tanh constants the in-file reference uses.
    let gelu_tanh = |v: f32| -> f32 {
        let x_cubed = v * v * v;
        let inner = 0.797_884_6 * (v + 0.044715 * x_cubed);
        0.5 * v * (1.0 + inner.tanh())
    };
    let out = read_f32(&client, &output_handle);
    for r in 0..m {
        let gd: f32 = gate[r * n..(r + 1) * n]
            .iter()
            .zip(input.iter())
            .map(|(&g, &x)| g * x)
            .sum();
        let ud: f32 = up[r * n..(r + 1) * n]
            .iter()
            .zip(input.iter())
            .map(|(&u, &x)| u * x)
            .sum();
        let want = gelu_tanh(gd) * ud;
        assert!(
            (out[r] - want).abs() < 5e-3 * want.abs().max(1.0),
            "row {r}: got {}, want {want} — dims bounded from the BACKING \
             size, not the params buffer",
            out[r]
        );
    }
}

/// The pair kernel: both output handles are offset views whose backing is
/// 2× the window (the chunk tail's pool-scratch shape). The len()-deriving
/// kernel read m_a/m_total from the backings → garbage halves; the params
/// kernel splits rows exactly at m_a = m.
#[test]
fn pair_kernel_reads_params_not_backing_sizes() {
    use riir_infer_gpu::gemv_cubecl::GemvCubeCL;

    let ctx = CubeCLContext::new().expect("cubecl context");
    let client = ctx.client();
    if !plane_ops(&client) {
        eprintln!("Skipping: device does not support plane ops");
        return;
    }

    let m = 4usize;
    let n = 16usize;
    let wa: Vec<f32> = (0..m * n)
        .map(|i| ((i as f32 * 0.618).sin()) * 0.5)
        .collect();
    let wb: Vec<f32> = (0..m * n)
        .map(|i| ((i as f32 * 0.234).sin()) * 0.5)
        .collect();
    let input: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.37).sin()) * 2.0).collect();

    let wa_h = client.create_from_slice(bytemuck::cast_slice(&wa));
    let wb_h = client.create_from_slice(bytemuck::cast_slice(&wb));
    let in_h = oversized_input(&client, &input);
    let oa = oversize_output(&client, m);
    let ob = oversize_output(&client, m);

    unsafe {
        GemvCubeCL::launch_pair::<ActiveRuntime>(
            &client,
            wa_h,
            wb_h,
            in_h,
            oa.clone(),
            ob.clone(),
            m,
            n,
            128,
        );
    }

    let out_a = read_f32(&client, &oa);
    let out_b = read_f32(&client, &ob);
    for r in 0..m {
        let want_a: f32 = wa[r * n..(r + 1) * n]
            .iter()
            .zip(input.iter())
            .map(|(&w, &x)| w * x)
            .sum();
        let want_b: f32 = wb[r * n..(r + 1) * n]
            .iter()
            .zip(input.iter())
            .map(|(&w, &x)| w * x)
            .sum();
        assert!(
            (out_a[r] - want_a).abs() < 1e-3,
            "a row {r}: got {}, want {want_a} — m_a bounded from the BACKING \
             size, not the params buffer",
            out_a[r]
        );
        assert!(
            (out_b[r] - want_b).abs() < 1e-3,
            "b row {r}: got {}, want {want_b} — m_total bounded from the \
             BACKING size, not the params buffer",
            out_b[r]
        );
    }
}
