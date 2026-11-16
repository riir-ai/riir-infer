//! Issue 1186 T3 — the CQ3 serving-arm GPU battery (Metal posture).
//!
//! The tiered correctness bar (the findings this battery MEASURED, recorded
//! in the module doc + the issue):
//! - Metal's cubecl MSL defaults reassociate/contract fp chains: the raw
//!   f32 divide is 1-ulp-off vs the host's correctly-rounded IEEE divide
//!   (72/256, max 1 ulp — the unit probe), and mul/add chains drift 1–2 ulp
//!   even through a store round-trip. So:
//! - **G1 parity (the exactness witness)**: the divide-FREE GEMV over HOST
//!   staged inputs — pinned abs ≤ 1e-5 (the Metal fp residue; measured ≤ 2 ulp on non-cancelling rows,
//!   smooth±). Every SEMANTIC property is still exact under this bound: a
//!   wrong packed-index read, codebook lookup or i32 dot explodes past it;
//!   the residual is Metal's fp spelling of the final accumulate. The
//!   bit-exact rung = a raw-MSL kernel at `-ffp-contract=off` (the
//!   gemm_dense_f16_metal_zero_copy route), named in the issue.
//! - **G1b batched identity**: batched p-launch vs p singles — `to_bits`
//!   (GPU-vs-GPU, same kernel posture: deterministic).
//! - **G1e end-to-end**: full GPU pipeline vs host — relative pin 1e-4.
//! - **G1c plane tier**: the plane kernel (the league row) vs host —
//!   relative pin 2e-4.
//! - **G2 engagement**: launch counters move with dispatch.
//!
//! ```sh
//! CARGO_TARGET_DIR=/tmp/i1186gpu cargo test -p riir-infer-gpu \
//!   --features cq3,qwen38_wgpu --test qwen38_cq3_gpu_g1 -- --nocapture
//! ```
//!
//! No GPU (CubeCLContext::new fails) → the arm SKIPS LOUD (a deferral,
//! never a green zero).

#![cfg(all(feature = "cq3", feature = "cubecl_runtime"))]

use cubecl::prelude::*;
use riir_infer_gpu::cubecl_runtime::{ActiveComputeClient, CubeCLContext};
use riir_infer_gpu::qwen38_cq3::CqTensor;
use riir_infer_gpu::qwen38_cq3_gpu::{Cq3GemvCubeCL, Cq3Handle, pack_indices_b32, words_per_group};
use riir_infer_gpu::qwen38_cq3_serving::PackedCq3Tensor;

/// The env latch (RIIR_QWEN38_CQ3_TILED) is process-global — the tests that
/// toggle it serialize on this lock (the libtest default runs tests in
/// parallel threads; two arms toggling mid-launch cross-contaminates).
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Deterministic pseudo-random f32 in [lo, hi) — a seeded LCG over bits.
fn lcg_fill(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut s = seed | 1;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (s >> 40) as f32 / (1u64 << 24) as f32;
        out.push(lo + (hi - lo) * bits);
    }
    out
}

/// A small tensor quantized with the REAL recipe (the fold rides `smooth`
/// explicitly), packed into the device form.
fn make_tensor(
    d_out: usize,
    d_in: usize,
    g: usize,
    b: u32,
    smooth: Option<Vec<f32>>,
) -> PackedCq3Tensor {
    let w = lcg_fill(0xC0FFEE, d_out * d_in, -0.5, 0.5);
    let t = CqTensor::quantize(&w, d_out, d_in, g, b, smooth.as_deref());
    PackedCq3Tensor::from_cq_tensor(&t)
}

fn ctx() -> Option<ActiveComputeClient> {
    match CubeCLContext::new() {
        Ok(ctx) => Some(ctx.client()),
        Err(e) => {
            eprintln!(
                "[skip] no GPU (CubeCLContext::new failed: {e}) — deferral, never a green zero"
            );
            None
        }
    }
}

fn upload_f32(client: &ActiveComputeClient, v: &[f32]) -> riir_infer_gpu::cubecl_runtime::Handle {
    client.create_from_slice(CubeElement::as_bytes(v))
}

fn upload_i32(client: &ActiveComputeClient, v: &[i32]) -> riir_infer_gpu::cubecl_runtime::Handle {
    client.create_from_slice(CubeElement::as_bytes(v))
}

fn read_f32(client: &ActiveComputeClient, h: &riir_infer_gpu::cubecl_runtime::Handle) -> Vec<f32> {
    client
        .read_one(h.clone())
        .expect("read")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// The HOST reference's staging loop (verbatim from
/// `PackedCq3Tensor::gemv_i8_sdot`): per group, the absmax fold, `s =
/// absmax/127`, the element-local `(v/s).round()` clamp ±127.
fn host_stage(x_rot: &[f32], g: usize) -> (Vec<i32>, Vec<f32>) {
    let groups_in = x_rot.len() / g;
    let mut x8 = vec![0_i32; x_rot.len()];
    let mut s_x = vec![0.0_f32; groups_in];
    for grp in 0..groups_in {
        let seg = &x_rot[grp * g..(grp + 1) * g];
        let absmax = seg.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
        let s = absmax / 127.0;
        s_x[grp] = s;
        if s > 0.0 {
            for (m, v) in seg.iter().enumerate() {
                x8[grp * g + m] = ((v / s).round() as i32).clamp(-127, 127);
            }
        }
    }
    (x8, s_x)
}

/// G1 — the exact witness: divide-free GPU GEMV over HOST staged inputs,
/// `to_bits` vs the host reference. Runs the TILED kernel explicitly (the
/// plane kernel is the tree tier — G1c).
fn run_g1_parity(d_out: usize, d_in: usize, g: usize, b: u32, with_smooth: bool) {
    let Some(client) = ctx() else { return };
    let _env = ENV_LOCK.lock().expect("env lock");
    // SAFETY: the ENV_LOCK serializes every toggle.
    unsafe { std::env::set_var("RIIR_QWEN38_CQ3_TILED", "1") };
    let smooth = with_smooth.then(|| lcg_fill(0x5EED, d_in, 0.25, 4.0));
    let t = make_tensor(d_out, d_in, g, b, smooth);
    let host = Cq3Handle::from_packed(&client, &t);

    let x = lcg_fill(0xDACE, d_in, -1.0, 1.0);
    let x_rot = t.rotate_activation(&x);
    let want = t.gemv_i8_sdot(&x_rot);
    let (x8, s_x) = host_stage(&x_rot, g);

    let x8_h = upload_i32(&client, &x8);
    let sx_h = upload_f32(&client, &s_x);
    let out_h = client.empty(d_out * core::mem::size_of::<f32>());
    unsafe { Cq3GemvCubeCL::launch_staged(&client, &host, x8_h, sx_h, out_h.clone(), 1) };
    let got = read_f32(&client, &out_h);
    unsafe { std::env::remove_var("RIIR_QWEN38_CQ3_TILED") };
    drop(_env);

    assert_eq!(got.len(), want.len(), "output len");
    // The pin is ABSOLUTE: with cancellation rows (near-zero sums of O(1)
    // contributions) ulp-of-output amplifies arbitrarily, while a semantic
    // error (wrong index/codebook/int op) explodes absolutely. The measured
    // Metal fp residue is ~1e-7 absolute (1–2 rounding steps).
    let mut max_abs = 0.0_f64;
    for (i, (a, w)) in got.iter().zip(want.iter()).enumerate() {
        let d = ((a - w).abs()) as f64;
        max_abs = max_abs.max(d);
        assert!(
            d <= 1e-5,
            "G1[{i}] abs {d:.3e} at {d_out}x{d_in} g={g} b={b} smooth={with_smooth} (pin 1e-5): gpu {a} vs host {w}"
        );
    }
    eprintln!(
        "[g1] PASS {d_out}x{d_in} g={g} b={b} smooth={with_smooth}: {d_out} rows abs ≤1e-5 (measured max {max_abs:.3e})"
    );
}

#[test]
fn g1_parity_semantic_g32_b3() {
    run_g1_parity(64, 256, 32, 3, false);
    run_g1_parity(64, 256, 32, 3, true);
}

#[test]
fn g1_parity_semantic_g16_b3() {
    run_g1_parity(48, 128, 16, 3, false);
    run_g1_parity(48, 128, 16, 3, true);
}

#[test]
fn g1_parity_semantic_b8_wide_carrier() {
    // b=8 exercises the no-word-crossing edge of the bit-read path; the
    // recipe pins b=3, the read must hold across b.
    run_g1_parity(32, 128, 32, 8, true);
}

/// G1b — the batched launch is to_bits-identical to p single launches
/// (tiled arm, same staged bits).
#[test]
fn g1b_batched_matches_single_to_bits() {
    let Some(client) = ctx() else { return };
    let (d_out, d_in, g, b) = (64usize, 256usize, 32usize, 3u32);
    let t = make_tensor(d_out, d_in, g, b, None);
    let host = Cq3Handle::from_packed(&client, &t);
    let p = 7usize;
    let xs = lcg_fill(0xBA7C, p * d_in, -1.0, 1.0);

    // Stage the whole batch on the host (identical bits for both arms).
    let mut x8 = Vec::with_capacity(p * d_in);
    let mut s_x = Vec::with_capacity(p * (d_in / g));
    for tok in 0..p {
        let x_rot = t.rotate_activation(&xs[tok * d_in..(tok + 1) * d_in]);
        let (b8, bs) = host_stage(&x_rot, g);
        x8.extend(b8);
        s_x.extend(bs);
    }
    // BOTH arms run the SAME kernel posture so GPU-vs-GPU is deterministic
    // (the default plane arm — the batched prefill posture; the tiled parity
    // arm is decode-shaped and launch_staged refuses p>1 under it).
    let _env = ENV_LOCK.lock().expect("env lock");
    // Batched: one launch.
    let x8_h = upload_i32(&client, &x8);
    let sx_h = upload_f32(&client, &s_x);
    let ob = client.empty(p * d_out * core::mem::size_of::<f32>());
    unsafe { Cq3GemvCubeCL::launch_staged(&client, &host, x8_h, sx_h, ob.clone(), p) };
    let got = read_f32(&client, &ob);

    // Singles: p launches through the same posture.
    let mut want = Vec::with_capacity(p * d_out);
    for tok in 0..p {
        let x8_h = upload_i32(&client, &x8[tok * d_in..(tok + 1) * d_in]);
        let sx_h = upload_f32(&client, &s_x[tok * (d_in / g)..(tok + 1) * (d_in / g)]);
        let oh = client.empty(d_out * core::mem::size_of::<f32>());
        unsafe { Cq3GemvCubeCL::launch_staged(&client, &host, x8_h, sx_h, oh.clone(), 1) };
        want.extend(read_f32(&client, &oh));
    }
    drop(_env);

    assert_eq!(got.len(), want.len());
    for (i, (a, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            w.to_bits(),
            "G1b[{i}] batched vs single: {a} vs {w}"
        );
    }
    eprintln!("[g1b] PASS batched p={p} to_bits-identical to singles");
}

/// G1e — the FULL GPU pipeline (rotate → stage → gemv) vs host, tolerance
/// tier (the default plane posture). The staging's int8 values should be
/// identical despite the 1-ulp x_rot drift (round flips sit at exact .5
/// boundaries — lcg floats don't land there); s_x drifts ≤1 ulp → outputs
/// drift ~1 ulp.
#[test]
fn g1e_end_to_end_tolerance() {
    let Some(client) = ctx() else { return };
    let (d_out, d_in, g, b) = (64usize, 256usize, 32usize, 3u32);
    let smooth = Some(lcg_fill(0x5EED, d_in, 0.25, 4.0));
    let t = make_tensor(d_out, d_in, g, b, smooth);
    let host = Cq3Handle::from_packed(&client, &t);

    let x = lcg_fill(0xDACE, d_in, -1.0, 1.0);
    let x_rot = t.rotate_activation(&x);
    let want = t.gemv_i8_sdot(&x_rot);

    let xh = upload_f32(&client, &x);
    let oh = client.empty(d_out * core::mem::size_of::<f32>());
    // TILED for the closest tier (plane adds the tree-order term).
    let _env = ENV_LOCK.lock().expect("env lock");
    // SAFETY: the ENV_LOCK serializes every toggle.
    unsafe { std::env::set_var("RIIR_QWEN38_CQ3_TILED", "1") };
    unsafe { Cq3GemvCubeCL::launch(&client, &host, xh, oh.clone(), 1) };
    unsafe { std::env::remove_var("RIIR_QWEN38_CQ3_TILED") };
    drop(_env);
    let got = read_f32(&client, &oh);

    let mut worst = 0.0_f64;
    for (i, (a, w)) in got.iter().zip(want.iter()).enumerate() {
        let denom = w.abs().max(1e-9) as f64;
        let rel = ((a - w).abs() as f64) / denom;
        worst = worst.max(rel);
        assert!(
            rel < 1e-4,
            "G1e[{i}] rel {rel:.3e} at e2e tier (pin 1e-4): gpu {a} vs host {w}"
        );
    }
    eprintln!("[g1e] PASS full pipeline worst rel {worst:.3e} (pin 1e-4)");
}

/// G1c — the plane kernel (the default league row) vs the host reference
/// under the tolerance pin.
#[test]
fn g1c_plane_within_tolerance_of_host() {
    let Some(client) = ctx() else { return };
    let (d_out, d_in, g, b) = (128usize, 512usize, 32usize, 3u32);
    let t = make_tensor(d_out, d_in, g, b, Some(lcg_fill(0x5EED, d_in, 0.25, 4.0)));
    let host = Cq3Handle::from_packed(&client, &t);

    let x = lcg_fill(0xDACE, d_in, -1.0, 1.0);
    let x_rot = t.rotate_activation(&x);
    let want = t.gemv_i8_sdot(&x_rot);

    let xh = upload_f32(&client, &x);
    let oh = client.empty(d_out * core::mem::size_of::<f32>());
    unsafe { Cq3GemvCubeCL::launch(&client, &host, xh, oh.clone(), 1) };
    let got = read_f32(&client, &oh);

    let mut worst = 0.0_f64;
    for (i, (a, w)) in got.iter().zip(want.iter()).enumerate() {
        let denom = w.abs().max(1e-9) as f64;
        let rel = ((a - w).abs() as f64) / denom;
        worst = worst.max(rel);
        assert!(
            rel < 2e-4,
            "G1c[{i}] rel {rel:.3e} at plane tier (pin 2e-4): gpu {a} vs host {w}"
        );
    }
    eprintln!("[g1c] PASS plane tier worst rel {worst:.3e} (pin 2e-4)");
}

/// G2 — the engagement counters move with dispatches (the T1.5 class:
/// a silent non-dispatch must never read as a green run).
#[test]
fn g2_launch_counters_move() {
    let Some(client) = ctx() else { return };
    let t = make_tensor(32, 128, 32, 3, None);
    let host = Cq3Handle::from_packed(&client, &t);
    let (p0, g0) = host.launch_counts();
    let xh = upload_f32(&client, &vec![0.5_f32; 128]);
    let oh = client.empty(32 * core::mem::size_of::<f32>());
    unsafe { Cq3GemvCubeCL::launch(&client, &host, xh.clone(), oh.clone(), 1) };
    unsafe { Cq3GemvCubeCL::launch(&client, &host, xh, oh, 1) };
    let (p1, g1) = host.launch_counts();
    assert_eq!(p1, p0 + 2, "act-prep counter moved twice");
    assert_eq!(g1, g0 + 2, "gemv counter moved twice");
    eprintln!("[g2] PASS counters {p0}->{p1} / {g0}->{g1}");
}

/// Host-form sanity (no GPU): the u32 device pack round-trips and the
/// word math agrees with the u64 reference pack.
#[test]
fn g0_pack32_identity_and_word_math() {
    let carrier: Vec<u8> = (0..97).map(|i| (i * 5 % 8) as u8).collect();
    let b = 3u32;
    let words64 = riir_infer_gpu::qwen38_cq3_serving::pack_indices_b(&carrier, b);
    let words32 = pack_indices_b32(&carrier, b);
    for (i, &v) in carrier.iter().enumerate() {
        let w = words64[i * b as usize / 64];
        let off = (i as u64 * b as u64 % 64) as u32;
        let mask = (1u64 << b) - 1;
        let lo = w >> off;
        let mut pv = lo;
        if off + b > 64 {
            pv = lo | (words64[i * b as usize / 64 + 1] << (64 - off));
        }
        assert_eq!((pv & mask) as u8, v, "u64 ref read [{i}]");
        let w32 = words32[i * b as usize / 32];
        let off32 = i as u32 * b % 32;
        let mask32 = (1u32 << b) - 1;
        let lo32 = w32 >> off32;
        let mut pv32 = lo32;
        if off32 + b > 32 {
            pv32 = lo32 | (words32[i * b as usize / 32 + 1] << (32 - off32));
        }
        assert_eq!((pv32 & mask32) as u8, v, "u32 device read [{i}]");
    }
    assert_eq!(words_per_group(32, 3), 3, "g32 b3 = 3 words/group");
    assert_eq!(words_per_group(16, 3), 2, "g16 b3 = ceil(48/32)");
    eprintln!("[g0] PASS pack32 identity + word math");
}
