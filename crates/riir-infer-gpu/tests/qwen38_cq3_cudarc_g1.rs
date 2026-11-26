#![cfg(all(feature = "cq3_cuda", not(target_os = "macos")))]
//! Issue 1186 item (d) — the CQ3 cudarc twin's G1 battery (4090-gated).
//!
//! The pins (mirroring the CubeCL battery's tiered bar, CUDA posture):
//! - **g1t (tiled bit-parity)** — the tiled GEMV vs the HOST reference
//!   (`PackedCq3Tensor::gemv_i8_sdot`) on HOST-STAGED inputs: `to_bits`
//!   equality (the shared-memory de-scale makes the final fold
//!   contract-proof; the int32 dot is exact).
//! - **g1q (staging parity)** — `cq3_quant_x8` vs the host per-group
//!   absmax staging: bits-exact (max/roundf/clamp are exact ops).
//! - **g1e (end-to-end staged)** — act-prep → quant → tiled vs the host
//!   full pipeline: `to_bits` on the whole staged path (the T3 decode-row
//!   shape; act-prep is an exact elementwise/butterfly chain — mul/add
//!   only, no contraction partner until the fold, which the shared-memory
//!   store pins).
//! - **g2 (wire form)** — the words-row length is the CubeCL handle's own
//!   formula (one definition of the wire; a drift corrupts every read).
//!
//! No CUDA device → the arm SKIPS LOUD (a deferral, never a green zero).
//!
//! ```sh
//! # On the 4090 (native CUDA, Windows):
//! cargo test -p riir-infer-gpu --features cq3_cuda --test qwen38_cq3_cudarc_g1 -- --nocapture
//! ```

use cudarc::driver::safe::CudaContext;
use riir_infer_gpu::qwen38_cq3::CqTensor;
use riir_infer_gpu::qwen38_cq3::pack_indices_b32;
use riir_infer_gpu::qwen38_cq3_cudarc::{Cq3CudarcKernels, words_row_len};
use riir_infer_gpu::qwen38_cq3_serving::PackedCq3Tensor;

/// Deterministic f32 in [lo, hi) — the LCG fill (the CubeCL battery's
/// generator, same constants).
fn lcg_fill(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut s = seed | 1;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u = ((s >> 33) as f32) / ((1u64 << 31) as f32);
        out.push(lo + u * (hi - lo));
    }
    out
}

/// A small deterministic CqTensor (g=32, b=3 — the pre-registered league
/// recipe; d_in a multiple of g). The host pipeline (fit_smooth +
/// quantize_residual) is the same one the serving load route runs.
fn fixture_tensor(d_out: usize, d_in: usize, smooth: bool) -> CqTensor {
    let g = 32usize;
    let b = 3u32;
    let w: Vec<f32> = lcg_fill(0xC13_0001, d_out * d_in, -0.5, 0.5);
    let cal: Vec<f32> = lcg_fill(0xC13_0002, 16 * d_in, -1.0, 1.0);
    let smooth =
        smooth.then(|| riir_infer_gpu::qwen38_cq3::fit_smooth(&w, d_out, d_in, &cal, 16, 0.25));
    riir_infer_gpu::qwen38_cq3::quantize_residual(
        &w,
        d_out,
        d_in,
        g,
        b,
        smooth.as_deref(),
        &cal,
        16,
    )
}

fn device_or_skip() -> Option<(std::sync::Arc<CudaContext>, Cq3CudarcKernels)> {
    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[cq3-cudarc-g1] SKIP (loud): no CUDA device — {e}");
            return None;
        }
    };
    let kernels = match Cq3CudarcKernels::new(&ctx) {
        Ok(k) => k,
        Err(e) => {
            eprintln!("[cq3-cudarc-g1] SKIP (loud): nvrtc/module — {e}");
            return None;
        }
    };
    Some((ctx, kernels))
}

/// The device-side weight + codebook form from a packed tensor (the same
/// pack path the CubeCL handle runs).
fn device_form(packed: &PackedCq3Tensor) -> (Vec<u32>, Vec<f32>, Vec<i32>) {
    let carrier = packed.unpack_indices();
    let words = pack_indices_b32(&carrier, packed.b);
    let scales: Vec<f32> = packed
        .scales
        .iter()
        .map(|&bits| half::f16::from_bits(bits).to_f32())
        .collect();
    let cb = packed.codebook();
    let c8: Vec<i32> = cb.c8.iter().map(|&v| i32::from(v)).collect();
    (words, scales, c8)
}

#[test]
fn g1t_tiled_bit_parity_vs_host() {
    let Some((ctx, kernels)) = device_or_skip() else {
        return;
    };
    let stream = ctx.new_stream().unwrap();
    let (d_out, d_in, tokens, g) = (48usize, 256usize, 3usize, 32usize);
    let groups = d_in / g;
    let cq = fixture_tensor(d_out, d_in, true);
    let packed = PackedCq3Tensor::from_cq_tensor(&cq);

    // Three tokens of ROTATED input staged by the HOST (the staged-input
    // pin: the kernels' fp freedom is only in the fold, never the dot).
    let xs: Vec<Vec<f32>> = (0..tokens)
        .map(|i| {
            let x = lcg_fill(0xC13_0010 + i as u64, d_in, -1.0, 1.0);
            packed.rotate_activation(&x)
        })
        .collect();

    let (words, scales, c8) = device_form(&packed);
    let s_x_flat: Vec<f32> = {
        let mut v = Vec::with_capacity(tokens * groups);
        for x in &xs {
            for grp in 0..groups {
                let seg = &x[grp * g..(grp + 1) * g];
                let absmax = seg.iter().fold(0.0_f32, |m, a| m.max(a.abs()));
                v.push(absmax / 127.0);
            }
        }
        v
    };
    // The HOST staging pin: exactly the rule `gemv_i8_sdot` stages
    // internally (per-group absmax/127, round-half-away clamp to
    // [-127, 127], an all-zero group stays zero) — the device dot must
    // see these codes, never raw f32, or the to_bits parity below is
    // meaningless.
    let x8_flat: Vec<i32> = {
        let mut v = vec![0_i32; tokens * d_in];
        for (tok, x) in xs.iter().enumerate() {
            for grp in 0..groups {
                let s = s_x_flat[tok * groups + grp];
                if s > 0.0 {
                    let base = tok * d_in + grp * g;
                    for (m, &val) in x[grp * g..(grp + 1) * g].iter().enumerate() {
                        v[base + m] = ((val / s).round() as i32).clamp(-127, 127);
                    }
                }
            }
        }
        v
    };

    let w_dev = stream.clone_htod(&words).unwrap();
    let sc_dev = stream.clone_htod(&scales).unwrap();
    let c8_dev = stream.clone_htod(&c8).unwrap();
    let x8_dev = stream.clone_htod(&x8_flat).unwrap();
    let sx_dev = stream.clone_htod(&s_x_flat).unwrap();
    let out_dev = stream.alloc_zeros::<f32>(tokens * d_out).unwrap();

    unsafe {
        kernels
            .launch_gemv_tiled(
                &stream,
                &w_dev,
                &sc_dev,
                &c8_dev,
                &x8_dev,
                &sx_dev,
                &out_dev,
                g,
                packed.b,
                packed.codebook().s_c,
                tokens,
                groups,
                d_out,
            )
            .expect("tiled launch");
    }
    stream.synchronize().unwrap();
    let mut got = vec![0_f32; tokens * d_out];
    stream.memcpy_dtoh(&out_dev, &mut got).unwrap();

    for (tok, x_rot) in xs.iter().enumerate() {
        let host = packed.gemv_i8_sdot(x_rot);
        for (r, (h, d)) in host
            .iter()
            .zip(got[tok * d_out..(tok + 1) * d_out].iter())
            .enumerate()
        {
            assert_eq!(
                h.to_bits(),
                d.to_bits(),
                "tiled bit-parity failed at token {tok} row {r} (host {h} vs dev {d})"
            );
        }
    }
}

#[test]
fn g1q_staging_bits_parity_vs_host() {
    let Some((ctx, kernels)) = device_or_skip() else {
        return;
    };
    let stream = ctx.new_stream().unwrap();
    let (d_in, tokens, g) = (256usize, 2usize, 32usize);
    let groups = d_in / g;
    let xr: Vec<f32> = lcg_fill(0xC13_0030, tokens * d_in, -3.0, 3.0);

    let xr_dev = stream.clone_htod(&xr).unwrap();
    let x8_dev = stream.alloc_zeros::<i32>(tokens * d_in).unwrap();
    let sx_dev = stream.alloc_zeros::<f32>(tokens * groups).unwrap();

    unsafe {
        kernels
            .launch_quant_x8(&stream, &xr_dev, &x8_dev, &sx_dev, g, tokens, groups)
            .expect("quant launch");
    }
    stream.synchronize().unwrap();
    let mut x8 = vec![0_i32; tokens * d_in];
    let mut s_x = vec![0_f32; tokens * groups];
    stream.memcpy_dtoh(&x8_dev, &mut x8).unwrap();
    stream.memcpy_dtoh(&sx_dev, &mut s_x).unwrap();

    for (pos, row) in xr.chunks_exact(d_in).enumerate() {
        for (grp, seg) in row.chunks_exact(g).enumerate() {
            let absmax = seg.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
            let s = absmax / 127.0;
            assert_eq!(
                s_x[pos * groups + grp].to_bits(),
                s.to_bits(),
                "staging scale drift at token {pos} group {grp}"
            );
            for (m, v) in seg.iter().enumerate() {
                let host_q = if s > 0.0 {
                    ((v / s).round() as i32).clamp(-127, 127)
                } else {
                    0
                };
                assert_eq!(
                    x8[pos * d_in + grp * g + m],
                    host_q,
                    "staging x8 drift at token {pos} group {grp} elem {m}"
                );
            }
        }
    }
}

#[test]
fn g1e_end_to_end_staged_bit_parity() {
    let Some((ctx, kernels)) = device_or_skip() else {
        return;
    };
    let stream = ctx.new_stream().unwrap();
    let (d_out, d_in, tokens, g) = (32usize, 128usize, 2usize, 32usize);
    let groups = d_in / g;
    let cq = fixture_tensor(d_out, d_in, true);
    let packed = PackedCq3Tensor::from_cq_tensor(&cq);

    // The full device pipeline per token: raw x → act_prep → quant → tiled.
    let smooth_len = usize::from(packed.smooth.is_some()) * d_in;
    let smooth: Vec<f32> = packed
        .smooth
        .as_ref()
        .map(|s| s.repeat(tokens))
        .unwrap_or_else(|| vec![0_f32; 16]);
    let rot = packed.rotation();
    let (s1, s2, perm) = rot.device_params();
    let perm: Vec<u32> = perm.to_vec();
    let s1: Vec<f32> = s1.to_vec();
    let s2: Vec<f32> = s2.to_vec();

    let (words, scales, c8) = device_form(&packed);
    let x: Vec<f32> = lcg_fill(0xC13_0040, tokens * d_in, -1.0, 1.0);

    let x_dev = stream.clone_htod(&x).unwrap();
    let sm_dev = stream.clone_htod(&smooth).unwrap();
    let perm_dev = stream.clone_htod(&perm).unwrap();
    let s1_dev = stream.clone_htod(&s1).unwrap();
    let s2_dev = stream.clone_htod(&s2).unwrap();
    let rot_dev = stream.alloc_zeros::<f32>(tokens * d_in).unwrap();
    let x8_dev = stream.alloc_zeros::<i32>(tokens * d_in).unwrap();
    let sx_dev = stream.alloc_zeros::<f32>(tokens * groups).unwrap();
    let w_dev = stream.clone_htod(&words).unwrap();
    let sc_dev = stream.clone_htod(&scales).unwrap();
    let c8_dev = stream.clone_htod(&c8).unwrap();
    let out_dev = stream.alloc_zeros::<f32>(tokens * d_out).unwrap();

    unsafe {
        kernels
            .launch_act_prep(
                &stream, &x_dev, &sm_dev, &perm_dev, &s1_dev, &s2_dev, &rot_dev, g, smooth_len,
                tokens, groups,
            )
            .expect("act_prep");
        kernels
            .launch_quant_x8(&stream, &rot_dev, &x8_dev, &sx_dev, g, tokens, groups)
            .expect("quant");
        kernels
            .launch_gemv_tiled(
                &stream,
                &w_dev,
                &sc_dev,
                &c8_dev,
                &x8_dev,
                &sx_dev,
                &out_dev,
                g,
                packed.b,
                packed.codebook().s_c,
                tokens,
                groups,
                d_out,
            )
            .expect("gemv");
    }
    stream.synchronize().unwrap();
    let mut got = vec![0_f32; tokens * d_out];
    stream.memcpy_dtoh(&out_dev, &mut got).unwrap();

    // Host pipeline per token (rotate → stage → gemv).
    for (pos, xrow) in x.chunks_exact(d_in).enumerate() {
        let rotated = packed.rotate_activation(xrow);
        let host = packed.gemv_i8_sdot(&rotated);
        for (r, (h, d)) in host
            .iter()
            .zip(got[pos * d_out..(pos + 1) * d_out].iter())
            .enumerate()
        {
            assert_eq!(
                h.to_bits(),
                d.to_bits(),
                "e2e staged parity failed at token {pos} row {r} (host {h} vs dev {d})"
            );
        }
    }
}

#[test]
fn g2_words_row_len_matches_the_cubecl_form() {
    // The device-form words count is the CubeCL handle's own formula — one
    // definition of the wire (a drift here corrupts every read).
    for &(d_in, b) in &[(5120usize, 3u32), (256usize, 4u32), (16384usize, 3u32)] {
        assert_eq!(
            words_row_len(d_in, b),
            (d_in * b as usize).div_ceil(32),
            "words_row_len({d_in}, {b})"
        );
    }
    // And the launcher's own buffer contract holds on a real packed tensor:
    // `words` covers rows·words_row_len(d_in, b) (the launch's # Safety
    // guarantee — a short words buffer is an OOB device read).
    for &(d_out, d_in) in &[(8usize, 256usize), (4usize, 5120usize)] {
        let cq = fixture_tensor(d_out, d_in, false);
        let packed = PackedCq3Tensor::from_cq_tensor(&cq);
        let (words, _, _) = device_form(&packed);
        assert_eq!(
            words.len(),
            d_out * words_row_len(d_in, packed.b),
            "words buffer contract (rows {d_out}, d_in {d_in}, b {})",
            packed.b
        );
    }
}
