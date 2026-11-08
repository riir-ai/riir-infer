#![cfg(feature = "cq3")]
//! Issue 1186 T1 — the CQ3 G1 fixture battery (riir-infer-gpu).
//!
//! Provenance: Cactus "Whole-Model Quantization Below 4 Bits Per Weight"
//! (NeurIPS 2026) via katgpt-rs Research 657 §A3 / issue 987 lead A3. The
//! never-unpack GEMV prototype on the qwen3.8 dense lane (issue 1176's
//! league posture; NOT Bonsai).
//!
//! The pins:
//! - **g1a (bit-exact, the G1 class)** — the fused LUT GEMV against
//!   reference dequant + plain sequential matmul, `to_bits` equality, with
//!   and without the SmoothQuant fold, at g = 128 (the paper's b_eff shape)
//!   and g = 16.
//! - **g1b** — the int8 SDOT arm against the f32 LUT arm under a PINNED
//!   measured tolerance (int8 is a second lossy surface — bounded, never
//!   claimed bit-exact).
//! - **g1c** — rotation orthogonality (norm + inner-product preservation).
//! - **g1d** — recipe-level accuracy vs the f32 reference at b = 4 and
//!   b = 3 (pinned from measurement; GPTQ residual is T2 — this pin is the
//!   NO-residual floor, and it is the floor the residual must improve on).
//! - **g1e** — the byte accounting: `bits_per_weight() == b + 16/g` (the
//!   paper's `b + 0.125` at g = 128), asserted to 1e-9.
//! - **g1f** — determinism: quantize twice / derive twice → bit-identical.

use riir_infer_gpu::qwen38_cq3::{Codebook, CqTensor, Rotation, Xorshift64Star};

const D_OUT: usize = 8;
const D_IN: usize = 256;
const W_SEED: u64 = 0x1186_F17E_0000_0001;
const X_SEED: u64 = 0x1186_F17E_0000_0002;

/// Seeded random weight matrix (row-major).
fn seeded_weights(seed: u64, len: usize) -> Vec<f32> {
    let mut rng = Xorshift64Star::new(seed);
    (0..len).map(|_| rng.next_normal() * 0.05).collect()
}

/// Seeded random activation vector.
fn seeded_vec(seed: u64, len: usize) -> Vec<f32> {
    let mut rng = Xorshift64Star::new(seed);
    (0..len).map(|_| rng.next_normal()).collect()
}

/// Plain sequential f32 matmul — the reference consumer of `dequant()`.
/// Sequential accumulation per row, matching the fused kernel's order
/// (Issue-940 determinism law: pinned order, no reassociation).
fn ref_matmul(w: &[f32], d_out: usize, d_in: usize, x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0_f32; d_out];
    for i in 0..d_out {
        let mut acc = 0.0_f32;
        for j in 0..d_in {
            acc += w[i * d_in + j] * x[j];
        }
        out[i] = acc;
    }
    out
}

fn rel_rms(a: &[f32], b: &[f32]) -> f64 {
    let mut se = 0.0_f64;
    let mut sy = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = f64::from(x) - f64::from(y);
        se += d * d;
        sy += f64::from(y) * f64::from(y);
    }
    (se.sqrt()) / (sy.sqrt())
}

/// The G1 core: fused LUT GEMV is bit-exact against dequant + plain matmul.
fn assert_bit_exact(d_out: usize, d_in: usize, g: usize, b: u32, smooth: Option<&[f32]>) {
    let w = seeded_weights(W_SEED + g as u64, d_out * d_in);
    let t = CqTensor::quantize(&w, d_out, d_in, g, b, smooth);
    let x = seeded_vec(X_SEED + g as u64, d_in);
    let x_rot = t.rotate_activation(&x);
    let w_ref = t.dequant();
    let want = ref_matmul(&w_ref, d_out, d_in, &x_rot);
    let got = t.gemv_f32_lut(&x_rot);
    assert_eq!(want.len(), got.len());
    for (i, (&want_row, &got_row)) in want.iter().zip(got.iter()).enumerate() {
        assert_eq!(
            want_row.to_bits(),
            got_row.to_bits(),
            "g1a: row {i} diverged from the reference dequant path"
        );
    }
}

#[test]
fn g1a_lut_gemv_bit_exact_vs_reference_dequant() {
    assert_bit_exact(D_OUT, D_IN, 128, 3, None);
    assert_bit_exact(D_OUT, D_IN, 16, 3, None);
    assert_bit_exact(D_OUT, D_IN, 32, 4, None);
}

#[test]
fn g1a_smooth_fold_stays_bit_exact() {
    // The SmoothQuant fold must not break the identity: the same x' = x/s
    // feeds both paths, so the fused kernel still matches the reference
    // dequant bit-for-bit.
    let mut rng = Xorshift64Star::new(0x1186_F17E_0000_0003);
    let smooth: Vec<f32> = (0..D_IN).map(|_| rng.next_f32() * 2.0 + 0.25).collect();
    assert_bit_exact(D_OUT, D_IN, 128, 3, Some(&smooth));
    assert_bit_exact(D_OUT, D_IN, 16, 3, Some(&smooth));

    // Algebra neutrality: (W·diag(s))(x/s) == W·x within f32 rounding.
    let w = seeded_weights(W_SEED, D_OUT * D_IN);
    let x = seeded_vec(X_SEED, D_IN);
    let t_s = CqTensor::quantize(&w, D_OUT, D_IN, 128, 6, Some(&smooth));
    let t_0 = CqTensor::quantize(&w, D_OUT, D_IN, 128, 6, None);
    let y_ref = ref_matmul(&w, D_OUT, D_IN, &x);
    let y_s = t_s.gemv_f32_lut(&t_s.rotate_activation(&x));
    let y_0 = t_0.gemv_f32_lut(&t_0.rotate_activation(&x));
    // b = 6 keeps the quantization floor low for this ALGEBRA comparison;
    // the smooth arm and the unsmoothed control must sit within the same
    // bound of the f32 reference (measured ≈ 3e-2 on these shapes — the
    // fixture's small per-block norms amplify relative error by 1/‖block‖;
    // the GPTQ residual that damps this lives in T2). Pin 1e-1.
    let tol = 1.0e-1;
    assert!(
        rel_rms(&y_s, &y_ref) < tol,
        "smoothed GEMV drifted from the f32 reference: {}",
        rel_rms(&y_s, &y_ref)
    );
    assert!(
        rel_rms(&y_0, &y_ref) < tol,
        "unsmoothed control drifted from the f32 reference: {}",
        rel_rms(&y_0, &y_ref)
    );
}

#[test]
fn g1b_i8_sdot_bounded_error() {
    // Pinned from measurement (T1, m3, dev build, these seeds): the int8
    // arm sits at rel-RMS ≈ 2.0e-2 against the f32 LUT arm at g=128/b=3
    // (the fixture's small per-block norms amplify by ≈1.8; a serving-shape
    // lane will sit lower). Pin 5e-2 = 2.5x headroom. int8 is a second
    // lossy surface — bounded, never bit-exact.
    for (g, b) in [(128_usize, 3_u32), (16, 3), (64, 4)] {
        let w = seeded_weights(W_SEED + g as u64, D_OUT * D_IN);
        let t = CqTensor::quantize(&w, D_OUT, D_IN, g, b, None);
        let x = seeded_vec(X_SEED + g as u64, D_IN);
        let x_rot = t.rotate_activation(&x);
        let y_f32 = t.gemv_f32_lut(&x_rot);
        let y_i8 = t.gemv_i8_sdot(&x_rot);
        let r = rel_rms(&y_i8, &y_f32);
        assert!(
            r < 5.0e-2,
            "g1b: int8 arm rel-RMS {r} breached the 5e-2 pin (g={g}, b={b})"
        );
    }
}

#[test]
fn g1c_rotation_is_orthogonal() {
    for g in [16_usize, 128_usize] {
        let rot = Rotation::fixed(g);
        let mut rng = Xorshift64Star::new(0x1186_F17E_0000_0004);
        let x: Vec<f32> = (0..g).map(|_| rng.next_normal()).collect();
        let y: Vec<f32> = (0..g).map(|_| rng.next_normal()).collect();
        let (nx, ny): (f64, f64) = (
            x.iter()
                .map(|v| f64::from(*v) * f64::from(*v))
                .sum::<f64>()
                .sqrt(),
            y.iter()
                .map(|v| f64::from(*v) * f64::from(*v))
                .sum::<f64>()
                .sqrt(),
        );
        let dxy: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum();
        let mut out = Vec::with_capacity(g);
        rot.apply(&x, &mut out);
        let mut out2 = Vec::with_capacity(g);
        rot.apply(&y, &mut out2);
        let (mx, my): (f64, f64) = (
            out.iter()
                .map(|v| f64::from(*v) * f64::from(*v))
                .sum::<f64>()
                .sqrt(),
            out2.iter()
                .map(|v| f64::from(*v) * f64::from(*v))
                .sum::<f64>()
                .sqrt(),
        );
        let dxy_rot: f64 = out
            .iter()
            .zip(out2.iter())
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum();
        let tol = 1.0e-4_f64;
        assert!(
            ((mx - nx) / nx).abs() < tol && ((my - ny) / ny).abs() < tol,
            "g1c: norm drift at g={g}: {nx} → {mx}, {ny} → {my}"
        );
        let d_ref = dxy / (nx * ny);
        let d_got = dxy_rot / (mx * my);
        assert!(
            (d_ref - d_got).abs() < tol,
            "g1c: inner product drift at g={g}: {d_ref} → {d_got}"
        );
    }
}

#[test]
fn g1d_recipe_level_accuracy_no_residual() {
    // The NO-GPTQ floor (the residual pass is T2 — it consumes the
    // matched-calibration arm's samples). Measured (m3, dev build, these
    // seeds): b = 4 ≈ 1.5e-1 rel-RMS, b = 3 larger — the fixture's small
    // per-block norms (σ_w = 0.05 ⇒ ‖block‖ ≈ 0.57) amplify relative error
    // by 1/‖block‖ ≈ 1.8; the pin holds ~2x headroom on the measured value
    // and it is the floor the T2 residual must improve on.
    let w = seeded_weights(W_SEED, D_OUT * D_IN);
    let x = seeded_vec(X_SEED, D_IN);
    let y_ref = ref_matmul(&w, D_OUT, D_IN, &x);
    for (b, pin) in [(4_u32, 3.0e-1_f64), (3, 5.0e-1)] {
        let t = CqTensor::quantize(&w, D_OUT, D_IN, 128, b, None);
        let y = t.gemv_f32_lut(&t.rotate_activation(&x));
        let r = rel_rms(&y, &y_ref);
        assert!(
            r < pin,
            "g1d: b={b} rel-RMS {r} breached the {pin} no-residual pin"
        );
    }
}

#[test]
fn g1e_b_eff_byte_accounting() {
    for (g, b) in [(128_usize, 3_u32), (16, 3), (64, 4), (256, 2)] {
        let w = seeded_weights(W_SEED + g as u64, D_OUT * D_IN);
        let t = CqTensor::quantize(&w, D_OUT, D_IN, g, b, None);
        let want = f64::from(b) + 16.0 / g as f64;
        let got = t.bits_per_weight();
        assert!(
            (got - want).abs() < 1.0e-9,
            "g1e: b_eff {got} != b + 16/g {want} (g={g}, b={b})"
        );
    }
    // The paper's headline shape explicitly: g = 128 ⇒ b + 0.125.
    let t = CqTensor::quantize(
        &seeded_weights(W_SEED, D_OUT * D_IN),
        D_OUT,
        D_IN,
        128,
        3,
        None,
    );
    assert!((t.bits_per_weight() - 3.125).abs() < 1.0e-9);
}

#[test]
fn g1f_determinism_same_inputs_bit_identical() {
    let w = seeded_weights(W_SEED, D_OUT * D_IN);
    let t1 = CqTensor::quantize(&w, D_OUT, D_IN, 128, 3, None);
    let t2 = CqTensor::quantize(&w, D_OUT, D_IN, 128, 3, None);
    assert_eq!(t1.indices, t2.indices, "g1f: indices diverged");
    assert_eq!(t1.scales, t2.scales, "g1f: scales diverged");
    // Bit pin on the codebook: f32 element equality via to_bits (deterministic
    // builds never produce NaN, so element equality IS the bit pin here).
    assert!(
        t1.codebook.centroids.len() == t2.codebook.centroids.len()
            && t1
                .codebook
                .centroids
                .iter()
                .zip(t2.codebook.centroids.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits()),
        "g1f: codebook diverged"
    );
    // Rotation re-derivation.
    let r1 = Rotation::fixed(128);
    let r2 = Rotation::fixed(128);
    let mut o1 = Vec::new();
    let mut o2 = Vec::new();
    let x = seeded_vec(X_SEED, D_IN);
    r1.apply(&x, &mut o1);
    r2.apply(&x, &mut o2);
    assert_eq!(o1, o2, "g1f: rotation re-derivation diverged");
    // Codebook re-build directly.
    let c1 = Codebook::build(128, 3);
    let c2 = Codebook::build(128, 3);
    assert_eq!(c1.centroids, c2.centroids);
    assert_eq!(c1.c8, c2.c8);
}
