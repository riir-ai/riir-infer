//! Issue 1188 T1 gates — the Q8_0 candidate posture on the projection
//! containers (quantize round-trip, quant-resident matvec, constructors,
//! invariants).
//!
//! The ablation's measurement authority rests on these: the Q8_0 arm must
//! reconstruct the dequanted weights within quantization error, the fused
//! GEMV must match a dequant-reference within tolerance, and the
//! constructors must refuse misshapen inputs loud.

use bytemuck::Zeroable;
use riir_infer_core::deltanet::ternary_weights::{GateProjWeights, ProjWeights};
use riir_infer_core::quant::q8kv::{
    BlockQ8_0, Q8_BLOCK_SIZE, dequantize_row_q8_0, gemv_q8_0_row, quantize_row_q8_0,
};

/// Deterministic weight-shaped fixture values (no RNG — the determinism
/// law; the sin/cos mix gives both signs and a realistic dynamic range).
fn fixture_values(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 * 0.37).sin() * 0.04) + ((i as f32 * 0.11).cos() * 0.01))
        .collect()
}

#[test]
fn proj_q8_round_trip_stays_within_quantization_error() {
    let (rows, cols) = (8usize, 128usize);
    let dense = fixture_values(rows * cols);
    let proj = ProjWeights::from_dense_q8_0(&dense, rows, cols);
    assert!(proj.invariants_hold());

    let back = proj.dequant_to_dense();
    let mut max_err = 0.0f32;
    for (a, b) in dense.iter().zip(back.iter()) {
        max_err = max_err.max((a - b).abs());
    }
    // Per-element bound: |x| <= amax; |x - d*q| <= d/2 = amax/254.
    // The fixture's per-block amax is bounded by 0.05 → d <= 4e-4.
    assert!(
        max_err < 0.05 / 127.0 * 1.5 + 1e-6,
        "round-trip max err {max_err}"
    );
    assert_eq!(proj.rows(), rows);
    assert_eq!(proj.cols(), cols);
    // 34 bytes per 32 elements = 8.5 bpw.
    assert_eq!(
        proj.bytes(),
        rows * (cols / Q8_BLOCK_SIZE) * std::mem::size_of::<BlockQ8_0>()
    );
}

#[test]
fn proj_q8_matvec_matches_dense_reference() {
    let (rows, cols) = (300usize, 256usize); // ≥ 256 rows → the rayon arm
    let dense = fixture_values(rows * cols);
    let proj = ProjWeights::from_dense_q8_0(&dense, rows, cols);
    let x = fixture_values(cols)
        .iter()
        .map(|v| v * 10.0)
        .collect::<Vec<_>>();

    let mut y = vec![0.0f32; rows];
    proj.matvec_into(&mut y, &x);

    // Reference: the dequanted weights, plain sequential dot per row.
    let back = proj.dequant_to_dense();
    let mut worst_rel = 0.0f64;
    for r in 0..rows {
        let mut reference = 0.0f32;
        for i in 0..cols {
            reference += back[r * cols + i] * x[i];
        }
        let rel = ((y[r] - reference) / reference.abs().max(1e-9)).abs() as f64;
        worst_rel = worst_rel.max(rel);
    }
    // The fused form scales per BLOCK (one f32 fold per 32-block) vs the
    // reference's per-element scale — a last-ulp-class difference per
    // block, tolerance-bounded here, never a bits match.
    assert!(
        worst_rel < 1e-4,
        "q8 matvec worst rel {worst_rel} vs dequant reference"
    );

    // The dequant-to-f32 arm reproduces the same dense buffer (dequant_to_dense
    // on a Q8_0 arm is the round-trip read side, already asserted above).
    assert_eq!(back.len(), rows * cols);
}

#[test]
fn proj_q8_refuses_non_multiple_cols_loud() {
    let dense = fixture_values(8 * 100); // 100 % 32 != 0
    let result = std::panic::catch_unwind(|| ProjWeights::from_dense_q8_0(&dense, 8, 100));
    assert!(
        result.is_err(),
        "cols=100 must refuse (not a multiple of 32)"
    );
}

#[test]
fn gate_proj_q8_round_trip_and_matvec() {
    let (rows, cols) = (48usize, 5120usize); // the Bonsai-2 a/b shape
    let dense = fixture_values(rows * cols);
    let gate = GateProjWeights::from_dense_q8_0(&dense, rows, cols);
    assert!(gate.invariants_hold());

    let back = gate.dequant_to_dense();
    assert_eq!(back.len(), rows * cols);
    let mut max_err = 0.0f32;
    for (a, b) in dense.iter().zip(back.iter()) {
        max_err = max_err.max((a - b).abs());
    }
    assert!(
        max_err < 0.05 / 127.0 * 1.5 + 1e-6,
        "gate max err {max_err}"
    );

    let x = fixture_values(cols);
    let mut y = vec![0.0f32; rows];
    gate.matvec_into(&mut y, &x);
    let mut worst_rel = 0.0f64;
    for r in 0..rows {
        let mut reference = 0.0f32;
        for i in 0..cols {
            reference += back[r * cols + i] * x[i];
        }
        let rel = ((y[r] - reference) / reference.abs().max(1e-9)).abs() as f64;
        worst_rel = worst_rel.max(rel);
    }
    assert!(worst_rel < 1e-4, "gate matvec worst rel {worst_rel}");

    // A Q8_0 gate is neither ternary nor dense-f32 (the fused-input hook
    // and the cudarc batched lane both refuse/fall back by these).
    assert!(gate.as_ternary().is_none());
    assert!(gate.as_dense().is_none());
}

#[test]
fn gemv_row_is_exact_against_its_own_fold_order() {
    // The fused kernel's fold: per-block int8·x dot (4-lane), then ONE
    // f16-scale multiply per block, blocks ascending. A reference with the
    // SAME fold must match to_bits — this pins the fold order itself (the
    // determinism law), not just the tolerance.
    let (n_blocks, scale) = (7usize, 0.02f32);
    let mut blocks = vec![BlockQ8_0::zeroed(); n_blocks];
    let mut x = Vec::with_capacity(n_blocks * Q8_BLOCK_SIZE);
    for b in 0..n_blocks {
        for i in 0..Q8_BLOCK_SIZE {
            let v = ((b * Q8_BLOCK_SIZE + i) as f32 * 0.53).sin() * scale;
            x.push(v);
        }
        quantize_row_q8_0(
            &x[b * Q8_BLOCK_SIZE..(b + 1) * Q8_BLOCK_SIZE],
            &mut blocks[b..b + 1],
        );
    }
    let fused = gemv_q8_0_row(&blocks, &x);

    let mut reference = 0.0f32;
    for b in 0..n_blocks {
        let d = half::f16::from_bits(blocks[b].d);
        let mut s0 = 0.0f32;
        let mut s1 = 0.0f32;
        let mut s2 = 0.0f32;
        let mut s3 = 0.0f32;
        for c in 0..Q8_BLOCK_SIZE / 4 {
            let j = c * 4;
            s0 += blocks[b].qs[j] as f32 * x[b * Q8_BLOCK_SIZE + j];
            s1 += blocks[b].qs[j + 1] as f32 * x[b * Q8_BLOCK_SIZE + j + 1];
            s2 += blocks[b].qs[j + 2] as f32 * x[b * Q8_BLOCK_SIZE + j + 2];
            s3 += blocks[b].qs[j + 3] as f32 * x[b * Q8_BLOCK_SIZE + j + 3];
        }
        reference += f32::from(d) * ((s0 + s1) + (s2 + s3));
    }
    assert_eq!(fused.to_bits(), reference.to_bits(), "fold-order exactness");
}

#[test]
fn q8_constructors_are_deterministic() {
    // Same dense input → to_bits-identical blocks (no hidden RNG or
    // parallel-order dependence in the requant path).
    let (rows, cols) = (4usize, 64usize);
    let dense = fixture_values(rows * cols);
    let a = ProjWeights::from_dense_q8_0(&dense, rows, cols);
    let b = ProjWeights::from_dense_q8_0(&dense, rows, cols);
    match (a, b) {
        (ProjWeights::Q8_0(ba, ra, ca), ProjWeights::Q8_0(bb, rb, cb)) => {
            assert_eq!((ra, ca), (rb, cb));
            assert_eq!(ba.len(), bb.len());
            for (x, y) in ba.iter().zip(bb.iter()) {
                assert_eq!(x.d, y.d, "scale bits differ");
                assert_eq!(x.qs, y.qs, "quantized values differ");
            }
        }
        _ => panic!("constructors must produce the Q8_0 arm"),
    }
}
