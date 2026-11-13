//! DI-SwiGLU — Algorithm 3, without FSBR smoothing (`alpha_smooth ≡ 1`),
//! with one deliberate upstream fix: the sigmoid is computed with the
//! numerically-stable two-branch form instead of the paper's shared
//! row-max frame.
//!
//! Decision-content transcription of detllm `ops/di_swiglu.py:1-30,35-65`
//! @ `9c3f1c9c` (NO LICENSE upstream — decision-content-only; overview
//! `.research/027`). The paper computes `sigma(x_i) = e^(x_i-M) /
//! (e^(x_i-M) + e^(-M))` with `M` the row max; when `M` is large (their
//! un-smoothed gates reach |x| ≈ 8+) BOTH exponentials underflow the
//! DI-Exp output resolution (~1/tpos ≈ 2^-8) and sigma collapses to 0
//! for moderately positive inputs where the true sigma ≈ 0.9. The stable
//! elementwise form
//!
//! ```text
//! sigma(x) = tpos / (tpos + E)   if x >= 0     E = DI-Exp(-|x|)·scale
//! sigma(x) = E / (tpos + E)      if x < 0
//! ```
//!
//! has no row coupling, uses the same DI-Exp machinery, and its argument
//! `-|x|` never underflows where sigma has curvature. sigma is quantized
//! to `[0, 128]` (7 fractional bits) and multiplied by both operands:
//! `y = x_gate · sigma · x_up` with dyadic bookkeeping
//! `(m_out = m_gate·m_up, k_out = k_gate + k_up + SIG_BITS - 1)`.
//!
//! Caller contract (upstream): gate/up each carry a common per-row
//! dyadic scale and have been requantized to ≤ ~12-bit magnitudes first
//! (the model layer's `requant_i32_common(target_bits=14)`) so
//! `gate·sigma·up` stays comfortably in int64.

use super::super::num::round_half_away_div;
use super::di_exp::di_exp_row;

/// Sigmoid fractional bits (`di_swiglu.py:35`).
pub const SIG_BITS: u32 = 8;
/// Fixed-point 1.0 of the sigmoid (`di_swiglu.py:36`) = 128.
pub const SIG_ONE: i64 = 1 << (SIG_BITS - 1);

/// One row of DI-SwiGLU. Returns `(y, m_out, k_out)`:
/// `value ≈ gate·sigma(gate)·up · m_out / 2^k_out` elementwise.
pub fn di_swiglu_row(
    gate: &[i64],
    up: &[i64],
    gm: i64,
    gk: u32,
    um: i64,
    uk: u32,
) -> (Vec<i64>, i64, u32) {
    assert_eq!(gate.len(), up.len(), "di_swiglu: gate/up length");
    // E = e^(-|x|)·tpos, per-row tpos (the two-branch stable form)
    let neg_abs: Vec<i64> = gate.iter().map(|g| -g.abs()).collect();
    let (e, tpos) = di_exp_row(&neg_abs, gm, gk);
    let mut y = Vec::with_capacity(gate.len());
    for (i, &g) in gate.iter().enumerate() {
        let denom = (tpos + e[i]).max(1);
        let num = if g >= 0 { tpos } else { e[i] };
        let sig = round_half_away_div(
            num.checked_shl(SIG_BITS - 1)
                .expect("di_swiglu: sigma numerator overflow"),
            denom,
        );
        y.push(
            g.checked_mul(sig)
                .expect("di_swiglu: gate·sigma overflow")
                .checked_mul(up[i])
                .expect("di_swiglu: ·up overflow"),
        );
    }
    let m_out = gm.checked_mul(um).expect("di_swiglu: m_out overflow");
    let k_out = gk + uk + (SIG_BITS - 1);
    (y, m_out, k_out)
}

/// The batched form: `gate`/`up` row-major `[rows, cols]` with per-row
/// scales. Returns `(y row-major int64, m_out per-row, k_out per-row)`.
pub fn di_swiglu(
    gate: crate::laya::riir::int_ref::tensor::Mat<'_, i64>,
    up: crate::laya::riir::int_ref::tensor::Mat<'_, i64>,
    gm: &[i64],
    gk: &[u32],
    um: &[i64],
    uk: &[u32],
) -> (Vec<i64>, Vec<i64>, Vec<u32>) {
    assert_eq!(gate.rows, up.rows, "di_swiglu: row mismatch");
    assert_eq!(gate.cols, up.cols, "di_swiglu: col mismatch");
    assert_eq!(gm.len(), gate.rows, "di_swiglu: gm length");
    assert_eq!(gk.len(), gate.rows, "di_swiglu: gk length");
    assert_eq!(um.len(), gate.rows, "di_swiglu: um length");
    assert_eq!(uk.len(), gate.rows, "di_swiglu: uk length");
    let mut y = Vec::with_capacity(gate.len());
    let mut m_out = Vec::with_capacity(gate.rows);
    let mut k_out = Vec::with_capacity(gate.rows);
    for r in 0..gate.rows {
        let (yr, mo, ko) = di_swiglu_row(gate.row(r), up.row(r), gm[r], gk[r], um[r], uk[r]);
        y.extend_from_slice(&yr);
        m_out.push(mo);
        k_out.push(ko);
    }
    (y, m_out, k_out)
}

#[cfg(test)]
mod goldens {
    use super::*;

    #[test]
    fn hand_golden_zero_gate() {
        // gate = 0: sigma(0) = tpos/(tpos + tpos) = 1/2 → sig = 64.
        // y = 0 · 64 · up = 0.
        let (y, m_out, k_out) = di_swiglu_row(&[0], &[7], 1, 4, 1, 4);
        assert_eq!(y, vec![0]);
        assert_eq!((m_out, k_out), (1, 4 + 4 + (SIG_BITS - 1)));
    }

    #[test]
    fn hand_golden_large_positive_gate_saturates() {
        // the upstream NOTES fix's headline defect: a large POSITIVE gate
        // must NOT collapse to 0. gate = +40·(1/16) = 2.5 (m=1, k=4):
        // DI-Exp(-40): q = floor(40/16) = 2, r = -8, unshifted = (-8>>1)
        // + 16 = 12, E = 12 >> 2 = 3; tpos = 16.
        // sigma = round(16·128/(16+3)) = round(2048/19) = 108.
        // y = 40 · 108 · 3 = 12960. (True sigma(2.5) ≈ 0.924 → sig 118;
        // the DI-Exp resolution is the gap — integer and deterministic.)
        let (y, _m, _k) = di_swiglu_row(&[40], &[3], 1, 4, 1, 4);
        assert_eq!(y[0], 12960);
    }

    #[test]
    fn hand_golden_negative_gate() {
        // gate = -16 (m=1, k=4 → true value -1.0): E = DI-Exp(-16):
        // q = 1, r = 0, unshifted = 16, result = 8; tpos = 16.
        // x < 0 branch: sigma = round(8·128/24) = round(42.67) = 43.
        // y = -16 · 43 · 5 = -3440.
        let (y, _m, _k) = di_swiglu_row(&[-16], &[5], 1, 4, 1, 4);
        assert_eq!(y[0], -3440);
    }

    #[test]
    fn sigmoid_saturates_to_the_rails() {
        // |gate| huge: E underflows to 0 → sigma = 128 (x ≥ 0) or 0 (x < 0)
        let (y_pos, _, _) = di_swiglu_row(&[1 << 30], &[9], 1, 4, 1, 4);
        assert_eq!(y_pos[0], (1_i64 << 30) * SIG_ONE * 9);
        let (y_neg, _, _) = di_swiglu_row(&[-(1 << 30)], &[9], 1, 4, 1, 4);
        assert_eq!(y_neg[0], 0);
    }

    #[test]
    fn sigma_symmetry_bracket() {
        // sigma(x) + sigma(-x) ≈ SIG_ONE within 1 (the two-branch form's
        // complementary identity, up to the two roundings). y = g·sig·1,
        // so sig = y/g exactly.
        for g in [1_i64, 4, 9, 16, 40, 64] {
            let (yp, _, _) = di_swiglu_row(&[g], &[1], 1, 4, 1, 4);
            let (yn, _, _) = di_swiglu_row(&[-g], &[1], 1, 4, 1, 4);
            let sig_p = yp[0] / g;
            let sig_n = -yn[0] / g;
            assert!(
                (sig_p + sig_n - SIG_ONE).abs() <= 1,
                "g={g}: {sig_p} + {sig_n} off {SIG_ONE}"
            );
        }
    }

    #[test]
    fn batched_bookkeeping() {
        // m_out = gm·um; k_out = gk + uk + SIG_BITS - 1, per row
        let gate = vec![1_i64, 2, 3, 4];
        let up = vec![1_i64, 1, 1, 1];
        let gmat = crate::laya::riir::int_ref::tensor::Mat::from((gate.as_slice(), 2, 2));
        let umat = crate::laya::riir::int_ref::tensor::Mat::from((up.as_slice(), 2, 2));
        let (y, m_out, k_out) = di_swiglu(gmat, umat, &[3, 5], &[4, 6], &[7, 11], &[8, 2]);
        assert_eq!(m_out, vec![21, 55]);
        assert_eq!(k_out, vec![4 + 8 + 7, 6 + 2 + 7]);
        assert_eq!(y.len(), 4);
    }
}
