//! DI-Exp — Algorithm 1: integer exponential via shift decomposition.
//!
//! Decision-content transcription of detllm `ops/di_exp.py:21-39` @
//! `9c3f1c9c` (NO LICENSE upstream — decision-content-only; overview
//! `.research/027`). Given `x ≤ 0` (already max-subtracted) with
//! per-row dyadic scale `m/2^k`, computes `result ≈ e^(x·m/2^k) · tpos`
//! where `tpos = round(2^k / m_f)` and
//! `m_f = m + (m >> 1) - (m >> 4)` — the paper's constant (`≈ 1.4375·m
//! ≈ m·log2(e)`; log2(e) = 1.4427, so the approximation is 0.36% low —
//! kept verbatim per spec). The implicit output scale `1/tpos` is NOT
//! dyadic; it is always consumed by a ratio (softmax / sigmoid), where
//! it cancels. Returns `(result, tpos)` with `result ∈ [0, tpos]`.

use super::super::num::round_half_away_div;

/// One row of DI-Exp: `x` is the row slice (all values ≤ 0), `m`/`k`
/// the row's dyadic scale (`k ≤ 62` — `2^k` must stay in int64).
/// Returns `(result, tpos)`; `tpos >= 1`.
pub fn di_exp_row(x: &[i64], m: i64, k: u32) -> (Vec<i64>, i64) {
    assert!(k <= 62, "di_exp: k must stay <= 62");
    // m_f = m + (m >> 1) - (m >> 4) — the paper's constant, verbatim
    let m_f = m + (m >> 1) - (m >> 4);
    let tpos = round_half_away_div(1_i64.checked_shl(k).expect("di_exp: 2^k overflow"), m_f).max(1);
    let t = -tpos;
    let mut result = Vec::with_capacity(x.len());
    for &xi in x {
        assert!(xi <= 0, "di_exp: input must be max-subtracted (<= 0)");
        // q = floor(x / t) — Python floor semantics (upstream floor_div);
        // x <= 0 and t < 0 make the quotient non-negative, where floor
        // and trunc coincide (the fused kernel's positive-domain rewrite
        // rests on the same fact).
        let q = floor_div(xi, t);
        let r = xi - q * t; // in (t, 0]
        let unshifted = (r >> 1) - t; // in (tpos/2, tpos]
        let qc = q.min(62);
        let v = unshifted
            .checked_shr(u32::try_from(qc).expect("di_exp: q out of range"))
            .expect("di_exp: shift overflow");
        result.push(v);
    }
    (result, tpos)
}

/// The batched form: `x` is row-major `[rows, cols]`, `m`/`k` per-row
/// (`len == rows`). Returns `(result, tpos)` — `result` row-major
/// `[rows, cols]`, `tpos` per-row.
pub fn di_exp(
    x: crate::laya::riir::int_ref::tensor::Mat<'_, i64>,
    m: &[i64],
    k: &[u32],
) -> (Vec<i64>, Vec<i64>) {
    assert_eq!(m.len(), x.rows, "di_exp: m length");
    assert_eq!(k.len(), x.rows, "di_exp: k length");
    let mut result = Vec::with_capacity(x.len());
    let mut tpos = Vec::with_capacity(x.rows);
    for r in 0..x.rows {
        let (res, tp) = di_exp_row(x.row(r), m[r], k[r]);
        result.extend_from_slice(&res);
        tpos.push(tp);
    }
    (result, tpos)
}

/// Python floor division (`intmath.floor_div`): rounds toward −inf for
/// either sign of the divisor. Upstream refuses `b == 0`.
#[inline]
pub fn floor_div(a: i64, b: i64) -> i64 {
    assert!(b != 0, "floor_div: zero divisor");
    let q = a / b; // trunc toward zero
    let r = a % b;
    if r != 0 && ((r < 0) != (b < 0)) {
        q - 1
    } else {
        q
    }
}

#[cfg(test)]
mod goldens {
    use super::*;

    #[test]
    fn zero_maps_to_one() {
        // upstream test_ops.py::TestDiExp::test_zero_maps_to_one:
        // e^0 = 1.0 exactly in this encoding — result == tpos
        let (res, tpos) = di_exp_row(&[0, 0, 0], 1, 4);
        assert_eq!(res, vec![tpos, tpos, tpos]);
        assert!(tpos >= 1);
    }

    #[test]
    fn hand_golden_half_row() {
        // Hand-computed: m=1, k=1 (scale 1/2), tpos = round(2^1/1) = 2.
        // x = -1: q = floor(-1/-2) = 0, r = -1, unshifted = (-1>>1) + 2 = 1,
        //         result = 1        (e^-0.5·2 = 1.21)
        // x = -2: q = 1, r = 0, unshifted = 2, result = 2 >> 1 = 1
        //         (e^-1·2 = 0.74)
        // x = -3: q = 1, r = -1, unshifted = (-1>>1) + 2 = 1,
        //         result = 1 >> 1 = 0 (e^-1.5·2 = 0.45)
        let (res, tpos) = di_exp_row(&[-1, -2, -3], 1, 1);
        assert_eq!(tpos, 2);
        assert_eq!(res, vec![1, 1, 0]);
    }

    #[test]
    fn hand_golden_k4() {
        // m=1, k=4: tpos = 16. x values are scores·(1/16).
        // x = 0  -> 16 (exact one)
        // x = -16 -> q=1, r=0, unshifted=16, result = 8  (e^-1·16 = 5.9)
        // x = -32 -> q=2, r=0, unshifted=16, result = 4  (e^-2·16 = 2.2)
        // x = -64 -> q=4, r=0, unshifted=16, result = 1  (e^-4·16 = 0.29)
        // x = -80 -> q=5, r=0, unshifted=16, result = 0  (e^-5·16 = 0.11)
        let (res, tpos) = di_exp_row(&[0, -16, -32, -64, -80], 1, 4);
        assert_eq!(tpos, 16);
        assert_eq!(res, vec![16, 8, 4, 1, 0]);
    }

    #[test]
    fn tail_underflows_to_zero() {
        // upstream test_ops.py::TestDiExp::test_tail_underflows_to_zero —
        // deeply negative inputs decay to exactly 0 (DI-Exp's own
        // underflow replaces Eq. 10 clipping). At m=1, k=20 the scale is
        // 2^-20: x = -(1<<30) is a true value of -2^10, x = -(1<<40) is
        // -2^20 — both far past e^-21 where the encoding underflows.
        // Hand-check x = -(1<<30): q = 2^10, r = 0, unshifted = 2^20,
        // result = 2^20 >> min(2^10, 62) = 0.
        let (res, _tpos) = di_exp_row(&[-(1 << 30), -(1 << 40)], 1, 20);
        assert_eq!(res, vec![0, 0]);
    }

    #[test]
    fn result_bounded_by_tpos() {
        // the doc contract: result ∈ [0, tpos] for a spread of inputs
        for k in [1_u32, 4, 8, 16] {
            let xs: Vec<i64> = (0..64).map(|i| -i * 7).collect();
            let (res, tpos) = di_exp_row(&xs, 97, k);
            assert_eq!(res[0], tpos, "x=0 must map to tpos");
            for v in &res {
                assert!((0..=tpos).contains(v), "result {v} out of [0, {tpos}]");
            }
            // monotone non-increasing in x
            for w in res.windows(2) {
                assert!(w[0] >= w[1], "di_exp not monotone at k={k}");
            }
        }
    }

    #[test]
    fn floor_div_is_python_semantics() {
        // rounds toward -inf with either divisor sign
        assert_eq!(floor_div(-7, 2), -4);
        assert_eq!(floor_div(7, 2), 3);
        assert_eq!(floor_div(-8, 2), -4);
        assert_eq!(floor_div(7, -2), -4);
        assert_eq!(floor_div(-7, -2), 3);
        assert_eq!(floor_div(-1, -2), 0); // floor(0.5) = 0 (the di_exp path)
        assert_eq!(floor_div(-3, -2), 1); // floor(1.5) = 1
    }

    #[test]
    #[should_panic(expected = "zero divisor")]
    fn floor_div_refuses_zero() {
        let _ = floor_div(1, 0);
    }

    #[test]
    #[should_panic(expected = "<= 0")]
    fn di_exp_refuses_positive_input() {
        let _ = di_exp_row(&[1], 1, 4);
    }
}
