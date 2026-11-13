//! DI-Softmax — Algorithm 2.
//!
//! Decision-content transcription of detllm `ops/di_softmax.py:8-14,28-30,33-58`
//! @ `9c3f1c9c` (NO LICENSE upstream — decision-content-only; overview
//! `.research/027`). Integer scores with a per-row dyadic scale and a
//! validity mask (causal + padding); masked positions are excluded from
//! the row max and the sum via an INTEGER sentinel — no float −inf
//! anywhere.
//!
//! Output: probabilities in `[0, 2^(p_out_bits-1)]` (stored int32) with
//! the fixed dyadic scale `(m=1, k=p_out_bits-1)` — upstream's default
//! `P_OUT_BITS = 15` (their §9.2 ablation: 7-bit probabilities cost ~2
//! PPL once scores are accurate; the paper's 8-bit default is not
//! ported). Probs are consumed by the exact hi/lo int8 split GEMM
//! ([`super::matmul::int_gemm_u16i8`]).
//!
//! Deliberately NOT ported: upstream's optional Eq. 10 clipping knob
//! (`clip_c`) — an ablation arm; their NOTES.md §4 shows DI-Exp's own
//! underflow below ≈ −21 is strictly more accurate than clipping.

use super::super::num::round_half_away_div;
use super::di_exp::di_exp_row;

/// Output precision (`di_softmax.py:29`).
pub const P_OUT_BITS: u32 = 15;
/// Fixed-point 1.0 of the output (`di_softmax.py:30`).
pub const PROB_ONE: i64 = 1 << (P_OUT_BITS - 1);

/// The integer sentinel standing in for float −inf (`i64::MIN + 1`,
/// leaving one unit of headroom below the sentinel for the max loop).
const NEG_SENTINEL: i64 = i64::MIN + 1;

/// One row of DI-Softmax: `scores` the row slice, `(m, k)` its dyadic
/// scale, `valid` the slot mask. Returns probs in `[0, 2^(p_out_bits-1)]`
/// at fixed scale `(1, p_out_bits-1)`; a row with NO valid slot returns
/// all zeros.
pub fn di_softmax_row(scores: &[i64], m: i64, k: u32, valid: &[bool], p_out_bits: u32) -> Vec<i32> {
    assert!(
        (1..=31).contains(&p_out_bits),
        "di_softmax: p_out_bits out of range"
    );
    assert_eq!(scores.len(), valid.len(), "di_softmax: scores/valid length");
    // sentinel-masked row max (no float -inf anywhere)
    let mut row_max = NEG_SENTINEL;
    let mut any_valid = false;
    for (i, &v) in valid.iter().enumerate() {
        if v {
            any_valid = true;
            row_max = row_max.max(scores[i]);
        }
    }
    if !any_valid {
        row_max = 0;
    }
    // xd = x - row_max, masked entries dummied to 0
    let mut xd = Vec::with_capacity(scores.len());
    for (i, &v) in valid.iter().enumerate() {
        if v {
            xd.push(scores[i] - row_max);
        } else {
            xd.push(0);
        }
    }
    let (mut exps, _tpos) = di_exp_row(&xd, m, k);
    for (i, &v) in valid.iter().enumerate() {
        if !v {
            exps[i] = 0;
        }
    }
    let mut denom: i64 = 0;
    for &e in &exps {
        denom = denom
            .checked_add(e)
            .expect("di_softmax: denominator overflow");
    }
    denom = denom.max(1);
    let shift = p_out_bits - 1;
    exps.iter()
        .map(|&e| {
            let num = e
                .checked_shl(shift)
                .expect("di_softmax: exps << p_out overflow");
            let p = round_half_away_div(num, denom);
            i32::try_from(p).expect("di_softmax: prob out of int32")
        })
        .collect()
}

/// The batched form: `scores` row-major `[rows, t]`, `valid` the same
/// shape flattened. Returns probs row-major `[rows, t]` int32.
pub fn di_softmax(
    scores: crate::laya::riir::int_ref::tensor::Mat<'_, i64>,
    m: &[i64],
    k: &[u32],
    valid: &[bool],
    p_out_bits: u32,
) -> Vec<i32> {
    assert_eq!(m.len(), scores.rows, "di_softmax: m length");
    assert_eq!(k.len(), scores.rows, "di_softmax: k length");
    assert_eq!(valid.len(), scores.len(), "di_softmax: valid length");
    let mut out = Vec::with_capacity(scores.len());
    for r in 0..scores.rows {
        let row = scores.row(r);
        let vr = &valid[r * scores.cols..(r + 1) * scores.cols];
        out.extend_from_slice(&di_softmax_row(row, m[r], k[r], vr, p_out_bits));
    }
    out
}

#[cfg(test)]
mod goldens {
    use super::*;

    #[test]
    fn hand_golden_row_sum_exact() {
        // Hand-computed: scores [32, 16, 0], m=1, k=4 (true values
        // [2, 1, 0]·1... at scale 1/16), all valid, p_out 15.
        // tpos = 16; exponents: x-max = [0, -16, -32] -> q=0,1,2, r=0 ->
        // unshifted 16 each -> exps [16, 8, 4]; denom 28;
        // probs = round_half_away([16, 8, 4] << 14, 28)
        //       = [9362, 4681, 2341] (16384·16/28 = 9362.28 -> 9362;
        //         16384·8/28 = 4681.14 -> 4681; 16384·4/28 = 2340.57 -> 2341)
        // sum = 16384 = PROB_ONE exactly.
        let p = di_softmax_row(&[32, 16, 0], 1, 4, &[true, true, true], P_OUT_BITS);
        assert_eq!(p, vec![9362, 4681, 2341]);
        let sum: i64 = p.iter().map(|&v| v as i64).sum();
        assert_eq!(sum, PROB_ONE);
    }

    #[test]
    fn row_sum_property() {
        // upstream test_ops.py::TestDiSoftmax::test_row_sum_property:
        // every row's probs sum to PROB_ONE within ± t (rounding slack)
        let t = 6_usize;
        let mut scores: Vec<i64> = Vec::new();
        let mut valid = Vec::new();
        for i in 0..(4 * t) {
            scores.push(((i as i64 * 37) % 91 - 45) * 3);
            valid.push(i % 5 != 3);
        }
        let mat = crate::laya::riir::int_ref::tensor::Mat::from((scores.as_slice(), 4_usize, t));
        let m = [1, 3, 97, 1];
        let kk = [4, 6, 8, 12];
        let p = di_softmax(mat, &m, &kk, &valid, P_OUT_BITS);
        for r in 0..4 {
            let sum: i64 = p[r * t..(r + 1) * t].iter().map(|&v| v as i64).sum();
            assert!(
                (sum - PROB_ONE).abs() <= t as i64,
                "row {r} sum {sum} off PROB_ONE"
            );
        }
    }

    #[test]
    fn masked_positions_contribute_nothing() {
        // upstream: masked slots get exactly 0 and removing them from
        // the row changes nothing (the sentinel exclusion is exact)
        let scores = [10, -4, 25, -9, 2];
        let valid = [true, false, true, false, true];
        let p = di_softmax_row(&scores, 1, 4, &valid, P_OUT_BITS);
        assert_eq!(p[1], 0);
        assert_eq!(p[3], 0);
        // recompute over only the valid positions — identical probs
        let p2 = di_softmax_row(&[10, 25, 2], 1, 4, &[true, true, true], P_OUT_BITS);
        assert_eq!([p[0], p[2], p[4]], p2.as_slice());
    }

    #[test]
    fn all_masked_row_is_zero() {
        // upstream test_ops.py::TestDiSoftmax::test_all_masked_row_is_zero
        let p = di_softmax_row(&[5, 5, 5], 1, 8, &[false, false, false], P_OUT_BITS);
        assert_eq!(p, vec![0, 0, 0]);
    }

    #[test]
    fn deterministic_in_mask_encoding() {
        // the same mask content in a different row position set gives
        // the same values (a pure function of values + mask, never of
        // iteration order or a HashMap)
        let a = di_softmax_row(&[3, 8, 1], 1, 4, &[true, true, false], P_OUT_BITS);
        let b = di_softmax_row(&[3, 8, 9], 1, 4, &[true, true, false], P_OUT_BITS);
        assert_eq!(a[0], b[0]);
        assert_eq!(a[1], b[1]);
        assert_eq!(a[2], 0);
        assert_eq!(b[2], 0);
    }

    #[test]
    fn probs_monotone_in_scores() {
        // softmax monotonicity through the pinned laws: strictly higher
        // score → prob ≥ the lower one's (DI-Exp is monotone per its own
        // golden; normalization preserves order — the denominator is a
        // positive row constant)
        let scores = [-9_i64, -2, 0, 5, 31];
        let p = di_softmax_row(&scores, 5, 6, &[true; 5], P_OUT_BITS);
        for w in p.windows(2) {
            assert!(w[0] <= w[1], "probs not monotone: {p:?}");
        }
        let sum: i64 = p.iter().map(|&v| v as i64).sum();
        assert!((sum - PROB_ONE).abs() <= 5);
    }
}
