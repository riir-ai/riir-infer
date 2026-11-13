//! DI-RMSNorm — Algorithm 4, adapted to a per-token-scaled input.
//!
//! Decision-content transcription of detllm `ops/di_rmsnorm.py:29-63` @
//! `9c3f1c9c` (NO LICENSE upstream — decision-content-only; overview
//! `.research/027`). Norm inputs carry a single per-token dyadic scale,
// so the scale cancels in `x/rms(x)` entirely and the core is scale-free:
//!
//! ```text
//! y ≈ x / sqrt(mean(x^2))    (integer, via the bitwise isqrt)
//! ```
//!
//! γ handling (upstream docs/NOTES.md): hidden-dim norms have γ folded
//! into the following linear's weights offline, so [`di_rmsnorm`]
//! returns the bare normalized value at fixed scale `(1,
//! OUT_FRAC_BITS)`; the QK-Norm variant [`di_rmsnorm_gamma`] multiplies
//! a per-channel dyadic γ into the data at runtime — the row scale
//! stays uniform because the data absorbs γ. eps is dropped (rms
//! clamped ≥ 1; upstream NOTES.md §7).
//!
//! Overflow discipline (upstream module doc): rows are pre-shifted so
//! `max|x'|` has MSB ≤ bit [`TARGET_HI`], hence `Σx'^2 ≤ n·2^52 ≤ 2^62`
//! for `n ≤ 1024` — every widening op below is still `checked_*` and
//! refuses loud. Up-shifts are exact; down-shifts round; the pre-shift
//! cancels exactly (numerator and rms use the same `x'`).

use super::super::num::{ilog2_floor, isqrt, round_half_away_div, rshift_round};

/// Output scale (`di_rmsnorm.py:29`): fixed `(m=1, k=OUT_FRAC_BITS)`;
/// `|y| ≤ sqrt(n)·2^OUT_FRAC_BITS ≤ 2^25` for `n ≤ 1024`.
pub const OUT_FRAC_BITS: u32 = 20;
/// Pre-shift target (`di_rmsnorm.py:30`): `max|x'|` MSB lands at bit 25.
const TARGET_HI: i64 = 25;

/// Scale-free integer RMSNorm: `x` row-major `[rows, n]` with any
/// per-token scale (it cancels). Returns int32 `[rows, n]` at fixed
/// scale `(1, OUT_FRAC_BITS)`; all-zero rows return zeros.
pub fn di_rmsnorm(x: crate::laya::riir::int_ref::tensor::Mat<'_, i64>) -> Vec<i32> {
    let mut y = Vec::with_capacity(x.len());
    for r in 0..x.rows {
        let row = x.row(r);
        let n = row.len() as i64;
        let g = row.iter().map(|v| v.abs()).max().unwrap_or(0);
        let lg = ilog2_floor(g.max(1) as u64) as i64;
        let sh = TARGET_HI - lg; // > 0: exact up-shift; < 0: rounded down-shift
        let up = sh.max(0) as u32;
        let down = (-sh).max(0) as u32;
        let mut xp = Vec::with_capacity(row.len());
        let mut ssum: i64 = 0;
        for &v in row {
            let shifted = rshift_round(v, down)
                .checked_shl(up)
                .expect("di_rmsnorm: pre-shift overflow");
            ssum = ssum
                .checked_add(
                    shifted
                        .checked_mul(shifted)
                        .expect("di_rmsnorm: x'^2 overflow"),
                )
                .expect("di_rmsnorm: sum-of-squares overflow");
            xp.push(shifted);
        }
        let mean = round_half_away_div(ssum, n);
        let rms = isqrt(mean as u64).max(1);
        for &v in &xp {
            let num = v
                .checked_shl(OUT_FRAC_BITS)
                .expect("di_rmsnorm: numerator overflow");
            let yv = round_half_away_div(num, rms as i64);
            y.push(i32::try_from(yv).expect("di_rmsnorm: int32 overflow"));
        }
    }
    y
}

/// QK-Norm variant (`di_rmsnorm.py:55`): `(x/rms)·γ` with γ given as
/// signed per-channel dyadic mantissas `gamma_m` / `2^gamma_k` over the
/// reduction dim (head_dim in the model). Returns `(y, k_out)` with
/// `y ≈ (x/rms)·γ_c / 2^k_out` at uniform row scale
/// `(1, k_out = OUT_FRAC_BITS + gamma_k)`; `|y| ≤ 2^25 · max|gamma_m|`
/// — the caller rescales/quantizes next.
pub fn di_rmsnorm_gamma(
    x: crate::laya::riir::int_ref::tensor::Mat<'_, i64>,
    gamma_m: &[i64],
    gamma_k: u32,
) -> (Vec<i64>, u32) {
    assert_eq!(gamma_m.len(), x.cols, "di_rmsnorm_gamma: gamma length");
    let bare = di_rmsnorm(x);
    let mut y = Vec::with_capacity(bare.len());
    for (i, &v) in bare.iter().enumerate() {
        y.push(
            (v as i64)
                .checked_mul(gamma_m[i % x.cols])
                .expect("di_rmsnorm_gamma: y·gamma overflow"),
        );
    }
    let k_out = OUT_FRAC_BITS + gamma_k;
    (y, k_out)
}

#[cfg(test)]
mod goldens {
    use super::*;
    use crate::laya::riir::int_ref::tensor::Mat;

    #[test]
    fn hand_golden_small_row() {
        // Hand-computed EXACT chain for row [3, 4] (n = 2):
        // g = 4, lg = 2, sh = 23 (exact up-shift), xp = [3·2^23, 4·2^23].
        // ssum = 25·2^46; mean = 25·2^45 (exact halving);
        // rms = isqrt(25·2^45) = 29658208
        //   (29658208^2 = 879609273535264 <= 879609302220800 < 29658209^2);
        // y0 = round(3·2^43 / 29658208): 889746·rms = 26388271935168,
        //   remainder 7131456 < rms/2 -> down -> 889746;
        // y1 = round(4·2^43 / 29658208): 1186328·rms = 35184362580224,
        //   remainder 9508608 < rms/2 -> down -> 1186328.
        // (Real-value check: x/rms(x) = [0.84853, 1.13137]·2^20 —
        // [889746, 1186328] are those rounded through the pinned laws.)
        let mat = Mat::from((&[3_i64, 4][..], 1, 2));
        let y = di_rmsnorm(mat);
        assert_eq!(y, vec![889746, 1186328]);
    }

    #[test]
    fn unit_row_maps_to_prob_one() {
        // n = 1: rms(x) = |x| exactly, so y = x·2^20/|x| = 2^20; and a
        // CONSTANT row of any width n has rms = the constant -> y = 2^20
        let one = Mat::from((&[7_i64][..], 1, 1));
        assert_eq!(di_rmsnorm(one), vec![1 << 20]);
        let row = [5_i64; 512];
        let many = Mat::from((row.as_slice(), 1, 512));
        assert_eq!(di_rmsnorm(many), vec![1 << 20; 512]);
    }

    #[test]
    fn scale_invariance_exact() {
        // upstream test_ops.py::TestDiRmsNorm::test_scale_invariance_exact:
        // same values at different (power-of-two) magnitudes give
        // IDENTICAL output — the pre-shift makes the core scale-free
        let mut x = Vec::new();
        for i in 0..(4 * 128) {
            let v = ((i * 2654435761_u64) % (1 << 20)) as i64;
            x.push(v - (1 << 19));
        }
        let m1 = Mat::from((x.as_slice(), 4, 128));
        let y1 = di_rmsnorm(m1);
        let x2: Vec<i64> = x.iter().map(|v| v << 4).collect();
        let m2 = Mat::from((x2.as_slice(), 4, 128));
        let y2 = di_rmsnorm(m2);
        assert_eq!(y1, y2);
    }

    #[test]
    fn zero_row_is_zero() {
        // upstream test_ops.py::TestDiRmsNorm::test_zero_row
        let z = [0_i64; 256];
        let mat = Mat::from((z.as_slice(), 2, 128));
        let y = di_rmsnorm(mat);
        assert_eq!(y, vec![0_i32; 256]);
    }

    #[test]
    fn gamma_variant_binds_scale() {
        // di_rmsnorm_gamma: y = bare·gamma_m at k_out = OUT_FRAC_BITS + gamma_k
        let mat = Mat::from((&[3_i64, 4, 3, 4][..], 2, 2));
        let gamma = [1000_i64, 1000];
        let (y, k_out) = di_rmsnorm_gamma(mat, &gamma, 10);
        let bare = di_rmsnorm(mat);
        assert_eq!(k_out, OUT_FRAC_BITS + 10);
        for (i, v) in y.iter().enumerate() {
            assert_eq!(*v, bare[i] as i64 * 1000);
        }
    }

    #[test]
    fn n_le_1024_overflow_discipline() {
        // the module's overflow note: n <= 1024 keeps sum(x'^2) <= 2^62;
        // a 1024-wide row of MSB-25 values must not refuse (and a wider
        // row of those WOULD — the loud check fires via checked_add).
        // A constant row has rms = the constant, ratio 1 -> y = 2^20.
        let row: Vec<i64> = vec![1_i64 << 25; 1024];
        let mat = Mat::from((row.as_slice(), 1, 1024));
        let y = di_rmsnorm(mat);
        for v in &y {
            assert_eq!(*v, 1 << 20);
        }
    }
}
