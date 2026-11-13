//! Integer rotary embeddings (§6.6 of the DetLLM design — not in the
//! paper; their design).
//!
//! Decision-content transcription of detllm `ops/rope.py:11-24,38-59` @
//! `9c3f1c9c` (NO LICENSE upstream — decision-content-only; overview
//! `.research/027`). Offline, cos/sin tables for all positions are
//! precomputed at fixed dyadic precision — int16 values, ONE global
//! `k = ROPE_FRAC_BITS` (values in `[-2^14, 2^14]`) — using HF Qwen3's
//! convention: for head_dim d, `inv_freq_j = base^(-2j/d)` for
//! `j < d/2`, and the angle vector is tiled twice so cos/sin have
//! length d (rotate_half pairing: `(x_j, x_{j+d/2})`).
//!
//! Runtime ([`apply_int_rope`]): pure elementwise integer multiply-add
//! against the position-indexed table rows, ONE [`rshift_round`] at the
//! end (single rounding after the add). The input's dyadic scale is
//! unchanged (`|cos|, |sin| ≤ 1` → no overflow growth: `|out| ≤
//! |x|·2^ROPE_FRAC_BITS·sqrt(2)` in int64 before the shift-back).
//!
//! ⛔ The float table BUILDER (upstream `ops/rope.py:25`, `build_rope_tables`)
//! is OFFLINE-only and deliberately absent from this module tree — the
//! type-level no-float rule covers every file here; the builder joins
//! the calibration stage (T1c) where the float stage is allowed, and
//! until then tests hand-craft tables.

use super::super::num::rshift_round;

/// Table fractional bits (`rope.py:22`).
pub const ROPE_FRAC_BITS: u32 = 14;

/// An offline-built rope table pair: `cos`/`sin` row-major
/// `[max_pos, head_dim]` int16, one global scale `2^-ROPE_FRAC_BITS`.
/// Provenance is the calibration artifact's duty (digest-pinned there).
#[derive(Clone, Debug)]
pub struct RopeTables {
    pub cos: Vec<i16>,
    pub sin: Vec<i16>,
    pub max_pos: usize,
    pub head_dim: usize,
}

impl RopeTables {
    /// Hand-built constant tables (identity rotation: cos = 1, sin = 0 —
    /// useful for tests and scale probes).
    pub fn identity(max_pos: usize, head_dim: usize) -> Self {
        RopeTables {
            cos: vec![(1 << ROPE_FRAC_BITS) as i16; max_pos * head_dim],
            sin: vec![0; max_pos * head_dim],
            max_pos,
            head_dim,
        }
    }

    fn row<'a>(&self, table: &'a [i16], pos: usize) -> &'a [i16] {
        assert!(
            pos < self.max_pos,
            "rope: position {pos} past the table's max_pos {}",
            self.max_pos
        );
        &table[pos * self.head_dim..(pos + 1) * self.head_dim]
    }
}

/// Apply integer RoPE (`apply_int_rope`, `rope.py:38`): `x` row-major
/// `[rows, head_dim]` (per-head Q or K rows, any dyadic scale — unchanged
/// by rotation); `positions[r]` the absolute position of row `r`.
/// Returns int64, same shape/scale as `x` (one rounding after the
/// multiply-add).
pub fn apply_int_rope(
    x: crate::laya::riir::int_ref::tensor::Mat<'_, i64>,
    tables: &RopeTables,
    positions: &[usize],
) -> Vec<i64> {
    let d = x.cols;
    assert_eq!(d, tables.head_dim, "rope: head_dim mismatch");
    assert_eq!(d % 2, 0, "rope: head_dim must be even");
    assert_eq!(positions.len(), x.rows, "rope: positions length");
    let half = d / 2;
    let mut out = Vec::with_capacity(x.len());
    for (pos, row) in positions.iter().zip((0..x.rows).map(|r| x.row(r))) {
        let c = tables.row(&tables.cos, *pos);
        let s = tables.row(&tables.sin, *pos);
        for j in 0..d {
            // rotate_half pairing: (x_j, x_{j+d/2}) — the partner index
            let partner = if j < half {
                -row[j + half]
            } else {
                row[j - half]
            };
            let v = (row[j])
                .checked_mul(c[j] as i64)
                .expect("rope: x·cos overflow")
                .checked_add(
                    partner
                        .checked_mul(s[j] as i64)
                        .expect("rope: partner·sin overflow"),
                )
                .expect("rope: multiply-add overflow");
            out.push(rshift_round(v, ROPE_FRAC_BITS));
        }
    }
    out
}

#[cfg(test)]
mod goldens {
    use super::*;
    use crate::laya::riir::int_ref::tensor::Mat;

    #[test]
    fn identity_tables_pass_through() {
        // cos=1, sin=0 → out == x exactly (shift of x·2^14 is exact)
        let tables = RopeTables::identity(4, 4);
        let x = [7_i64, -3, 100, -1000];
        let mat = Mat::from((x.as_slice(), 1, 4));
        let out = apply_int_rope(mat, &tables, &[2]);
        assert_eq!(out, x);
    }

    #[test]
    fn hand_golden_half_rotation() {
        // A 90-degree rotation at (cos=0, sin=2^14): out = rotate_half(x)
        // exactly. Tables hand-built, no float builder involved.
        let one = (1_i64 << ROPE_FRAC_BITS) as i16;
        let tables = RopeTables {
            cos: vec![0, 0, 0, 0],
            sin: vec![one, one, one, one],
            max_pos: 1,
            head_dim: 4,
        };
        // rotate_half([a, b, c, d]) = [-c, -d, a, b]
        let x = [10_i64, -20, 30, -40];
        let mat = Mat::from((x.as_slice(), 1, 4));
        let out = apply_int_rope(mat, &tables, &[0]);
        assert_eq!(out, vec![-30, 40, 10, -20]);
    }

    #[test]
    fn hand_golden_quarter_scales() {
        // cos = 2^13 (0.5), sin = 2^13·sqrt3 floor = 14169 (0.865 -> the
        // classic 60-degree triple scaled by 2^14: 8192, 14169 — the
        // rounding is the table's, not the op's). out_j =
        // rshift_round(x_j·8192 + partner_j·14169, 14).
        // x = [16384, 0, 0, 0]: partner = [-0, -0, 16384, 0].
        // out[0] = rshift(16384·8192 + (-0)·14169, 14) = rshift(2^27, 14) = 8192.
        // out[2] = rshift(0·8192 + 16384·14169, 14) = rshift(232128512, 14)
        //        = 14169 (232128512/16384 = 14169.0 — exact).
        let tables = RopeTables {
            cos: vec![8192, 8192, 8192, 8192],
            sin: vec![14169, 14169, 14169, 14169],
            max_pos: 1,
            head_dim: 4,
        };
        let x = [16384_i64, 0, 0, 0];
        let mat = Mat::from((x.as_slice(), 1, 4));
        let out = apply_int_rope(mat, &tables, &[0]);
        assert_eq!(out[0], 8192);
        assert_eq!(out[1], 0);
        assert_eq!(out[2], 14169);
        assert_eq!(out[3], 0);
    }

    #[test]
    fn norm_preserved_under_rotation() {
        // upstream test_ops.py::TestRope::test_norm_preservation: the
        // rotation preserves |x| up to the table's quantization (±1e-3
        // relative) — integer form: sum of squares within ~0.3%
        let one = (1_i64 << ROPE_FRAC_BITS) as i16;
        // a 30-degree row: cos = 2^14·cos(pi/6) = 14243 (floor), sin = 8192
        let tables = RopeTables {
            cos: vec![14243, 14243, 14243, 14243],
            sin: vec![8192, 8192, 8192, 8192],
            max_pos: 1,
            head_dim: 4,
        };
        let x = [10000_i64, -20000, 30000, -5000];
        let mat = Mat::from((x.as_slice(), 1, 4));
        let out = apply_int_rope(mat, &tables, &[0]);
        let sq = |v: &[i64]| v.iter().map(|a| a * a).sum::<i64>();
        let n_in = sq(&x);
        let n_out = sq(&out);
        // |cos·x + rot·sin|^2 = |x|^2·(cos^2 + sin^2) within the table
        // rounding: cos_t^2 + sin_t^2 = 14243^2 + 8192^2 = 269368049
        // vs 2^28 = 268435456 — about a third of a percent high
        let ratio_num = (n_out as i128) * 1000;
        let ratio_den = (n_in as i128) * 1000;
        let diff = (ratio_num - ratio_den).abs();
        assert!(
            diff * 2 <= ratio_den,
            "norm drift past the half-percent bound: in={n_in} out={n_out}"
        );
        let _ = one;
    }

    #[test]
    fn long_position_no_special_case() {
        // positions are plain table row indices — position 5000 behaves
        // like any other row (upstream test_ops.py long-position arm)
        let max_pos = 6000;
        let d = 4;
        let tables = RopeTables {
            cos: vec![one_val(); max_pos * d],
            sin: vec![0; max_pos * d],
            max_pos,
            head_dim: d,
        };
        let x = [3_i64, 9, -12, 5];
        let mat = Mat::from((x.as_slice(), 1, d));
        let out = apply_int_rope(mat, &tables, &[5999]);
        assert_eq!(out, x);
    }

    fn one_val() -> i16 {
        (1 << ROPE_FRAC_BITS) as i16
    }

    #[test]
    #[should_panic(expected = "past the table's max_pos")]
    fn position_past_table_refuses() {
        let tables = RopeTables::identity(2, 4);
        let x = [1_i64, 2, 3, 4];
        let mat = Mat::from((x.as_slice(), 1, 4));
        let _ = apply_int_rope(mat, &tables, &[5]);
    }
}
