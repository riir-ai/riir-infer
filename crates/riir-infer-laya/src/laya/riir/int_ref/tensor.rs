//! Tensor-level dyadic bookkeeping (Issue 1214 T1a).
//!
//! Decision-content transcription of detllm `dyadic.py` @ `9c3f1c9c`
//! (NO LICENSE upstream — decision-content-only posture; citations to
//! the pinned def lines below; overview: `.research/027`). A quantized
//! tensor is carried as `(data, m, k)` with `value ≈ data · m / 2^k`;
//! `m`/`k` are per-row scale vectors for activations (upstream: per-token
//! column vectors, shape `[..., 1]`) and per-channel vectors + scalar `k`
//! for weights. All arithmetic is integer-only and goes through the
//! frozen rounding laws of [`super::num`].
//!
//! Transcribed defs (pinned sha `9c3f1c9cdd42294087cc0507992c21dfde944bea`):
//! - `scaled_round_div` (`dyadic.py:45`), `fit_dyadic_ratio` (`:52`),
//!   `norm_scale` (`:75`) — the scale-mantissa bookkeeping;
//! - `quant_i8_pertoken` (`:93`), `requant_i32_common` (`:108`),
//!   `requant_i8_rowcol` (`:123`), `requant_i8_static` (`:142`) — the
//!   quantization/requantization family;
//! - `residual_renorm` (`:174`), `residual_from_delta` (`:187`),
//!   `residual_add` (`:193`) — the int32 residual stream with the
//!   per-token power-of-two scale `[2^RES_MIN_BITS+2, 2^RES_MAX_BITS)`
//!   renorm window (`dyadic.py:170-171`).
//!
//! Overflow law: every widening op is `checked_*` and refuses loud
//! (panic naming the op) — the lane is a reference; a silent wrap would
//! be a wrong bit no gate can see. This mirrors T0's scalar law at
//! tensor granularity.
//!
//! Divergence note (T0 scalar `Dyadic` vs this layer): upstream's
//! residual `k_res` is an unconstrained int64 — it may go NEGATIVE
//! (a value scaled up by `2^-k_res`) after an exact up-shift renorm on
//! a small row. T0's scalar type floors `k >= 1` and refuses
//! (`KUnderflow`); this layer follows upstream exactly (no floor) —
//! the scalar type is the standalone-arithmetic core, this layer is the
//! model's residual grid. The two never mix: the forward (T1b) runs
//! entirely on this layer.

use super::num::{ilog2_floor, round_half_away_div, rshift_round};

/// Residual stream cap (`dyadic.py:170`): keep `max|h| < 2^RES_MAX_BITS`.
pub const RES_MAX_BITS: i64 = 30;
/// Residual stream floor (`dyadic.py:171`): exact up-shift targets MSB
/// `RES_MIN_BITS + 2` = 26 (the window floor upstream commits to).
pub const RES_MIN_BITS: i64 = 24;

/// A row-major 2-D borrow of integer data (the lane's tensor view).
///
/// The lane is a reference: no strides, no genericity — a plain row-major
/// slice with explicit dims keeps every transcribed loop readable against
/// its upstream def.
#[derive(Clone, Copy)]
pub struct Mat<'a, T> {
    pub data: &'a [T],
    pub rows: usize,
    pub cols: usize,
}

impl<T> Mat<'_, T> {
    /// `(r, c)` element. Panics on out-of-range (a shape bug is loud).
    #[inline]
    pub fn get(&self, r: usize, c: usize) -> &T {
        &self.data[r * self.cols + c]
    }

    /// Row `r` as a contiguous slice.
    #[inline]
    pub fn row(&self, r: usize) -> &[T] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    /// Element count (`rows * cols`); every scale-vector length check
    /// goes through this.
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<'a, T> From<(&'a [T], usize, usize)> for Mat<'a, T> {
    fn from((data, rows, cols): (&'a [T], usize, usize)) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "Mat: data len {} does not match {rows} x {cols}",
            data.len()
        );
        Mat { data, rows, cols }
    }
}

/// `scaled_round_div` (`dyadic.py:45`): `round(num · 2^e / den)` with a
/// per-call (possibly negative) `e`, through the non-PoT rounding law.
#[inline]
pub fn scaled_round_div(num: i64, den: i64, e: i64) -> i64 {
    if e >= 0 {
        let shifted = num
            .checked_mul(
                1_i64
                    .checked_shl(e as u32)
                    .expect("scaled_round_div: 2^e overflow"),
            )
            .expect("scaled_round_div: numerator overflow");
        round_half_away_div(shifted, den)
    } else {
        let den_shifted = den
            .checked_mul(
                1_i64
                    .checked_shl((-e) as u32)
                    .expect("scaled_round_div: 2^-e overflow"),
            )
            .expect("scaled_round_div: denominator overflow");
        round_half_away_div(num, den_shifted)
    }
}

/// `norm_scale` (`dyadic.py:75`): renormalize the scale mantissa `m`
/// into `[2^(bits-1), 2^bits)` elementwise, adjusting `k` — exact left
/// shift when `m` grows, [`rshift_round`] when it shrinks. Keeps `k`
/// drift bounded across long op chains.
pub fn norm_scale(m: &mut [i64], k: &mut [i64], bits: i64) {
    assert_eq!(m.len(), k.len(), "norm_scale: m/k length mismatch");
    assert!(bits >= 1, "norm_scale: bits must be >= 1");
    for (mi, ki) in m.iter_mut().zip(k.iter_mut()) {
        let lg = ilog2_floor((*mi).max(1) as u64) as i64;
        let sh = (bits - 1) - lg;
        let up = sh.max(0) as u32;
        let down = (-sh).max(0) as u32;
        let mut m2 = rshift_round(mi.checked_shl(up).expect("norm_scale: m overflow"), down);
        let mut sh = sh;
        if m2 >= 1_i64 << bits {
            m2 = rshift_round(m2, 1);
            sh -= 1;
        }
        *mi = m2;
        *ki += sh;
    }
}

/// `fit_dyadic_ratio` (`dyadic.py:52`): approximate `num/den` by
/// `m/2^e` with `m` aimed at `[2^(bits-1), 2^bits)` (`e` may go
/// negative for ratios above the window). Worst-case relative error
/// `2^-bits`. Caller contract (upstream doc): `num >= 1`, `den >= 1`.
/// Returns `(m, e)` per element of `num`.
pub fn fit_dyadic_ratio(num: &[i64], den: i64, bits: i64) -> (Vec<i64>, Vec<i64>) {
    assert!(den >= 1, "fit_dyadic_ratio: den must be >= 1");
    assert!(bits >= 1, "fit_dyadic_ratio: bits must be >= 1");
    let mut out_m = Vec::with_capacity(num.len());
    let mut out_e = Vec::with_capacity(num.len());
    for &n in num {
        assert!(n >= 1, "fit_dyadic_ratio: num must be >= 1");
        let big = 1_i64 << bits;
        let l = ilog2_floor(n as u64) as i64 - ilog2_floor(den as u64) as i64;
        let mut e = (bits - 1) - l;
        let mut m = scaled_round_div(n, den, e);
        let mut adj: i64 = 0;
        if m >= big {
            adj -= 1;
        }
        if m < big >> 1 {
            adj += 1;
        }
        if adj != 0 {
            e += adj;
            m = scaled_round_div(n, den, e);
        }
        if m >= big {
            m = rshift_round(m, 1);
            e -= 1;
        }
        out_m.push(m);
        out_e.push(e);
    }
    (out_m, out_e)
}

/// Symmetric per-token int8 quantization (`quant_i8_pertoken`,
/// `dyadic.py:93`): `x` rows carry per-row dyadic scale `(m, k)`;
/// returns `(x8, m_out, k_out)` with `value ≈ x8 · m_out / 2^k_out`.
pub fn quant_i8_pertoken(x: Mat<'_, i64>, m: &[i64], k: &[i64]) -> (Vec<i8>, Vec<i64>, Vec<i64>) {
    assert_eq!(m.len(), x.rows, "quant_i8_pertoken: m length");
    assert_eq!(k.len(), x.rows, "quant_i8_pertoken: k length");
    let mut m = m.to_vec();
    let mut k = k.to_vec();
    // keep a·m within int64 headroom (upstream norm_scale pre-pass)
    norm_scale(&mut m, &mut k, 16);
    let mut x8 = Vec::with_capacity(x.len());
    let mut m_out = Vec::with_capacity(x.rows);
    let mut k_out = Vec::with_capacity(x.rows);
    for r in 0..x.rows {
        let row = x.row(r);
        let a = row.iter().map(|v| v.abs()).max().unwrap_or(0).max(1);
        for &v in row {
            let q = round_half_away_div(
                v.checked_mul(127)
                    .expect("quant_i8_pertoken: x·127 overflow"),
                a,
            );
            x8.push(clamp_i8(q));
        }
        let num = a
            .checked_mul(m[r])
            .expect("quant_i8_pertoken: a·m overflow");
        let (fm, fe) = fit_dyadic_ratio(&[num], 127, 16);
        m_out.push(fm[0]);
        k_out.push(k[r] + fe[0]);
    }
    (x8, m_out, k_out)
}

/// Requantize a GEMM output to a common per-row scale at ~`target_bits`
/// of magnitude (`requant_i32_common`, `dyadic.py:108`): per-row
/// `(row_m, row_k)` × per-channel `(col_m, col_k)` scales in, int32 +
/// per-row dyadic out. Shift-only (single [`rshift_round`] per element).
pub fn requant_i32_common(
    p: Mat<'_, i64>,
    row_m: &[i64],
    row_k: &[i64],
    col_m: &[i64],
    col_k: i64,
    target_bits: i64,
) -> (Vec<i32>, Vec<i64>, Vec<i64>) {
    assert_eq!(row_m.len(), p.rows, "requant_i32_common: row_m length");
    assert_eq!(row_k.len(), p.rows, "requant_i32_common: row_k length");
    assert_eq!(col_m.len(), p.cols, "requant_i32_common: col_m length");
    let mut y = Vec::with_capacity(p.len());
    let mut k_out = Vec::with_capacity(p.rows);
    for (r, &rk) in row_k.iter().enumerate() {
        let row = p.row(r);
        let scaled: Vec<i64> = row
            .iter()
            .enumerate()
            .map(|(c, &v)| {
                v.checked_mul(col_m[c])
                    .expect("requant_i32_common: P·col_m overflow")
            })
            .collect();
        let a = scaled.iter().map(|v| v.abs()).max().unwrap_or(0).max(1);
        let sh = (ilog2_floor(a as u64) as i64 - (target_bits - 1)).max(0) as u32;
        for &s in &scaled {
            let v = rshift_round(s, sh);
            let v = i32::try_from(v).expect("requant_i32_common: int32 overflow");
            y.push(v);
        }
        k_out.push(rk + col_k - sh as i64);
    }
    (y, row_m.to_vec(), k_out)
}

/// Requantize a GEMM output straight to symmetric per-token int8
/// (`requant_i8_rowcol`, `dyadic.py:123`) — division-based, uses the
/// full ±127 range.
pub fn requant_i8_rowcol(
    p: Mat<'_, i64>,
    row_m: &[i64],
    row_k: &[i64],
    col_m: &[i64],
    col_k: i64,
) -> (Vec<i8>, Vec<i64>, Vec<i64>) {
    assert_eq!(row_m.len(), p.rows, "requant_i8_rowcol: row_m length");
    assert_eq!(row_k.len(), p.rows, "requant_i8_rowcol: row_k length");
    assert_eq!(col_m.len(), p.cols, "requant_i8_rowcol: col_m length");
    let mut y8 = Vec::with_capacity(p.len());
    let mut m_out = Vec::with_capacity(p.rows);
    let mut k_out = Vec::with_capacity(p.rows);
    for r in 0..p.rows {
        let row = p.row(r);
        let mut scaled: Vec<i64> = row
            .iter()
            .enumerate()
            .map(|(c, &v)| {
                v.checked_mul(col_m[c])
                    .expect("requant_i8_rowcol: P·col_m overflow")
            })
            .collect();
        // headroom guard (upstream `dyadic.py:129-131`): keep scaled·127
        // within int64 — the pre-shift cancels in the ratio below, only
        // the k bookkeeping moves.
        let a0 = scaled.iter().map(|v| v.abs()).max().unwrap_or(0).max(1);
        let sh = (ilog2_floor(a0 as u64) as i64 - 45).max(0) as u32;
        if sh > 0 {
            for s in scaled.iter_mut() {
                *s = rshift_round(*s, sh);
            }
        }
        let a = scaled.iter().map(|v| v.abs()).max().unwrap_or(0).max(1);
        for &s in &scaled {
            let q = round_half_away_div(
                s.checked_mul(127)
                    .expect("requant_i8_rowcol: scaled·127 overflow"),
                a,
            );
            y8.push(clamp_i8(q));
        }
        let num = a
            .checked_mul(row_m[r])
            .expect("requant_i8_rowcol: a·row_m overflow");
        let (fm, fe) = fit_dyadic_ratio(&[num], 127, 16);
        m_out.push(fm[0]);
        k_out.push(row_k[r] + col_k + fe[0] - sh as i64);
    }
    (y8, m_out, k_out)
}

/// Requantize a GEMM output to int8 at a STATIC dyadic target scale
/// `s_m/2^s_k` (`requant_i8_static`, `dyadic.py:142`) — used for K and V
/// heading into the cache; ONE fused rounding.
///
/// `row_m`/`row_k` are per-row; `col_m` per-channel with scalar `col_k`;
/// `s_m`/`s_k` are PER-ELEMENT (`rows*cols`) — callers flatten upstream's
/// broadcast shapes before calling. `y = round(P · col_m · row_m · 2^s_k
/// / (2^(row_k+col_k) · s_m))`, clamped.
pub fn requant_i8_static(
    p: Mat<'_, i64>,
    row_m: &[i64],
    row_k: &[i64],
    col_m: &[i64],
    col_k: i64,
    s_m: &[i64],
    s_k: &[i64],
) -> Vec<i8> {
    assert_eq!(row_m.len(), p.rows, "requant_i8_static: row_m length");
    assert_eq!(row_k.len(), p.rows, "requant_i8_static: row_k length");
    assert_eq!(col_m.len(), p.cols, "requant_i8_static: col_m length");
    assert_eq!(s_m.len(), p.len(), "requant_i8_static: s_m length");
    assert_eq!(s_k.len(), p.len(), "requant_i8_static: s_k length");
    let mut y = Vec::with_capacity(p.len());
    for (r, (&rm, &rk)) in row_m.iter().zip(row_k.iter()).enumerate() {
        for (c, &cm) in col_m.iter().enumerate() {
            let idx = r * p.cols + c;
            let num = p
                .get(r, c)
                .checked_mul(cm)
                .expect("requant_i8_static: P·col_m overflow")
                .checked_mul(rm)
                .expect("requant_i8_static: ·row_m overflow");
            let shift = rk + col_k - s_k[idx];
            let up = (-shift).max(0) as u32;
            let down = shift.max(0) as u32;
            let num = num
                .checked_shl(up)
                .expect("requant_i8_static: numerator overflow");
            let den = s_m[idx]
                .checked_shl(down)
                .expect("requant_i8_static: s_m overflow");
            y.push(clamp_i8(round_half_away_div(num, den)));
        }
    }
    y
}

/// Bring per-row `max|h|` into `[2^26, 2^RES_MAX_BITS)` (`residual_renorm`,
/// `dyadic.py:174`), adjusting the per-row power-of-two scale `k_res`.
/// Left shifts are exact; right shifts round through [`rshift_round`].
pub fn residual_renorm(h: Mat<'_, i64>, k_res: &[i64]) -> (Vec<i32>, Vec<i64>) {
    assert_eq!(k_res.len(), h.rows, "residual_renorm: k_res length");
    let mut h_out = Vec::with_capacity(h.len());
    let mut k_out = Vec::with_capacity(h.rows);
    for (r, &kr) in k_res.iter().enumerate() {
        let row = h.row(r);
        let g = row.iter().map(|v| v.abs()).max().unwrap_or(0);
        let lg = ilog2_floor(g.max(1) as u64) as i64;
        let sh_down = (lg - (RES_MAX_BITS - 1)).max(0);
        let sh_up = if g > 0 {
            (RES_MIN_BITS + 2 - lg).max(0)
        } else {
            0
        };
        let sh_up = if sh_down > 0 { 0 } else { sh_up };
        for &v in row {
            let down = sh_down as u32;
            let up = sh_up as u32;
            let shifted = rshift_round(v, down)
                .checked_shl(up)
                .expect("residual_renorm: up-shift overflow");
            h_out.push(i32::try_from(shifted).expect("residual_renorm: int32 overflow"));
        }
        k_out.push(kr - sh_down + sh_up);
    }
    (h_out, k_out)
}

/// Turn a dyadic-scaled delta into residual-grid form
/// (`residual_from_delta`, `dyadic.py:187`).
pub fn residual_from_delta(d: Mat<'_, i64>, dm: &[i64], dk: &[i64]) -> (Vec<i32>, Vec<i64>) {
    assert_eq!(dm.len(), d.rows, "residual_from_delta: dm length");
    assert_eq!(dk.len(), d.rows, "residual_from_delta: dk length");
    let mut h = Vec::with_capacity(d.len());
    for (r, &dm_r) in dm.iter().enumerate() {
        for &v in d.row(r) {
            h.push(
                v.checked_mul(dm_r)
                    .expect("residual_from_delta: d·dm overflow"),
            );
        }
    }
    let hmat = Mat::from((h.as_slice(), d.rows, d.cols));
    residual_renorm(hmat, dk)
}

/// `h` (int32 data, per-row scale `2^-k_res`) `+= d · dm / 2^dk`
/// (`residual_add`, `dyadic.py:193`). Alignment rule (spec §6.8, the
/// same law T0 pins at scalar granularity): shift the finer operand down
/// to the coarser `k` with [`rshift_round`]; shifting up is exact. Then
/// renormalize into the residual window.
pub fn residual_add(
    h: Mat<'_, i64>,
    k_res: &[i64],
    d: Mat<'_, i64>,
    dm: &[i64],
    dk: &[i64],
) -> (Vec<i32>, Vec<i64>) {
    assert_eq!(h.rows, d.rows, "residual_add: row mismatch");
    assert_eq!(h.cols, d.cols, "residual_add: col mismatch");
    assert_eq!(k_res.len(), h.rows, "residual_add: k_res length");
    assert_eq!(dm.len(), d.rows, "residual_add: dm length");
    assert_eq!(dk.len(), d.rows, "residual_add: dk length");
    let mut sum = Vec::with_capacity(h.len());
    for (r, ((&dm_r, &dk_r), &kr)) in dm.iter().zip(dk.iter()).zip(k_res.iter()).enumerate() {
        let rel = dk_r - kr; // > 0: delta finer -> shift delta down
        let down = rel.max(0) as u32;
        let up = (-rel).max(0) as u32;
        for c in 0..h.cols {
            let dv = d
                .get(r, c)
                .checked_mul(dm_r)
                .expect("residual_add: d·dm overflow");
            let aligned = rshift_round(dv, down)
                .checked_shl(up)
                .expect("residual_add: up-shift overflow");
            let s = h
                .get(r, c)
                .checked_add(aligned)
                .expect("residual_add: h + d overflow");
            sum.push(s);
        }
    }
    let sum_mat = Mat::from((sum.as_slice(), h.rows, h.cols));
    residual_renorm(sum_mat, k_res)
}

/// Saturating cast to int8 (`intmath.clamp_i8`).
#[inline]
pub fn clamp_i8(v: i64) -> i8 {
    v.clamp(-128, 127) as i8
}
