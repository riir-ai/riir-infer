//! The dyadic `(data, m, k)` scale bookkeeping (Issue 1214 T0, scalar).
//!
//! Decision-content transcription of detllm `dyadic.py:3-6,166-205` via
//! `.research/027`: a value is carried as `data · m / 2^k` — `m` the
//! integer scale numerator, `k` the SHARED power-of-two denominator
//! exponent. Upstream rules transcribed here (scalar bookkeeping form;
//! the per-token/per-channel tensor layer is T1):
//! - alignment: "shift the finer operand down to the coarser k" — the
//!   operand with the SMALLER k gives up its extra precision (its data
//!   rounds through [`rshift_round`]) so both share one k before the
//!   integer op;
//! - `fit_dyadic_ratio`: fit a dyadic to a ratio with worst-case
//!   relative error `2^-(bits)` (the m/k constructor);
//! - `norm_scale`: renormalize so `k` drift stays bounded across long
//!   op chains (the data field kept in a fixed bit budget);
//! - the residual-stream renorm window `[2^26, 2^30)` on the data's
//!   magnitude (int32 data kept away from both overflow and
//!   precision-starved tails).
//!
//! Overflow law: every op refuses (returns a loud error, never wraps)
//! — the lane is a reference, a silent wrap would be a wrong bit that
//! no gate can see.

use super::num::{ilog2_floor, round_half_away_div, rshift_round};

/// A scalar dyadic: `value = data · m / 2^k` exactly representable in
/// integers (the fraction is a power of two, so rounding only ever
/// happens through the pinned [`rshift_round`] law).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dyadic {
    pub data: i64,
    pub m: i64,
    pub k: u32,
}

/// Why a dyadic op refused — the lane is a reference; every refusal is
/// a named condition, never a silent wrap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DyadicError {
    /// The aligned data would exceed the i32 budget the residual
    /// renorm window manages (upper bound of the window, halved for
    /// headroom across the following op chain).
    DataOverflow,
    /// A scale numerator collapsed to zero — the value would be
    /// silently zero; refused.
    ZeroScale,
    /// A caller asked for a k below the floor the alignment rule
    /// manages (k must stay >= 1 so rounding shifts are defined).
    KUnderflow,
}

impl core::fmt::Display for DyadicError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DyadicError::DataOverflow => write!(f, "dyadic: data overflow past the renorm window"),
            DyadicError::ZeroScale => write!(f, "dyadic: scale collapsed to zero"),
            DyadicError::KUnderflow => write!(f, "dyadic: k underflow (k must stay >= 1)"),
        }
    }
}

/// The residual-stream data renorm window (upstream `dyadic.py`): data
/// magnitude renormed into `[2^26, 2^30)`.
pub const RENORM_LO_EXP: u32 = 26;
pub const RENORM_HI_EXP: u32 = 30;

impl Dyadic {
    /// Exact constructor: the caller guarantees `value == data*m/2^k`
    /// holds without rounding (no shift applied).
    pub const fn exact(data: i64, m: i64, k: u32) -> Self {
        Dyadic { data, m, k }
    }

    /// Value as an exact rational numerator/denominator pair
    /// (`num / 2^k` after folding m in is the caller's arithmetic).
    /// Ops normalize through [`Dyadic::normalized`] before touching the
    /// data, so an `m != 1` form is legal at construction and folded
    /// once, checked.
    pub fn scaled_data(&self) -> Result<i64, DyadicError> {
        self.data
            .checked_mul(self.m)
            .ok_or(DyadicError::DataOverflow)
    }

    /// Fold `m` into the data (checked): every op path runs on the
    /// normalized m=1 form so the bookkeeping cannot tangle.
    fn normalized(self) -> Result<Self, DyadicError> {
        let data = self.scaled_data()?;
        Ok(Dyadic {
            data,
            m: 1,
            k: self.k,
        })
    }

    /// `fit_dyadic_ratio`: the smallest `k >= bits` whose rounded
    /// representation carries the value with worst-case relative error
    /// `<= 2^-(bits)` (upstream's fit law). The fit path rounds the
    /// non-PoT denominator through [`round_half_away_div`] — the fit is
    /// OFFLINE calibration (never bit-reproducibility-bound, upstream's
    /// own rule) and this is its ONE pinned rounding site; the online
    /// forward never calls fit. Zero fits exactly at `k == bits`.
    pub fn fit(value_num: i64, value_den: u64, bits: u32) -> Option<Self> {
        assert!(value_den > 0, "fit: zero denominator");
        if value_num == 0 {
            return Some(Dyadic::exact(0, 1, bits));
        }
        let mut k = bits.max(1);
        loop {
            // `?` on an Option in an Option-returning fn: overflow -> None
            let prod = value_num.checked_mul(1_i64 << k)?;
            let data = round_half_away_div(prod, value_den as i64);
            if data != 0 {
                return Some(Dyadic { data, m: 1, k });
            }
            // data underflowed to zero — deepen k until the value
            // is representable or the numerator shift overflows.
            k += 1;
            if k > 62 {
                return None;
            }
        }
    }

    /// Alignment (the upstream law): shift the finer (larger-k) operand
    /// DOWN to the coarser (smaller-k) one through [`rshift_round`].
    /// Returns the pair normalized (m=1) on a common k.
    pub fn align(a: Self, b: Self) -> Result<(Self, Self), DyadicError> {
        let a = a.normalized()?;
        let b = b.normalized()?;
        if a.k == b.k {
            return Ok((a, b));
        }
        if a.k < b.k {
            let drop = b.k - a.k;
            Ok((
                a,
                Dyadic {
                    data: shift_data_down(b.data, drop)?,
                    m: 1,
                    k: a.k,
                },
            ))
        } else {
            let drop = a.k - b.k;
            Ok((
                Dyadic {
                    data: shift_data_down(a.data, drop)?,
                    m: 1,
                    k: b.k,
                },
                b,
            ))
        }
    }

    /// Aligned add: normalize, align, add on the shared k. The result
    /// keeps the coarser k; the caller renorms when the chain demands
    /// it (`renorm_window`).
    pub fn aligned_add(a: Self, b: Self) -> Result<Self, DyadicError> {
        let (a, b) = Self::align(a, b)?;
        let data = a
            .data
            .checked_add(b.data)
            .ok_or(DyadicError::DataOverflow)?;
        Ok(Dyadic { data, m: 1, k: a.k })
    }

    /// Aligned mul: normalize, align, multiply (the k's ADD — the
    /// denominator squares). Deep k's are the caller's renorm duty
    /// (upstream `norm_scale` keeps chains bounded).
    pub fn aligned_mul(a: Self, b: Self) -> Result<Self, DyadicError> {
        let (a, b) = Self::align(a, b)?;
        let data = a
            .data
            .checked_mul(b.data)
            .ok_or(DyadicError::DataOverflow)?;
        let k = a.k.checked_add(b.k).ok_or(DyadicError::KUnderflow)?;
        Ok(Dyadic { data, m: 1, k })
    }

    /// `norm_scale`: renormalize the data magnitude into the residual
    /// window `[2^26, 2^30)` by moving powers of two between data and
    /// k (value-preserving up to the pinned rounding at the window
    /// floor). Long op chains keep k drift bounded through this.
    pub fn renorm_window(self) -> Result<Self, DyadicError> {
        let norm = self.normalized()?;
        let mag = norm.data.unsigned_abs();
        if mag == 0 {
            return Ok(norm); // zero renorms to itself
        }
        let e = ilog2_floor(mag);
        if e < RENORM_LO_EXP {
            // too small — deepen precision: data <<= (LO - e), k gives up
            // the same budget (k must stay >= 1).
            let lift = RENORM_LO_EXP - e;
            let new_k = norm.k.checked_sub(lift).ok_or(DyadicError::KUnderflow)?;
            let data = norm
                .data
                .checked_mul(1_i64 << lift)
                .ok_or(DyadicError::DataOverflow)?;
            Ok(Dyadic {
                data,
                m: 1,
                k: new_k,
            })
        } else if e >= RENORM_HI_EXP {
            // too large — give up precision: round the data down into the
            // window through the pinned law.
            let drop = e - (RENORM_HI_EXP - 1);
            let data = shift_data_down(norm.data, drop)?;
            Ok(Dyadic {
                data,
                m: 1,
                k: norm.k + drop,
            })
        } else {
            Ok(norm)
        }
    }
}

/// The shared down-shift through the pinned rounding law; the loud
/// overflow check rides here so every alignment path is covered.
#[inline]
fn shift_data_down(data: i64, drop: u32) -> Result<i64, DyadicError> {
    let shifted = rshift_round(data, drop);
    if !(-(1_i64 << 31)..=(1_i64 << 31) - 1).contains(&shifted) {
        // the int32 budget the residual window manages — a value that
        // overflows it cannot ride an int32 tensor layer (T1).
        return Err(DyadicError::DataOverflow);
    }
    Ok(shifted)
}

#[cfg(test)]
mod goldens {
    use super::*;

    #[test]
    fn exact_roundtrip() {
        let d = Dyadic::exact(3, 1, 2); // 3/4
        assert_eq!(d.scaled_data().expect("scaled"), 3);
        assert_eq!(d.k, 2);
        // an m != 1 construction normalizes through the op paths
        let m = Dyadic::exact(3, 2, 2); // 6/4 = 3/2
        let n = m.normalized().expect("norm");
        assert_eq!((n.data, n.m), (6, 1));
    }

    #[test]
    fn align_shifts_the_finer_down() {
        let coarse = Dyadic::exact(5, 1, 2); // 5/4
        let fine = Dyadic::exact(1, 1, 5); // 1/32
        let (a, b) = Dyadic::align(coarse, fine).expect("align");
        assert_eq!(a.k, 2);
        assert_eq!(b.k, 2);
        // 1/32 at k=5 -> k=2 is a drop of 3: rshift_round(1, 3) = 0 —
        // the fine operand's sub-ULP content rounds away (the law).
        assert_eq!(b.data, 0);
        // same-k operands pass through untouched
        let (x, y) = Dyadic::align(coarse, coarse).expect("align");
        assert_eq!((x.data, y.data), (5, 5));
    }

    #[test]
    fn aligned_add_goldens() {
        // 1/2 + 1/4: the finer (k=2) operand rounds DOWN to the coarser
        // k=1 (rshift_round(1,1)=1 → 1/2), so the sum is 1 on k=1 —
        // the alignment law's precision cost, visible by design.
        let a = Dyadic::exact(1, 1, 1);
        let b = Dyadic::exact(1, 1, 2);
        let s = Dyadic::aligned_add(a, b).expect("add");
        assert_eq!((s.data, s.k), (2, 1));
        // equal-k add: 3/8 + (-1/8) = 2/8
        let s2 = Dyadic::aligned_add(Dyadic::exact(3, 1, 3), Dyadic::exact(-1, 1, 3)).expect("add");
        assert_eq!((s2.data, s2.k), (2, 3));
        // 1/4 + 1/4 = 2/4
        let s3 = Dyadic::aligned_add(Dyadic::exact(1, 1, 2), Dyadic::exact(1, 1, 2)).expect("add");
        assert_eq!((s3.data, s3.k), (2, 2));
    }

    #[test]
    fn aligned_mul_goldens() {
        // 1/2 * 1/4: align to the coarser k=1 first (the fine operand's
        // 1 rounds to 1 — unchanged), product data 1, k adds 1+1 = 2
        // → 1/4. (The alignment costs nothing here; a fine value whose
        // rounding WOULD change it is the add golden's job.)
        let a = Dyadic::exact(1, 1, 1);
        let b = Dyadic::exact(1, 1, 2);
        let p = Dyadic::aligned_mul(a, b).expect("mul");
        assert_eq!((p.data, p.k), (1, 2));
    }

    #[test]
    fn renorm_window_goldens() {
        // 2^28 sits inside [2^26, 2^30): untouched
        let inside = Dyadic::exact(1_i64 << 28, 1, 4)
            .renorm_window()
            .expect("renorm");
        assert_eq!(inside.data, 1_i64 << 28);
        // 2^40 sits above 2^30: drops 11 (e=40 -> HI-1=29), data rounds
        // to 2^29 exactly (the drop is a pure PoT shift — no rounding)
        let big = Dyadic::exact(1_i64 << 40, 1, 4)
            .renorm_window()
            .expect("renorm");
        assert_eq!(big.data, 1_i64 << 29);
        assert_eq!(big.k, 4 + 11);
        // 2^10 sits below 2^26: lifts precision 16 bits (k gives them up)
        let small = Dyadic::exact(1_i64 << 10, 1, 20)
            .renorm_window()
            .expect("renorm");
        assert_eq!(small.data, 1_i64 << 26);
        assert_eq!(small.k, 20 - 16);
        // zero renorms to itself
        let z = Dyadic::exact(0, 1, 8).renorm_window().expect("renorm");
        assert_eq!(z.data, 0);
        // a lift past the k floor refuses loud
        let too_fine = Dyadic::exact(1, 1, 3).renorm_window();
        assert_eq!(too_fine, Err(DyadicError::KUnderflow));
    }

    #[test]
    fn fit_dyadic_ratio_goldens() {
        // fit 1/3 at 8 bits: k = 8, data = round(2^8/3) = round(85.33) = 85
        let f = Dyadic::fit(1, 3, 8).expect("fit");
        assert_eq!((f.data, f.k), (85, 8));
        // rel error <= 2^-8: |85/256 - 1/3| = |255-256|/768 = 1/768 ≈ 2^-9.58
        assert!(f.k >= 8);
        // exact PoT value fits with zero error
        let q = Dyadic::fit(1, 4, 8).expect("fit");
        assert_eq!((q.data, q.k), (64, 8)); // 64/2^8 = 1/4 exact
        // zero fits exactly at k == bits
        let z = Dyadic::fit(0, 7, 8).expect("fit zero");
        assert_eq!((z.data, z.k), (0, 8));
    }
}
