//! The frozen integer rounding laws + integer helpers (Issue 1214 T0).
//!
//! Decision-content transcription of detllm `intmath.py:7-17` ("Rounding
//! policy (frozen after milestone 1)") via `.research/027`:
//! - `rshift_round(a,k) = (a + (1<<(k-1))) >> k` — ties toward +inf — for
//!   ALL power-of-two rescales;
//! - `round_half_away_div` only where a non-power-of-two divisor appears
//!   (softmax normalization);
//! - `ilog2_floor`, bitwise `isqrt`.
//!
//! The decode argmax carries the pinned FIRST-INDEX tie rule
//! (katgpt-rs Issue 940 law): int32 logits tie exactly, so the pick
//! must be a function of the values alone — never a tolerance band,
//! never a HashMap order.

/// Arithmetic right shift with round-half-toward-+inf (the frozen law).
///
/// `(a + (1<<(k-1))) >> k` for `k >= 1`; `k == 0` is the identity.
/// Rust's `>>` on `i64` is arithmetic (sign-extending), and the added
/// bias makes the half-way case round UP toward +inf for either sign:
/// `rshift_round(-3, 1) == -1` (−1.5 → −1), `rshift_round(3, 1) == 2`.
#[inline]
pub fn rshift_round(a: i64, k: u32) -> i64 {
    if k == 0 {
        return a;
    }
    (a + (1_i64 << (k - 1))) >> k
}

/// Round-half-away-from-zero division by a positive non-power-of-two
/// divisor (the softmax-normalizer form; the ONLY place upstream allows
/// a non-PoT divisor).
///
/// Panics on a non-positive divisor: the softmax denominator is a sum
/// of non-negative int32 logits' exponentials — a non-positive divisor
/// is a caller-contract breach, refused loud, never rounded through.
#[inline]
pub fn round_half_away_div(a: i64, b: i64) -> i64 {
    assert!(b > 0, "round_half_away_div: non-positive divisor {b}");
    if a >= 0 {
        (a + b / 2) / b
    } else {
        (a - b / 2) / b
    }
}

/// Floor of log2 of a positive `v` (`ilog2_floor(1) == 0`). `v == 0`
/// has no floor log2 — refused loud.
#[inline]
pub fn ilog2_floor(v: u64) -> u32 {
    assert!(v != 0, "ilog2_floor: zero has no floor log2");
    63 - v.leading_zeros()
}

/// Bitwise integer square root (floor), per upstream `intmath.py`.
/// Newton-free: the classic bit-by-bit restoring method — branch-light,
/// overflow-free, exact for every u64.
#[inline]
pub fn isqrt(v: u64) -> u64 {
    if v < 2 {
        return v;
    }
    let mut root: u64 = 0;
    let mut rem: u64 = 0;
    // two bits at a time, from the top EVEN bit index down (the pair
    // alignment the restoring method needs; 64-bit values start at 62).
    let top_bit = 63 - v.leading_zeros();
    let mut shift = (top_bit & !1) as i32;
    while shift >= 0 {
        rem = (rem << 2) | (v >> shift as u32) & 0b11;
        shift -= 2;
        root <<= 1;
        let cand = (root << 1) | 1;
        if rem >= cand {
            rem -= cand;
            root |= 1;
        }
    }
    root
}

/// Argmax with the PINNED first-index tie rule (katgpt-rs Issue 940):
/// an exact tie picks the EARLIEST index — a pure function of the
/// slice's values and order, identical on every backend.
#[inline]
pub fn argmax_first(v: &[i32]) -> usize {
    assert!(!v.is_empty(), "argmax_first: empty slice");
    let mut best = 0usize;
    for i in 1..v.len() {
        // strictly greater — an equal value never displaces the earlier one
        if v[i] > v[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod goldens {
    use super::*;

    #[test]
    fn rshift_round_goldens() {
        // ties toward +inf at every parity boundary, both signs
        assert_eq!(rshift_round(4, 1), 2);
        assert_eq!(rshift_round(5, 1), 3); // 2.5 -> 3
        assert_eq!(rshift_round(-5, 1), -2); // -2.5 -> -2 (toward +inf)
        assert_eq!(rshift_round(-4, 1), -2);
        assert_eq!(rshift_round(7, 2), 2); // 1.75 -> 2
        assert_eq!(rshift_round(6, 2), 2); // 1.5 -> 2 (tie -> +inf)
        assert_eq!(rshift_round(-6, 2), -1); // -1.5 -> -1 (tie -> +inf)
        assert_eq!(rshift_round(1, 0), 1); // k=0 identity
        assert_eq!(rshift_round(-1, 0), -1);
        // wide-range spot: 2^40 + (2^19 - 1) rounds down at k=20,
        // 2^40 + 2^19 rounds up (exact tie)
        let big = 1_i64 << 40;
        assert_eq!(rshift_round(big + (1 << 19) - 1, 20), 1 << 20);
        // (2^40 + 2^19) / 2^20 = 2^20 + 1/2 — the exact tie rounds up
        assert_eq!(rshift_round(big + (1 << 19), 20), (1 << 20) + 1);
    }

    #[test]
    fn round_half_away_div_goldens() {
        assert_eq!(round_half_away_div(7, 2), 4); // 3.5 -> 4 (away from zero)
        assert_eq!(round_half_away_div(-7, 2), -4); // -3.5 -> -4 (away)
        assert_eq!(round_half_away_div(6, 2), 3);
        assert_eq!(round_half_away_div(10, 3), 3); // 3.33 -> 3
        assert_eq!(round_half_away_div(-10, 3), -3);
        assert_eq!(round_half_away_div(0, 5), 0);
    }

    #[test]
    #[should_panic(expected = "non-positive divisor")]
    fn round_half_away_div_refuses_zero_divisor() {
        let _ = round_half_away_div(10, 0);
    }

    #[test]
    fn ilog2_floor_goldens() {
        assert_eq!(ilog2_floor(1), 0);
        assert_eq!(ilog2_floor(2), 1);
        assert_eq!(ilog2_floor(3), 1);
        assert_eq!(ilog2_floor(4), 2);
        assert_eq!(ilog2_floor(1 << 26), 26);
        assert_eq!(ilog2_floor((1 << 26) + 1), 26);
        assert_eq!(ilog2_floor(u64::MAX), 63);
    }

    #[test]
    #[should_panic(expected = "no floor log2")]
    fn ilog2_floor_refuses_zero() {
        let _ = ilog2_floor(0);
    }

    #[test]
    fn isqrt_goldens() {
        for v in 0_u64..10_000 {
            let r = isqrt(v);
            assert!(r * r <= v, "isqrt({v}) = {r} overshot");
            assert!((r + 1) * (r + 1) > v, "isqrt({v}) = {r} undershot");
        }
        // exact squares and the u64-wide edge
        assert_eq!(isqrt(1 << 62), 1 << 31);
        assert_eq!(isqrt(u64::MAX), 4_294_967_295); // floor(sqrt(2^64 - 1))
        assert_eq!(isqrt(0), 0);
        assert_eq!(isqrt(2), 1);
    }

    #[test]
    fn argmax_first_goldens() {
        assert_eq!(argmax_first(&[1, 3, 3, 2]), 1); // tie -> EARLIEST
        assert_eq!(argmax_first(&[5]), 0);
        assert_eq!(argmax_first(&[-3, -1, -2]), 1);
        assert_eq!(argmax_first(&[i32::MIN, i32::MAX]), 1);
        assert_eq!(argmax_first(&[i32::MAX, i32::MAX, 0]), 0); // tie -> first
    }

    #[test]
    #[should_panic(expected = "empty slice")]
    fn argmax_first_refuses_empty() {
        let _ = argmax_first(&[]);
    }
}
