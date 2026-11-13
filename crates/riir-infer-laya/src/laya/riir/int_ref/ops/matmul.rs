//! The exact integer GEMM family (the matmul core of the lane).
//!
//! Decision-content transcription of detllm `backends.py:84-165` @
//! `9c3f1c9c` (NO LICENSE upstream — decision-content-only; overview
//! `.research/027`). "The ONLY backend-divergent code" upstream; here
//! there is exactly ONE backend — the CPU reference — and it computes
//! the exact int8×int8→int32 product-sum. Integer accumulation is
//! associative and commutative within range, so the result is
//! order-independent by construction (upstream: "bit-identical by
//! construction (verified by §9.3 tests, not assumed)"); the
//! order-invariance property test below replays chunked-K accumulation
//! orders and asserts identity (the G1 falsification arm of
//! `.issues/1214`).
//!
//! - [`int_gemm`] — the reference arm of `backends.py:84` (exact int32
//!   product-sum; upstream asserts `hd·127² < 2^31` and
//!   `intermediate·127² < 2^31` so the accumulator cannot overflow —
//!   the checked cast refuses loud if a caller breaches that);
//! - [`int_gemm_u8i8`] — `backends.py:138`, the −64 trick: unsigned
//!   `[0, 128]` left operand via `(p-64)@b + 64·colsum(b)`, exact;
//! - [`int_gemm_u16i8`] — `backends.py:152`, the hi/lo int8 split for
//!   15-bit probabilities `p = hi·2^7 + lo`, exact, returns int64 (the
//!   shifted hi part can exceed int32 for long sequences).
//!
//! `GUARD_ROWS` (upstream `backends.py:47`) defends cuBLASLt int8
//! kernels' speculative over-reads — a CUDA-only concern with no CPU
//! reference analog, noted here and deliberately not ported.

/// Exact integer GEMM: `a [m, k]` int8 × `b [k, n]` int8 → int32.
///
/// Accumulation is a sequential ascending-K i64 fold of exact products;
/// the int32 cast is checked and refuses loud (upstream's documented
/// bound: the accumulator stays below 2^31 for every model shape the
/// lane admits — `head_dim·127²` and `intermediate·127²` both < 2^31).
pub fn int_gemm(a: &[i8], m: usize, k: usize, b: &[i8], n: usize) -> Vec<i32> {
    assert_eq!(a.len(), m * k, "int_gemm: a shape");
    assert_eq!(b.len(), k * n, "int_gemm: b shape");
    let mut out = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            let mut acc: i64 = 0;
            for l in 0..k {
                let prod = (a[i * k + l] as i64) * (b[l * n + j] as i64);
                acc = acc
                    .checked_add(prod)
                    .expect("int_gemm: accumulator overflow");
            }
            out.push(i32::try_from(acc).expect("int_gemm: result out of int32"));
        }
    }
    out
}

/// Exact GEMM for an unsigned-range left operand (values in `[0, 128]`,
/// stored int32) against int8 `b`: computes `(p-64)@b + 64·colsum(b)`,
/// which equals `p@b` exactly in integer arithmetic (`backends.py:138`).
pub fn int_gemm_u8i8(p: &[i32], m: usize, k: usize, b: &[i8], n: usize) -> Vec<i32> {
    assert_eq!(p.len(), m * k, "int_gemm_u8i8: p shape");
    for &v in p {
        assert!((0..=128).contains(&v), "int_gemm_u8i8: p out of [0, 128]");
    }
    let shifted: Vec<i8> = p.iter().map(|&v| (v - 64) as i8).collect();
    let core = int_gemm(&shifted, m, k, b, n);
    let mut out = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            let mut colsum: i64 = 0;
            for l in 0..k {
                colsum = colsum
                    .checked_add(b[l * n + j] as i64)
                    .expect("int_gemm_u8i8: colsum overflow");
            }
            let corr = colsum * 64;
            let v = core[i * n + j] as i64;
            out.push(
                i32::try_from(v.checked_add(corr).expect("int_gemm_u8i8: overflow"))
                    .expect("int_gemm_u8i8: result out of int32"),
            );
        }
    }
    out
}

/// Exact GEMM for 15-bit probabilities (values in `[0, 2^14]`, stored
/// int32) against int8 `b` (`backends.py:152`): `p = hi·2^7 + lo` with
/// `hi ∈ [0, 128]` (the −64 trick) and `lo ∈ [0, 127]` (plain int8).
/// Returns int64 — the shifted hi part can exceed int32 for long
/// sequences.
pub fn int_gemm_u16i8(p: &[i32], m: usize, k: usize, b: &[i8], n: usize) -> Vec<i64> {
    let cap = 1_i64 << 14;
    for &v in p {
        let v = v as i64;
        assert!((0..=cap).contains(&v), "int_gemm_u16i8: p out of [0, 2^14]");
    }
    let hi: Vec<i32> = p.iter().map(|&v| v >> 7).collect();
    let lo: Vec<i32> = p.iter().map(|&v| v - (v >> 7 << 7)).collect();
    let hi_part = int_gemm_u8i8(&hi, m, k, b, n);
    let lo_i8: Vec<i8> = lo.iter().map(|&v| v as i8).collect();
    let lo_part = int_gemm(&lo_i8, m, k, b, n);
    let mut out = Vec::with_capacity(m * n);
    for i in 0..m * n {
        let v = ((hi_part[i] as i64) << 7)
            .checked_add(lo_part[i] as i64)
            .expect("int_gemm_u16i8: overflow");
        out.push(v);
    }
    out
}

#[cfg(test)]
mod goldens {
    use super::*;

    /// A naive triple-loop reference — the transcription of upstream's
    /// test_ops.py::TestGemm::test_matches_naive_loop oracle.
    fn naive_i8(a: &[i8], m: usize, k: usize, b: &[i8], n: usize) -> Vec<i32> {
        let mut out = Vec::with_capacity(m * n);
        for i in 0..m {
            for j in 0..n {
                let mut acc: i64 = 0;
                for l in 0..k {
                    acc += a[i * k + l] as i64 * b[l * n + j] as i64;
                }
                out.push(acc as i32);
            }
        }
        out
    }

    fn deterministic_i8(len: usize, seed: u64) -> Vec<i8> {
        // xorshift64* — a seeded, dependency-free generator (no
        // unseeded-global draws; the workspace's global-RNG gate class)
        let mut s = seed | 1;
        (0..len)
            .map(|_| {
                s ^= s >> 12;
                s ^= s << 25;
                s ^= s >> 27;
                let v = s.wrapping_mul(0x2545F4914F6CDD1D);
                (v >> 56) as i8
            })
            .collect()
    }

    #[test]
    fn hand_golden_small() {
        // hand-computed: a = [[1, -2], [3, 0]], b = [[4, -1], [5, 2]]
        // out[0,0] = 1·4 + (-2)·5 = -6; out[0,1] = 1·(-1) + (-2)·2 = -5;
        // out[1,0] = 3·4 + 0·5 = 12;  out[1,1] = 3·(-1) + 0·2 = -3.
        let a = [1_i8, -2, 3, 0];
        let b = [4_i8, -1, 5, 2];
        assert_eq!(int_gemm(&a, 2, 2, &b, 2), vec![-6, -5, 12, -3]);
    }

    #[test]
    fn matches_naive_loop() {
        // upstream test_ops.py::TestGemm::test_matches_naive_loop
        let (m, k, n) = (9, 12, 7);
        let a = deterministic_i8(m * k, 0xDEAD_0001);
        let b = deterministic_i8(k * n, 0xDEAD_0002);
        assert_eq!(int_gemm(&a, m, k, &b, n), naive_i8(&a, m, k, &b, n));
    }

    #[test]
    fn u8_trick_exact() {
        // upstream test_ops.py::TestGemm::test_u8_trick_exact
        let (m, k, n) = (5, 10, 6);
        let p: Vec<i32> = (0..m * k).map(|i| ((i * 37 + 11) % 129) as i32).collect();
        let b = deterministic_i8(k * n, 0xDEAD_0003);
        // naive unsigned reference in i64
        let mut want = Vec::with_capacity(m * n);
        for i in 0..m {
            for j in 0..n {
                let mut acc: i64 = 0;
                for l in 0..k {
                    acc += p[i * k + l] as i64 * b[l * n + j] as i64;
                }
                want.push(acc as i32);
            }
        }
        assert_eq!(int_gemm_u8i8(&p, m, k, &b, n), want);
    }

    #[test]
    fn u16_hilo_split_exact() {
        // upstream test_ops.py::TestGemm::test_u16_hilo_split_exact:
        // exact against the naive u16 reference, dtype int64
        let (m, k, n) = (4, 16, 5);
        let p: Vec<i32> = (0..m * k)
            .map(|i| ((i * 501 + 7) % (1 << 14)) as i32)
            .collect();
        let b = deterministic_i8(k * n, 0xDEAD_0004);
        let mut want = Vec::with_capacity(m * n);
        for i in 0..m {
            for j in 0..n {
                let mut acc: i64 = 0;
                for l in 0..k {
                    acc += p[i * k + l] as i64 * b[l * n + j] as i64;
                }
                want.push(acc);
            }
        }
        assert_eq!(int_gemm_u16i8(&p, m, k, &b, n), want);
    }

    #[test]
    fn order_invariance_chunked_accumulation() {
        // the G1 falsification arm (`.issues/1214`): permuting the K
        // accumulation's chunk order yields bit-identical output —
        // integer addition within range is associative, so ANY fold
        // order is the same bits; assert it on several permutations.
        let (m, k, n) = (6, 48, 4);
        let a = deterministic_i8(m * k, 0xDEAD_0005);
        let b = deterministic_i8(k * n, 0xDEAD_0006);
        let want = int_gemm(&a, m, k, &b, n);
        // chunk boundaries at several permutations of the K axis
        for chunk in [1_usize, 5, 7, 16, 48] {
            let mut got = Vec::with_capacity(m * n);
            for i in 0..m {
                for j in 0..n {
                    let mut acc: i64 = 0;
                    let mut l0 = 0;
                    while l0 < k {
                        let l1 = (l0 + chunk).min(k);
                        let mut part: i64 = 0;
                        for l in (l0..l1).rev() {
                            // reverse WITHIN the chunk too — the
                            // permutation must not matter
                            part += a[i * k + l] as i64 * b[l * n + j] as i64;
                        }
                        acc += part;
                        l0 = l1;
                    }
                    got.push(acc as i32);
                }
            }
            assert_eq!(got, want, "chunk={chunk} diverged");
        }
    }

    #[test]
    fn accumulator_overflow_refuses_loud() {
        // the lane's bound: hd·127^2 < 2^31 — a shape breaching it
        // refuses loud instead of wrapping
        let k = 200_000; // 200k·127·127 ≈ 3.2e9 > i32::MAX
        let a = vec![127_i8; k];
        let b = vec![127_i8; k];
        let res = std::panic::catch_unwind(|| int_gemm(&a, 1, k, &b, 1));
        assert!(res.is_err(), "overflow must refuse loud");
    }
}
