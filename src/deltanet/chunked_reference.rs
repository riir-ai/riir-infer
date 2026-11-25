//! Chunked-recurrence CPU reference — the Issue-1236 T1 G1 instrument.
//!
//! The lane's T0 verdict ([`.benchmarks/1265_i1236_t0_decomposition.md`])
//! routed the gdn_rec decode residue to lever class 1 (the chunked
//! intra-chunk form). Before any timing (the kernel-recovery law, T1), the
//! chunked family needs a bit-identity gate — and the gate needs a
//! **decomposition-level proof** that the fold order it pins is exact.
//! This module is that instrument:
//!
//! - [`sequential_oracle`] — the MR kernel's per-token contract
//!   (`deltanet_recurrence_multi_token_mr_f32`) in scalar Rust: decay →
//!   retrieval dot → delta → rank-1 update → output dot **after** the
//!   update, `scale = 1/√d`. Every dot folds through
//!   [`katgpt_types::simd::dot_f32_ordered`] — the ascending-sequential
//!   association, the determinism law's pinned form (Issue 940).
//! - [`chunked_horner`] — the **control** arm: chunk bookkeeping only,
//!   each chunk still evaluated as the per-element sequential chain from
//!   the carried state. This is the decomposition that MUST be
//!   bit-identical at every chunk length (the harness-proof arm; a failure
//!   here means the instrument itself is broken).
//! - [`chunked_pairwise`] — the **sensitivity probe** arm: the literature
//!   O(p²/d) association (explicit per-token decay products summed over
//!   the intra-chunk terms). Floating-point reassociation changes bits, so
//!   this arm is expected to DIFFER on multi-term chunks — that
//!   difference is the harness's own canary: a parity gate that cannot
//!   distinguish a wrong-order candidate from the oracle is vacuous.
//!
//! # The fold-order contract (what a T2 kernel must replicate)
//!
//! Per state element `(row, col)` of one head, per token `t`, in token
//! order, the exact op sequence is:
//!
//! ```text
//! s ← s · decay_t                                  // step 1, mul
//! kv_row = assoc_dot(s_row, k_t)                   // step 2, reduction
//! delta_t = beta_t · (v_t[row] − kv_row)           // step 3
//! s ← s + k_t[col] · delta_t                       // step 4, mul + add
//! out_t[row] = assoc_dot(s_row, q_t) · scale       // step 5, AFTER update
//! ```
//!
//! The chain is **Horner in the decay products** — the sequential form
//! never materializes a cumulative decay product. Any arm that expands the
//! chain into explicit products/sums (the pairwise form) reassociates the
//! arithmetic and therefore cannot pass a bit-identity gate. The two
//! admissible chunking shapes are: (a) the Horner control's shape — chunk
//! bookkeeping around unchanged per-element chains (the win there is
//! dispatch/working-set structure, not math parallelism); (b) a form whose
//! intra-chunk terms are each individually the exact sequential
//! contribution and whose combination folds in ascending token order —
//! provably identical here only in the degenerate one-term-per-chunk case.
//!
//! # Layer split (what this module does and does not pin)
//!
//! - **This module (Layer 1)** pins the DECOMPOSITION math on CPU: oracle
//!   vs chunk arms, both folding through the same ordered dots.
//! - **Layer 2 (the GPU G1 gate**, `mr_is_bit_identical_to_multi_token`
//!   extended with the T2 kernel arm) pins the Metal kernel: GPU-vs-GPU
//!   bits, where `plane_sum`'s butterfly IS the association. Layer 1
//!   says nothing about the butterfly; Layer 2 says nothing about the
//!   decomposition. A T2 candidate must pass both.
//!
//! Issues 940 + 1236 are the governing laws; the Case fixtures mirror the
//! MR module's G1 recipe (`lcg_fill`, same constants) so the evidence
//! stays comparable across lanes.

use katgpt_types::simd::dot_f32_ordered;

/// One G1 case: the MR kernel's input layout, CPU-side.
///
/// Layouts match the GPU harness exactly (`qkvx` rows of `3·v_dim` =
/// q | k | v per token; per-head scalars `beta`/`decay`; the carried state
/// `[n_head · d · d]`).
pub struct RecCase {
    /// `[p · 3 · v_dim]` — per-token `q | k | v` block (head-major).
    pub qkvx: Vec<f32>,
    /// `[p · n_head]` — update rate per (token, head).
    pub beta: Vec<f32>,
    /// `[p · n_head]` — decay per (token, head).
    pub decay: Vec<f32>,
    /// Carried state `[n_head · d · d]`.
    pub state: Vec<f32>,
}

impl RecCase {
    /// The MR module G1 test's `case()` recipe, mirrored byte-for-byte
    /// (same LCG constants, same ranges) so cross-lane evidence compares.
    #[must_use]
    pub fn mirror_gpu_case(
        n_head: usize,
        d: usize,
        p: usize,
        seed: u64,
        adversarial: bool,
    ) -> Self {
        let v_dim = n_head * d;
        let lcg = |seed: u64, n: usize, lo: f32, hi: f32| -> Vec<f32> {
            let mut x = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (0..n)
                .map(|_| {
                    x = x
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    let u = (x >> 40) as f32 / (1u64 << 24) as f32;
                    lo + (hi - lo) * u
                })
                .collect()
        };
        let decay = if adversarial {
            // Half the scalars near 0 (hard forget), half near 1 (hold) —
            // the B1 case set's decay-edge arm.
            lcg(seed ^ 0xD, p * n_head, 0.0, 1.0)
                .into_iter()
                .map(|x| {
                    if x < 0.5 {
                        1e-3 + x * 0.01
                    } else {
                        0.999 - x * 0.001
                    }
                })
                .collect()
        } else {
            lcg(seed ^ 0xD, p * n_head, 0.80, 0.999)
        };
        Self {
            qkvx: lcg(
                seed,
                p * 3 * v_dim,
                if adversarial { -0.35 } else { -0.18 },
                if adversarial { 0.35 } else { 0.18 },
            ),
            beta: lcg(
                seed ^ 0xB,
                p * n_head,
                if adversarial { 0.95 } else { 0.05 },
                if adversarial { 1.0 } else { 0.95 },
            ),
            decay,
            state: lcg(seed ^ 0x5, n_head * d * d, -0.5, 0.5),
        }
    }
}

/// The oracle's per-head scalar cursor: `(token · n_head + head)`.
#[inline]
fn scalar_idx(t: usize, head: usize, n_head: usize) -> usize {
    t * n_head + head
}

/// The sequential oracle — the MR kernel's contract in scalar Rust.
///
/// Steps in exact kernel order per (token, head, row): decay (step 1 on
/// the whole row before the retrieval — the kernel decays the registers
/// first), retrieval dot, delta, rank-1 update, output dot AFTER the
/// update, `scale = 1/√d`. Dots fold ascending-sequential
/// ([`dot_f32_ordered`]); a row-fused single pass is bit-identical only
/// if it keeps that exact per-row op sequence (the
/// `gated_deltanet_step_par_inplace` precedent).
///
/// Returns `(output, state)` with output `[p · v_dim]`, state
/// `[n_head · d · d]` — the kernel's two result buffers.
#[must_use]
pub fn sequential_oracle(
    case: &RecCase,
    n_head: usize,
    d: usize,
    p: usize,
) -> (Vec<f32>, Vec<f32>) {
    let v_dim = n_head * d;
    // The chain state: a caller-supplied carry when the case carries one
    // (the G1 fixtures do), else a zero state (sub-case windows resume
    // from the CALLER's chain — `chunked_horner` threads the updated
    // state explicitly, so the window's own state slot is empty).
    let mut state = if case.state.is_empty() {
        vec![0.0f32; n_head * d * d]
    } else {
        debug_assert_eq!(case.state.len(), n_head * d * d, "state shape");
        case.state.clone()
    };
    let mut output = vec![0.0f32; p * v_dim];
    let scale = 1.0 / (d as f32).sqrt();

    let k_base = v_dim; // k block base within a token row
    let v_base = 2 * v_dim;

    for t in 0..p {
        let t_row = t * 3 * v_dim;
        for h in 0..n_head {
            let beta_val = case.beta[scalar_idx(t, h, n_head)];
            let decay_val = case.decay[scalar_idx(t, h, n_head)];
            let head_off = h * d;
            let state_off = head_off * d;

            for row in 0..d {
                let row_off = state_off + row * d;
                let s_row = &mut state[row_off..row_off + d];

                // Step 1: decay (the row, once — kernel order).
                for s in s_row.iter_mut() {
                    *s *= decay_val;
                }
                // Step 2: retrieval dot.
                let k_head = &case.qkvx[t_row + k_base + head_off..t_row + k_base + head_off + d];
                let kv_row = dot_f32_ordered(s_row, k_head);
                // Step 3: delta.
                let delta = beta_val * (case.qkvx[t_row + v_base + head_off + row] - kv_row);
                // Step 4: rank-1 update (row only — the kernel's rows are
                // independent; the update touches `row`'s columns).
                for (c, s) in s_row.iter_mut().enumerate() {
                    *s += k_head[c] * delta;
                }
                // Step 5: output dot AFTER the update.
                let q_head = &case.qkvx[t_row + head_off..t_row + head_off + d];
                output[t * v_dim + head_off + row] = dot_f32_ordered(s_row, q_head) * scale;
            }
        }
    }
    (output, state)
}

/// The control arm — chunk bookkeeping around unchanged per-element
/// sequential chains.
///
/// Splits `[0, p)` into `ceil(p / chunk_len)` chunks; each chunk re-enters
/// the oracle's per-(token, head, row) loop with the carried state as the
/// chunk-local initial state and writes its own output slice. The
/// arithmetic per element is the oracle's EXACT op sequence (no
/// reassociation anywhere), so this MUST be bit-identical to
/// [`sequential_oracle`] at every `chunk_len` — including ragged tails
/// (`p % chunk_len != 0`) and `chunk_len ≥ p` (a single chunk). That
/// identity is the harness's own proof of life.
#[must_use]
pub fn chunked_horner(
    case: &RecCase,
    n_head: usize,
    d: usize,
    p: usize,
    chunk_len: usize,
) -> (Vec<f32>, Vec<f32>) {
    assert!(chunk_len > 0, "chunk_len must be positive");
    let v_dim = n_head * d;
    let mut state = case.state.clone();
    let mut output = vec![0.0f32; p * v_dim];

    let mut start = 0;
    while start < p {
        let end = (start + chunk_len).min(p);
        let sub = sub_case(case, start, end, v_dim, n_head);
        // The sub-case carries EMPTY state — the chain resumes from the
        // caller's updated `state`, which the oracle must ACCEPT. Seed the
        // window's chain from the carried state explicitly: zero-init the
        // oracle's chain would be WRONG (a chunk boundary must not reset
        // the recurrence). The oracle's zero-state fallback is for cold
        // starts only, so splice the carry in: run with a case whose state
        // IS the carry.
        let mut window = sub;
        window.state = state.clone();
        let (sub_out, sub_state) = sequential_oracle(&window, n_head, d, end - start);
        output[start * v_dim..end * v_dim].copy_from_slice(&sub_out);
        state = sub_state;
        start = end;
    }
    (output, state)
}

/// A window of the case as a standalone case (token offset `start`,
/// length `end − start`); the carried state is passed through unchanged —
/// the chunk-local oracle resumes exactly where the previous chunk left
/// the chain.
fn sub_case(case: &RecCase, start: usize, end: usize, v_dim: usize, n_head: usize) -> RecCase {
    let len = end - start;
    let qkvx_lo = start * 3 * v_dim;
    RecCase {
        qkvx: case.qkvx[qkvx_lo..qkvx_lo + len * 3 * v_dim].to_vec(),
        beta: case.beta[start * n_head..end * n_head].to_vec(),
        decay: case.decay[start * n_head..end * n_head].to_vec(),
        // The oracle carries no pre-step scalar state (decay folds into
        // the row loop), so the sub-case needs no state at all — the
        // oracle CLONEs this into its own chain state, which the caller
        // then discards. Empty is the honest shape: [n_head·d·d] zeros
        // would suggest a pre-step head state that does not exist.
        state: Vec::new(),
    }
}

/// The sensitivity probe — the literature O(p²/d) intra-chunk association.
///
/// Within a chunk `[a, b)`, token `t`'s state row is evaluated through the
/// EXPLICIT form: the carried contribution `(Π decays)·s_a` plus the
/// intra-chunk terms `(Π_{r<s≤t} decay_r)·u_s` summed in ascending `s`
/// order, where `u_s[col] = k_s[col]·δ_s` — mathematically the same
/// polynomial the sequential chain evaluates as Horner, but with the
/// products materialized and the sum reassociated. Floating-point
/// addition/multiplication is not associative, so this arm is expected to
/// differ from the oracle in the last ULPs on multi-term chunks; the
/// difference census is the harness's sensitivity control (a parity gate
/// that ties on a wrong-order candidate proves nothing).
///
/// The delta values `δ_s` themselves are NOT re-derived pairwise — they
/// ride the sequential chain inside each chunk (the retrieval dot makes
/// the deltas chain-dependent; a fully pairwise form would need the
/// intra-chunk causal attention term, which is the T2 design question
/// this instrument informs). The probe therefore isolates exactly one
/// reassociation: the state-row polynomial. That is the minimal
/// reassociation ANY O(p²/d) state-carry form must introduce, so a tie
/// here would mean even the full pairwise form could tie — the canary is
/// deliberately conservative.
#[must_use]
pub fn chunked_pairwise(
    case: &RecCase,
    n_head: usize,
    d: usize,
    p: usize,
    chunk_len: usize,
) -> (Vec<f32>, Vec<f32>) {
    assert!(chunk_len > 0, "chunk_len must be positive");
    let v_dim = n_head * d;
    let mut state = case.state.clone();
    let mut output = vec![0.0f32; p * v_dim];
    let scale = 1.0 / (d as f32).sqrt();
    let k_base = v_dim;

    let mut chunk_start = 0;
    while chunk_start < p {
        let chunk_end = (chunk_start + chunk_len).min(p);

        // Sequential chain INSIDE the chunk for the deltas (the
        // chain-dependent part; see the doc comment) — `chain_deltas`
        // walks the oracle's steps 1–4 per token and fills `deltas`.
        let mut deltas = vec![vec![0.0f32; n_head]; chunk_end - chunk_start];
        chain_deltas(case, n_head, d, chunk_start, chunk_end, &state, &mut deltas);

        // The pairwise state-row evaluation: per (head, row), replace the
        // chain with the explicit polynomial.
        // (Indexed head loop: `h` is head ARITHMETIC everywhere below —
        // offsets and scalar cursors — not iteration state.)
        #[allow(clippy::needless_range_loop)]
        for h in 0..n_head {
            let head_off = h * d;
            let state_off = head_off * d;
            for row in 0..d {
                let row_off = state_off + row * d;
                let mut new_row = vec![0.0f32; d];
                for (i, t) in (chunk_start..chunk_end).enumerate() {
                    let t_row = t * 3 * v_dim;
                    let decay_val = case.decay[scalar_idx(t, h, n_head)];
                    let k_head =
                        &case.qkvx[t_row + k_base + head_off..t_row + k_base + head_off + d];
                    // Decay product Π_{r ∈ (chunk_start..=t]} decay_r,
                    // materialized left-to-right.
                    let cum = if i == 0 {
                        decay_val
                    } else {
                        let mut c = decay_val;
                        for r in chunk_start..t {
                            c *= case.decay[scalar_idx(r, h, n_head)];
                        }
                        c
                    };
                    let delta = deltas[t - chunk_start][h];
                    // Carried term: only the chunk's FIRST token carries
                    // the pre-chunk state; the rest are intra-chunk terms.
                    if i == 0 {
                        for (c, nr) in new_row.iter_mut().enumerate() {
                            *nr += cum * state[row_off + c];
                        }
                    }
                    for (c, nr) in new_row.iter_mut().enumerate() {
                        let k_c = k_head[c];
                        let u = k_c * delta;
                        let w = if i == 0 {
                            1.0f32
                        } else {
                            let mut w = 1.0f32;
                            for r in t + 1..chunk_end {
                                w *= case.decay[scalar_idx(r, h, n_head)];
                            }
                            w
                        };
                        *nr += w * u;
                    }
                }
                state[row_off..row_off + d].copy_from_slice(&new_row);
                // Output dot AFTER the chunk-final update only would be
                // wrong for mid-chunk tokens — the oracle's outputs are
                // per-token, so the probe recomputes each token's output
                // from the EXPLICIT row state at t (the same pairwise
                // polynomial one token shorter).
                for t in chunk_start..chunk_end {
                    let t_row = t * 3 * v_dim;
                    let q_slice = &case.qkvx[t_row + head_off..t_row + head_off + d];
                    // The explicit state AT t (after t's update): carried
                    // + terms s ≤ t, same association.
                    let mut at_t = vec![0.0f32; d];
                    for s in chunk_start..=t {
                        let s_row = s * 3 * v_dim;
                        let s_decay = case.decay[scalar_idx(s, h, n_head)];
                        let s_k =
                            &case.qkvx[s_row + k_base + head_off..s_row + k_base + head_off + d];
                        let s_delta = deltas[s - chunk_start][h];
                        let cum_s = if s == chunk_start {
                            s_decay
                        } else {
                            let mut c = s_decay;
                            for r in chunk_start..s {
                                c *= case.decay[scalar_idx(r, h, n_head)];
                            }
                            c
                        };
                        if s == chunk_start {
                            for (c, a) in at_t.iter_mut().enumerate() {
                                *a += cum_s * state[row_off + c];
                            }
                        }
                        for (c, a) in at_t.iter_mut().enumerate() {
                            let w = {
                                let mut w = 1.0f32;
                                for r in s + 1..=t {
                                    w *= case.decay[scalar_idx(r, h, n_head)];
                                }
                                w
                            };
                            *a += w * (s_k[c] * s_delta);
                        }
                    }
                    output[t * v_dim + head_off + row] = dot_f32_ordered(&at_t, q_slice) * scale;
                }
            }
        }
        chunk_start = chunk_end;
    }
    (output, state)
}

/// Column index helper retained only as contract documentation — the flat
/// row offset is always `row · d + col` in this module (see `row_off`).
#[inline]
#[allow(dead_code)]
const fn row_of(col: usize, d: usize) -> usize {
    col % d
}

/// The sequential delta chain inside one chunk (the probe's
/// chain-dependent leg): walks the oracle's steps 1–4 per token so each
/// `delta[t][head]` matches the oracle's `δ_t` bit-for-bit, on a working
/// copy — the caller's state is never touched (the pairwise evaluation
/// reads the PRE-chunk carried state).
fn chain_deltas(
    case: &RecCase,
    n_head: usize,
    d: usize,
    start: usize,
    end: usize,
    state: &[f32],
    deltas: &mut [Vec<f32>],
) {
    let v_dim = n_head * d;
    let k_base = v_dim;
    let v_base = 2 * v_dim;
    // Working copy so the chain walk never perturbs the caller's state
    // (the pairwise evaluation reads the PRE-chunk carried state).
    let mut chain_state = state.to_vec();
    for t in start..end {
        let t_row = t * 3 * v_dim;
        // (Indexed head loop: `h` is head arithmetic — offsets, cursors,
        // and the per-head delta slot — not iteration state.)
        #[allow(clippy::needless_range_loop)]
        for h in 0..n_head {
            let head_off = h * d;
            let state_off = head_off * d;
            let beta_val = case.beta[scalar_idx(t, h, n_head)];
            let decay_val = case.decay[scalar_idx(t, h, n_head)];
            let k_head = &case.qkvx[t_row + k_base + head_off..t_row + k_base + head_off + d];
            for row in 0..d {
                let row_off = state_off + row * d;
                let s_row = &mut chain_state[row_off..row_off + d];
                for s in s_row.iter_mut() {
                    *s *= decay_val;
                }
                let kv_row = dot_f32_ordered(s_row, k_head);
                let delta = beta_val * (case.qkvx[t_row + v_base + head_off + row] - kv_row);
                deltas[t - start][h] = delta;
                for (c, s) in s_row.iter_mut().enumerate() {
                    *s += k_head[c] * delta;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const N_HEAD: usize = 4;
    const D: usize = 16; // small: CPU test budget; the CONTRACT is d-agnostic
    const P_GRID: &[usize] = &[1, 7, 16, 63, 64, 130];
    const CHUNK_GRID: &[usize] = &[1, 2, 3, 7, 16, 63, 64, 128, 512];

    /// Count of (p, chunk, adversarial) cases the control battery runs —
    /// the floor the green-zero rule reads.
    const CONTROL_CASES: usize = P_GRID.len() * CHUNK_GRID.len() * 2;

    /// T1 G1 control: the Horner arm is bit-identical at every chunk
    /// length × case — the harness's proof of life.
    #[test]
    fn chunked_horner_bit_identical_at_every_chunk_boundary() {
        let mut checked = 0usize;
        for adversarial in [false, true] {
            for &p in P_GRID {
                let case = RecCase::mirror_gpu_case(N_HEAD, D, p, 0x1236 + p as u64, adversarial);
                let (ref_out, ref_state) = sequential_oracle(&case, N_HEAD, D, p);
                assert!(
                    ref_out.iter().any(|&b| b.to_bits() != 0),
                    "oracle output all zero at p={p} adversarial={adversarial}"
                );
                for &chunk in CHUNK_GRID {
                    let (out, state) = chunked_horner(&case, N_HEAD, D, p, chunk);
                    let out_diff = out
                        .iter()
                        .zip(&ref_out)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    let st_diff = state
                        .iter()
                        .zip(&ref_state)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    assert_eq!(
                        (out_diff, st_diff),
                        (0, 0),
                        "chunked_horner chunk={chunk} p={p} adversarial={adversarial}: \
                         {out_diff} output / {st_diff} state words differ"
                    );
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, CONTROL_CASES, "the case grid must run in full");
    }

    /// T1 G1 sensitivity control: the pairwise arm MUST differ from the
    /// oracle on some multi-term chunk — the canary that keeps the gate
    /// from reading vacuous ties. (If floating-point reassociation were
    /// invisible to this fixture set, the bit-identity gate would be
    /// unable to catch a wrong-order T2 kernel.)
    #[test]
    fn pairwise_reassociation_differs_somewhere() {
        let mut differing_cases = 0usize;
        let mut multi_term_cases = 0usize;
        for adversarial in [false, true] {
            for &p in P_GRID {
                let case = RecCase::mirror_gpu_case(N_HEAD, D, p, 0x1236 + p as u64, adversarial);
                let (ref_out, ref_state) = sequential_oracle(&case, N_HEAD, D, p);
                for &chunk in CHUNK_GRID {
                    if chunk >= p {
                        continue; // single-chunk: the polynomial has the
                        // same one-carry shape; keep the census
                        // to multi-term chunks
                    }
                    multi_term_cases += 1;
                    let (out, state) = chunked_pairwise(&case, N_HEAD, D, p, chunk);
                    let diff = out
                        .iter()
                        .zip(&ref_out)
                        .chain(state.iter().zip(&ref_state))
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    if diff > 0 {
                        differing_cases += 1;
                    }
                }
            }
        }
        assert!(
            multi_term_cases > 0,
            "the sensitivity grid must contain multi-term chunks"
        );
        assert!(
            differing_cases > 0,
            "CANARY: the pairwise reassociation tied with the oracle on all \
             {multi_term_cases} multi-term cases — the bit-identity gate cannot \
             distinguish fold orders on this fixture set and is vacuous as a T2 \
             gate; widen the adversarial ranges before trusting any G1 pass"
        );
    }

    /// The chunk boundary must not perturb the ORACLE itself: feeding the
    /// oracle one token at a time (chunk_len = 1) is the identity the
    /// control arm relies on; pinned separately so a future oracle
    /// refactor that breaks single-step resumability reds here first.
    #[test]
    fn oracle_is_resumable_per_token() {
        for adversarial in [false, true] {
            let p = 33usize;
            let case = RecCase::mirror_gpu_case(N_HEAD, D, p, 0x607, adversarial);
            let (ref_out, ref_state) = sequential_oracle(&case, N_HEAD, D, p);
            let (out, state) = chunked_horner(&case, N_HEAD, D, p, 1);
            assert!(
                out.iter()
                    .zip(&ref_out)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
            assert!(
                state
                    .iter()
                    .zip(&ref_state)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
        }
    }
}
