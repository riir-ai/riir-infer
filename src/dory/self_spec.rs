//! Depth self-spec decode (issue 1221 T4 — the paper §6.2 rider).
//!
//! k=1 draft, full-depth verify, greedy EXACT rejection sampling. The
//! pre-registered gate: the emitted stream is token-identical to the
//! k=4 AR greedy reference under per-loop caches — because every
//! emitted token equals the k=4 target argmax at its position over
//! exactly the emitted prefix (the verify cache carries the same
//! prefix the reference carries, so the conditionals are the same
//! rows by the T1 prefill≡decode identity + forward determinism; the
//! draft only proposes and can never decide an emit).
//!
//! Round structure (one round emits up to γ+1 tokens):
//! 1. DRAFT — the k=1 cache chains γ greedy picks (its own rollout;
//!    proposal quality only).
//! 2. VERIFY — the full-depth cache replays the γ drafted tokens; row
//!    `vrows[i]` is the target logits after drafted token `i`.
//! 3. ACCEPT WALK — `t[0]` is decided by the round's decider row (the
//!    target logits of the previous emitted token); accepting `t[i]`
//!    hands `vrows[i]` to `t[i+1]`. A drafted token equal to the
//!    target argmax is accepted verbatim; on the first mismatch the
//!    TARGET ARGMAX is emitted instead and the round ends. If all γ
//!    accept, the last verify row's argmax is emitted free (the bonus
//!    token).
//! 4. ROUND-END REPAIR — the last emitted token is fed to BOTH caches:
//!    verify yields the next round's decider row; draft absorbs it (and
//!    a rejected round first rewinds the draft cache past its phantom
//!    suffix via [`DoryKvCache::truncate`], keeping it aligned with the
//!    emitted prefix).
//!
//! Cost law (sequential hardware — honest expectation, recorded in the
//! T4 cell): per emitted token ≈ one k=1 draft step + one k=4 verify
//! step ≈ 1.25× the k=4 AR reference's compute. Self-spec is a
//! LATENCY structure (draft/verify pipelining in a served posture),
//! never a compute win on one thread; the cell records the ratio, the
//! gate asserts the identity.
//!
//! # The sampling arm (plan 638 — the Bench-634 ExactQ law)
//!
//! Under SAMPLING the emitted stream is a draw from the k_full target
//! model's own distribution — a law, not a statistic: per position,
//! `P(emit x) = min(p(x), q(x)) + (1 − Σ_y min(p(y), q(y)))·resid(x)`
//! `= p(x)` for ANY target p and proposal q ([`induced_into`] is that
//! identity's instrument; `resid = max(0, p − q) / Z`). One round (up
//! to γ+1 tokens), sharing the greedy arm's boot and repair:
//!
//! 1. DRAFT — the k=1 chain SAMPLES γ tokens, each from the softmax of
//!    its own k=1 row (q captured at draft time — the Bench-634
//!    q-capture law). The chain feeds a drafted token only to reach
//!    the next row, so the LAST drafted token is not fed here (the
//!    deferred tail feed, paid in the repair).
//! 2. JUDGE — drafted token i is tested against the target row for
//!    position i (slot-aligned) with a fresh uniform; an ACCEPT feeds
//!    it to the verify cache to obtain the next target row (LAZY
//!    verify — a reject never runs the forwards its decision does not
//!    read). The first reject emits a draw from the residual and ends
//!    the round.
//! 3. BONUS — an all-accepted round samples one extra token from the
//!    last target row DIRECTLY (Leviathan Alg. 1's n = γ+1 branch:
//!    exact by construction — nobody proposes for that position).
//! 4. REPAIR — an all-accepted round pays its deferred tail feed; a
//!    rejected round's phantom suffix is rewound; the last emitted
//!    token is then fed to BOTH caches, exactly as in the greedy arm.
//!
//! Cost law of this arm (the counter identities the gate battery
//! pins): `drafted == rounds·γ`, `accepted + rejected + bonus ==
//! emitted` (uncapped), `verify_steps == accepted + rounds`,
//! `draft_steps == rounds·γ + bonus`. Laziness changes compute, never
//! the emitted law.

use std::fmt;

use super::forward::{DoryCachePolicy, DoryForward, DoryForwardError, DoryKvCache};
use super::spectra::argmax_and_margin;

/// Per-run counters (the throughput/cost cell's rows read these; never
/// a gate — report-only, per plan 636's law).
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct SelfSpecStats {
    pub rounds: usize,
    /// Drafted tokens (γ per round).
    pub drafted: usize,
    /// Drafted tokens accepted verbatim.
    pub accepted: usize,
    /// Free final tokens from all-accepted rounds.
    pub bonus: usize,
    /// Tokens emitted in total (accepted + rejected-target + bonus),
    /// after the `n` cap.
    pub emitted: usize,
    /// k=1 decode steps (draft chains + round-end repairs).
    pub draft_steps: usize,
    /// k_full decode steps (verify replays + round-end decider rows).
    pub verify_steps: usize,
}

/// Greedy k_full AR decode — the reference arm. `cache` is cleared by
/// the boot prefill; per-loop policy is the caller's posture (the T4
/// gate uses PerLoop both arms).
pub fn ar_greedy(
    fwd: &DoryForward<'_>,
    prompt: &[u32],
    n: usize,
    cache: &mut DoryKvCache,
) -> Result<Vec<u32>, DoryForwardError> {
    let mut rows = fwd.prefill(prompt, cache)?;
    let mut last = rows.pop().expect("non-empty prompt");
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let t = argmax_and_margin(&last).0 as u32;
        out.push(t);
        last = fwd.decode_step(t, cache)?;
    }
    Ok(out)
}

/// Greedy EXACT depth self-spec. `verify_cache` must be a PER-LOOP
/// cache at the checkpoint's full depth (the spec-mode policy — a
/// last-loop or partial-depth verify would change the target rows and
/// is refused loud); `draft_cache` must be at k=1. Both are cleared by
/// the boot prefills. Returns the emitted stream (capped at `n`) and
/// the round counters.
pub fn self_spec_greedy(
    fwd: &DoryForward<'_>,
    prompt: &[u32],
    n: usize,
    gamma: usize,
    verify_cache: &mut DoryKvCache,
    draft_cache: &mut DoryKvCache,
) -> Result<(Vec<u32>, SelfSpecStats), DoryForwardError> {
    assert!(gamma >= 1, "self-spec gamma must be >= 1");
    assert_spec_posture(fwd, verify_cache, draft_cache);

    let mut stats = SelfSpecStats::default();

    // Boot: the verify cache carries the prompt; its last row is the
    // first round's decider. The k=1 boot prefill arms the draft chain.
    let mut vrows = fwd.prefill(prompt, verify_cache)?;
    let mut decider = vrows.pop().expect("non-empty prompt");
    let mut drows = fwd.prefill(prompt, draft_cache)?;
    let mut dlast = drows.pop().expect("non-empty prompt");

    let mut emitted: Vec<u32> = Vec::new();
    while emitted.len() < n {
        stats.rounds += 1;

        // 1. DRAFT — γ greedy picks on the k=1 chain.
        stats.drafted += gamma;
        let mut drafted = Vec::with_capacity(gamma);
        for _ in 0..gamma {
            let t = argmax_and_margin(&dlast).0 as u32;
            drafted.push(t);
            dlast = fwd.decode_step(t, draft_cache)?;
            stats.draft_steps += 1;
        }

        // 2. VERIFY — the full-depth replay of the drafted suffix.
        let mut proposed_rows = Vec::with_capacity(gamma);
        for &t in &drafted {
            proposed_rows.push(fwd.decode_step(t, verify_cache)?);
            stats.verify_steps += 1;
        }

        // 3. ACCEPT WALK — greedy exact: a drafted token is accepted
        // iff it equals the target argmax that decides it; the first
        // mismatch emits the target argmax and ends the round.
        let mut accepted_here = 0usize;
        let mut round_reject: Option<u32> = None;
        for (i, &t) in drafted.iter().enumerate() {
            let tgt = argmax_and_margin(&decider).0 as u32;
            if t == tgt {
                emitted.push(t);
                accepted_here += 1;
                // This row decides the next drafted token (and, at the
                // end, the bonus token).
                decider = std::mem::take(&mut proposed_rows[i]);
            } else {
                emitted.push(tgt);
                round_reject = Some(tgt);
                break;
            }
        }
        stats.accepted += accepted_here;

        // The round's last emission: a reject emits the target argmax
        // in the walk; an all-accepted round emits the last verify
        // row's argmax free (the bonus token).
        let last_emitted = match round_reject {
            Some(t) => t,
            None => {
                let bonus = argmax_and_margin(&decider).0 as u32;
                emitted.push(bonus);
                stats.bonus += 1;
                bonus
            }
        };

        // 4. ROUND-END REPAIR — both caches rewind to the emitted
        // prefix (a rejected round leaves a phantom drafted suffix in
        // BOTH: the verify replay fed every drafted token before the
        // accept walk ran) and the last emitted token re-arms them:
        // verify returns the next round's decider row; draft returns
        // the next round's first draft row. On an all-accepted round
        // the truncates are exact no-ops.
        let keep = prompt.len() + emitted.len() - 1;
        verify_cache.truncate(keep);
        draft_cache.truncate(keep);
        decider = fwd.decode_step(last_emitted, verify_cache)?;
        stats.verify_steps += 1;
        dlast = fwd.decode_step(last_emitted, draft_cache)?;
        stats.draft_steps += 1;
    }

    emitted.truncate(n);
    stats.emitted = emitted.len();
    Ok((emitted, stats))
}

/// The spec-mode posture, shared by both arms: a PER-LOOP verify cache
/// at the checkpoint's full depth (the paper's spec-mode policy — a
/// last-loop or partial-depth verify would change the target rows) and
/// a k=1 draft cache. Refuses loud before any decode work.
fn assert_spec_posture(
    fwd: &DoryForward<'_>,
    verify_cache: &DoryKvCache,
    draft_cache: &DoryKvCache,
) {
    assert!(
        verify_cache.k_loops() == fwd.k_loops(),
        "verify cache k {} != the checkpoint's full depth {} (the target rows must be the full-depth forward)",
        verify_cache.k_loops(),
        fwd.k_loops()
    );
    assert!(
        verify_cache.policy() == DoryCachePolicy::PerLoop,
        "the self-spec verify posture is PER-LOOP caches (the paper's spec-mode policy); got {:?} — build the cache with DoryKvCache::new (or ::with_policy(.., PerLoop))",
        verify_cache.policy()
    );
    assert!(
        draft_cache.k_loops() == 1,
        "the depth self-spec draft is the k=1 posture; got k {}",
        draft_cache.k_loops()
    );
}

// ────────────────────── the sampling arm (plan 638) ──────────────────────

/// A sampling-arm failure. [`Self::Forward`] carries the forward's own
/// refusal (an out-of-range token); [`Self::NonFiniteLogits`] means a
/// model row was not a distribution (a NaN/±inf logit) — the sampler
/// refuses rather than emit from a non-law.
#[derive(Debug)]
pub enum SpecSampleError {
    Forward(DoryForwardError),
    NonFiniteLogits {
        /// Position of the emitted stream the row decides.
        position: usize,
        /// Which seam produced the row.
        source: &'static str,
    },
}

impl fmt::Display for SpecSampleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Forward(e) => write!(f, "self-spec sample forward: {e}"),
            Self::NonFiniteLogits { position, source } => write!(
                f,
                "self-spec sample: the {source} logits row deciding position {position} is not a distribution (a non-finite logit)"
            ),
        }
    }
}

impl std::error::Error for SpecSampleError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Forward(e) => Some(e),
            Self::NonFiniteLogits { .. } => None,
        }
    }
}

impl From<DoryForwardError> for SpecSampleError {
    fn from(e: DoryForwardError) -> Self {
        Self::Forward(e)
    }
}

/// Seeded xorshift64\* → f64 uniforms in [0, 1) from the top 53 bits
/// (the repo's determinism law: never `thread_rng`; the `adapter/mint`
/// precedent). The seed is scrambled through splitmix64's finalizer so
/// neighbouring seeds give unrelated streams. The consumption order is
/// part of the contract — per drafted token one draw; per judged
/// position two (the accept test and the residual's, the latter spent
/// even when an accept leaves it unused — an unconditional order, no
/// branch); per bonus one — so a pinned seed replays bit-identically.
#[derive(Debug, Clone)]
pub struct SpecRng(u64);

impl SpecRng {
    pub fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        // xorshift64* must start non-zero.
        Self((z ^ (z >> 31)) | 1)
    }

    /// The next raw 64-bit word.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// The next uniform in [0, 1) — 53 uniform bits.
    pub fn next_f64(&mut self) -> f64 {
        const INV_2_53: f64 = 1.0 / (1u64 << 53) as f64;
        (self.next_u64() >> 11) as f64 * INV_2_53
    }
}

/// `softmax(logits · inv_temperature)` in f64 into `out`: the max is
/// subtracted first, every fold sequential and ascending (the Issue-940
/// order law for a deterministic instrument). Callers refuse non-finite
/// rows before this (the arm's [`SpecSampleError::NonFiniteLogits`]).
pub fn softmax_f64_into(logits: &[f32], inv_temperature: f64, out: &mut [f64]) {
    assert_eq!(logits.len(), out.len(), "softmax row width");
    let Some((&first, _)) = logits.iter().zip(out.iter_mut()).next() else {
        return;
    };
    let m = f64::from(logits.iter().copied().fold(first, f32::max)) * inv_temperature;
    let mut s = 0.0f64;
    for (o, &l) in out.iter_mut().zip(logits.iter()) {
        let e = (f64::from(l) * inv_temperature - m).exp();
        *o = e;
        s += e;
    }
    let inv = 1.0 / s;
    for o in out.iter_mut() {
        *o *= inv;
    }
}

/// Draw an index from a mass vector with a uniform in [0, 1): the
/// first index whose ascending fold exceeds `u`. A `u` past the folded
/// total (rounding) lands on the last index with mass.
pub fn sample_categorical(probs: &[f64], u: f64) -> usize {
    let mut acc = 0.0f64;
    for (i, &m) in probs.iter().enumerate() {
        acc += m;
        if u < acc {
            return i;
        }
    }
    probs.iter().rposition(|&m| m > 0.0).unwrap_or(0)
}

/// `out = max(0, p − q) / Z`, returning the residual mass `Z` — the
/// target-minus-proposal tail a rejected draw comes from.
pub fn residual_into(p: &[f64], q: &[f64], out: &mut [f64]) -> f64 {
    assert_eq!(p.len(), q.len(), "residual width");
    assert_eq!(p.len(), out.len(), "residual out width");
    let mut z = 0.0f64;
    for ((o, &a), &b) in out.iter_mut().zip(p.iter()).zip(q.iter()) {
        let d = if a > b { a - b } else { 0.0 };
        *o = d;
        z += d;
    }
    if z > 0.0 {
        let inv = 1.0 / z;
        for o in out.iter_mut() {
            *o *= inv;
        }
    }
    z
}

/// One judge step: drafted token `drafted` was drawn from `q`; it is
/// ACCEPTED with probability `min(1, p(drafted)/q(drafted))`, else the
/// emission is a draw from the residual `max(0, p − q)` (in
/// `resid_scratch`). Returns the emitted token and whether the draft
/// was accepted.
pub fn spec_step(
    p: &[f64],
    q: &[f64],
    drafted: usize,
    u_accept: f64,
    u_residual: f64,
    resid_scratch: &mut [f64],
) -> (usize, bool) {
    let (pm, qm) = (p[drafted], q[drafted]);
    // u < min(1, p/q)  ⇔  u·q < min(q, p)   (q > 0); a zero proposal
    // mass can never be drafted from — read as an infinite ratio.
    let accept = if qm > 0.0 {
        u_accept * qm < pm.min(qm)
    } else {
        true
    };
    if accept {
        return (drafted, true);
    }
    // A reject implies p ≠ q, so the residual carries mass; the
    // target-direct fallback is exact too and keeps the step total.
    let emitted = if residual_into(p, q, resid_scratch) > 0.0 {
        sample_categorical(resid_scratch, u_residual)
    } else {
        sample_categorical(p, u_residual)
    };
    (emitted, false)
}

/// The INDUCED emitted law of one judge step, from the sampler's own
/// primitives: `min(p, q)` (the accept mass `q·min(1, p/q)`) plus the
/// reject mass `1 − Σ min(p, q)` carried by the residual. Equals `p`
/// exactly — the identity the sampling arm claims; the gates compare
/// the two. `scratch` is caller-owned at the row width.
pub fn induced_into(p: &[f64], q: &[f64], out: &mut [f64], scratch: &mut [f64]) {
    residual_into(p, q, scratch);
    let mut accept_mass = 0.0f64;
    for ((o, &a), &b) in out.iter_mut().zip(p.iter()).zip(q.iter()) {
        let v = a.min(b);
        *o = v;
        accept_mass += v;
    }
    let reject_mass = 1.0 - accept_mass;
    if reject_mass > 0.0 {
        for (o, &r) in out.iter_mut().zip(scratch.iter()) {
            *o += reject_mass * r;
        }
    }
}

/// The (p, q) mass pair every judge step read — the cell's receipt
/// rows (and the real pin's amortized draw tables: the forward runs
/// once, the draws are pure steps on captured rows).
#[derive(Debug, Default)]
pub struct SampleCapture {
    /// Target mass vectors, in judged order.
    pub p: Vec<Vec<f64>>,
    /// Proposal mass vectors, slot-aligned with `p`.
    pub q: Vec<Vec<f64>>,
}

impl SampleCapture {
    fn push(&mut self, p: &[f64], q: &[f64]) {
        self.p.push(p.to_vec());
        self.q.push(q.to_vec());
    }

    /// The captured pairs, slot-aligned.
    pub fn pairs(&self) -> impl Iterator<Item = (&[f64], &[f64])> {
        self.p
            .iter()
            .zip(self.q.iter())
            .map(|(a, b)| (a.as_slice(), b.as_slice()))
    }
}

/// Per-run counters of the sampling arm (report-only; the counter
/// identities are pinned by the gate battery, never asserted here).
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct SelfSpecSampleStats {
    /// The round counters both arms share (`bonus` = all-accepted
    /// rounds; `emitted` = after the cap).
    #[serde(flatten)]
    pub base: SelfSpecStats,
    /// Rounds that ended in a reject (== residual draws).
    pub rejected: usize,
    /// Tokens emitted before the cap (`accepted + rejected + bonus`).
    pub emitted_uncapped: usize,
    /// Rejects by drafted position (length γ).
    pub rejects_by_pos: Vec<usize>,
    /// Uniforms consumed (the determinism contract's witness).
    pub uniforms: usize,
}

/// The sampling arm's knobs: the temperature both arms share (the
/// identity is temperature-invariant — T scales p and q alike), the
/// seeded stream, and an optional capture of the judge rows.
pub struct SampleOpts<'a> {
    pub temperature: f64,
    pub rng: &'a mut SpecRng,
    pub capture: Option<&'a mut SampleCapture>,
}

/// Sampled k_full AR decode — the sampling arm's reference: every
/// token is a draw from the softmax of the k_full row at `temperature`.
/// The cache exits at `prompt + n` (the reference arm of [`ar_greedy`]).
pub fn ar_sample(
    fwd: &DoryForward<'_>,
    prompt: &[u32],
    n: usize,
    temperature: f64,
    cache: &mut DoryKvCache,
    rng: &mut SpecRng,
) -> Result<Vec<u32>, SpecSampleError> {
    assert_temperature(temperature);
    let inv_t = 1.0 / temperature;
    let mut rows = fwd.prefill(prompt, cache)?;
    let mut last = finite_row(rows.pop().expect("non-empty prompt"), 0, "ar-sample")?;
    let mut p = vec![0.0f64; last.len()];
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        softmax_f64_into(&last, inv_t, &mut p);
        let t = sample_categorical(&p, rng.next_f64()) as u32;
        out.push(t);
        last = finite_row(fwd.decode_step(t, cache)?, out.len(), "ar-sample")?;
    }
    Ok(out)
}

/// Depth self-spec SAMPLING (plan 638): k=1 draft, full-depth lazy
/// verify, the Bench-634 ExactQ accept/residual law — the emitted
/// stream is distributed exactly as [`ar_sample`] at the same
/// temperature. Same posture refusals as [`self_spec_greedy`]. Returns
/// the stream (capped at `n`) and the counters; both caches exit at
/// `prompt + n`.
pub fn self_spec_sample(
    fwd: &DoryForward<'_>,
    prompt: &[u32],
    n: usize,
    gamma: usize,
    verify_cache: &mut DoryKvCache,
    draft_cache: &mut DoryKvCache,
    opts: &mut SampleOpts<'_>,
) -> Result<(Vec<u32>, SelfSpecSampleStats), SpecSampleError> {
    sample_core(
        fwd,
        prompt,
        n,
        gamma,
        verify_cache,
        draft_cache,
        opts,
        Judge::Exact,
    )
}

/// The fail-required CONTROL (the Bench-634 PrefixMatch analogue):
/// SAMPLED drafts judged by argmax equality — a drafted token is
/// accepted iff it is the target row's argmax, else that argmax is
/// emitted; the bonus is the last target row's argmax. Every emission
/// is therefore a target argmax: the output is a point mass at the
/// mode (TV = 1 − p_max from the target) — the battery's proof that a
/// greedy judge of sampled drafts is NOT distribution-preserving.
pub fn self_spec_greedy_verify_control(
    fwd: &DoryForward<'_>,
    prompt: &[u32],
    n: usize,
    gamma: usize,
    verify_cache: &mut DoryKvCache,
    draft_cache: &mut DoryKvCache,
    rng: &mut SpecRng,
) -> Result<(Vec<u32>, SelfSpecSampleStats), SpecSampleError> {
    let mut opts = SampleOpts {
        temperature: 1.0,
        rng,
        capture: None,
    };
    sample_core(
        fwd,
        prompt,
        n,
        gamma,
        verify_cache,
        draft_cache,
        &mut opts,
        Judge::GreedyControl,
    )
}

/// How a drafted token is judged.
#[derive(Clone, Copy)]
enum Judge {
    /// `min(1, p/q)` accept, residual on reject — the law.
    Exact,
    /// Argmax equality, target argmax on mismatch — the control.
    GreedyControl,
}

fn assert_temperature(temperature: f64) {
    assert!(
        temperature.is_finite() && temperature > 0.0,
        "temperature must be finite and > 0; got {temperature}"
    );
}

fn finite_row(
    row: Vec<f32>,
    position: usize,
    source: &'static str,
) -> Result<Vec<f32>, SpecSampleError> {
    if row.iter().all(|l| l.is_finite()) {
        Ok(row)
    } else {
        Err(SpecSampleError::NonFiniteLogits { position, source })
    }
}

/// First-index argmax of a mass vector (the house strict-`>` tie rule).
fn argmax_mass(p: &[f64]) -> usize {
    let mut best = 0usize;
    let mut best_v = f64::NEG_INFINITY;
    for (i, &v) in p.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

/// The shared round machinery of both sampling postures (draft chain,
/// lazy verify, bonus, repair); `judge` picks the law.
#[allow(clippy::too_many_arguments)]
fn sample_core(
    fwd: &DoryForward<'_>,
    prompt: &[u32],
    n: usize,
    gamma: usize,
    verify_cache: &mut DoryKvCache,
    draft_cache: &mut DoryKvCache,
    opts: &mut SampleOpts<'_>,
    judge: Judge,
) -> Result<(Vec<u32>, SelfSpecSampleStats), SpecSampleError> {
    assert!(gamma >= 1, "self-spec gamma must be >= 1");
    assert_temperature(opts.temperature);
    assert_spec_posture(fwd, verify_cache, draft_cache);
    let inv_t = 1.0 / opts.temperature;
    let exact = matches!(judge, Judge::Exact);

    let mut stats = SelfSpecSampleStats {
        rejects_by_pos: vec![0; gamma],
        ..Default::default()
    };

    // Boot: the verify cache's last row is the first judge's target
    // row; the k=1 boot arms the draft chain.
    let mut vrows = fwd.prefill(prompt, verify_cache)?;
    let mut decider = finite_row(vrows.pop().expect("non-empty prompt"), 0, "boot-verify")?;
    let mut drows = fwd.prefill(prompt, draft_cache)?;
    let mut dlast = finite_row(drows.pop().expect("non-empty prompt"), 0, "boot-draft")?;
    let vocab = decider.len();
    assert_eq!(dlast.len(), vocab, "draft/target row width");

    // Scratch, sized once (the reuse-buffers discipline).
    let mut p_buf = vec![0.0f64; vocab];
    let mut r_buf = vec![0.0f64; vocab];
    let mut q_buf = vec![0.0f64; vocab * gamma];
    let mut drafted = vec![0u32; gamma];
    let mut emitted: Vec<u32> = Vec::with_capacity(n);

    while emitted.len() < n {
        stats.base.rounds += 1;
        stats.base.drafted += gamma;

        // 1. DRAFT — γ sampled proposals, each from its own k=1 row.
        for i in 0..gamma {
            let q = &mut q_buf[i * vocab..(i + 1) * vocab];
            softmax_f64_into(&dlast, inv_t, q);
            let t = sample_categorical(q, opts.rng.next_f64()) as u32;
            stats.uniforms += 1;
            drafted[i] = t;
            // The tail feed is deferred to the repair (its row proposes
            // nothing: the bonus is target-direct).
            if i + 1 < gamma {
                dlast = finite_row(
                    fwd.decode_step(t, draft_cache)?,
                    emitted.len() + i + 1,
                    "draft",
                )?;
                stats.base.draft_steps += 1;
            }
        }

        // 2. JUDGE — lazy verify: `decider` is the target row for
        // position i; only an ACCEPT feeds the drafted token to reach
        // position i+1.
        let mut all_accept = true;
        for i in 0..gamma {
            let q = &q_buf[i * vocab..(i + 1) * vocab];
            softmax_f64_into(&decider, inv_t, &mut p_buf);
            if exact {
                if let Some(c) = opts.capture.as_mut() {
                    c.push(&p_buf, q);
                }
                let u_accept = opts.rng.next_f64();
                let u_residual = opts.rng.next_f64();
                stats.uniforms += 2;
                let (x, accepted) = spec_step(
                    &p_buf,
                    q,
                    drafted[i] as usize,
                    u_accept,
                    u_residual,
                    &mut r_buf,
                );
                if accepted {
                    accept_emit(x as u32, &mut emitted, &mut stats);
                } else {
                    reject_emit(x as u32, i, &mut emitted, &mut stats, &mut all_accept);
                    break;
                }
            } else {
                let tgt = argmax_mass(&p_buf) as u32;
                if drafted[i] == tgt {
                    accept_emit(tgt, &mut emitted, &mut stats);
                } else {
                    reject_emit(tgt, i, &mut emitted, &mut stats, &mut all_accept);
                    break;
                }
            }
            // The next target row.
            let fed = *emitted.last().expect("accepted emits");
            decider = finite_row(fwd.decode_step(fed, verify_cache)?, emitted.len(), "verify")?;
            stats.base.verify_steps += 1;
        }

        // 3. BONUS — an all-accepted round emits from the last target
        // row: a draw under the law, the argmax under the control.
        if all_accept {
            softmax_f64_into(&decider, inv_t, &mut p_buf);
            let x = if exact {
                stats.uniforms += 1;
                sample_categorical(&p_buf, opts.rng.next_f64()) as u32
            } else {
                argmax_mass(&p_buf) as u32
            };
            emitted.push(x);
            stats.base.bonus += 1;
        }

        // 4. REPAIR — an all-accepted round pays its deferred tail
        // feed (the draft cache must reach the emitted prefix); a
        // rejected round's phantom suffix is rewound; then the last
        // emitted token re-arms both caches, as in the greedy arm.
        if all_accept {
            let _unused = fwd.decode_step(drafted[gamma - 1], draft_cache)?;
            stats.base.draft_steps += 1;
        }
        let keep = prompt.len() + emitted.len() - 1;
        // The lazy-verify invariant: the verify cache sits at exactly
        // the emitted prefix (accepted feeds only — a reject fed
        // nothing), so it never needs the rewind the draft cache does.
        debug_assert_eq!(
            verify_cache.len(),
            keep,
            "lazy verify: the verify cache must sit at the emitted prefix"
        );
        draft_cache.truncate(keep);
        debug_assert_eq!(
            draft_cache.len(),
            keep,
            "the draft rewind must land on the prefix"
        );
        let last = *emitted.last().expect("a round always emits");
        decider = finite_row(
            fwd.decode_step(last, verify_cache)?,
            emitted.len(),
            "verify-repair",
        )?;
        stats.base.verify_steps += 1;
        dlast = finite_row(
            fwd.decode_step(last, draft_cache)?,
            emitted.len(),
            "draft-repair",
        )?;
        stats.base.draft_steps += 1;
    }

    // The cap: a round may overshoot `n`; the caches exit aligned with
    // the EMITTED prefix.
    emitted.truncate(n);
    verify_cache.truncate(prompt.len() + n);
    draft_cache.truncate(prompt.len() + n);
    stats.base.emitted = emitted.len();
    stats.emitted_uncapped = stats.base.accepted + stats.rejected + stats.base.bonus;
    Ok((emitted, stats))
}

fn accept_emit(x: u32, emitted: &mut Vec<u32>, stats: &mut SelfSpecSampleStats) {
    emitted.push(x);
    stats.base.accepted += 1;
}

fn reject_emit(
    x: u32,
    pos: usize,
    emitted: &mut Vec<u32>,
    stats: &mut SelfSpecSampleStats,
    all_accept: &mut bool,
) {
    emitted.push(x);
    stats.rejected += 1;
    stats.rejects_by_pos[pos] += 1;
    *all_accept = false;
}
