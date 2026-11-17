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
