# Plan 638 — i1221 T4 sampling arm: pre-registration (the Bench-634 ExactQ law on the depth self-spec rider)

**Status:** OPEN — filed 2026-11-23 (zed-0a464098@amo) under the i1221 lease (the lapsed T4 pickup's unclaimed residue). Pre-registers the SAMPLING arm of the depth self-spec rider BEFORE any run: the accept/residual law, the gate battery with its bars and falsifiers, the real-pin cell's shape. The identity half (greedy EXACT) landed as Bench 1238; the TIMING rows stay window-parked (preflight-PASS only) and are NOT this plan's scope.

## Why pre-register

Bench 1238 closed the greedy arm with an explicit open row: "under SAMPLING the identity is provable (Bench-634 law) but the acceptance rate is an open measurement". A distribution-identity claim is a UQ-shaped claim — the bars must exist before the first sample is drawn, or the cell becomes a search for a statistic that passes. This plan pins: the law, the numeric posture, the statistic, N, the seeds' law, the bars, the fail-required control, and what a miss means.

## The law (what the arm claims)

- **Draft** q_i = softmax(k=1 logits at position i), T=1, untruncated; **target** p_i = softmax(k_full logits at position i), T=1, untruncated. Per-loop caches both arms (the spec-mode policy — the landed refusals carry over).
- **Accept** drafted x_i with probability `min(1, p_i(x_i)/q_i(x_i))`; on the first reject emit a draw from `normalize(max(0, p_i − q_i))` and end the round; on all-accept emit the bonus from `normalize(max(0, p_γ − q_γ))` (q_γ = the draft's next-proposal row — the Bench-634 q-capture law, at draft time).
- **Identity** (per position, no assumptions on p vs q): `P(emit x) = q(x)·min(1, p(x)/q(x)) + (1 − Σ_y min(p(y), q(y)))·resid(x) = p(x)`. Slot alignment: drafted token i is judged by the target row for position i (Bench-634's off-by-one law).
- **Lazy verify** (this arm only; the landed greedy arm is untouched): a drafted token is fed to the verify cache only once it is ACCEPTED. A reject therefore leaves the verify cache at exactly the emitted prefix (no verify truncate; only the draft cache rewinds its phantom suffix), and the cost law becomes `verify_steps == accepted + rounds`, `draft_steps == rounds·(γ+1)`. Laziness changes compute, never the emitted law (the skipped forwards produce rows no decision reads).
- **RNG**: seeded xorshift64\* → f64 in [0,1) from the top 53 bits (the repo law: never `thread_rng`; the `adapter/mint.rs` precedent). One uniform per draft draw, one per accept test, one per residual draw — a fixed consumption order, so a pinned seed replays bit-identically.
- **Numerics**: probabilities are f64 (softmax and folds sequential, ascending index — the Issue-940 order law for a deterministic instrument); Bench-634's 0.014 f32 correction-path noise is not accepted here — f64 makes the residual exact to ~1e-15 and the deterministic G2 receipt proves it.

## Gate battery (pre-registered; all pinned-seed, no tolerance bands for identity)

| id | gate | bar | falsifier / control |
|---|---|---|---|
| G1 | **policy-level TV** — the pure step on a fixed toy target over 32 tokens (peaked: p_max ≈ 0.98), two proposals: (a) mixture `0.6·p + 0.4·uniform` (acceptance ≈ 0.62, the residual path carries ≈ 38% of rounds, concentrated on the top token), (b) displaced (q's argmax ≠ p's argmax); 40,000 rounds, pinned seed | `TV(emitted, p) < 0.02` (the Bench-634 bar; a perfect sampler's expected TV here ≈ 0.0023 — 8× margin) | the **greedy-verify control** (sampled drafts judged by argmax equality — the PrefixMatch analogue) must read `TV > 0.10` (fail-required: it collapses toward the argmax) |
| G2 | **induced-law exactness** (deterministic, no sampling): `induced = q·min(1, p/q) + (1 − Σmin(p,q))·resid` equals `p` — on 200 seeded random (p,q) pairs and on every (p,q) row pair the E2E rollouts capture | `max_x |induced(x) − p(x)| ≤ 1e-12` | none needed (an algebra identity); a miss is an implementation defect, never noise |
| G3 | **E2E identity through the real forward** (synth 6-block, k=2, V=11): first-token draws, ref = direct sampler on the k_full row, spec = the sampler at γ ∈ {1, 4}; N = 200,000 per arm; `p_true` = the f64 softmax of the k_full row (known exactly) | `TV(spec_hist, p_true) < 0.02` AND `TV(ref_hist, p_true) < 0.02` (the floor; uniform-worst-case expectation at this N, V is ≈ 0.006) | the greedy-verify control `TV > 0.10` (fail-required) |
| G4 | **determinism + structure**: two passes at one seed are bit-identical (streams + counters); different seeds differ; `verify_steps == accepted + rounds`; `draft_steps == rounds·(γ+1)`; both caches exit at `prompt + n` | exact | — |
| G5 | **posture refusals**: LastLoop verify, partial-depth verify, non-k=1 draft refuse loud before any decode | exact | — |

G3 runs as an `#[ignore]`d `--release` cell: N is sized for the bar (200,000), not for a debug run, and at a reduced N the uniform-worst-case expectation sits too near 0.02 to be a bar — so there is NO reduced-N always-on twin of G3. The always-on battery is G1 + G2 + G4 + G5; G3's skip line names the release command.

## The real-pin cell (report-only + receipts; structural — no timing rows)

`examples/i1221_t4_sampling_cell.rs`, resumable JSONL per plan 636 (header lineage = pin blake3 + STAMP + prompt law + seed; mismatch refuses). On the Q5_K_M pin at k_full=4, T=1:

1. **Induced-law receipt** — along one sampled rollout per (prompt, γ), every captured (p_i, q_i) pair passes G2's bar (1e-12). This is the real pin's identity evidence: exact, load-independent, no sampling noise.
2. **Acceptance census** (the headline open measurement): rounds, accepted/drafted, rejects by position, bonus, tokens/round, the counter law — 2 prompts × γ ∈ {2, 4} × n = 12.
3. **Determinism receipt** — two passes, one seed: stream digests equal.
4. **First-token two-sample table** — N draws per arm on the CAPTURED first-position rows (the forward runs once; the draws are pure sampler steps): reported with its noise expectation, never gated (at any affordable N the real pin's 131k-vocab TV is noise-dominated — disclosed, not hidden).

Window discipline (plan 636): structural, so it may run concurrent-with-light-load — the concurrency is DISCLOSED in the header, and no wall number is quoted. The TIMING rows of Bench 1238 stay parked for a preflight-PASS window; this plan does not un-park them.

## Tasks

- [ ] T1 — the instrument: `src/dory/self_spec.rs` gains the sampling arm (`self_spec_sample`, the pure step + induced law + the f64 softmax/categorical helpers, the xorshift RNG) behind the existing `dory` feature; greedy arm untouched; module doc carries the law + the lazy-verify cost law.
- [ ] T2 — the gate battery `tests/i1221_t4_sampling_gates.rs` (G1, G2, G4, G5 always-on; G3 `#[ignore]` release-only) + the P43 `[[test]]` row in the same commit.
- [ ] T3 — the real-pin cell example + its P43 `[[example]]` row; the run (structural, disclosed concurrency); the bench record (number allocated at write time); issue 1221 T4 row + lease fold.
- [ ] T4 — clippy at default / no-default / `dory` postures; rustfmt on touched files; commit + push.

## What a miss means

- G1/G3 control does NOT fail (TV ≤ 0.10): the control is broken, the battery has no teeth — fix the control before reading any arm.
- G1 or G3 spec arm fails: an implementation defect in the step (slot alignment, the bonus row's q, a stale residual) — bisect with G2 (G2 green + G1 red ⇒ the sampling harness, not the law).
- Real-pin G2 receipt fails: f32 logits → f64 conversion or a non-finite row — refuse the cell, never quote the census.
