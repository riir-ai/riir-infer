# Plan 638 — i1221 T4 sampling arm: pre-registration (the Bench-634 ExactQ law on the depth self-spec rider)

**Status:** DONE — T1–T4 all LANDED (T3's cell closed 2026-11-25 as Bench 1260 under the re-claim; see the tasks block). Pre-registers the SAMPLING arm of the depth self-spec rider BEFORE any run: the accept/residual law, the gate battery with its bars and falsifiers, the real-pin cell's shape. The identity half (greedy EXACT) landed as Bench 1238; the TIMING rows stay window-parked (preflight-PASS only) and are NOT this plan's scope.

## Why pre-register

Bench 1238 closed the greedy arm with an explicit open row: "under SAMPLING the identity is provable (Bench-634 law) but the acceptance rate is an open measurement". A distribution-identity claim is a UQ-shaped claim — the bars must exist before the first sample is drawn, or the cell becomes a search for a statistic that passes. This plan pins: the law, the numeric posture, the statistic, N, the seeds' law, the bars, the fail-required control, and what a miss means.

## The law (what the arm claims)

- **Draft** q_i = softmax(k=1 logits at position i), T=1, untruncated; **target** p_i = softmax(k_full logits at position i), T=1, untruncated. Per-loop caches both arms (the spec-mode policy — the landed refusals carry over).
- **Accept** drafted x_i with probability `min(1, p_i(x_i)/q_i(x_i))`; on the first reject emit a draw from `normalize(max(0, p_i − q_i))` and end the round; on all-accept sample the bonus from the last target row `p_γ` directly (Leviathan Alg. 1's n = γ+1 branch — see the corrections below).
- **Identity** (per position, no assumptions on p vs q): `P(emit x) = min(p(x), q(x)) + (1 − Σ_y min(p(y), q(y)))·resid(x) = p(x)`. Slot alignment: drafted token i is judged by the target row for position i (Bench-634's off-by-one law).
- **Lazy verify** (this arm only; the landed greedy arm is untouched): a drafted token is fed to the verify cache only once it is ACCEPTED. A reject therefore leaves the verify cache at exactly the emitted prefix (no verify truncate; only the draft cache rewinds its phantom suffix), and the cost law becomes `verify_steps == accepted + rounds`, `draft_steps == rounds·γ + bonus`. Laziness changes compute, never the emitted law (the skipped forwards produce rows no decision reads).
- **RNG**: seeded xorshift64\* → f64 in [0,1) from the top 53 bits (the repo law: never `thread_rng`; the `adapter/mint.rs` precedent). One uniform per draft draw, one per accept test, one per residual draw — a fixed consumption order, so a pinned seed replays bit-identically.
- **Numerics**: probabilities are f64 (softmax and folds sequential, ascending index — the Issue-940 order law for a deterministic instrument); Bench-634's 0.014 f32 correction-path noise is not accepted here — f64 makes the residual exact to ~1e-15 and the deterministic G2 receipt proves it.

## Gate battery (pre-registered; all pinned-seed, no tolerance bands for identity)

| id | gate | bar | falsifier / control |
|---|---|---|---|
| G1 | **policy-level TV** — the pure step on a fixed toy target over 32 tokens (p_max ≈ 0.52), two proposals: (a) mixture `0.3·p + 0.7·uniform`, (b) displaced (q's argmax ≠ p's argmax); 40,000 rounds, pinned seed | `TV(emitted, p) < 0.02` (the Bench-634 bar; a perfect sampler's expected TV here ≈ 0.0025) | the **greedy-verify control** (sampled drafts judged by argmax equality — the PrefixMatch analogue) must read `TV > 0.30` (fail-required: it is a point mass at the argmax, TV = 1 − p_max ≈ 0.48) |
| G2 | **induced-law exactness** (deterministic, no sampling): `induced = min(p, q) + (1 − Σ min(p, q))·resid` equals `p` — on 200 seeded random (p,q) pairs and on every (p,q) pair the E2E rollouts capture | `max_x |induced(x) − p(x)| ≤ bar(V)` with `bar = max(1e-12, 4·V·ε)`, ε = the f64 machine epsilon (V = 11 ⇒ 1e-12; the pin's V = 131072 ⇒ 1.2e-10 — the sequential-fold noise scale, pinned by the row length, not by a reading) | none needed (an algebra identity); a miss is an implementation defect, never noise |
| G3 | **E2E identity through the real forward** (synth 6-block, k=2, V=11): first-token draws, ref = direct sampler on the k_full row, spec = the sampler at γ ∈ {1, 4}; N = 200,000 per arm; `p_true` = the f64 softmax of the k_full row (known exactly); T pinned by the ladder rule below | `TV(spec_hist, p_true) < 0.02` AND `TV(ref_hist, p_true) < 0.02` (the floor; uniform-worst-case expectation at this N, V is ≈ 0.006) | the greedy-verify control `TV > 0.10` (fail-required; by the ladder rule it is ≥ 0.25) |
| G4 | **determinism + structure**: two passes at one seed are bit-identical (streams + counters); different seeds differ; `drafted == rounds·γ`; `accepted + rejected + bonus == emitted_uncapped`; `verify_steps == accepted + rounds`; `draft_steps == rounds·γ + bonus`; both caches exit at `prompt + n` | exact | — |
| G5 | **posture refusals**: LastLoop verify, partial-depth verify, non-k=1 draft refuse loud before any decode | exact | — |

G3 runs as an `#[ignore]`d `--release` cell: N is sized for the bar (200,000), not for a debug run, and at a reduced N the uniform-worst-case expectation sits too near 0.02 to be a bar — so there is NO reduced-N always-on twin of G3. The always-on battery is G1 + G2 + G4 + G5; G3's skip line names the release command.

**Two law corrections folded before the first run (2026-11-23, same session — recorded, not silently edited):**

- **The bonus token is sampled from the last target row DIRECTLY** (Leviathan Alg. 1's n = γ+1 branch). The residual-bonus form first drafted here (`normalize(max(0, p_γ − q_γ))`) is also exact, but it costs a draft row nobody needs; the direct form makes the chain's last feed deferrable and changes the G4 counter law to `draft_steps == rounds·γ + bonus` (an all-accepted round pays its deferred tail feed).
- **The target must carry entropy for the control to have teeth.** The greedy-verify control is a point mass at the target's argmax, so its TV is exactly `1 − p_max` — on a near-deterministic target it is small (Bench 634's own E2E control read 0.014). G1's toy is therefore peaked at p_max ≈ 0.52, not 0.98 (control TV ≈ 0.48; a perfect sampler's expected TV ≈ 0.0025 — the control bar moves from `> 0.10` to `> 0.30` there, 6× margin). G3's temperature is pinned BY RULE, not by search: the smallest T in {1, 2, 4, 8, 16} at which the tested first-position target row has `p_max ≤ 0.75` (the identity is temperature-invariant — T scales both arms; the control's TV is then ≥ 0.25). No T in the ladder qualifying ⇒ G3 reports UNDECIDABLE on the synth and the identity rests on G1 + G2 + the real-pin receipt — never a silent pass.
- G1's mixture proposal is `0.3·p + 0.7·uniform` (acceptance ≈ 0.65 — the residual path carries ≈ 35% of rounds); the displaced proposal peaks on token 7 (acceptance ≈ 0.50).

## The real-pin cell (report-only + receipts; structural — no timing rows)

`examples/i1221_t4_sampling_cell.rs`, resumable JSONL per plan 636 (header lineage = pin blake3 + STAMP + prompt law + seed; mismatch refuses). On the Q5_K_M pin at k_full=4, T=1:

1. **Induced-law receipt** — along one sampled rollout per (prompt, γ), every captured (p_i, q_i) pair passes G2's bar (1.2e-10 at the pin's V). This is the real pin's identity evidence: exact, load-independent, no sampling noise.
2. **Acceptance census** (the headline open measurement): rounds, accepted/drafted, rejects by position, bonus, tokens/round, the counter law — 2 prompts × γ ∈ {2, 4} × n = 12.
3. **Determinism receipt** — two passes, one seed: stream digests equal.
4. **First-token two-sample table** — N draws per arm on the CAPTURED first-position rows (the forward runs once; the draws are pure sampler steps): reported with its noise expectation, never gated (at any affordable N the real pin's 131k-vocab TV is noise-dominated — disclosed, not hidden).

Window discipline (plan 636): structural, so it may run concurrent-with-light-load — the concurrency is DISCLOSED in the header, and no wall number is quoted. The TIMING rows of Bench 1238 stay parked for a preflight-PASS window; this plan does not un-park them.

## Fixture adjudication (2026-11-23, BEFORE the gated run — the first battery run's findings, recorded not fitted)

The first run of the battery returned three reds, all fixture properties (G5 and G1 were green):

- **The stock synth's target is a POINT MASS at every ladder temperature.** Its first-position logits are `[388181 … 5795502]` (the fixture bakes `output.weight` as `(i+1)·32` coordinates; the arch's logit formula multiplies by `output_scale/σ₀ ≈ 113`). p_max = 1.0000 at T ∈ {1, 2, 4, 8, 16} ⇒ the ladder rule reports **G3 UNDECIDABLE on the stock fixture**. The ladder was NOT widened to make it pass (T = ∞ would trivially qualify — exactly the fit the rule exists to forbid).
- **G4's reject path is unreachable on the stock fixture** (p ≡ q, the same point mass: every draft is accepted).
- **G2's pair floor was an arithmetic slip** (the gate caught its own author): judged pairs are `rounds·γ` per run, not `n·γ`; 150 was the correct count.

**The entropy-bearing variant** (test-local, post-load): the stock fixture with `lm_head × 2^-16` — the smallest power of two bringing max|logit| under 100 (5.9…88.4). Measured: p_max = 0.9999 / 0.990 / 0.900 / **0.680** / 0.429 at T = 1 / 2 / 4 / 8 / 16 ⇒ the unchanged ladder rule picks **T = 8**; the control's TV = 1 − 0.680 = 0.32. **KL(p‖q) = 7.8e-4 nats at T = 8** — the loop stays a near-no-op (the landed greedy census's finding, confirmed on a non-saturating head). A divergent attempt (`residual_alpha := 1` on every block) made the two depths numerically IDENTICAL (KL −0.000000 at every T): on these baked weights the recurrence reaches its fixed point within one pass, so no post-load rescale can make the synth depths diverge. Recorded; not used. **Divergent (p, q) through the forward is the real pin's and only the real pin's.**

Consequences (all pinned before the gated run):

- G3 and G4 run on the entropy-bearing variant. G3's statistic (first-token law through the full path, N = 200,000) stands unchanged; its control reads TV ≈ 0.32.
- **The E2E reject/repair path is exercised by a LONG STREAM** — n = 20,000, γ = 3, T = 8: at the measured ≈ 8e-4 rejects per judged token that is ≈ 16 expected rejects (P(0) ≈ 1e-7), so the assert `rejected ≥ 1` is robust rather than seed-lucky. The long stream carries a **cache-state receipt**: the verify-cache row decoded AFTER the stream is bit-identical to a fresh k_full reference decode of the emitted stream (the repair left exactly the reference's state).
- **The lazy-verify invariant is a `debug_assert` at every round end** (verify-cache length == the emitted prefix; draft-cache length == the same after its rewind) — live in the debug-profile always-on run, compiled out in release.
- The divergent-(p, q) E2E identity and the first real reject-path exercise are the real-pin cell's receipts (induced law at bar 1.2e-10; the acceptance census).

## Tasks

- [x] **T1 — the instrument**: `src/dory/self_spec.rs` gained the sampling arm (`self_spec_sample`, `ar_sample`, the pure step + induced law + the f64 softmax/categorical/residual helpers, `SpecRng`, `SampleCapture`, `SpecSampleError`, the fail-required `self_spec_greedy_verify_control` on the same round core) behind `dory`; greedy arm untouched (its posture asserts moved into the shared `assert_spec_posture`); lazy verify + deferred tail feed + target-direct bonus; the lazy-verify invariant is a `debug_assert` at every round end. LANDED `3d040f02`.
- [x] **T2 — the gate battery** `tests/i1221_t4_sampling_gates.rs` + the P43 row (same commit). **Measured 2026-11-23**: G1 spec TV 0.00936 (mixture) / 0.00737 (displaced) vs the 0.02 bar, accept rates 0.6645 / 0.4996 against the law's Σmin(p,q) 0.6603 / 0.4991, direct floor 0.00704, control TV **0.4835** (fail-required ✓); G2 induced law exact to 1.4e-16 (random + edge pairs), 0.0 (stock rows) / 2.2e-16 (entropy rows); G3 (release) spec TV **0.00032** (γ=1) / **0.00087** (γ=4) at the reference floor 0.00086, control **0.30787 = 1 − p_max**, 3,738 / 11,834 E2E rejects across the spec arms; G4 determinism + counter identities + cache alignment + cap overshoot + 3 sweep rejects at the pinned seeds; G4b (release) the 20,000-token stream: **204 rejects** (1.33e-2 per judged token, split [68, 68, 68] by position), the cache-state receipt bit-identical to a fresh k_full prefill of the stream; G5 five loud refusals. Always-on 5/5 in 0.16 s (debug); 7/7 in 36.7 s (release, both heavy cells included).
- [x] **T3 — the real-pin cell**: example LANDED `330b3293` (`examples/i1221_t4_sampling_cell.rs` + its P43 row); run 1 executed by the queued job (amo `01a20729-6915`, exit 0, 2026-11-23T07:02Z) but its rows + the schema-2 cell WIP sat stashed by the 11-24 sync sweep (57b16426) when the holder retired — recovered + repaired + re-run under the re-claim (zed-fba187f9@amo, 2026-11-25). **LANDED as Bench 1260** (`.benchmarks/1260_i1221_t4_sampling_cell.md`, run 2 schema 2 the record of record, run 1 preserved + disclosed): induced-law receipt 1.25e-16/6.0e-17 ≤ bar 1.2e-10 on every judged pair, acceptance census 75–100% with the predicted-acceptance cross-check consistent (Σmin 0.99–1.00, observed rejects on the expectation), determinism digests equal both prompts, counter laws hold, first-token table sane and report-only. **The prompt-collision finding: run 1 (and Bench 1238's "2 prompts") sampled ONE prompt twice — 1238's per-stream identity gates stay valid; only its prompt-count prose narrows.** Rows + run-1 rows + job log committed.
- [x] **T4 — clippy + fmt + push**: clippy `-D warnings` clean at default (lib), no-default (lib), `dory` (lib + the T3/T4 test targets + both T4 examples); rustfmt on every touched file; default lib 296 passed, `dory` lib 314 passed, landed greedy T4 gates 4/4 + spectra 12/12 unchanged; commits pushed through the landing. **Closes with T3's landing — CLOSED 2026-11-25.** ⚠ Sibling-owned red, not this plan's: `examples/i701_precomp_fit.rs` (wip `b763476a`, another session) has no `[[example]]` row, so `--all-targets` is red at every posture — mailed the owning inbox (`01a20724`).

## What a miss means

- G1/G3 control does NOT fail (TV ≤ 0.10): the control is broken, the battery has no teeth — fix the control before reading any arm.
- G1 or G3 spec arm fails: an implementation defect in the step (slot alignment, the bonus row's q, a stale residual) — bisect with G2 (G2 green + G1 red ⇒ the sampling harness, not the law).
- Real-pin G2 receipt fails: f32 logits → f64 conversion or a non-finite row — refuse the cell, never quote the census.
