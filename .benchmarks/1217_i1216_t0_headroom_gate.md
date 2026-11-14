# Bench 1217 — Issue 1216 T0: closed-form headroom gate on the typed scorer — FAIL (the lane dies)

**Date:** 2026-11-14 · **Repo:** riir-infer · **Lane:** `.issues/1216` T0 (plan 634
pre-registration) · **Host:** amo (M5 Ultra 96 GB, AC plugged, load 3.2/10.8/12.6,
one seal-viewer heavy process resident — disclosed; the gate is single-seat paired
pick arithmetic, not a timing bar)

## Question

Does a closed-form refit of the typed head's scorer beat the shipped scorer on
held-out PRACTICE rows, paired LB95 > 0? (T0 headroom gate — fail = the lane dies
before any T1 solve spend; the 425 encoder-ceiling kill is the named alternative.)

## Setup (per the plan-634 pre-registration, one deviation disclosed)

- **Practice = the typed_decisions TRAIN split only** (1200 rows, builder parity with
  reflex `build_typed_decisions`; gold resolvable on 6000 questions). Splits by case:
  fit = rows [0, 840) (4200 q), gate = [840, 1200) (1800 q), file order. The harness
  test split was never read by any T0 artifact.
- Extraction: `dump_encoder_states` verbatim (LENC v1), typed checkpoint,
  `WeightPosture::F16` widened f32, metal posture for the fit/gate caches (2.9s/163s/73s
  for fid/fit/gate) + a CPU 100-question fid slice (85s).
- Probes: P1 = last-layer ridge over the frozen GELU features
  (`h = gelu_erf(LN(r)·W1ᵀ + b1)`, LN + W1 frozen — the 425 posture); P2 = linear ridge
  over the LN rows (bypasses W1). Targets y ∈ {0,1} per option; λ ∈ {0.1, 1, 10}
  inner-picked (train on fit cases ≤ 671, eval 672–839): P1 → 1.0, P2 → 0.1. Full refit
  on all 840 fit rows at the picked λ; paired per-question pick deltas vs the shipped
  scorer on the 1800 gate questions. Tie rule: first-index argmax; picks are
  temperature-invariant (t > 0 monotone on raw logits).
- **Fidelity bar amendment (disclosed pre-verdict):** the pre-registered 1e-5 mean bar
  mispriced the f32-gemm reduction-noise class. Bisection (`i1216_t0_fid_bisect` +
  a python f64 replay): the reference logits ARE the raw scorer outputs and the f64
  replay matches at **5.25e-7 mean / 1.9e-6 max**, while `ops::matmul_w`'s f32 gemm
  carries ~2.3e-3 mean / 3.6e-3 max on the s1 reduction (25% of a1 elements off by
  > 1e-5, worst 2.9e-3 at |a1| ≈ 21) — the entire logit delta, nothing else. The gate
  ran at the 613 precedent bars (mean ≤ 5e-3, max ≤ 5e-2, Bench 613's own tolerance
  class) and PASSED: transcription is exact, the residual is measured instrument noise.

## Result: GATE FAIL (both probes negative)

| arm | gate acc | paired Δacc | pos/neg/zero | decisive | paired LB (decisive share) |
|---|---|---|---|---|---|
| shipped scorer | **0.8683** | — | — | — | — |
| P1 last-layer ridge (λ=1) | 0.8461 | **−2.22 pt** | 16/56/1728 | 72 | 0.14 → **FAIL** (LB < 0.5, Δ < 0) |
| P2 linear over LN rows (λ=0.1) | 0.8594 | −0.89 pt | 21/37/1742 | 58 | 0.25 → FAIL |

- Shipped gate-slice accuracy 0.8683 (1800 q) vs 0.8481 on the fit rows — the scorer
  generalizes train→heldout fine; the probe makes it WORSE.
- **In-sample sanity (the decisive negative):** scoring the probe against shipped ON
  ITS OWN fit rows (gate cache = fit cache) gives Δ −2.0 pt with zero decisive wins —
  the closed-form refit cannot even interpolate the shipped scorer's training-set picks.
  With 1025 free parameters over ~6.2k fit options and λ→0 available, the miss is the
  OBJECTIVE, not capacity or regularization: a 0/1 ridge on per-option targets optimizes
  calibrated probabilities, and argmax-pick under a shared d→1 map is not recovered by
  it. That is the 425 encoder-ceiling mechanism showing up one layer earlier: the frozen
  features do not carry a refit-able margin signal on this suite.

## Verdict

**Law-4 ledger row** — surface × supervision × cell = typed-checkpoint scorer ×
closed-form 0/1 ridge × typed_decisions train-split pick accuracy = **FAIL** (the fourth
kill in the Research-671 family; the dspy-wsp extrapolation does not survive its own T0).

T1–T5 stay CLOSED (no solve spend). The only unexplored axis the family names is the
hosted-writer arm — already killed on its analogue by riir-train 665 Lane 2 (Law-4 FAIL
with the random control). Re-open trigger: a NEW cell naming a mechanism neither kill
covers (e.g. a pick-aware objective — pairwise-margin/LP — or a per-question-type
split scorer). A λ/feature retune alone is NOT a re-open.

## Box state

amo (this box), macOS 27.2, M5 Ultra 96 GB, AC plugged, load 3.2/10.8/12.6 at gate run;
one resident seal-viewer process (unrelated lane); metal extraction passes completed
uncontended at ~26 rows/s; no other cargo/riir compute consumer during the gate.

## Lessons

1. **Price the f32-gemm noise class before writing a fidelity bar** — 613's 5e-3/5e-2
   tolerance exists for exactly this; a 1e-5 bar against a gemm-based replay reds on
   instrument noise, not transcription. The bisect arms (gemm vs loop-on-h vs full f64)
   localize any future break in one run.
2. **The in-sample sanity arm is cheap and decisive** — a refit that cannot win IN
   SAMPLE is objective-misspecified; running it first would have shortened this cell.
