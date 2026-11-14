# Plan 634 — Issue 1216 T0: the closed-form headroom gate on the typed scorer

**Status:** IN FLIGHT — zed-a0dc2e37@amo, 2026-11-14 (claim lease on the issue is the arbiter)
**Issue:** riir-infer `.issues/1216` (T0 only — T1+ stay open regardless of this gate's verdict)

## Pre-registration (fixed BEFORE any cell runs)

**Question.** Does a closed-form refit of the typed head's scorer beat the shipped scorer on
held-out PRACTICE rows, paired LB95 > 0? (T0 gate; fail = the lane dies before solve spend.)

**Data (contamination law).** PRACTICE = the typed_decisions **train** split only
(`.raw/datasets/typed_decisions/train-*.json`, 1200 rows, 6000 gold-resolvable questions;
builder parity with reflex `build_typed_decisions` filters — non-string instructions, missing
gold, unresolvable gold label skipped identically). The harness **test** split is never read by
any T0 artifact. Split by CASE: fit = first 840 rows (70%), gate = last 360 rows (30%), in file
order (`train-000` → `train-011`); a row's questions never straddle the boundary.

**Surfaces.**
- Extraction: `dump_encoder_states` verbatim (LENC v1) — marker rows `r_i` [k,d] (the scorer's
  LN input, post type-emb + 2 layers, FROZEN) + the forward path's raw logits per question.
  Posture pin: `WeightPosture::F16` widened f32 (the shipped load), `LAYA_DEVICE=metal`
  (disclosed box state; the fidelity gate arbitrates, a red ⇒ re-run quiet). d = 1024 (pinned
  encoder config).
- Fidelity gate (transcription): host replay of the scorer over the SAME frozen tensors must
  reproduce the cached raw logits — CPU 100-row slice: mean|Δ| ≤ 1e-5 (same-math gate,
  425's 5.28e-7 class); the metal cache's replay Δ is REPORTED beside it (device noise is
  disclosed, never folded into the CPU gate).

**Probes (all closed-form, ridge, per-option pooled regression).**
Features per option i of question q: `x_i = LN(r_i; s0w, s0b, eps)` (LN frozen — the 425
posture; torch LayerNorm semantics, biased variance, first-index ties).
- **P1 (primary, pre-registered gate arm):** last-layer ridge — features
  `h_i = gelu_erf(x_i·W1ᵀ + b1)` (W1, b1 frozen), targets `y_i ∈ {0,1}` (gold = 1), solve
  `[w3; b3]` by ridge λ ∈ {0.1, 1, 10} picked on an INNER split of the fit rows (last 20% of
  fit, by the same case order); report all three, the gate reads ONLY the inner-picked λ.
- **P2 (secondary, reported):** linear probe over `x_i` (bypasses the frozen W1 entirely) —
  same targets, same λ grid, same inner pick. Reported for the obstacle read; NEVER the gate.
- Gate metric: per-question pick accuracy on the gate slice — shipped pick = first-index argmax
  over the cached raw logits; probe pick = first-index argmax over the probe's z (temperature
  is monotone-positive ⇒ pick-equivalent). Paired per-question delta d_q ∈ {−1,0,+1};
  accuracy delta = mean(d_q); **GATE: LB95(delta) > 0** via the paired Wilson-style bound
  n_pos, n_neg, n_zero → p̂ over decisive questions with the reflex paired floor δ = 1.0 pt:
  the gate passes iff LB95 > 0 AND decisive ≥ 30 (else UNDECIDED — never folded into PASS).

**Expected sign / obstacle hypothesis.** The scorer's per-option threshold/margin
miscalibration (Research-562 calibration-gap line) is the named repairable obstacle; a FAIL
with a faithful fidelity gate = ability-absent on this cell (the 425 encoder-ceiling kill
extends to typed) — the negative joins the Law-4 ledger, T1+ stay closed.

## Tasks

- [x] Pre-registration written + committed before any cell runs
- [ ] T0.a — practice cases.jsonl (train-split builder parity, gold riding the row id) + the
      100-row CPU fidelity slice
- [ ] T0.b — metal extraction over the practice slice (dump_encoder_states, typed ckpt)
- [ ] T0.c — `examples/i1216_t0_headroom.rs`: cache reader + host scorer replay + P1/P2 ridge
      + paired verdict JSON
- [ ] T0.d — run; record `.benchmarks/1217_i1216_t0_headroom_gate.md` + verdict either way;
      update the issue (T0 line) + Research-671 addendum pointer on close

## Notes

- reflex's OWN cal/posture law (fused gates, threshold refit at cal ρ=30) is a T2 arm (A′/C),
  not a T0 surface — T0 reads raw logits only, one seat, no drafter.
- The cache ids carry `{casepos}:{qid}:{gold_idx}` — gold rides the row id through the
  existing dump bin, zero new extraction code.
- Box state at run time: recorded in the bench record (amo, AC, load, concurrent jobs).
