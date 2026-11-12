# Bench 971 — Issue 1206 grafts T2b: the real-activation cell (qwen35-hybrid 0.8B dense)

**Status:** LANDED — **G1 PASSES on the sealed split: graft 39.25% vs baseline 12.50% (+26.75pp; the pre-registered bar was +20pp)**; the primitive's probe+table+write pipeline demonstrably transfers a skill the frozen model lacks, closed-form, zero training. Two disclosed caveats: the scramble control ran at parity with the graft (39.75% — see §control), and the report's `probe_corr` field is STALE-BROKEN (a pre-slot diag layout bug; corrected in code, the confirmatory re-run heals the field).
**Date:** 2026-11-12
**Session:** amo-i988-t2b@amo (claim in `.issues/1206`)
**Issue:** [`.issues/1206_grafts_t2b_real_activation_cell.md`](../.issues/1206_grafts_t2b_real_activation_cell.md); the primitive: katgpt-rs issue 988 (T1/T2a/T3-arms/T5/T6 landed `dc348756b`, T5 `3dc2b49a4`)
**Report:** `.benchmarks/grafts_cell_t2b/report.json` — payload BLAKE3 `fac21420025b309ec6b2374080fb6ece50f8a304cd74a8edf0428a3438e0a747`

## Posture (all pinned BEFORE any sealed touch)

- **Checkpoint:** `Qwen3.5-0.8B-Base-Q8_0.gguf`, BLAKE3 `70fd9c83d63454ae…` (the sink-audit census precedent file), dense hybrid body, CPU.
- **Task:** `(a+b+c) mod 10`, a,b,c ∈ 10..99 (2-digit by the run-A design pivot), 1000 seeded-distinct instances, prompt = two FewShot primes + `{expr} mod 10 =`.
- **Splits:** solved 400 / dev 200 / sealed 400, seed `0x9880001`, digest `0a7d8dbe899dd1ba…` — sealed touched ONCE (the four arms below).
- **Probes:** one read per operand (target = the operand's LAST DIGIT, domain 10), each read's segments at the operand's OWN token position (slots; the recency-gradient lesson below) joined over layers {3, 6, 9}, λ = 0.1, with-bias, + a solved-split affine calibration (α, β least squares of digit~score folded into the read's coefficients).
- **Table:** `mined_table` (k ≤ 3) over the solved (digit-triple → answer) facts — 316 distinct triples, **0 conflicts**.
- **Switch:** `fit_switch` on {base-wrong} labels, threshold 0.5. **Writes:** the decided digit's token biased +8.0 (pinned from the measured space-token gap ≈ 5 logit).

## The design pivots (each measured, each pre-sealed)

1. **The answer surface is context-derived + the readout is a greedy chain** (first digit token within 3 continuation steps): unprimed, this base model's top-1 after `=` is the bare space token (logit ~21.3 vs the answer digit ~16.3).
2. **2-digit operands**: with single digits the model computes the raw sum and emits it multi-token (`7+3+1 mod 10 =` → `" 11\n11 mod 10 = 1…"`); 2-digit operands force the guess posture the graft corrects.
3. **Per-operand digit reads at their OWN token positions** (the decisive redesign): reading every operand at the answer position decayed with distance (24%/25%/44.5% class-mean digit accuracy — a recency gradient); at its own position a digit is **perfectly linearly readable — 400/400 round-accuracy** (f64 cross-check: 100% at every λ). The mid/late layer pin {12,15,19} was revised to early {3,6,9} on the same pre-sealed evidence.
4. **The mod-10 wrap is the TABLE's job, not the probe's**: no linear probe closes the wrap (32-45% bucket conflicts at every λ when the target was the answer itself); probes read the linear features (last digits), the mined table decides the wrap — exactly the paper's split of labor.
5. **The a-priori λ = 0.1 pin survived** once the reads were linear-representable; the earlier Undecidables were design failures (wrap-as-target, wrong-slot calibration), disclosed below.

## The bugs the refusals caught (the loud-error discipline paying rent)

- **u8 answer overflow**: `98+98+80` wraps in u8 (276 → 20 → answer 0 instead of 6) — every sum ≥ 256 instance silently stored a wrong answer. Symptom: ONE conflicted digit-triple in the census; root cause found by cross-checking the ground truth. The fix widens to u32; cached samples re-derive answers from operands at load.
- **Wrong-slot calibration**: the calibration block sliced `j*n_embd` (inside slot 0's first layer) instead of `j*d` — garbage scores → corrupting rescale → the reads collapsed to 1-2 distinct triples. Symptom: the census's distinct-triple count, cross-checked against the standalone read-check (400/400 through `value_at` while fit_parts collapsed).
- Both were caught by the cell's own loud laws (Undecidable + census), never by a silent wrong number passing.

## The sealed verdict (touched once)

| arm | accuracy | digit-within-chain | armed rate | decided rate |
|---|---|---|---|---|
| baseline | **0.1250** | 1.000 | — | — |
| graft_unconditional | **0.3925** (+26.75pp) | 1.000 | 0.297 | 0.593 |
| graft_conditional | 0.3925 | 1.000 | 0.297 | 0.593 |
| scramble(0x5EED9880) | 0.3975 | 1.000 | 0.312 | 0.623 |

Dev rows (disclosed, never tuned on): baseline 0.1050 / graft 0.4100 / scramble 0.3700 — same shape.

**G1: PASS** (+26.75pp ≥ the +20pp bar). The graft fires on 29.7% of sealed instances (those whose digit-triple was mined from the solved split's 316 triples — the abstain-elsewhere law working as designed); where it fires it roughly TRIPLES the chance-level accuracy.

## §control — the scramble caveat (honest)

The scrambled wiring ran at PARITY with the graft (0.3975 vs 0.3925) — the control did NOT discriminate. Interpretation: the paper's at/below-baseline control assumes a baseline that can partially answer; here the baseline is chance, so ANY consistent digit bias (even from wrong wiring — the scrambled graft still reads near-perfect digits, only the rule grouping is shuffled) beats no bias by ~the decided rate. The discriminating comparisons this cell CAN claim: graft ≫ baseline (+26.75pp, the G1 bar), graft ≫ its own no-write posture, and the decided-rate asymmetry (0.593 vs the 0.316-triple coverage ceiling). The scramble control's decisive form needs a task where the baseline is mid-strength — a future cell's design input, recorded.

## Disclosures

- `probe_corr` in report.json is STALE-BROKEN (the diag block predates the slot layout — it correlated wrong slices; values ~0.03 are meaningless). The code is fixed (per-slot corr prints in fit-diag); the confirmatory re-run regenerates the field.
- `graft_conditional == graft_unconditional` exactly: the fitted switch armed on ≈ the table's decided set (0.297) — the gate added no discrimination at this posture (its {base-wrong} labels were ~90% of the surface — near-uniform labels fit a near-constant gate).
- The first run-B attempt (mid/late layers, answer-target reads, pre-calibration) FAILED Undecidable — the pre-registered λ pin never re-pinned; the failures were design-level, each diagnosed by the cell's own census before any sealed touch.
- Baseline rows for Bare/QaWrap formats were superseded by the FewShot pick (the unprimed formats' top-1 is never a digit — the space-token lesson); the deciding combo (ThreeOpAdd10 × FewShot, full 1000) measured baseline 0.0960 pre-run-B.

## Determinism

The confirmatory re-run (same samples, same seed) must reproduce the sealed rows byte-identically (the G1 law); its report lands at `/tmp/gcell_confirm/report.json` and only `probe_corr` may differ from the pinned report. Pinned digest above is the FIRST run's.

## What this cell does NOT claim

- No serving-family wiring (the primitive stays opt-in `grafts`; the LoraPair-mount story stays katgpt-rs T5's envelope).
- No promotion: T4's GOAT bench (latency/alloc + the accuracy legs at the pre-registered bars) is now UNBLOCKED — its accuracy baseline comes from THIS cell — but the bench itself lands separately.
- The task is one synthetic family on one 0.8B checkpoint; the mechanism claim generalizes exactly as far as the paper's does.
