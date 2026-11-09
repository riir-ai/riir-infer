# Bench 961 — Issue 1180 T2b: the SinkHeavy protection half — the pin is NEUTRAL under score-based maxrecent at both contexts (pre-registered bar NOT met), but load-bearing under score-blind random eviction

**Status:** RECORD (T2b adjudicated; task ticked in the issue).
**Date:** 2026-11-09 (cells ran 2026-11-07 on amo)
**Session:** zed-15be441a@amo (adjudication); cells executed by rig job `01a1b10a-d2e7-7095-8aa8-7401dc3d6de5` (key `i1180-t2b`, Shared), script `~/i1180_data/run_t2b.sh`, `T2B_DONE`
**Issue:** [`.issues/1180_sink_audit_census_followups.md`](../.issues/1180_sink_audit_census_followups.md) (T2b)
**Box:** amo (M5 Ultra 96 GB, AC), worktree `~/git/riir-infer-i1180b` @ `18ea8a15`, `CARGO_TARGET_DIR=/tmp/i1180b_target`, release `needle_eviction_gate` with `sink_audit` (gemma2 backend: SentencePiece + BOS at pos 0, token-by-token prefill).
**Model:** `gemma-2-2b-it` f16 (5,235,213,856 B; sha256 `3c5fb672…` matches the m3 source; census 0.4071 → **SinkHeavy**, Bench 953), corpus `~/i1180_data/chat_probe`, 4 needles `--plant-fix 1` (all planted — depths logged), `--nopin-arms 1`, `--budget-frac 0.02,0.25`, arms `maxrecent,random@2pct`, chunk 512. Contexts: **C1 = 3800** (in-distribution; every position < 4096) and **C2 = 8000** (local layers attend past their trained 4096 sliding window — disclosed OOD, shared by the full reference). Walls: C1 8,732 s-class, C2 ~11,518 s-class (job log).

## The pre-registered bar (written 2026-11-06 in the issue, before any T2b cell ran)

> **LOAD-BEARING (bar MET)** iff at 2% maxrecent in C1: Δgen ≥ +0.10 nats/tok OR Δneedle ≥ +0.25 nats OR retrieval(pin) > retrieval(nopin). **NEUTRAL** iff |Δgen| < 0.02 AND |Δneedle| < 0.10 AND equal retrieval. Anything else = WEAK/MIXED, disclosed; nopin better = a NEGATIVE recorded as such. C2 and 25% are reported against the same rule, never folded into C1.

## C1 (3800, in-distribution)

| arm | meanΔ nats | genNLL | ret | sink_surv |
|---|---:|---:|---:|---:|
| full (reference) | 0.0000 | 3.4023 | 1.000 | 1.00 |
| maxrecent@2pct (pin 4) | 2.1799 | 3.3202 | 0.000 | 1.00 |
| maxrecent@2pct_nopin | 2.2249 | 3.3085 | 0.000 | **0.240** |
| random@2pct (pin 4) | 2.4645 | 3.0550 | 0.000 | 1.00 |
| random@2pct_nopin | 2.7259 | 4.0879 | 0.000 | **0.000** |
| maxrecent@25pct (pin 4) | 2.1375 | 3.2832 | 0.091 | 1.00 |
| maxrecent@25pct_nopin | 2.1368 | 3.2838 | 0.091 | **0.452** |

**C1 @ 2% adjudication:** Δgen = 3.3085 − 3.3202 = **−0.0117** (nopin *better*, magnitude inside the 0.02 band); Δneedle = 2.2249 − 2.1799 = **+0.0450** (< 0.25); retrieval equal (0.000 = 0.000). → **None of the three LOAD-BEARING conditions met; all three NEUTRAL conditions met → NEUTRAL.** 25%: meanΔ delta +0.0007, genNLL delta −0.0006, retrieval equal → NEUTRAL.

## C2 (8000, disclosed OOD — reported against the same rule, never folded)

| arm | meanΔ nats | genNLL | ret | sink_surv |
|---|---:|---:|---:|---:|
| full (reference) | 0.0000 | 6.8139 | 1.000 | 1.00 |
| maxrecent@2pct (pin 4) | 0.6535 | 6.5989 | 0.014 | 1.00 |
| maxrecent@2pct_nopin | 0.7079 | 6.6035 | 0.014 | **0.106** |
| random@2pct (pin 4) | −0.0409 | 6.3316 | 0.036 | 1.00 |
| random@2pct_nopin | 0.4444 | 6.7437 | 0.036 | **0.000** |
| maxrecent@25pct (pin 4) | 0.3333 | 6.6908 | 0.444 | 1.00 |
| maxrecent@25pct_nopin | 0.3385 | 6.6930 | 0.444 | **0.779** |

**C2 @ 2%:** Δgen +0.0046, Δneedle +0.0544, retrieval equal → NEUTRAL. 25%: +0.0052 / +0.0022 / equal → NEUTRAL. The full-reference genNLL 6.81 (vs C1's 3.40) is the disclosed OOD cost shared by every arm and the reference.

## Mechanism readout (the pre-stated hypothesis, adjudicated)

Hypothesis: "a score-based evictor can keep the BOS sink on score alone on a SinkHeavy model, in which case the pin is REDUNDANT under maxrecent."

- **Score alone does NOT reliably keep the sinks**: nopin `sink_row_survival` reads 0.240 (C1) / 0.106 (C2) at 2% — the BOS sink and its neighbours are evicted ~76–89% of the time without the pin. The *mechanism half* of the hypothesis (score keeps sinks) is REFUTED at this scale.
- **Yet the pin changes nothing measurable under maxrecent**: every maxrecent comparison lands inside the NEUTRAL band. At 2% of a 3800-row window (76 kept rows), the needle damage dominates (meanΔ ≈ 2.2 with or without sinks) and the sink contribution sits below the band. **The pin is redundant-under-maxrecent *in effect*, not through the hypothesised mechanism.** This is "a negative for 'load-bearing under maxrecent'" — exactly the class the pre-registered rule names — and **NOT a refutation of the SinkHeavy census** (0.4071, Bench 953) or of sink pins as a class.
- **Score-blind control (the pair the bar named): the pin IS load-bearing there.** random@2pct vs random_nopin: C1 Δneedle = 2.7259 − 2.4645 = **+0.2614** and C2 Δneedle = 0.4444 − (−0.0409) = **+0.4853**, with sink_surv 0.0 → 1.0 flip. When eviction cannot score rows, the sink pin protects (both contexts, large effect). The pin's value is real; score-based eviction simply already avoids most of the damage the pin guards against, at these budgets and scales.

## Instrument notes

- The c8000 run's report merge hit the same NaN→null defect as Bench 960's ("rewriting WITHOUT the old rows") — final reports complete post-rewrite, per-run logs carry full tables; follow-up filed in the issue.
- `--plant-fix 1` planted all 4 needles in both contexts (the legacy layout's needle-0 collision is bypassed); the 3800/8000 scales are reduced per the issue's cap rationale (gemma-2's 8K max position), never comparable to T2a's 64K absolute numbers.

## Verdict

**T2b: NEUTRAL under score-based maxrecent (pre-registered bar NOT met — disclosed as such per the rule), with the pin CONFIRMED load-bearing under score-blind random eviction (+0.26/+0.49 Δneedle).** The katgpt-rs-981 gate's protection half does not transfer to score-based eviction at reduced scale on this SinkHeavy checkpoint; the census verdict (SinkHeavy) and the audit vocabulary are unaffected. No promotion, no demotion — the audited-pin runtime gate's value rests on the T2a neutral half (Bench 960); a full-scale protection re-cell would need a long-context SinkHeavy host, which this model's 8K position cap cannot provide.
