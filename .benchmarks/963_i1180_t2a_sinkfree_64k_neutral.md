# Bench 963 — Issue 1180 T2a: the SinkFree 64K neutral half — the audited pin (4→1) costs nothing at either budget; the GOAT neutral half PASSES

**Status:** RECORD (T2a adjudicated; task ticked in the issue). **Numbering note (2026-11-09): 960 was double-named this morning — the riir-infer 1183 T5 record reserved `960_i1183_t5_head_only_acceptance_path_b_routes` in committed text (1183 + riir-train 679, 09:07–09:08, file unwritten) two hours before this cell's file landed (11:00, allocated from `.highwater` 959). No strict citation-weight margin (4–4); renumbered to 963 (next free) with every citing site moved in this commit — 960 stays the 1183 T5 record's number per first-committed-naming, and this file's original 960 allocation is void. The 1187-fix note below cites `f3385a8` through the 1180 issue (unchanged).**
**Date:** 2026-11-09 (cells ran 2026-11-07 on amo)
**Session:** zed-15be441a@amo (adjudication); cells executed by rig jobs `01a1b0e0` (incumbent) / `dc3d6de5`-family runner script `~/i1180_data/run_t2a.sh`
**Issue:** riir-infer Issue 1180 (T2a; CLOSED 2026-11-09 — record folded into HISTORY.md, file removed)
**Box:** amo (M5 Ultra 96 GB, AC), Shared rig key `i1180-t2a`, worktree `~/git/riir-infer-i1180` @ `18ea8a15`, `CARGO_TARGET_DIR=/tmp/i1180_target`, release `needle_eviction_gate` with `sink_audit`. Walls: incumbent 14,643.2 s + audited 5,386.6 s.
**Model:** `Qwen3.5-0.8B-Base-Q8_0` (811,843,200 B, the bench-947/953 checkpoint), corpus `~/i1180_data/chat_probe`, ctx **65536**, 4 needles with `--plant-fix 1` (the unplanted-needle-0 rig bug's fix, 1180 finding), budgets `--budget-frac 0.25,0.02`, arms `maxrecent`, nopin twins via `--nopin-arms 1`.

## The audited-pin decision (in-process, quoted from `audited.log`)

```
# sink-audit: n=512 layers=6 heads=8 f_attn_token0=0.076425 m_act=35.961 verdict=SinkFree -> audited pin n_sink=1 (window carried) | 25.7s
```

The bench-947 census value (0.0764 → SinkFree → pin 4→1) **reproduces in-process** on the full 64K cell — the audit rides the same forward and reduces the pin BEFORE the eviction arms run.

## The cells (meanΔ = pooled ΔNLL vs the full reference; ret = needle-row survival; sink = sink-row survival)

| arm | meanΔ nats | genNLL | needle ΔNLL (per needle) | ret | sink_surv |
|---|---:|---:|---|---:|---:|
| full (reference, shared) | 0.0000 | 3.2058 | [0.0757, 0.1230, 0.0253, 0.0013] | 1.000 | 1.00 |
| maxrecent@25pct (pin 4) | −0.0066 | 3.2442 | [0.0772, 0.0944, 0.0249, 0.0025] | 0.839 | 1.00 |
| maxrecent@25pct_nopin | −0.0066 | 3.2442 | identical to pinned | 0.839 | 1.00 |
| maxrecent@2pct (pin 4) | 1.4052 | 3.2049 | [1.629, 1.969, 1.185, 1.064] | 0.185 | 1.00 |
| maxrecent@2pct_nopin | 1.4049 | 3.2166 | [1.628, 1.969, 1.185, 1.063] | 0.185 | 0.50 |
| **audited** maxrecent@25pct (pin 1) | **−0.0066** | 3.2442 | identical to pin-4 25% | 0.839 | 1.00 |
| **audited** maxrecent@2pct (pin 1) | **1.4049** | 3.2166 | identical to nopin 2% | 0.185 | 0.50 |

The audited run reuses run 1's full reference (`--full-from`; `reference_loaded=true`), so every audited row is comparable row-for-row. The 25% rows are **byte-class identical** across pin 4 / pin 1 (same meanΔ to 7 decimals, same per-needle vectors, same evict-event count) — at 25% the budget keeps the sink rows inside the recent window regardless of pinning. At 2% the audited pin (n_sink=1) matches the *nopin* twin (meanΔ 1.4049, genNLL 3.2166, sink_surv 0.50): with one pinned row the score-based evictor's behaviour is statistically the nopin shape.

## Adjudication vs the pre-registered bar

Bar (issue line): "audited ≈ incumbent within the 1.0-nat retention rule at both budgets (the GOAT neutral half); the no-pin control shows the pin is load-bearing at 2%."

1. **GOAT neutral half: PASS.** audited vs incumbent meanΔ deltas: 25% → |−0.0066 − (−0.0066)| = 0.0000; 2% → |1.4049 − 1.4052| = **0.0003 nats** — both orders of magnitude inside the 1.0-nat rule. **The audited pin reduction 4→1 costs nothing measurable on the SinkFree checkpoint at 64K.** This is the katgpt-rs Issue-981 gate's runtime half landing GOAT-neutral on the real harness.
2. **Load-bearing control: NOT demonstrated here, and not expected to be.** nopin vs pin at 2%: Δgen(nopin−pin) = +0.0117 (nopin marginally worse — sub-noise), Δneedle = −0.0003, retrieval equal (0.185), nopin sink_surv 0.50 vs pinned 1.00. On a **SinkFree** model the sink rows are not special; the pin's load-bearing leg belongs to T2b's SinkHeavy model (Bench 961) — recorded here as the expected-absence, never folded into a "pin useless" claim.

## Known instrument defect (pre-existing, disclosed; data intact)

`needle_gate_report.json` merge fails when a field is NaN (`hub_row_survival` with `hubs=0` → serde_json emits `null`, the merge then refuses `null` as f32) and **rewrites WITHOUT the old rows**. Both runs completed with full rewrites (no rows lost — the per-run logs carry the complete tables), but any future incremental multi-process merge would silently drop earlier arms. Follow-up filed in the issue (T-fix line); the fix shape is `Option<f32>` for the can-be-NaN survey fields, or skip-null serialization.

## Verdict

**T2a PASS (GOAT neutral half).** The SinkFree half of the katgpt-rs-981 runtime gate holds on the real harness at full 64K scale: audited pin ≈ incumbent pin at both budgets, and the audit's pin reduction is free. Pair: [Bench 961](961_i1180_t2b_sinkheavy_protection.md) (the SinkHeavy protection half).
