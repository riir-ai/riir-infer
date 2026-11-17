# Issue 1235: the rec kernel's bit-exact fusion residue — the q-dot off the per-token critical path (the Bench-1233 named "layout/fusion restructure" class)

**Status:** OPEN — filed 2026-11-17 (amo-zed)
**Claimed:** amo-zed, 1794905378, expires 2026-11-19T08:49:38Z
(claim scope: T1 arm + G1 sweep join + T2 price cell at the pre-registered bars; <3% at p=512 closes CONFIRMATORY same-session. Worktree-isolated — the F2 re-application may touch riir-infer-gpu files mid-lane.) out of Issue 1233's CONFIRMATORY close (the chunked class was measured-NOT-BIT-IDENTICAL — Bench 705 — and folded; THIS is the other class Bench 1233 named, the bit-exact one).

## The premise

The serving gdn_rec path is the MR kernel at planes=8 (MR wins over the staged twin when armed — `ternary_deltanet_gpu_forward.rs` ~11753; default `RIIR_GDN_REC_MR` = 8). Its per-token per-row chain:

```
4 FMA (k partials) → plane_sum #1 → δ broadcast → 4 FMA (state update)
→ 4 FMA (q partials) → plane_sum #2 → out write
```

The **q-side (partials + plane_sum #2 + write) is OFF the recurrence dependency chain**: its result feeds only the output buffer, never the state. The next token's k-partial FMAs depend only on the updated state registers — they can issue BEFORE plane_sum #2 drains. An explicit restructure that schedules the q-dot late (accumulate q partials in registers, issue the butterfly after the NEXT token's k-partials, or batched per K-token flush) hands the compiler an order it may not find itself. **Bit-exact by construction**: every fp op's inputs are unchanged (same accumulators, same butterfly tree inside `plane_sum`, same per-token order of state mutations) — only the ISSUE ORDER of independent work moves. The existing G1 sweep (`mr_is_bit_identical_to_multi_token` + the arm loop) proves it unchanged.

Whether the MSL compiler already does this is an empirical question — the probe answers it. If it already interleaves, the arm reads ~0% and the lane folds cheaply; the residue is then priced as compiler-saturated.

## Pre-registered gates

1. **T1 implementation** — an arm latch in `deltanet_recurrence_mr_cubecl.rs` (default OFF; PROMOTE-ONLY env per the env-switch law — an opt-in arm's env only ever arms; the unarmed build compiles the incumbent bytes).
2. **G1 (mandatory, zero tolerance)** — the new arm joins `mr_is_bit_identical_to_multi_token`'s sweep (every ladder arm × the B1 case set, output AND state bits). Any diff = the arm is wrong, revert, no tolerance.
3. **T2 price cell** — `test_rec_mr_timing_probe` gains the fusion arm; scored at **p=512 (the serving chunk) + p=128/2048 (the ladder shapes)** vs the **MR8 serving arm** (the promoted posture — not the plain incumbent), interleaved median-of-9, release posture, exclusive GPU window (bmo), `bench_preflight` PROVENANCE quoted. **Bar: ≥3% kernel-isolated median at p=512** graduates to the promotion lane (the paired e2e A/B, the f16b-composed serving posture, GOAT gates, delegated-owner verdict, demote-env-first flip); under 3% → the lane closes CONFIRMATORY with the residue priced as compiler-saturated (recorded, done).
4. **Engagement proof first** (the Issue-1124 lesson): a launch-counter vacuous guard proves the arm reached the kernel in every measured posture.

## Scope guards

- Never touch the chunked class (`deltanet_delta_rule_chunked.rs` stays OFF — Issue 1233's close is the record).
- Never touch the decode resident lanes (issue 1181/1217 own those seams).
- The MR planes ladder stays as-is (`RIIR_GDN_REC_MR` semantics unchanged; the fusion arm is orthogonal and composes with any planes value).
- The staged twin's dispatch arm stays untouched (MR wins over it by the existing precedence).

## Named prior

- Bench 896 (CUDA): the r-axis arms (multiple rows per LANE) LOST to r4w4 on sm_89 — the OTHER never-built-on-Metal lever; the MR module doc names it "the follow-up arm if G2 shows the wall is per-chain latency". This issue does NOT build the r-axis (one variable at a time; the fusion probe's verdict prices whether a second ILP axis is worth filing).
- The q-dot-off-chain idea was recorded in Issue 1233's close (HISTORY 2026-11-17, commit `679ee528`) as "the bit-exact residue... its own issue if a lane opens it" — this is that lane.
