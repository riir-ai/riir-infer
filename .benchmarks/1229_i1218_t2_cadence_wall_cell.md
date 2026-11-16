# Bench 1229 — Issue 1218 T2: the fence-cadence wall cell (K8 re-anchor vs K32 armed) — G1 + stop identity PASS, the cadence bar FAILS on the drift-free read (within-run delta identical: −2.415 vs −2.413 ms/tok); K32 stays opt-in

**Status:** DONE — T2 adjudicated MEASURED-NEGATIVE (no cadence gain; mechanism-explained by Issue 1181's pool-bound-queue finding). The pre-registered ≥0.3 ms/tok adopt bar is NOT met on the honest read: the nominal cross-run resident delta (0.56 ms/tok, K8 15.35 → K32 14.79 ms/tok pair-mean) is REFUSED as a bar-passer by the run-level drift control — the within-run resident-vs-incumbent delta is IDENTICAL at both cadences (−2.415 K8 vs −2.413 K32 ms/tok, printed by the cell itself), while the incumbent arm shifted +0.5 ms/tok between runs (16.94/17.58/17.65 → 18.31/16.82/17.18) with no arm present to cause it. A lever that moved the resident arm would move the within-run delta; it read 0.002 ms/tok. T3 (GOAT cell) does not gate open.
**Date:** 2026-11-16/17 (cell ran 23:28–23:38Z 11-16)
**Session:** amo-i1218t2 (claim `58cbd6b8` line, lease continues from `003f00e4`), adjudicated by zed-023c35c1@amo
**Issue:** [`.issues/1218_resident_fence_flush_decomposition.md`](../.issues/1218_resident_fence_flush_decomposition.md)
**Box:** amo (M5 Ultra, 96 GiB), AC — ⚠ **NOT the quiet-exclusive premise the bars pre-registered against**: the window overlapped sibling lanes' builds (riir-refine i238 at ~400–600% CPU + two katgpt-core cargo tests; load 9.2/30 at 23:38Z, 4 heavy contenders). Disclosed and load-bearing for the verdict: the drift control exists precisely because the window was polluted. A quiet-window re-run is NOT owed — see §Mechanism.
**Rig incident (disclosed):** the job wrapper `b5247eb6` poison-looped (×131 by 23:38Z, no job log ever written) while the cell script `.i1218cell/run_cell.sh` ran to completion beside it — all five cells landed sequentially (log mtimes 06:33→06:38 +07, no interleaving/overwrite). The rig purged the job; the Exclusive slot freed for the sibling's i1226 walk (767fffab) without needing the cancel the sibling's escalation checkpoint was preparing. The wrapper failure mode (rig-side, script-side success) is recorded here so the next `.i1218cell`-style cell runs with an explicit `echo` sentinel at script end.

## The cells (release build, one binary, both postures; `bench_831_decode_long`; ctx=2048 n=512 pairs=3 ABBA; model Ternary-Bonsai-27B)

| cell | arm | result |
|---|---|---|
| k8_incumbent_ab | env unset (incumbent cadence 8) | incumbent 56.89 tok/s · resident 65.95 · ratio 0.863× · within-run delta **−2.415 ms/tok** · FNV `121982e1b7878d05` |
| k8_incumbent_stop | stop-identity at cadence 8 | visible prefix 65 tok — both arms IDENTICAL · overshoot 0 · TTFT 150.9 ms · bursts [8×8] |
| k32_armed_ab | `RIIR_RESIDENT_HARVEST_K=32` | incumbent 58.21 tok/s · resident 67.71 · ratio 0.860× · within-run delta **−2.413 ms/tok** · FNV `121982e1b7878d05` |
| k32_armed_stop | stop-identity at cadence 32 | visible prefix 65 tok — both arms IDENTICAL · overshoot 0 · TTFT 488.5 ms · bursts [32, 32] |
| k32_armed_stop65 | mid-burst `STOP_AT=65` (1 past a K=32 boundary) | visible prefix 30 tok — IDENTICAL · overshoot 3 tok (bounded, never visible) · bursts [32] |

Pair walls (ms/tok): K8 resident 15.08/15.82/15.16 (mean 15.35), incumbent 16.94/17.58/17.65 (mean 17.39); K32 resident 14.91/14.77/14.70 (mean 14.79), incumbent 18.31/16.82/17.18 (mean 17.44).

## The adjudication

- **G1 walk identity: PASS** — both AB runs print the same FNV walk hash (`121982e1b7878d05`) across both cadences; stop-identity cells: visible prefixes identical in all three stop cells, overshoot bounded (0, 0, 3) and never visible. The arm is correct; it just does not move the wall.
- **The pre-registered adopt bar (resident wall ≥0.3 ms/tok vs same-box resident-K8): FAIL on the drift-free read.** The cross-run pair-mean delta (0.56) nominally clears the bar, but the run-level drift control refutes it: the incumbent arm (cadence-8 code, byte-identical env) moved +0.5 ms/tok between the two runs — pure box drift, no arm present — while the within-run resident-vs-incumbent delta stayed fixed (−2.415 vs −2.413). The within-run delta is the drift-free instrument (both arms share the same minutes-wide window); it read a 0.002 ms/tok cadence effect. The katgpt-rs sequential-AB law (sequential arms on a loaded box differ 5–22%) is the exact class, and this window was loaded.
- **The pre-registered second leg (harvest share of the host window drops materially):** not reachable at this posture — with the wall unmoved, a host-window re-decomposition would measure the same fence share at any cadence; skipped rather than measured-to-nothing.
- **TTFT side-effect (recorded, not a bar):** the burst sizes change as designed (K8: 150.9 ms first burst of 8; K32: 488.5 ms first burst of 32) — a deeper cadence trades TTFT for batch amortization exactly as the mechanism predicts, with no wall change to show for it. Any future promotion talk would need this axis priced; moot with the bar failed.

## Mechanism (why this was the expected answer)

The lever was named by T1's counters as the harvest fence, and the arm raised the harvest cadence K — but Issue 1181's finding (Bench 1218) already showed the resident queue is **pool-bound ~1-deep, not ring-bound k=8** (enqueue p50 14.5 ms ≈ one GPU token; the serializer is the allocation backpressure inside the CubeCL launch path). A pool that deep never fills a K=8 ring, so K=32 changes nothing the host waits on — the ring depth is not the serializer. The within-run identity (0.002 ms/tok) is that prediction measured. The residual 0.68 ms/tok pool to the Issue-1168 P2 floor (15.32) lives in the pool/allocator path, not the cadence — the completion-counter drain (T2's named alternative) or the i1199/1199-class allocator rungs are the remaining levers, each needing its own pre-registered cell.

## Verdict

T2 **measured-negative** — `RIIR_RESIDENT_HARVEST_K` stays opt-in (the substrate knob `d2255b8b` + the bench wiring `b70707199` remain: unset byte-identical, promote-only, parser grammar pinned). No default change, no 1165-style promotion verdict needed (the negative path per the issue's own law). T3 does not gate open. The lease folds with this record; the named alternative lever (completion-counter drain) is filed as its own issue — [`.issues/1230_resident_completion_counter_drain.md`](../.issues/1230_resident_completion_counter_drain.md), bars pre-registered there with the 1229 lessons folded in (quiet-window hard premise, within-run delta as the instrument).
