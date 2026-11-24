# Issue 1247: the M3-Max wgpu per-driver-interaction latency hunt (the 1203 residue)

**Status:** In progress — claimed (zed-m3-i1247@m3, exp 2026-11-26T19:16Z). **T1 LANDED `bca26b22` (bench [1259](../.benchmarks/1259_i1247_m3_driver_interaction_decomposition.md), 2026-11-25):** the decomposition probe (cells A/B/C/D/F/G + parked H + pressure E) ran on m3 — quiet-window verdict: the absolute layer is NOT the write/submit primitive (allocation 1.7–7 µs size-flat, small write 3 µs, idle submit p50 2.6 µs, poll 20 µs, backlog depth ≤4096 never blocks, fence scan ≤1.3 µs, 24 GB pressure identical). The 1203 counter re-read RE-ATTRIBUTES the ms-class "write": the fork's Write arm is `flush()`-then-write, and the scheduler-level cap `tasks_max = 32` (RuntimeOptions DEFAULT_MAX_TASKS, env `CUBECL_WGPU_MAX_TASKS`) flushes ~32 accumulated compute tasks before EVERY write — the ms-class cost is the per-flush submit+commit chain amortized onto write arms. Cap-lift cell consistency: probe_16384 (REGULATE 0, wall unmoved, ENQUEUE_WRITE grew 19.4→45.4 s) = re-attribution, zero net. **NEXT (T2): the cadence cell in the probe (reproduce 32-task flushes + 64-write arms + 512-task polls over real passes) + the `CUBECL_WGPU_MAX_TASKS` A/B on the real forward (env-tunable TODAY; never benchmark a BUSY box, preflight PROVENANCE quoted).** Cell H (raw Metal) PARKED: the wgpu-core dispatch-layer extraction segfaults; needs the cubecl-wgpu backend-side extraction (follow-up, not a blocker). F's drain cost pins the per-CB retirement at ~8.6 µs on m3 — the F2 candidate is batching CB count per pass (1.5–4.2k/pass today). Spawned by Issue 1203's close (2026-11-24): the m3 stall is DIAGNOSED (per-driver-interaction latency, allocation-size-bucketed) but the ABSOLUTE layer is unexplained and unfixed — this issue owns the hunt and the posture law.

## What 1203 established (the record: riir-infer HISTORY.md 2026-11-24 arc; raw logs `.benchmarks/i1203_m3_stall_cells/`)

Every wgpu/Metal driver interaction on this box costs milliseconds where the healthy class is microseconds — and the per-unit cost BUCKETS BY ALLOCATION-SIZE CLASS (non-monotonic in ctx: 1024 → 230s, 4096 → 69s [mapped minimum], 24576 → 601s for the same counter counts):

- `write_buffer`: 3.2–19.7 ms per buffer (healthy: µs)
- `queue.submit()`: 5–33 ms per submit
- REGULATE (the ≥512-in-flight poll): 2.6–16.7 s per block
- Launch machinery (0.7–1.4 µs) and readback (0.04–0.06 ms) are EXONERATED — the mass is in the write/submit/wait path plus a 14–74% unattributed remainder below the Issue-761 counters.

Two layers: the ctx-bucket amplifier (exploited today: bench pass1 **18 tok/s @ ctx 4096 vs 4 tok/s @ 24576** — posture alone) × the absolute base (~66× residual at the best posture).

## This issue owns

1. **The absolute-layer hunt** — why is each `write_buffer`/`submit` ms-class on M3 Max + this vendored cubecl-wgpu 0.11.0-pre.2/wgpu-hal 30.0.0 stack when the same stack is µs-class on amo (m5ultra)? Candidates ranked: (a) the wgpu task/poll registry scanning cost scaling with live buffers (the REGULATE + unattributed-remainder shape fits a scan, and the bucket effect fits allocation-table size classes); (b) the Metal driver's per-command-buffer resource handling on AGX M3 Max (the #8768 per-pass driver-compile class is the closest prior); (c) `MTLCommandBuffer` commit/present scheduling on the shared queue without `wait_until_completed` (the async-pipeline law) interacting with M3 Max firmware. Instruments: wall clock + the Issue-761 counters ONLY (GPU timestamps return zeros on macOS 26+ per gfx-rs #9414); `sample(1)` for the parked-thread shape; the vendored fork's `write_to_buffer`/`submit` sites get micro-instrumented before any upstream report.
2. **The posture law for m3 dense-lane cells** — ctx 4096 is the mapped minimum for the pp512+decode workload shape; 1024 REGRESSES (2.3s → 230s class); 24576 costs 8.7×. Any m3 dense-lane cell (1201's FlashStates arm, league re-cells, the decode-ablate matrix — still box-impossible at 24.8 s/step, ~13 h) selects its ctx FROM THE BUCKET MAP and records it beside every number.
3. **The decode-ablate matrix** — box-impossible at the stalled rate on m3; if the absolute layer fixes, the matrix becomes runnable (~13 h at the 24576 rate; minutes-class if the fix reaches µs interactions). Not runnable on amo (the stall is m3-family-specific — amo is the qualified box, no stall to ablate).

## Refusals

- No upstream PR without the distill/outbound gates (the vendored fork is transcription territory).
- No demotion of the league ctx default fleet-wide — the bucket law is m3-family-specific (amo/bmo measured normal); the m3 arm selects its own ctx.
