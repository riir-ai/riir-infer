# Bench 1247 — the m3 per-driver-interaction decomposition (issue 1247 T1)

**Status:** RECORD — T1 instrument + first decomposition cells LANDED 2026-11-25 (zed-m3-i1247@m3, epoch 1795547784). The absolute layer is NOT the write/submit primitive cost — the µs-class cells below exonerate allocation, pressure, small-write path, idle submit, poll machinery, fence scan, and command-buffer backlog depth. What remains unexplained concentrates on the real-forward interaction pattern, and the 1203 counter re-read (below) re-attributes the "ms-class write" to the WRITE-ARM FLUSH, not `queue.write_buffer` itself.

**PROVENANCE:** quiet-window run `PROVENANCE: power=AC Power load=2.28 swap=1272.12M canary=skipped powermode=2(high)` (bench_preflight PASSED; the earlier shape runs rode load 6.5–17.4 — sibling builds — disclosed per run; the quiet run reproduces the same class structure).

## Instrument

`crates/riir-infer-gpu/src/bin/i1247_driver_interaction_probe.rs` (feature `metal_tensor_gemm` — the `metal` dep for the parked H cell; whole-file cfg + the named [[bin]] row). Public wgpu API only, no model load; wall clock only (GPU timestamps banned, #9414). Cells:

| cell | subject |
|---|---|
| A | `device.create_buffer` — the fresh per-write staging MTLBuffer wgpu-core pays per `write_buffer` |
| B | `queue.write_buffer` full path (fresh staging + memcpy + blit encode + bookkeeping) |
| C | `queue.submit` of an empty encoder |
| D | `device.poll(Wait{submission_index})` after a trivial submission |
| F | write latency vs in-flight CB backlog depth (64→4096 CBs in flight, real compute passes) |
| G | `poll(Poll)` fence-scan cost vs pending CB count (256/512/768) |
| H | raw Metal primitives — **PARKED** (the wgpu-core dispatch-layer `raw_device()` extraction segfaults; needs the cubecl-wgpu backend-side extraction) |
| E | 12–24 GB resident pressure repeat |

## Results (quiet window, m3, macOS 27.2.1)

| cell | reading | verdict |
|---|---|---|
| A create_buffer 4KB–64MB | 1.7–7.1 µs (size-flat) | **staging allocation EXONERATED** — not the ms class |
| B write_buffer 4KB | 3.0–3.8 µs | small writes µs-class — **the 1203 "ms per write" is NOT this primitive** |
| B write_buffer 64KB / 1MB / 64MB | 10–18 µs / 80–98 µs / 5.4–6.2 ms | scales with bytes = the memcpy; 64MB @ ~10 GB/s = WC-store bandwidth, mundane |
| C submit empty | p50 2.6–3.3 µs (mean skewed by load outliers) | µs-class |
| D poll(Wait) trivial | 19–26 µs | µs-class |
| F write @ backlog 64→4096 CBs | 11–86 µs, no monotone growth; drain poll 7–36 ms for 1024–4096 tiny CBs | **CB-budget blocking REFUTED at these depths** — 4096 in-flight CBs cost a 35 ms drain (≈8.6 µs/CB retirement) but never block writes ms-class |
| G poll(Poll) @ 256→768 pending CBs | 0.4→1.3 µs | fence scan super-linear only mildly; NOT the remainder |
| E pressure 12–24 GB | identical to idle (2.8–4.9 µs small writes) | **pressure EXONERATED** |
| H raw Metal | PARKED | follow-up cell |

## The 1203 counter re-read (the decisive re-attribution)

The vendored fork's Write arm (`enqueue_task::Write`, stream.rs:352-367) is **`flush()` THEN `write_to_buffer`** — ENQUEUE_WRITE covers a full flush (submit + regulate-within-flush) + the write. With `tasks_max = 32` (RuntimeOptions DEFAULT_MAX_TASKS — the scheduler-level flush trigger), **every Write task flushes the ~32 accumulated compute tasks first**: the 1530 "writes" of the pp-only cell each carried a pending encoder submit. The ms-class "per-write cost" is therefore the per-flush submit+driver commit chain amortized onto write arms — an interaction pattern (write interleaved with a 32-task encoder flush at ~64-write pending_write_max and 512-task REGULATE pacing), not a primitive cost.

Consistent with every 1203 reading: FLUSH count 4636 ≈ writes 1530×2 + exec-flushes (the Write-arm flush + flush_if_needed at 32 tasks + sync); REGULATE blocks (2.6–16.7 s) are GPU-paced drains of genuinely queued work; the cap-lift cell (probe_16384: REGULATE 0, wall unmoved, ENQUEUE_WRITE GREW 19.4→45.4 s) shows the drain cost re-homing into the write arm when the poll site is removed — pure re-attribution, zero net gain.

## What this pins for the issue

1. The absolute layer is NOT: allocation (A), pressure (E), backlog depth ≤4096 (F), fence scan (G), idle submit/poll (C/D).
2. It IS: the interaction pattern of the cubecl stream cadence on this box — 32-task scheduler flushes + 64-write poll flushes + 512-task REGULATE polls against an M3 Max driver whose per-CB retirement is ~8.6 µs (F drain) where amo's is µs-fast; the forward's 1.5–4.2k CBs per pass make the retirement ceiling the throughput bound.
3. Next instrument (T2): the cadence cell — reproduce the exact cubecl flush pattern (32-task flushes, 64-write arms, 512-task polls) over real compute passes in the probe, then A/B `CUBECL_WGPU_MAX_TASKS` (32→512) on the real forward — the scheduler-level cap is env-tunable TODAY (`CUBECL_WGPU_MAX_TASKS`), unlike the two Issue-1217 caps already ruled out (1203 E3: MAX_TOTAL_TASKS lift → wall unmoved; MAX_PENDING_WRITES untouched).
4. Cell H stays parked pending the backend-side extraction; it is a comparison nicety, not a blocker.

Raw runs: `/tmp/i1247/cell_run*.out` (run_final = load 2.3; run_quiet = load 2.28, the quoted table). Session: zed-m3-i1247@m3, 1795547784.
