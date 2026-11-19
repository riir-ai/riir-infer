# Bench 1243 — Issue 1231 T2: the pipelined snapshot-future harvest wall cell (ADOPT bar MET)

**Issue:** `../.issues/1231_resident_pipelined_snapshot_harvest.md` (T2; lease `amo-idle-hk` epoch 1795062768). **Box:** amo (M5 Ultra 96 GiB), AC, QUIET-EXCLUSIVE — load 1.43 at the run's start, 1.43–1.74 across the whole window (`uptime` + `ps` greps logged at cell start; zero cargo/riir/katgpt compute consumers; `CARGO_TARGET_DIR=/tmp/i1231cell` isolated). **Posture:** the bench_831 battery (ctx=2048, n=512, pairs=3, census uniform-cache override ON), burst K=8 (the resolver default), ring depth 2×K, release profile, features `cubecl_runtime,ternary_gemv,deltanet_recurrence_rowpar,params_handle_cache,ternary_gemm_batched`. Model: `riir-train/data/Ternary-Bonsai-27B-Q2_0.gguf` (the bench default path). Raw logs + the cell script: `.benchmarks/1243_i1231_t2_pipeline_wall_cell_logs/`.

## Verdict — ADOPT BAR MET (T2 PASS), T3 (serving wiring + GOAT + owner verdict) unblocked

| Bar (pre-registered, BEFORE the run) | Measured | Verdict |
|---|---|---|
| Pipelined improves ≥ 0.3 ms/tok vs the same-cell cadence arm (within-cell delta, median of pairs) | **−0.351 ms/tok** (cadence 14.98 → pipelined 14.63; pairs 1/2/3 all −0.351, ratio 0.977×; warm-up pair −0.198 excluded from the instrument) | ✅ MET |
| G1 walk identity (FNV + pairwise + across pairs) | FNV `121982e1b7878d05` — IDENTICAL to the cadence cell's ref hash (same-session cross-cell agreement); pairwise asserts green all 4 arms × 4 pairs | ✅ |
| Stop cell: visible prefix truncation-exact | 65 tokens (seed + generated), "both arms IDENTICAL" — cadence AND pipelined stop cells | ✅ |
| Stop overshoot ≤ 2K (≤ 16 at K=8) | 7 tokens past the stop (on-GPU cost of the K=8 cadence + the one-burst lead, never visible) | ✅ |
| TTFT (reported, not gated) | pipelined 222.2 ms vs cadence 142.7 ms — the one-burst lead costs ~80 ms of first-visible latency (the streaming-shape axis; the adopt decision carries it disclosed) | reported |

Throughput read: cadence 66.74 tok/s → pipelined **68.34 tok/s** (+2.4%) at the resident wall posture — the fence now overlaps the next enqueue phase instead of serializing after it, exactly the issue's mechanism. The delta sits just over the 0.3 bar (−0.351): real, stable (three pairs, identical), but NOT large — T3's GOAT gates decide the promotion, this cell only unblocks it.

## The EOS-drain defect the cell caught (fixed same session — the cell's first catch)

The first wall-cell run RED at G1: `pair 0 arm 1: walk diverged within the pair` — the pipelined arm's walk ended at **505 of 512 tokens** (the common 505-prefix byte-identical; no corruption, no reordering). Root cause: `resident_chain_pipelined` created its LAST snapshot future at the final burst boundary (i=504); the 7 tail tokens enqueued after that snapshot (505..511) were never harvested — the post-loop block drains only the outstanding snapshot. The cadence arm's post-loop `harvest()` drains the tail; the stop cell's `!drained_stop_burst` path already carried the correct pattern (`batch.extend(stream.harvest())`). **Fix (riir-ai, this session):** one EOS-drain block in `resident_chain_pipelined` after the outstanding-future block — the same contract the stop cell's tail path carries. Re-run: green, the FNV identity restored, the numbers above are the FIXED run. The stop cells passed before the fix (their tail path was already correct) and were not re-run.

## Cell inventory (all four, same binary)

1. `bench_831_decode_long_resident_ab` (the same-cell cadence arm): resident delta −1.787 ms/tok vs incumbent (16.75→14.96; ratio 0.893×) — the 1165-era resident gain reproduced at this posture; ref FNV `121982e1b7878d05`.
2. `bench_831_decode_long_resident_pipeline`: **the T2 instrument** — delta −0.351 ms/tok, G1 identity vs the cadence walk.
3. `bench_831_resident_stop_identity` (cadence): prefix identical, overshoot 0, TTFT 142.7 ms, wall 0.95 s / 65 visible.
4. `bench_831_resident_stop_identity_pipeline`: prefix identical, overshoot 7 (≤ 2K), TTFT 222.2 ms, wall 1.04 s / 65 visible; burst sizes all 8 (nine bursts).

## Disclosures

- The 1218-era run's rig-wrapper poison loop (Bench 1229's incident) did NOT recur: this cell ran as a plain background process on the box (no rig wrapper), an explicit `I1231-T2-CELLS-DONE` sentinel echoed at script end (the 1229 lesson applied).
- The 7.1 GB model load (~2 min) dominates each cell's wall; the walls are measured AFTER load, per the cell's own instrumentation.
- The out dir carried 1218-era leftovers (`k8_pipeline.log`, `stop64/65_*`); the T2 logs are the four `k8_cadence_*`/`k8_pipeline_*` files committed here.

## T3 (next, unblocked — not this cell's scope)

The serving loop wires the pipelined schedule behind a PROMOTE-ONLY env (the resolver pattern), then the GOAT gates + the 1165-style owner verdict. The env does NOT exist yet by design (a knob arming nothing is config rot, P22).
