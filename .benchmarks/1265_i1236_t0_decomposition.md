# Bench 1265 — Issue 1236 T0: the gdn_rec fixed-vs-slope decomposition (the chunked-form demote gate)

**Date:** 2026-11-26 (cell completed 2026-11-25T14:59Z; collected + analyzed 2026-11-25T22:2xZ) · **Cell box:** bmo (M5 Ultra 96 GB, the rig-exclusive queue) · **Analysis box:** amo · **Session:** zed-amo-i1236-t0@amo
**PROVENANCE:** the TIMING cell is the rig queue's bmo-exclusive posture (job `01a20a3a-e81f`, the 11-24T23:3xZ re-queue row in `.issues/1236`; detached worktree `~/git/i1236-t0-wt` refreshed in-job — worktree HEAD `7a9b8065` at collection time, detached at origin/develop per the re-queue text). The tee'd log carries no PROVENANCE line and no load reading — disclosed; the decision statistics are WITHIN-RUN ratios (fixed-vs-slope shares of the same session's walls), second-order robust to any mild contention the run may have overlapped, and the arm ratios cross-check Bench 1233 within its ±2.5% band (below). The ANALYSIS is a pure read of the captured output (zero timing claims of its own).
**Instrument:** the issue's exact re-queue command — `cargo test -p riir-infer-gpu --release --features production test_rec_mr_timing_probe -- --ignored --nocapture` with `CARGO_TARGET_DIR=/tmp/i1236_t0_tgt`, tee `/tmp/i1236_t0_out.txt` (preserved: [`i1236_t0_decomposition/i1236_t0_out.txt`](i1236_t0_decomposition/i1236_t0_out.txt)). The probe is the Bench-1233 instrument posture: n_v=48, hd=128, median of 9 interleaved runs per arm, MR planes ladder 1→32 with/without `qdefer`, at p ∈ {128, 512, 2048}. Zero new code (the T0 scope).

## The question (issue 1236 T0, pre-registered)

Decompose the gdn_rec MR-kernel wall at the serving shape p=512 into per-launch FIXED vs per-token SLOPE components. **Demote bar: if the per-token slope share is < 50% of the kernel wall, the chunked form's parallelism has nothing to amortize — the lane routes to the restructure class only.** A slope-dominated split licenses lever class 1 (the chunked intra-chunk form) with T1 opening.

## The data (planes=8 — the incumbent family; best arm per shape bolded in the raw log)

| p | planes=8 plain | planes=8 qdefer | run-best arm |
|---|---|---|---|
| 128 | 0.372 ms (1.180×) | 0.361 ms (1.216×) | planes=16/32 qdefer 0.352 ms (1.246×) |
| 512 | 0.884 ms (1.411×) | **0.866 ms (1.440×)** | planes=4 0.862 / qdefer family 0.866–0.887 |
| 2048 | **3.504 ms (1.548×)** | 3.558 ms (1.525×) | planes=8 plain 3.504 (planes=4 qdefer 3.532 next) |

Cross-check vs Bench 1233 (the confirmatory close, same instrument): planes=8 at p=512 read 1.448× there; here 1.411× (plain) / 1.440× (qdefer) — within the ±2.5% band; p=2048 planes=8 1.548× vs 1233's 1.514–1.516× — consistent. The planes=16/32 regression at p=2048 (3.83–3.91 vs 3.50 ms) re-confirms 1233's verdict: **planes=8 stands**; the rung-lever space above the promoted arm stays EMPTY.

## The decomposition (fixed vs slope at the serving shape)

Least-squares fit `wall(p) = fixed + slope·p` over the three serving-relevant planes=8 rows:

| arm | slope (µs/token) | fixed (ms) | slope share @ p=512 |
|---|---|---|---|
| planes=8 plain | 1.653 | 0.106 | **95.7%** |
| planes=8 qdefer | 1.690 | 0.081 | **99.9%** |

Segment marginals (planes=8 plain — the wall is NOT linear in p):
- p 128→512: (0.884 − 0.372) / 384 = **1.333 µs/token**
- p 512→2048: (3.504 − 0.884) / 1536 = **1.706 µs/token**

The marginal per-token cost RISES with scan length (superlinear regime at larger p — memory-bandwidth saturation / chunk-count effects in the MR geometry). The launch-fixed component at p=512 is **0.01–0.11 ms of the 0.884 ms wall ≈ 1–5%** (the two-point estimate at the serving segment gives 0.010 ms; the 3-point LSQ gives 0.081–0.106 ms — the honest band spans both, and every member of it is small).

## VERDICT: T0 PASSES its own demote bar by ~2× the bar — LANE ROUTES TO LEVER CLASS 1 (the chunked intra-chunk form)

The per-token slope share at p=512 is **95–99%**, far above the pre-registered 50% demote bar. The chunked form's parallelism has ~17× more per-token work (0.85 ms) than launch-fixed work (0.05 ms band midpoint) to amortize at the serving shape. **T1 opens** (the G1 harness FIRST — the kernel-recovery law: extend `mr_is_bit_identical_to_multi_token` to the chunked/carry family before any timing, the inter-chunk carry composed in scan order per Issue-940).

Secondary routing note: the restructure/fusion class (lever class 2) is DEMOTED to secondary at the serving shape — its target (the per-launch fixed cost) is 1–5% of the p=512 wall; even a total launch-cost elimination caps the win at ~5%. The p=128 launch floor is real (no arm went below 0.352 ms; the baseline itself is 0.439 ms) but p=128 is context, never a flip argument (the issue's own constraint). The one fusion-shaped observation worth carrying into T2's A/B set: `qdefer` wins the launch-bound regime (1.216–1.246× at p=128), washes at p=512 (1.440× vs 1.411×), and slightly trails at p=2048 — the chunked prototype's arm set should carry both variants, since a chunked form changes the launch/granularity mix.

## What this does NOT claim

- No serving change, no promotion, no kernel work landed: T0 is the instrument cell the issue pre-registered ("zero new code"); the incumbent MR planes=8 arm is untouched.
- Not a Metal-vs-CUDA comparison: the CUDA chunked prefill prior (Bench 896 / riir-ai Issue 971's REFUSED promotion) stays PRIOR — this cell prices the OPPORTUNITY on Metal (the AGX scheduler question is what T2/T3 answer).
- Not a bit-exactness claim: the probe is timing-only; bit-identity is T1's gate.
- The superlinearity observation (1.33 → 1.71 µs/token) is an instrument-level read at three points, not a fit claim; it licenses the chunked form's premise (a scan-length cost term worth splitting), never a projected number.

## Landing

- Claim → analysis: `fa0a91ba` (the claim row), this record + the issue update land together.
- The lane's next act: T1 (claim separately per the issue's task rule).
