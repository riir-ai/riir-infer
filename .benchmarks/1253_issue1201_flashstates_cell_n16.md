# Bench 1253 — Issue 1201 flashstates rung-1 cell (N=16, cap 2048, seed 0, frac 1/16): arms a0/a0p/a1/a2 at the league posture

**Date:** 2026-11-22 (data completed 2026-11-20 22:26 +07 on bmo)
**Issue:** [1201_flashstates_hybrid_state_serving.md](../.issues/1201_flashstates_hybrid_state_serving.md) · **Plan:** 630 T3 ladder rung 1 (the amended (16, 2048, seed 0) @ frac 1/16 pin)
**Model:** Qwen3.8-27B Q4_K_M (sha256 `e00082f7…`, the amo uzu27b copy shipped bit-identical over the bridge — the bartowski repack's attn_qkv Q8_0 refused the loader; the repacker lesson recorded in the issue)
**Instrument:** `crates/riir-infer-gpu/src/bin/flashstates_cell.rs` (the run's own commit pin; bars computed in-instrument, `bars.json`)
**Job:** bmo exclusive `01a1f72b-0a27-7a71-ad01-2dd208603bdd` (est 30h/timeout 45h), start ~13:0x +07 11-20, COMPLETE 22:26 +07 11-20; the run dir `/Users/katopz/git/riir-infer/.benchmarks/flashstates_cell/16_2048_s0/` on bmo (a0/a0p/a1/a2 rows+summaries+bars.json+manifest+`job_20261120T1100_resume.log`)

## Taint discipline (the row-split law v2, recorded pre-run in the issue)

a0's rows split at req 39 (poison-1: the lost-run + foreign-build window) and req 45 (poison-2: the sprint!050/seal-remake shared legs) — **46 clean-floor a0 rows (req ≥ 45)** feed the wall-clock bars; POISON-3 corrected: the post-13:4x stamps were REAL foreign builds (the four-process clang/ld graph a verified no-op build cannot produce); the wrapper self-match theory withdrawn (sprint!087 rewritten, seal-sprints `1bbcda8`, three real findings F1/F2/F3: shared-lane admission during exclusive runs, sticky POISONED labels, orphaned-compile probe gaps). a0p/a1/a2 rows born on the clean run. G1 rows taint-immune (bit-identity).

## The bars (pre-registered, computed in-instrument) — verdict: **FAIL (honest)**

| bar | definition | measured | verdict |
|---|---|---|---|
| 1 (a1) | uncached-token ratio ≤ 0.20 of a0 | **1.000** (a1 median_uncached 3270 = a0's 3270) | **FAIL** — the lane did not reduce uncached tokens |
| 2 (a1) | p90 TTFT ≤ 0.50 of a0 | **2.838** (49.6s → 141s-class p90; a1 ttft_p90 2.84× a0's) | **FAIL** — ~2.8× worse |
| 3 (a1) | decode median ≥ 0.98 of a0 | **0.960** (8.88 vs 9.25 tok/s median) | **FAIL** — ~4% slower |
| 4 (attribution) | a1 ≈ a0p on uncached (≤ 1.10×a0p) → the gain is cache composition | a0p **340** vs a1 **3270** (9.6× worse) | disclosure does NOT fire — the control dominates |
| 5 (a2) | median_uncached ≤ a1 at the same budget | a2 ≤ a1 (3270-class) | **PASS** |
| 6 | G1 resumed-vs-full bit-identity (a1) | PASS on 8 sampled requests | **PASS** — the resume machinery is byte-honest |
| 7 (a2 set) | uncached ≤ 0.20 / ttft ≤ 0.50 / decode ≥ 0.98 | 1.000 / 2.682 / 0.963 | **FAIL ×3** |

N=16 sessions, requests 91/arm, g1_checked 8. Full summaries: a0 (decode_median 9.246, ttft_p50 30.7s, ttft_p90 49.6s, median_uncached 3270), a0p (decode_median 9.257, ttft_p50 32.9s, p90 51.0s, **median_uncached 340**), a1 (decode_median 8.880, ttft_p50 60s-class, p90 141s-class, median_uncached 3270), a2 (decode_median 8.902, ttft_p50 60.2s, p90 132.9s, median_uncached 3270).

## The verdict + the read

**The flashstates multi-session lane (a1/a2) FAILS its efficiency bars at the league posture** — p90 TTFT ~2.7× worse, decode ~4% slower, zero uncached-token reduction — while the **a0p per-session-cache control dominates** the uncached axis (340 vs 3270, ~9.6×). Per the promotion law: fail = demote, the incumbent full-prefill + the a0p posture stay; flashstates was never promoted (opt-in default-off stands — no code change). The bar-4 attribution arm names the actual positive: a0p's per-session-cache composition IS the measured win on the uncached axis — that posture is the lane's real deliverable, already resident.

Taint honesty: bars 1/2/5 read a0 post-45 rows only (the pre-39/45 windows excluded per the row-split law); the arms share one window/box, so intermittent shared-runner build noise damps in the ratios (2.45×-class margins per plan 630) — the FAIL margins exceed the noise class.

## Provenance

- bmo QUIET at launch (load 2.0, no contenders) then shared-leg poisons honored as taint boundaries (39/45); the exclusive-window announcement mailed to the likely builders.
- Box: bmo (M5-class quiet runner), release build, the run's own commit pin; two instrument runs' manifests digest-consistent (the store-preservation discipline: this record's inputs are the run dir's committed copies — the Bench-148 loss mode stays closed).
- Session: zed-84acf899@amo (the collecting turn; the cell data by zed-00f2c1ae@amo's supervised relaunch lane, job `01a1f72b`).
