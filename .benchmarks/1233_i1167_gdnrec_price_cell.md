# Bench 1233 — Issue 1167: the gdn_rec engagement+price cell — the MR planes ladder re-scored with planes=16/32 and the serving chunk p=512 — planes=8 STANDS at every serving shape; cell closes CONFIRMATORY

**Status:** RECORD — cell complete ×2 windows (pre-registered bars in `.issues/1167`): **G1 bit-identity EXACT over all six ladder arms** (the B1 case set vs the incumbent multi-token kernel — the ladder extension's safety pin); **price: the DEFAULT planes=8 is the winner at the deciding shape p=512 (1.448×) and at p=2048 (1.514–1.516×)**; planes=16 (−3.6% at p=512) and planes=32 (−1.5%) land UNDER the pre-registered 3% graduation bar → **no arm graduates; the cell closes CONFIRMATORY — planes=8 stands at the post-f16B serving posture, and the remaining gdn_rec ground (Bench 1227's 32.42 ms/frame, 6.9% share) needs kernel WORK (the chunked intra-chunk form), not a rung flip.** The only shape where higher planes lead is p=128 (p32 1.268× vs p8 1.231×, +3.0%) — not a serving shape since the 1146 lane moved the chunk to 512; recorded for completeness, never a flip argument.
**Date:** 2026-11-17
**Session:** zed-5cfe765f@amo, 1795800000
**Issue:** `.issues/1167_metal_prefill_followup_lane_b_c.md` (the gdn_rec engagement+price cell — the lane's named next act after Bench 1227's re-rank)
**Box:** bmo (M5 Ultra, 96 GiB, AC), Exclusive lane, load 1.07–1.22 during both windows
**Instrument:** `test_rec_mr_timing_probe` (isolated-kernel, interleaved median-of-9, warmup, drain-read) + `mr_is_bit_identical_to_multi_token` (G1), `crates/riir-infer-gpu/src/deltanet_recurrence_mr_cubecl.rs`
**Pin:** bmo clone at `2bf61e3e` (the instrument commit: ladder [1,2,4,8] → [1,2,4,8,16,32]; probe shapes p=128 → **[128, 512, 2048]** — p=512 the SERVING chunk since the 1146 lane, never probed before this cell; valid_planes + validity-test negative control follow; DEFAULT arm stays LADDER[3]=8 index-pinned — zero serving change)
**Job:** `01a1e78e-b2d2` (bmo exclusive; both windows 03:31–03:32Z; the job ran the pre-G1 script revision — the G1 was run separately on the same box same posture, `ok` 0.08 s, log line `mr_is_bit_identical_to_multi_token ... ok`)
**Prior evidence carried as prior:** Bench 896 (CUDA sm_89) — the r8 arms LOST to r4w4; the Metal answer now agrees at the serving shapes. Plan 607 E1 (2026-10-31, M5) — p=128: p8 1.211× (this cell: 1.231×, the ladder's control discipline reproduced); p=2048: p4 1.493× vs p8 1.475× (this cell: p8 1.514–1.516×, p4 1.511–1.536× — the two top arms stay within noise of each other at the long reference; neither clears the bar over the other, and the default stays).

## The tables (two windows, median of 9 interleaved, ratio vs the incumbent kernel baseline)

**p=512 — THE SERVING SHAPE (the decider):**

| arm | window 1 | window 2 | two-window verdict |
|---|---:|---:|---|
| planes=2 | 1.332× | 1.332× | below |
| planes=4 | 1.427× | 1.427× | below |
| **planes=8 (DEFAULT)** | **1.448×** | **1.448×** | **TOP** |
| planes=16 | 1.396× | 1.396× | −3.6% vs p8 — UNDER the 3% bar |
| planes=32 | 1.426× | 1.426× | −1.5% vs p8 — UNDER the bar |

**p=2048 (long-launch reference):** p2 1.419–1.434× · p4 1.511–1.536× · **p8 1.514–1.516×** · p16 1.376–1.381× · p32 1.407–1.425× — p8 tops again (p4 within noise; no bar clear).

**p=128 (the pre-1146 chunk, not served):** p8 1.231× · p16 1.262× · **p32 1.268×** — higher planes lead by ~3.0% at the short shape; launch-overhead-dominated regime, no serving argument, recorded only.

## Adjudication (pre-registered bars, all read)

1. **G1** — EXACT over all six arms (incl. 16/32): `mr_is_bit_identical_to_multi_token ... ok`. The ladder extension is safe in-tree regardless of the price verdict.
2. **Engagement** — every arm dispatched and timed (the per-arm rows above; the launch-counter vacuous guard is the module's own).
3. **Price** — no arm ≥ planes=8 + 3% at p=512 → **CONFIRMATORY close**: planes=8 stands; the gdn_rec lever space above the promoted arm is EMPTY at the serving shapes.

## What remains for gdn_rec (the honest residue)

The 32.42 ms/frame (6.9% share, Bench 1227) is the MR kernel at its best ladder arm. The remaining lever classes are KERNEL WORK, not rungs: the chunked intra-chunk form (O(p²/d) parallel intra-chunk pass + inter-chunk carry — the CUDA lane's `RIIR_PREFILL_GDN_CHUNKED` class, Ada-MMA-specific there, promotion-REFUSED; a Metal twin would be a NEW kernel family), or a layout/fusion restructure of the reduction chain. Both are separate lanes with their own G1+A/B; this cell's verdict prices the ENTRY: 6.9% share × whatever fraction the new form reclaims — the chunked form's CUDA refusal (banked defense, Issue 971 watch) is the prior to beat. **Recommended disposition: gdn_rec is PRICED-CLOSED for rung work; any chunked-form kernel lane files its own issue.**

## Session-integrity notes (disclosed)

- The exclusive job ran the script revision that lacked the explicit G1 invocation (`-- --ignored` runs only `#[ignore]`d tests — the timing probe — while the G1 is a plain test); the G1 was run separately at the same pin/posture on the same box within minutes (green). The pre-registration's bar was read from that run, not from the job's log.
- The mid-flight script re-scp corrupted the job script's tail (bash reads scripts incrementally; the byte shift hit the trailing `echo`) — after both windows completed. No measurement touched.
