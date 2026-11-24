# Bench 1239 — Issue 1235: the rec kernel's q-defer fusion price cell — the arm is UNDER the bar at every shape (0.977× at the serving p=512); lane closes CONFIRMATORY, planes=8 stands

**Status:** RECORD — cell complete on the right tree (worktree at `d5f7e6b3`, the tree guard printed `q_defer refs: 18` before building): **the pre-registered bar (≥1.03 qdefer-vs-plain at the serving planes=8, p=512) reads 0.977× — the fusion arm is 2.3% SLOWER; every ladder arm shows the same under-performance polarity at every shape; the lane closes CONFIRMATORY. The MSL compiler's own scheduling already extracts the overlap the explicit deferral tried to win — the ~2-3% cost is the pipeline's register/branch overhead. The bit-exact residue of the gdn_rec reduction chain is priced as compiler-saturated at this form; planes=8 stands unchanged (the serving default).**
**Date:** 2026-11-17
**Session:** amo-zed, 1794904830 (T1 `d5f7e6b3`; the G1 sweep covers every ladder arm × q_defer {false, true}, bit-exact — `mr_is_bit_identical_to_multi_token ... ok` on Metal/amo; lib 406/0 at the gated posture)
**Issue:** `.issues/1235_rec_qdot_off_chain_fusion.md` (the Bench-1233 named "layout/fusion restructure" class — the bit-exact residue after Issue 1233 folded the chunked class)
**Box:** bmo (M5 Ultra, 96 GiB, AC), Exclusive lane; the rig job ran the probe in a worktree beside the clones (the sibling path-dep layout); job `01a1e98a` (log `dba22d26`), build 25.09s warm, probe 1.16s
**Instrument:** `test_rec_mr_timing_probe` extended — arms = the plain incumbent + every ladder arm × q_defer {false, true}, interleaved median-of-9, the `[bar]` row spelled out per shape; `--release`; drain-read sync
**Disclosure:** one clippy nit (`if let ((..))` double-parens, the probe's bar row) rode the T1 commit — fixed in this close commit, not re-measured (display formatting only, zero kernel codegen). Two earlier queue attempts failed honestly and are part of the record: attempt 1 measured bmo's MAIN clone (wrong tree — my `cd` overrode the rig's worktree; its numbers are a free Bench-1233 replication: p=512 planes=8 1.443× vs the record's 1.448×), attempt 2 died on the sibling path dep (`/tmp` worktree breaks `../katgpt-rs` — the worktree-must-sit-beside-the-clones lesson, now in the rig-job discipline).

## The tables (median of 9 interleaved; ratios vs the plain multi-token baseline)

**p=512 — THE SERVING SHAPE (the decider):**

| arm | plain | qdefer | delta |
|---|---:|---:|---|
| planes=2 | 1.352× | 1.333× | −1.4% |
| planes=4 | 1.402× | 1.376× | −1.9% |
| **planes=8 (DEFAULT)** | **1.441×** | **1.409×** | **−2.3% — THE BAR ROW: 0.977×, under (≥1.03 graduates)** |
| planes=16 | 1.379× | 1.342× | −2.7% |
| planes=32 | 1.375× | 1.360× | −1.1% |

**p=2048:** planes=8 plain 1.496× / qdefer 1.485× — bar row 0.992× (a wash). **p=128:** planes=8 plain 1.213× / qdefer 1.193× — bar row 0.984× (also under; planes=32 tops the short shape at 1.233×, matching Bench 1233's non-serving note).

**Control rows:** the plain ladder reproduces Bench 1233 within noise at every shape (p=512 planes=8 1.441× vs 1.448×; p=2048 1.496× vs 1.514×; p=128 the planes=4-tops/32-close pattern) — the negative is measured on a sound instrument.

## Adjudication (pre-registered bars, all read)

1. **Engagement** — the arm's own vacuous counter: the G1 sweep dispatched the deferred specialization (`rec_qdefer_launch_count > 0`), and the probe's qdefer rows populated at every arm — reached, not assumed.
2. **G1** — bit-exact at every arm (output AND state bits) — the arm is CORRECT; it is just not faster.
3. **Price** — 0.977× at the deciding shape, under the ≥1.03 bar, on the wrong side: **CONFIRMATORY close.** The deferral's premise (the q-butterfly sits between the state update and the next token's k-side on the critical path) is REFUTED — the compiler already schedules it off-chain; the explicit form pays for the pending registers + the drain branch.

## What remains for gdn_rec (the honest residue, now fully priced)

- The **chunked class**: Issue 1233's close — measured-NOT-BIT-IDENTICAL (Bench 705), the owner-numerics lane, CUDA-refused prior. Re-open only on an owner sign-off (files the retention-protocol issue, not a lane here).
- The **fusion class**: THIS record — compiler-saturated at the q-defer form. Re-open lever: a DIFFERENT explicit-scheduling form (e.g. the r-axis multiple-rows-per-lane arm the MR module doc names — Bench 896's CUDA prior is negative; the Metal answer would be its own priced cell), never a re-tune of this one.
- **Planes=8 stands** as the serving default; the ladder axis and both named kernel-work classes are now measured out on this silicon.
