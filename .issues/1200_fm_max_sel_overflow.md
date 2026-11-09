# Issue 1200 — FM selection count overflows FM_MAX_SEL (GPU OOB + host panic) past ~429/857 blocks — the 1191 T4 128k rho arms are dead-on-arrival

**Status:** OPEN — `- [ ]` tasks below. Diagnosis-only issue: found by the i1191-T4 midnight monitor lane (`zed-5592b062@amo`, root cause from log + pinned source), independently verified against the pinned source by the rig-ops lane (`zed-3ecf601a@amo`, 2026-11-10 ~01:45 local). NO fix landed — the envelope decision (raise vs refuse vs re-scope) belongs to the 1191 lane's pre-registered T4 semantics. Monitor-role discipline held: nobody touched the running process, the tree, or any file.

## Why

The 1191 T4 Metal cell (`job 01a1c0f5-00d6`, exclusive, started 22:37 local 11-09) died on its third arm: the 32k `rho=0.3` arm panicked at `ternary_deltanet_gpu_forward.rs:11988:32 — index out of bounds: the len is 512 but the index is 512` at pos ~27,400/32,768 (rc=101 at 17:15:19Z), ~5 min after its last checkpoint flush.

**The mechanism (verified in source):** `fm_top_rho_select_f32` computes the pick count as `k = ((rho * n).round() as u32).saturating_sub(1).min(n_blocks - 1)` — the CPU `RowRule::Budget` rule verbatim — then `sel_counts[kv] = k + 1`. **The clamp is against `n_blocks`, never against `FM_MAX_SEL` (=128, `flashmemory_budget_metal.rs:66`).** The flat buffer is `[n_kv * FM_MAX_SEL]` u32; whenever `round(rho·n_blocks) > 128` the kernel writes picks past its own kv row (clobbering the NEXT kv group's row; for the last kv group, past the buffer end — GPU-side, unchecked) and records `sel_counts = 129+`. The host fraction reader (`ternary_deltanet_gpu_forward.rs:11988`, `let block = sel[base + i]` with `base = kv * FM_MAX_SEL`, `i in 0..counts[kv]`) then walks past the buffer → exactly `index 512, len 512`.

**Overflow thresholds (f32):** `round(rho·n_blocks) ≥ 129` — at rho=0.3, n_blocks ≥ ~429 (pos ≥ ~27,328 at block 64: the 32k panic at pos ~27,392 matches within the refresh/sample cadence); at rho=0.15, n_blocks ≥ ~857 (pos ≥ ~54,784). Measured confirmation: the 32k rho=0.3 arm's per-position rate degrades smoothly up to 27,392 (11.57 tok/s) and the panic lands immediately after — the threshold is the arm's honest envelope, not a flake.

**Second latent defect (same run):** the launcher asserts `n_blocks ≤ 2048` (the shared-memory slice: 2048 score slots + 512 pair slots, `flashmemory_budget_metal.rs:361-363`). At 128k with block 64 the full length needs `ceil(131,073/64) = 2049` — the assert fires at the LAST block-open refresh (pos ≥ 131,008, 99.94%). So raising FM_MAX_SEL alone still kills both 128k rho arms at the tail; the ceiling and the block size must move together (the assert's own message says so).

**FM_MAX_SEL=128 caps the arm's honest envelope at ~54K tokens** (rho=0.15, block 64: 857 blocks × 64 = 54,848). The 128k pre-registered cells are outside it. The dense-vs-rho parity data at 32k (dense + rho_0.15 bins both persisted clean; max count round(0.15·512) = 77 ≤ 128 — no overflow, no clobber) is BANKABLE for the T4 bars regardless.

## Evidence (run.log + on-disk state, 2026-11-10 ~01:45 local)

- 32k: `dense.bin` ✓ (22:59, completed), `rho_0.15.bin` ✓ (23:35, completed, clean), `rho_0.30.bin` ⚠ **PARTIAL prefix** (00:09 checkpoint flush, cursor ≈ pos 27,000 — pre-threshold, the flushed samples are clean; NOT a completed arm; verify via the loader before any use).
- Panic: `ternary_deltanet_gpu_forward.rs:11988:32 — index out of bounds: the len is 512 but the index is 512`, 32k cell rc=101 at 17:15:19Z.
- 128k: `dense.bin` at 01:23 is a mid-flight prefix (T3 flush); dense arm was at 87,744/131,072, 19.72 tok/s at ~01:41 (ETA ~02:15 local). rho=0.15@128k is projected to panic at pos ≈ 54,784 (~42%, ~61 min at rho-arm rate); rho=0.3@128k at ~21%; nothing beyond dense will persist. (Projection, not yet measured at write time — the run.log will carry the measured panic positions after ~04:00 local.)

## Tasks

- [ ] T1 — the envelope fix (the lane's design call; three shapes on the table): (a) FAIL-CLOSED loud refusal at launch when `round(rho·n_blocks) + 1 > FM_MAX_SEL` — the arm refuses beyond its envelope instead of corrupting (the latent-defect doctrine: a documented loud refusal beats a silent clobber), or (b) raise `FM_MAX_SEL` to 512 (sel buffers 4×512 u32, smem untouched; ⚠ kernel (b)'s insertion sort is O(k²) on one thread — 307² per refresh × 2048 refreshes needs a wall-time check before adoption), or (c) re-scope the 128k cell to a larger block (block 256 → 512 blocks: rho=0.15 fits at count 77; rho=0.3 = 154 still overflows → rho=0.3@128k needs (a) or (b) regardless). The CPU `RowRule::Budget` clamp (`.min(n-1)`) is the CPU envelope; the GPU port needs the SLOT-capacity clamp on top — whichever shape wins, the invariant is: `sel_counts[kv] ≤ FM_MAX_SEL` by construction, plus a launch-time refusal naming the ceiling.
- [ ] T2 — the smem ceiling: `n_blocks ≤ 2048` vs 128k/block-64 = 2049 — raise the slice per the assert's own message OR pair the envelope refusal with it (refuse 128k/block-64 loud until the slice grows).
- [ ] T3 — real-boot confirmation of the fix on the league model (the 32k rho=0.3 cell that panicked is the natural regression cell: fix → `--resume` from the partial prefix (or `--fresh`) → completes past pos 27,400 without the panic; bins resume from cursor per the T3 runner).
- [ ] T4 — the blocked 1191 T4 128k rho cells re-run after T1+T2 land (the 128k dense.bin/completed arm is already bankable by then; the record lane decides whether the 128k axis re-enters the same bench record or a follow-up one).
- [ ] T5 — 1191 T4/T5 implications (the record lane's): 32k dense + rho_0.15 pair adjudicates the pre-registered bars NOW (|ΔΔNLL| ≤ 0.245, top-1 gap ≤ 0.131); the 128k axis is blocked by this issue; the envelope (~54K tokens at rho=0.15/block-64) is a documented finding either way — it is the honest operating envelope of the arm as shipped.

## Credit + provenance

Root cause: the midnight monitor lane (`zed-5592b062@amo`), 01:22-01:30 local 11-10, from run.log + pinned source. Independent verification (panic line, kernel clamp math, reader walk, smem assert) + the partial-bin and timeline corrections: rig-ops lane (`zed-3ecf601a@amo`). Monitored run: `job 01a1c0f5-00d6` (exclusive, amo, key i1191t4b), script `/Users/katopz/riir-bench/i1191_metal_run.sh`, artifacts `/Users/katopz/riir-bench/i1191_metal/`. No process/tree/file was touched by either lane.
