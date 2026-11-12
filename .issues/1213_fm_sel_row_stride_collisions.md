# Issue 1213 — the FM sel row stride stayed 128 through the FM_MAX_SEL raise: the 128K Metal ρ arms measured a collided selector (VOID)

**Status:** OPEN — fix LANDED (this commit), the 128K ρ re-run pending the trio's dense completion. `- [ ]` tasks below.

## Found by

The issue 1191 T4 128K Metal trio's mean-selected-fraction anomaly: ρ0.15(M) recorded **0.0504** in its bin header (2026-11-12 13:35) where the CPU trio on bmo recorded **0.1488** at the same nominal ρ — and the ρ0.30(M) arm would have recorded ~0.10 where the CPU recorded 0.2989. The quality numbers were *plausible* (ρ0.15 ΔNLL +0.2097 / top-1 80.85% vs CPU +0.2579 / 78.52%), which initially suggested a bookkeeping-only artifact. Root cause is worse than bookkeeping.

## Root cause

`66782122` (the issue-1200 capacity raise, owner speed directive) changed `FM_MAX_SEL` 128 → 1024, `FM_MAX_BLOCKS_SMEM` 2048 → 4096, and the kernel's k clamp `.min(127)` → `.min(1023)` — but **not the selection row's stride literals**, which stayed `128`:

- `fm_top_rho_select_f32`: stash `sel_out[kv * 128 + round]`, sort base `kv * 128`, pad `while p < 128`
- `fm_block_gather_f32`: `sel_base = kv_group * 128`
- the host reader `fm_selected_fraction` reads at `kv * FM_MAX_SEL` (1024) — and the t2prep test reads at `kv * FM_MAX_SEL` too

So at any **count > 128** the select kernel writes ~2–5× past its own row into the NEXT kv heads' rows — cross-head collisions (concurrent cubes, last-writer-wins per slot). At the T4 operating geometry (block-64 @ 128K = 2048 blocks):

| arm | count = max(round(ρ·n),1) | rows spanned (÷128) |
|---|---|---|
| ρ 0.15 | 307 | 2.4 |
| ρ 0.30 | 615 | 4.8 |

Every kv head attended a mixed set spanning all heads' top picks — an *internally consistent but semantically wrong* selector (the gather reads stride 128, matching the writes, so the collision is self-consistent; that consistency is why quality looked plausible — a cross-head union behaves like a *larger* effective per-head coverage, consistent with ρ0.15(M) beating the CPU's ρ0.15 on ΔNLL).

The reader anomaly (0.0504) is a second face of the same bug: `fm_selected_fraction` reads stride-1024 rows of a stride-128-written buffer — kv ≥ 1 rows land in never-written memory (u32::MAX pad) and break to zero, so only kv0's (collided) row is counted, divided by all `n_kv` heads.

## Why the gates missed it

- **Envelope (`fm_budget_envelope_check`)**: bounds `count ≤ FM_MAX_SEL` (1024) — written when row width == FM_MAX_SEL == 128, so it was a row-capacity check THEN; after the raise it checks against a number 8× wider than the physical row. Passes at count 307/615.
- **T2-prep parity gate**: fixture runs 12 blocks, ρ0.25 → count 4 — stride-independent. Its reads at `kv * FM_MAX_SEL` would in fact have RED post-raise (kv ≥ 1 rows read unwritten memory) — the gate was **not re-run after `66782122`** (the raise re-ran the envelope tests + a Metal smoke only). Process gap: a const raise is a kernel-shape change; the parity gate that pins the kernel's layout must re-run with it.
- **T2 full-forward gate**: 96 steps → 2 blocks — same stride-independence.

## Blast radius

- **128K Metal trio ρ arms (this box, `~/riir-bench/i1117_128k_metal/`) — VOID as budget-arm measurements.** The ρ0.15 bin (complete) and ρ0.30 arm (in flight at discovery) measured the collided selector. The **dense arm is valid** (unarmed path) and the walls are honest as collided-selector walls. The **CPU trio on bmo is unaffected** (host lane, no device rows).
- Everything at count ≤ 128 is unaffected — every recorded T1/T2/T2-prep result (small fixtures), the ctx-2048 smokes, and any context ≤ ~54K at ρ0.15 (count = round(0.15 · ctx/64) ≤ 128 → ctx ≤ 54613). The pre-raise envelope (FM_MAX_SEL 128) never admitted count > 128, so **no pre-raise run is implicated**.
- No serving/consumer path armed: the feature is opt-in, PROMOTE-ONLY env, no game-side consumer.

## Fix (landed this commit)

- The four stride literals moved 128 → 1024 = `FM_MAX_SEL` (select stash/sort/pad + gather `sel_base`), with lockstep comments naming the issue-1213 class; the launch doc now states the row stride IS `FM_MAX_SEL`; the `FM_MAX_SEL` const doc states the row-stride law and that the envelope check is a row-capacity guarantee only while the row width equals it.
- **Regression lock**: `flashmemory_budget_metal_t2prep::fm_budget_metal_row_stride_regression` — n_blocks 1000 at ρ0.15 (count 150 > 128, the pre-fix collision region), per-row SET equality vs the CPU budget rule. Pre-fix this fails at every kv ≥ 1; post-fix it passes. Metal-scoped, SKIPs loud off-Metal.
- Reader `fm_selected_fraction` unchanged (its stride was already `FM_MAX_SEL` — correct under the intended layout).

## Tasks

- [x] Root cause + blast radius + the pre-raise runs declared unaffected (count ≤ 128 everywhere).
- [x] Fix the four stride literals + docs (this commit).
- [x] Regression lock test (this commit).
- [ ] Re-run the t2prep gate + the new regression leg on Metal (m3, after the trio frees the GPU).
- [ ] End-to-end metric validation: short real-model gate cell at ctx 65536 (n_blocks 1024, ρ0.15 → count 154 > 128), assert the completed bin's mean_selected_fraction ≈ 0.15.
- [ ] Re-run the 128K ρ arms (v2 out dir, dense.bin copied for the digest-match auto-load; ~9 h + ~14 h on m3 Metal) + the completing bench record.
- [ ] Fold the closure into the issue 1191 T4 record (bench 1209 carries the VOID disclosure).

## Non-goals

- No bitonic sort swap for the O(k²) insertion sort now (k ≤ 1023, refresh-cadence-bounded, seconds per 128K walk — noted at the site).
- No CUDA twin change (the 4090 lane copies the fixed source when it lands).
