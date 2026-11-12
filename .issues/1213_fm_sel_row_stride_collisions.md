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
- [x] Re-run the t2prep gate + the new regression leg on Metal (m3, post-trio 2026-11-13 04:33: **2 passed** — the regression leg + the three-leg parity, 0.88 s).
- [x] End-to-end metric validation (PASS 2026-11-13 ~05:1x): ctx 16384 / 9300 steps / ρ0.5 on the fixed build reads **sel = 0.4926 ≈ ρ** through the armed forward + the fixed reader (pre-fix geometry read ~ρ/n_kv ≈ 0.125). The ρ0.9 companion arm was refused at the ctor by the 1200 loud envelope check (as designed — worst-case blocks × 0.9 needs 1843 slots > 1024); the first ctx-65536/256-step attempt was a geometry error (the walk reaches n_blocks 5; its 0.3362 sel is correct tiny-ctx sawtooth). Artifacts: `/tmp/i1213_validate2/{dense,rho_0.50}.bin` + `/tmp/i1213_validate2.log`.
- [ ] Re-run the 128K ρ arms (v2 out dir `~/riir-bench/i1117_128k_metal_v2/`, dense.bin copied for the digest-match auto-load; ~9 h + ~15 h on m3 Metal, launches after the validation pair frees the GPU) + the completing bench record.
- [ ] Fold the closure into the issue 1191 T4 record (bench 1209 carries the VOID disclosure — `.benchmarks/1209_i1117_128k_trio_cpu_valid_metal_rho_void.md`).

## Non-goals

- No bitonic sort swap for the O(k²) insertion sort now (k ≤ 1023, refresh-cadence-bounded, seconds per 128K walk — noted at the site).
- No CUDA twin change (the 4090 lane copies the fixed source when it lands).
