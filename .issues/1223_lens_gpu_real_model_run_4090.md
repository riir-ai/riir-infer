# Issue 1223: the lens full-basis REAL-model run — 4090 cell (i1222 follow-up)

**Status:** open — **Claimed:** shikuwa-4090-i1223rec, epoch 1794978699, expires 2026-11-20T11:11:39+00:00 (smoke-v3 record + deciding-run launch; prior lease zed-ba521963@amo expired 2026-11-18T00:16Z with the smoke unrecorded — taken per the expired-lease law).

## What this is

The GPU batched lens lane LANDED (i1222, riir-infer `e11ba253` + `b6dc97e0`):
13 CubeCL kernels + driver + gates (10 unit cells exact, composite parity
structural-gated across 5 stack shapes × 3 seeds). The remaining cell is the
REAL run: `lens_gpu_fit_t3` on `Qwen3.5-0.8B-Base-Q8_0` over the ag_news
corpus at the T3 protocol shape — the full-basis fit that motivated the lane.

## The run recipe

```sh
# sync first (the lane landed 2026-11-16, f20ac53b tip)
schtasks /run /tn gitsync   # then poll E:\git\sync-repos.log for SYNC DONE
cd E:\git\riir-infer
cargo run --release -p riir-infer-gpu --features lens_gpu --example lens_gpu_fit_t3 -- \
  --gguf ../riir-train/data/Qwen3.5-0.8B-Base-Q8_0.gguf \
  --corpus-dir ../riir-reflex/.raw/datasets/ag_news \
  --docs 1000 --t 128 --dim-batch 128 \
  --box-state "4090 <PROVENANCE line>"
```

(`-p riir-infer-gpu` is REQUIRED — `lens_gpu` lives on the gpu crate, not the
root package; the bare `--features` spelling refused with `error: no feature
named lens_gpu` + `help: package with the missing feature: riir-infer-gpu`.
Measured on the 4090 2026-11-16.)

**Run (in flight 2026-11-16):** first execution FAILED — `lens_gpu_fit_t3.rs` accumulation indexing OOB at the doc-1 K merge (`len 1048576, index 1048576` = d², point 1 row 0): `target[pi]`/`gpu_sums[pi]` are per-point [d × d] buffers, but both merge sites carried a `pi *` term inside the ROW index (`(pi*d + row)*d`) — correct only for point 0, out of bounds for every point ≥ 1. The J sweep carried the identical defect (would have fired after K). **Found by the first real-model execution** — the fixture/unit gates exercise the kernels, never this example's accumulation loops, and no prior run had `points.len() ≥ 2`. FIXED `7862a8c` (K merge + J sweep; comment pins the layout law).

**Smoke v2 (post `7862a8c`) found the SECOND real-geometry class** (`01a1e1f2`, exit 1): the plan-611 `f32_exact` params guard fired — `shape 58720256 exceeds the f32-exact bound 2^24` — the monolithic `n = batch·t_len·d` (128×448×1024) param exceeds f32-exact at real geometry (the fixture stacks never do). FIXED `498f3d5` + `1ceb459`: the three monolithic-count kernels now take FACTOR params — `LensScaleTableCL`/`LensHalfRule2CL` `[rows, row_len]` (the modulus row length always divides n; `tid / row_len < rows` ⇔ `tid < n` exactly), `LensAddInplaceCL` gains the `row_len` arg (2 call sites). The guard was never weakened — the params stay f32-exact, the shape is factored. Fixture gates re-run green: units 10/10, suffix parity PASS, clippy clean (rulewright mechanicals applied compile-gated). Smoke v3 queued `01a1e22b-e4aa` (4090 Exclusive).

**Smoke v3 RESULT (2026-11-18 record):** job `01a1e22b-e4aa` COMPLETED 2026-11-16 12:02–12:06 local — **9302.7 s/doc** (sweep 9294.4 + capture 8.4), G3 pass=true (max_rel 0.0 both axes) but **STRUCTURAL ONLY** — manifest `n_even=1, n_odd=0`: at docs=1 the odd half is empty, so parity was not exercised; the real tolerance calibration lands with the deciding run. Full record: `.benchmarks/1240_i1223_lens_gpu_smoke_v3_record.md`. **Deciding run SIZED from the measured wall: docs=10 → ~25.9 h (inside the ≤30 h target), halves 5/5 non-empty**, posture unchanged (t=128, dim_batch=128), launched via `agent-mail rig run 4090 --exclusive` — job `01a1ed16-36e3-7771-a152-07b0f5ea680e` (notify shikuwa@4090, timeout 40 h, wrapper `E:/tmp/i1223/deciding_run.cmd`), queued behind the in-flight plan-690 GPU cell.

**The run plan is re-priced (the recipe's ~20-30 h estimate was wrong >30×):** the failed run's timeline — start 07:26:01, panic 08:13:20, ZERO per-doc lines — means doc 1 never completed: ~47 min = capture + the FULL K sweep (1024 prefix JVPs × 48 points, 24 workers, k_threads defaults to available_parallelism). ~45+ min/doc on the 4090's 24 cores ⇒ 1000 docs ≈ 750+ h, infeasible as launched. Re-plan: **1-doc smoke first** (post-fix: validates the K merge through all 48 points, the J sweep through all 8 full-basis blocks, and the finalize path; measures the honest per-doc wall), then the deciding run is re-sized to what the acceptance needs at a feasible doc count (target ≤ ~30 h) — the G3 half-corpus parity gate needs INDEPENDENT halves, not 1000 docs; the count is set from the smoke's measured cost and recorded here.

**Artifacts of the failed attempt kept:** job `01a1e1bd` log (`C:\Users\katop\.agent-mail\jobs\logs\01a1e1bd-16b4-7280-b071-96e6437ab1ec.log`), rig `rig log 4090 01a1e1bd`; boot facts stand (CubeCL `wgpu<spirv>`, model load 5.6 s, corpus blake3 `22cb3c582d73cbc6`).

- The GGUF: `riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf` is the league
  model; the lens lane needs the 0.8B dense checkpoint — restore if missing
  via `./scripts/data_backup.sh pull data/Qwen3.5-0.8B-Base-Q8_0.gguf`
  (riir-train root; the LOCAL-KEEP pin class).
- The corpus: riir-reflex `.raw/datasets/ag_news` — if the 4090 checkout
  lacks it, `fetch_datasets.sh` (riir-reflex scripts) or copy the shard set.
- Estimated ~20-30 h for 1k docs at fp32 via wgpu-spirv (Vulkan on Windows)
  — est ~29 s/doc compute + capture ~4-5 s/doc + K sweep on CPU threads
  concurrent. Budget the box EXCLUSIVE (`--exclusive`).
- VRAM: dim_batch 128 ⇒ ~4-6 GB scratch + ~6 GB capture at T=128 — fits the
  24 GB card; raise `--dim-batch` toward 512 if headroom shows (fewer block
  passes).

## Acceptance

1. The fit completes with the G3 half-corpus gate PASS (the parity gate on
   the REAL model: the fixture numbers do NOT transfer — the first real run
   CALIBRATES the production read-level tolerance; record it in i1222).
2. Sampled CPU-vs-GPU read check on one doc (the `check-direct` analog —
   the GPU fit's reads vs the CPU lane's reads on shared docs, Pearson +
   rel-RMS recorded).
3. The artifacts + manifest land in `.lens_gpu_fit_t3/`; the record lands in
   i1222 (the tolerance calibration row) + a bench file.
4. If the reads beat the CPU lane's G2 steering-prediction bar at the same
   doc count, the full-basis restoration is validated end-to-end.

## Refs

- i1222 (the lane), katgpt-rs 999 T3, riir-train data_backup lane.
