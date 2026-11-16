# Issue 1223: the lens full-basis REAL-model run — 4090 cell (i1222 follow-up)

**Status:** open — **Claimed:** zed-ba521963@amo, epoch 1794788174, expires 2026-11-18T00:16:00+00:00 (4090 real-model run launch; amo session).

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

**Run (in flight 2026-11-16):** 4090 rig job `01a1e1bd-16b4-7280-b071-96e6437ab1ec`
(EXCLUSIVE, notify `zed-ba521963@amo`; wrapper `E:\git\lens_fit_t3_run.ps1` v2 —
preflight → build → run → artifact listing; log `E:\git\lens_fit_t3_run.log`).
Boot confirmed: CubeCL `wgpu<spirv>`, model load 5.6 s, corpus 1000 docs
128k tokens blake3 `22cb3c582d73cbc6`, J sweep 8×B128 blocks on GPU.

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
