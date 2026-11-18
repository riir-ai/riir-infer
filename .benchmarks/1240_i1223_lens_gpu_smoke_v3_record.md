# Bench 1240: i1223 lens_gpu_fit_t3 smoke-v3 — the first REAL-model full-basis run (record)

**Status:** RECORD (measurement landed by the run itself; this file is the missing record — the run completed 2026-11-16 12:02–12:06 local on the 4090, results were never recorded before the filing lease lapsed)

## What ran

`lens_gpu_fit_t3` smoke v3 (post `7862a8c` accumulation-index fix + `498f3d5`/`1ceb459` f32-exact factoring) — the FIRST end-to-end REAL-model execution of the GPU batched lens lane (i1222), job `01a1e22b-e4aa` (4090 Exclusive), log `C:\Users\katop\.agent-mail\jobs\logs\01a1e22b-e4aa-7183-9b82-c4f9d31a9fe2.log`.

```
cargo run --release -p riir-infer-gpu --features lens_gpu --example lens_gpu_fit_t3 -- \
  --gguf ../riir-train/data/Qwen3.5-0.8B-Base-Q8_0.gguf \
  --corpus-dir ../riir-reflex/.raw/datasets/ag_news \
  --docs 1 --t 128 --dim-batch 128 --box-state "4090 smoke-v3-post-1ceb459"
```

## Measured

| datum | value |
|---|---|
| per-doc wall | **9302.7 s/doc** (sweep 9294.4 s + capture 8.4 s) — ~2.58 h/doc |
| model load | 5.4 s (24 layers, 18 GDN / 6 attn, d=1024, vocab=248320) |
| stack | points=48, t=128, seeds [16,127), dim_batch=128 |
| K sweep | 1024 prefix passes (basis dirs) × 48 points, CPU workers |
| J sweep | 8 direction blocks of ≤128 on the GPU (49152 suffix passes total) |
| artifacts | `.lens_gpu_fit_t3/` — 1152 MiB f32 rows (fit_all/even/odd × k/j) + manifest.json (format `lens_gpu_fit_t3_v1`) |
| G3 parity | max_rel_k = 0.0000, max_rel_j = 0.0000, pass = true — **STRUCTURAL ONLY, see caveat** |
| corpus | ag_news, blake3(rows) `22cb3c582d73cbc6…`, 200 rows available |
| box state | 4090 (RTX 4090 24 GB, wgpu<spirv> → Vulkan on Windows), Exclusive rig job |

## The G3 caveat (the honest read)

manifest.json: `n_even=1, n_odd=0` — with docs=1 the even/odd split puts the single doc in the
even half and the odd half is EMPTY, so the G3 max_rel 0.0 is a structural smoke pass (the halves
exist, the finalize path runs, the gate machinery executes), NOT a meaningful parity datum.
**Parity needs ≥2 docs so both halves are non-empty.** The first real-run tolerance calibration
the i1223 acceptance asks for therefore lands with the DECIDING run, not this smoke.

## What the smoke DID prove

- The K merge + J sweep accumulation fixes hold at real geometry through ALL 48 points and all
  8 full-basis J blocks (the two defect classes found by smoke v1/v2 were doc-1-never-completed
  classes; v3 completed a full doc).
- The f32-exact params guard holds at real shape (58,720,256-element monolith now factored).
- The measured per-doc wall re-prices the run plan: the original recipe's ~20-30 h/1000-doc
  estimate was wrong >100×.

## Deciding-run sizing (derived from the measured wall)

| docs | est wall (9302.7 s/doc + ~10 s load) | halves |
|---|---|---|
| 1 (done) | 2.58 h | vacuous |
| 10 | **~25.9 h** — inside the ≤30 h target | 5/5 non-empty |
| 12 | ~31.0 h — over target | 6/6 |

**docs=10, t=128, dim_batch=128** (the measured posture; dim_batch raise toward 512 is a
separate unmeasured lever — not taken in the deciding run). Launched via
`agent-mail rig run 4090 --exclusive` (queues behind the in-flight plan-690 GPU cell).
