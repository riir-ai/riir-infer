# Bench 1256 — i1245 tail-logits root cause + fix: the plane-GEMV slice-length trap

**Session:** zed-8f8fb130@amo, epoch 1795461850+ · **Issue:** riir-infer 1245 (kernel-fix scope; filing + repro by zed-bb86e4df) · **Date:** 2026-11-24

## Root cause (bisected, reproduced, fixed)

`GemvCubeCL::launch` (the plane + tiled f32 GEMV kernels) derived their bounds from
`input.len()`/`output.len()`. A kernel-visible slice length is the **binding's backing
slice size** — the launcher registers it in the metadata; the
`BufferArg::from_raw_parts(handle, len)` length does **not** constrain it (the Issue-639
fact, one layer down; already documented on `rmsnorm_f32`).

`ingest_chunk_gpu`'s tail stage binds `tmp_b` (a `[chunk × e]` scratch) via an offset row
view and `logits_b` (a `[vocab]` buffer) via offset row windows. With `.len()` bounds,
**every chunk-tail GEMV launch read the whole scratch** (512 normed rows instead of the
last row) and **wrote past its row window** (the last chunk's view extends 229k elements
past the logits buffer). Results: mostly-correct-looking head logits, ~0.1% garbage rows
(1e37–1e38 / ±inf — adjacent-slab bytes read OOB), **nondeterministic garbage indices per
run** (pool-layout dependent), and silent OOB writes into neighbouring pool slices.

Why the lane's G1 cells never caught it: the serving default seeds decode through short
prompts (ragged tail → decode kernels only), and the league prefill cells measure
throughput, not chunk-boundary logits. Decode is a disjoint code path with exact-size
handles — always sane.

## The bisect (issue 1245's four postures were all the same posture)

A debug example (`examples/i1245_tail_bisect.rs`) + two unit tests
(`tests/i1245_tail_gemv_units.rs`, `tests/i1245_tail_stage_units.rs`) isolated it:

| read | result (pre-fix) |
|---|---|
| `x_b` last residual row | sane |
| `tmp_b[0..e]` (tail-norm output = GEMV input) | **sane** (0 non-finite) |
| `logits_b` (chunk tail) | garbage, garbage indices move per run |
| decode continuation from the chunk states | **sane** (layer loop correct) |
| plane GEMV, fresh exact-size handles | **passes** (the kernel is correct on exact bindings) |
| plane GEMV, input ≥ 32 KiB (pool-batched slice) | **fails** — `input.len()` = the slice size |
| stage miniature (batched norm → offset-view norm → offset-view GEMV) | reproduces; bitwise deterministic per layout |

The filer's "garbage on every posture/chunk size" was one posture: the league env
defaults (tensor-flash attention etc.) are demote-only and were armed in all repro runs,
and the decode control never enters the broken stage — the defect was neither.

## The fix (`gemv_cubecl.rs`)

Both f32 GEMV kernels (`gemv_plane_f32`, `gemv_tile_f32`) now take `n`/`m` from a
2-f32 params buffer (the Issue-697/698 pattern `gemv_batched_plane_f32` already uses);
`launch`, `launch_plane`, `launch_plane_wg`, `launch_tiled` pass it. One extra 8-byte
uniform per launch; no kernel-body math change (the dot loops are untouched, so the
decode path is bit-identical — the fp32 GEMV arm computes the same rows).

Known residue (NOT fixed here — no oversized-view caller today): the other
len()-deriving GEMV kernels (`gemv_q4k_cubecl` ×4 sites, `gemv_f16_cubecl`,
`gemv_geglu_*`, `gemv_cubecl.rs:126` pair kernels) carry the same trap for any future
caller that binds oversized views. The q6k kernel already takes params. Recorded as a
follow-up note in the issue.

## Validation

- `tests/i1245_tail_stage_units.rs`: 2/2 — the stage miniature is exact vs a CPU
  reference (0/32768 mismatches both runs) and bitwise deterministic run-to-run.
- `tests/i1245_tail_gemv_units.rs`: 2/2 — exact at m=32768 and m=18944.
- Real model (Qwen3.5-0.8B Q8_0, the i1243 corpus), all four bisect arms: chunk-tail
  logits **0 non-finite, range [-11.14, 20.26]**, absmax@25781 (the corpus's next-token
  region — plausible), **identical across arms** (deterministic), decode continuation
  unchanged (argmax 19746, head logits byte-identical to the pre-fix decode run).
- Lib suite (default qwen38_wgpu posture): **410 passed, 0 failed**; the G1 gates touched
  by the shared kernel (`qwen38_gdn_chunked_route_g1` 3/3, `qwen38_fused_norm_gating_g1`
  3/3) green.
- G2/G3 (league pp rows unchanged): not re-measured this session — the kernel bodies are
  byte-identical (params replace two register reads from `.len()`, which the MSL already
  materialized as constant loads); the pp cell re-read lands with the i1243 chunk-posture
  unblock, which is the next consumer of record.

## Provenance

amo (M5 Ultra), AC, load 4–8 (the bmo census ran throughout on the other box; no
GPU-exclusive claim needed — this cell is correctness, not timing).
