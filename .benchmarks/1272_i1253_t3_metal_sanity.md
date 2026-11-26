# Bench 1272 — Issue 1253 T3 cheap half: the first forward on Bonsai-2.5, Metal production posture

**Status:** COMPLETE — G1 PASS all cells, receipts reproduced (2026-11-26, session zed-i1253t3@amo).
**Issue:** riir-infer `.issues/1253` T3 (the cheap m3/amo Metal half; the 4090 fork-logits-parity G1 half stays with the GPU lane).
**Box:** amo (M5 Ultra 96 GB, Metal/wgpu, AC) — ⚠ CPU-CONTENDED: load 38.65 (the four i1244 hisa_census shards, ~4×720% CPU, GPU-free). **Every timing below is CONTEXT ONLY, NON-CLAIM** — the load class is outside any bench posture; only the correctness numbers are claims.

## What ran

`crates/riir-infer-gpu/src/bin/i1253_t3_metal_sanity.rs` (NEW, `required-features = ["production"]` + `riir-infer-core/bonsai2_hadamard` at the invocation): loads the checkpoint, builds `TernaryDeltanetGpuForward` on CubeCL (wgpu<msl>) with **no knob calls** — the untouched promoted serving posture — clamps `block_size` to 1024 for the small-context KV reservation, prefills the Issue-594-class probe, and greedy-walks.

Serving-path reach (the boot lines): `wgpu<msl>` runtime, the Issue-657 T5 zero-copy raw-Metal GEMM initialized (the ktile64+zerocopy serving DEFAULT), `[norm_rot_multi] REACH` — the fused norm+rotate staging dispatched. The `[Issue 727] metal-rs GEMM cache skipped` line is the expected informational (that A/B path arms only via its setter before `new()`; the production default is the zero-copy lane).

## The cells

### 2.5 (`Ternary-Bonsai-2.5-27B-PQ2_0.gguf`, 7,253,354,848 B — the PLAIN league file, not the MTP twin)

| probe | result |
|---|---|
| `"The capital of France is"` → rank of `" Paris"` (id 11751) | **rank 1**, logit 16.0054 vs next 12.2535 (**margin 3.75**), bar ≤ 5 — PASS |
| prefill logits FNV | `0xffde63bb7b24e27a` — byte-identical across 3 invocations (2 binaries) |
| first decode-lane logits FNV | `0x7e505936fc45f491` — byte-identical across runs |
| greedy walk (cap 32) | `[11751, 13, 248046]` → `" Paris.<|im_end|>"` — answers, ends the sentence, **clean chat-EOS stop at step 2** |
| finiteness | all logits finite, prefill + every decode step (loud guard, exit 1 on violation) |
| story walk (`"Once upon a time there was a"`, cap 24) | 24 steps, coherent narrative — `" small town in the mountains. In that town, a young woman named Elara was a blacksmith's daughter."`; decode FNV `0x47976c4badf73a3f` |

### Predecessor corroboration (`Ternary-Bonsai-2-27B-PQ2_0.gguf`, same probe)

| probe | result |
|---|---|
| rank of `" Paris"` | **rank 1**, logit 13.9980 vs next 11.0587 (margin 2.94) — PASS |
| prefill FNV / decode FNV | `0x81309dacf97be906` / `0x0175ff5fd3ac65b7` |
| greedy walk | `" Paris.\n#The capital of Germany is Berlin.\n#The capital of Italy is Rome. …"` — coherent pattern continuation (32 steps, no EOS) |

**Read:** the retarget did not degrade recall behavior — the 2.5 margin is WIDER than the predecessor's (3.75 vs 2.94 logits), and 2.5 additionally emits the trained chat-EOS after a complete answer where the predecessor enumerates. Both checkpoints rank `" Paris"` #1 by a decisive margin.

## Walls (context only, NON-CLAIM — contended box)

load+init ~13-14 s · prefill 6 tokens 0.3-0.4 s · decode ~25 steps in 0.2-0.8 s (~25-30 steps/s wall under 28-core CPU load). No perf claim is made or derived from these.

## What this closes and what stays open

- **Closed by this note:** T3's cheap half — the swap is now verified at the FORWARD level on the Metal serving posture, not just the header parse (T2's load cell). G1 first-token sanity + decode walk + finiteness + determinism receipts, all green.
- **Still open (T3 remainder):** the 4090-side fork-logits-parity G1 (`bonsai2_rotation_load` doc's real-file arm) — CUDA lane, rides the GPU-leg schedule.
- **Unblocked by T3:** T5 (the MTP-head specdec intake) — its dependency pair (T2 loader tolerance + T3 model verified) is now satisfied for the Metal side; T5's own cells (accept-rate-weighted net tok/s vs plain decode at the card presets) remain a separate measured lane.
