# Bench 964 — Issue 1177 Item 1: the cross-backend bit-parity G1 cell — CUDA (cudarc twin) vs Metal (wgpu league default) on the SAME Q4_K_M GGUF: 48-token greedy walk EXACT, FNV pins identical, band unexercised

**Status:** RECORD — Item 1's G1 **PASSES at the primary (exact) pin**; the pre-registered near-tie band was never exercised (no divergence to adjudicate). Issue 1177 closes: both residues landed.
**Date:** 2026-11-09 (Metal leg 12:2x +07; CUDA leg 17:0x +07 — job `01a1bed7-47f8-7df0-b54d-cac7fd8e199a`)
**Session:** zed-15be441a@amo, epoch 1794201411
**Issue:** [`.issues/1177_dense_lane_residues_cross_backend_parity_and_decode_gemv.md`](../.issues/1177_dense_lane_residues_cross_backend_parity_and_decode_gemv.md) (Item 1 — the band pre-registration lives there, written BEFORE any run)
**Walk:** the prose family (the harness default prompt, 45 chars → 10 tokens), 48 greedy tokens, ctx 2048 on BOTH sides (the 4090's 24 GB cannot hold f32 KV at the Metal-default ctx 24576 beside the weights; ctx is allocation-only), harness loop verbatim on both sides (the s==0 re-feed at pos = ids.len()).

## The model pin (A1/A8)

One GGUF on both backends, verified by hash on each box:

- **amo** (`~/uzu27b/models/qwen3.8-27b-Q4_K_M.gguf`, 16,810,714,336 B): sha256 `e00082f779fa385cee8c68a3ec8833a75778cc87272240b942f74e0b8243e520`
- **4090** (`F:/models/qwen3.8-27b-Q4_K_M.amo-copy.gguf`, chunked-shipped copy, 16,810,714,336 B): sha256 `e00082f779fa385cee8c68a3ec8833a75778cc87272240b942f74e0b8243e520` — **byte-exact** (Get-FileHash on arrival, printed in the job log beside the walk).

The 4090's pre-existing `qwen38-27b-dbirks-Q4_K_M.gguf` was REJECTED for this cell: 16,810,714,848 B (a 512-byte delta — different bytes = different file, never a parity input).

## The two legs

| leg | box / posture | load | walk ids |
|---|---|---|---|
| **Metal** | amo (M5 Ultra, quiet, preflight PASSED — `power=AC Power load=3.80 powermode=0(auto)`), `qwen38_wgpu_bench` release, `--features qwen38_wgpu,metal_tensor_gemm`, bare default = the promoted 1176 league posture (banner confirmed: dense matmul2d league DEFAULT + tensor-flash + half-B + Q6_K packed tail, all relaxed; `relaxed_precision=true`), 20.1 s load | idle GPU | `[walk] ids: [279, 2993, 421, 18378, 10849, 694, 12238, 1132, 264, 38936, 35101, 13, 561, 3594, 82738, 314, 11679, 11, 31694, 11, 321, 279, 36334, 61306, 314, 888, 49371, 1016, 13834, 494, 279, 2981, 71072, 13, 1049, 557, 279, 2993, 314, 6353, 421, 1801, 1208, 3166, 440, 7193, 11, 430]` |
| **CUDA** | 4090 (idle — GPU 0%, the rig job exclusive window), `qwen38_cudarc_walk` release (`ff5355a`), the cudarc twin at its DEFAULT env posture (all-fused arms — the twin's own canonical posture; the league envs are Metal-side), 42.4 s load | idle GPU | `[walk] ids: [279, 2993, 421, 18378, 10849, 694, 12238, 1132, 264, 38936, 35101, 13, 561, 3594, 82738, 314, 11679, 11, 31694, 11, 321, 279, 36334, 61306, 314, 888, 49371, 1016, 13834, 494, 279, 2981, 71072, 13, 1049, 557, 279, 2993, 314, 6353, 421, 1801, 1208, 3166, 440, 7193, 11, 430]` |

## The verdict

- **Sequences identical: 48/48 tokens, zero first divergence.**
- **FNV-1a (LE-u32 over the id sequence): `0x814125045f90b9` on BOTH sides.**
- Continuation text identical on both sides: " the kind that promised rain but delivered only a damp chill. The air smelled of salt, diesel, and the faint sweetness of overripe peaches from the market stalls. It was the kind of morning that made people move with purpose, as"
- **The pre-registered 1e-2 near-tie band: unexercised** (recorded as pre-registered-but-not-needed — the machinery stands for any future re-cell, e.g. a different checkpoint).

## Honest scope (what this pins, what it does not)

The claim is the **greedy-argmax decision floor on THIS model at THIS walk**: both backends read the same Q4_K_M weights (and the same Q6_K lm_head tail), the Metal side runs its lossy league arms (retention-certified 272/272 at bench 946), and the cross-backend kernel-order/fma-contraction delta stays below the argmax decision floor on these 48 steps. It is NOT a logits-distribution claim (no KL/CE read ran — the capture-logits band instrument exists for the flip path and was not needed), and NOT a claim about other checkpoints (a different GGUF re-runs the cell — the band block in the issue is the standing recipe).

Route note (disclosed, not a caveat): the Metal side prefills via the chunked ingest route; the CUDA side prefills token-by-token (the cudarc forward has no chunked ingest). The route ≡ multi-token equivalence on Metal is Issue 1150's own route G1 (bench 938), so the backend comparison is backend-vs-backend; the prefill routes differ by construction and the cell pins the WALK (post-prefill decode), which is the parity artifact 1150 named.

## Provenance / transfer

- Band pre-registration: issue line, commit `c9603b9` (before any run).
- CUDA bin: `qwen38_cudarc_walk` `ff5355a` (no sha/fnv in-bin — the wrapper prints Get-FileHash; FNV computed offline from the printed ids on both sides, identical code path).
- Transfer: chunked scp (16 × 1 GiB, per-chunk skip-resume over Tailscale DERP ~1 MB/s; the naive full-file scp restarted from zero on every drop — three drops observed), concat `copy /b` on arrival, size + sha256 verified before the walk. Chunks deleted post-verify; the amo-copy file retained on the 4090 as the pinned parity model. LAN-direct amo↔4090 was ARP-dead this posture (both sides reach the router, not each other) — noted for future bulk moves.
- Timed wake `01a1bef9` (18:30 +07 transfer-health check) mooted by early completion — fires harmlessly.
