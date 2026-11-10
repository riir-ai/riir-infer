# Bench 968 — plan 492 T0-residue (b): the main-path golden piece — cross-engine e2e token identity REFUTED (fold-order chaos), mechanism sealed

**Status:** DONE — the residue's open half is CLOSED with a real finding. The
e2e golden arm (Rust `qwen38_dspine_t2 --e2e` + Python
`dspine_golden_replay.py --e2e`, landed `6ca37bcf`..`d7ad0d7f`) drove BOTH
pipelines end-to-end on the same dumped inputs: the vendor's own
`DSpineDraftModel` (5 layers + `DSpineCodeBridge.__call__` + 
`RefinementTail.greedy`) vs our `draft_block` main path. **The chain tokens
do NOT agree cross-engine — 7/7 differ — and the mechanism is sealed: a
per-op fp32 fold-order difference (one bf16 ULP at the layer-0 input norm,
formula/eps/inputs/weights all pinned equal), amplified through the 5-layer
chain into complete topk disagreement.** Not a transcription bug — the
transcription is proven exact piecewise AND at the norm-op level; the
cross-engine bit-identity bar is unsatisfiable for any two implementations
that differ in fold order.

## Cells

- **Host:** amo (M5 Ultra, quiet; the run is CPU/MLX). 2026-11-10.
- **Posture:** the t2 shape-walk synthetic posture (seed 777, block 8, γ=7,
  ctx 32 rows, anchor 151644, synthetic bf16 lm_head + blake3 embed stand-in)
  — IDENTICAL inputs both sides, DUMPED whole by the Rust arm (lm_head 1.25 GB
  included: a Python LCG replay of 312M sequential draws would be an avoidable
  bit-exactness risk — the vendor arm is a PURE replay, zero reconstruction).
- **Rust side:** `draft_block` (the real main path, one call, vendor-exact
  rounded posture), blake3 pin `53d18516…` verified at load; chain
  `[35446, …]` conf 1.000 — **byte-identical to Bench 967's shape-walk chain**
  (same posture, deterministic forward ✓).
- **Vendor side:** `load_draft` + shadow config (the vendor loader refuses the
  checkpoint's sliding-window `layer_types`/`sliding_window` — their repo
  predates the SW lane; the override is a NO-OP at this posture, 40 rows ≪
  SW 8192, disclosed), `append_context` with the dumped captures,
  `model(None, block_ids, caches)` → `RefinementTail.greedy`.

## The measurement

| check | result |
|---|---|
| e2e candidates | 112/112 slots differ (complete disagreement) |
| e2e tokens | 7/7 differ — vendor `[219399, 157392, 27106, …]` vs rust `[35446, 11380, 107231, …]` |
| proposal rows max abs | 33.19 (final normed rows; norm_w-scaled chaos) |
| **layer-0 x_in max abs** | **7.812e-3 = exactly ONE bf16 ULP at [1,2)** |
| rust x vs dumped embeds | 2.4e-5 (f32 cast round-trip noise, sub-half-ULP) |

The bisect ladder: inputs equal → layer-0 input norm differs at ONE ULP →
5 layers + tail amplify to full topk disagreement. No candidate slot survives.

## The mechanism, sealed

- **Formula/eps/weights/inputs pinned equal** at layer 0: both sides read
  `rms_norm_eps = 1e-6`; the norm weight tensors come from the same
  checkpoint; the embed rows are the same dumped bf16 values.
- **The remaining delta is the fp32 fold order**: our `rn` folds mean-square
  sequentially ascending (the Issue-940 determinism law — a PINNED choice),
  MLX's fast kernel folds SIMD-pairwise. One f32 ULP in `inv` flips the
  bf16 rounding of individual products → 1-ULP x_in differences.
- **The norm staircase transcription itself is EXACT**: a direct MLX probe
  (amo, the vendor venv) shows `nn.RMSNorm` on bf16 matches our two-step
  `rnd(x·inv)` → `rnd(·γ)` staircase **0/4096 flips** — and does NOT match
  the single-round form (1264/4096 flips). Their norm IS the two-step
  staircase; our transcription of it is bit-perfect.
- **The piecewise pins never saw this** because every piece replay fed
  IDENTICAL bf16 inputs — the fold-order ULP enters between ops, and the
  chain is where it compounds. This is the 1193 lesson one class deeper: the
  gate bug was an op transcribed wrong inside the main loop; THIS finding is
  every op transcribed right, disagreeing only in reduction order —
  unfixable without giving up the 940 sequential-fold law (our side's own
  determinism pin), and equally present in the vendor-vs-vendor direction.

## What this means for plan 492 (the 679 premise, strengthened)

- **Cross-engine token identity is NOT a valid acceptance bar** for this
  draft — the bar stays what the plan already pinned: the harness
  fixture-greedy mean_j on OUR runtime (T3/T4), with the vendor chain-only
  2.882 as a STATISTICAL reference, never a row-wise one.
- The 1183-T5 failure decomposition (row-0 agreement 0.0 vs the vendor head
  on the real capture) is now MECHANISTICALLY EXPLAINED: the two engines'
  chains diverge from ULP-1 noise; the vendor head's expectations cannot
  transfer row-wise — **the distill premise (the head must see OUR runtime's
  numerics) is strengthened, not weakened**.
- Our internal consistency bars stay as landed: G1 serial==specdec (Bench
  965) on our side; the piecewise vendor parity pins (Bench 959/967) remain
  the transcription-correctness evidence at identical inputs.

## Landed

- `6ca37bcf` — the e2e arms (Rust dump + Python replay; the dump-everything
  posture: embeds, feats, the whole synthetic lm_head, tokens/candidates/
  confs, tail-trace diagnostics, layer-0 bisect dumps).
- Follow-ups in the same lane: the manifest parser (k=v tokens), the
  shadow-config (SW keys neutralized, disclosed), the tail's proposal-row
  convention (x[0,1:]), the candidates reshape, the layer-0 bisect +
  vendor-side hooks (`d7ad0d7f` head of this range — see git log 492-golden).
- The Python hook note: `l0.__call__ = rec` on an INSTANCE does not override
  (special methods are type-looked-up) — the attn/layer-out diag lines
  stayed silent; the x_in bisect carried the localization, the hooks are
  kept for a future class-based rewrite, disclosed.

Artifacts on amo: `/tmp/dspine_e2e/` (the full dump + shadow config; the
dumps regenerate deterministically from the pinned checkpoint + seed 777 —
`qwen38_dspine_t2 --e2e` rebuilds them in ~70 s).

`.benchmarks/.highwater` 967 → 968.
