# 005 — Checkpoint-composition algebra (Issue 1242 T4 design note)

**Status:** DESIGN NOTE — the composition algebra for checkpoint→checkpoint
transforms over the quant-recovery artifact family (deletion masks, KD/LoRA
overlays, scale-clip requants, weight soups). Implementation waits for the
second consumer (the two-instance bar, Issue 1242 T4); this note fixes the
definitions so the first two instances cannot drift apart.

## The transforms

Each artifact family is a typed transform `T: Base → Derived` over the same
architecture's weight set:

| family | transform shape | typed key |
|---|---|---|
| deletion mask (1208 `GateMaskOverlay`) | zero-out per-channel gates, base frozen | mask digest + base digest |
| KD/LoRA overlay (508) | low-rank delta on frozen base | rank + adapter seed + base digest |
| scale-clip requant (1243(c)) | per-family scale clip + requant | family set + clip params |
| weight soup (1243(d)) | convex weight-space average | member digests + weights |

## The three laws the algebra pins

1. **Memoizable content-addressed prefixes.** Every transform declares its
   INPUT digest; a composition `T2 ∘ T1` is addressable by
   `blake3(T2.key ‖ T1.output_digest)` — so a shared prefix (the same base
   through T1) computes ONCE and serves every suffix. The cache key binds
   the FULL ordered chain, never the set: `[T1,T2]` and `[T2,T1]` are
   different addresses (see law 2).
2. **Typed non-commuting ordering.** The transforms do not commute:
   mask-then-requant ≠ requant-then-mask (the requant's scale fit sees
   different live weights), and soup-then-overlay ≠ overlay-then-soup (the
   overlay's deltas are defined against a specific soup point). The typed
   key therefore carries ORDER; a composition attempt whose types declare
   incompatible ordering (e.g. an overlay keyed to base B applied after a
   soup that moved off B) refuses LOUD with the digest mismatch — never a
   silent wrong-model artifact.
3. **Idempotence collapse.** Applying the same family twice with identical
   keys collapses: `T ∘ T = T` for idempotent families (masks: masking
   already-masked gates is the same mask; scale clips: clipping at the same
   params is stable). Non-idempotent families (soup: averaging is not
   idempotent; LoRA stacking: deltas add) REFUSE the double-apply loudly
   instead of silently recomposing — the typed key's family tag carries the
   idempotence bit.

## What this buys

- The census instruments (1242 T1/T2) score any node of the composition DAG
  from its address alone — a candidate chain re-runs only its uncached
  suffix.
- The T3 break-even reads `F` (the fixed cost) off the chain's uncached
  suffix length instead of the whole build — the memo makes proxies cheaper
  honestly.
- The refuse paths make "which chain produced this artifact?" a digest
  question, answerable forever from the artifact itself.

## Implementation bar (when the second consumer lands)

The address type + the refuse paths are ~200 lines over the existing
digest plumbing (blake3, the overlay vehicle's base-digest binding already
exists in 1208). The census bins need an `--chain` flag accepting ordered
keys. Nothing else moves.
