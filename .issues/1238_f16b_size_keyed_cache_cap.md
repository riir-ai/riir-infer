# Issue 1238 — cap the f16-B size-keyed caches (shadow pool + twin input cache) for many-prompt-length serving

**Status:** IN PROGRESS — T1 measured (analytic key census, growth matters: up to 27 GiB at chunk 512); T2 in flight.
**Claimed:** katgpt-rs-93@amo, 1795100000, expires 2026-11-21T16:00:00Z

## The finding (from the verdict, re-checked in code)

Both f16-B twin caches are keyed by the frame's element count and never evicted:
- `f16b_shadow_pool` (`crates/riir-infer-gpu/src/ternary_deltanet_gpu_forward.rs`, `prefill_rotate_with_shadow` — `pool.entry(total * 2).or_insert_with(..)`);
- the twin's `input_cache` (`crates/riir-infer-gpu/src/gemm_ternary_metal_zero_copy.rs`, the cast-fallback f16 input).

At a fixed frame width steady state allocates nothing (G4 holds — Bench 1245). A serving process sees every distinct final-frame (tail) length, though, and each new length adds one GPU buffer per key that lives for the process lifetime. Under F2's default-on posture the shadow pool replaces `input_cache` growth on the served sites rather than adding to it, so this is a pre-existing growth class, NOT an F2 regression.

## Tasks

- [ ] **T1 — measure the growth:** count distinct keys and resident bytes for both caches over a realistic prompt-length mix (e.g. the dist head's chunk 128 with uniform 1..=chunk tails). Report bytes per key and the worst case at the serving context cap.
- [ ] **T2 — bound it (only if T1 shows growth that matters):** round keys up to a bucket (e.g. the next multiple of the frame width, so a tail reuses the full-frame buffer — a sub-slice bind is already the offset-aligned path) OR LRU-cap the map. Bit identity must stay byte-identical: re-run `i1226_f2_price_cell` in identity mode at chunk 128 + ragged tail.
- [ ] **T3 — G4 re-check:** steady-state zero allocation at a fixed width is unchanged.

## Not in scope

The f32 incumbent's own handle pools (separate substrate).
