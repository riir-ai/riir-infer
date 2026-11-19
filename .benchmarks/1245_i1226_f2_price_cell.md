# Bench 1245 — i1226 F2: the producer-writes-f16 PRICE CELL (pp4096 + pp32768)

**Status:** COMPLETE. **GAIN at both sizes, positive on all 5 pairs**, with C's logits byte-identical to B's end to end.
**Date:** 2026-11-19. Job `2eec23ab` (amo, exclusive), 13:58Z–14:46Z.
**Lease:** katgpt-rs-ec@amo (claim `18e09278`).
**Instrument:** `crates/riir-infer-gpu/tests/i1226_f2_price_cell.rs` (`6e420ad1`), which shares `tests/support/f16b_frame.rs` with the Bench-1225 frame A/B.
**Scope:** riir-infer. Lane record: `.issues/1226_f16b_lane_followups.md` (F2 + F3).

## The cell

The production frame posture is the Bench-1225 posture: relaxed matmul2d, tensor zerocopy + attn-tensor, every promoted rung (feature row `prefill_f16b,production,cubecl_runtime,multi_host,riir-infer-core/bonsai2_hadamard`, `--release`, `--test-threads=1`). Model: Bonsai-2-27B PQ2_0. Chunk 512; 8 frames (pp4096) and 64 frames (pp32768).

- **Arm B (the incumbent per the gate order):** the armed f16-B twin with its per-launch `cast_f32_f16_zc`. This is Bench 1225's armed posture.
- **Arm C:** B plus the producer shadow (`set_f16b_producer(true)`). The rotate writes the f16 pairs and the twin skips the cast encoder.
- **Arm A:** the f32 lane, route off. It runs once per pair AFTER the B/C block, so it is informational only and never part of the Δ.
- **Pair design:** 5 ABBA pairs (B C C B / C B B C), with 2 untimed warm passes per arm.

## Engagement (asserted, one counted pass per arm)

| size | B cast encoders | B skipped | C cast encoders | C skipped | misaligned | saved / layer / frame |
|---|---:|---:|---:|---:|---:|---:|
| pp4096 | 3456 | 0 | **0** | 3456 | 0 | 6.75 |
| pp32768 | 24960 | 0 | **0** | 24960 | 0 | 6.09 |

**C runs ZERO cast encoders at both sizes:** every twin-routed cast on the folded posture is replaced by the shadow, which confirms the scoping notes' "9/9 sites, no phase 2 on the serving posture". The per-layer-per-frame count differs between sizes (6.75 vs 6.09) because the count follows the twin's own per-site admission at each size. B's encoder count sets the denominator, and C removes all of it.

## Bit identity and anchors (asserted)

At both sizes: A × 5, B × 10 and C × 10 are each identical across their own passes. **C == B byte-for-byte** (the end-to-end shadow identity, which upgrades G1's per-kernel identity to the whole frame), and A ≠ B (the twin is live).

## The price

| size | B median (ms) | C median (ms) | Δ median (ms) | Δ range (ms) | % of B | positive on all pairs | prediction (ms) | measured / prediction-mid |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| pp4096 | 3830.85 | 3784.63 | **39.76** | [37.00, 66.54] | **1.04%** | yes, 5/5 | 518–657 | 0.07 |
| pp32768 | 81988.66 | 81736.50 | **257.30** | [225.37, 327.06] | **0.31%** | yes, 5/5 | 3744–4742 | 0.06 |

**The measured saving is ~6–7% of the Bench-1223 per-launch prediction.** The 0.15–0.19 ms tax was measured per launch in isolation; inside the pipelined frame most of the cast encoder overlaps with other in-flight GPU work. Only the serial residue is removed, about 11.5 µs per saved launch at pp4096 and 10.3 µs at pp32768. Removing the casts is a real, sign-consistent gain, but it is the size of a launch-overhead trim, not of the isolated tax. The absolute saving grows with prompt length (40 → 257 ms) while the share shrinks (attention's quadratic term dominates the pp32k frame).

**f32-lane line (informational, not interleaved):** C is 6.71% under A at pp4096 (A median 4056.69 ms) and 2.48% under A at pp32768 (A median 83819.20 ms). This is the whole f16-B lane's gain, the twin plus the producer.

## Provenance (F3)

- pp4096: `PROVENANCE: power=AC Power load=5.70 swap=3905.12M canary=skipped powermode=0(auto) ane=ok` at start; load 2.64 at end. **Clean:** the cell's log closed at 14:00:46Z, before the poison mark.
- pp32768: `PROVENANCE: power=AC Power load=2.64 swap=3905.12M canary=skipped powermode=0(auto) ane=ok` at start; `load=1.87` at end. **POISONED** (disclosed): the rig flagged Zed (pid 49644, a 4-day-old editor process going busy at ~55% of one core) at 14:01:24Z, at the start of the pp32k cell. **Kept per the rule recorded at the poison time:** re-run only if the result is not sign-consistent. All 5 pairs are positive, with a tight range (225–327 ms) against a 1.8% spread in A, and the ABBA order cancels steady load.
- Swap of 3.9 GB is disclosed (preflight `warn`). The canary was skipped (not built on this box).
- Binary built from the `6e420ad1` content (the run-time tree and the commit are byte-identical for the instrument); develop HEAD at launch was `18e09278`.
- Logs: `1245_i1226_f2_price_cell_logs/` (per-size cell log + start/end preflight).

## What it unblocks

The gate order's next rungs:
- **Tap retention on one posture:** satisfied BY CONSTRUCTION here. C's frame logits are byte-identical to B's at both sizes, so C inherits B's measured retention (Bench 1224 at pp4096, Bench 1234 at pp32768) exactly. Running the tap would compare identical bytes.
- **GOAT against Bench 1225's armed posture:** G1 is bit-identical to the incumbent. G2 is this cell. G3 holds because the producer touches only `launch_forward`/`launch_forward_copy` on the prefill rotate (decode's `launch_inverse` is untouched). G4 holds because the shadow pool keys by size with `or_insert_with`, so steady state allocates nothing.
- **Then the delegated-owner verdict** and the dist-lane default flip.

## Addendum — the delegated verdict's condition (b): identity at the dist frame width (2026-11-19, shared job `ad6187af`)

The verdict was **AGREE WITH CONDITIONS**: the dist head's default `--prefill-chunk` is 128, and the timed cell ran only chunk 512. Identity-only mode (`I1226P_IDENTITY_ONLY=1`, B C C B, every digest equal) was run on the FINAL predicate (condition (c): the shadow also requires the zero-copy flag + its site admission):

| shape | tok | B cast encoders | C cast encoders (skipped) | misaligned | C == B |
|---|---:|---:|---:|---:|---|
| chunk 128 × 32 + tail 37 | 4133 | 12672 | 0 (12672) | 0 | **byte-identical × 4** |
| chunk 512 × 8 + tail 2 (re-check) | 4098 | 3456 | 0 (3456) | 0 | **byte-identical × 4** |

Logs: `identity_c128_tail37.log`, `identity_c512_recheck.log`. Correctness only, so no timing claim (shared lane).
