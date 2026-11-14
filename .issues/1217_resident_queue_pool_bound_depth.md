# Issue 1217 — resident decode queue is pool-bound ~1-deep: lift the allocation ceiling (the deep-queue premise of plan 626)

**Status:** OPEN — filed 2026-11-14 from [Bench 1218](../.benchmarks/1218_i1181_t1_encode_share_demote_gate.md)'s second finding (the `issue1181_encode_share_probe` per-call instrumentation, amo, census posture, 27B folded). Unclaimed — filed as the follow-up lever lane, cross-linked from issue 1181 T2's break-even; claim with a lease before working.

## The measured fact

The resident decode lane (`resident_begin/enqueue/harvest`, plan 626, promoted default-on macOS greedy via issue 1165) runs an EFFECTIVE queue depth of ~1–2 tokens, not the ring's k=8:

- Per-call `resident_enqueue()` durations are bimodal: 72 of 576 calls fast (2.7–3.5 ms = pure encode, the first enqueue after each harvest when the queue is empty), the other 504 blocking at p50 14.50 ms / p90 14.87 ms — almost exactly one GPU token (wall 16.21 ms/tok).
- Host busy share (enqueue + harvest inside wall) = **100.0% in every pair**; harvest waits 24–26 ms per 8-token burst. A host that was truly 8-deep would finish a burst's enqueues in ~24 ms while the GPU needs ~130 ms — the harvest fence would be pre-satisfied and no enqueue would ever block.
- NOT the cleanup cadence: `cleanup_interval` defaults to 0 (disabled; `CUBECL_MEMORY_CLEANUP_INTERVAL` unset → `.unwrap_or(0)` at `ternary_deltanet_gpu_forward.rs:4874`), so `memory_cleanup()` never fires — the serializer is the ALLOCATION BACKPRESSURE inside the CubeCL launch path: token n+1's intermediate blocks cannot come out of the pool until token n's are freed.

## Why it matters

1. **~0.9 ms/token of decode wall is parked here**: the resident wall reads 16.21 ms/tok against the Issue-1168 P2 deep-queue floor 15.32 ms/tok. The promoted resident win is REAL (its adjudicated cells: G2 15.92 vs incumbent 18.26; the league re-cells; the 1165 default flip measured end-to-end) — but its MECHANISM is the logits-read removal, not deep queuing. The doc-comment claim "NO host sync" and the G2 record's "the queue-depth invariant holding" describe the aspiration, not the measured queue; the stable median was the periodic 1-deep pattern.
2. **It re-prices issue 1181's replay remedy**: the 3 ms/token host encode is overlapped inside the pool wait today; the replay arm buys wall only after depth lifts and encode becomes critical-path (the sequence is priced into 1181 T2's break-even).
3. If the pool can be made k-deep (pre-allocated per-slot intermediate arenas keyed by ring slot — the resident ring already carries the slot index; or a CubeCL-side block reservation), the wall approaches the P2 floor AND the harvest fence stops costing a token's wait per burst.

## Candidate fixes (each needs its own GOAT cell; G1 = token-exact walk identity as always)

- **Slot-indexed scratch arenas**: the resident path allocates its intermediates from a pool keyed by `t % k` — each ring slot owns a private scratch set, so k tokens coexist by construction. Zero new sync; the allocation ceases to be a serialization point. Cost: k× the intermediate residency (k=8 × the per-token working set — measure the KV/scratch footprint first; the Issue-864 KV line suggests the working set is known and bounded).
- **A depth probe first (T1 of this issue)**: instrument the pool — confirm the blocking site is allocation (a `pool_alloc` wait counter) and measure the true max depth the pool sustains today, before designing the fix. The bench-1218 inference is strong but indirect; this is the cheap decisive read.
- **CubeCL-side**: check whether the vendored cubecl 0.11.0-pre.2 exposes a reservation/reserve API (the upstream `c1d0125` code-cache work touched this neighborhood); a vendor patch is the last resort (the vendor law: prefer consumer-side fixes).

## Boundary

This is riir-infer substrate (the resident decode lane and its allocator); the consumer-side routing (serving posture) stays riir-ai. The env-switch law applies to any new arm (opt-in first; the promoted resident default only changes through the GOAT gates + the 1165-style owner verdict). The win must be greedy-lane-scoped at first (sampled stays incumbent), same as the resident lane's own scoping.
