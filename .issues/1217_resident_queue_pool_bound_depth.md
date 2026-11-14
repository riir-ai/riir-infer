# Issue 1217 — resident decode queue runs ~1-deep: the fork's two hardcoded poll(Wait) caps (MAX_TOTAL_TASKS=512 / MAX_PENDING_WRITES=64) pace the host to the GPU

**Status:** OPEN — filed 2026-11-14 from [Bench 1218](../.benchmarks/1218_i1181_t1_encode_share_demote_gate.md)'s second finding (the `issue1181_encode_share_probe` per-call instrumentation, amo, census posture, 27B folded). Unclaimed — filed as the follow-up lever lane, cross-linked from issue 1181 T2's break-even; claim with a lease before working.

## The measured fact

The resident decode lane (`resident_begin/enqueue/harvest`, plan 626, promoted default-on macOS greedy via issue 1165) runs an EFFECTIVE queue depth of ~1–2 tokens, not the ring's k=8:

- Per-call `resident_enqueue()` durations are bimodal: 72 of 576 calls fast (2.7–3.5 ms = pure encode, the first enqueue after each harvest when the queue is empty), the other 504 blocking at p50 14.50 ms / p90 14.87 ms — almost exactly one GPU token (wall 16.21 ms/tok).
- Host busy share (enqueue + harvest inside wall) = **100.0% in every pair**; harvest waits 24–26 ms per 8-token burst. A host that was truly 8-deep would finish a burst's enqueues in ~24 ms while the GPU needs ~130 ms — the harvest fence would be pre-satisfied and no enqueue would ever block.
- NOT the cleanup cadence: `cleanup_interval` defaults to 0 (disabled; `CUBECL_MEMORY_CLEANUP_INTERVAL` unset → `.unwrap_or(0)` at `ternary_deltanet_gpu_forward.rs:4874`), so `memory_cleanup()` never fires.
- **Root cause (vendor source, verified 2026-11-14 same session — the allocation-backpressure hypothesis first filed here is FALSIFIED): the two hardcoded poll(Wait) sites in the vendored fork's stream** (`vendor/cubecl-wgpu-0.11.0-pre.2/src/compute/stream.rs`) — **`SubmissionLoad::regulate`'s `MAX_TOTAL_TASKS = 512`** (after 512 accumulated tasks — ≈ ONE token at ~500+ dispatches/token for 64 layers — `device.poll(Wait { submission_index: previous })` blocks the host; the fork's own comment: "the in-flight cap that paces the encoder to the GPU") and **`MAX_PENDING_WRITES = 64`** in `write_to_buffer` ("on Metal/WGSL every dispatch with scalar args writes a uniform buffer" — a second, MORE frequent poll point). The existing env knob `CUBECL_WGPU_MAX_TASKS` (default 32, `runtime.rs:204`) moves only the flush cadence, never the two Wait sites. The Issue-761 counters (REGULATE_NANOS / WRITE_POLL_NANOS, always-on in the fork) show this directly: the `counter_probe` feature (dep-only, no dense-lane feature closure — `2301ca8`) exposes them to the i1181 probe cell's resident-only window (queued on bmo, job `01a1dabb-de8f`); its rows append to Bench 1218.

## Why it matters

1. **~0.9 ms/token of decode wall is parked here**: the resident wall reads 16.21 ms/tok against the Issue-1168 P2 deep-queue floor 15.32 ms/tok. The promoted resident win is REAL (its adjudicated cells: G2 15.92 vs incumbent 18.26; the league re-cells; the 1165 default flip measured end-to-end) — but its MECHANISM is the logits-read removal, not deep queuing. The doc-comment claim "NO host sync" and the G2 record's "the queue-depth invariant holding" describe the aspiration, not the measured queue; the stable median was the periodic 1-deep pattern.
2. **It re-prices issue 1181's replay remedy**: the 3 ms/token host encode is overlapped inside the poll(Wait) today; the replay arm buys wall only after the caps lift and encode becomes critical-path (the sequence is priced into 1181 T2's break-even).
3. If the caps lift (the cheap lever below) and the queue goes ring-deep, the wall approaches the P2 floor AND the harvest fence stops costing a token's wait per burst.

## Candidate fixes (each needs its own GOAT cell; G1 = token-exact walk identity as always)

- **Env-configurable caps (the cheap lever, FIRST)**: a vendor-patch row making `MAX_TOTAL_TASKS` + `MAX_PENDING_WRITES` env-tunable (the `CUBECL_WGPU_MAX_TASKS` precedent at `runtime.rs:204` is the exact shape; this fork already carries in-repo vendor patches — the Issue-761 counters landed the same way). Raising 512 past one token's dispatch count and 64 past the per-token uniform-write count removes the per-token poll(Wait) — the queue can go ring-deep (k=8) and the wall approaches the P2 floor. GOAT instrument = the `counter_probe` cell: REGULATE/WRITE_POLL share before vs after + wall delta; G1 walk-identity must hold (the caps are scheduling-only, output bit-identical).
- **Slot-indexed scratch arenas** (DEMOTED from first candidate — unnecessary under the root cause above; the pool was never the serializer): keep only if the counter window surprises (allocation waits visible after the cap lift).
- **Consumer-side fallback** (if the vendor patch is refused): fewer tasks/token via kernel fusion makes regulate fire less often per token — a league lane already in flight; the depth dividend is a side benefit to note there, never its primary claim.

## Boundary

This is riir-infer substrate (the resident decode lane and its allocator); the consumer-side routing (serving posture) stays riir-ai. The env-switch law applies to any new arm (opt-in first; the promoted resident default only changes through the GOAT gates + the 1165-style owner verdict). The win must be greedy-lane-scoped at first (sampled stays incumbent), same as the resident lane's own scoping.
