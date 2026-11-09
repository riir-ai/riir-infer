# Issue 1199 — the CPU-stage alternation tax: tail fwd inflates 2.6–2.9× when the multi-stream round-robin alternates per-stream stage instances (riir-infer 1198's measured root cause; the loopback overlap win the scheduler was built for is real once the tax is gone)

**Status:** OPEN — filed 2026-11-09 (zed-aac1c647@amo) out of 1198's T1 evidence. **Claimed:** zed-aac1c647@amo (amo), 1794250000, expires 2026-11-11T18:30:00+00:00.

## The measured fact

riir-infer 1198's step-timing decomposition (4090, Windows, quiet, HEAD e595342, T3.5 instrument armed):

| arm | tail fwd/step | tail recv/step | head recv/step | per-step wall |
|---|---|---|---|---|
| serial (1 stream/connection) | 0.19–0.25 ms | 0.19–0.24 ms | 0.19–0.24 ms | ≈ 0.42 ms |
| interleaved (2 streams, one connection, round-robin) | **0.57–0.60 ms (2.6–2.9×)** | 0.01–0.05 ms | 0.32–0.38 ms | ≈ 0.60 ms |

The interleaved pipeline IS formed (the tail never blocks — the one-pass-behind steady state the plan-623 T2.4 doc describes works), but the arm is **tail-bound on an inflated tail**: alternating the two per-stream `TernaryCpuStage` instances every step costs more than the overlap saves. RAYON_NUM_THREADS=1 leaves it unchanged (not pool contention); correctness asserts pass everywhere (purely a memory/locality effect).

## Why it matters

The multi-stream scheduler's thesis (Bench 908) is bubble fill: per-token cost `max(head, tail + rtt)` instead of their sum. The thesis is CONFIRMED in the compute-bound/GPU regimes (908's real-model loopback 1.64×, the 2-box GPU cell 2.37×) — but on loopback CPU stages at synth scale the alternation tax inverts it (ratios 1.2–3.3). Kill the tax and the loopback pipeline realizes the design intent: per-step ≈ `max(h, t)` ≈ 0.25 ms → ratio ≈ 0.6, the bar Bench 908 wrote would hold honestly.

## Suspected mechanism (to verify, not assumed)

Each `TernaryCpuStage` owns its own `HybridCache` (GDN recurrent state ≈ 1 MB for the synth geometry; attention-layer KV cache ≈ 2 MB at block 2048) + `HybridForwardScratch`. The tail's per-step working set therefore doubles when it alternates streams, exceeding L2-class residency — the state streams from L3/DRAM every step. The GDN recurrence is state-bound (S = exp(g)·S + k ⊗ (β(v − Sᵀk)) touches the whole state per step), so the inflation is memory-bandwidth-shaped, not compute-shaped. amo's larger ratio (2.68–3.28) vs the 4090's (1.23–1.37) is the same tax under different cache geometry.

## Tasks

- [ ] T1 — verify the mechanism directly: repeat the 1198 instrumented run with ONE stream's cache physically larger/smaller (the synth config's context_length/block_size) and confirm the tax scales with per-stream state, not with compute. A per-step L2-miss counter (macOS `performance counters` via `taskinfo`, or a poor-man's version: time the step with the OTHER stream's state touched between calls vs not) pins it.
- [ ] T2 — the fix ladder (each rung gated by the same instrumented read; promote only what measures): (a) per-stream scratch/state placement hints (align + non-temporal hints are weak on portable Rust — likely declined); (b) interleave the tail's alternation at LAYER granularity instead of TOKEN granularity (both streams' layer-2 steps back to back reuse layer-2 weights/state locality before moving to layer 3 — needs a protocol change: the tail processes one LAYER across streams per pass; the head's cadence unchanged, the wire order per stream preserved); (c) a tail-side step cache: keep both streams' GDN state in one arena laid out so alternating streams touches disjoint cache lines anyway (the state is per-stream contiguous today — measure whether an interleaved layout actually helps; likely declined — disjoint lines are already disjoint); (d) accept the tax and scope the scheduler's win claim to compute-bound regimes (the 1198 re-scope already does the claiming half).
- [ ] T3 — the honest verdict bench: the loopback instrumented read at the best rung vs the 1198 baseline, both postures (release + the 908 debug posture), quoted with `bench_preflight` PROVENANCE; the win claim only in the regime where it measures.

## Boundary

riir-infer only (the dist stage/forward + the multi.rs scheduler). Zero new deps; the katgpt-core edge untouched.
