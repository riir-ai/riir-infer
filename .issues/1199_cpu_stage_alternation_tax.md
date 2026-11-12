# Issue 1199 — the CPU-stage alternation tax: tail fwd inflates 2.6–2.9× when the multi-stream round-robin alternates per-stream stage instances (riir-infer 1198's measured root cause; the loopback overlap win the scheduler was built for is real once the tax is gone)

**Status:** OPEN — filed 2026-11-09 (zed-aac1c647@amo) out of 1198's T1 evidence.
**Claimed:** zed-aac1c647@amo (amo), 1794250000, expires 2026-11-11T18:30:00+00:00
**Claimed:** zed-93bcde10@amo, 1794438486, expires 2026-11-13T23:48:00+00:00
The T1–T3 alternation-tax lane starts now (prior lease expired unworked; this push is the arbiter).

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

- [x] T1 — verify the mechanism directly — **SYNTH-SCALE HALF ANSWERED (Bench 970)**: the poor-man's discriminator ran inside the T3 cell — at the synth geometry the rr inflation is NOT state-residency (tail recv inflated 3.7× alongside fwd 5.5×, and the per-stream working set is KBs — everything fits L2 regardless of alternation): the tax there is per-step thread/pipeline handoff cost amplified by the single-drain cadence. The COUNTER-SCALING leg (one stream's cache physically larger/smaller — the config-scaling probe that separates residency from scheduling) folds into the real-geometry exclusive cell, where the premise can actually hold.
- [x] T2 — the fix ladder — rung (b) IMPLEMENTED + GATED (Bench 970): the `StageCompute` layer-split quartet + the `TernaryCpuStage` cursor (mis-use refuses loud) + the armed `BatchN` branch (`RIIR_TAIL_LAYER_INTERLEAVE`, PROMOTE-ONLY, read per batch frame so rigs flip it in-process); core grew `forward_qwen_deltanet_ternary_layer_at` (one layer, pool-wrapped) + `forward_qwen_deltanet_ternary_finalize_into` (pool-wrapped LM head — the end leg keeps the incumbent pool posture); the whole-step path untouched. Bit-identity gates: `multi_stream_interleave_g1.rs` (oracle + composition + 6 mis-use refusals), full dist suite 70/70, clippy `-D warnings` (dist crate whole + core lib). (a) and (c) remain declined-by-analysis; (d) stays the 1198 re-scope's job.
- [ ] T3 — the honest verdict bench: **PROVISIONAL cell RUN — Bench 970** (synth geometry, release, two rounds, PROVENANCE: amo with 3 contender builds — never a GOAT quote): the rr tax reproduces at 5.5× per-step fwd (0.169 → 0.95 ms) but its shape is SCHEDULING-bound at this geometry (tail recv inflated 3.7× too; per-stream state is KBs — residency cannot matter), the fused batch already sidesteps the worst (wall 663→148 ms), and rung (b) LOSES at small scale (+5–16% wall — layers×streams pool dispatches dominate). **No promotion.** The DECIDING cell (real Bonsai-2 27B CPU stages: ~3 MB/stream state + scratch, 64 layers where the dispatches amortize) is parked on the exclusive quiet window — the same example re-run against the real checkpoint (`examples/i1199_interleave_cell`).

## Boundary

riir-infer only (the dist stage/forward + the multi.rs scheduler). Zero new deps; the katgpt-core edge untouched.
