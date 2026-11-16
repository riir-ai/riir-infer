# Bench 1221 — i1199 T3 deciding cell: the CPU-stage alternation tax at real Bonsai-2 27B geometry (bmo, exclusive, post-fix feature law)

**Verdict: rung (b) DECLINED at real geometry (+0.5% wall, noise — after the synth-geometry LOSS, +5–16%); the alternation tax reproduces but is BOUNDED at 1.43× (not the 2.6–3.3× band); the fused batch is the best arm and only reaches SERIAL PARITY — the loopback-CPU overlap thesis does not pay at 27B on this box. No promotion anywhere. Issue 1199 closed on this record.**

## Cell

- Example: `crates/riir-infer-dist/examples/i1199_interleave_cell` — the 1199 T3 deciding cell (real Bonsai-2 27B CPU stages: per-stream GDN+KV ≈ 3 MB + scratch, 64 layers where the dispatches amortize — the premise the synth cell could not test).
- Box: **bmo** (M5 Ultra 96 GB, 30 cores), Exclusive rig job `01a1e210-aeb6-7187-aabc-5abad96b066f`, release build.
- `PROVENANCE: power=AC Power load=2.20 swap=55236.44M canary=skipped powermode=0(auto)` — preflight PASSED, box quiet.
- Model: `Ternary-Bonsai-2-27B-PQ2_0.gguf` (7.2 GB, sha-pinned LOCAL-KEEP, Hadamard-folded — the loader's `prism.hadamard` refusal on the first attempt `01a1e1cb` is the fail-closed feature law working; the real arm requires `--features bonsai2_hadamard`).
- Arms: `--warm` warmup, serial (1 stream), rr (2-stream round-robin, per-frame path — the taxed arm), batch (fused, `RIIR_TAIL_LAYER_INTERLEAVE=0`), batch+il (fused, `=1`, the rung-(b) arm). 32 tokens/stream, 64 head steps in the 2-stream arms.

## The numbers

| arm | streams | wall | tail fwd/step | tail recv/step | head recv/step |
|---|---|---|---|---|---|
| serial | 1 | 2.051 s (32 tok) | **39.1 ms** | 24.8 ms | — |
| rr (taxed) | 2 | 4.590 s (64 tok) | **55.8 ms = 1.43× serial** | 15.8 ms | 22.7 ms |
| batch (fused) | 2 | 4.096 s (64 tok) | 1284.3 ms/fused-pass | 17.2 ms | 33.9 ms |
| batch+il (rung b) | 2 | 4.117 s (64 tok) | ~batch | ~batch | ~batch |

## What the cell says

1. **The tax reproduces but is bounded**: 1.43× on the tail fwd — far under the 2.6–2.9× (4090) / 2.68–3.28× (amo release) that motivated the issue. The T3 amortization hypothesis held: at 64 layers the layer-dispatch cost amortizes over ~ms-per-layer compute, and the per-stream working set no longer dominates the alternation step. The tax is real but NOT the wall-dominant term at this geometry.
2. **Rung (b) declined at real geometry**: batch+il wall 4.117 s vs batch 4.096 s = **+0.5% (noise)** — after the synth cell's outright LOSS (+5–16%, Bench 970). The `StageCompute` layer-split quartet + cursor stay LANDED (T2, bit-identity gated) but the interleave arm stays OFF (`RIIR_TAIL_LAYER_INTERLEAVE` promote-only, never promoted).
3. **The honest headline — the loopback-CPU overlap thesis does not pay at 27B here**: the best 2-stream arm (batch) only reaches serial parity (4.096 s vs 2 × serial 2.051 s = 4.102 s), and the rr arm is 12% WORSE than running the two streams strictly serially (4.590 vs 4.102). The Bench-908 bubble-fill thesis stays confirmed in its GPU/compute-bound regimes (908 real-model loopback 1.64×, 2-box GPU 2.37×) and refuted at 27B CPU-stage loopback on this box.
4. **The fused batch sidesteps the worst** (consistent with Bench 970's synth finding): one fused tail pass per pair (1284 ms) beats 39 round-robin steps (2 × 39 × 55.8 ms ≈ 4352 ms of tail fwd) — batching the streams' tail work is where the win lives, not interleaving layers inside it.

## Disposition

- T1 (mechanism) / T2 (fix ladder, rung b implemented + gated) / T3 (this cell) — complete.
- **No promotion**: rung (b) declined at both geometries; `RIIR_TAIL_LAYER_INTERLEAVE` stays promote-only, OFF.
- The 1198/908 scheduler keeps its proven regimes (GPU/compute-bound); loopback CPU stages at 27B stay serial-or-batched, never round-robin.
- Issue 1199 closed on this record (file removed; `HISTORY.md` carries the arc; the claim lease folded in the same commit).
