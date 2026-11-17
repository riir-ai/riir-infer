# Bench 1235 — i1221 T3(d): the loopability predictor cell on the dense pin

**Status:** CLOSED — cell COMPLETE 2026-11-17 (zed-b2e60a87@amo, amo, release CPU). Report-only readouts; the only gates in the lane are the lib identity fixtures (Bench record §Files). Read order leg (d) of [`.plans/636`](../.plans/636_i1221_t3_read_order_preregistration.md) done; (a) is next.

## What ran

The T3(d) dense-GGUF loopability predictor (Research 672 §3, the No-GD advocate's variant) on the dense subject pin, chunk = layers 8..16 (the stack's middle third: 6 DeltaNet + 2 Attention layers), ks = 1/2/4/8, 3 pseudo-token prompts (LCG, seed 0, seq 8 — the smoke's law; no tokenizer yet), full-prompt chunk passes (the Issue-929 `loop_probe` hybrid-native semantics; GDN states advance + attention KV rows overwrite per pass), the loop recurrence verbatim (`s_0 = x; u_t = N(s_{t-1}+x); s_t = N(F_chunk(u_t))`, N = unweighted RMS ε=1e-6 in division form), readouts per (prompt, k): λ_t ladders (Frobenius-pooled + last position), KL(full‖loop-k) + KL(loop-1‖loop-k) in f64 at the last position, argmax + margins + flip.

- Pin: `../riir-train/data/Qwen3.5-0.8B-Base-Q8_0.gguf` — blake3 `70fd9c83d63454ae72e07a29ebdb78f863cfad102d8939f6ec49b2aea32104e2`, arch qwen35, 24 layers (18 DeltaNet + 6 Attention), n_embd 1024, vocab 248320.
- STAMP: riir-infer `256c8647` (the instrument landing; the resume-scan fix `27296d6e` is runner-only, post-cell — the rows' computation lineage is 256c8647).
- Rows: `.benchmarks/i1221_t3d_loopability_cell/rows.jsonl` (header + 12 rows, committed; resume = rerun the same command, the lineage guard refuses a moved STAMP/blake3 exit 2).
- PROVENANCE: `power=AC Power load=5.40 swap=4449.19M canary=skipped powermode=0(auto)` — **concurrent-with-light-load DISCLOSED** (the F2 sibling lane was compiling; a single cargo job on 30 cores). The readouts (λ, KL, argmax, margins) are exact-determinism quantities — cross-run bit-identity below is the proof load does not reach them; the row `wall_ms` figures are throughput CONTEXT ONLY, load-contaminated, never a bar or a claim.
- Cell wall: 16.4 s (12 rows + 3 references; model load 2.5 s, release, 30 threads).

## The folded table (per-prompt rows agree to 2–3 significant figures — the chunk map's behavior is prompt-robust)

| k | λ_pooled ladder mean (p0/p1/p2) | KL(full‖loop-k) (p0/p1/p2) | KL(loop-1‖loop-k) (p0/p1/p2) | flip vs full |
|---|---|---|---|---|
| 1 | — | 4.098 / 3.174 / 4.581 | — | 3/3 |
| 2 | 0.1306 / 0.1263 / 0.1309 | 3.851 / 3.041 / 4.350 | 0.0428 / 0.0470 / 0.0434 | 3/3 |
| 4 | 0.5588 / 0.5652 / 0.5595 | 3.572 / 2.966 / 4.062 | 0.2369 / 0.2673 / 0.2375 | 3/3 |
| 8 | 0.7323 / 0.7409 / 0.7350 | 3.406 / 3.014 / 3.790 | 0.6251 / 0.7247 / 0.6131 | 3/3 |

Per-position shape: the λ ladders POOLED and LAST-POSITION agree closely (e.g. p0 k=2: 0.1306 pooled vs 0.1344 last) — contraction is position-uniform, not a last-token artifact.

## Verdict (measurement, never a gate)

1. **The chunk iteration CONTRACTS, strongly, from the first step.** Every λ_t at every (prompt, depth) is < 1: λ_2 ≈ 0.126–0.131 (the second pass's state move is ~7.7× smaller than the first's), rising to ≈ 0.73–0.74 at k=8 — the ladder RISES toward the map's local spectral radius as s_t converges into the anchor-injected fixed point, but never crosses 1 anywhere in the measured range. The loop-ability premise holds on dense hybrid weights WITHOUT any loop training.
2. **The loop is NOT inert, and it deepens monotonically.** KL(loop-1‖loop-k) grows ~0.043 → ~0.24 → ~0.63 across k = 2/4/8, near-identical across all three prompts. Iterating the chunk past depth 1 changes the readout at every additional depth — the computation genuinely deepens (the Dory k-scaling premise's dense-arm shadow).
3. **The looped readout is FAR from the full-depth readout:** KL(full‖loop-k) ≈ 2.9–4.6 nats at every depth. Expected — the loop variant replaces 8 DISTINCT layers with the same 8 layers iterated; an untrained chunk has no loop-invariance. This number is the baseline T3(a) calibrates against: on the TRAINED-looped Dory pin, per-loop KL should sit orders of magnitude below this dense-arm's (loop-trained invariance vs this chunk's none), while the λ signature (contract from a small λ_2, rise toward < 1) is the shape to compare against Dory's trained recurrent block.
4. **argmax flips at every (prompt, k) vs full-depth** — expected at pseudo-token prompts (reference margins 0.002 / 0.206 / 1.729; p0's reference is a near-tie). A measurement, never a quality claim.

## Determinism receipts

- Lib fixtures (run at `cargo test --lib --features deltanet_inference`, 349 passed / 0 failed): the range seam's composition pins — full-range entry ≡ incumbent BIT-IDENTICAL and a 3-way range split composes BIT-IDENTICALLY on random-weight hybrid models (zero weights would pass vacuously; the fill is the seeded LCG helper) — plus 10 loopability fixtures (wellformedness, not-inert, two-construction determinism, KL identity, exact-zero-denominator contraction, chunk-validation refusals, N idempotence).
- Cross-run bit-identity ON THE REAL PIN: the full cell re-run into a scratch lineage reproduced all 12 rows BIT-IDENTICAL with only `wall_ms` stripped (readouts are process-independent exact f32/f64).

## Resume / rerun

```sh
CARGO_TARGET_DIR=/tmp/<dir> cargo run --release --features deltanet_inference \
  --example i1221_t3d_loopability -- \
  --gguf ../riir-train/data/Qwen3.5-0.8B-Base-Q8_0.gguf \
  [--out .benchmarks/i1221_t3d_loopability_cell] [--ks 1,2,4,8] [--seq-len 8] \
  [--prompt-sets 3] [--seed 0] [--mid-start 8] [--chunk-len 8] [--fresh]
```

Resume verifies the header lineage (blake3 + STAMP + chunk + ks + prompt spec; mismatch REFUSES exit 2 — never append across lineages) and skips completed (prompt, k) pairs; per-row writes flush immediately. `--fresh` starts a new lineage.

## Files

- riir-infer `256c8647` — the instrument: `src/deltanet/loopability.rs` (new), `src/deltanet/forward.rs` (the `prefill_layer_range` verbatim extraction + `prefill_qwen_deltanet_range_into` + `PrefillContext` accessors + the two composition pins), `src/deltanet/mod.rs` (registration + the `fill_random_weights` test helper), `Cargo.toml` ([[example]] row), `examples/i1221_t3d_loopability.rs` (the resumable runner).
- riir-infer `27296d6e` — the resume-scan fix (LoopKRow is flattened into the row line; the scanner read the nested spelling and would have panicked on the first resumed run; resume-skip + lineage-refusal both proven live post-fix).
- `.issues/1221` — the T1-pickup lease folded (Lease closed row) + the fresh T3(d) cell claim; plan 636 status amended (leg (d) instrument landed → this record closes the cell).
