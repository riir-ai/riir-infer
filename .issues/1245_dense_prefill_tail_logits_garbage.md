# Issue 1245 — dense Metal lane: the chunked-prefill tail-logits read is garbage (lm_head materialization), decode path sane

**Status:** open — filed 2026-11-23 (zed-bb86e4df@amo) from the i1243 dense-golden-census wiring's smoke legs. The chunk posture of the new census bin is BLOCKED on this; the census runs `--stride 1` (decode path, proven sane) meanwhile. Fix owner: the dense Metal league lane (Issue 1150/1176 lineage) — the lm_head chunk GEMV stage of `ingest_chunk_gpu`.

**Claimed:** zed-bb86e4df@amo, 1795456853, expires 2026-11-25T18:00:00Z
(claim scope = THIS issue's filing + the reproduction evidence below; the kernel fix claims separately.)

## The finding

`Qwen38WgpuForward` chunked prefill: after ANY whole-chunk `ingest_chunk_gpu`, the materialized `logits_b` row (read by `read_logits` / `read_chunk_argmax` / `ingest_capture`'s per-chunk `last_logits`) is GARBAGE — magnitudes 1e37..1e38, +inf/−inf, and 156–280 non-finite entries of 248320 (count scales with chunk size: 64→280, 256→156, 512→250/278). The DECODE path (`forward_token_capture`) on the SAME model + build reads sane logits (pooled golden NLL 5.74/pos, oracle-vs-incumbent Δ −2.4e-15).

## Reproduction (amo, M5 Ultra, AC; 2026-11-23)

```
# commit f6c1c155 + the fixups; instrument bin i1243_dense_golden_census
CARGO_TARGET_DIR=/tmp/i1243_tgt cargo build --release -p riir-infer-gpu \
  --features i1243_dense_golden --bin i1243_dense_golden_census
/tmp/i1243_tgt/release/i1243_dense_golden_census \
  --gguf ../riir-train/data/Qwen3.5-0.8B-Base-Q8_0.gguf \
  --corpus tests/fixtures/i1188_family_census_bonsai2_27b_pq2.txt \
  --max-positions 8 --chunk 512 --label repro
# → "[i1243] non-finite golden NLL @scored 0 (pos 511, target 25781):
#    logits max inf, min -inf, non-finite count 250"
```

- Unarmed posture (features `i1243_dense_golden` only): garbage (max 8.5e37 @chunk 256).
- **ARMED posture (the league Metal default, +`metal_tensor_gemm`): still garbage** (max 1.45e38, 278 non-finite; the f16 half-B matmul2d arm boots — the banner line prints) with `RIIR_QWEN38_LMHEAD_Q6K=0` demoted (the fixture has tied embeddings, no `output.weight` — the Q6K arm refuses loud without the demote).
- Every chunk size: 64/256/512 all garbage from the FIRST chunk boundary (scored 0).
- Control: `--stride 1` (decode, `forward_token_capture` per token) — sane, same model/build/corpus.

## The suspect surface

`ingest_chunk_gpu`'s tail stage writes `logits_b` through the lm_head row-chunk GEMV family (`lm_head_chunks` Q8-class + `lm_head_q6k_chunks` Q6K-class, offset_start per row-chunk — `qwen38_dense_cubecl.rs` ~L2398–2428). Candidates: a stale/unwritten weight handle for this fixture's tied-embedding shape (the loader's `wte` reuse for lm_head?), the `tmp_b` tail-norm feeding wrong rows at prefill scale, or the offset arithmetic on a partial last chunk. The decode path's lm_head arm is a DIFFERENT code path (sane) — the defect is prefill-specific.

## Why it matters beyond this cell

Any consumer that seeds decode from a chunk prefill ≥ 1 whole chunk reads a garbage-driven argmax (`read_chunk_argmax` over garbage picks the 1e38 entries). The bench-946 league prefill cells measure THROUGHPUT (unaffected), but any correctness consumer of chunk-boundary logits (specdec prefill distributions, prefix-graph probes) is poisoned. The serving default seeds decode via short prompts (ragged-tail path — decode kernels only), which is why the lane's G1 identity cells never caught it.

## Fix bars (pre-registered)

- G1: chunk-boundary logits byte-sane (finite everywhere, plausible magnitudes) AND consistent with the decode path's continuation (teacher-forced NLL at a chunk boundary ≈ the decode path's NLL at the same position within f32 forward noise).
- G2: `read_chunk_argmax` == decode-path argmax at the same position (the two postures agree on the greedy token).
- G3: the league posture unchanged at the throughput shapes (the fix must not regress pp rows).
