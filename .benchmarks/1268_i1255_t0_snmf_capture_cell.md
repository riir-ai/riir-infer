# Bench 1268 — Issue 1255 T0: SNMF group-feature fixture cell (capture + semi-NMF + mixer statistics)

**Status:** LANDED 2026-11-26 — T0 complete on amo; the cell is `examples/snmf_census_cell.rs` (required-features `closed_form_write`).
Issue: `riir-infer/.issues/1255_snmf_group_feature_census_lane.md` (claim lease to 2026-11-28T11:32Z). Source: katgpt-rs research 695 (arXiv:2506.10920, ACL 2026).

## Posture

- Model: `Qwen3.5-0.8B-Base-Q8_0` (24 layers, n_embd 1024, **mlp 3584**; 18 DeltaNet + 6 Attention layers).
- Capture: the `closed_form_write` `CaptureSink` seam REUSED (the down-proj input rows = the SwiGLU activations — one capture machinery, two consumers, zero new forward code). One prefill over 768 token samples (ag_news canonical corpus, digest `0ea70685f7435cb9952f2c67fa03b8d3b36efd3a542cb90e28b96a2da4f7ac2a`), ALL 24 layers captured in that pass.
- SNMF: semi-NMF per representative layer (layer 0 DeltaNet + layer 3 Attention) — multiplicative Y updates (positive/negative-part split), closed-form ridge Z step `Z ← A Yᵀ(YYᵀ+λI)⁻¹` (Gauss-Jordan, partial pivoting), per-step column re-normalization with the scale folded into Y, hard WTA top-1% per feature column (36 neurons of 3584), k=100, λ=1e-2, ≤1500 iters, SplitMix64 seeded on the layer's capture digest. All math f64; every cross-cell reduction folds ORDER-COLLECTED chunk partials (rayon `collect` is order-preserving, final fold sequential) — no scheduling-dependent reduce order feeds the loop or receipts.
- Box state: amo SHARED-LOADED (the i1244 T2.2 shards + a queued Exclusive lane during all runs) — a correctness harness; every timing below is a wall-clock disclosure, never a perf claim.

## Receipts (4 runs, byte-identical across all of them)

| layer | mixer | capture digest | explained-norm (SPARSE ZY) | objective trace | support census | determinism |
|---|---|---|---|---|---|---|
| 0 | deltanet | `dbab1042…53fa4d2` | **0.1272** (rel resid 0.8728) | 0.9966 → 0.8728 @ 1500 it, ~85 s | 36/feature, union 1866/3584, features-per-neuron max 41 mean 1.93 | IDENTICAL ×4 |
| 3 | attention | `e33cd18d…abf20780f5d6` | **0.1008** (rel resid 0.8992) | 0.9997 → 0.8992 @ 1500 it, ~85 s | 36/feature, union 1761/3584, max 46 mean 2.04 | IDENTICAL ×4 |

- The objective is still descending at the 1500-iter cap (no plateau); the paper's 1.5–3.3k range is consistent — a longer-cap arm is a T2 rider, not a T0 claim.
- λ sensitivity (layer 0): λ 1e-2 → 1e-4 moves explained-norm 0.1272 → 0.1290 — **the ridge is NOT binding**.
- k sensitivity (layer 0): k=200 → 0.1768 (vs 0.1272 at k=100) — **the ceiling is k/samples, not λ**; the honest T0 posture records both.

## Mixer-type statistics cell (the T0 transfer question, measured — never assumed)

Per-layer stats over all 24 layers (mean/std/excess-kurtosis/near-zero-fraction/top-1%-energy-share), rollup per mixer on MEAN and MEDIAN (median = sorted[n/2], pinned order statistic):

| statistic | dn mean | attn mean | dn median | attn median | attn/dn median | diagnostic? |
|---|---|---|---|---|---|---|
| mean | −0.0002 | +0.0002 | −0.0000 | +0.0000 | — | negligible both (ratio not diagnostic) |
| std | 0.0894 | 0.1386 | 0.0655 | 0.1149 | 1.755 | yes, within guard |
| kurtosis | +3024.8 | +204.0 | +369.2 | +224.1 | 0.607 | yes, within guard (mean is outlier-driven) |
| near_zero | 0.0001 | 0.0001 | 0.0001 | 0.0001 | — | negligible both |
| top1pct_energy | 0.4518 | 0.4692 | 0.4562 | 0.4689 | 1.028 | yes, within guard |

**Verdict: NO material shift by median — every statistic within [0.5, 2.0]×.** The mean-kurtosis gap (+3025 vs +204) is SINGLE-OUTLIER-driven: layer 14 (DeltaNet) carries kurtosis 47538 > 10× its own mixer median — the cell names it explicitly instead of letting the mean read it. The DOMINANT axis of variation is DEPTH, not mixer type: std 0.04 → 0.39 and top-1% energy 0.31 → 0.72 grow with depth on BOTH mixers (massive-activation concentration deepening, consistent with the 1209 weight-audit outliers and the 1250 late-band readout strength). The paper's dense-SwiGLU statistics transfer to this hybrid checkpoint without a mixer-type correction at the fixture.

Defect caught in-pass (the cell's own stats, fixed before landing): the first kurtosis draft read −3.000 on every layer — the population form was missing the n factor (`m4·n/(m2²)`); the fixed readings (+53 … +47538) are heavy-tailed as the outlier literature expects. A ratio-of-negligibles false positive on the mean statistic also forced the negligibility clause (< 1% of mean std → "ratio not diagnostic") — flagging a ±0.0002 mean ratio as a "material shift" would have read noise.

## Determinism + coverage notes

- Determinism tripwire: full SNMF re-run per representative layer, Z/Y state digest asserted bit-equal — PASS ×8 runs total (2 layers × 4 full runs).
- The capture repeats bit-identically across runs (same corpus file, same window) — the capture digests above are stable artifact identities for T1 (input map + SNMFM1 artifact) and T2 (recursive hierarchy cell) to consume.
- Amo-side harvest disclosure: no prior census_screen harvest artifact existed on this box (the T0-prep record in the issue); the fresh-capture arm is the disclosed posture. The corpus + window + digests pin it for later re-use.

## Runbook

```bash
cargo run --release --features closed_form_write \
    --example snmf_census_cell -- \
    --gguf /Users/katopz/git/riir-train/data/Qwen3.5-0.8B-Base-Q8_0.gguf \
    --corpus /Users/katopz/git/riir-reflex/.raw/datasets/ag_news \
    --out .snmf_census_cell/
```
