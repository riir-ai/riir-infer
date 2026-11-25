# Bench 1264 — Issue 1250 T2: the J-share verbalizability cell (Qwen3.5-0.8B-Base-Q8_0)

**Date:** 2026-11-25/26 (amo)
**Lane:** riir-infer Issue 1250 T2 (katgpt-rs Research 693 §1.2, the Anthropic Fellows pair distill)
**Lease:** amo-i1250-t2 (claimed `247205d2`, folded at landing)

## What landed

- `src/readout_jshare.rs` (feature `readout_screen`, same opt-in as T1): the J-share pass — per unit, the Lawson–Hanson NNLS fit of the lens-normed unit onto its OWN T1 top-10 token rows (read from the RDMAP1 artifact, never re-derived), `J = ‖Eα*‖²/‖u‖²` clamped [0,1]; per-layer Rademacher ±1 control directions (SplitMix64 keyed on the file digest — transcendental-free, bit-reproducible) run through the SAME pipeline (lens norm → top-10 scan → NNLS), the pinned bars High p≥0.95 / Low p<0.50 (ties count ≤), the RDMAP2 v2 artifact (rows carry bucket + J bits + the support; the trailer carries the control count and each layer's ascending control distribution; the control seed base IS the header's file-digest field) + the per-band rollup (bucket census + J median/MAD).
- NNLS determinism + termination: first-index tie rules everywhere; Tikhonov damping δ = 1e-12 × max-diagonal in the passive solve (the atom sets are frequently rank-deficient — 10 atoms can exceed the unit rank); a zero-step eviction (λ = 0) BANS the atom permanently (its residual gradient is positive only through suppression that vanishes on entry — re-admitting cycles). Two livelock classes found and fixed by the fixture before the real cell: the singular-Gram churn (fixture: 8 dims vs 10 atoms) and the suppression cycle.
- `src/bin/readout-jshare.rs` (`--self-test` canary; `--gguf --artifact --out` real posture; digest-mismatch refusal BEFORE any compute), required-features row in the same commit (P30).
- Fixture gates 5 new (14 total in `readout_screen_fixture::run_all`): NNLS exactness (cone J=1 / orthogonal J=0 / the textbook suppression case), degeneracy + ties, control determinism + `top_k_tokens` = brute force, bucket edges (pinned bars, one `bucket_of` classification site), and the e2e (v1 → pass → v2 byte-stability + shape pin + engineered-unit J=1 + digest-mismatch refusal + report determinism).

## Pins (pre-registered before the real cell ran)

- Unit posture: the T1 lens unit `lens_norm(col, γ, ε)` (the same vector whose lens logits ranked the atoms). Atoms: the raw unembedding rows of the unit's own T1 top-10 (+w for Plus, −w for Minus — the sign trick's convention). ≤25-token cap: the source's cap; this lane's support is the pinned TOP_K=10 ≤ 25.
- J formula: squared-norm fraction `‖Eα*‖²/‖u‖²` (scale-invariant; α absorbs per-atom scale).
- Controls: 64 per layer, Rademacher ±1 from SplitMix64(file-digest, layer, k), full same-pipeline construction.
- Bars: High p ≥ 0.95, Low p < 0.50, Middle between (ties ≤). The source's 19–29%-high census is provenance, never a bar.

## The cell (amo, quiet-class box; load ~9 on 30 cores; AC)

- T1 regeneration on amo: 434.7 s, **artifact digest `c508d9170875cd60…` — BYTE-IDENTICAL to the m3 pin** (Bench 1262) — the T1 determinism claim now proven cross-box.
- T2 pass: **195.2 s** for 172,032 units (24 layers × 3584 × 2 signs) + 1,536 controls. RDMAP2: 15,661,135 bytes, blake3 `0411d383353c7a806b8409503378c025caf7f0ff2946ca28cec435054071c726`. Artifact at `/tmp/i1250_jshare_map.rdmap2` — NOT committed (binary; the report + digest pin are the record, the RDMAP1 not-committed precedent).

## Results

| band | units | High | Middle | Low | J median | J MAD |
|---|---|---|---|---|---|---|
| Early | 57,344 | 2,757 | 27,408 | 27,179 | 0.057 | 0.008 |
| Middle | 57,344 | 6,021 | 26,197 | 25,126 | 0.057 | 0.009 |
| Late | 57,344 | 20,504 | 18,800 | 18,040 | 0.066 | 0.018 |
| **all** | **172,032** | **29,282 (17.0%)** | **72,405 (42.1%)** | **70,345 (40.9%)** | — | — |

- **High-verbalizability share 17.0%** — just under the source's 19–29% provenance range (their bars/units differ; ours are the pinned percentile-vs-Rademacher-controls). Not a match claim — a corroboration-class read.
- **The late-band gradient is the strong finding**: 20,504 High units in Late vs 2,757 in Early (7.4×) — vocabulary-verbalizable neurons concentrate late, consistent with T1's late-band legibility finding and with the geometric read that early layers compute more distributed/non-vocabulary features.
- **Controls are tight**: Rademacher J median 0.056, max 0.113 across all bands — a random ±1 direction's top-10 lens rows explain ~5.6% of it; the unit population median (0.057–0.066) sits barely above, i.e. MOST units are not much more verbalizable than random, and the verbalizability signal lives in the High tail.
- **T3's ≥10% both-signs gate read**: the artifact now carries per-sign buckets; the T3 census (whether ≥10% of screened units have both signs reading distinct high-bucket concepts) can be answered from THIS artifact without a new pass — that is T3's own run (unclaimed, next lane).

## Verdict

T2 PASS as a lane row: the pass is deterministic (fixture-pinned, cross-run byte-stable), the real cell ran end-to-end on the real checkpoint, and the census numbers are sane against both the controls and the source's provenance range. T4's ordering-prior GOAT (screen-ordered sweep efficiency) now has its bucket field — still gated on the 1241 census sweep (in flight on bmo).
