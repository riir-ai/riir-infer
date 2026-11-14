# Bench 1220 — plan 633 stage record: OpenThai2.0-ASR on Apple ANE vs the CPU SIMD lane (M5 Ultra) — roofline calibration, conversion parity, posture A/B, G5 base comparison

**Plan:** [633](../.plans/633_ane_audio_lane_openthai_asr_bench.md) (P0 + P1 complete, P2.1–P2.4 + P2.6 stage-1 complete; **pending:** tvspeech n=100 ANE pipeline cell — queued amo-exclusive job `50421460`, est 20 m, held behind a POISONED ×16 sibling job `4e769f80` — and the p8 palettized arm, harness authored [`harnesses/p633_p25.py`](1220_p633_ane_asr_stage_tables/harnesses/p633_p25.py), not run).
**Ran:** amo (M5 Ultra 30-core, 96 GB, macOS 27.2, AC plugged, powermode auto), quiet windows 2026-11-14 → 2026-11-15; sessions zed-cd62a5ed@amo + zed-8d516799@amo. Every quoted row carries its own `PROVENANCE:` line (AC, load ≤ 2.85/30, swap 4172.62 M) — the quiet-rig law held; the load-contaminated smoke rows in [`1220_p633_ane_asr_stage_tables/p633_cells.jsonl`](1220_p633_ane_asr_stage_tables/p633_cells.jsonl) are tagged in-file and are **never quoted below**. This record is the STAGE record: the two pending cells append rows to the plan and amend the G1/G2 verdicts when they land; every number here is provenance-quoted from the row files copied beside this record (the `.raw/asr/` lane is ephemeral — P4.4 teardown — so the rows AND the harness scripts are preserved in [`1220_p633_ane_asr_stage_tables/`](1220_p633_ane_asr_stage_tables/)).

## Posture

- **Model:** OpenThai2.0-ASR streaming (dictation export), fine-tune `.nemo` sha256 `6094a83e9729f5bf1ddd45f459594bfef4ab3e7caa5bc58263b959b0a585f815` (2,562,969,600 B, bias-1.0 bake; the RAW variant byte-size-identical, sha256 `61f39dea77508e2d7656b126b8fba7d6ae42d788f0c6fe0d9f35f5fab583f5cb` — the two digests distinguish the bias bake). G5 base: `nemotron-3.5-asr-streaming-0.6b.nemo` sha256 `210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74` (2,368,284,501 B). ONNX dictation export (iapp recipe, the P1.2 lane): `encoder.onnx.data` sha256 `a139201333ab89d5e965bc12abb1c534aa285d82e40051c2dda605365c74f18b` (2,454,405,120 B), `encoder.int8.onnx.data` sha256 `39176685a17087bc45a2a002cdc760951ccc4781693780a59db2b2a05ae964a5` (883,175,424 B, = repo LFS oid).
- **Fixtures (pre-registered, P0.3):** seeded (633) 100-clip slices; manifests blake3 `3ea9d3b04aa0e5f8cdad49e4b899aa002ec09144f3f22a8ab50e7e0997b306da` (fleurs-th) + `1e1ae2f83d97aa8125c2265516d7276ce97ba4ba931f8bc22b0b6d99ff3285db` (tvspeech) — re-verified byte-identical at this record (the current-era digests; the plan's older `350529dd…`/`6b7d11b7…` rows were the pre-2026-11-14-23:47 build era and are superseded). Licence: FLEURS-th test CC-BY-4.0; TVSpeech CC-BY (curated, ungated); GigaSpeech2-th FAILED the gate and is OUT of every public record (P0.3).
- **Env (the numpy-pin era law):** uv venv `.raw/asr-env` — NeMo 3.1.0 pinned at the card's commit `cf724ac337`, torch 2.15.0 CPU, ort 1.32.0, onnx 1.23.2, coremltools 9 (no ONNX frontend — the direct-torch route), **numpy pinned 2.4.6 → 1.26.4** (pandas/sklearn/scipy/pyarrow repinned with it) so the torch frontend converts clean. The pin MOVED the P1.2 piecewise splitter's segmentation: the old-era P1.2 CER rows (21.08/20.82) are superseded by the reproducible current-era **22.69/27.09** — manifests rebuild byte-identical, P1.1 (torch-side decode) reproduced EXACTLY (9.29/12.34) across the same window, decode code unchanged; the pin is the only env delta between eras. Deterministic within each era; **the current era is the record of reference**. The int8 cells were run ONLY in the old era and were never re-run (fp wins on RTFx — the lane was abandoned); their CER rows carry the era caveat.
- **Harnesses (preserved verbatim in [`harnesses/`](1220_p633_ane_asr_stage_tables/harnesses/)):** `p633_fetch.py` (digests above) · `p633_p1.py` (P1.2 ONNX cell + manifests + parity; card-exact greedy decode: max_symbols_per_step=2, blank 14988, blank bias −1.0/−2.5 <2 s, `split_at_pauses` verbatim) · `p633_p11.py` (P1.1 via iapp ChindaASR `decode_pass`, standard delta −1.0) · `p633_p21.py` (encoder conversion: two graphs first/step, prompt layer in-graph, fixed shapes [1,128,49]/[1,128,65]) · `p633_p23.py` (encoder-only + full-pipeline cells, G1/G2/G3 inputs) · `p633_p24.py` (decoder-step + joint×4 graphs, Posture A/B timing, `P633_NEMO_BASE=1` = the G5 base arm) · `p633_p25.py` (p8 palettize: `palettize_weights` kmeans nbits=8 + census/parity/p50 modes — authored, not run). The roofline harness is tracked in-tree: `examples/p633_p02.rs` (`AnePeaks::m5`).
- **Scoring chain proven BEFORE any own cell:** the card's published 560 ms hypotheses scored 8.47 % / 12.41 % on my rebuilt refs = the card's numbers exactly (normalization + CER chain validated; P1 stage-1).

## Gate verdicts at this stage (every gate explicit; pending cells named)

| gate | bar | verdict | evidence |
|---|---|---|---|
| **G1** CER delta ≤ +0.5 pt per family vs P1.1 ref (9.29 fleurs / 12.34 tvspeech) | ≤ +0.5 | **FAIL at the fp16 tier (fleurs)**: 10.75 − 9.29 = **+1.46** (n=100, TWO passes byte-identical CER; rows ×2 in `p633_p23_cells.jsonl`) | tvspeech n=100 leg PENDING (queued 50421460); the n=3 smoke (+17.03) is a smoke row, never a gate |
| **G2** ANE pipeline RTFx ≥ 6× the BEST CPU cell | ≥ 6.0× | **FAIL — 0.57×** the CPU lane (pipe 28.09–28.58× vs best CPU 49.23× fleurs; enc-only 36.6–36.8× → 0.74×) | the owner's "expect 6×" bar far off at this tier; honest negative recorded, never explained away |
| **G3** normalized exact-match ≥ 98 % | ≥ 98 % | **UNMEASURED this stage** — the parity gates measured ANE-vs-NeMo token identity, not transcript-level exact-match; the G3 input rides the pending tvspeech + p8 cells (`p633_p23.py` computes it) | explicit gap, not a pass |
| **G4** ANE placement (cost-weighted) | 100 % encoder+joint | **PASS** — encoder 100.0 %/100.0 % cost-weighted on BOTH graphs (placed-op share 97.78/97.79 % — the 36-op int-mask island carries ~zero cost weight); joint 100 %/100 % at every batch b1/b28/b56/b112; **dec_step 0 %** — the 2-layer one-step LSTM is CPU-resident (a new island class beside beshkenadze's int-mask one) | `p633_p24_census.json`; the SYNC `MLComputePlan.load_from_path(compute_units=CPU_AND_NE)` returns `MLComputePlanCost` (supersedes the async-None instrument finding) |
| **G5** FT not slower than same-recipe base at the same tier | FT ≤ base | **PASS at tiers ≥28** (b28 0.87×, b56 0.77×, b112 0.65×); **marginal +6.6 % MISS at the per-step posture** (0.4095 vs 0.3841 ms) — recorded both, never pooled | the vocab-delta intuition (bigger vocab slower) INVERTS at large F: base degrades 2.9× b1→b112 vs FT's 1.8× |
| **Parity (encoder)** | token identity | **GREEN — 100 % token identity ×3 FLEURS clips** (12–39 steps, real-buffer chunks, caches threaded end-to-end); ANE fp16 encoded frames max-abs ≤ 0.134 / rel-RMS ≤ 3.8 % vs torch fp32; encoded_len match everywhere; 0 tok-diverge steps | `p633_p21_parity.json` |
| **Parity (decoder+joint)** | token identity | **GREEN — 100 % token identity + exact_match ×3 clips** (2 FLEURS + 1 TVSpeech); drift g ≤ 0.007 (ref scale 0.27), logits ≤ 42 on a 190-scale, per-step argmax agreement 100 %; **max_symbols=None — the model cfg has NO cap, the "=2" was iapp's recipe** | `p633_p24_parity.json`; the cumulative-`y_sequence` comparison-artifact law recorded for any future parity lane |
| **P0.2 Posture A** (per-call flat ±15 %, ≥ runtime_ms) | hold/refute | **HOLDS**: pair p50 0.4095 ms = 1.215× `runtime_ms` 0.3371 (full-iter incl. host argmax 0.4176 — host +8 µs); flat across tiers by construction | `p633_p24_cells.jsonl` |
| **P0.2 Posture B** (≥3× at short tiers → ≤1.5× by b56) | hold/refute | **FALSIFIER FIRED — the working-set collapse REFUTED for this shape**: excess = b1 **0.78×** (below the roofline row — the anticipated AnePeaks calibration row, the bandwidth fit is documented conservative), b28 0.92×, b56 1.03×, b112 1.22× — < 1.5× at EVERY tier, the ≥3× collapse never appears; b112 = 1.8× b1 for 112× the frames (dispatch amortization) | FluidAudio's published tier collapse reclassified as a batching/amortization effect for this shape, per P0.2's own falsifier clause |
| **CPU findings** | — | int8 is **CPU-NEGATIVE** on M5 Ultra MLAS (16–18× vs fp's 28–32× — DynamicQuantizeLinear islands cost more than int8 saves; the liquidcodec "int8 ≈ neutral" does NOT transfer); **4 threads wins** (49.23/45.30 vs 30.49/30.37 @1thr; 30thr 48.55/43.39 flat-to-negative; CER thread-invariant) | `p633_cells.jsonl` current-era rows |

**Ungate law:** the G1/G2 FAIL demotes the fp16 full-pipeline ANE posture by absence of wiring — nothing in-tree consumes it (the whole lane is ephemeral `.raw/asr/`; HEAD `6d434ef4` carries zero tracked-code changes from this plan). No serving claim, no default flip, nothing to revert.

## The cells

### CPU lane (the G2 denominator) — current era, quiet, n=100, wall RTFx

| cell | posture | fleurs-th | tvspeech | CER fleurs | CER tvspeech |
|---|---|---|---|---|---|
| P1.1 NeMo fp32 (iapp recipe, batch=1) | 1 thr | **5.45** | **5.66** | 9.29 | 12.34 |
| P1.2 ONNX fp (piecewise dictation) | 1 thr | 30.49 | 30.37 | **22.69** | **27.09** |
| P1.2 ONNX fp | 4 thr | **49.23** ← best CPU | **45.30** ← best CPU | 22.69 | 27.09 |
| P1.2 ONNX fp | 30 thr | 48.55 | 43.39 | 22.69 | 27.09 |
| P1.2 ONNX int8 (OLD ERA — pre-numpy-pin, never re-run) | 1 thr | 16.03 | 17.76 | 22.76ᵉ | 21.79ᵉ |

ᵉ old-era CER; superseded-era rows 28.11/32.16 RTFx + 21.08/20.82 CER remain in `p633_cells.jsonl` for lineage. The SIMD lane is **7.8× / 5.7× the NeMo reference at 1 thread** — P1.2 IS the honest denominator. G1 reference rows: P1.1 9.29/12.34, re-verified under the current env.

### ANE fp16 arm (the measured rows so far)

| cell | family | n | enc RTFx | pipe RTFx | CER | Δ vs ref | G2 ratio |
|---|---|---|---|---|---|---|---|
| P2.3-enc | fleurs-th | 100 | 36.80 | — | — | — | 0.74× (enc-only) |
| P2.3-enc (pass 2 — byte-stable) | fleurs-th | 100 | 36.61 | — | — | — | 0.74× |
| P2.5-pipe | fleurs-th | 100 | 36.80 | 28.58 | 10.75 | +1.46 | 0.581× |
| P2.5-pipe (pass 2 — CER identical) | fleurs-th | 100 | 36.61 | **28.09** | 10.75 | **+1.46** | **0.571×** |
| P2.5-pipe | tvspeech | **PENDING** | — | — | — | — | — (queued 50421460) |

Attribution (fleurs): decode tail 21 % of wall (enc 32.6 s of 41.8–42.5 s); of a real 13.08 s clip: encode 257.1 ms (enc-only RTFx 50.9), decode tail 90.7 ms = 6.93 ms per audio-second over 201 joint calls; **the CPU LSTM island ≈ 30 ms of the 90.7 (a third of the tail), priced ~0.15 ms/call** (A-pair 0.4095 − joint-alone ~0.26). The named levers, in measured-value order: the p8 palettized arm (instrumented, `p633_p25.py`), the P2.4a manual gate-decomposed LSTM (parked — matmul+sigmoid+tanh+mul are ANE-legal primitives), the host-side cache/copy gap.

### P2.4 timing (quiet AC, joint graphs, p50 / excess vs the P0.2 roofline `runtime_ms`)

| posture | FT (fine-tune) | base (G5) | FT/base | P0.2 runtime_ms |
|---|---|---|---|---|
| A — pair calls (dec_step + joint_b1) | 0.4095 ms (1.215×) | 0.3841 ms (1.139×) | **1.066 (+6.6 %)** | 0.3371 |
| A — full iter (host argmax incl.) | 0.4176 ms | 0.3904 ms | 1.070 | — |
| B — joint b1 | 0.2618 ms (0.777×) | 0.2487 ms (0.738×) | 1.053 | 0.3371 |
| B — joint b28 | 0.3127 ms (0.919×) | 0.3583 ms (1.053×) | **0.873** | 0.3404 |
| B — joint b56 | 0.3630 ms (1.033×) | 0.4735 ms (1.347×) | **0.767** | 0.3514 |
| B — joint b112 | 0.4720 ms (1.216×) | 0.7278 ms (1.875×) | **0.649** | 0.3881 |

tail_support p95 = 150 (posture A) / 50 (posture B) — above the percentile-floor law's n ≤ 100/p99 trap; p95 rows quoted beside p50 everywhere.

### Placement census (G4, cost = MLComputePlanCost.weight)

| graph | n_ops | placed | ANE placed-op | cost-weighted ANE | CPU island |
|---|---|---|---|---|---|
| encoder first [1,128,49] | 3803 | 1621 | 97.78 % | **100.0 %** | 36 int-mask ops |
| encoder step [1,128,65] | 3817 | 1626 | 97.79 % | **100.0 %** | 36 |
| dec_step (2-layer 1-step LSTM, h/c [2,1,640]) | 42 | 13 | **0 %** | **0 %** | the WHOLE graph |
| joint b1 / b28 / b56 / b112 | 15 | 7 | 100 % | **100 %** | none |
| base model (G5) — identical signature | — | — | same | same | same classes |

## Reading

1. **The honest shape:** on M5 Ultra at this tier, the ANE fp16 pipeline does NOT beat the CPU SIMD lane — 28.1× vs 49.2× (0.57×), and CER +1.46 misses G1. The owner's 6× expectation is far off at this posture, and the record says so plainly. The conversion is CORRECT (100 % token identity twice over) and the placement is PERFECT where the graphs exist (100 % cost-weighted) — the gap is per-call economics: one encoder call per 0.8 s of audio (chunk 56) + one CPU LSTM step + one ANE joint call per emitted token, with host-side cache plumbing between.
2. **The FluidAudio working-set attribution is REFUTED for this shape, twice:** Posture B stays under 1.5× at every tier (the ≥3× collapse never appears), and the G5 base cell shows the SMALLER-working-set model degrading FASTER across tiers (2.9× vs 1.8× b1→b112) — the opposite of a working-set cliff. Dispatch amortization is the visible mechanism (b112 = 1.8× b1 for 112× the frames).
3. **The joint (19.2 MB fp16 vocab matmul) measured 0.26–0.47 ms/call against a 0.34–0.39 ms roofline** — the `AnePeaks::m5()` bandwidth fit is confirmed conservative (measured BELOW the modeled DRAM streaming at b1), the anticipated calibration row, recorded against the module not explained away.
4. **What would move the number:** batching the decode tail (Posture B's amortization applies to real pipelines only if the autoregressive loop can batch — it can't per-token; this is P2.7's named risk), the palettized arm (p8 — halves the vocab matmul's weight traffic), removing the CPU LSTM island (P2.4a, priced 0.15 ms/call ≈ 30 ms/13 s-clip ≈ a third of the tail). None of these is assumed to close 0.57× → 6×; the p8 cell is the next MEASUREMENT, not a promise.

## Consequences

1. **P4.1 (this record) stage-complete**; the pending cells (tvspeech n=100 pipeline, p8 palettized) append rows + amend G1/G2/G3 when they land — the plan stays the ledger.
2. **No promotion anywhere** (ungate law above); P2.7 (B1 fusion) stays conditional on P2.5's final shape; P3 (the Rust SIMD lane) requires Phases 1–2 to PROVE the lane — at 0.57× the lane is not proven; P3.1 stays unopened.
3. The CPU lane findings stand alone as the record's positive contribution: the card's ONNX dictation export at 4 threads is the fastest Thai ASR posture measured on this box (49.2×/45.3×), int8 is CPU-negative, and the CER-era law (numpy pin moves segmentation) is recorded for every future NeMo/ONNX cell.
4. P4.4 teardown note: `.raw/asr/` (28 G) dies AFTER the pending cells land; this record + `1220_p633_ane_asr_stage_tables/` (16 row/fact/log files + 7 harness scripts) are the reproducibility set — digests above re-fetch every model, the harnesses re-run every cell.

## Provenance

- Repo: riir-infer `develop` @ `6d434ef4` (the lane's own landings are docs-only; zero tracked-code changes — boundary-clean by construction, no BOUNDARY.md surface touched, no dep added).
- Row files (verbatim copies in [`1220_p633_ane_asr_stage_tables/`](1220_p633_ane_asr_stage_tables/)): `p633_cells.jsonl` (P1 era rows, contamination tags intact) · `p633_p23_cells.jsonl` (P2.3/P2.5 + PROVENANCE per row) · `p633_p24_cells.jsonl` + `p633_base_p24_cells.jsonl` (Posture A/B + G5) · `p633_p21_{census,parity,facts}.json` · `p633_p24_{census,parity,facts}.json` · `p633_base_p24_census.json` · `p633_p23_{full,tv}.log` · `p633_env.log` · the slice manifests' content digests `fleurs_th_slice100.json` / `tvspeech_slice100.json`.
- PROVENANCE lines quoted per cell (the distinct quiet-window states): `power=AC Power load=2.85 swap=4172.62M powermode=0(auto)` (P2.3 pass 1) · `load=2.45` (P2.3 pass 2 + P2.5 pass 2) · `load=2.23` (smoke rows) · `load=2.17` (P2.4 FT timing + P1.3) · `load=2.16` (per_audio_s) · `load=2.14` (G5 base timing). All AC, powermode auto, no Low Power Mode; GPU-class arm n/a (ANE lane); no exclusivity incident (amo quiet windows between sibling load spikes; the seal-localnet docker VM load ~12 window refused the direct tvspeech run — fail-closed preflight, the cell queued exclusive instead).
- Box state at record-writing: amo AC, load 13.3/30 (sibling rustc/cargo + the poisoned sibling job), 63 GB free — no timing claim is made in this record beyond the provenance-quoted cells above.
- Scoring-chain oracle: the card's own benchmark lane (`iapp/`), their published numbers reproduced on my refs before any own cell (8.47 % / 12.41 %).
