# Bench 969 — Issue 1205: the Issue-022 T1.4 residual — the REAL parent-vs-quantized ΔS pair (english F16 vs the q8 storage artifact)

**Status:** RECORD — the residual riir-infer `7c5b5ef` deferred is CLOSED: both arms ran, the ΔS localization property fired (layer-concentrated, not uniform), and the total damage sits far under the pre-registered ε grid's smallest rung. The removal note's debt is paid in HISTORY.md (issue 1205's own row).

## Box state (provenance)

amo (M5 Ultra 96GB), AC Power, powermode `Automatic` (sudo-free detector: not run — no timing claims ride this record; the diagnostic is deterministic arithmetic over a fixed corpus, and every number below is a cosine-distance read, wall-clock-blind). Load at record time 7.00–11.21 (the m3 lane's rig jobs + the queued 1197 calibration boot were present) — IRRELEVANT to correctness: the capture driver is calibration-only, single-threaded forward, and the artifact records what it read, not when.

## Frozen inputs

- **Corpus:** `/tmp/audition_calib.txt` (from the 4090 `E:\git\riir-infer\.raw\twt\audition_calib.txt`, 289,910 bytes, 500 prompt lines) — corpus BLAKE3 `9f088e7b10f99f785007e47c1632a8f69beea331bf9e40e508b030e3c25afe10`.
  ⚠ **DISCLOSED divergence:** this is NOT the m3 GOAT corpus (`ea277368…`, never committed, exists only on m3) — the 4090's same-path file hashes differently (same size class, different content lineage; the T5.4 4090 session already recorded the corpus as m3-local and made its instrument box-portable instead). Every number here is a WITHIN-RUN controlled comparison (both arms read identical bytes); nothing is comparable to T5's m3 rows. Never pooled across corpora.
- **Parent:** `~/.cache/riir-reflex/laya/english/model.safetensors` (F16), weights BLAKE3 `5c701902d7fcc863…` (full digest in the artifact).
- **Quantized:** `~/.cache/riir-reflex/laya/english/derived/model.q8.safetensors` (the D2a storage artifact, converted this session by `laya-quant8`: 206 tensors / 126 quantized / 421,205,504 elements, 842,587,660 F16 → 447,707,500 Q8 bytes = 53.1%), weights BLAKE3 `8f78752701091606…` = the converter's own sidecar digest, re-verified at load by `resolve_weights_posture` (issue 1205's T1 instrument extension — a BLAKE3 mismatch would have refused the run loud).

## Instrument (issue 1205 T1)

`examples/twt_laya_profile` now resolves weights through `q8_artifact::resolve_weights_posture` (the env + sidecar-verify path every consumer shares) and stamps `weights_path` / `weights_posture` / `weights_blake3` beside `corpus_blake3`. Clippy `-D warnings` clean at the `twt_profile` posture. The Q8 payloads widen per-tensor through the identical `widen_q8_0` arithmetic the serving posture resolves with (`Encoder::from_map` → `Weight2D::from_weights` retains raw → per-consumer `wide()`), so the quantized arm IS the storage-tier numerics (Bench 0047's adopted posture), not an in-memory fake-quant.

## Protocol

`--checkpoint english --calib /tmp/audition_calib.txt --out <artifact>` (CPU posture, calibration-only), identical invocation both arms, env `LAYA_WEIGHTS_VARIANT=q8` on the quantized side only. Both arms folded **91,349 rows over 500 sequences, 0 skipped** — row-count identity asserted before any ΔS read.

## Results

- Parent arm: kill verdict SURVIVES (m(ε=0.05)=11 … m(1.2)=2, bar 16) — the phase structure replicates on this corpus.
- Quantized arm: kill verdict SURVIVES, identical block layout at every ε (the ε=0.1 partition read 0-1,2,3-4,5-6,7-11,12-13,14-18,19-27 on BOTH arms — the partition is quantization-stable at every rung).
- **ΔS map** (`twt::delta` semantics mirrored over the recorded `S_upper` triples; report JSON `.raw/twt/english_t14_delta_report.json`):
  - **max |ΔS| = 0.003652, RMS = 0.001191** over all 378 pairs.
  - **The localization property FIRED**: layer 18 concentrates incident |ΔS| mass 0.0603 — ~2× any neighbour (13–17 at ~0.030–0.033) and ~7× the per-pair mean. Layer 18 is a **full-attention** layer (global RoPE θ=160,000 vs the sliding θ=10,000).
  - **Per-type roll-up:** full-attention layers carry 0.0125 incident mass/layer vs sliding 0.0084 — a **1.48× concentration** on the global-θ layers.

## Reading (the localization property, not a verdict)

1. **The q8 storage tier barely moves the phase geometry**: whole-map RMS 0.0012 sits ~42× under the smallest pre-registered ε (0.05) — consistent with Bench 0046/0047's retention 0.956–1.000 and byte-identical decode. The tier's adoption is untouched; this record is the ΔS view of the same fact.
2. **The damage localizes on the global-attention layers** (1.48×/layer, peak at 18 = 2× neighbour): the wide-θ RoPE basis is where quantization noise costs the most phase similarity. This is a census datum for any future per-family quant budget (the Issue-1188 sensitivity-ranking lane's vocabulary: the `full_attention` family is where a tighter budget would buy the most), NOT a policy — no demotion, no re-quantization, the artifact stays as shipped.
3. Both partitions identical at every ε ⇒ the DP's block structure is quantization-robust: a collapsed emission planned off the F16 S matrix remains the SAME emission under the q8 weights.

## Artifacts

- `.raw/twt/english_t14_f16_profile.json` (parent S + provenance)
- `.raw/twt/english_t14_q8_profile.json` (quantized S + provenance)
- `.raw/twt/english_t14_delta_report.json` (the ΔS report: max/RMS/top-24/incident mass)

All gitignored (`.raw/`); the record carries the digests, the report JSON is regenerable from the two profile artifacts by the mirrored arithmetic (10 lines, formula quoted from `src/twt/delta.rs`).
