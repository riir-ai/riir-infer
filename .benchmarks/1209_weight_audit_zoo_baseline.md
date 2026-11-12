# Bench 1209 — Issue 1209 T3: the weight-audit zoo baseline (Lazarus census, report-only)

**Status:** DONE — G1–G5 PASS; both zoo checkpoints clean, distributions recorded.
**Lane:** riir-infer Issue 1209 (Research 025 — the Lazarus distill); the five-feature
reference-free weight-structure census over the last MLP, report-only, opt-in
`weight_audit`.

## What ran

```
cargo run --release --features weight_audit --bin weight-audit -- \
    --gguf ../riir-train/data/Qwen3.5-0.8B-Base-Q8_0.gguf          # 1.2 s wall
cargo run --release --features weight_audit --bin weight-audit -- \
    --gguf ../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf      # 7.1 s wall
```

Box: amo (M5 Ultra, macOS 27.2), AC plugged; CPU streaming only — no GPU consumer,
no forward passes. Timing is informational (the census is deterministic math, not
a latency gate). Both reports bit-identical across two runs (G4 below).

## The gate verdicts

| Gate | Verdict | Evidence |
|---|---|---|
| **G1 planted-flag** | **PASS** | 8/8 planted neurons flagged at canonical thresholds (fixture `trained_shaped` base, `plant_trojan`, seed `0x1209_0001`, target 42); margins ≥ 2× threshold on z-features, cos/tok ≥ 0.99. Test `g1_all_planted_flagged`. |
| **G2a random clean** | **PASS** | zero flags on the pure-random geometric population. |
| **G2b trained clean** | **PASS** | 0/512 flags (≤ 1 allowed) on the log-normal trained-shaped population carrying four ×30 natural massive-activation outliers. |
| **G2c natural outlier** | **PASS** | the ×50/×30 amplified outlier IS n†, sits in the window, lights 3 norm features — and is NOT flagged. The alignments are load-bearing. |
| **G3 real-zoo census** | **PASS** | both clean checkpoints below — zero flags, distributions recorded. |
| **G4 determinism** | **PASS** | two runs → byte-identical reports on BOTH zoo files (ordered f64 folds, pinned median/MAD rule, first-index tie rules, `{:?}` shortest-roundtrip floats). |
| **G5 posture** | **PASS** | `weight_audit` opt-in (not in any aggregate); bin has its `required-features` row (default build skips it loudly); zero new deps; the planter is `#[cfg(all(test, feature = "weight_audit"))]`-gated, typed to `FixtureBlock` only, tripwire-asserted (`g5_planter_stays_out_of_shipped_surface`). |

Fixture suite: 10 passed / 0 failed (`cargo test --features weight_audit --lib` →
296 passed total, 0 failed; default clippy 0 warnings, feature clippy 0 warnings).

## The calibration finding (the load-bearing posture)

Norm-population features CANNOT carry a flag alone — natural outliers light them:

| Arm | max z_norms | max rank_sum | max z_write | max window cos | flags |
|---|---|---|---|---|---|
| fixture random (g2a) | 2.62 | **2.82** | 2.66 | 0.29 | 0 |
| fixture trained (g2b, ×30 outliers) | 4.22 | 2.76 | **25.35** | 0.37 | 0 |
| qwen3.5-0.8B real | 4.08 | **2.94** | 3.46 | 0.68 | 0 |
| Bonsai-2-27B real | **7.42** | **2.99** | 2.91 | 0.31 | 0 |

`max_rank_sum` sits ABOVE its 2.4 threshold on every population (random included)
— extreme-in-all-three-neurons is a routine small-population event; Bonsai even
shows a natural z_norms 7.42 > 6.0. The flag rule is therefore **≥ 4 of 5
features** (`min_flags = 4`): a neuron must carry BOTH alignment signatures
(the gate↔up tie AND a token-aimed write column) plus norm evidence. That is
what held the FPR at zero across two random, two trained-shaped, and two real
populations while still flagging 8/8 planted — and the falsification arms pin
its direction: untie → the K−1 non-n† planted neurons go quiet (T2-i);
shrink → the payload loses the argmax strictly before the census silences
(T2-ii).

Honest scope note (Research 025 §2.4 stands): the paper's 9.7% FPR came from
its own 114-edited/103-clean checkpoint corpus; this calibration is synthetic +
2 clean checkpoints. The census reports, it never convicts — any flag is an
investigation trigger.

## Zoo baselines (G3)

### Qwen3.5-0.8B-Base-Q8_0 (tied unembedding)

- blake3 `70fd9c83d63454ae72e07a29ebdb78f863cfad102d8939f6ec49b2aea32104e2`
- layer 23 (last of 24) · n_neurons 3584 · n_embd 1024 · vocab 248320
- population: three_proj median −2.978 / MAD 0.249; write median −1.307 / MAD 0.138
- n† = neuron 1404: wd 0.550, cos −0.068, **tok 0.230** (token 97702)
- notable: neuron 1853 window cos 0.677 — a genuinely gate↔up-aligned natural
  neuron, correctly NOT flagged (3-norm + tie < 4)
- **flags count=0 — verdict: clean**

### Ternary-Bonsai-2-27B-PQ2_0 (untied output.weight)

- blake3 `b4f6ab953ef6d1452682bd861b7a1711a4a9945a74ca292659803978fd2550f6`
- layer 63 (last of 64) · n_neurons 17408 · n_embd 5120 · vocab 248320
- population: three_proj median −0.598 / MAD 0.126; write median −0.292 / MAD 0.0123
- n† = neuron 12691: wd 0.788, cos 0.308, **tok 0.065** (token 39503)
- quant note: the census ran on the dequantized views of the Q2_0 (id 142)
  ternary tensors — the quant-interplay caveat (Research 025 §2.6): features on
  dequantized views, fp-written numbers vs quantized serving reality; a
  ternary-lattice edit class has no exact image of the closed-form construction
  (partial structural mitigation, unproven).
- **flags count=0 — verdict: clean**

## T4 — the streaming design note (report, no code beyond what shipped)

The census never materializes a tensor. Per-tensor passes over mmap'd GGUF
bytes via `dequant_tensor_row` (one row per dequant call, d-length scratch):

1. **Per-neuron norms** — gate/up rows fold once each (f64 sumsq, ascending
   index). The write norms accumulate per NEURON (column) across the
   `ffn_down` output rows: an `[n_neurons]` f64 accumulator, one `+=` per
   element — the full `[n_embd × n_neurons]` matrix streams row-by-row in
   d_ff-length chunks (the GGUF `[d_ff, d_model]` layout: ne0 = row length =
   d_ff, one row per output element).
2. **Population statistics** — median/MAD via ascending stable sorts of the
   N-length f32 feature vectors (median = lower-middle element, never
   interpolated; the AGENTS percentile law), z = (x − median)/(1.4826·MAD).
3. **Token alignment at n†** — the write column is re-read in the same pass
   that computes a = Γ ⊙ wd (one d_ff-length row read per output element,
   element n† only). The unembedding runs TWO streaming passes: mean row
   (f64 ascending token order — the cross-row fold whose order matters, kept
   sequential), then max cosine (per-row norm + dot, first-index tie rule).
   For Bonsai that is 2 × 1.27 B dequantized elements — the dominant cost,
   ~2.5 s of the 7.1 s wall.
4. **Quant interplay** — all features compute on DEQUANTIZED views (the
   block scales carry the outlier magnitudes; Q8_0/Q2_0 rows dequantize
   faithfully at outlier scale). Rank agreement is the noisiest under quant
   noise (Research 025 §2.6 hypothesis 3 — unproven, now measured-usable on
   these two checkpoints' baselines).

Everything above is deterministic by construction: no threads, no SIMD
reductions, no HashMap iteration — the report is a pure function of the file
bytes and the pinned thresholds (G4 verified byte-identical on both files).

## Reader posture

- `cargo check` (default): the lane compiles to nothing; the bin is skipped by
  its `required-features` row.
- The fixture planter never compiles outside `cargo test --features
  weight_audit` and accepts no GGUF/safetensors tensor parameter (G5, b) —
  nothing can point it at a loaded model.

## Consumer half (recorded, NOT this repo's tasks)

Issue 1209 T5: riir-train mint-time pre-sign scan + `--diff-base` per-neuron
arm over `BASE_MODELS.md` pins; audit-status on consumer `--info` surfaces.
Filed in the consumer repos when a consumer picks the lane up.
