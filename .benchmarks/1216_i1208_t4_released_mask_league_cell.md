# Bench 1216 — Issue 1208 T4: the released-mask league cell (the GOAT cell, zero training)

Date: 2026-11-14, bmo (the quiet runner), exclusive lane, job `01a1d963-fc04`
(queued 09:28Z, walks done by 09:40Z — the G1/G3 walks were far cheaper than
the 3–4 h budget; the 4.7 GB download dominated the wall).
PROVENANCE (from the cell, verbatim): `load1=1.36 cores=30 epoch=1794649022
power=AC(assumed)` at start; `load1=5.12 cores=30 epoch=1794649196` at end.

## The artifact pair

| piece | value |
|---|---|
| mask | upstream `Jacklu0831/deletion-tuning-masks` → `grpo/qwen2.5-7b-simplerl-zoo` (GRPO-trained, SimpleRL-Zoo data) |
| their format | packbits_zstd, total_params 1,901,068,288, zero_params 3,161,923 (0.1663% sparsity) |
| ingest | `i1208_hf_ingest` — census verified EXACT both directions (28 sections × 67,895,296 gate weights = 1,901,068,288 bits; 3,161,923 deleted) |
| our artifact | 6,936,022 B (6.94 MB) — theirs 7.48 MB (our RLE+statistics-friendly layout compresses slightly better) |
| bound artifact blake3 | `280c99c245a0faa1a7346c49d9763a1b4d6e7e92245023d66080e6e08bc8ce8d` (the UNBOUND first-pass artifact was `a4384a89…d57f91b6`) |
| base bind | `cbc27156dd5b7190e5b33ba670103561869852325a0bb9f8fdfc40160371b3e7` = BLAKE3 over the 28 loaded gate tensors (1.9B f32 weights, row-major) — `decode(_, Some(base), _)` enforced it |
| base model | `kaetemi/Qwen2.5-7B-Q4_K_M-GGUF` — the TRUE base (`base_model:Qwen/Qwen2.5-7B`), 4,683,073,856 B, sha256 `41cffbf2…36376` verified pre-run, arch `qwen2` |

**Fidelity caveat (disclosed, not silent):** the mask was trained against the
base's bf16 weights; this cell runs on Q4_K_M dequantized to f32. Every bar
below compares POSTURES ON THE SAME WEIGHTS, so the bars are
quant-independent — G1's identity, G3's parity, and both G2 rows read the
same at any quantization. What the caveat touches is only how the deleted
positions' zeros interact with the surrounding quantized magnitudes (the
task-quality axis, which this cell does not measure — the paper's own eval
lane is their vLLM harness on bf16; a task-quality read is a consumer-side
question, never this cell's bar).

## G1 correctness — walk identity between postures: PASS

The 32-token decode walk (fixed synthetic ids, no tokenizer on lane), FNV
over the exact f32 logit bits + the argmax token stream:

| arm | argmax_fnv | logits_fnv |
|---|---|---|
| base | `e8caa29d76df01c5` | `2e2ac02fc9b589f7` |
| apply-at-load | `8fe685f12a617687` | `55b289fb14d74b81` |
| independent reference (test-side bit walk) | `8fe685f12a617687` | `55b289fb14d74b81` |
| all-keep identity mask | == base | == base |

- **apply ≡ reference: PASS** — bit-identical digests across all
  32×152k logits. The MSB→LSB transpose, the section offsets, and the
  apply path agree exactly; any off-by-one anywhere would have moved
  a digest.
- **identity pin: PASS** — the all-keep mask over the same sections
  walks == base (application never corrupts kept weights).
- **the mask is LIVE, not vacuous:** 3,161,923 weights zeroed across 28
  sections; the masked walk differs from base with **8/32 argmax flips**
  (deletion DOES move predictions at 0.166% sparsity — the task-quality
  axis is real; the paper's claim is the flips net out to a BETTER model,
  which is their eval's business).

(The issue's "materialized GGUF" posture reads as physically-zero weights
in a file; on the qwen2 lane the matmul sees 0.0 f32 either way, so the
independent-zeroing reference tests the same property without a 4.7 GB
rewrite — the cell's doc comment carries this disclosure.)

## G3 perf — decode parity: PASS (−0.159%)

Interleaved A/B rounds (32 tokens each, 4 rounds, warm-up both arms):

| round | base tok/s | masked tok/s | delta |
|---|---|---|---|
| 0 | 5.39 | 5.40 | +0.13% |
| 1 | 5.51 | 5.50 | −0.23% |
| 2 | 5.50 | 5.49 | −0.16% |
| 3 | 5.51 | 5.49 | −0.37% |

Mean **−0.159%** (bar ≤1%): PASS. As designed, the apply-at-load arm
shares base's FLOPs (zeroed weights multiply identically) — the parity is
structural, and the measurement confirms no accidental work joined the
masked path.

## G2 gain rows — measured

- **(a) artifact bytes:** 6,936,022 B vs 4,683,073,856 B = **675.2×**
  smaller (in the paper's hundreds-to-thousands× bracket, measured on our
  stack). Against the bf16 original (15.2 GB) it would be ~2,190×; the
  honest row is against the file we actually deploy from.
- **(b) swap wall:** the mask posture (decode+bind inside a load, then
  clone 1.805 s + apply 0.827 s = **2.632 s** total) vs a full GGUF
  reload of the same model (**10.969 s** warm) = **4.2× faster**. The
  apply itself is 0.827 s over 3,161,923 weights ≈ 0.26 ns/weight — the
  same bandwidth-bound zeroing the T2 cell measured (0.38 ns/weight at
  0.8B; faster here on bmo's memory). The clone (1.805 s) is the
  apply-at-load posture's fixed cost; the overlay posture skips it
  entirely (mount = masked copies built once). **Falsifier check: the
  swap win is 4.2×, far outside noise (base spread across rounds
  <1%) — the G2 gain claim SURVIVES the issue's falsifier.**

## Posture note

The issue's posture decision stays owner-gated per its own law. What this
cell adds to that decision: the substrate mechanics are GOAT-clean (G1
bit-exact, G3 free, G2(a) 675×), and the remaining question is purely
ECONOMIC — does the 4.2×/675× artifact advantage beat "just ship a second
GGUF" often enough to carry the lane. The overlay posture's numbers (no
clone, no apply) are already on record from bench 1208.

## Reproduction

```
# on any arm64 box with the staged fixture set:
./i1208_hf_ingest mask_dir mask.rimsk --base-gguf qwen2.5-7b-q4_k_m-imat.gguf
./i1208_t4_league_cell qwen2.5-7b-q4_k_m-imat.gguf mask.rimsk 32 4
```

riir-infer `bee2658` (bins) + the bind digest above pin the run. The T5
cross-box Wanda falsifier rides the same exclusive lane behind this job
(`01a1d96d`), its digests compared against bench 1208's amo record.
