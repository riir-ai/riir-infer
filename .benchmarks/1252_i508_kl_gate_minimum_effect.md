# Bench 1252 — plan 508 T1 pre-GPU KL minimum-effect gate (the anchor-KL instrument's first consumer)

**Date:** 2026-11-21T14:1xZ (zed-989683dc@4090). Plan: [riir-train 508](../../../riir-train/.plans/508_onpolicy_own_answer_kd_quant_recovery.md) T1; the verdict's recovery-signal risk (session `cc0f0014`) measured BEFORE any GPU-h burn, exactly as pinned.

## The instrument (1242 T1's first consumer)

`src/bin/i508_kl_gate.rs` (+ the `anchor_kl` feature now pulling `deltanet_inference`): loads BOTH GGUFs through the dense qwen35 loaders, forwards the fixed 256-token panel (the chat_probe corpus head, greedy — deterministic), computes per-position forward-KL(anchor=teacher ‖ candidate=student) via `anchor_kl::anchor_kl_full` (f64 sequential folds), reports the trace-set mean against the **pre-registered `MIN_EFFECT_KL_NATS` = 5e-3 bar** (pinned in-source at the bin — the verdict was pre-registered, not post-hoc).

## Cells

| student | mean KL(anchor‖student) | max | verdict |
|---|---|---|---|
| **Q8_0** (the plan's pinned student) | **0.000481 nats** | 0.005201 | **BELOW-MINIMUM-EFFECT** — the recovery cell cannot show a difference at this quant |
| **Q4_K_M** (the re-aimed student, requantized from the same fp16 via llama-quantize, 5.50 BPW) | **0.057347 nats** | 0.409852 | **SIGNAL-PRESENT** — 119× the Q8_0 mean, measurable above the bar |

Wall: 67.9 s / 68.2 s per cell (CPU forwards, 255 scored positions); loads 1.1–1.7 s. RAM 17 GB free; no other compute consumers (rig QUIET).

## Verdicts recorded

1. **The plan's original student (Q8_0) is REFUTED as the T1 vehicle — near-lossless**: 0.000481 nats mean KL vs its own fp16 source says the quant's information loss is invisible at this panel; a KD correction would chase noise. This is the honest no-signal outcome the verdict's risk called out, now measured.
2. **T1 re-aims at Q4_K_M** (the pre-registered alternative — "re-quantized from this same fp16, for example Q4_K"), which **re-opens T2's gate registration** per the plan's own rule (changing the student): the T2 GOAT gate's floor comparator becomes the UNCORRECTED Q4_K_M, and the modelless arm (1243's own-traces calibration / scale search) runs on the same Q4_K_M panel.
3. The teacher loads as **24 layers** through the dense loader — the MTP head (`blk.24`) is structurally excluded (verdict duty (i) confirmed by the loader itself, no assert needed at this posture).

## Artifacts

- The bin: `src/bin/i508_kl_gate.rs` (required-features `anchor_kl` in the same commit — the P30 law).
- The staged students: `E:/tmp/i508/Qwen3.5-0.8B-Base-f16.gguf` (teacher, BLAKE3 `ac3d9e64…`), `Qwen3.5-0.8B-Base-Q4_K_M.gguf` (the re-aimed student — blake3 owed at the T1 build's long-run-law artifacts).
- The probe: `scripts/plan508_tensor_compare.py` (bench-1249-era full-tensor pin, 320/320 PASS).

## Posture

CPU-only forwards (the KL instrument needs logits, not training); the GPU-h budget is UNTOUCHED (0 of <2 GPU-h spent — the gate did its job). Box QUIET per `agent-mail rig status`.
