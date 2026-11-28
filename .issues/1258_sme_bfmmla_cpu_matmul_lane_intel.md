# Issue 1258: SME/BFMMLA CPU-matmul lane intel — aarch64 instruction-path observation for any future CPU bf16 GEMM

**Claimed:** zed-5426a3b4@amo, 1795872400, expires 2026-12-01T01:00:00+00:00

**Status:** Open — observation-only lane intel, filed 2026-11-28 from the riir-refine hunch eg2cpu distill (Research 350, riir-refine Plan 311; delegated-verdict REVISE→DO round 1, session `e9e56fde-2068-46ae-bec7-55011c7d2451`). **No current consumer**: the codebase grep at filing time found ZERO bf16 CPU GEMM in riir-infer — every CPU matmul surface is f32 or dequant-to-f32. This note applies WHEN a CPU bf16 GEMM lands (a lane-intel pointer, never a serving-path claim, never a code change at filing).

## The two measured facts (source-measured, NOT our bar)

Provenance: undrgrnd-labs/hunch @ `22cf73010b0d` (MIT OR Apache-2.0), distilled as riir-refine Batch 429 / Research 350. Both figures are **hunch's own table on its own hardware** (Apple M5-class aarch64, their measurement posture) — never quoted as ours. Reproducing them on a quiet amo/bmo window is the one checkbox below, cross-referenced to riir-refine Plan 311 T3b so the measurement is not done twice.

1. **BFMMLA runs at ~half the fp32 FMLA rate** — ~2.1 G vs ~4.2 G bf16/fp32 matmul ops/s in their table. Regime: aarch64 with FEAT_BF16/BFMMLA, SME absent-or-off, encoder-shaped GEMMs. The hazard: a bf16 GEMM compiled onto BFMMLA without checking the alternative path loses ~half its throughput ceiling for free.
2. **SME takes the matmuls** — when FEAT_SME is present and enabled, the SME FMOPA path replaces BFMMLA for matmul workloads. The replacement, not a second hazard: on M5-class silicon the first question for a new bf16 GEMM is which path the runtime actually selects.

**Regime conditions (both directions):** aarch64 + FEAT_BF16 with SME absent-or-off = the hazard posture (BFMMLA half-rate); SME present = the replacement path (FMOPA). Neither applies to the Metal GPU lanes (BFMMLA/FMOPA are CPU instructions — a Metal kernel never issues either; the m5ultra tensor-stack prefill campaign and `ffn_gate_up` decode lever are GPU-side and are NOT consumers of this note).

## The riir-infer CPU matmul census at filing time (grep, 2026-11-28)

- `src/quant/svd_lbit.rs` — `gemm_nn`/`gemm_tn`: plain **f32** CPU GEMM.
- `katgpt_core::simd` (consumed via the `src/simd/` re-export; kernels live upstream since Plan 008 Step 7) — NEON/AVX2/WASM-SIMD128/scalar **f32** dot/sum/scale/exp/outer-product/matvec + ternary kernels. No bf16 kernel in the family.
- `src/transformer/ternary.rs` — ternary group matvec wrapper over the katgpt-rs Issue-578 NEON+scalar kernel (i8-class, not bf16).
- `src/quant/q4k.rs` / `src/quant/q6k.rs` — quant GEMV, dequant-to-f32-then-dot.
- bf16 touches are wire/load boundaries only: `src/multi_host.rs` (f32/bf16 LE pipe wire format), safetensors/gguf loaders (`bf16_to_f32` dequant at load).

`BFMMLA`, `FMOPA`, `FEAT_SME`, and `target_feature = "sme"` return ZERO hits in `src/` + `crates/` — no runtime detection, no per-target compile posture, nothing to retrofit today.

- [ ] **Qualify which riir-infer aarch64 CPU matmul paths take BFMMLA vs FMLA vs SME on M5-class silicon; measure only on a QUIET rig (the sequential-AB law — never a contended amo window).** Runs only if/when a CPU bf16 GEMM surface exists (or as a standalone census of what the compiler emits for the existing f32 GEMM family under `+bf16`/`+sme` target features); cross-referenced to riir-refine Plan 311 T3b (the committed f2 fixture carries the BFMMLA-vs-fp32 upgrade arm — one measurement session can discharge both, two boxes' records kept separate). Falsifier discipline: if the halves-rate class does NOT reproduce on our box, record the box split as regime content — never fold a non-reproduction into the source number.

## Routing

Filed unclaimed (observation-only; the `- [ ]` above is the only actionable unit). Mailed to the riir-infer lane at filing time; fleet issue-count + the boss's hourly re-fire are the backstop. Observation-only: **no code change, no benchmark claim** at filing — any measurement inherits the quiet-rig law above.
