# Issue 1206: grafts T2b — the real-activation cell (qwen35-hybrid 0.8B dense host)

**Status:** OPEN — T1+T2 code LANDED (the `grafts_cell` feature + `HiddenCapture` on the dense hybrid body + `src/grafts_cell.rs` + the `grafts_cell` bin; clippy -D warnings clean at grafts_cell/tests and default), run A (baseline measurement) in flight. **Design pivots recorded 2026-11-12 (both pre-registered BEFORE graft construction, both measured in run-A probes):** (1) the answer surface is CONTEXT-DERIVED — the base model's top-1 after `=` is the bare space token (logit ~21.3 vs the answer digit ~16.3), so the map derives from encoding reference-prompt+digit in context, and the readout is a GREEDY CHAIN (first digit token within 3 continuation steps wins; BIAS_DELTA 8.0 pinned from the measured gap — a smaller delta could never flip a base continuation); (2) ThreeOpAdd10 operands are TWO-DIGIT (10..99, 1000 seeded-distinct instances) — with single digits the model computes the raw sum and emits it multi-token (run-A gen probe: `7+3+1 mod 10 =` → `" 11\n11 mod 10 = 1…"`); 2-digit operands force the pattern-guessing posture the graft corrects (the paper's shape: the model guesses, the internal state knows, the graft reads the state). The forward was COHERENCE-PROVEN first (greedy gen: `"The capital of France is" → " Paris. The capital of Belgium is Brussels…"` — the dense-hybrid CPU path is sound before any accuracy row trusted it).

**Claimed:** amo-i988-t2b@amo, 1794433684, expires 2026-11-14T12:00:00Z

## Summary

The REAL cell required for any G1/GOAT accuracy claim on the `grafts` primitive (katgpt-rs Research 658 / issue 988): real hidden states + real logits from a real frozen model, closed-form fitted probes + a mined decision table + computed logit-bias writes, evaluated against the model's own baseline on held-out task instances. Synthetic-only fixtures (katgpt-rs T2a) make digit readability true by construction — this cell is the one that can actually fail.

## Host posture (pinned before graft construction)

- **Checkpoint:** `../riir-train/data/Qwen3.5-0.8B-Base-Q8_0.gguf` — the sink-audit census precedent (katgpt-rs issue 981) already ran this exact file through `load_qwen_deltanet_weights_gguf` + `forward_qwen_deltanet` (dense hybrid body). BLAKE3 computed at cell start and pinned in the report. (SmolLM2-135M was the issue-988 sketch; the paper's class is "small dense" — this 0.8B dense hybrid is the supported in-repo checkpoint: gemma-2-2b is RETIRED (owner law, no new cells), Bonsai-2-27B is the league model and 30× heavier per forward for a cell that wants hundreds of forwards.)
- **Body:** dense hybrid (`QwenDeltaNetWeights`, `HybridCache`, per-token `forward_qwen_deltanet`). The capture hook lands in the dense body only — the ternary body keeps its byte-identical path (the cell never touches it).
- **Tokenizer:** `BpeTokenizer::from_gguf` (the sink_audit dense path).
- **Feature:** `grafts_cell = ["katgpt-core/grafts"]` (opt-in; default build byte-identical — the capture field is cfg-gated out entirely, the Plan-287 G3 lesson).

## Task + split law (fixed BEFORE any graft is built)

- **Task family:** single-answer-token modular arithmetic, digit answers 0..9 (every answer exactly one BPE token — the harness refuses a multi-token answer loud). Candidates, measured for BASELINE HEADROOM before the task is pinned (the pre-registration step — the graft needs a model that actually errs to have anything to graft):
  - `three_op_add10`: `(a+b+c) mod 10`, a,b,c ∈ 0..9, 1000 instances — the default (instance-rich: unique pairs beyond probe memorization).
  - `two_op_mul10`: `(a*b) mod 10`, 100 pairs — fallback if three-op baseline leaves no headroom.
- **Prompt format:** `"{a}+{b}+{c} mod 10 ="` (and the `Q:/A:` wrapper as the alternate) — both baselines measured, better headroom pinned. Probe position: the LAST prompt token (hidden states after it is processed; the answer is the model's next-token choice there).
- **Splits (pinned seed, blake3-pinned manifest):** solved 400 / dev 200 / sealed 400 over a seeded Fisher–Yates shuffle of the instance space. The sealed split is touched ONLY at final eval; the dev split carries the disclosed tuning rows (armed rate, per-split accuracy); λ and bias delta are pinned a priori, never sealed-tuned.

## Probes / table / switch / writes (the katgpt-core primitive, consumed as-is)

- **Probe layers:** 3 mid/late layers at fractions {0.5, 0.65, 0.8}·n_layer (the paper's joined-14/20/21 pattern, generalized as fractions so the pin survives any checkpoint; exact indices printed + pinned in the report).
- **Reads:** one `GraftRead` per probe layer (segments = that layer, `fit_read` targets = the answer value as f32, domain = 10, ridge λ = 0.1 pinned a priori, with-bias). Quantized reads = nearest-int round of the probe score, clamped [0, 9] — the katgpt-core round-half-away law.
- **Table:** `mined_table` over the solved split's (read-triple → answer) examples — k ≤ 3 pinned, `Undecidable` refused loud, minimal-deciding-set reuse per the primitive's laws.
- **Switch:** `fit_switch` on solved-split labels {base-argmax wrong = 1}, threshold 0.5 — the conditional arm fires only where the model errs (the paper's conditional-graft shape). The UNCONDITIONAL arm (empty switches) also runs.
- **Writes:** one `GraftWrite` on cell 0: every decided value v biases digit token v with bias delta `+1.0` (pinned a priori; a sweep would tune on the answer distribution and is refused).

## Arms (sealed split, full-vocab argmax decoded to digit correctness)

1. **baseline** — clean logits argmax.
2. **graft-unconditional** — apply with empty switches; biases + argmax.
3. **graft-conditional** — apply with the fitted switch.
4. **scramble(seed)** — `Graft::scrambled(0x5EED_9880)` unconditional: the at/below-baseline control; wrong-by-construction wiring must NOT beat baseline (the T3 arm on real activations).

Report JSON (BLAKE3 over the canonical payload, the sink_audit G1 law): per-arm accuracy + decided counts + armed rate + full-vocab vs digit-restricted rows, checkpoint BLAKE3, split manifest digest, pinned layer indices, λ/threshold/delta, fit R²-adjacent diagnostics (probe score-vs-answer correlation on solved), scramble determinism check.

## Tasks

- [x] **T1** — feature `grafts_cell` (implies `katgpt-core/grafts` + `deltanet_inference`) + the cfg-gated `HiddenCapture` on `HybridForwardScratch` (dense body hook at the end-of-layer residual, the sink_audit Option pattern; flag-off = the field compiles out + the branch is one None check) + `src/grafts_cell.rs` (capture struct, task/split generation with `quant::kvq_harness::SplitMix64`, prompt encode, the in-context digit-token map with the single-token refusal). LANDED 2026-11-12.
- [x] **T2** — `src/bin/grafts_cell.rs` ([[bin]] + required-features): load → task baselines (three formats, candidate tasks) → task pin → splits pin → probes fit → table mine → switch fit → the four arms on sealed → report JSON + blake3. Determinism: same inputs byte-identical report (the G1 re-run). Additions beyond the original sketch: the whole run executes inside ONE `rayon::scope` (per-call scope entry measured at 2 latch-syscall classes/token — the first run-A attempt spent 105 min sys on ~84k calls), the per-sequence cache+scratch are RESET in place (Issue 838's lane — a fresh pair per sequence spent its life in madvise), and the coherent-forward probe gates any accuracy row. LANDED 2026-11-12.
- [x] **T3** — the pick was pre-registered (2026-11-12, before any graft construction): task ThreeOpAdd10 × format FewShot, baseline 0.0960 on the full 1000-instance deciding combo; the G1 bar: graft ≥ baseline + 20pp on sealed. **Run B LANDED — Bench 971** (`.benchmarks/971_grafts_t2b_real_activation_cell.md`, report payload blake3 `fac21420…`): **sealed G1 PASS — graft 0.3925 vs baseline 0.1250 (+26.75pp ≥ +20pp)**, armed 29.7% / decided 59.3% (the mined-triple coverage), scramble 0.3975 (the control ran at PARITY — disclosed: at a chance baseline any consistent digit bias beats no bias; the decisive scramble needs a mid-strength baseline, recorded as a future-cell design input). The road there caught three real bugs (each by the cell's own loud laws, each pre-sealed): the u8 answer overflow (`98+98+80` wrapped — every sum ≥ 256 instance stored a wrong answer), the wrong-slot calibration slice (`j*n_embd` inside slot 0 — the reads collapsed to 1-2 triples), and the answer-target wrap (no linear probe closes mod-10 — the probes read the DIGITS at their OWN token positions, the mined table decides the wrap; 316 triples, 0 conflicts, the reads 400/400 exact). The report's `probe_corr` field is STALE-BROKEN (pre-slot diag layout — disclosed in the bench record); the confirmatory determinism re-run heals the field.
- [ ] **T4** — fold the numbers back to katgpt-rs issue 988: T2b checked there (the real cell EXECUTED, G1 PASS at +26.75pp sealed), the T4 GOAT pre-registered bars written from THIS cell's measured baseline (sealed baseline 0.1250 → the accuracy bar; the µs/token G2 bar; the G3 KL/flip bars at delta 8.0), the cross-repo citation per the citation gate.

## Non-goals

- No training, no gradient code anywhere (BOUNDARY.md; the fits are closed-form ridge reads — deterministic constructions, the Plan-025 doctrine).
- No serving-family wiring; the LoraPair hot-swap mount story stays katgpt-rs T5's envelope (landed there 2026-11-12, `3dc2b49a4`).
- No ternary-body hook, no GPU lane — the cell is CPU dense-hybrid, one checkpoint, offline.

## Related

- katgpt-rs issue 988 (the primitive + T1/T2a/T3-arms/T5/T6 landed) + Research 658; katgpt-rs issue 995 (attn-match solver dedup — this cell consumes `katgpt-core` grafts which use the shared `linalg::ridge_solve`).
- Issue 981 sink_audit (the capture-seam pattern + the exact checkpoint precedent); issue 1150/946 (the qwen35 dense league posture).
- katgpt-rs Plan 025 (deterministic-overlay doctrine); npc_brain_graft precedent (riir-ai Proposal 055 — authored predecessor with ProbeSet + exact gate).
