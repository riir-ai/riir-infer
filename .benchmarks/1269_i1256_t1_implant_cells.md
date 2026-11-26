# Bench 1269 — Issue 1256 T1: the implant decode-seam cells (fixture + real tier, dense-hybrid 0.8B CPU)

**Status:** LANDED — **all four cells PASS at both tiers** (G1 silent identity, G1b false fire, G2 fired exactness, G3 composition); determinism PROVEN at both tiers (payload blake3 reproduced byte-for-byte on re-run). The first real-tier run also caught a cell-expectation bug (digit VALUE vs token ID) in one run — honest red, fixed, disclosed below.
**Date:** 2026-11-26
**Session:** shikuwa-4090-i1256t1 (the claim lease `e51bbbf`), 1795698636
**Issue:** riir-infer 1256 T1 (the cells staged in `0a44d25`; the runtime in `8386eba`)
**Reports:** `.benchmarks/i1256_t1_cells/report_real.json` (payload BLAKE3 `54d2518b5d717d96b4e12250011fb83097b5aac5441a49ec364f69c831352d75`) · `report_fixture.json` (payload BLAKE3 `fa774c8c4624228d577d4a499dcf7c438b4b665b5bca36ffd4a67ce2ac60c60f`)
**Code:** `src/implants_cell.rs` (the walk + fixture families + cells, lib), `src/bin/implants_cell.rs` (the bin: fixture/real tiers), `tests/implants_cell_gates.rs` (7 fixture gates, required-features `implants`)

## Posture (pinned BEFORE the real-tier touch)

- **Checkpoint:** `Qwen3.5-0.8B-Base-Q8_0.gguf`, BLAKE3 `70fd9c83d63454ae…` — the bench-971 league file, pin matched.
- **Body:** dense hybrid (`load_qwen_deltanet_weights_gguf` + `forward_qwen_deltanet`), **CPU, same-device scope** — the bench-971 host shape. The dense-hybrid body's forward in this tree is CPU-only (the GPU forward lives engine-side in riir-gpu, a different lane); the byte-identity scope is satisfied per-device on the CPU (bf16-class forwards differ across GPU types — the WSM repo's own disclosure, restated in the issue's G1 bar). A ternary checkpoint refuses loud.
- **Fixtures:** authored in `src/implants_cell.rs` — 15 single-digit addition rows (`{a}+{b}=`, a+b ≤ 9), 14 subtraction rows (positive, single-digit), 10 look-alike rows (operand swaps of UNTABLED pairs, paraphrases, adjacent operators, multi-digit operands, negative results), 4 prose rows. Synthetic-only per the WSM §4.1 process / grammar-not-answers discipline — zero benchmark test rows. Every prompt's FULL tokenization is its trigger rule (unique by construction — operand order is in the pattern), validated at authoring (`=` tail, unsplit operands, ≤ WINDOW_MAX 16).
- **Bias pin:** `IMPLANT_BIAS` 8.0 — the grafts_cell BIAS_DELTA precedent (the measured base top-1 gap after `=` is ~5 logits; 8 clears it; pinned a priori, never swept).
- **Answer surface:** context-derived per prompt via the continuation trick (`digit_continuation` — encode(prompt+digit), suffix exactly 1 token; the bench-971 lesson). Real tokenizer digit ids 16..=24 for 1..9.
- **Cells (the issue's pre-registered bars):** G1 armed-but-silent ≡ base (prompt-phase digests + first emission; the demoted mount on a row it WOULD fire on ≡ base FULL-walk — demoted never fires, structurally); G1b 0 fires on every look-alike + prose row (observe-level, prompt tokens); G2 the row's own fire at its own span (last prompt step, own implant) + forced token emitted EXACTLY + prompt-phase digests ≡ base + fired-step digest diverges + per-fire exactness (chained fires included); G3 solo/co first-fire agreement + the other implant silent through the row's prompt phase + co ≡ solo through the prompt phase AND the targeted emission + combined exactness 29/29 ≥ each single's ceiling.
- **Gates:** 7 fixture gates green (`gated: tests/implants_cell_gates.rs`, env-serialized — any mount reads the demote env at mount); default-posture clippy `-D warnings` clean (the gated-to-nothing law), `--features implants,implants_cell --all-targets` clean; the implants fixture gates 9/9 + G4 alloc gate still green (the observe path untouched).

## Results

| cell | fixture tier (SyntheticSource) | real tier (league checkpoint, CPU) |
|---|---|---|
| G1_silent_identity | PASS (14 rows, 0 prompt fires, digests equal; demote row digest-equal) | PASS (same counts) |
| G1b_false_fire | PASS (14 rows, total_fires 0) | PASS (same counts) |
| G2_fired_exactness | PASS (29/29 exact_forced, 10/10 lookalike_identical, 0 chained) | PASS (29/29 exact_forced, 10/10 lookalike_identical, 1 chained fire disclosed + exact-checked) |
| G3_composition | PASS (29 solo+co fire, 29 no_interference, 29 combined_exact, 14 silent) | PASS (same counts, 0 post-emission divergence) |

Run wall time: fixture 0.1 s; real ~305-350 s (43 walks × ~20-30 forwards, single CPU thread posture, release).

## Determinism (the grafts G1 law)

- Fixture tier: payload digest reproduced across runs.
- Real tier: two runs → payload BLAKE3 `54d2518b5d717d96b4e12250011fb83097b5aac5441a49ec364f69c831352d75` both.
- **The fix this forced (disclosed):** the first real-tier determinism pair DIVERGED — the report's `box_state` line embedded a live nvidia-smi snapshot (GPU util/memory move between runs under sibling load), riding the canonical payload. Fix: `box_state` is blanked in the hashed compact payload (live provenance, never a payload fact), documented in the bin header. Both tiers re-confirmed under the fixed hashing.

## The bug the first real run caught (honest red, root-caused, fixed)

The first real-tier run FAILED G2/G3 on every row — with the diagnosis notes showing the model emitting the CORRECT answer digit (`"2+2=" emitted Some("4")`, top2 id 19 at 25.09). Root cause: the cell's expectation helper re-derived the forced token from the prompt TEXT and returned the digit's integer VALUE (4), while the walk compares TOKEN IDS (4 → id 19 on the real tokenizer). The fixture tier MASKED the bug — its synthetic digit ids ARE the digit values (ids 0..=9) — the exact single-vocabulary trap the house discipline names. Fix: the expectation reads the AUTHORED TABLE (one source of truth) — `ArmsRig` now carries `table: Vec<TableRow>` (minted from the lanes) and the cells resolve `(implant, forced_token)` by prompt lookup; the text-re-derivation helpers are deleted. Fixture gates re-run green; the real tier went 29/29 exact.

## Design refinement the real tier forced: chained fires

After a forced `"2+2=4"`, the base model plausibly continues with further equations (`"4+2="` is IN the trigger table — the bench-971 continuation posture: `"11\n11 mod 10 = 1…"`). A strict one-fire/full-walk-equality assert would red on CORRECT behavior. The landed assert shape: the targeted span's exactness is the bar (prompt-phase identity + the row's own fire at its own span + the forced emission); post-prompt chained fires are COUNTED and exact-checked per fire (each fire's fold decides its own emission — resolved via the runtime's mounted action, `biases[0].0`), never silently pooled; same-step double fires refuse loud (attribution undefined); post-prompt divergence on look-alike/prose rows is DISCLOSED (`gen_diverged_disclosed` / `post_emission_diverged_disclosed`), not failed — the model generating genuinely tabled text is the trigger's own domain, not a fixture false fire. Real tier measured: 1 chained fire, exact. The pre-registered observe-level G1b bar (0 fires on the look-alike PROMPT tokens) stays strict.

## Box state (the shared-box disclosure)

- shikuwa (Windows 11, i7-13700K, 24 logical CPUs per `available_parallelism`), AC power; the cell is CPU-posture — the GPU-exclusivity law does not bind, disclosed anyway: runs 1-2 of the real tier executed while a sibling lane's `plm_smoke.exe` held the GPU at 100% (riir-train lane, cudarc consumer); the confirm pair ran with the GPU at ~10% / 596 MiB. Per-run `box_state` lines ride the reports (excluded from the canonical digest, above).
- Fixture/real reports carry per-run wall time; latency is NOT a claim of this record (exactness and identity are).

## What this cell does NOT claim

- No accuracy claim over any benchmark (the fixtures are synthetic; the implant forces answers on ITS OWN table — the G2 bar is exactness on the targeted surface, per the issue's pre-registration: "an exact implant's honest bar is exactness, not 'beats the model'").
- No cross-device claim (the byte-identity scope is this CPU; a GPU-class re-run needs the engine-side GPU lane; the 046 F2 margin-certificate form is the portable alternative, not exercised here).
- No serving-path claim (the lane stays opt-in `implants`/`implants_cell`, never in a default build).
- The residual-overlay write action (a later T1 rung per the issue) is not exercised — bias writes only.

## Next

- T2 (refit-ends transfer) / T3 (discovery loop) stay PARKED behind the issue's stop rule (a serving consumer within one cycle) — none is identified yet; the consumer lane is the named next engagement.
- The Issue-686 design-input fold (T4) stays riir-train-owned, pre-registered-when-dormant per the issue.
