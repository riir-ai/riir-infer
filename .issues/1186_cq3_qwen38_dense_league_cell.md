# Issue 1186 — CQ3 league cell on the qwen3.8 dense lane (Cactus whole-model quant, katgpt-rs issue 987 lead A3)

**Status:** IN PROGRESS — T1 + T2a LANDED 2026-11-09 (this box); T2b/T3/T4 open. Execution routes 4090 (league lane — the Bout law applies; run ALONE, box state recorded, engagement proof first).
**Claimed:** zed-m3-cq3@0, 1794167021, expires 2026-11-10T19:43:41+00:00

## What the lead asks (from 987)

A league cell pitting the paper's **never-unpack GEMV** (FWHT on FP16 activation + int8 SDOT + per-(g,b) LUT centroids, effective `b_eff = b + 0.125` bits-per-weight) against **llama.cpp Q3_K at matched bpw** on the qwen3.8 dense lane: pp2048 / pp4096 prefill + decode, the league's standard shape. The lane is THIS repo's (issue 1176 made the dense Metal league posture the DEFAULT; the CUDA twin is `qwen38_dense_cudarc`).

**NOT Bonsai** — Bonsai-2-27B is ternary-trained (wrong input class for a CQ-style recipe); the qwen3.8 dense lane is the designated consumer.

## The gate the vendor cell is MISSING (⛔ pre-registered now, before any wiring)

**Matched-calibration arm REQUIRED.** CQ calibrated on ~2,900 joint multimodal samples while the baselines calibrated on 128 text samples — the reported margins are partly a CALIBRATION-BUDGET artifact, not an algorithm margin. Any league row we publish carries BOTH arms: (a) the paper's asymmetric posture (for the record), (b) the matched-calibration arm (equal sample counts + equal modality mix). **The margins that transfer to our text lane are arm (b)'s**; quoting arm (a) alone is forbidden.

Additional standing bars:

- **Per-family conditional retention** per the lossy-surface promotion rule — CQ is a lossy surface (whole-row rotation + LUT centroids); aggregate perplexity alone can be flat while a family flips. Retention-gated per family, demote the family that drops.
- **Engagement proof first** — prove the never-unpack GEMV arm is REACHED in the measured posture before any A/B (the Issue-1124 T1.5 lesson).
- Vendor numbers are self-reported; our fixtures are ours (the x86_64-matrix lesson: a benchmark IS its fixture).

- [x] T1 — fixture: never-unpack GEMV prototype (FWHT-on-activation + int8 SDOT + per-(g,b) LUT) behind an opt-in feature on the qwen3.8 dense CPU/CUDA path; bit-exactness fixture for the LUT path vs a reference dequant (G1 class). **LANDED 2026-11-09** — `crates/riir-infer-gpu/src/qwen38_cq3.rs` (opt-in `cq3` = `["dep:half", "katgpt-core/meld"]`; the FWHT half consumes `katgpt_core::meld::walsh_hadamard_in_place_normalized` via a match dispatch over g ∈ {16,32,64,128,256} — substrate-first, no parallel Hadamard) + `tests/qwen38_cq3_g1.rs` (7-fixture battery, [[test]] required-features row landed in the same commit). Record:
  - Recipe steps 1–4 (SmoothQuant fold as caller-supplied per-channel scale; deterministic seeded rotation R_g = (1/√g)S₂H_gS₁P; FP16 group-norm → unit sphere with the NORM stored as FP16 — quantize divides by the round-tripped FP16 norm, dequant multiplies it back; one scalar Lloyd–Max codebook per (g,b) trained on 2048 synthetic unit vectors, 32 sweeps, seeded xorshift64* — never on weights). Step 5 (GPTQ Hessian residual) DEFERRED to T2 by design: it consumes the same calibration samples the matched-calibration arm fits.
  - **G1a BIT-EXACT PASS**: fused LUT GEMV ≡ dequant + plain sequential matmul, `to_bits` equality, with AND without the smooth fold, at g=128/16/32 — the never-unpack kernel preserves exact per-element semantics.
  - Two GEMV arms: `gemv_f32_lut` (bit-exact class) + `gemv_i8_sdot` (the phone-kernel shape: per-group int8 activation (absmax/127), pre-rounded int8 centroids, i32 dot with per-group de-scale `f32(acc)·s_c·s_x·f32(f16(norm))` — NEON SDOT semantics, `g·127² ≤ 2.06M` per group).
  - Measured pins (m3, dev build, fixture seeds): g1b int8-vs-f32 rel-RMS ≈ 2.0e-2 (pin 5e-2); g1d NO-GPTQ recipe floor b=4 ≈ 1.5e-1 (pin 3e-1), b=3 (pin 5e-1) — the fixture's small per-block norms (σ_w=0.05) amplify relative error by ≈1.8; this IS the floor the T2 residual must improve on. g1e asserts `b_eff = b + 16/g` to 1e-9 (g=128 ⇒ b+0.125 — the paper's number, computationally). g1f determinism (re-quantize + re-derive → bit-identical).
  - 40s → 10.5s suite time: derivation constants sized (2048 vectors, 32 sweeps) before any pin could fossilize.
  - OBSERVED, not mine: the i1179 probe examples carry pre-existing clippy E0432/E0601 reds under `--all-targets` (cfg posture, unrelated lane).
  - The T1 one-line lesson (recorded for T2): the dequant reference reconstructs the ROTATED weights — comparing it against the unrotated original reads sqrt(2) "uncorrelated" noise; the two-arm comparison must run on the same basis (W_rot·x_rot ≡ W·x held to 3.3e-7 in the algebra fixture).
- [ ] T2 — matched-calibration harness: equal-budget calibration for BOTH arms (ours + llama.cpp Q3_K baseline); the sample-count/mix ledger recorded beside every row.
  - [x] T2a (ours-arm calibration fit, LANDED 2026-11-09, this box) — recipe steps 1 + 5, CPU fixture lane behind the same opt-in `cq3` feature:
    - `fit_smooth` — SmoothQuant fold from calibration rows: `s_j = clip(diag(H)_j^α / wmax_j^(1−α), 1/8, 8)`, α = 0.25, MEDIAN-NORMALIZED before the clip (the fold must not move the layer's effective gain; the order is documented and load-bearing). Upper-median tie rule pinned.
    - `quantize_residual` — GPTQ-class BLOCKWISE Hessian residual (the paper's "blockwise H_s, damped Cholesky (H_s+λI)⁻¹, λ=0.01·mean diag"): per-(row, group) sweep over that group's `g×g` slice, feedback confined to the block, then a pass-2 renormalize+requantize (stored scale matches post-sweep block energy — dequant stays consistent). All f32 sequential, deterministic (g2a pins bit-identical re-quantize).
    - **The debugging arc, recorded because it cost three wrong theories:** (1) my first U-chain shortcut (U = (chol H)⁻ᵀ) gives UᵀU = (LᵀL)⁻¹ — the WRONG inverse; the full L⁻ᵀL⁻¹ must be re-factorized. (2) An operand-order inversion in the feedback (Frantar's err1 = (w−q)/d, W -= err1·U) made the compensation RE-ADD the error it should cancel. (3) The decisive cross-check (now fixture g2c) settled it: the module's sweep reduces the calibration objective and tracks an independent f64 exact-H⁻¹ reference within f32 slack.
    - **THE FINDING (drives T2's design): the residual's gain is a CALIBRATION-OBJECTIVE gain.** Measured at n_cal/d_in = 2:1: cal-objective −76% at b=4 (142.75→34.17), −86% at b=3 (879.84→121.64); held-out mean rel-RMS +23% (0.110→0.135) — the sweep partially overfits the calibration realization and the gain does NOT transfer to raw held-out GEMV error at this sample ratio. The fixture gates the objective (the sweep's contract) and DISCLOSES the held-out delta; whether the gain transfers to TASK loss is precisely what T3's matched-calibration league cell decides — the matched-budget harness is not optional bookkeeping, it is the validity condition for quoting any margin.
    - **Fixture lesson (calibration-set construction):** the first calibration generator (random ±1 mixing over 32 latents) averages to a near-ISOTROPIC covariance — nothing for the residual to exploit, and every paired delta read as noise for an hour. Replaced with an honest factor model (8 latents, spectral decay λ_l = 4·0.7^l, N(0,1) loadings). Synthetic calibration must be checked for its ACTUAL conditioning before any GPTQ-family claim is read off it.
    - Fixtures: g2a (cal-objective gate, b=4 + b=3, determinism, kernel bit-exactness through the fitted fold), g2b (smooth bounds/median/degenerate channels), g2c (module vs independent f64 Gauss–Jordan reference: both beat RTN; module ≤ 1.05× the reference). Suite 11 tests, 10.4 s.
  - [ ] T2b (the cross-tool harness + ledger) — llama.cpp arm at the equal budget, the ledger beside every row: league-prep cycle (T3's window), needs the model + llama.cpp checkout on the rig.
- [ ] T3 — the league cell: pp2048/pp4096 + decode vs llama.cpp Q3_K matched-bpw (4090 window, run ALONE, `PROVENANCE:` line quoted); per-family retention cells alongside the perf cells.
- [ ] T4 — verdict: GOAT gate per the league law (G1 correctness / G2 perf / G3 no-regression / G4 alloc); win → demote-only env promotion per the env-switch law; loss → recorded verdict, the corpus note filed.

## Constraints

- Opt-in until T3 passes (never in the release set before a GOAT); linear mixes only, no softmax; the league model naming law: Cactus artifacts are "Cactus <name>", ours keep their full lineage names — never a bare colliding stem (987's lineage check, 2026-11-08).
- 4090 routing: heavy-model league cells are rig jobs (schtasks route, sync-before-run), not workstation cells.

## Refs

- katgpt-rs `.issues/987_cactus_neurips2026_follow_ups.md` lead A3 · Research 657 §A3 · Cactus "Whole-Model Quantization Below 4 Bits Per Weight for On-Device Multimodal Models" (NeurIPS 2026, Mosoyan et al. — title/venue verified on cactuscompute.com/research 2026-11-08; paper-claim numbers ride Research 657's distillation)
- This repo: issue 1176 (the qwen3.8 dense league posture — the lane this cell runs on), the league records in riir-ai `.docs/09_performance/perf_rematch_league.md`
- The lossy-surface promotion rule + Report-the-Floor: katgpt-rs AGENTS.md §Feature Flag Discipline
