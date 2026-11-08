# Issue 1186 — CQ3 league cell on the qwen3.8 dense lane (Cactus whole-model quant, katgpt-rs issue 987 lead A3)

**Status:** OPEN — intake issue, filed from katgpt-rs `.issues/987_cactus_neurips2026_follow_ups.md` (Research 657 §A3, 2026-11-08). Issue-first: no GOAT plan opens until the matched-calibration arm has a runnable fixture. Execution routes 4090 (league lane — the Bout law applies; run ALONE, box state recorded, engagement proof first).

## What the lead asks (from 987)

A league cell pitting the paper's **never-unpack GEMV** (FWHT on FP16 activation + int8 SDOT + per-(g,b) LUT centroids, effective `b_eff = b + 0.125` bits-per-weight) against **llama.cpp Q3_K at matched bpw** on the qwen3.8 dense lane: pp2048 / pp4096 prefill + decode, the league's standard shape. The lane is THIS repo's (issue 1176 made the dense Metal league posture the DEFAULT; the CUDA twin is `qwen38_dense_cudarc`).

**NOT Bonsai** — Bonsai-2-27B is ternary-trained (wrong input class for a CQ-style recipe); the qwen3.8 dense lane is the designated consumer.

## The gate the vendor cell is MISSING (⛔ pre-registered now, before any wiring)

**Matched-calibration arm REQUIRED.** CQ calibrated on ~2,900 joint multimodal samples while the baselines calibrated on 128 text samples — the reported margins are partly a CALIBRATION-BUDGET artifact, not an algorithm margin. Any league row we publish carries BOTH arms: (a) the paper's asymmetric posture (for the record), (b) the matched-calibration arm (equal sample counts + equal modality mix). **The margins that transfer to our text lane are arm (b)'s**; quoting arm (a) alone is forbidden.

Additional standing bars:

- **Per-family conditional retention** per the lossy-surface promotion rule — CQ is a lossy surface (whole-row rotation + LUT centroids); aggregate perplexity alone can be flat while a family flips. Retention-gated per family, demote the family that drops.
- **Engagement proof first** — prove the never-unpack GEMV arm is REACHED in the measured posture before any A/B (the Issue-1124 T1.5 lesson).
- Vendor numbers are self-reported; our fixtures are ours (the x86_64-matrix lesson: a benchmark IS its fixture).

- [ ] T1 — fixture: never-unpack GEMV prototype (FWHT-on-activation + int8 SDOT + per-(g,b) LUT) behind an opt-in feature on the qwen3.8 dense CPU/CUDA path; bit-exactness fixture for the LUT path vs a reference dequant (G1 class).
- [ ] T2 — matched-calibration harness: equal-budget calibration for BOTH arms (ours + llama.cpp Q3_K baseline); the sample-count/mix ledger recorded beside every row.
- [ ] T3 — the league cell: pp2048/pp4096 + decode vs llama.cpp Q3_K matched-bpw (4090 window, run ALONE, `PROVENANCE:` line quoted); per-family retention cells alongside the perf cells.
- [ ] T4 — verdict: GOAT gate per the league law (G1 correctness / G2 perf / G3 no-regression / G4 alloc); win → demote-only env promotion per the env-switch law; loss → recorded verdict, the corpus note filed.

## Constraints

- Opt-in until T3 passes (never in the release set before a GOAT); linear mixes only, no softmax; the league model naming law: Cactus artifacts are "Cactus <name>", ours keep their full lineage names — never a bare colliding stem (987's lineage check, 2026-11-08).
- 4090 routing: heavy-model league cells are rig jobs (schtasks route, sync-before-run), not workstation cells.

## Refs

- katgpt-rs `.issues/987_cactus_neurips2026_follow_ups.md` lead A3 · Research 657 §A3 · Cactus "Whole-Model Quantization Below 4 Bits Per Weight for On-Device Multimodal Models" (NeurIPS 2026, Mosoyan et al. — title/venue verified on cactuscompute.com/research 2026-11-08; paper-claim numbers ride Research 657's distillation)
- This repo: issue 1176 (the qwen3.8 dense league posture — the lane this cell runs on), the league records in riir-ai `.docs/09_performance/perf_rematch_league.md`
- The lossy-surface promotion rule + Report-the-Floor: katgpt-rs AGENTS.md §Feature Flag Discipline
