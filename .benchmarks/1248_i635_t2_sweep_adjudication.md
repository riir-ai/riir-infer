# Bench 1248 — i635 T2 sweep adjudication: coverage REFUTED, recipe-2 ladder STOP

**Date:** 2026-11-20 · **Plan:** [635](../.plans/635_i1189_recipe2_recall_calibrated_rank_sweep.md) (issue 1189 recipe-2, T2 coverage axis) · **Session:** zed-467f5139@4090 (the adjudication wake, fired early — the sweep completed at 01:54Z)

## The sweep

8 arms = 4 layer windows × a/b determinism twins on the Gitega anchor (`examples/facts/i1189_gitega_anchor.json`, bench 1242), harness at riir-infer `27211f8`+ (the amended gates (i)–(iv), bench 1242), model `Qwen3.5-0.8B-Base-Q8_0` (data-root pin), rank 16 (T1's best-so-far; capacity-refuted), rounds 6. Master runner `E:\tmp\i635\t2\run_t2_all.ps1`, master log `t2_master.log`, per-arm logs + manifests under `E:\tmp\i635\t2\<window>_<arm>\`. Launched 2026-11-19T05:35:39 +07; last arm exited 08:54:00 +07 11-20 — **sweep wall ~34.3 h** (8 arms × 3.4–3.9 h; each b-run ~+10 min over its a-run, box-noise).

## Box state

4090 (shikuwa) exclusive-by-construct: the sweep held ~25.6 GB of the 32 GB (0.1–0.44 GB free throughout; the construct is CPU-bound, GPU idle). No cargo contender ran inside the window (the box's lanes were held off per the RAM disclosure; verified each wake). PROVENANCE class: correctness cells — walls quoted as scheduling data, never perf claims.

## The table (per-arm read from the sealed manifests/logs)

| window | arm | envelope digest (sealed) | fit | recall armed/baseline (Δ) | baseline-gate | repetition-gate |
|---|---|---|---|---|---|---|
| {8,9,10,11} | a | `270c2c21cbc43ea5…d20148` | 0.000 | 0/5 vs 0/5 (+0) | PASS | GREEN |
| {8,9,10,11} | b | `270c2c21cbc43ea5…d20148` | 0.000 | 0/5 vs 0/5 (+0) | PASS | GREEN |
| {10,11,12,13} | a | `8811c5d37ded31d8…4b0eb9` | 0.000 | 0/5 vs 0/5 (+0) | PASS | GREEN |
| {10,11,12,13} | b | `8811c5d37ded31d8…4b0eb9` | 0.000 | 0/5 vs 0/5 (+0) | PASS | GREEN |
| {12,13,14,15} | a | `038f1f63c161f61d…a2130` | 0.000 | 0/5 vs 0/5 (+0) | PASS | GREEN |
| {12,13,14,15} | b | `038f1f63c161f61d…a2130` | 0.000 | 0/5 vs 0/5 (+0) | PASS | GREEN |
| {14,15,16,17} | a | `1c8069df767aa1e0…` | 0.000 | 0/5 vs 0/5 (+0) | PASS | GREEN |
| {14,15,16,17} | b | `1c8069df767aa1e0…` | 0.000 | 0/5 vs 0/5 (+0) | PASS | GREEN |

Twice-run law: all four a/b envelope digests BYTE-IDENTICAL — determinism confirmed ×4. Heldpara probe logprobs collapsed (−57..−64 nats class) in every armed arm.

## Adjudication (per the plan's pre-registered bars)

- Bar: any window with fit-gain ≥ +0.05 over T1's best (0.500) AND G1 logprob Δ ≥ +0.3 → adopt. **No window cleared it — fit = 0.000 in all eight arms** (zero explained key-family variance at every layer band 8–17 on the Gitega anchor).
- Pre-registered consequence: **coverage REFUTED as a lever; T3 stands alone** — and with T3 also not adopted (bench 1244), **zero levers compose → T4 is moot, never runs → the T4-NO-GO law fires → T5 Cell B never opens → the recipe-2 ladder records STOP.**
- Closing finding: Recipe-1's France fact measured fit 0.500 (Bench 1219); the Gitega anchor measures 0.000 at every window and every arm. The anchor change moved the sweep to a fact family the closed-form rank-16 direction solve cannot reach at any layer band in the mid-to-late MLP block. Capacity (T1: rank 32/64 pin 0.500 on France), coverage (T2: 0.000 everywhere on Gitega), and information (T3: NOT ADOPTED, repetition attractor) are all measured-dead. The remaining ladder space is the owner-directed recipe space (issue 1189), not this plan's three levers.

## Gates held

Baseline-gate (amendment ii: baseline 0/5 ≤ 1/5) PASS in all eight arms; repetition-gate (amendment iv: zero armed collapses on the subject set) GREEN in all eight arms; demote (D) and reload (F) identities byte-identical in all eight arms; the flag-off Recipe-1 identity law untouched.

## Verdict

**T2 REFUTED (coverage). T4 MOOT (never runs). T5 MOOT. T6 verdict: recipe-2 measured-negative end-to-end — ladder STOP recorded in the plan rows.** The machinery verdict is unchanged from Bench 1219: tap seam, overlay-at-dequant, harness phases, sealed sidecars, demote/reload identities all green — the pipeline is sound; the recipe's levers are dead on this anchor.
