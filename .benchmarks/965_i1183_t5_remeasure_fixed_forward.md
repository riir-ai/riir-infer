# Bench 965 — the 1183 T5 re-measure on the FIXED DSpine forward: untrained mean_j 1.000 → 2.565 (the closure verdict's premise measured a broken forward)

**Status:** RECORD — corrects the 1183 T5 failure decomposition and the "worse than uzu" closure premise; the vendor-lane verdict revision is recommended owner-facing (verdict session recorded in HISTORY).

## Cell

`t5_head_only_fixture_greedy` — the EXACT 1183 T5 posture re-run on the fixed
forward (riir-infer `1f0ec28`, the message-gate sigmoid fix): the Qwen3.8-27B
DSpine b16 head (UNTRAINED, the checkpoint as shipped) serial-tail chain over
OUR Q4_K_M league target, Metal-posture context captures, greedy keyed argmax,
200 cycles, the league fixture prompt (603 tokens).

- dspine blake3 `53d185164e20b6ea51fd72a3f093492aff1596a7dda0693d45c7d057dc1cc15c`
- gguf blake3 `ce14c95531256552644397d8c38691c0fe6f228cd2dd2d8cce13021ecc5bd06a`
- G1-T5 PASS — serial == specdec over 32 committed tokens (bit-parity gate green on the fixed forward)
- host amo (M5 Ultra, AC); ~29 s/cycle; raw row `/tmp/i1183_t5_remeasure_fixed_forward.json` (copy below)

## The row

| metric | hyperbola forward (1183 T5, Bench-era) | FIXED forward (this bench) |
|---|---|---|
| mean_j (200 cycles) | 1.000 | **2.565** |
| emitted total | ~200 (1/cycle) | **514** (2.57/cycle) |
| j histogram | — | [0,64,60,35,23,7,3,2,1,2,0,0,0,3,0,0,0] |
| survival p=1 / p=2 / p=3 | — | 0.68 / 0.38 / 0.205 |

## Verdict

**Metric convention (checked, apples-to-apples):** both this bench and Bench 954 (uzu) use the SAME harness, the SAME 603-token league fixture, the SAME GGUF, and the SAME mean_j definition — `emitted_total/cycles` (accepted draft tokens per pass; the j=0 histogram key exists but carries 0 cycles in BOTH rows: 954 [0,161,29,7,…], 965 [0,64,60,35,…]). The 2.565-vs-1.27 comparison is therefore same-convention. The vendor's 2.882 (Bench 952-c4) remains cross-fixture (their engine, their numerics) and is never pooled with ours.

**rank0 vs mean_j reconciliation:** `rank0[p]` = the chain's rank-0 candidate LANE's token equals the target at verify row p (a rank-1-lane accuracy metric) — DSpine's rank0 rates read ≈ 0 while acceptance is 2.565. These measure different things: the chain rarely puts the target at the rank-0 lane, but the target frequently sits INSIDE the top-k candidate set at the WALKED lane (survival p1 = 0.68 — the set+walk carries the acceptance). No contradiction, and rank0 is NOT the distill loss's signal either way (the loss is the score-row CE over the full candidate set — plan 492's surface).

1. **The 1183 closure premise is overturned (delegated-owner verdict AGREE, option A — revised in place on the issue).** "The DSpine head is worse than uzu (1.27) and never saw our numerics (mean_j 1.000)" measured a forward whose message gate was `1/(1 − e^z)` (riir-infer 1193) — the head's real zero-shot acceptance on OUR target is 2.565: **the strongest MEASURED zero-shot head on this target (single 603-token fixture prompt, 200 cycles — held-out cells owed before any beat-lane claim), 2.0× uzu's trained 1.27 (Bench 954), same convention.** The "never saw our runtime's numerics" half of the verdict STANDS (rank0 ≈ 0).
2. **The beat-lane prize bar (≥3.2, the 1179 arithmetic → 100–135 tok/s vs the owner's 121.1) is NOT yet cleared** — the gap (2.565 → 3.2) is exactly the distillation delta plan 492's T2/T3 exist to buy; the DSpine family is the distill vehicle (uzu's re-open condition "a new head family" + this head's zero-shot lead). Promote/stop rule pre-registered in plan 492 before any training spend: promote on held-out mean_j ≥ 3.2 (same convention) + G1 serial==specdec PASS; stop if the distill delta < +0.3 over 2.565 after the first agreed budget; otherwise report without promoting.
3. **Plan 492's premise strengthens:** "the head never saw our runtime's numerics" stands (rank0 ≈ 0), and the distill surface now trains against an FD-exact backward (riir-train issue 683 closed the same day: tier3 median 7e-4).
4. **Correction of record:** every premise citing DSpine mean_j ≈ 1.0 (1183's closure, Bench 959's chain numbers, plan 492's Why, riir-train 679) must read mean_j 2.565 for zero-shot quality claims; the bit-exactness-vs-vendor claims (Bench 959 piecewise) stand. Residue owed regardless of T2/T3 (tasks filed in plan 492): the Bench 959 shape-walk re-run on the fixed forward + the main-path golden replay piece.

## Refs

- riir-infer 1193 (the gate fix; the blast-radius correction of record) ·
  riir-train issue 683 + plan 492 (the backward's three root causes closed
  the same day) · 1183 T5 (the hyperbola-era row this corrects) · Bench 954
  (uzu 1.27) · 1179 (the ≥3.2 prize arithmetic) · 952 condition-4 (the
  vendor chain-only 2.882 reference — note: on THEIR engine and numerics)
