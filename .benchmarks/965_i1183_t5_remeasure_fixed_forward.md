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

1. **The 1183 closure premise is overturned.** "The DSpine head is worse than
   uzu (1.27) and never saw our numerics (mean_j 1.000)" measured a forward
   whose message gate was `1/(1 − e^z)` (riir-infer 1193) — the head's real
   zero-shot acceptance on OUR target is 2.565: **2.0× uzu's trained 1.27
   (Bench 954) and above the 2.5 uzu-parity bar**, with the head UNTRAINED.
2. **The beat-lane prize bar (≥3.2, the 1179 arithmetic → 100–135 tok/s vs
   the owner's 121.1) is NOT yet cleared** — the gap (2.565 → 3.2) is
   exactly the distillation delta plan 492's T2/T3 exist to buy; the
   DSpine family is now unambiguously the distill vehicle (uzu's
   re-open condition "a new head family" + this head's zero-shot lead).
3. **Plan 492's premise strengthens:** "the head never saw our runtime's
   numerics" stands (the zero-shot chain still mismatches — rank0 rates ~0),
   and the distill surface now trains against an FD-exact backward
   (riir-train issue 683 closed the same day: tier3 median 7e-4).
4. **Correction of record:** every premise citing DSpine mean_j ≈ 1.0
   (1183's closure, Bench 959's chain numbers, plan 492's Why, riir-train
   679) must read mean_j 2.565 for zero-shot quality claims; the
   bit-exactness-vs-vendor claims (Bench 959 piecewise) stand.

## Refs

- riir-infer 1193 (the gate fix; the blast-radius correction of record) ·
  riir-train issue 683 + plan 492 (the backward's three root causes closed
  the same day) · 1183 T5 (the hyperbola-era row this corrects) · Bench 954
  (uzu 1.27) · 1179 (the ≥3.2 prize arithmetic) · 952 condition-4 (the
  vendor chain-only 2.882 reference — note: on THEIR engine and numerics)
