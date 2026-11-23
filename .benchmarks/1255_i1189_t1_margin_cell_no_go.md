# Bench 1255 — plan 637 T1: the recipe-3 margin-target deciding cell (Gitega anchor, `--margin-target`, twice-run) — T2 NO-GO, the pre-registered STOP fires

**Verdict: NO-GO — the plan 637 falsifier fired.** The differential battle-position write had real bite (held-out next-token logprob moved by −40 nats class) and never flipped the battle token on any training wording (depth 0 everywhere, incumbent competitors unchanged through all 6 refinement rounds); recall stayed 0/5 Δ 0; a control probe MOVED under the armed overlay (the leak gate's subject); the GO bar fails on its recall, depth, and control clauses. Machinery/identity green throughout (twice-run digest equality, demote + reload identities, sidecar sealed) — bench 1219's pattern again: the objective-shape axis records its fourth measured negative with the machinery standing. Per plan 637's narrowed STOP law: the recipe space is closed over the four pre-registered axes (capacity — bench 1242 T1 / coverage — bench 1248 T2 / information-attractor — bench 1244 T3 / objective-shape — this cell); the cap, protection-weights, and key-position degrees of freedom are HELD, not ruled out. Issue 1189's Cell-A recipe space closes measured-negative.

## The cell (as pre-registered; nothing re-tuned)

- Registered Gitega anchor verbatim (`examples/facts/i1189_gitega_anchor.json`), window {10,11,12,13} (recipe-1's reference window, NOT re-swept), rank 16, rounds 6, `--margin-target`, T = the harness defaults; `CARGO_TARGET_DIR=/tmp/i1189t0infer` isolation; amo `--shared`, one queue job (`2309ed09`): rounds-1 smoke (12:57Z) → full run A (13:55Z) → full run B (14:49Z).
- Cap pre-registration (from the phase-A census, bench-1242-faithful): margins 1.934 / 2.689 / 3.083 nats (competitors ` K` id 710 ×2, ` located` id 7172, all battles at pos 0); **`margin_cap_nats` held at the default 5.0** — max margin 3.083 rides untruncated, floor 3.0 adds after ⇒ D = 6.083 nats. The manifest echo carries the whole posture: `margin_target=true, margin_cap_nats=5.0, degeneracy_tau=0.1, target_floor_nats=3.0, battle_movement_k=3.0`.
- STAMP `5dd3be79` (the harness's last commit; the T0 harness at `c4ae2980` + core at katgpt-rs `f4e8f555c`); release CPU.

## Results (both runs byte-identical where identity applies)

| gate / reading | result | GO-bar clause |
|---|---|---|
| G2 twice-run identity | envelope + sidecar digest `0c4a7847…` in BOTH runs; D demote byte-identical (10 probes); F reload byte-identical (25.9–26.9 s incl. re-load) | (machinery, not gating) |
| construct | fit=0.000 rounds=6 miss=false dropped_energy=1.571e-13 gain=0.50 (fit is DIAGNOSTIC — not comparable across objectives, never gating) | — |
| battle after the write | `[Some(0), Some(0), Some(0)]` — battle position never moved past 0 on any training wording | **depth > 0 on ≥ 1 wording: FAIL (0/3)** |
| per-round competitor re-capture | `[710, 710, 7172]` identical at every round — the incumbent held the battle through all 6 rounds despite the write | corroborates depth 0 |
| recall (registered held-out set) | `armed 0/5 (baseline 0/5, Δ +0/-0)` | **baseline-delta ≥ 2/5: FAIL (0/5)** |
| baseline-gate (amendment ii) | baseline 0/5 ≤ 1/5 → PASS | (floor subject, not the bar) |
| repetition gate (amendment iv) | collapsed 0/5 → 0/5 → GREEN as pre-registered — while DISCLOSING the armed generations are visibly degenerate ("located in GG C GALYI KALITE" class) | green as measured |
| controls | control#1/#2 byte-identical; **control#0 MOVED** ("The 10 Best Hotels in San…" → "The 2011 World Cup") | **controls byte-identical: FAIL (1 of 3 controls moved)** |
| G1b look-alikes | not scored by the harness log (a T1 harness gap — the armed look-alike generations are visibly damaged: "Kigali, which has a population of 1," → "Dar-GG KULI GULI KÜL") | **FAIL on the visible evidence; the harness should score this arm explicitly (recorded as a follow-up, not a re-run)** |
| held-out logprob | `−5.69 → −48.16`, `−5.54 → −56.61`, `−6.20 → −50.05`, `−3.98 → −45.31`, `−10.25 → −50.63` — every held-out probe collapsed by 40+ nats | the bite was real and destructive |

## The failure anatomy (new, and the record's value)

Recipe-2's writes were polite: held-out logprob moved +0.345, generation unmoved. Recipe-3's write was violent: aiming `D = 6.083 nats` of movement along `unit((E_answer − E_competitor) ⊙ g)` at the battle-position keys sank the answer's own likelihood along with the competitor's — the armed probes' first-position landscape collapsed (40+ nat drops) while the greedy argmax at the battle stayed the incumbent (` K` / ` located` won every re-capture). The differential direction at this magnitude did not tilt the battle; it cratered the neighborhood. Combined with the control leak (a rank-16 down-proj change at layers 10–13 is a WEIGHT edit — it applies to every prompt; recipe-1/2's gentle writes kept controls byte-identical, this one did not), the reading is: **the battle-position differential target, at the margin the anchor demands, cannot flip a greedy decode without wrecking the local distribution — the objective-shape axis is exhausted on the pre-registered axes.** No rank/layer re-sweep, no paraphrase regeneration, no Cell B: the non-goals stand.

## Honest caveats

- The two full runs ran on amo `--shared` with sibling jobs live (load ~8–11; the reframe render job joined mid-window). Byte-determinism is algorithmic (no RNG) — the twice-run digest equality is load-independent — and no wall number is quoted as a claim; construct walls recorded only as provenance (3383.5 s / 3144.0 s).
- The smoke leg (rounds=1) passed before the full runs and is preserved at `/tmp/i1189_t1_smoke` (transient); the durable copies here are the two full logs + manifests.
- The G1b arm's non-scoring is a harness gap inherited from the recipe-1/2 cells; this record adjudicates it on the visible generations. A follow-up issue would make the harness score G1b explicitly — filed as prose here, not blocking this verdict (the recall + depth + control clauses already fail the bar).

## Artifacts

- Logs + manifests: `.benchmarks/i1189_t1_margin_cell/` (runA.log, runB.log, manifest.json ×2 — durable copies of the /tmp originals).
- Envelope/sidecar digests: `0c4a7847445a94334fd04a8c2929957d17d14a745007e26ddb30fd2e0c57276b` (both runs).
- Plan: `.plans/637` (T1 `[x]`, T2 `[x]` NO-GO + STOP, T3 close-out in the same fold). Issue 1189 carries the verdict + this bench ref.
