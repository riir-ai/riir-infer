# Bench 954 — Issue 1179 T1: the head-only acceptance price cell (Path A gate)

**Date:** 2026-11-08 · **Box:** bmo (M5 Ultra, AC, load 1.1-1.2/30, 18-82 GB mem free — `agent-mail rig status` quoted at run) · **Driver:** `agent-mail rig run bmo --shared` (jobs `01a1ba33`, follow-up `01a1baa3`)
**Posture:** feature `qwen38_specdec,metal_tensor_gemm`, `I1179_VENDOR_NUMERICS=1`, release; bin `qwen38_specdec_t1` (greedy, 200 cycles, `--sanity-steps` 32); the league DEFAULT target arms printed (dense-f16 matmul2d + tensor-flash prefill + half-B + Q6_K packed tail, Issues 1150/1171/1170).
**Artifacts (blake3-asserted at load, both PASS):** speculator `f6c67be2…638748` (Bench-1167 pin; the 1.0 GB artifact re-staged amo→bmo over the bridge after the bmo copy went missing — sizes+pin verified), GGUF `ce14c955…5bd06a`.
**Fixture:** `scripts/uzu_t2_prompt.txt`, 3234 chars → **603 tokens**; prefill 9.5-12.6 s.

## THE ROW (cell 1 — greedy, fixture — the decision cell)

```json
{"block_size":16,"capture_layers":[5,19,33,47,61],"cell":"t1_head_only_fixture_greedy","cycles":200,"emitted_total":255,"gguf_blake3":"ce14c95531256552644397d8c38691c0fe6f228cd2dd2d8cce13021ecc5bd06a","j_histogram":[0,161,29,7,1,2,0,0,0,0,0,0,0,0,0,0,0],"mean_j":1.27,"mode":"greedy","note":"uzu DFlash argmax chain (no weaver tree) over OUR Q4_K_M target; CPU f32 draft; int8-KV rounding not reproduced (orthogonal-basis, disclosed)","note_position_stats":"rank0[p] = draft argmax chain[p-1] == target argmax at verify row p, first-index ties, unconditional over all cycles; survival[p] = P(j > p)","prompt_kind":"fixture","prompt_tokens":603,"rank0_hits_by_position":[0,2,10,21,43,47,58,63,71,58,70,76,79,77,77],"rank0_rate_by_position":[0.0,0.01,0.05,0.105,0.215,0.235,0.29,0.315,0.355,0.29,0.35,0.38,0.395,0.385,0.385],"sampling":"greedy (keyed argmax, first-index ties)","seed":0,"spec_blake3":"f6c67be27ee8d81d7f133bf2724a969bd0b337cdbf2b7060e02803a219638748","survival_by_position":[1.0,0.195,0.05,0.015,0.01,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"ts":"epoch-1794130313"}
```

**mean_j = 1.27 tok/pass. G1-T1 PASS** (serial == specdec over 32 committed tokens — the harness's correctness gate; the divergence the salvage resolved (3848426's qkvg finding) is gone).

## The T1 decision gate — PATH A CLOSES

The issue's pre-registered gate (verdict rounds 1+2): **≥2.5 tok/pass head-only continues Path A; below closes it with the record.** Cell 1 reads **1.27** — under half the bar, against the vendor's own chain-only reference **2.882** (Bench 952's condition-4 cell, their engine, same Argmax chain shape). The beat arithmetic needed ~3.2 at serial 45-55 to clear the owner's 121 tok/s bar (Bench 1167: the vendor's own specdec-on cell reads 121.1 @ 3.18 tok/pass); at 1.27 the multiplier is nowhere near.

**Why the gate verdict is robust to the pending cells:** stochastic acceptance ≤ greedy in expectation (lossless rejection sampling cannot beat argmax's per-position agreement on the same model pair), and the vendor prompt is a prompt change, not a hypothesis change — no remaining cell can bridge a 2× gap. Cells 2-4 run for the RECORD (the issue's T1 as-written), not the verdict.

## The acceptance profile (why Path A fails)

- **rank0_rate** (draft chain[p-1] == target argmax at verify row p, unconditional): starts 0.0 at p=1, peaks **0.395 at p=13** — even at its best, the draft's argmax agrees with the target's barely 40% of the time; the first chain token almost never matches (0.01).
- **survival** collapses to ~0 by p=5 (1.0 → 0.195 → 0.05 → 0.015 → 0.01 → 0): the 15-token chain is rejected almost immediately on OUR target, so the keyed verify admits j=1 (161 of 200 cycles) with rare j=3-5.
- Two disclosed hypotheses for the vendor gap (their 2.882 chain-only on their engine): (a) the bench-952-f32 cell's finding inverted — the f32-exact draft read mean_j 1.20 vs their 2.882, and the vendor-exact int8 numerics (the salvage this cell ran on, incl. the prefix-cache write fix `d83ae45`) recovered only +0.07 → the quantization-chaos contribution is ~0.07 of a ~1.6 gap; the remainder is (b) the draft accepting differently against OUR Q4_K_M GGUF target than against THEIR runtime's target numerics — a target-side gap the draft was never distilled for. That is exactly Path B's premise: distill the head AGAINST OUR target, on the 4090, per the issue's T5 plan (already planned in parallel; the no-training-deferral rule keeps it open).

## Landing arc (the takeover)

The lane's state was split across boxes with the claim expired (zed-amo-specdec2, gone): amo held a 40-file stale-tree WIP (mostly an old tree copy over a newer HEAD; the REAL WIP = the +262 vendor-exact numerics block in qwen38_specdec.rs + the metal kernel additions), bmo held the same arc committed on `timer-build` (already in origin — the branch was resolved, not merged). Delegated-owner verdict `f9e6b2c2` (Claude, REVISE applied): salvage the real hunks (`c1bbbeb`), lint repair (`1be9401`), then this cell. The T1 run itself found the vendor prefix cache never written (`accept()` filled only the f32 cache — the vendor reads panicked at cycle 0); fixed `d83ae45`, G1 PASS on the fixed build.

**Verdict: T1 CLOSED — Path A CLOSES per the gate; Path B (riir-train head distillation, 4090, against OUR Q4_K_M target) is the beat lane.** The T1 issue's T2-T4 (verify pass, draft loop, e2e re-cell) are dead with Path A unless the owner re-opens on new evidence; T5 (the Path B plan) becomes the active lane and is an riir-train routing (this box does not route riir-train — the plan opens in riir-train by its owning session).
