# Bench 1224 — Issue 1225 T1: the f16-B per-family retention walk (the pre-registered numerics decider)

**Date:** 2026-11-16 · **Box:** amo (M5 Ultra, AC; load 2.5–4.6 across the four cells; no compute co-tenants — Zed exempt per the owner call, the rethink serve lane idle)
**Lease:** amo-idle-i1225f16b@amo (claim `a904584b`)
**Instrument:** `f16b_retention_walk` in `crates/riir-infer-gpu/tests/issue1225_f16b_retention_walk.rs` (new; the Issue-1146 lossy-surface precedent `issue1146_relaxed_kl.rs` widened per GEMM family via the matched-input tap)
**Pre-registration:** frozen in `.issues/1225` at commit `5776cf99` — BEFORE the twin landed (T2's twin + the tap route landed `wip` → this cell's build); floors never re-typed.
**Posture:** the serving default — tensor zerocopy (`RIIR_DIST_METAL_TENSOR` default ON), attn-tensor arm ON, KTILE64; the twin rides it with the B-side f16 patch + the per-launch `cast_f32_f16_zc` fallback. Model: the Bonsai-2-27B PQ2_0 league pin (`Ternary-Bonsai-2-27B-PQ2_0.gguf`). Frame width 128.

## The instrument (what makes the numbers attributable)

- **Matched-input tap:** at every twin-routed site of the tapped frame (each scored pass's final, deepest-context frame), the SAME input dispatches through BOTH the twin (feeds the trajectory) and the incumbent f32 body (tap-only scratch — never the real output). Per token row: argmax-over-m flip; per instance: worst scale-relative drift (strided ÷16). Family attribution is input-matched by construction — trajectory divergence never smears it.
- **Placebo control (HARD):** the twin compiled UNPATCHED — 8 controls (4 families × 2 postures × their first scored length) all read **KL 0.000e0, flips 0, worst_rel 0.000e0** → the instrument (cast pipeline, scratch dispatch, shared-queue ordering, census) adds exactly nothing; every armed number below is attributable to the lever.
- **Engagement (the Issue-1124 T1.5 lesson, counter-enforced):** twin dispatch total == incumbent served-site total at the same mask — **41472 == 41472 (P=512) and 73728 == 73728 (P=4096), both postures**. A twin fallthrough would read T < R. Zero-instance sites (gdn a/b, `Other`) are the model stack's own topology (the folded stack's a/b projections ride the deltanet prework lane, not `prefill_project`; `Other` is never named) — the incumbent never served them either (T==R proves it), so they are notes, not gaps.

## The pre-registered floors vs the measured worst per cell

| gate | floor | relaxed P=512 | strict P=512 | relaxed P=4096 | strict P=4096 |
|---|---|---|---|---|---|
| e2e KL max (worst family, nats) | ≤ 1e-3 | **3.221e-6** (zh) | **3.856e-6** (zh) | **5.973e-7** (zh) | **9.172e-8** (zh) |
| e2e top-1 | ≥ 15/16 | 32/32 | 32/32 | 8/8 | 8/8 |
| e2e ΔNLL mean | ±0.05 | worst +6.8e-4 | worst +1.3e-4 | worst +4.0e-4 | worst +1.1e-4 |
| per-family flip rate | ≤ 1e-3 | **5.10e-4** (gdn_z) | **3.21e-4** (attn_kv) | **4.08e-4** (gdn_z) | **3.68e-4** (gdn_z) |
| per-family worst_rel | ≤ 1e-2 | **6.06e-4** (ffn_gate) | **3.12e-4** (attn_kv) | **5.47e-4** (attn_kv) | **3.42e-4** (attn_kv) |
| placebo | == 0 exact | PASS ×4 | PASS ×4 | PASS ×4 | PASS ×4 |
| engagement | EQUAL | 41472==41472 | 41472==41472 | 73728==73728 | 73728==73728 |

## Per-GEMM-family retention (the demote axis — every family survives)

Flip rate = flipped token rows / compared rows at the tapped frames; worst_rel = worst scale-relative drift (strided ÷16) per instance, max over instances.

**Relaxed P=512 (8 scored lengths × 4 families; rows 40512–162048 per family):**

| family | instances | rows | flips | rate | worst_rel |
|---|---:|---:|---:|---:|---:|
| attn_q | 512 | 40512 | 3 | 7.4e-5 | 4.61e-4 |
| attn_kv | 512 | 40512 | 14 | 3.46e-4 | 5.75e-4 |
| attn_out_proj | 512 | 40512 | 3 | 7.4e-5 | 2.74e-4 |
| gdn_in_proj_qkv | 1536 | 121536 | 23 | 1.89e-4 | 4.60e-4 |
| gdn_in_proj_z | 1536 | 121536 | 62 | **5.10e-4** | 4.81e-4 |
| gdn_out_proj | 1536 | 121536 | 34 | 2.80e-4 | 4.07e-4 |
| ffn_gate | 2048 | 162048 | 34 | 2.10e-4 | **6.06e-4** |
| ffn_up | 2048 | 162048 | 48 | 2.96e-4 | 4.55e-4 |
| ffn_down | 2048 | 162048 | 53 | 3.27e-4 | 4.23e-4 |

**Strict P=512:** the same shape, ~2–3× tighter (worst rate 3.21e-4 attn_kv; worst_rel 3.12e-4 attn_kv; ffn_down 1.36e-4 / 1.75e-4).

**Relaxed P=4096 (2 scored lengths × 4 families):** worst rate 4.08e-4 (gdn_z), worst_rel 5.47e-4 (attn_kv); attn_out_proj 0 flips. **Strict P=4096:** worst rate 3.68e-4 (gdn_z), worst_rel 3.42e-4 (attn_kv); attn_out_proj + attn_q near-zero rates.

Flip samples (bounded, 8/site) show every flip at a near-tie: incumbent top-1 margins at flips read 1e-7–4e-4 — the twin reorders near-ties exactly as the tolerance class predicts, never a confident pick.

## Verdict: T1 PASS — no family demotes; the route graduates to T3

Every pre-registered floor held in every cell; no family approaches its demote bar (worst 5.1e-4 vs the 1e-3 bar — headroom ~2×; worst_rel 6.1e-4 vs 1e-2 — ~16×). The f16-B lever's numerics on the deployed path are the **same class as the already-promoted Issue-1146 relaxed rung** (per-family KL max 3.2–5.9e-6 relaxed / ≤ 3.9e-7 strict nats here vs 3.83e-6 there). The whole 9-family surface graduates together:

- **T2 (twin) is DONE in the same engagement** — the serving twin (`MetalTensorZeroCopyGemm::new_f16b` + `dispatch_f16b`: cast→gemm→blit in ONE command buffer; `dispatch_scratch` + `dedicated_output_for` for the tap; `F16_BODY_PATCHES` canonical in-module with the occurrence-count drift guard) landed with the route (`prefill_f16b` feature, default-OFF; `RIIR_PREFILL_F16B=1` PROMOTE-ONLY; `RIIR_PREFILL_F16B_FAMILIES` the per-family demote mask; `RIIR_PREFILL_F16B_TAP`/`_PLACEBO` the instrument arms) — engagement proven in situ by the counters above, not assumed.
- **Next: T3** — the paired Bench-937 frame-census A/B on the twin at the promoted posture (`TWIN_RELAXED` both arms), ≥1.5% e2e NET + this T1 → the T4 GOAT gate.

## Honest caveats (carried, not buried)

- **Corpus amendment (pre-registered, `5776cf99` + the walk's doc):** the frozen 1146 texts encode ~412–700 tokens; at P=4096 the walk feeds each family's tokens CYCLED (`cycle().take(p+2)` — the bench_948 corpus-construction freedom). The axis under test is numerics, not task skill; the content stays frozen.
- The tap reads only the final frame per scored pass (the deepest-context rows — where the committed pick lives). Shallow-context rows are covered by the P=512 protocol's shorter prefixes; a full-frame tap census is a re-run knob (`set_f16b_tap` arms any window), not a standing cost.
- `worst_rel` is a strided (÷16) subsample per instance — the full-element drift class was pinned by Bench 1223's G5 cell (1.69–1.96e-4 at the fixture level); the stride keeps the tap's host loop bounded and is disclosed.
- The league rows measured here are P=2048/4095 scored prefixes (P=4096 protocol) and the P=512 family — the census's P=32768 row is NOT walked (the dist lane's frame protocol tops at the scored lengths; a pp32k retention row is a T3 re-run knob if the frame A/B wants it).
- The cast tax is INSIDE the armed trajectories (the twin's per-launch fallback) — the retention numbers include the rounded-input reality; a producer-writes-f16 serving shape would only tighten the drift, never loosen it.

## Provenance

- Four cells: relaxed/strict × P=512/4096, one process each, `--release`, `--test-threads=1`, `CARGO_TARGET_DIR=/tmp/i1225f16b`. Logs: `/tmp/i1225f16b_walk_{relaxed,strict}{,512}.log` (copied into the bench dir).
- Loadavg at cells: 4.39→4.06 (P=4096 relaxed), 3.09→4.11 (P=4096 strict), 3.30→2.68 (P=512 relaxed), 2.64→2.51 (P=512 strict). No thermal throttle; no GPU co-tenants (`ps` census at start).
- The P=256 development smoke (68 passes, the instrument shakeout) is NOT part of this record — its numbers were discarded when the frame-loop base bug was fixed (the enumerate-index vs token-offset bug made every multi-frame prefix's final-frame detection fail; found by the dbg run, fixed, re-run — the smoke's post-fix run PASSED with engagement 23040==23040 before the record cells ran).
