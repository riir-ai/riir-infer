# Bench 1227 — Issue 1167: the post-f16B census re-rank — the FFN share holds ~49% WITH f16-B armed, gdn_rec 6.0%, the pp512 −4.9% control confirms the config rung, and the promoted-posture (prework-fused) table

**Status:** RECORD — census re-rank PASS ×2 (pre-fusion table + the promoted-posture fused table, both exit 0, engagement `twin 264192 dispatched / incumbent 0 served` asserted, non-vacuous bit-identity anchors held, G-C1 additivity residue −3.54 / +1.59 — noise-class). Re-ranks Bench 937 post-f16B: the f16-B twin (promoted dist default 2026-11-16, Bench 1226) armed for every GEMM site in both runs — the residual table at the serving posture for the first time. **The re-rank verdict (§The promoted-posture re-rank): gdn_rec 6.9% is the top remaining kernel ground (priced, needs its own engagement+price cell); the pp512 config rung −6.7% at the promoted posture (fully priced); FFN 54.9% → the 1226 lane's f16-B follow-ups; attention closed; prework fused 15.7× re-validated.** Both of the issue's named remaining grounds now carry numbers at the promoted posture — the lane's remaining ground is PRICED, not open.
**Date:** 2026-11-16
**Session:** zed-ddc61e9c@amo, 1794849000
**Issue:** `.issues/1167_metal_prefill_followup_lane_b_c.md`
**Box:** bmo (M5 Ultra, 96 GiB), AC, Exclusive lane (GPU-quiet by construction), load 1.4–2.9/30
**Model:** `Ternary-Bonsai-2-27B-PQ2_0.gguf` (league pin, absolute `BONSAI2_G1_MODEL` = `/Users/katopz/models/…`)
**Pin:** bmo worktree at `04667ea` (the instrumented-census commit; the instrument fixes `db0dfce` — see §Instrument — landed mid-bisect and both runs carry them)
**Jobs:** pre-fusion `01a1e5d4-…` (400 s, exit 0) · fused `01a1e5dc-…` (355 s, exit 0)

## The instrument arc (why three earlier reds were false alarms)

The first re-rank attempt aborted on its own engagement assert reading "routed 0 / total 264192" — three bisect cells (minimal probe ENGAGED; probe + census envs ENGAGED; full census REPRODUCED 0) pointed inside the census body, and the `I1146C_DIAG` per-chunk instrumentation (`04667ea`) caught it: **the assert had destructured `f16b_route_dispatch_counts()`'s (incumbent, twin) tuple backwards** — positional `let (routed, total)` named element 0 "routed" and asserted it > 0, but element 0 is the INCUMBENT's counter, which is 0 *by design* when the twin serves every site (the twin arm returns before the incumbent dispatch). The failed job's line was **twin 264192 fully engaged**. The substrate, the route latch, the envs, and the pins were all exonerated by the bisect before my own instrument confessed; the 1226 lane's walk-engagement counter model (T==R via the tap) was the datum that made the inversion visible. Fixed `db0dfce` (named destructure; the assert reads the twin).

Second (pre-existing, caught by the same diagnostics): the census's `run_pass` final-frame check compared `base + chunk_toks.len()` instead of `base * chunk + chunk_toks.len()` — `is_final` NEVER fired, `last_bits` never collected, and every anchor since the digest was added (Issue 1159) had been **vacuously green** (FNV offset-basis digest + argmax −inf = empty bits). The same enumerate-index class the i1226 walk fixed this morning. Fixed in the same commit; the vacuity warning + diagnostics stay in the instrument behind `I1146C_DIAG` (unset = byte-identical output).

## Posture

Bench-937's promoted+attn census posture **plus the f16-B twin route**: `I1146C_TENSOR=1`, `RIIR_METAL_TENSOR_RELAXED=1`, `RIIR_PREFILL_ATTN_METAL_TENSOR=1`, `I1146C_CHUNK=512 I1146C_FRAMES=8` (pp4096), `I1146C_F16B=1` (the substrate opt-in route arm added by this lane, pinned before `::new` per the construction-order law; engagement asserted via the route dispatch counts). Features = the `[[test]]` row + `metal_tensor_gemm` + `prefill_f16b`. 13 postures × (2 warmup + 5 timed). **Run A (§The census)** carries the pre-fusion prework (the `[[test]]` row's posture — the fused feature not in the set); **Run B (§Fused table)** adds `deltanet_prework_fused` = the PROMOTED serving posture (both runs' ALL walls are therefore not comparable to each other — that delta IS the fused-prework contribution at this posture, disclosed, not a lever).

⚠ Engagement (both runs, asserted): `twin 264192 dispatched / incumbent 0 served` — every one of the 264,192 prefill-GEMM dispatches routed the f16-B twin; the incumbent f32 body served zero sites. Per-family admission = ALL.

## Run A — the census (ms/frame, 8×512, tensor-zerocopy, f16-B ON, prework CHUNKED)

| posture | wall | Δ (class cost) | share | 937 (f32-B, pre-fusion) |
|---|---:|---:|---:|---|
| -ffn | 272.99 | 264.57 | **49.2%** | 50.8% |
| -deltanet | 319.03 | 218.53 | 40.7% | 39.3% |
| -attention | 488.15 | 49.42 | **9.2%** | 8.9% |
| -gdn_prework | 467.74 | 69.82 | 13.0% (chunked; fused in dist) | 12.2% |
| **-gdn_rec** | 505.12 | **32.44** | **6.0%** | 5.8% |
| -gdn_post | 528.56 | 9.01 | 1.7% | 1.7% |
| -embed | 528.74 | 8.83 | 1.6% | 1.7% |
| -gdn_beta | 535.74 | 1.83 | 0.3% | 0.3% |
| -gdn_residual | 536.81 | 0.75 | 0.1% | 0.1% |
| -gdn_norm | 537.16 | 0.40 | 0.1% | 0.1% |
| -gdn_rotin | 537.25 | 0.31 | 0.1% | <0.1% |
| -tail | 537.81 | −0.24 | ~0 | ~0 |
| **ALL** | **537.56** | baseline | — | 565.03 |

G-C1 coarse additivity: 541.11 of 537.56 attributed; residue −3.54 (noise-class). **Wide-chunk control: 4×2048 = 0.999 ms/token vs 1.050 at 512 (−4.9%)** — the pp512 config-class rung's ~5% bound CONFIRMED at the post-f16B posture (937: −5.0%; unchanged — the config slice is f16-B-independent, as a dispatch/config class should be).

## The fused table (Run B) — the PROMOTED posture (prework fused + f16-B ON)

Job `01a1e5dc-…` (355.49 s, exit 0, engagement asserted by the script's own greps). **The `-gdn_prework` Δ collapsing 69.82 → 4.45 = 15.7× is the fused arm's engagement proof** (the 937 pattern's 14.9×, reproduced with f16-B armed) — and **ALL 537.56 → 472.26 ms/frame (−12.1%)** is the fused-prework contribution at this posture:

| posture | wall | Δ (class cost) | share | Run A (chunked) Δ |
|---|---:|---:|---:|---:|
| -ffn | 212.97 | 259.29 | **54.9%** | 264.57 |
| -deltanet | 318.68 | 153.58 | 32.5% | 218.53 |
| -attention | 421.84 | 50.42 | 10.7% | 49.42 |
| -gdn_prework | 467.81 | **4.45** | 0.9% (**15.7× fused**) | 69.82 |
| **-gdn_rec** | 439.84 | **32.42** | **6.9%** | 32.44 |
| -gdn_post | 464.85 | 7.41 | 1.6% | 9.01 |
| -embed | 464.89 | 7.37 | 1.6% | 8.83 |
| -gdn_residual | 470.03 | 2.23 | 0.5% | 0.75 |
| -gdn_norm | 470.89 | 1.37 | 0.3% | 0.40 |
| -gdn_rotin | 471.27 | 0.99 | 0.2% | 0.31 |
| -gdn_beta | 472.37 | −0.11 | ~0 | 1.83 |
| -tail | 472.25 | 0.01 | ~0 | −0.24 |
| **ALL** | **472.26** | baseline | — | 537.56 |

G-C1 additivity residue 1.59 (noise-class). Side observation reproduced (937's own): the `gdn_out_proj` submission wall 1762.40 → 219.72 ms/frame-total — the chunked chain's queue backpressure, gone (submission walls are not exec time, demonstrated again). Wide-chunk control at the promoted posture: 0.860 vs 0.922 ms/token = **−6.7%** — the config-class slice's share GREW as the fusion shrank the denominator (was −4.9% pre-fusion).

## The promoted-posture re-rank (the verdict both runs exist for)

1. **gdn_rec 6.9% (32.42 ms/frame) is the largest non-FFN GPU class in the promoted posture** — the recurrence scan is the lane's top remaining KERNEL ground, priced. Its own engagement+price cell (the lane's price-gate rule) is the next kernel-class act, before any kernel is written.
2. **pp512 config-class rung: −6.7% wide-chunk at the promoted posture** (−4.9% pre-fusion) — the slice is real, bounded, and now measured at BOTH postures. Config-class (chunk-width policy / frame overlap), below kernel work in priority per the issue's own ranking, but fully priced.
3. **FFN 54.9%** — the residual's home WITH f16-B armed; the remaining FFN levers are the 1226 lane's f16-B follow-ups (F2 producer-writes-f16 cast-tax removal; the per-family retention decider). Not this lane's ground.
4. **attention 10.7%** — stays closed (937's verdict; even 2× moves e2e ~5%).
5. **gdn_prework 0.9%** — fused, closed, its promotion re-validated at the f16-B posture (15.7×).

## Re-rank vs Bench 937 (the issue's remaining ground, priced)

1. **gdn_rec 6.0% (32.44 ms/frame), was 5.8%** — the recurrence scan is now the largest NON-FFN kernel class after the prework fusion (in the promoted denominator ~472 ms it reads ~6.9%). This is the lane's named remaining kernel ground; it needs its own engagement+price cell before any kernel work (the lane's price-gate rule).
2. **pp512 config-class rung CONFIRMED at ~5%** — the wide-chunk control's −4.9% per-token is the direct read; the small-stage slice (norm/rotin/beta/post/residual/embed ≈ 21 ms/frame ≈ 3.9%) is its floor. Config-class (chunk-width policy / frame overlap), bounded, below kernel work in priority — but now MEASURED at the post-f16B posture, so the rung is fully priced: both of the issue's named remaining grounds carry numbers.
3. **FFN 49.2% is the residual's home WITH f16-B armed** — the twin already reclaimed the B-side bandwidth term; the remaining FFN levers are the f16-B follow-ups owned by the 1226 lane (F2 producer-writes-f16 removes the ~0.15–0.19 ms/launch cast tax; per-family retention = Bench 1224's decider). NOT this lane's ground — the lanes do not double-cover.
4. **attention 9.2% — stays closed** (937's verdict unchanged; even 2× moves e2e ~4.6%).
5. **gdn_prework 13.0% in Run A is a POSTURE ARTIFACT, not open ground** — the fused feature is promoted in the dist lane; Run B's fused table is the serving posture. Disclosed so nobody quotes Run A's prework row as a lever.
6. **Posture gap, carried honestly**: Run A lacks `deltanet_prework_fused` (the `[[test]]` row's posture), so its absolute walls run ~65 ms/frame above the promoted posture. Share RANKING is orientation-sufficient (every Δ reads the same frame); Run B pins the promoted absolute table.

## Instrument debt paid (stays in the census)

- The engagement assert now names `(incumbent, twin)` and asserts the twin — the tuple's order is a contract, never a position.
- `run_pass`'s final-frame check uses `tok_base = base * chunk` for both the offset and the is_final comparison — real anchors, digest `fnv d859da039f3767ed argmax 220 (4.7761)` at the ALL posture (vs the FNV offset basis + −inf the vacuous era printed).
- `I1146C_DIAG=1` (latch/admit dumps at four checkpoints, per-chunk route+counter+readback prints, vacuous-anchor warning) — unset keeps the run output byte-identical; the vacuity warning prints unconditionally when diag is on.
- The bmo-side red-run evidence: `.benchmarks/i1167_postf16b/census_postf16b.log` (the false-alarm abort), `census_rerank_full.log` (Run A), `census_rerank_fused.log` (Run B).

## Provenance

- Run A: bmo exclusive job `01a1e5d4-bbc8-75fd-a34e-f9374407c4a9`, 400.22 s, exit 0, load 1.4–2.9/30, no compute co-tenants (the script's ps census), AC.
- Run B: bmo exclusive job `01a1e5dc-8bae-7669-aa5f-a6459f921faf`, 355.49 s, exit 0, quiet, AC.
- The three false-alarm reds: job `01a1e525-…` (the original), the interactive bisect2 run, both at the same posture — all read the incumbent's zero against an engaged twin.
- Session: zed-ddc61e9c@amo (the i1167 claim lease, epoch 1794845186).
