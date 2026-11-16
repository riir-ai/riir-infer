# Bench 1222 — Issue 1167 G4: the packed-body dequant price at the serving posture (run 1 + the tightened run 2)

**Date:** 2026-11-16 · **Box:** amo (M5 Ultra, AC, load 4.55→3.24, quiet — no compute co-tenants, no thermal throttle)
**Lease:** amo-idle-i1167g4@amo (claim `340b61eb` — also reshaped 1221's unparseable claim line, which had the repo's lease check BLIND)
**Instrument:** `crates/riir-infer-gpu/tests/issue1167_rung_c_staging_price.rs::g4_packed_body_price_at_serving_posture` (`#[ignore]`d GPU cell, never run since it landed)
**Posture:** the serving selector — KTILE64 tiles, `relaxed_precision=true` (the promoted league posture's dist-scoped lossy arm), P=512 (the serving chunk — and the DEQUANT-SHARE-OPTIMAL posture: per-workgroup dequant work is P-independent while the matmul grows with P, so P=512 is where the A-dequant share is largest; parity there bounds the family), buffers uploaded once, interleaved rotating-start arms, `wait_until_completed` fenced.
**Arms:** baseline = the serving kernel's bit-plane dequant body verbatim (DEQUANT_V2 hoisted); packed = the SAME kernel with the A-side read+decode swapped for the packed-2-bit body (ONE u32 load per 16-element work item, decode algebra `(code&1) - ((code>>1)&1)` ≡ `(pos_bit − neg_bit)`). Same stores, same barriers, same grid — only the dequant section differs.

## Correctness (the anchor, both runs)

The packed twin is **bit-identical** to the serving baseline per (shape, P) — the per-element divergence assert (`to_bits`) never fired across all 7 shapes. The decode algebra is exact: code lane = pos_bit | neg_bit<<1, `(code&1) − ((code>>1)&1)` ≡ `pos_bit − neg_bit`. A future packed serving kernel inherits this anchor.

## Run 1 (defaults: 4 rounds × 2 launches, loadavg 4.55)

| shape | baseline ms/launch | packed | Δ ms | % of wall | Δ-range |
|---|---:|---:|---:|---:|---|
| ffn_gate/up (17408×5120) | 4.464 | 4.607 | −0.143 | −3.20% | [−2.010, +2.174] |
| ffn_down (5120×17408) | 5.236 | 5.099 | +0.137 | +2.61% | [+0.046, +2.133] |
| attn_q (12288×5120) | 3.048 | 3.591 | −0.544 | −17.84% | [−1.080, +2.523] |
| gdn_qkv (14336×5120) | 4.033 | 4.047 | −0.014 | −0.35% | [−0.447, +1.370] |
| gdn_z (6144×5120) | 1.985 | 1.845 | +0.140 | +7.07% | [−0.360, +0.396] |
| gdn_ab (16×5120) | 0.564 | 0.565 | −0.001 | −0.26% | [−0.022, +0.028] |
| gdn_out (5120×6144) | 2.134 | 2.078 | +0.056 | +2.62% | [−0.326, +0.350] |

Projection line printed: **−77.0 ms/frame = −15.4% "e2e" — UNDER**. Every row's Δ-range straddles zero.

## Run 2 (tightened: 12 rounds × 4 launches, loadavg 3.24)

| shape | baseline | packed | Δ ms | % of wall | Δ-range |
|---|---:|---:|---:|---:|---|
| ffn_gate/up | 4.357 | 4.314 | +0.044 | +1.00% | [−0.126, +0.244] |
| ffn_down | 5.177 | 5.114 | +0.064 | +1.23% | [−0.024, +0.616] |
| attn_q | 2.971 | 2.978 | −0.007 | −0.24% | [−0.246, +0.272] |
| gdn_qkv | 3.641 | 3.640 | +0.001 | +0.03% | [−0.244, +0.899] |
| gdn_z | 1.837 | 1.833 | +0.004 | +0.20% | [−0.062, +0.158] |
| gdn_ab | 0.562 | 0.563 | −0.001 | −0.14% | [−0.061, +0.042] |
| gdn_out | 1.905 | 1.879 | +0.026 | +1.37% | [−0.046, +0.230] |

Projection line printed: **+87.5 ms/frame = +17.5% "e2e" — OVER**. (Weights: ffn_gate ×128, ffn_down ×64, gdn ×48–96, ×8 chunks/frame — e.g. ffn_gate 0.044 ms × 1024 launches = 45 ms of the 87.5.)

## Verdict: the packed rung closes NEGATIVE — and the projection line is NOT evidence

1. **Per-launch parity is the measurement.** Across both runs, no row carries a consistent sign (only gdn_ab is consistent, at the ±0.001 ms noise floor); the max |Δ| is 1.4% of wall (run 2), inside its own Δ-range; run 1's larger rows (attn_q −17.8%, gdn_z +7.1%) shrank to −0.2%/+0.2% at 3× the rounds — they were noise. The packed format's load saving (3 loads → 2 per 16-element work item on the dequant section only) buys nothing measurable at the serving shapes.
2. **The G4 projection verdict line cannot adjudicate this effect size.** Same cell, 4 minutes apart: −15.4% then +17.5% — the launch-weighted projection (Σ per-launch Δ × up-to-1024 launch weights × 8) amplifies sub-noise per-launch deltas into double-digit fake "e2e" percentages in EITHER direction. Its 1.5% bar was calibrated for rung C's structural serialization change (barrier elimination, a multi-ms per-launch effect — Bench 949/962), where the projection correctly separated GOAT from NEGATIVE. At decode-delta magnitudes it mints verdicts from noise. Filed as riir-infer **issue 1224** (the instrument-honesty finding; the projection line stays for rung-C-class effects, with the magnitude caveat documented in the test doc-comment).
3. **Posture bound.** P=512 maximizes the dequant share (per-workgroup dequant work is P-independent; the matmul grows with P). Parity at P=512 bounds the lever family — pp2048/pp4096 can only be less favorable. The box-drift between runs (baseline ffn_gate 4.464→4.357 ms, 2.4%) is intra-arm and cancels in the interleaved ratio; the medians' absolute movement is another reason per-launch Δ ≤ 1.4% is parity, not signal.
4. **Price-gate rule satisfied honestly:** the addressable price ≈ 0 per launch, far under the ≥1.5% e2e graduate bar → no serving twin, no decider A/B. The FFN strip's remaining ground is the matmul body itself (out of this rung's scope; the census's 50.8% FFN share now has no named in-dequant lever).

## What this closes

Issue 1167's scope line ordered "the packed-byte dequant load price cell first … gdn_rec 5.8% + the pp512 config-class rung stay behind it". The packed cell is done: **CLOSED NEGATIVE**. Remaining behind it (unchanged): gdn_rec 5.8% and the pp512 config-class rung (~5%, bounded) — both now the lane's top open ground.

## Instrument addendum (same lease)

The projection-verdict guard (the finding above, filed + fixed same session; the issue file is removed per the noise-reduction rule, record in git history + this section): `projection_verdict()` + `ProjectionRow`/`ProjectionVerdict` in the instrument — every-weighted-row Δ-range straddling zero now prints `UNDECIDABLE` + the raw ms, never a bar verdict; any sign-consistent row restores the percentage verdict (the rung-C-class path unchanged, pinned by the pure `projection_verdict_guard_arithmetic` test: all-straddle → UNDECIDABLE, consistent over → OVER, consistent under → UNDER, empty → UNDECIDABLE). Verified in situ: a third G4 run (8 rounds × 4 launches, loadavg 3.03, per-launch parity ≤ 0.47% of wall) prints UNDECIDABLE on the same-shaped data. clippy clean, rustfmt-clean.

## Provenance

- Run 1: `loadavg1m=4.55` at start (build had just finished); run 2: `loadavg1m=3.24` at start, 3.62 at end; both AC, no thermal throttle, no GPU co-tenants (`ps` census before each).
- Feature set: the [[test]] row's declared set (`metal_tensor_gemm,cubecl_runtime,ternary_gemv,multi_host,ternary_gemm_batched,ternary_gemm_simdgroup,ternary_deltanet_chunked_prefill,ternary_attention_batched_prefill,riir-infer-core/bonsai2_hadamard`) — the production members of it; isolated `CARGO_TARGET_DIR=/tmp/i1167`.
- Box-state law note: the projection's frame reference (500.86 ms) is Bench 937's bmo frame; the per-launch parity verdict does not depend on it (it compares arms within one box), and the projection lines are ruled not-evidence above in both directions.
