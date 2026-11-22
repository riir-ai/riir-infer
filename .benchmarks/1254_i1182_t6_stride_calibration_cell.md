# Bench 1254 — Issue 1182 T6: the drift-guard stride calibration cell — stride 256 passes the pinned bar (0.9989 ≥ 0.990), stride 64 fails it (0.9840); the lane's calibrated posture is 256, the flip stays owner-gated

**Status:** LANDED (2026-11-22, sprint-071-idle@amo). The T6 calibration cell
(issue 1182's pre-registered flip path step 2) against the bar pinned in
step 1 (commit `1f3c0014`) — the deciding run for WHICH stride the opt-in
`drift_guard` lane carries. amo EXCLUSIVE rig job `01a204e6` (retry `01a2044e`
failed fast on the missing `-p riir-infer-gpu` in the cell script — the script
bug, fixed before the retry; no measurement was taken by the failed run).
Commit `110f242` lineage (T5's landing; the only delta this cell is the
`SAMPLE_STRIDE` const, flipped and restored by the cell harness itself).

## Box state (the preflight law)

amo (M5 Ultra, 30 cores, macOS 27.2), **EXCLUSIVE rig job** on an empty quiet
rig, **AC power**, `PROVENANCE: power=AC Power load=4.42 swap=698.81M
canary=skipped powermode=0(auto) ane=ok` — preflight PASSED (the ane axis
idle through the 3s window; the sgemm canary is not built on this box —
skipped, disclosed). One process at a time; no other GPU compute consumer.

## The pre-registration (pinned BEFORE the cell — `.issues/1182` T6, `1f3c0014`)

- **Bar: armed/bare pp2048 median-of-medians ratio ≥ 0.990** (tail ≤ 1.0%),
  ≥5 interleaved invocation pairs per arm per stride, each invocation the
  bench bin's own median-of-3 (`--pp 2048`, warm-up + 3 passes, padded to the
  ingest chunk — pure prefill).
- Per-cell carries: fallbacks 0 on clean + sampler engaged (the census line's
  stride ratio ≈ 1/SAMPLE_STRIDE) + retention walk byte-identical BARE vs
  ARMED (`[walk] ids:` line diff, 48 tokens over a varied text prompt).
- Stride selection law: the STRICTEST passing stride wins — stride 64 at
  equal pass (4× tighter breach detection); stride 256 only if 64 misses.
- Honest derivation note: the bar was pinned after Bench 966's measurements
  existed (2.0% @ 64 recorded, ~0.5% @ 256 estimated) but before the
  calibration cell — 1.0% adjudicates between them from the fleet's own
  ≥5%-class league-gate convention (an order of magnitude under), not from
  either number.

## Cells (5 interleaved BARE/ARMED pairs per stride, same session)

| posture | bare medians (tok/s) | armed medians (tok/s) | median ratio | bar ≥ 0.990 |
|---|---|---|---|---|
| stride 64 (HEAD const) | 913 · 938 · 935 · 936 · 934 → **935** | 915 · 921 · 920 · 917 · 920 → **920** | **0.9840** (tail 1.60%) | **FAIL** |
| stride 256 (const-flipped build) | 929 · 934 · 930 · 935 · 933 → **933** | 932 · 932 · 934 · 931 · 931 → **932** | **0.9989** (tail 0.11%) | **PASS** |

Model `/Users/katopz/uzu27b/models/qwen3.8-27b-Q4_K_M.gguf`, posture
`qwen38_wgpu,metal_tensor_gemm,drift_guard`, release, every invocation
rc=0 (20/20 prefill + 4 walks).

## Per-cell carries — all green at BOTH strides

- **Retention: byte-identical BARE vs ARMED** at stride 64 AND stride 256
  (the `[walk] ids:` 48-token greedy walks diff clean — 2/2).
- **Fallbacks 0** on every armed run (10/10 census lines read
  `fallbacks=0`, zero latch lines).
- **Engagement exact:** census `samples=124` @ stride 64 (8 shape families)
  vs `samples=31` @ stride 256 (the same 8 families) — the 1/stride ratio
  verified in the census itself (124 ≈ 4 × 31).
- **Clean divergence 0.000e0** on every sampled family at both strides
  (f32-readback resolution; the Bench 966 scope note carries over).

## Cross-session corroboration

The stride-64 bare medians (913–938, median 935) reproduce Bench 966's
(912/933/939, median 933) within noise; the stride-64 armed ratio 0.9840
(tail 1.60%) is consistent with Bench 966's 0.980 (2.0%) — the calibration
is reproducible across sessions twelve days apart on the same box class.

## Verdict: the lane's calibrated posture is SAMPLE_STRIDE = 256; the flip stays owner-gated

- **Stride 64 FAILS the pinned bar** (0.9840 < 0.990) — the tighter sampling
  costs more than the bar allows on this model/box.
- **Stride 256 PASSES** (0.9989 ≥ 0.990) with every per-cell carry green.
- The stride selection law decides: **the opt-in lane carries 256** — landed
  as the calibration outcome in the same commit as this bench (`SAMPLE_STRIDE`
  64 → 256; the feature stays opt-in, `RIIR_DRIFT_GUARD` promote-only
  unchanged; a breach at stride 256 is detected on average 4× later in the
  dispatch stream — the bounded-overhead tradeoff the bar was pinned for).
- **The default-posture flip remains step 3, OWNER-GATED** (the 2026-11-19
  delegated verdict's decision A): with this cell, the flip proposal is now
  evidence-complete — stride 256, tail 0.11% at pp2048, retention
  byte-identical, fallbacks 0, injected-fire proven (T5 + the PoC cells).
- Breach-DETECTION latency at 256 (the coverage half of the trade): at ~124
  → 31 samples per pp2048 pass, a family breaching on every sampled dispatch
  still latches within one pass at the served shape; the per-family sample
  counts (4–32 at stride 64) show every family samples at least once per
  pass at stride 256 except the smallest (0x300000001400: 1 sample — the
  known 4-dispatch family; a single sample still covers it once per pass).

## Honest scope

- ONE model (qwen3.8-27b-Q4_K_M) at pp2048 + one 48-token walk family — the
  946/966 retention-scope caveat carries over verbatim.
- The clean-arm divergences are 0 at f32 resolution (the calibration of
  `DIV_CENTER` against a real drift distribution remains the deliberate
  non-goal it was in Bench 966 — the synthetic tripwire + the fail-closed
  NaN law carry the calibration posture).
- The stride-256 bare arm was a FRESH build with only the const flipped —
  the compiler sees identical code otherwise (the const feeds the sampler's
  counter modulus only).
- Raw per-invocation logs were deleted with the scratch cell dir at teardown
  (`/tmp/i1182_t6/`); the 20 medians + census lines quoted above ARE the
  raw record (the bench file carries them in full).

## Follow-ups

- The owner flip call (step 3) now has its evidence package: this bench + the
  pinned bar + the landed stride-256 posture. Re-open trigger unchanged.
- If the owner declines the flip, the opt-in lane still carries 256 — the
  calibration is posture work, not promotion work.
- `DIV_CENTER`/`DIV_SCALE` calibration against a real drift distribution
  remains parked (deliberately — no drift-bearing cell exists for this lane).
