# Bench 966 — Issue 1182 T5: the drift-guard league cell — retention byte-identical, fallbacks 0, sampler engaged, 2.0% pp2048 tail; promotion stays owner-gated (opt-in stands)

**Status:** LANDED (2026-11-10, amo-zed-1191metal@amo). The T5 GOAT cell
for the `drift_guard` feature (issue 1182) on the REAL league posture —
the qwen3.8-27b dense-f16 arm (`qwen38_wgpu,metal_tensor_gemm,drift_guard`,
release, isolated `/tmp/i1182t5`), model
`/Users/katopz/uzu27b/models/qwen3.8-27b-Q4_K_M.gguf` (64 layers = 48
GDN / 16 attn, the 946 record's model). Commit `110f242` + the
observability additions (the bench's end-of-process guard census line +
the seam's env-gated first-skip disclosure).

## Box state (the preflight law)

amo (M5 Ultra, 30 cores, macOS 27.2), **QUIET** per `agent-mail rig
status` (load 5.5→2.4/30 over the run, mem 76-81/96 GB free, disk 343 GB
free), **AC power**, no other GPU compute consumer; one process at a
time (GPU-exclusive correctness cells). The 2.0% tail is a MEDIAN ratio
of interleaved same-session cells, not a p99 — quoted as such.

## Cells

| cell | result | bar (issue 1182 T5) |
|---|---|---|
| Retention (559-token prompt → one real prefill chunk + 48-token walk) | BARE vs ARMED walks **byte-identical 48/48** (`diff` clean) | unchanged ✓ |
| Clean-arm fallbacks | **0** across both armed cells; zero latch lines | 0 on clean ✓ |
| Sampler engagement | pp2048: **124 samples / 8 shape families** (stride ratio ≈ 1/64 of the ~7.9k dispatches — the fixed stride verified in the ratio); walk cell: 8 samples / 4 families | engaged ✓ |
| Clean-arm divergence | **0.000e0 on every sample** (f32-readback resolution; see scope) | no false latch ✓ |
| Injected drift | latches in 1 sample + the exact fallback serves — proven by the release-posture PoC cell (`drift_guard_poc`, 2/2 green, rig job `01a1c261` on quiet amo) | fires ✓ |
| Tail cost | pp2048 ×3: BARE median **933 tok/s** (912/933/939) vs ARMED median **914** (909/914/918) → **armed/bare = 0.980, a 2.0% median prefill tail** at stride 1/64 | measured, bar NOT pre-pinned (see verdict) |

The short-prompt walk (8 tokens) sample count 0 is CORRECT — the ragged
tail routes token-by-token through gemv (the matmul2d K×1 law), never
gemm; the guard is prefill-gemm-only by design. The >512-token prompt is
the retention cell that exercises prefill.

## Verdict: PROVISIONAL PASS — stays opt-in; promotion owner-gated

- Retention + fallback + engagement + injected-fire bars all PASS on the
  league model at the league posture.
- **The tail bar was NOT pre-pinned** (my T5 text said "within a pinned
  bar" and pinned no number — a pre-registration miss, recorded here).
  Measured: 2.0% median prefill tail at stride 64. Cheaper strides are a
  const change (stride 256 ≈ 0.5% tail, 4× slower breach detection) — a
  calibration act with its own cell, never a runtime env.
- Per the flag discipline the feature STAYS opt-in
  (`drift_guard`, not in `production`; `RIIR_DRIFT_GUARD` promote-only).
  The default-posture flip is the owner's call (a serving-lane cost
  accepted against a runtime drift-coverage gain).

## Honest scope

- The 0.000e0 clean divergences are read at f32 resolution: sub-ULP
  relaxed-vs-exact deltas round to identical f32 rows. The guard's job
  (catch gross family-level drift at runtime) is unaffected; the 3e-3
  center holds ~30× over the synthetic tripwire magnitude (1e-2) while
  reading 0 on trained weights — no false-latch pressure.
- ONE model (qwen3.8-27b-Q4_K_M) at pp2048 + one 559-token family — the
  946 retention-scope caveat carries over verbatim: a different
  checkpoint re-runs the cell.
- The clean-arm divergences being 0 also means the cell does NOT
  calibrate `DIV_CENTER` against a real distribution — the synthetic
  tripwire + the fail-closed NaN law are the calibration posture until a
  drift-bearing cell exists (deliberately: fabricating drift on trained
  weights beyond the tripwire is not a measurement this lane needs).

## Follow-ups

- The `debug_skip` first-skip disclosure (env `RIIR_DRIFT_GUARD_DEBUG=1`)
  landed with the census line — the failure taxonomy (prod dispatch err /
  exact twin missing / exact dispatch err / readback mismatch) is
  visible, never a silent "guard idle" read.
- Latched-family + mixed-path-only posture: unchanged open items from
  the issue (mixed-path arming needs the same twin over `MSL_SOURCE`).
