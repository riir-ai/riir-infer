# Issue 1222: lens GPU-class full-basis fit (katgpt-rs 999 T3 follow-up)

**Status:** open — claimed (GPU-class full-basis fit lane).

**Claimed:** zed-a081f5f1@amo, 1794829271, expires 2026-11-17T22:21Z

## Context

katgpt-rs Issue 999 T3 fits the corpus-averaged R/S lenses (Research 666)
on the real 0.8B checkpoint. The PoC-source protocol fits the R map `J` on
the **full d_model basis** (readable in ANY direction later, forever). The
CPU measured wall (riir-infer `0f796e6c` pilot, qwen3.5-0.8B T=128, 48
points, d=1024, 24 threads ≈ 21 effective cores):

- capture 4.2 s, prefix pass 4.2 s, suffix pass 2.0 s (single-threaded)
- full-basis R sweep = 48 × 1024 suffix passes/doc ≈ **25 core-hours/doc**
- 1k docs ≈ 2.9 CPU-years — dead on any CPU box in this fleet
- the PoC's own reference (~3 h / 1k docs) ran on an H100 (GPU)

The CPU lane landed the amendment: K (S lens) stays full-basis (the cheap
half, ~1024 prefix passes/doc); J (R lens) fits `--r-dirs N` top-corpus
token-detector directions (the G2 steering-write class). What is LOST is
read generality: a NEW R-read direction at depth needs a re-fit.

## What this issue wants

The full-basis fit restored, GPU-class. Options in cost order:

1. **dim_batch batched sweeps on the 4090 (CUDA)** — the tangent ops in
   `src/lens/*` are per-op linear maps; batched over B basis directions
   the projections become real GEMMs (the FLOPs are unchanged — this is a
   throughput lane, not an FLOP lane). The GDN recurrence jvp/vjp scans
   are sequential over T but batch over B cleanly (per-head blocks, heads
   never share state). The attention jvp/vjp batch over B the same way.
   Est. 2.3 PFLOP/doc at TF32 ≈ 80 TFLOPS → ~8 h for 1k docs — the
   issue-999 "hours-class" figure. Kernel surface: `lens/ops.rs` matvec
   twins + `lens/gdn.rs` scan + `lens/attn.rs` maps + the stack
   prefix/suffix drivers. Gates: the existing FD/adjoint/parity lens
   suite must pass at the batched posture (bit-exact vs the CPU
   one-direction reference on one real doc is the acceptance bar).
2. **Metal via wgpu/CubeCL on the M-series** — same math, this box; the
   M5 Ultra's AMX/tensor path makes 1k docs a ~1-2 day lane.
3. **dim_batch on CPU only** — batches directions into GEMM-shaped passes
   (better cache + vectorization), est. 3-5× over the current
   one-direction-at-a-time matvecs; still ~days for 1k docs. Only worth
   it if (1)/(2) are blocked.

## Constraints

- Never a serving path (the lens is offline diagnostics).
- The fit result must be the SAME estimator as the CPU lane (seed window
  [16, T−1), T-mean, doc mean) — bit-exact against the CPU reference on
  a shared doc set is the G1 bar.
- Feature-gate the lane (`lens_gpu` or an env under `lens`); the release
  set does not change.

## Provenance

- CPU lane + amendment: riir-infer `0f796e6c` (feat(lens): T3 harness),
  katgpt-rs Issue 999 T3 record.
- PoC source: `stdlogits/neuron-lens` @ `849bfe1678e28becceb1e3864afe5fc8297d02e4`
  (their transport.py runs d_model basis sweeps in dim_batch blocks on GPU).

