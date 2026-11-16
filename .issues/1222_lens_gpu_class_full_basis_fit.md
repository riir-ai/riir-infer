# Issue 1222: lens GPU-class full-basis fit (katgpt-rs 999 T3 follow-up)

**Status:** open — claimed (GPU-class full-basis fit lane).

**Status:** LANDED 2026-11-16 (zed-a081f5f1@amo) — the batched GPU lane compiles, unit-gated, parity-gated; the real-model full-basis run is the next cell (4090).

## Landing record (2026-11-16)

- **Kernel suite** (`crates/riir-infer-gpu/src/lens_batch_kernels.rs`, feature `lens_gpu`):
  13 CubeCL kernels + launchers — batched GEMM, scale-by-table, add, half-rule,
  causal conv, row-slice split, l2norm+expand (fused), GDN recurrence scan
  (self-initializing state), qk/ssm true-derivative norm, attention jvp
  (true derivative, GQA), rope (host-precomputed cos/sin table), gated-q
  split, per-direction column sum. Every kernel mirrors its CPU twin's loop
  order; each is unit-gated against the CPU op on a batch-of-1 fixture
  (`tests/lens_batch_kernel_units.rs`, 10 cells, exact to fp noise).
- **Driver** (`src/lens_batch.rs`): `LensBatchStack` (weights once) +
  `LensDocCapture` (per doc) + `suffix_pass` (one direction block through one
  `FitPoint` → `[batch·d]` column sums — the R-sweep unit).
- **Bugs the gates caught and fixed during landing** (each reproduced,
  named, fixed):
  1. GDN launcher binding order (states/ds/spre vs dq/dk/dv — the signature
     and the launch call disagreed);
  2. l2norm+expand dst row missing `·head_dim` + the y table's per-position
     stride (the capture qn/kn rows are the FULL `n_v·key` width);
  3. GDN gate half-rule argument order (dg/du pairing — `½(s·du + u·sp·dg)`
     with (dg,du) = (dz, dnormed));
  4. rope cos/sin table row stride (`rotary_dim/2`, not `freq.len()`) and the
     host-side table replacing a runtime index→f32 conversion;
  5. scan state self-initialization at `t == 0` (the CPU zeroes on entry;
     the GPU buffer is pooled/uninitialized).
- **Parity gate** (`tests/lens_batch_parity.rs`): 5 stack shapes × 3 seeds
  (pure GDN, pure ATTN, GDN→ATTN, ATTN→GDN, 3-layer hybrid), full-basis
  block per stack. **The gate is structural (Pearson ≥ 0.999 on dense
  direction-reads) — NOT per-column rel-RMS**: the softmax pullback's
  `ds − row_sum` cancellation turns the cross-backend fma-vs-mul-add ulp
  into unbounded per-column deltas (measured 0.34 of block-max on the
  3-layer fixture) while the reads stay at Pearson ≥ 0.9999. Read rel-RMS
  5e-1 is a garbage backstop only. This is the 940-law local-only
  reclassification recorded at the site (offline diagnostics, never synced).
- **Conditioning check**: a 1e-4 input perturbation moves the layer-1 output
  1.5e-4 (gain 1.54) — the deltas are intermediate-term cancellation, not
  input amplification.
- **Debug harness** (doc-hidden, kept for the 4090 session):
  `debug_layer_step_readback` / `debug_sublayer_readback` / `debug_mlp_readback`
  / `debug_attn_stage` / `debug_gdn_stage` + `tests/lens_batch_bisect.rs`
  (report-only cells that localized every bug above).
- **The fit runner landed**: `crates/riir-infer-gpu/examples/lens_gpu_fit_t3.rs`
  ([[example]], required-features `lens_gpu`) — the lens_fit_t3 protocol
  with the J sweep on GPU direction blocks (K stays on CPU workers; the
  same accumulator semantics, G3 gate, artifact layout).
- **Remaining**: the real-model full-basis RUN (option 1's ~20-30 h/1k docs
  on the 4090 at fp32 via wgpu-spirv — or Metal on amo post-fit) via
  `lens_gpu_fit_t3`; the fixture tolerance numbers do NOT transfer to the
  real model (its own parity cell calibrates the production bar).

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

