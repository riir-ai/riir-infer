# Issue 1226: the f16-B lane's follow-ups (the T4 verdict's carried caveats)

**Status:** OPEN — filed 2026-11-16 at the T4 promotion (delegated-owner verdict AGREE; the caveats the verdict carried forward, none blocking).

**Claimed:** amo-idle-i1225f16b@amo, 1794848737, expires 2026-11-17T16:00:00+00:00
(scope: F1 only — the pp32768 retention row via the walk's own P axis, relaxed + strict, one scored length per family; F2/F3 stay open)

**Repos:** [riir-infer]

## Tasks

- [ ] **F1 — the pp32768 retention row:** the T1 walk's frame protocol tops at pp4096 (scored prefixes 2048/4095); the census's pp32768 row has no retention read. A breach demotes THAT family via `RIIR_PREFILL_F16B_FAMILIES`, never the whole route (the verdict's own fix path). Re-use `issue1225_f16b_retention_walk` with `I1225_PS=32768` (block_size sizing + scored_lengths K=1 at 32767) — the walk's env already parameterizes P.
- [ ] **F2 — producer-writes-f16:** the residual stream emitted as f16 alongside f32 at the producer, removing the per-launch `cast_f32_f16_zc` pass (0.15–0.19 ms/launch, Bench 1223) from the frame. The measured NET already includes the tax, so this only widens the win; it is a serving-shape refinement (the producer seams), not a kernel change.
- [ ] **F3 — the next re-gate carries a `PROVENANCE:` line** (the verdict's note: power mode was not recorded on the 1224/1225/1226 cells) — `bench_preflight`-class box state beside any new quoted number.

## F2 scoping notes (2026-11-16, read-only — no twin or producer code touched)

The engagement map for whoever claims F2 (the twin's cast tax is per-LAUNCH — encoder + dispatch overhead dominates the bytes — so the win is the launch count, not bandwidth):

- **The producer census (who writes each twin-routed site's input):** every B-side input is a scratch buffer written by a CubeCL elementwise kernel (f32), never a pool boundary:
  - `AttnQ`/`AttnKv` ← `normx_b` — last writer before the twin is the **rotate** (`prefill_rotate(normx_b, None)` = IN-PLACE, prefill_attention_layer_batched); the norm (`prefill_norm` → `RmsNormBatchedCubeCL`) wrote it earlier.
  - `GdnQkv`/`GdnZ` ← `rotx_b` (folded files) — written by **rotate** out-of-place (`normx_b → rotx_b`); on pre-rotation files they read `normx_b` directly.
  - `GdnA`/`GdnB` ← `normx_b` (the Issue-032 PRIMAL — always pre-rotate) — producer is **rmsnorm** alone.
  - `FfnGate`/`FfnUp` ← `normx_b` — the post-attn `prefill_norm` rewrote it, then an IN-PLACE **rotate** (last writer).
  - `AttnOutProj` ← `attn_out_b` — the **attention** kernel; `GdnOutProj` ← `rec_b` — per-head **RmsNormBatched** + zgate + in-place rotate; `FfnDown` ← `hid_b` — the **SwiGLU/activation** kernel (+ in-place rotate).
- **The shape:** a shadow write is required in TWO kernel families — **rmsnorm** (covers `GdnA`/`GdnB` + pre-rotation `GdnQkv`/`GdnZ`) and **rotate** (covers everything that reads post-rotate values, incl. in-place variants). Norm-only misses AttnQ/Kv and FfnGate/Up on folded models (their last writer is the in-place rotate — the norm's shadow goes STALE); rotate-only misses `GdnA`/`GdnB`. Both needed for full coverage; phase-gate the engagement (start with the pair, keep `AttnOutProj`/`GdnOutProj`/`FfnDown` on the cast — 8 of 11 families covered in phase 1).
- **Numerics are free:** the shadow must be written with the same RNE rounding the cast kernel applies (`half(x)` / `f16_round_bits` — the crate already carries the bit-exact emulation in `dq_fakequant_cubecl::f16_round_bits` and CubeCL kernels can write a packed-u32 f16 pair without a `half` element type, the `f16_bits` lane). Elementwise `shadow[c] == cast(x[c])` is then a BIT-IDENTITY cell, and Bench 1224's per-family retention numbers carry over by construction — re-run the tap on ONE posture to verify, not the full walk.
- **The dispatch seam:** `dispatch_f16b` grows a prebound variant (bind the shadow CubeCL handle at buffer(3), skip encoder 1) — the twin's `input_cache` (byte-size-keyed dedicated Metal buffers) is unused on that path; the route consults per-site whether the input handle IS the shadow (a registry beside `set_f16b_site_mask`, the same latch pattern). Incumbent paths (GEMV fallback, simdgroup, ANE splits, the tap's incumbent arm, every decode lane) keep reading f32 — never touched.
- **The gate order:** G1 bit-identity (shadow vs cast, elementwise, one shape) → price cell (cast-launch count before/after + e2e pp4096/pp32k, QUIET box, PROVENANCE per F3) → tap retention on one posture → GOAT vs the armed f16b lane (the incumbent is BENCH 1225's armed posture, not the f32 lane) → delegated-owner verdict; promotion is default-ON inside the dist lane (it composes with the existing demote mask). Do NOT start before F1's verdict lands — a family demote at pp32768 changes the phase-1 coverage set.
