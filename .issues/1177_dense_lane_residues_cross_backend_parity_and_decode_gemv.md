# riir-infer Issue 1177 — dense-lane residues: the cross-backend bit-parity G1 cell (4090) + the parked dense-f16 decode GEMV lever

**Status:** Open — residue pointer filed 2026-11-05 when the parent lanes
closed (1150 + 1170; Claude verdict round 1 AGREE on the closure batch).
Two items the lane still owes, both priced/deferred, neither blocking a
landed surface. **Item 2's owed cells ALL LANDED 2026-11-08 ([Bench 956](../.benchmarks/956_i1177_owed_cells_attribution_ceiling_engagement_instability.md))** — the 6× was
stale-posture artifact (real packed-vs-f16 ctx gap 1.32×, stable n=5);
ceiling measured 700–750 GB/s (87–91% of the assumed 819); engagement
Δ+15,872/cell; walk byte-identical across 4 postures. Lever stays retired
(1184 finding + measured confirmation). Item 1 (4090 parity cell) remains.
**Lease closed:** zed-754da340@amo, 1794174540, was expires 2026-11-10T15:00:00+00:00 — holder session gone from amo's live-thread list (probe `01a1beaa` parked, no wake); under it Item 2's owed cells landed (Bench 956) and the lever closed — nothing left under that lease. **NOT re-claimed (2026-11-09, zed-15be441a@amo):** Item 1 is unblocked and ready — **the 4090 gate is OPEN** (verified 2026-11-09 ~05:1x +07 via Tailscale: 1% util, 731 MiB desktop-only, zero CUDA compute consumers; the filing-time "3 contenders" state is gone). Next executor claims per the lease law and runs the cell: build the cudarc forward at the league env set on the 4090, fixed walk on both backends, FNV/argmax pins registered here — **pre-register the f16-activation band from the f32 kernel-order precedent (`katgpt-rs closure/bridge_certified`) BEFORE the run, never post-hoc**.

Lease note (prior): Item 2 price cell on amo Metal; Item 1 stays 4090-queue-gated.
**Claimed:** zed-15be441a@amo, 1794201411, expires 2026-11-11T05:30:00Z
(scope: Item 1 — the cudarc walk-parity bin, the band pre-registration, the Metal+CUDA 48-token prose walk at ctx 2048, FNV/argmax pins registered here; the model is the bench-946 Q4_K_M, expected present on the 4090 at the probe bin's own F:/models default — sha256 verified against the amo copy before any run)

## Item 1 — the cross-backend bit-parity G1 cell (from Issue 1150's acceptance gates)

Issue 1150's G1 gate: Metal forward vs `qwen38_dense_cudarc.rs` (the
~7k-line CUDA twin, the bit-parity oracle) on the SAME qwen3.8-27B
Q4_K_M GGUF — FNV/argmax pins over a fixed token walk, tolerance band
set BEFORE the run (f32 kernel-order classes get the certified-precedent
treatment, never post-hoc bands). **Never run**: the lane's identity
evidence to date is Metal-vs-Metal (benches 938–946 walks digit-exact
within one backend) + the league demote byte-identity (bench 946). The
4090 is the only cudarc host and was BUSY (3 contenders) at filing.
Execute when the 4090 queue frees: build the cudarc forward at the
league env set, run the fixed walk on both backends, register the pins
(FNV + argmax) in this issue. Refusals/classes: the f16-activation
rounding order differs by construction across backends — pre-register
the band from the f32-class kernel-order precedent
(`katgpt-rs closure/bridge_certified`), never post-hoc.

## Item 2 — the parked dense-f16 decode GEMV lever (from Issue 1170 lever 2)

Decode on the dense-f16 posture reads ~1.9–22 tok/s class vs a ~15 tok/s
bandwidth bound per the 1150 audit framing (54 GB f16 at ~800 GB/s); the
`gemv_dense_f16` kernel is one simdgroup-per-row with strided reads,
while the Q6K tail's packed GEMV (bench 939) already proved the
saturating pattern (per-lane partials + one reduction). **Explicitly
parked non-serving in 1170**: the packed posture serves decode; this
lever matters only if the dense posture ever becomes the serving decode
posture or for parity rows. Price before any kernel work: the expected
ceiling is bandwidth-class (~15 tok/s), so the lever buys ~7× on a
non-serving surface — lead with a price cell, never the kernel.

### Item 2 price cell — FIRST READ + verdict round 1 REVISE (2026-11-05, zed-83e79a9e@amo)

**Status: PARKED pending attribution — the measurements below are real
but the code-vs-box split, the ceiling, and correctness proof are OWED
(Claude verdict round 1 REVISE, session 10bde1fe). Neither "~1.3×" nor
"79% of bound" is claimed.**

Cell ran on amo (M5 Ultra 96 GB, quiet, AC, exclusive GPU, co-tenant
guard passed), bench `qwen38_wgpu_bench --decode 32 --decode-ctx 2048`,
HEAD `d85ee80`+, 2 process runs per arm:

| cell | A: dense-f16 league default (`gemv_dense_f16`) | B: packed incumbent (`DENSE_MM2D=0 ATTN_TENSOR=0 F16ACT=0`) |
|---|---|---|
| raw decode 32-step | 11.67 / 11.67 tok/s | 8.76 / 8.82 tok/s |
| decode-ctx 2048 | 10.02 / 10.03 tok/s | **5.00 / 1.23 tok/s (UNADJUDICATED, n=2)** |
| load | 33.6 s (54 GB f16 materialization) | 9.9–11.1 s (16.8 GB packed) |

Witnesses: `[Issue 1150] dense-f16 matmul2d arm armed` + `[Issue 1170]
half-B armed` + `Q6_K packed tail armed (~2.1 GB resident)` (A) /
`[Issue 1176] DEMOTED` (B). Full logs `/tmp/i1177_price/` (transient).

**Corrections the verdict AGREED are recordable now:**
1. **The i1150 record's "~52 tok/s incumbent (issue-1153 cells)" is a
   mis-citation**: issue 1153 = Plan 623 Phase 3 (Bench 929) = the
   Bonsai ternary 2-box DIST decode row ("ours 2-box s1 52.28–52.59
   tok/s BEATS llama.cpp-RPC 2-box 48.6", amo⇄bmo TB wire, ~7 GB/token
   PQ2_0) — never the qwen3.8 packed single-box Metal decode (16.8
   GB/token). This cell's packed control reads 8.8 raw on amo.
2. **"The f16 posture must never serve decode" is stale post-1176**: the
   league default IS the dense posture (the demote interlocks prove it —
   `DENSE_MM2D=0` alone refuses: ATTN_TENSOR and F16ACT host on the
   dense context), so `gemv_dense_f16` decode is the serving Metal
   decode posture at HEAD.
3. **The 1.86/1.23 baseline is in doubt**: the decode path changed
   post-record (`27edb75`/`5ab9a51` 11-04 — the lm_head 10.7 GB f32
   tail → Q6_K packed GEMV ~2.1 GB + the `forward_token` decode loop;
   `d19223c` 11-05 made lmhead+f16act league default), and `5923713`
   resolved the 10-29 bimodality as a POSTURE split — the 1.86/1.23
   pair may not even be one posture. Co-tenancy cannot yet be credited
   with the 6×.

**OWED before any verdict or lever lane (verdict round 1 REVISE) — ALL LANDED 2026-11-08, [Bench 956](../.benchmarks/956_i1177_owed_cells_attribution_ceiling_engagement_instability.md):**
- [x] Attribution cell: A2 (`RIIR_QWEN38_LMHEAD_Q6K=0`) + A3 (also
  `F16ACT=0`) × 2 legs each — raw decode arm-insensitive (±0.4);
  ctx-2048: Q6K tail +1.2 tok/s, F16ACT ~0 at decode (prefill-only
  variant; engagement Δ identical). A1 baseline re-run at today's HEAD:
  12.06/12.11 raw, 10.33/10.37 ctx.
- [x] Measured ceiling: `examples/gemv_read_bw.rs` — the ENGAGED decode
  kernel over a 2 GiB shape, 3 processes: 715.5 / 748.5 / 699.4 GB/s
  median = **700–750 GB/s measured**, 87–91% of the assumed 819; the
  1184 finding's derived ~720 lands inside the band → f16 decode runs
  AT its kernel-class ceiling, no craft headroom.
- [x] Correctness proof: `[gemv-engagement]` instrument in the bench bin
  (dispatch_count delta per cell) — **Δ+15,872 per 32-step cell =
  496 GEMVs/step**, 10/10 legs; greedy-walk ids BYTE-IDENTICAL across
  A1/A2/A3/B1 (64 tokens, 4 postures).
- [x] Packed ctx-2048 instability: REFUTED at n=5 — 7.85/7.92/7.86/7.85/
  7.86 (spread 0.9%); the 11-05 "5.00/1.23" pair is stale-posture class
  (predates `d19223c`'s demote interlocks), never bimodality. Real
  packed-vs-f16 ctx gap = 1.32×; B raw 9.0 corroborates the 1184
  discriminator band (packedOFF ≈ 8.6-9).
- The re-priced lever lane stays CLOSED (the 1184 posture finding retired
  it at the byte-class level; §5 of the bench confirms at the measured
  level). Residue = Item 1 only (4090-queue-gated).

## Provenance

- Parent lanes closed 2026-11-05: riir-infer Issue 1150 (the dense
  forward arc, benches i1150/938–946), riir-infer Issue 1170 (lever 1 =
  half-B, bench 941).
- Claude verdict round 1 AGREE (2026-11-05): one residue issue covering
  both, always cited as "riir-infer Issue 1177" (riir-ai allocated its
  own 1177 the same day — always disambiguate cross-repo).
