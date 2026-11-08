# Bench 956 — Issue 1177 item 2 owed cells: the 6× attribution, the measured ceiling, the GEMV engagement trace, and the packed ctx-2048 instability adjudication

**Date:** 2026-11-08, amo (M5 Ultra 96 GB), session zed-754da340@amo
**Posture:** release `qwen38_wgpu,production` build at `bbbe744`+ (the engagement instrument + probe example landed in the same commit), HEAD above `463e2ce`/`678bb9d`; model `/Users/katopz/uzu27b/models/qwen3.8-27b-Q4_K_M.gguf` (the Bench-1167 pin, 16.81 GB). Box: AC power, load 2.5–3.0 (light), exclusive GPU (co-tenant guard passed before every cell), `pmset powermode` n/a. Cells: `qwen38_wgpu_bench --decode 32 --decode-ctx 2048`, 2 fresh processes per dense arm (leg 1 carries `--walk-tokens 64`), 5 fresh processes for the packed instability leg. Claude verdict round 1 (REVISE, session 10bde1fe) demanded exactly these four cells before any verdict; they are now LANDED — see [`.issues/1177`](../.issues/1177_dense_lane_residues_cross_backend_parity_and_decode_gemv.md).

## 1. The 6× attribution (code vs box)

| arm | env over the league default | raw decode (2 legs) | ctx-2048 decode (2 legs) |
|---|---|---|---|
| A1 league default | — | 12.06 / 12.11 | 10.33 / 10.37 |
| A2 −Q6K tail | `RIIR_QWEN38_LMHEAD_Q6K=0` | 11.71 / 11.73 | 9.12 / 9.08 |
| A3 −Q6K −F16ACT | `…LMHEAD_Q6K=0 …DENSE_MM2D_F16ACT=0` | 11.73 / 11.73 | 9.15 / 9.13 |

- Raw decode is arm-INSENSITIVE (±0.4 tok/s): the raw cell rides the f16 FFN GEMV stream; neither the lm_head tail nor the act posture moves it.
- ctx-2048: the Q6_K packed tail is worth **+1.2 tok/s** (10.35 vs 9.10); F16ACT is worth **~0 at decode** — it is a prefill-GEMM context variant; decode never dispatches it (the engagement Δ is IDENTICAL 15,872 across A1/A2/A3, see §3).
- The 11-05 cell's A-arm numbers (11.67/10.02) reproduce at today's HEAD within noise (12.06–12.11 / 10.33–10.37 — the small raw-cell rise is the 11-04→11-08 decode-path changes, `27edb75`/`5ab9a51`/`d19223c`). The A side of the "6×" was never the problem.

## 2. The packed ctx-2048 instability — REFUTED (n=5)

Five fresh processes at the EXACT doubtable shape (B env `DENSE_MM2D=0 ATTN_TENSOR=0 F16ACT=0`, ctx-2048, decode 32):

| leg | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| raw | 9.02 | 8.93 | 9.08 | 9.01 | 9.02 |
| ctx-2048 | 7.85 | 7.92 | 7.86 | 7.85 | 7.86 |

Spread 0.07 tok/s (~0.9%) — **STABLE. The 11-05 "5.00 / 1.23" pair does not reproduce and is now attributed to the stale-posture class, not bimodality**: those runs predate `d19223c` (the lmhead+F16ACT league default, 11-05) — at that date the B env set was in flux (the demote interlocks landed with it), and the depth-coupled hypothesis (4 min packed prefill before the ctx cell) is retired with it: the fresh runs carry the same depth and do not split. Co-tenancy never needs crediting.

The REAL packed-vs-f16 gap at ctx-2048 is **10.35 / 7.86 = 1.32×** — matching the 1184 posture finding's byte-class arithmetic (34.2 GB f16 vs ~packed bytes over similar walls; B raw 9.0 lands in the finding's "packedOFF ≈ 8.6-9 ⇒ 951 measured f16" discriminator band, corroborating the Bench-951 artifact confirmation the 1184 bmo run is landing independently).

## 3. The GEMV engagement trace

Instrument added to the bench bin (the Issue-1124 T1.5 law: assert the arm, never infer from env): `[gemv-engagement] dense dispatches Δ+N over the cell` — the dense context's raw dispatch counter (`dispatch_count`, gemv+matmul2d+q4k classes) captured after each ingest, printed per decode cell.

- **Δ = 15,872 per 32-step cell = 496 GEMV dispatches/step**, identical across A1/A2/A3 and every leg (10/10 legs) — decode demonstrably rides `dispatch_gemv`; the F16 gemv arm's structural invariant (it panics without the dense context) plus a measured count closes the engagement ask. No all-zero-class hazard: the walks produce coherent text (§4).

## 4. The greedy-walk digest

`--walk-tokens 64` (default prompt): **A1 == A2 == A3 == B1 BYTE-IDENTICAL ids** (64 tokens, 4 postures — f16 vs packed Q4_K quantization classes produce the same greedy picks at this depth; the `27edb75` all-zero precedent class is excluded by construction). Full id lines in `/tmp/i1177_owed/*.log` (transient) and in the issue record.

## 5. The measured ceiling — `examples/gemv_read_bw.rs` (new instrument)

The league roofline (~810–819 GB/s) was ASSUMED. The probe times the ENGAGED decode kernel (`gemv_dense_f16`, one simdgroup per row, coalesced f16 reads, commit + `wait_last` per dispatch — the decode-class timing pattern) over a 2 GiB saturating shape (m=n=32768 f16), median of 10 after 3 warm-ups, 3 fresh processes:

| run | median GB/s | min | max |
|---|---|---|---|
| 1 | 715.5 | 611.9 | 759.5 |
| 2 | 748.5 | 737.0 | 755.4 |
| 3 | 699.4 | 567.6 | 734.9 |

**Measured decode-class read BW ≈ 700–750 GB/s** (median-of-medians 715.5; the min legs are first-alloc noise). Against the assumed 819 GB/s roofline that is **87–91% measured** — and the 1184 finding's derived ~720 GB/s effective for the f16 league decode lands INSIDE the measured band: **the f16 decode GEMV stream already runs at its kernel-class ceiling. There is no kernel-craft headroom on f16; the lever's retirement (1184 finding claim 1) is confirmed at the measured level.** The packed route's ceiling is a different (ALU) bound — its lever lives in 1178's lanesliced lane, not here.

## Verdict

Item 2's owed cells are ALL LANDED. The re-priced lever lane stays CLOSED per the REVISE gate (attribution + ceiling landed, but the 1184 posture finding already retired the lever at the byte-class level: f16 is bandwidth-bound at 87-91% measured roofline — the remaining ~10-13% is not a 7× lever, and the packed lane moved to 1178 T2b). The lane's residue reduces to **Item 1 only** (the cross-backend bit-parity G1 cell, 4090-queue-gated).
