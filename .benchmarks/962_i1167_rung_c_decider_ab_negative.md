# Bench 962 — Issue 1167 rung C decider: the double-buffer staging variant wins 4/4 paired reps at 4.78 ms/frame = 0.85% e2e — UNDER the 1.5% bar → rung C closes NEGATIVE on the measured frame Δ (variant stays opt-in)

**Status:** RECORD (decider adjudicated; rung C closed per the pre-registered rule).
**Date:** 2026-11-09 (cell ran 2026-11-08 16:37–16:48 UTC on bmo)
**Session:** cell executed by bmo exclusive rig job `01a1bc05-8fb9-78c2-9ddc-d41f5ea19300` under the zed-amo-1167rungc@amo lease (fold owner zed-754da340@amo — both sessions nowake by 11-09); adjudicated by zed-15be441a@amo under the owner's nowake-takeover directive (lease folded in the issue).
**Issue:** [`.issues/1167_metal_prefill_followup_lane_b_c.md`](../.issues/1167_metal_prefill_followup_lane_b_c.md) (rung C, the decider half)
**Instrument:** the paired Bench-937 frame-census A/B (script `/tmp/i1167rungc_ab_job.sh` — the attempt-2 hardening: ABORTS LOUD exit 4 on any pass with no `(baseline)` line, closing the attempt-1 green-zero laundering class), gate sha `b490a9f2593c` (the G1b commit — the variant gate landed there: `set_metal_tensor_double_buffer` + the construction-time smem capture + `g1b_double_buffer_variant_gate` 4/4 PASS, worst_rel ≤ 4.228e-7, bit-identity to_bits EXACT).
**Box:** bmo (M5 Ultra 96 GB, AC), release census binary `bench_1146_prefill_frame_census`, exclusive lane, ~77 s/pass, 8 passes alternating.

## The pre-registered decider rule (Bench 949 / the issue)

> Paired Bench-937 frame-census A/B (3 alternating incumbent-first pairs + 1 reversed pair, order-bias probe), decider = ALL ms/frame vs the 500.86 reference; **≥7.5 ms/frame (1.5%) + G1 → GOAT lane via the `production` row; under → rung C closes NEGATIVE on the measured delta.**

## The cell (ms/frame, ALL; every pass rc=0; bit-identity anchors present in ALL logs)

| pass | arm | ALL ms/frame |
|---|---|---:|
| rep 1 (I-first) | incumbent | 565.30 |
| rep 1 (I-first) | variant | 560.59 |
| rep 2 (I-first) | incumbent | 564.67 |
| rep 2 (I-first) | variant | 560.80 |
| rep 3 (I-first) | incumbent | 565.45 |
| rep 3 (I-first) | variant | 560.21 |
| rep 4 (V-first, reversed) | variant | 560.52 |
| rep 4 (V-first, reversed) | incumbent | 565.37 |

- incumbent: 565.30 / 564.67 / 565.45 / 565.37 → **median 565.34** (mean 565.20, spread 0.78)
- variant: 560.59 / 560.80 / 560.21 / 560.52 → **median 560.56** (mean 560.53, spread 0.59)
- **Δ median = 4.78 ms/frame = 0.85% e2e** (Δ mean 4.67 = 0.83%); arm ranges DISJOINT (worst variant 560.80 < best incumbent 564.67); per-rep wall noise ±0.1% — the separation is ~8× the noise.
- Order-bias probe: the reversed pair (rep 4, variant-first) reads Δ 4.85 — consistent with the incumbent-first pairs (4.6–5.2); **no order artifact**.

## Adjudication

**UNDER the bar.** The ≥7.5 ms/frame (1.5%) decider bar is not met: the real pipelined exposure of the two per-K-tile barriers is **4.78 ms/frame (0.85% e2e)**, not the ~50 ms/frame fence-isolated projection Bench 949 priced. That projection was the fully-serialized UPPER bound by construction ("sync-fenced so a timed region cannot exclude work — the e2e projection applies them to the census's pipelined strip, which is the conservative direction"): in the real dispatch the other workgroups hide most of the staging serialization — exactly the hiding the G3 twins already measured in isolation (barrier exposure 1.5–2.2% of GEMM wall in situ vs the 10–12% isolated-probe share). The measured e2e is the launch-weighted projection discounted ~10.5× by pipelining. **Per the pre-registered rule, rung C closes NEGATIVE on the measured frame delta.**

Honest shape of the result: the variant is a small REAL win (4/4 reps, disjoint ranges, order-robust) — not noise, just sub-bar. It stays in-tree exactly as landed: G1b-verified (bit-identity to_bits EXACT, machine-verified at `b490a9f`), opt-in behind its promote-only env, **zero serving-path change** (never joined `production`; the construction-time smem capture from the G1b footgun fix is posture-neutral for the incumbent — it pins the captured value the incumbent already sized from). A future re-open needs the class's own named bar to move (e.g., the bar clearing via a different staging mechanism, not a re-run of this cell).

## The 949 projection vs the 962 measurement (recorded so nobody quotes 949's 10% as an e2e claim)

| quantity | value | class |
|---|---:|---|
| Bench 949 launch-weighted projection | ~50 ms/frame ≈ 10% e2e | fence-isolated UPPER bound (serialized probes) |
| Bench 962 paired in-situ A/B | **4.78 ms/frame ≈ 0.85% e2e** | the measured decider |
| discount | ~10.5× | the pipelining the projection itself named as the conservative direction |

## Provenance / lease chain

- Gate 1 (G1b): `b490a9f` + `77423da` (bit-identity EXACT machine-verified; posture-capture footgun closed root-cause).
- Gate 2 attempt 1 `01a1bbca`: VOID green-zero (relative model path onto an emptied bmo directory; loud SKIP laundered to exit 0 — the attempt-2 script hardening closes the class).
- Gate 2 attempt 2 `01a1bc05`: the cell above (absolute `BONSAI2_G1_MODEL`, abort-loud guard armed but not needed — every pass produced its `(baseline)` line).
- Fold owner `zed-754da340@amo` (wake armed 00:45 +07 11-09) never woke; the rig end-mail had no live recipient. Adjudicated 11-09 by zed-15be441a@amo (probe + owner takeover directive recorded in the issue).
