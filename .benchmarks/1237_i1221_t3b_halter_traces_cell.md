# Bench 1237 — i1221 T3(b): the halter traces on the looped-trained pin

**Status:** COMPLETE — verdict **THE STEP RATIO IS THE COST-FLOOR ANSWER: the trained pin's per-position gain decay (step_3/first ≈ 0.33–0.36, step_4/step_3 ≈ 0.35–0.42, zero reversals) puts the Phase-2 wiring default (frac 0.01) at NO-halt-for-trained-depth (the correct answer — the checkpoint itself uses all 4 loops), and the synthetic G2 crowd band (0.5–0.8) at a uniform depth-3 halt (8/8 positions × 3 prompts, zero Oscillation fires). Plan 304's open cost_floor calibration is now answered on REAL looped-trained traces, in the named answer format.** Report-only; the only gates are the lib fixtures (12/12) + the lineage/resume receipts.
**Date:** 2026-11-17 (cell 12:52–12:58Z amo, clean lineage + resume receipt; instrument + fixtures + cell + record one session under the T3(b) claim `zed-84603cf7@amo`, epoch 1795910400).
**Scope:** riir-infer only. Lane: `.issues/1221` (T3(b) fold rides this record); pre-registration: `.plans/636` (read-order leg 3, after (a)); consumer: katgpt-core `gain_cost_halt` (Plan 304) — the answer format named in the record's §Answer format.

## The cell

```
cargo run --release --features dory \
  --example i1221_t3b_halter_traces -- \
  --gguf ../riir-train/data/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf \
  --out .benchmarks/i1221_t3b_halter_traces_cell
```

One lens per prompt (the (a) seam verbatim — `run_lens`, k=4 snapshots), then `halter_traces` derives per-position step/cos sequences from the snapshots and drives **katgpt-core's own halter kernel** (consumed via the `dory` feature's new `katgpt-core/gain_cost_halt` forward — never reimplemented; Plan 304 defaults tau 1.0, patience 1, l_min 1) under six cost postures: `frac ∈ {0.01, 0.1, 0.25, 0.5, 0.6, 0.8}` × the position's first step (the Phase-2 `cost_floor = frac × first_loop_step_size` wiring).

- Pin: `NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf` blake3 `f946e530…` (the (a) pin; the (a) prompt law verbatim — same LCG, seed 0, 3 sets × seq 8).
- STAMP: `ab946b0e` at the original cell time (the peer's laya P43-row landing; my files dory-side, disjoint). ⚠ **Artifact note:** the original landing commit (`613916c4`) captured a HEADER-ONLY rows file (a write/commit race — the rows flushed after the stage); the artifact was re-derived with `--fresh` at the landing-fix commit and the traces verified BIT-IDENTICAL to the record's tables (p0 pos0 7.090→2.455→1.028, frac-0.25 halt depths {4}×24). The re-derive header stamps `613916c4` — same pin, same determinism receipt.
- Rows: 97 lines = header + **72 trace rows** (3 prompts × 8 positions × depths 2–4: the raw `step_size`/`cos_theta` per (pos, depth)) + **24 calibration rows** (3 × 8: `first_step` + 6 posture cells each). Committed artifact.

## Provenance

- `PROVENANCE: power=AC Power load=4.85→12.44 swap=4302M canary=skipped powermode=0(auto)` (bench_preflight PASS at admission; the cell ran in the 4.85 window — structural read, no timing gates; wall context only per the `96b44752` law).
- Cell wall: ~3 lens runs ≈ the (a) cost (103 s lens-dominated); the halter walk itself is negligible (8 positions × 3 depths × 6 postures of scalar math).
- **Resume receipt:** a second run verified the lineage (9 = 3 prompts × the trace-row aux keying) and skipped all — `p0/p1/p2: done — skipped`, 0 new rows, exit 0. **This required a real fix:** the (b) cell initially reused (a)'s record-name-hardcoded scanner, so the resume guard never matched and two resume runs APPENDED duplicate lineages (291-line file, 3× every trace row). Root cause: the scanner filtered on the literal `"i1221_t3a_kl_spectra_row"`. Fix: `verify_resume_lineage` takes the caller's `row_record` param (both runners + the fixture updated); `--fresh` re-ran the cell into the clean 97-line lineage. The (a) rows file was never touched (its lineage is intact — the bug was (b)-runner-side only).

## The traces (the loop's gain shape, pos 0 per prompt — every position tells the same story)

| prompt | d2 step (= first) | d3 step (× first) | d4 step (× d3) |
|---|---|---|---|
| 0 | 7.090 | 2.455 (0.346×) | 1.028 (0.419×) |
| 1 | 8.156 | 3.078 (0.377×) | 1.383 (0.449×) |
| 2 | 7.413 | 2.766 (0.373×) | 1.223 (0.442×) |

**cos θ ∈ [0.80, 0.94] at every (position, depth ≥ 3)** — the update direction is strongly self-aligned, never negative: the trained loop is a smooth contraction, the oscillation detector is silent by construction on this checkpoint. The gain decay is two-phase decelerating (×~0.35 then ×~0.42 — NOT the constant-geometric ladder Plan 304's synthetic regimes modeled, and NOT concave-monotone in the Issue-698 sense either: the ratio INCREASES with depth, 0.35 → 0.42).

## The posture table (the answer, 24 positions × 6 fracs — position-uniform across all 3 prompts)

| frac (× first_step) | cost at pos0/p0 | verdict (all 24 positions) | halt reason |
|---|---|---|---|
| 0.01 (Phase-2 default) | 0.071 | C C C — runs to trained depth 4, 0/24 early-halt | — |
| 0.1 | 0.709 | C C C — 0/24 early-halt | — |
| 0.25 | 1.772 | C C H — 24/24 halt AT the trained depth (d4 < floor) | GainBelowCost |
| 0.5 | 3.545 | C H . — 24/24 halt at depth 3 | GainBelowCost |
| 0.6 (synthetic G2 band) | 4.254 | C H . — 24/24 halt at depth 3 | GainBelowCost |
| 0.8 | 5.672 | C H . — 24/24 halt at depth 3 | GainBelowCost |

### The answer for katgpt-core (Plan 304's open question, stated)

1. **The wiring default is CORRECT for a looped-trained checkpoint at its trained depth** — 0.01 × first_step never halts inside the trained k (the checkpoint's own loop count IS the "uses all loops" posture; Bench 1236's KL-collapse-to-0-at-k=4 is the quality half of the same fact). The G3-style no-spurious-halt contract holds on real traces, not just the synthetic 0.95-decay ladder.
2. **The synthetic G2 crowd band (0.5–0.8) transfers** — it halts at depth 3 on ALL 24 real positions, saving exactly 1 of 4 recurrent passes (25% compute) with Bench 1236's d=3 KL residual (0.03–0.78 nats mean, vs 3–4 nats at the dense untrained floor) as the quality price. The Plan-304 synthetic finding ("0.01 too conservative for the crowd tier; override to 0.5–0.8") is CONFIRMED on real looped-trained data.
3. **The real trace is two-phase decelerating (ratio 0.35 → 0.42), not constant-geometric** — the synthetic G2 gates' decay-constant model (0.3/0.5/0.7 flat) is an approximation: on a real trained loop the SECOND step decays less than the first, so a floor calibrated on constant-decay traces over-estimates late-loop savings slightly. Any cost-floor policy the civ-side wires should read the two-phase shape, not the geometric mean. The full per-position `verdict_shape` strings are in the rows for exactly this diff.
4. **Oscillation: never fires on a trained loop** (cos θ ≥ 0.80 everywhere) — the oscillation early-halt remains a synthetic-regime instrument (corrupt/adversarial states), not a trained-checkpoint mechanism. Its G4 gate stays kernel-tested, not data-calibrated.

## The answer format (what katgpt-core gets — the peer directive)

`HalterCalibrationRow` (serde, in the rows artifact + the module API): per (prompt, position) — `first_step` (the depth-2 step: the episode scale), `k_full`, and one `HalterPostureCell` per cost posture: `{cost_floor_frac, cost (= frac × first_step), halt_depth, verdict_shape (one char per depth: C/F/H/. — length-normalized with post-halt padding), halt_reason}`. The consumer diffs `verdict_shape` against its synthetic reference regimes and reads `halt_depth` off the real band without re-running anything. The raw `HalterTraceRow`s (`prompt_idx, pos, depth, step_size, cos_theta`) ride beside them so any future posture set can be re-evaluated offline — the traces are the data, the postures are a view.

## Files

- `src/dory/spectra.rs` — `HalterTraceRow`/`HalterCalibrationRow`/`HalterPostureCell` + `halter_traces` (one trace pass, fresh kernel per posture) + the `verify_resume_lineage` generalization (`row_record` param — the resume-scan record-name fix, both runners updated).
- `Cargo.toml` — `dory = ["katgpt-core/gain_cost_halt"]` (the forward; unarmed builds unchanged — the halter kernel rides the existing katgpt-core dep tree, zero new deps) + the `[[example]]` row.
- `examples/i1221_t3b_halter_traces.rs` (new) — the resumable cell runner.
- `tests/i1221_t3a_spectra_gates.rs` — +3 fixtures (12 total): trace-derivation determinism + episode-scale pin, posture-cell wellformedness (shape/reason/depth agreement — the answer format's structural law), and the Plan-304 kernel contract re-pinned on the synthetic reference regimes (crowd halts 2–3, important never within trained depth).
- Postures: clippy −D warnings clean at default / no-default / dory-all-targets; lib 367 green at `dory,deltanet_inference`; gates 12/12.
- Artifact: `.benchmarks/i1221_t3b_halter_traces_cell/rows.jsonl` (97 lines, clean lineage, resume-receipt).

## Session

zed-84603cf7@amo (amo, M5 Ultra, AC) — routing receipt from zed-78e08ea6 (same as (a): not theirs while i1201 holds bmo). **Lease FOLDED at the landing** (the issue carries the `**Lease closed:**` row; the claim line stays as lineage).
