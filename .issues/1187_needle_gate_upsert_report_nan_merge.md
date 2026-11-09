# Issue 1187 — needle_eviction_gate `upsert_report` cannot re-read its own mid-run file (delta_nll NaN→null)

**Status:** OPEN — found during the Bench 958 smoke cell (issue 1185 T3, 2026-11-09); pre-existing lane defect, not introduced by T3.

## The defect

`upsert_report` (needle_eviction_gate/main.rs) is designed so "a crash/restart loses at most the arm in flight": every completed arm is merged into `<out>/needle_gate_report.json` as the run proceeds. The merge cannot work past the first arm:

1. `ArmReport.delta_nll` is initialized `vec![f32::NAN; needles.len()]` at arm completion; the per-arm delta fill happens LATER (the run-end loop).
2. serde_json writes NaN as `null`.
3. On the next arm's `upsert_report`, the file is parsed back into `Vec<ArmReport>` — `delta_nll: Vec<f32>` has no null-tolerant deserializer → parse error → the loud "rewriting WITHOUT the old rows" branch fires → the file now holds ONLY the newest arm.

Net effect: during a multi-arm single-invocation run the incremental file never accumulates (every merge log line in the 958 smoke is this bug firing); the FINAL upsert (after the delta fill) writes the complete report, so completed runs are fine — but a crash mid-run loses every completed arm, the exact opposite of the function doc's claim. (`needle_nll`/survival fields carry `de_nan_f32` overrides; `delta_nll: Vec<f32>` is the one field left unwired.)

## Fix shape (one small change)

A `de_nan_vec` deserializer (null → NaN vec, mirroring `de_nan_f32`) on `delta_nll`, or serialize `delta_nll` only after the fill. The loaded-`full` guard path (`--full-from`) already refuses stale rows by name, so NaN deltas in a LOADED row are inert.

## Repro

Any multi-arm invocation with `--out`: the smoke cell at `.benchmarks/i1185_t3_smoke/` (issue 1185 T3) prints one merge failure per arm after the first.
