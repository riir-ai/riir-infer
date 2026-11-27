# Bench 1274 — i1247 T2b: the CUBECL_WGPU_MAX_TASKS A/B + CELL I cadence cell (m3)

**Date:** 2026-11-27T18:25Z → 19:06Z (arms), probe re-run pending
**Box:** m3 (M3 Max 16-core, macOS 27.2.1), wgpu<msl> / vendored cubecl-wgpu 0.11.0-pre.2
**Claim:** riir-infer `.issues/1247` line 5 — zed-33fc5329@m3 (logical lane now zed-025d5169@m3), exp 2026-11-29T16:56Z
**Model:** /Users/kat/uzu27b/models/qwen3.8-27b-Q4_K_M.gguf (16.81 GB, Nov-11) — the bench default path is the amo home; m3 mirror passed explicitly

## Verdict

**NEGATIVE — no promotion.** Lifting the scheduler cap `tasks_max` 32 → 512 (`CUBECL_WGPU_MAX_TASKS=512`) moves the pp512 median wall **292.31 s → 285.88 s (−2.2%)** — inside run-to-run noise (arm A's own pass spread ±28% on pass 1; arm B spread ±0.7%). The 32-task flush cap is **not** the wall-clock bottleneck at the pp512 serving posture. T1's re-attribution (the ms-class "write" cost is the per-flush submit+commit chain amortized onto write arms) stands as a *counter* re-attribution, but the absolute layer's cost is **not flush-frequency-governed**: simply lifting the cap does not recover wall time. `CUBECL_WGPU_MAX_TASKS` stays at its default (unset = 32); the incumbent posture is kept. The m3 absolute-layer hunt continues on T1's remaining candidates (wgpu task/poll registry scan, Metal per-CB resource handling, commit scheduling).

## Arms (pure prefill pp512→512, ctx 24576, chunk 512, median of 3 after warm-up)

| arm | posture | warm-up | pass 1 | pass 2 | pass 3 | MEDIAN |
|---|---|---|---|---|---|---|
| A | cap default (32) | 226.21 s | 373.20 s | 292.31 s | 286.34 s | **292.31 s ≈ 1.75 tok/s** |
| B | CUBECL_WGPU_MAX_TASKS=512 | 292.53 s | 287.70 s | 285.88 s | 289.64 s | **285.88 s ≈ 1.79 tok/s** |

- Run-to-run honesty: arm A pass 1 (373 s) is a +28% outlier vs its own passes 2–3 (±1% each) — first-pass JIT/driver-state effect; arm B is tight (±0.7%). Medians are the read; delta −2.2% is noise-class.
- Both arms print "MEDIAN 2 tok/s" (integer rounding hides the 1.75 vs 1.79 distinction; use the seconds).
- Context posture note: arms ran at the bench default ctx 24576. The 1203 posture law maps ctx 4096 as the m3 minimum (24576 costs 8.7×); this A/B intentionally holds posture constant to isolate the cap variable. A ctx-4096 re-cell is NOT implied by this verdict (the cap non-effect should transfer, but it is unmeasured there).

## PROVENANCE

- Preflight (18:25Z, before arm A): `PROVENANCE: power=AC Power load=4.12 swap=1126.44M canary=skipped powermode=2(high) ane=ok` — PASSED (ceiling 6.0)
- Arm A window 18:25:05Z→18:46:25Z: start load 5.23/4.68/5.59 — clean throughout
- Arm B window 18:46:27Z→19:06:48Z: start load 4.47/4.78/4.88 — clean until ~19:05:45Z when sibling build lanes (s109wp31-sge worktree + s250-settings-stage bevy debug build) exploded load to 99; arm B pass 3 tail (+~20–60 s overlap) is the only touched read: 289.64 vs 285.88 = +1.3%, does not move the median. Caveat disclosed; median (pass 2) clean.
- GPU exclusivity: a seal-viewer premise probe (PID 77682, Plan 114 lane) held the GPU until 1:24 local; arms started after its exit — no co-tenant during any bench pass.
- canary=skipped: sgemm_shape_timing example not built on this box (preflight warn, not a gate for this cell — wall-clock A/B, no FLOP claim).

## CELL I (cadence) — CONTAMINATED FIRST RUN, CLEAN RE-RUN PENDING

CELL I ran 19:06:50Z→19:06:59Z **under load 99.51/74.89/41.80** (the sibling-build explosion began ~19:05:45Z, before the probe started). Its absolute numbers (per-pass wall 5764 µs; task-flush submit 3252.9 µs/pass over 96 submits; write-flush submit+poll 1490.7 µs/pass; REGULATE drain 96.8 µs/pass) are CPU-starvation-inflated and are DISCARDED as a measurement. Structure-only observation (to be re-read clean): the 32-task flush chain and the 64-write flush chain are the same order at this shape — consistent with the cap non-effect above. Clean re-run (2.4 s of bench time) lands in this record's addendum once load < 6.

## What this cell does NOT claim

- No decode-path claim (decode_steps=0; the arms are pure prefill).
- No claim that tasks_max=512 *hurts* (−2.2% is not a regression signal either; it is noise).
- No upstream-report implication: the negative caps the *cap* hypothesis only; T1 candidates (a)/(b)/(c) remain open.
