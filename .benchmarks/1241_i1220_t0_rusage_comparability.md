# Bench 1241 — issue 1220 T0 results: rusage CPU-seconds kill-read + the CER comparability matrix (amo job `01a1dfde`)

**Status:** LANDED 2026-11-19 (amo-idle-i932@amo) — the results of the T0 harness run (committed `a68e4735`) collected from the rig, independently verified, adjudicated. T0 CLOSED: V1 alive on fleurs (28% CPU reduction vs 1-thread), the fleurs quality advantage SURVIVES like-for-like scoring under BOTH contracts; the tvspeech ANE arm never completed (hang-shaped, zero rows — i1219 posture). Wall numbers on these rows are UNUSABLE (load-waived at load 25–29); only the CPU-seconds axis and the CER columns are load-tolerant.

## Posture

- Rig amo (M5 Ultra 96 GB), 2026-11-15 22:42–22:53 +07, amo job `01a1dfde` (shared lane), harness `.benchmarks/1220_p633_ane_asr_stage_tables/harnesses/p633_t0_rusage.py` @ `a68e4735`, one arm per PROCESS.
- Every row carries `load_waived: true` at load 25.16–28.83 — the preregistered T0 tolerance (CPU-seconds per audio-second is load-tolerant; wall RTFx is NOT and is recorded only as run-health context, never quoted as perf).
- Artifacts preserved from the volatile `/tmp/p633_t0` into `1220_p633_ane_asr_stage_tables/t0_run/` same-commit: the cells jsonl, the six per-clip rusage jsonl, the five hypothesis bundles.
- ANE decode recipe = streaming-560ms (`p633_p23` loop verbatim); ONNX = piecewise-dictation (`p633_p1.greedy_decode_long` verbatim) — the recipes deliberately differ (that IS the two serving postures being compared), the scorer matrix disentangles the scoring half.

## The cells (from `p633_t0_cells.jsonl`, verbatim)

| arm | family | n | audio_s | cpu_s (user+sys) | cpu_per_audio | cer_prep | cer_raw | note |
|---|---|---|---|---|---|---|---|---|
| onnx1t | fleurs-th | 100 | 1193.34 | 70.256 | **0.0589** | 22.69 | 24.07 | threads=1 |
| onnx1t | tvspeech | 100 | 2320.08 | 240.366 | **0.1036** | 27.09 | 30.36 | threads=1 |
| onnx4t | fleurs-th | 100 | 1193.34 | 335.164 | 0.2809 | 22.69 | 24.07 | hypotheses ≡ onnx1t (100/100) |
| onnx4t | tvspeech | 100 | 2320.08 | 664.914 | 0.2866 | 27.09 | 30.36 | hypotheses ≡ onnx1t |
| ane | fleurs-th | 100 | 1193.34 | 50.402 | **0.0422** | 9.06 | 10.75 | enc 52.2 s / dec 17.7 s / mel 0.36 s wall; host-side CPU only (ANE device compute is not CPU time — that IS the V1 measurement) |
| ane | tvspeech | — | — | — | — | — | — | **ARM HUNG — zero clip rows, 0-byte clip jsonl**; the job was timeout-killed inside this arm |

## T0 answer 1 — the V1 kill-read (CPU-seconds per audio-second)

- **fleurs: ANE 0.0422 vs CPU 1-thread 0.0589 = 0.716× — ANE uses 28.4% LESS host CPU than the 1-thread lane.** The preregistered kill read ("if the ANE posture does not use materially less CPU than the 1-thread lane, V1 is dead") therefore does NOT fire on fleurs: the headroom-release value case is ALIVE, with 28% as its measured magnitude.
- Vs CPU 4-thread: ANE uses 6.7× less CPU than onnx4t (0.0422 vs 0.2809 fleurs) — the co-tenant-friendly posture by a wide margin, which is the axis T2's additive arm prices.
- **tvspeech: UNMEASURED — the ANE arm hung before completing any clip** (T0 answer 3). The kill-read is adjudicated per-family: fleurs PASS, tvspeech NO VERDICT (blocked by i1219, not by a measured loss).

## T0 answer 2 — the comparability matrix (the open question RESOLVED)

Both arms scored under BOTH contracts on the same hypotheses:

| family | ONNX prep | ONNX raw | ANE prep | ANE raw | Δ prep (onnx−ane) | Δ raw |
|---|---|---|---|---|---|---|
| fleurs-th | 22.69 | 24.07 | **9.06** | **10.75** | **13.63** | **13.32** |
| tvspeech | 27.09 | 30.36 | — | — | — | — |

- **The fleurs quality advantage SURVIVES like-for-like under BOTH scorer contracts** (13.6 pts prep / 13.3 pts raw). The previously recorded 22.69-vs-10.75 comparison was scorer-mismatched in FORM (prep-scored ONNX row vs raw-scored ANE row), but the SUBSTANCE survives with a slightly larger like-for-like margin — the ONNX row's own prep/raw spread is +1.38, far smaller than the 13-point gap.
- The P1.1 NeMo reference (prep 9.29 fleurs) sits within 0.23 pts of the ANE arm's prep score (9.06) — the ANE streaming posture is scoring AT its NeMo reference under the prep contract, consistent with the parity-green conversion.
- **Verdict for the value-case gate: the per-family CER advantage is REAL on fleurs** — "a ~13-point better transcript at comparable host-CPU cost" is a genuine rung, and per the issue's quality-conditioned kill line this alone is a value-case input the owner weighs.
- tvspeech comparability: BLOCKED at the ANE arm (hang) — ONNX-side rows exist (27.09/30.36); the family's ANE posture stays i1219-gated.

## T0 answer 3 — the ane/tvspeech arm hang (i1219 cross-ref)

- The harness ran ane/fleurs to completion (100 rows, written 22:53 +07) then started ane/tvspeech: the clip jsonl was created (22:53) and stayed **0 bytes** — zero clips completed before the job was killed at its rig timeout. No panic, no error row.
- Same posture family as the two recorded i1219 reproductions (tvspeech + ANE sync inference through the `p633_p23` loop). Clip-level determinism is NOT established here — the harness writes rows only on clip completion and no in-flight index reached the log (the job log is not retained) — so this is recorded as a **hang-shaped failure at the same posture**, a third consistent observation, not a confirmed same-clip determinism row. Noted on i1219 same-commit.

## Independent verification receipt (this landing)

- Re-scored from the PRESERVED hypotheses with a from-scratch char-level Levenshtein (space-strip), corpus-length-weighted: **onnx1t 24.07 / ane 10.75 raw** — reproduces the harness cer_raw digits exactly.
- onnx1t vs onnx4t hypotheses byte-identical 100/100 (greedy decode is thread-count-invariant, matching the CPU-lane finding "CER thread-invariant"); references identical across all three arms (same slice manifest).
- cpu_per_audio recomputed from the per-clip rusage sums: ane fleurs Σ = 50.402 s over 1193.34 audio-s = 0.04223 — matches the cell.

## Consequences

- **T0 CLOSED** (both halves): V1 alive on fleurs; comparability resolved — the fleurs ANE quality advantage survives both contracts; tvspeech ANE blocked by i1219.
- **T1 stays owner/ops-gated** (the powermetrics sudoers entry — recorded on the issue; opens the moment it lands).
- **T2 stays window-gated**: it needs an exclusive window (bmo is committed to the i932 contract through ~2026-11-19; amo shared carries lane work) AND should NOT attempt the tvspeech ANE arm until i1219 resolves — its additive arm can run fleurs-only, or tvspeech on the ONNX arms + co-tenant measurement only.
- The value case so far, on fleurs: 28% host-CPU release at 0.97× the 1-thread wall (load-waived, indicative) with a ~13-point BETTER transcript — three non-dominated axes; the owner weighs speed-vs-quality per family after T1/T2.

## Provenance

amo job `01a1dfde` (shared, load-waived rows 25.16–28.83), 2026-11-15 22:42–22:53 +07; harness rev `a68e4735`; artifacts `1220_p633_ane_asr_stage_tables/t0_run/`; adjudication + independent re-score amo-idle-i932@amo 2026-11-19, riir-infer HEAD at landing `b6fca157` (pre-landing dirty tree unrelated). CPU-seconds axis load-tolerance preregistered in issue 1220 T0; wall columns never quoted as perf.
