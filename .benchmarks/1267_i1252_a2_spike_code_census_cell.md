# Bench 1267 — Issue 1252 A2: the spike-code census (instrument + the dense real cell)

**Cell class:** offline decision-data census (REPORT-ONLY; the issue's four
pre-registered refusals bind the lane's future — no kernel, no serving path,
no imported figures). The A2 deliverable: the `sparsity(k)` curve + the
whole-128-group skip ceiling + the structural-zero decomposition, measured on
OUR activations, BEFORE any format/kernel arm.

## What landed (this record's commit)

- `spike_code_census` feature (`closed_form_write` + `deltanet_ternary_inference`
  + `bonsai2_hadamard` — the capture seam's graph, the sink_audit folded
  posture); `src/spike_code_census.rs` (the census math + the SPKC1 artifact —
  canonical bytes + BLAKE3, the RDMAP class) + `spike_code_census_fixture.rs`
  (7 gates, the readout_screen fixture pattern — the bin `--self-test` and the
  lib test drive the same source); the `spike-code-census` bin (dense-body
  capture driver; the ternary GGUF refuses loud — the batched_mlp hook is the
  seam's only write-site, so the league ternary cell is the recorded
  follow-up slice); required-features rows in the same commit (the P30 law).
- En-route fixes the fixtures caught BEFORE landing: (1) the all-zero-row
  degenerate (`V_th = 0` → NaN spikes) pinned to zero spikes; (2) the
  floor-pick defect — a "nearest floor" excess ABSORBS the dead mass into the
  other shape's floor (the G3 fixture measured 0.31% where the honest bracket
  reads [0.31%, 5.08%]); the structural decomposition is now a BRACKET
  (excess over the loosest floor, excess over the tightest floor) with the
  shape pick as the point-estimate diagnostic; (3) the `closed_form_write`
  dense fixture's `Proj` match gained the `Ternary` arm (the spike feature
  chain compiles the variant into that build for the first time — refuse
  loud, never fill a ternary projection with dense numbers).

## The real cell (dense body — Qwen3.5-0.8B-Base-Q8_0)

- Capture: ALL 24 layers × 256 positions of the down-proj INPUT slabs (mlp
  = 3584) via the CaptureSink over `prefill_qwen_deltanet_into`; ag_news
  test-000 corpus (fleet loader); capture 11.0 s, census 0.4 s.
- Box: 4090 (shikuwa), 2026-11-26 ~10:3x local, CPU-only, AC, box quiet
  (18.6 GB RAM free), E:-light posture (build target on C:/tmp, artifacts on
  F:/tmp — the standing disk rule; E: still flagged pending the owner repair).
- Artifact: `F:/tmp/i1252/cell/spike_census.spkc`, digest
  `92d2e0bfc877c818d181bf768e7419a25dd780369980798735f0bb1dbfeaa018`
  (6,653 B; model digest `70fd9c83…` — the T1-artifact file digest class —
  corpus `a4405952…`).

## The decision read (the bars the issue pre-registered)

| read | measured | bar | verdict |
|---|---|---|---|
| element sparsity k=3 | 15.8–24.2% (layer band 12.3–24.2% over k=4..2) | — (the paper's 69% is THEIRS, never imported) | measured, far below the paper's regime |
| **whole-128-group zero rate** | **0.00000 in ALL 24 layers at ALL k ∈ {2,3,4}** | ≥ ~35% at k ≥ 3 unlocks the group-skip kernel arm | **BAR UNREACHABLE — the kernel arm stays gated OFF** |
| structural bracket (k=3) | excess_lo 0.4–8.9%, excess_hi 5.0–13.6% (layer-dependent; late layers read highest — L23 k=2 zero rate 33.4%) | decision data | the dead mass is layer-structured, real but small |
| shape | laplace fits nearer at every layer/k (heavy-tailed post-GLU) | — | the Laplace floor is the census's point-estimate floor |

**The honest verdict for the lane:** on this checkpoint's MLP activations the
whole-128-group skip ceiling is EMPTY — not one all-zero group in 6,144
layer-position groups × 3 k-values. The pre-registered kernel-unlock bar is
unreachable, so the spike-coding kernel/format arms stay refused exactly as
the issue's refusals pre-registered; the census closes A2 as
**decision-complete, kernel-negative**. A1 stays trigger-gated (no Q8KV
reopen); A3's monitor rides A2's artifact shape (the bound-vs-realized pair)
whenever a consumer arms it.

## Gates

- fixture battery 7/7 (bin `--self-test` + the lib test, same source);
  clippy `-D warnings` at the gated lib posture clean; the gated
  `--all-targets` posture clean FOR THIS LANE (3 residual warnings in
  `examples/i701_precomp_fit.rs` are PRE-EXISTING at HEAD — proven by the
  1250 T1 record's detached-worktree check; that file is another lane's);
  default-posture `cargo check` clean (the feature is opt-in; a default
  build compiles the lane to nothing).
- The twice-run law: this is a census (deterministic arithmetic, fixed-seed
  RNG nowhere in the cell path — the capture is a deterministic forward and
  the census math is fold-pinned); the artifact digest pins the run.

## The follow-up slice (recorded, not hidden)

The league TERNARY cell (Bonsai-2-27B PQ2_0) needs a capture hook in the
ternary body (the dense `batched_mlp` seam has no ternary write-site — the
sink_audit Q-capture precedent shows the wiring pattern via
HybridForwardScratch fields). Routed as the issue's follow-up; the dense
cell's group-zero-zero result makes the ternary cell's expected read
same-shaped (heavier quantization → fewer near-zero activations, not more),
so the bar-unreachable verdict is unlikely to flip — the ternary cell runs
when a consumer arms the lane, not as a standing obligation.

Session: zed-d102f326@4090 (shikuwa), 1795661651
