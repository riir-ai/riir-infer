# Issue 1196 — the synth Bonsai-2 fixture's METAL dense decode NaNs at step 16 (an unexercised hd-128 baseline path)

**Status:** OPEN — found by issue 1191 T2's first parity-cell run (2026-11-09); NOT a 1191 defect (its parity pivoted to the real model; the bisect below PROVES the 1191 arm clean on the league posture).

## The finding

`tests/common/synth_bonsai2.rs` (4 layers: 3 DeltaNet + 1 Attention, hd 128, folded) driven through `TernaryDeltanetGpuForward` (the METAL decode forward, feature `flashmemory_budget_metal_t2` posture, DISARMED — the incumbent dense path) produces **NaN logits from step 16** and every step after: teacher-forced walk vs the CPU dense lane on the identical stream agrees bit-clean for steps 0–15, then `top1 = 255` with NaN logits at step 16 and beyond (agreement 16/40, worst |Δlogit| 4.02 over the 40-step window; measured 2026-11-09, amo M5 Ultra).

The CPU dense lane on the same fixture is clean throughout (the G1 battery's lane). So the defect is in the METAL decode stack at the synth's geometry — most plausibly an hd-128-specific kernel assumption: the synth carries `attention.key_length = 128` while the league posture (Bonsai-2-27B, the only decode-verified Metal posture) carries head_dim 256, and several decode kernels carry hd-256-shaped constants (the shared-memory reduction trees, the split-K geometry, `Shared::<[f32]>::new_slice(256)` sizing). The first-16-steps-clean-then-NaN shape points at a state/stage that crosses a threshold at step 16 (the DeltaNet conv kernel is 4; the first block boundary is 64 — 16 matches neither), so the stage attribution needs the decode-stage census (`DECODE_STAGE_MASK` per-stage masking, `set_decode_stage_mask`) run per stage.

## Evidence

- Repro: construct `TernaryDeltanetGpuForward::new` on the synth folded file (the plan-622 kit), walk 40 xorshift tokens, compare logits vs `forward_qwen_deltanet_ternary`. Steps 0–15: bit-clean agreement. Step 16: NaN. (The bisect lived in the 1191 T2 session's temp probe — re-derivable from this text + the kit in ~10 min.)
- The same fixture on the CPU lane: clean (the G1 battery runs 700 steps).
- The league model on Metal dense: the SHIPPED posture (Issue 1151) — unaffected; this is a synth-geometry baseline defect, never hit by serving.

## Why it matters

- The synth kit is the workspace's standing small-model fixture; any future Metal-decode test on it silently inherits this (the 1191 T2 first run lost a cell to it).
- The hd-128 decode path may share the defect shape with other non-256 geometries (a shape-generalness audit of the decode kernels' hd constants is the real deliverable).

## Tasks

- [x] T1 — stage attribution: per-`DECODE_STAGE_MASK` masking runs (attention-only, deltanet-only, ffn-only, tail-only) on the synth walk; the NaN's stage named. **DONE (`tests/i1196_stage_attribution.rs`, `[[test]]` row + required-features, m3 Metal, `--release`): repro leg pins the SHAPE (finite prefix → NaN from a small step, stays NaN to the end; onset step is seed-dependent — this run's seed NaNs at 14, the T2 session's saw 16). COARSE verdict: masking OFF deltanet kills the NaN; masking OFF attention/ffn/tail_norm leaves it at step 14; masking OFF tail_lm_head also kills it (disclosure point — with the head skipped the walk emits the finite residual stream). PRODUCER = the DeltaNet region. SUB-BIT split (parent ON, each sub OFF): skipping proj/prerec/recurrence/outproj/concat/dense-escape/pre_betal2/out_gemv kills the NaN (the chain starves downstream); skipping **pre_conv (15), out_zgate (17), out_rot (18) leaves the NaN at step 14** — with rot skipped the out_gemv STILL NaNs, which is the stale/pad-read signature (the plan-622 arm-3 class: one NaN pad read poisons the logits; at hd-128 some buffer sized/strided for hd-256 leaves half a row NaN and a kernel reads past the valid region). Bounded: producer sits in the DeltaNet out leg at-or-before `gdn_out_gemv`, with `gdn_pre_betal2` or `gdn_recurrence` as the write-side candidates — T2's hd-constant hunt targets exactly those kernels.
- [ ] T2 — root cause in the named stage's kernels at hd 128 (the hd-256 constant hunt: `Shared::new_slice` sizings, reduction-tree start strides, split-K geometry guards).
- [ ] T3 — fix + a hd-128 Metal dense-vs-CPU parity gate on the synth fixture (the missing baseline gate this finding exposes); flag-off byte-identity unchanged (the league hd-256 posture pinned bit-identical by its existing anchors).
