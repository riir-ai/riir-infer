# Issue 1225: the f16-B serving lane — numerics decider + serving twin + the frame A/B (1167's f16-B rung graduated OVER)

**Status:** OPEN — filed 2026-11-16 from [Bench 1223](../.benchmarks/1223_i1167_g5_f16b_price.md): the f16-B price cleared the graduate bar by two orders (per-launch NET ≈ 65% of the GEMM wall, sign-consistent on all seven census shapes, disjoint Δ-ranges, strict-arm reproduced). The lever: the ternary prefill GEMM's B-side activations read f16 instead of f32 — the serving GEMM at prefill shapes is B-side-bandwidth-bound (each workgroup re-reads its B slice per K-tile; f16 halves exactly that term), and the win is posture-independent (relaxed AND strict arms agree).

**Claimed:** amo-idle-i1225f16b@amo, 1794835595, expires 2026-11-18T13:26:35+00:00
(scope: T1–T4, starting at T1 — the per-family retention decider at the shipped posture, P=512 AND P=4096; T2's twin shape depends on T1's per-family outcome)

**Repos:** [riir-infer] (kernels + host in `riir-infer-gpu`; the league-side adoption re-prices riir-ai Issue 1154's ceiling — separate lane, per 1167's boundary note)

## The pre-registered decider (each its own gate; order matters)

### T1 pre-registration (committed BEFORE the serving twin lands — the bars are frozen first)

**Metric set** (the Issue-1146 lossy-surface precedent, `tests/issue1146_relaxed_kl.rs`, widened per GEMM family):

- **End-to-end (deployed-path gate):** per frozen text family, final-token logits vs the f32-B reference pass — KL mean/max (nats, f64 log-softmax, sequential fold), top-1 agreement, ΔNLL mean. Corpus: the frozen four-family text set of `issue1146_relaxed_kl.rs` verbatim (code/math/prose/zh). Scored prefix lengths: K=8 spread over [P/2, P−1] at P=512; K=2 at P=4096.
- **Per-GEMM-family (the demote axis):** matched-input retention tap at the serving route — at every twin-routed site (armed passes, final frame of each scored pass), the SAME input dispatches through BOTH the twin (feeds the trajectory) and the incumbent (tap-only scratch): per token row, argmax-over-m flip; per instance, worst scale-relative drift (strided, every 16th element) and bounded flip samples. Families = the `PrefillSite` axis (attn_q, attn_kv, attn_out_proj, gdn qkv/z/a/b/out_proj, ffn gate/up/down). `embed` is a dequant lookup and the lm_head/tail is a separate dispatch path — neither is a twin-routed GEMM site; both are covered end-to-end by the trajectory read, never per-family.

**Floors (frozen; a family that fails demotes THAT family via the site mask, then the walk re-runs on the survivor set):**

1. **Placebo control (HARD):** twin compiled unpatched (numerics-identical) — end-to-end KL == 0.0 exactly, top-1 == n, per-family flips == 0, worst_rel == 0. Any nonzero = instrument defect; every armed number in that posture is VOID.
2. **Armed, per-GEMM-family:** flip rate ≤ 1e-3 of rows; worst_rel ≤ 1e-2 (the G5 input-rounding class bound).
3. **Armed, end-to-end:** per text family KL max ≤ 1e-3 nats; top-1 ≥ 15/16; ΔNLL mean within ±0.05 nats.
4. **Engagement (the Issue-1124 T1.5 lesson):** every admitted site's tap instance count > 0 per armed pass; a compile-failed/demoted twin route prints one loud counted line and the walk fails on the missing instances.

**Protocol:** one frozen read per (selector posture × arm): serving posture (tensor zerocopy + attn-tensor arms ON, KTILE64), relaxed selector DEFAULT arm (`RIIR_METAL_TENSOR_RELAXED` unset → ON) AND the strict arm (relaxed pin OFF), P=512 AND P=4096 (one forward per posture at block_size 4104; both P protocols share it — positions, not capacity, are what differ). Frame width 128 (the serving frame). GPU-exclusive (the AGENTS rule); box state printed. Record: `.benchmarks/` next number; verdict + per-family table land in the issue.

---

- [ ] **T1 — the numerics decider (pre-registered BEFORE the twin lands):** per-family retention at the shipped posture (relaxed selector DEFAULT arm + the strict arm) over one frozen read, per the lossy-surface promotion rule (deployed-path behavior, per-family conditional retention — never aggregate-only, never bit-identity: the input rounds by construction). Families: the dist prefill's token mix (embed, attn proj, FFN gate/up/down, GDN qkv/z/ab/out), Bonsai-2 league pin. Bar: no family's retention regress beyond the pre-registered floor; a family that drops demotes THAT family (per-family conditional gating — the KernelArmSelector already carries the pattern). Measure at P=512 AND P=4096 (the B-share grows with P — both retention and win must be measured at the league rows).
- [ ] **T2 — the serving twin:** `MetalTensorGemm` gains the f16-B input variant behind a feature flag (default-OFF), plus the input path. Producer-writes-f16 preferred (the producer emits the residual stream as f16 alongside f32 — the cast never lands on the frame); the per-launch `cast_f32_f16` fallback bounds the worst case (measured 0.15–0.19 ms/launch, Bench 1223). Engagement proof FIRST in situ (the Issue-1124 T1.5 lesson) — the decider's numbers are invalid without it.
- [ ] **T3 — the paired frame-census A/B:** the Bench-937 frame harness (3+1 alternating pairs, order-bias probe, the rung-C decider machinery) on the twin at the promoted posture, TWIN_RELAXED both arms; ≥1.5% e2e NET + T1 retention → GOAT gate.
- [ ] **T4 — GOAT + promotion:** G1 tolerance-class (T1), G2 perf (T3, --release), G3 decode-path regression unchanged (the feature flag is prefill-scoped; decode never routes through it), G4 alloc — then the feature-flag promotion discipline (default only if modelless-clean; this is a pure numerics-format change, no training dependency).
- [ ] **Doc + league:** riir-infer AGENTS lane row, riir-ai Issue 1154 re-price (the pp512 0.76× / pp4096 0.93× rows are the gap this mechanism targets — llama.cpp's kernel_mul_mm reads f16 B; this lane closes that gap class), HISTORY arc.

## Boundary

Substrate-side: kernels in `riir-infer-gpu`, seams in `riir-infer-dist` (same as 1167's). No riir-ai work required; the league-side adoption of the EXISTING arms is riir-ai's Issue 1154.
