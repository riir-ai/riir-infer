# Issue 1225: the f16-B serving lane — numerics decider + serving twin + the frame A/B (1167's f16-B rung graduated OVER)

**Status:** OPEN — filed 2026-11-16 from [Bench 1223](../.benchmarks/1223_i1167_g5_f16b_price.md): the f16-B price cleared the graduate bar by two orders (per-launch NET ≈ 65% of the GEMM wall, sign-consistent on all seven census shapes, disjoint Δ-ranges, strict-arm reproduced). The lever: the ternary prefill GEMM's B-side activations read f16 instead of f32 — the serving GEMM at prefill shapes is B-side-bandwidth-bound (each workgroup re-reads its B slice per K-tile; f16 halves exactly that term), and the win is posture-independent (relaxed AND strict arms agree).

**Claimed:** amo-idle-i1225f16b@amo, 1794835595, expires 2026-11-18T13:26:35+00:00
(scope: T1–T4, starting at T1 — the per-family retention decider at the shipped posture, P=512 AND P=4096; T2's twin shape depends on T1's per-family outcome)

**Repos:** [riir-infer] (kernels + host in `riir-infer-gpu`; the league-side adoption re-prices riir-ai Issue 1154's ceiling — separate lane, per 1167's boundary note)

## The pre-registered decider (each its own gate; order matters)

- [ ] **T1 — the numerics decider (pre-registered BEFORE the twin lands):** per-family retention at the shipped posture (relaxed selector DEFAULT arm + the strict arm) over one frozen read, per the lossy-surface promotion rule (deployed-path behavior, per-family conditional retention — never aggregate-only, never bit-identity: the input rounds by construction). Families: the dist prefill's token mix (embed, attn proj, FFN gate/up/down, GDN qkv/z/ab/out), Bonsai-2 league pin. Bar: no family's retention regress beyond the pre-registered floor; a family that drops demotes THAT family (per-family conditional gating — the KernelArmSelector already carries the pattern). Measure at P=512 AND P=4096 (the B-share grows with P — both retention and win must be measured at the league rows).
- [ ] **T2 — the serving twin:** `MetalTensorGemm` gains the f16-B input variant behind a feature flag (default-OFF), plus the input path. Producer-writes-f16 preferred (the producer emits the residual stream as f16 alongside f32 — the cast never lands on the frame); the per-launch `cast_f32_f16` fallback bounds the worst case (measured 0.15–0.19 ms/launch, Bench 1223). Engagement proof FIRST in situ (the Issue-1124 T1.5 lesson) — the decider's numbers are invalid without it.
- [ ] **T3 — the paired frame-census A/B:** the Bench-937 frame harness (3+1 alternating pairs, order-bias probe, the rung-C decider machinery) on the twin at the promoted posture, TWIN_RELAXED both arms; ≥1.5% e2e NET + T1 retention → GOAT gate.
- [ ] **T4 — GOAT + promotion:** G1 tolerance-class (T1), G2 perf (T3, --release), G3 decode-path regression unchanged (the feature flag is prefill-scoped; decode never routes through it), G4 alloc — then the feature-flag promotion discipline (default only if modelless-clean; this is a pure numerics-format change, no training dependency).
- [ ] **Doc + league:** riir-infer AGENTS lane row, riir-ai Issue 1154 re-price (the pp512 0.76× / pp4096 0.93× rows are the gap this mechanism targets — llama.cpp's kernel_mul_mm reads f16 B; this lane closes that gap class), HISTORY arc.

## Boundary

Substrate-side: kernels in `riir-infer-gpu`, seams in `riir-infer-dist` (same as 1167's). No riir-ai work required; the league-side adoption of the EXISTING arms is riir-ai's Issue 1154.
