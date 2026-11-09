# Issue 1197 — Metal-vs-CPU dense decode divergence on the plain synth fixture grows with position (the 1196 residue)

**Status:** OPEN — found by 1196 T3's plain-fixture leg (2026-11-09, m3, zed-fd805f52@m3); REPORT-ONLY row in the 1196 gate carries the baseline.

## The observation

On the convention-correct synth fixture (post-1196, a_log = −exp(1.0), decay < 1 — a contractive recurrence), Metal dense decode vs CPU dense diverge with a position-growing profile:

- plain (unfolded) fixture: max |Δ|/range 2.5e-3 at steps 0–1 → 7.7e-1 at step 6 → 1.57 at step 27 (top-1 0.875 over 40 steps — 5 near-tie flips);
- folded fixture: ~10× tighter — max |Δ|/range 1.0e-1, top-1 0.95 (the 1196 gate's law).

|Δ|/range > 1 means a logit crossing zero — a real amplification, not a tie artifact. Both lanes run the SAME math (bit-clean-equivalent at the first steps); the divergence grows through the recurrence/state path of an UNTRAINED random network whose weights have no redundancy to absorb fp reduction-order differences (block/flat tile order, plane_sum vs smem trees, strided-256 vs sequential sums). The league trained model never shows this class (well-separated logits, the 1191 T2 gates pin device-vs-host drift 0.349 ΔNLL at 96 steps on the REAL model).

## Candidate amplifiers (the study's legs)

- [ ] T1 — per-stage attribution on the PLAIN fixture with the same mask census: is the growth in the GDN chain (proj/prerec/recurrence/out), the attention layer at the synth shape (8 q heads / 2 kv heads / hd 128 — a shape the league never decodes: league attention is 4 kv × hd 256), or the FFN/norm chain?
- [ ] T2 — position dependence: decode is position-free per kernel; the growth must come from state (DeltaNet) or KV (attention). Identify which by fixture surgery (e.g. an all-DeltaNet layer_types fixture) or per-step state-digest census.
- [ ] T3 — bound or fix: if the amplifier is a kernel's reduction ORDER at this shape (e.g. the attention split-K route at kv 2×128, or the plain GEMV chain's workgroup width at n_embd 1024), either pin the shape's tolerance honestly, or match the reduction order at the divergent site. The league posture stays pinned bit-identical by its existing anchors — a shape-scoped tolerance or a guarded arm, never a global numerics change.
- [ ] T4 — decide the plain-fixture's standing: either the kit's plain composition gets its own gate at the honestly-measured bars, or the kit folds by default and the plain path is documented as measurement-only.

## Non-goals

- No change to the league serving numerics (hd-256 attention, trained models) — its anchors are green.
- No chase of the near-tie top-1 flips themselves (tie-class, disclosed).
