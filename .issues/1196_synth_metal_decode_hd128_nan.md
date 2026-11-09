# Issue 1196 — the synth Bonsai-2 fixture's METAL dense decode NaNs at step 16 — RESOLVED: a fixture bug (raw-positive ssm_a), every kernel exonerated

**Status:** RESOLVED 2026-11-09 (m3, zed-fd805f52@m3) — the finding's premise is overturned. The NaN was the synth kit's `ssm_a = +1.0` violating the Issue-594 loader convention, not an hd-128 Metal kernel assumption. Fixed in the kit; the standing hd-128 baseline gate landed; a follow-up accuracy observation filed (issue 1197).

**Lease closed:** zed-fd805f52@m3, epoch 1794232800, exp 2026-11-11T22:00:00+00:00 — resolution landed (the fix commit). A later claim by zed-34912c3b@amo (epoch 1794237218, exp 2026-11-11T15:13:38Z) was filed against the ORIGINAL finding and is superseded by this resolution — the issue is closed; the T2/T3 that claim re-scoped are moot (the kernels are exonerated; nothing to fix).

## Resolution (the evidence chain, in order)

1. **T1 stage attribution** (`tests/i1196_stage_attribution.rs`, the `DECODE_STAGE_MASK` census): producer = the DeltaNet region (mask-off deltanet kills the NaN; attention/ffn/tail_norm do not; tail_lm_head is the disclosure point). Sub-bit split bounded it to [pre_betal2, recurrence]; skipping pre_conv / out_zgate / out_rot left the NaN (their absence changes the values but the NaN persisted).
2. **Unfolded fixture CLEAN** — same geometry, no folding: the defect was never "the plain kernels at hd 128".
3. **Default-on folded rungs exonerated**: `RIIR_NORM_ROT_MULTI=0` and `RIIR_GDN_DENSE_PAIR_FUSED=0` both byte-identical NaN at step 14.
4. **The per-step buffer census** (`state_readback::read_decode_diagnostic_buffers`, a new diagnostic seam — readbacks only, never a hot path): at the onset step, L0.state held FINITE values up to 4.05e34 — unbounded GROWTH, then Inf → NaN; L1/L2 all-NaN downstream were victims of the poisoned residual. And L0.decay_buf at STEP 0 read `[1.002, 1.002, 1.014, 1.056, 2.465, 63.8, 1066.1, 924.1]` — decay > 1, impossible under the convention (decay = exp(a_log·softplus) ∈ (0,1] with a_log NEGATIVE).
5. **Root cause**: the kit wrote `ssm_a = +1.0` — a RAW positive A_log — while the loader contract (Issue 594) carries `a_log = −exp(A_log_raw)` (the converter applied −exp once). With a_log = +1, decay = exp(+softplus(·)) ≥ 1 on EVERY lane; the state grows without bound; Metal's fp path crossed the overflow cliff first (seed-dependent onset: 14 here, 16 in the original finding), CPU's later — the "bit-clean prefix then NaN, Metal-only" shape was the growth surfacing.
6. **The decisive legs**: Metal on a convention-correct fixture (a_log = −exp(1.0)) — CLEAN, 40 steps, bounded logits. The "hd-128 unexercised kernel path" hypothesis is doubly dead: hd 128 is the CANONICAL decode geometry (both recurrence kernels require exactly head_dim == 128; the league model's GDN head_dim is 128 — its ATTENTION head_dim is 256).

## Landed

- Kit fix: `ssm_a` payload → `−exp(1.0)` (the converter's form for a raw A of +1.0), with the convention documented at the write site. Consumers assert internal consistency, not absolute pins — no pin moved (measured: all root-manifest consumer gates re-run green; `issue028`'s q4-pair bar RECALIBRATED — see below).
- `state_readback::read_decode_diagnostic_buffers` — the per-step value-level diagnostic seam (1196 T1/T2's instrument; the manifest's values-half).
- `tests/i1196_stage_attribution.rs` — T1 attribution gate + the T3 standing baseline: folded hd-128 Metal-vs-CPU dense parity on the corrected fixture (measured top-1 0.95, worst |Δ|/range 1.0e-1 — an untrained random network's near-tie flips, bars pinned at 0.90 / 0.15 with the measurement recorded) + the a_log-convention regression tooth (a deliberately-bad fixture must produce decay > 1 — the kit bug can never silently return).
- `issue028_disaggregated_handoff` q4-pair bar recalibrated 0.05 → 0.35 with the honest history in the comment: the old pass read 0.0367 because both containers ran exponentially-growing logits (shared dominance); on the correct fixture the true ε(Q4_K)-through-4-random-layers value is 0.2543. Same law, honestly calibrated.

## Follow-up filed

- **Issue 1197** — the plain-fixture Metal-vs-CPU divergence grows with position (|Δ|/range 2e-3 at steps 0–1 → 1.6 by step 27; folded is ~10× tighter at 1.0e-1): fp reduction-order noise amplified through the untrained random network's direct path — which stage amplifies (attention 8q/2kv/hd128 shape? the plain GEMV chain? norm eps at this scale?) needs its own controlled study. Report-only row in the 1196 gate.

## Non-goals (original — now moot in the original framing)

- The hd-256-constant hunt: no kernel defect existed; the hunt is closed with the kernels exonerated by the fixed-fixture legs.
