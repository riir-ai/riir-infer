# Issue 1226: the f16-B lane's follow-ups (the T4 verdict's carried caveats)

**Status:** OPEN — filed 2026-11-16 at the T4 promotion (delegated-owner verdict AGREE; the caveats the verdict carried forward, none blocking).

**Claimed:** amo-idle-i1225f16b@amo, 1794848737, expires 2026-11-17T16:00:00+00:00
(scope: F1 only — the pp32768 retention row via the walk's own P axis, relaxed + strict, one scored length per family; F2/F3 stay open)

**Repos:** [riir-infer]

## Tasks

- [ ] **F1 — the pp32768 retention row:** the T1 walk's frame protocol tops at pp4096 (scored prefixes 2048/4095); the census's pp32768 row has no retention read. A breach demotes THAT family via `RIIR_PREFILL_F16B_FAMILIES`, never the whole route (the verdict's own fix path). Re-use `issue1225_f16b_retention_walk` with `I1225_PS=32768` (block_size sizing + scored_lengths K=1 at 32767) — the walk's env already parameterizes P.
- [ ] **F2 — producer-writes-f16:** the residual stream emitted as f16 alongside f32 at the producer, removing the per-launch `cast_f32_f16_zc` pass (0.15–0.19 ms/launch, Bench 1223) from the frame. The measured NET already includes the tax, so this only widens the win; it is a serving-shape refinement (the producer seams), not a kernel change.
- [ ] **F3 — the next re-gate carries a `PROVENANCE:` line** (the verdict's note: power mode was not recorded on the 1224/1225/1226 cells) — `bench_preflight`-class box state beside any new quoted number.
