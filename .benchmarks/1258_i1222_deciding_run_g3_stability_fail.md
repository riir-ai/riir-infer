# Bench 1258 — i1222/i1223 deciding run: completed at full scale; G3 half-corpus stability FAILED (machinery record; demotion-consistent)

**Status:** RECORD — the deciding run EXECUTED to completion (docs=10, halves 5/5 non-empty) and its pre-registered G3 stability gate FAILED. Delegated Claude verdict round 1 **AGREE** (session `9c4987f4-ea9a-46d6-badb-6bbc88f43208`, 2026-11-24): record-and-close, no debug item while the primitive is demoted. **This record is MACHINERY ONLY — never read as a lens promotion** (the primitive was DEMOTED 2026-11-17; re-open only via Research 666 §7's named triggers).

## The run

- Geometry: docs=10, t=128, dim_batch=128, points=48, seeds [16,127), d=1024 (24 layers, 18 GDN / 6 attn) — the Bench-1240-derived deciding posture (~25.9 h est, measured 26.2 h wall; 9,622.2 s/doc avg, sweep wall 94,309 s + capture).
- Corpus: ag_news, blake3(rows) `22cb3c582d73cbc6…` (the smoke's pinned corpus), 200 rows → 1,280 tokens.
- Execution: 4090 (RTX 4090 24 GB, wgpu<spirv> → Vulkan), Exclusive rig job `01a20678` (amo i1199 lane, box-state string `4090-i1223-deciding-docs10-post-7ce4c7e-resume-lane`), 2026-11-23 → 2026-11-24 13:26 +07; per-doc checkpoints gen_a↔gen_b (5 rotations, the long-runs law's durable state dir).
- Artifacts: `.lens_gpu_fit_t3/` on the 4090 — 1,152 MiB f32 rows (fit_all/even/odd × k/j, format `lens_gpu_fit_t3_v1`) + manifest.json.

## The gate result (the finding)

```
# G3: points=48 max_rel_k=1.1324 max_rel_j=0.9932 pass=false
Error: G3 half-corpus stability FAILED
```

- G3 = fit stability: the fit built from EVEN docs vs the fit built from ODD docs must agree ≤ 6% (relative Frobenius on K and J; `gate_g3`, `src/lens/fit.rs`). At 5/5 non-empty halves the measured disagreement is ~100% — the two halves' fits are uncorrelated.
- **Reading (verdict-concurred): not a code defect.** The even/odd accumulation paths are symmetric under the split by construction; rel ≈ 1.0 means the fit is corpus-dominated at real geometry. This corroborates the demotion verdict on the same instrument: katgpt-rs 999 closed DEMOTED 2026-11-17 (Research 666 §7: "the fit learned corpus statistics, not weight geometry"; amo T3 fit j-drift 47.6% at 40 docs).
- **Scale wording (per the verdict's correction):** G3 fails at BOTH 40-doc (amo, k 0.117 / j 0.476) and 10-doc (4090 GPU, k 1.13 / j 0.99) scales; drift is LARGER at the smaller halves, consistent with a corpus-dominated fit — smaller halves alone could explain much of the extra drift. NOT "demotion confirmed at full scale."
- The docs=1 smoke's G3 = 0.0 was STRUCTURAL ONLY (odd half empty; the halves-by-construction trivially agree). The first REAL G3 read at ≥2 docs is this run's failure. The production tolerance calibration the i1223 acceptance named lands only behind a passing stability gate — unreachable without a fit-stability fix, and fixing a demoted primitive's fit has no consumer.

## Reproducibility evidence

- The amo sweep lane (zed-074a264a@amo) re-dispatched the same cell 2026-11-24 ~21:47 +07 (Exclusive job `01a20dd3`/`20eebe5a`, box-state `4090 deciding-run docs=10 post-e31d1ea4`, riir-infer `e31d1ea4`) after finding the morning completion unrecorded: the resume lane re-read the same accumulation (docs_done=10, n_even=5, n_odd=5), re-finalized the five row files from the same gen_b accumulation, and read the IDENTICAL G3 report (max_rel_k=1.1324, max_rel_j=0.9932, pass=false), exiting code 1 identically. Two independent processes, one accumulation, byte-identical gate arithmetic.
- The morning run's own log (4090 `C:\Users\katop\.agent-mail\jobs\logs\01a20678-…log`): 10/10 checkpoints, same G3 line, same exit — the failure PREDATES this record's session; the re-run added reproducibility, not new data.

## Verdict record

Delegated Claude verdict (2026-11-24, session `9c4987f4-ea9a-46d6-badb-6bbc88f43208`): **AGREE** — record-and-close. Refinements applied: (a) the scale wording above (never "confirmed at full scale"); (b) i1222 + i1223 close in the SAME commit, leases folded to `**Lease closed:**` rows, lint green, files removed once the HISTORY row cites the hash; (c) re-open only on Research 666 §7's named triggers; (d) no debug item while the primitive is demoted. The verdict's #OwnerGate section also resolved riir-train plan-436 G3 (Option 1, INCONCLUSIVE) and riir-train Issue 635 T5 (DEFER) — recorded in those repos.

## Box state

4090: GPU 0.5–1% at dispatch (24 GB card, desktop exempt per the standing note), 17/32 GB mem free, disk 50 GB free; Exclusive lane both runs; no quoted latency claims (the 9,622 s/doc IS the wall datum, Exclusive-posture).
