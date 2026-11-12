# Bench 1209 — Issue 1117 T4: the 128K axis trio (CPU host legs on bmo; Metal dense valid; Metal ρ arms VOID per issue 1213) + the stride fix's gate/validation state

**Status:** RECORD — the 128K ρ(M) arms in this run are VOID (issue 1213 row-stride collisions); the valid re-run is queued (v2, ~24 h). This record pins the host-identity leg, the valid dense(M) arm, the incident, and the fix state.

## Recipe (both trios)

League checkpoint `Ternary-Bonsai-2-27B-PQ2_0.gguf`, teacher-forced decode, block 64 / refresh 64 / bucket 8192 / compare-from 0, ctx 131072, ρ grid {0.15, 0.30}, gate binary `flashmemory_decode_gate`. Metal trio built at `f5ddcb0d` (the 1204 fresh-recompute posture); CPU trio per `scripts/i1117_bmo_128k_run.sh` on bmo (worktree `riir-infer-benchwt`). **Corpora differ by box** (the AGENTS.md-glob recipe is box-local): m3 = 35 repos / 813129 B (`corpus_blake3 0ad8cbe8…`), bmo = 31 repos / 629060 B — the standing same-regime-aggregate caveat applies to every cross-box number; nothing here is paired-token.

## The 128K CPU trio (bmo, VALID — the host-identity leg)

Bins `~/riir-bench/i1117_128k/{dense,rho_0.15,rho_0.30}.bin`, report 2026-11-13 02:28 +0700 (job 6a300a3e). Collector `scripts/i1117_collect_128k.py`: ALL INTEGRITY CHECKS PASSED, gate cross-check OK, independent recompute:

| 128K CPU (bmo) | sel frac | tok/s | × dense | ΔNLL mean | ΔNLL p95 (support) | ΔNLL max | top-1 agree |
|---|---|---|---|---|---|---|---|
| dense | 1.0 | 2.03 | 1.00× | pooled NLL 2.4977 | — | — | — |
| ρ 0.15 | 0.1488 | 2.13 | **1.05×** | +0.2579 | +1.8520 (6554) | +14.7585 | 78.52% |
| ρ 0.30 | 0.2989 | 1.03 | 0.51× | +0.1108 | +1.0222 (6554) | +13.2340 | 85.89% |

32K reference (m3, `~/.cache/i1117/32k_rel`, ctx 32768, different corpus): dense 3.60 tok/s; ρ0.15 1.38× / +0.4048 / 70.56%; ρ0.30 0.76× / +0.2087 / 79.91%. **The 128K crossover moved**: ρ0.15's speedup fell 1.38× → 1.05× (scoring/gather overhead ∝ n_blocks grows with ctx while the attended share stays ρ), ρ0.30 fell to 0.51× — at 128K the CPU crossover sits between 0.15 and 0.30, below the ρ ≈ 0.2 the 32K cell measured.

## The 128K Metal trio (m3, `~/riir-bench/i1117_128k_metal/`, 2026-11-13 04:2x +0700)

| 128K Metal (m3) | sel frac | tok/s | × dense | ΔNLL mean | ΔNLL p95 (support) | ΔNLL max | top-1 agree | verdict |
|---|---|---|---|---|---|---|---|---|
| dense | 1.0 | 2.89 | 1.00× | pooled NLL 2.4405 | — | — | — | **VALID** (unarmed path) |
| ρ 0.15 | 0.0504 | 3.95 | 1.37× | +0.2097 | +1.5583 (6554) | +14.9225 | 80.85% | **VOID** (collided selector, issue 1213) |
| ρ 0.30 | 0.1081 | 2.44 | 0.84× | +0.1587 | +1.3425 (6554) | +13.0459 | 82.50% | **VOID** (collided selector, issue 1213) |

Collector: all integrity checks passed, gate cross-check OK. Walls: dense 45396 s, ρ0.15 33172 s, ρ0.30 53765 s. The bins and report stay on disk as the collided-selector record — **quote them only with the VOID flag**; the walls describe the collided selector, not the budget arm (the gather cost is selection-count-driven, so the walls are near-miss approximations, never citable numbers).

**Why VOID:** the issue-1200 raise (128 → 1024) moved `FM_MAX_SEL` + the envelope + the k clamp but not the selection row stride literals (`kv * 128`) — at count 307/615 every kv ≥ 1 head's writes collided into the next heads' rows. The sel-frac column is itself the bug's second face (the reader reads stride 1024 rows of a stride-128-written buffer). Full anatomy + blast radius: issue 1213 (fix `5425494e` + `19c12a3c`).

## Fix state (landed before this record)

- **Regression lock**: `flashmemory_budget_metal_t2prep::fm_budget_metal_row_stride_regression` (n_blocks 1000, ρ0.15, count 150 > 128; per-row SET equality vs the CPU budget rule). **Metal run 2026-11-13 04:33: 2 passed** (the new leg + the existing three-leg parity), 0.88 s.
- **End-to-end metric validation (PASS)**: the first probe (ctx 65536 / 256 steps) was a geometry error — the walk reaches n_blocks 5 (count 1) and its 0.3362 sel is correct tiny-ctx sawtooth, not a failure; the second attempt's ρ0.9 arm was refused at the ctor by the issue-1200 loud envelope check (as designed — at the gate's configured worst-case block count ρ0.9 needs 1843 slots > 1024). The completed probe — **ctx 16384, 9300 steps, ρ0.5, fixed build** — reads **sel = 0.4926 ≈ ρ end-to-end** through the armed forward + `fm_selected_fraction` (pre-fix the same geometry read ~ρ/n_kv ≈ 0.125: kv ≥ 1 rows landed in unwritten memory and broke to zero). dense reference arm: 9300 steps, 10.8 tok/s. This baseline plus the regression gate jointly pin the chain (kernel rows at stride `FM_MAX_SEL`, reader stride, arithmetic); the count > 128 forward geometry is exercised by the v2 128K re-run with the regression gate as the row-level guarantee.

## What the valid data already says (host-identity leg)

The T4 recorded bars (|ΔΔNLL| ≤ 0.245, top-1 gap ≤ 13.1 pp, ×1.4 headroom of the T2 ρ0.5/96-step measurement) are NOT yet applicable at 128K — the device-side (Metal) ρ arms are void, so the device-vs-host delta is unmeasured. What stands now: the host legs' internal shape (ρ0.30 quality > ρ0.15 quality on both axes at both ctx; quality improves with depth — both arms' worst buckets are the first 16K), and the 128K crossover movement above. The v2 re-run supplies the device side.

## Re-run plan (queued at this record's commit)

`~/riir-bench/i1117_128k_metal_v2/`: dense.bin copied (inputs digest match → the gate auto-loads it, ~12.6 h saved); the gate re-runs ρ {0.15, 0.30} fresh on the fixed build — ~9 h + ~15 h. Same recipe/args, same corpus file path (digest identity), `--out` the v2 dir. ETA 2026-11-14 ~05:00 +0700. The completing record (bench 1210+ or an appendix here) delivers the full T4 verdict: speedup/retention curve vs Metal dense, the 128K crossover ρ, per-bucket ΔNLL/top-1 at the ρ grid, the ΔΔNLL/top-1-gap bars vs these host legs, and the T5 promotion recommendation for the ρ ≤ 0.15 operating point.
