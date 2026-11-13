# Plan 632 — the int_ref self-verification guide + the fusion promotion (i1214 G1 PASS)

**Date:** 2026-11-14
**Status:** OPEN — this IS the research-note fusion promotion required by [`.issues/1214_integer_exact_reference_lane.md`](../.issues/1214_integer_exact_reference_lane.md) §"Status after G1 pass", filed in the commit that records the bench ([`.benchmarks/1213_i1214_t1c_receipt_g3_ppl_cell.md`](../.benchmarks/1213_i1214_t1c_receipt_g3_ppl_cell.md)).
**Claimed:** zed-41cd0608@amo, 1794576200, expires 2026-11-15T12:00:00Z
**G1 verdict: PASS — all four arms green, zero tolerance:**

| arm | result | where |
|---|---|---|
| (a) batch composition | row-exact (junk-sibling batch == solo row), CPU AND GPU paths | `tests/int_ref_real_artifact.rs::real_artifact_receipt` + the G1(c) test |
| (b) prefill/decode split | row-exact (cache path == full forward, all 28 layers' KV) | same file |
| (c) CPU vs GPU int GEMM | byte-identical logits hash `22d4844a…` on the real model (every linear on Metal `igemm_i8`); corpus-wide dual-compute: all 32 chunks × 32,768 tokens row-exact CPU==GPU, `ppl=22.931885 nll_fixed=0000001909272416` identical both postures (`gemm=metal` line) | `int_ref/gpu.rs` + `real_artifact_g1c_gpu_gemm_receipt` + the `I1214_GEMM=metal` G3 posture |
| (d) cross-vendor | BIT-IDENTICAL receipt line + G3 PPL + nll_fixed on the 4090 (x86_64/Windows) vs amo (M5 Ultra ARM/macOS) | bench 1213 §G1(d), agent-mail job `01a1d65b-cdc7` |

- [x] T1 — this file (the promotion itself: the guide lives here; the research note gains a one-line pointer, never a duplicate) — LANDED `2d3bf9b`
- [x] T2 — the §9.3 suite formalization — **SATISFIED BY THE LANDED SHAPE** (adjudicated 2026-11-14): the four arms already exist as named, HERMETIC lib tests under the toy mod (`forward_is_deterministic_and_input_sensitive` with its non-vacuity arm, `batch_rows_match_solo_runs_including_padding` = G1(a) incl. the masked-pad form, `prefill_decode_split_matches_full_prefill` = G1(b), `generate_matches_manual_greedy_loop` = the cache-growth path, `greedy_pick_tie_goes_to_first_index` = the pinned tie law both raw and scaled) — always-run under `cargo test --features int_ref --lib`, no env; the real-artifact target adds the env-gated receipts (skip-loud). A third `int_ref_determinism` target would DUPLICATE them. The one honest residue — count floors over the suites — needs repo-wide test-gate infrastructure riir-infer does not have (its CI is main-only, no test_gate.sh); DEFERRED to a repo-wide lane, not an int_ref one
- [x] T3 — the receipt RUNBOOK — satisfied: §"The guide" items 1–4 ARE the runbook (calibrate → eval ids → the receipt command + the expected line shapes → the GPU arm; the digest hand-check law carried)
- [ ] T4 — the ternary-skip arm's issue (research note §3, round-1 catch): activations-only dyadic treatment, no weight calibration error; files when a small ternary checkpoint is minted (**BLOCKED on riir-train**, NOT this repo — unblocks when a ternary checkpoint exists)
- [ ] T5 — consumer routing note (research note §3 B): quorum-committed decisions / frozen-brain NPC inference serve and verify THE INT MODEL's outputs — the owning repos (riir-chain / riir-ai) file their own consumption AFTER T2 (which is done); **BLOCKED on those repos' own filing** — the lane is ready, the consumers decide

## The guide (T1 — the substance)

**What exists now** (`develop` at `f9c87a2`):

1. **The calibration** (offline, float, one-time per checkpoint):
   `uv run python scripts/i1214_prepare.py --out artifacts/<name>` — SmoothQuant-analytic
   α=0.5/QK α=0.3, upstream-compatible artifact (`model.safetensors` int tensors + `config.json`
   scales). Pin the two blake3 digests the Rust loader prints; the artifact NEVER enters a repo
   (weights law) — it travels by digest-verified copy.
2. **The eval slice** (one frozen read): `scripts/i1214_eval_ids.py` — wikitext-2 TEST prefix,
   sha256-pinned ids JSON beside the artifact. Both lanes score THE SAME bytes.
3. **The receipts** (every rig, ~19 s release + ~34 min for the corpus-wide G3):
   ```
   I1214_ARTIFACT_DIR=<dir> cargo test -p riir-infer-laya --release --features int_ref \
     --test int_ref_real_artifact -- --nocapture
   ```
   Quote the ONE line — `i1214 receipt: model=… config=… logits_final=… greedy=[…]` — plus, for
   the full cell, `i1214 G3 int8: ppl=… nll_fixed=…`. Same bytes on every rig is the lane's
   entire point; a divergent hash is a STOP, never an investigation note.
4. **The GPU arm** (macOS): `I1214_GEMM=metal` runs every chunk on both the CPU reference and
   the Metal twin (`int_ref/gpu.rs`, ONE kernel, i64 ascending-k accumulation — the CPU's exact
   order), asserting row-exact equality. The `GemmFn` seam on `IntRefModel::forward_gemm` is the
   public injection point for any future backend (CUDA twin follows the same contract; the
   operand-guard rows are mandatory only if a vendor library path replaces the owned kernel).
5. **The laws the lane carries** (each with its in-source guard):
   - **No float on the numerical path** — the type-level no-float guard holds by test
     (planted-token + comment-stripping falsification arms in the T1a suite).
   - **Pinned tie rule** — `greedy_pick` is first-index (the Issue-940 law); a planted exact tie
     must decode identically on every backend (G1 suite arm, T2 formalizes).
   - **Checked arithmetic everywhere** — i64 accumulators + `i32::try_from` at the i32 wire;
     the GPU twin repeats the law host-side on readback (no conservative bound, no silent wrap).
   - **Self-verification only** — the lane certifies itself; it is NOT a drift_guard oracle and
     NOT a G5 upgrade (the round-1 categorical limit stands: ~1e-2-class int-vs-float error).
   - **Weights never enter a repo; artifacts travel by digest.**

**What this lane is for** (restated from the research note §3A, unchanged by the G1 result):
cross-rig byte-identity receipts for anything that needs "the same decision everywhere" —
a separately calibrated model any node can verify without trusting it. NOT throughput (int8
eager decode ~2.2 tok/s on amo CPU, 1.6 on the 4090 — informational G2); the float lanes win
every league bout. Consumer B (quorum/frozen-brain) stays with the owning repos until they
file it (T5).

**Tranche 2** (pre-registered in the issue, unchanged): integer GDN recurrence with its own
state-drift bound + ΔPPL bar — blocked on a hybrid checkpoint need, not started.
