# Bench 1213 — i1214 T1c end-to-end: the real-artifact receipt + the G3 ΔPPL cell

**Date:** 2026-11-14
**Issue:** riir-infer Issue 1214 (the int_ref DetLLM port — T1c)
**Verdict: G3 PASS — ΔPPL = +0.067 ≤ +1.0 bar. G1 FULL PASS — all four arms green, zero tolerance: (a) batch composition, (b) prefill/decode split, (c) CPU-vs-GPU-int-GEMM (probes byte-identical + corpus-wide dual-compute: all 32 chunks row-exact CPU==GPU, identical ppl/nll_fixed at `gemm=metal`), (d) cross-vendor bit-identical (amo M5 Ultra ARM/macOS vs 4090 x86_64/Windows). The fusion promotion is filed in the bench-record commit per the issue's status-after-G1 rule: `.plans/632_i1214_int_ref_self_verification_guide.md`.**

## G1(d) — the cross-vendor receipt (the lane's headline claim, PROVEN)

The artifact + eval ids scp'd to the 4090 (`E:/git/riir-infer/artifacts/qwen3-0.6b-int8/`, byte-count 781,358,936 = amo's copy), repo synced to tip `5d2abb2` via the gitsync scheduled task (git-over-ssh hangs there), then `agent-mail rig run 4090 --shared` job `01a1d65b-cdc7` (45 min, 2714.67 s of it the test):

| | amo (M5 Ultra ARM/macOS) | 4090 (x86_64 Windows) |
|---|---|---|
| model digest | `13224d13452c441d…` | `13224d13452c441d…` — IDENTICAL |
| config digest | `d1b23eadf744316c…` | `d1b23eadf744316c…` — IDENTICAL |
| logits_final | `22d4844a1d6d2e0a…74048edd` | IDENTICAL |
| greedy | `[8, 11185, 271, 785]` | IDENTICAL |
| G3 int8 PPL | 22.931885 | **22.931885 — IDENTICAL** |
| nll_fixed | `0000001909272416` | **`0000001909272416` — IDENTICAL** |
| G2 decode b1 | 2.165 tok/s | 1.618 tok/s (informational, box classes differ) |

The identical `nll_fixed` proves not just the lane but the fixed-order f64 eval reader is cross-vendor deterministic (no FMA contraction, sequential folds) — the receipt line means the same thing on both vendors, which IS the lane's one-hash purpose. 4090 G3 elapsed 2672.6 s vs amo's 1945.4 s (same 8-thread reader; different CPU classes, same numbers).

## The chain exercised

`scripts/i1214_prepare.py` (T1c part 2, `b2a3e5e`) → calibrated artifact on disk →
`crates/riir-infer-laya/src/laya/riir/int_ref/artifact.rs` loader (`1852255`) →
`IntRefModel` forward (`65373fb`+`420b4c4`) →
`tests/int_ref_real_artifact.rs` (`ea1eb27`, env `I1214_ARTIFACT_DIR`, skipped loud without it) →
G3 reader in the same file + `scripts/i1214_eval_ids.py` + `scripts/i1214_eval_baseline.py` (this cycle).

Two land-run bugs found and fixed in flight (both in the eval protocol, none in the lane):

1. **`rope_theta` eager-default getattr** (`scripts/i1214_prepare.py`, fixed `ea1eb27`):
   `getattr(cfg, "rope_theta", cfg.rope_parameters["rope_theta"])` evaluates its
   default argument eagerly — on transformers 4.x (no `rope_parameters` attribute)
   the fallback expression itself raised. Fixed to a genuinely lazy two-step; the
   calibration then ran clean on transformers 4.57.6 / datasets 4.8.5 / hub 0.36.2
   (the v5-era hub breaks transformers 4.x; the v4-era set keeps the script's
   `torch_dtype` kwarg valid — pinned in `.venv`, per-box).
2. **The G3 reader's own two eval bugs** (first run printed PPL 4.8e8 — garbage):
   (a) `norm_quant` is a per-TOKEN quant — `xm`/`xk` carry one scale PER ROW and
   the first reader applied `xm[0]` to every position; (b) HF next-token
   convention — logits row t scores the gold id at t+1, the first reader scored
   row t against token t. Both fixed in the same commit as the baseline script;
   the Python baseline reproduced the second bug exactly (bf16 read 44,189 PPL
   while misaligned), which is what isolated it to the protocol rather than the
   lane. After the fix both lanes read sane numbers (below). The lane itself
   (G1 receipt arms) was GREEN through all of this — argmax is scale-free, so
   the reader bugs could not touch it.

## The receipt (the lane's first cross-artifact G1 receipts on the real model)

```
i1214 receipt: model=13224d13452c441d config=d1b23eadf744316c
logits_final=22d4844a1d6d2e0a0018d769b6c0001cb6173fcb81f350aaaa3a763474048edd
greedy=[8, 11185, 271, 785]
```

- model digest `13224d13452c441d…` / config digest `d1b23eadf744316c…` — blake3
  over the artifact bytes (`artifacts/qwen3-0.6b-int8/`, gitignored:
  weights never enter a repo; the digests are the durable pin).
- logits_final = blake3 over the LE-byte encoding of the final-position int32
  logits row (prompt = the deterministic 7,919-strided id vector, t=8).
- Arms GREEN on the real model, zero tolerance: (G1a) batch-with-junk row ==
  solo row EXACT; (G1b) prefill/decode split (cache path) == full forward EXACT
  across all 28 layers' KV; greedy decode through the real cache growth path
  (capacity-stride slots, grow-by-doubling) — the exact path the T1b draft had
  wrong before the rewrite.

## G3 — ΔPPL on the frozen slice

Protocol (pre-registered in the issue; both lanes byte-identical rules):
wikitext-2-raw-v1 TEST, first 32,768 tokens (sha256 of the id JSON
`eec3bd417607d7b59b95f8f248d35d0d19c5a2c33c27c29a6840a19b664b1fa1`), chunks
1024 × 32 non-overlapping, HF next-token scoring (row t → token t+1), last
row's prediction dropped, f64 log-softmax on the real-domain logits
(`logits·xm[t]/2^xk[t]·head_m[v]/2^head_k`).

| lane | PPL | nll_sum | n_pred |
|---|---|---|---|
| bf16 (the bar's reference) | **22.864506** | 102450.119183 | 32736 |
| fp32 (calibration precision) | 22.853125 | 102433.821073 | 32736 |
| int8 (p_out=15) | **22.931885** | — | 32736 |

**ΔPPL(int8 − bf16) = +0.067 ≤ +1.0 — PASS** (4.8× headroom under 1/15 of the bar;
detllm's reference datum on this model class was −0.23 vs fp16 — our +0.067 vs
bf16 is the same "int8 ≈ unquantized" class).

int8 eval receipt (deterministic eval-side digest input, sequential fold):
`nll_fixed=0000001909272416` (u64 wrapping sum of per-position
`round(nll·2^20)`; Rust f64 arithmetic, no FMA contraction — cross-rig
reproducible by construction, to be VERIFIED on bmo under G1(d)).

## G2 — sampled decode (informational, PROVISIONAL)

int8 decode **2.165 tok/s at b1** over 10 steps (4.619 s), release build.
Box state at capture: amo, Apple M5 Ultra, 30 cores, AC power, 91% memory
free, load average 10.97/12.27/11.48 (a sibling cargo run held ~5 cores —
the number is load-class "loaded", not quiet), up 11 days. No bar, no
promotion talk — the lane is a self-verification reference, not a serving
lane (issue 1214's round-1 verdict).

## Cost

- calibration (48×1024 fp32 CPU passes + artifact write): ~7 min.
- G3 int8 read: 1945 s wall, 8 scoped threads (the eval reader parallelizes
  chunks; the lane stays pure per chunk — parallelism is reader-side).
- baseline bf16+fp32: ~4 min CPU.
- receipt test (no eval ids): 19.2 s release.

## What remains for the G1 suite (the issue's zero-tolerance set)

- (c) CPU int reference vs GPU int GEMM — **LANDED 2026-11-14 (`f9c87a2`)**:
  `int_ref/gpu.rs` — ONE MSL kernel (`igemm_i8`: one thread per output,
  i64 accumulation in the CPU's ascending-k order, i32 range check on
  readback — the exact twin, integer-only, no cache, no fallbacks), the
  `GemmFn` seam on `IntRefModel` (`forward_gemm`), and the
  `real_artifact_g1c_gpu_gemm_receipt` test: the REAL model with every
  linear on the Metal kernel reproduces the byte-identical logits hash
  `22d4844a…74048edd`; batch-composition held on the GPU path; kernel
  goldens bit-exact vs CPU on toy + ragged + all real geometries + the
  all-±127 worst case. The corpus-wide dual-compute posture
  (`I1214_GEMM=metal`: every G3 chunk runs BOTH lanes, row-exact
  asserted over all 32,768 tokens) runs in flight at record time — its
  green line completes the arm. **COMPLETED 2026-11-14: the corpus-wide
  dual-compute run is GREEN — `ppl=22.931885 n_pred=32736 chunks=32
  threads=8 gemm=metal elapsed=2013.5s nll_fixed=0000001909272416` —
  identical to the CPU-only values; every chunk's logits grid asserted
  row-exact equal. G1 is FULL PASS (all four arms); the fusion
  promotion filed as `.plans/632_i1214_int_ref_self_verification_guide.md`
  in the bench-record commit per the issue's status-after-G1 rule.**
- ~~(d) two different vendor boxes~~ — **DONE 2026-11-14**: the 4090
  (x86_64/Windows) quoted the byte-identical receipt line AND the identical
  G3 PPL + nll_fixed (table above); the artifact traveled by digest, never by
  repo commit (weights never enter a repo).
