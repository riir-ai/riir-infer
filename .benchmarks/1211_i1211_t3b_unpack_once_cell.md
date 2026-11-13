# Bench 1211 — Issue 1211 T3b: the unpack-once (QK word-read hoist) A/B cell

**Status:** DONE-NEGATIVE 2026-11-13 — latency bar MISSED, **DEMOTE** (the pre-registered gate's mechanical outcome). The unpack-once arm measured a consistent ~0.8–1.0% REGRESSION vs the folded baseline, the opposite direction of the ≥5% bar. The promoted default (`attention_decode_q8kv_folded`, the T3 fold) stands unchanged; the ephemeral arm (kernel twin + launcher pair + cell bin + Cargo row) is removed at this verdict, per the arm's own design note. Parity held trivially (the hoist is arithmetic-preserving — bit-identical outputs asserted at every shape, both postures), so the gate failed on latency alone.

## Box state (league law)

```
PROVENANCE: power=AC Power load=1.94 swap=5802.44M canary=skipped powermode=0(auto)
```

amo (M5 Ultra, Metal/wgpu-msl), AC power, auto power mode, load 1.94 (preflight PASSED, ceiling 6.0), swap 5.8 GiB disclosed. Release build, `CARGO_TARGET_DIR=/tmp/plan1211`. Two full cell runs (each: interleaved pairs ×40, warmup 5 — the sequential-A/B scheduler-tax law); both runs agree in direction at every shape.

## The arm (pre-registered in Issue 1211 T3b)

Baseline = the promoted folded kernel (`attention_decode_q8kv_folded`, default since T3; `RIIR_Q8KV_FOLD=0` demotes to the incumbent). Arm = `attention_decode_q8kv_folded_uo`: identical except the QK phase's inner 32-element loop is word-major — one `kv_qs[qs_block_base + w]` read serves its 4 bytes (the per-element form re-read the same u32 word once per byte, 4× per word, 8 words per block). Element-ascending FMA order preserved (`e = w*4 + j`), so the T1 reduction regroup is untouched.

Cell: `i1211_t3b_unpack_ab_cell` (feature `q8kv_fold_cell`), two-arm interleaved pairs (fold, uo), median per arm, shapes 512 / 4096 / 32768. Parity: the falsification arm extended to the uo kernel in `attention_q8kv_folded`'s tests (bit-exactness vs folded asserted + the ≤2× incumbent-vs-f64 bound, plain AND sink postures) — all green before the latency cell.

## Numbers

Run 1 (geomean **1.0096×** uo/folded):

| n_pos | folded med ms (p90) | unpack-once med ms (p90) | ratio |
|---|---|---|---|
| 512 | 0.4198 (0.5424, sup 4/40) | 0.4213 (0.5521, sup 4/40) | 1.0036× |
| 4096 | 1.9176 (1.9487, sup 4/40) | 1.9483 (1.9871, sup 4/40) | 1.0160× |
| 32768 | 16.1663 (16.2469, sup 4/40) | 16.3165 (16.4791, sup 4/40) | 1.0093× |

Run 2 (geomean **1.0083×**), same posture minutes later:

| n_pos | folded med ms (p90) | unpack-once med ms (p90) | ratio |
|---|---|---|---|
| 512 | 0.4237 (0.5699, sup 4/40) | 0.4268 (0.5544, sup 4/40) | 1.0072× |
| 4096 | 1.9410 (1.9783, sup 4/40) | 1.9600 (1.9904, sup 4/40) | 1.0098× |
| 32768 | 16.1687 (16.3070, sup 4/40) | 16.2955 (16.4367, sup 4/40) | 1.0078× |

Every shape, both runs, p50 AND p90: the arm at-or-above the baseline. Consistent direction, small magnitude — a real ~1% overhead, not noise.

## Verdict + why the premise failed (inference, labeled)

**DEMOTE** — the bar (≥5% gain AND parity) missed on latency. Parity passed (bit-identical: identical max-abs deviations vs f64 as the folded kernel at every shape — 4.759e-8 / 1.918e-8 / 1.597e-8 plain, 2.000e-8 sink — the hoist moved loads, never arithmetic).

The issue's premise ("the observed unpack redundancy is a REAL nearby win") is measured-REFUTED. Inferred mechanism (inference, not measured instrument): the flat 32-element loop is fully unrolled/optimized by the MSL compiler, which already CSEs the 4 redundant word loads per block — the source-level redundancy was never a memory-traffic redundancy in the generated shader. The nested word-major form adds per-element index arithmetic (`w*4+j` recompute) and unrolls less aggressively, costing ~1% net. The "4× per u32 word" observation was true of the SOURCE, not of the compiled kernel.

Corollary for the parked future work: the PV phase's 4× word sharing is cross-thread (four output dims per word, one byte each) — hardware-broadcast territory; a shared-memory staging rewrite remains the dp4a-class parked item (Issue 1211's unpriced future work), and THIS result lowers the prior on source-level load-redundancy claims in this kernel family generally: measure the generated shader, not the source.

## Reproduction

The arm is removed at the verdict (ephemeral by design; the diff shape is fully specified in "The arm" above — re-authoring is minutes). The cell invocation it ran under:

```sh
CARGO_TARGET_DIR=/tmp/plan1211 cargo build --release -p riir-infer-gpu \
  --features q8kv_fold_cell --bin i1211_t3b_unpack_ab_cell
../riir-reflex/scripts/bench_preflight.sh   # PROVENANCE first
/tmp/plan1211/release/i1211_t3b_unpack_ab_cell --iters 40 --warmup 5
```

The parity half remains standing in `attention_q8kv_folded`'s tests for the folded kernel (2 tests, both postures).

Session: zed-7d880162@amo, 1794530988 (lease: claim commit `9763b22`).
