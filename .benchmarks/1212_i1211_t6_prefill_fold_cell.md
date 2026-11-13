# Bench 1212 — Issue 1211 T6: the FOSCAR fold on the Q8-KV prefill flash arm (second site)

**Status:** DONE-NEGATIVE 2026-11-13 — latency bar MISSED, **DEMOTE** (the pre-registered gate's mechanical outcome). The folded twin measured a wash-to-slightly-slower result at every shape (geomean **1.0017×** folded/incumbent) against the ≥5% bar. Parity held trivially (bit-identical folded-vs-incumbent at the fixture, identical f64 deviations), so the gate failed on latency alone. The incumbent kernel (`qwen_attention_prefill_tiled_q8_f32`, the Issue-771 T2c-a arm) stands unchanged; the ephemeral twin (folded kernel + launcher + A/B bin + Cargo row) is removed at this verdict, per the arm's own design note.

This is the second independent demote of the fold family on Metal (Bench 1211 T3b the first) — together they pin the lane-level finding: **the Q8KV kernel family on Metal is bandwidth-governed, not ALU-governed**. The FOSCAR fold's real win is the decode kernel's specific shape (Bench 1210, 6.4% — promoted), not a general Q8KV principle.

## Box state (league law)

```
PROVENANCE: power=AC Power load=2.77 swap=5650.38M canary=skipped powermode=0(auto)
```

amo (M5 Ultra, Metal/wgpu-msl), AC power, auto power mode, load 2.77 (preflight PASSED, ceiling 6.0), swap 5.65 GiB disclosed. Release build, `CARGO_TARGET_DIR=/tmp/plan1211`. Interleaved pairs ×40, warmup 5, median per arm (the sequential-A/B scheduler-tax law).

## The arm (pre-registered in the claim commit `931b130`)

Baseline = `qwen_attention_prefill_tiled_q8_f32` (the Issue-771 Q8-KV prefill flash arm; DEFAULT-OFF, `RIIR_Q8KV_PREFILL=1`, serving gate `PREFILL_Q8KV_MIN_P = 16384`). Arm = `qwen_attention_prefill_tiled_q8_folded_f32`: identical except the per-byte dequant FMUL folds into the operands —

- **QK acc-fold**: each lane's 8 dims sit in ONE Q8 block, so the lane's 8-wide partial shares one scale; the fold drops the 8 per-byte scale FMULs and multiplies the raw `q·i8` partial once before the plane reduction (reduction order unchanged).
- **PV operand-fold**: `w·(i8·v_scale) → (w·v_scale)·i8` — one `wv` FMUL per (lane, position) replaces the 8 per-byte dequant FMULs; per-position online-softmax accumulation order unchanged.

Op accounting: ~22 fewer FMULs per (lane, position) of ~48 — unlike T3b's load-CSE case, this is a genuine arithmetic reduction the compiler cannot legally reassociate away. Measured anyway: a wash.

## Numbers

| p | incumbent med ms (p90) | folded med ms (p90) | ratio |
|---|---|---|---|
| 8192 | 179.4791 (181.0255, sup 4/40) | 179.8211 (181.1213, sup 4/40) | 1.0019× |
| 16384 | 715.0565 (719.1099, sup 4/40) | 716.2636 (719.3856, sup 4/40) | 1.0017× |
| 32768 | 2873.9986 (2884.7121, sup 4/40) | 2877.9595 (2891.3388, sup 4/40) | 1.0014× |

Geomean **1.0017×**. Shapes bracket the serving gate (16384 = the arm's engagement point; the arm never runs below MIN_P). Geometry: Bonsai-2-27B league keys **probed off the league pack itself** (`Ternary-Bonsai-2-27B-PQ2_0.gguf` qwen3.5 arch keys: `attention.head_count` 24, `head_count_kv` 4, `key_length` 256) — the fixture originally used qwen38_dflash2 defaults (32/8/128), which exposed a real fixture-side lesson: the kernel's lane decomposition (32 lanes × 8 dims, `blk = lane>>2` ≤ 7) is head_dim-256-scoped; at 128 both arms read cross-block garbage identically (the first parity run printed ~2.7e-1 errors for BOTH arms — caught before any timing, fixed to the league geometry; the launcher's `head_dim % 32` debug_assert is looser than the kernel's real 256-only contract — noted as a hardening remark in the issue, not changed here).

## Parity (the falsification arm — passed)

The module's `#[test]` (removed with the twin): f64 oracle over the same quantized bytes the kernels read (the lane's own quantizer for fixture generation — engagement by construction), causal softmax, gate divide. At p=257 (tile boundary + partial tile): incumbent vs f64 **2.837e-8**, folded vs f64 **2.837e-8**, folded-vs-incumbent delta **0.000e0** (bit-identical at the fixture — the association change did not move the max-abs), bound 5.674e-8 ✓.

## Why the fold is a wash here (inference, labeled)

The kernel is memory-bound: each (lane, position) moves 8 bytes of K/V global loads (K row + V row) against ~48 flops including the dequant chain. On Metal the dequant FMULs hide under the load latency — the ~22-FMUL cut does not shorten the critical path. Consistent with the decode fold's 6.4% (Bench 1210: its per-element load chain dominates differently) and with T3b (Bench 1211): **source-level ALU-side transforms don't move the Q8KV family on Metal; only load-shape changes do.** Two independent demotes (T3b source-level hoist, T6 arithmetic fold on a second site) now pin this lane-level law. The `i1211_fold_ab_cell` bin (T3's incumbent-vs-folded decode cell) remains the family's standing measurement vehicle.

## Reproduction

The arm is removed at the verdict (ephemeral; the diff shape is fully specified in "The arm" above — the twin kernel is the incumbent with the two fold sites swapped in, re-authoring is minutes). The cell invocation it ran under:

```sh
CARGO_TARGET_DIR=/tmp/plan1211 cargo build --release -p riir-infer-gpu \
  --features q8kv_fold_cell --bin i1211_t6_prefill_fold_ab
../riir-reflex/scripts/bench_preflight.sh   # PROVENANCE first
/tmp/plan1211/release/i1211_t6_prefill_fold_ab --iters 40 --warmup 5
```

Session: zed-7d880162@amo, 1794532455 (lease: claim commit `931b130`).
