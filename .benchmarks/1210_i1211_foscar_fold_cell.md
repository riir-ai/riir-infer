# Bench 1210 — Issue 1211 T3: the FOSCAR fold A/B cell (Q8KV decode attention, Metal)

**Status:** DONE — the pre-registered gate **PASSED end to end**: parity held at every
shape and posture, latency geomean 0.9360× (folded/incumbent) ≥ the 5% bar → the
folded kernel **PROMOTED into the q8kv dispatch path** (env-switch law demote:
`RIIR_Q8KV_FOLD=0`, exact literal, byte-identical incumbent restore).

**Lane:** riir-infer Issue 1211 (Research 026, arXiv:2611.09722 §4.1 — FOSCAR
affine folding over the token-wise Q8_0 KV groups). T1 (QK acc-fold) + T2 (PV
operand-fold into a shared `p'` table) authored by the claiming lane
(`zed-1211-foscar@amo`, claim commit `1d67e95`); T3 (parity + A/B + promotion)
executed by the continuation lane `zed-71cd4872@amo` per the handoff.

## Cell

```
cargo run --release -p riir-infer-gpu --features q8kv_fold_cell \
  --bin i1211_fold_ab_cell -- --iters 40 --warmup 5
```

Backend `cubecl_wgpu::runtime::WgpuRuntime` (Metal, `wgpu<msl>`); interleaved
(inc, fold) pairs ×40, 5 warmup pairs, median (pinned lower-middle rule), p90
with tail support.

**PROVENANCE: power=AC Power load=3.55 swap=5898.75M canary=skipped
powermode=0(auto)** — preflight PASSED; the swap-residency warning (5.9 GB) is
disclosed beside the numbers; no compute cargo/riir process ran during the cell
(`ps` swept before + after; rig `agent-mail rig status amo` QUIET, load
1.9→3.55/30).

## Results

| n_positions | incumbent med | folded med | ratio | ≥5%? |
|---|---|---|---|---|
| 512 | 0.5376 ms | 0.5031 ms | **0.9358×** | ✓ (6.4%) |
| 4096 | 2.0798 ms | 1.9396 ms | **0.9326×** | ✓ (6.7%) |
| 32768 | 17.2617 ms | 16.2170 ms | **0.9395×** | ✓ (6.1%) |
| **geomean** | — | — | **0.9360×** | ✓ **(6.4%)** |

Every shape individually clears the pre-registered ≥5% bar, including the
long-context cell (32768 positions — 128 tiles of 256). The p90s hold the same
ordering (folded ≤ incumbent per shape), so the gain is not a median artifact.
Support lines: p90 index lands at support 4/40 on both arms at every shape
(the n−1-max rule disclosed — both arms equally affected).

## Parity (the falsification arm — `attention_q8kv_folded` tests, Metal)

Folded-vs-f64 oracle must stay ≤ 2× incumbent-vs-f64 (the incumbent's own
error defines the folded kernel's bound — a measured class):

| posture | n_pos | incumbent vs f64 | folded vs f64 | bound | verdict |
|---|---|---|---|---|---|
| no-sink | 1 | 0.000e0 | 0.000e0 | 0.000e0 | ✓ (exact — single block) |
| no-sink | 32 | 6.920e-8 | 4.759e-8 | 1.384e-7 | ✓ (folded BELOW incumbent) |
| no-sink | 300 | 2.030e-8 | 1.918e-8 | 4.059e-8 | ✓ |
| no-sink | 512 | 1.903e-8 | 1.597e-8 | 3.806e-8 | ✓ |
| sink (4 rows) | 300 | 2.152e-8 | 2.000e-8 | 4.304e-8 | ✓ (fold-exempt sidecar exact-in-class) |

The fold regroups the reduction (per-block partial then one scale multiply)
without reordering element traversal; measured effect is folded error at-or-
below the incumbent's at every shape — the shared-memory `p'` round-trip costs
nothing numerically.

## Promotion record

- **Dispatch flip:** the three `gemma2_cubecl/dispatch.rs` Q8KV launch sites
  (`dispatch_attention_wo_q8kv_f32`, `dispatch_attention_wo_q8kv_q4k`, the
  `dispatch_attention_gpu` CPU-Q8 fallback) route through one
  `launch_q8kv_attention` helper → `AttentionQ8KVFolded` when
  `q8kv_fold_enabled()` (unset = ON), else the incumbent `AttentionQ8KVCubeCL`
  byte-identically. `launch_with_sink` remains serving-unwired (Issue 716's
  lane — untouched by this promotion).
- **Demote:** `RIIR_Q8KV_FOLD=0` (exact literal; `false`/`off` documented
  aliases), OnceLock, read once per process.
- **Safety:** both launchers carry the identical Gemma-2-2B geometry asserts
  (n_head=8, n_kv_head=4, head_dim=256) and the same binding-derives-units
  guards — any config the incumbent served, the folded kernel serves;
  anything else panics in both.
- **Regression proof:** `cargo test -p riir-infer-gpu --features
  cubecl_runtime,q8_kv_cache,q8kv_sink_guard --lib` = **327 passed / 0 failed**
  at the promoted posture (the full gemma2_cubecl + attention battery,
  ~357 s). Demote posture: gemma2 tests 13/0 + q8kv tests 14/0 with
  `RIIR_Q8KV_FOLD=0` armed.
- Clippy clean at three postures (cubecl_runtime; +q8_kv_cache,+sink,+cell;
  no-default-features +q8_kv_cache). rustfmt clean.

## What the fold bought (mechanism accounting)

Exactly one FMUL per KV element per phase removed (the honest accounting in
Issue 1211's premise): the QK phase drops the per-element scale FMUL for one
per-block partial multiply; the PV phase drops the per-element scale FMUL and
the per-element scale LOAD for one shared-memory broadcast read. At ~5 ops per
element that is the predicted single-digit %, landed at 6.1–6.7% — above the
pre-registered expectation, below any claim of transformational gain. The 8 KB
`p'` table's occupancy cost is inside these numbers (net positive).

## Not claimed here

- CUDA/wgpu-non-Metal postures: the cell ran Metal only; other backends keep
  the demote literal available and inherit the fold by default — a cross-
  backend re-cell is a new task, not part of this gate.
- `launch_with_sink` serving wiring (Issue 716's own lane).
- T4 (INT4-KV league arm on Ada) — deferred in the issue, lossy-arm law.
- T5 (GDN state-readout fold) — rides Issue 1207 (parked, 2.2% traffic).
- T6 (prefill q8kv second site) — optional after T3; the prefill arm is
  DEFAULT-OFF with its own gate.
