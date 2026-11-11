# Bench 970 — Issue 1199 rung (b): the fused-batch layer interleave, T3 provisional cell

**Status:** PROVISIONAL — the synth-geometry cell REFUTES promotion at small scale; the real-geometry (Bonsai-2 27B) deciding cell is parked on the exclusive quiet window. `RIIR_TAIL_LAYER_INTERLEAVE` stays PROMOTE-ONLY (exact `1`/`true`/`on`; unset/other = the incumbent path, byte-identical).
**Date:** 2026-11-12
**Session:** zed-93bcde10@amo, 1794438486 (claim `1a7c863`)
**Issue:** [`.issues/1199_cpu_stage_alternation_tax.md`](../.issues/1199_cpu_stage_alternation_tax.md)

## What landed (the code half of rung b)

- `riir-infer-core/src/deltanet/ternary_forward.rs`: `qwen_deltanet_ternary_layer_body` is now reachable per layer — new pub `forward_qwen_deltanet_ternary_layer_at` (ONE layer inside the same `on_decode_pool` wrapper the range entry uses) + `forward_qwen_deltanet_ternary_finalize_into` (final norm + LM head, pool-wrapped — the head matvec keeps the incumbent pool posture); `ternary_embed_token_into` / `ternary_lm_head_into` made pub (the split's begin/end legs). The whole-step path (`forward_qwen_deltanet_ternary_range`) is UNTOUCHED — byte-identical incumbent.
- `riir-infer-dist/src/stage.rs`: the `StageCompute` layer-split quartet (`layer_split_supported` / `layer_split_range` / `step_begin` / `step_layer` / `step_end`) with defaults = unsupported (GPU stages keep the incumbent batch path); `TernaryCpuStage` implements it with a strict-ordering cursor (double begin / layer skip / repeat / early end / calls outside a split each refuse LOUD — a mis-ordered interleave is a protocol bug, never a silent skip).
- `riir-infer-dist/src/multi.rs`: the `BatchN` arm branches — armed AND every row supports the split AND all rows report the SAME window → the interleave (all rows begin → one LAYER across every stream per pass → end + replies in the head's FIFO order); else the incumbent submit/finish rounds, byte-identical. fwd_ns accounting + a flush on the new branch.
- Gates: `tests/multi_stream_interleave_g1.rs` (3 tests: 2-stream interleave tokens == the 1-box oracle; split composition == whole-step bitwise at 7 positions × 2 ranges; the 6 misuse refusals) — own test binary (arms the env once; the incumbent batch path stays pinned by the unarmed `multi_stream_batch.rs`). Full dist suite 70/70 green; clippy `-D warnings` clean across the dist crate (lib+tests+example) and core's lib.

## The T3 provisional cell (this bench — release, in-process loopback, synth Bonsai-2: 4 layers, n_embd 1024, tail = layers 2–4)

Instrument: `RIIR_DIST_STEP_TIMING=1`, 192 tokens/stream, examples/i1199_interleave_cell.rs. Box state: **amo (M5 Ultra), PROVISIONAL posture — 3 contender builds live on the box (grafts_cell cargo, a riir-games-mmorpg build, this lane)** — never a GOAT quote; two rounds ran, both same direction.

| arm | wall (384 tok) | tail fwd/step | head recv/step |
|---|---|---|---|
| serial (s=1, per-frame) | 74 / 81 ms | 0.169 ms | 0.21–0.22 ms |
| rr (s=2, pair OFF — the 1198 taxed arm) | 663–690 ms | **0.92–0.96 ms (5.5×)** | 0.77–0.79 ms |
| batch (pair ON, interleave OFF) | 148–154 ms | (batch path) | 0.20–0.21 ms |
| batch + interleave (rung b) | **155–179 ms (+5–16% vs batch)** | (batch path) | 0.23–0.29 ms |

## The three findings

1. **The rr tax reproduces violently and its shape is NOT state-residency at this geometry.** Per-step tail fwd inflates 5.5× — but tail recv inflates 3.7× too (recv is a socket read!), and the synth per-stream state is KBs (everything fits L2 regardless of alternation). The dominant cost is the per-step THREAD/P.PIPELINE handoff chain (head → tail → pool worker → tail → head, ~3–4 wakeups/step/stream), amplified by the single-drain cadence's serialization (the head waits each reply). 1198's 2.6–2.9× on the 4090 at the same per-frame posture is the same class.
2. **The fused batch already sidesteps the worst of it** (wall 4.5× better than rr at synth scale): one BatchN wave + both rows + both replies replaces per-step round trips. The 908 2.37× GPU aggregate does NOT appear here — in-process loopback head/tail share one rayon pool, so the roles' compute serializes on it (0.40 vs 0.39 ms/token aggregate ≈ serial).
3. **Rung (b) loses at small scale (+5–16% wall):** the layer-at wrapper issues layers×streams pool dispatches per batch (4 vs 2 here); dispatch overhead dominates while the working set is too small for the residency hypothesis to matter. Two rounds, same direction.

## What this decides

- **No promotion.** The synth cell refutes rung (b) at small scale; the env stays PROMOTE-ONLY and the shipped default is byte-identical-off (the full suite ran unarmed).
- **The deciding cell for the residency hypothesis** is REAL geometry (Bonsai-2 27B CPU stages: per-stream GDN+KV ≈ 3 MB + scratch — the working set 1198 named — and 64 layers where 64×s dispatches amortize over ~ms-per-layer compute), quiet-box, exclusive. **Parked on the rig queue; this bench does not adjudicate it.** The exclusive re-run uses this same example (`cargo run --release -p riir-infer-dist --example i1199_interleave_cell`) against the real checkpoint the way 1198's T3.5 instrument drove it.
- **T1's poor-man's discriminator (locality vs compute) is answered at synth scale: locality is NOT the rr tax's driver there** — the recv inflation + tiny state rule it out; the alternation tax at this geometry is scheduling-bound. The real-geometry cell re-asks the question where the premise can hold.

## Residue sighting (not this lane's)

`cargo clippy -p riir-infer-core --all-targets -D warnings` fails at HEAD on `tests/i1188_q8_projection_gates.rs:13` — `dequantize_row_q8_0` imported with ZERO call sites in the file (grep-census proof; the file is committed code this lane never touched). Owning lane: the 1188 arc.
