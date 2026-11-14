# Bench 1218 — Issue 1181 T1: the per-token host encode-share demote gate (resident decode, census posture, 27B folded)

**Date:** 2026-11-14 · **Box:** amo (M5 Ultra 96 GB) · **Session:** amo-idle-i1181@amo
**PROVENANCE:** `power=AC Power load=5.40 swap=4196.62M canary=skipped powermode=0(auto)` — preflight PASSED at run start (the 1-min load was DECAYING from a 20.38 burst: the 5-min average was 11.94 and falling; the load-sensitive host-side figures carry that caveat, the GPU-bound walls do not — see the stability rows).
**Commit:** `2a744b5` (probe cell; the `c5e0707` clippy fix is signature-only, landed after the run was queued — the measured binary is `2a744b5`'s, semantics identical).
**Model:** `Ternary-Bonsai-2-27B-PQ2_0.gguf` (6.7 GB, folded=true, n_layer=64, vocab=248320, KV block 4096 → the Issue-864 512 MiB KV warn line; steady state unaffected).
**Run command:** the `issue1181_encode_share_probe` test header (`cargo test --release -p riir-infer-gpu --features cubecl_runtime,ternary_gemv,ternary_gemm_batched,ternary_deltanet_chunked_prefill,ternary_attention_batched_prefill,deltanet_recurrence_rowpar,params_handle_cache,riir-infer-core/bonsai2_hadamard --test issue1181_encode_share_probe -- --ignored --nocapture`), the bench_763 census posture (`set_uniform_cache_override(Some(true))`), 3 ABBA pairs × (32 incumbent + 96 resident) measured tokens after one warm-up pass of each arm, plus a walk-identity assert (the two lanes picked token-identical sequences — `ok`).

## The question (issue 1181 T1, pre-registered)

Is the per-token host command-ENCODE cost on the resident decode lane ≥ 1.0 ms/token? The bar anchors to 1.0 ms ≈ 6.3% of the amo resident loop (15.92 ms/tok, plan 626 T4). **DEMOTE unless measured encode ≥ 1.0 ms/token** — the bar is box-independent; the percentage re-derives per box.

## Result — the bar is MET 3× over: LANE OPEN

| figure | µs/token | notes |
|---|---|---|
| **resident enqueue p10** | **3552.4** | n=576 per-call durations; the fast cluster (first enqueue of each burst, post-harvest empty queue) spans min 2688.6 … p10 3552.4 |
| resident enqueue min | 2688.6 | the pure-encode floor |
| incumbent front median | 2979.1 | n=192 (`set_input_token` + `forward_dispatch_only` — the same per-op encode path, queue freshly drained); min 2664.0 / p90 3753.8 / max 4001.8 |
| Issue-1168 P1 front (pre-census) | 2455 | the historical cross-check — the census features did NOT materially change host encode (2.455 → 2.98 ms) |
| incumbent read (GPU wait) p50 | 15478.8 | n=192 — the P1 read decomposition reproduces (15.3–16.1 ms) |
| decode wall incumbent | 18.87 ms/tok | median of pairs (18.81–19.06 — vs the G2 reference 18.26, +0.6 ms residual-load slow, ratio intact) |
| decode wall resident | 16.21 ms/tok | median of pairs (**16.20 / 16.21 / 16.23 — ±0.03 across pairs, rock stable**; vs the G2's 15.92, ratio 0.859× vs G2's 0.872× — same shape) |

**VERDICT: LANE OPEN** — encode ≥ 1.0 ms/token MET (three corroborating pure-encode figures: 2.69 / 2.98 / 3.55 ms). The watch lane stays open; T2–T4 become live candidates.

## The second finding — the resident queue is POOL-BOUND ~1-deep, not ring-bound k=8 (filed as Issue 1217)

The per-call enqueue distribution is bimodal and the pattern decodes it: per 8-token burst, ONE enqueue is fast (2.7–3.5 ms = pure encode; 72 fast of 576 = 12.5%, so p10 lands inside the fast cluster) and SEVEN block at p50 14495.7 / p90 14868.0 µs — almost exactly one GPU token (the wall is 16.21 ms/tok, enqueue ≈ wall − encode). Host busy share (enqueue + harvest over wall) = **100.0% in every pair**; harvest waits 24.1–25.9 ms per burst.

Reconciliation: if the host truly ran 8 deep it would finish a burst's enqueues in 8×3 ms while the GPU needs 8×16 ms — the harvest fence would be pre-satisfied and enqueue would never block. Measured: enqueue BLOCKS ≈ one GPU token, harvest waits a full token's remainder → **the effective queue depth is ~1–2 outstanding tokens, not the k=8 ring cadence**.

~~The serialization is the allocation backpressure inside the CubeCL launch path~~ — **CORRECTED same session (vendor source read): the serializer is the two hardcoded poll(Wait) sites in the vendored fork's stream** (`vendor/cubecl-wgpu-0.11.0-pre.2/src/compute/stream.rs`): `SubmissionLoad::regulate`'s **`MAX_TOTAL_TASKS = 512`** (≈ one token's dispatch count at 64 layers → the poll(Wait) fires ~once per token) + `write_to_buffer`'s **`MAX_PENDING_WRITES = 64`** (uniform writes per scalar-arg dispatch on Metal/WGSL — a second, more frequent poll point). `cleanup_interval` defaults 0 (disabled), so `memory_cleanup()` never fires in these walks. The pool was never the serializer; issue 1217 carries the corrected mechanism + the env-configurable-caps fix lever. The Issue-761 counters (REGULATE_NANOS / WRITE_POLL_NANOS, always-on in the fork; exposed to this cell by the `counter_probe` feature, `2301ca8`) verify the decomposition — that row appends here from the bmo window (job `01a1dabb-de8f`).

Consequences, precisely scoped:
- The resident lane's PROMOTED win is real but its mechanism is the **logits-read removal** (the 1 MB per-token `read_one`), NOT deep queuing — the wall (16.21) sits ~0.9 ms/token ABOVE the Issue-1168 P2 deep-queue floor (15.32). The promoted-lane verdicts (G2 15.92 vs incumbent 18.26; the league re-cells; the 1165 default flip) adjudicated measured end-to-end numbers and stand — this finding corrects the MECHANISM NARRATIVE ("NO host sync" in the resident doc comment; "the queue-depth invariant holding" in the G2 record — the stable median was the periodic 1-deep pattern, not depth).
- The 1181 replay remedy's WALL value is therefore conditional: today the 3 ms/token encode is OVERLAPPED inside the poll(Wait) (removing it alone buys ~0 wall). Remedy order: (i) lift the two hardcoded caps (Issue 1217's lever — worth ~0.9 ms/token on the wall first), (ii) then encode sits on the critical path, (iii) then the ICB replay buys wall. T2's break-even estimate must price this sequence, not encode alone.

## What this does NOT claim

- Not a promotion/demotion of any shipped posture: the resident default (1165 flip) and its league cells are untouched; this is a diagnostic instrument row + the 1181 gate verdict.
- Not a CUDA lane claim: the 4090 prefill graphs arm is a different engine class (the issue's own scope note).
- The bmo confirmation cell (`rig run bmo --exclusive`, job `01a1daaa-fa45`) was queued behind that box's shared lane; it runs when the slot frees and appends a cross-box confirmation row here in a follow-up commit. The verdict does not wait on it: the margin is 3× the bar, wider than any plausible load inflation, and the encode floor (2689 µs min) is an order-independent statistic.
