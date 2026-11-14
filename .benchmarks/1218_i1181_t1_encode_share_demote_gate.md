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

## Cross-box confirmation — bmo (job `01a1dabb-de8f` / short `f5e35186`), 2026-11-15

**Date:** 2026-11-15 · **Box:** bmo (M5 Ultra 96 GB runner, exclusive lane, load 1.2–4.4) · **Session:** zed-07a26ed5@amo (adjudicating the landed job for the `amo-idle-i1181` lease lineage)
**Commit:** bmo riir-infer develop tip `f3588305` (the probe + counters feature unchanged from `2a744b5`'s semantics; the 1217 caps env patch `dcb140f` IS in the tree but the envs were UNSET — this row is the incumbent decomposition, not the caps A/B).
**Model:** same `Ternary-Bonsai-2-27B-PQ2_0.gguf`, folded=true, n_layer=64 (loaded from the riir-train/data absolute path).
**Same probe, same posture** (census + uniform-cache override, 3 ABBA pairs, walk-identity assert `ok`, 29.6 s test):

| figure | bmo (this row) | amo (T1 row above) |
|---|---|---|
| resident enqueue p10 | **2486.2 µs** (min 2289.2) | 3552.4 µs (min 2688.6) |
| incumbent front median | 2304.8 µs | 2979.1 µs |
| resident enqueue blocking p50 | 14400.4 µs | 14495.7 µs |
| resident harvest p50 / burst | 25291.8 µs | ~24–26 ms |
| decode wall incumbent | 18.03 ms/tok | 18.87 ms/tok |
| decode wall resident | **16.05 ms/tok** (16.05–16.06 across all 3 pairs) | 16.21 ms/tok |
| resident wall ratio | **0.890×** | 0.859× (0.872× vs G2) |
| host busy share | 100.0% every pair | 100.0% every pair |
| **counters (resident-only window)** | **REGULATE=75.0% (c=108) · WRITE_POLL=0.0% (c=0) · flush=77.8% (c=216) · launch=79.5% (n=108384) · exec=69.7% · write/read=0.0%** | (the counters row this cell was queued to produce) |

**Cross-box verdict: the T1 gate holds on bmo too** — encode ≥ the 1.0 ms bar by 2.3–2.5× (2.49 ms p10 / 2.29 ms min pure-encode), the lane is OPEN on both boxes, and the ~1-deep queue shape reproduces exactly (blocking enqueue p50 ≈ one GPU token; harvest ≈ 2.5 tokens per 8-burst; host busy 100%).

**The counters row is the first DIRECT on-hardware decomposition of the Issue-1217 mechanism, and it REFINES it:** REGULATE occupies **75.0%** of the resident-only window and fires c=108 ≈ once per token (the window is wall 1.540 s ≈ 96 resident tokens at 16.05 ms/tok; n=108384 launches ≈ 1129/token — the 512-cap crossing predicts ~2 fires/token, measured 1.1, the flush counter c=216 carrying the other half — the exact fire/cReset interplay is the vendored stream's, not re-derived here) — while **WRITE_POLL = 0.0% (c=0): the 64-cap write-poll site never fired in this window**. The corrected mechanism's SECOND poll point is therefore not a measurable serializer on this posture (the uniform writes evidently coalesce below 64 in-flight per poll epoch); the ~1-deep pacing is the REGULATE cap alone. Issue 1217's A/B arm B (raised 16384 on BOTH caps) stays the deciding cell — the pre-registered bars read the wall + the REGULATE share, both of which this row now pins incumbent-side (18.03 → target ≥ 0.4 ms/tok improvement; REGULATE 75.0% → materially down). Arm B queued as bmo job `01a1db54-a511` (the original `712bc463` arms died on the probe's cwd-relative GGUF default — `cargo test` runs from the package dir, so `../../riir-train/...` misses; the re-run carries the absolute `BONSAI_GGUF`).

Commits: this one (the row). Full job log: bmo `~/.agent-mail/jobs/logs/01a1dabb-de8f-77ae-bcde-21c9f5e35186.log`.

## The Issue-1217 caps A/B — arm B LANDED, verdict NO-ADOPT (measured-negative on the wall bar), 2026-11-15

**Date:** 2026-11-15 · **Box:** bmo (exclusive, load 1.7) · **Job:** `01a1db54-a511` (the `712bc463` re-run; both original arms had died on the probe's cwd-relative GGUF default — `cargo test` runs from the package dir, so the absolute `BONSAI_GGUF` is load-bearing)
**Arms:** A = the incumbent decomposition above (same box, same binary, ~40 min earlier — the pre-registration names job f5e35186 as arm A). B = same probe, `CUBECL_WGPU_MAX_TOTAL_TASKS=16384 CUBECL_WGPU_MAX_PENDING_WRITES=16384`, absolute `BONSAI_GGUF`, same feature set (⚠ no `counter_probe` — the counters line did not print; bar 3 reads through the enqueue distribution instead, see below).
**Engagement FIRST (proven):** the caps arm engaged — the enqueue distribution changed shape entirely: arm A's bimodal 12.5%-fast/87.5%-blocking (p50 14.40 ms ≈ one GPU token) became **unimodal pure-encode** (n=576: min 2277.0 / p10 2309.2 / p50 2369.2 / p90 2451.4 / max 2910.0 µs). Every blocking enqueue vanished.

| figure | arm A (incumbent caps) | arm B (16384/16384) | bar |
|---|---|---|---|
| resident wall | 16.05 ms/tok (16.05–16.06) | **16.00 ms/tok** (15.99–16.00) | ≥ 0.4 improvement — **FAIL (0.05, ≤ noise)** |
| resident enqueue p50 | 14400.4 µs (blocking) | 2369.2 µs (pure encode) | mechanism engaged ✓ |
| **resident harvest p50 / 8-burst** | **25291.8 µs** | **108985.5 µs** | — the released wait lands HERE |
| host busy share | 100.0% | 100.0% | conserved |
| incumbent wall (control) | 18.03 | 17.96 | consistent |
| G1 walk-identity | ok | ok | PASS |
| encode (lane-open corroboration) | p10 2486.2 | p10 2309.2 | ≥ 1 ms both |

**The wait is CONSERVED, not removed:** per 8-token burst, arm A's host wait ≈ 7 × 14.4 (blocking enqueues) + 25.3 (harvest) ≈ 128 ms ≈ 8 × 16.05; arm B's ≈ 8 × 2.3 (encode) + 109.0 (harvest) ≈ 127 ms ≈ 8 × 16.00. Lifting the caps moved the poll(Wait) out of `enqueue` — and the harvest fence (the plan-626 resident lane's drain-every-8-tokens design) absorbed every microsecond of it. **The resident lane is FENCE-BOUND, not cap-bound**: the ~0.9 ms/token between the resident wall and the Issue-1168 P2 deep-queue floor (15.32) is behind the harvest cadence, which the caps cannot touch.

**Verdict per the pre-registered bars (issue 1217):** G1 PASS both arms; wall improvement **0.05 ms/tok < 0.4 — ADOPT bar FAILS**; REGULATE/WRITE_POLL counters unprinted this run (bar 3 unmeasured directly — moot, bar 2 decides). **The env-configurable-caps lever records MEASURED-NEGATIVE on its adopt bar: the caps envs stay an experiment axis, no default change** (the smaller-but-positive clause applies: 0.05 ms/tok re-tiers the lever's claim and the 1181 T2 break-even re-prices — the sequence is now caps (done, ~0 wall) → **harvest-fence cadence (the named next lever, plan 626's own design: harvest every K>8 tokens or a completion-counter instead of the full drain)** → only then host encode sits on the critical path and the ICB replay's premise becomes testable).

Commits: this one (the arm-B row). Full job log: bmo `~/.agent-mail/jobs/logs/01a1db54-a511-7650-94bc-e8bb39e23928.log`; arm-B console tee: bmo `riir-infer/.benchmarks/i1217_caps_ab/armB_16384.log`.
