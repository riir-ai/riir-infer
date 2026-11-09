# Issue 1198 — `multi_stream_g1::two_streams_overlap_serial_vs_interleaved` reds on amo (ratio 2.7–3.3, want < 0.85) — pre-existing at the 1196 base, not a fixture-change regression

**Status:** OPEN — found 2026-11-09 while re-running the 1196 kit-consumer gates on amo (M5 Ultra); reproduces at the PRE-FIX base `83d491b` in a clean sibling worktree, so it is NOT the `c59e57f` fixture change.

## The finding

`cargo test --release -p riir-infer-dist --test multi_stream_g1 two_streams_overlap_serial_vs_interleaved`:

```
overlap: serial 25.108ms interleaved 82.251ms ratio 3.276   (base 83d491b, clean worktree)
overlap: serial 26.344ms interleaved 74.959ms ratio 2.845   (post-1196 5ff8df5)
overlap: serial 25.976ms interleaved 69.660ms ratio 2.682   (post-1196, alone re-run)
```

The bar wants interleaved/serial < 0.85 ("two streams round-robined take clearly LESS than running them serially"). Measured: interleaved is ~2.7–3.3× SLOWER than serial. All CORRECTNESS asserts in the same test pass (`reports[i].generated == want`, serial-vs-single identity) — this is the timing-ratio leg only.

## Attribution so far

- **Pre-existing**: reproduces at `83d491b` (pre-`c59e57f`) in a clean worktree — the 1196 fixture fix did not cause it. The m3 lane's 1196 close-out re-ran "multi_host 10" green (`plan623_multi_host_g1`) — a DIFFERENT binary than `multi_stream_g1`; this target was not in their re-run set.
- **Box-state caveat**: the test's own comment says "both arms scale with box load together; the real G2 cells belong to the bin on quiet hardware". amo was carrying sibling cargo builds during the first measurements, but the ALONE re-run (nothing else compiling) still reads 2.68 — and the base-commit worktree run reads 3.28 — so load alone does not explain it. Either amo scheduling serializes the two streams' threads (the round-robin design assumes interleave progress), or the plan-623 T2.4 round-robin regressed on this geometry.
- The serial arm runs TWO separate `run_multi` invocations (fresh connections each); the interleaved arm runs both streams over ONE connection round-robin. An interleaved/serial ratio > 1 means the single-connection round-robin is ~6× worse per-stream than fresh connections — the opposite of the design intent (Bench 908 measured the round-robin WIN).

## Structural analysis (zed-34912c3b@amo, 2026-11-09, code-read — the T2 leg advanced without a quiet box)

The wave loop (`run_waves`, drain-1-per-pass at `HeadOptions::default()` — `pair_decode = false`, `prefill_chunk = 1` so PROMPT_A's 5 positions ride per-position frames):

1. **Pipeline shape**: the head sends at most one work item per ready stream per pass, then drains exactly one reply. In decode steady state this leaves ONE activation in flight (pass N sends stream A, pass N+1 sends stream B while A's reply is already buffered — the tail runs one pass behind the head). The per-step period is `max(head_step, tail_step) + wire`, and the structural ratio is `max(h,t)/(h+t)` — **bounded ABOVE by 1.0** (the bubble-fill claim: interleaved can never be worse than serial, only approach it as h/t imbalance grows).
2. **The measured 2.7–3.3 EXCEEDS that bound.** A ratio > 1 is impossible in the model above; something pathological is added per trip on amo in the interleaved arm specifically (per-trip arithmetic: serial ≈ 25.1 ms / 66 reply trips ≈ 380 µs; interleaved ≈ 70–82 ms / 66 trips ≈ 1.06–1.24 ms — ~700 µs/trip of extra cost, roughly 2 wakeups' worth). Prime suspects: rayon-pool wakeup storms (every stage step forks onto the global pool via `on_rayon_worker`; the synth model is n_embd 1024 × 4 layers — per-step compute is µs-scale, so the trip is syscall/wakeup-dominated) or QoS-band thread placement penalizing the 4-instance alternation.
3. **Bench 908's passing 0.805 was a DIFFERENT posture**: debug build, contended box (serial 352.9 ms — compute inflated ~10×, which BALANCES h and t and makes the one-pass-behind pipeline form cleanly). Today's release build shrinks compute until ε dominates — the ratio's meaning changes posture-to-posture. The debug-profile re-run (planned) tests exactly this.
4. **The instrument exists**: `RIIR_DIST_STEP_TIMING=1` (the T3.5 scoreboard) prints per-role per-bucket decomposition (head recv/send/submit/finish; tail recv/fwd/pfwd/send) flushed every 64 steps — the armed run will name where the extra ~700 µs/trip lives without any code change.

### T1 (second-box) status

- **4090 repro QUEUED** (job `01a1c128`, `--shared`, 2026-11-09 ~16:20Z): the CPU-only test compiles and runs on the Windows box — a DIFFERENT OS/scheduler/core-topology axis. Ratio < 0.85 there ⇒ amo/macOS-specific (T2 box branch); ratio ≈ 2.7+ there ⇒ lane-structural. Includes the step-timing decomposition.
- bmo (M-series, the same-silicon axis) stays behind the i1117 exclusive job (~11 h); m3 is BUSY (14 contenders). An M-series second-box run queues after either frees.
- amo's own instrumented re-run queues behind the i1191 exclusive Metal job (281d1991, in flight all evening — timing isolation law).

### T2 decision tree (pre-registered before the measurements)

- Step-timing shows interleaved head `recv` ≈ 0 (never blocks) AND tail fwd per-step ≈ serial ⇒ pipeline formed and the extra cost is elsewhere — measure again (wakeup distribution).
- Interleaved head `recv` ≈ the whole trip ⇒ lockstep (tail-bound or pipeline collapsed) → compare tail fwd interleaved vs serial: if tail fwd inflated ⇒ stage-alternation cache effect; if tail fwd ≈ serial ⇒ the head is the blocker (rayon wake latency).
- RAYON_NUM_THREADS=1 re-run: if the ratio recovers (< 0.85), the pathology is the tiny-compute/wakeup regime, not the scheduler — the fix is then either the test's posture (pin a deterministic compute posture) or the bar's meaning (overlap-PRESENCE belongs to pair_decode, the co-located arm the BatchN lane was built for).
- Debug-profile re-run ≈ 0.805 on today's amo ⇒ the 908 record reproduces and the red is a RELEASE-posture artifact — the test then needs a posture-stable assertion (the ratio bar is posture-sensitive by construction).

## Tasks

- [ ] T1 — reproduce on a second box (bmo or m3) to split box-scheduling vs lane regression; record box state per the latency-claim law (`scripts/bench_preflight.sh` PROVENANCE or the Windows twin).
- [ ] T2 — if lane: bisect the plan-623 T2.4 round-robin (`fc6f546`, Bench 908) against its bench posture; if box: document the amo scheduling caveat on the test's ratio bar (or gate the bar on a quiet-box detector).

## Session

zed-34912c3b@amo, 1794237218 (the 1196 claim lease holder; 1196 itself resolved by the m3 lane — see HISTORY.md 2026-11-09).
