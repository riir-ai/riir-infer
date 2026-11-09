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

### T1 (second-box) — DONE 2026-11-09, verdict: LANE-STRUCTURAL (not amo, not rayon)

- **4090 (Windows, 24 cores, quiet — 4% load, 14.3 GB RAM free, HEAD e595342) REPRODUCES THE RED**: ratios **1.228 / 1.371 / 1.302** release ×3, **1.298** with RAYON_NUM_THREADS=1 (unchanged ⇒ NOT rayon wakeups). amo's 2.7–3.3 and the 4090's 1.2–1.4 are the same mechanism at different magnitude (cache geometry).
- **Step-timing decomposition names the mechanism** (the pre-registered decision tree, branch 2: *tail fwd inflated ⇒ stage-alternation cache effect*):
  - Serial arm (n=32): tail fwd 0.19–0.25 ms/step, head recv 0.19–0.24 ms/step — the lockstep ping-pong, head recv ≈ tail compute.
  - Interleaved arm (n=64): **tail fwd inflates to 0.57–0.60 ms/step (2.6–2.9× serial)**, tail recv collapses to 0.01–0.05 ms (never blocks — the one-pass-behind pipeline IS formed; the arm is TAIL-BOUND), head recv ≈ 0.32–0.38 ms/step (waits on the slow tail).
  - Arithmetic closes: per-step interleaved ≈ t_inflated ≈ 0.60 ms; serial ≈ h+t ≈ 0.42 ms; ratio ≈ 1.4 ≈ measured.
- **Mechanism (leading hypothesis, measured to this point)**: the tail alternates two per-stream stage instances per step — each carries GDN state (~1 MB) + KV cache (~2 MB) + scratch — so the per-step working set doubles and the state streams instead of staying L2-resident. The RAYON=1 control rules out pool contention; the correctness asserts pass everywhere, so it is purely the alternation tax on CPU-stage compute. The bubble-fill win (saved wakeups + overlap) is smaller than the tax at synth-compute scale — the structural bound analysis (`max(h,t)/(h+t)` ≤ 1.0) does not account for the tax, which is why ratios > 1 are possible at all.
- Regime note: Bench 908's real-model loopback smoke (1.64× WIN, ~90 ms/step compute) and the 2-box GPU cell (2.37×) are compute-bound/GPU-state regimes where the tax is absent or amortized — the round-robin's win is real THERE. The loopback-CPU-synth-release regime is where it inverts.
- **Debug-posture completion (4090, job 01a1c158)**: dev profile reads serial 562.6 ms / interleaved 510.5 ms = **0.907** — the compute-inflated regime recovers toward 908's 0.805 (17× slower steps, tax amortized, bubble-fill win visible) but STILL misses the 0.85 bar. So the old bar is not portable in ANY posture anymore: debug ≈ 0.80–0.91 (box-dependent), release > 1.0 (both boxes). The re-scope verdict stands reinforced: the tripwire is the only posture-stable assertion, and the win claim lives in the regimes that own it.
- Job provenance: first attempt `01a1c128` died in 1 s (rig-run argv malformation — the prose message was passed as the command); re-queued `01a1c153` (script scp'd to the 4090, token-split argv, `--notify` armed). Box state in the log (RAM/load/power lines); step-timing armed per the T3.5 instrument.

### T2 verdict (2026-11-09, evidence-complete)

Neither branch of the original T2 holds: NOT the box (second box reproduces), NOT a lane regression (pre-existing at base + the bar is structurally unreachable in release — see the structural analysis + the measured alternation tax). The bar's PREMISE is posture-bound: Bench 908's 0.805 was a debug-build measurement (compute inflated ~10×, the tax amortized, the bubble-fill win visible); the release posture inverts it on BOTH boxes. Resolution: re-scope the ratio bar (the overlap-PRESENCE claim moves to the regime that owns it — the bin G2 cells / compute-bound postures; the G1 loopback test keeps correctness + a catastrophic-pathology tripwire), and file the alternation tax as its own optimization rung (eliminating it would realize the design intent: per-step `max(h,t)` instead of the inflated tail).

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
