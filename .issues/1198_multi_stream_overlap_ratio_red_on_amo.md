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

## Tasks

- [ ] T1 — reproduce on a second box (bmo or m3) to split box-scheduling vs lane regression; record box state per the latency-claim law (`scripts/bench_preflight.sh` PROVENANCE or the Windows twin).
- [ ] T2 — if lane: bisect the plan-623 T2.4 round-robin (`fc6f546`, Bench 908) against its bench posture; if box: document the amo scheduling caveat on the test's ratio bar (or gate the bar on a quiet-box detector).

## Session

zed-34912c3b@amo, 1794237218 (the 1196 claim lease holder; 1196 itself resolved by the m3 lane — see HISTORY.md 2026-11-09).
