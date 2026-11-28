# Issue 1240 — ANE background-core POC + the ANE↔GPU contention cell (research GOAT; prior = anemll V8.1 DRAM measurement)

**Status:** OPEN

**Routing note (2026-11-23, zed-m3 housekeeping sweep):** the m3-held lease (`m3-session`, expired 2026-11-22T07:31:47Z) was a mis-claim — this lane's own T0 demands an EXCLUSIVE-QUIET Apple-silicon rig (`agent-mail rig status`; amo/bmo booking), and the pair cell's resident-decode posture is the plan-626 lane on M5 Ultra. The next claimer books a RIG lease (`rig run --exclusive --needs metal`), never an m3 lease; reflex Issue 123's ANE probe is the tripwire every cell here must respect.
**Lease closed:** zed-56f3ce1e@amo, 1795455611, was expires 2026-11-25T17:40:11Z — scope ADVANCED to landing: the T0–T1 pre-build landed under its fold-forward chain (`e7ba669f` the ANE encode-load driver + the pair-cell orchestrator, `32c342bf` the tick-line kill fallback); the pair cell itself stayed unrun (b159e516's non-exclusive runs were contender-killed ×2 and the exclusive re-queue was adjudicated but never booked); folded at the 2026-11-26 re-claim below.
**Lease closed:** zed-35dc6ba9@amo, 1795671452, was expires 2026-11-28T05:37:00Z — lapsed undischarged (the booking never landed; holder session gone). Scope passed through unchanged, re-claimed below.
**Claimed:** zed-3fa07e8b@amo, 1795846106, expires 2026-11-30T13:10:00Z
(scope: book the adjudicated exclusive pair-cell re-queue — amo, `rig run --exclusive --est 3h`, the run from a fresh origin/develop worktree with results to the durable `~/riir-bench/i1240_cells` out dir; preconditions re-verified 2026-11-28T13:08Z GREEN on amo — load bin warm `/tmp/i1240-target`, decode target warm `/tmp/i831m5` (2 bench_831 deps), ANE artifacts + manifest present, corpus present, model present; riir-ai main tree carries only docs WIP so the decode arm compiles clean source; harvest + the T4 GOAT adjudication via job-end mail. The `--needs metal` tag dropped from the booking — amo is the named rig and the tag class is cuda-documented; exclusivity is the load-bearing property.)

**Filed:** 2026-11-20, from the owner's hybrid-ANE discussion (Research 392 addendum, same day) + anemll's V8/V8.1/V8.2 hybrid article (X post 2122862729212145674, branch `v8-gpu-ane-prefill`). RESEARCH cell, not a product promotion: does concurrent ANE work cost our Metal decode (the reflex Issue-123 3–45% band), and can ANE serve cache-upkeep batch work as a background core? A NEGATIVE verdict is a valid landing (DONE-NEGATIVE class) — the demote stands, the record is the product.

## The question

reflex Issue 123 measured concurrent ANE activity skewing GPU numbers 3–45% (the `bench_preflight` ANE probe) — mechanism unattributed. anemll's V8.1 contention measurement (M6, `ARTICLE_CONTENTION_M6.md`) attributes it: **DRAM bandwidth + latency, not power/clock** — attention ~93 GB/s + FFN ~85 GB/s ≈ 180 GB/s demanded from ~155 GB/s delivered → ~134 shared, GPU memory latency ~doubles, and the ARBITRATION IS ASYMMETRIC (ANE FFN slows 0–6% beside synthetic GPU loads; memory-heavy GPU loads lose 23–78%; GPU held 1,735 MHz throughout; a 2 W memory stream hurts its partner more than 21 W matrix work). If the asymmetry holds on our silicon, "ANE as background core" (cache-upkeep batch encode beside resident decode) may be near-free GPU-side — or it may not; our M3/M5-class ANE generation + our mixed workload is unmeasured.

## Tasks

- [ ] T0 — provenance + posture: `bench_preflight` PROVENANCE line (reflex Issue-123 ANE axis armed), exclusive-box QUIET rig (`agent-mail rig status` — amo/bmo first; m3 only if quiet), box state + ANE generation recorded (ours ≠ their M6; generation drift is a documented class)
- [ ] T1 — the pair cell (the GOAT leg): resident Metal decode tok/s (riir-infer plan-626 lane, greedy, ctx-2048 posture) ± concurrent ANE batch-encode (laya-riir-ane encoder lane over a real corpus slice). Interleaved A/B (sequential arms measure the box — the ab_timing law), ≥3 rounds, decode AND encode deltas both recorded. Pre-registered bars: decode delta <5% = co-residency GO; 5–20% = scheduling-lane class; >20% = NO-GO (background core refuted on this silicon)
- [ ] T2 — attribution cell: what moved — DRAM contention vs dispatch interference? their telemetry method (bandwidth/clock/power) as far as our tooling permits; asymmetry check (expect ANE-side delta ≪ GPU-side per their arbitration result)
- [ ] T3 — cache-upkeep POC: a real batch-encode job (riir-refine `index_cache` embedding lane or reflex corpus embeddings) served on ANE while decode serves; end-to-end upkeep throughput + decode impact
- [ ] T4 — GOAT adjudication: verdicts per the pre-registered bars, delegated-owner ping-pong, bench record under `.benchmarks/` with box state; keep opt-in / demote per the env-switch law either way
- [ ] T5 — lane-intel writeback: anemll lessons adoptable here — fewer/bigger ANE calls (0.23 ms per-call firmware gap; their 18% FFN win); GPU keep-alive for bursty-GPU lanes (clock 502–874 → 1,735 MHz for +2–3 W); measure-the-pair doctrine (their per-unit-favored offloads ALL lost end to end); ANE thermal −30–45% after ~4 min continuous; never two threads on one Core AI function (silent corruption); ANE INT8 forms silently wrong → host-reference check (our G5 posture corroborated). Route: the ane lane doc + this issue's close note

## Non-goals

- No 27B ANE decode lane — their V8.2 decode (45.3 tok/s @96K, DFlash2 + SPLIT+GPU + keep-alive) is their product path; ours stays Metal. This issue prices ONLY the contention question + the background-core idea.
- No SIP-off paths ever — their weights-once trick (GPU maps ANE's compiled hwx in place, 10.8 GiB zero-copy) requires SIP disabled + root (datavault flag); research observation only, never on fleet boxes.
- No bench on a BUSY rig; never benchmark a box running a sibling agent's lane.

## Refs

- Research 392 addendum (2026-11-20): the V8/V8.1/V8.2 record — `../riir-ai/.research/392_Anemll_Forge_ANE_GDN_Numerics_Distill.md`
- anemll X article 2122862729212145674 (Version 2, 2026-11-19) + `ARTICLE_CONTENTION_M6.md` on branch `v8-gpu-ane-prefill`
- reflex Issue 123 (the 3–45% ANE→GPU skew band; the `bench_preflight` ANE axis)
- riir-infer plan 626 (the resident-decode lane) · reflex Issue 017 (the laya ANE lane, G5-ANE 76/76)
- Sequential-AB law: `../katgpt-rs/scripts/sequential_ab_timing_audit.py` class
