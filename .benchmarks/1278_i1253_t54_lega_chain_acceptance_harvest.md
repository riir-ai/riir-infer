# Bench 1278 — i1253 T5.4 leg-A chain acceptance (the economics gate leg, full 251-window pass)

**Date:** 2026-11-28 · **Box:** amo (M5 Ultra 96 GB) · **Verdict: LEG A HARVEST PASS — the bar-(a)-class agreement reads ≥ bar at the serving posture, and the chain economics inputs (E[m], tokens/round ceiling) are recorded for the T5.5 bar-(b) cell.**

## Run posture

`MTP_ECON_LEG=A /tmp/i1253t54/release/examples/mtp_block_economics` (detached, pid 85628), binary built from riir-infer `bcd15e0b` (04:17:10 +07, 23 s pre-launch), launched 2026-11-27T21:17:33Z / 04:17 +07 by zed-14567e6f@amo per the plan-639 line-124 posture. Log: `/tmp/i1253_t54_legA.log` (2,303 bytes — progress lines + the verdict block; the pre-hardening binary's crash-resilience gap was the accepted disclosed risk; the run did not need it). **EXIT=0 at 2026-11-28T05:25:45Z / 12:25 +07** — wall ≈ 8 h 08 m over 251 windows (windows 0..250 of the registered 256-window cap at stride 56 over the 14,075-token chat file; the same registered n as the T5.4-R combined read). Load: contended through the ride (llama-server + sibling lanes; 5/15-min 11–12 through the day per the plan's 08:12 pace note), 1-min 2.65 / 5-min 3.27 at exit — **counts are load-invariant per the pre-registered rate-cell amendment; the wall is not** (the 08:12 pace law's 2.4–3.5 min/window band is the disclosed wall record). Determinism: the cell's own pin printed `leg A determinism: window-0 pos-1 chain bit-identical across the pass`.

## The verdict block (verbatim from the log tail)

```
=== LEG A — chain acceptance (windows 251, start_window 0) ===
slot 0 (the block's own echo-slot read): 8545/15813 = 0.5404
slot 1: agree 4220/15562 = 0.2712 | conditional 4220/15562 = 0.2712
slot 2: agree 2296/15311 = 0.1500 | conditional 1367/4143 = 0.3300
slot 3: agree 1467/15060 = 0.0974 | conditional 510/1341 = 0.3803
slot 4: agree 855/14809 = 0.0577 | conditional 210/497 = 0.4225
slot 5: agree 592/14558 = 0.0407 | conditional 78/206 = 0.3786
slot 6: agree 459/14307 = 0.0321 | conditional 24/76 = 0.3158
slot 7: agree 401/14056 = 0.0285 | conditional 12/23 = 0.5217
E[m] apple8: sample-mean 0.408 over 14056 full-chain positions (of 251 windows) | from conditional rates 0.417
E[m] nvidia3 (mass ≥ 0.1): sample-mean 0.402 (the honest read — the shared-rate form would ignore the floor's reachability)
tokens/round ceiling (batched verify, echo + E[m]): apple8 1.408 | nvidia3 1.402
```

## Adjudication

- **Bar-(a)-class agreement at the serving posture (slot 0, the echo-slot): 0.5404 ≥ 0.4353 — PASS** (~3.4σ above the bar on n = 15,813: SE of the binomial ≈ 0.00396, margin 0.105 ≈ 26σ on the slot-0 population alone; the conservative read quotes ~26σ, not 3.4 — the 0.5139 T5.4-R single-step read sits in the same ~0.51–0.54 class, the two cells corroborate). The bar-(a) agreement clause of the T5.5 economics gate is **cleared at the serving posture** — the T5.5 Metal cell's precondition (bar (a) ≥ 0.4353) is satisfied by the same-pairing read at chain shape.
- **Chain economics inputs:** E[m] apple8 = 0.408 (sample-mean) / 0.417 (from conditional rates); E[m] nvidia3 (mass ≥ 0.1, the honest form — the shared-rate form would ignore the floor's reachability) = 0.402. **Tokens/round ceiling (batched verify, echo + E[m]): 1.408× apple8 / 1.402× nvidia3** — the upper bound leg B's net-throughput bar (b) (≥ 1.05×) prices against. The conditional-rate tail (slot 7 at 0.52 conditional on reaching it) shows the depth profile is reachability-limited, not accuracy-limited — the mass floor is the binding constraint, matching the nvidia3 honest-read framing.
- **What this cell is NOT:** leg B (the tok/s economics cell) stays QUIET-WINDOW-ONLY and is un-armed by this record; bar (b) ≥ 1.05× is adjudicated by leg B only. This record's ceiling is the analytic upper bound (echo + E[m]), never a net claim.

## Harvest provenance

Harvested 2026-11-28 13:2x +07 by zed-7bab0f17@amo under the PATH-B failover branch (the 16:20 harvest-primary chain pointed at zed-e821112a@amo; that inbox and every sibling lane of the chain are absent from the live thread list — overflow/ended; STEP 0 guards all green: no bench record existed, plan/benchmarks mtimes >15 min stale (last 08:28), the log EXIT marker present, no in-flight sibling writer). Session: zed-7bab0f17@amo, epoch 1795832278 (the i1250-B lease lane, executing the armed 13:20 backstop wake runbook).
