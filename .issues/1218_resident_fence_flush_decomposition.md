# Issue 1218 — the resident wall's residual host sync after the caps lift: the fence/flush decomposition cell

**Status:** open — filed 2026-11-15 from [Bench 1218 §"arm B"](../.benchmarks/1218_i1181_t1_encode_share_demote_gate.md)'s verdict (the 1217 caps A/B measured NEGATIVE on the wall bar with engagement proven).
**Claimed:** zed-a0615aba@amo, 1794693900, expires 2026-11-16T21:30:00+07:00 — scope: T1 (the counters-decomposition cell at the raised-caps posture, queued bmo exclusive behind the i1189 T4 feeding cell + i177 duel smoke), then T2's lever naming from its row; T3 stays gated on T2's pre-registered bars.

## The measured facts this issue starts from

- The 1217 caps lift WORKED as diagnosed and moved nothing: with `CUBECL_WGPU_MAX_TOTAL_TASKS=16384 CUBECL_WGPU_MAX_PENDING_WRITES=16384` the resident enqueue distribution went unimodal pure-encode (p50 2369 µs — the 14.4 ms blocking cluster gone, the queue genuinely runs ring-deep), and the wall read **16.00 ms/tok vs the incumbent's 16.05** (0.05 < the pre-registered 0.4 ADOPT bar).
- The host wait per 8-token burst is CONSERVED (~128 ms both arms): what the enqueue gave up, the **harvest fence absorbed** — p50/burst 25.29 → **108.99 ms**. The resident lane is FENCE-BOUND, not cap-bound.
- Arm A's incumbent counters (job `01a1dabb-de8f`): REGULATE=75.0% (c=108), WRITE_POLL=0.0% (c=0), **flush=77.8% (c=216)**, launch=79.5%, exec=69.7% — flush is ALREADY a top occupant beside regulate.
- The parked gap this issue chases: resident wall 16.00 vs the Issue-1168 P2 deep-queue floor **15.32 ms/tok** ≈ 0.68 ms/tok — which the caps lift provably does NOT reach (ring-deep queue, unchanged wall), so "depth" as such is not the mechanism; whatever separates 16.00 from 15.32 lives in the fence/flush/sync decomposition or the P2 posture differing in more than depth.

## Tasks

- [ ] **T1 — the decomposition cell (cheap, first):** re-run the arm-B posture WITH the counters window (`counter_probe` in the feature set — arm B's run lacked it, the counters line never printed): REGULATE should collapse toward 0 (engagement corroborated from the counter side); read what the 100% host-busy decomposes into at the raised caps — expect flush and/or the harvest fence's completion sync to dominate. One exclusive run, ~25 min.
- [ ] **T2 — the lever, named by T1 (conditional):** if the fence drain dominates → the cadence lever (harvest every K > 8 tokens, or a completion-counter drain that returns per-token results without the full-queue fence — plan 626's own resident surface, `resident_harvest`); if flush dominates → the flush cadence is the lever (its trigger/cap, the vendored stream's flush site). Design + pre-registered bars BEFORE any cell; the env-switch law applies (opt-in first).
- [ ] **T3 — GOAT cell per T2's lever** (G1 walk-identity both arms; wall bar honest — the 0.68 ms/tok pool is the ceiling, never promise it).

## Boundary

riir-infer substrate (the resident decode lane's own fence/flush mechanics; plan 626 surface). Consumer-side routing stays riir-ai. No default posture changes without the GOAT gates + the 1165-style owner verdict.
