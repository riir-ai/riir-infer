# Issue 1230 — the resident wall's completion-counter drain: poll ONE u32 per slot instead of fencing the queue (the 1218 lane's named alternative lever)

**Status:** Open — filed 2026-11-17 at Issue 1218's close (T2 measured-negative, [Bench 1229](../.benchmarks/1229_i1218_t2_cadence_wall_cell.md)): the fence-cadence lever cannot act (Issue 1181's pool-bound-queue finding — the serializer is allocation backpressure, not ring depth; the cell confirmed within-run identity −2.415 vs −2.413 ms/tok), so the harvest fence's residual (~78% of the host window per 1218 T1's decomposition) needs a DIFFERENT mechanism: per-token completion counters.
**Claimed:** amo-i1230, 1794873583, expires 2026-11-18T23:59:43Z

## The mechanism (from 1218 T2's own naming)

`ArgmaxLastCubeCL` (the plan-626 resident drain surface) writes a done-counter per slot; the host polls ONE u32 instead of blocking the queue in `client.read_one` for the full burst (~109 ms/burst at arm B). The ring depth becomes irrelevant to the wait — the host wakes per-token (or per mini-batch) on counter reads, the pool's backpressure (the true serializer per 1181) stays untouched, and the fence disappears from the host window.

## Why this lane, why now

- The parked pool this lever chases is MEASURED: resident wall 16.00–16.21 ms/tok at the raised-caps posture vs the Issue-1168 P2 deep-queue floor **15.32 ms/tok** ≈ 0.68 ms/tok — 1218 T2 proved depth is not the mechanism; the fence read is the named residual.
- The lever is substrate-side (resident decode fence mechanics, plan-626 surface — riir-infer), consumer wiring riir-ai-side, exactly like `RIIR_RESIDENT_HARVEST_K` (`d2255b8b` + `b70707199` — the resolver pattern + the bench_831 seams are already in place for a sibling knob).

## Pre-registered bars (2026-11-17, BEFORE any cell — the 1229 lessons folded in)

- **Quiet exclusive window is a HARD premise** (1229's cross-run delta was refused on a loaded box; the drift control cost the run its verdict): amo/bmo exclusive, no heavy co-tenants (`rig run --exclusive` + preflight `rig status` heavy-process check recorded beside the numbers).
- G1 walk identity both arms (the FNV walk pin `121982e1b7878d05` at ctx=2048 n=512 is the anchor) + stop-identity cells incl. a mid-burst stop (the 1229 stop battery is the template).
- **Adopt iff resident wall improves ≥ 0.3 ms/tok** vs the same-run incumbent (the within-run delta is the instrument of record — never a cross-run pair-mean) AND the fence read leaves the host window (1218 T1's decomposition counters re-run at the armed posture: the un-accounted ~78% share must collapse).
- TTFT must not regress beyond noise (1229 measured K32's TTFT cost — a per-token drain should IMPROVE it; regress = measured-negative on that axis alone).
- Any parity drift or wall regression = measured-negative, stays opt-in. Env-switch law: opt-in env first (PROMOTE-ONLY, the resolver pattern), no default change without the GOAT gates + the 1165-style owner verdict; the league bout (plan 626) is the promotion lane's confirmation cell.

## Tasks

- [ ] **T1 — the substrate seam**: the done-counter write site + the host poll path (a `RIIR_RESIDENT_DRAIN` resolver following the `resident_harvest_cadence()` pattern: valid-literal arms, everything else byte-identical incumbent, parser grammar pinned pure, armed value disclosed once on stderr).
- [ ] **T2 — the wall cell** at the pre-registered bars (the `run_cell.sh` template + the within-run-delta instrument; quiet exclusive REQUIRED).
- [ ] **T3 — GOAT cell** (only if T2 passes).

## Boundary

riir-infer substrate (the resident decode lane's fence/poll mechanics; plan-626 surface). Consumer-side routing stays riir-ai. No default posture changes without the GOAT gates + the 1165-style owner verdict.
