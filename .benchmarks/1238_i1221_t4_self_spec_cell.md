# Bench 1238 — i1221 T4: depth self-spec cell (the real looped-trained pin) — the greedy-EXACT identity gate + the acceptance census

**Status:** DONE-IDENTITY-HALF 2026-11-17 (zed-84603cf7@amo) — the pre-registered gate PASSED on the real pin (6/6 arm-cells token-identical to the k=4 AR reference under per-loop caches), the synth gate battery 4/4 green, the acceptance census landed (the k=1 draft NEVER missed: 24/24 drafted tokens accepted across γ=2/γ=4 × 2 prompts — depth self-spec on this pin is an always-accept greedy proposer). The TIMING arm is PENDING a preflight-PASS window: the first timed run was poisoned (a sibling's compile swarm took load to 41 mid-run — the rows file was DELETED, never quoted) and the re-run's window was taken by the i216-merge gate lane; the exact re-run command is in §Timing. Instrument + gates landed this commit.

## What ran

- **Instrument:** `src/dory/self_spec.rs` (new, behind `dory` — unarmed builds compile it to nothing): `ar_greedy` (the reference arm) + `self_spec_greedy` (k=1 draft γ tokens → full-depth verify replay → greedy EXACT accept walk → round-end repair). The round structure + the identity argument are in the module doc: every emitted token equals the k=4 target argmax at its position over exactly the emitted prefix, because the verify cache carries the same prefix the reference carries (T1 prefill≡decode identity + forward determinism) and the draft only proposes.
- **The repair seam:** `DoryKvCache::truncate(to)` (new in `forward.rs`) — drops cached positions `to..` from every slot (rows are uniform per slot, the no-hole invariant's other half). Needed because a rejected round leaves a phantom drafted suffix in BOTH caches (the verify replay feeds every drafted token before the accept walk runs); both rewind to `prompt.len() + emitted.len() - 1`, then the last emitted token re-arms them (verify → next round's decider row; draft → next round's first draft row). On all-accepted rounds the truncates are exact no-ops.
- **Gates** (`tests/i1221_t4_self_spec_gates.rs`, 4/4 green, synth 6-block k=2):
  1. `self_spec_stream_is_token_identical_to_the_ar_reference` — THE gate: 3 prompts × γ∈{1,3,5}, n=24, exact stream equality + cache alignment at exit (both caches end at prompt+n) + footprint posture pins.
  2. `draft_cache_truncate_keeps_decode_continuity` — the repair seam: a junk phantom at a position, truncated wholesale, then the real token decoded fresh — every vocab lane BIT-IDENTICAL to the clean path (a first draft of this gate compared a t6-row against a t5-row — the test was wrong, not the seam; the fix is the junk-replacement shape the repair actually uses).
  3. `self_spec_identity_across_seeds_with_stats_census` — 8 seeds × n=24 × γ=2, identity + invariants each, with the honest census: **the synth fixture is the ALWAYS-ACCEPT posture** (the k=1 argmax equals the k=2 target argmax at every position — loop near-no-op on random small weights) — the reject arm's proof therefore lives on the real pin (Bench 1236's KL(full‖exit_1) 13.8–24.1 guarantees divergence potential), and this gate discloses the posture rather than pretending a rejection was exercised.
  4. `self_spec_refuses_wrong_postures_loud` — LastLoop verify, partial-depth verify, non-k=1 draft all refuse before any decode (the spec-mode posture is per-loop caches, verify at the checkpoint's full depth).
- **Postures:** clippy `-D warnings` clean at default / no-default / dory all-targets (exit-0 verified); lib untouched at default features (the dory feature still gates everything); P43 rows for the new test + example in the same commit.

## The cell (real pin, Q5_K_M, k_full=4)

- Pin `f946e530…` (the T0 line's Q5_K_M GGUF), STAMP `2d86987e63a4` (HEAD at run; the landing commit carries the same instrument), release CPU, 2 prompts × 12 steps × {ref, γ=2, γ=4}.
- Rows: `.benchmarks/i1221_t4_self_spec_cell/rows.jsonl` (header lineage: pin blake3 + STAMP + prompt law + steps + gammas + `provenance: null` — the identity-only run, see §Timing).
- **IDENTITY: 6/6 arm-cells OK** — the self-spec streams are token-identical to the k=4 AR reference (`stream_blake3` equal per prompt across arms; asserted in-process, the run refuses on divergence).

### The acceptance census (the cell's headline finding)

| prompt | γ | rounds | accepted/drafted | bonus | emitted |
|---|---|---|---|---|---|
| p0 | 2 | 4 | **8/8** | 4 | 12 |
| p0 | 4 | 3 | **12/12** | 3 | 12 |
| p1 | 2 | 4 | **8/8** | 4 | 12 |
| p1 | 4 | 3 | **12/12** | 3 | 12 |

**The k=1 draft is a perfect greedy proposer on this pin** — 24/24 drafted tokens accepted at the trained depth, consistent with Bench 1236's rare argmax flips (KL(full‖exit_1) 13.8–24.1 is full-distribution divergence; the greedy argmax on sharp trained logits is depth-robust) and T2's argmax agreement 16/16 under a k change. Corroborates (a)'s effective-depth 1–4 finding: many positions' logits are already converged at depth 1. **Serving reading:** on GREEDY decode the k-scheduler question collapses toward "when is one loop enough" — the elastic-k halter story from Bench 1237 — rather than "when does speculation pay"; under SAMPLING the identity is provable (Bench-634 law) but the acceptance rate is an open measurement (T4's sampling arm, unclaimed).

### Cost structure (from the counters, load-independent)

With 100% acceptance every round emits γ+1 tokens for γ+1 draft steps (γ chain + 1 repair) + γ+1 verify steps (γ replay + 1 decider) — per emitted token: **1 k=1 step + 1 k_full step ≈ 1.25× the AR reference's compute** (the draft is ¼ of a verify pass at this pin's k=4). Sequential-hardware expectation confirmed structurally: self-spec is a LATENCY structure (draft/verify pipelining in a served posture), never a compute win on one thread — the paper's 1.37–1.71× is the concurrent arm.

## Timing (PENDING — the honest record of a poisoned measurement)

- First timed attempt (13:48): preflight PASSED at load 4.78 (`PROVENANCE: power=AC Power load=4.78 swap=4270.25M canary=skipped powermode=0(auto)`), but a sibling's compile swarm took load to 41.05 BEFORE the run started computing — every wall number landed under a REFUSED posture. **The rows file was deleted; none of those numbers are quoted anywhere.** (Identity receipts from that run were load-independent and re-proven by the landed identity run.)
- Re-run attempt: the window was taken by the i216-merge gate lane (amo BUSY, load 20+).
- **The pending arm:** in a preflight-PASS window (`../riir-reflex/scripts/bench_preflight.sh` PASS), re-run with `--fresh`:

```bash
CARGO_TARGET_DIR=/tmp/i1221t3b cargo run --release --features dory \
  --example i1221_t4_self_spec_cell -- --fresh \
  --provenance "<the preflight PROVENANCE line>"
```

The identity half needs no re-run (load-independent, landed); the timing half is the only open row. Budget ~15 min (the poisoned run's shape: ref ≈ 8 s/tok, spec γ=4 ≈ 12.75 s/tok UNDER LOAD 41 — direction consistent with the 1.25× structural ratio, never quotable).

## Box state

- amo, M5 Ultra, AC, powermode 0(auto); identity run at load 13.6–20.2 (siblings' lanes: i215 proof driver, then the i216-merge gate) — disclosed: the identity rows and counters are load-independent deterministic outputs; the preflight REFUSAL (load 12.53) was honored for the first timing attempt and the load-41 poisoning was caught and discarded. Disk 97→41 GB across the session (the i216-merge worktree's gate target consumed ~50 GB — sibling-owned).

## Files

- `src/dory/self_spec.rs` (the instrument) · `src/dory/forward.rs` (+`truncate`) · `src/dory/mod.rs` (registration)
- `tests/i1221_t4_self_spec_gates.rs` + the `[[test]]` row · `examples/i1221_t4_self_spec_cell.rs` + the `[[example]]` row
- Rows: `.benchmarks/i1221_t4_self_spec_cell/rows.jsonl` (identity lineage; timing pending)
- Issue: `.issues/1221` T4 — identity half DONE, timing row pending; the sampling arm (Bench-634 distribution identity) stays unclaimed
