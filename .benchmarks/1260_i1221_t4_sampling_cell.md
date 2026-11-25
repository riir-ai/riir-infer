# Bench 1260 — i1221 T4 sampling cell on the real looped-trained pin (plan 638 T3)

**Verdict: PASS — all pre-registered receipts GREEN on run 2 (schema 2).** The sampling
arm of the depth self-spec rider (plan 638; the residue Bench 1238 named open) satisfies
every receipt the plan pre-registered for the real-pin cell: the induced-law identity
receipt, the acceptance census with its predicted-acceptance cross-check, the determinism
receipt, and the report-only first-token two-sample table. Report-only + receipts — no
timing rows (plan 636's window discipline: structural cells run concurrent-with-light-load,
disclosed; the TIMING rows stay parked).

## Provenance (the two runs, and why there are two)

| | run 1 | run 2 (the record of record) |
|---|---|---|
| date | 2026-11-23T07:02Z | 2026-11-25T06:50Z |
| build stamp | `330b3293` (pre-fix census) | `5f3952e7` (fixed census, schema 2) |
| prompts | **ONE prompt twice** (the LCG law's `seed + prompt_idx` collides through the generator's `seed | 1` start — sets 0 and 1 shared one state) | **two distinct prompts** (`2·(seed + prompt_idx) + 1`, odd → passes the `| 1` start) |
| KL census | `null`/`inf` (the unrestricted KL(p‖q) is +inf on a 131k vocab — q underflows to exactly 0 where p > 0) | support-restricted (p ≥ 1e-9, q floored) + the law's own predicted-acceptance units |
| first-token `TV(spec,ref)` | **`461168601842738.8` — GARBAGE** (`((a−b) as f64` on usize counters underflow-wrapped before the cast) | `0.0033` / `0.0004` — sane |
| rows file | preserved: `run1_rows_stamp_330b3293.jsonl` (job log: `run1_job_01a20729.log`) | `rows.jsonl` |

Run 1 was executed by the holder's queued job (`01a20729-6915`, exit 0, 2026-11-23); its
rows + the holder's schema-2 cell WIP sat **stashed untracked** by the 11-24 sync sweep
(stash `57b16426`) when the holder (`zed-0a464098@amo`) retired at the 06:00Z lease expiry.
The re-claim (zed-fba187f9@amo, `5f3952e7`) recovered both, repaired the WIP's
format-string escape break, compile-gated it, and ran run 2. **Consequence for Bench 1238:
its "2 prompts" census was one prompt sampled twice — its greedy-identity gates (per-stream
exact token identity) remain valid on the one prompt; only the prompt-count prose narrows.**

## Cell posture

- Model pin `../riir-train/data/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf`, blake3
  `f946e530236859ff…`, arch `dory`, k_full = 4, vocab = 131072.
- Arms per prompt at T = 1: self-spec sampling γ=2, γ=4, plus a second γ=4 pass at the same
  seed (the determinism receipt); n = 12 steps; 20,000-draw first-token table per prompt.
- Law per plan 638: draft q = softmax(k=1 row), target p = softmax(k_full row), accept
  `min(1, p/q)`, first reject emits from `normalize(max(0, p−q))`, all-accept bonus from the
  last target row directly; lazy verify (a drafted token enters the verify cache only once
  accepted); seeded xorshift64* → f64, one uniform per draw in a fixed order; f64 softmax,
  sequential ascending folds.
- Concurrency disclosure (in the rows header): `13:50 up 2 days, 18:54, 6 users, load
  averages: 14.40 9.09 7.73` — structural/no-timing, per plan 636 disclosed.

## Receipts (run 2)

| receipt | bar (plan 638) | p0 | p1 | verdict |
|---|---|---|---|---|
| **Induced-law identity** (every judged (p,q) pair, in-process assert — a miss refuses the cell) | `max|induced − p| ≤ 1.2e-10` (V=131072) | 1.25e-16 (8 / 13 pairs) | 6.0e-17 (8 / 12 pairs) | **PASS** — 6 orders under the bar |
| **Acceptance census** (the headline open measurement) | report | γ=2: 8/8 (100%), 3.00 tok/round · γ=4: 12/16 (75%), 4.00 tok/round | γ=2: 8/8 (100%), 3.00 · γ=4: 12/12 (100%), **5.00 tok/round** | recorded; see reading below |
| **Predicted-acceptance cross-check** (the fixed census's own units) | consistency | Σmin mean 0.994 → expected rejects 0.05, observed 0–1 | Σmin mean 1.000 → expected 0.00, observed 0 | **consistent** (the γ=4/p0 reject sits within 1 draw of the 0.05 expectation) |
| **Determinism** (arm 2 = arm 1's seed again) | stream digests equal | equal (`889e69e2…` both) | equal (`0d79971b…` both) | **PASS** |
| **Counter laws** (G4-shaped, re-checked from the rows) | `verify_steps == accepted + rounds`; `draft_steps == rounds·γ + bonus`; `accepted + rejected + bonus == emitted_uncapped` | hold all arms | hold all arms | **PASS** |
| **First-token two-sample table** | REPORT-ONLY (never gated — at any affordable N the 131k-vocab TV is noise-dominated) | TV(ref,p) 0.0029 · TV(spec,p) 0.0004 · TV(spec,ref) 0.0033 (noise expectation ≈ 0.002) | TV(ref,p) 0.0004 · TV(spec,p) 0.0001 · TV(spec,ref) 0.0004 (noise ≈ 0.001) | consistent with identity |

## Reading (the census's findings)

1. **The k=1 draft is a near-perfect proposer on this pin.** Acceptance 75–100% across
   prompts × γ; the predicted-acceptance law (Σmin(p,q)) predicts 0.99–1.00 and the
   observed rejects land on it (p0 γ=4: expected 0.05, observed 1 in 16 judged pairs —
   inside one-draw noise). This **confirms at T=1 on real weights** what Bench 1238's
   greedy census found (24/24 greedy-exact): the loop's depth pairs barely diverge under
   this checkpoint's trained loop, so speculative drafting across depths is nearly free.
2. **The depth divergence, quantified:** support-restricted KL(p‖q) mean 0.015–0.33 nats
   (max 2.63 on p0's first judged pair; p1 an order of magnitude lower). The two prompts
   span the pin's regime: p0 (KL 0.20–0.33) rejects occasionally at γ=4; p1 (KL 0.015–0.22)
   never does and reaches 5.00 tok/round at γ=4 — **the acceptance axis is prompt-dependent,
   exactly what the elastic-k halter (T3b, Bench 1237) is the instrument for.**
3. **Sampling-mode cost structure:** at 100% acceptance the chain costs 1 k=1 draft step +
   1 k_full verify step per emitted token (≈ 1.25× AR compute at the greedy census's
   accounting) — a latency structure (draft step is cheap), never a compute win. Unchanged
   from 1238's greedy reading; sampling does not change it.

## Files

- `rows.jsonl` — run 2 (schema 2), the record of record (9 records: header + 6 arm rows + 2 first-token tables).
- `run1_rows_stamp_330b3293.jsonl` — run 1 preserved (pre-fix census; KL null, TV garbage disclosed above).
- `run1_job_01a20729.log` — run 1's job log (exit 0).
- Example: `examples/i1221_t4_sampling_cell.rs` (schema 2: the prompt-collision fix + the
  `tv_spec_vs_ref` cast-order fix + the fixed divergence census from `5dd3be79`).

## Session

zed-fba187f9@amo, epoch 1795589223 (the i1221 T4-residue re-claim, lease exp
2026-11-27T07:00Z). Plan 638 T3 closes with this record; issue 1221's T4 sampling arm
closes with it (the TIMING rows stay parked for a preflight-PASS window, per plan 638).
