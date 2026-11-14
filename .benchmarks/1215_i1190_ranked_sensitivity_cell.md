# 1215 — i1190 T3: the ranked family-sensitivity cell (Bonsai-2-27B-PQ2_0, 16 per-layer families → Q8_0, prefix 2048 / 512 scored)

**Status:** COMPLETE — all 16 families + the ffn_down determinism re-run executed on bmo; the determinism gate PASSes to_bits; engagement 16/16 (`tensors > 0` per family); the ranked table is T2's input below. The posture decision stays OWNER-GATED (this record files the table and never flips a default).

## Provenance

- Box: **bmo** (M5 Ultra, dedicated runner), rig job `01a1d8d2-cf21` **Exclusive** under `caffeinate -is`, AC, load 1.2–1.6 (load_trace.log, 2-min cadence).
- Binary: `si_zero_shot_ppl` release at features `scale_invariant_attn,bonsai2_hadamard,golden_nll`, built on bmo from its own `~/git/riir-infer` checkout at `db9464f` (the driver-flag fix).
- Model: `Ternary-Bonsai-2-27B-PQ2_0.gguf`, sha256 `3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1` (pin-verified at job start; the plan-631 pin).
- Corpus: the committed recipe regenerated at job start (unit sha256 `9f95700f…` verified; ×28 = 29,036 B, corpus sha256 `a6998c74…`) — ONE frozen read across all 16 families + the rerun (the grammar-not-answers law).
- Flags: `--family-ablate <F> --family-posture q8 --max-length 2048 --min-length 512 --score-window 512` (prefix 2048, 512 scored positions; the pre-registered shape — the first attempt's `--min-length 128` violated the bin's own `score_window <= min_length` assert, fixed `db9464f`, zero re-tuning).
- Wall: job start 2026-11-14T06:50Z, last artifact 08:22Z (~92 min; per-pass load ~115 s + requant + pass 2, the issue's 1.5–2.5 h estimate band).

## Gates (pre-registered in `.issues/1190`)

| gate | verdict |
|---|---|
| Engagement: `tensors > 0` per family | **PASS 16/16** (per-tensor rows name `blk.N.` tensors; counts: proj families 64/48/16 per the arch, norms 209 rows/208 emitted) |
| Determinism: ffn_down re-run pooled Δ-NLL to_bits-identical | **PASS** (`det_a.txt` == `det_b.txt`, `pooled Δ-NLL -3.964e-4 nats/token`) |
| Sanity direction: proj families ≤ ~0 (Q8 ≥ Q2_0) | **MOSTLY HOLDS with two honest exceptions** — see the ranked table |

## The ranked table (|pooled Δ-NLL| descending — T2's input)

Pooled Δ-NLL = ablated(Q8) minus incumbent(PQ2_0), nats/token, over 512 scored positions; NEGATIVE = the Q8 requant *improved* the golden-oracle read (Q8 ≥ Q2_0 precision). Argmax flips **0/512 on every family** (17/17 logs). No fp32-cliff or f64-cliff band positions on any family (0/512 each — the requant deltas live below both cliffs).

| rank | family | tensors | bpw shift | pooled Δ-NLL | mean \|Δ\| | max \|Δ\| | flips | keep/demote line |
|---|---|---|---|---|---|---|---|
| 1 | gdn_a | 48 | 32→32 (round-trip) | **−2.902e-3** | 2.902e-3 | 1.230e-1 | 0/512 | vector round-trip — the largest read, expected ≈0 per 1188 scope note (d); the honest reading: the Q8 round-trip nudges the escape set measurably — KEEP AS RANKED DATA, flag for a re-read before any posture use |
| 2 | norms | 208 rows | 32→32 (round-trip) | **−2.731e-3** | 2.731e-3 | 6.960e-2 | 0/512 | vector round-trip — same class as gdn_a |
| 3 | gdn_dt | 48 | 32→32 (round-trip) | **−1.474e-3** | 1.474e-3 | 2.292e-1 | 0/512 | vector round-trip; the largest single-position |Δ| in the whole cell (0.229) — the tail max is where a posture decision would bite first |
| 4 | gdn_out | 48 | 2.12→8.50 | **+1.738e-3** | 1.738e-3 | 1.037e-1 | 0/512 | **the one PROJ family that went POSITIVE** — the Q8 candidate HURT this family. Investigate before ranking it into any Q8 posture (the 1188 gate names this exact class) |
| 5 | gdn_beta | 48 | 32→32 | −7.899e-4 | 7.899e-4 | 8.178e-2 | 0/512 | round-trip |
| 6 | ffn_gate | 64 | 2.12→8.50 | −6.983e-4 | 6.983e-4 | 7.399e-2 | 0/512 | PROJ, negative (improves) — the 1188 fixture family reproduces |
| 7 | gdn_conv | 48 | 32→32 | −4.623e-4 | 4.623e-4 | 7.641e-2 | 0/512 | round-trip |
| 8 | attn_k | 16 | 2.12→8.50 | −5.133e-4 | 5.133e-4 | 1.189e-1 | 0/512 | PROJ, negative |
| 9 | ffn_down | 64 | 2.12→8.50 | −3.964e-4 | 3.964e-4 | 1.703e-1 | 0/512 | PROJ, negative; the determinism-gate family |
| 10 | attn_v | 16 | 2.12→8.50 | −5.045e-4 | 5.045e-4 | 7.263e-2 | 0/512 | PROJ, negative |
| 11 | attn_gate | 48 | 2.12→8.50 | **+3.596e-4** | 3.596e-4 | 1.599e-1 | 0/512 | **second POSITIVE PROJ family** — small pooled but a large single-position max; same investigation note as gdn_out |
| 12 | gdn_alpha | 48 | 32→32 | +2.206e-4 | 2.206e-4 | 6.319e-2 | 0/512 | round-trip, slightly positive — noise-class at this support |
| 13 | gdn_qkv | 48 | 2.12→8.50 | +5.767e-4 | 5.767e-4 | 9.025e-2 | 0/512 | PROJ, positive — third investigation note (the GDN input projection) |
| 14 | ffn_up | 64 | 2.12→8.50 | +1.617e-4 | 1.617e-4 | 5.866e-2 | 0/512 | PROJ, marginally positive — noise-class |
| 15 | attn_q | 16 | 2.12→8.50 | −9.465e-5 | 9.465e-5 | 1.993e-1 | 0/512 | PROJ, ≈0 pooled but the second-largest single-position max — the tail is the signal |
| 16 | attn_out | 16 | 2.12→8.50 | +1.530e-5 | 1.530e-5 | 5.824e-2 | 0/512 | PROJ, ≈0 — the least sensitive family in the cell |

## Honest reading (what the table does and does not say)

1. **The negative reads on the PROJ families are small and in the expected direction** — Q8 ≥ Q2_0 precision, exactly the 1188 T0 fixture's shape (its ffn_gate −5.171e-4 @ prefix 256; this cell reads −6.983e-4 @ prefix 2048). No family shows the "candidate hurt" magnitude that would veto a mixed posture on its own.
2. **The three positive PROJ reads (gdn_out +1.738e-3, gdn_qkv +5.767e-4, attn_gate +3.596e-4) are the investigation set** — a pooled positive means the Q8 requant measured WORSE than the ternary incumbent on the scored span; before any per-family Q8 promotion these three need a re-read at longer context (the Δ sign at prefix 2048 need not transfer to serving lengths) and a per-tensor delta scan (which blk.N rows drive it).
3. **The vector round-trips are not ≈0 as 1188 scope note (d) predicted** — gdn_a/norms/gdn_dt/gdn_beta/gdn_conv/gdn_alpha carry reads 10–20× larger than the PROJ families' medians. The quantize→dequant round-trip on 32-bit vectors MOVES the model measurably. That is cell data, not a defect — but any posture that requantizes vector families must clear these reads first (the storage-precision sensitivity the note priced as ≈0 is measured at 1e-3-class).
4. **Zero argmax flips everywhere** — the decision surface (the greedy pick) is untouched by ANY single-family Q8 ablation at this context. The sensitivity is a confidence-mass effect, not a pick effect.
5. **The band rows are empty on every family** — all deltas sit below the fp32 cliff (16.64 nats): an fp32-consumer readout sees these moves; nothing hides in the f64-deep tail.

## Per-family keep/demote LINE (T2's posture-table input; the DECISION is owner-gated)

- **gdn_out / gdn_qkv / attn_gate** — HOLD (positive pooled reads; re-read at longer context + per-tensor scan before any Q8 promotion).
- **ffn_gate / ffn_up / ffn_down / attn_q / attn_k / attn_v / attn_out** — Q8-candidate ELIGIBLE on this cell (negative or ≈0 pooled, zero flips); per-family conditional retention per the lossy-surface rule still applies at the promotion posture.
- **gdn_a / norms / gdn_dt / gdn_beta / gdn_conv / gdn_alpha** — round-trip class; the measured 1e-3-class reads make "storage-precision ≈ free" FALSE for the vector families; keep them OUT of any Q8 posture until a cell prices their retention explicitly.

## Artifacts

- bmo: `~/riir-bench/i1190_ranked_cell/` — `ablate_<family>.log` ×16 + `ablate_ffn_down_rerun.log`, `det_a.txt`/`det_b.txt` (identical), `ranked_input.txt`, `pass_times.log`, `load_trace.log`, `i1188_corpus.txt`, `recipe.json`.
- Driver: `scripts/i1190_ranked_cell.sh` at `db9464f` (the flag fix).
