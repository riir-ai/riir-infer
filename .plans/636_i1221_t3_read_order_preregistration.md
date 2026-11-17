# Plan 636 — i1221 T3 pre-registration: the real-looped checkpoint bench payload (read order + instrument inventory + resumability format)

**Status:** OPEN — filed 2026-11-17 (zed-b2e60a87@amo) under the i1221 lease. This plan pre-registers what `.issues/1221` T3 requires before its first bench: the read order, the in-stack instrument inventory (consume vs port), and the durable partial-row format. **No bench runs under this plan until the box window discipline allows it** (T3 cells are minutes-per-token CPU; the i1226 walk lane's exclusive-window poison precedent — riir-infer `96b44752` — is the rule: never start a competing CPU-heavy cell inside another lane's declared window). **Leg (d) DONE 2026-11-17** — instrument (`256c8647`), runner resume-scan fix (`27296d6e`), cell + bench record ([`.benchmarks/1235_i1221_t3d_loopability_dense_cell.md`](../.benchmarks/1235_i1221_t3d_loopability_dense_cell.md), `9c41f005`: 12 rows, contraction holds / loop deepens / no loop-invariance = T3(a)'s dense baseline). **Next act: leg (a)** — per-loop KL/argmax-flip spectra on the Dory pin (the Issue-898 `kl_depth_probe` port onto the T2 delta-report posture).

## Why pre-register

The issue's T3 is the unblock payload — the first real looped-trained checkpoint running the stack's instruments — and it names four instrument families ((a)–(d)) without pinning the order. The read order is registered BEFORE the first run so a partial campaign (the 20-min+ cells WILL straddle wake cycles) resumes into a stable plan instead of an ad-hoc next step. Report-only, no exit-code gates; every cell quotes box state + the STAMP/preflight discipline.

## The read order (pre-registered; cheapest-signal-first)

1. **T3(d) — the loopability predictor, dense-GGUF arm.** The ONLY cell that does not need the Dory forward to be fast: chunk a DENSE GGUF's middle block, iterate `s_t = N(F_blk(N(s_{t-1}+x)))`, read per-k contraction `λ_t = ‖Δs_t‖/‖Δs_{t-1}‖` + KL(full-depth‖loop-k). Small candidate set (the dense checkpoint on the pin shelf), short runs, and it EXTENDS an existing instrument (`loop_probe`, Issue 929 posture) rather than porting one. Its output also calibrates expectations for (a): a loopable dense block predicts small per-loop KL.
2. **T3(a) — per-loop KL/argmax-flip spectra on the Dory pin** (the Issue-898 `kl_depth_probe` port). k=1..4 per-loop logits over a fixed prompt set (the smoke's pseudo-token law until a tokenizer lands); per-loop KL divergence + argmax flip counts, incremental rows. This is T2's reported-delta instrument generalized to the per-loop axis — the machinery exists (`DoryCachePolicy` + the delta-report posture in `tests/i1221_dory_forward_smoke.rs`); the port is mostly loop-and-emit.
3. **T3(b) — halter-style per-loop step-size/angular traces → cost_floor calibration readout** (Plan 304's open calibration question with real data). Depends on (a)'s per-loop logit plumbing (the traces read the same per-loop state boundary); runs after (a) reuses its prompt fixture.
4. **T3(c) — tokens-per-task vs k** (the paper's 1.3–2.5× agentic law, direction check). LAST: needs the tokenizer lane (a real prompt set, a stopping rule) — the biggest unpriced dependency; it is the cell most likely to need the Metal posture (long generations). Blocked on the tokenizer decision, which is a separate pre-registration (T3 will amend this plan or file its own when the tokenizer lane lands).

Rationale: (d) needs no forward changes and validates the loop-contraction premise on dense weights first; (a) is the cheapest Dory-native signal and feeds (b); (c) is gated on a dependency that does not exist yet — putting it last keeps the campaign resumable without a dangling first step.

## Instrument inventory (consume vs port — substrate-first)

| instrument | in-stack home | posture |
|---|---|---|
| loopability predictor (d) | EXTENDS `src/deltanet/loop_probe.rs` (Issue 929) — the loop-alignment measurement becomes a predictor | dense GGUF through the existing loader; the middle-block iterator is new but shape-local |
| per-loop KL spectra (a) | the Issue-898 `kl_depth_probe` posture, ported onto the dory lane's per-loop logits (the T2 delta-report posture already emits per-step tables) | CPU, resumable rows |
| halter traces (b) | katgpt-core `gain_cost_halt` (`GainCostLoopHalter`) — consume the halter's trace types; the dory lane emits per-loop step-size/angular rows into them | CPU, after (a) |
| tokens-per-task (c) | none — needs the tokenizer lane + a stopping rule | BLOCKED (named above) |

No new deps; every cell rides the `dory` feature (unarmed builds compile it to nothing — the T5 gate law already claimed).

## Resumability format (the durable-state posture)

Each cell writes incremental JSONL rows (one per prompt × k × loop) to `.benchmarks/<cell-dir>/rows.jsonl` with a header row carrying: pin sha, STAMP commit, k values, prompt ids, box state, and a `schema` version. Resume = read the header, verify pin+STAMP match (mismatch = fresh dir, never append across lineages), skip completed (prompt, k) pairs by their row keys. The bench docs (numbered `.benchmarks/` files) cite the row dirs; the verdict text carries the folded table, never the raw rows.

## Gates

- `py ../katgpt-rs/scripts/dual_allocation_gate.py` + `py ../katgpt-rs/scripts/numbering_gate.py .` at allocation (run at filing).
- Every cell: STAMP = pin lineage, preflight disclosure (structural cells may run concurrent-with-light-load — disclosed; any cell with a timing ROW may not: the exclusive-window law above).
- Report-only: no exit-code gates on the instrument outputs; the only asserts are the fixture/identity gates that already exist (the T2 invariant battery).
