# Plan 637 — issue 1189 recipe-3: margin-calibrated teacher-forced recall (differential targets at the first answer position)

**Status:** OPEN — filed 2026-11-23 (amo-idle-i1189r3@amo lease on [issue 1189](../.issues/1189_qwen38_closed_form_write_overlay_cell.md), scope = THIS plan's authoring; cells lease separately per task). T0 code half unclaimed; T1 cell unclaimed. Plan-first law: no cell runs before T0 lands + the bars below stay as written.

## Why recipe-3 exists (the recipe-2 failure anatomy, now three benches deep)

Plan 635 measured all three of its levers negative on the Gitega anchor (Burundi → Gitega, the registered long-tail re-anchor, bench 1242):

- **T1 CAPACITY-REFUTED** — fit pinned at 0.500 at rank 16/32/64 (France fact; the direction family saturates).
- **T2 COVERAGE REFUTED** — all 8 layer-window arms fit 0.000, recall 0/5 Δ 0, determinism ×4 (bench 1248).
- **T3 NOT ADOPTED** — the paraphrase lever hit a relation-local repetition attractor; the armed "recall" was the write feeding itself (bench 1244).

Reading the three together with the solve's actual objective (katgpt-core `closed_form_weight_write.rs` `solve_fact`, read 2026-11-23 at the pinned tree) names the ONE axis no cell has touched:

1. **The target is absolute-likelihood-shaped.** At every answer position the solve regresses the key toward `unit(E[answer[j]] ⊙ g)` with magnitude `D_j = clamp(−lp, 0, cap) + floor` — the answer token's own embedding direction, sized by its raw logprob shortfall. Nothing in the objective references the token generation actually emits. On a mis-ranked fact (the stale-capital profile: Gitega sits at lp −4..−10 while the model emits Bujumbura), raising the answer's absolute likelihood without naming the incumbent cannot flip a greedy decode — and greedy decode IS what `gen_fit` scores.
2. **Refinement reweights, never reshapes.** `update_weights` boosts/decays the SAME absolute targets at the first divergent token (p<0.5 → ×1.5 …). The bench-1219 anatomy recorded the consequence: held-out logprob moved (+0.345, deterministic) while the generated answer did not. T3's attractor is the same shape — post-answer positions got boosted into feeding themselves because boosting likelihood was the only move the loop had.
3. **Depth-zero is the signature.** Fit is `gen_fit`'s achieved-token depth normalized. Fit 0.000 at every window means the write never flipped even the FIRST answer token on the anchor — the battle generation actually fights. Every recipe-2 lever tuned around that battle; none aimed at it.

The issue's own remaining-ladder record names the axis: **teacher-forced recall in the calibration set** (rank/layer swept and refuted; capacity, coverage, and information all measured). This plan instantiates it concretely: the calibration rows already ARE teacher-forced (`ask + answer` capture passes); what changes is the recall target's SHAPE at the position where recall is won or lost.

## The one-line recipe-3 hypothesis

**At the first answer position only, replace the absolute target direction `unit(E[answer₀] ⊙ g)` with the differential `unit((E[answer₀] − E[competing₀]) ⊙ g)` — where `competing₀` is the base model's own greedy first answer token, auto-captured once pre-write — sized by the margin the write must close (`lp(competing₀) − lp(answer₀)`), leaving positions j > 0 absolute: the write gains the ability to push AGAINST the incumbent answer, which is the only move a mis-ranked fact's recall actually requires.**

Mechanics, all inside the existing closed-form pipeline (zero new math beyond one vector subtraction and one logprob difference):

- **Auto-capture (deterministic, base-state):** at the top of `construct()` — before any overlay install, so the state is the pristine base for every fact — for each (fact, training ask): `competing₀ = free_generate(ask, 1, None)[0]`. Same precedent and determinism class as `ProtectSpec.control_asks`' model-answered continuations (already captured on the base model today). Greedy, first-index tie rule (the seam's law), no RNG.
- **Degenerate fallback (load-bearing):** when the base model already answers correctly, `competing₀ == answer₀` and the differential direction is the zero vector — the target falls back to the absolute direction verbatim. The margin lever only bites on mis-ranked facts (exactly the Gitega class); on a well-known fact it is a graceful no-op. This is measured behavior, never an error: bench 1242's screening showed the France fact recalls 4/5 at baseline, so under the flag most France rows degenerate — pin BOTH shapes in tests.
- **Margin sizing:** at position j = 0, `D_margin = clamp(lp(competing₀) − lp(answer₀), 0, shortfall_cap) + floor` (the shortfall the write must close IS the margin), direction `unit((E[answer₀] − E[competing₀]) ⊙ g)`; positions j ≥ 1 keep today's absolute targets and magnitudes unchanged. Bounded by the same `shortfall_cap_nats` — the extreme-negative-logprob explosion guard carries.
- **Config + flag:** `FactWriteConfig::margin_target: bool` default `false` — flag-off is byte-identical Recipe-1/2 by construction (the code path is an `if`). Authored override rides the `--facts` doc (optional `competing_answer` field, first token read) for probe-grade control; absent field = auto-capture.
- **Protection unchanged:** ask-position keys, look-alike contrast rows, control rows, earlier-fact rows — all weights and shapes exactly today. The differential direction is applied at the fact's own answer-position keys only; G1b look-alikes ("The capital of Italy is" → Rome) and controls stay the standing gates. The direction does push AWAY from the competing token's embedding — that is the point — and the look-alike gate is what proves it does not leak.

## Why this can move what recipe-2 could not (and what would falsify it)

- It attacks the measured failure surface directly: the first-token argmax battle (depth-zero) instead of absolute likelihood (moved, generation didn't).
- It is the wsp lineage's own missing half: wsp's 2/5 datum came from a richer recipe; the differential-of-embeddings target is the standard closed-form fix for "recall the answer OVER what the model currently says" — and it stays modelless (forward passes + linear solve + thin SVD; no gradient, no RNG).
- **Pre-registered falsifier:** if the margin target still produces depth-zero on the anchor (the solve cannot flip the first token even when aimed at the battle), the objective-shape axis is exhausted after capacity/coverage/information — the recipe space records STOP at T2 below. If it flips the first token but recall still scores 0/5 Δ 0 (post-answer collapse, the T3 attractor in a new costume), the repetition gate catches it and the same STOP fires. Either negative closes issue 1189's Cell-A recipe space measured-negative, with the machinery verdict standing green (Bench 1219).

## Tasks

- [ ] **T0 — the code half (katgpt-core + harness, flag-off byte-identical).**
  - `FactWriteConfig::margin_target` (default false) + `validate()` update; `FactSpec::competing_first_token: Option<Vec<u32>>` (per training ask; None = auto-capture).
  - `construct()` top: the base-state auto-capture pre-pass (per fact × ask, `free_generate(ask, 1, None)`; refuse loud on an empty generation — cannot happen under the seam's contract, refuse anyway).
  - `solve_fact()`: position-0 differential target + margin sizing under the flag; the degenerate fallback (competing == answer → absolute target verbatim); positions ≥ 1 untouched. Envelope `FactRow` + harness manifest echo gain the new fields ADDITIVELY (sealed pair bytes unchanged under flag-off; the flag changes pairs only under flag-on).
  - Harness `--margin-target` flag + the facts-doc field + demote identities: (a) flag-off byte-identity on the default doc AND the Gitega doc (the G2 class, twice-run digest equality), (b) the degenerate-fallback pin (competing == answer ⇒ pairs byte-identical to flag-off on that doc), (c) auto-capture determinism (two constructs, same digest), (d) the margin-target unit test (a synthetic two-token seam where E[a] ≠ E[c]: the position-0 target must be the differential, position-1 the absolute).
  - Feature posture: everything under the existing `closed_form_write` gate (katgpt-core opt-in; the harness example is already feature-gated). No serving path, no new feature key.
- [ ] **T1 — the deciding cell (ONE cell, the recipe has ONE lever).** The registered Gitega anchor VERBATIM (`examples/facts/i1189_gitega_anchor.json`, bench 1242's train/contrast/held-out set — the no-re-tune law holds at the anchor level, carried from plan 635's verdict): window {10,11,12,13} (Recipe-1's reference window — NOT re-swept; T2 proved window-shopping sterile for absolute targets and the margin lever changes the objective, not the geometry), rank 16, rounds 6, `--margin-target`. TWICE (G2 byte-identity + manifest equality), exclusive 4090-CPU or bmo window (~3.5 h × 2 at the plan-635 measured wall), `CARGO_TARGET_DIR` isolation, box-state provenance beside every figure (bench_preflight discipline; CPU-bound construct, GPU idle). Record: `.benchmarks/` cell record (number allocated at run time per the highwater law).
- [ ] **T2 — adjudication under the amended gates (all four carried from plan 635 T3's amendment — baseline-delta recall, held-out set disjointness, first-k + distinct-ratio hit rule, the repetition gate beside G1b).**
  - **GO bar:** baseline-delta recall ≥ 2/5 on the registered held-out set AND G1b look-alikes green AND controls byte-identical AND repetition gate green AND depth > 0 on ≥ 1 training wording (the liveness floor replacing fit — fit is NOT comparable across objectives and is recorded as diagnostic, never gating). GO → this plan records GO; Cell B scoping (27B layer-window translation, the plan-635 T5 shape) moves to a FOLLOW-UP plan + the owner verdict lane — never this plan's commit.
  - **NO-GO bar:** anything else → the T4-NO-GO law fires at the RECIPE-SPACE level: issue 1189 records STOP for the closed-form family on Cell A (capacity T1-refuted, coverage T2-refuted, information T3-refuted, objective-shape T2-here-refuted — the axis list is exhaustive against the solve's degrees of freedom), Cell B never opens, the issue closes measured-negative with the machinery verdict standing green. No third recipe.
- [ ] **T3 — records + close-out.** Issue 1189 Status updated with the verdict + bench ref; Research 659 §7 lineage line; plan marked COMPLETE; claim lines reshaped (`**Lease closed:**`) per the discipline. Commit + push.

## Discipline carried unchanged (from plan 635)

- Cells run EXCLUSIVE (one job at a time on the named box; never contend a live session — `agent-mail rig status` before queueing, `--est` always).
- Twice-run G2 on the deciding cell; determinism ×4 was T2's standard, byte-identity is the floor here.
- Anchor discipline: the bench-1242 screening set is FROZEN — probe re-tuning after registration is the experiment's death (the T5 no-re-tune law at the anchor level). The reserve anchor (Dodoma) stays reserved for anchor-as-instrument failure ONLY, never a negative result.
- `--phase-a-only` first on any new posture: the baseline tables must reproduce bench 1242's before any armed arm is believed.
- Numbering: `.highwater` +1 + folder re-scan at write time; the bump rides the allocation commit.
- No serving-path changes anywhere in this plan; the base artifact never moves (the overlay law); promotion talk stays OWNER-GATED exactly as T6 of 1189 records it.

## Non-goals

- No rank/layer re-sweep (T1/T2 of plan 635: measured refuted; re-running them under a new objective is dead compute until the objective itself wins a cell).
- No paraphrase-set regeneration (the T3 attractor finding + in-sample critique stand; the registered set is the instrument).
- No Cell B work of any kind (gated on T2 GO + its own plan + the owner lane).
- No riir-train hand-off: the write stays modelless/closed-form end to end — if T2 NO-GO fires, the honest record is that THIS family is exhausted on Cell A, not that training is the next step (that call is the owner's, on the record as it stands).

## Provenance

- Failure anatomy: bench 1219 (T4-NO-GO, machinery green), bench 1244 (T3 attractor), bench 1248 (T2 coverage refutation, determinism ×4), plan 635 (the complete recipe-2 record).
- Code read 2026-11-23 at the pinned tree: `katgpt-rs/crates/katgpt-core/src/closed_form_weight_write.rs` (`FactWriteConfig` defaults, `solve_fact` target computation ~L740–775, `gen_fit`, `measure_gain`, `update_weights`, `construct` control-ask pre-pass precedent), `riir-infer/examples/closed_form_write_cell.rs` (flag surface, `--phase-a-only`, facts-doc schema, held-out set machinery), `examples/facts/i1189_gitega_anchor.json` (the registered anchor).
- Owner direction 2026-11-09 ("ternary too low bit → try qwen3.8 instead") stands as the lane's mandate; recipe-3 is executed under it, no new owner input required to run the plan as written.
