# Issue 1237: Decision-core census lane over laya encoders (ACE application)

**Status:** MEASURED 2026-11-19 — [Bench 1247](../.benchmarks/1247_i1237_decision_core_census.md): R1 relevance is rank-1 (≥99.3% energy) on all three checkpoints; typed LOW_DIM_CORE CLAIMED at r=1 (keep 0.999 vs PCA 0.197); english NOT CLAIMED (core ties PCA — the decision direction IS the top-variance direction); multilingual UNINFORMATIVE (99.5% one class on this suite); english↔typed AGREEMENT CLAIMED (cos² 0.951 = 974× random at r=1). Code landing GATED on this box's katgpt-rs working tree reaching `relevance_core` (see T-land).
**Lease closed:** katgpt-rs-93@amo, 1795103000, was expires 2026-11-21T18:00:00Z — T1 + T-ml + T-R0 done (Bench 1247); T-land left UNCLAIMED for whichever lane sees amo's katgpt-rs unblocked first (git apply the patch, clippy, commit)

**The idea.** The ACE extraction operator (balanced SVD of activation covariance and per-sample tail-Jacobian outer products — closed-form, label-free, f64) applied OFFLINE to the laya decision lane's encoders: per suite, extract the decision core at the encoder's readout layer, causally validate it through the decision head (keep/remove/flip — the keep/remove protocol already ships as katgpt-rs `subspace_intervention`), and measure cross-checkpoint core agreement (the paper's headline: necessary+sufficient cores recur across independently trained models, and a linear operator transfers across model families).

**Shape (the `sink_audit` precedent):** opt-in feature + a census bin, OFFLINE once-per-checkpoint, never a serving-path cost. CPU-viable on small checkpoints (finite-difference Jacobians at scalar-head scale; the paper's Jacobian count converges by 64–128 samples); the large-checkpoint arm gates on a GPU window. Producer dependency: the primitive itself is katgpt-rs Plan 640 (`relevance_core`) — this issue is its consumer; do not fork the math here.

**What it would measure first (pre-registered when claimed):** (1) decision-core rank + necessity/sufficiency per suite (do laya decisions ride a low-dim core?); (2) cross-checkpoint basis agreement (`basis_similarity`/procrustes) within one encoder family across sizes; (3) the transport read: fit the cross-core operator on one pair of checkpoints, score on another (the paper's 60%-vs-77%-own transfer shape).

**Downstream (named, not wired):** if a core is sufficient at rank r ≪ D, the lossy-surface law governs any dist-lane activation-projection arm (the f16-B pattern: retention-gated, per-family, demote-only) — an idea-class follow-up that starts only after (1) measures sufficiency. The rethink decision family is the eventual product consumer; nothing here ships a serving posture or a rethink-class surface — this stays public-substrate analysis.

**Provenance:** paper pinned in Research 683 (`joshseth/cores @ 527948bb`, MIT); the ACE math summarized there §2. Box state: quote `bench_preflight.sh` PROVENANCE on any quoted latency; accuracy tables need no box state.


## Pre-registration (katgpt-rs-93@amo, 2026-11-19, BEFORE any number)

- **Readout R1:** the head's scorer-input marker rows (`EncodedQuestion.marker_rows`, after type-emb + both head layers). `f` = the per-marker scorer logit (LN → Linear+GELU → Linear), Jacobian ANALYTIC in f64 from `scorer_tensors()` (exact, no finite differences). R0 (encoder output through the head layers, finite differences) is a named follow-up, not this slice.
- **Suite:** riir-reflex `tests/fixtures/tetris_oracle_laya_en_v4.jsonl` (read as data by path, no dep), the first 200 records, every option, as the oracle's `noul` forward (`<state_sentence>\n<option sentence>`). The decision is `sign(l_true − l_false)`. **Fit on records 0..100, evaluate the causal reads on 100..200** (held out). Cov sample ≥ 4·d rows (Bench 988's rule, disclosed per run).
- **Checkpoints:** english + typed (same family, ModernBERT-large, d = 1024), plus multilingual (d = 768) solo for (1).
- **Ranks:** `RankRule::Energy(0.90)` plus a fixed sweep r ∈ {1, 2, 4, 8, 16, 32, 64}.
- **Causal (mean-ablation):** keep = μ + QQᵀ(h−μ), remove = h − QQᵀ(h−μ), applied to both marker rows. Controls at the same r: the PCA top-r of Cov and a seeded random orthonormal r.
- **Verdicts:**
  - SUFFICIENT at r ⇔ keep-retention ≥ 0.95 AND it beats both controls' keep.
  - NECESSARY at r ⇔ remove-flip ≥ 0.25 AND ≥ 2× random's remove-flip.
  - A low-dim core is claimed ⇔ SUFFICIENT at some r ≤ 32.
- **(2) english vs typed:** mean cos² of the principal angles between the two cores at the english E0.90 rank (capped at 32), against the random baseline r/d. Agreement is claimed ⇔ ≥ 5× baseline. Caveat disclosed: separately fine-tuned heads share coordinates only through the common ModernBERT-large init.
- **(3) transport:** deferred (needs a second paired family).

## Tasks

- [x] **T1 — R1 census** (pre-registered above): Bench 1247, job `901d28cb`.
- [ ] **T-land — commit the census code (R1 + T-ml + R0 seams; patch refreshed 2026-11-20)** (`.benchmarks/1247_i1237_decision_core_census_logs/code_at_measurement.patch`, applies clean at `15954292`'s tree). It needs `katgpt-core` with `relevance_core` at `../katgpt-rs`: cargo validates `katgpt-core/relevance_core` at resolution, so landing early breaks every riir-infer build on a stale katgpt-rs checkout. amo's katgpt-rs is 23 commits behind (blocked by uncommitted `.agents/skills/{boundary-guard,doc-sync}/SKILL.md` WIP that conflicts with upstream). Unblock: that WIP is committed or reconciled, then `git apply` + clippy + commit.
- [x] **T-R0 — the encoder-output readout** (Bench 1247 R0 addendum, job `687a79c3`: a rank-1–2 core, necessary, english↔typed agreement 0.733 = 750×; core ties PCA at R0 for english/typed; multilingual rule-CLAIMED at r=4 by one eval forward, NOT robust; full-eval re-run `6b9950c9` (2302 forwards): CLAIMED at r=2, sign-consistent +4..+11 forwards over PCA at r≥2, the big gap at r=1 +437) (through the two head layers, finite differences): the deeper read R1 cannot give.
  **Pre-registration (2026-11-20, before any R0 number):**
  - Readout: the encoder-output marker rows (new seams `encoder_hidden` + `head_rows_from_hidden`, feature-gated). `f` = the scorer logit after both head layers. Seam fidelity ≤ 1e-3 against the lane's logits (the smoke run measured 7.6e-7).
  - Relevance is estimated by central finite differences along the top-k = 64 activity (PCA) directions of the fit marker rows. The step is 1e-2 × √λ_j, on 128 fit forwards spread over records 0–99 (256 marker rows). ACE runs in that k-subspace (Cov = diag λ) and maps back. Disclosed approximation: directions outside the top-64 span are never probed; the captured trace share is reported.
  - Causal reads at R0 on 256 eval forwards spread over records 100–199. Ranks {1, 2, 4, 8, 16} ∪ E0.90; same arms (core / PCA / random) and verdict rules as R1.
  - Runs: english + typed on the tetris v4 suite, multilingual on the balanced es/th/ja suite. Then english ↔ typed R0 core agreement at the english R0 E0.90 rank (≥ 5× random = AGREEMENT).
  - Side finding from the smoke run, filed as Issue 1239: `EncodedQuestion.hidden` is the post-head residual (max |Δ| 1281.6 vs encoder-only).
- [x] **T-ml — a balanced multilingual suite** (Bench 1247 addendum: decidable now; relevance is rank-2, necessary at every r, rank-2 keep 0.999 ties PCA → NOT CLAIMED; r=1 core beats both controls 0.795 vs 0.374/0.652) so multilingual sufficiency becomes decidable.
  **Pre-registration (2026-11-20, before the census number):**
  - Suite: the tetris v4 fixture rendered into es / th / ja from its closed template vocabulary (`gen_suite.py`, phrase tables, no LLM translation), records interleaved round-robin across the three languages.
  - Question choice: a pilot on records 200–239 (disjoint from the census's 0–199), candidates in order Q1 = the translated "Does the stack look clean?", Q2 = the translated "Does the piece leave holes under it?", gate true-rate ∈ [0.2, 0.8]. Pilot result: Q1 0.323, Q2 0.671 → **Q1 chosen** (first to pass; the same question as Bench 1247's English suite).
  - Census: multilingual checkpoint, first 200 records, with Bench 1247's protocol and verdict rules unchanged.
- [-] **(3) transport** — deferred (needs a second paired family).
