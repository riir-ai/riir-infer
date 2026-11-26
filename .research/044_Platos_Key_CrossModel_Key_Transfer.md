# Research 044: Plato's Key — Cross-Model Key Transfer by Closed-Form Rotation

> **Source:** "Plato's Key" — Hazy Research (Stanford) blog, 2026-11-25,
> <https://hazyresearch.stanford.edu/blog/2026-11-25-platos-key> (Vivien Cheng et al.;
> blog-first, no arXiv ID surfaced as of 2026-11-26). Base hypothesis: Huh, Cheung,
> Wang, Isola, "The Platonic Representation Hypothesis" (ICML 2024,
> arXiv:2405.14850). Adjacent: Jha et al., "Harnessing the Universal Geometry of
> Embeddings" (vec2vec, arXiv:2505.21840, May 2025).
> **Date:** 2026-11-26
> **Status:** Active — PoC cell filed at `riir-infer/.issues/1254_platos_key_cross_model_transfer_cell.md`
> **Related Research:** katgpt-rs Research 459 (canonical intent space — the `katgpt-canon`
> adapter substrate this consumes); this repo's 041/043 are unrelated lanes
> **Related Plans:** none yet (issue-first per the ungoat discipline; a plan follows only if the cell passes)
> **Classification:** Public (riir-infer is a sanctioned public substrate repo; the paper is public)

---

## TL;DR

A **rotation + one global scale, fit in closed form** (orthogonal Procrustes on
paired-recall keys, thin SVD), lets one frozen LLM use another's representations:
a donor's hidden state at the last subject token, rotated and scaled, swapped in
for the recipient's own state at the corresponding layer, flips the recipient's
factual-recall answer (median transfer accuracy 41% vs 55% self-swap, 20 model
pairs, 12 base LLMs, eight families, 1.5B–27B). Rotations chain (A→B→C retains
99% of A→C) and compress (256 principal components retain 74%) — consistent with
LLMs being rotated views of one shared lower-dimensional key space. A rotated
donor key recovers **18% of the recipient's dark memories** (facts it stores but
fails to recall), with no weight updates.

**Distilled for this stack (modelless, inference-time):** the whole protocol is
GD-free by construction — closed-form SVD fit offline per model pair, frozen as a
BLAKE3-sealed artifact, applied at serving as one matmul + one state replace at a
chosen (layer, position). The generic fit primitive ALREADY SHIPS upstream:
`katgpt_spectral::procrustes` (`orthogonal_procrustes`, mean-centering built in,
`ProcrustesConfig { center, force_det }`), GOAT-gated as `katgpt-canon`'s
`ProcrustesAdapter` (Bench 562: residual 0.0000%, round-trip 4.47e-8, BLAKE3
bit-identical, 16.17 µs @ d=256, 0 allocs hot path). What does NOT ship is the
protocol around it: paired-recall fact mining, subject-token key capture, the
recipient-side state-REPLACE seam, functional transfer eval, dark-memory probes.

---

## 1. Paper core findings (numbers of record, as published 2026-11-25)

| Finding | Number |
|---|---|
| Transfer accuracy (rotation map, 20 pairs) | median **41%** (range 34–46%) |
| Self-swap reference (recipient's own other-fact key) | median 55% → rotation retains median **73%** (60–93%) |
| Flexible maps (ridge w/o rescale, whitened rotation, MLP) beat the rotation | only **7%** of comparisons (Holm-corrected p < 0.05) |
| Unsupervised (no paired facts) Wasserstein Procrustes | median 87% of the supervised rotation |
| Layer selection by mKNN (the PRH geometric metric) instead of transfer-val | median transfer **4.5%** — ~1/10th; mKNN peaks in the first 5% of depth, transfer peaks at **22–38%** |
| Chained rotations (A→B→C, composed, B never run) | median **99%** of direct A→C (40/47 chains ≥ 95%, none < 92%) |
| Compressed keys (256 PCs per LLM) | median **74%** of full rotation; 16 PCs → 4%, 512 PCs → 87% |
| Dark memories among donor-recalls/recipient-misses | **56%** qualify (ICL / paraphrase / rank-1-of-≤50 checks) |
| Rotated donor key recovers dark memories | **18%** (267/1,463); best control ≤ 1.1% |

Mechanism frame: factual recall as key–value lookup — early-to-middle layers build
a key at the last subject token; later layers retrieve the value from the weights.
The paper swaps the **key**, never the value, never the weights. Scale matters:
keeping the recipient's usual spread beats least-squares-fitted scales.

## 2. Internal prior art — the controlling records (grep-verified this session)

The first-pass `rg` sweep returned zero and was WRONG — the authoritative workspace
grep surfaced three load-bearing internal records. The §1.5 verdict below is graded
against them.

**(a) The generic fit primitive ships — twice.** `katgpt-spectral/src/procrustes.rs`
ships `orthogonal_procrustes(a, b, n, d, out_rotation, scratch)` with
`ProcrustesConfig::center` (default true — the paper's mean-centering) and
det enforcement; `katgpt-canon` wraps it as `ProcrustesAdapter` plus
`SubspaceAdapter` (joint SVD, cross-arch shared subspace) and `MaskAdapter`
(katgpt-rs `.docs/05_adaptation/canonical_intent_space.md`; GOAT G1/G2/G4 PASS,
Bench 562, 2026-07-28).
**Signal-diff (§3.6):** canon fits its map from anchor pairs of **steering/intent
directions** for snapshot-swap and steering adapters; Plato's Key fits from
**paired-recall fact keys** and evaluates **functional recall transfer by state
replacement**. Different fit data (directions vs subject-token keys), different
eval (direction agreement vs argmax flip), different apply site (adapter output vs
mid-forward state replace). Coverage of the FIT is real; coverage of the PROTOCOL
is not. No new primitive needed — the PoC consumes or locally reimplements the fit
(issue_925 test-local one-sided Jacobi precedent).

**(b) riir-train Benches 424–427: the stack's own cross-arch NEGATIVE — the burden this note carries.**
(Ownership corrected after verdict round 1: the canon demotion benches live in
RIIR-TRAIN — `424_canon_p3_rust_discrimination`, `425_canon_p3b_gemma_layer_sweep`,
`426_canon_p3c_corpus_and_length`, `427_canon_p4_recipe_d_length_matched` —
referenced unqualified from katgpt-rs `.docs/05_adaptation/canonical_intent_space.md`;
katgpt-rs's OWN 424–427 are unrelated benches — dd_tree / gdn_tree / tilr /
manifold_erasure / mance_svd. Bench 562 (canon GOAT) IS katgpt-rs.) katgpt-canon's
cross-arch Super-GOAT headline is PERMANENTLY DEMOTED: four
hidden-state methods failed G6 ≥ 0.5 cross-arch agreement — per-model centroid +
Procrustes **−0.33**, difference-of-means +0.46 (token-count confound), intermediate-layer
probe +0.19, length-detrended −0.15, length-matched corpus +0.009 — "the failure is
cross-arch disagreement, structurally ~0, not length, not noise"; the modelless path
was declared exhausted, "reopens only on non-hidden-state construction" (Proposal 010 draft).
**Signal-diff:** 424–427 measured **steering-direction agreement** (cosine between
per-model derived directions, unsupervised alignment). Plato's Key measures
**functional recall transfer** (paired-recall supervised fit, layer selection by
validation transfer at 22–38% depth, argmax flip on held-out facts). Different
measurement — BUT the demotion is the falsifier context the PoC cell must carry.
⛔ Scale discipline (verdict round 1): the PRIMARY cell must run IN the paper's
validated range (1.5–27B) — a below-floor null cannot settle architecture vs
scale, so it cannot extend 424–427. Primary pair: Ternary-Bonsai-2.5-27B
(`qwen35` GDN hybrid, header-verified) × Gemma-2-2B (dense, restore-gated per
riir-train Issue 617) — cross-arch, both in range; a same-arch in-range pair
(Bonsai-2 ↔ Bonsai-2.5, both header-verified `qwen35` 64-block) isolates
training variance from architecture; the 0.8B × 1B pair is demoted to a cheap
smoke whose null is pre-registered INCONCLUSIVE-below-floor. A pass does NOT
auto-reopen canon's steering claim (different measurement, different task); an
in-range null DOES extend the negative to a third measurement class (steering
directions, probe layers, now functional keys).

**(c) The forward-intervention seam exists in kind.** `PrefillContext.census_column_scale:
Option<(layer, column, f)>` + the `batched_mlp` hook (`src/deltanet/forward.rs`,
feature `neuron_census`, Issue 1241) is a **multiplicative activation push**,
single model, census readout.
**Signal-diff:** the swap needs **REPLACEMENT of one position's full state** under a
**cross-model map** with a **recall consumer**. Sibling seam shape (set-once before
a read, cleared after), not coverage — a `state_replace`-shaped hook is new, small,
and naturally test-level first.

**(d) Adjacent lanes on the factual-memory axis.** `closed_form_write_cell`
(Issue 1189) writes facts into ONE model's weights in closed form (construct →
sealed envelope → demote/reload identity) — no cross-model map, no runtime swap.
`readout_screen` (Issue 1250) is a weights-only logit-lens readout — the paper's
dark-memory rank probe is forward-side, a different instrument. Specdec lanes
(1179/1183) are two-model runtime cooperation at the TOKEN level (draft/verify),
not the state level. Deletion-mask lane (1208) ships the artifact-law template this
idea borrows: base-BLAKE3 binding, RLE+zstd, optional Ed25519, router-frozen,
monotonic mount — the exact freeze shape a donor-key table should wear.

**(e) Matrix machinery all exists:** one-sided Jacobi SVD (test-local f64,
`katgpt-attn/tests/issue_925_procrustes_probe.rs`), symmetric Jacobi
(`katgpt-core` `SymmetricEigScratch`), PCA/compression via katgpt-core spectral
(`basis_split` whitening + rank truncation). Zero new linear algebra required.

*Verification note:* §2's internal records were grep-verified this session with the
workspace grep tool after a first-pass `rg` sweep returned false zeros; the 424–427
REPO ownership and the donor-file-on-disk claims were corrected in verdict round 1
(the reviewer checked disk; the coordinator re-ran `ls` on both `.benchmarks/` dirs
and `riir-train/data/` to confirm — MiniCPM5-1B-F16.gguf is ABSENT despite the
riir-train AGENTS.md LOCAL-KEEP row, which is stale vs disk). Verdict round 2 added
the ARCHITECTURE LAW: a model's architecture is read from `general.architecture`
in the GGUF header (probe `scripts/plan402_gguf_probe.py`), never from the filename —
the round-2 draft paired a DFLASH DRAFT HEAD (`Qwen3.8-27B-DFlash2`, 5-block,
non-causal, cannot recall) as if it were a full LM, and every in-range model on
disk is `qwen35`-family, so the in-range cross-arch pair requires the restore-gated
gemma-2-2b (coordinator re-ran the header probe on all three files to confirm).
Tool note (reworded per verdict round 3): `rg` was not ABSENT — it resolves
through a shell-function wrapper that a non-interactive shell can fail to
resolve; the first-pass zero-hit sweeps were tool-RESOLUTION artifacts. The
lesson stands: a zero-hit result from a tool that may not have run is not
evidence — the workspace grep tool is the verification instrument of record.

## 3. External prior art (searched 2026-11-26)

- **PRH** (arXiv:2405.14850, ICML 2024): geometric similarity (mKNN) across models
  — the paper itself shows mKNN layer selection yields ~4.5% transfer: geometric
  similarity ≠ functional interchangeability. Plato's Key is the functional test.
- **vec2vec** (arXiv:2505.21840): unsupervised translation between embedding
  spaces via TRAINED adapters — no functional recall swap, no closed form.
- **ROT** (Du et al., 2026): rotating hidden states toward contextual vectors —
  WITHIN-model hidden-state correction for recall; no cross-model map.
- **Cross-Architecture Semantic Geometry** (SSRN 2026): PCA + orthogonal
  Procrustes on residual activations — geometric pipeline, no functional swap.
- **Hazy's MLPs-as-key-value-memories line** ("Stuffing MLPs Full of Facts",
  2025-11-28 blog): the memory frame the paper operationalizes.
- Nothing found (2 searches + landscape sweep) that does the functional
  cross-model recall swap or the dark-memory unlock before 2026-11-25.

## 4. Distillation — modelless component inventory (Path 0, compact)

The paper is already GD-free, so every component is modelless-validable; the
table records what ships vs what the cell adds.

| Component | Modelless | Ships today | Delta for the cell |
|---|---|---|---|
| Rotation fit (Procrustes, centered) | closed-form SVD | YES — `katgpt_spectral::procrustes` (or test-local Jacobi, issue_925 precedent) | none |
| Global scale (spread match) | closed form | report norms exist | ~10-line addition (paper's scale law, NOT least-squares — the paper shows LS scales underperform) |
| Paired-recall fact mining | forward probes | CPU forwards ship | fact set (16 relations), direct-prompt recall filter |
| Donor key capture @ (layer, subject pos) | forward read | forwards ship | residual readout at position |
| Recipient state REPLACE @ (layer, pos) | forward hook | sibling of `census_column_scale` (multiplicative, scale≠replace) | test-level replace hook |
| Layer selection ℓA/ℓB | validation sweep offline | — | sweep 22–38% depth band + mKNN contrast arm |
| Transfer eval + controls (identity / random rotation / shuffled keys) | argmax walks | league walk machinery | fixture |
| Dark-memory detection (ICL / paraphrase / rank-1) | forward probes | rank surfaces near readout_screen | three checks, report-only arm |
| Chained rotations (compose R·R) | matmul | — | free |
| Key compression to k PCs | spectral | katgpt-core `basis_split`/Jacobi | truncation arm |

**Track separation (one verdict per track):** modelless track — everything above,
PRIMARY, filed as Issue 1254. Training track — "donor-rotated keys as distillation
targets" **audited discard**: the distill lanes' teachers are text-output models;
a pre-answer hidden state is not a supervision target without a consumer that
consumes representations, and the paper shows closed-form ≈ trained maps (MLP
rarely better), so there is no training gain to chase. Self-adaptive track — the
fit is offline-per-pair, not runtime-adaptive; no EMA/refinement loop is claimed.

## 5. Fusion — what paper × stack produces that neither has alone

1. **The adjudication cell (primary value).** riir-train's Benches 424–427
   measured cross-arch hidden-state agreement ≈ 0; the paper measures functional
   transfer succeeding. Running the paper's protocol on OUR in-range cross-arch
   pair (Bonsai-2.5-27B × Gemma-2-2B — the 4090/amo rig lane, never m3)
   adjudicates the tension: pass ⇒ paired-recall functional eval is the missing
   ingredient (and the canon demotion's scope note gains a measured boundary);
   in-range null ⇒ the negative extends to a third measurement class. Either
   result is a durable record — this is why the cell is pre-registered with BOTH
   outcomes.
2. **Frozen donor-key table (the serving primitive, post-cell).** Fit once per
   model pair offline; freeze {R, s, means, ℓA, ℓB, donor-key rows for a subject
   set} as a BLAKE3-sealed artifact in the deletion-mask RIMSK shape (base-BLAKE3
   binding, monotonic mount, no runtime minting); at serving, a **sigmoid-gated**
   low-confidence recall (never softmax) consults the table and swaps — the donor
   model never runs at serving time. Zero weight writes, one matmul, armed
   promote-only, unarmed boots byte-identical (env-switch law).
3. **Chained-rotation hub law.** One map per checkpoint into a shared space
   instead of pairwise tables (the paper's 99% chain retention) — matters when N
   serving checkpoints each need donor tables; composes with (2).

## 6. Game-context and consumer-context reframes (honest, thin)

- **Game:** the stack's NPC cognition is modelless latent belief kernels, not LLM
  forwards — the analog (two environment variants' belief geometries aligned by
  one rotation into a shared KG space) exists conceptually but no consumer asks
  for it; recorded, not filed.
- **Consumer (rulewright, the priority-1 two-sided check):** retrieval corpora all
  share ONE embedder (`ModellessEmbedder`), so rule-corpus spaces are already
  co-geometric — no rotation is needed anywhere on the rulewright path. The
  mechanism does not improve the self-evolve stack; it serves the inference-perf
  league (fusion-priority 3). Recorded honestly: the ladder orders effort, and
  this files at priority 3.

## 7. Verdict

**Gain** — mechanism externally proven (paper, fresh 2026-11-25) and actionable
for the stack; NOT Super-GOAT: the generic primitive already ships (canon, Bench
562), the paper is itself the mechanism's published prior art, and the product
selling point ("serving models rescue recall with another model's frozen keys")
has **no consumer lane in the serving stack today** — the decision family serves
classifications, not factual recall. Fusion idea (frozen key-table serving +
adjudication value) is novelty-TBD and is filed as the issue below — no
Super-GOAT wording is claimed, and tiers re-grade ONLY on a passing cell.

Not PASS: three concrete unblocks exist — the adjudication cell (424–427
tension), the replace-shaped sibling seam, and the artifact-law template the
serving primitive would wear.

**MOAT gate:** riir-infer — public inference-substrate novelty (cross-model state
transfer protocol + forward-state intervention seam), upstream-clean, no
game/chain/cognition semantics. Fits. katgpt-rs — no new primitive owed (canon
ships it); this note is NOT filed there. Boundary note: `katgpt-spectral` is not
yet in riir-infer's dep allowlist — the cell is TEST-LOCAL first (the
`issue_925_procrustes_probe.rs` precedent), and any production promotion adds the
allowlist row + BOUNDARY.md edit in the same commit as the dep.

**Falsifiers (pre-registered, full gates in Issue 1254):**
- PASS: swap transfer ≥ 15% absolute AND ≥ 3× the random-rotation control AND ≥ 2×
  the identity control, controls clean (no NaN, no all-flip pathology) — measured
  on the PRIMARY in-range pair (Ternary-Bonsai-2.5-27B × Llama-3.2-3B BASE,
  cross-dim fit in the shared top-256-PC space per the paper's compression
  protocol; recipient amended from gemma-2-2b to the llama BASE model by the
  delegated verdict below).
- FAIL/NEGATIVE: in-range transfer ≤ random-rotation control + noise ⇒ the
  cross-arch negative (riir-train Benches 424–427) extends to functional key
  transfer at validated scale; record NEGATIVE, demote, close. The below-floor
  0.8B × 1B arm can NEVER drive the negative (scale-confounded by construction).
- Scale-law check: the paper's spread-match scale vs a least-squares scale — the
  LS-underperforms claim leans on the blog alone; if LS ≈ spread on our pair,
  record the divergence (no demote).
- G3 byte-identity: hook-compiled-unarmed forward is bit-identical (FNV/argmax
  walk pins). G4: swap apply alloc-free at serving shape.
- mKNN contrast arm expected to underperform transfer-selected layers (~4.5%
  paper) — adjudicates geometric-vs-functional on our pair.

### PoC decision addendum (2026-11-26)

Delegated verdict (session `5b062d19`, round 1 AGREE, closed round 2): A1's
recipient of record is **Llama-3.2-3B BASE** (F16 GGUF, HF-fetched, sha256 +
source URL recorded, Llama 3.2 Community License — LOCAL USE ONLY, never a
shipped artifact), not the archived gemma-2-2b-it checkpoint. Reason: the
riir-infer owner law (2026-10-26) — "no new gemma-2-2b execution cells
anywhere" — forbids the gemma arm under its text (the offline-research-cell
reading is exactly what it was written to stop); arm B also removes the
IT-vs-BASE protocol deviation. Reviewer findings folded into the cell: the
llama path needs its own `PostLayerHook` fork (the Issue-395 seam shape, G3
tested); llama3 RoPE scaling (`rope_freqs.weight`) must be loaded and applied
(per-dim freq factors, fail-loud on shape mismatch — never a silent fallback);
the A1 run is gated on a recipient baseline recall floor before any transfer
number. A is rejected, not deferred — the gemma archive stays unrestored.
Full amendment: `../.issues/1254` (the runnable cell of record).

## References

- Plato's Key blog (2026-11-25) — fetched verbatim this session; numbers above are
  as-published-blog, NOT reproduced here (blog-first; arXiv/code TBD — re-verify
  against the paper when it lands).
- arXiv:2405.14850 (PRH), arXiv:2505.21840 (vec2vec).
- riir-train: `.benchmarks/424_canon_p3_rust_discrimination.md` … `427_canon_p4_recipe_d_length_matched.md`
  (the canon cross-arch demotion; 423_canon_p1 is the same family — note both repos
  hold a 423, a historical cross-repo collision number); `data/BASE_MODELS.md`
  (Bonsai lineage: 2.5 = agentic retrain of Qwen3.8-27B); `scripts/plan402_gguf_probe.py`
  (the header probe — architecture is read from the GGUF header, never a filename).
- katgpt-rs: `.docs/05_adaptation/canonical_intent_space.md` (references the
  riir-train benches unqualified), Bench 562 (`562_katgpt_canon_goat.md`),
  `.benchmarks/125_reexport_map.md`, `crates/katgpt-spectral/src/procrustes.rs`,
  `crates/katgpt-attn/tests/issue_925_procrustes_probe.rs`.
- riir-infer: `src/deltanet/forward.rs` (`census_column_scale`), `examples/closed_form_write_cell.rs`,
  Issues 1189 / 1208 / 1241 / 1250.
