# Research 35: DNeo — MoFA structure-preserving target-feature encoding, rejection memory, and the layer-selection protocol (the specdec family's first ARCHITECTURE-level recipe record)

> **Source:** paper [arXiv:2611.21925](https://arxiv.org/abs/2611.21925) "DNeo: Your Parallel Draft Model Needs Better Target Feature Encoding" (Sandler, Bergsma, Elhoushi, Zhang, Gray, Hestness, Dey, Fioretto — UVA + Cerebras Systems, 2026-11-16; PDF read in full this session via r.jina.ai, 23 pp). Weights + inference code announced released (paper front page); **license NOT yet verified** — T0 item in the recipe issue before any code/weights reference beyond the paper text.
> **Date:** 2026-11-20
> **Status:** RECORD — Gain (lane intel + the family's first architecture-level training-recipe record + a modelless extraction inventory; **no lane opens** — the specdec beat lane stays PAUSED per the 2026-11-18 delegated verdict d9c2bb28; the recipe files as **riir-train Issue 703, PARKED with its own pre-registered stop line, do-not-claim** — the 698/702 posture).
> **Related Research:** riir-infer R031 (Sigma distill-spec — the acceptance-training lever + the LK datum), R032 (MMFlash — the qwen3.8 head INSTANCE + objective-reweighting half), R034 (2DSD — the DATA-recipe half, looped targets); katgpt-rs R661 (SimpleHC — the internal "no residual mixing" cousin, §3.4), katgpt-rs R316 (DSpark confidence-scheduled specdec — the loss + prefix-scheduler lineage, plan 339); katgpt-rs R012/R015 (PPoT rejection memory — the NAME-COLLISION cousin, signal-diffed §3.3)
> **Related Plans/issues:** riir-infer Issues [1179](../.issues/1179_qwen38_specdec_verify_lane.md) + [1183](../.issues/1183_dspine_b16_draft_model_support.md) (PAUSED, delegated verdict d9c2bb28 2026-11-18; re-open triggers: (1) trained head ≥3.2 held-out mean_j, (2) serving-side consumer, (3) a NEW draft-head objective/architecture that pre-registers its own stop line; Bonsai-target ranks above a qwen3.8 restart), riir-train plan 492 (STOPPED at its pre-registered stop branch 2026-11-14: trained DSpine mean_j 2.740 vs the ≥3.2 gate), riir-train Issues 698 (MMFlash objective recipe) + 702 (2DSD data recipe) — the two parked recipe precedents, **riir-train Issue 703 (filed by this note — the architecture recipe)**
> **Classification:** Public (riir-infer)

---

## TL;DR

DNeo is the parallel-drafter architecture SoTA: 1.3× faster generation than DFlash-2 at equal drafter footprint, TPF (τ) 3.41–5.37 at T=1 on Qwen3-4B (b=16), beating EAGLE-3's τ at T=1. Its gain decomposes into five mechanisms, each independently ablated (their Table 8, additive ladder DFlash 2.072 → DNeo 3.062, **+0.989 τ ≈ one full extra token per forward**): **MoFA** (a softmax-weighted per-layer mixture replacing the dense feature encoder — no "residual mixing" by construction; the single largest component, +0.651 τ), the **DNeo layer protocol** (a fixed closed-form schedule: layer 1 + second half; uniform selection measurably DEGRADES as c grows), **rejection memory** (cross-attention to the verifier's already-computed hidden states at REJECTED draft positions — free features; +0.176–0.189), the **fused causal head** (DSpark + DFlash-2 heads fused in shared rank-r latent space, full-vocabulary; +0.217 over DSpark), and **prefix-only attention** (drop masked-block self-attention; latency-only, +0.004 τ).

**For this stack:** no consumable artifact for our targets exists (Qwen3-4B drafter only; license unverified), and the mechanisms all require RETRAINING a drafter — so DNeo enters exactly as re-open **trigger (3)** of the paused beat lane ("a new draft-head objective/architecture proposal that pre-registers its own stop line"): the proposal is riir-train Issue 703. The modelless surface is unusually large and is recorded here as an inventory (§4), not implemented (no live consumer; ungoat discipline). Two corroborations land for free: our vendor checkpoints pin exactly the uniform-selection taps DNeo refutes (§3.2), and our own SimpleHC note (katgpt-rs R661) already states the residual-mixing law internally (§3.4).

---

## 1. What DNeo is (the five mechanisms, with the paper's numbers)

Background: parallel drafters (DFlash/DFlash-2/DSpark family) condition the drafter on target hidden states via **feature encoding** — c target-layer hidden states concatenated and DENSELY down-projected (matrix `Fc ∈ R^{cd×d}`, RMSNorm). DNeo finds and fixes three flaws of that design.

1. **Residual mixing (the discovered effect) — §5.4.1.** Each encoded scalar `f_j = Σ_i w_{j,i}·h^{(l_i)}` receives contributions from ALL coordinates of every layer state — the dense encoder entangles coordinates across layers that all write into ONE residual stream. **Ablation (their Table 5, 50K rollouts): dense `Fc` 1.938 → restricted-`Fc` (c diagonal matrices, off-diagonal blocks masked) 2.361 (+0.423) → DNeo+MoFA 2.526 vs DNeo+restricted-`Fc` 2.517 (within 0.3%).** Coordinate-preserving encoding ≈ the ENTIRE win; `Fc`'s deficit is not capacity (it contains the restricted form as a special case) — it must learn a coordinate structure that the restricted/MoFA forms build in.
2. **MoFA (Mixture-of-Features Attention) — §4.1.** `f_i = Σ_k u_k·h^{(l_k)}_i` with `u = softmax(w)`, c learned scalars (~0.05% of the dense encoder's parameters). Assumes `d(drafter) == d(target)` (coordinates pass through unchanged; the general case re-aims the drafter's K/V input dims). The gate doubles as a readable per-layer importance profile. Largest component: **+0.651 τ** (subtractive ablation, Table 4).
3. **DNeo layer protocol — §4.5 / A.2.** Trained MoFA gates allocate mass to LATER layers (20→35 of 36; layer 1 ≈ identity). Shipped protocol is a **fixed closed-form schedule needing no training**: keep layer 1, draw the remaining c−1 indices uniformly from the second half (`i ∈ (1, ⌊L/2⌋+⌊2L/2c⌋, ⌊L/2⌋+⌊3L/2c⌋, …, L)`). **Uniform (full-depth) selection DEGRADES with c (2.467 → 2.447 for c 5→15) while the DNeo protocol improves (2.481 → 2.516); MoFA over ALL 36 layers (2.478) loses to the protocol at c=5 (2.505)** — uninformative layers dilute the gate.
4. **Rejection memory — §4.3.** The target verifies the whole draft block in ONE parallel forward, so its hidden states at the REJECTED postfix are already computed — free. DNeo encodes the rejected postfix with the SAME MoFA encoder and appends the features as auxiliary cross-attention keys/values for the next round's masked block (input `X_{t+1} = [context f, x_bonus∘[m], rejection f]`; rejection features supply K/V, never predictions). **+0.176–0.189 τ over memory-less (Tables 7/8). Cross-attention beats fusion: DFlow-style MLP fusion +0.030 (no-MLP +0.009 ≈ nothing); fusion+memory is WORSE than memory alone (−0.034)** — broadcast fusion perturbs every masked input regardless of relevance ("fusion-as-bottleneck"). Cost: a serial dependency between rounds → training MUST be on-policy (rollout distillation; the paper uses Draft-OPD-style on-policy distillation).
5. **Fused DNeo causal head — §4.2 + prefix-only attention — §4.4.** Parallel drafters suffer suffix decay; DSpark's low-rank head (full-vocab, low expressivity) and DFlash-2's bilinear score head (context-aware, top-k=16 only) are fused in the shared rank-r latent space with per-position learned scalars α_j/β_j, up-projected full-vocab: **+0.217 over DSpark head (Table 6)**. Prefix-only attention drops masked-block bidirectional self-attention (2-tap causal convs + the head carry local deps): +0.004 τ but removes the masked-block K/V projections and shrinks block attention O(b²)→O(b) — the drafter gets FASTER (their A.6: total speculative latency below DFlash-2 and DSpark).

**Results (their Tables 2/3):** Qwen3-4B, b=16, 1×H100/SGLang. T=1: speedup 2.25×–3.55× vs AR, τ 3.41–5.37 (beats DFlash-2 by up to +32–40% τ, EAGLE-3 by up to +21% τ); throughput parity with the AR baseline at B=64 (10.32K vs 10.34K tok/s) where every other drafter falls 10–18% below. Training: DSpark loss `Σ_j exp(−(j−1)/b)·[0.9·L1 + 0.1·CE]`, on-policy distillation (4 rollouts/prompt, Llama-Nemotron post-training data), lr 1e-5 linear-warm-then-decay, 100K (ablation) / 1.6M (full) rollouts.

---

## 2. Why it matters here (the family map + the lane state)

The DFlash-family record in this repo now covers five complementary halves, of which DNeo is the **first ARCHITECTURE-level one**:

| record | half | consumable for our targets? |
|---|---|---|
| R031 (Sigma distill-spec) | acceptance-TRAINING lever (LK datum: acceptance-trained variant +14–15% mean accepted length) | checkpoint = continuous-dLM 3B, not adopted |
| R032 (MMFlash) | head INSTANCE (a real `FSCCS/Qwen3.8-27B-MMFlash` DFlash-format checkpoint) + objective-reweighting recipe | YES on paper — unconsumed, parked with 698 |
| R034 (2DSD) | DATA recipe (drafter data curriculum), looped-model-specific | no (Ouro/Nanbeige/Dory targets) |
| **this note (DNeo)** | **ARCHITECTURE recipe: encoder + layer selection + rejection memory + head fusion + attention shape** | no (Qwen3-4B only; license unverified) — enters as the trigger-(3) proposal, riir-train Issue 703 |

Lane state (all live-state-verified this session): 1179 T1 CLOSED (uzu head mean_j 1.27 < the 2.5 continuation gate, Bench 954 — Path A dead); plan 492 STOPPED at its pre-registered stop branch (trained DSpine 2.740 vs the ≥3.2 beat-arithmetic gate, Bench 709 — "the objective works, the acceptance ceiling doesn't"); the lane PAUSED 2026-11-18 (d9c2bb28). The datapoints on OUR Q4_K_M target: trained DSpine **2.740** (Bench 709 — the trained base of record, the STOP anchor), DSpine zero-shot 2.565 (Bench 965), uzu 1.27 (Bench 954), and a "2.83 (z-lab class)" measured-of-record figure cited in 1183 + plan 628 **whose producing bench is not named in either doc — context only, never a gate anchor** — gate ≥3.2. DNeo's ablation ladder says exactly why the ceiling held: every head we measured or trained carries the DENSE encoder (+0.651 τ left on the table on their fixture), uniform taps, no rejection memory (+0.18), and a single-head correction (+0.22). The architecture, not the objective, is the next lever class — which is what trigger (3) names.

**Positioning vs DSpine (do NOT cross-fixture-compare):** DSpine conditions over DEPTH (gated adjacent injection, ReplaySSM for the hybrid target, transition cache); DNeo fixes the ENCODING of depth-selected target features + adds cross-ROUND memory. The mechanisms are orthogonal and composable — a future beat-lane head could carry DSpine-style depth injection AND DNeo-style encoding/memory. DNeo's paper target is dense (Qwen3-4B); for the hybrid targets this lane actually cares about (Bonsai/qwen3.8-hybrid), DSpine's ReplaySSM remains the draft-side hybrid mechanism, and DNeo's rejection memory has a Bonsai-specific free-work story (§3.5).

---

## 3. Internal coverage + signal-diffs (read before claiming anything novel)

### 3.1 The closest code cousin: `riir-ai/crates/riir-gpu/src/dflash_config.rs` — `softmax_normalize_fusion` + `DFlareFusionConfig`

Ships today (verified this session): a softmax-normalization utility over a fusion-weight vector, exported behind `dflare_training` alongside `DFlareFusionConfig` and `build_target_layer_ids`. This is the MoFA gate's SHAPE — softmax-normalized per-layer fusion weights — as a training-plane INIT utility for the DFlare lane (DFlare = per-layer features, the paper's Table-1 row that already has structure-preserving encoding). **Signal-diff (why DNeo is still new here):** (a) DFlare fuses per-layer features PER DRAFT LAYER; MoFA fuses ONE shared feature reused by all drafter layers; (b) our utility is an init helper — no residual-mixing diagnosis, no restricted-`Fc` isolation experiment, no gate-as-importance readout; (c) the DNeo layer protocol and rejection memory have no analog anywhere in the tree. The honest filing: when a head-training lane re-opens, the MoFA gate CONSUMES `softmax_normalize_fusion` (the init is already correct-by-construction for c scalars) — never a parallel utility.

### 3.2 Our vendor tap-sets are exactly the uniform-selection class DNeo refutes

Verified this session: `QWEN38_DFLASH2_TAP_LAYERS = [5, 19, 33, 47, 61]` (0-indexed outputs; = the GGUF's `dflash.target_layers [6,20,34,48,62]`, the z-lab checkpoint's own pins) and `QWEN38_DSPINE_TAP_LAYERS = [1, 16, 31, 46, 61]` (the DSpine checkpoint's `target_layer_ids`) in `riir-infer-gpu/src/qwen38_dense_cudarc.rs`. Both are uniform full-depth selection with offsets — DNeo's measurements say this shape loses τ that grows with c, and the DNeo protocol (layer 1 + second half) is the closed-form replacement. **Consequence (load-bearing for 703): taps are checkpoint-pinned — changing them on a frozen vendor artifact is a train/inference mismatch. The protocol is a TRAINING-RECIPE knob: it binds only when WE train the drafter (which is exactly what 703 proposes).** For L=64, c=5 the protocol form gives ≈ `[1, 44, 51, 57, 64]` (1-indexed inputs ≈ our 0-indexed-output convention shifted; exact indices derived at recipe time from the pinned formula, never hand-typed).

### 3.3 "Rejection memory" NAME COLLISION — katgpt-rs Plan 027 / PPoT

`katgpt-rs` ships a "rejection memory" (Plan 027 adaptive PPoT rescue + the `ppot/knowledge` module: a ring buffer of "don't" insights biasing resampling strategy across a generation session). **Signal-diff:** PPoT's memory consumes accept/reject EVENTS at the token-RULE level to re-weight resampling STRATEGIES (a modelless bandit over sampling policies); DNeo's consumes TARGET HIDDEN STATES at rejected POSITIONS as cross-attention conditioning for the next draft round (a trained-drafter architecture surface). Different layer (resampling policy vs draft conditioning), different signal (events vs hidden states), different consumer. No consolidation — the shared word is coincidental; both keep their names (PPoT's is the older in-tree usage).

### 3.4 The residual-mixing law is ALREADY stated internally — katgpt-rs R661 (SimpleHC)

R661 distills SimpleHC (Hyper-Connections family): "learned read/write gates with identity transitions (**no residual mixing**) + shared memory access", with the component-study finding that Sinkhorn-constrained dense mixing kills the mixing benefit. That is the same design law DNeo discovers for feature encoding — **don't densely mix states that share a residual stream; gate them coordinate-preservingly** — arrived at independently, on a different surface (a model's own sublayer connections vs a drafter's conditioning encoder). The two publications + one internal note make this a three-way convergent law; this note records the lineage and cites R661 as the internal anchor. Fusion implication for any future architecture work: the law's scope is "any mixture over residual-stream states" — encoder design (DNeo), residual connections (SimpleHC/mHC), and by extension any learned blend we build over layer-like states.

### 3.5 The Bonsai-specific free-work story: rejection memory × GDN state snapshots

DNeo's rejection features are free because verification already computed the target's states at rejected positions. For a GDN-hybrid target (Bonsai), the recurrent-side analog ALREADY ships as a corpus rule: `gdn-state-snapshot-per-position-verify-early-exit` (riir-refine kernel_opt, entries_gpu_152 G — per-position recurrent-state snapshots as the speculation early-exit substrate). The snapshots ARE the per-position state record a rejection-memory drafter would read: on a Bonsai target, rejection memory's marginal cost is even lower than on a dense target. Combined with DSpine's ReplaySSM (the draft-side hybrid mechanism, riir-infer 1183), this is the concrete shape of a **Bonsai-first specdec lane** — the ordering d9c2bb28 already prefers over a qwen3.8 restart. RECORDED AS LANE INTEL, untested, no plan.

### 3.6 Measurement honesty: the TPF ceiling arithmetic joins an existing corpus family

The acceptance arithmetic `E[τ] = 1 + Σ_j Π_{k≤j} p_k` (iid: `(1−p^{b+1})/(1−p)`) is an ANTI-FRAUD BOUND: a reported TPF above the ceiling implied by measured per-position acceptances is an instrumentation bug, never a win. This is the same family as the shipped rust_perf rules `specdec-composite-decomposed-into-per-step-time-and-accepted-length` + `degenerate-output-invalidates-throughput-claim` (B369). Recorded here as corroboration; no corpus edit (the rule class already exists — a new rule would need its own disqualifier, and the ceiling bound is a special case of "decompose the composite").

---

## 4. Modelless extraction inventory (recorded, NOT implemented — no live consumer; ungoat discipline)

The no-GD advocate pass extracted 16 items; the ones that survive coordinator review as genuinely modelless (derivable/measurable without training), each with its validation gate. **None has a live consumer today** — the beat lane is paused and no vendor DNeo artifact exists for our targets — so per the filing law these stay INVENTORY, not plans. They bind when 703 (or any conditioning-architecture lane) opens.

| # | extraction | ships without training? | gate |
|---|---|---|---|
| 1 | Eq-14 layer-selection schedule: fixed f(L, c), precompute per (L,c) | YES — closed form | matched-c TPF ≥ uniform with the gap widening in c; within ~0.5% of trained gates (the paper's own 0.4% precedent) |
| 2 | Offline per-layer importance probe: logit-lens alignment / CKA(h^(l), final hidden or output embeddings) over a calibration corpus → derived u_l (our `lens/` lane computes per-layer logit projections already) | YES | derived-u TPF within ~1% of trained-u; Spearman ρ ≥ 0.7 vs trained gates where they exist |
| 3 | Uniform-dilution ordering law: capacity added without importance-aligned allocation hurts | YES — falsifiable sign test | reproduce at c ∈ {5, 15}; any crossover refutes |
| 4 | Diagonal-mass census: decompose any trained dense encoder into diagonal + off-diagonal, measure ‖off-diag‖/‖F‖ | YES — weight reading (the weight-audit class) | small off-diagonal share corroborates restricted-sufficiency per encoder |
| 5 | TPF ceiling arithmetic + monotone p̄_j (suffix decay) | YES — arithmetic over measured p | measured τ within ~5% of the ceiling; a τ above the ceiling = instrumentation bug |
| 6 | Adaptive-b controller: keep drafting while Π p_k × marginal tokens > marginal verify cost | YES — closed-form control | adaptive-b ≥ best fixed-b over a logged sweep |
| 7 | exp(−(j−1)/b) as a VALUATION PRIOR (never as a loss weight — that role needs training) | YES as prior | modeled decay matches measured p̄_j; downstream ≥ uniform-prior control |
| 8 | Fusion-as-bottleneck law: attention-gated injection beats broadcast fusion; combined-worse-than-single is the law's signature | YES as design law | A/B at matched params: attention ≥ fusion |
| 9 | Rejection-memory free-resource LEDGER (b states/round, dtype, zero extra target forwards) — the consumer stays trained-architecture | ledger YES | census matches theory |
| 10 | Prefix-only cost model: removes O(b·d²) K/V MACs + O(b²)→O(b) attention per layer | YES | measured −Δlatency ≈ predicted; \|Δτ\| ≤ ~0.01 |
| 11 | d≠d shim: deterministic unitary (Hadamard/DCT) projection — the deterministically-constructed-overlay class, never fitted | YES | orthonormality to machine precision; BLAKE3-pin the matrix |
| 12 | Lossless verification math (rejection sampling ⇒ output distribution ≡ target) | YES — it IS the G1 | distributional identity + deterministic replay |

Honest NEEDS-TRAINING residue (never extracted modellessly): the drafter network itself, MoFA's gate VALUES (the probe of #2 approximates them), the fused causal head's parameters, the position-dependent α_j/β_j, rejection-memory cross-attention weights.

---

## 5. Verdict

**Pinned claim (§1.5 form):** offline layer-importance probing + the closed-form Eq-14 second-half schedule, recorded as modelless extractions for target-feature-conditioned drafters on the paused specdec lanes, distinguished from the paper's trained MoFA-gate derivation by requiring no drafter training, and from R031–R034 by carrying the family's first architecture-level record. §4 searches: headline ("DNeo Cerebras MoFA" — found only the paper itself + HF papers page, 4 days old), component ("logit lens layer selection draft model feature encoding" — EDD/ACL 2025 and LogitSpec are adjacent-but-different: EDD treats the LLM as encoder for draft input, LogitSpec retrieves by last-logit; neither derives draft-conditioning layer selection from an offline layer-importance probe), rejection landscape (DFlow 2609.06498 + DEdit 2609.38510 = the class DNeo improves on; both in-paper). **Layer SELECTION itself has published prior art and the claim is scoped to the derivation, never to selection per se: EAGLE/EAGLE-3 inject a FIXED layer set (low/mid/high grouping, sampled shallow→deep — the fixed-set convention every EAGLE-family drafter inherits), and the feature-level drafter family (HASS "Learning Harmonized Representations for Speculative Decoding"; GRIFFIN arXiv:2502.11018) picks/aligns target features by LEARNED or training-time mechanisms. No published prior art found for the OFFLINE-PROBE-DERIVED schedule specifically (measure a per-layer importance spectrum, then apply the closed-form second-half formula) — a narrow claim, and the honest one.**

**Per-track tiers:**

- **Track (a) modelless inference: Gain (record-class).** The extraction inventory (§4) is real and gated but consumer-less today; a katgpt-core primitive filing would be a no-consumer filing (banned). The note + the recipe issue carry it.
- **Track (b) self-adaptive runtime: Gain (record-class), folded.** The rejection-memory LEDGER is modelless; the consumer is trained-drafter architecture; the GDN-snapshot fusion (§3.5) is lane intel for the Bonsai-first ordering, untested.
- **Track (c) model-based: Gain — recipe filing.** DNeo is trigger (3)'s architecture class; the proposal with its own pre-registered stop line = **riir-train Issue 703** (PARKED, do-not-claim, same posture as 698/702; resumption waits on the owner verdict on the beat lane itself).

Not Super-GOAT: no new capability class ships (nothing is wired), and the mechanisms are published prior art by construction. Not GOAT: no runnable proof path until a lane re-opens (ungoat discipline — an issue, never a record-plan). **MOAT gate (riir-infer):** lane intel + recipe recorded on the public inference-substrate surface where the specdec lanes live — in scope, no code lands, zero riir-* dep questions.

**PASS-Redirects (synthesis):** Zheng et al. [ACL 2025 "Faster Speculative Decoding via Effective Draft Decoder (EDD)"] — adjacent feature-level drafting, no layer-importance probe; already covered by the §4 search line. He et al. [arXiv:2507.01449 "LogitSpec"] — last-logit retrieval drafting, different mechanism, no redirect owed beyond this line.

## 6. Filing

- **riir-train Issue 703** — the DNeo drafter-architecture recipe (MoFA encoder consuming `softmax_normalize_fusion`, the Eq-14 tap-set builder, rejection memory over the verify pass's own states, fused rank-r head, prefix-only shape; PROMOTE gate held-out mean_j ≥ 3.2 + G1; STOP line delta < +0.3 over **2.740** (Bench 709) at the capped budget; target decision — qwen3.8 vs Bonsai-first — is part of the owner verdict, per d9c2bb28's ordering).
- No code, no plan, no corpus edit this session (the recipe issues 698/702/703 compose at claim time: 698 objective, 702 data, 703 architecture).
