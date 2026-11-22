# Research 039: Quantized Draft-KV Speculative Decoding — the K/V Bit-Split × Quantizer-Geometry Co-Design Principle, the K4V4 Draft-Cache Operating Point, and the Memory-Not-Speed Disclosure

> **Source:** arXiv:2611.20565 "Memory-Efficient Speculative Decoding with Quantized Draft KV Cache" — Saksham Gupta & Michael Li (CMU), v1 2026-11-16, accepted at the AXIOM Foundations of Efficient Deep Learning Workshop @ NeurIPS 2026. Code: `github.com/sgdescent/KV_Reduce` (MIT). Full text read this session (PDF via r.jina.ai, 13 pp).
> **Date:** 2026-11-22
> **Status:** RECORD — Gain (record-class: the co-design principle + the draft-cache memory operating point + the honest negative-result disclosures; **no lane opens, no plan filed** — the specdec lanes it feeds are PAUSED (Issues 1179/1183, delegated-owner verdict `d9c2bb28` 2026-11-18) and the Q8KV lane it informs is default-on at symmetric Q8, where the bit-split question cannot arise)
> **Related Research:** 031 (sigma distillspec — the specdec league row + the cross-fixture law this note rides), 026 (FOSCAR — the fold/layout axis coupling + the axis-speed ablation), 034 (2DSD looped specdec), 032/037 (the specdec family), katgpt-rs R487 (massive activations + sink-aware Q8KV — the outlier-absorption cousin), katgpt-rs R159 (KVarN), R020 (TurboQuant vs KIVI), R063/R065 (OCTOPUS/RotorQuant — the OTHER "bit split" family: direction/norm triplet splits, NOT the K/V split)
> **Related Issues:** 1179 (qwen3.8 specdec verify lane — PAUSED, file on disk confirms), 1183 (DSpine b16 draft lane — PAUSED, file on disk confirms), 1211 (Q8KV fold lane — DEFAULT-ON), 919/716 lineage (`kvq_ab.rs` A/B backends — the per-token + channel-exemption geometry), 1182 (drift guard — the proxy-law consumer)
> **Classification:** Public (riir-infer)

---

## TL;DR

Two-model speculative decoding stores two KV caches; this paper quantizes **only the draft cache** (target cache + verifier stay BF16, so the correction rule is mathematically untouched) and lands two durable contributions for this stack:

1. **The co-design principle (the headline finding):** the K/V bit-split conclusion is NOT intrinsic to attention — it is an artifact of the key quantizer's geometry. Under a naive per-token key quantizer, K8V4 beats K4V8 by **+10.87 pp** acceptance (macro; controlled single-variable test +28.70 pp, and K4V2−K2V4 +15.05 pp); switch keys to grouped per-channel (KIVI layout) and the same contrasts collapse to **−0.25 pp** / −0.33 pp / **−3.66 pp (SIGN REVERSED)**. 16K passkey retrieval: the K4V2-vs-K2V4 gap shrinks 81.25 pp → 16.15 pp from the key-quantizer change alone. **Bit width and quantizer layout must be co-designed; keys do not universally demand more bits — the layout absorbs their outliers.**
2. **The draft-cache operating point:** K4V4 (per-channel key groups of 32 tokens, per-token values, KIVI layout) across six target/draft pairs (Qwen2.5/Qwen3/Llama 3/OLMo 2/SmolLM2, 360M–8B): **draft cache −66.20%, combined target+draft KV −23.95%, acceptance 57.93% → 57.14% (−0.79 pp macro, −1.36% relative)**. Long context holds (16K/32K: ~31% combined saved, ≤0.94 pp acceptance drop). Packed storage is real: 896 MiB → 260.2 MiB per sequence at 32K.

**And the honest half this note carries forward as discipline:** the packed Triton prototype is **18–33% SLOWER** than BF16 speculative decoding (the paper claims memory, never speed); peak GPU memory is unchanged (the cache packs after a BF16 prefill); the batched verifier is not exact. For us: **memory claims are storage claims, not peak claims, not speed claims** — and any future packed-cache kernel work goes through the FOSCAR fold (never rebuild-then-attend).

Nothing here opens a lane: specdec is PAUSED (1179/1183), and our Q8KV lane runs symmetric Q8 where no bit split exists. The record arms the **when** (a specdec lane re-opens → the draft side boots K4V4-class as the memory posture candidate; an asymmetric K/V arm ever opens under the bandwidth law → the layout co-design doctrine applies first).

---

## 1. Paper core findings (verified against the fetched full text, v1)

1. **Draft-only quantization preserves the target correction rule by construction.** Target cache + verifier BF16; the draft proposes, the target accepts/rejects against its own exact cache. Sequential-verifier outputs matched target-only greedy on all checks (38,400 audit checks in App. F).
2. **RQ1 — the geometry interaction.** Per-token symmetric quantizer (their Eq 1: one scale per token × KV head): K8V4−K4V8 = **+10.87 pp** macro [+3.75, +19.68] across six pairs; per-pair up to **+31.18 pp** (Qwen2.5 3B/1.5B). Grouped per-channel keys (32-token groups, affine, 128-token BF16 residual window): macro **−0.25 pp** [−0.56, +0.04], every within-pair |gap| < 1 pp. The controlled Qwen intervention (models, prompts, value quantizer, budget, decode loop all fixed; ONLY the key quantizer changes): K8V4−K4V8 **+28.70 → −0.33 pp**; K4V2−K2V4 **+15.05 → −3.66 pp** (preferred split FLIPS). Verifier-free 16K passkey: per-token keys K4V2 166/192 vs K2V4 10/192 (81.25 pp); grouped keys 189/192 vs 158/192 (16.15 pp) — the interaction needs no verifier.
3. **RQ2 — the operating point.** K4V4: draft cache −66.20% macro (65.55–66.33% per pair; −70.56% with no residual window), combined −23.95% (11.30–33.17% by draft cache share), acceptance Δmacro −0.79 pp [−1.63, +0.04]; KL from BF16 draft 0.00730, top-1 agreement 95.99%. 256-token continuations: Δ −0.59…+0.33 pp, all CIs include zero.
4. **RQ3 — generalization + real bytes + the honest prototype.** 16K/32K (Qwen2.5 3B/1.5B, PG-19): ~71% draft saved, ~31% combined, acceptance −0.94/−0.73 pp. Packed K4V4 at 32K: 896.0 → 260.2 MiB (−635.8 MiB/seq; metadata costs 1 bit/key-elem + 0.25 bit/value-elem, so 4-bit payload ≠ 75% saving). **The packed Triton prototype is slower, not faster**: read-packed 0.708 ms/layer vs rebuild-then-attend 1.448 ms vs native BF16 SDPA 0.103 ms at 32K; end-to-end +18–33% latency vs BF16 SD; peak allocated memory unchanged (pack-after-BF16-prefill); the batched verifier mismatches target-only on 2–5 of 12–24 prompts **even with a BF16 draft** (cause unfound — timings declared preliminary by the authors).
5. **Appendix F — the proxy law.** Precision policies chosen by speculative acceptance vs by plain draft quality (KL/top-1) coincide at mild budgets (Spearman **0.964 @1K, 0.878 @4K** over a 20-point K/V grid); at a 3-bit budget the acceptance-profiled choice is 0.48 pp WORSE. Direct acceptance is the confirmation metric, not the search tool.
6. **The noise-test cross-link (their App A).** Keys ARE intrinsically more noise-sensitive: Gaussian noise at α = 0.03·RMS drops key-noise top-1 agreement to 61.7% (KL 1.46) vs value-noise 99.2% (KL 0.0016); at α = 0.1 key noise leaves 10.9% vs value 95.3%. The reconciliation with finding 2: per-channel grouping **absorbs key outliers** — sensitivity is handled by layout, not by raw bits.
7. **Group-size + residual-window ablations.** Key groups 16/32/64/128 tokens (K4V2−K2V4: −5.83/−2.94/−1.57/−1.87 pp; only 16-token excludes zero); the 128-token BF16 residual window is a configuration choice, not an optimum (no clear acceptance gain; removing it lifts draft saving 66.33% → 70.56%).

---

## 2. Prior-art state (§4 external sweep + internal workspace sweep) — nothing here is ours to claim

| claim surface | published state | in-workspace state | consequence |
|---|---|---|---|
| Draft-only KV quantization for two-model SD (class) | **This paper IS the published record** (2026-11-16, workshop-accepted); nearest published cousin QuantSpec (Tiwari et al., ICML 2025 — *self*-speculative hierarchical INT4-draft/INT8-target, a different paradigm the paper itself distinguishes) | no record in any `.research/` (grep `2611.20565`, "Quantized Draft KV", "draft cache quant": zero; specdec family notes 031/032/034/037 carry acceptance/head economics, never draft-cache memory) | distillation record, not novelty claim |
| KIVI layout (per-channel keys / per-token values) | Liu et al., ICML 2024 — the paper follows it | known + used as a comparison baseline (katgpt-rs R020 TurboQuant table, R159 KVarN plan) | the layout ships in the literature; the paper's contribution is the interaction WITH bit split |
| "Keys need more bits" stance | Quantize What Counts (Hariri et al., ACL 2026 Findings) — the paper REFUTES it under KIVI geometry | not adopted anywhere | the refutation is the record's value |
| The bit-split × layout interaction principle | **This paper** (Tables 3–4 + Fig 2) | **new to the workspace** — all "bit split" grep hits are OCTOPUS/RotorQuant direction/norm triplet splits (R063/R065/Bench 022/023), a different axis entirely | the principle is the note's durable payload |
| Acceptance vs draft-quality as precision-policy proxy | This paper App. F | adjacent surface exists: `kvq_harness.rs` retention gates (Issue 919 T3), drift guard rel-RMS (Issue 1182) | doctrine row, no code change now |

---

## 3. Distillation

### 3.1 Vocabulary translation (paper → this stack)

| Paper | This stack | Status |
|---|---|---|
| draft cache (BF16, the second KV cache) | no specdec draft cache exists in any shipped lane — 1179 T1 is head-only verify; DSpine (1183) is paused pre-serving | gap, gated behind lane re-open |
| K4V4, per-channel keys (32-token groups) / per-token values | `src/quant/q8kv.rs` `quantize_row_q8_0`: **per-token rows** (one row per (layer, pos), kvd-wide — `kvq_ab.rs` stores `pos * kvd`) blocked into **32-element channel blocks** with per-block symmetric absmax (llama.cpp `block_q8_0` wire) | partial — per-token family with channel sub-blocking, NOT KIVI's cross-token per-channel keys |
| per-channel key exemptions / outlier absorption | Issue 716/919: `ExemptQ8KvCache` — measured top-S channels held at f16 in a per-(layer, pos) sidecar, zeroed out of their block's absmax | ships (measurement lane) — the same outlier-absorption goal, different mechanism |
| acceptance rate as the serving metric | kvq_harness retention gates; drift-guard rel-RMS divergence | adjacent |
| BF16 target cache (correction rule untouched) | our specdec verify lanes keep the target exact by construction (1179: the verify lane prices head-only acceptance; target forward untouched) | consistent by design |
| packed uint8 payload + BF16 metadata | q8kv wire already carries f16 block scales (Plan 586 pinned the f16 storage site); no packed-KV attention kernel | no kernel — and the paper's own prototype says don't build one for speed yet |

### 3.2 Fusion rows

**F1 — The co-design doctrine → the Q8KV lane (forward-arming, no code now).**
Our Q8KV geometry is pinned by code: per-token rows blocked into 32-element channel blocks (`quantize_row_q8_0`; the block scale is per-block symmetric absmax, f16-stored per Plan 586). That is the paper's **per-token family** (finer than their Eq-1 whole-row baseline: one channel outlier poisons its own 32-channel block, not the row) — NOT KIVI's per-channel keys whose groups span 32 tokens. Two consequences, both forward-looking:

- *If a bandwidth-law-motivated asymmetric arm ever opens* (drop V below Q8 — a load-shape change, the one lever Benches 1211/1212 say moves the Q8KV family on Metal), the paper predicts a per-token-family A/B will **inflate apparent key sensitivity** (+10.87 pp-class K8V4 advantage) where a per-channel key layout would show a wash. Doctrine: any such arm evaluates under the DEPLOYED layout, and a per-channel-keys variant backend (the `kvq_ab` A/B pattern extends naturally) co-designs before any "keys need more bits" conclusion. Our channel-exemption sidecar (Issue 716) is a third absorption mechanism in the same family — the note records that the choice between {more key bits, per-channel key layout, channel exemption} is a layout decision, not a bits decision.
- *The two-sided trade the paper alone does not state:* FOSCAR's axis ablation (Research 026) measured token/token fastest and channel/channel at 0.57–0.73× of token/token (INT2, B200) — per-channel key grouping buys acceptance-robustness at a measured kernel-speed cost, and it changes the fold shape (channel-wise K = operand-fold into Q vs token-wise K = acc-fold). The co-design is therefore acceptance × kernel-speed × fold-shape, never acceptance alone.

**F2 — The K4V4 draft-cache operating point → the PAUSED specdec lanes' memory row.**
When 1179/1183 re-open (the pause verdict ranks a **Bonsai-target** specdec lane above restarting the qwen3.8 one; re-open triggers: a trained head ≥3.2 held-out mean_j, a serving-side consumer demanding draft speculation, or a new draft-head objective/architecture), the draft side should boot K4V4-class (per-channel keys / per-token values) as the DEFAULT memory posture candidate, with the paper's cross-family record as the prior: −66% draft cache, −23.95% combined, −0.79 pp acceptance macro; long-context ~31% combined at ≤0.94 pp. For a Bonsai-2-27B PQ2_0 target the draft's KV share of combined memory is large, so the combined saving should land toward the paper's 33% upper end — **inference, flagged, never measured here**. Cross-fixture law (Research 031): the paper's numbers are their models/boxes/A5000s — the first cell when a lane opens is a paired acceptance A/B on the actual target/draft pair, one frozen read, pre-registered bar. Bonus posture note: the residual window is config, not optimum — start at window 0 (saves 70.56% vs 66.33%) unless the paired A/B says otherwise.

**F3 — The proxy law → harness/monitor design (doctrine, no code now).**
Precision policy chosen by cheap draft-quality proxies (KL, top-1 agreement) ≈ policy chosen by direct acceptance profiling (Spearman 0.964/0.878; the acceptance-only allocator slightly worse at 3 bits). If mixed-precision KV ever lands, `kvq_harness` (919 T3) and the drift guard (1182) search policy space with the cheap proxy and confirm with the serving metric — never profile acceptance across the whole grid.

**F4 — The negative-result disclosures (carried as discipline).**
(a) Packed-cache attention: the paper's rebuild-then-attend Triton path is 2× the read-packed path and ~7× native BF16 — any future packed-KV kernel here goes through the FOSCAR fold (never reconstruct; Research 026's Q8KV decode fold is the in-stack template), and the paper's prototype numbers say the speed case is UNPROVEN at 4-bit draft scale. (b) Memory claims: storage ≠ peak — pack-after-prefill moves no peak; a true peak win needs streaming prefill (their named future work). (c) Verifier discipline: their batched verifier mismatched even with a BF16 draft — our specdec lanes' exactness claims ride the sequential/one-call-at-a-time path, never a batched-rollback path without its own audit.

**F5 — The noise-test reconciliation → reframes the principle.**
Keys are intrinsically far more noise-sensitive than values (α = 0.03·RMS: key-noise KL 1.46 vs value-noise KL 0.0016 — ≈900× on KL; top-1 disagreement 38.3% vs 0.8% — ≈48×; at α = 0.1 key noise leaves 10.9% vs value 95.3% top-1), yet per-channel grouping erases the bit-split gap. The mechanism is outlier ABSORPTION by layout. This unifies three published + in-stack responses to the same key-outlier pressure: more key bits (Quantize What Counts), better key layout (KIVI/this paper), exempt the outlier channels (our Issue 716/919). The general law this note records: **key precision demand is a property of the outlier-handling mechanism, not of keys.**

### 3.3 Consumer-surface reframe (step-4 honesty)

No rulewright or game-runtime surface consumes specdec draft-KV memory: the rulewright's fix lanes never draft tokens, and the game planes sync raw `MapPos` scalars across the sync boundary — speculative decoding is serving-substrate-only. The decision-serving family (reflex/rethink) runs modelless lanes, no specdec. One honest sentence instead of a forced reframe.

### 3.4 Path 0 (§3.5) — not applicable

The paper has no training loop and no training deferral arises: the value is quantizer layout + measurement doctrine, all computable at inference. No adversarial panel (clearly inference-side; the §0 technique-stripping pass already ran — nothing survives stripping: the "principle" IS the payload).

---

## 4. Closest cousins in-workspace (fusion check — why they don't kill the record)

| cousin | what it carries | why it doesn't cover this |
|---|---|---|
| Research 026 FOSCAR (Issue 1211, default-on) | Q8KV decode-attention fold over an EXISTING quantized cache; axis ablation | quantizes at symmetric Q8 (no bit split); never touches specdec draft caches or the bit-split question; contributes the kernel-speed axis to F1 |
| Research 031 sigma distillspec (the specdec league row) | draft-head acceptance economics (AL, mean_j), the cross-fixture law, the paused-lane state | acceptance/latency economics only — no draft-cache memory surface, no quantization |
| katgpt-rs R487 + Issue 716 | massive-activation sink/channel outlier handling in a per-token Q8KV layout | outlier-channel exemption mechanism; never the bit-split × layout interaction, never specdec |
| katgpt-rs R159 KVarN / R020 TurboQuant / R063+R065 OCTOPUS-RotorQuant | KV quality mechanisms (variance normalization, rotation VQ, octahedral triplet + direction/norm bit split) | different axes entirely; "bit split" in those records is direction/norm, not K/V |

---

## 5. What this note does NOT claim (mirroring the paper's own §H, plus cross-fixture law)

- We do not claim any K/V bit split works for our models — no in-stack measurement exists; the paper's pairs (Qwen2.5/Qwen3/Llama 3/OLMo 2/SmolLM2) are not our serving targets.
- We do not claim a speedup from packed draft caches — the paper measured the opposite (18–33% slower), and its batched verifier was not exact.
- We do not claim our Q8KV lane should change: it runs symmetric Q8 (8.5 bpw) where the bit-split question cannot arise, and the bandwidth law (Benches 1211/1212) governs its perf axis.
- We do not claim the geometry characterization of `q8kv.rs` beyond what the code shows (per-token rows, 32-element channel blocks, symmetric per-block absmax, f16 scales) — the per-token-family classification is a reading of the pinned code, quoted in §3.1.
- We do not open a lane or file a plan: the consuming lanes are PAUSED (1179/1183) and the doctrine rows arm on triggers recorded in §3.2, per the ungoat discipline (no proof path ⇒ no GOAT filing; this is a RECORD note — the distillation is the durable artifact).

---

## Verdict

**Gain (record-class)** — the smallest honest artifact set: this note + the `.research/.highwater` bump. The co-design principle and the K4V4 draft-cache operating point are new to the workspace records (§2 sweep); both arm on named triggers (specdec lane re-open; asymmetric K/V arm under the bandwidth law) rather than on any current consumer. Public classification (riir-infer): quant layouts and serving doctrine — no moat content.
