# Research 32: MMFlash — visual-dependence-weighted diffusion drafting, a third DFlash-class head instance for Qwen3.8-27B, and the specdec lane intel

> **Source:** paper [arXiv:2611.18482](https://arxiv.org/abs/2611.18482) "MMFlash: Faster Speculative Decoding for Vision-Language Models" (Feng, Chen, Wang, Fang, Wang — NUS, 2026-11-13, arXiv CC-BY-4.0 HTML read in full this session) + model card [FSCCS/Qwen3.8-27B-MMFlash](https://huggingface.co/FSCCS/Qwen3.8-27B-MMFlash) (HF, 2026-11-16, **license not stated on card — T0 item**) + repo [github.com/fscdc/MMFlash](https://github.com/fscdc/MMFlash) (structure fetched this session; training code builds on SpecForge, drafter follows DFlash, export = a `DFlashDraftModel` checkpoint stock SGLang/vLLM `DFLASH` loads with no patches).
> **Date:** 2026-11-18
> **Status:** RECORD — Gain (lane intel + training recipe; **no lane opens** — the specdec beat lane it feeds is owner-gated; the recipe files as riir-train Issue 698 under the same owner verdict). **Claude verdict: AGREE round 3 (2026-11-18, session c20b572c; rounds 1–2 REVISE — all findings folded: the head-INSTANCE framing per 1179's round-1 ruling, the rank0 premise marked RETRACTED-CLASS with the ⚠ block, the expectations bound led by the LK prior per R031 §4, the bars anchored on the trained base with the +0.32 compound bar FRESH-re-derived, the three-zone T5 outcome rule)**
> **Related Research:** riir-infer R031 (the Sigma distill-spec sibling note, in flight at filing — carries the acceptance-training calibration datum: an acceptance-trained variant gained +14–15% mean accepted length; the sharp expectations bound below cites it), katgpt-rs R533 (Uno Ψ-spec — carries the LK-losses signal-diff per R031's related line)
> **Related Plans/issues:** riir-infer Issue 1179 (qwen3.8 specdec verify lane — T1 closed, Path B is the beat lane), riir-infer Issue 1183 (DSpine lane — T5 stop-branch), riir-train plan 492 (STOPPED at its pre-registered stop branch, mean_j 2.740; resumption requires the owner verdict), riir-train Issue 679 (STOP-BAR FIRED record), riir-train Issue 693 (composed-asymmetry cells; Cell B lane-gated on the same verdict), riir-train Issue 493 (removed; the 4090 chat specdec closure whose reopen conditions 1179 quotes), **riir-train Issue 698 (filed by this note — the recipe)**
> **Classification:** Public (riir-infer)

---

## TL;DR

MMFlash extends DFlash block-diffusion drafting to VLMs and its two findings transfer to our lanes **without owning anything multimodal**: (1) **a public Qwen3.8-27B drafter checkpoint exists** (`FSCCS/Qwen3.8-27B-MMFlash`, 1.73B bf16, block 16, DFlash-class, DFLASH drop-in) — a third, independently-trained DFlash-class head INSTANCE for the beat lane (the reopen condition 1179's verdict-round-1 rewording names is an INSTANCE above the named 2.648 ceiling, **not a new family**), measurable on our T5 head-only harness with zero training; (2) **acceptance loss concentrates on a token class** — drafter acceptance falls 46%→22% across input-dependence quartiles, Q4 holds 44% of first rejections, and the gap persists after simply training on more multimodal data; **objective reweighting, not more data, moves it**. The recipe's text-only port is the plan-492-successor objective (chain-survival positional weights × focal miss-boost — filed riir-train Issue 698), with the sharp expectations bound (led by the honest prior): the LK AR-head datum (+8–10% relative → 2.96–3.01) and even the optimistic self-spec-regime magnitude (+14–15% → 3.13–3.16, inflation-caveated in R031) BOTH land UNDER our ≥3.2 promote gate — the gap is +16.8% relative, above the whole published band; only a compound above it, or a gate re-derivation, clears.

**Distilled for riir-infer (league lane intel + harness upgrades; nothing trained here):**
per-stratum acceptance reporting beside mean_j; the chain-survival product as an ONLINE depth policy (no training, works on any drafter including the stalled zero-shot); the sparse text-half draft-context window as a long-context serving lever; block-size 8-vs-16 as an eval row. All records; no feature lands.

---

## 1. Paper core findings

- **Diagnostic (Eq 1):** per-token visual dependence `d = KL(p(·|prefix,I) ‖ p(·|prefix))` — two target forwards per profiled position (with/without the image), computed OFFLINE from the target alone, no drafter, quantile-normalized over the corpus (rank → [0,1]).
- **Empirical law (Fig 3/5):** acceptance falls monotonically 46% (Q1) → 22% (Q4), 2.1× gap; Q4 holds 44% of first-rejection (truncation) points vs 7% in Q1; **persists after multimodal training** (multimodal data alone: +5 pts Q1, <3 pts Q4). Block-level: 33% (Q1) → 14% (Q4).
- **Loss (Eq 2–4):** `w_k = b_k·(1+α·g_k·(1−sg(q_k)))`, α=1, λ=0.5, replacing DFlash's positional decay `exp(−(k−1)/γ)`:
  - `g_k = g̃_k·exp(−H_k)` — dependence × target confidence ("where the input both CHANGES and SETTLES the prediction");
  - `b_k = Π_{i≤k}[(1−λ)·sg(q_i)+λ]` — verification-aware chain survival (gradient at k matters only if 1..k survive);
  - `(1−sg(q_k))` — the drafter's current miss probability; the boost fades as tokens are learned.
- **Sparse draft context (video, 41K→1.6K tokens):** 4 sinks + stride-32 visual + all text + 256 local window, SAME pattern train+inference. Training-free adoption alone: **+11.1% acceptance, +4.7% speedup**. Measured: 50.1% of drafter attention lands in the most recent 256 tokens (0.8% of context).
- **Results:** Qwen3.5-4B/9B 3.00–3.05× greedy; **Qwen3.8-27B τ 5.36–7.57 / 3.54–4.71×** (their Table 5 + card — CROSS-FIXTURE: multimodal benchmarks, H200, SGLang — never quoted as ours). Ablations: 5-layer drafter sweet spot; 8 taps = +19.9% training time for ≤0.01×; test-time block 8 vs trained-at 16: −5.3% speedup, still beats baselines.

## 2. The consumables for the league lanes

### 2.1 The checkpoint — a head-instance candidate for the T5 cell

`FSCCS/Qwen3.8-27B-MMFlash` is a DFlash-class drafter for OUR league target. The consume map is shorter than DSpine's was:

| piece | status |
|---|---|
| target tap set | card: layers **1, 16, 31, 46, 61** — EXACTLY the DSPINE tap arm plan 492 T1 landed (`QWEN38_DSPINE_TAP_LAYERS = [1,16,31,46,61]`, `forward_verify_chunk_taps_dspine`) — **the tap seam exists** |
| draft architecture | DFlash block-diffusion, block 16 — the 1179 machinery inventory's "MISSING on Metal dense" row (k-position batched verify + KV rollback + lossless sampling) still applies (1179 T2/T3, unclaimed) |
| container | `DFlashDraftModel` safetensors, DFLASH drop-in — the uzu/Path-A loader class; plain bf16 (no uint4-affine wrinkle) |
| gates | head-only acceptance cell FIRST (the Bench-954/965 convention: mean_j = emitted_total/cycles, j=0 keys 0 cycles, G1 serial==specdec over 32 committed tokens), e2e only after; **cross-fixture disclosure mandatory** — card/paper numbers are multimodal/H200, never ours |
| caveats | **license not stated on the card or repo README — T0 verification before any download**; trained on ~900K MULTIMODAL samples (LLaVA-OV-1.5-Instruct regenerated) — text-chat acceptance on our fixture is UNMEASURED and could land anywhere between uzu's 1.27 and above the z-lab 2.648 ceiling; the cell decides |

Why it matters: the beat lane's HEAD condition, as 1179's verdict-round-1 rewording fixes it, is **"a new head INSTANCE above the named ceiling, not a new family"** — the named ceiling being the z-lab 2.648 class (the closure-era wording "a new head family or a target swap" is superseded in-file: z-lab and uzu were ruled the SAME DFlash family). The uzu head (1.27) is that family's worse-trained instance; DSpine (2.565 zero-shot / 2.740 trained, stopped) is the one alternative FAMILY measured; **MMFlash is a third, independently-trained DFlash-class INSTANCE from a different recipe — exactly the instance condition's shape, IF it measures above the ceiling on our fixture (unmeasured; the cell decides)**. Measuring it costs one T5 cell, no training, no beat-lane claim (it rides the same owner-gated lane; the cell is the DECISIVE INPUT to the owner verdict, never an auto-start — the Issue-693 Cell-B posture).

### 2.2 The diagnostic — acceptance stratification as a harness upgrade

The paper's core measurement is portable to our harness verbatim: report acceptance **per stratum** beside aggregate mean_j. On our text lane the dependence axis has no visual half — the honest stratifiers available without a VLM are: target entropy H per position (already materialized in any distill pass), drafter confidence q, position-in-block, and token class. The no-GD advocate's framing stands: mean_j conflates corpus composition with head quality (their law: acceptance is class-dependent, so the aggregate moves with the mix); per-stratum rows separate "the head is bad at a class" from "the corpus shifted". Unblocked today, zero training, no owner gate — a harness-reporting upgrade on the existing T5/T3 rig.

### 2.3 The recipe — training track, filed as riir-train Issue 698

The loss recipe ports to the plan-492 successor with **zero extra target forwards** (all signals — q per position, target entropy — already materialized in the distill pass). Full recipe, pre-registered bars, expectations bound, budget, and the do-NOT-port row: **riir-train Issue 698**. Summary here:

- **Lead arm:** `w_k = b_k·(1+α·(1−sg(q_k)))` — chain-survival × focal, dropping the visual-dependence term (no text-only analog for the "changes" half — honest NO-ANALOG; do not fake one). Keep mixed-frac as side policy (the only knob that ever moved the head), replace the uniform score-row weight.
- **Mechanistic alignment with the measured profile (valid numbers only):** the trained arm's survival decays p1/p2/p3 = 0.635/0.415/0.240 (Bench 709) — accepted mass thins with depth. Under `b_k`, position 0 carries weight 1 by construction (empty product) and deep rows down-weight by actual reachability — gradient mass aligns with the survival profile and with the truncation semantics (a gradient at k pays off only if 1..k survive — the paper's verification-aware argument, unconditional); λ ∈ {0.25, 0.5} load-bearing (never 0). ⚠ **NOT claimed:** the earlier "chain head never matches, rank0 ≈ 0" diagnostic came from the RETRACTED T5 cell (mean_j 1.000, the broken message-gate hyperbola forward; plan 492 records that premise dead — on the FIXED forward the zero-shot head reads 2.565, so position 0 demonstrably matches sometimes); re-measure rank0 on the fixed forward if the successor wants that decomposition.
- **Expectations bound (sharp, led by the honest prior — R031 §4's ruling, lines 75–78):** the LK AR-head datum (+8–10% relative) lands 2.96–3.01 from the 2.740 base; the optimistic self-spec-regime magnitude (+14–15%, inflation-caveated: draft/verifier share weights there, cross-model transfer not evidenced) lands 3.13–3.16. BOTH sit under the ≥3.2 promote gate, whose gap is +0.46 = +16.8% relative over the trained base — above the entire published band. **No single acceptance rung reaches 3.2** (R031's arithmetic; both panel advocates converged on it independently); the only paths are a compound exceeding the bands, or a gate re-derivation (3.2 came from 1179's beat arithmetic at serial 45–55 — if a retrained head changes serial cost, the bar moves WITH THE ARITHMETIC, never by fiat).
- **Budget:** ~100–190 GPU-h stage-gated on the 4090 (lead arm ~21 GPU-h); spend halts at the first dead branch.

## 3. Path 0 inventory — the modelless table

| paper component | verdict |
|---|---|
| visual-KL dependence census `d` (Eq 1) | **VLM-gated, not ported** — no generative-VLM consumer in the stack; annotation, never a feature |
| quantile normalization (rank→[0,1]) | extracted (trivial; consumed only if `d` ever ships) |
| **acceptance-stratified harness rows** | **extracted — durable record** (§2.2; unblocked, no owner gate) |
| **chain-survival product as ONLINE depth policy** | **extracted — the strongest no-GD runtime lever**: the paper's training weight `b_k` is equally a runtime survival estimator — the drafter emits per-step confidence for free; stop proposing when the running product < τ. Works on ANY drafter including the stalled 2.565 zero-shot DSpine. Proposal-class (needs its own cell + pre-registered bar), never a default flip |
| sparse text-half draft-context window (sinks + text + local) | **extracted as a long-context serving lever** — training-free +11.1% acceptance on their video posture; ours to re-measure on a ≥4K-context cell; the window fraction is per-model (their 50.1%@256 is NOT ours — re-measure before pinning) |
| block-size 8-vs-16 robustness | **extracted** — a b8 eval row beside b16 per artifact (their ablation: trained-16/run-8 = −5.3% speedup, still > baselines; also unlocks a dynamic-block posture without retraining) |
| target-entropy `exp(−H)` gating | extracted into the RECIPE only (riir-train 698 ablation arm); no standalone runtime consumer |
| loss reweighting itself | NOT modelless — training track, Issue 698 |

## 4. Verdict

**Gain — lane intel + recipe. NOT Super-GOAT, NOT a katgpt-core primitive.**

- One-line reasoning: every mechanism class is published (LK losses arXiv:2602.23881 — direct acceptance-rate objectives; D-PACE arXiv:2605.18810 — dynamic position-aware weights; verification-aware training arXiv:2608.30135; MMFlash is the SOTA *instance* of the reweighting class) — we CONSUME a published method's checkpoint and recipe; no new capability class, no open primitive (the diagnostics need a generative target forward; the modelless engine has none).
- Per-track split (one verdict per track): **inference track** = the §2/§3 consumables (records + harness upgrades; Gain); **training track** = riir-train Issue 698 (Gain; execution owner-gated under the 679/492 stop verdict).
- MOAT gate: riir-infer ✓ (league/substrate lane intel; the note lives here); no katgpt-rs file; no riir-ai/chain/db surface.

## 5. Proof paths (ungoat law — every filing carries its gate)

- **T5-cell proposal** (head-instance candidate): head-only acceptance on our harness, Bench-954/965 convention, G1 serial==specdec/32, blake3-pinned checkpoint + target at load (A1/A8), cross-fixture disclosure on every quoted foreign number. Three-zone outcome rule, pre-stated: at-or-below the uzu 1.27 class ⇒ the recipe's gain did not transfer — record + close; between 1.27 and the 2.648-class ceiling ⇒ record as a BELOW-CEILING instance, the lane stays stopped, the reading goes to the owner verdict as calibration input; above the 2.648-class ceiling ⇒ the instance condition is met and the reading re-arms the lane's owner-verdict file.
- **Stratified rows** (harness upgrade): ordering law fixture — strata must be monotone on our corpus; control arm = label-shuffled strata must flatten within CI.
- **Recipe bars** (Issue 698): per-lever +0.10 mean_j (anchored on the 2.740 trained base) two-run stable; compound-continuation +0.32 (FRESH-re-derived, see Issue 698); promote ≥3.2 absolute unchanged; **falsifier: b+focal <+0.05 ⇒ the reweighting class is acceptance-ceiling-bound (plan 492's verdict generalizes to the objective axis) — record, route to Issue 693 T2 (composed GRPO), close**.
- **Depth-policy + sparse-window proposals**: their own cells with pre-registered bars BEFORE any posture change; stationary-surface law respected (these are LEVEL readouts + one policy arm, no dynamics claim).

## 6. Dedup + cousins (why nothing here is killed by what ships)

| cousin | relationship |
|---|---|
| uzu DFlash head (1179 Path A, mean_j 1.27) | the same DFlash ARCHITECTURE class, worse-trained instance (1179's round-1 ruling: z-lab and uzu are ONE family) — MMFlash is the class's independently-trained counterexample INSTANCE; the T5 cell prices the delta |
| DSpine (1183, 2.565/2.740) | the one alternative head FAMILY measured on this target + the SAME tap seam; MMFlash's recipe composes with (never duplicates) plan 492's mixed-frac — different axes of the same loss line |
| LK losses (2602.23881, R533/R031) | replaces the loss FUNCTION vs MMFlash's reweighting of cross-entropy — same goal class, different mechanism; R031 §4 supplies the expectations bound (LK prior leads, self-spec magnitude caveated) |
| z-lab ceiling 2.648 | the acceptance wall the beat lane quotes; MMFlash's head is exactly the kind of new instance the reopen condition names |
| Issue 693 T2 (composed GRPO) | the escalation if the reweight arm fails its falsifier — b_k is the in-pass surrogate of the composed reward; sequencing recorded in Issue 698 |

## 7. Caveats

- **License unverified** on the model card and repo README (T0 before any download; the paper is CC-BY-4.0, the code builds on Apache-2.0 SpecForge — neither covers the checkpoint).
- The checkpoint's text-chat acceptance is **unmeasured**; the training corpus is multimodal instruction data — transfer is a hypothesis, the cell decides.
- R031 is a sibling session's in-flight note (zed-36aeec2d@m3); the expectations bound cites R031 §4 **lines 75–78 specifically** (its LK-prior ruling + the 2.96–3.01 / 3.13–3.16 arithmetic, anchors corrected at R031's own verdict round 1) — if R031 lands with corrections, the bound re-reads from the landed note.
- `.research/.highwater` advanced 31→32 over the sibling's uncommitted in-flight bump (their 031 file untouched; monotonic, both valid).
- All foreign numbers in this note are cross-fixture (multimodal benchmarks, H200, SGLang) and are never quoted as ours — the standing 1179 verdict-round-1 discipline.

---

Session: zed-21e8a973@m3, epoch 1794962704 · Panels: No-GD advocate 72b04f41 (16 extractions) + Model-based advocate af966468 (10 items, grounded in plan 492/679/693/Bench 709) · Coordinator spot-checks passed: issue 679/693/R533/Bench 709 on disk, survival 0.635/0.415/0.240 verified in 679 (the VALID Bench-709 numbers — the rank0 diagnostic is RETRACTED-CLASS and carries no spot-check pass; see §2.3's ⚠ block), tap set [1,16,31,46,61] verified in plan 492 T1.

> **PASS-Redirects (synthesis):** Kim et al. [2DSD: Two-Dimensional Speculative Decoding for Looped Language Models, github.com/SqueezeAILab/2DSD @ 431aade, 2026-11-19] — the looped-target sibling of this note: the fourth DFlash-family instance + the unified (r1, r2, k) two-axis schedule + last-loop KV sharing (lagged window W=64); its drafter DATA-recipe half files riir-train Issue 702 (complementary to 698's objective half), PARKED under the same d9c2bb28 pause; riir-infer `.research/034_2DSD_looped_2d_specdec.md` carries the record.
