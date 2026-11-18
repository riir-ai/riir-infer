# Research 31: Sigma-distill-spec — the continuous-dLM self-speculative family's first shipped checkpoint, the X25 acceptance datum, and the league row

> **Source:** model card [nvidia/Nemotron-Labs-Sigma-distill-spec-3B](https://huggingface.co/nvidia/Nemotron-Labs-Sigma-distill-spec-3B) (HF, released 2026-11-12, NVIDIA Nemotron Open Model License) + paper [arXiv:2610.02665](https://arxiv.org/abs/2610.02665) "Large Language Continuous Diffusion Models" (Yang et al., NVIDIA/Cornell/GT, 2026-10-02, CC-BY-4.0; HTML read in full this session) + family context arXiv:2607.05722 (tri-mode, R055), arXiv:2602.23881 (LK losses, ICML 2026), arXiv:2510.04147 (SSD). All card numbers quoted from the fetched card this session; all paper numbers from the fetched HTML this session.
> **Date:** 2026-11-18
> **Status:** RECORD — Gain (record-class: the league row + the acceptance-training calibration datum + one doctrine refinement; **no lane opens, no plan filed** — the specdec beat lane it feeds is owner-gated, see §5)
> **Related Research:** katgpt-rs R055 (the tri-mode family anchor — self-speculation mode + the L_LK/L_CE drafter-alignment loss distilled there, §7), katgpt-rs R533 (Uno Ψ-spec self-drafter — carries the decisive LK-losses signal-diff), katgpt-rs R638 (LoopUno), katgpt-rs R485 + riir-train R376 (the paper's own PASS-Redirects — schedule/commitment-order axes), riir-refine R334 (Packed-SS — diffusion drafting held NOTE-ONLY/off-lane)
> **Related Plans/issues:** riir-infer Issue 1179 (the qwen3.8 specdec verify lane — T1 closed, Path B is the beat lane), riir-infer Issue 1183 (DSpine lane — T5 stop-branch), riir-train plan 492 (DSpine-class head distill — **STOPPED at its pre-registered stop branch 2026-11-14, mean_j 2.740; resumption requires an owner verdict**), riir-train plan 376 (Uno TV self-drafter — carries the LK warm-start caveat), riir-train R486, riir-ai plan 614 (the M5 verify-kernel L_eff gate this feeds)
> **Classification:** Public (riir-infer)

---

## TL;DR

NVIDIA shipped the first **checkpoint** of the Sigma self-speculative family (post-paper release): a 3B continuous diffusion LM whose **diffusion head drafts 32-token blocks (a single drafting call at the checkpoint's native NFE=2, no self-conditioning)** and whose **own AR head verifies**, with the card claiming the output is **exactly the greedy AR output** — and an acceptance-trained variant whose mean accepted length rose **+14–15%** over the pre-acceptance student. For this stack the value is **not** the checkpoint (PyTorch-only, GPU-only, own backbone — it drafts only for itself and cannot serve qwen3.8/bonsai lanes); it is three durable records: **(1)** the draft-family league row (AL 5.25–10.49 bonus-excluded — the highest raw-AL family published, at a 3-forward-per-round cost structure), **(2)** the X25 acceptance-training datum calibrating the LK-family ceiling our stopped distill lane (plan 492) stopped under, and **(3)** the greedy-exactness doctrine refinement for the walk-16 "lossless-AR vs lossy-diffusion" row. Nothing here is novel — the class (SSD, arXiv:2510.04147) and the objective family (LK losses, arXiv:2602.23881 — already signal-diffed in R533) are published and in-stack distilled; this note is the greppable home for the family row the live owner verdict on riir-infer 1179/1183 will want.

---

## 1. What the checkpoint is (card facts; the paper contains NO specdec)

**The paper (arXiv:2610.02665) has no speculative-decoding content.** Its TOC carries no specdec section; its distillation is PDD (Parallel Decoding Distillation, Shaul et al., adapted to blockwise continuous diffusion; App. I), its steering is embedding-space CFG + score temperature (§4.2), its few-step lane is DDIM uniform-γ + SUSReg + PDD (§5). The `distill-spec` checkpoints are **post-paper releases**; their recipe exists only on the card:

- **Two-head structure** (paper §3.2.2): the Sigma backbone trains a dual-head output — the AR head (causal stream, `L_ar`) and the diffusion head (blockwise NELBO stream, `L_BD`, weight `w_diff` 0.3 after 200B tokens). This dual head is what makes self-speculation possible without a second model.
- **Card recipe**: Phased Distribution Matching Distillation from a multiscale SUSReg-regularized 3B teacher (MS-SUSReg-3B) → student (uniform weight average of P34/P35, 600 updates each) → continued with an **"LK acceptance objective ('X25')"** so the diffusion head drafts block-32 that the model's own AR head verifies. Drafting is one diffusion call at native **NFE=2**, self-conditioning **off** (both trained and served SC-off).
- **Exactness claim (card, verbatim)**: "the returned text is **exactly the greedy autoregressive output** — acceptance changes only how many tokens each round commits, never which tokens."
- **Measured (card table — single release-handoff snapshot, bonus-EXCLUDED AL, prompts restricted to non-looping/non-capped sets, GB200 + SGLang + BF16, one request at a time, speedup vs its own AR)**:

  | benchmark | prompts | AL (bonus-excl) | speedup vs own AR |
  |---|---|---|---|
  | GSM8K (CoT) | 1303 | 10.49 | 3.28× |
  | HumanEval | 158 | 7.90 | 2.61× |
  | MBPP | 499 | 9.21 | 2.61× |
  | SPEED-Bench | 703 | 5.25 | 1.98× |

- **Acceptance-training delta (card)**: paired vs the pre-acceptance Phased-DMD4 student over the same prompts: **+14.9 / +14.1 / +15.4%** AL on GSM8K / HumanEval / MBPP; "AR-decoding accuracy is unchanged by acceptance training." (The **8B sibling card reports bonus-INCLUSIVE** AL on full prompt sets — the 3B card itself warns the two are not comparable.)
- **Not consumable by us**: PyTorch ≥ 2.4 + transformers ≥ 4.55 via `trust_remote_code`; **GPU-accelerated systems only, CPU inference not supported**; 4096-token context; it drafts **for its own AR head only** (different tokenizer/backbone from qwen3.8-27B and Bonsai — a cross-family draft is structurally unavailable); greedy-oriented serving contract. Architecture (paper App. E): 26 layers, width 3072, FFN 9216, 32Q/8KV heads, head dim 128, RMSNorm, SwiGLU, YaRN RoPE, 16-D unit-sphere diffusion embeddings, learned Bernstein noise schedule (card: γ endpoints [−5.4150, 4.8549]).

## 2. Prior-art state (§4 searches + internal sweep) — nothing here is novel

| claim surface | prior art / in-stack state | consequence |
|---|---|---|
| Diffusion model drafts for itself, AR verifies (class) | **SSD — "Self Speculative Decoding for Diffusion Large Language Models"** (Gao et al., arXiv:2510.04147, Oct 2025, OpenReview) + Packed-SS (arXiv:2610.33379, held NOTE-ONLY in riir-refine R334 with the lossless-AR-vs-lossy-diffusion doctrine row) + the tri-mode report itself (R055, mode 3, with 5.99× TPF claims) | the class is published; no novelty |
| Acceptance-rate-directed draft training (objective family) | **LK losses** (Samarin et al., Nebius, arXiv:2602.23881, ICML 2026, CC-BY-4.0): KL-min and acceptance share the global optimum but capacity-limited drafts converge where minimizing KL ≠ maximizing acceptance; forms include −log-acceptance and hybrid λ·KL+(1−λ)·TV; **+8–10% average acceptance length** across 4 draft architectures × 6 targets (8B–685B); "no computational overhead… directly integrated into any existing speculator training framework". **Already signal-diffed in-stack**: katgpt-rs R533's landscape row (pure TV from scratch is unstable — warm-start regime caveat; encoded as plan 376's constraint), R486's positioning row, R055's references | the objective family is published AND in-stack adjudicated; the card's "LK acceptance objective (X25)" names this family (X25's exact coefficients are not published — inference, flagged) |
| The L_LK + L_CE drafter-alignment formula | R055 §7 (from the tri-mode report): `L = λ_KL·L_LK + λ_CE·L_CE`, LK-hybrid distribution matching on top-K union support + CE on verifier argmax, active positions = accepted prefix + first rejected, τ=3.0/τ=1.0 | formula already distilled |
| Continuous dLM at scale / schedules / steering | the paper itself: three PASS-Redirects already carry it (katgpt-rs R055 — family anchor + "no consumer in our discrete-masked lane"; R485 — uniform-γ discretization converges with UGC's equal-√q-mass grid; riir-train R376 — commitment-order axis) | covered |
| Fleet consumer check | workspace grep this session: **no continuous-diffusion / score-temperature / Bernstein-schedule lane exists in any `*.rs`** (Bernstein hits are unrelated empirical-Bernstein bounds machinery in `hint_regret`/`ugc_schedule`); the fleet's own diffusion-drafter cousin program is **Uno/LoopUno** (R533/R638 + riir-train plan 376 — a TV-loss diffusion-LoRA self-drafter for looped hosts), a different host class | no lane to arm; family recorded |

## 3. The league row (cross-fixture discipline — never printed as our measurement)

Where the Sigma family sits on the draft-head acceptance axis our lanes track:

| family / lane | AL (mean accepted length) | cost structure | provenance class |
|---|---|---|---|
| DFlash | 5.92 | small head, ≈free draft | 8B same-budget retrain, 7-bench mean, T=0 (paper-reported; carried in riir-infer 1183's cross-fixture table) |
| Domino | 7.05 | small head | same class |
| DSpark | 7.44 | small head | same class |
| DFlash2 | 7.55 | small head | same class; our 4090 loop banked 1.242× (riir-ai 846 lose table) |
| DSpine | 7.83 | 5-layer/2B draft | same class; 27B card 4.48–7.20; **on OUR target: untrained 2.565 (Bench 965), distill-trained 2.740 (plan 492 arm-1, stopped)** |
| **Sigma self-spec (this note)** | **5.25–10.49 (bonus-excluded, per-bench)** | **≈2–3 forwards/round (derived, unresolved; ≤33-row verify)** | 3B own-backbone, GB200/SGLang/BF16, greedy — **cross-fixture: different model, box, runtime, temperature; never projected onto bmo/4090 numbers** |
| our target measured | uzu head 1.27 (Bench 954) · DSpine 2.565–2.740 | small head vs 27B Q4_K_M | our boxes, our fixtures |

Reading rules (the 1179 verdict-round-1 discipline):

- **AL conventions differ and are material.** The 3B card's AL is bonus-EXCLUDED (a round commits AL+1); the 8B sibling card is bonus-INCLUSIVE; our `mean_j` is `emitted_total/cycles` (Bench 954 definition — committed tokens per cycle), i.e. our convention already sits on the bonus-INCLUSIVE side. At AL 2–11 a ±1/round convention shift is a material fraction; **pin the convention per row before any cross-family comparison**. The 3B card's own comparability warning is the precedent. Direction for the plan-614 comparison below: measured against our bonus-inclusive `mean_j`, the card's bonus-EXCLUDED AL **understates** Sigma by ≈1 token — never the reverse.
- **Per-bench, never pooled.** Sigma's own spread is 2.0× across benches (5.25 → 10.49); a single pooled number hides that. Our league should carry per-bench rows beside any mean.
- **(AL, cost) pairs, never raw AL.** Derived arithmetic (ours, from the card's own numbers; **derived, unresolved**): a bonus-excluded round commits AL+1 tokens (GSM8K 11.49). A 3-forward round (2 diffusion NFE + 1 verify) gives ideal ≈ 11.49/3 ≈ 3.83× vs measured 3.28× ⇒ overhead ≈ 0.86, ≈ 3.8 committed tokens/forward; a 2-forward reading gives ideal ≈ 5.75× ⇒ overhead ≈ 0.57, ≈ 5.7 tok/forward — arithmetically possible, so the round is **bounded to 2–3 forwards and unresolved from the card alone** (it says "1 diffusion call" while quoting native NFE 2); 4 forwards is ruled out (ideal 2.87× < measured). A k≈33-row verify pass is not one AR decode step of cost, so forwards are a weak proxy. Under either reading, per-forward efficiency still favors the small-head lanes (a ≈free draft buys ≈AL tokens per ≈1 verify pass — e.g. raw-AL 7.83 ⇒ ≈7.8, convention unpinned per row); **the Sigma family wins on raw AL and on the quality-locked single-model story** (no extra 2B draft resident). Neither half is a measurement on our hardware.
- **Speedup ≠ AL.** Their 3.28× is against their own AR on GB200; our beat arithmetic (1179) runs at serial ~12 tok/s on M5 Ultra where draft cost is nearly free (Bench 955/1167 decomposition). No projection either direction.
- **riir-ai plan 614 context**: the M5 verify-kernel gate needs acceptance `L_eff ≥ 4.78–10.3` at today's kernels. DSpine's card (4.48–7.20) was the first family claimed to plausibly enter that range; **Sigma's 7.90–10.49 (greedy, bonus-excl) enters it too** — a second family datum, same caveat (cross-fixture, own model). On the convention axis this comparison **understates** Sigma: bonus-inclusive equivalents read ≈8.9–11.5, while our `mean_j` 2.74 is already bonus-inclusive.

## 4. The X25 datum — calibration for the stopped distill lane (owner-gated)

The relevant history, all recorded in live lane docs:

- the riir-train Issue 493 acceptance-training ladder on the z-lab DFlash2 family: **monotone DOWN** (2.385/1.949/2.233/2.076 vs z-lab 2.648) — the naive acceptance-shaped form is refuted in-stack on the old family. (493's own issue file is removed per the noise-reduction rule — git history; the ladder numbers are carried by the riir-ai Issue 846 lose table, and plan 492's binding-priors row cites the 2.385 head to riir-ai Bench 820.)
- riir-train plan 492 (DSpine-class vs our Q4_K_M runtime): objective = score-row CE + mixed teacher/on-policy prefix frac + depth-0 ×2 + miss-only aux full-vocab CE. Arm-1 (frac 0.25): mean_j **2.740**, G1 PASS, promote bar ≥ 3.2 FAIL, delta +0.08 < the +0.3 stop line → **pre-registered STOP, report without promoting** (Bench 709; 2026-11-14). The plan's own diagnosis: "the objective works, the acceptance ceiling doesn't" (trajectory front-loaded 3.636@c10, decaying).
- LK losses' premise (arXiv:2602.23881) matches that diagnosis verbatim: KL-min and acceptance diverge for capacity-limited drafts. Its measured effect on **AR draft heads**: +8–10% acceptance length. R533's signal-diff adds the in-stack constraint: warm-start only, never TV-from-scratch.
- **Sigma X25 instance**: +14–15% AL on a **continuous-diffusion student drafting for its own AR head** — the self-spec error-correlation regime (draft/verifier share weights), which inflates acceptance gains vs a cross-model draft; transfer of that magnitude to an AR head on a foreign 27B target is **not evidenced**, and LK's own AR-head numbers (+8–10%) are the better prior.

**The arithmetic the owner verdict needs** (derived; both adversarial-panel advocates converged on it independently; **anchors corrected at verdict round 1**): plan 492's pre-registered gates measured delta against the **UNTRAINED baseline** (box zero-shot 2.660, recorded 2.565 — plan 492 §T3), not against the trained 2.740: the +0.3 stop bar read 2.960 (box baseline; 2.865 against the recorded 2.565) and **2.740 already failed both** (+0.08 over 2.660); the promote bar is **3.2 = +16.8% over 2.740** and is the only live reference number — a re-opened lane pre-registers FRESH bars rather than inheriting these. On published magnitudes from the 2.740 base: the LK AR-head prior (+8–10%) lands **2.96–3.01**; the diffusion-instance magnitude (+14–15%) lands **3.13–3.16**. **No single acceptance rung reaches 3.2**; the only arithmetic path is compound (raise the base drafter, then acceptance-continue the better base). This is recorded as calibration input, **not** as a proposal: the stop-branch record says resumption requires the owner verdict on the beat lane (1179/1183), and no agent lane claims it before that lands.

## 5. Doctrine refinement — the walk-16 "lossless-AR vs lossy-diffusion" row

riir-refine R334's Packed-SS row held diffusion drafting off-lane under a lossless-AR vs lossy-diffusion framing. The Sigma artifact forces a two-line split:

1. **At T=0 the verify mode is lossless by construction** — if acceptance is prefix-match against the AR head's greedy output, the committed stream is exactly greedy AR regardless of what produced the draft; quality collapses out of the comparison and AL-per-cost is the only axis. (This is also the natural G1 for any future exact-drafter row: byte-identity vs a serial k=1 run — the 1179 G1 definition already spells exactly this harness.)
2. **The sampled arm is unlicensed, not lossy-by-default.** Losslessness lives in the ACCEPTANCE RULE (min(1, p/q) rejection needs a per-token proposal distribution); a continuous-diffusion drafter exposes no per-token categorical proposal, so true rejection sampling is undefined for it unless the vendor documents a rule. The card is silent on the sampled arm. Any future lane touching this family must treat τ>0 diffusion-draft rows as **no-verdict on losslessness** until the acceptance rule is read at a pin — never "lossy" (unproven) and never "lossless" (the greedy claim does not transfer).

## 6. Path 0 inventory (complete — the funnel's outputs)

| component (paper/card) | analog in stack? | training-free extraction? | disposition |
|---|---|---|---|
| Self-spec two-head decode (diffusion draft + AR verify) | Partial — verify-loop machinery specified (1179 T2/T3, unbuilt; blocked on serial ≥ ~40 + owner verdict); `SpeculativeVerifier`-class pattern exists in katgpt-rs; D2fDrafterVerifier idea recorded in R055 as "a variant, not a new system" | architectural only; class published (SSD) | recorded (this note); **no lane** |
| LK acceptance objective / X25 continuation | Objective family published (2602.23881) + in-stack signal-diff (R533, plan 376 warm-start caveat); no riir-train implementation of an LK rung exists | no (it is a training objective) — the DEFINITION + magnitude priors extract | calibration datum (§4); plan-492-class re-open lever **owner-gated** |
| Phased DMD | no analog (PDD is the paper's lane; DMD family published Yin et al.) | no | recorded; no consumer |
| AL measurement protocol (bonus convention, per-bench rows, (AL, cost) pairs, posture fields) | partial — mean_j harness exists (Bench 954/965 definitions); the convention pin + per-bench + cost columns do not | **YES — pure measurement work** | recorded as the named harness follow-up IF the beat lane re-opens (rides that lane; no standalone issue filed) |
| CFG w / score temperature τ / commit-stride / uniform-γ | no consumer (no continuous dLM anywhere in fleet `*.rs`) | closed-form but surface-less | recorded; no action |
| The checkpoint bytes as a measured baseline row | no | measurement-only, hosted PyTorch/GPU harness — possible on the 4090 only; license check (NVIDIA Nemotron Open Model License) before any byte | optional, owner-adjacent; not filed |

## 7. Verdict

**Gain (record-class).** The tier reasoning, explicitly: not Super-GOAT (nothing novel — class and objective are published, in-stack signal-diffs exist); not GOAT (no gain is provable over an existing approach — no lane consumes anything here); not Pass (the checkpoint row, the X25 calibration datum, and the §5 doctrine split are exactly what the LIVE owner verdict on 1179/1183 and any plan-492-class re-open will grep for — the distillation itself is the durable artifact, which is the note-not-redirect case). MOAT gate: riir-infer is the specdec league's home (fusion-priority-3 inference-perf lane; `.research/` 028–030 are the league-intel precedent) — routing confirmed. The recipe half points at riir-train plan 492's class but **no plan is filed** (owner-gated stop branch; filing one would spend a lane that is not open).

**Proof path (the ungoat discipline):** every number above carries its source (card table / paper section / bench number / derived-and-labeled); the falsifiers are named (§4's transfer-attenuation caveat, §5's sampled-arm no-verdict rule, §3's convention pin); the gate that would decide any future step is named in each row.

---

### PASS-Redirects (synthesis)

> **PASS-Redirects (synthesis):** Gao et al. [arXiv:2510.04147 "Self Speculative Decoding for Diffusion Large Language Models"] — the published class for diffusion-self-drafting + verify; Sigma-distill-spec is its first NVIDIA-shipped continuous-dLM checkpoint instance; recorded here as the family row, no lane.
> **PASS-Redirects (synthesis):** Samarin et al. [arXiv:2602.23881 "LK Losses: Direct Acceptance Rate Optimization for Speculative Decoding"] — already signal-diffed in katgpt-rs R533 (warm-start regime caveat; −log-acceptance and hybrid forms) and encoded in riir-train plan 376; this note adds only the X25 instance magnitude (+14–15% on a self-spec diffusion student) as calibration for the owner-gated plan-492-class re-open.
> **PASS-Redirects (synthesis):** the paper arXiv:2610.02665 itself was already triple-redirected at distill time (katgpt-rs R055 family anchor, R485 schedule axis, riir-train R376 commitment order) — this note adds the post-paper CHECKPOINT row those redirects could not carry (they pre-date the 2026-11-12 release).

### Numbering + gate record

`.research/.highwater` read at write time: 30 → this note allocates **031**; highwater bumped to 31 in the same commit. **`scripts/dual_allocation_gate.py` PASSES** — run this session after `git fetch origin` at verdict round 1: 0 twin / 0 independent / 0 counter-only. (The coordinator's own terminal invocation had been refused mid-session by an environment failure — every command, including plain `ls` in previously-working roots, returned a workspace-ambiguity error after earlier identical invocations succeeded; the reviewer's post-fetch run above is the gate evidence, this sentence the outage record.) If a sibling box still allocated 031 concurrently, the citation-weight rule adjudicates and this note renumbers.
