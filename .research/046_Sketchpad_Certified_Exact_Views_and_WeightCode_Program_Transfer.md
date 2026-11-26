# Research 46: Sketchpad Certified-Exact Views + WeightCode Program Transfer — the two WSM reference follow-ups

> **Source (two-paper fusion note, the Research 045 §3 follow-ups):**
> - **Sketchpad: certified exact arithmetic with a small vision-language model** — Mostafa 'Neo' Mohsenvand (MIT Media Lab), [arXiv:2611.04434](https://arxiv.org/abs/2611.04434), v1 2026-11-03. 25 pp. Code **Apache-2.0** (github.com/NeoVand/sketchpad — runtime, algebra, baselines, Lean proofs in `paper/lean/`); model + adapter + certificate on HF (`mohsenvand/sketchpad-0.8b-preview`). WSM ref [31].
> - **WeightCode: Installing New Capabilities into Neural Networks via Weight-Space Programming** — Bo Li, Taoyong Cui, Bob Zhang, …, Ling Yang, Yingcheng Wu, [arXiv:2611.25476](https://arxiv.org/abs/2611.25476), v1 2026-11-19. Code: github.com/Gen-Verse/WeightCode. WSM ref [32] (their named concurrent work).
> **Date:** 2026-11-26
> **Status:** Active — Gain (both, track-separated). Feeds riir-infer Issue 1256 (proof-path upgrade + T2 reference arm); training recipe = design input with trigger, NO new riir-train plan (the named tension, adjudicated in the verdict).
> **Related Research:** riir-infer 045 (WSM — the parent distill; this note discharges its two open §3 follow-ups), 018 (Qwenputer — exact-by-construction in-weights), 044 (Plato's Key — ends re-solve seam), 013 (TokenGate); katgpt-rs 621 (Handwired exact), 648 (WSP parent — WeightCode's class), 678 (iWSP), 690 (LoopedWSP); katgpt-rs Issue 958 `sst_certified` (the internal certificate to signal-diff); riir-chain/riir-dapps/riir-neuron-db/katgpt-rs Lean instances (our FV discipline).
> **Grep disambiguation:** "Latent Sketchpad (Zhang 2025)" (katgpt-rs 555's cite) and "Visual Sketchpad" (Hu et al. 2024, this paper's own ref [87]) are DIFFERENT papers — grep `2611.04434` and `Mohsenvand` for THIS one.
> **Classification:** Public (riir-infer).

---

## TL;DR

**Sketchpad** certifies a *trained* artifact exact for ALL input lengths by FINITE checking: pin each step's input to a bounded view of external memory (a 3-cell sheet view), enumerate the reachable view set (2,446 rows), and check margins on the deployment device — four Lean 4 theorems (Mathlib, axiom-clean) prove (T1) exactness-on-views ⟺ exactness-at-every-length, (T2) the bound is NECESSARY — with history-reading steps NO finite test suffices (explicit counterexample decoder passing any finite suite, failing at 2M+4), (T3) margins ≥ 2τ certify under bounded logit perturbation — so the certificate carries across precision/batching/hardware, (T4) composition error decomposition (wrong table / wrong transcription / wrong combining program). The artifact: rank-16 LoRA (10.2M, 1.2%) on Qwen3.5-0.8B — **our league checkpoint family, including the 6-full-attn + 18-GDN hybrid structure our deltanet lane serves** — trained on the view tables with a margin-hinge loop (hinge 4, replay < 2, stop ≥ 3; **19.8 min on one H100**); a scalar gate g multiplies every adapter branch: **g=0 is bit-for-bit the base model** (verified on logits/hidden/cached states), g=1 reproduces all margins. Honest negatives recorded: the dialogue adapter lost 5.8 ± 1.7 GSM8K on fresh items (outside its pre-registered band); base model passes 12/2,446 views zero-shot, 158/2,380 with 8 in-context examples.

**WeightCode** quantifies the two claims our transfer/library records lacked numbers for: **weight programs (executable weight-writing code, not fixed edits) re-executed on 5 independently trained ViTs retain 83–110% of source gains, while copying their fixed weight edits yields negligible gains** — the program is the carrier, the edit bytes are not; and **programs accumulate** on an already-edited network (+5.23–14.50 contrast robustness while retaining the installed noise gain). Authoring is the WSP class we hold saturated (katgpt-rs 648 is the parent both cite); hosts are ViT/ImageNet + a toy command net — vision, not our serving scope; mechanism-level transfer only.

**Distilled for this stack:** the import is a **proof-path class**, not a mechanism: (F1) Issue 1256's implant exactness surface is certifiable by view enumeration — the fire gate's read table is a bounded set, so G2's "exact on fired spans" sampling upgrades to exhaustive enumeration with margins, while the model's downstream recurrence stays sample-evaluated (T2 law — the SPLIT is the content); (F2) the margin contract (≥ 2τ on the deployment path) is the portable cross-hardware form of G1's same-box+same-GPU scope; (F3) trained-then-certified joins closed-form/Handwired as the third construction route, riding the existing 1256 G-bars; (F4) 1256 T2's cell gains the program-vs-edit reference arm with WeightCode's 83–110%-vs-negligible expectation. Verdict: **Gain** (both), no Super-GOAT (importing a published class; our compile-side is saturated), no new plan (consumer-gated trigger, same shape as riir-train 686).

---

## 1. Paper core findings (numbers of record)

### 1.1 Sketchpad (arXiv:2611.04434)

| Finding | Datum |
|---|---|
| T1 exactness reduction | exact on every run of every length ⟺ exact on the finite reachable view set (Moore's machine-identification + regeneration; paint injectivity needed only for the converse) |
| T2 necessity | a 2-state lookup + one attention head passes every finite test suite (and every input up to length N; `no_finite_check`: exact to N fails at N+2) yet errs at 2M+4 — **no finite test certifies when steps read history** |
| T3 margin robustness | margin m(v) > 2τ ⟹ exact under any logit perturbation ≤ τ; deployment certificate on M4/WebGPU fp16: min margin 3.02 (batch τ measured 1.09; batching samples compositions, does not bound all); 4-bit decoder moved margins +9.4 → −8.9 (the class T3 excludes) |
| T4 composition | wrong answer ⟹ wrong table ∨ wrong transcription ∨ wrong combining program; certified tables remove the first term |
| Tables | SUM 242 · MUL 1,100 · ADD 220 · ADZ 220 · FLS 2 · NXT 11 · SUB 242 · EST 400 · DEC 9 = **2,446 views**; glyphs finite (32×32 sprites); prompt = op token + 3 image tokens + `=` + 2 placeholders, nothing else |
| Training | rank-16 LoRA, 150 matrices (24 attn + 54 GDN + 72 FFN), 10.2M params (1.2%); CE + margin hinge (h=4) + LwF distill penalty (λ=1, 128-tok plain text); full certificate every 100 steps, replay margin < 2, stop ≥ 3 — **passed every row at step 1,200, stopped 1,600, 19.8 min on one H100**; plain-text PPL ×1.0048 |
| Dormancy gate | y = xW + g·s·(xAᵀ)Bᵀ; g=0 ⟹ bit-for-bit base (logits/hidden/cached, 2,500 chat passes + 2,688 views, ONNX CPU); g set by the RUNTIME, not the model; within-reply switching inherits base recall 0.809 |
| Zero-shot floor | base model: 12/2,446 views (all trivial single-digit SUM copies); +8 in-context solved examples: 158/2,380 — **the tables are learned, not elicited** |
| Comparison | 220/220 pre-registered (sums ≤ 1,000 digits, products ≤ 20 digits); GPT-6.1 Sol 217; Qwen3.5-0.8B alone 23; silent errors on 300/1,000-digit sums are the frontier failure mode |
| Dialogue adapter | invocation precision 0.994 / recall 0.989 held-out; fresh-item rescore (pre-registered): GSM8K **−5.8 ± 1.7 (band miss — honest negative)**, MMLU −2.0 ± 1.7 (held); the loss sits in multi-step reasoning, not the sheet |
| Mechanism | logit-lens decision depth median 19–20 (the 5th full-attention layer); class-mean intervention flips 79–80/80 steps after depth 20, 0 up to 15 |

### 1.2 WeightCode (arXiv:2611.25476)

| Finding | Datum |
|---|---|
| Authoring | frozen LLM writes executable weight programs (read/construct/modify weights), task knowledge + existing weights, **no GD**; grammar or 8 demonstrations suffice |
| Composition install | unseen command compositions 0% → 99.8% exact match; new pickup action 0% → 98.4%; original command/navigation preserved |
| ImageNet | 4 programs, net gains 8.84–17.62 pts, clean accuracy preserved; beats the best of 8 fine-tuning methods under matched examples/checks/retention |
| **Transfer (the load-bearing row)** | **10 programs re-executed on 5 independently trained ViTs, no further LM calls: 83–110% of source gains retained; copying the programs' FIXED weight edits: negligible gains** |
| **Accumulation** | 6 programs on an already-edited network: +5.23–14.50 contrast robustness, previous noise gain retained |
| Attribution | causal interventions: installed noise-correction gates drive robustness; selective activation preserves clean accuracy |
| Hosts | ViT/ImageNet + a toy command/navigation net — vision; the LLM-host space is WSM's (Research 045) |

---

## 2. Internal prior art — signal-diffs (grep-verified 2026-11-26; instrument live-checked: `certify|certificate|exactness` sweeps hit real code, listed below)

| Sketchpad/WeightCode property | Our closest analog | Where | Delta (signal-diff) |
|---|---|---|---|
| **Universal exactness certificate (finite views → all lengths, Lean)** | `sst_certified` (katgpt-rs Issue 958): `certify_decoded_tail` — a certificate for tail reconstruction with **RELATIVE tolerance band [1e-5, 1]** | katgpt-core/src/sst_certified.rs | Different claim class: reconstruction-tolerance certificate on ONE decode vs behavioral-exactness certificate over an enumerated input set with an all-lengths reduction. No internal analog for the latter |
| **Per-step closed-form certificate** | `refinement_bridge.rs` — "terminal-mass certificate", "each later step approximate with a closed-form certificate" | katgpt-rs/src/refinement_bridge.rs | Certificates bound ERROR MASS of approximate steps; Sketchpad's certify ARGMAX EQUALITY (margins). Both are margin-shaped, but the bridge certifies its own algebra, not a trained network's behavior |
| **Deployment-path parity monitoring** | `drift_guard` (default-ON): fixed-stride 1/256 sampling through the exact twin, sigmoid threshold, per-shape-bucket demote | riir-infer (2026-11-23 flip) | Detection vs certification: drift_guard SAMPLES the serving posture and demotes on drift; Sketchpad ENUMERATES the full reachable input set pre-deployment with margins and proves universality. Complementary, not overlapping |
| **Dormancy of a runtime artifact** | `latent_steering_bridge` G1 (bit-identical at zero steering weight, riir-ai Plan 391); flag-off law (compile-time byte-identity); `closed_form_write` demote/reload identity probes | shipped lanes | Same CONTRACT (silent ≡ base), different artifact: our lanes gate latent directions / compile-time features; Sketchpad's gate makes a TRAINED adapter bit-silent via scalar-zero branches — the trained-adapter twin. Modality-gated adapters (their ref [101], arXiv:2609.26182) publish the same exact-equality test — class has published prior art, import is the gate SHAPE for any future trained implant |
| **Exact arithmetic in/near the forward pass** | Handwired 621 (exact by construction, bit-exact), 665 emitter, closed_form_write (closed-form writes, i1189 cell), Qwenputer 018 (compiled RISC-V, exact by construction) | saturated compile-side | Sketchpad is the TRAINED-then-CERTIFIED complement: exactness achieved by training against a certificate-driven objective, then proven. Construction route differs; proof obligation identical (exactness) |
| **Certificate-bound artifacts** | GraftSlot A2 monotonic mount + BLAKE3 digest pins; `mount` refuses base-digest mismatch (closed_form_write L107–117); envelope hash law | shipped | Sketchpad's certificate binds model-file hashes + device + library version + per-view margins in fixed order, rechecks the weakest tenth per new device — an EXTENDED certificate envelope (behavioral margins, not just artifact digests). Vocabulary overlaps, content extends |
| **Adapter placement on the qwen35 hybrid** | our deltanet lane serves the same 6-full + 18-GDN structure; laya/CUDA/Metal adapters; rank-16 is our house LoRA rank | riir-infer | Their 150-matrix placement table (attn q/k/v/o ×6, GDN in/out ×18, FFN ×3 ×24) is a placement map for OUR checkpoint family — context, not a gap |
| **Program-vs-edit transfer (WeightCode)** | our transfer records move KEYS/edits: 668 Tree-Ising (one reference-free edit transfer datum), 044 Plato's Key (ends re-solve seam — does not ship), riir-train 424–427 (unsupervised direction transfer CLOSED NEGATIVE) | 045 §2 row 6 | Nobody internally quantified program-re-execution retention vs copied-edit retention. WeightCode's 83–110%-vs-negligible is the reference datum for 1256 T2's cell design (transfer the builder, re-solve the ends — the design 045 already proposes, now with an external number to expect) |
| **Accumulation/library** | 1256 T1's composable library (proposed, unbuilt); WSM's 23-implant co-mount | 045 §2 row 5 | First quantified accumulation evidence (programs on an already-edited net retaining prior gains). Evidence for the axis, not a mechanism we lack |

**Internal zero-hit claim (documented per the negative-coverage rule):** "no internal universal-behavioral-exactness certificate ships" — greps run: `certify|certificate|exactness|view set|bounded view` over katgpt-rs + riir-infer src (18 hits, every hit read: sst_certified relative-tolerance, refinement_bridge closed-form error certs, fa_minimize automaton exactness (code-level algebra, not a network), readout_screen NNLS exactness (fixture algebra), deltanet/tokengate exactness laws (fold-order/storage invariants), adapter geometry test tolerance — ALL are code-algebra or integrity claims; none certifies a fitted artifact's behavioral outputs over an enumerated input set). The Lean .proofs instances prove OUR code's invariants, not trained-model behavior. Claim stands.

---

## 3. External prior art (§4 searches run 2026-11-26)

- Certification lineage is the paper's OWN related-work section, independently confirmed by search: Moore 1956 (machine identification), Chow 1978 (FSM testing), Cai/Shin/Song ICLR 2017 (recursion verification), **Lewandowski/Machado/Schuurmans arXiv:2601.08061** (certified universal computation — checks all 1,857 rules of a universal system by exact match; Sketchpad specializes it and adds the margin/perturbation/device layers), Gross et al. NeurIPS 2024 (compact mechanistic proofs), Back de Luca & Fountoulakis arXiv:2605.22964 (certification-from-examples hardness), Yang et al. (length-generalization bounds, ICML 2026). No competing "certified exact arithmetic on a deployed pretrained LLM via view enumeration" work found beyond this lineage — the note imports the class, never claims to originate it.
- WeightCode: confirmed the concurrent WSP framework (our 648 is the class parent it builds on); the program-retention quantification is its own contribution.
- Related-class priors worth recording: **Activated LoRA** (arXiv:2509.03864-family, their ref [100]; adapter active only post-invocation, base cache reused) and **modality-gated adapters** (arXiv:2609.26182, exact-equality preservation tests) — the gate-dormancy class has published members; inductive scratchpad (arXiv:2410.02696-family, ref [41]) + PENCIL (arXiv:2503.14337, ref [42]) are the published "bound what a step reads" context disciplines T1 sits at the limit of.

---

## 4. Path 0 — component inventory

| Component | Track | Coverage / extraction | Disposition |
|---|---|---|---|
| T1 finite-view reduction (Lean, axiom-clean) | modelless math | **not covered** internally; extraction = the theorem itself (importable as a proof obligation template) | **note (this)** + Issue 1256 proof-path upgrade |
| T2 necessity counterexample (no finite test with history reads) | modelless math | **not covered**; constrains EXACTNESS claims only (quality claims on suites unaffected) | note — eval-law row |
| T3 margin contract (≥ 2τ, deployment-path, batch-τ measured) | modelless math | drift_guard samples (detection); margin form **uncovered** | note + 1256 G1 portable form |
| T4 composition decomposition | modelless math | **uncovered**; relevant to multi-implant G3 | note + 1256 G3 note |
| View-enumeration certificate protocol (device-bound, hash-pinned, weakest-tenth recheck) | modelless protocol | mount digest law partial (hash pinning only) | 1256 proof path |
| Margin-hinge certified-adapter training (hinge 4 / replay <2 / stop ≥3 / LwF; 19.8 min H100) | **training (c)** | riir-train owns the domain; no consumer lane today → **design input + trigger, NO plan now** (the named tension — verdict adjudicates) | note §5 F3 + 1256 route row |
| Scalar gate g dormancy | trained-adapter runtime | latent_steering G1 = same contract, latent-direction artifact; gate SHAPE importable for any trained implant | note + 1256 gate-shape note |
| Sheet/external-memory runtime (pinned views, runtime moves view, writes no digit) | runtime architecture | **uncovered** — our exactness lanes are in-weights (Qwenputer compiles IN; Sketchpad reads rendered OUT) | context only; no lane (a sheet lane would need a VLM host + browser runtime — out of serving scope) |
| Dialogue adapter + honest negative | track-c evidence | our DORMANT-pattern caution has a measured external twin | note |
| WeightCode program-retention + accumulation | modelless decision content | **uncovered** (nobody quantified program-vs-edit retention internally) | note + 1256 T2 reference arm |
| WeightCode authoring pipeline (grammar/8-demos → program) | WSP class | **SATURATED** (648/678/665/677/621) | signal-diff row only |

---

## 5. Fusion — what these × our stack produce that neither has alone

- **F1 — The implant-lane certificate (the primary import).** Issue 1256 T1's exactness surface SPLITS: the implant's own behavior (fire gate over its bounded token-id read table → silent-or-write) IS certifiable Sketchpad-style — enumerate the reachable (window → decision) views, record margins on the deployment path, T1 gives all-lengths universality for the gate+write contract — while the model's downstream behavior on forced tokens remains recurrence-over-history and stays sample-evaluated FOREVER (T2: no finite test certifies it). G2's "exact match on every span where it fires" upgrades from sampling to enumeration; G1b's false-fire bar becomes the table's silence rows (look-alike views must carry margin ≥ 2τ toward NO-fire). The certificate rides the existing mount law (hash-pinned artifact + per-view margin table, weakest-tenth recheck on new devices).
- **F2 — The margin contract as G1's portable form.** 1256's G1 currently scopes silent identity to same-box + same-GPU (bf16 greedy decode differs across GPUs — the WSM disclosure). Sketchpad's T3 is the alternative contract: margins ≥ 2τ certify across any perturbation ≤ τ, with τ MEASURED per posture (batch composition, precision). A future implant cell can quote margins instead of cross-hardware bit-identity — looser but sound, and portable.
- **F3 — Trained-then-certified as the track-c construction route.** Closed-form (closed_form_write) and agent-compiled (Handwired/665) are our two exact construction routes; Sketchpad adds the third: train against the certificate (margin-hinge loop, replay-on-low-margin, stop-when-certifiable) — 19.8 min on one H100 ≈ a 4090 cell. Filed as a ROUTE inside 1256 (riding its existing G-bars + the F1 certificate as the proof path) with the trigger = the lane's consumer cell passing and a track-c arm being wanted; NOT a standalone riir-train plan — the consumer-gate discipline (1256's stop rule) is the leash, same shape as 686's dormancy. Recipe + cost + bars recorded here so the future plan is pre-registered cheaply.
- **F4 — T2's transfer design gains its reference arm.** 1256 T2's cell (core built on 0.8B, ends re-solved on a sibling) adds the WeightCode contrast: re-EXECUTE the builder/program on the sibling vs transplant the fixed edit bytes — the external expectation is 83–110% vs negligible. If our cell reproduces the contrast, the refit-ends design is externally corroborated; if not, the ViT↔LLM host difference is the named confound.
- **F5 — The T2 eval-law lands on our recurrence lanes.** Any lane claiming EXACTNESS for behavior that reads its own history (looped lanes' sample-based evals, specdec acceptance, multi-implant composition) must phrase the claim as quality-on-samples, never exactness — there exist passing-on-any-finite-suite decoders that fail at unbounded length (and length-bounded testing doesn't rescue: exact-to-N fails at N+2). Quality claims are untouched. This is a claim-hygiene law, not new machinery.

---

## 6. Verdict

**Per-track separation (no cascade):**

| Track | Verdict | Tier | Files |
|---|---|---|---|
| Sketchpad (a) modelless — theorems + certificate protocol as proof-path class | **Gain** — uncovered proof-path class, actionable on 1256's G-bars | GOAT-candidate (pending any cell; no exactness cell exists yet) | this note + Issue 1256 amendments |
| Sketchpad (c) training — margin-hinge certified adapter | **Design input with trigger** — no consumer lane; 19.8-min cost recorded; NOT a plan now (named tension, adjudicated) | — | this note §5 F3 + 1256 route row |
| WeightCode (a) modelless — program-retention + accumulation decision content | **Gain (evidence-grade)** — reference arm for T2, evidence for the library axis; no new primitive (WSP class saturated) | — | this note + 1256 T2 row |

**Tiers:** Super-GOAT **NO** (Q1 fails — the class is the papers' own published contribution with a deep lineage; we import). GOAT **not yet** (no cell). Ungoat discipline: every actionable row above names its proof path (F1's enumeration protocol, F2's measured-τ contract, F4's two-arm contrast); no GOAT claim without a run.

**MOAT gate (riir-infer):** fits — inference-substrate proof-path + lane design input on our league checkpoint family, public repo, upstream-clean. The Lean-proof obligation, if we ever formalize F1 for our own implant, lands in the riir-infer-local proof convention (the certificate JSON + margins are the artifact; the reduction theorem is cited, not re-proven — Sketchpad's Lean files are Apache-2.0 and quotable).

**Named weak points (pre-registered for the verdict):**
1. **The Path 0.5 tension** — the margin-hinge training recipe is applicable-but-consumerless; I filed design-input-with-trigger instead of a riir-train plan, betting on the stop-rule discipline. The counter-argument: "never defer GPU-training tasks" + the recipe is CHEAP (19.8 min H100). If the reviewer rules a plan, it opens as a small riir-train plan with the F1 certificate as its GOAT gate.
2. **F1's boundary choice** — I split certifiable-implant-surface from uncertifiable-downstream-recurrence. The alternative reading: forced-token writes change the KV the model later attends, so even the "silent" contract is only silent until first fire — that's fine (G1 tests the all-silent posture), but the post-fire composition claims must stay sample-evaluated. If the reviewer reads the split as blurrier, F1's wording narrows.
3. **WeightCode is vision-hosted** — mechanism-level transfer only; no LLM-host retention numbers exist.
4. **One note for two papers** — both arXiv IDs + author names are in the header for grep findability; if the reviewer prefers split notes, 046 stays and 047 spawns as the WeightCode half with a pointer.

---

## References

- Sketchpad — arXiv:2611.04434 (Mohsenvand 2026; Apache-2.0 code + Lean proofs; WSM ref [31])
- WeightCode — arXiv:2611.25476 (Li, Cui, Zhang et al. 2026; WSM ref [32])
- Lewandowski, Machado & Schuurmans — arXiv:2601.08061 (the certified-decoding prior Sketchpad specializes)
- Modality-Gated Deep Adapters — arXiv:2609.26182 (the gate exact-equality class)
- Research 045 §3 rows updated to `distilled: riir-infer 046` on landing.
