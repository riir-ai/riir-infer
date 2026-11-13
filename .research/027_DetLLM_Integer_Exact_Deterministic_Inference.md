# Research 027: DetLLM — Integer-Exact Deterministic Inference

> **Source:** nathanrs/detllm — "Fully deterministic cross-device LLMs inference" @ `9c3f1c9cdd42294087cc0507992c21dfde944bea` (GitHub, **NO LICENSE file — decision-content-only transcription posture, outbound-PR FAIL-CLOSED**)
> **Date:** 2026-11-13
> **Status:** PROMOTED — the G1 pass landed (2026-11-14; all four arms green, zero tolerance — bench 1213); the fusion section §3A is now [`.plans/632_i1214_int_ref_self_verification_guide.md`](../.plans/632_i1214_int_ref_self_verification_guide.md) (the issue's status-after-G1 rule); §3B consumers stay with their owning repos (plan 632 T5)
> **Related Research:** 024 (LeapQuant GDN state quant — static storage scales), 002 (e8 int8 tables)
> **Classification:** Public (riir-infer note; mechanism is public upstream)

---

## TL;DR

detllm runs Qwen3-0.6B **end-to-end in exact integer arithmetic** — int8 dyadic-scale weights/activations, integer softmax/RMSNorm/SwiGLU/exp/RoPE, no float anywhere between input ids and int32 logits. Because integer addition is exactly associative and commutative, bit-identity across runs, batch compositions, prefill/decode splits and hardware vendors becomes **a property of the arithmetic**, proven by a zero-tolerance 9-machine demo: one int8 logits hash (A100/H100/EPYC/Xeon/M1 Pro, batch 1/8, split prefill) vs **nine distinct fp16 hashes**. Our determinism today is **discipline** (katgpt-rs Issue 940 fold-order pins, `candle_vec_sum` order-mirroring, G5 tolerance parity — the laya-parity gate defined in riir-reflex `AGENTS.md`); detllm shows it can be **arithmetic**.

**Answer to the owner's lane question ("int lane for modelless too?"):** it is a **model lane, not modelless** — it runs trained weights; the reflex modelless engine is already integer-deterministic by construction (BLAKE3 hashing). But it slots into the lane family beside `laya`'s cpu/metal/cuda/ane backends as an **integer-exact reference lane** whose value is **self-verification** (bit-identical to itself on every rig; cross-rig byte-identity receipts), not throughput — it is not an oracle for the float lanes.

---

## 1. Source Findings

**Why float forks:** FP addition is non-associative; high-performance kernels pick reduction order by throughput, and that order shifts with batch composition, prefill-vs-decode GEMM shapes, kernel heuristics (split-K, atomics), and silicon. "Deterministic mode" flags pin run-to-run order **on one machine at one shape** — not batch-invariant, not cross-hardware. Int8 *storage* is not int8 *arithmetic* (SmoothQuant/OmniQuant keep float non-linear ops — the I-LLM arXiv:2405.17849 observation).

**The stack** (all quotes at pinned sha):

- `intmath.py` — THE single semantics module: "Rounding policy (frozen after milestone 1)": `rshift_round(a,k) = (a + (1<<(k-1))) >> k` (ties toward +inf) for all power-of-two rescales; `round_half_away_div` only where a non-power-of-two divisor appears (softmax normalization). Plus `ilog2_floor`, bitwise `isqrt`.
- `dyadic.py` — tensors carried as `(data, m, k)`, value ≈ `data·m/2^k`; per-token column scales for activations, per-channel for weights. `fit_dyadic_ratio` (worst-case rel error 2^-bits), `norm_scale` ("keeps k drift bounded across long op chains"), residual stream as int32 + power-of-two scale renormed into `[2^26, 2^30)`, alignment rule "shift the finer operand down to the coarser k", and `requant_i8_static` for **K/V heading into the cache** (one fused rounding at a static dyadic scale).
- `ops/di_exp.py` — integer exp by shift decomposition with the paper's constant `m_f = m + (m>>1) − (m>>4) ≈ 1.4375·m ≈ m·log2(e)`; the implicit non-dyadic output scale "is always consumed by a ratio (softmax / sigmoid), where it cancels".
- `ops/di_softmax.py` — sentinel-masked row max (no float −inf anywhere), `P_OUT_BITS = 15` default (the paper's 8-bit probs "cost ~2 PPL once scores are accurate" per their ablation), probs fed to the GEMM as an exact hi/lo int8 split.
- `guard.py` — `NoFloatMode` (TorchDispatchMode) raising `FloatLeakError` on any float tensor produced/consumed in the guarded region: "mechanically prevents 'convenient .float()' regressions".
- `backends.py` — "the ONLY backend-divergent code"; both backends compute the exact int8×int8→int32 product-sum, "bit-identical by construction (verified by §9.3 tests, not assumed)". `reference` "names a ROLE — the correctness oracle and cross-device determinism witness — not a device". `GUARD_ROWS = 64`: cuBLASLt int8 CUTLASS kernels issue speculative reads up to 16 KB past operand ends — operands need guard rows.
- `prepare.py` — offline calibration is **training-free**: SmoothQuant-style analytic factors, α=0.5.
- Acceptance spec §9.3: "Determinism (the actual point — must all pass exactly, no tolerance)" across batch composition, prefill/decode split, and devices.

**Measured (their docs/RESULTS.md, A100):** WikiText2 PPL 20.72 (int8) vs 20.95 (fp16). Decode b1 **106.3 tok/s = 3.6×** fp16 *eager* (29.3); b8 566.7 (3.6×). **Prefill int8 eager 1,057 vs fp16 48,898 tok/s** — prefill is a ~46× loss; the decode win is CUDA-graphed against an eager baseline. Honest read: this is a *verifiability* property with acceptable decode, not a throughput lane.

**The demo's sharpest datum:** despite every fp16 row disagreeing on logits from step 0, eight of nine still produced identical greedy text — until M1 Pro step 457, where a top-2 gap of **0.016 (one fp16 ulp)** tied exactly and the lowest-id tie-break forked the completion. "Every generation is one near-tie away from two 'identical' deployments quietly disagreeing."

---

## 2. Distillation — Path-0 Component Inventory

The value is the **math property** (integer associativity), not any training loop. Per-component coverage vs our substrate:

| # | Component | Shipped analog | Signal-diff | Verdict |
|---|---|---|---|---|
| 1 | Frozen integer rounding-semantics module | katgpt-rs Issue 940 determinism law + `closure/bridge_certified.rs` (pinned f32 fold order + strict tie rules) | Ours pins float ORDER per surface; detllm makes order irrelevant **between int runs**. No integer-forward semantics module exists | **extract** |
| 2 | Dyadic `(data,m,k)` per-token dynamic rescale THROUGH the forward | `q8_artifact.rs`/`blocked_artifact.rs`, EXL3 trellis, e8 tables (research 002) | Those are STATIC storage scales with float compute; detllm rescales per token during integer compute | **extract** |
| 3 | DI-Exp shift decomposition + ratio-cancellation of the non-dyadic implicit scale | none | Our forwards are float (Metal `relaxed=false` twin is still float). Integer sigmoid = exp/(exp+1) fixed-point via the same ratio trick — a direct cousin of the sigmoid law | **extract** |
| 4 | DI-Softmax p_out=15 hi/lo int8 split feeding the GEMM | none | — | **extract** |
| 5 | DI-RMSNorm / DI-SwiGLU / int RoPE / bitwise isqrt | none | — | **extract** |
| 6 | NoFloatMode runtime guard | none mechanical | Rust equivalent is TYPE-LEVEL (a module with no `f32` in scope) — compile-time by construction, stronger than dispatch-mode | **extract as pattern** |
| 7 | Zero-tolerance cross-device acceptance suite | Retention cells (byte-identical **per box**); G5 parity (tolerance, p-drift ≤ 1e-3) | Per-box/per-posture vs cross-vendor; tolerance vs exact | **extract as gate class** |
| 8 | Exact CPU reference as oracle ROLE | `drift_guard.rs` exact twin (`crates/riir-infer-gpu/src/drift_guard.rs`) — same-MSL `relaxed=false`, same platform | detllm's `reference` role certifies the INTEGER lane across configurations; drift_guard's twin checks float-vs-float. Transfer = the ROLE pattern for the int lane. ⛔ NOT a drift_guard oracle seat (round-1 verdict: int-vs-float is ~1e-2-class per GEMM — above drift_guard's `DIV_CENTER = 3e-3`; a different numerical model, not an exact float twin) | **extract as ROLE pattern, never as a drift_guard consumer** |
| 9 | Training-free analytic calibration (SmoothQuant α=0.5) | `fake_quant.rs` + q8/q4 converters (LAYA_WEIGHTS_VARIANT=q8 adopted) | Our q8 lane is fake-quant: int storage, float compute; detllm converts offline into a true-integer artifact | **extract** |

Workspace prior-art check (corrected — the first sweep used `rg`, which is NOT installed on this shell and silently returned nothing; re-run with `grep -r`): the closest prior distill is **riir-refine Research 116** — source verification of **arXiv:2609.00363** "Deterministic LLM Inference Across GPU Kernels: Power-of-Two INT8 Quantization Scales and the Limits of Tolerance-Based Conformance", whose lever **riir-ai Issue 845** is **RESOLVED-SCOPED — PASSED** (riir-ai HISTORY.md row 1604: scale-site fold invariance 0 diffs on 3 Bonsai shapes, non-vacuous native control). Signal-diff vs detllm: 2609.00363/845 snap **weight-side quant scales to PoT** on an otherwise-float pipeline — determinism was EMPIRICAL (196/196 bitwise measured, per-token activation scales stay non-PoT; float softmax/norm remain); detllm is **whole-pipeline integer compute** — activations, dyadic rescale-through-the-forward, integer non-linear ops — determinism BY CONSTRUCTION. detllm completes what 2609.00363 started; our fleet's own 845 gate ("scale-site fold invariance, non-vacuous control") is the same gate discipline this lane needs at forward granularity. Other grep hits are vocabulary cousins, not mechanisms: 569 (Hadamard dyadic-exactness anchors), 007 (rank-1 dyadic update — LA algebra), 377 EGGROLL (integer-only TRAINING via ES — training-side), 026 (per-box bit-identical parity, already cited).

**Published prior art (§4 sweep):** I-LLM (arXiv:2405.17849) and I-BERT/I-ViT — integer-only for *efficiency on edge*, no determinism claims; Thinking Machines "Defeating Nondeterminism in LLM Inference" + SGLang/ICML deterministic-TP work — batch-invariant FLOAT kernels with pinned order (the discipline approach detllm rejects as symptom-treatment); arXiv:2609.00363 — PoT weights-only scale snapping (the workspace's own 116/845 record above). detllm's delta is unconditional cross-device determinism as acceptance criteria over the WHOLE pipeline. Our delta if ported: upstream-clean Rust substrate port + the self-verifying lane integration — integration novelty only, and we credit the 2609.00363/845 lineage as the in-fleet predecessor.

---

## 3. Fusion

**A (P0, this repo) — the integer-exact reference lane.** An opt-in `int_ref` backend beside `cuda.rs`/`metal.rs`: offline analytic calibration of one of our checkpoints into an int artifact, a Rust integer forward (type-level no-float by construction). **What it is: a separately calibrated, self-verifying MODEL — bit-identical to itself on every rig — not an oracle for the float lanes.** Consumers: (a) **cross-rig byte-identity receipts** (the headline) — one logits-hash line that means the same thing on amo, bmo and the 4090 (today "byte-identical" is a per-box claim); (b) **its own parity is exact** — int-vs-int reference-path parity is bit-exact, while the float lanes' G5 laya-parity (defined in riir-reflex `AGENTS.md` §GOAT gates) stays a tolerance check — the int lane does NOT upgrade it. ⛔ **Categorical limit (round-1 verdict, binding):** int-vs-float is a different numerical model, ~1e-2-class per-GEMM error — above drift_guard's `DIV_CENTER = 3e-3` — so the lane can neither sit in drift_guard's oracle seat nor certify float-lane outputs. It certifies itself.

**Ternary skip (round-1 verdict catch):** ternary weights {−1,0,+1} are already integer — the dyadic treatment reduces to the activation side (norms, softmax, accumulation, RoPE) and the weight-calibration error source vanishes entirely. The natural second G1 target once a small ternary checkpoint is minted; the 27B league model is too large for an eager-class reference lane.

**B (second step, private consumers) — verifiability propagates.** Quorum-committed decisions and frozen-brain NPC inference need cross-node bit-identical cognition: the chain side already proves commit-path determinism; an integer-exact forward lets the COMPUTE side carry the same property. Framing (round-1 verdict): those consumers would **serve and verify the INT MODEL's outputs** — a separately calibrated model whose decisions any node can verify without trusting it — never the fp lanes' outputs. Sequenced AFTER P0 proves G1 bit-exactness on the Rust port — filed as consumer-side work by the owning repos then, not speculative wiring now.

**One-liner:** di_exp's ratio-cancellation yields an exact integer sigmoid (fixed-point exp/(exp+1)) if any sync-crossing latent gate ever needs cross-rig exactness.

---

## 4. Verdict

**Tier: GAIN** — mechanism-distill, new primitive class for the substrate, with the in-fleet predecessor named (refine 116 / ai 845 — scale-snapping, the step below full-pipeline integer). Not scored Super-GOAT yet: the capability proof is upstream PyTorch on their artifact and their demo; per the defend-wrong discipline a capability claim on OUR stack needs the PoC. The P0 lane therefore files as an issue (`.issues/1214`) with a pre-registered proof path — zero-tolerance determinism suite on the Rust port — and the plan/guide materialize on G1 pass. Honest scope note: the lane enters as **self-verification** (int-vs-int), never a serving-default claim — detllm's 3.6× is vs fp16 *eager* and int8 prefill is ~46× slower than fp16; our optimized float lanes (ktile64 tensor stack, resident decode) would win any league bout on throughput. **And its exactness is int-vs-int only:** it is a second, separately calibrated model whose outputs any rig can verify — it neither checks nor upgrades the float lanes' numerics (float G5 stays tolerance-based; drift_guard keeps its same-platform float twin).

**MOAT gate:** riir-infer = public substrate moat (quant formats, kernels) ✓ in scope; upstream-clean Rust port, zero new deps ✓. Numbered-docs bump rides the allocation commit.

**License hazard (binding):** the repo carries **no LICENSE** (tree + pyproject checked) → decision-content-only transcription posture: no vendoring, no outbound PR (fail-closed), port from the documented algorithms and spec text with quote-pinned provenance. Do not depend on their HF artifact either — calibrate OUR checkpoints.

---

## 5. Provenance

- Clone: `nathanrs/detllm` @ `9c3f1c9cdd42294087cc0507992c21dfde944bea`, shallow, `.raw/detllm/` (removed after landing).
- Quotes: `src/detllm/intmath.py:7-17`, `dyadic.py:3-6,166-205`, `ops/di_exp.py:2-11`, `ops/di_softmax.py:8-14,28-30`, `guard.py:1-5`, `backends.py:1-15,30-34`, `prepare.py:267-288,342-359`, `docs/SPEC.md` headings, `docs/RESULTS.md:111-136`. License: absent from tree and `pyproject.toml`.
- Cousins verified live: `crates/riir-infer-gpu/src/drift_guard.rs` exists (with `DIV_CENTER = 3e-3` — the round-1 verdict's categorical-limit datum); `dot_f32_ordered` HAS 23 tracked code hits across katgpt-rs (definition home `katgpt-types/src/simd/dot.rs`; consumers in katgpt-spectral, katgpt-personality, katgpt-hla, katgpt-micro-belief — a correction of this note's own first claim, which a broken `rg` sweep voided); the katgpt-rs Issue-940 law's canonical precedent is `katgpt-core/src/closure/bridge_certified.rs` + `.docs/10_audits/float_reduction_order_tie_band_determinism_audit.md`. G5 = the laya-parity gate defined in riir-reflex `AGENTS.md` §GOAT gates (top-1 ≥ 99.9% + p-drift ≤ 1e-3). Qwen3.5 is GDN-hybrid — research 018: "our league runs qwen3.8 GDN hybrids"; the only Qwen3.5-0.8B on disk is a Q8_0 GGUF (pre-quantized — unusable as a calibration parent).
- Tooling lesson (this session, recorded because it corrupted a first draft): `rg` is not installed on this shell — every `rg`-based sweep above silently returned nothing until re-run with `grep -r`. Any future prior-art claim in this repo must name the grep binary that produced it.
- Round 1 (Claude verdict `65a5289f-f5a6-4f78-bfa1-9f8478eab987`, REVISE→fixed): drift_guard-oracle consumer removed as a category error; consumer (3) restated to own-parity-only; accuracy bar pre-registered with named checkpoint; argmax tie rule + test arm added to G1; ternary-skip fusion added; cross-repo citations repo-qualified. Round 2 (same session, REVISE→fixed): G1/G3 target retargeted dense (Qwen3.5 is hybrid GDN — out of scope for this port); integer GDN recurrence pre-registered as a separate tranche; bf16 baseline pinned; stale TL;DR oracle line deleted; in-fleet predecessor (refine 116 / ai 845 RESOLVED-SCOPED PASSED) added to the prior-art section.
