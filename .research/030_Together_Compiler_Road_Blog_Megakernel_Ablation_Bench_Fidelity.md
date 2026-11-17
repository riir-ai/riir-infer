# Research 30: Together Compiler Road Blog — Megakernel Ablation + Compiler-Benchmark Fidelity (TKCC Supplement, PUBLIC SKELETON)

> **Source:** "Last year we built megakernels. This year we deleted them: The road to Together Compiler" — Together AI blog, together.ai/blog/road-to-together-compiler, published 2026-11-16. POST-ONLY source: no code release, no repo — **the blog URL is the pin** (the Batch-353/363 live-fetch precedent). All numbers below quoted from the fetched page this session.
> **Date:** 2026-11-17
> **Status:** RECORD — Gain (staged leads filed in riir-refine `.distill/001`; **no pivot, no new plan, no implementation issue**). Public skeleton: this note carries the blog's findings + the corpus-facing atoms only. The league-side convergences (bench numbers, queue state, design corroboration) are **private-side** — see riir-ai R402's road-blog addendum (private repo).
> **Related:** riir-ai R402 (private; the controlling TKCC record — owns Part 3's bitter lessons + the agentic-loop falsification), riir-refine corpus `pdl-admission-by-image-set-marker` (gpu_111), `grid-dependency-wait-scoped-to-dependent-operands` (gpu_23), `spill-fragility-law-accumulator-regcap-cliff` (gpu_44), `grid-collapsing-mega-kernel-occupancy-trap` (gpu_07), `fused-serving-stack-measured-negative-ledger` (gpu_57), `adoption-trial-rotate-operands-per-launch-dram-streaming` (gpu_60)
> **Classification:** Public (riir-infer)

---

## TL;DR

The road blog is the public journey post behind the paper R402 already distilled. **Part 3 restates R402's bitter lessons unchanged** (same numbers: KB −19–31%, bare prompt 6.20× vs expert-strategy ensemble 4.92×, logging DAG up to 43% of tool calls, frequent agent comms 7.72× vs 4.82×) — nothing new there. **What the paper does NOT carry, and this note distills, is Part 1 (Torch Inductor internals) and Part 2 (the megakernel ablation)**:

- **Part 1 — Inductor leaves performance on the table:** 607 tuning flags (198 added in the past year) with non-independent effects (`aggressive_fusion` alone −6 TPS, alongside three other flags +3 TPS); the `freezing` flag buys +10–26% at large batch and **−49% at batch 1** (the FP32 weight transform is not reversed on the matvec path). Isolated-kernel benchmarking is unfaithful in three directions at once (leads 1–3 below). Greedy pairwise fusion keeps an elementwise kernel (6 CTAs) and an indexing kernel (256 CTAs) separate because they share no memory — while 126 of 132 SMs sit idle.
- **Part 2 — the megakernel was the wrong target:** CUDA Graphs on default Inductor kernels reach 87% of the megakernel (864 vs 990 TPS, Llama-1B H100); adding PDL reaches 96% (953); fine-grained PDL wait placement reaches **98% (975)**. Memory preloading — the megakernel's showcase mechanism — is worth **3.3%** of the forward. Three of five producer-consumer edges per layer need FULL barriers, so persistent-kernel overlap is mostly notional. Fixed launch envelopes cost GEMMs 1.2–1.5× (20 warps forced where 12 fit the wgmma register tiles). Splitting ops OUT of fused kernels WINS (+6.9% MoE dispatch onto copy engines vs Mixture-of-Kittens; +17% FlashInfer MegaMoE with dispatch factored out).

**Verdict for us: Gain (staged leads) — NO pivot.** The agentic-optimizer queue already exists (private riir-ai Issue 1177, via R402); the megakernel question was never open in this workspace (every prior item is conceptual/LOW); the PDL family is already mined into the corpus with measured negatives on other silicon; and our serving lanes already ship the blog's winning recipe (per-lane hand kernels + CUDA Graphs default-on + launch-count fusion + real-input measurement law — documented in this repo's public `.benchmarks/`, e.g. 873/879, and the mined corpus itself).

---

## 1. What is new vs shipped — the staged-lead table

Canonical lead scheme (used identically in this table and in the `.distill/001` addendum): **leads 1–3 are NEW rules; leads 4–5 are riders on owned families; lead 6 is a rider + one NEW face.** Every atom was deduped this session by 3+-vocabulary greps over the kernel_opt + rust_perf corpora; owners named per row; the conversion batch re-runs dedup at claim time.

| # | Atom (blog evidence) | Corpus state | Disposition |
|---|---|---|---|
| 1 | **`benchmark-input-distribution-realism`** — Inductor benchmarks `torch.embedding` with ZERO token ids → every lookup hits one row → 2.1× cache-inflated timing vs random ids; MoE grouped-GEMM latency overestimated 4–21% on random inputs vs realistic routing (GLM 5.3). The law: a kernel timed on synthetic inputs lies whenever its memory behavior is input-distribution-dependent — zeros AND randoms are both lies, in opposite directions | **No owner.** gpu_60 owns operand RESIDENCY (cached-vs-streamed trial reversal — a different axis); cold-l3 (rust_perf) owns deliberate chilling. The TKCC paper's own sibling instances (random routing activates 13–38% more experts; the top-k histogram kernel 20 µs random → 62 ms on real ReLU-zeros) are unconverted in the corpus and fold into this lead's future batch | **STAGED (lead 1)** |
| 2 | **`isolated-bench-cache-state-vs-live`** — L2-flushed isolated benches read 39–58% SLOWER than the same kernels live (block-mask slices +57.9/+52.4%, fused QKV-split +44.5%, embedding +38.8%) — cache reuse from predecessor kernels is real work a flush prices as cold | **No owner** for the selection-bench face (gpu_60 is the adoption-trial face; cold-l3 is the deliberate-cold pole) | **STAGED (lead 2)** |
| 3 | **`harness-overhead-contaminates-us-kernels`** — Inductor's bench reported 2.5 µs for an op doing NO GPU work; other readings ~3.6 µs over Nsight's GPU times (CPU launch overhead counted after a flush) → wrong kernel picked (Triton 7.4 µs chosen over live-cuBLAS 6.4 µs, −15%). In-stack instance: katgpt-rs `timed_region_guard_gate` (public AGENTS.md: 7 of 34 timed regions satisfied by ABSENT work) — same disease, our gate is the loud-zero defence | **No corpus owner** for the launch-overhead-in-the-timer face (the measurement-honesty family lacks it) | **STAGED (lead 3, thin)** |
| 4 | **PDL sync-cost table + megakernel parity** — inter-kernel cost (ns, H100/B200): `cudaLaunchKernelEx` 2077/2496, CUDA Graph 1051/915, PDL 731/814, **Graph+PDL 645/642**, `grid.sync()` 1043/1281, **megakernel grid sync 646/795** — the decomposition datum: a persistent kernel's sync advantage over graphs+PDL is ≈0. Plus: PDL wait placed just before the first read of predecessor output recovers ~98% of paged shared-memory preloading (itself worth 3.3%); loader-warp spinning in the megakernel ADDS 1.3 µs (SM queue contention); the downproj→QKV boundary is bandwidth-saturated so early weight loads are a wash. Full-barrier census: 3 of 5 producer-consumer edges per layer FULL, only the attention edge fine-grained | **Owned family:** gpu_111 (admission: image-set marker + size gate + workload class; the vLLM #59737 rider is source-reported public content — router-chain −9.12%, pure-prefill TTFT regresses 0.54–0.49% at C16/C32, GB200), gpu_23 (wait placement law), gpu_18 (sm≥90 capability probe), gpu_57 (measured negative on a B300 hybrid stack) | **RIDER (lead 4)** on gpu_111 (absolute-ns table + parity datum) + a quantified instance for gpu_23. No new rule |
| 5 | **`split-from-fused-dispatch-positives`** — K3 expert dispatch on the copy engine beats Mixture-of-Kittens' MoE megakernel by 6.9% under imbalanced routing; FlashInfer MegaMoE +17% with dispatch factored out of the fused kernel — the POSITIVE pole beside the corpus's owned negatives (stripe-sms rider's 9–49% whole-stage-fusion loss; gpu_57's ledger; gpu_07's occupancy trap) | Negative pole owned; positive split instances unowned | **RIDER (lead 5)** at the mega-kernel trap / measured-negative families |
| 6 | **`fixed-launch-envelope-costs-gemm` + `instruction-cache-thrash-whole-model-binary`** — one 20-warp envelope (16 compute + 4 control) forced onto GEMMs wanting 12 warps: wgmma accumulates in registers, best tiles stop fitting above 12 → **GEMMs 1.2–1.5× slower** (Hopper; Blackwell tcgen05 exempt via tensor-memory accumulation; AMD same wall). Plus the icache face: a whole-model binary exceeds SM icache → evict + refetch on op repeat | gpu_44 `spill-fragility-law-accumulator-regcap-cliff` owns the register-envelope cliff with four measured sm_89 cells — same law, different silicon; the **icache face is unowned** | **RIDER (lead 6)** on gpu_44 (envelope datum) + the icache face is the lead's NEW half |

Not staged (recorded here only): Kimi K3 cta_group mixing (PTX forbids single-CTA KDA + CTA-pair MLA in one kernel → a megakernel is structurally incapable of running both at their best config) + TP-MoE/ADP cluster-size forcing −5–19% — sm_10x-class, no league relevance on current hardware (sm_89 + Metal).

---

## 2. PDL — why there is no implementation queue

The mined corpus family already carries the admission law (gpu_111: image-set marker + size gate + workload class, fail-closed to plain launch), the in-kernel wait placement (gpu_23), the sm≥90 capability probe (gpu_18), and measured negatives on other-silicon serving stacks (gpu_57: PDL rejected on a B300 hybrid; the vLLM #59737 rider: pays on short decode-side chains, pure-prefill TTFT regresses). This blog is sm_90-class evidence (H100/B200) — **outside our current hardware** (sm_89 + Metal), and the sm≥90 gate means the lever is unreachable on today's boxes. Our-shape pricing is recorded private-side (riir-ai R402 addendum). Re-open trigger: an sm_90+ box joins the league, or the decode-side chain-length profile changes.

## 3. Ownership boundaries (do not re-distill)

- Part 3 in full, the tiered-persistent-engine mechanism, the cost model, and the KB-vs-corpus signal-diff: **private riir-ai R402 owns these** — this note deliberately does not carry them.
- The internal league convergences (Metal fusion-neutrality, WDDM launch pricing, our-shape PDL pricing, the promotion-law validation, the agentic-harness corroboration): **private-side, R402's addendum.**

## 4. Routing

- This note: riir-infer (inference-substrate material; public skeleton).
- Staged leads 1–6: riir-refine `.distill/001` (blog-transcription class; claim plan-first per the pools; the TKCC paper's own unconverted instances fold into lead 1's batch).
- Private-side convergences + queue state: riir-ai R402 road-blog addendum.
- NO riir-infer implementation issue (see §2).

NOTHING INVENTED — every number above quoted from the fetched blog this session; every corpus verdict a live grep this session. Session: zed-m3-tkcc-road, epoch 1794890926.
