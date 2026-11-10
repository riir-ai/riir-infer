# Research 023: FlashStates — Hybrid State Prefix Caching for Agentic Serving

> **Source:** "FlashStates: Unlocking Flexible Hybrid State Management for Agentic LLM Inference" — FutureMLS-Lab (Li, Xu, Zhang, Wang, Zhou, Xu), 2026-11-09. alphaXiv `2611.flashstate`; code https://github.com/FutureMLS-Lab/FlashStates (Apache-2.0).
> **Date:** 2026-11-10
> **Status:** Active — verdict Gain; follow-ups filed as `riir-infer/.issues/1201`
> **Related Research:** riir-infer 015 (multi-host), 016 (PHOTON decoding)
> **Related Issues:** riir-infer 1201 (hybrid-state serving work items)
> **Classification:** Public (paper distillation + public-substrate mapping; no private-tier content)

---

## TL;DR

FlashStates fixes hybrid-LLM prefix caching (attention layers reuse KV; recurrent layers need a state checkpoint at the *exact* resume position). vLLM saves recurrent checkpoints only at KV-block boundaries and evicts KV by recency, so every agentic turn recomputes the tail of the previous one; SGLang checkpoints denser but memory-pressured. FlashStates: (1) **ASC** — capture the checkpoint at the true resume point (end of request; optionally mid-prefill as a kernel side-output), (2) **cost-aware retention** — one tuning-free GreedyDual-Size priority (prefill-seconds saved per cache block *actually freed*, shared blocks not double-counted, last-use watermark, inner-before-leaf) driving BOTH admission and eviction of whole states, (3) optional **host tier** — write-through of whole states to a Mooncake store with the same priority. On GLM-5.3-Flash NVFP4 @ 1M ctx, 48 sessions, 2×B200: **3.14×** vLLM prefill throughput, **96%** fewer median uncached prompt tokens, **88%** lower p90 TTFT; decode **3.02×**.

**Distilled for riir-infer (modelless serving substrate):** three deterministic, training-free mechanisms — placement (capture at the true resume point), retention (one dependency-aware cost-density priority for admission + eviction), capacity (host tier evicting by the same priority). **The single-stream serving half already ships**: `riir-infer-gpu/src/qwen38_prefix_cache.rs` (Issue 742 T4) captures whole-prefix KV+GDN checkpoints at caller-chosen anchors (turn end), restores, and re-fills only the suffix. The FlashStates deltas are concrete and filed (Issue 1201): **multi-session** scale-out on the already-named radix path (katgpt-rs Issue 771 `RadixPrefixTree`, Bench 762), **cost-density retention** replacing recency + leaf-last, and **in-chunk kernel export**.

---

## 1. Paper core findings

- **Hybrid state** = the attention KV for every token of a prefix **plus** a recurrent checkpoint at that prefix's exact end. A recurrent checkpoint cannot be rolled back to recover an earlier one — this is why placement matters.
- **Miss taxonomy** (their profiling of vLLM + SGLang on GLM): **placement miss** (no checkpoint saved at the needed point), **dependency miss** (checkpoint evicted while its KV remains — the prefix is unusable), **capacity miss** (memory removes the KV too). Placement + dependency dominate below 40 sessions; capacity grows from 40.
- **ASC (arbitrary state checkpointing):** a finished request's state is captured at its exact end ("frontier"); `--mamba-prefill-export` lets a prefill chunk run past a cache-worthy position and have the backend write the state there. Scope honesty: in-chunk export was exercised only on GLM's KDA kernel; Qwen3.8-Flash-Next and Nemotron 3 Super use exact request-completion capture.
- **Retention:** priority = (extra prefill seconds if evicted) ÷ (cache capacity eviction actually releases — KV blocks shared with another surviving state count as NOT freed), combined GreedyDual-Size-style with a last-use watermark. One priority drives admission AND eviction. The prefill-cost model is fitted once per model (two coefficients, e.g. GLM `1.862e-5 3.33e-6`). **Leaf-last:** a session's newest state (the resume frontier) is evicted only after every inner checkpoint of the same session; a leaf idle > S seconds ranks as inner.
- **Host tier:** each finished request's hybrid state written through to Mooncake in host memory; the master evicts whole "recovery units" (state + the pages only it needs) by the same priority (`unit_gds`).
- **Numbers** (48 sessions, per-GPU): GLM-5.3-Flash 3.14× prefill / 3.02× decode, median uncached tokens 24.7K → 1.03K, p90 TTFT 20.4 s → 2.36 s; Nemotron 3 Super 2.27×/2.32×; **Qwen3.8-Flash-Next** (512K) 1.54× both; Kimi-K3.1-Pro on GB300 +36% peak output vs vLLM with p90 TTFT 2.0 s vs 9.1 s.
- **Ablation is the honesty core** (Qwen3.8-Flash-Next, 512K, decode K tok/min/GPU): at 32 sessions ASC+LRU is **below vLLM** (26.0 vs 30.1, −14%); at 48 it is +42%. Adding admission → 28.2/36.8; adding dependency-aware eviction → 30.4/38.8 (+61% over vLLM). **Placement alone is not enough — retention is load-bearing.**
- **Headline caveats (the paper's own + alphaXiv's review):** the 3.14× is one model at one load on one hardware setup; "prefill throughput" counts cached tokens too (it is not generated tokens); the retention policy's optimization analysis assumes exact continuation / independent tool calls / fluid capacity — design rationale, not a proof; nearest predecessor **Marconi's** tunable α swings throughput up to 16% and its online tuner is unreliable (FlashState's motivation for the tuning-free priority); concurrent **HiState** (SSD tiering) is contrasted, not benchmarked.

## 2. Distillation

### Vocabulary translation (paper → our stack)

| FlashStates | Our stack | Status |
|---|---|---|
| hybrid state (KV + recurrent checkpoint) | `qwen38_prefix_cache.rs` (Issue 742 T4): whole-prefix checkpoint = attention KV tail *implicitly live* + full GDN state (48 rec mats + 48 conv windows) in device side-buffers; `CheckpointLayer::{Attention{key,value}, DeltaNet{rec,conv}}` in `riir-infer-dist/src/spec_fill.rs` for the rewind lane | **SHIPS — single-stream serving (742 T4) + rewind-scoped (spec_fill)** |
| frontier checkpoint (capture at request end) | 742 T4 captures at **caller-chosen semantic anchors** (the doc end before the question, a turn end) — the same positions FlashStates would pick; automatic end-of-request capture is a trigger-policy delta, not a capability gap | **partial (trigger automation)** |
| multi-session (48 concurrent) reuse | 742 T4 is explicitly single-stream; the successor is ALREADY NAMED in its header — `katgpt_kv::radix_prefix::RadixPrefixTree` + `PagedKVCache` chunk-page seam (opt-in `radix_prefix_cache`, katgpt-rs Issue 771 / Bench 762: 2.45× hit-rate at equal page budget) | **gap — path named, unbuilt on this lane** |
| in-chunk export (kernel side-output) | — (deltanet kernels expose no state side-output; 742 T4 snapshots at anchor boundaries between forwards) | gap |
| GDS cost-density admission + eviction | 742 T4 evicts by recency + the `note_write` lineage prune (a checkpoint whose KV tail a younger write could clobber is pruned first — the single-stream dependency-miss rule); no cost-density priority; leaf-last absent | gap |
| Mooncake `unit_gds` host tier | the persistence half is ALSO pre-named: riir-infer Issue 1108 T2 (Research 014 §3.2, the Locals paper's cross-restart cache) — BLAKE3-committed frozen state, thawed at boot | gap — evaluation rides 1108 T2 |
| compact spec-state layout | per-request `StageCheckpoint` buffers in spec_fill | partial analog |

### Closest cousins (checked, not name-matched)

1. **`riir-infer-gpu/src/qwen38_prefix_cache.rs` (Issue 742 T4)** — the single-stream serving prefix cache: whole-prefix KV+GDN checkpoints at semantic anchors, restore + suffix re-fill, `note_write` lineage pruning, CUDA-graph-safe, rides `ternary_gemv_cuda_raw` (CUDA-only). *Signal diff:* one session, exact-match longest-prefix, recency eviction, caller-chosen anchors — vs FlashStates' N sessions, cost-density admission+eviction, leaf-last, automatic request-end trigger, in-chunk kernel export. **The FlashStates delta is 742 T4 × (many sessions, better retention, kernel-side capture) — an extension of shipped substrate, not a build-from-zero.**
2. **`riir-infer-dist/src/spec_fill.rs`** — hybrid checkpoint/restore for the spec-rewind lane. *Signal diff:* rewind-consumable **within one request** (checkpoint → consume drafts → restore on reject); second-closest cousin, cited for the capture shape (`CheckpointLayer::{Attention, DeltaNet{rec,conv}}` + `HybridStateSegment`).
3. **riir-ai Research 342 (FreeToken, arXiv:2608.16157)** — the internal prior distill whose GDN-checkpoint design input 742 T4 cites; the workspace already walked this family once. This note extends that lineage to the multi-session + retention axis.
4. **`gdn_delta_roll`** (riir-infer Issue 049 per the AGENTS.md lane table, opt-in) — GDN state journal: a state *persistence* seam, not a cache policy.
5. **riir-ai Issue 1165 resident lane** (reported in riir-ai AGENTS.md; code not read this session) — ctx-2048 resident serving = a single-session resident context; FlashStates generalizes to N sessions under an eviction policy.

### Fusion / consumer mapping

- **League intel (priority-3 lane, inference-perf):** vLLM **and** SGLang — both perf-rematch opponents — measurably collapse on concurrent agentic hybrid workloads (vLLM p90 TTFT 20.4 s, 24.7K median uncached tokens at 48 sessions). A concurrent-agentic cell is a **winnable lane** for our serving posture; the ablation model being **Qwen3.8-Flash-Next** says our qwen3.8 lane's model class is exactly the agentic serving target — and the qwen3.8 lane is the one that already carries 742 T4.
- **Substrate delta (extension, not invention):** the missing halves are (a) **multi-session** — the radix successor is pre-named and pre-measured (Issue 771 / Bench 762: 2.45× hit-rate at equal budget; the 742 T4 header forbids a private tree), (b) **retention** — recency + lineage pruning → cost-density GDS + leaf-last (the paper's ablation proves this half is load-bearing: ASC+LRU is NEGATIVE at 32 sessions), (c) **in-chunk export** — deltanet kernel side-output (Metal + CUDA arms per the compute law).
- **Persistence is pre-named too:** riir-infer Issue 1108 T2's freeze/thaw fusion (BLAKE3-committed frozen state, thaw at boot) is the natural host for the FlashStates host-tier question at our scale — the T5 verdict evaluates against that lane, not a from-scratch LocalKv spill.

---

## 3. Verdict

**Gain.** One line: ships (Apache-2.0, reproducible benches) and is actionable — measured opponent weakness on a league lane we can contest, and three named deltas on substrate we already own (742 T4 single-stream serving cache + the Issue 771 radix path + Issue 1108 T2 persistence) — but it originates no primitive for us to claim (GreedyDual-Size cost-density is established; Marconi is the published predecessor; HiState/ReplaySSM/vLLM RFC 64508 crowd the same month), so the honest tier is adoption/adaptation, not GOAT/Super-GOAT. The gap is SMALLER than a first-pass read suggests — single-stream capture ships — and the note's value is precisely scoping what remains: many sessions, cost-aware retention, in-chunk export.

**MOAT gate (riir-infer, public substrate):** strengthens the moat — names the serving-state gaps on lanes this repo owns (deltanet kernels, dist stage checkpoints) and records league-grade opponent weakness with reproducible numbers. Decision serving (reflex) and game runtime (riir-ai) untouched.

**Files:** this note + `riir-infer/.issues/1201` (work items + pre-registered proof path). No plan yet — the GOAT cell design is Issue 1201 T1; the plan opens when the cell is pre-registered.

## 4. Published prior art (§4 sweep)

**Internal-first:** zero prior distills of FlashStates or Marconi anywhere in the workspace `.research/` folders (grepped this session) — **but the direct internal cousin EXISTS and was missed by the first sweep**: riir-ai Research 342 (FreeToken, arXiv:2608.16157), whose GDN-checkpoint serving design 742 T4 already consumed. The first pre-flight grep for this note searched only "FlashState\|Marconi" and its `prefix_cach` grep over the workspace was PAGINATED and never paged — the verdict reviewer caught `qwen38_prefix_cache.rs`; both facts re-verified at source this session. Recorded here as the sweep-method lesson, not just the citation.

- **Marconi** (arXiv:2411.19379, MLSys'25) — first hybrid prefix caching; admission/eviction with tunable α; FlashStates' stated deltas: tuning-free single priority + placement decoupled from block boundaries.
- **HiState** (arXiv:2610.41508, 2026-10-30) — hierarchical: KV in fast memory, historical checkpoints on SSD, adaptive intervals.
- **Sparse Prefix Caching for Hybrid and Recurrent LLM Serving** (arXiv:2605.05219) — sparse exact-state checkpoint positions as a design point.
- **vLLM RFC #64508** (2026-10-29) — checkpoint-internal paging for hybrid recurrent state (upstream acknowledging the same gap).
- **ReplaySSM** (NVIDIA blog, 2026-10-12) — SSM state as checkpoint + recent-inputs buffer.
- **RadixFlow** (arXiv:2610.18143) — execution bubbles under prefix caching (scheduler half).

## 5. What remains

- `riir-infer/.issues/1201` (open): T1 cell pre-registration, T2 multi-session radix extension of 742 T4, T3 cost-density retention + leaf-last, T4 in-chunk GDN export, T5 persistence verdict via Issue 1108 T2.
- `.raw/flashstate` clone removed after commit (provenance pinned below).

## Provenance

Repo clone `FutureMLS-Lab/FlashStates` @ `e2ea9a7864b41081a0147a3968e73bd5c6875a7d` (main repo only; `third_party/vllm` + `third_party/mooncake` submodules NOT cloned), Apache-2.0. alphaXiv overview fetched 2026-11-10. Implementation facts quoted from `docs/what-flashstate-changes.md` and `benchmarks/README.md` at that sha. The paper PDF (`docs/FlashStates_Release.pdf`) was NOT parsed this session — the cost-model arithmetic (the exact form behind the two coefficients) is inferred, not quoted. The substrate facts about 742 T4 / Issue 771 / Issue 1108 T2 were verified by reading `riir-infer-gpu/src/qwen38_prefix_cache.rs` headers at source this session (round-1 REVISE correction — the original draft wrongly marked single-stream serving capture as a gap).
