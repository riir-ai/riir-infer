# Research 026: FOSCAR — Affine Folding for Decode Attention over Low-Bit KV Caches

> **Source:** "FOSCAR: Accelerating Decode Attention over 2- and 4-Bit KV Caches through Affine Folding" — Zhuang, Ponnusamy, Wu, Song, Zhou (Together AI + U Sydney), [arXiv:2611.09722](https://arxiv.org/abs/2611.09722) (v1 2026-11-06). Open-source kernel library: `FutureMLS-Lab/FOSCAR`.
> **Date:** 2026-11-12
> **Status:** Active — verdict Gain; follow-up filed as [`.issues/1211`](../.issues/1211_foscar_kv_fold_decode.md)
> **Related Research:** 024 (LeapQuant — the GDN state-quant lane; the fusion axis), 023 (FlashStates), 020 (LSA), 007 (Triadic GDN)
> **Related Issues:** 1211 (the KV-fold decode lane — proof path), 1207 (LeapQuant GDN state Q8 — the state-readout fold rides it), 919 (ternary_kvq — the KV-half precedent)
> **Classification:** Public

---

## TL;DR

FOSCAR never reconstructs a cached KV value inside decode attention. Each quantization group's shared scale folds into the **other matmul operand** (operand-fold, their Eq 3) or the **group-local accumulated partial** (acc-fold, Eq 4), and the additive offset becomes an output correction — exact arithmetic, online softmax preserved, with a strict ordering law (QK corrections before softmax, PV corrections before the denominator division). The measured win is in **instruction-issue slots**, the scarce resource as stored bit width drops: per-element reconstruction alone exceeds B200's issue peak at INT2, folding cuts demand ~8×, and the kernel reaches 91.3%/60.6% of peak HBM at INT4/INT2 where prior INT2 kernels stall at ≤31%.

**Distilled for riir-infer (public serving substrate):** the Q8KV decode lane (`crates/riir-infer-gpu/src/attention_q8kv_cubecl.rs`) dequantizes every element inline (`value = d * qs[i]`) in both phases — the fold applies directly and exactly (Q8_0 is symmetric, z=0, so only scale-folding is needed). The (α,β) compensation law (their Eq 5) covers any operand we construct from integer codes by bit mapping — the ternary i8-mma lanes, the q4k metal tensor kernel. The break-even law (`g = 2·h_q`) and the per-byte issue-demand census (their Eq 2) are reusable admission predicates for our decode arms. Follow-up: `.issues/1211`.

---

## 1. Paper core findings (verified against the fetched full text, v1)

1. **Instruction issue, not HBM, is the low-bit bottleneck.** Their Eq 2: streaming at peak HBM stays within issue peak only when `a_issue/B_elem ≤ C_issue/B_HBM`. H100→B200: peak HBM ×2.39, issue throughput ×1.11. Reconstruction (INT→FP + `s·q+z` + format convert) costs 6 lane-slots/4 values on H100 → 3/6 slots/byte at INT4/INT2 vs a 9.99 peak; on B200 INT2 reconstruction is **8 slots/byte vs a 4.65 peak — exceeds**. Folding drops demand to 0.88–1.00 (B200) / 2.13–4.25 (H100) slots/byte.
2. **Affine folding, exact arithmetic.** For `C = A(s·q+z)`: per-reduction-index groups → **operand-fold** (scale A once per group, offset = a separate reduction of A); per-output-index groups → **acc-fold** (scale the group-local partial, offset × row-sum of A). Axis mapping for attention: QK^T reduces over D → token-wise K = acc-fold, channel-wise K = operand-fold into Q; PV reduces over L → token-wise V = operand-fold into P, channel-wise V = acc-fold. Acc-fold partials are **group-local** (cannot cross a group boundary unscaled). Corrections: QK before softmax, PV before the denominator division — softmax is untouched, so exactness carries through online softmax.
3. **Bit-level tensor-core operand construction + the (α,β) compensation.** Hopper: mantissa injection `0x4300 | q → q+128` (α=1, β=128), 1 ALU/elem. Blackwell: direct bit placement — INT2 bit patterns ARE E2M1 values `q/2` (α=½, β=0, 0.19 ALU/elem); INT4 → E4M3 `q·2⁻⁹` (α=2⁻⁹, 0.38). Their Eq 5 folds any such mapping into the group constants: `x = s'y + z', s' = s/α, z' = z − β·s'`, precomputed FP32 at store time. Zero per-element numeric conversion.
4. **Results.** INT2 geo-mean 3.36×/2.67× over FlashInfer BF16 (H100/B200); INT4 2.83×/2.30× (matches TRT-LLM NVFP4 on B200 at 1.00× while returning BF16 output). Kernel arithmetic error **vs the dequantized cache**: 1.68–1.76e-3 — matches BF16 decode; folding adds no error. Ablation: folding vs per-element reconstruction = **7.36–8.86× on B200** (1.07–1.15× on H100, where reconstruction is still within issue peak). Serving Qwen3-8B @100K ctx: +170%/+341% aggregate throughput over 2-bit/16-bit baselines.
5. **Break-even law (their App C):** folding costs `2·h_q/g` scalar affine ops per stored value vs 1 for reconstruction; break-even at `g = 2·h_q`; GQA `g ≥ 32 > 2h_q` → folding wins up to 64× fewer scalar ops. Head-ratio sweep: gain **grows** with H_KV (3.89×/7.68× at H_KV=32, batch 1/8, H100); at h_q=32 (H_KV=1, tiny cache) folding can LOSE (0.77–1.32×).
6. **Axis ablation (their App B):** token/token fastest everywhere; channel/channel slowest (0.57–0.73× of token/token at INT2 on B200) — channel-wise K re-scales Q per token group and channel-wise V drains group-local partials per group.

---

## 2. Distillation

### Vocabulary translation (paper → our stack)

| FOSCAR | Our stack | Status |
|---|---|---|
| decode attention over quantized KV, per-element `s·q+z` reconstruction | `crates/riir-infer-gpu/src/attention_q8kv_cubecl.rs` (Q8_0 KV, **inline per-element dequant** in both Q·K and PV phases; f32 sink sidecar per Issue 716) | **gap — reconstructs** |
| KV quantization at store time | `src/deltanet/ternary_kvq.rs` (Issue 919 T3, full-attention layers of the hybrid), `src/quant/q8kv.rs` + `src/quant/kvq_ab.rs` A/B backends (RawF32 / Q8Absmax / ExemptQ8) | **ships** |
| scale × group partial (implicit acc-fold, weight side) | the block-quant weight GEMM lanes (`gemm_q4k_*`, `gemm_ternary_cmma_i8_*`, `exl3_dequant`) accumulate per block then scale the partial | **ships on the weight side only** |
| bit-mapped operand construction, (α,β) compensation | ternary i8-mma lanes map 2-bit ternary codes into i8 operands; `gemm_q4k_mma.metal` builds f16 operands from 4-bit codes | **the Eq 5 compensation law is new here** |
| offset/zero-point correction (`z·rowsum(A)`) | our KV/weight formats are symmetric (Q8_0, ternary: z=0); q4_K carries `dmin` | **no offset-correction kernel exists** |
| issue-demand census (Eq 2, lane-slots/byte) | no per-byte issue-cost instrument on our decode arms | **reusable diagnostic** |
| break-even `g = 2·h_q` admission predicate | admission-gate discipline exists (pregate/budget precedents); this predicate is new | **reusable** |
| ⚠ "fold" (ours) | `attention_decode_folded_f32` / `fold_dispatch` = query-token coalescing per workgroup (Plan 179) — **an unrelated sense of the word**; the folded-dispatch lane predates and does not cover this | vocab-collision note |

### Closest cousins (checked by formula read, not name match)

1. **`attention_q8kv_cubecl.rs`** — per-element inline dequant, one FMUL per element per phase. *Signal diff:* reconstructs every value (O(T·D) FMULs) vs folding — acc-fold QK per (pos, block) + operand-fold of P per (pos, block); saves one FMUL per element out of ~5 per-element ops; Q8_0 is symmetric (z=0) so the offset-correction half of the algebra is vacuous at Q8 and matters only if an asymmetric format (dmin-carrying) ever feeds this lane. Complementary: same cache bytes, cheaper operand path (full recipe: Issue 1211).
2. **`ternary_kvq.rs` (Issue 919)** — quantizes KV **rows at store time**; the kernel half is not addressed there. The fold is the kernel-half completion for any quantized-KV lane.
3. **Block-scale × partial in the weight GEMM lanes** — the acc-fold *shape* already ships there, but only as the weight-side epilogue; operand-fold, offset corrections, the attention ordering law, and the issue-throughput framing do not ship anywhere.

### The concrete Q8_0 mapping in this stack

Our Q8_0 KV cache (`src/quant/q8kv.rs`, consumed by `attention_q8kv_cubecl.rs` decode + `qwen_prefill_q8kv_cubecl.rs` prefill) groups **token-wise**: one scale per (position, kv_head, 32-dim block), z=0 symmetric. Plugging into §1.2's axis table: QK^T reduces over head_dim with the scale varying along that reduction AND per position → **acc-fold** — per (pos, block), a raw 32-wide dot then one multiply of the partial by the block scale; the scale differs per position, so no operand pre-scaling can span positions. PV reduces over positions with the scale varying along that same axis → **operand-fold into P** — `p'(pos, block) = p · v_scale(pos, block)` computed once per (position, block), each lane-thread FMA-ing raw converted i8 values against it. Both folds remove exactly one FMUL per element (the scale multiply) out of ~5 per-element ops — at Q8 the honest expected gain is single-digit %, and the paper's own ablation bounds the upside class (1.07–1.15× where reconstruction is near the issue peak; Q8 reconstruction is cheaper again, and Metal/wgpu Q8 decode is memory-bound — the issue-bound framing does not transfer without a census). One incidental find while reading the kernel: the packed word is re-read per element (`kv_qs[word_idx]` inside the 32-element loop — 4× redundant per u32); an unpack-once rewrite is a nearby orthogonal win, filed as its own arm in Issue 1211 T3b so it never confounds the fold A/B.

### Prior art (external, delegated web check)

- The folding **algebra** pre-exists attention: Jacob et al. [arXiv:1712.05877](https://arxiv.org/abs/1712.05877) (2017) — the canonical (scale, zero-point) accumulator algebra for integer-only GEMM/conv; Marlin epilogues ([arXiv:2408.11743](https://arxiv.org/abs/2408.11743)) rescale group partial sums. FOSCAR's novelty is scoped to the attention application + the issue-throughput framing — **not ours to claim**.
- No uncited competitor does s·q+z folding for attention (checked 2026-11-12); nearest uncited contemporary: **FuTransformer** [arXiv:2610.23796](https://arxiv.org/abs/2610.23796) — INT4-KV decode on Hopper with fragment-layout *per-element* dequant (the reconstruction posture, not folding).
- **The linear-attention / GDN folding lane is empty publicly** — the 2025–26 state-quant wave (LeapQuant, LRB/BDR, GGA, DAMP, Minima) quantizes the recurrent state but none folds a dequant affine into state-readout operands.
- **Ada (sm_89) lane:** only BitDecoding ([arXiv:2503.18773](https://arxiv.org/abs/2503.18773)) targets Ada; FOSCAR is Hopper/Blackwell (wgmma/tcgen05) only.

### Fusion

- **LeapQuant (Research 024, filed today) × FOSCAR:** Issue 1207's lane quantizes the GDN recurrent state; the state readout (`Ŝ₀ᵀΓx`, their Eq 10 fixed-state term) and buffer-record readouts are matmuls over that (future) quantized state — the state-dequant affine can fold into the x/key operands exactly as FOSCAR folds K. Publicly unclaimed lane; rides Issue 1207 as a future arm, not filed separately (noise law).
- **League arm:** INT4-KV decode on the 4090 (Ada) for the league models' full-attention layers — Bonsai-2-27B carries 16 of them; the league decode cell (tg128 0.970×) is memory-bound and the KV half is F16 today. Lossy arm → per-family retention gates + league law (FNV/argmax digit-exact). Deferred in Issue 1211; not started.

---

## 3. Verdict

**Tier: Gain.** Exact, modelless, inference-only algebra with a concrete consumer (the Q8KV decode lane) and shipping proof infrastructure (`kvq_ab.rs`). Not GOAT yet — the gain at Q8 is expected modest (FOSCAR's own data: folding buys 1.07–1.15× where reconstruction is within issue peak; Q8 reconstruction is 4× cheaper than INT2 again, so the honest bar is a kernel-level single-digit %, demote-if-fail per Issue 1211's pre-registered gate). Not Super-GOAT — the mechanism is external and published; no novel combination of our primitives yet produces a capability competitors lack (the GDN-state fusion is recorded, unproven).

**MOAT gate (riir-infer row):** inference-substrate novelty (quant formats, kernels) shipped public + upstream-clean — fits exactly; this note + Issue 1211 are public-surface artifacts. Not a reflex/rethink/ai concern: no decision-serving, game-runtime, or economy surface is touched.

**Path 0 / panel:** N/A — pure inference-substrate paper (KV cache class skips the model-based advocate per the skill).

**Proof path (Ungoat law):** fixture = `kvq_ab.rs` backends at the Q8Absmax posture; falsification arm = numeric parity of the folded kernel vs the inline-dequant incumbent (tolerance class stated in the issue); pre-registered gate = decode-attention kernel latency A/B, ≥5% or demote.

**Pre-flight live-state (this session):** `riir-infer/.research/.highwater` = 025 read before allocating 026; `riir-infer/.issues/.highwater` = 1210 read before allocating 1211.
