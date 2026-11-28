# Plan 627: GLM-5.3-Flash 1M-context lane on M5 Ultra — arch support + LSA bucket indexer (owner-pulled)

**Status:** Filed + claimed — owner pull landed 2026-11-03 (this conversation): M5 Ultra fleet (amo/bmo 96GB now, +4×128GB planned) will run GLM-5.3-Flash with 1M context as the USUAL posture; LSA prefill is the owner-stated need. Design input: [`.research/020_LSA_Latent_Sparse_Attention.md`](../.research/020_LSA_Latent_Sparse_Attention.md) + [`.research/021_ZHA_Hierarchical_Tree_Indexer.md`](../.research/021_ZHA_Hierarchical_Tree_Indexer.md) (ZHA, arXiv:2610.42112 — design rows adopted below: top-p mass-rule A/B as a third retention arm, crossover-band pre-registration, trained-tree arm ONLY if the PCA floor misses G1; the KL reference-query distill PoC is riir-train [`.issues/666`](../riir-train/.issues/666_zha_reference_query_distill_poc.md) — no model backward, PQ2/kernel-frozen compatible, the distill-arm vehicle if Phase 4's floor misses); lane lineage: Plan 624 (sparse prefill arm) → bench 1157 → issue 1165 (τ=0.85 envelope promotion, ≥16K, Metal — the dense-side baseline this lane extends).

**Lease closed:** zed-m3-lsa-distill, 1793698334, was expires 2026-11-10T00:00:00+00:00 — expired undischarged 13 days; folded to the closed-row form (the expired line made this file's lease AMBIGUOUS, blinding the repo-wide claim check; the live claim below stands).
**Housekeeping 2026-11-23 (zed-bb86e4df@amo, boss sweep): EXPIRED UNDISCHARGED 13 days — re-claimable.**
**Lease closed:** zed-6addfb44@amo, 1795458782, was expires 2026-11-25T18:33:00+00:00 — expired undischarged (zero landing commits; scope was the Phase 0 pre-arrival track); adjudicated expired 2026-11-26 by the idle lane.
**Lease closed:** zed-7e18a471@amo, 1795657452, was expires 2026-11-28T01:44:12Z — lapsed undischarged (zero landing commits; scope was the Phase 0 pre-arrival track); adjudicated 2026-11-28T01:44Z by the 4090 wake-continuation lane per the loaded route-or-hands-off recipe (lapse + the lane's Phase 0 scope is portable network+CPU work → claimed below instead of routed).
**Claimed:** zed-4534384e@4090, 1795835100, expires 2026-11-28T08:00:00Z
(scope: the same Phase 0 pre-arrival track — arch probe from the HF metadata, serving-posture survey, memory-budget doc — executed from the 4090 box (network+CPU only; the ~650GB weights pull stays owner-gated per the plan's own row); probe instrument lands as `scripts/plan627_glm53_hf_probe.py` (the plan402_gguf_probe pattern); Phase 1+ re-claims per the lease law.)

## The two mountains (honest scoping)

1. **Serving GLM-5.3-Flash at all** — 320B total / 18B active MoE, hybrid sparse+linear attention, MLA latent, mHC, natively multimodal. HF card lists SGLang/vLLM/TokenSpeed/Transformers/KTransformers/Unsloth — no llama.cpp, no MLX. On M5 Ultra the paths are: (a) OUR stack (riir-infer GLM lane — the league premise), (b) quantized single-box (ternary/PQ2 ≈ 63–80 GB weights — our Bonsai specialty), (c) FP8 across 4×128GB via the Plan-623 multi-host lane (quality-preserving).
2. **LSA prefill speedup** — the paper's own curve on THIS model (H200): 3.5×/attention-layer prefill @1M vs its trained sparse baseline; 22.7× is the 10M cell; vs a DENSE baseline the sibling row (Qwen) shows 8.2×/layer @1M. End-to-end @1M vs the sparse baseline is modest (1.09× CP8 / 1.51× TP8 in Table 3) — the honest expectation for OUR first rung is per-layer and dense-baseline-relative, measured, never promised.

## Goal

One pre-registered gate: **does the PCA-training-free LSA bucket indexer beat our dense prefill arm by ≥2× per attention layer at pp1M on m5ultra, at G1 accuracy parity on real-tensor long-context captures?** Win → promote; lose → demote loser, recorded negative (the HGA precedent shape).

## Hardware gate (owner note, 2026-11-03)

The 4×128GB M5 Ultra boxes are **reserved for this lane but not on site yet** ("soon"). Two tracks:

- **Pre-arrival (96GB pair: amo builder / bmo runner)** — Phase 0 acquisition + arch probe (CPU work), Phase 1 CPU-reference forward + parity, Phase 2 quant-lane development (ternary/PQ2 recipe on the MoE experts), Phase 3 harness CONSTRUCTION (real-tensor captures at ≤128k), Metal bring-up at short contexts. No 1M claims off 96GB.
- **GATED on 128GB arrival** — pp512k/pp1M bench cells, the PQ2 single-box @1M posture (63–80GB weights + ~10–15GB 1M-KV + routing buffers is over-budget on 96GB, comfortable on 128GB), and the FP8 4-box multi-host posture (Plan-623 lane). The G2 gate (≥2×/layer @pp1M) runs only after the gate opens; the pre-arrival G2 bar is set at the ≤128k cells instead.

## Phases

### Phase 0 — acquisition + landscape
- [-] Download `zai-org/GLM-5.3-Flash` checkpoint (HF; verify SHA256 per-token, record BLAKE3 in `data/BASE_MODELS.md`-style pin — third-party pin law). Default target `SDXC1TB` — internal free space cannot hold the ~650GB bf16 checkout; OWNER CONFIRM before the multi-hundred-GB pull (or the FP8-class variant if the repo ships one) — **owner-gated, unchanged; probe-verified 2026-11-28: usedStorage 611.6–656.7 GB across two same-hour HF API reads (LFS accounting drift), 62 shards, fp8 quant_method with 1,509 modules_to_not_convert rows**
- [x] Arch probe (the `plan402_gguf_probe` pattern for safetensors/config): layer map (sparse vs linear attention split), MLA latent dims, mHC placement, MoE expert count/active, rotary layout (paper says none — verify in config), tokenizer — **LANDED 2026-11-28 (zed-4534384e@4090): `scripts/plan627_glm53_hf_probe.py` + the §Phase-0-probe section below**
- [x] Survey community posture: MLX / llama.cpp GLM-5.3 support state (the serving-vehicle fork; llama.cpp = the league opponent) — **DONE 2026-11-28 (same session): llama.cpp supports glm5_next first-class; MLX has 20 community conversions; see §survey below**
- [x] Memory budget doc: 1M KV under MLA ≈ 1 GB/layer/sparse-layer (paper datum) → total KV + weights per posture (96GB single-box ternary vs 4×128GB FP8 multi-host) — **DONE 2026-11-28: measured 1.00 GB/layer bf16 (kv_lora_rank 512, no rope dims) × 11 DSA layers = 11.0 GB @1M; the paper datum CONFIRMED by config; §budget below**

### Phase 1 — GLM arch lane in riir-infer (the big lift)
- [ ] `glm5` arch feature: safetensors loader (MoE + tied structures), text-lane forward skeleton
- [ ] MLA latent attention forward (CPU reference first, parity posture; then Metal via the 1146 tensor stack)
- [ ] Hybrid linear-attention layers (GDN-class recurrence — reuse the `deltanet/` lane) + mHC
- [ ] MoE routing + expert dispatch on Metal (grouped GEMM posture)
- [ ] `- [-]` vision tower — DEFERRED (text lane first; multimodal encode is a separate lane)
- [ ] Parity gate: CPU reference vs Metal, frozen captures, p-drift ≤ 1e-3 (the G5 discipline)

### Phase 2 — quantization posture
- [ ] FP8/quant eval on the weights; ternary/PQ2 lane applied to MoE experts (Bonsai recipe) for the single-box 128GB posture
- [ ] Lossy-surface law: per-family conditional retention gate (per the UQ/lossy promotion rule), never aggregate-PPL-only

### Phase 3 — long-context bench cells + accuracy harness (do not defer)
- [ ] League manifest expansion pp256k / pp512k / pp1M on m5ultra — owner-sanctioned this conversation (2026-11-03); llama.cpp baseline row only if it supports the model by then, else dense-self baseline
- [ ] Real-tensor long-context accuracy captures: needle + multi-hop trail tasks over REAL documents (random-key NIAH is BANNED — the HGA lesson, Plan 397 / Research 379); PaperTrail-style fabricated-trail construction is the reference design
- [ ] Box-state preflight on every cell (bench_preflight posture; rig runs via `agent-mail rig run` — bmo quiet runner for long cells, exclusive for benches)

### Phase 4 — LSA bucket indexer (the owner-visible speedup)
- [ ] `lsa_bucket` feature (default-off): RoPE-free/MLA-latent index space (the pinned `index_space.py` contract), d(idx)=64, pool 4:1, Gram-Ward buckets (n̄b≈8, unit-norm centroids), bucket cache (mmap-friendly — NeuronShard-adjacent layout)
- [ ] Flat search first: threshold top-k (sample→θ→compact→rescore→packed score‖id radix select — the App. F kernel shape) as wgpu/Metal compute or CPU SIMD (d=64 makes a CPU indexer viable for the poc)
- [ ] PCA training-free projections = the FIRST arm (zero training); distillation (W^idx_Q,K + τ) only if the floor misses G1 — rides the riir-train Plan 476 owner-gated lane shape
- [ ] Retention-rule A/B arm (ZHA row 1): top-p mass routing vs τ-envelope vs top-k at the pre-arrival ≤128k G2 bar — three arms, one gate, demote losers
- [ ] Crossover-band pre-registration (ZHA row 4 / build-hygiene law): predict N* (the bucket count where the indexer starts paying) BEFORE the pp256k+ cells, land the measured crossover in-band
- [ ] Tree/beam search deferred until flat wins (paper: tree loses accuracy past 1M on GLM; decode tree only pays batched ≥1M; ZHA row 5 — a trained-tree arm opens ONLY if the PCA floor misses G1 at pp1M, with the ZHA per-level mass schedule as its starting shape)
- [ ] GOAT gate (pre-registered HERE, before any measurement): G1 accuracy parity on real captures (≥ dense-self on needle/trail families, per-family not aggregate) · G2 ≥2× per-attention-layer prefill @pp1M m5ultra vs dense arm (--release, exclusive rig, box-state quoted) · G3 flag-off byte-identical at ALL existing league cells (≤32k — the crossover law says the flag must not touch them) · G4 alloc-free hot path
- [ ] Promote-to-default only on G1–G4 + owner; demote the dense arm in the ≥256k cells if won (keep dense <16K per the issue-1165 conditions)

## Non-goals
- 10M posture (CP8/H200 class) — not on this fleet; revisit if a 10M consumer appears
- CuTe DSL kernel parity (H200/B300-specific) — our arms are Metal/wgpu; speedups will differ and are measured, not transcribed
- Training N-LSA-style from scratch — out of charter

## Sibling etiquette
 amo = builder; bmo = quiet runner for long cells; heavy benches exclusive-run. The i1167 packed-byte lane (claimed to 11-04) is untouched; if it lands first, the Phase-1 Metal kernels build on it.

## 2026-11-28 — Phase 0 pre-arrival probe + survey + memory budget (zed-4534384e@4090)

Claimed scope executed (lease row above). Probe instrument: `scripts/plan627_glm53_hf_probe.py` (the plan402_gguf_probe pattern — metadata-only, stdlib, console-encoding-defended, re-runnable; full output in the run of record 2026-11-28T01:47Z, cache `E:/tmp/plan627_probe`). All facts pinned to HF repo state sha `46a050bec13002ad29ee795c16e61cf4fa3c4a9c` (lastModified 2026-10-15). Pre-staging reads: `.research/020` (LSA) + `.research/021` (ZHA) + this plan's own design rows.

### Arch map (probe of record)

| Axis | Measured (config.json) | Phase-1 consequence |
|---|---|---|
| Layer split | 45 layers: 34 `linear_attention` (KDA) + 11 `deepseek_sparse_attention` (DSA/MLA) at [3,7,11,15,19,23,27,31,35,39,43] | the plan's hybrid assumption CONFIRMED with the exact split; `full_attn_layers` also enumerated in `linear_attn_config` |
| KDA geometry | 64 heads × head_dim 128, short_conv_kernel 4, `gate_lower_bound` −5.0 (sigmoid-gated delta family) | maps onto the existing `deltanet/` lane (the Phase-1 reuse box CONFIRMED); deltas vs our GDN: the gate lower bound, short conv shape, per-layer list |
| MLA | q_lora_rank 1536, kv_lora_rank 512, qk_nope 256, v_head_dim 256, `mla_use_nope` true, **qk_rope_head_dim = 0** | the plan's "paper says none — verify in config" row RESOLVED: NO rotary on attention; 64 q heads over the compressed latent |
| Trained indexer (GLM's own, per DSA layer) | `index_head_dim` 128, `index_n_heads` 32, `index_kpool` 4, `index_kpool_compress` true, `index_kpool_always_select_tail` true, `index_topk` 2048, `indexer_rope_interleave` true | the checkpoint SHIPS the LSA-class indexer the paper measured against; `index_kpool 4:1` matches the paper's block-sparse family row; the indexer carries rope (interleaved) even though attention does not — the RoPE-free index-space rule (research 020 finding 7) applies to OUR indexer design, not to theirs |
| MoE | first_k_dense_replace 3; 288 routed + 1 shared; top-8; noaux_tc; sigmoid scoring; routed_scaling 2.5; moe_intermediate 2048 (dense 12288); router fp32; swiglu_limit 10.0 | grouped-GEMM dispatch posture; swiglu clamp is Qwen3-family (the riir-refine `clamp-emits-tail-error-term` numerics rule applies verbatim); sigmoid router = the house sigmoid law, no softmax anywhere |
| mHC | `mhc` true, `hc_mult` 4, `hc_eps` 1e-06, `hc_sinkhorn_iters` 20 | a REAL module (hc_attn_base/fn/scale + hc_ffn_base/fn/scale weights per layer, in the not_convert list) — the Phase-1 forward skeleton's "+ mHC" box is load-bearing: the residual stream is 4-wide (4096×4), sinkhorn-normalized mixing |
| MTP | `num_nextn_predict_layers` 1; layer 45 carries eh_proj/enorm/hnorm/shared_head (DeepSeek-V3 MTP shape) + its own MLA+indexer block; `index_share_for_mtp_iteration` true | an in-checkpoint draft layer — the i1253 `mtp_draft_block` lane shape has a direct future consumer here (never a Phase-1 requirement) |
| Context / vocab | max_position 1,048,576 (1M native); vocab 154,880; tie_word_embeddings FALSE | untied head = lm_head is a real matrix in PQ2 planning; 1M positions native, no YaRN interpolation row |
| Tokenizer | tokenizers-backend BPE; model_max_length 1,048,576; eos/pad `<|endoftext|>` (id 154820; gen_config eos list 154820/154827/154829); GLM special-token family (gMASK/sop/system/user/assistant/observation, image/video/audio sentinels) | the laya lane's `tokenizers` dep reads this class already; chat template = the gMASK/sop GLM shape with think tags + tool_call XML-ish args |
| Multimodal | vision tower (24 blocks + merger + patch_embed) + image/video tokens, `image-text-to-text` | Phase-1 vision box stays DEFERRED per plan; the not_convert list names the full visual.* set |
| Storage (HF-reported) | params: BF16 6.93B + F8_E4M3 314.40B + F32 ~0.3M = 321.32B total; usedStorage read 656.7 GB (01:2xZ) then 611.6 GB (01:47Z) across two same-hour API reads; quant_method fp8 with 1,509 modules_to_not_convert rows; 62 shards | **RESOLVED by shard-header probe (same session, 01:58Z): storage IS fp8** — sampled shard headers (1, 32, 62) read F8_E4M3 1-B/param data + 1:1-paired F32 scale tensors (329 F32 + 329 F8 in shard 1; 630:625 in shard 32; ~5.00 GB data/shard uniform, last shard the 1.17 GB bf16 lm_head/norm tail); true repo data ≈ 61 × 5.0 + 1.2 ≈ **306–328 GB, NOT 612–657 GB** — HF's usedStorage figure is ~2× LFS accounting overhead; SDXC1TB and the 4×128GB FP8 posture both gain headroom; per-tensor dtype ground truth: F8 experts + F32 scales + BF16 norms/embeddings/lm_head tail |

### Serving-posture survey (community, 2026-11-28T01:2x–01:3xZ)

| Vehicle | State | Evidence (search of record, GitHub/HF API) |
|---|---|---|
| llama.cpp (the league opponent) | **SUPPORTS glm5_next first-class** | PR #27773 "add GLM-5.3-Flash (GLM5-Next) support" MERGED; follow-ups merged: #31009 (QSA without kq_mask, + DSA), #30042 (removed the glm5-next sparse-attn gather path), #31253 (`-sm tensor` for LLM_ARCH_GLM5_NEXT), #32887 (query chunking), #30932 (input optimizations); OPEN: #30268 (server input-layer embeddings), #30093 (BUG: dflash specdec aborts — `t_layer_inp` never set for glm5-next); GGUF ecosystem mature: unsloth/GLM-5.3-Flash-GGUF (748K downloads), bartowski BF16 GGUF, 15+ conversions incl. IQ2_S/REAP-pruned variants. League consequence: a llama.cpp GLM-5.3-Flash baseline row is OFF-THE-SHELF when the hardware gate opens |
| vLLM | **Mature, actively perf-tuned** | 63 glm5 hits: merged perf PRs — TMA MoE up/down (+3.1–3.4% E2E), latency_specialized kernel migration (+3.8% E2E), fp8 KV cache for the dlfash draft (−2.85 GB/rank), split FP8 load, shared-expert fusion, CUDA-graph capture fixes (SM120), Rust-frontend HF renderer; open: #68848 KV-scale load bug (GLM5.2) |
| SGLang (the LSA paper's E2E vehicle) | **Deep indexer-specific work** | 178 glm5 hits: NPU interleaved prefill CP, **INT8/mxFP8 index cache for the KPool pooled-key indexer** (open #51016 — indexer-cache quantization is a live frontier), AMD fused DSA indexer Q/K rope+quant+K-cache write, comm-comp overlap prefill triton, PTPC FP8 KDA projections (gfx950) |
| MLX (the Mac path) | **Community conversions only, 20 repos** | mlx-community GLM-5.3-Flash-oQ8-mtp / oQ6-mtp (MTP-RETAINING), TensorFold 4bit/2bit/oQ4-MTP, pipenetwork mixed-4/8 + REAP25/37/50 pruned, inferencerlabs Q9; no official mlx-lm arch row observed — the conversions ARE the support; MTP-retaining quants already ship |
| Card-claimed (not probed) | TokenSpeed / Transformers / KTransformers / Unsloth | the HF card lists them; one community Space claims single-5090 serving (heavy quant) — unverified |

### Memory budget (the gate's box)

KV @1M (bf16, measured from config): DSA layers = (kv_lora_rank 512 + rope 0) × 2 B × 1,048,576 = **1.00 GB/layer × 11 = 11.0 GB** — the plan's paper datum CONFIRMED. Indexer KPool cache @1M: (T/4 × 128 dim) bf16 upper bound = 64 MiB/layer × 11 = **0.69 GB** (the LSA paper's GLM datum 331 MiB/layer @10M scales to ~33 MiB/layer @1M — same class). KDA recurrent state: 64 × 128² × fp32 = 4.0 MiB/layer × 34 = **0.13 GB per sequence** (context-independent — the linear-attention dividend: 34 of 45 layers carry NO length-growing KV at all). @128k: 1.38 + 0.09 + 0.13 GB — trivial.

| Posture | Weights | KV @1M | Working verdict |
|---|---|---|---|
| 96GB single-box, PQ2 ternary experts | plan est. 63–80 GB (Phase-2 recipe pending) | 11.0 + 0.7 + KDA 0.13 | 74–92 GB + mHC-4× activations + OS — **over/at budget; the hardware-gate verdict STANDS**; decode-tight even if it fits |
| 128GB single-box, PQ2 | same | same | comfortable (~92 GB peak worst case); the G2 @pp1M gate opens here |
| 4×128GB FP8 multi-host (Plan-623 lane) | fp8 runtime ≈ 314–328 GB experts+bf16 minor | 11.0 GB replicated or sharded per the paper's CP posture (+ bucket-cache per-rank replication) | fits 512 GB with headroom; the FP8-class variant question resolves at pull-time dtype verification |
| 4090 24GB | — | — | prefill-microscope only (chunked probes ≤128k); never a 1M posture |

Active-param estimate (top-8 routing): ~15–18B/token (8×25.2M expert slices + shared + dense-3 + attention/KDA + untied head over 154,880 vocab) — consistent with the card's 18B-active class.

### Probe findings that re-shape Phase 1 assumptions

1. **KDA = the deltanet lane's cousin, with deltas** — reuse CONFIRMED, but the forward skeleton must add: the −5.0 gate lower bound (sigmoid clamp), the short conv (kernel 4) over q/k/v, and the per-layer type dispatch (34/45 KDA).
2. **No-rotary attention + rope-interleaved indexer** — the plan's rotary row resolved; any RoPE machinery in the MLA path is indexer-only (interleaved half-rotation on 128-dim index heads).
3. **The trained indexer ships in-checkpoint** (wk/wq_b/weights_proj/k_norm + kpool compress gates) — Phase 4's PCA-floor arm can be validated AGAINST the shipped indexer's selection behavior on real captures (the paper's trained-together-baseline caveat made concrete).
4. **mHC is load-bearing** — the residual stream is 4-wide with sinkhorn-normalized per-layer mixing; NOT prose. Phase-1 skeleton needs it before any parity capture.
5. **swiglu_limit 10.0** — the Qwen3-family activation clamp; the riir-refine `clamp-emits-tail-error-term` rule applies verbatim when we clamp in a numerics contract.
6. **Storage dtype question — RESOLVED 01:58Z, pre-pull** — the probe's `--shard-headers` arm (ranged safetensors-header reads of shards 1/32/62, metadata-only, never tensor data) proves the experts STORE as F8_E4M3 with 1:1-paired fp32 scales; the earlier 2.04 B/param usedStorage read was HF LFS accounting, not bytes-on-tensor. Pull size ≈ **306–328 GB** (not 612–657); Phase-2 PQ2 planning starts from fp8-sourced experts (not bf16) — the requant path is fp8→ternary, and the per-block fp32 scales are the calibration reference the recipe can lean on.
7. **Untied head + 154,880 vocab** — lm_head is 4096×154880 ≈ 1.27B params (bf16 minor set) — a real PQ2 line item, not an embedding tie.
8. **MTP layer in-checkpoint** — records the future specdec cousin lane (i1253 shape); no Phase-1 action.

### Phase 0 disposition

- [x] Arch probe — LANDED (script + this section; run of record 2026-11-28T01:47Z)
- [x] Survey — LANDED (this section; four vehicles + GGUF ecosystem)
- [x] Memory budget — LANDED (this section; the paper datum confirmed by config)
- [-] Weights pull — stays OWNER-GATED (the ~306–328 GB verified-real pull (fp8+scale+bf16 storage measured at the shard headers 2026-11-28T01:58Z; HF usedStorage overstates ~2×) + SDXC1TB target = the owner's confirm; everything agent-side is now unblocked for Phase 1 re-claim per the lease law)
