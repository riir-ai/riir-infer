# Research 043: Ternary Bonsai 2.5 27B — release distill + league retarget

> **Source:** PrismML "Introducing Ternary Bonsai 2.5 27B" (prismml.com/news/bonsai-2-5-27b, 2026-11-24) + whitepaper `bonsai-2-5-27b-whitepaper.pdf` (PrismML-Eng/Bonsai-demo @ main) + model card `prism-ml/Ternary-Bonsai-2.5-27B-GGUF` (Apache-2.0) + collection `prism-ml/bonsai-25-27b`
> **Date:** 2026-11-25
> **Status:** Active — retarget executed in-session; follow-up cells filed (Issue 1253, riir-ai Issue 1263)
> **Related Research:** 026 (FOSCAR fold), 031/034/037/039 (specdec cluster), 038 (looped NVFP4 lane intel)
> **Related Plans:** none (retarget, not a new mechanism; follow-ups are issues)
> **Classification:** Public

---

## TL;DR

Bonsai 2.5 27B is an **agentic retrain of the Bonsai-2 checkpoint family**: same Qwen3.8-27B hybrid-attention architecture, same ternary g128 blockwise-Hadamard format (1.76 bpw, 5.93 GB PTQ1_0), same arch tag our loader keys on — "The ternary format, storage accounting, and rotated weight basis are unchanged from Bonsai 2". It fixes exactly what Bonsai 2 was worst at (agentic coding 63% → 90% of FP16; tool calling 67.0 → 77.2 vs FP16 77.7) and ships an MTP draft head in every packing. **The league checkpoint target of this repo moves from `Ternary-Bonsai-2-27B-PQ2_0.gguf` to `Ternary-Bonsai-2.5-27B-PQ2_0.gguf`** — executed same-session (data pin + every live default), with the four riir-ai model-pinned parity tests kept on the predecessor until the re-pin cell runs.

**Distilled for riir-infer (inference substrate, public):**
the retarget is format-identical, so the `bonsai2_hadamard` lane loads 2.5 unchanged; the genuinely NEW runtime facts are (a) an in-GGUF MTP draft head (1.5–1.8× NVIDIA decode, RTX 4090 204 tok/s = 1.83×, prefill cost 9–23%) our specdec lane can consume, and (b) Q2_0 ternary is now MERGED in stock llama.cpp (#29600) with FWHT kernels upstream per backend — the fork-only moat PrismML held at Bonsai-2 release is gone, which changes what our league's opponent row can run.

---

## 1. Release facts (whitepaper + card, verified against the file list)

| Item | Value |
|---|---|
| Released | 2026-11-24, Apache-2.0 (+ NOTICE.txt) |
| Base | Qwen3.8-27B hybrid attention (~75% linear GDN / ~25% full attn), SwiGLU, RoPE, RMSNorm |
| Params | 24.35B language (64 blocks) + 0.47B vision tower (27 blocks) + 2.54B emb/lm-head = 27.36B |
| Context | 262K |
| Format | ternary g128, FP16 group scales, blockwise Hadamard basis, folded offline — **unchanged from Bonsai 2**; 1.76 bpw; 5.93 GB PTQ1_0 |
| GGUF files | `PQ2_0` / `PTQ1_0` / `Q2_0` + `-mtp` twin of each + `mmproj-BF16` / `mmproj-Q8_0` + llama.cpp MTP presets (cuda/metal/sycl/hexagon) |
| Reasoning | `xhigh` (default) / `medium`; **`low` unsupported** (behaves ≈ xhigh) |
| Sampling (thinking) | temp 1.0, top-p 0.95, top-k 20, min-p 0 (llama.cpp defaults min-p 0.05 — pass `--min-p 0`), presence 0, rep 1.0 |
| Sampling (instruct) | temp 0.7, top-p 0.80, top-k 20, min-p 0, presence 1.5 |
| MLX | text-only affine 2-bit g128 + bundled loader applying the Hadamard transforms; KL 4e-6…3e-5 vs llama.cpp |

Quality vs Bonsai 2 (thinking, xhigh, pass@1): tool-calling avg 67.0 → **77.2** (FP16 77.7); AppWorld 51.2 → **90.5**; agentic coding avg 37.9 → **54.2** (90% of FP16; DeepSWE 0 → 27.0, SWE-bench-V 60.8 → 75.6, TB2.1 52.8 → 60.1); general 5-category 81.6 → **82.3** (FP16 84.4). Beats every GSQ-RCO IQ2/IQ3 and Unsloth UD-IQ2_XS build under 10 GB; IQ3_XXS (1.7× larger) still edges coding avg 55.5.

Speed (plain decode, PQ2_0, current PrismML runtime vs the Sept-2026 Bonsai-2 release build): RTX 4090 98 tok/s (1.09×), H200 184 (1.56×); prefill pp2048 4090 3444 (1.06×), PTQ1_0 3452 (2.06×). With MTP: 4090 **204 tok/s (1.83×)**, H200 306 (1.53×). Vs Bonsai-2 release day with the kernels in testing: 2.3–3.0× decode on 6/7 NVIDIA GPUs (1.92× on the 4090).

Upstream status (changes the league opponent row): **Q2_0 Bonsai-27B-family support merged to stock llama.cpp as PR #29600**; FWHT kernels upstream CPU #27779, CUDA #29096/#29100, Metal #29094/#29095/#29602, Vulkan #29772, SYCL #29243/#29605, MUSA #30167. PTQ1_0 format+CPU under review #29672, CUDA #29676. Stock llama.cpp Q2_0 matches the fork (KL 3.6e-4, identical greedy text) at 109 vs 150 tok/s on H200. PQ2_0 + the fast kernels stay fork-only.

Kernels in testing (not yet released): chunk-staged Gated DeltaNet prefill; fused norm+Hadamard PTQ1_0 path; Hopper PTQ1_0 prefill; PQ2_0 multi-column mat-vec for single-token decode on A100/B200 (B200 sums in a different order — mean KL 1.2e-4, disclosed); spec-decode verify kernels (draft head reads the prompt alongside the main model; Metal + SYCL multi-token verify); **Apple base-3 collapse kernel for PTQ1_0 spec-dec verify (2–4 tokens) — "computes the same integer dot product (proved exact in Lean)", +26/36/72% verify throughput at 2/3/4 tokens on M5 Pro, opt-in because float accumulation order differs**.

## 2. Distillation — what this release means for each lane

**Loader / format lane (`bonsai2_hadamard`):** no format change → the feature name stays correct (it names the Bonsai-2 FORMAT family, which 2.5 shares). The base (non-MTP) GGUFs load through the existing `qwen35`-arch + `prism.hadamard.*` contract unchanged. The `-mtp` twins add draft-head tensors our loader has never seen — tolerance for unknown tensor names is the open question; the league default is the non-MTP file so nothing gates on it (Issue 1253 T2 checks the `-mtp` file explicitly).

**League lane:** the checkpoint target moves. Consequences: (a) every live default in this repo now names `Ternary-Bonsai-2.5-27B-PQ2_0.gguf` (env `BONSAI_GGUF`/`BONSAI2_GGUF` overrides unchanged); (b) riir-train carries the new pin (BASE_MODELS VERIFIED row + BACKUP_MANIFEST pin_sha256) with the 2-27B PQ2_0 retained as a league-predecessor pin — it stays re-downloadable and satisfies the four predecessor-pinned parity tests in riir-ai; (c) the league rows (4090 tg128 0.970×, m5ultra decode-resident bout, prefill FLIP rows) are **per-model measurements** — they do not transfer to 2.5 and the re-bout is owed (riir-ai Issue 1263). The opponent baseline moved too: stock llama.cpp can now run the Q2_0 band upstream, and the fork's testing kernels change the fastest-plain-setup math the whitepaper's Table 7 documents.

**Specdec lane:** the MTP draft head is PrismML-built, rides IN the GGUF, and their presets (3 tokens, min-prob 0.1 on NVIDIA; Apple 1–8 tokens) are published. Our specdec cluster (Research 031/034/037/039; `qwen38_specdec` Issue 1179 paused; `mtp_head_acceptance` + `mtp_chat_tokens` examples already in riir-infer-dist) gains a real, already-distributed draft head for the league model — the 1.83× on the exact GPU our 4090 league row uses is the headline. Their base-3 collapse verify kernel (Lean-exact integer dot product, opt-in over accumulation order) is the same FV discipline our `katgpt-core`/`riir-chain` proofs carry and is the natural G1-exactness precedent for any ternary verify arm we build.

**Fusion idea (novelty TBD — issue, not plan):** folded ternary × MTP-verify: our folded-PQ2_0 decode path + their draft-head tensors + our Issue-940 pinned fold order, gating accept with a base-3 exact verify instead of a float tolerance — the two lossy-surface lessons (lossy-surface promotion rule; Issue-940 law) both bite here, which is why the exact-integer verify is the interesting half.

## 3. Retarget execution record (this session, 2026-11-25)

- **riir-train:** downloaded `Ternary-Bonsai-2.5-27B-PQ2_0.gguf` from `prism-ml/Ternary-Bonsai-2.5-27B-GGUF` into `data/`; BASE_MODELS.md VERIFIED row added; BACKUP_MANIFEST.tsv pin row added (sha256 == upstream LFS oid, verify command in the row); data/README.md KEEP set updated; AGENTS.md Models rows updated (2.5 = league model, 2-27B = league predecessor, both LOCAL-KEEP pins).
- **riir-infer:** every live default naming `Ternary-Bonsai-2-27B-PQ2_0.gguf` swapped to `Ternary-Bonsai-2.5-27B-PQ2_0.gguf` (src/bin + tests + root `examples/` incl. the twt_* lane and `requant_q4_prefill`'s requant parent + riir-infer-gpu tests/bins/src + riir-infer-dist tests/examples + root Cargo.toml comment + lane scripts) — all verified model-agnostic (self-parity / walk / A-B / census cells; the only in-source pin on the folded lane, `g1_032`'s `PREROT_PIN_P2048`, is keyed to the OLD gen-1 `Ternary-Bonsai-27B-Q2_0.gguf` pre-rotation file and stays untouched). Historical records (.benchmarks/.research/.plans/.issues/HISTORY/docs + committed fixtures like `Ternary-Bonsai-2-27B-PQ2_0.kvdiag.json`) deliberately untouched. `hisa_plan`'s RESTORE_POINTER updated to the upstream pin wording (a pin is re-downloaded from HF, never via `pull`).
- **riir-ai (league lane):** unpinned league test defaults + lane scripts swapped; **four model-pinned files kept on the predecessor** (`bench_980_t4` EXPECTED_TOP1 11751, `g1_bonsai2_metal_parity` `== 11751`, `bench_947` FA/G3 FNV walk pins, `g1_bonsai2_rotation_parity` fork-parity 11751) — swapping model+pin together requires a measured 2.5 cell; filed as riir-ai Issue 1263. Refine's kernel_opt corpus entry + mining queue snapshot untouched (mined-rule provenance / historical snapshot).
- **Verification:** header probe (`plan402_gguf_probe.py`) on the downloaded file → arch + geometry census; synthetic loader-contract gate `bonsai2_rotation_load` re-run green (fast local gate; real-file fork-parity G1 is the 4090-side cell, Issue 1253).

## 4. Verdict

**Tier: Gain (lane intel + retarget), no Super-GOAT claim.** This is a model release, not a new mechanism; the whitepaper's own techniques (MTP head, base-3 collapse verify) are consumption targets for lanes we already run, with published numbers, not novel primitives we'd ship. Prior-art check: no workspace note mentions bonsai-2.5 (grepped all `.research`/`.plans`); the specdec cousins (031/034/037/039) pre-date it and are the consumers, not blockers. MOAT gate: riir-infer public substrate moat — a current league checkpoint + loader compatibility + the specdec intake lane is exactly this repo's charter; the riir-ai league row + rethink serving posture ride their own repos' follow-ups.

**Closest shipped cousins:** `qwen38_specdec` (Issue 1179, paused — the MTP head is a re-open candidate for the Bonsai target), `mtp_head_acceptance.rs` / `mtp_chat_tokens.rs` (the acceptance harnesses already exist in riir-infer-dist), `bonsai2_hadamard` folded lane (loads 2.5 unchanged), riir-ai `bench_preflight`/league cells (re-bout owed).

## 5. Follow-ups (proof-path-bearing; Issue 1253 carries the full list)

- **T1 (done here)** header-probe + synthetic gate on the new checkpoint. **amo advanced T3's cheap half 2026-11-26 (riir-ai Issue 1264 + `b1d79dc4f`): loader contract GREEN, top-1 11751 " Paris" UNCHANGED at 2.5 (CPU + Metal agree), top-5 moved 4/5 — the model swap is forward-verified at the first-token class. ai-1264 also found the four predecessor-pinned parity gates stale at develop (q8-activation default vs float-era bars + the G3 fingerprint drift, candidate `ternary_q2_resident`/Issue 935) — that finding is the controlling record for any "gates green" claim.**
- **T2 — `-mtp` load census (DONE 2026-11-26)**: the `-mtp` twin loads correctly by design — PrismML ships the draft head as a nextn block (`block_count` 65 = 64 main + 1, `qwen35.nextn_predict_layers = 1`, draft tensors `blk.64.*` at Q4_0/F32), and our shipped nextn exclusion (Issue 742 T9.2, pinned test) keeps the main stack at 64; all types standard, the eager type walk passes loud-clean. Instrument: `examples/i1253_mtp_load_census.rs`. T5 needs NO new loader surface — the draft tensors stay readable by name via `tensor_slice`.
- **T3** real-model G1: fork-parity class cell on 2.5 (the bonsai2_rotation_load doc's "real-file validation is the 4090-side G1" arm) + the Issue-594-style first-token sanity.
- **T4** league re-bout on 2.5 (4090 CUDA + m5ultra rows; opponent = current fork + the upstream-Q2_0 stock lane; riir-ai Issue 1263 owns the riir-ai side incl. the four pin re-measures).
- **T5** specdec intake: consume the MTP head through `mtp_head_acceptance` at the card's presets (3, 0.1 NVIDIA / 1–8 Apple); pre-registered bar = acceptance-rate + net tok/s vs plain decode on the league prompts, G1 exactness via the base-3 integer-verify idea (fusion, opt-in flag).
