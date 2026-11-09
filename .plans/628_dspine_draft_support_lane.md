# Plan 628: the DSpine b16 draft-model support lane (tencent/Qwen3.8-27B-DSpine-b16 → riir-infer specdec)

**Status:** Filed 2026-11-08 (session zed-m3-dspine-file@m3, epoch 1794128516) — CLAIMED zed-amo-1183@amo. **PROGRESS 2026-11-09: Phase 0 DONE (T0, vendor pin 6450af7 — the README "Coming soon" was stale, the tip carries 27B) · Phase 1 DONE (checkpoint blake3 53d18516…, on-disk at ~/uzu27b/models/Qwen3.8-27B-DSpine-b16/; loader landed with T2) · Phase 2 COMPLETE (T2 `c8de365` oracle + the G1(a) vendor-MLX golden diff, [Bench 959](../.benchmarks/959_i1183_t2_golden_diff_vendor_mlx.md): idx/tokens bit-exact vs the vendor walk; the diff caught the sender-base score bug + the transposed TG read + a green-zero fixture — all fixed/regression-tested/guarded) · T5 DECISION 2026-11-09: the head-only gate read mean_j 1.000 < 2.5 — the vendor-artifact lane CLOSES like uzu; Phase 3/4 + T6 are DEAD with it (re-open only on a trained head clearing the gate); Path-B = riir-train Issue 679 (`c15257e6`); [Bench 960](../.benchmarks/960_i1183_t5_head_only_acceptance_path_b_routes.md) lands with the record cells.** Tracks [Issue 1183](../.issues/1183_dspine_b16_draft_model_support.md); the gates (G1–G4), the boundary split and the decision gates live there.
**Claimed:** zed-amo-1183@amo (amo), 1794154200, expires 2026-11-10T16:00:00+00:00

## Ground truth at filing (re-verify at T0)

- **DSpine** = parallel speculative drafter; the causal chain unfolds over DEPTH: gated adjacent injection (immediate predecessor only) after every layer; block drafted in one pass; block size 16 (`b16` = BLOCK size — the dtype is bf16), SW 8192, 5 layers, ~2B params, Apache-2.0. Sources: [paper](https://arxiv.org/abs/2609.36173) · [project page](https://dspine-project.github.io/) · [repo](https://github.com/tencent/dspine) · [27B checkpoint](https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16) (2026-11-07).
- The repo's supported-models table still says Qwen3.8-27B "Coming soon" (README stale vs the HF drop); only the **Qwen3-8B** draft is code-documented upstream — T0 reconciles the 27B config against the 8B reference.
- Serving = SGLang 0.5.22, dynamic verification referencing DSpark, ReplaySSM (linear-attn + f32 SSM state) — the 27B target is the hybrid GDN model our lanes already serve.
- Motivation: the riir-ai plan 614 Phase-3 M5 gate needs acceptance L_eff ≥ 4.78–10.3 (Bench 1162); best measured head 2.83. DSpine's claimed 4.5–7.2 (cross-fixture) is the first family plausibly in range — T5 decides on OUR target.

## Phases

### Phase 0 — vendor ground truth (Issue 1183 T0)
- [ ] Clone `github.com/tencent/dspine` into `.raw/dspine` (research clone, `rm` when done); pin the commit SHA into Issue 1183
- [ ] Read + record as a PROGRESS block: config.json field map; the injection gate/message/write-back algebra; the block-attention causal posture (project page says block-causal — DFlash2 was non-causal `is_causal:false`); the transfer-space projections (exact target tensors — LM-head output embeddings?); the transition-cache scoring form; the ReplaySSM draft shape + replay-at-verify semantics; the dynamic-verification rule (how many of the 16 positions verify per cycle, when it adapts)
- [ ] Reconcile the README "Coming soon" vs the 27B HF card (the card is fresher); note any 8B→27B config drift

### Phase 1 — acquisition + loader (T1)
- [x] Download `tencent/Qwen3.8-27B-DSpine-b16` → weights-from-disk lane (never in-repo, A1/A8); blake3 pin into Issue 1183; owner-noted if the pull lands on SDXC — DONE 2026-11-08 (~/uzu27b/models/, blake3 53d18516…)
- [x] Loader: raw-u16 bf16 safetensors streaming (the `qwen38_dflash2_gpu.rs` posture — no f32 materialization; host peak = one tensor); config parse vs the T0 field map — DONE with T2 (`c8de365`)
- [x] `bytemuck::cast_slice` zero-copy views; `half` only where host math needs it (already an optional dep); never scalar-emulate bf16

### Phase 2 — CPU oracle (T2)
- [x] `qwen38_dspine.rs` CPU oracle: block forward, per-layer gated adjacent injection, unified transfer space, predecessor-conditioned decoding + transition cache (K×K pairs + last-write-redo), ReplaySSM replay — DONE `c8de365` (ReplaySSM is TARGET-side only per T0 — the draft carries no SSM tensors; out of draft scope by the T0 correction)
- [x] Piecewise parity fixtures vs the vendor reference (Bench-952 rig pattern): injection gate, transfer projection, transition cache, SSM replay — a stated tolerance per piece — DONE + the vendor replay [Bench 959](../.benchmarks/959_i1183_t2_golden_diff_vendor_mlx.md) (all PASS; two oracle bugs + a green-zero fixture caught and fixed)
- [x] Determinism-law audit (katgpt-rs Issue 940): every reduction feeding an accept decision pins its fold order — sequential ascending everywhere; the confidence mass pins the Triton contract formula

### Phase 3 — GPU port (T3) — Metal-first, cudarc twin second — DEAD WITH THE T5 CLOSURE (2026-11-09; re-opens only on a trained head clearing the gate, riir-train Issue 679)
- [ ] Metal (cubecl/wgpu on the dense lane): draft attention over SW-8192 KV rings; injection as a fused gather+scale per layer; the 256-pair transition scoring as one batched kernel; bf16→f32 dequant in-kernel
- [ ] cudarc twin (the riir-ai 755-T1 port order); CPU↔Metal↔CUDA parity on frozen captures
- [ ] Zero-alloc steady state (G4): preallocated rings/buffers; the 1179 T3 KV-discipline shape

### Phase 4 — verify-loop integration (T4 — GATED on 1179 T2/T3) — DEAD WITH THE T5 CLOSURE (same re-open condition as T3)
- [ ] k=16 rows per cycle on the dense verify path; the k-invariance pre-check (1179 T2's definition) at k=16
- [ ] Dynamic-verification posture from T0; KV + SSM-state rollback at the accept boundary; lossless rejection sampling (T=0.95 top-k=20)
- [ ] Feature `qwen38_dspine` opt-in, PROMOTE-ONLY env; feature-off byte-identical (league walk pins)

### Phase 5 — cells + records (T5/T6)
- [x] T5 head-only acceptance cell — DONE 2026-11-09, VERDICT: CLOSES (mean_j 1.000, j=1 all 200 cycles, G1-T5 PASS; the decision gate lives in Issue 1183; record cells → Bench 960)
- [ ] T6 e2e vendor-method cell riding 1179 T4's harness: FairnessManifest + size-gap disclosure; league WATCH-class row; bench number at landing — DEAD WITH THE CLOSURE (no promotion exists to land; Bench 960 + the HISTORY lane-close line carry the landing)
- [ ] AGENTS.md lane-table row + HISTORY.md entry at landing (doc-sync)

## Risks / open questions (honest)

- The 27B checkpoint is one day old and upstream's code table still says "Coming soon" — the 27B config may drift from the 8B reference the code documents (T0 reconciles before any loader work).
- **ReplaySSM is the least-documented piece** (SGLang flags, not the paper's core): whether draft-side SSM replay lives INSIDE the drafter or is a verify-side posture is a T0 call; budget slack there.
- Dynamic verification (DSpark-referenced) may change the effective verify-k per cycle — the G1 k-invariance reference must be defined against the vendor rule, never a fixed 16.
- Acceptance expectations are cross-fixture (SGLang/BF16 upstream cells vs our Q4_K_M GGUF target): the uzu T1 lesson says measure on OUR target, never print the card.
- Shared-file collisions with the in-flight 1179 session (see Status) — lease discipline before any shared edit.
