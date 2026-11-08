# Plan 628: the DSpine b16 draft-model support lane (tencent/Qwen3.8-27B-DSpine-b16 → riir-infer specdec)

**Status:** Filed 2026-11-08 (session zed-m3-dspine-file@m3, epoch 1794128516) — UNCLAIMED. Tracks [Issue 1183](../.issues/1183_dspine_b16_draft_model_support.md); the gates (G1–G4), the boundary split and the decision gates live there. T0–T3 + T5 are unblocked now (CPU/loader side); T4 verify-loop integration is GATED on Issue 1179 T2/T3 (open, actively in flight — vendor-exact numerics landed `d83ae45c`, active T1 lease zed-1b3207ea@amo to 11-10). ⛔ The specdec files are SHARED and dirty right now (sibling WIP on `qwen38_dense_cubecl.rs` + the i1179 probes): claim a lease on 1179/1183 before touching any shared file.

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
- [ ] Download `tencent/Qwen3.8-27B-DSpine-b16` → weights-from-disk lane (never in-repo, A1/A8); blake3 pin into Issue 1183; owner-noted if the pull lands on SDXC
- [ ] Loader: raw-u16 bf16 safetensors streaming (the `qwen38_dflash2_gpu.rs` posture — no f32 materialization; host peak = one tensor); config parse vs the T0 field map
- [ ] `bytemuck::cast_slice` zero-copy views; `half` only where host math needs it (already an optional dep); never scalar-emulate bf16

### Phase 2 — CPU oracle (T2)
- [ ] `qwen38_dspine.rs` CPU oracle: block forward, per-layer gated adjacent injection, unified transfer space, predecessor-conditioned decoding + transition cache (K×K pairs + last-write-redo), ReplaySSM replay
- [ ] Piecewise parity fixtures vs the vendor reference (Bench-952 rig pattern): injection gate, transfer projection, transition cache, SSM replay — a stated tolerance per piece
- [ ] Determinism-law audit (katgpt-rs Issue 940): every reduction feeding an accept decision pins its fold order

### Phase 3 — GPU port (T3) — Metal-first, cudarc twin second
- [ ] Metal (cubecl/wgpu on the dense lane): draft attention over SW-8192 KV rings; injection as a fused gather+scale per layer; the 256-pair transition scoring as one batched kernel; bf16→f32 dequant in-kernel
- [ ] cudarc twin (the riir-ai 755-T1 port order); CPU↔Metal↔CUDA parity on frozen captures
- [ ] Zero-alloc steady state (G4): preallocated rings/buffers; the 1179 T3 KV-discipline shape

### Phase 4 — verify-loop integration (T4 — GATED on 1179 T2/T3)
- [ ] k=16 rows per cycle on the dense verify path; the k-invariance pre-check (1179 T2's definition) at k=16
- [ ] Dynamic-verification posture from T0; KV + SSM-state rollback at the accept boundary; lossless rejection sampling (T=0.95 top-k=20)
- [ ] Feature `qwen38_dspine` opt-in, PROMOTE-ONLY env; feature-off byte-identical (league walk pins)

### Phase 5 — cells + records (T5/T6)
- [ ] T5 head-only acceptance cell (speed-independent; the decision gate lives in Issue 1183)
- [ ] T6 e2e vendor-method cell riding 1179 T4's harness: FairnessManifest + size-gap disclosure; league WATCH-class row; bench number at landing
- [ ] AGENTS.md lane-table row + HISTORY.md entry at landing (doc-sync)

## Risks / open questions (honest)

- The 27B checkpoint is one day old and upstream's code table still says "Coming soon" — the 27B config may drift from the 8B reference the code documents (T0 reconciles before any loader work).
- **ReplaySSM is the least-documented piece** (SGLang flags, not the paper's core): whether draft-side SSM replay lives INSIDE the drafter or is a verify-side posture is a T0 call; budget slack there.
- Dynamic verification (DSpark-referenced) may change the effective verify-k per cycle — the G1 k-invariance reference must be defined against the vendor rule, never a fixed 16.
- Acceptance expectations are cross-fixture (SGLang/BF16 upstream cells vs our Q4_K_M GGUF target): the uzu T1 lesson says measure on OUR target, never print the card.
- Shared-file collisions with the in-flight 1179 session (see Status) — lease discipline before any shared edit.
