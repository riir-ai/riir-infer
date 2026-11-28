# Issue 1254: PoC cell — Plato's Key cross-model key transfer (closed-form rotation, functional recall swap)

**Status:** Open — P0a+P0b LANDED 2026-11-28 (branch `i1254-p0a-p0b`, commits `ea8c88b`+`19e1391`: the rope_freqs.weight loader arm wired + the PAIRED-spelling disposition measured on the real checkpoint + the P0 runtime probe bin; recipient load + scaled table (max ratio 32.0) + donor PQ2_0 load + layer_capture tap all GREEN; A1/A2 cells next — the A1 recipient download (Llama-3.2-3B-Base.Q8_0 3.42 GB, sha256 3a9d5b61...f2c6d, provenance recorded) is ON DISK at riir-train/data. A1 fact corpus LANDED 2026-11-28 (commit on this branch: `src/i1254_corpus.rs` + the tracked `src/i1254_a1_corpus.jsonl` + the `i1254_corpus_audit` receipt bin — 1387 facts / 16 relations, subject-wise blake3→SplitMix64 70/10/20 split, split 991/107/289, corpus digest 34b18a6a1c61e42fe25ad2f7eefcd14e90ac9bc85456b07f7038c7b44c297f13, split digest 19573a87ce2100078dc346dd16b6c6ea4ffa948e2eae6e239a7c1dc5b8f44fdc, record `.benchmarks/1279_i1254_a1_corpus_receipt.md`; also repairs the P0a commit's broken Cargo.toml (unresolved e0715b1 conflict markers + absolute dep paths committed in ea8c88b — the manifest now parses and is fully relative). Next: donor band capture + PCA r=256 Procrustes fit + REPLACE-hook eval cells — GPU-queue-gated behind the 4090 census job)
**Claimed:** zed-2611a378@4090, 1795902000, expires 2026-11-29T06:00Z
**Lease closed:** zed-m3-research-lane, 1795663960, was expires 2026-11-28T03:32Z — lapsed unworked (m3 dark through the window); re-claimed same hour by the 4090 housekeeping continuation per the lease law (the 03:40Z verify-only wake's claim-if-actionable arm executed early in-turn at the arbiter's execute-now order; push is the arbiter)

Filing record: `../.research/044_Platos_Key_CrossModel_Key_Transfer.md` (distillation + prior art + falsifiers). This issue is the runnable PoC cell.

## Verdict record (2026-11-26)

Delegated verdict, session `5b062d19-6a0c-4e89-a0a2-ae7c7477416d` (2 rounds,
final AGREE): the owner-law text decides it — "no new gemma-2-2b execution
cells anywhere" was written to stop the offline-research-cell reading; the
carve-out covers gemma2 architecture lanes serving live products (laya,
clef-ANE), not the retired checkpoint. Arm B taken in full with the reviewer's
additions (RoPE-scaling disposition above, the baseline recall gate, download
provenance + license note). A rejected, not deferred. Unchanged: A2, A3,
G1/G3/G4 thresholds, PC-space controls, rig routing (`--exclusive --est`).

## Claim (pinned)

A closed-form orthogonal-Procrustes map (rotation + one global scale,
mean-centered, fit on paired-recall facts), frozen as a BLAKE3 artifact and
applied as a single hidden-state REPLACE at the recipient's subject-token
layer, transfers factual recall across two frozen LLMs — adjudicated on THIS
stack's IN-RANGE cross-arch pair (Llama-3.2-3B BASE recipient ×
Ternary-Bonsai-2.5-27B donor) against the stack's own cross-arch negative
(riir-train Benches 424–427
`424_canon_p3_rust_discrimination` … `427_canon_p4_recipe_d_length_matched`:
hidden-state agreement structurally ~0 for steering directions; katgpt-rs's
own 424–427 are unrelated benches).

## Why here

riir-infer owns model-based inference substrate (BOUNDARY.md); the cell consumes
on-disk checkpoints + the CPU/GPU forwards. The generic rotation fit already ships
upstream (`katgpt_spectral::procrustes`, Bench 562 GOAT) but katgpt-spectral is not
in this repo's dep allowlist — the cell is TEST-LOCAL first (one-sided Jacobi f64,
verbatim precedent: `katgpt-attn/tests/issue_925_procrustes_probe.rs`); a production
promotion adds the allowlist row + BOUNDARY.md edit in the same commit as the dep.
No new linear algebra, no training, no weight writes anywhere.

⛔ **Scale discipline (verdict round 1):** the paper's validated range is 1.5–27B.
A below-floor null cannot separate architecture from scale, so it can never extend
the 424–427 negative — the ADJUDICATION runs on the in-range pair; the small pair
is a smoke arm with its null pre-registered INCONCLUSIVE-below-floor. The 27B
forwards are heavy work: they route to the 4090/amo rig queue
(`agent-mail rig run --est`); m3 never executes them.

## Cell design

⛔ **Architecture law (verdict round 2):** a model's architecture is read from
`general.architecture` in the GGUF header (probe: `py scripts/plan402_gguf_probe.py
<file.gguf>` from riir-train), NEVER from the filename. Header-verified 2026-11-26:
`Qwen3.8-27B-DFlash2` is `dflash` — a 5-block NON-CAUSAL draft head that cannot
recall facts and has no subject-token layer; `Ternary-Bonsai-2/2.5` are `qwen35`
(64-block GDN hybrid). DFlash2 / dspark / mtp files are NEVER recipients or donors.

**Arms (in-range primary first; heavy arms route to 4090/amo via the rig queue):**

- **A1 — PRIMARY adjudication (cross-arch, in-range): AMENDED 2026-11-26,
  delegated-verdict session `5b062d19` round 1 AGREE — the recipient of record
  is `Llama-3.2-3B` BASE (dense llama, d=3072, 28 blocks, GQA 8 kv heads,
  F16 GGUF, HF-fetched with sha256 + source URL recorded — the pre-registered
  download arm). The originally-adjudicated recipient
  (`riir-train/data/gemma-2-2b-it-f16.gguf`) is REJECTED, not deferred: it is a
  new gemma-2-2b execution cell, which the riir-infer owner law (2026-10-26,
  "no new gemma-2-2b execution cells anywhere") forbids under its text, and
  the gemma archive stays unrestored. Arm B also fixes the IT-vs-BASE protocol
  deviation (the paper used BASE models). × donor
  `riir-train/data/Ternary-Bonsai-2.5-27B-PQ2_0.gguf` (`qwen35` GDN hybrid,
  packed ternary, d=5120, on disk). Both members sit inside the paper's
  validated 1.5–27B span (3B / 27B). License: Llama 3.2 Community License —
  LOCAL USE ONLY, never a shipped artifact (the license note rides Research
  044).
- **A1 protocol addition — cross-dim fit:** llama d=3072 ≠ qwen35 d=5120. Fit in
  a SHARED top-r PC space (r=256 — exactly the paper's compression arm, which
  retains 74%): PCA each side's keys to r dims, square r×r Procrustes in that
  space, swap = project → rotate → reconstruct with the recipient's OWN PCs +
  training mean. Semi-orthogonal full-dim fit is the fallback arm. Layer
  selection uses DEPTH PERCENTAGE (22–38%), never absolute indices (28 vs 64
  blocks). ⛔ **Leak law:** the PCs and training means are fit on the TRAIN
  split of the paired facts ONLY — fitting them on all facts leaks test-set
  geometry into the swap and inflates transfer.
- **A1 controls are defined INSIDE the same PC space** (no identity map exists
  between 3072 and 5120 raw dims): identity control = the r×r identity through
  the identical project→reconstruct path; random-rotation control = a Haar-random
  r×r orthogonal matrix through that same path. G1's "≥ 2× identity" compares
  against the PC-space identity, never a raw-dim one.
- **A2 — same-architecture in-range control (isolates training variance):**
  recipient `Ternary-Bonsai-2-27B-PQ2_0.gguf` × donor
  `Ternary-Bonsai-2.5-27B-PQ2_0.gguf` — both header-verified `qwen35` 64-block,
  same PrismML Bonsai lineage, different training generations (2.5 = agentic
  retrain of Qwen3.8-27B per riir-train `data/BASE_MODELS.md`; Bonsai-2's exact
  base is a cell-time T0 check — if it shares 2.5's base, relabel the arm
  same-base/different-post-train). Same d=5120, square full-dim fit.
- **A3 — below-floor smoke (INCONCLUSIVE-below-floor by pre-registration):**
  recipient `Qwen3.5-0.8B-Base-Q8_0.gguf` × donor `MiniCPM5-1B-F16.gguf` — ⛔ the
  MiniCPM GGUF is NOT on disk (only `minicpm5-tokenizer/`; the riir-train
  AGENTS.md LOCAL-KEEP row is stale vs disk, observed 2026-11-26). Restore first:
  `cd ../riir-train && ./scripts/data_backup.sh pull data/MiniCPM5-1B-F16.gguf`.
  A3 results never feed the G1 verdict.

**Common protocol:**
- **Facts:** 16-relation set (capitals / authors / currencies / …), subject-wise
  train/val/test split; paired-recall filter = both models answer correctly on the
  direct prompt. Target ≥ 2k pairs (paper used < 5k training facts).
- **Fit:** mean-center → orthogonal Procrustes (test-local Jacobi, f64 internals,
  deterministic — tie rule pinned) → global scale s by spread match (NOT
  least-squares; the paper shows LS scales underperform — verified by the
  scale-law arm below, since that claim leans on the blog alone).
- **Swap:** donor key = hidden state at last subject token, layer ℓA; replace the
  recipient's state at layer ℓB, same position, with s·R·h_donor, then continue the
  forward. ℓA/ℓB selected on validation transfer over the 22–38% depth band.
  Forward hook (the Issue-395 `PostLayerHook` shape): the recipient REPLACE and
  the recipient-side key CAPTURE both ride one seam — the harness drives the
  token loop, so it arms the hook per call (armed only at `pos ==` subject
  position; `NoHook` monomorphizes away). A1's replace/capture point is the
  llama DENSE path (the `forward_llama_hooked` fork, this cell's own seam);
  the Bonsai donor side is CAPTURE-ONLY (qwen35 GDN hidden-state readout —
  the `HybridForwardScratch` harvest precedent, sink_audit; the ternary-path
  capture seam is its own G3-gated commit — the harness refuses loud at the
  donor step until it lands, never a silent half-run). The optional
  reverse-direction arm is what puts a MUTATE hook on the qwen35
  packed-ternary path — cost named, not assumed shared.
- ⛔ **RoPE-scaling disposition (verdict round 1 finding):** llama3 GGUFs may
  carry `rope_freqs.weight` (per-dim frequency factors). The loader handles it
  — load + bake into the recipient's RoPE frequency table (`freq[i] · w[i]`,
  llama.cpp semantics; absent tensor or all-ones weights ⇒ byte-identical
  table) — and a shape mismatch REFUSES LOUD, never a silent fallback. The
  A1 run also clamps the recipient KV-cache context to the actual sequence
  budget (`--ctx`, default 512) — llama-3.2-3B's GGUF `context_length`
  (131072) must never size the cache.
- ⛔ **Baseline recall gate (verdict round 1 addition):** before ANY transfer
  number, the recipient must answer the direct prompts on its own at ≥ the
  paired-recall floor (the paired pool ≥ the run's min-pairs); a recipient
  whose baseline recall is broken (tokenizer mismatch, rope mis-applied) is a
  REFUSAL naming the phase, never a transfer table.
- **Download provenance:** the recipient of record is
  `https://huggingface.co/RichardErkhov/1024m_-_Llama-3.2-3B-Base-gguf`
  → `Llama-3.2-3B-Base.Q8_0.gguf` (3.42 GB; the straight mirror of the
  official llama.cpp conversion; sha256 recorded at the run). Disclosed
  deviation: Q8_0 weights vs the paper's full precision — no F16 BASE GGUF
  mirror exists (checked 2026-11-26: the only BASE GGUF repos carry quants
  up to Q8_0; the Instruct F16 mirrors violate the BASE protocol);
  per-weight 8-bit noise sits ~2 orders below the 15% effect bar, and the
  activations stay f32 either way — recorded in the run report, never
  silently. The recipient config carries **llama3 rope_scaling (factor 32,
  llama3 type)** — the `rope_freqs.weight` loader arm this cell added is
  load-bearing (the verdict round-1 finding, confirmed at the source
  config). Llama 3.2 Community License — LOCAL USE ONLY, never a shipped
  artifact.
- **Donor side (landed `af50b72`, superseding the "seam pending" note):**
  the ternary-path capture ALREADY existed — the ternary forward's
  `layer_capture` tap (the fork's `l_out` snapshot: post-layer residual per
  layer, copies only when a buffer is supplied, unarmed byte-identical by
  construction). The harness's P3 = donor body detect (globals-tensor dtype
  discriminator, sink_audit's law, before any load) → PQ2_0 ternary load →
  Issue-929 ctx clamp → per-fact prefill with `layer_capture` armed at the
  subject position (the 22–38% band) → donor greedy-recall paired filter →
  recipient band capture via the Issue-395 hook. P4 = deterministic
  70/10/20 split, PCA r=256 per band side, Procrustes per (ℓA, ℓB) combo,
  VAL-split layer selection. P5 = the test arms + the machinery ceiling +
  the G1 arithmetic verbatim. The A2-class MUTATE hook on the ternary path
  stays a named future seam (cost named, not assumed shared).
- **Arms per pair:** transfer (primary) · controls: identity map, random rotation,
  shuffled donor keys · self-swap ceiling at ℓB · mKNN-selected-layer contrast ·
  **scale-law arm** (spread-match scale vs least-squares scale — if LS ≈ spread,
  record the divergence, no demote) · chained-rotation spot check (one triple,
  A1 only) · dark-memory report-only arm (donor-recalls/recipient-misses: ICL /
  paraphrase / rank-1-of-≤50 checks; A1 only).

## Gates (pre-registered)

- **G1 PASS:** A1 (in-range) test-transfer ≥ 15% absolute AND ≥ 3× random-rotation
  control AND ≥ 2× identity control; controls clean. A2 reported alongside
  (architecture-vs-training read); A3 smoke, no verdict weight.
- **G1 NEGATIVE:** A1 transfer ≤ random-rotation control + noise ⇒ the cross-arch
  negative (riir-train Benches 424–427) EXTENDS to functional key transfer at
  validated 27B scale — record NEGATIVE in Research 044's PoC Addendum, demote,
  close. ⛔ A3's null can NEVER drive this (scale-confounded by construction).
- **G3:** hook-compiled-unarmed forward bit-identical (FNV/argmax walk pins, the
  env-switch law's shape) — per forward path.
- **G4:** swap apply alloc-free at serving shape (one d×d matvec + one copy).
- Dark-memory arm: REPORT-ONLY (paper's 18% is on 1.5–27B pairs; no bar until A1).
- Execution routing: `agent-mail rig run` to 4090 or amo, `--est` on every job,
  `--exclusive` for the 27B forward sweeps; m3 authors and adjudicates only.

## Outputs

`.benchmarks/<next>_i1254_*` cell record (verdict + raw tables); PoC Addendum in
Research 044; tiers re-graded there on the result (the frozen donor-key-table
serving framing stays unclaimed until a PASS).

## Refs

Research 044 · katgpt-rs `.docs/05_adaptation/canonical_intent_space.md` + Bench 562
(`562_katgpt_canon_goat.md`, katgpt-rs) · **riir-train** Benches 424–427
(`424_canon_p3_rust_discrimination` … `427_canon_p4_recipe_d_length_matched`) ·
riir-infer Issues 1189 (closed_form_write), 1208 (RIMSK artifact law), 1241 (census
push seam), 1250 (readout_screen) · Hazy Research blog 2026-11-25 · the llama
cell seam: `src/transformer/llama.rs` (`forward_llama_hooked`) +
`src/rope.rs` (`RopeFreqTable::new_weighted`) + the `rope_freqs.weight`
loader arm in `src/gguf_loader.rs`.
