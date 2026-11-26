# Plan 639 — i1253 T5: the GGUF-embedded MTP draft-block lane (specdec intake)

- [ ] Plan 639 — slices T5.1–T5.5 below; measured cells pre-registered here, executed in quiet windows only.

**Status:** FILED 2026-11-26 (session zed-i1253t3@amo, epoch 1795719635) — the lane breakdown for `.issues/1253` T5; claim slices individually, this plan is not a claim.
**Depends:** T2 (loader tolerance — LANDED `5d1e2cf`, Bench 1271) + T3 (model verified — cheap half LANDED `dc90cdaa`, Bench 1272; 4090 parity arm open with the GPU lane).

## The target (what T5 consumes)

The league checkpoint's **embedded draft block** — `blk.64.nextn.*` in the MTP twin
(`Ternary-Bonsai-2.5-27B-PQ2_0-mtp.gguf`, 7,492,338,944 B, sha256 `ededc9fc…`, T2's file of
record): 15 tensors at `blk.64`, 8 Q4_0 (attn_q/k/v/output, ffn_gate/up/down,
`nextn.eh_proj`) + 7 F32 (block norms + `nextn.enorm`/`hnorm`/`shared_head_norm`),
declared by `qwen35.nextn_predict_layers = 1`. T2 proved the loader ADMITS the file and
EXCLUDES the block (typed census + loud disclosure); T5 makes the block draft.

The card presets name the cell shapes (standard MTP-vendor postures): **NVIDIA 3 draft
tokens / min-prob 0.10** and **Apple 1–8 draft tokens** — the sweep axes for the
acceptance and net-throughput cells.

## Substrate map (consume, never fork)

| substrate | home | role |
|---|---|---|
| `MtpTwinCensus` + declare-or-refuse gate | `riir-infer-core/src/gguf_loader.rs` (T2) | the admission half — T5.1 extends, never bypasses |
| `DraftSource` + `run_head_spec` + the 2-stage runner | `riir-infer-dist/src/spec_fill.rs` | the verify machinery — K-draft verify rounds, acceptance ledger, net tok/s |
| `MtpHeadDrafter`/`BmthHead` | `riir-infer-dist/src/spec_mtp.rs` | the TRAINED-head drafter precedent (a different artifact — BMTH v1 bin, not the GGUF block); its acceptance harness (`examples/mtp_head_acceptance.rs`) is the measurement skeleton T5.4 adapts |
| qwen38_dspine draft-loader + CPU oracle | `riir-infer` (`qwen38_dspine` feature, Issue 1183) | the vendor-draft-head consumption precedent (load → oracle → accept/verify) |
| `forward_qwen_deltanet_ternary` / GPU ternary forward | `riir-infer-core` / `riir-infer-gpu` | the main-stack forward the verify leg runs |

**The one prior of record:** Bench 924 (i924) measured the TRAINED-head spec lane on the
CPU 2-stage pair at **5.35 spec vs 7.69 ordinary tok/s — measured NEGATIVE** at E[m] 1.73.
T5's bars must beat THAT posture's economics on the Metal decode lane (where the draft
block rides the GPU with the stack, not a host-side head), or the lane is
filed-and-parked like 1207.

## Slices

- [ ] **T5.1 — the draft block's typed admission** (`riir-infer-core`, opt-in feature
  `mtp_draft_block`): a typed `MtpDraftBlock` load alongside the main stack — parse the
  15 `blk.{n_main}.nextn.*` + block-norm tensors into the struct (dequant Q4_0 → the
  runtime's working precision, the same tolerance the main stack uses), extend
  `MtpTwinCensus` with a `draft_admitted` class (declared + admitted loads BOTH the main
  stack and the block; the disclosure line changes from "stays for the MTP lane" to
  "admitted to the draft lane"), feature-off byte-identical (the T2 census gate is the
  default posture; the feature arms the second read). Synthetic-gate battery: declared
  admits (typed fields pinned), undeclared still refuses, missing-tensor refuses loud
  (a file declaring layers but carrying a partial block is the torn-twin class).
- [ ] **T5.2 — the draft forward (CPU reference first)**: one nextn block step —
  `enorm`/`hnorm` + `eh_proj` compose the block input from (prev hidden, prev token
  embed), the block runs attention (own KV over draft positions) + FFN, then
  `shared_head_norm` + the main lm_head produce draft logits. Bit-parity law: the CPU
  reference is the oracle (the reference-keeps-the-job law); the G1 target the issue
  names — integer base-3 collapse verify — pins the draft-block math against the
  whitepaper §6's Lean-exact kernel precedent (the fold-order laws ride Issue 940).
- [ ] **T5.3 — the drafter seat**: `MtpDraftBlockDrafter` as a `DraftSource` over the
  typed block; K presets map to the card cells (K=3 + min-prob 0.10 = the NVIDIA card;
  K∈1..8 = the Apple card); greedy-draft by default (the acceptance harness's teacher
  posture when measuring head quality alone).
- [ ] **T5.4 — the acceptance + net-throughput cells** (`examples/mtp_head_acceptance.rs`
  adapted or a sibling example — the existing harness measures the BMTH head; the
  GGUF-block arm is a new example consuming T5.3): E[m] per verify round, per-slot rate
  beyond echo, decode tok/s (draft vs repeat-last vs ordinary) on the league prompt set
  (mtp_chat_tokens' transcript is the serving shape), **quiet-window only** (amo/bmo
  exclusive; the load-class disclosure rides every number).
- [ ] **T5.5 — the Metal decode lane cell + the GOAT adjudication**: the draft block on
  the Metal serving posture (one extra block forward per verify round, micro-batched
  verify per spec_fill's design), net tok/s (accept-weighted) vs plain decode at the
  card presets. **Pre-registered bars:** (a) correctness — draft-token agreement with
  the model's own greedy continuation ≥ the 924 trained-head posture's 0.4353 chat
  per-slot rate (the embedded block should CLEAR its trained approximation; below it,
  the block intake is wrong before economics are discussed); (b) economics — net tok/s
  ≥ plain decode × 1.05 at the NVIDIA card preset on the Metal posture (the 924 CPU
  prior is negative; the embedded block must beat parity by a real margin or the lane
  parks); (c) G4 — the draft step alloc-free at the serving posture. A miss is a
  recorded NEGATIVE with the numbers; no retune-and-retry without a new pre-registration.

## Falsifiers / honest-exit conditions

- The 2.5 twin's draft block does not exist in the PLAIN league file (7,253,354,848 B) —
  cells run on the MTP twin only; the plain file is the control (draft lane refuses on
  it, loud).
- If T5.2's CPU reference cannot reach agreement-class quality (bar (a)) even
  teacher-forced, the block's post-final basis mismatch (the spec_mtp disclosure's
  basis law, measured for the BMTH head) applies to the embedded block too — record the
  NEGATIVE and park; do not build T5.3/T5.4 on a dead drafter.
- The measured cells need a QUIET box (exclusive rig job); a contended-box number is
  not a verdict — this plan refuses to cite one.

## Neighbors (not this plan)

riir-ai Issue 1263 (league re-pin + re-bout, owns T3's 4090 arm + T4); Issue 1183's
paused DSpine lane (the OTHER draft head — qwen3.8-target, paused by verdict; this plan
is the Bonsai-target lane the 1183 verdict ranked above it).
