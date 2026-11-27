# Plan 639 — i1253 T5: the GGUF-embedded MTP draft-block lane (specdec intake)

- [ ] Plan 639 — slices T5.1–T5.5 below; measured cells pre-registered here, executed in quiet windows only.

**Status:** FILED 2026-11-26 (session zed-i1253t3@amo, epoch 1795719635) — the lane breakdown for `.issues/1253` T5; claim slices individually, this plan is not a claim.
**Progress:** T5.1 + T5.2 LANDED 2026-11-27 (`a3b455cd` + `fbf10fe3`, zed-ecba9366@amo) — the typed admission + the CPU reference forward, synthetic gates green; **T5.3 LANDED 2026-11-27 (`70b62d0c`, zed-14567e6f@amo)** — the drafter seat + card presets + the greedy chain, gates 5/5; **bar (a) EXECUTED 2026-11-27 (below) — 0.1739 < 0.4353, the pre-registered falsifier FIRED; T5.5's bar (a) precondition FAILED, the lane records the NEGATIVE**; T5.4 (the acceptance/net-throughput example) is UNREACHABLE while the drafter cannot even echo+1 at bar quality on real text — the honest-exit clause applies.
**Verdict (delegated-owner, Claude, 2026-11-26 ~20:4xZ, session 464973ce): AGREE with conditions, both discharged same session** — (1) the basis contract ACCEPTED as pinned (norm-before-rotation flagged as the load-bearing order; the draft tensors' primal storage is header-verified data, not assumption); the open question (which hidden the fork hands over) RESOLVED by source read: `qwen35.cpp` norms FIRST then publishes `t_h_nextn` = `t_embd` from the SAME post-`output_norm` tensor — the module doc now carries the evidence; (2) the DEFECT (the per-call `inp_sa` `to_vec` allocation — bar (c) as written) FIXED + gated: `inp_sa` is a scratch field, and `tests/mtp_draft_step_g4_alloc` (own binary, counting allocator + liveness canary, 2/2) pins the step at ZERO allocations so it cannot come back.
**T5.4 intake UPDATE (2026-11-27, riir-refine housekeeping continuation lane on amo):** the 2.5 `-mtp` twin is NOW ON THE 4090 (job 01a21af5: curl'd to `E:\git\riir-train\data\Ternary-Bonsai-2.5-27B-PQ2_0-mtp.gguf`, 7,492,338,944 B, sha256 `ededc9fc0a8a…` == the plan's pin byte-prefix; the runner's bash is WSL — the `/mnt/e/...` spelling, learned over 4 retries: GNU `timeout` absent, `E:/...` not a bash path). The original route claim ("the twin downloads there, the test runs from bmo's own synced clone") is AMENDED by measurement: bmo has NO mtp twin either and the 4090's rig bash proved a hostile cell host (3 dead jobs); the download rode a 4090 shared job because HF pull speed there (~27 MB/s) beats an amo-side pull of 7.1 GB, but THE TEST runs on amo with `MTP_GGUF` pointed at a synced copy (the twin now LIVES on amo at `riir-train/data/` — restored HF-direct from the manifest pin's upstream spec at cell time, sha256 pin-checked; amo's HF pull measured ~13 MB/s vs the 4090's ~27, and the 4090 LAN/Tailscale path proved hostile for bulk — the "LAN pull at cell time" clause is superseded; the twin is a pinned, re-downloadable artifact, keep or delete freely).

**BAR (a) EXECUTED 2026-11-27 10:49 +07 (zed-f3efb14d@amo, direct contended run, NOT a rig cell):** `mtp_draft_step_gates --ignored` arm 4 (`draft_step_real_teacher_forced_agreement`), release, real twin on disk — **4/23 = 0.1739 greedy top-1 agreement vs bar (a) ≥ 0.4353 — the falsifier FIRED** (bit-identical ×2: 12.24 s then 11.78 s, `BARA_EXIT=0` both; exit 0 is the test's own `assert!(total > 0)`, not a bar pass — the bar reads the printed rate). The `gguf_loader` disclosure lines confirm the real admission (15-tensor blk.64 class, the draft lane consumes it). **Posture disclosure (the plan's contended-number rule):** amo at run time carried the 4 i1244 shard retries (load ~38/30, CPU ~99-100%) + this lane's capped footprint (`-j 8` build, `RAYON_NUM_THREADS=8` cell) — the plan refuses contended numbers for the MEASURED cells (T5.4/T5.5 economics), but bar (a) is a correctness RATE (deterministic numerics, bit-identical ×2 under load), not an economics claim; the falsifier's input is valid at any load. Cell wall ≈ 12 s + ~21 min one-time (twin pull + fresh `/tmp/i1253-bara` release build); the twin is retained on amo for any re-open cell.

**Diagnosis (why the block reads dead-on-arrival on random seeds):** the cell's walk seeds `0..24` positions with synthetic tokens `(i*7919+13) % vocab` (NOT a real transcript); the draft step consumes `h_prev` = the main stack's `hidden_copy` — the POST-`final_norm`, PRE-rotation-capture hidden (`ternary_forward.rs:1681-1687`: the norm runs at 1681, the copy at 1687 BEFORE the forward-rotate, so the hand-off is the final-normed PRIMAL residual stream). Suspects, re-stated against that fact: (1) the compose then applies `hnorm` to an ALREADY-final-normed hidden — a double-norm, where the fork's `t_h_nextn = t_embd` provenance implies the block consumes the final-normed tensor through `hnorm` as ITS OWN norm (i.e. either the hand-off should be the pre-`final_norm` residual, or `hnorm` should be skipped/identity given the file's gamma is not identity — unverifiable without the fork's gamma values); (2) the head leg norms with `shared_head_norm` (present in the file) — the fork may expect the main `output_norm` at this position, or no second norm at all; (3) the synthetic seeds cannot exercise a chat-trained nextn block's input distribution. The compose-order probe cannot see any of these (zeroed projections + one-hot compose make norm families invisible).

**Verdict: NEGATIVE recorded, honest-exit clause exercised** — per the falsifier section: below bar (a) the block intake is wrong BEFORE economics are discussed; record the NEGATIVE, do not build T5.4 on it, no retune-and-retry without a new pre-registration. The re-open shape is named (the three suspects above + a real-transcript seed), needs its own pre-registered cell.

**Routes (filing-is-not-routing, per the verdict):** T5.3 (the `DraftSource` seat) — LANDED; T5.4/T5.5 bar (a) real teacher-forced cell — EXECUTED 2026-11-27 (amo, direct contended run, rate 0.1739 < 0.4353 — falsifier fired, NEGATIVE recorded); bar (b)/(c) Metal cells — UNREACHABLE (bar (a) failed; the honest-exit clause parks the lane). The rig fallback job 0ca1ac85 is cancelled same-session (superseded by the direct run).
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

- [x] **T5.1 — the draft block's typed admission** (`riir-infer-core`, opt-in feature
  `mtp_draft_block`, LANDED 2026-11-27 `a3b455cd`): a typed `MtpDraftBlock` load alongside the main stack —
  the 15 `blk.{n_main}` tensors parsed into the struct (real names + geometry pinned from the actual 2.5
  MTP twin header 2026-11-27: 11 attention/FFN tensors under the STANDARD main-layer names + 4
  `nextn.*`; attn_q [5120 × 12288] = the Issue-594 q‖gate concat, eh_proj [10240 × 5120], per-head
  norms [256]; Q4_0 dequant through the loader's existing arm — never a second decoder),
  `MtpTwinCensus::draft_admittable` (declared + covered + trailing > 0), per-tensor shape pins vs
  config, torn-twin refuses LOUD naming the tensor, feature-off compiles to nothing (the T2 census
  gate is the default posture). Synthetic-gate battery `mtp_draft_block_gates` 6/6: declared admits
  (shapes pinned + the Q4_0 round-trip EXACT at the fixture's f16-exact scale), torn/clean/shape
  refusals, undeclared census; the real-file arm rides `MTP_GGUF` (SKIPs loud without the twin).
- [x] **T5.2 — the draft forward (CPU reference)**: LANDED 2026-11-27 `fbf10fe3` —
  `src/mtp_draft.rs` `mtp_draft_step` mirrors the fork's `graph_mtp`: wte row inverse-rotated to
  primal (the latent-lookup law), concat(enorm(embed), hnorm(h)) → eh_proj (primal in/out — the
  real twin's draft tensors carry ZERO hadamard weight_names entries, header-verified 2026-11-27),
  gated attention over the block's OWN KVCache (Issue-594 q‖gate split, per-head q/k norms,
  partial RoPE, GQA), post-attn norm → SwiGLU → residual anchored on the SAVED pre-norm value,
  head = shared_head_norm (else output_norm) normed on a COPY → forward-rotated → folded lm_head
  matvec (the `ternary_lm_head_into` mirror; `shared_head_head` present refuses loud — the dense
  arm is T5.4). The BASIS CONTRACT pinned in the module doc: the residual stream is PRIMAL (folds
  apply D·H at each folded matmul's input; the hidden variant returns the rotated head-basis copy
  — the caller recovers with `rotate_inverse_inplace`, the Issue-1111 recipe). THREE real bugs
  found + fixed: TWO by the compose-order probe pre-landing (the FFN residual anchor — norming
  h_prev in place without the pre-norm save fed the NORMED value into the final add — and the head
  norm destroying the caller's h_draft), the THIRD post-landing by katgpt-rs-88's fork source read
  (`c41b6bc5`: the block's OWN attn_norm was loaded + shape-pinned but never READ — the q/k/v
  projections consumed the un-normed compose output; invisible to every synthetic arm — the probe
  zeroes the attention projections, the fixture arms pin no attention-leg value; fixed + gated by
  a canary-verified v-only single-position arm that reads 2·e0 without the fix). Gates
  `mtp_draft_step_gates` 4/4 + the `--ignored` real teacher-forced arm
  (`MTP_GGUF`, bar (a) ≥ 0.4353 = the 924 prior): end-to-end determinism bit-identical ×2,
  primal-vs-rotated basis observable, enorm-first/hnorm-second compose order, attn_norm applied
  pre-qkv.
- [x] **T5.3 — the drafter seat**: `MtpDraftBlockDrafter` as a `DraftSource` over the
  typed block; K presets map to the card cells (K=3 + min-prob 0.10 = the NVIDIA card;
  K∈1..8 = the Apple card); greedy-draft by default (the acceptance harness's teacher
  posture when measuring head quality alone).
  LANDED 2026-11-27 `70b62d0c` (zed-14567e6f@amo, claim exp 2026-11-28T21:20Z folded on
  landing): `crates/riir-infer-dist/src/spec_mtp_block.rs` behind the opt-in dist
  `mtp_draft_block` feature (forwards the root crate's); `MtpDraftPreset`
  (nvidia_draft3 / apple(k), domain refusals) + `MtpDraftBlockDrafter` —
  `seed(h_final primal, last_landed, next_pos)` → `draft_chain()` greedy over
  `mtp_draft_step` (in-place h chain, the block's own KV discipline: committed-prefix
  rows persist, rejected tails rewritten, no explicit truncation); `propose` = d_0 echo
  + d_j = chain step j; **the 2-stage spec runner posture cannot feed `h_final`
  (tail-side) — the seat degrades to the lossless echo + `take_error` (the BMTH
  honest-basis disclosure); the lawful postures are the full-stack T5.4 teacher
  harness + T5.5 Metal cell, seeded per round**. min-prob = top-token mass 1/Z
  (ascending f32 fold, first-index argmax; a scalar normalizer read, never a softmax
  weighting; non-finite gates out loud). Gates `mtp_draft_block_drafter_gates` 5/5
  (required-features row rides the same commit — the green-zero law). Drive-by FIX:
  the `01932e53` rebase had duplicated the `attn_output` `dense_matvec` in
  `mtp_draft_step` (2× the matvec per draft step, K-multiplied in the chain); removed,
  T5.2 gates 4/4 unchanged. Post-mining note: `cargo refine fix` landed
  shrink-to-fit; the 2 `collapsible_if` let-chain collapses were manual (rulewright
  miss on the class, edition-2024 let-chains).
- [ ] **T5.4 — the acceptance + net-throughput cells** (`examples/mtp_head_acceptance.rs`
  adapted or a sibling example — the existing harness measures the BMTH head; the
  GGUF-block arm is a new example consuming T5.3): E[m] per verify round, per-slot rate
  beyond echo, decode tok/s (draft vs repeat-last vs ordinary) on the league prompt set
  (mtp_chat_tokens' transcript is the serving shape), **quiet-window only** (amo/bmo
  exclusive; the load-class disclosure rides every number). **PARKED 2026-11-27: bar (a) FAILED (0.1739 < 0.4353, falsifier fired) — the drafter cannot seed a chain on real forwards; this cell is unreachable until the re-open cell below clears bar (a) under its own pre-registration.**
- [ ] **T5.4-R (the re-open cell, PRE-REGISTERED 2026-11-27, zed-f021648e@amo):** the re-open is DIAGNOSE-then-REMEASURE, and the diagnosis leg is complete BEFORE this cell runs (recorded below) — the numerics stay byte-unchanged. The new cell is arm 5 `draft_step_chat_transcript_teacher_forced_agreement` (`#[ignore]`, release, real twin + real chat tokens):
  - **Token source:** `MTP_CHAT_TOKENS` env, default `../riir-train/data/bonsai_mtp/tokens.chat.tokens` — regenerated by the documented example (`cargo run --release -p riir-infer-dist --example mtp_chat_tokens`, the Bench 924 chat transcript family: 38 hand-written serving-shaped turns in `dist_stage::chat_turn`'s exact template, ~14k tokens); SKIP loud when absent, naming the regeneration command.
  - **Leg A (the bar leg, the 924 shape):** sliding 64-token windows over the chat file, stride 56, capped at 256 windows (n ≈ 250); per window FRESH main + draft caches, main forwards at pos 0..63, the draft step at pos 63 consumes (hidden@62 primal, tok@63) and its greedy top-1 is compared against the MAIN model's greedy top-1 at pos 63. **BAR (a)-R: Leg A rate ≥ 0.4353** — the 924 prior's own measurement shape (TF per-slot @ split 63, chat stream, 4000 rows; the pairing definition is Bench 917's: head-vs-THE-MODEL'S-OWN-argmax on the same row, never vs the gold token — arm 4's pairing is already exactly this, the semantics carry unchanged).
  - **Leg B (context row, never the bar):** one sequential teacher-forced walk over the first min(file, 4096) tokens (the serving ctx), the arm-4 loop shape, agreement over every position ≥ 1 (n ≈ 4k).
  - **Power note (why the first firing was WEAK, recorded pre-run):** arm 4's n=23 — at p=0.4353, P(X≤4) ≈ 0.010; the synthetic-seed distribution + n=23 made the first NEGATIVE suggestive, not decisive. Leg A's n≈250 separates 0.43 from 0.17 at ~7σ.
  - **Posture:** rate cells are load-valid (the bar-(a) precedent: deterministic numerics, bit-identical ×2 under load ~38/30, `RAYON_NUM_THREADS=8` cap, disclosure rides the result line) — this line AMENDS the draft text's "quiet-window exclusive" BEFORE the run, on that precedent; the quiet-window rule stays binding for T5.4/T5.5 ECONOMICS cells.
  - **Closing clause (unchanged):** a second miss closes the lane as a recorded NEGATIVE (the block stays loadable + excluded, the `mtp_draft_block` feature stays opt-in).
  **Claimed:** zed-f021648e@amo, 1795752492, expires 2026-11-29T04:08:12Z

**T5.4-R pre-run suspect adjudication (source-level, 2026-11-27, zed-f021648e@amo) — recorded BEFORE the cell, superseding the Diagnosis paragraph's open suspects:**
- **Suspect (1) the double-norm compose — REFUTED as a defect.** The fork's own graph IS the double-norm compose: `qwen35.cpp`'s main graph norms FIRST then publishes `t_h_nextn = t_embd` from the SAME post-`output_norm` tensor, and the MTP graph's `hnorm` applies ON TOP — "two norms with different gammas — by design, not a bias" (`src/mtp_draft.rs` basis-contract section, resolved by source read + verdict-recorded 2026-11-26, session 464973ce). Our hand-off (post-`final_norm` primal per `ternary_forward.rs:1681-1687`) + step 2's `hnorm` on top replicate the fork's contract EXACTLY. The Diagnosis paragraph's fix framing ("hand-off should be pre-norm, or hnorm skipped") was the polarity inversion the 11-27 correction warned about — there is nothing to repair: the cell measured the right wiring at the wrong distribution and the wrong n.
- **Suspect (2) the head-leg norm — REFUTED.** The real twin CARRIES `nextn.shared_head_norm` (`src/gguf_loader.rs` — the 15-tensor class's 7 F32 norms include `nextn.enorm/hnorm/shared_head_norm`; only `nextn.embed_tokens` + `nextn.shared_head_head` are absent, the census-disclosed Options). Step 4 prefers `shared_head_norm` when present (`src/mtp_draft.rs`, the `match block.shared_head_norm` arm) — the head leg is the block's OWN norm, never a fallback guess.
- **Suspect (3) the synthetic seed — STANDS, and it is the live confound.** `(i*7919+13)%v` seeds are off-distribution for a chat-trained nextn block, AND the main stack's own greedy picks on garbage tokens are near-arbitrary — two low-quality picks agree rarely. The 924 prior was measured on the CHAT transcript (Bench 924: mix head, TF @ split 63, chat stream). Leg A + Leg B measure at the prior's distribution and n.
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
- **FIRED 2026-11-27 (bar (a) leg):** If T5.2's CPU reference cannot reach agreement-class
  quality (bar (a)) even teacher-forced, the block's post-final basis mismatch (the spec_mtp
  disclosure's basis law, measured for the BMTH head) applies to the embedded block too —
  record the NEGATIVE and park; do not build T5.3/T5.4 on a dead drafter. **→ RECORDED 0.1739 vs ≥ 0.4353.** The clause's letter says "park, do not build T5.4"; T5.3 is already LANDED (built on the synthetic contract, its gates stand) and stays landed; the lane parks at the re-open cell (T5.4-R above) rather than an outright close — the three named suspects are cheap to falsify first, and the 5.35-vs-7.69 economics prior is already negative, so this lane's GOAT path was never wide.
- The measured cells need a QUIET box (exclusive rig job); a contended-box number is
  not a verdict — this plan refuses to cite one.

## Neighbors (not this plan)

riir-ai Issue 1263 (league re-pin + re-bout, owns T3's 4090 arm + T4); Issue 1183's
paused DSpine lane (the OTHER draft head — qwen3.8-target, paused by verdict; this plan
is the Bonsai-target lane the 1183 verdict ranked above it).
