//! # riir-infer-core — the model-based inference layer
//!
//! Proposal 041 / T1.1: the model layer extracted out of `riir-engine`
//! (transformer, deltanet, quantization, GGUF loading, CPU reference
//! paths). This crate is **modelless-inference shaped**: weight loading,
//! quantized forward passes, and architecture definitions — zero
//! cognition imports (measured, T1.4: the only non-move-set references
//! are the upstream `katgpt_quant::turboquant` re-export), zero training
//! code (Issue 741 evicted Tier C to riir-train-engine first).
//!
//! **Phase 1 layout:** the crate lives inside riir-ai and every module is
//! re-exported from `riir_engine` at the SAME paths (T1.2, the Issue 739 /
//! Bench 723 shape), so no consumer import changes. Phase 2 (repo
//! promotion) is gated on T1.5 build-cost evidence — see
//! `.proposals/041_riir_infer_model_based_inference_split.md`.
//!
//! Modules move in dependency order: leaves first (`types`, `simd`,
//! `rope`, `gemma_layer`, `llama_layer`, `quant`, `wall`,
//! `safetensors_loader`), then the interdependent blob atomically
//! (`transformer` + `deltanet` + `ternary_layer` + `spec_types` +
//! `dflash` + `gguf_loader` — a 3-cycle forces one chunk).

// ── Chunk A: leaves (moved 2026-08-27, Proposal 041 T1.1) ──────────────

/// Shared engine type aliases — pure leaf, zero `crate::` imports at move
/// time. 253 reference sites in riir-engine resolve through the re-export.
pub mod types;

/// SIMD helpers used by the model layer (leaf).
pub mod simd;

/// `RoPE` tables (leaf).
pub mod rope;

/// Issue 1244 T1 — the header-only HiSA planner (report-only; pure math
/// over `HisaHeader`, the GGUF header's own facts — `rope.freq_base`,
/// `attention.key_length`, `rope.dimension_count`, `full_attention_interval`;
/// Research 040). The bin front is `src/bin/hisa_plan.rs`.
pub mod hisa_plan;

/// Issue 1244 T2 — the HiSA rotation-suppression census arm (report-only;
/// the score-path instrument beside [`hisa_plan`]'s closed-form prior).
/// Opt-in under `hisa_census`; the census driver owns the only armed copy
/// (the scratch field compiles out of every other build).
#[cfg(feature = "hisa_census")]
pub mod hisa_census;

/// Issue 1244 T2.1b — the census sweep driver (grids, cell table, the
/// frozen-read pass runner, the resumable ledger, the canonical report);
/// the bin front is `src/bin/hisa_census.rs`.
#[cfg(feature = "hisa_census")]
pub mod hisa_census_run;

/// Issue-1244 T3 landing tooling: merge the shard censuses into one
/// canonical report + classify per (layer, chunk) — the typed verdict and
/// the T2.3 predicted-vs-measured columns. The bin front is
/// `src/bin/hisa_merge.rs`.
#[cfg(feature = "hisa_census")]
pub mod hisa_census_merge;

/// Gemma architecture layer helpers (leaf).
pub mod gemma_layer;

/// Llama architecture layer helpers (leaf).
pub mod llama_layer;

/// Quantization primitives: `Q2_0` / `Q4_K` / `Q5_K` / `Q6_K` / `Q8_KV`.
pub mod quant;

/// WALL attention (leaf; depends only on `simd`).
pub mod wall;

/// Wall Attention CONFIGURATION (Issue 019 Phase C.1 de-fork; re-homed
/// from riir-engine by Plan 610 S5 — the gemma-cluster unlock). A pure
/// re-export of the canonical `katgpt_types::WallConfig` behind the
/// `wall_attention` feature; the engine re-exports this module at its
/// historical `riir_engine::wall_config` path, so every consumer path
/// (riir-gpu's wall_decode/wall_mla, katgpt config plumbing) is unchanged.
pub mod wall_config;

/// safetensors weight loading (leaf; depends on `gemma_layer` + `types`).
pub mod safetensors_loader;

// ── Chunk B: the interdependent blob (moved atomically — 3-cycle
// transformer → ternary_layer → deltanet → transformer, 2026-08-27) ────

/// The transformer model layer (gemma2/gemma4/llama/mtp/raven/dllm/prefill
/// + tests). Largest module in the crate.
pub mod transformer;

/// The `DeltaNet` model layer (weights, forward, ternary weights/forward,
/// `flashmemory_gqa`, `tree_forward`, profiling). Gated exactly as in
/// riir-engine.
#[cfg(feature = "deltanet_inference")]
pub mod deltanet;

/// Issue 1189 — the closed-form fact-write lane: overlay-at-dequant apply,
/// the teacher-forced capture sink, and the `FactWriteModel` adapter over
/// katgpt-core's constructor (consumed, never forked). Gated wholesale:
/// `QuantErrorLora` itself is feature-gated katgpt-core-side, so the module
/// cannot resolve in a default build. Default builds compile it out
/// entirely (the sink_audit precedent); empty overlays = byte-identical demote.
#[cfg(feature = "closed_form_write")]
pub mod closed_form_write;

/// Issue 1256 T1 — the exact-implant runtime lane: N silent modules over a
/// bounded token-id window (exact read tables → pure-function fire gate →
/// sampling-seam bias writes), per-implant fire counters + demote envs,
/// and the A2 monotonic mount (law re-stated, surgical_read_patch
/// precedent — the grafts probe modality stays katgpt-core's). Gated
/// wholesale: a default build compiles it to nothing (the byte-identical
/// -off law); the observe path is zero-alloc (fixture-pinned G4).
#[cfg(feature = "implants")]
pub mod implants;
#[cfg(feature = "implants")]
pub mod implants_cert;
#[cfg(feature = "implants_cell")]
pub mod implants_cell;

/// The katgpt-rs issue-988 T2b real-activation cell (riir-infer Issue
/// 1206): the capture seam + the offline graft fit/eval harness over the
/// dense qwen35-hybrid body. The PRIMITIVE is katgpt-core's (`grafts`,
/// forwarded by this feature); this module hosts the hidden-state
/// capture (`HiddenCapture` — the sink_audit Option pattern), the
/// task/split generation, and the report. The serving path never mounts
/// it; a default build compiles the module (and the scratch field) out
/// entirely.
#[cfg(feature = "grafts_cell")]
pub mod grafts_cell;

/// Ternary-weight transformer structs — only compiles with the ternary
/// substrate on (mirrors the engine-side gate + comment).
#[cfg(feature = "ternary_inference")]
pub mod ternary_layer;

/// Speculative-decode spec types.
pub mod spec_types;

/// `DFlash` drafter.
pub mod dflash;

/// GGUF weight loading.
pub mod gguf_loader;

/// Issue 1253 T5.2 (plan 639) — the MTP draft block's CPU reference
/// forward (`mtp_draft_block` + the ternary lane it serves; the admission
/// half lives in `gguf_loader`).
#[cfg(all(feature = "mtp_draft_block", feature = "deltanet_ternary_inference"))]
pub mod mtp_draft;

/// Issue 028 — the dual-PTQ disaggregated container (`.pf` one-file law +
/// two-file pair) and the in-process prefill→decode phase handoff over the
/// shared `HybridCache`. Opt-in with the ternary lane it serves.
#[cfg(feature = "deltanet_ternary_inference")]
pub mod disaggregated;

/// Natural-text corpus loading for the offline calibration bins (the
/// `chat_probe` HF-pages shape + plain text). One loader, both instruments.
pub mod corpus_text;

/// Tensor-family taxonomy for the qwen35-hybrid GGUF grammar (Issue 1188
/// T0): the mixed-precision sensitivity ranking tool's classification
/// substrate — name → family → storage class, plus the census walk.
/// Always compiled; classification only, never a serving path.
pub mod family_census;

/// Weight-structure audit lane (Issue 1209, Research 025 — the Lazarus
/// distill): the five-feature reference-free census over the last MLP's
/// SwiGLU block. REPORT-ONLY, opt-in `weight_audit` — a flag means
/// "inspect", never "reject"; never a serving path; zero new deps. The
/// synthetic fixture planter lives OUTSIDE this surface:
/// `weight_audit_fixture` is `#[cfg(all(test, feature = "weight_audit"))]`
/// — test-support only, typed to the fixture's own block.
#[cfg(feature = "weight_audit")]
pub mod weight_audit;
#[cfg(all(test, feature = "weight_audit"))]
mod weight_audit_fixture;

/// Generation-check instruments (Issue 1242 T2 — the QDA distill's
/// `gencheck`: truncation-at-limit census + suffix-cycle/repetition census
/// over long generations). Pure text analysis, zero deps, ungated (the
/// instrument is the reason the module is tiny + self-contained; a serving
/// path never calls it). The plan-508 T2 gate consumes it via the
/// anchor_kl bin (the same definitions).
pub mod gencheck;

/// Offline interventional neuron census (Issue 1241 T1 — the Fermion
/// distill, katgpt-rs Research 684): per-layer chunked dose-grid pushes
/// over the down_proj columns, matched-random control bands via
/// `katgpt_pruners::mech_attribution` (Issue 1237's landed scorer
/// primitives — consumed, never forked), a canonical BLAKE3 artifact +
/// a resumable cursor/rows state dir. Opt-in `neuron_census`, offline
/// by construction (G3/G5): serving bins never link it, never a
/// serving-path claim (the lens-lane demotion binds the claim this
/// lane refuses to make). The GGUF cell is T2 (routes to the 4090
/// `--needs cuda` window per the issue).
#[cfg(feature = "neuron_census")]
pub mod neuron_census;
/// The G1 fixture (planted-edit recovery battery) — feature-gated (not
/// test-gated) so the `neuron-census` bin's `--self-test` drives the
/// same source the tests assert; an unarmed build compiles it to
/// nothing.
#[cfg(feature = "neuron_census")]
pub mod neuron_census_fixture;
/// The GGUF census session (Issue 1241 T2-prep): the
/// [`Qwen35CensusSession`] adapter (owned weights, column-scale
/// push/restore protocol over the real checkpoint) + the mechanics
/// gate the `neuron-census --mechanics-check` posture runs on CPU.
/// The T2 cell (observable/probe tuning + the full sweep) routes to
/// the 4090 `--needs cuda` window per the issue.
#[cfg(feature = "neuron_census")]
pub mod neuron_census_gguf;

#[cfg(feature = "readout_screen")]
pub mod readout_jshare;
/// The static readout pre-screen (Issue 1250 T1 — katgpt-rs Research
/// 693, the Anthropic Fellows pair distill): the weights-only logit
/// lens over every down_proj column, sign-split (the type-2 finding),
/// top-K lens tokens per unit + per-band sharpness/margin rollups —
/// the strictly-cheaper tier in front of the census screen and the
/// interventional census. Opt-in `readout_screen`, offline by
/// construction (weights only, no forward pass, no serving path),
/// zero new deps. The `bucket` field ships UNASSIGNED in v1 — T2's
/// J-share verbalizability pass fills it. The fixture gates live in
/// `readout_screen_fixture` (feature-gated, not test-gated: the
/// `readout-screen --self-test` bin posture drives the same source).
#[cfg(feature = "readout_screen")]
pub mod readout_screen;
#[cfg(feature = "readout_screen")]
pub mod readout_screen_fixture;

/// The spike-CODING census (Issue 1252 A2 — the SpikingBrain distill,
/// katgpt-rs Research 694): the offline decision-data pass — per-layer
/// MLP-activation capture via the `closed_form_write` CaptureSink seam
/// (dense AND ternary bodies share the `batched_mlp` hook), then per
/// k ∈ {2,3,4}: the adaptive-threshold spike-count census (`V_th =
/// mean|x|/k`, `s = round(x/V_th)`), the element-zero `sparsity(k)`
/// point, the whole-128-group zero rate (the packed ternary matvec's
/// skip granularity), and the structural-zero decomposition against the
/// pinned Gaussian/Laplacian floors. Decision data, NEVER a kernel and
/// never a serving path (the issue's four pre-registered refusals
/// bind: no GPU spike simulation, no decode activation quantization,
/// no kernel before A2's bars, no imported energy figures). The census
/// math is tokenizer/model-free (fixture-gated); the artifact is
/// canonical bytes + BLAKE3 (`SPKC1`, the RDMAP class). Opt-in
/// `spike_code_census` (implies the capture seam's feature graph);
/// offline by construction; Issue-940 fold order; no RNG. Fixture
/// gates live in `spike_code_census_fixture` (feature-gated, not
/// test-gated: the `spike-code-census --self-test` bin posture drives
/// the same source).
#[cfg(feature = "spike_code_census")]
pub mod spike_code_census;
#[cfg(feature = "spike_code_census")]
pub mod spike_code_census_fixture;

/// The trigger-vote screen (Issue 1251 T1 — katgpt-rs Research 693
/// Part 2 App. C.4, the checkpoint-ingest backdoor probe; the
/// published class is cited, never claimed): a weights-only vote over
/// a (base, finetune) GGUF pair — per-neuron top-15 changed tokens by
/// `|Δgate·e_t| + |Δup·e_t|`, the C.4 layer-band admission rule, and
/// the pooled ranking over the admitted band. Opt-in under the same
/// `readout_screen` feature (the vote shares the lane's pinned fold,
/// selection and artifact conventions — zero new deps). Offline by
/// construction (weights only, no forward pass, no serving path).
/// Fixture gates live in `trigger_vote_fixture` (feature-gated, not
/// test-gated: the `trigger-vote --self-test` bin posture drives the
/// same source).
#[cfg(feature = "readout_screen")]
pub mod trigger_vote;
#[cfg(feature = "readout_screen")]
pub mod trigger_vote_fixture;

/// The planted-fixture lane (Issue 1251 T2 — the synthetic backdoor
/// planter behind the vote's T3 recovery cell): the in-place Q8_0 GGUF
/// patcher, the two planters (T2a disclosed-circular; T2b non-circular
/// upstream-feature cores), the PLNT1 manifest, and the pre-write
/// self-falsifier. Opt-in under the same `readout_screen` feature;
/// offline by construction (a fixture tool, never a serving path).
#[cfg(feature = "readout_screen")]
pub mod trigger_vote_cell;
/// The cell fixture gates (Issue 1251 T2/T3 G1 — the whole chain on
/// tiny deterministic GGUFs the fixture writes itself).
#[cfg(feature = "readout_screen")]
pub mod trigger_vote_cell_fixture;
/// The T3 recovery cell (Issue 1251 T3 — the pinned bars): the two
/// baselines (summed-change over the vote's own statistic family;
/// embedding-shift) and the scorer (≥⅔ of ≥3 planted triggers in the
/// vote's top-10, each at ≥2× the better-ranked baseline).
#[cfg(feature = "readout_screen")]
pub mod trigger_vote_cell_score;
/// The safetensors vote source (Issue 1251 T4 — the detector's native
/// HF-format lane): the same [`VoteTensors`] trait over a minimal
/// in-repo safetensors reader (zero new deps), so the supply-chain
/// screen reads the format third-party checkpoints actually arrive in.
#[cfg(feature = "readout_screen")]
pub mod trigger_vote_safetensors;

/// KL-vs-anchor ranked-selection census (Issue 1242 T1 — the QDA distill's
/// anchor-KL instrument): per-token forward-KL from logits slices, the
/// top-k-union variant, the trace-set aggregation. OPT-IN under
/// `anchor_kl` (offline instruments; a serving path never reaches it).
#[cfg(feature = "anchor_kl")]
pub mod anchor_kl;

/// f64 golden-NLL readout (Issue 959 T6, opt-in `golden_nll`): the
/// tail-faithful per-token oracle for lossy-quant gates over
/// katgpt-core's `fixed_log_softmax_target_f64`. Instrument-grade —
/// never a serving path.
#[cfg(feature = "golden_nll")]
pub mod golden_nll;

/// Frozen-weight adapter artifacts (Issue 1208, opt-in `deletion_mask`):
/// the deletion-mask codec leaf. The deployment half of adapters that
/// write zero new weight values; never a serving path until T2 wires
/// apply-at-load behind the same feature.
#[cfg(feature = "deletion_mask")]
pub mod adapter;
/// Offline R/S Jacobian-lens lane (Research 666, katgpt-rs Issue 999,
/// opt-in `lens`): RelP hand-rule op maps — the LN/identity/half rules and
/// the exact adjoints (RoPE, projections, causal conv1d) the layer-stack
/// lens fit composes into `linearize`/`jvp`/`vjp`. Same posture as
/// `sink_audit` — offline once per checkpoint, never a serving path; the
/// fit IS the laya stability lane's deferred "headful lens variant".
#[cfg(feature = "lens")]
pub mod lens;

/// Plan 622 TokenGate lane switches (T1.5): the `tokengate` compile-time
/// feature + the `RIIR_TOKENGATE` runtime demote. See the module doc.
pub mod tokengate;

/// Issue 1221 — the Dory looped-nGPT lane: the GGUF loader for the
/// `dory` arch (I/R/O partition + loop metadata read from the header,
/// never hardcoded; the T0 census values are loud pins) + the lane's
/// dims view. Opt-in `dory` — unarmed builds compile it to nothing and
/// a dory GGUF handed to a non-dory lane refuses loud by arch name.
#[cfg(feature = "dory")]
pub mod dory;

/// Multi-host pipeline substrate (Plan 623 Phase 1): the layer-range
/// vocabulary, the activation wire frame, the stage handshake, and the
/// per-range HybridCache state-segment serializer. Opt-in — the lane's
/// wire surface; the range-forward compute lives ungated inside the
/// forward bodies it splits.
#[cfg(feature = "multi_host")]
pub mod multi_host;

/// TWT phase-collapse lane (Issue 022): the S-matrix profiler, the
/// min-max DP partition, the ΔS quant-damage map. Opt-in — never a
/// default without the Phase-5 GOAT gate.
#[cfg(feature = "twt_profile")]
pub mod twt;

/// FA-constrained exact posterior sampling for the dLLM decode lane
/// (Issue 035 P0): the pure joint sampler over automaton-constrained
/// blocks — CSR automaton tensors, edge-space forward–backward, exact
/// state-path-then-tokens draws. Ungated pure math (the decode-loop
/// wiring is the `fa_constraint` arm, P1); allocation-free hot path.
pub mod fa_posterior;
pub mod fa_schema;

/// Language-preserving minimization for [`fa_posterior::Automaton`] (issue
/// 035, the G2 axis): trim + partition refinement at block granularity.
/// `fa_schema::compile` applies it; hand-built automata can too. The merged
/// states carry identical per-token flow, so the exact joint sampler's
/// token-sequence distribution is preserved — the test battery pins that
/// distributionally, not just language-level.
pub mod fa_minimize;

/// SentencePiece/BPE tokenizers for GGUF-embedded vocabularies (native-only;
/// the sentencepiece-sys C++ backend cannot compile for wasm32).
#[cfg(not(target_arch = "wasm32"))]
pub mod tokenizer;

/// Issue 1122 T2 — the cross-call history n-gram drafter's offline
/// acceptance-rate estimator (pure token-math over captured decode logs;
/// the measurement instrument, not the gated T1 drafter arm).
pub mod rollout_history;

// Upstream re-export the model layer consumes (`crate::turboquant` —
// transformer/mod.rs + tests). Same gated form as riir-engine's copy;
// riir-engine keeps its own for its consumers.
#[cfg(feature = "turboquant")]
pub use katgpt_quant::turboquant;

// Issue 832 — NaN-safe comparator gates for the float sort sites fixed
// under this issue (triage + fix record: .issues/832_*).
#[cfg(test)]
mod issue832_float_order_tests;
