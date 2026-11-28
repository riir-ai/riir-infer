//! Cross-layer KV-sharing init — the closed-form alignment fit + the
//! three-arm measurement harness (katgpt-rs Issue 1250 arm B T5; the YASA
//! distill, Research 698).
//!
//! The pre-registration is PINNED in
//! `../katgpt-rs/.issues/1250_yasa_poc_arms_tied_indexer_kv_init.md`
//! §"T5 pre-registration" — this module IMPLEMENTS it, it does not
//! re-litigate it. The measured pair is the league checkpoint's
//! full-attention layers **(3, 7)** (producer 3, consumer 7; every 4th
//! block is full attention so no two FA layers are adjacent — the
//! 3-DeltaNet gap is a disclosed confound in the pre-registration).
//!
//! Three arms over one loaded checkpoint, deltas isolate the sharing:
//!
//! 1. **baseline** — the checkpoint as-is (the plain forward).
//! 2. **naive copy** — the consumer reads the PRODUCER's cached K/V
//!    verbatim; its own `W_k`/`W_v` are skipped, its queries and output
//!    projection unchanged.
//! 3. **aligned init** — the consumer reads the producer's cache THROUGH
//!    the closed-form init fitted over anchor tokens:
//!    - key side: per kv head, the RoPE-commuting Procrustes rotation
//!      taking producer post-gain pre-RoPE keys to consumer post-gain
//!      pre-RoPE keys, applied to the consumer's queries as `Rᵀ`
//!      (post-gain, pre-RoPE — pinned for determinism; it commutes with
//!      RoPE so the insertion point is otherwise free);
//!    - value side: per kv head, the rank-`hd/2` truncated least-squares
//!      map taking producer values to consumer values, applied per q-head
//!      group between the attention output and the consumer's output
//!      projection.
//!
//! OFFLINE measurement lane (`kv_share_align`, opt-in — never a serving
//! path; a serving bin never links this module, never a serving-path
//! claim — the lens-lane posture). Structural reference:
//! [`crate::deltanet::ternary_kvq`] (the Issue-919 quantized-KV seam) —
//! the same layer-loop-reimplementation-with-swapped-plumbing discipline,
//! G0 correspondence pin included.
//!
//! # Key-side derivation (RoPE commutation)
//!
//! `rope_commuting_procrustes_paired` (katgpt-spectral, katgpt-rs Issue
//! 1251 — consumed, never forked) solves
//! `min_R ‖B − A·Rᵀ‖²_F` restricted to planar rotations on an explicit
//! coordinate-pair list (the Frobenius objective decouples per pair; the
//! closed-form Kabsch angle per pair). The pair list is the forward's own
//! rotate-half convention: `(i, i + rotary_dim/2)` for `i in
//! 0..rotary_dim/2` — BOTH rope kernels in [`crate::rope`] pair exactly
//! these coordinates (partial `apply_partial_rope_with_freq` and full
//! `apply_rope_with_freq` alike), so a rotation constant on each pair
//! commutes with RoPE: `R·rot_θ(k) = rot_θ(R·k)` per pair.
//!
//! The consumer's own cache rows would hold `rope(k7) ≈ rope(R·k3) =
//! R·rope(k3)`. Scoring the producer's cached keys with the query
//! pre-rotated by `Rᵀ` gives `(Rᵀq)·rope(k3) = q·(R·rope(k3))` — exactly
//! the score the consumer's own key would have produced, at every
//! position, without ever materializing the shared keys. The composed
//! per-head rotation is block-diagonal in `hd` dimensions: the rotary
//! planar blocks from the paired fit ⊕ a FREE orthogonal block on the
//! non-rotary dims `rotary_dim..hd` from
//! [`katgpt_spectral::procrustes::orthogonal_procrustes`] on the dim
//! slices (off-block entries exactly 0; `rotary_dim == hd` skips the free
//! half — the pre-registration's "dims 64..256" row is the
//! `rope_dimension_count = 64, head_dim = 256` instance).
//!
//! Gains note (the pre-registration's constraint discharge): the anchors
//! are POST-GAIN (captured right after the QK-norm step), so the
//! "gain uniform within each pair or folded first" clause is discharged
//! by capture — the fit never sees a pre-gain key.
//!
//! # Value-side derivation (the Gram/pseudoinverse algebra, shown in full)
//!
//! Per kv head, anchor matrices `A = V3` and `B = V7` — `T × hd`,
//! row-major, row `t` = the value at anchor token `t`. The map acts on
//! COLUMN vectors: `v7 = W·v3`. In row form each anchor obeys
//! `b_tᵀ = (W·a_t)ᵀ = a_tᵀ·Wᵀ`, hence `B = A·Wᵀ` and the least-squares
//! objective is
//!
//! ```text
//!     min_W  ‖B − A·Wᵀ‖²_F
//! ```
//!
//! Normal equations: `Aᵀ(A·Wᵀ − B) = 0 ⇒ (AᵀA)·Wᵀ = AᵀB`. With the
//! Gram `G = AᵀA` (hd × hd, symmetric PSD over the anchors) and the
//! cross `C = AᵀB` (hd × hd):
//!
//! ```text
//!     Wᵀ = G⁻¹·C        (so W = (G⁻¹C)ᵀ = Cᵀ·G⁻¹)
//! ```
//!
//! ⚠ **Transpose note (a deliberate correction of the task sketch):** a
//! sketch that writes "`W = G⁻¹C`" is naming `Wᵀ` under this row-anchor
//! convention — applying `G⁻¹C` directly to a producer value COLUMN
//! yields the transposed map. This module stores
//! `v_map = (U_r Λ_r⁻¹ U_rᵀ C)ᵀ` so the exec-side fold is verbatim
//! `v7 = W·v3` (verified by the planted-recovery gate G2: a planted
//! `v7 = M·v3` is recovered as `v_map ≈ M`, not `Mᵀ`).
//!
//! Solve via the symmetric eigendecomposition `G = UΛUᵀ`
//! (`katgpt_core::linalg::symmetric_eig` — deterministic Householder+QL,
//! f64, bit-identical; Issue 940 fold order throughout):
//!
//! ```text
//!     G⁻¹C = U·Λ⁻¹·Uᵀ·C      computed as  P = UᵀC;  row_k(P) *= 1/λ_k;  Wᵀ = U·P
//! ```
//!
// Rank truncation (Eckart–Young **of the map's inverse-gram spectrum**,
// honestly disclosed): recompose with only the top-`r` eigenpairs,
// `r = hd/2` (128 for the league head_dim 256 — PINNED by the
// pre-registration; the rank sweep is a follow-up column, never a bar
// read). This truncates the RESOLVED MAP's spectrum (the singular values
// of `G⁻¹C` through the orthogonal `U` basis), NOT the value covariance
// — the approximation quality is exactly what `fit_residual` reports.
//
// ⚠ **Projection semantics (measured, gate battery):** with `C = G·M_trueᵀ`
// the truncated solution is `W_trunc = M_true·P_r`, where `P_r = U_rU_rᵀ`
// projects onto the ANCHORS' top-r Gram eigenspace. The map is recovered
// losslessly only on the anchor-visible subspace; the complement is
// unseeable BY CONSTRUCTION, and `fit_residual` — which weighs error in
// the anchor metric — under-reports exactly those directions (measured in
// `kv_share_align_gates`: residual ≤ 0.05 beside a raw-Frobenius error
// ~0.8 on an arbitrary plant). A lossless-recovery assertion needs the
// plant inside span(U_r) (gate G2a's two-step). For the real cell this
// means: the rank-128 map is the best rank-128 anchor-metric map; its
// residual is the honest quality read the method promises, not a global
// Frobenius guarantee.
// Pseudoinverse floor: contributions with `λ_k ≤ 1e-10·λ_max` are DROPPED
//! (the Moore-Penrose-with-floor form — deterministic, and it makes the
//! effective rank honest when the anchors under-determine the map).
//! Eigenpair order: descending eigenvalue, ties by ascending index.
//!
//! # Exec-side fold order (why the task's `O7'` precompute sketch moved)
//!
//! The consumer's own computation would be
//! `out = O7 · Rot_h · (W_g·attn_h per head)`, gate first (the gate is
//! diagonal per head slice and `W_g` is block-diagonal per head, so the
//! two commute within a head — the gate runs verbatim at step 6, the fold
//! rides after it). The task sketch precomputes `O7' = O7·blockdiag(W_g)`
//! for a single matvec; that composes the fold on the WRONG side of the
//! folded-basis (hadamard) rotation: `Rot_h` is block-diagonal on the
//! rotation-block grid, `W_g` on the head grid, and rotations on NESTED
//! block grids do not commute in general (the hadamard blocks are
//! fixed ±1/Walsh structure, `W_g`'s blocks are arbitrary). The exec
//! therefore applies `W_g` per head BEFORE the in-place hadamard rotation
//! and projects with the UNfolded dense `O7` (dequantized once in
//! [`KvSharePlan::build`] — the "modified layer weight copy ONCE" of the
//! task, adjusted to the order-correct form). Extra cost: one hd × hd
//! matvec per q head per token, skipped on bit-exactly-identity heads.
//! When EVERY head's rotation and map are bit-exactly identity, the whole
//! arm DEMOTES to the naive path byte-identically (gate G1 pins it) — the
//! armed-empty byte-identity law, same shape as every other overlay lane.
//!
//! # Cache survival (the per-layer layout)
//!
//! [`crate::transformer::MultiLayerKVCache`] — the plain forward's
//! `HybridCache.kv_cache` — already keeps ONE `KVCache` per layer, so the
//! producer's rows (0..=pos) survive until the consumer's step in the
//! same pass; the harness forward reads
//! `cache.kv_cache.layers[producer_layer]` and never touches the
//! consumer's slot. The share arms therefore REQUIRE
//! `producer_layer < consumer_layer` — a producer at or after the
//! consumer would leave the row for the current position unwritten at
//! read time (same-pass causal order) — and [`fit_kv_share_alignment`]
//! refuses the inverted pair loud rather than documenting an unenforced
//! invariant.
//!
//! # Divergences from the task sketch (all documented, none silent)
//!
//! - `q_rot`/`v_map` are `Vec<Vec<f32>>` (per-head `[hd × hd]` row-major),
//!   not fixed `[f32; 256·256]` arrays — the sizing is config-driven
//!   everywhere (head_dim differs in the tiny gate fixtures; a literal
//!   array would be a head_dim-256 pin).
//! - The value-map algebra is `W = (G⁻¹C)ᵀ` (derivation above) — the
//!   sketch's `W = M⁻¹C` is the transposed map under the row-anchor
//!   convention.
//! - The `O7'` single-matvec precompute is replaced by the per-head
//!   fold-then-rotate form (order argument above).
//! - `producer ≥ consumer` is REFUSED, not just documented (causality
//!   argument above).
//! - The identity demote is BIT-EXACT (bit-pattern equality on the
//!   stored matrices), which is what gate G1 needs: at identity the
//!   aligned arm runs the ternary output projection (the naive path's
//!   kernel), never a dense matvec.

use crate::deltanet::forward::{HybridCache, HybridForwardScratch, effective_rotary_dim};
use crate::deltanet::rotation::{TernaryRotationConfig, rotate_forward_inplace};
use crate::deltanet::ternary_forward::{
    bitlinear, bitlinear_pair, qwen_deltanet_ternary_layer_body, ternary_embed_token_into,
    ternary_lm_head_into,
};
use crate::deltanet::ternary_weights::{DeltaNetTernaryLayerWeights, QwenDeltaNetTernaryWeights};
use crate::types::{Config, DeltaNetLayerType, rmsnorm_with_gamma_eps, swiglu};
use katgpt_core::linalg::{SymmetricEigScratch, symmetric_eig};
use katgpt_spectral::procrustes::{
    ProcrustesConfig, ProcrustesScratch, orthogonal_procrustes, rope_commuting_procrustes_paired,
};

/// Pseudoinverse floor relative to λ_max (the pre-registration's
/// "eigenvalue floor 1e-10 relative to λ_max").
const PINV_FLOOR_REL: f64 = 1e-10;
/// QL iteration cap per eigenvalue (the Issue-931 caller convention).
const SYMMETRIC_EIG_MAX_ITERS: usize = 200;

// ─────────────────────────────────────────────────────────────────────────
// The fitted artifact
// ─────────────────────────────────────────────────────────────────────────

/// The fitted cross-layer KV-sharing init (one call of
/// [`fit_kv_share_alignment`]). All matrices are per-KV-HEAD
/// (length `config.n_kv_head`), `[hd × hd]` row-major, hd =
/// `config.head_dim` — config-driven dims everywhere, never literals.
#[derive(Debug, Clone)]
pub struct KvShareFit {
    /// Producer layer index (the cache donor).
    pub producer_layer: usize,
    /// Consumer layer index (the cache reader).
    pub consumer_layer: usize,
    /// Per kv head: the TRANSPOSED rotation `Rᵀ` applied to the consumer's
    /// queries (post-gain, pre-RoPE). Row-major `[hd × hd]`.
    pub q_rot: Vec<Vec<f32>>,
    /// Per kv head: the value map applied to the producer's cached value
    /// (column form `v7 = W·v3`). Row-major `[hd × hd]`.
    pub v_map: Vec<Vec<f32>>,
    /// Per kv head: `‖θ‖₂` over the fitted rotary planar angles (radians,
    /// ordered fold per Issue 940). The NO-OP falsifier reads this:
    /// every head `< 1e-3` ⇒ the key spaces are already aligned.
    pub angles_norm: Vec<f32>,
    /// Pooled relative value-map residual
    /// `‖V7 − A·Wᵀ‖_F / ‖V7‖_F` over all heads' anchors (f64 folds, one
    /// f32 at rest).
    pub fit_residual: f32,
    /// BLAKE3 over the canonical fit inputs (recipe in
    /// [`fit_kv_share_alignment`]) — binds the artifact to the anchor
    /// tensors + the fit geometry.
    pub anchor_digest: [u8; 32],
    /// Head dim the matrices were fitted at (`config.head_dim`).
    pub head_dim: usize,
    /// Effective rotary dim (`effective_rotary_dim(config)`).
    pub rotary_dim: usize,
    /// The pinned rank cut actually applied (`min(hd/2, healthy λ count)`
    /// upper-bounded per head — the value is the PINNED cut `hd/2`; a head
    /// whose spectrum fell below the pseudoinverse floor resolves fewer).
    pub value_rank: usize,
    /// Anchor count the fit consumed (`anchor_tokens.len()`).
    pub n_anchors: usize,
}

impl KvShareFit {
    /// Bit-exact identity check on one stored matrix (the G1 demote
    /// predicate — `i == j ⇒ bits == 1.0`, else `bits == 0.0`).
    fn matrix_is_identity(m: &[f32], d: usize) -> bool {
        debug_assert_eq!(m.len(), d * d, "matrix shape");
        for i in 0..d {
            for j in 0..d {
                let expect = if i == j { 1.0 } else { 0.0 };
                if m[i * d + j] != expect {
                    return false;
                }
            }
        }
        true
    }

    /// All heads' key rotations are bit-exactly identity.
    #[must_use]
    pub fn q_is_identity(&self) -> bool {
        self.q_rot
            .iter()
            .all(|r| Self::matrix_is_identity(r, self.head_dim))
    }

    /// All heads' value maps are bit-exactly identity.
    #[must_use]
    pub fn v_is_identity(&self) -> bool {
        self.v_map
            .iter()
            .all(|w| Self::matrix_is_identity(w, self.head_dim))
    }

    /// The NO-OP falsifier: every head's fitted angle norm `< 1e-3` rad —
    /// the producer/consumer key spaces are already aligned (recorded,
    /// never read as negative).
    #[must_use]
    pub fn is_noop(&self) -> bool {
        self.angles_norm.iter().all(|a| *a < 1e-3)
    }
}

// ─────────────────────────────────────────────────────────────────────────
// The fit
// ─────────────────────────────────────────────────────────────────────────

/// Per-head anchor captures for one fit pass (private).
struct KvShareCapture {
    producer_layer: usize,
    consumer_layer: usize,
    n_kv: usize,
    hd: usize,
    /// `[slot][head][T · hd]` — slot 0 = producer, slot 1 = consumer.
    k: [Vec<Vec<f32>>; 2],
    v: [Vec<Vec<f32>>; 2],
}

impl KvShareCapture {
    fn new(producer_layer: usize, consumer_layer: usize, config: &Config) -> Self {
        let n_kv = config.n_kv_head;
        let hd = config.head_dim;
        Self {
            producer_layer,
            consumer_layer,
            n_kv,
            hd,
            k: [vec![Vec::new(); n_kv], vec![Vec::new(); n_kv]],
            v: [vec![Vec::new(); n_kv], vec![Vec::new(); n_kv]],
        }
    }

    fn slot_of(&self, layer_idx: usize) -> Option<usize> {
        if layer_idx == self.producer_layer {
            Some(0)
        } else if layer_idx == self.consumer_layer {
            Some(1)
        } else {
            None
        }
    }

    fn capture(&mut self, slot: usize, k_buf: &[f32], v_buf: &[f32]) {
        let hd = self.hd;
        for g in 0..self.n_kv {
            self.k[slot][g].extend_from_slice(&k_buf[g * hd..(g + 1) * hd]);
            self.v[slot][g].extend_from_slice(&v_buf[g * hd..(g + 1) * hd]);
        }
    }
}

/// The capture pass's own attention scratch — field-for-field the
/// `AttentionLayerScratch` sizing contract (`forward.rs::new`), privately
/// owned because the plain struct's buffers are `pub(super)`
/// (deltanet-internal — a crate-level module cannot reach them).
struct CaptureScratch {
    qg_buf: Vec<f32>,
    q_buf: Vec<f32>,
    gate_buf: Vec<f32>,
    k_buf: Vec<f32>,
    v_buf: Vec<f32>,
    attn_out: Vec<f32>,
    head_scores: Vec<f32>,
}

impl CaptureScratch {
    fn new(config: &Config) -> Self {
        let q_dim = config.n_head * config.head_dim;
        let kvd = crate::types::kv_dim(config);
        Self {
            qg_buf: vec![0.0; q_dim * 2],
            q_buf: vec![0.0; q_dim],
            gate_buf: vec![0.0; q_dim],
            k_buf: vec![0.0; kvd],
            v_buf: vec![0.0; kvd],
            attn_out: vec![0.0; q_dim],
            head_scores: vec![0.0; config.n_head * config.block_size],
        }
    }
}

/// The capture variant over the PRIVATE scratch (the real body; the
/// The capture variant (the real body — see the doc above
/// [`CaptureScratch`] for the scratch note).
#[allow(clippy::too_many_arguments)]
fn forward_attention_layer_ternary_capture(
    x: &mut [f32],
    layer: &DeltaNetTernaryLayerWeights,
    cache: &mut crate::transformer::KVCache,
    pos: usize,
    layer_idx: usize,
    config: &Config,
    rope_freq: &crate::rope::RopeFreqTable,
    cs: &mut CaptureScratch,
    rotation: Option<&TernaryRotationConfig>,
    rotation_buf: &mut [f32],
    capture: Option<&mut KvShareCapture>,
) {
    let n_embd = config.n_embd;
    let n_head = config.n_head;
    let n_kv = config.n_kv_head;
    let hd = config.head_dim;
    let q_dim = n_head * hd;
    let kvd = n_kv * hd;
    let rotary_dim = effective_rotary_dim(config);

    let x_in = &x[..n_embd];

    // 1. Gated Q projection + K + V (ternary) — verbatim (Issue 594/980).
    let rotated_qkv = rotation.is_some();
    if rotated_qkv {
        let rot = rotation.unwrap();
        let signs = rot.signs_for_width(n_embd);
        rotation_buf[..n_embd].copy_from_slice(x_in);
        rotate_forward_inplace(&mut rotation_buf[..n_embd], signs, rot.block_size);
    }
    let x_qkv: &[f32] = if rotated_qkv {
        &rotation_buf[..n_embd]
    } else {
        x_in
    };
    bitlinear(&mut cs.qg_buf, &layer.attn_wq, x_qkv, None);
    for h in 0..n_head {
        let src = h * 2 * hd;
        let dst = h * hd;
        cs.q_buf[dst..dst + hd].copy_from_slice(&cs.qg_buf[src..src + hd]);
        cs.gate_buf[dst..dst + hd].copy_from_slice(&cs.qg_buf[src + hd..src + 2 * hd]);
    }
    bitlinear(&mut cs.k_buf, &layer.attn_wk, x_qkv, None);
    bitlinear(&mut cs.v_buf, &layer.attn_wv, x_qkv, None);

    // 2. QK-norm (verbatim) — the CAPTURE point (post-gain, PRE-RoPE).
    let eps = config.rms_norm_eps;
    for h in 0..n_head {
        let off = h * hd;
        rmsnorm_with_gamma_eps(&mut cs.q_buf[off..off + hd], &layer.attn_q_norm, eps);
    }
    for h in 0..n_kv {
        let off = h * hd;
        rmsnorm_with_gamma_eps(&mut cs.k_buf[off..off + hd], &layer.attn_k_norm, eps);
    }
    if let Some(cap) = capture
        && let Some(slot) = cap.slot_of(layer_idx)
    {
        cap.capture(slot, &cs.k_buf, &cs.v_buf);
    }

    // 3. Partial RoPE (verbatim).
    if rotary_dim == hd {
        crate::rope::apply_rope_with_freq(
            &mut cs.q_buf,
            &mut cs.k_buf,
            pos,
            hd,
            rope_freq.as_slice(),
        );
    } else {
        crate::rope::apply_partial_rope_with_freq(
            &mut cs.q_buf,
            &mut cs.k_buf,
            pos,
            hd,
            rotary_dim,
            rope_freq.as_slice(),
        );
    }

    // 4. Store K, V in cache (verbatim).
    let pos_off = pos * kvd;
    unsafe {
        std::ptr::copy_nonoverlapping(cs.k_buf.as_ptr(), cache.key.as_mut_ptr().add(pos_off), kvd);
        std::ptr::copy_nonoverlapping(
            cs.v_buf.as_ptr(),
            cache.value.as_mut_ptr().add(pos_off),
            kvd,
        );
    }

    // 5. Multi-head attention with GQA (verbatim — over the CACHE rows,
    //    not the current row).
    let scale = 1.0 / (hd as f32).sqrt();
    cs.attn_out[..q_dim].fill(0.0);
    let t_n = pos + 1;
    unsafe {
        crate::transformer::attention_heads_parallel_bias(
            &cs.q_buf,
            &cache.key,
            &cache.value,
            &mut cs.attn_out,
            &mut cs.head_scores,
            n_head,
            n_kv,
            kvd,
            hd,
            t_n,
            scale,
            0.0,
            config.block_size,
            None,
        );
    }

    // 6. Output gating (verbatim).
    for i in 0..q_dim {
        cs.attn_out[i] *= crate::simd::fast_sigmoid(cs.gate_buf[i]);
    }

    // 7. Output projection (ternary, folded — verbatim).
    if let Some(rot) = rotation {
        let signs = rot.signs_for_width(q_dim);
        rotate_forward_inplace(&mut cs.attn_out[..q_dim], signs, rot.block_size);
    }
    bitlinear(
        &mut x[..n_embd],
        &layer.attn_wo,
        &cs.attn_out[..q_dim],
        None,
    );
}

/// The model-level capture pass: the plain ternary forward's structure
/// bounded to layers `0..=consumer_layer` (a token's K/V at layer L never
/// depends on layers after it — the early stop is a pure prefix of the
/// same computation), with the capture variant at the two fit targets and
/// the shared plain layer body everywhere else. Teacher-forced: one token
/// per position, cache + GDN state fresh for the pass.
#[allow(clippy::too_many_arguments)]
fn run_capture_pass(
    weights: &QwenDeltaNetTernaryWeights,
    config: &Config,
    rope_freq: &crate::rope::RopeFreqTable,
    producer_layer: usize,
    consumer_layer: usize,
    anchor_tokens: &[usize],
) -> Result<KvShareCapture, String> {
    let n = config.n_embd;
    let mut cfg = config.clone();
    // Scratch sizing (head_scores rides config.block_size): the pass must
    // never index past the buffers, whatever the caller's cap was.
    cfg.block_size = cfg.block_size.max(anchor_tokens.len() + 8);
    let mut capture = KvShareCapture::new(producer_layer, consumer_layer, config);
    let mut cache =
        HybridCache::with_layer_types_bounded(&cfg, &weights.layer_types, anchor_tokens.len() + 8);
    let mut scratch = HybridForwardScratch::new(&cfg);
    let mut cs = CaptureScratch::new(&cfg);
    let rotation = weights.rotation.as_ref();

    crate::deltanet::on_decode_pool(move || {
        let mut x = vec![0.0f32; n];
        for (pos, &token) in anchor_tokens.iter().enumerate() {
            ternary_embed_token_into(&mut x, weights, n, token);
            for layer_idx in 0..=consumer_layer {
                let layer_weights = &weights.layers[layer_idx];
                let is_linear = weights.layer_types[layer_idx] == DeltaNetLayerType::DeltaNet;
                let is_target = layer_idx == producer_layer || layer_idx == consumer_layer;
                if is_target && !is_linear {
                    // a–b (verbatim from `qwen_deltanet_ternary_layer_body`).
                    scratch.residual[..n].copy_from_slice(&x[..n]);
                    rmsnorm_with_gamma_eps(
                        &mut x[..n],
                        &layer_weights.input_norm,
                        cfg.rms_norm_eps,
                    );
                    let cap = Some(&mut capture);
                    forward_attention_layer_ternary_capture(
                        &mut x,
                        layer_weights,
                        &mut cache.kv_cache.layers[layer_idx],
                        pos,
                        layer_idx,
                        &cfg,
                        rope_freq,
                        &mut cs,
                        rotation,
                        &mut scratch.rotation_buf,
                        cap,
                    );
                    kvshare_post_attention_tail(
                        &mut x,
                        layer_weights,
                        &cfg,
                        &mut scratch,
                        rotation,
                    );
                } else {
                    qwen_deltanet_ternary_layer_body(
                        &mut x,
                        layer_weights,
                        is_linear,
                        &mut cache.deltanet_state.recurrent_states[layer_idx],
                        &mut cache.deltanet_state.conv_states[layer_idx],
                        &mut cache.kv_cache.layers[layer_idx],
                        pos,
                        layer_idx,
                        &cfg,
                        &mut scratch,
                        rope_freq,
                        rotation,
                        None,
                        None,
                        None,
                    );
                }
            }
        }
        Ok(capture)
    })
}

/// The post-attention per-layer tail (residual add → MLP residual save →
/// pre-MLP RMSNorm → SwiGLU → rotated down projection → residual add).
/// VERBATIM from `ternary_forward::ternary_post_attention_tail` steps d–i
/// minus the hook branches (this lane passes no hooks — the hook plumbing
/// stays in the plain forward, the `ternary_kvq` precedent). Shared by the
/// capture pass and the share arms so the two cannot drift.
#[allow(clippy::too_many_arguments)]
fn kvshare_post_attention_tail(
    x: &mut [f32],
    layer_weights: &DeltaNetTernaryLayerWeights,
    config: &Config,
    scratch: &mut HybridForwardScratch,
    rotation: Option<&TernaryRotationConfig>,
) {
    let n = config.n_embd;
    let x = &mut x[..n];

    // d. Residual add.
    for (xi, r) in x[..n].iter_mut().zip(&scratch.residual[..n]) {
        *xi += *r;
    }

    // e. Save residual for MLP.
    scratch.residual[..n].copy_from_slice(&x[..n]);

    // f. Pre-MLP RMSNorm (dense field).
    rmsnorm_with_gamma_eps(
        &mut x[..n],
        &layer_weights.post_attn_norm,
        config.rms_norm_eps,
    );

    // g. SwiGLU MLP (ternary projections) — Issue 980 folded: rotate the
    //    shared input ONCE into the rotation scratch.
    let x_ffn: &[f32] = if let Some(rot) = rotation {
        let signs = rot.signs_for_width(n);
        scratch.rotation_buf[..n].copy_from_slice(&x[..n]);
        rotate_forward_inplace(&mut scratch.rotation_buf[..n], signs, rot.block_size);
        &scratch.rotation_buf[..n]
    } else {
        &x[..n]
    };
    bitlinear_pair(
        &mut scratch.gate,
        &layer_weights.gate_proj,
        &mut scratch.up,
        &layer_weights.up_proj,
        x_ffn,
        None,
    );
    swiglu(&mut scratch.hidden, &scratch.gate, &scratch.up);

    // h. Down projection (ternary) — folded: rotate `hidden` in place.
    if let Some(rot) = rotation {
        let signs = rot.signs_for_width(config.mlp_hidden);
        rotate_forward_inplace(&mut scratch.hidden, signs, rot.block_size);
    }
    bitlinear(&mut x[..n], &layer_weights.down_proj, &scratch.hidden, None);

    // i. Residual add.
    for (xi, r) in x.iter_mut().zip(&scratch.residual[..n]) {
        *xi += *r;
    }
}

/// Fit the cross-layer KV-sharing alignment over anchor tokens.
///
/// Runs the teacher-forced capture pass (the plain forward's arithmetic —
/// the fit must see the checkpoint's own keys/values), then per kv head:
///
/// 1. keys — `rope_commuting_procrustes_paired(K3, K7, T, hd, pairs)` with
///    `pairs = (i, i + rotary_dim/2)` (the forward's own rotate-half
///    convention), plus `orthogonal_procrustes` on the non-rotary dim
///    slice when `rotary_dim < hd`; composed `R` stored TRANSPOSED in
///    `q_rot` (the exec applies `Rᵀ` to the consumer's queries);
/// 2. values — the rank-truncated Gram/pseudoinverse map (derivation in
///    the module doc), stored column-form in `v_map`;
/// 3. per-head `‖θ‖₂` of the rotary planar angles + the pooled
///    value-map residual + the canonical anchor digest.
///
/// # Errors (all loud, typed `String` — the G3 refusal battery)
///
/// layer kinds (a DeltaNet layer as producer or consumer), layer identity,
/// the inverted pair (`producer ≥ consumer` — same-pass causality), empty
/// anchors, out-of-range token ids, degenerate anchors (zero cross
/// covariance), and the substrate's own refusals (underdetermined fit,
/// odd dims) surfaced verbatim.
#[allow(clippy::too_many_arguments)]
pub fn fit_kv_share_alignment(
    weights: &QwenDeltaNetTernaryWeights,
    config: &Config,
    rope_freq: &crate::rope::RopeFreqTable,
    producer_layer: usize,
    consumer_layer: usize,
    anchor_tokens: &[u32],
) -> Result<KvShareFit, String> {
    let hd = config.head_dim;
    let n_kv = config.n_kv_head;
    let rotary_dim = effective_rotary_dim(config);

    // ── validation (loud, before any pass) ─────────────────────────────
    if weights.layer_types.len() != config.n_layer {
        return Err(format!(
            "layer_types length {} != config.n_layer {}",
            weights.layer_types.len(),
            config.n_layer
        ));
    }
    if hd < 2 || !hd.is_multiple_of(2) {
        return Err(format!(
            "kv_share_align needs an even head_dim >= 2 (rotate-half pairs), got {hd}"
        ));
    }
    if !rotary_dim.is_multiple_of(2) || rotary_dim > hd {
        return Err(format!(
            "effective rotary dim {rotary_dim} must be even and <= head_dim {hd}"
        ));
    }
    if producer_layer >= config.n_layer || consumer_layer >= config.n_layer {
        return Err(format!(
            "layer pair ({producer_layer}, {consumer_layer}) out of range (n_layer {})",
            config.n_layer
        ));
    }
    if producer_layer == consumer_layer {
        return Err(format!(
            "producer_layer == consumer_layer == {producer_layer}: sharing a layer with itself is undefined"
        ));
    }
    if producer_layer > consumer_layer {
        // Same-pass causal order: the harness forward reads the producer's
        // cache rows 0..=pos at the consumer's step, which only exist if
        // the producer ran earlier in the loop (see the module doc).
        return Err(format!(
            "producer_layer {producer_layer} > consumer_layer {consumer_layer}: \
             the same-pass harness requires producer < consumer"
        ));
    }
    for (name, layer) in [("producer", producer_layer), ("consumer", consumer_layer)] {
        if weights.layer_types[layer] != DeltaNetLayerType::Attention {
            return Err(format!(
                "{name}_layer {layer} is a DeltaNet layer — the KV-sharing fit targets full-attention layers only"
            ));
        }
    }
    if anchor_tokens.is_empty() {
        return Err("anchor_tokens is empty — nothing to fit".to_string());
    }
    let anchors: Vec<usize> = anchor_tokens
        .iter()
        .map(|&t| {
            let t = t as usize;
            if t >= config.vocab_size {
                return Err(format!(
                    "anchor token id {t} out of range (vocab_size {})",
                    config.vocab_size
                ));
            }
            Ok(t)
        })
        .collect::<Result<_, _>>()?;

    // ── capture pass ───────────────────────────────────────────────────
    let cap = run_capture_pass(
        weights,
        config,
        rope_freq,
        producer_layer,
        consumer_layer,
        &anchors,
    )?;
    let t = anchors.len();

    // ── per-head fits ──────────────────────────────────────────────────
    let value_rank = hd / 2;

    let mut q_rot = Vec::with_capacity(n_kv);
    let mut v_map = Vec::with_capacity(n_kv);
    let mut angles_norm = Vec::with_capacity(n_kv);
    let mut res_num = 0.0f64;
    let mut res_den = 0.0f64;

    // Canonical digest over the fit inputs (deterministic recipe).
    let mut digest = blake3::Hasher::new();
    digest.update(b"i1250-kvshare-fit-v1");
    digest.update(&(producer_layer as u64).to_le_bytes());
    digest.update(&(consumer_layer as u64).to_le_bytes());
    digest.update(&(n_kv as u64).to_le_bytes());
    digest.update(&(hd as u64).to_le_bytes());
    digest.update(&(rotary_dim as u64).to_le_bytes());
    digest.update(&(value_rank as u64).to_le_bytes());
    digest.update(&(t as u64).to_le_bytes());

    for g in 0..n_kv {
        let k3 = &cap.k[0][g];
        let k7 = &cap.k[1][g];
        let v3 = &cap.v[0][g];
        let v7 = &cap.v[1][g];
        debug_assert_eq!(k3.len(), t * hd, "k3 capture rows");
        digest.update(bytemuck::cast_slice::<f32, u8>(k3));
        digest.update(bytemuck::cast_slice::<f32, u8>(k7));
        digest.update(bytemuck::cast_slice::<f32, u8>(v3));
        digest.update(bytemuck::cast_slice::<f32, u8>(v7));

        let key = fit_head_rotation(k3, k7, t, hd, rotary_dim)
            .map_err(|e| format!("kv head {g}: {e}"))?;
        let mut rt = vec![0.0f32; hd * hd];
        for i in 0..hd {
            for j in 0..hd {
                rt[i * hd + j] = key.r_full[j * hd + i];
            }
        }
        q_rot.push(rt);
        let theta_sq: f64 = key.angles.iter().map(|a| a * a).sum();
        angles_norm.push(theta_sq.sqrt() as f32);

        let value = fit_head_value_map(v3, v7, t, hd, value_rank)
            .map_err(|e| format!("kv head {g}: {e}"))?;
        v_map.push(value.w);
        res_num += value.residual_sq;
        res_den += value.target_sq;
    }

    if res_den <= 0.0 {
        return Err("degenerate value anchors (‖V7‖_F == 0)".to_string());
    }
    let fit_residual = ((res_num / res_den).sqrt()) as f32;

    Ok(KvShareFit {
        producer_layer,
        consumer_layer,
        q_rot,
        v_map,
        angles_norm,
        fit_residual,
        anchor_digest: *digest.finalize().as_bytes(),
        head_dim: hd,
        rotary_dim,
        value_rank,
        n_anchors: t,
    })
}

/// One kv head's fitted key rotation: the composed block-diagonal `R`
/// (`[hd × hd]` row-major — rotary planar blocks ⊕ the free orthogonal
/// block on the non-rotary dims, off-blocks exactly 0) and the per-pair
/// planar angles in radians (f64, atan2 of the fitted block cells).
#[derive(Debug, Clone)]
pub struct KvShareKeyFit {
    pub r_full: Vec<f32>,
    pub angles: Vec<f64>,
}

/// Fit ONE kv head's key rotation from captured anchor pairs — rows =
/// tokens, POST-GAIN PRE-RoPE keys, `[t × hd]` row-major. `k7 ≈ R·k3`
/// least squares under the RoPE-commuting constraint (the forward's own
/// rotate-half pair list `(i, i + rotary_dim/2)`); the non-rotary dims
/// `rotary_dim..hd` take a free orthogonal rotation via the unconstrained
/// solve. Deterministic (the substrate's Issue-940 folds). Pub so a
/// consumer with its own capture plumbing can fit without the model-level
/// pass (the gate battery drives it with exact anchors — the closed form's
/// own recovery bar lives there).
pub fn fit_head_rotation(
    k3: &[f32],
    k7: &[f32],
    t: usize,
    hd: usize,
    rotary_dim: usize,
) -> Result<KvShareKeyFit, String> {
    if k3.len() != t * hd || k7.len() != t * hd {
        return Err(format!(
            "anchor shape mismatch: expected {t} × {hd}, got {} / {}",
            k3.len(),
            k7.len()
        ));
    }
    if hd < 2 || !hd.is_multiple_of(2) {
        return Err(format!(
            "kv_share_align needs an even head_dim >= 2 (rotate-half pairs), got {hd}"
        ));
    }
    if !rotary_dim.is_multiple_of(2) || rotary_dim > hd || rotary_dim < 2 {
        return Err(format!(
            "effective rotary dim {rotary_dim} must be even, >= 2, <= head_dim {hd}"
        ));
    }
    let half = rotary_dim / 2;
    let pairs: Vec<(usize, usize)> = (0..half).map(|i| (i, i + half)).collect();
    let d_free = hd - rotary_dim;
    let fit_config = ProcrustesConfig {
        center: false,
        special_orthogonal: false,
        compute_residual: true,
        compute_det: false,
        min_anchors: 0, // the substrate default 2·d applies
    };

    let mut rot_scratch = ProcrustesScratch::new(t, hd);
    let mut r_rot = vec![0.0f32; hd * hd];
    rope_commuting_procrustes_paired(
        k3,
        k7,
        t,
        hd,
        &pairs,
        &mut r_rot,
        &mut rot_scratch,
        &fit_config,
    )
    .map_err(|e| format!("rotary procrustes fit failed: {e}"))?;

    // Per-pair angles, extracted from the fitted block-diagonal R
    // (R[ca·hd + ca] = cos θ, R[cb·hd + ca] = sin θ, cb = ca + half).
    let mut angles = Vec::with_capacity(half);
    for (ca, cb) in &pairs {
        let cos = f64::from(r_rot[ca * hd + ca]);
        let sin = f64::from(r_rot[cb * hd + ca]);
        angles.push(sin.atan2(cos));
    }

    let mut r_full = r_rot; // planar blocks placed, unlisted dims identity
    if d_free > 0 {
        let mut free_scratch = ProcrustesScratch::new(t, d_free);
        let mut a_free_buf = vec![0.0f32; t * d_free];
        let mut b_free_buf = vec![0.0f32; t * d_free];
        let mut r_free = vec![0.0f32; d_free * d_free];
        // Strided dim slices: a_free[t][c] = K3[t][rotary_dim + c].
        for (row, chunk) in a_free_buf.chunks_mut(d_free).enumerate() {
            for (c, cell) in chunk.iter_mut().enumerate() {
                *cell = k3[row * hd + rotary_dim + c];
            }
        }
        for (row, chunk) in b_free_buf.chunks_mut(d_free).enumerate() {
            for (c, cell) in chunk.iter_mut().enumerate() {
                *cell = k7[row * hd + rotary_dim + c];
            }
        }
        orthogonal_procrustes(
            &a_free_buf,
            &b_free_buf,
            t,
            d_free,
            &mut r_free,
            &mut free_scratch,
            &fit_config,
        )
        .map_err(|e| format!("free-block procrustes fit failed: {e}"))?;
        // Compose: overwrite the bottom-right diagonal block; the planar
        // blocks live strictly in [0, rotary_dim)² so the two halves never
        // overlap — off-block entries stay exactly 0.
        for i in 0..d_free {
            for j in 0..d_free {
                r_full[(rotary_dim + i) * hd + (rotary_dim + j)] = r_free[i * d_free + j];
            }
        }
    }

    Ok(KvShareKeyFit { r_full, angles })
}

/// One kv head's fitted value map: `w` (`[hd × hd]` row-major, COLUMN
/// form — `v7 = W·v3`) plus the residual folds
/// (`residual_sq = ‖B − A·Wᵀ‖²_F`, `target_sq = ‖B‖²_F`) for pooled
/// reporting.
#[derive(Debug, Clone)]
pub struct KvShareValueFit {
    pub w: Vec<f32>,
    pub residual_sq: f64,
    pub target_sq: f64,
}

/// Fit ONE kv head's value map from captured anchor pairs (rows = tokens,
/// `[t × hd]` row-major): the rank-truncated Gram/pseudoinverse least
/// squares of the module doc (f64 symmetric eig, deterministic order,
/// Issue-940 folds, pseudoinverse floor `1e-10·λ_max`). Pub alongside
/// [`fit_head_rotation`] for the same reason.///
/// # Errors
/// shape mismatches, `rank == 0`, non-finite eigenvalues, degenerate
/// anchors.
pub fn fit_head_value_map(
    v3: &[f32],
    v7: &[f32],
    t: usize,
    hd: usize,
    rank: usize,
) -> Result<KvShareValueFit, String> {
    if v3.len() != t * hd || v7.len() != t * hd {
        return Err(format!(
            "anchor shape mismatch: expected {t} × {hd}, got {} / {}",
            v3.len(),
            v7.len()
        ));
    }
    if rank == 0 || rank > hd {
        return Err(format!("rank {rank} out of range (1..={hd})"));
    }

    // G = AᵀA, C = AᵀB in f64 (Issue 940 ordered folds over anchors).
    let mut gram = vec![0.0f64; hd * hd];
    let mut cross = vec![0.0f64; hd * hd];
    for (row, a_row) in v3.chunks_exact(hd).enumerate() {
        let b_row = &v7[row * hd..(row + 1) * hd];
        for i in 0..hd {
            let ai = f64::from(a_row[i]);
            for j in 0..hd {
                gram[i * hd + j] += ai * f64::from(a_row[j]);
                cross[i * hd + j] += ai * f64::from(b_row[j]);
            }
        }
    }

    let mut eig_scratch = SymmetricEigScratch::new();
    let mut eigvals = vec![0.0f64; hd];
    let mut eigvecs = vec![0.0f64; hd * hd];
    symmetric_eig(
        &mut eigvals,
        &mut eigvecs,
        &gram,
        &mut eig_scratch,
        hd,
        SYMMETRIC_EIG_MAX_ITERS,
    );
    for (i, v) in eigvals.iter().enumerate() {
        if !v.is_finite() {
            return Err(format!("non-finite eigenvalue at {i}"));
        }
    }

    // Deterministic order: descending eigenvalue, ties by ascending index
    // (the comparator never returns Equal across distinct pairs unless
    // both λ and index agree — impossible for distinct indices).
    let mut order: Vec<usize> = (0..hd).collect();
    order.sort_by(|&a, &b| {
        eigvals[b]
            .partial_cmp(&eigvals[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let lambda_max = eigvals[order[0]];
    if lambda_max <= 0.0 {
        return Err(format!(
            "degenerate value anchors (Gram spectrum max {lambda_max})"
        ));
    }
    let floor = lambda_max * PINV_FLOOR_REL;

    // Wᵀ = Σ_{k ∈ top-r, λ_k > floor} (1/λ_k) · u_k · (u_kᵀ C):
    //   P = Uᵀ·C, keep/scale the top-r rows, Wᵀ = U_r · P_kept.
    let mut p = vec![0.0f64; hd * hd];
    // p[k][j] = u_kᵀ · C[:, j] — u_k is eigvecs COLUMN k: the contract is
    // `eigvecs[i*n + j] = V[i, j]`, so component i of eigenvector k is
    // `eigvecs[k + i*n]` (start at k, stride n — the doc's "length-n
    // stride-n slice starting at eigvecs[k]").
    for k in 0..hd {
        for j in 0..hd {
            let mut acc = 0.0f64;
            for i in 0..hd {
                acc += eigvecs[k + i * hd] * cross[i * hd + j];
            }
            p[k * hd + j] = acc;
        }
    }
    let mut wt = vec![0.0f64; hd * hd];
    for &k in order.iter().take(rank.min(hd)) {
        let lambda = eigvals[k];
        if lambda <= floor {
            // Below the pseudoinverse floor: dropped (Moore-Penrose with
            // floor — the effective rank is honest).
            continue;
        }
        let inv = 1.0 / lambda;
        // wt += (1/λ) · u_k ⊗ p_row_k  (u_k[i] = eigvecs[k + i·hd])
        for i in 0..hd {
            let u_ik = eigvecs[k + i * hd];
            if u_ik == 0.0 {
                continue;
            }
            let p_row = &p[k * hd..(k + 1) * hd];
            let wt_row = &mut wt[i * hd..(i + 1) * hd];
            for j in 0..hd {
                wt_row[j] += inv * u_ik * p_row[j];
            }
        }
    }

    // The exec-side map is the TRANSPOSE (v7 = W·v3, column form).
    let mut w = vec![0.0f32; hd * hd];
    for i in 0..hd {
        for j in 0..hd {
            w[i * hd + j] = wt[j * hd + i] as f32;
        }
    }

    // Residual folds: ‖B − A·Wᵀ‖²_F and ‖B‖²_F (predicted[t][j] =
    // Σ_i A[t][i]·W[j][i] — W acts on the value column).
    let mut residual_sq = 0.0f64;
    let mut target_sq = 0.0f64;
    for row in 0..t {
        let a_row = &v3[row * hd..(row + 1) * hd];
        let b_row = &v7[row * hd..(row + 1) * hd];
        for j in 0..hd {
            let mut acc = 0.0f64;
            let w_row = &w[j * hd..(j + 1) * hd];
            for i in 0..hd {
                acc += f64::from(a_row[i]) * f64::from(w_row[i]);
            }
            let diff = f64::from(b_row[j]) - acc;
            residual_sq += diff * diff;
            target_sq += f64::from(b_row[j]) * f64::from(b_row[j]);
        }
    }

    Ok(KvShareValueFit {
        w,
        residual_sq,
        target_sq,
    })
}

// ─────────────────────────────────────────────────────────────────────────
// The exec plan + the harness forward
// ─────────────────────────────────────────────────────────────────────────

/// The eval-side execution plan: a [`KvShareFit`] bound to concrete layer
/// indices with the identity flags and the ONCE-built dense output
/// projection. Build once per (fit, checkpoint); reuse across windows and
/// arms.
#[derive(Debug)]
pub struct KvSharePlan<'a> {
    /// The fitted transforms.
    pub fit: &'a KvShareFit,
    /// The cache donor layer.
    pub producer_layer: usize,
    /// The cache reader layer.
    pub consumer_layer: usize,
    /// Per-head bit-exact-identity flags (precomputed — the per-token
    /// hot path must not rescan hd² cells per head).
    pub q_rot_identity: Vec<bool>,
    /// Per-head bit-exact-identity flags for the value maps.
    pub v_map_identity: Vec<bool>,
    /// All heads' rotations identity.
    pub q_identity: bool,
    /// All heads' maps identity — when true the arm's projection DEMOTES
    /// to the ternary kernel (the naive path's), byte-identically (G1).
    pub v_identity: bool,
    /// Dense dequant of the consumer's `attn_wo` `[n_embd × q_dim]`
    /// row-major — `Some` iff `!v_identity` (built ONCE in
    /// [`KvSharePlan::build`]; the per-head W application runs before the
    /// folded-basis rotation, so the projection itself stays UNfolded —
    /// see the module doc's fold-order note).
    pub o_dense: Option<Vec<f32>>,
}

impl<'a> KvSharePlan<'a> {
    /// Bind a fit to its layer pair + checkpoint. Validates the pair
    /// against the layer kinds and the fit's own geometry.
    pub fn build(
        fit: &'a KvShareFit,
        producer_layer: usize,
        consumer_layer: usize,
        weights: &QwenDeltaNetTernaryWeights,
        config: &Config,
    ) -> Result<Self, String> {
        if producer_layer != fit.producer_layer || consumer_layer != fit.consumer_layer {
            return Err(format!(
                "plan pair ({producer_layer}, {consumer_layer}) != fit pair ({}, {})",
                fit.producer_layer, fit.consumer_layer
            ));
        }
        if weights.layer_types.len() != config.n_layer || consumer_layer >= config.n_layer {
            return Err("plan pair out of range for the checkpoint".to_string());
        }
        if weights.layer_types[consumer_layer] != DeltaNetLayerType::Attention {
            return Err(format!(
                "consumer_layer {consumer_layer} is a DeltaNet layer — the share arms target full-attention layers only"
            ));
        }
        if fit.head_dim != config.head_dim || fit.q_rot.len() != config.n_kv_head {
            return Err(format!(
                "fit geometry (hd {}, {} heads) != config (hd {}, {} heads)",
                fit.head_dim,
                fit.q_rot.len(),
                config.head_dim,
                config.n_kv_head
            ));
        }
        let q_rot_identity: Vec<bool> = fit
            .q_rot
            .iter()
            .map(|r| KvShareFit::matrix_is_identity(r, fit.head_dim))
            .collect();
        let v_map_identity: Vec<bool> = fit
            .v_map
            .iter()
            .map(|w| KvShareFit::matrix_is_identity(w, fit.head_dim))
            .collect();
        let q_identity = q_rot_identity.iter().all(|b| *b);
        let v_identity = v_map_identity.iter().all(|b| *b);
        let o_dense = if v_identity {
            None
        } else {
            // The "modified layer weight copy ONCE" — dequantized, UNfolded
            // (the per-head W application happens before the rotation; see
            // the module doc).
            let wo = &weights.layers[consumer_layer].attn_wo;
            let (rows, cols) = (wo.rows(), wo.cols());
            debug_assert_eq!(rows, config.n_embd, "attn_wo rows");
            debug_assert_eq!(cols, config.n_head * config.head_dim, "attn_wo cols");
            Some(wo.dequant_to_dense())
        };
        Ok(Self {
            fit,
            producer_layer,
            consumer_layer,
            q_rot_identity,
            v_map_identity,
            q_identity,
            v_identity,
            o_dense,
        })
    }

    /// The q head's kv group (the attention kernel's own GQA mapping:
    /// `h · n_kv / n_head` integer division).
    #[inline]
    fn group_of(&self, h: usize, config: &Config) -> usize {
        h * config.n_kv_head / config.n_head
    }
}

/// The harness arm (what differs between the eval passes).
pub enum KvShareArm<'a> {
    /// The plain forward — the G0 control.
    Plain,
    /// The consumer reads the producer's cache verbatim: its own K/V
    /// projections skipped, no transforms.
    Naive {
        producer_layer: usize,
        consumer_layer: usize,
    },
    /// The aligned init through a fitted plan.
    Aligned(&'a KvSharePlan<'a>),
}

/// The share arms' own attention scratch — field-for-field the
/// `AttentionLayerScratch` sizing contract, privately owned (the plain
/// struct's buffers are `pub(super)`, deltanet-internal). Reused across
/// tokens and arms; zero steady-state allocation.
pub struct KvShareScratch {
    qg_buf: Vec<f32>,
    q_buf: Vec<f32>,
    gate_buf: Vec<f32>,
    /// RoPE dummy: the q-only rope call takes both buffers; the producer's
    /// keys are ALREADY roped (stored post-RoPE), so K rides a zeroed
    /// scratch whose roped output is discarded.
    k_dummy: Vec<f32>,
    attn_out: Vec<f32>,
    head_scores: Vec<f32>,
    fold_tmp: Vec<f32>,
}

impl KvShareScratch {
    #[must_use]
    pub fn new(config: &Config) -> Self {
        let q_dim = config.n_head * config.head_dim;
        let kvd = crate::types::kv_dim(config);
        Self {
            qg_buf: vec![0.0; q_dim * 2],
            q_buf: vec![0.0; q_dim],
            gate_buf: vec![0.0; q_dim],
            k_dummy: vec![0.0; kvd],
            attn_out: vec![0.0; q_dim],
            head_scores: vec![0.0; config.n_head * config.block_size],
            fold_tmp: vec![0.0; config.head_dim],
        }
    }
}

/// The dense row-major matvec `out[r] = Σ_c w[r·cols + c]·x[c]` — the
/// sequential ascending fold (the Issue-940 order-pinned form; the mtp
/// draft's private helper shape, local to this lane).
fn dense_matvec_rowmajor(w: &[f32], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(w.len(), rows * cols, "dense matvec shape");
    debug_assert_eq!(x.len(), cols, "dense matvec input width");
    for (r, orow) in out[..rows].iter_mut().enumerate() {
        let row = &w[r * cols..(r + 1) * cols];
        let mut acc = 0.0f32;
        for (acc_cell, (wv, xv)) in row.iter().zip(x.iter()).enumerate() {
            let _ = acc_cell;
            acc += wv * xv;
        }
        *orow = acc;
    }
}

/// The share arms' one full-attention layer: the plain
/// `forward_attention_layer_ternary` structure with steps 1–3 reduced to
/// the Q side (the consumer's own K/V are the sharing's whole point),
/// step 4 removed (no store — the consumer's cache slot stays untouched),
/// step 5 redirected at the PRODUCER's per-layer cache rows, and the
/// aligned transforms inserted at their pinned points (q rotation
/// post-gain pre-RoPE; value fold after the gate, before the folded-basis
/// rotation).
#[allow(clippy::too_many_arguments)]
fn forward_attention_layer_ternary_share(
    x: &mut [f32],
    layer: &DeltaNetTernaryLayerWeights,
    producer_cache: &crate::transformer::KVCache,
    pos: usize,
    config: &Config,
    rope_freq: &crate::rope::RopeFreqTable,
    kv: &mut KvShareScratch,
    rotation: Option<&TernaryRotationConfig>,
    rotation_buf: &mut [f32],
    aligned: Option<&KvSharePlan<'_>>,
) {
    let n_embd = config.n_embd;
    let n_head = config.n_head;
    let n_kv = config.n_kv_head;
    let hd = config.head_dim;
    let q_dim = n_head * hd;
    let kvd = n_kv * hd;
    let rotary_dim = effective_rotary_dim(config);

    let x_in = &x[..n_embd];

    // 1. Gated Q projection ONLY (the naive and aligned arms both skip the
    //    consumer's W_k/W_v — that skip IS the sharing).
    let rotated_qkv = rotation.is_some();
    if rotated_qkv {
        let rot = rotation.unwrap();
        let signs = rot.signs_for_width(n_embd);
        rotation_buf[..n_embd].copy_from_slice(x_in);
        rotate_forward_inplace(&mut rotation_buf[..n_embd], signs, rot.block_size);
    }
    let x_qkv: &[f32] = if rotated_qkv {
        &rotation_buf[..n_embd]
    } else {
        x_in
    };
    bitlinear(&mut kv.qg_buf, &layer.attn_wq, x_qkv, None);
    for h in 0..n_head {
        let src = h * 2 * hd;
        let dst = h * hd;
        kv.q_buf[dst..dst + hd].copy_from_slice(&kv.qg_buf[src..src + hd]);
        kv.gate_buf[dst..dst + hd].copy_from_slice(&kv.qg_buf[src + hd..src + 2 * hd]);
    }

    // 2. QK-norm on Q only (K is the producer's post-norm cached row).
    let eps = config.rms_norm_eps;
    for h in 0..n_head {
        let off = h * hd;
        rmsnorm_with_gamma_eps(&mut kv.q_buf[off..off + hd], &layer.attn_q_norm, eps);
    }

    // 2.5 ALIGNED: per-head query rotation by R_gᵀ — post-gain, pre-RoPE
    //     (the pinned insertion point; it commutes with RoPE either way).
    if let Some(plan) = aligned
        && !plan.q_identity
    {
        for h in 0..n_head {
            let g = plan.group_of(h, config);
            if plan.q_rot_identity[g] {
                continue;
            }
            let r = &plan.fit.q_rot[g];
            let qh = &mut kv.q_buf[h * hd..(h + 1) * hd];
            let tmp = &mut kv.fold_tmp[..hd];
            for (i, t_cell) in tmp.iter_mut().enumerate() {
                let row = &r[i * hd..(i + 1) * hd];
                let mut acc = 0.0f32;
                for (wv, xv) in row.iter().zip(qh.iter()) {
                    acc += wv * xv;
                }
                *t_cell = acc;
            }
            qh.copy_from_slice(tmp);
        }
    }

    // 3. RoPE on Q only — the SAME kernel the plain path runs (the dummy K
    //    rides along; the kernels treat the two buffers' heads
    //    independently). The producer's keys were roped at store time.
    if rotary_dim == hd {
        crate::rope::apply_rope_with_freq(
            &mut kv.q_buf,
            &mut kv.k_dummy,
            pos,
            hd,
            rope_freq.as_slice(),
        );
    } else {
        crate::rope::apply_partial_rope_with_freq(
            &mut kv.q_buf,
            &mut kv.k_dummy,
            pos,
            hd,
            rotary_dim,
            rope_freq.as_slice(),
        );
    }

    // 4. NO store — the consumer's cache slot is not written in a share
    //    arm (nothing to write; its own K/V were never computed).

    // 5. GQA attention over the PRODUCER's cache rows 0..=pos — per-layer
    //    `MultiLayerKVCache` keeps them from the producer's own step
    //    earlier in this same pass.
    let scale = 1.0 / (hd as f32).sqrt();
    kv.attn_out[..q_dim].fill(0.0);
    let t_n = pos + 1;
    unsafe {
        crate::transformer::attention_heads_parallel_bias(
            &kv.q_buf,
            &producer_cache.key,
            &producer_cache.value,
            &mut kv.attn_out,
            &mut kv.head_scores,
            n_head,
            n_kv,
            kvd,
            hd,
            t_n,
            scale,
            0.0,
            config.block_size,
            None,
        );
    }

    // 6. Output gating (verbatim).
    for i in 0..q_dim {
        kv.attn_out[i] *= crate::simd::fast_sigmoid(kv.gate_buf[i]);
    }

    // 6.5 ALIGNED: per-head value fold — v7 = W_g · (this head's attention
    //     output). Skipped on bit-exactly-identity heads; the fold rides
    //     BEFORE the folded-basis rotation (the order argument in the
    //     module doc — the rotation does not commute with W_g).
    if let Some(plan) = aligned
        && !plan.v_identity
    {
        for h in 0..n_head {
            let g = plan.group_of(h, config);
            if plan.v_map_identity[g] {
                continue;
            }
            let w = &plan.fit.v_map[g];
            let sl = &mut kv.attn_out[h * hd..(h + 1) * hd];
            let tmp = &mut kv.fold_tmp[..hd];
            for (i, t_cell) in tmp.iter_mut().enumerate() {
                let row = &w[i * hd..(i + 1) * hd];
                let mut acc = 0.0f32;
                for (wv, xv) in row.iter().zip(sl.iter()) {
                    acc += wv * xv;
                }
                *t_cell = acc;
            }
            sl.copy_from_slice(tmp);
        }
    }

    // 7. Folded-basis output rotation (verbatim).
    if let Some(rot) = rotation {
        let signs = rot.signs_for_width(q_dim);
        rotate_forward_inplace(&mut kv.attn_out[..q_dim], signs, rot.block_size);
    }

    // 7'. Output projection: ternary kernel when every head's value map is
    //     bit-exactly identity (the naive path's kernel — G1's bit
    //     identity rides here); the ONCE-built dense O7 otherwise.
    let v_identity = aligned.is_none_or(|p| p.v_identity);
    if v_identity {
        bitlinear(
            &mut x[..n_embd],
            &layer.attn_wo,
            &kv.attn_out[..q_dim],
            None,
        );
    } else {
        let o = aligned
            .expect("non-identity fold requires the aligned plan")
            .o_dense
            .as_ref()
            .expect("o_dense built when v_identity is false");
        dense_matvec_rowmajor(o, n_embd, q_dim, &kv.attn_out[..q_dim], &mut x[..n_embd]);
    }
}

/// The ternary hybrid forward with the cross-layer KV-sharing arm. The
/// plain forward's structure byte-for-byte (embed → hybrid layer loop →
/// final norm → rotated LM head) with the CONSUMER layer's attention
/// swapped per arm; every other layer — including the producer — runs the
/// shared plain layer body, so the producer's cache rows are the real
/// forward's and survive in the per-layer cache until the consumer's step.
///
/// `KvShareArm::Plain` runs NO special path of its own — it is the same
/// loop with the consumer slot unbound, and gate G0 pins its logits
/// bit-identically to [`forward_qwen_deltanet_ternary`](crate::deltanet::ternary_forward::forward_qwen_deltanet_ternary).
#[allow(clippy::too_many_arguments)]
pub fn forward_qwen_deltanet_ternary_kvshare<'a>(
    x: &'a mut [f32],
    weights: &QwenDeltaNetTernaryWeights,
    cache: &mut HybridCache,
    arm: &KvShareArm<'_>,
    token: usize,
    pos: usize,
    config: &Config,
    scratch: &'a mut HybridForwardScratch,
    rope_freq: &crate::rope::RopeFreqTable,
    kv: &mut KvShareScratch,
) -> &'a mut [f32] {
    // Run on a worker of the decode pool (the plain forward's posture —
    // no 17th thread beside the pool; Bench 880). Arithmetic unchanged.
    crate::deltanet::on_decode_pool(move || {
        forward_qwen_deltanet_ternary_kvshare_body(
            x, weights, cache, arm, token, pos, config, scratch, rope_freq, kv,
        )
    })
}

/// The share pair this forward call resolves (None = plain).
fn share_pair<'a>(arm: &'a KvShareArm<'a>) -> Option<(usize, usize, Option<&'a KvSharePlan<'a>>)> {
    match arm {
        KvShareArm::Plain => None,
        KvShareArm::Naive {
            producer_layer,
            consumer_layer,
        } => Some((*producer_layer, *consumer_layer, None)),
        KvShareArm::Aligned(plan) => Some((plan.producer_layer, plan.consumer_layer, Some(plan))),
    }
}

#[allow(clippy::too_many_arguments)]
fn forward_qwen_deltanet_ternary_kvshare_body<'a>(
    x: &'a mut [f32],
    weights: &QwenDeltaNetTernaryWeights,
    cache: &mut HybridCache,
    arm: &KvShareArm<'_>,
    token: usize,
    pos: usize,
    config: &Config,
    scratch: &'a mut HybridForwardScratch,
    rope_freq: &crate::rope::RopeFreqTable,
    kv: &mut KvShareScratch,
) -> &'a mut [f32] {
    let n = config.n_embd;
    let v_dim = config.deltanet_linear_n_value_heads * config.deltanet_linear_head_dim;

    let (producer_layer, consumer_layer, plan) = match share_pair(arm) {
        Some((p, c, pl)) => (Some(p), Some(c), pl),
        None => (None, None, None),
    };
    if let Some(cons) = consumer_layer {
        debug_assert!(
            weights.layer_types[cons] == DeltaNetLayerType::Attention,
            "kv_share_align: the consumer layer {cons} is a DeltaNet layer (the fit and the plan both refuse this; the exec asserts)"
        );
    }

    // Rotation state (Issue 980) — verbatim.
    let rotation = weights.rotation.as_ref();
    let max_rot_len = n.max(v_dim).max(config.mlp_hidden);
    debug_assert!(
        scratch.rotation_buf.len() >= max_rot_len,
        "rotation scratch {} < required {max_rot_len} (HybridForwardScratch::new sizes it)",
        scratch.rotation_buf.len()
    );

    // 1. Embedding lookup + folded-basis restore — verbatim.
    ternary_embed_token_into(x, weights, n, token);

    // 2. Layer loop with hybrid dispatch (verbatim structure).
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        let is_linear = weights.layer_types[layer_idx] == DeltaNetLayerType::DeltaNet;
        let is_consumer = consumer_layer == Some(layer_idx);

        if is_consumer && !is_linear {
            // a. Save residual + b. pre-attention RMSNorm (the layer
            //    body's steps a-b, verbatim).
            scratch.residual[..n].copy_from_slice(&x[..n]);
            rmsnorm_with_gamma_eps(&mut x[..n], &layer_weights.input_norm, config.rms_norm_eps);

            // c. The share-arm attention step over the PRODUCER's cache.
            let producer = producer_layer.expect("share arm names a producer layer");
            let producer_cache = &cache.kv_cache.layers[producer];
            forward_attention_layer_ternary_share(
                &mut x[..n],
                layer_weights,
                producer_cache,
                pos,
                config,
                rope_freq,
                kv,
                rotation,
                &mut scratch.rotation_buf,
                plan,
            );

            // d–i. Shared tail (no hooks in this lane).
            kvshare_post_attention_tail(&mut x[..n], layer_weights, config, scratch, rotation);
        } else {
            // The plain per-layer body (norm → attention/DeltaNet → MLP,
            // both residual adds) — the shared extraction the plain
            // forward itself runs; a drift indicts the shared body, not
            // this loop (the Issue 022 T3.1 parity law, structural).
            qwen_deltanet_ternary_layer_body(
                x,
                layer_weights,
                is_linear,
                &mut cache.deltanet_state.recurrent_states[layer_idx],
                &mut cache.deltanet_state.conv_states[layer_idx],
                &mut cache.kv_cache.layers[layer_idx],
                pos,
                layer_idx,
                config,
                scratch,
                rope_freq,
                rotation,
                None,
                None,
                None,
            );
        }
    }

    // 3–4. Final RMSNorm + LM head (shared helper, Issue 1117).
    ternary_lm_head_into(x, weights, config, scratch, rotation, None);

    &mut x[..config.vocab_size]
}
