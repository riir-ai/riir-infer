//! Issue 1253 T5.2 (plan 639) — the MTP draft block's CPU REFERENCE forward.
//!
//! One teacher-forced draft step over [`MtpDraftBlock`] + the main ternary
//! stack's globals, mirroring the Prism fork's `qwen35.cpp::graph_mtp`:
//!
//! 1. embed = the `wte` row of `last_token`, INVERSE-rotated to the primal
//!    basis (the fold stores latent rows on lookup tables — the fork's
//!    `hdmd->inv` arm; `token_embd.weight` is the one verified
//!    inverse-after-lookup table);
//! 2. `concat(enorm(embed), hnorm(h_prev))` → `eh_proj` — the compose head;
//! 3. one gated-attention block (Issue-594 q‖gate split, per-head q/k
//!    norms, partial RoPE, GQA over the block's OWN [`KVCache`]) →
//!    residual → `post_attention_norm` → SwiGLU FFN → residual;
//! 4. head norm (`shared_head_norm`, else the main stack's `output_norm`)
//!    → forward-rotated copy → the folded `lm_head` matvec (the
//!    `ternary_lm_head_into` mirror; `shared_head_head` absent on the real
//!    twin, the main head is the fallback — the `Option` contract).
//!
//! ## Basis contract (the crux, pinned before any run)
//!
//! **Which hidden the fork hands over — the verdict's open question,
//! resolved by source read 2026-11-27:** `qwen35.cpp`'s main graph norms
//! FIRST, then publishes BOTH consumers from the SAME normed tensor —
//! `cur = build_norm(cur, model.output_norm, …, LLM_NORM_RMS, -1);
//! cb(cur, "h_nextn", -1); res->t_h_nextn = cur; … res->t_embd = cur;` —
//! so `t_h_nextn` (the MTP block's `inp->h`) is the hidden AFTER
//! `output_norm`, and the MTP graph's `hnorm` then applies ON TOP (two
//! norms with different gammas — by design, not a bias; the same shape
//! the embed side carries: `enorm` norms the already-token-embedding
//! vector). This module's `h_prev` contract (post-final-norm, primal) is
//! the fork's own; the hidden variant's rotated copy recovers it with
//! `rotate_inverse_inplace` (the Issue-1111 recipe).
//!
//! The Hadamard fold stores `W_f = W·D·H` and applies `H·D` at each folded
//! matmul's input locally — the RESIDUAL STREAM IS PRIMAL everywhere
//! (`tests/bonsai2_rotation_load.rs::hidden_variant_returns_head_basis`
//! pins the one visible exception: the hidden variant returns the rotated
//! head-basis copy). This module therefore consumes `h_prev` in the
//! PRIMAL basis. The draft block's own tensors are NOT folded (verified:
//! the real twin's `prism.hadamard.weight_names` carries zero
//! `nextn`/`blk.64` entries), so steps 2–3 run primal-in/primal-out with
//! no rotation; only the head matvec (step 4) rotates.
//!
//! The CPU reference is the ORACLE (the reference-keeps-the-job law): a
//! GPU/Metal arm (T5.5) is admissible only against this step's numbers.
//! Fold order rides Issue 940 (sequential ascending folds; the ternary
//! matvec is the shipped order-pinned kernel).

#[cfg(all(feature = "mtp_draft_block", feature = "deltanet_ternary_inference"))]
use crate::deltanet::rotation::{rotate_forward_inplace, rotate_inverse_inplace};
#[cfg(all(feature = "mtp_draft_block", feature = "deltanet_ternary_inference"))]
use crate::deltanet::ternary_weights::{QwenDeltaNetTernaryWeights, ternary_matvec_parallel};
#[cfg(all(feature = "mtp_draft_block", feature = "deltanet_ternary_inference"))]
use crate::gguf_loader::MtpDraftBlock;
#[cfg(all(feature = "mtp_draft_block", feature = "deltanet_ternary_inference"))]
use crate::transformer::KVCache;
#[cfg(all(feature = "mtp_draft_block", feature = "deltanet_ternary_inference"))]
use crate::types::{Config, rmsnorm_with_gamma_eps, swiglu};

/// Grow-once scratch for [`mtp_draft_step`] — no allocation in the step
/// itself (the G4 bar the serving posture inherits from T5.5).
#[cfg(all(feature = "mtp_draft_block", feature = "deltanet_ternary_inference"))]
pub struct MtpDraftStepScratch {
    /// The primal embedding row of `last_token` (`[n_embd]`).
    pub embed_primal: Vec<f32>,
    /// The compose input `concat(enorm(embed), hnorm(h))` (`[2·n_embd]`).
    pub concat: Vec<f32>,
    /// The attention-leg residual anchor `[n_embd]` — the eh_proj output
    /// (the fork's `inpSA`; the attention residual adds onto it).
    pub inp_sa: Vec<f32>,
    /// Gated-Q output `[2·q_dim]` (q ‖ gate interleaved per head, Issue 594).
    pub qg: Vec<f32>,
    /// Q after the split `[q_dim]`.
    pub q: Vec<f32>,
    /// The attention-output gate `[q_dim]` (pre-sigmoid).
    pub gate: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub head_scores: Vec<f32>,
    /// SwiGLU gate `[mlp_hidden]`.
    pub ffn_gate: Vec<f32>,
    /// SwiGLU up `[mlp_hidden]`.
    pub ffn_up: Vec<f32>,
    /// SwiGLU hidden `[mlp_hidden]`.
    pub ffn_hidden: Vec<f32>,
    /// The forward-rotated head input `[n_embd]` (the folded lm_head's
    /// copy — the `hidden_copy` mirror in `ternary_lm_head_into`).
    pub head_rotated: Vec<f32>,
    /// The FFN-leg residual anchor `[n_embd]` — the attention-leg output
    /// saved BEFORE `post_attention_norm` (the fork's `ffn_residual`; the
    /// final add consumes the pre-norm value, never the normed one).
    pub ffn_residual: Vec<f32>,
}

#[cfg(all(feature = "mtp_draft_block", feature = "deltanet_ternary_inference"))]
impl MtpDraftStepScratch {
    pub fn new(config: &Config) -> Self {
        let n = config.n_embd;
        let q_dim = config.n_head * config.head_dim;
        let kvd = config.n_kv_head * config.head_dim;
        Self {
            embed_primal: vec![0.0; n],
            concat: vec![0.0; 2 * n],
            inp_sa: vec![0.0; n],
            qg: vec![0.0; 2 * q_dim],
            q: vec![0.0; q_dim],
            gate: vec![0.0; q_dim],
            k: vec![0.0; kvd],
            v: vec![0.0; kvd],
            attn_out: vec![0.0; q_dim],
            head_scores: vec![0.0; config.n_head * config.block_size],
            ffn_gate: vec![0.0; config.mlp_hidden],
            ffn_up: vec![0.0; config.mlp_hidden],
            ffn_hidden: vec![0.0; config.mlp_hidden],
            head_rotated: vec![0.0; n],
            ffn_residual: vec![0.0; n],
        }
    }
}

/// One teacher-forced MTP draft step (CPU REFERENCE — the oracle for every
/// later arm).
///
/// `h_prev` is the main stack's post-final-norm hidden at the last landed
/// position, in the PRIMAL basis (see the module basis contract; recover
/// from the hidden variant with `rotate_inverse_inplace`). `pos` is the
/// DRAFT position's RoPE phase (the next position after the main stack's
/// last). Writes `logits[..vocab]`; `h_draft` is left in `h_prev` (in
/// place, the caller's buffer — the next greedy draft step consumes it
/// directly).
///
/// # Errors
/// Refuses loud when `logits.len() < config.vocab_size` (a truncated out
/// buffer would silently drop the tail of the distribution).
#[cfg(all(feature = "mtp_draft_block", feature = "deltanet_ternary_inference"))]
#[allow(clippy::too_many_arguments)]
pub fn mtp_draft_step(
    block: &MtpDraftBlock,
    weights: &QwenDeltaNetTernaryWeights,
    cache: &mut KVCache,
    pos: usize,
    h_prev: &mut [f32],
    last_token: usize,
    config: &Config,
    rope_freq: &crate::rope::RopeFreqTable,
    scratch: &mut MtpDraftStepScratch,
    logits: &mut [f32],
) -> Result<(), anyhow::Error> {
    let n = config.n_embd;
    let n_head = config.n_head;
    let n_kv = config.n_kv_head;
    let hd = config.head_dim;
    let q_dim = n_head * hd;
    let kvd = n_kv * hd;
    let mlp = config.mlp_hidden;
    let vocab = config.vocab_size;
    anyhow::ensure!(
        logits.len() >= vocab,
        "mtp_draft_step: logits buffer {} < vocab {vocab} — refusing",
        logits.len()
    );
    anyhow::ensure!(h_prev.len() >= n, "h_prev shorter than n_embd — refusing");

    // 1. The embedding row → primal. token_embd.weight rows are stored
    //    LATENT on a folded file (the inverse-after-lookup contract); the
    //    inverse transform is a no-op-shaped identity on an unfolded one
    //    (`rotation = None` skips both rotate calls — byte-identical).
    let wte_row = |out: &mut [f32]| weights.wte.dequant_row_into(last_token, out);
    let mut embed = std::mem::take(&mut scratch.embed_primal);
    wte_row(&mut embed[..n]);
    if let Some(rot) = &weights.rotation {
        let signs = rot.signs_for_width(n);
        rotate_inverse_inplace(&mut embed[..n], signs, rot.block_size);
    }
    scratch.embed_primal = embed;

    // 2. The compose head: concat(enorm(embed), hnorm(h_prev)) → eh_proj.
    //    Both norms are full-width RMSNorm with a gamma (the fork's
    //    build_norm with a weight — LLM_NORM_RMS).
    rmsnorm_with_gamma_eps(
        &mut scratch.embed_primal[..n],
        &block.enorm,
        config.rms_norm_eps,
    );
    rmsnorm_with_gamma_eps(&mut h_prev[..n], &block.hnorm, config.rms_norm_eps);
    scratch.concat[..n].copy_from_slice(&scratch.embed_primal[..n]);
    scratch.concat[n..2 * n].copy_from_slice(&h_prev[..n]);
    // eh_proj is [n_embd × 2·n_embd] stored row-major (rows = out): the
    // dense matvec out[r] = Σ_c w[r·cols + c]·x[c], sequential ascending
    // fold (Issue 940).
    dense_matvec(&block.eh_proj, n, 2 * n, &scratch.concat, h_prev);

    // 3. The block: gated attention → residual → post-attn norm → SwiGLU
    //    → residual. `inpSA` = the eh_proj output (the fork's residual
    //    anchor for the attention leg) — saved in the SCRATCH (a per-call
    //    `to_vec` here allocated every step; the verdict defect, fixed —
    //    bar (c) is a pre-registered gate, not a doc aspiration).
    scratch.inp_sa[..n].copy_from_slice(&h_prev[..n]);

    // 3a. Gated Q projection + split (the Issue-594 per-head interleave).
    dense_matvec(&block.attn_q, 2 * q_dim, n, &h_prev[..n], &mut scratch.qg);
    for h in 0..n_head {
        let src = h * 2 * hd;
        let dst = h * hd;
        scratch.q[dst..dst + hd].copy_from_slice(&scratch.qg[src..src + hd]);
        scratch.gate[dst..dst + hd].copy_from_slice(&scratch.qg[src + hd..src + 2 * hd]);
    }
    dense_matvec(&block.attn_k, kvd, n, &h_prev[..n], &mut scratch.k);
    dense_matvec(&block.attn_v, kvd, n, &h_prev[..n], &mut scratch.v);

    // 3b. Per-head QK-norm (RMSNorm over each head's [head_dim] slice).
    let eps = config.rms_norm_eps;
    for h in 0..n_head {
        let off = h * hd;
        rmsnorm_with_gamma_eps(&mut scratch.q[off..off + hd], &block.attn_q_norm, eps);
    }
    for h in 0..n_kv {
        let off = h * hd;
        rmsnorm_with_gamma_eps(&mut scratch.k[off..off + hd], &block.attn_k_norm, eps);
    }

    // 3c. RoPE (partial per Issue 594 — the caller's table is built for
    //     `effective_rotary_dim`), then K/V into the draft cache.
    let rotary_dim = crate::deltanet::forward::effective_rotary_dim(config);
    if rotary_dim == hd {
        crate::rope::apply_rope_with_freq(
            &mut scratch.q,
            &mut scratch.k,
            pos,
            hd,
            rope_freq.as_slice(),
        );
    } else {
        crate::rope::apply_partial_rope_with_freq(
            &mut scratch.q,
            &mut scratch.k,
            pos,
            hd,
            rotary_dim,
            rope_freq.as_slice(),
        );
    }
    let pos_off = pos * kvd;
    cache.key[pos_off..pos_off + kvd].copy_from_slice(&scratch.k);
    cache.value[pos_off..pos_off + kvd].copy_from_slice(&scratch.v);

    // 3d. GQA attention over the draft cache (same kernel, no softcap).
    let scale = 1.0 / (hd as f32).sqrt();
    scratch.attn_out[..q_dim].fill(0.0);
    let t_n = pos + 1;
    unsafe {
        crate::transformer::attention_heads_parallel_bias(
            &scratch.q,
            &cache.key,
            &cache.value,
            &mut scratch.attn_out,
            &mut scratch.head_scores,
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

    // 3e. Output gate (Issue 594): attn *= sigmoid(gate), then wo.
    for i in 0..q_dim {
        scratch.attn_out[i] *= crate::simd::fast_sigmoid(scratch.gate[i]);
    }
    dense_matvec(
        &block.attn_output,
        n,
        q_dim,
        &scratch.attn_out[..q_dim],
        h_prev,
    );
    dense_matvec(
        &block.attn_output,
        n,
        q_dim,
        &scratch.attn_out[..q_dim],
        h_prev,
    );
    // 3f. The attention residual (onto the saved inpSA).
    for (xi, r) in h_prev[..n].iter_mut().zip(&scratch.inp_sa[..n]) {
        *xi += *r;
    }

    // 3g. Pre-FFN norm + SwiGLU + down proj + the FFN residual. The
    //     anchor is the attention-leg OUTPUT saved BEFORE the norm (the
    //     fork's `ffn_residual = cur` — the main loop's `scratch.residual`
    //     shape; norming h_prev in place without this save feeds the
    //     normed value into the final add — the compose-order probe
    //     caught exactly that).
    scratch.ffn_residual[..n].copy_from_slice(&h_prev[..n]);
    rmsnorm_with_gamma_eps(&mut h_prev[..n], &block.post_attention_norm, eps);
    dense_matvec(&block.ffn_gate, mlp, n, &h_prev[..n], &mut scratch.ffn_gate);
    dense_matvec(&block.ffn_up, mlp, n, &h_prev[..n], &mut scratch.ffn_up);
    swiglu(&mut scratch.ffn_hidden, &scratch.ffn_gate, &scratch.ffn_up);
    dense_matvec(&block.ffn_down, n, mlp, &scratch.ffn_hidden, h_prev);
    for (xi, r) in h_prev[..n].iter_mut().zip(&scratch.ffn_residual[..n]) {
        *xi += *r;
    }

    // 4. The head: shared_head_norm (else the main stack's output_norm) →
    //    forward-rotated copy → the folded lm_head matvec. Primal in,
    //    PRIMAL logits out (the fork's contract — the head matvec absorbs
    //    the rotation; the distribution needs no un-rotation). The norm
    //    runs on the COPY — `h_prev` stays the primal h_draft the caller
    //    is owed (the fork's `cur = build_norm(...)` is a new tensor;
    //     norming h_prev in place here destroyed it — the compose-order
    //     probe caught the head-normed value leaking into the caller's
    //     buffer).
    scratch.head_rotated[..n].copy_from_slice(&h_prev[..n]);
    match block.shared_head_norm.as_ref() {
        Some(g) => rmsnorm_with_gamma_eps(&mut scratch.head_rotated[..n], g, eps),
        None => rmsnorm_with_gamma_eps(&mut scratch.head_rotated[..n], &weights.final_norm, eps),
    }
    if let Some(rot) = &weights.rotation {
        let signs = rot.signs_for_width(n);
        rotate_forward_inplace(&mut scratch.head_rotated[..n], signs, rot.block_size);
    }
    match block.shared_head_head.as_ref() {
        // The real twin omits it — the dense head path is the T5.4 follow-up
        // arm; today an actual `shared_head_head` refuses loud rather than
        // silently dequantizing an unmeasured path.
        Some(_) => anyhow::bail!(
            "mtp_draft_step: nextn.shared_head_head is present but the dense-head arm is \
             not landed (T5.4) — refusing rather than guessing (Issue 1253 T5.2)"
        ),
        None => ternary_matvec_parallel(
            &weights.lm_head,
            &scratch.head_rotated[..n],
            &mut logits[..vocab],
        ),
    }
    Ok(())
}

/// The dense (f32) row-major matvec: `out[r] = Σ_c w[r·cols + c]·x[c]`,
/// sequential ascending fold — the Issue-940 order-pinned form for a
/// sync-crossing reduction. The draft step is a decode-side oracle; it
/// takes the plain scalar loop (SIMD chunking is a measured follow-up,
/// never a silent reorder).
#[cfg(all(feature = "mtp_draft_block", feature = "deltanet_ternary_inference"))]
fn dense_matvec(w: &[f32], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(w.len(), rows * cols, "dense matvec shape");
    debug_assert_eq!(x.len(), cols, "dense matvec input width");
    for (r, orow) in out[..rows].iter_mut().enumerate() {
        let row = &w[r * cols..(r + 1) * cols];
        let mut acc = 0.0f32;
        for c in 0..cols {
            acc += row[c] * x[c];
        }
        *orow = acc;
    }
}
