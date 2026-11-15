//! GDN-layer lens maps — T2c of katgpt-rs Issue 999 (Research 666).
//!
//! The qwen3.5 Gated-DeltaNet layer ([`crate::deltanet::forward`]
//! `forward_deltanet_layer`) linearized op by op, doc-level (T positions,
//! zero initial state — the fit's standalone-doc posture):
//!
//! 1. **Projections** — `in_proj_qkv` / `in_proj_z` / `in_proj_a` /
//!    `in_proj_b` are plain linear maps ([`crate::types::matmul`] +
//!    [`crate::lens::ops::matvec_t`]).
//! 2. **Causal conv1d + SiLU** — the conv is linear (exact adjoint:
//!    [`crate::lens::ops::conv1d_causal_jvp`] / `_vjp`, zero initial state =
//!    the doc-level left-pad); the SiLU after it takes the Identity rule
//!    (capture `silu'` per channel per position). ⚠ The forward applies
//!    `fast_sigmoid` inside its silu — the capture uses the exact
//!    [`crate::lens::ops::silu_grad`]; the drift is fit-side noise,
//!    disclosed (T2d pins bit-parity through the serving seam).
//! 3. **Gate scalars** — `β = σ(b)`, `decay = exp(a_log·softplus(a+dt_bias))`:
//!    exact scalar chain rule ([`gdn_gates_capture`]).
//! 4. **Tiled expand + L2 norm** — the forward copies the k/q heads, L2-
//!    normalizes the SOURCE heads, then tiles ([`crate::deltanet::forward`]
//!    `expand_l2_normalized_heads_into`). The norm keeps its TRUE derivative
//!    (`J = inv·(I − yyᵀ)`, symmetric — jvp/vjp share the form); the tile is
//!    a permutation whose adjoint sums the repeat cotangents.
//! 5. **Gated delta-rule recurrence** — TRUE derivative, the scan core
//!    ([`gdn_recurrence_jvp`] / [`gdn_recurrence_vjp`]): per head per step
//!    `S̃ = decay·S_prev`, `d = β(v − S̃k)`, `S = S̃ + d⊗k`,
//!    `y = S q·scale` — mirroring `gated_deltanet_step_par_inplace`'s
//!    arithmetic order.
//! 6. **ssm_norm** — per-head RMSNorm: the TRUE-derivative maps REUSED from
//!    T2b ([`crate::lens::attn::QkNormConsts`] — same rank-1-plus-diagonal
//!    shape, gamma = `linear_norm`).
//! 7. **silu(z) output gate** — the two-factor half rule REUSED from T2a
//!    ([`crate::lens::ops::MlpConsts`] + `half_rule_jvp`/`half_rule_vjp`,
//!    the elementwise core without the fused down-projection — `out_proj`
//!    is a separate op here).
//! 8. **out_proj** — plain linear.
//!
//! Provenance: `stdlogits/neuron-lens` @ `849bfe16` (MIT), `layers.py` GDN
//! map, ported to this crate's row-major layout and the Issue-594 row names.
//! The maps themselves allocate nothing — buffers are caller-provided
//! ([`GdnLayerScratch`]); the offline fit allocates scratch once per doc.
//! Never a serving path.
//!
//! Bit-parity note: [`gdn_layer_forward`] is the DOC-level reference the
//! tests differentiate; T2d pins it against `forward_qwen_deltanet`'s
//! decode-step seams (the conv's sliding state, the gate scalars) the same
//! way T2a's maps pin against the forward kernels.

use crate::lens::attn::{QkNormConsts, qk_norm_true_jvp, qk_norm_true_vjp};
use crate::lens::ops::{
    MlpConsts, conv1d_causal_jvp, conv1d_causal_vjp, half_rule_jvp, half_rule_vjp, matvec_t,
    sigmoid, silu, silu_grad,
};

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

/// The GDN layer's geometry (recurrence heads, head widths, conv width).
#[derive(Clone, Copy, Debug)]
pub struct GdnGeometry {
    pub t_len: usize,
    pub n_k_heads: usize,
    pub n_v_heads: usize,
    pub key_dim: usize,
    pub val_dim: usize,
    pub kernel_size: usize,
}

impl GdnGeometry {
    pub fn q_dim(&self) -> usize {
        self.n_k_heads * self.key_dim
    }

    pub fn k_dim(&self) -> usize {
        self.n_k_heads * self.key_dim
    }

    pub fn v_dim(&self) -> usize {
        self.n_v_heads * self.val_dim
    }

    /// The conv channels: `[q | k | v]` (the forward's `conv_buf` layout).
    pub fn conv_dim(&self) -> usize {
        self.q_dim() + self.k_dim() + self.v_dim()
    }

    /// The tiled expand factor (`ggml_repeat` modulo semantics, Issue 594).
    pub fn repeat(&self) -> usize {
        self.n_v_heads / self.n_k_heads
    }

    /// The read scale `1/sqrt(key_dim)` (the forward's).
    pub fn scale(&self) -> f32 {
        1.0 / (self.key_dim as f32).sqrt()
    }

    /// One head's state block `[val_dim][key_dim]`, all heads in the doc.
    pub fn states_len(&self) -> usize {
        self.t_len * self.n_v_heads * self.val_dim * self.key_dim
    }
}

// ---------------------------------------------------------------------------
// L2 norm — true derivative (symmetric rank-1-plus-scalar map)
// ---------------------------------------------------------------------------

/// Captured L2-norm constants for one buffer of heads (the forward's
/// `l2_normalize`: `y = x/‖x‖`, identity when `‖x‖² == 0`).
#[derive(Clone, Copy, Debug)]
pub struct L2NormConsts<'a> {
    /// The normalized output `y` (`[heads · head_dim]`).
    pub y: &'a [f32],
    /// Per-head `1/‖x‖` (`[heads]`); `0` marks the degenerate identity.
    pub inv: &'a [f32],
}

/// Capture `y` + per-head `1/‖x‖` (one position, all heads of a buffer).
pub fn l2_norm_capture(x: &[f32], head_dim: usize, y_out: &mut [f32], inv_out: &mut [f32]) {
    let heads = x.len() / head_dim;
    debug_assert_eq!(y_out.len(), x.len(), "l2_norm_capture: y_out shape");
    debug_assert_eq!(inv_out.len(), heads, "l2_norm_capture: inv_out heads");
    for (h, inv_slot) in inv_out.iter_mut().enumerate().take(heads) {
        let off = h * head_dim;
        let head = &x[off..off + head_dim];
        let mut sum_sq = 0.0f32;
        for &v in head {
            sum_sq += v * v;
        }
        if sum_sq > 0.0 {
            let inv = 1.0 / sum_sq.sqrt();
            *inv_slot = inv;
            for (i, y_slot) in y_out[off..off + head_dim].iter_mut().enumerate() {
                *y_slot = head[i] * inv;
            }
        } else {
            // the forward's guard: identity — the tangent map is the identity
            *inv_slot = 0.0;
            y_out[off..off + head_dim].copy_from_slice(head);
        }
    }
}

/// L2-norm tangent map (true derivative): `dv = inv·(v − y·⟨y, v⟩)` per head.
///
/// The Jacobian `inv·(I − yyᵀ)` is symmetric, so this one function serves
/// both `jvp` and `vjp` ([`l2_norm_vjp`] delegates). In place — the input
/// buffer is overwritten with its image. The degenerate heads (`inv == 0`)
/// pass through untouched, mirroring the forward's guard.
pub fn l2_norm_apply(c: &L2NormConsts, head_dim: usize, v: &mut [f32]) {
    let heads = c.y.len() / head_dim;
    debug_assert_eq!(c.inv.len(), heads, "l2_norm_apply: inv heads");
    debug_assert_eq!(v.len(), c.y.len(), "l2_norm_apply: v shape");
    for h in 0..heads {
        let off = h * head_dim;
        let inv = c.inv[h];
        if inv == 0.0 {
            continue;
        }
        let y_h = &c.y[off..off + head_dim];
        let mut dot = 0.0f32;
        for i in 0..head_dim {
            dot += y_h[i] * v[off + i];
        }
        for i in 0..head_dim {
            v[off + i] = inv * (v[off + i] - y_h[i] * dot);
        }
    }
}

/// L2-norm `jvp` (in place on the tangent).
pub fn l2_norm_jvp(c: &L2NormConsts, head_dim: usize, dx: &mut [f32]) {
    l2_norm_apply(c, head_dim, dx);
}

/// L2-norm `vjp` — the exact adjoint (the map is symmetric).
pub fn l2_norm_vjp(c: &L2NormConsts, head_dim: usize, dy: &mut [f32]) {
    l2_norm_apply(c, head_dim, dy);
}

// ---------------------------------------------------------------------------
// Tiled expand — permutation and its summing adjoint
// ---------------------------------------------------------------------------

/// Tiled-expand `jvp`: `dst` head `j·repeat + r` receives `src` head `j`'s
/// tangent (the forward's `copy_within` tiling).
pub fn expand_tiled_jvp(
    dsrc: &[f32],
    n_src_heads: usize,
    head_dim: usize,
    repeat: usize,
    ddst: &mut [f32],
) {
    let block = n_src_heads * head_dim;
    debug_assert_eq!(dsrc.len(), block, "expand_tiled_jvp: dsrc shape");
    debug_assert_eq!(ddst.len(), block * repeat, "expand_tiled_jvp: ddst shape");
    for r in 0..repeat {
        ddst[r * block..(r + 1) * block].copy_from_slice(dsrc);
    }
}

/// Tiled-expand `vjp` — the exact adjoint: SUM the repeat cotangents back
/// into the source head (the tile duplicates, the adjoint accumulates).
pub fn expand_tiled_vjp(
    gdst: &[f32],
    n_src_heads: usize,
    head_dim: usize,
    repeat: usize,
    gsrc: &mut [f32],
) {
    let block = n_src_heads * head_dim;
    debug_assert_eq!(gdst.len(), block * repeat, "expand_tiled_vjp: gdst shape");
    debug_assert_eq!(gsrc.len(), block, "expand_tiled_vjp: gsrc shape");
    for v in gsrc.iter_mut() {
        *v = 0.0;
    }
    for r in 0..repeat {
        for i in 0..block {
            gsrc[i] += gdst[r * block + i];
        }
    }
}

// ---------------------------------------------------------------------------
// Gate scalars — exact chain rule
// ---------------------------------------------------------------------------

/// Numerically-stable softplus (the forward's guarded form; its derivative
/// is the logistic sigmoid).
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        0.0
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Capture the gate scalars for one position (per head):
///
/// - `β = σ(b)`; its tangent coefficient `spb = σ'(b)`.
/// - `a_val = a_raw + dt_bias`; `g = a_log·softplus(a_val)`;
///   `decay = exp(g)`; its tangent coefficient
///   `ddecay/da_raw = decay·a_log·σ(a_val)` (`dt_bias` and `a_log` are
///   WEIGHTS — constants of the linearization).
///
/// The forward computes β via `fast_sigmoid` — the capture uses the exact
/// [`sigmoid`]; disclosed fit-side drift (see the module doc).
#[allow(clippy::too_many_arguments)]
pub fn gdn_gates_capture(
    a_raw: &[f32],
    b_raw: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    beta_out: &mut [f32],
    decay_out: &mut [f32],
    dbeta_coef_out: &mut [f32],
    ddecay_coef_out: &mut [f32],
) {
    let n = a_raw.len();
    debug_assert_eq!(b_raw.len(), n, "gdn_gates_capture: b heads");
    debug_assert_eq!(a_log.len(), n, "gdn_gates_capture: a_log heads");
    debug_assert_eq!(dt_bias.len(), n, "gdn_gates_capture: dt_bias heads");
    debug_assert_eq!(beta_out.len(), n, "gdn_gates_capture: beta_out heads");
    debug_assert_eq!(decay_out.len(), n, "gdn_gates_capture: decay_out heads");
    debug_assert_eq!(dbeta_coef_out.len(), n, "gdn_gates_capture: coef heads");
    debug_assert_eq!(ddecay_coef_out.len(), n, "gdn_gates_capture: coef heads");
    for h in 0..n {
        let s = sigmoid(b_raw[h]);
        beta_out[h] = s;
        dbeta_coef_out[h] = s * (1.0 - s);
        let a_val = a_raw[h] + dt_bias[h];
        let g = a_log[h] * softplus(a_val);
        let decay = g.exp();
        decay_out[h] = decay;
        ddecay_coef_out[h] = decay * a_log[h] * sigmoid(a_val);
    }
}

// ---------------------------------------------------------------------------
// Gated delta-rule recurrence — true derivative scan core
// ---------------------------------------------------------------------------

/// Captured recurrence constants for one doc (states + the post-expand/
/// norm activations the scan differentiates through).
#[derive(Clone, Copy, Debug)]
pub struct GdnRecurrenceConsts<'a> {
    pub t_len: usize,
    pub n_v_heads: usize,
    pub key_dim: usize,
    pub val_dim: usize,
    /// The read scale `1/sqrt(key_dim)`.
    pub scale: f32,
    /// Expanded + L2-normalized q (`[T · n_v · key]`).
    pub q: &'a [f32],
    /// Expanded + L2-normalized k (`[T · n_v · key]`).
    pub k: &'a [f32],
    /// Post-conv v (`[T · n_v · val]`).
    pub v: &'a [f32],
    /// Per head per step (`[T · n_v]`).
    pub beta: &'a [f32],
    /// Per head per step (`[T · n_v]`).
    pub decay: &'a [f32],
    /// Post-update states `S_t` (`[T · n_v · val · key]`, row-major
    /// `[val][key]` per head); `S_0 = 0` (the doc-level fit's posture).
    pub states: &'a [f32],
}

/// Doc-level recurrence forward — the FD reference and T2d's capture.
///
/// Mirrors `gated_deltanet_step_par_inplace`'s arithmetic order per row:
/// decay in place (`s_row *= g`), `kv = ⟨s_row, k⟩`, `d = β(v_r − kv)`,
/// `s_row += d·k`, `o_r = ⟨s_row, q⟩·scale`. The dots fold sequentially
/// (the serving kernel's SIMD tree may differ in the last ulp; T2d captures
/// through the serving seam when it needs bit parity). Zero initial state.
#[allow(clippy::too_many_arguments)]
pub fn gdn_recurrence_capture(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    beta: &[f32],
    decay: &[f32],
    t_len: usize,
    n_v_heads: usize,
    key_dim: usize,
    val_dim: usize,
    states_out: &mut [f32],
    y_out: &mut [f32],
) {
    let per_head = val_dim * key_dim;
    debug_assert_eq!(
        q.len(),
        t_len * n_v_heads * key_dim,
        "gdn_rec capture: q shape"
    );
    debug_assert_eq!(k.len(), q.len(), "gdn_rec capture: k shape");
    debug_assert_eq!(
        v.len(),
        t_len * n_v_heads * val_dim,
        "gdn_rec capture: v shape"
    );
    debug_assert_eq!(beta.len(), t_len * n_v_heads, "gdn_rec capture: beta shape");
    debug_assert_eq!(decay.len(), beta.len(), "gdn_rec capture: decay shape");
    debug_assert_eq!(
        states_out.len(),
        t_len * n_v_heads * per_head,
        "gdn_rec capture: states shape"
    );
    debug_assert_eq!(
        y_out.len(),
        t_len * n_v_heads * val_dim,
        "gdn_rec capture: y shape"
    );
    let scale = 1.0 / (key_dim as f32).sqrt();

    for t in 0..t_len {
        for h in 0..n_v_heads {
            let (g, b) = (decay[t * n_v_heads + h], beta[t * n_v_heads + h]);
            let cur_off = (t * n_v_heads + h) * per_head;
            if t > 0 {
                // prev → cur copy BEFORE the mutable borrow (disjoint ranges,
                // but index-level copies keep the borrow checker trivially
                // happy and avoid a temporary).
                let prev_off = ((t - 1) * n_v_heads + h) * per_head;
                for i in 0..per_head {
                    states_out[cur_off + i] = states_out[prev_off + i];
                }
            }
            let cur = &mut states_out[cur_off..cur_off + per_head];
            if t == 0 {
                cur.fill(0.0);
            }
            let q_head = &q[t * n_v_heads * key_dim + h * key_dim
                ..t * n_v_heads * key_dim + (h + 1) * key_dim];
            let k_head = &k[t * n_v_heads * key_dim + h * key_dim
                ..t * n_v_heads * key_dim + (h + 1) * key_dim];
            let v_head = &v[t * n_v_heads * val_dim + h * val_dim
                ..t * n_v_heads * val_dim + (h + 1) * val_dim];
            let y_head = &mut y_out[t * n_v_heads * val_dim + h * val_dim
                ..t * n_v_heads * val_dim + (h + 1) * val_dim];
            for (r, s_row) in cur.chunks_exact_mut(key_dim).enumerate() {
                for c in s_row.iter_mut() {
                    *c *= g;
                }
                let mut kv = 0.0f32;
                for c in 0..key_dim {
                    kv += s_row[c] * k_head[c];
                }
                let d = b * (v_head[r] - kv);
                for c in 0..key_dim {
                    s_row[c] += d * k_head[c];
                }
                let mut o = 0.0f32;
                for c in 0..key_dim {
                    o += s_row[c] * q_head[c];
                }
                y_head[r] = o * scale;
            }
        }
    }
}

/// Recurrence `jvp` — the TRUE forward-mode derivative scan.
///
/// Tangents `(dq, dk, dv, dbeta, ddecay)` (activation layouts; `dq`/`dk` in
/// the EXPANDED `[T · n_v · key]` layout) → `dy` `[T · n_v · val]`. The
/// update tangent is `dS = dS̃ + dd⊗k + d·dk` — BOTH factors of the outer
/// product `d⊗k` carry tangent terms (the adjoint gate catches the missing
/// `d·dk` as a k-only identity break). The running tangent state `ds` and
/// the per-step decayed-state materialization `spre` are caller scratch,
/// both `[n_v · val · key]`; `ds` is zeroed here.
#[allow(clippy::too_many_arguments)]
pub fn gdn_recurrence_jvp(
    c: &GdnRecurrenceConsts,
    dq: &[f32],
    dk: &[f32],
    dv: &[f32],
    dbeta: &[f32],
    ddecay: &[f32],
    ds: &mut [f32],
    spre: &mut [f32],
    dy_out: &mut [f32],
) {
    let (t_len, n_v_heads, key_dim, val_dim) = (c.t_len, c.n_v_heads, c.key_dim, c.val_dim);
    let per_head = val_dim * key_dim;
    let scale = c.scale;
    debug_assert_eq!(
        dq.len(),
        t_len * n_v_heads * key_dim,
        "gdn_rec jvp: dq shape"
    );
    debug_assert_eq!(dk.len(), dq.len(), "gdn_rec jvp: dk shape");
    debug_assert_eq!(
        dv.len(),
        t_len * n_v_heads * val_dim,
        "gdn_rec jvp: dv shape"
    );
    debug_assert_eq!(dbeta.len(), t_len * n_v_heads, "gdn_rec jvp: dbeta shape");
    debug_assert_eq!(ddecay.len(), dbeta.len(), "gdn_rec jvp: ddecay shape");
    debug_assert_eq!(ds.len(), n_v_heads * per_head, "gdn_rec jvp: ds scratch");
    debug_assert_eq!(
        spre.len(),
        n_v_heads * per_head,
        "gdn_rec jvp: spre scratch"
    );
    debug_assert_eq!(
        dy_out.len(),
        t_len * n_v_heads * val_dim,
        "gdn_rec jvp: dy shape"
    );
    for x in ds.iter_mut() {
        *x = 0.0;
    }

    for t in 0..t_len {
        for h in 0..n_v_heads {
            let idx = t * n_v_heads + h;
            let (g, b, dg, db) = (c.decay[idx], c.beta[idx], ddecay[idx], dbeta[idx]);
            // tangent state: dS̃ = g·dS_prev + dg·S_prev (elementwise);
            // S̃ = g·S_prev materialized for the delta terms (S_0 = 0).
            // PER-HEAD blocks — heads never share the running state.
            let ds_head = &mut ds[h * per_head..(h + 1) * per_head];
            let spre_head = &mut spre[h * per_head..(h + 1) * per_head];
            let prev_off = if t == 0 {
                0
            } else {
                ((t - 1) * n_v_heads + h) * per_head
            };
            for i in 0..per_head {
                let sp = if t == 0 { 0.0 } else { c.states[prev_off + i] };
                ds_head[i] = g * ds_head[i] + dg * sp;
                spre_head[i] = g * sp;
            }
            let cur_off = idx * per_head;
            let s_cur = &c.states[cur_off..cur_off + per_head];
            let q_head = &c.q[t * n_v_heads * key_dim + h * key_dim
                ..t * n_v_heads * key_dim + (h + 1) * key_dim];
            let k_head = &c.k[t * n_v_heads * key_dim + h * key_dim
                ..t * n_v_heads * key_dim + (h + 1) * key_dim];
            let dk_head = &dk[t * n_v_heads * key_dim + h * key_dim
                ..t * n_v_heads * key_dim + (h + 1) * key_dim];
            let dq_head = &dq[t * n_v_heads * key_dim + h * key_dim
                ..t * n_v_heads * key_dim + (h + 1) * key_dim];
            let v_head = &c.v[t * n_v_heads * val_dim + h * val_dim
                ..t * n_v_heads * val_dim + (h + 1) * val_dim];
            let dv_head = &dv[t * n_v_heads * val_dim + h * val_dim
                ..t * n_v_heads * val_dim + (h + 1) * val_dim];
            let dy_head = &mut dy_out[t * n_v_heads * val_dim + h * val_dim
                ..t * n_v_heads * val_dim + (h + 1) * val_dim];
            for r in 0..val_dim {
                let ds_row = &mut ds_head[r * key_dim..(r + 1) * key_dim];
                let spre_row = &spre_head[r * key_dim..(r + 1) * key_dim];
                let mut sk = 0.0f32;
                let mut kv_dot = 0.0f32;
                for ci in 0..key_dim {
                    sk += spre_row[ci] * k_head[ci];
                    kv_dot += ds_row[ci] * k_head[ci] + spre_row[ci] * dk_head[ci];
                }
                let vm = v_head[r] - sk;
                let d_r = b * vm; // the FORWARD's delta value (for the d·dk term)
                let dd = b * (dv_head[r] - kv_dot) + db * vm;
                for ci in 0..key_dim {
                    ds_row[ci] += dd * k_head[ci] + d_r * dk_head[ci];
                }
                let mut dy_read = 0.0f32;
                let mut dy_state = 0.0f32;
                let s_row = &s_cur[r * key_dim..(r + 1) * key_dim];
                for ci in 0..key_dim {
                    dy_read += ds_row[ci] * q_head[ci];
                    dy_state += s_row[ci] * dq_head[ci];
                }
                dy_head[r] = (dy_read + dy_state) * scale;
            }
        }
    }
}

/// Recurrence `vjp` — the exact adjoint reverse scan.
///
/// Cotangent `gy` `[T · n_v · val]` (and `y`, the captured forward output,
/// unused here but kept for signature symmetry with the attention vjp —
/// the softmax pullback needs it, this scan does not) → gradients
/// `(gq, gk, gv, gbeta, gdecay)` in the activation layouts (zeroed here).
/// The running cotangent state `gs` and the per-step `gspre` buffer are
/// caller scratch, both `[n_v · val · key]`; `gs` is zeroed here.
///
/// Derivation per step (reverse order): the read adds `scale·gy⊗q` to the
/// state cotangent and `scale·Sᵀ·gy` to `gq`; the outer product splits the
/// state cotangent into `gS̃` and `(gk via d, gd)`; the delta scalar adds
/// `gβ`, `gv`, and the rank-1 terms to `gk`/`gS̃`; the decay closes with
/// `gdecay = ⟨gS̃, S_prev⟩` and `gS_prev = decay·gS̃`.
#[allow(clippy::too_many_arguments)]
pub fn gdn_recurrence_vjp(
    c: &GdnRecurrenceConsts,
    _y: &[f32],
    gy: &[f32],
    gq: &mut [f32],
    gk: &mut [f32],
    gv: &mut [f32],
    gbeta: &mut [f32],
    gdecay: &mut [f32],
    gs: &mut [f32],
    gspre: &mut [f32],
) {
    let (t_len, n_v_heads, key_dim, val_dim) = (c.t_len, c.n_v_heads, c.key_dim, c.val_dim);
    let per_head = val_dim * key_dim;
    let scale = c.scale;
    debug_assert_eq!(
        gy.len(),
        t_len * n_v_heads * val_dim,
        "gdn_rec vjp: gy shape"
    );
    debug_assert_eq!(
        gq.len(),
        t_len * n_v_heads * key_dim,
        "gdn_rec vjp: gq shape"
    );
    debug_assert_eq!(gk.len(), gq.len(), "gdn_rec vjp: gk shape");
    debug_assert_eq!(
        gv.len(),
        t_len * n_v_heads * val_dim,
        "gdn_rec vjp: gv shape"
    );
    debug_assert_eq!(gbeta.len(), t_len * n_v_heads, "gdn_rec vjp: gbeta shape");
    debug_assert_eq!(gdecay.len(), gbeta.len(), "gdn_rec vjp: gdecay shape");
    debug_assert_eq!(gs.len(), n_v_heads * per_head, "gdn_rec vjp: gs scratch");
    debug_assert_eq!(
        gspre.len(),
        n_v_heads * per_head,
        "gdn_rec vjp: gspre scratch"
    );
    for x in gq.iter_mut() {
        *x = 0.0;
    }
    for x in gk.iter_mut() {
        *x = 0.0;
    }
    for x in gv.iter_mut() {
        *x = 0.0;
    }
    for x in gbeta.iter_mut() {
        *x = 0.0;
    }
    for x in gdecay.iter_mut() {
        *x = 0.0;
    }
    for x in gs.iter_mut() {
        *x = 0.0;
    }

    for t in (0..t_len).rev() {
        for h in 0..n_v_heads {
            let idx = t * n_v_heads + h;
            let (g, b) = (c.decay[idx], c.beta[idx]);
            let cur_off = idx * per_head;
            let s_cur = &c.states[cur_off..cur_off + per_head];
            let prev_off = if t == 0 {
                0
            } else {
                ((t - 1) * n_v_heads + h) * per_head
            };
            let q_head = &c.q[t * n_v_heads * key_dim + h * key_dim
                ..t * n_v_heads * key_dim + (h + 1) * key_dim];
            let k_head = &c.k[t * n_v_heads * key_dim + h * key_dim
                ..t * n_v_heads * key_dim + (h + 1) * key_dim];
            let v_head = &c.v[t * n_v_heads * val_dim + h * val_dim
                ..t * n_v_heads * val_dim + (h + 1) * val_dim];
            let gy_head = &gy[t * n_v_heads * val_dim + h * val_dim
                ..t * n_v_heads * val_dim + (h + 1) * val_dim];
            let gq_head = &mut gq[t * n_v_heads * key_dim + h * key_dim
                ..t * n_v_heads * key_dim + (h + 1) * key_dim];
            let gk_head = &mut gk[t * n_v_heads * key_dim + h * key_dim
                ..t * n_v_heads * key_dim + (h + 1) * key_dim];
            let gv_head = &mut gv[t * n_v_heads * val_dim + h * val_dim
                ..t * n_v_heads * val_dim + (h + 1) * val_dim];
            let gs_head = &mut gs[h * per_head..(h + 1) * per_head];
            let gspre_head = &mut gspre[h * per_head..(h + 1) * per_head];
            let mut gb_acc = 0.0f32;
            // 1. read: gq += scale·Sᵀ·gy ; gs += scale·gy⊗q
            for r in 0..val_dim {
                let gy_r = gy_head[r] * scale;
                let s_row = &s_cur[r * key_dim..(r + 1) * key_dim];
                let gs_row = &mut gs_head[r * key_dim..(r + 1) * key_dim];
                for ci in 0..key_dim {
                    gq_head[ci] += gy_r * s_row[ci];
                    gs_row[ci] += gy_r * q_head[ci];
                }
            }
            // 2. S̃ = decay·S_prev materialization
            for (i, gspre_slot) in gspre_head.iter_mut().enumerate() {
                let sp = if t == 0 { 0.0 } else { c.states[prev_off + i] };
                *gspre_slot = g * sp;
            }
            // 3. outer product + delta
            for r in 0..val_dim {
                let spre_row = &gspre_head[r * key_dim..(r + 1) * key_dim];
                let gs_row = &mut gs_head[r * key_dim..(r + 1) * key_dim];
                let mut sk = 0.0f32;
                for ci in 0..key_dim {
                    sk += spre_row[ci] * k_head[ci];
                }
                let d_r = b * (v_head[r] - sk);
                let mut gd_r = 0.0f32;
                for ci in 0..key_dim {
                    gd_r += gs_row[ci] * k_head[ci];
                }
                // outer-product term: gk += gs_row · d_r
                for ci in 0..key_dim {
                    gk_head[ci] += gs_row[ci] * d_r;
                }
                // delta term
                let vm = v_head[r] - sk;
                gb_acc += gd_r * vm;
                gv_head[r] += b * gd_r;
                let gsk_r = -b * gd_r;
                for ci in 0..key_dim {
                    gk_head[ci] += gsk_r * spre_row[ci];
                    // gS̃ accumulation in place: gs_row becomes gS̃ row
                    gs_row[ci] += gsk_r * k_head[ci];
                }
            }
            gbeta[idx] = gb_acc;
            // 4. decay: gdecay = ⟨gS̃, S_prev⟩ ; gS_prev = g·gS̃
            let mut gdec = 0.0f32;
            for (i, g_slot) in gs_head.iter_mut().enumerate() {
                let sp = if t == 0 { 0.0 } else { c.states[prev_off + i] };
                gdec += *g_slot * sp;
                *g_slot *= g;
            }
            gdecay[idx] = gdec;
        }
    }
}

// ---------------------------------------------------------------------------
// Layer composition — weights, capture, consts, scratch, jvp/vjp
// ---------------------------------------------------------------------------

/// The GDN layer's weight slices (row-major `[rows, cols]`, `y = W @ x`).
#[derive(Clone, Copy, Debug)]
pub struct GdnLayerWeights<'a> {
    /// `in_proj_qkv` — `[conv_dim][n_embd]`.
    pub w_qkv: &'a [f32],
    /// `in_proj_z` — `[v_dim][n_embd]`.
    pub w_z: &'a [f32],
    /// `in_proj_a` — `[n_v][n_embd]`.
    pub w_a: &'a [f32],
    /// `in_proj_b` — `[n_v][n_embd]`.
    pub w_b: &'a [f32],
    /// `out_proj` — `[n_embd][v_dim]`.
    pub w_out: &'a [f32],
    /// `conv1d_weight` — `[conv_dim][kernel]` (depthwise).
    pub conv_w: &'a [f32],
    /// `a_log` — `[n_v]` (already `-exp(A_log_raw)` from the GGUF, Issue 594).
    pub a_log: &'a [f32],
    /// `dt_bias` — `[n_v]`.
    pub dt_bias: &'a [f32],
    /// `linear_norm` — `[val_dim]` (per-head ssm_norm gamma).
    pub linear_norm: &'a [f32],
    pub n_embd: usize,
    pub rms_eps: f64,
}

/// The captured layer activations for one doc ([`gdn_layer_forward`]'s
/// output — the FD reference values AND the constants the maps consume).
#[derive(Default)]
pub struct GdnLayerCapture {
    /// `in_proj_qkv` output (the conv's input) `[T · conv_dim]`.
    pub qkv_proj: Vec<f32>,
    /// `silu'` of the conv pre-activation `[T · conv_dim]`.
    pub conv_silu_grad: Vec<f32>,
    /// Post-conv q / k / v `[T · q_dim]` / `[T · k_dim]` / `[T · v_dim]`.
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    /// Gate values + tangent coefficients `[T · n_v]`.
    pub beta: Vec<f32>,
    pub decay: Vec<f32>,
    pub dbeta_coef: Vec<f32>,
    pub ddecay_coef: Vec<f32>,
    /// Expand + L2-normalized q / k `[T · n_v · key]`.
    pub qn: Vec<f32>,
    pub kn: Vec<f32>,
    /// Per-source-head `1/‖x‖` `[T · n_k]`.
    pub qn_inv: Vec<f32>,
    pub kn_inv: Vec<f32>,
    /// Recurrence states `[T · n_v · val · key]`.
    pub states: Vec<f32>,
    /// Recurrence read (pre-ssm_norm) `[T · v_dim]`.
    pub y_pre: Vec<f32>,
    /// ssm_norm per-head rsqrt `[T · n_v]`.
    pub norm_r: Vec<f32>,
    /// ssm_norm output `[T · v_dim]` (the gate's value factor).
    pub normed: Vec<f32>,
    /// `silu(z)` and `silu'(z)` `[T · v_dim]`.
    pub gate_s: Vec<f32>,
    pub gate_sp: Vec<f32>,
    /// The layer output `[T · n_embd]`.
    pub y: Vec<f32>,
}

/// Doc-level GDN-layer forward — mirrors `deltanet_layer_recurrent_body`'s
/// op order over a standalone doc (zero conv/state seeds). This is the FD
/// reference the tests differentiate AND T2d's capture pass.
#[allow(clippy::too_many_arguments)]
pub fn gdn_layer_forward(
    x: &[f32],
    w: &GdnLayerWeights,
    geo: &GdnGeometry,
    cap: &mut GdnLayerCapture,
) {
    let t_len = geo.t_len;
    let n_embd = w.n_embd;
    let conv_dim = geo.conv_dim();
    let v_dim = geo.v_dim();
    let n_v = geo.n_v_heads;
    let n_k = geo.n_k_heads;
    let key_dim = geo.key_dim;
    let val_dim = geo.val_dim;
    let q_dim = geo.q_dim();
    let k_dim = geo.k_dim();
    debug_assert_eq!(x.len(), t_len * n_embd, "gdn_layer_forward: x shape");
    debug_assert_eq!(w.w_qkv.len(), conv_dim * n_embd, "w_qkv shape");
    debug_assert_eq!(w.w_z.len(), v_dim * n_embd, "w_z shape");
    debug_assert_eq!(w.w_a.len(), n_v * n_embd, "w_a shape");
    debug_assert_eq!(w.w_b.len(), n_v * n_embd, "w_b shape");
    debug_assert_eq!(w.w_out.len(), n_embd * v_dim, "w_out shape");
    debug_assert_eq!(w.conv_w.len(), conv_dim * geo.kernel_size, "conv_w shape");

    // 1-4. Projections + gate captures (per position).
    cap.qkv_proj = vec![0.0; t_len * conv_dim];
    cap.beta = vec![0.0; t_len * n_v];
    cap.decay = vec![0.0; t_len * n_v];
    cap.dbeta_coef = vec![0.0; t_len * n_v];
    cap.ddecay_coef = vec![0.0; t_len * n_v];
    cap.gate_s = vec![0.0; t_len * v_dim];
    cap.gate_sp = vec![0.0; t_len * v_dim];
    let mut z = vec![0.0f32; v_dim];
    let mut a = vec![0.0f32; n_v];
    let mut b = vec![0.0f32; n_v];
    for t in 0..t_len {
        let x_t = &x[t * n_embd..(t + 1) * n_embd];
        crate::types::matmul(
            &mut cap.qkv_proj[t * conv_dim..(t + 1) * conv_dim],
            w.w_qkv,
            x_t,
            conv_dim,
            n_embd,
        );
        crate::types::matmul(&mut z, w.w_z, x_t, v_dim, n_embd);
        crate::types::matmul(&mut a, w.w_a, x_t, n_v, n_embd);
        crate::types::matmul(&mut b, w.w_b, x_t, n_v, n_embd);
        gdn_gates_capture(
            &a,
            &b,
            w.a_log,
            w.dt_bias,
            &mut cap.beta[t * n_v..(t + 1) * n_v],
            &mut cap.decay[t * n_v..(t + 1) * n_v],
            &mut cap.dbeta_coef[t * n_v..(t + 1) * n_v],
            &mut cap.ddecay_coef[t * n_v..(t + 1) * n_v],
        );
        for (i, z_slot) in z.iter_mut().enumerate() {
            let z_val = *z_slot;
            cap.gate_s[t * v_dim + i] = silu(z_val);
            cap.gate_sp[t * v_dim + i] = silu_grad(z_val);
        }
    }

    // 5. Causal conv1d (doc-level, zero initial state) + SiLU.
    let mut conv_pre = vec![0.0f32; t_len * conv_dim];
    // the linear conv map: conv1d_causal_jvp IS the forward of the linear op
    conv1d_causal_jvp(
        w.conv_w,
        geo.kernel_size,
        &cap.qkv_proj,
        conv_dim,
        &mut conv_pre,
    );
    cap.conv_silu_grad = vec![0.0; t_len * conv_dim];
    let mut conv_out = vec![0.0f32; t_len * conv_dim];
    for i in 0..t_len * conv_dim {
        cap.conv_silu_grad[i] = silu_grad(conv_pre[i]);
        conv_out[i] = conv_pre[i] * sigmoid(conv_pre[i]);
    }

    // 6. Split [q | k | v].
    cap.q = vec![0.0; t_len * q_dim];
    cap.k = vec![0.0; t_len * k_dim];
    cap.v = vec![0.0; t_len * v_dim];
    for t in 0..t_len {
        let row = &conv_out[t * conv_dim..(t + 1) * conv_dim];
        cap.q[t * q_dim..(t + 1) * q_dim].copy_from_slice(&row[..q_dim]);
        cap.k[t * k_dim..(t + 1) * k_dim].copy_from_slice(&row[q_dim..q_dim + k_dim]);
        cap.v[t * v_dim..(t + 1) * v_dim].copy_from_slice(&row[q_dim + k_dim..]);
    }

    // 7-8. L2-normalize source heads, then tile (expand).
    let block = n_k * key_dim;
    let repeat = geo.repeat();
    cap.qn = vec![0.0; t_len * n_v * key_dim];
    cap.kn = vec![0.0; t_len * n_v * key_dim];
    cap.qn_inv = vec![0.0; t_len * n_k];
    cap.kn_inv = vec![0.0; t_len * n_k];
    let mut y_src = vec![0.0f32; block];
    let mut inv_src = vec![0.0f32; n_k];
    for t in 0..t_len {
        l2_norm_capture(
            &cap.q[t * q_dim..(t + 1) * q_dim],
            key_dim,
            &mut y_src,
            &mut inv_src,
        );
        cap.qn_inv[t * n_k..(t + 1) * n_k].copy_from_slice(&inv_src);
        expand_tiled_jvp(
            &y_src,
            n_k,
            key_dim,
            repeat,
            &mut cap.qn[t * n_v * key_dim..(t + 1) * n_v * key_dim],
        );
        l2_norm_capture(
            &cap.k[t * k_dim..(t + 1) * k_dim],
            key_dim,
            &mut y_src,
            &mut inv_src,
        );
        cap.kn_inv[t * n_k..(t + 1) * n_k].copy_from_slice(&inv_src);
        expand_tiled_jvp(
            &y_src,
            n_k,
            key_dim,
            repeat,
            &mut cap.kn[t * n_v * key_dim..(t + 1) * n_v * key_dim],
        );
    }

    // 9. Gated delta-rule recurrence.
    cap.states = vec![0.0; geo.states_len()];
    cap.y_pre = vec![0.0; t_len * v_dim];
    gdn_recurrence_capture(
        &cap.qn,
        &cap.kn,
        &cap.v,
        &cap.beta,
        &cap.decay,
        t_len,
        n_v,
        key_dim,
        val_dim,
        &mut cap.states,
        &mut cap.y_pre,
    );

    // 10. ssm_norm (per-head RMSNorm) + 11. silu(z) gate + 12. out_proj.
    cap.norm_r = vec![0.0; t_len * n_v];
    cap.normed = vec![0.0; t_len * v_dim];
    cap.y = vec![0.0; t_len * n_embd];
    let mut gate_out = vec![0.0f32; v_dim];
    for t in 0..t_len {
        QkNormConsts::capture(
            &cap.y_pre[t * v_dim..(t + 1) * v_dim],
            w.linear_norm,
            val_dim,
            w.rms_eps,
            &mut cap.norm_r[t * n_v..(t + 1) * n_v],
            &mut cap.normed[t * v_dim..(t + 1) * v_dim],
        );
        for (i, g_slot) in gate_out.iter_mut().enumerate() {
            *g_slot = cap.gate_s[t * v_dim + i] * cap.normed[t * v_dim + i];
        }
        crate::types::matmul(
            &mut cap.y[t * n_embd..(t + 1) * n_embd],
            w.w_out,
            &gate_out,
            n_embd,
            v_dim,
        );
    }
}

/// Borrowed view over the weights + captured constants the layer maps need.
#[derive(Clone, Copy)]
pub struct GdnLayerConsts<'a> {
    pub w: GdnLayerWeights<'a>,
    pub geo: GdnGeometry,
    pub cap: &'a GdnLayerCapture,
}

/// Caller-provided scratch for [`gdn_layer_jvp`] / [`gdn_layer_vjp`]
/// (allocate once per doc via [`GdnLayerScratch::new`]; the maps themselves
/// allocate nothing).
pub struct GdnLayerScratch {
    dqkv_raw: Vec<f32>,
    dconv: Vec<f32>,
    dqkv: Vec<f32>,
    dq: Vec<f32>,
    dk: Vec<f32>,
    dv: Vec<f32>,
    dz: Vec<f32>,
    da: Vec<f32>,
    db: Vec<f32>,
    dbeta: Vec<f32>,
    ddecay: Vec<f32>,
    dsrc: Vec<f32>,
    dqn: Vec<f32>,
    dkn: Vec<f32>,
    ds: Vec<f32>,
    spre: Vec<f32>,
    dy_pre: Vec<f32>,
    dnormed: Vec<f32>,
    dout: Vec<f32>,
    gout: Vec<f32>,
    gdz: Vec<f32>,
    gdu: Vec<f32>,
    gdy_pre: Vec<f32>,
    gdqn: Vec<f32>,
    gdkn: Vec<f32>,
    gdv: Vec<f32>,
    gbeta: Vec<f32>,
    gdecay: Vec<f32>,
    gdsrc: Vec<f32>,
    gdq: Vec<f32>,
    gdk: Vec<f32>,
    gs: Vec<f32>,
    gspre: Vec<f32>,
    gqkv: Vec<f32>,
    gconv_pre: Vec<f32>,
    gqkv_raw: Vec<f32>,
    gda: Vec<f32>,
    gdb: Vec<f32>,
    gdx_tmp: Vec<f32>,
}

impl GdnLayerScratch {
    pub fn new(geo: &GdnGeometry) -> Self {
        let n_embd_t = geo.t_len; // placeholder to silence unused warning path
        let _ = n_embd_t;
        let conv_dim = geo.conv_dim();
        let v_dim = geo.v_dim();
        let qk_dim = geo.q_dim();
        let tq_exp = geo.t_len * geo.n_v_heads * geo.key_dim; // expanded q/k tangents
        let heads = geo.n_v_heads * geo.val_dim * geo.key_dim;
        let tc = geo.t_len * conv_dim;
        let tv = geo.t_len * v_dim;
        let tq = geo.t_len * qk_dim;
        Self {
            dqkv_raw: vec![0.0; tc],
            dconv: vec![0.0; tc],
            dqkv: vec![0.0; tc],
            dq: vec![0.0; tq],
            dk: vec![0.0; tq],
            dv: vec![0.0; tv],
            dz: vec![0.0; tv],
            da: vec![0.0; geo.t_len * geo.n_v_heads],
            db: vec![0.0; geo.t_len * geo.n_v_heads],
            dbeta: vec![0.0; geo.t_len * geo.n_v_heads],
            ddecay: vec![0.0; geo.t_len * geo.n_v_heads],
            dsrc: vec![0.0; qk_dim],
            dqn: vec![0.0; tq_exp],
            dkn: vec![0.0; tq_exp],
            ds: vec![0.0; heads],
            spre: vec![0.0; heads],
            dy_pre: vec![0.0; tv],
            dnormed: vec![0.0; tv],
            dout: vec![0.0; tv],
            gout: vec![0.0; tv],
            gdz: vec![0.0; tv],
            gdu: vec![0.0; tv],
            gdy_pre: vec![0.0; tv],
            gdqn: vec![0.0; tq_exp],
            gdkn: vec![0.0; tq_exp],
            gdv: vec![0.0; tv],
            gbeta: vec![0.0; geo.t_len * geo.n_v_heads],
            gdecay: vec![0.0; geo.t_len * geo.n_v_heads],
            gdsrc: vec![0.0; qk_dim],
            gdq: vec![0.0; tq],
            gdk: vec![0.0; tq],
            gs: vec![0.0; heads],
            gspre: vec![0.0; heads],
            gqkv: vec![0.0; tc],
            gconv_pre: vec![0.0; tc],
            gqkv_raw: vec![0.0; tc],
            gda: vec![0.0; geo.t_len * geo.n_v_heads],
            gdb: vec![0.0; geo.t_len * geo.n_v_heads],
            gdx_tmp: vec![0.0; 0], // sized lazily by the maps (needs n_embd)
        }
    }
}

/// Layer `jvp`: tangent `dx` `[T · n_embd]` → `dy` `[T · n_embd]`, composed
/// op by op in the forward's order (see the module doc).
pub fn gdn_layer_jvp(c: &GdnLayerConsts, dx: &[f32], s: &mut GdnLayerScratch, dy_out: &mut [f32]) {
    let geo = c.geo;
    let w = &c.w;
    let cap = c.cap;
    let t_len = geo.t_len;
    let n_embd = w.n_embd;
    let conv_dim = geo.conv_dim();
    let v_dim = geo.v_dim();
    let n_v = geo.n_v_heads;
    let n_k = geo.n_k_heads;
    let key_dim = geo.key_dim;
    let val_dim = geo.val_dim;
    let q_dim = geo.q_dim();
    let k_dim = geo.k_dim();
    let qkv_dim = q_dim + k_dim;
    let block = n_k * key_dim;
    let repeat = geo.repeat();
    debug_assert_eq!(dx.len(), t_len * n_embd, "gdn_layer_jvp: dx shape");
    debug_assert_eq!(dy_out.len(), t_len * n_embd, "gdn_layer_jvp: dy shape");
    if s.gdx_tmp.len() != n_embd {
        s.gdx_tmp = vec![0.0; n_embd];
    }

    // 1. projections (per position)
    for t in 0..t_len {
        let x_t = &dx[t * n_embd..(t + 1) * n_embd];
        crate::types::matmul(
            &mut s.dqkv_raw[t * conv_dim..(t + 1) * conv_dim],
            w.w_qkv,
            x_t,
            conv_dim,
            n_embd,
        );
        crate::types::matmul(
            &mut s.dz[t * v_dim..(t + 1) * v_dim],
            w.w_z,
            x_t,
            v_dim,
            n_embd,
        );
        crate::types::matmul(&mut s.da[t * n_v..(t + 1) * n_v], w.w_a, x_t, n_v, n_embd);
        crate::types::matmul(&mut s.db[t * n_v..(t + 1) * n_v], w.w_b, x_t, n_v, n_embd);
    }
    // 2. doc-level causal conv + silu diagonal
    conv1d_causal_jvp(
        w.conv_w,
        geo.kernel_size,
        &s.dqkv_raw,
        conv_dim,
        &mut s.dconv,
    );
    for i in 0..t_len * conv_dim {
        s.dqkv[i] = cap.conv_silu_grad[i] * s.dconv[i];
    }
    for t in 0..t_len {
        s.dq[t * q_dim..(t + 1) * q_dim]
            .copy_from_slice(&s.dqkv[t * conv_dim..t * conv_dim + q_dim]);
        s.dk[t * k_dim..(t + 1) * k_dim]
            .copy_from_slice(&s.dqkv[t * conv_dim + q_dim..t * conv_dim + qkv_dim]);
        s.dv[t * v_dim..(t + 1) * v_dim]
            .copy_from_slice(&s.dqkv[t * conv_dim + qkv_dim..(t + 1) * conv_dim]);
    }
    // 3. gate tangent coefficients
    for i in 0..t_len * n_v {
        s.dbeta[i] = cap.dbeta_coef[i] * s.db[i];
        s.ddecay[i] = cap.ddecay_coef[i] * s.da[i];
    }
    // 4. l2norm + expand jvp (source heads first, then tile)
    for t in 0..t_len {
        let qc = L2NormConsts {
            y: &cap.qn[t * n_v * key_dim..t * n_v * key_dim + block],
            inv: &cap.qn_inv[t * n_k..(t + 1) * n_k],
        };
        s.dsrc.copy_from_slice(&s.dq[t * q_dim..(t + 1) * q_dim]);
        l2_norm_jvp(&qc, key_dim, &mut s.dsrc);
        expand_tiled_jvp(
            &s.dsrc,
            n_k,
            key_dim,
            repeat,
            &mut s.dqn[t * n_v * key_dim..(t + 1) * n_v * key_dim],
        );
        let kc = L2NormConsts {
            y: &cap.kn[t * n_v * key_dim..t * n_v * key_dim + block],
            inv: &cap.kn_inv[t * n_k..(t + 1) * n_k],
        };
        s.dsrc.copy_from_slice(&s.dk[t * k_dim..(t + 1) * k_dim]);
        l2_norm_jvp(&kc, key_dim, &mut s.dsrc);
        expand_tiled_jvp(
            &s.dsrc,
            n_k,
            key_dim,
            repeat,
            &mut s.dkn[t * n_v * key_dim..(t + 1) * n_v * key_dim],
        );
    }
    // 5. recurrence scan
    let rc = GdnRecurrenceConsts {
        t_len,
        n_v_heads: n_v,
        key_dim,
        val_dim,
        scale: geo.scale(),
        q: &cap.qn,
        k: &cap.kn,
        v: &cap.v,
        beta: &cap.beta,
        decay: &cap.decay,
        states: &cap.states,
    };
    gdn_recurrence_jvp(
        &rc,
        &s.dqn,
        &s.dkn,
        &s.dv,
        &s.dbeta,
        &s.ddecay,
        &mut s.ds,
        &mut s.spre,
        &mut s.dy_pre,
    );
    // 6. ssm_norm jvp (T2b's true-derivative maps, per position)
    for t in 0..t_len {
        let qkc = QkNormConsts {
            x: &cap.y_pre[t * v_dim..(t + 1) * v_dim],
            r: &cap.norm_r[t * n_v..(t + 1) * n_v],
        };
        qk_norm_true_jvp(
            &qkc,
            w.linear_norm,
            val_dim,
            &s.dy_pre[t * v_dim..(t + 1) * v_dim],
            &mut s.dnormed[t * v_dim..(t + 1) * v_dim],
        );
    }
    // 7. silu(z) gate half rule + 8. out projection
    for t in 0..t_len {
        let mc = MlpConsts {
            s: &cap.gate_s[t * v_dim..(t + 1) * v_dim],
            u: &cap.normed[t * v_dim..(t + 1) * v_dim],
            sp: &cap.gate_sp[t * v_dim..(t + 1) * v_dim],
        };
        half_rule_jvp(
            &mc,
            &s.dz[t * v_dim..(t + 1) * v_dim],
            &s.dnormed[t * v_dim..(t + 1) * v_dim],
            &mut s.dout[t * v_dim..(t + 1) * v_dim],
        );
        crate::types::matmul(
            &mut dy_out[t * n_embd..(t + 1) * n_embd],
            w.w_out,
            &s.dout[t * v_dim..(t + 1) * v_dim],
            n_embd,
            v_dim,
        );
    }
}

/// Layer `vjp`: cotangent `gy` `[T · n_embd]` → `gdx` `[T · n_embd]`
/// (zeroed here), the exact adjoint of [`gdn_layer_jvp`].
pub fn gdn_layer_vjp(c: &GdnLayerConsts, gy: &[f32], s: &mut GdnLayerScratch, gdx: &mut [f32]) {
    let geo = c.geo;
    let w = &c.w;
    let cap = c.cap;
    let t_len = geo.t_len;
    let n_embd = w.n_embd;
    let conv_dim = geo.conv_dim();
    let v_dim = geo.v_dim();
    let n_v = geo.n_v_heads;
    let n_k = geo.n_k_heads;
    let key_dim = geo.key_dim;
    let val_dim = geo.val_dim;
    let q_dim = geo.q_dim();
    let k_dim = geo.k_dim();
    let qkv_dim = q_dim + k_dim;
    let block = n_k * key_dim;
    let repeat = geo.repeat();
    debug_assert_eq!(gy.len(), t_len * n_embd, "gdn_layer_vjp: gy shape");
    debug_assert_eq!(gdx.len(), t_len * n_embd, "gdn_layer_vjp: gdx shape");
    if s.gdx_tmp.len() != n_embd {
        s.gdx_tmp = vec![0.0; n_embd];
    }
    for v in gdx.iter_mut() {
        *v = 0.0;
    }

    // 8. out projection adjoint
    for t in 0..t_len {
        matvec_t(
            &mut s.gout[t * v_dim..(t + 1) * v_dim],
            w.w_out,
            &gy[t * n_embd..(t + 1) * n_embd],
            n_embd,
            v_dim,
        );
    }
    // 7. gate half-rule adjoint (elementwise core)
    for t in 0..t_len {
        let mc = MlpConsts {
            s: &cap.gate_s[t * v_dim..(t + 1) * v_dim],
            u: &cap.normed[t * v_dim..(t + 1) * v_dim],
            sp: &cap.gate_sp[t * v_dim..(t + 1) * v_dim],
        };
        half_rule_vjp(
            &mc,
            &s.gout[t * v_dim..(t + 1) * v_dim],
            &mut s.gdz[t * v_dim..(t + 1) * v_dim],
            &mut s.gdu[t * v_dim..(t + 1) * v_dim],
        );
    }
    // 6. ssm_norm vjp
    for t in 0..t_len {
        let qkc = QkNormConsts {
            x: &cap.y_pre[t * v_dim..(t + 1) * v_dim],
            r: &cap.norm_r[t * n_v..(t + 1) * n_v],
        };
        qk_norm_true_vjp(
            &qkc,
            w.linear_norm,
            val_dim,
            &s.gdu[t * v_dim..(t + 1) * v_dim],
            &mut s.gdy_pre[t * v_dim..(t + 1) * v_dim],
        );
    }
    // 5. recurrence reverse scan
    let rc = GdnRecurrenceConsts {
        t_len,
        n_v_heads: n_v,
        key_dim,
        val_dim,
        scale: geo.scale(),
        q: &cap.qn,
        k: &cap.kn,
        v: &cap.v,
        beta: &cap.beta,
        decay: &cap.decay,
        states: &cap.states,
    };
    gdn_recurrence_vjp(
        &rc,
        &cap.y_pre,
        &s.gdy_pre,
        &mut s.gdqn,
        &mut s.gdkn,
        &mut s.gdv,
        &mut s.gbeta,
        &mut s.gdecay,
        &mut s.gs,
        &mut s.gspre,
    );
    // 4. expand adjoint (fold repeats) + l2norm vjp
    for t in 0..t_len {
        expand_tiled_vjp(
            &s.gdqn[t * n_v * key_dim..(t + 1) * n_v * key_dim],
            n_k,
            key_dim,
            repeat,
            &mut s.gdsrc,
        );
        let qc = L2NormConsts {
            y: &cap.qn[t * n_v * key_dim..t * n_v * key_dim + block],
            inv: &cap.qn_inv[t * n_k..(t + 1) * n_k],
        };
        s.gdq[t * q_dim..(t + 1) * q_dim].copy_from_slice(&s.gdsrc);
        l2_norm_vjp(&qc, key_dim, &mut s.gdq[t * q_dim..(t + 1) * q_dim]);
        expand_tiled_vjp(
            &s.gdkn[t * n_v * key_dim..(t + 1) * n_v * key_dim],
            n_k,
            key_dim,
            repeat,
            &mut s.gdsrc,
        );
        let kc = L2NormConsts {
            y: &cap.kn[t * n_v * key_dim..t * n_v * key_dim + block],
            inv: &cap.kn_inv[t * n_k..(t + 1) * n_k],
        };
        s.gdk[t * k_dim..(t + 1) * k_dim].copy_from_slice(&s.gdsrc);
        l2_norm_vjp(&kc, key_dim, &mut s.gdk[t * k_dim..(t + 1) * k_dim]);
    }
    // 3. gate coefficient adjoints
    for i in 0..t_len * n_v {
        s.gdb[i] = cap.dbeta_coef[i] * s.gbeta[i];
        s.gda[i] = cap.ddecay_coef[i] * s.gdecay[i];
    }
    // 2. conv adjoint (silu diagonal then the transposed conv)
    for t in 0..t_len {
        s.gqkv[t * conv_dim..t * conv_dim + q_dim]
            .copy_from_slice(&s.gdq[t * q_dim..(t + 1) * q_dim]);
        s.gqkv[t * conv_dim + q_dim..t * conv_dim + qkv_dim]
            .copy_from_slice(&s.gdk[t * k_dim..(t + 1) * k_dim]);
        s.gqkv[t * conv_dim + qkv_dim..(t + 1) * conv_dim]
            .copy_from_slice(&s.gdv[t * v_dim..(t + 1) * v_dim]);
    }
    for i in 0..t_len * conv_dim {
        s.gconv_pre[i] = cap.conv_silu_grad[i] * s.gqkv[i];
    }
    conv1d_causal_vjp(
        w.conv_w,
        geo.kernel_size,
        &s.gconv_pre,
        conv_dim,
        &mut s.gqkv_raw,
    );
    // 1. projection adjoints — accumulate into gdx (out-proj of z/a/b/qkv)
    for t in 0..t_len {
        let gx_t = &mut gdx[t * n_embd..(t + 1) * n_embd];
        matvec_t(
            &mut s.gdx_tmp,
            w.w_z,
            &s.gdz[t * v_dim..(t + 1) * v_dim],
            v_dim,
            n_embd,
        );
        for (i, g_slot) in gx_t.iter_mut().enumerate() {
            *g_slot += s.gdx_tmp[i];
        }
        matvec_t(
            &mut s.gdx_tmp,
            w.w_a,
            &s.gda[t * n_v..(t + 1) * n_v],
            n_v,
            n_embd,
        );
        for (i, g_slot) in gx_t.iter_mut().enumerate() {
            *g_slot += s.gdx_tmp[i];
        }
        matvec_t(
            &mut s.gdx_tmp,
            w.w_b,
            &s.gdb[t * n_v..(t + 1) * n_v],
            n_v,
            n_embd,
        );
        for (i, g_slot) in gx_t.iter_mut().enumerate() {
            *g_slot += s.gdx_tmp[i];
        }
        matvec_t(
            &mut s.gdx_tmp,
            w.w_qkv,
            &s.gqkv_raw[t * conv_dim..(t + 1) * conv_dim],
            conv_dim,
            n_embd,
        );
        for (i, g_slot) in gx_t.iter_mut().enumerate() {
            *g_slot += s.gdx_tmp[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Tests — adjointness + finite-difference cross-checks
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic LCG — no unseeded-global draws (the global_rng_gate law).
    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f64 / 16_777_216.0 - 1.0) as f32
        }
        fn vec(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| self.next_f32()).collect()
        }
    }

    fn dot(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| x as f64 * y as f64)
            .sum::<f64>()
    }

    fn close(a: f64, b: f64, tol: f64, what: &str) {
        let scale = a.abs().max(b.abs()).max(1.0);
        assert!(
            (a - b).abs() / scale < tol,
            "{what}: {a:.6} vs {b:.6} rel {:.2e}",
            (a - b).abs() / scale
        );
    }

    #[test]
    fn l2_norm_matches_finite_difference() {
        let head_dim = 8usize;
        let heads = 3usize;
        let mut rng = Lcg(0xA11CE);
        let x = rng.vec(heads * head_dim);
        let mut y = vec![0.0f32; x.len()];
        let mut inv = vec![0.0f32; heads];
        l2_norm_capture(&x, head_dim, &mut y, &mut inv);
        let c = L2NormConsts { y: &y, inv: &inv };
        // jvp vs FD (unit tangent per probe)
        let h = 1e-3f32;
        for probe in [2usize, heads * head_dim / 2] {
            let mut dx = vec![0.0f32; x.len()];
            dx[probe] = 1.0;
            l2_norm_jvp(&c, head_dim, &mut dx);
            let y_of = |xx: &[f32]| -> f64 {
                let mut y2 = vec![0.0f32; xx.len()];
                let mut inv2 = vec![0.0f32; heads];
                l2_norm_capture(xx, head_dim, &mut y2, &mut inv2);
                y2[probe] as f64
            };
            let mut x_hi = x.clone();
            x_hi[probe] += h;
            let mut x_lo = x.clone();
            x_lo[probe] -= h;
            let fd = (y_of(&x_hi) - y_of(&x_lo)) / (2.0 * h as f64);
            close(dx[probe] as f64, fd, 2e-3, "l2_norm jvp fd");
        }
        // vjp vs FD of L = ⟨y, w⟩
        let w = rng.vec(x.len());
        let mut g = w.clone();
        l2_norm_vjp(&c, head_dim, &mut g);
        let loss = |xx: &[f32]| -> f64 {
            let mut y2 = vec![0.0f32; xx.len()];
            let mut inv2 = vec![0.0f32; heads];
            l2_norm_capture(xx, head_dim, &mut y2, &mut inv2);
            dot(&y2, &w)
        };
        for probe in [2usize, heads * head_dim / 2] {
            let hh = 1e-3f32;
            let mut x_hi = x.clone();
            x_hi[probe] += hh;
            let mut x_lo = x.clone();
            x_lo[probe] -= hh;
            let fd = (loss(&x_hi) - loss(&x_lo)) / (2.0 * hh as f64);
            close(g[probe] as f64, fd, 2e-3, "l2_norm vjp fd");
        }
        // adjointness (symmetric map: jvp ≡ vjp on different vectors)
        let mut a1 = rng.vec(x.len());
        let mut a2 = rng.vec(x.len());
        let j1 = a1.clone();
        let j2 = a2.clone();
        l2_norm_jvp(&c, head_dim, &mut a1);
        l2_norm_vjp(&c, head_dim, &mut a2);
        close(dot(&a1, &j2), dot(&j1, &a2), 1e-4, "l2_norm adjoint");
    }

    #[test]
    fn expand_tiled_is_adjoint() {
        let n_src = 2usize;
        let head_dim = 4usize;
        let repeat = 3usize;
        let mut rng = Lcg(0xB0B);
        let dsrc = rng.vec(n_src * head_dim);
        let gdst = rng.vec(n_src * head_dim * repeat);
        let mut ddst = vec![0.0f32; gdst.len()];
        expand_tiled_jvp(&dsrc, n_src, head_dim, repeat, &mut ddst);
        // identity: the jvp copies — slot check
        for r in 0..repeat {
            for i in 0..n_src * head_dim {
                assert!((ddst[r * n_src * head_dim + i] - dsrc[i]).abs() < 1e-6);
            }
        }
        let mut gsrc = vec![0.0f32; dsrc.len()];
        expand_tiled_vjp(&gdst, n_src, head_dim, repeat, &mut gsrc);
        close(dot(&ddst, &gdst), dot(&dsrc, &gsrc), 1e-5, "expand adjoint");
    }

    #[test]
    fn gdn_gates_match_finite_difference() {
        let n = 4usize;
        let mut rng = Lcg(0xC0FFEE);
        let a_raw = rng.vec(n);
        let b_raw = rng.vec(n);
        let a_log: Vec<f32> = (0..n).map(|_| -rng.next_f32().abs() - 0.5).collect();
        let dt_bias = rng.vec(n);
        let mut beta = vec![0.0f32; n];
        let mut decay = vec![0.0f32; n];
        let mut spb = vec![0.0f32; n];
        let mut ddec = vec![0.0f32; n];
        gdn_gates_capture(
            &a_raw, &b_raw, &a_log, &dt_bias, &mut beta, &mut decay, &mut spb, &mut ddec,
        );
        let h = 1e-3f32;
        for probe in 0..n {
            // beta: dβ/db vs spb
            let mut b_hi = b_raw.clone();
            b_hi[probe] += h;
            let mut b_lo = b_raw.clone();
            b_lo[probe] -= h;
            let s_of = |bb: &[f32]| sigmoid(bb[probe]) as f64;
            let fd_b = (s_of(&b_hi) - s_of(&b_lo)) / (2.0 * h as f64);
            close(spb[probe] as f64, fd_b, 2e-3, "gates beta fd");
            // decay: ddecay/da vs coef
            let mut a_hi = a_raw.clone();
            a_hi[probe] += h;
            let mut a_lo = a_raw.clone();
            a_lo[probe] -= h;
            let dec_of =
                |aa: &[f32]| (a_log[probe] * softplus(aa[probe] + dt_bias[probe])).exp() as f64;
            let fd_d = (dec_of(&a_hi) - dec_of(&a_lo)) / (2.0 * h as f64);
            close(ddec[probe] as f64, fd_d, 2e-3, "gates decay fd");
        }
    }

    /// Recurrence: adjointness + FD on the scan core (small asymmetric
    /// geometry, val ≠ key to pin the layouts).
    #[test]
    fn gdn_recurrence_is_adjoint_and_matches_fd() {
        let (t_len, n_v, key, val) = (5usize, 2usize, 4usize, 3usize);
        let per_head = key * val;
        let mut rng = Lcg(0xD00D);
        let q = rng.vec(t_len * n_v * key);
        let k = rng.vec(t_len * n_v * key);
        let v = rng.vec(t_len * n_v * val);
        let beta: Vec<f32> = (0..t_len * n_v)
            .map(|_| rng.next_f32() * 0.5 + 0.25)
            .collect();
        let decay: Vec<f32> = (0..t_len * n_v)
            .map(|_| rng.next_f32() * 0.5 + 0.25)
            .collect();
        let mut states = vec![0.0f32; t_len * n_v * per_head];
        let mut y = vec![0.0f32; t_len * n_v * val];
        gdn_recurrence_capture(
            &q,
            &k,
            &v,
            &beta,
            &decay,
            t_len,
            n_v,
            key,
            val,
            &mut states,
            &mut y,
        );
        let c = GdnRecurrenceConsts {
            t_len,
            n_v_heads: n_v,
            key_dim: key,
            val_dim: val,
            scale: 1.0 / (key as f32).sqrt(),
            q: &q,
            k: &k,
            v: &v,
            beta: &beta,
            decay: &decay,
            states: &states,
        };
        let dq = rng.vec(q.len());
        let dk = rng.vec(k.len());
        let dv = rng.vec(v.len());
        let dbeta = rng.vec(beta.len());
        let ddecay = rng.vec(decay.len());
        let gy = rng.vec(y.len());
        let mut ds = vec![0.0f32; n_v * per_head];
        let mut spre = vec![0.0f32; n_v * per_head];
        let mut dy = vec![0.0f32; y.len()];
        gdn_recurrence_jvp(
            &c, &dq, &dk, &dv, &dbeta, &ddecay, &mut ds, &mut spre, &mut dy,
        );
        let mut gq = vec![0.0f32; q.len()];
        let mut gk = vec![0.0f32; k.len()];
        let mut gv = vec![0.0f32; v.len()];
        let mut gbeta = vec![0.0f32; beta.len()];
        let mut gdecay = vec![0.0f32; decay.len()];
        let mut gs = vec![0.0f32; n_v * per_head];
        let mut gspre = vec![0.0f32; n_v * per_head];
        gdn_recurrence_vjp(
            &c,
            &y,
            &gy,
            &mut gq,
            &mut gk,
            &mut gv,
            &mut gbeta,
            &mut gdecay,
            &mut gs,
            &mut gspre,
        );
        let lhs = dot(&dy, &gy);
        let rhs = dot(&dq, &gq)
            + dot(&dk, &gk)
            + dot(&dv, &gv)
            + dot(&dbeta, &gbeta)
            + dot(&ddecay, &gdecay);
        close(lhs, rhs, 1e-4, "recurrence adjoint");

        // FD: y[t1, h1, r1] w.r.t. k[t0, h1, c1] (t0 < t1 — the state path),
        // and w.r.t. q[t1, h1, c1] (same-step read), v/beta/decay at t1.
        // The FD compares against the vjp with gy = unit at the probe output.
        let (t1, h1, r1) = (t_len - 1, n_v - 1, val - 1);
        let (t0, c1) = (t_len / 2, key / 2);
        let h = 1e-3f32;
        let mut gy_unit = vec![0.0f32; y.len()];
        gy_unit[(t1 * n_v + h1) * val + r1] = 1.0;
        let mut gq2 = vec![0.0f32; q.len()];
        let mut gk2 = vec![0.0f32; k.len()];
        let mut gv2 = vec![0.0f32; v.len()];
        let mut gb2 = vec![0.0f32; beta.len()];
        let mut gd2 = vec![0.0f32; decay.len()];
        let mut gs2 = vec![0.0f32; n_v * per_head];
        let mut gsp2 = vec![0.0f32; n_v * per_head];
        gdn_recurrence_vjp(
            &c, &y, &gy_unit, &mut gq2, &mut gk2, &mut gv2, &mut gb2, &mut gd2, &mut gs2, &mut gsp2,
        );
        let y_of = |name: &str, buf: &[f32]| -> f64 {
            let (qq, kk, vv, bb, dd) = match name {
                "q" => (buf, &k[..], &v[..], &beta[..], &decay[..]),
                "k" => (&q[..], buf, &v[..], &beta[..], &decay[..]),
                "v" => (&q[..], &k[..], buf, &beta[..], &decay[..]),
                "b" => (&q[..], &k[..], &v[..], buf, &decay[..]),
                _ => (&q[..], &k[..], &v[..], &beta[..], buf),
            };
            let mut st2 = vec![0.0f32; states.len()];
            let mut y2 = vec![0.0f32; y.len()];
            gdn_recurrence_capture(qq, kk, vv, bb, dd, t_len, n_v, key, val, &mut st2, &mut y2);
            y2[(t1 * n_v + h1) * val + r1] as f64
        };
        for (name, buf, idx) in [
            ("k", &k, ((t0 * n_v + h1) * key) + c1),
            ("q", &q, ((t1 * n_v + h1) * key) + c1),
            ("v", &v, ((t1 * n_v + h1) * val) + r1),
            ("b", &beta, t1 * n_v + h1),
            ("d", &decay, t1 * n_v + h1),
        ] {
            let mut b_hi = buf.clone();
            b_hi[idx] += h;
            let mut b_lo = buf.clone();
            b_lo[idx] -= h;
            let fd = (y_of(name, &b_hi) - y_of(name, &b_lo)) / (2.0 * h as f64);
            let grad2 = match name {
                "k" => gk2[idx],
                "q" => gq2[idx],
                "v" => gv2[idx],
                "b" => gb2[idx],
                _ => gd2[idx],
            };
            close(grad2 as f64, fd, 3e-3, &format!("recurrence vjp fd {name}"));
        }
    }

    /// The composed layer: adjointness + FD at a small asymmetric geometry
    /// (n_k=1, n_v=2 so repeat=2; key=3, val=4; kernel=3).
    #[test]
    fn gdn_layer_is_adjoint_and_matches_fd() {
        let geo = GdnGeometry {
            t_len: 4,
            n_k_heads: 1,
            n_v_heads: 2,
            key_dim: 3,
            val_dim: 4,
            kernel_size: 3,
        };
        let n_embd = 8usize;
        let conv_dim = geo.conv_dim();
        let v_dim = geo.v_dim();
        let n_v = geo.n_v_heads;
        let mut rng = Lcg(0xE111);
        let w = GdnLayerWeights {
            w_qkv: rng.vec(conv_dim * n_embd).leak(),
            w_z: rng.vec(v_dim * n_embd).leak(),
            w_a: rng.vec(n_v * n_embd).leak(),
            w_b: rng.vec(n_v * n_embd).leak(),
            w_out: rng.vec(n_embd * v_dim).leak(),
            conv_w: rng.vec(conv_dim * geo.kernel_size).leak(),
            a_log: (0..n_v)
                .map(|_| -rng.next_f32().abs() - 0.5)
                .collect::<Vec<f32>>()
                .leak(),
            dt_bias: rng.vec(n_v).leak(),
            linear_norm: rng.vec(geo.val_dim).leak(),
            n_embd,
            rms_eps: 1e-6,
        };
        let x = rng.vec(geo.t_len * n_embd);
        let mut cap = GdnLayerCapture::default();
        gdn_layer_forward(&x, &w, &geo, &mut cap);
        let c = GdnLayerConsts { w, geo, cap: &cap };
        let mut s = GdnLayerScratch::new(&geo);
        let dx = rng.vec(x.len());
        let gy = rng.vec(x.len());
        let mut dy = vec![0.0f32; x.len()];
        let mut gdx = vec![0.0f32; x.len()];
        gdn_layer_jvp(&c, &dx, &mut s, &mut dy);
        gdn_layer_vjp(&c, &gy, &mut s, &mut gdx);
        close(dot(&dy, &gy), dot(&dx, &gdx), 1e-4, "gdn layer adjoint");

        // FD: ∂y[slot]/∂x[probe] vs the vjp with gy = unit at the slot
        // (the layer is nonlinear in x — a single-coordinate FD pairs with
        // the unit-cotangent vjp entry, never a random tangent; the T2b
        // lesson).
        let probe = x.len() / 2;
        let slot = x.len() - 1;
        let hh = 1e-3f32;
        let mut x_hi = x.clone();
        x_hi[probe] += hh;
        let mut x_lo = x.clone();
        x_lo[probe] -= hh;
        let y_of = |xx: &[f32]| -> f64 {
            let mut capn = GdnLayerCapture::default();
            gdn_layer_forward(xx, &w, &geo, &mut capn);
            capn.y[slot] as f64
        };
        let fd = (y_of(&x_hi) - y_of(&x_lo)) / (2.0 * hh as f64);
        let mut gy_unit = vec![0.0f32; x.len()];
        gy_unit[slot] = 1.0;
        let mut gdx2 = vec![0.0f32; x.len()];
        gdn_layer_vjp(&c, &gy_unit, &mut s, &mut gdx2);
        // The layer's ONLY ruled op is the silu(z)·normed output gate (the
        // RelP half rule: its local Jacobian is ½·the true product Jacobian,
        // on BOTH branches). Every other op in the chain keeps the true
        // derivative, and out_proj (linear) lets the scalar ½ commute — so
        // the layer's input gradient is EXACTLY half the FD truth. Assert
        // that relation, not FD equality (the T2a precedent: rules get
        // adjointness; true-derivative ops get FD).
        close(
            gdx2[probe] as f64,
            0.5 * fd,
            3e-3,
            "gdn layer vjp fd (half-rule factor)",
        );
    }
}
