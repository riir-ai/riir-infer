//! Attention-layer lens maps — T2b of katgpt-rs Issue 999 (Research 666).
//!
//! The qwen3.5 full-attention layer ([`crate::deltanet::forward`]
//! `forward_attention_layer`) linearized op by op:
//!
//! 1. **Gated Q projection** — `attn_wq` emits `2·q_dim` outputs laid out
//!    `[q_h(hd), gate_h(hd)]` per head; the de-interleave is a permutation,
//!    so its adjoint ([`qg_interleave`]) is the inverse permutation.
//! 2. **QK-norm** — per-head RMSNorm under the TRUE derivative (the issue's
//!    T2 rule split: attention-path norms keep the exact Jacobian; only the
//!    decoder RMSNorms take the LN rule). Rank-1-plus-diagonal, see
//!    [`QkNormConsts`].
//! 3. **RoPE** — already exact ([`crate::lens::ops::rope_jvp`]/
//!    [`crate::lens::ops::rope_vjp`], partial-`rotary_dim` dispatch tested
//!    there); the layer composition calls it between QK-norm and attention.
//! 4. **Softmax attention** — true derivative through the softmax
//!    ([`attention_jvp`]/[`attention_vjp`]), causal, GQA with the forward's
//!    strided group map `kv_group = h·n_kv/n_head` and
//!    `scale = 1/sqrt(head_dim)` (mirroring `attention_heads_parallel_bias`).
//! 5. **Output gate** — `σ(gate)·attn_out`: the two-factor half rule already
//!    shipped ([`crate::lens::ops::GateConsts`] + `gate_half_rule_*`); the
//!    layer composition captures with the gate pre-activation. ⚠ The forward
//!    applies `crate::simd::fast_sigmoid` — capture may use it for parity or
//!    the exact [`crate::lens::ops::sigmoid`]; the drift is fit-side noise,
//!    disclosed at the composition site (T2d).
//! 6. **o_proj** — plain linear ([`crate::lens::ops::matvec_t`] serves the
//!    adjoint).
//!
//! Provenance: `stdlogits/neuron-lens` @ `849bfe16` (MIT), `layers.py`
//! grouped-head softmax attention + q/k norm maps, ported to this crate's
//! row-major layout. Zero allocation: every buffer is caller-provided; the
//! fit is offline (once per checkpoint), never a serving path.
//!
//! Bit-parity note: the capture/forward here is the DOC-level reference
//! (all `t` attending `j ≤ t`); the T2d linearize pins it against
//! `forward_qwen_deltanet`'s decode-step seams the same way
//! [`crate::lens::ops::conv1d_causal_jvp`] pins against the GDN loop (T2c).

// ---------------------------------------------------------------------------
// Softmax attention — true derivative, causal, GQA
// ---------------------------------------------------------------------------

/// Captured attention constants for one doc (the `linearize` pass computes
/// these once from the post-RoPE q/k/v activations).
#[derive(Clone, Copy, Debug)]
pub struct AttentionConsts<'a> {
    /// Sequence length `T` (every position attends to `j ≤ t`).
    pub t_len: usize,
    /// Query heads.
    pub n_head: usize,
    /// Key/value heads (GQA; `n_kv ≤ n_head`).
    pub n_kv: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Logit scale `1/sqrt(head_dim)` (the forward's).
    pub scale: f32,
    /// Softmax weights, layout `[t_len · n_head · t_len]` — row `t`, head
    /// `h`, key `j` at `attn[(t·n_head + h)·t_len + j]`; only `j ≤ t` is
    /// meaningful. One value per query head (GQA shares rows across the
    /// group's heads, never merged — the adjoint is per-head).
    pub attn: &'a [f32],
}

impl<'a> AttentionConsts<'a> {
    /// Query-vector length (`n_head · head_dim`).
    pub fn q_dim(&self) -> usize {
        self.n_head * self.head_dim
    }

    /// Key/value-vector length (`n_kv · head_dim`).
    pub fn kv_dim(&self) -> usize {
        self.n_kv * self.head_dim
    }

    /// The forward's GQA group map: `kv_group = h · n_kv / n_head`
    /// (`attention_heads_parallel_bias`'s strided grouping — heads in
    /// contiguous blocks of `n_head/n_kv` share one kv head).
    #[inline(always)]
    pub fn kv_group(&self, h: usize) -> usize {
        h * self.n_kv / self.n_head
    }
}

/// The forward's logit scale.
#[inline(always)]
pub fn attention_scale(head_dim: usize) -> f32 {
    1.0 / (head_dim as f32).sqrt()
}

/// Doc-level causal GQA attention forward — computes the softmax weights
/// into `attn` and the attended values into `y`, both over the whole doc.
///
/// Layouts (row-major, per position): `q` `[T · q_dim]`, `k`/`v`
/// `[T · kv_dim]`, `y` `[T · q_dim]`, `attn` `[T · n_head · T]`. This is the
/// FD reference the tests differentiate; T2d pins it against the serving
/// forward's decode-step seams.
#[allow(clippy::too_many_arguments)]
pub fn attention_capture(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_head: usize,
    n_kv: usize,
    head_dim: usize,
    attn: &mut [f32],
    y: &mut [f32],
) {
    let t_len = q.len() / (n_head * head_dim);
    debug_assert_eq!(
        k.len(),
        t_len * n_kv * head_dim,
        "attention_capture: k shape"
    );
    debug_assert_eq!(v.len(), k.len(), "attention_capture: v shape");
    debug_assert_eq!(y.len(), q.len(), "attention_capture: y shape");
    debug_assert_eq!(
        attn.len(),
        t_len * n_head * t_len,
        "attention_capture: attn shape"
    );
    let kv_dim = n_kv * head_dim;
    let q_dim = n_head * head_dim;
    let scale = attention_scale(head_dim);

    for t in 0..t_len {
        let t_n = t + 1;
        for h in 0..n_head {
            let kv = h * n_kv / n_head;
            let q_off = t * q_dim + h * head_dim;
            let q_h = &q[q_off..q_off + head_dim];
            let a_row = &mut attn[(t * n_head + h) * t_len..(t * n_head + h) * t_len + t_n];
            let mut max_logit = f32::NEG_INFINITY;
            for (j, a_slot) in a_row.iter_mut().enumerate().take(t_n) {
                let k_off = j * kv_dim + kv * head_dim;
                let mut logit = 0.0f32;
                for d in 0..head_dim {
                    logit += q_h[d] * k[k_off + d];
                }
                logit *= scale;
                *a_slot = logit;
                if logit > max_logit {
                    max_logit = logit;
                }
            }
            let mut sum = 0.0f32;
            for a_slot in a_row.iter_mut().take(t_n) {
                let e = (*a_slot - max_logit).exp();
                *a_slot = e;
                sum += e;
            }
            let inv = 1.0 / sum;
            for a_slot in a_row.iter_mut().take(t_n) {
                *a_slot *= inv;
            }
            // y[t, h] = Σ_j a[t,h,j] · v[j, kv]
            let y_off = t * q_dim + h * head_dim;
            for d in 0..head_dim {
                y[y_off + d] = 0.0;
            }
            for (j, &a) in a_row.iter().enumerate().take(t_n) {
                if a == 0.0 {
                    continue;
                }
                let v_off = j * kv_dim + kv * head_dim;
                for d in 0..head_dim {
                    y[y_off + d] += a * v[v_off + d];
                }
            }
        }
    }
}

/// Softmax-attention `jvp` — the TRUE derivative (the issue's T2 rule: the
/// attention core keeps its exact Jacobian).
///
/// Tangents `(dq, dk, dv)` (same layouts as the activations) → `dout`
/// `[T · q_dim]`. Per query row: `ds = (⟨dq,k⟩ + ⟨q,dk⟩)·scale`,
/// `da = a ⊙ (ds − ⟨a ⊙ ds⟩_row)`, `dout = da·v + a·dv`.
/// `ds_scratch` is caller-provided `[t_len]` (reused per `(t, h)` row).
#[allow(clippy::too_many_arguments)]
pub fn attention_jvp(
    c: &AttentionConsts,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    dq: &[f32],
    dk: &[f32],
    dv: &[f32],
    ds_scratch: &mut [f32],
    dout: &mut [f32],
) {
    let t_len = c.t_len;
    let head_dim = c.head_dim;
    let kv_dim = c.kv_dim();
    let q_dim = c.q_dim();
    let scale = c.scale;
    debug_assert_eq!(q.len(), t_len * q_dim, "attention jvp: q shape");
    debug_assert_eq!(k.len(), t_len * kv_dim, "attention jvp: k shape");
    debug_assert_eq!(v.len(), k.len(), "attention jvp: v shape");
    debug_assert_eq!(dq.len(), q.len(), "attention jvp: dq shape");
    debug_assert_eq!(dk.len(), k.len(), "attention jvp: dk shape");
    debug_assert_eq!(dv.len(), v.len(), "attention jvp: dv shape");
    debug_assert_eq!(dout.len(), q.len(), "attention jvp: dout shape");
    debug_assert!(
        ds_scratch.len() >= t_len,
        "attention jvp: ds_scratch < t_len"
    );

    for t in 0..t_len {
        let t_n = t + 1;
        let ds_row = &mut ds_scratch[..t_n];
        for h in 0..c.n_head {
            let kv = c.kv_group(h);
            let q_off = t * q_dim + h * head_dim;
            let q_h = &q[q_off..q_off + head_dim];
            let dq_h = &dq[q_off..q_off + head_dim];
            let a_row = &c.attn[(t * c.n_head + h) * t_len..(t * c.n_head + h) * t_len + t_n];
            let mut row_sum = 0.0f32;
            for j in 0..t_n {
                let k_off = j * kv_dim + kv * head_dim;
                let mut ds = 0.0f32;
                for d in 0..head_dim {
                    ds += dq_h[d] * k[k_off + d] + q_h[d] * dk[k_off + d];
                }
                ds *= scale;
                ds_row[j] = ds;
                row_sum += a_row[j] * ds;
            }
            let dout_off = t * q_dim + h * head_dim;
            for d in 0..head_dim {
                dout[dout_off + d] = 0.0;
            }
            for j in 0..t_n {
                let da = a_row[j] * (ds_row[j] - row_sum);
                let a = a_row[j];
                let v_off = j * kv_dim + kv * head_dim;
                for d in 0..head_dim {
                    dout[dout_off + d] += da * v[v_off + d] + a * dv[v_off + d];
                }
            }
        }
    }
}

/// Softmax-attention `vjp` — the exact adjoint of [`attention_jvp`].
///
/// Cotangent `gout` `[T · q_dim]` → gradients `(gq, gk, gv)` (activation
/// layouts; zeroed here). Per query row `t` (with `y` the captured forward
/// output, needed for the softmax pullback's `⟨g, y⟩` term):
/// `gs = a ⊙ (⟨g, v_j⟩ − ⟨g, y_t⟩)`, `gq += gs·k·scale`,
/// `gk += gs·q·scale`, `gv += a·g`.
#[allow(clippy::too_many_arguments)]
pub fn attention_vjp(
    c: &AttentionConsts,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    y: &[f32],
    gout: &[f32],
    gq: &mut [f32],
    gk: &mut [f32],
    gv: &mut [f32],
) {
    let t_len = c.t_len;
    let head_dim = c.head_dim;
    let kv_dim = c.kv_dim();
    let q_dim = c.q_dim();
    let scale = c.scale;
    debug_assert_eq!(q.len(), t_len * q_dim, "attention vjp: q shape");
    debug_assert_eq!(k.len(), t_len * kv_dim, "attention vjp: k shape");
    debug_assert_eq!(v.len(), k.len(), "attention vjp: v shape");
    debug_assert_eq!(y.len(), q.len(), "attention vjp: y shape");
    debug_assert_eq!(gout.len(), q.len(), "attention vjp: gout shape");
    debug_assert_eq!(gq.len(), q.len(), "attention vjp: gq shape");
    debug_assert_eq!(gk.len(), k.len(), "attention vjp: gk shape");
    debug_assert_eq!(gv.len(), v.len(), "attention vjp: gv shape");
    for x in gq.iter_mut() {
        *x = 0.0;
    }
    for x in gk.iter_mut() {
        *x = 0.0;
    }
    for x in gv.iter_mut() {
        *x = 0.0;
    }

    for t in 0..t_len {
        let t_n = t + 1;
        for h in 0..c.n_head {
            let kv = c.kv_group(h);
            let q_off = t * q_dim + h * head_dim;
            let q_h = &q[q_off..q_off + head_dim];
            let g_h = &gout[q_off..q_off + head_dim];
            let y_h = &y[q_off..q_off + head_dim];
            let mut dot_gy = 0.0f32;
            for d in 0..head_dim {
                dot_gy += g_h[d] * y_h[d];
            }
            let a_row = &c.attn[(t * c.n_head + h) * t_len..(t * c.n_head + h) * t_len + t_n];
            let gq_h = &mut gq[q_off..q_off + head_dim];
            for (j, &a) in a_row.iter().enumerate().take(t_n) {
                let k_off = j * kv_dim + kv * head_dim;
                let mut dot_gv = 0.0f32;
                for d in 0..head_dim {
                    dot_gv += g_h[d] * v[k_off + d];
                }
                let gs = a * (dot_gv - dot_gy);
                if gs == 0.0 && a == 0.0 {
                    continue;
                }
                for d in 0..head_dim {
                    gq_h[d] += gs * k[k_off + d] * scale;
                    gk[k_off + d] += gs * q_h[d] * scale;
                    gv[k_off + d] += a * g_h[d];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// QK-norm — per-head RMSNorm under the true derivative
// ---------------------------------------------------------------------------

/// Captured QK-norm constants for ONE position (pre-norm activation `x` and
/// the per-head `rsqrt` values the forward computed).
///
/// True derivative of `y_i = γ_i · x_i · r` with
/// `r = (mean(x_h²) + eps)^(−1/2)` per head `h`. With `a = γ⊙r` and
/// `u = γ⊙x·r³/hd` the Jacobian is `J = diag(a) − u·xᵀ`, so
///
/// ```text
/// dy_i = γ_i·r·dx_i − γ_i·x_i·r³·⟨x_h, dx_h⟩/hd        (jvp)
/// dx_i = γ_i·r·g_i − x_i·⟨u_h, g_h⟩                     (vjp — Jᵀ: no γ on the rank-1 term)
/// ```
///
/// The capture mirrors [`crate::types::rmsnorm_with_gamma_eps`]'s numerics:
/// f32 sum-of-squares, `eps` cast to f32 once (the f64 param is API compat).
#[derive(Clone, Copy, Debug)]
pub struct QkNormConsts<'a> {
    /// The pre-norm activation `x` (one position, all heads, `[heads·hd]`).
    pub x: &'a [f32],
    /// Per-head `rsqrt` (`[heads]`).
    pub r: &'a [f32],
}

impl<'a> QkNormConsts<'a> {
    /// Capture per-head `rsqrt` + the normalized output `y = γ ⊙ x ⊙ r`.
    ///
    /// `r_out` is `[heads]`, `y_out` `[heads·hd]`. The sum-of-squares is a
    /// sequential f32 fold — the serving kernel's SIMD tree may differ in
    /// the last ulp; T2d captures `r` from the forward's own seam when it
    /// needs bit parity.
    pub fn capture(
        x: &[f32],
        gamma: &[f32],
        head_dim: usize,
        eps: f64,
        r_out: &mut [f32],
        y_out: &mut [f32],
    ) {
        let heads = x.len() / head_dim;
        debug_assert_eq!(r_out.len(), heads, "QkNormConsts::capture: r_out heads");
        debug_assert_eq!(y_out.len(), x.len(), "QkNormConsts::capture: y_out shape");
        debug_assert_eq!(
            gamma.len(),
            head_dim,
            "QkNormConsts::capture: gamma per head"
        );
        let eps_f32 = eps as f32;
        for (h, r_slot) in r_out.iter_mut().enumerate().take(heads) {
            let off = h * head_dim;
            let head = &x[off..off + head_dim];
            let mut sum_sq = 0.0f32;
            for &v in head {
                sum_sq += v * v;
            }
            let r = 1.0 / (sum_sq / head_dim as f32 + eps_f32).sqrt();
            *r_slot = r;
            for (i, &v) in head.iter().enumerate() {
                y_out[off + i] = v * gamma[i] * r;
            }
        }
    }
}

/// QK-norm `jvp` (true derivative) — `dx` → `dy`, one position, per head.
///
/// `gamma` is the norm's weight (one `[hd]` row shared by every head, the
/// forward's layout). The rank-1 term uses the pre-norm `x` and captured
/// `r` from the consts.
pub fn qk_norm_true_jvp(
    c: &QkNormConsts,
    gamma: &[f32],
    head_dim: usize,
    dx: &[f32],
    dy: &mut [f32],
) {
    let heads = c.x.len() / head_dim;
    debug_assert_eq!(c.r.len(), heads, "qk_norm jvp: r heads");
    debug_assert_eq!(gamma.len(), head_dim, "qk_norm jvp: gamma per head");
    debug_assert_eq!(dx.len(), c.x.len(), "qk_norm jvp: dx shape");
    debug_assert_eq!(dy.len(), dx.len(), "qk_norm jvp: dy shape");
    for h in 0..heads {
        let off = h * head_dim;
        let x_h = &c.x[off..off + head_dim];
        let r = c.r[h];
        let r3 = r * r * r;
        let mut dot_xdx = 0.0f32;
        for i in 0..head_dim {
            dot_xdx += x_h[i] * dx[off + i];
        }
        let coef = r3 * dot_xdx / head_dim as f32;
        for i in 0..head_dim {
            dy[off + i] = gamma[i] * (r * dx[off + i] - x_h[i] * coef);
        }
    }
}

/// QK-norm `vjp` — the exact adjoint of [`qk_norm_true_jvp`]. With
/// `u = γ⊙x·r³/hd`: `dx_i = γ_i·r·g_i − x_i·⟨u_h, g_h⟩` — the rank-1 term
/// carries NO γ (it is `x·uᵀ` in `Jᵀ`, not `γ⊙x·uᵀ`).
pub fn qk_norm_true_vjp(
    c: &QkNormConsts,
    gamma: &[f32],
    head_dim: usize,
    dy: &[f32],
    dx: &mut [f32],
) {
    let heads = c.x.len() / head_dim;
    debug_assert_eq!(c.r.len(), heads, "qk_norm vjp: r heads");
    debug_assert_eq!(gamma.len(), head_dim, "qk_norm vjp: gamma per head");
    debug_assert_eq!(dy.len(), c.x.len(), "qk_norm vjp: dy shape");
    debug_assert_eq!(dx.len(), dy.len(), "qk_norm vjp: dx shape");
    for h in 0..heads {
        let off = h * head_dim;
        let x_h = &c.x[off..off + head_dim];
        let r = c.r[h];
        let r3 = r * r * r;
        let mut dot_ug = 0.0f32;
        for i in 0..head_dim {
            dot_ug += gamma[i] * x_h[i] * r3 * dy[off + i];
        }
        dot_ug /= head_dim as f32;
        for i in 0..head_dim {
            // Jᵀ = diag(γ⊙r) − x·uᵀ: the diagonal term wears γ, the rank-1
            // term does not.
            dx[off + i] = gamma[i] * r * dy[off + i] - x_h[i] * dot_ug;
        }
    }
}

// ---------------------------------------------------------------------------
// Gated Q projection — interleave permutation and its inverse
// ---------------------------------------------------------------------------

/// De-interleave the gated Q projection's tangent: `attn_wq` emits
/// `[q_h(hd), gate_h(hd)]` per head (the forward's split at
/// `forward_attention_layer` step 1) — split one tangent into `(dq, dgate)`.
pub fn qg_deinterleave(
    dqg: &[f32],
    n_head: usize,
    head_dim: usize,
    dq: &mut [f32],
    dgate: &mut [f32],
) {
    debug_assert_eq!(
        dqg.len(),
        n_head * 2 * head_dim,
        "qg_deinterleave: dqg shape"
    );
    debug_assert_eq!(dq.len(), n_head * head_dim, "qg_deinterleave: dq shape");
    debug_assert_eq!(dgate.len(), dq.len(), "qg_deinterleave: dgate shape");
    for h in 0..n_head {
        let src = h * 2 * head_dim;
        let dst = h * head_dim;
        dq[dst..dst + head_dim].copy_from_slice(&dqg[src..src + head_dim]);
        dgate[dst..dst + head_dim].copy_from_slice(&dqg[src + head_dim..src + 2 * head_dim]);
    }
}

/// The inverse permutation ([`qg_deinterleave`]'s exact adjoint): fold
/// cotangents `(gq, ggate)` back into the projection's output layout.
pub fn qg_interleave(gq: &[f32], ggate: &[f32], n_head: usize, head_dim: usize, gqg: &mut [f32]) {
    debug_assert_eq!(gqg.len(), n_head * 2 * head_dim, "qg_interleave: gqg shape");
    debug_assert_eq!(gq.len(), n_head * head_dim, "qg_interleave: gq shape");
    debug_assert_eq!(ggate.len(), gq.len(), "qg_interleave: ggate shape");
    for h in 0..n_head {
        let src = h * head_dim;
        let dst = h * 2 * head_dim;
        gqg[dst..dst + head_dim].copy_from_slice(&gq[src..src + head_dim]);
        gqg[dst + head_dim..dst + 2 * head_dim].copy_from_slice(&ggate[src..src + head_dim]);
    }
}

// ---------------------------------------------------------------------------
// Tests — adjointness per map + finite-difference cross-checks
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

    /// Small geometry: T=6, 4 Q heads over 2 KV heads, hd=8.
    fn small_geo() -> (usize, usize, usize, usize) {
        (6, 4, 2, 8)
    }

    fn random_doc<'a>(
        rng: &mut Lcg,
        t_len: usize,
        n_head: usize,
        n_kv: usize,
        head_dim: usize,
        attn_buf: &'a mut [f32],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, AttentionConsts<'a>) {
        let q = rng.vec(t_len * n_head * head_dim);
        let k = rng.vec(t_len * n_kv * head_dim);
        let v = rng.vec(t_len * n_kv * head_dim);
        let mut y = vec![0.0f32; q.len()];
        attention_capture(&q, &k, &v, n_head, n_kv, head_dim, attn_buf, &mut y);
        let consts = AttentionConsts {
            t_len,
            n_head,
            n_kv,
            head_dim,
            scale: attention_scale(head_dim),
            attn: attn_buf,
        };
        (q, k, v, y, consts)
    }

    #[test]
    fn attention_core_is_adjoint() {
        let (t_len, n_head, n_kv, head_dim) = small_geo();
        let mut rng = Lcg(0xA777);
        let mut attn_buf = vec![0.0f32; t_len * n_head * t_len];
        let (q, k, v, _y, c) = random_doc(&mut rng, t_len, n_head, n_kv, head_dim, &mut attn_buf);
        let dq = rng.vec(q.len());
        let dk = rng.vec(k.len());
        let dv = rng.vec(v.len());
        let gout = rng.vec(q.len());
        let mut ds = vec![0.0f32; t_len];
        let mut dout = vec![0.0f32; q.len()];
        let mut gq = vec![0.0f32; q.len()];
        let mut gk = vec![0.0f32; k.len()];
        let mut gv = vec![0.0f32; v.len()];
        attention_jvp(&c, &q, &k, &v, &dq, &dk, &dv, &mut ds, &mut dout);
        attention_vjp(&c, &q, &k, &v, &_y, &gout, &mut gq, &mut gk, &mut gv);
        let lhs = dot(&dout, &gout);
        let rhs = dot(&dq, &gq) + dot(&dk, &gk) + dot(&dv, &gv);
        close(lhs, rhs, 1e-4, "attention adjoint");
    }

    /// jvp against central differences of the capture forward, per input
    /// slot (q, k, v), at both geometries — the small one and qwen3.5-0.8B's
    /// (8Q/2KV, hd=256).
    #[test]
    fn attention_jvp_matches_finite_difference() {
        for &(t_len, n_head, n_kv, head_dim, seed) in &[
            (6usize, 4usize, 2usize, 8usize, 0x51EED),
            (5, 8, 2, 256, 0x51EEF),
        ] {
            let mut rng = Lcg(seed);
            let mut attn_buf = vec![0.0f32; t_len * n_head * t_len];
            let (q, k, v, _y, c) =
                random_doc(&mut rng, t_len, n_head, n_kv, head_dim, &mut attn_buf);
            let q_dim = n_head * head_dim;
            let kv_dim = n_kv * head_dim;
            let h = 1e-3f32;
            // one (t, head, d) probe slot per input side
            let (t0, h0, d0) = (t_len / 2, n_head - 1, head_dim / 2);
            let q_idx = t0 * q_dim + h0 * head_dim + d0;
            let kv_head = h0 * n_kv / n_head;
            let k_idx = (t0 - 1) * kv_dim + kv_head * head_dim + d0;
            let v_idx = t0 * kv_dim + kv_head * head_dim + d0;
            for (name, buf, idx) in [("q", &q, q_idx), ("k", &k, k_idx), ("v", &v, v_idx)] {
                let mut unit = vec![0.0f32; buf.len()];
                unit[idx] = 1.0;
                let mut ds = vec![0.0f32; t_len];
                let mut dout = vec![0.0f32; q.len()];
                let (dqq, dkk, dvv) = match name {
                    "q" => (unit.clone(), vec![0.0; k.len()], vec![0.0; v.len()]),
                    "k" => (vec![0.0; q.len()], unit.clone(), vec![0.0; v.len()]),
                    _ => (vec![0.0; q.len()], vec![0.0; k.len()], unit),
                };
                attention_jvp(&c, &q, &k, &v, &dqq, &dkk, &dvv, &mut ds, &mut dout);
                // FD at the probe's output slot (t0, h0, d0); for k the probe
                // sits at j = t0−1 ≤ t0 so the row still sees it.
                let y_hi = {
                    let mut b = buf.clone();
                    b[idx] += h;
                    let (qq, kk, vv) = match name {
                        "q" => (b, k.clone(), v.clone()),
                        "k" => (q.clone(), b, v.clone()),
                        _ => (q.clone(), k.clone(), b),
                    };
                    let mut attn2 = vec![0.0f32; t_len * n_head * t_len];
                    let mut y2 = vec![0.0f32; q.len()];
                    attention_capture(&qq, &kk, &vv, n_head, n_kv, head_dim, &mut attn2, &mut y2);
                    y2[q_idx]
                };
                let y_lo = {
                    let mut b = buf.clone();
                    b[idx] -= h;
                    let (qq, kk, vv) = match name {
                        "q" => (b, k.clone(), v.clone()),
                        "k" => (q.clone(), b, v.clone()),
                        _ => (q.clone(), k.clone(), b),
                    };
                    let mut attn2 = vec![0.0f32; t_len * n_head * t_len];
                    let mut y2 = vec![0.0f32; q.len()];
                    attention_capture(&qq, &kk, &vv, n_head, n_kv, head_dim, &mut attn2, &mut y2);
                    y2[q_idx]
                };
                let fd = ((y_hi - y_lo) / (2.0 * h)) as f64;
                close(
                    dout[q_idx] as f64,
                    fd,
                    3e-3,
                    &format!("attention jvp {name} fd"),
                );
            }
        }
    }

    /// vjp is the gradient of the scalarization L = ⟨y, w⟩ — central
    /// differences on each input side, both geometries, GQA sharing covered
    /// by the kv-head probe (its gradient must fold the whole group).
    #[test]
    fn attention_vjp_matches_finite_difference() {
        for &(t_len, n_head, n_kv, head_dim, seed) in &[
            (6usize, 4usize, 2usize, 8usize, 0x61EED),
            (5, 8, 2, 256, 0x61EEF),
        ] {
            let mut rng = Lcg(seed);
            let mut attn_buf = vec![0.0f32; t_len * n_head * t_len];
            let (q, k, v, y, c) =
                random_doc(&mut rng, t_len, n_head, n_kv, head_dim, &mut attn_buf);
            let q_dim = n_head * head_dim;
            let kv_dim = n_kv * head_dim;
            let w = rng.vec(q.len());
            let mut gq = vec![0.0f32; q.len()];
            let mut gk = vec![0.0f32; k.len()];
            let mut gv = vec![0.0f32; v.len()];
            attention_vjp(&c, &q, &k, &v, &y, &w, &mut gq, &mut gk, &mut gv);
            let (t0, h0, d0) = (t_len / 2, n_head / 2, head_dim / 2);
            let q_idx = t0 * q_dim + h0 * head_dim + d0;
            let kv_head = h0 * n_kv / n_head;
            let k_idx = (t0 - 1) * kv_dim + kv_head * head_dim + d0;
            let v_idx = t0 * kv_dim + kv_head * head_dim + d0;
            let h = 1e-3f32;
            for (name, buf, idx, grad) in [
                ("q", &q, q_idx, &gq),
                ("k", &k, k_idx, &gk),
                ("v", &v, v_idx, &gv),
            ] {
                let loss = |b: &[f32]| -> f64 {
                    let (qq, kk, vv) = match name {
                        "q" => (b, &k[..], &v[..]),
                        "k" => (&q[..], b, &v[..]),
                        _ => (&q[..], &k[..], b),
                    };
                    let mut attn2 = vec![0.0f32; t_len * n_head * t_len];
                    let mut y2 = vec![0.0f32; q.len()];
                    attention_capture(qq, kk, vv, n_head, n_kv, head_dim, &mut attn2, &mut y2);
                    dot(&y2, &w)
                };
                let mut b_hi = buf.clone();
                b_hi[idx] += h;
                let mut b_lo = buf.clone();
                b_lo[idx] -= h;
                let fd = (loss(&b_hi) - loss(&b_lo)) / (2.0 * h as f64);
                close(
                    grad[idx] as f64,
                    fd,
                    3e-3,
                    &format!("attention vjp {name} fd"),
                );
            }
        }
    }

    #[test]
    fn qk_norm_true_matches_forward_fd() {
        let head_dim = 32usize;
        let heads = 4usize;
        let eps = 1e-6f64;
        let mut rng = Lcg(0x71EED);
        let x = rng.vec(heads * head_dim);
        let gamma = rng.vec(head_dim);
        let mut r = vec![0.0f32; heads];
        let mut y = vec![0.0f32; x.len()];
        QkNormConsts::capture(&x, &gamma, head_dim, eps, &mut r, &mut y);
        // capture's y must equal the serving kernel (per-head rmsnorm)
        let mut y_fwd = x.clone();
        for h in 0..heads {
            crate::types::rmsnorm_with_gamma_eps(
                &mut y_fwd[h * head_dim..(h + 1) * head_dim],
                &gamma,
                eps,
            );
        }
        for i in 0..y.len() {
            assert!(
                (y[i] - y_fwd[i]).abs() < 1e-4,
                "qk_norm capture vs forward: {i}: {} vs {}",
                y[i],
                y_fwd[i]
            );
        }
        // jvp vs FD of the forward: the tangent is the UNIT vector at the
        // probe — a single-coordinate FD measures one column of J, so the
        // tangent must select that column, not a random direction.
        let c = QkNormConsts { x: &x, r: &r };
        let h = 1e-3f32;
        for probe in [3usize, heads * head_dim / 2] {
            let mut dx = vec![0.0f32; x.len()];
            dx[probe] = 1.0;
            let mut dy = vec![0.0f32; x.len()];
            qk_norm_true_jvp(&c, &gamma, head_dim, &dx, &mut dy);
            let mut x_hi = x.clone();
            x_hi[probe] += h;
            let mut x_lo = x.clone();
            x_lo[probe] -= h;
            let y_of = |xx: &[f32]| -> f64 {
                let mut r2 = vec![0.0f32; heads];
                let mut y2 = vec![0.0f32; xx.len()];
                QkNormConsts::capture(xx, &gamma, head_dim, eps, &mut r2, &mut y2);
                y2[probe] as f64
            };
            let fd = (y_of(&x_hi) - y_of(&x_lo)) / (2.0 * h as f64);
            close(dy[probe] as f64, fd, 2e-3, "qk_norm jvp fd");
        }
        // vjp vs FD of L = ⟨y, w⟩
        let w = rng.vec(x.len());
        let mut dx_v = vec![0.0f32; x.len()];
        qk_norm_true_vjp(&c, &gamma, head_dim, &w, &mut dx_v);
        for probe in [3usize, heads * head_dim / 2] {
            let loss = |xx: &[f32]| -> f64 {
                let mut r2 = vec![0.0f32; heads];
                let mut y2 = vec![0.0f32; xx.len()];
                QkNormConsts::capture(xx, &gamma, head_dim, eps, &mut r2, &mut y2);
                dot(&y2, &w)
            };
            let mut x_hi = x.clone();
            x_hi[probe] += h;
            let mut x_lo = x.clone();
            x_lo[probe] -= h;
            let fd = (loss(&x_hi) - loss(&x_lo)) / (2.0 * h as f64);
            close(dx_v[probe] as f64, fd, 2e-3, "qk_norm vjp fd");
        }
    }

    #[test]
    fn qk_norm_is_adjoint() {
        let head_dim = 16usize;
        let heads = 3usize;
        let eps = 1e-6f64;
        let mut rng = Lcg(0x81EED);
        let x = rng.vec(heads * head_dim);
        let gamma = rng.vec(head_dim);
        let mut r = vec![0.0f32; heads];
        let mut y = vec![0.0f32; x.len()];
        QkNormConsts::capture(&x, &gamma, head_dim, eps, &mut r, &mut y);
        let c = QkNormConsts { x: &x, r: &r };
        let dx = rng.vec(x.len());
        let g = rng.vec(x.len());
        let mut dy = vec![0.0f32; x.len()];
        let mut dx_v = vec![0.0f32; x.len()];
        qk_norm_true_jvp(&c, &gamma, head_dim, &dx, &mut dy);
        qk_norm_true_vjp(&c, &gamma, head_dim, &g, &mut dx_v);
        close(dot(&dy, &g), dot(&dx, &dx_v), 1e-4, "qk_norm adjoint");
    }

    #[test]
    fn qg_split_is_adjoint() {
        let n_head = 4usize;
        let head_dim = 8usize;
        let mut rng = Lcg(0x91EED);
        let dqg = rng.vec(n_head * 2 * head_dim);
        let mut dq = vec![0.0f32; n_head * head_dim];
        let mut dgate = vec![0.0f32; n_head * head_dim];
        qg_deinterleave(&dqg, n_head, head_dim, &mut dq, &mut dgate);
        // round trip: the interleave of the split restores dqg exactly
        let mut round = vec![0.0f32; dqg.len()];
        qg_interleave(&dq, &dgate, n_head, head_dim, &mut round);
        for i in 0..dqg.len() {
            assert!((round[i] - dqg[i]).abs() < 1e-6, "qg round-trip [{i}]");
        }
        // adjoint: ⟨J·dqg, (bq, bg)⟩ == ⟨dqg, Jᵀ·(bq, bg)⟩ with INDEPENDENT
        // cotangent slots (never halves of dqg — the interleaved layout's
        // halves are not (q, gate)).
        let bq = rng.vec(n_head * head_dim);
        let bg = rng.vec(n_head * head_dim);
        let mut folded = vec![0.0f32; dqg.len()];
        qg_interleave(&bq, &bg, n_head, head_dim, &mut folded);
        close(
            dot(&dq, &bq) + dot(&dgate, &bg),
            dot(&dqg, &folded),
            1e-6,
            "qg adjoint",
        );
    }
}
