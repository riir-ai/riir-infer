//! Op-level forward (`jvp`) / adjoint (`vjp`) pairs for the RelP lens fit.
//!
//! Every function here is a linear map applied to tangent/cotangent buffers;
//! the nonlinear constants (norm scales, gate activations, silu derivatives)
//! are captured once per position by the `linearize` pass and passed in as
//! slices. See [`crate::lens`] (module docs) for the rule set and provenance.

// ---------------------------------------------------------------------------
// Scalar activations + derivatives (the Identity-rule constants)
// ---------------------------------------------------------------------------

/// Sigmoid-activation function.
#[inline(always)]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// SiLU (swish): `x · σ(x)`.
#[inline(always)]
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// `silu'(x) = σ(x)·(1 + x·(1 − σ(x)))` — exact local derivative
/// (the Identity rule's linearization constant for the gate branch).
#[inline(always)]
pub fn silu_grad(x: f32) -> f32 {
    let s = sigmoid(x);
    s * (1.0 + x * (1.0 - s))
}

// ---------------------------------------------------------------------------
// LN rule — RMSNorm tangent map (diagonal, self-adjoint)
// ---------------------------------------------------------------------------

/// LN-rule diagonal scale for one position: `scale[i] = γ[i] · rsqrt(mean(x²)+ε)`.
///
/// Under the LN rule the norm's `rsqrt` is held constant, so the tangent map
/// is the diagonal `(r ⊙ γ)` — and a diagonal map is its own adjoint
/// ([`diag_apply`] serves both `jvp` and `vjp`).
///
/// `eps` is f64 to match the forward's [`katgpt_types`-style] eps convention
/// (`rmsnorm_with_gamma_eps`); the resulting scale is f32.
pub fn norm_diag_scale(x: &[f32], gamma: &[f32], eps: f64, scale: &mut [f32]) {
    debug_assert_eq!(
        x.len(),
        gamma.len(),
        "norm_diag_scale: x/gamma length mismatch"
    );
    debug_assert_eq!(
        x.len(),
        scale.len(),
        "norm_diag_scale: scale length mismatch"
    );
    let n = x.len();
    debug_assert!(n > 0, "norm_diag_scale: empty position");
    let mut sum_sq = 0.0f64;
    for &v in x {
        sum_sq += (v as f64) * (v as f64);
    }
    let r = 1.0 / (sum_sq / n as f64 + eps).sqrt();
    for i in 0..n {
        scale[i] = gamma[i] * r as f32;
    }
}

/// Apply the (self-adjoint) LN-rule diagonal map in place: `v[i] *= scale[i]`.
#[inline(always)]
pub fn diag_apply(scale: &[f32], v: &mut [f32]) {
    debug_assert_eq!(scale.len(), v.len(), "diag_apply: length mismatch");
    for (v, s) in v.iter_mut().zip(scale) {
        *v *= s;
    }
}

// ---------------------------------------------------------------------------
// Linear maps — exact adjoints over the crate's row-major convention
// ---------------------------------------------------------------------------

/// Transposed matvec: `x = Wᵀ · y` for row-major `W [rows, cols]`.
///
/// The exact adjoint of [`crate::types::matmul`] (`y = W @ x`). Scalar loop
/// over the row-major layout (column reads are strided); the fit is offline —
/// a cache-blocking / SIMD transposed kernel is a follow-up, not a PoC need.
pub fn matvec_t(x: &mut [f32], weight: &[f32], y: &[f32], rows: usize, cols: usize) {
    debug_assert_eq!(
        weight.len(),
        rows * cols,
        "matvec_t: weight len != rows*cols"
    );
    debug_assert_eq!(x.len(), cols, "matvec_t: x len != cols");
    debug_assert_eq!(y.len(), rows, "matvec_t: y len != rows");
    for v in x.iter_mut() {
        *v = 0.0;
    }
    for (i, &y_i) in y.iter().enumerate().take(rows) {
        let row = &weight[i * cols..(i + 1) * cols];
        if y_i == 0.0 {
            continue;
        }
        for (j, &w_ij) in row.iter().enumerate() {
            x[j] += w_ij * y_i;
        }
    }
}

// ---------------------------------------------------------------------------
// RoPE — exact rotation; jvp reuses the forward, vjp is the inverse rotation
// ---------------------------------------------------------------------------

/// RoPE `jvp`: apply the FORWARD's exact rotation to tangent buffers.
///
/// The rotation is linear and position-determined, so the tangent map is the
/// same rotation. Delegates to [`crate::rope`] so the fit's pairing convention
/// (rotate-half: pairs `(i, i + rotary_dim/2)`) is the forward's by
/// construction. `rotary_dim == head_dim` selects the full-rotation path,
/// mirroring the forward's dispatch (`effective_rotary_dim`).
pub fn rope_jvp(
    dq: &mut [f32],
    dk: &mut [f32],
    pos: usize,
    head_dim: usize,
    rotary_dim: usize,
    freq_table: &[f32],
) {
    if rotary_dim == head_dim {
        crate::rope::apply_rope_with_freq(dq, dk, pos, head_dim, freq_table);
    } else {
        crate::rope::apply_partial_rope_with_freq(dq, dk, pos, head_dim, rotary_dim, freq_table);
    }
}

/// RoPE `vjp`: the inverse rotation (same pairing, negated sin).
///
/// Per-pair matrix `[[cos, −sin], [sin, cos]]` is orthogonal; its transpose
/// swaps the sign of the off-diagonal sin terms. In-place per head — each
/// pair's two indices are read before either is written.
pub fn rope_vjp(
    dq: &mut [f32],
    dk: &mut [f32],
    pos: usize,
    head_dim: usize,
    rotary_dim: usize,
    freq_table: &[f32],
) {
    if pos == 0 || rotary_dim == 0 {
        return; // identity, mirroring the forward's fast paths
    }
    let half = rotary_dim / 2;
    debug_assert!(freq_table.len() >= half, "rope_vjp: freq_table too short");
    debug_assert!(rotary_dim <= head_dim, "rope_vjp: rotary_dim > head_dim");
    debug_assert!(
        rotary_dim.is_multiple_of(2),
        "rope_vjp: rotary_dim must be even"
    );

    let mut cos_sin = [0.0f32; 512];
    let use_stack = half <= 256;
    let mut heap_buf: Vec<f32>;
    let buf: &mut [f32] = if use_stack {
        &mut cos_sin[..half * 2]
    } else {
        heap_buf = vec![0.0f32; half * 2];
        &mut heap_buf
    };
    let pos_f = pos as f32;
    for i in 0..half {
        let angle = pos_f * freq_table[i];
        let (sin_a, cos_a) = angle.sin_cos();
        buf[i] = cos_a;
        buf[half + i] = sin_a;
    }
    let (cos_table, sin_table) = buf.split_at(half);

    let n_heads_q = dq.len() / head_dim;
    let n_heads_k = dk.len() / head_dim;
    for h in 0..n_heads_q {
        rotate_head_rev(
            &mut dq[h * head_dim..h * head_dim + rotary_dim],
            cos_table,
            sin_table,
        );
    }
    for h in 0..n_heads_k {
        rotate_head_rev(
            &mut dk[h * head_dim..h * head_dim + rotary_dim],
            cos_table,
            sin_table,
        );
    }
}

/// In-place inverse rotation of one head's first `rotary_dim` dims
/// (rotate-half pairing `(i, i + half)`, sin negated).
fn rotate_head_rev(v: &mut [f32], cos_table: &[f32], sin_table: &[f32]) {
    let half = cos_table.len();
    for i in 0..half {
        let x = v[i];
        let y = v[i + half];
        let (c, s) = (cos_table[i], sin_table[i]);
        v[i] = c * x + s * y;
        v[i + half] = -s * x + c * y;
    }
}

// ---------------------------------------------------------------------------
// Half rule — SwiGLU MLP (gate · up) tangent map
// ---------------------------------------------------------------------------

/// Captured per-neuron MLP constants for one position (the `linearize`
/// pass computes these once from the gate/up pre-activations).
#[derive(Clone, Copy, Debug)]
pub struct MlpConsts<'a> {
    /// `s = silu(g)` — the gate branch's value.
    pub s: &'a [f32],
    /// `u` — the up branch's value.
    pub u: &'a [f32],
    /// `sp = silu'(g)` — the gate branch's local derivative.
    pub sp: &'a [f32],
}

impl<'a> MlpConsts<'a> {
    /// Capture from gate/up pre-activations into caller-provided buffers.
    ///
    /// `g`/`u_pre` are the raw projections (`W_gate·x̂`, `W_up·x̂`); fills
    /// `s_out`, `u_out` (copy), `sp_out`. Lengths must all equal `mlp_hidden`.
    pub fn capture(
        g: &[f32],
        u_pre: &[f32],
        s_out: &mut [f32],
        u_out: &mut [f32],
        sp_out: &mut [f32],
    ) {
        debug_assert_eq!(
            g.len(),
            u_pre.len(),
            "MlpConsts::capture: g/u length mismatch"
        );
        debug_assert_eq!(
            g.len(),
            s_out.len(),
            "MlpConsts::capture: s_out length mismatch"
        );
        debug_assert_eq!(
            g.len(),
            u_out.len(),
            "MlpConsts::capture: u_out length mismatch"
        );
        debug_assert_eq!(
            g.len(),
            sp_out.len(),
            "MlpConsts::capture: sp_out length mismatch"
        );
        for i in 0..g.len() {
            s_out[i] = silu(g[i]);
            u_out[i] = u_pre[i];
            sp_out[i] = silu_grad(g[i]);
        }
    }
}

/// MLP `jvp` under the half rule: tangent `(dg, du)` → `dY = W_down · dact`,
/// `dact = ½(s⊙du + u⊙sp⊙dg)`.
///
/// `w_down` is row-major `[n_embd, mlp_hidden]` (rows = residual dim);
/// `dact_scratch` is caller-provided `[mlp_hidden]` (zero-alloc).
pub fn mlp_half_rule_jvp(
    c: &MlpConsts,
    dg: &[f32],
    du: &[f32],
    w_down: &[f32],
    n_embd: usize,
    dact_scratch: &mut [f32],
    dy: &mut [f32],
) {
    let hidden = c.s.len();
    debug_assert_eq!(hidden, c.u.len(), "mlp jvp: consts length mismatch");
    debug_assert_eq!(hidden, c.sp.len(), "mlp jvp: consts length mismatch");
    debug_assert_eq!(hidden, dg.len(), "mlp jvp: dg length mismatch");
    debug_assert_eq!(hidden, du.len(), "mlp jvp: du length mismatch");
    debug_assert_eq!(
        hidden,
        dact_scratch.len(),
        "mlp jvp: scratch length mismatch"
    );
    debug_assert_eq!(
        w_down.len(),
        n_embd * hidden,
        "mlp jvp: w_down shape mismatch"
    );
    debug_assert_eq!(dy.len(), n_embd, "mlp jvp: dy length mismatch");
    for i in 0..hidden {
        dact_scratch[i] = 0.5 * (c.s[i] * du[i] + c.u[i] * c.sp[i] * dg[i]);
    }
    crate::types::matmul(dy, w_down, dact_scratch, n_embd, hidden);
}

/// MLP `vjp` — the exact adjoint of [`mlp_half_rule_jvp`]:
/// `dact = W_downᵀ · dY`, `dg = ½·u⊙sp⊙dact`, `du = ½·s⊙dact`.
pub fn mlp_half_rule_vjp(
    c: &MlpConsts,
    dy: &[f32],
    w_down: &[f32],
    n_embd: usize,
    dact_scratch: &mut [f32],
    dg: &mut [f32],
    du: &mut [f32],
) {
    let hidden = c.s.len();
    debug_assert_eq!(hidden, c.u.len(), "mlp vjp: consts length mismatch");
    debug_assert_eq!(hidden, c.sp.len(), "mlp vjp: consts length mismatch");
    debug_assert_eq!(
        hidden,
        dact_scratch.len(),
        "mlp vjp: scratch length mismatch"
    );
    debug_assert_eq!(hidden, dg.len(), "mlp vjp: dg length mismatch");
    debug_assert_eq!(hidden, du.len(), "mlp vjp: du length mismatch");
    debug_assert_eq!(
        w_down.len(),
        n_embd * hidden,
        "mlp vjp: w_down shape mismatch"
    );
    debug_assert_eq!(dy.len(), n_embd, "mlp vjp: dy length mismatch");
    matvec_t(dact_scratch, w_down, dy, n_embd, hidden);
    for i in 0..hidden {
        let d = dact_scratch[i];
        dg[i] = 0.5 * c.u[i] * c.sp[i] * d;
        du[i] = 0.5 * c.s[i] * d;
    }
}

// ---------------------------------------------------------------------------
// Half rule — GDN output gate (σ(g) · z) tangent map
// ---------------------------------------------------------------------------

/// Captured per-channel gate constants for one position/time-step.
#[derive(Clone, Copy, Debug)]
pub struct GateConsts<'a> {
    /// `σ(g)` — the gate branch's sigmoid value.
    pub sg: &'a [f32],
    /// `z` — the value branch (what the gate scales).
    pub z: &'a [f32],
    /// `σ'(g)` — the gate branch's local derivative.
    pub spg: &'a [f32],
}

impl<'a> GateConsts<'a> {
    /// Capture from the gate pre-activation `g` and the value `z`.
    pub fn capture(
        g: &[f32],
        z: &[f32],
        sg_out: &mut [f32],
        z_out: &mut [f32],
        spg_out: &mut [f32],
    ) {
        debug_assert_eq!(g.len(), z.len(), "GateConsts::capture: g/z length mismatch");
        debug_assert_eq!(
            g.len(),
            sg_out.len(),
            "GateConsts::capture: sg_out length mismatch"
        );
        debug_assert_eq!(
            g.len(),
            z_out.len(),
            "GateConsts::capture: z_out length mismatch"
        );
        debug_assert_eq!(
            g.len(),
            spg_out.len(),
            "GateConsts::capture: spg_out length mismatch"
        );
        for i in 0..g.len() {
            let s = sigmoid(g[i]);
            sg_out[i] = s;
            z_out[i] = z[i];
            spg_out[i] = s * (1.0 - s);
        }
    }
}

/// Gate `jvp` under the half rule: tangent `(dg, dz)` → `dy` in place,
/// `dy = ½(σg⊙dz + z⊙σ'g⊙dg)`.
pub fn gate_half_rule_jvp(c: &GateConsts, dg: &[f32], dz: &[f32], dy: &mut [f32]) {
    let n = c.sg.len();
    debug_assert_eq!(n, c.z.len(), "gate jvp: consts length mismatch");
    debug_assert_eq!(n, c.spg.len(), "gate jvp: consts length mismatch");
    debug_assert_eq!(n, dg.len(), "gate jvp: dg length mismatch");
    debug_assert_eq!(n, dz.len(), "gate jvp: dz length mismatch");
    debug_assert_eq!(n, dy.len(), "gate jvp: dy length mismatch");
    for i in 0..n {
        dy[i] = 0.5 * (c.sg[i] * dz[i] + c.z[i] * c.spg[i] * dg[i]);
    }
}

/// Gate `vjp` — the exact adjoint of [`gate_half_rule_jvp`] (self-adjoint up
/// to the shared `½` factor: the map is diagonal in `(dg, dz)`, so the adjoint
/// re-applies the same two diagonal factors to the swapped outputs).
pub fn gate_half_rule_vjp(c: &GateConsts, dy: &[f32], dg: &mut [f32], dz: &mut [f32]) {
    let n = c.sg.len();
    debug_assert_eq!(n, c.z.len(), "gate vjp: consts length mismatch");
    debug_assert_eq!(n, c.spg.len(), "gate vjp: consts length mismatch");
    debug_assert_eq!(n, dy.len(), "gate vjp: dy length mismatch");
    debug_assert_eq!(n, dg.len(), "gate vjp: dg length mismatch");
    debug_assert_eq!(n, dz.len(), "gate vjp: dz length mismatch");
    for i in 0..n {
        let d = dy[i];
        dg[i] = 0.5 * c.z[i] * c.spg[i] * d;
        dz[i] = 0.5 * c.sg[i] * d;
    }
}

// ---------------------------------------------------------------------------
// Causal depthwise conv1d — linear; exact adjoint
// ---------------------------------------------------------------------------

/// Causal depthwise conv1d `jvp`: `dy[t,c] = Σ_k K[c,k] · dx[t−k, c]`
/// (`x[t−k] = 0` for `t < k`; output length == input length).
///
/// `kernel` is `[channels, kernel_w]` row-major. This is the left-padded
/// causal convention; the T2d fit pins bit-parity against the GDN forward's
/// exact loop before composing (T2c), the same way `rope_jvp` pins against
/// [`crate::rope`].
pub fn conv1d_causal_jvp(
    kernel: &[f32],
    kernel_w: usize,
    dx: &[f32],
    channels: usize,
    dy: &mut [f32],
) {
    let t_len = dx.len() / channels;
    debug_assert_eq!(dx.len(), t_len * channels, "conv jvp: dx len != t*channels");
    debug_assert_eq!(dy.len(), dx.len(), "conv jvp: dy length mismatch");
    debug_assert_eq!(
        kernel.len(),
        channels * kernel_w,
        "conv jvp: kernel shape mismatch"
    );
    for v in dy.iter_mut() {
        *v = 0.0;
    }
    for t in 0..t_len {
        for c in 0..channels {
            let mut acc = 0.0f32;
            let k_max = kernel_w.min(t + 1);
            for k in 0..k_max {
                acc += kernel[c * kernel_w + k] * dx[(t - k) * channels + c];
            }
            dy[t * channels + c] = acc;
        }
    }
}

/// Conv1d `vjp` — the exact adjoint of [`conv1d_causal_jvp`]:
/// `dx[t,c] = Σ_k K[c,k] · dy[t+k, c]` (for `t + k < T`).
pub fn conv1d_causal_vjp(
    kernel: &[f32],
    kernel_w: usize,
    dy: &[f32],
    channels: usize,
    dx: &mut [f32],
) {
    let t_len = dy.len() / channels;
    debug_assert_eq!(dy.len(), t_len * channels, "conv vjp: dy len != t*channels");
    debug_assert_eq!(dx.len(), dy.len(), "conv vjp: dx length mismatch");
    debug_assert_eq!(
        kernel.len(),
        channels * kernel_w,
        "conv vjp: kernel shape mismatch"
    );
    for v in dx.iter_mut() {
        *v = 0.0;
    }
    for t in 0..t_len {
        for c in 0..channels {
            let mut acc = 0.0f32;
            let k_max = kernel_w.min(t_len - t);
            for k in 0..k_max {
                acc += kernel[c * kernel_w + k] * dy[(t + k) * channels + c];
            }
            dx[t * channels + c] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests — adjointness per map + RoPE round-trip vs the forward
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic LCG — no unseeded-global draws in gate tests.
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

    /// Adjointness: `⟨J·dx, dy⟩ == ⟨dx, Jᵀ·dy⟩` within f32 accumulation noise.
    fn assert_adjoint(jvp: &[f32], vjp_result: &[f32], dx: &[f32], dy: &[f32]) {
        let lhs = dot(jvp, dy);
        let rhs = dot(dx, vjp_result);
        let scale = lhs.abs().max(rhs.abs()).max(1.0);
        let rel = (lhs - rhs).abs() / scale;
        assert!(
            rel < 1e-4,
            "adjointness violated: lhs {lhs:.6} rhs {rhs:.6} rel {rel:.2e}"
        );
    }

    #[test]
    fn silu_grad_matches_finite_difference() {
        let mut rng = Lcg(0x9999);
        for _ in 0..64 {
            let x = rng.next_f32() * 8.0;
            let h = 1e-3f32;
            let fd = (silu(x + h) - silu(x - h)) / (2.0 * h);
            let g = silu_grad(x);
            assert!((fd - g).abs() < 2e-3, "silu_grad({x}) {g} vs fd {fd}");
        }
    }

    #[test]
    fn norm_diag_is_self_adjoint_and_matches_rmsnorm() {
        let mut rng = Lcg(0x1234);
        let n = 128;
        let x = rng.vec(n);
        let gamma = rng.vec(n);
        let eps = 1e-6;
        let mut scale = vec![0.0f32; n];
        norm_diag_scale(&x, &gamma, eps, &mut scale);

        // The diagonal must reproduce the forward's normed output when applied
        // to x itself: scale ⊙ x == rmsnorm_with_gamma_eps(x).
        let mut forward_x = x.clone();
        crate::types::rmsnorm_with_gamma_eps(&mut forward_x, &gamma, eps);
        let mut applied = x.clone();
        diag_apply(&scale, &mut applied);
        for i in 0..n {
            assert!(
                (applied[i] - forward_x[i]).abs() < 1e-4,
                "LN-rule scale ⊙ x != forward rmsnorm at {i}: {} vs {}",
                applied[i],
                forward_x[i]
            );
        }

        // Self-adjointness: ⟨S·dx, dy⟩ == ⟨dx, S·dy⟩ (diagonal, f32-exact both
        // sides up to multiply order).
        let dx = rng.vec(n);
        let dy = rng.vec(n);
        let mut sx = dx.clone();
        diag_apply(&scale, &mut sx);
        let mut sy = dy.clone();
        diag_apply(&scale, &mut sy);
        assert_adjoint(&sx, &sy, &dx, &dy);
    }

    #[test]
    fn matvec_t_is_adjoint_of_matmul() {
        let mut rng = Lcg(0x5678);
        let (rows, cols) = (48, 96);
        let w = rng.vec(rows * cols);
        let x = rng.vec(cols);
        let dy = rng.vec(rows);
        let mut y = vec![0.0f32; rows];
        crate::types::matmul(&mut y, &w, &x, rows, cols);
        let mut dx = vec![0.0f32; cols];
        matvec_t(&mut dx, &w, &dy, rows, cols);
        assert_adjoint(&y, &dx, &x, &dy);
    }

    /// RoPE round trip: `vjp(jvp(x)) == x` for full and partial rotation —
    /// pins the vjp's pairing against the FORWARD's exact convention.
    #[test]
    fn rope_vjp_undoes_rope_jvp() {
        let head_dim = 64usize;
        let theta = 1_000_000.0f32;
        for &rotary_dim in &[head_dim, head_dim / 2] {
            let freq = crate::rope::RopeFreqTable::new(theta, rotary_dim);
            let mut rng = Lcg(0xABCD);
            let n_q_heads = 4usize;
            let n_k_heads = 2usize;
            let q_len = n_q_heads * head_dim;
            let k_len = n_k_heads * head_dim;
            for &pos in &[1usize, 5, 130, 1023] {
                let q = rng.vec(q_len);
                let k = rng.vec(k_len);
                let mut dq = q.clone();
                let mut dk = k.clone();
                rope_jvp(&mut dq, &mut dk, pos, head_dim, rotary_dim, freq.as_slice());
                rope_vjp(&mut dq, &mut dk, pos, head_dim, rotary_dim, freq.as_slice());
                for i in 0..q_len {
                    assert!(
                        (dq[i] - q[i]).abs() < 1e-3,
                        "rope round-trip q[{i}] pos {pos} rotary {rotary_dim}: {} vs {}",
                        dq[i],
                        q[i]
                    );
                }
                for i in 0..k_len {
                    assert!(
                        (dk[i] - k[i]).abs() < 1e-3,
                        "rope round-trip k[{i}] pos {pos} rotary {rotary_dim}: {} vs {}",
                        dk[i],
                        k[i]
                    );
                }
            }
        }
    }

    /// Adjointness of the rotation itself: ⟨R·dq, dq′⟩ == ⟨dq, Rᵀ·dq′⟩.
    #[test]
    fn rope_is_adjoint() {
        let head_dim = 32usize;
        let rotary_dim = head_dim;
        let theta = 10_000.0f32;
        let freq = crate::rope::RopeFreqTable::new(theta, rotary_dim);
        let mut rng = Lcg(0xBEEF);
        let len = 3 * head_dim;
        let dq = rng.vec(len);
        let dq2 = rng.vec(len);
        let mut rot = dq.clone();
        let mut rot_t = dq2.clone();
        rope_jvp(&mut rot, &mut [], 7, head_dim, rotary_dim, freq.as_slice());
        rope_vjp(
            &mut rot_t,
            &mut [],
            7,
            head_dim,
            rotary_dim,
            freq.as_slice(),
        );
        assert_adjoint(&rot, &rot_t, &dq, &dq2);
    }

    #[test]
    fn mlp_half_rule_is_adjoint() {
        let mut rng = Lcg(0x2468);
        let (n_embd, hidden) = (32, 96);
        let g = rng.vec(hidden);
        let u_pre = rng.vec(hidden);
        let mut s = vec![0.0f32; hidden];
        let mut u = vec![0.0f32; hidden];
        let mut sp = vec![0.0f32; hidden];
        MlpConsts::capture(&g, &u_pre, &mut s, &mut u, &mut sp);
        let c = MlpConsts {
            s: &s,
            u: &u,
            sp: &sp,
        };
        let w_down = rng.vec(n_embd * hidden);
        let dg = rng.vec(hidden);
        let du = rng.vec(hidden);
        // Map: (dg,du) → dy = W_down·dact, dact = ½(s⊙du + u⊙sp⊙dg).
        let mut dact = vec![0.0f32; hidden];
        let mut dy = vec![0.0f32; n_embd];
        mlp_half_rule_jvp(&c, &dg, &du, &w_down, n_embd, &mut dact, &mut dy);
        // Adjoint: dy → (dg2,du2) via dact2 = W_downᵀ·dy. The vjp reuses the
        // scratch for dact2, so keep the forward dact around for the identity.
        let dact_fwd = dact.clone();
        let mut dg2 = vec![0.0f32; hidden];
        let mut du2 = vec![0.0f32; hidden];
        mlp_half_rule_vjp(&c, &dy, &w_down, n_embd, &mut dact, &mut dg2, &mut du2);
        // ⟨W_down·dact, dy⟩ == ⟨dact, W_downᵀ·dy⟩, and the half rule is
        // diagonal, so the full identity reduces to ⟨dact_fwd, dact⟩ vs
        // ⟨(dg,du), (dg2,du2)⟩.
        let lhs = dot(&dact_fwd, &dact);
        let rhs = dot(&dg, &dg2) + dot(&du, &du2);
        assert!(
            (lhs - rhs).abs() / lhs.abs().max(1.0) < 1e-4,
            "mlp half rule adjointness: lhs {lhs:.6} rhs {rhs:.6}"
        );
    }

    #[test]
    fn gate_half_rule_is_adjoint() {
        let mut rng = Lcg(0x1357);
        let n = 64;
        let g = rng.vec(n);
        let z = rng.vec(n);
        let mut sg = vec![0.0f32; n];
        let mut z_out = vec![0.0f32; n];
        let mut spg = vec![0.0f32; n];
        GateConsts::capture(&g, &z, &mut sg, &mut z_out, &mut spg);
        let c = GateConsts {
            sg: &sg,
            z: &z_out,
            spg: &spg,
        };
        let dg = rng.vec(n);
        let dz = rng.vec(n);
        let mut dy = vec![0.0f32; n];
        gate_half_rule_jvp(&c, &dg, &dz, &mut dy);
        // diagonal in (dg,dz): ⟨dy, dy′⟩ == ⟨(dg,dz), (dg′,dz′)⟩ for dy′ = jvp(dg′,dz′)
        let dg2 = rng.vec(n);
        let dz2 = rng.vec(n);
        let mut dy2 = vec![0.0f32; n];
        gate_half_rule_jvp(&c, &dg2, &dz2, &mut dy2);
        let mut dg_v = vec![0.0f32; n];
        let mut dz_v = vec![0.0f32; n];
        gate_half_rule_vjp(&c, &dy2, &mut dg_v, &mut dz_v);
        let lhs = dot(&dy, &dy2);
        let rhs = dot(&dg, &dg_v) + dot(&dz, &dz_v);
        assert!(
            (lhs - rhs).abs() / lhs.abs().max(1.0) < 1e-4,
            "gate half rule adjointness: lhs {lhs:.6} rhs {rhs:.6}"
        );
    }

    #[test]
    fn conv1d_causal_is_adjoint() {
        let mut rng = Lcg(0x8642);
        let (channels, kernel_w, t_len) = (8usize, 4usize, 24usize);
        let kernel = rng.vec(channels * kernel_w);
        let dx = rng.vec(t_len * channels);
        let mut dy = vec![0.0f32; dx.len()];
        conv1d_causal_jvp(&kernel, kernel_w, &dx, channels, &mut dy);
        let dy2 = rng.vec(t_len * channels);
        let mut dx_v = vec![0.0f32; dx.len()];
        conv1d_causal_vjp(&kernel, kernel_w, &dy2, channels, &mut dx_v);
        assert_adjoint(&dy, &dx_v, &dx, &dy2);
    }

    /// The conv `jvp` must equal direct evaluation of the recurrence on the
    /// tangent (finite-difference cross-check through a LINEAR map is exact).
    #[test]
    fn conv1d_causal_matches_direct_evaluation() {
        let mut rng = Lcg(0x7777);
        let (channels, kernel_w, t_len) = (4usize, 3usize, 12usize);
        let kernel = rng.vec(channels * kernel_w);
        let x = rng.vec(t_len * channels);
        let mut dy = vec![0.0f32; x.len()];
        conv1d_causal_jvp(&kernel, kernel_w, &x, channels, &mut dy);
        // Direct: pad-left and slide.
        for t in 0..t_len {
            for c in 0..channels {
                let mut acc = 0.0f32;
                for k in 0..kernel_w {
                    if t >= k {
                        acc += kernel[c * kernel_w + k] * x[(t - k) * channels + c];
                    }
                }
                assert!(
                    (acc - dy[t * channels + c]).abs() < 1e-5,
                    "conv jvp direct mismatch at (t={t}, c={c})"
                );
            }
        }
    }
}
