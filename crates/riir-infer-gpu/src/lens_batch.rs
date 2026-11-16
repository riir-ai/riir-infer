//! Host driver for the batched lens R-sweep (riir-infer Issue 1222).
//!
//! [`LensBatchStack`] uploads the stack weights ONCE, a per-doc capture is
//! uploaded with [`LensDocCapture::upload`], and [`suffix_pass`] runs ONE
//! direction block (`batch` directions) through the suffix tangent stack
//! for one [`FitPoint`], returning the per-direction column sums
//! `[batch · d]` — the R-sweep unit the fit example loops over
//! points × blocks (the SAME uploaded `dv` block serves every point).
//!
//! The composition mirrors `riir_infer_core::lens::stack`'s
//! `stack_suffix_jvp` op-for-op in the batched layouts
//! ([`crate::lens_batch_kernels`]). K (the prefix sweep) stays CPU-side —
//! it is ~1/25 of the R sweep's cost, and its per-layer mid/out recording
//! shape is a later GPU increment.
//!
//! Never a serving path — offline diagnostics only.

#![cfg(feature = "lens_gpu")]

use crate::lens_batch_kernels::{
    LensAddInplaceCL, LensAttnJvpCL, LensColSumCL, LensConv1dCL, LensCopyRowSliceCL, LensGdnScanCL,
    LensGemmCL, LensHalfRule2CL, LensL2NormExpandCL, LensQgSplitCL, LensQkNormCL, LensRopeJvpCL,
    LensScaleTableCL,
};
use riir_infer_core::lens::attn::AttnGeometry;
use riir_infer_core::lens::gdn::GdnGeometry;
use riir_infer_core::lens::stack::{FitPoint, StackCapture, StackWeights};

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::{ComputeClient, Runtime};

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

#[cfg(feature = "cubecl_runtime")]
use crate::cubecl_runtime::{create_f32, read_f32};

// ---------------------------------------------------------------------------
// Dims
// ---------------------------------------------------------------------------

/// The stack's batched-sweep geometry (host-side; derives every kernel
/// launch shape).
#[derive(Clone, Debug)]
pub struct LensBatchDims {
    pub batch: usize,
    pub t_len: usize,
    pub n_layers: usize,
    pub n_embd: usize,
    pub mlp_hidden: usize,
    pub layer_is_gdn: Vec<bool>,
    /// The GDN sublayer dims (`None` on an attention-only stack).
    pub gdn: Option<GdnGeometry>,
    /// The attention sublayer dims (`None` on a GDN-only stack).
    pub attn: Option<AttnGeometry>,
}

impl LensBatchDims {
    pub fn new(w: &StackWeights, batch: usize) -> Self {
        let geo = &w.geo;
        // The sublayer geos are type-uniform (the geometry is global); take
        // them from the first layer of each kind.
        let gdn = geo.layer_is_gdn.iter().any(|g| *g).then(|| geo.gdn_geo());
        let attn = geo.layer_is_gdn.iter().any(|g| !g).then(|| geo.attn_geo());
        Self {
            batch,
            t_len: geo.t_len,
            n_layers: geo.n_layers,
            n_embd: geo.n_embd,
            mlp_hidden: geo.mlp_hidden,
            layer_is_gdn: geo.layer_is_gdn.clone(),
            gdn,
            attn,
        }
    }
}

// ---------------------------------------------------------------------------
// Uploaded weights (once per stack)
// ---------------------------------------------------------------------------

#[cfg(feature = "cubecl_runtime")]
#[derive(Clone)]
struct GdnWeightHandles {
    w_qkv: Handle,
    w_z: Handle,
    w_a: Handle,
    w_b: Handle,
    w_out: Handle,
    conv_w: Handle,
    linear_norm: Handle,
}

#[cfg(feature = "cubecl_runtime")]
#[derive(Clone)]
struct AttnWeightHandles {
    w_qg: Handle,
    w_k: Handle,
    w_v: Handle,
    w_o: Handle,
    q_norm: Handle,
    k_norm: Handle,
    /// Host-precomputed per-position `(cos, sin)` table `[t_len, 2·half]` —
    /// the same host-side `sin_cos` values the CPU rope computes, so the
    /// rotation's parity is inherited (the kernel never converts a
    /// position index to f32).
    rope_cs: Handle,
}

#[cfg(feature = "cubecl_runtime")]
#[derive(Clone)]
enum SubWeightHandles {
    Gdn(GdnWeightHandles),
    Attn(AttnWeightHandles),
}

#[cfg(feature = "cubecl_runtime")]
#[derive(Clone)]
struct LayerWeightHandles {
    sub: SubWeightHandles,
    w_gate: Handle,
    w_up: Handle,
    w_down: Handle,
}

/// The uploaded stack weights + persistent batched scratch (drop releases
/// the handles).
#[cfg(feature = "cubecl_runtime")]
pub struct LensBatchStack {
    pub dims: LensBatchDims,
    layers: Vec<LayerWeightHandles>,
    /// Persistent scratch (sized by `batch`; reused across passes).
    t2: Handle,
    dmlp: Handle,
    dg: Handle,
    du: Handle,
    dact: Handle,
    /// GDN scratch (present iff the stack has GDN layers).
    gdn: Option<GdnScratch>,
    /// Attention scratch (present iff the stack has attention layers).
    attn: Option<AttnScratch>,
    colsum: Handle,
}

#[cfg(feature = "cubecl_runtime")]
struct GdnScratch {
    dqkv_raw: Handle,
    dconv: Handle,
    dqkv: Handle,
    dz: Handle,
    da: Handle,
    db: Handle,
    dbeta: Handle,
    ddecay: Handle,
    dq: Handle,
    dk: Handle,
    dv: Handle,
    dqn: Handle,
    dkn: Handle,
    ds: Handle,
    spre: Handle,
    dy_pre: Handle,
    dnormed: Handle,
    dout: Handle,
}

#[cfg(feature = "cubecl_runtime")]
struct AttnScratch {
    dqg: Handle,
    dgate: Handle,
    dq: Handle,
    dk: Handle,
    dv: Handle,
    dq2: Handle,
    dk2: Handle,
    dgated: Handle,
    dgated2: Handle,
}

#[cfg(feature = "cubecl_runtime")]
impl LensBatchStack {
    /// Upload the stack's weights + allocate the batched scratch.
    pub fn new<R: Runtime>(client: &ComputeClient<R>, w: &StackWeights, batch: usize) -> Self {
        let dims = LensBatchDims::new(w, batch);
        let (t_len, d, hidden) = (dims.t_len, dims.n_embd, dims.mlp_hidden);
        let layers = w
            .layers
            .iter()
            .map(|lw| {
                let sub = match &lw.sub {
                    riir_infer_core::lens::stack::SubWeights::Gdn(gw) => {
                        SubWeightHandles::Gdn(GdnWeightHandles {
                            w_qkv: create_f32(client, gw.w_qkv),
                            w_z: create_f32(client, gw.w_z),
                            w_a: create_f32(client, gw.w_a),
                            w_b: create_f32(client, gw.w_b),
                            w_out: create_f32(client, gw.w_out),
                            conv_w: create_f32(client, gw.conv_w),
                            linear_norm: create_f32(client, gw.linear_norm),
                        })
                    }
                    riir_infer_core::lens::stack::SubWeights::Attn(aw) => {
                        SubWeightHandles::Attn(AttnWeightHandles {
                            w_qg: create_f32(client, aw.w_qg),
                            w_k: create_f32(client, aw.w_k),
                            w_v: create_f32(client, aw.w_v),
                            w_o: create_f32(client, aw.w_o),
                            q_norm: create_f32(client, aw.q_norm),
                            k_norm: create_f32(client, aw.k_norm),
                            rope_cs: create_f32(
                                client,
                                &rope_cs_table(
                                    aw.rope_freq,
                                    t_len,
                                    dims.attn
                                        .as_ref()
                                        .map(|a| a.rotary_dim)
                                        .unwrap_or(aw.rope_freq.len() * 2),
                                ),
                            ),
                        })
                    }
                };
                LayerWeightHandles {
                    sub,
                    w_gate: create_f32(client, lw.w_gate),
                    w_up: create_f32(client, lw.w_up),
                    w_down: create_f32(client, lw.w_down),
                }
            })
            .collect();
        // Scratch — client.empty takes BYTES.
        let e = |n: usize| client.empty(n * std::mem::size_of::<f32>());
        let td_b = batch * t_len * d;
        let th_b = batch * t_len * hidden;
        let gdn = dims.gdn.as_ref().map(|gdn| {
            let (n_v, key, val) = (gdn.n_v_heads, gdn.key_dim, gdn.val_dim);
            let conv_dim = gdn.conv_dim();
            let q_dim = gdn.q_dim();
            GdnScratch {
                dqkv_raw: e(batch * t_len * conv_dim),
                dconv: e(batch * t_len * conv_dim),
                dqkv: e(batch * t_len * conv_dim),
                dz: e(batch * t_len * n_v * val),
                da: e(batch * t_len * n_v),
                db: e(batch * t_len * n_v),
                dbeta: e(batch * t_len * n_v),
                ddecay: e(batch * t_len * n_v),
                dq: e(batch * t_len * q_dim),
                dk: e(batch * t_len * q_dim),
                dv: e(batch * t_len * n_v * val),
                dqn: e(batch * t_len * n_v * key),
                dkn: e(batch * t_len * n_v * key),
                ds: e(batch * n_v * val * key),
                spre: e(batch * n_v * val * key),
                dy_pre: e(batch * t_len * n_v * val),
                dnormed: e(batch * t_len * n_v * val),
                dout: e(batch * t_len * n_v * val),
            }
        });
        let attn = dims.attn.as_ref().map(|attn| {
            let q_dim = attn.q_dim();
            let kv_dim = attn.kv_dim();
            let qg_dim = attn.qg_dim();
            AttnScratch {
                dqg: e(batch * t_len * qg_dim),
                dgate: e(batch * t_len * q_dim),
                dq: e(batch * t_len * q_dim),
                dk: e(batch * t_len * kv_dim),
                dv: e(batch * t_len * kv_dim),
                dq2: e(batch * t_len * q_dim),
                dk2: e(batch * t_len * kv_dim),
                dgated: e(batch * t_len * q_dim),
                dgated2: e(batch * t_len * q_dim),
            }
        });
        Self {
            t2: e(td_b),
            dmlp: e(td_b),
            dg: e(th_b),
            du: e(th_b),
            dact: e(th_b),
            gdn,
            attn,
            colsum: e(batch * d),
            layers,
            dims,
        }
    }
}

// ---------------------------------------------------------------------------
// Uploaded capture (once per doc)
// ---------------------------------------------------------------------------

#[cfg(feature = "cubecl_runtime")]
#[derive(Clone)]
struct GdnCaptureHandles {
    conv_silu_grad: Handle,
    qn: Handle,
    kn: Handle,
    /// The post-conv value `[T · v_dim]` (the scan's shared value row).
    v: Handle,
    qn_inv: Handle,
    kn_inv: Handle,
    beta: Handle,
    decay: Handle,
    dbeta_coef: Handle,
    ddecay_coef: Handle,
    states: Handle,
    y_pre: Handle,
    norm_r: Handle,
    gate_s: Handle,
    gate_sp: Handle,
    normed: Handle,
}

#[cfg(feature = "cubecl_runtime")]
#[derive(Clone)]
struct AttnCaptureHandles {
    q_pre: Handle,
    k_pre: Handle,
    q_r: Handle,
    k_r: Handle,
    q: Handle,
    k: Handle,
    v: Handle,
    attn: Handle,
    att: Handle,
    gate_sg: Handle,
    gate_spg: Handle,
}

#[cfg(feature = "cubecl_runtime")]
#[derive(Clone)]
#[allow(clippy::large_enum_variant)] // per-layer metadata, ~n_layers instances; no memory concern
enum SubCaptureHandles {
    Gdn(GdnCaptureHandles),
    Attn(AttnCaptureHandles),
}

#[cfg(feature = "cubecl_runtime")]
#[derive(Clone)]
struct LayerCaptureHandles {
    sub: SubCaptureHandles,
    input_scale: Handle,
    post_scale: Handle,
    mlp_s: Handle,
    mlp_u: Handle,
    mlp_sp: Handle,
}

/// One doc's uploaded capture constants (upload after `stack_forward`;
/// drop releases the handles).
#[cfg(feature = "cubecl_runtime")]
pub struct LensDocCapture {
    layers: Vec<LayerCaptureHandles>,
    final_scale: Handle,
}

#[cfg(feature = "cubecl_runtime")]
impl LensDocCapture {
    pub fn upload<R: Runtime>(client: &ComputeClient<R>, cap: &StackCapture) -> Self {
        let layers = cap
            .layers
            .iter()
            .map(|lc| {
                let sub = match &lc.sub {
                    riir_infer_core::lens::stack::SubCapture::Gdn(g) => {
                        SubCaptureHandles::Gdn(GdnCaptureHandles {
                            conv_silu_grad: create_f32(client, &g.conv_silu_grad),
                            qn: create_f32(client, &g.qn),
                            kn: create_f32(client, &g.kn),
                            v: create_f32(client, &g.v),
                            qn_inv: create_f32(client, &g.qn_inv),
                            kn_inv: create_f32(client, &g.kn_inv),
                            beta: create_f32(client, &g.beta),
                            decay: create_f32(client, &g.decay),
                            dbeta_coef: create_f32(client, &g.dbeta_coef),
                            ddecay_coef: create_f32(client, &g.ddecay_coef),
                            states: create_f32(client, &g.states),
                            y_pre: create_f32(client, &g.y_pre),
                            norm_r: create_f32(client, &g.norm_r),
                            gate_s: create_f32(client, &g.gate_s),
                            gate_sp: create_f32(client, &g.gate_sp),
                            normed: create_f32(client, &g.normed),
                        })
                    }
                    riir_infer_core::lens::stack::SubCapture::Attn(a) => {
                        SubCaptureHandles::Attn(AttnCaptureHandles {
                            q_pre: create_f32(client, &a.q_pre),
                            k_pre: create_f32(client, &a.k_pre),
                            q_r: create_f32(client, &a.q_r),
                            k_r: create_f32(client, &a.k_r),
                            q: create_f32(client, &a.q),
                            k: create_f32(client, &a.k),
                            v: create_f32(client, &a.v),
                            attn: create_f32(client, &a.attn),
                            att: create_f32(client, &a.att),
                            gate_sg: create_f32(client, &a.gate_sg),
                            gate_spg: create_f32(client, &a.gate_spg),
                        })
                    }
                    riir_infer_core::lens::stack::SubCapture::None => {
                        unreachable!("capture pass fills every sublayer")
                    }
                };
                LayerCaptureHandles {
                    sub,
                    input_scale: create_f32(client, &lc.input_scale),
                    post_scale: create_f32(client, &lc.post_scale),
                    mlp_s: create_f32(client, &lc.mlp_s),
                    mlp_u: create_f32(client, &lc.mlp_u),
                    mlp_sp: create_f32(client, &lc.mlp_sp),
                }
            })
            .collect();
        Self {
            layers,
            final_scale: create_f32(client, &cap.final_scale),
        }
    }
}

// ---------------------------------------------------------------------------
// The batched suffix pass
// ---------------------------------------------------------------------------

/// Run ONE suffix pass for `point` over the resident direction block: the
/// injected tangent `dv` `[batch·t_len·d]` (uploaded by the caller — the
/// SAME block serves every point, so upload once per doc per block) is
/// propagated from the point through the remaining stack, and the final
/// post-norm tangent's rows `[row_lo, row_hi)` are summed per direction.
///
/// Returns `[batch · d]` f32 (the accumulator's `add_doc_column` unit).
///
/// # Safety
///
/// All buffers are sized by [`LensBatchStack::new`] +
/// [`LensDocCapture::upload`] against the same dims — the launches assert
/// bindings in debug.
pub unsafe fn suffix_pass<R: Runtime>(
    client: &ComputeClient<R>,
    stack: &LensBatchStack,
    cap: &LensDocCapture,
    dv: &Handle,
    point: FitPoint,
    row_lo: usize,
    row_hi: usize,
) -> Result<Vec<f32>, cubecl::server::ServerError> {
    let dims = &stack.dims;
    let (batch, t_len, d) = (dims.batch, dims.t_len, dims.n_embd);
    let td_b = batch * t_len * d;
    // The working tangent: a device copy of dv (the residual adds
    // accumulate in place from here).
    let t = client.empty(td_b * std::mem::size_of::<f32>());
    unsafe {
        crate::elementwise_cubecl::CopyCubeCL::launch(client, dv.clone(), t.clone(), td_b);
    }
    match point {
        FitPoint::Out(l) => {
            for li in (l + 1)..dims.n_layers {
                unsafe {
                    layer_step(client, stack, cap, li, &t)?;
                }
            }
        }
        FitPoint::Mid(l) => {
            // The write enters layer l's post-norm: the MLP half only.
            unsafe {
                mlp_step(client, stack, cap, l, &t)?;
            }
            for li in (l + 1)..dims.n_layers {
                unsafe {
                    layer_step(client, stack, cap, li, &t)?;
                }
            }
        }
    }
    // Final norm (LN-rule diag, self-adjoint) — into t2, then reduce.
    unsafe {
        LensScaleTableCL::launch(
            client,
            t.clone(),
            cap.final_scale.clone(),
            stack.t2.clone(),
            td_b,
            t_len * d,
        );
    }
    unsafe {
        LensColSumCL::launch(
            client,
            stack.t2.clone(),
            stack.colsum.clone(),
            batch,
            t_len,
            d,
            row_lo,
            row_hi,
        );
    }
    read_f32(client, stack.colsum.clone())
}

/// One layer's jvp step on the batched tangent `t` (input-norm diag →
/// sublayer → residual → MLP half rule → residual) — the twin of
/// `stack::layer_jvp_step_simple`.
#[cfg(feature = "cubecl_runtime")]
/// # Safety
///
/// Kernel-launch contract: `t` is a `batch·t_len·d` handle; the capture and
/// stack handles were uploaded against the same dims.
unsafe fn layer_step<R: Runtime>(
    client: &ComputeClient<R>,
    stack: &LensBatchStack,
    cap: &LensDocCapture,
    li: usize,
    t: &Handle,
) -> Result<(), cubecl::server::ServerError> {
    let dims = &stack.dims;
    let (batch, t_len, d) = (dims.batch, dims.t_len, dims.n_embd);
    let td_b = batch * t_len * d;
    let lc = &cap.layers[li];
    // Input-norm diag (in place — the scale kernel's read/write indices are
    // identical, no cross-thread hazard).
    unsafe {
        LensScaleTableCL::launch(
            client,
            t.clone(),
            lc.input_scale.clone(),
            t.clone(),
            td_b,
            t_len * d,
        );
    }
    unsafe {
        sublayer_step(client, stack, cap, li, t)?;
    }
    unsafe {
        LensAddInplaceCL::launch(client, t.clone(), stack.t2.clone(), td_b);
    }
    unsafe {
        mlp_step(client, stack, cap, li, t)?;
    }
    Ok(())
}

/// The MLP half of one layer's step (post-norm diag → gate/up GEMMs → half
/// rule → down GEMM → residual add) — the twin of `stack::mlp_jvp` + the
/// caller's residual add.
#[cfg(feature = "cubecl_runtime")]
/// # Safety
///
/// Kernel-launch contract: `t` is a `batch·t_len·d` handle; the capture and
/// stack handles were uploaded against the same dims.
unsafe fn mlp_step<R: Runtime>(
    client: &ComputeClient<R>,
    stack: &LensBatchStack,
    cap: &LensDocCapture,
    li: usize,
    t: &Handle,
) -> Result<(), cubecl::server::ServerError> {
    let dims = &stack.dims;
    let (batch, t_len, d, hidden) = (dims.batch, dims.t_len, dims.n_embd, dims.mlp_hidden);
    let td_b = batch * t_len * d;
    let th_b = batch * t_len * hidden;
    let lc = &cap.layers[li];
    let lw = &stack.layers[li];
    // t2 = t · post_scale (t unchanged — the residual's pre-norm tangent).
    unsafe {
        LensScaleTableCL::launch(
            client,
            t.clone(),
            lc.post_scale.clone(),
            stack.t2.clone(),
            td_b,
            t_len * d,
        );
    }
    unsafe {
        LensGemmCL::launch(
            client,
            stack.t2.clone(),
            lw.w_gate.clone(),
            stack.dg.clone(),
            batch * t_len,
            hidden,
            d,
        );
    }
    unsafe {
        LensGemmCL::launch(
            client,
            stack.t2.clone(),
            lw.w_up.clone(),
            stack.du.clone(),
            batch * t_len,
            hidden,
            d,
        );
    }
    // dact = ½(s·du + u·sp·dg).
    unsafe {
        LensHalfRule2CL::launch(
            client,
            lc.mlp_s.clone(),
            lc.mlp_u.clone(),
            lc.mlp_sp.clone(),
            stack.du.clone(),
            stack.dg.clone(),
            stack.dact.clone(),
            th_b,
            t_len * hidden,
        );
    }
    unsafe {
        LensGemmCL::launch(
            client,
            stack.dact.clone(),
            lw.w_down.clone(),
            stack.dmlp.clone(),
            batch * t_len,
            d,
            hidden,
        );
    }
    unsafe {
        LensAddInplaceCL::launch(client, t.clone(), stack.dmlp.clone(), td_b);
    }
    Ok(())
}

/// The sublayer half of one layer's step (GDN or attention) — writes the
/// sublayer output tangent into `stack.t2`.
#[cfg(feature = "cubecl_runtime")]
/// # Safety
///
/// Kernel-launch contract: `t` is a `batch·t_len·d` handle; the capture and
/// stack handles were uploaded against the same dims.
unsafe fn sublayer_step<R: Runtime>(
    client: &ComputeClient<R>,
    stack: &LensBatchStack,
    cap: &LensDocCapture,
    li: usize,
    t: &Handle,
) -> Result<(), cubecl::server::ServerError> {
    let dims = &stack.dims;
    let (batch, t_len, d) = (dims.batch, dims.t_len, dims.n_embd);
    let p_rows = batch * t_len;
    let lc = &cap.layers[li];
    let lw = &stack.layers[li];
    match (&lc.sub, &lw.sub) {
        (SubCaptureHandles::Gdn(gc), SubWeightHandles::Gdn(gw)) => {
            let gdn = dims.gdn.as_ref().expect("gdn dims");
            let gsc = stack.gdn.as_ref().expect("gdn scratch");
            let (n_k, n_v, key, val) = (gdn.n_k_heads, gdn.n_v_heads, gdn.key_dim, gdn.val_dim);
            let conv_dim = gdn.conv_dim();
            let q_dim = gdn.q_dim();
            let v_dim = gdn.v_dim();
            let ptd_n_v = batch * t_len * n_v;
            // 1. Projections.
            unsafe {
                LensGemmCL::launch(
                    client,
                    t.clone(),
                    gw.w_qkv.clone(),
                    gsc.dqkv_raw.clone(),
                    p_rows,
                    conv_dim,
                    d,
                );
            }
            unsafe {
                LensGemmCL::launch(
                    client,
                    t.clone(),
                    gw.w_z.clone(),
                    gsc.dz.clone(),
                    p_rows,
                    v_dim,
                    d,
                );
            }
            unsafe {
                LensGemmCL::launch(
                    client,
                    t.clone(),
                    gw.w_a.clone(),
                    gsc.da.clone(),
                    p_rows,
                    n_v,
                    d,
                );
            }
            unsafe {
                LensGemmCL::launch(
                    client,
                    t.clone(),
                    gw.w_b.clone(),
                    gsc.db.clone(),
                    p_rows,
                    n_v,
                    d,
                );
            }
            // 2. Causal conv + silu diagonal.
            unsafe {
                LensConv1dCL::launch(
                    client,
                    gsc.dqkv_raw.clone(),
                    gw.conv_w.clone(),
                    gsc.dconv.clone(),
                    batch,
                    t_len,
                    conv_dim,
                    gdn.kernel_size,
                );
            }
            unsafe {
                LensScaleTableCL::launch(
                    client,
                    gsc.dconv.clone(),
                    gc.conv_silu_grad.clone(),
                    gsc.dqkv.clone(),
                    batch * t_len * conv_dim,
                    t_len * conv_dim,
                );
            }
            // Split the conv row into (dq, dk, dv).
            unsafe {
                LensCopyRowSliceCL::launch(
                    client,
                    gsc.dqkv.clone(),
                    gsc.dq.clone(),
                    batch,
                    t_len,
                    conv_dim,
                    q_dim,
                    0,
                );
            }
            unsafe {
                LensCopyRowSliceCL::launch(
                    client,
                    gsc.dqkv.clone(),
                    gsc.dk.clone(),
                    batch,
                    t_len,
                    conv_dim,
                    q_dim,
                    q_dim,
                );
            }
            unsafe {
                LensCopyRowSliceCL::launch(
                    client,
                    gsc.dqkv.clone(),
                    gsc.dv.clone(),
                    batch,
                    t_len,
                    conv_dim,
                    v_dim,
                    2 * q_dim,
                );
            }
            // 3. Gate tangent coefficients.
            unsafe {
                LensScaleTableCL::launch(
                    client,
                    gsc.db.clone(),
                    gc.dbeta_coef.clone(),
                    gsc.dbeta.clone(),
                    ptd_n_v,
                    t_len * n_v,
                );
            }
            unsafe {
                LensScaleTableCL::launch(
                    client,
                    gsc.da.clone(),
                    gc.ddecay_coef.clone(),
                    gsc.ddecay.clone(),
                    ptd_n_v,
                    t_len * n_v,
                );
            }
            // 4. L2 norm + expand (q, k).
            unsafe {
                LensL2NormExpandCL::launch(
                    client,
                    gsc.dq.clone(),
                    gc.qn.clone(),
                    gc.qn_inv.clone(),
                    gsc.dqn.clone(),
                    batch,
                    t_len,
                    n_k,
                    n_v / n_k,
                    key,
                    n_v * key,
                );
            }
            unsafe {
                LensL2NormExpandCL::launch(
                    client,
                    gsc.dk.clone(),
                    gc.kn.clone(),
                    gc.kn_inv.clone(),
                    gsc.dkn.clone(),
                    batch,
                    t_len,
                    n_k,
                    n_v / n_k,
                    key,
                    n_v * key,
                );
            }
            // 5-6. Recurrence scan + ssm norm (the scan consumes the
            // EXPANDED dq/dk — the CPU twin passes dqn/dkn).
            unsafe {
                LensGdnScanCL::launch(
                    client,
                    gc.qn.clone(),
                    gc.kn.clone(),
                    gc.v.clone(),
                    gc.beta.clone(),
                    gc.decay.clone(),
                    gsc.dbeta.clone(),
                    gsc.ddecay.clone(),
                    gsc.dqn.clone(),
                    gsc.dkn.clone(),
                    gsc.dv.clone(),
                    gc.states.clone(),
                    gsc.ds.clone(),
                    gsc.spre.clone(),
                    gsc.dy_pre.clone(),
                    batch,
                    t_len,
                    n_v,
                    key,
                    val,
                    gdn.scale(),
                );
            }
            unsafe {
                LensQkNormCL::launch(
                    client,
                    gc.y_pre.clone(),
                    gc.norm_r.clone(),
                    gw.linear_norm.clone(),
                    gsc.dy_pre.clone(),
                    gsc.dnormed.clone(),
                    batch,
                    t_len,
                    n_v,
                    val,
                );
            }
            // 7-8. silu(z) gate half rule + out projection:
            // dact = ½(s·du + u·sp·dg) with (dg, du) = (dz, dnormed)
            // → ½(gate_s·dnormed + normed·gate_sp·dz). The x/y argument
            // order matters — x pairs with the a-table (the value branch).
            unsafe {
                LensHalfRule2CL::launch(
                    client,
                    gc.gate_s.clone(),
                    gc.normed.clone(),
                    gc.gate_sp.clone(),
                    gsc.dnormed.clone(),
                    gsc.dz.clone(),
                    gsc.dout.clone(),
                    batch * t_len * v_dim,
                    t_len * v_dim,
                );
            }
            unsafe {
                LensGemmCL::launch(
                    client,
                    gsc.dout.clone(),
                    gw.w_out.clone(),
                    stack.t2.clone(),
                    p_rows,
                    d,
                    v_dim,
                );
            }
            Ok(())
        }
        (SubCaptureHandles::Attn(ac), SubWeightHandles::Attn(aw)) => {
            let attn = dims.attn.as_ref().expect("attn dims");
            let asc = stack.attn.as_ref().expect("attn scratch");
            let (n_head, n_kv, hd) = (attn.n_head, attn.n_kv, attn.head_dim);
            let q_dim = attn.q_dim();
            let kv_dim = attn.kv_dim();
            let qg_dim = attn.qg_dim();
            let rotary_dim = attn.rotary_dim;
            // 1. Gated Q / K / V projection tangents + interleave split.
            unsafe {
                LensGemmCL::launch(
                    client,
                    t.clone(),
                    aw.w_qg.clone(),
                    asc.dqg.clone(),
                    p_rows,
                    qg_dim,
                    d,
                );
            }
            unsafe {
                LensGemmCL::launch(
                    client,
                    t.clone(),
                    aw.w_v.clone(),
                    asc.dv.clone(),
                    p_rows,
                    kv_dim,
                    d,
                );
            }
            unsafe {
                LensGemmCL::launch(
                    client,
                    t.clone(),
                    aw.w_k.clone(),
                    asc.dk.clone(),
                    p_rows,
                    kv_dim,
                    d,
                );
            }
            unsafe {
                LensQgSplitCL::launch(
                    client,
                    asc.dqg.clone(),
                    asc.dq.clone(),
                    asc.dgate.clone(),
                    batch,
                    t_len,
                    n_head,
                    hd,
                );
            }
            // 2. QK-norm tangents (true derivative).
            unsafe {
                LensQkNormCL::launch(
                    client,
                    ac.q_pre.clone(),
                    ac.q_r.clone(),
                    aw.q_norm.clone(),
                    asc.dq.clone(),
                    asc.dq2.clone(),
                    batch,
                    t_len,
                    n_head,
                    hd,
                );
            }
            unsafe {
                LensQkNormCL::launch(
                    client,
                    ac.k_pre.clone(),
                    ac.k_r.clone(),
                    aw.k_norm.clone(),
                    asc.dk.clone(),
                    asc.dk2.clone(),
                    batch,
                    t_len,
                    n_kv,
                    hd,
                );
            }
            // 3. RoPE (in place).
            unsafe {
                LensRopeJvpCL::launch(
                    client,
                    asc.dq2.clone(),
                    asc.dk2.clone(),
                    aw.rope_cs.clone(),
                    batch,
                    t_len,
                    n_head,
                    n_kv,
                    hd,
                    rotary_dim,
                );
            }
            // 5. Softmax attention (true derivative).
            unsafe {
                LensAttnJvpCL::launch(
                    client,
                    ac.q.clone(),
                    ac.k.clone(),
                    ac.v.clone(),
                    ac.attn.clone(),
                    asc.dq2.clone(),
                    asc.dk2.clone(),
                    asc.dv.clone(),
                    asc.dgated.clone(),
                    batch,
                    t_len,
                    n_head,
                    n_kv,
                    hd,
                    attn.scale(),
                );
            }
            // 6. Output gate (half rule): dy = ½(sg·dz + z·spg·dg) with
            // (dg, dz) = (dgate, dgated) — note the swapped argument order
            // vs the MLP arm (the gate's value branch is `att`, not
            // `normed`).
            unsafe {
                LensHalfRule2CL::launch(
                    client,
                    ac.gate_sg.clone(),
                    ac.att.clone(),
                    ac.gate_spg.clone(),
                    asc.dgated.clone(),
                    asc.dgate.clone(),
                    asc.dgated2.clone(),
                    batch * t_len * q_dim,
                    t_len * q_dim,
                );
            }
            // 7. o_proj.
            unsafe {
                LensGemmCL::launch(
                    client,
                    asc.dgated2.clone(),
                    aw.w_o.clone(),
                    stack.t2.clone(),
                    p_rows,
                    d,
                    q_dim,
                );
            }
            Ok(())
        }
        _ => unreachable!("sub capture/weights kind mismatch"),
    }
}

/// The per-position `(cos, sin)` table for the attention rope: `[T, 2·half]`
/// with `cos` at `[0, half)` and `sin` at `[half, 2·half)` — `half =
/// rotary_dim / 2` (the partial rope's pairing stride; only the first
/// `half` freq entries participate). The exact host-side values
/// `rope::apply_*_rope_with_freq`'s phase-1 precompute makes.
fn rope_cs_table(freq: &[f32], t_len: usize, rotary_dim: usize) -> Vec<f32> {
    let half = rotary_dim / 2;
    debug_assert!(
        freq.len() >= half,
        "rope_cs_table: freq shorter than rotary_dim/2"
    );
    let mut out = vec![0.0f32; t_len * 2 * half];
    for t in 0..t_len {
        let pos_f = t as f32;
        let base = t * 2 * half;
        for (i, &f) in freq[..half].iter().enumerate() {
            let angle = pos_f * f;
            let (sin_a, cos_a) = angle.sin_cos();
            out[base + i] = cos_a;
            out[base + half + i] = sin_a;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Debug bisect harness (doc-hidden; the parity test's surgical probe —
// runs ONE layer step on `t` and reads the working tangent back).
// ---------------------------------------------------------------------------

/// Run one layer's full jvp step on the working tangent `t` (in place) and
/// read the result back. Doc-hidden debug harness for the parity test's
/// per-layer bisect — NOT part of the driver's stable surface.
#[doc(hidden)]
#[cfg(feature = "cubecl_runtime")]
pub unsafe fn debug_layer_step_readback<R: Runtime>(
    client: &ComputeClient<R>,
    stack: &LensBatchStack,
    cap: &LensDocCapture,
    li: usize,
    t: &Handle,
) -> Result<Vec<f32>, cubecl::server::ServerError> {
    unsafe {
        layer_step(client, stack, cap, li, t)?;
    }
    read_f32(client, t.clone())
}

/// Doc-hidden: run ONLY the sublayer half of layer `li`'s step on `t`
/// (in place up to the residual add — the sublayer output lands in
/// `stack.t2`) and read `t2` back.
#[doc(hidden)]
#[cfg(feature = "cubecl_runtime")]
pub unsafe fn debug_sublayer_readback<R: Runtime>(
    client: &ComputeClient<R>,
    stack: &LensBatchStack,
    cap: &LensDocCapture,
    li: usize,
    t: &Handle,
) -> Result<Vec<f32>, cubecl::server::ServerError> {
    unsafe {
        sublayer_step(client, stack, cap, li, t)?;
    }
    read_f32(client, stack.t2.clone())
}

/// Doc-hidden bisect probe: run the attention arm's pipeline for layer
/// `li` on `t` and read back the buffer named by `stage`:
/// 0 = `dqg` (after the w_qg GEMM), 1 = `dq` (after the qg split),
/// 2 = `dq2` (after Q qk-norm), 3 = `dk2` (after K qk-norm),
/// 4 = `dq2` (after RoPE), 5 = `dgated` (after the attention jvp),
/// 6 = `dgated2` (after the gate half rule).
#[doc(hidden)]
#[cfg(feature = "cubecl_runtime")]
pub unsafe fn debug_attn_stage<R: Runtime>(
    client: &ComputeClient<R>,
    stack: &LensBatchStack,
    cap: &LensDocCapture,
    li: usize,
    t: &Handle,
    stage: usize,
) -> Result<Vec<f32>, cubecl::server::ServerError> {
    let dims = &stack.dims;
    let (batch, t_len, d) = (dims.batch, dims.t_len, dims.n_embd);
    let p_rows = batch * t_len;
    let lc = &cap.layers[li];
    let lw = &stack.layers[li];
    let (SubCaptureHandles::Attn(ac), SubWeightHandles::Attn(aw)) = (&lc.sub, &lw.sub) else {
        panic!("debug_attn_stage: not an attention layer");
    };
    let attn = dims.attn.as_ref().expect("attn dims");
    let asc = stack.attn.as_ref().expect("attn scratch");
    let (n_head, n_kv, hd) = (attn.n_head, attn.n_kv, attn.head_dim);
    let q_dim = attn.q_dim();
    let kv_dim = attn.kv_dim();
    let qg_dim = attn.qg_dim();
    // Stages 1-6 need the dqg GEMM first; stage 0 reads it directly.
    unsafe {
        LensGemmCL::launch(
            client,
            t.clone(),
            aw.w_qg.clone(),
            asc.dqg.clone(),
            p_rows,
            qg_dim,
            d,
        );
    }
    pollster::block_on(client.sync()).ok();
    if stage == 0 {
        return read_f32(client, asc.dqg.clone());
    }
    unsafe {
        LensGemmCL::launch(
            client,
            t.clone(),
            aw.w_k.clone(),
            asc.dk.clone(),
            p_rows,
            kv_dim,
            d,
        );
    }
    unsafe {
        LensGemmCL::launch(
            client,
            t.clone(),
            aw.w_v.clone(),
            asc.dv.clone(),
            p_rows,
            kv_dim,
            d,
        );
    }
    unsafe {
        LensQgSplitCL::launch(
            client,
            asc.dqg.clone(),
            asc.dq.clone(),
            asc.dgate.clone(),
            batch,
            t_len,
            n_head,
            hd,
        );
    }
    pollster::block_on(client.sync()).ok();
    if stage == 1 {
        return read_f32(client, asc.dq.clone());
    }
    if stage == 7 {
        return read_f32(client, asc.dk.clone());
    }
    unsafe {
        LensQkNormCL::launch(
            client,
            ac.q_pre.clone(),
            ac.q_r.clone(),
            aw.q_norm.clone(),
            asc.dq.clone(),
            asc.dq2.clone(),
            batch,
            t_len,
            n_head,
            hd,
        );
    }
    pollster::block_on(client.sync()).ok();
    if stage == 2 {
        // Force a read-barrier on the INPUT handle first (read_one is the
        // documented sync point; sync() alone has proven insufficient for
        // kernel→kernel dependency tracking on this vendored runtime).
        let _ = read_f32(client, asc.dq.clone());
        let first = read_f32(client, asc.dq2.clone())?;
        let second = read_f32(client, asc.dq2.clone())?;
        if first != second {
            println!("  [debug] dq2 read1 != read2 — nondeterministic readback");
        }
        return Ok(first);
    }
    unsafe {
        LensQkNormCL::launch(
            client,
            ac.k_pre.clone(),
            ac.k_r.clone(),
            aw.k_norm.clone(),
            asc.dk.clone(),
            asc.dk2.clone(),
            batch,
            t_len,
            n_kv,
            hd,
        );
    }
    pollster::block_on(client.sync()).ok();
    if stage == 3 {
        return read_f32(client, asc.dk2.clone());
    }
    unsafe {
        LensRopeJvpCL::launch(
            client,
            asc.dq2.clone(),
            asc.dk2.clone(),
            aw.rope_cs.clone(),
            batch,
            t_len,
            n_head,
            n_kv,
            hd,
            attn.rotary_dim,
        );
    }
    pollster::block_on(client.sync()).ok();
    if stage == 4 {
        return read_f32(client, asc.dq2.clone());
    }
    pollster::block_on(client.sync()).ok();
    if stage == 5 {
        // The kernel consumes dk2 too — rerun rope on a fresh copy first
        // is unnecessary; rope is idempotent-DESTRUCTIVE (in place), so the
        // stage-5 caller must accept dk2 also advanced. Run the jvp now.
        unsafe {
            LensAttnJvpCL::launch(
                client,
                ac.q.clone(),
                ac.k.clone(),
                ac.v.clone(),
                ac.attn.clone(),
                asc.dq2.clone(),
                asc.dk2.clone(),
                asc.dv.clone(),
                asc.dgated.clone(),
                batch,
                t_len,
                n_head,
                n_kv,
                hd,
                attn.scale(),
            );
        }
        return read_f32(client, asc.dgated.clone());
    }
    unsafe {
        LensAttnJvpCL::launch(
            client,
            ac.q.clone(),
            ac.k.clone(),
            ac.v.clone(),
            ac.attn.clone(),
            asc.dq2.clone(),
            asc.dk2.clone(),
            asc.dv.clone(),
            asc.dgated.clone(),
            batch,
            t_len,
            n_head,
            n_kv,
            hd,
            attn.scale(),
        );
    }
    unsafe {
        LensHalfRule2CL::launch(
            client,
            ac.gate_sg.clone(),
            ac.att.clone(),
            ac.gate_spg.clone(),
            asc.dgated.clone(),
            asc.dgate.clone(),
            asc.dgated2.clone(),
            batch * t_len * q_dim,
            t_len * q_dim,
        );
    }
    let _ = t_len * q_dim;
    pollster::block_on(client.sync()).ok();
    read_f32(client, asc.dgated2.clone())
}

/// Doc-hidden bisect probe (GDN arm): run layer `li`'s GDN pipeline up to
/// `stage` and read the named buffer:
/// 0 = `dz` (w_z GEMM), 1 = `da` (w_a GEMM), 2 = `dqkv` (conv+silu diag),
/// 3 = `dq` (split), 4 = `dk` (split), 5 = `dv` (split),
/// 6 = `dbeta`, 7 = `ddecay`, 8 = `dqn` (l2norm+expand),
/// 9 = `dkn`, 10 = `dy_pre` (scan).
#[doc(hidden)]
#[cfg(feature = "cubecl_runtime")]
pub unsafe fn debug_gdn_stage<R: Runtime>(
    client: &ComputeClient<R>,
    stack: &LensBatchStack,
    cap: &LensDocCapture,
    li: usize,
    t: &Handle,
    stage: usize,
) -> Result<Vec<f32>, cubecl::server::ServerError> {
    let dims = &stack.dims;
    let (batch, t_len, d) = (dims.batch, dims.t_len, dims.n_embd);
    let p_rows = batch * t_len;
    let lc = &cap.layers[li];
    let lw = &stack.layers[li];
    let (SubCaptureHandles::Gdn(gc), SubWeightHandles::Gdn(gw)) = (&lc.sub, &lw.sub) else {
        panic!("debug_gdn_stage: not a GDN layer");
    };
    let gdn = dims.gdn.as_ref().expect("gdn dims");
    let gsc = stack.gdn.as_ref().expect("gdn scratch");
    let (n_k, n_v, key, val) = (gdn.n_k_heads, gdn.n_v_heads, gdn.key_dim, gdn.val_dim);
    let conv_dim = gdn.conv_dim();
    let q_dim = gdn.q_dim();
    let v_dim = gdn.v_dim();
    let ptd_n_v = batch * t_len * n_v;
    unsafe {
        LensGemmCL::launch(
            client,
            t.clone(),
            gw.w_qkv.clone(),
            gsc.dqkv_raw.clone(),
            p_rows,
            conv_dim,
            d,
        );
    }
    unsafe {
        LensGemmCL::launch(
            client,
            t.clone(),
            gw.w_z.clone(),
            gsc.dz.clone(),
            p_rows,
            v_dim,
            d,
        );
    }
    unsafe {
        LensGemmCL::launch(
            client,
            t.clone(),
            gw.w_a.clone(),
            gsc.da.clone(),
            p_rows,
            n_v,
            d,
        );
    }
    unsafe {
        LensGemmCL::launch(
            client,
            t.clone(),
            gw.w_b.clone(),
            gsc.db.clone(),
            p_rows,
            n_v,
            d,
        );
    }
    match stage {
        0 => return read_f32(client, gsc.dz.clone()),
        1 => return read_f32(client, gsc.da.clone()),
        _ => {}
    }
    unsafe {
        LensConv1dCL::launch(
            client,
            gsc.dqkv_raw.clone(),
            gw.conv_w.clone(),
            gsc.dconv.clone(),
            batch,
            t_len,
            conv_dim,
            gdn.kernel_size,
        );
    }
    unsafe {
        LensScaleTableCL::launch(
            client,
            gsc.dconv.clone(),
            gc.conv_silu_grad.clone(),
            gsc.dqkv.clone(),
            batch * t_len * conv_dim,
            t_len * conv_dim,
        );
    }
    unsafe {
        LensCopyRowSliceCL::launch(
            client,
            gsc.dqkv.clone(),
            gsc.dq.clone(),
            batch,
            t_len,
            conv_dim,
            q_dim,
            0,
        );
    }
    unsafe {
        LensCopyRowSliceCL::launch(
            client,
            gsc.dqkv.clone(),
            gsc.dk.clone(),
            batch,
            t_len,
            conv_dim,
            q_dim,
            q_dim,
        );
    }
    unsafe {
        LensCopyRowSliceCL::launch(
            client,
            gsc.dqkv.clone(),
            gsc.dv.clone(),
            batch,
            t_len,
            conv_dim,
            v_dim,
            2 * q_dim,
        );
    }
    match stage {
        2 => return read_f32(client, gsc.dqkv.clone()),
        3 => return read_f32(client, gsc.dq.clone()),
        4 => return read_f32(client, gsc.dk.clone()),
        5 => return read_f32(client, gsc.dv.clone()),
        _ => {}
    }
    unsafe {
        LensScaleTableCL::launch(
            client,
            gsc.db.clone(),
            gc.dbeta_coef.clone(),
            gsc.dbeta.clone(),
            ptd_n_v,
            t_len * n_v,
        );
    }
    unsafe {
        LensScaleTableCL::launch(
            client,
            gsc.da.clone(),
            gc.ddecay_coef.clone(),
            gsc.ddecay.clone(),
            ptd_n_v,
            t_len * n_v,
        );
    }
    match stage {
        6 => return read_f32(client, gsc.dbeta.clone()),
        7 => return read_f32(client, gsc.ddecay.clone()),
        _ => {}
    }
    unsafe {
        LensL2NormExpandCL::launch(
            client,
            gsc.dq.clone(),
            gc.qn.clone(),
            gc.qn_inv.clone(),
            gsc.dqn.clone(),
            batch,
            t_len,
            n_k,
            n_v / n_k,
            key,
            n_v * key,
        );
    }
    unsafe {
        LensL2NormExpandCL::launch(
            client,
            gsc.dk.clone(),
            gc.kn.clone(),
            gc.kn_inv.clone(),
            gsc.dkn.clone(),
            batch,
            t_len,
            n_k,
            n_v / n_k,
            key,
            n_v * key,
        );
    }
    match stage {
        8 => return read_f32(client, gsc.dqn.clone()),
        9 => return read_f32(client, gsc.dkn.clone()),
        _ => {}
    }
    unsafe {
        LensGdnScanCL::launch(
            client,
            gc.qn.clone(),
            gc.kn.clone(),
            gc.v.clone(),
            gc.beta.clone(),
            gc.decay.clone(),
            gsc.dbeta.clone(),
            gsc.ddecay.clone(),
            gsc.dqn.clone(),
            gsc.dkn.clone(),
            gsc.dv.clone(),
            gc.states.clone(),
            gsc.ds.clone(),
            gsc.spre.clone(),
            gsc.dy_pre.clone(),
            batch,
            t_len,
            n_v,
            key,
            val,
            gdn.scale(),
        );
    }
    debug_assert_eq!(stage, 10, "debug_gdn_stage: stage out of range");
    read_f32(client, gsc.dy_pre.clone())
}

/// Doc-hidden: run layer `li`'s MLP half on `t` (in place) and read back.
#[doc(hidden)]
#[cfg(feature = "cubecl_runtime")]
pub unsafe fn debug_mlp_readback<R: Runtime>(
    client: &ComputeClient<R>,
    stack: &LensBatchStack,
    cap: &LensDocCapture,
    li: usize,
    t: &Handle,
) -> Result<Vec<f32>, cubecl::server::ServerError> {
    unsafe {
        mlp_step(client, stack, cap, li, t)?;
    }
    read_f32(client, t.clone())
}
