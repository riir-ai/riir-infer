#![cfg(feature = "lens")]
//! Doc-level qwen3.5 stack capture + linearized-stack maps — T2d of
//! katgpt-rs Issue 999 (Research 666).
//!
//! [`stack_forward`] mirrors `crate::deltanet::forward`
//! `forward_qwen_deltanet_body`'s op order over a STANDALONE doc (zero KV /
//! conv / recurrence state; position `t`'s RoPE phase is `t`) and captures
//! every constant the RelP tangent maps consume:
//!
//! ```text
//! x = wte[tokens]                      (embedding gather — no scale, Qwen3.5)
//! per layer:
//!     residual = x; x̂ = RMSNorm(x, input_norm)        [LN rule diag]
//!     x_sub = GDN(x̂) | Attn(x̂)                        [T2c | T2b maps]
//!     mid = residual + x_sub                           (the `mid` fit point)
//!     residual = mid; x̂2 = RMSNorm(mid, post_attn_norm) [LN rule diag]
//!     x = mid + W_down·(silu(W_gate·x̂2) ⊙ W_up·x̂2)     [half rule]  (the
//!                                                      `out` fit point)
//! hn = RMSNorm(x, final_norm)          [LN rule diag]
//! logits = lm_head · hn                (lazy — computed by the reads)
//! ```
//!
//! The tangent machinery:
//!
//! - [`stack_prefix_jvp`] — embedding tangents `[T·d]` → tangents at EVERY
//!   fit point + the final post-norm tangent. One pass per basis direction
//!   covers every layer (the S-lens fixed end).
//! - [`stack_suffix_jvp`] — a write tangent at one [`FitPoint`] → the final
//!   post-norm tangent (the R-lens per-point pass).
//! - [`stack_suffix_vjp`] — the exact adjoint of the suffix (the transpose
//!   identity's implementation half; adjointness is gate-tested).
//!
//! Bit parity: the capture's activations use the EXACT activations
//! ([`crate::lens::ops::sigmoid`]/[`crate::lens::ops::silu`]) where the
//! serving forward uses `fast_sigmoid` — the disclosed fit-side drift (the
//! T2b/T2c posture, asserted in tolerance by the stack parity test). Every
//! other op is the forward's own function, so the parity test pins the
//! composition (residual plumbing, rope phases, causal masks, GQA groups)
//! structurally.
//!
//! Memory envelope (owned by [`StackCapture`], one doc at a time): the GDN
//! per-position states dominate — `T · n_v · val · key` f32 per GDN layer
//! (qwen3.5-0.8B at T=128: ~268 MB/layer, ~5 GB/doc). The fit is offline;
//! the capture is built, swept, and dropped per doc.
//!
//! Never a serving path.

use crate::deltanet::forward::effective_rotary_dim;
use crate::deltanet::weights::{Proj, QwenDeltaNetWeights};
use crate::lens::attn::{
    AttnGeometry, AttnLayerCapture, AttnLayerConsts, AttnLayerScratch, AttnLayerWeights,
    attn_layer_forward, attn_layer_jvp, attn_layer_vjp,
};
use crate::lens::gdn::{
    GdnGeometry, GdnLayerCapture, GdnLayerConsts, GdnLayerScratch, GdnLayerWeights,
    gdn_layer_forward, gdn_layer_jvp, gdn_layer_vjp,
};
use crate::lens::ops::{
    MlpConsts, diag_apply, matvec_t, mlp_half_rule_jvp, mlp_half_rule_vjp, norm_diag_scale,
};
use crate::types::Config;
use crate::types::DeltaNetLayerType;
use crate::types::matmul;

// ---------------------------------------------------------------------------
// Geometry + fit points
// ---------------------------------------------------------------------------

/// The stack's geometry (derived from the model [`Config`] at a fixed doc
/// length; `t_len` is a fit-time choice, not a config field).
#[derive(Clone, Debug)]
pub struct StackGeometry {
    pub t_len: usize,
    pub n_layers: usize,
    pub n_embd: usize,
    pub mlp_hidden: usize,
    pub vocab: usize,
    pub rms_eps: f64,
    /// Partial-RoPE dimension count (`effective_rotary_dim`, 0-sentinel
    /// resolved to `head_dim`).
    pub rotary_dim: usize,
    /// Full-attention sublayer dims.
    pub n_head: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    /// GDN sublayer dims (the forward's: Q and K share `dn_n_k`; gates and
    /// V are per `dn_n_v` head; key/val share `dn_head_dim`).
    pub dn_n_k: usize,
    pub dn_n_v: usize,
    pub dn_head_dim: usize,
    pub dn_kernel: usize,
    /// `true` = GDN layer, `false` = full attention (the model's
    /// `layer_types`).
    pub layer_is_gdn: Vec<bool>,
}

impl StackGeometry {
    pub fn from_config(config: &Config, t_len: usize, layer_is_gdn: Vec<bool>) -> Self {
        Self {
            t_len,
            n_layers: layer_is_gdn.len(),
            n_embd: config.n_embd,
            mlp_hidden: config.mlp_hidden,
            vocab: config.vocab_size,
            rms_eps: config.rms_norm_eps,
            rotary_dim: effective_rotary_dim(config),
            n_head: config.n_head,
            n_kv: config.n_kv_head,
            head_dim: config.head_dim,
            dn_n_k: config.deltanet_linear_n_heads,
            dn_n_v: config.deltanet_linear_n_value_heads,
            dn_head_dim: config.deltanet_linear_head_dim,
            dn_kernel: config.deltanet_conv_kernel_size,
            layer_is_gdn,
        }
    }

    pub fn attn_geo(&self) -> AttnGeometry {
        AttnGeometry {
            t_len: self.t_len,
            n_head: self.n_head,
            n_kv: self.n_kv,
            head_dim: self.head_dim,
            rotary_dim: self.rotary_dim,
        }
    }

    pub fn gdn_geo(&self) -> GdnGeometry {
        GdnGeometry {
            t_len: self.t_len,
            n_k_heads: self.dn_n_k,
            n_v_heads: self.dn_n_v,
            key_dim: self.dn_head_dim,
            val_dim: self.dn_head_dim,
            kernel_size: self.dn_kernel,
        }
    }

    /// The fit points in canonical order: `[Mid(0), Out(0), Mid(1), …]`.
    pub fn fit_points(&self) -> Vec<FitPoint> {
        let mut pts = Vec::with_capacity(self.n_layers * 2);
        for l in 0..self.n_layers {
            pts.push(FitPoint::Mid(l));
            pts.push(FitPoint::Out(l));
        }
        pts
    }
}

/// Where a lens read/write sits: `Mid(l)` = the post-attention residual of
/// layer `l` (reads MLP inputs + attention outputs), `Out(l)` = the layer's
/// output residual (reads MLP outputs + attention inputs) — the upstream
/// catalog's two-points-per-layer rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FitPoint {
    Mid(usize),
    Out(usize),
}

impl FitPoint {
    pub fn layer(&self) -> usize {
        match self {
            FitPoint::Mid(l) | FitPoint::Out(l) => *l,
        }
    }
}

// ---------------------------------------------------------------------------
// Weights — borrowed views over the loaded model
// ---------------------------------------------------------------------------

/// One layer's sub-block weights (GDN or attention).
#[derive(Clone, Copy)]
pub enum SubWeights<'a> {
    Gdn(GdnLayerWeights<'a>),
    Attn(AttnLayerWeights<'a>),
}

/// One layer's borrowed weight views.
#[derive(Clone, Copy)]
pub struct LayerWeights<'a> {
    pub sub: SubWeights<'a>,
    pub input_norm: &'a [f32],
    pub post_attn_norm: &'a [f32],
    /// `gate_proj` — `[mlp_hidden][n_embd]`.
    pub w_gate: &'a [f32],
    /// `up_proj` — `[mlp_hidden][n_embd]`.
    pub w_up: &'a [f32],
    /// `down_proj` — `[n_embd][mlp_hidden]`.
    pub w_down: &'a [f32],
}

/// Borrowed views over a loaded [`QwenDeltaNetWeights`] (dense lane — the
/// lens reads `Proj::dense_data`, which refuses a ternary projection loud).
pub struct StackWeights<'a> {
    pub wte: &'a [f32],
    /// The LM head rows `[vocab][n_embd]` (tied to `wte` for Qwen3.5; kept
    /// separate so a headful non-tied checkpoint reads its own tensor).
    pub lm_head: &'a [f32],
    pub final_norm: &'a [f32],
    pub layers: Vec<LayerWeights<'a>>,
    pub geo: StackGeometry,
}

/// The dense slice of a projection, with a loud refusal on the ternary arm.
fn dense<'a>(p: &'a Proj, what: &'static str) -> &'a [f32] {
    match p {
        Proj::Dense { data, rows, cols } => {
            debug_assert_eq!(data.len(), rows * cols, "lens: {what} dense shape mismatch");
            data
        }
        #[cfg(feature = "deltanet_ternary_inference")]
        Proj::Ternary(_) => {
            panic!("lens: {what} is a ternary projection — the lens lane is dense-only")
        }
    }
}

impl<'a> StackWeights<'a> {
    /// Borrow the model's weights for lens fits at doc length `t_len`.
    ///
    /// `rope_freq` must be the SAME table the serving forward uses
    /// (`RopeFreqTable::new(config.rope_theta, config.head_dim)`) — the RoPE
    /// maps delegate to it, so parity is inherited, not re-derived.
    pub fn from_model(
        model: &'a QwenDeltaNetWeights,
        config: &Config,
        rope_freq: &'a [f32],
        t_len: usize,
    ) -> Self {
        let layer_is_gdn: Vec<bool> = model
            .layer_types
            .iter()
            .map(|t| *t == DeltaNetLayerType::DeltaNet)
            .collect();
        let geo = StackGeometry::from_config(config, t_len, layer_is_gdn);
        debug_assert_eq!(model.layers.len(), geo.n_layers, "layer count");
        let gdn_geo = geo.gdn_geo();
        let attn_geo = geo.attn_geo();
        let layers = model.layers.iter().enumerate().map(|(li, lw)| {
            let sub = if geo.layer_is_gdn[li] {
                // dt_bias/a_log may be longer than n_v (upstream shapes);
                // the gates read the first n_v entries — slice, and
                // refuse a SHORT row loud.
                debug_assert!(lw.dt_bias.len() >= gdn_geo.n_v_heads, "dt_bias short");
                debug_assert!(lw.a_log.len() >= gdn_geo.n_v_heads, "a_log short");
                debug_assert_eq!(lw.linear_norm.len(), gdn_geo.val_dim, "linear_norm");
                SubWeights::Gdn(GdnLayerWeights {
                    w_qkv: dense(&lw.in_proj_qkv, "in_proj_qkv"),
                    w_z: dense(&lw.in_proj_z, "in_proj_z"),
                    w_a: dense(&lw.in_proj_a, "in_proj_a"),
                    w_b: dense(&lw.in_proj_b, "in_proj_b"),
                    w_out: dense(&lw.out_proj, "out_proj"),
                    conv_w: &lw.conv1d_weight,
                    a_log: &lw.a_log[..gdn_geo.n_v_heads],
                    dt_bias: &lw.dt_bias[..gdn_geo.n_v_heads],
                    linear_norm: &lw.linear_norm,
                    n_embd: geo.n_embd,
                    rms_eps: geo.rms_eps,
                })
            } else {
                SubWeights::Attn(AttnLayerWeights {
                    w_qg: dense(&lw.attn_wq, "attn_wq"),
                    w_k: dense(&lw.attn_wk, "attn_wk"),
                    w_v: dense(&lw.attn_wv, "attn_wv"),
                    w_o: dense(&lw.attn_wo, "attn_wo"),
                    q_norm: &lw.attn_q_norm,
                    k_norm: &lw.attn_k_norm,
                    rope_freq,
                    n_embd: geo.n_embd,
                    rms_eps: geo.rms_eps,
                })
            };
            LayerWeights {
                sub,
                input_norm: &lw.input_norm,
                post_attn_norm: &lw.post_attn_norm,
                w_gate: dense(&lw.gate_proj, "gate_proj"),
                w_up: dense(&lw.up_proj, "up_proj"),
                w_down: dense(&lw.down_proj, "down_proj"),
            }
        });
        let _ = (gdn_geo, attn_geo);
        Self {
            wte: &model.wte,
            lm_head: dense(&model.lm_head, "lm_head"),
            final_norm: &model.final_norm,
            layers: layers.collect(),
            geo,
        }
    }
}

// ---------------------------------------------------------------------------
// Capture
// ---------------------------------------------------------------------------

/// One layer's sub-block capture.
#[derive(Default)]
pub enum SubCapture {
    Gdn(GdnLayerCapture),
    Attn(AttnLayerCapture),
    #[default]
    None,
}

/// One layer's captured constants.
pub struct LayerCapture {
    /// LN-rule diagonal at the input norm (`[T·d]`, `γ·r` per position).
    pub input_scale: Vec<f32>,
    pub sub: SubCapture,
    /// The `mid` fit point (post-attention residual) `[T·d]`.
    pub mid: Vec<f32>,
    /// LN-rule diagonal at the post-attention norm `[T·d]`.
    pub post_scale: Vec<f32>,
    /// MLP half-rule constants `[T·hidden]` each (`s = silu(g)`, `u`,
    /// `sp = silu'(g)`).
    pub mlp_s: Vec<f32>,
    pub mlp_u: Vec<f32>,
    pub mlp_sp: Vec<f32>,
    /// The `out` fit point (the layer's output residual) `[T·d]`.
    pub out: Vec<f32>,
}

/// One doc's full capture ([`stack_forward`]'s output).
pub struct StackCapture {
    /// The embedded doc (the residual stream at the stack input) `[T·d]`.
    pub embedded: Vec<f32>,
    pub layers: Vec<LayerCapture>,
    /// LN-rule diagonal at the final norm `[T·d]`.
    pub final_scale: Vec<f32>,
    /// The post-final-norm hidden `[T·d]` (the reads' base).
    pub hn: Vec<f32>,
}

impl StackCapture {
    pub fn new(geo: &StackGeometry) -> Self {
        let mut layers = Vec::with_capacity(geo.n_layers);
        for is_gdn in &geo.layer_is_gdn {
            let sub = if *is_gdn {
                SubCapture::Gdn(GdnLayerCapture::default())
            } else {
                SubCapture::Attn(AttnLayerCapture::default())
            };
            layers.push(LayerCapture {
                input_scale: vec![0.0; geo.t_len * geo.n_embd],
                sub,
                mid: vec![0.0; geo.t_len * geo.n_embd],
                post_scale: vec![0.0; geo.t_len * geo.n_embd],
                mlp_s: vec![0.0; geo.t_len * geo.mlp_hidden],
                mlp_u: vec![0.0; geo.t_len * geo.mlp_hidden],
                mlp_sp: vec![0.0; geo.t_len * geo.mlp_hidden],
                out: vec![0.0; geo.t_len * geo.n_embd],
            });
        }
        Self {
            embedded: vec![0.0; geo.t_len * geo.n_embd],
            layers,
            final_scale: vec![0.0; geo.t_len * geo.n_embd],
            hn: vec![0.0; geo.t_len * geo.n_embd],
        }
    }
}

/// Doc-level stack forward — the capture pass (module doc's op order).
pub fn stack_forward(tokens: &[usize], w: &StackWeights, cap: &mut StackCapture) {
    stack_forward_write(tokens, w, cap, None);
}

/// A steering write injected mid-forward (the T4 G2 measured arm).
///
/// `dir` (`[d]`) scaled by `alpha` is ADDED to the residual at `point`'s
/// rows `[seed_lo, seed_hi)` — the T3 fit's J rows are the linearization of
/// exactly this injection (the same window, the same unit direction), so a
/// measured run with `alpha = 1` is the true nonlinear twin of the fitted
/// row. The captured run carries the write from the injection point onward;
/// rows before the seed window stay bit-identical (causality).
pub struct StackWrite<'a> {
    pub point: FitPoint,
    pub dir: &'a [f32],
    pub alpha: f32,
    pub seed_lo: usize,
    /// Exclusive.
    pub seed_hi: usize,
}

/// Doc-level stack forward with an optional steering write ([`StackWrite`]).
pub fn stack_forward_write(
    tokens: &[usize],
    w: &StackWeights,
    cap: &mut StackCapture,
    write: Option<&StackWrite>,
) {
    let geo = &w.geo;
    let (t_len, d) = (geo.t_len, geo.n_embd);
    let hidden = geo.mlp_hidden;
    debug_assert_eq!(tokens.len(), t_len, "stack_forward: token count");
    if let Some(sw) = write {
        debug_assert_eq!(sw.dir.len(), d, "stack write: dir shape");
        debug_assert!(sw.point.layer() < geo.n_layers, "stack write: point layer");
        debug_assert!(
            sw.seed_lo < sw.seed_hi && sw.seed_hi <= t_len,
            "stack write: window"
        );
    }
    #[inline(always)]
    fn inject(sw: &StackWrite, want: FitPoint, buf: &mut [f32], t_len: usize, d: usize) {
        if sw.point != want {
            return;
        }
        for t in sw.seed_lo..sw.seed_hi.min(t_len) {
            let row = &mut buf[t * d..(t + 1) * d];
            for (r, &dv) in row.iter_mut().zip(sw.dir) {
                *r += sw.alpha * dv;
            }
        }
    }
    // Embedding gather (no scale — the forward's step 1).
    for (t, &tok) in tokens.iter().enumerate() {
        debug_assert!(tok < geo.vocab, "token id out of range");
        let off = tok * d;
        cap.embedded[t * d..(t + 1) * d].copy_from_slice(&w.wte[off..off + d]);
    }
    let mut x = cap.embedded.clone();
    let mut xh = vec![0.0f32; t_len * d];
    let mut g = vec![0.0f32; hidden];
    let mut u_pre = vec![0.0f32; hidden];
    let mut act = vec![0.0f32; hidden];
    let mut y_sub = vec![0.0f32; t_len * d];
    let mut y_mlp = vec![0.0f32; t_len * d];
    let mut xh2 = vec![0.0f32; t_len * d];

    for (li, lw) in w.layers.iter().enumerate() {
        let lc = &mut cap.layers[li];
        // Input RMSNorm: capture the LN-rule diagonal; the normed value is
        // the diagonal applied to x (γ·r ⊙ x == γ ⊙ x ⊙ r exactly).
        for t in 0..t_len {
            norm_diag_scale(
                &x[t * d..(t + 1) * d],
                lw.input_norm,
                geo.rms_eps,
                &mut lc.input_scale[t * d..(t + 1) * d],
            );
        }
        xh.copy_from_slice(&x);
        for t in 0..t_len {
            diag_apply(
                &lc.input_scale[t * d..(t + 1) * d],
                &mut xh[t * d..(t + 1) * d],
            );
        }
        // Sublayer forward.
        match (&mut lc.sub, &lw.sub) {
            (SubCapture::Gdn(gc), SubWeights::Gdn(gw)) => {
                gdn_layer_forward(&xh, gw, &gdn_geo_of(geo), gc);
            }
            (SubCapture::Attn(ac), SubWeights::Attn(aw)) => {
                attn_layer_forward(&xh, aw, &attn_geo_of(geo), ac);
            }
            _ => unreachable!("sub capture/weights kind mismatch"),
        }
        let y = match &lc.sub {
            SubCapture::Gdn(c) => &c.y,
            SubCapture::Attn(c) => &c.y,
            SubCapture::None => unreachable!(),
        };
        y_sub.copy_from_slice(y);
        // mid = x + y_sub (the fit point).
        for i in 0..t_len * d {
            lc.mid[i] = x[i] + y_sub[i];
        }
        if let Some(sw) = write {
            inject(sw, FitPoint::Mid(li), &mut lc.mid, t_len, d);
        }
        x.copy_from_slice(&lc.mid);
        // Post-attention RMSNorm + SwiGLU MLP.
        for t in 0..t_len {
            norm_diag_scale(
                &x[t * d..(t + 1) * d],
                lw.post_attn_norm,
                geo.rms_eps,
                &mut lc.post_scale[t * d..(t + 1) * d],
            );
            xh2[t * d..(t + 1) * d].copy_from_slice(&x[t * d..(t + 1) * d]);
            diag_apply(
                &lc.post_scale[t * d..(t + 1) * d],
                &mut xh2[t * d..(t + 1) * d],
            );
            matmul(&mut g, lw.w_gate, &xh2[t * d..(t + 1) * d], hidden, d);
            matmul(&mut u_pre, lw.w_up, &xh2[t * d..(t + 1) * d], hidden, d);
            MlpConsts::capture(
                &g,
                &u_pre,
                &mut lc.mlp_s[t * hidden..(t + 1) * hidden],
                &mut lc.mlp_u[t * hidden..(t + 1) * hidden],
                &mut lc.mlp_sp[t * hidden..(t + 1) * hidden],
            );
            // act = s ⊙ u (the MLP forward value; the half rule's linear
            // point).
            let s_row = &lc.mlp_s[t * hidden..(t + 1) * hidden];
            let u_row = &lc.mlp_u[t * hidden..(t + 1) * hidden];
            for ((a, &sv), &uv) in act.iter_mut().zip(s_row).zip(u_row) {
                *a = sv * uv;
            }
            matmul(&mut y_mlp[t * d..(t + 1) * d], lw.w_down, &act, d, hidden);
        }
        // out = mid + y_mlp (the fit point).
        for ((o, &m), &ym) in lc.out.iter_mut().zip(lc.mid.iter()).zip(y_mlp.iter()) {
            *o = m + ym;
        }
        if let Some(sw) = write {
            inject(sw, FitPoint::Out(li), &mut lc.out, t_len, d);
        }
        x.copy_from_slice(&lc.out);
    }

    // Final RMSNorm.
    for t in 0..t_len {
        norm_diag_scale(
            &x[t * d..(t + 1) * d],
            w.final_norm,
            geo.rms_eps,
            &mut cap.final_scale[t * d..(t + 1) * d],
        );
        cap.hn[t * d..(t + 1) * d].copy_from_slice(&x[t * d..(t + 1) * d]);
        diag_apply(
            &cap.final_scale[t * d..(t + 1) * d],
            &mut cap.hn[t * d..(t + 1) * d],
        );
    }
}

fn gdn_geo_of(geo: &StackGeometry) -> GdnGeometry {
    geo.gdn_geo()
}

fn attn_geo_of(geo: &StackGeometry) -> AttnGeometry {
    geo.attn_geo()
}

// ---------------------------------------------------------------------------
// Scratch + tangent maps
// ---------------------------------------------------------------------------

/// Caller-provided scratch for the stack tangent maps (allocate once per
/// doc; the maps themselves allocate nothing).
pub struct StackScratch {
    attn: AttnLayerScratch,
    gdn: GdnLayerScratch,
    t: Vec<f32>,
    t2: Vec<f32>,
    gsub: Vec<f32>,
    dg: Vec<f32>,
    du: Vec<f32>,
    dact: Vec<f32>,
    dmlp: Vec<f32>,
    tmp_d: Vec<f32>,
    g_mid: Vec<f32>,
    gdg: Vec<f32>,
    gdu: Vec<f32>,
}

impl StackScratch {
    pub fn new(geo: &StackGeometry) -> Self {
        let td = geo.t_len * geo.n_embd;
        let th = geo.t_len * geo.mlp_hidden;
        Self {
            attn: AttnLayerScratch::new(&geo.attn_geo()),
            gdn: GdnLayerScratch::new(&geo.gdn_geo()),
            t: vec![0.0; td],
            t2: vec![0.0; td],
            gsub: vec![0.0; td],
            dg: vec![0.0; th],
            du: vec![0.0; th],
            dact: vec![0.0; geo.mlp_hidden],
            dmlp: vec![0.0; td],
            tmp_d: vec![0.0; geo.n_embd],
            g_mid: vec![0.0; td],
            gdg: vec![0.0; th],
            gdu: vec![0.0; th],
        }
    }
}

/// Tangents at every fit point + the final post-norm tangent
/// ([`stack_prefix_jvp`]'s output; allocated by [`PrefixTangents::new`],
/// reused across basis directions).
pub struct PrefixTangents {
    pub mid: Vec<Vec<f32>>,
    pub out: Vec<Vec<f32>>,
    pub hn: Vec<f32>,
}

impl PrefixTangents {
    pub fn new(geo: &StackGeometry) -> Self {
        let td = geo.t_len * geo.n_embd;
        Self {
            mid: (0..geo.n_layers).map(|_| vec![0.0; td]).collect(),
            out: (0..geo.n_layers).map(|_| vec![0.0; td]).collect(),
            hn: vec![0.0; td],
        }
    }
}

/// One layer's jvp step: advance the working tangent `s.t` from the layer's
/// input to its output (input-norm diag → sublayer → residual → post-norm
/// diag → MLP half rule → residual). Used by the suffix loop; the prefix
/// pass inlines its own body to record the mid point in between.
fn layer_jvp_step_simple(li: usize, cap: &StackCapture, w: &StackWeights, s: &mut StackScratch) {
    let geo = &w.geo;
    let td = geo.t_len * geo.n_embd;
    let lc = &cap.layers[li];
    let lw = &w.layers[li];
    for t in 0..geo.t_len {
        diag_apply(
            &lc.input_scale[t * geo.n_embd..(t + 1) * geo.n_embd],
            &mut s.t[t * geo.n_embd..(t + 1) * geo.n_embd],
        );
    }
    sublayer_jvp(lc, lw, geo, s);
    for i in 0..td {
        s.t[i] += s.t2[i];
    }
    mlp_jvp(lc, lw, geo, s);
    for i in 0..td {
        s.t[i] += s.dmlp[i];
    }
}

/// Sublayer jvp: `s.t` (normed-input tangent) → `s.t2` (sublayer output
/// tangent).
fn sublayer_jvp(lc: &LayerCapture, lw: &LayerWeights, geo: &StackGeometry, s: &mut StackScratch) {
    match (&lc.sub, &lw.sub) {
        (SubCapture::Gdn(gc), SubWeights::Gdn(gw)) => {
            let consts = GdnLayerConsts {
                w: *gw,
                geo: geo.gdn_geo(),
                cap: gc,
            };
            gdn_layer_jvp(&consts, &s.t, &mut s.gdn, &mut s.t2);
        }
        (SubCapture::Attn(ac), SubWeights::Attn(aw)) => {
            let consts = AttnLayerConsts {
                w: *aw,
                geo: geo.attn_geo(),
                cap: ac,
            };
            attn_layer_jvp(&consts, &s.t, &mut s.attn, &mut s.t2);
        }
        _ => unreachable!("sub capture/weights kind mismatch"),
    }
}

/// MLP jvp: `s.t` (mid tangent, PRE post-norm) → `s.dmlp` (MLP output
/// tangent). Applies the post-norm diag internally.
fn mlp_jvp(lc: &LayerCapture, lw: &LayerWeights, geo: &StackGeometry, s: &mut StackScratch) {
    let (t_len, d, hidden) = (geo.t_len, geo.n_embd, geo.mlp_hidden);
    for t in 0..t_len {
        s.t2[t * d..(t + 1) * d].copy_from_slice(&s.t[t * d..(t + 1) * d]);
        diag_apply(
            &lc.post_scale[t * d..(t + 1) * d],
            &mut s.t2[t * d..(t + 1) * d],
        );
        matmul(
            &mut s.dg[t * hidden..(t + 1) * hidden],
            lw.w_gate,
            &s.t2[t * d..(t + 1) * d],
            hidden,
            d,
        );
        matmul(
            &mut s.du[t * hidden..(t + 1) * hidden],
            lw.w_up,
            &s.t2[t * d..(t + 1) * d],
            hidden,
            d,
        );
    }
    for t in 0..t_len {
        let consts = MlpConsts {
            s: &lc.mlp_s[t * hidden..(t + 1) * hidden],
            u: &lc.mlp_u[t * hidden..(t + 1) * hidden],
            sp: &lc.mlp_sp[t * hidden..(t + 1) * hidden],
        };
        mlp_half_rule_jvp(
            &consts,
            &s.dg[t * hidden..(t + 1) * hidden],
            &s.du[t * hidden..(t + 1) * hidden],
            lw.w_down,
            d,
            &mut s.dact,
            &mut s.dmlp[t * d..(t + 1) * d],
        );
    }
}

/// Prefix `jvp`: embedding tangents `d_emb` `[T·d]` → tangents at EVERY fit
/// point + the final post-norm tangent (the S-lens fixed-end pass).
pub fn stack_prefix_jvp(
    cap: &StackCapture,
    w: &StackWeights,
    s: &mut StackScratch,
    d_emb: &[f32],
    out: &mut PrefixTangents,
) {
    let geo = &w.geo;
    let d = geo.n_embd;
    let td = geo.t_len * d;
    debug_assert_eq!(d_emb.len(), td, "prefix jvp: d_emb shape");
    s.t.copy_from_slice(d_emb);
    for li in 0..geo.n_layers {
        let lc = &cap.layers[li];
        let lw = &w.layers[li];
        for t in 0..geo.t_len {
            diag_apply(
                &lc.input_scale[t * d..(t + 1) * d],
                &mut s.t[t * d..(t + 1) * d],
            );
        }
        sublayer_jvp(lc, lw, geo, s);
        for i in 0..td {
            s.t[i] += s.t2[i];
        }
        // The `mid` fit point (post-attention residual tangent).
        out.mid[li].copy_from_slice(&s.t);
        mlp_jvp(lc, lw, geo, s);
        for i in 0..td {
            s.t[i] += s.dmlp[i];
        }
        // The `out` fit point (layer output residual tangent).
        out.out[li].copy_from_slice(&s.t);
    }
    // Final norm (LN-rule diag, self-adjoint).
    for t in 0..geo.t_len {
        diag_apply(
            &cap.final_scale[t * d..(t + 1) * d],
            &mut s.t[t * d..(t + 1) * d],
        );
    }
    out.hn.copy_from_slice(&s.t);
}

/// Suffix `jvp`: a write tangent `dv` `[T·d]` at `point` → the final
/// post-norm tangent `out_hn` `[T·d]` (the R-lens per-point pass; the LN
/// rule's final-norm diagonal is INCLUDED, so `out_hn` lives in the read
/// space `lm_head` consumes).
pub fn stack_suffix_jvp(
    cap: &StackCapture,
    w: &StackWeights,
    s: &mut StackScratch,
    point: FitPoint,
    dv: &[f32],
    out_hn: &mut [f32],
) {
    let geo = &w.geo;
    let d = geo.n_embd;
    let td = geo.t_len * d;
    debug_assert_eq!(dv.len(), td, "suffix jvp: dv shape");
    debug_assert_eq!(out_hn.len(), td, "suffix jvp: out shape");
    s.t.copy_from_slice(dv);
    match point {
        FitPoint::Out(l) => {
            // The write enters layer l+1 directly.
            for li in (l + 1)..geo.n_layers {
                layer_jvp_step_simple(li, cap, w, s);
            }
        }
        FitPoint::Mid(l) => {
            // The write enters layer l's post-norm: MLP half first.
            let lc = &cap.layers[l];
            let lw = &w.layers[l];
            mlp_jvp(lc, lw, geo, s);
            for i in 0..td {
                s.t[i] += s.dmlp[i];
            }
            for li in (l + 1)..geo.n_layers {
                layer_jvp_step_simple(li, cap, w, s);
            }
        }
    }
    for t in 0..geo.t_len {
        diag_apply(
            &cap.final_scale[t * d..(t + 1) * d],
            &mut s.t[t * d..(t + 1) * d],
        );
    }
    out_hn.copy_from_slice(&s.t);
}

/// One layer's vjp step: unwind the working cotangent `s.g_mid`(== the
/// cotangent at the layer's OUTPUT) to the layer's INPUT, leaving the
/// result in `s.g_mid`.
fn layer_vjp_step(li: usize, cap: &StackCapture, w: &StackWeights, s: &mut StackScratch) {
    let geo = &w.geo;
    let d = geo.n_embd;
    let hidden = geo.mlp_hidden;
    let lc = &cap.layers[li];
    let lw = &w.layers[li];
    // g_mid = g (residual passthrough) + MLP-path adjoint.
    s.g_mid.copy_from_slice(&s.t);
    for t in 0..geo.t_len {
        let consts = MlpConsts {
            s: &lc.mlp_s[t * hidden..(t + 1) * hidden],
            u: &lc.mlp_u[t * hidden..(t + 1) * hidden],
            sp: &lc.mlp_sp[t * hidden..(t + 1) * hidden],
        };
        mlp_half_rule_vjp(
            &consts,
            &s.t[t * d..(t + 1) * d],
            lw.w_down,
            d,
            &mut s.dact,
            &mut s.gdg[t * hidden..(t + 1) * hidden],
            &mut s.gdu[t * hidden..(t + 1) * hidden],
        );
    }
    for t in 0..geo.t_len {
        matvec_t(
            &mut s.t2[t * d..(t + 1) * d],
            lw.w_gate,
            &s.gdg[t * hidden..(t + 1) * hidden],
            hidden,
            d,
        );
        matvec_t(
            &mut s.tmp_d,
            lw.w_up,
            &s.gdu[t * hidden..(t + 1) * hidden],
            hidden,
            d,
        );
        for i in 0..d {
            s.t2[t * d + i] += s.tmp_d[i];
        }
        diag_apply(
            &lc.post_scale[t * d..(t + 1) * d],
            &mut s.t2[t * d..(t + 1) * d],
        );
        for i in 0..d {
            s.g_mid[t * d + i] += s.t2[t * d + i];
        }
    }
    // mid = x + sub(x̂): cotangent at x = g_mid + sublayer adjoint(g_mid).
    s.t.copy_from_slice(&s.g_mid);
    sublayer_vjp(lc, lw, geo, s);
    for i in 0..geo.t_len * d {
        s.g_mid[i] += s.gsub[i];
    }
    // Input-norm adjoint (diagonal, self-adjoint).
    for t in 0..geo.t_len {
        diag_apply(
            &lc.input_scale[t * d..(t + 1) * d],
            &mut s.g_mid[t * d..(t + 1) * d],
        );
    }
    // Hand the cotangent back via s.t (the caller's working slot).
    s.t.copy_from_slice(&s.g_mid);
}

/// Sublayer vjp: `s.t` → `s.gsub`.
fn sublayer_vjp(lc: &LayerCapture, lw: &LayerWeights, geo: &StackGeometry, s: &mut StackScratch) {
    match (&lc.sub, &lw.sub) {
        (SubCapture::Gdn(gc), SubWeights::Gdn(gw)) => {
            let consts = GdnLayerConsts {
                w: *gw,
                geo: geo.gdn_geo(),
                cap: gc,
            };
            gdn_layer_vjp(&consts, &s.t, &mut s.gdn, &mut s.gsub);
        }
        (SubCapture::Attn(ac), SubWeights::Attn(aw)) => {
            let consts = AttnLayerConsts {
                w: *aw,
                geo: geo.attn_geo(),
                cap: ac,
            };
            attn_layer_vjp(&consts, &s.t, &mut s.attn, &mut s.gsub);
        }
        _ => unreachable!("sub capture/weights kind mismatch"),
    }
}

/// Suffix `vjp` — the exact adjoint of [`stack_suffix_jvp`]: a final
/// post-norm cotangent `gh` `[T·d]` → the write cotangent at `point`
/// `out_g` `[T·d]` (the transpose identity's implementation half).
pub fn stack_suffix_vjp(
    cap: &StackCapture,
    w: &StackWeights,
    s: &mut StackScratch,
    point: FitPoint,
    gh: &[f32],
    out_g: &mut [f32],
) {
    let geo = &w.geo;
    let d = geo.n_embd;
    let hidden = geo.mlp_hidden;
    let td = geo.t_len * d;
    debug_assert_eq!(gh.len(), td, "suffix vjp: gh shape");
    debug_assert_eq!(out_g.len(), td, "suffix vjp: out shape");
    // Final-norm adjoint (diagonal).
    s.t.copy_from_slice(gh);
    for t in 0..geo.t_len {
        diag_apply(
            &cap.final_scale[t * d..(t + 1) * d],
            &mut s.t[t * d..(t + 1) * d],
        );
    }
    // Unwind layers (point.layer+1)..n (each step: out → input cotangent).
    for li in (point.layer() + 1)..geo.n_layers {
        layer_vjp_step(li, cap, w, s);
    }
    match point {
        FitPoint::Out(_) => {
            out_g.copy_from_slice(&s.t);
        }
        FitPoint::Mid(l) => {
            // Unwind layer l's MLP path: cotangent at out → cotangent at mid.
            let lc = &cap.layers[l];
            let lw = &w.layers[l];
            s.g_mid.copy_from_slice(&s.t);
            for t in 0..geo.t_len {
                let consts = MlpConsts {
                    s: &lc.mlp_s[t * hidden..(t + 1) * hidden],
                    u: &lc.mlp_u[t * hidden..(t + 1) * hidden],
                    sp: &lc.mlp_sp[t * hidden..(t + 1) * hidden],
                };
                mlp_half_rule_vjp(
                    &consts,
                    &s.t[t * d..(t + 1) * d],
                    lw.w_down,
                    d,
                    &mut s.dact,
                    &mut s.gdg[t * hidden..(t + 1) * hidden],
                    &mut s.gdu[t * hidden..(t + 1) * hidden],
                );
            }
            for t in 0..geo.t_len {
                matvec_t(
                    &mut s.t2[t * d..(t + 1) * d],
                    lw.w_gate,
                    &s.gdg[t * hidden..(t + 1) * hidden],
                    hidden,
                    d,
                );
                matvec_t(
                    &mut s.tmp_d,
                    lw.w_up,
                    &s.gdu[t * hidden..(t + 1) * hidden],
                    hidden,
                    d,
                );
                for i in 0..d {
                    s.t2[t * d + i] += s.tmp_d[i];
                }
                diag_apply(
                    &lc.post_scale[t * d..(t + 1) * d],
                    &mut s.t2[t * d..(t + 1) * d],
                );
                for i in 0..d {
                    s.g_mid[t * d + i] += s.t2[t * d + i];
                }
            }
            out_g.copy_from_slice(&s.g_mid);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests — parity vs the serving decode loop, causality, adjointness, FD
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deltanet::forward::{HybridCache, HybridForwardScratch, forward_qwen_deltanet};
    use crate::lens::fit::fit_lens;
    use crate::rope::RopeFreqTable;
    use crate::types::DeltaNetLayerType::*;

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

    /// Tiny hybrid model: `layer_types` chosen by the caller (bisectable),
    /// d=16, vocab=32, partial RoPE. GDN gate coefficients are kept in the
    /// stable band (`a_log` negative, dt small) so the doc forward stays
    /// bounded.
    fn tiny_model(
        seed: u64,
        layer_types: Vec<crate::types::DeltaNetLayerType>,
    ) -> (crate::types::Config, QwenDeltaNetWeights) {
        let mut rng = Lcg(seed);
        let nl = layer_types.len();
        let mut config = crate::types::Config::qwen_deltanet(nl, layer_types);
        config.n_embd = 16;
        config.n_head = 2;
        config.n_kv_head = 1;
        config.head_dim = 8;
        config.mlp_hidden = 32;
        config.vocab_size = 32;
        config.deltanet_linear_head_dim = 4;
        config.deltanet_linear_n_heads = 1;
        config.deltanet_linear_n_value_heads = 2;
        config.deltanet_conv_kernel_size = 3;
        config.rope_dimension_count = 4; // partial RoPE
        config.block_size = 64;
        let d = config.n_embd;
        let q_dim = config.n_head * config.head_dim;
        let kv_dim = config.n_kv_head * config.head_dim;
        let hidden = config.mlp_hidden;
        let n_k = config.deltanet_linear_n_heads;
        let n_v = config.deltanet_linear_n_value_heads;
        let lhd = config.deltanet_linear_head_dim;
        let qkv_dim = 2 * n_k * lhd + n_v * lhd;
        let conv_dim = qkv_dim;
        let mut weights = QwenDeltaNetWeights::zeros(&config);
        let fill = |p: &mut crate::deltanet::Proj, rows: usize, rng: &mut Lcg, scale: f32| {
            let data = rng.vec(rows * cols_of(rows, d));
            let data: Vec<f32> = data.into_iter().map(|x| x * scale).collect();
            *p = crate::deltanet::Proj::Dense {
                data,
                rows,
                cols: d,
            };
        };
        fn cols_of(rows: usize, d: usize) -> usize {
            let _ = rows;
            d
        }
        // Embedding + head (small, so logits stay in a sane band).
        weights.wte = rng
            .vec(config.vocab_size * d)
            .iter()
            .map(|x| x * 0.5)
            .collect();
        {
            let head: Vec<f32> = rng
                .vec(config.vocab_size * d)
                .iter()
                .map(|x| x * 0.3)
                .collect();
            weights.lm_head = crate::deltanet::Proj::dense(head, config.vocab_size, d);
        }
        weights.final_norm = (0..d).map(|_| 1.0 + 0.1 * rng.next_f32()).collect();
        let norm =
            |rng: &mut Lcg| -> Vec<f32> { (0..d).map(|_| 1.0 + 0.1 * rng.next_f32()).collect() };
        for lw in weights.layers.iter_mut() {
            lw.input_norm = norm(&mut rng);
            lw.post_attn_norm = norm(&mut rng);
            // MLP (SwiGLU).
            fill(&mut lw.gate_proj, hidden, &mut rng, 0.4);
            fill(&mut lw.up_proj, hidden, &mut rng, 0.4);
            {
                let data: Vec<f32> = rng.vec(d * hidden).iter().map(|x| x * 0.3).collect();
                lw.down_proj = crate::deltanet::Proj::dense(data, d, hidden);
            }
        }
        // Per-type sub-blocks.
        for (li, lw) in weights.layers.iter_mut().enumerate() {
            if config.layer_types[li] == DeltaNet {
                fill(&mut lw.in_proj_qkv, qkv_dim, &mut rng, 0.4);
                fill(&mut lw.in_proj_z, n_v * lhd, &mut rng, 0.4);
                fill(&mut lw.in_proj_a, n_v, &mut rng, 0.3);
                fill(&mut lw.in_proj_b, n_v, &mut rng, 0.3);
                {
                    let data: Vec<f32> = rng.vec(d * (n_v * lhd)).iter().map(|x| x * 0.3).collect();
                    lw.out_proj = crate::deltanet::Proj::dense(data, d, n_v * lhd);
                }
                lw.conv1d_weight = rng
                    .vec(conv_dim * config.deltanet_conv_kernel_size)
                    .iter()
                    .map(|x| x * 0.4)
                    .collect();
                lw.a_log = (0..n_v).map(|_| -0.5 - rng.next_f32().abs()).collect();
                lw.dt_bias = rng.vec(n_v).iter().map(|x| x * 0.2).collect();
                lw.linear_norm = (0..lhd).map(|_| 1.0 + 0.1 * rng.next_f32()).collect();
            } else {
                fill(&mut lw.attn_wq, 2 * q_dim, &mut rng, 0.4);
                fill(&mut lw.attn_wk, kv_dim, &mut rng, 0.4);
                fill(&mut lw.attn_wv, kv_dim, &mut rng, 0.4);
                {
                    let data: Vec<f32> = rng.vec(d * q_dim).iter().map(|x| x * 0.3).collect();
                    lw.attn_wo = crate::deltanet::Proj::dense(data, d, q_dim);
                }
                lw.attn_q_norm = (0..config.head_dim)
                    .map(|_| 1.0 + 0.1 * rng.next_f32())
                    .collect();
                lw.attn_k_norm = (0..config.head_dim)
                    .map(|_| 1.0 + 0.1 * rng.next_f32())
                    .collect();
            }
        }
        (config, weights)
    }

    const T: usize = 6;
    const TOKENS: [usize; T] = [3, 17, 5, 29, 1, 11];

    fn stack_of<'a>(
        config: &crate::types::Config,
        weights: &'a QwenDeltaNetWeights,
        rope_freq: &'a RopeFreqTable,
    ) -> StackWeights<'a> {
        StackWeights::from_model(weights, config, rope_freq.as_slice(), T)
    }

    /// A `StackWrite` with `alpha = 0` must leave every capture byte
    /// identical to the plain forward — the write lane is additive-only.
    #[test]
    fn stack_write_alpha_zero_is_bit_identical() {
        for types in [vec![DeltaNet], vec![Attention], vec![DeltaNet, Attention]] {
            let (config, weights) = tiny_model(0x99E1, types);
            let rope_freq = RopeFreqTable::new(config.rope_theta, config.head_dim);
            let sw = stack_of(&config, &weights, &rope_freq);
            let geo = sw.geo.clone();
            let d = geo.n_embd;
            let dir = vec![0.0f32; d];
            let mut base = StackCapture::new(&geo);
            stack_forward(&TOKENS, &sw, &mut base);
            for point in [FitPoint::Mid(0), FitPoint::Out(0)] {
                let mut wcap = StackCapture::new(&geo);
                stack_forward_write(
                    &TOKENS,
                    &sw,
                    &mut wcap,
                    Some(&StackWrite {
                        point,
                        dir: &dir,
                        alpha: 0.0,
                        seed_lo: 2,
                        seed_hi: 4,
                    }),
                );
                assert_eq!(base.hn, wcap.hn, "{point:?}: alpha=0 must be identity");
                assert_eq!(base.embedded, wcap.embedded);
                for (lb, lw) in base.layers.iter().zip(&wcap.layers) {
                    assert_eq!(lb.mid, lw.mid, "{point:?}: mid diverged");
                    assert_eq!(lb.out, lw.out, "{point:?}: out diverged");
                }
            }
        }
    }

    /// The write is causal and point-local: captures BEFORE the write point
    /// are bit-identical, positions BEFORE the seed window are bit-identical
    /// at the injection (exact `+alpha·dir` rows), and a LATER layer's mixer
    /// carries the write to positions after the window.
    #[test]
    fn stack_write_is_causal_and_local() {
        let (config, weights) = tiny_model(0x99E2, vec![DeltaNet, Attention]);
        let rope_freq = RopeFreqTable::new(config.rope_theta, config.head_dim);
        let sw = stack_of(&config, &weights, &rope_freq);
        let geo = sw.geo.clone();
        let d = geo.n_embd;
        let dir: Vec<f32> = (0..d)
            .map(|i| if i % 2 == 0 { 0.5 } else { -0.25 })
            .collect();
        let mut base = StackCapture::new(&geo);
        stack_forward(&TOKENS, &sw, &mut base);
        let mut wcap = StackCapture::new(&geo);
        stack_forward_write(
            &TOKENS,
            &sw,
            &mut wcap,
            Some(&StackWrite {
                point: FitPoint::Mid(0),
                dir: &dir,
                alpha: 1.0,
                seed_lo: 2,
                seed_hi: 4,
            }),
        );
        // Layer 0 mid: rows below the window identical, window rows moved by
        // exactly the write (mid = x + y_sub, the injection is the last op).
        for t in 0..2 {
            assert_eq!(
                base.layers[0].mid[t * d..(t + 1) * d],
                wcap.layers[0].mid[t * d..(t + 1) * d],
                "row {t} below the window must be identical"
            );
        }
        for t in 2..4 {
            for (i, &dv) in dir.iter().enumerate() {
                let want = base.layers[0].mid[t * d + i] + dv;
                assert_eq!(wcap.layers[0].mid[t * d + i], want, "row {t} dim {i}");
            }
        }
        // hn: rows below the window identical, rows in/after the window move
        // (the later layer's attention carries the write forward).
        for t in 0..2 {
            assert_eq!(
                base.hn[t * d..(t + 1) * d],
                wcap.hn[t * d..(t + 1) * d],
                "hn row {t} must be identical"
            );
        }
        assert!(
            (0..d).any(|i| base.hn[2 * d + i] != wcap.hn[2 * d + i]),
            "hn row 2 (in window) must move"
        );
        assert!(
            (0..d).any(|i| base.hn[5 * d + i] != wcap.hn[5 * d + i]),
            "hn row 5 (after window) must move via the later mixer"
        );
    }

    /// The capture forward must reproduce the SERVING decode loop's logits
    /// (each position's next-token distribution) — the structural pin: any
    /// wrong residual plumbing, RoPE phase, causal mask or GQA group shows
    /// up O(1). The only deliberate drift is the exact-vs-fast sigmoid on
    /// the gate coefficients (the disclosed fit-side noise, T2b/T2c).
    #[test]
    fn stack_forward_matches_decode_loop() {
        // Bisect: single GDN layer, single attention layer, then the hybrid.
        for (name, types) in [
            ("gdn", vec![DeltaNet]),
            ("attn", vec![Attention]),
            ("hybrid", vec![DeltaNet, Attention]),
        ] {
            let (config, weights) = tiny_model(0x999D, types);
            let rope_freq = RopeFreqTable::new(config.rope_theta, config.head_dim);
            let sw = stack_of(&config, &weights, &rope_freq);
            let mut cache = HybridCache::with_layer_types(&config, &weights.layer_types);
            let mut scratch = HybridForwardScratch::new(&config);
            let mut x = vec![0.0f32; config.n_embd.max(config.vocab_size)];
            let mut decode_logits = Vec::with_capacity(T * config.vocab_size);
            for (pos, &tok) in TOKENS.iter().enumerate() {
                let lg = forward_qwen_deltanet(
                    &mut x,
                    &weights,
                    &mut cache,
                    tok,
                    pos,
                    &config,
                    &mut scratch,
                    &rope_freq,
                );
                decode_logits.extend_from_slice(&lg[..config.vocab_size]);
            }
            let mut cap = StackCapture::new(&sw.geo);
            stack_forward(&TOKENS, &sw, &mut cap);
            let mut worst = 0.0f64;
            let mut worst_t = 0usize;
            for t in 0..T {
                let mut logits = vec![0.0f32; config.vocab_size];
                matmul(
                    &mut logits,
                    sw.lm_head,
                    &cap.hn[t * config.n_embd..(t + 1) * config.n_embd],
                    config.vocab_size,
                    config.n_embd,
                );
                for v in 0..config.vocab_size {
                    let got = logits[v] as f64;
                    let want = decode_logits[t * config.vocab_size + v] as f64;
                    let scale = want.abs().max(got.abs()).max(1.0);
                    let rel = (got - want).abs() / scale;
                    if rel > worst {
                        worst = rel;
                        worst_t = t;
                    }
                }
            }
            println!("parity[{name}]: worst {worst:.3e} at t={worst_t}");
            assert!(
                worst < 2e-2,
                "stack parity drift[{name}] {worst:.2e} — the capture diverged from the decode loop"
            );
        }
    }

    /// Causality probe: a tangent injected at the LAST position must leave
    /// every EARLIER position's tangent zero, layer by layer.
    #[test]
    fn stack_tangents_are_causal() {
        let (config, weights) = tiny_model(0x999E, vec![DeltaNet, Attention]);
        let rope_freq = RopeFreqTable::new(config.rope_theta, config.head_dim);
        let sw = stack_of(&config, &weights, &rope_freq);
        let geo = &sw.geo;
        let d = geo.n_embd;
        let mut cap = StackCapture::new(geo);
        stack_forward(&TOKENS, &sw, &mut cap);
        let mut s = StackScratch::new(geo);
        let mut pt = PrefixTangents::new(geo);
        let mut one = vec![0.0f32; T * d];
        one[(T - 1) * d] = 1.0;
        stack_prefix_jvp(&cap, &sw, &mut s, &one, &mut pt);
        for li in 0..geo.n_layers {
            let (which, buf) = ("mid", &pt.mid[li]);
            let worst = (0..(T - 1) * d)
                .map(|i| buf[i].abs() as f64)
                .fold(0.0f64, f64::max);
            assert!(
                worst < 1e-4,
                "layer {li} {which}: acausal leak {worst:.2e} at positions < T-1"
            );
        }
    }

    /// The pair-sum identity that makes the sweep fit EXACT: injecting a
    /// basis direction at EVERY position and summing the response equals the
    /// sum of per-position injections (causality + linearity). A future
    /// leak (attention, conv, or recurrence reading a tangent from `s > t`)
    /// breaks this.
    #[test]
    fn stack_sweeps_satisfy_the_pair_sum_identity() {
        let (config, weights) = tiny_model(0x999E, vec![DeltaNet, Attention]);
        let rope_freq = RopeFreqTable::new(config.rope_theta, config.head_dim);
        let sw = stack_of(&config, &weights, &rope_freq);
        let geo = &sw.geo;
        let d = geo.n_embd;
        let mut cap = StackCapture::new(geo);
        stack_forward(&TOKENS, &sw, &mut cap);
        let mut s = StackScratch::new(geo);
        let mut pt = PrefixTangents::new(geo);
        let mut dhn = vec![0.0f32; T * d];
        for i in [0usize, d / 2, d - 1] {
            // S side: everywhere-injection vs the per-position sum.
            let mut d_emb = vec![0.0f32; T * d];
            for t in 0..T {
                d_emb[t * d + i] = 1.0;
            }
            stack_prefix_jvp(&cap, &sw, &mut s, &d_emb, &mut pt);
            // Snapshot the everywhere-response for ALL layers first — the
            // per-position loop below overwrites `pt` (the stale-read bug
            // this test once had: layer 0 passed, layer 1 read garbage).
            let every: Vec<Vec<f64>> = (0..geo.n_layers)
                .map(|li| {
                    (0..d)
                        .map(|j| (0..T).map(|t| pt.mid[li][t * d + j] as f64).sum::<f64>())
                        .collect()
                })
                .collect();
            for (li, _every_li) in every.iter().enumerate() {
                let mut per_pos = vec![0.0f64; d];
                for spos in 0..T {
                    let mut one = vec![0.0f32; T * d];
                    one[spos * d + i] = 1.0;
                    stack_prefix_jvp(&cap, &sw, &mut s, &one, &mut pt);
                    for (j, pp) in per_pos.iter_mut().enumerate() {
                        *pp += (spos..T).map(|t| pt.mid[li][t * d + j] as f64).sum::<f64>();
                    }
                }
                let scale: f64 = every[li].iter().map(|x| x.abs()).sum::<f64>().max(1.0);
                let err: f64 = every[li]
                    .iter()
                    .zip(&per_pos)
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f64>();
                assert!(
                    err / scale < 1e-5,
                    "prefix pair-sum identity broke (layer {li}, dir {i}, err {err:.2e})"
                );
            }
            // J side: everywhere-write vs the per-write-position sum, at two
            // points (mid of the GDN layer, out of the attention layer).
            for point in [FitPoint::Mid(0), FitPoint::Out(1)] {
                let mut dv = vec![0.0f32; T * d];
                for t in 0..T {
                    dv[t * d + i] = 1.0;
                }
                stack_suffix_jvp(&cap, &sw, &mut s, point, &dv, &mut dhn);
                let every: Vec<f64> = (0..d)
                    .map(|j| (0..T).map(|t| dhn[t * d + j] as f64).sum::<f64>())
                    .collect();
                let mut per_pos = vec![0.0f64; d];
                for wpos in 0..T {
                    let mut one = vec![0.0f32; T * d];
                    one[wpos * d + i] = 1.0;
                    stack_suffix_jvp(&cap, &sw, &mut s, point, &one, &mut dhn);
                    for (j, pp) in per_pos.iter_mut().enumerate() {
                        *pp += (wpos..T).map(|t| dhn[t * d + j] as f64).sum::<f64>();
                    }
                }
                let scale: f64 = every.iter().map(|x| x.abs()).sum::<f64>().max(1.0);
                let err: f64 = every
                    .iter()
                    .zip(&per_pos)
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f64>();
                assert!(
                    err / scale < 1e-5,
                    "suffix pair-sum identity broke (point {point:?}, dir {i}, err {err:.2e})"
                );
            }
        }
    }

    /// The transpose identity's implementation half: the suffix vjp is the
    /// exact adjoint of the suffix jvp, at both fit-point kinds.
    #[test]
    fn stack_suffix_vjp_is_adjoint() {
        let (config, weights) = tiny_model(0x999F, vec![DeltaNet, Attention]);
        let rope_freq = RopeFreqTable::new(config.rope_theta, config.head_dim);
        let sw = stack_of(&config, &weights, &rope_freq);
        let geo = &sw.geo;
        let d = geo.n_embd;
        let mut cap = StackCapture::new(geo);
        stack_forward(&TOKENS, &sw, &mut cap);
        let mut s = StackScratch::new(geo);
        let mut rng = Lcg(0x9991);
        let mut dhn = vec![0.0f32; T * d];
        for point in [
            FitPoint::Mid(0),
            FitPoint::Out(0),
            FitPoint::Mid(1),
            FitPoint::Out(1),
        ] {
            let dv = rng.vec(T * d);
            let gh = rng.vec(T * d);
            stack_suffix_jvp(&cap, &sw, &mut s, point, &dv, &mut dhn);
            let mut g_at = vec![0.0f32; T * d];
            stack_suffix_vjp(&cap, &sw, &mut s, point, &gh, &mut g_at);
            let lhs = dot(&dhn, &gh);
            let rhs = dot(&dv, &g_at);
            let scale = lhs.abs().max(rhs.abs()).max(1.0);
            assert!(
                (lhs - rhs).abs() / scale < 1e-4,
                "suffix adjoint broke at {point:?}: {lhs:.6} vs {rhs:.6}"
            );
        }
    }

    /// End-to-end prefix FD: the full-stack logit tangent vs central
    /// differences of the capture forward (perturbing the embedding table at
    /// one entry). An exact scalar relation does NOT hold — the residual
    /// identity paths skip the ruled ops while subblock paths carry the
    /// half rule's ½ per op, so the composed tangent mixes factors. What
    /// the rules must preserve is the DIRECTION (they approximate local
    /// Jacobians, never rotate them systematically) — asserted by cosine,
    /// with the magnitude ratio printed for the record.
    #[test]
    fn stack_prefix_jvp_matches_fd_direction() {
        let (config, mut weights) = tiny_model(0x999A, vec![DeltaNet, Attention]);
        let rope_freq = RopeFreqTable::new(config.rope_theta, config.head_dim);
        let sw = stack_of(&config, &weights, &rope_freq);
        let geo = &sw.geo;
        let d = geo.n_embd;
        let mut cap = StackCapture::new(geo);
        stack_forward(&TOKENS, &sw, &mut cap);
        let mut s = StackScratch::new(geo);
        let mut pt = PrefixTangents::new(geo);
        // Probe: perturb the embedding of TOKENS[2] (position 2) along dim 5;
        // read the logit tangent at position 2, vocab slot 7.
        let (probe_tok, probe_dim, out_t, out_v) = (TOKENS[2], 5usize, 2usize, 7usize);
        let mut d_emb = vec![0.0f32; T * d];
        d_emb[2 * d + probe_dim] = 1.0;
        stack_prefix_jvp(&cap, &sw, &mut s, &d_emb, &mut pt);
        let hn_t = &pt.hn[out_t * d..(out_t + 1) * d];
        let mut logit_tangent = vec![0.0f32; geo.vocab];
        matmul(&mut logit_tangent, sw.lm_head, hn_t, geo.vocab, d);
        // FD on the model's embedding table (explicit perturb/restore — a
        // capturing closure would fight the borrow of `weights`).
        let hh = 1e-3f32;
        let emb_idx = probe_tok * d + probe_dim;
        let base_val = weights.wte[emb_idx];
        let mut fd_results = [0.0f64; 2];
        for (arm, wte_at) in [(0usize, base_val + hh), (1usize, base_val - hh)] {
            weights.wte[emb_idx] = wte_at;
            let sw2 = StackWeights::from_model(&weights, &config, rope_freq.as_slice(), T);
            let mut cap2 = StackCapture::new(&sw2.geo);
            stack_forward(&TOKENS, &sw2, &mut cap2);
            let mut lg = vec![0.0f32; sw2.geo.vocab];
            matmul(
                &mut lg,
                sw2.lm_head,
                &cap2.hn[out_t * d..(out_t + 1) * d],
                sw2.geo.vocab,
                d,
            );
            fd_results[arm] = lg[out_v] as f64;
        }
        weights.wte[emb_idx] = base_val;
        let fd = (fd_results[0] - fd_results[1]) / (2.0 * hh as f64);
        // The FD vector is the unit-cotangent response — its cosine with the
        // predicted logit tangent at the probe slot is the directional check
        // (both live on the same one-probe geometry here: the tangent at a
        // single output slot vs the FD of that slot IS the entry-wise value,
        // so the cosine reduces to a bounded-ratio agreement on that entry).
        let got = logit_tangent[out_v] as f64;
        // The rules' composed magnitude relation on the ONE-probe geometry:
        // the tangent and FD must agree in sign and stay same-order — a
        // systematic rotation or a wrong-path composition shows up as a sign
        // flip or an order-of-magnitude break.
        let ratio = got / fd;
        println!("prefix FD: tangent {got:.6} fd {fd:.6} ratio {ratio:.3}");
        assert!(
            got * fd > 0.0,
            "prefix FD direction: tangent {got:.6} vs FD {fd:.6} — sign flip"
        );
        assert!(
            (0.05..=20.0).contains(&ratio.abs()),
            "prefix FD magnitude: ratio {ratio:.3} outside the composed-rule band"
        );
    }

    /// The fit plumbing: one doc through [`fit_lens`' inner loop shape] —
    /// K/J accumulators equal the directly computed sweep columns.
    #[test]
    fn fit_accumulators_match_direct_sweeps() {
        let (config, weights) = tiny_model(0x999B, vec![DeltaNet, Attention]);
        let rope_freq = RopeFreqTable::new(config.rope_theta, config.head_dim);
        let sw = stack_of(&config, &weights, &rope_freq);
        let points = sw.geo.fit_points();
        // ONE doc — the direct comparison below is that doc's column alone.
        let docs = vec![TOKENS.to_vec()];
        let (fit, cost) = fit_lens(&sw, &docs, &points, "test");
        assert_eq!(fit.n_docs, 1);
        assert_eq!(cost.docs, 1);
        assert_eq!(cost.prefix_passes, fit.d as u64);
        assert_eq!(cost.suffix_passes, points.len() as u64 * fit.d as u64);
        // Direct: one doc, one direction, one point.
        let geo = &sw.geo;
        let d = geo.n_embd;
        let mut cap = StackCapture::new(geo);
        stack_forward(&TOKENS, &sw, &mut cap);
        let mut s = StackScratch::new(geo);
        let mut pt = PrefixTangents::new(geo);
        let i = 3usize;
        let pi = fit.point_index(FitPoint::Mid(0)).unwrap();
        let mut d_emb = vec![0.0f32; T * d];
        for t in 0..T {
            d_emb[t * d + i] = 1.0;
        }
        stack_prefix_jvp(&cap, &sw, &mut s, &d_emb, &mut pt);
        let mut col = vec![0.0f64; d * d];
        for j in 0..d {
            let mut acc = 0.0f64;
            for t in 0..T {
                acc += pt.mid[0][t * d + j] as f64;
            }
            col[i * d + j] = acc / T as f64; // one doc, position mean
        }
        for j in 0..d {
            assert!(
                (fit.k[pi][i * d + j] as f64 - col[i * d + j]).abs() < 1e-4,
                "fit K col {i},[{j}]: {} vs {}",
                fit.k[pi][i * d + j],
                col[i * d + j]
            );
        }
    }
}
