//! Issue 1214 T1b — the dense-attention integer forward (the model layer).
//!
//! Decision-content transcription of detllm `src/detllm/model.py`
//! (`IntQwen3`) @ `9c3f1c9cdd42294087cc0507992c21dfde944bea` (NO LICENSE
//! upstream — decision-content-only posture; overview `.research/027`).
//! Loads ONLY a prepared integer artifact ([`IntRefArtifact`]) and runs
//! the integer forward: every value on the numerical path is an integer
//! with `(data, m, k)` dyadic bookkeeping (the T0/T1a conventions).
//!
//! Ported: the DYNAMIC-cache dense path — embedding → residual grid →
//! per-layer attention (per-token int8 Q, static per-(kv-head, channel)
//! int8 K straight into the cache, static per-kv-head int8 V, GQA score
//! bands with slot-triangular causality, DI-softmax, exact probs·V) →
//! o-proj → SwiGLU MLP → final norm → LM head → greedy pick.
//!
//! Deliberately NOT ported (performance machinery upstream proves
//! bit-identical to this path; the reference carries the same bits):
//! the static-capacity CUDA-graph cache mode (upstream's own doc: the
//! geometric growth never moves filled slots, so growth cannot affect
//! results — the doubling cache here is result-identical), the Triton
//! fused decode kernel, and the torch.compile seams.
//!
//! Batching/masking law (upstream semantics, §9.3 invariance): RIGHT
//! padding with a bool valid flag per cache slot; masked slots are
//! excluded from softmax max/sum (integer sentinel) and contribute
//! exact zeros to probs·V, so a sequence's outputs are bit-identical
//! regardless of co-batched content (G1 arm:
//! `batch_rows_match_solo_runs_including_padding`). K/V are quantized
//! ONCE (static artifact scales) when produced, then only read —
//! prefill and decode see the same ints by construction (G1 arm:
//! `prefill_decode_split_matches_full_prefill`).
//!
//! Scale provenance — the offline calibration contract (upstream
//! `prepare.py`): the QK smoothing dyadic `c` (`qs_m`/`qs_k`) multiplies
//! Q at runtime and folds into K's static cache scale (`kq_m`/`kq_k`)
//! offline, leaving the per-kv-head score scale (`ks_m`/`ks_k`) with the
//! attention 1/sqrt(head_dim) folded in (upstream `prepare.py`:
//! `kq = c·s_kh` per channel, `ks = s_kh/sqrt(hd)` per head, `qs = c`).
//! The runtime-consumed consequence: the softmax's exp arguments equal
//! `q·k/sqrt(hd)` exactly as scaled — `v(qs)·v(ks)/v(kq) == 1/sqrt(hd)`
//! per channel. Because sqrt(hd) is not dyadic for general hd, this is
//! a FLOAT-calibrated, per-channel-fitted relation with NO exact integer
//! identity — the load path checks shapes and mantissa positivity and
//! trusts the calibration (as upstream does); it does NOT pretend to
//! re-verify the fit. (An earlier exact-identity assert here was WRONG:
//! it dropped the 1/sqrt(hd) fold and would have refused every
//! upstream-faithful artifact.)
//!
//! Integer-only by construction — the module-level no-float guard scans
//! this file.

use super::num::rshift_round;
use super::ops::{
    OUT_FRAC_BITS, P_OUT_BITS, RopeTables, apply_int_rope, di_rmsnorm, di_rmsnorm_gamma,
    di_softmax_row, di_swiglu, int_gemm,
};
use super::tensor::{
    Mat, norm_scale, quant_i8_pertoken, requant_i8_rowcol, requant_i8_static, requant_i32_common,
    residual_add, residual_from_delta,
};

/// Model shape (upstream `config.json` in the artifact dir).
#[derive(Clone, Copy, Debug)]
pub struct IntRefConfig {
    pub n_layers: usize,
    pub hidden: usize,
    pub n_q: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub vocab: usize,
    pub intermediate: usize,
}

/// One int8 GEMM's weights: `w8t` is PRE-TRANSPOSED `[K_in, N]`
/// (upstream `.w8t` — the matrix the GEMM consumes), `m` the
/// per-output-channel mantissas `[N]`, `k` the shared exponent.
#[derive(Clone, Debug)]
pub struct LinearWeights {
    pub w8t: Vec<i8>,
    pub m: Vec<i64>,
    pub k: i64,
}

/// Per-layer static scales (upstream `LayerWeights` + the per-layer
/// constants of the artifact config). Everything here is an OFFLINE
/// calibration output (T1c produces it; T1b consumes it) — NO derived
/// fields: the load-time derivations live on [`IntRefModel`].
#[derive(Clone, Debug)]
pub struct LayerWeights {
    pub q: LinearWeights,
    pub k: LinearWeights,
    pub v: LinearWeights,
    pub o: LinearWeights,
    pub gate: LinearWeights,
    pub up: LinearWeights,
    pub down: LinearWeights,
    /// QK-Norm per-channel dyadic γ (head_dim), folded OUT of the
    /// hidden-dim norms and applied at runtime (upstream `model.py:200-218`).
    pub q_norm_m: Vec<i64>,
    pub q_norm_k: u32,
    pub k_norm_m: Vec<i64>,
    pub k_norm_k: u32,
    /// QK smoothing per-channel dyadic c — folded into K's static cache
    /// scale offline (`model.py:319-324`); multiplying Q by it cancels
    /// identically in the score dot product. Length `n_kv * head_dim`.
    /// Contract (calibration-side, upstream `prepare.py`): `v(kq)[c] =
    /// v(qs)[c]·sqrt(hd)·v(ks)[kvh]` so the softmax consumes `q·k/sqrt(hd)`.
    pub qs_m: Vec<i64>,
    pub qs_k: u32,
    /// Attention-input per-channel smoothing (1/s; s folded into the
    /// consumer weights offline; `model.py:187-198`). Length `hidden`.
    pub as_m: Vec<i64>,
    pub as_k: u32,
    /// MLP-input smoothing. Length `hidden`.
    pub ms_m: Vec<i64>,
    pub ms_k: u32,
    /// Down-input smoothing. Length `intermediate`.
    pub ds_m: Vec<i64>,
    pub ds_k: u32,
    /// Merge-head o-input smoothing (per q-head channel). Length
    /// `n_q * head_dim`.
    pub os_m: Vec<i64>,
    pub os_k: i64,
    /// K's static per-(kv-head, channel) cache scale (c folded). Length
    /// `n_kv * head_dim`.
    pub kq_m: Vec<i64>,
    pub kq_k: Vec<i64>,
    /// Score scale per kv-head (the K cache scale the QK dot product
    /// consumes; 1/sqrt(head_dim) folded upstream). Length `n_kv`.
    pub ks_m: Vec<i64>,
    pub ks_k: Vec<i64>,
    /// V's static per-kv-head cache scale. Length `n_kv`.
    pub vq_m: Vec<i64>,
    pub vq_k: Vec<i64>,
}

/// The whole calibrated integer artifact (upstream `model.safetensors` +
/// `config.json`, in memory).
#[derive(Clone, Debug)]
pub struct IntRefArtifact {
    pub cfg: IntRefConfig,
    /// int8 embedding table `[vocab, hidden]` with per-row dyadic scale.
    pub embed_e8: Vec<i8>,
    pub embed_m: Vec<i64>,
    pub embed_k: Vec<i64>,
    /// LM head (`head.w8t` etc.).
    pub head: LinearWeights,
    /// Offline cos/sin tables `[max_pos, head_dim]` int16 (the float
    /// builder is T1c/offline; tests hand-craft tables).
    pub rope: RopeTables,
    pub layers: Vec<LayerWeights>,
}

/// The KV cache (upstream `KVCache`, dynamic mode): per-layer int8 K/V
/// planes `[B, n_kv, CAP, head_dim]` plus the per-slot validity mask.
/// Capacity is RESULT-INERT (upstream doc: growth never moves filled
/// slots, so it cannot affect results); slots grow by doubling.
pub struct KvCache {
    pub k8: Vec<Vec<i8>>,
    pub v8: Vec<Vec<i8>>,
    pub valid: Vec<Vec<bool>>,
    /// Slots written before the current chunk (`S_past`).
    pub past: usize,
    b: usize,
    n_kv: usize,
    hd: usize,
}

impl KvCache {
    pub fn new(b: usize, n_layers: usize, n_kv: usize, hd: usize) -> Self {
        KvCache {
            k8: vec![Vec::new(); n_layers],
            v8: vec![Vec::new(); n_layers],
            valid: vec![Vec::new(); b],
            past: 0,
            b,
            n_kv,
            hd,
        }
    }

    /// Ensure the per-layer planes hold `need` slots, growing by
    /// doubling (content-preserving: slots never move).
    fn ensure_layer(&mut self, li: usize, need: usize) {
        let plane = self.b * self.n_kv * self.hd;
        let cap = self.k8[li].len() / plane;
        if need <= cap {
            return;
        }
        let mut new_cap = cap.max(4);
        while new_cap < need {
            new_cap *= 2;
        }
        let mut k = vec![0_i8; new_cap * plane];
        let mut v = vec![0_i8; new_cap * plane];
        k[..cap * plane].copy_from_slice(&self.k8[li]);
        v[..cap * plane].copy_from_slice(&self.v8[li]);
        self.k8[li] = k;
        self.v8[li] = v;
    }

    /// Extend the validity masks to `need` slots (false = unwritten).
    fn ensure_valid(&mut self, need: usize) {
        for row in self.valid.iter_mut() {
            if row.len() < need {
                row.resize(need, false);
            }
        }
    }

    /// Slots per layer plane the buffers currently hold (the ONE slot
    /// stride — writes and reads both go through this).
    fn capacity(&self, li: usize) -> usize {
        self.k8[li].len() / (self.b * self.n_kv * self.hd)
    }

    /// Slots written before the current chunk.
    pub fn seq_len(&self) -> usize {
        self.past
    }
}

/// The linear-GEMM seam: `a [m, k] · w8t [k, n] → [m, n] int32`. The
/// CPU forward installs [`int_gemm`]; the G1(c) arm installs the GPU
/// twin — the two MUST agree bit-for-bit (zero tolerance), which is
/// exactly what `tests/int_ref_real_artifact.rs` receipts on the real
/// model.
pub type GemmFn<'a> = dyn Fn(&[i8], usize, usize, &[i8], usize) -> Vec<i32> + 'a;

/// Greedy pick over one logits row: `argmax(logits·head_m)` with the
/// PINNED first-index tie rule (upstream's lowest-token-id tie-break;
/// katgpt-rs Issue 940 law). `logits_row` is the int32 GEMM output for
/// one position; the per-vocab column scale `head_m` must be applied
/// (the positive row scale is argmax-invariant; column scales differ).
pub fn greedy_pick(logits_row: &[i32], head_m: &[i64]) -> usize {
    assert_eq!(
        logits_row.len(),
        head_m.len(),
        "greedy_pick: row/head_m length"
    );
    let mut best = 0_usize;
    let mut best_v = i64::MIN;
    for (i, (&l, &m)) in logits_row.iter().zip(head_m.iter()).enumerate() {
        let scaled = (i64::from(l))
            .checked_mul(m)
            .expect("greedy_pick: logits·head_m overflow");
        if scaled > best_v {
            best_v = scaled;
            best = i;
        }
    }
    best
}

/// Load-time shape check for one [`LinearWeights`].
fn lin_check(what: &str, l: &LinearWeights, k_in: usize, n_out: usize) {
    assert_eq!(l.w8t.len(), k_in * n_out, "{what}: w8t shape");
    assert_eq!(l.m.len(), n_out, "{what}: per-channel scale length");
    assert!(
        l.m.iter().all(|&m| m > 0),
        "{what}: scale mantissas must be positive"
    );
}

/// The integer model: the artifact + its load-time derived scales,
/// ready to run.
pub struct IntRefModel {
    a: IntRefArtifact,
    group: usize,
    /// Attention-output merged col scale per layer: per-q-head V scales
    /// aligned to a shared exponent by integer shifts of the artifact
    /// constants (`model.py:188-201`).
    attn_cols: Vec<Vec<i64>>,
    /// The shared exponent per layer: `max(vq_k) + (P_OUT_BITS - 1)`.
    attn_ks: Vec<i64>,
    /// K's row exponent entering the static requant per layer:
    /// `OUT_FRAC_BITS + k_norm_k` (`model.py:203`).
    k_rowks: Vec<i64>,
}

impl IntRefModel {
    /// Load-time derivation and validation (`model.py:124-206`): the
    /// merged attention col scale, the K row exponent, the shape
    /// checks, and the qs/kq/ks calibration contract (a violation would
    /// make the score bookkeeping silently wrong — refused loud).
    pub fn new(a: IntRefArtifact) -> Self {
        let c = a.cfg;
        assert!(
            c.n_layers >= 1 && c.hidden >= 1 && c.vocab >= 1,
            "int_ref: empty shape"
        );
        assert!(
            c.n_kv >= 1 && c.n_q.is_multiple_of(c.n_kv),
            "int_ref: n_q must be a nonzero multiple of n_kv"
        );
        assert!(
            c.head_dim >= 2 && c.head_dim.is_multiple_of(2),
            "int_ref: head_dim must be even (rope pairing)"
        );
        assert!(
            (c.head_dim as i64) * 127 * 127 < (1_i64 << 31)
                && (c.intermediate as i64) * 127 * 127 < (1_i64 << 31),
            "int_ref: head_dim/intermediate breach the int32 GEMM bound (x·127^2 < 2^31)"
        );
        let group = c.n_q / c.n_kv;
        let kv_dim = c.n_kv * c.head_dim;
        let q_dim = c.n_q * c.head_dim;
        assert_eq!(
            a.rope.head_dim, c.head_dim,
            "int_ref: rope head_dim mismatch"
        );
        assert_eq!(a.embed_e8.len(), c.vocab * c.hidden, "int_ref: embed shape");
        assert_eq!(a.embed_m.len(), c.vocab, "int_ref: embed scale length");
        assert_eq!(a.embed_k.len(), c.vocab, "int_ref: embed exponent length");
        assert!(
            a.embed_m.iter().all(|&m| m > 0),
            "int_ref: embed mantissas must be positive"
        );
        lin_check("head", &a.head, c.hidden, c.vocab);
        let mut attn_cols = Vec::with_capacity(c.n_layers);
        let mut attn_ks = Vec::with_capacity(c.n_layers);
        let mut k_rowks = Vec::with_capacity(c.n_layers);
        for (li, lw) in a.layers.iter().enumerate() {
            lin_check("q", &lw.q, c.hidden, q_dim);
            lin_check("k", &lw.k, c.hidden, kv_dim);
            lin_check("v", &lw.v, c.hidden, kv_dim);
            lin_check("o", &lw.o, q_dim, c.hidden);
            lin_check("gate", &lw.gate, c.hidden, c.intermediate);
            lin_check("up", &lw.up, c.hidden, c.intermediate);
            lin_check("down", &lw.down, c.intermediate, c.hidden);
            assert_eq!(lw.q_norm_m.len(), c.head_dim, "layer {li}: q_norm length");
            assert_eq!(lw.k_norm_m.len(), c.head_dim, "layer {li}: k_norm length");
            assert_eq!(lw.qs_m.len(), kv_dim, "layer {li}: qs length");
            assert_eq!(lw.kq_m.len(), kv_dim, "layer {li}: kq length");
            assert_eq!(lw.kq_k.len(), kv_dim, "layer {li}: kq_k length");
            assert_eq!(lw.ks_m.len(), c.n_kv, "layer {li}: ks length");
            assert_eq!(lw.ks_k.len(), c.n_kv, "layer {li}: ks_k length");
            assert_eq!(lw.vq_m.len(), c.n_kv, "layer {li}: vq length");
            assert_eq!(lw.vq_k.len(), c.n_kv, "layer {li}: vq_k length");
            assert_eq!(lw.as_m.len(), c.hidden, "layer {li}: as length");
            assert_eq!(lw.ms_m.len(), c.hidden, "layer {li}: ms length");
            assert_eq!(lw.ds_m.len(), c.intermediate, "layer {li}: ds length");
            assert_eq!(lw.os_m.len(), q_dim, "layer {li}: os length");
            let scales_ok = lw
                .qs_m
                .iter()
                .chain(lw.kq_m.iter())
                .chain(lw.ks_m.iter())
                .chain(lw.vq_m.iter())
                .chain(lw.as_m.iter())
                .chain(lw.ms_m.iter())
                .chain(lw.ds_m.iter())
                .chain(lw.os_m.iter())
                .all(|&m| m > 0);
            assert!(scales_ok, "layer {li}: scale mantissas must be positive");
            let vk_shared = lw.vq_k.iter().copied().max().unwrap_or(0);
            let mut cols = Vec::with_capacity(q_dim);
            for qh in 0..c.n_q {
                let kvh = qh / group;
                let shift =
                    u32::try_from(vk_shared - lw.vq_k[kvh]).expect("int_ref: negative vq shift");
                for _ in 0..c.head_dim {
                    cols.push(
                        lw.vq_m[kvh]
                            .checked_shl(shift)
                            .expect("int_ref: attn col scale overflow"),
                    );
                }
            }
            attn_cols.push(cols);
            attn_ks.push(vk_shared + i64::from(P_OUT_BITS) - 1);
            k_rowks.push(i64::from(OUT_FRAC_BITS) + i64::from(lw.k_norm_k));
        }
        Self {
            a,
            group,
            attn_cols,
            attn_ks,
            k_rowks,
        }
    }

    pub fn config(&self) -> &IntRefConfig {
        &self.a.cfg
    }

    /// The LM head's per-vocab mantissas (consumers computing logit
    /// values and greedy picks need them).
    pub fn head_m(&self) -> &[i64] {
        &self.a.head.m
    }

    /// The LM head's shared exponent (pairs [`head_m`](Self::head_m):
    /// real logit ≈ `logits · xm/2^xk · head_m[v]/2^head_k`).
    pub fn head_k(&self) -> i64 {
        self.a.head.k
    }

    fn layer(&self, li: usize) -> &LayerWeights {
        &self.a.layers[li]
    }

    /// `_norm_quant` (`model.py:187-198`): RMSNorm (γ folded into the
    /// consumer weights offline), optional per-channel activation
    /// smoothing multiply, then per-token int8 quant. `h` is
    /// `[rows, hidden]` int64 (the residual grid widened); the residual
    /// scale cancels in the norm entirely. The claimed row scale after a
    /// smoothing multiply is `(1, OUT_FRAC_BITS + sm_k)` — exact, because
    /// the consumer weights carry the matching `1/s` fold offline.
    fn norm_quant(
        &self,
        h: Mat<'_, i64>,
        sm_m: Option<&[i64]>,
        sm_k: u32,
    ) -> (Vec<i8>, Vec<i64>, Vec<i64>) {
        let y32 = di_rmsnorm(h);
        let rows = h.rows;
        let cols = h.cols;
        let ones = vec![1_i64; rows];
        match sm_m {
            None => {
                let y: Vec<i64> = y32.iter().map(|&v| i64::from(v)).collect();
                let kq = vec![i64::from(OUT_FRAC_BITS); rows];
                quant_i8_pertoken(Mat::from((y.as_slice(), rows, cols)), &ones, &kq)
            }
            Some(sm) => {
                assert_eq!(sm.len(), cols, "norm_quant: smoothing length");
                let mut y: Vec<i64> = Vec::with_capacity(y32.len());
                for row in y32.chunks(cols) {
                    for (v, m) in row.iter().zip(sm.iter()) {
                        y.push(
                            i64::from(*v)
                                .checked_mul(*m)
                                .expect("norm_quant: smoothing overflow"),
                        );
                    }
                }
                let kq = vec![i64::from(OUT_FRAC_BITS) + i64::from(sm_k); rows];
                quant_i8_pertoken(Mat::from((y.as_slice(), rows, cols)), &ones, &kq)
            }
        }
    }

    /// `_qk_head_pipeline` (`model.py:200-218`): per-(token, head)
    /// requant at 16-bit magnitude → QK-Norm(γ) → int RoPE. `p` is the
    /// int32 GEMM output `[B*T, n_heads*head_dim]`; returns int64
    /// `[B, n_heads, T, head_dim]` at fixed scale `(1, OUT_FRAC_BITS +
    /// gamma_k)` (the requant m/k are dropped — the norm is scale-free).
    #[allow(clippy::too_many_arguments)]
    fn qk_head_pipeline(
        &self,
        p: &[i32],
        rows: usize,
        n_heads: usize,
        hd: usize,
        xm: &[i64],
        xk: &[i64],
        w_m: &[i64],
        w_k: i64,
        gamma_m: &[i64],
        gamma_k: u32,
        positions: &[usize],
        b: usize,
        t: usize,
    ) -> (Vec<i64>, u32) {
        let pw: Vec<i64> = p.iter().map(|&v| i64::from(v)).collect();
        let (y32, _m, _k) = requant_i32_common(
            Mat::from((pw.as_slice(), rows, n_heads * hd)),
            xm,
            xk,
            w_m,
            w_k,
            16,
        );
        // the [rows, n_heads*hd] layout IS [rows*n_heads, hd] row-major
        let y: Vec<i64> = y32.iter().map(|&v| i64::from(v)).collect();
        let (yn, k_out) = di_rmsnorm_gamma(
            Mat::from((y.as_slice(), rows * n_heads, hd)),
            gamma_m,
            gamma_k,
        );
        // permute [B, T, H, hd] -> [B, H, T, hd]; positions ride (b, t)
        let mut perm = Vec::with_capacity(yn.len());
        let mut pos_exp = Vec::with_capacity(yn.len() / hd);
        for bi in 0..b {
            for hi in 0..n_heads {
                for ti in 0..t {
                    let src = ((bi * t + ti) * n_heads + hi) * hd;
                    perm.extend_from_slice(&yn[src..src + hd]);
                    pos_exp.push(positions[bi * t + ti]);
                }
            }
        }
        let permuted = Mat::from((perm.as_slice(), b * n_heads * t, hd));
        let out = apply_int_rope(permuted, &self.a.rope, &pos_exp);
        (out, k_out)
    }

    /// One transformer layer (attention + MLP), residual in/out
    /// (`model.py:288-427`). `h` is `[B*T, hidden]` int64 (the residual
    /// grid widened from int32), `k_res` per-row; `chunk_valid` is the
    /// right-padding flag per `[B, T]` token; cache writes and reads
    /// both go through the plane's CAPACITY stride (never the needed
    /// count — capacity is result-inert, stride drift is not).
    #[allow(clippy::too_many_arguments)]
    fn run_layer(
        &self,
        gemm: &GemmFn<'_>,
        li: usize,
        h: Mat<'_, i64>,
        k_res: &[i64],
        positions: &[usize],
        chunk_valid: &[bool],
        cache: &mut KvCache,
        b: usize,
        t: usize,
    ) -> (Vec<i32>, Vec<i64>) {
        let c = self.config();
        let lw = self.layer(li);
        let hd = c.head_dim;
        let group = self.group;
        let rows = b * t;
        let past = cache.past;
        let q_dim = c.n_q * hd;
        let kv_dim = c.n_kv * hd;

        // ---- attention ----
        let (x8, xm, xk) = self.norm_quant(h, Some(lw.as_m.as_slice()), lw.as_k);
        let p_q = gemm(x8.as_slice(), rows, c.hidden, lw.q.w8t.as_slice(), q_dim);
        let p_k = gemm(x8.as_slice(), rows, c.hidden, lw.k.w8t.as_slice(), kv_dim);
        let p_v = gemm(x8.as_slice(), rows, c.hidden, lw.v.w8t.as_slice(), kv_dim);

        let (q_raw, qkk) = self.qk_head_pipeline(
            p_q.as_slice(),
            rows,
            c.n_q,
            hd,
            &xm,
            &xk,
            lw.q.m.as_slice(),
            lw.q.k,
            lw.q_norm_m.as_slice(),
            lw.q_norm_k,
            positions,
            b,
            t,
        );
        let (kk, _kkk) = self.qk_head_pipeline(
            p_k.as_slice(),
            rows,
            c.n_kv,
            hd,
            &xm,
            &xk,
            lw.k.m.as_slice(),
            lw.k.k,
            lw.k_norm_m.as_slice(),
            lw.k_norm_k,
            positions,
            b,
            t,
        );

        // QK smoothing: multiply Q by the per-channel dyadic c (exact
        // int64; c is folded into K's static cache scale offline, so it
        // cancels identically in the score dot product — the contract
        // new() asserts).
        let q_rows = b * c.n_q * t;
        let mut qs = Vec::with_capacity(q_raw.len());
        for row in 0..q_rows {
            let qh = (row / t) % c.n_q;
            let kvh = qh / group;
            for (qd, qv) in q_raw[row * hd..(row + 1) * hd].iter().enumerate() {
                qs.push(
                    qv.checked_mul(lw.qs_m[kvh * hd + qd])
                        .expect("int_ref: q smoothing overflow"),
                );
            }
        }
        // Q's claimed row exponent: the Q pipeline's k + the smoothing k
        // (exact — K's weights carry the matching fold offline)
        let qkk_s = i64::from(qkk) + i64::from(lw.qs_k);
        // Q: dynamic per-(token, head) int8
        let ones_q = vec![1_i64; q_rows];
        let kq_vec = vec![qkk_s; q_rows];
        let (q8, qm, qk) =
            quant_i8_pertoken(Mat::from((qs.as_slice(), q_rows, hd)), &ones_q, &kq_vec);
        // K: static per-(head, channel) int8 (quantized ONCE, straight
        // into the cache; row (1, k_rowk), col (1, 0), target kq)
        let kn_rows = b * c.n_kv * t;
        let ones_k = vec![1_i64; kn_rows];
        let krowk = vec![self.k_rowks[li]; kn_rows];
        let ones_c = vec![1_i64; hd];
        let mut k_sm = Vec::with_capacity(kn_rows * hd);
        let mut k_sk = Vec::with_capacity(kn_rows * hd);
        for row in 0..kn_rows {
            let kvh = (row / t) % c.n_kv;
            for d in 0..hd {
                k_sm.push(lw.kq_m[kvh * hd + d]);
                k_sk.push(lw.kq_k[kvh * hd + d]);
            }
        }
        let k8 = requant_i8_static(
            Mat::from((kk.as_slice(), kn_rows, hd)),
            &ones_k,
            &krowk,
            &ones_c,
            0,
            &k_sm,
            &k_sk,
        );
        // V: static per-kv-head int8 from the GEMM output directly —
        // rows (xm, xk), cols PER-CHANNEL v.m, target per-kv-head vq
        let pv64: Vec<i64> = p_v.iter().map(|&v| i64::from(v)).collect();
        let mut v_m_per = Vec::with_capacity(kv_dim);
        for kvh in 0..c.n_kv {
            for d in 0..hd {
                v_m_per.push(lw.v.m[kvh * hd + d]);
            }
        }
        let mut v_sm = Vec::with_capacity(rows * kv_dim);
        let mut v_sk = Vec::with_capacity(rows * kv_dim);
        for _r in 0..rows {
            for kvh in 0..c.n_kv {
                for _d in 0..hd {
                    v_sm.push(lw.vq_m[kvh]);
                    v_sk.push(lw.vq_k[kvh]);
                }
            }
        }
        let v8_flat = requant_i8_static(
            Mat::from((pv64.as_slice(), rows, kv_dim)),
            &xm,
            &xk,
            &v_m_per,
            lw.v.k,
            &v_sm,
            &v_sk,
        );
        // permute [B, T, n_kv, hd] -> [B, n_kv, T, hd] for the cache
        let mut v8 = vec![0_i8; v8_flat.len()];
        for bi in 0..b {
            for kvh in 0..c.n_kv {
                for ti in 0..t {
                    let src = ((bi * t + ti) * c.n_kv + kvh) * hd;
                    let dst = ((bi * c.n_kv + kvh) * t + ti) * hd;
                    v8[dst..dst + hd].copy_from_slice(&v8_flat[src..src + hd]);
                }
            }
        }

        // cache writes at slots [past .. past+t), CAPACITY stride
        let need = past + t;
        cache.ensure_layer(li, need);
        cache.ensure_valid(need);
        let cap = cache.capacity(li);
        for bi in 0..b {
            for kvh in 0..c.n_kv {
                for ti in 0..t {
                    let src = ((bi * c.n_kv + kvh) * t + ti) * hd;
                    let dst = ((bi * c.n_kv + kvh) * cap + past + ti) * hd;
                    let kc = &mut cache.k8[li];
                    kc[dst..dst + hd].copy_from_slice(&k8[src..src + hd]);
                    let vc = &mut cache.v8[li];
                    vc[dst..dst + hd].copy_from_slice(&v8[src..src + hd]);
                }
            }
            cache.valid[bi][past..past + t].copy_from_slice(&chunk_valid[bi * t..(bi + 1) * t]);
        }

        // ---- scores → DI-softmax → context ----
        // per (b, qh, t): integer dot products over the valid slot band;
        // GQA indexes kvh = qh/group directly (upstream's decode path
        // does the same — no repeat_interleave materialization;
        // bit-identical by integer associativity).
        let slots = need;
        let mut merge = vec![0_i64; rows * q_dim];
        let mut scores_buf = vec![0_i64; slots];
        let mut mask_row = vec![false; slots];
        let mut attn_buf = vec![0_i64; hd];
        for bi in 0..b {
            for qh in 0..c.n_q {
                let kvh = qh / group;
                for ti in 0..t {
                    let qrow = ((bi * c.n_q + qh) * t + ti) * hd;
                    let band = past + ti + 1;
                    for (s, slot) in scores_buf[..band].iter_mut().enumerate() {
                        let mut acc: i64 = 0;
                        if cache.valid[bi][s] {
                            let kbase = ((bi * c.n_kv + kvh) * cap + s) * hd;
                            for d in 0..hd {
                                acc = acc
                                    .checked_add(
                                        i64::from(q8[qrow + d])
                                            * i64::from(cache.k8[li][kbase + d]),
                                    )
                                    .expect("int_ref: score dot overflow");
                            }
                        }
                        *slot = acc;
                    }
                    for (mr, vr) in mask_row[..band].iter_mut().zip(cache.valid[bi].iter()) {
                        *mr = *vr;
                    }
                    // score scale: (qm·ks_m[kvh], qk + ks_k[kvh]),
                    // normalized to a 16-bit mantissa window (the asserts
                    // mirror upstream's lshift_t handling of negative k)
                    let qidx = (bi * c.n_q + qh) * t + ti;
                    let mut sm = qm[qidx]
                        .checked_mul(lw.ks_m[kvh])
                        .expect("int_ref: score scale overflow");
                    let mut sk = qk[qidx] + lw.ks_k[kvh];
                    norm_scale(
                        std::slice::from_mut(&mut sm),
                        std::slice::from_mut(&mut sk),
                        16,
                    );
                    // the int64 wall: di_exp computes 2^k, so the claimed
                    // exponent must stay ≤ 62. norm_scale(16) normally pins
                    // k far below, but a degenerate (≈zero) row's fit can
                    // push it past — renormalize the pair by shifting m down
                    // (value-preserving up to one rshift_round rounding).
                    if sk > 62 {
                        let drop =
                            u32::try_from(sk - 62).expect("int_ref: score exponent overflow");
                        sm = rshift_round(sm, drop);
                        sk = 62;
                    }
                    assert!(sk >= 0, "int_ref: score exponent underflowed past zero");
                    let sk_u = u32::try_from(sk).expect("int_ref: score exponent overflow");
                    let probs = di_softmax_row(
                        &scores_buf[..band],
                        sm,
                        sk_u,
                        &mask_row[..band],
                        P_OUT_BITS,
                    );
                    // context: probs × V8 exact in int64 — the same value
                    // upstream's hi/lo int8 split GEMM produces (integer
                    // associativity; the split is a GPU int8-GEMM
                    // accommodation, exactness pinned by the matmul tests)
                    attn_buf.fill(0);
                    for (s, &p32) in probs[..band].iter().enumerate() {
                        let p = i64::from(p32);
                        if p == 0 {
                            continue;
                        }
                        let vbase = ((bi * c.n_kv + kvh) * cap + s) * hd;
                        for (d, av) in attn_buf.iter_mut().enumerate() {
                            *av = av
                                .checked_add(p * i64::from(cache.v8[li][vbase + d]))
                                .expect("int_ref: context accumulation overflow");
                        }
                    }
                    let dst = ((bi * t + ti) * c.n_q + qh) * hd;
                    merge[dst..dst + hd].copy_from_slice(&attn_buf);
                }
            }
        }

        // merge-head requant (o-input smoothing multiply, then int8;
        // row scale starts at exact 1/2^os_k: the probs k is folded in
        // the col scale — merge·col_m/2^col_k == Σ p_true·v_true exactly)
        let mut os_p = Vec::with_capacity(merge.len());
        for row in merge.chunks(q_dim) {
            for (v, m) in row.iter().zip(lw.os_m.iter()) {
                os_p.push(v.checked_mul(*m).expect("int_ref: os_m overflow"));
            }
        }
        let ones_r = vec![1_i64; rows];
        let rk0 = vec![lw.os_k; rows];
        let (a8, am, ak) = requant_i8_rowcol(
            Mat::from((os_p.as_slice(), rows, q_dim)),
            &ones_r,
            &rk0,
            self.attn_cols[li].as_slice(),
            self.attn_ks[li],
        );
        let p_o = gemm(a8.as_slice(), rows, q_dim, lw.o.w8t.as_slice(), c.hidden);
        let po64: Vec<i64> = p_o.iter().map(|&v| i64::from(v)).collect();
        let (d32, dm, dk) = requant_i32_common(
            Mat::from((po64.as_slice(), rows, c.hidden)),
            &am,
            &ak,
            lw.o.m.as_slice(),
            lw.o.k,
            20,
        );
        let d64: Vec<i64> = d32.iter().map(|&v| i64::from(v)).collect();
        let (h2, k2) = residual_add(
            h,
            k_res,
            Mat::from((d64.as_slice(), rows, c.hidden)),
            &dm,
            &dk,
        );

        // ---- MLP ----
        self.mlp_tail(gemm, li, h2.as_slice(), &k2, rows)
    }

    /// The MLP tail (`model.py:404-425`): norm+smooth → gate/up GEMMs →
    /// requant at 14-bit magnitude each → DI-SwiGLU (the requant OUTPUTS'
    /// per-row m/k feed the op) → down-input smoothing multiply →
    /// dynamic per-token int8 → down GEMM → requant at 20 bits →
    /// residual add.
    fn mlp_tail(
        &self,
        gemm: &GemmFn<'_>,
        li: usize,
        h32: &[i32],
        k_res: &[i64],
        rows: usize,
    ) -> (Vec<i32>, Vec<i64>) {
        let c = self.config();
        let lw = self.layer(li);
        let h64: Vec<i64> = h32.iter().map(|&v| i64::from(v)).collect();
        let hmat = Mat::from((h64.as_slice(), rows, c.hidden));
        let (x8, xm, xk) = self.norm_quant(hmat, Some(lw.ms_m.as_slice()), lw.ms_k);
        let p_g = (gemm)(
            x8.as_slice(),
            rows,
            c.hidden,
            lw.gate.w8t.as_slice(),
            c.intermediate,
        );
        let p_u = (gemm)(
            x8.as_slice(),
            rows,
            c.hidden,
            lw.up.w8t.as_slice(),
            c.intermediate,
        );
        let pg64: Vec<i64> = p_g.iter().map(|&v| i64::from(v)).collect();
        let (gate32, gxm, gxk) = requant_i32_common(
            Mat::from((pg64.as_slice(), rows, c.intermediate)),
            &xm,
            &xk,
            lw.gate.m.as_slice(),
            lw.gate.k,
            14,
        );
        // up uses ITS OWN col scale
        let pu64: Vec<i64> = p_u.iter().map(|&v| i64::from(v)).collect();
        let (up32, uxm, uxk) = requant_i32_common(
            Mat::from((pu64.as_slice(), rows, c.intermediate)),
            &xm,
            &xk,
            lw.up.m.as_slice(),
            lw.up.k,
            14,
        );
        let gate64: Vec<i64> = gate32.iter().map(|&v| i64::from(v)).collect();
        let up64: Vec<i64> = up32.iter().map(|&v| i64::from(v)).collect();
        let gk_u32: Vec<u32> = gxk
            .iter()
            .map(|&k| u32::try_from(k).expect("int_ref: gate exponent negative"))
            .collect();
        let uk_u32: Vec<u32> = uxk
            .iter()
            .map(|&k| u32::try_from(k).expect("int_ref: up exponent negative"))
            .collect();
        let (y, ym, yk) = di_swiglu(
            Mat::from((gate64.as_slice(), rows, c.intermediate)),
            Mat::from((up64.as_slice(), rows, c.intermediate)),
            &gxm,
            &gk_u32,
            &uxm,
            &uk_u32,
        );
        // down-input smoothing multiply, then dynamic per-token int8
        // (the claimed row scale (ym, yk + ds_k) is exact — the down
        // weights carry the matching 1/s fold offline)
        let mut y_ds = Vec::with_capacity(y.len());
        for row in y.chunks(c.intermediate) {
            for (v, m) in row.iter().zip(lw.ds_m.iter()) {
                y_ds.push(v.checked_mul(*m).expect("int_ref: ds_m overflow"));
            }
        }
        let ydk: Vec<i64> = yk
            .iter()
            .map(|&k| i64::from(k) + i64::from(lw.ds_k))
            .collect();
        let (y8, y8m, y8k) = quant_i8_pertoken(
            Mat::from((y_ds.as_slice(), rows, c.intermediate)),
            &ym,
            &ydk,
        );
        let p_d = (gemm)(
            y8.as_slice(),
            rows,
            c.intermediate,
            lw.down.w8t.as_slice(),
            c.hidden,
        );
        let pd64: Vec<i64> = p_d.iter().map(|&v| i64::from(v)).collect();
        let (d32, dm, dk) = requant_i32_common(
            Mat::from((pd64.as_slice(), rows, c.hidden)),
            &y8m,
            &y8k,
            lw.down.m.as_slice(),
            lw.down.k,
            20,
        );
        let d64: Vec<i64> = d32.iter().map(|&v| i64::from(v)).collect();
        residual_add(
            hmat,
            k_res,
            Mat::from((d64.as_slice(), rows, c.hidden)),
            &dm,
            &dk,
        )
    }

    /// `forward` (`model.py:220-286`): ids `[B, T]`, positions `[B, T]`,
    /// optional `chunk_valid` `[B, T]` (None = all real), optional cache.
    /// Returns int32 logits `[B*T, vocab]` plus the per-row output scale
    /// `(xm, xk)`: logit value ≈ `logits · xm·head_m[v] / 2^(xk+head_k)`.
    /// The CPU forward — every linear through [`int_gemm`]. The receipt
    /// path; see [`forward_gemm`](Self::forward_gemm) for the G1(c) GPU
    /// twin.
    pub fn forward(
        &self,
        ids: &[u32],
        positions: &[usize],
        chunk_valid: Option<&[bool]>,
        cache: Option<&mut KvCache>,
        b: usize,
        t: usize,
    ) -> (Vec<i32>, Vec<i64>, Vec<i64>) {
        self.forward_gemm(
            ids,
            positions,
            chunk_valid,
            cache,
            b,
            t,
            &|a, m, k, w, n| int_gemm(a, m, k, w, n),
        )
    }

    /// The forward with the LINEAR GEMMs supplied by `gemm` — the G1(c)
    /// seam: the whole pipeline runs with the GPU twin
    /// (`IntGemmGpu::gemm`) in place of [`int_gemm`] and MUST produce
    /// bit-identical logits (zero tolerance). Everything between the
    /// linears (norms, attention bookkeeping, quant/requant, softmax,
    /// SwiGLU) stays the CPU integer ops — the arm isolates the GEMM as
    /// the only substitution.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_gemm(
        &self,
        ids: &[u32],
        positions: &[usize],
        chunk_valid: Option<&[bool]>,
        cache: Option<&mut KvCache>,
        b: usize,
        t: usize,
        gemm: &GemmFn<'_>,
    ) -> (Vec<i32>, Vec<i64>, Vec<i64>) {
        let c = self.config();
        assert!(b >= 1 && t >= 1, "forward: empty batch or chunk");
        assert_eq!(ids.len(), b * t, "forward: ids shape");
        assert_eq!(positions.len(), b * t, "forward: positions shape");
        let owned_valid;
        let chunk_valid: &[bool] = match chunk_valid {
            Some(v) => {
                assert_eq!(v.len(), b * t, "forward: chunk_valid shape");
                v
            }
            None => {
                owned_valid = vec![true; b * t];
                &owned_valid
            }
        };
        let mut own_cache;
        let cache: &mut KvCache = match cache {
            Some(ca) => {
                assert_eq!(ca.b, b, "forward: cache batch mismatch");
                ca
            }
            None => {
                own_cache = KvCache::new(b, c.n_layers, c.n_kv, c.head_dim);
                &mut own_cache
            }
        };

        // embedding lookup → residual grid
        let rows = b * t;
        let mut e8 = Vec::with_capacity(rows * c.hidden);
        let mut em = Vec::with_capacity(rows);
        let mut ek = Vec::with_capacity(rows);
        for &id in ids {
            let id = id as usize;
            assert!(id < c.vocab, "forward: id {id} past vocab");
            e8.extend_from_slice(&self.a.embed_e8[id * c.hidden..(id + 1) * c.hidden]);
            em.push(self.a.embed_m[id]);
            ek.push(self.a.embed_k[id]);
        }
        let e64: Vec<i64> = e8.iter().map(|&v| i64::from(v)).collect();
        let (h32, k_res) =
            residual_from_delta(Mat::from((e64.as_slice(), rows, c.hidden)), &em, &ek);
        // the residual grid is int32 on the wire (the renorm window);
        // each layer widens it to i64 at its op entry
        let mut h32 = h32;
        let mut k_res = k_res;

        for li in 0..c.n_layers {
            let h64: Vec<i64> = h32.iter().map(|&v| i64::from(v)).collect();
            let (h2, k2) = self.run_layer(
                gemm,
                li,
                Mat::from((h64.as_slice(), rows, c.hidden)),
                &k_res,
                positions,
                chunk_valid,
                cache,
                b,
                t,
            );
            h32 = h2;
            k_res = k2;
        }
        // the chunk is committed (upstream sets cache.length = S after
        // the layer loop in dynamic mode)
        cache.past += t;

        let h64: Vec<i64> = h32.iter().map(|&v| i64::from(v)).collect();
        let hw = Mat::from((h64.as_slice(), rows, c.hidden));
        let (x8, xm, xk) = self.norm_quant(hw, None, 0);
        let logits = gemm(
            x8.as_slice(),
            rows,
            c.hidden,
            self.a.head.w8t.as_slice(),
            c.vocab,
        );
        (logits, xm, xk)
    }

    /// Greedy generation with the cache (`model.py:493-531`, equal-length
    /// prompts): returns `max_new` token ids per batch row.
    pub fn generate(&self, ids: &[u32], b: usize, t0: usize, max_new: usize) -> Vec<Vec<usize>> {
        let c = self.config();
        assert!(max_new >= 1, "generate: max_new must be >= 1");
        assert_eq!(ids.len(), b * t0, "generate: ids shape");
        let mut cache = KvCache::new(b, c.n_layers, c.n_kv, c.head_dim);
        let positions: Vec<usize> = (0..b * t0).map(|i| i % t0).collect();
        let (logits, _xm, _xk) = self.forward(ids, &positions, None, Some(&mut cache), b, t0);
        let mut out = Vec::with_capacity(b);
        let mut cur = Vec::with_capacity(b);
        for bi in 0..b {
            // the prompt's last position's logits row (×vocab — logits is
            // [B*T0, vocab] row-major)
            let r0 = ((bi + 1) * t0 - 1) * c.vocab;
            let tok = greedy_pick(&logits[r0..r0 + c.vocab], self.head_m());
            out.push(vec![tok]);
            cur.push(u32::try_from(tok).expect("generate: token past u32"));
        }
        for step in 1..max_new {
            let positions: Vec<usize> = (0..b).map(|_| t0 + step - 1).collect();
            let (lg, _xm, _xk) = self.forward(&cur, &positions, None, Some(&mut cache), b, 1);
            for bi in 0..b {
                let r0 = bi * c.vocab;
                let tok = greedy_pick(&lg[r0..r0 + c.vocab], self.head_m());
                out[bi].push(tok);
                cur[bi] = u32::try_from(tok).expect("generate: token past u32");
            }
        }
        out
    }
}

#[cfg(test)]
mod toy {
    use super::*;

    pub(super) const V: usize = 16;

    /// xorshift64* — a seeded, dependency-free generator (the
    /// workspace's global-RNG gate class: no unseeded draws).
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            Rng(seed | 1)
        }
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545F4914F6CDD1D)
        }
        fn i8s(&mut self, span: i64) -> i8 {
            ((self.next() % (2 * span as u64 + 1)) as i64 - span) as i8
        }
        fn mant(&mut self, hi: u64) -> i64 {
            1 + (self.next() % hi) as i64
        }
    }

    fn lin(rng: &mut Rng, k_in: usize, n_out: usize) -> LinearWeights {
        LinearWeights {
            w8t: (0..k_in * n_out).map(|_| rng.i8s(64)).collect(),
            m: (0..n_out).map(|_| rng.mant(1 << 14)).collect(),
            k: 22,
        }
    }

    /// A self-consistent toy artifact (2L, hidden 8, 2 q-heads / 1
    /// kv-head, hd 4, vocab 16, ffn 16): every path live — linears k=22,
    /// norms k=11 (one signed γ channel), smoothings k=10, embed_k=20,
    /// identity rope over 64 positions. The qs/kq/ks family satisfies
    /// the upstream calibration semantics (v(kq) = v(qs)·sqrt(hd)·v(ks):
    /// hd=4, v(qs)=5/2^10, v(ks)=1/2 → v(kq)=5/2^10, i.e. kq_m = qs_m
    /// << 7 at kq_k = 17) with a live K cache — |k8| spreads across the
    /// int8 range instead of collapsing to zeros.
    pub(super) fn toy_artifact() -> IntRefArtifact {
        let cfg = IntRefConfig {
            n_layers: 2,
            hidden: 8,
            n_q: 2,
            n_kv: 1,
            head_dim: 4,
            vocab: V,
            intermediate: 16,
        };
        let mut rng = Rng::new(0x1214_1214_1214_1215);
        let hd = cfg.head_dim;
        let kv_dim = cfg.n_kv * hd;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            let qs_m = vec![5, 7, 5, 7];
            // v(kq) = v(qs)·sqrt(hd)·v(ks) = 5/2^10 · 2 · 1/2 = 5/2^10
            // → kq_m = qs_m << 7 at kq_k = 17 (5·2^7 / 2^17 = 5/2^10)
            let kq_m: Vec<i64> = qs_m.iter().map(|&m| m << 7).collect();
            layers.push(LayerWeights {
                q: lin(&mut rng, cfg.hidden, cfg.n_q * hd),
                k: lin(&mut rng, cfg.hidden, kv_dim),
                v: lin(&mut rng, cfg.hidden, kv_dim),
                o: lin(&mut rng, cfg.n_q * hd, cfg.hidden),
                gate: lin(&mut rng, cfg.hidden, cfg.intermediate),
                up: lin(&mut rng, cfg.hidden, cfg.intermediate),
                down: lin(&mut rng, cfg.intermediate, cfg.hidden),
                q_norm_m: {
                    let mut v: Vec<i64> = (0..hd).map(|_| rng.mant(1 << 10)).collect();
                    v[hd - 1] = -v[hd - 1]; // one signed gamma channel
                    v
                },
                q_norm_k: 11,
                k_norm_m: (0..hd).map(|_| rng.mant(1 << 10)).collect(),
                k_norm_k: 11,
                qs_m,
                qs_k: 10,
                as_m: (0..cfg.hidden).map(|_| rng.mant(1 << 10)).collect(),
                as_k: 10,
                ms_m: (0..cfg.hidden).map(|_| rng.mant(1 << 10)).collect(),
                ms_k: 10,
                ds_m: (0..cfg.intermediate).map(|_| rng.mant(1 << 10)).collect(),
                ds_k: 10,
                os_m: (0..cfg.n_q * hd).map(|_| rng.mant(1 << 10)).collect(),
                os_k: 10,
                ks_m: vec![1],
                ks_k: vec![1],
                kq_m,
                kq_k: vec![17; kv_dim],
                vq_m: vec![32768],
                vq_k: vec![20],
            });
        }
        IntRefArtifact {
            cfg,
            embed_e8: (0..cfg.vocab * cfg.hidden).map(|_| rng.i8s(32)).collect(),
            embed_m: (0..cfg.vocab).map(|_| rng.mant(1 << 10)).collect(),
            embed_k: vec![20; cfg.vocab],
            head: lin(&mut rng, cfg.hidden, cfg.vocab),
            rope: RopeTables::identity(64, cfg.head_dim),
            layers,
        }
    }

    pub(super) fn toy_model() -> IntRefModel {
        IntRefModel::new(toy_artifact())
    }
}

#[cfg(test)]
mod tests {
    use super::toy::{V, toy_model};
    use super::*;

    #[test]
    fn forward_is_deterministic_and_input_sensitive() {
        let m = toy_model();
        let ids = [1_u32, 5, 2, 9, 3, 0];
        let pos: Vec<usize> = (0..6).collect();
        let (l1, xm1, xk1) = m.forward(&ids, &pos, None, None, 1, 6);
        let (l2, xm2, xk2) = m.forward(&ids, &pos, None, None, 1, 6);
        assert_eq!(l1, l2, "same input must be bit-identical");
        assert_eq!(xm1, xm2);
        assert_eq!(xk1, xk2);
        // non-vacuity: a different input must not collapse to identical
        // logits (a dead artifact would pass the determinism half)
        let ids2 = [4_u32, 4, 4, 4, 4, 4];
        let (l3, _xm3, _xk3) = m.forward(&ids2, &pos, None, None, 1, 6);
        assert_ne!(l1, l3, "different inputs produced identical logits");
    }

    #[test]
    fn batch_rows_match_solo_runs_including_padding() {
        let m = toy_model();
        // row 0: [2, 7] + a masked pad slot; row 1: [9, 3, 1] unpadded
        let ids = [2_u32, 7, 0, 9, 3, 1];
        let valid = [true, true, false, true, true, true];
        let pos: Vec<usize> = (0..6).collect();
        let (lb, xmb, _xkb) = m.forward(&ids, &pos, Some(&valid), None, 2, 3);
        let (s0, xm0, _xk0) = m.forward(&[2_u32, 7], &[0, 1], None, None, 1, 2);
        let (s1, xm1, _xk1) = m.forward(&[9_u32, 3, 1], &[0, 1, 2], None, None, 1, 3);
        // row 0's VALID positions (0, 1) must match its solo run exactly
        // (the pad slot is masked out of every score band); row 1, fully
        // valid, must match its own solo run — co-batched content cannot
        // perturb a sequence's outputs (upstream §9.3).
        assert_eq!(
            &lb[0..2 * V],
            &s0[..],
            "padded row diverged from its solo run"
        );
        assert_eq!(&xmb[0..2], &xm0[..]);
        assert_eq!(
            &lb[3 * V..6 * V],
            &s1[..],
            "co-batched row diverged from its solo run"
        );
        assert_eq!(&xmb[3..6], &xm1[..]);
    }

    #[test]
    fn prefill_decode_split_matches_full_prefill() {
        let m = toy_model();
        let ids = [3_u32, 11, 6, 12];
        let pos: Vec<usize> = (0..4).collect();
        let (full, _xm, _xk) = m.forward(&ids, &pos, None, None, 1, 4);
        let mut cache = KvCache::new(1, 2, 1, 4);
        let (p1, _a, _b) = m.forward(&ids[0..2], &[0, 1], None, Some(&mut cache), 1, 2);
        let (p2, _c, _d) = m.forward(&ids[2..4], &[2, 3], None, Some(&mut cache), 1, 2);
        assert_eq!(&p1[..], &full[0..2 * V], "first chunk diverged");
        assert_eq!(
            &p2[..],
            &full[2 * V..4 * V],
            "second chunk diverged (cache boundary not exact)"
        );
        assert_eq!(cache.seq_len(), 4);
    }

    #[test]
    fn generate_matches_manual_greedy_loop() {
        let m = toy_model();
        let ids = [5_u32, 2, 8];
        let g = m.generate(&ids, 1, 3, 3);
        // manual loop: prefill, then one token at a time through the
        // same cache (this also exercises the cache growth path — the
        // prefill fills 3 of 4 slots, decode grows past capacity)
        let mut cache = KvCache::new(1, 2, 1, 4);
        let (lg, _xm, _xk) = m.forward(&ids, &[0, 1, 2], None, Some(&mut cache), 1, 3);
        let mut tok = greedy_pick(&lg[2 * V..3 * V], m.head_m());
        let mut manual = vec![tok];
        let decode_positions = [3_usize, 4];
        for p in decode_positions {
            let (l1, _xm, _xk) = m.forward(&[tok as u32], &[p], None, Some(&mut cache), 1, 1);
            tok = greedy_pick(&l1[0..V], m.head_m());
            manual.push(tok);
        }
        assert_eq!(
            g[0], manual,
            "generate diverged from the manual greedy loop"
        );
    }

    #[test]
    fn greedy_pick_tie_goes_to_first_index() {
        let hm = [3_i64, 3, 3, 3];
        assert_eq!(
            greedy_pick(&[5, 9, 9, 1], &hm),
            1,
            "exact tie must keep the earlier index"
        );
        // a tie that only appears after the per-vocab column scale:
        // 2·6 == 3·4 == 12
        let hm2 = [6_i64, 4, 1, 9];
        assert_eq!(
            greedy_pick(&[2, 3, 12, 1], &hm2),
            0,
            "scaled tie must keep the earlier index"
        );
        // strictly greater displaces
        assert_eq!(greedy_pick(&[2, 3, 13, 1], &hm2), 2);
    }
}
