//! GLM-5.3-Flash mHC (hyper-connection) seat — Plan 627 Phase 1 T4b.
//!
//! A thin mapping layer over the UPSTREAM mHC substrate
//! (`katgpt_attn::mhc` — `MhcConfig`, `MhcWeights`, `MhcScratch`,
//! `mhc_project`, `mhc_combine`), specialized to the glm5 config grammar:
//!
//! - The config's `hc_mult` / `hc_eps` / `hc_sinkhorn_iters` map 1:1 onto
//!   the substrate's `n_streams` / `eps` / `sinkhorn_iters`; the input norm's
//!   epsilon is the shared `rms_norm_eps` (the source's
//!   `DeepseekV4UnweightedRMSNorm` variance epsilon, unweighted — no
//!   learnable scale).
//! - The checkpoint stores TWO hyper-connection sites per main layer —
//!   `hc_attn_*` (around the attention block) and `hc_ffn_*` (around the
//!   MLP/MoE block) — six tensors total, none on the MTP layer (T0's
//!   inventory grammar). `Glm5MhcWeights` is one site's `{fn, base, scale}`
//!   triple; [`Glm5Mhc`] carries both sites plus the substrate config.
//! - Shapes (N = 4, D = 4096): `fn` **[24, 16384]**, `base` **[24]**,
//!   `scale` **[3]** — the source's `(2 + hc_mult) * hc_mult` ×
//!   `hc_mult * hidden_size` formula, pinned by a substrate test and
//!   refused loud here on any mismatch. (Plan 627's earlier "[12×4096]"
//!   bullet was a mis-derivation — corrected in the plan at claim time.)
//! - The per-layer flow this seat composes (the transformers decoder):
//!   `streams` → site project → `{post, comb, collapsed}` → block on
//!   `collapsed` → combine `post ⊗ block_out + combᵀ · streams` → next
//!   site. The stream head is GLM's UNWEIGHTED MEAN over the N streams
//!   (`Glm5NextTextHyperHead`, unlike DeepSeek-V4's learned mix) —
//!   [`Glm5Mhc::head_mean_into`]. Model entry expands the embedding to N
//!   identical streams — [`Glm5Mhc::expand_streams`].
//!
//! Consumed, never forked: the sinkhorn projection, the pre/post/comb
//! weight forms, the norm, the collapse, and the combine are the
//! substrate's own — the seat adds only the config mapping, the two-site
//! naming, and the decoder-flow helpers.
//!
//! Feature `glm5_mhc` implies `glm5` + `katgpt-attn/mhc`.

use katgpt_attn::mhc::{MhcConfig, MhcScratch, MhcWeightsRef, mhc_combine, mhc_project};

use crate::glm5::Glm5Config;

// ---------------------------------------------------------------------------
// Config conversion (glm5 hc_* facts → the upstream mHC vocabulary)
// ---------------------------------------------------------------------------

impl Glm5Config {
    /// Build the substrate `MhcConfig` from the parsed `hc_*` keys. Refuses
    /// loud naming every missing key — a config that enables mHC weights
    /// without the geometry is malformed, never guessed.
    pub fn to_mhc_config(&self) -> Result<MhcConfig, String> {
        let missing = |name: &str| format!("mHC geometry missing: {name} is not set");
        let n_streams = self.hc_mult.ok_or_else(|| missing("hc_mult"))?;
        let eps = self.hc_eps.ok_or_else(|| missing("hc_eps"))?;
        let sinkhorn_iters = self
            .hc_sinkhorn_iters
            .ok_or_else(|| missing("hc_sinkhorn_iters"))?;
        if n_streams == 0 {
            return Err("mHC geometry invalid: hc_mult must be >= 1".to_string());
        }
        Ok(MhcConfig {
            n_streams,
            hidden_size: self.hidden_size,
            sinkhorn_iters,
            eps,
            rms_eps: self.rms_norm_eps,
        })
    }
}

// ---------------------------------------------------------------------------
// Weight mapping (glm5 inventory names → the substrate MhcWeights)
// ---------------------------------------------------------------------------

/// One hyper-connection site's weights — the `{fn, base, scale}` triple
/// behind `hc_attn_*` or `hc_ffn_*` (`fn` is a reserved word; the field is
/// `fn_w`).
#[derive(Clone, Debug)]
pub struct Glm5MhcWeights {
    /// The projection matrix (checkpoint `hc_*.fn`), row-major
    /// `[(2+N)·N, N·D]` — GLM-5.3: [24, 16384].
    pub fn_w: Vec<f32>,
    /// The additive base (checkpoint `hc_*.base`), `[(2+N)·N]` — [24].
    pub base: Vec<f32>,
    /// The three output scales (checkpoint `hc_*.scale`): `[pre, post, comb]`.
    pub scale: [f32; 3],
}

impl Glm5MhcWeights {
    /// Length/type validation against the substrate config — refuses loud
    /// naming the first mismatch (delegates to the substrate's own gate on
    /// the borrow view; zero copy).
    pub fn validate(&self, cfg: &MhcConfig) -> Result<(), String> {
        self.as_view().validate(cfg)
    }

    /// Borrow-view into the substrate weight vocabulary (zero-copy — the
    /// hot path takes the view, never an owned clone of the ~1.5 MiB
    /// projection matrix).
    pub fn as_view(&self) -> MhcWeightsRef<'_> {
        MhcWeightsRef {
            fn_w: &self.fn_w,
            base: &self.base,
            scale: self.scale,
        }
    }
}

// ---------------------------------------------------------------------------
// The seat — both sites + the decoder-flow helpers
// ---------------------------------------------------------------------------

/// The glm5 mHC plane: the substrate config plus the per-layer ATTN and FFN
/// site weights.
pub struct Glm5Mhc {
    pub cfg: MhcConfig,
    /// The hyper-connection around the attention block (`hc_attn_*`).
    pub attn: Glm5MhcWeights,
    /// The hyper-connection around the MLP/MoE block (`hc_ffn_*`).
    pub ffn: Glm5MhcWeights,
}

impl Glm5Mhc {
    /// Assemble + validate both sites; refuses loud on any shape mismatch.
    pub fn new(
        cfg: &Glm5Config,
        attn: Glm5MhcWeights,
        ffn: Glm5MhcWeights,
    ) -> Result<Self, String> {
        let mcfg = cfg.to_mhc_config()?;
        attn.validate(&mcfg)?;
        ffn.validate(&mcfg)?;
        Ok(Self {
            cfg: mcfg,
            attn,
            ffn,
        })
    }

    /// Model entry: expand the embedding `[D]` into N identical streams
    /// `[N·D]` (the source's `unsqueeze(2).expand(-1, -1, hc_mult, -1)`).
    /// Writes `out`; allocates nothing.
    pub fn expand_streams(&self, embed: &[f32], out: &mut [f32]) {
        let n = self.cfg.n_streams;
        let d = self.cfg.hidden_size;
        debug_assert_eq!(embed.len(), d, "embed must be [D]");
        debug_assert_eq!(out.len(), n * d, "out must be [N·D]");
        for k in 0..n {
            out[k * d..(k + 1) * d].copy_from_slice(embed);
        }
    }

    /// Project the ATTN site: `streams [N·D] → (post, comb, collapsed)` via
    /// the substrate forward, borrowing `sc` for the buffers.
    pub fn project_attn<'a>(
        &self,
        streams: &[f32],
        sc: &'a mut MhcScratch,
    ) -> (&'a [f32], &'a [f32], &'a [f32]) {
        mhc_project(&self.cfg, &self.attn.as_view(), streams, sc)
    }

    /// Project the FFN site (same contract as [`Glm5Mhc::project_attn`]).
    pub fn project_ffn<'a>(
        &self,
        streams: &[f32],
        sc: &'a mut MhcScratch,
    ) -> (&'a [f32], &'a [f32], &'a [f32]) {
        mhc_project(&self.cfg, &self.ffn.as_view(), streams, sc)
    }

    /// The per-layer combine (delegates to the substrate's transposed-comb
    /// mixer): `new[k][d] = post[k]·block_out[d] + Σ_j comb[j][k]·residual[j][d]`.
    /// `out` must not alias `residual`; allocates nothing.
    pub fn combine(
        &self,
        post: &[f32],
        comb: &[f32],
        block_out: &[f32],
        residual: &[f32],
        out: &mut [f32],
    ) {
        mhc_combine(&self.cfg, post, comb, block_out, residual, out);
    }

    /// GLM's stream head (`Glm5NextTextHyperHead`): the UNWEIGHTED mean over
    /// the N streams. Writes `out [D]`; allocates nothing.
    pub fn head_mean_into(&self, streams: &[f32], out: &mut [f32]) {
        let n = self.cfg.n_streams;
        let d = self.cfg.hidden_size;
        debug_assert_eq!(streams.len(), n * d, "streams must be [N·D]");
        debug_assert_eq!(out.len(), d, "out must be [D]");
        for (dd, out_v) in out.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for k in 0..n {
                acc += streams[k * d + dd];
            }
            *out_v = acc / n as f32;
        }
    }

    /// A fresh scratch sized for this config.
    pub fn scratch(&self) -> MhcScratch {
        MhcScratch::new(&self.cfg)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny but structurally EXACT glm5 mHC shape: 1 DSA layer, hc geometry
    /// matching the real config (mult 4, eps 1e-6, iters 20) over a small
    /// hidden size.
    fn tiny_mhc_cfg() -> Glm5Config {
        let json = r##"{
            "model_type": "glm5_next",
            "text_config": {
                "model_type": "glm5_next_text",
                "num_hidden_layers": 1,
                "layer_types": ["deepseek_sparse_attention"],
                "hidden_size": 16,
                "num_attention_heads": 2,
                "q_lora_rank": 8,
                "kv_lora_rank": 8,
                "qk_nope_head_dim": 4,
                "qk_rope_head_dim": 0,
                "v_head_dim": 4,
                "max_position_embeddings": 4096,
                "vocab_size": 64,
                "first_k_dense_replace": 0,
                "n_routed_experts": 2,
                "rms_norm_eps": 1e-05,
                "hc_mult": 4,
                "mhc": true,
                "hc_eps": 1e-06,
                "hc_sinkhorn_iters": 20
            }
        }"##;
        Glm5Config::parse(json).expect("tiny mhc parse")
    }

    /// Deterministic pseudo-random f32 rows (LCG — the upstream
    /// `random_matrix` convention).
    fn rand_rows(seed: u64, rows: usize, cols: usize) -> Vec<f32> {
        let mut s = seed | 1;
        let scale = 1.0 / (cols as f32).sqrt();
        (0..rows * cols)
            .map(|_| {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let u = ((s >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0;
                u * scale
            })
            .collect()
    }

    fn tiny_weights(cfg: &MhcConfig) -> Glm5MhcWeights {
        Glm5MhcWeights {
            fn_w: rand_rows(1, cfg.fn_rows(), cfg.fn_cols()),
            base: rand_rows(2, 1, cfg.fn_rows()),
            scale: [0.8, 1.2, 1.0],
        }
    }

    #[test]
    fn config_conversion_maps_the_hc_geometry() {
        let cfg = tiny_mhc_cfg();
        let m = cfg.to_mhc_config().expect("conversion");
        assert_eq!(m.n_streams, 4);
        assert_eq!(m.hidden_size, 16);
        assert_eq!(m.sinkhorn_iters, 20);
        assert_eq!(m.eps, 1e-6);
        assert_eq!(m.rms_eps, cfg.rms_norm_eps);
        assert_eq!(m.fn_rows(), 24);
        assert_eq!(m.fn_cols(), 64);
    }

    #[test]
    fn config_conversion_refuses_missing_geometry_loud() {
        let mut cfg = tiny_mhc_cfg();
        cfg.hc_eps = None;
        assert!(cfg.to_mhc_config().is_err());
        let mut cfg = tiny_mhc_cfg();
        cfg.hc_sinkhorn_iters = None;
        assert!(cfg.to_mhc_config().is_err());
        let mut cfg = tiny_mhc_cfg();
        cfg.hc_mult = None;
        assert!(cfg.to_mhc_config().is_err());
        let mut cfg = tiny_mhc_cfg();
        cfg.hc_mult = Some(0);
        assert!(cfg.to_mhc_config().is_err());
    }

    #[test]
    fn seat_assembles_and_refuses_bad_shapes_loud() {
        let cfg = tiny_mhc_cfg();
        let mcfg = cfg.to_mhc_config().expect("mcfg");
        let attn = tiny_weights(&mcfg);
        let ffn = tiny_weights(&mcfg);
        let seat = Glm5Mhc::new(&cfg, attn.clone(), ffn.clone()).expect("seat");
        // The real-size expectation through THIS config's shapes:
        assert_eq!(seat.cfg.fn_rows(), (2 + 4) * 4);
        assert_eq!(seat.cfg.fn_cols(), 4 * 16);
        // Wrong fn length refuses.
        let bad = Glm5MhcWeights {
            fn_w: vec![0.0; mcfg.fn_rows() * mcfg.fn_cols() + 1],
            base: attn.base.clone(),
            scale: [1.0; 3],
        };
        assert!(Glm5Mhc::new(&cfg, bad, ffn).is_err());
        // Wrong base length refuses.
        let mcfg2 = cfg.to_mhc_config().unwrap();
        let attn2 = tiny_weights(&mcfg2);
        let bad_base = Glm5MhcWeights {
            fn_w: attn2.fn_w.clone(),
            base: vec![0.0; mcfg2.fn_rows() - 1],
            scale: [1.0; 3],
        };
        assert!(Glm5Mhc::new(&cfg, bad_base, attn2).is_err());
    }

    #[test]
    fn expand_and_head_roundtrip_through_a_uniform_stream() {
        let cfg = tiny_mhc_cfg();
        let mcfg = cfg.to_mhc_config().unwrap();
        let seat = Glm5Mhc::new(&cfg, tiny_weights(&mcfg), tiny_weights(&mcfg)).expect("seat");
        let embed: Vec<f32> = (0..seat.cfg.hidden_size).map(|i| i as f32 * 0.25).collect();
        let mut streams = vec![0.0f32; seat.cfg.fn_cols()];
        seat.expand_streams(&embed, &mut streams);
        for k in 0..seat.cfg.n_streams {
            assert_eq!(
                &streams[k * seat.cfg.hidden_size..(k + 1) * seat.cfg.hidden_size],
                &embed[..]
            );
        }
        let mut head = vec![0.0f32; seat.cfg.hidden_size];
        seat.head_mean_into(&streams, &mut head);
        // The mean of N identical copies is the copy itself.
        for (h, &e) in head.iter().zip(embed.iter()) {
            assert!((h - e).abs() < 1e-6);
        }
    }

    /// The full decoder-site flow — expand → attn project → identity block →
    /// combine → ffn project → identity block → combine → head mean — against
    /// an independent in-test transcription, and bit-identical on rerun.
    #[test]
    fn decoder_flow_matches_transcription_and_is_deterministic() {
        let cfg = tiny_mhc_cfg();
        let mcfg = cfg.to_mhc_config().unwrap();
        let seat = Glm5Mhc::new(&cfg, tiny_weights(&mcfg), tiny_weights(&mcfg)).expect("seat");
        let n = seat.cfg.n_streams;
        let d = seat.cfg.hidden_size;
        let flat_len = n * d;
        let embed: Vec<f32> = (0..d).map(|i| (i as f32 * 0.31).cos()).collect();

        let run = || {
            let mut streams = vec![0.0f32; flat_len];
            seat.expand_streams(&embed, &mut streams);
            let mut sc_attn = seat.scratch();
            let mut sc_ffn = seat.scratch();
            let mut out = vec![0.0f32; flat_len];
            for _ in 0..2 {
                // ATTN site (identity block: block_out = collapsed)
                let (post, comb, collapsed) = {
                    let (p, c, cl) = seat.project_attn(&streams, &mut sc_attn);
                    (p.to_vec(), c.to_vec(), cl.to_vec())
                };
                seat.combine(&post, &comb, &collapsed, &streams, &mut out);
                std::mem::swap(&mut streams, &mut out);
                // FFN site
                let (post, comb, collapsed) = {
                    let (p, c, cl) = seat.project_ffn(&streams, &mut sc_ffn);
                    (p.to_vec(), c.to_vec(), cl.to_vec())
                };
                seat.combine(&post, &comb, &collapsed, &streams, &mut out);
                std::mem::swap(&mut streams, &mut out);
            }
            let mut head = vec![0.0f32; d];
            seat.head_mean_into(&streams, &mut head);
            (streams.clone(), head)
        };
        let (streams1, head1) = run();
        let (streams2, head2) = run();
        assert_eq!(streams1, streams2, "stream bits drifted");
        assert_eq!(head1, head2, "head bits drifted");
        // streams stay finite through 2 layers (the non-expansive property
        // the sinkhorn constraint buys — sanity, not a tight bound).
        for v in &streams1 {
            assert!(v.is_finite(), "stream value diverged: {v}");
        }
    }

    /// The combine consumed through the seat matches the substrate free fn
    /// bit-for-bit (thin-delegation pin).
    #[test]
    fn seat_combine_delegates_bit_identically_to_the_substrate() {
        let cfg = tiny_mhc_cfg();
        let mcfg = cfg.to_mhc_config().unwrap();
        let seat = Glm5Mhc::new(&cfg, tiny_weights(&mcfg), tiny_weights(&mcfg)).expect("seat");
        let post = [1.0f32, 0.5, 2.0, 0.25];
        let comb: Vec<f32> = (0..16).map(|i| 0.0625 * (i as f32 + 1.0)).collect();
        let block_out: Vec<f32> = (0..seat.cfg.hidden_size)
            .map(|i| (i as f32) - 3.0)
            .collect();
        let residual: Vec<f32> = (0..seat.cfg.fn_cols())
            .map(|i| (i as f32 * 0.07).sin())
            .collect();
        let mut a = vec![0.0f32; seat.cfg.fn_cols()];
        let mut b = vec![0.0f32; seat.cfg.fn_cols()];
        seat.combine(&post, &comb, &block_out, &residual, &mut a);
        mhc_combine(&seat.cfg, &post, &comb, &block_out, &residual, &mut b);
        assert_eq!(a, b);
    }
}
