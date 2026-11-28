//! GLM-5.3-Flash KDA-layer CPU-reference seat — Plan 627 Phase 1 T4a.
//!
//! A thin wrapper over the UPSTREAM KDA substrate
//! (`katgpt_attn::gdn2::kda_forward` — `KdaConfig`, `KdaWeights`,
//! `KdaLayerCache`, `KdaForwardScratch`), specialized to the glm5 shape:
//!
//! - The forget gate is GLM's SIGMOID LOWER-BOUND form
//!   (`kda_forward_token_lower_bound_gate`, bound = `linear_lower_bound`,
//!   measured −5.0): `gk = bound · sigmoid(exp(A_log)·(f_b(f_a(h)) +
//!   dt_bias))` — the transformers `Glm5NextTextForgetGate` branch taken
//!   when `linear_lower_bound` is set. The decay factor stays in
//!   `[exp(bound), 1]` (≈ [0.0067, 1]), so the substrate's `alpha_eps`
//!   floor never fires.
//! - The checkpoint stores SEPARATE `q_conv1d` / `k_conv1d` / `v_conv1d`
//!   weights (T0's inventory grammar) — they map 1:1 onto the substrate's
//!   three ShortConv filters (the transformers modeling fuses them into one
//!   grouped conv over the qkv concat, which is per-channel identical).
//! - The output gate is LOW-RANK (`g_a_proj` [head_dim, d] → `g_b_proj`
//!   [proj, head_dim]); the substrate's `KdaWeights` carries the full-rank
//!   `g_proj` form, so the map composes `g_b @ g_a` ONCE at load. The
//!   composition reassociates two matmuls into one — fine for the CPU
//!   reference (its synthetic tests are self-consistent); real-capture
//!   parity is the weights-gated gate that adjudicates the composed form.
//!
//! Consumed, never forked: the recurrence, l2norm (fla convention,
//! `x/sqrt(Σx²+1e-6)`), q-scale, beta sigmoid, SiLU conv, and gated output
//! norm are the substrate's own — the seat adds only the shape mapping.
//!
//! mHC (the 4-wide sinkhorn-normalized residual streams) is its OWN
//! follow-up box (plan 627): the probe's "needed before any parity
//! capture" binds real-tensor captures, which are weights-gated regardless.
//!
//! Feature `glm5_kda` implies `glm5` + `katgpt-attn/gdn2_attention`.

use katgpt_attn::gdn2::kda_forward::{
    kda_forward_token_lower_bound_gate, KdaConfig, KdaForwardScratch, KdaLayerCache, KdaWeights,
};

use crate::glm5::Glm5Config;

// ---------------------------------------------------------------------------
// Config conversion (glm5 linear_* facts → the upstream KDA vocabulary)
// ---------------------------------------------------------------------------

impl Glm5Config {
    /// Build the substrate `KdaConfig` from the parsed `linear_*` keys.
    /// Refuses loud naming every missing key — a config that declares KDA
    /// layers without the geometry is malformed, never guessed.
    pub fn to_kda_config(&self) -> Result<KdaConfig, String> {
        let missing = |name: &str| format!("linear geometry missing: {name} is not set");
        let n_heads = self
            .linear_num_heads
            .ok_or_else(|| missing("linear_num_heads"))?;
        let head_dim = self
            .linear_head_dim
            .ok_or_else(|| missing("linear_head_dim"))?;
        let conv = self
            .linear_conv_kernel_dim
            .ok_or_else(|| missing("linear_conv_kernel_dim"))?;
        // The gate bound is NOT a KdaConfig field (its struct literals are
        // pinned across two repos) — but a KDA-seat config without it is
        // malformed, so its PRESENCE is checked here; the value itself is
        // consumed by the seat (`gate_lower_bound`).
        self.linear_lower_bound
            .ok_or_else(|| missing("linear_lower_bound"))?;
        Ok(KdaConfig {
            head_dim,
            n_heads,
            hidden_size: self.hidden_size,
            conv_kernel_size: conv,
            alpha_eps: 1e-5,
            rms_eps: self.rms_norm_eps,
        })
    }
}

// ---------------------------------------------------------------------------
// Weight mapping (glm5 inventory names → the substrate KdaWeights)
// ---------------------------------------------------------------------------

/// The glm5 KDA weight family (T0's inventory grammar, `self_attn.*` rows):
/// separate q/k/v projections + conv1d filters, the low-rank forget gate
/// (`f_a`/`f_b` + `dt_bias` + `A_log`), the per-head beta (`b_proj`), the
/// LOW-RANK output gate (`g_a`/`g_b`), the gated output norm, and `o_proj`.
#[derive(Clone)]
pub struct Glm5KdaWeights {
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    /// Depthwise FIR taps `[n_channels × kernel]`, per channel
    /// (the substrate's tap convention: index k weights the sample k steps ago).
    pub q_conv1d: Vec<f32>,
    pub k_conv1d: Vec<f32>,
    pub v_conv1d: Vec<f32>,
    /// Log-space per-head decay `[n_heads]`.
    pub a_log: Vec<f32>,
    /// Forget-gate low-rank pieces: `[head_dim, d]` then `[proj, head_dim]`.
    pub f_a_proj: Vec<f32>,
    pub f_b_proj: Vec<f32>,
    /// Per-channel gate bias `[proj]`.
    pub dt_bias: Vec<f32>,
    /// Per-head beta pre-sigmoid `[n_heads, d]`.
    pub b_proj: Vec<f32>,
    /// Low-rank output gate: `[head_dim, d]` then `[proj, head_dim]`.
    pub g_a_proj: Vec<f32>,
    pub g_b_proj: Vec<f32>,
    /// Gated output norm gamma `[head_dim]`.
    pub o_norm: Vec<f32>,
    pub o_proj: Vec<f32>,
}

impl Glm5KdaWeights {
    /// Validate every family shape against the parsed geometry. Refuses the
    /// FIRST mismatch with both sizes named.
    pub fn validate_shapes(&self, cfg: &Glm5Config) -> Result<(), String> {
        let d = cfg.hidden_size;
        let n_heads = cfg
            .linear_num_heads
            .ok_or("linear geometry missing: linear_num_heads")?;
        let dk = cfg
            .linear_head_dim
            .ok_or("linear geometry missing: linear_head_dim")?;
        let ks = cfg
            .linear_conv_kernel_dim
            .ok_or("linear geometry missing: linear_conv_kernel_dim")?;
        let proj = dk * n_heads;
        let conv = |name: &str, got: usize, want: usize| {
            if got == want {
                Ok(())
            } else {
                Err(format!("{name}: {got} != {want}"))
            }
        };
        conv("q_proj", self.q_proj.len(), proj * d)?;
        conv("k_proj", self.k_proj.len(), proj * d)?;
        conv("v_proj", self.v_proj.len(), proj * d)?;
        conv("q_conv1d", self.q_conv1d.len(), proj * ks)?;
        conv("k_conv1d", self.k_conv1d.len(), proj * ks)?;
        conv("v_conv1d", self.v_conv1d.len(), proj * ks)?;
        conv("a_log", self.a_log.len(), n_heads)?;
        conv("f_a_proj", self.f_a_proj.len(), dk * d)?;
        conv("f_b_proj", self.f_b_proj.len(), proj * dk)?;
        conv("dt_bias", self.dt_bias.len(), proj)?;
        conv("b_proj", self.b_proj.len(), n_heads * d)?;
        conv("g_a_proj", self.g_a_proj.len(), dk * d)?;
        conv("g_b_proj", self.g_b_proj.len(), proj * dk)?;
        conv("o_norm", self.o_norm.len(), dk)?;
        conv("o_proj", self.o_proj.len(), d * proj)?;
        Ok(())
    }

    /// Map into the substrate `KdaWeights`. The low-rank output gate is
    /// composed ONCE here (`g_proj[r][c] = Σ_k g_b[r][k]·g_a[k][c]`) — the
    /// two-matmul reference reassociates into one; see the module doc.
    pub fn to_kda_weights(&self, cfg: &Glm5Config) -> KdaWeights {
        let d = cfg.hidden_size;
        let n_heads = cfg.linear_num_heads.expect("validate_shapes ran");
        let dk = cfg.linear_head_dim.expect("validate_shapes ran");
        let proj = dk * n_heads;

        // g_proj = g_b (proj × dk) @ g_a (dk × d) — row-major output [proj × d].
        let mut g_proj = vec![0.0f32; proj * d];
        for r in 0..proj {
            let g_b_row = &self.g_b_proj[r * dk..(r + 1) * dk];
            for (k, &gbk) in g_b_row.iter().enumerate() {
                if gbk == 0.0 {
                    continue;
                }
                let g_a_row = &self.g_a_proj[k * d..(k + 1) * d];
                for (c, &gak) in g_a_row.iter().enumerate() {
                    g_proj[r * d + c] += gbk * gak;
                }
            }
        }

        KdaWeights {
            q_proj: self.q_proj.clone(),
            k_proj: self.k_proj.clone(),
            v_proj: self.v_proj.clone(),
            q_conv_weight: self.q_conv1d.clone(),
            k_conv_weight: self.k_conv1d.clone(),
            v_conv_weight: self.v_conv1d.clone(),
            a_log: self.a_log.clone(),
            f_a_proj: self.f_a_proj.clone(),
            f_b_proj: self.f_b_proj.clone(),
            dt_bias: self.dt_bias.clone(),
            beta_proj: self.b_proj.clone(),
            g_proj,
            o_norm_weight: self.o_norm.clone(),
            o_proj: self.o_proj.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// The per-layer forward seat
// ---------------------------------------------------------------------------

/// One KDA layer's decode seat: the upstream cache + scratch, constructed
/// once and reused across tokens (the alloc-free decode discipline).
pub struct Glm5KdaLayer {
    cfg: Glm5Config,
    kda: KdaConfig,
    weights: KdaWeights,
    cache: KdaLayerCache,
    scratch: KdaForwardScratch,
    gate_lower_bound: f32,
}

impl Glm5KdaLayer {
    /// Build the seat for one KDA layer. The seat runs its OWN scoped
    /// validation — model type, linear geometry (`to_kda_config`), and shape
    /// agreement (`validate_shapes`) — NOT the config-wide `validate()`, whose
    /// ≥1-DSA-layer rule is the DSA lane's posture (a pure-KDA fixture is a
    /// valid KDA-seat input and must not be refused by a sparse-lane rule).
    pub fn new(cfg: &Glm5Config, weights: Glm5KdaWeights) -> Result<Self, String> {
        if cfg.model_type != "glm5_next_text" && cfg.model_type != "glm5_next" {
            return Err(format!("unsupported model_type: {}", cfg.model_type));
        }
        weights.validate_shapes(cfg)?;
        let kda = cfg.to_kda_config()?;
        let bound = cfg
            .linear_lower_bound
            .ok_or("linear geometry missing: linear_lower_bound")?;
        let cache = KdaLayerCache::new(&kda);
        let scratch = KdaForwardScratch::new(&kda);
        Ok(Self {
            kda,
            weights: weights.to_kda_weights(cfg),
            cache,
            scratch,
            cfg: cfg.clone(),
            gate_lower_bound: bound,
        })
    }

    /// The layer's config view.
    pub fn config(&self) -> &Glm5Config {
        &self.cfg
    }

    /// Reset the recurrent state + conv rings (new sequence).
    pub fn reset(&mut self) {
        self.cache.reset();
    }

    /// One decode step: appends this token through the short convs, advances
    /// the per-head delta-rule states, and returns a slice of the layer
    /// scratch holding the layer output `[hidden_size]`. The sigmoid
    /// lower-bound forget gate rides the substrate's dedicated entry point
    /// (`linear_lower_bound` is glm5's pinned posture — the softplus form is
    /// the kimi lane's).
    pub fn forward_token(&mut self, h: &[f32]) -> Result<&[f32], String> {
        if h.len() != self.cfg.hidden_size {
            return Err(format!(
                "hidden state dim mismatch: {} != {}",
                h.len(),
                self.cfg.hidden_size
            ));
        }
        let out = kda_forward_token_lower_bound_gate(
            &self.kda,
            &self.weights,
            &mut self.cache,
            &mut self.scratch,
            h,
            self.gate_lower_bound,
        );
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny but structurally EXACT glm5 KDA shape: 1 linear_attention
    /// layer, the real geometry RATIO (4 heads × 8), the lower bound set.
    fn tiny_kda_cfg() -> Glm5Config {
        let json = r##"{
            "model_type": "glm5_next",
            "text_config": {
                "model_type": "glm5_next_text",
                "num_hidden_layers": 1,
                "layer_types": ["linear_attention"],
                "hidden_size": 64,
                "num_attention_heads": 4,
                "q_lora_rank": 32,
                "kv_lora_rank": 16,
                "qk_nope_head_dim": 8,
                "qk_rope_head_dim": 0,
                "v_head_dim": 8,
                "max_position_embeddings": 4096,
                "vocab_size": 128,
                "first_k_dense_replace": 0,
                "n_routed_experts": 2,
                "rms_norm_eps": 1e-05,
                "linear_num_heads": 4,
                "linear_head_dim": 8,
                "linear_conv_kernel_dim": 4,
                "linear_lower_bound": -5.0
            }
        }"##;
        Glm5Config::parse(json).expect("tiny kda parse")
    }

    /// Deterministic pseudo-random f32 rows (LCG — no new dep, the
    /// upstream `random_matrix` convention).
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

    fn rand_vec(seed: u64, n: usize) -> Vec<f32> {
        rand_rows(seed, 1, n)
    }

    fn tiny_weights(cfg: &Glm5Config) -> Glm5KdaWeights {
        let d = cfg.hidden_size;
        let n_h = cfg.linear_num_heads.expect("geometry");
        let dk = cfg.linear_head_dim.expect("geometry");
        let ks = cfg.linear_conv_kernel_dim.expect("geometry");
        let proj = dk * n_h;
        Glm5KdaWeights {
            q_proj: rand_rows(1, proj, d),
            k_proj: rand_rows(2, proj, d),
            v_proj: rand_rows(3, proj, d),
            q_conv1d: rand_rows(4, proj, ks),
            k_conv1d: rand_rows(5, proj, ks),
            v_conv1d: rand_rows(6, proj, ks),
            a_log: rand_vec(7, n_h),
            f_a_proj: rand_rows(8, dk, d),
            f_b_proj: rand_rows(9, proj, dk),
            dt_bias: rand_vec(10, proj),
            b_proj: rand_rows(11, n_h, d),
            g_a_proj: rand_rows(12, dk, d),
            g_b_proj: rand_rows(13, proj, dk),
            o_norm: rand_vec(14, dk),
            o_proj: rand_rows(15, d, proj),
        }
    }

    #[test]
    fn config_conversion_maps_the_linear_geometry() {
        let cfg = tiny_kda_cfg();
        let kda = cfg.to_kda_config().expect("conversion");
        assert_eq!(kda.n_heads, 4);
        assert_eq!(kda.head_dim, 8);
        assert_eq!(kda.conv_kernel_size, 4);
        assert_eq!(kda.hidden_size, 64);
        assert_eq!(kda.rms_eps, cfg.rms_norm_eps);
    }

    #[test]
    fn conversion_refuses_missing_geometry_loud() {
        // Strip the linear keys: the seat must refuse naming the first
        // missing key, never guess.
        let json = r##"{
            "model_type": "glm5_next",
            "text_config": {
                "model_type": "glm5_next_text",
                "num_hidden_layers": 1,
                "layer_types": ["linear_attention"],
                "hidden_size": 64,
                "num_attention_heads": 4,
                "kv_lora_rank": 16,
                "qk_nope_head_dim": 8,
                "max_position_embeddings": 4096,
                "vocab_size": 128
            }
        }"##;
        let cfg = Glm5Config::parse(json).expect("parse");
        let err = cfg
            .to_kda_config()
            .expect_err("missing geometry must refuse");
        assert!(err.contains("linear_num_heads"), "err: {err}");
    }

    #[test]
    fn shape_validation_refuses_a_short_row() {
        let cfg = tiny_kda_cfg();
        let mut w = tiny_weights(&cfg);
        w.q_proj.pop();
        let err = w
            .validate_shapes(&cfg)
            .expect_err("short q_proj must refuse");
        assert!(err.contains("q_proj"), "err: {err}");
    }

    #[test]
    fn seat_new_refuses_broken_shapes_and_runs_decodes() {
        let cfg = tiny_kda_cfg();
        let d = cfg.hidden_size;
        // Broken weights refuse at construction.
        let mut bad = tiny_weights(&cfg);
        bad.o_proj.truncate(bad.o_proj.len() - 1);
        assert!(Glm5KdaLayer::new(&cfg, bad).is_err());
        // A good seat decodes, advances deterministically, and resets.
        let mut seat = Glm5KdaLayer::new(&cfg, tiny_weights(&cfg)).expect("seat");
        let first = seat
            .forward_token(&rand_vec(100, d))
            .expect("decode 1")
            .to_vec();
        assert_eq!(first.len(), d);
        assert!(first.iter().all(|v| v.is_finite()));
        seat.reset();
        let again = seat
            .forward_token(&rand_vec(100, d))
            .expect("decode 2")
            .to_vec();
        assert_eq!(first, again, "reset + re-decode must reproduce");
    }

    #[test]
    fn decode_is_deterministic_across_two_tokens() {
        let cfg = tiny_kda_cfg();
        let d = cfg.hidden_size;
        let mut a = Glm5KdaLayer::new(&cfg, tiny_weights(&cfg)).expect("seat a");
        let mut b = Glm5KdaLayer::new(&cfg, tiny_weights(&cfg)).expect("seat b");
        for i in 0..3u64 {
            let h = rand_vec(200 + i, d);
            let oa = a.forward_token(&h).expect("a").to_vec();
            let ob = b.forward_token(&h).expect("b").to_vec();
            assert_eq!(oa, ob, "token {i} diverged between identical seats");
        }
    }

    #[test]
    fn lower_bound_posture_is_the_seat_gate() {
        // The seat rides the lower-bound entry point by construction: run a
        // longer sequence with a positive-drift hidden state (large positive
        // g region, where softplus ~ identity would over-decay) and assert
        // outputs stay finite with the gate form in play. The FORM-level
        // divergence test lives on the substrate; this pins the seat's
        // posture wiring (the bound travels from the config).
        let cfg = tiny_kda_cfg();
        assert_eq!(cfg.linear_lower_bound, Some(-5.0));
        let d = cfg.hidden_size;
        let mut seat = Glm5KdaLayer::new(&cfg, tiny_weights(&cfg)).expect("seat");
        for i in 0..8u64 {
            let mut h = rand_vec(300 + i, d);
            for v in h.iter_mut() {
                *v += 0.5;
            }
            let out = seat.forward_token(&h).expect("decode").to_vec();
            assert!(out.iter().all(|v| v.is_finite()), "token {i} non-finite");
        }
    }
}
