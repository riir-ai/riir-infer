//! GLM-5.3-Flash MoE-block CPU-reference seat — Plan 627 Phase 1 T5.
//!
//! A thin mapping layer over the UPSTREAM MoE substrate
//! (`katgpt_transformer::moe` — `MoeConfig`, `MoeWeights`, `MoeScratch`... —
//! the DeepSeek-V3 `noaux_tc`-class sigmoid router with the selection-only
//! bias law already G1-gated there), specialized to the glm5 grammar:
//!
//! - The gate is `DeepseekV3TopkRouter.forward` VERBATIM (GLM's
//!   `Glm5NextTextTopkRouter` inherits it unchanged, `topk_method` deleted):
//!   fp32 router matmul → `sigmoid` scores (never softmax) → biased scores
//!   `s + b` for SELECTION only → raw-score renormalization
//!   `/ (Σ + 1e-20)` (`norm_topk_prob: true`) → `× routed_scaling_factor`
//!   (2.5). The grouped selection degenerates at the transformers defaults
//!   `n_group = 1` / `topk_group = 1` to a plain biased top-K over all
//!   experts — which is exactly the substrate's `select_topk_indices`.
//!   The seat REFUSES `n_group ≠ 1` loud (the grouped lane is not
//!   implemented — fail-closed, never a silent wrong posture).
//! - The experts are `Glm5NextTextExperts`: per-expert SEPARATE
//!   `gate/up/down` triplets (T0's inventory grammar — the checkpoint's own
//!   layout, not the Qwen/MiniMax packed `gate_up_proj`), clamped-SwiGLU
//!   activation `gate.clamp(max=10) · up.clamp(±10) → silu(gate)·up`
//!   (`swiglu_limit` 10.0), routed at the FULL hidden dim (NO latent MoE
//!   wrapper — `routed_expert_hidden_size: None`).
//! - The shared expert is `Glm5NextTextMLP` at
//!   `moe_intermediate_size × n_shared_experts` — the SAME clamped-SwiGLU
//!   class, always on, ungated, added AFTER the routed sum
//!   (`DeepseekV2Moe.forward`: `routed + shared(residuals)`).
//! - Numerics note: the substrate's renorm carries a uniform-1/K fallback
//!   when the selected raw-score sum < 1e-20; the V3 source divides by
//!   `Σ + 1e-20` unconditionally. The two forms agree everywhere in the
//!   reachable domain (sigmoid scores sum < 1e-20 requires all-eight logits
//    ≤ −30 — dead zone); the substrate's form is kept (incumbent
//!   byte-identity) and the difference is disclosed here. Real-capture
//!   parity adjudicates.
//!
//! Consumed, never forked: the router math, the selection-only bias, the
//! renormalization, the scaling, the expert dispatch, and the shared/routed
//! accumulation are the substrate's own — the seat adds only the config
//! mapping (with the transformers defaults), the shape gates, and the
//! separate-triplet expert layout.
//!
//! Feature `glm5_moe` implies `glm5` + `katgpt-transformer/transformer_moe`.

use katgpt_transformer::moe::{
    MoeConfig, MoeForwardScratch, MoeWeights, SwiGluExpertWeights, moe_forward_token,
};

use crate::glm5::Glm5Config;

// ---------------------------------------------------------------------------
// Config conversion (glm5 MoE facts → the upstream MoeConfig vocabulary)
// ---------------------------------------------------------------------------

impl Glm5Config {
    /// Build the substrate `MoeConfig` from the parsed MoE keys. Absent keys
    /// take the transformers documented defaults (see the field docs on
    /// [`Glm5Config`]); a present `n_group ≠ 1` refuses loud.
    pub fn to_moe_config(&self) -> Result<MoeConfig, String> {
        let missing = |name: &str| format!("MoE geometry missing: {name} is not set");
        let n_routed = self
            .n_routed_experts
            .ok_or_else(|| missing("n_routed_experts"))?;
        let k = self
            .num_experts_per_tok
            .ok_or_else(|| missing("num_experts_per_tok"))?;
        let n_shared = self
            .n_shared_experts
            .ok_or_else(|| missing("n_shared_experts"))?;
        let moe_inter = self
            .moe_intermediate_size
            .ok_or_else(|| missing("moe_intermediate_size"))?;
        let n_group = self.n_group.unwrap_or(1);
        if n_group != 1 {
            return Err(format!(
                "MoE n_group = {n_group} unsupported: the grouped-selection lane \
                 is not implemented (glm5 default is 1 — fail-closed, never a \
                 silent wrong posture)"
            ));
        }
        Ok(MoeConfig {
            num_experts: n_routed,
            num_shared_experts: n_shared,
            num_experts_per_token: k,
            moe_intermediate_size: moe_inter,
            hidden_size: self.hidden_size,
            use_sigmoid_router: true,
            renormalize: self.norm_topk_prob.unwrap_or(true),
            routed_expert_hidden_size: None,
            latent_moe_use_norm: false,
            rms_norm_eps: self.rms_norm_eps,
            situ_beta: 1.0,
            situ_linear_beta: None,
            routed_scaling_factor: Some(self.routed_scaling_factor.unwrap_or(2.5)),
            swiglu_limit: Some(self.swiglu_limit.unwrap_or(10.0)),
        })
    }
}

// ---------------------------------------------------------------------------
// Weight assembly (glm5 separate-triplet experts → the substrate MoeWeights)
// ---------------------------------------------------------------------------

/// The glm5 MoE layer weights — the separate per-expert triplets from the
/// checkpoint (T0 grammar: `mlp.experts.{e}.{gate,up,down}_proj.weight`,
/// `mlp.shared_experts.*`, `mlp.gate.weight` + `e_score_correction_bias`).
#[derive(Clone, Debug)]
pub struct Glm5MoeWeights {
    /// Router matrix (`mlp.gate.weight`), row-major `[E, d]`.
    pub router_weight: Vec<f32>,
    /// Selection-only bias (`mlp.gate.e_score_correction_bias`), `[E]`.
    pub e_score_correction_bias: Vec<f32>,
    /// Routed experts, one `gate/up/down` triplet each (`E` triplets; gate/up
    /// `[moe_i, d]`, down `[d, moe_i]`).
    pub experts: Vec<SwiGluExpertWeights>,
    /// The always-on shared expert triplet (`moe_i·n_shared` intermediate).
    pub shared_experts: SwiGluExpertWeights,
}

impl Glm5MoeWeights {
    /// Length validation against the substrate config — refuses loud naming
    /// the first mismatch.
    pub fn validate(&self, cfg: &MoeConfig) -> Result<(), String> {
        let e = cfg.n_routed();
        let d = cfg.d();
        let moe_i = cfg.d_ffn();
        let shared_i = cfg.d_ffn_shared();
        if self.router_weight.len() != e * d {
            return Err(format!(
                "router weight length {}: expected E·d = {e}·{d} = {}",
                self.router_weight.len(),
                e * d
            ));
        }
        if self.e_score_correction_bias.len() != e {
            return Err(format!(
                "e_score_correction_bias length {}: expected E = {e}",
                self.e_score_correction_bias.len()
            ));
        }
        if self.experts.len() != e {
            return Err(format!(
                "expert count {}: expected E = {e}",
                self.experts.len()
            ));
        }
        for (i, ex) in self.experts.iter().enumerate() {
            if ex.gate_proj.len() != moe_i * d
                || ex.up_proj.len() != moe_i * d
                || ex.down_proj.len() != d * moe_i
            {
                return Err(format!(
                    "expert {i} shapes wrong: gate {} up {} down {} — expected \
                     gate/up {moe_i}·{d}, down {d}·{moe_i}",
                    ex.gate_proj.len(),
                    ex.up_proj.len(),
                    ex.down_proj.len()
                ));
            }
        }
        if self.shared_experts.gate_proj.len() != shared_i * d
            || self.shared_experts.up_proj.len() != shared_i * d
            || self.shared_experts.down_proj.len() != d * shared_i
        {
            return Err(format!(
                "shared expert shapes wrong: gate {} up {} down {} — expected \
                 gate/up {shared_i}·{d}, down {d}·{shared_i}",
                self.shared_experts.gate_proj.len(),
                self.shared_experts.up_proj.len(),
                self.shared_experts.down_proj.len()
            ));
        }
        Ok(())
    }

    /// Convert ONCE into the substrate `MoeWeights` vocabulary (the hot path
    /// consumes the converted form — never a per-call clone of the expert
    /// matrices; routed_expert_* stay `None`, the GLM shape has no latent
    /// wrapper).
    pub fn into_substrate(self) -> MoeWeights {
        MoeWeights {
            router_weight: self.router_weight,
            e_score_correction_bias: self.e_score_correction_bias,
            experts: self.experts,
            shared_experts: vec![self.shared_experts],
            routed_expert_down_proj: None,
            routed_expert_up_proj: None,
            routed_expert_norm_weight: None,
        }
    }
}

// ---------------------------------------------------------------------------
// The seat
// ---------------------------------------------------------------------------

/// The glm5 MoE block: the substrate config plus the CONVERTED substrate
/// weights (converted ONCE at construction — the hot path never re-maps).
pub struct Glm5Moe {
    pub cfg: MoeConfig,
    pub weights: MoeWeights,
}

impl Glm5Moe {
    /// Assemble + validate + convert; refuses loud on any shape mismatch or
    /// an unsupported config (n_group ≠ 1).
    pub fn new(cfg: &Glm5Config, weights: Glm5MoeWeights) -> Result<Self, String> {
        let mcfg = cfg.to_moe_config()?;
        weights.validate(&mcfg)?;
        Ok(Self {
            cfg: mcfg,
            weights: weights.into_substrate(),
        })
    }

    /// Single-token MoE forward (delegates to the substrate's zero-alloc
    /// `moe_forward_token`): shared expert + scaled routed sum, NO residual
    /// add (the caller owns the residual, as in the source decoder).
    pub fn forward_token(
        &self,
        hidden_in: &[f32],
        hidden_out: &mut [f32],
        scratch: &mut MoeForwardScratch,
    ) {
        moe_forward_token(&self.weights, &self.cfg, hidden_in, hidden_out, scratch);
    }

    /// A fresh scratch sized for this config.
    pub fn scratch(&self) -> MoeForwardScratch {
        MoeForwardScratch::new(&self.cfg)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny but structurally EXACT glm5 MoE shape: real ratios (E=8, k=2,
    /// moe_i=8, d=16, shared 1), the transformers defaults absent (the seat
    /// must fill them).
    fn tiny_moe_cfg() -> Glm5Config {
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
                "n_routed_experts": 8,
                "n_shared_experts": 1,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 8,
                "rms_norm_eps": 1e-05
            }
        }"##;
        Glm5Config::parse(json).expect("tiny moe parse")
    }

    /// Deterministic pseudo-random f32 rows (LCG — the upstream convention).
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

    fn triplet(seed: u64, d_in: usize, d_out: usize) -> SwiGluExpertWeights {
        SwiGluExpertWeights {
            gate_proj: rand_rows(seed, d_out, d_in),
            up_proj: rand_rows(seed + 100, d_out, d_in),
            down_proj: rand_rows(seed + 200, d_in, d_out),
        }
    }

    fn tiny_weights(cfg: &MoeConfig) -> Glm5MoeWeights {
        let e = cfg.n_routed();
        let d = cfg.d();
        let moe_i = cfg.d_ffn();
        Glm5MoeWeights {
            router_weight: rand_rows(1, e, d),
            e_score_correction_bias: rand_rows(2, 1, e),
            experts: (0..e).map(|i| triplet(10 + i as u64, d, moe_i)).collect(),
            shared_experts: triplet(99, d, cfg.d_ffn_shared()),
        }
    }

    #[test]
    fn config_conversion_applies_the_transformers_defaults() {
        let cfg = tiny_moe_cfg();
        let m = cfg.to_moe_config().expect("conversion");
        assert_eq!(m.num_experts, 8);
        assert_eq!(m.num_experts_per_token, 2);
        assert_eq!(m.num_shared_experts, 1);
        assert_eq!(m.moe_intermediate_size, 8);
        assert_eq!(m.hidden_size, 16);
        assert!(m.use_sigmoid_router, "glm5 router is sigmoid");
        assert!(m.renormalize, "norm_topk_prob defaults true");
        assert_eq!(m.routed_expert_hidden_size, None, "NO latent wrapper");
        assert_eq!(m.routed_scaling_factor, Some(2.5), "GlmMoeDsa default");
        assert_eq!(m.swiglu_limit, Some(10.0), "Glm5NextText default");
        // d_ffn_shared = moe_i * n_shared = 8.
        assert_eq!(m.d_ffn_shared(), 8);
    }

    #[test]
    fn config_conversion_refuses_missing_geometry_and_n_group_ne_1_loud() {
        let mut cfg = tiny_moe_cfg();
        cfg.n_routed_experts = None;
        assert!(cfg.to_moe_config().is_err());
        let mut cfg = tiny_moe_cfg();
        cfg.moe_intermediate_size = None;
        assert!(cfg.to_moe_config().is_err());
        let mut cfg = tiny_moe_cfg();
        cfg.n_group = Some(4);
        let err = cfg.to_moe_config().unwrap_err();
        assert!(err.contains("n_group"), "refusal must name the knob: {err}");
    }

    #[test]
    fn seat_assembles_and_refuses_bad_shapes_loud() {
        let cfg = tiny_moe_cfg();
        let mcfg = cfg.to_moe_config().expect("mcfg");
        let ok = tiny_weights(&mcfg);
        Glm5Moe::new(&cfg, ok.clone()).expect("seat");
        let mut bad = ok.clone();
        bad.router_weight.truncate(bad.router_weight.len() - 1);
        assert!(Glm5Moe::new(&cfg, bad).is_err());
        let mcfg2 = cfg.to_moe_config().unwrap();
        let ok2 = tiny_weights(&mcfg2);
        let mut bad_experts = ok2.clone();
        bad_experts.experts.pop();
        assert!(Glm5Moe::new(&cfg, bad_experts).is_err());
    }

    /// The selection-only bias law through the seat: a huge bias on one
    /// expert CHANGES the selection but the selected weights still renormalize
    /// from RAW sigmoid scores (sum to routed_scaling_factor = 2.5).
    #[test]
    fn bias_is_selection_only_and_scaling_lands_in_the_weights() {
        let cfg = tiny_moe_cfg();
        let mcfg = cfg.to_moe_config().unwrap();
        let mut w = tiny_weights(&mcfg);
        w.e_score_correction_bias = vec![0.0; mcfg.n_routed()];
        let seat = Glm5Moe::new(&cfg, w).expect("seat");
        let hidden = vec![0.1; seat.cfg.d()];
        let mut scratch = seat.scratch();
        let mut out = vec![0.0f32; seat.cfg.d()];
        seat.forward_token(&hidden, &mut out, &mut scratch);
        let sum_plain: f32 = scratch.topk_weights.iter().sum();
        assert!(
            (sum_plain - 2.5).abs() < 1e-4,
            "raw-score renorm × 2.5: {sum_plain}"
        );
        // Now bias one expert hugely — it must enter the selection.
        let mcfg2 = cfg.to_moe_config().unwrap();
        let mut w2 = tiny_weights(&mcfg2);
        w2.e_score_correction_bias[3] = 100.0;
        let seat2 = Glm5Moe::new(&cfg, w2).expect("seat2");
        let mut scratch2 = seat2.scratch();
        let mut out2 = vec![0.0f32; seat2.cfg.d()];
        seat2.forward_token(&hidden, &mut out2, &mut scratch2);
        assert!(
            scratch2.topk_indices.contains(&3),
            "biased expert must be selected: {:?}",
            scratch2.topk_indices
        );
        // ...but the weights STILL sum to 2.5 (bias never leaks into them).
        let sum_biased: f32 = scratch2.topk_weights.iter().sum();
        assert!((sum_biased - 2.5).abs() < 1e-4, "{sum_biased}");
    }

    /// Determinism: two runs bit-identical; output finite.
    #[test]
    fn forward_is_deterministic_and_finite() {
        let cfg = tiny_moe_cfg();
        let mcfg = cfg.to_moe_config().unwrap();
        let seat = Glm5Moe::new(&cfg, tiny_weights(&mcfg)).expect("seat");
        let hidden: Vec<f32> = (0..seat.cfg.d()).map(|i| (i as f32 * 0.21).sin()).collect();
        let run = || {
            let mut scratch = seat.scratch();
            let mut out = vec![0.0f32; seat.cfg.d()];
            seat.forward_token(&hidden, &mut out, &mut scratch);
            out
        };
        let a = run();
        let b = run();
        assert_eq!(a, b, "output bits drifted");
        assert!(a.iter().all(|v| v.is_finite()));
    }

    /// The seat's forward is BYTE-IDENTICAL to a direct substrate call over
    /// the same converted config + weights (thin-delegation pin — the seat
    /// adds mapping, never math).
    #[test]
    fn seat_forward_delegates_bit_identically_to_the_substrate() {
        let cfg = tiny_moe_cfg();
        let mcfg = cfg.to_moe_config().unwrap();
        let raw = tiny_weights(&mcfg);
        let direct = raw.clone().into_substrate();
        let seat = Glm5Moe::new(&cfg, raw).expect("seat");
        let hidden: Vec<f32> = (0..seat.cfg.d()).map(|i| (i as f32 * 0.17).cos()).collect();
        let mut a = vec![0.0f32; seat.cfg.d()];
        let mut sc_a = seat.scratch();
        seat.forward_token(&hidden, &mut a, &mut sc_a);
        let mut b = vec![0.0f32; seat.cfg.d()];
        let mut sc_b = MoeForwardScratch::new(&mcfg);
        moe_forward_token(&direct, &mcfg, &hidden, &mut b, &mut sc_b);
        assert_eq!(a, b);
    }
}
