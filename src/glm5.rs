//! GLM-5.3-Flash (`glm5_next`) arch vocabulary — Plan 627 Phase 1 T0.
//!
//! The typed config + layer-kind dispatch + safetensors inventory
//! validation for the GLM-5.3-Flash hybrid arch (34 KDA linear-attention
//! layers + 11 DSA/MLA sparse-attention layers, 288-expert sigmoid MoE,
//! mHC hyper-connections, MTP draft layer). Everything here is
//! METADATA-ONLY: parsed from the HF `config.json` shape and validated
//! against the safetensors index `weight_map` — no weights are loaded,
//! no forward runs (the forward legs re-claim per the plan's lease law).
//!
//! Substrate-first verdict (Plan 627, 2026-11-28): the MLA vocabulary is
//! owned upstream by `katgpt-attn::mla::MlaConfig` — this module carries
//! the glm5-side config FACTS and converts to the upstream type at the
//! forward leg; the loader substrate is `crate::safetensors_loader`; the
//! KDA recurrence consumes `crate::deltanet` at the forward leg. Zero
//! local MLA forking.
//!
//! Name grammar (probed from the real checkpoint index, 76,108 tensors,
//! pin sha 46a050bec13002ad29ee795c16e61cf4fa3c4a9c):
//! - prefix `model.language_model.` (the multimodal wrapper nests the
//!   text stack under `language_model`); the visual tower (`model.visual.*`)
//!   is NOT validated here — the text lane defers multimodal (plan Phase 1).
//! - fp8 quant rows carry a `weight_scale_inv` twin (DeepSeek fp8 convention).

use std::collections::BTreeMap;

use serde::Deserialize;

// ---------------------------------------------------------------------------
// Config vocabulary (serde over the HF config.json shape)
// ---------------------------------------------------------------------------

/// GLM-5.3-Flash text-stack configuration (the fields the lane reads).
///
/// Fields not in this struct are ignored by serde — the probe pins the
/// full set; this struct carries what the loader/dispatch/forward consume.
#[derive(Debug, Clone, Deserialize)]
pub struct Glm5Config {
    pub model_type: String,
    pub num_hidden_layers: usize,
    /// Per-layer attention kind, in layer order (`linear_attention` |
    /// `deepseek_sparse_attention`).
    #[serde(default)]
    pub layer_types: Vec<String>,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    #[serde(default)]
    pub q_lora_rank: Option<usize>,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    /// glm5 ships 0 (NO rotary on attention; the trained indexer carries
    /// its own interleaved rope — plan finding 2).
    #[serde(default)]
    pub qk_rope_head_dim: usize,
    #[serde(default)]
    pub v_head_dim: Option<usize>,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    /// Dense-FFN layers before the MoE blocks (glm5: 3).
    #[serde(default)]
    pub first_k_dense_replace: usize,
    #[serde(default)]
    pub n_routed_experts: Option<usize>,
    #[serde(default)]
    pub n_shared_experts: Option<usize>,
    #[serde(default)]
    pub num_experts_per_tok: Option<usize>,
    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,
    #[serde(default)]
    pub intermediate_size: Option<usize>,
    #[serde(default)]
    pub num_nextn_predict_layers: Option<usize>,
    /// mHC hyper-connections (glm5: `hc_mult` 4, sinkhorn 20).
    #[serde(default)]
    pub hc_mult: Option<usize>,
    #[serde(default)]
    pub mhc: Option<bool>,
    /// mHC numerical floor (`hc_eps`, glm5: 1e-6) — added after the comb
    /// softmax and inside every sinkhorn divisor. The mHC seat refuses loud
    /// when `hc_mult` is set but this is absent.
    #[serde(default)]
    pub hc_eps: Option<f32>,
    /// mHC sinkhorn iteration count (`hc_sinkhorn_iters`, glm5: 20) — ONE
    /// column normalization before the loop, then `iters − 1` row/col pairs.
    #[serde(default)]
    pub hc_sinkhorn_iters: Option<usize>,
    /// RMSNorm epsilon for the latent norms (`q_a_layernorm` / `kv_a_layernorm`).
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    /// The trained KPool indexer, per DSA layer (plan finding 3).
    #[serde(default)]
    pub index_kpool: Option<usize>,
    #[serde(default)]
    pub index_topk: Option<usize>,
    #[serde(default)]
    pub index_head_dim: Option<usize>,
    #[serde(default)]
    pub index_n_heads: Option<usize>,
    /// KDA (linear-attention) geometry — the top-level `linear_*` keys
    /// (transformers `Glm5NextTextConfig`): heads × head_dim, the short-conv
    /// kernel, and the sigmoid forget-gate lower bound (glm5: 64 × 128,
    /// kernel 4, −5.0). The KDA seat (`glm5_kda`) refuses loud when a KDA
    /// layer exists but the geometry is absent.
    #[serde(default)]
    pub linear_num_heads: Option<usize>,
    #[serde(default)]
    pub linear_head_dim: Option<usize>,
    #[serde(default)]
    pub linear_conv_kernel_dim: Option<usize>,
    /// The sigmoid forget-gate lower bound — the transformers
    /// `linear_lower_bound` / `safe_gate_lower_bound` branch selector; `None`
    /// means the kimi softplus form (not glm5's posture).
    #[serde(default)]
    pub linear_lower_bound: Option<f32>,
}

fn default_rms_norm_eps() -> f32 {
    1e-5
}

/// The per-layer attention kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// KDA — gated delta-rule linear attention (the `deltanet` lane's
    /// reuse target).
    Kda,
    /// DSA — DeepSeek-style sparse MLA with the trained KPool indexer.
    Dsa,
}

impl Glm5Config {
    /// Parse from raw HF `config.json` bytes (nested or flat text config).
    /// Returns the TEXT config either way. Two explicit steps instead of
    /// serde flatten — flatten cannot express "nested if present, else the
    /// same fields inline".
    pub fn parse(json: &str) -> Result<Glm5Config, Glm5Error> {
        let v: serde_json::Value =
            serde_json::from_str(json).map_err(|e| Glm5Error::ConfigParse(e.to_string()))?;
        let text = v.get("text_config").unwrap_or(&v);
        serde_json::from_value(text.clone()).map_err(|e| Glm5Error::ConfigParse(e.to_string()))
    }

    /// Structural validation, loud and named (the loader refuses before
    /// any weight math runs). Accepts both the root spelling
    /// (`glm5_next`) and the text-stack spelling (`glm5_next_text`).
    pub fn validate(&self) -> Result<(), Glm5Error> {
        if self.model_type != "glm5_next_text" && self.model_type != "glm5_next" {
            return Err(Glm5Error::UnsupportedModelType(self.model_type.clone()));
        }
        if self.num_hidden_layers == 0 {
            return Err(Glm5Error::ZeroLayers);
        }
        if self.layer_types.len() != self.num_hidden_layers {
            return Err(Glm5Error::LayerTypeCount {
                expected: self.num_hidden_layers,
                got: self.layer_types.len(),
            });
        }
        // Every layer type must be one of the two known kinds, and the DSA
        // layers must match the config's kv/q latent geometry.
        for (i, lt) in self.layer_types.iter().enumerate() {
            match lt.as_str() {
                "linear_attention" => {}
                "deepseek_sparse_attention" => {}
                other => return Err(Glm5Error::UnknownLayerType(i, other.to_string())),
            }
        }
        if self.kv_lora_rank == 0 || self.qk_nope_head_dim == 0 {
            return Err(Glm5Error::ZeroLatentDims);
        }
        if self.dsa_layer_count() == 0 {
            return Err(Glm5Error::NoDsaLayers);
        }
        if self.hidden_size == 0 || self.vocab_size == 0 {
            return Err(Glm5Error::ZeroModelDims);
        }
        Ok(())
    }

    /// The per-layer kind table (length == `num_hidden_layers`).
    pub fn layer_kinds(&self) -> Vec<LayerKind> {
        self.layer_types
            .iter()
            .map(|lt| match lt.as_str() {
                "deepseek_sparse_attention" => LayerKind::Dsa,
                _ => LayerKind::Kda,
            })
            .collect()
    }

    /// Indices of the DSA/MLA layers (glm5: [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]).
    pub fn dsa_layers(&self) -> Vec<usize> {
        self.layer_kinds()
            .iter()
            .enumerate()
            .filter_map(|(i, k)| (*k == LayerKind::Dsa).then_some(i))
            .collect()
    }

    /// Indices of the KDA layers (glm5: 34 layers).
    pub fn kda_layers(&self) -> Vec<usize> {
        self.layer_kinds()
            .iter()
            .enumerate()
            .filter_map(|(i, k)| (*k == LayerKind::Kda).then_some(i))
            .collect()
    }

    pub fn dsa_layer_count(&self) -> usize {
        self.dsa_layers().len()
    }

    pub fn kda_layer_count(&self) -> usize {
        self.kda_layers().len()
    }

    /// The MoE layer count: layers after the dense prefix (the MTP layer,
    /// stored at index `num_hidden_layers`, is NOT counted — it is not in
    /// `layer_types`).
    pub fn moe_layer_count(&self) -> usize {
        self.num_hidden_layers
            .saturating_sub(self.first_k_dense_replace)
    }

    pub fn routed_experts(&self) -> usize {
        self.n_routed_experts.unwrap_or(0)
    }

    /// v-head dim (glm5 pins it equal to `qk_nope_head_dim`).
    pub fn v_head_dim(&self) -> usize {
        self.v_head_dim.unwrap_or(self.qk_nope_head_dim)
    }
}

// ---------------------------------------------------------------------------
// Errors (typed, loud)
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum Glm5Error {
    #[error("config parse failed: {0}")]
    ConfigParse(String),
    #[error("unsupported model_type `{0}` (this module pins glm5_next_text)")]
    UnsupportedModelType(String),
    #[error("num_hidden_layers is zero")]
    ZeroLayers,
    #[error("layer_types count {got} != num_hidden_layers {expected}")]
    LayerTypeCount { expected: usize, got: usize },
    #[error("layer {0}: unknown layer type `{1}`")]
    UnknownLayerType(usize, String),
    #[error("zero latent dims (kv_lora_rank/qk_nope_head_dim)")]
    ZeroLatentDims,
    #[error("no DSA layers — the sparse lane needs at least one")]
    NoDsaLayers,
    #[error("zero model dims (hidden_size/vocab_size)")]
    ZeroModelDims,
    #[error("inventory: {0}")]
    Inventory(String),
}

// ---------------------------------------------------------------------------
// Weight-inventory validation (metadata only — name grammar as data)
// ---------------------------------------------------------------------------

/// The expected text-stack tensor families, derived from the config.
///
/// Each row: (family suffix after `model.language_model.layers.{i}.`,
/// count expression). The prefix was probed from the real checkpoint
/// (the multimodal wrapper nests the text stack under `language_model`).
struct FamilySpec {
    /// Per-layer suffix with `{i}` substituted at validation time.
    per_layer: &'static [&'static str],
    /// Which layer indices carry it.
    layers: FamilyLayers,
}

enum FamilyLayers {
    /// Main layers 0..num_hidden_layers (NOT the MTP layer).
    MainLayersOnly,
    /// Main layers + the MTP layer (0..=num_hidden_layers).
    AllLayers,
    KdaOnly,
    DsaOnly,
    MoEOnly,
    DenseOnly,
    /// The MTP module (stored at layer index == num_hidden_layers).
    MtpOnly,
}

const FAMILY_ROOT: &[&str] = &[
    "model.language_model.embed_tokens.weight",
    "model.language_model.norm.weight",
    "lm_head.weight", // untied (glm5: tie_word_embeddings = false)
];

const FAMILY_MTP: &[&str] = &[
    "eh_proj.weight",
    "enorm.weight",
    "hnorm.weight",
    "shared_head.norm.weight",
];

/// The 16 DSA/MLA per-layer rows (norms + latents + the trained indexer),
/// spelled once as a const — used by the DSA family spec AND the MTP-row
/// insertion.
const MLA_FAMILY_ROWS: &[&str] = &[
    "self_attn.q_a_layernorm.weight",
    "self_attn.q_a_proj.weight",
    "self_attn.q_a_proj.weight_scale_inv",
    "self_attn.q_b_proj.weight",
    "self_attn.q_b_proj.weight_scale_inv",
    "self_attn.kv_a_layernorm.weight",
    "self_attn.kv_a_proj_with_mqa.weight",
    "self_attn.kv_a_proj_with_mqa.weight_scale_inv",
    "self_attn.kv_b_proj.weight",
    "self_attn.indexer.wk.weight",
    "self_attn.indexer.wq_b.weight",
    "self_attn.indexer.weights_proj.weight",
    "self_attn.indexer.k_norm.weight",
    "self_attn.indexer.k_norm.bias",
    "self_attn.indexer.index_kpool_compress_ape",
    "self_attn.indexer.index_kpool_compress_gate",
];

fn family_specs(cfg: &Glm5Config) -> Vec<FamilySpec> {
    let _ = cfg; // rows below are already layer-class-scoped
    vec![
        // Every layer (main + MTP): the two norms. (o_proj is its own
        // family: bare `.weight` on all layers incl. MTP + a scale twin on
        // the fp8 rows — DSA layers and MTP — probed 46 bare / 12 scale.)
        FamilySpec {
            per_layer: &["input_layernorm.weight", "post_attention_layernorm.weight"],
            layers: FamilyLayers::AllLayers,
        },
        // mHC hyper-connections: 6 parameter tensors per MAIN layer, no
        // `.weight` suffix. The MTP layer (index == num_hidden_layers)
        // carries NO mHC — probed: 45 rows per hc family on the real
        // checkpoint (layers 0..44 only).
        FamilySpec {
            per_layer: &[
                "hc_attn_base",
                "hc_attn_fn",
                "hc_attn_scale",
                "hc_ffn_base",
                "hc_ffn_fn",
                "hc_ffn_scale",
            ],
            layers: FamilyLayers::MainLayersOnly,
        },
        // KDA layers: gated delta-rule + short-conv machinery.
        FamilySpec {
            per_layer: &[
                "self_attn.A_log",
                "self_attn.dt_bias",
                "self_attn.b_proj.weight",
                "self_attn.f_a_proj.weight",
                "self_attn.f_b_proj.weight",
                "self_attn.g_a_proj.weight",
                "self_attn.g_b_proj.weight",
                "self_attn.q_conv1d.weight",
                "self_attn.k_conv1d.weight",
                "self_attn.v_conv1d.weight",
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_norm.weight",
            ],
            layers: FamilyLayers::KdaOnly,
        },
        // DSA layers: MLA latents + the trained KPool indexer.
        FamilySpec {
            per_layer: MLA_FAMILY_ROWS,
            layers: FamilyLayers::DsaOnly,
        },
        // MoE layers: router + shared experts (+ scale twins). The 288
        // routed experts are checked separately (per-expert triple).
        FamilySpec {
            per_layer: &[
                "mlp.gate.weight",
                "mlp.gate.e_score_correction_bias",
                "mlp.shared_experts.gate_proj.weight",
                "mlp.shared_experts.gate_proj.weight_scale_inv",
                "mlp.shared_experts.up_proj.weight",
                "mlp.shared_experts.up_proj.weight_scale_inv",
                "mlp.shared_experts.down_proj.weight",
                "mlp.shared_experts.down_proj.weight_scale_inv",
            ],
            layers: FamilyLayers::MoEOnly,
        },
        // Dense layers: plain GeGLU triplet (+ scale twins).
        FamilySpec {
            per_layer: &[
                "mlp.gate_proj.weight",
                "mlp.gate_proj.weight_scale_inv",
                "mlp.up_proj.weight",
                "mlp.up_proj.weight_scale_inv",
                "mlp.down_proj.weight",
                "mlp.down_proj.weight_scale_inv",
            ],
            layers: FamilyLayers::DenseOnly,
        },
        // MTP module.
        FamilySpec {
            per_layer: FAMILY_MTP,
            layers: FamilyLayers::MtpOnly,
        },
    ]
}

/// The per-expert rows (`mlp.experts.{e}.{gate,up,down}_proj.weight` +
/// `weight_scale_inv`) for every MoE layer.
fn expected_expert_rows(cfg: &Glm5Config, layers: &[usize]) -> Vec<String> {
    let mut rows = Vec::new();
    for &i in layers {
        for e in 0..cfg.routed_experts() {
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                rows.push(format!(
                    "model.language_model.layers.{i}.mlp.experts.{e}.{proj}.weight"
                ));
                rows.push(format!(
                    "model.language_model.layers.{i}.mlp.experts.{e}.{proj}.weight_scale_inv"
                ));
            }
        }
    }
    rows
}

/// Layer o_proj: every layer carries a bare `o_proj.weight` (bf16 rows on
/// KDA layers, the fp8 DATA rows on DSA/MTP); the fp8 layers (11 DSA +
/// MTP) additionally carry the `weight_scale_inv` twin — probed 46 bare /
/// 12 scale on the real checkpoint.
fn expected_o_proj_rows(cfg: &Glm5Config) -> Vec<String> {
    let mut rows = Vec::new();
    for i in 0..=cfg.num_hidden_layers {
        rows.push(format!(
            "model.language_model.layers.{i}.self_attn.o_proj.weight"
        ));
    }
    for &i in cfg.dsa_layers().iter() {
        rows.push(format!(
            "model.language_model.layers.{i}.self_attn.o_proj.weight_scale_inv"
        ));
    }
    rows.push(format!(
        "model.language_model.layers.{}.self_attn.o_proj.weight_scale_inv",
        cfg.num_hidden_layers
    ));
    rows
}

/// Result of an inventory validation: what was checked and what the
/// weight_map carried beyond the validated text-stack families.
#[derive(Debug, Default)]
pub struct InventoryReport {
    pub checked: usize,
    /// The visual tower + anything outside the text-stack grammar.
    pub outside_grammar: usize,
}

impl InventoryReport {
    pub fn is_text_stack_complete(&self) -> bool {
        true // only constructed on success
    }
}

/// Validate a safetensors `weight_map` (tensor name → shard file) against
/// the config: every expected text-stack family present, no missing row,
/// nothing unaccounted inside the text-stack prefix. The visual tower is
/// counted but not validated (multimodal deferred, plan Phase 1).
///
/// This is the metadata gate the REAL loader runs before touching any
/// shard bytes — cheap (name sets only), and it pins the name grammar the
/// forward leg's weight mapping consumes.
pub fn validate_weight_map(
    cfg: &Glm5Config,
    weight_map: &BTreeMap<String, String>,
) -> Result<InventoryReport, Glm5Error> {
    cfg.validate()?;

    let prefix = "model.language_model.layers.";
    let mut expected: BTreeMap<String, ()> = BTreeMap::new();

    let kda = cfg.kda_layers();
    let dsa = cfg.dsa_layers();
    let dense: Vec<usize> = (0..cfg.first_k_dense_replace).collect();
    // MoE rows live on every layer past the dense prefix AND on the MTP
    // layer (probed: router + shared experts + 288 routed experts at
    // index 45; the expert rows below share this list).
    let mut moe: Vec<usize> = (cfg.first_k_dense_replace..cfg.num_hidden_layers).collect();
    moe.push(cfg.num_hidden_layers);

    for spec in family_specs(cfg) {
        let layers: Vec<usize> = match spec.layers {
            FamilyLayers::MainLayersOnly => (0..cfg.num_hidden_layers).collect(),
            FamilyLayers::AllLayers => (0..=cfg.num_hidden_layers).collect(),
            FamilyLayers::KdaOnly => kda.clone(),
            FamilyLayers::DsaOnly => dsa.clone(),
            FamilyLayers::MoEOnly => moe.clone(),
            FamilyLayers::DenseOnly => dense.clone(),
            FamilyLayers::MtpOnly => vec![cfg.num_hidden_layers],
        };
        for &i in layers.iter() {
            for suffix in spec.per_layer {
                expected.insert(format!("{prefix}{i}.{suffix}"), ());
            }
        }
    }
    // DSA-family rows also ride the MTP layer (index 45 — probed:
    // the MTP module carries the full MLA + indexer set).
    for suffix in MLA_FAMILY_ROWS {
        let row = format!("{prefix}{}.{suffix}", cfg.num_hidden_layers);
        expected.insert(row, ());
    }
    for name in FAMILY_ROOT {
        expected.insert((*name).to_string(), ());
    }
    for row in expected_o_proj_rows(cfg) {
        expected.insert(row, ());
    }
    for row in expected_expert_rows(cfg, &moe) {
        expected.insert(row, ());
    }

    // Missing = any expected name absent from the map.
    let mut missing: Vec<&String> = expected
        .keys()
        .filter(|n| !weight_map.contains_key(n.as_str()))
        .collect();
    if !missing.is_empty() {
        missing.sort();
        let show: Vec<&String> = missing.iter().take(5).copied().collect();
        return Err(Glm5Error::Inventory(format!(
            "{} text-stack tensor(s) missing from the weight_map (first: {:?})",
            missing.len(),
            show
        )));
    }

    // Unaccounted = anything under the text-stack prefix that the grammar
    // did not expect (a foreign or renamed row — refuse loud, never skip).
    let mut unaccounted: Vec<&String> = weight_map
        .keys()
        .filter(|n| n.starts_with(prefix) || FAMILY_ROOT.contains(&n.as_str()))
        .filter(|n| !expected.contains_key(n.as_str()))
        .collect();
    if !unaccounted.is_empty() {
        unaccounted.sort();
        let show: Vec<&String> = unaccounted.iter().take(5).copied().collect();
        return Err(Glm5Error::Inventory(format!(
            "{} unaccounted text-stack tensor(s) in the weight_map (first: {:?})",
            unaccounted.len(),
            show
        )));
    }

    Ok(InventoryReport {
        checked: expected.len(),
        outside_grammar: weight_map
            .keys()
            .filter(|n| !(n.starts_with(prefix) || FAMILY_ROOT.contains(&n.as_str())))
            .count(),
    })
}

// ---------------------------------------------------------------------------
// Tests — golden fixture (the REAL probed numbers) + refusal battery
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// The REAL glm5_next_text config values, pinned from the probe of
    /// record (HF zai-org/GLM-5.3-Flash @ sha 46a050bec13002ad29ee795c16e61cf4fa3c4a9c,
    /// fetched 2026-11-28). Nested shape (as shipped).
    const GOLDEN_NESTED: &str = r##"{
        "architectures": ["Glm5NextForConditionalGeneration"],
        "model_type": "glm5_next",
        "text_config": {
            "model_type": "glm5_next_text",
            "num_hidden_layers": 45,
            "layer_types": ["linear_attention","linear_attention","linear_attention","deepseek_sparse_attention","linear_attention","linear_attention","linear_attention","deepseek_sparse_attention","linear_attention","linear_attention","linear_attention","deepseek_sparse_attention","linear_attention","linear_attention","linear_attention","deepseek_sparse_attention","linear_attention","linear_attention","linear_attention","deepseek_sparse_attention","linear_attention","linear_attention","linear_attention","deepseek_sparse_attention","linear_attention","linear_attention","linear_attention","deepseek_sparse_attention","linear_attention","linear_attention","linear_attention","deepseek_sparse_attention","linear_attention","linear_attention","linear_attention","deepseek_sparse_attention","linear_attention","linear_attention","linear_attention","deepseek_sparse_attention","linear_attention","linear_attention","linear_attention","deepseek_sparse_attention","linear_attention"],
            "hidden_size": 4096,
            "num_attention_heads": 64,
            "num_key_value_heads": 64,
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 256,
            "qk_rope_head_dim": 0,
            "v_head_dim": 256,
            "max_position_embeddings": 1048576,
            "vocab_size": 154880,
            "first_k_dense_replace": 3,
            "n_routed_experts": 288,
            "n_shared_experts": 1,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 2048,
            "intermediate_size": 12288,
            "num_nextn_predict_layers": 1,
            "hc_mult": 4,
            "mhc": true,
            "hc_eps": 1e-06,
            "hc_sinkhorn_iters": 20,
            "rms_norm_eps": 1e-05,
            "index_kpool": 4,
            "index_topk": 2048,
            "index_head_dim": 128,
            "index_n_heads": 32
        }
    }"##;

    #[test]
    fn golden_config_parses_and_validates() {
        let cfg = Glm5Config::parse(GOLDEN_NESTED).expect("golden parse");
        cfg.validate().expect("golden validate");
        assert_eq!(cfg.num_hidden_layers, 45);
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.kv_lora_rank, 512);
        assert_eq!(cfg.q_lora_rank, Some(1536));
        assert_eq!(cfg.qk_rope_head_dim, 0, "glm5 ships NO attention rope");
        assert_eq!(cfg.max_position_embeddings, 1_048_576);
        assert_eq!(cfg.vocab_size, 154_880);
        assert_eq!(cfg.first_k_dense_replace, 3);
        assert_eq!(cfg.routed_experts(), 288);
        assert_eq!(cfg.num_nextn_predict_layers, Some(1));
        assert_eq!(cfg.hc_mult, Some(4));
        assert_eq!(cfg.hc_eps, Some(1e-6));
        assert_eq!(cfg.hc_sinkhorn_iters, Some(20));
        assert_eq!(cfg.index_kpool, Some(4));
        assert_eq!(cfg.index_topk, Some(2048));
        assert_eq!(cfg.rms_norm_eps, 1e-5, "the real config ships rms_norm_eps");
    }

    #[test]
    fn flat_text_config_also_parses() {
        let nested = Glm5Config::parse(GOLDEN_NESTED).expect("nested parse");
        // Flatten: strip the wrapper, inline text_config at the root.
        let v: serde_json::Value = serde_json::from_str(GOLDEN_NESTED).unwrap();
        let mut flat = v["text_config"].clone();
        flat["model_type"] = serde_json::Value::String("glm5_next_text".into());
        let flat_cfg = Glm5Config::parse(&flat.to_string()).expect("flat parse");
        assert_eq!(flat_cfg.num_hidden_layers, nested.num_hidden_layers);
        assert_eq!(flat_cfg.kv_lora_rank, nested.kv_lora_rank);
    }

    #[test]
    fn layer_kind_table_matches_the_probe() {
        let cfg = Glm5Config::parse(GOLDEN_NESTED).unwrap();
        let dsa = cfg.dsa_layers();
        let kda = cfg.kda_layers();
        assert_eq!(
            dsa,
            vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43],
            "the every-4th-layer DSA split, probed"
        );
        assert_eq!(kda.len(), 34);
        assert_eq!(kda[0], 0);
        assert_eq!(kda[1], 1);
        assert_eq!(kda[2], 2);
        assert_eq!(kda[3], 4, "layer 3 is DSA; KDA resumes at 4");
        assert_eq!(*kda.last().unwrap(), 44);
        assert_eq!(cfg.moe_layer_count(), 42, "45 main layers - 3 dense");
    }

    #[test]
    fn refusal_battery_wrong_model_type() {
        let mut v: serde_json::Value = serde_json::from_str(GOLDEN_NESTED).unwrap();
        v["text_config"]["model_type"] = serde_json::Value::String("gemma2".into());
        let cfg = Glm5Config::parse(&v.to_string()).unwrap();
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("unsupported model_type"), "{err}");
    }

    #[test]
    fn refusal_battery_layer_count_mismatch() {
        let mut v: serde_json::Value = serde_json::from_str(GOLDEN_NESTED).unwrap();
        v["text_config"]["num_hidden_layers"] = serde_json::Value::from(44);
        let cfg = Glm5Config::parse(&v.to_string()).unwrap();
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("layer_types count"), "{err}");
    }

    #[test]
    fn refusal_battery_unknown_layer_type() {
        let mut v: serde_json::Value = serde_json::from_str(GOLDEN_NESTED).unwrap();
        v["text_config"]["layer_types"][10] = serde_json::Value::String("full_attention".into());
        let cfg = Glm5Config::parse(&v.to_string()).unwrap();
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("unknown layer type"), "{err}");
    }

    #[test]
    fn refusal_battery_garbage_json() {
        let err = Glm5Config::parse("{ not json").unwrap_err();
        assert!(err.to_string().contains("config parse failed"), "{err}");
    }

    /// A synthetic weight_map over a TINY config that exercises the exact
    /// grammar: 2 layers (1 KDA + 1 DSA), dense prefix 1, 2 experts, MTP.
    fn tiny_cfg() -> Glm5Config {
        let json = r##"{
            "model_type": "glm5_next",
            "text_config": {
                "model_type": "glm5_next_text",
                "num_hidden_layers": 2,
                "layer_types": ["linear_attention", "deepseek_sparse_attention"],
                "hidden_size": 64,
                "num_attention_heads": 4,
                "kv_lora_rank": 16,
                "qk_nope_head_dim": 8,
                "qk_rope_head_dim": 0,
                "max_position_embeddings": 4096,
                "vocab_size": 128,
                "first_k_dense_replace": 1,
                "n_routed_experts": 2,
                "n_shared_experts": 1,
                "num_experts_per_tok": 1,
                "num_nextn_predict_layers": 1
            }
        }"##;
        Glm5Config::parse(json).expect("tiny parse")
    }

    fn tiny_expected_names(cfg: &Glm5Config) -> Vec<String> {
        // The same expectation math the validator runs, spelled out for
        // the fixture (independent spelling of the grammar). The MTP
        // layer (index == num_hidden_layers): norms + bare o_proj + MLA
        // family + o_proj scale + MoE rows, but NO mHC (probed rule).
        let mtp = cfg.num_hidden_layers;
        let mut names = Vec::new();
        names.push("model.language_model.embed_tokens.weight".to_string());
        names.push("model.language_model.norm.weight".to_string());
        names.push("lm_head.weight".to_string());
        for i in 0..=cfg.num_hidden_layers {
            names.push(format!(
                "model.language_model.layers.{i}.input_layernorm.weight"
            ));
            names.push(format!(
                "model.language_model.layers.{i}.post_attention_layernorm.weight"
            ));
            names.push(format!(
                "model.language_model.layers.{i}.self_attn.o_proj.weight"
            ));
            if i < cfg.num_hidden_layers {
                for hc in [
                    "hc_attn_base",
                    "hc_attn_fn",
                    "hc_attn_scale",
                    "hc_ffn_base",
                    "hc_ffn_fn",
                    "hc_ffn_scale",
                ] {
                    names.push(format!("model.language_model.layers.{i}.{hc}"));
                }
            }
        }
        // KDA layer 0.
        for s in [
            "self_attn.A_log",
            "self_attn.dt_bias",
            "self_attn.b_proj.weight",
            "self_attn.f_a_proj.weight",
            "self_attn.f_b_proj.weight",
            "self_attn.g_a_proj.weight",
            "self_attn.g_b_proj.weight",
            "self_attn.q_conv1d.weight",
            "self_attn.k_conv1d.weight",
            "self_attn.v_conv1d.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_norm.weight",
        ] {
            names.push(format!("model.language_model.layers.0.{s}"));
        }
        // DSA-family rows: DSA layer 1 AND the MTP layer (both carry the
        // full MLA + indexer set).
        for i in [1, mtp] {
            for s in [
                "self_attn.q_a_layernorm.weight",
                "self_attn.q_a_proj.weight",
                "self_attn.q_a_proj.weight_scale_inv",
                "self_attn.q_b_proj.weight",
                "self_attn.q_b_proj.weight_scale_inv",
                "self_attn.kv_a_layernorm.weight",
                "self_attn.kv_a_proj_with_mqa.weight",
                "self_attn.kv_a_proj_with_mqa.weight_scale_inv",
                "self_attn.kv_b_proj.weight",
                "self_attn.indexer.wk.weight",
                "self_attn.indexer.wq_b.weight",
                "self_attn.indexer.weights_proj.weight",
                "self_attn.indexer.k_norm.weight",
                "self_attn.indexer.k_norm.bias",
                "self_attn.indexer.index_kpool_compress_ape",
                "self_attn.indexer.index_kpool_compress_gate",
                "self_attn.o_proj.weight_scale_inv",
            ] {
                names.push(format!("model.language_model.layers.{i}.{s}"));
            }
        }
        // Dense layer 0 mlp triplet; MoE rows on layer 1 AND the MTP layer
        // (probed rule: the MTP layer carries router + shared + routed
        // experts, but no mHC).
        for s in [
            "mlp.gate_proj.weight",
            "mlp.gate_proj.weight_scale_inv",
            "mlp.up_proj.weight",
            "mlp.up_proj.weight_scale_inv",
            "mlp.down_proj.weight",
            "mlp.down_proj.weight_scale_inv",
        ] {
            names.push(format!("model.language_model.layers.0.{s}"));
        }
        for i in [1, mtp] {
            for s in [
                "mlp.gate.weight",
                "mlp.gate.e_score_correction_bias",
                "mlp.shared_experts.gate_proj.weight",
                "mlp.shared_experts.gate_proj.weight_scale_inv",
                "mlp.shared_experts.up_proj.weight",
                "mlp.shared_experts.up_proj.weight_scale_inv",
                "mlp.shared_experts.down_proj.weight",
                "mlp.shared_experts.down_proj.weight_scale_inv",
            ] {
                names.push(format!("model.language_model.layers.{i}.{s}"));
            }
            for e in 0..cfg.routed_experts() {
                for proj in ["gate_proj", "up_proj", "down_proj"] {
                    names.push(format!(
                        "model.language_model.layers.{i}.mlp.experts.{e}.{proj}.weight"
                    ));
                    names.push(format!(
                        "model.language_model.layers.{i}.mlp.experts.{e}.{proj}.weight_scale_inv"
                    ));
                }
            }
        }
        // MTP module extras.
        for s in [
            "eh_proj.weight",
            "enorm.weight",
            "hnorm.weight",
            "shared_head.norm.weight",
        ] {
            names.push(format!("model.language_model.layers.{mtp}.{s}"));
        }
        names
    }

    #[test]
    fn inventory_accepts_a_complete_tiny_map() {
        let cfg = tiny_cfg();
        let map: BTreeMap<String, String> = tiny_expected_names(&cfg)
            .into_iter()
            .map(|n| (n, "model-00001-of-00001.safetensors".to_string()))
            .collect();
        let report = validate_weight_map(&cfg, &map).expect("tiny map validates");
        assert!(report.checked > 0);
        assert_eq!(report.outside_grammar, 0, "nothing outside the text stack");
    }

    #[test]
    fn inventory_refuses_mhc_on_the_mtp_layer() {
        // The probed grammar: the MTP layer carries NO mHC rows — the
        // validator must refuse one as unaccounted.
        let cfg = tiny_cfg();
        let mut map: BTreeMap<String, String> = tiny_expected_names(&cfg)
            .into_iter()
            .map(|n| (n, "s.safetensors".to_string()))
            .collect();
        map.insert(
            format!(
                "model.language_model.layers.{}.hc_attn_base",
                cfg.num_hidden_layers
            ),
            "s.safetensors".to_string(),
        );
        let err = validate_weight_map(&cfg, &map).unwrap_err();
        assert!(err.to_string().contains("unaccounted text-stack"), "{err}");
    }

    #[test]
    fn inventory_refuses_a_missing_row() {
        let cfg = tiny_cfg();
        let mut map: BTreeMap<String, String> = tiny_expected_names(&cfg)
            .into_iter()
            .map(|n| (n, "s.safetensors".to_string()))
            .collect();
        map.remove("model.language_model.layers.1.self_attn.indexer.wk.weight");
        let err = validate_weight_map(&cfg, &map).unwrap_err();
        assert!(
            err.to_string().contains("missing from the weight_map"),
            "{err}"
        );
    }

    #[test]
    fn inventory_refuses_an_unaccounted_text_stack_row() {
        let cfg = tiny_cfg();
        let mut map: BTreeMap<String, String> = tiny_expected_names(&cfg)
            .into_iter()
            .map(|n| (n, "s.safetensors".to_string()))
            .collect();
        map.insert(
            "model.language_model.layers.0.self_attn.rotary_emb.inv_freq".to_string(),
            "s.safetensors".to_string(),
        );
        let err = validate_weight_map(&cfg, &map).unwrap_err();
        assert!(err.to_string().contains("unaccounted text-stack"), "{err}");
    }

    #[test]
    fn inventory_ignores_the_visual_tower_but_counts_it() {
        let cfg = tiny_cfg();
        let mut map: BTreeMap<String, String> = tiny_expected_names(&cfg)
            .into_iter()
            .map(|n| (n, "s.safetensors".to_string()))
            .collect();
        map.insert(
            "model.visual.blocks.0.attn.qkv.weight".to_string(),
            "s.safetensors".to_string(),
        );
        let report = validate_weight_map(&cfg, &map).expect("visual rows pass through");
        assert_eq!(report.outside_grammar, 1);
    }

    #[test]
    fn inventory_refuses_without_validation_of_a_broken_config() {
        let mut cfg = tiny_cfg();
        cfg.num_hidden_layers = 99; // now layer_types.len() != layers
        let map: BTreeMap<String, String> = BTreeMap::new();
        let err = validate_weight_map(&cfg, &map).unwrap_err();
        assert!(err.to_string().contains("layer_types count"), "{err}");
    }

    /// The full-size expected-count math, cross-checked against the REAL
    /// probed checkpoint census (76,108 tensors total; 347 visual rows —
    /// exact-set diffed 2026-11-28, zero missing / zero extra).
    #[test]
    fn full_size_expectation_math_matches_the_probe_census() {
        let cfg = Glm5Config::parse(GOLDEN_NESTED).unwrap();

        let kda_rows = cfg.kda_layer_count() * 14;
        let dsa_rows = cfg.dsa_layer_count() * 16;
        let dsa_o_proj_scale_rows = cfg.dsa_layer_count(); // the fp8 twins
        let mtp_mla_rows = 16; // the MTP layer carries the full MLA set
        let mtp_o_proj_scale = 1;
        let norm_rows = (cfg.num_hidden_layers + 1) * 2;
        let o_proj_bare_rows = cfg.num_hidden_layers + 1;
        let mhc_rows = cfg.num_hidden_layers * 6; // MAIN layers only
        let root_rows = 3;
        let mtp_module_rows = 4;
        let dense_rows = cfg.first_k_dense_replace * 6;
        let moe_router_rows = (cfg.moe_layer_count() + 1) * 8; // + the MTP layer
        let expert_rows = (cfg.moe_layer_count() + 1) * cfg.routed_experts() * 3 * 2;

        let text_stack = kda_rows
            + dsa_rows
            + dsa_o_proj_scale_rows
            + mtp_mla_rows
            + mtp_o_proj_scale
            + norm_rows
            + o_proj_bare_rows
            + mhc_rows
            + root_rows
            + mtp_module_rows
            + dense_rows
            + moe_router_rows
            + expert_rows;

        // The real index: 76,108 tensors, 347 of them the visual tower
        // (exact set-diff against the grammar: zero missing, zero extra).
        let real_total: usize = 76_108;
        let real_visual: usize = 347;
        assert_eq!(
            text_stack,
            real_total - real_visual,
            "the grammar math must reproduce the probed census exactly"
        );
    }
}
