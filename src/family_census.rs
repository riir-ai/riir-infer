//! Tensor-family taxonomy for the qwen35-hybrid GGUF grammar (Issue 1188
//! T0 — the mixed-precision sensitivity ranking tool's classification
//! substrate).
//!
//! The tool ranks model tensor FAMILIES by their Δ-NLL under a candidate
//! quantization posture; the family of a tensor is derived from its GGUF
//! tensor NAME (the [`plan402_gguf_probe.py` precedent: read the model
//! BEFORE building against it]). The grammar is the one
//! [`crate::gguf_loader`] consumes for `qwen35` hybrid checkpoints
//! (`qwen35_deltanet_gguf_names` / `qwen35_attention_gguf_names`):
//! `token_embd.weight`, `output.weight`, `blk.N.<suffix>`.
//!
//! Every family carries a [`StorageClass`] — the weights field its tensors
//! materialize into — which decides how a candidate posture applies:
//! [`StorageClass::Proj`] / [`StorageClass::GateProj`] tensors re-quantize
//! quant-resident (the serving shape); [`StorageClass::Vector`] tensors are
//! plain f32 at runtime, so their candidate read is a quantize→dequant
//! round-trip (storage-precision sensitivity, forward untouched).
//!
//! Instrument-grade: classification only, never a serving path.

use crate::gguf_loader::{GgufFile, GgmlType};
use std::fmt;

/// A model tensor family (the ranking unit).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Family {
    /// `token_embd.weight` — the embedding table (tied lm_head reads it too).
    TokenEmbd,
    /// `output.weight` — the untied lm_head (absent on tied checkpoints).
    Output,
    /// Full-attention layer Q projection (gated: q fused with its gate, Issue 594).
    AttnQ,
    /// Full-attention layer K projection.
    AttnK,
    /// Full-attention layer V projection.
    AttnV,
    /// Full-attention layer output projection.
    AttnOut,
    /// DeltaNet z gate (`blk.N.attn_gate.weight` → `in_proj_z`).
    AttnGate,
    /// DeltaNet fused qkv input projection (`blk.N.attn_qkv.weight`).
    GdnQkv,
    /// DeltaNet `in_proj_a` (`blk.N.ssm_alpha.weight`).
    GdnAlpha,
    /// DeltaNet `in_proj_b` (`blk.N.ssm_beta.weight`).
    GdnBeta,
    /// DeltaNet output projection (`blk.N.ssm_out.weight`).
    GdnOut,
    /// SwiGLU gate projection (both layer kinds).
    FfnGate,
    /// SwiGLU up projection (both layer kinds).
    FfnUp,
    /// SwiGLU down projection (both layer kinds).
    FfnDown,
    /// DeltaNet conv1d weight (`blk.N.ssm_conv1d.weight`).
    GdnConv,
    /// DeltaNet time-step bias (`blk.N.ssm_dt.bias`).
    GdnDt,
    /// DeltaNet `a_log` (`blk.N.ssm_a`).
    GdnA,
    /// Every RMS norm vector (input / post-attention / q/k / linear / final).
    Norms,
}

impl Family {
    /// Every family, census order (globals first, then per-layer groups,
    /// then the vector classes — the order the ranked table prints in).
    pub const ALL: [Family; 18] = [
        Family::TokenEmbd,
        Family::Output,
        Family::AttnQ,
        Family::AttnK,
        Family::AttnV,
        Family::AttnOut,
        Family::AttnGate,
        Family::GdnQkv,
        Family::GdnAlpha,
        Family::GdnBeta,
        Family::GdnOut,
        Family::FfnGate,
        Family::FfnUp,
        Family::FfnDown,
        Family::GdnConv,
        Family::GdnDt,
        Family::GdnA,
        Family::Norms,
    ];

    /// The CLI/census spelling (snake; `parse` is its exact inverse).
    pub const fn slug(self) -> &'static str {
        match self {
            Family::TokenEmbd => "token_embd",
            Family::Output => "output",
            Family::AttnQ => "attn_q",
            Family::AttnK => "attn_k",
            Family::AttnV => "attn_v",
            Family::AttnOut => "attn_out",
            Family::AttnGate => "attn_gate",
            Family::GdnQkv => "gdn_qkv",
            Family::GdnAlpha => "gdn_alpha",
            Family::GdnBeta => "gdn_beta",
            Family::GdnOut => "gdn_out",
            Family::FfnGate => "ffn_gate",
            Family::FfnUp => "ffn_up",
            Family::FfnDown => "ffn_down",
            Family::GdnConv => "gdn_conv",
            Family::GdnDt => "gdn_dt",
            Family::GdnA => "gdn_a",
            Family::Norms => "norms",
        }
    }

    /// Parse a CLI family name (the inverse of [`Family::slug`]).
    pub fn parse(s: &str) -> Option<Family> {
        Family::ALL.iter().copied().find(|f| f.slug() == s)
    }

    /// How this family's tensors materialize into the weights struct.
    pub const fn storage_class(self) -> StorageClass {
        match self {
            Family::TokenEmbd
            | Family::Output
            | Family::AttnQ
            | Family::AttnK
            | Family::AttnV
            | Family::AttnOut
            | Family::AttnGate
            | Family::GdnQkv
            | Family::GdnOut
            | Family::FfnGate
            | Family::FfnUp
            | Family::FfnDown => StorageClass::Proj,
            Family::GdnAlpha | Family::GdnBeta => StorageClass::GateProj,
            Family::GdnConv | Family::GdnDt | Family::GdnA | Family::Norms => StorageClass::Vector,
        }
    }
}

impl fmt::Display for Family {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.slug())
    }
}

/// The weights field a family's tensors materialize into — decides how a
/// candidate posture applies ([`Family::storage_class`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageClass {
    /// A 2-D projection in `ProjWeights` (quant-resident re-quantization).
    Proj,
    /// A 2-D projection in `GateProjWeights` (the DeltaNet a/b gates).
    GateProj,
    /// A plain runtime-f32 vector — the candidate read is a
    /// quantize→dequant round-trip (forward code untouched).
    Vector,
}

impl fmt::Display for StorageClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StorageClass::Proj => f.write_str("proj"),
            StorageClass::GateProj => f.write_str("gate_proj"),
            StorageClass::Vector => f.write_str("vector"),
        }
    }
}

/// One classified tensor: family + layer index (None for globals).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Classified {
    pub family: Family,
    /// `blk.N` layer index; `None` for the global tensors.
    pub layer: Option<usize>,
}

/// Classify one qwen35-hybrid GGUF tensor name.
///
/// Unknown names are a typed miss ([`None`]-equivalent via `Err`) — the
/// census REPORTS them (a census reads), it never guesses. The matcher is
/// EXACT-suffix on the post-`blk.N.` remainder (never `contains`: `ssm_a`
/// is a strict prefix of `ssm_alpha.weight`, and a contains-match would
/// misfile it).
pub fn classify(name: &str) -> Option<Classified> {
    let (layer, rest) = match name.strip_prefix("blk.") {
        Some(after) => {
            let dot = after.find('.')?;
            let idx: usize = after[..dot].parse().ok()?;
            (Some(idx), &after[dot + 1..])
        }
        None => (None, name),
    };
    // Globals are ONLY the three bare names — a per-layer suffix without
    // its `blk.N.` prefix is foreign grammar and must MISS (never guess).
    if layer.is_none()
        && !matches!(rest, "token_embd.weight" | "output.weight" | "output_norm.weight")
    {
        return None;
    }
    let family = match rest {
        "token_embd.weight" => Family::TokenEmbd,
        "output.weight" => Family::Output,
        "attn_q.weight" => Family::AttnQ,
        "attn_k.weight" => Family::AttnK,
        "attn_v.weight" => Family::AttnV,
        "attn_output.weight" => Family::AttnOut,
        "attn_gate.weight" => Family::AttnGate,
        "attn_qkv.weight" => Family::GdnQkv,
        "ssm_alpha.weight" => Family::GdnAlpha,
        "ssm_beta.weight" => Family::GdnBeta,
        "ssm_out.weight" => Family::GdnOut,
        "ffn_gate.weight" => Family::FfnGate,
        "ffn_up.weight" => Family::FfnUp,
        "ffn_down.weight" => Family::FfnDown,
        "ssm_conv1d.weight" => Family::GdnConv,
        "ssm_dt.bias" => Family::GdnDt,
        "ssm_a" => Family::GdnA,
        "attn_norm.weight"
        | "post_attention_norm.weight"
        | "ssm_norm.weight"
        | "attn_q_norm.weight"
        | "attn_k_norm.weight"
        | "output_norm.weight" => Family::Norms,
        _ => return None,
    };
    Some(Classified { family, layer })
}

/// One census row (per tensor).
#[derive(Debug, Clone)]
pub struct CensusRow {
    pub name: String,
    pub classified: Option<Classified>,
    pub ggml_type: GgmlType,
    /// GGML shape order (ne0 = in-features/innermost first).
    pub shape: Vec<usize>,
    pub byte_len: u64,
}

/// The per-family rollup (the ranked table's input).
#[derive(Debug, Clone)]
pub struct FamilyRollup {
    pub family: Family,
    pub tensors: usize,
    pub elements: u64,
    pub bytes: u64,
    /// Distinct GGML types across the family's tensors (the incumbent
    /// posture evidence, e.g. "Q2_0" or "BF16,Dense-vector mix").
    pub types: Vec<String>,
}

impl FamilyRollup {
    fn empty(family: Family) -> Self {
        Self {
            family,
            tensors: 0,
            elements: 0,
            bytes: 0,
            types: Vec::new(),
        }
    }
}

/// The full census over an open GGUF.
#[derive(Debug, Clone)]
pub struct FamilyCensus {
    pub rows: Vec<CensusRow>,
    pub unclassified: usize,
    pub rollups: Vec<FamilyRollup>,
}

/// Walk an open GGUF's tensor list and classify every name.
///
/// Total walk — the header is already parsed, this touches no tensor data.
pub fn census_of(file: &GgufFile) -> FamilyCensus {
    let mut rows: Vec<CensusRow> = Vec::with_capacity(file.tensor_infos.len());
    for info in &file.tensor_infos {
        rows.push(CensusRow {
            name: info.name.clone(),
            classified: classify(&info.name),
            ggml_type: info.ggml_type,
            shape: info.shape.clone(),
            byte_len: info.byte_len,
        });
    }
    let unclassified = rows.iter().filter(|r| r.classified.is_none()).count();

    // Rollup per family, in Family::ALL order, then the unclassified tail.
    let mut rollups: Vec<FamilyRollup> = Vec::with_capacity(Family::ALL.len() + 1);
    for family in Family::ALL {
        let mut r = FamilyRollup::empty(family);
        for row in &rows {
            if row.classified.map(|c| c.family) == Some(family) {
                r.tensors += 1;
                r.elements += row.shape.iter().product::<usize>() as u64;
                r.bytes += row.byte_len;
                let t = format!("{:?}", row.ggml_type);
                if !r.types.contains(&t) {
                    r.types.push(t);
                }
            }
        }
        if r.tensors > 0 {
            rollups.push(r);
        }
    }
    FamilyCensus {
        rows,
        unclassified,
        rollups,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The strict-prefix trap: `ssm_a` must not swallow `ssm_alpha.weight`
    /// (exact-suffix matching, never `contains`).
    #[test]
    fn ssm_a_does_not_swallow_ssm_alpha() {
        let a = classify("blk.3.ssm_a").expect("ssm_a classifies");
        assert_eq!(a.family, Family::GdnA);
        assert_eq!(a.layer, Some(3));
        let alpha = classify("blk.3.ssm_alpha.weight").expect("ssm_alpha classifies");
        assert_eq!(alpha.family, Family::GdnAlpha);
    }

    #[test]
    fn globals_and_norms_classify() {
        let t = classify("token_embd.weight").expect("token_embd");
        assert_eq!(t.family, Family::TokenEmbd);
        assert_eq!(t.layer, None);
        let o = classify("output.weight").expect("output");
        assert_eq!(o.family, Family::Output);
        let on = classify("output_norm.weight").expect("output_norm");
        assert_eq!(on.family, Family::Norms);
        assert_eq!(on.layer, None);
    }

    #[test]
    fn attention_layer_names_classify() {
        for (name, family) in [
            ("blk.4.attn_q.weight", Family::AttnQ),
            ("blk.4.attn_k.weight", Family::AttnK),
            ("blk.4.attn_v.weight", Family::AttnV),
            ("blk.4.attn_output.weight", Family::AttnOut),
            ("blk.4.attn_q_norm.weight", Family::Norms),
            ("blk.4.attn_k_norm.weight", Family::Norms),
            ("blk.4.attn_norm.weight", Family::Norms),
            ("blk.4.post_attention_norm.weight", Family::Norms),
            ("blk.4.ffn_gate.weight", Family::FfnGate),
            ("blk.4.ffn_up.weight", Family::FfnUp),
            ("blk.4.ffn_down.weight", Family::FfnDown),
        ] {
            let c = classify(name)
                .unwrap_or_else(|| panic!("{name} must classify"));
            assert_eq!(c.family, family, "{name}");
            assert_eq!(c.layer, Some(4), "{name}");
        }
    }

    #[test]
    fn deltanet_layer_names_classify() {
        for (name, family) in [
            ("blk.5.attn_qkv.weight", Family::GdnQkv),
            ("blk.5.attn_gate.weight", Family::AttnGate),
            ("blk.5.ssm_out.weight", Family::GdnOut),
            ("blk.5.ssm_conv1d.weight", Family::GdnConv),
            ("blk.5.ssm_dt.bias", Family::GdnDt),
            ("blk.5.ssm_norm.weight", Family::Norms),
            ("blk.5.ffn_gate.weight", Family::FfnGate),
        ] {
            let c = classify(name)
                .unwrap_or_else(|| panic!("{name} must classify"));
            assert_eq!(c.family, family, "{name}");
            assert_eq!(c.layer, Some(5), "{name}");
        }
    }

    #[test]
    fn unknown_and_malformed_names_miss() {
        assert!(classify("blk.0.unknown_thing.weight").is_none());
        assert!(classify("blk.x.attn_q.weight").is_none(), "non-numeric layer");
        assert!(classify("blk.0").is_none(), "no dot after the layer index");
        assert!(classify("nextn.layer.0.weight").is_none(), "foreign grammar");
    }

    /// Every family's slug round-trips through `parse`, and the storage
    /// classes partition exactly as the ablation application expects
    /// (12 proj + 2 gate_proj + 4 vector = all 18).
    #[test]
    fn slugs_round_trip_and_classes_partition() {
        for f in Family::ALL {
            assert_eq!(Family::parse(f.slug()), Some(f), "{}", f.slug());
            assert!(
                matches!(
                    f.storage_class(),
                    StorageClass::Proj | StorageClass::GateProj | StorageClass::Vector
                ),
                "{} has no storage class",
                f.slug()
            );
        }
        let count = |want: StorageClass| {
            Family::ALL
                .iter()
                .filter(|f| f.storage_class() == want)
                .count()
        };
        assert_eq!(count(StorageClass::Proj), 12);
        assert_eq!(count(StorageClass::GateProj), 2);
        assert_eq!(count(StorageClass::Vector), 4);
    }
}
