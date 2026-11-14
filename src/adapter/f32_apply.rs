//! Deletion-mask application to dense f32 weights (Issue 1208 T2).
//!
//! Two postures over the same `DecodedMask`, per the issue:
//!
//! 1. **Apply-at-load (preferred)** — [`apply_mask_at_load`] zeros the
//!    deleted weights inside the loaded weight copy. The copy diverges
//!    from the file; the divergence IS the mask.
//! 2. **In-forward overlay** — [`GateMaskOverlay`] holds masked SHADOW
//!    COPIES of the armed layers' gate projections; the base weights
//!    stay pristine and hot-swap is pointer-swap. Bit-exact vs
//!    apply-at-load by construction: the same kernel runs, only the
//!    data differs. (A custom masked kernel was REJECTED: it must
//!    mirror the upstream simd accumulation bit-for-bit and re-fails
//!    G1 on every kernel change — the G1 gate would be guarding the
//!    mirror, not the mask.)
//!
//! Bit-order contract (pinned with the codec): mask bit `w` covers
//! weight `w` of the tensor in row-major order — exactly the order of
//! `dequant_f16_to_f32` output and `Proj::dense` layout. A section's
//! `bit_len` must equal the tensor's weight count.
//!
//! Falsifier mapping: G1 (masked ≡ elementwise-zeroed reference, the
//! forward-level identity pin) lives in `tests/i1208_mask_apply.rs`;
//! G2 (load-wall %, decode tok/s delta) is league-cell measurement.

use super::mask_codec::{DecodedMask, MaskCodecError};
use crate::deltanet::weights::QwenDeltaNetWeights;
use crate::llama_layer::LlamaTransformerWeights;

/// The GGUF tensor name for layer `i`'s gate projection (the
/// `qwen35_deltanet_gguf_names` convention — the mask section key).
pub fn gate_tensor_name(layer: usize) -> String {
    format!("blk.{layer}.ffn_gate.weight")
}

/// Zero `dst[w]` wherever the mask deletes weight `w`. `mask_offset`
/// is the section-relative bit base (the caller passes the section's
/// `bit_offset` when `dst` is a raw window of the codec's bit stream;
/// in-tensor application passes 0 and `bits` must start at the
/// section's first bit). Returns the number of weights zeroed.
///
/// Bit `w` of the window = 1 → keep; 0 → `dst[w] = 0.0` exactly
/// (positive zero; any prior sign disappears, which is what
/// "elementwise-zeroed" means for the reference contract).
pub fn apply_mask_f32(dst: &mut [f32], bits: &[u8], mask_offset: usize) -> usize {
    let mut zeroed = 0usize;
    for (w, slot) in dst.iter_mut().enumerate() {
        let bit_index = mask_offset + w;
        let keep = bits
            .get(bit_index / 8)
            .is_some_and(|b| (b >> (bit_index % 8)) & 1 == 1);
        if !keep {
            *slot = 0.0;
            zeroed += 1;
        }
    }
    zeroed
}

/// Per-section application record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaskApplicationReport {
    pub tensor_name: String,
    pub weights_total: usize,
    pub weights_zeroed: usize,
}

/// Apply-at-load: walk the mask's sections, zero the matching layer
/// gate projections in `weights`. Sections naming unknown tensors are
/// skipped (the mask is silent where it does not bind — same law as
/// [`super::q2_apply::apply_to_tensor`]); a section whose length
/// disagrees with the tensor refuses loudly. Returns one record per
/// APPLIED section.
pub fn apply_mask_at_load(
    weights: &mut QwenDeltaNetWeights,
    mask: &DecodedMask,
) -> Result<Vec<MaskApplicationReport>, MaskCodecError> {
    let mut reports = Vec::new();
    for section in &mask.manifest.sections {
        let Some((layer_index, _)) = section
            .tensor_name
            .strip_prefix("blk.")
            .and_then(|rest| rest.strip_suffix(".ffn_gate.weight"))
            .and_then(|mid| mid.parse::<usize>().ok().map(|i| (i, mid)))
        else {
            continue;
        };
        let Some(layer) = weights.layers.get_mut(layer_index) else {
            return Err(MaskCodecError::SectionRange(format!(
                "section {} names layer {} beyond the model's {} layers",
                section.tensor_name,
                layer_index,
                weights.layers.len()
            )));
        };
        let data = layer.gate_proj.dense_data_mut();
        if data.len() as u64 != section.bit_len {
            return Err(MaskCodecError::SectionRange(format!(
                "section {} covers {} bits, tensor holds {} weights",
                section.tensor_name,
                section.bit_len,
                data.len()
            )));
        }
        let zeroed = apply_mask_f32(data, &mask.bits, section.bit_offset as usize);
        reports.push(MaskApplicationReport {
            tensor_name: section.tensor_name.clone(),
            weights_total: data.len(),
            weights_zeroed: zeroed,
        });
    }
    Ok(reports)
}

/// Apply-at-load over the llama-family weight struct (the `qwen2` lane
/// — Issue 1208 T4's released-mask league cell). Same law as
/// [`apply_mask_at_load`]: unknown sections skip silently, out-of-range
/// layers and bit-length disagreements refuse loudly. The qwen2
/// tensors are plain `Vec<f32>` (no `Proj` wrapper), so the section
/// walk is direct.
pub fn apply_mask_at_load_llama(
    weights: &mut LlamaTransformerWeights,
    mask: &DecodedMask,
) -> Result<Vec<MaskApplicationReport>, MaskCodecError> {
    let mut reports = Vec::new();
    for section in &mask.manifest.sections {
        let Some((layer_index, _)) = section
            .tensor_name
            .strip_prefix("blk.")
            .and_then(|rest| rest.strip_suffix(".ffn_gate.weight"))
            .and_then(|mid| mid.parse::<usize>().ok().map(|i| (i, mid)))
        else {
            continue;
        };
        let Some(layer) = weights.layers.get_mut(layer_index) else {
            return Err(MaskCodecError::SectionRange(format!(
                "section {} names layer {} beyond the model's {} layers",
                section.tensor_name,
                layer_index,
                weights.layers.len()
            )));
        };
        let data = layer.gate_proj.as_mut_slice();
        if data.len() as u64 != section.bit_len {
            return Err(MaskCodecError::SectionRange(format!(
                "section {} covers {} bits, tensor holds {} weights",
                section.tensor_name,
                section.bit_len,
                data.len()
            )));
        }
        let zeroed = apply_mask_f32(data, &mask.bits, section.bit_offset as usize);
        reports.push(MaskApplicationReport {
            tensor_name: section.tensor_name.clone(),
            weights_total: data.len(),
            weights_zeroed: zeroed,
        });
    }
    Ok(reports)
}

/// The in-forward overlay: masked shadow copies of armed layers' gate
/// projections. Mounted ONCE per mask (construction-time allocation);
/// the steady-state forward pays one `Option` check per layer and runs
/// the same kernel over the shadow copy when armed. The base weights
/// are never touched (pinned by the G1 battery).
#[derive(Debug, Clone)]
pub struct GateMaskOverlay {
    /// `layer_index -> masked gate data` (row-major, same layout as the
    /// base `Proj::dense`). Only masked layers have entries.
    shadows: Vec<(usize, Vec<f32>)>,
}

impl GateMaskOverlay {
    /// Build the overlay from a decoded mask + the pristine weights.
    /// Sections naming unknown tensors are skipped (the bind-where-bound
    /// law); length disagreements refuse loudly.
    pub fn mount(
        weights: &QwenDeltaNetWeights,
        mask: &DecodedMask,
    ) -> Result<Self, MaskCodecError> {
        let mut shadows = Vec::new();
        for section in &mask.manifest.sections {
            let Some(layer_index) = section
                .tensor_name
                .strip_prefix("blk.")
                .and_then(|rest| rest.strip_suffix(".ffn_gate.weight"))
                .and_then(|mid| mid.parse::<usize>().ok())
            else {
                continue;
            };
            let Some(layer) = weights.layers.get(layer_index) else {
                return Err(MaskCodecError::SectionRange(format!(
                    "section {} names layer {} beyond the model's {} layers",
                    section.tensor_name,
                    layer_index,
                    weights.layers.len()
                )));
            };
            let data = layer.gate_proj.dense_data();
            if data.len() as u64 != section.bit_len {
                return Err(MaskCodecError::SectionRange(format!(
                    "section {} covers {} bits, tensor holds {} weights",
                    section.tensor_name,
                    section.bit_len,
                    data.len()
                )));
            }
            let mut copy = data.to_vec();
            apply_mask_f32(&mut copy, &mask.bits, section.bit_offset as usize);
            shadows.push((layer_index, copy));
        }
        Ok(Self { shadows })
    }

    /// The masked gate data for `layer`, if the overlay arms it.
    pub fn shadow_for(&self, layer: usize) -> Option<&[f32]> {
        self.shadows
            .iter()
            .find(|(idx, _)| *idx == layer)
            .map(|(_, data)| data.as_slice())
    }

    /// Whether `layer`'s gate projection is armed (drives the forward's
    /// one-None-check dispatch).
    pub fn arms(&self, layer: usize) -> bool {
        self.shadows.iter().any(|(idx, _)| *idx == layer)
    }

    pub fn armed_layers(&self) -> impl Iterator<Item = usize> + '_ {
        self.shadows.iter().map(|(idx, _)| *idx)
    }
}

/// The scratch field type for the cfg-gated forward hook (sibling of the
/// grafts_cell capture / sink_audit capture pattern: `None` everywhere
/// except the armed run; a flag-off build compiles the field out).
pub type GateMaskOverlayField = Option<GateMaskOverlay>;
