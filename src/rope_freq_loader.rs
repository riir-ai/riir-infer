// Issue 1254 P0a — wire the `rope_freqs.weight` loader arm.
//
// The struct field (`rope_freq_weights`) + the ONE-table constructor
// (`build_rope_freq_table`) + the weighted table (`RopeFreqTable::new_weighted`)
// already landed upstream; the loader's tensor READ did not — both llama
// loaders hard-wired `rope_freq_weights: None`, so a scaled checkpoint
// silently ran plain frequencies (the loud comment at each site named the
// missing half). This file IS that half: try-read `rope_freqs.weight`,
// refuse loud on any non-F32 shape, plain table otherwise.
//
// Run report (T0 runtime side): see `.benchmarks/1277_i1254_claim_and_t0_runtime.md`.

use anyhow::{Context, Result, bail};

/// Read the optional per-dim RoPE frequency factors (`rope_freqs.weight`,
/// llama.cpp semantics) from an open GGUF.
///
/// Returns:
/// - `Ok(None)` — the tensor is absent: plain-table checkpoint.
/// - `Ok(Some(w))` — `head_dim/2` f32 factors.
/// - `Err(_)` — the tensor EXISTS but is not F32 with exactly
///   `head_dim/2` elements: refuse loud (never a silent fallback —
///   a truncated/garbled factor vector would bake wrong frequencies
///   into the table for every forward).
pub fn read_rope_freq_weights(
    gguf: &crate::gguf_loader::GgufFile,
    head_dim: usize,
) -> Result<Option<Vec<f32>>> {
    let Some(info) = gguf.tensor_info("rope_freqs.weight") else {
        return Ok(None);
    };
    let expected = head_dim / 2;
    if info.ggml_type != crate::gguf_loader::GgmlType::F32 {
        bail!(
            "rope_freqs.weight: expected F32 factors, got {:?} — refusing loud",
            info.ggml_type
        );
    }
    if info.n_elements() != expected {
        bail!(
            "rope_freqs.weight: expected head_dim/2 = {expected} elements, got {} — refusing loud",
            info.n_elements()
        );
    }
    let slice = gguf
        .tensor_slice("rope_freqs.weight")
        .context("rope_freqs.weight: data out of bounds")?;
    // F32 little-endian, zero-copy cast when aligned (the loader's own F32 arm).
    #[cfg(target_endian = "little")]
    if let Ok(w) = bytemuck::try_cast_slice::<_, f32>(slice) {
        return Ok(Some(w.to_vec()));
    }
    let mut w = Vec::with_capacity(expected);
    for chunk in slice.as_chunks::<4>().0 {
        w.push(f32::from_le_bytes(*chunk));
    }
    Ok(Some(w))
}

/// Wire the read into a freshly-built weights struct (the one-line arm the
/// two loader sites share; the loaders' struct literals keep
/// `rope_freq_weights: None` and this overrides it after the literal).
pub fn finish_llama_weights(
    gguf: &crate::gguf_loader::GgufFile,
    config: &crate::types::Config,
    mut weights: crate::llama_layer::LlamaTransformerWeights,
) -> Result<crate::llama_layer::LlamaTransformerWeights> {
    weights.rope_freq_weights = read_rope_freq_weights(gguf, config.head_dim)?;
    Ok(weights)
}
