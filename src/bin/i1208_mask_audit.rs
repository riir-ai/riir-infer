//! Issue 1208 T6 — the mask audit arms. REPORT-ONLY, exit 0.
//!
//! - `i1208_mask_audit <model.gguf> <mask.rimsk> [seed] [k]` — the
//!   `--mask-geometry` posture (per-gate-tensor stable rank / spectrum
//!   drift / principal-subspace overlap vs the paper's FFT envelope).
//! - `i1208_mask_audit --fk-lab <model.gguf> <mask.rimsk> <corpus>
//!   [probe_tokens] [layers]` — the forward-KL posture: decode the
//!   base and the overlay-masked model on the same probe walk,
//!   KL(base ∥ masked) per position, the mean + percentile tail
//!   printed with the digest-pinned KL table (the T7 τ-sweep's input;
//!   the corpus rides the chat_probe convention — token-stream digest
//!   in the header, the T5 pattern).
//!
//! The instrument for geometry is the seeded Halko truncated SVD
//! (deterministic; exact on the tiny goldens, truncated on real
//! matrices — the seed and k are printed so any box rebuilds the row).

use riir_infer_core::adapter::geometry::audit_mask;
use riir_infer_core::adapter::mask_codec::decode;
use riir_infer_core::gguf_loader::load_qwen_deltanet_weights_gguf;
use std::path::Path;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(|s| s.as_str()) == Some("--fk-lab") {
        return fk_lab(&args[1..]);
    }
    mask_geometry(&args)
}

fn mask_geometry(args: &[String]) -> anyhow::Result<()> {
    let (model, mask_path) = match args {
        [m, k, ..] => (m.clone(), k.clone()),
        _ => anyhow::bail!("usage: i1208_mask_audit <model.gguf> <mask.rimsk> [seed] [k]"),
    };
    let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0xC0FFEE);
    let k: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);

    let mask_blob = std::fs::read(&mask_path)?;
    let decoded = decode(&mask_blob, None, None)?;
    let (config, weights) = load_qwen_deltanet_weights_gguf(Path::new(&model))?;
    let (rows_, cols) = (config.mlp_hidden, config.n_embd);

    println!(
        "# mask-geometry | mask {} ({} sections) | model {} layers | gate [mlp_hidden {rows_}, n_embd {cols}] | seed {seed:#x} k={k}",
        mask_path,
        decoded.manifest.sections.len(),
        weights.layers.len()
    );
    println!(
        "# FFT envelope axis: trained masks drift ≤1.4x, overlap 0.43-0.46; identity -> overlap 1.0"
    );
    let layers = &weights.layers;
    let rows = audit_mask(&decoded, seed, |layer| {
        layers.get(layer).map(|l| {
            let data = l.gate_proj.dense_data();
            debug_assert_eq!(data.len(), rows_ * cols);
            (data.to_vec(), rows_, cols)
        })
    });
    if rows.is_empty() {
        println!("# no gate sections bound (nothing to audit)");
        return Ok(());
    }
    for row in &rows {
        println!(
            "# {} | stable rank {:.2} -> {:.2} | drift max {:.3} mean {:.3} | overlap {:.3} (k={})",
            row.tensor_name,
            row.stable_rank_base,
            row.stable_rank_masked,
            row.drift_max,
            row.drift_mean,
            row.overlap,
            row.k
        );
    }
    Ok(())
}

/// The `--fk-lab` posture: two walks (base / overlay-masked) over the
/// same probe tokens, the per-position forward-KL table.
fn fk_lab(args: &[String]) -> anyhow::Result<()> {
    use riir_infer_core::adapter::f32_apply::GateMaskOverlay;
    use riir_infer_core::adapter::fkl::forward_kl_walk;
    use riir_infer_core::corpus_text::{load_corpus_pages, load_corpus_text};
    use riir_infer_core::deltanet::forward::{
        HybridCache, HybridForwardScratch, forward_qwen_deltanet,
    };
    use riir_infer_core::gguf_loader::{GgufFile, load_qwen_deltanet_weights_gguf};
    use riir_infer_core::rope::RopeFreqTable;
    use riir_infer_core::tokenizer::BpeTokenizer;

    let (model, mask_path, corpus_path) = match args {
        [m, k, c, ..] => (m.clone(), k.clone(), c.clone()),
        _ => anyhow::bail!(
            "usage: i1208_mask_audit --fk-lab <model.gguf> <mask.rimsk> <corpus> [probe_tokens=64]"
        ),
    };
    let probe_tokens: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);

    let mask_blob = std::fs::read(&mask_path)?;
    let decoded = decode(&mask_blob, None, None)?;

    let probe = GgufFile::open(Path::new(&model))?;
    let tok = BpeTokenizer::from_gguf(&probe)?;
    drop(probe);
    let (config, weights) = load_qwen_deltanet_weights_gguf(Path::new(&model))?;
    let rope_freq = RopeFreqTable::new(config.rope_theta, config.rope_dimension_count);

    // the probe: natural text -> token ids, the token-stream digest in
    // the header (the T5 pattern)
    let cpath = Path::new(&corpus_path);
    let text = if cpath.is_dir() {
        load_corpus_pages(cpath)?
            .into_iter()
            .map(|(_, body)| body)
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        load_corpus_text(cpath)?
    };
    let token_stream: Vec<usize> = tok.encode(&text).into_iter().take(probe_tokens).collect();
    let corpus_digest: [u8; 32] =
        *blake3::hash(bytemuck::cast_slice::<usize, u8>(&token_stream)).as_bytes();
    println!(
        "# fk-lab | mask {} ({} sections) | probe {} tokens, corpus digest {}",
        mask_path,
        decoded.manifest.sections.len(),
        token_stream.len(),
        hex(&corpus_digest)
    );

    // the two walks: the base (pristine) and the overlay-masked mount;
    // the overlay is built ONCE (masked shadow copies — no second full
    // weight copy beyond the mount, the scope law)
    let overlay = GateMaskOverlay::mount(&weights, &decoded)?;
    let walk = |overlay: Option<GateMaskOverlay>| -> anyhow::Result<Vec<Vec<f32>>> {
        let mut cache = HybridCache::with_layer_types(&config, &weights.layer_types);
        let mut scratch = HybridForwardScratch::new(&config);
        #[cfg(feature = "deletion_mask")]
        {
            scratch.gate_mask_overlay = overlay;
        }
        #[cfg(not(feature = "deletion_mask"))]
        {
            let _ = overlay;
        }
        let mut x = vec![0.0f32; config.vocab_size.max(config.n_embd) * 2];
        let mut per_position = Vec::with_capacity(token_stream.len());
        for (step, &tok_id) in token_stream.iter().enumerate() {
            let out = forward_qwen_deltanet(
                &mut x,
                &weights,
                &mut cache,
                tok_id,
                step,
                &config,
                &mut scratch,
                &rope_freq,
            );
            per_position.push(out[..config.vocab_size].to_vec());
        }
        Ok(per_position)
    };
    let t0 = std::time::Instant::now();
    let base_walk = walk(None)?;
    let masked_walk = walk(Some(overlay))?;
    let report = forward_kl_walk(&base_walk, &masked_walk);
    println!(
        "# walks: {:.1} s (2 × {} positions × vocab {})",
        t0.elapsed().as_secs_f64(),
        token_stream.len(),
        config.vocab_size
    );
    println!(
        "# KL(base ∥ masked) nats/position: mean {:.6} | p50 {:.6} | p95 {:.6} | p99 {:.6} (tail support {})",
        report.mean, report.p50, report.p95, report.p99, report.p99_tail_support
    );
    println!(
        "# KL table digest {} | per-position head: {}",
        hex(&report.table_digest),
        report
            .per_position
            .iter()
            .take(8)
            .map(|v| format!("{v:.4}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("# REPORT-ONLY exit 0 — the T7 τ-sweep consumes this table");
    Ok(())
}
