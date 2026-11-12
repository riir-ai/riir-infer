//! Issue 1208 T6 — the mask audit arm (`--mask-geometry`): per-gate-
//! tensor geometry of a deletion mask vs the pristine weights (stable
//! rank, spectrum drift, principal-subspace overlap — the paper's FFT
//! envelope axis). REPORT-ONLY, exit 0.
//!
//! The instrument is the seeded Halko truncated SVD (deterministic;
//! exact on the tiny goldens, truncated on real matrices — the seed
//! and k are printed so any box rebuilds the row).
//!
//! Run: cargo run --release --features deletion_mask,deltanet_inference \
//!   --bin i1208_mask_audit -- <model.gguf> <mask.rimsk> [seed] [k]
use riir_infer_core::adapter::geometry::audit_mask;
use riir_infer_core::adapter::mask_codec::decode;
use riir_infer_core::gguf_loader::load_qwen_deltanet_weights_gguf;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (model, mask_path) = match args.as_slice() {
        [m, k] => (m.clone(), k.clone()),
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
