//! Issue 1208 T4 — the upstream mask ingest: a Deletion Tuning
//! `packbits_zstd` mask directory (`model.safetensors` + `config.json`)
//! → a RIMSK artifact this repo's codec can decode, with the census
//! (total_params / zero_params) verified EXACTLY against their
//! config.json on the way through.
//!
//! The optional `--base-gguf` arm loads a qwen2 GGUF and computes the
//! BLAKE3 over its loaded gate projections — the digest the artifact
//! binds and every downstream `decode(_, Some(base), _)` enforces
//! (wrong-base law). Without it the artifact is emitted with a zero
//! digest and a warning: it decodes, but nothing binds.
//!
//! Run:
//! cargo run --release --features deletion_mask \
//!   --bin i1208_hf_ingest -- <mask_dir> <out.rimsk> [--base-gguf <qwen2.gguf>]
use anyhow::{Context, Result, bail};
use blake3::Hasher;
use riir_infer_core::adapter::hf_ingest::{HfMaskCensus, ingest_hf_mask};
use riir_infer_core::gguf_loader::load_qwen2_weights_gguf;
use std::path::Path;

/// BLAKE3 over the loaded gate projections, per layer in order — the
/// checkpoint-digest convention shared with the Wanda sweep header
/// (`i1208_wanda_score`): the SAME weights always digest the same.
fn gate_tensor_digest(gguf: &Path) -> Result<([u8; 32], usize, usize), anyhow::Error> {
    let (_config, weights) = load_qwen2_weights_gguf(gguf)
        .with_context(|| format!("loading qwen2 weights from {}", gguf.display()))?;
    let mut hasher = Hasher::new();
    let mut count = 0usize;
    for layer in &weights.layers {
        hasher.update(&(layer.gate_proj.len() as u64).to_le_bytes());
        // f32 bytes in tensor (row-major) order — bit-exact bind
        for v in &layer.gate_proj {
            hasher.update(&v.to_le_bytes());
        }
        count += layer.gate_proj.len();
    }
    let digest = *hasher.finalize().as_bytes();
    Ok((digest, weights.layers.len(), count))
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mask_dir = PathBufArg::parse(&mut args, "mask_dir")?;
    let out_path = PathBufArg::parse(&mut args, "out.rimsk")?;
    let mut base_gguf: Option<std::path::PathBuf> = None;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--base-gguf" => {
                base_gguf = Some(
                    args.next()
                        .map(std::path::PathBuf::from)
                        .expect("--base-gguf needs a path"),
                );
            }
            other => bail!("unknown argument {other:?}"),
        }
    }

    let safetensors_path = mask_dir.join("model.safetensors");
    let config_path = mask_dir.join("config.json");
    let st_bytes = std::fs::read(&safetensors_path)
        .with_context(|| format!("reading {}", safetensors_path.display()))?;
    let census = HfMaskCensus::parse(
        &std::fs::read(&config_path)
            .with_context(|| format!("reading {}", config_path.display()))?,
    )?;
    println!(
        "# census: format={} total_params={} zero_params={} (sparsity {:.4}%)",
        census.format,
        census.total_params,
        census.zero_params,
        100.0 * census.zero_params as f64 / census.total_params.max(1) as f64
    );

    let (base_digest, base_note): ([u8; 32], String) = match &base_gguf {
        Some(gguf) => {
            let (digest, layers, weights) = gate_tensor_digest(gguf)?;
            println!(
                "# base bind: blake3(gate tensors of {layers} layers, {weights} weights) = {}",
                hex(&digest)
            );
            (digest, format!("bound to {}", gguf.display()))
        }
        None => (
            [0u8; 32],
            "UNBOUND (zero digest) — decode(_, Some(base), _) will refuse; \
             rerun with --base-gguf to bind"
                .to_string(),
        ),
    };
    if base_gguf.is_none() {
        println!("# WARNING: {base_note}");
    }

    let (manifest, bits) = ingest_hf_mask(&st_bytes, &census, base_digest)?;
    let artifact = manifest.pack(&bits, None)?;
    std::fs::write(&out_path, &artifact)
        .with_context(|| format!("writing {}", out_path.display()))?;

    let deleted: u64 = manifest
        .sections
        .iter()
        .map(|s| s.bit_len)
        .sum::<u64>()
        .saturating_sub(bits.iter().map(|b| b.count_ones() as u64).sum::<u64>());
    println!(
        "# sections={} bit_len_sum={} deleted={} artifact_bytes={} -> {} ({})",
        manifest.sections.len(),
        manifest.sections.iter().map(|s| s.bit_len).sum::<u64>(),
        deleted,
        artifact.len(),
        out_path.display(),
        base_note
    );
    // The BLAKE3 of the artifact itself — the record pins this.
    println!(
        "# artifact blake3 = {}",
        hex(blake3::hash(&artifact).as_bytes())
    );
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Tiny arg helper keeping main() flat.
struct PathBufArg;
impl PathBufArg {
    fn parse(args: &mut impl Iterator<Item = String>, what: &str) -> Result<std::path::PathBuf> {
        args.next()
            .map(std::path::PathBuf::from)
            .ok_or_else(|| anyhow::anyhow!("usage: i1208_hf_ingest <mask_dir> <out.rimsk> [--base-gguf <gguf>] (missing {what})"))
    }
}
