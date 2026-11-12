//! The weight-audit census front (riir-infer Issue 1209, Research 025
//! — the Lazarus distill): the five-feature reference-free census over
//! the last MLP of a loaded GGUF, one deterministic report on stdout.
//!
//! REPORT-ONLY: a flag means "inspect", never "reject". Offline once
//! per checkpoint; zero serving-path cost; no forward passes.
//!
//! ```text
//! cargo run --release --features weight_audit --bin weight-audit -- \
//!     --gguf ../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf [--layer N]
//! ```

use anyhow::{Context, Result, bail};
use riir_infer_core::weight_audit::{
    GgufMlpSource, MlpTensors, ReportHeader, Thresholds, census_mlp, render_report,
};
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut gguf_path: Option<PathBuf> = None;
    let mut layer: Option<usize> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--gguf" => {
                gguf_path = Some(PathBuf::from(args.next().context("--gguf needs a path")?));
            }
            "--layer" => {
                let v = args.next().context("--layer needs a number")?;
                layer = Some(v.parse::<usize>().context("--layer parse")?);
            }
            other => bail!("unknown arg {other:?} (usage: --gguf <path> [--layer N])"),
        }
    }
    let path = gguf_path.context("--gguf <path> is required")?;
    if !path.is_file() {
        bail!("--gguf {} is not a file", path.display());
    }

    let gguf = riir_infer_core::gguf_loader::GgufFile::open(&path)
        .with_context(|| format!("open {}", path.display()))?;
    let src = GgufMlpSource::open(&gguf, layer)?;

    // The report stamp: blake3 over the whole file bytes (streamed; the
    // transport-integrity layer the census composes with — Research 025
    // §2.7. A matching hash of a backdoored upstream is still a perfect
    // match; the census reads the structure the hash cannot).
    let digest = file_blake3(&path)?;

    let t0 = std::time::Instant::now();
    let audit = census_mlp(&src, Thresholds::CANONICAL)?;
    let elapsed = t0.elapsed();
    let header = ReportHeader {
        file: path.display().to_string(),
        arch: gguf.architecture().map(str::to_string),
        file_digest_blake3: Some(digest),
    };
    print!("{}", render_report(&audit, &header));
    eprintln!(
        "[weight-audit] census {} in {:.1}s",
        src.label(),
        elapsed.as_secs_f32()
    );
    Ok(())
}

/// Streamed blake3 over the file (4 MiB chunks).
fn file_blake3(path: &PathBuf) -> Result<String> {
    use blake3::Hasher;
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut hasher = Hasher::new();
    let mut buf = vec![0u8; 4 * 1024 * 1024];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize().to_hex().to_string())
}
