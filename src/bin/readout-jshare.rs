//! readout-jshare — the Issue-1250 T2 J-share verbalizability front.
//!
//! Postures:
//! - `--self-test` — runs the G1 fixture gates in-process (the NNLS
//!   exactness + degeneracy arms, the control determinism, the bucket
//!   edges, the e2e over the fixture artifact). No model file needed.
//! - `--gguf <path> --artifact <rdmap1> --out <rdmap2>` — the T2 pass
//!   over a real checkpoint: reads the T1 artifact's atom supports
//!   (never re-derived), verifies the artifact's whole-file BLAKE3
//!   against the live checkpoint, solves the per-unit NNLS J-share
//!   against its own top-10 lens rows, builds the per-layer Rademacher
//!   control population, buckets every unit High/Middle/Low by its
//!   percentile, and writes the canonical RDMAP2 artifact + digest +
//!   the deterministic report. Weights only — no forward pass, no
//!   serving surface.
//!
//! Reader protection: `required-features = ["readout_screen"]` — a
//! default build never compiles this bin (the green-zero law).

use anyhow::Context as _;
use riir_infer_core::gguf_loader::GgufFile;
use riir_infer_core::readout_jshare::{
    build_j_artifact, j_artifact_digest, render_j_report, run_jshare,
};
use riir_infer_core::readout_screen::{GgufReadoutSource, hex};
use riir_infer_core::readout_screen_fixture;
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let mut self_test = false;
    let mut model: Option<String> = None;
    let mut artifact: Option<String> = None;
    let mut out: Option<String> = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--self-test" => self_test = true,
            "--gguf" => model = args.next(),
            "--artifact" => artifact = args.next(),
            "--out" => out = args.next(),
            "--help" | "-h" => {
                print_help();
                return;
            }
            other => {
                eprintln!("readout-jshare: unknown argument {other:?} (see --help)");
                std::process::exit(2);
            }
        }
    }

    if self_test {
        match readout_screen_fixture::run_all() {
            Ok(receipt) => {
                println!("readout-jshare self-test: {} gate(s) PASS", receipt.len());
                for line in receipt {
                    println!("{line}");
                }
            }
            Err(e) => {
                eprintln!("readout-jshare self-test FAILED: {e:#}");
                std::process::exit(1);
            }
        }
        return;
    }

    let Some(model_path) = model else {
        eprintln!("readout-jshare: --gguf <path> is required (or --self-test)");
        std::process::exit(2);
    };
    let Some(artifact_path) = artifact else {
        eprintln!("readout-jshare: --artifact <rdmap1> is required (the T1 atom supports)");
        std::process::exit(2);
    };
    let Some(out_path) = out else {
        eprintln!("readout-jshare: --out <rdmap2 path> is required");
        std::process::exit(2);
    };

    let t0 = Instant::now();
    let gguf = GgufFile::open(std::path::Path::new(&model_path))
        .with_context(|| format!("open {}", model_path))
        .unwrap_or_else(|e| {
            eprintln!("readout-jshare: {e:#}");
            std::process::exit(1);
        });
    let src = GgufReadoutSource::open(&gguf, std::path::Path::new(&model_path), 1e-6)
        .unwrap_or_else(|e| {
            eprintln!("readout-jshare: {e:#}");
            std::process::exit(1);
        });
    let v1 = std::fs::read(&artifact_path)
        .with_context(|| format!("read {artifact_path}"))
        .unwrap_or_else(|e| {
            eprintln!("readout-jshare: {e:#}");
            std::process::exit(1);
        });
    let js = run_jshare(&v1, &src).unwrap_or_else(|e| {
        eprintln!("readout-jshare: {e:#}");
        std::process::exit(1);
    });
    let bytes = build_j_artifact(&js);
    let digest = j_artifact_digest(&bytes);
    std::fs::write(&out_path, &bytes)
        .with_context(|| format!("write {out_path}"))
        .unwrap_or_else(|e| {
            eprintln!("readout-jshare: {e:#}");
            std::process::exit(1);
        });
    print!("{}", render_j_report(&js, &hex(&digest)));
    eprintln!(
        "readout-jshare: {} unit rows in {:?} → {} ({} bytes, digest {})",
        js.layers.iter().map(|l| l.units.len()).sum::<usize>(),
        t0.elapsed(),
        out_path,
        bytes.len(),
        hex(&digest)
    );
}

fn print_help() {
    println!(
        "readout-jshare — the Issue-1250 T2 J-share verbalizability pass\n\
         \x20 (weights-only offline census; opt-in feature `readout_screen`)\n\
         \n\
         \x20 readout-jshare --self-test\n\
         \x20 readout-jshare --gguf <checkpoint.gguf> --artifact <rdmap1> --out <rdmap2>\n"
    );
}
