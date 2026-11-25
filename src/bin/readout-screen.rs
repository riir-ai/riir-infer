//! readout-screen — the Issue-1250 static readout pre-screen front.
//!
//! Postures:
//! - `--self-test` — runs the G1 fixture gates in-process (the lens
//!   math, the bit-exact sign trick, the tie rules, the band formula,
//!   artifact stability, the refusal battery). The canary vehicle — no
//!   model file needed.
//! - `--gguf <path> --out <artifact>` — the T1 readout map over a real
//!   checkpoint: every layer × every down_proj column × both signs,
//!   the top-K lens tokens per unit, the canonical RDMAP1 artifact +
//!   BLAKE3 digest, and the deterministic text report (per-band
//!   rollups + the global top-units table). Weights only — no forward
//!   pass, no probes, no serving surface.
//!
//! Reader protection: `required-features = ["readout_screen"]` — a
//! default build never compiles this bin (the green-zero law).

use anyhow::Context as _;
use riir_infer_core::gguf_loader::GgufFile;
use riir_infer_core::readout_screen::{
    GgufReadoutSource, ROW_BYTES, ReadoutTensors, TOP_K, artifact_digest, build_artifact, hex,
    render_report, screen,
};
use riir_infer_core::readout_screen_fixture;
use riir_infer_core::tokenizer::BpeTokenizer;
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let mut self_test = false;
    let mut model: Option<String> = None;
    let mut out: Option<String> = None;
    let mut no_strings = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--self-test" => self_test = true,
            "--gguf" => model = args.next(),
            "--out" => out = args.next(),
            "--no-strings" => no_strings = true,
            "--help" | "-h" => {
                print_help();
                return;
            }
            other => {
                eprintln!("readout-screen: unknown argument {other:?} (see --help)");
                std::process::exit(2);
            }
        }
    }

    if self_test {
        run_self_test();
        return;
    }

    let Some(model_path) = model else {
        eprintln!("readout-screen: --gguf <path> is required (or --self-test)");
        std::process::exit(2);
    };
    let Some(out_path) = out else {
        eprintln!("readout-screen: --out <artifact path> is required");
        std::process::exit(2);
    };
    if let Err(e) = run_screen(
        std::path::Path::new(&model_path),
        std::path::Path::new(&out_path),
        no_strings,
    ) {
        eprintln!("readout-screen FAIL: {e:#}");
        std::process::exit(1);
    }
}

fn print_help() {
    println!(
        "readout-screen — the Issue-1250 static readout pre-screen (weights-only logit lens)\n\
         \n\
         USAGE:\n\
         \x20 readout-screen --self-test\n\
         \x20 readout-screen --gguf <checkpoint.gguf> --out <artifact.rdmap> [--no-strings]\n\
         \n\
         The sweep reads every layer's ffn_down columns (both signs) through the\n\
         final-norm logit lens and writes the canonical RDMAP1 artifact + digest.\n\
         TOP_K is pinned at {TOP_K}. The report prints per-band rollups\n\
         (early/middle/late) of top1 sharpness and top1−top2 margin."
    );
}

/// The T1 cell over a real checkpoint.
fn run_screen(
    model: &std::path::Path,
    out: &std::path::Path,
    no_strings: bool,
) -> anyhow::Result<()> {
    let t0 = Instant::now();
    let gguf = GgufFile::open(model).with_context(|| format!("opening {}", model.display()))?;
    let rms_eps = gguf
        .metadata_f64("qwen35.attention.layer_norm_rms_epsilon")
        .or_else(|| gguf.metadata_f64("attention.layer_norm_rms_epsilon"))
        .unwrap_or(1e-5);
    let src = GgufReadoutSource::open(&gguf, model, rms_eps)?;
    let tok = BpeTokenizer::from_gguf(&gguf).ok();
    println!(
        "readout-screen: {} layers={} n_ff={} n_embd={} vocab={} unemb={} top_k={TOP_K}",
        model.display(),
        src.n_layers(),
        src.n_ff(),
        src.n_embd(),
        src.vocab(),
        src.label(),
    );
    let s = screen(&src)?;
    let artifact = build_artifact(&s);
    let digest = artifact_digest(&artifact);
    let digest_hex = hex(&digest);
    std::fs::write(out, &artifact).with_context(|| format!("writing {}", out.display()))?;
    let strings: Option<&dyn Fn(u32) -> String> = match (no_strings, tok.as_ref()) {
        (true, _) | (false, None) => None,
        // The closure borrows the tokenizer for the report's top-units
        // table only — the artifact carries ids.
        (false, Some(t)) => Some(&move |id: u32| decode_one(t, id)),
    };
    let report = render_report(&s, &digest_hex, strings);
    print!("{report}");
    eprintln!(
        "readout-screen: artifact={} digest={} rows={} elapsed={:.1?}",
        out.display(),
        digest_hex,
        artifact.len() / ROW_BYTES,
        t0.elapsed()
    );
    Ok(())
}

/// Decode one token id to a printable piece (whitespace visualized;
/// control chars escaped).
fn decode_one(tok: &BpeTokenizer, id: u32) -> String {
    tok.decode(&[id as usize])
        .replace('\n', "\\n")
        .replace('\t', "\\t")
        .replace('\r', "\\r")
}

/// The G1 fixture gates (the same source the cargo tests assert).
fn run_self_test() {
    match readout_screen_fixture::run_all() {
        Ok(receipt) => {
            println!("readout-screen self-test PASS");
            for line in receipt {
                println!("{line}");
            }
        }
        Err(e) => {
            eprintln!("readout-screen self-test FAIL: {e:#}");
            std::process::exit(1);
        }
    }
}
