//! The trigger-vote front (riir-infer Issue 1251 T1; katgpt-rs
//! Research 693 Part 2 App. C.4 — the checkpoint-ingest backdoor
//! screen, re-derived on our lanes; the published class is cited,
//! never claimed).
//!
//! Reader protection: `required-features = ["readout_screen"]` — a
//! default build never compiles this bin (the green-zero law).
//!
//! ```text
//! trigger-vote --self-test
//! trigger-vote --base <base.gguf> --ft <finetune.gguf> \
//!              [--out <artifact.vote1>] [--top N]
//! ```

use riir_infer_core::gguf_loader::GgufFile;
use riir_infer_core::readout_screen::hex;
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_core::trigger_vote::{
    GgufVoteSource, Verdict, VoteTensors, artifact_digest, build_artifact, render_report,
    trigger_vote,
};
use riir_infer_core::trigger_vote_fixture;
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let mut self_test = false;
    let mut base: Option<String> = None;
    let mut ft: Option<String> = None;
    let mut out: Option<String> = None;
    let mut top: usize = 30;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--self-test" => self_test = true,
            "--base" => base = args.next(),
            "--ft" => ft = args.next(),
            "--out" => out = args.next(),
            "--top" => match args.next().and_then(|v| v.parse::<usize>().ok()) {
                Some(n) if n > 0 => top = n,
                _ => {
                    eprintln!("trigger-vote: --top needs a positive integer");
                    std::process::exit(2);
                }
            },
            "--help" | "-h" => {
                print_help();
                return;
            }
            other => {
                eprintln!("trigger-vote: unknown argument {other:?} (see --help)");
                std::process::exit(2);
            }
        }
    }

    if self_test {
        match trigger_vote_fixture::run_all() {
            Ok(receipt) => {
                println!("trigger-vote self-test: {} gate(s) PASS", receipt.len());
                for line in receipt {
                    println!("  {line}");
                }
            }
            Err(e) => {
                eprintln!("trigger-vote self-test FAIL: {e:#}");
                std::process::exit(1);
            }
        }
        return;
    }

    let (Some(base_path), Some(ft_path)) = (base, ft) else {
        eprintln!(
            "trigger-vote: --base <base.gguf> --ft <finetune.gguf> are required (or --self-test)"
        );
        std::process::exit(2);
    };
    if let Err(e) = run_vote(
        std::path::Path::new(&base_path),
        std::path::Path::new(&ft_path),
        out.as_deref(),
        top,
    ) {
        eprintln!("trigger-vote FAIL: {e:#}");
        std::process::exit(1);
    }
}

fn print_help() {
    println!(
        "trigger-vote — the Issue 1251 trigger-vote screen (checkpoint-ingest backdoor probe)\n\
         \n\
         USAGE:\n\
         \x20 trigger-vote --self-test\n\
         \x20 trigger-vote --base <base.gguf> --ft <finetune.gguf> [--out <artifact.vote1>] [--top N]\n\
         \n\
         Weights-only vote over the (base, finetune) pair: every MLP neuron votes\n\
         for its top-15 changed tokens by |Δgate·e_t| + |Δup·e_t| (the C.4 rule;\n\
         e_t = the BASE token-embd row); the layer band admits contiguous layers\n\
         whose top-100 lists share ≥10 tokens, and the pooled band ranking is\n\
         the trigger candidate list. Writes the canonical VOTE1 artifact + digest."
    );
}

fn run_vote(
    base_path: &std::path::Path,
    ft_path: &std::path::Path,
    out: Option<&str>,
    top: usize,
) -> anyhow::Result<()> {
    let t0 = Instant::now();
    let base_gguf =
        GgufFile::open(base_path).with_context(|| format!("opening {}", base_path.display()))?;
    let ft_gguf =
        GgufFile::open(ft_path).with_context(|| format!("opening {}", ft_path.display()))?;
    let base_src = GgufVoteSource::open(&base_gguf, base_path)?;
    let ft_src = GgufVoteSource::open(&ft_gguf, ft_path)?;
    println!(
        "trigger-vote: base={} ft={} ({} layers, n_ff={}, n_embd={}, vocab={})",
        base_path.display(),
        ft_path.display(),
        base_src.n_layers(),
        base_src.n_ff(),
        base_src.n_embd(),
        base_src.vocab(),
    );
    let tok = BpeTokenizer::from_gguf(&base_gguf).ok();
    let v = trigger_vote(&base_src, &ft_src)?;
    println!("vote pass: {} ms", t0.elapsed().as_millis());
    print!("{}", render_report(&v, tok.as_ref(), top));
    if let Some(out_path) = out {
        let bytes = build_artifact(&v);
        std::fs::write(out_path, &bytes).with_context(|| format!("writing {out_path}"))?;
        let digest = artifact_digest(&bytes);
        println!(
            "artifact: {out_path} ({} bytes, blake3 {})",
            bytes.len(),
            hex(&digest)
        );
    }
    match v.verdict {
        Verdict::Voted => println!("verdict: VOTED"),
        Verdict::ZeroDelta => println!("verdict: ZERO-DELTA (base and finetune are bit-identical)"),
        Verdict::ZeroVotes => {
            println!("verdict: ZERO-VOTES (rows changed but every score is exactly 0)")
        }
    }
    Ok(())
}

use anyhow::Context;
