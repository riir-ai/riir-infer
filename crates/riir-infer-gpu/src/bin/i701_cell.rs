//! i701_cell — the Issue-701 cell-2 CONSUMER half: the paired per-family
//! retention cell on the pre-compensated artifact (the splice producer's
//! output), driving the dense-league GPU forward (`Qwen38WgpuForward`).
//!
//! Protocol (the i1188 shape, ON ONE FROZEN READ per arm):
//!   for arm in {base, precomp}: load GGUF → ingest the calibration
//!   prefix, then teacher-force the scored window reading the FULL logits
//!   per position; report pooled NLL + argmax-flips vs the base arm's
//!   picks, per family-class attribution rides the artifact itself (each
//!   family was spliced separately by the producer — this cell scores the
//!   WHOLE spliced set against base; the per-family split comes from
//!   running the producer with `--families <one>` per cell).
//!
//! Kill bar (riir-train issue 701): tie (ΔNLL ≈ 0) or ANY per-family
//! retention regression ⇒ the lane dies.
//!
//! ```bash
//! cargo run --release -p riir-infer-gpu --features cq3 --bin i701_cell -- \
//!     --base /Users/katopz/git/riir-train/data/Qwen3.5-0.8B-Base-Q8_0.gguf \
//!     --precomp /tmp/i701_artifact.gguf --corpus <text> \
//!     [--prefix 512] [--window 512]
//! ```
//!
//! Offline measurement tool; the forwards are bit-identical to the
//! league posture (the promoted arms ride the same env defaults).

use std::path::PathBuf;

use riir_infer_core::corpus_text::load_corpus_text;
use riir_infer_core::gguf_loader::GgufFile;
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_gpu::qwen38_dense_cubecl::Qwen38WgpuForward;

fn main() {
    let mut base = None;
    let mut precomp = None;
    let mut corpus = None;
    let mut prefix = 512usize;
    let mut window = 512usize;
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--base" => base = Some(PathBuf::from(&args[i + 1])),
            "--precomp" => precomp = Some(PathBuf::from(&args[i + 1])),
            "--corpus" => corpus = Some(PathBuf::from(&args[i + 1])),
            "--prefix" => prefix = args[i + 1].parse().expect("prefix"),
            "--window" => window = args[i + 1].parse().expect("window"),
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(2);
            }
        }
        i += 2;
    }
    let base = base.unwrap_or_else(|| {
        PathBuf::from("/Users/katopz/git/riir-train/data/Qwen3.5-0.8B-Base-Q8_0.gguf")
    });
    let precomp = precomp.expect("--precomp <artifact.gguf>");
    let corpus = corpus.expect("--corpus <text>");
    let r = run(&base, &precomp, &corpus, prefix, window);
    if let Err(e) = r {
        eprintln!("[i701_cell] FAIL: {e}");
        std::process::exit(1);
    }
}

/// Teacher-force one arm: ingest `prefix` tokens, then score the next
/// `window` positions with FULL logits reads (NLL + argmax). Returns the
/// per-position (nats, argmax) rows.
fn score_arm(
    gguf: &PathBuf,
    tokens: &[u32],
    prefix: usize,
    window: usize,
) -> Result<(Vec<f64>, Vec<u32>), String> {
    let mut fwd = Qwen38WgpuForward::new(gguf, prefix + window + 64, 64)
        .map_err(|e| format!("boot {}: {e}", gguf.display()))?;
    let vocab = {
        let gg = GgufFile::open(gguf).map_err(|e| e.to_string())?;
        gg.metadata_u64("qwen35.vocab_size").unwrap_or(0) as usize
    };
    let _ = vocab;
    fwd.ingest(&tokens[..prefix], 0)?;
    let mut nlls = Vec::with_capacity(window);
    let mut picks = Vec::with_capacity(window);
    for w in 0..window {
        // Teacher-forced decode: feed the TRUE next token at pos prefix+w-1
        // reads the logits FOR that position's continuation. The decode
        // step at pos p consumes token[p] and produces the distribution
        // for token[p+1] — score logits against tokens[p+1].
        let pos = prefix + w - 1;
        let (_argmax) = fwd.forward_token(tokens[pos], pos)?;
        let logits = fwd.last_logits()?;
        let target = tokens[pos + 1] as usize;
        // f64 logsumexp over the raw logits (the harness convention).
        let mut max_l = f32::NEG_INFINITY;
        for &l in &logits {
            if l > max_l {
                max_l = l;
            }
        }
        let mut sum = 0.0f64;
        for &l in &logits {
            sum += (f64::from(l) - f64::from(max_l)).exp();
        }
        let lse = f64::from(max_l) + sum.ln();
        let inc = lse - f64::from(logits[target]);
        nlls.push(inc);
        picks.push(argmax_first(&logits));
    }
    Ok((nlls, picks))
}

fn argmax_first(v: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            best = i;
        }
    }
    best as u32
}

fn run(
    base: &PathBuf,
    precomp: &PathBuf,
    corpus: &PathBuf,
    prefix: usize,
    window: usize,
) -> Result<(), String> {
    let text = load_corpus_text(corpus).map_err(|e| e.to_string())?;
    let gg = GgufFile::open(base).map_err(|e| e.to_string())?;
    let tok = BpeTokenizer::from_gguf(&gg).map_err(|e| e.to_string())?;
    drop(gg);
    let toks: Vec<u32> = tok.encode(&text).into_iter().map(|t| t as u32).collect();
    if toks.len() < prefix + window + 1 {
        return Err(format!(
            "corpus {} tokens < prefix {prefix} + window {window} + 1",
            toks.len()
        ));
    }
    println!(
        "# i701_cell: base {} vs precomp {} | prefix {prefix} window {window}",
        base.display(),
        precomp.display()
    );

    let t0 = std::time::Instant::now();
    let (base_nll, base_pick) = score_arm(base, &toks, prefix, window)?;
    println!(
        "# base arm: pooled NLL {:.4} ({:.4}/tok) in {:.1}s",
        base_nll.iter().sum::<f64>(),
        base_nll.iter().sum::<f64>() / window as f64,
        t0.elapsed().as_secs_f32()
    );
    let t1 = std::time::Instant::now();
    let (pre_nll, pre_pick) = score_arm(precomp, &toks, prefix, window)?;
    println!(
        "# precomp arm: pooled NLL {:.4} ({:.4}/tok) in {:.1}s",
        pre_nll.iter().sum::<f64>(),
        pre_nll.iter().sum::<f64>() / window as f64,
        t1.elapsed().as_secs_f32()
    );

    let flips = base_pick
        .iter()
        .zip(pre_pick.iter())
        .filter(|(a, b)| a != b)
        .count();
    let delta = pre_nll.iter().sum::<f64>() - base_nll.iter().sum::<f64>();
    println!("# VERDICT: ΔNLL(pooled, precomp-base) = {delta:+.4} | argmax flips {flips}/{window}");
    if delta > 1e-9 {
        println!("# KILL BAR: retention REGRESSED (ΔNLL > 0)");
        std::process::exit(1);
    } else if delta.abs() < 1e-9 {
        println!("# KILL BAR: TIE (ΔNLL = 0) — pre-compensation moved nothing");
        std::process::exit(1);
    } else {
        println!("# PASS: pre-compensation improved held-out NLL (flips noted)");
    }
    Ok(())
}
