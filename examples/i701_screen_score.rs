//! i701_screen_score — the Issue-701 cell-2 CONSUMER half: the paired
//! base-vs-precomp-screen cell over the DENSE qwen35 lane.
//!
//! The i1188 harness prices families on the ternary posture (its loader
//! requires ternary globals — `load_ternary_proj` refuses a dense
//! `token_embd`), so the dense-lane consumer lands HERE as an example
//! beside the producer (the same `cells as riir-infer examples` shape the
//! T0 note pinned; the controlling record is riir-train issue 701).
//!
//! Walk: load the baseline GGUF and the spliced precomp artifact with the
//! SAME dense loader, tokenize the SAME held-out corpus, prefill the SAME
//! prefix, then score the SAME `--score-window` next-token predictions —
//! per-position NLL + argmax on both files, paired ascending. The deciding
//! read is the paired pooled Δ-NLL (artifact − base) plus the argmax-flip
//! census. Kill bar (riir-train issue 701, unchanged): tie (|Δ| within
//! noise) or any regression ⇒ the screen kills the lane here; only a
//! negative pooled Δ on the artifact survives to the per-family battery.
//!
//! Determinism: same files + same corpus ⇒ byte-identical output (fixed
//! ascending fold order per the Issue-940 law; no RNG anywhere; both runs
//! share the identical f32 forward — the paired design, no requant step
//! in-run). Offline tool: zero serving-path surface.
//!
//! ```bash
//! cargo run --release --features deltanet_inference \
//!     --example i701_screen_score -- \
//!     --gguf <base.gguf> --artifact <spliced.gguf> --corpus <held_out.txt> \
//!     [--max-length 2048] [--score-window 512]
//! ```

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use riir_infer_core::deltanet::forward::{
    HybridCache, HybridForwardScratch, PrefillContext, effective_rotary_dim, forward_qwen_deltanet,
    prefill_qwen_deltanet_into,
};
use riir_infer_core::deltanet::weights::QwenDeltaNetWeights;
use riir_infer_core::gguf_loader::{GgufFile, load_qwen_deltanet_weights_gguf};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_core::types::Config;

fn blake3_file(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    Ok(blake3::hash(&bytes).to_string())
}

/// One scored position: the file's NLL for `next_token` under the logits
/// predicted after feeding `[0..=t]`, plus the argmax pick.
#[derive(Clone, Copy)]
struct ScoredRow {
    nll: f64,
    argmax: usize,
    next_token: usize,
}

/// Walk ONE file: prefill `[0..P-W)`, then step the scored window, reading
/// each step's full logits before feeding the token. Identical code path
/// for both files — the pairing is the design.
fn score_file(
    path: &Path,
    config: &Config,
    seq: &[u32],
    prefix: usize,
    window: usize,
) -> Result<Vec<ScoredRow>> {
    let (mut _cfg, weights) = load_qwen_deltanet_weights_gguf(path)
        .with_context(|| format!("load {}", path.display()))?;
    score_weights(&weights, config, seq, prefix, window)
}

fn score_weights(
    weights: &QwenDeltaNetWeights,
    config: &Config,
    seq: &[u32],
    prefix: usize,
    window: usize,
) -> Result<Vec<ScoredRow>> {
    let feed_len = prefix - window;
    let rope = RopeFreqTable::new(config.rope_theta, effective_rotary_dim(config));
    let mut cache = HybridCache::with_layer_types(config, &weights.layer_types);
    let mut scratch = HybridForwardScratch::new(config);
    let mut prefill_ctx = PrefillContext::new(config, prefix.max(1));

    let tokens: Vec<usize> = seq.iter().map(|&t| t as usize).collect();
    let v = config.vocab_size;
    let mut x = vec![0.0f32; v.max(config.n_embd) + v];
    let mut prefill_logits = vec![0.0f32; v];

    // Prefill [0..feed_len): logits_out = the prediction of token feed_len.
    prefill_qwen_deltanet_into(
        weights,
        config,
        &mut cache,
        &tokens[..feed_len],
        &mut scratch,
        &rope,
        &mut prefill_ctx,
        &mut prefill_logits,
        None,
    );

    let mut rows = Vec::with_capacity(window);
    let mut cur_logits = prefill_logits;
    {
        // Session-state probe (the NaN hunt): the prefill's own head output.
        let fin = cur_logits.iter().filter(|l| l.is_finite()).count();
        let mx = cur_logits.iter().fold(f32::NEG_INFINITY, |a, b| b.max(a));
        println!(
            "# [probe] after prefill: finite {fin}/{} logits, max {mx}",
            cur_logits.len()
        );
    }
    let mut next_pos = feed_len; // the next fed token's position
    for (k, &nt) in tokens.iter().enumerate().take(prefix).skip(feed_len) {
        // NLL + argmax of cur_logits (the prediction of token k).
        let (nll, argmax) = nll_argmax(&cur_logits, nt);
        rows.push(ScoredRow {
            nll,
            argmax,
            next_token: nt,
        });
        if k + 1 < prefix {
            // Feed token k at position next_pos; the returned slice is the
            // logits predicting token k+1. x is sized for the head write.
            let out = forward_qwen_deltanet(
                &mut x,
                weights,
                &mut cache,
                nt,
                next_pos,
                config,
                &mut scratch,
                &rope,
            );
            cur_logits.copy_from_slice(&out[..v]);
            next_pos += 1;
            if k == feed_len {
                // The FIRST step's logits: the handoff probe.
                let fin = cur_logits.iter().filter(|l| l.is_finite()).count();
                let mx = cur_logits.iter().fold(f32::NEG_INFINITY, |a, b| b.max(a));
                println!(
                    "# [probe] after first step (pos {next_pos}): finite {fin}/{} logits, max {mx}",
                    cur_logits.len()
                );
            }
        }
    }
    Ok(rows)
}

/// f64 NLL for `next_token` under `logits` + the argmax pick. Max-subtracted
/// log-softmax; the logsumexp folds ascending (the Issue-940 order law).
fn nll_argmax(logits: &[f32], next_token: usize) -> (f64, usize) {
    let mut max = f32::NEG_INFINITY;
    for &l in logits {
        if l > max {
            max = l;
        }
    }
    let mut sum = 0.0f64;
    for &l in logits {
        sum += ((l - max) as f64).exp();
    }
    let lse = max as f64 + sum.ln();
    let picked = logits[next_token] as f64;
    let nll = lse - picked;
    let mut argmax = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (i, &l) in logits.iter().enumerate() {
        if l > best {
            best = l;
            argmax = i;
        }
    }
    (nll, argmax)
}

fn parse_args(
    argv: &[String],
) -> Result<(PathBuf, PathBuf, PathBuf, usize, usize, Option<PathBuf>)> {
    let mut gguf: Option<PathBuf> = None;
    let mut artifact: Option<PathBuf> = None;
    let mut corpus: Option<PathBuf> = None;
    let mut max_length = 2048usize;
    let mut window = 512usize;
    let mut dump: Option<PathBuf> = None;
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--gguf" => gguf = Some(PathBuf::from(&argv[i + 1])),
            "--artifact" => artifact = Some(PathBuf::from(&argv[i + 1])),
            "--corpus" => corpus = Some(PathBuf::from(&argv[i + 1])),
            "--max-length" => max_length = argv[i + 1].parse()?,
            "--score-window" => window = argv[i + 1].parse()?,
            "--dump" => dump = Some(PathBuf::from(&argv[i + 1])),
            other => bail!("unknown arg {other}"),
        }
        i += 2;
    }
    let gguf = gguf.context("--gguf <base.gguf> is required")?;
    let artifact = artifact.context("--artifact <spliced.gguf> is required")?;
    let corpus = corpus.context("--corpus <held-out text> is required")?;
    ensure!(
        window >= 8 && window <= max_length,
        "8 <= score-window <= max-length (got window {window}, prefix {max_length})"
    );
    Ok((gguf, artifact, corpus, max_length, window, dump))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let r = run(&args[1..]);
    if let Err(e) = r {
        eprintln!("[i701_screen_score] FAIL: {e:#}");
        std::process::exit(1);
    }
}

fn run(argv: &[String]) -> Result<()> {
    let (base_path, artifact_path, corpus_path, prefix, window, dump) = parse_args(argv)?;

    // Tokenize ONCE — both files walk the identical token sequence.
    let gg = GgufFile::open(&base_path).context("open base for tokenizer")?;
    let tok = BpeTokenizer::from_gguf(&gg).context("tokenizer")?;
    drop(gg);
    let text = std::fs::read_to_string(&corpus_path)
        .with_context(|| format!("corpus {}", corpus_path.display()))?;
    let seq: Vec<u32> = tok.encode(&text).into_iter().map(|t| t as u32).collect();
    anyhow::ensure!(
        seq.len() >= prefix,
        "corpus tokenizes to {} tokens < prefix {prefix}",
        seq.len()
    );
    println!(
        "# i701_screen_score | base {} | artifact {} | corpus {} -> {} tokens | prefix {prefix} window {window}",
        base_path.display(),
        artifact_path.display(),
        corpus_path.display(),
        seq.len(),
    );
    println!(
        "# base blake3 {} | artifact blake3 {}",
        blake3_file(&base_path)?,
        blake3_file(&artifact_path)?,
    );

    // The config comes from the base load; the artifact is a byte-splice of
    // the same file (same header) — one config drives both walks.
    let (config, base_weights) = load_qwen_deltanet_weights_gguf(&base_path)
        .with_context(|| format!("load {}", base_path.display()))?;
    let t0 = std::time::Instant::now();
    let base_rows = score_weights(&base_weights, &config, &seq, prefix, window)?;
    println!(
        "# base walk: {} scored positions in {:.1}s",
        base_rows.len(),
        t0.elapsed().as_secs_f32()
    );
    drop(base_weights);

    let t1 = std::time::Instant::now();
    let art_rows = score_file(&artifact_path, &config, &seq, prefix, window)?;
    println!(
        "# artifact walk: {} scored positions in {:.1}s",
        art_rows.len(),
        t1.elapsed().as_secs_f32()
    );

    // ── the paired report (ascending fold, Issue-940 law) ──
    anyhow::ensure!(
        base_rows.len() == art_rows.len() && !base_rows.is_empty(),
        "paired walks scored different position counts"
    );
    let n = base_rows.len() as f64;
    let mut sum_base = 0.0f64;
    let mut sum_art = 0.0f64;
    let mut abs_sum = 0.0f64;
    let mut flips = 0usize;
    let mut flip_base_lost = 0usize; // base right, artifact wrong
    let mut flip_art_lost = 0usize; // artifact right, base wrong
    let mut nonfinite_base = 0usize;
    let mut nonfinite_art = 0usize;
    let mut first_bad: Option<(usize, bool)> = None; // (index, is_artifact_side)
    let mut worst = Vec::<(f64, usize)>::new();
    for (i, (b, a)) in base_rows.iter().zip(art_rows.iter()).enumerate() {
        let d = a.nll - b.nll;
        if !b.nll.is_finite() {
            nonfinite_base += 1;
            if first_bad.is_none() {
                first_bad = Some((i, false));
            }
        }
        if !a.nll.is_finite() {
            nonfinite_art += 1;
            if first_bad.is_none() {
                first_bad = Some((i, true));
            }
        }
        sum_art += a.nll;
        sum_base += b.nll;
        abs_sum += d.abs();
        if a.argmax != b.argmax {
            flips += 1;
            let b_right = b.argmax == b.next_token;
            let a_right = a.argmax == a.next_token;
            if b_right && !a_right {
                flip_base_lost += 1;
            }
            if a_right && !b_right {
                flip_art_lost += 1;
            }
        }
        worst.push((d, i));
    }
    // NaN-aware ordering (a non-finite delta sorts last, never panics) + a
    // LOUD census — a NaN NLL is decision data (the file's forward diverged
    // at that position), never a crash.
    worst.sort_by(|x, y| y.0.abs().total_cmp(&x.0.abs()));
    // The per-position sidecar: the walk costs ~6 min/file — never make a
    // re-analysis pay it twice. index,nll_base,nll_artifact,delta,argmax_base,argmax_art,next_token
    if let Some(dp) = dump {
        let mut s =
            String::from("index,nll_base,nll_artifact,delta,argmax_base,argmax_art,next_token\n");
        for (i, (b, a)) in base_rows.iter().zip(art_rows.iter()).enumerate() {
            s.push_str(&format!(
                "{},{:.6},{:.6},{:.6},{},{},{}\n",
                i,
                b.nll,
                a.nll,
                a.nll - b.nll,
                b.argmax,
                a.argmax,
                b.next_token
            ));
        }
        std::fs::write(&dp, s).with_context(|| format!("dump {}", dp.display()))?;
        println!("# per-position dump: {}", dp.display());
    }
    println!("# non-finite NLL: base {nonfinite_base} | artifact {nonfinite_art}");
    let pooled = (sum_art - sum_base) / n;
    if let Some((i, art)) = first_bad {
        println!(
            "#   FIRST BAD at scored index {i} ({}) — pooled sums carry the poison, the verdict line is UNDECIDABLE",
            if art { "artifact side" } else { "base side" }
        );
    }
    println!("# scored positions: {n:.0}");
    println!(
        "# mean NLL base {:.6} | artifact {:.6}",
        sum_base / n,
        sum_art / n
    );
    println!("# paired pooled Δ-NLL (artifact − base) {pooled:+.6}");
    println!("# mean |Δ| {:.6}", abs_sum / n);
    println!(
        "# argmax flips {flips}/{:.0} (base-right→art-wrong {flip_base_lost}, art-right→base-wrong {flip_art_lost})",
        n
    );
    println!("# worst 10 positions (Δ, index):");
    for (d, i) in worst.iter().take(10) {
        println!("#   [{i}] Δ {d:+.6}");
    }
    // The pre-registered kill bar (riir-train issue 701, unchanged):
    // the SCREEN survives only on a negative pooled Δ with BOTH sides fully
    // finite. Tie, regression, or any non-finite poison kills the lane here
    // (a NaN side is UNDECIDABLE, never folded into a verdict).
    if nonfinite_base > 0 || nonfinite_art > 0 {
        println!(
            "# VERDICT: UNDECIDABLE — non-finite NLL present (base {nonfinite_base}, artifact {nonfinite_art}); fix the divergence source before any read"
        );
        std::process::exit(2);
    }
    if pooled < 0.0 {
        println!(
            "# VERDICT: WIN — pooled Δ {pooled:+.6} < 0 (the screen survives to the per-family battery)"
        );
    } else {
        println!(
            "# VERDICT: KILL — pooled Δ {pooled:+.6} ≥ 0 (tie or regression; the lane dies here per the issue bar)"
        );
        std::process::exit(1);
    }
    Ok(())
}
