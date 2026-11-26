//! spike-code-census — Issue 1252 A2 front (the readout-screen bin shape):
//! `--self-test` runs the fixture gates in-process (the canary vehicle);
//! `--gguf/--corpus/--out` is the census cell (capture via the
//! closed_form_write CaptureSink seam over the DENSE body, then the
//! per-k spike-code census — SPKC1 artifact + digest + report).
//!
//! The TERNARY body has no batched_mlp capture hook yet (the dense
//! prefill's `batched_mlp` is the seam's only write-site) — the league
//! ternary cell is the follow-up slice recorded in the issue; this bin
//! REFUSES a ternary GGUF loud (the sink_audit dispatch lesson: detect
//! the body BEFORE any weight load) rather than mis-shaping the file.

use std::path::PathBuf;

use anyhow::{Context as _, Result, bail};
use riir_infer_core::closed_form_write::CaptureSink;
use riir_infer_core::corpus_text::load_corpus_text;
use riir_infer_core::deltanet::effective_rotary_dim;
use riir_infer_core::deltanet::forward::{
    HybridCache, HybridForwardScratch, PrefillContext, prefill_qwen_deltanet_into,
};
use riir_infer_core::gguf_loader::{GgmlType, GgufFile, load_qwen_deltanet_weights_gguf};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::spike_code_census::{K_LADDER, SPKC_MAGIC, SpikeCensusArtifact, census_layer};
use riir_infer_core::tokenizer::BpeTokenizer;

fn main() -> Result<()> {
    let args = Args::parse()?;
    if args.self_test {
        let n = riir_infer_core::spike_code_census_fixture::run_all().context("fixture battery")?;
        println!("# self-test: {n} fixture gates PASS");
        return Ok(());
    }
    let gguf = args
        .gguf
        .clone()
        .context("--gguf required (or --self-test)")?;
    let corpus = args.corpus.clone().context("--corpus required")?;
    let out = args.out.clone().context("--out required")?;

    // Body dispatch BEFORE any weight load (the sink_audit lesson): a
    // ternary GGUF refuses loud — the dense prefill's batched_mlp hook is
    // the capture seam's only write-site, and the dense loader mis-shapes
    // a PQ2_0 file.
    let probe = GgufFile::open(&gguf).with_context(|| format!("open {}", gguf.display()))?;
    let is_ternary = probe
        .tensor_info("token_embd.weight")
        .map(|info| matches!(info.ggml_type, GgmlType::Q2_0 | GgmlType::PTQ1_0))
        .unwrap_or(false);
    if is_ternary {
        bail!(
            "ternary GGUF: the dense prefill's batched_mlp hook is the capture seam's only \
             write-site — the league ternary cell is the recorded follow-up slice (issue 1252); \
             refusing rather than mis-shaping the file"
        );
    }

    // Corpus + tokenizer (the fleet corpus loader — the ag_news JSON
    // pages' text extraction, not a raw file read).
    let corpus_text =
        load_corpus_text(&corpus).with_context(|| format!("load corpus {}", corpus.display()))?;
    let corpus_digest = blake3::hash(corpus_text.as_bytes()).to_hex().to_string();
    let tokens = {
        let tok = BpeTokenizer::from_gguf(&probe).context("gpt2 BPE tokenizer from gguf")?;
        tok.encode(&corpus_text)
    };
    if tokens.len() < args.tokens {
        bail!(
            "corpus encodes {} tokens < the census window {} — shorten --tokens or grow the corpus",
            tokens.len(),
            args.tokens
        );
    }

    // Model digest (the artifact's identity pin) — the file bytes.
    let t0 = std::time::Instant::now();
    let model_bytes = std::fs::read(&gguf).with_context(|| format!("read {}", gguf.display()))?;
    let model_digest = blake3::hash(&model_bytes).to_hex().to_string();
    let (config, weights) = load_qwen_deltanet_weights_gguf(&gguf)
        .with_context(|| format!("load {}", gguf.display()))?;
    println!(
        "# model: qwen35 hybrid | body=dense | layers={} n_embd={} | load {:.1}s | file {} B",
        config.n_layer,
        config.n_embd,
        t0.elapsed().as_secs_f32(),
        model_bytes.len()
    );
    drop(model_bytes);

    // Capture: ALL layers, the census window's positions.
    let layers: Vec<usize> = (0..config.n_layer).collect();
    let seq: Vec<usize> = tokens[..args.tokens].to_vec();
    let mut cache = HybridCache::with_layer_types(&config, &weights.layer_types);
    let mut scratch = HybridForwardScratch::new(&config);
    let mut prefill_ctx = PrefillContext::new(&config, args.tokens);
    prefill_ctx.closed_form_capture = Some(CaptureSink::new(layers.clone()));
    let rope = RopeFreqTable::new(config.rope_theta, effective_rotary_dim(&config));
    let v = config.vocab_size;
    let mut logits = vec![0.0f32; v];
    let t1 = std::time::Instant::now();
    prefill_qwen_deltanet_into(
        &weights,
        &config,
        &mut cache,
        &seq,
        &mut scratch,
        &rope,
        &mut prefill_ctx,
        &mut logits,
        None,
    );
    let sink = prefill_ctx
        .closed_form_capture
        .take()
        .context("capture sink vanished")?;
    println!(
        "# capture pass: {:.1}s ({} layers x {} positions, mlp={} n={})",
        t1.elapsed().as_secs_f32(),
        sink.layers.len(),
        sink.rows,
        sink.mlp,
        sink.n
    );
    assert_eq!(sink.rows, args.tokens, "capture rows must equal the window");

    // Census: per layer, per k. Rows = each position's activation vector.
    let t2 = std::time::Instant::now();
    let mut stats = Vec::with_capacity(layers.len() * K_LADDER.len());
    for (li, _layer) in layers.iter().enumerate() {
        let keys = &sink.keys[li];
        let rows: Vec<Vec<f32>> = (0..sink.rows)
            .map(|p| keys[p * sink.mlp..(p + 1) * sink.mlp].to_vec())
            .collect();
        for &k in K_LADDER.iter() {
            let st = census_layer(&rows, k).with_context(|| format!("layer {li} k {k}"))?;
            stats.push(st);
        }
    }
    println!("# census arithmetic: {:.1}s", t2.elapsed().as_secs_f32());

    let artifact = SpikeCensusArtifact {
        model_digest_hex: model_digest,
        corpus_digest_hex: corpus_digest,
        tokens: args.tokens,
        width: sink.mlp,
        k_ladder: K_LADDER.to_vec(),
        layers,
        stats,
    };
    let out_dir = PathBuf::from(&out);
    let digest = artifact.write(&out_dir).context("write artifact")?;
    let report = artifact.render(&gguf.display().to_string());
    print!("{report}");
    println!(
        "# artifact: {} (digest {}) [{}]",
        out_dir.join("spike_census.spkc").display(),
        hex(&digest),
        std::str::from_utf8(SPKC_MAGIC).unwrap_or("SPKC")
    );
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

struct Args {
    gguf: Option<PathBuf>,
    corpus: Option<PathBuf>,
    out: Option<PathBuf>,
    tokens: usize,
    self_test: bool,
}

impl Args {
    fn parse() -> Result<Args> {
        let mut a = Args {
            gguf: None,
            corpus: None,
            out: None,
            tokens: 256,
            self_test: false,
        };
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let mut val = || it.next().with_context(|| format!("{flag} needs a value"));
            match flag.as_str() {
                "--gguf" => a.gguf = Some(PathBuf::from(val()?)),
                "--corpus" => a.corpus = Some(PathBuf::from(val()?)),
                "--out" => a.out = Some(PathBuf::from(val()?)),
                "--tokens" => a.tokens = val()?.parse().context("--tokens parse")?,
                "--self-test" => a.self_test = true,
                other => bail!(
                    "unknown flag {other} (flags: --gguf --corpus --out --tokens --self-test)"
                ),
            }
        }
        Ok(a)
    }
}
