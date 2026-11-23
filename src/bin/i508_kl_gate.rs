//! i508_kl_gate — plan 508's pre-GPU KL minimum-effect gate (the verdict's
//! recovery-signal risk, session `cc0f0014`): at 0.56% weight error the Q8_0
//! student may be near-lossless vs the fp16 teacher — if the teacher→student
//! token-KL on the prompt panel is already within run-noise, the honest
//! outcome is "Q8_0 recovery can't show a difference" and T1 re-aims at a
//! lossier student (Q4_K requant from this same fp16, re-opening T2's
//! registration). This bin MEASURES that KL (the 1242 anchor-KL instrument's
//! first consumer): both models forward the same token panel, the per-
//! position forward-KL(anchor=teacher ‖ candidate=student) folds into the
//! trace-set mean, and the verdict prints against the pre-registered bar.
//!
//! ```bash
//! cargo run --release --features anchor_kl --bin i508_kl_gate -- \
//!     ../riir-train/data/pin/qwen3.5-0.8b-base-fp16/../.. (see DEFAULTS)
//! ```
//!
//! Determinism: greedy forwards (no sampling), the corpus head as the fixed
//! panel, f64 KL folds per the Issue-940 law. Report-only — no gates flip.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};

use riir_infer_core::anchor_kl::{AnchorKlScratch, anchor_kl_full, mean_kl};
use riir_infer_core::corpus_text::load_corpus_text;
use riir_infer_core::deltanet::{
    HybridCache, HybridForwardScratch, effective_rotary_dim, forward_qwen_deltanet,
};
use riir_infer_core::gguf_loader::{GgufFile, load_qwen_deltanet_weights_gguf};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::tokenizer::BpeTokenizer;

const DEFAULT_TEACHER: &str = "E:/tmp/i508/Qwen3.5-0.8B-Base-f16.gguf";
const DEFAULT_STUDENT: &str = "../riir-train/data/Qwen3.5-0.8B-Base-Q8_0.gguf";
const DEFAULT_CORPUS: &str = "../riir-train/data/chat_probe";
const DEFAULT_N: usize = 256;

/// The pre-registered minimum-effect bar: the trace-set mean KL BELOW which
/// the recovery cell "can't show a difference" (nats; the KL of two
/// near-identical distributions on 256-token panels reads ~1e-4..1e-3-class
/// when only Q8_0 rounding separates them — the bar sits at the top of that
/// band, pinned HERE so the verdict is pre-registered, not post-hoc).
const MIN_EFFECT_KL_NATS: f64 = 5e-3;

fn main() -> Result<()> {
    let mut teacher = PathBuf::from(DEFAULT_TEACHER);
    let mut student = PathBuf::from(DEFAULT_STUDENT);
    let mut corpus = PathBuf::from(DEFAULT_CORPUS);
    let mut n = DEFAULT_N;
    let mut it = std::env::args().skip(1);
    while let Some(f) = it.next() {
        match f.as_str() {
            "--n" => n = it.next().expect("--n value").parse().expect("n usize"),
            "--corpus" => corpus = PathBuf::from(it.next().expect("--corpus value")),
            other => {
                let p = PathBuf::from(other);
                if teacher == *DEFAULT_TEACHER && !teacher.exists() || other.contains("f16") {
                    teacher = p;
                } else {
                    student = p;
                }
            }
        }
    }

    // Tokenize the FIXED panel (the corpus head) — one tokenizer serves both
    // models (same qwen35 vocab, probed).
    let probe = GgufFile::open(&student).context("open student")?;
    let tokens = {
        let tok = BpeTokenizer::from_gguf(&probe).context("BPE tokenizer")?;
        let text = load_corpus_text(&corpus).context("load corpus")?;
        tok.encode(&text)
    };
    if tokens.len() < n {
        bail!("corpus encodes {} tokens < the panel {}", tokens.len(), n);
    }
    let panel = &tokens[..n];
    println!("# panel: {} tokens from {}", panel.len(), corpus.display());

    // Load both models (dense qwen35 loaders — same arch family probed).
    println!("# loading teacher (f16) ...");
    let t0 = std::time::Instant::now();
    let (t_config, t_weights) = load_qwen_deltanet_weights_gguf(&teacher)
        .with_context(|| format!("load teacher {}", teacher.display()))?;
    println!(
        "# teacher: layers={} n_embd={} vocab={} | load {:.1}s",
        t_config.n_layer,
        t_config.n_embd,
        t_config.vocab_size,
        t0.elapsed().as_secs_f32()
    );
    let t1 = std::time::Instant::now();
    let (s_config, s_weights) = load_qwen_deltanet_weights_gguf(&student)
        .with_context(|| format!("load student {}", student.display()))?;
    println!(
        "# student: layers={} n_embd={} vocab={} | load {:.1}s",
        s_config.n_layer,
        s_config.n_embd,
        s_config.vocab_size,
        t1.elapsed().as_secs_f32()
    );
    if t_config.vocab_size != s_config.vocab_size {
        bail!(
            "vocab mismatch: teacher {} vs student {} — KL undefined",
            t_config.vocab_size,
            s_config.vocab_size
        );
    }

    let rope_t = RopeFreqTable::new(t_config.rope_theta, effective_rotary_dim(&t_config));
    let rope_s = RopeFreqTable::new(s_config.rope_theta, effective_rotary_dim(&s_config));
    let vocab = t_config.vocab_size;

    let mut cache_t = HybridCache::with_layer_types_bounded(&t_config, &t_weights.layer_types, n);
    let mut scratch_t = HybridForwardScratch::new(&t_config);
    let mut cache_s = HybridCache::with_layer_types_bounded(&s_config, &s_weights.layer_types, n);
    let mut scratch_s = HybridForwardScratch::new(&s_config);

    let mut kl_rows: Vec<f64> = Vec::with_capacity(n - 1);
    let mut kl_scratch = AnchorKlScratch::default();
    let t2 = std::time::Instant::now();

    // Forward both models over the panel, KL at every scored position
    // (position i scores the token at i+1 — the teacher's logits ARE the
    // anchor; the student's the candidate).
    let mut logits_t = vec![0.0f32; vocab];
    let mut logits_s = vec![0.0f32; vocab];
    for (i, &token) in panel.iter().enumerate() {
        let lt = forward_qwen_deltanet(
            &mut logits_t,
            &t_weights,
            &mut cache_t,
            token,
            i,
            &t_config,
            &mut scratch_t,
            &rope_t,
        );
        let ls = forward_qwen_deltanet(
            &mut logits_s,
            &s_weights,
            &mut cache_s,
            token,
            i,
            &s_config,
            &mut scratch_s,
            &rope_s,
        );
        if i + 1 < panel.len() {
            let r = anchor_kl_full(lt, ls, &mut kl_scratch)
                .ok_or_else(|| anyhow::anyhow!("anchor-KL read failed at pos {i}"))?;
            kl_rows.push(r.kl_full);
        }
    }
    println!(
        "# forwards: {:.1}s ({} positions scored)",
        t2.elapsed().as_secs_f32(),
        kl_rows.len()
    );

    let mean = mean_kl(&kl_rows);
    let max = kl_rows.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!("# mean KL(anchor‖student) = {mean:.6} nats | max = {max:.6} nats");
    println!("# pre-registered minimum-effect bar: {MIN_EFFECT_KL_NATS} nats");
    if mean < MIN_EFFECT_KL_NATS {
        println!(
            "VERDICT: BELOW-MINIMUM-EFFECT — the Q8_0 recovery cell cannot show a difference \
             at this quant; T1 re-aims at a lossier student (Q4_K requant from the same fp16; \
             re-opens T2's registration)"
        );
    } else {
        println!(
            "VERDICT: SIGNAL-PRESENT — the recovery cell may proceed; the effect size is \
             measurable above the bar"
        );
    }
    Ok(())
}
