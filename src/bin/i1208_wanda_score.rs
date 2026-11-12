//! Issue 1208 T5 — the Wanda scoring arm: a forward-only sweep of a
//! fixed calibration corpus over the qwen35-hybrid dense lane, folding
//! per-column input squares at each armed layer's gate GEMV, then the
//! saliency artifact `s_ij = |W_ij| · ‖x_j‖` (BLAKE3-stamped, the
//! T7 τ-sweep's input).
//!
//! Determinism law (Issue-940): the fold is sequential token order —
//! the sweep digest rebuilds byte-for-byte on any box given the same
//! checkpoint + corpus. The artifact header carries the corpus digest
//! (BLAKE3 over the token stream) and the checkpoint digest (BLAKE3
//! over the loaded gate tensors), so a cross-box mismatch is a loud
//! digest comparison, never a tolerance.
//!
//! Run: cargo run --release --features deletion_mask,deltanet_inference \
//!   --bin i1208_wanda_score -- <model.gguf> <corpus.txt> <out.iwanda> [layers] [max_tokens]
use riir_infer_core::adapter::wanda::{ScoredArtifact, WandaGateCapture, saliency_for};
use riir_infer_core::corpus_text::load_corpus_text;
use riir_infer_core::deltanet::forward::{
    HybridCache, HybridForwardScratch, forward_qwen_deltanet,
};
use riir_infer_core::gguf_loader::{GgufFile, load_qwen_deltanet_weights_gguf};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::tokenizer::BpeTokenizer;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (model, corpus_path, out_path) = match args.as_slice() {
        [m, c, o] => (m.clone(), c.clone(), o.clone()),
        _ => anyhow::bail!(
            "usage: i1208_wanda_score <model.gguf> <corpus.txt> <out.iwanda> [layers=0,1,..] [max_tokens]"
        ),
    };
    let layers: Vec<usize> = args
        .get(3)
        .map(|s| {
            s.split(',')
                .map(|p| p.parse::<usize>())
                .collect::<Result<_, _>>()
        })
        .transpose()?
        .unwrap_or_else(Vec::new);
    let max_tokens: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(512);

    // the armed layer set: explicit, else every layer
    let probe = GgufFile::open(Path::new(&model))?;
    let tok = BpeTokenizer::from_gguf(&probe)?;
    drop(probe);
    let (config, weights) = load_qwen_deltanet_weights_gguf(Path::new(&model))?;
    let layers = if layers.is_empty() {
        (0..config.n_layer).collect()
    } else {
        layers
    };
    let n = config.n_embd;
    println!(
        "# model: qwen35 hybrid dense | layers={}/{} armed {:?} | n_embd={n}",
        layers.len(),
        config.n_layer,
        layers
    );

    // the corpus: natural text -> token ids (deterministic BPE); the
    // digest binds the TOKEN STREAM (the exact forward input), not the
    // raw bytes
    let text = load_corpus_text(Path::new(&corpus_path))?;
    let token_stream: Vec<usize> = tok.encode(&text).into_iter().take(max_tokens).collect();
    let corpus_digest: [u8; 32] =
        *blake3::hash(bytemuck::cast_slice::<usize, u8>(&token_stream)).as_bytes();
    println!(
        "# corpus: {} tokens (cap {max_tokens}) digest {}",
        token_stream.len(),
        hex(&corpus_digest)
    );

    // the checkpoint digest: BLAKE3 over every gate tensor's loaded bytes
    let mut ckpt = blake3::Hasher::new();
    for (i, layer) in weights.layers.iter().enumerate() {
        ckpt.update(&(i as u32).to_le_bytes());
        for v in layer.gate_proj.dense_data() {
            ckpt.update(&v.to_le_bytes());
        }
    }
    let checkpoint_digest: [u8; 32] = *ckpt.finalize().as_bytes();

    // the sweep: sequential tokens, the capture armed on the scratch
    let mut cache = HybridCache::with_layer_types(&config, &weights.layer_types);
    let mut scratch = HybridForwardScratch::new(&config);
    scratch.wanda_gate_capture = Some(WandaGateCapture::new(&layers, n));
    let rope_freq = RopeFreqTable::new(config.rope_theta, config.rope_dimension_count);
    let mut x = vec![0.0f32; config.vocab_size.max(n) * 2];
    let t0 = std::time::Instant::now();
    for (step, &tok_id) in token_stream.iter().enumerate() {
        forward_qwen_deltanet(
            &mut x,
            &weights,
            &mut cache,
            tok_id,
            step,
            &config,
            &mut scratch,
            &rope_freq,
        );
    }
    // the capture mutated INSIDE the forward (through the scratch field);
    // take it back out for scoring
    let capture = scratch
        .wanda_gate_capture
        .take()
        .ok_or_else(|| anyhow::anyhow!("capture vanished during the sweep"))?;
    let sweep_digest = capture.sweep_digest();
    println!(
        "# sweep: {} tokens in {:.1} s | sweep digest {}",
        capture.tokens_seen(),
        t0.elapsed().as_secs_f64(),
        hex(&sweep_digest)
    );

    // the saliency tables
    let mut tables = Vec::new();
    for &layer in capture.layers() {
        let norms = capture
            .column_norms(layer)
            .ok_or_else(|| anyhow::anyhow!("layer {layer} missing from the capture"))?;
        let data = weights.layers[layer].gate_proj.dense_data();
        let cols = n;
        let scores = saliency_for(data, &norms, cols);
        tables.push((
            format!("blk.{layer}.ffn_gate.weight"),
            data.len() / cols,
            cols,
            scores,
        ));
    }
    let artifact = ScoredArtifact {
        checkpoint_digest,
        corpus_digest,
        sweep_digest,
        tables,
    };
    std::fs::write(&out_path, artifact.to_bytes())?;
    println!(
        "# artifact: {} ({} tables, blob digest {})",
        out_path,
        artifact.tables.len(),
        hex(&artifact.blob_digest())
    );
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
