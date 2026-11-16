//! Issue 1208 T7 — the closed-form mask minting bin: a forward-only
//! calibration sweep (the same posture as `i1208_wanda_score`), then a
//! deterministic solver pass per armed layer, then a RIMSK v1 artifact
//! (the T1 codec — base-BLAKE3 bound, router-safe by construction).
//!
//! Solvers (see `adapter::mint`): `wanda` (the T5 saliency axis at a
//! rate-derived τ), `sparsegpt` (the upstream chunked reconstruction —
//! the SELECTION mask packs into the RIMSK artifact; pass
//! `--emit-compensated` to ALSO write the delta sidecar, a different
//! artifact class per the scope's split), `magnitude` and `random` (the
//! mandatory floors).
//!
//! Determinism law (Issue-940): the same model + corpus + flags mint a
//! byte-identical artifact on any box — the digest line is the pin.
//!
//! Run: cargo run --release --features deletion_mask,deltanet_inference \
//!   --bin i1208_mask_mint -- <model.gguf> <corpus> <out.rimsk> \
//!   [--solver wanda|sparsegpt|magnitude|random] [--rate F] [--seed N] \
//!   [--layers 0,1,..] [--max-tokens N] [--emit-compensated <out.bin>]
use riir_infer_core::adapter::mask_codec::{MaskManifest, MaskSection};
use riir_infer_core::adapter::mint::{
    SparseGptGateCapture, magnitude_keep, random_keep, sparsegpt_reconstruct, wanda_tau_for_rate,
    wanda_tau_keep,
};
use riir_infer_core::adapter::wanda::{WandaGateCapture, saliency_for};
use riir_infer_core::corpus_text::{load_corpus_pages, load_corpus_text};
use riir_infer_core::deltanet::forward::{
    HybridCache, HybridForwardScratch, forward_qwen_deltanet,
};
use riir_infer_core::gguf_loader::{GgufFile, load_qwen_deltanet_weights_gguf};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::tokenizer::BpeTokenizer;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        anyhow::bail!(
            "usage: i1208_mask_mint <model.gguf> <corpus> <out.rimsk> \
             [--solver wanda|sparsegpt|magnitude|random] [--rate F] [--seed N] \
             [--layers 0,1,..] [--max-tokens N] [--emit-compensated <out.bin>]"
        );
    }
    let model = args.remove(0);
    let corpus_path = args.remove(0);
    let out_path = args.remove(0);

    let mut solver = "wanda".to_string();
    let mut rate = 0.001_f64; // the paper's class: <0.05%; default 0.1%
    let mut seed = 0_u64;
    let mut layers_arg = String::new();
    let mut max_tokens = 512_usize;
    let mut emit_compensated: Option<String> = None;
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let mut val = |name: &str| -> anyhow::Result<String> {
            it.next()
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("--{name} needs a value"))
        };
        match flag.as_str() {
            "--solver" => solver = val("solver")?,
            "--rate" => rate = val("rate")?.parse()?,
            "--seed" => seed = val("seed")?.parse()?,
            "--layers" => layers_arg = val("layers")?,
            "--max-tokens" => max_tokens = val("max-tokens")?.parse()?,
            "--emit-compensated" => emit_compensated = Some(val("emit-compensated")?),
            other => anyhow::bail!("unknown flag {other}"),
        }
    }
    match solver.as_str() {
        "wanda" | "sparsegpt" | "magnitude" | "random" => {}
        other => anyhow::bail!("unknown solver {other} (wanda|sparsegpt|magnitude|random)"),
    }
    if solver != "random" && rate <= 0.0 {
        anyhow::bail!("--rate must be > 0 for {solver}");
    }

    // the armed layer set: explicit, else every layer (the wanda_score
    // posture verbatim)
    let probe = GgufFile::open(Path::new(&model))?;
    let tok = BpeTokenizer::from_gguf(&probe)?;
    drop(probe);
    let (config, weights) = load_qwen_deltanet_weights_gguf(Path::new(&model))?;
    let layers: Vec<usize> = if layers_arg.is_empty() || layers_arg == "all" {
        (0..config.n_layer).collect()
    } else {
        layers_arg
            .split(',')
            .map(|p| p.parse::<usize>())
            .collect::<Result<_, _>>()?
    };
    let n = config.n_embd;
    println!(
        "# model: qwen35 hybrid dense | solver={solver} rate={rate} | layers={}/{} armed {:?} | n_embd={n}",
        layers.len(),
        config.n_layer,
        layers
    );

    // the corpus -> token stream; the digest binds the TOKEN STREAM
    let cpath = Path::new(&corpus_path);
    let text = if cpath.is_dir() {
        let pages = load_corpus_pages(cpath)?;
        pages
            .into_iter()
            .map(|(_, body)| body)
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        load_corpus_text(cpath)?
    };
    let token_stream: Vec<usize> = tok.encode(&text).into_iter().take(max_tokens).collect();
    let corpus_digest: [u8; 32] =
        *blake3::hash(bytemuck::cast_slice::<usize, u8>(&token_stream)).as_bytes();
    println!(
        "# corpus: {} tokens (cap {max_tokens}) digest {}",
        token_stream.len(),
        hex(&corpus_digest)
    );

    // the checkpoint digest: BLAKE3 over every gate tensor's loaded
    // bytes — the artifact's base binding (the wrong-base law; the
    // full-model digest must cover EVERY layer, armed or not, since the
    // mask binds the whole checkpoint)
    let mut ckpt = blake3::Hasher::new();
    for (i, layer) in weights.layers.iter().enumerate() {
        ckpt.update(&(i as u32).to_le_bytes());
        for v in layer.gate_proj.dense_data() {
            ckpt.update(&v.to_le_bytes());
        }
    }
    let checkpoint_digest: [u8; 32] = *ckpt.finalize().as_bytes();

    // the sweep: the solver decides which capture rides
    let mut cache = HybridCache::with_layer_types(&config, &weights.layer_types);
    let mut scratch = HybridForwardScratch::new(&config);
    if solver == "sparsegpt" {
        scratch.sparsegpt_gate_capture = Some(SparseGptGateCapture::new(&layers, n));
    } else {
        scratch.wanda_gate_capture = Some(WandaGateCapture::new(&layers, n));
    }
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
    println!(
        "# sweep: {} tokens in {:.1} s",
        token_stream.len(),
        t0.elapsed().as_secs_f64()
    );

    // per-layer solve -> keep masks. Solver inputs differ; the wanda
    // path needs per-column norms (its capture), the sparsegpt path the
    // per-layer Gram (its capture) — each taken out of the scratch.
    let mut keeps: Vec<(String, Vec<bool>)> = Vec::new();
    let mut compensated_out: Option<Vec<Vec<f32>>> = None;
    match solver.as_str() {
        "wanda" => {
            let cap = scratch
                .wanda_gate_capture
                .take()
                .ok_or_else(|| anyhow::anyhow!("capture vanished during the sweep"))?;
            println!("# wanda sweep digest {}", hex(&cap.sweep_digest()));
            for &layer in &layers {
                let norms = cap
                    .column_norms(layer)
                    .ok_or_else(|| anyhow::anyhow!("layer {layer} missing from the capture"))?;
                let data = weights.layers[layer].gate_proj.dense_data();
                let scores = saliency_for(data, &norms, n);
                let tau = wanda_tau_for_rate(&scores, rate);
                keeps.push((
                    format!("blk.{layer}.ffn_gate.weight"),
                    wanda_tau_keep(&scores, tau),
                ));
            }
        }
        "sparsegpt" => {
            let cap = scratch
                .sparsegpt_gate_capture
                .take()
                .ok_or_else(|| anyhow::anyhow!("capture vanished during the sweep"))?;
            println!(
                "# sparsegpt sweep: {} tokens | digest {}",
                cap.tokens_seen(),
                hex(&cap.sweep_digest())
            );
            let mut comp_all = Vec::new();
            for &layer in &layers {
                let gram = cap
                    .gram(layer)
                    .ok_or_else(|| anyhow::anyhow!("layer {layer} missing from the capture"))?
                    .to_vec();
                let data = weights.layers[layer].gate_proj.dense_data();
                let out = sparsegpt_reconstruct(data, &gram, n, rate, 128, 0.01);
                println!(
                    "#   blk.{layer}: deletions {} loss {:.6}",
                    out.keep.iter().filter(|&&k| !k).count(),
                    out.loss
                );
                keeps.push((format!("blk.{layer}.ffn_gate.weight"), out.keep));
                comp_all.push(out.compensated);
            }
            compensated_out = Some(comp_all);
        }
        "magnitude" => {
            for &layer in &layers {
                let data = weights.layers[layer].gate_proj.dense_data();
                keeps.push((
                    format!("blk.{layer}.ffn_gate.weight"),
                    magnitude_keep(data, rate),
                ));
            }
        }
        "random" => {
            let per_layer = weights.layers[layers[0]].gate_proj.dense_data().len();
            for &layer in &layers {
                keeps.push((
                    format!("blk.{layer}.ffn_gate.weight"),
                    random_keep(
                        per_layer,
                        rate,
                        seed ^ (layer as u64).wrapping_mul(0x9E37_79B9),
                    ),
                ));
            }
        }
        _ => unreachable!("solver validated at the flag parse"),
    }

    // pack: sequential sections, bit ranges in keeps order; 1 = keep
    // (LSB-first) per the codec's convention.
    let mut sections = Vec::new();
    let mut bit_offset = 0_u64;
    for (name, keep) in &keeps {
        sections.push(MaskSection::new(
            name.clone(),
            bit_offset,
            keep.len() as u64,
        )?);
        bit_offset += keep.len() as u64;
    }
    let total_bits = bit_offset as usize;
    let mut bits = vec![0_u8; total_bits.div_ceil(8)];
    for (name, keep) in &keeps {
        let _ = name;
        let sec = sections
            .iter()
            .find(|s| s.tensor_name == *name)
            .ok_or_else(|| anyhow::anyhow!("section vanished"))?;
        let base = sec.bit_offset as usize;
        for (i, &k) in keep.iter().enumerate() {
            if k {
                let p = base + i;
                bits[p / 8] |= 1 << (p % 8);
            }
        }
    }
    let manifest = MaskManifest::new(checkpoint_digest, sections)?;
    let artifact = manifest.pack(&bits, None)?;
    std::fs::write(&out_path, &artifact)?;
    println!(
        "# artifact: {} ({} sections, {} bits, {} bytes) — quote the digest:",
        out_path,
        keeps.len(),
        total_bits,
        artifact.len()
    );
    println!("# rimsk blake3 {}", hex(blake3::hash(&artifact).as_bytes()));

    // the delta sidecar (solver ii-b's compensation — a DIFFERENT
    // artifact class, never packed into the RIMSK v1 artifact)
    if let (Some(path), Some(all)) = (emit_compensated, compensated_out) {
        let mut blob = Vec::new();
        blob.extend_from_slice(b"I1208COMP1");
        blob.extend_from_slice(&checkpoint_digest);
        blob.extend_from_slice(&corpus_digest);
        blob.extend_from_slice(&(rate.to_le_bytes()));
        blob.extend_from_slice(&(all.len() as u32).to_le_bytes());
        for (idx, comp) in all.iter().enumerate() {
            blob.extend_from_slice(&(layers[idx] as u32).to_le_bytes());
            blob.extend_from_slice(&(comp.len() as u64).to_le_bytes());
            for v in comp {
                blob.extend_from_slice(&v.to_le_bytes());
            }
        }
        std::fs::write(Path::new(&path), &blob)?;
        println!(
            "# compensated sidecar: {path} ({} tensors, blake3 {})",
            all.len(),
            hex(blake3::hash(&blob).as_bytes())
        );
    }
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
