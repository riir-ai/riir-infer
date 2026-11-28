//! `i1254_capture` — the Plato's-Key cell's resumable per-fact capture
//! phase (Issue 1254 A1).
//!
//! One side per invocation (`--side donor|recipient`): loads its model
//! (donor = ternary PQ2_0 via the ternary forward's `layer_capture` tap;
//! recipient = llama BASE via the Issue-395 [`CaptureHook`]), walks the
//! canonical corpus ([`riir_infer_core::i1254_corpus`]) in order, and
//! writes per-fact band keys + recall rows into the resumable capture
//! store. The fit phase joins the two stores (the P4 driver, next unit;
//! the fit math is the landed [`riir_infer_core::i1254_fit`]).
//!
//! Resume law (owner compute decree): a killed/rebooted run re-enters and
//! skips every fact already rowed (its keys file validated first) — the
//! append-only store IS the checkpoint.
//!
//! ```bash
//! cargo run --release --bin i1254_capture --features deltanet_ternary_inference -- \
//!     --side donor --donor /path/Ternary-Bonsai-2.5-27B-PQ2_0.gguf \
//!     --state .i1254_state/ --ctx 512
//! cargo run --release --bin i1254_capture -- \
//!     --side recipient --recipient /path/Llama-3.2-3B-Base.Q8_0.gguf \
//!     --state .i1254_state/ --ctx 512
//! ```
//!
//! A feature-off build refuses the donor side LOUD (exit 3) — never a
//! silent half-run. Not a benchmark: wall clocks are provenance only.
//!
//! [`CaptureHook`]: riir_infer_core::i1254_capture::CaptureHook

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};

use riir_infer_core::gguf_loader::{load_llama_weights_gguf, GgufFile};
use riir_infer_core::i1254_capture::{
    argmax_first, depth_band, file_blake3, recall_decision, subject_tokens, CaptureHook,
    CaptureMeta, CaptureRow, CaptureStore,
};
use riir_infer_core::i1254_corpus::{
    build_prompt, corpus_digest, load_corpus, prompt_parts, relation_of, split_assignment_digest,
    split_of, Fact, Split, MATCH_WINDOW_TOKENS,
};
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_core::transformer::{
    forward_llama, forward_llama_hooked, ForwardContext, MultiLayerKVCache, NoHook,
};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut side: Option<String> = None;
    let mut recipient: Option<String> = None;
    let mut donor: Option<String> = None;
    let mut state = PathBuf::from(".i1254_state/");
    let mut ctx = 512usize;
    let mut limit: Option<usize> = None;

    while let Some(a) = args.next() {
        match a.as_str() {
            "--side" => side = Some(args.next().context("--side needs a value")?),
            "--recipient" => recipient = Some(args.next().context("--recipient needs a value")?),
            "--donor" => donor = Some(args.next().context("--donor needs a value")?),
            "--state" => state = PathBuf::from(args.next().context("--state needs a value")?),
            "--ctx" => {
                ctx = args
                    .next()
                    .context("--ctx needs a value")?
                    .parse()
                    .context("--ctx must be an integer")?
            }
            "--limit" => {
                limit = Some(
                    args.next()
                        .context("--limit needs a value")?
                        .parse()
                        .context("--limit must be an integer")?,
                )
            }
            other => bail!("unknown arg {other:?}"),
        }
    }
    let side = side.context("--side donor|recipient is required")?;
    let facts = load_corpus().map_err(|e| anyhow::anyhow!("corpus: {e}"))?;
    eprintln!(
        "[p0] corpus: {} facts · corpus_digest {} · split_digest {}",
        facts.len(),
        corpus_digest(),
        split_assignment_digest(&facts)
    );

    let res = match side.as_str() {
        "recipient" => {
            let path = recipient.context("--recipient PATH is required for --side recipient")?;
            run_recipient(&path, &state, &facts, ctx, limit)
        }
        "donor" => {
            let path = donor.context("--donor PATH is required for --side donor")?;
            run_donor(&path, &state, &facts, ctx, limit)
        }
        other => bail!("--side must be donor|recipient, got {other:?}"),
    };
    if let Err(e) = &res {
        let msg = format!("{e:#}");
        if msg.contains("DONOR CAPTURE NEEDS THE TERNARY FEATURE") {
            std::process::exit(3);
        }
    }
    res
}

/// The row's split char, RECOMPUTED from the corpus at write time (the
/// join re-checks it — one law, two enforcement points).
fn split_char_of(fact: &Fact) -> char {
    match split_of(&fact.subj) {
        Split::Train => 'T',
        Split::Val => 'V',
        Split::Test => 'S',
    }
}

fn run_recipient(
    path: &str,
    state: &Path,
    facts: &[Fact],
    ctx: usize,
    limit: Option<usize>,
) -> Result<()> {
    let recipient_path = PathBuf::from(path);
    let gguf = GgufFile::open(&recipient_path)
        .with_context(|| format!("opening recipient {}", recipient_path.display()))?;
    let arch = gguf.architecture().unwrap_or("unknown").to_string();
    if arch != "llama" {
        bail!(
            "recipient GGUF header architecture is '{arch}', expected 'llama' \
             (the architecture law: headers, never filenames)"
        );
    }
    let tok = BpeTokenizer::from_gguf(&gguf).context("recipient BPE tokenizer")?;
    drop(gguf);
    let (mut config, weights) = load_llama_weights_gguf(&recipient_path)
        .with_context(|| format!("loading recipient weights {}", recipient_path.display()))?;
    // ⛔ ctx clamp: the checkpoint's context_length must never size the
    // KV cache — clamp to the actual sequence budget.
    config.block_size = ctx;
    let rope_table = weights.build_rope_freq_table(&config);
    eprintln!(
        "[p0] recipient: d={} layers={} kv_heads={} head_dim={} vocab={} rope_factors={} ctx_clamped={}",
        config.n_embd,
        config.n_layer,
        config.n_kv_head,
        config.head_dim,
        config.vocab_size,
        weights.rope_freq_weights.is_some(),
        config.block_size
    );
    eprintln!(
        "[p0] provenance: blake3 {} …",
        file_blake3(&recipient_path)?
    );

    let band = depth_band(config.n_layer);
    let band_static: &'static [usize] = Box::leak(band.clone().into_boxed_slice());
    eprintln!(
        "[p0] depth band 22–38%: layers {band:?} of {}",
        config.n_layer
    );

    // ── G3 real-model identity: plain ≡ hooked(NoHook), bit-exact ──
    {
        let probe = "The capital of France is";
        let tokens = tok.encode_with_bos(probe);
        let run = |hooked: bool| -> Vec<Vec<f32>> {
            let cache = &mut MultiLayerKVCache::new(&config);
            let mut fctx = ForwardContext::new(&config);
            fctx.rope_freq_table = rope_table.clone();
            let mut hook = NoHook;
            let mut snap = Vec::new();
            for (pos, &t) in tokens.iter().enumerate() {
                let logits = if hooked {
                    forward_llama_hooked(&mut fctx, &weights, cache, t, pos, &config, &mut hook)
                } else {
                    forward_llama(&mut fctx, &weights, cache, t, pos, &config)
                };
                snap.push(logits.to_vec());
            }
            snap
        };
        let plain = run(false);
        let hooked = run(true);
        for (pos, (a, b)) in plain.iter().zip(&hooked).enumerate() {
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "G3 REAL-MODEL IDENTITY FAILED at pos {pos} logit {i} — the seam is not a no-op"
                );
            }
        }
        eprintln!("[p1] G3 PASS: {} positions bit-identical", plain.len());
    }

    let store = CaptureStore::open(state, "recipient").context("capture store")?;
    let done = store.load_done()?;
    eprintln!("[p2] resume: {} fact(s) already captured", done.len());

    let mut attempted = 0usize;
    let mut hits = 0usize;
    let mut skips_boundary = 0usize;
    let mut skips_ctx = 0usize;
    let mut resumed = 0usize;
    let t0 = Instant::now();
    for (fi, fact) in facts.iter().enumerate() {
        if done.contains(&fi) {
            resumed += 1;
            continue;
        }
        if limit.is_some_and(|l| attempted >= l) {
            break;
        }
        attempted += 1;
        let rel = relation_of(fact.rel).map_err(|e| anyhow::anyhow!("fact {fi}: {e}"))?;
        let prompt = build_prompt(rel, &fact.subj);
        let (prefix, _suffix) = prompt_parts(rel, &fact.subj);
        let split_char = split_char_of(fact);
        let row_base =
            |boundary_ok: bool, recall: bool, note: &str, subj_pos: usize, n_tokens: usize| {
                CaptureRow {
                    fact: fi,
                    rel: fact.rel.to_string(),
                    split: split_char,
                    subj_pos,
                    n_tokens,
                    recall,
                    boundary_ok,
                    note: note.to_string(),
                }
            };
        let Ok((tokens, subj_pos)) = subject_tokens(&tok, &prompt, &prefix) else {
            skips_boundary += 1;
            store.append_row(&row_base(false, false, "boundary", 0, 0))?;
            continue;
        };
        if tokens.len() + MATCH_WINDOW_TOKENS + 1 > ctx {
            skips_ctx += 1;
            store.append_row(&row_base(true, false, "ctx", subj_pos, tokens.len()))?;
            continue;
        }
        // Prefill with the capture hook armed at the subject position only
        // (the other positions run the plain forward — NoHook monomorphizes
        // away there by construction).
        let cache = &mut MultiLayerKVCache::new(&config);
        let mut fctx = ForwardContext::new(&config);
        fctx.rope_freq_table = rope_table.clone();
        let mut hook = CaptureHook::new(band_static);
        let mut logits = Vec::new();
        for (pos, &t) in tokens.iter().enumerate() {
            let out = if pos == subj_pos {
                hook = CaptureHook::new(band_static);
                forward_llama_hooked(&mut fctx, &weights, cache, t, pos, &config, &mut hook)
                    .to_vec()
            } else {
                forward_llama(&mut fctx, &weights, cache, t, pos, &config).to_vec()
            };
            logits = out;
        }
        // Greedy window (the paired filter's recipient half).
        let mut gen_toks: Vec<usize> = Vec::with_capacity(MATCH_WINDOW_TOKENS);
        let mut next = argmax_first(&logits);
        for _ in 0..MATCH_WINDOW_TOKENS {
            if next == tok.eos_id() {
                break;
            }
            gen_toks.push(next);
            let pos = tokens.len() + gen_toks.len() - 1;
            let out = forward_llama(&mut fctx, &weights, cache, next, pos, &config).to_vec();
            next = argmax_first(&out);
        }
        let hit = recall_decision(&gen_toks, &tok, fact, rel);
        if hit {
            hits += 1;
        }
        // Keys: the captured band slots (never a half-captured write —
        // every slot must have snapped, else refuse loud).
        let mut planes: Vec<Vec<f32>> = Vec::with_capacity(band.len());
        for (slot, captured) in hook.captured.iter().enumerate() {
            match captured {
                Some(v) => planes.push(v.clone()),
                None => bail!(
                    "fact {fi}: band slot {slot} (layer {}) did not capture — the hook saw fewer layers than the band",
                    band[slot]
                ),
            }
        }
        store.write_keys(fi, &planes)?;
        store.append_row(&row_base(true, hit, "", subj_pos, tokens.len()))?;
        if attempted.is_multiple_of(25) {
            eprintln!(
                "[p2] {attempted} facts ({hits} recalled) in {:.1?}",
                t0.elapsed()
            );
        }
    }
    let meta = CaptureMeta {
        side: "recipient".into(),
        model_path: recipient_path.display().to_string(),
        model_blake3: file_blake3(&recipient_path)?,
        n_layer: config.n_layer,
        band,
        dim: config.n_embd,
        ctx,
        facts_attempted: attempted,
        recall_hits: hits,
        skips_boundary,
        skips_ctx,
    };
    store.write_meta(&meta)?;
    eprintln!(
        "[done] recipient capture: {attempted} attempted ({resumed} resumed-past), {hits} recalled, \
         boundary skips {skips_boundary}, ctx skips {skips_ctx} in {:.1?}",
        t0.elapsed()
    );
    Ok(())
}

fn run_donor(
    path: &str,
    state: &Path,
    facts: &[Fact],
    ctx: usize,
    limit: Option<usize>,
) -> Result<()> {
    #[cfg(not(feature = "deltanet_ternary_inference"))]
    {
        let _ = (path, state, facts, ctx, limit);
        bail!(
            "DONOR CAPTURE NEEDS THE TERNARY FEATURE (exit 3): rebuild with \
             --features deltanet_ternary_inference — the donor of record \
             (Ternary-Bonsai-2.5-27B PQ2_0) runs the ternary body; a feature-off \
             build refuses, never a silent skip."
        );
    }
    #[cfg(feature = "deltanet_ternary_inference")]
    {
        use riir_infer_core::deltanet::forward::{
            effective_rotary_dim, HybridCache, HybridForwardScratch,
        };
        use riir_infer_core::deltanet::ternary_forward::forward_qwen_deltanet_ternary_with_capture;
        use riir_infer_core::gguf_loader::load_qwen_deltanet_ternary_weights_gguf;
        use riir_infer_core::rope::RopeFreqTable;

        let donor_path = PathBuf::from(path);
        // The ternary body discriminator (sink_audit's law): the globals
        // tensor's dtype decides — BEFORE any weight load.
        {
            use riir_infer_core::gguf_loader::GgmlType;
            let probe = GgufFile::open(&donor_path)
                .with_context(|| format!("open {}", donor_path.display()))?;
            let is_ternary = probe
                .tensor_info("token_embd.weight")
                .map(|info| matches!(info.ggml_type, GgmlType::Q2_0 | GgmlType::PTQ1_0))
                .unwrap_or(false);
            if !is_ternary {
                bail!(
                    "donor GGUF body is DENSE — the donor of record is the PQ2_0 \
                     ternary checkpoint; a dense-donor posture (A2-class) is not \
                     this cell's arm"
                );
            }
        }
        let (mut don_config, don_weights) = load_qwen_deltanet_ternary_weights_gguf(&donor_path)
            .with_context(|| format!("load donor (ternary) {}", donor_path.display()))?;
        let gguf = GgufFile::open(&donor_path)?;
        let tok = BpeTokenizer::from_gguf(&gguf).context("donor BPE tokenizer")?;
        drop(gguf);
        // Issue-929 clamp: never the model's 128k-class block_size.
        don_config.block_size = ctx;
        let don_rope = RopeFreqTable::new(don_config.rope_theta, effective_rotary_dim(&don_config));
        eprintln!(
            "[p3] donor: layers={} n_embd={} folded={} vocab={} ctx_clamped={}",
            don_config.n_layer,
            don_config.n_embd,
            don_weights.rotation.is_some(),
            don_config.vocab_size,
            don_config.block_size
        );
        eprintln!("[p3] provenance: blake3 {} …", file_blake3(&donor_path)?);

        let band = depth_band(don_config.n_layer);
        eprintln!(
            "[p3] depth band 22–38%: layers {band:?} of {}",
            don_config.n_layer
        );

        let store = CaptureStore::open(state, "donor").context("capture store")?;
        let done = store.load_done()?;
        eprintln!("[p3] resume: {} fact(s) already captured", done.len());

        let mut attempted = 0usize;
        let mut hits = 0usize;
        let mut skips_boundary = 0usize;
        let mut skips_ctx = 0usize;
        let mut resumed = 0usize;
        let t0 = Instant::now();
        for (fi, fact) in facts.iter().enumerate() {
            if done.contains(&fi) {
                resumed += 1;
                continue;
            }
            if limit.is_some_and(|l| attempted >= l) {
                break;
            }
            attempted += 1;
            let rel = relation_of(fact.rel).map_err(|e| anyhow::anyhow!("fact {fi}: {e}"))?;
            let prompt = build_prompt(rel, &fact.subj);
            let (prefix, _suffix) = prompt_parts(rel, &fact.subj);
            let split_char = split_char_of(fact);
            let row_base =
                |boundary_ok: bool, recall: bool, note: &str, subj_pos: usize, n_tokens: usize| {
                    CaptureRow {
                        fact: fi,
                        rel: fact.rel.to_string(),
                        split: split_char,
                        subj_pos,
                        n_tokens,
                        recall,
                        boundary_ok,
                        note: note.to_string(),
                    }
                };
            let Some((tokens, subj_pos)) = subject_tokens(&tok, &prompt, &prefix)
                .map_err(|e| e.to_string())
                .ok()
            else {
                skips_boundary += 1;
                store.append_row(&row_base(false, false, "boundary", 0, 0))?;
                continue;
            };
            if tokens.len() + MATCH_WINDOW_TOKENS + 1 > ctx {
                skips_ctx += 1;
                store.append_row(&row_base(true, false, "ctx", subj_pos, tokens.len()))?;
                continue;
            }
            let cache = &mut HybridCache::with_layer_types_bounded(
                &don_config,
                &don_weights.layer_types,
                ctx,
            );
            let scratch = &mut HybridForwardScratch::new(&don_config);
            // The forward returns a slice INTO the caller's buffer.
            let mut logits = vec![0.0f32; don_config.vocab_size];
            let mut band_keys: Vec<Vec<f32>> = band
                .iter()
                .map(|_| vec![0.0f32; don_config.n_embd])
                .collect();
            // The forward writes INTO the caller's `logits` buffer and
            // returns the live prefix — capture its length (the borrow
            // ends with the statement), never a self-copy.
            let mut out_len = logits.len();
            for (pos, &t) in tokens.iter().enumerate() {
                let capture = if pos == subj_pos {
                    Some(&mut band_keys[..])
                } else {
                    None
                };
                let out = forward_qwen_deltanet_ternary_with_capture(
                    &mut logits,
                    &don_weights,
                    cache,
                    t,
                    pos,
                    &don_config,
                    scratch,
                    &don_rope,
                    capture,
                );
                out_len = out.len();
            }
            let mut next = argmax_first(&logits[..out_len]);
            // Greedy window (the paired filter's donor half).
            let mut gen_toks: Vec<usize> = Vec::with_capacity(MATCH_WINDOW_TOKENS);
            for _ in 0..MATCH_WINDOW_TOKENS {
                if next == tok.eos_id() {
                    break;
                }
                gen_toks.push(next);
                let pos = tokens.len() + gen_toks.len() - 1;
                let out = forward_qwen_deltanet_ternary_with_capture(
                    &mut logits,
                    &don_weights,
                    cache,
                    next,
                    pos,
                    &don_config,
                    scratch,
                    &don_rope,
                    None,
                );
                next = argmax_first(out);
            }
            let hit = recall_decision(&gen_toks, &tok, fact, rel);
            if hit {
                hits += 1;
            }
            store.write_keys(fi, &band_keys)?;
            store.append_row(&row_base(true, hit, "", subj_pos, tokens.len()))?;
            if attempted.is_multiple_of(10) {
                eprintln!(
                    "[p3] {attempted} facts ({hits} recalled) in {:.1?}",
                    t0.elapsed()
                );
            }
        }
        let meta = CaptureMeta {
            side: "donor".into(),
            model_path: donor_path.display().to_string(),
            model_blake3: file_blake3(&donor_path)?,
            n_layer: don_config.n_layer,
            band,
            dim: don_config.n_embd,
            ctx,
            facts_attempted: attempted,
            recall_hits: hits,
            skips_boundary,
            skips_ctx,
        };
        store.write_meta(&meta)?;
        eprintln!(
            "[done] donor capture: {attempted} attempted ({resumed} resumed-past), {hits} recalled, \
             boundary skips {skips_boundary}, ctx skips {skips_ctx} in {:.1?}",
            t0.elapsed()
        );
        Ok(())
    }
}
