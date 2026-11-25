//! hisa_census — the Issue-1244 T2.1b census sweep driver front: the
//! rotation-suppression census over the REAL ternary checkpoint (report
//! only). T2.2 runs this; T3 reads its report.
//!
//! ```bash
//! # The smoke cell (one layer × one chunk × one band, short read):
//! cargo run --release --features hisa_census --bin hisa_census -- \
//!     ../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf \
//!     --read-len 256 --layers 3 --chunks 0 --bands 0 \
//!     --out /tmp/hisa_smoke_report.txt --state-dir /tmp/hisa_smoke_state
//! # The full sweep (resumable — kill/resume is bit-identical):
//! cargo run --release --features hisa_census --bin hisa_census -- \
//!     ../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf \
//!     --read-len 2048 --out .benchmarks/hisa_census_report.txt \
//!     --state-dir .hisa_census_state --resume
//! ```
//!
//! REPORT-ONLY: exit 0 on a successful sweep; every refusal is loud and
//! typed. Never a serving path.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, bail};

use riir_infer_core::corpus_text::load_corpus_text;
use riir_infer_core::deltanet::effective_rotary_dim;
use riir_infer_core::deltanet::forward::{HybridCache, HybridForwardScratch};
use riir_infer_core::gguf_loader::{GgmlType, GgufFile, load_qwen_deltanet_ternary_weights_gguf};
use riir_infer_core::hisa_census::HisaCensusArm;
use riir_infer_core::hisa_census_run::{
    CellResult, CellSpec, CensusLedger, DEFAULT_READ_LEN, SweepIdentity, band_grid,
    build_cell_table, canonical_report_bytes, chunk_grid, committed_specs, filter_cell_table,
    flip_count, manifest_digest, mean, render_report, run_read_pass,
};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_core::types::{DeltaNetLayerType, kv_dim};

struct Args {
    model: PathBuf,
    corpus: PathBuf,
    read_len: usize,
    null_offset: Option<usize>,
    layers: Option<Vec<usize>>,
    chunks: Option<Vec<usize>>,
    bands: Option<Vec<usize>>,
    limit: Option<usize>,
    out: Option<PathBuf>,
    state_dir: Option<PathBuf>,
    resume: bool,
}

fn parse_list(s: &str) -> Vec<usize> {
    s.split(',')
        .map(|p| {
            p.trim()
                .parse()
                .unwrap_or_else(|_| panic!("bad list value `{p}`"))
        })
        .collect()
}

fn parse_args() -> Args {
    let mut a = Args {
        model: PathBuf::from("../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf"),
        corpus: PathBuf::from("../riir-train/data/chat_probe"),
        read_len: DEFAULT_READ_LEN,
        null_offset: None,
        layers: None,
        chunks: None,
        bands: None,
        limit: None,
        out: None,
        state_dir: None,
        resume: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(f) = it.next() {
        let mut val = |name: &str| -> String {
            it.next()
                .unwrap_or_else(|| panic!("--{name} needs a value"))
        };
        match f.as_str() {
            "--corpus" => a.corpus = val("corpus").into(),
            "--read-len" => a.read_len = val("read-len").parse().unwrap(),
            "--null-offset" => a.null_offset = Some(val("null-offset").parse().unwrap()),
            "--layers" => a.layers = Some(parse_list(&val("layers"))),
            "--chunks" => a.chunks = Some(parse_list(&val("chunks"))),
            "--bands" => a.bands = Some(parse_list(&val("bands"))),
            "--limit" => a.limit = Some(val("limit").parse().unwrap()),
            "--out" => a.out = Some(val("out").into()),
            "--state-dir" => a.state_dir = Some(val("state-dir").into()),
            "--resume" => a.resume = true,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other if other.starts_with('-') => panic!("unknown arg {other}"),
            other => a.model = other.into(),
        }
    }
    a
}

fn print_help() {
    println!(
        "hisa_census <model.gguf> [--corpus PATH] [--read-len N] [--null-offset O]\n\
         [--layers L1,L2] [--chunks I1,I2] [--bands I1,I2] [--limit K]\n\
         [--out report.txt] [--state-dir DIR] [--resume]\n\n\
         Issue 1244 T2.1b: the rotation-suppression census sweep driver\n\
         (report-only). Per (attention layer, 16-feature chunk, dyadic\n\
         log-Δ band): Remove + Replace passes over ONE frozen teacher-forced\n\
         read; pooled golden Δ-NLL + argmax flips vs the base pass, the\n\
         E_remove/E_observe and E_replace/E_observe mass ratios.\n\n\
         --state-dir enables the resumable ledger (per-cell rows, tmp+rename\n\
         commit); --resume continues a matching-digest sweep, refuses a\n\
         foreign one loud. The null document is the corpus slice at\n\
         --null-offset (default one read past the primary; must be disjoint\n\
         and leave null_offset + read_len + 1 tokens)."
    );
}

fn main() -> Result<()> {
    let args = parse_args();
    let n = args.read_len;
    if n == 0 {
        bail!("--read-len 0: the read needs at least one scored step");
    }
    let null_offset = args.null_offset.unwrap_or(n);
    if null_offset < n {
        bail!(
            "--null-offset {null_offset} overlaps the primary read (read_len {n}) — the E_null noise floor needs a DISJOINT document"
        );
    }

    // Ternary-body v1: refuse anything else loud (the dense mirror is a
    // named follow-up, never a silent fallback).
    let probe =
        GgufFile::open(&args.model).with_context(|| format!("open {}", args.model.display()))?;
    let is_ternary = probe
        .tensor_info("token_embd.weight")
        .map(|info| matches!(info.ggml_type, GgmlType::Q2_0 | GgmlType::PTQ1_0))
        .unwrap_or(false);
    if !is_ternary {
        bail!(
            "hisa_census v1 is the TERNARY body only (token_embd.weight Q2_0/PTQ1_0); {} is not — the dense mirror is a named follow-up",
            args.model.display()
        );
    }
    let model_len = std::fs::metadata(&args.model)
        .with_context(|| format!("stat {}", args.model.display()))?
        .len();

    let corpus_text = load_corpus_text(&args.corpus)
        .with_context(|| format!("load corpus {}", args.corpus.display()))?;
    let tokens = {
        let tok = BpeTokenizer::from_gguf(&probe).context("gpt2 BPE tokenizer from gguf")?;
        tok.encode(&corpus_text)
    };
    let needed = null_offset + n + 1;
    if tokens.len() < needed {
        bail!(
            "corpus encodes {} tokens < the needed {} (read {n} + null at offset {null_offset} + 1 gold) — shorten --read-len or grow the corpus",
            tokens.len(),
            needed
        );
    }
    println!(
        "# corpus: {} | {} tokens encoded (need {needed}) | read={n} null_offset={null_offset}",
        args.corpus.display(),
        tokens.len()
    );

    let (config, weights) = load_qwen_deltanet_ternary_weights_gguf(&args.model)
        .with_context(|| format!("load (ternary) {}", args.model.display()))?;
    let attn_layers: Vec<usize> = weights
        .layer_types
        .iter()
        .enumerate()
        .filter_map(|(i, &t)| (t == DeltaNetLayerType::Attention).then_some(i))
        .collect();
    if attn_layers.is_empty() {
        bail!("no full-attention layers in the checkpoint — the census has no surface");
    }
    println!(
        "# model: qwen35 hybrid | body=ternary | layers={} attn_layers={} n_embd={} n_head={} n_kv_head={} hd={} rotary={} folded={}",
        config.n_layer,
        attn_layers.len(),
        config.n_embd,
        config.n_head,
        config.n_kv_head,
        config.head_dim,
        effective_rotary_dim(&config),
        weights.rotation.is_some()
    );

    // Census posture: the cache is sized to exactly the read (the
    // Issue-929 lesson — never the model's 128k-class block_size).
    let mut census_config = config.clone();
    census_config.block_size = n;
    let kvd = kv_dim(&census_config);
    let rotary = effective_rotary_dim(&census_config);
    if rotary == 0 {
        bail!("effective_rotary_dim == 0 — no rotating pairs, the census has no surface");
    }
    let rope_freq = RopeFreqTable::new(census_config.rope_theta, rotary);

    let chunks = chunk_grid(rotary);
    let bands = band_grid(n);
    let table = build_cell_table(&attn_layers, &chunks, &bands);
    let selected = filter_cell_table(
        &table,
        args.layers.as_deref(),
        args.chunks.as_deref(),
        args.bands.as_deref(),
        args.limit,
        &chunks,
        &bands,
    )
    .map_err(anyhow::Error::msg)?;
    if selected.is_empty() {
        bail!("the filters selected ZERO cells — a green zero is refused; widen the filters");
    }

    // The read + golds; the null document (disjoint, same length).
    let read: Vec<usize> = tokens[..n].to_vec();
    let golds: Vec<usize> = tokens[1..=n].to_vec();
    let null_read: Vec<usize> = tokens[null_offset..null_offset + n].to_vec();
    let null_golds: Vec<usize> = tokens[null_offset + 1..null_offset + n + 1].to_vec();

    // Ledger.
    let tokens_digest = blake3::hash(&{
        let mut b = Vec::with_capacity(tokens.len() * 8);
        for t in &tokens {
            b.extend_from_slice(&(*t as u64).to_le_bytes());
        }
        b
    })
    .to_hex()
    .to_string();
    let config_sig = format!(
        "layers={} n_embd={} n_head={} n_kv_head={} hd={} rotary={} theta={}e0",
        config.n_layer,
        config.n_embd,
        config.n_head,
        config.n_kv_head,
        config.head_dim,
        rotary,
        census_config.rope_theta
    );
    let identity = SweepIdentity {
        model: &args.model.display().to_string(),
        model_len,
        config_sig: &config_sig,
        tokens_digest: &tokens_digest,
        read_len: n,
        null_offset,
        attn_layers: &attn_layers,
        chunks: &chunks,
        bands: &bands,
    };
    let digest = manifest_digest(&identity, &table);
    let ledger = match args.state_dir.as_deref() {
        Some(dir) => {
            if args.resume {
                let led =
                    CensusLedger::open(dir, &digest, table.len()).map_err(anyhow::Error::msg)?;
                println!(
                    "# ledger: resumed {} ({}/{} cells committed)",
                    dir.display(),
                    led.committed(),
                    table.len()
                );
                Some(led)
            } else {
                let led =
                    CensusLedger::create(dir, &digest, table.len()).map_err(anyhow::Error::msg)?;
                println!("# ledger: fresh {} ({} cells)", dir.display(), table.len());
                Some(led)
            }
        }
        None => None,
    };
    let committed_specs = match &ledger {
        Some(led) => committed_specs(led).map_err(anyhow::Error::msg)?,
        None => Vec::new(),
    };
    let pending: Vec<&CellSpec> = selected
        .iter()
        .filter(|c| !committed_specs.contains(c))
        .collect();
    println!(
        "# cells: {} selected, {} committed, {} pending | full table {} cells",
        selected.len(),
        committed_specs.len(),
        pending.len(),
        table.len()
    );
    let pending_count = pending.len();

    // One cache + scratch for the whole sweep (the Issue-838 H1 reset
    // between passes — no per-pass reallocation).
    let mut cache = HybridCache::with_layer_types_bounded(&census_config, &weights.layer_types, n);
    let mut scratch = HybridForwardScratch::new(&census_config);

    // ── the base pass (no arm) ──
    let t0 = Instant::now();
    let (base_nlls, base_top1, _) = run_read_pass(
        &weights,
        &census_config,
        &read,
        &golds,
        &rope_freq,
        &mut cache,
        &mut scratch,
        None,
    )
    .map_err(anyhow::Error::msg)?;
    let base_nll_mean = mean(&base_nlls);
    let pass_wall = t0.elapsed().as_secs_f64();
    println!(
        "# base pass: {:.1}s (steps {n}, mean NLL {:.6})",
        pass_wall, base_nll_mean
    );
    let eta_secs = (pending_count as f64) * 2.0 * pass_wall.max(1e-6)
        + (pending_count as f64) * pass_wall.max(1e-6) / (layer_estimate(&selected) as f64);
    println!(
        "# eta: {} pending cells × ~2.1 passes × {:.1}s ≈ {:.0} min",
        pending_count,
        pass_wall.max(1e-6),
        eta_secs / 60.0
    );

    // ── the null snapshot pass ──
    let t0 = Instant::now();
    let _ = run_read_pass(
        &weights,
        &census_config,
        &null_read,
        &null_golds,
        &rope_freq,
        &mut cache,
        &mut scratch,
        None,
    )
    .map_err(anyhow::Error::msg)?;
    let mut null_keys: std::collections::HashMap<usize, Arc<Vec<f32>>> =
        std::collections::HashMap::new();
    for &layer in attn_layers
        .iter()
        .filter(|l| selected.iter().any(|c| c.layer == **l))
    {
        let key = &cache.kv_cache.layers[layer].key;
        let want = n * kvd;
        if key.len() < want {
            bail!(
                "layer {layer}: key buffer {} < {want} (t_null={n} × kvd={kvd}) — the cache posture shrank",
                key.len()
            );
        }
        null_keys.insert(layer, Arc::new(key[..want].to_vec()));
    }
    println!(
        "# null snapshot: {:.1}s ({} layer buffers, t_null={n})",
        t0.elapsed().as_secs_f64(),
        null_keys.len()
    );

    // ── the sweep: per probed layer, observe once, then cells ──
    let mut pass_wall_avg = pass_wall.max(1e-6);
    let mut passes_done = 2u64;
    let mut results: Vec<CellResult> = Vec::new();
    let mut layer_groups: Vec<usize> = selected.iter().map(|c| c.layer).collect();
    layer_groups.sort_unstable();
    layer_groups.dedup();
    for &layer in &layer_groups {
        // Observe pass (the E_real base; every band of the posture in one pass).
        let t0 = Instant::now();
        let (_, _, arm) = run_read_pass(
            &weights,
            &census_config,
            &read,
            &golds,
            &rope_freq,
            &mut cache,
            &mut scratch,
            Some(HisaCensusArm::observe(layer, bands.clone())),
        )
        .map_err(anyhow::Error::msg)?;
        let arm = arm.expect("observe arm vanished");
        println!(
            "# layer {layer} observe: {:.1}s",
            t0.elapsed().as_secs_f64()
        );
        let e_observe = |band: (usize, usize)| -> (f64, u64) {
            arm.bands
                .iter()
                .find(|c| c.band == band)
                .map(|c| (c.energy, c.count))
                .unwrap_or_else(|| panic!("band {band:?} missing from the observe accumulator"))
        };

        // Cell passes, in canonical order.
        for cell in selected.iter().filter(|c| c.layer == layer) {
            if committed_specs.contains(cell) {
                continue;
            }
            let (e_obs, c_obs) = e_observe(cell.band);
            let null_keys = null_keys
                .get(&layer)
                .ok_or_else(|| anyhow::anyhow!("layer {layer}: no null snapshot"))?
                .clone();

            // Remove (suppression posture).
            let t0 = Instant::now();
            let (nll_rem, top1_rem, arm) = run_read_pass(
                &weights,
                &census_config,
                &read,
                &golds,
                &rope_freq,
                &mut cache,
                &mut scratch,
                Some(HisaCensusArm::remove(layer, cell.chunk, cell.band)),
            )
            .map_err(anyhow::Error::msg)?;
            let arm_remove = arm.expect("remove arm vanished");
            let (e_rem, c_rem) = (arm_remove.energy, arm_remove.count);
            drop(arm_remove);

            // Replace (E_null noise floor).
            let (nll_rep, top1_rep, arm) = run_read_pass(
                &weights,
                &census_config,
                &read,
                &golds,
                &rope_freq,
                &mut cache,
                &mut scratch,
                Some(
                    HisaCensusArm::replace(layer, cell.chunk, cell.band, null_keys, n, kvd)
                        .map_err(anyhow::Error::msg)?,
                ),
            )
            .map_err(anyhow::Error::msg)?;
            let arm = arm.expect("replace arm vanished");
            let wall = t0.elapsed().as_secs_f64();
            passes_done += 2;
            pass_wall_avg =
                (pass_wall_avg * (passes_done - 2) as f64 + wall / 2.0) / passes_done as f64;

            let result = CellResult {
                spec: *cell,
                e_observe: e_obs,
                count_observe: c_obs,
                e_remove: e_rem,
                count_remove: c_rem,
                e_replace: arm.energy,
                count_replace: arm.count,
                swapped_skipped: arm.swapped_skipped,
                base_nll_mean,
                nll_remove_mean: mean(&nll_rem),
                nll_replace_mean: mean(&nll_rep),
                d_nll_remove: mean(&nll_rem) - base_nll_mean,
                d_nll_replace: mean(&nll_rep) - base_nll_mean,
                flips_remove: flip_count(&base_top1, &top1_rem),
                flips_replace: flip_count(&base_top1, &top1_rep),
                steps: n as u64,
            };
            if let Some(led) = &ledger {
                led.commit(&result).map_err(anyhow::Error::msg)?;
            }
            println!(
                "# cell layer={} chunk=({},{}) band=({},{}) E_rem/E_obs={:.3} E_rep/E_obs={:.3} dNLL_rem={:.3e} dNLL_rep={:.3e} flips={}/{} ({:.1}s, {}/{} pending after)",
                result.spec.layer,
                result.spec.chunk.0,
                result.spec.chunk.1,
                result.spec.band.0,
                result.spec.band.1,
                safe_ratio(result.e_remove, result.e_observe),
                safe_ratio(result.e_replace, result.e_observe),
                result.d_nll_remove,
                result.d_nll_replace,
                result.flips_remove,
                result.flips_replace,
                wall,
                committed_specs.len() + results.len() + 1,
                selected.len()
            );
            results.push(result);
        }
    }

    // ── the report ──
    let canonical = canonical_report_bytes(&identity, &results);
    let skipped_total: u64 = results.iter().map(|r| r.swapped_skipped).sum();
    if skipped_total > 0 {
        eprintln!(
            "[hisa_census] DEFECT: swapped_skipped total = {skipped_total} (>0) — the driver's t_null guarantee broke; do NOT read this report's E_replace column"
        );
    }
    let report = render_report(&canonical);
    match &args.out {
        Some(path) => {
            std::fs::write(path, &report)
                .with_context(|| format!("write report {}", path.display()))?;
            println!("# report: {} ({} bytes)", path.display(), report.len());
        }
        None => {
            print!("{}", String::from_utf8_lossy(&report));
        }
    }
    println!(
        "# done: {} cells this run | per-pass wall {:.1}s | digest {}",
        results.len(),
        pass_wall_avg,
        blake3::hash(&canonical).to_hex()
    );
    Ok(())
}

fn safe_ratio(num: f64, den: f64) -> f64 {
    if den > 0.0 && num.is_finite() {
        num / den
    } else {
        f64::NAN
    }
}

/// Distinct probed layers in the selected table — the observe-pass count
/// divisor for the ETA (one observe pass per layer, not per cell).
fn layer_estimate(selected: &[CellSpec]) -> usize {
    let mut g: Vec<usize> = selected.iter().map(|c| c.layer).collect();
    g.sort_unstable();
    g.dedup();
    g.len().max(1)
}
