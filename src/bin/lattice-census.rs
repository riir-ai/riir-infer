//! lattice-census — riir-train plan 503 T0.1's census CLI: one ternary pass
//! per checkpoint over the first N tokens, per-head GDN state geometry
//! (entropy effective rank / stable rank / query-alignment gain) snapshotted
//! at the pinned capture positions, one fixed-precision BLAKE3-pinned report.
//!
//! ```bash
//! cargo run --release --features lattice_census[,bonsai2_hadamard] \
//!     --bin lattice-census -- \
//!     ../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf \
//!     --n 512 --cap-every 64 --out .benchmarks/lattice_census_report.json
//! # T0.2 probe arm (fixed-coeff FiLM over the GDN layer input):
//! #   ... --film-seed 0x50E1C5   (census re-runs under the injection;
//! #   the shift vs the unarmed run is the probe's read-out)
//! ```
//!
//! Determinism (G1): the report's `blake3` digest is taken over the
//! canonical fixed-precision payload, so a re-run over the same checkpoint +
//! corpus + window reproduces it byte-for-byte (the Issue-940 law: f64
//! sequential ascending folds, fixed Jacobi sweep order, pinned tie rules).
//!
//! Posture: REPORT-ONLY — the census reads a frozen checkpoint and writes a
//! JSON report; it never gates, never tunes, never enters a serving path.
//! The Phase-2 lattice arm (plan 503) is judged against this baseline later;
//! the census itself claims nothing.
//!
//! ⚠ Query-alignment rows: the transient `q_normed` capture arm is the
//! compile-window wiring step (the module doc carries the note); until it
//! lands the report carries `g1=na` rows and `query_armed=false` —
//! self-describing, never silently absent.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};

use riir_infer_core::corpus_text::load_corpus_text;
use riir_infer_core::deltanet::effective_rotary_dim;
use riir_infer_core::deltanet::lattice_census::{
    FilmFilter, FilmMatvecHook, LatticeCensusReport, MAX_CAPTURE_POSITIONS, census_state_snapshot,
};
use riir_infer_core::deltanet::{
    forward_qwen_deltanet_ternary, forward_qwen_deltanet_ternary_with_hook,
};
use riir_infer_core::gguf_loader::{GgmlType, GgufFile, load_qwen_deltanet_ternary_weights_gguf};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_core::types::DeltaNetLayerType;

const DEFAULT_MODEL: &str = "../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf";
const DEFAULT_CORPUS: &str = "../riir-train/data/chat_probe";
const DEFAULT_N: usize = 512;
const DEFAULT_CAP_EVERY: usize = 64;

struct Args {
    model: PathBuf,
    corpus: PathBuf,
    /// Total tokens to consume (one sequence, the corpus head — the v1
    /// posture; the paper's 1024-seq posture is the pinned v2 extension).
    n: usize,
    /// Capture every this many positions (position 1 always captured).
    cap_every: usize,
    /// T0.2: run the census under the fixed-coeff FiLM injection keyed on
    /// this seed (absent = the unarmed baseline run).
    film_seed: Option<u64>,
    out: Option<PathBuf>,
}

fn parse_args() -> anyhow::Result<Args> {
    let mut a = Args {
        model: PathBuf::from(DEFAULT_MODEL),
        corpus: PathBuf::from(DEFAULT_CORPUS),
        n: DEFAULT_N,
        cap_every: DEFAULT_CAP_EVERY,
        film_seed: None,
        out: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(f) = it.next() {
        match f.as_str() {
            "--n" => a.n = it.next().expect("--n value").parse().expect("n usize"),
            "--cap-every" => {
                a.cap_every = it
                    .next()
                    .expect("--cap-every value")
                    .parse()
                    .expect("cap-every usize")
            }
            "--film-seed" => {
                let v = it.next().expect("--film-seed value");
                let v = v.trim_start_matches("0x").trim_start_matches("0X");
                a.film_seed = Some(u64::from_str_radix(v, 16).expect("film-seed hex u64"));
            }
            "--corpus" => a.corpus = PathBuf::from(it.next().expect("--corpus value")),
            "--out" => a.out = Some(PathBuf::from(it.next().expect("--out value"))),
            other => {
                if !other.starts_with('-') {
                    a.model = PathBuf::from(other);
                } else {
                    bail!("unknown flag {other}");
                }
            }
        }
    }
    Ok(a)
}

fn main() -> Result<()> {
    let args = parse_args()?;

    // The ternary dispatch FIRST (the sink_audit lesson: the dense loader
    // mis-shapes a PQ2_0 ternary file — the dtype is the discriminator).
    let probe =
        GgufFile::open(&args.model).with_context(|| format!("open {}", args.model.display()))?;
    let is_ternary = probe
        .tensor_info("token_embd.weight")
        .map(|info| matches!(info.ggml_type, GgmlType::Q2_0 | GgmlType::PTQ1_0))
        .unwrap_or(false);
    if !is_ternary {
        bail!(
            "lattice-census v1 drives the TERNARY body only ({} is not Q2_0/PTQ1_0) — \
             the dense twin is the compile-window extension",
            args.model.display()
        );
    }

    let corpus_text = load_corpus_text(&args.corpus)
        .with_context(|| format!("load corpus {}", args.corpus.display()))?;
    let tokens = {
        let tok = BpeTokenizer::from_gguf(&probe).context("gpt2 BPE tokenizer from gguf")?;
        tok.encode(&corpus_text)
    };
    if tokens.len() < args.n {
        bail!(
            "corpus encodes {} tokens < the census window {} — shorten --n or grow the corpus",
            tokens.len(),
            args.n
        );
    }

    // Capture schedule: position 1 always, then every --cap-every, capped.
    let mut positions: Vec<usize> = Vec::new();
    let mut p = 1usize;
    while p <= args.n && positions.len() < MAX_CAPTURE_POSITIONS {
        positions.push(p);
        p += args.cap_every;
    }
    if positions.len() >= MAX_CAPTURE_POSITIONS && p <= args.n {
        bail!(
            "capture schedule exceeds the snapshot budget {} — raise --cap-every (budget: \
             ~n_gdn_layers × 1 MiB per position)",
            MAX_CAPTURE_POSITIONS
        );
    }
    println!(
        "# census: {} positions (pos 1 + every {}) over {} tokens | film={:?}",
        positions.len(),
        args.cap_every,
        args.n,
        args.film_seed
    );

    let t0 = std::time::Instant::now();
    let (config, weights) = load_qwen_deltanet_ternary_weights_gguf(&args.model)
        .with_context(|| format!("load (ternary) {}", args.model.display()))?;
    let n_gdn = weights
        .layer_types
        .iter()
        .filter(|&&t| t == DeltaNetLayerType::DeltaNet)
        .count();
    let folded = weights.rotation.is_some();
    println!(
        "# model: qwen35 hybrid ternary | layers={} gdn_layers={n_gdn} n_embd={} | folded={folded} | load {:.1}s",
        config.n_layer,
        config.n_embd,
        t0.elapsed().as_secs_f32()
    );
    if folded {
        // The folded (rotated-QKVZ) layers bypass the fused input-proj hook by
        // construction — the T0.2 seam on a folded checkpoint is the
        // matvec-wrap hook below, keyed on the armed layer's resident qkv/z
        // planes, with the Rβ equivariance form.
        println!(
            "# NOTE: folded checkpoint — T0.2 arms via the matvec-wrap hook \
             (γ⊙x_rot + Rβ equivariance form; input-proj fusion is bypassed \
             on rotated layers by construction)"
        );
    }

    let n_v_heads = config.deltanet_linear_n_value_heads;
    let d_k = config.deltanet_linear_head_dim;
    let d_v = config.deltanet_linear_head_dim;

    // The FiLM probe posture: arm ONE GDN layer (the first, deterministic) —
    // the filter keyed on that layer's resident qkv/z plane addresses.
    let arm_layer_idx = weights
        .layer_types
        .iter()
        .enumerate()
        .find(|(_, t)| **t == DeltaNetLayerType::DeltaNet)
        .map(|(i, _)| i)
        .expect("checkpoint has at least one GDN layer (T0.2 probe posture)");
    let film_hook = args.film_seed.map(|seed| {
        let filter = FilmFilter::fixed_coeff(config.n_embd, arm_layer_idx, seed);
        let signs: Vec<i8> = weights
            .rotation
            .as_ref()
            .and_then(|r| r.signs_for_width(config.n_embd).map(|s| s.to_vec()))
            .unwrap_or_else(|| vec![1i8; config.n_embd]);
        let hook = FilmMatvecHook::arm_layer(
            filter,
            &weights.layers[arm_layer_idx].in_proj_qkv,
            &weights.layers[arm_layer_idx].in_proj_z,
            &signs,
        );
        println!(
            "# film armed: layer {} (first GDN) seed=0x{:x} folded={folded} | γ±{:.3} β±{:.3}",
            arm_layer_idx,
            seed,
            hook.filter
                .gamma
                .iter()
                .map(|g| (g - 1.0).abs())
                .fold(0.0f32, f32::max),
            hook.filter
                .beta
                .iter()
                .map(|b| b.abs())
                .fold(0.0f32, f32::max),
        );
        hook
    });

    let rope_freq = RopeFreqTable::new(config.rope_theta, effective_rotary_dim(&config));
    let mut cache = riir_infer_core::deltanet::HybridCache::with_layer_types_bounded(
        &config,
        &weights.layer_types,
        args.n,
    );
    let mut scratch = riir_infer_core::deltanet::HybridForwardScratch::new(&config);
    let mut logits = vec![0.0f32; config.vocab_size];

    let mut points = Vec::with_capacity(positions.len());
    let t1 = std::time::Instant::now();

    // T0.2 engagement proof (rides the driver): a fresh UNARMED pos-1 forward
    // vs a fresh ARMED pos-1 forward on the armed layer's state, compared
    // bit-wise (to_bits — NaN-proof). A non-identity filter that leaves the
    // armed layer's pos-1 state byte-identical = the seam never engaged →
    // refuse loud; a fake-shift report is worse than no report.
    if let Some(hook) = film_hook.as_ref() {
        let mut c_un = riir_infer_core::deltanet::HybridCache::with_layer_types_bounded(
            &config,
            &weights.layer_types,
            1,
        );
        let mut s_un = riir_infer_core::deltanet::HybridForwardScratch::new(&config);
        let mut l_un = vec![0.0f32; config.vocab_size];
        forward_qwen_deltanet_ternary(
            &mut l_un, &weights, &mut c_un, tokens[0], 0, &config, &mut s_un, &rope_freq,
        );
        let mut c_ar = riir_infer_core::deltanet::HybridCache::with_layer_types_bounded(
            &config,
            &weights.layer_types,
            1,
        );
        let mut s_ar = riir_infer_core::deltanet::HybridForwardScratch::new(&config);
        let mut l_ar = vec![0.0f32; config.vocab_size];
        forward_qwen_deltanet_ternary_with_hook(
            &mut l_ar,
            &weights,
            &mut c_ar,
            tokens[0],
            0,
            &config,
            &mut s_ar,
            &rope_freq,
            None,
            Some(hook),
            None,
            None,
        );
        let un = &c_un.deltanet_state.recurrent_states[arm_layer_idx];
        let ar = &c_ar.deltanet_state.recurrent_states[arm_layer_idx];
        let same = un.len() == ar.len()
            && un
                .iter()
                .zip(ar.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits());
        if same {
            bail!(
                "T0.2 engagement proof FAILED: armed layer {} pos-1 state byte-identical to \
                 unarmed — the matvec-wrap seam did not engage (address keying or filter arms)",
                arm_layer_idx
            );
        }
        println!(
            "# engagement proof: armed layer {} pos-1 state differs from unarmed (seam engaged)",
            arm_layer_idx
        );
    }

    for (i, &token) in tokens.iter().take(args.n).enumerate() {
        let pos = i;
        // T0.2 injection: the armed run drives the matvec-wrap hook (the
        // single armed layer's qkv/z FiLM); the unarmed run passes hook=None
        // and is byte-identical to the incumbent forward.
        match film_hook.as_ref() {
            Some(hook) => forward_qwen_deltanet_ternary_with_hook(
                &mut logits,
                &weights,
                &mut cache,
                token,
                pos,
                &config,
                &mut scratch,
                &rope_freq,
                None,
                Some(hook),
                None,
                None,
            ),
            None => forward_qwen_deltanet_ternary(
                &mut logits,
                &weights,
                &mut cache,
                token,
                pos,
                &config,
                &mut scratch,
                &rope_freq,
            ),
        };
        if positions.binary_search(&pos.wrapping_add(1)).is_ok() {
            // Snapshot AFTER position `pos`'s update = the state AT
            // `pos+1` — the capture-point position label.
            let snap_pos = pos + 1;
            let mut geometry = Vec::with_capacity(n_gdn);
            for layer_idx in 0..config.n_layer {
                if weights.layer_types[layer_idx] != DeltaNetLayerType::DeltaNet {
                    continue;
                }
                let state = &cache.deltanet_state.recurrent_states[layer_idx];
                geometry.push(census_state_snapshot(state, None, n_v_heads, d_k, d_v)?);
            }
            points.push(riir_infer_core::deltanet::lattice_census::CapturePoint {
                pos: snap_pos,
                geometry,
            });
        }
    }
    println!(
        "# census pass: {:.1}s (ternary body, {} snapshots)",
        t1.elapsed().as_secs_f32(),
        points.len()
    );

    let report = LatticeCensusReport {
        positions: points.iter().map(|p| p.pos).collect(),
        points,
        shape: (n_gdn, n_v_heads, d_k, d_v),
        query_armed: false,
    };
    let payload = report.canonical_payload();
    let digest = report.digest();

    // Fixed-precision summary (the readable top line; the payload is the pin).
    let means = report.mean_r_ent_per_layer();
    let mean_all = if means.is_empty() {
        0.0
    } else {
        means.iter().sum::<f64>() / means.len() as f64
    };
    println!("# mean r_ent (all layers × points): {mean_all:.4} | digest={digest}");

    match args.out {
        Some(out) => {
            std::fs::write(&out, &payload).with_context(|| format!("write {}", out.display()))?;
            println!(
                "# report → {} ({} bytes, blake3={digest})",
                out.display(),
                payload.len()
            );
        }
        None => {
            print!("{payload}");
            println!("# blake3={digest}");
        }
    }
    Ok(())
}
