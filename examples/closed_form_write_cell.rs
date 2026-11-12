//! closed_form_write_cell — the Issue-1189 T3 harness: the closed-form
//! fact-write cell end to end on the REAL dense qwen35-hybrid (Cell A:
//! `Qwen3.5-0.8B-Base-Q8_0`), CPU-deterministic.
//!
//! Phases:
//!   A. baseline probes — fresh binding, no overlays
//!   B. [`construct`] — the katgpt-core closed-form solve + refinement
//!      (the adapter installs pairs as it goes; the envelope seals)
//!   C. armed probes — the same probe set under the installed pairs
//!   D. demote identity — empty-overlay every edited layer; the probes must
//!      be byte-identical to baseline (logprob bits + generation tokens)
//!   E. sidecar — `manifest.json` + raw f32 pair files (the consumer's
//!      loadable artifact, layer-fold order, per-pair provenance)
//!   F. reload-and-remount identity — fresh model load, base-digest check,
//!      mount(sidecar pairs); probes must be byte-identical to armed
//!
//! The gate ADJUDICATIONS (G1 wording-recall floors, G1b look-alike
//! zero-change, G2 cross-process digest equality, G3 league retention)
//! stay T4's — this harness REPORTS those tables and only ASSERTS the two
//! identity laws (D, F), which are correctness, not quality.
//!
//! ```bash
//! cargo run --release --features closed_form_write \
//!     --example closed_form_write_cell -- \
//!     --gguf /Users/katopz/git/riir-train/data/Qwen3.5-0.8B-Base-Q8_0.gguf \
//!     --out .closed_form_write_cell/
//! # facts default to one inline fixture (France → Paris); --facts <json>
//! # replaces the whole set (schema: FactsDoc below).
//! ```
//!
//! Timings printed per phase are wall-clock on a possibly-loaded box — a
//! correctness harness, never a perf claim; no bench number leaves here.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use katgpt_core::closed_form_weight_write::{
    FactSpec, FactWriteConfig, FactWriteModel, ProtectSpec, construct,
};
use katgpt_core::quant_error_lora::QuantErrorLora;
use riir_infer_core::closed_form_write::QwenFactWriteModel;
use riir_infer_core::gguf_loader::{GgufFile, load_qwen_deltanet_weights_gguf};
use riir_infer_core::tokenizer::BpeTokenizer;

/// The default Cell A model (box-stable on amo + bmo; `--gguf` overrides).
const DEFAULT_GGUF: &str = "/Users/katopz/git/riir-train/data/Qwen3.5-0.8B-Base-Q8_0.gguf";

// ── fact-sheet input ────────────────────────────────────────────────────

/// `--facts` JSON schema. `answer` is the completion the fact pins (the
/// solve appends EOS as the stop token — the answer contract is content +
/// stop). `heldout_asks` and `lookalike_asks` are eval-only probes; they
/// never enter the solve.
#[derive(serde::Deserialize)]
struct FactsDoc {
    facts: Vec<FactDoc>,
    #[serde(default)]
    control_asks: Vec<String>,
    #[serde(default)]
    lookalike_asks: Vec<String>,
}

#[derive(serde::Deserialize)]
struct FactDoc {
    train_asks: Vec<String>,
    answer: String,
    #[serde(default)]
    contrast_asks: Vec<String>,
    #[serde(default)]
    heldout_asks: Vec<String>,
}

/// The inline default: one fact, its contrast family, one held-out wording,
/// three general controls. Real knowledge, base-model completion form.
fn default_facts_doc() -> FactsDoc {
    FactsDoc {
        facts: vec![FactDoc {
            train_asks: vec![
                "The capital of France is".into(),
                "The capital city of France is called".into(),
            ],
            answer: " Paris".into(),
            contrast_asks: vec![
                "The capital of Italy is".into(),
                "The capital of Germany is".into(),
                "The capital of Spain is".into(),
            ],
            heldout_asks: vec!["France's capital city is".into()],
        }],
        control_asks: vec![
            "The weather this morning was".into(),
            "A good book can be".into(),
            "The train arrived at the station and".into(),
        ],
        lookalike_asks: vec!["The capital of Italy is".into()],
    }
}

// ── probes ──────────────────────────────────────────────────────────────

struct Probe {
    label: String,
    ask: Vec<u32>,
    /// Content tokens whose teacher-forced logprob is probed (empty =
    /// generation-only probe).
    answer: Vec<u32>,
}

struct Reading {
    label: String,
    logprob: Option<f64>,
    gen_toks: Vec<u32>,
    gen_text: String,
}

/// One reading pass over every probe against the model's CURRENT overlay
/// state (each call = fresh session state — the adapter's statelessness law).
fn read_probes(
    m: &mut QwenFactWriteModel<'_>,
    probes: &[Probe],
    max_new: usize,
    tok: &BpeTokenizer,
) -> Vec<Reading> {
    probes
        .iter()
        .map(|p| {
            let logprob = (!p.answer.is_empty()).then(|| {
                let mut seq = p.ask.clone();
                seq.extend_from_slice(&p.answer);
                let mut sum = 0.0f64;
                for pos in p.ask.len()..seq.len() {
                    sum += m.token_logprob(&seq, pos);
                }
                sum
            });
            let toks = m.free_generate(&p.ask, max_new, None);
            let us: Vec<usize> = toks.iter().map(|&t| t as usize).collect();
            Reading {
                label: p.label.clone(),
                logprob,
                gen_text: tok.decode(&us),
                gen_toks: toks,
            }
        })
        .collect()
}

/// The identity law: same labels, same logprob BITS, same generation tokens.
fn assert_readings_identical(phase: &str, want: &[Reading], got: &[Reading]) {
    assert_eq!(want.len(), got.len(), "{phase}: reading count drifted");
    for (w, g) in want.iter().zip(got.iter()) {
        assert_eq!(w.label, g.label, "{phase}: probe order drifted");
        match (w.logprob, g.logprob) {
            (Some(a), Some(b)) => assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{phase}[{}]: logprob moved: {a} vs {b}",
                w.label
            ),
            (None, None) => {}
            _ => panic!("{phase}[{}]: logprob presence drifted", w.label),
        }
        assert_eq!(
            w.gen_toks, g.gen_toks,
            "{phase}[{}]: generation tokens moved",
            w.label
        );
    }
}

// ── sidecar manifest ────────────────────────────────────────────────────

#[derive(serde::Serialize, serde::Deserialize)]
struct ConfigEcho {
    lambda: f64,
    contrast_weight: f64,
    earlier_fact_weight: f64,
    control_weight: f64,
    mu: f64,
    max_rel_norm: f64,
    rank: usize,
    max_rounds: usize,
    boost_cap: f64,
    boost_ratio_cap: f64,
    target_floor_nats: f64,
    shortfall_cap_nats: f64,
    gain_lo: f64,
    gain_hi: f64,
}

impl From<FactWriteConfig> for ConfigEcho {
    fn from(c: FactWriteConfig) -> Self {
        Self {
            lambda: c.lambda,
            contrast_weight: c.contrast_weight,
            earlier_fact_weight: c.earlier_fact_weight,
            control_weight: c.control_weight,
            mu: c.mu,
            max_rel_norm: c.max_rel_norm,
            rank: c.rank,
            max_rounds: c.max_rounds,
            boost_cap: c.boost_cap,
            boost_ratio_cap: c.boost_ratio_cap,
            target_floor_nats: c.target_floor_nats,
            shortfall_cap_nats: c.shortfall_cap_nats,
            gain_lo: c.gain_lo,
            gain_hi: c.gain_hi,
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct FactRowEcho {
    fact_index: usize,
    ask_digest_hex: String,
    answer: Vec<u32>,
    contrast_digest_hex: String,
    layers: Vec<usize>,
    rank: usize,
    dropped_energy: f64,
    fit: f64,
    rounds: usize,
    miss: bool,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct PairMeta {
    fact: usize,
    layer: usize,
    rank: usize,
    in_dim: usize,
    out_dim: usize,
    alpha: f32,
    /// Raw little-endian f32 file under the out dir: `a` rows then `b` rows.
    file: String,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ProbeEcho {
    label: String,
    logprob: Option<f64>,
    gen_toks: Vec<u32>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Manifest {
    format_version: u32,
    lane: String,
    model: String,
    base_digest_hex: String,
    envelope_digest_hex: String,
    sealed: bool,
    constructed_not_armed: bool,
    edited_layers: Vec<usize>,
    config: ConfigEcho,
    facts: Vec<FactRowEcho>,
    /// Layer-fold order (the armed table's canonical order): a layer's pairs
    /// adjacent, fact-asc within. The reload path mounts this order as-is.
    pairs: Vec<PairMeta>,
    probes_armed: Vec<ProbeEcho>,
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn head(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

fn write_pair(path: &Path, p: &QuantErrorLora) -> Result<()> {
    use std::io::Write;
    let mut f =
        std::fs::File::create(path).with_context(|| format!("create {}", path.display()))?;
    for v in &p.a {
        f.write_all(&v.to_le_bytes())?;
    }
    for v in &p.b {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn read_pair(path: &Path, pm: &PairMeta) -> Result<QuantErrorLora> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let need = 4 * (pm.rank * pm.in_dim + pm.out_dim * pm.rank);
    if bytes.len() != need {
        bail!(
            "{}: {} bytes, expected {need} (rank {} in {} out {})",
            path.display(),
            bytes.len(),
            pm.rank,
            pm.in_dim,
            pm.out_dim
        );
    }
    let mut it = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c));
    let a: Vec<f32> = it.by_ref().take(pm.rank * pm.in_dim).collect();
    let b: Vec<f32> = it.collect();
    Ok(QuantErrorLora {
        a,
        b,
        alpha: pm.alpha,
        rank: pm.rank,
        in_dim: pm.in_dim,
        out_dim: pm.out_dim,
    })
}

// ── args ────────────────────────────────────────────────────────────────

struct Args {
    gguf: PathBuf,
    out: PathBuf,
    facts: Option<PathBuf>,
    layers: Option<String>,
    rank: Option<usize>,
    rounds: Option<usize>,
    max_new: usize,
}

impl Args {
    fn parse() -> Result<Args> {
        let mut a = Args {
            gguf: PathBuf::from(DEFAULT_GGUF),
            out: PathBuf::from(".closed_form_write_cell"),
            facts: None,
            layers: None,
            rank: None,
            rounds: None,
            max_new: 12,
        };
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let mut val = || it.next().with_context(|| format!("{flag} needs a value"));
            match flag.as_str() {
                "--gguf" => a.gguf = PathBuf::from(val()?),
                "--out" => a.out = PathBuf::from(val()?),
                "--facts" => a.facts = Some(PathBuf::from(val()?)),
                "--layers" => a.layers = Some(val()?),
                "--rank" => a.rank = Some(val()?.parse().context("--rank parse")?),
                "--rounds" => a.rounds = Some(val()?.parse().context("--rounds parse")?),
                "--max-new" => a.max_new = val()?.parse().context("--max-new parse")?,
                other => bail!(
                    "unknown flag {other} (flags: --gguf --out --facts --layers --rank --rounds --max-new)"
                ),
            }
        }
        Ok(a)
    }
}

fn parse_layers(s: &str) -> Result<Vec<usize>> {
    let mut ls: Vec<usize> = s
        .split(',')
        .map(|t| t.trim().parse().context("layer id parse"))
        .collect::<Result<_>>()?;
    ls.sort_unstable();
    ls.dedup();
    Ok(ls)
}

// ── main ────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse()?;

    let t_load = Instant::now();
    let (config, mut weights) = load_qwen_deltanet_weights_gguf(&args.gguf)
        .with_context(|| format!("load {}", args.gguf.display()))?;
    let tok = {
        let gguf = GgufFile::open(&args.gguf).context("re-open gguf for tokenizer")?;
        BpeTokenizer::from_gguf(&gguf).context("gpt2 BPE tokenizer from gguf")?
    };
    println!(
        "# model: qwen35 hybrid | layers={} n_embd={} vocab={} | load {:.1}s",
        config.n_layer,
        config.n_embd,
        config.vocab_size,
        t_load.elapsed().as_secs_f32()
    );
    let base_digest = riir_infer_core::closed_form_write::weights_digest(&weights);

    let doc = match &args.facts {
        Some(p) => {
            let raw =
                std::fs::read_to_string(p).with_context(|| format!("read {}", p.display()))?;
            serde_json::from_str(&raw).with_context(|| format!("parse {}", p.display()))?
        }
        None => default_facts_doc(),
    };

    let eos = tok.eos_id() as u32;
    let enc = |s: &str| -> Vec<u32> {
        tok.encode(s)
            .into_iter()
            .map(|t| t as u32)
            .collect::<Vec<u32>>()
    };
    let answer_content: Vec<Vec<u32>> = doc.facts.iter().map(|f| enc(&f.answer)).collect();
    let facts: Vec<FactSpec> = doc
        .facts
        .iter()
        .map(|f| FactSpec {
            train_asks: f.train_asks.iter().map(|s| enc(s)).collect(),
            answer: {
                let mut a = enc(&f.answer);
                a.push(eos);
                a
            },
            contrast_asks: f.contrast_asks.iter().map(|s| enc(s)).collect(),
        })
        .collect();
    let controls = if doc.control_asks.is_empty() {
        default_facts_doc().control_asks
    } else {
        doc.control_asks.clone()
    };
    let protect = ProtectSpec {
        control_asks: controls.iter().map(|s| enc(s)).collect(),
    };

    // Probe set: per-fact held-out wordings (logprob + generation), the
    // look-alike family, and the general controls (generation only).
    let mut probes: Vec<Probe> = Vec::new();
    for (i, f) in doc.facts.iter().enumerate() {
        for (k, h) in f.heldout_asks.iter().enumerate() {
            probes.push(Probe {
                label: format!("heldout[f{i}]#{k}"),
                ask: enc(h),
                answer: answer_content[i].clone(),
            });
        }
    }
    for (k, l) in doc.lookalike_asks.iter().enumerate() {
        probes.push(Probe {
            label: format!("lookalike#{k}"),
            ask: enc(l),
            answer: Vec::new(),
        });
    }
    for (k, c) in controls.iter().enumerate() {
        probes.push(Probe {
            label: format!("control#{k}"),
            ask: enc(c),
            answer: Vec::new(),
        });
    }
    assert!(
        !probes.is_empty(),
        "no probes — add heldout/lookalike/control asks"
    );

    let edited = match &args.layers {
        Some(s) => parse_layers(s)?,
        None => {
            let start = (config.n_layer / 2).saturating_sub(2);
            let end = (start + 4).min(config.n_layer);
            (start..end).collect()
        }
    };
    if edited.is_empty() || edited.windows(2).any(|w| w[0] >= w[1]) {
        bail!("edited layers must be non-empty and strictly ascending");
    }
    if edited.last().copied().unwrap() >= config.n_layer {
        bail!("edited layer past the model (n_layer={})", config.n_layer);
    }
    println!("# edited layers: {edited:?}");

    let solve_cfg = FactWriteConfig {
        rank: args.rank.unwrap_or_else(|| FactWriteConfig::default().rank),
        max_rounds: args
            .rounds
            .unwrap_or_else(|| FactWriteConfig::default().max_rounds),
        ..FactWriteConfig::default()
    };

    // ── phase A: baseline ──
    println!("# ==== phase A: baseline probes (no overlays) ====");
    let t = Instant::now();
    let baseline: Vec<Reading> = {
        let mut m = QwenFactWriteModel::bind(&mut weights, &config, edited.clone());
        read_probes(&mut m, &probes, args.max_new, &tok)
    };
    println!("# {:.1}s", t.elapsed().as_secs_f32());

    // ── phase B: construct ──
    println!("# ==== phase B: construct (closed-form solve + refinement) ====");
    let t = Instant::now();
    let outcome = {
        let mut m = QwenFactWriteModel::bind(&mut weights, &config, edited.clone());
        construct(&mut m, &solve_cfg, &facts, &protect)
    };
    println!("# {:.1}s", t.elapsed().as_secs_f32());
    for (i, row) in outcome.envelope.rows().iter().enumerate() {
        println!(
            "# fact[{i}]: fit={:.3} rounds={} miss={} dropped_energy={:.3e} gain={:.2} layers={:?} rank={}",
            row.fit,
            row.rounds,
            row.miss,
            row.dropped_energy,
            outcome.gain[i],
            row.layers,
            row.rank
        );
    }
    println!(
        "# envelope: sealed={} digest={}",
        outcome.envelope.is_sealed(),
        hex(&outcome.envelope.digest())
    );

    // ── phase C: armed probes ──
    println!("# ==== phase C: armed probes (pairs installed) ====");
    let t = Instant::now();
    let armed: Vec<Reading> = {
        let mut m = QwenFactWriteModel::bind(&mut weights, &config, edited.clone());
        read_probes(&mut m, &probes, args.max_new, &tok)
    };
    println!("# {:.1}s", t.elapsed().as_secs_f32());
    for (b, a) in baseline.iter().zip(armed.iter()) {
        match (b.logprob, a.logprob) {
            (Some(bl), Some(al)) => println!(
                "# {:24} logprob {:+.4} → {:+.4} (Δ {:+.4})",
                b.label,
                bl,
                al,
                al - bl
            ),
            _ => println!("# {:24}", b.label),
        }
        println!(
            "# {:24} gen {:?} → {:?}",
            "",
            head(&b.gen_text, 48),
            head(&a.gen_text, 48)
        );
    }
    // The G1 pre-read (REPORT; the floor adjudication is T4's): does the
    // armed generation contain the fact's answer on held-out wordings?
    for (i, f) in doc.facts.iter().enumerate() {
        let ans = f.answer.trim();
        let hits: Vec<bool> = armed
            .iter()
            .filter(|r| r.label.starts_with(&format!("heldout[f{i}]")))
            .map(|r| r.gen_text.contains(ans))
            .collect();
        let n_hit = hits.iter().filter(|h| **h).count();
        println!(
            "# recall-pre[f{i}]: {ans:?} in armed generations: {n_hit}/{}",
            hits.len()
        );
    }

    // ── phase D: demote identity ──
    println!("# ==== phase D: demote identity (empty-overlay every edited layer) ====");
    let demoted: Vec<Reading> = {
        let mut m = QwenFactWriteModel::bind(&mut weights, &config, edited.clone());
        for &l in &edited {
            m.set_layer_overlays(l, &[]);
        }
        read_probes(&mut m, &probes, args.max_new, &tok)
    };
    assert_readings_identical("D demote-vs-baseline", &baseline, &demoted);
    println!(
        "# D demote identity: byte-identical ✓ ({} probes)",
        demoted.len()
    );

    // ── phase E: sidecar ──
    println!("# ==== phase E: sidecar (manifest + pair files) ====");
    std::fs::create_dir_all(&args.out).with_context(|| format!("mkdir {}", args.out.display()))?;
    let layers_per_fact: Vec<Vec<usize>> = outcome
        .envelope
        .rows()
        .iter()
        .map(|r| r.layers.clone())
        .collect();
    let raw_pairs = outcome.envelope.pairs_all();
    let mut metas: Vec<(usize, usize, &QuantErrorLora)> = Vec::new();
    let mut k = 0usize;
    for (f, ls) in layers_per_fact.iter().enumerate() {
        for &l in ls {
            metas.push((f, l, &raw_pairs[k].1));
            k += 1;
        }
    }
    debug_assert_eq!(k, raw_pairs.len(), "pair count vs per-fact layer counts");
    // The armed table's canonical order: stable sort by layer (fact-asc
    // within — the same law `pairs_in_layer_fold_order` pins for mount).
    metas.sort_by_key(|&(_, l, _)| l);

    let mut pair_rows: Vec<PairMeta> = Vec::with_capacity(metas.len());
    for (f, l, p) in &metas {
        let file = format!("f{f}_l{l}.bin");
        write_pair(&args.out.join(&file), p)?;
        pair_rows.push(PairMeta {
            fact: *f,
            layer: *l,
            rank: p.rank,
            in_dim: p.in_dim,
            out_dim: p.out_dim,
            alpha: p.alpha,
            file,
        });
    }
    let manifest = Manifest {
        format_version: 1,
        lane: "closed_form_write/1".into(),
        model: args.gguf.display().to_string(),
        base_digest_hex: hex(&base_digest),
        envelope_digest_hex: hex(&outcome.envelope.digest()),
        sealed: outcome.envelope.is_sealed(),
        constructed_not_armed: outcome.envelope.constructed_not_armed(),
        edited_layers: edited.clone(),
        config: ConfigEcho::from(solve_cfg),
        facts: outcome
            .envelope
            .rows()
            .iter()
            .map(|r| FactRowEcho {
                fact_index: r.fact_index,
                ask_digest_hex: hex(&r.ask_digest),
                answer: r.answer.clone(),
                contrast_digest_hex: hex(&r.contrast_digest),
                layers: r.layers.clone(),
                rank: r.rank,
                dropped_energy: r.dropped_energy,
                fit: r.fit,
                rounds: r.rounds,
                miss: r.miss,
            })
            .collect(),
        probes_armed: armed
            .iter()
            .map(|r| ProbeEcho {
                label: r.label.clone(),
                logprob: r.logprob,
                gen_toks: r.gen_toks.clone(),
            })
            .collect(),
        pairs: pair_rows,
    };
    let manifest_path = args.out.join("manifest.json");
    std::fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).context("manifest serialize")?,
    )
    .with_context(|| format!("write {}", manifest_path.display()))?;
    println!(
        "# sidecar: {} ({} pair files) — digest {}",
        manifest_path.display(),
        manifest.pairs.len(),
        manifest.envelope_digest_hex
    );

    // ── phase F: reload-and-remount identity ──
    println!("# ==== phase F: reload + remount from the sidecar ====");
    let t = Instant::now();
    let read_manifest: Manifest = {
        let raw = std::fs::read_to_string(&manifest_path)
            .with_context(|| format!("read {}", manifest_path.display()))?;
        serde_json::from_str(&raw).context("manifest parse")?
    };
    let (config2, mut weights2) = load_qwen_deltanet_weights_gguf(&args.gguf)
        .with_context(|| format!("reload {}", args.gguf.display()))?;
    let digest2 = riir_infer_core::closed_form_write::weights_digest(&weights2);
    if hex(&digest2) != read_manifest.base_digest_hex {
        bail!(
            "reload digest {} != sidecar base digest {} — refusing to remount",
            hex(&digest2),
            read_manifest.base_digest_hex
        );
    }
    let mut rows: Vec<(usize, QuantErrorLora)> = Vec::with_capacity(read_manifest.pairs.len());
    for pm in &read_manifest.pairs {
        rows.push((pm.layer, read_pair(&args.out.join(&pm.file), pm)?));
    }
    let table = riir_infer_core::closed_form_write::mount(rows);
    weights2.closed_form_overlays = Some(table);
    let armed2: Vec<Reading> = {
        let mut m =
            QwenFactWriteModel::bind(&mut weights2, &config2, read_manifest.edited_layers.clone());
        read_probes(&mut m, &probes, args.max_new, &tok)
    };
    assert_readings_identical("F reload-vs-armed", &armed, &armed2);
    println!(
        "# F reload identity: byte-identical ✓ ({:.1}s incl. re-load)",
        t.elapsed().as_secs_f32()
    );

    println!("# ==== RESULT: PASS — D + F identities green, sidecar sealed ====");
    Ok(())
}
