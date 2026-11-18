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
//! # --paraphrases (plan-635 T3): swaps fact 0's train asks for the authored
//! # 5-paraphrase calibration set and widens the G1 probe set with the same
//! # asks (labels `paraphrase#k`) plus the authored HELD-OUT paraphrase set
//! # (labels `heldpara[f0]#k` — the amended floor's subject); flag-off is
//! # byte-identical Recipe-1. Authored for the DEFAULT doc only — composing
//! # it with --facts refuses loud.
//! # --phase-a-only (plan-635 T3 amendment): the baseline instrument — runs
//! # phase A, prints the amended recall tables + the baseline-selection
//! # gate, and exits before construct. The re-anchor/probe-set evidence run.
//! ```
//!
//! G1 recall metric — AS AMENDED at T3 (plan-635, amendment (i)–(iv)):
//! (i) recall scored as CHANGE from the phase-A baseline per probe;
//! (ii) the floor's subject is a HELD-OUT paraphrase probe set sharing no
//!     ask strings with the training asks, selected so baseline recall
//!     ≤ 1/5 (the baseline-gate line reports it);
//! (iii) a hit only if the answer appears within the first
//!     [`RECALL_FIRST_K`] generated tokens AND the window is not a
//!     repetition run ([`RECALL_DISTINCT_RATIO`] distinct-token floor);
//! (iv) the repetition check reports as its own gate beside G1b.
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
    /// T3 amendment (ii): held-out paraphrase probes for this fact — the
    /// amended floor's subject. Eval-only (never enters the solve); a doc
    /// opting in must keep baseline recall ≤ 1/5 on them (the harness
    /// reports the gate).
    #[serde(default)]
    heldout_para_asks: Vec<String>,
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
            heldout_para_asks: Vec::new(),
        }],
        control_asks: vec![
            "The weather this morning was".into(),
            "A good book can be".into(),
            "The train arrived at the station and".into(),
        ],
        lookalike_asks: vec!["The capital of Italy is".into()],
    }
}

/// Plan-635 T3: the fixed 5-paraphrase calibration set for the default fact
/// (same relation, different surface forms, one hyponym probe) — authored
/// data committed HERE so the cell stays reproducible; never generated text.
/// `--paraphrases` swaps fact 0's train asks for exactly this set and widens
/// the G1 probe set with the same asks; flag-off stays byte-identical
/// Recipe-1 (the disjointness is test-pinned). Entry 4 is the hyponym probe:
/// the relation routes through CITY (a hyponym of the capital-city frame).
const PARAPHRASE_TRAIN_ASKS: [&str; 5] = [
    "France's capital, the city of the Eiffel Tower, is",
    "In western Europe, the capital of France is",
    "The city where France's government sits, its capital, is",
    "The largest French city, the country's capital, is",
    "The capital of the French Republic is",
];

/// Plan-635 T3 amendment (ii): the HELD-OUT paraphrase probe set for the
/// default fact — five surface forms of the same relation that share NO ask
/// strings with the calibration set (or the doc's asks), never contain the
/// answer, and use wording angles the calibration set does not (Louvre,
/// the City-of-Light nickname, a question form, the Élysée seat, a
/// name-frame). Authored data committed HERE; joins the probe set under
/// `--paraphrases` as `heldpara[f0]#k` and is the amended G1 floor's
/// subject. The in-string disjointness and the no-answer-leak law are
/// test-pinned.
const HELD_OUT_PARAPHRASE_PROBES: [&str; 5] = [
    "The French capital, home to the Louvre, is",
    "The City of Light, France's capital, is",
    "Which city is the capital of France? It is",
    "The French city where the Elysee Palace stands is",
    "In the heart of France, its capital city goes by the name of",
];

/// The probe-label prefix shared by the probe builder and the recall-pre-para
/// report's selector — one const so the two sites cannot drift.
const PARAPHRASE_LABEL_PREFIX: &str = "paraphrase#";

/// The calibration ask STRINGS per fact under the flag — pure (tokenization
/// happens later in `main`), so the identity tests need no tokenizer.
/// Flag-off returns exactly the doc's own asks (the Recipe-1 posture);
/// flag-on returns the authored set for fact 0 (the default doc is
/// single-fact under the flag — `main` refuses a `--facts` composition).
fn train_ask_strings(doc: &FactsDoc, paraphrases: bool) -> Vec<Vec<&str>> {
    doc.facts
        .iter()
        .enumerate()
        .map(|(i, f)| {
            if paraphrases && i == 0 {
                PARAPHRASE_TRAIN_ASKS.to_vec()
            } else {
                f.train_asks.iter().map(String::as_str).collect()
            }
        })
        .collect()
}

// ── probes ──────────────────────────────────────────────────────────────

struct Probe {
    label: String,
    ask: Vec<u32>,
    /// Content tokens whose teacher-forced logprob is probed (empty =
    /// generation-only probe).
    answer: Vec<u32>,
}

// ── the amended G1 recall metric (T3 amendment (iii)) ───────────────────

/// A hit scores only if the answer appears within the first k generated
/// tokens. Pin: 8 of the harness's 12-token generation — loose enough for a
/// paraphrase roundabout ("Well, the capital … is Paris"), tight enough to
/// exclude the repetition tail the T3 cell measured (the attractor sat at
/// tokens 3+). Pre-registered with the amendment; never re-tuned after a
/// deciding cell starts.
const RECALL_FIRST_K: usize = 8;

/// The repetition-run floor: a scoring window whose distinct-token ratio
/// (distinct ids / k) falls below this is a repetition run and scores NO
/// hit — even when the answer's substring rides the repetition (the T3
/// attractor `" the capital Paris Paris Paris…"` measured 3/8 = 0.375,
/// below the floor; a clean generation measures ≥ 6/8). Pinned at 0.5.
const RECALL_DISTINCT_RATIO: f64 = 0.5;

/// Distinct-token ratio of a token window — the repetition-run detector
/// (amendment (iv)). Pure, so the identity tests need no tokenizer.
fn distinct_token_ratio(window: &[u32]) -> f64 {
    if window.is_empty() {
        return 0.0;
    }
    let distinct: std::collections::HashSet<u32> = window.iter().copied().collect();
    distinct.len() as f64 / window.len() as f64
}

/// The amended hit rule: answer in the first [`RECALL_FIRST_K`] tokens AND
/// the window not a repetition run.
fn recall_hit(gen_toks: &[u32], answer: &str, tok: &BpeTokenizer) -> bool {
    let k = gen_toks.len().min(RECALL_FIRST_K);
    if k == 0 {
        return false;
    }
    let window = &gen_toks[..k];
    if distinct_token_ratio(window) < RECALL_DISTINCT_RATIO {
        return false;
    }
    let us: Vec<usize> = window.iter().map(|&t| t as usize).collect();
    tok.decode(&us).contains(answer)
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

/// Per-probe amended-metric read-out: (hit, window distinct-token ratio).
fn probe_recall(r: &Reading, answer: &str, tok: &BpeTokenizer) -> (bool, f64) {
    let k = r.gen_toks.len().min(RECALL_FIRST_K);
    let ratio = if k == 0 {
        0.0
    } else {
        distinct_token_ratio(&r.gen_toks[..k])
    };
    (recall_hit(&r.gen_toks, answer, tok), ratio)
}

/// One recall row across the two postures.
struct RecallRow {
    label: String,
    baseline_hit: bool,
    baseline_ratio: f64,
    /// Baseline answer-token logprob (the SUM over the answer's content
    /// tokens) — the phase-A instrument's mis-ranked-vs-unknown
    /// discriminator.
    baseline_logprob: Option<f64>,
    /// The measured baseline generation text — the stale-capital/
    /// mis-ranking profile is READ, never assumed.
    baseline_gen: String,
    armed_hit: Option<bool>,
    armed_ratio: Option<f64>,
}

/// The amended G1 recall report (T3 amendment (i)–(iv)). Per fact: per-probe
/// baseline-delta scoring over the fact's recall-subject probes (held-out
/// wordings, the in-sample paraphrases when flagged, the held-out paraphrase
/// set), group summaries, the amendment-(ii) baseline-selection gate on the
/// heldpara group, and the amendment-(iv) repetition gate. `armed = None` is
/// the phase-A-only instrument — baseline-only tables, no deltas. REPORT
/// only: the floor adjudication stays T4's; the harness asserts only the
/// D/F identity laws.
fn print_recall_report(
    doc: &FactsDoc,
    tok: &BpeTokenizer,
    baseline: &[Reading],
    armed: Option<&[Reading]>,
    paraphrases: bool,
    answer_tokens: &[Vec<u32>],
) {
    if let Some(a) = armed {
        assert_eq!(
            baseline.len(),
            a.len(),
            "recall report: baseline/armed alignment drifted"
        );
    }
    for (i, f) in doc.facts.iter().enumerate() {
        let ans = f.answer.trim();
        // The answer's content-token count — the per-token mean-lp divisor
        // (summed lp penalises multi-token answers; the mean is the
        // like-for-like column).
        let n_ans = answer_tokens[i].len().max(1);
        // The fact's recall-subject groups, by label prefix.
        let mut groups: Vec<(&str, Vec<usize>)> = vec![("heldout", Vec::new())];
        if paraphrases {
            // The in-sample calibration asks — kept for continuity, marked.
            groups.push((PARAPHRASE_LABEL_PREFIX, Vec::new()));
        }
        groups.push(("heldpara", Vec::new()));
        for (idx, r) in baseline.iter().enumerate() {
            for (name, rows) in groups.iter_mut() {
                let hits = match *name {
                    "heldout" => r.label.starts_with(&format!("heldout[f{i}]")),
                    "heldpara" => r.label.starts_with(&format!("heldpara[f{i}]")),
                    other => other == PARAPHRASE_LABEL_PREFIX && r.label.starts_with(other),
                };
                if hits {
                    rows.push(idx);
                }
            }
        }
        let mut collapsed_base = 0usize;
        let mut collapsed_armed = 0usize;
        let mut subject = 0usize;
        for (name, idxs) in &groups {
            if idxs.is_empty() {
                continue;
            }
            let rows: Vec<RecallRow> = idxs
                .iter()
                .map(|&idx| {
                    let (bh, br) = probe_recall(&baseline[idx], ans, tok);
                    subject += 1;
                    if br < RECALL_DISTINCT_RATIO {
                        collapsed_base += 1;
                    }
                    let (ah, ar) = match armed {
                        Some(a) => {
                            let (h, r) = probe_recall(&a[idx], ans, tok);
                            if r < RECALL_DISTINCT_RATIO {
                                collapsed_armed += 1;
                            }
                            (Some(h), Some(r))
                        }
                        None => (None, None),
                    };
                    RecallRow {
                        label: baseline[idx].label.clone(),
                        baseline_hit: bh,
                        baseline_ratio: br,
                        baseline_logprob: baseline[idx].logprob,
                        baseline_gen: baseline[idx].gen_text.clone(),
                        armed_hit: ah,
                        armed_ratio: ar,
                    }
                })
                .collect();
            for r in &rows {
                match (r.armed_hit, r.armed_ratio) {
                    (Some(ah), Some(ar)) => println!(
                        "# recall[{}] baseline={}({:.2}) armed={}({:.2}) Δ{}",
                        r.label,
                        hit_word(r.baseline_hit),
                        r.baseline_ratio,
                        hit_word(ah),
                        ar,
                        delta_word(r.baseline_hit, ah),
                    ),
                    _ => println!(
                        "# recall[{}] baseline={}({:.2}) lp={:?} mean_lp={:?} armed=pending",
                        r.label,
                        hit_word(r.baseline_hit),
                        r.baseline_ratio,
                        r.baseline_logprob.map(|v| (v * 100.0).round() / 100.0),
                        r.baseline_logprob
                            .map(|v| ((v / n_ans as f64) * 100.0).round() / 100.0),
                    ),
                }
                if armed.is_none() {
                    // The phase-A instrument prints the measured baseline
                    // generation — the mis-ranking profile is READ here,
                    // never assumed.
                    println!("# gen[{}] {:?}", r.label, head(&r.baseline_gen, 64));
                }
            }
            let b_hits = rows.iter().filter(|r| r.baseline_hit).count();
            let tag = if *name == PARAPHRASE_LABEL_PREFIX {
                "recall-pre-para"
            } else if *name == "heldout" {
                "recall-pre"
            } else {
                "recall-heldpara"
            };
            match armed {
                Some(_) => {
                    let a_hits = rows.iter().filter(|r| r.armed_hit == Some(true)).count();
                    let gained = rows
                        .iter()
                        .filter(|r| !r.baseline_hit && r.armed_hit == Some(true))
                        .count();
                    let lost = rows
                        .iter()
                        .filter(|r| r.baseline_hit && r.armed_hit == Some(false))
                        .count();
                    let mark = if *name == PARAPHRASE_LABEL_PREFIX {
                        " [in-sample]"
                    } else if *name == "heldpara" {
                        " [floor subject]"
                    } else {
                        ""
                    };
                    println!(
                        "# {tag}[f{i}]: armed {a_hits}/{} (baseline {b_hits}/{}, Δ +{gained}/-{lost}){mark}",
                        rows.len(),
                        rows.len(),
                    );
                }
                None => {
                    let mark = if *name == "heldpara" {
                        " [floor subject]"
                    } else {
                        ""
                    };
                    println!("# {tag}[f{i}]: baseline {b_hits}/{}{mark}", rows.len(),);
                }
            }
            // The amendment-(ii) selection gate rides the heldpara group.
            if *name == "heldpara" {
                if b_hits * 5 <= rows.len() {
                    println!(
                        "# baseline-gate[f{i}] (amendment ii): baseline {b_hits}/{} ≤ 1/5 → PASS",
                        rows.len()
                    );
                } else {
                    println!(
                        "# baseline-gate[f{i}] (amendment ii): baseline {b_hits}/{} > 1/5 → FAIL — the fact is too well-known to test recall gains; re-anchor branch (plan-635 T3)",
                        rows.len()
                    );
                }
            }
        }
        if subject == 0 {
            continue;
        }
        // The amendment-(iv) repetition gate, beside G1b.
        match armed {
            Some(_) => {
                let verdict = if collapsed_armed == 0 { "GREEN" } else { "RED" };
                println!(
                    "# repetition-gate[f{i}] (amendment iv): collapsed baseline={collapsed_base}/{subject} armed={collapsed_armed}/{subject} → {verdict}"
                );
            }
            None => println!(
                "# repetition-gate[f{i}] (amendment iv): collapsed baseline={collapsed_base}/{subject} armed=pending → GREEN (baseline side)"
            ),
        }
    }
}

fn hit_word(h: bool) -> &'static str {
    if h { "hit" } else { "miss" }
}

fn delta_word(baseline: bool, armed: bool) -> &'static str {
    match (baseline, armed) {
        (false, true) => "+1",
        (true, false) => "-1",
        _ => "0",
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
    /// T3 amendment (i): the phase-A baseline readings ride the manifest so
    /// the baseline-delta evidence is durable in the artifact. Optional on
    /// read (older sidecars lack it).
    #[serde(default)]
    probes_baseline: Vec<ProbeEcho>,
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
    paraphrases: bool,
    phase_a_only: bool,
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
            paraphrases: false,
            phase_a_only: false,
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
                "--paraphrases" => a.paraphrases = true,
                "--phase-a-only" => a.phase_a_only = true,
                other => bail!(
                    "unknown flag {other} (flags: --gguf --out --facts --layers --rank --rounds --max-new --paraphrases --phase-a-only)"
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
    if args.paraphrases && args.facts.is_some() {
        bail!(
            "--paraphrases is authored for the DEFAULT fact doc (France → Paris) — pass no --facts"
        );
    }

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
    if args.paraphrases {
        println!(
            "# paraphrase calibration: {} authored train asks (plan-635 T3; flag-off = Recipe-1)",
            PARAPHRASE_TRAIN_ASKS.len()
        );
    }
    let facts: Vec<FactSpec> = doc
        .facts
        .iter()
        .zip(train_ask_strings(&doc, args.paraphrases))
        .map(|(f, asks)| FactSpec {
            train_asks: asks.iter().map(|s| enc(s)).collect(),
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
        // T3 amendment (ii): the fact's own held-out paraphrase probes (a
        // doc opting in carries them; the default doc carries none at
        // flag-off — its set rides the flag, below — so flag-off stays
        // byte-identical Recipe-1).
        for (k, h) in f.heldout_para_asks.iter().enumerate() {
            probes.push(Probe {
                label: format!("heldpara[f{i}]#{k}"),
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
    // Plan-635 T3: the same 5 paraphrases join the probe set — the G1 floor's
    // subject widens to them; the held-out wording stays scored separately
    // and never enters the solve (disjointness test-pinned).
    if args.paraphrases {
        for (k, p) in PARAPHRASE_TRAIN_ASKS.iter().enumerate() {
            probes.push(Probe {
                label: format!("{PARAPHRASE_LABEL_PREFIX}{k}"),
                ask: enc(p),
                answer: answer_content[0].clone(),
            });
        }
        // T3 amendment (ii): the authored HELD-OUT paraphrase set joins as
        // the amended floor's subject (labels heldpara[f0]#k).
        for (k, h) in HELD_OUT_PARAPHRASE_PROBES.iter().enumerate() {
            probes.push(Probe {
                label: format!("heldpara[f0]#{k}"),
                ask: enc(h),
                answer: answer_content[0].clone(),
            });
        }
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

    // The phase-A-only instrument (plan-635 T3 amendment): the probe-set
    // selection + re-anchor evidence run — no construct, no sidecar.
    if args.phase_a_only {
        println!("# ==== amended recall tables (phase-A-only instrument) ====");
        print_recall_report(
            &doc,
            &tok,
            &baseline,
            None,
            args.paraphrases,
            &answer_content,
        );
        println!(
            "# ==== RESULT: PHASE-A-ONLY — baseline instrument complete (no construct, no sidecar) ===="
        );
        return Ok(());
    }

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
    // The amended G1 recall tables (plan-635 T3 amendment (i)–(iv)) —
    // baseline-delta scoring, the held-out floor subject, the baseline
    // selection gate, and the repetition gate beside G1b. The per-probe
    // logprob/gen lines above stay the raw read-out the identities pin.
    print_recall_report(
        &doc,
        &tok,
        &baseline,
        Some(&armed),
        args.paraphrases,
        &answer_content,
    );

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
        probes_baseline: baseline
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

// ── tests (plan-635 T3 demote-identity battery; runs via
//    `cargo test --example closed_form_write_cell --features closed_form_write`)
#[cfg(test)]
mod tests {
    use super::*;

    /// Flag-off law: the default doc IS Recipe-1's calibration set verbatim
    /// — the two train asks, one held-out wording — and no paraphrase string
    /// leaks into the flag-off train set (the flag can never silently widen
    /// the flag-off posture).
    #[test]
    fn flag_off_calibration_is_recipe1_verbatim() {
        let doc = default_facts_doc();
        assert_eq!(doc.facts.len(), 1);
        let f = &doc.facts[0];
        assert_eq!(
            f.train_asks,
            vec![
                "The capital of France is".to_string(),
                "The capital city of France is called".to_string(),
            ]
        );
        assert_eq!(f.heldout_asks, vec!["France's capital city is".to_string()]);
        let sets = train_ask_strings(&doc, false);
        assert_eq!(sets.len(), 1);
        assert_eq!(
            sets[0],
            vec![
                "The capital of France is",
                "The capital city of France is called",
            ]
        );
        for p in PARAPHRASE_TRAIN_ASKS.iter() {
            assert!(
                !sets[0].contains(p),
                "paraphrase {p:?} leaked into flag-off"
            );
        }
    }

    /// Flag-on law: fact 0's train asks are exactly the authored five, in
    /// order; the held-out wording stays OUT (it is the G1 read-out, never a
    /// calibration input).
    #[test]
    fn flag_on_train_set_is_the_authored_five() {
        let doc = default_facts_doc();
        let sets = train_ask_strings(&doc, true);
        assert_eq!(sets.len(), 1);
        assert_eq!(sets[0], PARAPHRASE_TRAIN_ASKS.to_vec());
        assert!(!sets[0].contains(&"France's capital city is"));
    }

    /// The paraphrase set's shape: exactly five, unique, same-relation
    /// surface forms that end in the teacher-forceable pre-answer frame and
    /// keep the fact's subject.
    #[test]
    fn paraphrase_set_shape() {
        assert_eq!(PARAPHRASE_TRAIN_ASKS.len(), 5);
        let uniq: std::collections::HashSet<_> = PARAPHRASE_TRAIN_ASKS.iter().copied().collect();
        assert_eq!(uniq.len(), 5, "duplicate paraphrase wording");
        for p in PARAPHRASE_TRAIN_ASKS.iter() {
            assert!(
                p.contains("France") || p.contains("French"),
                "paraphrase {p:?} lost the subject"
            );
            assert!(
                p.trim_end().ends_with("is"),
                "paraphrase {p:?} lost the pre-answer frame"
            );
        }
    }

    /// The probe builder and the recall report share ONE label prefix — the
    /// selector cannot drift from the labels it counts.
    #[test]
    fn paraphrase_label_prefix_is_the_shared_selector() {
        let labels: Vec<String> = (0..PARAPHRASE_TRAIN_ASKS.len())
            .map(|k| format!("{PARAPHRASE_LABEL_PREFIX}{k}"))
            .collect();
        assert_eq!(labels.len(), 5);
        assert!(
            labels
                .iter()
                .all(|l| l.starts_with(PARAPHRASE_LABEL_PREFIX))
        );
    }

    /// T3 amendment (ii): the held-out paraphrase set is HELD-OUT — five
    /// unique asks sharing no string with any calibration ask, any doc ask
    /// (either direction: equality AND containment), and never leaking the
    /// answer. This is the law that makes the amended floor's subject
    /// out-of-sample.
    #[test]
    fn heldout_para_set_is_disjoint_from_all_asks_and_never_leaks_the_answer() {
        let doc = default_facts_doc();
        let f = &doc.facts[0];
        let answer = f.answer.trim();
        let mut all_asks: Vec<&str> = Vec::new();
        all_asks.extend(f.train_asks.iter().map(String::as_str));
        all_asks.extend(PARAPHRASE_TRAIN_ASKS.iter().copied());
        all_asks.extend(f.contrast_asks.iter().map(String::as_str));
        all_asks.extend(f.heldout_asks.iter().map(String::as_str));
        all_asks.extend(doc.lookalike_asks.iter().map(String::as_str));
        // Shape: exactly five, unique, no answer leak.
        assert_eq!(HELD_OUT_PARAPHRASE_PROBES.len(), 5);
        let uniq: std::collections::HashSet<_> =
            HELD_OUT_PARAPHRASE_PROBES.iter().copied().collect();
        assert_eq!(uniq.len(), 5, "duplicate held-out paraphrase wording");
        for h in HELD_OUT_PARAPHRASE_PROBES.iter() {
            assert!(
                !h.contains(answer),
                "held-out probe {h:?} leaks the answer {answer:?}"
            );
            // Disjointness vs every ask: no equality, no containment either
            // direction.
            for a in &all_asks {
                assert_ne!(h, a, "held-out probe IS an existing ask: {h:?}");
                assert!(
                    !h.contains(a),
                    "held-out probe {h:?} contains existing ask {a:?}"
                );
                assert!(
                    !a.contains(h),
                    "existing ask {a:?} contains held-out probe {h:?}"
                );
            }
        }
        // Also within-set disjointness (containment both directions).
        for (i, hi) in HELD_OUT_PARAPHRASE_PROBES.iter().enumerate() {
            for (j, hj) in HELD_OUT_PARAPHRASE_PROBES.iter().enumerate() {
                if i != j {
                    assert!(
                        !hi.contains(hj),
                        "held-out probe {hi:?} contains sibling {hj:?}"
                    );
                }
            }
        }
        // The calibration set itself must not leak the answer either (the
        // training asks encode the relation by frame, never by answer).
        for p in PARAPHRASE_TRAIN_ASKS.iter() {
            assert!(!p.contains(answer), "train ask {p:?} leaks the answer");
        }
    }

    /// T3 amendment (iii): the repetition-run detector — the T3 attractor
    /// shape (`" the capital Paris Paris Paris…"`, measured 3/8 distinct)
    /// must fall below the floor; a clean generation must clear it; the
    /// first-k truncation must keep the guard on the WINDOW (a diverse tail
    /// past the first k never rescues a repeated window).
    #[test]
    fn repetition_guard_pins_the_attractor_shape() {
        let attractor: Vec<u32> = vec![10, 11, 12, 12, 12, 12, 12, 12, 12];
        let ratio = distinct_token_ratio(&attractor[..RECALL_FIRST_K]);
        assert!(
            ratio < RECALL_DISTINCT_RATIO,
            "attractor window {ratio} must fall below the floor"
        );
        let clean: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let ratio = distinct_token_ratio(&clean[..RECALL_FIRST_K]);
        assert!(ratio >= RECALL_DISTINCT_RATIO);
        // Early-EOS correct answers are not punished: a 2-token window of
        // distinct ids measures 1.0.
        assert_eq!(distinct_token_ratio(&[7, 0]), 1.0);
        // The guard reads the FIRST-k window only — a diverse tail past k
        // cannot rescue a repeated window.
        let mixed: Vec<u32> = vec![12, 12, 12, 12, 12, 12, 12, 12, 1, 2, 3, 4];
        assert!(
            distinct_token_ratio(&mixed[..RECALL_FIRST_K]) < RECALL_DISTINCT_RATIO,
            "diverse tail past the window must not rescue a repeated window"
        );
        // The floor is exact at 4/8: four distinct ids in eight tokens sit
        // ON the floor (pass); three fall below (repetition).
        let on_floor: Vec<u32> = vec![1, 2, 3, 4, 1, 2, 3, 4];
        assert_eq!(
            distinct_token_ratio(&on_floor),
            RECALL_DISTINCT_RATIO,
            "4/8 must sit exactly on the floor"
        );
        let below: Vec<u32> = vec![1, 2, 3, 3, 3, 3, 3, 3];
        assert!(distinct_token_ratio(&below) < RECALL_DISTINCT_RATIO);
    }

    /// The held-out probes carry their own label prefix (heldpara) distinct
    /// from the in-sample paraphrase# labels — the report's group selector
    /// can never conflate the floor subject with the calibration asks.
    #[test]
    fn heldpara_labels_never_collide_with_calibration_labels() {
        for k in 0..HELD_OUT_PARAPHRASE_PROBES.len() {
            let l = format!("heldpara[f0]#{k}");
            assert!(!l.starts_with(PARAPHRASE_LABEL_PREFIX));
            assert!(l.starts_with("heldpara[f0]#"));
        }
    }
}
