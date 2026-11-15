//! lens_gates_t4 — katgpt-rs Issue 999 T4: the real gates over the T3 fit.
//!
//! Runs the pre-registered G1/G2 gates (Research 666 §2.3) against the
//! production T3 fit artifacts (`lens_fit_t3`'s `fit_all.k|j` + manifest):
//!
//! - **G1 (S read)** — mines the L0-class MLP poles from a HELD-OUT corpus
//!   slice (the token stream past the fit's docs, so the firing tokens are
//!   held out by construction), then reads each pole's read direction
//!   (`â ± b̂`, the unit gate/up rows as the residual sees them, per SiLU
//!   lobe) through the fitted K at the layer's `Mid` point and checks the
//!   pre-registered bar: top-10 hit rate ≥ 25% with a mismatched-pole
//!   control < 1%. No minable poles ⇒ **UNSEEN** (scope finding, never a
//!   refutation). Even/odd half fits + a mining-corpus-masked variant run
//!   as diagnostics.
//! - **G2 (R read)** — steers with the fitted J directions (the restricted
//!   fit's token-detector unit-`W_E` rows) at `Out` points: the predicted
//!   `Δlogits = W_U·(Jᵀ_row)` is correlated with the MEASURED steering (a
//!   true forward with the write injected at the fit's seed window, the
//!   response summed over positions and meaned over docs — the exact twin
//!   of the fit's accumulation). Bar: matched correlation ≥ 0.45 at the
//!   matched depth band (layers in `[n/4, 3n/4)`), mismatched control
//!   within ±0.04.
//!
//! G3 is already in the fit's manifest; G4 (one vocab-side matvec per read,
//! zero-alloc; fit cost + box state) is pinned by the library tests and the
//! fit's `fit_cost` record — this cell only consumes them.
//!
//! ```text
//! cargo run --release --features lens --example lens_gates_t4 -- \
//!   --fit-dir /tmp/lens_t3_run \
//!   [--gguf ../riir-train/data/Qwen3.5-0.8B-Base-Q8_0.gguf]
//!   [--corpus-dir ../riir-reflex/.raw/datasets/ag_news]
//!   [--skip-docs N] [--mining-docs 16] [--t 128]
//!   [--g1-layers 0] [--top-firings 8] [--min-firings 3] [--top-k 10]
//!   [--g2-points out:4,out:8,out:12,out:16,out:20] [--g2-dirs 8]
//!   [--g2-docs 3] [--g2-alpha 1.0]
//!   [--box-state "<PROVENANCE line>"]
//! ```
//!
//! Writes `t4_gates.json` + `t4_g1_poles.json` into the fit dir.

use anyhow::{Context, Result, bail};
use riir_infer_core::gguf_loader::{GgufFile, load_qwen_deltanet_weights_gguf};
use riir_infer_core::lens::fit::{
    G1Report, G2Report, LensFit, RareMask, gate_g1, gate_g2, mean_embedding, pearson,
    r_read_delta_dir_into,
};
use riir_infer_core::lens::stack::{
    FitPoint, StackCapture, StackWeights, StackWrite, stack_forward, stack_forward_write,
};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::tokenizer::BpeTokenizer;
use serde_json::json;
use std::path::{Path, PathBuf};

const DEFAULT_GGUF: &str = "../riir-train/data/Qwen3.5-0.8B-Base-Q8_0.gguf";
const DEFAULT_CORPUS_DIR: &str = "../riir-reflex/.raw/datasets/ag_news";

// ── args ────────────────────────────────────────────────────────────────

struct Args {
    fit_dir: PathBuf,
    gguf: PathBuf,
    corpus_dir: Option<PathBuf>,
    corpus_file: Option<PathBuf>,
    skip_docs: Option<usize>,
    mining_docs: usize,
    t: usize,
    g1_layers: Vec<usize>,
    top_firings: usize,
    min_firings: usize,
    top_k: usize,
    g2_points: Vec<(bool, usize)>, // (is_out, layer)
    g2_dirs: usize,
    g2_docs: usize,
    g2_alpha: f32,
    box_state: String,
}

impl Args {
    fn parse() -> Result<Args> {
        let mut a = Args {
            fit_dir: PathBuf::from("/tmp/lens_t3_run"),
            gguf: PathBuf::from(DEFAULT_GGUF),
            corpus_dir: Some(PathBuf::from(DEFAULT_CORPUS_DIR)),
            corpus_file: None,
            skip_docs: None,
            mining_docs: 16,
            t: 128,
            g1_layers: vec![0],
            top_firings: 8,
            min_firings: 3,
            top_k: 10,
            g2_points: [4usize, 8, 12, 16, 20].iter().map(|&l| (true, l)).collect(),
            g2_dirs: 8,
            g2_docs: 3,
            g2_alpha: 1.0,
            box_state: String::new(),
        };
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let mut val = || it.next().with_context(|| format!("{flag} needs a value"));
            match flag.as_str() {
                "--fit-dir" => a.fit_dir = PathBuf::from(val()?),
                "--gguf" => a.gguf = PathBuf::from(val()?),
                "--corpus-dir" => a.corpus_dir = Some(PathBuf::from(val()?)),
                "--corpus" => {
                    a.corpus_file = Some(PathBuf::from(val()?));
                    a.corpus_dir = None;
                }
                "--skip-docs" => a.skip_docs = Some(val()?.parse().context("--skip-docs parse")?),
                "--mining-docs" => a.mining_docs = val()?.parse().context("--mining-docs parse")?,
                "--t" => a.t = val()?.parse().context("--t parse")?,
                "--g1-layers" => {
                    a.g1_layers = val()?
                        .split(',')
                        .map(|s| s.trim().parse().context("--g1-layers parse"))
                        .collect::<Result<Vec<usize>>>()?;
                }
                "--top-firings" => a.top_firings = val()?.parse().context("--top-firings parse")?,
                "--min-firings" => a.min_firings = val()?.parse().context("--min-firings parse")?,
                "--top-k" => a.top_k = val()?.parse().context("--top-k parse")?,
                "--g2-points" => {
                    let mut pts = Vec::new();
                    for part in val()?.split(',') {
                        let part = part.trim();
                        let (kind, l) = part
                            .split_once(':')
                            .with_context(|| format!("--g2-points: bad {part}"))?;
                        let layer = l.parse().with_context(|| format!("--g2-points: {part}"))?;
                        match kind {
                            "out" => pts.push((true, layer)),
                            "mid" => pts.push((false, layer)),
                            other => bail!("--g2-points: unknown kind {other} (out|mid)"),
                        }
                    }
                    a.g2_points = pts;
                }
                "--g2-dirs" => a.g2_dirs = val()?.parse().context("--g2-dirs parse")?,
                "--g2-docs" => a.g2_docs = val()?.parse().context("--g2-docs parse")?,
                "--g2-alpha" => a.g2_alpha = val()?.parse().context("--g2-alpha parse")?,
                "--box-state" => a.box_state = val()?,
                other => bail!("unknown flag {other}"),
            }
        }
        if a.mining_docs < 2 {
            bail!("--mining-docs must be ≥ 2 (the pole miner needs more than one doc)");
        }
        if a.g2_dirs < 2 || a.g2_dirs > 64 {
            bail!("--g2-dirs must be in [2, 64] (the control needs mismatched pairs)");
        }
        Ok(a)
    }
}

// ── corpus (the lens_fit_t3 reader, byte-identical row order) ───────────

fn read_rows(dir: &Path, shards_max: usize) -> Result<Vec<String>> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("train-") && n.ends_with(".json"))
                .unwrap_or(false)
        })
        .collect();
    paths.sort();
    if paths.is_empty() {
        bail!("no train-*.json shards under {}", dir.display());
    }
    paths.truncate(shards_max);
    let mut rows = Vec::with_capacity(paths.len());
    for p in &paths {
        let raw =
            std::fs::read_to_string(p).with_context(|| format!("read shard {}", p.display()))?;
        let v: serde_json::Value =
            serde_json::from_str(&raw).with_context(|| format!("parse {}", p.display()))?;
        let shard_rows = v
            .get("rows")
            .and_then(|r| r.as_array())
            .with_context(|| format!("shard {} has no rows[]", p.display()))?;
        for r in shard_rows {
            if let Some(text) = r
                .get("row")
                .and_then(|r| r.get("text"))
                .and_then(|t| t.as_str())
            {
                rows.push(text.to_string());
            }
        }
    }
    if rows.is_empty() {
        bail!(
            "every shard under {} parsed to zero text rows",
            dir.display()
        );
    }
    Ok(rows)
}

/// Tokenize rows in order until `want_tokens` are collected (the fit's
/// reader verbatim — the held-out guarantee is that the fit consumed the
/// FIRST `skip·t` tokens of this exact stream).
fn tokenize_rows(tok: &BpeTokenizer, rows: &[String], want_tokens: usize) -> Result<Vec<usize>> {
    let mut tokens: Vec<usize> = Vec::with_capacity(want_tokens + 512);
    for (i, row) in rows.iter().enumerate() {
        let joined = if i == 0 {
            row.clone()
        } else {
            format!("\n{row}")
        };
        tokens.extend(tok.encode(&joined));
        if tokens.len() >= want_tokens {
            break;
        }
    }
    if tokens.len() < want_tokens {
        bail!(
            "corpus exhausted at {} tokens (< {want_tokens}) — add shards",
            tokens.len()
        );
    }
    Ok(tokens)
}

// ── fit artifacts ─────────────────────────────────────────────────────────────

/// Load one `fit_*.k|j` file: `points` blocks of `block_len` f32 LE.
fn load_fit_blocks(
    path_base: &Path,
    name: &str,
    points: usize,
    block_len: usize,
) -> Result<Vec<Vec<f32>>> {
    let path = path_base.join(name);
    let raw = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    let total = points * block_len;
    if raw.len() != total * 4 {
        bail!(
            "{}: {} B, expected {} f32 — the fit shape moved",
            path.display(),
            raw.len(),
            total
        );
    }
    let flat: Vec<f32> = raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    Ok(flat.chunks_exact(block_len).map(|c| c.to_vec()).collect())
}

// ── pole mining (G1) ────────────────────────────────────────────────────

/// One pole's top firings, kept sorted by value descending.
struct TopFirings {
    vals: Vec<f32>,
    tokens: Vec<u32>,
    /// The gate pre-activation's silu value at each firing (its sign IS the
    /// gate's sign — the SiLU lobe the pole sits on).
    sgns: Vec<f32>,
    count: u32,
}

impl TopFirings {
    fn new(cap: usize) -> Self {
        Self {
            vals: Vec::with_capacity(cap),
            tokens: Vec::with_capacity(cap),
            sgns: Vec::with_capacity(cap),
            count: 0,
        }
    }

    fn offer(&mut self, val: f32, token: u32, sgn: f32, cap: usize) {
        self.count += 1;
        if self.vals.len() < cap {
            let at = self
                .vals
                .binary_search_by(|p| val.total_cmp(p).reverse())
                .unwrap_or_else(|at| at);
            self.vals.insert(at, val);
            self.tokens.insert(at, token);
            self.sgns.insert(at, sgn);
            return;
        }
        if val <= self.vals[cap - 1] {
            return;
        }
        let at = self
            .vals
            .binary_search_by(|p| val.total_cmp(p).reverse())
            .unwrap_or_else(|at| at);
        if at < cap {
            self.vals.insert(at, val);
            self.tokens.insert(at, token);
            self.sgns.insert(at, sgn);
            self.vals.truncate(cap);
            self.tokens.truncate(cap);
            self.sgns.truncate(cap);
        }
    }

    /// The mode token over the kept firings (ties → smallest id).
    fn mode_token(&self) -> Option<(u32, f64)> {
        if self.tokens.is_empty() {
            return None;
        }
        let mut best: Option<(u32, usize)> = None;
        for &t in &self.tokens {
            let c = self.tokens.iter().filter(|&&x| x == t).count();
            let better = match best {
                None => true,
                Some((bt, bc)) => c > bc || (c == bc && t < bt),
            };
            if better {
                best = Some((t, c));
            }
        }
        best.map(|(t, c)| (t, c as f64 / self.tokens.len() as f64))
    }
}

/// Mine the MLP poles of `layers` over `docs`: per neuron and pole, the top
/// firings' tokens from the captured `mlp_s`/`mlp_u` values (act = s⊙u, the
/// forward's own MLP value — no extra matvecs).
fn mine_poles(
    w: &StackWeights,
    docs: &[Vec<usize>],
    layers: &[usize],
    cap_f: usize,
) -> Result<Vec<Vec<Vec<TopFirings>>>> {
    // [layer][pole 0=+/1=−][neuron]
    let hidden = w.geo.mlp_hidden;
    let mut mined: Vec<Vec<Vec<TopFirings>>> = layers
        .iter()
        .map(|_| {
            vec![
                (0..hidden)
                    .map(|_| TopFirings::new(cap_f))
                    .collect::<Vec<_>>(),
                (0..hidden)
                    .map(|_| TopFirings::new(cap_f))
                    .collect::<Vec<_>>(),
            ]
        })
        .collect();
    let mut cap = StackCapture::new(&w.geo);
    for doc in docs {
        stack_forward(doc, w, &mut cap);
        for (li, &layer) in layers.iter().enumerate() {
            let lc = &cap.layers[layer];
            for (t, &tok) in doc.iter().enumerate() {
                let tok = tok as u32;
                let s = &lc.mlp_s[t * hidden..(t + 1) * hidden];
                let u = &lc.mlp_u[t * hidden..(t + 1) * hidden];
                for n in 0..hidden {
                    let a = s[n] * u[n];
                    if a > 0.0 {
                        mined[li][0][n].offer(a, tok, s[n], cap_f);
                    } else if a < 0.0 {
                        mined[li][1][n].offer(-a, tok, s[n], cap_f);
                    }
                }
            }
        }
    }
    Ok(mined)
}

fn main() -> Result<()> {
    let args = Args::parse()?;

    // ── the fit's manifest ──
    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(args.fit_dir.join("manifest.json"))
            .context("read fit manifest.json")?,
    )
    .context("parse fit manifest.json")?;
    let fit_docs = manifest["docs"].as_u64().context("manifest docs")? as usize;
    let fit_t = manifest["t"].as_u64().context("manifest t")? as usize;
    let fit_d = manifest["d"].as_u64().context("manifest d")? as usize;
    let j_count = manifest["j_dirs"]["count"]
        .as_u64()
        .context("manifest j count")? as usize;
    let j_tokens: Vec<usize> = manifest["j_dirs"]["token_ids"]
        .as_array()
        .context("manifest j token_ids")?
        .iter()
        .map(|v| v.as_u64().map(|x| x as usize).context("token id"))
        .collect::<Result<Vec<usize>>>()?;
    let n_points = manifest["points"]
        .as_array()
        .context("manifest points")?
        .len();
    let seed_lo = manifest["seed_window"][0].as_u64().context("seed lo")? as usize;
    let seed_hi = manifest["seed_window"][1].as_u64().context("seed hi")? as usize;
    if fit_t != args.t {
        bail!(
            "fit T={} vs --t {} — the mining corpus must chop at the fit's doc length",
            fit_t,
            args.t
        );
    }
    let skip_docs = args.skip_docs.unwrap_or(fit_docs);

    // ── model + tokenizer (the fit's loading path verbatim) ──
    let t_load = std::time::Instant::now();
    let (config, weights) = load_qwen_deltanet_weights_gguf(&args.gguf)
        .with_context(|| format!("load {}", args.gguf.display()))?;
    let tok = {
        let gguf = GgufFile::open(&args.gguf).context("re-open gguf for tokenizer")?;
        BpeTokenizer::from_gguf(&gguf).context("gpt2 BPE tokenizer from gguf")?
    };
    println!(
        "# model: {} layers={} d={} vocab={} | load {:.1}s",
        args.gguf.display(),
        config.n_layer,
        config.n_embd,
        config.vocab_size,
        t_load.elapsed().as_secs_f32()
    );
    if config.n_embd != fit_d {
        bail!("model d={} vs fit d={}", config.n_embd, fit_d);
    }
    let rope = RopeFreqTable::new(config.rope_theta, config.head_dim);
    let w = StackWeights::from_model(&weights, &config, rope.as_slice(), args.t);
    let geo = &w.geo;
    let d = geo.n_embd;
    let vocab = geo.vocab;
    let points = geo.fit_points();
    if points.len() != n_points {
        bail!(
            "model points={} vs fit manifest points={}",
            points.len(),
            n_points
        );
    }

    // ── the fits ──
    let mean_emb = mean_embedding(w.wte, vocab, d);
    let k_all = load_fit_blocks(&args.fit_dir, "fit_all.k", points.len(), d * d)?;
    let j_all = load_fit_blocks(&args.fit_dir, "fit_all.j", points.len(), j_count * d)?;
    let k_even = load_fit_blocks(&args.fit_dir, "fit_even.k", points.len(), d * d)?;
    let j_even = load_fit_blocks(&args.fit_dir, "fit_even.j", points.len(), j_count * d)?;
    let k_odd = load_fit_blocks(&args.fit_dir, "fit_odd.k", points.len(), d * d)?;
    let j_odd = load_fit_blocks(&args.fit_dir, "fit_odd.j", points.len(), j_count * d)?;
    let fit_all = LensFit {
        d,
        points: points.clone(),
        k: k_all,
        j: j_all,
        mean_emb: mean_emb.clone(),
        n_docs: fit_docs as u64,
    };
    let fit_even = LensFit {
        d,
        points: points.clone(),
        k: k_even,
        j: j_even,
        mean_emb: mean_emb.clone(),
        n_docs: fit_docs.div_ceil(2) as u64,
    };
    let fit_odd = LensFit {
        d,
        points: points.clone(),
        k: k_odd,
        j: j_odd,
        mean_emb,
        n_docs: (fit_docs / 2) as u64,
    };

    // ── held-out corpus (the token stream past the fit's docs) ──
    let rows = match (&args.corpus_dir, &args.corpus_file) {
        (Some(dir), _) => read_rows(dir, 200)?,
        (_, Some(f)) => {
            let text =
                std::fs::read_to_string(f).with_context(|| format!("read {}", f.display()))?;
            text.as_bytes()
                .chunks(8 * 1024)
                .map(|c| String::from_utf8_lossy(c).into_owned())
                .collect::<Vec<_>>()
        }
        _ => bail!("pass --corpus-dir or --corpus"),
    };
    let want = (skip_docs + args.mining_docs) * args.t;
    let stream = tokenize_rows(&tok, &rows, want)?;
    let docs: Vec<Vec<usize>> = stream[skip_docs * args.t..]
        .chunks_exact(args.t)
        .take(args.mining_docs)
        .map(|c| c.to_vec())
        .collect();
    if docs.len() < args.mining_docs {
        bail!(
            "only {} held-out docs past doc {skip_docs} (want {}) — add shards",
            docs.len(),
            args.mining_docs
        );
    }
    println!(
        "# held-out: docs [{}, {}) of the fit's token stream ({} docs × T={})",
        skip_docs,
        skip_docs + docs.len(),
        docs.len(),
        args.t
    );

    // ══ G1 — the S read names the firing token ══════════════════════════
    let mut g1_report = serde_json::Map::new();
    let t_g1 = std::time::Instant::now();
    // One capture pass over the held-out docs mines EVERY requested layer.
    let mined_layers = mine_poles(&w, &docs, &args.g1_layers, args.top_firings)?;
    for (li, &layer) in args.g1_layers.iter().enumerate() {
        let (pos, neg) = (&mined_layers[li][0], &mined_layers[li][1]);
        // The silent floor: 5% of the median per-pole peak (poles below it
        // never fire enough to define a token). Recorded in the report.
        let mut peaks: Vec<f32> = pos
            .iter()
            .chain(neg.iter())
            .filter(|p| p.count > 0)
            .map(|p| p.vals[0])
            .collect();
        if peaks.is_empty() {
            println!("# G1 L{layer}: UNSEEN — no firing poles at all");
            g1_report.insert(format!("L{layer}"), json!({"unseen": true, "poles": 0}));
            continue;
        }
        peaks.sort_by(|a, b| a.total_cmp(b));
        let median_peak = peaks[peaks.len() / 2];
        let floor = 0.05 * median_peak;
        // Pole directions: the read rows â ± b̂ as the residual sees them
        // (gate/up rows ⊙ the MLP-input norm's weight, unit-normalized),
        // per SiLU lobe — the reference estimator's `read_rows`.
        let gamma = &w.layers[layer].post_attn_norm;
        let mut poles: Vec<(usize, Vec<f32>)> = Vec::new();
        let mut held_out: Vec<(usize, usize)> = Vec::new();
        let mut pole_records = Vec::new();
        let mut detector_count = 0usize;
        for (neuron, pn) in pos.iter().enumerate() {
            for (sign, pole) in [(1i8, pn), (-1i8, &neg[neuron])] {
                if pole.count < args.min_firings as u32 || pole.vals[0] < floor {
                    continue;
                }
                let (token, share) = pole
                    .mode_token()
                    .with_context(|| format!("L{layer} n{neuron} s{sign}: no firings"))?;
                if share >= 1.0 / 3.0 {
                    detector_count += 1;
                }
                let lobe_positive = pole.sgns.iter().sum::<f32>() >= 0.0;
                // â = unit(w_gate[n] ⊙ γ), b̂ = unit(w_up[n] ⊙ γ).
                let mut a_row: Vec<f32> = gamma
                    .iter()
                    .zip(&w.layers[layer].w_gate[neuron * d..(neuron + 1) * d])
                    .map(|(&g, &x)| g * x)
                    .collect();
                let mut b_row: Vec<f32> = gamma
                    .iter()
                    .zip(&w.layers[layer].w_up[neuron * d..(neuron + 1) * d])
                    .map(|(&g, &x)| g * x)
                    .collect();
                let na: f32 = a_row.iter().map(|x| x * x).sum::<f32>().sqrt();
                let nb: f32 = b_row.iter().map(|x| x * x).sum::<f32>().sqrt();
                if !(na > 1e-12 && nb > 1e-12) {
                    continue; // degenerate row
                }
                for x in a_row.iter_mut() {
                    *x /= na;
                }
                for x in b_row.iter_mut() {
                    *x /= nb;
                }
                let c: Vec<f32> = match (sign > 0, lobe_positive) {
                    (true, true) => a_row.iter().zip(&b_row).map(|(a, b)| a + b).collect(),
                    (false, true) => a_row.iter().zip(&b_row).map(|(a, b)| a - b).collect(),
                    (true, false) => b_row.iter().map(|b| -b).collect(),
                    (false, false) => b_row.clone(),
                };
                let idx = poles.len();
                poles.push((
                    fit_all
                        .point_index(FitPoint::Mid(layer))
                        .expect("mid point"),
                    c,
                ));
                held_out.push((idx, token as usize));
                pole_records.push(json!({
                    "neuron": neuron, "sign": sign, "lobe": if lobe_positive {"+"} else {"-"},
                    "token": token, "mode_share": share, "top1": pole.vals[0],
                    "firings": pole.count,
                }));
            }
        }
        let refs: Vec<(usize, &[f32])> = poles.iter().map(|(pi, c)| (*pi, c.as_slice())).collect();
        let mask = RareMask::allow_all(vocab);
        let mask_mining = RareMask::from_docs(&docs, vocab, 2);
        // The four variants are independent reads over the same poles — one
        // thread per fit (the vocab matvec is memory-bound; this is the
        // cheap 4×).
        let (rep, rep_even, rep_odd, rep_masked) = {
            let mut buf1 = vec![0.0f32; d];
            let mut sc1 = vec![0.0f32; vocab];
            let mut buf2 = vec![0.0f32; d];
            let mut sc2 = vec![0.0f32; vocab];
            let mut buf3 = vec![0.0f32; d];
            let mut sc3 = vec![0.0f32; vocab];
            let mut buf4 = vec![0.0f32; d];
            let mut sc4 = vec![0.0f32; vocab];
            std::thread::scope(|s| {
                let a = s.spawn(|| {
                    gate_g1(
                        w.wte, vocab, &fit_all, &refs, &held_out, &mask, args.top_k, &mut buf1,
                        &mut sc1,
                    )
                });
                let b = s.spawn(|| {
                    gate_g1(
                        w.wte, vocab, &fit_even, &refs, &held_out, &mask, args.top_k, &mut buf2,
                        &mut sc2,
                    )
                });
                let c = s.spawn(|| {
                    gate_g1(
                        w.wte, vocab, &fit_odd, &refs, &held_out, &mask, args.top_k, &mut buf3,
                        &mut sc3,
                    )
                });
                let e = s.spawn(|| {
                    gate_g1(
                        w.wte,
                        vocab,
                        &fit_all,
                        &refs,
                        &held_out,
                        &mask_mining,
                        args.top_k,
                        &mut buf4,
                        &mut sc4,
                    )
                });
                (
                    a.join().expect("g1 primary panicked"),
                    b.join().expect("g1 even panicked"),
                    c.join().expect("g1 odd panicked"),
                    e.join().expect("g1 masked panicked"),
                )
            })
        };
        println!(
            "# G1 L{layer}: poles={} (detectors {detector_count}) hit={:.3} control={:.3} pass={} | even {:.3} odd {:.3} masked {:.3}",
            rep.poles,
            rep.hit_rate,
            rep.control_rate,
            rep.pass,
            rep_even.hit_rate,
            rep_odd.hit_rate,
            rep_masked.hit_rate
        );
        let g1j = |r: &G1Report| {
            json!({
                "poles": r.poles, "top_k": r.top_k, "hit_rate": r.hit_rate,
                "control_rate": r.control_rate, "unseen": r.unseen, "pass": r.pass,
            })
        };
        g1_report.insert(
            format!("L{layer}"),
            json!({
                "primary": g1j(&rep),
                "even": g1j(&rep_even),
                "odd": g1j(&rep_odd),
                "masked_mining_corpus": g1j(&rep_masked),
                "detectors": detector_count,
                "floor": floor,
                "floor_rule": "silent if peak < 0.05 × median(per-pole peak, poles with ≥1 firing)",
                "min_firings": args.min_firings,
            }),
        );
        std::fs::write(
            args.fit_dir.join(format!("t4_g1_poles_L{layer}.json")),
            serde_json::to_string_pretty(&pole_records).context("poles serialize")?,
        )?;
    }
    println!(
        "# G1 done over layers {:?} in {:.1}s",
        args.g1_layers,
        t_g1.elapsed().as_secs_f32()
    );

    // ══ G2 — the R read predicts the measured steering ══════════════════
    let t_g2 = std::time::Instant::now();
    // The sampled J directions: spread over the fit's 64 token-detector
    // rows; the write vectors must be the fit's EXACT unit rows.
    let dir_ids: Vec<usize> = (0..args.g2_dirs)
        .map(|i| i * j_count / args.g2_dirs)
        .collect();
    let dir_vecs: Vec<Vec<f32>> = dir_ids
        .iter()
        .map(|&j| {
            let t = j_tokens[j];
            let row = &w.wte[t * d..(t + 1) * d];
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            row.iter().map(|x| x / norm).collect()
        })
        .collect();
    let mut points_g2: Vec<(String, FitPoint, usize)> = Vec::new();
    for (is_out, layer) in &args.g2_points {
        let p = if *is_out {
            FitPoint::Out(*layer)
        } else {
            FitPoint::Mid(*layer)
        };
        let pi = fit_all
            .point_index(p)
            .with_context(|| format!("point {p:?} not in the fit"))?;
        points_g2.push((
            format!("{}:{layer}", if *is_out { "out" } else { "mid" }),
            p,
            pi,
        ));
    }
    let mid_band: Vec<usize> = points_g2
        .iter()
        .filter(|(_, p, _)| {
            let l = p.layer();
            l >= geo.n_layers / 4 && l < (geo.n_layers * 3).div_ceil(4)
        })
        .map(|(_, _, pi)| *pi)
        .collect();
    let g2_doc_set = &docs[..args.g2_docs.min(docs.len())];
    let mut hn_base = vec![0.0f32; args.t * d];
    let mut cap_base = StackCapture::new(geo);
    let mut cap_w = StackCapture::new(geo);
    // meas[point][dir] = Σ_docs (1/window) Σ_{t'} Δhn — then one lm_head
    // matvec per (point, dir) gives the measured Δlogits.
    let mut meas_hn: Vec<Vec<Vec<f32>>> = points_g2
        .iter()
        .map(|_| vec![vec![0.0f32; d]; args.g2_dirs])
        .collect();
    let mut baselines = 0usize;
    let mut writes = 0usize;
    for doc in g2_doc_set {
        stack_forward(doc, &w, &mut cap_base);
        baselines += 1;
        hn_base.copy_from_slice(&cap_base.hn);
        for (pi, (_, point, _)) in points_g2.iter().enumerate() {
            for (di, dv) in dir_vecs.iter().enumerate() {
                stack_forward_write(
                    doc,
                    &w,
                    &mut cap_w,
                    Some(&StackWrite {
                        point: *point,
                        dir: dv,
                        alpha: args.g2_alpha,
                        seed_lo,
                        seed_hi,
                    }),
                );
                writes += 1;
                let acc = &mut meas_hn[pi][di];
                for (j, acc_j) in acc.iter_mut().enumerate() {
                    let mut s = 0.0f64;
                    for off in (j..args.t * d).step_by(d) {
                        s += (cap_w.hn[off] - hn_base[off]) as f64;
                    }
                    *acc_j += (s / (seed_hi - seed_lo) as f64) as f32;
                }
            }
        }
    }
    let nd = g2_doc_set.len() as f32;
    let mut preds: Vec<Vec<Vec<f32>>> = Vec::with_capacity(points_g2.len());
    let mut meas: Vec<Vec<Vec<f32>>> = Vec::with_capacity(points_g2.len());
    let mut tmp = vec![0.0f32; d];
    let mut delta = vec![0.0f32; vocab];
    for (pi, (_, _, fit_pi)) in points_g2.iter().enumerate() {
        let mut pr = Vec::with_capacity(args.g2_dirs);
        let mut mr = Vec::with_capacity(args.g2_dirs);
        for di in 0..args.g2_dirs {
            r_read_delta_dir_into(
                w.lm_head,
                vocab,
                &fit_all,
                *fit_pi,
                dir_ids[di],
                &mut tmp,
                &mut delta,
            );
            pr.push(delta.clone());
            let row: Vec<f32> = meas_hn[pi][di].iter().map(|x| x / nd).collect();
            mr.push(plain_head_matvec(w.lm_head, vocab, &row));
        }
        preds.push(pr);
        meas.push(mr);
    }
    // Gate per point + the pooled mid band; controls = the off-diagonal
    // ordered pairs (pred_i vs meas_j, i ≠ j).
    let g2j = |r: &G2Report| {
        json!({
            "n": r.n, "n_controls": r.n_controls, "r_mean": r.r_mean,
            "control_mean": r.control_mean, "pass": r.pass,
        })
    };
    let mut g2_points_json = serde_json::Map::new();
    let mut mid_preds: Vec<&[f32]> = Vec::new();
    let mut mid_meas: Vec<&[f32]> = Vec::new();
    let mut mid_cp: Vec<&[f32]> = Vec::new();
    let mut mid_cm: Vec<&[f32]> = Vec::new();
    for (pi, (name, _, fit_pi)) in points_g2.iter().enumerate() {
        let preds_refs: Vec<&[f32]> = preds[pi].iter().map(|v| v.as_slice()).collect();
        let meas_refs: Vec<&[f32]> = meas[pi].iter().map(|v| v.as_slice()).collect();
        let mut cp = Vec::new();
        let mut cm = Vec::new();
        for (i, pred_i) in preds[pi].iter().enumerate().take(args.g2_dirs) {
            for (j, meas_j) in meas[pi].iter().enumerate().take(args.g2_dirs) {
                if i != j {
                    cp.push(pred_i.as_slice());
                    cm.push(meas_j.as_slice());
                }
            }
        }
        let rep = gate_g2(&preds_refs, &meas_refs, &cp, &cm);
        // The per-pair magnitude ratio (how close the linearization is to
        // quantitatively right, beyond the correlation).
        let mut ratios = Vec::new();
        for i in 0..args.g2_dirs {
            let p: f64 = preds[pi][i].iter().map(|x| x.abs() as f64).sum();
            let m: f64 = meas[pi][i].iter().map(|x| x.abs() as f64).sum();
            if m > 0.0 {
                ratios.push(p / m);
            }
        }
        ratios.sort_by(|a, b| a.total_cmp(b));
        let per_pair: Vec<f64> = (0..args.g2_dirs)
            .map(|i| pearson(&preds[pi][i], &meas[pi][i]))
            .collect();
        println!(
            "# G2 {name} (fit point {}): r_mean={:.3} control={:.3} pass={} pairs={:?} ratio_med={:.3}",
            fit_pi,
            rep.r_mean,
            rep.control_mean,
            rep.pass,
            per_pair
                .iter()
                .map(|x| format!("{x:.2}"))
                .collect::<Vec<_>>(),
            ratios.get(ratios.len() / 2).copied().unwrap_or(f64::NAN)
        );
        g2_points_json.insert(
            name.clone(),
            json!({
                "report": g2j(&rep),
                "per_pair_pearson": per_pair,
                "pred_over_meas_abs_median": ratios.get(ratios.len() / 2).copied(),
                "in_mid_band": mid_band.contains(fit_pi),
            }),
        );
        if mid_band.contains(fit_pi) {
            mid_preds.extend(preds_refs.iter().copied());
            mid_meas.extend(meas_refs.iter().copied());
            mid_cp.extend(cp.iter().copied());
            mid_cm.extend(cm.iter().copied());
        }
    }
    let mid_rep = gate_g2(&mid_preds, &mid_meas, &mid_cp, &mid_cm);
    println!(
        "# G2 mid band pooled: r_mean={:.3} control={:.3} pass={} | baselines {baselines} writes {writes} | {:.1}s",
        mid_rep.r_mean,
        mid_rep.control_mean,
        mid_rep.pass,
        t_g2.elapsed().as_secs_f32()
    );

    // ── report ──
    let report = json!({
        "issue": "katgpt-rs 999 T4",
        "fit_dir": args.fit_dir.display().to_string(),
        "gguf": args.gguf.display().to_string(),
        "box_state": if args.box_state.is_empty() {
            json!("(none passed — run ../riir-reflex/scripts/bench_preflight.sh and pass --box-state)")
        } else {
            json!(args.box_state)
        },
        "fit_manifest": {
            "docs": fit_docs, "t": fit_t, "d": fit_d,
            "j_dirs": {
                "count": j_count,
                "sampled": dir_ids.iter().map(|&j| json!({"row": j, "token": j_tokens[j]})).collect::<Vec<_>>(),
            },
            "seed_window": [seed_lo, seed_hi],
            "g3": manifest["g3"],
            "fit_cost": manifest["fit_cost"],
        },
        "held_out": {"skip_docs": skip_docs, "docs": docs.len(), "g2_docs": g2_doc_set.len()},
        "g1": g1_report,
        "g1_bar": "hit ≥ 0.25, control < 0.01 (UNSEEN if no poles — scope finding)",
        "g2": {
            "alpha": args.g2_alpha,
            "points": g2_points_json,
            "mid_band": {"report": g2j(&mid_rep), "bar": "r_mean ≥ 0.45, |control| ≤ 0.04", "pass": mid_rep.pass},
        },
    });
    std::fs::write(
        args.fit_dir.join("t4_gates.json"),
        serde_json::to_string_pretty(&report).context("report serialize")?,
    )?;
    println!("# wrote {}", args.fit_dir.join("t4_gates.json").display());
    Ok(())
}

/// `Δlogits = lm_head · v` — the plain row-major matvec (the measured arm's
/// read; `v` lives in the post-final-norm space the head consumes).
fn plain_head_matvec(lm_head: &[f32], vocab: usize, v: &[f32]) -> Vec<f32> {
    let d = v.len();
    let mut out = vec![0.0f32; vocab];
    for (o, row) in out.iter_mut().zip(lm_head.chunks_exact(d)) {
        let mut s = 0.0f64;
        for (r, &x) in row.iter().zip(v) {
            s += *r as f64 * x as f64;
        }
        *o = s as f32;
    }
    out
}
