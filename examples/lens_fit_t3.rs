//! lens_fit_t3 — katgpt-rs Issue 999 T3: the real-corpus R/S lens fit.
//!
//! Fits the corpus-averaged transposable lenses ([`riir_infer_core::lens`])
//! over a real text corpus on a headful checkpoint (the issue's primary:
//! `Qwen3.5-0.8B-Base-Q8_0`), at the pre-registered protocol shape:
//! two points per layer (`Mid`/`Out`), seed positions `[16, T−1)`, docs
//! split into even/odd halves for the G3 stability gate (the halves' sums
//! add exactly to the full-corpus fit).
//!
//! The fit is offline and compute-bound (the R sweep is
//! `points · d` suffix passes per doc). Docs are chopped from a
//! continuous token stream; each doc is captured once, then its basis
//! sweeps run on a worker scope (direction slices; each worker owns its
//! scratch + accumulators — every (point, direction) slot is written by
//! exactly one worker, so the fit is deterministic regardless of
//! scheduling).
//!
//! ```text
//! cargo run --release --features lens --example lens_fit_t3 -- \
//!   --gguf ../riir-train/data/Qwen3.5-0.8B-Base-Q8_0.gguf \
//!   --corpus-dir ../riir-reflex/.raw/datasets/ag_news \
//!   --docs 1000 --t 128 --out .lens_fit_t3 \
//!   [--seed-lo 16] [--seed-hi 127] [--threads 24] [--shards-max 200]
//!   [--box-state "<PROVENANCE line>"] [--check-direct]
//! ```
//!
//! Artifacts (`--out`): `fit_even.k|j`, `fit_odd.k|j`, `fit_all.k|j`
//! (raw f32, layout in `manifest.json`), the G3 report, the fit cost
//! (G4's record) and the corpus digest.

use anyhow::{Context, Result, bail};
use riir_infer_core::gguf_loader::{GgufFile, load_qwen_deltanet_weights_gguf};
use riir_infer_core::lens::fit::{
    ColAccum, FitCost, G3Report, LensFit, SlicedAccum, gate_g3, mean_embedding,
};
use riir_infer_core::lens::stack::{
    FitPoint, PrefixTangents, StackCapture, StackScratch, StackWeights, stack_forward,
    stack_prefix_jvp, stack_suffix_jvp,
};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::tokenizer::BpeTokenizer;
use serde_json::json;
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

const DEFAULT_GGUF: &str = "../riir-train/data/Qwen3.5-0.8B-Base-Q8_0.gguf";
const DEFAULT_CORPUS_DIR: &str = "../riir-reflex/.raw/datasets/ag_news";

// ── args ────────────────────────────────────────────────────────────────

struct Args {
    gguf: PathBuf,
    corpus_dir: Option<PathBuf>,
    corpus_file: Option<PathBuf>,
    docs: usize,
    t: usize,
    threads: usize,
    seed_lo: usize,
    seed_hi: Option<usize>,
    shards_max: usize,
    out: PathBuf,
    box_state: String,
    check_direct: bool,
    check_samples: usize,
    r_dirs: usize,
    resume: bool,
}

impl Args {
    fn parse() -> Result<Args> {
        let mut a = Args {
            gguf: PathBuf::from(DEFAULT_GGUF),
            corpus_dir: Some(PathBuf::from(DEFAULT_CORPUS_DIR)),
            corpus_file: None,
            docs: 1000,
            t: 128,
            threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(8)
                .saturating_sub(2)
                .max(1),
            seed_lo: 16,
            seed_hi: None,
            shards_max: 200,
            out: PathBuf::from(".lens_fit_t3"),
            box_state: String::new(),
            check_direct: false,
            check_samples: 8,
            r_dirs: 0,
            resume: false,
        };
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let mut val = || it.next().with_context(|| format!("{flag} needs a value"));
            match flag.as_str() {
                "--gguf" => a.gguf = PathBuf::from(val()?),
                "--corpus-dir" => a.corpus_dir = Some(PathBuf::from(val()?)),
                "--corpus" => {
                    a.corpus_file = Some(PathBuf::from(val()?));
                    a.corpus_dir = None;
                }
                "--docs" => a.docs = val()?.parse().context("--docs parse")?,
                "--t" => a.t = val()?.parse().context("--t parse")?,
                "--threads" => a.threads = val()?.parse().context("--threads parse")?,
                "--seed-lo" => a.seed_lo = val()?.parse().context("--seed-lo parse")?,
                "--seed-hi" => a.seed_hi = Some(val()?.parse().context("--seed-hi parse")?),
                "--shards-max" => a.shards_max = val()?.parse().context("--shards-max parse")?,
                "--out" => a.out = PathBuf::from(val()?),
                "--box-state" => a.box_state = val()?,
                "--check-direct" => a.check_direct = true,
                "--check-samples" => {
                    a.check_samples = val()?.parse().context("--check-samples parse")?
                }
                "--r-dirs" => a.r_dirs = val()?.parse().context("--r-dirs parse")?,
                "--resume" => a.resume = true,
                other => bail!(
                    "unknown flag {other} (flags: --gguf --corpus-dir --corpus --docs --t \
                     --threads --seed-lo --seed-hi --shards-max --out --box-state --check-direct \
                     --check-samples --r-dirs --resume)"
                ),
            }
        }
        if a.docs < 2 {
            bail!("--docs must be ≥ 2 (the G3 halves need an even and an odd doc)");
        }
        Ok(a)
    }
}

// ── corpus ──────────────────────────────────────────────────────────────

/// The corpus text rows in read order: ag_news-style shards
/// (`{"rows":[{"row":{"text":...}}]}`) sorted by name, or a plain text
/// file (one synthetic row per 8 KiB chunk keeps the row pipeline
/// uniform).
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

/// Tokenize rows in order until `want_tokens` are collected (a row may
/// cross the budget — the stream is chopped into docs afterwards, so
/// overshoot is free).
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

fn corpus_digest(rows: &[String]) -> String {
    let mut h = blake3::Hasher::new();
    for r in rows {
        h.update(r.as_bytes());
        h.update(b"\n");
    }
    h.finalize().to_hex().to_string()
}

// ── fit sums ────────────────────────────────────────────────────────────

/// Per-matrix f64 sums over docs, split by doc parity; the full-corpus
/// fit is the elementwise sum (means divide by the summed doc count).
/// `dirs` is the direction dimension (K: the full basis `d`; J: the
/// read-direction count), `d` the output dimension.
struct ParitySums {
    even: Vec<f64>,
    odd: Vec<f64>,
    n_even: u64,
    n_odd: u64,
    d: usize,
    dirs: usize,
    points: usize,
}

impl ParitySums {
    fn new(d: usize, dirs: usize, points: usize) -> Self {
        Self {
            even: vec![0.0; points * dirs * d],
            odd: vec![0.0; points * dirs * d],
            n_even: 0,
            n_odd: 0,
            d,
            dirs,
            points,
        }
    }

    fn note_doc(&mut self, parity: usize) {
        if parity == 0 {
            self.n_even += 1;
        } else {
            self.n_odd += 1;
        }
    }

    fn add_doc(&mut self, parity: usize, partial: &[f64]) {
        debug_assert_eq!(partial.len(), self.points * self.dirs * self.d);
        let target = if parity == 0 {
            &mut self.even
        } else {
            &mut self.odd
        };
        for (t, &s) in target.iter_mut().zip(partial) {
            *t += s;
        }
    }

    fn finalize_into(&self, use_even: bool, use_both: bool, out: &mut Vec<Vec<f32>>) {
        let n = if use_both {
            self.n_even + self.n_odd
        } else if use_even {
            self.n_even
        } else {
            self.n_odd
        };
        assert!(n > 0, "finalize on zero docs");
        let d = self.d;
        let dirs = self.dirs;
        let inv = 1.0 / n as f64;
        *out = (0..self.points)
            .map(|p| {
                (0..dirs * d)
                    .map(|i| {
                        let e = self.even[p * dirs * d + i];
                        let o = self.odd[p * dirs * d + i];
                        let s = if use_both {
                            e + o
                        } else if use_even {
                            e
                        } else {
                            o
                        };
                        (s * inv) as f32
                    })
                    .collect()
            })
            .collect();
    }
}

fn write_f32_rows(path: &Path, rows: &[Vec<f32>]) -> Result<u64> {
    let mut f =
        std::fs::File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut bytes = 0u64;
    for r in rows {
        for v in r {
            f.write_all(&v.to_le_bytes())?;
            bytes += 4;
        }
    }
    Ok(bytes)
}

// ── checkpoint / resume (the long run's crash guard) ────────────────────

fn save_checkpoint(dir: &Path, k: &ParitySums, j: &ParitySums, docs_done: usize) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let tmp = dir.join("ckpt.tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        for buf in [&k.even, &k.odd, &j.even, &j.odd] {
            for v in buf {
                f.write_all(&v.to_le_bytes())?;
            }
        }
    }
    let meta = json!({
        "docs_done": docs_done,
        "n_even": k.n_even,
        "n_odd": k.n_odd,
        "k_len": k.even.len(),
        "j_len": j.even.len(),
    });
    std::fs::write(dir.join("ckpt.json"), serde_json::to_string(&meta)?)?;
    std::fs::rename(&tmp, dir.join("ckpt.sums"))?;
    Ok(())
}

/// Returns `(k_len, j_len, n_even, n_odd, docs_done)`.
fn load_checkpoint(
    dir: &Path,
    k: &mut ParitySums,
    j: &mut ParitySums,
) -> Result<(usize, usize, u64, u64, usize)> {
    let path = dir.join("ckpt.sums");
    if !path.exists() {
        return Ok((0, 0, 0, 0, 0));
    }
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("ckpt.json")).context("ckpt.json read")?,
    )?;
    let docs_done = meta["docs_done"].as_u64().context("ckpt docs_done")? as usize;
    let n_even = meta["n_even"].as_u64().context("ckpt n_even")?;
    let n_odd = meta["n_odd"].as_u64().context("ckpt n_odd")?;
    let k_len = meta["k_len"].as_u64().context("ckpt k_len")? as usize;
    let j_len = meta["j_len"].as_u64().context("ckpt j_len")? as usize;
    if k_len != k.even.len() || j_len != j.even.len() {
        bail!(
            "checkpoint shape mismatch: ckpt k={} j={} vs run k={} j={} — the flags changed; \
             delete the checkpoint or restore the flags",
            k_len,
            j_len,
            k.even.len(),
            j.even.len()
        );
    }
    let raw = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    let want = 2 * (k_len + j_len);
    if raw.len() != want * 8 {
        bail!("checkpoint bytes {} != expected {}", raw.len(), want * 8);
    }
    let mut off = 0usize;
    let take = |buf: &mut [f64], raw: &[u8], off: &mut usize| {
        for v in buf.iter_mut() {
            let b: [u8; 8] = raw[*off..*off + 8].try_into().expect("f64 slice");
            *v = f64::from_le_bytes(b);
            *off += 8;
        }
    };
    take(&mut k.even, &raw, &mut off);
    take(&mut k.odd, &raw, &mut off);
    take(&mut j.even, &raw, &mut off);
    take(&mut j.odd, &raw, &mut off);
    k.n_even = n_even;
    k.n_odd = n_odd;
    j.n_even = n_even;
    j.n_odd = n_odd;
    Ok((k_len, j_len, n_even, n_odd, docs_done))
}

// ── the sweep worker ────────────────────────────────────────────────────

/// The R sweep's read-direction set: the full basis (the PoC's fit form —
/// `J` readable in ANY direction later) or `n` explicit model-space
/// vectors (the CPU-scale amendment: top corpus-frequency token
/// embeddings, unit-normalized — the token-detector write class G2
/// steers with).
enum DirSet {
    Basis,
    Explicit(Vec<Vec<f32>>),
}

impl DirSet {
    /// Fill `dv[t·d + j] = v[j]` for the seed rows under direction `k`.
    fn inject(&self, dv: &mut [f32], k: usize, d: usize, t_len: usize, lo: usize, hi: usize) {
        for x in dv.iter_mut() {
            *x = 0.0;
        }
        match self {
            DirSet::Basis => {
                for t in lo..hi.min(t_len) {
                    dv[t * d + k] = 1.0;
                }
            }
            DirSet::Explicit(vs) => {
                let v = &vs[k];
                debug_assert_eq!(v.len(), d, "read direction dim");
                for t in lo..hi.min(t_len) {
                    let row = &mut dv[t * d..t * d + d];
                    row.copy_from_slice(v);
                }
            }
        }
    }
}

/// The worker's slice of the sweep directions + the protocol's seed
/// window.
struct SweepSlice {
    dir_lo: usize,
    dir_len: usize,
    seed_lo: usize,
    seed_hi: usize,
}

/// One worker's K sweep: basis directions `[dir_lo, dir_lo+dir_len)`
/// through the full point grid (seed rows, `sum_all_rows = false`).
fn sweep_k_slice(
    cap: &StackCapture,
    w: &StackWeights,
    points: &[FitPoint],
    sl: SweepSlice,
    acc: &mut SlicedAccum,
) {
    let geo = &w.geo;
    let d = geo.n_embd;
    let t_len = geo.t_len;
    let mut scratch = StackScratch::new(geo);
    let mut pt = PrefixTangents::new(geo);
    let mut d_emb = vec![0.0f32; t_len * d];
    // S sweeps: one prefix pass per basis direction fills a K column for
    // every point at once (the fixed-end law).
    for i in sl.dir_lo..sl.dir_lo + sl.dir_len {
        for v in d_emb.iter_mut() {
            *v = 0.0;
        }
        for t in 0..t_len {
            d_emb[t * d + i] = 1.0;
        }
        stack_prefix_jvp(cap, w, &mut scratch, &d_emb, &mut pt);
        for (pi, p) in points.iter().enumerate() {
            let buf = match p {
                FitPoint::Mid(l) => &pt.mid[*l],
                FitPoint::Out(l) => &pt.out[*l],
            };
            acc.add_doc_column_window(pi, i, buf, sl.seed_lo, sl.seed_hi, false);
        }
    }
}

/// One worker's J sweep: read directions `[dir_lo, dir_lo+dir_len)` of
/// the DirSet, injected at the seed positions only; the response sum
/// runs to the last position (`sum_all_rows = true`).
fn sweep_j_slice(
    cap: &StackCapture,
    w: &StackWeights,
    points: &[FitPoint],
    r_dirs: &DirSet,
    sl: SweepSlice,
    acc: &mut SlicedAccum,
) {
    let geo = &w.geo;
    let d = geo.n_embd;
    let t_len = geo.t_len;
    let mut scratch = StackScratch::new(geo);
    let mut dv = vec![0.0f32; t_len * d];
    let mut dhn = vec![0.0f32; t_len * d];
    for (pi, p) in points.iter().enumerate() {
        for k in sl.dir_lo..sl.dir_lo + sl.dir_len {
            r_dirs.inject(&mut dv, k, d, t_len, sl.seed_lo, sl.seed_hi);
            stack_suffix_jvp(cap, w, &mut scratch, *p, &dv, &mut dhn);
            acc.add_doc_column_window(pi, k, &dhn, sl.seed_lo, sl.seed_hi, true);
        }
    }
}

/// Expand per-worker slice sums into the dense `(point·dirs + dir)·d + j`
/// layout (raw f64, in dir order — slices are disjoint dir ranges).
fn merge_slice_sums(
    slices: Vec<(usize, Vec<Vec<f64>>)>,
    dirs: usize,
    d: usize,
    points: usize,
) -> Vec<f64> {
    let mut dense = vec![0.0f64; points * dirs * d];
    for (lo, per_point) in slices {
        for (p, rows) in per_point.into_iter().enumerate() {
            for (li, row) in rows.chunks_exact(d).enumerate() {
                let dst = (p * dirs + lo + li) * d;
                dense[dst..dst + d].copy_from_slice(row);
            }
        }
    }
    dense
}

/// The fit's shape knobs shared by the threaded sweep driver.
struct FitShape {
    d: usize,
    j_dirs: usize,
    seed_lo: usize,
    seed_hi: usize,
    threads: usize,
}

/// One threaded doc sweep over the captured doc: the K scope splits the
/// full basis, the J scope splits the DirSet; both run all workers.
/// Returns the dense f64 partials (K over the basis, J over the DirSet).
fn sweep_doc_threaded(
    cap: &StackCapture,
    w: &StackWeights,
    points: &[FitPoint],
    r_dirs: &DirSet,
    shape: &FitShape,
) -> (Vec<f64>, Vec<f64>) {
    let d = shape.d;
    let j_dirs = shape.j_dirs;
    let seed = (shape.seed_lo, shape.seed_hi);
    let threads = shape.threads;
    // K scope: basis slices.
    let k_chunk = d.div_ceil(threads);
    let k_slices: Vec<(usize, Vec<Vec<f64>>)> = std::thread::scope(|s| {
        let mut handles = Vec::new();
        let mut lo = 0usize;
        while lo < d {
            let len = k_chunk.min(d - lo);
            let cap_ref = &cap;
            let w_ref = &w;
            let points_ref = &points;
            handles.push(s.spawn(move || {
                let mut acc = SlicedAccum::new(d, points_ref.len(), lo, len);
                sweep_k_slice(
                    cap_ref,
                    w_ref,
                    points_ref,
                    SweepSlice {
                        dir_lo: lo,
                        dir_len: len,
                        seed_lo: seed.0,
                        seed_hi: seed.1,
                    },
                    &mut acc,
                );
                (lo, acc.sums())
            }));
            lo += len;
        }
        handles
            .into_iter()
            .map(|h| h.join().expect("K worker panicked"))
            .collect()
    });
    let k = merge_slice_sums(k_slices, d, d, points.len());
    // J scope: DirSet slices.
    let j_chunk = j_dirs.div_ceil(threads);
    let j_slices: Vec<(usize, Vec<Vec<f64>>)> = std::thread::scope(|s| {
        let mut handles = Vec::new();
        let mut lo = 0usize;
        while lo < j_dirs {
            let len = j_chunk.min(j_dirs - lo);
            let cap_ref = &cap;
            let w_ref = &w;
            let points_ref = &points;
            let dirs_ref = &r_dirs;
            handles.push(s.spawn(move || {
                let mut acc = SlicedAccum::new(d, points_ref.len(), lo, len);
                sweep_j_slice(
                    cap_ref,
                    w_ref,
                    points_ref,
                    dirs_ref,
                    SweepSlice {
                        dir_lo: lo,
                        dir_len: len,
                        seed_lo: seed.0,
                        seed_hi: seed.1,
                    },
                    &mut acc,
                );
                (lo, acc.sums())
            }));
            lo += len;
        }
        handles
            .into_iter()
            .map(|h| h.join().expect("J worker panicked"))
            .collect()
    });
    let j = merge_slice_sums(j_slices, j_dirs, d, points.len());
    (k, j)
}

// ── main ────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse()?;
    let seed_hi = args.seed_hi.unwrap_or(args.t.saturating_sub(1));
    if args.seed_lo >= seed_hi || seed_hi > args.t {
        bail!(
            "seed window [{}, {}) empty or past T={} — the T3 protocol is [16, T−1)",
            args.seed_lo,
            seed_hi,
            args.t
        );
    }

    // ── model + tokenizer ──
    let t_load = Instant::now();
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

    // ── corpus → docs ──
    let rows = match (&args.corpus_dir, &args.corpus_file) {
        (Some(dir), _) => read_rows(dir, args.shards_max)?,
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
    let digest = corpus_digest(&rows);
    let want = args.docs * args.t;
    let t_tok = Instant::now();
    let stream = tokenize_rows(&tok, &rows, want)?;
    println!(
        "# corpus: {} rows → {} tokens ({:.1}s) blake3(rows)={}",
        rows.len(),
        stream.len(),
        t_tok.elapsed().as_secs_f32(),
        &digest[..16]
    );
    let docs: Vec<Vec<usize>> = stream
        .chunks_exact(args.t)
        .take(args.docs)
        .map(|c| c.to_vec())
        .collect();
    if docs.len() < args.docs {
        bail!(
            "only {} full docs from the corpus (want {}) — add shards",
            docs.len(),
            args.docs
        );
    }

    // ── stack weights (the geometry rides the weights, as `from_model` built it) ──
    let rope = RopeFreqTable::new(config.rope_theta, config.head_dim);
    let w = StackWeights::from_model(&weights, &config, rope.as_slice(), args.t);
    let geo = &w.geo;
    let points = geo.fit_points();
    let d = geo.n_embd;

    // ── the R sweep's read-direction set ──
    // `--r-dirs 0` = the full basis (the PoC's fit form; GPU-class cost).
    // `--r-dirs N` = the CPU-scale amendment: the N most frequent corpus
    // tokens' embedding rows, unit-normalized — the token-detector write
    // class G2 steers with. K is ALWAYS the full basis.
    let (r_dir_set, j_dirs, dir_ids): (DirSet, usize, Vec<usize>) = if args.r_dirs == 0 {
        (DirSet::Basis, d, Vec::new())
    } else {
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for &t in &stream {
            *counts.entry(t).or_insert(0) += 1;
        }
        let mut ids: Vec<(usize, usize)> = counts.into_iter().collect();
        ids.sort_unstable_by_key(|&(_, c)| std::cmp::Reverse(c));
        ids.truncate(args.r_dirs * 2);
        let mut vecs = Vec::with_capacity(ids.len());
        let mut kept = Vec::new();
        for (t, _) in ids {
            if t >= geo.vocab {
                continue;
            }
            let row = &w.wte[t * d..(t + 1) * d];
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            let usable = norm.is_finite() && norm > 1e-12;
            if !usable || !row.iter().all(|x| x.is_finite()) {
                continue; // degenerate row — dropped loud below
            }
            vecs.push(row.iter().map(|x| x / norm).collect::<Vec<f32>>());
            kept.push(t);
            if kept.len() == args.r_dirs {
                break;
            }
        }
        if kept.len() < args.r_dirs {
            bail!(
                "only {} non-degenerate read directions from the top-{} corpus tokens",
                kept.len(),
                args.r_dirs * 2
            );
        }
        println!(
            "# r-dirs: {} token-detector directions (top corpus tokens, unit W_E rows): {:?}…",
            kept.len(),
            &kept[..kept.len().min(8)]
        );
        (DirSet::Explicit(vecs), kept.len(), kept)
    };
    println!(
        "# stack: {} layers ({} GDN / {} attn) points={} d={} t={} seeds=[{}, {}) threads={}",
        geo.n_layers,
        geo.layer_is_gdn.iter().filter(|g| **g).count(),
        geo.layer_is_gdn.iter().filter(|g| !**g).count(),
        points.len(),
        d,
        geo.t_len,
        args.seed_lo,
        seed_hi,
        args.threads,
    );

    // ── pilot timings (one capture + one prefix + one suffix pass) ──
    {
        let mut cap = StackCapture::new(geo);
        let t0 = Instant::now();
        stack_forward(&docs[0], &w, &mut cap);
        let cap_s = t0.elapsed().as_secs_f64();
        let mut scratch = StackScratch::new(geo);
        let mut pt = PrefixTangents::new(geo);
        let mut d_emb = vec![0.0f32; args.t * d];
        d_emb[args.t * d - 1] = 1.0;
        let t1 = Instant::now();
        stack_prefix_jvp(&cap, &w, &mut scratch, &d_emb, &mut pt);
        let prefix_s = t1.elapsed().as_secs_f64();
        let mut dv = vec![0.0f32; args.t * d];
        for t in args.seed_lo..seed_hi {
            dv[t * d + d / 2] = 1.0;
        }
        let mut dhn = vec![0.0f32; args.t * d];
        let t2 = Instant::now();
        stack_suffix_jvp(
            &cap,
            &w,
            &mut scratch,
            FitPoint::Mid(geo.n_layers / 2),
            &dv,
            &mut dhn,
        );
        let suffix_s = t2.elapsed().as_secs_f64();
        let per_doc = cap_s + d as f64 * prefix_s + points.len() as f64 * d as f64 * suffix_s;
        println!(
            "# pilot: capture {:.2}s | prefix {:.2}s | suffix(mid) {:.2}s → projected {:.1} min/doc, {:.1} h for {} docs",
            cap_s,
            prefix_s,
            suffix_s,
            per_doc / 60.0,
            per_doc * args.docs as f64 / 3600.0,
            args.docs
        );
    }

    // ── the fit ──
    let mut sums_k = ParitySums::new(d, d, points.len());
    let mut sums_j = ParitySums::new(d, j_dirs, points.len());
    let mut cost = FitCost {
        docs: 0,
        basis_dirs: d,
        prefix_passes: 0,
        suffix_passes: 0,
        capture_secs: 0.0,
        sweep_secs: 0.0,
        box_state: if args.box_state.is_empty() {
            format!(
                "amo M5 Ultra 96GB AC threads={} (no bench_preflight PROVENANCE passed)",
                args.threads
            )
        } else {
            args.box_state.clone()
        },
    };
    let mut docs_done = 0usize;
    let mut start_docs = 0usize;
    // Resume: reload the parity sums + counters from the checkpoint (the
    // checkpoint is authoritative — a crash loses at most
    // `--ckpt-every` docs of work).
    if args.resume {
        let (nk, nj, ne, no, done) = load_checkpoint(&args.out, &mut sums_k, &mut sums_j)?;
        cost.prefix_passes = (done as u64) * (d as u64);
        cost.suffix_passes = (done as u64) * (points.len() * j_dirs) as u64;
        cost.docs = done as u64;
        let _ = (nk, nj, ne, no);
        docs_done = done;
        start_docs = done;
        println!(
            "# resume: {} docs already fitted (even {} / odd {}) — continuing at doc {}",
            done, sums_k.n_even, sums_k.n_odd, done
        );
    }
    let ckpt_every: usize = std::env::var("LENS_T3_CKPT_EVERY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);
    let t_fit = Instant::now();
    for (di, tokens) in docs.iter().enumerate().skip(docs_done) {
        let parity = di % 2;
        let tc = Instant::now();
        let mut cap = StackCapture::new(geo);
        stack_forward(tokens, &w, &mut cap);
        cost.capture_secs += tc.elapsed().as_secs_f64();
        let ts = Instant::now();
        let (pk, pj) = sweep_doc_threaded(
            &cap,
            &w,
            &points,
            &r_dir_set,
            &FitShape {
                d,
                j_dirs,
                seed_lo: args.seed_lo,
                seed_hi,
                threads: args.threads,
            },
        );
        cost.sweep_secs += ts.elapsed().as_secs_f64();
        sums_k.note_doc(parity);
        sums_j.note_doc(parity);
        sums_k.add_doc(parity, &pk);
        sums_j.add_doc(parity, &pj);
        cost.docs += 1;
        cost.prefix_passes += d as u64;
        cost.suffix_passes += (points.len() * j_dirs) as u64;
        docs_done += 1;
        let this_run = docs_done - start_docs;
        if this_run.is_multiple_of(5) || docs_done == args.docs {
            let el = t_fit.elapsed().as_secs_f64();
            println!(
                "# [{}/{}] {:.1}s/doc avg this run, eta {:.1} h",
                docs_done,
                args.docs,
                el / this_run as f64,
                el / this_run as f64 * (args.docs - docs_done) as f64 / 3600.0
            );
        }
        if docs_done.is_multiple_of(ckpt_every) || docs_done == args.docs {
            save_checkpoint(&args.out, &sums_k, &sums_j, docs_done)?;
        }
    }

    // ── sampled bit-exact direct check (doc 0, single-threaded) ──
    if args.check_direct {
        println!(
            "# check-direct: single-threaded re-run of doc 0 over {} sampled dirs (K) and all {} J dirs",
            args.check_samples, j_dirs
        );
        let mut cap = StackCapture::new(geo);
        stack_forward(&docs[0], &w, &mut cap);
        // Evenly spaced sample of the basis for K; ALL J directions (they
        // are already the restricted set).
        let k_step = d / args.check_samples.max(1);
        let k_ids: Vec<usize> = (0..args.check_samples).map(|i| i * k_step).collect();
        let mut dense_k = ColAccum::new(d, points.len());
        let mut dense_j = SlicedAccum::new(d, points.len(), 0, j_dirs);
        dense_k.note_doc();
        let mut scratch = StackScratch::new(geo);
        let mut pt = PrefixTangents::new(geo);
        let mut d_emb = vec![0.0f32; args.t * d];
        let mut dv = vec![0.0f32; args.t * d];
        let mut dhn = vec![0.0f32; args.t * d];
        for &i in &k_ids {
            for v in d_emb.iter_mut() {
                *v = 0.0;
            }
            for t in 0..args.t {
                d_emb[t * d + i] = 1.0;
            }
            stack_prefix_jvp(&cap, &w, &mut scratch, &d_emb, &mut pt);
            for (pi, p) in points.iter().enumerate() {
                let buf = match p {
                    FitPoint::Mid(l) => &pt.mid[*l],
                    FitPoint::Out(l) => &pt.out[*l],
                };
                dense_k.add_doc_column_window(pi, i, buf, args.seed_lo, seed_hi, false);
            }
        }
        for (pi, p) in points.iter().enumerate() {
            for k in 0..j_dirs {
                r_dir_set.inject(&mut dv, k, d, geo.t_len, args.seed_lo, seed_hi);
                stack_suffix_jvp(&cap, &w, &mut scratch, *p, &dv, &mut dhn);
                dense_j.add_doc_column_window(pi, k, &dhn, args.seed_lo, seed_hi, true);
            }
        }
        // Threaded re-run of doc 0, restricted to the same samples.
        let (pk, pj) = sweep_doc_threaded(
            &cap,
            &w,
            &points,
            &r_dir_set,
            &FitShape {
                d,
                j_dirs,
                seed_lo: args.seed_lo,
                seed_hi,
                threads: args.threads,
            },
        );
        let want_k = dense_k.finalize();
        let want_j = dense_j.finalize(1);
        for (pi, _p) in points.iter().enumerate() {
            for &i in &k_ids {
                for j in 0..d {
                    let a = want_k[pi][i * d + j];
                    let b = pk[(pi * d + i) * d + j] as f32;
                    assert!(
                        a == b,
                        "check-direct: K point {pi} dir {i} j {j}: {a} vs {b}"
                    );
                }
            }
            for k in 0..j_dirs {
                for j in 0..d {
                    let a = want_j[pi][k * d + j];
                    let b = pj[(pi * j_dirs + k) * d + j] as f32;
                    assert!(
                        a == b,
                        "check-direct: J point {pi} dir {k} j {j}: {a} vs {b}"
                    );
                }
            }
        }
        println!(
            "# check-direct: OK (K sampled ×{} + J full, bit-exact across {} points)",
            args.check_samples,
            points.len()
        );
    }

    // ── finalize: even / odd / all fits ──
    let mean_emb = mean_embedding(w.wte, geo.vocab, d);
    let mut fit_even = LensFit {
        d,
        points: points.clone(),
        k: Vec::new(),
        j: Vec::new(),
        mean_emb: mean_emb.clone(),
        n_docs: sums_k.n_even,
    };
    let mut fit_odd = LensFit {
        d,
        points: points.clone(),
        k: Vec::new(),
        j: Vec::new(),
        mean_emb: mean_emb.clone(),
        n_docs: sums_k.n_odd,
    };
    let mut fit_all = LensFit {
        d,
        points: points.clone(),
        k: Vec::new(),
        j: Vec::new(),
        mean_emb,
        n_docs: sums_k.n_even + sums_k.n_odd,
    };
    sums_k.finalize_into(true, false, &mut fit_even.k);
    sums_j.finalize_into(true, false, &mut fit_even.j);
    sums_k.finalize_into(false, false, &mut fit_odd.k);
    sums_j.finalize_into(false, false, &mut fit_odd.j);
    sums_k.finalize_into(true, true, &mut fit_all.k);
    sums_j.finalize_into(true, true, &mut fit_all.j);

    // ── G3 ──
    let g3: G3Report = gate_g3(&fit_even, &fit_odd);

    // ── persist ──
    std::fs::create_dir_all(&args.out).with_context(|| format!("mkdir {}", args.out.display()))?;
    let mut files = serde_json::Map::new();
    for (name, fit) in [
        ("fit_even", &fit_even),
        ("fit_odd", &fit_odd),
        ("fit_all", &fit_all),
    ] {
        let bytes_k = write_f32_rows(&args.out.join(format!("{name}.k")), &fit.k)?;
        let bytes_j = write_f32_rows(&args.out.join(format!("{name}.j")), &fit.j)?;
        files.insert(
            name.to_string(),
            json!({"k_bytes": bytes_k, "j_bytes": bytes_j, "n_docs": fit.n_docs}),
        );
    }
    let manifest = json!({
        "issue": "katgpt-rs 999 T3",
        "gguf": args.gguf.display().to_string(),
        "corpus_rows": rows.len(),
        "corpus_digest_blake3": digest,
        "docs": cost.docs,
        "t": geo.t_len,
        "seed_window": [args.seed_lo, seed_hi],
        "d": d,
        "points": points.iter().map(|p| match p {
            FitPoint::Mid(l) => json!({"mid": l}),
            FitPoint::Out(l) => json!({"out": l}),
        }).collect::<Vec<_>>(),
        "k_dirs": {"mode": "basis", "count": d},
        "j_dirs": if dir_ids.is_empty() {
            json!({"mode": "basis", "count": j_dirs})
        } else {
            json!({"mode": "embed_top_freq_unitnorm", "count": j_dirs, "token_ids": dir_ids})
        },
        "layout": "each fit_*.k / fit_*.j = points.len() blocks of (dirs·d) f32 LE, row-major [dir][j]; block order = points[]",
        "files": files,
        "fit_cost": {
            "docs": cost.docs,
            "basis_dirs": cost.basis_dirs,
            "prefix_passes": cost.prefix_passes,
            "suffix_passes": cost.suffix_passes,
            "capture_secs": cost.capture_secs,
            "sweep_secs": cost.sweep_secs,
            "box_state": cost.box_state,
        },
        "g3": {
            "points": g3.points,
            "max_rel_k": g3.max_rel_k,
            "max_rel_j": g3.max_rel_j,
            "pass": g3.pass,
        },
    });
    std::fs::write(
        args.out.join("manifest.json"),
        serde_json::to_string_pretty(&manifest).context("manifest serialize")?,
    )?;
    println!(
        "# G3: max_rel_k={:.4} max_rel_j={:.4} pass={} (bar ≤ 0.06)",
        g3.max_rel_k, g3.max_rel_j, g3.pass
    );
    println!(
        "# cost: {} docs, prefix {} / suffix {} passes, capture {:.1}s sweep {:.1}s",
        cost.docs, cost.prefix_passes, cost.suffix_passes, cost.capture_secs, cost.sweep_secs
    );
    println!("# wrote {} (manifest.json + fit_*.k|j)", args.out.display());
    Ok(())
}
