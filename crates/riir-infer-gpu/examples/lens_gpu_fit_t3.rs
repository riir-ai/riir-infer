//! lens_gpu_fit_t3 — katgpt-rs Issue 999 T3 follow-up (riir-infer Issue
//! 1222): the full-basis R-sweep fit on the GPU.
//!
//! Same protocol as [`riir_infer_core`]'s `lens_fit_t3` example (two points
//! per layer, seed positions `[16, T−1)`, even/odd halves for G3), with the
//! J sweep (the R lens — `points · dirs` suffix passes, the GPU-class bulk)
//! running on [`LensBatchStack`] direction blocks and the K sweep (the S
//! lens — d prefix passes, ~1/25 of the cost) staying on CPU worker
//! threads. The GPU J column sums are exact-linear in the directions, so
//! the fit artifact is the same estimator; the cross-backend fp delta is
//! the pullback-cancellation class documented in
//! [`riir_infer_gpu::lens_batch_kernels`] (offline diagnostics, never a
//! serving path).
//!
//! ```text
//! cargo run --release --features lens_gpu --example lens_gpu_fit_t3 -- \
//!   --gguf ../riir-train/data/Qwen3.5-0.8B-Base-Q8_0.gguf \
//!   --corpus-dir ../riir-reflex/.raw/datasets/ag_news \
//!   --docs 1000 --t 128 --dim-batch 128 \
//!   [--k-threads 16] [--out .lens_gpu_fit_t3] [--box-state "..."] [--no-resume]
//! ```
//!
//! `--r-dirs 0` (the default) = the full basis — the point of this lane.
//!
//! Checkpoint/resume (the compute law: >20 min runs resume, never
//! restart): after every doc the accumulators flush to
//! `<out>/gen_{a,b}/` + an atomic `LATEST` pointer, so a kill or cancel
//! loses at most one doc. A re-run over the SAME corpus/shape loads the
//! checkpoint and continues with the identical accumulation order
//! (bit-identical to an uninterrupted run); a mismatch refuses LOUD
//! (pass `--no-resume` to start fresh). Born of the 2026-11-18
//! deciding-run cancel that lost 3 completed docs (issue 1223).

use anyhow::{Context, Result, bail};
use riir_infer_core::gguf_loader::{GgufFile, load_qwen_deltanet_weights_gguf};
use riir_infer_core::lens::fit::{FitCost, G3Report, LensFit, gate_g3, mean_embedding};
use riir_infer_core::lens::stack::{
    FitPoint, PrefixTangents, StackCapture, StackScratch, StackWeights, stack_forward,
    stack_prefix_jvp,
};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_gpu::cubecl_runtime::{CubeCLContext, create_f32};
use riir_infer_gpu::lens_batch::{LensBatchStack, LensDocCapture, suffix_pass};
use serde_json::json;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

const DEFAULT_GGUF: &str = "../riir-train/data/Qwen3.5-0.8B-Base-Q8_0.gguf";
const DEFAULT_CORPUS_DIR: &str = "../riir-reflex/.raw/datasets/ag_news";

struct Args {
    gguf: PathBuf,
    corpus_dir: Option<PathBuf>,
    docs: usize,
    t: usize,
    dim_batch: usize,
    k_threads: usize,
    seed_lo: usize,
    out: PathBuf,
    box_state: String,
    no_resume: bool,
}

impl Args {
    fn parse() -> Result<Args> {
        let mut a = Args {
            gguf: PathBuf::from(DEFAULT_GGUF),
            corpus_dir: Some(PathBuf::from(DEFAULT_CORPUS_DIR)),
            docs: 1000,
            t: 128,
            dim_batch: 128,
            k_threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(8),
            seed_lo: 16,
            out: PathBuf::from(".lens_gpu_fit_t3"),
            box_state: String::new(),
            no_resume: false,
        };
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--gguf" => a.gguf = PathBuf::from(it.next().context("--gguf value")?),
                "--corpus-dir" => {
                    a.corpus_dir = Some(PathBuf::from(it.next().context("--corpus-dir value")?))
                }
                "--docs" => a.docs = it.next().context("--docs value")?.parse()?,
                "--t" => a.t = it.next().context("--t value")?.parse()?,
                "--dim-batch" => a.dim_batch = it.next().context("--dim-batch value")?.parse()?,
                "--k-threads" => a.k_threads = it.next().context("--k-threads value")?.parse()?,
                "--seed-lo" => a.seed_lo = it.next().context("--seed-lo value")?.parse()?,
                "--out" => a.out = PathBuf::from(it.next().context("--out value")?),
                "--box-state" => a.box_state = it.next().context("--box-state value")?,
                "--no-resume" => a.no_resume = true,
                other => bail!("unknown arg {other} (see the module docs)"),
            }
        }
        Ok(a)
    }
}

/// Deterministic LCG for the G3 split (the parity of the doc index).
fn parity_of(di: usize) -> usize {
    di % 2
}

fn read_rows(dir: &Path, shards_max: usize) -> Result<Vec<String>> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("read_dir {}", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "jsonl" || e == "json"))
        .collect();
    paths.sort();
    paths.truncate(shards_max);
    let mut rows = Vec::new();
    for p in &paths {
        let text = std::fs::read_to_string(p).with_context(|| format!("read {}", p.display()))?;
        for line in text.lines().take(10_000) {
            if line.len() > 80 {
                rows.push(line.to_string());
            }
        }
    }
    anyhow::ensure!(!rows.is_empty(), "no rows from {}", dir.display());
    Ok(rows)
}

fn tokenize_rows(tok: &BpeTokenizer, rows: &[String], want_tokens: usize) -> Result<Vec<usize>> {
    let mut stream = Vec::with_capacity(want_tokens + 1024);
    for r in rows {
        if stream.len() >= want_tokens {
            break;
        }
        let ids = tok.encode(r);
        stream.extend(ids);
    }
    anyhow::ensure!(
        stream.len() >= want_tokens,
        "corpus produced {} tokens < {} wanted",
        stream.len(),
        want_tokens
    );
    stream.truncate(want_tokens);
    Ok(stream)
}

fn corpus_digest(rows: &[String]) -> String {
    use blake3::Hasher;
    let mut h = Hasher::new();
    for r in rows {
        h.update(r.as_bytes());
        h.update(b"\n");
    }
    h.finalize().to_hex().to_string()
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

/// Per-doc checkpoint: two alternating generation directories + an atomic
/// `LATEST` pointer. A crash mid-flush leaves `LATEST` naming the previous
/// COMPLETE generation — the pointer only flips after every file of the
/// new generation is on disk.
fn flush_partial(
    dir: &Path,
    generation: &str,
    state: &serde_json::Value,
    accs: &[(&str, &[Vec<f64>])],
) -> Result<()> {
    let gdir = dir.join(generation);
    std::fs::create_dir_all(&gdir)?;
    for (name, rows) in accs {
        let tmp = gdir.join(format!("{name}.bin.tmp"));
        {
            let mut f =
                std::fs::File::create(&tmp).with_context(|| format!("create {}", tmp.display()))?;
            for r in *rows {
                for v in r {
                    f.write_all(&v.to_le_bytes())?;
                }
            }
            f.sync_all().ok();
        }
        std::fs::rename(&tmp, gdir.join(format!("{name}.bin")))?;
    }
    let stmp = gdir.join("state.json.tmp");
    std::fs::write(&stmp, serde_json::to_string(state)?)?;
    std::fs::rename(&stmp, gdir.join("state.json"))?;
    let ltmp = dir.join("LATEST.tmp");
    std::fs::write(&ltmp, generation)?;
    std::fs::rename(&ltmp, dir.join("LATEST"))?;
    Ok(())
}

/// Load the checkpoint generation `LATEST` names. `None` when no
/// checkpoint exists; a PRESENT-but-unreadable checkpoint is a hard error
/// (the operator decides — never a silent fresh start of a multi-hour run).
fn load_partial(dir: &Path) -> Result<Option<(serde_json::Value, Vec<Vec<f64>>)>> {
    let latest = dir.join("LATEST");
    if !latest.exists() {
        return Ok(None);
    }
    let generation = std::fs::read_to_string(&latest)?.trim().to_string();
    anyhow::ensure!(
        !generation.is_empty(),
        "empty LATEST pointer in {}",
        dir.display()
    );
    let gdir = dir.join(&generation);
    let state: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(gdir.join("state.json"))?)
            .with_context(|| format!("corrupt checkpoint state in {}", gdir.display()))?;
    let mut accs = Vec::new();
    for name in ["k_even", "j_even", "k_odd", "j_odd"] {
        let bytes = std::fs::read(gdir.join(format!("{name}.bin")))
            .with_context(|| format!("checkpoint bin {name} unreadable in {}", gdir.display()))?;
        accs.push(
            bytes
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().expect("8 bytes")))
                .collect::<Vec<f64>>(),
        );
    }
    Ok(Some((state, accs)))
}

fn main() -> Result<()> {
    let args = Args::parse()?;
    let seed_hi = args.t.saturating_sub(1);
    anyhow::ensure!(
        args.seed_lo < seed_hi && seed_hi <= args.t,
        "seed window [{}, {}) empty — the T3 protocol is [16, T−1)",
        args.seed_lo,
        seed_hi
    );
    let dim_batch = args.dim_batch;

    // ── GPU context (before the model load: fail fast on a GPU-less box) ──
    let Ok(ctx) = CubeCLContext::new() else {
        bail!("no GPU device — the lens GPU lane needs one (Issue 1222)");
    };
    let client = ctx.client();

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
    let rows = match &args.corpus_dir {
        Some(dir) => read_rows(dir, 200)?,
        None => bail!("pass --corpus-dir"),
    };
    let digest = corpus_digest(&rows);
    let want = args.docs * args.t;
    let stream = tokenize_rows(&tok, &rows, want)?;
    println!(
        "# corpus: {} rows → {} tokens for {} docs, blake3(rows)={}",
        rows.len(),
        stream.len(),
        args.docs,
        &digest[..16]
    );
    let docs: Vec<Vec<usize>> = stream
        .chunks_exact(args.t)
        .take(args.docs)
        .map(|c| c.to_vec())
        .collect();

    // ── stack weights ──
    let rope = RopeFreqTable::new(config.rope_theta, config.head_dim);
    let w = StackWeights::from_model(&weights, &config, rope.as_slice(), args.t);
    let geo = w.geo.clone();
    let points = geo.fit_points();
    let d = geo.n_embd;
    println!(
        "# stack: {} layers ({} GDN / {} attn) points={} d={} t={} seeds=[{}, {}) dim_batch={}",
        geo.n_layers,
        geo.layer_is_gdn.iter().filter(|g| **g).count(),
        geo.layer_is_gdn.iter().filter(|g| !**g).count(),
        points.len(),
        d,
        geo.t_len,
        args.seed_lo,
        seed_hi,
        dim_batch,
    );
    let blocks = d.div_ceil(dim_batch);
    println!("# J sweep: {blocks} direction blocks of ≤{dim_batch} on the GPU");

    // ── the fit accumulators (even/odd halves for G3) ──
    // Layout: [points][dirs·d] f64 sums, J over the full basis, K likewise.
    let mut j_even = vec![vec![0.0f64; d * d]; points.len()];
    let mut j_odd = vec![vec![0.0f64; d * d]; points.len()];
    let mut k_even = vec![vec![0.0f64; d * d]; points.len()];
    let mut k_odd = vec![vec![0.0f64; d * d]; points.len()];
    let (mut n_even, mut n_odd) = (0u64, 0u64);
    let mut cost = FitCost {
        docs: 0,
        basis_dirs: d,
        prefix_passes: 0,
        suffix_passes: 0,
        capture_secs: 0.0,
        sweep_secs: 0.0,
        box_state: if args.box_state.is_empty() {
            "unrecorded (--box-state)".to_string()
        } else {
            args.box_state.clone()
        },
    };
    let t_fit = Instant::now();

    // The K sweep's CPU scratch (built once per doc, below).
    let mut k_scratch = StackScratch::new(&geo);
    let mut k_pt = PrefixTangents::new(&geo);
    let mut d_emb = vec![0.0f32; args.t * d];
    let mut cap = StackCapture::new(&geo);
    let mut gpu_stack = None;

    // ── resume (the compute law: >20 min runs resume, never restart) ──
    let mut docs_done = 0usize;
    let mut gen_toggle = false;
    if args.no_resume {
        println!("# resume: disabled (--no-resume) — starting fresh");
    } else if let Some((state, accs)) = load_partial(&args.out)? {
        let matches = state["corpus_digest"].as_str() == Some(digest.as_str())
            && state["docs"].as_u64() == Some(args.docs as u64)
            && state["t"].as_u64() == Some(args.t as u64)
            && state["seed_lo"].as_u64() == Some(args.seed_lo as u64)
            && state["seed_hi"].as_u64() == Some(seed_hi as u64)
            && state["dim_batch"].as_u64() == Some(dim_batch as u64)
            && state["points"].as_u64() == Some(points.len() as u64)
            && state["d"].as_u64() == Some(d as u64);
        let expect = points.len() * d * d;
        let bins_ok = accs.iter().all(|a| a.len() == expect);
        if !matches || !bins_ok {
            bail!(
                "checkpoint in {} does not match this run (digest/docs/t/seed/dim_batch/points/d or bin lengths) — pass --no-resume to start fresh",
                args.out.display()
            );
        }
        let get =
            |i: usize| -> Vec<Vec<f64>> { accs[i].chunks(d * d).map(|c| c.to_vec()).collect() };
        k_even = get(0);
        j_even = get(1);
        k_odd = get(2);
        j_odd = get(3);
        n_even = state["n_even"].as_u64().context("checkpoint n_even")?;
        n_odd = state["n_odd"].as_u64().context("checkpoint n_odd")?;
        docs_done = state["docs_done"]
            .as_u64()
            .context("checkpoint docs_done")? as usize;
        cost.docs = state["cost_docs"].as_u64().unwrap_or(0) as usize;
        cost.prefix_passes = state["cost_prefix_passes"].as_u64().unwrap_or(0);
        cost.suffix_passes = state["cost_suffix_passes"].as_u64().unwrap_or(0);
        cost.capture_secs = state["cost_capture_secs"].as_f64().unwrap_or(0.0);
        cost.sweep_secs = state["cost_sweep_secs"].as_f64().unwrap_or(0.0);
        gen_toggle = docs_done % 2 == 1;
        println!(
            "# resume: docs_done={} n_even={} n_odd={} from {} — skipping the first {} docs (bit-identical accumulation order)",
            docs_done,
            n_even,
            n_odd,
            args.out.display(),
            docs_done
        );
    } else {
        println!(
            "# resume: no checkpoint in {} — starting fresh",
            args.out.display()
        );
    }

    for (di, tokens) in docs.iter().enumerate().skip(docs_done) {
        let tc = Instant::now();
        stack_forward(tokens, &w, &mut cap);
        cost.capture_secs += tc.elapsed().as_secs_f64();
        let parity = parity_of(di);

        // ── K sweep (CPU workers, d prefix passes) ──
        let ts = Instant::now();
        {
            let k_chunk = d.div_ceil(args.k_threads);
            let mut lo = 0usize;
            let worker_sums = std::thread::scope(|s: &std::thread::Scope<'_, '_>| {
                let mut hs = Vec::new();
                while lo < d {
                    let len = k_chunk.min(d - lo);
                    let cap_ref = &cap;
                    let w_ref = &w;
                    let points_ref = &points;
                    let seed = (args.seed_lo, seed_hi);
                    hs.push(s.spawn(move || {
                        let mut sums = vec![vec![0.0f64; len * d]; points_ref.len()];
                        let mut scratch = StackScratch::new(&w_ref.geo);
                        let mut pt = PrefixTangents::new(&w_ref.geo);
                        let mut d_emb = vec![0.0f32; w_ref.geo.t_len * w_ref.geo.n_embd];
                        for i in lo..lo + len {
                            for v in d_emb.iter_mut() {
                                *v = 0.0;
                            }
                            for t in 0..w_ref.geo.t_len {
                                d_emb[t * w_ref.geo.n_embd + i] = 1.0;
                            }
                            stack_prefix_jvp(cap_ref, w_ref, &mut scratch, &d_emb, &mut pt);
                            for (pi, p) in points_ref.iter().enumerate() {
                                let buf = match p {
                                    FitPoint::Mid(l) => &pt.mid[*l],
                                    FitPoint::Out(l) => &pt.out[*l],
                                };
                                // K: seed-window mean (the S lens's row rule).
                                let inv_w = 1.0 / (seed.1 - seed.0) as f64;
                                let local = i - lo;
                                for j in 0..w_ref.geo.n_embd {
                                    let mut col = 0.0f64;
                                    for t in seed.0..seed.1 {
                                        col += buf[t * w_ref.geo.n_embd + j] as f64;
                                    }
                                    sums[pi][local * w_ref.geo.n_embd + j] += col * inv_w;
                                }
                            }
                        }
                        (lo, sums)
                    }));
                    lo += len;
                }
                hs.into_iter()
                    .map(|h| h.join().expect("K worker panicked"))
                    .collect::<Vec<_>>()
            });
            for (slo, sums) in worker_sums {
                let len = k_chunk.min(d - slo);
                let target = if parity == 0 { &mut k_even } else { &mut k_odd };
                for (pi, rows_p) in sums.iter().enumerate() {
                    for (li, row) in rows_p.chunks_exact(d).enumerate() {
                        // target[pi] is per-point [d × d]: row = global emb
                        // row (slo + li), col = j — pi selects the buffer,
                        // never the row (pi*d here overflows every point ≥ 1).
                        let dst = (slo + li) * d;
                        for (j, &v) in row.iter().enumerate() {
                            target[pi][dst + j] += v;
                        }
                    }
                }
                let _ = len;
            }
        }
        let _ = (&mut k_scratch, &mut k_pt, &mut d_emb); // reserved (unused in the worker-scope form)

        // ── J sweep (GPU direction blocks) ──
        let gpu = gpu_stack.get_or_insert_with(|| LensBatchStack::new(&client, &w, dim_batch));
        let doc_h = LensDocCapture::upload(&client, &cap);
        let mut dv = vec![0.0f32; dim_batch * geo.t_len * d];
        let mut gpu_sums = vec![vec![0.0f64; d * d]; points.len()];
        let ts_gpu = Instant::now();
        for block in 0..blocks {
            let dir_lo = block * dim_batch;
            let dir_len = dim_batch.min(d - dir_lo);
            // Build the block's injected tangent: basis dir k at row k.
            for v in dv.iter_mut() {
                *v = 0.0;
            }
            for k in 0..dir_len {
                for t in args.seed_lo..seed_hi {
                    dv[(k * geo.t_len + t) * d + dir_lo + k] = 1.0;
                }
            }
            let dv_h = create_f32(&client, &dv);
            for (pi, p) in points.iter().enumerate() {
                let sums = unsafe { suffix_pass(&client, gpu, &doc_h, &dv_h, *p, 0, geo.t_len) }
                    .expect("GPU suffix readback");
                // Row rule: sum_all_rows (the J lens sums the response over
                // EVERY position) with the seed-window divisor (1/(hi−lo)) —
                // the accumulator's exact semantics.
                let inv_w = 1.0 / (seed_hi - args.seed_lo) as f64;
                for k in 0..dir_len {
                    // gpu_sums[pi] is per-point [d × d]: row = direction
                    // index (dir_lo + k), col = j — pi selects the buffer.
                    let base = (dir_lo + k) * d;
                    for (j, &s) in sums[k * d..(k + 1) * d].iter().enumerate() {
                        gpu_sums[pi][base + j] += (s as f64) * inv_w;
                    }
                }
                cost.suffix_passes += dir_len as u64;
            }
        }
        let target_j = if parity == 0 { &mut j_even } else { &mut j_odd };
        for (pi, rows_p) in gpu_sums.iter().enumerate() {
            for (idx, &v) in rows_p.iter().enumerate() {
                target_j[pi][idx] += v;
            }
        }
        let sweep_wall = ts.elapsed().as_secs_f64();
        let _ = ts_gpu;
        cost.sweep_secs += sweep_wall;
        cost.prefix_passes += d as u64;
        cost.docs += 1;
        if parity == 0 {
            n_even += 1;
        } else {
            n_odd += 1;
        }
        let el = t_fit.elapsed().as_secs_f64();
        println!(
            "# [{}/{}] {:.1}s/doc avg (sweep wall {:.1}s), eta {:.1} h",
            di + 1,
            args.docs,
            el / (di + 1) as f64,
            sweep_wall,
            el / (di + 1) as f64 * (args.docs - di - 1) as f64 / 3600.0
        );

        // Per-doc checkpoint — the alternating-generation flip is
        // crash-safe: LATEST always names a complete generation.
        gen_toggle = !gen_toggle;
        let generation = if gen_toggle { "gen_b" } else { "gen_a" };
        let state = json!({
            "format": "lens_gpu_fit_t3_partial_v1",
            "corpus_digest": digest,
            "docs": args.docs,
            "t": args.t,
            "seed_lo": args.seed_lo,
            "seed_hi": seed_hi,
            "dim_batch": dim_batch,
            "points": points.len(),
            "d": d,
            "docs_done": di + 1,
            "n_even": n_even,
            "n_odd": n_odd,
            "cost_docs": cost.docs,
            "cost_prefix_passes": cost.prefix_passes,
            "cost_suffix_passes": cost.suffix_passes,
            "cost_capture_secs": cost.capture_secs,
            "cost_sweep_secs": cost.sweep_secs,
        });
        flush_partial(
            &args.out,
            generation,
            &state,
            &[
                ("k_even", &k_even),
                ("j_even", &j_even),
                ("k_odd", &k_odd),
                ("j_odd", &j_odd),
            ],
        )?;
        println!(
            "# checkpoint: docs_done={} → {} in {}",
            di + 1,
            generation,
            args.out.display()
        );
    }
    drop(cap);
    drop(k_scratch);
    drop(k_pt);
    drop(d_emb);

    // ── finalize: even / odd / all fits ──
    let finalize = |sums: &[Vec<f64>], n: u64| -> Vec<Vec<f32>> {
        let inv = 1.0 / n as f64;
        sums.iter()
            .map(|p| p.iter().map(|&s| (s * inv) as f32).collect())
            .collect()
    };
    let mean_emb = mean_embedding(w.wte, geo.vocab, d);
    let make_fit = |k: &[Vec<f32>], j: &[Vec<f32>], n: u64| -> LensFit {
        LensFit {
            d,
            points: points.clone(),
            k: k.to_vec(),
            j: j.to_vec(),
            mean_emb: mean_emb.clone(),
            n_docs: n,
        }
    };
    let fit_even = make_fit(
        &finalize(&k_even, n_even),
        &finalize(&j_even, n_even),
        n_even,
    );
    let fit_odd = make_fit(&finalize(&k_odd, n_odd), &finalize(&j_odd, n_odd), n_odd);
    let mut k_all = vec![vec![0.0f64; d * d]; points.len()];
    let mut j_all = vec![vec![0.0f64; d * d]; points.len()];
    for pi in 0..points.len() {
        for idx in 0..d * d {
            k_all[pi][idx] = k_even[pi][idx] + k_odd[pi][idx];
            j_all[pi][idx] = j_even[pi][idx] + j_odd[pi][idx];
        }
    }
    let n_all = n_even + n_odd;
    let fit_all = make_fit(&finalize(&k_all, n_all), &finalize(&j_all, n_all), n_all);

    // ── G3 (half-corpus stability) ──
    let g3: G3Report = gate_g3(&fit_even, &fit_odd);
    println!(
        "# G3: points={} max_rel_k={:.4} max_rel_j={:.4} pass={}",
        g3.points, g3.max_rel_k, g3.max_rel_j, g3.pass
    );

    // ── artifacts ──
    std::fs::create_dir_all(&args.out)?;
    let mut manifest = serde_json::to_string_pretty(&json!({
        "format": "lens_gpu_fit_t3_v1",
        "gguf": args.gguf,
        "corpus_digest": digest,
        "docs": args.docs,
        "t": args.t,
        "seed_lo": args.seed_lo,
        "seed_hi": seed_hi,
        "dim_batch": dim_batch,
        "d": d,
        "points": points.iter().map(|p| format!("{p:?}")).collect::<Vec<_>>(),
        "n_even": n_even,
        "n_odd": n_odd,
        "layout": "rows=[points][dirs·d] raw f32, dir-major then dim (fit.k / fit.j)",
        "cost": {
            "docs": cost.docs,
            "basis_dirs": cost.basis_dirs,
            "prefix_passes": cost.prefix_passes,
            "suffix_passes": cost.suffix_passes,
            "capture_secs": cost.capture_secs,
            "sweep_secs": cost.sweep_secs,
            "box_state": cost.box_state,
        },
        "g3": {
            "max_rel_k": g3.max_rel_k,
            "max_rel_j": g3.max_rel_j,
            "pass": g3.pass,
        },
    }))?;
    manifest.push('\n');
    std::fs::write(args.out.join("manifest.json"), manifest)?;
    let mut total = 0u64;
    for (name, rows) in [
        ("fit_even.k", &fit_even.k),
        ("fit_even.j", &fit_even.j),
        ("fit_odd.k", &fit_odd.k),
        ("fit_odd.j", &fit_odd.j),
        ("fit_all.k", &fit_all.k),
        ("fit_all.j", &fit_all.j),
    ] {
        total += write_f32_rows(&args.out.join(name), rows)?;
    }
    println!(
        "# artifacts: {} ({} MiB of f32 rows) — G3 pass={}",
        args.out.display(),
        total / (1024 * 1024),
        g3.pass
    );
    if !g3.pass {
        bail!("G3 half-corpus stability FAILED — see the report above");
    }
    Ok(())
}
