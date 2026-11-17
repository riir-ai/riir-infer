//! i1221 T3(d) — the loopability predictor cell, dense-GGUF arm (plan 636
//! read-order leg 1; Research 672 §3).
//!
//! Chunks a DENSE qwen35 hybrid checkpoint's middle layer range, iterates
//! the loop recurrence over it ([`riir_infer_core::deltanet::loopability`]), and
//! writes resumable JSONL rows under `.benchmarks/` (plan 636's durable
//! partial-row format). Report-only — the rows are measurements, never
//! gates; the only asserts in the lane live in the lib fixtures.
//!
//! # Usage
//!
//! ```text
//! cargo run --release --features deltanet_inference \
//!   --example i1221_t3d_loopability -- \
//!   --gguf ../riir-train/data/Qwen3.5-0.8B-Base-Q8_0.gguf \
//!   [--out .benchmarks/i1221_t3d_loopability_cell] \
//!   [--ks 1,2,4,8] [--seq-len 8] [--prompt-sets 3] [--seed 0] \
//!   [--mid-start 8] [--chunk-len 8] [--fresh]
//! ```
//!
//! # Resumability
//!
//! `<out>/rows.jsonl` line 1 is the header (schema, model blake3, repo
//! STAMP, chunk, ks, prompt spec, box state); every following line is one
//! `(prompt_idx, k)` row, flushed to disk immediately after it is computed.
//! A rerun resumes: the header must match the current run byte-for-byte on
//! every lineage field (blake3, STAMP, chunk, ks, seed, seq-len) — a
//! mismatch REFUSES (exit 2, never append across lineages); completed
//! `(prompt_idx, k)` pairs are skipped. `--fresh` truncates and starts a
//! new lineage.
//!
//! # Prompt law
//!
//! Pseudo-token prompts (the smoke's law until a tokenizer lands): each
//! set is `seq_len` tokens drawn uniform in `[4, vocab_size)` from an LCG
//! seeded with `seed + prompt_idx` — deterministic, recorded in the header,
//! never an unseeded draw.

use std::collections::BTreeSet;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use riir_infer_core::deltanet::loopability::{
    LoopKRow, LoopabilityChunk, full_depth_reference, loop_variant_row,
};
use riir_infer_core::deltanet::{HybridForwardScratch, effective_rotary_dim};
use riir_infer_core::gguf_loader::load_qwen_deltanet_weights_gguf;
use riir_infer_core::rope::RopeFreqTable;

const SCHEMA_VERSION: u32 = 1;
const ROWS_FILE: &str = "rows.jsonl";

struct Args {
    gguf: PathBuf,
    out: PathBuf,
    ks: Vec<usize>,
    seq_len: usize,
    prompt_sets: usize,
    seed: u64,
    mid_start: Option<usize>,
    chunk_len: Option<usize>,
    fresh: bool,
}

fn usage() -> ! {
    eprintln!(
        "usage: i1221_t3d_loopability --gguf <model.gguf> [--out DIR] [--ks 1,2,4,8] \
         [--seq-len 8] [--prompt-sets 3] [--seed 0] [--mid-start N] [--chunk-len N] [--fresh]"
    );
    std::process::exit(2);
}

fn parse_args() -> Args {
    let mut args = Args {
        gguf: PathBuf::new(),
        out: PathBuf::from(".benchmarks/i1221_t3d_loopability_cell"),
        ks: vec![1, 2, 4, 8],
        seq_len: 8,
        prompt_sets: 3,
        seed: 0,
        mid_start: None,
        chunk_len: None,
        fresh: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = |name: &str| -> String {
            it.next()
                .unwrap_or_else(|| panic!("{name} needs a value (exit via usage)"))
        };
        match flag.as_str() {
            "--gguf" => args.gguf = PathBuf::from(value("--gguf")),
            "--out" => args.out = PathBuf::from(value("--out")),
            "--ks" => {
                let raw = value("--ks");
                args.ks = raw
                    .split(',')
                    .map(|s| {
                        s.trim()
                            .parse::<usize>()
                            .unwrap_or_else(|_| panic!("bad ks entry '{s}'"))
                    })
                    .collect();
                assert!(!args.ks.is_empty(), "--ks needs at least one depth");
            }
            "--seq-len" => args.seq_len = value("--seq-len").parse().expect("--seq-len usize"),
            "--prompt-sets" => {
                args.prompt_sets = value("--prompt-sets").parse().expect("--prompt-sets usize")
            }
            "--seed" => args.seed = value("--seed").parse().expect("--seed u64"),
            "--mid-start" => {
                args.mid_start = Some(value("--mid-start").parse().expect("--mid-start usize"))
            }
            "--chunk-len" => {
                args.chunk_len = Some(value("--chunk-len").parse().expect("--chunk-len usize"))
            }
            "--fresh" => args.fresh = true,
            other => {
                eprintln!("unknown flag {other}");
                usage();
            }
        }
    }
    if args.gguf.as_os_str().is_empty() {
        eprintln!("--gguf is required");
        usage();
    }
    args
}

/// The LCG the forward.rs tests use — one deterministic pseudo-token stream
/// per prompt set (no unseeded draw; the seed + law live in the header).
struct Lcg(u64);

impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn pseudo_tokens(seed: u64, seq_len: usize, vocab_size: usize) -> Vec<usize> {
    let mut rng = Lcg(seed.wrapping_add(0xD07));
    (0..seq_len)
        .map(|_| 4 + (rng.next_f64() * (vocab_size - 4) as f64) as usize)
        .collect()
}

fn blake3_file(path: &Path) -> String {
    let mut file = fs::File::open(path).unwrap_or_else(|e| panic!("open {:?}: {e}", path));
    let mut hasher = blake3::Hasher::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = std::io::Read::read(&mut file, &mut buf).expect("read model file");
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    hasher.finalize().to_hex().to_string()
}

fn repo_stamp() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

#[derive(serde::Serialize)]
struct Header<'a> {
    record: &'static str,
    schema_version: u32,
    model: ModelLine<'a>,
    chunk: LoopabilityChunk,
    ks: &'a [usize],
    prompts: PromptLine,
    box_state: BoxState,
    repo_commit: &'a str,
}

#[derive(serde::Serialize)]
struct ModelLine<'a> {
    path: &'a str,
    blake3: &'a str,
    arch: &'static str,
    n_layer: usize,
    n_embd: usize,
    vocab_size: usize,
    attention_layers: usize,
    deltanet_layers: usize,
}

#[derive(serde::Serialize)]
struct PromptLine {
    sets: usize,
    seq_len: usize,
    seed: u64,
    law: &'static str,
}

#[derive(serde::Serialize)]
struct BoxState {
    utc_rfc3339: String,
    profile: &'static str,
    threads: usize,
}

#[derive(serde::Serialize)]
struct RowOut {
    record: &'static str,
    prompt_idx: usize,
    #[serde(flatten)]
    row: LoopKRow,
    wall_ms: f64,
}

fn main() {
    let args = parse_args();
    let t_boot = Instant::now();

    // ── model + lineage ─────────────────────────────────────────────────
    let t_load = Instant::now();
    let (config, weights) = load_qwen_deltanet_weights_gguf(&args.gguf)
        .unwrap_or_else(|e| panic!("load {}: {e}", args.gguf.display()));
    let load_s = t_load.elapsed().as_secs_f64();
    let n_layers = weights.layers.len();
    let attention_layers = weights
        .layer_types
        .iter()
        .filter(|t| **t == riir_infer_core::types::DeltaNetLayerType::Attention)
        .count();
    let blake3_hex = blake3_file(&args.gguf);
    let stamp = repo_stamp();
    println!(
        "[i1221-t3d] model {} (blake3 {}, {n_layers} layers, {attention_layers} attention) loaded in {load_s:.1}s; stamp {}",
        args.gguf.display(),
        &blake3_hex[..16],
        stamp
    );

    // ── chunk defaults: the middle third, validated against the stack ───
    let mid_start = args.mid_start.unwrap_or(n_layers / 3);
    let chunk_len = args.chunk_len.unwrap_or(n_layers / 3);
    let chunk = LoopabilityChunk {
        mid_start,
        chunk_len,
    }
    .validated(n_layers);
    println!(
        "[i1221-t3d] chunk layers {}..{} (input half 0..{}, output half {}..{n_layers})",
        chunk.mid_start,
        chunk.mid_start + chunk.chunk_len,
        chunk.mid_start,
        chunk.mid_start + chunk.chunk_len
    );

    // ── resume lineage ──────────────────────────────────────────────────
    fs::create_dir_all(&args.out).unwrap_or_else(|e| panic!("mkdir {:?}: {e}", args.out));
    let rows_path = args.out.join(ROWS_FILE);
    let mut completed: BTreeSet<(usize, usize)> = BTreeSet::new();
    if rows_path.exists() && !args.fresh {
        let text =
            fs::read_to_string(&rows_path).unwrap_or_else(|e| panic!("read {rows_path:?}: {e}"));
        let mut lines = text.lines();
        let header_line = lines.next().unwrap_or("");
        let header: serde_json::Value = serde_json::from_str(header_line)
            .unwrap_or_else(|e| panic!("{rows_path:?} line 1 is not a header ({e}) — a corrupt lineage is never appended; rerun with --fresh or a new --out"));
        let lineage_bad = |why: String| -> ! {
            eprintln!(
                "REFUSE: {rows_path:?} header lineage mismatch: {why}\n  \
                 never append across lineages — rerun with --fresh or a new --out"
            );
            std::process::exit(2);
        };
        if header["record"] != "i1221_t3d_loopability_header" {
            lineage_bad("record is not the t3d header".into());
        }
        if header["schema_version"].as_u64() != Some(SCHEMA_VERSION as u64) {
            lineage_bad(format!(
                "schema {:?} != {SCHEMA_VERSION}",
                header["schema_version"]
            ));
        }
        if header["model"]["blake3"].as_str() != Some(blake3_hex.as_str()) {
            lineage_bad("model blake3 differs (different checkpoint)".into());
        }
        if header["repo_commit"].as_str() != Some(stamp.as_str()) {
            lineage_bad("repo STAMP differs (different code lineage)".into());
        }
        if header["chunk"]["mid_start"].as_u64() != Some(chunk.mid_start as u64)
            || header["chunk"]["chunk_len"].as_u64() != Some(chunk.chunk_len as u64)
        {
            lineage_bad("chunk differs".into());
        }
        let want_ks: Vec<u64> = args.ks.iter().map(|&k| k as u64).collect();
        let got_ks: Vec<u64> = header["ks"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_u64()).collect())
            .unwrap_or_default();
        if got_ks != want_ks {
            lineage_bad("ks differs".into());
        }
        if header["prompts"]["sets"].as_u64() != Some(args.prompt_sets as u64)
            || header["prompts"]["seq_len"].as_u64() != Some(args.seq_len as u64)
            || header["prompts"]["seed"].as_u64() != Some(args.seed)
        {
            lineage_bad("prompt spec differs".into());
        }
        for line in lines {
            if line.trim().is_empty() {
                continue;
            }
            let row: serde_json::Value = serde_json::from_str(line).unwrap_or_else(|e| {
                panic!("{rows_path:?} has a corrupt row ({e}) — rerun with --fresh")
            });
            if row["record"] == "i1221_t3d_loopability_row" {
                let p = row["prompt_idx"].as_u64().expect("row prompt_idx") as usize;
                // LoopKRow is flattened into the row line — k sits at the
                // top level, never nested.
                let k = row["k"].as_u64().expect("row k") as usize;
                completed.insert((p, k));
            }
        }
        println!(
            "[i1221-t3d] resume: {} of {} rows already durable",
            completed.len(),
            args.prompt_sets * args.ks.len()
        );
    }

    // ── header write (fresh lineage only) ───────────────────────────────
    let model_path_str = args.gguf.display().to_string();
    if !rows_path.exists() {
        let header = Header {
            record: "i1221_t3d_loopability_header",
            schema_version: SCHEMA_VERSION,
            model: ModelLine {
                path: &model_path_str,
                blake3: &blake3_hex,
                arch: "qwen35",
                n_layer: n_layers,
                n_embd: config.n_embd,
                vocab_size: config.vocab_size,
                attention_layers,
                deltanet_layers: n_layers - attention_layers,
            },
            chunk,
            ks: &args.ks,
            prompts: PromptLine {
                sets: args.prompt_sets,
                seq_len: args.seq_len,
                seed: args.seed,
                law: "LCG uniform in [4, vocab), per-set seed = seed + prompt_idx (no tokenizer yet — the smoke's pseudo-token law)",
            },
            box_state: BoxState {
                utc_rfc3339: utc_now(),
                profile: if cfg!(debug_assertions) {
                    "debug"
                } else {
                    "release"
                },
                threads: std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1),
            },
            repo_commit: &stamp,
        };
        let mut w = BufWriter::new(
            fs::File::create(&rows_path).unwrap_or_else(|e| panic!("create {rows_path:?}: {e}")),
        );
        serde_json::to_writer(&mut w, &header).expect("header serialize");
        writeln!(w).expect("header newline");
        w.flush().expect("header flush");
        println!(
            "[i1221-t3d] fresh lineage header written to {}",
            rows_path.display()
        );
    }

    // ── the cell ────────────────────────────────────────────────────────
    let rope_freq = RopeFreqTable::new(config.rope_theta, effective_rotary_dim(&config));
    let mut scratch = HybridForwardScratch::new(&config);
    let rows_file = fs::OpenOptions::new()
        .append(true)
        .open(&rows_path)
        .unwrap_or_else(|e| panic!("append {rows_path:?}: {e}"));
    let mut out = BufWriter::new(rows_file);

    let mut written = 0usize;
    let mut skipped = 0usize;
    let t_cell = Instant::now();
    for prompt_idx in 0..args.prompt_sets {
        let prompt_tokens = pseudo_tokens(
            args.seed + prompt_idx as u64,
            args.seq_len,
            config.vocab_size,
        );
        let t_ref = Instant::now();
        let full =
            full_depth_reference(&weights, &config, &prompt_tokens, &mut scratch, &rope_freq);
        println!(
            "[i1221-t3d] prompt {prompt_idx}: reference argmax {} margin {:.4} in {:.1}s",
            full.argmax,
            full.margin,
            t_ref.elapsed().as_secs_f64()
        );
        for &k in &args.ks {
            if completed.contains(&(prompt_idx, k)) {
                skipped += 1;
                continue;
            }
            let t_row = Instant::now();
            let row = loop_variant_row(
                &weights,
                &config,
                &prompt_tokens,
                chunk,
                k,
                &full,
                &mut scratch,
                &rope_freq,
            );
            let wall_ms = t_row.elapsed().as_secs_f64() * 1000.0;
            let flip_note = if row.argmax_flip { " FLIP" } else { "" };
            println!(
                "[i1221-t3d] prompt {prompt_idx} k={k}: lambda_mean {:?} kl_full {:.4e} kl_loop1 {:?} margins {:.3}/{:.3}{} ({:.1} ms)",
                row.lambda_mean_pooled.unwrap_or(f64::NAN),
                row.kl_full_vs_loop,
                row.kl_loop1_vs_loopk.unwrap_or(f64::NAN),
                row.margin_full,
                row.margin_loop,
                flip_note,
                wall_ms
            );
            let row_out = RowOut {
                record: "i1221_t3d_loopability_row",
                prompt_idx,
                row,
                wall_ms,
            };
            serde_json::to_writer(&mut out, &row_out).expect("row serialize");
            writeln!(out).expect("row newline");
            out.flush().expect("row flush — the durable per-row write");
            written += 1;
        }
    }
    out.flush().ok();

    let total = args.prompt_sets * args.ks.len();
    println!(
        "[i1221-t3d] done: {written} written, {skipped} resumed, {total} total — rows at {} (cell wall {:.1}s, boot-to-done {:.1}s)",
        rows_path.display(),
        t_cell.elapsed().as_secs_f64(),
        t_boot.elapsed().as_secs_f64()
    );
    if written == 0 && skipped == total {
        println!("[i1221-t3d] nothing to do — every row was already durable");
    }
}

fn utc_now() -> String {
    // No chrono dep: seconds-precision UTC via the unix epoch, RFC3339 by
    // hand (days → y/m/d civil-from-days, Howard Hinnant's algorithm).
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_secs();
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let (h, m, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1461 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y + 1 } else { y };
    format!("{y:04}-{:02}-{:02}T{h:02}:{m:02}:{s:02}Z", mo, d)
}
