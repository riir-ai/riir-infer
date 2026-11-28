//! i1221 T3(c) — the tokens-per-task vs loop-count cell (plan 636
//! read-order leg 4, LAST; the tokenizer-decision amendment
//! 2026-11-28 discharged the park's blocker — the pin's own embedded
//! gpt2 BPE rides [`riir_infer_core::tokenizer::BpeTokenizer::from_gguf`],
//! consumed never forked, and the labeled probe suite is
//! [`riir_infer_core::dory::probe_suite`]).
//!
//! The paper's 1.3–2.5× agentic law, direction check: for each of the
//! 16 labeled micro-tasks and each k ∈ {1, 2, 4}, GREEDY-decode until
//! the task's verifier passes (or eos, or the per-task token cap) —
//! the measured number is the generation length to the first correct
//! answer, per loop depth. Timing rows → the exclusive-window law
//! binds (plan 636): run this ONLY on a quiet box, disclosed.
//!
//! # Usage
//!
//! ```text
//! cargo run --release --features dory \
//!   --example i1221_t3c_tokens_per_task -- \
//!   --gguf ../riir-train/data/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf \
//!   [--out .benchmarks/i1221_t3c_tokens_per_task_cell] \
//!   [--ks 1,2,4] [--max-gen 64] [--fresh]
//! ```
//!
//! Resumable JSONL (plan 636's law): one row per (task, k), flushed
//! immediately; the header lineage binds pin blake3, repo STAMP, the k
//! set, max_gen, AND the suite digest — a suite edit starts a fresh
//! lineage (mismatch REFUSES exit 2); `--fresh` starts one explicitly.
//! Report-only: the rows are measurements, never gates; the per-k
//! summary never folds cap-hit/missed rows into the mean. The prompt
//! text is the checkpoint's OWN trained instruct format (`[INST]
//! … [/INST]`, special tokens matched by the tokenizer pass).

use std::fs;
use std::io::Write;
use std::path::PathBuf;

use riir_infer_core::dory::probe_suite::{
    self as suite, ROW_RECORD, TokensPerTaskRow, blake3_file, make_row, summarize_k,
    verify_resume_lineage_c, wrap_inst,
};
use riir_infer_core::dory::spectra::argmax_of;
use riir_infer_core::dory::{DoryForward, DoryKvCache, load_dory_weights};
use riir_infer_core::tokenizer::BpeTokenizer;

fn main() {
    let mut gguf = PathBuf::from("../riir-train/data/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf");
    let mut out = PathBuf::from(".benchmarks/i1221_t3c_tokens_per_task_cell");
    let mut ks: Vec<usize> = vec![1, 2, 4];
    let mut max_gen = 64usize;
    let mut fresh = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--gguf" => gguf = PathBuf::from(args.next().expect("--gguf PATH")),
            "--out" => out = PathBuf::from(args.next().expect("--out DIR")),
            "--ks" => {
                let raw = args.next().expect("--ks 1,2,4");
                ks = raw
                    .split(',')
                    .map(|s| s.trim().parse().expect("k usize"))
                    .collect();
                assert!(!ks.is_empty(), "--ks needs at least one k");
            }
            "--max-gen" => max_gen = args.next().expect("--max-gen N").parse().expect("N"),
            "--fresh" => fresh = true,
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(2);
            }
        }
    }

    let pin_digest = blake3_file(&gguf);
    let repo_commit = repo_stamp();
    println!(
        "[i1221-t3c] pin {} blake3 {}… stamp {}… ks={ks:?} max_gen={max_gen} suite {}…",
        gguf.display(),
        &pin_digest[..16],
        &repo_commit[..12.min(repo_commit.len())],
        &suite::suite_digest()[..12],
    );

    // ── load: weights + the pin's own embedded tokenizer ──
    let t0 = std::time::Instant::now();
    let file = riir_infer_core::gguf_loader::GgufFile::open(&gguf).expect("open gguf");
    let tok = BpeTokenizer::from_gguf(&file).expect("embedded gpt2 BPE tokenizer");
    let weights = load_dory_weights(&gguf).expect("load dory weights");
    let k_full = weights.meta.num_loops;
    let vocab = weights.dims.vocab;
    println!(
        "[i1221-t3c] loaded: k_full={k_full} vocab={vocab} tok_vocab={} bos={} eos={} in {:.1}s",
        tok.vocab_size(),
        tok.bos_id(),
        tok.eos_id(),
        t0.elapsed().as_secs_f32(),
    );
    for &k in &ks {
        assert!(
            (1..=k_full).contains(&k),
            "k={k} outside the trained 1..={k_full} range — the elastic-k question is 1..=num_loops"
        );
    }

    // ── prompts: the suite's fixed order, wrapped in the trained format ──
    let prompts: Vec<Vec<u32>> = suite::SUITE
        .iter()
        .map(|t| {
            tok.encode_with_bos(&wrap_inst(t.prompt))
                .into_iter()
                .map(|u| u as u32)
                .collect()
        })
        .collect();

    // ── resumable rows ──
    fs::create_dir_all(&out).expect("create out dir");
    let rows_path = out.join("rows.jsonl");
    let mut done =
        match verify_resume_lineage_c(&rows_path, &pin_digest, &repo_commit, &ks, max_gen) {
            Ok(d) => d,
            Err(msg) => {
                eprintln!("REFUSE: {:?} {msg}", rows_path);
                std::process::exit(2);
            }
        };
    let mut rows_file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&rows_path)
        .expect("open rows file");
    if done.is_empty() || fresh {
        if fresh && rows_path.exists() {
            let _ = fs::remove_file(&rows_path);
            done.clear();
            rows_file = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&rows_path)
                .expect("reopen rows file");
        }
        let header = format!(
            concat!(
                "{{\"record\":\"{}\",\"ts\":\"{}\",\"model\":{{\"path\":\"{}\",\"blake3\":\"{}\",",
                "\"arch\":\"dory\",\"k_full\":{},\"tok_vocab\":{},\"eos\":{}}},\"repo_commit\":\"{}\",",
                "\"probe\":{{\"ks\":{:?},\"max_gen\":{},\"suite_blake3\":\"{}\",\"tasks\":{}}}}}\n"
            ),
            suite::HEADER_RECORD,
            rfc3339_now(),
            gguf.display(),
            pin_digest,
            k_full,
            tok.vocab_size(),
            tok.eos_id(),
            repo_commit,
            ks,
            max_gen,
            suite::suite_digest(),
            suite::SUITE.len(),
        );
        rows_file
            .write_all(header.as_bytes())
            .expect("write header");
        rows_file.flush().expect("flush header");
        println!("[i1221-t3c] new lineage: header written");
    } else {
        println!(
            "[i1221-t3c] resume: {} row(s) present, lineage verified — missing pairs only",
            done.len()
        );
    }

    // ── the cell: greedy decode to verifier-pass, per (task, k) ──
    let cell_start = std::time::Instant::now();
    let total_pairs = suite::SUITE.len() * ks.len();
    for (p, task) in suite::SUITE.iter().enumerate() {
        for &k in &ks {
            if done.contains(&(p, k)) {
                continue;
            }
            let arm_start = std::time::Instant::now();
            let mut cache = DoryKvCache::new(&weights, k);
            let fwd = DoryForward::new(&weights);
            let tokens = &prompts[p];
            let logits = fwd.prefill(tokens, &mut cache).expect("prefill");
            let mut last = logits.last().expect("non-empty prefill").clone();

            let mut gen_toks: Vec<u32> = Vec::with_capacity(max_gen);
            let mut outcome = "cap_hit";
            let mut tokens_to_pass = 0usize;
            let eos = tok.eos_id() as u32;
            loop {
                let next = argmax_of(&last) as u32;
                if next == eos {
                    outcome = "missed_eos";
                    break;
                }
                gen_toks.push(next);
                let decoded = tok.decode(&gen_toks.iter().map(|&t| t as usize).collect::<Vec<_>>());
                if suite::step_satisfied(task, &decoded) {
                    outcome = "passed";
                    tokens_to_pass = gen_toks.len();
                    break;
                }
                if gen_toks.len() >= max_gen {
                    break; // cap_hit (the initial value)
                }
                last = fwd.decode_step(next, &mut cache).expect("decode step");
            }

            let row = make_row(
                p,
                k,
                task,
                outcome,
                tokens_to_pass,
                gen_toks.len(),
                tok.decode(&gen_toks.iter().map(|&t| t as usize).collect::<Vec<_>>()),
                &gen_toks,
            );
            let line = serde_json::to_string(&row).expect("row json");
            rows_file.write_all(line.as_bytes()).expect("write row");
            rows_file.write_all(b"\n").expect("newline");
            rows_file.flush().expect("flush row");
            println!(
                "[i1221-t3c] p{p} ({}) k={k}: {} at {} tok (gen {}), {:.1}s",
                task.id,
                outcome,
                if outcome == "passed" {
                    tokens_to_pass
                } else {
                    gen_toks.len()
                },
                gen_toks.len(),
                arm_start.elapsed().as_secs_f64(),
            );
        }
    }

    // ── the direction-check table ──
    let all: Vec<TokensPerTaskRow> = read_rows(&rows_path);
    println!("[i1221-t3c] per-k summary (mean over PASSED rows only — cap/missed never folded):");
    for &k in &ks {
        let s = summarize_k(k, &all);
        let mean = s
            .mean_tokens_to_pass
            .map(|m| format!("{m:.2}"))
            .unwrap_or_else(|| "n/a".into());
        println!(
            "[i1221-t3c]   k={k}: n={} passed={} missed_eos={} cap_hit={} mean_ttp={mean} \
             min={:?} max={:?}",
            s.n, s.passed, s.missed_eos, s.cap_hit, s.min_tokens_to_pass, s.max_tokens_to_pass,
        );
    }
    println!(
        "[i1221-t3c] cell done: {total_pairs} pair(s), total {:.1}s",
        cell_start.elapsed().as_secs_f64()
    );
}

/// Read back every row of record `ROW_RECORD` (the summary reads the
/// FILE, not run state — a resumed cell summarizes the whole lineage).
/// The row struct is fully serde-general (owned tag fields), so the
/// read is a real deserialization, not a Value projection.
fn read_rows(rows_path: &std::path::Path) -> Vec<TokensPerTaskRow> {
    let content = fs::read_to_string(rows_path).unwrap_or_default();
    content
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .filter(|r| r.get("record").and_then(|x| x.as_str()) == Some(ROW_RECORD))
        .filter_map(|v| serde_json::from_value::<TokensPerTaskRow>(v).ok())
        .collect()
}

fn repo_stamp() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into())
}

fn rfc3339_now() -> String {
    std::process::Command::new("date")
        .arg("-u")
        .args(["+%Y-%m-%dT%H:%M:%SZ"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}
