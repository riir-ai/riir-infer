//! i1221 T3(b) — the halter-trace cell (plan 636 read-order leg 3;
//! Plan 304's open `GainCostLoopHalter` cost_floor calibration with
//! REAL looped-trained traces).
//!
//! Reuses (a)'s lens exactly: one k=4 prefill with per-pass state
//! snapshots armed ([`riir_infer_core::dory::spectra::run_lens`]), then
//! [`riir_infer_core::dory::spectra::halter_traces`] derives per-
//! position step-size / angular-change sequences from the snapshots and
//! feeds katgpt-core's OWN halter kernel (consumed via the `dory`
//! feature's `katgpt-core/gain_cost_halt` forward — never reimplemented)
//! under each candidate cost posture.
//!
//! # The answer format (what katgpt-core gets)
//!
//! Per (prompt, position): `first_step` (the depth-2 step — the episode
//! scale Plan 304's `cost_floor = frac × first_loop_step_size` wiring
//! multiplies) + one cell per cost posture (`cost_floor_frac`, absolute
//! `cost`, `halt_depth`, the `verdict_shape` string over depths 2..=k,
//! `halt_reason`). The consumer diffs `verdict_shape` against its
//! synthetic reference regimes (geometric decay 0.3/0.5/0.7 at floor
//! 0.6) without re-running anything — the postures span the Phase-2
//! wiring default (0.01) through the synthetic crowd band (0.5–0.8).
//!
//! # Usage
//!
//! ```text
//! cargo run --release --features dory \
//!   --example i1221_t3b_halter_traces -- \
//!   --gguf ../riir-train/data/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf \
//!   [--out .benchmarks/i1221_t3b_halter_traces_cell] \
//!   [--seq-len 8] [--prompt-sets 3] [--seed 0] [--fresh]
//! ```
//!
//! Resumable JSONL (plan 636's law): header lineage verified on every
//! field; completed PROMPTS are skipped whole (the lens is one run per
//! prompt and both row families fall out of it); mismatch REFUSES
//! exit 2. Report-only: the rows are measurements, never gates; the
//! only asserts are the lib fixtures + the runner's real-pin receipt
//! (the trace determinism line).

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use riir_infer_core::dory::spectra::{
    halter_traces, run_lens, verify_resume_lineage,
};
use riir_infer_core::dory::{DoryForward, DoryKvCache, load_dory_weights};

const SCHEMA_VERSION: u32 = 1;
const ROWS_FILE: &str = "rows.jsonl";
/// The cost postures: the Phase-2 wiring default (0.01), two mid rungs,
/// and the synthetic G2 crowd band (0.5/0.6/0.8 — Bench 304's override
/// finding). Fractions of `first_step`.
const COST_FLOOR_FRACS: [f32; 6] = [0.01, 0.1, 0.25, 0.5, 0.6, 0.8];

fn main() {
    let mut gguf =
        PathBuf::from("../riir-train/data/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf");
    let mut out = PathBuf::from(".benchmarks/i1221_t3b_halter_traces_cell");
    let mut seq_len = 8usize;
    let mut prompt_sets = 3usize;
    let mut seed: u64 = 0;
    let mut fresh = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--gguf" => gguf = PathBuf::from(args.next().expect("--gguf PATH")),
            "--out" => out = PathBuf::from(args.next().expect("--out DIR")),
            "--seq-len" => seq_len = args.next().expect("--seq-len N").parse().expect("N"),
            "--prompt-sets" => {
                prompt_sets = args.next().expect("--prompt-sets N").parse().expect("N")
            }
            "--seed" => seed = args.next().expect("--seed N").parse().expect("N"),
            "--fresh" => fresh = true,
            other => {
                eprintln!("unknown arg {other} (--gguf --out --seq-len --prompt-sets --seed --fresh)");
                std::process::exit(2);
            }
        }
    }

    // ── lineage ──
    let pin_digest = blake3_file(&gguf);
    let repo_commit = repo_stamp();
    println!(
        "[i1221-t3b] pin {} blake3 {}… stamp {}…",
        gguf.display(),
        &pin_digest[..16],
        &repo_commit[..12.min(repo_commit.len())]
    );

    // ── load ──
    let t0 = std::time::Instant::now();
    let weights = load_dory_weights(&gguf).expect("load dory weights");
    let k_full = weights.meta.num_loops;
    let vocab = weights.dims.vocab;
    let n_embd = weights.dims.n_embd;
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    println!(
        "[i1221-t3b] loaded: k_full={k_full} vocab={vocab} n_embd={n_embd} in {:.1}s; \
         cost postures {COST_FLOOR_FRACS:?} (fractions of first_step)",
        t0.elapsed().as_secs_f32(),
    );

    // ── prompts: the (a) law verbatim (the same LCG, seed + idx) ──
    let prompts: Vec<Vec<u32>> = (0..prompt_sets)
        .map(|p| pseudo_tokens(seed + p as u64, seq_len, vocab))
        .collect();

    // ── resumable rows ──
    fs::create_dir_all(&out).expect("create out dir");
    let rows_path = out.join(ROWS_FILE);
    if fresh {
        let _ = fs::remove_file(&rows_path);
    }
    let done = verify_resume_lineage(
        &rows_path,
        &pin_digest,
        &repo_commit,
        k_full,
        seq_len,
        prompt_sets,
        seed,
        "i1221_t3b_halter_trace_row",
    );
    // The prompt is the resume unit (one lens per prompt; both row
    // families fall out of it). A complete trace set carries depths
    // 2..=k — the two endpoints prove it; a partial set (torn write)
    // refuses loud: the JSONL is append-only, never duplicate-patched.
    let prompt_done =
        |p: usize| done.contains(&(p, 2)) && done.contains(&(p, k_full));
    let prompt_torn =
        |p: usize| done.contains(&(p, 2)) != done.contains(&(p, k_full));
    let mut rows_file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&rows_path)
        .expect("open rows");
    if done.is_empty() {
        let header = format!(
            "{{\"record\":\"i1221_t3b_halter_header\",\"schema_version\":{SCHEMA_VERSION},\
             \"model\":{{\"path\":{},\"blake3\":\"{}\",\"arch\":\"dory\",\"k_full\":{},\
             \"n_embd\":{},\"vocab\":{}}},\
             \"prompts\":{{\"sets\":{},\"seq_len\":{},\"seed\":{},\
             \"law\":\"LCG uniform in [4, vocab), per-set seed = seed + prompt_idx (the (a) law verbatim)\"}},\
             \"cost_floor_fracs\":{},\
             \"halter_posture\":\"Plan 304 defaults: tau 1.0, oscillation_patience 1, l_min 1\",\
             \"box_state\":{{\"utc_rfc3339\":\"{}\",\"profile\":\"release\",\"threads\":{}}},\
             \"repo_commit\":\"{}\"}}\n",
            json_str(&gguf.to_string_lossy()),
            pin_digest,
            k_full,
            n_embd,
            vocab,
            prompt_sets,
            seq_len,
            seed,
            json_str(&format!("{COST_FLOOR_FRACS:?}")),
            rfc3339_now(),
            threads,
            repo_commit,
        );
        rows_file.write_all(header.as_bytes()).expect("write header");
        rows_file.flush().expect("flush header");
        println!("[i1221-t3b] new lineage: header written");
    } else {
        println!(
            "[i1221-t3b] resume: {} prompt(s) present, lineage verified — missing prompts only",
            done.len()
        );
    }

    // ── the cell: one lens per prompt, traces + postures fall out ──
    for (p, tokens) in prompts.iter().enumerate() {
        if prompt_torn(p) {
            eprintln!(
                "REFUSE: p{p} has a torn row set (depths 2..={k_full} incomplete) — \
                 the rows file is append-only; re-run with --fresh"
            );
            std::process::exit(2);
        }
        if prompt_done(p) {
            println!("[i1221-t3b] p{p}: done — skipped");
            continue;
        }
        let mut cache = DoryKvCache::new(&weights, k_full);
        let fwd = DoryForward::new(&weights);
        let mut snaps = Vec::new();
        let _full = run_lens(&fwd, tokens, &mut cache, &mut snaps).expect("lens run");

        let (trace_rows, calib_rows) =
            halter_traces(p, &snaps, tokens.len(), n_embd, &COST_FLOOR_FRACS);

        // Per-prompt receipt (report lines): the step ladder at pos 0 —
        // the shape the halter reads. Two runs of this example must
        // produce identical numbers (determinism receipt).
        let ladder: Vec<String> = (0..k_full - 1)
            .map(|i| {
                format!(
                    "{:.4}",
                    trace_rows[i * tokens.len() /* pos 0 */].step_size
                )
            })
            .collect();
        println!("[i1221-t3b] p{p} pos0 step ladder (d2..d{k_full}): {}", ladder.join(" → "));

        for row in &trace_rows {
            let line = serde_json::to_string(row).expect("trace json");
            rows_file.write_all(line.as_bytes()).expect("write trace");
            rows_file.write_all(b"\n").expect("newline");
        }
        for row in &calib_rows {
            let line = serde_json::to_string(row).expect("calib json");
            rows_file.write_all(line.as_bytes()).expect("write calib");
            rows_file.write_all(b"\n").expect("newline");
        }
        rows_file.flush().expect("flush prompt");

        // Per-prompt posture summary (report lines, the record's table):
        // per posture, how many positions halt before the trained depth.
        for (ci, &frac) in COST_FLOOR_FRACS.iter().enumerate() {
            let mut halted = 0usize;
            let mut depth_sum = 0usize;
            for row in &calib_rows {
                let cell = &row.postures[ci];
                if cell.halt_reason != "-" && cell.halt_depth < k_full {
                    halted += 1;
                }
                depth_sum += cell.halt_depth;
            }
            println!(
                "[i1221-t3b] p{p} frac={frac}: early-halt {halted}/{} positions, \
                 mean halt depth {:.2}",
                calib_rows.len(),
                depth_sum as f64 / calib_rows.len() as f64,
            );
        }
    }
    println!(
        "[i1221-t3b] cell done: {prompt_sets} prompt(s) × {seq_len} positions × \
         {} postures, k=1..={k_full}",
        COST_FLOOR_FRACS.len(),
    );
}

/// LCG pseudo-token prompt (the (a) law verbatim): uniform in [4, vocab),
/// deterministic per set seed, never an unseeded draw.
fn pseudo_tokens(seed: u64, seq_len: usize, vocab: usize) -> Vec<u32> {
    let mut rng = Lcg(seed);
    (0..seq_len)
        .map(|_| 4 + (rng.next_f64() * (vocab - 4) as f64) as u32)
        .collect()
}

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

fn rfc3339_now() -> String {
    std::process::Command::new("date")
        .arg("-u")
        .arg("+%Y-%m-%dT%H:%M:%SZ")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Minimal JSON string escaping.
fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
