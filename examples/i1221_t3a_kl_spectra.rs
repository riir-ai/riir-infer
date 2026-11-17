//! i1221 T3(a) — the per-loop KL/argmax-flip spectra cell (plan 636
//! read-order leg 2; the Issue-898 `kl_depth_probe` semantics ported
//! onto the T2 delta-report posture for the Dory looped checkpoint).
//!
//! The lens ([`riir_infer_core::dory::spectra::run_lens`]) prefills once
//! at the checkpoint's trained loop count K with per-pass state
//! snapshots armed, then replays the readout on every snapshot —
//! "exiting at loop d" IS the readout of a d-loop prefill by
//! construction (the Dory readout depends only on the final recurrent
//! state; the fixtures pin the lens-premise identity at EVERY position,
//! stronger than katgpt-rs Bench 899's pos-0-only form which their
//! shared-cache readout forced).
//!
//! # Usage
//!
//! ```text
//! cargo run --release --features dory \
//!   --example i1221_t3a_kl_spectra -- \
//!   --gguf ../riir-train/data/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf \
//!   [--out .benchmarks/i1221_t3a_kl_spectra_cell] \
//!   [--seq-len 8] [--prompt-sets 3] [--seed 0] [--fresh]
//! ```
//!
//! k spans 1..=K with K = `dory.num_loops` from the header (4 on the
//! pin — the plan's "k=1..4" is exactly the trained range). Rows are
//! one per (prompt, depth), flushed immediately, resumable: the header
//! must match on every lineage field (pin blake3, STAMP, k_full,
//! seq-len, prompt-sets, seed) — a mismatch REFUSES exit 2; `--fresh`
//! starts a new lineage. Prompts: the smoke's pseudo-token LCG (per-set
//! seed = seed + prompt_idx, uniform in [4, vocab)), never an unseeded
//! draw. Report-only: the rows are measurements, never gates; the only
//! asserts are the lib fixtures and the two per-prompt real-pin
//! receipts this runner prints (kill-switch, lens premise).

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use riir_infer_core::dory::spectra::{
    agreement_exit, argmax_and_margin, argmax_of, effective_depth, exit_row, kl_loop1_vs_exit,
    kl_positions, monotone_fraction, run_lens, spectra_row, verify_resume_lineage,
};
use riir_infer_core::dory::{DoryForward, DoryKvCache, load_dory_weights};

const SCHEMA_VERSION: u32 = 1;
const ROWS_FILE: &str = "rows.jsonl";
const EXIT_THRESHOLD_NATS: f64 = 1.0;

fn main() {
    let mut gguf =
        PathBuf::from("../riir-train/data/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf");
    let mut out = PathBuf::from(".benchmarks/i1221_t3a_kl_spectra_cell");
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
        "[i1221-t3a] pin {} blake3 {}… stamp {}…",
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
        "[i1221-t3a] loaded: k_full={k_full} vocab={vocab} n_embd={n_embd} blocks={} \
         (I{} R{} O{}) cache_mode={} in {:.1}s",
        weights.blocks.len(),
        weights.meta.input_layers,
        weights.meta.recurrent_layers,
        weights.meta.output_layers,
        weights.meta.kv_cache_mode,
        t0.elapsed().as_secs_f32(),
    );

    // ── prompts: the smoke's pseudo-token LCG, per-set seed = seed + idx ──
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
    );
    let mut rows_file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&rows_path)
        .expect("open rows");
    if done.is_empty() {
        let header = format!(
            "{{\"record\":\"i1221_t3a_kl_spectra_header\",\"schema_version\":{SCHEMA_VERSION},\
             \"model\":{{\"path\":{},\"blake3\":\"{}\",\"arch\":\"dory\",\"k_full\":{},\
             \"n_embd\":{},\"vocab\":{},\"blocks\":{},\"input_layers\":{},\"recurrent_layers\":{},\
             \"output_layers\":{},\"kv_cache_mode\":\"{}\"}},\
             \"ks\":\"1..={}\",\
             \"prompts\":{{\"sets\":{},\"seq_len\":{},\"seed\":{},\
             \"law\":\"LCG uniform in [4, vocab), per-set seed = seed + prompt_idx (no tokenizer yet - the smoke's pseudo-token law)\"}},\
             \"threshold_nats\":{},\
             \"box_state\":{{\"utc_rfc3339\":\"{}\",\"profile\":\"release\",\"threads\":{}}},\
             \"repo_commit\":\"{}\"}}\n",
            json_str(&gguf.to_string_lossy()),
            pin_digest,
            k_full,
            n_embd,
            vocab,
            weights.blocks.len(),
            weights.meta.input_layers,
            weights.meta.recurrent_layers,
            weights.meta.output_layers,
            weights.meta.kv_cache_mode,
            k_full,
            prompt_sets,
            seq_len,
            seed,
            EXIT_THRESHOLD_NATS,
            rfc3339_now(),
            threads,
            repo_commit,
        );
        rows_file.write_all(header.as_bytes()).expect("write header");
        rows_file.flush().expect("flush header");
        println!("[i1221-t3a] new lineage: header written");
    } else {
        println!(
            "[i1221-t3a] resume: {} row(s) present, lineage verified — missing pairs only",
            done.len()
        );
    }

    // ── the cell ──
    let mut total_wall = 0.0f64;
    let lens_start_cell = std::time::Instant::now();
    for (p, tokens) in prompts.iter().enumerate() {
        let all_done = (1..=k_full).all(|d| done.contains(&(p, d)));
        if all_done {
            println!("[i1221-t3a] p{p}: all depths done — summary only");
            summarize_prompt(p, &rows_path, seq_len, EXIT_THRESHOLD_NATS);
            continue;
        }
        let lens_start = std::time::Instant::now();
        let mut cache = DoryKvCache::new(&weights, k_full);
        let fwd = DoryForward::new(&weights);
        let mut snaps = Vec::new();
        let full_logits = run_lens(&fwd, tokens, &mut cache, &mut snaps).expect("lens run");
        total_wall += lens_start.elapsed().as_secs_f64();

        // Receipt 1 — the kill-switch (report line, never an assert):
        // the lens run's final logits vs a plain unarmed prefill's.
        {
            let mut plain_cache = DoryKvCache::new(&weights, k_full);
            let plain = fwd.prefill(tokens, &mut plain_cache).expect("plain prefill");
            let same = logits_identical(&full_logits, &plain);
            println!("[i1221-t3a] p{p} kill-switch (lens ≡ plain final logits): {same}");
        }
        // Receipt 2 — the lens premise at its strongest Dory form: the
        // exit on the LAST snapshot ≡ the full logits at EVERY position
        // (the readout depends only on the final recurrent state).
        {
            let d_last = snaps.len();
            let mut exit_cache = DoryKvCache::new(&weights, 1);
            let exit_last =
                exit_row(&fwd, &snaps[d_last - 1], tokens.len(), &mut exit_cache).expect("exit");
            let same = logits_identical(&exit_last, &full_logits);
            println!(
                "[i1221-t3a] p{p} lens premise (exit_{d_last} ≡ full logits, all positions): {same}"
            );
        }

        // exit_1 always computed (the deepening axis's anchor); exits are
        // pure functions of the snapshots, so a resumed d=1 row stays
        // consistent.
        let mut c1 = DoryKvCache::new(&weights, 1);
        let exit1 = exit_row(&fwd, &snaps[0], tokens.len(), &mut c1).expect("exit1");

        for d in 1..=snaps.len() {
            if done.contains(&(p, d)) {
                continue;
            }
            let mut exit_cache = DoryKvCache::new(&weights, 1);
            let exit = exit_row(&fwd, &snaps[d - 1], tokens.len(), &mut exit_cache).expect("exit run");
            let kl = kl_positions(&full_logits, &exit);
            let kl1 = kl_loop1_vs_exit(&exit1, &exit);
            let (am_full, m_full) = argmax_and_margin(&full_logits[full_logits.len() - 1]);
            let (am_exit, m_exit) = argmax_and_margin(&exit[exit.len() - 1]);
            let flips = full_logits
                .iter()
                .zip(exit.iter())
                .filter(|(a, b)| argmax_of(a) != argmax_of(b))
                .count();
            let wall_ms = lens_start.elapsed().as_secs_f64() * 1000.0;
            let row = spectra_row(
                p,
                d,
                &kl,
                &kl1,
                am_full,
                am_exit,
                m_full,
                m_exit,
                flips,
                wall_ms,
            );
            let line = serde_json::to_string(&row).expect("row json");
            rows_file.write_all(line.as_bytes()).expect("write row");
            rows_file.write_all(b"\n").expect("newline");
            rows_file.flush().expect("flush row");
            println!(
                "[i1221-t3a] p{p} d={d}: KL(full‖exit) mean={:.4} max={:.4} kl1_mean={:.4} \
                 flip(last)={} flips(all)={flips}",
                row.kl_mean,
                row.kl_max,
                row.kl_loop1_mean,
                row.argmax_flip_last,
            );
        }
        summarize_prompt(p, &rows_path, seq_len, EXIT_THRESHOLD_NATS);
    }
    println!(
        "[i1221-t3a] cell done: {prompt_sets} prompt(s) × exits 1..={k_full}, \
         lens wall {total_wall:.1}s, total {:.1}s",
        lens_start_cell.elapsed().as_secs_f64()
    );
}

fn logits_identical(a: &[Vec<f32>], b: &[Vec<f32>]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.len() == y.len() && x.iter().zip(y.iter()).all(|(u, v)| u.to_bits() == v.to_bits()))
}

/// LCG pseudo-token prompt (the smoke's law): uniform in [4, vocab),
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

/// Minimal JSON string escaping for the header's path field.
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

/// Per-prompt Bench-899 summary from the rows (live run or resume — the
/// rows file is the single source; report lines, never asserts).
fn summarize_prompt(p: usize, rows_path: &Path, seq_len: usize, threshold: f64) {
    let content = fs::read_to_string(rows_path).expect("read rows");
    let mut profiles: Vec<Vec<f64>> = vec![Vec::new(); seq_len];
    let mut exit_argmaxes: Vec<Vec<usize>> = vec![Vec::new(); seq_len];
    for line in content.lines().skip(1) {
        let v: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if v.get("record").and_then(|x| x.as_str()) != Some("i1221_t3a_kl_spectra_row")
            || v.get("prompt_idx").and_then(|x| x.as_u64()) != Some(p as u64)
        {
            continue;
        }
        let d = v.get("depth").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
        if d == 0 {
            continue;
        }
        let kl: Vec<f64> = v
            .get("kl_full_vs_exit")
            .and_then(|x| x.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_f64()).collect())
            .unwrap_or_default();
        let am = v
            .get("argmax_exit_last")
            .and_then(|x| x.as_u64())
            .unwrap_or(0) as usize;
        for (pos, klv) in kl.iter().enumerate() {
            if pos < seq_len {
                let slot = &mut profiles[pos];
                while slot.len() < d {
                    slot.push(f64::NAN);
                }
                slot[d - 1] = *klv;
            }
        }
        for (_pos, slot) in exit_argmaxes.iter_mut().enumerate().take(seq_len) {
            while slot.len() < d {
                slot.push(usize::MAX);
            }
            slot[d - 1] = am;
        }
    }
    for pos in 0..seq_len {
        let ed = effective_depth(&profiles[pos], threshold);
        let ae = agreement_exit_clean(&exit_argmaxes[pos]);
        let mf = monotone_fraction(&profiles[pos]);
        println!(
            "[i1221-t3a] p{p} pos{pos}: effective_depth={ed:?} agreement_exit={ae:?} monotone={mf:.3}"
        );
    }
}

/// [`agreement_exit`] with the resume placeholder filtered (usize::MAX
/// marks a hole only a torn/partial lineage could leave; holes read
/// UNDECIDABLE, never folded into a verdict).
fn agreement_exit_clean(argmaxes: &[usize]) -> Option<usize> {
    if argmaxes.contains(&usize::MAX) {
        return None;
    }
    agreement_exit(argmaxes)
}
