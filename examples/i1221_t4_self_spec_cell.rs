//! i1221 T4 — the depth self-spec cell (the real looped-trained pin).
//!
//! Arms per prompt: the k=4 AR greedy reference, then self-spec at
//! γ=2 and γ=4 (k=1 draft, k=4 verify, greedy EXACT rejection). The
//! in-process gate: the self-spec stream is token-identical to the
//! reference (asserted; the run refuses on divergence — this is the
//! pre-registered gate, not a report line). The rows carry the stream
//! digest + the round counters + wall ms; the record's cost table
//! reads them (sequential CPU expectation: spec ≈ 1.25× AR compute —
//! a latency structure, never a compute win on one thread).
//!
//! Resumable JSONL (plan 636's law): header lineage verified on every
//! field; completed (prompt, arm) pairs skipped whole; mismatch
//! REFUSES exit 2. Timing rows carry bench_preflight PROVENANCE in the
//! header — run `../riir-reflex/scripts/bench_preflight.sh` first and
//! pass `--provenance "<line>"`; without it the timing columns are
//! written as null (a run without provenance never quotes a number).
//!
//! ```text
//! cargo run --release --features dory \
//!   --example i1221_t4_self_spec_cell -- \
//!   --gguf ../riir-train/data/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf \
//!   [--out .benchmarks/i1221_t4_self_spec_cell] \
//!   [--steps 12] [--prompts 2] [--seed 0] [--fresh] \
//!   [--provenance "PROVENANCE: ..."]
//! ```

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use riir_infer_core::dory::self_spec::{ar_greedy, self_spec_greedy};
use riir_infer_core::dory::spectra::verify_resume_lineage;
use riir_infer_core::dory::{DoryForward, DoryKvCache, load_dory_weights};

const SCHEMA_VERSION: u32 = 1;
const ROWS_FILE: &str = "rows.jsonl";
const ROW_RECORD: &str = "i1221_t4_self_spec_row";
const GAMMAS: [usize; 2] = [2, 4];

fn main() {
    let mut gguf = PathBuf::from("../riir-train/data/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf");
    let mut out = PathBuf::from(".benchmarks/i1221_t4_self_spec_cell");
    let mut steps = 12usize;
    let mut prompts = 2usize;
    let mut seed: u64 = 0;
    let mut fresh = false;
    let mut provenance = String::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--gguf" => gguf = PathBuf::from(args.next().expect("--gguf PATH")),
            "--out" => out = PathBuf::from(args.next().expect("--out DIR")),
            "--steps" => steps = args.next().expect("--steps N").parse().expect("N"),
            "--prompts" => prompts = args.next().expect("--prompts N").parse().expect("N"),
            "--seed" => seed = args.next().expect("--seed N").parse().expect("N"),
            "--fresh" => fresh = true,
            "--provenance" => provenance = args.next().expect("--provenance LINE"),
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(2);
            }
        }
    }
    // Timing rows are legal ONLY with bench_preflight provenance (the
    // box-state law): absent --provenance, wall_ms writes as null and
    // no s/tok number is printed — identity + counters only.
    let timed = !provenance.is_empty();

    // ── lineage ──
    let pin_digest = blake3_file(&gguf);
    let repo_commit = repo_stamp();
    println!(
        "[i1221-t4] pin {} blake3 {}… stamp {}…",
        gguf.display(),
        &pin_digest[..16],
        &repo_commit[..12.min(repo_commit.len())]
    );

    // ── load ──
    let t0 = std::time::Instant::now();
    let weights = load_dory_weights(&gguf).expect("load dory weights");
    let k_full = weights.meta.num_loops;
    let vocab = weights.dims.vocab;
    println!(
        "[i1221-t4] loaded: k_full={k_full} vocab={vocab} in {:.1}s; steps={steps} prompts={prompts} gammas={GAMMAS:?}",
        t0.elapsed().as_secs_f32(),
    );

    // ── prompts: the lane's LCG law ──
    let toks: Vec<Vec<u32>> = (0..prompts)
        .map(|p| pseudo_tokens(seed + p as u64, 8, vocab))
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
        8,
        prompts,
        seed,
        ROW_RECORD,
    );
    let mut rows_file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&rows_path)
        .expect("open rows");
    if done.is_empty() {
        let header = format!(
            "{{\"record\":\"i1221_t4_self_spec_header\",\"schema_version\":{SCHEMA_VERSION},\
             \"model\":{{\"path\":{},\"blake3\":\"{}\",\"arch\":\"dory\",\"k_full\":{},\
             \"vocab\":{}}},\
             \"prompts\":{{\"sets\":{},\"seq_len\":8,\"seed\":{},\
             \"law\":\"LCG uniform in [4, vocab), per-set seed = seed + prompt_idx (the lane law)\"}},\
             \"steps\":{},\"gammas\":{},\
             \"provenance\":{},\
             \"box_state\":{{\"utc_rfc3339\":\"{}\",\"profile\":\"release\"}},\
             \"repo_commit\":\"{}\"}}\n",
            json_str(&gguf.to_string_lossy()),
            pin_digest,
            k_full,
            vocab,
            prompts,
            seed,
            steps,
            json_str(&format!("{GAMMAS:?}")),
            if provenance.is_empty() {
                "null".to_string()
            } else {
                json_str(&provenance)
            },
            rfc3339_now(),
            repo_commit,
        );
        rows_file
            .write_all(header.as_bytes())
            .expect("write header");
        rows_file.flush().expect("flush header");
        println!("[i1221-t4] new lineage: header written");
    } else {
        println!(
            "[i1221-t4] resume: {} arm-cell(s) present, lineage verified — missing only",
            done.len()
        );
    }

    // ── the cell ──
    for (p, tokens) in toks.iter().enumerate() {
        // Reference arm: k_full AR greedy under per-loop caches.
        if !done.contains(&(p, 0)) {
            let mut cache = DoryKvCache::new(&weights, k_full);
            let fwd = DoryForward::new(&weights);
            let t = std::time::Instant::now();
            let stream = ar_greedy(&fwd, tokens, steps, &mut cache).expect("ar greedy");
            let wall_ms = if timed {
                t.elapsed().as_secs_f64() * 1e3
            } else {
                0.0
            };
            write_row(
                &mut rows_file,
                serde_json::json!({
                    "record": ROW_RECORD,
                    "prompt_idx": p,
                    "arm": 0,
                    "arm_name": "ar_greedy_k_full",
                    "steps": steps,
                    "emitted": stream.len(),
                    "stream_blake3": blake3_stream(&stream),
                    "wall_ms": if timed { serde_json::json!(wall_ms) } else { serde_json::Value::Null },
                    "stats": {"rounds": steps, "drafted": 0, "accepted": 0,
                              "bonus": 0, "draft_steps": 0, "verify_steps": steps},
                }),
            );
            if timed {
                println!(
                    "[i1221-t4] p{p} ref: {} tok in {:.1}s ({:.2} s/tok)",
                    stream.len(),
                    wall_ms / 1e3,
                    wall_ms / 1e3 / steps as f64
                );
            } else {
                println!(
                    "[i1221-t4] p{p} ref: {} tok (untimed — no provenance)",
                    stream.len()
                );
            }
        } else {
            println!("[i1221-t4] p{p} ref: done — skipped");
        }

        // Self-spec arms.
        for (gi, &gamma) in GAMMAS.iter().enumerate() {
            let arm = gi + 1; // (prompt, 1..=gamma-index) row keys
            if done.contains(&(p, arm)) {
                println!("[i1221-t4] p{p} g{gamma}: done — skipped");
                continue;
            }
            let mut verify_cache = DoryKvCache::new(&weights, k_full);
            let mut draft_cache = DoryKvCache::new(&weights, 1);
            let fwd = DoryForward::new(&weights);
            let t = std::time::Instant::now();
            let (stream, st) = self_spec_greedy(
                &fwd,
                tokens,
                steps,
                gamma,
                &mut verify_cache,
                &mut draft_cache,
            )
            .expect("self spec");
            let wall_ms = if timed {
                t.elapsed().as_secs_f64() * 1e3
            } else {
                0.0
            };

            // THE gate, in-process on the real pin: the stream must
            // equal the reference token-for-token. Read the reference
            // row from the rows file (it landed above / on resume).
            let ref_digest = reference_digest_for(&rows_path, p).unwrap_or_else(|e| {
                panic!("reference row for p{p} unreadable: {e} (the ref arm must land before the spec arms)")
            });
            let got = blake3_stream(&stream);
            assert_eq!(
                got, ref_digest,
                "p{p} g{gamma}: self-spec stream diverged from the k={k_full} AR reference — THE T4 gate FAILED"
            );

            write_row(
                &mut rows_file,
                serde_json::json!({
                    "record": ROW_RECORD,
                    "prompt_idx": p,
                    "arm": arm,
                    "arm_name": "self_spec_greedy",
                    "steps": steps,
                    "emitted": stream.len(),
                    "stream_blake3": blake3_stream(&stream),
                    "wall_ms": if timed { serde_json::json!(wall_ms) } else { serde_json::Value::Null },
                    "stats": serde_json::json!(st),
                }),
            );
            if timed {
                println!(
                    "[i1221-t4] p{p} g{gamma}: {} tok in {:.1}s ({:.2} s/tok) — rounds {} accepted {}/{} bonus {} (IDENTITY OK)",
                    stream.len(),
                    wall_ms / 1e3,
                    wall_ms / 1e3 / st.emitted.max(1) as f64,
                    st.rounds,
                    st.accepted,
                    st.drafted,
                    st.bonus
                );
            } else {
                println!(
                    "[i1221-t4] p{p} g{gamma}: {} tok (untimed) — rounds {} accepted {}/{} bonus {} (IDENTITY OK)",
                    stream.len(),
                    st.rounds,
                    st.accepted,
                    st.drafted,
                    st.bonus
                );
            }
        }
    }
    println!(
        "[i1221-t4] cell done: {prompts} prompt(s) × {} arm(s) × {steps} steps",
        GAMMAS.len() + 1
    );
}

fn write_row(f: &mut fs::File, row: serde_json::Value) {
    let line = serde_json::to_string(&row).expect("row json");
    f.write_all(line.as_bytes()).expect("write row");
    f.write_all(b"\n").expect("newline");
    f.flush().expect("flush row");
}

/// The reference arm's stream digest for `prompt_idx`, read back from
/// the rows file (the identity gate compares digests, so the reference
/// never needs a second decode).
fn reference_digest_for(rows_path: &Path, prompt_idx: usize) -> Result<String, std::io::Error> {
    let content = fs::read_to_string(rows_path)?;
    for line in content.lines().skip(1) {
        let v: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if v.get("record").and_then(|x| x.as_str()) == Some(ROW_RECORD)
            && v.get("prompt_idx").and_then(|x| x.as_u64()) == Some(prompt_idx as u64)
            && v.get("arm").and_then(|x| x.as_u64()) == Some(0)
        {
            return Ok(v
                .get("stream_blake3")
                .and_then(|x| x.as_str())
                .expect("row carries stream_blake3")
                .to_string());
        }
    }
    panic!(
        "reference row for p{prompt_idx} not found (the ref arm must land before the spec arms)"
    );
}

fn blake3_stream(stream: &[u32]) -> String {
    let mut h = blake3::Hasher::new();
    for t in stream {
        h.update(&t.to_le_bytes());
    }
    h.finalize().to_hex().to_string()
}

/// LCG pseudo-token prompt (the lane law): uniform in [4, vocab).
fn pseudo_tokens(seed: u64, seq_len: usize, vocab: usize) -> Vec<u32> {
    let mut rng = Lcg(seed | 1);
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
