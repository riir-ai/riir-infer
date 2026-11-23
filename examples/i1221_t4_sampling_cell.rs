//! i1221 T4 — the SAMPLING cell on the real looped-trained pin (plan
//! 638; the residue of Bench 1238's T4: "under SAMPLING the identity is
//! provable but the acceptance rate is an open measurement").
//!
//! Structural — NO timing rows (Bench 1238's timing rows stay parked
//! for a preflight-PASS window; this cell never quotes a wall number,
//! so it may run concurrent-with-light-load per plan 636, disclosed).
//!
//! Arms per prompt at T = 1 (the raw k_full row's softmax — untruncated,
//! the identity's cleanest posture): self-spec sampling at γ = 2 and
//! γ = 4 (k=1 draft, k_full lazy verify), plus a SECOND γ = 4 pass at
//! the same seed (the determinism receipt). Each arm writes one row
//! carrying:
//!
//! - the stream digest + the round counters (the acceptance census);
//! - the INDUCED-LAW RECEIPT: every (p, q) pair the judge read passes
//!   `max |induced − p| ≤ max(1e-12, 4·V·ε)` (1.2e-10 at the pin's V) —
//!   the real pin's identity evidence, exact and load-independent
//!   (asserted in-process: a miss refuses the cell);
//! - the KL(p‖q) census — what the pin's depth pairs actually diverge
//!   by (the synth's near-no-op is not this pin's regime);
//! - the first-token two-sample table on the CAPTURED position-0 rows
//!   (the forward ran once; the draws are pure steps): REPORT-ONLY — at
//!   any affordable N the 131k-vocab TV is noise-dominated, so the table
//!   carries its own noise expectation and gates nothing.
//!
//! Resumable JSONL (plan 636's law): header lineage verified on every
//! field; completed (prompt, arm) pairs skipped whole; mismatch REFUSES
//! exit 2.
//!
//! ```text
//! cargo run --release --features dory --example i1221_t4_sampling_cell -- \
//!   --gguf ../riir-train/data/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf \
//!   [--out .benchmarks/i1221_t4_sampling_cell] [--steps 12] [--prompts 2] \
//!   [--seed 0] [--draws 20000] [--fresh]
//! ```

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use riir_infer_core::dory::self_spec::{
    SampleCapture, SampleOpts, SpecRng, induced_into, sample_categorical, self_spec_sample,
    spec_step,
};
use riir_infer_core::dory::spectra::verify_resume_lineage;
use riir_infer_core::dory::{DoryForward, DoryKvCache, load_dory_weights};

const SCHEMA_VERSION: u32 = 1;
const ROWS_FILE: &str = "rows.jsonl";
const ROW_RECORD: &str = "i1221_t4_sampling_row";
const HEADER_RECORD: &str = "i1221_t4_sampling_header";
/// (arm, γ): arm 0 = γ 2, arm 1 = γ 4, arm 2 = γ 4 again at the same
/// seed (the determinism receipt).
const ARMS: [(usize, usize); 3] = [(0, 2), (1, 4), (2, 4)];

fn main() {
    let mut gguf = PathBuf::from("../riir-train/data/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf");
    let mut out = PathBuf::from(".benchmarks/i1221_t4_sampling_cell");
    let mut steps = 12usize;
    let mut prompts = 2usize;
    let mut seed: u64 = 0;
    let mut draws = 20_000usize;
    let mut fresh = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--gguf" => gguf = PathBuf::from(args.next().expect("--gguf PATH")),
            "--out" => out = PathBuf::from(args.next().expect("--out DIR")),
            "--steps" => steps = args.next().expect("--steps N").parse().expect("N"),
            "--prompts" => prompts = args.next().expect("--prompts N").parse().expect("N"),
            "--seed" => seed = args.next().expect("--seed N").parse().expect("N"),
            "--draws" => draws = args.next().expect("--draws N").parse().expect("N"),
            "--fresh" => fresh = true,
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(2);
            }
        }
    }

    // ── lineage ──
    let pin_digest = blake3_file(&gguf);
    let repo_commit = repo_stamp();
    println!(
        "[i1221-t4s] pin {} blake3 {}… stamp {}…",
        gguf.display(),
        &pin_digest[..16],
        &repo_commit[..12.min(repo_commit.len())]
    );

    // ── load ──
    let t0 = std::time::Instant::now();
    let weights = load_dory_weights(&gguf).expect("load dory weights");
    let k_full = weights.meta.num_loops;
    let vocab = weights.dims.vocab;
    let induced_bar = f64::max(1e-12, 4.0 * vocab as f64 * f64::EPSILON);
    println!(
        "[i1221-t4s] loaded: k_full={k_full} vocab={vocab} in {:.1}s; steps={steps} prompts={prompts} draws={draws} (structural: no timing rows)",
        t0.elapsed().as_secs_f32()
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
        let load = std::process::Command::new("uptime")
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_else(|| "unknown".to_string());
        let header = format!(
            "{{\"record\":\"{HEADER_RECORD}\",\"schema_version\":{SCHEMA_VERSION},\
             \"model\":{{\"path\":{},\"blake3\":\"{}\",\"arch\":\"dory\",\"k_full\":{},\"vocab\":{}}},\
             \"prompts\":{{\"sets\":{},\"seq_len\":8,\"seed\":{},\"law\":\"LCG uniform in [4, vocab), per-set seed = seed + prompt_idx (the lane law)\"}},\
             \"steps\":{},\"arms\":[{{\"arm\":0,\"gamma\":2}},{{\"arm\":1,\"gamma\":4}},{{\"arm\":2,\"gamma\":4,\"note\":\"determinism receipt: arm 1's seed again\"}}],\
             \"temperature\":1.0,\
             \"timing\":\"none (structural cell — plan 636: concurrent-with-light-load allowed, disclosed)\",\
             \"concurrency_disclosure\":{},\
             \"box_state\":{{\"utc_rfc3339\":\"{}\",\"profile\":\"release\"}},\
             \"repo_commit\":\"{}\"}}\n",
            json_str(&gguf.to_string_lossy()),
            pin_digest,
            k_full,
            vocab,
            prompts,
            seed,
            steps,
            json_str(&format!("uptime at start: {load}")),
            rfc3339_now(),
            repo_commit,
        );
        rows_file
            .write_all(header.as_bytes())
            .expect("write header");
        rows_file.flush().expect("flush header");
        println!("[i1221-t4s] new lineage: header written (concurrency disclosed: {load})");
    } else {
        println!(
            "[i1221-t4s] resume: {} arm-cell(s) present, lineage verified — missing only",
            done.len()
        );
    }

    // ── the cell ──
    let fwd = DoryForward::new(&weights);
    let mut scratch = vec![0.0f64; vocab];
    let mut induced = vec![0.0f64; vocab];
    let mut ref_digest: Option<String> = None;
    for (p, tokens) in toks.iter().enumerate() {
        let mut first_pair: Option<(Vec<f64>, Vec<f64>)> = None;
        for (arm, gamma) in ARMS {
            if done.contains(&(p, arm)) {
                println!("[i1221-t4s] p{p} arm{arm} (γ={gamma}): done — skipped");
                continue;
            }
            let mut verify_cache = DoryKvCache::new(&weights, k_full);
            let mut draft_cache = DoryKvCache::new(&weights, 1);
            // Arms 1 and 2 share a seed (the determinism receipt); arm 0
            // has its own.
            let arm_seed = seed
                ^ (p as u64)
                    .wrapping_mul(0x9E37_79B9)
                    .wrapping_add(if arm == 0 { 11 } else { 17 });
            let mut rng = SpecRng::new(arm_seed);
            let mut cap = SampleCapture::default();
            let mut opts = SampleOpts {
                temperature: 1.0,
                rng: &mut rng,
                capture: Some(&mut cap),
            };
            let (stream, st) = self_spec_sample(
                &fwd,
                tokens,
                steps,
                gamma,
                &mut verify_cache,
                &mut draft_cache,
                &mut opts,
            )
            .expect("self spec sample");
            let digest = blake3_stream(&stream);

            // The induced-law receipt over every pair the judge read, plus
            // the DIVERGENCE CENSUS in the law's own units: `Σ min(p, q)`
            // is the acceptance a q-drawn proposal is PREDICTED to receive
            // at that position (finite; a KL is not — `Σ p·ln(p/q)` is +∞
            // on a 131k vocab wherever q underflows to exactly 0 with
            // p > 0, which the first run serialized as null). The
            // support-restricted KL (tokens with p ≥ 1e-9, q floored) is
            // carried as the nats-scale reading.
            let mut worst = 0.0f64;
            let mut accept_law_sum = 0.0f64;
            let mut accept_law_min = 1.0f64;
            let mut expected_rejected = 0.0f64;
            let mut kl_sum = 0.0f64;
            let mut kl_max = 0.0f64;
            let mut judged = 0usize;
            for (pp, qq) in cap.pairs() {
                induced_into(pp, qq, &mut induced, &mut scratch);
                let d = pp
                    .iter()
                    .zip(induced.iter())
                    .map(|(&a, &b)| (a - b).abs())
                    .fold(0.0, f64::max);
                worst = worst.max(d);
                let law_accept: f64 = pp.iter().zip(qq.iter()).map(|(&a, &b)| a.min(b)).sum();
                accept_law_sum += law_accept;
                accept_law_min = accept_law_min.min(law_accept);
                expected_rejected += 1.0 - law_accept;
                let kl = kl_support_nats(pp, qq);
                kl_sum += kl;
                kl_max = kl_max.max(kl);
                judged += 1;
            }
            if judged > 0 {
                assert!(
                    worst <= induced_bar,
                    "p{p} arm{arm}: the induced law deviates from the target by {worst:.3e} > {induced_bar:.1e} on the REAL pin — refusing the cell"
                );
            }
            // The determinism receipt: arm 2 must reproduce arm 1.
            if arm == 1 {
                ref_digest = Some(digest.clone());
                first_pair = cap.pairs().next().map(|(a, b)| (a.to_vec(), b.to_vec()));
            }
            if arm == 2 {
                let want = ref_digest
                    .clone()
                    .unwrap_or_else(|| reference_digest_for(&rows_path, p, 1).expect("arm 1 row"));
                assert_eq!(
                    digest, want,
                    "p{p}: the second γ=4 pass at the same seed diverged — determinism FAILED"
                );
            }

            let b = &st.base;
            write_row(
                &mut rows_file,
                serde_json::json!({
                    "record": ROW_RECORD,
                    "prompt_idx": p,
                    "arm": arm,
                    "gamma": gamma,
                    "steps": steps,
                    "emitted": stream.len(),
                    "stream_blake3": digest,
                    "stats": serde_json::json!(st),
                    "acceptance_rate": if b.drafted > 0 {
                        serde_json::json!(b.accepted as f64 / b.drafted as f64)
                    } else {
                        serde_json::Value::Null
                    },
                    "induced_receipt": {
                        "judged_pairs": judged,
                        "max_abs_dev": worst,
                        "bar": induced_bar,
                    },
                    "divergence_census": {
                        "judged_pairs": judged,
                        "acceptance_law_mean": if judged > 0 { serde_json::json!(accept_law_sum / judged as f64) } else { serde_json::Value::Null },
                        "acceptance_law_min": if judged > 0 { serde_json::json!(accept_law_min) } else { serde_json::Value::Null },
                        "expected_rejected": expected_rejected,
                        "observed_rejected": st.rejected,
                        "kl_support_nats": {
                            "mean": if judged > 0 { serde_json::json!(kl_sum / judged as f64) } else { serde_json::Value::Null },
                            "max": kl_max,
                            "support": "tokens with p >= 1e-9; q floored at 1e-300 (an unrestricted KL is +inf on this vocab)",
                        },
                    },
                }),
            );
            println!(
                "[i1221-t4s] p{p} arm{arm} γ={gamma}: {} tok · rounds {} accepted {}/{} ({:.1}%) rejected {} bonus {} · tok/round {:.2} · induced max|Δ| {:.1e} (bar {:.1e}, {judged} pairs) · predicted accept (Σmin) mean {:.3} min {:.3} · expected rejected {:.2} vs observed {} · KL(support) mean {:.3} max {:.3} nats{}",
                stream.len(),
                b.rounds,
                b.accepted,
                b.drafted,
                100.0 * b.accepted as f64 / b.drafted as f64,
                st.rejected,
                b.bonus,
                st.emitted_uncapped as f64 / b.rounds as f64,
                worst,
                induced_bar,
                accept_law_sum / judged.max(1) as f64,
                accept_law_min,
                expected_rejected,
                st.rejected,
                kl_sum / judged.max(1) as f64,
                kl_max,
                if arm == 2 { " · DETERMINISM OK" } else { "" },
            );
        }

        // The first-token two-sample table on the captured position-0
        // rows (report-only; its noise expectation travels with it).
        if let Some((p0, q0)) = first_pair.take() {
            let table = first_token_table(&p0, &q0, draws, seed ^ 0x5EED_0000 ^ p as u64);
            write_row(
                &mut rows_file,
                serde_json::json!({
                    "record": "i1221_t4_sampling_first_token",
                    "prompt_idx": p,
                    "draws": draws,
                    "tv_ref_vs_p": table.tv_ref,
                    "tv_spec_vs_p": table.tv_spec,
                    "tv_spec_vs_ref": table.tv_spec_ref,
                    "noise_expectation": format!(
                        "a perfect sampler's TV vs p at N={draws} over a distribution with these top masses is ≈ {:.3}; the bars are not gates",
                        table.tv_expected
                    ),
                    "p_max": table.p_max,
                    "q_max": table.q_max,
                    "top": table.rows,
                }),
            );
            println!(
                "[i1221-t4s] p{p} first-token table: TV(ref,p)={:.4} TV(spec,p)={:.4} TV(spec,ref)={:.4} (noise expectation ≈ {:.3}; report-only)",
                table.tv_ref, table.tv_spec, table.tv_spec_ref, table.tv_expected
            );
        }
    }
    println!(
        "[i1221-t4s] cell done: {prompts} prompt(s) × {} arm(s) × {steps} steps (structural; no timing rows)",
        ARMS.len()
    );
}

struct FirstTokenTable {
    tv_ref: f64,
    tv_spec: f64,
    tv_spec_ref: f64,
    tv_expected: f64,
    p_max: f64,
    q_max: f64,
    rows: Vec<serde_json::Value>,
}

/// The two-sample first-token table on the captured (p₀, q₀): `draws`
/// direct draws from p₀ (the reference) and `draws` pure judge steps
/// (draft from q₀, then `spec_step`). Report-only.
fn first_token_table(p0: &[f64], q0: &[f64], draws: usize, seed: u64) -> FirstTokenTable {
    let v = p0.len();
    let mut ref_hist = vec![0u64; v];
    let mut spec_hist = vec![0u64; v];
    let mut scratch = vec![0.0f64; v];
    let mut rng = SpecRng::new(seed);
    for _ in 0..draws {
        ref_hist[sample_categorical(p0, rng.next_f64())] += 1;
        let t = sample_categorical(q0, rng.next_f64());
        let (x, _) = spec_step(p0, q0, t, rng.next_f64(), rng.next_f64(), &mut scratch);
        spec_hist[x] += 1;
    }
    let tv = |a: &[u64], law: &[f64]| {
        a.iter()
            .zip(law.iter())
            .map(|(&c, &pi)| (c as f64 / draws as f64 - pi).abs())
            .sum::<f64>()
            * 0.5
    };
    // A perfect sampler's expected TV vs its law: Σ E|f_i − p_i| with
    // the binomial's mean absolute deviation, summed ascending.
    let tv_expected: f64 = p0
        .iter()
        .map(|&pi| {
            let n = draws as f64;
            let var = n * pi * (1.0 - pi);
            // E|X − np| ≈ sqrt(2 var / π) for np ≥ ~5; ≈ 2 np (1−p) for
            // the rare-bin regime.
            let e = if n * pi >= 5.0 {
                (2.0 * var / std::f64::consts::PI).sqrt()
            } else {
                2.0 * n * pi * (1.0 - pi)
            };
            e / n
        })
        .sum();
    // The top rows by target mass.
    let mut order: Vec<usize> = (0..v).collect();
    order.sort_by(|&a, &b| {
        p0[b]
            .partial_cmp(&p0[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let rows = order
        .iter()
        .take(8)
        .map(|&i| {
            serde_json::json!({
                "token": i,
                "p": p0[i],
                "q": q0[i],
                "ref_freq": ref_hist[i] as f64 / draws as f64,
                "spec_freq": spec_hist[i] as f64 / draws as f64,
            })
        })
        .collect();
    FirstTokenTable {
        tv_ref: tv(&ref_hist, p0),
        tv_spec: tv(&spec_hist, p0),
        tv_spec_ref: {
            let n = draws as f64;
            spec_hist
                .iter()
                .zip(ref_hist.iter())
                .map(|(&a, &b)| ((a - b) as f64 / n).abs())
                .sum::<f64>()
                * 0.5
        },
        tv_expected,
        p_max: p0.iter().copied().fold(0.0, f64::max),
        q_max: q0.iter().copied().fold(0.0, f64::max),
        rows,
    }
}

/// KL(p ‖ q) over the support that carries mass: tokens with
/// `p ≥ 1e-9`, q floored at 1e-300. The unrestricted sum is +∞ on a
/// large vocab (q underflows to exactly 0 where p > 0), which says
/// nothing about the two laws' usable disagreement — the first run's
/// rows serialized that as null.
fn kl_support_nats(p: &[f64], q: &[f64]) -> f64 {
    const SUPPORT: f64 = 1e-9;
    const FLOOR: f64 = 1e-300;
    let mut acc = 0.0f64;
    for (&a, &b) in p.iter().zip(q.iter()) {
        if a >= SUPPORT {
            acc += a * (a / b.max(FLOOR)).ln();
        }
    }
    acc
}

fn write_row(f: &mut fs::File, row: serde_json::Value) {
    let line = serde_json::to_string(&row).expect("row json");
    f.write_all(line.as_bytes()).expect("write row");
    f.write_all(b"\n").expect("newline");
    f.flush().expect("flush row");
}

/// The stream digest of an already-landed arm row (the determinism
/// receipt on resume reads arm 1's row back).
fn reference_digest_for(rows_path: &Path, prompt_idx: usize, arm: usize) -> Option<String> {
    let content = fs::read_to_string(rows_path).ok()?;
    content.lines().skip(1).find_map(|line| {
        let v: serde_json::Value = serde_json::from_str(line).ok()?;
        (v.get("record").and_then(|x| x.as_str()) == Some(ROW_RECORD)
            && v.get("prompt_idx").and_then(|x| x.as_u64()) == Some(prompt_idx as u64)
            && v.get("arm").and_then(|x| x.as_u64()) == Some(arm as u64))
        .then(|| {
            v.get("stream_blake3")
                .and_then(|x| x.as_str())
                .expect("row carries stream_blake3")
                .to_string()
        })
    })
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
