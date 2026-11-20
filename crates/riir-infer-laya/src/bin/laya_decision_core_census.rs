//! Issue 1237 — the decision-core census over a laya checkpoint (ACE at the
//! R1 readout: the head's scorer-input marker rows).
//!
//! ```sh
//! cargo run --release -p riir-infer-laya --features decision_core_census,laya-riir-metal \
//!   --bin laya-decision-core-census -- census --suite <tetris_v4.jsonl> \
//!   --checkpoint english --records 200 --out <dir>
//! cargo run ... -- compare --a <dir>/english_core.f64 --b <dir>/typed_core.f64 --r 32
//! ```
//!
//! `census` (the pre-registration in `.issues/1237`): every option of the first
//! `--records` suite records runs the oracle's `noul` forward
//! (`<state_sentence>\n<option sentence>`); the two marker rows (false, true)
//! are the samples, the decision is `sign(l_true − l_false)`. Records
//! `[0, n/2)` FIT (Cov + the analytic relevance moment), `[n/2, n)` EVALUATE
//! the causal keep/remove reads against the PCA (`S = I`, relevance_core's
//! documented positive control) and seeded-random controls at every rank.
//! Writes `<ckpt>_report.json` + `<ckpt>_core.f64` (r_max × d, LE f64).
//!
//! `compare`: principal-angle basis similarity of two cores at rank `r`
//! against the random baseline `r/d`.

use std::path::{Path, PathBuf};

use katgpt_core::relevance_core::{
    CoreScratch, RankRule, accumulate_sens, core_rank_energy, extract_core_from_matrices,
};
use katgpt_core::subspace_intervention::{basis_similarity, random_basis_into};
use riir_infer_laya::laya::config::Checkpoint;
use riir_infer_laya::laya::riir::RiirAgent;
use riir_infer_laya::laya::riir::decision_core::{CoreProjector, Scorer64, ScorerScratch};
use riir_infer_laya::laya::weights::weights_root;
use serde_json::{Value, json};

const RANKS: [usize; 7] = [1, 2, 4, 8, 16, 32, 64];
const ENERGY: f64 = 0.90;
const RANDOM_SEED: u64 = 0x1237_ACE0;
/// Fidelity gate: forwards whose raw lane logits are compared to
/// [`Scorer64`] (f64 re-expression of the same scorer).
const FIDELITY_FORWARDS: usize = 32;
const FIDELITY_TOL: f64 = 1e-3;

fn die(msg: &str) -> ! {
    eprintln!("laya-decision-core-census: {msg}");
    std::process::exit(2);
}

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1).cloned())
}

fn checkpoint(name: &str) -> Checkpoint {
    match name {
        "english" => Checkpoint::English,
        "typed" | "typed-decisions" => Checkpoint::TypedDecisions,
        "multilingual" => Checkpoint::Multilingual,
        other => die(&format!("unknown checkpoint {other:?}")),
    }
}

/// One forward's samples: the two marker rows (false, true), its record.
struct Sample {
    record: usize,
    rows: [Vec<f64>; 2],
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("census") => census(&args),
        Some("census-r0") => census_r0(&args),
        Some("compare") => compare(&args),
        _ => die(
            "usage: census --suite P --checkpoint C --records N --out DIR | compare --a F --b F --r R",
        ),
    }
}

/// The suite file's BLAKE3 + its first `n` non-meta records.
fn load_suite(path: &str, n: usize) -> (String, Vec<Value>) {
    let text = std::fs::read_to_string(path).expect("read suite");
    let digest = blake3::hash(text.as_bytes()).to_hex().to_string();
    let records: Vec<Value> = text
        .lines()
        .map(|l| serde_json::from_str(l).expect("suite JSONL"))
        .filter(|r: &Value| r["state_id"] != "_meta")
        .take(n)
        .collect();
    if records.len() < 2 {
        die("suite has fewer than 2 records");
    }
    (digest, records)
}

/// One oracle forward: the `noul` question over `<state>\n<option>`.
struct SuiteForward {
    record: usize,
    payload: Value,
    qdef: Value,
}

/// Every option of every record as its oracle forward, in suite order.
fn suite_forwards(records: &[Value]) -> Vec<SuiteForward> {
    let mut out = Vec::new();
    for (record, rec) in records.iter().enumerate() {
        let question = rec["question"].as_str().expect("question");
        let state_line = rec["state_sentence"].as_str();
        let qdef = json!({"type": "noul", "instructions": question, "criteria": null});
        for opt in rec["options"].as_array().expect("options") {
            let sentence = opt["sentence"].as_str().expect("sentence");
            let payload = Value::String(match state_line {
                Some(s) => format!("{s}\n{sentence}"),
                None => sentence.to_string(),
            });
            out.push(SuiteForward {
                record,
                payload,
                qdef: qdef.clone(),
            });
        }
    }
    out
}

fn census(args: &[String]) {
    let suite = arg(args, "--suite").unwrap_or_else(|| die("--suite required"));
    let ckpt_name = arg(args, "--checkpoint").unwrap_or_else(|| "english".into());
    let n_records: usize = arg(args, "--records").map_or(200, |v| v.parse().expect("--records"));
    let out = PathBuf::from(arg(args, "--out").unwrap_or_else(|| die("--out required")));
    std::fs::create_dir_all(&out).expect("create --out");

    let (suite_blake3, records) = load_suite(&suite, n_records);
    let fit_end = records.len() / 2;

    let agent = RiirAgent::load(&weights_root(), checkpoint(&ckpt_name))
        .unwrap_or_else(|e| die(&format!("load {ckpt_name}: {e}")));
    let scorer = Scorer64::new(&agent.scorer_tensors());
    let d = scorer.dim();
    let mut ssc = ScorerScratch::new(d);
    eprintln!(
        "loaded {ckpt_name} (d {d}, device {}); {} records, fit [0,{fit_end})",
        agent.device(),
        records.len()
    );

    // ── Forwards ────────────────────────────────────────────────────────
    let t0 = std::time::Instant::now();
    let mut samples: Vec<Sample> = Vec::new();
    let mut fidelity_max = 0.0f64;
    let forwards = suite_forwards(&records);
    for (
        fi,
        SuiteForward {
            record,
            payload,
            qdef,
        },
    ) in forwards.iter().enumerate()
    {
        let enc = agent
            .encode_question(payload, qdef)
            .unwrap_or_else(|e| die(&format!("encode: {e}")));
        if enc.markers.len() != 2 {
            die("noul forward must carry exactly two markers");
        }
        let rows = [
            enc.marker_rows[..d]
                .iter()
                .map(|&x| f64::from(x))
                .collect::<Vec<_>>(),
            enc.marker_rows[d..2 * d]
                .iter()
                .map(|&x| f64::from(x))
                .collect(),
        ];
        if samples.len() < FIDELITY_FORWARDS {
            let fwd = agent
                .forward_question(payload, qdef)
                .unwrap_or_else(|e| die(&format!("forward: {e}")));
            for (m, row) in rows.iter().enumerate() {
                let l = scorer.logit(row, &mut ssc);
                fidelity_max = fidelity_max.max((l - f64::from(fwd.logits[m])).abs());
            }
        }
        samples.push(Sample {
            record: *record,
            rows,
        });
        if fi % 500 == 0 {
            eprintln!(
                "  forward {fi}/{}: {:.1}s",
                forwards.len(),
                t0.elapsed().as_secs_f64()
            );
        }
    }
    let forward_secs = t0.elapsed().as_secs_f64();
    if fidelity_max > FIDELITY_TOL {
        die(&format!(
            "fidelity gate FAILED: Scorer64 vs lane logits max |Δ| {fidelity_max:.3e} > {FIDELITY_TOL:e}"
        ));
    }
    let (fit, eval): (Vec<&Sample>, Vec<&Sample>) =
        samples.iter().partition(|s| s.record < fit_end);
    let n_fit_rows = 2 * fit.len();

    // ── Fit: mean, Cov, relevance moment ───────────────────────────────
    let mut mean = vec![0.0; d];
    for s in &fit {
        for row in &s.rows {
            for (m, x) in mean.iter_mut().zip(row) {
                *m += x;
            }
        }
    }
    mean.iter_mut().for_each(|m| *m /= n_fit_rows as f64);
    let mut cov = vec![0.0; d * d];
    let mut sens = vec![0.0; d * d];
    let mut n_seen = 0u64;
    let mut centered = vec![0.0; d];
    let mut grad = vec![0.0; d];
    for s in &fit {
        for row in &s.rows {
            for i in 0..d {
                centered[i] = row[i] - mean[i];
            }
            for i in 0..d {
                let ci = centered[i];
                for k in i..d {
                    cov[i * d + k] += ci * centered[k];
                }
            }
            scorer.logit_grad(row, &mut grad, &mut ssc);
            accumulate_sens(&mut sens, d, &grad, &mut n_seen).expect("accumulate_sens");
        }
    }
    for i in 0..d {
        for k in i..d {
            let v = cov[i * d + k] / n_fit_rows as f64;
            cov[i * d + k] = v;
            cov[k * d + i] = v;
        }
    }

    // ── Extract: ACE core, PCA control (S = I), random control ─────────
    let mut scratch = CoreScratch::new(d);
    let mut energies = vec![0.0; d];
    let mut q_probe = vec![0.0; d * d];
    extract_core_from_matrices(
        &cov,
        &sens,
        d,
        RankRule::Fixed(1),
        &mut q_probe,
        &mut energies,
        &mut scratch,
    )
    .unwrap_or_else(|e| die(&format!("extract (spectrum): {e}")));
    let r_energy = core_rank_energy(&energies, ENERGY);
    let r_max = r_energy.max(*RANKS.last().unwrap()).min(d);
    let mut q_core = vec![0.0; r_max * d];
    let core = extract_core_from_matrices(
        &cov,
        &sens,
        d,
        RankRule::Fixed(r_max),
        &mut q_core,
        &mut energies,
        &mut scratch,
    )
    .unwrap_or_else(|e| die(&format!("extract core: {e}")));
    let mut ident = vec![0.0; d * d];
    (0..d).for_each(|i| ident[i * d + i] = 1.0);
    let mut q_pca = vec![0.0; r_max * d];
    let mut pca_energies = vec![0.0; d];
    extract_core_from_matrices(
        &cov,
        &ident,
        d,
        RankRule::Fixed(r_max),
        &mut q_pca,
        &mut pca_energies,
        &mut scratch,
    )
    .unwrap_or_else(|e| die(&format!("extract pca: {e}")));
    let mut rnd32 = vec![0.0f32; r_max * d];
    random_basis_into(&mut rnd32, d, RANDOM_SEED);
    let q_rand: Vec<f64> = rnd32.iter().map(|&x| f64::from(x)).collect();

    // ── Causal reads on the held-out half ──────────────────────────────
    let decision = |rows: &[&[f64]; 2], sc: &mut ScorerScratch| -> bool {
        scorer.logit(rows[1], sc) > scorer.logit(rows[0], sc)
    };
    let base: Vec<bool> = eval
        .iter()
        .map(|s| decision(&[&s.rows[0], &s.rows[1]], &mut ssc))
        .collect();
    let base_true_rate = base.iter().filter(|&&b| b).count() as f64 / base.len().max(1) as f64;

    let mut ranks: Vec<usize> = RANKS.iter().copied().filter(|&r| r <= r_max).collect();
    if !ranks.contains(&r_energy) {
        ranks.push(r_energy);
        ranks.sort_unstable();
    }
    let arms: [(&str, &[f64]); 3] = [("core", &q_core), ("pca", &q_pca), ("random", &q_rand)];
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
    let mut table = Vec::new();
    for &r in &ranks {
        let mut row = serde_json::Map::new();
        row.insert("r".into(), json!(r));
        for (name, q) in arms {
            let p = CoreProjector { mean: &mean, q, r };
            let (keep_agree, remove_flip) = eval_arm(&eval, &base, &p, &scorer, d, threads);
            row.insert(format!("{name}_keep_retention"), json!(keep_agree));
            row.insert(format!("{name}_remove_flip"), json!(remove_flip));
        }
        let get = |k: &str| row[k].as_f64().unwrap();
        let sufficient = get("core_keep_retention") >= 0.95
            && get("core_keep_retention") > get("pca_keep_retention")
            && get("core_keep_retention") > get("random_keep_retention");
        let necessary = get("core_remove_flip") >= 0.25
            && get("core_remove_flip") >= 2.0 * get("random_remove_flip");
        row.insert("SUFFICIENT".into(), json!(sufficient));
        row.insert("NECESSARY".into(), json!(necessary));
        eprintln!("  r {r:4}: {}", Value::Object(row.clone()));
        table.push(Value::Object(row));
    }
    let low_dim_core = table
        .iter()
        .any(|r| r["SUFFICIENT"] == true && r["r"].as_u64().unwrap() <= 32);

    // ── Persist ─────────────────────────────────────────────────────────
    let core_path = out.join(format!("{ckpt_name}_core.f64"));
    let bytes: Vec<u8> = q_core.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&core_path, &bytes).expect("write core");
    let total_e: f64 = energies.iter().map(|e: &f64| e.max(0.0)).sum();
    let report = json!({
        "issue": 1237,
        "readout": "R1 scorer-input marker rows (analytic Jacobian)",
        "checkpoint": ckpt_name,
        "device": agent.device(),
        "d": d,
        "suite": suite,
        "suite_blake3": suite_blake3,
        "records": records.len(),
        "fit_records": fit_end,
        "fit_rows": n_fit_rows,
        "eval_forwards": eval.len(),
        "cov_rows_over_d": n_fit_rows as f64 / d as f64,
        "fidelity_max_abs_logit": fidelity_max,
        "eval_base_true_rate": base_true_rate,
        "r_energy_090": r_energy,
        "r_max": r_max,
        "core_gap_ratio_at_rmax": core.gap_ratio,
        "core_participation_ratio": core.participation_ratio,
        "energy_share_top": RANKS.iter().map(|&r| {
            let s: f64 = energies.iter().take(r).map(|e: &f64| e.max(0.0)).sum();
            json!({"r": r, "share": s / total_e})
        }).collect::<Vec<_>>(),
        "forward_secs": forward_secs,
        "table": table,
        "LOW_DIM_CORE": low_dim_core,
    });
    let report_path = out.join(format!("{ckpt_name}_report.json"));
    std::fs::write(&report_path, serde_json::to_string_pretty(&report).unwrap())
        .expect("write report");
    println!(
        "{}",
        serde_json::to_string(&json!({
            "checkpoint": ckpt_name, "r_energy_090": r_energy, "LOW_DIM_CORE": low_dim_core,
            "fidelity_max_abs_logit": fidelity_max, "report": report_path,
        }))
        .unwrap()
    );
}

/// `(keep-retention, remove-flip)` of one projector over the eval forwards,
/// threaded by contiguous chunks (per-thread scratch; order-free counts).
fn eval_arm(
    eval: &[&Sample],
    base: &[bool],
    p: &CoreProjector<'_>,
    scorer: &Scorer64,
    d: usize,
    threads: usize,
) -> (f64, f64) {
    let chunk = eval.len().div_ceil(threads).max(1);
    let (keep, flip) = std::thread::scope(|s| {
        let handles: Vec<_> = eval
            .chunks(chunk)
            .zip(base.chunks(chunk))
            .map(|(es, bs)| {
                s.spawn(move || {
                    let mut sc = ScorerScratch::new(d);
                    let mut buf = [vec![0.0; d], vec![0.0; d]];
                    let (mut keep, mut flip) = (0usize, 0usize);
                    for (smp, &b) in es.iter().zip(bs) {
                        for (row, out) in smp.rows.iter().zip(buf.iter_mut()) {
                            p.keep(row, out);
                        }
                        let kd = scorer.logit(&buf[1], &mut sc) > scorer.logit(&buf[0], &mut sc);
                        keep += usize::from(kd == b);
                        for (row, out) in smp.rows.iter().zip(buf.iter_mut()) {
                            p.remove(row, out);
                        }
                        let rd = scorer.logit(&buf[1], &mut sc) > scorer.logit(&buf[0], &mut sc);
                        flip += usize::from(rd != b);
                    }
                    (keep, flip)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("eval thread"))
            .fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
    });
    let n = eval.len().max(1) as f64;
    (keep as f64 / n, flip as f64 / n)
}

fn read_core(path: &Path, d: usize) -> Vec<f64> {
    let bytes =
        std::fs::read(path).unwrap_or_else(|e| die(&format!("read {}: {e}", path.display())));
    let (words, _) = bytes.as_chunks::<8>();
    let v: Vec<f64> = words.iter().map(|c| f64::from_le_bytes(*c)).collect();
    if !v.len().is_multiple_of(d) {
        die("core file length is not a multiple of d");
    }
    v
}

fn compare(args: &[String]) {
    let a = PathBuf::from(arg(args, "--a").unwrap_or_else(|| die("--a required")));
    let b = PathBuf::from(arg(args, "--b").unwrap_or_else(|| die("--b required")));
    let d: usize = arg(args, "--d").map_or(1024, |v| v.parse().expect("--d"));
    let r: usize = arg(args, "--r").map_or(32, |v| v.parse().expect("--r"));
    let (qa, qb) = (read_core(&a, d), read_core(&b, d));
    if qa.len() < r * d || qb.len() < r * d {
        die("a core holds fewer than r rows");
    }
    let to32 = |q: &[f64]| q[..r * d].iter().map(|&x| x as f32).collect::<Vec<f32>>();
    let sim = f64::from(basis_similarity(&to32(&qa), &to32(&qb), d));
    let baseline = r as f64 / d as f64;
    let ratio = sim / baseline;
    println!(
        "{}",
        serde_json::to_string(&json!({
            "r": r, "d": d, "basis_similarity": sim, "random_baseline": baseline,
            "ratio": ratio, "AGREEMENT": ratio >= 5.0,
        }))
        .unwrap()
    );
}

/// One R0 sample: an encoder output + its two marker positions.
struct R0Forward {
    hidden: Vec<f32>,
    markers: Vec<usize>,
    qtype: usize,
}

/// The two scorer logits after the head over `hidden` (the decision is
/// `l[1] > l[0]`).
fn r0_logits(
    agent: &RiirAgent,
    scorer: &Scorer64,
    f: &R0Forward,
    hidden: &[f32],
    sc: &mut ScorerScratch,
) -> [f64; 2] {
    let rows = agent
        .head_rows_from_hidden(hidden, f.qtype, &f.markers)
        .unwrap_or_else(|e| die(&format!("head: {e}")));
    let d = scorer.dim();
    let mut out = [0.0; 2];
    let mut row = vec![0.0; d];
    for (m, o) in out.iter_mut().enumerate() {
        for (r, &x) in row.iter_mut().zip(&rows[m * d..(m + 1) * d]) {
            *r = f64::from(x);
        }
        *o = scorer.logit(&row, sc);
    }
    out
}

/// Every `stride`-th index of `0..n` (at most `cap`), spread over the range.
fn spread(n: usize, cap: usize) -> Vec<usize> {
    let stride = n.div_ceil(cap.max(1)).max(1);
    (0..n).step_by(stride).take(cap).collect()
}

/// Issue 1237 T-R0: the decision core at the ENCODER OUTPUT (marker
/// positions), through both head layers + the scorer. The relevance moment
/// is estimated by central finite differences along the top-`k` activity
/// (PCA) directions of the fit marker rows — the ACE extract then runs in
/// that k-dim subspace (Cov there is diag(λ)) and maps back. Disclosed
/// approximation: directions outside the top-k activity span carry
/// `1 − captured_trace_share` of the activity and are never probed.
fn census_r0(args: &[String]) {
    let suite = arg(args, "--suite").unwrap_or_else(|| die("--suite required"));
    let ckpt_name = arg(args, "--checkpoint").unwrap_or_else(|| "english".into());
    let n_records: usize = arg(args, "--records").map_or(200, |v| v.parse().expect("--records"));
    let k: usize = arg(args, "--k").map_or(64, |v| v.parse().expect("--k"));
    let n_sens: usize = arg(args, "--sens").map_or(128, |v| v.parse().expect("--sens"));
    let n_eval: usize = arg(args, "--eval").map_or(256, |v| v.parse().expect("--eval"));
    let eps_rel: f64 = arg(args, "--eps").map_or(1e-2, |v| v.parse().expect("--eps"));
    let out = PathBuf::from(arg(args, "--out").unwrap_or_else(|| die("--out required")));
    std::fs::create_dir_all(&out).expect("create --out");
    let (suite_blake3, records) = load_suite(&suite, n_records);
    let fit_end = records.len() / 2;
    let agent = RiirAgent::load(&weights_root(), checkpoint(&ckpt_name))
        .unwrap_or_else(|e| die(&format!("load {ckpt_name}: {e}")));
    let scorer = Scorer64::new(&agent.scorer_tensors());
    let d = scorer.dim();
    let mut ssc = ScorerScratch::new(d);
    let t0 = std::time::Instant::now();

    // ── Encoder pass: every forward's pre-head hidden ──────────────────
    let forwards = suite_forwards(&records);
    let mut fit_idx = Vec::new();
    let mut eval_idx = Vec::new();
    let mut r0: Vec<R0Forward> = Vec::with_capacity(forwards.len());
    let mut hidden_vs_encode_max = 0.0f64;
    let mut seam_fidelity_max = 0.0f64;
    for (fi, fw) in forwards.iter().enumerate() {
        let (hidden, markers, qtype) = agent
            .encoder_hidden(&fw.payload, &fw.qdef)
            .unwrap_or_else(|e| die(&format!("encoder: {e}")));
        if markers.len() != 2 {
            die("noul forward must carry exactly two markers");
        }
        let f = R0Forward {
            hidden,
            markers,
            qtype,
        };
        if fi < FIDELITY_FORWARDS {
            // Seam fidelity: head-from-hidden == the lane's own logits; and
            // the EncodedQuestion.hidden semantics read (pre- vs post-head).
            let fwd = agent
                .forward_question(&fw.payload, &fw.qdef)
                .unwrap_or_else(|e| die(&format!("forward: {e}")));
            let l = r0_logits(&agent, &scorer, &f, &f.hidden, &mut ssc);
            for (lm, &fm) in l.iter().zip(&fwd.logits) {
                seam_fidelity_max = seam_fidelity_max.max((lm - f64::from(fm)).abs());
            }
            let enc = agent
                .encode_question(&fw.payload, &fw.qdef)
                .unwrap_or_else(|e| die(&format!("encode: {e}")));
            for (a, b) in enc.hidden.iter().zip(&f.hidden) {
                hidden_vs_encode_max = hidden_vs_encode_max.max(f64::from((a - b).abs()));
            }
        }
        if fw.record < fit_end {
            fit_idx.push(fi);
        } else {
            eval_idx.push(fi);
        }
        r0.push(f);
        if fi % 500 == 0 {
            eprintln!(
                "  encoder {fi}/{}: {:.1}s",
                forwards.len(),
                t0.elapsed().as_secs_f64()
            );
        }
    }
    if seam_fidelity_max > FIDELITY_TOL {
        die(&format!(
            "seam fidelity FAILED: max |Δlogit| {seam_fidelity_max:.3e}"
        ));
    }
    let row_of = |f: &R0Forward, m: usize| -> Vec<f64> {
        let p = f.markers[m];
        f.hidden[p * d..(p + 1) * d]
            .iter()
            .map(|&x| f64::from(x))
            .collect()
    };

    // ── Activity at R0: mean + Cov over the fit marker rows ────────────
    let n_rows = 2 * fit_idx.len();
    let mut mean = vec![0.0; d];
    for &i in &fit_idx {
        for m in 0..2 {
            for (a, x) in mean.iter_mut().zip(row_of(&r0[i], m)) {
                *a += x;
            }
        }
    }
    mean.iter_mut().for_each(|a| *a /= n_rows as f64);
    let mut cov = vec![0.0; d * d];
    let mut c = vec![0.0; d];
    for &i in &fit_idx {
        for m in 0..2 {
            for (ci, (x, mu)) in c.iter_mut().zip(row_of(&r0[i], m).iter().zip(&mean)) {
                *ci = x - mu;
            }
            for a in 0..d {
                for b in a..d {
                    cov[a * d + b] += c[a] * c[b];
                }
            }
        }
    }
    for a in 0..d {
        for b in a..d {
            let v = cov[a * d + b] / n_rows as f64;
            cov[a * d + b] = v;
            cov[b * d + a] = v;
        }
    }
    let trace: f64 = (0..d).map(|a| cov[a * d + a]).sum();

    // ── PCA basis P (top-k rows) via relevance_core's S = I control ────
    let mut scratch = CoreScratch::new(d);
    let mut ident = vec![0.0; d * d];
    (0..d).for_each(|a| ident[a * d + a] = 1.0);
    let mut p_rows = vec![0.0; k * d];
    let mut lam = vec![0.0; d];
    extract_core_from_matrices(
        &cov,
        &ident,
        d,
        RankRule::Fixed(k),
        &mut p_rows,
        &mut lam,
        &mut scratch,
    )
    .unwrap_or_else(|e| die(&format!("pca: {e}")));
    let captured: f64 = lam.iter().take(k).map(|e: &f64| e.max(0.0)).sum::<f64>() / trace;
    eprintln!(
        "  pca top-{k} captures {captured:.4} of tr(Cov); {:.1}s",
        t0.elapsed().as_secs_f64()
    );

    // ── Relevance in P coords: central FD along each p_j ───────────────
    let sens_idx: Vec<usize> = spread(fit_idx.len(), n_sens)
        .into_iter()
        .map(|j| fit_idx[j])
        .collect();
    let mut sens_k = vec![0.0; k * k];
    let mut n_seen = 0u64;
    let mut g = vec![0.0; k];
    let mut work = Vec::new();
    for (si, &i) in sens_idx.iter().enumerate() {
        let f = &r0[i];
        for m in 0..2 {
            let pos = f.markers[m];
            for j in 0..k {
                let step = eps_rel * lam[j].max(0.0).sqrt();
                let pj = &p_rows[j * d..(j + 1) * d];
                let mut probe = |sign: f64| -> f64 {
                    work.clear();
                    work.extend_from_slice(&f.hidden);
                    for (w, &pv) in work[pos * d..(pos + 1) * d].iter_mut().zip(pj) {
                        *w += (sign * step * pv) as f32;
                    }
                    r0_logits(&agent, &scorer, f, &work, &mut ssc)[m]
                };
                g[j] = (probe(1.0) - probe(-1.0)) / (2.0 * step);
            }
            accumulate_sens(&mut sens_k, k, &g, &mut n_seen).expect("accumulate_sens");
        }
        if si % 16 == 0 {
            eprintln!(
                "  sens {si}/{}: {:.1}s",
                sens_idx.len(),
                t0.elapsed().as_secs_f64()
            );
        }
    }

    // ── ACE in the k-subspace (Cov_k = diag λ), map back through P ─────
    let mut cov_k = vec![0.0; k * k];
    (0..k).for_each(|j| cov_k[j * k + j] = lam[j]);
    let mut kscratch = CoreScratch::new(k);
    let mut energies = vec![0.0; k];
    let mut qk = vec![0.0; k * k];
    let core = extract_core_from_matrices(
        &cov_k,
        &sens_k,
        k,
        RankRule::Fixed(k),
        &mut qk,
        &mut energies,
        &mut kscratch,
    )
    .unwrap_or_else(|e| die(&format!("extract core: {e}")));
    let r_energy = core_rank_energy(&energies, ENERGY);
    let mut q_core = vec![0.0; k * d];
    for r in 0..k {
        for j in 0..k {
            let w = qk[r * k + j];
            for a in 0..d {
                q_core[r * d + a] += w * p_rows[j * d + a];
            }
        }
    }
    let mut rnd32 = vec![0.0f32; k * d];
    random_basis_into(&mut rnd32, d, RANDOM_SEED);
    let q_rand: Vec<f64> = rnd32.iter().map(|&x| f64::from(x)).collect();

    // ── Causal reads at R0 on the held-out eval subset ─────────────────
    let eval_sel: Vec<usize> = spread(eval_idx.len(), n_eval)
        .into_iter()
        .map(|j| eval_idx[j])
        .collect();
    let decide = |l: [f64; 2]| l[1] > l[0];
    let base: Vec<bool> = eval_sel
        .iter()
        .map(|&i| decide(r0_logits(&agent, &scorer, &r0[i], &r0[i].hidden, &mut ssc)))
        .collect();
    let base_true_rate = base.iter().filter(|&&b| b).count() as f64 / base.len().max(1) as f64;
    let mut ranks: Vec<usize> = [1usize, 2, 4, 8, 16]
        .into_iter()
        .filter(|&r| r <= k)
        .collect();
    if r_energy <= k && !ranks.contains(&r_energy) {
        ranks.push(r_energy);
        ranks.sort_unstable();
    }
    let arms: [(&str, &[f64]); 3] = [("core", &q_core), ("pca", &p_rows), ("random", &q_rand)];
    let mut buf = vec![0.0; d];
    let mut table = Vec::new();
    for &r in &ranks {
        let mut row = serde_json::Map::new();
        row.insert("r".into(), json!(r));
        for (name, q) in arms {
            let proj = CoreProjector { mean: &mean, q, r };
            let (mut keep, mut flip) = (0usize, 0usize);
            for (&i, &b) in eval_sel.iter().zip(&base) {
                let f = &r0[i];
                for (mode, counter) in [(0, &mut keep), (1, &mut flip)] {
                    work.clear();
                    work.extend_from_slice(&f.hidden);
                    for m in 0..2 {
                        let h = row_of(f, m);
                        match mode {
                            0 => proj.keep(&h, &mut buf),
                            _ => proj.remove(&h, &mut buf),
                        }
                        let pos = f.markers[m];
                        for (w, &v) in work[pos * d..(pos + 1) * d].iter_mut().zip(&buf) {
                            *w = v as f32;
                        }
                    }
                    let dec = decide(r0_logits(&agent, &scorer, f, &work, &mut ssc));
                    *counter += usize::from(if mode == 0 { dec == b } else { dec != b });
                }
            }
            let n = eval_sel.len().max(1) as f64;
            row.insert(format!("{name}_keep_retention"), json!(keep as f64 / n));
            row.insert(format!("{name}_remove_flip"), json!(flip as f64 / n));
        }
        let get = |key: &str| row[key].as_f64().unwrap();
        let sufficient = get("core_keep_retention") >= 0.95
            && get("core_keep_retention") > get("pca_keep_retention")
            && get("core_keep_retention") > get("random_keep_retention");
        let necessary = get("core_remove_flip") >= 0.25
            && get("core_remove_flip") >= 2.0 * get("random_remove_flip");
        row.insert("SUFFICIENT".into(), json!(sufficient));
        row.insert("NECESSARY".into(), json!(necessary));
        eprintln!("  r {r:3}: {}", Value::Object(row.clone()));
        table.push(Value::Object(row));
    }
    let low_dim_core = table
        .iter()
        .any(|r| r["SUFFICIENT"] == true && r["r"].as_u64().unwrap() <= 32);
    let total_e: f64 = energies.iter().map(|e: &f64| e.max(0.0)).sum();
    let energy_share: Vec<Value> = [1usize, 2, 4, 8, 16, 32]
        .into_iter()
        .filter(|&r| r <= k)
        .map(|r| {
            let s: f64 = energies.iter().take(r).map(|e: &f64| e.max(0.0)).sum();
            json!({"r": r, "share": s / total_e})
        })
        .collect();
    let report = json!({
        "issue": 1237,
        "readout": "R0 encoder-output marker rows through head layers + scorer (FD in the top-k activity subspace)",
        "checkpoint": ckpt_name, "device": agent.device(), "d": d, "k": k,
        "suite": suite, "suite_blake3": suite_blake3, "records": records.len(), "fit_records": fit_end,
        "fit_rows": n_rows, "sens_forwards": sens_idx.len(), "eval_forwards": eval_sel.len(),
        "eps_rel": eps_rel, "pca_captured_trace_share": captured,
        "seam_fidelity_max_abs_logit": seam_fidelity_max,
        "encode_question_hidden_vs_encoder_only_max_abs": hidden_vs_encode_max,
        "eval_base_true_rate": base_true_rate, "r_energy_090": r_energy,
        "core_participation_ratio": core.participation_ratio,
        "energy_share_top": energy_share,
        "secs": t0.elapsed().as_secs_f64(),
        "table": table, "LOW_DIM_CORE": low_dim_core,
    });
    let path = out.join(format!("{ckpt_name}_r0_report.json"));
    std::fs::write(&path, serde_json::to_string_pretty(&report).unwrap()).expect("write report");
    let bytes: Vec<u8> = q_core.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(out.join(format!("{ckpt_name}_r0_core.f64")), bytes).expect("write core");
    println!("{}", serde_json::to_string(&json!({
        "checkpoint": ckpt_name, "r_energy_090": r_energy, "LOW_DIM_CORE": low_dim_core,
        "seam_fidelity": seam_fidelity_max, "hidden_vs_encode": hidden_vs_encode_max, "report": path,
    })).unwrap());
}
