//! Issue 1240 T1 — the ANE encode-load arm of the ANE↔decode contention
//! cell: one process that loads the `laya-riir-ane` whole-graph encoder
//! (constructor-selected, `RiirAgent::load_ane`) and drives a sustained
//! batch-encode workload over a REAL corpus slice, so a concurrent Metal
//! decode cell measures the pair's DRAM/dispatch interference.
//!
//! The workload: massive_intent_en utterances (the reflex t20k dataset
//! pages) as the state text, a synthetic choice question (k ∈ 2..=5
//! options) per row — the forward shape is exactly the G5-ANE parity
//! seam's (`forward_question`), exercised at both manifest buckets
//! (L64 / L128; every 3rd row carries a context suffix that pushes it
//! toward the long bucket). Decision OUTPUTS are irrelevant here — this
//! is a load generator; the lane's loud refusals (out-of-bucket,
//! head-budget) are counted, never swallowed.
//!
//! Output protocol (the orchestrator gates on these lines):
//!   `ANE_LOAD_DEVICE <cpu|metal|ane>`  — the constructor's own label
//!   `ANE_LOAD_READY`                   — after load + warmup forwards
//!   `ANE_LOAD_TICK secs=.. forwards=.. tok=.. fps=.. tps=.. L64=.. L128=.. err=..`
//!   `ANE_LOAD_TOTAL secs=.. forwards=.. tok=.. fps=.. tps=.. buckets=L64:..,L128:.. err=.. ref=..`
//!
//! Run (the cell's own posture — NEVER quote numbers from a BUSY box):
//!   LAYA_ANE_ARTIFACTS_DIR=~/riir-bench/i1240_ane_artifacts \
//!   cargo run --release -p riir-infer-laya --features laya-riir-ane \
//!     --example i1240_ane_encode_load -- \
//!     --checkpoint english --corpus ../riir-reflex/.raw/datasets_t20k/massive_intent_en \
//!     --secs 60
//!
//! Numbers from this example are VALID only beside their box-state
//! provenance (bench_preflight + rig exclusivity); alone it proves
//! reach (the device label line) and sustains load.

#![cfg(all(target_os = "macos", feature = "laya-riir-ane"))]

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde_json::Value as Json;
use serde_json::{Value, json};

use riir_infer_laya::laya::config::Checkpoint;
use riir_infer_laya::laya::riir::RiirAgent;
use riir_infer_laya::laya::weights::weights_root;

struct Args {
    checkpoint: String,
    artifacts: Option<PathBuf>,
    corpus: PathBuf,
    secs: u64,
    warmup: usize,
    tick_secs: u64,
    max_pages: usize,
}

fn parse_args() -> Args {
    let mut a = Args {
        checkpoint: "english".into(),
        artifacts: None,
        corpus: PathBuf::from("../riir-reflex/.raw/datasets_t20k/massive_intent_en"),
        secs: 60,
        warmup: 8,
        tick_secs: 10,
        max_pages: 30,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let val = |i: &mut usize, name: &str| -> String {
            *i += 1;
            args.get(*i)
                .unwrap_or_else(|| panic!("{name} needs a value"))
                .clone()
        };
        match args[i].as_str() {
            "--checkpoint" => a.checkpoint = val(&mut i, "--checkpoint"),
            "--artifacts" => a.artifacts = Some(val(&mut i, "--artifacts").into()),
            "--corpus" => a.corpus = val(&mut i, "--corpus").into(),
            "--secs" => a.secs = val(&mut i, "--secs").parse().expect("--secs number"),
            "--warmup" => a.warmup = val(&mut i, "--warmup").parse().expect("--warmup number"),
            "--tick-secs" => a.tick_secs = val(&mut i, "--tick-secs").parse().expect("number"),
            "--max-pages" => a.max_pages = val(&mut i, "--max-pages").parse().expect("number"),
            other => panic!("unknown flag {other}"),
        }
        i += 1;
    }
    a
}

fn checkpoint(name: &str) -> Checkpoint {
    Checkpoint::ALL
        .into_iter()
        .find(|c| c.fixture_name() == name || c.subfolder() == name)
        .unwrap_or_else(|| {
            panic!(
                "unknown checkpoint {name} (expected one of: {})",
                Checkpoint::ALL
                    .iter()
                    .map(|c| c.fixture_name())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
}

/// The artifact root: `--artifacts`, else `LAYA_ANE_ARTIFACTS_DIR`, else
/// the reflex checkout's committed (artifact-less) tree — which the
/// digest verify at load will refuse loud, naming the remedy.
fn ane_roots(args: &Args) -> (PathBuf, PathBuf) {
    if let Some(p) = &args.artifacts {
        return (p.clone(), p.join("manifest.json"));
    }
    if let Some(p) = std::env::var_os("LAYA_ANE_ARTIFACTS_DIR") {
        let root = PathBuf::from(p);
        return (root.clone(), root.join("manifest.json"));
    }
    let root = PathBuf::from("../riir-reflex/assets/ane");
    (root.clone(), root.join("manifest.json"))
}

/// One corpus page (the datasets-server JSON form): `rows[].row.text` is
/// the utterance, `rows[].row.label_text` the intent label.
fn load_page_texts(page: &Path) -> Vec<(String, String)> {
    let data: Json = serde_json::from_str(
        &std::fs::read_to_string(page).unwrap_or_else(|e| panic!("read {}: {e}", page.display())),
    )
    .expect("dataset page is JSON");
    let mut out = Vec::new();
    for row in data["rows"].as_array().expect("rows array") {
        let text = row["row"]["text"].as_str().unwrap_or_default().to_string();
        let label = row["row"]["label_text"]
            .as_str()
            .unwrap_or_default()
            .to_string();
        if !text.is_empty() {
            out.push((text, label));
        }
    }
    out
}

/// One encode row in the agent wire form. Every 3rd row (deterministic)
/// carries a two-sentence context suffix so the slice exercises BOTH
/// manifest buckets instead of sitting entirely in L64.
fn build_row(idx: usize, text: &str, label: &str) -> (String, Value) {
    // k varies 2..=5 by index; option keys derived from the intent label so
    // the corpus variance reaches the tokenizer, never a constant shape.
    // A long base text caps k — the k=5 block plus a 50-char utterance
    // already crosses the 128-token bucket ceiling (measured: 132).
    let k = if text.chars().count() >= 20 {
        2 + (idx % 2)
    } else {
        2 + (idx % 4)
    };
    // Context suffix only on LOW-K short rows — the k=5 question block is
    // already ~120 tokens, so a suffix lands any longer row past the 128
    // bucket ceiling (the lane refuses; the cell wants load, not refusals).
    let state = if idx % 3 == 2 && k <= 2 && text.chars().count() < 25 {
        format!(
            "{text} — The user has asked several related things this week and the \
             assistant keeps the running context beside each new request."
        )
    } else {
        text.to_string()
    };
    let mut crit = serde_json::Map::new();
    for j in 0..k {
        let key: String = if j == 0 {
            label.to_string()
        } else {
            format!("{label}_alt{j}")
        };
        crit.insert(
            key,
            json!(if j == 0 {
                "this intent matches the request as stated".to_string()
            } else {
                "a related but distinct intent the request could also mean".to_string()
            }),
        );
    }
    let q = json!({
        "type": "choice",
        "instructions": "Pick the intent this user utterance expresses.",
        "criteria": Value::Object(crit),
    });
    (state, q)
}

fn main() -> Result<(), String> {
    let args = parse_args();
    let ckpt = checkpoint(&args.checkpoint);

    let weights = weights_root();
    let (ane_root, manifest_path) = ane_roots(&args);
    if !manifest_path.exists() {
        return Err(format!(
            "ANE manifest missing at {} — convert first (reflex scripts/ane_convert.py) \
             or point --artifacts/LAYA_ANE_ARTIFACTS_DIR at an artifact root",
            manifest_path.display()
        ));
    }

    // The corpus slice: pages in sorted order, capped by --max-pages.
    let mut pages: Vec<PathBuf> = std::fs::read_dir(&args.corpus)
        .map_err(|e| format!("corpus dir {}: {e}", args.corpus.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("train-") && n.ends_with(".json"))
                .unwrap_or(false)
        })
        .collect();
    pages.sort();
    if pages.len() > args.max_pages {
        pages.truncate(args.max_pages);
    }
    if pages.is_empty() {
        return Err(format!(
            "no train-*.json pages under {}",
            args.corpus.display()
        ));
    }
    let mut texts: Vec<(String, String)> = Vec::new();
    for p in &pages {
        texts.extend(load_page_texts(p));
    }
    if texts.is_empty() {
        return Err("corpus slice empty".into());
    }
    eprintln!(
        "[i1240-load] corpus: {} rows from {} page(s)",
        texts.len(),
        pages.len()
    );

    let agent = RiirAgent::load_ane(&weights, ckpt, &ane_root, &manifest_path)
        .map_err(|e| format!("load_ane: {e}"))?;
    println!("ANE_LOAD_DEVICE {}", agent.device());
    if agent.device() != "ane" {
        return Err(format!(
            "device label is {}, not ane — the load arm would prove nothing",
            agent.device()
        ));
    }
    if let Some(max) = agent.ane_bucket_max() {
        eprintln!("[i1240-load] ane bucket max = {max}");
    }

    // Warmup: load caches + Core ML compile land BEFORE the READY line, so
    // the orchestrator's decode arm never times the load phase.
    for i in 0..args.warmup {
        let (state, q) = build_row(i, &texts[i % texts.len()].0, &texts[i % texts.len()].1);
        agent
            .forward_question(&json!(state), &q)
            .map_err(|e| format!("warmup[{i}]: {e}"))?;
    }
    println!("ANE_LOAD_READY");

    let deadline = Instant::now() + Duration::from_secs(args.secs);
    let tick_every = Duration::from_secs(args.tick_secs);
    let t0 = Instant::now();
    let mut forwards = 0u64;
    let mut toks = 0u64;
    let mut errs = 0u64;
    let mut l64 = 0u64;
    let mut l128 = 0u64;
    let mut out_of_bucket = 0u64;
    let mut next_tick = t0 + tick_every;
    let mut idx = 0usize;

    while Instant::now() < deadline {
        let (text, label) = &texts[idx % texts.len()];
        let (state, q) = build_row(idx, text, label);
        match agent.forward_question(&json!(state), &q) {
            Ok(f) => {
                forwards += 1;
                toks += f.seq_len as u64;
                if f.seq_len <= 64 {
                    l64 += 1;
                } else {
                    l128 += 1;
                }
            }
            Err(e) => {
                errs += 1;
                // Out-of-bucket rows are a COUNTED disclosure (the G5 gate's
                // own floor law), not silent; anything else is loud once.
                let msg = format!("{e}");
                if msg.contains("bucket") {
                    out_of_bucket += 1;
                    if out_of_bucket == 1 {
                        eprintln!(
                            "[i1240-load] first out-of-bucket row (idx={idx}, chars={}, k={}): {msg}",
                            text.chars().count(),
                            q["criteria"].as_object().map(|m| m.len()).unwrap_or(0),
                        );
                    }
                } else {
                    return Err(format!("forward[{idx}]: {e}"));
                }
            }
        }
        idx += 1;
        if Instant::now() >= next_tick {
            let secs = t0.elapsed().as_secs_f64();
            println!(
                "ANE_LOAD_TICK secs={:.1} forwards={forwards} tok={toks} \
                 fps={:.1} tps={:.1} L64={l64} L128={l128} err={errs} oob={out_of_bucket}",
                secs,
                forwards as f64 / secs,
                toks as f64 / secs,
            );
            next_tick += tick_every;
        }
    }

    let secs = t0.elapsed().as_secs_f64().max(1e-9);
    println!(
        "ANE_LOAD_TOTAL secs={:.1} forwards={forwards} tok={toks} fps={:.1} tps={:.1} \
         buckets=L64:{l64},L128:{l128} err={errs} ref={out_of_bucket}",
        secs,
        forwards as f64 / secs,
        toks as f64 / secs,
    );
    Ok(())
}
