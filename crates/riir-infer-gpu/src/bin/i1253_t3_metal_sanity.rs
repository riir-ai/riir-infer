//! `i1253_t3_metal_sanity` — Issue 1253 T3 cheap half: the first forward on
//! the retargeted league checkpoint, on the Metal serving posture.
//!
//! T2 (Bench 1271) proved the 2.5 twin LOADS; the swap is only "verified"
//! when a forward runs. This bin runs the Issue-594-class first-token
//! sanity — "The capital of France is" must rank a Paris-class token at the
//! top of the next-token distribution — plus one greedy decode walk, on the
//! UNTOUCHED `production` posture (every promoted rung default-on, no knob
//! calls, the exact serving path the league drives).
//!
//! Outputs (the bench note's inputs, all CORRECTNESS numbers — timings are
//! context only, never a perf claim):
//! - the top-8 next-token table after prefill (rank · id · surface · logit),
//! - the rank of ` Paris` (bar: rank ≤ 5, first-index ties per Issue 940),
//! - a finiteness guard on every logits read (NaN/Inf fails loud, exit 1),
//! - the walk: greedy tokens with their ids, the decoded surface, the
//!   first decode-lane logits' FNV (the determinism receipt — the same
//!   binary + file re-runs byte-identically), EOS stop disclosed.
//!
//! Usage:
//! ```text
//! cargo build --release -p riir-infer-gpu --features production --bin i1253_t3_metal_sanity
//! ./target/release/i1253_t3_metal_sanity \
//!   [--gguf ../riir-train/data/Ternary-Bonsai-2.5-27B-PQ2_0.gguf] [--gen 32]
//! ```

use std::path::Path;
use std::time::Instant;

use riir_infer_core::gguf_loader::{GgufFile, load_qwen_deltanet_ternary_weights_gguf};
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_gpu::{CubeCLContext, TernaryDeltanetGpuForward};

const DEFAULT_MODEL: &str = "../riir-train/data/Ternary-Bonsai-2.5-27B-PQ2_0.gguf";
const PROBE_PROMPT: &str = "The capital of France is";
/// The Issue-594-class target surface: leading space, first BPE id wins.
const PROBE_ANSWER: &str = " Paris";
/// The sanity bar: the answer's rank must be at most this (1 expected on a
/// healthy checkpoint; 5 keeps the gate honest about near-ties without
/// loosening it into a noop).
const RANK_BAR: usize = 5;
/// Small-context KV reservation: the walk never exceeds ~64 positions.
const BLOCK_CLAMP: usize = 1024;

const EOS: usize = 248_046;

fn argmax(v: &[f32]) -> usize {
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            bi = i;
        }
    }
    bi
}

fn logits_fnv(v: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &x in v {
        let b = x.to_bits() as u64;
        h = (h ^ b).wrapping_mul(0x1000_0000_01b3);
    }
    h
}

fn finite(v: &[f32], where_: &str) -> Result<(), String> {
    if v.iter().all(|x| x.is_finite()) {
        return Ok(());
    }
    let bad = v.iter().filter(|x| !x.is_finite()).count();
    Err(format!(
        "FINITENESS FAIL at {where_}: {bad} non-finite logit(s) of {} — the forward is corrupt, refusing the verdict",
        v.len()
    ))
}

struct Args {
    gguf: String,
    walk: usize,
    prompt: Option<String>,
}

fn parse_args() -> Args {
    let mut args = Args {
        gguf: DEFAULT_MODEL.to_string(),
        walk: 32,
        prompt: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--gguf" => {
                args.gguf = it.next().unwrap_or_else(|| args.gguf.clone());
            }
            "--gen" => {
                args.walk = it.next().and_then(|v| v.parse().ok()).unwrap_or(args.walk);
            }
            "--prompt" => {
                args.prompt = it.next();
            }
            other => eprintln!("unknown arg {other} (ignored)"),
        }
    }
    args
}

fn main() {
    if let Err(e) = run() {
        eprintln!("[i1253-t3] REFUSED: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args();
    let model_path = args.gguf.clone();
    let n_gen = args.walk.max(1);
    let probe_prompt = args
        .prompt
        .clone()
        .unwrap_or_else(|| PROBE_PROMPT.to_string());
    let probe_is_named = args.prompt.is_none();

    eprintln!("[i1253-t3] loading {model_path} …");
    let t0 = Instant::now();
    let (mut config, weights) = load_qwen_deltanet_ternary_weights_gguf(Path::new(&model_path))
        .map_err(|e| format!("load: {e}"))?;
    let gguf = GgufFile::open(Path::new(&model_path)).map_err(|e| format!("reopen: {e}"))?;
    let tok = BpeTokenizer::from_gguf(&gguf).map_err(|e| format!("tokenizer: {e}"))?;
    config.block_size = config.block_size.min(BLOCK_CLAMP);
    let ctx = CubeCLContext::new().map_err(|e| format!("GPU init: {e}"))?;
    let mut fwd = TernaryDeltanetGpuForward::new(&ctx, &config, &weights);
    eprintln!(
        "[i1253-t3] forward ready ({:.0}s) — production posture, block_size clamped to {}",
        t0.elapsed().as_secs(),
        config.block_size
    );

    // ── G1: the first-token sanity (the named Paris probe; a --prompt
    // override runs the decode-walk shape instead and skips the rank gate) ──
    let mut prompt_ids = tok.encode(&probe_prompt);
    let bos = config.bos_token;
    if prompt_ids.first() != Some(&bos) {
        prompt_ids.insert(0, bos);
    }
    let answer_id = tok
        .encode(PROBE_ANSWER)
        .first()
        .copied()
        .ok_or_else(|| format!("probe answer {PROBE_ANSWER:?} encodes to zero tokens"))?;
    if !probe_is_named {
        // An overridden prompt carries no pre-registered answer: the rank
        // table still prints, the gate is the walk + finiteness only.
        return run_walk_only(&mut fwd, &weights, &tok, prompt_ids, n_gen);
    }

    fwd.reset_state();
    let tp = Instant::now();
    let logits = fwd.prefill(&prompt_ids);
    let prefill_wall = tp.elapsed();
    finite(&logits, "prefill logits")?;
    let first_fnv = logits_fnv(&logits);
    println!(
        "prefill wall (context only): {} tokens in {:.1}s",
        prompt_ids.len(),
        prefill_wall.as_secs_f32()
    );

    // Rank of the answer token among the full vocab (first-index tie rule:
    // a token ranks AHEAD of an equal-scoring later token).
    let answer_logit = logits[answer_id];
    let rank = logits.iter().filter(|&&x| x > answer_logit).count() + 1;
    let top = argmax(&logits);

    println!(
        "probe: {probe_prompt:?}  ({} prompt tokens, bos={bos})",
        prompt_ids.len()
    );
    println!("prefill logits FNV: {first_fnv:#018x}");
    println!("top-8 next tokens:");
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (r, &id) in order.iter().take(8).enumerate() {
        let surface = tok.decode(&[id]);
        let mark = if id == answer_id {
            "  <- probe answer"
        } else {
            ""
        };
        println!(
            "  {:>2}. id={:<7} logit={:<9.4} {:?}{mark}",
            r + 1,
            id,
            logits[id],
            surface
        );
    }
    println!(
        "rank of {PROBE_ANSWER:?} (id {answer_id}): {rank} (bar ≤ {RANK_BAR}) — {}",
        if rank <= RANK_BAR { "PASS" } else { "FAIL" }
    );
    if rank > RANK_BAR {
        return Err(format!(
            "G1 first-token sanity FAILED: {PROBE_ANSWER:?} ranked {rank} > {RANK_BAR}"
        ));
    }
    run_walk(&mut fwd, &weights, &tok, top, n_gen)?;
    println!(
        "[i1253-t3] VERDICT: G1 PASS (rank {rank} ≤ {RANK_BAR}), walk clean, forward ran on the production Metal posture"
    );
    Ok(())
}

/// The `--prompt` override posture: no pre-registered answer, so the rank
/// gate is skipped and the walk + finiteness are the verdict.
fn run_walk_only(
    fwd: &mut TernaryDeltanetGpuForward,
    weights: &riir_infer_core::deltanet::ternary_weights::QwenDeltaNetTernaryWeights,
    tok: &BpeTokenizer,
    prompt_ids: Vec<usize>,
    n_gen: usize,
) -> Result<(), String> {
    fwd.reset_state();
    let logits = fwd.prefill(&prompt_ids);
    finite(&logits, "prefill logits")?;
    println!(
        "probe: (override) {} prompt tokens; prefill logits FNV: {:#018x}",
        prompt_ids.len(),
        logits_fnv(&logits)
    );
    run_walk(fwd, weights, tok, argmax(&logits), n_gen)
}

/// The decode walk, given the prefill's argmax. Shared by both postures.
fn run_walk(
    fwd: &mut TernaryDeltanetGpuForward,
    weights: &riir_infer_core::deltanet::ternary_weights::QwenDeltaNetTernaryWeights,
    tok: &BpeTokenizer,
    first: usize,
    n_gen: usize,
) -> Result<(), String> {
    let mut out_ids: Vec<usize> = Vec::with_capacity(n_gen + 1);
    let mut next = first;
    out_ids.push(next);
    let mut second_fnv: u64 = 0;
    let td = Instant::now();
    for step in 0..n_gen {
        if next == EOS {
            println!("walk: EOS at step {step} — stopping");
            break;
        }
        fwd.set_input_token(weights, next);
        let l = fwd.forward_token();
        finite(&l, &format!("decode step {step}"))?;
        if step == 0 {
            second_fnv = logits_fnv(&l);
        }
        next = argmax(&l);
        out_ids.push(next);
    }
    let decode_wall = td.elapsed();
    let text = tok.decode(&out_ids);
    println!(
        "walk ({} tokens): {:?}",
        out_ids.len(),
        &out_ids[..out_ids.len().min(12)]
    );
    println!("  {text:?}");
    println!("first decode-lane logits FNV: {second_fnv:#018x} (determinism receipt)");
    println!(
        "walls (context only, NON-CLAIM — box CPU contended): {} decode steps {:.1}s",
        out_ids.len().saturating_sub(1),
        decode_wall.as_secs_f32(),
    );
    println!("[i1253-t3] walk clean on the production Metal posture");
    Ok(())
}
