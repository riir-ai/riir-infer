//! Issue 1177 Item 1 — the CUDA-side walk-parity harness: the SAME fixed
//! greedy-argmax walk as the Metal bench's `[walk] ids:` artifact, run by
//! `qwen38_dense_cudarc` (the bit-parity oracle) on a CUDA host.
//!
//! The Metal side is `qwen38_wgpu_bench --walk-tokens 48` (its default
//! prompt, its walk loop verbatim). This bin mirrors that shape EXACTLY:
//! same default prompt string, same tokenizer, same sequential prefill
//! (per-token `forward_token` over the prompt — the cudarc forward has no
//! chunked ingest; the Metal chunked-route ≡ multi-token equivalence is
//! Issue 1150's own route G1, bench 938), same walk loop (the s==0
//! re-feed of the last prompt token at pos = ids.len(), the bench's
//! convention), same `[walk] ids: [...]` output line so the two logs
//! diff by eye and hash identically.
//!
//! The pre-registered band lives in the issue (exact-pin primary; a flip
//! is adjudicated against the 1e-2 near-tie band via the capture-logits
//! re-run — NOT this primary path). FNV-1a over the ids and the model
//! sha256 pin are computed by the run wrapper on both sides, never in-bin
//! (the Metal bench prints ids only; both logs stay format-identical).
//!
//! Run (4090, PowerShell wrapper prints Get-FileHash first):
//!   cargo build --release --bin qwen38_cudarc_walk --features ternary_gemv_cuda_raw
//!   QWEN38_GGUF=F:/models/qwen38-27b-dbirks-Q4_K_M.gguf target/release/qwen38_cudarc_walk.exe --ctx 2048 --walk-tokens 48

use riir_infer_core::gguf_loader::GgufFile;
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_gpu::qwen38_dense_cudarc::Qwen38DenseForward;

/// The Metal bench's DEFAULT_PROMPT verbatim (the prose family walk).
const DEFAULT_PROMPT: &str = "The harbor town woke slowly under a grey sky,";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut gguf = std::env::var("QWEN38_GGUF")
        .unwrap_or_else(|_| r"F:/models/qwen38-27b-dbirks-Q4_K_M.gguf".to_owned());
    let mut ctx = 2_048usize;
    let mut walk_tokens = 48usize;
    let mut prompt = DEFAULT_PROMPT.to_string();
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--gguf" => {
                i += 1;
                gguf = args[i].clone();
            }
            "--ctx" => {
                i += 1;
                ctx = args[i].parse().expect("ctx n");
            }
            "--walk-tokens" => {
                i += 1;
                walk_tokens = args[i].parse().expect("walk n");
            }
            "--prompt" => {
                i += 1;
                prompt = args[i].clone();
            }
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(2);
            }
        }
        i += 1;
    }

    if !std::path::Path::new(&gguf).exists() {
        eprintln!("[i1177-parity] SKIP (loud): no model at {gguf} — set QWEN38_GGUF");
        std::process::exit(3);
    }

    println!("[i1177-parity] loading {gguf} (ctx {ctx}) — ~15 GiB upload, idle GPU assumed");
    let t0 = std::time::Instant::now();
    let mut fwd = Qwen38DenseForward::new(std::path::Path::new(&gguf), ctx)
        .unwrap_or_else(|e| panic!("[i1177-parity] load failed: {e}"));
    println!(
        "[i1177-parity] loaded in {:.1}s (vocab {}, n_embd {}, layers {})",
        t0.elapsed().as_secs_f32(),
        fwd.cfg.vocab_size,
        fwd.cfg.n_embd,
        fwd.cfg.layer_types.len()
    );

    let gg = GgufFile::open(std::path::Path::new(&gguf))
        .unwrap_or_else(|e| panic!("[i1177-parity] reopen: {e}"));
    let tok =
        BpeTokenizer::from_gguf(&gg).unwrap_or_else(|e| panic!("[i1177-parity] tokenizer: {e}"));
    let ids: Vec<u32> = tok.encode(&prompt).into_iter().map(|t| t as u32).collect();
    println!(
        "[bench] prompt: {} chars -> {} tokens",
        prompt.len(),
        ids.len()
    );

    // Sequential prefill — positions 0..len-1, the cudarc forward's only
    // prompt route (see the module-level note on route ≡ multi-token).
    for (i, &t) in ids.iter().enumerate() {
        fwd.forward_token(t, i)
            .unwrap_or_else(|e| panic!("prefill {i}: {e}"));
    }

    // The walk — the Metal bench's loop verbatim (s==0 re-feeds the last
    // prompt token at pos = ids.len(); then each argmax continues).
    let mut next = 0u32;
    let mut out_ids: Vec<u32> = vec![];
    for s in 0..walk_tokens {
        let pos = ids.len() + s;
        next = fwd
            .forward_token(if s == 0 { *ids.last().unwrap() } else { next }, pos)
            .unwrap_or_else(|e| panic!("walk: {e}"));
        out_ids.push(next);
    }
    println!("[walk] ids: {:?}", out_ids);
    let text: Vec<String> = out_ids
        .iter()
        .map(|&id| tok.decode(&[id as usize]))
        .collect();
    println!("[walk] text: {}", text.join(""));
}
