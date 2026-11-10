// The bin's own posture is its required-features row
// (flashmemory_budget_gate — the CPU reference lane's core features
// forwarded + the Metal arm); the row protects the reader (the
// green-zero law), this cfg mirrors it for direct rustc sanity.
#![cfg(feature = "flashmemory_budget_gate")]
//! Issue 1117 T3/T4 — the FlashMemory budget (top-ρ) decode gate.
//!
//! The end-to-end quality + perf read the per-layer fidelity numbers
//! (riir-train Bench 671) cannot answer: does per-layer block selection
//! survive every attention layer compounding through the real hybrid stack?
//!
//! # Rig
//!
//! Teacher-forced decode over a long-context token slice: both arms step the
//! REAL forward one token at a time over the same ground-truth token stream
//! (no sampling — the comparison isolates the attention-read pattern). The
//! DENSE arm (`forward_qwen_deltanet_ternary`, bounded cache) runs once; each
//! ρ arm (`forward_qwen_deltanet_ternary_flashmemory`, block 64, refresh 64,
//! incremental centroids) runs against its own fresh cache + state.
//!
//! # Arm persistence (2026-10-30 — the killed-run lesson)
//!
//! The first 128K launch (2026-10-27) died mid-dense to an unattributed kill
//! at 41% with the report still end-of-run-only: ~7.6 h of compute, zero
//! durable output. Each arm now persists to `<out>/<name>.bin` the moment it
//! completes, and EVERY arm (dense since f98e921; the ρ arms since the
//! 2026-11-04 post-mortem — the m3 v2 run lost ~27 h of ρ 0.15 at 94% with
//! no bin at all) also rewrites its bin every `--bucket` steps as a
//! complete=0 prefix, temp+rename atomic, so a mid-arm kill keeps the
//! counted prefix readable post-hoc. On startup a COMPLETE bin whose digest
//! matches the current inputs is LOADED instead of recomputed; since Issue
//! 1191 T3 a complete=0 prefix is ALSO loadable under `--resume` (the arm
//! continues from its cursor — the state rebuild is the deterministic
//! replay, no serialization needed). The digest pins ctx/steps/block/refresh/rho/compare-from +
//! the corpus bytes + the first 64 MiB of the gguf — a changed corpus or
//! model at the same paths refuses the stale bin loud (`--fresh` recomputes
//! anyway). Forward code paths are untouched by this layer.
//!
//! # Metrics (per position, bucketed)
//!
//! - `ΔNLL = NLL_sparse − NLL_dense` per position (mean / p95 / max, with the
//!   bucket's tail support reported beside the p95 — a percentile over a
//!   handful of rows is the max wearing a name).
//! - top-1 agreement: argmax_sparse == argmax_dense per position.
//! - retention: sampled mean selected-token fraction over the attention
//!   layers (every `--retain-every` steps).
//! - tok/s per arm (the T4 read; CPU posture, box state recorded in the
//!   bench doc by the operator — this binary prints timing only).
//!
//! # Corpus
//!
//! `--corpus-file` is REQUIRED — pass a plain-text file (≥ ctx+1 tokens after
//! BPE). The token stream is the file's first ctx+1 tokens; positions
//! compared start at `--compare-from` (0 = compare everything).
//!
//! # Example (the league posture)
//!
//! ```sh
//! cargo run --release --features flashmemory_budget,deltanet_ternary_inference,bonsai2_hadamard \
//!   --bin flashmemory_decode_gate -- \
//!   --gguf ../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf \
//!   --corpus-file /tmp/corpus.txt --ctx 32768 --rho-list 0.05,0.10,0.15,0.30 \
//!   --bucket 4096 --out /tmp/fm_decode_gate
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use riir_infer_core::deltanet::flashmemory_gqa::FlashMemoryConfig;
use riir_infer_core::deltanet::flashmemory_gqa::{FlashMemoryDecodeState, kv_reduction_ratio};
use riir_infer_core::deltanet::ternary_forward::{
    forward_qwen_deltanet_ternary, forward_qwen_deltanet_ternary_flashmemory,
};
use riir_infer_core::deltanet::{HybridCache, HybridForwardScratch};
use riir_infer_core::gguf_loader::{GgufFile, load_qwen_deltanet_ternary_weights_gguf};
use riir_infer_core::quant::kvq_harness::nll;
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::tokenizer::BpeTokenizer;

// Issue 1191 T4 — the Metal arms' forward + arm module (this bin lives in
// the gpu crate: the boundary law keeps riir-infer-core free of riir-*
// deps, and the gate drives the GPU arms — its natural home is HERE).
// Use sites spell the full path (the import died in the e1ee836 move;
// the bin's required-features row still guarantees the lib features).

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

const DEFAULT_GGUF: &str = "../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf";

struct Args {
    gguf: PathBuf,
    corpus_file: PathBuf,
    ctx: usize,
    rho_list: Vec<f32>,
    block: usize,
    refresh: usize,
    bucket: usize,
    compare_from: usize,
    max_steps: usize,
    retain_every: usize,
    out: PathBuf,
    fresh: bool,
    /// Issue 1191 T3: continue incomplete arms from their prefix-bin
    /// cursor (the rows recovered; the skipped steps replay for state).
    resume: bool,
    /// Issue 1191 T4: the METAL posture — every arm drives the GPU decode
    /// forward with the budget arm armed per-ρ (dense = disarmed). The
    /// walls are Metal walls; the CPU reference rows live in the bmo
    /// record and are NEVER pooled with these.
    metal: bool,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        gguf: PathBuf::from(
            std::env::var("BONSAI_PQ2_0_GGUF").unwrap_or_else(|_| DEFAULT_GGUF.to_string()),
        ),
        corpus_file: PathBuf::new(),
        ctx: 32_768,
        rho_list: vec![0.05, 0.10, 0.15, 0.30],
        block: 64,
        refresh: 64,
        bucket: 4096,
        compare_from: 0,
        max_steps: 0,
        retain_every: 16,
        // Issue 1191 T3 — the DURABLE default (the Bench-893-v1 failure
        // class: the recipe + partials lost WITH /tmp). NEVER /tmp.
        out: PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into()))
            .join(".cache/i1117"),
        fresh: false,
        resume: false,
        metal: false,
    };
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        let name: &str = &args[i];
        let value = || args[i + 1].clone();
        let mut step = 2usize; // `--flag value` pairs; valueless flags step 1
        match name {
            "--gguf" => a.gguf = value().into(),
            "--corpus-file" => a.corpus_file = value().into(),
            "--ctx" => a.ctx = value().parse().unwrap(),
            "--rho-list" => {
                a.rho_list = value()
                    .split(',')
                    .map(|s| s.trim().parse().unwrap())
                    .collect();
            }
            "--block" => a.block = value().parse().unwrap(),
            "--refresh" => a.refresh = value().parse().unwrap(),
            "--bucket" => a.bucket = value().parse().unwrap(),
            "--compare-from" => a.compare_from = value().parse().unwrap(),
            "--max-steps" => a.max_steps = value().parse().unwrap(),
            "--retain-every" => a.retain_every = value().parse().unwrap(),
            "--out" => a.out = value().into(),
            "--fresh" => {
                a.fresh = true;
                step = 1;
            }
            "--resume" => {
                a.resume = true;
                step = 1;
            }
            "--metal" => {
                a.metal = true;
                step = 1;
            }
            _ => anyhow::bail!("unknown arg {name} (see the module doc)"),
        }
        i += step;
    }
    if a.corpus_file.as_os_str().is_empty() {
        anyhow::bail!("--corpus-file is required (plain text, >= ctx+1 tokens)");
    }
    if a.rho_list.is_empty() {
        anyhow::bail!("--rho-list must name at least one rho");
    }
    for &rho in &a.rho_list {
        assert!(rho > 0.0 && rho <= 1.0, "rho {rho} outside (0, 1]");
    }
    assert!(a.bucket > 0 && a.block > 0 && a.refresh > 0);
    if a.max_steps == 0 {
        a.max_steps = a.ctx;
    }
    assert!(
        a.max_steps <= a.ctx,
        "--max-steps {} exceeds --ctx {}",
        a.max_steps,
        a.ctx
    );
    Ok(a)
}

/// One arm's per-position observation.
struct ArmScore {
    nll: Vec<f32>,
    argmax: Vec<usize>,
    /// Sampled mean selected-token fraction (sparse arms only; dense = 1.0).
    mean_selected_fraction: f64,
    wall_secs: f64,
}

// ── Arm persistence (the killed-run lesson; see the module doc) ────────────

const ARM_BIN_MAGIC: &[u8; 4] = b"FMG1";

fn inputs_digest(args: &Args) -> Result<[u8; 32]> {
    let mut h = blake3::Hasher::new();
    h.update(ARM_BIN_MAGIC);
    h.update(&args.ctx.to_le_bytes());
    h.update(&args.max_steps.to_le_bytes());
    h.update(&args.block.to_le_bytes());
    h.update(&args.refresh.to_le_bytes());
    h.update(&args.bucket.to_le_bytes());
    h.update(&args.compare_from.to_le_bytes());
    // Issue 1191 T4: the POSTURE is part of the claim — Metal and CPU
    // bins never cross-load (the walls are not comparable either).
    h.update(&[u8::from(args.metal)]);
    h.update(
        &args
            .rho_list
            .iter()
            .flat_map(|r| r.to_le_bytes())
            .collect::<Vec<u8>>(),
    );
    let corpus = std::fs::read(&args.corpus_file)
        .with_context(|| format!("digest corpus {}", args.corpus_file.display()))?;
    h.update(&(corpus.len() as u64).to_le_bytes());
    h.update(&corpus);
    // First 64 MiB of the gguf: catches a swapped model without hashing ~5 GB.
    let mut gguf = std::fs::File::open(&args.gguf)
        .with_context(|| format!("digest gguf {}", args.gguf.display()))?;
    let mut head = vec![0u8; 64 * 1024 * 1024];
    let n = std::io::Read::read(&mut gguf, &mut head)?;
    h.update(&(n as u64).to_le_bytes());
    h.update(&head[..n]);
    Ok(*h.finalize().as_bytes())
}

/// Serializes one completed (or in-flight dense prefix) arm.
///
/// Layout: magic[4] · digest[32] · steps u64 · complete u8 · n_written u64 ·
/// mean_selected_fraction f64 · wall_secs f64 · n_written × (nll f32 · argmax u64).
struct ArmBinMeta {
    expected_steps: usize,
    complete: bool,
    mean_selected_fraction: f64,
    wall_secs: f64,
}

fn write_arm_bin(
    path: &std::path::Path,
    digest: &[u8; 32],
    meta: &ArmBinMeta,
    nll: &[f32],
    argmax: &[usize],
) -> Result<()> {
    let mut buf = Vec::with_capacity(4 + 32 + 8 + 1 + 8 + 8 + 8 + nll.len() * 12);
    buf.extend_from_slice(ARM_BIN_MAGIC);
    buf.extend_from_slice(digest);
    buf.extend_from_slice(&(meta.expected_steps as u64).to_le_bytes());
    buf.push(u8::from(meta.complete));
    buf.extend_from_slice(&(nll.len() as u64).to_le_bytes());
    buf.extend_from_slice(&meta.mean_selected_fraction.to_le_bytes());
    buf.extend_from_slice(&meta.wall_secs.to_le_bytes());
    for (n, a) in nll.iter().zip(argmax) {
        buf.extend_from_slice(&n.to_le_bytes());
        buf.extend_from_slice(&(*a as u64).to_le_bytes());
    }
    // Temp + rename: a kill mid-write never truncates the previous bin.
    let tmp = path.with_extension("bin.tmp");
    std::fs::write(&tmp, &buf).with_context(|| format!("write {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("rename into {}", path.display()))?;
    Ok(())
}

/// Load a bin; `allow_prefix` accepts complete=0 bins and returns the
/// cursor (the strict form rejects them — the completed-bin reload law).
fn load_arm_bin_opts(
    path: &std::path::Path,
    digest: &[u8; 32],
    expected_steps: usize,
    allow_prefix: bool,
) -> Result<(ArmScore, usize)> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let bad = |why: &str| anyhow::anyhow!("{}: {}", path.display(), why);
    if bytes.len() < 4 + 32 + 8 + 1 + 8 + 8 + 8 {
        return Err(bad("truncated header"));
    }
    if &bytes[..4] != ARM_BIN_MAGIC {
        return Err(bad("bad magic"));
    }
    if &bytes[4..36] != digest as &[u8] {
        return Err(bad("inputs digest mismatch (corpus/model/args changed)"));
    }
    let steps = u64::from_le_bytes(bytes[36..44].try_into().unwrap()) as usize;
    if steps != expected_steps {
        return Err(bad(&format!("steps {steps} != expected {expected_steps}")));
    }
    let complete = bytes[44] == 1;
    let n = u64::from_le_bytes(bytes[45..53].try_into().unwrap()) as usize;
    if complete && n != steps {
        return Err(bad(&format!("n_written {n} != steps {steps}")));
    }
    if !complete {
        if !allow_prefix {
            return Err(bad("in-flight prefix (complete=0; pass --resume)"));
        }
        if n > steps {
            return Err(bad(&format!("prefix n_written {n} > steps {steps}")));
        }
    }
    let mean_sel = f64::from_le_bytes(bytes[53..61].try_into().unwrap());
    let wall = f64::from_le_bytes(bytes[61..69].try_into().unwrap());
    let body = &bytes[69..];
    if body.len() != n * 12 {
        return Err(bad(&format!("body {} != {} rows", body.len(), n)));
    }
    let (chunks, rest) = body.as_chunks::<12>();
    if !rest.is_empty() {
        return Err(bad("trailing bytes"));
    }
    let mut nll = Vec::with_capacity(n);
    let mut argmax = Vec::with_capacity(n);
    for row in chunks {
        nll.push(f32::from_le_bytes(row[..4].try_into().unwrap()));
        argmax.push(u64::from_le_bytes(row[4..12].try_into().unwrap()) as usize);
    }
    let rows = ArmScore {
        nll,
        argmax,
        mean_selected_fraction: mean_sel,
        wall_secs: wall,
    };
    let cursor = if complete { steps } else { n };
    Ok((rows, cursor))
}

/// The strict completed-bin form (the incumbent call sites).
fn load_arm_bin(
    path: &std::path::Path,
    digest: &[u8; 32],
    expected_steps: usize,
) -> Result<ArmScore> {
    load_arm_bin_opts(path, digest, expected_steps, false).map(|(rows, _)| rows)
}

fn arm_bin_path(out: &std::path::Path, name: &str) -> std::path::PathBuf {
    out.join(format!("{name}.bin"))
}

fn percentile_with_support(rows: &mut [f32], p: f64) -> (f32, usize) {
    rows.sort_by(|a, b| a.total_cmp(b));
    let n = rows.len();
    // Nearest rank (ceil(p·n) − 1): truncating n·p lands on n − 1 (the MAX)
    // for every n ≤ 1/(1 − p).
    let idx = ((p * n as f64).ceil() as usize).clamp(1, n.max(1)) - 1;
    // Tail support = observations at or above the reported order statistic —
    // under ~10 the "p95" is the max wearing a name and is reported as such.
    (rows[idx], n - idx)
}

fn main() -> Result<()> {
    let args = parse_args()?;
    std::fs::create_dir_all(&args.out)?;

    // ── Model + tokenizer ──────────────────────────────────────────────────
    println!("# loading {}", args.gguf.display());
    let load_start = Instant::now();
    let (config, weights) = load_qwen_deltanet_ternary_weights_gguf(&args.gguf)
        .with_context(|| format!("load {}", args.gguf.display()))?;
    println!(
        "# loaded in {:.1}s — layers={} attn={} embd={} vocab={} kv_heads={} head_dim={}",
        load_start.elapsed().as_secs_f64(),
        config.n_layer,
        weights
            .layer_types
            .iter()
            .filter(|&&lt| lt == riir_infer_core::types::DeltaNetLayerType::Attention)
            .count(),
        config.n_embd,
        config.vocab_size,
        config.n_kv_head,
        config.head_dim,
    );

    let gguf = GgufFile::open(&args.gguf).context("re-open gguf for tokenizer")?;
    let tok = BpeTokenizer::from_gguf(&gguf).context("gpt2 BPE tokenizer from gguf")?;
    let text = std::fs::read_to_string(&args.corpus_file)
        .with_context(|| format!("read corpus {}", args.corpus_file.display()))?;
    let all_tokens = tok.encode(&text);
    let need = args.ctx + 1;
    anyhow::ensure!(
        all_tokens.len() >= need,
        "corpus tokenizes to {} tokens < ctx+1 = {} — pass a longer file",
        all_tokens.len(),
        need
    );
    let tokens: Vec<usize> = all_tokens[..need].to_vec();
    println!(
        "# corpus {} → {} tokens (need {need}); stepped positions 0..{}",
        args.corpus_file.display(),
        all_tokens.len(),
        args.max_steps
    );

    let rope = RopeFreqTable::new(config.rope_theta, config.head_dim);
    let n_attn = weights
        .layer_types
        .iter()
        .filter(|&&lt| lt == riir_infer_core::types::DeltaNetLayerType::Attention)
        .count();
    let last_attn_layer = weights
        .layer_types
        .iter()
        .rposition(|&lt| lt == riir_infer_core::types::DeltaNetLayerType::Attention)
        .expect("at least one attention layer");

    let steps = args.max_steps;
    let x_len = config.vocab_size.max(config.n_embd);

    let digest = inputs_digest(&args)?;

    // Issue 1191 T3 — the corpus recipe record: the v1 failure class (the
    // corpus file list lost with /tmp) is impossible by construction —
    // the recipe lands at startup, before any compute.
    {
        let corpus_bytes = std::fs::read(&args.corpus_file).unwrap_or_default();
        let recipe = serde_json::json!({
            "digest_hex": hex(&digest),
            "gguf": args.gguf.display().to_string(),
            "corpus_file": args.corpus_file.display().to_string(),
            "corpus_len": corpus_bytes.len(),
            "corpus_blake3": blake3::hash(&corpus_bytes).to_string(),
            "ctx": args.ctx,
            "max_steps": args.max_steps,
            "block": args.block,
            "refresh": args.refresh,
            "bucket": args.bucket,
            "compare_from": args.compare_from,
            "rho_list": args.rho_list,
            "posture": if args.metal { "metal" } else { "cpu" },
            "written_at": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        });
        let recipe_path = args.out.join("recipe.json");
        let tmp = recipe_path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(&recipe).unwrap())
            .with_context(|| format!("write {}", tmp.display()))?;
        std::fs::rename(&tmp, &recipe_path)
            .with_context(|| format!("rename into {}", recipe_path.display()))?;
        println!("# recipe → {}", recipe_path.display());
    }

    // ── Dense reference arm (load-or-compute; the bin persists progressively) ─
    let dense_bin = arm_bin_path(&args.out, "dense");
    let dense = if !args.fresh
        && let Ok(loaded) = load_arm_bin(&dense_bin, &digest, steps)
    {
        println!(
            "# arm: dense (reference) — LOADED {} ({:.1}s wall recorded)",
            dense_bin.display(),
            loaded.wall_secs
        );
        loaded
    } else if args.resume
        && let Ok((prefix, cursor)) = load_arm_bin_opts(&dense_bin, &digest, steps, true)
        && cursor > 0
        && cursor < steps
    {
        // --resume: replay 0..cursor through a fresh cache (deterministic —
        // the state rebuild costs the skipped steps' compute, never their
        // wall), then continue from the cursor.
        println!(
            "# arm: dense (reference) — RESUME from cursor {cursor}/{} ({} rows recovered)",
            steps, cursor
        );
        let t0 = Instant::now();
        let mut cache =
            HybridCache::with_layer_types_bounded(&config, &weights.layer_types, steps + 2);
        let mut scratch = HybridForwardScratch::new(&config);
        let mut x = vec![0.0f32; x_len];
        let mut dense_nll = prefix.nll.clone();
        dense_nll.resize(steps, 0.0);
        let mut dense_argmax = prefix.argmax.clone();
        dense_argmax.resize(steps, 0);
        for pos in cursor..steps {
            let logits = forward_qwen_deltanet_ternary(
                &mut x,
                &weights,
                &mut cache,
                tokens[pos],
                pos,
                &config,
                &mut scratch,
                &rope,
            );
            dense_nll[pos] = nll(logits, tokens[pos + 1]) as f32;
            dense_argmax[pos] = argmax(logits);
            if pos > cursor && pos % args.bucket == 0 {
                let meta = ArmBinMeta {
                    expected_steps: steps,
                    complete: false,
                    mean_selected_fraction: 1.0,
                    wall_secs: prefix.wall_secs + t0.elapsed().as_secs_f64(),
                };
                let _ = write_arm_bin(
                    &dense_bin,
                    &digest,
                    &meta,
                    &dense_nll[..pos],
                    &dense_argmax[..pos],
                );
            }
        }
        let dense_wall = prefix.wall_secs + t0.elapsed().as_secs_f64();
        drop(cache);
        let meta = ArmBinMeta {
            expected_steps: steps,
            complete: true,
            mean_selected_fraction: 1.0,
            wall_secs: dense_wall,
        };
        write_arm_bin(&dense_bin, &digest, &meta, &dense_nll, &dense_argmax)?;
        println!(
            "#   dense arm (resumed) persisted → {}",
            dense_bin.display()
        );
        ArmScore {
            nll: dense_nll,
            argmax: dense_argmax,
            mean_selected_fraction: 1.0,
            wall_secs: dense_wall,
        }
    } else if args.metal {
        // Issue 1191 T4 — the METAL dense arm: the GPU decode forward,
        // budget arm DISARMED (the incumbent dense dispatch).
        println!("# arm: dense (reference, METAL)");
        riir_infer_gpu::flashmemory_budget_metal::set_flashmemory_budget_metal(false);
        let ctx = riir_infer_gpu::cubecl_runtime::CubeCLContext::new()
            .expect("Metal ctx (--metal requires a Metal device)");
        let mut gpu_config = config.clone();
        // The Metal KV single-buffer bound (riir-ai Issue 994 / riir-infer
        // Issue 1114): clamp to the largest anchor-validated rung — the
        // 128K axis IS that rung (131,072 ≤ 131,136), and the walk needs
        // ≥ steps + 2 rows.
        gpu_config.block_size = gpu_config.block_size.min(131_072).max(steps + 2);
        let mut gpu = riir_infer_gpu::ternary_deltanet_gpu_forward::TernaryDeltanetGpuForward::new(
            &ctx,
            &gpu_config,
            &weights,
        );
        let t0 = Instant::now();
        let mut dense_nll = vec![0f32; steps];
        let mut dense_argmax = vec![0usize; steps];
        for pos in 0..steps {
            if pos > 0 && pos % 64 == 0 {
                let el = t0.elapsed().as_secs_f64();
                eprintln!(
                    "# progress dense(M) {pos}/{steps} — {el:.0}s — {:.2} tok/s",
                    pos as f64 / el
                );
            }
            gpu.set_input_token(&weights, tokens[pos]);
            let logits = gpu.forward_token();
            dense_nll[pos] = nll(&logits, tokens[pos + 1]) as f32;
            dense_argmax[pos] = argmax(&logits);
            if pos > 0 && pos % args.bucket == 0 {
                let meta = ArmBinMeta {
                    expected_steps: steps,
                    complete: false,
                    mean_selected_fraction: 1.0,
                    wall_secs: t0.elapsed().as_secs_f64(),
                };
                let _ = write_arm_bin(
                    &dense_bin,
                    &digest,
                    &meta,
                    &dense_nll[..pos],
                    &dense_argmax[..pos],
                );
            }
        }
        let dense_wall = t0.elapsed().as_secs_f64();
        let meta = ArmBinMeta {
            expected_steps: steps,
            complete: true,
            mean_selected_fraction: 1.0,
            wall_secs: dense_wall,
        };
        write_arm_bin(&dense_bin, &digest, &meta, &dense_nll, &dense_argmax)?;
        println!("#   dense (METAL) arm persisted → {}", dense_bin.display());
        ArmScore {
            nll: dense_nll,
            argmax: dense_argmax,
            mean_selected_fraction: 1.0,
            wall_secs: dense_wall,
        }
    } else {
        println!("# arm: dense (reference)");
        let t0 = Instant::now();
        let mut cache =
            HybridCache::with_layer_types_bounded(&config, &weights.layer_types, steps + 2);
        let mut scratch = HybridForwardScratch::new(&config);
        let mut x = vec![0.0f32; x_len];
        let mut dense_nll = vec![0f32; steps];
        let mut dense_argmax = vec![0usize; steps];
        for pos in 0..steps {
            if pos > 0 && pos % 64 == 0 {
                let el = t0.elapsed().as_secs_f64();
                eprintln!(
                    "# progress dense {pos}/{steps} — {el:.0}s — {:.2} tok/s",
                    pos as f64 / el
                );
            }
            let logits = forward_qwen_deltanet_ternary(
                &mut x,
                &weights,
                &mut cache,
                tokens[pos],
                pos,
                &config,
                &mut scratch,
                &rope,
            );
            dense_nll[pos] = nll(logits, tokens[pos + 1]) as f32;
            dense_argmax[pos] = argmax(logits);
            // Progressive prefix: a mid-dense kill keeps the counted rows.
            if pos > 0 && pos % args.bucket == 0 {
                let meta = ArmBinMeta {
                    expected_steps: steps,
                    complete: false,
                    mean_selected_fraction: 1.0,
                    wall_secs: t0.elapsed().as_secs_f64(),
                };
                let _ = write_arm_bin(
                    &dense_bin,
                    &digest,
                    &meta,
                    &dense_nll[..pos],
                    &dense_argmax[..pos],
                );
            }
        }
        let dense_wall = t0.elapsed().as_secs_f64();
        drop(cache);
        let meta = ArmBinMeta {
            expected_steps: steps,
            complete: true,
            mean_selected_fraction: 1.0,
            wall_secs: dense_wall,
        };
        write_arm_bin(&dense_bin, &digest, &meta, &dense_nll, &dense_argmax)?;
        println!("#   dense arm persisted → {}", dense_bin.display());
        ArmScore {
            nll: dense_nll,
            argmax: dense_argmax,
            mean_selected_fraction: 1.0,
            wall_secs: dense_wall,
        }
    };
    println!(
        "#   dense wall {:.1}s — {:.1} tok/s",
        dense.wall_secs,
        steps as f64 / dense.wall_secs
    );

    // ── Sparse arms (load-or-compute; each persists the moment it completes) ─
    let mut arms: Vec<(f32, ArmScore)> = Vec::new();
    for &rho in &args.rho_list {
        let rho_bin = arm_bin_path(&args.out, &format!("rho_{rho:.2}"));
        if !args.fresh
            && let Ok(loaded) = load_arm_bin(&rho_bin, &digest, steps)
        {
            println!(
                "# arm: flashmemory rho={rho} — LOADED {} ({:.1}s wall recorded, sel frac {:.4})",
                rho_bin.display(),
                loaded.wall_secs,
                loaded.mean_selected_fraction
            );
            arms.push((rho, loaded));
            continue;
        }
        let resumed = if args.resume {
            load_arm_bin_opts(&rho_bin, &digest, steps, true)
                .ok()
                .filter(|(_, cursor)| *cursor > 0 && *cursor < steps)
        } else {
            None
        };
        println!(
            "# arm: flashmemory rho={rho} (block {} refresh {}{})",
            args.block,
            args.refresh,
            if args.metal { ", METAL" } else { "" }
        );
        if args.metal {
            // Issue 1191 T4 — the METAL budget arm at this rho: the GPU
            // decode forward CONSTRUCTED per rho (the budget state is
            // per-construction), the arm armed via the authoritative
            // setter + the env row (rho/block/refresh reach the ctor).
            unsafe {
                std::env::set_var("RIIR_FM_RHO", format!("{rho}"));
                std::env::set_var("RIIR_FM_BLOCK", format!("{}", args.block));
                std::env::set_var("RIIR_FM_REFRESH", format!("{}", args.refresh));
            }
            riir_infer_gpu::flashmemory_budget_metal::set_flashmemory_budget_metal(true);
            let ctx = riir_infer_gpu::cubecl_runtime::CubeCLContext::new()
                .expect("Metal ctx (--metal requires a Metal device)");
            let mut gpu_config = config.clone();
            // The same 128K single-buffer bound (see the dense arm).
            gpu_config.block_size = gpu_config.block_size.min(131_072).max(steps + 2);
            let mut gpu =
                riir_infer_gpu::ternary_deltanet_gpu_forward::TernaryDeltanetGpuForward::new(
                    &ctx,
                    &gpu_config,
                    &weights,
                );
            let t0 = Instant::now();
            let mut arm_nll = vec![0f32; steps];
            let mut arm_argmax = vec![0usize; steps];
            let mut selected_sum = 0f64;
            let mut selected_n = 0f64;
            for pos in 0..steps {
                if pos > 0 && pos % 64 == 0 {
                    let el = t0.elapsed().as_secs_f64();
                    eprintln!(
                        "# progress rho={rho}(M) {pos}/{steps} — {el:.0}s — {:.2} tok/s",
                        pos as f64 / el
                    );
                }
                gpu.set_input_token(&weights, tokens[pos]);
                let logits = gpu.forward_token();
                arm_nll[pos] = nll(&logits, tokens[pos + 1]) as f32;
                arm_argmax[pos] = argmax(&logits);
                if pos % args.retain_every == 0
                    && pos > 0
                    && let Some(frac) = gpu.fm_selected_fraction(last_attn_layer, pos)
                {
                    selected_sum += frac;
                    selected_n += 1.0;
                }
                if pos > 0 && pos % args.bucket == 0 {
                    let meta = ArmBinMeta {
                        expected_steps: steps,
                        complete: false,
                        mean_selected_fraction: if selected_n > 0.0 {
                            selected_sum / selected_n
                        } else {
                            1.0
                        },
                        wall_secs: t0.elapsed().as_secs_f64(),
                    };
                    let _ = write_arm_bin(
                        &rho_bin,
                        &digest,
                        &meta,
                        &arm_nll[..pos],
                        &arm_argmax[..pos],
                    );
                }
            }
            let wall = t0.elapsed().as_secs_f64();
            let mean_frac = if selected_n > 0.0 {
                selected_sum / selected_n
            } else {
                1.0
            };
            println!(
                "#   wall {:.1}s — {:.1} tok/s — mean selected fraction {:.4} (last attn layer, sampled every {})",
                wall,
                steps as f64 / wall,
                mean_frac,
                args.retain_every
            );
            write_arm_bin(
                &rho_bin,
                &digest,
                &ArmBinMeta {
                    expected_steps: steps,
                    complete: true,
                    mean_selected_fraction: mean_frac,
                    wall_secs: wall,
                },
                &arm_nll,
                &arm_argmax,
            )?;
            println!(
                "#   rho={rho} (METAL) arm persisted → {}",
                rho_bin.display()
            );
            arms.push((
                rho,
                ArmScore {
                    nll: arm_nll,
                    argmax: arm_argmax,
                    mean_selected_fraction: mean_frac,
                    wall_secs: wall,
                },
            ));
            continue;
        }
        let t0 = Instant::now();
        let mut cache =
            HybridCache::with_layer_types_bounded(&config, &weights.layer_types, steps + 2);
        let mut scratch = HybridForwardScratch::new(&config);
        let mut fm = FlashMemoryDecodeState::new(
            &config,
            &weights.layer_types,
            steps + 2,
            FlashMemoryConfig {
                block_size: args.block,
                refresh_period: args.refresh,
                threshold: 0.5, // unused by the budget rule
            },
            rho,
        );
        let mut x = vec![0.0f32; x_len];
        let mut arm_nll = vec![0f32; steps];
        let mut arm_argmax = vec![0usize; steps];
        let mut selected_sum = 0f64;
        let mut selected_n = 0f64;
        let start_pos = if let Some((prefix, cursor)) = resumed.as_ref() {
            // --resume: replay 0..cursor (state rebuild — deterministic),
            // keep the recovered rows, continue from the cursor.
            println!(
                "# arm: flashmemory rho={rho} — RESUME from cursor {cursor}/{} ({} rows recovered, sel frac {:.4} so far)",
                steps, cursor, prefix.mean_selected_fraction
            );
            arm_nll[..*cursor].copy_from_slice(&prefix.nll[..*cursor]);
            arm_argmax[..*cursor].copy_from_slice(&prefix.argmax[..*cursor]);
            selected_sum = prefix.mean_selected_fraction * (*cursor as f64);
            selected_n = *cursor as f64;
            *cursor
        } else {
            0
        };
        for pos in start_pos..steps {
            if pos > 0 && pos % 64 == 0 {
                let el = t0.elapsed().as_secs_f64();
                eprintln!(
                    "# progress rho={rho} {pos}/{steps} — {el:.0}s — {:.2} tok/s",
                    pos as f64 / el
                );
            }
            let logits = forward_qwen_deltanet_ternary_flashmemory(
                &mut x,
                &weights,
                &mut cache,
                &mut fm,
                tokens[pos],
                pos,
                &config,
                &mut scratch,
                &rope,
            );
            arm_nll[pos] = nll(logits, tokens[pos + 1]) as f32;
            arm_argmax[pos] = argmax(logits);
            if pos % args.retain_every == 0
                && pos > 0
                && let Some(st) = &fm.layers[last_attn_layer]
            {
                let selection = st.selector.selection();
                let (selected, total, _) = kv_reduction_ratio(selection, &st.block_cache, pos + 1);
                if total > 0 {
                    selected_sum += selected as f64 / total as f64;
                    selected_n += 1.0;
                }
            }
            // Per-bucket prefix: a mid-arm kill keeps the counted rows — the
            // same progressive persistence the dense arm has. The m3 v2 run
            // lost ~27 h of ρ 0.15 at 94% because only the dense arm wrote
            // prefixes (issue 1117, the 2026-11-04 post-mortem). The prefix
            // is post-hoc collection value (a restart recomputes the arm —
            // resume covers completed bins only); mean_selected_fraction
            // carries the running sample so a collected prefix still reads
            // the KV-bytes proxy.
            if pos > 0 && pos % args.bucket == 0 {
                let meta = ArmBinMeta {
                    expected_steps: steps,
                    complete: false,
                    mean_selected_fraction: if selected_n > 0.0 {
                        selected_sum / selected_n
                    } else {
                        1.0
                    },
                    wall_secs: t0.elapsed().as_secs_f64(),
                };
                let _ = write_arm_bin(
                    &rho_bin,
                    &digest,
                    &meta,
                    &arm_nll[..pos],
                    &arm_argmax[..pos],
                );
            }
        }
        let this_wall = t0.elapsed().as_secs_f64();
        let wall = this_wall + resumed.as_ref().map_or(0.0, |(p, _)| p.wall_secs);
        drop(cache);
        let mean_frac = if selected_n > 0.0 {
            selected_sum / selected_n
        } else {
            1.0
        };
        println!(
            "#   wall {:.1}s — {:.1} tok/s (effective) — mean selected fraction {:.4} (last attn layer, sampled every {})",
            wall,
            steps as f64 / wall,
            mean_frac,
            args.retain_every
        );
        write_arm_bin(
            &rho_bin,
            &digest,
            &ArmBinMeta {
                expected_steps: steps,
                complete: true,
                mean_selected_fraction: mean_frac,
                wall_secs: wall,
            },
            &arm_nll,
            &arm_argmax,
        )?;
        println!("#   rho={rho} arm persisted → {}", rho_bin.display());
        arms.push((
            rho,
            ArmScore {
                nll: arm_nll,
                argmax: arm_argmax,
                mean_selected_fraction: mean_frac,
                wall_secs: wall,
            },
        ));
    }

    // ── Comparison + report ────────────────────────────────────────────────
    let mut md = String::new();
    let mut json_rows = Vec::new();
    md.push_str(&format!(
        "# FlashMemory budget decode gate (Issue 1117 T3/T4)\n\n\
         model `{}` · block {} · refresh {} · steps {steps} · bucket {} · compare-from {} · incremental centroids\n\n\
         | arm | mean sel frac | tok/s | pooled ΔNLL mean | pooled ΔNLL p95 (support) | pooled ΔNLL max | top-1 agree |\n\
         |---|---|---|---|---|---|---|\n",
        args.gguf.display(),
        args.block,
        args.refresh,
        args.bucket,
        args.compare_from,
    ));
    for (rho, arm) in &arms {
        let deltas: Vec<f32> = arm.nll[args.compare_from..]
            .iter()
            .zip(&dense.nll[args.compare_from..])
            .map(|(a, b)| a - b)
            .collect();
        let agree = arm.argmax[args.compare_from..]
            .iter()
            .zip(&dense.argmax[args.compare_from..])
            .filter(|(a, b)| a == b)
            .count();
        let n_cmp = deltas.len();
        let mean: f32 = deltas.iter().sum::<f32>() / n_cmp as f32;
        let mut sorted = deltas.clone();
        let (p95, support) = percentile_with_support(&mut sorted, 0.95);
        let max = sorted[n_cmp - 1];
        let agree_pct = 100.0 * agree as f64 / n_cmp as f64;
        md.push_str(&format!(
            "| ρ {rho} | {:.4} | {:.1} | {mean:+.4} | {p95:+.4} ({support}) | {max:+.4} | {agree_pct:.2}% |\n",
            arm.mean_selected_fraction,
            steps as f64 / arm.wall_secs,
        ));
        json_rows.push(serde_json::json!({
            "rho": rho,
            "mean_selected_fraction": arm.mean_selected_fraction,
            "tok_per_s": steps as f64 / arm.wall_secs,
            "delta_nll_mean": mean,
            "delta_nll_p95": p95,
            "delta_nll_p95_support": support,
            "delta_nll_max": max,
            "top1_agreement_pct": agree_pct,
        }));

        // Per-bucket rows (sparse arm detail).
        md.push_str(&format!(
            "\n### ρ {rho} — per-bucket (width {})\n\n| bucket | n | ΔNLL mean | ΔNLL p95 (support) | ΔNLL max | top-1 agree |\n|---|---|---|---|---|---|\n",
            args.bucket
        ));
        let mut bstart = args.compare_from - args.compare_from % args.bucket;
        while bstart < steps {
            let bend = (bstart + args.bucket).min(steps);
            let lo = bstart.max(args.compare_from);
            if bend > lo {
                let mut bdelta: Vec<f32> = arm.nll[lo..bend]
                    .iter()
                    .zip(&dense.nll[lo..bend])
                    .map(|(a, b)| a - b)
                    .collect();
                let bn = bdelta.len();
                let bmean: f32 = bdelta.iter().sum::<f32>() / bn as f32;
                let (bp95, bsupport) = percentile_with_support(&mut bdelta, 0.95);
                let bmax = bdelta.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let bagree = arm.argmax[lo..bend]
                    .iter()
                    .zip(&dense.argmax[lo..bend])
                    .filter(|(a, b)| a == b)
                    .count();
                md.push_str(&format!(
                    "| [{bstart}, {bend}) | {bn} | {bmean:+.4} | {bp95:+.4} ({bsupport}) | {bmax:+.4} | {:.2}% |\n",
                    100.0 * bagree as f64 / bn as f64
                ));
            }
            bstart += args.bucket;
        }
    }

    // Dense arm summary line + run config.
    md.push_str(&format!(
        "\nDense reference: {:.1} tok/s; pooled NLL mean {:.4}.\n",
        steps as f64 / dense.wall_secs,
        dense.nll[args.compare_from..].iter().sum::<f32>() as f64
            / (steps - args.compare_from) as f64
    ));

    println!("{md}");
    std::fs::write(args.out.join("report.md"), &md)?;
    std::fs::write(
        args.out.join("report.json"),
        serde_json::to_string_pretty(&serde_json::json!({
            "issue": 1117,
            "gguf": args.gguf.display().to_string(),
            "corpus": args.corpus_file.display().to_string(),
            "ctx": args.ctx,
            "steps": steps,
            "block": args.block,
            "refresh": args.refresh,
            "bucket": args.bucket,
            "compare_from": args.compare_from,
            "attn_layers": n_attn,
            "dense_tok_per_s": steps as f64 / dense.wall_secs,
            "arms": json_rows,
        }))?,
    )?;
    println!("# wrote {}", args.out.join("report.md").display());
    Ok(())
}

fn argmax(logits: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}
