//! `i1250_kvshare_cell` — katgpt-rs Issue 1250 arm B T5: the pre-registered
//! cross-layer KV-sharing init cell (the YASA distill, Research 698).
//!
//! Load the LEAGUE checkpoint (`Ternary-Bonsai-2.5-27B-PQ2_0.gguf`, folded —
//! this bin requires `bonsai2_hadamard` at the run), fit the closed-form
//! KV-sharing init over the fit corpus' first `--anchors` tokens (default
//! 4096, `tokens.fit.tokens`), then score the three arms over the eval
//! corpus (`tokens.chat.tokens`, 14,075 tokens; the pre-registration pins
//! its SHA-256 and this bin verifies it fail-closed):
//!
//! 1. **baseline** — the checkpoint as-is (the plain forward);
//! 2. **naive copy** — consumer layer 7 reads layer 3's cached K/V
//!    verbatim (its own `W_k`/`W_v` skipped);
//! 3. **aligned init** — the same share THROUGH the fitted plan
//!    (RoPE-commuting key rotation on the queries, rank-128 value map
//!    folded per q-head group).
//!
//! Eval protocol (the pre-registration): teacher-forced NLL, 1024-token
//! windows with a 512-token unscored warm-up prefix each (13 windows at
//! 14,075 tokens; the tail remainder joins the last window); per-window
//! mean NLL per arm; `delta_arm = mean NLL(arm) − mean NLL(baseline)`
//! paired per window. The T7 bar: aligned delta ≤ **0.5× naive delta**
//! with a bootstrap 95% CI over the 13 windows (10,000 resamples,
//! SplitMix64 seeded from `blake3(eval_bytes) XOR blake3(fit_bytes)`);
//! CI straddling the bar boundary = UNDECIDED, never PASS; aligned worse
//! than naive at the CI level = NEGATIVE; fitted angle norms all
//! `‖θ‖₂ < 1e-3` = NO-OP (the key spaces are already aligned — recorded,
//! not negative).
//!
//! Output: per-window rows + deltas + the verdict on stdout, and a JSON
//! artifact (`--out`) with the per-window NLLs, the fitted diagnostics
//! (per-head angle norms, value-map residual, anchor digest), SHA-256 +
//! BLAKE3 of both token files, the model's BLAKE3, and box-state
//! provenance (the `scripts/bench_preflight.sh` PROVENANCE line when the
//! sibling script is present, plus the inline sysctl/pmset quote — the
//! twt_perf_tg_pp shape; a failed read degrades to a printed marker, it
//! never fails the run).
//!
//! OFFLINE lane only — never a serving path. The REAL cell run is
//! owner-side (a 27B load + ~90 min of CPU decode per arm at the league
//! posture); this session builds + hermetic-gates it only.
//!
//! ```text
//! cargo run --release --features kv_share_align,bonsai2_hadamard \
//!   --bin i1250_kvshare_cell -- \
//!   --gguf ../riir-train/data/Ternary-Bonsai-2.5-27B-PQ2_0.gguf \
//!   --fit-tokens ../riir-train/data/bonsai_mtp/tokens.fit.tokens \
//!   --eval-tokens ../riir-train/data/bonsai_mtp/tokens.chat.tokens \
//!   --anchors 4096 --out .benchmarks/i1250_kvshare_cell.json
//! ```
#![cfg(feature = "kv_share_align")]

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use sha2::{Digest as _, Sha256};

use riir_infer_core::deltanet::forward::{HybridCache, HybridForwardScratch, effective_rotary_dim};
use riir_infer_core::deltanet::ternary_forward::forward_qwen_deltanet_ternary;
use riir_infer_core::gguf_loader::load_qwen_deltanet_ternary_weights_gguf;
use riir_infer_core::kv_share_align::{
    KvShareArm, KvSharePlan, KvShareScratch, fit_kv_share_alignment,
    forward_qwen_deltanet_ternary_kvshare,
};
use riir_infer_core::rope::RopeFreqTable;

/// The pre-registration's pinned eval-corpus SHA-256.
///
/// CORRECTION (2026-11-28, zed-7bab0f17@amo): the ORIGINAL pre-registration
/// text labeled this digest "SHA-256" but the hex was actually the file's
/// BLAKE3 (the pinning command chain `blake3 || b3sum || shasum` short-
/// circuited on the first tool). The file itself NEVER drifted (mtime
/// 2026-11-27 11:12, before and after the pin); this constant now carries
/// the file's true SHA-256. The v2 cell's refusal (fail-closed at this
/// check) is the gate working as built. Disclosure folded into the
/// katgpt-rs `.issues/1250` pre-registration text in the same session.
/// (14,075 chat-shaped tokens; regenerable via
/// `cargo run --release -p riir-infer-dist --example mtp_chat_tokens`).
/// A drift REFUSES — the eval file is part of the pre-registration.
const PINNED_EVAL_SHA256: &str = "2e8ef853b86fecf39a6054841725556c6f4ac10d6bf20cfb0cbdb9afcdbfff29";
/// The pre-registration's pinned FA pair (closest gap; the confound is
/// disclosed there). Overridable — a different pair is a NEW
/// pre-registration, not a flag flip.
const DEFAULT_PRODUCER_LAYER: usize = 3;
const DEFAULT_CONSUMER_LAYER: usize = 7;
const WINDOW: usize = 1024;
const WARMUP: usize = 512;
const RESAMPLES: usize = 10_000;
/// The NO-OP falsifier: every head's fitted angle norm below this (rad).
const NOOP_ANGLE_NORM: f32 = 1e-3;

fn main() -> Result<()> {
    let mut gguf_path = PathBuf::from("../riir-train/data/Ternary-Bonsai-2.5-27B-PQ2_0.gguf");
    let mut fit_path = PathBuf::from("../riir-train/data/bonsai_mtp/tokens.fit.tokens");
    let mut eval_path = PathBuf::from("../riir-train/data/bonsai_mtp/tokens.chat.tokens");
    let mut anchors = 4096usize;
    let mut producer_layer = DEFAULT_PRODUCER_LAYER;
    let mut consumer_layer = DEFAULT_CONSUMER_LAYER;
    let mut resamples = RESAMPLES;
    let mut out: Option<PathBuf> = None;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = |name: &str| {
            args.next()
                .ok_or_else(|| anyhow::anyhow!("--{name} needs a value"))
        };
        match a.as_str() {
            "--gguf" => gguf_path = PathBuf::from(val("gguf")?),
            "--fit-tokens" => fit_path = PathBuf::from(val("fit-tokens")?),
            "--eval-tokens" => eval_path = PathBuf::from(val("eval-tokens")?),
            "--anchors" => anchors = val("anchors")?.parse()?,
            "--producer" => producer_layer = val("producer")?.parse()?,
            "--consumer" => consumer_layer = val("consumer")?.parse()?,
            "--resamples" => resamples = val("resamples")?.parse()?,
            "--out" => out = Some(PathBuf::from(val("out")?)),
            other => bail!("unknown arg {other} (see the module doc for the usage line)"),
        }
    }
    if (producer_layer, consumer_layer) != (DEFAULT_PRODUCER_LAYER, DEFAULT_CONSUMER_LAYER) {
        println!(
            "# PAIR OVERRIDE ({producer_layer}, {consumer_layer}) — the pre-registration pins (3, 7); \
             a different pair is a NEW pre-registration, never a retune"
        );
    }

    // ── inputs: refuse LOUD naming what is missing (exit 2 posture) ────
    for (label, path) in [
        ("model", gguf_path.as_path()),
        ("fit tokens", fit_path.as_path()),
        ("eval tokens", eval_path.as_path()),
    ] {
        if !path.is_file() {
            eprintln!(
                "REFUSE: the {label} file is absent at {} — the cell is pre-registered on these inputs; \
nothing here synthesizes a substitute",
                path.display()
            );
            std::process::exit(2);
        }
    }
    if anchors < 512 {
        bail!(
            "--anchors {anchors} < 512: the value-map Gram needs the anchor panel the pre-registration pins"
        );
    }

    // ── token files: u32-LE, SHA-256 verified against the pin ──────────
    let fit_bytes = std::fs::read(&fit_path).context("read fit tokens")?;
    let eval_bytes = std::fs::read(&eval_path).context("read eval tokens")?;
    let mut sha = Sha256::new();
    sha.update(&eval_bytes);
    let eval_sha = hex(&sha.finalize());
    if eval_sha != PINNED_EVAL_SHA256 {
        eprintln!(
            "REFUSE: the eval corpus' SHA-256 drifted from the pre-registration pin:\n  pinned {PINNED_EVAL_SHA256}\n  actual {eval_sha}\n\
the eval file is part of the pre-registration — regenerate it with \
`cargo run --release -p riir-infer-dist --example mtp_chat_tokens` (it must reproduce the pin) \
or file a NEW pre-registration"
        );
        std::process::exit(2);
    }
    ensure!(
        fit_bytes.len().is_multiple_of(4) && eval_bytes.len().is_multiple_of(4),
        "token files are u32-LE"
    );
    let fit_tokens: Vec<usize> = fit_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b) as usize)
        .collect();
    let eval_tokens: Vec<usize> = eval_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b) as usize)
        .collect();
    ensure!(
        fit_tokens.len() >= anchors,
        "fit corpus has {} tokens < --anchors {anchors}",
        fit_tokens.len()
    );
    println!(
        "# corpora: fit {} tokens (first {anchors} used) | eval {} tokens (sha256 pinned ✓)",
        fit_tokens.len(),
        eval_tokens.len()
    );

    // ── load the league checkpoint ─────────────────────────────────────
    let t0 = Instant::now();
    let (mut config, weights) = load_qwen_deltanet_ternary_weights_gguf(&gguf_path)
        .with_context(|| format!("load {}", gguf_path.display()))?;
    println!(
        "# model: {} | layers={} heads={}/{} kv={} hd={} rot={} hadamard={} | load {:.1}s",
        gguf_path.display(),
        config.n_layer,
        config.n_head,
        config.n_kv_head,
        weights.layers[producer_layer].attn_wq.cols(),
        config.head_dim,
        config.rope_dimension_count,
        weights.rotation.is_some(),
        t0.elapsed().as_secs_f32()
    );
    for (name, layer) in [("producer", producer_layer), ("consumer", consumer_layer)] {
        ensure!(
            weights.layer_types[layer] == riir_infer_core::types::DeltaNetLayerType::Attention,
            "{name}_layer {layer} is not a full-attention layer (the fit refuses; check the pair against full_attention_interval)"
        );
    }

    // Cache-window cap BEFORE any cache construction (the row_logit_floor
    // law): the widest pass is the fit (anchors) or an eval window.
    let window_span = WARMUP + WINDOW + 8;
    config.block_size = config.block_size.max(anchors + 8).max(window_span);
    let rope_freq = RopeFreqTable::new(config.rope_theta, effective_rotary_dim(&config));

    // ── fit ────────────────────────────────────────────────────────────
    println!("# fitting on the first {anchors} anchor tokens (teacher-forced capture pass)…");
    let t1 = Instant::now();
    let anchor_slice: Vec<u32> = fit_tokens[..anchors].iter().map(|&t| t as u32).collect();
    let fit = fit_kv_share_alignment(
        &weights,
        &config,
        &rope_freq,
        producer_layer,
        consumer_layer,
        &anchor_slice,
    )
    .map_err(anyhow::Error::msg)?;
    println!(
        "# fit done in {:.1}s: {} kv heads, hd {} (rotary {}), rank {}, anchors {}",
        t1.elapsed().as_secs_f32(),
        fit.angles_norm.len(),
        fit.head_dim,
        fit.rotary_dim,
        fit.value_rank,
        fit.n_anchors
    );
    for (g, a) in fit.angles_norm.iter().enumerate() {
        println!("#   head {g}: ‖θ‖₂ = {a:.6} rad");
    }
    println!(
        "#   value-map residual (relative, anchor metric): {:.6}",
        fit.fit_residual
    );
    println!(
        "#   anchor digest: {}",
        fit.anchor_digest
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    );
    let noop = fit.is_noop();
    if noop {
        println!(
            "# NO-OP falsifier: every head's ‖θ‖₂ < {NOOP_ANGLE_NORM} — the producer/consumer key \
spaces are already aligned on this pair; the arm closes NO-OP (recorded, not negative)"
        );
    }
    let plan = KvSharePlan::build(&fit, producer_layer, consumer_layer, &weights, &config)
        .map_err(anyhow::Error::msg)?;

    // ── the three arms over the eval windows ───────────────────────────
    let n_windows = (eval_tokens.len().saturating_sub(WARMUP)) / WINDOW;
    ensure!(
        n_windows >= 1,
        "eval corpus too short for one warmup+window"
    );
    println!(
        "# eval: {n_windows} windows × ({WARMUP} warmup + {WINDOW} scored); the tail joins the last window"
    );

    let mut windows: Vec<[f64; 3]> = Vec::with_capacity(n_windows); // [baseline, naive, aligned]
    for arm_name in ["baseline", "naive", "aligned"] {
        let t2 = Instant::now();
        let mut cache =
            HybridCache::with_layer_types_bounded(&config, &weights.layer_types, window_span);
        let mut scratch = HybridForwardScratch::new(&config);
        let mut kv = KvShareScratch::new(&config);
        let mut x = vec![0.0f32; config.vocab_size.max(config.n_embd)];
        let mut arm_windows = vec![0.0f64; n_windows];
        let mut arm_scored = vec![0usize; n_windows];
        for w in 0..n_windows {
            let start = w * WINDOW;
            let end = (start + WARMUP + WINDOW).min(eval_tokens.len());
            cache.reset();
            let mut nll = 0.0f64;
            let mut scored = 0usize;
            let arm = match arm_name {
                "baseline" => KvShareArm::Plain,
                "naive" => KvShareArm::Naive {
                    producer_layer,
                    consumer_layer,
                },
                "aligned" => KvShareArm::Aligned(&plan),
                _ => unreachable!(),
            };
            for (pos, &tok) in eval_tokens[start..end].iter().enumerate() {
                let logits = if matches!(arm, KvShareArm::Plain) {
                    forward_qwen_deltanet_ternary(
                        &mut x,
                        &weights,
                        &mut cache,
                        tok,
                        pos,
                        &config,
                        &mut scratch,
                        &rope_freq,
                    )
                } else {
                    forward_qwen_deltanet_ternary_kvshare(
                        &mut x,
                        &weights,
                        &mut cache,
                        &arm,
                        tok,
                        pos,
                        &config,
                        &mut scratch,
                        &rope_freq,
                        &mut kv,
                    )
                };
                if pos >= WARMUP && pos + 1 < end - start {
                    let target = eval_tokens[start + pos + 1];
                    // f64 logsumexp with max-subtraction, sequential folds
                    // (the si_zero_shot_ppl convention — deterministic).
                    let mut max_l = f32::NEG_INFINITY;
                    for &l in logits.iter() {
                        max_l = max_l.max(l);
                    }
                    let mut sum = 0.0f64;
                    for &l in logits.iter() {
                        sum += (f64::from(l) - f64::from(max_l)).exp();
                    }
                    let lse = f64::from(max_l) + sum.ln();
                    nll += lse - f64::from(logits[target]);
                    scored += 1;
                }
            }
            arm_windows[w] = nll / scored.max(1) as f64;
            arm_scored[w] = scored;
        }
        println!(
            "# arm {arm_name} done in {:.1}s",
            t2.elapsed().as_secs_f32()
        );
        for w in 0..n_windows {
            let slot = match arm_name {
                "baseline" => 0,
                "naive" => 1,
                _ => 2,
            };
            if windows.len() <= w {
                windows.push([0.0; 3]);
            }
            windows[w][slot] = arm_windows[w];
            let _ = arm_scored[w]; // uniform across arms by construction
        }
    }

    // ── deltas + the pre-registered bar ────────────────────────────────
    let naive_delta: Vec<f64> = windows.iter().map(|w| w[1] - w[0]).collect();
    let aligned_delta: Vec<f64> = windows.iter().map(|w| w[2] - w[0]).collect();
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    let (naive_d, aligned_d) = (mean(&naive_delta), mean(&aligned_delta));

    println!("\n# per-window mean NLL (warmup excluded)");
    println!(
        "# {:>3} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "w", "baseline", "naive", "aligned", "d_naive", "d_aligned"
    );
    for (w, win) in windows.iter().enumerate() {
        println!(
            "# {:>3} {:>12.6} {:>12.6} {:>12.6} {:>+12.6} {:>+12.6}",
            w, win[0], win[1], win[2], naive_delta[w], aligned_delta[w]
        );
    }
    println!(
        "# pooled deltas over {n_windows} windows: naive {naive_d:+.6} | aligned {aligned_d:+.6}"
    );

    // Bootstrap CIs: SplitMix64 seeded from blake3(eval) XOR blake3(fit).
    let seed_digest = {
        let e = blake3::hash(&eval_bytes);
        let f = blake3::hash(&fit_bytes);
        let mut x = [0u8; 32];
        for (o, (&a, &b)) in x.iter_mut().zip(e.as_bytes().iter().zip(f.as_bytes())) {
            *o = a ^ b;
        }
        x
    };
    let mut rng = SplitMix64(u64::from_le_bytes(seed_digest[..8].try_into().unwrap()));
    let half_cis = bootstrap_ci(&naive_delta, &aligned_delta, 0.5, resamples, &mut rng);
    let neg_cis = bootstrap_ci(&naive_delta, &aligned_delta, 1.0, resamples, &mut rng);
    println!(
        "# bootstrap({resamples}) 95% CI: [aligned − 0.5·naive] mean {:.6} CI [{:.6}, {:.6}]",
        half_cis.0, half_cis.1, half_cis.2
    );
    println!(
        "# bootstrap({resamples}) 95% CI: [aligned − naive]     mean {:.6} CI [{:.6}, {:.6}]",
        neg_cis.0, neg_cis.1, neg_cis.2
    );

    let verdict = if noop {
        "NO-OP"
    } else if neg_cis.1 > 0.0 {
        // aligned worse than naive at the CI level — the CI's LOWER bound
        // above zero ("at the 95% CI" per the pre-registration and this
        // bin's own doc; CORRECTION 2026-11-28, zed-7bab0f17@amo: the
        // original condition `mean > 0 && hi > 0` fired on a CI that
        // STRADDLES zero (this run: mean 0.0028, CI [−0.0107, +0.0127]),
        // printing NEGATIVE where the pre-registered gates read PASS — the
        // bug was verdict-LABELING only; the recorded NLL counts are
        // deterministic and unchanged. The i1250-B T7 adjudication record
        // carries the correction + the fold.
        "NEGATIVE"
    } else if half_cis.0 <= 0.0 && half_cis.2 <= 0.0 {
        "PASS"
    } else if half_cis.0 <= 0.0 {
        // point estimate at/below the bar, CI straddles it: UNDECIDED,
        // never PASS (the re-open bar is more windows, never a re-tune).
        "UNDECIDED"
    } else {
        "FAIL"
    };
    println!(
        "# T7 bar (aligned ≤ 0.5× naive at the 95% CI): verdict {verdict}\n#   (NO-OP = the angle falsifier; NEGATIVE = aligned > naive at the CI; \
UNDECIDED = the CI straddles the bar; FAIL = the point estimate misses the bar without the NEGATIVE CI)"
    );

    // ── the JSON artifact ──────────────────────────────────────────────
    let model_blake3 = blake3_file(&gguf_path)?;
    let mut sha_f = Sha256::new();
    sha_f.update(&fit_bytes);
    let fit_sha = hex(&sha_f.finalize());
    let artifact = serde_json::json!({
        "issue": "katgpt-rs/riir-infer i1250 arm B T5",
        "pre_registration": "katgpt-rs .issues/1250 §T5 (PINNED 2026-11-28)",
        "model": {
            "path": gguf_path.display().to_string(),
            "bytes": gguf_path.metadata().map(|m| m.len()).unwrap_or(0),
            "blake3": model_blake3,
        },
        "pair": { "producer_layer": producer_layer, "consumer_layer": consumer_layer },
        "corpora": {
            "fit": { "path": fit_path.display().to_string(), "tokens": fit_tokens.len(), "anchors_used": anchors, "sha256": fit_sha, "blake3": hex(blake3::hash(&fit_bytes).as_bytes()) },
            "eval": { "path": eval_path.display().to_string(), "tokens": eval_tokens.len(), "sha256": eval_sha, "sha256_pinned": PINNED_EVAL_SHA256, "blake3": hex(blake3::hash(&eval_bytes).as_bytes()) },
        },
        "fit": {
            "head_dim": fit.head_dim,
            "rotary_dim": fit.rotary_dim,
            "value_rank": fit.value_rank,
            "n_anchors": fit.n_anchors,
            "angles_norm": fit.angles_norm,
            "fit_residual": fit.fit_residual,
            "anchor_digest": hex(&fit.anchor_digest),
        },
        "eval_protocol": { "window": WINDOW, "warmup": WARMUP, "windows": n_windows, "resamples": resamples, "seed_digest": hex(&seed_digest) },
        "per_window": windows.iter().enumerate().map(|(w, win)| serde_json::json!({
            "window": w, "baseline": win[0], "naive": win[1], "aligned": win[2],
            "delta_naive": naive_delta[w], "delta_aligned": aligned_delta[w],
        })).collect::<Vec<_>>(),
        "deltas": { "naive": naive_d, "aligned": aligned_d },
        "bootstrap": {
            "half": { "mean": half_cis.0, "ci95_lo": half_cis.1, "ci95_hi": half_cis.2 },
            "neg": { "mean": neg_cis.0, "ci95_lo": neg_cis.1, "ci95_hi": neg_cis.2 },
        },
        "verdict": verdict,
        "box_state": box_state(),
    });
    let artifact_str = serde_json::to_string_pretty(&artifact)?;
    match &out {
        Some(p) => {
            std::fs::write(p, &artifact_str).with_context(|| format!("write {}", p.display()))?;
            println!(
                "# artifact written: {} ({} bytes)",
                p.display(),
                artifact_str.len()
            );
        }
        None => {
            println!("# artifact (--out unset; stdout only):\n{artifact_str}");
        }
    }

    // A verdict is a RECORDED measurement, not a process failure — any of
    // the five outcomes exits 0 with the verdict in the artifact.
    println!("\nVERDICT: {verdict}");
    Ok(())
}

// ── helpers ──────────────────────────────────────────────────────────────

/// SplitMix64 (the pre-registration names the generator).
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Bootstrap the mean of `aligned_delta_i − ratio·naive_delta_i` over
/// per-window pairs, resampling windows with replacement. Returns
/// (mean, ci_lo, ci_hi) — percentile 2.5 / 97.5 over the sorted resample
/// means (n = `resamples` ≥ 1000, so the percentile index keeps healthy
/// tail support — the n·p ≥ n-1 pathology cannot fire here).
fn bootstrap_ci(
    naive: &[f64],
    aligned: &[f64],
    ratio: f64,
    resamples: usize,
    rng: &mut SplitMix64,
) -> (f64, f64, f64) {
    let n = naive.len();
    let mut means = Vec::with_capacity(resamples);
    for _ in 0..resamples {
        let mut acc = 0.0f64;
        for _ in 0..n {
            // One uniform draw per resampled window — the multiply-shift
            // form (the 128-bit product's high word is an unbiased bounded
            // index, no modulo bias).
            let x = rng.next();
            let idx = ((x as u128 * n as u128) >> 64) as usize;
            acc += aligned[idx] - ratio * naive[idx];
        }
        means.push(acc / n as f64);
    }
    means.sort_by(|a, b| a.partial_cmp(b).expect("finite bootstrap means"));
    let lo = means[((0.025) * (resamples - 1) as f64) as usize];
    let hi = means[((0.975) * (resamples - 1) as f64) as usize];
    (means.iter().sum::<f64>() / resamples as f64, lo, hi)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Streamed BLAKE3 of a file (the league GGUF is multi-GiB — never a full
/// read into memory).
fn blake3_file(path: &Path) -> Result<String> {
    let mut hasher = blake3::Hasher::new();
    let mut f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    std::io::copy(&mut f, &mut hasher).context("blake3 stream")?;
    Ok(hex(hasher.finalize().as_bytes()))
}

/// Box-state provenance (the twt_perf_tg_pp shape: measurement-only shell
/// quotes; a failed read degrades to a marker, never fails the run), plus
/// the reflex `bench_preflight.sh` PROVENANCE line when the sibling
/// script exists.
fn box_state() -> serde_json::Value {
    let sh = |cmd: &str| -> String {
        std::process::Command::new("sh")
            .arg("-c")
            .arg(cmd)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_else(|| "(unavailable)".to_owned())
    };
    let preflight = Path::new("../riir-reflex/scripts/bench_preflight.sh");
    let provenance = if preflight.is_file() {
        sh(&format!(
            "sh {} 2>/dev/null | grep '^PROVENANCE:' || echo '(no PROVENANCE line)'",
            preflight.display()
        ))
    } else {
        "(bench_preflight.sh not found — the inline quote below is the provenance)".to_owned()
    };
    serde_json::json!({
        "utc": sh("date -u '+%Y-%m-%dT%H:%M:%SZ'"),
        "loadavg": sh("sysctl -n vm.loadavg"),
        "power": sh("pmset -g batt | grep -E 'AC Power|Battery Power'"),
        "powermode": sh("pmset -g | grep -i '^ powermode' || pmset -g | grep -i powermode"),
        "cores": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0),
        "bench_preflight_provenance": provenance,
    })
}
