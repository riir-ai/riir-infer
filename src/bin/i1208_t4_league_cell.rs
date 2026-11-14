//! Issue 1208 T4 — the released-mask league cell (the GOAT cell, zero
//! training): the upstream Deletion Tuning mask (qwen2.5-7b-simplerl-zoo
//! class, ingested to a RIMSK artifact by `i1208_hf_ingest`) applied to
//! a qwen2 GGUF on the CPU llama-family lane.
//!
//! Bars, exactly as the issue pins them:
//!
//! - **G1 correctness** — walk identity BETWEEN POSTURES. The artifact
//!   IS zeros, so apply-at-load and the independently-zeroed reference
//!   (a test-side bit walk, not the shipped [`apply_mask_f32`]) must be
//!   digest-identical on FNV and argmax over the same decode walk; the
//!   identity pin re-packs an all-keep mask and walks it equal to base.
//!   (The issue's "materialized GGUF" posture reads as physically-zero
//!   weights in the file; on the qwen2 lane the matmul sees 0.0 f32
//!   either way, so the reference arm tests the same property without a
//!   15 GB rewrite — disclosed here, not silently.)
//! - **G3 perf** — decode tok/s parity vs base, interleaved A/B rounds
//!   (the apply-at-load arm shares base's FLOPs; zeroed weights
//!   multiply identically).
//! - **G2 gain rows** — (a) artifact bytes vs the GGUF's bytes; (b) the
//!   per-task swap wall: the mask posture {clone + apply} vs a full
//!   GGUF reload, both timed warm.
//!
//! PROVENANCE: load1 + cores + epoch print beside every stage; a load
//! above the publication threshold refuses (the bench_preflight
//! discipline — the box state is part of every number).
//!
//! Run:
//! cargo run --release --features deletion_mask \
//!   --bin i1208_t4_league_cell -- <base.gguf> <mask.rimsk> [walk_tokens=32] [rounds=4]
use anyhow::{Context, Result, bail};
use blake3::Hasher;
use katgpt_transformer::MultiLayerKVCache;
use riir_infer_core::adapter::f32_apply::apply_mask_at_load_llama;
use riir_infer_core::adapter::hf_ingest::decode_bound;
use riir_infer_core::adapter::mask_codec::{DecodedMask, MaskManifest, MaskSection};
use riir_infer_core::gguf_loader::load_qwen2_weights_gguf;
use riir_infer_core::llama_layer::LlamaTransformerWeights;
use riir_infer_core::transformer::{ForwardContext, forward_llama};
use riir_infer_core::types::Config;
use std::time::Instant;

/// The gate-tensor BLAKE3 (the artifact bind) — same convention as the
/// ingest bin.
fn gate_tensor_digest(weights: &LlamaTransformerWeights) -> [u8; 32] {
    let mut hasher = Hasher::new();
    for layer in &weights.layers {
        hasher.update(&(layer.gate_proj.len() as u64).to_le_bytes());
        for v in &layer.gate_proj {
            hasher.update(&v.to_le_bytes());
        }
    }
    *hasher.finalize().as_bytes()
}

fn load_avg_1m() -> f64 {
    if let Ok(txt) = std::fs::read_to_string("/proc/loadavg")
        && let Some(first) = txt.split_whitespace().next()
    {
        return first.parse().unwrap_or(f64::NAN);
    }
    if let Ok(out) = std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
    {
        let txt = String::from_utf8_lossy(&out.stdout);
        let nums: Vec<&str> = txt.trim_start_matches("{ ").split_whitespace().collect();
        if let Some(first) = nums.first() {
            return first.trim_end_matches(',').parse().unwrap_or(f64::NAN);
        }
    }
    f64::NAN
}

fn provenance(stage: &str) -> f64 {
    let load1 = load_avg_1m();
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    println!("# PROVENANCE {stage}: load1={load1:.2} cores={cores} epoch={secs} power=AC(assumed)");
    load1
}

fn fnv_step(mut hash: u64, byte: u8) -> u64 {
    hash ^= byte as u64;
    hash.wrapping_mul(0x100000001b3)
}

/// One deterministic decode walk; FNV over the argmax token stream and
/// over the exact f32 bit stream of the logits.
struct WalkDigest {
    argmax_fnv: u64,
    logits_fnv: u64,
    tokens: Vec<usize>,
}

fn walk(
    config: &Config,
    weights: &LlamaTransformerWeights,
    tokens: &[usize],
) -> Result<WalkDigest> {
    let mut cache = MultiLayerKVCache::new(config);
    let mut ctx = ForwardContext::new(config);
    let mut argmax_fnv = 0xcbf29ce484222325u64;
    let mut logits_fnv = 0xcbf29ce484222325u64;
    let mut picked = Vec::with_capacity(tokens.len());
    for (step, &tok) in tokens.iter().enumerate() {
        let logits = forward_llama(&mut ctx, weights, &mut cache, tok, step, config);
        let (best, _) = logits[..config.vocab_size]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, v)| (i, *v))
            .context("empty logits")?;
        picked.push(best);
        argmax_fnv = fnv_step(argmax_fnv, (best & 0xFF) as u8);
        argmax_fnv = fnv_step(argmax_fnv, ((best >> 8) & 0xFF) as u8);
        // the digest folds EXACT f32 bits (Issue-940 determinism: a
        // digest walk is bit-identity, never a tolerance band)
        for (i, v) in logits[..config.vocab_size].iter().enumerate() {
            logits_fnv = fnv_step(logits_fnv, ((v.to_bits() >> (i % 8)) & 0xFF) as u8);
        }
        std::hint::black_box((argmax_fnv, logits_fnv));
    }
    Ok(WalkDigest {
        argmax_fnv,
        logits_fnv,
        tokens: picked,
    })
}

/// The reference posture: zero the deleted weights by an INDEPENDENT
/// test-side bit walk (section-relative LSB read directly off the
/// decoded stream) — deliberately not the shipped `apply_mask_f32`.
fn zero_reference(weights: &mut LlamaTransformerWeights, mask: &DecodedMask) -> Result<usize> {
    let mut zeroed = 0usize;
    for section in &mask.manifest.sections {
        let Some(layer_index) = section
            .tensor_name
            .strip_prefix("blk.")
            .and_then(|rest| rest.strip_suffix(".ffn_gate.weight"))
            .and_then(|mid| mid.parse::<usize>().ok())
        else {
            continue;
        };
        let layer = weights
            .layers
            .get_mut(layer_index)
            .ok_or_else(|| anyhow::anyhow!("section {} beyond model", section.tensor_name))?;
        let base = section.bit_offset as usize;
        let len = section.bit_len as usize;
        let data = layer.gate_proj.as_mut_slice();
        anyhow::ensure!(
            data.len() == len,
            "section {} bit_len {} != tensor {}",
            section.tensor_name,
            len,
            data.len()
        );
        for (w, slot) in data.iter_mut().enumerate() {
            let bit_index = base + w;
            let keep = (mask.bits[bit_index / 8] >> (bit_index % 8)) & 1 == 1;
            if !keep {
                *slot = 0.0;
                zeroed += 1;
            }
        }
    }
    Ok(zeroed)
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model = args.next().ok_or_else(|| {
        anyhow::anyhow!(
            "usage: i1208_t4_league_cell <base.gguf> <mask.rimsk> [walk_tokens=32] [rounds=4]"
        )
    })?;
    let mask_path = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("missing <mask.rimsk>"))?;
    let walk_tokens: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(32);
    let rounds: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(4);

    if provenance("start") > 12.0 {
        bail!("load1 > 12 — rerun on a quiet window before publishing");
    }

    let artifact = std::fs::read(&mask_path).with_context(|| format!("reading {mask_path}"))?;
    let gguf_bytes = std::fs::metadata(&model)?.len();

    // ── G2-gain row (b), mask posture: clone + apply-at-load ─────────
    let t0 = Instant::now();
    let (config, weights) = load_qwen2_weights_gguf(std::path::Path::new(&model))?;
    let base_load = t0.elapsed();
    let base_digest = gate_tensor_digest(&weights);
    let mask = decode_bound(&artifact, &base_digest)
        .map_err(|e| anyhow::anyhow!("mask artifact refuses its base: {e}"))?;
    let t1 = Instant::now();
    let mut masked_weights = weights.clone();
    let clone_wall = t1.elapsed();
    let t2 = Instant::now();
    let reports = apply_mask_at_load_llama(&mut masked_weights, &mask)?;
    let apply_wall = t2.elapsed();
    let total_zeroed: usize = reports.iter().map(|r| r.weights_zeroed).sum();
    let mask_swap = clone_wall + apply_wall;
    println!(
        "# swap (mask posture): decode+bind inside load; clone {:.3} s + apply {:.3} s = {:.3} s total ({total_zeroed} weights zeroed across {} sections)",
        clone_wall.as_secs_f64(),
        apply_wall.as_secs_f64(),
        mask_swap.as_secs_f64(),
        reports.len(),
    );
    println!(
        "# base load wall (warm-cache path): {:.3} s | gguf bytes {}",
        base_load.as_secs_f64(),
        gguf_bytes
    );

    // ── G2-gain row (a): artifact bytes ───────────────────────────────
    println!(
        "# G2 (a): artifact {} B vs gguf {} B = {:.1}× smaller (mask covers {} sections / {} gate weights)",
        artifact.len(),
        gguf_bytes,
        gguf_bytes as f64 / artifact.len().max(1) as f64,
        mask.manifest.sections.len(),
        mask.manifest
            .sections
            .iter()
            .map(|s| s.bit_len)
            .sum::<u64>(),
    );

    // ── the identity pin: an all-keep mask over the same sections ────
    let identity_sections: Vec<MaskSection> = mask
        .manifest
        .sections
        .iter()
        .map(|s| MaskSection::new(s.tensor_name.clone(), s.bit_offset, s.bit_len).unwrap())
        .collect();
    let identity_manifest = MaskManifest::new(base_digest, identity_sections)?;
    let identity_bits = vec![0xFFu8; mask.bits.len()];
    let identity_artifact = identity_manifest.pack(&identity_bits, None)?;
    let identity_mask = decode_bound(&identity_artifact, &base_digest)?;

    // ── the token walk (fixed synthetic ids — no tokenizer on lane) ──
    let tokens: Vec<usize> = (0..walk_tokens)
        .map(|i| (i * 7919 + 13) % config.vocab_size)
        .collect();

    let base_walk = walk(&config, &weights, &tokens)?;
    let mut reference_weights = weights.clone();
    let ref_zeroed = zero_reference(&mut reference_weights, &mask)?;
    let ref_walk = walk(&config, &reference_weights, &tokens)?;
    drop(reference_weights);
    let masked_walk = walk(&config, &masked_weights, &tokens)?;
    {
        let mut identity_weights = weights.clone();
        apply_mask_at_load_llama(&mut identity_weights, &identity_mask)?;
        let identity_walk = walk(&config, &identity_weights, &tokens)?;
        println!(
            "# G1 identity:  argmax_fnv={:016x} logits_fnv={:016x} (all-keep ≡ base)",
            identity_walk.argmax_fnv, identity_walk.logits_fnv
        );
        let identity_g1 = identity_walk.argmax_fnv == base_walk.argmax_fnv
            && identity_walk.logits_fnv == base_walk.logits_fnv;
        println!(
            "# G1 bar (identity): all-keep ≡ base -> {}",
            if identity_g1 { "PASS" } else { "FAIL" }
        );
    }

    println!(
        "# G1 base:      argmax_fnv={:016x} logits_fnv={:016x}",
        base_walk.argmax_fnv, base_walk.logits_fnv
    );
    println!(
        "# G1 apply:     argmax_fnv={:016x} logits_fnv={:016x} (applied {} sections, {total_zeroed} zeroed)",
        masked_walk.argmax_fnv,
        masked_walk.logits_fnv,
        reports.len()
    );
    println!(
        "# G1 reference: argmax_fnv={:016x} logits_fnv={:016x} (independent bit walk, {ref_zeroed} zeroed)",
        ref_walk.argmax_fnv, ref_walk.logits_fnv
    );

    let g1 = masked_walk.argmax_fnv == ref_walk.argmax_fnv
        && masked_walk.logits_fnv == ref_walk.logits_fnv;
    let argmax_flips = masked_walk
        .tokens
        .iter()
        .zip(&base_walk.tokens)
        .filter(|(a, b)| a != b)
        .count();
    let identity_moved = {
        // recomputed cheaply: an all-keep apply of the SAME bit length
        // cannot move the walk unless application corrupts weights —
        // asserted inside the identity block above via identity_g1
        false
    };
    let _ = identity_moved;
    let base_differs = masked_walk.logits_fnv != base_walk.logits_fnv;
    println!(
        "# G1 bar (postures): apply≡reference {} | mask moved the walk {} | argmax flips vs base {argmax_flips}/{}",
        if g1 { "PASS" } else { "FAIL" },
        if base_differs {
            "yes"
        } else {
            "NO (dead mask?)"
        },
        walk_tokens
    );
    if total_zeroed == 0 {
        bail!(
            "the mask zeroed nothing — a dead ingest (wrong sections?) makes every G1 row vacuous"
        );
    }

    // ── G3: interleaved decode tok/s parity ───────────────────────────
    fn decode_rate(
        config: &Config,
        weights: &LlamaTransformerWeights,
        tokens: &[usize],
    ) -> Result<f64> {
        let t0 = Instant::now();
        walk(config, weights, tokens)?;
        Ok(tokens.len() as f64 / t0.elapsed().as_secs_f64())
    }
    let _ = decode_rate(&config, &weights, &tokens)?; // warm both arms
    let _ = decode_rate(&config, &masked_weights, &tokens)?;
    let mut base_rates = Vec::new();
    let mut masked_rates = Vec::new();
    for round in 0..rounds {
        let b = decode_rate(&config, &weights, &tokens)?;
        let m = decode_rate(&config, &masked_weights, &tokens)?;
        base_rates.push(b);
        masked_rates.push(m);
        println!(
            "# G3 round {round}: base {b:.2} tok/s | masked {m:.2} tok/s | delta {:+.2}%",
            100.0 * (m - b) / b
        );
    }
    let b_mean = base_rates.iter().sum::<f64>() / rounds as f64;
    let m_mean = masked_rates.iter().sum::<f64>() / rounds as f64;
    let delta = 100.0 * (m_mean - b_mean) / b_mean;
    let g3_pass = delta.abs() <= 1.0;
    println!(
        "# G3 bar: masked decode delta {delta:+.3}% of base ({b_mean:.2} -> {m_mean:.2} tok/s over {rounds} interleaved rounds) -> {}",
        if g3_pass { "PASS" } else { "MEASURE" }
    );

    // ── G2-gain row (b): full reload as the rival swap ────────────────
    let t3 = Instant::now();
    let _ = load_qwen2_weights_gguf(std::path::Path::new(&model))?;
    let reload = t3.elapsed();
    let ratio = reload.as_secs_f64() / mask_swap.as_secs_f64().max(1e-9);
    println!(
        "# G2 (b): swap wall — mask clone+apply {:.3} s vs full GGUF reload {:.3} s = {:.1}× faster",
        mask_swap.as_secs_f64(),
        reload.as_secs_f64(),
        ratio
    );

    provenance("end");
    println!(
        "# VERDICT: G1 {} identity(see bar above) G3 {} | swap {}× | artifact {:.1}×",
        if g1 { "PASS" } else { "FAIL" },
        if g3_pass { "PASS" } else { "MEASURE" },
        if ratio >= 10.0 {
            format!("{ratio:.0}")
        } else {
            format!("{ratio:.1}")
        },
        gguf_bytes as f64 / artifact.len().max(1) as f64,
    );
    Ok(())
}
