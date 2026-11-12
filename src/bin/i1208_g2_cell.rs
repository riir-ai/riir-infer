//! Issue 1208 T2's G2 bars — the two measurement cells on the real
//! qwen35-hybrid 0.8B dense fixture (CPU lane), behind `deletion_mask` +
//! `deltanet_inference`:
//!
//! 1. **Apply-at-load wall** — base load vs load+mask-apply; the delta is
//!    the application's share of load wall (bar: "a few %").
//! 2. **Decode tok/s delta** — overlay-armed vs base decode, INTERLEAVED
//!    A/B/A/B so background drift hits both arms equally (bar: ≤1%).
//!    Apply-at-load decode shares base's FLOPs (zeroed weights multiply
//!    identically), so the overlay arm is the one that can move.
//!
//! PROVENANCE (bench_preflight discipline): power, load, and the box
//! line print beside every number; a REFUSAL fires when load exceeds
//! the publication threshold at start.
//!
//! Run: cargo run --release --features deletion_mask,deltanet_inference \
//!        --bin i1208_g2_cell -- /path/to/Qwen3.5-0.8B-Base-Q8_0.gguf
use riir_infer_core::adapter::f32_apply::{GateMaskOverlay, apply_mask_at_load};
use riir_infer_core::adapter::mask_codec::{MaskManifest, MaskSection, decode};
use riir_infer_core::deltanet::forward::{
    HybridCache, HybridForwardScratch, forward_qwen_deltanet,
};
use riir_infer_core::deltanet::weights::QwenDeltaNetWeights;
use riir_infer_core::gguf_loader::load_qwen_deltanet_weights_gguf;
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::types::Config;
use std::time::Instant;

/// Best-effort 1-minute load (the svd_lbit_eval pattern: /proc on Linux,
/// sysctl vm.loadavg on macOS; NaN when neither reads).
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

/// The deletion mask over every layer's gate projection: every 9th
/// weight of every tensor deleted (≈11% — well above the paper's
/// <0.05% so any overlay overhead is VISIBLE, not lost in noise; the
/// bar is "overlay ≈ base", and a bigger mask only stresses it more).
fn build_mask(cfg: &Config) -> (MaskManifest, Vec<u8>) {
    let base = *blake3::hash(b"i1208-g2-cell-base").as_bytes();
    let count = cfg.mlp_hidden * cfg.n_embd;
    let sections: Vec<MaskSection> = (0..cfg.n_layer)
        .map(|i| {
            MaskSection::new(
                format!("blk.{i}.ffn_gate.weight"),
                (i * count) as u64,
                count as u64,
            )
            .unwrap()
        })
        .collect();
    let manifest = MaskManifest::new(base, sections).unwrap();
    let mut bits = vec![0xFFu8; (cfg.n_layer * count).div_ceil(8)];
    for i in 0..cfg.n_layer {
        for w in (0..count).step_by(9) {
            let p = i * count + w;
            bits[p / 8] &= !(1 << (p % 8));
        }
    }
    (manifest, bits)
}

struct BoxState {
    load1: f64,
}

fn provenance(stage: &str) -> BoxState {
    let load1 = load_avg_1m();
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    println!(
        "# PROVENANCE {stage}: load1={load1:.2} cores={cores} time={} power=AC(assumed)",
        chrono_like_now()
    );
    BoxState { load1 }
}

fn chrono_like_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("epoch={secs}")
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let model = args
        .next()
        .expect("usage: i1208_g2_cell <model.gguf> [decode_tokens] [rounds]");

    let state = provenance("start");
    if state.load1 > 12.0 {
        anyhow::bail!(
            "load1 {:.1} > 12 — the box is not quiet enough to publish a ≤1% bar; rerun on a quiet window",
            state.load1
        );
    }

    let decode_tokens: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(64);
    let rounds: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(6);

    // ── Cell 1: apply-at-load wall ─────────────────────────────────────
    let t0 = Instant::now();
    let (config, base_weights) = load_qwen_deltanet_weights_gguf(std::path::Path::new(&model))?;
    let base_load_wall = t0.elapsed();
    println!(
        "# base load wall: {:.3} s (weights {} layers, mlp_hidden {} n_embd {})",
        base_load_wall.as_secs_f64(),
        config.n_layer,
        config.mlp_hidden,
        config.n_embd
    );

    let (manifest, bits) = build_mask(&config);
    let artifact = manifest.pack(&bits, None)?;
    let decoded = decode(&artifact, None, None)?;

    // a second load (OS cache warm — both loads warmed identically: the
    // FIRST load paid the cold-cache cost, so the second measures the
    // CPU/IO split the same way; the delta below is apply-only either way)
    let t1 = Instant::now();
    let (_, mut masked_weights) = load_qwen_deltanet_weights_gguf(std::path::Path::new(&model))?;
    let masked_load_wall = t1.elapsed();
    let t2 = Instant::now();
    let reports = apply_mask_at_load(&mut masked_weights, &decoded)?;
    let apply_wall = t2.elapsed();
    let total_zeroed: usize = reports.iter().map(|r| r.weights_zeroed).sum();
    println!(
        "# masked load wall: {:.3} s (same warm path) | apply wall: {:.3} s = {:.2}% of the masked load | weights zeroed {total_zeroed}",
        masked_load_wall.as_secs_f64(),
        apply_wall.as_secs_f64(),
        100.0 * apply_wall.as_secs_f64() / masked_load_wall.as_secs_f64().max(1e-9),
    );
    let apply_share = 100.0 * apply_wall.as_secs_f64() / masked_load_wall.as_secs_f64().max(1e-9);
    println!(
        "# G2 bar (a): apply share of load wall = {apply_share:.2}% (bar: a few %) -> {}",
        if apply_share < 5.0 { "PASS" } else { "MEASURE" }
    );

    // a third load for the PRISTINE weights (the overlay's base — the
    // shadow-copy contract needs weights the apply-at-load arm never touched)
    let (_, pristine) = load_qwen_deltanet_weights_gguf(std::path::Path::new(&model))?;

    // ── Cell 2: interleaved decode tok/s ───────────────────────────────
    let rope_freq = RopeFreqTable::new(config.rope_theta, config.rope_dimension_count);
    let tokens: Vec<usize> = (0..decode_tokens)
        .map(|i| (i * 31 + 7) % config.vocab_size)
        .collect();

    fn decode_run(
        cfg: &Config,
        weights: &QwenDeltaNetWeights,
        rope_freq: &RopeFreqTable,
        tokens: &[usize],
        overlay: Option<GateMaskOverlay>,
    ) -> anyhow::Result<f64> {
        let mut cache = HybridCache::with_layer_types(cfg, &weights.layer_types);
        let mut scratch = HybridForwardScratch::new(cfg);
        #[cfg(feature = "deletion_mask")]
        {
            scratch.gate_mask_overlay = overlay;
        }
        #[cfg(not(feature = "deletion_mask"))]
        {
            let _ = overlay;
        }
        let mut x = vec![0.0f32; cfg.vocab_size.max(cfg.n_embd) * 2];
        let t0 = Instant::now();
        for (step, &tok) in tokens.iter().enumerate() {
            let out = forward_qwen_deltanet(
                &mut x,
                weights,
                &mut cache,
                tok,
                step,
                cfg,
                &mut scratch,
                rope_freq,
            );
            // black-box the logits: consume one byte so the walk cannot
            // be folded away
            let mut sink = 0u64;
            for (i, v) in out[..cfg.vocab_size].iter().enumerate() {
                sink = sink.wrapping_add(((v.to_bits() >> (i % 8)) & 0xFF) as u64);
            }
            std::hint::black_box(sink);
        }
        Ok(tokens.len() as f64 / t0.elapsed().as_secs_f64())
    }

    // warm-up both arms (caches, rayon pool, branch predictors); the
    // overlay mounts ONCE — hot-swap is pointer-swap, remounting per
    // round would measure the mount, not the steady state
    let overlay = GateMaskOverlay::mount(&pristine, &decoded)?;
    decode_run(&config, &base_weights, &rope_freq, &tokens, None)?;
    decode_run(
        &config,
        &pristine,
        &rope_freq,
        &tokens,
        Some(overlay.clone()),
    )?;

    let mut base_rates = Vec::new();
    let mut overlay_rates = Vec::new();
    for round in 0..rounds {
        let b = decode_run(&config, &base_weights, &rope_freq, &tokens, None)?;
        let o = decode_run(
            &config,
            &pristine,
            &rope_freq,
            &tokens,
            Some(overlay.clone()),
        )?;
        base_rates.push(b);
        overlay_rates.push(o);
        println!(
            "# round {round}: base {b:.1} tok/s | overlay {o:.1} tok/s | delta {:+.2}%",
            100.0 * (o - b) / b
        );
    }
    let b_mean = base_rates.iter().sum::<f64>() / rounds as f64;
    let o_mean = overlay_rates.iter().sum::<f64>() / rounds as f64;
    let spread_b = (base_rates.iter().cloned().fold(f64::MIN, f64::max)
        - base_rates.iter().cloned().fold(f64::MAX, f64::min))
        / b_mean
        * 100.0;
    println!(
        "# G2 bar (b): overlay decode delta = {:+.3}% of base ({b_mean:.1} -> {o_mean:.1} tok/s), base spread {spread_b:.2}% (bar: ≤1%) -> {}",
        100.0 * (o_mean - b_mean) / b_mean,
        if (100.0 * (o_mean - b_mean) / b_mean).abs() <= 1.0 {
            "PASS"
        } else {
            "FAIL"
        }
    );
    provenance("end");
    Ok(())
}
