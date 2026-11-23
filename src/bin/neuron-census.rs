//! neuron-census — the Issue-1241 census front (the Fermion distill,
//! katgpt-rs Research 684).
//!
//! Postures:
//! - `--self-test` — runs the G1 fixture gates in-process (the
//!   planted-edit recovery battery + the parallel/serial twin gates)
//!   and prints the receipt. The canary vehicle — no model file
//!   needed.
//! - `--mechanics-check --model <gguf>` — the push/protocol gate on a
//!   REAL checkpoint: band ≡ full bit-identity, identity KL ≡ 0
//!   exactly (both postures), dose monotonicity on a live column,
//!   weights-digest canary.
//! - `--model <gguf> [--out DIR] [--layers 0-8,16-23] [--threads N]
//!   [--chunk K] [--limit N] [--resume]` — the T2 sweep cell over the
//!   activation-scale push + the layer-band prefix cache (the KL
//!   observable, 8 probes). Writes the state dir (cursor + rows
//!   ledger) and, at completion, the canonical artifact with its
//!   BLAKE3 trailer. `--limit N` caps every layer's candidates (the
//!   smoke posture — never resumed).
//!
//! Reader protection: `required-features = ["neuron_census"]` — a
//! default build never compiles this bin (the green-zero law).

use riir_infer_core::neuron_census::{
    CensusConfig, CensusCursor, CensusVerdict, mint_artifact, run_sweep_par, verify_artifact,
};
use riir_infer_core::neuron_census_gguf::{CensusShared, mechanics_check};
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let mut self_test = false;
    let mut mechanics = false;
    let mut model: Option<String> = None;
    let mut out_dir: Option<String> = None;
    let mut layers_spec: Option<String> = None;
    let mut threads: Option<usize> = None;
    let mut chunk: Option<usize> = None;
    let mut limit: Option<usize> = None;
    let mut resume = false;
    let mut abs_floor: Option<f64> = None;
    let mut band_ratio: Option<f64> = None;
    let mut probes: Vec<String> = Vec::new();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--self-test" => self_test = true,
            "--mechanics-check" => mechanics = true,
            "--model" => model = args.next(),
            "--out" => out_dir = args.next(),
            "--layers" => layers_spec = args.next(),
            "--threads" => threads = args.next().and_then(|v| v.parse().ok()),
            "--chunk" => chunk = args.next().and_then(|v| v.parse().ok()),
            "--limit" => limit = args.next().and_then(|v| v.parse().ok()),
            "--resume" => resume = true,
            "--abs-floor" => abs_floor = args.next().and_then(|v| v.parse().ok()),
            "--band-ratio" => band_ratio = args.next().and_then(|v| v.parse().ok()),
            "--probe" => {
                if let Some(p) = args.next() {
                    probes.push(p);
                }
            }
            "--help" | "-h" => {
                print_help();
                return;
            }
            other => {
                eprintln!("neuron-census: unknown argument {other:?} (see --help)");
                std::process::exit(2);
            }
        }
    }

    if self_test {
        run_self_test();
        return;
    }

    if mechanics {
        let Some(model_path) = model else {
            eprintln!("neuron-census: --mechanics-check needs --model <gguf>");
            std::process::exit(2);
        };
        match mechanics_check(std::path::Path::new(&model_path), &probes) {
            Ok(line) => println!("neuron-census {line}"),
            Err(e) => {
                eprintln!("neuron-census mechanics FAIL: {e:#}");
                std::process::exit(1);
            }
        }
        return;
    }

    if let Some(model_path) = model {
        if let Err(e) = run_sweep_cell(
            std::path::Path::new(&model_path),
            out_dir.as_deref(),
            layers_spec.as_deref(),
            threads,
            chunk,
            limit,
            resume,
            abs_floor,
            band_ratio,
            &probes,
        ) {
            eprintln!("neuron-census sweep FAIL: {e:#}");
            std::process::exit(1);
        }
        return;
    }

    print_help();
    std::process::exit(2);
}

/// Parse `--layers "0-8,16-23"` (inclusive ranges, comma-separated) —
/// `None` = every layer. Overlap refuses loud (two rows per candidate
/// would double-sweep).
fn parse_layers(spec: &str, n_layers: usize) -> anyhow::Result<Vec<usize>> {
    let mut selected = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        let (a, b) = match part.split_once('-') {
            Some((a, b)) => (a.trim().parse::<usize>()?, b.trim().parse::<usize>()?),
            None => {
                let v: usize = part.parse()?;
                (v, v)
            }
        };
        anyhow::ensure!(a <= b, "layer range {a}-{b} descends");
        for l in a..=b {
            anyhow::ensure!(
                l < n_layers,
                "layer {l} out of range (the checkpoint has {n_layers})"
            );
            anyhow::ensure!(!selected.contains(&l), "layer {l} selected twice");
            selected.push(l);
        }
    }
    Ok(selected)
}

#[allow(clippy::too_many_arguments)]
fn run_sweep_cell(
    model_path: &std::path::Path,
    out_dir: Option<&str>,
    layers_spec: Option<&str>,
    threads: Option<usize>,
    chunk: Option<usize>,
    limit: Option<usize>,
    resume: bool,
    abs_floor: Option<f64>,
    band_ratio: Option<f64>,
    probes: &[String],
) -> anyhow::Result<()> {
    if resume && limit.is_some() {
        anyhow::bail!("--resume with --limit is refused — a limited run is a smoke, never resumed");
    }
    let cfg = CensusConfig {
        abs_floor: abs_floor.unwrap_or(CensusConfig::default().abs_floor),
        band_ratio: band_ratio.unwrap_or(CensusConfig::default().band_ratio),
        ..CensusConfig::default()
    };
    let out = out_dir.map_or_else(
        || std::path::PathBuf::from(".census/neuron_census"),
        std::path::PathBuf::from,
    );

    let t0 = Instant::now();
    let shared = CensusShared::from_gguf(model_path, probes)?;
    let load_s = t0.elapsed().as_secs_f64();
    let n_layers = shared.census_layers().len();

    let selected = match layers_spec {
        Some(spec) => parse_layers(spec, n_layers)?,
        None => (0..n_layers).collect(),
    };
    let filter = if selected.len() == n_layers {
        None
    } else {
        // Contiguous ascending run → one Range; otherwise refuse (the
        // session maps a Range; a scattered pick is a caller bug).
        for w in selected.windows(2) {
            anyhow::ensure!(
                w[1] == w[0] + 1,
                "--layers must be one contiguous range (got {:?})",
                selected
            );
        }
        Some(selected[0]..selected[selected.len() - 1] + 1)
    };

    let workers = threads.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    });
    let chunk = chunk.unwrap_or(16);
    let total: usize = shared
        .census_layers()
        .iter()
        .filter(|l| selected.contains(&(l.layer as usize)))
        .map(|l| limit.map_or(l.neurons, |lim| l.neurons.min(lim)))
        .sum();
    println!(
        "[census] load {load_s:.1}s · {} layers · {} candidates · {workers} threads · chunk {chunk}{}{}",
        selected.len(),
        total,
        if limit.is_some() {
            " · LIMIT (smoke)"
        } else {
            ""
        },
        if resume { " · RESUME" } else { "" },
    );

    let mut models: Vec<Box<dyn riir_infer_core::neuron_census::CensusModel>> =
        Vec::with_capacity(workers);
    for _ in 0..workers {
        models.push(Box::new(shared.session(filter.clone())?));
    }

    let t1 = Instant::now();
    let mut last_print = Instant::now();
    let mut done = 0usize;
    let sweep = run_sweep_par(
        models,
        chunk,
        &cfg,
        &out,
        resume,
        limit,
        &mut |_li, width, cursor: CensusCursor| {
            // One hook call = one row committed (in serial candidate order),
            // so the bin counts its own exact progress; the cursor names the
            // layer/column the commit landed at.
            done += 1;
            if last_print.elapsed().as_secs_f64() >= 1.0 {
                let dt = last_print.elapsed().as_secs_f64();
                let rate = (done as f64) / (t1.elapsed().as_secs_f64());
                let remaining = total.saturating_sub(done);
                let eta = if rate > 0.0 {
                    remaining as f64 / rate
                } else {
                    f64::NAN
                };
                println!(
                    "[census] L{} col {}/{} · {done}/{total} · {rate:.2}/s · eta {:.0}m",
                    cursor.layer_index, cursor.next_column, width, eta,
                );
                last_print = Instant::now();
                let _ = dt;
            }
        },
    )?;
    let elapsed = t1.elapsed().as_secs_f64();
    let switches = sweep
        .rows
        .iter()
        .filter(|r| r.verdict == CensusVerdict::SwitchFound)
        .count();
    let artifact = mint_artifact(&sweep, &shared.weights_digest(), &cfg);
    let payload_len = verify_artifact(&artifact)
        .ok_or_else(|| anyhow::anyhow!("minted artifact failed its own BLAKE3 trailer"))?;
    let artifact_path = out.join("neuron_census.canonical");
    std::fs::write(&artifact_path, &artifact)?;
    let digest_hex: String = artifact[artifact.len() - 32..]
        .iter()
        .take(8)
        .map(|b| format!("{b:02x}"))
        .collect();

    println!(
        "[census] done: {} rows ({elapsed:.0}s, {:.2}/s effective) · SwitchFound {switches} · \
         artifact {} ({} B, payload {payload_len}, trailer {})",
        sweep.rows.len(),
        sweep.rows.len() as f64 / elapsed,
        artifact_path.display(),
        artifact.len(),
        digest_hex,
    );

    let mut ranked: Vec<_> = sweep.rows.iter().collect();
    ranked.sort_by(|a, b| {
        b.effect
            .partial_cmp(&a.effect)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    println!("[census] top-10 by effect (layer:column dose verdict effect band_mean):");
    for r in ranked.iter().take(10) {
        println!(
            "  L{:<2}:{:<5} dose {:.1} {:?} effect {:.6} band {:.6}",
            r.layer, r.column, r.best_dose, r.verdict, r.effect, r.band_mean,
        );
    }
    Ok(())
}

fn run_self_test() {
    let cfg = CensusConfig::default();
    match riir_infer_core::neuron_census_fixture::g1_receipt(&cfg) {
        Ok(r) => {
            println!("neuron-census G1 PASS (pre-registered bars: max(floor, 28×band))");
            println!("  planted column recovered rank-1: {}", r.top_column);
            println!("  effect        = {:.6}", r.effect);
            println!("  band mean     = {:.6}", r.band_mean);
            println!("  ratio         = {:.2} (bar 28.00)", r.ratio);
            println!("  false fires   = {} (pin 0)", r.false_fires);
            println!("  artifact len  = {} B (trailer verified)", r.artifact_len);
            println!("  leaderboard   = {:?}", r.leaderboard);
            println!(
                "  fixture seed  = {:#018x}",
                riir_infer_core::neuron_census_fixture::FIXTURE_SEED
            );
        }
        Err(e) => {
            eprintln!("neuron-census G1 FAIL: {e}");
            std::process::exit(1);
        }
    }
    match riir_infer_core::neuron_census_fixture::par_receipt(&cfg) {
        Ok(r) => {
            println!("neuron-census PAR PASS (parallel/serial twin gates)");
            println!("  rows equal serial bit-for-bit: {} rows", r.rows);
            println!(
                "  chunked resume artifact identical: {}",
                r.resume_identical
            );
            println!("  workers × chunk = {} × {}", r.workers, r.chunk);
        }
        Err(e) => {
            eprintln!("neuron-census PAR FAIL: {e}");
            std::process::exit(1);
        }
    }
}

fn print_help() {
    println!(
        "neuron-census — the Issue-1241 offline interventional neuron census\n\
         \n\
         USAGE:\n\
         \x20 neuron-census --self-test                        run the G1 + parallel/serial fixture gates\n\
         \x20 neuron-census --mechanics-check --model <gguf>   the protocol gate on a real checkpoint\n\
         \x20        [--probe \"text\"] ...                      (band ≡ full bits, identity KL ≡ 0, dose\n\
         \x20                                                  monotone, digest canary)\n\
         \x20 neuron-census --model <gguf> [--out DIR]         the T2 sweep cell (KL observable,\n\
         \x20        [--layers 0-8,16-23] [--threads N]        layer-band prefix cache, parallel workers;\n\
         \x20        [--chunk K] [--limit N] [--resume]        state dir + canonical artifact with BLAKE3\n\
         \x20        [--abs-floor F] [--band-ratio R]          trailer; --limit is the smoke posture)\n\
         \n\
         The lane is opt-in (`neuron_census` feature), offline by construction, and never\n\
         touches a serving path."
    );
}
