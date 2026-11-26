//! The trigger-vote front (riir-infer Issue 1251 T1; katgpt-rs
//! Research 693 Part 2 App. C.4 — the checkpoint-ingest backdoor
//! screen, re-derived on our lanes; the published class is cited,
//! never claimed).
//!
//! Reader protection: `required-features = ["readout_screen"]` — a
//! default build never compiles this bin (the green-zero law).
//!
//! ```text
//! trigger-vote --self-test
//! trigger-vote --base <base.gguf> --ft <finetune.gguf> \
//!              [--out <artifact.vote1>] [--top N]
//! trigger-vote --plant <t2a|t2b> --base <base.gguf> --out <planted.gguf> \
//!              [--layer N] [--triggers N] [--fan N] [--noise N] [--seed N]
//! trigger-vote --cell <planted.gguf> --base <base.gguf> [--t2a-passed]
//! trigger-vote --cell <organism.safetensors> --base <base.safetensors> \
//!              --cell-st --cell-triggers <ids> [--out <artifact.vote1>]
//! ```
//!
//! `--plant` builds the Issue-1251 T2 fixture: plans (analysis + the
//! pre-write self-falsifier printed BEFORE any byte moves), materializes
//! the planted GGUF copy + the PLNT1 manifest beside it. `--cell` runs
//! the T3 recovery cell over a planted pair: detector → baselines →
//! pinned bar, printing the pre-registered verdict. Report-only — the
//! exit code never encodes the verdict.

use riir_infer_core::gguf_loader::GgufFile;
use riir_infer_core::readout_screen::hex;
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_core::trigger_vote::{
    GgufVoteSource, Verdict, VoteTensors, artifact_digest, build_artifact, render_report,
    trigger_vote,
};

use riir_infer_core::trigger_vote_fixture;
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let mut self_test = false;
    let mut base: Option<String> = None;
    let mut ft: Option<String> = None;
    let mut out: Option<String> = None;
    let mut top: usize = 30;
    let mut plant: Option<riir_infer_core::trigger_vote_cell::PlantMode> = None;
    let mut cell: Option<String> = None;
    let mut layer = riir_infer_core::trigger_vote_cell::LAYER_MIDDLE;
    let mut triggers = riir_infer_core::trigger_vote_cell::TRIGGERS_MIN;
    let mut fan = riir_infer_core::trigger_vote_cell::FAN_PER_TRIGGER;
    let mut noise = riir_infer_core::trigger_vote_cell::NOISE_NEURONS;
    let mut seed = riir_infer_core::trigger_vote_cell::DEFAULT_SEED;
    let mut t2a_passed = false;
    let mut cell_st = false;
    let mut cell_triggers: Option<Vec<u32>> = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--self-test" => self_test = true,
            "--base" => base = args.next(),
            "--ft" => ft = args.next(),
            "--out" => out = args.next(),
            "--plant" => match args
                .next()
                .as_deref()
                .and_then(|v| riir_infer_core::trigger_vote_cell::PlantMode::parse(v).ok())
            {
                Some(m) => plant = Some(m),
                None => {
                    eprintln!("trigger-vote: --plant needs circular|t2a or upstream|t2b");
                    std::process::exit(2);
                }
            },
            "--cell" => cell = args.next(),
            "--layer" => match args.next().and_then(|v| v.parse::<usize>().ok()) {
                Some(n) => layer = n,
                _ => {
                    eprintln!("trigger-vote: --layer needs an integer");
                    std::process::exit(2);
                }
            },
            "--triggers" => match args.next().and_then(|v| v.parse::<usize>().ok()) {
                Some(n) if n > 0 => triggers = n,
                _ => {
                    eprintln!("trigger-vote: --triggers needs a positive integer");
                    std::process::exit(2);
                }
            },
            "--fan" => match args.next().and_then(|v| v.parse::<usize>().ok()) {
                Some(n) if n > 0 => fan = n,
                _ => {
                    eprintln!("trigger-vote: --fan needs a positive integer");
                    std::process::exit(2);
                }
            },
            "--noise" => match args.next().and_then(|v| v.parse::<usize>().ok()) {
                Some(n) => noise = n,
                _ => {
                    eprintln!("trigger-vote: --noise needs an integer");
                    std::process::exit(2);
                }
            },
            "--seed" => match args.next().and_then(|v| v.parse::<u64>().ok()) {
                Some(n) => seed = n,
                _ => {
                    eprintln!("trigger-vote: --seed needs a u64");
                    std::process::exit(2);
                }
            },
            "--t2a-passed" => t2a_passed = true,
            "--cell-st" => cell_st = true,
            "--cell-triggers" => {
                let Some(v) = args.next() else {
                    eprintln!("trigger-vote: --cell-triggers needs a comma-separated id list");
                    std::process::exit(2);
                };
                match v
                    .split(',')
                    .map(|x| x.trim().parse::<u32>())
                    .collect::<Result<Vec<_>, _>>()
                {
                    Ok(list) if !list.is_empty() => cell_triggers = Some(list),
                    _ => {
                        eprintln!("trigger-vote: --cell-triggers needs ids like 7147,32963");
                        std::process::exit(2);
                    }
                }
            }
            "--top" => match args.next().and_then(|v| v.parse::<usize>().ok()) {
                Some(n) if n > 0 => top = n,
                _ => {
                    eprintln!("trigger-vote: --top needs a positive integer");
                    std::process::exit(2);
                }
            },
            "--help" | "-h" => {
                print_help();
                return;
            }
            other => {
                eprintln!("trigger-vote: unknown argument {other:?} (see --help)");
                std::process::exit(2);
            }
        }
    }

    if self_test {
        match trigger_vote_fixture::run_all().and_then(|mut r| {
            r.extend(riir_infer_core::trigger_vote_cell_fixture::run_all()?);
            Ok(r)
        }) {
            Ok(receipt) => {
                println!("trigger-vote self-test: {} gate(s) PASS", receipt.len());
                for line in receipt {
                    println!("  {line}");
                }
            }
            Err(e) => {
                eprintln!("trigger-vote self-test FAIL: {e:#}");
                std::process::exit(1);
            }
        }
        return;
    }

    if let Some(mode) = plant {
        let Some(base_path) = base else {
            eprintln!("trigger-vote: --plant needs --base <base.gguf>");
            std::process::exit(2);
        };
        let Some(dst) = out else {
            eprintln!("trigger-vote: --plant needs --out <planted.gguf>");
            std::process::exit(2);
        };
        if let Err(e) = run_plant(
            std::path::Path::new(&base_path),
            std::path::Path::new(&dst),
            mode,
            layer,
            triggers,
            fan,
            noise,
            seed,
        ) {
            eprintln!("trigger-vote plant FAIL: {e:#}");
            std::process::exit(1);
        }
        return;
    }

    if let Some(planted_path) = cell {
        let Some(base_path) = base else {
            eprintln!("trigger-vote: --cell needs --base <base.gguf>");
            std::process::exit(2);
        };
        if let Err(e) = run_cell(
            std::path::Path::new(&base_path),
            std::path::Path::new(&planted_path),
            t2a_passed,
            cell_st,
            cell_triggers.as_deref(),
            out.as_deref(),
        ) {
            eprintln!("trigger-vote cell FAIL: {e:#}");
            std::process::exit(1);
        }
        return;
    }

    let (Some(base_path), Some(ft_path)) = (base, ft) else {
        eprintln!(
            "trigger-vote: --base <base.gguf> --ft <finetune.gguf> are required (or --self-test)"
        );
        std::process::exit(2);
    };
    if let Err(e) = run_vote(
        std::path::Path::new(&base_path),
        std::path::Path::new(&ft_path),
        out.as_deref(),
        top,
    ) {
        eprintln!("trigger-vote FAIL: {e:#}");
        std::process::exit(1);
    }
}

fn print_help() {
    println!(
        "trigger-vote — the Issue 1251 trigger-vote screen (checkpoint-ingest backdoor probe)\n\
         \n\
         USAGE:\n\
         \x20 trigger-vote --self-test\n\
         \x20 trigger-vote --base <base.gguf> --ft <finetune.gguf> [--out <artifact.vote1>] [--top N]\n\
         \x20 trigger-vote --plant <t2a|t2b> --base <base.gguf> --out <planted.gguf>\n\
         \x20            [--layer N] [--triggers N] [--fan N] [--noise N] [--seed N]\n\
         \x20 trigger-vote --cell <planted.gguf> --base <base.gguf> [--t2a-passed]\n\
         \n\
         Weights-only vote over the (base, finetune) pair: every MLP neuron votes\n\
         for its top-15 changed tokens by |Δgate·e_t| + |Δup·e_t| (the C.4 rule;\n\
         e_t = the BASE token-embd row); the layer band admits contiguous layers\n\
         whose top-100 lists share ≥10 tokens, and the pooled band ranking is\n\
         the trigger candidate list. Writes the canonical VOTE1 artifact + digest.\n\
         --plant builds the T2 synthetic-backdoor fixture (T2a disclosed-circular;\n\
         T2b non-circular upstream-feature cores); --cell scores the pinned T3 bar."
    );
}

#[allow(clippy::too_many_arguments)]
fn run_plant(
    base_path: &std::path::Path,
    dst_path: &std::path::Path,
    mode: riir_infer_core::trigger_vote_cell::PlantMode,
    layer: usize,
    triggers: usize,
    fan: usize,
    noise: usize,
    seed: u64,
) -> anyhow::Result<()> {
    use riir_infer_core::trigger_vote_cell::{
        PlantConfig, materialize, plan, render_plan, stream_digest,
    };
    let t0 = std::time::Instant::now();
    let gguf =
        GgufFile::open(base_path).with_context(|| format!("opening {}", base_path.display()))?;
    let cfg = PlantConfig {
        mode,
        layer,
        triggers,
        fan_per_trigger: fan,
        noise_neurons: noise,
        seed,
    };
    println!("plant: planning ({}) …", mode.name());
    let p = plan(&gguf, &cfg)?;
    print!("{}", render_plan(&p));
    println!("plant: writing {} + manifest …", dst_path.display());
    let manifest = materialize(base_path, dst_path, &p)?;
    println!(
        "planted: {} (blake3 {}) — base digest {}",
        dst_path.display(),
        hex(&manifest.planted_digest),
        hex(&manifest.base_digest)
    );
    println!(
        "manifest: {} (mode {} seed {:#x} layer {})",
        riir_infer_core::trigger_vote_cell::manifest_path(dst_path).display(),
        mode.name(),
        seed,
        manifest.layer
    );
    let _ = stream_digest(base_path)?; // cheap; already computed inside materialize
    println!("plant pass: {} ms", t0.elapsed().as_millis());
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_cell(
    base_path: &std::path::Path,
    planted_path: &std::path::Path,
    t2a_passed: bool,
    safetensors: bool,
    cell_triggers: Option<&[u32]>,
    out_cell_artifact: Option<&str>,
) -> anyhow::Result<()> {
    use riir_infer_core::trigger_vote_cell::manifest_path;
    let t0 = std::time::Instant::now();
    let (manifest, cell_triggers) = if safetensors {
        // The safetensors lane carries no manifest: the planted tokens
        // are the DOCUMENTED triggers (the organism's card), passed via
        // --cell-triggers.
        let Some(list) = cell_triggers else {
            eprintln!(
                "trigger-vote: --cell-st needs --cell-triggers <comma-separated documented trigger ids>"
            );
            std::process::exit(2);
        };
        (None, Some(list.to_vec()))
    } else {
        let mpath = manifest_path(planted_path);
        let manifest = riir_infer_core::trigger_vote_cell::PlantManifest::load(&mpath)
            .with_context(|| format!("loading {} (plant first)", mpath.display()))?;
        // Verify the planted file against the manifest's digest BEFORE
        // any tensor work (the provenance law).
        let planted_digest = riir_infer_core::trigger_vote_cell::stream_digest(planted_path)?;
        anyhow::ensure!(
            planted_digest == manifest.planted_digest,
            "planted file digest {} != manifest's {} — re-plant or point --cell at the manifest's file",
            hex(&planted_digest),
            hex(&manifest.planted_digest)
        );
        let list = cell_triggers
            .map(|l| l.to_vec())
            .unwrap_or_else(|| manifest.planted_tokens());
        (Some(manifest), Some(list))
    };
    let cell_triggers = cell_triggers.as_deref();
    // The sources borrow their files for 'static: the cell runs to
    // process end, so the files are leaked deliberately (one mmap each;
    // the OS reclaims at exit — a CLI one-shot, not a server).
    let (bs, fs): (Box<dyn VoteTensors>, Box<dyn VoteTensors>) = if safetensors {
        // The HF-native lane: both sides read as safetensors (the
        // supply-chain posture — the files arrive in transformers
        // format, no conversion). The manifest half is skipped; the
        // planted-token list arrives via --cell-triggers.
        // --base may name MULTIPLE shards, comma-separated (the
        // model.safetensors.index.json convention: each tensor lives
        // wholly in one shard).
        let base_shards: Vec<std::path::PathBuf> = base_path
            .to_str()
            .context("--base path is not valid UTF-8")?
            .split(',')
            .map(|p| std::path::PathBuf::from(p.trim()))
            .collect();
        let base_st: &'static riir_infer_core::trigger_vote_safetensors::StShardSet =
            Box::leak(Box::new(
                riir_infer_core::trigger_vote_safetensors::StShardSet::open(&base_shards)
                    .with_context(|| format!("opening {} as safetensors", base_path.display()))?,
            ));
        let ft_st: &'static riir_infer_core::trigger_vote_safetensors::StShardSet =
            Box::leak(Box::new(
                riir_infer_core::trigger_vote_safetensors::StShardSet::open(&[
                    planted_path.to_path_buf()
                ])
                .with_context(|| format!("opening {} as safetensors", planted_path.display()))?,
            ));
        (
            Box::new(
                riir_infer_core::trigger_vote_safetensors::SafetensorsVoteSource::open(base_st)?,
            ),
            Box::new(
                riir_infer_core::trigger_vote_safetensors::SafetensorsVoteSource::open(ft_st)?,
            ),
        )
    } else {
        let base_gguf: &'static GgufFile = Box::leak(Box::new(
            GgufFile::open(base_path)
                .with_context(|| format!("opening {}", base_path.display()))?,
        ));
        let planted_gguf: &'static GgufFile = Box::leak(Box::new(
            GgufFile::open(planted_path)
                .with_context(|| format!("opening {}", planted_path.display()))?,
        ));
        (
            Box::new(GgufVoteSource::open(base_gguf, base_path)?),
            Box::new(GgufVoteSource::open(planted_gguf, planted_path)?),
        )
    };
    let (mode_line, t2a) = match &manifest {
        Some(m) => (
            format!(
                "mode {} seed {:#x} layer {}",
                m.mode.name(),
                m.seed,
                m.layer
            ),
            if m.mode == riir_infer_core::trigger_vote_cell::PlantMode::Upstream {
                Some(t2a_passed)
            } else {
                None
            },
        ),
        None => (
            "safetensors organism (documented triggers)".to_string(),
            None,
        ),
    };
    println!(
        "cell: base={} planted={} ({} triggers={cell_triggers:?}) t2a_passed={}",
        base_path.display(),
        planted_path.display(),
        mode_line,
        t2a_passed
    );
    let rep = if let Some(m) = &manifest {
        riir_infer_core::trigger_vote_cell_score::run_cell(&*bs, &*fs, m, t2a)?
    } else {
        // The safetensors organism cell: detector + baselines + the
        // report scored against the documented trigger list.
        let vote = riir_infer_core::trigger_vote::trigger_vote(&*bs, &*fs)?;
        let baselines =
            riir_infer_core::trigger_vote_cell_score::baseline_rankings(&*bs, &*fs, &vote)?;
        let rep = riir_infer_core::trigger_vote_cell_score::score_cell(
            riir_infer_core::trigger_vote_cell::PlantMode::Upstream,
            cell_triggers.unwrap(),
            &vote,
            &baselines,
            t2a_passed.then_some(true),
        );
        if let Some(out_path) = out_cell_artifact {
            let bytes = riir_infer_core::trigger_vote::build_artifact(&vote);
            std::fs::write(out_path, &bytes).with_context(|| format!("writing {out_path}"))?;
            println!(
                "artifact: {out_path} (blake3 {})",
                hex(&riir_infer_core::trigger_vote::artifact_digest(&bytes))
            );
        }
        rep
    };
    print!("{}", rep.text);
    println!("cell pass: {} ms", t0.elapsed().as_millis());
    Ok(())
}

use anyhow::Context;
fn run_vote(
    base_path: &std::path::Path,
    ft_path: &std::path::Path,
    out: Option<&str>,
    top: usize,
) -> anyhow::Result<()> {
    let t0 = Instant::now();
    let base_gguf =
        GgufFile::open(base_path).with_context(|| format!("opening {}", base_path.display()))?;
    let ft_gguf =
        GgufFile::open(ft_path).with_context(|| format!("opening {}", ft_path.display()))?;
    let base_src = GgufVoteSource::open(&base_gguf, base_path)?;
    let ft_src = GgufVoteSource::open(&ft_gguf, ft_path)?;
    println!(
        "trigger-vote: base={} ft={} ({} layers, n_ff={}, n_embd={}, vocab={})",
        base_path.display(),
        ft_path.display(),
        base_src.n_layers(),
        base_src.n_ff(),
        base_src.n_embd(),
        base_src.vocab(),
    );
    let tok = BpeTokenizer::from_gguf(&base_gguf).ok();
    let v = trigger_vote(&base_src, &ft_src)?;
    println!("vote pass: {} ms", t0.elapsed().as_millis());
    print!("{}", render_report(&v, tok.as_ref(), top));
    if let Some(out_path) = out {
        let bytes = build_artifact(&v);
        std::fs::write(out_path, &bytes).with_context(|| format!("writing {out_path}"))?;
        let digest = artifact_digest(&bytes);
        println!(
            "artifact: {out_path} ({} bytes, blake3 {})",
            bytes.len(),
            hex(&digest)
        );
    }
    match v.verdict {
        Verdict::Voted => println!("verdict: VOTED"),
        Verdict::ZeroDelta => println!("verdict: ZERO-DELTA (base and finetune are bit-identical)"),
        Verdict::ZeroVotes => {
            println!("verdict: ZERO-VOTES (rows changed but every score is exactly 0)")
        }
    }
    Ok(())
}
