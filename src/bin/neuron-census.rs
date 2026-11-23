//! neuron-census — the Issue-1241 census front (the Fermion distill,
//! katgpt-rs Research 684).
//!
//! Postures:
//! - `--self-test` — runs the G1 fixture gates in-process (the
//!   planted-edit recovery battery: rank-1 recovery, the
//!   pre-registered `max(floor, 28×band)` bar, the zero-false-fire
//!   control arm, artifact mint + BLAKE3 trailer verify) and prints
//!   the receipt. This is the canary vehicle — no model file needed.
//! - `--mechanics-check --model <gguf>` — the push/restore protocol
//!   gate on a REAL checkpoint (CPU): identity push ≡ base
//!   bit-identically, dose monotonicity on a live column, restore
//!   digest identity. T2-prep.
//! - `--model <gguf>` bare — the T2 full-sweep cell. REFUSES LOUD: the
//!   observable/probe science and the sweep posture are tuned at the
//!   4090 `--needs cuda` window per the issue (the runner, the
//!   state-dir protocol, the artifact minter and the mechanics gate
//!   all exist).
//!
//! Reader protection: `required-features = ["neuron_census"]` — a
//! default build never compiles this bin (the green-zero law).

fn main() {
    let mut args = std::env::args().skip(1);
    let mut self_test = false;
    let mut mechanics = false;
    let mut model: Option<String> = None;
    let mut out_dir: Option<String> = None;
    let mut probes: Vec<String> = Vec::new();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--self-test" => self_test = true,
            "--mechanics-check" => mechanics = true,
            "--model" => model = args.next(),
            "--out" => out_dir = args.next(),
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
        match riir_infer_core::neuron_census_gguf::mechanics_check(
            std::path::Path::new(&model_path),
            &probes,
        ) {
            Ok(line) => println!("neuron-census {line}"),
            Err(e) => {
                eprintln!("neuron-census mechanics FAIL: {e:#}");
                std::process::exit(1);
            }
        }
        return;
    }

    if model.is_some() {
        // The T2 SWEEP cell is a separate landing: the observable/probe
        // science + the full-sweep posture are tuned at the 4090
        // `--needs cuda` window (the runner, the state-dir protocol,
        // the artifact minter and the mechanics gate exist — see
        // `src/neuron_census.rs` + `src/neuron_census_gguf.rs`).
        eprintln!(
            "neuron-census: the full-sweep cell is T2 (routed to the 4090 --needs cuda window);\n\
             what runs today:\n\
             \x20 neuron-census --mechanics-check --model <gguf> [--probe \"text\"] ...\n\
             \x20 neuron-census --self-test"
        );
        if out_dir.is_some() {
            eprintln!("neuron-census: --out is likewise T2 (ignored on this refusal)");
        }
        std::process::exit(2);
    }

    print_help();
    std::process::exit(2);
}

fn run_self_test() {
    let cfg = riir_infer_core::neuron_census::CensusConfig::default();
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
}

fn print_help() {
    println!(
        "neuron-census — the Issue-1241 offline interventional neuron census\n\
         \n\
         USAGE:\n\
         \x20 neuron-census --self-test                        run the G1 fixture gates (no model needed)\n\
         \x20 neuron-census --mechanics-check --model <gguf>   the push/restore protocol gate on a real\n\
         \x20        [--probe \"text\"] ...                      checkpoint (CPU; identity ≡ base, dose\n\
         \x20                                                  monotonicity, restore digest identity)\n\
         \x20 neuron-census --model <gguf> [--out DIR]         the T2 full-sweep cell (routed to the\n\
         \x20                                                  4090 --needs cuda window; refuses here)\n\
         \n\
         The lane is opt-in (`neuron_census` feature), offline by construction, and never\n\
         touches a serving path."
    );
}
