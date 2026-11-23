//! neuron-census — the Issue-1241 census front (the Fermion distill,
//! katgpt-rs Research 684).
//!
//! Postures:
//! - `--self-test` — runs the G1 fixture gates in-process (the
//!   planted-edit recovery battery: rank-1 recovery, the
//!   pre-registered `max(floor, 28×band)` bar, the zero-false-fire
//!   control arm, artifact mint + BLAKE3 trailer verify) and prints
//!   the receipt. This is the canary vehicle — no model file needed.
//! - `--model <gguf>` — the T2 real-checkpoint cell. REFUSES LOUD
//!   until the GGUF census adapter lands (the adapter + the cell route
//!   to the 4090 `--needs cuda` window per the issue; the runner, the
//!   state-dir protocol and the artifact minter already exist).
//!
//! Reader protection: `required-features = ["neuron_census"]` — a
//! default build never compiles this bin (the green-zero law).

fn main() {
    let mut args = std::env::args().skip(1);
    let mut self_test = false;
    let mut model: Option<String> = None;
    let mut out_dir: Option<String> = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--self-test" => self_test = true,
            "--model" => model = args.next(),
            "--out" => out_dir = args.next(),
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
        return;
    }

    if model.is_some() {
        // T2 is a separate landing: the GGUF census adapter + the
        // layer table + the logit-lens observable. The runner, the
        // resumable state protocol and the canonical artifact minter
        // exist (see `src/neuron_census.rs`); this refusal is the
        // loud T1/T2 seam, never a silent no-op.
        eprintln!(
            "neuron-census: --model is the T2 cell and its GGUF adapter is not landed yet;\n\
             the runner + G1 fixture gates are testable today:\n\
             cargo test -p riir-infer-core --features neuron_census --lib\n\
             neuron-census --self-test"
        );
        if out_dir.is_some() {
            eprintln!("neuron-census: --out is likewise T2 (ignored on this refusal)");
        }
        std::process::exit(2);
    }

    print_help();
    std::process::exit(2);
}

fn print_help() {
    println!(
        "neuron-census — the Issue-1241 offline interventional neuron census\n\
         \n\
         USAGE:\n\
         \x20 neuron-census --self-test                run the G1 fixture gates (no model needed)\n\
         \x20 neuron-census --model <gguf> [--out DIR] the T2 real-checkpoint cell (refuses until\n\
         \x20                                          the GGUF adapter lands)\n\
         \n\
         The lane is opt-in (`neuron_census` feature), offline by construction, and never\n\
         touches a serving path."
    );
}
