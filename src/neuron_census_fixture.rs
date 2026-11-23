//! Issue-1241 G1 fixture (the planted-edit recovery battery).
//!
//! Toy hermetic MLP, deterministic from one pinned SplitMix64 stream:
//! one census layer, a single-coordinate readout (the observable), and
//! a dominant planted path — the planted neuron's hidden activation is
//! engineered large and stable across reads while the rest of the
//! population stays quiet, so the closed-form column-scale edit moves
//! the observable orders of magnitude harder than a matched-random
//! push. That engineering is the fixture against the PRE-REGISTERED
//! bar (the issue: "pre-register per fixture" — Fermion's 40-vs-1.4 is
//! his behavioral fixture's scale, never portable).
//!
//! [`g1_receipt`] is the gate's one entry point (the `neuron-census`
//! bin's `--self-test` drives it); the `#[cfg(test)]` battery here
//! adds the kill/resume, foreign-cursor, control-draw and verdict
//! pins. Determinism (Issue 940): f64 sequential folds, first-index
//! ties.
//!
//! Feature-gated (not test-gated) so the bin can run it; an unarmed
//! build compiles the whole lane to nothing (G3).

use crate::neuron_census::*;

// ── pinned fixture constants ─────────────────────────────────────────

const D_MODEL: usize = 16;
const N_NEURONS: usize = 24;
const N_READS: usize = 16;
/// The planted column (hidden from the census; the gate asserts
/// against it).
const PLANTED_COLUMN: usize = 11;
/// The planted edit's multiplier (the closed-form column-scale edit).
const PLANTED_SCALE: f32 = 3.0;
/// One fixture seed, one SplitMix64 stream — every matrix and probe
/// read from it, in a fixed construction order.
pub const FIXTURE_SEED: u64 = 0x1241_A17C_0000_0001;
/// The readout coordinate the observable reads (w2 row 0).
const READOUT_ROW: usize = 0;

/// SplitMix64 — the fixture's only randomness (pinned).
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// f32 in [-1, 1) from 24 bits of the stream.
    fn next_sym(&mut self) -> f32 {
        let v = (self.next() >> 40) as u32 & 0x00FF_FFFF;
        (v as f32 / 0x80_0000 as f32) - 1.0
    }

    /// f32 in [0.5, 1.0) — the probe inputs (positive, so the planted
    /// path's pre-activation stays positive under its positive row).
    fn next_probe(&mut self) -> f32 {
        0.75 + 0.25 * self.next_sym()
    }
}

/// The toy MLP: h = relu(W1 x), observable = (W2·h)[READOUT_ROW] with
/// the planted view composed multiplicatively into W2's columns.
/// Row-major [out][in] (the GGUF `ffn_down` axes note in
/// `weight_audit.rs`).
pub struct ToyMlp {
    w1: Vec<f32>, // [N_NEURONS][D_MODEL]
    w2: Vec<f32>, // [D_MODEL][N_NEURONS] — neuron n's write column is
    // w2[i * N_NEURONS + n] (a ColumnScale site, exactly
    // the census's candidate axis).
    inputs: Vec<f32>, // [N_READS][D_MODEL] in [0.5, 1)
    /// The permanent planted view (column, multiplier).
    planted: (usize, f32),
    /// The temporary push view (column, multiplier) — composed ON TOP
    /// of the plant, never replacing it.
    push: Option<(usize, f32)>,
}

impl ToyMlp {
    pub fn new(seed: u64) -> Self {
        let mut rng = SplitMix64(seed);
        let mut w1 = vec![0.0f32; N_NEURONS * D_MODEL];
        for v in w1.iter_mut() {
            *v = rng.next_sym() * 0.05; // quiet population by default
        }
        // The planted path: a large positive weight row aligned with
        // the positive probe inputs — h[PLANTED] is large and stable.
        for j in 0..D_MODEL {
            w1[PLANTED_COLUMN * D_MODEL + j] = 0.5;
        }
        let mut w2 = vec![0.0f32; D_MODEL * N_NEURONS];
        for v in w2.iter_mut() {
            *v = rng.next_sym() * 0.25;
        }
        let mut inputs = vec![0.0f32; N_READS * D_MODEL];
        for v in inputs.iter_mut() {
            *v = rng.next_probe();
        }
        Self {
            w1,
            w2,
            inputs,
            planted: (PLANTED_COLUMN, PLANTED_SCALE),
            push: None,
        }
    }

    fn forward_observable(&self, read: usize) -> f64 {
        let x = &self.inputs[read * D_MODEL..(read + 1) * D_MODEL];
        let mut h = [0.0f32; N_NEURONS];
        for (n, row) in self.w1.as_chunks::<D_MODEL>().0.iter().enumerate() {
            let mut acc = 0.0f32;
            for (j, w) in row.iter().enumerate() {
                acc += w * x[j];
            }
            h[n] = acc.max(0.0);
        }
        let (pc, ps) = self.planted;
        let row = &self.w2[READOUT_ROW * N_NEURONS..(READOUT_ROW + 1) * N_NEURONS];
        let mut obs = 0.0f64;
        for (n, &wn) in row.iter().enumerate() {
            let mut w = wn;
            if n == pc {
                w *= ps;
            }
            if let Some((col, f)) = self.push
                && n == col
            {
                w *= f;
            }
            obs += (w * h[n]) as f64;
        }
        obs
    }
}

/// The census model adapter over the toy MLP. The push is evaluated
/// through the temporary view — the base weights never mutate, the
/// capture loop allocates nothing.
pub struct ToyCensusModel {
    mlp: ToyMlp,
    layers: Vec<CensusLayer>,
    digest: [u8; 32],
}

impl ToyCensusModel {
    pub fn new(mlp: ToyMlp) -> Self {
        // The digest binds the BASE weights (the plant is a view, not
        // bytes) — two adapters over the same fixture share it, which
        // is what the resume cursor binds.
        let mut bytes = Vec::with_capacity((mlp.w1.len() + mlp.w2.len() + mlp.inputs.len()) * 4);
        for v in mlp.w1.iter().chain(mlp.w2.iter()).chain(mlp.inputs.iter()) {
            bytes.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        let digest = *blake3::hash(&bytes).as_bytes();
        let layers = vec![CensusLayer {
            layer: 0,
            neurons: N_NEURONS,
        }];
        Self {
            mlp,
            layers,
            digest,
        }
    }
}

impl CensusModel for ToyCensusModel {
    fn model_digest(&self) -> [u8; 32] {
        self.digest
    }

    fn layers(&self) -> &[CensusLayer] {
        &self.layers
    }

    fn base_observable(&mut self, read: usize) -> f64 {
        self.mlp.forward_observable(read)
    }

    fn push_observable(&mut self, _layer_index: usize, column: u32, f: f32, read: usize) -> f64 {
        self.mlp.push = Some((column as usize, f));
        let obs = self.mlp.forward_observable(read);
        self.mlp.push = None;
        obs
    }
}

/// The G1 receipt — the gate's pre-registration record (all numbers
/// beside the bar they cleared).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct G1Receipt {
    pub top_column: u32,
    pub effect: f64,
    pub band_mean: f64,
    /// `effect / band_mean` — the dominance ratio beside the 28× bar.
    pub ratio: f64,
    pub false_fires: usize,
    /// Leaderboard (column, effect), best first — the rank-1 evidence.
    pub leaderboard: [(u32, f64); 6],
    pub artifact_len: usize,
}

/// Run the G1 gate end to end: full sweep → rank-1 assertion → the
/// pre-registered bar → the control-arm false-fire pin → artifact
/// mint + trailer verify. `Err` carries the failing evidence.
pub fn g1_receipt(cfg: &CensusConfig) -> Result<G1Receipt, String> {
    let mut model = ToyCensusModel::new(ToyMlp::new(FIXTURE_SEED));
    let mut sweep = CensusSweep::default();
    let mut draw = ControlDraw::new(N_NEURONS);
    let layer = CensusLayer {
        layer: 0,
        neurons: N_NEURONS,
    };
    for column in 0..N_NEURONS {
        let ids = draw.draw(cfg, 0, column as u32);
        sweep
            .adjudicate_candidate(
                &mut model,
                Candidate {
                    layer_index: 0,
                    layer: &layer,
                    column: column as u32,
                    control_ids: &ids,
                },
                N_READS,
                cfg,
                &mut |m, _li, c, f, read| m.push_observable(0, c, f, read),
            )
            .map_err(|e| e.to_string())?;
    }

    let mut ranked: Vec<&CensusRow> = sweep.rows.iter().collect();
    ranked.sort_by(|a, b| b.effect.partial_cmp(&a.effect).expect("finite effects"));
    let top = ranked[0];
    let leaderboard: [(u32, f64); 6] = ranked
        .iter()
        .take(6)
        .map(|r| (r.column, r.effect))
        .collect::<Vec<_>>()
        .try_into()
        .expect("six rows");

    if top.column as usize != PLANTED_COLUMN {
        return Err(format!(
            "rank-1 recovery missed: top column {} effect {}\nleaderboard {leaderboard:#?}",
            top.column, top.effect
        ));
    }
    if top.verdict != CensusVerdict::SwitchFound {
        return Err(format!("top row must be SwitchFound, got {top:?}"));
    }
    if !(top.effect >= cfg.abs_floor && top.effect >= cfg.band_ratio * top.band_mean) {
        return Err(format!(
            "bar miss: effect {} vs floor {} vs 28×band {}",
            top.effect,
            cfg.abs_floor,
            cfg.band_ratio * top.band_mean
        ));
    }

    let mut false_fires = 0usize;
    let mut fired: Vec<(u32, f64, f64)> = Vec::new();
    for row in sweep.rows.iter() {
        if row.column as usize == PLANTED_COLUMN {
            continue;
        }
        if row.verdict == CensusVerdict::SwitchFound {
            false_fires += 1;
            fired.push((row.column, row.effect, row.band_mean));
        }
    }
    if false_fires != 0 {
        return Err(format!("control arm false-fired: {fired:#?}"));
    }

    let artifact = mint_artifact(&sweep, &model.model_digest(), cfg);
    let payload_len = verify_artifact(&artifact)
        .ok_or_else(|| "minted artifact failed its own BLAKE3 trailer".to_string())?;
    if payload_len != artifact.len() - 32 {
        return Err("artifact payload length mismatch".to_string());
    }

    Ok(G1Receipt {
        top_column: top.column,
        effect: top.effect,
        band_mean: top.band_mean,
        ratio: top.effect / top.band_mean,
        false_fires,
        leaderboard,
        artifact_len: artifact.len(),
    })
}

/// The parallel/serial twin receipt — the `run_sweep_par` driver's
/// gates on the same fixture model: rows equal the serial sweep
/// bit-for-bit, a chunked kill/resume lands the identical artifact.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParReceipt {
    pub rows: usize,
    pub resume_identical: bool,
    pub workers: usize,
    pub chunk: usize,
}

/// Run the parallel driver over fresh fixture models and pin it against
/// the serial sweep: same rows (bit-exact), same artifact, and a
/// mid-sweep kill + resume that converges to the uninterrupted bytes.
pub fn par_receipt(cfg: &CensusConfig) -> Result<ParReceipt, String> {
    use crate::neuron_census::run_sweep_par;

    const WORKERS: usize = 4;
    const CHUNK: usize = 3;

    let mut serial_model = ToyCensusModel::new(ToyMlp::new(FIXTURE_SEED));
    let serial = run_sweep(&mut serial_model, cfg, None, false, &mut |_, _, _| {})
        .map_err(|e| e.to_string())?;
    let serial_artifact = mint_artifact(&serial, &serial_model.model_digest(), cfg);

    let dir = std::env::temp_dir().join(format!("neuron_census_par_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let models: Vec<Box<dyn CensusModel>> = (0..WORKERS)
        .map(|_| Box::new(ToyCensusModel::new(ToyMlp::new(FIXTURE_SEED))) as Box<dyn CensusModel>)
        .collect();
    let par = run_sweep_par(models, CHUNK, cfg, &dir, false, None, &mut |_, _, _| {})
        .map_err(|e| e.to_string())?;
    if par.rows != serial.rows {
        return Err(format!(
            "parallel rows diverged from serial ({} vs {} rows, first diff at {:?})",
            par.rows.len(),
            serial.rows.len(),
            par.rows
                .iter()
                .zip(serial.rows.iter())
                .position(|(a, b)| a != b),
        ));
    }
    let par_artifact = mint_artifact(&par, &serial_model.model_digest(), cfg);
    if par_artifact != serial_artifact {
        return Err("parallel artifact diverged from serial".to_string());
    }

    // Chunked kill + resume: kill after ~half the candidates (the
    // progress hook fires on the committer thread — this main thread —
    // so a panic here unwinds through the scope and joins the workers),
    // then resume to the identical artifact.
    let kdir =
        std::env::temp_dir().join(format!("neuron_census_par_resume_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&kdir);
    let kill_at = serial.rows.len() / 2;
    let done = std::cell::Cell::new(0usize);
    let models: Vec<Box<dyn CensusModel>> = (0..WORKERS)
        .map(|_| Box::new(ToyCensusModel::new(ToyMlp::new(FIXTURE_SEED))) as Box<dyn CensusModel>)
        .collect();
    // Silence the expected kill's panic print (the receipt, not a
    // failure log — the hook is restored before returning).
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let killed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = run_sweep_par(models, CHUNK, cfg, &kdir, false, None, &mut |_, _, _| {
            let d = done.get() + 1;
            done.set(d);
            if d == kill_at {
                panic!("simulated kill mid-parallel-sweep");
            }
        });
    }));
    std::panic::set_hook(prev_hook);
    if killed.is_err() {
        // Expected — the kill fired. The committed prefix is the ledger.
    } else {
        return Err("the simulated parallel kill must fire".to_string());
    }
    let models: Vec<Box<dyn CensusModel>> = (0..WORKERS)
        .map(|_| Box::new(ToyCensusModel::new(ToyMlp::new(FIXTURE_SEED))) as Box<dyn CensusModel>)
        .collect();
    let resumed = run_sweep_par(models, CHUNK, cfg, &kdir, true, None, &mut |_, _, _| {})
        .map_err(|e| e.to_string())?;
    let resumed_artifact = mint_artifact(&resumed, &serial_model.model_digest(), cfg);
    let identical = resumed_artifact == serial_artifact;
    if !identical {
        return Err("parallel resume artifact diverged from the uninterrupted sweep".to_string());
    }

    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&kdir);
    Ok(ParReceipt {
        rows: serial.rows.len(),
        resume_identical: identical,
        workers: WORKERS,
        chunk: CHUNK,
    })
}

// ── the cfg(test) battery ────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use katgpt_pruners::mech_attribution::{ControlVerdict, control_verdict};

    /// G1: the receipt clears every pre-registered assertion.
    #[test]
    fn g1_planted_column_scale_edit_recovers_rank1() {
        let cfg = CensusConfig::default();
        let r = g1_receipt(&cfg).expect("G1 must pass");
        assert_eq!(r.top_column as usize, super::PLANTED_COLUMN);
        assert_eq!(r.false_fires, 0);
        assert!(r.ratio >= cfg.band_ratio, "ratio {} below 28×", r.ratio);
        // The leaderboard's top margin over the runner-up is part of
        // the record (rank-1 must not be a hair).
        assert!(
            r.leaderboard[0].1 > r.leaderboard[1].1,
            "rank-1 margin vanished: {:#?}",
            r.leaderboard
        );
    }

    /// Kill/resume: a hard kill after the 7th candidate (the panic
    /// fires through the progress hook, AFTER the cursor committed)
    /// resumes to a bit-identical artifact — the rows ledger + cursor
    /// are the whole state.
    #[test]
    fn resume_matches_uninterrupted_sweep_bit_identically() {
        use std::cell::Cell;
        use std::panic::{AssertUnwindSafe, catch_unwind};

        const KILL_AFTER: usize = 7;
        let cfg = CensusConfig::default();
        let dir = std::env::temp_dir().join(format!("neuron_census_resume_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        let mut m = ToyCensusModel::new(ToyMlp::new(FIXTURE_SEED));
        let full = run_sweep(&mut m, &cfg, None, false, &mut |_, _, _| {}).expect("full sweep");
        let full_artifact = mint_artifact(&full, &m.model_digest(), &cfg);

        let mut mk = ToyCensusModel::new(ToyMlp::new(FIXTURE_SEED));
        let done = Cell::new(0usize);
        let killed = catch_unwind(AssertUnwindSafe(|| {
            let _ = run_sweep(&mut mk, &cfg, Some(&dir), false, &mut |_, _, _| {
                let d = done.get() + 1;
                done.set(d);
                if d == KILL_AFTER {
                    panic!("simulated kill mid-sweep");
                }
            });
        }));
        assert!(killed.is_err(), "the simulated kill must fire");
        assert_eq!(
            done.get(),
            KILL_AFTER,
            "the kill must land on the hook count"
        );

        let mut m2 = ToyCensusModel::new(ToyMlp::new(FIXTURE_SEED));
        let resumed =
            run_sweep(&mut m2, &cfg, Some(&dir), true, &mut |_, _, _| {}).expect("resume");
        let resumed_artifact = mint_artifact(&resumed, &m2.model_digest(), &cfg);

        assert_eq!(
            resumed.rows.len(),
            full.rows.len(),
            "resume must carry the prior candidates' rows"
        );
        assert_eq!(full_artifact, resumed_artifact, "resume artifact diverged");
        assert_eq!(
            verify_artifact(&resumed_artifact),
            Some(resumed_artifact.len() - 32),
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A stale cursor (different model digest) refuses loud — a cursor
    /// is model-bound, never silently restarted.
    #[test]
    fn resume_refuses_a_foreign_cursor() {
        let cfg = CensusConfig::default();
        let dir =
            std::env::temp_dir().join(format!("neuron_census_foreign_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("cursor"), "v1 deadbeef 0 3\n").unwrap();

        let mut m = ToyCensusModel::new(ToyMlp::new(FIXTURE_SEED));
        let err = run_sweep(&mut m, &cfg, Some(&dir), true, &mut |_, _, _| {});
        assert!(err.is_err(), "a foreign cursor must refuse");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Control draws: disjoint from the candidate, in-range, stable
    /// across fresh draws with the same inputs.
    #[test]
    fn control_draw_disjoint_and_stable() {
        let cfg = CensusConfig::default();
        let mut d1 = ControlDraw::new(N_NEURONS);
        let mut d2 = ControlDraw::new(N_NEURONS);
        for column in 0..N_NEURONS {
            let a = d1.draw(&cfg, 0, column as u32);
            let b = d2.draw(&cfg, 0, column as u32);
            assert_eq!(a, b, "draw diverged at column {column}");
            assert_eq!(a.len(), MIN_CONTROLS);
            assert!(
                !a.contains(&column),
                "draw contains its own candidate at column {column}"
            );
            assert!(a.iter().all(|&x| x < N_NEURONS));
        }
    }

    /// The verdict contract on this lane's own numbers: a
    /// band-dominated effect reads NoSwitch (a confident negative,
    /// never folded); too-few controls reads Undecided (an absence,
    /// never a negative).
    #[test]
    fn verdict_refuses_when_band_dominates() {
        let cfg = CensusConfig::default();
        let mut controls = vec![0.01f64; MIN_CONTROLS];
        let v = control_verdict(0.02, N_READS, &controls, &cfg.control_params());
        assert!(matches!(v, ControlVerdict::NoSwitch { .. }));
        controls[0] = 1.0; // the band max now dominates the candidate
        let v = control_verdict(0.02, N_READS, &controls, &cfg.control_params());
        assert!(matches!(v, ControlVerdict::NoSwitch { .. }));
        let v = control_verdict(1.0, N_READS, &controls[..4], &cfg.control_params());
        assert!(matches!(v, ControlVerdict::Undecided { .. }));
    }

    // ── the T2 access check: artifact reader + set operation ────────

    use crate::neuron_census::{AccessClass, AccessSide, access_check, parse_artifact};

    /// Hand-minted rows (no model): one row per spec, verdict carried
    /// explicitly. Mint sorts by (layer, column) at canon time.
    fn row(layer: u8, column: u32, verdict: CensusVerdict, effect: f64) -> CensusRow {
        CensusRow {
            layer,
            column,
            direction_sign: 1,
            best_dose: 1.0,
            effect,
            verdict,
            band_mean: if verdict == CensusVerdict::Undecided {
                f64::NAN
            } else {
                effect / 10.0
            },
            band_max: if verdict == CensusVerdict::Undecided {
                f64::NAN
            } else {
                effect / 5.0
            },
        }
    }

    fn mint_rows(digest: &[u8; 32], cfg: &CensusConfig, rows: &[CensusRow]) -> Vec<u8> {
        let mut sweep = CensusSweep::default();
        sweep.rows = rows.to_vec();
        mint_artifact(&sweep, digest, cfg)
    }

    /// The issue's G1 fixture vocabulary, on synthetic artifacts: a
    /// site switched in BOTH models classifies replicated; a site
    /// switched in one only classifies present-inert on that side;
    /// mint→parse round-trips every field bit-exactly.
    #[test]
    fn access_check_classifies_replicated_and_present_inert() {
        let cfg = CensusConfig::default();
        let band = 1e-3f64;
        // K1 replicated; K2 base-only; K3 chat-only; K4 no-switch in
        // both (never enters a set); K5 undecided in base (disclosed,
        // never a member).
        let base = mint_rows(
            &[1u8; 32],
            &cfg,
            &[
                row(3, 10, CensusVerdict::SwitchFound, 5.0),
                row(3, 20, CensusVerdict::SwitchFound, 2.0),
                row(3, 40, CensusVerdict::NoSwitch, 0.0001),
                row(3, 50, CensusVerdict::Undecided, f64::NAN),
            ],
        );
        let chat = mint_rows(
            &[2u8; 32],
            &cfg,
            &[
                row(3, 10, CensusVerdict::SwitchFound, 4.0),
                row(3, 30, CensusVerdict::SwitchFound, 3.0),
                row(3, 40, CensusVerdict::NoSwitch, 0.0002),
            ],
        );
        let rep = access_check(&base, &chat, band).expect("comparable artifacts");
        assert_eq!(rep.switches_base, 2);
        assert_eq!(rep.switches_chat, 2);
        assert_eq!(rep.undecided_base, 1);
        assert_eq!(rep.undecided_chat, 0);
        assert_eq!(rep.replicated, 1);
        assert_eq!(rep.present_inert_base, 1);
        assert_eq!(rep.present_inert_chat, 1);
        assert_eq!(rep.noise_marginal, 0);
        assert_eq!(rep.sites.len(), 3);
        // Roundtrip: parse returns the minted rows bit-exactly.
        let (h, rows) = parse_artifact(&base).expect("parse");
        assert_eq!(h.weights_digest, [1u8; 32]);
        assert_eq!(h.n_rows, 4);
        assert_eq!(h.push_scale, cfg.push_scale);
        assert_eq!(h.abs_floor.to_bits(), cfg.abs_floor.to_bits());
        assert_eq!(h.band_ratio.to_bits(), cfg.band_ratio.to_bits());
        assert_eq!(h.seed, cfg.seed);
        assert_eq!(rows.len(), 4);
        let orig = [
            row(3, 10, CensusVerdict::SwitchFound, 5.0),
            row(3, 20, CensusVerdict::SwitchFound, 2.0),
            row(3, 40, CensusVerdict::NoSwitch, 0.0001),
            row(3, 50, CensusVerdict::Undecided, f64::NAN),
        ];
        for (got, want) in rows.iter().zip(orig.iter()) {
            assert_eq!(got.layer, want.layer);
            assert_eq!(got.column, want.column);
            assert_eq!(got.verdict, want.verdict);
            assert_eq!(got.effect.to_bits(), want.effect.to_bits());
            assert_eq!(got.band_mean.to_bits(), want.band_mean.to_bits());
        }
        // The prior feed carries the confident rows strongest-first
        // (K1's weaker side 4.0 > K3's 3.0 > K2's 2.0).
        let prior = rep.authoring_prior();
        assert_eq!(prior.len(), 3);
        assert_eq!(prior[0].column, 10);
        assert_eq!(prior[1].column, 30);
        assert_eq!(prior[2].column, 20);
    }

    /// The quant-noise law: rows under the band type NoiseMarginal —
    /// a both-side under-band pair, a one-side under-band singleton,
    /// a mixed band pair — never pooled with the confident classes.
    #[test]
    fn access_check_noise_band_types_marginal_sites() {
        let cfg = CensusConfig::default();
        let band = 1e-3f64;
        let base = mint_rows(
            &[1u8; 32],
            &cfg,
            &[
                row(7, 1, CensusVerdict::SwitchFound, 0.0004), // both-side under band
                row(7, 2, CensusVerdict::SwitchFound, 0.0009), // base-only under band
                row(7, 3, CensusVerdict::SwitchFound, 2.0),    // confident base-only
            ],
        );
        let chat = mint_rows(
            &[2u8; 32],
            &cfg,
            &[
                row(7, 1, CensusVerdict::SwitchFound, 0.0006),
                row(7, 4, CensusVerdict::SwitchFound, 0.0008), // chat-only under band
            ],
        );
        let rep = access_check(&base, &chat, band).expect("comparable artifacts");
        assert_eq!(rep.replicated, 0);
        assert_eq!(rep.present_inert_base, 1); // only K3
        assert_eq!(rep.present_inert_chat, 0);
        assert_eq!(rep.noise_marginal, 3); // K1 both-under, K2 base-under, K4 chat-under
        // Exact class spot-checks.
        let k1 = rep.sites.iter().find(|s| s.column == 1).unwrap();
        assert_eq!(k1.class, AccessClass::NoiseMarginal);
        let k3 = rep.sites.iter().find(|s| s.column == 3).unwrap();
        assert_eq!(
            k3.class,
            AccessClass::PresentInert {
                present_in: AccessSide::Base,
                effect: 2.0
            }
        );
        let k2 = rep.sites.iter().find(|s| s.column == 2).unwrap();
        assert_eq!(k2.class, AccessClass::NoiseMarginal);
        // Band edge: exactly-at-band is confident (>= band).
        let base_edge = mint_rows(
            &[1u8; 32],
            &cfg,
            &[row(7, 9, CensusVerdict::SwitchFound, band)],
        );
        let chat_edge = mint_rows(&[2u8; 32], &cfg, &[]);
        let rep_edge = access_check(&base_edge, &chat_edge, band).unwrap();
        assert_eq!(rep_edge.present_inert_base, 1);
    }

    /// Refusals: config mismatch (floor/band/scale/seed differ) bails
    /// loud; a flipped byte fails the trailer; a truncated artifact
    /// fails; an unknown verdict byte bails in row decode.
    #[test]
    fn access_check_refuses_config_mismatch_and_tampered_bytes() {
        let cfg = CensusConfig::default();
        let mut cfg2 = cfg.clone();
        cfg2.abs_floor = 1e-4; // different floor ⇒ not the same measurement
        let a = mint_rows(
            &[1u8; 32],
            &cfg,
            &[row(1, 1, CensusVerdict::SwitchFound, 1.0)],
        );
        let b = mint_rows(
            &[2u8; 32],
            &cfg2,
            &[row(1, 1, CensusVerdict::SwitchFound, 1.0)],
        );
        assert!(
            access_check(&a, &b, 1e-3).is_err(),
            "config mismatch must refuse"
        );

        let mut tampered = a.clone();
        let mid = tampered.len() / 2;
        tampered[mid] ^= 0xff;
        assert!(
            parse_artifact(&tampered).is_err(),
            "trailer must catch tampering"
        );

        let truncated = &a[..a.len() - 40];
        assert!(parse_artifact(truncated).is_err(), "truncation must refuse");

        // Same rows under the SAME config are comparable even though
        // the weights digests differ (base-vs-chat is the point).
        let b_same_cfg = mint_rows(
            &[2u8; 32],
            &cfg,
            &[row(1, 1, CensusVerdict::SwitchFound, 1.0)],
        );
        let rep = access_check(&a, &b_same_cfg, 1e-3).expect("same-config artifacts comparable");
        assert_eq!(rep.replicated, 1);
        assert_ne!(rep.weights_digest_base, rep.weights_digest_chat);
    }
}
