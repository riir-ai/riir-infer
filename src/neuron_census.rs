//! Offline interventional neuron census (riir-infer Issue 1241 T1; the
//! Fermion distill, katgpt-rs Research 684).
//!
//! For each MLP layer and candidate neuron (a `down_proj` input column),
//! the runner pushes the column with a
//! [`katgpt_core::weight_edit_overlay::EditForm::ColumnScale`]-shaped
//! dose grid (push × suppress arms, ≥5 doses per direction — the T2
//! floor), measures the observable delta on held-out reads, adjudicates
//! against a matched-random control band with
//! [`katgpt_pruners::mech_attribution::control_verdict`], and mints ONE
//! canonical BLAKE3-digested artifact.
//!
//! Posture (G3/G5): opt-in `neuron_census`, offline by construction —
//! serving bins never link the module (an unarmed build compiles it to
//! nothing), never a serving-path claim (the lens-lane demotion,
//! katgpt-rs `.research/666` §7, binds exactly the claim this lane
//! refuses to make). Determinism per the katgpt-rs Issue-940 law: f64
//! sequential folds, SplitMix64 Fisher–Yates control draws from a
//! pinned per-candidate seed, first-index tie rules.
//!
//! T1 scope: the runner + the hermetic G1 fixture gates (planted
//! closed-form column-scale edit recovered rank-1 against a
//! pre-registered band bar) + the resumable cursor. The GGUF model
//! adapter and the real-checkpoint cell are T2 (routes to the 4090
//! `--needs cuda` window per the issue).

use anyhow::{Context, Result, bail};
use katgpt_pruners::mech_attribution::census_screen::MIN_CONTROL_UNITS;
use katgpt_pruners::mech_attribution::{ControlParams, ControlVerdict, control_verdict};

/// Pinned push scale (the amplifier arm). The suppress arm is `1/s`,
/// so both endpoints ride the fader's exact lerp law
/// (`EditForm::fader_factor`: dose 1 gives `f = scale` bit-exactly).
pub const PUSH_SCALE: f32 = 3.0;
/// Dose ladder per direction — ≥5 points (the issue's `min_doses` floor).
pub const DOSE_LADDER: [f32; 5] = [0.2, 0.4, 0.6, 0.8, 1.0];
/// Held-out reads behind every effect number (the `min_reads` floor).
pub const MIN_READS: usize = 8;
/// Matched-random controls per candidate (the issue's
/// `min_control_neurons` floor — also `census_screen::MIN_CONTROL_UNITS`).
pub const MIN_CONTROLS: usize = MIN_CONTROL_UNITS;

/// Canonical artifact magic ("NEUCEN1") + format version byte.
pub const ARTIFACT_MAGIC: &[u8; 8] = b"NEUCEN1\x01";

/// Census sweep configuration. [`Default`] pins the pre-registered bars.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CensusConfig {
    /// Push scale (the suppress arm is its reciprocal).
    pub push_scale: f32,
    /// Absolute effect floor (observable units; pre-registered per
    /// fixture — the G1 bar is `max(abs_floor, band_ratio × band mean)`).
    pub abs_floor: f64,
    /// Effect must reach `band_ratio × mean |control|` (Fermion's 40-vs-1.4
    /// ≈ 28× scale, pre-registered per fixture, never portable).
    pub band_ratio: f64,
    /// Control-draw seed base (per-candidate seeds fold the column id).
    pub seed: u64,
}

impl Default for CensusConfig {
    fn default() -> Self {
        Self {
            push_scale: PUSH_SCALE,
            abs_floor: 0.05,
            band_ratio: 28.0,
            seed: 0x0001_241C_5E51_C001,
        }
    }
}

impl CensusConfig {
    /// The control-verdict params this config adjudicates with.
    #[inline]
    pub fn control_params(&self) -> ControlParams {
        let mut p = ControlParams::new(MIN_READS, self.abs_floor, self.band_ratio);
        p.min_controls = MIN_CONTROLS;
        p
    }
}

/// One census layer: the model's layer id and its neuron count (the
/// `down_proj` input width — one candidate per column).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CensusLayer {
    pub layer: u8,
    pub neurons: usize,
}

/// The model the census sweeps. Implementations own their forwards and
/// scratch; the runner never allocates per candidate (the capture loop
/// is read → push → read over the model's reused buffers). `Send` so
/// the parallel driver can move one session per worker thread (the
/// adapter's push must be weight-neutral for that to be sound — the
/// qwen35 adapter's activation-scale push is; a weight-mutating
/// implementation is only sound under `run_sweep`'s serial driver).
pub trait CensusModel: Send {
    /// Identity of the swept artifact (weights digest) — the cursor and
    /// the artifact bind it.
    fn model_digest(&self) -> [u8; 32];
    /// The sweep's layer table.
    fn layers(&self) -> &[CensusLayer];
    /// Base observable for held-out read `read` (a per-sample scalar —
    /// the answer-token logit, a probe readout; fixture-defined).
    fn base_observable(&mut self, read: usize) -> f64;
    /// Push column `column` of `layer_index` with multiplier `f`, then
    /// measure the same observable. Implementations must leave the base
    /// weights UNCHANGED between calls (scratch-copy or revert-inside).
    fn push_observable(&mut self, layer_index: usize, column: u32, f: f32, read: usize) -> f64;
}

/// Typed census verdict (mirrors `ControlVerdict` minus the band
/// payload — the band numbers are columns of their own).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum CensusVerdict {
    SwitchFound = 1,
    NoSwitch = 2,
    Undecided = 3,
}

/// Per-neuron census row — the artifact's payload.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CensusRow {
    pub layer: u8,
    pub column: u32,
    /// Direction sign of the best arm: `+1` push (`f = s·dose`), `-1`
    /// suppress (`f = dose/s`).
    pub direction_sign: i8,
    /// Best dose (the ladder point that produced `effect`).
    pub best_dose: f32,
    /// Best |observable delta| over the grid × reads.
    pub effect: f64,
    pub verdict: CensusVerdict,
    pub band_mean: f64,
    pub band_max: f64,
}

/// Accumulating census — rows land in sweep order; the artifact
/// canonicalizes (sorts by (layer, column)) at mint.
#[derive(Debug, Default)]
pub struct CensusSweep {
    pub rows: Vec<CensusRow>,
}

/// The winning arm's multiplier: push `f = s·dose`, suppress `f = dose/s`
/// (both endpoints exact at dose 1 by the fader lerp law).
#[inline]
fn arm_factor(sign: i8, dose: f32, s: f32) -> f32 {
    if sign > 0 { s * dose } else { dose / s }
}

/// The matched-random control push closure: measure one control
/// column at exactly the candidate's winning multiplier.
pub type ControlEffectFn<'a> =
    &'a mut dyn FnMut(&mut dyn CensusModel, usize, u32, f32, usize) -> f64;

/// One census candidate's coordinates: which layer (index + table row),
/// which column, and its drawn control set. Grouped so the
/// adjudication takes one context argument, not a ladder of scalars.
pub struct Candidate<'a> {
    pub layer_index: usize,
    pub layer: &'a CensusLayer,
    pub column: u32,
    pub control_ids: &'a [usize],
}

impl CensusSweep {
    /// Adjudicate one candidate: best arm over the dose grid × reads,
    /// then the matched-random band at the winner's arm.
    ///
    /// `control_ids` are disjoint column ids in the same layer (the
    /// runner supplies the deterministic draw); `control_effect` must
    /// measure a control column at EXACTLY the candidate's winning
    /// multiplier. The CAPTURE loop (base read → push read) allocates
    /// nothing; the verdict-side scratch (control effects, the draw
    /// output) is one small Vec each per candidate — T2's G4 cell
    /// prices whether they move to caller-owned scratch.
    pub fn adjudicate_candidate(
        &mut self,
        model: &mut dyn CensusModel,
        cand: Candidate<'_>,
        reads: usize,
        cfg: &CensusConfig,
        control_effect: ControlEffectFn<'_>,
    ) -> Result<()> {
        let Candidate {
            layer_index,
            layer,
            column,
            control_ids,
        } = cand;
        if reads < MIN_READS {
            bail!(
                "reads {reads} below the MIN_READS floor {MIN_READS} — no verdict, never a negative"
            );
        }
        if control_ids.len() < MIN_CONTROLS {
            bail!(
                "controls {} below the MIN_CONTROLS floor {MIN_CONTROLS} — no verdict, never a negative",
                control_ids.len()
            );
        }
        let s = cfg.push_scale;
        // Best arm over (direction × dose): the arm-dose mean over reads,
        // sequential f64 fold, strictly-greater updates (first-index ties).
        let mut best_effect = 0.0f64;
        let mut best_sign = 1i8;
        let mut best_dose = 0.0f32;
        let mut first = true;
        for sign in [1i8, -1i8] {
            for &dose in DOSE_LADDER.iter() {
                let f = arm_factor(sign, dose, s);
                let mut acc = 0.0f64;
                for read in 0..reads {
                    let base = model.base_observable(read);
                    let pushed = model.push_observable(layer_index, column, f, read);
                    acc += (pushed - base).abs();
                }
                let mean = acc / reads as f64;
                if first || mean > best_effect {
                    best_effect = mean;
                    best_sign = sign;
                    best_dose = dose;
                    first = false;
                }
            }
        }

        // Matched-random band at the winner's arm — every control pushed
        // exactly as hard as the candidate was.
        let f = arm_factor(best_sign, best_dose, s);
        let mut control_effects = Vec::with_capacity(control_ids.len());
        for &cid in control_ids {
            if cid as u32 == column {
                bail!("control draw collided with the candidate — the draw is tampered");
            }
            let mut acc = 0.0f64;
            for read in 0..reads {
                let base = model.base_observable(read);
                let pushed = control_effect(&mut *model, layer_index, cid as u32, f, read);
                acc += (pushed - base).abs();
            }
            control_effects.push(acc / reads as f64);
        }

        let verdict = control_verdict(best_effect, reads, &control_effects, &cfg.control_params());
        let (v, band_mean, band_max) = match verdict {
            ControlVerdict::SwitchFound { band, .. } => {
                (CensusVerdict::SwitchFound, band.mean_abs, band.max_abs)
            }
            ControlVerdict::NoSwitch { band_measured, .. } => (
                CensusVerdict::NoSwitch,
                band_measured.mean_abs,
                band_measured.max_abs,
            ),
            ControlVerdict::Undecided { .. } => (CensusVerdict::Undecided, f64::NAN, f64::NAN),
        };
        self.rows.push(CensusRow {
            layer: layer.layer,
            column,
            direction_sign: best_sign,
            best_dose,
            effect: best_effect,
            verdict: v,
            band_mean,
            band_max,
        });
        Ok(())
    }
}

/// Deterministic per-candidate control draw: SplitMix64 seeded with
/// `seed ^ (layer << 32) ^ column·φ`, Fisher–Yates over the layer's
/// column pool (the pool scratch is reused across candidates — zero
/// steady-state allocation), first `MIN_CONTROLS` survivors disjoint
/// from the candidate.
pub struct ControlDraw {
    pool: Vec<usize>,
}

impl ControlDraw {
    pub fn new(layer_neurons: usize) -> Self {
        Self {
            pool: (0..layer_neurons).collect(),
        }
    }

    pub fn draw(&mut self, cfg: &CensusConfig, layer_index: usize, column: u32) -> Vec<usize> {
        // The draw is a PURE function of (seed, layer_index, column):
        // the pool scratch is reset to identity before each shuffle, so
        // a resumed sweep reproduces the uninterrupted run's control
        // sets bit-identically (draw history must not matter).
        for (i, p) in self.pool.iter_mut().enumerate() {
            *p = i;
        }
        let mut state = cfg.seed
            ^ ((layer_index as u64) << 32)
            ^ (column as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        for i in (1..self.pool.len()).rev() {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            let j = (z ^ (z >> 31)) as usize % (i + 1);
            self.pool.swap(i, j);
        }
        let mut out = Vec::with_capacity(MIN_CONTROLS);
        for &p in self.pool.iter() {
            if p as u32 != column {
                out.push(p);
                if out.len() == MIN_CONTROLS {
                    break;
                }
            }
        }
        out
    }
}

/// The sweep cursor — canonical text in the durable state dir
/// (`v1 <digest-hex> <layer_index> <next_column>`); the model digest
/// binds it, a mismatched cursor refuses loud.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CensusCursor {
    pub layer_index: usize,
    pub next_column: usize,
}

/// The full sweep, resumable. `state_dir` holds the atomic cursor plus
/// the canonical rows ledger (`rows.bin`, fixed 35-byte stride, one
/// appended row per completed candidate): the cursor rename is the
/// commit point — on load, rows beyond the cursor position are
/// discarded (a kill mid-append may leave a partial tail; a cursor is
/// model-bound, a mismatching one refuses loud).
pub fn run_sweep(
    model: &mut dyn CensusModel,
    cfg: &CensusConfig,
    state_dir: Option<&std::path::Path>,
    resume: bool,
    progress: &mut dyn FnMut(usize, usize, CensusCursor),
) -> Result<CensusSweep> {
    let layers: Vec<CensusLayer> = model.layers().to_vec();
    let digest_hex = hex(&model.model_digest());
    let (mut sweep, mut cursor, cursor_path, rows_path) = match state_dir {
        Some(dir) => {
            let (s, c, cp, rp) = open_census_state(dir, &digest_hex, &layers, resume, None)?;
            (s, c, Some(cp), Some(rp))
        }
        None => (
            CensusSweep::default(),
            CensusCursor {
                layer_index: 0,
                next_column: 0,
            },
            None,
            None,
        ),
    };

    let mut draw = ControlDraw::new(layers.first().map_or(0, |l| l.neurons));
    for (li, layer) in layers.iter().enumerate().skip(cursor.layer_index) {
        if draw.pool.len() != layer.neurons {
            draw = ControlDraw::new(layer.neurons);
        }
        let start = if li == cursor.layer_index {
            cursor.next_column
        } else {
            0
        };
        for column in start..layer.neurons {
            let column = column as u32;
            // The seed folds the REAL layer id, not the table position —
            // identical here (an unfiltered table's ids are its positions)
            // and posture-invariant under a `--layers` filter (the parallel
            // driver's filtered tables would otherwise draw different
            // controls for the same candidate).
            let ids = draw.draw(cfg, usize::from(layer.layer), column);
            sweep.adjudicate_candidate(
                model,
                Candidate {
                    layer_index: li,
                    layer,
                    column,
                    control_ids: &ids,
                },
                MIN_READS,
                cfg,
                &mut |m, li, c, f, read| m.push_observable(li, c, f, read),
            )?;
            cursor = CensusCursor {
                layer_index: li,
                next_column: column as usize + 1,
            };
            if let (Some(rp), Some(cp)) = (&rows_path, &cursor_path) {
                let row = *sweep.rows.last().expect("adjudication just pushed one row");
                append_row(rp, &row)?;
                write_cursor(cp, &cursor, &digest_hex)?;
            }
            progress(li, layer.neurons, cursor);
        }
    }
    Ok(sweep)
}

/// Global completed-candidate count at a cursor (whole layers before
/// it plus its in-layer column position), under the driver's per-layer
/// candidate `limit` (a limited posture's widths are the layer widths
/// clamped to it; `None` = the full table).
fn global_done(cursor: &CensusCursor, layers: &[CensusLayer], limit: Option<usize>) -> usize {
    let mut n = 0usize;
    for layer in layers.iter().take(cursor.layer_index) {
        n += limited_width(layer.neurons, limit);
    }
    let cur_w = layers
        .get(cursor.layer_index)
        .map_or(0, |l| limited_width(l.neurons, limit));
    n + cursor.next_column.min(cur_w)
}

/// A layer's sweep width under the per-layer candidate limit.
fn limited_width(neurons: usize, limit: Option<usize>) -> usize {
    match limit {
        Some(l) => neurons.min(l),
        None => neurons,
    }
}

/// The flattened candidate position → `(table layer index, column)`.
/// `lw` is the per-layer limited widths; a position past the table is
/// unreachable by construction (the callers pull `g < total`).
fn locate_candidate(g: usize, lw: &[usize]) -> (usize, usize) {
    let mut rest = g;
    for (li, &w) in lw.iter().enumerate() {
        if rest < w {
            return (li, rest);
        }
        rest -= w;
    }
    unreachable!(
        "candidate position {g} past the flattened table (total {})",
        lw.iter().sum::<usize>()
    );
}

/// Open (or resume) a census state dir — the ONE home of the durable
/// protocol both drivers share: the atomic cursor plus the canonical
/// rows ledger. The cursor rename is the commit point; on resume, rows
/// beyond the cursor are discarded (a kill mid-append may leave a
/// partial tail) and a mismatching cursor refuses loud. A fresh dir is
/// initialized (cursor written, ledger truncated) whether or not
/// `resume` was asked — `resume` with no cursor present is a fresh run.
fn open_census_state(
    dir: &std::path::Path,
    digest_hex: &str,
    layers: &[CensusLayer],
    resume: bool,
    limit: Option<usize>,
) -> Result<(
    CensusSweep,
    CensusCursor,
    std::path::PathBuf,
    std::path::PathBuf,
)> {
    std::fs::create_dir_all(dir).with_context(|| format!("census state dir {}", dir.display()))?;
    let p = dir.join("cursor");
    let r = dir.join("rows.bin");
    let mut sweep = CensusSweep::default();
    let mut cursor = CensusCursor {
        layer_index: 0,
        next_column: 0,
    };
    if resume && p.exists() {
        let text = std::fs::read_to_string(&p)
            .with_context(|| format!("reading cursor {}", p.display()))?;
        cursor = parse_cursor(&text, digest_hex)
            .with_context(|| format!("cursor {} is stale or malformed", p.display()))?;
        let keep = global_done(&cursor, layers, limit);
        let prior = load_rows(&r, keep).with_context(|| format!("rows ledger {}", r.display()))?;
        if prior.len() != keep {
            bail!(
                "rows ledger holds {} rows but the cursor commits {} — state is inconsistent",
                prior.len(),
                keep
            );
        }
        sweep.rows = prior;
    } else {
        write_cursor(&p, &cursor, digest_hex)?;
        std::fs::write(&r, b"")
            .with_context(|| format!("truncating rows ledger {}", r.display()))?;
    }
    Ok((sweep, cursor, p, r))
}

/// The parallel census sweep driver — the serial [`run_sweep`]'s row
/// semantics on many cores. Workers pull CHUNKS of consecutive
/// candidates from the serial order (the flatten of the model's layer
/// table, column-major within a layer) and adjudicate them on their own
/// [`CensusModel`] session; rows stream back per candidate and the main
/// thread commits ONLY the in-order prefix — ledger append, cursor
/// advance, `progress` — so the durable state and the kill/resume
/// semantics are byte-identical to the serial sweep's (a resume
/// continues at the committed cursor; in-flight candidates are simply
/// re-run). Determinism: the control draw is a pure function of
/// (seed, layer id, column) and the adjudication folds are sequential
/// per candidate — the parallel rows equal the serial rows bit-for-bit
/// (pinned by the fixture's par/serial twin gates).
///
/// `models` supplies one session per worker (the caller builds them —
/// typically cheap handles over shared weights); `limit` caps every
/// layer's candidates (the smoke posture; a limited run is never
/// resumed — the cursor under a different limit refuses loud via the
/// keep check).
pub fn run_sweep_par(
    models: Vec<Box<dyn CensusModel>>,
    chunk: usize,
    cfg: &CensusConfig,
    state_dir: &std::path::Path,
    resume: bool,
    limit: Option<usize>,
    progress: &mut dyn FnMut(usize, usize, CensusCursor),
) -> Result<CensusSweep> {
    if models.is_empty() {
        bail!("run_sweep_par needs at least one worker session");
    }
    if chunk == 0 {
        bail!("chunk must be >= 1");
    }
    let layers: Vec<CensusLayer> = models[0].layers().to_vec();
    let digest_hex = hex(&models[0].model_digest());
    let lw: Vec<usize> = layers
        .iter()
        .map(|l| limited_width(l.neurons, limit))
        .collect();
    let total: usize = lw.iter().sum();
    let (mut sweep, mut cursor, cursor_path, rows_path) =
        open_census_state(state_dir, &digest_hex, &layers, resume, limit)?;
    let mut committed = global_done(&cursor, &layers, limit);
    if committed >= total {
        return Ok(sweep);
    }

    let next = std::sync::atomic::AtomicUsize::new(committed);
    let abort = std::sync::atomic::AtomicBool::new(false);
    let (tx, rx) = std::sync::mpsc::channel::<(usize, Result<CensusRow, anyhow::Error>)>();
    let mut first_err: Option<(usize, anyhow::Error)> = None;
    let mut io_err: Option<anyhow::Error> = None;
    let mut pending: std::collections::BTreeMap<usize, CensusRow> = Default::default();

    std::thread::scope(|scope| {
        let lw = &lw;
        let layers = &layers;
        let total = &total;
        let cfg = &cfg;
        for mut model in models {
            let tx = tx.clone();
            let next = &next;
            let abort = &abort;
            scope.spawn(move || {
                let mut draw = ControlDraw::new(0);
                let mut scratch = CensusSweep::default();
                loop {
                    if abort.load(std::sync::atomic::Ordering::Relaxed) {
                        break;
                    }
                    let start = next.fetch_add(chunk, std::sync::atomic::Ordering::Relaxed);
                    if start >= *total {
                        break;
                    }
                    let end = (start + chunk).min(*total);
                    for g in start..end {
                        let (li, col) = locate_candidate(g, lw);
                        let layer = &layers[li];
                        if draw.pool.len() != layer.neurons {
                            draw = ControlDraw::new(layer.neurons);
                        }
                        let ids = draw.draw(cfg, usize::from(layer.layer), col as u32);
                        let r = scratch.adjudicate_candidate(
                            model.as_mut(),
                            Candidate {
                                layer_index: li,
                                layer,
                                column: col as u32,
                                control_ids: &ids,
                            },
                            MIN_READS,
                            cfg,
                            &mut |m, li, c, f, read| m.push_observable(li, c, f, read),
                        );
                        let row = scratch.rows.pop();
                        debug_assert!(
                            r.is_ok() || row.is_none(),
                            "a failed adjudication must not leave a row"
                        );
                        let _ = tx.send((
                            g,
                            r.map(|_| row.expect("a passed adjudication pushed exactly one row")),
                        ));
                    }
                }
            });
        }
        drop(tx); // the workers hold the clones; recv ends when they all finish
        while committed < *total && first_err.is_none() {
            let (g, r) = match rx.recv() {
                Ok(x) => x,
                Err(_) => break, // every sender gone (workers finished)
            };
            match r {
                Ok(row) => {
                    pending.insert(g, row);
                }
                Err(e) => {
                    first_err = Some((g, e));
                    abort.store(true, std::sync::atomic::Ordering::Relaxed);
                    break;
                }
            }
            // Commit the in-order prefix the new row may have completed.
            // A durable-state IO failure is stored, not `?`-returned (the
            // scope closure returns ()); the run aborts and the caller
            // refuses after the workers join — the committed prefix is
            // exactly the ledger, so the run is resumable.
            while io_err.is_none()
                && let Some(row) = pending.remove(&committed)
            {
                let (li, col) = locate_candidate(committed, lw);
                cursor = CensusCursor {
                    layer_index: li,
                    next_column: col + 1,
                };
                if let Err(e) = append_row(&rows_path, &row)
                    .and_then(|()| write_cursor(&cursor_path, &cursor, &digest_hex))
                {
                    io_err = Some(e);
                    abort.store(true, std::sync::atomic::Ordering::Relaxed);
                    break;
                }
                sweep.rows.push(row);
                committed += 1;
                progress(li, layers[li].neurons, cursor);
            }
            if io_err.is_some() {
                break;
            }
        }
    });

    if let Some(e) = io_err {
        bail!("census durable state failed at {committed} committed candidates: {e}");
    }
    if let Some((g, e)) = first_err {
        let (li, col) = locate_candidate(g, &lw);
        bail!(
            "candidate {g} (layer {} column {col}) failed: {e}",
            layers[li].layer
        );
    }
    Ok(sweep)
}

/// Canonical artifact: header + rows sorted by (layer, column) + BLAKE3
/// digest trailer. Every float is its bit pattern (little-endian); two
/// runs on the same model + config are byte-identical.
pub fn mint_artifact(sweep: &CensusSweep, digest: &[u8; 32], cfg: &CensusConfig) -> Vec<u8> {
    let mut rows: Vec<&CensusRow> = sweep.rows.iter().collect();
    rows.sort_by_key(|r| (r.layer, r.column));
    let mut out = Vec::with_capacity(64 + rows.len() * 56);
    out.extend_from_slice(ARTIFACT_MAGIC);
    out.extend_from_slice(digest);
    out.extend_from_slice(&cfg.push_scale.to_bits().to_le_bytes());
    out.extend_from_slice(&cfg.abs_floor.to_bits().to_le_bytes());
    out.extend_from_slice(&cfg.band_ratio.to_bits().to_le_bytes());
    out.extend_from_slice(&cfg.seed.to_le_bytes());
    out.extend_from_slice(&(rows.len() as u32).to_le_bytes());
    for r in rows {
        out.push(r.layer);
        out.extend_from_slice(&r.column.to_le_bytes());
        out.push(r.direction_sign as u8);
        out.extend_from_slice(&r.best_dose.to_bits().to_le_bytes());
        out.extend_from_slice(&r.effect.to_bits().to_le_bytes());
        out.push(r.verdict as u8);
        out.extend_from_slice(&r.band_mean.to_bits().to_le_bytes());
        out.extend_from_slice(&r.band_max.to_bits().to_le_bytes());
    }
    let h = blake3::hash(&out);
    out.extend_from_slice(h.as_bytes());
    out
}

/// Verify an artifact's BLAKE3 trailer (the digest is the LAST 32
/// bytes; everything before it is the hashed payload). Returns the
/// payload length, or None on any mismatch/truncation.
pub fn verify_artifact(bytes: &[u8]) -> Option<usize> {
    if bytes.len() < 32 + 32 + 8 {
        return None;
    }
    let (payload, trailer) = bytes.split_at(bytes.len() - 32);
    if blake3::hash(payload).as_bytes() != trailer {
        return None;
    }
    if &payload[0..8] != ARTIFACT_MAGIC {
        return None;
    }
    Some(payload.len())
}

// ── artifact reader + the access-check set operation (T2) ───────────

/// Byte stride of one canonical artifact row (mint and parse agree —
/// a mismatch either way corrupts every row after the first).
const ARTIFACT_ROW_STRIDE: usize = 35;
/// Header size: magic 8 + digest 32 + push_scale f32 4 + abs_floor
/// f64 8 + band_ratio f64 8 + seed u64 8 + row count u32 4.
const ARTIFACT_HEADER_LEN: usize = 72;

/// The decoded artifact header — identity + the config the sweep ran
/// under. Two artifacts are access-check comparable ONLY under the
/// same config (floors/bands/seed differ ⇒ verdicts are not the same
/// measurement); the weights digests may (and for base-vs-chat, must)
/// differ.
#[derive(Clone, Debug, PartialEq)]
pub struct CensusArtifactHeader {
    pub weights_digest: [u8; 32],
    pub push_scale: f32,
    pub abs_floor: f64,
    pub band_ratio: f64,
    pub seed: u64,
    pub n_rows: usize,
}

impl CensusArtifactHeader {
    /// The comparable-science tuple — everything but the model
    /// identity. Equality is required before a set operation.
    fn config_key(&self) -> (u64, u64, u64, u64) {
        (
            u64::from(self.push_scale.to_bits()),
            self.abs_floor.to_bits(),
            self.band_ratio.to_bits(),
            self.seed,
        )
    }
}

/// Parse a minted artifact: verify the trailer, decode the header and
/// every row. Refuses loud on any truncation, unknown verdict byte,
/// or payload-length mismatch (strict — no trailing junk).
pub fn parse_artifact(bytes: &[u8]) -> Result<(CensusArtifactHeader, Vec<CensusRow>)> {
    let payload_len =
        verify_artifact(bytes).context("census artifact trailer/canonicality check failed")?;
    let payload = &bytes[..payload_len];
    let n_rows = u32::from_le_bytes(payload[68..72].try_into().expect("fixed-size slice")) as usize;
    let expected = ARTIFACT_HEADER_LEN + n_rows * ARTIFACT_ROW_STRIDE;
    if payload_len != expected {
        bail!(
            "census artifact holds {} rows but the payload is {} bytes (expected {}) — truncated or padded",
            n_rows,
            payload_len,
            expected
        );
    }
    let header = CensusArtifactHeader {
        weights_digest: payload[8..40].try_into().expect("fixed-size slice"),
        push_scale: f32::from_le_bytes(payload[40..44].try_into().expect("fixed-size slice")),
        abs_floor: f64::from_le_bytes(payload[44..52].try_into().expect("fixed-size slice")),
        band_ratio: f64::from_le_bytes(payload[52..60].try_into().expect("fixed-size slice")),
        seed: u64::from_le_bytes(payload[60..68].try_into().expect("fixed-size slice")),
        n_rows,
    };
    let mut rows = Vec::with_capacity(n_rows);
    let mut off = ARTIFACT_HEADER_LEN;
    for i in 0..n_rows {
        let r = &payload[off..off + ARTIFACT_ROW_STRIDE];
        let verdict_byte = r[18];
        let verdict = match verdict_byte {
            1 => CensusVerdict::SwitchFound,
            2 => CensusVerdict::NoSwitch,
            3 => CensusVerdict::Undecided,
            other => bail!(
                "census artifact row {i} carries unknown verdict byte {other} — not a minted artifact"
            ),
        };
        rows.push(CensusRow {
            layer: r[0],
            column: u32::from_le_bytes(r[1..5].try_into().expect("fixed-size slice")),
            direction_sign: r[5] as i8,
            best_dose: f32::from_le_bytes(r[6..10].try_into().expect("fixed-size slice")),
            effect: f64::from_le_bytes(r[10..18].try_into().expect("fixed-size slice")),
            verdict,
            band_mean: f64::from_le_bytes(r[19..27].try_into().expect("fixed-size slice")),
            band_max: f64::from_le_bytes(r[27..35].try_into().expect("fixed-size slice")),
        });
        off += ARTIFACT_ROW_STRIDE;
    }
    Ok((header, rows))
}

/// Which side of an access check a site was found in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccessSide {
    Base,
    Chat,
}

impl std::fmt::Display for AccessSide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AccessSide::Base => write!(f, "base"),
            AccessSide::Chat => write!(f, "chat"),
        }
    }
}

/// One (layer, column) site's access-check classification.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AccessClass {
    /// Switched in BOTH models above the noise band — a shared
    /// sensitive site (the authoring did not move it).
    Replicated,
    /// Switched in one model only (above the band) — the other is
    /// inert at this site. This is the authoring-prior signal.
    PresentInert { present_in: AccessSide, effect: f64 },
    /// A classification quant-fragile under the noise band (Bench
    /// 1252: Q8_0-vs-fp16 KL noise mean 4.81e-4, max 5.2e-3 — the
    /// census floor 1e-4 sits INSIDE that noise). Carried typed,
    /// never pooled with the confident classes.
    NoiseMarginal,
}

/// One classified site with its coordinates + the effects it carried.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AccessSite {
    pub layer: u8,
    pub column: u32,
    pub class: AccessClass,
    /// Effect in whichever side(s) held the row (NaN where absent).
    pub effect_base: f64,
    pub effect_chat: f64,
}

/// The access-check report: the set operation over two census
/// artifacts' switch-sets (rows verdicted `SwitchFound`), typed per
/// the issue's G1 fixture vocabulary. `Undecided` rows never enter a
/// set; `NoSwitch` rows are confident negatives on their own side.
#[derive(Clone, Debug, Default)]
pub struct AccessCheckReport {
    pub weights_digest_base: [u8; 32],
    pub weights_digest_chat: [u8; 32],
    pub noise_band: f64,
    /// Switch-set sizes BEFORE band typing.
    pub switches_base: usize,
    pub switches_chat: usize,
    pub replicated: usize,
    pub present_inert_base: usize,
    pub present_inert_chat: usize,
    pub noise_marginal: usize,
    /// `Undecided`-verdict row counts (absence of evidence — never a
    /// set member, disclosed here).
    pub undecided_base: usize,
    pub undecided_chat: usize,
    /// Full classification table, sorted by (layer, column).
    pub sites: Vec<AccessSite>,
}

impl AccessCheckReport {
    /// Confident present-inert rows sorted strongest-first (the
    /// authoring-prior feed for T3/T4).
    pub fn authoring_prior(&self) -> Vec<&AccessSite> {
        let mut rows: Vec<&AccessSite> = self
            .sites
            .iter()
            .filter(|s| {
                matches!(
                    s.class,
                    AccessClass::PresentInert { .. } | AccessClass::Replicated
                )
            })
            .collect();
        rows.sort_by(|a, b| {
            let ka = a.effect_base.max(a.effect_chat);
            let kb = b.effect_base.max(b.effect_chat);
            kb.partial_cmp(&ka).unwrap_or(std::cmp::Ordering::Equal)
        });
        rows
    }
}

/// The access check: parse-free entry taking the two minted artifacts
/// (bytes) + the noise band in nats. Refuses loud when the two
/// artifacts were swept under DIFFERENT configs (floor/band/scale/
/// seed — the verdicts would not be the same measurement). The two
/// weights digests are reported, never compared.
pub fn access_check(
    base_bytes: &[u8],
    chat_bytes: &[u8],
    noise_band: f64,
) -> Result<AccessCheckReport> {
    let (h_base, rows_base) = parse_artifact(base_bytes).context("base artifact")?;
    let (h_chat, rows_chat) = parse_artifact(chat_bytes).context("chat artifact")?;
    if h_base.config_key() != h_chat.config_key() {
        bail!(
            "census artifacts were swept under different configs (base: scale={} floor={:?} band={:?} seed={}; chat: scale={} floor={:?} band={:?} seed={}) — the switch-sets are not comparable",
            h_base.push_scale,
            h_base.abs_floor,
            h_base.band_ratio,
            h_base.seed,
            h_chat.push_scale,
            h_chat.abs_floor,
            h_chat.band_ratio,
            h_chat.seed,
        );
    }
    if !(noise_band.is_finite() && noise_band > 0.0) {
        bail!("noise band must be a positive finite number of nats, got {noise_band}");
    }

    // Switch-sets keyed by (layer, column); Undecided disclosed, never
    // a set member. Rows are sorted at mint, so first-wins is the
    // canonical row if a key somehow repeats.
    let mut base_map: std::collections::HashMap<(u8, u32), &CensusRow> =
        std::collections::HashMap::new();
    let mut chat_map: std::collections::HashMap<(u8, u32), &CensusRow> =
        std::collections::HashMap::new();
    let (mut undecided_base, mut undecided_chat) = (0usize, 0usize);
    let (mut switches_base, mut switches_chat) = (0usize, 0usize);
    for r in &rows_base {
        match r.verdict {
            CensusVerdict::SwitchFound => {
                switches_base += 1;
                base_map.entry((r.layer, r.column)).or_insert(r);
            }
            CensusVerdict::Undecided => undecided_base += 1,
            CensusVerdict::NoSwitch => {}
        }
    }
    for r in &rows_chat {
        match r.verdict {
            CensusVerdict::SwitchFound => {
                switches_chat += 1;
                chat_map.entry((r.layer, r.column)).or_insert(r);
            }
            CensusVerdict::Undecided => undecided_chat += 1,
            CensusVerdict::NoSwitch => {}
        }
    }

    // Union walk in canonical (layer, column) order.
    let mut keys: Vec<(u8, u32)> = base_map.keys().chain(chat_map.keys()).copied().collect();
    keys.sort_unstable();
    keys.dedup();

    let mut report = AccessCheckReport {
        weights_digest_base: h_base.weights_digest,
        weights_digest_chat: h_chat.weights_digest,
        noise_band,
        switches_base,
        switches_chat,
        undecided_base,
        undecided_chat,
        ..Default::default()
    };

    for key in keys {
        let b = base_map.get(&key).copied();
        let c = chat_map.get(&key).copied();
        let eb = b.map(|r| r.effect).unwrap_or(f64::NAN);
        let ec = c.map(|r| r.effect).unwrap_or(f64::NAN);
        let class = match (b, c) {
            (Some(b), Some(c)) => {
                if b.effect >= noise_band && c.effect >= noise_band {
                    AccessClass::Replicated
                } else {
                    AccessClass::NoiseMarginal
                }
            }
            (Some(b), None) => {
                if b.effect >= noise_band {
                    AccessClass::PresentInert {
                        present_in: AccessSide::Base,
                        effect: b.effect,
                    }
                } else {
                    AccessClass::NoiseMarginal
                }
            }
            (None, Some(c)) => {
                if c.effect >= noise_band {
                    AccessClass::PresentInert {
                        present_in: AccessSide::Chat,
                        effect: c.effect,
                    }
                } else {
                    AccessClass::NoiseMarginal
                }
            }
            (None, None) => unreachable!("key came from one of the maps"),
        };
        match class {
            AccessClass::Replicated => report.replicated += 1,
            AccessClass::PresentInert { present_in, .. } => match present_in {
                AccessSide::Base => report.present_inert_base += 1,
                AccessSide::Chat => report.present_inert_chat += 1,
            },
            AccessClass::NoiseMarginal => report.noise_marginal += 1,
        }
        report.sites.push(AccessSite {
            layer: key.0,
            column: key.1,
            class,
            effect_base: eb,
            effect_chat: ec,
        });
    }
    Ok(report)
}

// ── rows ledger + cursor plumbing ────────────────────────────────────

/// One canonical census row, fixed stride.
const ROW_STRIDE: usize = 35;

fn encode_row(r: &CensusRow) -> [u8; ROW_STRIDE] {
    let mut b = [0u8; ROW_STRIDE];
    b[0] = r.layer;
    b[1..5].copy_from_slice(&r.column.to_le_bytes());
    b[5] = r.direction_sign as u8;
    b[6..10].copy_from_slice(&r.best_dose.to_bits().to_le_bytes());
    b[10..18].copy_from_slice(&r.effect.to_bits().to_le_bytes());
    b[18] = r.verdict as u8;
    b[19..27].copy_from_slice(&r.band_mean.to_bits().to_le_bytes());
    b[27..35].copy_from_slice(&r.band_max.to_bits().to_le_bytes());
    b
}

fn decode_row(b: &[u8]) -> Option<CensusRow> {
    if b.len() < ROW_STRIDE {
        return None;
    }
    let verdict = match b[18] {
        1 => CensusVerdict::SwitchFound,
        2 => CensusVerdict::NoSwitch,
        3 => CensusVerdict::Undecided,
        _ => return None,
    };
    let sign = match b[5] {
        255 => -1i8,
        1 => 1i8,
        _ => return None,
    };
    Some(CensusRow {
        layer: b[0],
        column: u32::from_le_bytes(b[1..5].try_into().ok()?),
        direction_sign: sign,
        best_dose: f32::from_bits(u32::from_le_bytes(b[6..10].try_into().ok()?)),
        effect: f64::from_bits(u64::from_le_bytes(b[10..18].try_into().ok()?)),
        verdict,
        band_mean: f64::from_bits(u64::from_le_bytes(b[19..27].try_into().ok()?)),
        band_max: f64::from_bits(u64::from_le_bytes(b[27..35].try_into().ok()?)),
    })
}

fn append_row(path: &std::path::Path, row: &CensusRow) -> Result<()> {
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .append(true)
        .open(path)
        .with_context(|| format!("opening rows ledger {}", path.display()))?;
    f.write_all(&encode_row(row))
        .with_context(|| format!("appending row to {}", path.display()))?;
    Ok(())
}

fn load_rows(path: &std::path::Path, keep: usize) -> Result<Vec<CensusRow>> {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e).with_context(|| format!("reading rows ledger {}", path.display())),
    };
    // Floor to the stride — a kill mid-append leaves a partial tail;
    // the cursor (not the file length) is the commit point.
    let whole = bytes.len() / ROW_STRIDE;
    let mut rows = Vec::with_capacity(whole.min(keep));
    for i in 0..whole.min(keep) {
        let row = decode_row(&bytes[i * ROW_STRIDE..(i + 1) * ROW_STRIDE])
            .with_context(|| format!("rows ledger {} row {i} malformed", path.display()))?;
        rows.push(row);
    }
    Ok(rows)
}

fn write_cursor(path: &std::path::Path, cursor: &CensusCursor, digest_hex: &str) -> Result<()> {
    // temp-then-rename (atomic; the cursor line is the whole state — a
    // torn read refuses at parse).
    let tmp = path.with_extension("tmp");
    std::fs::write(
        &tmp,
        format!(
            "v1 {} {} {}\n",
            digest_hex, cursor.layer_index, cursor.next_column
        ),
    )
    .with_context(|| format!("writing cursor {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("renaming cursor {}", path.display()))?;
    Ok(())
}

fn parse_cursor(text: &str, digest_hex: &str) -> Result<CensusCursor> {
    let line = text.trim();
    let mut it = line.split_whitespace();
    let v = it.next().unwrap_or("");
    if v != "v1" {
        bail!("cursor version {v:?} unsupported");
    }
    let d = it.next().unwrap_or("");
    if d != digest_hex {
        bail!("cursor model digest {d} != swept digest {digest_hex}");
    }
    let li: usize = it
        .next()
        .unwrap_or("")
        .parse()
        .map_err(|_| anyhow::anyhow!("cursor layer index malformed"))?;
    let nc: usize = it
        .next()
        .unwrap_or("")
        .parse()
        .map_err(|_| anyhow::anyhow!("cursor next-column malformed"))?;
    if it.next().is_some() {
        bail!("cursor line has trailing fields");
    }
    Ok(CensusCursor {
        layer_index: li,
        next_column: nc,
    })
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}
