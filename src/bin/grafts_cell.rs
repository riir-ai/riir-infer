//! grafts_cell — the katgpt-rs issue-988 T2b REAL-ACTIVATION cell
//! (riir-infer Issue 1206): baseline measurement, graft fitting, and the
//! four sealed arms over the dense qwen35-hybrid body, ending in a
//! BLAKE3-pinned report JSON.
//!
//! Two-run protocol (the pre-registration law):
//!   run A — `--baseline-only`: measures the candidate tasks × formats,
//!           writes `baselines.json`; the task + format are PINNED from
//!           its table (headroom band) BEFORE any graft is built.
//!   run B — the full cell: `--task <kind> --format <f>` (REQUIRED);
//!           splits pinned by seed, probes fitted on the solved split,
//!           dev rows disclosed, the four arms on the sealed split,
//!           `report.json` + payload BLAKE3.
//!
//! ```bash
//! cargo run --release --features grafts_cell --bin grafts_cell -- \
//!     --baseline-only --out .benchmarks/grafts_cell_t2b
//! cargo run --release --features grafts_cell --bin grafts_cell -- \
//!     --task ThreeOpAdd10 --format Bare --out .benchmarks/grafts_cell_t2b
//! ```
//!
//! Determinism (G1): the report's `payload_blake3` is taken over the
//! canonical payload with the digest field blanked — a re-run over the
//! same checkpoint + seed must reproduce it byte-for-byte.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use riir_infer_core::deltanet::{
    HybridCache, HybridForwardScratch, effective_rotary_dim, forward_qwen_deltanet,
};
use riir_infer_core::gguf_loader::{GgmlType, GgufFile, load_qwen_deltanet_weights_gguf};
use riir_infer_core::grafts_cell::{
    ArmRow, CellReport, FitParts, HiddenCapture, ProbeSample, PromptFormat, RowReader, TaskKind,
    digit_slot_positions, fit_parts, make_split, pearson, probe_layers, split_digest,
};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::tokenizer::BpeTokenizer;

use katgpt_core::graft::{Graft, GraftWrite, LogitWriter, WriteEntry};

// ── Pinned a priori (the issue's pre-registration) ───────────────────────

/// Ridge λ for every closed-form fit (pinned before any sealed touch).
const RIDGE: f32 = 0.1;
/// The write bias delta, PINNED from the run-A logit-gap measurement:
/// the base model's top-1 after "=" is the bare space token at logit
/// ~21.3 vs the answer digit at ~16.3 (gap ≈ 5) — a delta that cannot
/// clear the gap would never flip a base continuation, which is the
/// write seam's whole job. A sweep would tune the answer distribution;
/// the gap-derived pin is the pre-registered posture (disclosed in the
/// bench record).
const BIAS_DELTA: f32 = 8.0;

/// The greedy-chain readout bound: the answer = the first digit token
/// emitted within this many continuation steps (the run-A lesson: the
/// base model continues with prose/punctuation; the digit lands 1-2
/// steps deep).
const GREEDY_CHAIN_STEPS: usize = 3;
/// The switch arming threshold.
const SWITCH_THRESHOLD: f32 = 0.5;
/// The scramble control's seed.
const SCRAMBLE_SEED: u64 = 0x5EED_9880;
/// The split seed (the shuffle + cut).
const SPLIT_SEED: u64 = 0x988_0001;
/// Split fractions over the task space (0.4 / 0.2 / 0.4).
const SOLVED_FRAC: f64 = 0.4;
const DEV_FRAC: f64 = 0.2;

/// The cache capacity bound (longest prompt the cell can ask + headroom;
/// FewShot ThreeOpAdd10 is ~35 tokens — a generous fixed bound keeps one
/// reusable cache honest).
const MAX_POSITIONS: usize = 128;

const DEFAULT_MODEL: &str = "../riir-train/data/Qwen3.5-0.8B-Base-Q8_0.gguf";
const DEFAULT_OUT: &str = ".benchmarks/grafts_cell_t2b";
const ISSUE_TAG: &str = "riir-infer issue 1206 (katgpt-rs issue 988 T2b)";

struct Args {
    model: PathBuf,
    out: PathBuf,
    baseline_only: bool,
    task: Option<TaskKind>,
    format: Option<PromptFormat>,
    /// Ridge λ override (the a-priori 0.1 pin produced an Undecidable
    /// table on the first run-B attempt — the pre-registered λ failed,
    /// disclosed; a re-pin reads the SOLVED+DEV diagnostics only, never
    /// sealed, and the bench record carries both).
    ridge: Option<f32>,
    /// Save the captured fit samples (solved+dev) to this file after the
    /// capture pass, then continue (the cache makes λ re-pins cheap —
    /// the capture pass costs ~23 min, the fit seconds).
    save_samples: Option<PathBuf>,
    /// Load previously-saved samples instead of capturing.
    load_samples: Option<PathBuf>,
    /// Stop after the fit + diagnostics (the decide-check against cached
    /// samples; no dev/sealed passes).
    fit_only: bool,
    /// Off-task capture mode (998 T2): generate the pinned 3-class
    /// corpus, capture at the fit geometry, save the (fit, heldout)
    /// cache to this path, exit. Task + format are PINNED
    /// (ThreeOpAdd10/FewShot — the geometry's owner).
    offtask_capture: Option<PathBuf>,
    /// Posture-2 fresh TEST corpus capture (998 T2 posture-2
    /// pre-registration): `OFFTASK_SEED_TEST`, the pinned 600-row test
    /// plan, saved as a single-segment cache. Same pinned task/format.
    offtask_test_capture: Option<PathBuf>,
    /// Posture-3 FRESH test corpus capture (katgpt-rs issue 998
    /// posture-3 pre-registration, verdict session 40a812a8):
    /// `OFFTASK_SEED_TEST_P3`, same pinned 600-row test plan — the only
    /// posture-3 gate surface; the spent posture-2 corpus feeds the tail
    /// diagnostics only. Must be captured only AFTER the pre-registration
    /// commit (the ruling's contamination ban, same as posture 2).
    offtask_test_capture_p3: Option<PathBuf>,
    /// Sealed capture-only arm check (998 T2 pre-registration, gated leg
    /// 1): load a grafts artifact, capture the 400 sealed prompts at the
    /// fit geometry, compute `switches_armed_at` per prompt — never
    /// decode, never score — and write the counts + bits digest to
    /// `<out>/sealed_arm_check.json`.
    sealed_arm_check: Option<PathBuf>,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        model: PathBuf::from(DEFAULT_MODEL),
        out: PathBuf::from(DEFAULT_OUT),
        baseline_only: false,
        task: None,
        format: None,
        ridge: None,
        save_samples: None,
        load_samples: None,
        fit_only: false,
        offtask_capture: None,
        offtask_test_capture: None,
        offtask_test_capture_p3: None,
        sealed_arm_check: None,
    };
    let mut it = std::env::args().skip(1);
    let mut task_str = String::new();
    let mut fmt_str = String::new();
    while let Some(f) = it.next() {
        match f.as_str() {
            "--model" => a.model = PathBuf::from(it.next().context("--model needs a value")?),
            "--out" => a.out = PathBuf::from(it.next().context("--out needs a value")?),
            "--baseline-only" => a.baseline_only = true,
            "--task" => task_str = it.next().context("--task needs a value")?,
            "--format" => fmt_str = it.next().context("--format needs a value")?,
            "--ridge" => {
                a.ridge = Some(
                    it.next()
                        .context("--ridge needs a value")?
                        .parse()
                        .context("--ridge must parse as f32")?,
                )
            }
            "--save-samples" => {
                a.save_samples = Some(PathBuf::from(
                    it.next().context("--save-samples needs a value")?,
                ))
            }
            "--load-samples" => {
                a.load_samples = Some(PathBuf::from(
                    it.next().context("--load-samples needs a value")?,
                ))
            }
            "--fit-only" => a.fit_only = true,
            "--offtask-capture" => {
                a.offtask_capture = Some(PathBuf::from(
                    it.next().context("--offtask-capture needs a value")?,
                ))
            }
            "--offtask-test-capture" => {
                a.offtask_test_capture = Some(PathBuf::from(
                    it.next().context("--offtask-test-capture needs a value")?,
                ))
            }
            "--offtask-test-capture-p3" => {
                a.offtask_test_capture_p3 = Some(PathBuf::from(
                    it.next().context("--offtask-test-capture-p3 needs a value")?,
                ))
            }
            "--sealed-arm-check" => {
                a.sealed_arm_check = Some(PathBuf::from(
                    it.next().context("--sealed-arm-check needs a value")?,
                ))
            }
            other => bail!("unknown flag {other}"),
        }
    }
    if !task_str.is_empty() {
        a.task = Some(match task_str.as_str() {
            "ThreeOpAdd10" => TaskKind::ThreeOpAdd10,
            "TwoOpMul10" => TaskKind::TwoOpMul10,
            "TwoOpAdd10" => TaskKind::TwoOpAdd10,
            other => bail!("unknown task {other} (ThreeOpAdd10 | TwoOpMul10 | TwoOpAdd10)"),
        });
    }
    if !fmt_str.is_empty() {
        a.format = Some(match fmt_str.as_str() {
            "FewShot" => PromptFormat::FewShot,
            "Bare" => PromptFormat::Bare,
            "QaWrap" => PromptFormat::QaWrap,
            other => bail!("unknown format {other} (FewShot | Bare | QaWrap)"),
        });
    }
    if a.baseline_only {
        if a.task.is_some() || a.format.is_some() {
            bail!("--baseline-only measures every candidate; pass no --task/--format");
        }
    } else if a.task.is_none() || a.format.is_none() {
        if a.offtask_capture.is_none()
            && a.sealed_arm_check.is_none()
            && a.offtask_test_capture.is_none()
            && a.offtask_test_capture_p3.is_none()
        {
            bail!("full-cell runs pin the task + format from run A: --task <k> --format <f>");
        }
        // The off-task + arm-check modes PIN the task + format (the fit
        // geometry's owner); task/format flags beside them are a caller
        // bug — refuse, never silently diverge.
        if a.task.is_some() || a.format.is_some() {
            bail!(
                "--offtask-capture/--offtask-test-capture(-p3)/--sealed-arm-check pin ThreeOpAdd10/FewShot themselves; pass no --task/--format"
            );
        }
    }
    if [
        a.offtask_capture.is_some(),
        a.offtask_test_capture.is_some(),
        a.offtask_test_capture_p3.is_some(),
        a.sealed_arm_check.is_some(),
    ]
    .iter()
    .filter(|&&b| b)
    .count()
        > 1
    {
        bail!(
            "--offtask-capture/--offtask-test-capture(-p3)/--sealed-arm-check are separate modes; one per run"
        );
    }
    Ok(a)
}

// ── The host + the sequence runner ──────────────────────────────────────

struct Host {
    config: riir_infer_core::types::Config,
    weights: riir_infer_core::deltanet::QwenDeltaNetWeights,
    rope_freq: RopeFreqTable,
    tok: BpeTokenizer,
    probe_layers: Vec<usize>,
    vocab_size: usize,
}

/// The per-sequence runner: owns the cache + scratch pair (taken out of
/// the Host once — the cell is single-threaded by design), resets in
/// place per sequence (Issue 838's lane; a fresh pair per sequence
/// spent its life in madvise), and exposes the one-token step with the
/// capture armed at the current position. Lives INSIDE a rayon worker
/// (the whole run is wrapped in one `rayon::scope`: `on_rayon_worker`
/// then runs every forward inline — the per-call scope entry was
/// measured at 2 syscall classes × 84k calls, the run-A sys-time blowup).
struct Runner<'a> {
    host: &'a Host,
    cache: riir_infer_core::deltanet::HybridCache,
    /// The scratch OWNS the capture (the forward hook mounts it by
    /// value); the runner reads rows back through the scratch.
    scratch: riir_infer_core::deltanet::HybridForwardScratch,
    logits: Vec<f32>,
    /// The armed (position, slot) pairs for the CURRENT sequence (the
    /// operand last-digit positions, set per prompt).
    slot_positions: Vec<(usize, usize)>,
}

impl<'a> Runner<'a> {
    fn new(host: &'a Host) -> Result<Self> {
        let cache = HybridCache::with_layer_types_bounded(
            &host.config,
            &host.weights.layer_types,
            MAX_POSITIONS,
        );
        let mut scratch = HybridForwardScratch::new(&host.config);
        scratch.grafts_hidden = Some(HiddenCapture::new(&host.probe_layers, host.config.n_embd));
        Ok(Self {
            host,
            cache,
            scratch,
            logits: vec![0.0; host.vocab_size],
            slot_positions: Vec::new(),
        })
    }

    /// Arm the capture slots for the next prompt (the operand last-digit
    /// positions).
    fn arm_slots(&mut self, slots: &[(usize, usize)]) {
        self.slot_positions = slots.to_vec();
        if let Some(cap) = self.scratch.grafts_hidden.as_mut() {
            cap.arm_slots(slots);
        }
    }

    /// One token step at `pos`; the capture fires when `capture_pos` is
    /// Some(pos). Returns the logits (a copy — the caller biases copies
    /// for the arms) and whether the capture fired.
    fn step(&mut self, token: usize, pos: usize, capture_pos: Option<usize>) -> (Vec<f32>, bool) {
        // The capture fires at ANY armed slot position (the slot map is
        // armed per prompt; capture() resolves pos→slot). `capture_pos`
        // is the legacy single-position arm — kept for the chain's
        // last-prompt-position semantics.
        let _ = capture_pos;
        let armed = self.slot_positions.iter().any(|(p, _)| *p == pos);
        forward_qwen_deltanet(
            &mut self.logits,
            &self.host.weights,
            &mut self.cache,
            token,
            pos,
            &self.host.config,
            &mut self.scratch,
            &self.host.rope_freq,
        );
        let fired = armed;
        (self.logits.clone(), fired)
    }

    /// The captured rows (the scratch's own capture), copied out.
    fn captured_rows(&self) -> Vec<f32> {
        self.scratch
            .grafts_hidden
            .as_ref()
            .map(|c| c.rows().to_vec())
            .unwrap_or_default()
    }

    /// Run a prompt from a fresh cache: feed `tokens` sequentially, the
    /// capture armed at `capture_pos` (the caller passes tokens.len()-1
    /// for the last-prompt-position probe, None for none). Returns the
    /// final-position logits + the captured rows (empty when unarmed).
    fn run_prompt(&mut self, tokens: &[usize], capture_pos: Option<usize>) -> (Vec<f32>, Vec<f32>) {
        self.cache.reset();
        let mut rows = Vec::new();
        for (pos, &t) in tokens.iter().enumerate() {
            let (_logits, fired) = self.step(t, pos, capture_pos);
            if fired {
                rows = self.captured_rows();
            }
        }
        (self.logits.clone(), rows)
    }

    /// Continue the sequence by ONE token (greedy chain): process
    /// `token` at `pos`, capture armed at that pos, return the logits +
    /// rows for the graft's read.
    fn continue_step(&mut self, token: usize, pos: usize) -> (Vec<f32>, Vec<f32>) {
        let (logits, _fired) = self.step(token, pos, Some(pos));
        (logits, self.captured_rows())
    }
}

/// The digit set for one (task, format) — the answer surface.
struct DigitSurface {
    map: [usize; 10],
    set: std::collections::BTreeSet<usize>,
}

impl DigitSurface {
    fn new(tok: &BpeTokenizer, task: TaskKind, format: PromptFormat) -> Result<Self> {
        let map = digit_map(tok, task, format)?;
        let set = map.iter().copied().collect();
        Ok(Self { map, set })
    }

    fn token_of(&self, d: u8) -> usize {
        self.map[usize::from(d)]
    }

    fn is_digit(&self, tok: usize) -> bool {
        self.set.contains(&tok)
    }
}

/// The biased-argmax writer: folds graft biases into a logits copy.
struct BiasWriter<'b> {
    logits: &'b mut [f32],
}

impl LogitWriter for BiasWriter<'_> {
    fn add_bias(&mut self, token: u32, bias: f32) {
        self.logits[token as usize] += bias;
    }
}

fn argmax(xs: &[f32]) -> usize {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best
}

/// Load the host (dense-hybrid only; a ternary file refuses loud).
fn load_host(model: &Path) -> Result<Host> {
    let probe = GgufFile::open(model).with_context(|| format!("open {}", model.display()))?;
    let is_ternary = probe
        .tensor_info("token_embd.weight")
        .map(|info| matches!(info.ggml_type, GgmlType::Q2_0 | GgmlType::PTQ1_0))
        .unwrap_or(false);
    if is_ternary {
        bail!("grafts_cell: ternary checkpoint refused (the cell is the dense hybrid body)");
    }
    let tok = BpeTokenizer::from_gguf(&probe).context("BPE tokenizer from gguf")?;
    drop(probe);
    let (config, weights) = load_qwen_deltanet_weights_gguf(model)
        .with_context(|| format!("load {}", model.display()))?;
    let rope_freq = RopeFreqTable::new(config.rope_theta, effective_rotary_dim(&config));
    let layers = probe_layers(config.n_layer);
    let (n_embd, vocab_size, n_layer) = (config.n_embd, config.vocab_size, config.n_layer);
    println!(
        "# model: qwen35 hybrid dense | layers={n_layer} n_embd={n_embd} vocab={vocab_size} | probe layers {layers:?}"
    );
    Ok(Host {
        config,
        weights,
        rope_freq,
        tok,
        probe_layers: layers,
        vocab_size,
    })
}

/// The per-(task, format) digit-token map, derived IN CONTEXT from a
/// reference prompt of the actual pinned task (the surface is exactly
/// what a greedy decode emits at the answer position — the run-A
/// lesson's failure class killed at the root).
fn digit_map(tok: &BpeTokenizer, task: TaskKind, format: PromptFormat) -> Result<[usize; 10]> {
    let inst: &[u8] = match task {
        TaskKind::ThreeOpAdd10 => &[73, 34, 51],
        _ => &[7, 3],
    };
    let reference = task.prompt(inst, format);
    riir_infer_core::grafts_cell::digit_tokens(tok, &reference)
}

/// Per-task-format baseline over the full instance space: the GREEDY
/// CHAIN readout — step the decode up to GREEDY_CHAIN_STEPS positions;
/// the first digit token emitted is the answer; no digit within the
/// bound = wrong (the run-A lesson: this base model continues with
/// prose/punctuation, the digit lands 1-2 steps deep, never at step 0).
fn measure_baseline(
    runner: &mut Runner,
    surface: &DigitSurface,
    task: TaskKind,
    format: PromptFormat,
) -> Result<(f64, Vec<bool>)> {
    let instances = task.instances();
    let mut correct = 0usize;
    let mut digit_am = 0usize;
    let mut wrong_flags = Vec::with_capacity(instances.len());
    for inst in &instances {
        let tokens = runner.host.tok.encode(&task.prompt(inst, format));
        if tokens.is_empty() || tokens.len() + GREEDY_CHAIN_STEPS + 1 > MAX_POSITIONS {
            bail!("prompt chain exceeds the cache bound for '{:?}'", task);
        }
        let slots = digit_slot_positions(&runner.host.tok, task, format, inst)?;
        runner.arm_slots(
            &slots
                .iter()
                .enumerate()
                .map(|(s, &p)| (p, s))
                .collect::<Vec<_>>(),
        );
        let (mut logits, _) = runner.run_prompt(&tokens, None);
        let answer = surface.token_of(task.answer(inst));
        let mut ok = false;
        let mut digit_seen = false;
        for pos in (tokens.len()..).take(GREEDY_CHAIN_STEPS + 1) {
            let am = argmax(&logits);
            if surface.is_digit(am) {
                digit_seen = true;
                ok = am == answer;
                break;
            }
            let (next, _) = runner.continue_step(am, pos);
            logits = next;
        }
        if digit_seen {
            digit_am += 1;
        }
        wrong_flags.push(!ok);
        if ok {
            correct += 1;
        }
    }
    let n = instances.len();
    println!("#   chain readout: digit within {GREEDY_CHAIN_STEPS}+1 steps {digit_am}/{n}");
    Ok((correct as f64 / n as f64, wrong_flags))
}

/// Split fractions per task space size.
fn split_counts(n: usize) -> (usize, usize) {
    let solved = ((n as f64 * SOLVED_FRAC) as usize).max(1);
    let dev = ((n as f64 * DEV_FRAC) as usize).max(1);
    (solved, dev)
}

/// The capture pass over `indices`: prompts run with the capture armed
/// at the LAST prompt position; rows + answers + base-wrong labels
/// (the baseline's OWN greedy-chain verdict at that instance) collected.
fn capture_samples(
    runner: &mut Runner,
    surface: &DigitSurface,
    task: TaskKind,
    format: PromptFormat,
    instances: &[Vec<u8>],
    indices: &[usize],
) -> Result<Vec<ProbeSample>> {
    let mut out = Vec::with_capacity(indices.len());
    for &idx in indices {
        let inst = &instances[idx];
        let tokens = runner.host.tok.encode(&task.prompt(inst, format));
        let slots = digit_slot_positions(&runner.host.tok, task, format, inst)?;
        runner.arm_slots(
            &slots
                .iter()
                .enumerate()
                .map(|(s, &p)| (p, s))
                .collect::<Vec<_>>(),
        );
        let (mut logits, rows) = runner.run_prompt(&tokens, None);
        // The base verdict follows the SAME greedy-chain readout as the
        // baseline (the switch label = the base got it wrong through the
        // chain, not at one position).
        let answer = surface.token_of(task.answer(inst));
        let mut base_ok = false;
        let mut first_rows = rows;
        for pos in (tokens.len()..).take(GREEDY_CHAIN_STEPS + 1) {
            let am = argmax(&logits);
            if surface.is_digit(am) {
                base_ok = am == answer;
                break;
            }
            let (next, _) = runner.continue_step(am, pos);
            logits = next;
        }
        out.push(ProbeSample {
            instance: idx,
            base_wrong: !base_ok,
            answer: task.answer(inst),
            operands: inst.clone(),
            rows: std::mem::take(&mut first_rows),
        });
    }
    Ok(out)
}

/// Assemble a graft posture from the fitted parts (switches = the
/// conditional gate, or empty for unconditional).
fn build_graft(parts: &FitParts, writes: &GraftWrite, conditional: bool) -> Result<Graft> {
    let switches = if conditional {
        vec![parts.switch.clone()]
    } else {
        Vec::new()
    };
    Graft::new(
        parts.reads.clone(),
        parts.table.clone(),
        vec![writes.clone()],
        switches,
    )
    .map_err(anyhow::Error::msg)
}

/// Eval one arm over a split: the greedy chain with the graft applied
/// at EVERY step (the graft reads the CURRENT activations, decides, and
/// biases the digit tokens; the baseline arm skips the apply). The
/// first digit token emitted within the bound is the answer.
#[allow(clippy::too_many_arguments)]
fn eval_arm(
    runner: &mut Runner,
    surface: &DigitSurface,
    task: TaskKind,
    format: PromptFormat,
    instances: &[Vec<u8>],
    indices: &[usize],
    graft: Option<&Graft>,
    arm_name: &str,
) -> Result<ArmRow> {
    let n_embd = runner.host.config.n_embd;
    let mut correct = 0usize;
    let mut digit_seen_n = 0usize;
    let mut decided_sum = 0usize;
    let mut fired = 0usize;
    let mut applies = 0usize;
    for &idx in indices {
        let inst = &instances[idx];
        let tokens = runner.host.tok.encode(&task.prompt(inst, format));
        let answer = surface.token_of(task.answer(inst));
        let slots = digit_slot_positions(&runner.host.tok, task, format, inst)?;
        runner.arm_slots(
            &slots
                .iter()
                .enumerate()
                .map(|(s, &p)| (p, s))
                .collect::<Vec<_>>(),
        );
        let (mut logits, rows) = runner.run_prompt(&tokens, None);
        // The held-arm law (998 T2 pre-registration): the switch is
        // evaluated ONCE per prompt, at the fit-geometry prefill
        // captures, and the decision holds for the whole greedy chain —
        // task identity belongs to the prompt, never to a decode step.
        // For every pre-998 posture this is byte-identical to per-call
        // `apply` (the old switch armed every fit-surface row → held
        // true → `apply_with_arm(true, …) ≡ apply(…)`; switchless grafts
        // hold true trivially). The step-0 equivalence is fixture-pinned
        // in katgpt-core.
        let held_arm = graft.map(|g| {
            let reader = RowReader::new(&runner.host.probe_layers, n_embd, &rows);
            g.switches_armed_at(&reader)
        });
        let mut ok = false;
        let mut digit_seen = false;
        let mut armed_rows: Option<Vec<f32>> = Some(rows);
        for pos in (tokens.len()..).take(GREEDY_CHAIN_STEPS + 1) {
            let mut step_logits = logits.clone();
            let mut applied_decided = 0usize;
            if let Some(g) = graft
                && let Some(r) = &armed_rows
            {
                let reader = RowReader::new(&runner.host.probe_layers, n_embd, r);
                let mut input_vals = vec![0u16; g.n_inputs()];
                let mut out_vals = vec![0u16; g.n_outputs()];
                let applied = {
                    let mut w = BiasWriter {
                        logits: &mut step_logits,
                    };
                    let bit = held_arm.unwrap_or(false);
                    g.apply_with_arm(bit, &reader, &mut w, &mut input_vals, &mut out_vals)
                };
                applied_decided = applied.decided;
                applies += 1;
                if applied.fired {
                    fired += 1;
                }
            }
            decided_sum += applied_decided;
            let am = argmax(&step_logits);
            if surface.is_digit(am) {
                digit_seen = true;
                ok = am == answer;
                break;
            }
            let (next, _) = runner.continue_step(am, pos);
            logits = next;
            // Deeper steps captured inside continue_step (armed at `pos`);
            // a chained graft re-arms from the fresh rows.
            armed_rows = Some(runner.captured_rows());
        }
        if digit_seen {
            digit_seen_n += 1;
        }
        if ok {
            correct += 1;
        }
    }
    let n = indices.len();
    Ok(ArmRow {
        arm: arm_name.to_string(),
        accuracy: correct as f64 / n as f64,
        digit_argmax_rate: digit_seen_n as f64 / n as f64,
        n,
        decided: graft.map(|_| decided_sum as f64 / n.max(1) as f64),
        armed_rate: graft.map(|_| fired as f64 / applies.max(1) as f64),
    })
}

fn main() -> Result<()> {
    let args = parse_args()?;
    std::fs::create_dir_all(&args.out).with_context(|| format!("mkdir {}", args.out.display()))?;

    // Checkpoint BLAKE3 (streamed; pinned in every artifact).
    let t0 = std::time::Instant::now();
    let mut h = blake3::Hasher::new();
    let mut f = std::fs::File::open(&args.model)
        .with_context(|| format!("open {}", args.model.display()))?;
    std::io::copy(&mut f, &mut h).context("hash checkpoint")?;
    drop(f);
    let checkpoint_blake3 = h.finalize().to_hex().to_string();
    println!(
        "# checkpoint blake3 {} ({:.1}s)",
        &checkpoint_blake3[..16],
        t0.elapsed().as_secs_f32()
    );

    let host = load_host(&args.model)?;

    // The WHOLE run lives inside one rayon scope: `on_rayon_worker` then
    // executes every forward INLINE on this worker (the per-call scope
    // entry cost the run-A sys-time blowup — 2 latch syscall classes per
    // token, ~84k calls per pass).
    let mut result: Result<()> = Ok(());
    rayon::scope(|_s| {
        result = if args.offtask_capture.is_some() {
            run_offtask_capture(
                &args,
                &host,
                riir_infer_core::grafts_cell::OFFTASK_SEED,
                riir_infer_core::grafts_cell::OFFTASK_FIT_N
                    + riir_infer_core::grafts_cell::OFFTASK_HELDOUT_N,
                "offtask-capture",
            )
        } else if args.offtask_test_capture.is_some() {
            run_offtask_capture(
                &args,
                &host,
                riir_infer_core::grafts_cell::OFFTASK_SEED_TEST,
                riir_infer_core::grafts_cell::OFFTASK_TEST_N,
                "offtask-test-capture",
            )
        } else if args.offtask_test_capture_p3.is_some() {
            run_offtask_capture(
                &args,
                &host,
                riir_infer_core::grafts_cell::OFFTASK_SEED_TEST_P3,
                riir_infer_core::grafts_cell::OFFTASK_TEST_N,
                "offtask-test-capture-p3",
            )
        } else if args.sealed_arm_check.is_some() {
            run_sealed_arm_check(&args, &host, checkpoint_blake3)
        } else {
            run_cell(&args, &host, checkpoint_blake3)
        };
    });
    result
}

/// The task + format the off-task and arm-check modes PIN (the fit
/// geometry's owner — 998 T2 pre-registration).
const PINNED_TASK: TaskKind = TaskKind::ThreeOpAdd10;
const PINNED_FORMAT: PromptFormat = PromptFormat::FewShot;

/// Box-state line for the run log (998 T2 pre-registration: the amo
/// capture runs beside the confirmatory determinism run — wall-clock
/// fields must never be misread as a quiet-box figure later).
fn log_box_state(tag: &str) {
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);
    // macOS: sysctl vm.loadavg; Linux: /proc/loadavg. A read failure is
    // disclosed as n/a, never fabricated.
    let load = std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .or_else(|| {
            std::fs::read_to_string("/proc/loadavg")
                .ok()
                .map(|s| s.split_whitespace().take(3).collect::<Vec<_>>().join(" "))
        })
        .unwrap_or_else(|| "n/a".into());
    println!(
        "# box[{tag}]: cpus={cpus} loadavg={load} (share-box — wall-clock fields carry this line)"
    );
}

/// The off-task capture mode: generate the pinned 3-class corpus,
/// capture at the fit geometry, save the cache. `seed` + `total`
/// select the corpus (original: seed `OFFTASK_SEED`, 1000 rows,
/// (fit, heldout) pair; posture-2 fresh test: `OFFTASK_SEED_TEST`,
/// 600 rows, single-segment cache).
fn run_offtask_capture(args: &Args, host: &Host, seed: u64, total: usize, tag: &str) -> Result<()> {
    let out_path = args
        .offtask_capture
        .clone()
        .or_else(|| args.offtask_test_capture.clone())
        .or_else(|| args.offtask_test_capture_p3.clone())
        .unwrap();
    let test_mode = args.offtask_test_capture.is_some() || args.offtask_test_capture_p3.is_some();
    println!(
        "# off-task capture [{tag}]: seed {seed:#x}, {total} rows, task {PINNED_TASK:?}/{PINNED_FORMAT:?} — pinned by 998 T2{}",
        if test_mode {
            " (POSTURE FRESH TEST corpus — captured after the placement-rule commit, the ruling's contamination ban)"
        } else {
            ""
        }
    );
    log_box_state(tag);
    let mut runner = Runner::new(host)?;
    let task = PINNED_TASK;
    let format = PINNED_FORMAT;
    let tok = &host.tok;
    let n_embd = host.config.n_embd;

    // The three ABSOLUTE capture indices (the fit geometry) + the
    // prompt-length targets derived from them.
    let slots = riir_infer_core::grafts_cell::fit_geometry_slots(tok, task, format)?;
    assert_eq!(slots.len(), 3, "fit geometry: three slots");
    let primes_len = slots[0] - 1;
    let target_min = primes_len + riir_infer_core::grafts_cell::OFFTASK_LEN_MARGIN;
    let target_cap = primes_len + riir_infer_core::grafts_cell::OFFTASK_LEN_CAP;
    println!(
        "# fit geometry: capture indices {slots:?} (primes_len {primes_len}) — prompt token target [{target_min}, {target_cap}]"
    );
    let slot_arms: Vec<(usize, usize)> = slots.iter().enumerate().map(|(s, &p)| (p, s)).collect();

    let mut all_rows = Vec::with_capacity(total);
    let t0 = std::time::Instant::now();
    for i in 0..total {
        let class = if test_mode {
            riir_infer_core::grafts_cell::offtask_test_class(i)
        } else {
            riir_infer_core::grafts_cell::offtask_class(i)
        };
        // Grow the prompt sentence-by-sentence (the deterministic stream)
        // until its token length clears the geometry depth + margin.
        let mut text = String::new();
        let mut j = 0usize;
        let len = loop {
            text.push_str(&riir_infer_core::grafts_cell::offtask_sentence_seed(
                seed, i, j,
            ));
            text.push(' ');
            let cur = tok.encode(text.trim_end()).len();
            if cur >= target_min {
                break cur;
            }
            j += 1;
            if j > 64 {
                bail!(
                    "offtask {i}: sentence stream exceeded 64 sentences (generator pathological)"
                );
            }
        };
        if len > target_cap {
            bail!("offtask {i}: prompt {len} tokens exceeds the cap {target_cap} — geometry drift");
        }
        let tokens = tok.encode(text.trim_end());
        if slots.iter().any(|&p| p >= tokens.len()) {
            bail!(
                "offtask {i}: capture index beyond the prompt ({} tokens)",
                tokens.len()
            );
        }
        if tokens.len() + 2 > MAX_POSITIONS {
            bail!("offtask {i}: prompt exceeds the cache bound");
        }
        runner.arm_slots(&slot_arms);
        let (_logits, rows) = runner.run_prompt(&tokens, None);
        if rows.len()
            != riir_infer_core::grafts_cell::CAPTURE_SLOTS * host.probe_layers.len() * n_embd
        {
            bail!("offtask {i}: capture did not fire ({} rows)", rows.len());
        }
        all_rows.push(riir_infer_core::grafts_cell::OfftaskRow {
            index: i,
            class,
            rows,
        });
        if (i + 1) % 100 == 0 {
            println!(
                "# offtask {}/{} ({:.1}s) — last len {len} tokens, class {class}",
                i + 1,
                total,
                t0.elapsed().as_secs_f32()
            );
        }
    }
    if test_mode {
        let json = serde_json::to_vec(&all_rows).context("serialize offtask test")?;
        std::fs::write(&out_path, &json)
            .with_context(|| format!("write {}", out_path.display()))?;
        println!(
            "# off-task TEST cache {} rows → {} ({:.1}s) blake3 {}",
            all_rows.len(),
            out_path.display(),
            t0.elapsed().as_secs_f32(),
            &blake3::hash(&json).to_hex()[..16]
        );
    } else {
        let fit_n = riir_infer_core::grafts_cell::OFFTASK_FIT_N;
        riir_infer_core::grafts_cell::save_offtask(
            &out_path,
            &all_rows[..fit_n],
            &all_rows[fit_n..],
        )?;
        println!(
            "# off-task cache {} fit + {} heldout → {} ({:.1}s)",
            fit_n,
            all_rows.len() - fit_n,
            out_path.display(),
            t0.elapsed().as_secs_f32()
        );
    }
    Ok(())
}

/// The sealed capture-only arm check (998 T2 gated leg 1's sealed half):
/// load a grafts artifact, capture the 400 sealed prompts at the fit
/// geometry, count `switches_armed_at` — never decode, never score,
/// never read an outcome (inside the one-sealed-read-per-posture law).
fn run_sealed_arm_check(args: &Args, host: &Host, checkpoint_blake3: String) -> Result<()> {
    let artifact_path = args.sealed_arm_check.as_ref().unwrap();
    log_box_state("sealed-arm-check");
    let task = PINNED_TASK;
    let format = PINNED_FORMAT;
    let mut runner = Runner::new(host)?;
    let n_embd = host.config.n_embd;

    // The artifact: strict GRFZ parse (the export's envelope), decode,
    // verify the digest against the canonical bytes — a corrupted
    // sidecar refuses loud here, same as at mount.
    let bytes = std::fs::read(artifact_path)
        .with_context(|| format!("read {}", artifact_path.display()))?;
    if bytes.len() < 52 || &bytes[0..4] != b"GRFZ" {
        bail!(
            "sealed-arm-check: not a GRFZ artifact ({})",
            artifact_path.display()
        );
    }
    // The GRFZ envelope framing is the exporter's own (magic 4 + version
    // 4 + digest 32 + epoch 8 + canonical_len 4 + canonical); rebuild the
    // katgpt-core artifact from its fields, then decode through the
    // validating path (format version + digest + canonicality).
    let fmt_version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    let mut digest32 = [0u8; 32];
    digest32.copy_from_slice(&bytes[8..40]);
    let epoch = u64::from_le_bytes([
        bytes[40], bytes[41], bytes[42], bytes[43], bytes[44], bytes[45], bytes[46], bytes[47],
    ]);
    let canonical_len = u32::from_le_bytes([bytes[48], bytes[49], bytes[50], bytes[51]]) as usize;
    if bytes.len() != 52 + canonical_len {
        bail!("sealed-arm-check: GRFZ envelope length mismatch");
    }
    let artifact = katgpt_core::GraftArtifact {
        format_version: fmt_version,
        epoch,
        graft_digest: digest32,
        canonical: bytes[52..].to_vec(),
    };
    let graft = artifact
        .decode()
        .map_err(anyhow::Error::msg)
        .with_context(|| "decode graft artifact")?;
    println!(
        "# sealed-arm-check: artifact {} graft digest {} (switches {}) — capture-only, no decode",
        artifact_path.display(),
        artifact.graft_digest_hex(),
        graft.switches.len(),
    );

    // The sealed split (the same pinned seed — the 400 prompts the
    // Bench 971 accuracy pins describe).
    let instances = task.instances();
    let (solved_n, dev_n) = split_counts(instances.len());
    let split = make_split(instances.len(), solved_n, dev_n, SPLIT_SEED)?;
    let slots = riir_infer_core::grafts_cell::fit_geometry_slots(&host.tok, task, format)?;
    let slot_arms: Vec<(usize, usize)> = slots.iter().enumerate().map(|(s, &p)| (p, s)).collect();

    let mut armed = 0usize;
    let mut bits = Vec::with_capacity(split.sealed.len());
    let t0 = std::time::Instant::now();
    for (k, &idx) in split.sealed.iter().enumerate() {
        let inst = &instances[idx];
        let tokens = runner.host.tok.encode(&task.prompt(inst, format));
        runner.arm_slots(&slot_arms);
        let (_logits, rows) = runner.run_prompt(&tokens, None);
        if rows.is_empty() {
            bail!("sealed-arm-check: capture did not fire at sealed row {k}");
        }
        let reader = RowReader::new(&host.probe_layers, n_embd, &rows);
        let bit = graft.switches_armed_at(&reader);
        bits.push(u8::from(bit));
        if bit {
            armed += 1;
        }
        if (k + 1) % 100 == 0 {
            println!(
                "# sealed-arm-check {}/{} ({:.1}s)",
                k + 1,
                split.sealed.len(),
                t0.elapsed().as_secs_f32()
            );
        }
    }
    let bits_digest = blake3::hash(&bits).to_hex().to_string();
    println!(
        "# sealed-arm-check: {armed}/{} armed — bits blake3 {} ({:.1}s); ANY disarm = lane-A GOAT FAIL (998 T2)",
        bits.len(),
        &bits_digest[..16],
        t0.elapsed().as_secs_f32()
    );
    let payload = serde_json::json!({
        "issue": "katgpt-rs issue 998 T2 gated leg 1 (sealed capture-only arm check)",
        "checkpoint_blake3": checkpoint_blake3,
        "artifact": artifact_path.to_string_lossy(),
        "artifact_graft_digest": artifact.graft_digest_hex(),
        "task": task,
        "format": format,
        "n": bits.len(),
        "armed": armed,
        "disarmed": bits.len() - armed,
        "bits_blake3": bits_digest,
        "law": "any disarm = lane-A GOAT FAIL; threshold never moves on held-out evidence",
    });
    let path = args.out.join("sealed_arm_check.json");
    std::fs::write(&path, serde_json::to_vec_pretty(&payload)?)
        .with_context(|| format!("write {}", path.display()))?;
    println!("# report → {}", path.display());
    Ok(())
}

/// The cell body (run A or run B), executed on a rayon worker.
fn run_cell(args: &Args, host: &Host, checkpoint_blake3: String) -> Result<()> {
    let mut runner = Runner::new(host)?;

    // ── Run A: the baseline table over every candidate ──
    if args.baseline_only {
        let cap_iter = [
            TaskKind::ThreeOpAdd10,
            TaskKind::TwoOpMul10,
            TaskKind::TwoOpAdd10,
        ];
        let mut rows = Vec::with_capacity(cap_iter.len());
        for task in cap_iter {
            for format in [
                PromptFormat::FewShot,
                PromptFormat::Bare,
                PromptFormat::QaWrap,
            ] {
                let t = std::time::Instant::now();
                let surface = DigitSurface::new(&host.tok, task, format)?;
                let (acc, wrong) = measure_baseline(&mut runner, &surface, task, format)?;
                rows.push(serde_json::json!({
                    "task": task,
                    "format": format,
                    "accuracy": acc,
                    "wrong_n": wrong.iter().filter(|&&w| w).count(),
                    "n": task.space_len(),
                    "secs": t.elapsed().as_secs_f64(),
                }));
                println!(
                    "# baseline task={task:?} format={format:?}: acc {:.4} ({} wrong / {})",
                    acc,
                    wrong.iter().filter(|&&w| w).count(),
                    task.space_len()
                );
            }
        }
        let payload = serde_json::json!({
            "issue": ISSUE_TAG,
            "checkpoint_blake3": checkpoint_blake3,
            "baselines": rows,
        });
        let path = args.out.join("baselines.json");
        std::fs::write(&path, serde_json::to_vec_pretty(&payload)?)
            .with_context(|| format!("write {}", path.display()))?;
        println!("# run A complete → {}", path.display());
        return Ok(());
    }

    // ── Run B: the full cell (task + format pinned from run A) ──
    let task = args.task.unwrap();
    let format = args.format.unwrap();
    let instances = task.instances();
    let (solved_n, dev_n) = split_counts(instances.len());
    let split = make_split(instances.len(), solved_n, dev_n, SPLIT_SEED)?;
    let sdig = split_digest(task, format, &split, SPLIT_SEED);
    println!(
        "# split: solved={} dev={} sealed={} seed={SPLIT_SEED:#x} digest={}",
        split.solved.len(),
        split.dev.len(),
        split.sealed.len(),
        &sdig[..16]
    );

    // The fit surface (solved + dev captured in ONE deterministic pass
    // order — ascending indices per split as pinned).
    let t_fit = std::time::Instant::now();
    let surface = DigitSurface::new(&host.tok, task, format)?;
    let (solved_samples, dev_samples) = if let Some(path) = &args.load_samples {
        let (s, d) = riir_infer_core::grafts_cell::load_samples(path)?;
        println!(
            "# samples loaded: {} solved + {} dev from {}",
            s.len(),
            d.len(),
            path.display()
        );
        // The answer field in a cache predates the u8-overflow fix —
        // RE-DERIVE from the operands (the answer is a pure function of
        // them); a stale stored value carries no load.
        let mut s = s;
        let mut d = d;
        for sample in s.iter_mut().chain(d.iter_mut()) {
            sample.answer = task.answer(&sample.operands);
        }
        (s, d)
    } else {
        let solved = capture_samples(
            &mut runner,
            &surface,
            task,
            format,
            &instances,
            &split.solved,
        )?;
        let dev = capture_samples(&mut runner, &surface, task, format, &instances, &split.dev)?;
        println!(
            "# capture: {} solved + {} dev rows in {:.1}s",
            solved.len(),
            dev.len(),
            t_fit.elapsed().as_secs_f32()
        );
        if let Some(path) = &args.save_samples {
            riir_infer_core::grafts_cell::save_samples(path, &solved, &dev)?;
            println!("# samples saved → {}", path.display());
        }
        (solved, dev)
    };

    let ridge = args.ridge.unwrap_or(RIDGE);
    if ridge != RIDGE {
        println!(
            "# λ RE-PIN {ridge} (the a-priori {RIDGE} pin failed: the first run-B attempt mined an Undecidable table — the pre-registered λ is disclosed as failed; the re-pin reads the fit diagnostics below, solved+dev only, never sealed)"
        );
    }

    // Fit diagnostics BEFORE the mine: per-read score spread on the
    // read's OWN slot rows (the slot-era layout; the pre-slot block
    // correlated wrong slices and produced meaningless probe_corr — the
    // 0.03-row in the first report is that bug's output, disclosed).
    {
        let n_embd = host.config.n_embd;
        let k = host.probe_layers.len();
        let d = k * n_embd;
        let n_reads = solved_samples[0].operands.len();
        for j in 0..n_reads {
            let mut xs = Vec::with_capacity(solved_samples.len() * d);
            for s in &solved_samples {
                xs.extend_from_slice(&s.rows[j * d..(j + 1) * d]);
            }
            let targets: Vec<f32> = solved_samples
                .iter()
                .map(|s| f32::from(s.operands[j] % 10))
                .collect();
            let layers_u32: Vec<u32> = host
                .probe_layers
                .iter()
                .map(|&l| riir_infer_core::grafts_cell::read_id(j, l))
                .collect();
            if let Ok(read) = katgpt_core::graft::fit_read(
                &xs,
                solved_samples.len(),
                d,
                &layers_u32,
                &vec![n_embd; k],
                &targets,
                ridge,
                10,
                true,
            ) {
                let mut lo = f32::INFINITY;
                let mut hi = f32::NEG_INFINITY;
                let mut scores = Vec::with_capacity(solved_samples.len());
                for s in &solved_samples {
                    let base = j * d;
                    let mut sc = read.bias;
                    let mut off = 0usize;
                    for seg in &read.segments {
                        sc += seg.score(&s.rows[base + off..base + off + n_embd]);
                        off += n_embd;
                    }
                    scores.push(sc);
                    lo = lo.min(sc);
                    hi = hi.max(sc);
                }
                let answers: Vec<f32> = solved_samples
                    .iter()
                    .map(|s| f32::from(s.operands[j] % 10))
                    .collect();
                let corr = pearson(&scores, &answers);
                println!(
                    "# fit-diag read {j} (slot {j}): score spread [{lo:.3}, {hi:.3}] corr {corr:.4}"
                );
            }
        }
    }

    // The direct read check: fit read 0 on slot-0 rows EXACTLY as
    // fit_parts does, then round-accuracy over the solved split — the
    // f32 solver's verdict, isolated from the rest of the pipeline.
    {
        let n_embd = host.config.n_embd;
        let k = host.probe_layers.len();
        let d = k * n_embd;
        let mut xs = Vec::with_capacity(solved_samples.len() * d);
        for s in &solved_samples {
            xs.extend_from_slice(&s.rows[0..d]);
        }
        let targets: Vec<f32> = solved_samples
            .iter()
            .map(|s| f32::from(s.operands[0] % 10))
            .collect();
        let layers_u32: Vec<u32> = host
            .probe_layers
            .iter()
            .map(|&l| riir_infer_core::grafts_cell::read_id(0, l))
            .collect();
        match katgpt_core::graft::fit_read(
            &xs,
            solved_samples.len(),
            d,
            &layers_u32,
            &vec![n_embd; k],
            &targets,
            ridge,
            10,
            true,
        ) {
            Ok(read) => {
                let mut ok = 0usize;
                let mut lo = f32::INFINITY;
                let mut hi = f32::NEG_INFINITY;
                for s in &solved_samples {
                    // The REAL read path: value_at over the slot-aware
                    // reader (id resolution + quantize — no parallel
                    // implementation).
                    let reader = RowReader::new(&host.probe_layers, n_embd, &s.rows);
                    let v = read.value_at(&reader);
                    let sc = {
                        // the pre-quantize score, for the spread row
                        let mut sc = read.bias;
                        let mut off = 0usize;
                        for seg in &read.segments {
                            sc += seg.score(&s.rows[off..off + n_embd]);
                            off += n_embd;
                        }
                        sc
                    };
                    lo = lo.min(sc);
                    hi = hi.max(sc);
                    if usize::from(v) == usize::from(s.operands[0] % 10) {
                        ok += 1;
                    }
                }
                println!(
                    "# read-check slot0: value_at round-acc {ok}/{} | score spread [{lo:.3}, {hi:.3}]",
                    solved_samples.len()
                );
            }
            Err(e) => println!("# read-check slot0: fit error {e:?}"),
        }
    }

    // Fit (the primitive's closed-form lanes; Undecidable passes through
    // loud and the cell discloses it — the module's conflict census
    // prints the exact driver before the error).
    let parts = fit_parts(
        &solved_samples,
        &host.probe_layers,
        host.config.n_embd,
        ridge,
    )
    .map_err(anyhow::Error::msg)
    .context("graft fit (solved split)")?;
    let cell0 = parts.table.rule_for_cell(0).is_some();
    if !cell0 {
        bail!("mined table has no rule for output cell 0 (the single answer column)");
    }

    // Writes: every decided value biases its digit token (+BIAS_DELTA);
    // the tokens are the PINNED FORMAT's map (the form the model emits).
    let digit_tok = surface.map;
    let writes = GraftWrite::new(
        0,
        (0..10u16)
            .map(|v| WriteEntry {
                value: v,
                biases: vec![(digit_tok[v as usize] as u32, BIAS_DELTA)],
            })
            .collect::<Vec<_>>(),
    )
    .map_err(anyhow::Error::msg)?;

    let graft_uncond = build_graft(&parts, &writes, false)?;
    let graft_cond = build_graft(&parts, &writes, true)?;
    let graft_scramble = graft_uncond.scrambled(SCRAMBLE_SEED);

    // Fit diagnostics (solved split only).
    let distinct_triples = parts
        .triples
        .iter()
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    let triple_coverage = distinct_triples as f64 / parts.triples.len() as f64;
    let mut probe_corr = Vec::with_capacity(host.probe_layers.len());
    for (j, read) in parts.reads.iter().enumerate() {
        // Read j scores slot j's FULL block (k layers × n_embd), each
        // segment against its own layer's slice — the pre-slot form
        // sliced j*n_embd and fed every segment the same wrong slice
        // (the stale-broken 0.03 row the pinned report discloses; the
        // healed fit-diag block above shows the correct shape).
        let k_seg = read.segments.len();
        let off = j * k_seg * host.config.n_embd;
        let scores: Vec<f32> = solved_samples
            .iter()
            .map(|s| {
                let mut sc = read.bias;
                for (i, seg) in read.segments.iter().enumerate() {
                    let lo = off + i * host.config.n_embd;
                    sc += seg.score(&s.rows[lo..lo + host.config.n_embd]);
                }
                sc
            })
            .collect();
        let answers: Vec<f32> = solved_samples.iter().map(|s| f32::from(s.answer)).collect();
        probe_corr.push(pearson(&scores, &answers));
    }
    println!(
        "# fit: triple coverage {:.3} | probe corr {:?}",
        triple_coverage,
        probe_corr
            .iter()
            .map(|c| format!("{c:.4}"))
            .collect::<Vec<_>>()
            .join(" ")
    );

    if args.fit_only {
        println!("# --fit-only: stopping before the dev/sealed passes");
        return Ok(());
    }

    // Dev rows (the disclosed tuning surface).
    let dev_rows = vec![
        eval_arm(
            &mut runner,
            &surface,
            task,
            format,
            &instances,
            &split.dev,
            None,
            "baseline",
        )?,
        eval_arm(
            &mut runner,
            &surface,
            task,
            format,
            &instances,
            &split.dev,
            Some(&graft_uncond),
            "graft_unconditional",
        )?,
        eval_arm(
            &mut runner,
            &surface,
            task,
            format,
            &instances,
            &split.dev,
            Some(&graft_cond),
            "graft_conditional",
        )?,
        eval_arm(
            &mut runner,
            &surface,
            task,
            format,
            &instances,
            &split.dev,
            Some(&graft_scramble),
            "scramble",
        )?,
    ];
    for r in &dev_rows {
        println!(
            "# dev   {:<19} acc {:.4} digit-argmax {:.4} armed {:?} decided {:?}",
            r.arm,
            r.accuracy,
            r.digit_argmax_rate,
            r.armed_rate.map(|v| format!("{v:.3}")),
            r.decided.map(|v| format!("{v:.3}"))
        );
    }

    // Sealed split — the four arms, touched ONCE.
    let sealed_rows = vec![
        eval_arm(
            &mut runner,
            &surface,
            task,
            format,
            &instances,
            &split.sealed,
            None,
            "baseline",
        )?,
        eval_arm(
            &mut runner,
            &surface,
            task,
            format,
            &instances,
            &split.sealed,
            Some(&graft_uncond),
            "graft_unconditional",
        )?,
        eval_arm(
            &mut runner,
            &surface,
            task,
            format,
            &instances,
            &split.sealed,
            Some(&graft_cond),
            "graft_conditional",
        )?,
        eval_arm(
            &mut runner,
            &surface,
            task,
            format,
            &instances,
            &split.sealed,
            Some(&graft_scramble),
            "scramble",
        )?,
    ];
    println!("# ── sealed ──");
    for r in &sealed_rows {
        println!(
            "# sealed {:<19} acc {:.4} digit-argmax {:.4} armed {:?} decided {:?}",
            r.arm,
            r.accuracy,
            r.digit_argmax_rate,
            r.armed_rate.map(|v| format!("{v:.3}")),
            r.decided.map(|v| format!("{v:.3}"))
        );
    }

    let mut report = CellReport {
        issue: ISSUE_TAG,
        checkpoint: args.model.to_string_lossy().into_owned(),
        checkpoint_blake3,
        task,
        prompt_format: format,
        split_seed: SPLIT_SEED,
        split_digest: sdig,
        probe_layers: host.probe_layers.clone(),
        ridge: RIDGE,
        switch_threshold: SWITCH_THRESHOLD,
        bias_delta: BIAS_DELTA,
        scramble_seed: SCRAMBLE_SEED,
        n_embd: host.config.n_embd,
        solved_n: solved_samples.len(),
        dev_n: dev_samples.len(),
        sealed_n: split.sealed.len(),
        solved_triple_coverage: triple_coverage,
        probe_corr,
        dev_rows,
        sealed_rows,
        payload_blake3: String::new(),
    };
    report.stamp_digest()?;
    let path = args.out.join("report.json");
    std::fs::write(&path, report.canonical_json()?)
        .with_context(|| format!("write {}", path.display()))?;
    println!(
        "# report {} (payload blake3 {})",
        path.display(),
        &report.payload_blake3[..16]
    );
    Ok(())
}
