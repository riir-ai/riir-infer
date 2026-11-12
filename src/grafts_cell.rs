//! `grafts_cell` — the katgpt-rs issue-988 T2b REAL-ACTIVATION cell
//! (riir-infer Issue 1206): real hidden states + real logits from a real
//! frozen model, closed-form fitted probes + a mined decision table +
//! computed logit-bias writes, evaluated against the model's own baseline
//! on held-out task instances.
//!
//! The PRIMITIVE lives in `katgpt-core` (`grafts` — reads/rules/writes/
//! switches, `Graft::apply`, `mined_table`, `scrambled`); this repo hosts
//! the capture seam + the harness. Synthetic-only fixtures (katgpt-rs
//! T2a) make digit readability true by construction — this cell is the
//! one that can actually fail.
//!
//! Laws (the issue pins them; the code only enforces):
//! - **Checkpoint:** `Qwen3.5-0.8B-Base-Q8_0.gguf` through the dense
//!   hybrid body (`load_qwen_deltanet_weights_gguf` +
//!   `forward_qwen_deltanet`) — the sink-audit census precedent
//!   (katgpt-rs issue 981). BLAKE3 of the file pinned in the report.
//! - **Split law:** solved/dev/sealed fixed by a pinned-seed shuffle
//!   BEFORE any graft is built; the sealed split is touched only at
//!   final eval; λ/threshold/bias-delta pinned a priori, never
//!   sealed-tuned.
//! - **One read path:** the fit and the eval both read through
//!   [`RowReader`] + `GraftRead::value_at` — the primitive's own
//!   quantization folds, never a parallel implementation (DRY; the
//!   round-half-away law has one home in katgpt-core).
//! - **Determinism:** same inputs → byte-identical report (BLAKE3 over
//!   the canonical payload, the sink_audit G1 law).
//! - **Honesty:** the scramble arm (wrong-by-construction wiring) runs
//!   on the sealed split; an `Undecidable` table is disclosed, never
//!   retried against sealed.
//!
//! Offline only — the serving path never mounts the capture (the field
//! is cfg-gated; a default build compiles it out entirely).

use anyhow::{Context, Result, bail};
use serde::Serialize;

use crate::quant::kvq_harness::SplitMix64;

// ── The capture seam ─────────────────────────────────────────────────────

/// Residual-stream capture for the graft lanes: the hidden state at the
/// END of each probe layer, at up to `SLOTS` armed positions (the
/// per-operand last-digit token positions — the run-A recency-gradient
/// lesson: reading every operand at the ANSWER position decayed the far
/// operands to ~24% linear readability; at its own position a digit is
/// still embedding-hot). Sized once; the forward writes rows in place.
pub const CAPTURE_SLOTS: usize = 3;

pub struct HiddenCapture {
    /// The probe layers (ascending; the forward binary-searches its own
    /// layer index here).
    layers: Vec<usize>,
    /// `(prompt position, slot)` arms — the forward writes the slot's
    /// rows when the processed position matches.
    armed: Vec<(usize, usize)>,
    n_embd: usize,
    /// `[slot][layer][embd]` — slot-major, then layer-major.
    rows: Vec<f32>,
}

impl HiddenCapture {
    /// Arm a capture for `layers` × `CAPTURE_SLOTS` slots at `n_embd`
    /// width. Panics on an empty layer set (the cell always probes ≥ 2
    /// layers — a loud caller bug).
    pub fn new(layers: &[usize], n_embd: usize) -> Self {
        assert!(!layers.is_empty(), "grafts capture: empty layer set");
        assert!(
            layers.windows(2).all(|w| w[0] < w[1]),
            "grafts capture: layer set must be ascending + distinct"
        );
        Self {
            layers: layers.to_vec(),
            armed: Vec::new(),
            n_embd,
            rows: vec![0.0; CAPTURE_SLOTS * layers.len() * n_embd],
        }
    }

    /// The probe layer ids (ascending).
    #[inline]
    pub fn layers(&self) -> &[usize] {
        &self.layers
    }

    /// The width of one row.
    #[inline]
    pub fn n_embd(&self) -> usize {
        self.n_embd
    }

    /// Arm the slot→position map (replaces any previous arms). The
    /// forward calls [`Self::capture`] per layer; only the armed
    /// positions write.
    #[inline]
    pub fn arm_slots(&mut self, slot_positions: &[(usize, usize)]) {
        self.armed = slot_positions.to_vec();
    }

    /// Disarm (no position writes).
    #[inline]
    pub fn disarm(&mut self) {
        self.armed.clear();
    }

    /// The forward's per-layer hook (the dense body calls this at the
    /// end-of-layer residual with the position being processed). A
    /// layer the set does not name, or a position no slot arms, is a
    /// skip, never an error.
    #[inline]
    pub fn capture(&mut self, layer_idx: usize, pos: usize, x: &[f32]) {
        let Some(slot) = self.armed.iter().find(|(p, _)| *p == pos).map(|&(_, s)| s) else {
            return;
        };
        let Ok(k) = self.layers.binary_search(&layer_idx) else {
            return;
        };
        debug_assert_eq!(x.len(), self.n_embd, "grafts capture: width mismatch");
        let base = (slot * self.layers.len() + k) * self.n_embd;
        self.rows[base..base + self.n_embd].copy_from_slice(&x[..self.n_embd]);
    }

    /// The captured rows, slot-major then layer-major.
    #[inline]
    pub fn rows(&self) -> &[f32] {
        &self.rows
    }
}

/// The graft read id space: `id = slot * 100 + layer` — slot 0..2 (the
/// operand's last-digit token position), layer = the probe layer. The
/// `ActivationReader` maps ids to the captured rows; the reads'
/// segments name these ids (the trait is host-seamed: the host owns the
/// address space).
#[inline]
#[must_use]
pub fn read_id(slot: usize, layer: usize) -> u32 {
    assert!(slot < CAPTURE_SLOTS, "grafts read id: slot out of range");
    assert!(layer < 100, "grafts read id: layer out of range");
    (slot * 100 + layer) as u32
}

/// The `katgpt_core::graft::ActivationReader` over slot-major rows —
/// the ONE read path for fit and eval alike (a `ProbeSample`'s rows and
/// a `HiddenCapture`'s rows share the layout, so the same reader serves
/// both). `read(id)` resolves the id's (slot, layer) to the row slice.
pub struct RowReader<'a> {
    layers: Vec<u32>,
    n_embd: usize,
    rows: &'a [f32],
}

impl<'a> RowReader<'a> {
    /// Panics when `CAPTURE_SLOTS * layers.len() * n_embd != rows.len()`
    /// — a loud caller bug (the row buffer is the capture's own).
    pub fn new(layers: &[usize], n_embd: usize, rows: &'a [f32]) -> RowReader<'a> {
        assert_eq!(
            CAPTURE_SLOTS * layers.len() * n_embd,
            rows.len(),
            "grafts row reader: rows is slots × layers × n_embd"
        );
        Self {
            layers: layers.iter().map(|&l| l as u32).collect(),
            n_embd,
            rows,
        }
    }

    fn row_for(&self, id: u32) -> Option<&[f32]> {
        let slot = (id / 100) as usize;
        let layer = (id % 100) as usize;
        let k = self.layers.iter().position(|&l| l as usize == layer)?;
        let base = (slot * self.layers.len() + k) * self.n_embd;
        Some(&self.rows[base..base + self.n_embd])
    }
}

impl katgpt_core::graft::ActivationReader for RowReader<'_> {
    fn read(&self, id: u32) -> &[f32] {
        // A read naming an id the rows do not hold is a caller bug; the
        // empty slice trips the trait contract's loud length assert
        // downstream (with the read's own message) instead of a panic
        // here.
        self.row_for(id).unwrap_or(&[])
    }
}

// ── Task + split generation (deterministic, pinned) ──────────────────────

/// The task-space seed for the sampled instance space (pinned; distinct
/// instances, deterministic order).
pub const TASK_SPACE_SEED: u64 = 0x988_0BE7;

/// The modular-arithmetic task family (single-answer-token; the harness
/// refuses a multi-token answer loud).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum TaskKind {
    /// `(a+b+c) mod 10`, a,b,c ∈ 10..99 — 1000 seeded-distinct
    /// instances. TWO-DIGIT operands BY DESIGN (the run-A lesson): with
    /// single digits the model computes the raw sum and emits it
    /// multi-token; with 2-digit operands it cannot, and the answer
    /// surface becomes the pattern-following single-digit guess the
    /// graft corrects (the paper's shape: the model guesses, the
    /// internal state knows, the graft reads the state).
    ThreeOpAdd10,
    /// `(a*b) mod 10`, a,b ∈ 0..9 — 100 pairs (single digits; the model
    /// knows its times tables — the baseline row says how much
    /// headroom is left).
    TwoOpMul10,
    /// `(a+b) mod 10`, a,b ∈ 0..9 — 100 pairs.
    TwoOpAdd10,
}

impl TaskKind {
    /// The instance count of the space.
    #[must_use]
    pub fn space_len(&self) -> usize {
        match self {
            Self::ThreeOpAdd10 => 1000,
            Self::TwoOpMul10 | Self::TwoOpAdd10 => 100,
        }
    }

    /// Deterministic instance space: ThreeOpAdd10 samples 1000 distinct
    /// 2-digit triples from a pinned-seed stream (ascending draw order);
    /// the two-op tasks enumerate their full 100-pair spaces.
    #[must_use]
    pub fn instances(&self) -> Vec<Vec<u8>> {
        match self {
            Self::ThreeOpAdd10 => {
                let mut rng = SplitMix64(TASK_SPACE_SEED);
                let mut seen = std::collections::BTreeSet::new();
                while seen.len() < 1000 {
                    let a = 10 + (rng.next_u64() % 90) as u8;
                    let b = 10 + (rng.next_u64() % 90) as u8;
                    let c = 10 + (rng.next_u64() % 90) as u8;
                    seen.insert((a, b, c));
                }
                seen.into_iter().map(|(a, b, c)| vec![a, b, c]).collect()
            }
            Self::TwoOpMul10 => (0..100u32)
                .map(|i| vec![(i / 10) as u8, (i % 10) as u8])
                .collect(),
            Self::TwoOpAdd10 => (0..100u32)
                .map(|i| vec![(i / 10) as u8, (i % 10) as u8])
                .collect(),
        }
    }

    /// The answer for one instance (the cell's output value, 0..9).
    /// The sum widens to u32 FIRST — in u8, 98+98+80 wraps (276 → 20),
    /// and every wrapped instance silently stored a wrong answer (the
    /// (8,8,0) census conflict was the symptom; the baseline's tail
    /// rows are wrong the same way — the u8 overflow class).
    #[must_use]
    pub fn answer(&self, inst: &[u8]) -> u8 {
        match self {
            Self::ThreeOpAdd10 => {
                ((u32::from(inst[0]) + u32::from(inst[1]) + u32::from(inst[2])) % 10) as u8
            }
            Self::TwoOpMul10 => ((u32::from(inst[0]) * u32::from(inst[1])) % 10) as u8,
            Self::TwoOpAdd10 => ((u32::from(inst[0]) + u32::from(inst[1])) % 10) as u8,
        }
    }

    /// The pinned few-shot prime block (two solved examples — correct
    /// answers, style-only; the tested operands never appear). The
    /// ThreeOpAdd10 primes use 2-digit operands matching the task's
    /// operand class (the style pin includes the operand SHAPE).
    #[must_use]
    fn primes(&self) -> &'static str {
        match self {
            Self::ThreeOpAdd10 => "12+34+56 mod 10 = 2\n23+45+67 mod 10 = 5\n",
            Self::TwoOpAdd10 => "1+2 mod 10 = 3\n4+5 mod 10 = 9\n",
            Self::TwoOpMul10 => "1*2 mod 10 = 2\n3*4 mod 10 = 2\n",
        }
    }

    /// The prompt for one instance under `format`.
    #[must_use]
    pub fn prompt(&self, inst: &[u8], format: PromptFormat) -> String {
        let ops: Vec<String> = inst.iter().map(|d| d.to_string()).collect();
        let expr = match self {
            Self::ThreeOpAdd10 | Self::TwoOpAdd10 => ops.join("+"),
            Self::TwoOpMul10 => ops.join("*"),
        };
        match format {
            PromptFormat::Bare => format!("{expr} mod 10 ="),
            PromptFormat::QaWrap => format!("Q: {expr} mod 10 =\nA:"),
            PromptFormat::FewShot => format!("{}{expr} mod 10 =", self.primes()),
        }
    }
}

/// The pinned prompt format candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum PromptFormat {
    /// `"{expr} mod 10 ="`
    Bare,
    /// `"Q: {expr} mod 10 =\nA:"` (the wrapper gives the answer position a
    /// distinctive shape the base model has seen in instruction-shaped
    /// text).
    QaWrap,
    /// Two PINNED solved examples prime the continuation pattern, then
    /// the question: a base model follows the in-context `= <digit>`
    /// shape and the answer surface becomes the space-prefixed digit
    /// token the LM head actually ranks (the run-A lesson: unprimed,
    /// the greedy top-1 after `"="` is the bare space token — never a
    /// digit). The primes are correct answers; they pin the STYLE, not
    /// any tested operand.
    FewShot,
}

/// The pinned split: instance indices into the task space.
#[derive(Debug, Clone, Serialize)]
pub struct Split {
    /// Solved split — the probe/table/switch fit surface.
    pub solved: Vec<usize>,
    /// Dev split — disclosed tuning rows only.
    pub dev: Vec<usize>,
    /// Sealed split — touched ONLY at final eval.
    pub sealed: Vec<usize>,
}

/// Shuffle `0..n` with a pinned-seed Fisher–Yates and cut the splits.
/// The shuffle consumes the RNG in one fixed order; identical seed →
/// identical split (the split digest pins it).
/// # Errors
/// Refuses when solved+dev swallow the space (no sealed remainder).
pub fn make_split(n: usize, solved: usize, dev: usize, seed: u64) -> Result<Split> {
    if solved + dev >= n {
        bail!("split: solved {solved} + dev {dev} must leave a sealed remainder of {n}");
    }
    let mut order: Vec<usize> = (0..n).collect();
    let mut rng = SplitMix64(seed);
    for i in (1..order.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
    Ok(Split {
        solved: order[..solved].to_vec(),
        dev: order[solved..solved + dev].to_vec(),
        sealed: order[solved + dev..].to_vec(),
    })
}

/// BLAKE3 over the canonical split encoding (the split manifest digest
/// that goes in the report — pins the split against re-generation
/// drift).
#[must_use]
pub fn split_digest(task: TaskKind, format: PromptFormat, split: &Split, seed: u64) -> String {
    let mut h = blake3::Hasher::new();
    h.update(format!("task={task:?} format={format:?} seed={seed}\n").as_bytes());
    for (name, rows) in [
        ("solved", &split.solved),
        ("dev", &split.dev),
        ("sealed", &split.sealed),
    ] {
        h.update(name.as_bytes());
        for &i in rows {
            h.update(&(i as u64).to_le_bytes());
        }
    }
    h.finalize().to_hex().to_string()
}

// ── The fit surface ──────────────────────────────────────────────────────

/// The three operand last-digit TOKEN POSITIONS within the encoded
/// prompt (the capture slots). Layout-derived: the task expression
/// `"{a}+{b}+{c} mod 10 ="` tokenizes with digits as individual tokens
/// (the Qwen digit-run rule), so the suffix's last-digit offsets are
/// fixed: 1, 4, 7 (0-based within the expression). The positions are
/// asserted against the actual decode — a tokenizer drift refuses loud.
/// # Errors
/// Any asserted digit position decoding to the wrong digit.
pub fn digit_slot_positions(
    tok: &crate::tokenizer::BpeTokenizer,
    task: crate::grafts_cell::TaskKind,
    format: crate::grafts_cell::PromptFormat,
    inst: &[u8],
) -> Result<Vec<usize>> {
    let prompt = task.prompt(inst, format);
    let tokens = tok.encode(&prompt);
    let primes_len = match format {
        crate::grafts_cell::PromptFormat::FewShot => tok.encode(task.primes()).len(),
        _ => 0,
    };
    // Expression layout: [d1a, d1b, '+', d2a, d2b, '+', d3a, d3b, ...]
    // — the last-digit offsets within the expression.
    let expr_offsets: Vec<usize> = match inst.len() {
        3 => vec![1, 4, 7],
        2 => vec![1, 4],
        _ => bail!("digit slots: unsupported operand count {}", inst.len()),
    };
    let mut slots = Vec::with_capacity(expr_offsets.len());
    for (j, &off) in expr_offsets.iter().enumerate() {
        let pos = primes_len + off;
        let Some(&t) = tokens.get(pos) else {
            bail!(
                "digit slots: position {pos} out of range (prompt len {})",
                tokens.len()
            );
        };
        let expect = format!("{}", inst[j] % 10);
        let got = tok.decode(&[t]);
        if got != expect {
            bail!(
                "digit slots: position {pos} decodes to '{got}', expected '{expect}' — the expression layout shifted"
            );
        }
        slots.push(pos);
    }
    Ok(slots)
}

/// One instance's captured probe rows + the task answer — a fit-surface
/// row.
pub struct ProbeSample {
    /// The instance index into the task space.
    pub instance: usize,
    /// Captured rows, slot-major then layer-major (`CAPTURE_SLOTS ×
    /// probe_layers.len() × n_embd` — the [`HiddenCapture::rows`] layout,
    /// which [`RowReader`] resolves by `(slot, layer)` id).
    pub rows: Vec<f32>,
    /// The task answer (0..9).
    pub answer: u8,
    /// The instance's operands (the per-read last-digit targets derive
    /// from these).
    pub operands: Vec<u8>,
    /// Was the BASE model's argmax at this position wrong? (The switch
    /// label.)
    pub base_wrong: bool,
}

/// The pinned probe-layer plan: EARLY layers (fractions
/// {0.125, 0.25, 0.375}·n_layer, rounded, deduped, ascending) — the
/// digit-identity regime. The a-priori mid/late pin {0.5, 0.65, 0.8}
/// was revised by the pre-sealed diagnostics (2026-11-12, disclosed in
/// the bench record): at layers 12/15/19 the operand last digits read
/// at only 24/25/45% vs 10% chance (a recency gradient — the signal
/// decays with distance from the probe position); the early residual
/// stream carries the operands closer to their embeddings. Sealed was
/// never touched by any of this.
#[must_use]
pub fn probe_layers(n_layer: usize) -> Vec<usize> {
    let mut v: Vec<usize> = [0.125f64, 0.25, 0.375]
        .iter()
        .map(|f| ((f * n_layer as f64) as usize).max(1).min(n_layer - 1))
        .collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// Digit-token map for one prompt format: digit d (0..9) → the exact
/// continuation token id AT THE ANSWER POSITION. The form is
/// context-dependent (measured: after `"="` the model ranks the
/// space-prefixed digit token; after `"A:"` the bare form) — and BPE
/// string-start encoding can differ from mid-text continuation, so the
/// map is derived IN CONTEXT: encode the reference prompt, encode the
/// reference prompt with the digit appended, and the one-token delta is
/// the answer surface. This kills the run-A lesson's failure class
/// (a bare-form map silently mismatching the argmax → 0.0000 on every
/// task): the surface is exactly what a greedy decode would emit.
/// # Errors
/// Any digit whose in-context continuation is not exactly one token, or
/// two digits sharing one continuation id.
pub fn digit_tokens(tok: &crate::tokenizer::BpeTokenizer, reference: &str) -> Result<[usize; 10]> {
    let reference = reference.to_string();
    let base = tok.encode(&reference);
    let mut out = [usize::MAX; 10];
    for (d, slot) in out.iter_mut().enumerate() {
        let with = tok.encode(&format!("{reference}{d}"));
        let suffix = with
            .len()
            .checked_sub(base.len())
            .context("digit continuation shorter than base")?;
        if suffix != 1 {
            bail!(
                "digit token: '{d}' continues the reference prompt with {suffix} tokens (need exactly 1)"
            );
        }
        *slot = with[with.len() - 1];
    }
    let distinct = out.iter().collect::<std::collections::BTreeSet<_>>();
    if distinct.len() != 10 {
        bail!(
            "digit token: two digits share one continuation id — the answer surface is ambiguous"
        );
    }
    Ok(out)
}

/// The fit matrix over `samples`: row-major `n × (k·n_embd)`, layer-major
/// within a row (the `fit_read`/`fit_switch` segment order).
#[must_use]
pub fn fit_matrix(samples: &[ProbeSample]) -> Vec<f32> {
    let mut x = Vec::new();
    for s in samples {
        x.extend_from_slice(&s.rows);
    }
    x
}

/// The fitted parts (reads + table + switch) with fit-surface
/// diagnostics.
pub struct FitParts {
    pub reads: Vec<katgpt_core::graft::GraftRead>,
    pub table: katgpt_core::graft::GraftTable,
    pub switch: katgpt_core::graft::GraftSwitch,
    /// The solved-split quantized read-triples (diagnostics: the table's
    /// in-fit input space).
    pub triples: Vec<Vec<u16>>,
}

/// Fit the three graft parts from solved-split samples, consuming the
/// katgpt-core primitive as-is: one `GraftRead` per probe layer
/// (targets = the answer), the table mined over the quantized
/// read-triples, and a `GraftSwitch` on the {base-wrong} labels.
///
/// # Errors
/// Any katgpt-core fit error passes through, including `Undecidable` —
/// disclosed, never retried against sealed.
pub fn fit_parts(
    samples: &[ProbeSample],
    probe_layers: &[usize],
    n_embd: usize,
    ridge: f32,
) -> Result<FitParts, katgpt_core::graft::GraftError> {
    use katgpt_core::graft::{GraftExample, fit_read, fit_switch, mined_table};

    if samples.is_empty() {
        return Err(katgpt_core::graft::GraftError::Usage(
            "grafts cell: empty fit surface",
        ));
    }
    let k = probe_layers.len();
    let d = k * n_embd;
    let x = fit_matrix(samples);
    let labels: Vec<bool> = samples.iter().map(|s| s.base_wrong).collect();

    // ONE READ PER OPERAND (the λ-sweep redesign): the mod-10 answer is
    // a WRAP (piecewise-linear) function of the digit sum — no linear
    // probe closes the wrap (measured: 32-45% bucket conflicts at every
    // λ). The paper's own split of labor: probes read LINEAR features
    // (here: each operand's last digit), and the mined TABLE decides
    // the nonlinear wrap from the digit reads. Read j targets operand
    // j's last digit; its segments read the operand's OWN token
    // position (slot j) at every probe layer — the run-A recency
    // gradient measured all-at-one-position reads decaying to ~24-45%;
    // at its own position a digit is still embedding-hot.
    let n_reads = samples[0].operands.len();
    assert!(
        n_reads <= CAPTURE_SLOTS,
        "grafts cell: one read per operand needs CAPTURE_SLOTS slots"
    );
    let seg_dims: Vec<usize> = vec![n_embd; k];
    let mut reads = Vec::with_capacity(n_reads);
    for j in 0..n_reads {
        // Read j's surface = slot j's rows only (k layers × n_embd wide,
        // contiguous within each sample's slot-major row block).
        let mut xs = Vec::with_capacity(samples.len() * d);
        for s in samples {
            xs.extend_from_slice(&s.rows[j * d..(j + 1) * d]);
        }
        let layers_u32: Vec<u32> = probe_layers.iter().map(|&l| read_id(j, l)).collect();
        let targets: Vec<f32> = samples
            .iter()
            .map(|s| f32::from(s.operands[j] % 10))
            .collect();
        reads.push(fit_read(
            &xs,
            samples.len(),
            d,
            &layers_u32,
            &seg_dims,
            &targets,
            ridge,
            10,
            true,
        )?);
    }

    // Affine calibration per read, pinned on the SOLVED split only:
    // ridge shrinkage compresses scores relative to the answer scale
    // (measured: a 10-bucket domain read a 4.5-wide spread — bucket
    // collisions were the Undecidable root cause), so the cell
    // least-squares-fits `answer ~ score` and FOLDS the correction into
    // the read's own coefficients (w' = α·w, b' = α·b + β): the folded
    // score becomes α·score + β and `value_at` needs no change. The
    // primitive is untouched; the statistics never see dev/sealed.
    for (j, read) in reads.iter_mut().enumerate() {
        // Slot j's row block starts at j*d and holds k layer rows — the
        // score folds every segment against ITS layer's slice (the
        // pre-slot layout sliced j*n_embd — INSIDE slot 0's first layer
        // — and calibrated against garbage scores, corrupting the read).
        let off = j * d;
        let (mut sx, mut sy, mut sxx, mut sxy) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let n = samples.len() as f64;
        for s in samples {
            let mut sc = read.bias;
            for (seg_i, seg) in read.segments.iter().enumerate() {
                let row = &s.rows[off + seg_i * n_embd..off + (seg_i + 1) * n_embd];
                sc += seg.score(row);
            }
            // The calibration target is the read's OWN fit target — the
            // operand's last digit (the answer-wrap story is the
            // TABLE's job, not the read's).
            let (x, y) = (f64::from(sc), f64::from(s.operands[j] % 10));
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
        }
        let denom = n * sxx - sx * sx;
        // The NEGATED comparison is deliberate (the fit-lane law): a NaN
        // denom must refuse (NaN > 1e-9 is false — the positive form
        // cannot express the refusal).
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(denom > 1e-9) {
            return Err(katgpt_core::graft::GraftError::Usage(
                "grafts cell: degenerate score spread (calibration refused)",
            ));
        }
        // The TRUE sign of the slope is load-bearing: a negative slope
        // (the ridge fit converged to an anti-correlated direction —
        // measured) calibrates just as validly through α < 0; clamping
        // it positive flipped every mapping into β's neighborhood and
        // collapsed the reads to ~1 bucket (the 2-distinct-triples
        // census). Only true degeneracy (|α| ≈ 0) refuses.
        let alpha = (n * sxy - sx * sy) / denom;
        // The NEGATED comparison is deliberate (the fit-lane law): a
        // NaN alpha must refuse.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(alpha.abs() > 1e-6) {
            return Err(katgpt_core::graft::GraftError::Usage(
                "grafts cell: zero score-answer slope (calibration refused)",
            ));
        }
        let beta = (sy - alpha * sx) / n;
        for seg in &mut read.segments {
            for w in &mut seg.weights {
                *w *= alpha as f32;
            }
        }
        read.bias = (alpha * f64::from(read.bias) + beta) as f32;
    }

    // Quantized read-triples over the solved split — through the REAL
    // read path (`value_at` over a row reader), never a parallel fold.
    let mut examples = Vec::with_capacity(samples.len());
    let mut triples = Vec::with_capacity(samples.len());
    for s in samples.iter() {
        let reader = RowReader::new(probe_layers, n_embd, &s.rows);
        let triple: Vec<u16> = reads.iter().map(|r| r.value_at(&reader)).collect();
        triples.push(triple.clone());
        examples.push(GraftExample {
            input: triple,
            output: vec![u16::from(s.answer)],
        });
    }
    // The conflict census (pre-mine): how many quantized triples carry
    // >1 distinct answer on the solved split — the exact Undecidable
    // driver, printed before the mine refuses so the re-pin is guided
    // by numbers, not by a bare loud error.
    let mut triple_answers: std::collections::HashMap<Vec<u16>, std::collections::BTreeSet<u8>> =
        std::collections::HashMap::new();
    for (ex, s) in examples.iter().zip(samples.iter()) {
        triple_answers
            .entry(ex.input.clone())
            .or_default()
            .insert(s.answer);
    }
    let conflicted = triple_answers.values().filter(|v| v.len() > 1).count();
    println!(
        "# conflict census: {} distinct triples, {} conflicted (carry >1 answer) — the Undecidable driver",
        triple_answers.len(),
        conflicted
    );
    let table = mined_table(&examples, k, 1, 3)?;

    // The switch: labels = {base-argmax wrong} over the same surface;
    // segments = every slot's every probe layer (the gate reads the
    // full captured state, slot-qualified ids).
    let all_ids: Vec<u32> = (0..n_reads)
        .flat_map(|slot| probe_layers.iter().map(move |&l| read_id(slot, l)))
        .collect();
    let switch = fit_switch(
        &x,
        samples.len(),
        d * n_reads,
        &all_ids,
        &vec![n_embd; k * n_reads],
        &labels,
        ridge,
        0.5,
        true,
    )?;

    Ok(FitParts {
        reads,
        table,
        switch,
        triples,
    })
}

// ── Sample cache (the λ re-pin lane's capture-pass elision) ──────────────

/// The sample-cache row (serde form of [`ProbeSample`]): the λ re-pin
/// lane and the fixture exporter (the `grafts_export` bin) keep the
/// ~23-minute capture pass out of every experiment.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct SampleRow {
    pub instance: usize,
    pub rows: Vec<f32>,
    pub answer: u8,
    pub operands: Vec<u8>,
    pub base_wrong: bool,
}

impl SampleRow {
    #[must_use]
    pub fn from_sample(s: &ProbeSample) -> Self {
        Self {
            instance: s.instance,
            rows: s.rows.clone(),
            answer: s.answer,
            operands: s.operands.clone(),
            base_wrong: s.base_wrong,
        }
    }
}

impl From<SampleRow> for ProbeSample {
    fn from(r: SampleRow) -> Self {
        Self {
            instance: r.instance,
            rows: r.rows,
            // The stored answer may predate the u8-overflow fix — callers
            // re-derive it from the operands (a pure function of them;
            // a stale stored value carries no load).
            answer: r.answer,
            operands: r.operands,
            base_wrong: r.base_wrong,
        }
    }
}

/// Load a `(solved, dev)` sample cache written by the cell's
/// `--save-samples`.
/// # Errors
/// IO or JSON failure.
pub fn load_samples(path: &std::path::Path) -> Result<(Vec<ProbeSample>, Vec<ProbeSample>)> {
    let (s, d): (Vec<SampleRow>, Vec<SampleRow>) = serde_json::from_reader(
        std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?,
    )
    .context("load samples")?;
    Ok((
        s.into_iter().map(Into::into).collect(),
        d.into_iter().map(Into::into).collect(),
    ))
}

/// Write a `(solved, dev)` sample cache (the cell's `--save-samples` dual).
/// # Errors
/// IO or JSON failure.
pub fn save_samples(
    path: &std::path::Path,
    solved: &[ProbeSample],
    dev: &[ProbeSample],
) -> Result<()> {
    let s_ser: Vec<SampleRow> = solved.iter().map(SampleRow::from_sample).collect();
    let d_ser: Vec<SampleRow> = dev.iter().map(SampleRow::from_sample).collect();
    std::fs::write(
        path,
        serde_json::to_vec(&(&s_ser, &d_ser)).context("serialize samples")?,
    )
    .with_context(|| format!("write {}", path.display()))
}

// ── Report (BLAKE3 over the canonical payload — the G1 law) ──────────────

/// One arm's split verdict row.
#[derive(Debug, Clone, Serialize)]
pub struct ArmRow {
    pub arm: String,
    /// Exact-match accuracy (argmax decodes to the answer digit token).
    pub accuracy: f64,
    /// The rate the argmax was A digit token at all (wrong-digit rows
    /// count here but not in `accuracy`).
    pub digit_argmax_rate: f64,
    pub n: usize,
    /// Mean decided cells per apply (the graft arms only).
    pub decided: Option<f64>,
    /// Armed rate per apply (the graft arms only).
    pub armed_rate: Option<f64>,
}

/// The cell's report — `serde_json` of a struct is field-order-stable,
/// so the canonical payload is deterministic.
#[derive(Debug, Serialize)]
pub struct CellReport {
    pub issue: &'static str,
    pub checkpoint: String,
    pub checkpoint_blake3: String,
    pub task: TaskKind,
    pub prompt_format: PromptFormat,
    pub split_seed: u64,
    pub split_digest: String,
    pub probe_layers: Vec<usize>,
    pub ridge: f32,
    pub switch_threshold: f32,
    pub bias_delta: f32,
    pub scramble_seed: u64,
    pub n_embd: usize,
    pub solved_n: usize,
    pub dev_n: usize,
    pub sealed_n: usize,
    /// In-fit table input-space coverage on the solved split (distinct
    /// triples / samples — the mining is exact by construction; the row
    /// discloses how degenerate the quantized space is).
    pub solved_triple_coverage: f64,
    /// Probe-vs-answer Pearson correlation per probe layer (solved
    /// split, f64 sequential folds).
    pub probe_corr: Vec<f64>,
    /// Dev-split rows (the disclosed tuning surface — never sealed-tuned).
    pub dev_rows: Vec<ArmRow>,
    pub sealed_rows: Vec<ArmRow>,
    /// BLAKE3 over the canonical payload bytes WITH this field empty
    /// (the sink_audit shape: a verifier blanks the field, re-serializes,
    /// re-hashes, compares).
    pub payload_blake3: String,
}

impl CellReport {
    /// Canonical JSON bytes (the payload the digest pins).
    /// # Errors
    /// Serialization failure (a report is always serializable; loud
    /// anyway).
    pub fn canonical_json(&self) -> Result<Vec<u8>> {
        serde_json::to_vec_pretty(self).context("serialize cell report")
    }

    /// Stamp `payload_blake3` — call once after all other fields are
    /// set. The digest covers the payload with the field EMPTY.
    /// # Errors
    /// Serialization failure.
    pub fn stamp_digest(&mut self) -> Result<()> {
        self.payload_blake3 = String::new();
        let bytes = self.canonical_json()?;
        self.payload_blake3 = blake3::hash(&bytes).to_hex().to_string();
        Ok(())
    }
}

/// Pearson correlation (f64 sequential folds — an offline artifact, the
/// Issue-940 law's offline cousin).
#[must_use]
pub fn pearson(xs: &[f32], ys: &[f32]) -> f64 {
    let n = xs.len();
    if n == 0 || n != ys.len() {
        return f64::NAN;
    }
    let (mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let (x, y) = (f64::from(xs[i]), f64::from(ys[i]));
        sx += x;
        sy += y;
        sxx += x * x;
        syy += y * y;
        sxy += x * y;
    }
    let nf = n as f64;
    let cov = sxy - sx * sy / nf;
    let vx = sxx - sx * sx / nf;
    let vy = syy - sy * sy / nf;
    // The NEGATED comparison is deliberate (the fit-lane law): a NaN
    // variance must refuse (NaN > 0 is false, so the negated form
    // catches it — the positive form cannot express the refusal).
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(vx > 0.0) || !(vy > 0.0) {
        return f64::NAN;
    }
    cov / (vx.sqrt() * vy.sqrt())
}
