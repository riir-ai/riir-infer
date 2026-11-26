//! `implants_cell` — the Issue-1256 T1 cell machinery: the decode-seam
//! walk (observe → fire → bias-fold → argmax), the synthetic fixture
//! source, and the four pre-registered cells.
//!
//! Two postures run the SAME cells through the SAME code:
//! - **fixture tier** ([`SyntheticSource`], deterministic hash logits,
//!   no checkpoint) — the CI-runnable gate (`tests/implants_cell_gates.rs`);
//! - **real tier** (the bin's host over the league checkpoint, CPU,
//!   same-device scope — the bench-971 host shape).
//!
//! Cells (the issue's pre-registered bars):
//! - **G1 silent identity** ([`cell_g1_silent_identity`]) — armed-but-
//!   silent mounted ≡ unmounted per step (logit-bit digests + generated
//!   tokens), and the demoted mount on a row it WOULD fire on ≡ base.
//! - **G1b false fire** ([`cell_g1b_false_fire`]) — 0 fires on every
//!   look-alike + prose row (paraphrases, operand swaps, adjacent
//!   operators, multi-digit, negative results, plain prose).
//! - **G2 fired exactness** ([`cell_g2_fired_exactness`]) — every fired
//!   span emits its forced token exactly; prompt-phase logits are
//!   bit-identical to base up to the fired step; look-alike rows are
//!   walk-identical to base (the co-measured non-fired spans).
//! - **G3 composition** ([`cell_g3_composition`]) — 2 implants
//!   co-mounted: each fires alone on its own family, silent on the
//!   other's, zero interference (co-mounted walk ≡ single-mounted walk),
//!   combined exactness ≥ each single.
//!
//! Fixture contamination law: every fixture is authored in this module
//! (synthetic look-alikes per the WSM §4.1 process / the
//! grammar-not-answers doctrine) — no benchmark test rows anywhere.
//!
//! The bias pin: [`IMPLANT_BIAS`] 8.0 is the grafts_cell BIAS_DELTA
//! precedent (the measured base top-1 gap after `=` is ~5 logits; 8
//! clears it without tuning — pinned a priori, never swept).

use crate::implants::{
    FireRef, ImplantAction, ImplantArtifact, ImplantError, ImplantRuntime, ImplantSpec, TokenPat,
    TriggerRule, WINDOW_MAX,
};

/// The forced-token bias (the grafts_cell BIAS_DELTA 8.0 precedent: the
/// measured base top-1 gap after `=` is ~5 logits; 8 clears it, pinned
/// a priori, never swept).
pub const IMPLANT_BIAS: f32 = 8.0;

/// The synthetic source's "base model" favorite bonus after `=` —
/// mirrors the measured ~5-logit space-token gap (bench 971) so the
/// fixture tier exercises the same forced-vs-base margin the real tier
/// sees.
const SYNTH_FAV_BONUS: f32 = 5.0;

/// Demote env for the addition implant (the drift_guard posture: exact
/// literal `0` at mount).
pub const ADD_DEMOTE_ENV: &str = "RIIR_IMPLANTS_CELL_ADD_DEMOTE";
/// Demote env for the subtraction implant.
pub const SUB_DEMOTE_ENV: &str = "RIIR_IMPLANTS_CELL_SUB_DEMOTE";

// ── Fixture families (authored, contamination-free) ─────────────────────

/// Single-digit addition pairs (a+b ≤ 9 so the answer is one token).
/// (5,5)=10 excluded — the answer must stay single-digit.
pub fn add_pairs() -> Vec<(u8, u8)> {
    let mut v = Vec::new();
    for a in 2..=5_u8 {
        for b in 2..=5_u8 {
            if a + b <= 9 {
                v.push((a, b));
            }
        }
    }
    v
}

/// Single-digit subtraction pairs (positive result, one token).
pub fn sub_pairs() -> Vec<(u8, u8)> {
    let mut v = Vec::new();
    for a in 4..=7_u8 {
        for b in 2..a {
            v.push((a, b));
        }
    }
    v
}

/// Look-alike rows: none is in either trigger table. Classes: operand
/// swaps of UNTABLED pairs, paraphrases, adjacent operators, multi-digit
/// operands, negative results.
pub const LOOKALIKE_PROMPTS: &[&str] = &[
    "6+7=",
    "7+6=",
    "16+27=",
    "6*7=",
    "What is 6 plus 7?",
    "3-9=",
    "2-7=",
    "12-5=",
    "9*2=",
    "What is 9 minus 2?",
];

/// Plain prose rows (no trigger anywhere).
pub const PROSE_PROMPTS: &[&str] = &[
    "The quick brown fox.",
    "Hello world.",
    "abcd efgh",
    "zz zz zz.",
];

// ── The fixture codec ────────────────────────────────────────────────────

/// How a tier turns fixture strings into token ids and resolves the
/// digit answer surface. The synthetic tier uses a fixed char map; the
/// real tier wraps the checkpoint's BPE tokenizer (the continuation
/// trick — bench 971's context-derived surface).
pub trait FixtureCodec {
    /// Encode a fixture prompt.
    ///
    /// # Errors
    /// Tier-specific: an unencodable character (synthetic) or a
    /// tokenizer failure (real).
    fn encode(&self, s: &str) -> Result<Vec<usize>, String>;
    /// Decode ONE token (pattern-shape validation only).
    fn decode_one(&self, t: usize) -> String;
    /// The token id the digit `d` takes when CONTINUING `prompt` (the
    /// answer surface). Refuses when the digit is not exactly one token.
    ///
    /// # Errors
    /// The continuation is not exactly one token, or the continuation id
    /// is not the digit's own id (synthetic).
    fn digit_continuation(&self, prompt: &str, d: u8) -> Result<usize, String>;
}

// Synthetic vocabulary: digits 0..=9, '+' 10, '-' 11, '=' 12, '*' 13,
// '?' 14, 'a'..'z' 15..=40, '.' 41, ' ' 42.
const SY_PLUS: usize = 10;
const SY_MINUS: usize = 11;
const SY_EQ: usize = 12;
const SY_STAR: usize = 13;
const SY_DOT: usize = 41;
const SY_SPACE: usize = 42;
const SY_VOCAB: usize = 48;
/// The synthetic "favorite" after `=`: the letter 'x' (a WRONG answer
/// posture the implants must override — the bench-971 space-token lesson
/// mirrored).
const SY_FAV: usize = 15 + (b'x' - b'a') as usize;

fn syn_char(c: char) -> Option<usize> {
    let id = match c {
        '0'..='9' => c as u8 - b'0',
        '+' => SY_PLUS as u8,
        '-' => SY_MINUS as u8,
        '=' => SY_EQ as u8,
        '*' => SY_STAR as u8,
        '?' => 14,
        '.' => SY_DOT as u8,
        ' ' => SY_SPACE as u8,
        'a'..='z' => 15 + (c as u8 - b'a'),
        'A'..='Z' => 15 + (c as u8 - b'A'),
        _ => return None,
    } as usize;
    Some(id)
}

fn syn_token(t: usize) -> String {
    match t {
        0..=9 => (t as u8 + b'0') as char,
        SY_PLUS => '+',
        SY_MINUS => '-',
        SY_EQ => '=',
        SY_STAR => '*',
        14 => '?',
        SY_DOT => '.',
        SY_SPACE => ' ',
        15..=40 => (b'a' + (t - 15) as u8) as char,
        _ => '\u{FFFD}',
    }
    .to_string()
}

/// The fixture-tier codec (fixed char map, no tokenizer).
pub struct SyntheticCodec;

impl FixtureCodec for SyntheticCodec {
    fn encode(&self, s: &str) -> Result<Vec<usize>, String> {
        let mut out = Vec::with_capacity(s.len());
        for c in s.chars() {
            let Some(id) = syn_char(c) else {
                return Err(format!("synthetic codec: unencodable char {c:?} in {s:?}"));
            };
            out.push(id);
        }
        Ok(out)
    }

    fn decode_one(&self, t: usize) -> String {
        syn_token(t)
    }

    fn digit_continuation(&self, prompt: &str, d: u8) -> Result<usize, String> {
        let base = self.encode(prompt)?;
        let with = self.encode(&format!("{prompt}{d}"))?;
        if with.len() != base.len() + 1 {
            return Err(format!("digit {d}: continuation is not one token"));
        }
        let last = *with.last().expect("non-empty by the length check");
        if last != d as usize {
            return Err(format!(
                "digit {d}: continuation id {last} is not the digit"
            ));
        }
        Ok(last)
    }
}

// ── The synthetic logits source ──────────────────────────────────────────

/// Deterministic hash-based logits (the fixture tier's "model"): per-slot
/// noise in [0, 1) from a splitmix64 finalizer over (slot, token, pos),
/// plus a favorite bonus after `=` — a deterministic WRONG-answer posture
/// the implants must override. Integer hashing only: identical bytes on
/// every platform.
pub struct SyntheticSource {
    salt: u64,
}

impl SyntheticSource {
    /// A source with the pinned lane salt (one value, never swept).
    #[must_use]
    pub fn new() -> Self {
        Self {
            salt: 0x1256_9880_5EED_0001,
        }
    }

    fn pseudo(&self, v: usize, token: usize, pos: usize) -> f32 {
        let mut x = (v as u64) ^ 0x9E37_79B9_7F4A_7C15;
        x ^= (token as u64).wrapping_mul(0xFF51_AFD7_ED55_8CCD);
        x ^= (pos as u64).wrapping_mul(0xC4CE_B9FE_1A85_EC53);
        x ^= self.salt;
        x = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
        x ^= x >> 31;
        ((x >> 11) as f64 / (1u64 << 53) as f64) as f32
    }
}

impl Default for SyntheticSource {
    fn default() -> Self {
        Self::new()
    }
}

impl LogitsSource for SyntheticSource {
    fn vocab(&self) -> usize {
        SY_VOCAB
    }

    fn step(&mut self, token: usize, pos: usize, out: &mut [f32]) {
        let fav = if token == SY_EQ { SY_FAV } else { SY_SPACE };
        for (v, slot) in out.iter_mut().enumerate() {
            *slot = self.pseudo(v, token, pos) + if v == fav { SYNTH_FAV_BONUS } else { 0.0 };
        }
    }
}

// ── The decode seam ──────────────────────────────────────────────────────

/// A logits provider for one decode walk. `step` fills `out` (sized to
/// [`LogitsSource::vocab`]) with the logits for `token` at `pos` — the
/// distribution over the NEXT token.
pub trait LogitsSource {
    /// The vocabulary size (the logits width).
    fn vocab(&self) -> usize;
    /// One forward step; `out` is exactly [`LogitsSource::vocab`] wide.
    fn step(&mut self, token: usize, pos: usize, out: &mut [f32]);
    /// Drop any cross-row stream state (the real host's KV cache). The
    /// walk calls this at its start — every row decodes independently.
    fn reset(&mut self) {}
}

/// One fired rule resolved at a decode step (where it fired, what it
/// wrote — telemetry and assertion input).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FireStep {
    /// The walk step index (0-based over prompt + generation steps).
    pub step: usize,
    /// The implant index (mount order).
    pub implant: u16,
    /// The fired action index within that implant.
    pub action: u16,
    /// Top-1 and top-2 (id, logit) at the fired step, AFTER the bias
    /// fold — the diagnosis input when an exactness bar fails.
    pub top2: [(usize, f32); 2],
}

/// One decode walk's outcome.
#[derive(Debug, Clone)]
pub struct WalkOutcome {
    /// The generated tokens (the prompt is not included).
    pub tokens: Vec<usize>,
    /// Per-step BLAKE3 over the POST-fold f32 bit patterns (prompt steps
    /// + generation steps). The byte-identity comparator.
    pub step_digests: Vec<[u8; 32]>,
    /// The fires, in order.
    pub fires: Vec<FireStep>,
}

/// Reused walk buffers (fire buffer + logits scratch — the G4 scratch
/// discipline; allocation happens once here, never per walk when the
/// caller reuses the struct).
pub struct WalkBufs {
    fire: Vec<FireRef>,
    logits: Vec<f32>,
}

impl WalkBufs {
    /// Buffers sized for a vocabulary.
    #[must_use]
    pub fn new(vocab: usize) -> Self {
        Self {
            fire: Vec::new(),
            logits: vec![0.0; vocab],
        }
    }
}

/// First-index argmax (the Issue-940 strict tie rule: earliest index
/// wins; never a tolerance band, never map order).
fn argmax_first(xs: &[f32]) -> usize {
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

fn top2(xs: &[f32]) -> [(usize, f32); 2] {
    let mut best = [(0usize, f32::NEG_INFINITY); 2];
    for (i, &v) in xs.iter().enumerate() {
        if v > best[0].1 {
            best[1] = best[0];
            best[0] = (i, v);
        } else if v > best[1].1 {
            best[1] = (i, v);
        }
    }
    best
}

/// One forward + observe + fold + digest step.
fn step_and_observe<S: LogitsSource>(
    source: &mut S,
    runtime: &mut ImplantRuntime,
    bufs: &mut WalkBufs,
    token: usize,
    pos: usize,
    fires: &mut Vec<FireStep>,
    step: usize,
) -> [u8; 32] {
    {
        let logits: &mut [f32] = &mut bufs.logits;
        source.step(token, pos, logits);
        bufs.fire.clear();
        runtime.observe_into(token as u32, &mut bufs.fire);
        for fire in bufs.fire.iter() {
            let action = runtime.action(fire);
            for &(t, b) in &action.biases {
                logits[t as usize] += b;
            }
            fires.push(FireStep {
                step,
                implant: fire.implant,
                action: fire.action,
                top2: top2(logits),
            });
        }
    }
    let mut hasher = blake3::Hasher::new();
    for v in &bufs.logits {
        hasher.update(&v.to_bits().to_le_bytes());
    }
    let mut d = [0u8; 32];
    d.copy_from_slice(hasher.finalize().as_bytes());
    d
}

/// The decode-seam walk: process `prompt`, then greedily generate up to
/// `max_new` tokens through the implant runtime (observe → fire →
/// bias-fold → argmax, first-index ties). The runtime's token window is
/// the caller's responsibility (a fresh [`ImplantRuntime`] per fixture
/// row keeps rows independent — serving keeps one live window).
#[must_use]
pub fn decode_walk<S: LogitsSource>(
    source: &mut S,
    prompt: &[usize],
    max_new: usize,
    runtime: &mut ImplantRuntime,
    bufs: &mut WalkBufs,
) -> WalkOutcome {
    debug_assert_eq!(bufs.logits.len(), source.vocab(), "logits width mismatch");
    source.reset();
    let mut tokens = Vec::with_capacity(max_new);
    let mut step_digests = Vec::with_capacity(prompt.len() + max_new);
    let mut fires = Vec::new();
    for (pos, &t) in prompt.iter().enumerate() {
        let d = step_and_observe(source, runtime, bufs, t, pos, &mut fires, pos);
        step_digests.push(d);
    }
    for (i, cur_pos) in (0..max_new).zip(prompt.len()..) {
        let next = argmax_first(&bufs.logits);
        tokens.push(next);
        let step = prompt.len() + i;
        let d = step_and_observe(source, runtime, bufs, next, cur_pos, &mut fires, step);
        step_digests.push(d);
    }
    WalkOutcome {
        tokens,
        step_digests,
        fires,
    }
}

// ── Lane authoring ───────────────────────────────────────────────────────

/// One trigger-table row (the assertion input: which implant fires, what
/// must be emitted).
#[derive(Debug, Clone)]
pub struct TableRow {
    /// The fixture prompt (its full tokenization is the trigger pattern).
    pub prompt: String,
    /// The implant index in mount order (0 = addition, 1 = subtraction).
    pub implant: u16,
    /// The forced token id (the answer digit's continuation token).
    pub forced_token: usize,
}

/// The two authored implant lanes + their trigger table.
#[derive(Debug, Clone)]
pub struct AuthoredLanes {
    /// The addition implant (`{a}+{b}=` → the sum digit).
    pub add: ImplantSpec,
    /// The subtraction implant (`{a}-{b}=` → the difference digit).
    pub sub: ImplantSpec,
    /// Every tabled row (addition rows then subtraction rows).
    pub table: Vec<TableRow>,
}

/// Author both lanes through `codec`: each row's FULL prompt tokenization
/// becomes one trigger rule (unique by construction — operand order is
/// in the pattern), whose action forces the answer digit's continuation
/// token at [`IMPLANT_BIAS`]. Refuses loud on any tokenization shape
/// violation (a digit split across tokens, a missing `=` tail, a pattern
/// over [`WINDOW_MAX`]) — the cell never runs on a silently wrong table.
///
/// # Errors
/// Any encode/validation failure, or an [`ImplantSpec`] construction
/// refusal (they cannot fire for well-formed fixtures — returned as text).
pub fn author_lanes<C: FixtureCodec>(codec: &C) -> Result<AuthoredLanes, String> {
    let mut table = Vec::new();
    let add = author_family(
        codec,
        "i1256_add10",
        ADD_DEMOTE_ENV,
        '+',
        &add_pairs(),
        &mut table,
    )?;
    let sub = author_family(
        codec,
        "i1256_sub10",
        SUB_DEMOTE_ENV,
        '-',
        &sub_pairs(),
        &mut table,
    )?;
    Ok(AuthoredLanes { add, sub, table })
}

fn author_family<C: FixtureCodec>(
    codec: &C,
    name: &str,
    demote_env: &str,
    op: char,
    pairs: &[(u8, u8)],
    table: &mut Vec<TableRow>,
) -> Result<ImplantSpec, String> {
    // Addition mounts at slot 0, subtraction at slot 1 (table order).
    let idx = u16::from(!table.is_empty());
    let mut rules = Vec::with_capacity(pairs.len());
    let mut actions = Vec::with_capacity(pairs.len());
    for &(a, b) in pairs {
        let prompt = format!("{a}{op}{b}=");
        let answer = if op == '+' { a + b } else { a - b };
        debug_assert!(answer <= 9, "fixture answers stay single-digit");
        let toks = codec.encode(&prompt)?;
        // Shape validation: last token is '=', second-to-last is the
        // second operand, first token is the first operand (the answer
        // surface derivation needs the exact prompt tail).
        if toks.len() < 3 {
            return Err(format!(
                "{name}: prompt {prompt:?} tokenizes under 3 tokens"
            ));
        }
        if codec.decode_one(toks[toks.len() - 1]) != "=" {
            return Err(format!(
                "{name}: prompt {prompt:?} does not end in an '=' token ({:?})",
                codec.decode_one(toks[toks.len() - 1])
            ));
        }
        if codec.decode_one(toks[toks.len() - 2]) != b.to_string() {
            return Err(format!(
                "{name}: operand {b} of {prompt:?} split across tokens ({:?})",
                codec.decode_one(toks[toks.len() - 2])
            ));
        }
        if codec.decode_one(toks[0]) != a.to_string() {
            return Err(format!(
                "{name}: operand {a} of {prompt:?} split across tokens ({:?})",
                codec.decode_one(toks[0])
            ));
        }
        if toks.len() > WINDOW_MAX {
            return Err(format!(
                "{name}: prompt {prompt:?} is {len} tokens, over WINDOW_MAX {WINDOW_MAX}",
                len = toks.len()
            ));
        }
        let forced = codec.digit_continuation(&prompt, answer)?;
        rules.push(TriggerRule {
            pattern: toks.iter().map(|&t| TokenPat::Exact(t as u32)).collect(),
            action: actions.len() as u16,
        });
        actions.push(ImplantAction {
            biases: vec![(forced as u32, IMPLANT_BIAS)],
        });
        table.push(TableRow {
            prompt,
            implant: idx,
            forced_token: forced,
        });
    }
    ImplantSpec::new(name, rules, actions, Some(demote_env.to_string()))
        .map_err(|e: ImplantError| format!("{name}: {e}"))
}

// ── Mount rigs ───────────────────────────────────────────────────────────

/// Minted artifacts for both lanes + the authored assertion table (the
/// ONE source of truth for which implant owns a row and which token id
/// its action forces — never re-derived from the prompt text: the
/// digit's integer value is not the token id).
#[derive(Debug, Clone)]
pub struct ArmsRig {
    /// The addition artifact (slot 0).
    pub add: ImplantArtifact,
    /// The subtraction artifact (slot 1).
    pub sub: ImplantArtifact,
    /// The authored trigger table (addition rows then subtraction rows).
    pub table: Vec<TableRow>,
}

impl ArmsRig {
    /// Mint both artifacts at epoch 1 and carry the authored table.
    #[must_use]
    pub fn mint(lanes: &AuthoredLanes) -> Self {
        Self {
            add: ImplantArtifact::mint(&lanes.add, 1),
            sub: ImplantArtifact::mint(&lanes.sub, 1),
            table: lanes.table.clone(),
        }
    }

    /// The authored row for `prompt`.
    ///
    /// # Errors
    /// The prompt names no authored row (a cell/fixture mismatch — loud).
    pub fn row(&self, prompt: &str) -> Result<&TableRow, String> {
        self.table
            .iter()
            .find(|r| r.prompt == prompt)
            .ok_or_else(|| format!("rig: prompt {prompt:?} names no authored row"))
    }
}

/// The mount posture for one walk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MountPosture {
    /// No mounts (the base arm).
    None,
    /// Both implants mounted armed.
    Armed,
    /// Both mounts resolve demoted (the exact-literal `0` env set around
    /// the mount; the typed [`ImplantError::DemotedAtMount`] result is
    /// the expected outcome and is asserted).
    Demoted,
}

impl ArmsRig {
    /// A fresh two-slot runtime in `posture`.
    ///
    /// # Errors
    /// Any mount refusal other than the posture's expected outcome.
    pub fn new_runtime(&self, posture: MountPosture) -> Result<ImplantRuntime, String> {
        let mut rt = ImplantRuntime::new(2);
        if posture == MountPosture::None {
            return Ok(rt);
        }
        if posture == MountPosture::Demoted {
            // The demote posture resolves at MOUNT time — set the exact
            // literal, mount, then clear (the drift_guard seam exercised
            // end-to-end). SAFETY: env mutation is single-threaded here
            // by contract — the gates test serializes env users behind
            // ENV_LOCK, and the bin calls this from single-threaded
            // `main`; no other thread reads env concurrently.
            unsafe {
                std::env::set_var(ADD_DEMOTE_ENV, "0");
                std::env::set_var(SUB_DEMOTE_ENV, "0");
            }
        }
        let mounts = [
            (0usize, &self.add, ADD_DEMOTE_ENV),
            (1usize, &self.sub, SUB_DEMOTE_ENV),
        ];
        for (i, artifact, env) in mounts {
            let res = rt.mount(i, artifact);
            match posture {
                MountPosture::Armed => {
                    res.map_err(|e| format!("mount slot {i}: {e}"))?;
                }
                MountPosture::Demoted => match res {
                    Err(ImplantError::DemotedAtMount) => {
                        debug_assert!(rt.is_demoted(i), "demoted mount must mark the slot");
                    }
                    _ => {
                        // SAFETY: same single-threaded contract as above.
                        unsafe {
                            std::env::remove_var(env);
                            std::env::remove_var(ADD_DEMOTE_ENV);
                            std::env::remove_var(SUB_DEMOTE_ENV);
                        }
                        return Err(format!("mount slot {i}: demoted posture mounted armed"));
                    }
                },
                MountPosture::None => unreachable!("None handled above"),
            }
        }
        if posture == MountPosture::Demoted {
            // SAFETY: same single-threaded contract as above.
            unsafe {
                std::env::remove_var(ADD_DEMOTE_ENV);
                std::env::remove_var(SUB_DEMOTE_ENV);
            }
        }
        Ok(rt)
    }
}

// ── Cell results ─────────────────────────────────────────────────────────

/// One cell's verdict (typed, serializable).
#[derive(Debug, Clone, serde::Serialize)]
pub struct CellResult {
    /// The cell name (`G1_silent_identity`, `G1b_false_fire`,
    /// `G2_fired_exactness`, `G3_composition`).
    pub name: &'static str,
    /// The pass verdict.
    pub pass: bool,
    /// Named counters.
    pub counts: Vec<(&'static str, usize)>,
    /// Free-form notes (failures carry the diagnosis here).
    pub notes: Vec<String>,
}

impl CellResult {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            pass: true,
            counts: Vec::new(),
            notes: Vec::new(),
        }
    }

    fn count(mut self, k: &'static str, v: usize) -> Self {
        self.counts.push((k, v));
        self
    }

    fn note(mut self, s: String) -> Self {
        self.pass = false;
        self.notes.push(s);
        self
    }

    fn check(self, ok: bool, s: String) -> Self {
        if ok { self } else { self.note(s) }
    }
}

// ── G1 — silent identity ─────────────────────────────────────────────────

/// G1: armed-but-silent ≡ base, and demoted-on-a-firing-row ≡ base.
/// Silent rows = prose + look-alikes (never fire armed); the demote arm
/// walks the FIRST table row (which would fire when armed) under the
/// demoted mount. Byte comparator: per-step logit-bit digests + the
/// generated token ids.
///
/// # Errors
/// Encoding/mount failures (cell machinery misuse — never a measurement).
pub fn cell_g1_silent_identity<S: LogitsSource, C: FixtureCodec>(
    source: &mut S,
    codec: &C,
    rig: &ArmsRig,
    max_new: usize,
) -> Result<CellResult, String> {
    let mut cell = CellResult::new("G1_silent_identity");
    let mut silent_rows: Vec<String> = PROSE_PROMPTS.iter().map(|s| (*s).to_string()).collect();
    silent_rows.extend(LOOKALIKE_PROMPTS.iter().map(|s| (*s).to_string()));

    let bufs = &mut WalkBufs::new(source.vocab());
    // Base arm.
    let mut base_rows = Vec::with_capacity(silent_rows.len());
    let mut row_prompts = Vec::with_capacity(silent_rows.len());
    for row in &silent_rows {
        let toks = codec.encode(row)?;
        let mut rt = rig.new_runtime(MountPosture::None)?;
        base_rows.push((
            toks.len(),
            decode_walk(source, &toks, max_new, &mut rt, bufs),
        ));
        row_prompts.push(row.clone());
    }
    // Silent-mounted arm. The load-bearing identity: NO fire during the
    // prompt phase, prompt-phase digests identical, first emission
    // identical. Post-prompt divergence is DISCLOSED, not failed — a
    // mid-generation fire means the model generated genuinely tabled
    // text, which is correct trigger behavior, not a fixture false fire.
    let mut silent_rows_out = Vec::with_capacity(silent_rows.len());
    let mut silent_fires = 0usize;
    let mut gen_diverged = 0usize;
    for row in &silent_rows {
        let toks = codec.encode(row)?;
        let mut rt = rig.new_runtime(MountPosture::Armed)?;
        let out = decode_walk(source, &toks, max_new, &mut rt, bufs);
        silent_fires += out.fires.iter().filter(|f| f.step < toks.len()).count();
        silent_rows_out.push(out);
    }
    for (i, ((p, b), s)) in base_rows.iter().zip(silent_rows_out.iter()).enumerate() {
        let row = &row_prompts[i];
        let prompt_fires = s.fires.iter().filter(|f| f.step < *p).count();
        cell = cell.check(
            prompt_fires == 0,
            format!("G1: silent-mounted fired during the prompt phase on {row:?}"),
        );
        cell = cell.check(
            b.step_digests[..*p] == s.step_digests[..*p],
            format!("G1: silent-mounted prompt-phase digests diverge on {row:?}"),
        );
        let first_equal = b.tokens.first() == s.tokens.first();
        cell = cell.check(
            first_equal,
            format!("G1: silent-mounted first emission diverges on {row:?}"),
        );
        if s.step_digests != b.step_digests {
            gen_diverged += 1;
        }
    }
    // Demoted arm on the first table row (would fire when armed).
    let demote_row = codec.encode(&rig_first_row_prompt())?;
    let mut rt_base = rig.new_runtime(MountPosture::None)?;
    let base_demote = decode_walk(source, &demote_row, max_new, &mut rt_base, bufs);
    let mut rt_dem = rig.new_runtime(MountPosture::Demoted)?;
    let demoted = decode_walk(source, &demote_row, max_new, &mut rt_dem, bufs);
    cell = cell.check(
        base_demote.step_digests == demoted.step_digests,
        format!(
            "G1: demoted-mount digests diverge on {:?}",
            rig_first_row_prompt()
        ),
    );
    cell = cell.check(
        base_demote.tokens == demoted.tokens,
        format!(
            "G1: demoted-mount tokens diverge on {:?}",
            rig_first_row_prompt()
        ),
    );
    cell = cell.check(
        demoted.fires.is_empty(),
        "G1: demoted mount fired".to_string(),
    );
    cell = cell
        .count("silent_rows", silent_rows.len())
        .count("silent_fires", silent_fires)
        .count("gen_diverged_disclosed", gen_diverged)
        .count(
            "demote_row_digest_equal",
            usize::from(base_demote.step_digests == demoted.step_digests),
        );
    Ok(cell)
}

fn rig_first_row_prompt() -> String {
    // The first authored addition row — the demote arm's firing-row.
    let (a, b) = add_pairs()[0];
    format!("{a}+{b}=")
}

// ── G1b — false fire ─────────────────────────────────────────────────────

/// G1b: on every look-alike + prose row, every implant fires 0 times
/// (observe-level; the walk-level no-regression proof rides G2). The
/// runtime window advances across the whole row stream per row (a fresh
/// runtime per row — rows independent).
///
/// # Errors
/// Encoding failures.
pub fn cell_g1b_false_fire<C: FixtureCodec>(
    codec: &C,
    rig: &ArmsRig,
) -> Result<CellResult, String> {
    let mut cell = CellResult::new("G1b_false_fire");
    let mut rows: Vec<String> = LOOKALIKE_PROMPTS.iter().map(|s| (*s).to_string()).collect();
    rows.extend(PROSE_PROMPTS.iter().map(|s| (*s).to_string()));
    let mut total_fires = 0usize;
    let mut fired_rows = 0usize;
    let mut fire_buf = Vec::new();
    for row in &rows {
        let toks = codec.encode(row)?;
        let mut rt = rig.new_runtime(MountPosture::Armed)?;
        let mut row_fires = 0usize;
        for t in toks {
            fire_buf.clear();
            rt.observe_into(t as u32, &mut fire_buf);
            row_fires += fire_buf.len();
        }
        row_fires += rt.fires().iter().sum::<u64>() as usize;
        total_fires += row_fires;
        if row_fires > 0 {
            fired_rows += 1;
            cell = cell.note(format!("G1b: {row:?} fired {row_fires} time(s)"));
        }
    }
    cell = cell
        .count("lookalike_prose_rows", rows.len())
        .count("rows_fired", fired_rows)
        .count("total_fires", total_fires)
        .check(total_fires == 0, "G1b: false fires present".to_string());
    Ok(cell)
}

// ── G2 — fired exactness ─────────────────────────────────────────────────

/// G2: on every tabled row the armed walk fires at the row's own span
/// (the last prompt step, by the row's implant) and emits the forced
/// token EXACTLY at that decision point; the prompt-phase logits are
/// bit-identical to base up to (not including) the fired step; the fired
/// step's digest diverges (the write happened); every attributable fire
/// emits its own action's forced token (chained fires included — the
/// model may continue with further tabled equations, and an exact
/// trigger writes exactly there too); on every look-alike row the
/// prompt phase is fire-free and digest-identical, first emission
/// identical (the co-measured non-fired spans).
///
/// # Errors
/// Encoding/mount failures.
pub fn cell_g2_fired_exactness<S: LogitsSource, C: FixtureCodec>(
    source: &mut S,
    codec: &C,
    rig: &ArmsRig,
    max_new: usize,
) -> Result<CellResult, String> {
    let mut cell = CellResult::new("G2_fired_exactness");
    let bufs = &mut WalkBufs::new(source.vocab());

    // Tabled rows: armed vs base.
    let mut exact = 0usize;
    let mut chained_fires = 0usize;
    let mut fires_without_emission = 0usize;
    for row_prompt in g2_table_prompts() {
        let toks = codec.encode(&row_prompt)?;
        let p = toks.len();
        let mut rt_base = rig.new_runtime(MountPosture::None)?;
        let base = decode_walk(source, &toks, max_new, &mut rt_base, bufs);
        let mut rt_arm = rig.new_runtime(MountPosture::Armed)?;
        let armed = decode_walk(source, &toks, max_new, &mut rt_arm, bufs);
        let authored = rig.row(&row_prompt)?;
        let forced = authored.forced_token;
        let own = authored.implant;
        // The row's own fire, at the row's own span. Chained fires
        // (later steps) are counted + exact-checked, never a failure:
        // the model may continue with further tabled equations.
        let first_fire_ok = !armed.fires.is_empty()
            && armed.fires[0].step == p - 1
            && armed.fires[0].implant == own;
        cell = cell.check(
            first_fire_ok,
            format!(
                "G2: {row_prompt:?} fire shape wrong ({fires:?}, expected own at step {want})",
                fires = armed
                    .fires
                    .iter()
                    .map(|f| (f.step, f.implant))
                    .collect::<Vec<_>>(),
                want = p - 1
            ),
        );
        // Same-step double fires make attribution undefined — refuse
        // loud (never pool two folds into one emission verdict).
        cell = cell.check(
            !armed.fires.windows(2).any(|w| w[0].step == w[1].step),
            format!("G2: {row_prompt:?} same-step double fire — attribution undefined"),
        );
        // Exact forced emission at the targeted span's decision point.
        let exact_here = !armed.tokens.is_empty() && armed.tokens[0] == forced;
        cell = cell.check(
            exact_here,
            format!(
                "G2: {row_prompt:?} emitted {:?} not the forced digit (top2 {:?})",
                armed.tokens.first().map(|t| codec.decode_one(*t)),
                armed.fires.first().map(|f| f.top2)
            ),
        );
        // Prompt phase identical up to (not incl.) the fired step.
        cell = cell.check(
            armed.step_digests[..p - 1] == base.step_digests[..p - 1],
            format!("G2: {row_prompt:?} prompt-phase digests diverge before the fire"),
        );
        // The fired step diverged (the bias write happened).
        cell = cell.check(
            armed.step_digests[p - 1] != base.step_digests[p - 1],
            format!("G2: {row_prompt:?} fired-step digest did not move (no write?)"),
        );
        // Per-fire exactness: each fire's fold decides emission
        // tokens[step - (p-1)]; it must be that fire's own forced token.
        for f in &armed.fires {
            let k = f.step - (p - 1);
            let Some(&tok) = armed.tokens.get(k) else {
                fires_without_emission += 1;
                continue; // the final step's fold decides nothing emitted
            };
            let fire_ref = FireRef {
                implant: f.implant,
                action: f.action,
            };
            let fire_forced = rt_arm.action(&fire_ref).biases[0].0 as usize;
            cell = cell.check(
                tok == fire_forced,
                format!(
                    "G2: {row_prompt:?} chained fire at step {step} emitted {got:?} not its forced {want}",
                    step = f.step,
                    got = codec.decode_one(tok),
                    want = codec.decode_one(fire_forced)
                ),
            );
            if f.step != p - 1 {
                chained_fires += 1;
            }
        }
        if first_fire_ok && exact_here {
            exact += 1;
        }
    }
    // Look-alike rows: the non-fired spans. The prompt phase must be
    // fire-free and digest-identical, first emission identical;
    // post-prompt divergence is disclosed (model-generated tabled text
    // is the trigger's own domain, not the fixture's false-fire bar).
    let mut look_identical = 0usize;
    for row_prompt in LOOKALIKE_PROMPTS {
        let toks = codec.encode(row_prompt)?;
        let p = toks.len();
        let mut rt_base = rig.new_runtime(MountPosture::None)?;
        let base = decode_walk(source, &toks, max_new, &mut rt_base, bufs);
        let mut rt_arm = rig.new_runtime(MountPosture::Armed)?;
        let armed = decode_walk(source, &toks, max_new, &mut rt_arm, bufs);
        let prompt_fires = armed.fires.iter().filter(|f| f.step < p).count();
        let same = prompt_fires == 0
            && armed.step_digests[..p] == base.step_digests[..p]
            && armed.tokens.first() == base.tokens.first();
        cell = cell.check(
            same,
            format!("G2: look-alike {row_prompt:?} prompt-phase walk diverged from base"),
        );
        if armed.step_digests != base.step_digests {
            chained_fires += armed.fires.iter().filter(|f| f.step >= p).count();
        }
        if same {
            look_identical += 1;
        }
    }
    let n_table = g2_table_prompts().len();
    cell = cell
        .count("table_rows", n_table)
        .count("exact_forced", exact)
        .count("lookalike_rows", LOOKALIKE_PROMPTS.len())
        .count("lookalike_identical", look_identical)
        .count("chained_fires_disclosed", chained_fires)
        .count("fires_without_emission", fires_without_emission)
        .check(
            exact == n_table,
            format!("G2: exactness {exact}/{n_table} — below the exact bar"),
        )
        .check(
            look_identical == LOOKALIKE_PROMPTS.len(),
            "G2: look-alike no-regression incomplete".to_string(),
        );
    Ok(cell)
}

fn g2_table_prompts() -> Vec<String> {
    let mut v: Vec<String> = add_pairs()
        .iter()
        .map(|(a, b)| format!("{a}+{b}="))
        .collect();
    v.extend(sub_pairs().iter().map(|(a, b)| format!("{a}-{b}=")));
    v
}

// ── G3 — composition ─────────────────────────────────────────────────────

/// G3: two implants co-mounted — each fires alone on its own family at
/// its own span and the OTHER implant is silent through that row's
/// prompt phase; the co-mounted walk matches the single-mounted walk
/// through the prompt phase AND the targeted emission (no interference
/// at the composition-relevant surface); every attributable co fire
/// emits its own action's forced token; combined exactness on all
/// tabled rows ≥ each single's ceiling; prose + look-alike rows stay
/// fire-free and digest-identical through their prompt phase at the co
/// posture. Post-emission divergence (chained fires on model-generated
/// text — a cross-family equation the model emitted) is DISCLOSED, not
/// failed: the stream is shared, and each implant correctly doing its
/// own job on genuinely tabled text is not composition interference.
///
/// # Errors
/// Encoding/mount failures.
pub fn cell_g3_composition<S: LogitsSource, C: FixtureCodec>(
    source: &mut S,
    codec: &C,
    rig: &ArmsRig,
    max_new: usize,
) -> Result<CellResult, String> {
    let mut cell = CellResult::new("G3_composition");
    let bufs = &mut WalkBufs::new(source.vocab());

    // Single-mount rigs: addition only (slot 0), subtraction only (slot 1).
    let solo_add = SoloRig {
        artifact: &rig.add,
        slot: 0,
    };
    let solo_sub = SoloRig {
        artifact: &rig.sub,
        slot: 1,
    };

    let mut solo_fire_ok = 0usize;
    let mut no_interference = 0usize;
    let mut combined_exact = 0usize;
    let mut post_emission_diverged = 0usize;
    let n_add = add_pairs().len();
    let n_sub = sub_pairs().len();
    for row_prompt in g2_table_prompts() {
        let toks = codec.encode(&row_prompt)?;
        let p = toks.len();
        let authored = rig.row(&row_prompt)?;
        let own = authored.implant;
        let forced = authored.forced_token;
        // Co-mounted walk.
        let mut rt_co = rig.new_runtime(MountPosture::Armed)?;
        let co = decode_walk(source, &toks, max_new, &mut rt_co, bufs);
        // Single-mounted walk (own implant only).
        let solo_rig = if own == 0 { &solo_add } else { &solo_sub };
        let mut rt_solo = solo_rig.runtime()?;
        let solo = decode_walk(source, &toks, max_new, &mut rt_solo, bufs);
        // First-fire shape, both postures: own implant, own span.
        let solo_fires =
            !solo.fires.is_empty() && solo.fires[0].step == p - 1 && solo.fires[0].implant == own;
        let co_fires =
            !co.fires.is_empty() && co.fires[0].step == p - 1 && co.fires[0].implant == own;
        cell = cell.check(
            solo_fires && co_fires,
            format!(
                "G3: {row_prompt:?} fire shape (solo {sf:?}, co {cf:?})",
                sf = solo
                    .fires
                    .iter()
                    .map(|f| (f.step, f.implant))
                    .collect::<Vec<_>>(),
                cf = co
                    .fires
                    .iter()
                    .map(|f| (f.step, f.implant))
                    .collect::<Vec<_>>(),
            ),
        );
        // The OTHER implant never fires during this row's prompt phase.
        let other = 1 - own;
        cell = cell.check(
            !co.fires.iter().any(|f| f.step < p && f.implant == other),
            format!(
                "G3: {row_prompt:?} — the other implant (slot {other}) fired in this row's prompt phase"
            ),
        );
        // Same-step double fires: attribution undefined — refuse loud.
        cell = cell.check(
            !co.fires.windows(2).any(|w| w[0].step == w[1].step),
            format!("G3: {row_prompt:?} same-step double fire — attribution undefined"),
        );
        // No interference at the composition-relevant surface: the
        // prompt phase digests AND the targeted emission match solo.
        let same = co.step_digests[..p] == solo.step_digests[..p]
            && co.tokens.first() == solo.tokens.first();
        cell = cell.check(
            same,
            format!("G3: {row_prompt:?} co-mounted walk interfered with the solo walk"),
        );
        // Combined exactness (the targeted emission).
        let exact_here = !co.tokens.is_empty() && co.tokens[0] == forced;
        cell = cell.check(
            exact_here,
            format!("G3: {row_prompt:?} co-mounted emission not exact"),
        );
        // Per-fire exactness on the co walk (chained fires included).
        for f in &co.fires {
            let k = f.step - (p - 1);
            let Some(&tok) = co.tokens.get(k) else {
                continue; // the final step's fold decides nothing emitted
            };
            let fire_ref = FireRef {
                implant: f.implant,
                action: f.action,
            };
            let fire_forced = rt_co.action(&fire_ref).biases[0].0 as usize;
            cell = cell.check(
                tok == fire_forced,
                format!(
                    "G3: {row_prompt:?} chained fire at step {step} emitted {got:?} not its forced {want}",
                    step = f.step,
                    got = codec.decode_one(tok),
                    want = codec.decode_one(fire_forced)
                ),
            );
        }
        if co.step_digests != solo.step_digests {
            post_emission_diverged += 1;
        }
        if solo_fires && co_fires {
            solo_fire_ok += 1;
        }
        if same {
            no_interference += 1;
        }
        if exact_here {
            combined_exact += 1;
        }
    }
    // Silent rows at the co posture: fire-free + digest-identical
    // through the prompt phase, first emission identical.
    let mut silent_ok = 0usize;
    let silent_prompts: Vec<&str> = PROSE_PROMPTS
        .iter()
        .chain(LOOKALIKE_PROMPTS.iter())
        .copied()
        .collect();
    for row_prompt in silent_prompts {
        let toks = codec.encode(row_prompt)?;
        let p = toks.len();
        let mut rt_base = rig.new_runtime(MountPosture::None)?;
        let base = decode_walk(source, &toks, max_new, &mut rt_base, bufs);
        let mut rt_co = rig.new_runtime(MountPosture::Armed)?;
        let co = decode_walk(source, &toks, max_new, &mut rt_co, bufs);
        let prompt_fires = co.fires.iter().filter(|f| f.step < p).count();
        let ok = prompt_fires == 0
            && co.step_digests[..p] == base.step_digests[..p]
            && co.tokens.first() == base.tokens.first();
        cell = cell.check(
            ok,
            format!("G3: silent row {row_prompt:?} not silent/digest-identical at co"),
        );
        if ok {
            silent_ok += 1;
        }
    }
    let n_table = g2_table_prompts().len();
    let best_single = n_add.max(n_sub);
    cell = cell
        .count("table_rows", n_table)
        .count("solo_co_fire_ok", solo_fire_ok)
        .count("no_interference", no_interference)
        .count("combined_exact", combined_exact)
        .count("silent_rows_ok", silent_ok)
        .count("best_single_exact_ceiling", best_single)
        .count(
            "post_emission_diverged_disclosed",
            post_emission_diverged,
        )
        .check(
            combined_exact == n_table && combined_exact >= best_single,
            format!(
                "G3: combined exactness {combined_exact}/{n_table} (best single ceiling {best_single})"
            ),
        );
    Ok(cell)
}

/// A single-implant rig (one slot, one artifact — the G3 singles).
struct SoloRig<'a> {
    artifact: &'a ImplantArtifact,
    slot: usize,
}

impl SoloRig<'_> {
    fn runtime(&self) -> Result<ImplantRuntime, String> {
        let mut rt = ImplantRuntime::new(2);
        rt.mount(self.slot, self.artifact)
            .map_err(|e| format!("solo mount slot {}: {e}", self.slot))?;
        Ok(rt)
    }
}
