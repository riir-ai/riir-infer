//! The static readout + verbalizability pre-screen for the neuron census
//! (riir-infer Issue 1250 T1; katgpt-rs Research 693 — the Anthropic
//! Fellows pair distill, alignment.anthropic.com/2026/neuron-basis-theory).
//!
//! **What v0 (T1) is:** a weights-only logit-lens readout map of every
//! MLP neuron's down-proj write column, **sign-split** (`+w` and `−w`
//! are separate units — the paper's type-2 finding makes one neuron two
//! concept units), for every layer, over one open GGUF. Per unit: the
//! top-`TOP_K` tokens by lens logit (ids only; strings are
//! tokenizer-derivable), the top-1 sharpness and the top-1−top-2
//! margin. Per layer band (early/middle/late thirds): pinned rollups of
//! those two statistics — the T4/T5 discrimination input ("the screen
//! fails" vs "the logit lens is too weak at middle layers").
//!
//! **What v0 deliberately is not:** no J-share/verbalizability number —
//! that is T2 (NNLS against lens directions). The per-unit `bucket`
//! field exists in the artifact schema and ships `UNASSIGNED` in v1;
//! T2 fills it and the per-band bucket rollup. Nothing here assumes a
//! dense-SwiGLU 8–32B transfer (the paper's own population) — T4
//! measures that on our ternary/GDN checkpoints, never this module.
//!
//! **Lens definition (pinned):** `logits_t = ⟨U_t, rmsnorm_Γ(w)⟩`
//! where `w` is the neuron's `n_embd`-length down column (GGUF row
//! semantics: `ffn_down` is `[n_ff × n_embd]`, one row per OUTPUT
//! element; neuron n's column is the n-th element of every row — the
//! `weight_audit` reading), `Γ` is `output_norm.weight`, and
//! `rmsnorm` is the final-norm shape `x·Γ/sqrt(mean(x²)+eps)` folded
//! sequentially in f64 (the Issue-940 law: this artifact's numbers are
//! digested, so the fold order is pinned — NOT the serving SIMD norm,
//! which is platform-strided by design).
//!
//! **The sign trick (exact, not approximate):** `logits(−w) = −logits(w)`
//! bit-exactly in IEEE arithmetic (negation commutes with the ordered
//! f64 dot), so one ascending scan per neuron maintains BOTH the top-K
//! (for `+w`) and the bottom-K (for `−w`, values stored negated) — half
//! the flops of computing both signs separately, byte-identical results.
//! Ties: value-first, then ascending token id (first-index law); the
//! ascending scan makes later equal tokens never win.
//!
//! Posture: opt-in `readout_screen`, offline by construction — weights
//! only, no forward pass, no serving path, zero new deps (rayon +
//! blake3 already in-tree). Deterministic: same file → bit-identical
//! artifact + report (the parallelism partitions whole neurons per
//! rayon task; every unit's ranking is computed by exactly one task in
//! ascending token order). The artifact is canonical binary
//! (`RDMAP1\x01` + packed rows) + BLAKE3 digest, bound to the source
//! file's whole-file BLAKE3.

use crate::gguf_loader::GgufFile;
use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use std::fmt::Write as _;

/// Tokens retained per unit per sign (pinned — 1250 T1 "top-k"; T2/T4
/// read the artifact, never re-derive).
pub const TOP_K: usize = 10;

/// Population floors (a real checkpoint is far above these; the fixture
/// is the only legal tenant of the minimum band).
const MIN_LAYERS: usize = 1;
const MIN_EMBD: usize = 8;
const MIN_FF: usize = 8;
const MIN_VOCAB: usize = 16;

/// Artifact magic + format version (v1: the T1 schema — bucket ships
/// UNASSIGNED until T2 fills it).
pub const ARTIFACT_MAGIC: &[u8; 7] = b"RDMAP1\x01";
pub const ARTIFACT_VERSION: u32 = 1;

/// Header size: magic 7 + version 4 + vocab/n_embd/n_ff/n_layers/top_k
/// 4×5 + rms_eps 4 + unemb_source 1 + reserved 3 + file digest 32 = 71.
pub const HEADER_BYTES: usize = 71;

/// Per-unit J-share bucket (T2 fills; v1 ships [`Bucket::Unassigned`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Bucket {
    /// v1 default — T2's verbalizability pass assigns High/Middle/Low.
    Unassigned = 0,
    High = 1,
    Middle = 2,
    Low = 3,
}

/// Sign of the read unit (the type-2 split: one neuron, two units).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Sign {
    Plus = 0,
    Minus = 1,
}

/// Layer band: deterministic thirds
/// (`((layer * 3) / n_layers).min(2)` — 24 layers → 8/8/8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Band {
    Early = 0,
    Middle = 1,
    Late = 2,
}

/// Pinned band assignment (`min(2)` guards `layer = n_layers` never
/// occurring but keeps the formula total).
#[inline]
pub fn band_of(layer: usize, n_layers: usize) -> Band {
    match ((layer * 3) / n_layers.max(1)).min(2) {
        0 => Band::Early,
        1 => Band::Middle,
        _ => Band::Late,
    }
}

// ── Deterministic folds (the katgpt-rs Issue-940 law) ────────────────

/// Sequential ascending-index f64 dot of two f32 slices, promoted per
/// element. Plain `+=` — never a platform SIMD kernel: this artifact's
/// numbers are digested and must be bit-reproducible. Pub: the fixture
/// gates fold with the SAME function the screen uses (one fold law).
#[inline]
pub fn ordered_dot_f64(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f64;
    for i in 0..a.len() {
        acc += a[i] as f64 * b[i] as f64;
    }
    acc
}

/// Sequential ascending-index f64 sum of squares.
#[inline]
fn ordered_sumsq_f64(row: &[f32]) -> f64 {
    let mut acc = 0.0f64;
    for &v in row {
        acc += v as f64 * v as f64;
    }
    acc
}

/// The lens norm, final-norm shape, pinned sequential f64:
/// `out[i] = Γ[i] · w[i] / sqrt(mean(w²) + eps)` (f32 in, f64 fold,
/// f32 out). Deliberately NOT [`katgpt_core::types::
/// rmsnorm_with_gamma_eps`] — that one is platform-strided for serving;
/// this artifact needs the pinned fold. Same mathematical shape.
pub fn lens_norm(w: &[f32], gamma: &[f32], eps: f64) -> Vec<f32> {
    debug_assert_eq!(w.len(), gamma.len());
    let n = w.len();
    let sum_sq = ordered_sumsq_f64(w);
    let inv_rms = 1.0 / ((sum_sq / n as f64) + eps).sqrt();
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        out[i] = (w[i] as f64 * gamma[i] as f64 * inv_rms) as f32;
    }
    out
}

// ── The tensor source abstraction ────────────────────────────────────

/// Every MLP layer's write columns + the readout head, as the screen
/// consumes it. The GGUF source ([`GgufReadoutSource`]) and the test
/// fixture implement the same trait, so fixture arms and real runs run
/// ONE code path.
///
/// Row semantics (the GGUF layout): `down_out_row(layer, j)` is output
/// element `j`'s `n_ff`-length row; neuron `n`'s write column is the
/// `n`-th element of every row of that layer. `unemb_rows()` is the
/// readout table row-major `[vocab × n_embd]` (dequantized ONCE — the
/// screen streams it per unit batch).
pub trait ReadoutTensors: Sync {
    fn n_layers(&self) -> usize;
    fn n_ff(&self) -> usize;
    fn n_embd(&self) -> usize;
    fn vocab(&self) -> usize;
    fn gamma(&self) -> &[f32];
    fn rms_eps(&self) -> f64;
    fn down_out_row(&self, layer: usize, j: usize) -> Result<Vec<f32>>;
    /// The full unembedding, row-major `[vocab × n_embd]` f32.
    fn unemb_rows(&self) -> Result<Vec<f32>>;
    /// Source label for the report (`gguf:output.weight` /
    /// `gguf:token_embd.weight (tied)` / `fixture`).
    fn label(&self) -> String;
    /// Whole-source BLAKE3 (`None` when the source is not a file — the
    /// fixture). Binds the artifact to the exact bytes read.
    fn file_digest(&self) -> Option<[u8; 32]> {
        None
    }
}

/// A GGUF-backed source: EVERY `blk.N` carrying the ffn trio + the
/// untied (`output.weight`) or tied (`token_embd.weight`) unembedding +
/// `output_norm.weight` as Γ (the `weight_audit` open rules, generalized
/// to all layers). The whole-file BLAKE3 is hashed streaming at open
/// (the artifact's provenance binding).
pub struct GgufReadoutSource<'a> {
    gguf: &'a GgufFile,
    n_layers: usize,
    n_ff: usize,
    n_embd: usize,
    vocab: usize,
    down: Vec<String>,
    unemb: String,
    unemb_label: String,
    gamma: Vec<f32>,
    rms_eps: f64,
    file_digest: Option<[u8; 32]>,
}

impl<'a> GgufReadoutSource<'a> {
    /// Open the screen source over every ffn-carrying block (ascending
    /// layer index; a layer missing any ffn tensor refuses loud — a
    /// partial sweep would silently skew the per-band rollups).
    /// `path` feeds the whole-file BLAKE3 (streaming, <1 s at GB-class
    /// files; the artifact's provenance binding).
    pub fn open(gguf: &'a GgufFile, path: &std::path::Path, rms_eps: f64) -> Result<Self> {
        // Ascending layer census over `blk.N.ffn_down.weight`.
        let mut layer_ids: Vec<usize> = Vec::new();
        for info in &gguf.tensor_infos {
            if let Some(rest) = info.name.strip_prefix("blk.")
                && let Some((idx, suffix)) = rest.split_once('.')
                && suffix == "ffn_down.weight"
                && let Ok(i) = idx.parse::<usize>()
            {
                layer_ids.push(i);
            }
        }
        if layer_ids.is_empty() {
            bail!("no blk.N.ffn_down.weight tensor in this GGUF");
        }
        layer_ids.sort_unstable();
        layer_ids.dedup();
        // Contiguity check: a hole means a missing layer would land in
        // the wrong band.
        let n_layers = layer_ids.len();
        for (pos, &id) in layer_ids.iter().enumerate() {
            if id != pos {
                bail!(
                    "ffn layers are not contiguous (blk.{id} at position {pos}) — \
                     refusing the partial sweep"
                );
            }
        }
        let down: Vec<String> = layer_ids
            .iter()
            .map(|i| format!("blk.{i}.ffn_down.weight"))
            .collect();
        let gate_probe = format!("blk.{}.ffn_gate.weight", layer_ids[0]);
        if gguf.tensor_info(&gate_probe).is_none() {
            bail!("layer {}: tensor '{gate_probe}' not found", layer_ids[0]);
        }
        let down_shape = gguf.tensor_info(&down[0]).unwrap().shape.clone();
        if down_shape.len() != 2 {
            bail!("ffn_down must be 2-D (shape {down_shape:?})");
        }
        // GGUF order: ne0 = row length = n_ff, ne1 = rows = n_embd (the
        // weight_audit reading — one row per OUTPUT element).
        let n_ff = down_shape[0];
        let n_embd = down_shape[1];
        // Unembedding: untied output.weight preferred, else the tied table.
        let (unemb, unemb_label) = if gguf.tensor_info("output.weight").is_some() {
            ("output.weight".to_string(), "output.weight".to_string())
        } else if gguf.tensor_info("token_embd.weight").is_some() {
            (
                "token_embd.weight".to_string(),
                "token_embd.weight (tied)".to_string(),
            )
        } else {
            bail!("no output.weight and no token_embd.weight — no unembedding head");
        };
        let ue_shape = gguf.tensor_info(&unemb).unwrap().shape.clone();
        if ue_shape.len() != 2 || ue_shape[0] != n_embd {
            bail!("unembedding {unemb} shape {ue_shape:?} disagrees with n_embd {n_embd}");
        }
        let vocab = ue_shape[1];
        // Γ = output_norm.weight (row-0 dequant reads the F32 vector).
        let gamma = gguf
            .dequant_tensor_row("output_norm.weight", 0, n_embd)
            .with_context(|| "output_norm.weight read failed")?;
        if gamma.len() != n_embd {
            bail!(
                "output_norm.weight length {} != n_embd {n_embd}",
                gamma.len()
            );
        }
        Ok(Self {
            gguf,
            n_layers,
            n_ff,
            n_embd,
            vocab,
            down,
            unemb,
            unemb_label,
            gamma,
            rms_eps,
            file_digest: file_digest_blake3(path),
        })
    }
}

/// Whole-file BLAKE3, streamed (a GB-class checkpoint hashes in about
/// a second; the artifact binds to these bytes). `pub(crate)`: shared
/// with the Issue-1251 trigger-vote source (same binding convention).
pub(crate) fn file_digest_blake3(path: &std::path::Path) -> Option<[u8; 32]> {
    use blake3::Hasher;
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let mut h = Hasher::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).ok()?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Some(*h.finalize().as_bytes())
}

impl ReadoutTensors for GgufReadoutSource<'_> {
    fn n_layers(&self) -> usize {
        self.n_layers
    }
    fn n_ff(&self) -> usize {
        self.n_ff
    }
    fn n_embd(&self) -> usize {
        self.n_embd
    }
    fn vocab(&self) -> usize {
        self.vocab
    }
    fn gamma(&self) -> &[f32] {
        &self.gamma
    }
    fn rms_eps(&self) -> f64 {
        self.rms_eps
    }
    fn down_out_row(&self, layer: usize, j: usize) -> Result<Vec<f32>> {
        self.gguf
            .dequant_tensor_row(&self.down[layer], j, self.n_ff)
    }
    fn unemb_rows(&self) -> Result<Vec<f32>> {
        self.gguf
            .dequant_f16_to_f32(&self.unemb)
            .with_context(|| format!("dequant {}", self.unemb))
    }
    fn label(&self) -> String {
        format!("gguf:{}", self.unemb_label)
    }
    fn file_digest(&self) -> Option<[u8; 32]> {
        self.file_digest
    }
}

// ── The screen ───────────────────────────────────────────────────────

/// One unit's readout row (a neuron at one sign).
#[derive(Debug, Clone)]
pub struct UnitRow {
    pub layer: usize,
    pub col: usize,
    pub sign: Sign,
    pub band: Band,
    pub bucket: Bucket,
    /// Top-`TOP_K` tokens, best first (value desc, token-asc ties).
    pub tokens: Vec<u32>,
    /// The matching lens logits, f32 bits stored canonically.
    pub logits: Vec<f32>,
    /// top1 − top2 (f64 subtract of the f32 values; `None` when fewer
    /// than 2 tokens — impossible above `MIN_VOCAB`, kept total).
    pub margin: Option<f32>,
}

impl UnitRow {
    fn from_topk(layer: usize, col: usize, sign: Sign, band: Band, mut tk: SideK) -> Self {
        let margin = if tk.vals.len() >= 2 {
            Some(tk.vals[0] - tk.vals[1])
        } else {
            None
        };
        if sign == Sign::Minus {
            // The bottom-K of +w, stored negated (the bit-exact sign
            // trick). Best-first for −w = ascending +w value.
            for v in &mut tk.vals {
                *v = -*v;
            }
        }
        Self {
            layer,
            col,
            sign,
            band,
            bucket: Bucket::Unassigned,
            tokens: tk.tokens,
            logits: tk.vals,
            margin,
        }
    }
}

/// A fixed-capacity running selection (value desc, token-asc ties) for
/// the top side; its mirror (value asc, token-asc ties) for the bottom
/// side. Ascending-token scans never see a later token win a tie, so
/// the running state IS the final selection.
///
/// `pub(crate)`: the Issue-1251 trigger-vote screen folds per-neuron
/// top-15 selections with the SAME selection law — one bounded-select
/// mechanism for the whole readout lane (no parallel re-derivation).
/// The bottom-side (`largest = false`) mirror stays 1250's sign-trick
/// lane; the vote lane only ever runs the largest side over `|Δ|`
/// scores (already nonnegative).
#[derive(Clone)]
pub(crate) struct SideK {
    tokens: Vec<u32>,
    vals: Vec<f32>,
    cap: usize,
    /// true = keep the LARGEST values (top side); false = keep the
    /// SMALLEST (bottom side).
    largest: bool,
}

impl SideK {
    pub(crate) fn new(cap: usize, largest: bool) -> Self {
        Self {
            tokens: Vec::with_capacity(cap),
            vals: Vec::with_capacity(cap),
            cap,
            largest,
        }
    }

    /// No token retained (every candidate scored ≤ the abstain floor;
    /// a `VOTES_PER_NEURON` > vocab source simply retains fewer — the
    /// vote artifact pads with `u32::MAX`).
    pub(crate) fn is_empty(&self) -> bool {
        self.vals.is_empty()
    }

    /// Consume the selection: `(tokens, vals)`, both best-first.
    pub(crate) fn into_parts(self) -> (Vec<u32>, Vec<f32>) {
        (self.tokens, self.vals)
    }

    /// Does candidate `(val, tok)` beat the current worst slot?
    #[inline]
    fn beats(&self, val: f32, tok: u32) -> bool {
        if self.vals.len() < self.cap {
            return true;
        }
        let wv = self.vals[self.cap - 1];
        let wt = self.tokens[self.cap - 1];
        if self.largest {
            val > wv || (val == wv && tok < wt)
        } else {
            val < wv || (val == wv && tok < wt)
        }
    }

    #[inline]
    pub(crate) fn push(&mut self, val: f32, tok: u32) {
        if !self.beats(val, tok) {
            return;
        }
        // Insertion position: the `largest` side keeps vals DESCENDING,
        // so the comparator is `val` vs `v` (reverse); the `smallest`
        // side keeps vals ASCENDING, comparator `v` vs `val`. Ties (the
        // comparator reads Equal) land BEFORE the equal run — but the
        // ascending token scan already guarantees a later equal token
        // never reaches `beats`, so the tie arm is dead in practice and
        // kept total only.
        let mut pos = if self.largest {
            self.vals
                .binary_search_by(|&v| val.partial_cmp(&v).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or_else(|p| p)
        } else {
            self.vals
                .binary_search_by(|&v| v.partial_cmp(&val).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or_else(|p| p)
        };
        // Equal values: the newcomer is a LATER token, so it ranks
        // AFTER the existing equal run (the first-index law). Without
        // this walk the Ok(pos) arm of binary_search would insert the
        // later token BEFORE the earlier equal one.
        while pos < self.vals.len() && self.vals[pos] == val {
            pos += 1;
        }
        if self.vals.len() < self.cap {
            self.vals.insert(pos, val);
            self.tokens.insert(pos, tok);
        } else {
            // pos ≤ cap−1 always (the candidate beat the worst slot).
            self.vals.insert(pos, val);
            self.tokens.insert(pos, tok);
            self.vals.pop();
            self.tokens.pop();
        }
    }
}

/// One neuron's both-sign selection state (one ascending vocab scan).
#[derive(Clone)]
struct NeuronScan {
    top: SideK,
    bottom: SideK,
}

/// One layer's screen result (both signs, col-major).
#[derive(Debug, Clone)]
pub struct LayerScreen {
    pub layer: usize,
    pub band: Band,
    /// `2 × n_ff` rows: neuron 0 (+), neuron 0 (−), neuron 1 (+), …
    pub units: Vec<UnitRow>,
}

/// Per-band pinned rollup (median = lower-middle element, MAD = same
/// rule over |x − median| — never an interpolated percentile; the
/// percentile-index law pins these to order statistics).
#[derive(Debug, Clone)]
pub struct BandRollup {
    pub band: Band,
    pub n_units: usize,
    /// Sharpness = top1 lens logit. mean (f64 fold), median, MAD, max.
    pub sharp_mean: f64,
    pub sharp_median: f32,
    pub sharp_mad: f32,
    pub sharp_max: f32,
    /// Margin = top1 − top2. mean, median, MAD, max.
    pub margin_mean: f64,
    pub margin_median: f32,
    pub margin_mad: f32,
    pub margin_max: f32,
    /// Units whose sharpness ≥ the checkpoint-wide sharpness median
    /// (the lens-strength proxy per band: T4/T5 distinguish "the screen
    /// fails" from "the lens is weak in this band" with this count).
    pub readable_units: usize,
}

/// The full screen result.
#[derive(Debug, Clone)]
pub struct ReadoutScreen {
    pub n_layers: usize,
    pub n_ff: usize,
    pub n_embd: usize,
    pub vocab: usize,
    pub source_label: String,
    pub top_k: usize,
    pub rms_eps: f64,
    pub layers: Vec<LayerScreen>,
    pub rollups: Vec<BandRollup>,
    /// The checkpoint-wide sharpness median (the readable-units bar).
    pub sharp_median_all: f32,
    pub file_digest: Option<[u8; 32]>,
}

/// Run the screen over the whole source: every layer × every neuron ×
/// both signs. Parallel over neurons within a layer (each neuron's
/// ranking is one task's ascending scan — deterministic by
/// construction); layers sequential (each materializes its columns,
/// 14 MB-class, then drops them).
pub fn screen(src: &dyn ReadoutTensors) -> Result<ReadoutScreen> {
    let n_layers = src.n_layers();
    let n_ff = src.n_ff();
    let d = src.n_embd();
    let v = src.vocab();
    if n_layers < MIN_LAYERS || d < MIN_EMBD || n_ff < MIN_FF || v < MIN_VOCAB {
        bail!(
            "degenerate source: layers={n_layers} n_embd={d} n_ff={n_ff} vocab={v} \
             (minimums {MIN_LAYERS}/{MIN_EMBD}/{MIN_FF}/{MIN_VOCAB})"
        );
    }
    if src.gamma().len() != d {
        bail!("gamma length {} != n_embd {d}", src.gamma().len());
    }
    if TOP_K >= v {
        bail!("TOP_K {TOP_K} >= vocab {v}");
    }
    // The unembedding, dequantized once, read-only across threads.
    let unemb = src.unemb_rows()?;
    if unemb.len() != v * d {
        bail!(
            "unembedding length {} != vocab {v} × n_embd {d}",
            unemb.len()
        );
    }
    for (i, &x) in unemb.iter().enumerate() {
        if !x.is_finite() {
            bail!("non-finite unembedding weight at [{i}] — refusing the screen");
        }
    }
    let gamma = src.gamma().to_vec();
    let eps = src.rms_eps();

    // Sequential layer pass; parallel neuron pass inside.
    let mut layers = Vec::with_capacity(n_layers);
    for layer in 0..n_layers {
        // Materialize this layer's write columns: col[n][t'] with
        // col[n][r] = down_out_row(r)[n].
        let mut cols = vec![0.0f32; n_ff * d];
        for r in 0..d {
            let row = src.down_out_row(layer, r)?;
            if row.len() != n_ff {
                bail!(
                    "layer {layer} ffn_down row {r} length {} != n_ff {n_ff}",
                    row.len()
                );
            }
            for (n, &val) in row.iter().enumerate() {
                if !val.is_finite() {
                    bail!("non-finite ffn_down weight at layer {layer} row {r} col {n}");
                }
                cols[n * d + r] = val;
            }
        }
        // Parallel over neurons; every unit's scan is one task.
        let scans: Vec<NeuronScan> = cols
            .par_chunks(d)
            .map(|col| {
                let normed = lens_norm(col, &gamma, eps);
                let mut scan = NeuronScan {
                    top: SideK::new(TOP_K, true),
                    bottom: SideK::new(TOP_K, false),
                };
                for t in 0..v {
                    let row = &unemb[t * d..(t + 1) * d];
                    let val = ordered_dot_f64(row, &normed) as f32;
                    let tok = t as u32;
                    scan.top.push(val, tok);
                    scan.bottom.push(val, tok);
                }
                scan
            })
            .collect();
        let band = band_of(layer, n_layers);
        let mut units = Vec::with_capacity(n_ff * 2);
        for (n, scan) in scans.into_iter().enumerate() {
            units.push(UnitRow::from_topk(layer, n, Sign::Plus, band, scan.top));
            units.push(UnitRow::from_topk(layer, n, Sign::Minus, band, scan.bottom));
        }
        layers.push(LayerScreen { layer, band, units });
    }

    // Rollups: checkpoint-wide sharpness median first (the readable bar).
    let mut sharp_all: Vec<f32> = layers
        .iter()
        .flat_map(|l| l.units.iter().step_by(2).map(|u| u.logits[0]))
        .collect();
    sharp_all.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let sharp_median_all = sharp_all[(sharp_all.len() - 1) / 2];

    let mut rollups = Vec::with_capacity(3);
    for band in [Band::Early, Band::Middle, Band::Late] {
        let mut sharp: Vec<f32> = Vec::new();
        let mut margin: Vec<f32> = Vec::new();
        for l in &layers {
            if l.band != band {
                continue;
            }
            for u in l.units.iter().step_by(2) {
                sharp.push(u.logits[0]);
                if let Some(m) = u.margin {
                    margin.push(m);
                }
            }
        }
        if sharp.is_empty() {
            continue;
        }
        let (s_med, s_mad) = median_mad(&sharp);
        let (m_med, m_mad) = median_mad(&margin);
        let s_mean: f64 = sharp.iter().map(|&x| x as f64).sum::<f64>() / sharp.len() as f64;
        let m_mean: f64 = margin.iter().map(|&x| x as f64).sum::<f64>() / margin.len() as f64;
        let s_max = sharp.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let m_max = margin.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        rollups.push(BandRollup {
            band,
            n_units: sharp.len(),
            sharp_mean: s_mean,
            sharp_median: s_med,
            sharp_mad: s_mad,
            sharp_max: s_max,
            margin_mean: m_mean,
            margin_median: m_med,
            margin_mad: m_mad,
            margin_max: m_max,
            readable_units: sharp.iter().filter(|&&x| x >= sharp_median_all).count(),
        });
    }

    Ok(ReadoutScreen {
        n_layers,
        n_ff,
        n_embd: d,
        vocab: v,
        source_label: src.label(),
        top_k: TOP_K,
        rms_eps: eps,
        layers,
        rollups,
        sharp_median_all,
        file_digest: src.file_digest(),
    })
}

/// Pinned median (ascending sort, lower-middle element) + MAD (same
/// rule over |x − median|). MAD == 0 is legal here (a degenerate-but-real
/// population) and reported as-is — the screen is a report, not a gate.
fn median_mad(values: &[f32]) -> (f32, f32) {
    let mut s = values.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = s[(s.len() - 1) / 2];
    let mut dev: Vec<f32> = values.iter().map(|&v| (v - med).abs()).collect();
    dev.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = dev[(dev.len() - 1) / 2];
    (med, mad)
}

// ── The artifact ─────────────────────────────────────────────────────

/// Packed row: `[layer u8][col u32][sign u8][bucket u8][pad u16]` (9
/// bytes) + `TOP_K × [token u32][bits u32]` (8·TOP_K), little-endian,
/// f32 stored as bits (the canonical-bytes convention).
pub const ROW_BYTES: usize = 9 + 8 * TOP_K;

/// Serialize the artifact: header + rows in (layer asc, col asc,
/// Plus-before-Minus) order — the `layers` vec's own order.
pub fn build_artifact(s: &ReadoutScreen) -> Vec<u8> {
    let n_units: usize = s.layers.iter().map(|l| l.units.len()).sum();
    let mut out = Vec::with_capacity(HEADER_BYTES + n_units * ROW_BYTES);
    out.extend_from_slice(ARTIFACT_MAGIC);
    out.extend_from_slice(&ARTIFACT_VERSION.to_le_bytes());
    out.extend_from_slice(&(s.vocab as u32).to_le_bytes());
    out.extend_from_slice(&(s.n_embd as u32).to_le_bytes());
    out.extend_from_slice(&(s.n_ff as u32).to_le_bytes());
    out.extend_from_slice(&(s.n_layers as u32).to_le_bytes());
    out.extend_from_slice(&(s.top_k as u32).to_le_bytes());
    out.extend_from_slice(&(s.rms_eps as f32).to_bits().to_le_bytes());
    out.push(match s.source_label.contains("tied") {
        true => 1,
        false => 0,
    });
    out.extend_from_slice(&[0u8; 3]); // reserved
    let mut fd = [0u8; 32];
    if let Some(d) = s.file_digest {
        fd = d;
    }
    out.extend_from_slice(&fd);
    for l in &s.layers {
        for u in &l.units {
            out.push(u.layer as u8);
            out.extend_from_slice(&(u.col as u32).to_le_bytes());
            out.push(u.sign as u8);
            out.push(u.bucket as u8);
            out.extend_from_slice(&[0u8; 2]); // pad to 8
            for i in 0..TOP_K {
                let tok = u.tokens.get(i).copied().unwrap_or(u32::MAX);
                let bits = u.logits.get(i).map(|f| f.to_bits()).unwrap_or(0);
                out.extend_from_slice(&tok.to_le_bytes());
                out.extend_from_slice(&bits.to_le_bytes());
            }
        }
    }
    out
}

/// BLAKE3 digest of the artifact bytes.
pub fn artifact_digest(bytes: &[u8]) -> [u8; 32] {
    *blake3::hash(bytes).as_bytes()
}

/// Parse the artifact header (the bin's `--verify` posture; a truncated
/// or foreign file refuses loud). Returns `(version, vocab, n_embd,
/// n_ff, units, top_k)` as usize.
pub fn parse_header(bytes: &[u8]) -> Result<(u32, usize, usize, usize, usize, usize)> {
    if bytes.len() < HEADER_BYTES || &bytes[..7] != ARTIFACT_MAGIC {
        bail!("not an RDMAP1 artifact (magic/length mismatch)");
    }
    let rd_u32 = |off: usize| -> u32 {
        u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
    };
    let version = rd_u32(7);
    if version != ARTIFACT_VERSION {
        bail!("artifact version {version} != {ARTIFACT_VERSION}");
    }
    let vocab = rd_u32(11) as usize;
    let n_embd = rd_u32(15) as usize;
    let n_ff = rd_u32(19) as usize;
    let n_layers = rd_u32(23) as usize;
    let top_k = rd_u32(27) as usize;
    let n_units = bytes.len() - HEADER_BYTES;
    if !n_units.is_multiple_of(ROW_BYTES) {
        bail!("artifact body {} is not a multiple of {ROW_BYTES}", n_units);
    }
    let units = n_layers
        .checked_mul(n_ff)
        .and_then(|p| p.checked_mul(2))
        .context("unit count overflow")?;
    Ok((version, vocab, n_embd, n_ff, units, top_k))
}

// ── The report ───────────────────────────────────────────────────────

/// Render the deterministic text report (floats print with Rust's
/// shortest round-trip `{:?}`; the text is a G4 artifact — two runs are
/// byte-identical). `token_strings` maps token id → printable piece for
/// the top-units table (`None` keeps ids only).
pub fn render_report(
    s: &ReadoutScreen,
    artifact_digest_hex: &str,
    token_strings: Option<&dyn Fn(u32) -> String>,
) -> String {
    let mut out = String::with_capacity(8192);
    let _ = writeln!(out, "readout_screen v{} (Issue 1250 T1)", ARTIFACT_VERSION);
    let _ = writeln!(
        out,
        "source={} layers={} n_ff={} n_embd={} vocab={} top_k={} rms_eps={:?}",
        s.source_label, s.n_layers, s.n_ff, s.n_embd, s.vocab, s.top_k, s.rms_eps
    );
    if let Some(d) = &s.file_digest {
        let _ = writeln!(out, "file_digest_blake3={}", hex(d));
    }
    let _ = writeln!(out, "artifact_digest_blake3={artifact_digest_hex}");
    let _ = writeln!(
        out,
        "units={} ({} neurons × 2 signs; bucket v1=UNASSIGNED — T2 assigns J-share)",
        s.layers.iter().map(|l| l.units.len()).sum::<usize>(),
        s.n_ff
    );
    let band_name = |b: Band| match b {
        Band::Early => "early",
        Band::Middle => "middle",
        Band::Late => "late",
    };
    let _ = writeln!(
        out,
        "sharp_median_all={:?} (the readable-units bar)",
        s.sharp_median_all
    );
    for r in &s.rollups {
        let _ = writeln!(
            out,
            "band {:>6} n={:>5} sharp mean={:?} med={:?} mad={:?} max={:?} | margin mean={:?} med={:?} mad={:?} max={:?} | readable={}",
            band_name(r.band),
            r.n_units,
            r.sharp_mean,
            r.sharp_median,
            r.sharp_mad,
            r.sharp_max,
            r.margin_mean,
            r.margin_median,
            r.margin_mad,
            r.margin_max,
            r.readable_units
        );
    }
    // Global top-20 units by sharpness (stable: value desc, then
    // layer/col/sign ascending — the collected order is already
    // (layer, col, +−), so a stable sort by value desc keeps it).
    let mut all: Vec<&UnitRow> = s
        .layers
        .iter()
        .flat_map(|l| l.units.iter().step_by(2))
        .collect();
    all.sort_by(|a, b| {
        b.logits[0]
            .partial_cmp(&a.logits[0])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let _ = writeln!(out, "top units by sharpness (+w sign):");
    for u in all.iter().take(20) {
        let toks: Vec<String> = (0..5)
            .filter_map(|i| {
                u.tokens.get(i).map(|&t| match token_strings {
                    Some(f) => format!("{}:{}", t, f(t)),
                    None => format!("{t}"),
                })
            })
            .collect();
        let _ = writeln!(
            out,
            "  l={:>2} c={:>4} sharp={:?} margin={:?} top5=[{}]",
            u.layer,
            u.col,
            u.logits[0],
            u.margin.unwrap_or(f32::NAN),
            toks.join(", ")
        );
    }
    out
}

pub fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{:02x}", b);
    }
    s
}

#[cfg(test)]
mod tests {
    #[test]
    fn g1_selection_matches_brute_force() {
        crate::readout_screen_fixture::gate_selection_matches_brute_force()
            .expect("selection gate");
    }

    #[test]
    fn g1_engineered_unit_reads_its_own_token() {
        crate::readout_screen_fixture::gate_engineered_unit().expect("engineered unit");
    }

    #[test]
    fn g1_sign_trick_bit_exact() {
        crate::readout_screen_fixture::gate_sign_trick_bit_exact().expect("sign trick");
    }

    #[test]
    fn g1_ties_first_index() {
        crate::readout_screen_fixture::gate_ties_first_index().expect("ties");
    }

    #[test]
    fn g1_band_formula_pinned() {
        crate::readout_screen_fixture::gate_band_formula().expect("bands");
    }

    #[test]
    fn g1_artifact_stable() {
        crate::readout_screen_fixture::gate_artifact_stable().expect("artifact");
    }

    #[test]
    fn g1_refusals_loud() {
        crate::readout_screen_fixture::gate_refusals_loud().expect("refusals");
    }

    #[test]
    fn g1_lens_norm_reference() {
        crate::readout_screen_fixture::gate_lens_norm_reference().expect("lens norm");
    }

    #[test]
    fn g1_report_deterministic() {
        crate::readout_screen_fixture::gate_report_deterministic().expect("report");
    }
}
