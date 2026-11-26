//! The trigger-vote screen for finetuned checkpoints (riir-infer Issue
//! 1251 T1; katgpt-rs Research 693 Part 2, Appendix C.4 — the Anthropic
//! Fellows pair's "neuron voting" backdoor probe, re-derived on OUR
//! lanes).
//!
//! **The published class is cited, never claimed novel:** arXiv:2611.19304
//! ("A Backdoor Trigger Is Legible in Fine-Tune Weights") and PEFTGuard
//! (IEEE S&P 2025) own the mechanism; this module is the supply-chain
//! SCREEN for checkpoints we ingest — the detective half of the 268/422
//! provenance story. The fixture lane (T2) rides our own
//! `closed_form_write`/`weight_edit_overlay` machinery so the planter
//! and the detector share one substrate and one falsifier.
//!
//! **What T1 is:** a weights-only vote over a (base, finetune) GGUF
//! pair. For every MLP neuron `i` (a `ffn_gate.weight`/`ffn_up.weight`
//! row), the per-token pre-activation delta is
//! `Δgate[i]·e_t = (W_gate_ft[i] − W_gate_base[i])·e_t` (and the `up`
//! twin), with `e_t` the **base** embedding row — the pinned proxy for
//! the MLP input (weights-only: the residual stream is not computable
//! here; the rmsnorm-γ variant is a T4+ ablation, recorded as a pin,
//! not an assumption). Each neuron votes for its top-`VOTES_PER_NEURON`
//! tokens by `|Δgate·e_t| + |Δup·e_t|` (the magnitude-of-change rule,
//! C.4's "changed most"; sum-of-abs is the pinned symmetric
//! combination). Tokens are ranked by vote count — a backdoored
//! finetune concentrates votes on its trigger tokens.
//!
//! **Pinned determinism (the Issue-940 law — this artifact's numbers
//! are digested):** dots fold sequentially ascending in f64 via
//! [`crate::readout_screen::ordered_dot_f64`] (never a platform SIMD
//! kernel); scores cast to f32 once, ranked value-desc with the
//! **first-index tie rule** (lower token id first) via the same
//! [`SideK`] selection the 1250 lens lane uses; the final token ranking
//! sorts `(votes desc, token asc)` with a total order — same file pair
//! → bit-identical artifact.
//!
//! **The layer-band admission rule (C.4, pinned reading):** each layer's
//! top-`BAND_WIDTH` token list must share ≥ `BAND_MIN_OVERLAP` tokens
//! with the previous layer's list to stay linked. The screen takes the
//! **best maximal contiguous run** of linked layers — mass (total
//! votes inside the run) desc, earlier start on a tie. A single layer is
//! a legal band; an unlinked layer starts a new run. The final ranking
//! pools votes over the band's layers only.
//!
//! **Verdicts:** [`Verdict::Voted`] (a band exists), [`Verdict::ZeroDelta`]
//! (every weight row bit-identical — no finetune to screen), and
//! [`Verdict::ZeroVotes`] (rows changed but every score landed exactly 0
//! — the delta is orthogonal to the whole embedding table). Zero-score
//! candidates never vote: a delta the embeddings cannot see casts no
//! vote.
//!
//! **Row-skip law:** a neuron whose gate AND up rows are bit-identical
//! between base and finetune is skipped outright (f32-bit compare of
//! the dequantized rows; dequantization is deterministic, so identical
//! bytes → identical rows). n_active counts only changed rows — the
//! artifact records the skipped rows as all-`u32::MAX` padding.
//!
//! Posture: opt-in `readout_screen`, offline by construction — weights
//! only, no forward pass, no serving path, zero new deps (rayon +
//! blake3 already in-tree). Parallelism partitions whole neurons; every
//! neuron's ranking is one task's ascending token scan, so the result
//! is deterministic by construction. The artifact is canonical binary
//! (`VOTE1\x01`) + BLAKE3 digest, bound to both source files' whole-file
//! digests.

use crate::gguf_loader::GgufFile;
use crate::readout_screen::{SideK, file_digest_blake3, ordered_dot_f64};
use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use std::fmt::Write as _;

/// Votes per neuron (paper C.4: the top-15 changed tokens).
pub const VOTES_PER_NEURON: usize = 15;
/// Width of the per-layer admission lists (C.4: top-100).
pub const BAND_WIDTH: usize = 100;
/// Minimum shared tokens between consecutive layers' top-`BAND_WIDTH`
/// lists for the layers to link (C.4: ≥10).
pub const BAND_MIN_OVERLAP: usize = 10;
/// Ranked tokens shipped in the artifact (bounded; the report prints a
/// prefix).
pub const RANKED_N: usize = 4096;

/// Population floors (a real checkpoint is far above these; the fixture
/// is the only legal tenant of the minimum band).
const MIN_LAYERS: usize = 1;
const MIN_EMBD: usize = 8;
const MIN_FF: usize = 8;
const MIN_VOCAB: usize = 16;

/// Artifact magic + format version.
pub const ARTIFACT_MAGIC: &[u8; 6] = b"VOTE1\x01";
pub const ARTIFACT_VERSION: u32 = 1;

/// Header: magic 6 + version 4 + votes_per_neuron/band_width/
/// band_min_overlap/ranked_n/n_layers/n_ff/n_embd/vocab 4×9 + verdict
/// 1 + reserved 3 + band_start 4 + band_end 4 + n_active_rows 8 +
/// base_digest 32 + ft_digest 32 = 126.
pub const HEADER_BYTES: usize = 126;

/// Per-neuron row: layer u32 + neuron u32 +
/// `VOTES_PER_NEURON × (token u32 + score bits u32)` = 128. An inactive
/// (skipped) row ships `u32::MAX` tokens with zero bits.
pub const ROW_BYTES: usize = 8 + 8 * VOTES_PER_NEURON;

// ── Verdicts ─────────────────────────────────────────────────────────

/// The screen's verdict (u8 in the artifact header).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Verdict {
    /// A contiguous band of linked layers exists; the ranking is live.
    Voted = 0,
    /// Every weight row bit-identical between base and finetune —
    /// nothing to screen.
    ZeroDelta = 1,
    /// Rows changed, but every score landed exactly 0 — the delta is
    /// orthogonal to the entire embedding table. No vote, no band.
    ZeroVotes = 2,
}

impl Verdict {
    fn from_u8(b: u8) -> Result<Self> {
        match b {
            0 => Ok(Verdict::Voted),
            1 => Ok(Verdict::ZeroDelta),
            2 => Ok(Verdict::ZeroVotes),
            other => bail!("unknown verdict byte {other}"),
        }
    }
}

// ── The tensor source abstraction ────────────────────────────────────

/// One checkpoint side's MLP weight rows + the embedding table, as the
/// vote consumes it. The GGUF source ([`GgufVoteSource`]) and the test
/// fixture implement the same trait, so fixture arms and real runs run
/// ONE code path.
///
/// Row semantics (the GGUF layout): `ffn_gate.weight`/`ffn_up.weight`
/// are `[n_ff × n_embd]` row-major — neuron `i`'s gate vector is ROW
/// `i` (`ne[0] = n_embd` = row length, `ne[1] = n_ff` = rows; the
/// `ffn_down` transpose reading is the mirrored case). `emb_rows()` is
/// the token embedding, row-major `[vocab × n_embd]` (row `t` = `e_t`).
pub trait VoteTensors: Sync {
    fn n_layers(&self) -> usize;
    fn n_ff(&self) -> usize;
    fn n_embd(&self) -> usize;
    fn vocab(&self) -> usize;
    /// The full gate matrix of one layer, row-major `[n_ff × n_embd]`.
    fn gate_rows(&self, layer: usize) -> Result<Vec<f32>>;
    /// The full up matrix of one layer, row-major `[n_ff × n_embd]`.
    fn up_rows(&self, layer: usize) -> Result<Vec<f32>>;
    /// The token embedding table, row-major `[vocab × n_embd]`.
    fn emb_rows(&self) -> Result<Vec<f32>>;
    /// Source label for the report.
    fn label(&self) -> String;
    /// Whole-source BLAKE3 (`None` when the source is not a file — the
    /// fixture). Both sides carry digests for the identity
    /// short-circuit and the artifact's provenance binding.
    fn file_digest(&self) -> Option<[u8; 32]> {
        None
    }
}

/// A GGUF-backed vote source: every `blk.N` carrying the gate/up pair
/// (a layer missing either tensor refuses loud — a partial sweep would
/// silently skew the band rule), plus `token_embd.weight` (the INPUT
/// embedding the delta dots read; the output/unembedding head is
/// deliberately NOT a fallback — the vote reads pre-activations, not
/// logits). The whole-file BLAKE3 is hashed streaming at open (the
/// artifact's provenance binding).
pub struct GgufVoteSource<'a> {
    gguf: &'a GgufFile,
    path_digest: Option<[u8; 32]>,
    n_layers: usize,
    n_ff: usize,
    n_embd: usize,
    vocab: usize,
    gate: Vec<String>,
    up: Vec<String>,
    emb: String,
}

impl<'a> GgufVoteSource<'a> {
    /// Open the vote source over every gate/up-carrying block (ascending
    /// layer index; contiguity asserted — a hole would silently shift
    /// the band).
    pub fn open(gguf: &'a GgufFile, path: &std::path::Path) -> Result<Self> {
        let mut layer_ids: Vec<usize> = Vec::new();
        for info in &gguf.tensor_infos {
            if let Some(rest) = info.name.strip_prefix("blk.")
                && let Some((idx, suffix)) = rest.split_once('.')
                && suffix == "ffn_gate.weight"
                && let Ok(i) = idx.parse::<usize>()
            {
                layer_ids.push(i);
            }
        }
        if layer_ids.is_empty() {
            bail!("no blk.N.ffn_gate.weight tensor in this GGUF");
        }
        layer_ids.sort_unstable();
        layer_ids.dedup();
        let n_layers = layer_ids.len();
        for (pos, &id) in layer_ids.iter().enumerate() {
            if id != pos {
                bail!(
                    "ffn layers are not contiguous (blk.{id} at position {pos}) — \
                     refusing the partial sweep"
                );
            }
        }
        let gate: Vec<String> = layer_ids
            .iter()
            .map(|i| format!("blk.{i}.ffn_gate.weight"))
            .collect();
        let up: Vec<String> = layer_ids
            .iter()
            .map(|i| format!("blk.{i}.ffn_up.weight"))
            .collect();
        for name in &up {
            if gguf.tensor_info(name).is_none() {
                bail!("layer tensor '{name}' not found — refusing the partial sweep");
            }
        }
        // gate/up: ne[0] = row length = n_embd, ne[1] = rows = n_ff.
        let gate_shape = gguf.tensor_info(&gate[0]).unwrap().shape.clone();
        let up_shape = gguf.tensor_info(&up[0]).unwrap().shape.clone();
        if gate_shape.len() != 2 || up_shape.len() != 2 {
            bail!("ffn_gate/ffn_up must be 2-D (gate {gate_shape:?}, up {up_shape:?})");
        }
        if gate_shape != up_shape {
            bail!(
                "ffn_gate shape {gate_shape:?} != ffn_up shape {up_shape:?} — \
                 refusing the asymmetric layer"
            );
        }
        let (n_embd, n_ff) = (gate_shape[0], gate_shape[1]);
        // Every layer must match layer 0's geometry (a heterogeneous
        // checkpoint would break the per-layer row census).
        for (l, name) in gate.iter().enumerate().skip(1) {
            let s = gguf.tensor_info(name).unwrap().shape.clone();
            if s != gate_shape {
                bail!("layer {l} gate shape {s:?} != layer 0 {gate_shape:?}");
            }
        }
        // The INPUT embedding (pre-activation proxy); never the
        // output head.
        if gguf.tensor_info("token_embd.weight").is_none() {
            bail!("no token_embd.weight — the vote reads input embeddings, not logits");
        }
        let emb_shape = gguf.tensor_info("token_embd.weight").unwrap().shape.clone();
        if emb_shape.len() != 2 || emb_shape[0] != n_embd {
            bail!(
                "token_embd shape {emb_shape:?} disagrees with n_embd {n_embd} \
                 (want [n_embd × vocab])"
            );
        }
        let vocab = emb_shape[1];
        Ok(Self {
            gguf,
            path_digest: file_digest_blake3(path),
            n_layers,
            n_ff,
            n_embd,
            vocab,
            gate,
            up,
            emb: "token_embd.weight".to_string(),
        })
    }
}

impl VoteTensors for GgufVoteSource<'_> {
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
    fn gate_rows(&self, layer: usize) -> Result<Vec<f32>> {
        self.gguf
            .dequant_f16_to_f32(&self.gate[layer])
            .with_context(|| format!("dequant {}", self.gate[layer]))
    }
    fn up_rows(&self, layer: usize) -> Result<Vec<f32>> {
        self.gguf
            .dequant_f16_to_f32(&self.up[layer])
            .with_context(|| format!("dequant {}", self.up[layer]))
    }
    fn emb_rows(&self) -> Result<Vec<f32>> {
        self.gguf
            .dequant_f16_to_f32(&self.emb)
            .with_context(|| format!("dequant {}", self.emb))
    }
    fn label(&self) -> String {
        "gguf:token_embd.weight".to_string()
    }
    fn file_digest(&self) -> Option<[u8; 32]> {
        self.path_digest
    }
}

// ── The vote ─────────────────────────────────────────────────────────

/// One layer's vote table + admission bookkeeping.
#[derive(Debug, Clone)]
pub struct LayerVote {
    pub layer: usize,
    /// Per-token vote counts (`vocab` long; > 0 only for voted tokens).
    pub votes: Vec<u32>,
    /// The layer's top-`BAND_WIDTH` tokens (votes desc, token asc).
    pub top: Vec<u32>,
    /// Overlap with the previous layer's top list (`None` for layer 0;
    /// `Some(0)` when either side is empty).
    pub overlap_prev: Option<usize>,
    /// Weight rows that differed (bit-wise) between base and finetune.
    pub n_active: usize,
}

/// One active neuron's top-15 selection (scores f32 bits in the
/// artifact; tokens only in memory).
#[derive(Debug, Clone)]
pub struct NeuronVoteRow {
    pub layer: usize,
    pub neuron: usize,
    /// Voted tokens, best first (score desc, token asc).
    pub tokens: Vec<u32>,
    /// The matching scores (f32; `|Δgate·e_t| + |Δup·e_t|`).
    pub scores: Vec<f32>,
}

/// The full vote result.
#[derive(Debug, Clone)]
pub struct TriggerVote {
    pub n_layers: usize,
    pub n_ff: usize,
    pub n_embd: usize,
    pub vocab: usize,
    pub verdict: Verdict,
    /// The admitted contiguous band `[start, end]` (inclusive; `None`
    /// when no band).
    pub band: Option<(usize, usize)>,
    pub layers: Vec<LayerVote>,
    /// Active neurons' rows, sorted by (layer, neuron).
    pub rows: Vec<NeuronVoteRow>,
    /// The band's pooled ranking: `(token, votes)`, votes desc, token
    /// asc, at most `RANKED_N` entries (empty unless [`Verdict::Voted`]).
    pub ranked: Vec<(u32, u32)>,
    /// Weight rows that differed across the whole checkpoint.
    pub n_active_rows: u64,
    pub base_digest: Option<[u8; 32]>,
    pub ft_digest: Option<[u8; 32]>,
}

/// Votes > 0 count (total vote mass of a layer).
fn layer_mass(l: &LayerVote) -> u64 {
    l.votes.iter().map(|&v| v as u64).sum()
}

/// Rank voted tokens: votes desc, token asc. Returns at most `k` ids.
fn rank_top(votes: &[u32], k: usize) -> Vec<u32> {
    let mut idx: Vec<u32> = (0..votes.len() as u32)
        .filter(|&t| votes[t as usize] > 0)
        .collect();
    idx.sort_by(|&a, &b| votes[b as usize].cmp(&votes[a as usize]).then(a.cmp(&b)));
    idx.truncate(k);
    idx
}

/// Overlap count of two sorted-ascending token lists (both are sorted:
/// `rank_top` emits votes desc then token asc — sort a copy ascending
/// for the two-pointer walk).
fn overlap_of(a: &[u32], b: &[u32]) -> usize {
    let mut sa: Vec<u32> = a.to_vec();
    let mut sb: Vec<u32> = b.to_vec();
    sa.sort_unstable();
    sb.sort_unstable();
    let (mut i, mut j, mut n) = (0usize, 0usize, 0usize);
    while i < sa.len() && j < sb.len() {
        match sa[i].cmp(&sb[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                n += 1;
                i += 1;
                j += 1;
            }
        }
    }
    n
}

/// The best maximal contiguous run of linked layers: mass desc, earlier
/// start on a tie. A layer links to its predecessor iff both tops are
/// nonempty and the overlap ≥ [`BAND_MIN_OVERLAP`].
fn pick_band(layers: &[LayerVote]) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize, u64)> = None;
    let consider =
        |best: &mut Option<(usize, usize, u64)>, s: usize, e: usize, mass: u64| match best {
            None => *best = Some((s, e, mass)),
            Some((bs, _, bm)) if mass > *bm || (mass == *bm && s < *bs) => {
                *best = Some((s, e, mass));
            }
            _ => {}
        };
    let mut run_start: Option<usize> = None;
    for l in 0..layers.len() {
        let has = !layers[l].top.is_empty();
        let linked_prev = l > 0
            && has
            && !layers[l - 1].top.is_empty()
            && layers[l].overlap_prev.unwrap_or(0) >= BAND_MIN_OVERLAP;
        match (run_start, linked_prev, has) {
            (None, _, true) => run_start = Some(l),
            (Some(_), false, true) => {
                // Close the open run and open a new one at l.
                let s = run_start.take().unwrap();
                let mass: u64 = (s..l).map(|i| layer_mass(&layers[i])).sum();
                consider(&mut best, s, l - 1, mass);
                run_start = Some(l);
            }
            (Some(_), true, _) => {}
            (_, _, false) => {
                if let Some(s) = run_start.take() {
                    // s < l always (a run opens only on a has-layer and
                    // l itself is not one).
                    let mass: u64 = (s..l).map(|i| layer_mass(&layers[i])).sum();
                    consider(&mut best, s, l - 1, mass);
                }
            }
        }
    }
    if let Some(s) = run_start {
        let e = layers.len() - 1;
        let mass: u64 = (s..=e).map(|i| layer_mass(&layers[i])).sum();
        consider(&mut best, s, e, mass);
    }
    best.map(|(s, e, _)| (s, e))
}

/// Run the vote over a (base, finetune) pair. The pair must agree on
/// the whole geometry (a mismatch refuses loud naming the field — never
/// a silent partial screen).
pub fn trigger_vote(base: &dyn VoteTensors, ft: &dyn VoteTensors) -> Result<TriggerVote> {
    let (n_layers, n_ff, n_embd, vocab) =
        (base.n_layers(), base.n_ff(), base.n_embd(), base.vocab());
    if ft.n_layers() != n_layers {
        bail!(
            "layer census mismatch: base {n_layers} vs ft {} — the pair is not \
             the same architecture",
            ft.n_layers()
        );
    }
    if ft.n_ff() != n_ff {
        bail!(
            "n_ff mismatch: base {n_ff} vs ft {} — the pair is not the same \
             architecture",
            ft.n_ff()
        );
    }
    if ft.n_embd() != n_embd {
        bail!(
            "n_embd mismatch: base {n_embd} vs ft {} — the pair is not the \
             same architecture",
            ft.n_embd()
        );
    }
    if ft.vocab() != vocab {
        bail!(
            "vocab mismatch: base {vocab} vs ft {} — the pair is not the \
             same tokenizer space",
            ft.vocab()
        );
    }
    if n_layers < MIN_LAYERS || n_embd < MIN_EMBD || n_ff < MIN_FF || vocab < MIN_VOCAB {
        bail!(
            "degenerate source: layers={n_layers} n_embd={n_embd} n_ff={n_ff} \
             vocab={vocab} (minimums {MIN_LAYERS}/{MIN_EMBD}/{MIN_FF}/{MIN_VOCAB})"
        );
    }
    if VOTES_PER_NEURON >= vocab {
        bail!("VOTES_PER_NEURON {VOTES_PER_NEURON} >= vocab {vocab}");
    }
    let base_digest = base.file_digest();
    let ft_digest = ft.file_digest();
    // Identity short-circuit BEFORE any tensor materialization:
    // identical files dequantize identically, so every row is skipped —
    // the pass cannot produce anything but ZeroDelta, and loading a
    // 50 GB-class pair to prove it would be pure waste. (The fixture
    // passes `None` digests and exercises the full pass instead.)
    if matches!((base_digest, ft_digest), (Some(a), Some(b)) if a == b) {
        let layers_out = (0..n_layers)
            .map(|layer| LayerVote {
                layer,
                votes: vec![0u32; vocab],
                top: Vec::new(),
                overlap_prev: if layer == 0 { None } else { Some(0) },
                n_active: 0,
            })
            .collect();
        return Ok(TriggerVote {
            n_layers,
            n_ff,
            n_embd,
            vocab,
            verdict: Verdict::ZeroDelta,
            band: None,
            layers: layers_out,
            rows: Vec::new(),
            ranked: Vec::new(),
            n_active_rows: 0,
            base_digest,
            ft_digest,
        });
    }
    // Pinned: the BASE embedding table is the input proxy for both
    // sides (the trigger direction is defined at the base's input
    // space; the ft-embedding variant is a T4+ ablation).
    let emb = base.emb_rows()?;
    if emb.len() != vocab * n_embd {
        bail!(
            "embedding length {} != vocab {vocab} × n_embd {n_embd}",
            emb.len()
        );
    }
    for (i, &x) in emb.iter().enumerate() {
        if !x.is_finite() {
            bail!("non-finite embedding element at [{i}] — refusing the vote");
        }
    }

    let mut layers_out: Vec<LayerVote> = Vec::with_capacity(n_layers);
    let mut rows: Vec<NeuronVoteRow> = Vec::new();
    let mut n_active_total: u64 = 0;

    for layer in 0..n_layers {
        let bg = base.gate_rows(layer)?;
        let fg = ft.gate_rows(layer)?;
        let bu = base.up_rows(layer)?;
        let fu = ft.up_rows(layer)?;
        let want = n_ff * n_embd;
        if bg.len() != want || fg.len() != want || bu.len() != want || fu.len() != want {
            bail!(
                "layer {layer} tensor length mismatch (gate {}/{} up {}/{} want {want})",
                bg.len(),
                fg.len(),
                bu.len(),
                fu.len()
            );
        }
        for (name, t) in [
            ("ffn_gate base", &bg),
            ("ffn_gate ft", &fg),
            ("ffn_up base", &bu),
            ("ffn_up ft", &fu),
        ] {
            for (idx, &x) in t.iter().enumerate() {
                if !x.is_finite() {
                    bail!("non-finite {name} weight at layer {layer} element {idx}");
                }
            }
        }
        // Active-row census: bit-compare the dequantized rows (gate OR
        // up differing marks the row).
        let active: Vec<usize> = (0..n_ff)
            .filter(|&i| {
                let (a, b) = (
                    &bg[i * n_embd..(i + 1) * n_embd],
                    &fg[i * n_embd..(i + 1) * n_embd],
                );
                let (c, d) = (
                    &bu[i * n_embd..(i + 1) * n_embd],
                    &fu[i * n_embd..(i + 1) * n_embd],
                );
                a.iter().zip(b).any(|(x, y)| x.to_bits() != y.to_bits())
                    || c.iter().zip(d).any(|(x, y)| x.to_bits() != y.to_bits())
            })
            .collect();
        n_active_total += active.len() as u64;

        let mut votes = vec![0u32; vocab];
        if !active.is_empty() {
            // One task per active neuron: the delta rows are L1-resident,
            // the embedding table streams ascending once.
            let scanned: Vec<Option<(Vec<u32>, Vec<f32>)>> = active
                .par_iter()
                .map(|&i| {
                    let (bg_r, fg_r) = (
                        &bg[i * n_embd..(i + 1) * n_embd],
                        &fg[i * n_embd..(i + 1) * n_embd],
                    );
                    let (bu_r, fu_r) = (
                        &bu[i * n_embd..(i + 1) * n_embd],
                        &fu[i * n_embd..(i + 1) * n_embd],
                    );
                    let mut dg = vec![0f32; n_embd];
                    let mut du = vec![0f32; n_embd];
                    for t in 0..n_embd {
                        dg[t] = fg_r[t] - bg_r[t];
                        du[t] = fu_r[t] - bu_r[t];
                    }
                    let mut sel = SideK::new(VOTES_PER_NEURON, true);
                    for t in 0..vocab {
                        let e_row = &emb[t * n_embd..(t + 1) * n_embd];
                        let g = ordered_dot_f64(&dg, e_row);
                        let u = ordered_dot_f64(&du, e_row);
                        // Zero scores never vote: a delta the
                        // embeddings cannot see casts no vote.
                        let s = (g.abs() + u.abs()) as f32;
                        if s > 0.0 {
                            sel.push(s, t as u32);
                        }
                    }
                    if sel.is_empty() {
                        None
                    } else {
                        Some(sel.into_parts())
                    }
                })
                .collect();
            for (pos, scan) in scanned.into_iter().enumerate() {
                let i = active[pos];
                if let Some((tokens, scores)) = scan {
                    for &tok in &tokens {
                        votes[tok as usize] += 1;
                    }
                    rows.push(NeuronVoteRow {
                        layer,
                        neuron: i,
                        tokens,
                        scores,
                    });
                }
            }
        }
        let top = rank_top(&votes, BAND_WIDTH);
        let overlap_prev = if layer == 0 {
            None
        } else {
            Some(overlap_of(&top, &layers_out[layer - 1].top))
        };
        layers_out.push(LayerVote {
            layer,
            votes,
            top,
            overlap_prev,
            n_active: active.len(),
        });
    }

    // rows are pushed in (layer asc, neuron asc) order by construction.
    let band = pick_band(&layers_out);
    let verdict = if n_active_total == 0 {
        Verdict::ZeroDelta
    } else if band.is_none() {
        Verdict::ZeroVotes
    } else {
        Verdict::Voted
    };
    let ranked = match (verdict, band) {
        (Verdict::Voted, Some((s, e))) => {
            let mut pooled = vec![0u32; vocab];
            for l in &layers_out[s..=e] {
                for (p, &v) in pooled.iter_mut().zip(l.votes.iter()) {
                    *p += v;
                }
            }
            let pairs: Vec<(u32, u32)> = rank_top(&pooled, RANKED_N)
                .into_iter()
                .map(|t| (t, pooled[t as usize]))
                .collect();
            pairs
        }
        _ => Vec::new(),
    };

    Ok(TriggerVote {
        n_layers,
        n_ff,
        n_embd,
        vocab,
        verdict,
        band,
        layers: layers_out,
        rows,
        ranked,
        n_active_rows: n_active_total,
        base_digest,
        ft_digest,
    })
}

// ── The artifact ─────────────────────────────────────────────────────

/// Serialize the vote: header + per-layer overlap table +
/// `n_layers × n_ff` neuron rows (inactive rows padded) + ranking.
pub fn build_artifact(v: &TriggerVote) -> Vec<u8> {
    let mut out = Vec::with_capacity(
        HEADER_BYTES + v.n_layers * 4 + v.n_layers * v.n_ff * ROW_BYTES + v.ranked.len() * 8,
    );
    out.extend_from_slice(ARTIFACT_MAGIC);
    out.extend_from_slice(&ARTIFACT_VERSION.to_le_bytes());
    out.extend_from_slice(&(VOTES_PER_NEURON as u32).to_le_bytes());
    out.extend_from_slice(&(BAND_WIDTH as u32).to_le_bytes());
    out.extend_from_slice(&(BAND_MIN_OVERLAP as u32).to_le_bytes());
    out.extend_from_slice(&(v.ranked.len() as u32).to_le_bytes());
    out.extend_from_slice(&(v.n_layers as u32).to_le_bytes());
    out.extend_from_slice(&(v.n_ff as u32).to_le_bytes());
    out.extend_from_slice(&(v.n_embd as u32).to_le_bytes());
    out.extend_from_slice(&(v.vocab as u32).to_le_bytes());
    out.push(v.verdict as u8);
    out.extend_from_slice(&[0u8; 3]); // reserved
    let (bs, be) = v.band.unwrap_or((u32::MAX as usize, 0));
    out.extend_from_slice(&(bs as u32).to_le_bytes());
    out.extend_from_slice(&(be as u32).to_le_bytes());
    out.extend_from_slice(&v.n_active_rows.to_le_bytes());
    out.extend_from_slice(&v.base_digest.unwrap_or([0u8; 32]));
    out.extend_from_slice(&v.ft_digest.unwrap_or([0u8; 32]));
    debug_assert_eq!(out.len(), HEADER_BYTES);
    for l in &v.layers {
        let o = l.overlap_prev.unwrap_or(usize::MAX);
        out.extend_from_slice(&(o as u32).to_le_bytes());
    }
    // Rows in (layer asc, neuron asc); a cursor walk over the sorted
    // active rows emits padding for skipped neurons.
    let mut cursor = 0usize;
    for layer in 0..v.n_layers {
        for neuron in 0..v.n_ff {
            let hit = cursor < v.rows.len()
                && v.rows[cursor].layer == layer
                && v.rows[cursor].neuron == neuron;
            out.extend_from_slice(&(layer as u32).to_le_bytes());
            out.extend_from_slice(&(neuron as u32).to_le_bytes());
            if hit {
                let r = &v.rows[cursor];
                for k in 0..VOTES_PER_NEURON {
                    let tok = r.tokens.get(k).copied().unwrap_or(u32::MAX);
                    let bits = r.scores.get(k).map(|f| f.to_bits()).unwrap_or(0);
                    out.extend_from_slice(&tok.to_le_bytes());
                    out.extend_from_slice(&bits.to_le_bytes());
                }
                cursor += 1;
            } else {
                for _ in 0..VOTES_PER_NEURON {
                    out.extend_from_slice(&u32::MAX.to_le_bytes());
                    out.extend_from_slice(&0u32.to_le_bytes());
                }
            }
        }
    }
    debug_assert_eq!(cursor, v.rows.len());
    for (tok, votes) in &v.ranked {
        out.extend_from_slice(&tok.to_le_bytes());
        out.extend_from_slice(&votes.to_le_bytes());
    }
    out
}

/// BLAKE3 digest of the artifact bytes.
pub fn artifact_digest(bytes: &[u8]) -> [u8; 32] {
    *blake3::hash(bytes).as_bytes()
}

/// Parsed artifact header.
#[derive(Debug, Clone)]
pub struct VoteHeader {
    pub version: u32,
    pub votes_per_neuron: usize,
    pub band_width: usize,
    pub band_min_overlap: usize,
    pub ranked_n: usize,
    pub n_layers: usize,
    pub n_ff: usize,
    pub n_embd: usize,
    pub vocab: usize,
    pub verdict: Verdict,
    pub band_start: u32,
    pub band_end: u32,
    pub n_active_rows: u64,
    pub base_digest: [u8; 32],
    pub ft_digest: [u8; 32],
}

/// Parse the artifact header (a truncated or foreign file refuses
/// loud).
pub fn parse_header(bytes: &[u8]) -> Result<VoteHeader> {
    if bytes.len() < HEADER_BYTES || &bytes[..6] != ARTIFACT_MAGIC {
        bail!("not a VOTE1 artifact (magic/length mismatch)");
    }
    let rd_u32 = |off: usize| -> u32 {
        u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
    };
    let version = rd_u32(6);
    if version != ARTIFACT_VERSION {
        bail!("VOTE1 version {version} != supported {ARTIFACT_VERSION}");
    }
    let mut base_digest = [0u8; 32];
    let mut ft_digest = [0u8; 32];
    base_digest.copy_from_slice(&bytes[62..94]);
    ft_digest.copy_from_slice(&bytes[94..126]);
    Ok(VoteHeader {
        version,
        votes_per_neuron: rd_u32(10) as usize,
        band_width: rd_u32(14) as usize,
        band_min_overlap: rd_u32(18) as usize,
        ranked_n: rd_u32(22) as usize,
        n_layers: rd_u32(26) as usize,
        n_ff: rd_u32(30) as usize,
        n_embd: rd_u32(34) as usize,
        vocab: rd_u32(38) as usize,
        verdict: Verdict::from_u8(bytes[42])?,
        band_start: rd_u32(46),
        band_end: rd_u32(50),
        n_active_rows: u64::from_le_bytes(bytes[54..62].try_into().unwrap()),
        base_digest,
        ft_digest,
    })
}

/// Human-readable report. `tok` (optional) decodes the ranked tokens'
/// strings for the printout (the artifact carries ids only).
pub fn render_report(
    v: &TriggerVote,
    tok: Option<&crate::tokenizer::BpeTokenizer>,
    top_n: usize,
) -> String {
    let mut out = String::with_capacity(4096);
    let _ = writeln!(out, "trigger_vote v{} (Issue 1251 T1)", ARTIFACT_VERSION);
    let _ = writeln!(
        out,
        "layers={} n_ff={} n_embd={} vocab={} verdict={} n_active_rows={}",
        v.n_layers,
        v.n_ff,
        v.n_embd,
        v.vocab,
        match v.verdict {
            Verdict::Voted => "VOTED",
            Verdict::ZeroDelta => "ZERO-DELTA",
            Verdict::ZeroVotes => "ZERO-VOTES",
        },
        v.n_active_rows
    );
    match v.band {
        Some((s, e)) => {
            let _ = writeln!(out, "band: layers [{s}..={e}] ({} layers)", e - s + 1);
        }
        None => {
            let _ = writeln!(out, "band: none");
        }
    }
    let _ = writeln!(out, "per-layer (active, votes, overlap_prev, top-8):");
    for l in &v.layers {
        let prev = match l.overlap_prev {
            None => "-".to_string(),
            Some(o) => o.to_string(),
        };
        let tops: Vec<String> = l
            .top
            .iter()
            .take(8)
            .map(|&t| match tok {
                Some(k) => k.decode(&[t as usize]),
                None => format!("#{t}"),
            })
            .collect();
        let mass: u64 = l.votes.iter().map(|&x| x as u64).sum();
        let _ = writeln!(
            out,
            "  layer {:>3}: active={:>5} votes={:>7} overlap={} top=[{}]",
            l.layer,
            l.n_active,
            mass,
            prev,
            tops.join(", ")
        );
    }
    if !v.ranked.is_empty() {
        let _ = writeln!(out, "ranked (top {top_n} of {}):", v.ranked.len());
        for (pos, (t, votes)) in v.ranked.iter().take(top_n).enumerate() {
            let s = match tok {
                Some(k) => k.decode(&[*t as usize]),
                None => String::new(),
            };
            let _ = writeln!(
                out,
                "  {:>3}. token {:>6} {:>5} votes  {:?}",
                pos + 1,
                t,
                votes,
                s
            );
        }
    }
    if let (Some(a), Some(b)) = (v.base_digest, v.ft_digest) {
        let _ = writeln!(out, "base blake3 {}", crate::readout_screen::hex(&a));
        let _ = writeln!(out, "ft   blake3 {}", crate::readout_screen::hex(&b));
    }
    out
}

#[cfg(test)]
mod tests {
    fn run_gate(name: &str, f: fn() -> anyhow::Result<()>) {
        f().unwrap_or_else(|e| panic!("{name}: {e:#}"));
    }

    #[test]
    fn t1_planted_trigger_ranked_first() {
        run_gate(
            "planted_trigger_ranked_first",
            crate::trigger_vote_fixture::gate_planted_trigger_ranked_first,
        );
    }

    #[test]
    fn t1_band_chaining_contiguous() {
        run_gate(
            "band_chaining_contiguous",
            crate::trigger_vote_fixture::gate_band_chaining_contiguous,
        );
    }

    #[test]
    fn t1_ties_first_index() {
        run_gate(
            "ties_first_index",
            crate::trigger_vote_fixture::gate_ties_first_index,
        );
    }

    #[test]
    fn t1_identity_zero_delta() {
        run_gate(
            "identity_zero_delta",
            crate::trigger_vote_fixture::gate_identity_zero_delta,
        );
    }

    #[test]
    fn t1_zero_scores_no_votes() {
        run_gate(
            "zero_scores_no_votes",
            crate::trigger_vote_fixture::gate_zero_scores_no_votes,
        );
    }

    #[test]
    fn t1_row_skip_only_active_votes() {
        run_gate(
            "row_skip_only_active_votes",
            crate::trigger_vote_fixture::gate_row_skip_only_active_votes,
        );
    }

    #[test]
    fn t1_artifact_stable() {
        run_gate(
            "artifact_stable",
            crate::trigger_vote_fixture::gate_artifact_stable,
        );
    }

    #[test]
    fn t1_dims_refusal() {
        run_gate(
            "dims_refusal",
            crate::trigger_vote_fixture::gate_dims_refusal,
        );
    }

    #[test]
    fn t1_layout_pinned() {
        run_gate(
            "layout_pinned",
            crate::trigger_vote_fixture::gate_layout_pinned,
        );
    }

    #[test]
    fn t1_parse_roundtrip() {
        run_gate(
            "parse_roundtrip",
            crate::trigger_vote_fixture::gate_parse_roundtrip,
        );
    }

    #[test]
    fn t1_run_all_receipt() {
        let receipts = crate::trigger_vote_fixture::run_all().expect("run_all");
        assert_eq!(receipts.len(), 10, "gate count drifted");
    }
}
