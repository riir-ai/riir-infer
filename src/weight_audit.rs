//! Weight-structure audit lane (riir-infer Issue 1209; Research 025 —
//! the Lazarus distill, the published Microsoft Research detector).
//!
//! A **reference-free, weight-only** census over the last MLP's SwiGLU
//! block: five population-statistical features per neuron, computed
//! against the layer's OWN population (median/MAD z-scores, internal
//! ranks, internal cosines) — no baseline checkpoint, no delta, no
//! forward pass. This is the instrument katgpt-rs Issue 152 concluded
//! did not exist; its class boundary (Research 025 §2.4) stands: the
//! census is a tripwire for footprint-leaving sparse edits (the
//! Lazarus family), makes NO claim against R422-class
//! distribution-indistinguishable constructions, and its natural
//! massive-activation neurons are the standing false-flag hazard —
//! **it reports, it never convicts**.
//!
//! Posture (G5): opt-in `weight_audit`, report-only (never blocks a
//! load, never in a serving path), zero new deps, one deterministic
//! report (ordered f64 folds per the katgpt-rs Issue-940 law; pinned
//! first-index tie rules; the report text is the artifact and two runs
//! on the same file are bit-identical — G4). The census bin follows the
//! `family_census` shape: an offline reader over an open GGUF.
//!
//! Thresholds (v1): raw z-score / rank / cosine thresholds calibrated
//! on the synthetic fixture and pinned as constants — honestly labeled
//! in Research 025 as un-fitted (no logistic); the calibration margins
//! are recorded in the lane's bench note. Detection scope: the
//! Lazarus-shaped edit family until measured otherwise.
//!
//! Tensor axes (the GGUF layout, probed on both zoo models):
//! `ffn_gate`/`ffn_up` are `[n_embd, n_neurons]` — ne0 = row length =
//! d_model, one row per neuron. `ffn_down` is `[n_neurons, n_embd]` —
//! ne0 = row length = d_ff, one row per OUTPUT element j; neuron n's
//! write column is the n-th element of every row (length d_model).

use crate::gguf_loader::GgufFile;
use anyhow::{Context, Result, bail};
use std::cmp::Ordering;
use std::fmt::Write as _;

/// Normal-consistency MAD scale (robust z = (x − median)/(MAD_SCALE·MAD)).
const MAD_SCALE: f32 = 1.4826;

/// Alignment-feature scan window: the two alignment features evaluate on
/// the top-`TOP_M_SCAN` write-norm neurons (the paper evaluates them at
/// n† = top-1; the window covers multi-neuron plantings, which the
/// paper's Theorem 2 makes one neuron per payload slot). A neuron
/// outside the window can hold at most the three norm-population
/// features — below `Thresholds::min_flags` by construction.
pub const TOP_M_SCAN: usize = 32;

/// A neuron whose every norm is below this is a structural zero row
/// (all-zero blocks are plausible in ternary quant): excluded from the
/// population, counted in the report, never flagged.
const ZERO_NORM_FLOOR: f64 = 1e-30;

// ── Canonical thresholds (v1, fixture-calibrated; see the bench note) ──

/// The census decision thresholds. Canonical values are pinned here so
/// the zoo baseline and any future run read against ONE posture; the
/// fixture calibration margins live in the bench note.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Thresholds {
    /// Robust z on the three-projection sum `ln‖wg‖+ln‖wu‖+ln‖wd‖`.
    pub z_norms: f32,
    /// Summed extremity-of-rank across the three norm rankings (each in
    /// [0, 1], 3.0 = top in all three; ties → first-index, stable sort).
    pub rank_sum: f32,
    /// Robust z on `ln‖wd‖` (the write magnitude).
    pub z_write: f32,
    /// cosine(gate row, up row) — the Analytic tie.
    pub cos_align: f32,
    /// max_t cosine(Γ⊙wd, unemb_row_t − mean_row) — the aimed column.
    pub tok_align: f32,
    /// A neuron flags at ≥ this many exceeded features (of the 5).
    pub min_flags: usize,
}

impl Thresholds {
    /// The fixture-calibrated canonical posture (v1). Margins against
    /// the fixture populations are recorded in the lane's bench note.
    pub const CANONICAL: Thresholds = Thresholds {
        z_norms: 6.0,
        rank_sum: 2.4,
        z_write: 6.0,
        cos_align: 0.85,
        tok_align: 0.85,
        min_flags: 4,
    };
}

impl Default for Thresholds {
    fn default() -> Self {
        Thresholds::CANONICAL
    }
}

// ── Deterministic folds (the katgpt-rs Issue-940 law) ────────────────

/// Sequential ascending-index f64 dot of two f32 rows, promoted per
/// element. Plain `+=` — never a platform SIMD kernel: this report's
/// numbers must be bit-reproducible.
#[inline]
fn ordered_dot_f64(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f64;
    for i in 0..a.len() {
        acc += a[i] as f64 * b[i] as f64;
    }
    acc
}

/// Sequential ascending-index f64 sum of squares of an f32 row.
#[inline]
fn ordered_sumsq_f64(row: &[f32]) -> f64 {
    let mut acc = 0.0f64;
    for &v in row {
        acc += v as f64 * v as f64;
    }
    acc
}

/// Total-order compare for finite f32s (the census refuses non-finite
/// values at read time, so `unwrap_or` is unreachable for sorted data).
fn cmp_f32(a: &f32, b: &f32) -> Ordering {
    a.partial_cmp(b).unwrap_or(Ordering::Equal)
}

// ── The tensor source abstraction ────────────────────────────────────

/// One SwiGLU MLP block + the readout head, as the census consumes it.
/// The GGUF source ([`GgufMlpSource`]) and the test fixture implement
/// the same trait, so fixture arms and zoo runs run ONE census code
/// path.
///
/// Row semantics (the GGUF layout, see the module header): `gate_row(n)`
/// / `up_row(n)` are neuron n's `n_embd`-length rows; `down_out_row(j)`
/// is output element j's `n_neurons`-length row of `ffn_down` — neuron
/// n's write column is `down_out_row(j)[n]` over all j.
pub trait MlpTensors {
    fn n_neurons(&self) -> usize;
    fn n_embd(&self) -> usize;
    fn vocab(&self) -> usize;
    fn gate_row(&self, n: usize) -> Result<Vec<f32>>;
    fn up_row(&self, n: usize) -> Result<Vec<f32>>;
    fn down_out_row(&self, j: usize) -> Result<Vec<f32>>;
    fn unemb_row(&self, t: usize) -> Result<Vec<f32>>;
    /// The final-norm gain vector Γ (length `n_embd`).
    fn gamma(&self) -> &[f32];
    /// Source label for the report (`gguf:output.weight` / `fixture`).
    fn label(&self) -> String;
    /// The audited block's layer index (0 for synthetic blocks).
    fn layer(&self) -> usize {
        0
    }
}

/// A GGUF-backed source: the last MLP (or an explicit layer) of an open
/// GGUF + the untied (`output.weight`) or tied (`token_embd.weight`)
/// unembedding + `output_norm.weight` as Γ.
pub struct GgufMlpSource<'a> {
    gguf: &'a GgufFile,
    layer: usize,
    n_neurons: usize,
    n_embd: usize,
    vocab: usize,
    gate: String,
    up: String,
    down: String,
    unemb: String,
    /// `output.weight` vs the tied table — recorded in the report.
    unemb_label: String,
    gamma: Vec<f32>,
}

impl<'a> GgufMlpSource<'a> {
    /// Open the census source. `layer = None` picks the highest
    /// `blk.N` carrying an `ffn_gate.weight` (the paper's "last MLP");
    /// an explicit layer must carry the ffn trio or the open refuses
    /// loud. Unembedding preference: `output.weight`, else the tied
    /// `token_embd.weight` (recorded in the report either way).
    pub fn open(gguf: &'a GgufFile, layer: Option<usize>) -> Result<Self> {
        let mut last: Option<usize> = None;
        for info in &gguf.tensor_infos {
            if let Some(rest) = info.name.strip_prefix("blk.")
                && let Some((idx, suffix)) = rest.split_once('.')
                && suffix == "ffn_gate.weight"
                && let Ok(i) = idx.parse::<usize>()
            {
                last = Some(match last {
                    Some(m) if m >= i => m,
                    _ => i,
                });
            }
        }
        let layer = match layer {
            Some(l) => l,
            None => last.with_context(|| "no blk.N.ffn_gate.weight tensor in this GGUF")?,
        };
        let gate = format!("blk.{layer}.ffn_gate.weight");
        let up = format!("blk.{layer}.ffn_up.weight");
        let down = format!("blk.{layer}.ffn_down.weight");
        for name in [&gate, &up, &down] {
            if gguf.tensor_info(name).is_none() {
                bail!("layer {layer}: tensor '{name}' not found");
            }
        }
        let gate_shape = gguf.tensor_info(&gate).unwrap().shape.clone();
        let down_shape = gguf.tensor_info(&down).unwrap().shape.clone();
        // GGUF order: ne0 = row length (innermost), ne1 = rows.
        if gate_shape.len() != 2 || down_shape.len() != 2 {
            bail!("ffn tensors must be 2-D (gate {gate_shape:?}, down {down_shape:?})");
        }
        let n_embd = gate_shape[0];
        let n_neurons = gate_shape[1];
        if down_shape[0] != n_neurons || down_shape[1] != n_embd {
            bail!(
                "ffn_down shape {down_shape:?} disagrees with ffn_gate {gate_shape:?} \
                 (neuron/output axis swap?)"
            );
        }
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
        // Γ = output_norm.weight (F32 vector; the row-0 dequant reads it).
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
            layer,
            n_neurons,
            n_embd,
            vocab,
            gate,
            up,
            down,
            unemb,
            unemb_label,
            gamma,
        })
    }

    /// The audited block's layer index.
    pub fn layer(&self) -> usize {
        self.layer
    }
}

impl MlpTensors for GgufMlpSource<'_> {
    fn n_neurons(&self) -> usize {
        self.n_neurons
    }
    fn n_embd(&self) -> usize {
        self.n_embd
    }
    fn vocab(&self) -> usize {
        self.vocab
    }
    fn gate_row(&self, n: usize) -> Result<Vec<f32>> {
        self.gguf.dequant_tensor_row(&self.gate, n, self.n_embd)
    }
    fn up_row(&self, n: usize) -> Result<Vec<f32>> {
        self.gguf.dequant_tensor_row(&self.up, n, self.n_embd)
    }
    fn down_out_row(&self, j: usize) -> Result<Vec<f32>> {
        self.gguf.dequant_tensor_row(&self.down, j, self.n_neurons)
    }
    fn unemb_row(&self, t: usize) -> Result<Vec<f32>> {
        self.gguf.dequant_tensor_row(&self.unemb, t, self.n_embd)
    }
    fn gamma(&self) -> &[f32] {
        &self.gamma
    }
    fn label(&self) -> String {
        format!("gguf:{}", self.unemb_label)
    }
    fn layer(&self) -> usize {
        self.layer
    }
}

// ── The census ───────────────────────────────────────────────────────

/// One window neuron's full feature row (the top-`TOP_M_SCAN` write-norm
/// neurons; these are the only neurons the alignment features evaluate,
/// and the only neurons that can reach `min_flags`).
#[derive(Debug, Clone)]
pub struct WindowRow {
    pub neuron: usize,
    /// ‖wd_n‖ (the down-column write norm).
    pub write_norm: f32,
    /// ln‖wg‖+ln‖wu‖+ln‖wd‖ robust z.
    pub z_norms: f32,
    /// Summed extremity-of-rank over the three norm rankings.
    pub rank_sum: f32,
    /// ln‖wd‖ robust z.
    pub z_write: f32,
    /// cosine(wg_n, wu_n).
    pub cos_align: f32,
    /// max_t cosine(Γ⊙wd, unemb_t − mean) — evaluated at n† only;
    /// `None` for the rest of the window.
    pub tok_align: Option<f32>,
    /// The argmax token of the alignment (first-index tie rule).
    pub tok_align_token: Option<usize>,
    /// How many thresholded features this row exceeds.
    pub flags: usize,
}

/// The one-layer census result (everything the report renders).
#[derive(Debug, Clone)]
pub struct LayerAudit {
    pub n_neurons: usize,
    pub n_embd: usize,
    pub vocab: usize,
    pub layer: usize,
    pub source_label: String,
    /// Neurons excluded as structural zero rows.
    pub excluded_zero_norm: usize,
    /// The three-projection population's pinned median / MAD.
    pub three_proj_median: f32,
    pub three_proj_mad: f32,
    /// The write-norm log-population's pinned median / MAD.
    pub write_median: f32,
    pub write_mad: f32,
    /// n† = the write-norm argmax (first-index tie rule).
    pub ndag: usize,
    /// The top-`TOP_M_SCAN` rows by write norm, descending (stable).
    pub window: Vec<WindowRow>,
    /// Flagged neuron indices, ascending.
    pub flags: Vec<usize>,
    pub thresholds: Thresholds,
    /// Population maxima (the calibration evidence the report carries).
    pub pop_max_z_norms: f32,
    pub pop_max_rank_sum: f32,
    pub pop_max_z_write: f32,
}

/// Run the five-feature census over one MLP block.
///
/// Population passes: gate + up rows (per-neuron norms), the down rows
/// (per-neuron write-column norms), then the unembedding mean + the
/// token alignment at n†. All folds ordered f64; all sorts stable with
/// first-index tie rules; non-finite weights refuse loud.
pub fn census_mlp(src: &dyn MlpTensors, thresholds: Thresholds) -> Result<LayerAudit> {
    let n = src.n_neurons();
    let d = src.n_embd();
    let v = src.vocab();
    if n < 16 || d < 16 || v < 8 {
        bail!("degenerate block: n={n} d={d} v={v} (minimum 16/16/8)");
    }
    if src.gamma().len() != d {
        bail!("gamma length {} != n_embd {d}", src.gamma().len());
    }

    // Pass A: per-neuron norms. wg/wu from their rows; wd accumulated
    // per neuron (column) across the down_out_rows.
    let mut wg_norm = vec![0.0f32; n];
    let mut wu_norm = vec![0.0f32; n];
    let mut wd_norm = vec![0.0f32; n];
    let mut wd_ss = vec![0.0f64; n];
    for i in 0..n {
        let g = src.gate_row(i)?;
        let u = src.up_row(i)?;
        check_finite(&g, &format!("ffn_gate row {i}"))?;
        check_finite(&u, &format!("ffn_up row {i}"))?;
        wg_norm[i] = (ordered_sumsq_f64(&g).sqrt()) as f32;
        wu_norm[i] = (ordered_sumsq_f64(&u).sqrt()) as f32;
    }
    for j in 0..d {
        let r = src.down_out_row(j)?;
        if r.len() != n {
            bail!("ffn_down row {j} length {} != n_neurons {n}", r.len());
        }
        check_finite(&r, &format!("ffn_down row {j}"))?;
        for (neuron, &val) in r.iter().enumerate() {
            wd_ss[neuron] += val as f64 * val as f64;
        }
    }
    for (i, &ss) in wd_ss.iter().enumerate() {
        wd_norm[i] = (ss.sqrt()) as f32;
    }
    check_finite(&wg_norm, "gate norms")?;
    check_finite(&wu_norm, "up norms")?;
    check_finite(&wd_norm, "down norms")?;

    // Structural zero rows: excluded from every population, never flagged.
    let is_zero = |i: usize| {
        (wd_norm[i] as f64) <= ZERO_NORM_FLOOR
            && (wg_norm[i] as f64) <= ZERO_NORM_FLOOR
            && (wu_norm[i] as f64) <= ZERO_NORM_FLOOR
    };
    let excluded = (0..n).filter(|&i| is_zero(i)).count();
    let active: Vec<usize> = (0..n).filter(|&i| !is_zero(i)).collect();
    if active.len() < 16 {
        bail!(
            "active population {}/{} below the 16 floor after zero-row exclusion",
            active.len(),
            n
        );
    }

    // Feature 1: three-projection sum robust z (log norms, active only).
    let three_proj: Vec<f32> = active
        .iter()
        .map(|&i| wg_norm[i].ln() + wu_norm[i].ln() + wd_norm[i].ln())
        .collect();
    let (tp_median, tp_mad) = robust_stats(&three_proj)?;
    let z_norms: Vec<f32> = robust_z(&three_proj, tp_median, tp_mad);

    // Feature 5: write-magnitude robust z.
    let write_pop: Vec<f32> = active.iter().map(|&i| wd_norm[i].ln()).collect();
    let (w_median, w_mad) = robust_stats(&write_pop)?;
    let z_write: Vec<f32> = robust_z(&write_pop, w_median, w_mad);

    // Feature 2: summed extremity-of-rank over the three norm rankings.
    let rank_sum: Vec<f32> = {
        let wg_a: Vec<f32> = active.iter().map(|&i| wg_norm[i]).collect();
        let wu_a: Vec<f32> = active.iter().map(|&i| wu_norm[i]).collect();
        let wd_a: Vec<f32> = active.iter().map(|&i| wd_norm[i]).collect();
        let rw = ranks_ascending(&wg_a);
        let ru = ranks_ascending(&wu_a);
        let rd = ranks_ascending(&wd_a);
        let m = active.len() - 1;
        (0..active.len())
            .map(|i| extremity(rw[i], m) + extremity(ru[i], m) + extremity(rd[i], m))
            .collect()
    };

    let pop_max_z_norms = fold_max(&z_norms);
    let pop_max_rank_sum = fold_max(&rank_sum);
    let pop_max_z_write = fold_max(&z_write);

    // n† = write-norm argmax over the active population (first index on
    // ties: `>` never `>=`).
    let mut ndag_a = 0usize;
    for i in 1..active.len() {
        if wd_norm[active[i]] > wd_norm[active[ndag_a]] {
            ndag_a = i;
        }
    }
    let ndag = active[ndag_a];

    // Window: top-M by write norm, descending, stable (ties keep the
    // original neuron order = first-index rule).
    let mut window_a: Vec<usize> = (0..active.len()).collect();
    window_a.sort_by(|&a, &b| cmp_f32(&wd_norm[active[b]], &wd_norm[active[a]]));
    window_a.truncate(TOP_M_SCAN.min(active.len()));

    // Feature 3 per window row: cosine(gate, up) at the neuron.
    let mut rows: Vec<WindowRow> = Vec::with_capacity(window_a.len());
    for &ai in &window_a {
        let neuron = active[ai];
        let g = src.gate_row(neuron)?;
        let u = src.up_row(neuron)?;
        let dot = ordered_dot_f64(&g, &u);
        let ng = ordered_sumsq_f64(&g).sqrt();
        let nu = ordered_sumsq_f64(&u).sqrt();
        let cos = if ng > 0.0 && nu > 0.0 {
            (dot / (ng * nu)) as f32
        } else {
            0.0
        };
        let flags = count_flags(
            z_norms[ai],
            rank_sum[ai],
            z_write[ai],
            Some(cos),
            None,
            thresholds,
        );
        rows.push(WindowRow {
            neuron,
            write_norm: wd_norm[neuron],
            z_norms: z_norms[ai],
            rank_sum: rank_sum[ai],
            z_write: z_write[ai],
            cos_align: cos,
            tok_align: None,
            tok_align_token: None,
            flags,
        });
    }

    // Feature 4 at n† only: the aimed-column cosine.
    let (tok, tok_tok) = tok_align_at(src, ndag)?;
    if let Some(r) = rows.iter_mut().find(|r| r.neuron == ndag) {
        r.tok_align = Some(tok);
        r.tok_align_token = Some(tok_tok);
        r.flags = count_flags(
            r.z_norms,
            r.rank_sum,
            r.z_write,
            Some(r.cos_align),
            r.tok_align,
            thresholds,
        );
    }

    let mut flags: Vec<usize> = rows
        .iter()
        .filter(|r| r.flags >= thresholds.min_flags)
        .map(|r| r.neuron)
        .collect();
    flags.sort_unstable();

    Ok(LayerAudit {
        n_neurons: n,
        n_embd: d,
        vocab: v,
        layer: src.layer(),
        source_label: src.label(),
        excluded_zero_norm: excluded,
        three_proj_median: tp_median,
        three_proj_mad: tp_mad,
        write_median: w_median,
        write_mad: w_mad,
        ndag,
        window: rows,
        flags,
        thresholds,
        pop_max_z_norms,
        pop_max_rank_sum,
        pop_max_z_write,
    })
}

/// Feature 4 at `ndag`: a = Γ ⊙ wd_col(n†); two streaming passes over
/// the unembedding (mean, then max cosine) — no materialization. The
/// write column is read in the same pass as the alignment extraction:
/// `a[j] = Γ[j] · down_out_row(j)[ndag]`.
fn tok_align_at(src: &dyn MlpTensors, ndag: usize) -> Result<(f32, usize)> {
    let d = src.n_embd();
    let v = src.vocab();
    let gamma = src.gamma();
    let mut a = vec![0.0f64; d];
    for j in 0..d {
        let r = src.down_out_row(j)?;
        a[j] = gamma[j] as f64 * r[ndag] as f64;
    }
    let a_norm = {
        let mut s = 0.0f64;
        for &x in &a {
            s += x * x;
        }
        s.sqrt()
    };
    // The f32 view folds once, deterministically, before every dot.
    let a_f32: Vec<f32> = a.iter().map(|&x| x as f32).collect();

    // Pass 1: mean unembedding row (f64, ascending token order).
    let mut mean = vec![0.0f64; d];
    for t in 0..v {
        let r = src.unemb_row(t)?;
        check_finite(&r, &format!("unemb row {t}"))?;
        for (j, &val) in r.iter().enumerate() {
            mean[j] += val as f64;
        }
    }
    for x in mean.iter_mut() {
        *x /= v as f64;
    }
    let mean_f32: Vec<f32> = mean.iter().map(|&x| x as f32).collect();

    // Pass 2: max cosine (first-index tie rule: strictly greater).
    let mut best = f64::NEG_INFINITY;
    let mut best_t = 0usize;
    for t in 0..v {
        let r = src.unemb_row(t)?;
        let mut centered = vec![0.0f32; d];
        for j in 0..d {
            centered[j] = r[j] - mean_f32[j];
        }
        let dot = ordered_dot_f64(&a_f32, &centered);
        let dn = (ordered_sumsq_f64(&centered).sqrt()) as f64;
        if dn > 0.0 && a_norm > 0.0 {
            let c = dot / (a_norm * dn);
            if c > best {
                best = c;
                best_t = t;
            }
        }
    }
    if best.is_finite() {
        Ok((best as f32, best_t))
    } else {
        bail!("token alignment undefined (zero write column or zero rows)")
    }
}

// ── Feature helpers ──────────────────────────────────────────────────

fn fold_max(values: &[f32]) -> f32 {
    let mut m = f32::NEG_INFINITY;
    for &v in values {
        if v > m {
            m = v;
        }
    }
    m
}

fn check_finite(row: &[f32], what: &str) -> Result<()> {
    for (i, &v) in row.iter().enumerate() {
        if !v.is_finite() {
            bail!("non-finite weight at {what}[{i}] — refusing the census");
        }
    }
    Ok(())
}

/// Pinned median (ascending sort, lower-middle element — never
/// interpolated) + pinned MAD (same rule over |x − median|).
fn robust_stats(values: &[f32]) -> Result<(f32, f32)> {
    if values.is_empty() {
        bail!("empty population");
    }
    let mut s = values.to_vec();
    s.sort_by(cmp_f32);
    let med = s[(s.len() - 1) / 2];
    let mut dev: Vec<f32> = values.iter().map(|&v| (v - med).abs()).collect();
    dev.sort_by(cmp_f32);
    let mad = dev[(dev.len() - 1) / 2];
    if mad <= 0.0 {
        bail!("degenerate population: MAD == 0 (every neuron identical?)");
    }
    Ok((med, mad))
}

fn robust_z(values: &[f32], median: f32, mad: f32) -> Vec<f32> {
    let scale = MAD_SCALE * mad;
    values.iter().map(|&v| (v - median) / scale).collect()
}

/// Ascending stable ranks (ties → first index gets the lower rank).
fn ranks_ascending(values: &[f32]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_by(|&a, &b| cmp_f32(&values[a], &values[b]));
    let mut r = vec![0usize; values.len()];
    for (pos, &i) in idx.iter().enumerate() {
        r[i] = pos;
    }
    r
}

/// Extremity of rank `r` in a population of `m + 1`: 0 at the median,
/// 1 at either extreme (pinned formula; ties inherit the first-index
/// rank, the conservative direction).
#[inline]
fn extremity(r: usize, m: usize) -> f32 {
    2.0 * ((r as f32) / (m as f32) - 0.5).abs()
}

fn count_flags(
    z_norms: f32,
    rank_sum: f32,
    z_write: f32,
    cos_align: Option<f32>,
    tok_align: Option<f32>,
    t: Thresholds,
) -> usize {
    let mut c = 0;
    if z_norms >= t.z_norms {
        c += 1;
    }
    if rank_sum >= t.rank_sum {
        c += 1;
    }
    if z_write >= t.z_write {
        c += 1;
    }
    if cos_align.is_some_and(|v| v >= t.cos_align) {
        c += 1;
    }
    if tok_align.is_some_and(|v| v >= t.tok_align) {
        c += 1;
    }
    c
}

// ── The report ───────────────────────────────────────────────────────

/// File-level header facts the bin stamps into the report.
#[derive(Debug, Clone)]
pub struct ReportHeader {
    pub file: String,
    pub arch: Option<String>,
    /// blake3 over the whole file bytes (`None` when unavailable).
    pub file_digest_blake3: Option<String>,
}

/// Render the deterministic report. Floats print with Rust's shortest
/// round-trip `{:?}`; every ordering is pinned; the text is the G4
/// artifact (two runs → byte-identical).
pub fn render_report(a: &LayerAudit, h: &ReportHeader) -> String {
    let mut out = String::with_capacity(8192);
    let _ = writeln!(out, "weight_audit v1");
    let _ = writeln!(out, "file={}", h.file);
    if let Some(arch) = &h.arch {
        let _ = writeln!(out, "arch={arch}");
    }
    if let Some(dg) = &h.file_digest_blake3 {
        let _ = writeln!(out, "file_digest_blake3={dg}");
    }
    let _ = writeln!(
        out,
        "layer={} unembedding={} n_neurons={} n_embd={} vocab={}",
        a.layer, a.source_label, a.n_neurons, a.n_embd, a.vocab
    );
    let _ = writeln!(out, "excluded_zero_norm={}", a.excluded_zero_norm);
    let _ = writeln!(
        out,
        "pop three_proj_median={:?} three_proj_mad={:?} write_median={:?} write_mad={:?}",
        a.three_proj_median, a.three_proj_mad, a.write_median, a.write_mad
    );
    let _ = writeln!(
        out,
        "pop max_z_norms={:?} max_rank_sum={:?} max_z_write={:?}",
        a.pop_max_z_norms, a.pop_max_rank_sum, a.pop_max_z_write
    );
    let t = a.thresholds;
    let _ = writeln!(
        out,
        "thresholds z_norms={:?} rank_sum={:?} z_write={:?} cos_align={:?} tok_align={:?} min_flags={} top_m={}",
        t.z_norms, t.rank_sum, t.z_write, t.cos_align, t.tok_align, t.min_flags, TOP_M_SCAN
    );
    let _ = writeln!(out, "ndag={}", a.ndag);
    for r in &a.window {
        let _ = writeln!(
            out,
            "window n={:>6} wd={:?} z_norms={:?} rank={:?} z_write={:?} cos={:?} tok={:?} tok_token={:?} flags={}",
            r.neuron,
            r.write_norm,
            r.z_norms,
            r.rank_sum,
            r.z_write,
            r.cos_align,
            r.tok_align,
            r.tok_align_token,
            r.flags
        );
    }
    let _ = writeln!(out, "flags count={} list={:?}", a.flags.len(), a.flags);
    let verdict = if a.flags.is_empty() {
        "clean"
    } else {
        "FLAGGED"
    };
    let _ = writeln!(out, "verdict: {verdict}");
    out
}
