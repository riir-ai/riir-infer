//! `spike_code_census` — Issue 1252 A2: the spike-CODING census arm (the
//! offline decision-data pass; the SpikingBrain distill, katgpt-rs
//! Research 694 + the issue's option lanes).
//!
//! One offline pass per checkpoint: capture the per-layer MLP activations
//! (the down-proj INPUT slabs — the quantization surface an activation
//! format would ride), then measure, per layer and per k ∈ {2,3,4}:
//!
//! * **spike-count histogram** — per row, `|{i : s_i ≠ 0}|` under the
//!   adaptive-threshold code `V_th = mean|x|/k`, `s = round(x/V_th)`;
//!   histogrammed over rows (the paper's "spike count" is THEIRS, ours is
//!   measured — never imported);
//! * **element-zero rate** — the `sparsity(k)` point;
//! * **whole-128-group zero rate** — the only skip granularity coherent
//!   with the packed ternary matvec layout (a group all-zero means the
//!   fused lane can skip the weight block; element-gather never wins at
//!   31% density, the pre-registered refusal's arithmetic);
//! * **structural-zero decomposition** — the Gaussian floor
//!   `2Φ(0.399/k)−1` and the Laplacian floor `1−e^(−1/2k)` (the paper's
//!   thresholding-zero predictors under each row shape); the measured
//!   excess over the better-fitting floor is the post-GLU structural dead
//!   mass ρ₀ (the predictor-validation arm: the census spectrum vs held-out
//!   slices rides the same artifact).
//!
//! The bars that unlock the kernel lane live in the issue (group-zero
//! ≥ ~35% at k ≥ 3 on a league checkpoint); THIS module only measures —
//! decision data, never a kernel, never a serving path.
//!
//! # Shape (the `readout_screen` precedent)
//!
//! Feature-gated pure-local opt-in (`spike_code_census`); the math module
//! is tokenizer/model-free (fixture-gated without any GGUF); the capture
//! driver rides the `closed_form_write` CaptureSink seam (the i1189 lane's
//! `batched_mlp` hook — dense AND ternary bodies share it); the artifact is
//! canonical bytes + BLAKE3 (`SPKC1`, the RDMAP class); the bin is the
//! census front (`--self-test` = the fixture canary; required-features row
//! in the same commit — the P30 law). Issue-940 fold order everywhere;
//! first-index ties; no RNG (the census is deterministic by construction).

use std::path::Path;

use anyhow::{Context as _, Result};

// ── the census math (pure, fixture-gated — no model, no tokenizer) ─────

/// The group granularity: 128 contiguous channels (the packed ternary
/// matvec's block width — the issue's pinned skip granularity).
pub const GROUP_WIDTH: usize = 128;

/// The k ladder the census sweeps (the issue's pins; k=3 is the paper's
/// headline class).
pub const K_LADDER: [usize; 3] = [2, 3, 4];

/// Standard normal CDF (Abramowitz-Stegun 7.1.26, f64 — the floor
/// formulas' only transcendental; deterministic, no deps).
fn phi(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let poly = t
        * (0.319_381_530
            + t * (-0.356_563_782
                + t * (1.781_477_937 + t * (-1.821_255_978 + t * 1.330_274_429))));
    let p = 1.0 - (-(x * x) / 2.0).exp() / (std::f64::consts::TAU.sqrt()) * poly;
    if x >= 0.0 { p } else { 1.0 - p }
}

/// The Gaussian zero floor at parameter k: `2Φ(0.3989/k)−1` (0.3989 =
/// √(2/π)/2 — the mean-abs of a unit Gaussian halved; the issue's
/// transcription).
pub fn gaussian_zero_floor(k: usize) -> f64 {
    let kf = k as f64;
    2.0 * phi(0.398_942_280_401_432_7 / kf) - 1.0
}

/// The Laplacian zero floor at parameter k: `1−e^(−1/(2k))`.
pub fn laplace_zero_floor(k: usize) -> f64 {
    let kf = k as f64;
    1.0 - (-1.0 / (2.0 * kf)).exp()
}

/// Per-k census of ONE layer's activation rows (`rows[row] = the mlp-wide
/// activation vector`, row-major f32). Deterministic: Issue-940 sequential
/// folds; first-index ties; no RNG.
pub fn census_layer(rows: &[Vec<f32>], k: usize) -> Result<LayerSpikeStats> {
    if rows.is_empty() {
        anyhow::bail!("spike census: empty row set");
    }
    let width = rows[0].len();
    if width == 0 {
        anyhow::bail!("spike census: zero-width rows");
    }
    if rows.iter().any(|r| r.len() != width) {
        anyhow::bail!("spike census: ragged rows");
    }
    if k == 0 {
        anyhow::bail!("spike census: k=0 (division by zero threshold)");
    }
    let kf = k as f64;
    let groups = width.div_ceil(GROUP_WIDTH);
    // Histogram bins: spike count per row, 0..=width. Stored as the raw
    // per-row counts (the artifact carries counts, not just the histogram,
    // so the reader can re-bin) — plus the compact histogram for the report.
    let mut spike_counts = Vec::with_capacity(rows.len());
    let mut zero_elems: u64 = 0;
    let mut total_elems: u64 = 0;
    let mut zero_groups: u64 = 0;
    let mut total_groups: u64 = 0;
    let mut sum_abs: f64 = 0.0;
    for r in rows {
        // The adaptive threshold: V_th = mean|x| / k (the paper's form;
        // Issue-940 sequential fold for the mean).
        let mut mean_abs = 0.0f64;
        for &x in r {
            mean_abs += (x as f64).abs();
        }
        mean_abs /= width as f64;
        sum_abs += mean_abs;
        if mean_abs == 0.0 {
            // The degenerate all-zero row: V_th would be 0 and s = round(x/0)
            // is undefined — the census PINS it to zero spikes (the group-skip
            // reading wants dead rows skipped, never NaN-counted).
            spike_counts.push(0);
            zero_elems += width as u64;
            total_elems += width as u64;
            zero_groups += groups as u64;
            total_groups += groups as u64;
            continue;
        }
        let v_th = mean_abs / kf;
        let mut spikes = 0u64;
        let mut group_zeros = 0u64;
        let mut group_count = 0u64;
        for g in 0..groups {
            let lo = g * GROUP_WIDTH;
            let hi = (lo + GROUP_WIDTH).min(width);
            let mut group_zero = true;
            for &x in &r[lo..hi] {
                // s = round(x / V_th); s == 0 iff |x| < V_th/2 (round-to-1
                // at the exact tie — the predicate is the strict interior).
                let s = ((x as f64) / v_th).round();
                if s != 0.0 {
                    spikes += 1;
                    group_zero = false;
                }
            }
            if group_zero {
                group_zeros += 1;
            }
            group_count += 1;
        }
        spike_counts.push(spikes);
        zero_elems += (width as u64) - spikes;
        total_elems += width as u64;
        zero_groups += group_zeros;
        total_groups += group_count;
    }
    let hist_width = width + 1;
    let mut histogram = vec![0u64; hist_width];
    for &c in &spike_counts {
        histogram[c as usize] += 1;
    }
    let elem_zero_rate = zero_elems as f64 / total_elems as f64;
    let group_zero_rate = zero_groups as f64 / total_groups as f64;
    let mean_abs_overall = sum_abs / rows.len() as f64;
    Ok(LayerSpikeStats {
        rows: rows.len(),
        width,
        k,
        spike_counts,
        histogram,
        elem_zero_rate,
        group_zero_rate,
        gaussian_floor: gaussian_zero_floor(k),
        laplace_floor: laplace_zero_floor(k),
        mean_abs: mean_abs_overall,
    })
}

/// Per-layer, per-k census read-out.
#[derive(Debug, Clone)]
pub struct LayerSpikeStats {
    /// Rows captured (positions × sequences).
    pub rows: usize,
    /// Activation width (mlp).
    pub width: usize,
    /// The k this census ran at.
    pub k: usize,
    /// Spike count per row (|{i: s_i ≠ 0}|).
    pub spike_counts: Vec<u64>,
    /// Histogram of `spike_counts` (bin = count, value = rows).
    pub histogram: Vec<u64>,
    /// Element-zero rate (the sparsity(k) point).
    pub elem_zero_rate: f64,
    /// Whole-128-group zero rate (the skip ceiling).
    pub group_zero_rate: f64,
    /// The Gaussian floor at this k.
    pub gaussian_floor: f64,
    /// The Laplacian floor at this k.
    pub laplace_floor: f64,
    /// Mean per-row mean|x| (the threshold's numerator, for the record).
    pub mean_abs: f64,
}

impl LayerSpikeStats {
    /// The structural dead-mass BRACKET (ρ₀) over the two pinned floors:
    /// `(over_loosest, over_tightest)` = (measured − max_floor,
    /// measured − min_floor) clamped at 0. The issue's decomposition is a
    /// bracket, not a point: the true shape of the live part is an
    /// empirical question (the shape column is the point-estimate pick),
    /// and a single-floor reading hides the signal on either side — the
    /// nearest-floor pick absorbs the dead mass into the other shape's
    /// floor, and the loosest-floor pick under-reports when the live part
    /// is genuinely Gaussian (the G3 fixture caught both defects
    /// pre-landing: nearest-floor read 0.31% where the bracket is
    /// [0.31%, 5.08%]).
    pub fn structural_excess_bracket(&self) -> (f64, f64) {
        let hi = self.gaussian_floor.max(self.laplace_floor);
        let lo = self.gaussian_floor.min(self.laplace_floor);
        (
            (self.elem_zero_rate - hi).max(0.0),
            (self.elem_zero_rate - lo).max(0.0),
        )
    }

    /// The bracket's UPPER leg (over the tightest floor) — the
    /// unambiguous dead-mass reading the kernel-bar arithmetic reads.
    pub fn structural_excess(&self) -> f64 {
        self.structural_excess_bracket().1
    }

    /// Which floor sits nearer the MEASURED zero rate (true = Gaussian) —
    /// the report's shape DIAGNOSTIC and the bracket's point-estimate
    /// pick.
    pub fn gaussian_better(&self) -> bool {
        (self.elem_zero_rate - self.gaussian_floor).abs()
            <= (self.elem_zero_rate - self.laplace_floor).abs()
    }
}

// ── the artifact (SPKC1 — canonical bytes + BLAKE3, the RDMAP class) ───

/// Artifact magic + format version.
pub const SPKC_MAGIC: &[u8; 4] = b"SPKC";
pub const SPKC_VERSION: u8 = 1;

/// The census artifact: per (layer, k) stats. Layers in ascending order;
/// the k ladder inner. Canonical encoding: little-endian, f64-as-bits for
/// rates (the RDMAP class), counts as u64.
#[derive(Debug, Clone)]
pub struct SpikeCensusArtifact {
    /// Model identity (the weights BLAKE3 hex, as the sink_audit lane pins).
    pub model_digest_hex: String,
    /// Corpus identity (the token-source digest hex — the corpus file's
    /// BLAKE3, or a content hash of the token vector).
    pub corpus_digest_hex: String,
    /// Tokens captured per layer pass.
    pub tokens: usize,
    /// Activation width (mlp).
    pub width: usize,
    /// The k ladder (ascending).
    pub k_ladder: Vec<usize>,
    /// Layers (ascending).
    pub layers: Vec<usize>,
    /// Stats, indexed `[layer_slot * k_ladder.len() + k_slot]`.
    pub stats: Vec<LayerSpikeStats>,
}

impl SpikeCensusArtifact {
    /// Canonical bytes: header + rows. Deterministic (sorted inputs pinned
    /// by construction; f64 rates as-bits; u64 counts LE).
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(SPKC_MAGIC);
        b.push(SPKC_VERSION);
        b.extend_from_slice(&(self.tokens as u64).to_le_bytes());
        b.extend_from_slice(&(self.width as u64).to_le_bytes());
        b.extend_from_slice(&(self.k_ladder.len() as u64).to_le_bytes());
        for &k in &self.k_ladder {
            b.extend_from_slice(&(k as u64).to_le_bytes());
        }
        b.extend_from_slice(&(self.layers.len() as u64).to_le_bytes());
        for &l in &self.layers {
            b.extend_from_slice(&(l as u64).to_le_bytes());
        }
        // Digests as raw 32-byte halves (hex strings are the report's
        // business; the artifact carries the raw bytes).
        let md = blake3_hex_to_bytes(&self.model_digest_hex);
        b.extend_from_slice(&md);
        let cd = blake3_hex_to_bytes(&self.corpus_digest_hex);
        b.extend_from_slice(&cd);
        for s in &self.stats {
            b.extend_from_slice(&(s.rows as u64).to_le_bytes());
            b.extend_from_slice(&(s.k as u64).to_le_bytes());
            b.extend_from_slice(s.elem_zero_rate.to_bits().to_le_bytes().as_slice());
            b.extend_from_slice(s.group_zero_rate.to_bits().to_le_bytes().as_slice());
            b.extend_from_slice(s.gaussian_floor.to_bits().to_le_bytes().as_slice());
            b.extend_from_slice(s.laplace_floor.to_bits().to_le_bytes().as_slice());
            b.extend_from_slice(s.mean_abs.to_bits().to_le_bytes().as_slice());
            b.extend_from_slice(&s.zero_elems_sum().to_le_bytes());
            b.extend_from_slice(&s.total_elems_sum().to_le_bytes());
            b.extend_from_slice(&s.zero_groups_sum().to_le_bytes());
            b.extend_from_slice(&s.total_groups_sum().to_le_bytes());
        }
        b
    }

    /// BLAKE3 of [`Self::canonical_bytes`].
    pub fn digest(&self) -> [u8; 32] {
        *blake3::hash(&self.canonical_bytes()).as_bytes()
    }

    /// The report text (the record's table source; deterministic order).
    pub fn render(&self, label: &str) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "# spike-code census — {label}\n# model {} | corpus {} | tokens {} | width {}\n",
            short_hex(&self.model_digest_hex),
            short_hex(&self.corpus_digest_hex),
            self.tokens,
            self.width
        ));
        s.push_str("# layer  k  elem_zero  group_zero  gauss_floor  lapl_floor  excess_lo  excess_hi  shape\n");
        for (i, &l) in self.layers.iter().enumerate() {
            for (ki, &k) in self.k_ladder.iter().enumerate() {
                let st = &self.stats[i * self.k_ladder.len() + ki];
                let (elo, ehi) = st.structural_excess_bracket();
                s.push_str(&format!(
                    "{:>5}  {:>2}  {:9.5}  {:10.5}  {:11.5}  {:10.5}  {:9.5}  {:9.5}  {}\n",
                    l,
                    k,
                    st.elem_zero_rate,
                    st.group_zero_rate,
                    st.gaussian_floor,
                    st.laplace_floor,
                    elo,
                    ehi,
                    if st.gaussian_better() {
                        "gauss"
                    } else {
                        "lapl"
                    }
                ));
            }
        }
        s
    }

    /// Write the artifact (canonical bytes + digest sidecar).
    pub fn write(&self, dir: &Path) -> Result<[u8; 32]> {
        std::fs::create_dir_all(dir).with_context(|| format!("mkdir {}", dir.display()))?;
        let bytes = self.canonical_bytes();
        let d = self.digest();
        std::fs::write(dir.join("spike_census.spkc"), &bytes)
            .with_context(|| format!("write {}", dir.join("spike_census.spkc").display()))?;
        std::fs::write(dir.join("spike_census.blake3"), hex(&d))
            .with_context(|| "write digest".to_string())?;
        Ok(d)
    }
}

impl LayerSpikeStats {
    fn zero_elems_sum(&self) -> u64 {
        // Recovered: rate * total, but the canonical row carries the COUNTS
        // via the histogram sum bookkeeping — store the explicit counts.
        let total = (self.rows as u64) * (self.width as u64);
        let zeros = (self.elem_zero_rate * total as f64).round() as u64;
        let _ = zeros;
        // Exact: histogram[0] is rows with ZERO spikes; count bookkeeping
        // rides the rates (the histogram bins recover counts exactly).
        let mut zeros = 0u64;
        for (bin, &rows) in self.histogram.iter().enumerate() {
            zeros += ((self.width as u64) - bin as u64) * rows;
        }
        zeros
    }

    fn total_elems_sum(&self) -> u64 {
        (self.rows as u64) * (self.width as u64)
    }

    fn zero_groups_sum(&self) -> u64 {
        let total = (self.rows as u64) * (self.width.div_ceil(GROUP_WIDTH) as u64);
        (self.group_zero_rate * total as f64).round() as u64
    }

    fn total_groups_sum(&self) -> u64 {
        (self.rows as u64) * (self.width.div_ceil(GROUP_WIDTH) as u64)
    }
}

// ── small helpers ───────────────────────────────────────────────────────

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn short_hex(s: &str) -> String {
    s.chars().take(12).collect()
}

fn blake3_hex_to_bytes(s: &str) -> [u8; 32] {
    let mut out = [0u8; 32];
    let bytes = s.as_bytes();
    for (i, slot) in out.iter_mut().enumerate() {
        let hi = bytes
            .get(i * 2)
            .and_then(|c| (*c as char).to_digit(16))
            .unwrap_or(0) as u8;
        let lo = bytes
            .get(i * 2 + 1)
            .and_then(|c| (*c as char).to_digit(16))
            .unwrap_or(0) as u8;
        *slot = (hi << 4) | lo;
    }
    out
}

// ── the A2 gates (in-module; the fixture drives both postures) ─────────

#[cfg(all(test, feature = "spike_code_census"))]
mod tests {
    #[test]
    fn a2_g1_gaussian_floor_recovered() {
        crate::spike_code_census_fixture::run_all().expect("fixture battery");
    }
}
