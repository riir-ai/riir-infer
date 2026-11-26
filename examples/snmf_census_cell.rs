//! snmf_census_cell — Issue 1255 T0: the SNMF group-feature fixture cell
//! over the REAL dense qwen35-hybrid (`Qwen3.5-0.8B-Base-Q8_0`), the
//! research-695 factorization (arXiv:2506.10920) run on OUR checkpoint.
//!
//! Capture REUSES the `closed_form_write` [`CaptureSink`] seam (the
//! down-proj input rows = the SwiGLU activations, the exact tensor the
//! paper decomposes) — one capture machinery, two consumers; zero new
//! forward code. One prefill over a fixed corpus window captures EVERY
//! layer's activation slab (the spike-code-census bin shape), so the
//! mixer-type question (does the per-layer mixer — attention vs DeltaNet
//! — shift the activation statistics?) is MEASURED across all layers,
//! never assumed.
//!
//! Phases:
//!   A. capture — corpus text → tokens[..window] → one prefill, all
//!      layers' activation slabs (`keys`), per-layer capture digest
//!   B. stats — per-layer activation statistics (mean/std/kurtosis/
//!      near-zero fraction/top-1% energy share), rollups per mixer type,
//!      the T0 mixer verdict line
//!   C. SNMF — semi-NMF (Ding 2010) per representative layer (first
//!      DeltaNet MLP + first Attention MLP): multiplicative Y updates
//!      (positive/negative-part split), the closed-form ridge Z step
//!      `Z ← A Yᵀ (YYᵀ + λI)⁻¹`, per-step column re-normalization (the
//!      scale folded into Y to preserve the product) + the hard WTA
//!      (top 1% of |z| per feature column), SplitMix64 seeded on the
//!      layer's capture digest — never an unseeded-global draw
//!   D. receipts — explained-norm of the SPARSE ZY (the honest number),
//!      per-feature neuron support census (support size, union,
//!      features-per-neuron overlap), objective trace
//!   E. determinism tripwire — the SNMF re-runs on the same capture +
//!      seed; the final Z/Y digest must be bit-identical
//!   F. T1 input map — Y's per-feature top tokens named against the
//!      corpus (decode via the GGUF BPE tokenizer), the readability
//!      census in the 1250 T1 shape (readable-features count + per-band
//!      rollups over Y-energy quartiles), and the `SNMFM1` artifact
//!      (canonical bytes + BLAKE3 trailer, digest printed)
//!
//! Determinism posture: every cross-cell float reduction folds chunk
//! partials COLLECTED IN ORDER (rayon `collect` is order-preserving, the
//! final fold is sequential) — no scheduling-dependent reduce order ever
//! feeds the loop or the receipts. All math f64.
//!
//! Fixture-tier posture: a reconstruction receipt, never an interpretive
//! or perf claim; T1 (input map + artifact) and T2 (hierarchy cell) ride
//! the same capture.
//!
//! ```bash
//! cargo run --release --features closed_form_write \
//!     --example snmf_census_cell -- \
//!     --gguf /Users/katopz/git/riir-train/data/Qwen3.5-0.8B-Base-Q8_0.gguf \
//!     --corpus /Users/katopz/git/riir-reflex/.raw/datasets/ag_news \
//!     --out .snmf_census_cell/
//! ```
//!
//! Timings printed are wall-clock on a possibly-loaded box — correctness
//! harness, never a perf claim.

use std::path::PathBuf;

use anyhow::{Context as _, Result, bail};
use rayon::prelude::*;
use riir_infer_core::closed_form_write::CaptureSink;
use riir_infer_core::corpus_text::load_corpus_text;
use riir_infer_core::deltanet::effective_rotary_dim;
use riir_infer_core::deltanet::forward::{
    HybridCache, HybridForwardScratch, PrefillContext, prefill_qwen_deltanet_into,
};
use riir_infer_core::gguf_loader::{GgmlType, GgufFile, load_qwen_deltanet_weights_gguf};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_core::types::DeltaNetLayerType;

/// The default Cell A model (box-stable on amo + bmo; `--gguf` overrides).
const DEFAULT_GGUF: &str = "/Users/katopz/git/riir-train/data/Qwen3.5-0.8B-Base-Q8_0.gguf";
/// The default capture corpus (the reflex canonical datasets dir; the
/// ag_news split pages the fleet corpus loader speaks). `--corpus` overrides.
const DEFAULT_CORPUS: &str = "/Users/katopz/git/riir-reflex/.raw/datasets/ag_news";

/// Window default: token samples per factorization (n columns of A). A
/// 768-sample window gives ~7.7 samples per feature at k=100.
const DEFAULT_WINDOW: usize = 768;
const DEFAULT_K: usize = 100;
const DEFAULT_ITERS: usize = 1500;
const DEFAULT_LAMBDA: f64 = 1e-2;
const DEFAULT_WTA_PCT: f64 = 1.0;
/// Early stop: relative objective change under this over [`CONV_PATIENCE`]
/// consecutive checks ends the loop (the paper converges in ~1.5–3.3k
/// iterations; the fixture cell may converge sooner).
const CONV_TOL: f64 = 1e-12;
const CONV_PATIENCE: usize = 6;
const OBJ_EVERY: usize = 25;

// ── args ────────────────────────────────────────────────────────────────

struct Args {
    gguf: PathBuf,
    corpus: PathBuf,
    out: PathBuf,
    window: usize,
    k: usize,
    iters: usize,
    lambda: f64,
    wta_pct: f64,
}

fn parse_args() -> Result<Args> {
    let mut gguf = PathBuf::from(DEFAULT_GGUF);
    let mut corpus = PathBuf::from(DEFAULT_CORPUS);
    let mut out = PathBuf::from(".snmf_census_cell");
    let mut window = DEFAULT_WINDOW;
    let mut k = DEFAULT_K;
    let mut iters = DEFAULT_ITERS;
    let mut lambda = DEFAULT_LAMBDA;
    let mut wta_pct = DEFAULT_WTA_PCT;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut val = |name: &str| -> Result<String> {
            it.next()
                .ok_or_else(|| anyhow::anyhow!("--{name} requires a value"))
        };
        match a.as_str() {
            "--gguf" => gguf = PathBuf::from(val("gguf")?),
            "--corpus" => corpus = PathBuf::from(val("corpus")?),
            "--out" => out = PathBuf::from(val("out")?),
            "--tokens" => window = val("tokens")?.parse()?,
            "--k" => k = val("k")?.parse()?,
            "--iters" => iters = val("iters")?.parse()?,
            "--lambda" => lambda = val("lambda")?.parse()?,
            "--wta-pct" => wta_pct = val("wta-pct")?.parse()?,
            other => bail!("unknown arg {other:?} (see the module doc for usage)"),
        }
    }
    if window < 4 * k {
        bail!("--tokens {window} < 4·k {k}: too few samples per feature for a fixture receipt");
    }
    if !(0.0 < wta_pct && wta_pct <= 50.0) {
        bail!("--wta-pct {wta_pct} outside (0, 50]");
    }
    Ok(Args {
        gguf,
        corpus,
        out,
        window,
        k,
        iters,
        lambda,
        wta_pct,
    })
}

// ── deterministic RNG (SplitMix64, seeded on the capture digest) ────────

struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in (0, 1].
    fn next_unit(&mut self) -> f64 {
        let v = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        if v <= 0.0 { f64::EPSILON } else { v }
    }
}

// ── SNMF (semi-NMF, research 695 / Ding 2010) ──────────────────────────

/// The factorization state. `a`/`at` hold the capture ([n × d_a] rows and
/// its transpose), `z` is [d_a × k] row-major (features as COLUMNS —
/// `z[r * k + j]` is neuron r's weight in feature j), `y` is [k × n]
/// row-major (coefficients ≥ 0).
struct Snmf {
    n: usize,
    d: usize,
    k: usize,
    a: Vec<Vec<f64>>,
    at: Vec<Vec<f64>>,
    z: Vec<f64>,
    y: Vec<f64>,
    lambda: f64,
    wta: usize,
}

impl Snmf {
    fn new(a_rows: &[Vec<f64>], k: usize, lambda: f64, wta_pct: f64, seed: u64) -> Self {
        let n = a_rows.len();
        let d = a_rows[0].len();
        let wta = (((wta_pct / 100.0) * d as f64).ceil() as usize).clamp(1, d);
        let mut rng = SplitMix64(seed);
        let y: Vec<f64> = (0..k * n).map(|_| rng.next_unit()).collect();
        Self {
            n,
            d,
            k,
            a: a_rows.to_vec(),
            at: Vec::new(),
            z: vec![0.0; d * k],
            y,
            lambda,
            wta,
        }
    }

    /// Materialize the transpose (once, before the first ridge step).
    fn build_transpose(&mut self) {
        self.at = vec![vec![0.0; self.n]; self.d];
        for (s, row) in self.a.iter().enumerate() {
            for (r, &v) in row.iter().enumerate() {
                self.at[r][s] = v;
            }
        }
    }

    /// Z ← A Yᵀ (YYᵀ + λI)⁻¹ (the closed-form ridge step). The normal
    /// matrix inverts by Gauss-Jordan with partial pivoting (λI keeps it
    /// SPD; a zero pivot is a caller-bug refusal, never a silent fix).
    fn ridge_z_step(&mut self) {
        let (n, d, k) = (self.n, self.d, self.k);
        // M = YYᵀ + λI.
        let mut m = vec![0.0f64; k * k];
        for j in 0..k {
            for i in j..k {
                let yj = &self.y[j * n..(j + 1) * n];
                let yi = &self.y[i * n..(i + 1) * n];
                let dot: f64 = yj.iter().zip(yi.iter()).map(|(&x, &v)| x * v).sum();
                m[j * k + i] = dot;
                m[i * k + j] = dot;
            }
            m[j * k + j] += self.lambda;
        }
        // Augmented [M | I], row-major.
        let w = 2 * k;
        let mut aug = vec![0.0f64; k * w];
        for j in 0..k {
            aug[j * w..j * w + k].copy_from_slice(&m[j * k..(j + 1) * k]);
            aug[j * w + k + j] = 1.0;
        }
        for col in 0..k {
            let mut piv = col;
            let mut best = aug[col * w + col].abs();
            for r in (col + 1)..k {
                let v = aug[r * w + col].abs();
                if v > best {
                    best = v;
                    piv = r;
                }
            }
            if best < 1e-300 {
                panic!(
                    "ridge step: singular normal matrix (lambda={})",
                    self.lambda
                );
            }
            if piv != col {
                for c in 0..w {
                    aug.swap(col * w + c, piv * w + c);
                }
            }
            let inv = 1.0 / aug[col * w + col];
            for c in 0..w {
                aug[col * w + c] *= inv;
            }
            for r in 0..k {
                if r == col {
                    continue;
                }
                let f = aug[r * w + col];
                if f == 0.0 {
                    continue;
                }
                for c in 0..w {
                    aug[r * w + c] -= f * aug[col * w + c];
                }
            }
        }
        // minv_t[j][i] = (M⁻¹)[i][j] — contiguous per feature for the Z dots.
        let mut minv_t = vec![0.0f64; k * k];
        for i in 0..k {
            for j in 0..k {
                minv_t[j * k + i] = aug[i * w + k + j];
            }
        }
        // B = A Yᵀ then Z_row = b_row · M⁻¹, per neuron row — rayon over
        // row chunks; every cell folds in a fixed ascending order, so the
        // scheduling cannot move a bit.
        let (at, y, minv_t) = (&self.at, &self.y, &minv_t);
        let chunk = (d / 24).max(1);
        let ranges: Vec<(usize, usize)> = (0..d)
            .step_by(chunk)
            .map(|r0| (r0, (r0 + chunk).min(d)))
            .collect();
        let updates: Vec<(usize, Vec<f64>)> = ranges
            .par_iter()
            .map(|&(r0, r1)| {
                let mut out = vec![0.0f64; (r1 - r0) * k];
                let mut b = vec![0.0f64; k];
                for (ri, r) in (r0..r1).enumerate() {
                    let at_row = &at[r];
                    for (i, y_row) in y.chunks_exact(n).enumerate() {
                        b[i] = at_row.iter().zip(y_row.iter()).map(|(&x, &v)| x * v).sum();
                    }
                    for (j, mrow) in minv_t.chunks_exact(k).enumerate() {
                        out[ri * k + j] = b.iter().zip(mrow.iter()).map(|(&x, &v)| x * v).sum();
                    }
                }
                (r0, out)
            })
            .collect();
        for (r0, out) in updates {
            self.z[r0 * k..r0 * k + out.len()].copy_from_slice(&out);
        }
    }

    /// One multiplicative Y update (semi-NMF: positive/negative-part
    /// split of A), reading the CURRENT (sparse) Z.
    fn y_update(&mut self) {
        let (n, d, k) = (self.n, self.d, self.k);
        // Zt [k × d] (the features as rows) + ZZ = ZᵀZ, both fixed folds.
        let zt: Vec<Vec<f64>> = (0..k)
            .map(|j| (0..d).map(|r| self.z[r * k + j]).collect())
            .collect();
        let mut zz = vec![0.0f64; k * k];
        for j in 0..k {
            for i in j..k {
                let dot: f64 = zt[j].iter().zip(zt[i].iter()).map(|(&x, &v)| x * v).sum();
                zz[j * k + i] = dot;
                zz[i * k + j] = dot;
            }
        }
        // Per sample s: za[j] = zt[j]·a[s]; den = max(−za, 0) + ZZ[j]·y_col;
        // Y ← Y ⊙ num ⊘ den (num = max(za, 0)). Rayon over samples with an
        // ORDERED collect — the apply below is sequential.
        let (a_ref, y_ref) = (&self.a, &self.y);
        let updates: Vec<(usize, Vec<f64>)> = (0..n)
            .into_par_iter()
            .map(|s| {
                let a_row = &a_ref[s];
                let mut new_col = vec![0.0f64; k];
                for j in 0..k {
                    let za: f64 = a_row.iter().zip(zt[j].iter()).map(|(&x, &v)| x * v).sum();
                    let neg = (-za).max(0.0);
                    let zz_y: f64 = zz[j * k..(j + 1) * k]
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| v * y_ref[i * n + s])
                        .sum();
                    let den = neg + zz_y;
                    let num = za.max(0.0);
                    new_col[j] = if den > 1e-12 {
                        y_ref[j * n + s] * num / (den + 1e-12)
                    } else {
                        0.0
                    };
                }
                (s, new_col)
            })
            .collect();
        for (s, col) in updates {
            for (j, v) in col.into_iter().enumerate() {
                self.y[j * n + s] = v;
            }
        }
    }

    /// Column re-normalization (the scale folded into Y so ZY is
    /// preserved) + the hard WTA per feature: keep the top [`Self::wta`]
    /// |z| entries (ties at the threshold keep the lowest neuron index
    /// first — a fixed strict tie rule), zero the rest.
    fn normalize_and_wta(&mut self) {
        let (d, k) = (self.d, self.k);
        let mut keep = vec![false; d];
        for j in 0..k {
            let nrm: f64 = (0..d)
                .map(|r| self.z[r * k + j] * self.z[r * k + j])
                .sum::<f64>()
                .sqrt();
            if nrm <= 1e-300 {
                continue;
            }
            for s in 0..self.n {
                self.y[j * self.n + s] *= nrm;
            }
            for r in 0..d {
                self.z[r * k + j] /= nrm;
            }
            let mut col: Vec<(f64, usize)> = (0..d).map(|r| (self.z[r * k + j].abs(), r)).collect();
            col.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            keep.iter_mut().for_each(|v| *v = false);
            for &(_, r) in col.iter().take(self.wta) {
                keep[r] = true;
            }
            for (r, &keep_r) in keep.iter().enumerate() {
                if !keep_r {
                    self.z[r * k + j] = 0.0;
                }
            }
        }
    }

    /// Relative residual ‖A − ZY‖_F / ‖A‖_F with the CURRENT (sparse) Z.
    /// The float reductions fold chunk partials COLLECTED IN ORDER — a
    /// scheduling-independent result (it feeds the early stop).
    fn rel_residual(&self) -> f64 {
        let (n, d, k) = (self.n, self.d, self.k);
        let a_partials: Vec<f64> = self
            .a
            .par_iter()
            .map(|row| row.iter().map(|&x| x * x).sum::<f64>())
            .collect();
        let a_norm = a_partials.iter().sum::<f64>().sqrt();
        let (a_ref, z_ref, y_ref) = (&self.a, &self.z, &self.y);
        let ss_partials: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|s| {
                let a_row = &a_ref[s];
                let mut acc = 0.0;
                for r in 0..d {
                    let mut zy = 0.0;
                    for j in 0..k {
                        zy += z_ref[r * k + j] * y_ref[j * n + s];
                    }
                    let e = a_row[r] - zy;
                    acc += e * e;
                }
                acc
            })
            .collect();
        let ss = ss_partials.iter().sum::<f64>();
        if a_norm <= 0.0 {
            return 0.0;
        }
        ss.sqrt() / a_norm
    }

    /// The final-state digest: Z bits + Y bits, fixed order — the
    /// determinism tripwire's identity.
    fn state_digest(&self) -> String {
        use blake3::Hasher;
        let mut h = Hasher::new();
        h.update(&(self.d as u64).to_le_bytes());
        h.update(&(self.k as u64).to_le_bytes());
        h.update(&(self.n as u64).to_le_bytes());
        for v in &self.z {
            h.update(&v.to_bits().to_le_bytes());
        }
        for v in &self.y {
            h.update(&v.to_bits().to_le_bytes());
        }
        h.finalize().to_hex().to_string()
    }

    /// Run the loop; returns the relative-residual trace (per check).
    fn run(&mut self, iters: usize) -> Vec<(usize, f64)> {
        self.build_transpose();
        self.ridge_z_step();
        self.normalize_and_wta();
        let mut trace = Vec::new();
        let mut last = self.rel_residual();
        trace.push((0, last));
        let mut stall = 0usize;
        for it in 1..=iters {
            self.y_update();
            self.ridge_z_step();
            self.normalize_and_wta();
            if it % OBJ_EVERY == 0 {
                let cur = self.rel_residual();
                trace.push((it, cur));
                if (last - cur).abs() < CONV_TOL * last.max(1e-300) {
                    stall += 1;
                    if stall >= CONV_PATIENCE {
                        break;
                    }
                } else {
                    stall = 0;
                }
                last = cur;
            }
        }
        trace
    }
}

// ── capture + stats ─────────────────────────────────────────────────────

struct LayerCapture {
    mixer: &'static str,
    rows: Vec<Vec<f64>>,
    capture_digest: String,
}

fn capture_digest_of(rows: &[Vec<f32>], mlp: usize) -> String {
    use blake3::Hasher;
    let mut h = Hasher::new();
    h.update(&(rows.len() as u64).to_le_bytes());
    h.update(&(mlp as u64).to_le_bytes());
    for row in rows {
        for v in row {
            h.update(&v.to_bits().to_le_bytes());
        }
    }
    h.finalize().to_hex().to_string()
}

/// (mean, std, excess kurtosis, near-zero fraction, top-1% energy share).
/// Every reduction folds ORDER-COLLECTED chunk partials.
fn act_stats(rows: &[Vec<f64>]) -> (f64, f64, f64, f64, f64) {
    let n_total: usize = rows.iter().map(|r| r.len()).sum();
    let mean_partials: Vec<f64> = rows.par_iter().map(|r| r.iter().sum::<f64>()).collect();
    let mean = mean_partials.iter().sum::<f64>() / n_total as f64;
    let central: Vec<(f64, f64, usize)> = rows
        .par_iter()
        .map(|r| {
            let mut m2 = 0.0;
            let mut m4 = 0.0;
            let mut nearz = 0usize;
            for &x in r {
                let c = x - mean;
                let c2 = c * c;
                m2 += c2;
                m4 += c2 * c2;
                if x.abs() < 1e-6 {
                    nearz += 1;
                }
            }
            (m2, m4, nearz)
        })
        .collect();
    let m2 = central.iter().map(|c| c.0).sum::<f64>();
    let m4 = central.iter().map(|c| c.1).sum::<f64>();
    let nearz: usize = central.iter().map(|c| c.2).sum();
    let std = (m2 / n_total as f64).sqrt();
    let kurt = if std > 0.0 {
        m4 * n_total as f64 / (m2 * m2) - 3.0
    } else {
        0.0
    };
    let nearz_frac = nearz as f64 / n_total as f64;
    let mut mags: Vec<f64> = rows
        .par_iter()
        .flat_map_iter(|r| r.iter().map(|x| x.abs()))
        .collect();
    mags.sort_by(|a, b| b.total_cmp(a));
    let energy: f64 = mags.iter().map(|m| m * m).sum();
    let top = ((n_total as f64) * 0.01).ceil() as usize;
    let top_energy: f64 = mags[..top.min(mags.len())].iter().map(|m| m * m).sum();
    let top_share = if energy > 0.0 {
        top_energy / energy
    } else {
        0.0
    };
    (mean, std, kurt, nearz_frac, top_share)
}

/// Top samples named per feature in the input map.
const TOP_TOKENS_PER_FEATURE: usize = 8;

/// The T1 artifact: `SNMFM1` canonical bytes — header + per-feature
/// support (neuron u32 + z f32-bits) + per-feature top samples (sample
/// u32 + y f32-bits) + BLAKE3 trailer over everything before. The layer,
/// k, d_a, n, lambda, wta and BOTH digests bind the identity; z/y ride
/// f32-as-bits (the RDMAP1 precedent) — precision loss disclosed by the
/// version byte.
#[allow(clippy::too_many_arguments)]
fn write_snmfm1(
    path: &std::path::Path,
    layer: usize,
    k: usize,
    d: usize,
    n: usize,
    lambda: f64,
    wta: usize,
    capture_digest: &str,
    corpus_digest: &str,
    z: &[f64],
    y: &[f64],
    top: &[Vec<(usize, f64)>],
) -> Result<String> {
    use blake3::Hasher;
    let mut b = Vec::new();
    b.extend_from_slice(b"SNMFM1\x01");
    b.extend_from_slice(&(layer as u32).to_le_bytes());
    b.extend_from_slice(&(k as u32).to_le_bytes());
    b.extend_from_slice(&(d as u32).to_le_bytes());
    b.extend_from_slice(&(n as u32).to_le_bytes());
    b.extend_from_slice(&lambda.to_bits().to_le_bytes());
    b.extend_from_slice(&(wta as u32).to_le_bytes());
    let cap = blake3::hash(capture_digest.as_bytes());
    b.extend_from_slice(cap.as_bytes());
    let cor = blake3::hash(corpus_digest.as_bytes());
    b.extend_from_slice(cor.as_bytes());
    for v in z {
        b.extend_from_slice(&(*v as f32).to_bits().to_le_bytes());
    }
    for v in y {
        b.extend_from_slice(&(*v as f32).to_bits().to_le_bytes());
    }
    for t in top {
        for (s, v) in t {
            b.extend_from_slice(&(*s as u32).to_le_bytes());
            b.extend_from_slice(&(*v as f32).to_bits().to_le_bytes());
        }
    }
    let mut h = Hasher::new();
    h.update(&b);
    let trailer = h.finalize();
    b.extend_from_slice(trailer.as_bytes());
    std::fs::write(path, &b).with_context(|| format!("write {}", path.display()))?;
    Ok(trailer.to_hex().to_string())
}

/// The 1250-T1-shape readability rule, pinned: a decoded token is WORD-ish
/// when its trimmed text is >= 3 chars and every char is ASCII
/// alphanumeric or one of ` -'`; a FEATURE reads readable when >= 1 of its
/// top-8 tokens is word-ish.
fn token_is_wordish(text: &str) -> bool {
    let t = text.trim();
    t.chars().count() >= 3
        && t.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == ' ' || c == '-' || c == '\'')
}

fn main() -> Result<()> {
    let args = parse_args()?;
    if !args.corpus.exists() {
        bail!(
            "corpus {} missing — pass --corpus (a reflex-canonical datasets dir or a plain text file)",
            args.corpus.display()
        );
    }

    // Body dispatch BEFORE any weight load (the sink_audit lesson): a
    // ternary GGUF refuses loud — the dense prefill's batched_mlp hook is
    // the capture seam's only write-site.
    let probe =
        GgufFile::open(&args.gguf).with_context(|| format!("open {}", args.gguf.display()))?;
    let is_ternary = probe
        .tensor_info("token_embd.weight")
        .map(|info| matches!(info.ggml_type, GgmlType::Q2_0 | GgmlType::PTQ1_0))
        .unwrap_or(false);
    if is_ternary {
        bail!(
            "ternary GGUF: the dense prefill's batched_mlp hook is the capture seam's only \
             write-site (issue 1252's dispatch law) — refusing rather than mis-shaping the file"
        );
    }

    // Corpus + tokenizer.
    let corpus_text = load_corpus_text(&args.corpus)
        .with_context(|| format!("load corpus {}", args.corpus.display()))?;
    let corpus_digest = blake3::hash(corpus_text.as_bytes()).to_hex().to_string();
    let tokens = {
        let tok = BpeTokenizer::from_gguf(&probe).context("BPE tokenizer from gguf")?;
        tok.encode(&corpus_text)
    };
    if tokens.len() < args.window {
        bail!(
            "corpus encodes {} tokens < the window {} — grow the corpus or lower --tokens",
            tokens.len(),
            args.window
        );
    }
    let seq: Vec<usize> = tokens[..args.window].to_vec();

    // Model.
    let t0 = std::time::Instant::now();
    let (config, weights) = load_qwen_deltanet_weights_gguf(&args.gguf)
        .with_context(|| format!("load {}", args.gguf.display()))?;
    let mlp = config.mlp_hidden;
    let n_layers = weights.layers.len();
    println!(
        "# model: qwen35 hybrid | layers={n_layers} n_embd={} mlp={mlp} | load {:.1}s",
        config.n_embd,
        t0.elapsed().as_secs_f32()
    );

    // Representative layers: the first DeltaNet MLP + the first Attention MLP.
    let first_dn = weights
        .layer_types
        .iter()
        .position(|t| *t == DeltaNetLayerType::DeltaNet)
        .context("no DeltaNet layer in the fixture")?;
    let first_at = weights
        .layer_types
        .iter()
        .position(|t| *t == DeltaNetLayerType::Attention)
        .context("no Attention layer in the fixture")?;
    let dn_count = weights
        .layer_types
        .iter()
        .filter(|t| **t == DeltaNetLayerType::DeltaNet)
        .count();
    println!(
        "# mixer map: {dn_count} DeltaNet + {} Attention layers; SNMF representatives: layer {first_dn} (DeltaNet) + layer {first_at} (Attention)",
        n_layers - dn_count
    );

    // Phase A: capture — one prefill, ALL layers.
    let layers_all: Vec<usize> = (0..n_layers).collect();
    let mut cache = HybridCache::with_layer_types(&config, &weights.layer_types);
    let mut scratch = HybridForwardScratch::new(&config);
    let mut prefill_ctx = PrefillContext::new(&config, args.window);
    prefill_ctx.closed_form_capture = Some(CaptureSink::new(layers_all.clone()));
    let rope = RopeFreqTable::new(config.rope_theta, effective_rotary_dim(&config));
    let mut logits = vec![0.0f32; config.vocab_size];
    let t1 = std::time::Instant::now();
    prefill_qwen_deltanet_into(
        &weights,
        &config,
        &mut cache,
        &seq,
        &mut scratch,
        &rope,
        &mut prefill_ctx,
        &mut logits,
        None,
    );
    let sink = prefill_ctx
        .closed_form_capture
        .take()
        .context("capture sink vanished")?;
    assert_eq!(sink.rows, args.window, "capture rows must equal the window");
    assert_eq!(sink.mlp, mlp, "capture mlp width must equal config");
    println!(
        "# capture: {:.1}s ({} layers × {} positions × {} units)",
        t1.elapsed().as_secs_f32(),
        sink.layers.len(),
        sink.rows,
        sink.mlp
    );

    // Per-layer captures (f64 rows + digest). Sink keys are [rows × mlp]
    // row-major in `layers` order.
    let mut captures: Vec<LayerCapture> = Vec::with_capacity(n_layers);
    for (li, &layer) in layers_all.iter().enumerate() {
        let keys = &sink.keys[li];
        let f32_rows: Vec<Vec<f32>> = (0..sink.rows)
            .map(|p| keys[p * sink.mlp..(p + 1) * sink.mlp].to_vec())
            .collect();
        let digest = capture_digest_of(&f32_rows, sink.mlp);
        let rows: Vec<Vec<f64>> = f32_rows
            .into_iter()
            .map(|r| r.into_iter().map(|x| x as f64).collect())
            .collect();
        let mixer = if weights.layer_types[layer] == DeltaNetLayerType::DeltaNet {
            "deltanet"
        } else {
            "attention"
        };
        captures.push(LayerCapture {
            mixer,
            rows,
            capture_digest: digest,
        });
    }

    // Phase B: activation statistics + the mixer-type rollup.
    println!("# ── activation statistics (per layer) ──");
    println!("# layer mixer mean std kurtosis near_zero top1pct_energy");
    let stat_names = ["mean", "std", "kurtosis", "near_zero", "top1pct_energy"];
    // Per-mixer per-stat values (the rollup folds them into mean + median;
    // kurtosis is single-outlier-dominated, so the verdict reads the median).
    let mut dn_vals: Vec<Vec<f64>> = vec![Vec::new(); 5];
    let mut at_vals: Vec<Vec<f64>> = vec![Vec::new(); 5];
    for (li, c) in captures.iter().enumerate() {
        let (mean, std, kurt, nearz, top) = act_stats(&c.rows);
        println!(
            "# {li:>5} {:>9} {:+.4} {:.4} {:+.3} {:.4} {:.4}",
            c.mixer, mean, std, kurt, nearz, top
        );
        let bucket = if c.mixer == "deltanet" {
            &mut dn_vals
        } else {
            &mut at_vals
        };
        for (acc, v) in bucket.iter_mut().zip([mean, std, kurt, nearz, top]) {
            acc.push(v);
        }
    }
    let stat_mean = |vals: &[Vec<f64>], i: usize| -> f64 {
        let v = &vals[i];
        if v.is_empty() {
            0.0
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        }
    };
    // Median = sorted[n/2] (the lower median, a pinned order statistic —
    // never interpolated).
    let stat_median = |vals: &[Vec<f64>], i: usize| -> f64 {
        let mut v = vals[i].clone();
        if v.is_empty() {
            0.0
        } else {
            v.sort_by(|a, b| a.total_cmp(b));
            v[v.len() / 2]
        }
    };
    println!("# ── mixer rollup (mean of per-layer stats) ──");
    let mut shifted: Vec<&str> = Vec::new();
    let gstd = (stat_mean(&dn_vals, 1) + stat_mean(&at_vals, 1)) / 2.0;
    for (i, name) in stat_names.iter().enumerate() {
        let dvm = stat_mean(&dn_vals, i);
        let avm = stat_mean(&at_vals, i);
        let dvt = stat_median(&dn_vals, i);
        let avt = stat_median(&at_vals, i);
        // A statistic vanishing for BOTH mixers (< 1% of the mean std)
        // carries no diagnostic ratio — flagging it would read noise.
        let negligible = dvm.abs() < 0.01 * gstd && avm.abs() < 0.01 * gstd;
        if negligible {
            println!(
                "# {name:>15}: mean dn {dvm:+.4} at {avm:+.4} | median dn {dvt:+.4} at {avt:+.4} | negligible for both mixers (ratio not diagnostic)"
            );
            continue;
        }
        let ratio = if dvt.abs() > 1e-30 {
            avt / dvt
        } else {
            f64::INFINITY
        };
        println!(
            "# {name:>15}: mean dn {dvm:+.4} at {avm:+.4} | median dn {dvt:+.4} at {avt:+.4} | attn/dn median {ratio:.3}"
        );
        if !(0.5..=2.0).contains(&ratio) {
            shifted.push(name);
        }
    }
    // Single-outlier naming: a layer whose kurtosis exceeds 10× its own
    // mixer's median kurtosis — the layer the rollup's MEAN would read.
    let dn_med_k = stat_median(&dn_vals, 2);
    let at_med_k = stat_median(&at_vals, 2);
    for (li, c) in captures.iter().enumerate() {
        let med = if c.mixer == "deltanet" {
            dn_med_k
        } else {
            at_med_k
        };
        let (_, _, kurt, _, _) = act_stats(&c.rows);
        if med > 0.0 && kurt > 10.0 * med {
            println!(
                "# outlier layer {li} ({}): kurtosis {kurt:.1} > 10× its mixer median {med:.1} — the mean rollup reads THIS layer",
                c.mixer
            );
        }
    }
    // The T0 mixer verdict: a "material shift" is any statistic whose
    // attention/deltanet ratio leaves [0.5, 2.0] (order-of-magnitude
    // guard, fixture-tier — the measured answer, never an assumption).
    if shifted.is_empty() {
        println!(
            "# mixer verdict: every statistic within [0.5, 2.0]× — no material shift detected at the fixture"
        );
    } else {
        println!(
            "# mixer verdict: MATERIAL SHIFT in {} (attention/deltanet ratio outside [0.5, 2.0]) — the paper's dense-SwiGLU statistics do not transfer unexamined",
            shifted.join(", ")
        );
    }

    // Phases C+D+E+F: SNMF on the two representative layers + determinism
    // + the T1 input map + artifact.
    let reps = [first_dn, first_at];
    let mut receipt = String::new();
    let tok = BpeTokenizer::from_gguf(&probe).context("BPE tokenizer from gguf (T1)")?;
    let decoded: Vec<String> = seq.iter().map(|&t| tok.decode(&[t])).collect();
    println!(
        "# T1: {} window samples named (corpus decode ready)",
        decoded.len()
    );
    receipt.push_str(&format!(
        "# Issue 1255 T0 — SNMF group-feature fixture cell receipt\n\n\
         - model: {} (layers={n_layers} n_embd={} mlp={mlp})\n\
         - corpus: {} (digest {corpus_digest})\n\
         - window: {} token samples | k={} iters<={} lambda={} wta={}%\n\
         - capture seam: closed_form_write CaptureSink (down-proj input rows; one prefill, all layers)\n\n",
        args.gguf.display(),
        config.n_embd,
        args.corpus.display(),
        args.window,
        args.k,
        args.iters,
        args.lambda,
        args.wta_pct,
    ));
    let mut lambda_rider = String::new();
    for &layer in &reps {
        let c = &captures[layer];
        let seed = u64::from_le_bytes(
            blake3::hash(c.capture_digest.as_bytes()).as_bytes()[..8].try_into()?,
        );
        println!("# ── SNMF: layer {layer} ({}) ──", c.mixer);
        println!("# capture digest {} | seed {seed:#018x}", c.capture_digest);
        let mut snmf = Snmf::new(&c.rows, args.k, args.lambda, args.wta_pct, seed);
        let t2 = std::time::Instant::now();
        let trace = snmf.run(args.iters);
        let snmf_secs = t2.elapsed().as_secs_f32();
        let rel = snmf.rel_residual();
        let explained = 1.0 - rel;
        let digest = snmf.state_digest();

        // Per-feature neuron support census.
        let k = snmf.k;
        let d = snmf.d;
        let mut union = std::collections::HashSet::new();
        let mut per_neuron = vec![0usize; d];
        for j in 0..k {
            for (r, cnt) in per_neuron.iter_mut().enumerate() {
                if snmf.z[r * k + j] != 0.0 {
                    union.insert(r);
                    *cnt += 1;
                }
            }
        }
        let max_overlap = per_neuron.iter().copied().max().unwrap_or(0);
        let mean_overlap = per_neuron.iter().sum::<usize>() as f64 / union.len().max(1) as f64;
        let mut support_sample = String::new();
        for j in 0..3.min(k) {
            let mut col: Vec<(f64, usize)> = (0..d).map(|r| (snmf.z[r * k + j].abs(), r)).collect();
            col.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            let tops: Vec<String> = col[..5]
                .iter()
                .map(|(m, r)| format!("{r}({m:.3})"))
                .collect();
            support_sample.push_str(&format!("  - feature {j}: {}\n", tops.join(" ")));
        }

        let iters_run = trace.last().map(|(i, _)| i + 1).unwrap_or(1);
        println!(
            "# explained-norm {explained:.4} (rel residual {rel:.4}) | iters {iters_run} in {snmf_secs:.1}s | support {}/feature | union {} neurons | max/mean features-per-neuron {max_overlap}/{mean_overlap:.2}",
            snmf.wta,
            union.len()
        );
        println!("# support sample (top-5 neurons per feature, |z|):\n{support_sample}");

        // Phase E: determinism tripwire — same capture + seed re-run.
        let mut snmf2 = Snmf::new(&c.rows, args.k, args.lambda, args.wta_pct, seed);
        snmf2.run(args.iters);
        let det = digest == snmf2.state_digest();
        println!(
            "# determinism: {}",
            if det {
                "IDENTICAL (bit-equal re-run)"
            } else {
                "REFUSE — re-run diverged"
            }
        );
        if !det {
            bail!("determinism tripwire failed on layer {layer}");
        }

        // Lambda-sensitivity rider (first representative only): is the
        // ridge binding on the reconstruction receipt?
        if layer == reps[0] {
            let mut snmf_lo = Snmf::new(&c.rows, args.k, args.lambda / 100.0, args.wta_pct, seed);
            let _trace_lo = snmf_lo.run(args.iters);
            let rel_lo = snmf_lo.rel_residual();
            let mut snmf_hi = Snmf::new(&c.rows, args.k * 2, args.lambda, args.wta_pct, seed);
            let _trace_hi = snmf_hi.run(args.iters);
            let rel_hi = snmf_hi.rel_residual();
            let line = format!(
                "  - lambda sensitivity: explained-norm at lambda={} -> {:.4} (vs {:.4} at lambda={})\n",
                args.lambda / 100.0,
                1.0 - rel_lo,
                explained,
                args.lambda
            );
            lambda_rider.push_str(&line);
            let line = format!(
                "  - k sensitivity: explained-norm at k={} -> {:.4} (vs {:.4} at k={})\n",
                args.k * 2,
                1.0 - rel_hi,
                explained,
                args.k
            );
            lambda_rider.push_str(&line);
            println!(
                "# lambda sensitivity: explained-norm at lambda={} -> {:.4} (vs {explained:.4} at lambda={})",
                args.lambda / 100.0,
                1.0 - rel_lo,
                args.lambda
            );
            println!(
                "# k sensitivity: explained-norm at k={} -> {:.4} (vs {explained:.4} at k={})",
                args.k * 2,
                1.0 - rel_hi,
                args.k
            );
        }

        // ── T1: the input map — Y's per-feature top tokens ──
        let k = snmf.k;
        let n = snmf.n;
        let mut top_samples: Vec<Vec<(usize, f64)>> = Vec::with_capacity(k);
        let mut input_map_lines = String::new();
        let mut readable_features = 0usize;
        let mut y_energy: Vec<(f64, usize)> = Vec::with_capacity(k);
        for j in 0..k {
            let row = &snmf.y[j * n..(j + 1) * n];
            let mut ord: Vec<(f64, usize)> = row
                .iter()
                .copied()
                .enumerate()
                .map(|(s, v)| (v, s))
                .collect();
            ord.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            let tops: Vec<(usize, f64)> = ord
                .iter()
                .take(TOP_TOKENS_PER_FEATURE)
                .map(|&(v, s)| (s, v))
                .collect();
            y_energy.push((row.iter().sum::<f64>(), j));
            let toks: Vec<String> = tops
                .iter()
                .map(|&(s, _)| decoded[s].trim().to_string())
                .collect();
            let n_wordish = toks.iter().filter(|t| token_is_wordish(t)).count();
            if n_wordish > 0 {
                readable_features += 1;
            }
            input_map_lines.push_str(&format!(
                "  - feature {j}: {} (wordish {n_wordish}/{})\n",
                toks.join(" | "),
                toks.len()
            ));
            top_samples.push(tops);
        }
        // Per-band rollup over Y-energy quartiles (sorted by row energy,
        // quartile membership by index; band 0 = highest energy).
        y_energy.sort_by(|a, b| b.0.total_cmp(&a.0));
        let q = (k / 4).max(1);
        let mut band_readable = [0usize; 4];
        let mut band_wordish_tokens = [0usize; 4];
        for (rank, &(energy, j)) in y_energy.iter().enumerate() {
            let band = (rank / q).min(3);
            let _ = energy;
            let row_tops = &top_samples[j];
            let n_wordish = row_tops
                .iter()
                .map(|&(s, _)| usize::from(token_is_wordish(decoded[s].trim())))
                .sum::<usize>();
            band_wordish_tokens[band] += n_wordish;
            if n_wordish > 0 {
                band_readable[band] += 1;
            }
        }
        println!(
            "# T1 input map: {readable_features}/{k} features readable (>=1 wordish top token) | band readable [Q0..Q3] = {band_readable:?}/{k} per band of {q} | band wordish tokens {band_wordish_tokens:?}"
        );
        for line in input_map_lines.lines().take(6) {
            println!("# {line}");
        }

        // ── T1 artifact: SNMFM1 ──
        std::fs::create_dir_all(&args.out)
            .with_context(|| format!("mkdir {}", args.out.display()))?;
        let art_path = args.out.join(format!("snmfm1_layer{layer}.bin"));
        let art_digest = write_snmfm1(
            &art_path,
            layer,
            k,
            snmf.d,
            n,
            args.lambda,
            snmf.wta,
            &c.capture_digest,
            &corpus_digest,
            &snmf.z,
            &snmf.y,
            &top_samples,
        )?;
        println!("# SNMFM1: {} ({})", art_path.display(), art_digest);

        let (first_it, first_r) = trace.first().copied().unwrap_or((0, f64::NAN));
        let (last_it, last_r) = trace.last().copied().unwrap_or((0, f64::NAN));
        receipt.push_str(&format!(
            "## layer {layer} ({mix})\n\n\
             - capture digest: `{cd}` (SplitMix64 seed {seed:#018x})\n\
             - explained-norm (SPARSE ZY): **{explained:.4}** (rel residual {rel:.4})\n\
             - objective trace: it {first_it} → {last_it} ({first_r:.4} → {last_r:.4})\n\
             - support: {} neurons/feature | union {} | features-per-neuron max {max_overlap} mean {mean_overlap:.2}\n\
             - determinism: bit-identical re-run ({det})\n\n\
             - readable features: {readable_features}/{k} | band readable [Q0..Q3] = {band_readable:?}\n\
             - artifact: SNMFM1 {art_digest}\n\n\
             ### support sample\n{support_sample}\n\
             ### input map (top-8 samples per feature)\n{input_map_lines}\n",
            snmf.wta,
            union.len(),
            mix = c.mixer,
            cd = c.capture_digest,
        ));
    }

    if !lambda_rider.is_empty() {
        receipt.push_str("## lambda sensitivity\n\n");
        receipt.push_str(&lambda_rider);
    }

    // Receipt md.
    std::fs::create_dir_all(&args.out).with_context(|| format!("mkdir {}", args.out.display()))?;
    let receipt_path = args.out.join("receipt.md");
    std::fs::write(&receipt_path, &receipt)
        .with_context(|| format!("write {}", receipt_path.display()))?;
    println!("# receipt: {}", receipt_path.display());
    Ok(())
}
