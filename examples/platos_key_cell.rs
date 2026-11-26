//! `platos_key_cell` — the Issue-1254 PoC harness: the Plato's-Key
//! cross-model key-transfer cell (closed-form orthogonal-Procrustes rotation
//! + one spread-match scale, frozen on paired-recall facts, applied as a
//!   single hidden-state REPLACE at the recipient's subject-token layer).
//!
//! Design of record: the sibling `.issues/1254_platos_key_cross_model_transfer_cell.md`
//! (amended 2026-11-26, delegated-verdict session `5b062d19`: recipient of
//! record = Llama-3.2-3B BASE, donor = Ternary-Bonsai-2.5-27B, adjudication on
//! the in-range pair only); distillation: the sibling `.research/044` note. The
//! blog's numbers are the paper's, never reproduced here — this harness
//! measures THIS stack.
//!
//! Modes
//! ─────
//! **`--self-test`** (no models): a synthetic protocol dry-run. Two fake
//! "models" are constructed so that a PLANTED orthogonal map relates their
//! key distributions and a linear readout plays recall; the full pipeline —
//! randomized PCA → one-sided Jacobi Procrustes (f64, pinned tie rule) →
//! spread-match scale → controls → G1 gate arithmetic — must recover the
//! planted map and report PASS. This validates the machinery and the gate
//! arithmetic; it is a protocol dry-run and makes NO science claim (the
//! synthetic truncation is benign by construction — the real cell's r=256
//! truncation meets real geometry).
//!
//! **Real mode** (`--recipient … --donor … --facts …`): the pre-registered
//! cell on the real checkpoints. This commit lands the recipient side end to
//! end (P0 load + rope disposition, P1 real-model G3 identity, P2 baseline
//! recall gate) and REFUSES LOUD at the donor step (P3): the ternary-path
//! capture seam is its own G3-gated commit, and a half-armed cell never
//! prints a transfer number.
//!
//! ```bash
//! # the machinery self-test (runs on any box, no model files):
//! cargo run --release --example platos_key_cell -- --self-test
//!
//! # the real cell (rig posture; refuses loud at P3 until the donor seam lands):
//! cargo run --release --example platos_key_cell -- \
//!     --recipient /path/Llama-3.2-3B-f16.gguf \
//!     --donor /path/Ternary-Bonsai-2.5-27B-PQ2_0.gguf \
//!     --facts examples/facts/platos_key_facts_sample.json \
//!     --out .platos_key_cell/
//! ```
//!
//! Exit codes: 0 = ran to completion (verdict text inside the report);
//! 2 = config/env refusal; 3 = donor capture seam pending (P3, by design);
//! 4 = baseline recall gate failed (no transfer number is ever printed past
//! a failed baseline — the verdict round-1 addition).
//!
//! ⛔ Not a benchmark: every timing printed is wall-clock on a possibly
//! loaded box — a correctness harness, never a perf claim.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};

use katgpt_transformer::MultiLayerKVCache;
use riir_infer_core::gguf_loader::{GgufFile, load_llama_weights_gguf};
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_core::transformer::{
    ForwardContext, NoHook, PostLayerHook, forward_llama, forward_llama_hooked,
};

/// Greedy answer window (the i1189 recall metric's shape).
const RECALL_FIRST_K: usize = 8;
/// Distinct-token floor inside the window (a repetition run is not recall).
const RECALL_DISTINCT_RATIO: f64 = 0.4;

// ── facts input ─────────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
struct FactsDoc {
    facts: Vec<FactDoc>,
}

#[derive(serde::Deserialize)]
struct FactDoc {
    /// The subject span, VERBATIM inside `prompt` (the subject-token
    /// boundary is located by `prompt.find(&subject)` — a subject the
    /// prompt does not contain exactly is an authoring error, refused).
    subject: String,
    /// The full direct prompt, subject embedded, ending in a template
    /// suffix ("The capital of France is").
    prompt: String,
    /// The answer the recall check greps for (exact substring of the
    /// first [`RECALL_FIRST_K`] generated tokens' decode).
    answer: String,
}

fn load_facts(path: &Path) -> Result<Vec<FactDoc>> {
    let raw =
        std::fs::read_to_string(path).with_context(|| format!("facts file {}", path.display()))?;
    let doc: FactsDoc = serde_json::from_str(&raw)
        .with_context(|| format!("facts schema in {}", path.display()))?;
    if doc.facts.is_empty() {
        bail!("facts file {} holds zero facts", path.display());
    }
    for (i, f) in doc.facts.iter().enumerate() {
        if f.subject.is_empty() || !f.prompt.contains(&f.subject) {
            bail!(
                "fact {i}: subject {:?} must appear verbatim inside the prompt",
                f.subject
            );
        }
        if f.answer.is_empty() {
            bail!("fact {i}: empty answer");
        }
        // The subject must NOT reach the prompt's tail — the template suffix
        // after it is what the swapped state generates into.
        let end = f.prompt.find(&f.subject).unwrap() + f.subject.len();
        if end == f.prompt.len() {
            bail!("fact {i}: subject ends the prompt — a template suffix is required");
        }
    }
    Ok(doc.facts)
}

// ── deterministic RNG (SplitMix64) + seeded orthogonal matrices ─────────

struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    /// Uniform [0, 1) in f64.
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Centered, roughly Gaussian (Irwin–Hall sum of 4 uniforms − 2).
    fn gauss(&mut self) -> f64 {
        (0..4).map(|_| self.next_f64()).sum::<f64>() - 2.0
    }
    fn gauss_vec(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.gauss()).collect()
    }
}

/// Gram–Schmidt orthonormalize the COLUMNS of a row-major matrix.
fn orthonormalize_columns(cols: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut out: Vec<Vec<f64>> = Vec::with_capacity(cols.len());
    for c in cols {
        let mut v = c;
        for b in &out {
            let dot: f64 = v.iter().zip(b).map(|(x, y)| x * y).sum();
            for (x, y) in v.iter_mut().zip(b) {
                *x -= dot * y;
            }
        }
        let n: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if n > 1e-10 {
            for x in &mut v {
                *x /= n;
            }
            out.push(v);
        }
    }
    out
}

/// A seeded random orthogonal matrix (columns Gram–Schmidt'd from a
/// deterministic Gaussian draw). Called Haar-like in the reports — it is a
/// seeded random orthogonal control, not a verified Haar sample.
fn random_orthogonal(n: usize, rng: &mut SplitMix64) -> Vec<Vec<f64>> {
    let cols: Vec<Vec<f64>> = (0..n).map(|_| rng.gauss_vec(n)).collect();
    orthonormalize_columns(cols)
}

// ── the f64 fit machinery ───────────────────────────────────────────────

/// A fitted PC space: training mean + `r` orthonormal components (f64).
#[derive(Clone)]
struct PcSpace {
    mean: Vec<f64>,
    /// `components[i]` = the i-th component, `d` long.
    components: Vec<Vec<f64>>,
}

impl PcSpace {
    fn project(&self, x: &[f64]) -> Vec<f64> {
        let centered: Vec<f64> = x.iter().zip(&self.mean).map(|(a, b)| a - b).collect();
        self.components
            .iter()
            .map(|c| c.iter().zip(&centered).map(|(a, b)| a * b).sum())
            .collect()
    }
    /// Back to data space with the TRAINING mean (the issue's law: the
    /// recipient reconstructs with its own PCs + training mean).
    fn reconstruct(&self, z: &[f64]) -> Vec<f64> {
        let mut x = self.mean.clone();
        for (zi, c) in z.iter().zip(&self.components) {
            for (xi, ci) in x.iter_mut().zip(c) {
                *xi += zi * ci;
            }
        }
        x
    }
}

/// Deterministic randomized PCA (Halko): Y = XΩ → QR → B = QᵀX → exact
/// Jacobi eigenpairs of BBᵀ → top-r back-projected components.
///
/// Oversampling is fixed (+32) and Ω is a seeded Gaussian draw, so the
/// whole fit is a deterministic function of (samples, r, seed).
fn fit_pca(samples: &[Vec<f64>], r: usize, seed: u64) -> PcSpace {
    let n = samples.len();
    let d = samples[0].len();
    assert!(n >= 2, "PCA needs ≥ 2 samples");
    assert!(r <= d.min(n), "r {} must be ≤ min(n {}, d {})", r, n, d);
    let t0 = Instant::now();

    let mut rng = SplitMix64::new(seed);
    let l = (r + 32).min(d);
    // Ω (d × l), column-major for the product below.
    let omega: Vec<Vec<f64>> = (0..l).map(|_| rng.gauss_vec(d)).collect();

    // Y = X Ω  (n × l)
    let mut y = vec![vec![0.0f64; l]; n];
    for (row, s) in samples.iter().enumerate() {
        for (col, oc) in omega.iter().enumerate() {
            y[row][col] = s.iter().zip(oc).map(|(a, b)| a * b).sum();
        }
    }
    // QR of Y's COLUMNS (l columns, each n long), with a re-orthogonalization
    // pass — twice-modified Gram–Schmidt; at l ≤ r+32 the tail error is far
    // under the fit's needs.
    let y_cols: Vec<Vec<f64>> = (0..l)
        .map(|c| (0..n).map(|r_| y[r_][c]).collect())
        .collect();
    let q_cols = orthonormalize_columns(orthonormalize_columns(y_cols));
    assert!(
        q_cols.len() >= r,
        "PCA QR produced {} columns for r={r} (rank-deficient draw)",
        q_cols.len()
    );

    // B = Qᵀ X  (l × d)
    let mut b = vec![vec![0.0f64; d]; l];
    for (bi, qc) in b.iter_mut().zip(&q_cols) {
        for (row, s) in samples.iter().enumerate() {
            let qv = qc[row];
            if qv != 0.0 {
                for (bj, sx) in bi.iter_mut().zip(s) {
                    *bj += qv * sx;
                }
            }
        }
    }

    // G = B Bᵀ (l × l, symmetric) — exact cyclic Jacobi eigendecomposition.
    let g = symmetric_gram(&b);
    let (eigvals, eigvecs) = jacobi_eigen(&g);

    // mean
    let mut mean = vec![0.0f64; d];
    for s in samples {
        for (mi, si) in mean.iter_mut().zip(s) {
            *mi += si;
        }
    }
    for mi in &mut mean {
        *mi /= n as f64;
    }

    // Top-r eigenpairs (descending), back-projected through Bᵀ.
    let mut order: Vec<usize> = (0..eigvals.len()).collect();
    order.sort_by(|a, b| eigvals[*b].partial_cmp(&eigvals[*a]).unwrap());
    let mut components = Vec::with_capacity(r);
    for &idx in order.iter().take(r) {
        let u = &eigvecs[idx]; // length l
        let mut v = vec![0.0f64; d];
        for (ui, brow) in u.iter().zip(&b) {
            if *ui != 0.0 {
                for (vj, bv) in v.iter_mut().zip(brow) {
                    *vj += ui * bv;
                }
            }
        }
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-12 {
            for x in &mut v {
                *x /= norm;
            }
            components.push(v);
        }
    }
    assert!(
        components.len() == r,
        "PCA produced {} components, wanted {r}",
        components.len()
    );
    eprintln!(
        "    pca: {}×{} → r={r}, top λ {:.3e}..{:.3e} in {:.1?}",
        n,
        d,
        eigvals[order[0]],
        eigvals[*order.last().unwrap()],
        t0.elapsed()
    );
    PcSpace { mean, components }
}

/// B·Bᵀ for row-major `b` (l × d) — symmetric l × l.
fn symmetric_gram(b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let l = b.len();
    let mut g = vec![vec![0.0f64; l]; l];
    for i in 0..l {
        for j in i..l {
            let dot: f64 = b[i].iter().zip(&b[j]).map(|(x, y)| x * y).sum();
            g[i][j] = dot;
            g[j][i] = dot;
        }
    }
    g
}

/// Classic cyclic Jacobi eigendecomposition of a symmetric matrix
/// (deterministic: fixed sweep order, convergence at a relative off-diagonal
/// threshold, hard sweep cap). Returns (eigenvalues, eigenvectors-as-rows).
fn jacobi_eigen(a_in: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a_in.len();
    let mut a = a_in.to_vec();
    let mut v = vec![vec![0.0f64; n]; n];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    let scale: f64 = a
        .iter()
        .flat_map(|r| r.iter())
        .map(|x| x.abs())
        .sum::<f64>()
        .max(1e-300);

    for _sweep in 0..60 {
        // Matrix-index form throughout — the sweep order IS the pinned tie
        // rule; iterator forms would obscure it.
        let off: f64 = {
            let mut off = 0.0;
            #[allow(clippy::needless_range_loop)]
            for i in 0..n {
                for j in i + 1..n {
                    off += a[i][j] * a[i][j];
                }
            }
            off
        };
        if off.sqrt() < 1e-14 * scale {
            break;
        }
        for i in 0..n {
            for j in i + 1..n {
                if a[i][j].abs() < 1e-300 {
                    continue;
                }
                // Deterministic rotation (no branch on comparison signs beyond
                // the atan2 — the tie law: the angle is a pure function).
                // ⚠ atan2(Y, X): Y = 2·a_ij, X = a_jj − a_ii — tan(2θ) =
                // 2a_ij/(a_jj − a_ii) zeroes the pivot (the swapped order
                // computes the cotangent angle and never diagonalizes).
                let theta = 0.5 * (a[i][j] * 2.0).atan2(a[j][j] - a[i][i]);
                let (s, c) = theta.sin_cos();
                #[allow(clippy::needless_range_loop)]
                for k in 0..n {
                    let aik = a[i][k];
                    let ajk = a[j][k];
                    a[i][k] = c * aik - s * ajk;
                    a[j][k] = s * aik + c * ajk;
                }
                #[allow(clippy::needless_range_loop)]
                for k in 0..n {
                    let aki = a[k][i];
                    let akj = a[k][j];
                    a[k][i] = c * aki - s * akj;
                    a[k][j] = s * aki + c * akj;
                }
                #[allow(clippy::needless_range_loop)]
                for k in 0..n {
                    let vki = v[k][i];
                    let vkj = v[k][j];
                    v[k][i] = c * vki - s * vkj;
                    v[k][j] = s * vki + c * vkj;
                }
            }
        }
    }

    let eigvals = (0..n).map(|i| a[i][i]).collect();
    let eigvecs: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|k| v[k][i]).collect()).collect();
    (eigvals, eigvecs)
}

/// Orthogonal Procrustes: `R = argmin‖A·R − B‖_F`, both n×r — solved by the
/// pre-registered one-sided Jacobi SVD of M = AᵀB (f64 internals, the
/// `issue_925_procrustes_probe.rs` precedent; tie rule pinned: lexicographic
/// (i, j) sweep order, the angle is a pure f64 function, hard sweep cap).
fn procrustes(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let r = a[0].len();
    assert_eq!(b.len(), n, "paired row counts");
    assert_eq!(b[0].len(), r, "paired widths");

    // M = Aᵀ B  (r × r) — the Procrustes cross-product. The solver returns
    // the COLUMN-convention rotation (z_rec ≈ R·z_don, the shape
    // `SwapMap::apply` consumes): with C = AᵀB = ŨΣṼᵀ, min‖A·R − B‖_F has
    // its row-convention optimum at ŨṼᵀ, and the column convention inverts
    // it — R = Ṽ·Ũᵀ.
    let mut m = vec![vec![0.0f64; r]; r];
    for (ar, br) in a.iter().zip(b) {
        for (i, av) in ar.iter().enumerate() {
            for (j, bv) in br.iter().enumerate() {
                m[i][j] += av * bv;
            }
        }
    }

    // The SVD of C via the exact Jacobi eigendecomposition of CᵀC
    // (r × r symmetric PSD): C = AᵀB = ŨΣṀᵀ with Ṁ = eigenvectors of
    // CᵀC (as columns, descending λ), Ũ = C·Ṁ·Σ⁻¹. Deterministic: the
    // cyclic Jacobi sweep order is fixed and the sort is a strict total
    // order (descending eigenvalue, ties to the lower index — the pinned
    // tie rule).
    let g = symmetric_gram_t(&m);
    let (eigvals, eigvec_rows) = jacobi_eigen(&g);
    let mut order: Vec<usize> = (0..eigvals.len()).collect();
    order.sort_by(|x, y| {
        eigvals[*y]
            .partial_cmp(&eigvals[*x])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(x.cmp(y))
    });

    // V₁ = eigenvectors as COLUMNS: v1[k][col].
    let mut v1 = vec![vec![0.0f64; r]; r];
    for (col, &idx) in order.iter().enumerate() {
        for k in 0..r {
            v1[k][col] = eigvec_rows[idx][k];
        }
    }

    // Σ (descending), clamped at zero — CᵀC is PSD but rounding can read a
    // tiny negative on a near-null eigenvalue.
    let sig: Vec<f64> = order
        .iter()
        .map(|&idx| eigvals[idx].max(0.0).sqrt())
        .collect();
    assert!(
        sig[0] > 1e-12,
        "procrustes: degenerate cross-product (largest singular value {:.3e})",
        sig[0]
    );

    // U₁ = C·V₁·Σ⁻¹, columns normalized (the Σ⁻¹ fold already normalizes;
    // the explicit pass guards the tail).
    let mut u1 = vec![vec![0.0f64; r]; r];
    for (k, m_k) in m.iter().enumerate() {
        for (col, sig_v) in sig.iter().enumerate() {
            if *sig_v > 1e-300 {
                u1[k][col] = m_k
                    .iter()
                    .zip(&v1)
                    .map(|(cv, vrow)| cv * vrow[col])
                    .sum::<f64>()
                    / sig_v;
            }
        }
    }

    // Row-convention optimum: R_row = U₁·V₁ᵀ. Column convention (what
    // `SwapMap::apply` consumes, z_rec ≈ R·z_don): R = (U₁·V₁ᵀ)ᵀ = V₁·U₁ᵀ
    // — rot[i][j] = Σ_k V₁[i][k]·U₁[j][k].
    let mut rot = vec![vec![0.0f64; r]; r];
    for i in 0..r {
        for j in 0..r {
            rot[i][j] = (0..r).map(|k| v1[i][k] * u1[j][k]).sum();
        }
    }
    rot
}

/// CᵀC for row-major `c` (r × r) — symmetric PSD r × r.
fn symmetric_gram_t(c: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let r = c.len();
    let mut g = vec![vec![0.0f64; r]; r];
    for i in 0..r {
        for j in i..r {
            let dot: f64 = (0..r).map(|k| c[k][i] * c[k][j]).sum();
            g[i][j] = dot;
            g[j][i] = dot;
        }
    }
    g
}

/// Matrix–vector: row-major `m` × `x`.
fn mat_vec(m: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    m.iter()
        .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
        .collect()
}

// ── the swap pipeline (one path for every arm) ──────────────────────────

struct SwapMap {
    /// Recipient PCs + training mean (reconstruction side).
    pcs: PcSpace,
    /// Donor PCs + training mean (projection side).
    pcs_donor: PcSpace,
    /// The r×r rotation in PC coordinates.
    rot: Vec<Vec<f64>>,
    /// The global scale.
    scale: f64,
}

impl SwapMap {
    /// donor key → recipient-space state (the exact map the hook applies,
    /// here in f64 fit coordinates; the serving apply is the f32 matvec).
    fn apply(&self, donor_key: &[f64]) -> Vec<f64> {
        let z: Vec<f64> = self.pcs_donor.project(donor_key);
        let zr: Vec<f64> = mat_vec(&self.rot, &z);
        let zs: Vec<f64> = zr.iter().map(|v| v * self.scale).collect();
        self.pcs.reconstruct(&zs)
    }

    /// The self-swap ceiling: the recipient's own key through the SAME
    /// machinery path — project→rotate→reconstruct with the recipient's OWN
    /// PCs and the IDENTITY rotation (the machinery-preservation ceiling:
    /// how much recall survives the PC round trip itself, isolating
    /// truncation loss from rotation quality; the cross-fit rotation is
    /// meaningless on recipient-side coordinates — it maps donor-PC coords,
    /// not recipient-PC coords).
    fn apply_self(&self, recipient_key: &[f64]) -> Vec<f64> {
        let z: Vec<f64> = self.pcs.project(recipient_key);
        self.pcs.reconstruct(&z)
    }
}

/// Spread-match scale (the paper's law): the ratio of the recipient-side
/// projected spread to the donor-side projected spread, over the TRAIN
/// split (never test — the leak law).
fn spread_scale(
    pcs_rec: &PcSpace,
    pcs_don: &PcSpace,
    _rot: &[Vec<f64>],
    rec_keys: &[Vec<f64>],
    don_keys: &[Vec<f64>],
) -> f64 {
    let rec_spread = projected_spread(pcs_rec, rec_keys);
    let don_spread = projected_spread(pcs_don, don_keys);
    if don_spread <= 1e-300 {
        return 1.0;
    }
    // The rotated donor coordinates inherit the rotation's norm-preservation,
    // so the spread ratio is computed on the recipient target side vs the
    // donor projected side (both in their own PC coordinates, scale-free).
    rec_spread / don_spread
}

/// Least-squares scale arm (the pre-registered scale-law contrast):
/// `s_ls = ⟨R·z_don, z_rec⟩ / ⟨R·z_don, R·z_don⟩` over the train pairs.
fn ls_scale(
    pcs_rec: &PcSpace,
    pcs_don: &PcSpace,
    rot: &[Vec<f64>],
    rec_keys: &[Vec<f64>],
    don_keys: &[Vec<f64>],
) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for (rk, dk) in rec_keys.iter().zip(don_keys) {
        let z_don = pcs_don.project(dk);
        let z_rot = mat_vec(rot, &z_don);
        let z_rec = pcs_rec.project(rk);
        num += z_rot.iter().zip(&z_rec).map(|(a, b)| a * b).sum::<f64>();
        den += z_rot.iter().map(|a| a * a).sum::<f64>();
    }
    if den <= 1e-300 {
        return 1.0;
    }
    num / den
}

fn projected_spread(pcs: &PcSpace, keys: &[Vec<f64>]) -> f64 {
    let zs: Vec<Vec<f64>> = keys.iter().map(|k| pcs.project(k)).collect();
    let r = zs[0].len();
    let mut var_sum = 0.0;
    for dim in 0..r {
        let mean = zs.iter().map(|z| z[dim]).sum::<f64>() / zs.len() as f64;
        var_sum += zs.iter().map(|z| (z[dim] - mean).powi(2)).sum::<f64>() / zs.len() as f64;
    }
    var_sum.sqrt()
}

// ── the synthetic protocol dry-run (self-test) ──────────────────────────

struct SyntheticWorld {
    /// The linear "recall readout" (K × d0): argmax plays the answer.
    readout: Vec<Vec<f64>>,
    /// Recipient's own keys (paired-correct by construction).
    recipient_keys: Vec<Vec<f64>>,
    /// Donor keys = R*ᵀ · recipient targets + noise.
    donor_keys: Vec<Vec<f64>>,
    /// Gold object per fact.
    gold: Vec<usize>,
    /// The planted rotation (held for the recovery readout; the fit never
    /// sees it — the dry-run asserts functional recovery, not matrix equality).
    #[allow(dead_code)]
    planted: Vec<Vec<f64>>,
    d0: usize,
}

fn build_synthetic(seed: u64) -> SyntheticWorld {
    const D0: usize = 64;
    const R_SUB: usize = 12;
    const K: usize = 8;
    const N: usize = 512;

    let mut rng = SplitMix64::new(seed);

    // The readout rows.
    let mut readout: Vec<Vec<f64>> = (0..K).map(|_| rng.gauss_vec(D0)).collect();
    for row in &mut readout {
        let norm: f64 = row.iter().map(|x| x * x).sum::<f64>().sqrt();
        for x in row {
            *x /= norm;
        }
    }

    // The planted rotation.
    let planted = random_orthogonal(D0, &mut rng);

    // The recipient-side subspace basis (D0 × R_SUB, orthonormal columns).
    let sub_cols: Vec<Vec<f64>> = (0..R_SUB).map(|_| rng.gauss_vec(D0)).collect();
    let basis = orthonormalize_columns(sub_cols);

    // The donor's OWN geometry — independent readout rows + basis, the
    // cross-ARCH shape (the two key spaces share nothing but the planted
    // map). A shared geometry would make the identity control transfer by
    // construction (the same-arch shape A2 adjudicates, not A1).
    let readout_don: Vec<Vec<f64>> = (0..K).map(|_| rng.gauss_vec(D0)).collect();
    let mut readout_don_n = Vec::with_capacity(K);
    for row in &readout_don {
        let norm: f64 = row.iter().map(|x| x * x).sum::<f64>().sqrt();
        readout_don_n.push(row.iter().map(|x| x / norm).collect::<Vec<f64>>());
    }
    let basis_don = orthonormalize_columns((0..R_SUB).map(|_| rng.gauss_vec(D0)).collect());

    let mut recipient_keys = Vec::with_capacity(N);
    let mut donor_keys = Vec::with_capacity(N);
    let mut gold = Vec::with_capacity(N);
    for j in 0..N {
        let o = (j * 7 + 3) % K;
        // Recipient latent v = 3·unit(readout[o]) + basis·g — reads o under
        // the recipient readout by construction.
        let mut v = readout[o].clone();
        for x in &mut v {
            *x *= 3.0;
        }
        let g: Vec<f64> = (0..R_SUB).map(|_| rng.gauss() * 0.25).collect();
        for (gi, b) in g.iter().zip(&basis) {
            for (vi, bv) in v.iter_mut().zip(b) {
                *vi += gi * bv;
            }
        }
        // Recipient's own key: small jitter, still reads o.
        let mut k_rec = v.clone();
        for x in &mut k_rec {
            *x += rng.gauss() * 0.02;
        }
        // Donor latent: the SAME gold object under the DONOR's geometry.
        let mut u = readout_don_n[o].clone();
        for x in &mut u {
            *x *= 3.0;
        }
        let gd: Vec<f64> = (0..R_SUB).map(|_| rng.gauss() * 0.25).collect();
        for (gi, b) in gd.iter().zip(&basis_don) {
            for (vi, bv) in u.iter_mut().zip(b) {
                *vi += gi * bv;
            }
        }
        // Donor key: the planted map of the donor latent, small jitter.
        let mut k_don = mat_vec(&transposed(&planted), &u);
        for x in &mut k_don {
            *x += rng.gauss() * 0.02;
        }
        recipient_keys.push(k_rec);
        donor_keys.push(k_don);
        gold.push(o);
    }

    SyntheticWorld {
        readout,
        recipient_keys,
        donor_keys,
        gold,
        planted,
        d0: D0,
    }
}

fn transposed(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = m.len();
    let cols = m[0].len();
    let mut t = vec![vec![0.0f64; rows]; cols];
    for (i, row) in m.iter().enumerate() {
        for (j, x) in row.iter().enumerate() {
            t[j][i] = *x;
        }
    }
    t
}

/// The linear readout: argmax (first index — the tie law).
fn readout_argmax(readout: &[Vec<f64>], h: &[f64]) -> usize {
    let mut best = 0usize;
    let mut best_v = f64::NEG_INFINITY;
    for (k, row) in readout.iter().enumerate() {
        let s: f64 = row.iter().zip(h).map(|(a, b)| a * b).sum();
        if s > best_v {
            best_v = s;
            best = k;
        }
    }
    best
}

/// Fit the swap on a key split, in the synthetic world's single shared space
/// (both sides are d0-dim; the REAL cell fits in each side's own PC space —
/// the synthetic dry-run exercises the same project→rotate→reconstruct path
/// with each side's own PCA).
fn fit_synthetic(world: &SyntheticWorld, train_idx: &[usize], r_pc: usize, seed: u64) -> SwapMap {
    let rec_train: Vec<Vec<f64>> = train_idx
        .iter()
        .map(|&i| world.recipient_keys[i].clone())
        .collect();
    let don_train: Vec<Vec<f64>> = train_idx
        .iter()
        .map(|&i| world.donor_keys[i].clone())
        .collect();

    let pcs_rec = fit_pca(&rec_train, r_pc, seed ^ 0xA11CE);
    let pcs_don = fit_pca(&don_train, r_pc, seed ^ 0xD0CA7);

    // Procrustes on the paired PC coordinates (f64, one-sided Jacobi).
    let mut a = Vec::with_capacity(train_idx.len());
    let mut b = Vec::with_capacity(train_idx.len());
    for &i in train_idx {
        a.push(pcs_don.project(&world.donor_keys[i]));
        b.push(pcs_rec.project(&world.recipient_keys[i]));
    }
    let rot = procrustes(&a, &b);

    let scale = spread_scale(&pcs_rec, &pcs_don, &rot, &rec_train, &don_train);
    SwapMap {
        pcs: pcs_rec,
        pcs_donor: pcs_don,
        rot,
        scale,
    }
}

fn run_self_test() -> Result<()> {
    let t0 = Instant::now();
    eprintln!("[self-test] building the synthetic world (planted rotation)…");
    let world = build_synthetic(0x1254);

    let n = world.recipient_keys.len();
    let n_train = n * 3 / 4;
    let n_val = n / 8;
    let train_idx: Vec<usize> = (0..n_train).collect();
    let val_idx: Vec<usize> = (n_train..n_train + n_val).collect();
    let test_idx: Vec<usize> = (n_train + n_val..n).collect();

    eprintln!("[self-test] fitting the swap (PCA → Procrustes → scale)…");

    // Machinery probe 1 — Procrustes recovers a planted r×r rotation on
    // noiseless pairs, to f64 precision. Isolates the fit from the PCA path.
    {
        let mut prng = SplitMix64::new(0x99);
        let rt = random_orthogonal(16, &mut prng);
        let a: Vec<Vec<f64>> = (0..200).map(|_| prng.gauss_vec(16)).collect();
        let b: Vec<Vec<f64>> = a.iter().map(|row| mat_vec(&rt, row)).collect();
        let r_hat = procrustes(&a, &b);
        let err: f64 = r_hat
            .iter()
            .zip(&rt)
            .flat_map(|(row_r, row_t)| row_r.iter().zip(row_t).map(|(x, y)| (x - y) * (x - y)))
            .sum::<f64>()
            .sqrt();
        assert!(
            err < 1e-8,
            "procrustes failed to recover a planted rotation: {err:.3e}"
        );
    }

    // Machinery probe 2 — the PCA round trip preserves a planted map:
    // project → identity-rotate → reconstruct must return the input when the
    // input lives in the captured subspace.
    {
        let mut prng = SplitMix64::new(0x98);
        // Anisotropic draw: a planted 6-dim subspace carries ALL the
        // variance (basis directions scaled 10×); r=8 of d=12 captures it
        // whole, so the round trip must be near-exact. (Isotropic samples
        // would lose energy under ANY truncation — a broken probe, not a
        // broken PCA.)
        let basis_cols: Vec<Vec<f64>> = (0..12).map(|_| prng.gauss_vec(12)).collect();
        let basis = orthonormalize_columns(basis_cols); // 12 columns, keep 6
        let samples: Vec<Vec<f64>> = (0..300)
            .map(|_| {
                let g: Vec<f64> = (0..6).map(|_| prng.gauss() * 10.0).collect();
                let mut v = vec![0.0f64; 12];
                for (gi, b) in g.iter().zip(basis.iter().take(6)) {
                    for (vi, bv) in v.iter_mut().zip(b) {
                        *vi += gi * bv;
                    }
                }
                v
            })
            .collect();
        // r = the planted rank (6): truncation at the data rank is the
        // round-trip-exact posture; requesting r above the rank would be a
        // degenerate probe (the QR of XΩ has exactly rank-6 columns).
        let pcs = fit_pca(&samples, 6, 0x77);
        let mut max_err = 0.0f64;
        for s in samples.iter().take(20) {
            let z = pcs.project(s);
            let x = pcs.reconstruct(&z);
            let err = x
                .iter()
                .zip(s)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            max_err = max_err.max(err);
        }
        eprintln!(
            "[self-test] pca round-trip probe (r=6 of d=12, planted 6-dim subspace): max err = {max_err:.3e}"
        );
        assert!(
            max_err < 1.0,
            "PCA round trip lost an in-subspace vector: {max_err:.3e}"
        );
    }

    let swap = fit_synthetic(&world, &train_idx, 16, 0x5EED);

    // Recovery readout: the recovered rotation against the planted one, in
    // the shared PC coordinates — report-only (the gates are functional).
    let mut plant_err = 0.0;
    for (i, &ti) in train_idx.iter().enumerate().take(64) {
        let _ = i;
        let mapped = swap.apply(&world.donor_keys[ti]);
        let err: f64 = mapped
            .iter()
            .zip(&world.recipient_keys[ti])
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        plant_err += err.sqrt();
    }
    plant_err /= 64.0;

    // ── arms over the test split ──
    let mut arms: Vec<(String, usize)> = vec![
        ("transfer".into(), 0),
        ("identity".into(), 0),
        ("random_rotation".into(), 0),
        ("shuffled_donor".into(), 0),
        ("self_swap".into(), 0),
        ("ls_scale".into(), 0),
    ];
    let n_test = test_idx.len();

    // The LS-scale twin of the transfer map (scale-law arm).
    let ls = ls_scale(
        &swap.pcs,
        &swap.pcs_donor,
        &swap.rot,
        &train_idx
            .iter()
            .map(|&i| world.recipient_keys[i].clone())
            .collect::<Vec<_>>(),
        &train_idx
            .iter()
            .map(|&i| world.donor_keys[i].clone())
            .collect::<Vec<_>>(),
    );
    let swap_ls = SwapMap {
        pcs: swap.pcs.clone(),
        pcs_donor: swap.pcs_donor.clone(),
        rot: swap.rot.clone(),
        scale: ls,
    };

    // The random-rotation control through the IDENTICAL path.
    let mut rng = SplitMix64::new(0x1D3A);
    let random_rot = random_orthogonal(swap.rot.len(), &mut rng);
    let swap_random = SwapMap {
        pcs: swap.pcs.clone(),
        pcs_donor: swap.pcs_donor.clone(),
        rot: random_rot,
        scale: swap.scale,
    };
    // The identity control: the r×r identity through the same path.
    let r_dim = swap.rot.len();
    let identity: Vec<Vec<f64>> = (0..r_dim)
        .map(|i| (0..r_dim).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();
    let swap_identity = SwapMap {
        pcs: swap.pcs.clone(),
        pcs_donor: swap.pcs_donor.clone(),
        rot: identity,
        scale: swap.scale,
    };

    for (t, &ti) in test_idx.iter().enumerate() {
        let gold_obj = world.gold[ti];
        // transfer
        let h = swap.apply(&world.donor_keys[ti]);
        if readout_argmax(&world.readout, &h) == gold_obj {
            arms[0].1 += 1;
        }
        // identity
        let h = swap_identity.apply(&world.donor_keys[ti]);
        if readout_argmax(&world.readout, &h) == gold_obj {
            arms[1].1 += 1;
        }
        // random rotation
        let h = swap_random.apply(&world.donor_keys[ti]);
        if readout_argmax(&world.readout, &h) == gold_obj {
            arms[2].1 += 1;
        }
        // shuffled donor keys (the donor key of a DIFFERENT test fact)
        let other = test_idx[(t + 101) % n_test];
        let h = swap.apply(&world.donor_keys[other]);
        if readout_argmax(&world.readout, &h) == gold_obj {
            arms[3].1 += 1;
        }
        // self-swap ceiling: the recipient's own key through the same
        // project→rotate→reconstruct path with its OWN PCs (apply_self)
        let h = swap.apply_self(&world.recipient_keys[ti]);
        if readout_argmax(&world.readout, &h) == gold_obj {
            arms[4].1 += 1;
        }
        // LS-scale twin
        let h = swap_ls.apply(&world.donor_keys[ti]);
        if readout_argmax(&world.readout, &h) == gold_obj {
            arms[5].1 += 1;
        }
    }

    let rate = |hits: usize| hits as f64 / n_test as f64;
    let transfer = rate(arms[0].1);
    let identity = rate(arms[1].1);
    let random = rate(arms[2].1);

    // The pre-registered G1 arithmetic, verbatim.
    let g1_abs = transfer >= 0.15;
    let g1_vs_random = transfer >= 3.0 * random;
    let g1_vs_identity = transfer >= 2.0 * identity;
    let pass = g1_abs && g1_vs_random && g1_vs_identity;

    println!("\n═══ platos_key_cell — SELF-TEST (protocol dry-run, no science claim) ═══");
    println!(
        "facts: {} (train {} / val {} / test {}), d0 = {}, r_pc = 16",
        n,
        train_idx.len(),
        val_idx.len(),
        n_test,
        world.d0
    );
    println!("plant-recovery mean reconstruction error (train head): {plant_err:.4e}");
    for (name, hits) in &arms {
        println!("  {name:>16}: {hits:>4}/{} = {:.3}", n_test, rate(*hits));
    }
    println!(
        "  scale: spread = {:.4}, ls = {:.4} (scale-law arm: LS vs spread transfer {:.3} vs {:.3})",
        swap.scale,
        ls,
        rate(arms[5].1),
        transfer
    );
    println!(
        "G1: abs(≥0.15)={}  ≥3×random={}  ≥2×identity={}  ⇒ {}",
        g1_abs,
        g1_vs_random,
        g1_vs_identity,
        if pass { "PASS" } else { "FAIL" }
    );

    // The self-test's own bar: the machinery MUST recover the planted map —
    // transfer at ceiling, controls at floor. A miss is broken machinery,
    // not a science result.
    let ceiling = rate(arms[4].1);
    assert!(
        transfer >= 0.9 && ceiling >= 0.9,
        "self-test: the pipeline failed to recover the planted rotation \
         (transfer {:.3}, self-swap ceiling {:.3}) — the machinery is broken, \
         fix before any real cell",
        transfer,
        ceiling
    );
    assert!(
        transfer >= 2.0 * identity && transfer >= 3.0 * random,
        "self-test: controls not separated from transfer (identity {:.3}, random {:.3})",
        identity,
        random
    );
    println!(
        "self-test: machinery validated in {:.1?} — PASS",
        t0.elapsed()
    );
    Ok(())
}

// ── the real cell (recipient side this commit; donor side refuses loud) ──

/// The Issue-395 hook, capture shape: records the post-layer residual at the
/// armed layers. The harness arms it per CALL (subject position only).
/// Present but unwired until the donor-capture seam lands (the P3 refusal);
/// the recipient-side band capture will ride this same type.
#[allow(dead_code)]
struct CaptureHook {
    capture_layers: &'static [usize],
    captured: Vec<Option<Vec<f32>>>,
}
impl CaptureHook {
    #[allow(dead_code)]
    fn new(capture_layers: &'static [usize]) -> Self {
        Self {
            capture_layers,
            captured: vec![None; capture_layers.len()],
        }
    }
}
impl PostLayerHook for CaptureHook {
    fn after_layer(&mut self, layer_idx: usize, residual: &mut [f32]) {
        if let Some(slot) = self.capture_layers.iter().position(|&l| l == layer_idx) {
            self.captured[slot] = Some(residual.to_vec());
        }
    }
}

/// The Issue-395 hook, replace shape: writes a prepared state into the
/// post-layer residual at the armed layer (the swap's replace point).
/// The Issue-395 hook, replace shape — unwired until the donor seam lands
/// (the swap arm's apply point). `#[allow(dead_code)]` carries the same
/// pending-seam reason.
#[allow(dead_code)]
struct ReplaceHook {
    layer: usize,
    state: Vec<f32>,
}
impl PostLayerHook for ReplaceHook {
    fn after_layer(&mut self, layer_idx: usize, residual: &mut [f32]) {
        if layer_idx == self.layer {
            let n = residual.len().min(self.state.len());
            residual[..n].copy_from_slice(&self.state[..n]);
        }
    }
}

fn argmax_first(logits: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

fn distinct_token_ratio(window: &[usize]) -> f64 {
    if window.is_empty() {
        return 0.0;
    }
    let mut seen = std::collections::HashSet::new();
    for t in window {
        seen.insert(*t);
    }
    seen.len() as f64 / window.len() as f64
}

fn recall_hit(gen_toks: &[usize], answer: &str, tok: &BpeTokenizer) -> bool {
    let k = gen_toks.len().min(RECALL_FIRST_K);
    if k == 0 {
        return false;
    }
    let window = &gen_toks[..k];
    if distinct_token_ratio(window) < RECALL_DISTINCT_RATIO {
        return false;
    }
    tok.decode(window).contains(answer)
}

/// Locate the subject's last token index with the BPE-locality check.
/// Returns None when the boundary check fails (the fact is skipped and
/// counted, never silently).
fn subject_last_token(tok: &BpeTokenizer, fact: &FactDoc) -> Option<(Vec<usize>, usize)> {
    let start = fact.prompt.find(&fact.subject)?;
    let end = start + fact.subject.len();
    let prefix_subj = &fact.prompt[..end];
    let full = tok.encode_with_bos(&fact.prompt);
    let ps = tok.encode_with_bos(prefix_subj);
    if ps.len() < 2 || full.len() < ps.len() + 1 {
        return None;
    }
    // BPE locality: the full prompt's token prefix must equal the prefix's
    // own tokenization (a merge across the boundary breaks the index law).
    if full[..ps.len()] != ps[..] {
        return None;
    }
    Some((full, ps.len() - 1))
}

fn run_real(args: &RealArgs) -> Result<()> {
    let _t0 = Instant::now(); // wall-clock provenance for the phase logs (never a perf claim)
    // ── P0: load recipient + tokenizer, rope disposition, ctx clamp ──
    let recipient_path = PathBuf::from(&args.recipient);
    let gguf = GgufFile::open(&recipient_path)
        .with_context(|| format!("opening recipient {}", recipient_path.display()))?;
    let arch = gguf.architecture().unwrap_or("unknown").to_string();
    if arch != "llama" {
        // The architecture law: read from the header, never the filename.
        bail!(
            "recipient GGUF header architecture is '{arch}', expected 'llama' \
             (the architecture law: headers, never filenames)"
        );
    }
    let tok = BpeTokenizer::from_gguf(&gguf).context("recipient BPE tokenizer")?;
    drop(gguf);
    let (mut config, weights) = load_llama_weights_gguf(&recipient_path)
        .with_context(|| format!("loading recipient weights {}", recipient_path.display()))?;

    // ⛔ ctx clamp: llama-3.2-3B's GGUF context_length (131072) must never
    // size the KV cache — clamp to the actual sequence budget.
    config.block_size = args.ctx;
    eprintln!(
        "[p0] recipient: d={} layers={} kv_heads={} head_dim={} vocab={} rope_theta={:.0} \
         rope_factors={} ctx_clamped={}",
        config.n_embd,
        config.n_layer,
        config.n_kv_head,
        config.head_dim,
        config.vocab_size,
        config.rope_theta,
        weights.rope_freq_weights.is_some(),
        config.block_size
    );
    if config.n_embd * config.n_layer > 0 {
        // (sanity: the header numbers above are the run's provenance rows)
    }
    let file_len = std::fs::metadata(&recipient_path)?.len();
    eprintln!(
        "[p0] provenance: path={} size={} (sha256 recorded at the download step)",
        recipient_path.display(),
        file_len
    );

    // The weighted RoPE table — the ONE constructor (Issue 1254 disposition).
    let rope_table = weights.build_rope_freq_table(&config);

    let facts = load_facts(Path::new(&args.facts))?;
    eprintln!("[p0] facts: {} loaded from {}", facts.len(), args.facts);

    // Depth band (22–38%), never absolute indices.
    let band = depth_band(config.n_layer);
    let band_static: &'static [usize] = Box::leak(band.into_boxed_slice());
    eprintln!(
        "[p0] depth band 22–38%: layers {band_static:?} of {}",
        config.n_layer
    );

    // ── P1: real-model G3 identity (plain ≡ hooked(NoHook), bit-exact) ──
    eprintln!("[p1] G3 real-model identity probe…");
    {
        let probe = "The capital of France is";
        let tokens = tok.encode_with_bos(probe);
        let run_plain = |tokens: &[usize]| -> Vec<Vec<f32>> {
            let cache = &mut MultiLayerKVCache::new(&config);
            let mut ctx = ForwardContext::new(&config);
            ctx.rope_freq_table = rope_table.clone();
            let mut snap = Vec::new();
            for (pos, &t) in tokens.iter().enumerate() {
                let logits = forward_llama(&mut ctx, &weights, cache, t, pos, &config);
                snap.push(logits.to_vec());
            }
            snap
        };
        let run_hooked = |tokens: &[usize]| -> Vec<Vec<f32>> {
            let cache = &mut MultiLayerKVCache::new(&config);
            let mut ctx = ForwardContext::new(&config);
            ctx.rope_freq_table = rope_table.clone();
            let mut hook = NoHook;
            let mut snap = Vec::new();
            for (pos, &t) in tokens.iter().enumerate() {
                let logits =
                    forward_llama_hooked(&mut ctx, &weights, cache, t, pos, &config, &mut hook);
                snap.push(logits.to_vec());
            }
            snap
        };
        let plain = run_plain(&tokens);
        let hooked = run_hooked(&tokens);
        for (pos, (a, b)) in plain.iter().zip(&hooked).enumerate() {
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "G3 REAL-MODEL IDENTITY FAILED: plain vs hooked(NoHook) diverged \
                     at pos {pos} logit {i} — the seam is not a no-op"
                );
            }
        }
        eprintln!(
            "[p1] G3 PASS: {} positions × {} logits bit-identical",
            plain.len(),
            plain[0].len()
        );
    }

    // ── P2: the baseline recall gate (recipient alone; NO transfer number
    //    past a failed gate — the verdict round-1 addition) ──
    eprintln!("[p2] baseline recall gate (floor {})…", args.baseline_floor);
    let mut baseline_hits = 0usize;
    let mut usable: Vec<(usize, Vec<usize>, usize)> = Vec::new(); // (fact idx, tokens, subject pos)
    let mut boundary_skips = 0usize;
    for (i, fact) in facts.iter().enumerate() {
        let Some((tokens, subj_pos)) = subject_last_token(&tok, fact) else {
            boundary_skips += 1;
            continue;
        };
        if tokens.len() + RECALL_FIRST_K + 1 > args.ctx {
            boundary_skips += 1; // prompt exceeds the clamped ctx budget
            continue;
        }
        let cache = &mut MultiLayerKVCache::new(&config);
        let mut ctx = ForwardContext::new(&config);
        ctx.rope_freq_table = rope_table.clone();
        let mut gen_toks: Vec<usize> = Vec::with_capacity(RECALL_FIRST_K);
        let mut logits = Vec::new();
        for (pos, &t) in tokens.iter().enumerate() {
            logits = forward_llama(&mut ctx, &weights, cache, t, pos, &config).to_vec();
        }
        let mut next = argmax_first(&logits);
        for _ in 0..RECALL_FIRST_K {
            if next == tok.eos_id() {
                break;
            }
            gen_toks.push(next);
            let pos = tokens.len() + gen_toks.len() - 1;
            logits = forward_llama(&mut ctx, &weights, cache, next, pos, &config).to_vec();
            next = argmax_first(&logits);
        }
        if recall_hit(&gen_toks, &fact.answer, &tok) {
            baseline_hits += 1;
            usable.push((i, tokens, subj_pos));
        }
    }
    let baseline_acc = baseline_hits as f64 / facts.len() as f64;
    eprintln!(
        "[p2] baseline recall: {}/{} = {:.3} (boundary skips: {})",
        baseline_hits,
        facts.len(),
        baseline_acc,
        boundary_skips
    );
    if baseline_acc < args.baseline_floor {
        bail!(
            "BASELINE RECALL GATE FAILED: recipient accuracy {:.3} < floor {:.3} — \
             no transfer number is reported (exit 4). Fix the recipient posture \
             (tokenizer, rope factors, prompt format) before any cell run.",
            baseline_acc,
            args.baseline_floor
        );
    }
    let paired_target = 2_000usize;
    if usable.len() < paired_target {
        eprintln!(
            "[p2] ⚠ paired pool {} < the pre-registered ≥2k target — the run may \
             proceed below target with the shortfall disclosed in the report",
            usable.len()
        );
    }
    if usable.len() < 32 {
        bail!(
            "paired pool {} below the minimum adjudicable set (32) — \
             widen the facts corpus",
            usable.len()
        );
    }

    // ── P3: donor capture — the pending seam ──
    eprintln!("[p3] donor capture…");
    bail!(
        "DONOR CAPTURE SEAM PENDING (exit 3): the ternary-path hidden-state \
         capture hook (Ternary-Bonsai-2.5-27B donor, qwen35 GDN hybrid, the \
         HybridForwardScratch harvest precedent) is its own G3-gated commit — \
         the harness refuses here, never a silent half-run. The recipient-side \
         phases above (P0 load/rope, P1 G3, P2 baseline gate {:.3}) are the \
         download+baseline rig job's deliverables.",
        baseline_acc
    );
}

fn depth_band(n_layer: usize) -> Vec<usize> {
    let lo = ((n_layer as f64 * 0.22).round() as usize).max(1);
    let hi = ((n_layer as f64 * 0.38).round() as usize).min(n_layer - 1);
    (lo..=hi).collect()
}

struct RealArgs {
    recipient: String,
    #[allow(dead_code)]
    donor: String,
    facts: String,
    /// The report directory — consumed by the report-writing phase that
    /// lands with the donor seam; created eagerly so a run's dir exists.
    #[allow(dead_code)]
    out: PathBuf,
    ctx: usize,
    baseline_floor: f64,
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut self_test = false;
    let mut recipient_opt: Option<String> = None;
    let mut donor: Option<String> = None;
    let mut facts: Option<String> = None;
    let mut out = PathBuf::from(".platos_key_cell/");
    let mut ctx = 512usize;
    let mut baseline_floor = 0.4f64;

    while let Some(a) = args.next() {
        match a.as_str() {
            "--self-test" => self_test = true,
            "--recipient" => {
                recipient_opt = Some(args.next().context("--recipient needs a value")?)
            }
            "--donor" => donor = Some(args.next().context("--donor needs a value")?),
            "--facts" => facts = Some(args.next().context("--facts needs a value")?),
            "--out" => out = PathBuf::from(args.next().context("--out needs a value")?),
            "--ctx" => {
                ctx = args
                    .next()
                    .context("--ctx needs a value")?
                    .parse()
                    .context("--ctx must be an integer")?
            }
            "--baseline-floor" => {
                baseline_floor = args
                    .next()
                    .context("--baseline-floor needs a value")?
                    .parse()
                    .context("--baseline-floor must be a float in [0, 1]")?;
                if !(0.0..=1.0).contains(&baseline_floor) {
                    bail!("--baseline-floor must be in [0, 1]");
                }
            }
            other => bail!("unknown arg {other:?} (see the module docs)"),
        }
    }

    if self_test {
        if recipient_opt.is_some() || facts.is_some() {
            bail!("--self-test takes no model/fact args");
        }
        return run_self_test();
    }

    let recipient = recipient_opt.context("real mode requires --recipient")?;
    let donor = donor.unwrap_or_else(|| {
        // The donor of record (Issue 1254); overridable for A2/A3 postures.
        "/Users/katopz/git/riir-train/data/Ternary-Bonsai-2.5-27B-PQ2_0.gguf".to_string()
    });
    let facts = facts.context("real mode requires --facts")?;
    std::fs::create_dir_all(&out)?;
    let real = RealArgs {
        recipient,
        donor,
        facts,
        out,
        ctx,
        baseline_floor,
    };
    let res = run_real(&real);
    if let Err(e) = &res {
        // Distinguish the designed refusals from failures by their message
        // markers (the exit-code contract in the module docs).
        let msg = format!("{e:#}");
        if msg.contains("DONOR CAPTURE SEAM PENDING") {
            std::process::exit(3);
        }
        if msg.contains("BASELINE RECALL GATE FAILED") {
            std::process::exit(4);
        }
    }
    res
}
