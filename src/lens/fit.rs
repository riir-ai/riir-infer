//! Lens fit + reads + gates — T2d of katgpt-rs Issue 999 (Research 666).
//!
//! The corpus-averaged R/S lenses over the doc-level stack
//! ([`crate::lens::stack`]):
//!
//! - **S lens (backward)** — `K_P = mean_t Σ_{s≤t} ∂h_P(t)/∂e(s)` per fit
//!   point `P`. By causality, `Σ_{s≤t} ∂h_P(t)/∂e(s) · u` is EXACTLY the
//!   prefix jvp with tangent `u` at EVERY input position (positions `s > t`
//!   contribute zero), so one prefix pass per basis direction fills one
//!   column of `K_P` for every point at once (the fixed-end law).
//! - **R lens (forward)** — `J_P = mean_t Σ_{t'≥t} ∂h_hn(t')/∂h_P(t)`.
//!   Same superposition: one suffix jvp per (point, basis direction) with
//!   the write at EVERY position of `P`; summing the response over all
//!   final positions `t'` counts each causal pair `(τ ≤ t')` exactly once.
//!   The final-norm's LN-rule diagonal is folded into the suffix, so `J_P`
//!   maps straight into the space `lm_head` reads.
//! - **Reads** — S: `score(v) = ⟨h, K·(W_E[v] − mean(W_E))⟩`, computed as
//!   ONE `d×d` transpose-matvec (`tmp = Kᵀh`) + ONE vocab-side matvec
//!   (`wte·tmp`) minus a constant — one matvec per direction over the
//!   vocabulary (G4). R: `Δlogits = W_U·(J·v)`; the softmax read adds the
//!   doc's base `hn` before the head.
//! - **Gates G1–G4** — the pre-registered bars (Issue 999 T4): G1 S-lens
//!   top-10 hit rate ≥ 25% with a <1% mismatched-pole control (UNSEEN when
//!   no poles — a scope finding, never a refutation); G2 R-read vs measured
//!   steering correlation ≥ 0.45 (control ±0.04); G3 even/odd half-corpus
//!   fits agree ≤ 6% per point; G4 one vocab-side matvec per direction,
//!   zero-alloc, fit cost recorded with box state.
//!
//! The fit is offline (once per checkpoint), never a serving path. Sweeps
//! run one basis direction at a time — `dim_batch` block batching is the
//! T3 perf lever, recorded here as an option, not yet applied.

use crate::lens::ops::matvec_t;
use crate::lens::stack::{
    FitPoint, PrefixTangents, StackCapture, StackScratch, StackWeights, stack_forward,
    stack_prefix_jvp, stack_suffix_jvp,
};
use crate::types::matmul;

// ---------------------------------------------------------------------------
// Accumulators
// ---------------------------------------------------------------------------

/// Per-point, per-direction column accumulator over the fitted matrices:
/// `sum[(point·d + dir)·d + j]` accumulates `Σ_t (per_pos[t·d + j]) / T`
/// per doc (the 1/T position mean folds in here); `n_docs` counts added
/// docs for the finalize divide. Column `dir` of point `point` is the
/// mean response to the `dir`-th basis direction — exactly `K[:, dir]` /
/// `J[:, dir]` in the corpus-averaged lens definition.
#[derive(Clone)]
pub struct ColAccum {
    sum: Vec<f64>,
    d: usize,
    points: usize,
    n_docs: u64,
}

impl ColAccum {
    pub fn new(d: usize, points: usize) -> Self {
        Self {
            sum: vec![0.0; points * d * d],
            d,
            points,
            n_docs: 0,
        }
    }

    /// Add one doc's column: `per_pos` is the `[T·d]` tangent at `point`
    /// under the `dir`-th basis direction; the position-summed, T-meaned
    /// vector joins that column.
    pub fn add_doc_column(&mut self, point: usize, dir: usize, per_pos: &[f32], t_len: usize) {
        let d = self.d;
        debug_assert_eq!(per_pos.len(), t_len * d, "ColAccum: per_pos shape");
        debug_assert!(point < self.points, "ColAccum: point index");
        debug_assert!(dir < d, "ColAccum: dir index");
        let inv_t = 1.0 / t_len as f64;
        let base = (point * d + dir) * d;
        for j in 0..d {
            let mut acc = 0.0f64;
            for t in 0..t_len {
                acc += per_pos[t * d + j] as f64;
            }
            self.sum[base + j] += acc * inv_t;
        }
    }

    pub fn note_doc(&mut self) {
        self.n_docs += 1;
    }

    pub fn n_docs(&self) -> u64 {
        self.n_docs
    }

    /// Finalize the per-point means (`sum / n_docs`); refuses an empty
    /// accumulator loud (a green zero is never a fit).
    pub fn finalize(&self) -> Vec<Vec<f32>> {
        assert!(
            self.n_docs > 0,
            "ColAccum::finalize on zero docs — no fit was performed"
        );
        let d = self.d;
        let inv = 1.0 / self.n_docs as f64;
        (0..self.points)
            .map(|p| {
                (0..d * d)
                    .map(|i| (self.sum[p * d * d + i] * inv) as f32)
                    .collect()
            })
            .collect()
    }
}

/// The fitted lens: per fit point, the S matrix `K` (`[d_P][d_E]` — the
/// prefix pair-sum, meaned over docs) and the R matrix `J` (`[d_hn][d_P]` —
/// the suffix pair-sum, final-norm diagonal included), row-major, plus the
/// weight-derived embedding mean.
pub struct LensFit {
    pub d: usize,
    pub points: Vec<FitPoint>,
    pub k: Vec<Vec<f32>>,
    pub j: Vec<Vec<f32>>,
    /// `mean(W_E)` over the vocabulary rows (`[d]`) — the S read's
    /// centering vector (weight-derived, not fitted).
    pub mean_emb: Vec<f32>,
    pub n_docs: u64,
}

impl LensFit {
    /// Point index in the canonical ordering (the fit's own `points`).
    pub fn point_index(&self, p: FitPoint) -> Option<usize> {
        self.points.iter().position(|&q| q == p)
    }
}

/// Fit cost + provenance (G4's record).
#[derive(Clone, Debug)]
pub struct FitCost {
    pub docs: u64,
    pub basis_dirs: usize,
    pub prefix_passes: u64,
    pub suffix_passes: u64,
    pub capture_secs: f64,
    pub sweep_secs: f64,
    /// The box state at fit time (the `bench_preflight.sh` `PROVENANCE:`
    /// line where applicable; tests record `"test"`).
    pub box_state: String,
}

// ---------------------------------------------------------------------------
// The fit
// ---------------------------------------------------------------------------

/// Fit the R/S lenses over `docs` (token id sequences, each of the stack's
/// `t_len`). Captures and sweeps one doc at a time (the capture's memory is
/// dominated by the GDN per-position states — see [`crate::lens::stack`]).
pub fn fit_lens(
    w: &StackWeights,
    docs: &[Vec<usize>],
    points: &[FitPoint],
    box_state: &str,
) -> (LensFit, FitCost) {
    let geo = &w.geo;
    let d = geo.n_embd;
    let t_len = geo.t_len;
    let mut fit = LensFit {
        d,
        points: points.to_vec(),
        k: Vec::new(),
        j: Vec::new(),
        mean_emb: mean_embedding(w.wte, geo.vocab, d),
        n_docs: 0,
    };
    let mut acc_k = ColAccum::new(d, points.len());
    let mut acc_j = ColAccum::new(d, points.len());
    let mut cost = FitCost {
        docs: 0,
        basis_dirs: d,
        prefix_passes: 0,
        suffix_passes: 0,
        capture_secs: 0.0,
        sweep_secs: 0.0,
        box_state: box_state.to_string(),
    };
    let mut cap = StackCapture::new(geo);
    let mut scratch = StackScratch::new(geo);
    let mut d_emb = vec![0.0f32; t_len * d];
    let mut dv = vec![0.0f32; t_len * d];
    let mut pt = PrefixTangents::new(geo);
    let mut dhn = vec![0.0f32; t_len * d];
    for tokens in docs {
        debug_assert_eq!(tokens.len(), t_len, "fit doc length");
        let t0 = std::time::Instant::now();
        stack_forward(tokens, w, &mut cap);
        cost.capture_secs += t0.elapsed().as_secs_f64();
        acc_k.note_doc();
        acc_j.note_doc();
        cost.docs += 1;
        let t1 = std::time::Instant::now();
        // S sweeps: one prefix pass per basis direction fills a K column for
        // EVERY point at once (the fixed-end law).
        for i in 0..d {
            for v in d_emb.iter_mut() {
                *v = 0.0;
            }
            for t in 0..t_len {
                d_emb[t * d + i] = 1.0;
            }
            stack_prefix_jvp(&cap, w, &mut scratch, &d_emb, &mut pt);
            cost.prefix_passes += 1;
            for (pi, p) in points.iter().enumerate() {
                let buf = match p {
                    FitPoint::Mid(_) => &pt.mid[p.layer()],
                    FitPoint::Out(_) => &pt.out[p.layer()],
                };
                acc_k.add_doc_column(pi, i, buf, t_len);
            }
        }
        // R sweeps: one suffix pass per (point, basis direction).
        for (pi, &p) in points.iter().enumerate() {
            for i in 0..d {
                for v in dv.iter_mut() {
                    *v = 0.0;
                }
                for t in 0..t_len {
                    dv[t * d + i] = 1.0;
                }
                stack_suffix_jvp(&cap, w, &mut scratch, p, &dv, &mut dhn);
                cost.suffix_passes += 1;
                acc_j.add_doc_column(pi, i, &dhn, t_len);
            }
        }
        cost.sweep_secs += t1.elapsed().as_secs_f64();
    }
    fit.k = acc_k.finalize();
    fit.j = acc_j.finalize();
    fit.n_docs = acc_k.n_docs();
    (fit, cost)
}

/// `mean(W_E)` over the vocabulary rows.
pub fn mean_embedding(wte: &[f32], vocab: usize, d: usize) -> Vec<f32> {
    let mut mean = vec![0.0f64; d];
    for v in 0..vocab {
        let row = &wte[v * d..(v + 1) * d];
        for (j, &x) in row.iter().enumerate() {
            mean[j] += x as f64;
        }
    }
    mean.iter().map(|&x| (x / vocab as f64) as f32).collect()
}

// ---------------------------------------------------------------------------
// Reads — one vocab-side matvec per direction, zero-alloc (G4)
// ---------------------------------------------------------------------------

/// S read: scores over the whole vocabulary for pole direction `h_dir` at
/// fit point `point_idx`.
///
/// `score(v) = ⟨tmp, W_E[v]⟩ − ⟨tmp, mean(W_E)⟩` with `tmp = Kᵀ·h` — one
/// `d×d` transpose-matvec + one vocab-side matvec, both into caller
/// buffers. `tmp` is `[d]`, `scores` is `[vocab]`.
pub fn s_scores_into(
    wte: &[f32],
    vocab: usize,
    fit: &LensFit,
    point_idx: usize,
    h_dir: &[f32],
    tmp: &mut [f32],
    scores: &mut [f32],
) {
    let d = fit.d;
    debug_assert_eq!(h_dir.len(), d, "s_scores: h shape");
    debug_assert_eq!(tmp.len(), d, "s_scores: tmp shape");
    debug_assert_eq!(scores.len(), vocab, "s_scores: scores shape");
    debug_assert_eq!(wte.len(), vocab * d, "s_scores: wte shape");
    matvec_t(tmp, &fit.k[point_idx], h_dir, d, d);
    matmul(scores, wte, tmp, vocab, d);
    let center: f64 = fit
        .mean_emb
        .iter()
        .zip(tmp.iter())
        .map(|(&m, &t)| m as f64 * t as f64)
        .sum();
    let c = center as f32;
    for s in scores.iter_mut() {
        *s -= c;
    }
}

/// R read: the predicted logit DELTA over the vocabulary for a write `v_dir`
/// at fit point `point_idx` — `Δlogits = W_U·(J·v)` (the G2 correlation
/// quantity). `tmp` is `[d]`, `delta` is `[vocab]`.
pub fn r_read_delta_into(
    lm_head: &[f32],
    vocab: usize,
    fit: &LensFit,
    point_idx: usize,
    v_dir: &[f32],
    tmp: &mut [f32],
    delta: &mut [f32],
) {
    let d = fit.d;
    debug_assert_eq!(v_dir.len(), d, "r_read: v shape");
    debug_assert_eq!(tmp.len(), d, "r_read: tmp shape");
    debug_assert_eq!(delta.len(), vocab, "r_read: delta shape");
    debug_assert_eq!(lm_head.len(), vocab * d, "r_read: head shape");
    matmul(tmp, &fit.j[point_idx], v_dir, d, d);
    matmul(delta, lm_head, tmp, vocab, d);
}

/// R read, softmax form: `softmax(W_U·(base_hn + J·v))` — the write's
/// predicted next-token distribution at the read position (the corpus-
/// averaged `J` applied at that position's base `hn`; Research 666's
/// `softmax(W_U·norm(J_L h))` with the norm folded into `J`).
#[allow(clippy::too_many_arguments)]
pub fn r_read_logits_into(
    lm_head: &[f32],
    vocab: usize,
    fit: &LensFit,
    point_idx: usize,
    base_hn: &[f32],
    v_dir: &[f32],
    tmp: &mut [f32],
    logits: &mut [f32],
) {
    let d = fit.d;
    debug_assert_eq!(base_hn.len(), d, "r_read_logits: base shape");
    matmul(tmp, &fit.j[point_idx], v_dir, d, d);
    for (t, &b) in tmp.iter_mut().zip(base_hn) {
        *t += b;
    }
    matmul(logits, lm_head, tmp, vocab, d);
}

/// Numerically-stable softmax (max-subtracted).
pub fn softmax_into(logits: &[f32], out: &mut [f32]) {
    debug_assert_eq!(logits.len(), out.len(), "softmax: shape");
    let mut m = f32::NEG_INFINITY;
    for &l in logits {
        if l > m {
            m = l;
        }
    }
    let mut sum = 0.0f64;
    for (o, &l) in out.iter_mut().zip(logits) {
        let e = (l - m).exp();
        *o = e;
        sum += e as f64;
    }
    let inv = 1.0 / sum;
    for o in out.iter_mut() {
        *o *= inv as f32;
    }
}

// ---------------------------------------------------------------------------
// Rare-token mask (G1's token scoring filter)
// ---------------------------------------------------------------------------

/// Token-frequency mask over the fitting corpus: a token scoring below
/// `min_count` occurrences is excluded from read top-k (frequent tokens
/// dominate raw S scores; the mask keeps the read honest on rare firings).
pub struct RareMask {
    counts: Vec<u32>,
    pub min_count: u32,
}

impl RareMask {
    pub fn from_docs(docs: &[Vec<usize>], vocab: usize, min_count: u32) -> Self {
        let mut counts = vec![0u32; vocab];
        for doc in docs {
            for &t in doc {
                counts[t] += 1;
            }
        }
        Self { counts, min_count }
    }

    /// The all-allowed mask (no corpus — e.g. the gate harness's planted
    /// constructions).
    pub fn allow_all(vocab: usize) -> Self {
        Self {
            counts: vec![u32::MAX; vocab],
            min_count: 1,
        }
    }

    pub fn allows(&self, v: usize) -> bool {
        self.counts[v] >= self.min_count
    }
}

/// Top-`k` token indices by score, masked tokens excluded, descending
/// (stable: equal scores keep vocabulary order).
pub fn top_k_indices(scores: &[f32], mask: &RareMask, k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).filter(|&v| mask.allows(v)).collect();
    idx.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    idx.truncate(k);
    idx
}

// ---------------------------------------------------------------------------
// Gates G1–G4 (the pre-registered bars, Issue 999 T4 / Research 666 §2.3)
// ---------------------------------------------------------------------------

/// Pearson correlation (population form; zero-variance inputs read 0.0 —
/// never NaN into a gate verdict).
pub fn pearson(x: &[f32], y: &[f32]) -> f64 {
    debug_assert_eq!(x.len(), y.len(), "pearson: shape");
    let n = x.len() as f64;
    let (mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for i in 0..x.len() {
        let (a, b) = (x[i] as f64, y[i] as f64);
        sx += a;
        sy += b;
        sxx += a * a;
        syy += b * b;
        sxy += a * b;
    }
    let cov = sxy - sx * sy / n;
    let vx = sxx - sx * sx / n;
    let vy = syy - sy * sy / n;
    if vx <= 0.0 || vy <= 0.0 {
        return 0.0;
    }
    cov / (vx.sqrt() * vy.sqrt())
}

/// G1 report: S-lens top-k hit rate on held-out firing tokens + the
/// mismatched-pole control.
#[derive(Clone, Copy, Debug)]
pub struct G1Report {
    pub poles: usize,
    pub top_k: usize,
    /// Mean over poles of "the pole's held-out token is in its own top-k".
    pub hit_rate: f64,
    /// Mean over ordered mismatched pairs (i, j), i≠j, of "pole i's top-k
    /// contains pole j's held-out token".
    pub control_rate: f64,
    pub unseen: bool,
    pub pass: bool,
}

/// **G1** — the S-lens names the firing token. `poles` is
/// `[(point_idx, direction)]`, `held_out` is `[(pole index, token)]`.
/// Bar (extrapolated from the repo's L0 figure — Issue 999 T4): hit ≥ 0.25,
/// control < 0.01. No poles ⇒ **UNSEEN** (scope finding, never a
/// refutation).
#[allow(clippy::too_many_arguments)]
pub fn gate_g1(
    wte: &[f32],
    vocab: usize,
    fit: &LensFit,
    poles: &[(usize, &[f32])],
    held_out: &[(usize, usize)],
    mask: &RareMask,
    top_k: usize,
    tmp: &mut [f32],
    scores: &mut [f32],
) -> G1Report {
    let mut rep = G1Report {
        poles: poles.len(),
        top_k,
        hit_rate: 0.0,
        control_rate: 0.0,
        unseen: poles.is_empty(),
        pass: false,
    };
    if rep.unseen {
        return rep;
    }
    debug_assert_eq!(poles.len(), held_out.len(), "g1: pole/held-out count");
    let mut hits = 0.0f64;
    let mut control_hits = 0.0f64;
    let mut control_pairs = 0.0f64;
    let mut tops: Vec<Vec<usize>> = Vec::with_capacity(poles.len());
    for (pole, ho) in poles.iter().zip(held_out) {
        s_scores_into(wte, vocab, fit, pole.0, pole.1, tmp, scores);
        let top = top_k_indices(scores, mask, top_k);
        if top.contains(&ho.1) {
            hits += 1.0;
        }
        tops.push(top);
    }
    for (i, top_i) in tops.iter().enumerate() {
        for (j, ho_j) in held_out.iter().enumerate() {
            if i == j {
                continue;
            }
            control_pairs += 1.0;
            if top_i.contains(&ho_j.1) {
                control_hits += 1.0;
            }
        }
    }
    rep.hit_rate = hits / poles.len() as f64;
    rep.control_rate = if control_pairs > 0.0 {
        control_hits / control_pairs
    } else {
        0.0
    };
    rep.pass = rep.hit_rate >= 0.25 && rep.control_rate < 0.01;
    rep
}

/// G2 report: R-read vs measured steering correlation + control.
#[derive(Clone, Copy, Debug)]
pub struct G2Report {
    pub n: usize,
    pub n_controls: usize,
    pub r_mean: f64,
    pub control_mean: f64,
    pub pass: bool,
}

/// **G2** — the R read predicts measured steering. `preds`/`meas` are paired
/// full-vocabulary vectors (the predicted `Δlogits` from
/// [`r_read_delta_into`] vs the measured steering log-prob deltas).
/// Bar: r ≥ 0.45 at the matched depth band, mismatched control within ±0.04.
pub fn gate_g2(
    preds: &[&[f32]],
    meas: &[&[f32]],
    control_preds: &[&[f32]],
    control_meas: &[&[f32]],
) -> G2Report {
    let mut rep = G2Report {
        n: preds.len(),
        n_controls: control_preds.len(),
        r_mean: 0.0,
        control_mean: 0.0,
        pass: false,
    };
    if rep.n > 0 {
        let mut acc = 0.0f64;
        for i in 0..rep.n {
            acc += pearson(preds[i], meas[i]);
        }
        rep.r_mean = acc / rep.n as f64;
    }
    if rep.n_controls > 0 {
        let mut acc = 0.0f64;
        for i in 0..rep.n_controls {
            acc += pearson(control_preds[i], control_meas[i]);
        }
        rep.control_mean = acc / rep.n_controls as f64;
    }
    let control_ok = rep.n_controls == 0 || rep.control_mean.abs() <= 0.04;
    rep.pass = rep.r_mean >= 0.45 && control_ok;
    rep
}

/// G3 report: even/odd half-corpus fit agreement.
#[derive(Clone, Copy, Debug)]
pub struct G3Report {
    pub points: usize,
    pub max_rel_k: f64,
    pub max_rel_j: f64,
    pub pass: bool,
}

/// **G3** — fit stability: even/odd half-corpus fits agree ≤ 6% per point
/// (relative Frobenius difference on both K and J).
pub fn gate_g3(a: &LensFit, b: &LensFit) -> G3Report {
    debug_assert_eq!(a.points.len(), b.points.len(), "g3: point count");
    debug_assert_eq!(a.d, b.d, "g3: d");
    let mut rep = G3Report {
        points: a.points.len(),
        max_rel_k: 0.0,
        max_rel_j: 0.0,
        pass: false,
    };
    for (va, vb) in a.k.iter().zip(&b.k) {
        let rel = rel_frobenius(va, vb);
        if rel > rep.max_rel_k {
            rep.max_rel_k = rel;
        }
    }
    for (va, vb) in a.j.iter().zip(&b.j) {
        let rel = rel_frobenius(va, vb);
        if rel > rep.max_rel_j {
            rep.max_rel_j = rel;
        }
    }
    rep.pass = rep.max_rel_k <= 0.06 && rep.max_rel_j <= 0.06;
    rep
}

/// `‖a − b‖_F / max(‖(a+b)/2‖_F, 1e-12)`.
fn rel_frobenius(a: &[f32], b: &[f32]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for i in 0..a.len() {
        let (x, y) = (a[i] as f64, b[i] as f64);
        num += (x - y) * (x - y);
        let m = 0.5 * (x + y);
        den += m * m;
    }
    if den <= 1e-24 {
        return 0.0;
    }
    num.sqrt() / den.sqrt()
}

// **G4** is structural: the reads write into caller-provided buffers and
// perform exactly one vocab-side matvec per direction (the lens-internal
// `d×d` projection is the direction→lens-space multiply); the zero-alloc
// proof is the counting-allocator gate test, and the fit cost is
// [`FitCost`] (recorded with box state by [`fit_lens`]). No runtime
// function — see `tests::g4_reads_are_zero_alloc`.

// ---------------------------------------------------------------------------
// Tests — reads, gates, G4
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    // -- G4 instrument: a counting global allocator, armed per-thread so
    //    sibling test threads' allocations never pollute the count.
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicUsize, Ordering};

    static ALLOC_CALLS: AtomicUsize = AtomicUsize::new(0);
    thread_local! {
        static COUNTING: Cell<bool> = const { Cell::new(false) };
    }

    struct CountingAlloc;

    unsafe impl GlobalAlloc for CountingAlloc {
        unsafe fn alloc(&self, l: Layout) -> *mut u8 {
            if COUNTING.try_with(|c| c.get()).unwrap_or(false) {
                ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
            }
            unsafe { System.alloc(l) }
        }

        unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
            if COUNTING.try_with(|c| c.get()).unwrap_or(false) {
                ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
            }
            unsafe { System.alloc_zeroed(l) }
        }

        unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
            unsafe { System.dealloc(p, l) }
        }

        unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
            if COUNTING.try_with(|c| c.get()).unwrap_or(false) {
                ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
            }
            unsafe { System.realloc(p, l, n) }
        }
    }

    #[global_allocator]
    static GLOBAL: CountingAlloc = CountingAlloc;

    /// Hand-built lens fixture: d=4, vocab=8. Token 3 aligns with `e0`,
    /// token 5 with `e1`, token 6 with `2·e0` (MASKED in the G1 test).
    fn fixture() -> (Vec<f32>, usize, LensFit) {
        let d = 4usize;
        let vocab = 8usize;
        let mut wte = vec![0.0f32; vocab * d];
        let set_row = |wte: &mut [f32], v: usize, r: [f32; 4]| {
            wte[v * d..(v + 1) * d].copy_from_slice(&r);
        };
        set_row(&mut wte, 0, [0.1, 0.0, 0.2, 0.0]);
        set_row(&mut wte, 1, [0.0, 0.3, 0.0, 0.1]);
        set_row(&mut wte, 2, [-0.2, 0.1, 0.0, 0.2]);
        set_row(&mut wte, 3, [10.0, 0.1, 0.2, 0.0]);
        set_row(&mut wte, 4, [0.0, -0.1, 0.3, 0.1]);
        set_row(&mut wte, 5, [0.2, 10.0, 0.0, 0.3]);
        set_row(&mut wte, 6, [20.0, 0.0, 0.1, 0.0]);
        set_row(&mut wte, 7, [0.0, 0.2, -0.1, 0.4]);
        let points = vec![FitPoint::Mid(0), FitPoint::Out(0)];
        let mut k = vec![vec![0.0f32; d * d]; points.len()];
        for block in &mut k {
            for i in 0..d {
                block[i * d + i] = 1.0; // identity lens
            }
        }
        let j: Vec<Vec<f32>> = k
            .iter()
            .map(|b| b.iter().map(|&x| 2.0 * x).collect())
            .collect();
        let mean_emb = mean_embedding(&wte, vocab, d);
        (
            wte,
            vocab,
            LensFit {
                d,
                points,
                k,
                j,
                mean_emb,
                n_docs: 1,
            },
        )
    }

    #[test]
    fn s_read_matches_direct_definition() {
        let (wte, vocab, fit) = fixture();
        let h = [1.0f32, 0.0, 0.0, 0.0]; // pole along e0
        let mut tmp = vec![0.0f32; fit.d];
        let mut scores = vec![0.0f32; vocab];
        s_scores_into(&wte, vocab, &fit, 0, &h, &mut tmp, &mut scores);
        // Direct: score(v) = ⟨K·(W_E[v] − mean), h⟩ with K = I.
        for v in 0..vocab {
            let row = &wte[v * fit.d..(v + 1) * fit.d];
            let mut dot = 0.0f64;
            for j in 0..fit.d {
                dot += (row[j] - fit.mean_emb[j]) as f64 * h[j] as f64;
            }
            assert!(
                (scores[v] as f64 - dot).abs() < 1e-4,
                "s score {v}: {} vs {dot}",
                scores[v]
            );
        }
        // Token 3 (aligned with e0) outscores the unaligned tokens.
        assert!(scores[3] > scores[0] && scores[3] > scores[1] && scores[3] > scores[7]);
    }

    #[test]
    fn r_read_matches_direct_definition() {
        let (_wte, _vocab, fit) = fixture();
        let vocab = _vocab;
        // lm_head: identity-ish over the first d rows + zero rows beyond.
        let mut head = vec![0.0f32; vocab * fit.d];
        for v in 0..fit.d {
            head[v * fit.d + v] = 1.0;
        }
        let v = [0.5f32, -1.0, 2.0, 0.0];
        let mut tmp = vec![0.0f32; fit.d];
        let mut delta = vec![0.0f32; vocab];
        r_read_delta_into(&head, vocab, &fit, 1, &v, &mut tmp, &mut delta);
        // J = 2·I ⇒ J·v = 2v; head row v (v < d) reads it back.
        for (t, &dv_) in delta.iter().enumerate().take(fit.d) {
            let want = 2.0 * v[t];
            assert!((dv_ - want).abs() < 1e-5, "r delta {t}");
        }
        for &dv_ in delta.iter().skip(fit.d) {
            assert!(dv_.abs() < 1e-6, "r delta head-zero row");
        }
        // Softmax form: base shifts every logit equally → softmax unchanged.
        let base = [1.0f32, 2.0, 3.0, 4.0];
        let mut logits1 = vec![0.0f32; vocab];
        r_read_logits_into(&head, vocab, &fit, 1, &base, &v, &mut tmp, &mut logits1);
        r_read_delta_into(&head, vocab, &fit, 1, &v, &mut tmp, &mut delta);
        for t in 0..vocab {
            assert!((logits1[t] - (base.get(t).copied().unwrap_or(0.0) + delta[t])).abs() < 1e-5);
        }
        let mut sm = vec![0.0f32; vocab];
        softmax_into(&logits1, &mut sm);
        let total: f64 = sm.iter().map(|&x| x as f64).sum();
        assert!((total - 1.0).abs() < 1e-5, "softmax sums to {total}");
    }

    #[test]
    fn top_k_respects_rare_mask() {
        let (wte, vocab, fit) = fixture();
        let h = [1.0f32, 0.0, 0.0, 0.0];
        let mut tmp = vec![0.0f32; fit.d];
        let mut scores = vec![0.0f32; vocab];
        s_scores_into(&wte, vocab, &fit, 0, &h, &mut tmp, &mut scores);
        // Token 6 (2·e0) outscores token 3 but is masked; token 3 must lead.
        let mut counts = vec![1u32; vocab];
        counts[6] = 0;
        let mask = RareMask {
            counts,
            min_count: 1,
        };
        let top = top_k_indices(&scores, &mask, 3);
        assert_eq!(top[0], 3, "masked token 6 must not lead: {top:?}");
        assert!(!top.contains(&6));
        // Unmasked, token 6 leads.
        let mask_all = RareMask::allow_all(vocab);
        let top_all = top_k_indices(&scores, &mask_all, 3);
        assert_eq!(top_all[0], 6);
    }

    #[test]
    fn g1_planted_poles_hit_and_control_misses() {
        let (wte, vocab, fit) = fixture();
        let poles: Vec<(usize, &[f32])> = vec![
            (0usize, &[1.0f32, 0.0, 0.0, 0.0]), // pole 0 along e0 → token 3
            (1usize, &[0.0f32, 1.0, 0.0, 0.0]), // pole 1 along e1 → token 5
        ];
        let held_out = vec![(0usize, 3usize), (1usize, 5usize)];
        let mask = RareMask::allow_all(vocab);
        let mut tmp = vec![0.0f32; fit.d];
        let mut scores = vec![0.0f32; vocab];
        let rep = gate_g1(
            &wte,
            vocab,
            &fit,
            &poles,
            &held_out,
            &mask,
            2,
            &mut tmp,
            &mut scores,
        );
        assert_eq!(rep.poles, 2);
        assert!(!rep.unseen);
        assert!(
            (rep.hit_rate - 1.0).abs() < 1e-9,
            "planted poles must hit: {rep:?}"
        );
        assert!(
            rep.control_rate < 0.01,
            "mismatched control must miss: {rep:?}"
        );
        assert!(rep.pass);
        // UNSEEN: no poles is a scope finding, never a fail.
        let empty: Vec<(usize, &[f32])> = Vec::new();
        let rep0 = gate_g1(
            &wte,
            vocab,
            &fit,
            &empty,
            &[],
            &mask,
            2,
            &mut tmp,
            &mut scores,
        );
        assert!(rep0.unseen && !rep0.pass);
    }

    #[test]
    fn g2_pearson_bars() {
        let perfect: Vec<&[f32]> = vec![&[1.0, 2.0, 3.0], &[0.5, 1.0, 1.5]];
        let measured: Vec<&[f32]> = vec![&[2.0, 4.0, 6.0], &[1.0, 2.0, 3.0]];
        let rep = gate_g2(&perfect, &measured, &[], &[]);
        assert!((rep.r_mean - 1.0).abs() < 1e-9);
        assert!(rep.pass);
        // Mismatched control inside ±0.04 keeps the pass.
        let cp: Vec<&[f32]> = vec![&[1.0, 2.0, 3.0]];
        let cm: Vec<&[f32]> = vec![&[3.0, 2.0, 1.0]]; // r = −1 → control fails
        let rep_bad = gate_g2(&perfect, &measured, &cp, &cm);
        assert!(!rep_bad.pass && rep_bad.control_mean.abs() > 0.04);
        // Orthogonal control (r = 0) passes beside a ≥ 0.45 read.
        let cn: Vec<&[f32]> = vec![&[1.0, -1.0, 1.0, -1.0]];
        let cnm: Vec<&[f32]> = vec![&[1.0, 1.0, -1.0, -1.0]];
        let rep_ok = gate_g2(&perfect, &measured, &cn, &cnm);
        assert!(rep_ok.pass, "{rep_ok:?}");
        // Zero-variance inputs read 0.0, never NaN.
        assert_eq!(pearson(&[1.0, 1.0, 1.0], &[2.0, 3.0, 4.0]), 0.0);
    }

    #[test]
    fn g3_half_corpus_bar() {
        let (_wte, _vocab, mut fit_a) = fixture();
        let mut fit_b = LensFit {
            d: fit_a.d,
            points: fit_a.points.clone(),
            k: fit_a.k.clone(),
            j: fit_a.j.clone(),
            mean_emb: fit_a.mean_emb.clone(),
            n_docs: 2,
        };
        assert!(gate_g3(&fit_a, &fit_b).pass);
        // 5% perturbation on every entry → ≤ 6% bar holds.
        for block in fit_b.k.iter_mut() {
            for x in block.iter_mut() {
                *x *= 1.05;
            }
        }
        assert!(gate_g3(&fit_a, &fit_b).pass, "5% drift passes the 6% bar");
        // 20% perturbation → fails.
        for block in fit_b.k.iter_mut() {
            for x in block.iter_mut() {
                *x *= 1.2;
            }
        }
        let rep = gate_g3(&fit_a, &fit_b);
        assert!(!rep.pass && rep.max_rel_k > 0.06, "{rep:?}");
        let _ = &mut fit_a;
    }

    /// **G4** — the read path performs ZERO allocations with caller-provided
    /// buffers (the counting allocator, armed on this thread only).
    #[test]
    fn g4_reads_are_zero_alloc() {
        let (wte, vocab, fit) = fixture();
        let mut head = vec![0.0f32; vocab * fit.d];
        for v in 0..fit.d {
            head[v * fit.d + v] = 1.0;
        }
        let h = [1.0f32, 0.0, 0.0, 0.0];
        let mut tmp = vec![0.0f32; fit.d];
        let mut out = vec![0.0f32; vocab];
        COUNTING.with(|c| c.set(true));
        ALLOC_CALLS.store(0, Ordering::Relaxed);
        s_scores_into(&wte, vocab, &fit, 0, &h, &mut tmp, &mut out);
        let s_allocs = ALLOC_CALLS.load(Ordering::Relaxed);
        ALLOC_CALLS.store(0, Ordering::Relaxed);
        r_read_delta_into(&head, vocab, &fit, 1, &h, &mut tmp, &mut out);
        let r_allocs = ALLOC_CALLS.load(Ordering::Relaxed);
        COUNTING.with(|c| c.set(false));
        assert_eq!(s_allocs, 0, "s_scores allocated {s_allocs} times");
        assert_eq!(r_allocs, 0, "r_read allocated {r_allocs} times");
    }
}
