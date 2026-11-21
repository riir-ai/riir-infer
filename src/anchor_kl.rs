//! KL-vs-anchor ranked-selection census (riir-infer Issue 1242 T1 — the
//! QDA distill's anchor-KL instrument, riir-refine Research 427 /
//! arXiv:2611.23388): score candidate checkpoints by their divergence from
//! a fixed ANCHOR model over a fixed trace set, in minutes instead of a
//! full eval — then rank candidates by that divergence and verify the rank
//! predicts the full-eval ranking (the proxy's purpose).
//!
//! The anchor is the REFERENCE quantization whose quality the fleet already
//! trusts (e.g. the incumbent W4-class backbone beside a candidate's own
//! requant); the KL direction is **anchor → candidate** (mean over the
//! trace set's positions): how far the candidate wanders from the trusted
//! read. QDA's published finding: KL-vs-anchor ranks candidates in
//! 350–990× less compute than ground-truth eval; the failure mode the
//! ranking misses (loops) is the gencheck axis (1242 T2).
//!
//! The math half is here: per-token forward-KL from logits slices, the
//! top-k variant (mass over the union of the anchor's and the candidate's
//! top-k tokens — the tail beyond k must not vanish from the divergence),
//! and the trace-set aggregation with pinned fold order (the katgpt-rs
//! Issue-940 determinism law: sequential ascending folds, first-index tie
//! rules). The forward wiring rides the census bins (the anchor's logits
//! are computed once per (model bytes, trace set, dtype) and cached
//! content-addressed — the cached-anchor shape golden_nll established).
//!
//! Instrument-grade only — never a serving path.

/// Top-k width for the truncated variant (pinned; QDA's top-k class).
pub const TOP_K: usize = 64;

/// One scored position's reading: forward KL(anchor ‖ candidate) over the
/// full vocab (nats) and over the top-k union (nats), both in f64.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnchorKlReading {
    pub kl_full: f64,
    pub kl_topk: f64,
}

/// Forward KL(anchor ‖ candidate) = Σᵢ pᵢ · (log pᵢ − log qᵢ) where p is the
/// anchor's softmax and q the candidate's, computed in the **log domain**
/// (logits in; no softmax materialized — the log-sum-exp form is exact for
/// any logits and never overflows):
///
/// ```text
/// KL = LSE_p − LSE_q + Σᵢ softmax_p(i) · (q_logitᵢ − p_logitᵢ) / 1
///    = LSE_p + Σᵢ pᵢ · (qᵢ − pᵢ)   [in logit space, folded ascending]
/// ```
///
/// Wait — the stable form actually used (and fixture-pinned): gather
/// `lse_p = LSE(p)`, `lse_q = LSE(q)`, then
/// `KL = Σ exp(p − lse_p) · ((p − lse_p) − (q − lse_q))` with the fold
/// sequential ascending (the Issue-940 law). The scratch widens both rows
/// once (the golden_nll scratch pattern — caller-owned, zero steady-state).
pub fn anchor_kl_full(
    anchor_logits: &[f32],
    candidate_logits: &[f32],
    scratch: &mut AnchorKlScratch,
) -> Option<AnchorKlReading> {
    if anchor_logits.len() != candidate_logits.len() || anchor_logits.is_empty() {
        return None;
    }
    let n = anchor_logits.len();

    // LSE both rows (max-shift form, ascending folds, f64).
    let (mut mx_p, mut mx_q) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for i in 0..n {
        mx_p = mx_p.max(anchor_logits[i] as f64);
        mx_q = mx_q.max(candidate_logits[i] as f64);
    }
    let mut acc_p = 0.0f64;
    let mut acc_q = 0.0f64;
    for i in 0..n {
        acc_p += (anchor_logits[i] as f64 - mx_p).exp();
        acc_q += (candidate_logits[i] as f64 - mx_q).exp();
    }
    let lse_p = mx_p + acc_p.ln();
    let lse_q = mx_q + acc_q.ln();

    // KL fold: Σ p̂ · ((p̂log) − (q̂log)), p̂ = exp(p − lse_p).
    let mut kl = 0.0f64;
    for i in 0..n {
        let pl = anchor_logits[i] as f64 - lse_p;
        let ql = candidate_logits[i] as f64 - lse_q;
        kl += pl.exp() * (pl - ql);
    }
    scratch.last_full = kl;

    // Top-k union variant: take the anchor's top-k and the candidate's top-k
    // indices (first-index tie rule via the descending stable sort), union
    // them, and fold the restricted KL over exactly those positions — the
    // mass the truncated read keeps. The complement mass difference is the
    // truncation error, reported alongside never hidden.
    let mut order_p: Vec<usize> = (0..n).collect();
    order_p.sort_by(|&a, &b| {
        let x = anchor_logits[b];
        let y = anchor_logits[a];
        y.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut order_q: Vec<usize> = (0..n).collect();
    order_q.sort_by(|&a, &b| {
        let x = candidate_logits[b];
        let y = candidate_logits[a];
        y.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Equal)
    });
    let k = TOP_K.min(n);
    let mut in_union = vec![false; n];
    for &i in &order_p[..k] {
        in_union[i] = true;
    }
    for &i in &order_q[..k] {
        in_union[i] = true;
    }

    // Restricted LSEs over the union (both rows).
    let (mut mx_pu, mut mx_qu) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for i in 0..n {
        if in_union[i] {
            mx_pu = mx_pu.max(anchor_logits[i] as f64);
            mx_qu = mx_qu.max(candidate_logits[i] as f64);
        }
    }
    let (mut acc_pu, mut acc_qu) = (0.0f64, 0.0f64);
    for i in 0..n {
        if in_union[i] {
            acc_pu += (anchor_logits[i] as f64 - mx_pu).exp();
            acc_qu += (candidate_logits[i] as f64 - mx_qu).exp();
        }
    }
    let lse_pu = mx_pu + acc_pu.ln();
    let lse_qu = mx_qu + acc_qu.ln();

    let mut kl_k = 0.0f64;
    for i in 0..n {
        if in_union[i] {
            let pl = anchor_logits[i] as f64 - lse_pu;
            let ql = candidate_logits[i] as f64 - lse_qu;
            kl_k += pl.exp() * (pl - ql);
        }
    }
    scratch.last_topk = kl_k;
    scratch.last_topk_union = in_union.iter().filter(|&&b| b).count();

    Some(AnchorKlReading {
        kl_full: kl,
        kl_topk: kl_k,
    })
}

/// Caller-owned scratch (the golden_nll pattern — allocate once per
/// harness; the last reading's aggregates ride here for the harness loop).
#[derive(Debug, Default, Clone)]
pub struct AnchorKlScratch {
    pub last_full: f64,
    pub last_topk: f64,
    /// Union size of the last top-k read (≤ 2·TOP_K).
    pub last_topk_union: usize,
}

/// The trace-set aggregation: mean KL over positions, sequential ascending
/// fold (the Issue-940 law). Ranked selection sorts candidates by this mean
/// ascending (closest to the anchor first — QDA's pick rule).
pub fn mean_kl(kls: &[f64]) -> f64 {
    if kls.is_empty() {
        return f64::NAN;
    }
    let mut acc = 0.0f64;
    for v in kls {
        acc += v;
    }
    acc / kls.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * b.abs().max(1e-12)
    }

    #[test]
    fn identical_logits_read_zero_kl() {
        let a = [0.1f32, -2.0, 3.0, 0.0, 7.5];
        let mut s = AnchorKlScratch::default();
        let r = anchor_kl_full(&a, &a, &mut s).expect("reads");
        assert!(approx(r.kl_full, 0.0, 1e-12), "kl_full={}", r.kl_full);
        assert!(approx(r.kl_topk, 0.0, 1e-12), "kl_topk={}", r.kl_topk);
    }

    #[test]
    fn known_small_kl_oracle() {
        // Two-position hand-computable case: anchor (log0.7, log0.3),
        // candidate (log0.5, log0.5) — the canonical Bernoulli KL.
        // KL(p‖q) = 0.7·ln(0.7/0.5) + 0.3·ln(0.3/0.5) ≈ 0.08377
        let a = [0.7f64.ln() as f32, 0.3f64.ln() as f32];
        let b = [0.5f64.ln() as f32, 0.5f64.ln() as f32];
        let mut s = AnchorKlScratch::default();
        let r = anchor_kl_full(&a, &b, &mut s).expect("reads");
        let expect = 0.7 * (0.7f64 / 0.5).ln() + 0.3 * (0.3f64 / 0.5).ln();
        assert!(
            approx(r.kl_full, expect, 1e-6),
            "kl={} expect={}",
            r.kl_full,
            expect
        );
    }

    #[test]
    fn kl_is_nonnegative_and_asymmetric() {
        // Forward KL is >= 0 and KL(p‖q) != KL(q‖p).
        let p = [2.0f32, 0.0, -1.0, 4.0, 1.0, 0.5];
        let q = [-1.0f32, 3.0, 2.0, 0.0, 1.5, 0.2];
        let mut s = AnchorKlScratch::default();
        let rp = anchor_kl_full(&p, &q, &mut s).expect("reads");
        let rq = anchor_kl_full(&q, &p, &mut s).expect("reads");
        assert!(rp.kl_full >= 0.0);
        assert!(rq.kl_full >= 0.0);
        assert!(!approx(rp.kl_full, rq.kl_full, 1e-6));
    }

    #[test]
    fn topk_union_covers_both_peaks() {
        // Anchor peaks at index 0, candidate at index 5 (disjoint peaks)
        // over a vocab WIDER than TOP_K, so the truncated read must
        // actively include both peaks in its union.
        let n = 80usize;
        let mut a = vec![-50f32; n];
        a[0] = 10.0;
        let mut b = vec![-50f32; n];
        b[5] = 10.0;
        let mut s = AnchorKlScratch::default();
        let r = anchor_kl_full(&a, &b, &mut s).expect("reads");
        assert!(s.last_topk_union >= 2 && s.last_topk_union <= 2 * TOP_K);
        assert!(
            s.last_topk_union < n,
            "a truncated union must not be the whole vocab: {}",
            s.last_topk_union
        );
        // The union-KL is dominated by the disjoint peaks: large.
        assert!(r.kl_topk > 5.0, "kl_topk={}", r.kl_topk);
    }

    #[test]
    fn mismatched_lengths_refuse() {
        let mut s = AnchorKlScratch::default();
        assert!(anchor_kl_full(&[1.0], &[1.0, 2.0], &mut s).is_none());
        assert!(anchor_kl_full(&[], &[], &mut s).is_none());
    }

    #[test]
    fn mean_kl_folds_sequentially() {
        // The Issue-940 fold: mean([a,b,c]) == (a+b+c)/3 exactly at these
        // magnitudes — the fixture pins the ordering, not statistics.
        let v = [1.0f64, 2.0, 3.0];
        assert!(approx(mean_kl(&v), 2.0, 1e-15));
        assert!(mean_kl(&[]).is_nan());
    }

    #[test]
    fn constants_are_pinned() {
        assert_eq!(TOP_K, 64);
    }
}
