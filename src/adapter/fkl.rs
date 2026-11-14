//! Issue 1208 T6 — the forward-KL audit arm (`--fk-lab`): KL(base ∥
//! masked) over a fixed probe walk, the closed-form quality axis the
//! T7 τ-sweep consumes (the paper's retention metric, on our stack).
//!
//! Semantics (pinned): at each walk position the base and masked
//! logits are softmaxed over the SAME vocabulary, and the position's
//! contribution is `KL(base ∥ masked) = Σ_v p_b(v)·(ln p_b(v) − ln
//! p_m(v))` — forward KL, mass where the BASE is confident (a masked
//! position the base never attends to costs nothing; the reverse
//! ordering would score where the MASKED model concentrates — the
//! wrong axis for "did deletion move my outputs"). Positions fold in
//! sequential walk order (the Issue-940 determinism law; the fold is
//! plain `+=` in f64, never a platform-lane SIMD kernel).
//!
//! Report-only, exit 0. Goldens: identical distributions → exactly 0;
//! a hand-computed perturbation → the closed-form value.

/// Log-softmax in f64 over one position's logits (the max-shift is the
/// stable form; the result is the ln-probability vector).
fn log_softmax_f64(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().fold(f64::MIN, |m, v| m.max(*v as f64));
    let mut out: Vec<f64> = logits.iter().map(|v| *v as f64 - max).collect();
    let lse: f64 = out.iter().map(|v| v.exp()).sum::<f64>().ln();
    for v in &mut out {
        *v -= lse;
    }
    out
}

/// One position's forward-KL contribution in nats: KL(base ∥ masked)
/// over the softmaxed distributions. `base` and `masked` must be the
/// same length (the same vocabulary — the caller's contract; a walk
/// through one model cannot disagree with itself).
pub fn position_forward_kl(base: &[f32], masked: &[f32]) -> f64 {
    debug_assert_eq!(base.len(), masked.len(), "KL needs the same vocabulary");
    let pb = log_softmax_f64(base);
    let pm = log_softmax_f64(masked);
    let mut kl = 0.0f64;
    for (lb, lm) in pb.iter().zip(pm.iter()) {
        let p = lb.exp();
        kl += p * (lb - lm);
    }
    kl
}

/// The walk-level audit: per-position KL folded in order, plus the
/// mean / percentile tail. Positions are consecutive (`walk_base[i]`
/// is position i's full logits slice). The percentile index is the
/// ceiling form (`ceil(p·n) − 1`, never `n·p` truncation — the
/// percentile law: `n·p as usize` lands on the MAX at small n; tail
/// support `n − idx` is printed so a weak tail is visible).
pub struct KlReport {
    pub per_position: Vec<f64>,
    pub mean: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    /// `n − idx` for p99 — under 10 is a weak tail (print, never hide).
    pub p99_tail_support: usize,
    /// BLAKE3 over the per-position table (LE f64 bytes, in order) —
    /// the determinism pin (same walk → same digest).
    pub table_digest: [u8; 32],
}

pub fn forward_kl_walk<B: AsRef<[f32]>, M: AsRef<[f32]>>(
    walk_base: &[B],
    walk_masked: &[M],
) -> KlReport {
    assert_eq!(
        walk_base.len(),
        walk_masked.len(),
        "the two walks must cover the same positions"
    );
    let per_position: Vec<f64> = walk_base
        .iter()
        .zip(walk_masked.iter())
        .map(|(b, m)| position_forward_kl(b.as_ref(), m.as_ref()))
        .collect();
    let n = per_position.len();
    let mut sorted = per_position.clone();
    sorted.sort_by(f64::total_cmp);
    let idx = |p: f64| -> usize { ((p * n as f64).ceil() as usize).max(1) - 1 };
    let p50 = sorted[idx(0.50)];
    let p95 = sorted[idx(0.95)];
    let p99 = sorted[idx(0.99)];
    let mean = per_position.iter().sum::<f64>() / n.max(1) as f64;
    let mut hasher = blake3::Hasher::new();
    for v in &per_position {
        hasher.update(&v.to_le_bytes());
    }
    KlReport {
        p99_tail_support: n - idx(0.99),
        per_position,
        mean,
        p50,
        p95,
        p99,
        table_digest: *hasher.finalize().as_bytes(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn identical_distributions_give_exactly_zero() {
        let logits: Vec<f32> = (0..64).map(|i| (i as f32 * 0.37).sin()).collect();
        assert_eq!(position_forward_kl(&logits, &logits), 0.0);
    }

    #[test]
    fn hand_computed_perturbation_matches_closed_form() {
        // base logits [ln 4, ln 1] → p_b = [0.8, 0.2]
        // masked logits [ln 1, ln 1] → p_m = [0.5, 0.5]
        // KL = 0.8·ln(0.8/0.5) + 0.2·ln(0.2/0.5)
        // (the f32 inputs bound the comparison at ~1e-8: the f64 math
        // downstream is exact, the INPUTS carry f32 rounding)
        let base = [4.0f32.ln(), 1.0f32.ln()];
        let masked = [1.0f32, 1.0f32];
        let want = 0.8 * (0.8f64 / 0.5).ln() + 0.2 * (0.2f64 / 0.5).ln();
        let got = position_forward_kl(&base, &masked);
        assert!((got - want).abs() < 1e-8, "got {got}, want {want}");
    }

    #[test]
    fn kl_is_nonnegative_and_asymmetric() {
        // KL(base||masked) >= 0 always; the reverse ordering differs
        let base = [3.0f32, 0.1, -2.0];
        let masked = [0.5, 2.0, 0.5];
        let fwd = position_forward_kl(&base, &masked);
        let rev = position_forward_kl(&masked, &base);
        assert!(fwd >= 0.0 && rev >= 0.0);
        assert!((fwd - rev).abs() > 1e-6, "KL must be asymmetric");
    }

    #[test]
    fn walk_report_goldens_and_digest_determinism() {
        // two positions: the first identical (KL 0), the second the
        // hand-computed perturbation — the mean is the closed form's half
        let base_p1 = [4.0f32.ln(), 1.0f32.ln()];
        let masked_p1 = [1.0f32, 1.0f32];
        let walk_base = vec![base_p1, base_p1];
        let walk_masked = vec![base_p1, masked_p1];
        let report = forward_kl_walk(&walk_base, &walk_masked);
        assert_eq!(report.per_position[0], 0.0);
        let want = 0.8 * (0.8f64 / 0.5).ln() + 0.2 * (0.2f64 / 0.5).ln();
        assert!((report.per_position[1] - want).abs() < 1e-8);
        assert!((report.mean - want / 2.0).abs() < 1e-8);
        // percentiles on n=2 (ceil form): idx(p) = ceil(p·n)−1
        //   p99 → ceil(1.98)−1 = 1 → the max;  p95 → ceil(1.9)−1 = 1;
        //   p50 → ceil(1.0)−1 = 0 → the lower element (here 0.0 — the
        //   identical first position)
        assert_eq!(report.p99, report.per_position[1]);
        assert_eq!(report.p95, report.per_position[1]);
        assert_eq!(report.p50, report.per_position[0]);
        assert_eq!(report.p99_tail_support, 1);
        // determinism: the same walk digests identically
        let again = forward_kl_walk(&walk_base, &walk_masked);
        assert_eq!(hex(&report.table_digest), hex(&again.table_digest));
        // and a different walk moves it
        let moved = forward_kl_walk(&walk_base, &[masked_p1, masked_p1]);
        assert_ne!(hex(&report.table_digest), hex(&moved.table_digest));
    }
}
