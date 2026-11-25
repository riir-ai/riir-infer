//! The HiSA rotation-suppression census arm (riir-infer Issue 1244 T2.1;
//! Research 040, arXiv:2611.26363).
//!
//! REPORT-ONLY instrument: an offline diagnostic beside
//! [`crate::hisa_plan`]'s closed-form prior — the planner says WHICH
//! (chunk, band) windows the geometry predicts free to suppress; this
//! census MEASURES whether the model carries signal there. Zero
//! serving-path code; nothing synced; the armed posture is never a
//! production boot (the census driver is the only producer — the scratch
//! field compiles out of every other build).
//!
//! # The experiment
//!
//! Per (full-attention layer, 16-feature rotating chunk, log-Δ distance
//! band) the arm ablates the chunk's contribution to the attention
//! scores on the band's key positions and the driver reads, over ONE
//! frozen token read:
//!
//! - pooled teacher-forced Δ-NLL + argmax flips (effect size; the NLL
//!   math is the driver's — f64 positive-summand folds per Issue 940),
//! - the band's pooled softmax mass under three postures: [`HisaMode::Observe`]
//!   (incumbent-identical scores — the E_real base), [`HisaMode::Remove`]
//!   (the chunk zeroed on the band — the suppression posture),
//!   [`HisaMode::Replace`] (the chunk's partial dot swapped for the null
//!   document's same-position key row — the E_null noise floor).
//!   `E_remove/E_observe` and `E_replace/E_observe` are the paper's
//!   suppression-vs-noise read; a `no_suppression` verdict (T3) closes
//!   the kernel lane loud.
//!
//! # Bit-identity law
//!
//! Out-of-band scores are `simd_dot_f32(q, k, hd) * scale` — the exact
//! `attention_heads_parallel_bias` (softcap = 0, no bias) arithmetic in
//! the same order; the identity gate (an Observe arm, or a band beyond
//! the read) reproduces the incumbent bitwise, head_scores included, in
//! BOTH dispatch regimes. Only in-band rows carry the ablation
//! perturbation. The short-sequence threshold is a LOCAL mirror of
//! transformer-attention's `PARALLEL_HEADS_MIN_SEQ` (`pub(super)`
//! there — invisible here); the two-regime identity tests pin the pair,
//! so an upstream drift reds the gate instead of silently splitting the
//! census from the model it measures.
//!
//! # Scope
//!
//! The TERNARY attention body only (v1) — the Issue-1244 postures target
//! the Bonsai ternary forward; the dense body's mirror is a follow-up if
//! a dense posture is wanted. The FlashMemory decode body refuses the
//! arm (its selected-token softmax is undefined under a band ablation).
//! Scale-invariant and HiSA are mutually exclusive (a full-row logit
//! affine over an ablated softmax is undefined); the dispatch asserts it
//! when both features compile.
//!
//! # Sync boundary
//!
//! None — an offline diagnostic. Deterministic by construction: heads
//! ascending, keys ascending, f32 score arithmetic in the incumbent's
//! order, energy folded sequentially in f64 (Issue 940: a local
//! instrument, fixed fold order, no RNG, no wall clock).

use std::sync::Arc;

/// The census posture of the score path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HisaMode {
    /// Incumbent-identical scores; band mass accumulates per band cell.
    /// The E_real base — one pass per LAYER covers every band of the
    /// posture (the scores are unmodified, so the band grid is free).
    Observe,
    /// In-band keys lose the chunk's partial dot (the suppression op).
    Remove,
    /// In-band keys' chunk partial dot replaced by the null document's
    /// same-position key row (the noise-floor control).
    Replace,
}

/// One distance band of an [`HisaMode::Observe`] arm's accumulator.
#[derive(Clone, Copy, Debug)]
pub struct HisaBandCell {
    /// Half-open query→key distance band `[lo, hi)`; the query sits at
    /// `t_n − 1`, so key slot `t` carries distance `t_n − 1 − t`.
    pub band: (usize, usize),
    /// Pooled softmax mass, f64, sequential folds (heads ascending,
    /// keys ascending).
    pub energy: f64,
    /// In-band (head, key) samples pooled into `energy`.
    pub count: u64,
}

/// The arm — census CONFIG + the band-energy accumulator riding
/// [`crate::deltanet::forward::AttentionLayerScratch`] (the
/// `scale_invariant` precedent: session config, not a buffer; the census
/// driver owns the only armed copy, every other construction leaves the
/// field `None`).
#[derive(Debug)]
pub struct HisaCensusArm {
    /// The probed full-attention layer. The dispatch consults this; every
    /// other layer runs the incumbent path untouched.
    pub layer: usize,
    pub mode: HisaMode,
    /// `Observe`: the band cells (disjoint, ascending by convention — the
    /// kernel records into the FIRST matching band).
    pub bands: Vec<HisaBandCell>,
    /// `Remove`/`Replace`: half-open feature range of the rotating chunk
    /// within one head (`[lo, hi)`; partial RoPE rotates `[0, rotary_dim)`
    /// of every head, so the 16-feature chunks tile that range).
    pub chunk: (usize, usize),
    /// `Remove`/`Replace`: the target distance band `[lo, hi)`.
    pub band: (usize, usize),
    /// `Replace`: the null document's post-RoPE key rows for the probed
    /// layer, flat `[t_null × kv_dim]` — the exact `cache.key` layout.
    /// Read-only (shared `Arc`).
    pub null_keys: Option<Arc<Vec<f32>>>,
    /// Valid null rows. In-band keys at `t >= t_null` keep their incumbent
    /// score and count in [`Self::swapped_skipped`] — the report discloses
    /// a nonzero count as a driver bug (the constructor + driver guarantee
    /// `t_null >= t_n` for every step of the read).
    pub t_null: usize,
    /// `Remove`/`Replace` pooled band mass (f64, sequential folds).
    pub energy: f64,
    /// In-band (head, key) samples pooled into `energy`.
    pub count: u64,
    /// `Replace` only: in-band keys with no null row (tripwire — expect 0).
    pub swapped_skipped: u64,
}

impl HisaCensusArm {
    /// The E_real base arm: incumbent-identical scores, every band of the
    /// posture accumulated in ONE pass.
    pub fn observe(layer: usize, bands: Vec<(usize, usize)>) -> Self {
        Self {
            layer,
            mode: HisaMode::Observe,
            bands: bands
                .into_iter()
                .map(|band| HisaBandCell {
                    band,
                    energy: 0.0,
                    count: 0,
                })
                .collect(),
            chunk: (0, 0),
            band: (0, 0),
            null_keys: None,
            t_null: 0,
            energy: 0.0,
            count: 0,
            swapped_skipped: 0,
        }
    }

    /// The suppression arm: the chunk is zeroed on the band.
    pub fn remove(layer: usize, chunk: (usize, usize), band: (usize, usize)) -> Self {
        Self {
            layer,
            mode: HisaMode::Remove,
            bands: Vec::new(),
            chunk,
            band,
            null_keys: None,
            t_null: 0,
            energy: 0.0,
            count: 0,
            swapped_skipped: 0,
        }
    }

    /// The noise-floor arm: the chunk's partial dot on the band is swapped
    /// for the null document's same-position key row. `null_keys` must be
    /// the probed layer's `cache.key` layout (`t_null × kv_dim` flat);
    /// the read's every step stays `<= t_null` (validated here by length,
    /// at the driver by construction).
    pub fn replace(
        layer: usize,
        chunk: (usize, usize),
        band: (usize, usize),
        null_keys: Arc<Vec<f32>>,
        t_null: usize,
        kv_dim: usize,
    ) -> Result<Self, String> {
        if t_null == 0 {
            return Err("hisa_census: replace arm with t_null == 0 — no null rows to swap".into());
        }
        if null_keys.len() != t_null * kv_dim {
            return Err(format!(
                "hisa_census: null key buffer is {} floats, expected t_null({t_null}) × kv_dim({kv_dim}) = {}",
                null_keys.len(),
                t_null * kv_dim
            ));
        }
        Ok(Self {
            layer,
            mode: HisaMode::Replace,
            bands: Vec::new(),
            chunk,
            band,
            null_keys: Some(null_keys),
            t_null,
            energy: 0.0,
            count: 0,
            swapped_skipped: 0,
        })
    }

    /// Is key slot `t` (query at `t_n − 1`) inside this arm's target band?
    #[inline]
    fn targets(&self, t: usize, t_n: usize) -> bool {
        let d = t_n - 1 - t;
        match self.mode {
            HisaMode::Observe => self.bands.iter().any(|c| d >= c.band.0 && d < c.band.1),
            HisaMode::Remove | HisaMode::Replace => d >= self.band.0 && d < self.band.1,
        }
    }

    /// The `Observe` band index carrying distance `d` (first match).
    #[inline]
    fn observe_band_index(&self, d: usize) -> Option<usize> {
        self.bands
            .iter()
            .position(|c| d >= c.band.0 && d < c.band.1)
    }

    /// The modified in-band score (`Remove`/`Replace` only). `full` is the
    /// incumbent-order full-head dot; the chunk partials are second-pass
    /// sums over exactly the chunk's dims — the ablation is the
    /// perturbation, the out-of-band rows stay incumbent-exact. Slice
    /// bounds: `key_cache[k_off..][..head_dim]` per the kernel's contract;
    /// the Replace null-row bound is the constructor + driver's
    /// `t_null >= t_n` (a skipped swap surfaces in `swapped_skipped`, it
    /// never panics inside the forward).
    #[allow(clippy::too_many_arguments)]
    fn modified_score(
        &mut self,
        t: usize,
        kv_dim: usize,
        kv_group_offset: usize,
        q_head: &[f32],
        full: f32,
        scale: f32,
        key_cache: &[f32],
    ) -> f32 {
        let (lo, hi) = self.chunk;
        let k_off = t * kv_dim + kv_group_offset;
        let chunk_real =
            crate::simd::simd_dot_f32(&q_head[lo..hi], &key_cache[k_off + lo..k_off + hi], hi - lo);
        match self.mode {
            HisaMode::Remove => (full - chunk_real) * scale,
            HisaMode::Replace => {
                let null = self
                    .null_keys
                    .as_deref()
                    .expect("replace arm without null keys — constructor invariant");
                if t < self.t_null {
                    let n_off = t * kv_dim + kv_group_offset;
                    let chunk_null = crate::simd::simd_dot_f32(
                        &q_head[lo..hi],
                        &null[n_off + lo..n_off + hi],
                        hi - lo,
                    );
                    (full - chunk_real + chunk_null) * scale
                } else {
                    // No null row at this distance: keep the incumbent score
                    // and count the skip — the report discloses a nonzero
                    // count as a driver bug, never a silent mix.
                    self.swapped_skipped += 1;
                    full * scale
                }
            }
            // Observe never dispatches here (the kernel guards on the mode).
            HisaMode::Observe => full * scale,
        }
    }

    /// Record one normalized weight into the arm's accumulator.
    #[inline]
    fn record(&mut self, t: usize, t_n: usize, w: f32) {
        let d = t_n - 1 - t;
        match self.mode {
            HisaMode::Observe => {
                if let Some(idx) = self.observe_band_index(d) {
                    self.bands[idx].energy += w as f64;
                    self.bands[idx].count += 1;
                }
            }
            HisaMode::Remove | HisaMode::Replace => {
                self.energy += w as f64;
                self.count += 1;
            }
        }
    }
}

/// The census attention kernel — the incumbent
/// `attention_heads_parallel_bias` (softcap 0, no bias) arithmetic with
/// the arm's ablation on the in-band rows and the band-mass accumulator.
/// Per-head math is identical to the incumbent in BOTH dispatch regimes;
/// heads run sequentially here (determinism first — the probed layer is
/// one of 64 and the prefill is dominated by the MLP/GDN blocks).
///
/// # Safety
/// The `attention_heads_parallel_bias` contract: `q.len() >= n_head *
/// head_dim`; `key_cache`/`value_cache` sized `t_n * kv_dim` with
/// `n_kv_head * head_dim <= kv_dim`; `attn_out.len() >= n_head * head_dim`
/// non-aliasing (zeroed by the caller, accumulated with `+=` like the
/// incumbent); `head_scores.len() >= n_head * block_size`; `t_n <=
/// block_size`. In the `Replace` posture the driver additionally
/// guarantees `arm.t_null >= t_n` (in-band rows below it fall back to the
/// incumbent score and count in `swapped_skipped` — never a panic).
///
/// The score region follows the incumbent's regime: short sequences
/// (`t_n < 512` or one head) share region 0 across heads (last head
/// wins), longer sequences give each head its own block — mirrored, and
/// pinned by the identity tests in both regimes.
#[allow(clippy::too_many_arguments)]
pub unsafe fn attention_heads_hisa(
    q: &[f32],
    key_cache: &[f32],
    value_cache: &[f32],
    attn_out: &mut [f32],
    head_scores: &mut [f32],
    n_head: usize,
    n_kv_head: usize,
    kv_dim: usize,
    head_dim: usize,
    t_n: usize,
    scale: f32,
    block_size: usize,
    arm: &mut HisaCensusArm,
) {
    // Local mirror of transformer-attention's PARALLEL_HEADS_MIN_SEQ
    // (pub(super) there). A drift upstream flips the incumbent's regime at
    // the boundary; the identity tests at 37 and 513 keys straddle it and
    // red instead of letting the census diverge from the model it measures.
    const HISA_PARALLEL_HEADS_MIN_SEQ: usize = 512;
    let short_regime = t_n < HISA_PARALLEL_HEADS_MIN_SEQ || n_head <= 1;
    for h in 0..n_head {
        let q_off = h * head_dim;
        let kv_group_offset = (h * n_kv_head / n_head) * head_dim;
        let q_head = &q[q_off..q_off + head_dim];
        let score_base = if short_regime { 0 } else { h * block_size };
        // Pass 1: incumbent-order full dots; the ablation lands only on
        // in-band rows of a Remove/Replace arm (Observe is score-identical).
        let mut max_score = f32::NEG_INFINITY;
        for t in 0..t_n {
            let k_off = t * kv_dim + kv_group_offset;
            let full =
                crate::simd::simd_dot_f32(q_head, &key_cache[k_off..k_off + head_dim], head_dim);
            let score = if arm.mode != HisaMode::Observe && arm.targets(t, t_n) {
                arm.modified_score(t, kv_dim, kv_group_offset, q_head, full, scale, key_cache)
            } else {
                full * scale
            };
            unsafe {
                *head_scores.get_unchecked_mut(score_base + t) = score;
            }
            max_score = max_score.max(score);
        }
        // Pass 2: exp(scores − max), sum — the incumbent shape and order.
        let mut sum = 0.0f32;
        for t in 0..t_n {
            let slot = unsafe { head_scores.get_unchecked_mut(score_base + t) };
            let e = (*slot - max_score).exp();
            *slot = e;
            sum += e;
        }
        // Pass 3: normalize + band-mass accumulation + weighted value sum,
        // one fixed order (heads ascending, keys ascending; f64 folds).
        let inv_sum = 1.0 / sum;
        for t in 0..t_n {
            let w = unsafe { *head_scores.get_unchecked(score_base + t) * inv_sum };
            if arm.targets(t, t_n) {
                arm.record(t, t_n, w);
            }
            let v_off = t * kv_dim + kv_group_offset;
            for d in 0..head_dim {
                unsafe {
                    *attn_out.get_unchecked_mut(q_off + d) +=
                        w * *value_cache.get_unchecked(v_off + d);
                }
            }
        }
    }
}

#[cfg(all(test, feature = "hisa_census"))]
mod tests {
    use super::*;

    /// Deterministic LCG fill — no RNG crate, no global state (Issue 940).
    fn lcg_fill(len: usize, seed: u64) -> Vec<f32> {
        let mut s = seed | 1;
        (0..len)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let v = ((s >> 33) as f32 / (u32::MAX as f32)) * 2.0 - 1.0;
                if v == 0.0 { 0.5 } else { v }
            })
            .collect()
    }

    fn assert_bitwise(a: &[f32], b: &[f32], what: &str) {
        assert_eq!(a.len(), b.len(), "{what} length");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{what}[{i}]: {x} vs {y} — the census kernel must reproduce the incumbent bit-exactly off the ablated path"
            );
        }
    }

    /// The incumbent the ternary body dispatches (softcap 0, no bias).
    #[allow(clippy::too_many_arguments)]
    unsafe fn run_incumbent(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        out: &mut [f32],
        scores: &mut [f32],
        n_head: usize,
        n_kv: usize,
        kv_dim: usize,
        hd: usize,
        t_n: usize,
        scale: f32,
        block: usize,
    ) {
        unsafe {
            crate::transformer::attention_heads_parallel_bias(
                q, k, v, out, scores, n_head, n_kv, kv_dim, hd, t_n, scale, 0.0, block, None,
            );
        }
    }

    /// G1 identity, short regime (37 keys < 512, 3 heads): an Observe arm
    /// must be score-transparent — attn_out AND head_scores bitwise.
    #[test]
    fn observe_short_regime_matches_incumbent_bitwise() {
        let n_head = 3;
        let n_kv = 1;
        let hd = 8;
        let kv_dim = n_kv * hd;
        let block = 1024;
        let t_n = 37;
        let scale = 1.0 / (hd as f32).sqrt();
        let q = lcg_fill(n_head * hd, 1);
        let k = lcg_fill(t_n * kv_dim, 2);
        let v = lcg_fill(t_n * kv_dim, 3);

        let mut o1 = vec![0.0; n_head * hd];
        let mut s1 = vec![0.0; n_head * block];
        unsafe {
            run_incumbent(
                &q, &k, &v, &mut o1, &mut s1, n_head, n_kv, kv_dim, hd, t_n, scale, block,
            );
        }

        let mut arm = HisaCensusArm::observe(7, vec![(0, 1), (1, 2), (2, 4), (8, 16), (32, 64)]);
        let mut o2 = vec![0.0; n_head * hd];
        let mut s2 = vec![0.0; n_head * block];
        unsafe {
            attention_heads_hisa(
                &q, &k, &v, &mut o2, &mut s2, n_head, n_kv, kv_dim, hd, t_n, scale, block, &mut arm,
            );
        }
        assert_bitwise(&o1, &o2, "attn_out");
        assert_bitwise(&s1, &s2, "head_scores");
        // Every band of the read carried at least one key — the accumulator
        // is live, not vacuously green.
        for (i, cell) in arm.bands.iter().enumerate() {
            assert!(cell.count > 0, "band {i} recorded nothing");
        }
    }

    /// G1 identity, parallel regime (513 keys >= 512): per-head score
    /// regions this time — still bitwise on both outputs.
    #[test]
    fn observe_parallel_regime_matches_incumbent_bitwise() {
        let n_head = 3;
        let n_kv = 1;
        let hd = 8;
        let kv_dim = n_kv * hd;
        let block = 1024;
        let t_n = 513;
        let scale = 1.0 / (hd as f32).sqrt();
        let q = lcg_fill(n_head * hd, 11);
        let k = lcg_fill(t_n * kv_dim, 12);
        let v = lcg_fill(t_n * kv_dim, 13);

        let mut o1 = vec![0.0; n_head * hd];
        let mut s1 = vec![0.0; n_head * block];
        unsafe {
            run_incumbent(
                &q, &k, &v, &mut o1, &mut s1, n_head, n_kv, kv_dim, hd, t_n, scale, block,
            );
        }

        let mut arm = HisaCensusArm::observe(2, vec![(0, 1), (256, 512)]);
        let mut o2 = vec![0.0; n_head * hd];
        let mut s2 = vec![0.0; n_head * block];
        unsafe {
            attention_heads_hisa(
                &q, &k, &v, &mut o2, &mut s2, n_head, n_kv, kv_dim, hd, t_n, scale, block, &mut arm,
            );
        }
        assert_bitwise(&o1, &o2, "attn_out");
        assert_bitwise(&s1, &s2, "head_scores");
    }

    /// The Remove perturbation, exactly: one in-band key loses its chunk
    /// partial dot; the test replicates the incumbent pass shape on the
    /// expected scores and compares bitwise (weights, output, energy).
    #[test]
    fn remove_ablation_is_the_exact_partial_subtraction() {
        let hd = 4;
        let n_head = 1;
        let n_kv = 1;
        let kv_dim = hd;
        let t_n = 3;
        let scale = 0.5; // 1/sqrt(4)
        let chunk = (2, 4);
        let band = (1, 2); // only key t=1 (distance 1)

        // q = [0,0,1,1]; the in-band key carries its whole dot in the chunk.
        let q = vec![0.0f32, 0.0, 1.0, 1.0];
        let k0 = lcg_fill(hd, 21);
        let k1 = vec![0.0f32, 0.0, 1.0, 1.0];
        let k2 = lcg_fill(hd, 22);
        let mut k = Vec::with_capacity(t_n * kv_dim);
        k.extend_from_slice(&k0);
        k.extend_from_slice(&k1);
        k.extend_from_slice(&k2);
        let v = lcg_fill(t_n * kv_dim, 23);

        let mut arm = HisaCensusArm::remove(0, chunk, band);
        let mut out = vec![0.0; n_head * hd];
        let mut scores = vec![0.0; n_head * 64];
        unsafe {
            attention_heads_hisa(
                &q,
                &k,
                &v,
                &mut out,
                &mut scores,
                n_head,
                n_kv,
                kv_dim,
                hd,
                t_n,
                scale,
                64,
                &mut arm,
            );
        }

        // Expected: replicate the incumbent 3-pass shape on the expected
        // scores (s1 = (full − chunk)·scale = 0). The full dots go through
        // simd_dot_f32 — the kernel's own arithmetic, same fold order, or
        // the bitwise compare would measure the test's sum, not the arm.
        let full0 = crate::simd::simd_dot_f32(&q, &k0, hd);
        let full2 = crate::simd::simd_dot_f32(&q, &k2, hd);
        let expected_scores = [full0 * scale, 0.0f32, full2 * scale];
        let max = expected_scores
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut exp_vals = [0.0f32; 3];
        let mut sum = 0.0f32;
        for i in 0..3 {
            exp_vals[i] = (expected_scores[i] - max).exp();
            sum += exp_vals[i];
        }
        let inv = 1.0 / sum;
        let mut expected_out = vec![0.0f32; hd];
        for t in 0..3 {
            let w = exp_vals[t] * inv;
            for d in 0..hd {
                expected_out[d] += w * v[t * kv_dim + d];
            }
        }
        assert_bitwise(&out, &expected_out, "attn_out");
        assert_bitwise(&scores[..3], &exp_vals, "head_scores(exp)");

        // The band mass = the in-band weight, in f64, count 1.
        let expected_w1 = (exp_vals[1] * inv) as f64;
        assert_eq!(arm.energy.to_bits(), expected_w1.to_bits());
        assert_eq!(arm.count, 1);
    }

    /// Replace with the null document's row IDENTICAL to the real row is
    /// the Observe posture bitwise — the swap is a no-op, not an
    /// approximation.
    #[test]
    fn replace_with_identical_null_row_is_observe_bitwise() {
        let hd = 4;
        let n_head = 1;
        let n_kv = 1;
        let kv_dim = hd;
        let t_n = 3;
        let scale = 0.5;
        let chunk = (2, 4);
        let band = (1, 2);
        let q = vec![0.0f32, 0.0, 1.0, 1.0];
        let k1 = vec![0.0f32, 0.0, 1.0, 1.0];
        let mut k = lcg_fill(t_n * kv_dim, 31);
        k[1 * kv_dim..2 * kv_dim].copy_from_slice(&k1);
        let v = lcg_fill(t_n * kv_dim, 32);

        let mut obs = HisaCensusArm::observe(0, vec![band]);
        let mut out_obs = vec![0.0; n_head * hd];
        let mut scores_obs = vec![0.0; n_head * 64];
        unsafe {
            attention_heads_hisa(
                &q,
                &k,
                &v,
                &mut out_obs,
                &mut scores_obs,
                n_head,
                n_kv,
                kv_dim,
                hd,
                t_n,
                scale,
                64,
                &mut obs,
            );
        }

        let mut rep =
            HisaCensusArm::replace(0, chunk, band, std::sync::Arc::new(k.clone()), t_n, kv_dim)
                .expect("valid replace arm");
        let mut out_rep = vec![0.0; n_head * hd];
        let mut scores_rep = vec![0.0; n_head * 64];
        unsafe {
            attention_heads_hisa(
                &q,
                &k,
                &v,
                &mut out_rep,
                &mut scores_rep,
                n_head,
                n_kv,
                kv_dim,
                hd,
                t_n,
                scale,
                64,
                &mut rep,
            );
        }
        assert_bitwise(&out_obs, &out_rep, "attn_out");
        assert_bitwise(&scores_obs, &scores_rep, "head_scores");
    }

    /// Replace with a DIFFERENT null row swaps exactly that chunk's
    /// partial dot into the score.
    #[test]
    fn replace_swaps_the_null_row_chunk_exactly() {
        let hd = 4;
        let n_head = 1;
        let n_kv = 1;
        let kv_dim = hd;
        let t_n = 3;
        let scale = 0.5;
        let chunk = (2, 4);
        let band = (1, 2);
        let q = vec![0.0f32, 0.0, 1.0, 1.0];
        let k1 = vec![0.0f32, 0.0, 1.0, 1.0];
        let mut k = lcg_fill(t_n * kv_dim, 41);
        k[1 * kv_dim..2 * kv_dim].copy_from_slice(&k1);
        let v = lcg_fill(t_n * kv_dim, 42);
        let mut null = k.clone();
        null[1 * kv_dim..2 * kv_dim].copy_from_slice(&[1.0f32, 0.0, 1.0, 0.5]);

        let mut rep =
            HisaCensusArm::replace(0, chunk, band, std::sync::Arc::new(null), t_n, kv_dim)
                .expect("valid replace arm");
        let mut out = vec![0.0; n_head * hd];
        let mut scores = vec![0.0; n_head * 64];
        unsafe {
            attention_heads_hisa(
                &q,
                &k,
                &v,
                &mut out,
                &mut scores,
                n_head,
                n_kv,
                kv_dim,
                hd,
                t_n,
                scale,
                64,
                &mut rep,
            );
        }

        // s1 = (full − chunk_real + chunk_null)·scale; chunk_real = 2,
        // chunk_null = 1·1 + 1·0.5 = 1.5 → s1 = 0.75. Full dots via
        // simd_dot_f32 — the kernel's own fold order.
        let full0 = crate::simd::simd_dot_f32(&q, &k[0..hd], hd);
        let full2 = crate::simd::simd_dot_f32(&q, &k[2 * hd..3 * hd], hd);
        let expected_scores = [full0 * scale, 0.75f32, full2 * scale];
        let max = expected_scores
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut exp_vals = [0.0f32; 3];
        let mut sum = 0.0f32;
        for i in 0..3 {
            exp_vals[i] = (expected_scores[i] - max).exp();
            sum += exp_vals[i];
        }
        let inv = 1.0 / sum;
        let mut expected_out = vec![0.0f32; hd];
        for t in 0..3 {
            let w = exp_vals[t] * inv;
            for d in 0..hd {
                expected_out[d] += w * v[t * kv_dim + d];
            }
        }
        assert_bitwise(&out, &expected_out, "attn_out");
        assert_bitwise(&scores[..3], &exp_vals, "head_scores(exp)");
    }

    /// A Remove band beyond the read is the incumbent bitwise — the gate
    /// for "the arm changed nothing it did not target".
    #[test]
    fn remove_with_empty_band_matches_incumbent_bitwise() {
        let n_head = 3;
        let n_kv = 1;
        let hd = 8;
        let kv_dim = n_kv * hd;
        let block = 1024;
        let t_n = 37;
        let scale = 1.0 / (hd as f32).sqrt();
        let q = lcg_fill(n_head * hd, 51);
        let k = lcg_fill(t_n * kv_dim, 52);
        let v = lcg_fill(t_n * kv_dim, 53);

        let mut o1 = vec![0.0; n_head * hd];
        let mut s1 = vec![0.0; n_head * block];
        unsafe {
            run_incumbent(
                &q, &k, &v, &mut o1, &mut s1, n_head, n_kv, kv_dim, hd, t_n, scale, block,
            );
        }

        let mut arm = HisaCensusArm::remove(7, (0, 16), (10_000, 10_001));
        let mut o2 = vec![0.0; n_head * hd];
        let mut s2 = vec![0.0; n_head * block];
        unsafe {
            attention_heads_hisa(
                &q, &k, &v, &mut o2, &mut s2, n_head, n_kv, kv_dim, hd, t_n, scale, block, &mut arm,
            );
        }
        assert_bitwise(&o1, &o2, "attn_out");
        assert_bitwise(&s1, &s2, "head_scores");
        assert_eq!(arm.count, 0, "an out-of-read band must record nothing");
    }

    /// The Replace tripwire: an in-band key with no null row keeps its
    /// incumbent score and counts in swapped_skipped — visible, never
    /// silent.
    #[test]
    fn replace_without_a_null_row_counts_the_skip() {
        let hd = 4;
        let n_head = 1;
        let n_kv = 1;
        let kv_dim = hd;
        let t_n = 3;
        let scale = 0.5;
        let chunk = (2, 4);
        let band = (1, 2);
        let q = vec![0.0f32, 0.0, 1.0, 1.0];
        let k1 = vec![0.0f32, 0.0, 1.0, 1.0];
        let mut k = lcg_fill(t_n * kv_dim, 61);
        k[1 * kv_dim..2 * kv_dim].copy_from_slice(&k1);
        let v = lcg_fill(t_n * kv_dim, 62);

        // Only ONE null row (position 0) — the in-band key at t=1 has none.
        let mut rep = HisaCensusArm::replace(
            0,
            chunk,
            band,
            std::sync::Arc::new(k[..kv_dim].to_vec()),
            1,
            kv_dim,
        )
        .expect("valid replace arm");
        let mut out = vec![0.0; n_head * hd];
        let mut scores = vec![0.0; n_head * 64];
        unsafe {
            attention_heads_hisa(
                &q,
                &k,
                &v,
                &mut out,
                &mut scores,
                n_head,
                n_kv,
                kv_dim,
                hd,
                t_n,
                scale,
                64,
                &mut rep,
            );
        }
        assert_eq!(rep.swapped_skipped, 1, "the skipped swap must be counted");

        // The kept score is the incumbent's: the UNMODIFIED full dot at
        // t=1, via simd_dot_f32 (the kernel's own arithmetic).
        let full0 = crate::simd::simd_dot_f32(&q, &k[0..hd], hd);
        let full1 = crate::simd::simd_dot_f32(&q, &k[1 * hd..2 * hd], hd);
        let full2 = crate::simd::simd_dot_f32(&q, &k[2 * hd..3 * hd], hd);
        let expected_scores = [full0 * scale, full1 * scale, full2 * scale];
        let max = expected_scores
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut exp_vals = [0.0f32; 3];
        for i in 0..3 {
            exp_vals[i] = (expected_scores[i] - max).exp();
        }
        assert_bitwise(&scores[..3], &exp_vals, "head_scores(exp)");
    }

    /// The constructor refuses a malformed null buffer loud.
    #[test]
    fn replace_constructor_refuses_a_malformed_buffer() {
        let err =
            HisaCensusArm::replace(0, (0, 4), (1, 2), std::sync::Arc::new(vec![0.0; 7]), 2, 4)
                .expect_err("7 floats is not 2 × 4");
        assert!(err.contains("expected t_null"));
        let err = HisaCensusArm::replace(0, (0, 4), (1, 2), std::sync::Arc::new(Vec::new()), 0, 4)
            .expect_err("t_null == 0 carries no null document");
        assert!(err.contains("t_null == 0"));
    }
}
