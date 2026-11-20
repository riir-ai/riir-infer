//! Issue 1237 — the decision-core census math (R1 readout: the scorer-input
//! marker rows).
//!
//! The ACE extraction itself is katgpt-core's `relevance_core` (consumed,
//! never forked). This module owns only what is laya-specific:
//!
//! - [`Scorer64`] — the head's scorer (`LN → Linear(d,d) → GELU(erf) →
//!   Linear(d,1)`) re-expressed in f64 with its ANALYTIC input gradient, so
//!   the relevance moment `S = Σ ∇logit ∇logitᵀ` needs no finite
//!   differences;
//! - [`CoreProjector`] — the mean-ablation keep / remove projections the
//!   causal reads apply to marker rows.
//!
//! Offline census only (`O(d²)` per row) — never a serving path.

use crate::laya::riir::head::ScorerTensors;

/// The scorer in f64 with its input gradient. Weights widened once.
pub struct Scorer64 {
    d: usize,
    eps: f64,
    s0w: Vec<f64>,
    s0b: Vec<f64>,
    /// `[d_out, d_in]` row-major (torch `Linear.weight`; `y = W x`).
    s1w: Vec<f64>,
    s1b: Vec<f64>,
    s3w: Vec<f64>,
    s3b: f64,
}

/// Per-call buffers for [`Scorer64`] (allocated once per `d`).
pub struct ScorerScratch {
    xhat: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    dz: Vec<f64>,
    u: Vec<f64>,
}

impl ScorerScratch {
    pub fn new(d: usize) -> Self {
        Self {
            xhat: vec![0.0; d],
            y: vec![0.0; d],
            z: vec![0.0; d],
            dz: vec![0.0; d],
            u: vec![0.0; d],
        }
    }
}

const INV_SQRT2: f64 = std::f64::consts::FRAC_1_SQRT_2;
/// `1/√(2π)`.
const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

fn widen(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| f64::from(x)).collect()
}

impl Scorer64 {
    pub fn new(t: &ScorerTensors) -> Self {
        let d = t.d;
        assert_eq!(t.s1w.len(), d * d, "scorer.1 is [d, d]");
        assert_eq!(t.s3w.len(), d, "scorer.3 is [1, d]");
        Self {
            d,
            eps: f64::from(t.eps),
            s0w: widen(&t.s0w),
            s0b: widen(&t.s0b),
            s1w: widen(&t.s1w),
            s1b: widen(&t.s1b),
            s3w: widen(&t.s3w),
            s3b: f64::from(t.s3b[0]),
        }
    }

    pub fn dim(&self) -> usize {
        self.d
    }

    /// The forward through `z` (shared by [`Self::logit`] and
    /// [`Self::logit_grad`]); returns `1/σ`.
    fn forward_z(&self, h: &[f64], sc: &mut ScorerScratch) -> f64 {
        let d = self.d;
        let mean = h.iter().sum::<f64>() / d as f64;
        let var = h.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / d as f64;
        let inv_sigma = 1.0 / (var + self.eps).sqrt();
        for (i, &hi) in h.iter().enumerate().take(d) {
            sc.xhat[i] = (hi - mean) * inv_sigma;
            sc.y[i] = sc.xhat[i] * self.s0w[i] + self.s0b[i];
        }
        for o in 0..d {
            let row = &self.s1w[o * d..(o + 1) * d];
            sc.z[o] = self.s1b[o] + row.iter().zip(&sc.y).map(|(w, y)| w * y).sum::<f64>();
        }
        inv_sigma
    }

    fn head_logit(&self, z: &[f64]) -> f64 {
        let mut acc = self.s3b;
        for (w, &zi) in self.s3w.iter().zip(z) {
            acc += w * 0.5 * zi * (1.0 + libm::erf(zi * INV_SQRT2));
        }
        acc
    }

    /// The scorer logit of one marker row.
    pub fn logit(&self, h: &[f64], sc: &mut ScorerScratch) -> f64 {
        self.forward_z(h, sc);
        self.head_logit(&sc.z)
    }

    /// The logit and `∂logit/∂h` (written into `grad`).
    pub fn logit_grad(&self, h: &[f64], grad: &mut [f64], sc: &mut ScorerScratch) -> f64 {
        let d = self.d;
        let inv_sigma = self.forward_z(h, sc);
        let logit = self.head_logit(&sc.z);
        // dz = w3 ⊙ gelu'(z), gelu'(z) = Φ(z) + z φ(z).
        for o in 0..d {
            let z = sc.z[o];
            let cdf = 0.5 * (1.0 + libm::erf(z * INV_SQRT2));
            let pdf = INV_SQRT_2PI * (-0.5 * z * z).exp();
            sc.dz[o] = self.s3w[o] * (cdf + z * pdf);
        }
        // dy = W1ᵀ dz; u = dy ⊙ w0.
        sc.u.fill(0.0);
        for o in 0..d {
            let g = sc.dz[o];
            if g == 0.0 {
                continue;
            }
            let row = &self.s1w[o * d..(o + 1) * d];
            for (u, w) in sc.u.iter_mut().zip(row) {
                *u += w * g;
            }
        }
        for i in 0..d {
            sc.u[i] *= self.s0w[i];
        }
        // LayerNorm backward: (u − mean(u) − x̂·mean(u⊙x̂)) / σ.
        let mu = sc.u.iter().sum::<f64>() / d as f64;
        let mux = sc.u.iter().zip(&sc.xhat).map(|(a, b)| a * b).sum::<f64>() / d as f64;
        for ((g, u), x) in grad.iter_mut().zip(&sc.u).zip(&sc.xhat) {
            *g = (u - mu - x * mux) * inv_sigma;
        }
        logit
    }
}

/// Mean-ablation projections onto / off an orthonormal core `Q` (`r × d`
/// row-major): keep `μ + QᵀQ(h−μ)`, remove `h − QᵀQ(h−μ)`.
pub struct CoreProjector<'a> {
    pub mean: &'a [f64],
    pub q: &'a [f64],
    pub r: usize,
}

impl CoreProjector<'_> {
    /// Writes the in-core component `QᵀQ(h−μ)` into `out`.
    fn in_core(&self, h: &[f64], out: &mut [f64]) {
        let d = self.mean.len();
        out.fill(0.0);
        for k in 0..self.r {
            let qk = &self.q[k * d..(k + 1) * d];
            let c: f64 = (0..d).map(|i| qk[i] * (h[i] - self.mean[i])).sum();
            for i in 0..d {
                out[i] += c * qk[i];
            }
        }
    }

    pub fn keep(&self, h: &[f64], out: &mut [f64]) {
        self.in_core(h, out);
        for (o, m) in out.iter_mut().zip(self.mean) {
            *o += m;
        }
    }

    pub fn remove(&self, h: &[f64], out: &mut [f64]) {
        self.in_core(h, out);
        for (o, x) in out.iter_mut().zip(h) {
            *o = x - *o;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5
    }

    fn toy(d: usize) -> Scorer64 {
        let mut s = 7u64;
        let mut v = |n: usize| (0..n).map(|_| lcg(&mut s)).collect::<Vec<f32>>();
        Scorer64::new(&ScorerTensors {
            s0w: v(d).iter().map(|x| 1.0 + x).collect(),
            s0b: v(d),
            s1w: v(d * d),
            s1b: v(d),
            s3w: v(d),
            s3b: vec![0.1],
            d,
            eps: 1e-5,
        })
    }

    /// The analytic gradient matches central finite differences.
    #[test]
    fn grad_matches_finite_difference() {
        let d = 24;
        let sc64 = toy(d);
        let mut sc = ScorerScratch::new(d);
        let mut seed = 99u64;
        let h: Vec<f64> = (0..d).map(|_| f64::from(lcg(&mut seed)) * 3.0).collect();
        let mut g = vec![0.0; d];
        sc64.logit_grad(&h, &mut g, &mut sc);
        let eps = 1e-6;
        for i in 0..d {
            let (mut hp, mut hm) = (h.clone(), h.clone());
            hp[i] += eps;
            hm[i] -= eps;
            let fd = (sc64.logit(&hp, &mut sc) - sc64.logit(&hm, &mut sc)) / (2.0 * eps);
            assert!(
                (fd - g[i]).abs() < 1e-6 * (1.0 + fd.abs()),
                "i {i}: fd {fd} vs {}",
                g[i]
            );
        }
    }

    /// keep + remove reconstruct `h + μ`; a full-rank core keeps `h`.
    #[test]
    fn projector_identities() {
        let d = 3;
        let mean = [1.0, -1.0, 0.5];
        let q = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let h = [2.0, 3.0, -4.0];
        let (mut k, mut rm) = ([0.0; 3], [0.0; 3]);
        let p = CoreProjector {
            mean: &mean,
            q: &q,
            r: 1,
        };
        p.keep(&h, &mut k);
        p.remove(&h, &mut rm);
        assert_eq!(k, [2.0, -1.0, 0.5]);
        for i in 0..d {
            assert!((k[i] + rm[i] - h[i] - mean[i]).abs() < 1e-12);
        }
        let full = CoreProjector {
            mean: &mean,
            q: &q,
            r: 3,
        };
        full.keep(&h, &mut k);
        assert_eq!(k, h);
    }
}
