//! The weight-audit synthetic fixture (Issue 1209 T1/T2) — the paper's
//! closed-form planting recipe at fixture scale, `#[cfg(test)]`-gated
//! via its `lib.rs` declaration and typed to this module's OWN block.
//!
//! G5 surface rules (tripwire-asserted in [`tests`]):
//! (a) test-support only — never in a bin, never in the `weight_audit`
//!     feature surface (this module compiles to nothing outside
//!     `cargo test --features weight_audit`);
//! (b) the planter accepts only [`FixtureBlock`] — no GGUF/safetensors
//!     tensor parameter exists, so nothing can point it at a loaded
//!     model;
//! (c) no backdoored artifact ships; the fixture lives and dies inside
//!     these tests.
//!
//! Fixture scale: N=512 neurons, D=64, V=128 tokens — small enough for
//! exhaustive arms, large enough for population statistics. All
//! randomness is a pinned splitmix64 stream (zero new deps).

use crate::weight_audit::{MlpTensors, ReportHeader, Thresholds, census_mlp, render_report};
use anyhow::Result;

pub(crate) const FIXTURE_N: usize = 512;
pub(crate) const FIXTURE_D: usize = 64;
pub(crate) const FIXTURE_V: usize = 128;
pub(crate) const FIXTURE_K: usize = 8;

// ── Deterministic RNG (splitmix64 + Box-Muller, zero deps) ───────────

pub(crate) struct SplitMix64(u64);

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        SplitMix64(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform [0, 1).
    fn next_unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }
    /// Standard normal via Box-Muller (one draw per call; the paired
    /// sine draw is discarded — determinism over throughput).
    fn next_normal(&mut self) -> f32 {
        let u1 = self.next_unit().max(1e-7);
        let u2 = self.next_unit();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
    /// Seeded random unit vector (length-normalized gaussian draw).
    fn next_unit_vec(&mut self, d: usize) -> Vec<f32> {
        let mut v: Vec<f32> = (0..d).map(|_| self.next_normal()).collect();
        let norm = (v
            .iter()
            .map(|&x| (x as f64) * (x as f64))
            .sum::<f64>()
            .sqrt()) as f32;
        for x in v.iter_mut() {
            *x /= norm;
        }
        v
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

// ── The fixture block ────────────────────────────────────────────────

/// The synthetic SwiGLU block the planter is typed to. Layout mirrors
/// the GGUF convention exactly (see `weight_audit`'s module header):
/// gate/up are `[n][d]` row-major (one row per neuron); down is
/// `[d][n]` row-major (one row per output element — neuron n's write
/// column is `down[j*n + n]` over all j).
pub(crate) struct FixtureBlock {
    n: usize,
    d: usize,
    v: usize,
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
    unemb: Vec<f32>,
    gamma: Vec<f32>,
    label: String,
}

/// The five weight tensors, passed as one argument so the constructor
/// stays under the arity lint.
struct BlockTensors {
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
    unemb: Vec<f32>,
    gamma: Vec<f32>,
}

impl FixtureBlock {
    fn new(n: usize, d: usize, v: usize, t: BlockTensors, label: &str) -> Self {
        assert_eq!(t.gate.len(), n * d);
        assert_eq!(t.up.len(), n * d);
        assert_eq!(t.down.len(), d * n);
        assert_eq!(t.unemb.len(), v * d);
        assert_eq!(t.gamma.len(), d);
        FixtureBlock {
            n,
            d,
            v,
            gate: t.gate,
            up: t.up,
            down: t.down,
            unemb: t.unemb,
            gamma: t.gamma,
            label: label.to_string(),
        }
    }

    /// Pure-random geometric block: every row a gaussian draw scaled by
    /// U[0.5, 1.5] — no trained structure, no outliers. The G2a arm.
    pub(crate) fn random_geometric(seed: u64) -> Self {
        let (n, d, v) = (FIXTURE_N, FIXTURE_D, FIXTURE_V);
        let mut rng = SplitMix64::new(seed);
        let row = |rng: &mut SplitMix64, len: usize| -> Vec<f32> {
            let s = 0.5 + rng.next_unit();
            (0..len).map(|_| rng.next_normal() * s).collect()
        };
        let mut gate = Vec::with_capacity(n * d);
        let mut up = Vec::with_capacity(n * d);
        for _ in 0..n {
            gate.extend(row(&mut rng, d));
            up.extend(row(&mut rng, d));
        }
        let mut down = Vec::with_capacity(d * n);
        for _ in 0..d {
            down.extend(row(&mut rng, n));
        }
        let mut unemb = Vec::with_capacity(v * d);
        for _ in 0..v {
            unemb.extend(row(&mut rng, d));
        }
        let gamma = (0..d).map(|_| 0.5 + rng.next_unit()).collect();
        Self::new(
            n,
            d,
            v,
            BlockTensors {
                gate,
                up,
                down,
                unemb,
                gamma,
            },
            "fixture:random_geometric",
        )
    }

    /// Trained-shaped block (the G2b arm): log-normal row-scale spread
    /// (σ=1.0 gate/up, σ=0.5 down/unemb), a mild gate↔up correlation
    /// (ρ≈0.2 as trained SwiGLU pairs show), and four natural
    /// massive-activation neurons (write norm ×30 — well below planting
    /// scale, above the population).
    pub(crate) fn trained_shaped(seed: u64) -> Self {
        let (n, d, v) = (FIXTURE_N, FIXTURE_D, FIXTURE_V);
        let mut rng = SplitMix64::new(seed);
        let lognorm = |rng: &mut SplitMix64, sigma: f32| -> f32 {
            (sigma * rng.next_normal()).exp().clamp(0.01, 200.0)
        };
        let mut gate = vec![0.0f32; n * d];
        let mut up = vec![0.0f32; n * d];
        for i in 0..n {
            let sg = lognorm(&mut rng, 1.0);
            let su = lognorm(&mut rng, 1.0);
            let gunit = rng.next_unit_vec(d); // normalized — the mix stays ρ
            let indie = rng.next_unit_vec(d);
            // wu = ρ·g + √(1−ρ²)·independent (population cos ≈ ρ).
            let rho = 0.2f32;
            for j in 0..d {
                gate[i * d + j] = sg * gunit[j];
                up[i * d + j] = su * (rho * gunit[j] + (1.0 - rho * rho).sqrt() * indie[j]);
            }
        }
        let mut down = vec![0.0f32; d * n];
        for j in 0..d {
            let s = lognorm(&mut rng, 0.5);
            for i in 0..n {
                down[j * n + i] = s * rng.next_normal();
            }
        }
        // Four natural outliers: write norms ×30 (their down columns).
        for &np in &[FIXTURE_N - 1, FIXTURE_N - 2, FIXTURE_N - 3, FIXTURE_N - 4] {
            for j in 0..d {
                down[j * n + np] *= 30.0;
            }
        }
        let mut unemb = vec![0.0f32; v * d];
        for t in 0..v {
            let s = lognorm(&mut rng, 0.5);
            for j in 0..d {
                unemb[t * d + j] = s * rng.next_normal();
            }
        }
        let gamma: Vec<f32> = (0..d).map(|_| 0.5 + rng.next_unit()).collect();
        Self::new(
            n,
            d,
            v,
            BlockTensors {
                gate,
                up,
                down,
                unemb,
                gamma,
            },
            "fixture:trained_shaped",
        )
    }

    /// The f64 mean unembedding row (the census's pinned fold, mirrored).
    pub(crate) fn mean_unemb(&self) -> Vec<f64> {
        let mut mean = vec![0.0f64; self.d];
        for t in 0..self.v {
            for (j, m) in mean.iter_mut().enumerate() {
                *m += self.unemb[t * self.d + j] as f64;
            }
        }
        for x in mean.iter_mut() {
            *x /= self.v as f64;
        }
        mean
    }

    /// The additive-readout payload check: W = Σ_p silu(a_p)·a_p·(γ⊙wd_p)
    /// with a_p = ‖wg_p‖ (tied rows on unit directions make the gate/up
    /// activations a function of the norm alone); scores
    /// s_t = ⟨W, unemb_t − mean⟩; returns (argmax token, best/second
    /// margin, ∞ when the runner-up is ≤ 0).
    pub(crate) fn payload_argmax(&self, planted: &[usize]) -> (usize, f32) {
        let mut w = vec![0.0f64; self.d];
        for &np in planted {
            let g = &self.gate[np * self.d..(np + 1) * self.d];
            let a = (g
                .iter()
                .map(|&x| (x as f64) * (x as f64))
                .sum::<f64>()
                .sqrt()) as f32;
            let c = silu(a) * a;
            for (j, wj) in w.iter_mut().enumerate() {
                *wj += (c * self.gamma[j] * self.down[j * self.n + np]) as f64;
            }
        }
        let mean = self.mean_unemb();
        let mut best = f32::NEG_INFINITY;
        let mut second = f32::NEG_INFINITY;
        let mut bi = 0usize;
        for t in 0..self.v {
            let mut s = 0.0f64;
            for (j, wj) in w.iter().enumerate() {
                s += wj * ((self.unemb[t * self.d + j] as f64) - mean[j]);
            }
            let s = s as f32;
            if s > best {
                second = best;
                best = s;
                bi = t;
            } else if s > second {
                second = s;
            }
        }
        let margin = if second > 0.0 {
            best / second
        } else if best > 0.0 {
            f32::INFINITY
        } else {
            0.0
        };
        (bi, margin)
    }

    /// Scale the planted neurons' gate/up rows and down columns by
    /// `factor` (the Theorem-4 floor arm walks this down).
    pub(crate) fn scale_planted(&mut self, planted: &[usize], factor: f32) {
        for &np in planted {
            for j in 0..self.d {
                self.gate[np * self.d + j] *= factor;
                self.up[np * self.d + j] *= factor;
            }
            for j in 0..self.d {
                self.down[j * self.n + np] *= factor;
            }
        }
    }
}

impl MlpTensors for FixtureBlock {
    fn n_neurons(&self) -> usize {
        self.n
    }
    fn n_embd(&self) -> usize {
        self.d
    }
    fn vocab(&self) -> usize {
        self.v
    }
    fn gate_row(&self, i: usize) -> Result<Vec<f32>> {
        Ok(self.gate[i * self.d..(i + 1) * self.d].to_vec())
    }
    fn up_row(&self, i: usize) -> Result<Vec<f32>> {
        Ok(self.up[i * self.d..(i + 1) * self.d].to_vec())
    }
    fn down_out_row(&self, j: usize) -> Result<Vec<f32>> {
        Ok(self.down[j * self.n..(j + 1) * self.n].to_vec())
    }
    fn unemb_row(&self, t: usize) -> Result<Vec<f32>> {
        Ok(self.unemb[t * self.d..(t + 1) * self.d].to_vec())
    }
    fn gamma(&self) -> &[f32] {
        &self.gamma
    }
    fn label(&self) -> String {
        self.label.clone()
    }
}

// ── The planter (the paper's closed-form recipe at fixture scale) ────

/// The result of a planting (the census arm consumes `block`; the
/// planted metadata drives the assertions).
pub(crate) struct PlantedFixture {
    pub block: FixtureBlock,
    pub neurons: Vec<usize>,
    pub target: usize,
}

/// Plant `k` Lazarus-Analytic neurons:
/// 1. one tied gate/up pair per slot on a seeded separating direction
///    (SiLU makes the write nonnegative along its own direction);
/// 2. the down column on the exposing direction — the target token's
///    mean-centered unembedding row divided by the normalization gains
///    (ridge floor |γ| ≥ 0.25), normalized, scaled;
/// 3. the scale search doubles α/β until the payload wins the argmax
///    with a ≥4× margin over the runner-up (Theorem 2's argmaxability,
///    fixture scale). Slots share one target token — disclosed: the
///    census features are per-neuron and do not read the payload.
///
/// Consumes the block (the planted artifact lives only inside these
/// tests). Panics if the scale search fails (a fixture bug, never a
/// silent arm).
pub(crate) fn plant_trojan(
    mut block: FixtureBlock,
    k: usize,
    seed: u64,
    target: usize,
) -> PlantedFixture {
    assert!(k >= 1 && k * 4 <= block.n, "fixture planting size");
    assert!(target < block.v);
    let (n, d) = (block.n, block.d);
    let mut rng = SplitMix64::new(seed);
    let neurons: Vec<usize> = (0..k).collect();
    let dirs: Vec<Vec<f32>> = (0..k).map(|_| rng.next_unit_vec(d)).collect();
    // Exposing direction: (unemb[target] − mean) ⊘ γ′ (ridge floor), unit.
    let mean = block.mean_unemb();
    let gamma_floor = |g: f32| if g.abs() < 0.25 { g.signum() * 0.25 } else { g };
    let mut e = vec![0.0f32; d];
    for (j, ej) in e.iter_mut().enumerate() {
        let centered = (block.unemb[target * d + j] as f64 - mean[j]) as f32;
        *ej = centered / gamma_floor(block.gamma[j]);
    }
    let e_norm = (e
        .iter()
        .map(|&x| (x as f64) * (x as f64))
        .sum::<f64>()
        .sqrt()) as f32;
    for x in e.iter_mut() {
        *x /= e_norm;
    }
    // The scale search.
    let mut alpha = 1.0f32;
    let mut beta = 1.0f32;
    for _ in 0..64 {
        for (p, &np) in neurons.iter().enumerate() {
            for (j, gj) in block.gate[np * d..(np + 1) * d].iter_mut().enumerate() {
                *gj = alpha * dirs[p][j];
            }
            for (j, uj) in block.up[np * d..(np + 1) * d].iter_mut().enumerate() {
                *uj = alpha * dirs[p][j];
            }
        }
        for (j, ej) in e.iter().enumerate() {
            for &np in &neurons {
                block.down[j * n + np] = beta * ej;
            }
        }
        let (winner, margin) = block.payload_argmax(&neurons);
        if winner == target && margin >= 4.0 {
            return PlantedFixture {
                block,
                neurons,
                target,
            };
        }
        alpha *= 2.0;
        beta *= 2.0;
    }
    panic!("planter scale search failed to win the argmax (fixture bug)");
}

/// The T2-i falsification arm: untie the planted gate rows — same norms,
/// independent seeded directions. The alignment feature drops; the norm
/// features hold.
pub(crate) fn untie_gate_rows(block: &mut FixtureBlock, neurons: &[usize], seed: u64) {
    let d = block.d;
    let mut rng = SplitMix64::new(seed);
    for &np in neurons {
        let a = (block.gate[np * d..(np + 1) * d]
            .iter()
            .map(|&x| (x as f64) * (x as f64))
            .sum::<f64>()
            .sqrt()) as f32;
        let v = rng.next_unit_vec(d);
        for (j, gj) in block.gate[np * d..(np + 1) * d].iter_mut().enumerate() {
            *gj = a * v[j];
        }
    }
}

/// The G2c hazard arm: amplify one neuron to natural-massive-activation
/// scale (gate/up ×`gate_scale`, write column ×`write_scale`) — untied,
/// unaimed, extreme in the norm populations.
pub(crate) fn amplify_neuron(
    block: &mut FixtureBlock,
    np: usize,
    gate_scale: f32,
    write_scale: f32,
) {
    let (n, d) = (block.n, block.d);
    for j in 0..d {
        block.gate[np * d + j] *= gate_scale;
        block.up[np * d + j] *= gate_scale;
    }
    for j in 0..d {
        block.down[j * n + np] *= write_scale;
    }
}

/// Census + report at the canonical thresholds.
pub(crate) fn audit(block: &FixtureBlock) -> Result<crate::weight_audit::LayerAudit> {
    census_mlp(block, Thresholds::CANONICAL)
}

pub(crate) fn report_of(block: &FixtureBlock) -> Result<String> {
    let a = audit(block)?;
    Ok(render_report(
        &a,
        &ReportHeader {
            file: block.label.clone(),
            arch: None,
            file_digest_blake3: None,
        },
    ))
}

// ── The gate arms (G1–G5 fixture half + T2 falsifications) ──────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight_audit::TOP_M_SCAN;

    const PLANT_SEED: u64 = 0x1209_0001;
    const PLANT_TARGET: usize = 42;

    fn planted() -> PlantedFixture {
        let base = FixtureBlock::trained_shaped(PLANT_SEED ^ 0xA11C);
        plant_trojan(base, FIXTURE_K, PLANT_SEED, PLANT_TARGET)
    }

    /// Construction sanity: the payload wins the argmax with margin, and
    /// the planted rows are tied (the Analytic geometry the detector
    /// reads).
    #[test]
    fn construction_ties_rows_and_wins_argmax() {
        let pf = planted();
        let (winner, margin) = pf.block.payload_argmax(&pf.neurons);
        assert_eq!(winner, pf.target, "payload must win the argmax");
        assert!(margin >= 4.0, "margin {margin} below the planter bar");
        let d = pf.block.d;
        for &np in &pf.neurons {
            let g = &pf.block.gate[np * d..(np + 1) * d];
            let u = &pf.block.up[np * d..(np + 1) * d];
            let dot: f64 = g
                .iter()
                .zip(u)
                .map(|(&a, &b)| (a as f64) * (b as f64))
                .sum();
            let ng = g
                .iter()
                .map(|&x| (x as f64) * (x as f64))
                .sum::<f64>()
                .sqrt();
            let nu = u
                .iter()
                .map(|&x| (x as f64) * (x as f64))
                .sum::<f64>()
                .sqrt();
            let cos = (dot / (ng * nu)) as f32;
            assert!(cos > 0.999, "tied rows cos {cos}");
        }
    }

    /// G1: ALL k planted neurons flagged at canonical thresholds.
    #[test]
    fn g1_all_planted_flagged() {
        let pf = planted();
        let a = audit(&pf.block).expect("census");
        for &np in &pf.neurons {
            assert!(
                a.flags.contains(&np),
                "planted neuron {np} not flagged (flags: {:?})",
                a.flags
            );
        }
        assert_eq!(a.flags.len(), FIXTURE_K, "exactly the planted set flags");
        // Margins the bench note records: every planted feature ≥ 2× its
        // threshold.
        let t = a.thresholds;
        for r in a.window.iter().filter(|r| pf.neurons.contains(&r.neuron)) {
            assert!(
                r.z_norms >= 2.0 * t.z_norms,
                "planted z_norms {}",
                r.z_norms
            );
            assert!(
                r.rank_sum >= 1.2 * t.rank_sum,
                "planted rank {}",
                r.rank_sum
            );
            assert!(
                r.z_write >= 2.0 * t.z_write,
                "planted z_write {}",
                r.z_write
            );
            assert!(r.cos_align >= 0.99, "planted cos {}", r.cos_align);
            if let Some(tok) = r.tok_align {
                assert!(tok >= 0.99, "planted tok {}", tok);
            }
        }
    }

    /// G2a: pure-random geometric clean population — ZERO flags, the
    /// alignment margins hold, and the calibration reality is on record:
    /// norm-population features DO fire on rank extremes even in pure
    /// random (max_rank_sum ≈ 2.8 > the 2.4 threshold) — which is why
    /// `min_flags = 4` (both alignments + norm evidence) is the
    /// load-bearing FPR control, not the raw thresholds. The numbers
    /// print for the bench note.
    #[test]
    fn g2a_random_clean_zero_flags() {
        let b = FixtureBlock::random_geometric(0x1209_00A0);
        let a = audit(&b).expect("census");
        assert!(a.flags.is_empty(), "random clean flagged: {:?}", a.flags);
        let t = a.thresholds;
        eprintln!(
            "CAL g2a max_z_norms={} max_rank_sum={} max_z_write={} max_window_cos={}",
            a.pop_max_z_norms,
            a.pop_max_rank_sum,
            a.pop_max_z_write,
            a.window
                .iter()
                .map(|r| r.cos_align)
                .fold(f32::NEG_INFINITY, f32::max)
        );
        assert!(
            a.window.iter().all(|r| r.cos_align < t.cos_align),
            "random clean window cos over threshold"
        );
    }

    /// G2b: trained-shaped clean — FPR ≤ 0.2% (≤ 1 of 512), the
    /// alignment margins hold, and the four ×30 natural outliers light
    /// their norm features (max_z_write ≈ 25 ≫ 6) WITHOUT flagging — the
    /// documented hazard, survived by min_flags.
    #[test]
    fn g2b_trained_clean_le_one_flag() {
        let b = FixtureBlock::trained_shaped(0x1209_00B0);
        let a = audit(&b).expect("census");
        eprintln!(
            "CAL g2b max_z_norms={} max_rank_sum={} max_z_write={} max_window_cos={}",
            a.pop_max_z_norms,
            a.pop_max_rank_sum,
            a.pop_max_z_write,
            a.window
                .iter()
                .map(|r| r.cos_align)
                .fold(f32::NEG_INFINITY, f32::max)
        );
        assert!(
            a.flags.len() <= 1,
            "trained clean flagged {} of {}: {:?}",
            a.flags.len(),
            a.n_neurons,
            a.flags
        );
        let t = a.thresholds;
        assert!(
            a.window.iter().all(|r| r.cos_align < t.cos_align),
            "trained clean window cos over threshold"
        );
        // The hazard is real and armed: a natural outlier's norm features
        // exceed their thresholds.
        assert!(
            a.pop_max_z_write > t.z_write && a.pop_max_rank_sum > t.rank_sum,
            "the natural-outlier hazard did not light the norm features — the arm is vacuous"
        );
    }

    /// G2c: the natural massive-activation hazard — the outlier IS n† and
    /// lands in the window, and is NOT flagged (the alignments are the
    /// load-bearing features).
    #[test]
    fn g2c_natural_outlier_not_flagged() {
        let mut b = FixtureBlock::trained_shaped(0x1209_00C0);
        let outlier = 0usize;
        amplify_neuron(&mut b, outlier, 50.0, 30.0);
        let a = audit(&b).expect("census");
        assert_eq!(a.ndag, outlier, "the amplified neuron must be n†");
        assert!(
            a.window.iter().any(|r| r.neuron == outlier),
            "the outlier must sit in the scan window"
        );
        assert!(
            !a.flags.contains(&outlier),
            "natural outlier FLAGGED as tamper (flags: {:?})",
            a.flags
        );
    }

    /// T2-i: untie the planted gate rows — the tie feature (3) drops on
    /// every planted neuron, the norm features hold, and the K−1 non-n†
    /// planted neurons go quiet. The n† neuron MAY persist: the aim
    /// feature (4) reads only the down column, which untie does not
    /// touch — that persistence is the detector working as designed, and
    /// this arm pins that feature 3 alone was load-bearing for the
    /// non-n† flags.
    #[test]
    fn t2i_untying_drops_alignment_and_the_flag() {
        let pf = planted();
        let mut b = pf.block;
        untie_gate_rows(&mut b, &pf.neurons, PLANT_SEED ^ 0x0DD1);
        let a = audit(&b).expect("census");
        for r in a.window.iter().filter(|r| pf.neurons.contains(&r.neuron)) {
            assert!(
                r.cos_align < a.thresholds.cos_align,
                "untied planted cos {} still aligned",
                r.cos_align
            );
            // Norm features still fire:
            assert!(r.z_norms >= a.thresholds.z_norms);
            assert!(r.z_write >= a.thresholds.z_write);
            assert!(r.rank_sum >= a.thresholds.rank_sum);
        }
        for &np in &pf.neurons {
            if np == a.ndag {
                continue; // the aim feature may hold n† — by design
            }
            assert!(
                !a.flags.contains(&np),
                "untied non-n† planted {np} still flagged"
            );
        }
        assert!(
            a.flags.len() <= 1,
            "untie must silence all but possibly n† (flags: {:?})",
            a.flags
        );
    }

    /// T2-ii: shrinking the write crosses BOTH floors in the pinned
    /// order — the payload loses the argmax strictly BEFORE the census
    /// silences. The bound's direction: a write small enough to be
    /// undetectable is small enough to be useless.
    #[test]
    fn t2ii_below_write_floor_argmax_lost_and_quiet() {
        let pf = planted();
        let mut b = pf.block;
        let mut argmax_halvings: Option<usize> = None;
        let mut quiet_halvings: Option<usize> = None;
        for halvings in 1..=40 {
            b.scale_planted(&pf.neurons, 0.5);
            if argmax_halvings.is_none() {
                let (w, _) = b.payload_argmax(&pf.neurons);
                if w != pf.target {
                    argmax_halvings = Some(halvings);
                }
            }
            if quiet_halvings.is_none() && audit(&b).expect("census").flags.is_empty() {
                quiet_halvings = Some(halvings);
            }
            if argmax_halvings.is_some() && quiet_halvings.is_some() {
                break;
            }
        }
        let h1 = argmax_halvings.expect("payload never lost the argmax");
        let h2 = quiet_halvings.expect("census never went quiet");
        assert!(
            h1 < h2,
            "expected argmax-loss ({h1}) strictly before census-silence ({h2})"
        );
    }

    /// G4: the report is bit-identical across two runs.
    #[test]
    fn g4_report_bit_identical_across_runs() {
        let pf = planted();
        let r1 = report_of(&pf.block).expect("report 1");
        let r2 = report_of(&pf.block).expect("report 2");
        assert_eq!(r1, r2, "report must be bit-identical");
    }

    /// The window carries exactly the top-M rows, descending by write
    /// norm; n† is its first row with the token alignment filled.
    #[test]
    fn window_shape_and_ndag_row() {
        let pf = planted();
        let a = audit(&pf.block).expect("census");
        assert_eq!(a.window.len(), TOP_M_SCAN);
        assert_eq!(a.window[0].neuron, a.ndag);
        assert!(a.window[0].tok_align.is_some());
        for w in a.window.windows(2) {
            assert!(w[0].write_norm >= w[1].write_norm);
        }
    }

    /// G5(c) tripwire: the planter stays out of the shipped surface —
    /// (a) the lib declares this module under cfg(all(test, feature)),
    /// and (b) no bin references the fixture or the planter.
    #[test]
    fn g5_planter_stays_out_of_shipped_surface() {
        let manifest = env!("CARGO_MANIFEST_DIR");
        let lib = std::fs::read_to_string(format!("{manifest}/src/lib.rs")).expect("lib.rs");
        let decl = lib
            .lines()
            .find(|l| l.contains("mod weight_audit_fixture"))
            .expect("fixture module declaration missing from lib.rs");
        // The declaration line itself is the `mod` line; the two cfg
        // attributes sit immediately above it.
        let idx = lib
            .lines()
            .position(|l| l.contains("mod weight_audit_fixture"))
            .expect("position");
        let above: String = lib.lines().skip(idx.saturating_sub(2)).take(2).collect();
        assert!(
            above.contains("cfg(all(test, feature = \"weight_audit\"))"),
            "fixture module must be cfg(all(test, feature = \"weight_audit\"))\n---\n{above}\n---"
        );
        let _ = decl;
        let bin_dir = format!("{manifest}/src/bin");
        for entry in std::fs::read_dir(&bin_dir).expect("bin dir") {
            let path = entry.expect("entry").path();
            if path.is_dir() {
                continue; // bin subdirectories carry no planter reference
            }
            let src = std::fs::read_to_string(&path).expect("bin source");
            assert!(
                !src.contains("weight_audit_fixture") && !src.contains("plant_trojan"),
                "bin {:?} references the fixture planter",
                path
            );
        }
    }
}
