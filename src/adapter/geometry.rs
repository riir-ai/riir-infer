//! Mask geometry audit (Issue 1208 T6, `--mask-geometry` arm — the
//! paper's FFT-envelope reference axis: drift ≤1.4×, overlap 0.43–0.46
//! for trained masks; `identity -> overlap 1.0` is the golden law).
//!
//! Report-only, exit 0 — measurement, never a gate.
//!
//! Instrument: the repo's seeded Halko truncated SVD
//! (`quant::svd_lbit::rand_trunc_svd`, the svd_lbit lane's own
//! instrument) — deterministic per seed, exact when `k` covers the
//! matrix's full rank (every tiny golden), disclosed as truncated on
//! real matrices. Metrics per masked gate tensor vs its base:
//!
//! - **stable rank** `‖A‖_F² / σ₁²`
//! - **spectrum drift** `σ_i(masked) / σ_i(base)` over the computed
//!   spectrum — max and mean both disclosed (the envelope pins max)
//! - **principal overlap** `Σ_ij (u_a,i · u_b,j)² / k` over the top-k
//!   LEFT singular vectors — 1.0 = identical subspace (the identity
//!   mask must land there), 0.0 = orthogonal
//!
//! The mask operand is the codec's `DecodedMask`; only sections that
//! bind a gate tensor are audited (the bind-where-bound law).

use super::mask_codec::DecodedMask;
use crate::quant::svd_lbit::rand_trunc_svd;

/// Per-tensor audit row.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorGeometry {
    pub tensor_name: String,
    pub stable_rank_base: f32,
    pub stable_rank_masked: f32,
    /// `σ_i(masked)/σ_i(base)` over the computed spectrum, descending.
    pub spectrum_drift: Vec<f32>,
    /// Max of `spectrum_drift` (the envelope's ≤1.4× axis).
    pub drift_max: f32,
    /// Mean of `spectrum_drift`.
    pub drift_mean: f32,
    /// Top-k left-subspace overlap masked-vs-base (1.0 = identical).
    pub overlap: f32,
    /// The basis width used (≤ min(dout, din)).
    pub k: usize,
}

fn frobenius_sq(m: &[f32]) -> f64 {
    m.iter()
        .map(|v| {
            let v = f64::from(*v);
            v * v
        })
        .sum()
}

/// Geometry of one matrix vs its base, both row-major `[dout, din]`.
pub fn audit_matrix(
    tensor_name: &str,
    base: &[f32],
    masked: &[f32],
    dout: usize,
    din: usize,
    seed: u64,
) -> TensorGeometry {
    let full = dout.min(din);
    let k = full.min(64);
    let svd_base = rand_trunc_svd(base, dout, din, k, seed);
    let svd_masked = rand_trunc_svd(masked, dout, din, k, seed);

    let stable_rank_base =
        (frobenius_sq(base) / f64::from(svd_base.sigma[0] * svd_base.sigma[0])) as f32;
    let stable_rank_masked =
        (frobenius_sq(masked) / f64::from(svd_masked.sigma[0] * svd_masked.sigma[0])) as f32;

    let spectrum_drift: Vec<f32> = svd_masked
        .sigma
        .iter()
        .zip(svd_base.sigma.iter())
        .map(|(m, b)| {
            if *b == 0.0 {
                // a zero base σ means the direction is numerically absent;
                // drift is defined by the masked value alone
                *m
            } else {
                m / b
            }
        })
        .collect();
    let n = spectrum_drift.len() as f32;
    let drift_max = spectrum_drift.iter().cloned().fold(f32::MIN, f32::max);
    let drift_mean = spectrum_drift.iter().sum::<f32>() / n;

    // principal overlap: Σ_ij (u_a,i · u_b,j)² / k over the top-k LEFT
    // vectors (u is [dout, k], column-major)
    let kk = svd_base.k.min(svd_masked.k);
    let mut dot_sq_sum = 0.0f64;
    for i in 0..kk {
        for j in 0..kk {
            let mut dot = 0.0f64;
            for r in 0..dout {
                let a = f64::from(svd_masked.u[r * svd_masked.k + i]);
                let b = f64::from(svd_base.u[r * svd_base.k + j]);
                dot += a * b;
            }
            dot_sq_sum += dot * dot;
        }
    }
    let overlap = (dot_sq_sum / kk as f64) as f32;

    TensorGeometry {
        tensor_name: tensor_name.to_owned(),
        stable_rank_base,
        stable_rank_masked,
        spectrum_drift,
        drift_max,
        drift_mean,
        overlap,
        k: kk,
    }
}

/// Audit every gate section the mask binds, against the pristine
/// weights (looked up via `gate_lookup(layer) -> (data, rows, cols)`).
/// Returns rows in the mask's section order.
pub fn audit_mask(
    mask: &DecodedMask,
    seed: u64,
    mut gate_lookup: impl FnMut(usize) -> Option<(Vec<f32>, usize, usize)>,
) -> Vec<TensorGeometry> {
    let mut rows = Vec::new();
    for section in &mask.manifest.sections {
        let Some(layer_index) = section
            .tensor_name
            .strip_prefix("blk.")
            .and_then(|rest| rest.strip_suffix(".ffn_gate.weight"))
            .and_then(|mid| mid.parse::<usize>().ok())
        else {
            continue;
        };
        let Some((mut data, dout, din)) = gate_lookup(layer_index) else {
            continue;
        };
        if data.len() != section.bit_len as usize {
            continue;
        }
        // the masked copy: zero the deleted weights (the codec's bit
        // order — bit w covers weight w, row-major)
        for (w, slot) in data.iter_mut().enumerate() {
            let bit_index = section.bit_offset as usize + w;
            let keep = mask
                .bits
                .get(bit_index / 8)
                .is_some_and(|b| (b >> (bit_index % 8)) & 1 == 1);
            if !keep {
                *slot = 0.0;
            }
        }
        rows.push(audit_matrix(
            &section.tensor_name,
            &gate_lookup(layer_index)
                .map(|(d, _, _)| d)
                .unwrap_or_default(),
            &data,
            dout,
            din,
            seed,
        ));
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Halko SVD at full rank on a small matrix must reconstruct the
    /// matrix to tight tolerance (the goldens' exactness precondition).
    fn reconstruct_err(m: &[f32], dout: usize, din: usize) -> f32 {
        let svd = rand_trunc_svd(m, dout, din, dout.min(din), 0xC0FFEE);
        let mut err_sq = 0.0f64;
        for r in 0..dout {
            for c in 0..din {
                let mut approx = 0.0f64;
                for t in 0..svd.k {
                    approx += f64::from(svd.u[r * svd.k + t])
                        * f64::from(svd.sigma[t])
                        * f64::from(svd.v[c * svd.k + t]);
                }
                let d = f64::from(m[r * din + c]) - approx;
                err_sq += d * d;
            }
        }
        err_sq.sqrt() as f32
    }

    #[test]
    fn full_rank_halko_reconstructs_small_matrices() {
        // a small well-conditioned matrix, row-major 4x3
        let m = [
            3.0f32, 0.0, 0.0, //
            0.0, 2.0, 0.0, //
            0.0, 0.0, 1.0, //
            1.0, 1.0, 0.0,
        ];
        let err = reconstruct_err(&m, 4, 3);
        assert!(err < 1.0e-2, "reconstruction error {err}");
    }

    #[test]
    fn identity_mask_lands_on_overlap_1_drift_1() {
        let m: Vec<f32> = (0..12)
            .map(|i| ((i * 7) % 13) as f32 * 0.25 - 1.5)
            .collect();
        let row = audit_matrix("t", &m, &m, 4, 3, 42);
        assert!(
            (row.overlap - 1.0).abs() < 1.0e-2,
            "identity overlap {}",
            row.overlap
        );
        for d in &row.spectrum_drift {
            assert!((d - 1.0).abs() < 1.0e-2, "identity drift {d}");
        }
        assert!((row.stable_rank_base - row.stable_rank_masked).abs() < 1.0e-3);
    }

    #[test]
    fn deletion_moves_the_spectrum_and_shrinks_overlap() {
        // base: structured low-rank-ish matrix; masked: one row fully
        // deleted — overlap must drop below 1 but stay positive
        let mut base = vec![0.0f32; 16]; // 4x4
        for (i, v) in base.iter_mut().enumerate() {
            *v = ((i * 5) % 11) as f32 * 0.2 - 1.0;
        }
        let mut masked = base.clone();
        for slot in &mut masked[..4] {
            *slot = 0.0; // delete row 0 entirely
        }
        let row = audit_matrix("t", &base, &masked, 4, 4, 7);
        assert!(
            row.overlap < 0.999,
            "deletion must move the subspace: {}",
            row.overlap
        );
        assert!(row.overlap > 0.0, "still correlated: {}", row.overlap);
        assert!(
            row.drift_mean < 1.0,
            "deletion shrinks the spectrum: {}",
            row.drift_mean
        );
        assert!(row.stable_rank_masked < row.stable_rank_base * 1.5);
    }

    #[test]
    fn orthogonal_subspaces_overlap_near_zero() {
        // two matrices whose row spaces are orthogonal: A on rows, B the
        // complement rows zeroed — the top left-vectors live on disjoint
        // coordinates
        let mut a = vec![0.0f32; 8]; // 4x2
        let mut b = vec![0.0f32; 8];
        for r in 0..2 {
            for c in 0..2 {
                a[r * 2 + c] = ((r + c) % 2) as f32 + 0.5;
            }
        }
        for r in 2..4 {
            for c in 0..2 {
                b[r * 2 + c] = ((r + c) % 2) as f32 + 0.5;
            }
        }
        let row = audit_matrix("t", &a, &b, 4, 2, 9);
        assert!(
            row.overlap < 0.35,
            "disjoint-row matrices overlap low: {}",
            row.overlap
        );
    }

    #[test]
    fn audit_mask_routes_sections_and_skips_unbound() {
        use crate::adapter::mask_codec::{MaskManifest, MaskSection, decode};
        let base_digest = [9u8; 32];
        // one gate section (4x3 = 12 weights, delete 2) + one non-gate
        let manifest = MaskManifest::new(
            base_digest,
            vec![
                MaskSection::new("blk.0.ffn_gate.weight", 0, 12).unwrap(),
                MaskSection::new("output_norm.weight", 12, 12).unwrap(),
            ],
        )
        .unwrap();
        let mut bits = vec![0xFFu8; 24];
        bits[0] &= !(1 | 2); // delete weights 0,1 of layer 0
        let decoded = decode(&manifest.pack(&bits, None).unwrap(), None, None).unwrap();

        let base = (0..12)
            .map(|i| ((i * 7) % 13) as f32 * 0.25 - 1.5)
            .collect::<Vec<_>>();
        let rows = audit_mask(&decoded, 42, |layer| {
            if layer == 0 {
                Some((base.clone(), 4, 3))
            } else {
                None
            }
        });
        assert_eq!(rows.len(), 1, "non-gate sections skipped");
        assert_eq!(rows[0].tensor_name, "blk.0.ffn_gate.weight");
    }
}
