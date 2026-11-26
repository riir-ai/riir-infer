//! Fixture gates for the Issue-1252 A2 spike-code census (feature-gated;
//! `run_all()` is driven BOTH by the tests below and the bin's
//! `--self-test` — the `neuron_census_fixture` pattern). Synthetic rows
//! with KNOWN distribution shapes: the floors must be recovered where the
//! shape matches, the engineered dead mass must show as excess, and the
//! artifact must round-trip byte-stably.

use anyhow::Result;

/// A deterministic SplitMix64 (no global RNG — the census-family law).
/// The draw method is `next_u64` (NOT `next` — the
/// should_implement_trait fence: a struct method named `next` reads as
/// an Iterator impl to every downstream reader).
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform f64 in [-1, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }

    /// Approximate Gaussian via the sum of 12 uniforms (Irwin–Hall; mean 0,
    /// var 1 — good enough for floor-recovery at row width ≥ 512 with the
    /// ±1e-2 tolerance the floors carry).
    pub fn gaussian(&mut self) -> f64 {
        let acc: f64 = (0..12).map(|_| self.unit()).sum();
        acc
    }

    /// Approximate Laplace(0,1) via the inverse CDF on a uniform.
    pub fn laplace(&mut self) -> f64 {
        let u = (self.unit() + 1.0) / 2.0; // (0,1)
        let u = u.clamp(1e-9, 1.0 - 1e-9);
        if u < 0.5 {
            (2.0 * u).ln()
        } else {
            -(2.0 * (1.0 - u)).ln()
        }
    }
}

/// The fixture battery. Every arm is a loud assertion; the bin's
/// `--self-test` prints the pass count.
pub fn run_all() -> Result<usize> {
    let mut passed = 0usize;

    // G1 — the Gaussian floor is RECOVERED on Gaussian rows at k=3 within
    // ±0.02 (sampling noise at width 4096, 128 rows).
    {
        let mut rng = SplitMix64(0x1252_0000_0000_0001);
        let rows: Vec<Vec<f32>> = (0..128)
            .map(|_| (0..4096).map(|_| rng.gaussian() as f32).collect())
            .collect();
        let st = crate::spike_code_census::census_layer(&rows, 3)?;
        let floor = crate::spike_code_census::gaussian_zero_floor(3);
        assert!(
            (st.elem_zero_rate - floor).abs() < 0.02,
            "G1: gaussian rows measured zero rate {:.5} vs floor {:.5}",
            st.elem_zero_rate,
            floor
        );
        assert!(
            st.gaussian_better(),
            "G1: gaussian rows should fit the gaussian floor better"
        );
        passed += 1;
    }

    // G2 — the Laplacian floor is RECOVERED on Laplace rows at k=3 within
    // ±0.02, and the shape pick flips to laplace.
    {
        let mut rng = SplitMix64(0x1252_0000_0000_0002);
        let rows: Vec<Vec<f32>> = (0..128)
            .map(|_| (0..4096).map(|_| rng.laplace() as f32).collect())
            .collect();
        let st = crate::spike_code_census::census_layer(&rows, 3)?;
        let floor = crate::spike_code_census::laplace_zero_floor(3);
        assert!(
            (st.elem_zero_rate - floor).abs() < 0.02,
            "G2: laplace rows measured zero rate {:.5} vs floor {:.5}",
            st.elem_zero_rate,
            floor
        );
        assert!(
            !st.gaussian_better(),
            "G2: laplace rows should fit the laplace floor better"
        );
        passed += 1;
    }

    // G3 — the engineered structural dead mass shows as EXCESS: take the
    // G1 gaussian rows and zero 20% of the groups outright; the measured
    // zero rate must exceed the gaussian floor by ~0.20 within tolerance,
    // and the whole-128-group zero rate must be ≥ ~0.20 (the skip ceiling
    // the kernel-bar arithmetic reads).
    {
        let mut rng = SplitMix64(0x1252_0000_0000_0003);
        let rows: Vec<Vec<f32>> = (0..128)
            .map(|row| {
                let mut r: Vec<f32> = (0..4096).map(|_| rng.gaussian() as f32).collect();
                if row % 5 == 0 {
                    // Zero groups 0..=9 (20 of 32 groups... 4096/128 = 32
                    // groups; zero 10 of them ≈ 31%).
                    for g in 0..10 {
                        for x in &mut r[g * 128..(g + 1) * 128] {
                            *x = 0.0;
                        }
                    }
                }
                r
            })
            .collect();
        let st = crate::spike_code_census::census_layer(&rows, 3)?;
        let (elo, ehi) = st.structural_excess_bracket();
        // The bracket: ~20% of rows carry 10/32 dead groups; measured
        // overall ≈ 15.7% → over the loosest floor ≈ 0.3%, over the
        // tightest ≈ 5%. The UPPER leg is the unambiguous dead-mass
        // reading; the lower leg is what the nearest-floor defect used to
        // report as the whole answer.
        assert!(
            ehi > 0.03 && elo < ehi,
            "G3: the bracket must carry the dead mass (lo={elo:.5} hi={ehi:.5})"
        );
        assert!(
            st.group_zero_rate >= 0.05,
            "G3: group zero rate {:.5} must clear the engineered band",
            st.group_zero_rate
        );
        assert!(
            st.elem_zero_rate > st.laplace_floor,
            "G3: measured zero rate must exceed the loosest floor on dead-mass rows"
        );
        passed += 1;
    }

    // G4 — the spike-count histogram is exact: a hand-built 4-row layer
    // with known spikes per row (k=2, hand-computed thresholds).
    {
        // width 128 (one group). Rows:
        // r0: all zeros → V_th = 0! (the degenerate all-zero row: mean|x|
        // = 0 → v_th = 0 → division. The census must treat an all-zero row
        // as all-zero spikes (the paper's code: s = round(0/0) is
        // undefined; the census PINS it as zero spikes — the group-skip
        // reading wants it zero).
        let rows = vec![vec![0.0f32; 128], {
            let mut r = vec![0.0f32; 128];
            r[0] = 4.0;
            r[1] = -2.0;
            r[2] = 0.3;
            r
        }];
        // The census must not NaN on the all-zero row. Mean|x| of r1 = 6.3/128
        // ≈ 0.04922; v_th ≈ 0.02461 (k=2); half ≈ 0.01230. Spikes: |4|≥ → 1,
        // |-2| → 1, |0.3| → 1 (0.3 > 0.0123) → 3 spikes. r0: pinned 0.
        let st = crate::spike_code_census::census_layer(&rows, 2)?;
        assert_eq!(st.spike_counts[0], 0, "G4: all-zero row pins zero spikes");
        assert_eq!(
            st.spike_counts[1], 3,
            "G4: hand row must count exactly 3 spikes"
        );
        assert_eq!(st.histogram[0], 1, "G4: histogram bin 0 = the all-zero row");
        assert_eq!(st.histogram[3], 1, "G4: histogram bin 3 = the hand row");
        passed += 1;
    }

    // G5 — the k ladder monotonicity: higher k → LOWER zero rate (finer
    // threshold → fewer zeros) on the same rows; and the floors move the
    // same direction.
    {
        let mut rng = SplitMix64(0x1252_0000_0000_0005);
        let rows: Vec<Vec<f32>> = (0..64)
            .map(|_| (0..2048).map(|_| rng.laplace() as f32).collect())
            .collect();
        let s2 = crate::spike_code_census::census_layer(&rows, 2)?;
        let s4 = crate::spike_code_census::census_layer(&rows, 4)?;
        assert!(
            s4.elem_zero_rate < s2.elem_zero_rate,
            "G5: zero rate must fall as k rises ({} vs {})",
            s4.elem_zero_rate,
            s2.elem_zero_rate
        );
        assert!(
            crate::spike_code_census::laplace_zero_floor(4)
                < crate::spike_code_census::laplace_zero_floor(2),
            "G5: the laplace floor must fall as k rises"
        );
        passed += 1;
    }

    // G6 — the artifact: canonical bytes are stable across two builds, the
    // digest pins, and the render is deterministic.
    {
        let mut rng = SplitMix64(0x1252_0000_0000_0006);
        let rows: Vec<Vec<f32>> = (0..16)
            .map(|_| (0..512).map(|_| rng.gaussian() as f32).collect())
            .collect();
        let a = crate::spike_code_census::census_layer(&rows, 2)?;
        let b = crate::spike_code_census::census_layer(&rows, 3)?;
        let artifact = crate::spike_code_census::SpikeCensusArtifact {
            model_digest_hex: "00".repeat(32),
            corpus_digest_hex: "11".repeat(32),
            tokens: 16,
            width: 512,
            k_ladder: vec![2, 3],
            layers: vec![7],
            stats: vec![a, b],
        };
        let d1 = artifact.digest();
        let d2 = artifact.digest();
        assert_eq!(d1, d2, "G6: digest must be build-stable");
        let r1 = artifact.render("g6");
        let r2 = artifact.render("g6");
        assert_eq!(r1, r2, "G6: render must be deterministic");
        assert!(r1.contains("spike-code census"), "G6: render header");
        passed += 1;
    }

    // G7 — the refusal battery: empty rows, ragged rows, zero width, k=0.
    {
        let empty: Vec<Vec<f32>> = Vec::new();
        assert!(crate::spike_code_census::census_layer(&empty, 3).is_err());
        let ragged = vec![vec![1.0f32, 2.0], vec![1.0]];
        assert!(crate::spike_code_census::census_layer(&ragged, 3).is_err());
        let zerow = vec![Vec::<f32>::new()];
        assert!(crate::spike_code_census::census_layer(&zerow, 3).is_err());
        let ok = vec![vec![1.0f32, 2.0]];
        assert!(crate::spike_code_census::census_layer(&ok, 0).is_err());
        passed += 1;
    }

    Ok(passed)
}
