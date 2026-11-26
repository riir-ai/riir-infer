//! Issue 1249 — the GDN erase-stability gate: the two-cell pin over the
//! shipped rank-1 erase operator.
//!
//! **Cell (a) — the classical DeltaNet bound (Yang et al. 2412.06464 lineage,
//! prior art, NOT MemT's).** The erase `A = I − β·k·kᵀ` (per head) has
//! eigenvalues `{1 (×d−1), 1 − β‖k‖²}` ⇒ non-expansive iff
//! **0 ≤ β‖k‖² ≤ 2**. The shipped lane satisfies this BY CONSTRUCTION
//! (`expand_l2_normalized_heads_into` L2-normalizes k, β is a sigmoid scalar
//! in (0,1), decay ∈ (0,1]) — this gate asserts it, and formalizes the
//! `riir-infer-gpu` cudarc fixture class (β·‖k‖² driven to 5..30 by
//! bypassing the production normalizer) as the REFUSED posture the same
//! predicate must catch.
//!
//! **Cell (b) — the MemT channel-wise law (arXiv:2610.26825 App. C).** A
//! channel-wise gate b on unit-norm k makes `A = I − kᵀ(b⊙k)` non-normal:
//! sup‖A‖ → 2/√3 ≈ 1.1547, peaking at ρ = Σ bᵢkᵢ² = 1/3; the symmetrized
//! form `u = √β⊙k` gives `I − uᵀu` non-expansive by construction
//! (sup ≤ 1 + ε). The deterministic grid covers the analytic peak — never
//! random draws alone (the falsification arm drops the law honestly if the
//! sweep cannot reach 1.14 in our parameterization).
//!
//! Both cells assert the PER-STEP state-norm behavior of the shipped
//! `gated_deltanet_step` (production posture) or the direct operator
//! algebra (cell b) — no GPU, no features; pure-CPU (`riir-infer-core`).

use riir_infer_core::deltanet::forward::gated_deltanet_step;

/// Deterministic SplitMix64 PRNG (Issue-809 law: a seeded draw carries a pin
/// row in the test itself — seed + consumption order are the pin).
struct SplitMix64(u64);
impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    /// Uniform in [-1, 1).
    fn next_signed(&mut self) -> f32 {
        self.next_f32() * 2.0 - 1.0
    }
}

/// Seed pin: fixed, named; the random-regime arms consume it in construction
/// order (one draw per element, k then v per token). Changing either reds.
const SEED_A: u64 = 0x1249_A001;

fn l2_normalize(x: &mut [f32]) {
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    if sum_sq > 0.0 {
        let inv = 1.0 / sum_sq.sqrt();
        for v in x.iter_mut() {
            *v *= inv;
        }
    }
}

fn state_norm(state: &[f32]) -> f32 {
    state.iter().map(|v| v * v).sum::<f32>().sqrt()
}

// ── Cell (a) — the production posture is non-expansive ────────────────────

/// The shipped scalar-β step on its production posture (L2-normalized k,
/// β ∈ (0,1) via sigmoid, decay ∈ (0,1]) keeps the per-step state norm
/// non-increasing: ratio ≤ 1 + 1e-6, across a sinusoid regime and a seeded
/// random-walk regime, for 512 steps.
#[test]
fn cell_a_production_posture_state_norm_non_increasing() {
    let (n_v_heads, key_dim, val_dim) = (4usize, 32usize, 32usize);
    let state_dim = n_v_heads * key_dim * val_dim;
    let beta: Vec<f32> = (0..n_v_heads)
        .map(|h| 1.0 / (1.0 + (-(0.25 + 0.5 * h as f32)).exp()))
        .collect();
    let decay: Vec<f32> = (0..n_v_heads)
        .map(|h| (-0.01 - 0.05 * h as f32).exp())
        .collect();

    // ── Arm 1 — the HOMOGENEOUS erase (v = 0): the rank-1 erase itself is
    // non-expansive, so the state norm falls MONOTONICALLY from any seeded
    // state, both regimes. This is the bound the issue names; the sinusoid
    // vs random-walk axes vary the k/q content the erase reads.
    for regime in ["sinusoid", "random_walk"] {
        let mut rng = SplitMix64(SEED_A);
        let mut state = vec![0.0f32; state_dim];
        let mut walk = vec![0.0f32; n_v_heads * key_dim];
        // Seed the state with content (the erase has something to contract).
        for v in state.iter_mut() {
            *v = rng.next_signed() * 0.1;
        }
        let zero_v = vec![0.0f32; n_v_heads * val_dim];
        let mut prev = state_norm(&state);
        for t in 0..200u64 {
            let mut k = vec![0.0f32; n_v_heads * key_dim];
            let mut q = vec![0.0f32; n_v_heads * key_dim];
            for h in 0..n_v_heads {
                for c in 0..key_dim {
                    let idx = h * key_dim + c;
                    k[idx] = match regime {
                        "sinusoid" => (t as f32 * 0.05 + c as f32).sin(),
                        _ => {
                            walk[idx] += rng.next_signed() * 0.1;
                            walk[idx]
                        }
                    };
                    q[idx] = (t as f32 * 0.03 + c as f32).cos();
                }
                l2_normalize(&mut k[h * key_dim..(h + 1) * key_dim]);
                l2_normalize(&mut q[h * key_dim..(h + 1) * key_dim]);
            }
            let out = gated_deltanet_step(
                &q, &k, &zero_v, &mut state, &beta, &decay, n_v_heads, key_dim, val_dim,
            );
            assert!(out.iter().all(|x| x.is_finite()));
            let cur = state_norm(&state);
            assert!(
                cur <= prev + 1e-6,
                "regime {regime} homogeneous step {t}: state norm GREW {prev} -> {cur} \
                 (the rank-1 erase is expansive on the production posture)"
            );
            prev = cur;
        }
        assert!(
            prev < state_norm(&vec![0.0; state_dim]) + 1.0 && prev <= prev,
            "unreachable guard"
        );
    }

    // ── Arm 2 — the DRIVEN bound: with input, the per-step norm obeys
    // ||S'|| <= g·||S|| + β·||v|| per head (the erase non-expansive + the
    // rank-1 injection's exact one-norm). Asserted exactly, per step.
    for regime in ["sinusoid", "random_walk"] {
        let mut rng = SplitMix64(SEED_A);
        let mut state = vec![0.0f32; state_dim];
        let mut walk = vec![0.0f32; n_v_heads * key_dim];
        let head_norm = |state: &[f32], h: usize, kd: usize, vd: usize| {
            state[h * kd * vd..(h + 1) * kd * vd]
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt()
        };
        for t in 0..512u64 {
            let mut k = vec![0.0f32; n_v_heads * key_dim];
            let mut q = vec![0.0f32; n_v_heads * key_dim];
            let mut v = vec![0.0f32; n_v_heads * val_dim];
            for h in 0..n_v_heads {
                for c in 0..key_dim {
                    let idx = h * key_dim + c;
                    k[idx] = match regime {
                        "sinusoid" => (t as f32 * 0.05 + c as f32).sin(),
                        _ => {
                            walk[idx] += rng.next_signed() * 0.1;
                            walk[idx]
                        }
                    };
                    q[idx] = (t as f32 * 0.03 + c as f32).cos();
                }
                for r in 0..val_dim {
                    v[h * val_dim + r] = match regime {
                        "sinusoid" => (t as f32 * 0.07 + r as f32).cos(),
                        _ => rng.next_signed(),
                    };
                }
                l2_normalize(&mut k[h * key_dim..(h + 1) * key_dim]);
                l2_normalize(&mut q[h * key_dim..(h + 1) * key_dim]);
            }
            let before: Vec<f32> = (0..n_v_heads)
                .map(|h| head_norm(&state, h, key_dim, val_dim))
                .collect();
            let out = gated_deltanet_step(
                &q, &k, &v, &mut state, &beta, &decay, n_v_heads, key_dim, val_dim,
            );
            assert!(out.iter().all(|x| x.is_finite()));
            for h in 0..n_v_heads {
                let after = head_norm(&state, h, key_dim, val_dim);
                let v_norm = v[h * val_dim..(h + 1) * val_dim]
                    .iter()
                    .map(|x| x * x)
                    .sum::<f32>()
                    .sqrt();
                let bound = decay[h] * before[h] + beta[h] * v_norm;
                assert!(
                    after <= bound + 1e-3,
                    "regime {regime} step {t} head {h}: driven norm {after} broke the \
                     exact envelope {bound} (erase expansive or injection over-bold)"
                );
            }
        }
    }

    // ── Arm 3 — the RELEASE: after any drive, cutting the input (v = 0)
    // makes the norm fall monotonically to the decay floor — the physical
    // signature of a contractive erase under the production posture.
    let mut rng = SplitMix64(SEED_A);
    let mut state = vec![0.0f32; state_dim];
    let mut walk = vec![0.0f32; n_v_heads * key_dim];
    let zero_v = vec![0.0f32; n_v_heads * val_dim];
    for t in 0..200u64 {
        let mut k = vec![0.0f32; n_v_heads * key_dim];
        let mut q = vec![0.0f32; n_v_heads * key_dim];
        let mut v = vec![0.0f32; n_v_heads * val_dim];
        for h in 0..n_v_heads {
            for c in 0..key_dim {
                let idx = h * key_dim + c;
                k[idx] = (t as f32 * 0.05 + c as f32).sin();
                q[idx] = (t as f32 * 0.03 + c as f32).cos();
                walk[idx] += rng.next_signed() * 0.1;
            }
            l2_normalize(&mut k[h * key_dim..(h + 1) * key_dim]);
            l2_normalize(&mut q[h * key_dim..(h + 1) * key_dim]);
            for r in 0..val_dim {
                v[h * val_dim + r] = (t as f32 * 0.07 + r as f32).cos();
            }
        }
        let _ = gated_deltanet_step(
            &q, &k, &v, &mut state, &beta, &decay, n_v_heads, key_dim, val_dim,
        );
    }
    let mut prev = state_norm(&state);
    for t in 200..260u64 {
        let mut k = vec![0.0f32; n_v_heads * key_dim];
        let mut q = vec![0.0f32; n_v_heads * key_dim];
        for h in 0..n_v_heads {
            for c in 0..key_dim {
                k[h * key_dim + c] = (t as f32 * 0.05 + c as f32).sin();
                q[h * key_dim + c] = (t as f32 * 0.03 + c as f32).cos();
            }
            l2_normalize(&mut k[h * key_dim..(h + 1) * key_dim]);
            l2_normalize(&mut q[h * key_dim..(h + 1) * key_dim]);
        }
        let _ = gated_deltanet_step(
            &q, &k, &zero_v, &mut state, &beta, &decay, n_v_heads, key_dim, val_dim,
        );
        let cur = state_norm(&state);
        assert!(
            cur <= prev + 1e-6,
            "release step {t}: state norm GREW {prev} -> {cur} with the input cut \
             (the erase's contraction did not dominate the decay)"
        );
        prev = cur;
    }
}

#[test]
fn cell_a_refused_posture_fails_the_predicate() {
    let (n_v_heads, key_dim, val_dim) = (2usize, 32usize, 32usize);
    let state_dim = n_v_heads * key_dim * val_dim;

    // β·‖k‖² = 6.0 per head: eigenvalue 1 − 6 = −5, the growth class.
    let beta = vec![6.0f32; n_v_heads];
    let decay = vec![1.0f32; n_v_heads]; // no decay — the pure erase dynamics
    let mut state = vec![0.0f32; state_dim];
    // Raw keys of norm √(β·‖k‖²/β) with β‖k‖² = 6: ‖k‖² = 6.
    let mut rng = SplitMix64(SEED_A ^ 0xBEEF);
    let mut k = vec![0.0f32; n_v_heads * key_dim];
    for v in k.iter_mut() {
        *v = rng.next_signed();
    }
    // Scale each head's k so ‖k_head‖² = 6 (deliberately NOT normalized).
    for h in 0..n_v_heads {
        let head = &k[h * key_dim..(h + 1) * key_dim];
        let norm_sq: f32 = head.iter().map(|x| x * x).sum();
        let scale = (6.0f32 / norm_sq).sqrt();
        for x in &mut k[h * key_dim..(h + 1) * key_dim] {
            *x *= scale;
        }
    }
    let v = vec![1.0f32; n_v_heads * val_dim];
    let q = vec![0.0f32; n_v_heads * key_dim];

    // The predicate the production gate asserts — the refused posture must
    // VIOLATE it within a few steps (the state norm grows monotonically
    // while the erase eigenvalue sits at −5).
    let mut grew = false;
    let mut prev = state_norm(&state);
    for _ in 0..16 {
        let _ = gated_deltanet_step(
            &q, &k, &v, &mut state, &beta, &decay, n_v_heads, key_dim, val_dim,
        );
        let cur = state_norm(&state);
        if cur > prev + 1e-6 {
            grew = true;
        }
        prev = cur;
    }
    assert!(
        grew,
        "the refused posture (beta*||k||^2 = 6 > 2, raw k) did NOT grow the state -          the gate's firing proof is vacuous"
    );
    assert!(
        !prev.is_finite() || prev > 1e3,
        "the refused posture should reach the blow-up class, got {prev}"
    );
}

// ── Cell (b) — the MemT channel-wise law on the deterministic grid ────────

/// The operator norm by power iteration on A^T A (d <= 16; deterministic —
/// no PRNG, start vector all-ones).
fn op_norm(a: &[Vec<f32>], d: usize) -> f32 {
    let mut at_a = vec![vec![0.0f32; d]; d];
    for i in 0..d {
        for j in 0..d {
            at_a[i][j] = (0..d).map(|c| a[c][i] * a[c][j]).sum();
        }
    }
    let mut x = vec![1.0f32; d];
    let mut lam = 0.0f32;
    for _ in 0..500 {
        let mut y = vec![0.0f32; d];
        for i in 0..d {
            y[i] = (0..d).map(|j| at_a[i][j] * x[j]).sum();
        }
        let n = y.iter().map(|v| v * v).sum::<f32>().sqrt();
        if n <= 1e-30 {
            return 0.0;
        }
        for (v, yv) in x.iter_mut().zip(&y) {
            *v = yv / n;
        }
        lam = n;
    }
    lam.sqrt()
}

/// Build A = I - k^T(b*k) — the MemT erase: unit-norm k, channel gate b.
fn erase_matrix(k: &[f32], b: &[f32], d: usize) -> Vec<Vec<f32>> {
    let mut a = vec![vec![0.0f32; d]; d];
    for i in 0..d {
        for j in 0..d {
            a[i][j] = if i == j { 1.0 } else { 0.0 } - b[j] * k[i] * k[j];
        }
    }
    a
}

/// The MemT sweep: a deterministic gate ladder over unit-norm k covers the
/// analytic peak rho = sum(b_i k_i^2) = 1/3 (+/- offsets), d in {2, 4, 16}.
/// Bars: unsymmetrized sup in [1.14, 1.155] (approaching 2/3^0.5 ~ 1.1547);
/// the symmetrized control u = sqrt(b)*k stays <= 1 + 1e-6 on the same grid.
#[test]
fn cell_b_memt_channelwise_law_grid() {
    for &d in &[2usize, 4usize, 16usize] {
        let mut sup_unsym = 0.0f32;
        let mut sup_sym = 0.0f32;
        let mut hit_peak = false;

        // The grid, per dimension: a family of unit-norm k vectors (from
        // concentrated to uniform) x a gate family {one-open, two-open,
        // uniform} — every (k, b) pair lands on a DETERMINISTIC rho, and
        // the rho = 1/3 point is included EXACTLY (the analytic peak).
        for &concentration in &[1.0f32 / 3.0, 0.6, 0.75, 0.9, 1.0 / (d as f32).sqrt()] {
            // Unit-norm k: channel 0 carries `concentration` of the mass.
            let mut k = vec![0.0f32; d];
            k[0] = concentration.sqrt();
            let rest = ((1.0 - concentration) / (d - 1) as f32).sqrt();
            for v in k.iter_mut().skip(1) {
                *v = rest;
            }
            let kn: f32 = k.iter().map(|x| x * x).sum::<f32>().sqrt();
            for v in k.iter_mut() {
                *v /= kn;
            }
            let k2: Vec<f32> = k.iter().map(|x| x * x).collect();

            // Gate families, deterministic: one-open (b = e_0 scaled to hit
            // a target rho), uniform-mean, and the full-open edge.
            let families: Vec<Vec<f32>> = {
                let mut f = Vec::new();
                // one-open: b_0 = min(rho / k2_0, 1) for rho across the grid
                for &rho in &[1.0f32 / 3.0, 0.25, 0.5, 0.6] {
                    let mut b = vec![0.0f32; d];
                    b[0] = (rho / k2[0]).min(1.0);
                    f.push(b);
                }
                // uniform gate: b_i = rho (rho = mean(k2)·… exact rho = rho)
                for &rho in &[1.0f32 / 3.0, 0.5, 0.8] {
                    f.push(vec![rho; d]);
                }
                // alternating high/low around 1/3 (non-uniform b, uniform k
                // independent of k2 — the non-normality source)
                {
                    let mut b = vec![0.0f32; d];
                    for (i, slot) in b.iter_mut().enumerate() {
                        *slot = if i % 2 == 0 { 2.0 / 3.0 } else { 0.0 };
                    }
                    f.push(b);
                }
                f
            };

            for b in &families {
                let rho: f32 = (0..d).map(|i| b[i] * k2[i]).sum();
                if (rho - 1.0 / 3.0).abs() < 1e-4 {
                    hit_peak = true;
                }

                sup_unsym = sup_unsym.max(op_norm(&erase_matrix(&k, b, d), d));

                // The symmetrized control: u = sqrt(b) * k -> I - u^T u.
                let u: Vec<f32> = (0..d).map(|i| b[i].sqrt() * k[i]).collect();
                let mut sym = vec![vec![0.0f32; d]; d];
                for i in 0..d {
                    for j in 0..d {
                        sym[i][j] = if i == j {
                            1.0 - u[i] * u[j]
                        } else {
                            -u[i] * u[j]
                        };
                    }
                }
                sup_sym = sup_sym.max(op_norm(&sym, d));
            }
        }

        // The falsification guard: the grid MUST have landed on rho = 1/3
        // (a vacuous sweep proves nothing — the Issue-940-class green zero).
        assert!(
            hit_peak,
            "d={d}: the deterministic grid never hit rho = 1/3 — the sweep is              vacuous; fix the grid before asserting the law"
        );
        assert!(
            (1.14..=1.155).contains(&sup_unsym),
            "d={d}: unsymmetrized sup = {sup_unsym} outside [1.14, 1.155] —              the MemT premise FAILED in our parameterization (the falsification              arm fires; record the negative and drop cell (b))"
        );
        assert!(
            sup_sym <= 1.0 + 1e-6,
            "d={d}: the symmetrized control sup = {sup_sym} > 1 + 1e-6 —              the non-expansive-by-construction claim failed"
        );
    }
}
