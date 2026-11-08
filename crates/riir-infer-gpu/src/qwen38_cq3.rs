//! CQ3 never-unpack GEMV prototype — the Cactus whole-model-quantization
//! recipe's kernel half, fixture lane for the qwen3.8 dense league cell.
//!
//! Provenance: Cactus, "Whole-Model Quantization Below 4 Bits Per Weight for
//! On-Device Multimodal Models" (NeurIPS 2026, Mosoyan et al.) via katgpt-rs
//! Research 657 §A3 + issue 987 lead A3; this repo's issue 1186 T1. The lane
//! is the qwen3.8 dense league lane (issue 1176 made the dense Metal posture
//! the league default; NOT Bonsai — ternary-trained, wrong input class).
//!
//! The recipe (activation-facing tensors; steps 1–4 of the paper's five).
//! Step 5 — the GPTQ-style Hessian residual — is DEFERRED to T2: it consumes
//! the same calibration samples the matched-calibration arm fits, so the
//! residual pass lands with the harness, not with this kernel fixture.
//!
//! 1. SmoothQuant fold `s_k = clip(diag(H)^α/|w|^(1−α), 1/8, 8)` — the fold
//!    arrives here as a caller-supplied per-input-channel scale (the H
//!    census is T2's calibration half; the fixture synthesizes one). Folded
//!    INTO the weights (`w·s`); the activation is divided (`x/s`).
//! 2. Fixed structured rotation `R_g = (1/√g)·S₂·H_g·S₁·P` — deterministic
//!    seeded sign patterns + a fixed permutation (replaces QuIP#/
//!    TurboQuant's randomized rotations; zero storage — the parameters
//!    re-derive from `g` alone). The FWHT half rides katgpt-core's
//!    `walsh_hadamard_in_place_normalized` (substrate-first, meld.rs).
//! 3. FP16 group norm → unit sphere: each (row × input-group of `g`) block
//!    is divided by its norm, and THAT norm is stored as FP16 — 2 B per g
//!    weights = `16/g` bits/weight (at g = 128 that is the paper's
//!    `b + 0.125`). Quantize divides by the round-tripped FP16 norm; the
//!    dequant/GEMV multiplies it back — one norm, never two.
//! 4. ONE scalar Lloyd–Max codebook per `(g, b)`, trained on a SYNTHETIC
//!    random-unit-vector distribution (never on the weights), shared across
//!    every tensor and model; weights store b-bit nearest-centroid indices.
//!
//! The kernel never unpacks: the activation is rotated once per vector
//! (blockwise `R_g`), then each group contributes through the codebook LUT
//! with the FP16 norm scale fused into the product — no dequantized weight
//! tensor is ever materialized. Two arms:
//!
//! - [`CqTensor::gemv_f32_lut`] — f32 LUT gather, BIT-EXACT against
//!   reference dequant + plain matmul (the G1 fixture pins `to_bits`
//!   equality; sequential accumulation order per the Issue-940 determinism
//!   law; fixed first-index tie rule on nearest-centroid).
//! - [`CqTensor::gemv_i8_sdot`] — the phone-kernel shape: activation
//!   quantized to int8 per group post-rotation, centroids pre-rounded to
//!   int8, integer dot with i32 accumulate (the NEON SDOT semantic; the
//!   per-group i32 bound `g·127² ≤ 2.06M` and the per-row de-scale order
//!   are documented at the kernel). Measured against the f32 arm under a
//!   pinned tolerance — int8 is a second lossy surface and is NOT claimed
//!   bit-exact.
//!
//! Opt-in (`cq3`) until the T3 league cell passes the GOAT gate; never in
//! the release set before that. `b_eff = b + 16/g` bits/weight is asserted
//! computationally by the g1e fixture. This module is the CPU fixture lane:
//! indices ride a u8 carrier (b ≤ 8) and the packing into a b-bit wire
//! format is the GPU lane's job at T3.

use katgpt_core::meld::walsh_hadamard_in_place_normalized;

/// Group sizes the structured rotation supports (`g` must divide `d_in`).
/// g = 128 is the paper's `b_eff = b + 0.125` shape.
pub const SUPPORTED_G: [usize; 5] = [16, 32, 64, 128, 256];

/// Deterministic derivation seed for the rotation's sign patterns and
/// permutation. Fixed at crate lifetime — the rotation is part of the wire
/// contract once any artifact ships (changing it forks every quantized
/// tensor).
pub const ROTATION_SEED: u64 = 0x1186_C0DE_0000_0001;

/// Deterministic derivation seed for the synthetic codebook training set.
/// Same contract note as [`ROTATION_SEED`].
pub const CODEBOOK_SEED: u64 = 0x1186_C0DE_0000_0002;

/// Synthetic unit vectors sampled per `(g, b)` codebook build. Trained
/// NEVER on real weights — the paper's shared-across-models property.
/// 2048 vectors × g components keeps the fixture battery fast while the
/// Lloyd–Max means stay stable (262k samples at g = 128).
const CODEBOOK_UNIT_VECTORS: usize = 2048;

/// Lloyd–Max iteration cap. Convergence is checked by exact-centroid drift
/// (f64 means stop moving); 32 sweeps is far past the plateau for 2^b ≤ 256
/// on a 262k-sample set — the cap just bounds the worst case
/// deterministically.
const CODEBOOK_MAX_ITERS: usize = 32;

/// Deterministic xorshift64* RNG for fixture/training derivations only —
/// never for production sampling. Every consumer seeds it from a named
/// const (see [`ROTATION_SEED`]/[`CODEBOOK_SEED`]); no global RNG state.
pub struct Xorshift64Star {
    state: u64,
    pending: Option<f32>,
}

impl Xorshift64Star {
    /// `seed == 0` clamps to 1 (xorshift is stuck at zero).
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.max(1),
            pending: None,
        }
    }

    #[must_use]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform `[0, 1)` at 24-bit resolution.
    #[must_use]
    pub fn next_f32(&mut self) -> f32 {
        f32::from_bits(0x3F80_0000 | ((self.next_u64() >> 40) as u32)) - 1.0
    }

    /// Standard normal via Box–Muller; the second of each uniform pair is
    /// cached, so draws come out in a fixed deterministic sequence.
    #[must_use]
    pub fn next_normal(&mut self) -> f32 {
        if let Some(v) = self.pending.take() {
            return v;
        }
        let u1 = self.next_f32().max(f32::MIN_POSITIVE); // log(0) guard
        let u2 = self.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f32::consts::TAU * u2;
        self.pending = Some(r * theta.sin());
        r * theta.cos()
    }
}

/// The fixed structured rotation `R_g = (1/√g)·S₂·H_g·S₁·P` for one group
/// size. Orthogonal by construction (sign patterns are ±1 diagonals, H_g is
/// the normalized Walsh–Hadamard, P is a permutation); applied identically
/// to weight row-blocks (as column vectors) and activation blocks, so
/// `(W·Rᵀ)(R·x) = W·x` exactly in exact arithmetic.
pub struct Rotation {
    g: usize,
    s1: Vec<f32>,
    s2: Vec<f32>,
    perm: Vec<u32>,
}

impl Rotation {
    /// Derive the fixed rotation for `g` (must be in [`SUPPORTED_G`]).
    /// Deterministic in (seed, g): same inputs, byte-identical parameters.
    #[must_use]
    pub fn fixed(g: usize) -> Self {
        assert!(
            SUPPORTED_G.contains(&g),
            "cq3 rotation: g must be one of {SUPPORTED_G:?}, got {g}"
        );
        let mut rng = Xorshift64Star::new(ROTATION_SEED ^ ((g as u64) << 32));
        let mut s1 = Vec::with_capacity(g);
        let mut s2 = Vec::with_capacity(g);
        for _ in 0..g {
            s1.push(if rng.next_f32() < 0.5 { 1.0 } else { -1.0 });
            s2.push(if rng.next_f32() < 0.5 { 1.0 } else { -1.0 });
        }
        // Fisher–Yates with the seeded stream (u64 modulo — deterministic
        // and documented; this is derivation-time, not a hot loop).
        let mut perm: Vec<u32> = (0..g as u32).collect();
        for i in (1..g).rev() {
            let j = (rng.next_u64() % (i as u64 + 1)) as usize;
            perm.swap(i, j);
        }
        Self { g, s1, s2, perm }
    }

    #[must_use]
    pub fn g(&self) -> usize {
        self.g
    }

    /// Apply to one `g`-length block IN PLACE: `x ↦ S₂·FWHT(S₁·P·x)`.
    /// (The normalized FWHT carries the `1/√g`.)
    fn apply_block(&self, blk: &mut [f32]) {
        debug_assert_eq!(blk.len(), self.g);
        let g = self.g;
        // P — gather through the permutation.
        let mut tmp = vec![0.0_f32; g];
        for (dst, &src) in tmp.iter_mut().zip(self.perm.iter()) {
            *dst = blk[src as usize];
        }
        // S₁ — elementwise sign.
        for (v, s) in tmp.iter_mut().zip(self.s1.iter()) {
            *v *= s;
        }
        // H_g / √g — the substrate's normalized Walsh–Hadamard butterfly.
        fwht_block(&mut tmp);
        // S₂ — elementwise sign.
        for (v, s) in tmp.iter_mut().zip(self.s2.iter()) {
            *v *= s;
        }
        blk.copy_from_slice(&tmp);
    }

    /// Apply blockwise across a whole vector (`len` must be a multiple of
    /// `g`). Allocation-free when `out` arrives with sufficient capacity
    /// (the caller-owned scratch convention).
    pub fn apply(&self, x: &[f32], out: &mut Vec<f32>) {
        assert_eq!(
            x.len() % self.g,
            0,
            "cq3 rotation: length {} not a multiple of g = {}",
            x.len(),
            self.g
        );
        out.clear();
        out.extend_from_slice(x);
        for blk in out.chunks_exact_mut(self.g) {
            self.apply_block(blk);
        }
    }
}

/// Dispatch the normalized FWHT over the supported group sizes (the
/// substrate primitive is const-generic; the supported set IS the paper's).
fn fwht_block(x: &mut [f32]) {
    match x.len() {
        16 => walsh_hadamard_in_place_normalized::<16>(x.try_into().expect("len 16")),
        32 => walsh_hadamard_in_place_normalized::<32>(x.try_into().expect("len 32")),
        64 => walsh_hadamard_in_place_normalized::<64>(x.try_into().expect("len 64")),
        128 => walsh_hadamard_in_place_normalized::<128>(x.try_into().expect("len 128")),
        256 => walsh_hadamard_in_place_normalized::<256>(x.try_into().expect("len 256")),
        other => unreachable!("rotation::fixed validated g; got {other}"),
    }
}

/// One scalar Lloyd–Max codebook per `(g, b)` — trained on a synthetic
/// random-unit-vector distribution (the post-group-norm value distribution),
/// never on weights. Centroids are sorted ascending; `c8`/`s_c` are the
/// pre-rounded int8 twin consumed by [`CqTensor::gemv_i8_sdot`].
pub struct Codebook {
    pub b: u32,
    /// `2^b` centroids, ascending.
    pub centroids: Vec<f32>,
    /// Centroids pre-rounded to int8 at `s_c` resolution.
    pub c8: Vec<i8>,
    /// The int8 resolution: `absmax(centroids) / 127`.
    pub s_c: f32,
}

impl Codebook {
    /// Build deterministically from `(g, b)` (seeded synthetic set only).
    #[must_use]
    pub fn build(g: usize, b: u32) -> Self {
        assert!((1..=8).contains(&b), "cq3 codebook: b in 1..=8, got {b}");
        let mut rng = Xorshift64Star::new(CODEBOOK_SEED ^ ((g as u64) << 32) ^ (u64::from(b)));
        // Synthetic training set: components of g-dim unit vectors (the
        // recipe's post-group-norm distribution).
        let mut sample: Vec<f32> = Vec::with_capacity(CODEBOOK_UNIT_VECTORS * g);
        let mut buf = vec![0.0_f32; g];
        for _ in 0..CODEBOOK_UNIT_VECTORS {
            for v in buf.iter_mut() {
                *v = rng.next_normal();
            }
            let mut ss = 0.0_f32;
            for v in &buf {
                ss += v * v;
            }
            let inv = if ss > f32::EPSILON {
                1.0 / ss.sqrt()
            } else {
                0.0
            };
            for v in &buf {
                sample.push(v * inv);
            }
        }
        sample.sort_by(f32::total_cmp);

        // Lloyd–Max: quantile init on the sorted sample, then assign/mean
        // iterations (f64 means for conditioning; derivation-time lane).
        let k = 1usize << b;
        let n = sample.len();
        let mut centroids: Vec<f64> = (0..k)
            .map(|i| {
                let q = (i as f64 + 0.5) / k as f64;
                let idx = ((q * (n - 1) as f64).round() as usize).min(n - 1);
                f64::from(sample[idx])
            })
            .collect();
        for _ in 0..CODEBOOK_MAX_ITERS {
            let mut sums = vec![0.0_f64; k];
            let mut counts = vec![0_u64; k];
            for (ci, &v) in sample
                .iter()
                .map(|v| (nearest_f64(&centroids, f64::from(*v)), v))
            {
                sums[ci] += f64::from(v);
                counts[ci] += 1;
            }
            let mut drift = 0.0_f64;
            for (c, (&s, &cnt)) in centroids.iter_mut().zip(sums.iter().zip(counts.iter())) {
                if cnt > 0 {
                    let mean = s / cnt as f64;
                    drift += (mean - *c).abs();
                    *c = mean;
                }
            }
            if drift == 0.0 {
                break;
            }
        }
        let mut centroids: Vec<f32> = centroids.iter().map(|c| *c as f32).collect();
        centroids.sort_by(f32::total_cmp);
        let absmax = centroids.iter().fold(0.0_f32, |m, c| m.max(c.abs()));
        let s_c = if absmax > 0.0 { absmax / 127.0 } else { 1.0 };
        let c8: Vec<i8> = centroids
            .iter()
            .map(|c| ((c / s_c).round() as i32).clamp(-127, 127) as i8)
            .collect();
        Self {
            b,
            centroids,
            c8,
            s_c,
        }
    }

    /// Nearest-centroid index; ties resolve to the LOWER index (first-wins
    /// strict `<` — the determinism law's fixed tie rule). Linear over
    /// `2^b ≤ 256` — fixture-lane cost; the T3 GPU lane precomputes
    /// partition points instead.
    #[must_use]
    pub fn nearest(&self, v: f32) -> u8 {
        let mut best = 0_usize;
        let mut best_d = (v - self.centroids[0]).abs();
        for (i, &c) in self.centroids.iter().enumerate().skip(1) {
            let d = (v - c).abs();
            if d < best_d {
                best_d = d;
                best = i;
            }
        }
        best as u8
    }
}

/// First-wins nearest-centroid over f64 centroids (Lloyd–Max inner loop).
fn nearest_f64(centroids: &[f64], v: f64) -> usize {
    let mut best = 0_usize;
    let mut best_d = (v - centroids[0]).abs();
    for (i, &c) in centroids.iter().enumerate().skip(1) {
        let d = (v - c).abs();
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best
}

/// The quantized tensor: recipe steps 1–4 applied, weights stored as b-bit
/// codebook indices + per-(row, input-group) FP16 norm scales. Carries its
/// rotation and codebook so the kernels are single-argument.
pub struct CqTensor {
    pub d_out: usize,
    pub d_in: usize,
    pub g: usize,
    pub b: u32,
    pub rotation: Rotation,
    pub codebook: Codebook,
    /// SmoothQuant fold (per input channel); `None` = unfused. The GEMV
    /// divides the activation by this before rotating (`x' = x/s`).
    pub smooth: Option<Vec<f32>>,
    /// b-bit nearest-centroid index per weight (u8 carrier, b ≤ 8; the
    /// packed b-bit wire format is the T3 GPU lane's job).
    pub indices: Vec<u8>,
    /// FP16 group-norm bits per (row, input-group): `d_out · (d_in / g)`
    /// entries. The stored value is the group's norm — quantize divided by
    /// it, dequant multiplies it back.
    pub scales: Vec<u16>,
}

impl CqTensor {
    /// Quantize `w` (row-major `d_out × d_in`) under the recipe: optional
    /// SmoothQuant fold → blockwise structured rotation → FP16 group norm
    /// (divide by the round-tripped FP16 norm; the norm itself is what is
    /// stored) → nearest-centroid indices. Dequantization reconstructs
    /// exactly the values the indices were chosen against (no double
    /// rounding mismatch).
    pub fn quantize(
        w: &[f32],
        d_out: usize,
        d_in: usize,
        g: usize,
        b: u32,
        smooth: Option<&[f32]>,
    ) -> Self {
        assert_eq!(w.len(), d_out * d_in, "cq3 quantize: w len mismatch");
        assert_eq!(d_in % g, 0, "cq3 quantize: d_in must be a multiple of g");
        assert!(
            smooth.is_none() || smooth.expect("checked").len() == d_in,
            "cq3 quantize: smooth len mismatch"
        );
        let rotation = Rotation::fixed(g);
        let codebook = Codebook::build(g, b);
        let groups_in = d_in / g;
        let mut indices = vec![0_u8; d_out * d_in];
        let mut scales = vec![0_u16; d_out * groups_in];
        let mut block = vec![0.0_f32; g];
        for i in 0..d_out {
            let row = &w[i * d_in..(i + 1) * d_in];
            for grp in 0..groups_in {
                for m in 0..g {
                    let mut v = row[grp * g + m];
                    if let Some(s) = smooth {
                        v *= s[grp * g + m];
                    }
                    block[m] = v;
                }
                rotation.apply_block(&mut block);
                let mut ss = 0.0_f32;
                for v in &block {
                    ss += v * v;
                }
                // The stored FP16 scale is the group NORM (the paper's "FP16
                // group norm → unit sphere"): quantize DIVIDES by it, the
                // dequant/GEMV multiplies it back — one norm, never two.
                let norm = if ss > f32::EPSILON { ss.sqrt() } else { 0.0 };
                let norm_f16 = half::f16::from_f32(norm);
                let gf = norm_f16.to_f32();
                scales[i * groups_in + grp] = norm_f16.to_bits();
                for m in 0..g {
                    indices[i * d_in + grp * g + m] = codebook.nearest(block[m] / gf);
                }
            }
        }
        Self {
            d_out,
            d_in,
            g,
            b,
            rotation,
            codebook,
            smooth: smooth.map(|s| s.to_vec()),
            indices,
            scales,
        }
    }

    /// Reference dequantization: materialize `f32(f16(norm)) · centroid[idx]`
    /// — the norm (stored per group) multiplies the centroid back up. The
    /// G1 bit-exactness fixture compares [`Self::gemv_f32_lut`] against a
    /// plain matmul over THIS matrix — proving the fused LUT kernel
    /// preserves exact per-element semantics.
    #[must_use]
    pub fn dequant(&self) -> Vec<f32> {
        let groups_in = self.d_in / self.g;
        let mut out = vec![0.0_f32; self.d_out * self.d_in];
        for i in 0..self.d_out {
            for grp in 0..groups_in {
                let gf = half::f16::from_bits(self.scales[i * groups_in + grp]).to_f32();
                let base = i * self.d_in + grp * self.g;
                for m in 0..self.g {
                    out[base + m] = self.codebook.centroids[self.indices[base + m] as usize] * gf;
                }
            }
        }
        out
    }

    /// Rotate (and optionally SmoothQuant-divide) the activation: the
    /// never-unpack kernel's whole activation-side work, done ONCE per
    /// vector — `(W·Rᵀ)(R·x) = W·x` with the smooth fold folded in.
    #[must_use]
    pub fn rotate_activation(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.d_in, "cq3 rotate_activation: len mismatch");
        let mut out = Vec::with_capacity(self.d_in);
        if let Some(s) = &self.smooth {
            out.extend(x.iter().zip(s.iter()).map(|(&v, &s)| v / s));
        } else {
            out.extend_from_slice(x);
        }
        for blk in out.chunks_exact_mut(self.g) {
            self.rotation.apply_block(blk);
        }
        out
    }

    /// The f32 fused-LUT GEMV: per group, gather centroids by index, fuse
    /// the FP16 norm scale, accumulate sequentially in f32. BIT-EXACT
    /// against `dequant()` + a plain sequential matmul on the same `x_rot`
    /// (same two rounded products per element, same accumulation order —
    /// the G1 fixture pins `to_bits` equality).
    #[must_use]
    pub fn gemv_f32_lut(&self, x_rot: &[f32]) -> Vec<f32> {
        assert_eq!(x_rot.len(), self.d_in, "cq3 gemv_f32_lut: len mismatch");
        let groups_in = self.d_in / self.g;
        let mut out = vec![0.0_f32; self.d_out];
        for (i, out_i) in out.iter_mut().enumerate() {
            let mut acc = 0.0_f32;
            for grp in 0..groups_in {
                let gf = half::f16::from_bits(self.scales[i * groups_in + grp]).to_f32();
                let base = i * self.d_in + grp * self.g;
                for m in 0..self.g {
                    let w = self.codebook.centroids[self.indices[base + m] as usize] * gf;
                    acc += w * x_rot[grp * self.g + m];
                }
            }
            *out_i = acc;
        }
        out
    }

    /// The int8 SDOT-shaped arm (the phone kernel's execution shape): the
    /// rotated activation is quantized to int8 PER GROUP (absmax/127),
    /// centroids are pre-rounded to int8 at `s_c`, and each group
    /// contributes an i32 dot (NEON SDOT semantic — `g·127² ≤ 2.06M` per
    /// group, comfortably inside i32). The per-group de-scale
    /// `f32(acc)·s_c·s_x·f32(f16(norm))` runs at the group boundary in that
    /// fixed left-associated order (documented; a tolerance fixture, not a
    /// bit-exactness claim — int8 is a second lossy surface).
    ///
    /// The activation quantization is staged ONCE per call (it does not
    /// depend on the output row), matching how a real kernel front-loads it.
    #[must_use]
    pub fn gemv_i8_sdot(&self, x_rot: &[f32]) -> Vec<f32> {
        assert_eq!(x_rot.len(), self.d_in, "cq3 gemv_i8_sdot: len mismatch");
        debug_assert!(
            self.d_in * 127 * 127 < i32::MAX as usize,
            "cq3 gemv_i8_sdot: per-row i32 bound d_in·127² exceeded"
        );
        let g = self.g;
        let groups_in = self.d_in / g;
        // Stage the int8 activation per group once.
        let mut x8 = vec![0_i32; self.d_in];
        let mut s_x = vec![0.0_f32; groups_in];
        for grp in 0..groups_in {
            let seg = &x_rot[grp * g..(grp + 1) * g];
            let absmax = seg.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
            let s = absmax / 127.0;
            s_x[grp] = s;
            let base = grp * g;
            if s > 0.0 {
                for (m, v) in seg.iter().enumerate() {
                    x8[base + m] = ((v / s).round() as i32).clamp(-127, 127);
                }
            }
        }
        let mut out = vec![0.0_f32; self.d_out];
        for (i, out_i) in out.iter_mut().enumerate() {
            let mut y = 0.0_f32;
            for grp in 0..groups_in {
                let gf = half::f16::from_bits(self.scales[i * groups_in + grp]).to_f32();
                let base = i * self.d_in + grp * g;
                let mut acc = 0_i32;
                for m in 0..g {
                    acc += i32::from(self.codebook.c8[self.indices[base + m] as usize])
                        * x8[grp * g + m];
                }
                y += acc as f32 * self.codebook.s_c * s_x[grp] * gf;
            }
            *out_i = y;
        }
        out
    }

    /// The wire bits-per-weight: `b` index bits + one FP16 scale per group
    /// (`16/g`). At `g = 128` this is the paper's `b_eff = b + 0.125`.
    /// The u8 carrier in this fixture lane is NOT the wire — this reports
    /// the packed b-bit figure the T3 GPU lane ships.
    #[must_use]
    pub fn bits_per_weight(&self) -> f64 {
        let n = (self.d_out * self.d_in) as f64;
        (n * f64::from(self.b) + (self.scales.len() as f64) * 16.0) / n
    }
}
