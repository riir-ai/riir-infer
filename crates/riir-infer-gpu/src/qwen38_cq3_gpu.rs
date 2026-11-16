//! Issue 1186 T3 — the CQ3 serving arm's CubeCL kernels (the decode GEMV +
//! the activation-prep pass + the batched prefill shape).
//!
//! Scope (the serving src/ lease, items a–c): the never-unpack GEMV in the
//! paper's i8-SDOT shape, the activation side (SmoothQuant divide + the
//! blockwise structured rotation `S₂·FWHT(S₁·P·x)`), and the prefill half
//! as option (i) — the same GEMV shape run p-times in ONE batched launch
//! (the paper has no CQ prefill path; the cell discloses this arm; the
//! novel tile GEMM (ii) is the named follow-up).
//!
//! # Bit-exactness tiers (the correctness bar)
//!
//! - **Tiled arms are `to_bits`-exact** against the host reference
//!   ([`PackedCq3Tensor::rotate_activation`] +
//!   [`PackedCq3Tensor::gemv_i8_sdot`]): the act-prep workgroup replays the
//!   host butterfly's exact serial op order (P gather → S₁ signs → the
//!   `s = a+b; d = a-b` butterfly in the same chunk/stage order → S₂
//!   signs → the single `inv` multiply), and the tiled GEMV replays the
//!   host's sequential group fold (absmax fold, element-local int8
//!   staging, i32 mads, the fixed left-associated
//!   `f32(acc)·s_c·s_x·gf` de-scale). No fast-math attribute — strict
//!   IEEE add/sub/mul, matching the incumbent lane's contraction posture.
//! - **Plane arms are the tolerance tier**: each lane accumulates its
//!   groups' de-scaled contributions sequentially, but the cross-lane
//!   `plane_sum` is the hardware tree — a different (order-defined)
//!   accumulation than the host's. Same numeric class as the incumbent
//!   Q4_K plane kernel; the fixture pins the tolerance, the league cell
//!   reads the plane arm.
//!
//! # Device form
//!
//! Packed indices ride **u32 words** (LSB-first; repacked from the
//! container's u64 form at load — WGSL-class integer ops are 32-bit).
//! Scales upload PRE-EXPANDED to f32 (the host `half` conversion is the
//! exact reference; +4% device footprint, disclosed — removes the whole
//! in-kernel f16-subnormal class, the Issue-593 trap). The codebook's int8
//! centroids upload as i32. With `g·b` bits per group the per-group word
//! count is `ceil(g·b/32)`; at the pre-registered g=32/b=3 that is exactly
//! 3 words per group.
//!
//! Opt-in `cq3` (+ `cubecl_runtime`), never in `production` before the T3
//! GOAT (the env-switch law). Engagement proof: the construction line +
//! the per-pass launch counters on [`Cq3Handle`] (the Issue-1124 T1.5
//! lesson).

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

use crate::qwen38_cq3_serving::PackedCq3Tensor;

/// Pack one `b`-bit index stream into whole `u32` words (LSB-first, padded
/// to a word boundary) — the DEVICE upload form. Values must fit `b` bits.
/// The container's u64 form repacks through the u8 carrier here (the
/// battery pins pack/unpack identity both ways).
#[must_use]
pub fn pack_indices_b32(carrier: &[u8], b: u32) -> Vec<u32> {
    assert!((1..=8).contains(&b), "cq3 pack32: b in 1..=8, got {b}");
    let bits = carrier.len() as u64 * u64::from(b);
    let n_words = bits.div_ceil(32) as usize;
    let mut words = vec![0_u32; n_words];
    for (i, &v) in carrier.iter().enumerate() {
        assert!(
            u32::from(v) < (1_u32 << b),
            "cq3 pack32: carrier value {v} does not fit {b} bits"
        );
        let bit = i as u64 * u64::from(b);
        let w = (bit / 32) as usize;
        let off = (bit % 32) as u32;
        words[w] |= u32::from(v) << off;
        if off + b > 32 {
            words[w + 1] |= u32::from(v) >> (32 - off);
        }
    }
    words
}

/// Words per group on the device form: `ceil(g·b / 32)`.
#[must_use]
pub fn words_per_group(g: usize, b: u32) -> usize {
    (g as u64 * u64::from(b)).div_ceil(32) as usize
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

/// Activation prep — one WORKGROUP per (group, token), thread 0 replays the
/// host butterfly's exact serial op order into a shared scratch slice:
/// SmoothQuant divide → P gather → S₁ signs → the normalized FWHT butterfly
/// (`s = a+b; d = a-b` per pair, same chunk/stage order as
/// `walsh_hadamard_in_place_normalized`) → S₂ signs → one `inv` multiply
/// per element.
///
/// Batched shape: X = groups, Y = token (`p` columns of the batched input).
/// `params = [g, smooth_len]`; `inv` is a separate arg computed HOST-side
/// with the host's exact expression (`1.0 / (g as f32).sqrt()`).
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn cq3_act_prep(
    x: &[f32],
    smooth: &[f32],
    perm: &[u32],
    s1: &[f32],
    s2: &[f32],
    out: &mut [f32],
    params: &[f32],
) {
    let g = params[0usize] as u32;
    let smooth_len = params[1usize] as u32;
    let inv = params[2usize];
    // tokens rides params — the batch buffer holds p·d_in, so the per-token
    // dims MUST derive through it (the x.len()/g trap: at p>1 that reads
    // p× the real group count and corrupts every index).
    let tokens = params[3usize] as u32;
    let d_in = (x.len() as u32) / tokens.max(1u32);
    let groups = d_in / g;
    // One WORKGROUP per (group, token): the group id is the CUBE id, never
    // the global thread id (ABSOLUTE_POS_X spans all workgroups' threads —
    // with CubeDim 32 that misroutes every group past 0 to terminate).
    let grp = CUBE_POS_X;
    let pos = CUBE_POS_Y;
    if grp >= groups || pos >= tokens {
        terminate!();
    }
    let base = pos * d_in + grp * g;

    // g ≤ SUPPORTED_G's 256 (the recipe's supported set); the slice is the
    // comptime max, the runtime g bounds every loop.
    let mut scratch = Shared::<[f32]>::new_slice(comptime!(256usize));

    if UNIT_POS == 0u32 {
        // SmoothQuant divide + P gather + S₁ signs, element order 0..g.
        // The host divides FIRST (divided[j] = x[j]/s[j]), THEN gathers —
        // so the smooth index is the SOURCE slot perm[m], never m.
        let mut m = 0u32;
        while m < g {
            let src_slot = perm[m as usize];
            let src = x[(base + src_slot) as usize];
            // Two if blocks, never a conditional expression as value — the
            // NativeExpand macro bug (the sign_extend_i8 precedent).
            let mut v = src;
            if smooth_len > 0u32 {
                v = src / smooth[(base + src_slot) as usize];
            }
            v *= s1[m as usize];
            scratch[m as usize] = v;
            m += 1u32;
        }
        // The normalized FWHT butterfly — the host's exact stage/chunk/pair
        // order (`s = a + b; d = a - b`, computed before either store).
        let mut h = 1u32;
        while h < g {
            let mut c = 0u32;
            while c < g {
                let mut i = 0u32;
                while i < h {
                    let a = scratch[(c + i) as usize];
                    let b = scratch[(c + i + h) as usize];
                    scratch[(c + i) as usize] = a + b;
                    scratch[(c + i + h) as usize] = a - b;
                    i += 1u32;
                }
                c += 2u32 * h;
            }
            h <<= 1u32;
        }
        // S₂ signs, then the single normalized multiply per element.
        let mut j = 0u32;
        while j < g {
            let mut v = scratch[j as usize];
            v *= s2[j as usize];
            v *= inv;
            out[(base + j) as usize] = v;
            j += 1u32;
        }
    }
}

/// Int8 staging — one WORKGROUP per (group, token), thread 0 runs the
/// host's exact serial order: the absmax fold, `s = absmax/127`, then the
/// element-local `(v/s).round()` clamp ±127.
///
/// This pass carries the ONLY divides in the serving pipeline (Metal's f32
/// divide is 1-ulp-off vs the host's correctly-rounded IEEE divide — the
/// recorded finding, the unit probe pins it) — so the GEMV kernels below
/// are divide-FREE and `to_bits`-exact given identical staged inputs.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn cq3_stage_quant(x_rot: &[f32], x8: &mut [i32], s_x: &mut [f32], params: &[f32]) {
    let g = params[0usize] as u32;
    let tokens = params[3usize] as u32;
    // One WORKGROUP per (group, token) — CUBE ids, never ABSOLUTE_POS (the
    // misroute trap the act-prep probe caught). The per-token dims derive
    // through tokens (the x.len()/g batch trap).
    let grp = CUBE_POS_X;
    let pos = CUBE_POS_Y;
    let d_in = (x_rot.len() as u32) / tokens.max(1u32);
    let groups = d_in / g;
    if grp >= groups || pos >= tokens {
        terminate!();
    }
    let base = pos * d_in + grp * g;

    if UNIT_POS == 0u32 {
        let mut absmax = f32::new(0.0f32);
        let mut m = 0u32;
        while m < g {
            let v = x_rot[(base + m) as usize];
            let mut a = v;
            if v < f32::new(0.0f32) {
                a = -v;
            }
            if a > absmax {
                absmax = a;
            }
            m += 1u32;
        }
        let s = absmax / f32::new(127.0f32);
        s_x[(pos * groups + grp) as usize] = s;
        let mut m2 = 0u32;
        while m2 < g {
            let v = x_rot[(base + m2) as usize];
            let mut q = f32::new(0.0f32);
            if s > f32::new(0.0f32) {
                q = (v / s).round();
            }
            let mut qi = i32::cast_from(q);
            if qi > i32::new(127i64) {
                qi = i32::new(127i64);
            }
            if qi < i32::new(-127i64) {
                qi = i32::new(-127i64);
            }
            x8[(base + m2) as usize] = qi;
            m2 += 1u32;
        }
    }
}

/// Tiled CQ3 GEMV (the `to_bits` tier) — one thread per (row, token), the
/// host's exact sequential group fold over the STAGED inputs: the packed
/// b-bit index reads, the i32 dot over the int8 codebook, and the fixed
/// left-associated de-scale `f32(acc)·s_c·s_x·gf` accumulated into `y`.
/// NO divides anywhere — given identical x8/s_x bits the outputs are
/// bit-identical to [`crate::qwen38_cq3_serving::PackedCq3Tensor::gemv_i8_sdot`].
///
/// The de-scale rides a SHARED-MEMORY round trip: MSL defaults to fast
/// fp-contract, and a bare `y += product` would fuse the final multiply
/// into the add (one rounding, 1-ulp-class off the host's two-op form).
/// Storing the product forces its rounding; the second loop's `y += load`
/// is a pure add with nothing to contract.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn cq3_gemv_tiled(
    words: &[u32],
    scales: &[f32],
    c8: &[i32],
    x8: &[i32],
    s_x: &[f32],
    out: &mut [f32],
    params: &[f32],
) {
    let g = params[0usize] as u32;
    let b = params[1usize] as u32;
    let s_c = params[3usize];
    let tokens = params[4usize] as u32;
    // Per-token dims derive through tokens (the x.len()/g batch trap).
    let groups_in = ((x8.len() as u32) / tokens.max(1u32)) / g;
    let d_in = groups_in * g;
    // One thread per (row, token) in an 8-thread workgroup: the row is the
    // CUBE id — ABSOLUTE_POS_X spans every workgroup's threads (the same
    // misroute class the act-prep probe caught; with CubeDim 8 it mapped
    // row 1's threads to row 8 and left rows 1–7 stale).
    let row = CUBE_POS_X;
    let pos = ABSOLUTE_POS_Y;
    let d_out = (out.len() as u32) / tokens.max(1u32);
    if row >= d_out || pos >= tokens {
        terminate!();
    }
    let mask = (1u32 << b) - 1u32;
    let row_bit = (row * d_in) as usize * b as usize;
    let tok_base = pos * d_in;

    // Per-(row, token) contributions; the comptime 1024 covers g=16 at
    // d_in 16384 (from_packed refuses past it).
    let mut contrib = Shared::<[f32]>::new_slice(comptime!(1024usize));

    let mut k = 0u32;
    while k < groups_in {
        let grp = k;
        let mut acc = i32::new(0i64);
        let mut m = 0u32;
        while m < g {
            let pos_w = row_bit + ((grp * g + m) as usize) * b as usize;
            let w = pos_w / 32;
            let off = (pos_w % 32) as u32;
            let lo = words[w] >> off;
            let mut pv = lo;
            if off + b > 32u32 {
                pv = lo | (words[w + 1] << (32u32 - off));
            }
            let idx = (pv & mask) as usize;
            acc += c8[idx] * x8[(tok_base + grp * g + m) as usize];
            m += 1u32;
        }
        // The fixed left-associated product chain — rounded by the store.
        let gf = scales[(row * groups_in + grp) as usize];
        contrib[k as usize] =
            f32::cast_from(acc) * s_c * s_x[(pos * groups_in + grp) as usize] * gf;
        k += 1u32;
    }
    // NO barrier here: lanes past 0 terminated above, and a barrier reached
    // by a subset of the workgroup is UB (the staged kernel's law: every
    // unit must reach every sync). The single-writer-per-workgroup shape
    // makes the barrier semantically unnecessary anyway.
    // Pure adds — the sequential host fold. The ×1.0 spelling is
    // contraction-proof by algebra: fma(y, 1.0, c) = round(y·1.0 + c) =
    // round(y + c) — EXACTLY the host's two-op rounding, whichever way the
    // Metal compiler spells it (the bare `y += product` form fuses and
    // single-rounds, the measured 1–2 ulp drift).
    let mut y = f32::new(0.0f32);
    let mut k2 = 0u32;
    while k2 < groups_in {
        y = y * f32::new(1.0f32) + contrib[k2 as usize];
        k2 += 1u32;
    }
    out[(pos * d_out + row) as usize] = y;
}

/// Plane CQ3 GEMV (the perf tier) — one plane per (row, token), lane `j`
/// owns groups `j, j+PLANE_DIM, ...`; the same divide-free per-group math
/// as the tiled kernel; the cross-lane `plane_sum` is the hardware tree
/// (the tolerance tier — see the module doc).
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn cq3_gemv_plane(
    words: &[u32],
    scales: &[f32],
    c8: &[i32],
    x8: &[i32],
    s_x: &[f32],
    out: &mut [f32],
    params: &[f32],
) {
    let g = params[0usize] as u32;
    let b = params[1usize] as u32;
    let s_c = params[3usize];
    let tokens = params[4usize] as u32;
    // Per-token dims derive through tokens (the x.len()/g batch trap).
    let groups_in = ((x8.len() as u32) / tokens.max(1u32)) / g;
    let d_in = groups_in * g;
    let row = ABSOLUTE_POS_X / PLANE_DIM;
    let pos = ABSOLUTE_POS_Y;
    let d_out = (out.len() as u32) / tokens.max(1u32);
    if row >= d_out || pos >= tokens {
        terminate!();
    }
    let lane = UNIT_POS_PLANE;
    let mask = (1u32 << b) - 1u32;
    let row_bit = (row * d_in) as usize * b as usize;
    let tok_base = pos * d_in;

    let mut y = f32::new(0.0f32);
    let mut grp = lane;
    while grp < groups_in {
        let mut acc = i32::new(0i64);
        let mut m = 0u32;
        while m < g {
            let pos_w = row_bit + ((grp * g + m) as usize) * b as usize;
            let w = pos_w / 32;
            let off = (pos_w % 32) as u32;
            let lo = words[w] >> off;
            let mut pv = lo;
            if off + b > 32u32 {
                pv = lo | (words[w + 1] << (32u32 - off));
            }
            let idx = (pv & mask) as usize;
            acc += c8[idx] * x8[(tok_base + grp * g + m) as usize];
            m += 1u32;
        }
        let gf = scales[(row * groups_in + grp) as usize];
        y += f32::cast_from(acc) * s_c * s_x[(pos * groups_in + grp) as usize] * gf;
        grp += PLANE_DIM;
    }
    let result = plane_sum(y);
    if lane == 0u32 {
        out[(pos * d_out + row) as usize] = result;
    }
}

// ---------------------------------------------------------------------------
// Handle + launchers
// ---------------------------------------------------------------------------

/// The GPU-side CQ3 weight handle: the device buffers + dims + the
/// engagement counters. Clone like the other weight handles (Handles are
/// cheap `Arc` clones); the counters ride an [`AtomicUsize`] pair behind a
/// shared inner so clones count the same lane.
#[cfg(feature = "cubecl_runtime")]
#[derive(Clone)]
pub struct Cq3Handle {
    inner: std::sync::Arc<Cq3HandleInner>,
}

#[cfg(feature = "cubecl_runtime")]
struct Cq3HandleInner {
    words: Handle,
    scales: Handle,
    c8: Handle,
    /// The rotation derivation's P permutation (u32, g entries).
    perm: Handle,
    s1: Handle,
    s2: Handle,
    /// The SmoothQuant fold (f32, d_in); a zeroed dummy when unarmed (the
    /// kernel consults `params[1]`, never the length).
    smooth: Handle,
    /// Handle consts the per-launch params compose from (tokens rides the
    /// per-launch buffer — the x.len()/g batch trap).
    smooth_len_const: usize,
    inv_const: f32,
    wpg_const: usize,
    s_c_const: f32,
    pub d_out: usize,
    pub d_in: usize,
    pub g: usize,
    pub b: u32,
    /// Engagement counters (the Issue-1124 T1.5 class): act-prep and gemv
    /// launch counts, both atomic — the construction line names the lane,
    /// the smoke asserts these move.
    prep_launches: AtomicUsize,
    gemv_launches: AtomicUsize,
    /// Decode scratch: rotated x (f32, d_in) + staged x8 (i32, d_in) + s_x
    /// (f32, groups), grown once.
    decode_scratch: Mutex<Option<DecodeScratch>>,
    /// Lazily grown batched scratch: `(p, ...)` — one entry, the league's
    /// tile count is fixed per run.
    batch_scratch: Mutex<Option<(usize, BatchScratch)>>,
}

#[cfg(feature = "cubecl_runtime")]
struct DecodeScratch {
    x_rot: Handle,
    x8: Handle,
    s_x: Handle,
    params_prep: Handle,
    params_gemv: Handle,
}

#[cfg(feature = "cubecl_runtime")]
struct BatchScratch {
    x_rot: Handle,
    x8: Handle,
    s_x: Handle,
    params_prep: Handle,
    params_gemv: Handle,
}

#[cfg(feature = "cubecl_runtime")]
use std::sync::Mutex;
#[cfg(feature = "cubecl_runtime")]
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(feature = "cubecl_runtime")]
impl Cq3Handle {
    /// Build from the packed tensor: repack the u64 words to the u32 device
    /// form, pre-expand the f16 scales to f32 (the exact host reference —
    /// the Issue-593 subnormal class never reaches a kernel), cast the int8
    /// codebook to i32, and derive the rotation + codebook from (g, b) the
    /// same way the host reference does.
    pub fn from_packed<R: Runtime>(client: &ComputeClient<R>, t: &PackedCq3Tensor) -> Self {
        // Carrier → u32 words (device form).
        let carrier = t.unpack_indices();
        let words = pack_indices_b32(&carrier, t.b);
        // Scales pre-expanded (the exact reference conversion).
        let scales: Vec<f32> = t
            .scales
            .iter()
            .map(|&bits| half::f16::from_bits(bits).to_f32())
            .collect();
        let cb = t.codebook();
        let c8: Vec<i32> = cb.c8.iter().map(|&v| i32::from(v)).collect();
        let rot = t.rotation();
        let (rot_s1, rot_s2, rot_perm) = rot.device_params();
        let rot_perm: Vec<u32> = rot_perm.to_vec();
        let rot_s1: Vec<f32> = rot_s1.to_vec();
        let rot_s2: Vec<f32> = rot_s2.to_vec();
        let g = t.g;
        // The fold rides a DUMMY d_in buffer when unarmed (a size-0 buffer
        // arg corrupts the launch — measured: the no-smooth case returned
        // unrelated values); smooth_len=0 in params is the only gate the
        // kernel consults.
        let smooth = t.smooth.clone().unwrap_or_else(|| vec![0.0_f32; t.d_in]);
        let wpg = words_per_group(g, t.b);
        let groups_cap = (t.d_in / g).max(1);
        assert!(
            groups_cap <= 1024,
            "cq3 serving: groups per tensor {} exceeds the tiled kernel's shared slice (1024) — \
             the recipe pins g=32 (27B max 432); a wider model needs the cap raised",
            groups_cap
        );

        let words_h = client.create_from_slice(bytemuck::cast_slice(&words));
        let scales_h = client.create_from_slice(bytemuck::cast_slice(&scales));
        let c8_h = client.create_from_slice(bytemuck::cast_slice(&c8));
        let perm_h = client.create_from_slice(bytemuck::cast_slice(&rot_perm));
        let s1_h = client.create_from_slice(bytemuck::cast_slice(&rot_s1));
        let s2_h = client.create_from_slice(bytemuck::cast_slice(&rot_s2));
        let smooth_h = client.create_from_slice(bytemuck::cast_slice(&smooth));
        // The per-launch params compose from these consts + tokens at each
        // dispatch (the scratch carries the cached buffers).
        eprintln!(
            "[cq3-serving] armed: {}x{} g={} b={} b_eff={:.3} words={}KiB smooth={} wpg={} \
             (launch counters on; opt-in cq3, never production before the T3 GOAT)",
            t.d_out,
            t.d_in,
            g,
            t.b,
            t.bits_per_weight(),
            words.len() * 4 / 1024,
            if smooth.is_empty() { "off" } else { "on" },
            wpg,
        );

        Self {
            inner: std::sync::Arc::new(Cq3HandleInner {
                words: words_h,
                scales: scales_h,
                c8: c8_h,
                perm: perm_h,
                s1: s1_h,
                s2: s2_h,
                smooth: smooth_h,
                smooth_len_const: if t.smooth.is_some() { t.d_in } else { 0 },
                inv_const: 1.0_f32 / (g as f32).sqrt(),
                wpg_const: wpg,
                s_c_const: cb.s_c,
                d_out: t.d_out,
                d_in: t.d_in,
                g,
                b: t.b,
                prep_launches: AtomicUsize::new(0),
                gemv_launches: AtomicUsize::new(0),
                decode_scratch: Mutex::new(None),
                batch_scratch: Mutex::new(None),
            }),
        }
    }

    /// Dims for the dispatch site's shape assertions.
    pub fn dims(&self) -> (usize, usize) {
        (self.inner.d_out, self.inner.d_in)
    }

    /// Launch counters (the engagement-proof readout).
    pub fn launch_counts(&self) -> (usize, usize) {
        (
            self.inner.prep_launches.load(Ordering::Relaxed),
            self.inner.gemv_launches.load(Ordering::Relaxed),
        )
    }
}

#[cfg(feature = "cubecl_runtime")]
impl std::fmt::Debug for Cq3Handle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Cq3Handle")
            .field("d_out", &self.inner.d_out)
            .field("d_in", &self.inner.d_in)
            .field("g", &self.inner.g)
            .field("b", &self.inner.b)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Public launcher
// ---------------------------------------------------------------------------

/// The CQ3 GEMV launcher — decode (batch 1) and prefill (p tokens) in one
/// entry. Prefill is option (i), the never-unpack GEMV shape run p-times in
/// one batched launch (disclosed; the paper has no CQ prefill path).
#[cfg(feature = "cubecl_runtime")]
pub struct Cq3GemvCubeCL;

#[cfg(feature = "cubecl_runtime")]
impl Cq3GemvCubeCL {
    /// Dispatch: activation prep (rotated x in the handle's scratch) → the
    /// GEMV kernel (plane or tiled). `p == 1` is the decode shape; larger
    /// `p` rides the batched buffers.
    ///
    /// # Safety
    ///
    /// `input` holds `p * d_in` f32, `output` holds `p * d_out` f32.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        h: &Cq3Handle,
        input: Handle,
        output: Handle,
        p: usize,
    ) {
        let inner = &h.inner;
        let g = inner.g;
        let groups_in = inner.d_in / g;

        // Three passes: rotate (f32) → stage (int8 + per-group scale) →
        // GEMV. The GEMV is divide-free — the exactness bar lives there.
        let (x_rot, x8, s_x, params_prep, params_gemv) = Self::scratch(client, h, p);
        let prep_wg = CubeDim::new_1d(32u32);
        let groups_per_launch = (groups_in as u32).max(1);
        unsafe {
            cq3_act_prep::launch_unchecked::<R>(
                client,
                CubeCount::Static(groups_per_launch, p as u32, 1),
                prep_wg,
                BufferArg::from_raw_parts(input.clone(), p * inner.d_in),
                BufferArg::from_raw_parts(inner.smooth.clone(), inner.d_in),
                BufferArg::from_raw_parts(inner.perm.clone(), g),
                BufferArg::from_raw_parts(inner.s1.clone(), g),
                BufferArg::from_raw_parts(inner.s2.clone(), g),
                BufferArg::from_raw_parts(x_rot.clone(), p * inner.d_in),
                BufferArg::from_raw_parts(params_prep.clone(), 4),
            );
            cq3_stage_quant::launch_unchecked::<R>(
                client,
                CubeCount::Static(groups_per_launch, p as u32, 1),
                prep_wg,
                BufferArg::from_raw_parts(x_rot.clone(), p * inner.d_in),
                BufferArg::from_raw_parts(x8.clone(), p * inner.d_in),
                BufferArg::from_raw_parts(s_x.clone(), p * groups_in),
                BufferArg::from_raw_parts(params_prep.clone(), 4),
            );
        }
        inner.prep_launches.fetch_add(1, Ordering::Relaxed);

        // GEMV — plane by default (the league row), tiled behind the
        // exact-parity env (the fixture reads it). No runtime scalar args:
        // s_c + tokens ride params.
        let tiled = std::env::var("RIIR_QWEN38_CQ3_TILED").as_deref() == Ok("1") && p == 1;
        unsafe {
            if tiled {
                cq3_gemv_tiled::launch_unchecked::<R>(
                    client,
                    CubeCount::Static((inner.d_out as u32).max(1), p as u32, 1),
                    CubeDim::new_1d(8u32),
                    BufferArg::from_raw_parts(
                        inner.words.clone(),
                        words_host_len(inner.d_out * inner.d_in, inner.b),
                    ),
                    BufferArg::from_raw_parts(inner.scales.clone(), inner.d_out * (inner.d_in / g)),
                    BufferArg::from_raw_parts(inner.c8.clone(), 1usize << inner.b),
                    BufferArg::from_raw_parts(x8.clone(), p * inner.d_in),
                    BufferArg::from_raw_parts(s_x.clone(), p * groups_in),
                    BufferArg::from_raw_parts(output.clone(), p * inner.d_out),
                    BufferArg::from_raw_parts(params_gemv.clone(), 5),
                );
            } else {
                let plane_size = 32u32;
                let rows_per_wg = 256u32 / plane_size;
                let num_wg = ((inner.d_out as u32).div_ceil(rows_per_wg)).max(1);
                cq3_gemv_plane::launch_unchecked::<R>(
                    client,
                    CubeCount::Static(num_wg, p as u32, 1),
                    CubeDim::new_1d(256u32),
                    BufferArg::from_raw_parts(
                        inner.words.clone(),
                        words_host_len(inner.d_out * inner.d_in, inner.b),
                    ),
                    BufferArg::from_raw_parts(inner.scales.clone(), inner.d_out * (inner.d_in / g)),
                    BufferArg::from_raw_parts(inner.c8.clone(), 1usize << inner.b),
                    BufferArg::from_raw_parts(x8.clone(), p * inner.d_in),
                    BufferArg::from_raw_parts(s_x.clone(), p * groups_in),
                    BufferArg::from_raw_parts(output.clone(), p * inner.d_out),
                    BufferArg::from_raw_parts(params_gemv.clone(), 5),
                );
            }
        }
        inner.gemv_launches.fetch_add(1, Ordering::Relaxed);
    }

    /// The parity lane's entry: GEMV over PRE-STAGED inputs (x8/s_x computed
    /// by the HOST reference's exact loop) — divide-free, the exactness
    /// witness vs [`PackedCq3Tensor::gemv_i8_sdot`]. The tiled arm is
    /// DECODE-SHAPED (p = 1, the parity witness); the batched prefill rides
    /// the plane arm (the tolerance tier — Metal's fp contraction/reassoc
    /// class, the module doc records it).
    ///
    /// # Safety
    ///
    /// `x8` holds `p * d_in` i32, `s_x` holds `p * groups_in` f32, `output`
    /// holds `p * d_out` f32.
    pub unsafe fn launch_staged<R: Runtime>(
        client: &ComputeClient<R>,
        h: &Cq3Handle,
        x8: Handle,
        s_x: Handle,
        output: Handle,
        p: usize,
    ) {
        let inner = &h.inner;
        let g = inner.g;
        let groups_in = inner.d_in / g;
        // The tiled kernel is the parity witness at the decode shape; a
        // batched launch routes plane (the disclosed tier).
        let tiled = std::env::var("RIIR_QWEN38_CQ3_TILED").as_deref() == Ok("1") && p == 1;
        // Per-launch gemv params ([g, b, wpg, s_c, tokens]) — the Q4K-batched
        // pattern (a 20-byte device upload per call, trivial).
        let params = [
            g as f32,
            inner.b as f32,
            inner.wpg_const as f32,
            inner.s_c_const,
            p as f32,
        ];
        let params_h = client.create_from_slice(bytemuck::cast_slice(&params));
        unsafe {
            if tiled {
                cq3_gemv_tiled::launch_unchecked::<R>(
                    client,
                    CubeCount::Static((inner.d_out as u32).max(1), p as u32, 1),
                    CubeDim::new_1d(8u32),
                    BufferArg::from_raw_parts(
                        inner.words.clone(),
                        words_host_len(inner.d_out * inner.d_in, inner.b),
                    ),
                    BufferArg::from_raw_parts(inner.scales.clone(), inner.d_out * (inner.d_in / g)),
                    BufferArg::from_raw_parts(inner.c8.clone(), 1usize << inner.b),
                    BufferArg::from_raw_parts(x8, p * inner.d_in),
                    BufferArg::from_raw_parts(s_x, p * groups_in),
                    BufferArg::from_raw_parts(output, p * inner.d_out),
                    BufferArg::from_raw_parts(params_h.clone(), 5),
                );
            } else {
                let rows_per_wg = 8u32;
                let num_wg = ((inner.d_out as u32).div_ceil(rows_per_wg)).max(1);
                cq3_gemv_plane::launch_unchecked::<R>(
                    client,
                    CubeCount::Static(num_wg, p as u32, 1),
                    CubeDim::new_1d(256u32),
                    BufferArg::from_raw_parts(
                        inner.words.clone(),
                        words_host_len(inner.d_out * inner.d_in, inner.b),
                    ),
                    BufferArg::from_raw_parts(inner.scales.clone(), inner.d_out * (inner.d_in / g)),
                    BufferArg::from_raw_parts(inner.c8.clone(), 1usize << inner.b),
                    BufferArg::from_raw_parts(x8, p * inner.d_in),
                    BufferArg::from_raw_parts(s_x, p * groups_in),
                    BufferArg::from_raw_parts(output, p * inner.d_out),
                    BufferArg::from_raw_parts(params_h, 5),
                );
            }
        }
        inner.gemv_launches.fetch_add(1, Ordering::Relaxed);
    }

    /// The three-pass scratch + the per-launch params buffers: decode
    /// reuses handle-carried buffers; the batched shape grows cached p-sized
    /// buffers (one entry — the league's tile count is fixed per run).
    /// params_prep = [g, smooth_len, inv, tokens]; params_gemv = [g, b, wpg,
    /// s_c, tokens] — tokens rides per-launch (the x.len()/g batch trap).
    fn scratch<R: Runtime>(
        client: &ComputeClient<R>,
        h: &Cq3Handle,
        p: usize,
    ) -> (Handle, Handle, Handle, Handle, Handle) {
        let d_in = h.inner.d_in;
        let g = h.inner.g;
        let groups_in = d_in / g;
        let f32_b = core::mem::size_of::<f32>();
        let i32_b = core::mem::size_of::<i32>();
        if p == 1 {
            let mut slot = h.inner.decode_scratch.lock().expect("cq3 scratch lock");
            if slot.is_none() {
                let prep = [
                    g as f32,
                    h.inner.smooth_len_const as f32,
                    h.inner.inv_const,
                    1.0_f32,
                ];
                let gemv = [
                    g as f32,
                    h.inner.b as f32,
                    h.inner.wpg_const as f32,
                    h.inner.s_c_const,
                    1.0_f32,
                ];
                *slot = Some(DecodeScratch {
                    x_rot: client.empty(d_in * f32_b),
                    x8: client.empty(d_in * i32_b),
                    s_x: client.empty(groups_in * f32_b),
                    params_prep: client.create_from_slice(bytemuck::cast_slice(&prep)),
                    params_gemv: client.create_from_slice(bytemuck::cast_slice(&gemv)),
                });
            }
            let s = slot.as_ref().expect("just set");
            (
                s.x_rot.clone(),
                s.x8.clone(),
                s.s_x.clone(),
                s.params_prep.clone(),
                s.params_gemv.clone(),
            )
        } else {
            let mut slot = h.inner.batch_scratch.lock().expect("cq3 batch lock");
            let ok = matches!(&*slot, Some((cached, _)) if *cached == p);
            if !ok {
                let prep = [
                    g as f32,
                    h.inner.smooth_len_const as f32,
                    h.inner.inv_const,
                    p as f32,
                ];
                let gemv = [
                    g as f32,
                    h.inner.b as f32,
                    h.inner.wpg_const as f32,
                    h.inner.s_c_const,
                    p as f32,
                ];
                *slot = Some((
                    p,
                    BatchScratch {
                        x_rot: client.empty(p * d_in * f32_b),
                        x8: client.empty(p * d_in * i32_b),
                        s_x: client.empty(p * groups_in * f32_b),
                        params_prep: client.create_from_slice(bytemuck::cast_slice(&prep)),
                        params_gemv: client.create_from_slice(bytemuck::cast_slice(&gemv)),
                    },
                ));
            }
            let s = match &*slot {
                Some((_, s)) => s,
                None => unreachable!("just set"),
            };
            (
                s.x_rot.clone(),
                s.x8.clone(),
                s.s_x.clone(),
                s.params_prep.clone(),
                s.params_gemv.clone(),
            )
        }
    }
}

/// Device word count for `n` weights at `b` bits.
#[must_use]
pub fn words_host_len(n: usize, b: u32) -> usize {
    (n as u64 * u64::from(b)).div_ceil(32) as usize
}

#[cfg(all(test, feature = "cubecl_runtime", target_os = "macos"))]
mod unit_probe {
    use super::*;
    use crate::cubecl_runtime::{ActiveComputeClient, ActiveRuntime, CubeCLContext};
    use crate::qwen38_cq3::CqTensor;

    fn lcg(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let bits = (s >> 40) as f32 / (1u64 << 24) as f32;
                lo + (hi - lo) * bits
            })
            .collect()
    }

    #[cube(launch_unchecked)]
    fn probe_div_kernel(x: &[f32], s: &[f32], out: &mut [f32]) {
        let i = ABSOLUTE_POS_X as usize;
        let n = x.len();
        if i < n {
            out[i] = x[i] / s[i];
        }
    }

    #[cube(launch_unchecked)]
    fn probe_mul_kernel(x: &[f32], s: &[f32], out: &mut [f32]) {
        let i = ABSOLUTE_POS_X as usize;
        let n = x.len();
        if i < n {
            out[i] = x[i] * s[i];
        }
    }

    #[test]
    fn probe_divide_precision() {
        let Ok(ctx) = CubeCLContext::new() else {
            eprintln!("[skip] no GPU");
            return;
        };
        let client: ActiveComputeClient = ctx.client();
        let x = lcg(0xDACE, 256, -1.0, 1.0);
        let s = lcg(0x5EED, 256, 0.25, 4.0);
        let want: Vec<f32> = x.iter().zip(s.iter()).map(|(a, b)| a / b).collect();
        let xh = client.create_from_slice(bytemuck::cast_slice(&x));
        let sh = client.create_from_slice(bytemuck::cast_slice(&s));
        let oh = client.empty(256 * core::mem::size_of::<f32>());
        unsafe {
            probe_div_kernel::launch_unchecked::<ActiveRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(256u32),
                BufferArg::from_raw_parts(xh, 256),
                BufferArg::from_raw_parts(sh, 256),
                BufferArg::from_raw_parts(oh.clone(), 256),
            );
        }
        let bytes = client.read_one(oh).expect("read");
        let got: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut mm = 0;
        let mut max_ulp = 0_u64;
        for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
            let d = a.to_bits().abs_diff(b.to_bits());
            if d != 0 {
                if mm < 3 {
                    eprintln!("[divprobe] [{i}] gpu {a} vs host {b} ulp {d}");
                }
                mm += 1;
                max_ulp = max_ulp.max(d as u64);
            }
        }
        eprintln!("[divprobe] divide mismatches {mm}/256 max ulp {max_ulp}");
    }

    #[test]
    fn probe_act_prep_matches_rotate_activation() {
        let Ok(ctx) = CubeCLContext::new() else {
            eprintln!("[skip] no GPU");
            return;
        };
        let client: ActiveComputeClient = ctx.client();
        let d_in = 256usize;
        let g = 32usize;
        let smooth = lcg(0x5EED, d_in, 0.25, 4.0);
        let w = lcg(0xC0FFEE, 16 * d_in, -0.5, 0.5);
        let t = CqTensor::quantize(&w, 16, d_in, g, 3, Some(&smooth));
        let packed = crate::qwen38_cq3_serving::PackedCq3Tensor::from_cq_tensor(&t);

        let x = lcg(0xDACE, d_in, -1.0, 1.0);
        let want = packed.rotate_activation(&x);

        // Direct kernel launch (unit tests reach the private fns).
        let rot = packed.rotation();
        let (rs1, rs2, rperm) = rot.device_params();
        let perm_h = client.create_from_slice(bytemuck::cast_slice(rperm));
        let s1_h = client.create_from_slice(bytemuck::cast_slice(rs1));
        let s2_h = client.create_from_slice(bytemuck::cast_slice(rs2));
        let smooth_h = client.create_from_slice(bytemuck::cast_slice(&smooth));
        let params = [
            g as f32,
            smooth.len() as f32,
            1.0_f32 / (g as f32).sqrt(),
            1.0_f32,
        ];
        let params_h = client.create_from_slice(bytemuck::cast_slice(&params));
        let x_h = client.create_from_slice(bytemuck::cast_slice(&x));
        let out_h = client.empty(d_in * core::mem::size_of::<f32>());
        let groups_in = d_in / g;
        unsafe {
            cq3_act_prep::launch_unchecked::<ActiveRuntime>(
                &client,
                CubeCount::Static(groups_in as u32, 1, 1),
                CubeDim::new_1d(32u32),
                BufferArg::from_raw_parts(x_h, d_in),
                BufferArg::from_raw_parts(smooth_h, d_in),
                BufferArg::from_raw_parts(perm_h, g),
                BufferArg::from_raw_parts(s1_h, g),
                BufferArg::from_raw_parts(s2_h, g),
                BufferArg::from_raw_parts(out_h.clone(), d_in),
                BufferArg::from_raw_parts(params_h, 4),
            );
        }
        let bytes = client.read_one(out_h).expect("read");
        let got: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut mismatches = 0;
        for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
            if a.to_bits() != b.to_bits() {
                if mismatches < 4 {
                    eprintln!("[probe] [{i}] gpu {a} vs host {b}");
                }
                mismatches += 1;
            }
        }
        eprintln!("[probe] {mismatches}/{d_in} bit-mismatches");
        eprintln!(
            "[probe] got[0..3]={:?} want[0..3]={:?}",
            &got[..3],
            &want[..3]
        );
        // Quantify the drift class: exact ulp distance on the mismatches.
        let mut max_ulp = 0_u64;
        for (a, b) in got.iter().zip(want.iter()) {
            let d = a.to_bits().abs_diff(b.to_bits()) as u64;
            max_ulp = max_ulp.max(d);
        }
        eprintln!("[probe] smooth-armed mismatches: {mismatches}/{d_in}, max ulp delta {max_ulp}");
        // Bisect arm: no-smooth run (empty fold) — isolates the divide as
        // the 1-ulp source if the drift vanishes.
        let t2 = CqTensor::quantize(&w, 16, d_in, g, 3, None);
        let packed2 = crate::qwen38_cq3_serving::PackedCq3Tensor::from_cq_tensor(&t2);
        let want2 = packed2.rotate_activation(&x);
        let rot2 = packed2.rotation();
        let (r1, r2_, rp) = rot2.device_params();
        let p_h = client.create_from_slice(bytemuck::cast_slice(rp));
        let a_h = client.create_from_slice(bytemuck::cast_slice(r1));
        let b_h = client.create_from_slice(bytemuck::cast_slice(r2_));
        let empty: Vec<f32> = vec![0.0_f32; d_in];
        let empty_h = client.create_from_slice(bytemuck::cast_slice(&empty));
        let params2 = [g as f32, 0.0_f32, 1.0_f32 / (g as f32).sqrt(), 1.0_f32];
        let p2_h = client.create_from_slice(bytemuck::cast_slice(&params2));
        let x_h2 = client.create_from_slice(bytemuck::cast_slice(&x));
        let out_h2 = client.empty(d_in * core::mem::size_of::<f32>());
        unsafe {
            cq3_act_prep::launch_unchecked::<ActiveRuntime>(
                &client,
                CubeCount::Static(groups_in as u32, 1, 1),
                CubeDim::new_1d(32u32),
                BufferArg::from_raw_parts(x_h2, d_in),
                BufferArg::from_raw_parts(empty_h, 0),
                BufferArg::from_raw_parts(p_h, g),
                BufferArg::from_raw_parts(a_h, g),
                BufferArg::from_raw_parts(b_h, g),
                BufferArg::from_raw_parts(out_h2.clone(), d_in),
                BufferArg::from_raw_parts(p2_h, 4),
            );
        }
        let bytes2 = client.read_one(out_h2).expect("read2");
        let got2: Vec<f32> = bytes2
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut mm2 = 0;
        for (i, (a, b)) in got2.iter().zip(want2.iter()).enumerate() {
            if a.to_bits() != b.to_bits() {
                if mm2 < 3 {
                    eprintln!("[probe-nosmooth] [{i}] gpu {a} vs host {b}");
                }
                mm2 += 1;
            }
        }
        eprintln!("[probe-nosmooth] mismatches: {mm2}/{d_in}");
        assert_eq!(
            mm2, 0,
            "act_prep bit-exact vs rotate_activation (no smooth)"
        );
    }
}
