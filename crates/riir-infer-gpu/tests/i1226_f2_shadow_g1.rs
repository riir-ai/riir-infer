//! i1226 F2 — G1, the producer-writes-f16 shadow's BIT-IDENTITY cell.
//!
//! The gate the F2 lane is armed on: the rotate kernels' packed-u32 f16
//! shadow (`f16_bits` RNE emulation, written as a side output of the
//! rotation) must be bit-identical — elementwise — to what the twin's cast
//! encoder (`cast_f32_f16_zc`, `dst[i] = half(src[i])`) produces from the
//! SAME rotated f32 output. The authority is the REAL cast pipeline (via the
//! instrument-only `cast_f16_reference`), never a host-side oracle: this is
//! the encoder the shadow removes from the frame.
//!
//! All FOUR rotate call shapes (i1226 scoping notes, verdict round 2):
//! in-place on `normx_b` at width n; out-of-place copy to `rotx_b`; in-place
//! on `attn_out_b` at width q_dim; in-place on `rec_b`/`hid_b` at width
//! v_dim/mlp — the rotate picks its `signs_for_width(width)` table by width,
//! so the cell must not assume one width. Both schedules: hblock 1024 (the
//! radix-4 serving path, default-on) and hblock 256 (the plain schedule).
//!
//! Bit-identity here carries Bench 1224's per-family retention over by
//! construction: the twin consumes the SAME rounded values it always cast.
#![cfg(all(feature = "prefill_f16b", target_os = "macos"))]

use cubecl::prelude::*;
use riir_infer_core::deltanet::rotation::TernaryRotationConfig;
use riir_infer_gpu::cubecl_runtime::{ActiveRuntime, CubeCLContext};
use riir_infer_gpu::deltanet_rotation_cubecl::RotationTablesCubeCL;
use riir_infer_gpu::{MetalTensorZeroCopyGemm, RotationCubeCL};

/// Deterministic f32 stream in [-3, 3] — finite by construction (the rotate
/// outputs are finite; the G1 cell pins finite inputs, the NaN payload class
/// is documented unreachable on the armed path).
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        // SplitMix64-style finalizer on an incrementing state — enough
        // dispersion for the rounding classes (near-tie mantissas included).
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let z = self.0;
        let h = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        let h = (h ^ (h >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let h = h ^ (h >> 31);
        ((h >> 40) as f32 / (1u32 << 24) as f32 - 0.5) * 6.0
    }
}

fn input_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut lcg = Lcg(seed);
    (0..n).map(|_| lcg.next_f32()).collect()
}

fn signs_vec(n: usize, seed: u64) -> Vec<i8> {
    let mut lcg = Lcg(seed);
    (0..n)
        .map(|_| if lcg.next_f32() >= 0.0 { 1 } else { -1 })
        .collect()
}

/// One G1 case: shadow-rotate `input` [p × width] → read the shadow pairs;
/// plain-rotate the SAME input → cast the rotated f32 through THE cast
/// pipeline → compare bit-for-bit.
fn shadow_bit_identity_case(label: &str, width: usize, p: usize, hblock: usize, seed: u64) {
    let ctx = CubeCLContext::new().expect("GPU init");
    let client = ctx.client();
    let gemm = MetalTensorZeroCopyGemm::new_f16b(
        ctx.wgpu_device().expect("wgpu device").clone(),
        ctx.wgpu_queue().expect("wgpu queue").clone(),
    )
    .expect("f16b twin (armed posture — the placebo flag defaults off)");

    let signs = signs_vec(width, seed);
    let cfg = TernaryRotationConfig {
        block_size: hblock,
        signs: vec![(width, signs)],
        gdn_v_grouped: true,
        gdn_v_heads: 4,
        gdn_k_groups: 2,
        inverse_embedding: true,
    };
    let tables = RotationTablesCubeCL::build::<ActiveRuntime>(&client, &cfg)
        .expect("tables build within bound");

    let total = p * width;
    let input = input_vec(total, seed.wrapping_add(1));
    let input_handle = client.create_from_slice(f32::as_bytes(&input));

    // The shadow run: a COPY of the input, rotated by the shadow kernel.
    let shadow_input = client.create_from_slice(f32::as_bytes(&input));
    let shadow = client.empty(total * std::mem::size_of::<u16>());
    let signs_h = tables.signs_for_width(width).clone();
    unsafe {
        RotationCubeCL::launch_forward_shadow::<ActiveRuntime>(
            &client,
            shadow_input.clone(),
            shadow.clone(),
            signs_h.clone(),
            total,
            width,
            hblock,
        );
    }

    // The reference run: the plain rotate (bit-identical to the shadow
    // rotate's f32 half by the existing rotate tests) on a fresh copy, then
    // THE cast pipeline over its output.
    let ref_input = client.create_from_slice(f32::as_bytes(&input));
    unsafe {
        RotationCubeCL::launch_forward::<ActiveRuntime>(
            &client,
            ref_input.clone(),
            signs_h.clone(),
            total,
            width,
            hblock,
        );
    }
    client.flush();
    let ref_f16 = gemm
        .cast_f16_reference(&client, &ref_input, total)
        .expect("cast reference");

    // The shadow, read back as u16s (packed pairs → lo | hi<<16).
    client.flush();
    let shadow_bytes = client.read_one(shadow).expect("read shadow");
    let shadow_u32: Vec<u32> = shadow_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(shadow_u32.len(), total / 2, "shadow pair count");
    let mut mismatches = 0usize;
    let mut first = None;
    for (k, &pair) in shadow_u32.iter().enumerate() {
        let lo = (pair & 0xFFFF) as u16;
        let hi = (pair >> 16) as u16;
        let want_lo = ref_f16[2 * k];
        let want_hi = ref_f16[2 * k + 1];
        if lo != want_lo || hi != want_hi {
            mismatches += 1;
            if first.is_none() {
                first = Some((
                    2 * k,
                    lo,
                    want_lo,
                    hi,
                    want_hi,
                    input[2 * k],
                    input[2 * k + 1],
                ));
            }
        }
    }
    assert_eq!(
        mismatches, 0,
        "{label}: shadow vs cast bit-identity broke at {mismatches} pair(s) \
         (first: {:?})",
        first
    );
}

/// The serving schedule (hblock 1024 → radix-4) at the model widths: the
/// in-place `normx_b` shape (AttnQ/Kv, FfnGate/Up), the out-of-place
/// `rotx_b` shape (GdnQkv/GdnZ), the `attn_out_b` shape (AttnOutProj, q_dim
/// class), and the `rec_b`/`hid_b` shape (GdnOutProj/FfnDown, v_dim/mlp
/// class). Widths are the Bonsai-2 folded family set.
#[test]
#[ignore = "requires the Metal GPU; run explicitly with --release"]
fn f2_shadow_bit_identity_all_shapes_hblock1024() {
    // In-place normx (width n_embd) + out-of-place rotx (same width, the
    // copy kernel path).
    shadow_bit_identity_case("normx/rotx n=5120 (r4)", 5120, 8, 1024, 0xF2001);
    // attn_out (q_dim class) — a different width table + odd-ish geometry.
    shadow_bit_identity_case("attn_out q_dim=2048 (r4)", 2048, 8, 1024, 0xF2002);
    // rec/hid (v_dim/mlp class).
    shadow_bit_identity_case("rec/hid mlp=17408 (r4)", 17408, 4, 1024, 0xF2003);
    // The 6144 table (the second n_embd-class width in the fixture set).
    shadow_bit_identity_case("normx n=6144 (r4)", 6144, 8, 1024, 0xF2004);
}

/// The plain schedule (hblock 256) — the non-1024 rotate path, where the
/// shadow epilogue reads `smem` directly after the butterfly's final barrier.
#[test]
#[ignore = "requires the Metal GPU; run explicitly with --release"]
fn f2_shadow_bit_identity_all_shapes_hblock256() {
    shadow_bit_identity_case("normx/rotx n=5120 (plain)", 5120, 8, 256, 0xF2011);
    shadow_bit_identity_case("attn_out q_dim=2048 (plain)", 2048, 8, 256, 0xF2012);
    shadow_bit_identity_case("rec/hid mlp=17408 (plain)", 17408, 4, 256, 0xF2013);
}

/// The tie-exercise class: mantissas engineered to land ON the RNE tie
/// (13th-bit patterns) and at the f16 rounding boundary — the rounding-shape
/// coverage the random streams only sample. Width kept small (one block).
#[test]
#[ignore = "requires the Metal GPU; run explicitly with --release"]
fn f2_shadow_bit_identity_tie_classes() {
    // Tie inputs: every element's f32 mantissa carries the 0x1000/0x1FFF
    // pattern the RNE rule branches on (rem == half, odd/even h10).
    let tie_pattern = 0x3F80_0000u32 | (0x1000u32 << 12); // rem == half, h10 even → stays
    let tie_up = 0x3F80_0000u32 | ((0x1000u32 << 12) | 0x1000); // rem == half, h10 odd → rounds up
    let carry = 0x3F80_0000u32 | 0x7F_FFFFu32; // rounds up into a new exp
    let tiny = 0x0000_0001u32; // f32 subnormal → ±0
    let inputs: Vec<f32> = [tie_pattern, tie_up, carry, tiny, 0x3F80_0000, 0xBF80_0000]
        .iter()
        .map(|&b| f32::from_bits(b))
        .chain(std::iter::repeat(0.5f32).take(502))
        .collect();
    let width = 512usize;
    let p = 1usize;
    let total = p * width;
    assert_eq!(inputs.len(), total);

    let ctx = CubeCLContext::new().expect("GPU init");
    let client = ctx.client();
    let gemm = MetalTensorZeroCopyGemm::new_f16b(
        ctx.wgpu_device().expect("wgpu device").clone(),
        ctx.wgpu_queue().expect("wgpu queue").clone(),
    )
    .expect("f16b twin");
    let signs = signs_vec(width, 0xF3001);
    let cfg = TernaryRotationConfig {
        block_size: 256,
        signs: vec![(width, signs)],
        gdn_v_grouped: true,
        gdn_v_heads: 4,
        gdn_k_groups: 2,
        inverse_embedding: true,
    };
    let tables = RotationTablesCubeCL::build::<ActiveRuntime>(&client, &cfg).expect("tables");

    let shadow_input = client.create_from_slice(f32::as_bytes(&inputs));
    let shadow = client.empty(total * std::mem::size_of::<u16>());
    let signs_h = tables.signs_for_width(width).clone();
    let ref_input = client.create_from_slice(f32::as_bytes(&inputs));
    unsafe {
        RotationCubeCL::launch_forward_shadow::<ActiveRuntime>(
            &client,
            shadow_input,
            shadow.clone(),
            signs_h.clone(),
            total,
            width,
            256,
        );
        RotationCubeCL::launch_forward::<ActiveRuntime>(
            &client,
            ref_input.clone(),
            signs_h,
            total,
            width,
            256,
        );
    }
    client.flush();
    let ref_f16 = gemm
        .cast_f16_reference(&client, &ref_input, total)
        .expect("cast reference");
    let shadow_bytes = client.read_one(shadow).expect("read shadow");
    let shadow_u32: Vec<u32> = shadow_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(shadow_u32.len(), total / 2, "shadow pair count");
    for (k, &pair) in shadow_u32.iter().enumerate() {
        let lo = (pair & 0xFFFF) as u16;
        let hi = (pair >> 16) as u16;
        assert_eq!(
            lo,
            ref_f16[2 * k],
            "tie-class lo mismatch at element {}",
            2 * k
        );
        assert_eq!(
            hi,
            ref_f16[2 * k + 1],
            "tie-class hi mismatch at element {}",
            2 * k + 1
        );
    }
}
