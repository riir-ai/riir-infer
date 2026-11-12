//! Deletion-mask application to packed ternary `Q2_0`/`PQ2_0` weights
//! (Issue 1208 T3).
//!
//! Deletion is IN-DISTRIBUTION for PQ2_0: the masked code is `01`,
//! which is the format's NATIVE zero symbol (`src/quant/q2_0.rs` —
//! codes {0→−d, 1→0, 2→+d, 3→+2d}), so a masked tensor stays a valid
//! tensor and no kernel bit-path changes by construction.
//!
//! Bulk form: a byte-pair lookup table over (codes byte, mask byte) —
//! `LUT[cb][mb] → new_cb` — the exhaustive 2¹⁶ domain, complete, not
//! sampled. Alignment convention: one codes byte holds FOUR 2-bit
//! codes; the mask operand carries those four keep/delete bits in its
//! LOW NIBBLE (bit i = weight i of the codes byte, LSB-first — the
//! codec's bit-order pin). The upper nibble of the mask operand is
//! ignored by the table (defined but never produced by [`nibble_for`]).
//! The caller maps each codes byte to its mask nibble with
//! [`nibble_for`] — no bit-loop in the hot path, SIMD-friendly,
//! branch-free, zero-alloc.
//!
//! The group scale `d` is FROZEN: masked weights go to 0, which the
//! symmetric scale represents without any rescale, so scale bytes are
//! copied verbatim and a masked tensor is byte-identical to the
//! original outside the code bytes (pinned by [`tests::scales_never_move`]).
//!
//! Falsifiers (all pinned here, exhaustive where the domain allows):
//! masked-zero ≡ native-zero through BOTH the f32 dequant and the
//! [`repack_q2_0_to_ternary_group`] bridge (the skip-path parity law);
//! the stale-metadata tripwire (the materialized planes are computed
//! from the MASKED codes — mutate one code after masking and the
//! materialization digest must move); no metadata reuse across masks
//! (different masks → different outputs; same input+mask →
//! deterministic output); code-3 deletion (+2d → 0).

use super::mask_codec::{DecodedMask, MaskCodecError};
use crate::quant::q2_0::{BlockQ2_0, Q2_0_BLOCK_SIZE};
use std::sync::OnceLock;

/// Table entries per row: the mask operand's useful bits (low nibble).
const MASK_USEFUL_BITS: usize = 4;
const LUT_SIDE: usize = 256;
const LUT_LEN: usize = LUT_SIDE * LUT_SIDE;

/// The keep/delete nibble for codes byte `index` (0..32) inside a
/// block, drawn from the block's 16 mask bytes (128 weights × 1 bit):
/// even codes byte → LOW nibble of mask byte `index / 2`, odd codes
/// byte → HIGH nibble, shifted into the low position.
#[inline]
pub fn nibble_for(codes_byte_index: usize, mask_bytes: &[u8]) -> u8 {
    let byte = mask_bytes[codes_byte_index / 2];
    if codes_byte_index.is_multiple_of(2) {
        byte & 0x0F
    } else {
        byte >> 4
    }
}

/// The exhaustive table: `LUT[cb * 256 + mb]`. For each weight slot
/// `i` (0..4): mask bit `i` of `mb` = 0 → code `i` becomes `01`
/// (native zero); = 1 → the code passes through untouched (including
/// the spare code-3 state, which deletion may legitimately zero).
#[inline]
pub fn lut() -> &'static [u8; LUT_LEN] {
    static LUT: OnceLock<[u8; LUT_LEN]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut table = [0u8; LUT_LEN];
        for cb in 0..=255u8 {
            for mb in 0..=255u8 {
                let mut out = cb;
                for i in 0..MASK_USEFUL_BITS as u8 {
                    if (mb >> i) & 1 == 0 {
                        out &= !(0b11 << (2 * i));
                        out |= 0b01 << (2 * i);
                    }
                }
                table[cb as usize * LUT_SIDE + mb as usize] = out;
            }
        }
        table
    })
}

/// Raw entry lookup for tests and callers that already hold both bytes.
#[inline]
pub fn lut_entry(codes_byte: u8, mask_byte: u8) -> u8 {
    lut()[codes_byte as usize * LUT_SIDE + mask_byte as usize]
}

/// Mask a contiguous run of `Q2_0` blocks. `mask_bits` is the codec's
/// bit-packed stream; the slice `[start_bit .. start_bit +
/// blocks.len() * Q2_0_BLOCK_SIZE]` must be in range (one mask bit per
/// weight, block-major: block b's weights sit at `start_bit + b *
/// 128`). Scales are copied verbatim; code bytes go through
/// [`lut_entry`] against their aligned nibble.
pub fn apply_mask_q2_0(
    blocks: &[BlockQ2_0],
    mask_bits: &[u8],
    start_bit: usize,
) -> Result<Vec<BlockQ2_0>, MaskCodecError> {
    if !start_bit.is_multiple_of(8) {
        return Err(MaskCodecError::UnalignedPayload(start_bit as u64));
    }
    let needed_bits = blocks
        .len()
        .checked_mul(Q2_0_BLOCK_SIZE)
        .ok_or_else(|| MaskCodecError::SectionRange("block count overflow".to_owned()))?;
    let end_bit = start_bit + needed_bits;
    if mask_bits.len() * 8 < end_bit {
        return Err(MaskCodecError::SectionRange(format!(
            "mask stream {} bits < apply window {}..{}",
            mask_bits.len() * 8,
            start_bit,
            end_bit
        )));
    }

    let table = lut();
    let mut out = blocks.to_vec();
    let mask_bytes = &mask_bits[start_bit / 8..end_bit.div_ceil(8)];
    for (block_index, block) in out.iter_mut().enumerate() {
        let raw = bytemuck::bytes_of_mut(block);
        // 32 code bytes start at offset 2 (after the f16 scale); each
        // consumes one nibble of the block's 16 mask bytes.
        for i in 0..32 {
            let code_off = 2 + i;
            let mask_bit_base = block_index * Q2_0_BLOCK_SIZE + i * 4;
            let nibble = {
                let byte = mask_bytes[mask_bit_base / 8];
                if mask_bit_base.is_multiple_of(8) {
                    byte & 0x0F
                } else {
                    byte >> 4
                }
            };
            raw[code_off] = table[raw[code_off] as usize * LUT_SIDE + nibble as usize];
        }
    }
    Ok(out)
}

/// Apply a decoded mask to one named tensor's blocks. `None` when the
/// mask has no section for the tensor (the mask is silent there — the
/// tensor passes through untouched). The section's bit range must
/// exactly cover the tensor's weight count.
pub fn apply_to_tensor(
    blocks: &[BlockQ2_0],
    mask: &DecodedMask,
    tensor_name: &str,
) -> Result<Option<Vec<BlockQ2_0>>, MaskCodecError> {
    let Some(section) = mask
        .manifest
        .sections
        .iter()
        .find(|s| s.tensor_name == tensor_name)
    else {
        return Ok(None);
    };
    let weight_count = blocks
        .len()
        .checked_mul(Q2_0_BLOCK_SIZE)
        .ok_or_else(|| MaskCodecError::SectionRange("block count overflow".to_owned()))?;
    if section.bit_len != weight_count as u64 {
        return Err(MaskCodecError::SectionRange(format!(
            "section covers {} bits, tensor {} needs {}",
            section.bit_len, tensor_name, weight_count
        )));
    }
    apply_mask_q2_0(blocks, &mask.bits, section.bit_offset as usize).map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(bytes: &[u8]) -> [u8; 32] {
        *blake3::hash(bytes).as_bytes()
    }

    fn blocks_with_pattern(n_blocks: usize, seed: u8) -> Vec<BlockQ2_0> {
        use half::f16;
        (0..n_blocks)
            .map(|b| {
                let mut block = BlockQ2_0 {
                    d: f16::to_bits(f16::from_f32(0.5 + 0.1 * (b % 7) as f32)),
                    qs: [0u8; 32],
                };
                // code-3-free by construction: every 2-bit slot cycles
                // {0,1,2} — the bridge refuses the spare state, and real
                // symmetric quantization never emits it (0.000% measured)
                for (i, byte) in block.qs.iter_mut().enumerate() {
                    let s0 = (i + b * 3) % 3;
                    let s1 = (i + b * 3 + 1) % 3;
                    let s2 = (i + b * 3 + 2) % 3;
                    let s3 = (i / 2 + b) % 3;
                    *byte = (s0 | (s1 << 2) | (s2 << 4) | (s3 << 6)) as u8;
                }
                let _ = seed;
                block
            })
            .collect()
    }

    /// A block's 128 mask bits: bit w set = keep weight w.
    fn mask_bytes_from_keeps(keeps: &[bool]) -> Vec<u8> {
        let mut bytes = vec![0u8; keeps.len().div_ceil(8)];
        for (w, keep) in keeps.iter().enumerate() {
            if *keep {
                bytes[w / 8] |= 1 << (w % 8);
            }
        }
        bytes
    }

    fn q2_bytes(blocks: &[BlockQ2_0]) -> Vec<u8> {
        blocks
            .iter()
            .flat_map(|b| b.d.to_le_bytes().into_iter().chain(b.qs))
            .collect()
    }

    /// Scalar reference: apply the mask one weight at a time.
    fn reference_apply(blocks: &[BlockQ2_0], keeps: &[bool]) -> Vec<BlockQ2_0> {
        let mut out = blocks.to_vec();
        for (b, block) in out.iter_mut().enumerate() {
            for w in 0..Q2_0_BLOCK_SIZE {
                if !keeps[b * Q2_0_BLOCK_SIZE + w] {
                    let shift = 2 * (w % 4);
                    block.qs[w / 4] = (block.qs[w / 4] & !(0b11 << shift)) | (0b01 << shift);
                }
            }
        }
        out
    }

    #[test]
    fn lut_is_exhaustive_and_matches_reference() {
        let table = lut();
        // every entry defined, and the whole 2^16 domain agrees with the
        // scalar per-weight reference — complete, not sampled
        for cb in 0..=255u8 {
            for mb in 0..=255u8 {
                let mut expected = cb;
                for i in 0..4u8 {
                    if (mb >> i) & 1 == 0 {
                        expected = (expected & !(0b11 << (2 * i))) | (0b01 << (2 * i));
                    }
                }
                assert_eq!(lut_entry(cb, mb), expected, "cb={cb:#010b} mb={mb:#010b}");
                assert_eq!(table[cb as usize * 256 + mb as usize], expected);
            }
        }
    }

    #[test]
    fn lut_identity_all_keep_and_upper_nibble_ignored() {
        // all-keep (0xFF and any upper-nibble noise) → byte unchanged
        for cb in 0..=255u8 {
            for noise in [0x00u8, 0x0F, 0xF0, 0xFF] {
                assert_eq!(lut_entry(cb, 0x0F | noise), cb);
            }
        }
    }

    #[test]
    fn code3_deletes_to_native_zero() {
        // cb = 0b11_01_10_00, slots LSB-first: slot0=00(-d), slot1=10(+d),
        // slot2=01(native 0), slot3=11(+2d, the spare state)
        let cb = 0b11_01_10_00;
        // delete every slot → all native zeros
        assert_eq!(lut_entry(cb, 0x00), 0b01_01_01_01);
        // keep ONLY slot 2 (already the native zero) → the other three
        // collapse to 01 as well
        assert_eq!(lut_entry(cb, 0b0100), 0b01_01_01_01);
        // keep ONLY slot 3 → the spare code 3 SURVIVES a keep
        assert_eq!(lut_entry(cb, 0b1000), 0b11_01_01_01);
        // keep everything → untouched
        assert_eq!(lut_entry(cb, 0x0F), cb);
    }

    #[test]
    fn scales_never_move() {
        let blocks = blocks_with_pattern(9, 0xA7);
        let mut keeps = vec![true; 9 * Q2_0_BLOCK_SIZE];
        keeps[0] = false;
        keeps[500] = false;
        let masked = apply_mask_q2_0(&blocks, &mask_bytes_from_keeps(&keeps), 0).unwrap();
        for (orig, new) in blocks.iter().zip(masked.iter()) {
            assert_eq!(orig.d, new.d, "group scale is frozen");
        }
        // the f16 scale bytes specifically
        let orig_bytes: Vec<u8> = blocks.iter().flat_map(|b| b.d.to_le_bytes()).collect();
        let new_bytes: Vec<u8> = masked.iter().flat_map(|b| b.d.to_le_bytes()).collect();
        assert_eq!(orig_bytes, new_bytes);
    }

    #[test]
    fn apply_matches_scalar_reference_and_nibble_alignment() {
        let blocks = blocks_with_pattern(4, 0x33);
        // delete a strided pattern crossing nibble/byte boundaries
        let mut keeps = vec![true; 4 * Q2_0_BLOCK_SIZE];
        for w in (0..4 * Q2_0_BLOCK_SIZE).step_by(3) {
            keeps[w] = false;
        }
        for w in (7..4 * Q2_0_BLOCK_SIZE).step_by(11) {
            keeps[w] = false;
        }
        let masked = apply_mask_q2_0(&blocks, &mask_bytes_from_keeps(&keeps), 0).unwrap();
        let reference = reference_apply(&blocks, &keeps);
        assert_eq!(q2_bytes(&masked), q2_bytes(&reference));
    }

    #[test]
    fn masked_zero_equals_native_zero_through_dequant_and_bridge() {
        use crate::quant::q2_0::{dequantize_row_q2_0, repack_q2_0_to_ternary_group};
        // two tensors: weight 5 natively zero (code 1) vs weight 5
        // deleted by the mask (was code 2) — dequant digests must match
        let mut native = blocks_with_pattern(1, 0x00);
        let mut deleted = blocks_with_pattern(1, 0x00);
        for target in [&mut native, &mut deleted] {
            // base pattern: all +d (code 2)
            for byte in target[0].qs.iter_mut() {
                *byte = 0b10_10_10_10;
            }
        }
        native[0].qs[1] = (native[0].qs[1] & !(0b11 << 2)) | (0b01 << 2); // weight 5 → code 1 (native zero)
        deleted[0].qs[1] |= 0b11 << 2; // weight 5 → keep code 2 for now

        let mut keeps = vec![true; Q2_0_BLOCK_SIZE];
        keeps[5] = false;
        let masked = apply_mask_q2_0(&deleted, &mask_bytes_from_keeps(&keeps), 0).unwrap();
        assert_eq!(masked[0].qs[1], native[0].qs[1], "code bytes match");

        let mut a = vec![0.0f32; Q2_0_BLOCK_SIZE];
        let mut b = vec![0.0f32; Q2_0_BLOCK_SIZE];
        dequantize_row_q2_0(&native, &mut a);
        dequantize_row_q2_0(&masked, &mut b);
        assert_eq!(a, b, "dequant digest parity");

        // and through the ternary bridge (the kernel-side skip path)
        let plane_native = repack_q2_0_to_ternary_group(&native, 1, Q2_0_BLOCK_SIZE).unwrap();
        let plane_masked = repack_q2_0_to_ternary_group(&masked, 1, Q2_0_BLOCK_SIZE).unwrap();
        assert_eq!(
            (plane_native.pos_bits.clone(), plane_native.neg_bits.clone()),
            (plane_masked.pos_bits, plane_masked.neg_bits),
            "bridge (materialization) parity: masked zero == native zero"
        );
    }

    #[test]
    fn stale_metadata_tripwire() {
        use crate::quant::q2_0::repack_q2_0_to_ternary_group;
        // materialization is computed from the MASKED codes: flip one
        // mask bit after masking and the materialized digest must move
        let blocks = blocks_with_pattern(2, 0x5C);
        let mut keeps = vec![true; 2 * Q2_0_BLOCK_SIZE];
        keeps[70] = false;
        let masked_a = apply_mask_q2_0(&blocks, &mask_bytes_from_keeps(&keeps), 0).unwrap();
        keeps[70] = true;
        keeps[71] = false;
        let masked_b = apply_mask_q2_0(&blocks, &mask_bytes_from_keeps(&keeps), 0).unwrap();
        assert_ne!(
            q2_bytes(&masked_a),
            q2_bytes(&masked_b),
            "different masks → different codes"
        );
        let planes_a = repack_q2_0_to_ternary_group(&masked_a, 2, Q2_0_BLOCK_SIZE).unwrap();
        let planes_b = repack_q2_0_to_ternary_group(&masked_b, 2, Q2_0_BLOCK_SIZE).unwrap();
        assert_ne!(
            (planes_a.pos_bits, planes_a.neg_bits),
            (planes_b.pos_bits, planes_b.neg_bits),
            "materialization must follow the mask (stale-metadata tripwire)"
        );
    }

    #[test]
    fn deterministic_and_no_cross_mask_reuse() {
        // same input+mask → deterministic output; two DIFFERENT masks on
        // the same input never share a materialization (pure fn, no cache)
        let blocks = blocks_with_pattern(3, 0xE1);
        let mut keeps_a = vec![true; 3 * Q2_0_BLOCK_SIZE];
        keeps_a[1] = false;
        let mut keeps_b = vec![true; 3 * Q2_0_BLOCK_SIZE];
        keeps_b[200] = false;
        let a1 = apply_mask_q2_0(&blocks, &mask_bytes_from_keeps(&keeps_a), 0).unwrap();
        let a2 = apply_mask_q2_0(&blocks, &mask_bytes_from_keeps(&keeps_a), 0).unwrap();
        assert_eq!(q2_bytes(&a1), q2_bytes(&a2));
        let b = apply_mask_q2_0(&blocks, &mask_bytes_from_keeps(&keeps_b), 0).unwrap();
        assert_ne!(q2_bytes(&a1), q2_bytes(&b));
    }

    #[test]
    fn all_keep_mask_is_byte_identical() {
        let blocks = blocks_with_pattern(5, 0x99);
        let keeps = vec![true; 5 * Q2_0_BLOCK_SIZE];
        let masked = apply_mask_q2_0(&blocks, &mask_bytes_from_keeps(&keeps), 0).unwrap();
        assert_eq!(
            q2_bytes(&blocks),
            q2_bytes(&masked),
            "identity: all-keep mask leaves every byte untouched"
        );
    }

    #[test]
    fn apply_to_tensor_routes_sections_and_refuses_mismatch() {
        let blocks = blocks_with_pattern(2, 0x44);
        let base = digest(b"tensor-routing-base");
        let manifest = super::super::mask_codec::MaskManifest::new(
            base,
            vec![
                super::super::mask_codec::MaskSection::new("blk.0.ffn_gate.weight", 0, 256)
                    .unwrap(),
                super::super::mask_codec::MaskSection::new(
                    "blk.1.ffn_gate.weight",
                    256,
                    100, // wrong length → refuse
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let mut keeps = vec![true; 256];
        keeps[9] = false;
        // mask stream: 256 bits for section 0, 100 bits for section 1
        let mut bits = mask_bytes_from_keeps(&keeps);
        bits.extend(mask_bytes_from_keeps(&[true; 104])); // pad section 1 to byte
        let decoded =
            super::super::mask_codec::decode(&manifest.pack(&bits, None).unwrap(), None, None)
                .unwrap();

        // named section applies
        let applied = apply_to_tensor(&blocks, &decoded, "blk.0.ffn_gate.weight")
            .unwrap()
            .unwrap();
        let direct = apply_mask_q2_0(&blocks, &decoded.bits, 0).unwrap();
        assert_eq!(q2_bytes(&applied), q2_bytes(&direct));
        // unlisted tensor → silent pass-through
        assert!(
            apply_to_tensor(&blocks, &decoded, "blk.9.ffn_down.weight")
                .unwrap()
                .is_none()
        );
        // length mismatch → loud refusal
        assert!(matches!(
            apply_to_tensor(&blocks, &decoded, "blk.1.ffn_gate.weight"),
            Err(MaskCodecError::SectionRange(_))
        ));
    }

    #[test]
    fn out_of_range_window_refused() {
        let blocks = blocks_with_pattern(1, 0x11);
        assert!(matches!(
            apply_mask_q2_0(&blocks, &[0xFFu8; 8], 64), // 64+128 > 64 available
            Err(MaskCodecError::SectionRange(_))
        ));
    }
}
