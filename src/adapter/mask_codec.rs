//! The deletion-mask artifact codec (Issue 1208 T1).
//!
//! A deletion mask is a per-weight binary keep/delete vector over a
//! frozen model's gate projections: 1 = keep, 0 = delete (→ native
//! zero). The deployed adapter writes zero new weight values, so the
//! whole artifact is a manifest + a bit stream — this module owns the
//! on-disk format end to end (self-contained leaf: only external
//! crates, no riir-infer internals).
//!
//! File layout (little-endian throughout):
//!
//! ```text
//! magic            5B   b"RIMSK"
//! format_version   u32  (1)
//! base_model       32B  BLAKE3 of the bound base model — refused
//!                       BEFORE any payload work when the caller passes
//!                       the loaded model's digest (wrong-base law)
//! payload_blake3   32B  BLAKE3 over the UNCOMPRESSED bit-packed mask
//! payload_len_raw  u64  mask bit length (always a multiple of 8)
//! zero_code        u8   v1 pins PQ2_0's native zero (code 1)
//! sig_len          u32  0 = unsigned, else 64 (Ed25519 over every
//!                       byte of the artifact up to the signature)
//! section_count    u32
//! sections         per entry: name_len u16, name UTF-8,
//!                       bit_offset u64, bit_len u64   (into the mask
//!                       bit stream; non-overlapping)
//! compressed_len   u64
//! payload          RLE-then-zstd of the bit-packed mask
//! signature        sig_len bytes (optional)
//! ```
//!
//! Bit order (pinned; the T3 PQ2_0 byte-pair LUT consumes these bytes
//! directly): LSB-first within each byte — byte `b`, bit `j` is weight
//! `8*b + j`.
//!
//! Refusal battery (all loud + typed, [`MaskCodecError`]): bad magic,
//! unknown version, wrong base digest, bad zero-code, truncated
//! header/map/payload/signature, tampered payload (BLAKE3 mismatch),
//! decompression failure or expansion beyond the declared raw length
//! (bomb guard), malformed RLE, out-of-range/overlapping sections,
//! duplicate names, trailing bytes, router/gate-logit tensor names
//! (the paper's router-frozen law, enforced by the format at pack AND
//! decode time), malformed/bad signature.
//!
//! Identity pin: an empty mask (no sections, or every section bit set)
//! is [`DecodedMask::is_identity`] and must behave as the base — the
//! byte-identical FORWARD half of that pin is T2's G1.
//!
//! Controlling records: Issue 1208; riir-train `.research/493`
//! (track (a) terminal state). Feature `deletion_mask` is the
//! pre-pinned promotion bar.

use std::io::Read;

use ed25519_dalek::{Signer, Verifier};
pub use ed25519_dalek::{SigningKey, VerifyingKey};

pub const MAGIC: [u8; 5] = *b"RIMSK";
pub const FORMAT_VERSION: u32 = 1;
/// PQ2_0's native zero symbol: codes {0→−1, 1→0, 2→+1, 3→+2}
/// (`src/quant/q2_0.rs`). v1 artifacts may only carry this value.
pub const PQ2_0_ZERO_CODE: u8 = 1;
const ZSTD_COMPRESSION_LEVEL: i32 = 3;
const SIGNATURE_LEN: usize = 64;
/// Router / gate-logit substrings a section map may never name: the
/// mask addresses gate PROJECTIONS (incl. MoE expert gates, e.g.
/// `ffn_gate_exps`), never the routing math.
const REFUSED_NAME_SUBSTRINGS: [&str; 3] = ["ffn_gate_inp", "router", "gate_logit"];

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MaskCodecError {
    #[error("bad magic: expected RIMSK, found {0:02x?}")]
    BadMagic([u8; 5]),
    #[error("unknown format version {found} (supported: {supported})")]
    UnknownVersion { found: u32, supported: u32 },
    #[error(
        "base-model digest mismatch: artifact binds {found:02x?}, this model is {expected:02x?}"
    )]
    BaseMismatch { expected: [u8; 32], found: [u8; 32] },
    #[error("bad zero-code {found}: v1 pins the PQ2_0 native zero (code 1)")]
    BadZeroCode { found: u8 },
    #[error("truncated header")]
    TruncatedHeader,
    #[error("truncated section map")]
    TruncatedMap,
    #[error("truncated payload")]
    TruncatedPayload,
    #[error("truncated signature")]
    TruncatedSignature,
    #[error("payload hash mismatch: declared {expected:02x?}, decoded {found:02x?}")]
    PayloadTampered { expected: [u8; 32], found: [u8; 32] },
    #[error("payload failed to decompress: {0}")]
    PayloadCorrupt(String),
    #[error("payload raw length {0} is not byte-aligned")]
    UnalignedPayload(u64),
    #[error("RLE stream malformed: {0}")]
    RleMalformed(String),
    #[error("section {0} out of payload range or overlapping another section")]
    SectionRange(String),
    #[error("duplicate section {0}")]
    DuplicateSection(String),
    #[error("empty tensor name")]
    EmptyName,
    #[error("router/gate-logit tensor refused by format: {0} (router-frozen law)")]
    RouterTensorRefused(String),
    #[error("malformed signature: expected 64 bytes, found {0}")]
    MalformedSignature(usize),
    #[error("signature verification failed")]
    BadSignature,
    #[error("artifact is unsigned but a trusted key was supplied")]
    UnsignedArtifact,
    #[error("{0} trailing bytes after the artifact end")]
    TrailingBytes(usize),
}

/// Router / gate-logit name classifier — the format-level enforcement
/// of the paper's router-frozen law. Case-insensitive substring match.
pub fn refused_tensor_name(tensor_name: &str) -> bool {
    let lowered = tensor_name.to_lowercase();
    REFUSED_NAME_SUBSTRINGS.iter().any(|s| lowered.contains(s))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaskSection {
    pub tensor_name: String,
    pub bit_offset: u64,
    pub bit_len: u64,
}

impl MaskSection {
    pub fn new(
        tensor_name: impl Into<String>,
        bit_offset: u64,
        bit_len: u64,
    ) -> Result<Self, MaskCodecError> {
        let tensor_name = tensor_name.into();
        if tensor_name.is_empty() {
            return Err(MaskCodecError::EmptyName);
        }
        if refused_tensor_name(&tensor_name) {
            return Err(MaskCodecError::RouterTensorRefused(tensor_name));
        }
        Ok(Self {
            tensor_name,
            bit_offset,
            bit_len,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaskManifest {
    /// BLAKE3 digest of the bound base model.
    pub base_model_blake3: [u8; 32],
    /// Zero-code constant carried by the artifact; v1 pins
    /// [`PQ2_0_ZERO_CODE`].
    pub zero_code: u8,
    /// Tensor name → bit range into the mask bit stream.
    pub sections: Vec<MaskSection>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedMask {
    pub manifest: MaskManifest,
    /// Bit-packed mask, LSB-first: bit `j` of byte `b` is weight
    /// `8*b + j`; 1 = keep, 0 = delete (→ native zero).
    pub bits: Vec<u8>,
    pub signature: Option<[u8; SIGNATURE_LEN]>,
}

impl DecodedMask {
    /// True when applying this mask is a no-op: no sections at all, or
    /// every section bit is 1 (keep). The forward-level half of the
    /// identity pin (byte-identical output vs the base) is T2's G1.
    pub fn is_identity(&self) -> bool {
        self.manifest.sections.iter().all(|s| {
            (s.bit_offset..s.bit_offset.saturating_add(s.bit_len)).all(|p| {
                self.bits
                    .get((p / 8) as usize)
                    .is_some_and(|b| ((b >> (p % 8)) & 1) == 1)
            })
        })
    }
}

fn max_section_end(sections: &[MaskSection]) -> Result<u64, MaskCodecError> {
    let mut max_end = 0u64;
    for section in sections {
        let end = section
            .bit_offset
            .checked_add(section.bit_len)
            .ok_or_else(|| MaskCodecError::SectionRange(section.tensor_name.clone()))?;
        if end > max_end {
            max_end = end;
        }
    }
    Ok(max_end)
}

fn validate_sections(sections: &[MaskSection]) -> Result<(), MaskCodecError> {
    let mut sorted: Vec<&MaskSection> = sections.iter().collect();
    sorted.sort_by_key(|s| (s.bit_offset, s.bit_len));
    let mut seen: Vec<&str> = Vec::with_capacity(sections.len());
    let mut prev_end: Option<u64> = None;
    for section in sorted {
        if section.tensor_name.is_empty() {
            return Err(MaskCodecError::EmptyName);
        }
        if refused_tensor_name(&section.tensor_name) {
            return Err(MaskCodecError::RouterTensorRefused(
                section.tensor_name.clone(),
            ));
        }
        if section.bit_len == 0 {
            return Err(MaskCodecError::SectionRange(section.tensor_name.clone()));
        }
        if seen.contains(&section.tensor_name.as_str()) {
            return Err(MaskCodecError::DuplicateSection(
                section.tensor_name.clone(),
            ));
        }
        seen.push(&section.tensor_name);
        let end = section
            .bit_offset
            .checked_add(section.bit_len)
            .ok_or_else(|| MaskCodecError::SectionRange(section.tensor_name.clone()))?;
        if prev_end.is_some_and(|bound| section.bit_offset < bound) {
            return Err(MaskCodecError::SectionRange(section.tensor_name.clone()));
        }
        prev_end = Some(end);
    }
    Ok(())
}

impl MaskManifest {
    pub fn new(
        base_model_blake3: [u8; 32],
        sections: Vec<MaskSection>,
    ) -> Result<Self, MaskCodecError> {
        validate_sections(&sections)?;
        Ok(Self {
            base_model_blake3,
            zero_code: PQ2_0_ZERO_CODE,
            sections,
        })
    }

    /// Serialize the artifact. `bits` is the bit-packed mask covering
    /// every section (LSB-first; trailing bytes beyond the last
    /// section are carried verbatim).
    pub fn pack(
        &self,
        bits: &[u8],
        signing_key: Option<&SigningKey>,
    ) -> Result<Vec<u8>, MaskCodecError> {
        if self.zero_code != PQ2_0_ZERO_CODE {
            return Err(MaskCodecError::BadZeroCode {
                found: self.zero_code,
            });
        }
        validate_sections(&self.sections)?;
        let needed = max_section_end(&self.sections)?.div_ceil(8) as usize;
        if bits.len() < needed {
            return Err(MaskCodecError::SectionRange(format!(
                "mask buffer {} bytes < sections need {}",
                bits.len(),
                needed
            )));
        }

        let payload_blake3 = blake3::hash(bits);
        let rle = rle_encode(bits);
        let compressed = zstd::stream::encode_all(&rle[..], ZSTD_COMPRESSION_LEVEL)
            .map_err(|e| MaskCodecError::PayloadCorrupt(e.to_string()))?;

        let mut out = Vec::with_capacity(96 + compressed.len());
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&self.base_model_blake3);
        out.extend_from_slice(payload_blake3.as_bytes());
        out.extend_from_slice(&((bits.len() as u64) * 8).to_le_bytes());
        out.push(self.zero_code);
        let sig_len = if signing_key.is_some() {
            SIGNATURE_LEN as u32
        } else {
            0
        };
        out.extend_from_slice(&sig_len.to_le_bytes());
        out.extend_from_slice(&(self.sections.len() as u32).to_le_bytes());
        for section in &self.sections {
            let name = section.tensor_name.as_bytes();
            out.extend_from_slice(&(name.len() as u16).to_le_bytes());
            out.extend_from_slice(name);
            out.extend_from_slice(&section.bit_offset.to_le_bytes());
            out.extend_from_slice(&section.bit_len.to_le_bytes());
        }
        out.extend_from_slice(&(compressed.len() as u64).to_le_bytes());
        out.extend_from_slice(&compressed);

        if let Some(key) = signing_key {
            let signature = key.sign(&out);
            out.extend_from_slice(&signature.to_bytes());
        }
        Ok(out)
    }
}

/// Decode an artifact. `expected_base` is the loaded model's BLAKE3 —
/// a mismatch refuses BEFORE the map/payload are touched (wrong-base
/// law). `trusted_key`, when given, demands a valid Ed25519 signature
/// over the artifact (an unsigned artifact then refuses loudly).
pub fn decode(
    bytes: &[u8],
    expected_base: Option<&[u8; 32]>,
    trusted_key: Option<&VerifyingKey>,
) -> Result<DecodedMask, MaskCodecError> {
    let mut cur = bytes;
    let magic: [u8; 5] = take(&mut cur, 5, MaskCodecError::TruncatedHeader)?
        .try_into()
        .map_err(|_| MaskCodecError::TruncatedHeader)?;
    if magic != MAGIC {
        return Err(MaskCodecError::BadMagic(magic));
    }
    let version = read_u32(&mut cur)?;
    if version != FORMAT_VERSION {
        return Err(MaskCodecError::UnknownVersion {
            found: version,
            supported: FORMAT_VERSION,
        });
    }
    let base_model_blake3: [u8; 32] = take(&mut cur, 32, MaskCodecError::TruncatedHeader)?
        .try_into()
        .map_err(|_| MaskCodecError::TruncatedHeader)?;
    if let Some(expected) = expected_base
        && *expected != base_model_blake3
    {
        return Err(MaskCodecError::BaseMismatch {
            expected: *expected,
            found: base_model_blake3,
        });
    }
    let payload_blake3: [u8; 32] = take(&mut cur, 32, MaskCodecError::TruncatedHeader)?
        .try_into()
        .map_err(|_| MaskCodecError::TruncatedHeader)?;
    let payload_len_raw = read_u64(&mut cur)?;
    if payload_len_raw % 8 != 0 {
        return Err(MaskCodecError::UnalignedPayload(payload_len_raw));
    }
    let zero_code = take(&mut cur, 1, MaskCodecError::TruncatedHeader)?[0];
    if zero_code != PQ2_0_ZERO_CODE {
        return Err(MaskCodecError::BadZeroCode { found: zero_code });
    }
    let sig_len = read_u32(&mut cur)?;
    let section_count = read_u32(&mut cur)?;

    let mut sections = Vec::with_capacity(section_count.min(1024) as usize);
    let mut names: Vec<String> = Vec::with_capacity(section_count.min(1024) as usize);
    for _ in 0..section_count {
        let name_len = u16::from_le_bytes(
            take(&mut cur, 2, MaskCodecError::TruncatedMap)?
                .try_into()
                .map_err(|_| MaskCodecError::TruncatedMap)?,
        ) as usize;
        let name_bytes = take(&mut cur, name_len, MaskCodecError::TruncatedMap)?;
        let tensor_name = std::str::from_utf8(name_bytes)
            .map_err(|_| MaskCodecError::TruncatedMap)?
            .to_owned();
        if refused_tensor_name(&tensor_name) {
            return Err(MaskCodecError::RouterTensorRefused(tensor_name));
        }
        if tensor_name.is_empty() {
            return Err(MaskCodecError::EmptyName);
        }
        if names.contains(&tensor_name) {
            return Err(MaskCodecError::DuplicateSection(tensor_name));
        }
        names.push(tensor_name.clone());
        let bit_offset = read_u64(&mut cur)?;
        let bit_len = read_u64(&mut cur)?;
        sections.push(MaskSection {
            tensor_name,
            bit_offset,
            bit_len,
        });
    }
    validate_sections(&sections)?;
    if max_section_end(&sections)? > payload_len_raw {
        let offender = sections
            .iter()
            .find(|s| s.bit_offset.saturating_add(s.bit_len) > payload_len_raw)
            .map(|s| s.tensor_name.clone())
            .unwrap_or_default();
        return Err(MaskCodecError::SectionRange(offender));
    }
    let expected_byte_len = (payload_len_raw / 8) as usize;

    let compressed_len =
        usize::try_from(read_u64(&mut cur)?).map_err(|_| MaskCodecError::TruncatedPayload)?;
    let compressed = take(&mut cur, compressed_len, MaskCodecError::TruncatedPayload)?;

    let signature: Option<[u8; SIGNATURE_LEN]> = if sig_len == 0 {
        None
    } else {
        if sig_len as usize != SIGNATURE_LEN {
            return Err(MaskCodecError::MalformedSignature(sig_len as usize));
        }
        let sig = take(&mut cur, SIGNATURE_LEN, MaskCodecError::TruncatedSignature)?;
        let sig_bytes: [u8; SIGNATURE_LEN] = sig
            .try_into()
            .map_err(|_| MaskCodecError::TruncatedSignature)?;
        Some(sig_bytes)
    };
    if !cur.is_empty() {
        return Err(MaskCodecError::TrailingBytes(cur.len()));
    }

    let raw = inflate_bounded(compressed, expected_byte_len)?;
    let bits = rle_decode(&raw, payload_len_raw)?;
    let found = blake3::hash(&bits);
    if found.as_bytes() != &payload_blake3 {
        return Err(MaskCodecError::PayloadTampered {
            expected: payload_blake3,
            found: *found.as_bytes(),
        });
    }

    if let Some(key) = trusted_key {
        let signature = signature.ok_or(MaskCodecError::UnsignedArtifact)?;
        let signature = ed25519_dalek::Signature::from_slice(&signature)
            .map_err(|_| MaskCodecError::MalformedSignature(signature.len()))?;
        let signed_prefix_len = bytes
            .len()
            .checked_sub(SIGNATURE_LEN)
            .ok_or(MaskCodecError::TruncatedSignature)?;
        key.verify(&bytes[..signed_prefix_len], &signature)
            .map_err(|_| MaskCodecError::BadSignature)?;
    }

    Ok(DecodedMask {
        manifest: MaskManifest {
            base_model_blake3,
            zero_code,
            sections,
        },
        bits,
        signature,
    })
}

fn take<'a>(
    cur: &mut &'a [u8],
    n: usize,
    truncated: MaskCodecError,
) -> Result<&'a [u8], MaskCodecError> {
    if cur.len() < n {
        return Err(truncated);
    }
    let (head, rest) = cur.split_at(n);
    *cur = rest;
    Ok(head)
}

fn read_u32(cur: &mut &[u8]) -> Result<u32, MaskCodecError> {
    let bytes: [u8; 4] = take(cur, 4, MaskCodecError::TruncatedHeader)?
        .try_into()
        .map_err(|_| MaskCodecError::TruncatedHeader)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(cur: &mut &[u8]) -> Result<u64, MaskCodecError> {
    let bytes: [u8; 8] = take(cur, 8, MaskCodecError::TruncatedHeader)?
        .try_into()
        .map_err(|_| MaskCodecError::TruncatedHeader)?;
    Ok(u64::from_le_bytes(bytes))
}

/// Decompress with a hard output cap (decompression-bomb guard): the
/// decompressed content is the RLE STREAM, whose worst case is 9 bytes
/// per mask BIT (alternating runs) — the cap is `mask_bytes * 72`; the
/// EXACT bit accounting is `rle_decode`'s job, not this layer's. The
/// output buffer grows with ACTUAL decoded bytes only — a huge declared
/// length never pre-allocates.
fn inflate_bounded(compressed: &[u8], mask_len: usize) -> Result<Vec<u8>, MaskCodecError> {
    let decoder = zstd::stream::Decoder::new(compressed)
        .map_err(|e| MaskCodecError::PayloadCorrupt(e.to_string()))?;
    // 9 bytes per bit worst case × 8 bits per byte.
    let rle_cap = mask_len
        .checked_mul(9 * 8)
        .ok_or_else(|| MaskCodecError::PayloadCorrupt("mask length overflows".to_owned()))?;
    // Read ONE byte past the cap: at-or-under = hand to the RLE decoder;
    // cap+1 = the stream expands beyond any representable RLE (bomb).
    let mut limited = decoder.take(rle_cap as u64 + 1);
    let mut raw = Vec::new();
    limited
        .read_to_end(&mut raw)
        .map_err(|e| MaskCodecError::PayloadCorrupt(e.to_string()))?;
    if raw.len() > rle_cap {
        return Err(MaskCodecError::PayloadCorrupt(format!(
            "RLE stream expands beyond the representable cap: {} bytes > {}",
            raw.len(),
            rle_cap
        )));
    }
    Ok(raw)
}

/// Runs of `(count u64 LE, value u8)` over the bit stream, LSB-first;
/// canonical form emits maximal runs.
fn rle_encode(bits: &[u8]) -> Vec<u8> {
    let total_bits = bits.len() * 8;
    let mut out = Vec::new();
    let mut run_value = 0u8;
    let mut run_len = 0u64;
    for p in 0..total_bits {
        let v = (bits[p / 8] >> (p % 8)) & 1;
        if run_len > 0 && v != run_value {
            out.extend_from_slice(&run_len.to_le_bytes());
            out.push(run_value);
            run_len = 0;
        }
        run_value = v;
        run_len += 1;
    }
    if run_len > 0 {
        out.extend_from_slice(&run_len.to_le_bytes());
        out.push(run_value);
    }
    out
}

fn rle_decode(raw: &[u8], payload_len_raw: u64) -> Result<Vec<u8>, MaskCodecError> {
    let mut bits = vec![0u8; (payload_len_raw / 8) as usize];
    let mut cur = raw;
    let mut pos = 0u64;
    while !cur.is_empty() {
        if cur.len() < 9 {
            return Err(MaskCodecError::RleMalformed(format!(
                "truncated run at bit {} ({} bytes left)",
                pos,
                cur.len()
            )));
        }
        let count = u64::from_le_bytes(
            cur[..8]
                .try_into()
                .map_err(|_| MaskCodecError::RleMalformed("bad run count".to_owned()))?,
        );
        let value = cur[8];
        if value > 1 {
            return Err(MaskCodecError::RleMalformed(format!(
                "run value {value} at bit {pos}"
            )));
        }
        cur = &cur[9..];
        if pos + count > payload_len_raw {
            return Err(MaskCodecError::RleMalformed(format!(
                "run to bit {} exceeds declared length {payload_len_raw}",
                pos + count
            )));
        }
        if value == 1 {
            for p in pos..pos + count {
                bits[(p / 8) as usize] |= 1 << (p % 8);
            }
        }
        pos += count;
    }
    if pos != payload_len_raw {
        return Err(MaskCodecError::RleMalformed(format!(
            "decoded {pos} bits, artifact declares {payload_len_raw}"
        )));
    }
    Ok(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(bytes: &[u8]) -> [u8; 32] {
        *blake3::hash(bytes).as_bytes()
    }

    #[test]
    fn refused_names_classify() {
        assert!(refused_tensor_name("blk.0.ffn_gate_inp.weight"));
        assert!(refused_tensor_name("blk.3.Router_Proj.weight"));
        assert!(refused_tensor_name("model.gate_logit.bias"));
        assert!(!refused_tensor_name("blk.0.ffn_gate.weight"));
        assert!(!refused_tensor_name("blk.0.ffn_gate_exps.weight"));
        assert!(!refused_tensor_name("blk.0.ffn_up.weight"));
    }

    #[test]
    fn section_constructors_refuse_routers_and_empty() {
        assert!(matches!(
            MaskSection::new("blk.0.ffn_gate_inp.weight", 0, 8),
            Err(MaskCodecError::RouterTensorRefused(_))
        ));
        assert!(matches!(
            MaskSection::new("", 0, 8),
            Err(MaskCodecError::EmptyName)
        ));
    }

    #[test]
    fn manifest_new_refuses_overlap_dup_and_zero_len() {
        let base = digest(b"base");
        let good = |n: &str, o: u64, l: u64| MaskSection::new(n, o, l).unwrap();
        assert!(matches!(
            MaskManifest::new(
                base,
                vec![good("a.weight", 0, 64), good("b.weight", 32, 64)]
            ),
            Err(MaskCodecError::SectionRange(_))
        ));
        assert!(matches!(
            MaskManifest::new(
                base,
                vec![good("a.weight", 0, 64), good("a.weight", 64, 64)]
            ),
            Err(MaskCodecError::DuplicateSection(_))
        ));
        assert!(matches!(
            MaskManifest::new(base, vec![good("a.weight", 0, 0)]),
            Err(MaskCodecError::SectionRange(_))
        ));
    }

    #[test]
    fn rle_roundtrip_and_rejects() {
        let bits = [0b1010_1010u8, 0b0000_1111, 0xFF];
        let encoded = rle_encode(&bits);
        let decoded = rle_decode(&encoded, 24).unwrap();
        assert_eq!(decoded, bits);
        // overflow beyond the declared length
        let mut over = encoded.clone();
        over[..8].copy_from_slice(&25u64.to_le_bytes());
        assert!(matches!(
            rle_decode(&over, 24),
            Err(MaskCodecError::RleMalformed(_))
        ));
        // short stream
        assert!(matches!(
            rle_decode(&encoded[..encoded.len() - 9], 24),
            Err(MaskCodecError::RleMalformed(_))
        ));
        // bad run value
        let mut bad_value = encoded.clone();
        let last = bad_value.len() - 1;
        bad_value[last] = 2;
        assert!(matches!(
            rle_decode(&bad_value, 24),
            Err(MaskCodecError::RleMalformed(_))
        ));
    }

    #[test]
    fn pack_refuses_short_buffer_and_bad_zero_code() {
        let base = digest(b"base");
        let section = MaskSection::new("blk.0.ffn_gate.weight", 0, 16).unwrap();
        let manifest = MaskManifest::new(base, vec![section]).unwrap();
        assert!(matches!(
            manifest.pack(&[0xFFu8], None),
            Err(MaskCodecError::SectionRange(_))
        ));
        let mut bad_zero = manifest.clone();
        bad_zero.zero_code = 2;
        assert!(matches!(
            bad_zero.pack(&[0xFF, 0xFF], None),
            Err(MaskCodecError::BadZeroCode { found: 2 })
        ));
        // pack-side defense: a struct-literal manifest naming a router
        let router_manifest = MaskManifest {
            base_model_blake3: base,
            zero_code: PQ2_0_ZERO_CODE,
            sections: vec![MaskSection {
                tensor_name: "blk.0.ffn_gate_inp.weight".to_owned(),
                bit_offset: 0,
                bit_len: 16,
            }],
        };
        assert!(matches!(
            router_manifest.pack(&[0xFF, 0xFF], None),
            Err(MaskCodecError::RouterTensorRefused(_))
        ));
    }

    #[test]
    fn roundtrip_preserves_bits() {
        let base = digest(b"roundtrip");
        let sections = vec![
            MaskSection::new("blk.0.ffn_gate.weight", 0, 24).unwrap(),
            MaskSection::new("blk.0.ffn_gate_exps.weight", 24, 16).unwrap(),
        ];
        let manifest = MaskManifest::new(base, sections).unwrap();
        let mut bits = [0xFFu8; 5];
        bits[0] = 0b1111_0111; // bit 3 deleted
        bits[3] = 0b1111_1011; // bit 27 deleted (offset-24 section)
        let artifact = manifest.pack(&bits, None).unwrap();
        let decoded = decode(&artifact, Some(&base), None).unwrap();
        assert_eq!(decoded.bits, bits);
        assert_eq!(decoded.manifest, manifest);
        assert!(decoded.signature.is_none());
        assert!(!decoded.is_identity());
    }

    #[test]
    fn identity_mask_flags_and_trailing_bytes_refuse() {
        let base = digest(b"identity");
        let sections = vec![MaskSection::new("blk.1.ffn_gate.weight", 0, 8).unwrap()];
        let manifest = MaskManifest::new(base, sections).unwrap();
        // all-ones over the section → identity
        let artifact = manifest.pack(&[0xFF], None).unwrap();
        let decoded = decode(&artifact, None, None).unwrap();
        assert!(decoded.is_identity());
        // one deleted bit → not identity
        let one_zero = [0xFEu8];
        let artifact = manifest.pack(&one_zero, None).unwrap();
        let decoded = decode(&artifact, None, None).unwrap();
        assert!(!decoded.is_identity());
        // trailing garbage after a valid artifact → loud refusal
        let mut padded = artifact;
        padded.push(0);
        assert!(matches!(
            decode(&padded, None, None),
            Err(MaskCodecError::TrailingBytes(1))
        ));
    }
}
