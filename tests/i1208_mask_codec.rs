//! Issue 1208 T1 — the deletion-mask codec's file-level battery: the
//! committed golden fixture (byte-identity on re-pack), the typed
//! refusal battery, the identity pin, and the Ed25519 path.

use riir_infer_core::adapter::mask_codec::{
    DecodedMask, FORMAT_VERSION, MAGIC, MaskCodecError, MaskManifest, MaskSection, PQ2_0_ZERO_CODE,
    SigningKey, VerifyingKey, decode, refused_tensor_name,
};

const GOLDEN_PATH: &str = "tests/fixtures/i1208_mask_codec_golden.bin";
const GOLDEN_BLAKE3: &str = "cc552c8fda5240bb021aca51f985a9f20afd14938c207cb882f8a8c9dad0bd0a";
const TEST_SIGNING_SEED: [u8; 32] = *b"i1208-golden-signing-seed-000000";

fn digest(bytes: &[u8]) -> [u8; 32] {
    *blake3::hash(bytes).as_bytes()
}

/// The golden manifest: 3 gate sections over 112 mask bytes (896 bits).
fn golden_manifest() -> MaskManifest {
    MaskManifest::new(
        digest(b"i1208-golden-base"),
        vec![
            MaskSection::new("blk.0.ffn_gate.weight", 0, 256).unwrap(),
            MaskSection::new("blk.1.ffn_gate.weight", 256, 128).unwrap(),
            MaskSection::new("blk.0.ffn_gate_exps.weight", 384, 512).unwrap(),
        ],
    )
    .unwrap()
}

/// The golden mask: all-keep except six pinned delete bits.
fn golden_bits() -> Vec<u8> {
    let mut bits = vec![0xFFu8; 112];
    for &p in &[3u64, 17, 255, 400, 401, 511] {
        bits[(p / 8) as usize] &= !(1 << (p % 8));
    }
    bits
}

#[test]
#[ignore]
fn generate_golden_fixture() {
    let artifact = golden_manifest().pack(&golden_bits(), None).unwrap();
    std::fs::write(GOLDEN_PATH, &artifact).unwrap();
    println!("golden blake3: {}", blake3::hash(&artifact).to_hex());
}

#[test]
fn golden_fixture_decodes_and_repacks_byte_identical() {
    let artifact = std::fs::read(GOLDEN_PATH).expect("golden fixture committed");
    assert_eq!(
        blake3::hash(&artifact).to_hex().as_str(),
        GOLDEN_BLAKE3,
        "golden fixture drifted — regenerate deliberately"
    );
    let decoded = decode(&artifact, Some(&digest(b"i1208-golden-base")), None).unwrap();
    assert_eq!(decoded.manifest, golden_manifest());
    assert_eq!(decoded.bits, golden_bits());
    assert!(!decoded.is_identity());
    assert!(decoded.signature.is_none());
    let repacked = decoded.manifest.pack(&decoded.bits, None).unwrap();
    assert_eq!(repacked, artifact, "format determinism pin");
}

fn wire_version(artifact: &mut [u8], version: u32) {
    artifact[5..9].copy_from_slice(&version.to_le_bytes());
}

fn wire_zero_code(artifact: &mut [u8], code: u8) {
    artifact[81] = code;
}

#[test]
fn refusal_bad_magic() {
    let mut artifact = golden_manifest().pack(&golden_bits(), None).unwrap();
    artifact[0] = b'X';
    assert!(matches!(
        decode(&artifact, None, None),
        Err(MaskCodecError::BadMagic(_))
    ));
}

#[test]
fn refusal_unknown_version() {
    let mut artifact = golden_manifest().pack(&golden_bits(), None).unwrap();
    wire_version(&mut artifact, FORMAT_VERSION + 1);
    assert!(matches!(
        decode(&artifact, None, None),
        Err(MaskCodecError::UnknownVersion { .. })
    ));
}

#[test]
fn refusal_wrong_base_before_payload_work() {
    let artifact = golden_manifest().pack(&golden_bits(), None).unwrap();
    let mut wrong = digest(b"another-model");
    wrong[0] ^= 0xFF;
    assert!(matches!(
        decode(&artifact, Some(&wrong), None),
        Err(MaskCodecError::BaseMismatch { .. })
    ));
    // and the payload blob is gone before the mismatch could matter:
    // truncate it away entirely — the base refusal must still fire.
    let beheaded = &artifact[..41];
    assert!(matches!(
        decode(beheaded, Some(&wrong), None),
        Err(MaskCodecError::BaseMismatch { .. })
    ));
}

#[test]
fn refusal_bad_zero_code() {
    let mut artifact = golden_manifest().pack(&golden_bits(), None).unwrap();
    wire_zero_code(&mut artifact, 2);
    assert!(matches!(
        decode(&artifact, None, None),
        Err(MaskCodecError::BadZeroCode { found: 2 })
    ));
}

#[test]
fn refusal_truncated_map_payload_signature() {
    let artifact = golden_manifest().pack(&golden_bits(), None).unwrap();
    // cut inside the section map (header ends at 90; map spans the
    // three entries after it)
    assert!(matches!(
        decode(&artifact[..95], None, None),
        Err(MaskCodecError::TruncatedMap)
    ));
    // cut inside the compressed payload
    assert!(matches!(
        decode(&artifact[..artifact.len() - 10], None, None),
        Err(MaskCodecError::TruncatedPayload)
    ));
}

#[test]
fn refusal_tampered_payload() {
    // splice a valid zstd of WRONG content into an otherwise valid
    // artifact: decompression succeeds, the BLAKE3 must catch it.
    let artifact = golden_manifest().pack(&golden_bits(), None).unwrap();
    // the RLE of an all-zero 112-BYTE mask (896 bits) is one (896, 0)
    // run = 9 bytes — same bit accounting, different content, so the
    // BLAKE3 (not the RLE walk) is what must catch it.
    let wrong_rle = {
        let mut r = Vec::new();
        r.extend_from_slice(&896u64.to_le_bytes());
        r.push(0);
        r
    };
    let wrong_compressed = zstd::stream::encode_all(&wrong_rle[..], 3).unwrap();

    let len_off = compressed_len_offset(&artifact);
    let stored_len =
        u64::from_le_bytes(artifact[len_off..len_off + 8].try_into().unwrap()) as usize;
    let payload_off = len_off + 8;

    let mut spliced = artifact.clone();
    spliced[len_off..len_off + 8].copy_from_slice(&(wrong_compressed.len() as u64).to_le_bytes());
    spliced.splice(payload_off..payload_off + stored_len, wrong_compressed);

    assert!(matches!(
        decode(&spliced, None, None),
        Err(MaskCodecError::PayloadTampered { .. })
    ));
}

#[test]
fn refusal_decode_side_router_name() {
    // craft an artifact whose section NAME is a router: same-length
    // in-place patch of the golden name keeps every cursor intact, so
    // the format-level router refusal is what fires.
    let artifact = golden_manifest().pack(&golden_bits(), None).unwrap();
    let original = b"blk.0.ffn_gate.weight";
    let patched = b"router.a.b.c.weight.1";
    assert_eq!(original.len(), patched.len());
    assert_eq!(patched.len(), 21);
    assert!(refused_tensor_name(std::str::from_utf8(patched).unwrap()));
    let off = find(artifact.as_slice(), original).expect("golden name in artifact");
    let mut patched_artifact = artifact.clone();
    patched_artifact[off..off + original.len()].copy_from_slice(patched);
    assert!(matches!(
        decode(&patched_artifact, None, None),
        Err(MaskCodecError::RouterTensorRefused(_))
    ));
}

fn compressed_len_offset(artifact: &[u8]) -> usize {
    // after magic(5)+version(4)+base(32)+payload_blake3(32)+raw_len(8)
    // +zero(1)+sig_len(4)+section_count(4) = 90, the map entries begin;
    // three entries, then the compressed_len u64.
    let mut off = 90usize;
    for _ in 0..3 {
        let name_len = u16::from_le_bytes(artifact[off..off + 2].try_into().unwrap()) as usize;
        off += 2 + name_len + 16;
    }
    off
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

#[test]
fn signature_roundtrip_verify_and_refuse() {
    let signing = SigningKey::from_bytes(&TEST_SIGNING_SEED);
    let verifying = VerifyingKey::from(&signing);
    let unsigned = golden_manifest().pack(&golden_bits(), None).unwrap();
    let signed = golden_manifest()
        .pack(&golden_bits(), Some(&signing))
        .unwrap();
    assert_eq!(unsigned.len() + 64, signed.len());

    let decoded = decode(&signed, None, Some(&verifying)).unwrap();
    assert!(decoded.signature.is_some());
    // trusted key + unsigned artifact → loud refusal
    assert!(matches!(
        decode(&unsigned, None, Some(&verifying)),
        Err(MaskCodecError::UnsignedArtifact)
    ));
    // flip one signature byte → verification failure
    let mut tampered_sig = signed.clone();
    let last = tampered_sig.len() - 1;
    tampered_sig[last] ^= 0x01;
    assert!(matches!(
        decode(&tampered_sig, None, Some(&verifying)),
        Err(MaskCodecError::BadSignature)
    ));
    // flip one payload byte → tamper refusal (hash fires first)
    let mut tampered_payload = signed.clone();
    tampered_payload[60] ^= 0x01;
    assert!(matches!(
        decode(&tampered_payload, None, Some(&verifying)),
        Err(MaskCodecError::PayloadTampered { .. })
    ));
    // a truncated signature field (sig_len 63 → malformed)
    let mut malformed = unsigned.clone();
    malformed[82..86].copy_from_slice(&63u32.to_le_bytes());
    assert!(matches!(
        decode(&malformed, None, None),
        Err(MaskCodecError::MalformedSignature(63))
    ));
}

#[test]
fn identity_and_wire_constants_pinned() {
    // wire constants the refusal battery depends on
    assert_eq!(MAGIC, *b"RIMSK");
    assert_eq!(FORMAT_VERSION, 1);
    assert_eq!(PQ2_0_ZERO_CODE, 1);
    // all-keep mask over the golden sections is identity
    let mut bits = vec![0xFFu8; 112];
    let manifest = golden_manifest();
    let decoded: DecodedMask = decode(&manifest.pack(&bits, None).unwrap(), None, None).unwrap();
    assert!(decoded.is_identity());
    bits[0] &= 0xFE;
    let decoded = decode(&manifest.pack(&bits, None).unwrap(), None, None).unwrap();
    assert!(!decoded.is_identity());
}
