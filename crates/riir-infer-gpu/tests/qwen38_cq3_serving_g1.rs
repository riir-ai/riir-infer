#![cfg(feature = "cq3")]
//! Issue 1186 T2b serving half — the packed data plane + the `.cq3`
//! container fixture battery (hermetic, no GPU, no model):
//!
//! - **s1 pack round trip** — carrier → packed words → carrier is
//!   bit-identical at every supported bit width, and the pack is
//!   word-deterministic (re-pack of the unpacked carrier reproduces the
//!   words exactly).
//! - **s2 packed GEMV bit-exactness** — the packed-form
//!   [`PackedCq3Tensor::gemv_i8_sdot`] is `to_bits`-equal to the
//!   carrier-form [`CqTensor::gemv_i8_sdot`] across g/b/smooth arms —
//!   the GPU kernels' reference contract.
//! - **s3 rotate_activation twin** — the packed form's activation path
//!   (smooth divide + blockwise rotation) is `to_bits`-equal to the
//!   recipe module's.
//! - **s4 container round trip** — write (the module's writers, the
//!   single format authority) → read: names, dims, smooth, scales and
//!   packed words all identical; the reader refuses truncation, bad
//!   magic, a trailing-byte tail and a bogus format loud, by name.
//! - **s5 sidecar verify** — the blake3 sidecar verifies on a clean
//!   artifact and refuses a tampered byte by name (the recorded digest
//!   in the error).
//! - **s6 bpw pin** — `bits_per_weight` = b + 16/g exactly at the
//!   pre-registered g=32/b=3 (3.5).

use riir_infer_gpu::qwen38_cq3::{CqTensor, Xorshift64Star};
use riir_infer_gpu::qwen38_cq3_serving::{
    ARTIFACT_FORMAT_V1, ARTIFACT_MAGIC, Cq3Artifact, Cq3ArtifactTensor, PackedCq3Tensor,
    pack_indices_b, read_index_b, unpack_indices_b, verify_blake3_sidecar, write_artifact,
};

/// Seeded deterministic weights (the fixture-lane RNG; never a global).
fn seeded_weights(seed: u64, n: usize) -> Vec<f32> {
    let mut rng = Xorshift64Star::new(seed);
    (0..n).map(|_| rng.next_normal()).collect()
}

/// Deterministic per-arm seed (arms stay independent without a global).
fn arm_seed(g: usize, b: u32, with_smooth: bool) -> u64 {
    0x1186_0002 ^ ((g as u64) << 8) ^ (u64::from(b) << 4) ^ u64::from(with_smooth)
}

fn tmpdir(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!(
        "i1186_serving_{tag}_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    std::fs::create_dir_all(&d).expect("mkdir");
    d
}

fn synth_tensor(seed: u64, g: usize, b: u32, d_out: usize, smooth: Option<&[f32]>) -> CqTensor {
    let d_in = 2 * g;
    let w = seeded_weights(seed, d_out * d_in);
    CqTensor::quantize(&w, d_out, d_in, g, b, smooth)
}

#[test]
fn s1_pack_round_trip_bit_exact() {
    for b in [1_u32, 3, 4, 7, 8] {
        let g = 32_usize;
        let d_out = 4_usize;
        let d_in = 2 * g;
        // Carrier values deliberately span the whole b-bit domain.
        let carrier: Vec<u8> = (0..d_out * d_in)
            .map(|i| ((i * 37 + 11) % (1_usize << b)) as u8)
            .collect();
        let words = pack_indices_b(&carrier, b);
        assert_eq!(
            words.len(),
            (carrier.len() as u64 * u64::from(b)).div_ceil(64) as usize,
            "b{b}: word count"
        );
        let back = unpack_indices_b(&words, carrier.len(), b);
        assert_eq!(back, carrier, "b{b}: round trip");
        let repacked = pack_indices_b(&back, b);
        assert_eq!(repacked, words, "b{b}: re-pack determinism");
        // Single-index reads agree with the bulk unpack at spot positions.
        for pos in [0_usize, 1, 17, d_out * d_in / 2, d_out * d_in - 1] {
            assert_eq!(
                read_index_b(&words, pos, b),
                carrier[pos],
                "b{b}: spot read at {pos}"
            );
        }
    }
}

#[test]
fn s2_packed_gemv_matches_carrier_i8_sdot() {
    for g in [16_usize, 32] {
        for b in [3_u32, 4] {
            for with_smooth in [false, true] {
                let d_out = 6_usize;
                let d_in = 2 * g;
                let smooth_vec = with_smooth.then(|| {
                    (0..d_in)
                        .map(|i| 1.25_f32 + 0.125 * (i % 5) as f32)
                        .collect::<Vec<_>>()
                });
                let t = synth_tensor(
                    arm_seed(g, b, with_smooth),
                    g,
                    b,
                    d_out,
                    smooth_vec.as_deref(),
                );
                let packed = PackedCq3Tensor::from_cq_tensor(&t);
                let mut rng = Xorshift64Star::new(0x5EED_0002 ^ arm_seed(g, b, with_smooth));
                let x: Vec<f32> = (0..t.d_in).map(|_| rng.next_normal()).collect();
                let x_rot = t.rotate_activation(&x);
                let want = t.gemv_i8_sdot(&x_rot);
                let got = packed.gemv_i8_sdot(&x_rot);
                assert_eq!(got.len(), want.len(), "g{g} b{b}: len");
                for (i, (a, e)) in got.iter().zip(want.iter()).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        e.to_bits(),
                        "g{g} b{b} smooth={with_smooth}: row {i} diverged ({a} vs {e})"
                    );
                }
            }
        }
    }
}

#[test]
fn s3_rotate_activation_twin() {
    for g in [16_usize, 32, 64] {
        let d_out = 3_usize;
        let smooth: Vec<f32> = (0..2 * g)
            .map(|i| 1.0_f32 / (1.0 + 0.25 * (i % 3) as f32))
            .collect();
        let t = synth_tensor(0x1186_0003 ^ (g as u64), g, 3, d_out, Some(&smooth));
        let packed = PackedCq3Tensor::from_cq_tensor(&t);
        let mut rng = Xorshift64Star::new(0x5EED_0003 ^ (g as u64));
        let x: Vec<f32> = (0..t.d_in).map(|_| rng.next_normal()).collect();
        let want = t.rotate_activation(&x);
        let got = packed.rotate_activation(&x);
        for (a, e) in got.iter().zip(want.iter()) {
            assert_eq!(a.to_bits(), e.to_bits(), "g{g}: rotate diverged");
        }
        // Unfused arm too.
        let t0 = synth_tensor(0x1186_0004 ^ (g as u64), g, 3, d_out, None);
        let p0 = PackedCq3Tensor::from_cq_tensor(&t0);
        let want0 = t0.rotate_activation(&x);
        let got0 = p0.rotate_activation(&x);
        for (a, e) in got0.iter().zip(want0.iter()) {
            assert_eq!(a.to_bits(), e.to_bits(), "g{g}: unfused rotate diverged");
        }
    }
}

#[test]
fn s4_container_round_trip_and_refusals() {
    let dir = tmpdir("container");
    let path = dir.join("rt.cq3");

    let t1 = synth_tensor(0x1186_0005, 32, 3, 4, None);
    let t2_smooth: Vec<f32> = (0..64).map(|i| 0.5_f32 + 0.25 * (i % 4) as f32).collect();
    let t2 = synth_tensor(0x1186_0006, 32, 3, 5, Some(&t2_smooth));
    let artifact = Cq3Artifact {
        g: 32,
        b: 3,
        alpha: 0.25,
        format: ARTIFACT_FORMAT_V1,
        tensors: vec![
            Cq3ArtifactTensor {
                name: "blk.0.ffn_gate.weight".into(),
                tensor: PackedCq3Tensor::from_cq_tensor(&t1),
            },
            Cq3ArtifactTensor {
                name: "blk.1.ffn_down.weight".into(),
                tensor: PackedCq3Tensor::from_cq_tensor(&t2),
            },
        ],
    };
    write_artifact(&path, &artifact).expect("write");

    let bytes = std::fs::read(&path).expect("read back");
    assert_eq!(&bytes[..7], ARTIFACT_MAGIC, "magic");
    assert!(bytes.len() > 32, "records present after the 32-byte header");

    let parsed = Cq3Artifact::read(&path).expect("read parses");
    assert_eq!(parsed.g, 32);
    assert_eq!(parsed.b, 3);
    assert_eq!(parsed.alpha.to_bits(), 0.25_f32.to_bits());
    assert_eq!(parsed.tensors.len(), 2);
    for (orig, back) in artifact.tensors.iter().zip(parsed.tensors.iter()) {
        assert_eq!(orig.name, back.name);
        assert_eq!(orig.tensor.d_out, back.tensor.d_out);
        assert_eq!(orig.tensor.d_in, back.tensor.d_in);
        assert_eq!(
            orig.tensor.packed, back.tensor.packed,
            "{}: words",
            orig.name
        );
        assert_eq!(
            orig.tensor.scales, back.tensor.scales,
            "{}: scales",
            orig.name
        );
        match (&orig.tensor.smooth, &back.tensor.smooth) {
            (None, None) => {}
            (Some(a), Some(b_)) => {
                assert_eq!(a.len(), b_.len());
                for (va, vb) in a.iter().zip(b_.iter()) {
                    assert_eq!(va.to_bits(), vb.to_bits());
                }
            }
            _ => panic!("{}: smooth presence mismatch", orig.name),
        }
    }
    let lookup = parsed.tensor("blk.1.ffn_down.weight").expect("lookup");
    assert_eq!(lookup.packed, artifact.tensors[1].tensor.packed);
    assert!(
        parsed.tensor("blk.9.missing.weight").is_err(),
        "missing refuses"
    );

    // Refusal battery: truncated, bad magic, bogus format, trailing bytes.
    let trunc = dir.join("trunc.cq3");
    std::fs::write(&trunc, &bytes[..bytes.len() - 3]).expect("write trunc");
    let err = Cq3Artifact::read(&trunc).expect_err("truncated refuses");
    assert!(err.contains("truncated"), "err: {err}");

    let mut bad = bytes.clone();
    bad[0] = b'X';
    let badmagic = dir.join("badmagic.cq3");
    std::fs::write(&badmagic, &bad).expect("write badmagic");
    let err = Cq3Artifact::read(&badmagic).expect_err("bad magic refuses");
    assert!(err.contains("bad magic"), "err: {err}");

    let mut fmt_bytes = bytes.clone();
    fmt_bytes[8] = 9; // format field's little-endian first byte
    let badfmt = dir.join("badfmt.cq3");
    std::fs::write(&badfmt, &fmt_bytes).expect("write badfmt");
    let err = Cq3Artifact::read(&badfmt).expect_err("bogus format refuses");
    assert!(err.contains("format 9"), "err: {err}");

    let mut trail = bytes;
    trail.push(0_u8);
    let trailp = dir.join("trail.cq3");
    std::fs::write(&trailp, &trail).expect("write trail");
    let err = Cq3Artifact::read(&trailp).expect_err("trailing bytes refuse");
    assert!(err.contains("trailing bytes"), "err: {err}");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn s5_sidecar_verify_clean_and_tampered() {
    let dir = tmpdir("sidecar");
    let path = dir.join("sc.cq3");
    let t = synth_tensor(0x1186_0007, 32, 3, 4, None);
    let artifact = Cq3Artifact {
        g: 32,
        b: 3,
        alpha: 0.25,
        format: ARTIFACT_FORMAT_V1,
        tensors: vec![Cq3ArtifactTensor {
            name: "blk.0.ffn_gate.weight".into(),
            tensor: PackedCq3Tensor::from_cq_tensor(&t),
        }],
    };
    write_artifact(&path, &artifact).expect("write");
    // Sidecar in the quantize bin's format: "<hex>  <path>\n".
    let digest = {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&std::fs::read(&path).expect("read"));
        let mut s = String::new();
        for byte in hasher.finalize().as_bytes() {
            s.push_str(&format!("{byte:02x}"));
        }
        s
    };
    std::fs::write(
        dir.join("sc.cq3.blake3"),
        format!("{digest}  {}\n", path.display()),
    )
    .expect("sidecar");
    let got = verify_blake3_sidecar(&path).expect("clean verifies");
    assert_eq!(got, digest);

    // Tamper one payload byte — the verify refuses with the recorded digest.
    let mut bytes = std::fs::read(&path).expect("read");
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0x01;
    std::fs::write(&path, &bytes).expect("tamper");
    let err = verify_blake3_sidecar(&path).expect_err("tampered refuses");
    assert!(err.contains("blake3 mismatch"), "err: {err}");
    assert!(
        err.contains(&digest),
        "err carries the recorded digest: {err}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn s6_bits_per_weight_pin() {
    let t = synth_tensor(0x1186_0008, 32, 3, 4, None);
    let p = PackedCq3Tensor::from_cq_tensor(&t);
    let bpw = p.bits_per_weight();
    let want = 3.0_f64 + 16.0 / 32.0;
    assert!(
        (bpw - want).abs() < 1e-9,
        "bpw {bpw} != the pre-registered {want}"
    );
    // The scales geometry the formula counts: one FP16 per (row, group).
    assert_eq!(p.scales.len(), t.d_out * (t.d_in / t.g));
}
