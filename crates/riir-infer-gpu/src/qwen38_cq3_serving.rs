//! Issue 1186 T2b serving half — the CQ3 data plane between the artifact
//! and the T3 GPU lanes.
//!
//! Three pieces, one home (the format's single authority):
//! 1. [`PackedCq3Tensor`] — the b-bit packed representation the device
//!    uploads consume (whole `u64` words, LSB-first; the `u8` carrier the
//!    fixture lane holds is packed at load and never stored beside it).
//!    The packed-form host GEMV ([`PackedCq3Tensor::gemv_i8_sdot`]) is the
//!    GPU kernels' bit-exactness reference: identical loop order and
//!    arithmetic to [`CqTensor::gemv_i8_sdot`], only the index READ path
//!    differs (bits instead of the carrier), so the outputs are
//!    `to_bits`-equal by construction — the serving battery pins it.
//! 2. The `.cq3` container (CQ3ART1 v1) reader + writer — THE single
//!    format authority; the `cq3_quantize` bin writes through these. The
//!    field order below is the wire (never reordered without a format
//!    bump):
//!    `MAGIC[7] | pad[1] | format u32 | g u32 | b u32 | α bits u32 |
//!     reserved u32 | count u32` then per record
//!    `name_len u32 | name | d_out u32 | d_in u32 | smooth u8
//!     [+ d_in f32] | carrier_len u64 | carrier bytes | scales_len u64
//!     | scales u16`.
//!    The WIRE carries the `u8` carrier (v1, landed with the bin); the
//!    b-bit packing is the DEVICE representation, applied at load.
//! 3. [`verify_blake3_sidecar`] — the verify-then-use posture law for the
//!    T3 lane (the quantize bin's `<out>.blake3` sidecar).
//!
//! Rotation + codebook ride the `(g, b)` derivation (the seeded consts in
//! the recipe module — the g1f determinism pin): nothing but g/b/smooth/
//! indices/scales crosses any boundary, and the packed form re-derives
//! both on demand.
//!
//! Opt-in `cq3`, never in `production` before the T3 league cell's GOAT
//! gate (the env-switch law). This is the host-reference lane: deriving
//! the codebook per call is disclosed derivation-time cost — the GPU
//! lanes derive/upload once at load.

use std::io::Write;
use std::path::{Path, PathBuf};

use crate::qwen38_cq3::{Codebook, CqTensor, Rotation, SUPPORTED_G};

/// The container magic (`CQ3ART1`, 7 bytes — the bin's landed v1 value).
pub const ARTIFACT_MAGIC: &[u8; 7] = b"CQ3ART1";

/// The only container format this reader accepts (refuse-loud on anything
/// else — never a silent best-effort parse).
pub const ARTIFACT_FORMAT_V1: u32 = 1;

/// Pack one `b`-bit index stream (LSB-first, whole `u64` words, padded to
/// a word boundary — the device upload shape). Values must fit `b` bits.
#[must_use]
pub fn pack_indices_b(carrier: &[u8], b: u32) -> Vec<u64> {
    assert!((1..=8).contains(&b), "cq3 pack: b in 1..=8, got {b}");
    let bits = carrier.len() as u64 * u64::from(b);
    let n_words = bits.div_ceil(64) as usize;
    let mut words = vec![0_u64; n_words];
    for (i, &v) in carrier.iter().enumerate() {
        assert!(
            u64::from(v) < (1_u64 << b),
            "cq3 pack: carrier value {v} does not fit {b} bits"
        );
        let bit = i as u64 * u64::from(b);
        let w = (bit / 64) as usize;
        let off = (bit % 64) as u32;
        words[w] |= u64::from(v) << off;
        if off + b > 64 {
            words[w + 1] |= u64::from(v) >> (64 - off);
        }
    }
    words
}

/// Read one `b`-bit index at weight position `pos` (the GEMV inner read).
#[must_use]
pub fn read_index_b(words: &[u64], pos: usize, b: u32) -> u8 {
    debug_assert!((1..=8).contains(&b), "cq3 read: b in 1..=8, got {b}");
    let bit = pos as u64 * u64::from(b);
    let w = (bit / 64) as usize;
    let off = (bit % 64) as u32;
    let mask = (1_u64 << b) - 1;
    let lo = words[w] >> off;
    // A weight crossing a word boundary pulls the low bits of the next
    // word; `off > 0` whenever this arm fires, so the shift is < 64.
    let v = if off + b > 64 {
        lo | (words[w + 1] << (64 - off))
    } else {
        lo
    };
    (v & mask) as u8
}

/// Unpack a packed stream back to the `u8` carrier (`n` weights).
#[must_use]
pub fn unpack_indices_b(words: &[u64], n: usize, b: u32) -> Vec<u8> {
    (0..n).map(|i| read_index_b(words, i, b)).collect()
}

/// One quantized weight matrix in the packed (device-shaped) form: `b`-bit
/// indices + FP16 group norms + the SmoothQuant fold. Rotation + codebook
/// are NOT stored — they derive deterministically from `(g, b)`.
#[derive(Clone, Debug)]
pub struct PackedCq3Tensor {
    pub d_out: usize,
    pub d_in: usize,
    pub g: usize,
    pub b: u32,
    /// SmoothQuant fold (per input channel); `None` = unfused. The GEMV
    /// divides the activation by this before rotating (`x' = x/s`).
    pub smooth: Option<Vec<f32>>,
    /// FP16 group-norm bits, `d_out · (d_in / g)` entries.
    pub scales: Vec<u16>,
    /// `b`-bit packed indices, LSB-first whole `u64` words.
    pub packed: Vec<u64>,
}

impl PackedCq3Tensor {
    /// Pack from parts (the reader's entry — it validates before packing,
    /// so these asserts are internal invariants, never hostile input).
    ///
    /// # Panics
    /// On a shape/bit-width inconsistency (the caller validates first).
    #[must_use]
    pub fn from_parts(
        d_out: usize,
        d_in: usize,
        g: usize,
        b: u32,
        smooth: Option<&[f32]>,
        carrier: &[u8],
        scales: &[u16],
    ) -> Self {
        assert_eq!(carrier.len(), d_out * d_in, "cq3 pack: carrier len");
        assert_eq!(scales.len(), d_out * (d_in / g), "cq3 pack: scales len");
        assert_eq!(d_in % g, 0, "cq3 pack: d_in must be a multiple of g");
        assert!(
            SUPPORTED_G.contains(&g),
            "cq3 pack: g must be one of {SUPPORTED_G:?}, got {g}"
        );
        assert!((1..=8).contains(&b), "cq3 pack: b in 1..=8, got {b}");
        if let Some(s) = smooth {
            assert_eq!(s.len(), d_in, "cq3 pack: smooth len");
        }
        Self {
            d_out,
            d_in,
            g,
            b,
            smooth: smooth.map(<[f32]>::to_vec),
            scales: scales.to_vec(),
            packed: pack_indices_b(carrier, b),
        }
    }

    /// Pack from the fixture-lane recipe type (the carrier is consumed —
    /// the packed form never stores it beside the words).
    #[must_use]
    pub fn from_cq_tensor(t: &CqTensor) -> Self {
        Self::from_parts(
            t.d_out,
            t.d_in,
            t.g,
            t.b,
            t.smooth.as_deref(),
            &t.indices,
            &t.scales,
        )
    }

    /// The fixed structured rotation for this tensor's `g` (derivation).
    #[must_use]
    pub fn rotation(&self) -> Rotation {
        Rotation::fixed(self.g)
    }

    /// The scalar Lloyd–Max codebook for this tensor's `(g, b)`
    /// (derivation). Derivation-time cost — the GPU lanes upload the
    /// derived `c8`/`s_c` once at load.
    #[must_use]
    pub fn codebook(&self) -> Codebook {
        Codebook::build(self.g, self.b)
    }

    /// Back to the `u8` carrier (round-trip half; the container writer
    /// unpacks per record at write time).
    #[must_use]
    pub fn unpack_indices(&self) -> Vec<u8> {
        unpack_indices_b(&self.packed, self.d_out * self.d_in, self.b)
    }

    /// Rotate (and optionally SmoothQuant-divide) the activation — the
    /// whole activation-side work, done ONCE per vector. Arithmetic twins
    /// [`CqTensor::rotate_activation`].
    #[must_use]
    pub fn rotate_activation(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.d_in, "cq3 rotate_activation: len mismatch");
        let mut divided = Vec::with_capacity(self.d_in);
        match &self.smooth {
            Some(s) => divided.extend(x.iter().zip(s.iter()).map(|(&v, &s)| v / s)),
            None => divided.extend_from_slice(x),
        }
        let mut out = Vec::with_capacity(self.d_in);
        self.rotation().apply(&divided, &mut out);
        out
    }

    /// The int8 SDOT-shaped GEMV over the PACKED indices — the GPU
    /// kernels' bit-exactness reference. Identical loop order, i32
    /// accumulation and per-group de-scale (`f32(acc)·s_c·s_x·f16(norm)`,
    /// that fixed left-associated order) to [`CqTensor::gemv_i8_sdot`];
    /// only the index read goes through the bits, so the outputs are
    /// `to_bits`-equal to the carrier form (the battery pins it).
    ///
    /// The activation quantization is staged ONCE per call (it does not
    /// depend on the output row), matching how a real kernel front-loads
    /// it.
    #[must_use]
    pub fn gemv_i8_sdot(&self, x_rot: &[f32]) -> Vec<f32> {
        assert_eq!(x_rot.len(), self.d_in, "cq3 gemv_i8_sdot: len mismatch");
        debug_assert!(
            self.d_in * 127 * 127 < i32::MAX as usize,
            "cq3 gemv_i8_sdot: per-row i32 bound d_in·127² exceeded"
        );
        let g = self.g;
        let groups_in = self.d_in / g;
        let cb = self.codebook();
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
                    let idx = read_index_b(&self.packed, base + m, self.b) as usize;
                    acc += i32::from(cb.c8[idx]) * x8[grp * g + m];
                }
                y += acc as f32 * cb.s_c * s_x[grp] * gf;
            }
            *out_i = y;
        }
        out
    }

    /// The packed bits-per-weight: `b + 16/g` (one FP16 scale per group).
    /// At the pre-registered `g = 32, b = 3` this is the 3.5 that matches
    /// Q3_K's ~3.44 from the heavy side.
    #[must_use]
    pub fn bits_per_weight(&self) -> f64 {
        let n = (self.d_out * self.d_in) as f64;
        (n * f64::from(self.b) + (self.scales.len() as f64) * 16.0) / n
    }
}

/// One named tensor record in a `.cq3` container.
#[derive(Clone, Debug)]
pub struct Cq3ArtifactTensor {
    pub name: String,
    pub tensor: PackedCq3Tensor,
}

/// A parsed `.cq3` container (the T3 serving lane's load-time entry).
#[derive(Clone, Debug)]
pub struct Cq3Artifact {
    pub g: usize,
    pub b: u32,
    pub alpha: f32,
    pub format: u32,
    pub tensors: Vec<Cq3ArtifactTensor>,
}

/// Write the 32-byte container header. Field order IS the wire.
///
/// # Errors
/// Propagates the underlying `std::io` error.
pub fn write_header<W: Write>(
    w: &mut W,
    g: usize,
    b: u32,
    alpha: f32,
    count: usize,
) -> std::io::Result<()> {
    w.write_all(ARTIFACT_MAGIC)?;
    w.write_all(&[0_u8; 1])?;
    w.write_all(&ARTIFACT_FORMAT_V1.to_le_bytes())?;
    w.write_all(&(g as u32).to_le_bytes())?;
    w.write_all(&b.to_le_bytes())?;
    w.write_all(&alpha.to_bits().to_le_bytes())?;
    w.write_all(&0_u32.to_le_bytes())?; // reserved
    w.write_all(&(count as u32).to_le_bytes())
}

/// Write one tensor record (the carrier is re-derived from the packed
/// words — the record's `carrier_len` counts the unpacked bytes).
///
/// # Errors
/// Propagates the underlying `std::io` error.
pub fn write_tensor_record<W: Write>(
    w: &mut W,
    name: &str,
    t: &PackedCq3Tensor,
) -> std::io::Result<()> {
    let carrier = t.unpack_indices();
    w.write_all(&(name.len() as u32).to_le_bytes())?;
    w.write_all(name.as_bytes())?;
    w.write_all(&(t.d_out as u32).to_le_bytes())?;
    w.write_all(&(t.d_in as u32).to_le_bytes())?;
    match &t.smooth {
        Some(s) => {
            w.write_all(&[1_u8])?;
            for v in s {
                w.write_all(&v.to_le_bytes())?;
            }
        }
        None => w.write_all(&[0_u8])?,
    }
    w.write_all(&(carrier.len() as u64).to_le_bytes())?;
    w.write_all(&carrier)?;
    w.write_all(&(t.scales.len() as u64).to_le_bytes())?;
    for s in &t.scales {
        w.write_all(&s.to_le_bytes())?;
    }
    Ok(())
}

/// Whole-file convenience writer (fixtures + tools). The quantize bin
/// streams through [`write_header`]/[`write_tensor_record`] instead (its
/// progress prints + fsync + sidecar stay its own).
///
/// # Errors
/// Names the failed phase on any I/O error.
pub fn write_artifact(path: &Path, a: &Cq3Artifact) -> Result<(), String> {
    let file = std::fs::File::create(path).map_err(|e| format!("out: {e}"))?;
    let mut w = std::io::BufWriter::with_capacity(1 << 22, file);
    write_header(&mut w, a.g, a.b, a.alpha, a.tensors.len())
        .map_err(|e| format!("write header: {e}"))?;
    for t in &a.tensors {
        write_tensor_record(&mut w, &t.name, &t.tensor)
            .map_err(|e| format!("write {}: {e}", t.name))?;
    }
    w.flush().map_err(|e| format!("flush: {e}"))?;
    let file = w.into_inner().map_err(|e| format!("into_inner: {e}"))?;
    file.sync_all().map_err(|e| format!("fsync: {e}"))?;
    Ok(())
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Verify the quantize bin's `<path>.blake3` sidecar against the file's
/// actual BLAKE3 digest — the verify-then-use posture law (T3 refuses a
/// drifted artifact by name, never loads it). Returns the actual digest.
///
/// # Errors
/// Names the sidecar, the I/O phase, or both digests on a mismatch.
pub fn verify_blake3_sidecar(path: &Path) -> Result<String, String> {
    let sidecar = PathBuf::from(format!("{}.blake3", path.display()));
    let text =
        std::fs::read_to_string(&sidecar).map_err(|e| format!("{}: {e}", sidecar.display()))?;
    let recorded = text
        .split_whitespace()
        .next()
        .ok_or_else(|| format!("{}: empty sidecar", sidecar.display()))?
        .to_ascii_lowercase();
    let mut file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut hasher = blake3::Hasher::new();
    std::io::copy(&mut file, &mut hasher)
        .map_err(|e| format!("blake3 {}: read: {e}", path.display()))?;
    let actual = hex_encode(hasher.finalize().as_bytes());
    if recorded != actual {
        return Err(format!(
            "{}: blake3 mismatch (sidecar {recorded}, actual {actual})",
            path.display()
        ));
    }
    Ok(actual)
}

impl Cq3Artifact {
    /// Parse a `.cq3` container (v1 only). Refuses loud — bad magic,
    /// wrong format, truncated records, out-of-range recipe params,
    /// length inconsistencies, trailing bytes — never a silent
    /// best-effort parse (the verify-then-use posture law).
    ///
    /// # Errors
    /// Names the file and the offending field.
    pub fn read(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if bytes.len() < 32 {
            return Err(format!(
                "{}: truncated header ({} < 32 bytes)",
                path.display(),
                bytes.len()
            ));
        }
        if &bytes[..7] != ARTIFACT_MAGIC {
            return Err(format!("{}: bad magic", path.display()));
        }
        let fmt = u32::from_le_bytes(bytes[8..12].try_into().expect("32-byte header"));
        if fmt != ARTIFACT_FORMAT_V1 {
            return Err(format!("{}: format {fmt} != v1", path.display()));
        }
        let g = u32::from_le_bytes(bytes[12..16].try_into().expect("header")) as usize;
        if !SUPPORTED_G.contains(&g) {
            return Err(format!("{}: g {g} not in {SUPPORTED_G:?}", path.display()));
        }
        let b = u32::from_le_bytes(bytes[16..20].try_into().expect("header"));
        if !(1..=8).contains(&b) {
            return Err(format!("{}: b {b} not in 1..=8", path.display()));
        }
        let alpha = f32::from_bits(u32::from_le_bytes(
            bytes[20..24].try_into().expect("header"),
        ));
        let count = u32::from_le_bytes(bytes[28..32].try_into().expect("header")) as usize;
        let mut pos = 32usize;
        let mut tensors = Vec::with_capacity(count);
        for rec in 0..count {
            if pos + 4 > bytes.len() {
                return Err(format!(
                    "{}: record {rec}: truncated name_len",
                    path.display()
                ));
            }
            let name_len =
                u32::from_le_bytes(bytes[pos..pos + 4].try_into().expect("4 bytes")) as usize;
            if name_len == 0 || name_len > 4096 {
                return Err(format!(
                    "{}: record {rec}: implausible name_len {name_len}",
                    path.display()
                ));
            }
            pos += 4;
            if pos + name_len > bytes.len() {
                return Err(format!("{}: record {rec}: truncated name", path.display()));
            }
            let name = std::str::from_utf8(&bytes[pos..pos + name_len])
                .map_err(|_| format!("{}: record {rec}: name is not UTF-8", path.display()))?
                .to_string();
            pos += name_len;
            let take_u32 = |pos: &mut usize, what: &str| -> Result<u32, String> {
                if *pos + 4 > bytes.len() {
                    return Err(format!(
                        "{}: record {rec}: truncated {what}",
                        path.display()
                    ));
                }
                let v = u32::from_le_bytes(bytes[*pos..*pos + 4].try_into().expect("4 bytes"));
                *pos += 4;
                Ok(v)
            };
            let d_out = take_u32(&mut pos, "d_out")? as usize;
            let d_in = take_u32(&mut pos, "d_in")? as usize;
            if !d_in.is_multiple_of(g) {
                return Err(format!(
                    "{}: {name}: d_in {d_in} not a multiple of g {g}",
                    path.display()
                ));
            }
            if pos >= bytes.len() {
                return Err(format!("{}: {name}: truncated smooth flag", path.display()));
            }
            let smooth_flag = bytes[pos];
            pos += 1;
            let smooth = match smooth_flag {
                0 => None,
                1 => {
                    let n = d_in * 4;
                    if pos + n > bytes.len() {
                        return Err(format!("{}: {name}: truncated smooth", path.display()));
                    }
                    let v: Vec<f32> = bytes[pos..pos + n]
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| f32::from_le_bytes(*c))
                        .collect();
                    pos += n;
                    Some(v)
                }
                other => {
                    return Err(format!(
                        "{}: {name}: smooth flag {other} is neither 0 nor 1",
                        path.display()
                    ));
                }
            };
            let take_len = |pos: &mut usize, what: &str| -> Result<u64, String> {
                if *pos + 8 > bytes.len() {
                    return Err(format!("{}: {name}: truncated {what} len", path.display()));
                }
                let v = u64::from_le_bytes(bytes[*pos..*pos + 8].try_into().expect("8 bytes"));
                *pos += 8;
                Ok(v)
            };
            let carrier_len = take_len(&mut pos, "carrier")? as usize;
            if carrier_len != d_out * d_in {
                return Err(format!(
                    "{}: {name}: carrier_len {carrier_len} != d_out·d_in {}",
                    path.display(),
                    d_out * d_in
                ));
            }
            if pos + carrier_len > bytes.len() {
                return Err(format!("{}: {name}: truncated carrier", path.display()));
            }
            let carrier = bytes[pos..pos + carrier_len].to_vec();
            pos += carrier_len;
            let scales_len = take_len(&mut pos, "scales")? as usize;
            let expect_scales = d_out * (d_in / g);
            if scales_len != expect_scales {
                return Err(format!(
                    "{}: {name}: scales_len {scales_len} != {expect_scales}",
                    path.display()
                ));
            }
            if pos + scales_len * 2 > bytes.len() {
                return Err(format!("{}: {name}: truncated scales", path.display()));
            }
            let scales: Vec<u16> = bytes[pos..pos + scales_len * 2]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| u16::from_le_bytes(*c))
                .collect();
            pos += scales_len * 2;
            let tensor = PackedCq3Tensor::from_parts(
                d_out,
                d_in,
                g,
                b,
                smooth.as_deref(),
                &carrier,
                &scales,
            );
            tensors.push(Cq3ArtifactTensor { name, tensor });
        }
        if pos != bytes.len() {
            return Err(format!(
                "{}: {} trailing bytes after the last record",
                path.display(),
                bytes.len() - pos
            ));
        }
        Ok(Self {
            g,
            b,
            alpha,
            format: fmt,
            tensors,
        })
    }

    /// The named tensor (T3's per-layer lookup; refuses by name).
    ///
    /// # Errors
    /// Names the missing tensor and the artifact.
    pub fn tensor(&self, name: &str) -> Result<&PackedCq3Tensor, String> {
        self.tensors
            .iter()
            .find(|t| t.name == name)
            .map(|t| &t.tensor)
            .ok_or_else(|| format!("artifact: tensor {name} not present"))
    }
}
