//! The artifact loader (Issue 1214 T1c, Rust half) — reads the
//! prepared integer artifact (upstream `prepare.py` @ `9c3f1c9c` layout:
//! `model.safetensors` of int tensors + `config.json` of scalar
//! constants) into an [`IntRefArtifact`], blake3-digesting both files
//! for the pin.
//!
//! The parse is INT-Typed only: I8/I16/I32/I64 decode little-endian;
//! every other dtype — floats first — refuses loud (the mirror image of
//! upstream's own "float tensor in artifact" refusal at load). The
//! float lane's reader (`super::super::weights`) widens to f32 and
//! pins the F16/Q8 checkpoint family — a different container by
//! contract, deliberately not reused.
//!
//! Tensor names follow upstream `prepare.py` verbatim (`layers.{i}.q.w8t`,
//! `layers.{i}.kq_m`, `rope.cos`, ...); an UNKNOWN name refuses (the
//! typo detector — upstream writes exactly this set). Scalar constants
//! ride `config.json` (`n_layers`, per-layer `q.k`/`as_k`/`qs_k`/...);
//! semantic shape law stays with [`IntRefModel::new`].

use super::model::{IntRefArtifact, IntRefConfig, LayerWeights, LinearWeights};
use super::ops::{ROPE_FRAC_BITS, RopeTables};

/// One decoded safetensors tensor: dtype + shape + the decoded payload.
#[derive(Clone, Debug)]
pub struct StTensor {
    pub dtype: StDtype,
    pub shape: Vec<usize>,
    pub data: StData,
}

/// The integer dtypes the lane admits (safetensors spellings).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StDtype {
    I8,
    I16,
    I32,
    I64,
}

impl StDtype {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "I8" => Some(StDtype::I8),
            "I16" => Some(StDtype::I16),
            "I32" => Some(StDtype::I32),
            "I64" => Some(StDtype::I64),
            _ => None,
        }
    }

    fn width(self) -> usize {
        match self {
            StDtype::I8 => 1,
            StDtype::I16 => 2,
            StDtype::I32 => 4,
            StDtype::I64 => 8,
        }
    }
}

/// The decoded payload, one variant per admitted dtype.
#[derive(Clone, Debug)]
pub enum StData {
    I8(Vec<i8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
}

impl StData {
    fn decode(dtype: StDtype, bytes: &[u8]) -> Self {
        match dtype {
            StDtype::I8 => StData::I8(bytes.iter().map(|&b| b as i8).collect()),
            StDtype::I16 => StData::I16(
                bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| i16::from_le_bytes([c[0], c[1]]))
                    .collect(),
            ),
            StDtype::I32 => StData::I32(
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            ),
            StDtype::I64 => StData::I64(
                bytes
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                    .collect(),
            ),
        }
    }
}

/// A parsed container: tensor name → decoded tensor.
#[derive(Debug)]
pub struct StMap {
    tensors: Vec<(String, StTensor)>,
}

impl StMap {
    pub fn get(&self, name: &str) -> Option<&StTensor> {
        self.tensors.iter().find(|(n, _)| n == name).map(|(_, t)| t)
    }

    fn require(&self, name: &str) -> Result<&StTensor, String> {
        self.get(name)
            .ok_or_else(|| format!("artifact: missing tensor {name}"))
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.iter().map(|(n, _)| n.as_str())
    }

    fn expect_i64(&self, name: &str, want: usize) -> Result<Vec<i64>, String> {
        let t = self.require(name)?;
        match &t.data {
            StData::I64(v) => {
                if v.len() != want {
                    return Err(format!(
                        "artifact: {name} has {} elements, want {want}",
                        v.len()
                    ));
                }
                Ok(v.clone())
            }
            _ => Err(format!("artifact: {name} must be I64")),
        }
    }

    fn expect_i8(&self, name: &str, want: usize) -> Result<Vec<i8>, String> {
        let t = self.require(name)?;
        match &t.data {
            StData::I8(v) => {
                if v.len() != want {
                    return Err(format!(
                        "artifact: {name} has {} elements, want {want}",
                        v.len()
                    ));
                }
                Ok(v.clone())
            }
            _ => Err(format!("artifact: {name} must be I8")),
        }
    }

    fn expect_i16(&self, name: &str, want: usize) -> Result<Vec<i16>, String> {
        let t = self.require(name)?;
        match &t.data {
            StData::I16(v) => {
                if v.len() != want {
                    return Err(format!(
                        "artifact: {name} has {} elements, want {want}",
                        v.len()
                    ));
                }
                Ok(v.clone())
            }
            _ => Err(format!("artifact: {name} must be I16")),
        }
    }
}

/// Parse a safetensors container: 8-byte LE header length + JSON header
/// (`{name: {dtype, shape, data_offsets}}`, offsets relative to the data
/// section at `8 + header_len`) + raw data. Layout per the safetensors
/// spec, as documented on the float lane's own reader.
pub fn parse_container(bytes: &[u8]) -> Result<StMap, String> {
    if bytes.len() < 8 {
        return Err(format!(
            "artifact: container too short ({} bytes)",
            bytes.len()
        ));
    }
    let mut len_b = [0_u8; 8];
    len_b.copy_from_slice(&bytes[..8]);
    let header_len = u64::from_le_bytes(len_b) as usize;
    if header_len > bytes.len() - 8 {
        return Err(format!(
            "artifact: header length {header_len} past the file ({} bytes)",
            bytes.len()
        ));
    }
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + header_len])
        .map_err(|e| format!("artifact: header JSON: {e}"))?;
    let obj = header
        .as_object()
        .ok_or_else(|| "artifact: header must be a JSON object".to_string())?;
    let data_start = 8 + header_len;
    let mut tensors = Vec::with_capacity(obj.len());
    for (name, entry) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype_s = entry["dtype"]
            .as_str()
            .ok_or_else(|| format!("artifact: {name}: missing dtype"))?;
        let dtype = StDtype::from_str(dtype_s).ok_or_else(|| {
            format!("artifact: {name}: dtype {dtype_s} is not an int tensor (floats refuse at the lane door)")
        })?;
        let shape = entry["shape"]
            .as_array()
            .ok_or_else(|| format!("artifact: {name}: missing shape"))?
            .iter()
            .map(|v| {
                v.as_u64()
                    .map(|x| x as usize)
                    .ok_or_else(|| format!("artifact: {name}: bad shape entry"))
            })
            .collect::<Result<Vec<usize>, String>>()?;
        let offsets = entry["data_offsets"]
            .as_array()
            .ok_or_else(|| format!("artifact: {name}: missing data_offsets"))?;
        if offsets.len() != 2 {
            return Err(format!(
                "artifact: {name}: data_offsets must be [begin, end]"
            ));
        }
        let begin = offsets[0].as_u64().unwrap_or(u64::MAX) as usize;
        let end = offsets[1].as_u64().unwrap_or(0) as usize;
        let numel: usize = shape.iter().product();
        let expect = numel
            .checked_mul(dtype.width())
            .ok_or_else(|| format!("artifact: {name}: size overflow"))?;
        if end.saturating_sub(begin) != expect {
            return Err(format!(
                "artifact: {name}: data span {} bytes != the {} layout ({expect} for {numel} elements)",
                end.saturating_sub(begin),
                dtype_s
            ));
        }
        let abs_begin = data_start
            .checked_add(begin)
            .ok_or_else(|| format!("artifact: {name}: offset overflow"))?;
        let abs_end = abs_begin
            .checked_add(end.saturating_sub(begin))
            .ok_or_else(|| format!("artifact: {name}: offset overflow"))?;
        if bytes.len() < abs_end {
            return Err(format!(
                "artifact: {name}: data ends at {abs_end} but the file has {} bytes",
                bytes.len()
            ));
        }
        tensors.push((
            name.clone(),
            StTensor {
                dtype,
                shape,
                data: StData::decode(dtype, &bytes[abs_begin..abs_end]),
            },
        ));
    }
    Ok(StMap { tensors })
}

/// The config.json scalar constants the loader consumes (a typed view of
/// the upstream `prepare.py` schema).
struct LayerCfg<'a> {
    q_k: i64,
    k_k: i64,
    v_k: i64,
    o_k: i64,
    gate_k: i64,
    up_k: i64,
    down_k: i64,
    qs_k: i64,
    as_k: i64,
    ms_k: i64,
    ds_k: i64,
    os_k: i64,
    q_norm_k: i64,
    k_norm_k: i64,
    _marker: core::marker::PhantomData<&'a ()>,
}

fn cfg_i64(v: &serde_json::Value, key: &str) -> Result<i64, String> {
    v.as_i64()
        .ok_or_else(|| format!("artifact: config {key} must be an integer"))
}

fn layer_cfg(layer: &serde_json::Value, li: usize) -> Result<LayerCfg<'_>, String> {
    let lin_k = |name: &str| -> Result<i64, String> {
        let obj = layer
            .get(name)
            .ok_or_else(|| format!("artifact: config layers[{li}].{name} missing"))?;
        obj.get("k")
            .ok_or_else(|| format!("artifact: config layers[{li}].{name}.k missing"))?
            .as_i64()
            .ok_or_else(|| format!("artifact: config layers[{li}].{name}.k must be an integer"))
    };
    let scalar = |name: &str| -> Result<i64, String> {
        cfg_i64(
            layer
                .get(name)
                .ok_or_else(|| format!("artifact: config layers[{li}].{name} missing"))?,
            &format!("layers[{li}].{name}"),
        )
    };
    Ok(LayerCfg {
        q_k: lin_k("q")?,
        k_k: lin_k("k")?,
        v_k: lin_k("v")?,
        o_k: lin_k("o")?,
        gate_k: lin_k("gate")?,
        up_k: lin_k("up")?,
        down_k: lin_k("down")?,
        qs_k: scalar("qs_k")?,
        as_k: scalar("as_k")?,
        ms_k: scalar("ms_k")?,
        ds_k: scalar("ds_k")?,
        os_k: scalar("os_k")?,
        q_norm_k: scalar("q_norm_k")?,
        k_norm_k: scalar("k_norm_k")?,
        _marker: core::marker::PhantomData,
    })
}

/// Assemble the artifact from a parsed container + the config.json
/// value. Tensor-level shape law lives HERE (names, lengths vs config);
/// the semantic load checks stay with [`IntRefModel::new`].
pub fn assemble_artifact(
    map: &StMap,
    config: &serde_json::Value,
) -> Result<IntRefArtifact, String> {
    let cfg_i = |key: &str| -> Result<usize, String> {
        config
            .get(key)
            .ok_or_else(|| format!("artifact: config {key} missing"))?
            .as_u64()
            .map(|v| v as usize)
            .ok_or_else(|| format!("artifact: config {key} must be a non-negative integer"))
    };
    let n_layers = cfg_i("n_layers")?;
    let hidden = cfg_i("hidden")?;
    let n_q = cfg_i("n_q_heads")?;
    let n_kv = cfg_i("n_kv_heads")?;
    let hd = cfg_i("head_dim")?;
    let vocab = cfg_i("vocab")?;
    let intermediate = cfg_i("intermediate")?;
    let max_pos = cfg_i("max_pos")?;
    let rope_bits = cfg_i("rope_frac_bits")?;
    if rope_bits != ROPE_FRAC_BITS as usize {
        return Err(format!(
            "artifact: rope_frac_bits {rope_bits} != the lane's ROPE_FRAC_BITS"
        ));
    }
    let kv_dim = n_kv * hd;
    let q_dim = n_q * hd;
    // upstream prepare.py's overflow assertions, re-checked at load
    if (hd as i64) * 127 * 127 >= (1_i64 << 31)
        || (intermediate as i64) * 127 * 127 >= (1_i64 << 31)
    {
        return Err("artifact: head_dim/intermediate breach the int32 GEMM bound".to_string());
    }
    if (max_pos as i64) * 128 * 127 >= (1_i64 << 31) {
        return Err(
            "artifact: max_pos breaches the probs·V accumulation bound (upstream config law)"
                .to_string(),
        );
    }

    let lin = |name: &str, k_in: usize, n_out: usize, k: i64| -> Result<LinearWeights, String> {
        Ok(LinearWeights {
            w8t: map.expect_i8(&format!("{name}.w8t"), k_in * n_out)?,
            m: map.expect_i64(&format!("{name}.m"), n_out)?,
            k,
        })
    };

    let head_k = config
        .get("head")
        .ok_or("artifact: config head missing")?
        .get("k")
        .ok_or("artifact: config head.k missing")?
        .as_i64()
        .ok_or("artifact: config head.k must be an integer")?;
    let head = lin("head", hidden, vocab, head_k)?;

    let cos = map.expect_i16("rope.cos", max_pos * hd)?;
    let sin = map.expect_i16("rope.sin", max_pos * hd)?;
    let rope = RopeTables {
        cos,
        sin,
        max_pos,
        head_dim: hd,
    };

    let embed_e8 = map.expect_i8("embed.e8", vocab * hidden)?;
    let embed_m = map.expect_i64("embed.m", vocab)?;
    let embed_k = map.expect_i64("embed.k", vocab)?;

    let mut layers = Vec::with_capacity(n_layers);
    for li in 0..n_layers {
        let p = format!("layers.{li}.");
        let lc = layer_cfg(
            config
                .get("layers")
                .ok_or("artifact: config layers missing")?
                .get(li)
                .ok_or_else(|| format!("artifact: config layers[{li}] missing"))?,
            li,
        )?;
        let layer = LayerWeights {
            q: lin(&format!("{p}q"), hidden, q_dim, lc.q_k)?,
            k: lin(&format!("{p}k"), hidden, kv_dim, lc.k_k)?,
            v: lin(&format!("{p}v"), hidden, kv_dim, lc.v_k)?,
            o: lin(&format!("{p}o"), q_dim, hidden, lc.o_k)?,
            gate: lin(&format!("{p}gate"), hidden, intermediate, lc.gate_k)?,
            up: lin(&format!("{p}up"), hidden, intermediate, lc.up_k)?,
            down: lin(&format!("{p}down"), intermediate, hidden, lc.down_k)?,
            q_norm_m: map.expect_i64(&format!("{p}q_norm.m"), hd)?,
            q_norm_k: u32::try_from(lc.q_norm_k)
                .map_err(|_| "artifact: q_norm_k negative".to_string())?,
            k_norm_m: map.expect_i64(&format!("{p}k_norm.m"), hd)?,
            k_norm_k: u32::try_from(lc.k_norm_k)
                .map_err(|_| "artifact: k_norm_k negative".to_string())?,
            qs_m: map.expect_i64(&format!("{p}qs_m"), kv_dim)?,
            qs_k: u32::try_from(lc.qs_k)
                .map_err(|_| format!("artifact: layers[{li}].qs_k negative"))?,
            as_m: map.expect_i64(&format!("{p}as_m"), hidden)?,
            as_k: u32::try_from(lc.as_k)
                .map_err(|_| format!("artifact: layers[{li}].as_k negative"))?,
            ms_m: map.expect_i64(&format!("{p}ms_m"), hidden)?,
            ms_k: u32::try_from(lc.ms_k)
                .map_err(|_| format!("artifact: layers[{li}].ms_k negative"))?,
            ds_m: map.expect_i64(&format!("{p}ds_m"), intermediate)?,
            ds_k: u32::try_from(lc.ds_k)
                .map_err(|_| format!("artifact: layers[{li}].ds_k negative"))?,
            os_m: map.expect_i64(&format!("{p}os_m"), q_dim)?,
            os_k: lc.os_k,
            kq_m: map.expect_i64(&format!("{p}kq_m"), kv_dim)?,
            kq_k: map.expect_i64(&format!("{p}kq_k"), kv_dim)?,
            ks_m: map.expect_i64(&format!("{p}ks_m"), n_kv)?,
            ks_k: map.expect_i64(&format!("{p}ks_k"), n_kv)?,
            vq_m: map.expect_i64(&format!("{p}vq_m"), n_kv)?,
            vq_k: map.expect_i64(&format!("{p}vq_k"), n_kv)?,
        };
        layers.push(layer);
    }

    // the typo detector: upstream prepare.py writes EXACTLY this set
    let mut expect = std::collections::BTreeSet::new();
    for n in [
        "embed.e8", "embed.m", "embed.k", "rope.cos", "rope.sin", "head.w8t", "head.m",
    ] {
        expect.insert(n.to_string());
    }
    for li in 0..n_layers {
        let p = format!("layers.{li}.");
        for suffix in [
            "q.w8t", "q.m", "k.w8t", "k.m", "v.w8t", "v.m", "o.w8t", "o.m", "gate.w8t", "gate.m",
            "up.w8t", "up.m", "down.w8t", "down.m", "q_norm.m", "k_norm.m", "qs_m", "as_m", "ms_m",
            "ds_m", "os_m", "kq_m", "kq_k", "ks_m", "ks_k", "vq_m", "vq_k",
        ] {
            expect.insert(format!("{p}{suffix}"));
        }
    }
    let got: std::collections::BTreeSet<&str> = map.names().collect();
    for name in &got {
        if !expect.contains(*name) {
            return Err(format!(
                "artifact: unexpected tensor {name} (upstream writes exactly the documented set)"
            ));
        }
    }

    Ok(IntRefArtifact {
        cfg: IntRefConfig {
            n_layers,
            hidden,
            n_q,
            n_kv,
            head_dim: hd,
            vocab,
            intermediate,
        },
        embed_e8,
        embed_m,
        embed_k,
        head,
        rope,
        layers,
    })
}

/// Blake3 digests of the two artifact files (the pin).
#[derive(Clone, Debug)]
pub struct ArtifactDigests {
    pub model: String,
    pub config: String,
}

/// Load an artifact dir (`model.safetensors` + `config.json`): read,
/// digest, parse, assemble.
pub fn load_dir(dir: &std::path::Path) -> Result<(IntRefArtifact, ArtifactDigests), String> {
    let st_path = dir.join("model.safetensors");
    let cfg_path = dir.join("config.json");
    let st_bytes =
        std::fs::read(&st_path).map_err(|e| format!("artifact: {}: {e}", st_path.display()))?;
    let cfg_bytes =
        std::fs::read(&cfg_path).map_err(|e| format!("artifact: {}: {e}", cfg_path.display()))?;
    let digests = ArtifactDigests {
        model: blake3::hash(&st_bytes).to_string(),
        config: blake3::hash(&cfg_bytes).to_string(),
    };
    let map = parse_container(&st_bytes)?;
    let config: serde_json::Value =
        serde_json::from_slice(&cfg_bytes).map_err(|e| format!("artifact: config JSON: {e}"))?;
    let artifact = assemble_artifact(&map, &config)?;
    Ok((artifact, digests))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::laya::riir::int_ref::model::IntRefModel;

    /// The test-side safetensors WRITER (the format's other half):
    /// 8-byte LE header length + JSON header + concatenated payloads.
    /// Round-tripping through this proves the reader against the exact
    /// byte layout upstream prepare.py writes.
    fn write_container(entries: &[(&str, &str, Vec<i64>)]) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        let mut blobs: Vec<Vec<u8>> = Vec::new();
        let mut offset = 0_usize;
        for (name, dtype, values) in entries {
            let (width, bytes): (usize, Vec<u8>) = match (*dtype, values.as_slice()) {
                ("I8", v) => {
                    let mut b = Vec::with_capacity(v.len());
                    for &x in v {
                        b.push(x as i8 as u8);
                    }
                    (1, b)
                }
                ("I16", v) => {
                    let mut b = Vec::with_capacity(v.len() * 2);
                    for &x in v {
                        b.extend_from_slice(&(x as i16).to_le_bytes());
                    }
                    (2, b)
                }
                ("I32", v) => {
                    let mut b = Vec::with_capacity(v.len() * 4);
                    for &x in v {
                        b.extend_from_slice(&(x as i32).to_le_bytes());
                    }
                    (4, b)
                }
                ("I64", v) => {
                    let mut b = Vec::with_capacity(v.len() * 8);
                    for &x in v {
                        b.extend_from_slice(&x.to_le_bytes());
                    }
                    (8, b)
                }
                _ => panic!("test writer: dtype {dtype} unsupported"),
            };
            let end = offset + bytes.len();
            header.insert(
                (*name).to_string(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": [values.len()],
                    "data_offsets": [offset, end],
                }),
            );
            blobs.push(bytes);
            offset = end;
            let _ = width;
        }
        let header_json = serde_json::to_string(&serde_json::Value::Object(header)).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
        out.extend_from_slice(header_json.as_bytes());
        for b in blobs {
            out.extend_from_slice(&b);
        }
        out
    }

    fn toy_config(n_layers: usize) -> serde_json::Value {
        serde_json::json!({
            "n_layers": n_layers,
            "hidden": 8,
            "n_q_heads": 2,
            "n_kv_heads": 1,
            "head_dim": 4,
            "vocab": 16,
            "intermediate": 16,
            "max_pos": 64,
            "rope_frac_bits": ROPE_FRAC_BITS,
            "head": {"k": 22},
            "layers": (0..n_layers).map(|_| serde_json::json!({
                "q": {"k": 22}, "k": {"k": 22}, "v": {"k": 22},
                "o": {"k": 22}, "gate": {"k": 22}, "up": {"k": 22},
                "down": {"k": 22},
                "qs_k": 10, "as_k": 10, "ms_k": 10, "ds_k": 10,
                "os_k": 10, "q_norm_k": 11, "k_norm_k": 11,
            })).collect::<Vec<_>>(),
        })
    }

    fn toy_entries(n_layers: usize) -> Vec<(&'static str, &'static str, Vec<i64>)> {
        // embed rows must be NONZERO: an all-zero embedding table drives
        // the norm outputs (and their claimed k) pathological, and the
        // V static requant's divisor shift overflows — the checked-op
        // refusal firing on a degenerate calibration, by design
        let embeds: Vec<i64> = (0..16 * 8).map(|i| ((i % 7) as i64) - 3).collect();
        let mut e: Vec<(&'static str, &'static str, Vec<i64>)> = vec![
            ("embed.e8", "I8", embeds),
            ("embed.m", "I64", vec![100; 16]),
            ("embed.k", "I64", vec![20; 16]),
            ("head.w8t", "I8", vec![3; 8 * 16]),
            ("head.m", "I64", vec![7; 16]),
            ("rope.cos", "I16", vec![1 << 14; 64 * 4]),
            ("rope.sin", "I16", vec![0; 64 * 4]),
        ];
        for li in 0..n_layers {
            let p = format!("layers.{li}.");
            let mut push = |suffix: &'static str, dtype: &'static str, n: usize, val: i64| {
                e.push((
                    Box::leak(format!("{p}{suffix}").into_boxed_str()),
                    dtype,
                    vec![val; n],
                ));
            };
            push("q.w8t", "I8", 8 * 8, 1);
            push("q.m", "I64", 8, 10);
            push("k.w8t", "I8", 8 * 4, 1);
            push("k.m", "I64", 4, 10);
            push("v.w8t", "I8", 8 * 4, 1);
            push("v.m", "I64", 4, 10);
            push("o.w8t", "I8", 8 * 8, 1);
            push("o.m", "I64", 8, 10);
            push("gate.w8t", "I8", 8 * 16, 1);
            push("gate.m", "I64", 16, 10);
            push("up.w8t", "I8", 8 * 16, 1);
            push("up.m", "I64", 16, 10);
            push("down.w8t", "I8", 16 * 8, 1);
            push("down.m", "I64", 8, 10);
            push("q_norm.m", "I64", 4, 500);
            push("k_norm.m", "I64", 4, 500);
            push("qs_m", "I64", 4, 5);
            push("as_m", "I64", 8, 9);
            push("ms_m", "I64", 8, 9);
            push("ds_m", "I64", 16, 9);
            push("os_m", "I64", 8, 9);
            push("kq_m", "I64", 4, 640);
            push("kq_k", "I64", 4, 17);
            push("ks_m", "I64", 1, 1);
            push("ks_k", "I64", 1, 1);
            push("vq_m", "I64", 1, 32768);
            push("vq_k", "I64", 1, 20);
        }
        e
    }

    #[test]
    fn container_round_trips_into_a_live_model() {
        let n_layers = 2_usize;
        let bytes = write_container(&toy_entries(n_layers));
        let map = parse_container(&bytes).expect("parse");
        let config = toy_config(n_layers);
        let artifact = assemble_artifact(&map, &config).expect("assemble");
        // the semantic load checks run; a forward returns sane shapes
        let m = IntRefModel::new(artifact);
        let (logits, xm, _xk) = m.forward(&[1_u32, 2, 3], &[0, 1, 2], None, None, 1, 3);
        assert_eq!(logits.len(), 3 * 16);
        assert_eq!(xm.len(), 3);
        // determinism through the LOADED artifact, not just the builder
        let (lg2, _, _) = m.forward(&[1_u32, 2, 3], &[0, 1, 2], None, None, 1, 3);
        assert_eq!(logits, lg2);
    }

    #[test]
    fn float_dtype_refuses_at_the_lane_door() {
        // an F32-typed entry must be refused loud (the mirror image of
        // upstream's own float-leak refusal)
        let bytes = write_container(&[("embed.k", "I64", vec![20; 16])]);
        // splice an F32 dtype into the header
        let patched = String::from_utf8(bytes)
            .unwrap()
            .replace("\"I64\"", "\"F32\"");
        let err = parse_container(patched.as_bytes()).expect_err("float must refuse");
        assert!(err.contains("not an int tensor"), "unexpected error: {err}");
    }

    #[test]
    fn missing_tensor_names_itself() {
        let entries: Vec<_> = toy_entries(1)
            .into_iter()
            .filter(|(n, _, _)| *n != "head.m")
            .collect();
        let bytes = write_container(&entries);
        let map = parse_container(&bytes).expect("parse");
        let err = assemble_artifact(&map, &toy_config(1)).expect_err("missing tensor must refuse");
        assert!(
            err.contains("missing tensor head.m"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn unexpected_tensor_refuses_as_a_typo() {
        let mut entries = toy_entries(1);
        entries.push(("layers.0.qs_M", "I64", vec![5; 4])); // case typo
        let bytes = write_container(&entries);
        let map = parse_container(&bytes).expect("parse");
        let err = assemble_artifact(&map, &toy_config(1)).expect_err("extra tensor must refuse");
        assert!(err.contains("unexpected tensor"), "unexpected error: {err}");
    }

    #[test]
    fn wrong_element_count_refuses_loud() {
        let mut entries = toy_entries(1);
        entries[1] = ("embed.m", "I64", vec![100; 15]); // want 16
        let bytes = write_container(&entries);
        let map = parse_container(&bytes).expect("parse");
        let err = assemble_artifact(&map, &toy_config(1)).expect_err("wrong count must refuse");
        assert!(
            err.contains("embed.m has 15 elements"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rope_frac_bits_mismatch_refuses() {
        let bytes = write_container(&toy_entries(1));
        let map = parse_container(&bytes).expect("parse");
        let mut config = toy_config(1);
        config["rope_frac_bits"] = serde_json::json!(12);
        let err = assemble_artifact(&map, &config).expect_err("rope mismatch must refuse");
        assert!(err.contains("ROPE_FRAC_BITS"), "unexpected error: {err}");
    }
}
