//! The safetensors vote source (Issue 1251 T4) — the detector's native
//! HF-format lane: most third-party checkpoints arrive as `.safetensors`
//! (transformers format), not GGUF, and the supply-chain screen must
//! read what actually arrives. Implements the SAME [`VoteTensors`] trait
//! the GGUF source implements, over a minimal in-repo safetensors
//! reader (the format is an 8-byte LE header length + JSON header +
//! raw tensor blobs — no external crate, zero new deps).
//!
//! **Tensor-name mapping (llama-family HF naming):**
//! - `model.embed_tokens.weight` → the input embedding (the vote's
//!   pinned `e_t` proxy);
//! - `model.layers.{N}.mlp.gate_proj.weight` / `mlp.up_proj.weight` →
//!   the per-neuron rows (row-major `[intermediate × hidden]` — the
//!   same rows-are-neurons convention the GGUF lane reads).
//!
//! Covers Qwen2 / Llama / Mistal-class checkpoints. Gemma-2/3 differences
//! (tie_word_embeddings, different norm names) do not affect the MLP
//! slice this source reads.
//!
//! **dtypes:** BF16 / F16 / F32 → exact f32 upcast (the vote's f32-bit
//! row-skip compares the upcast rows; bf16→f32 is exact, so bit-identical
//! bf16 rows stay bit-identical).
//!
//! Posture: opt-in under `readout_screen`, offline (weights only, no
//! forward pass), zero new deps.

use crate::readout_screen::file_digest_blake3;
use crate::trigger_vote::VoteTensors;
use anyhow::{Context, Result, bail, ensure};
use std::collections::HashMap;
use std::path::Path;

/// One safetensors tensor's metadata.
#[derive(Debug, Clone)]
pub struct StTensorInfo {
    pub name: String,
    pub dtype: StDtype,
    pub shape: Vec<usize>,
    /// Absolute byte offset of the tensor data in the file.
    pub data_start: u64,
    /// Total bytes of tensor data.
    pub byte_len: u64,
}

/// The safetensors dtypes this lane reads (everything else refuses
/// loud — a silent wrong-dtype read would corrupt the vote silently).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StDtype {
    F32,
    F16,
    Bf16,
}

impl StDtype {
    fn parse(s: &str) -> Result<Self> {
        match s {
            "F32" => Ok(StDtype::F32),
            "F16" => Ok(StDtype::F16),
            "BF16" => Ok(StDtype::Bf16),
            other => bail!("unsupported safetensors dtype '{other}' (F32/F16/BF16 only)"),
        }
    }

    /// Bytes per element.
    pub fn elem_bytes(self) -> usize {
        match self {
            StDtype::F32 => 4,
            StDtype::F16 | StDtype::Bf16 => 2,
        }
    }
}

/// A parsed safetensors file (mmapped; tensor blobs are zero-copy
/// views into the map).
pub struct StFile {
    mmap: memmap2::Mmap,
    tensor_map: HashMap<String, StTensorInfo>,
    pub path: std::path::PathBuf,
}

/// A shard set: the `model.safetensors.index.json` convention where a
/// checkpoint's tensors spread over several files, each tensor living
/// WHOLLY in one shard (never split). Lookup routes by tensor name;
/// the digest is the BLAKE3 of the per-shard whole-file digests in the
/// set's order (stable per file list).
pub struct StShardSet {
    shards: Vec<StFile>,
    /// Per-tensor shard index (the shard carrying it).
    routes: HashMap<String, usize>,
    digest: Option<[u8; 32]>,
}

impl StShardSet {
    /// Open every shard; refuse a tensor carried by two shards (a
    /// duplicated tensor is a corrupted set, never a lookup ambiguity
    /// to resolve silently).
    pub fn open(paths: &[std::path::PathBuf]) -> Result<Self> {
        ensure!(!paths.is_empty(), "empty shard list");
        let mut shards = Vec::with_capacity(paths.len());
        let mut routes = HashMap::with_capacity(paths.len() * 512);
        let mut digest_hasher = blake3::Hasher::new();
        for (i, p) in paths.iter().enumerate() {
            let f = StFile::open(p)?;
            for name in f.tensor_map.keys() {
                let prior = routes.insert(name.clone(), i);
                ensure!(
                    prior.is_none(),
                    "tensor '{name}' carried by shards {} and {i} — refusing the ambiguous set",
                    prior.unwrap_or(usize::MAX)
                );
            }
            if let Some(d) = file_digest_blake3(p) {
                digest_hasher.update(&d);
            }
            shards.push(f);
        }
        Ok(Self {
            shards,
            routes,
            digest: Some(*digest_hasher.finalize().as_bytes()),
        })
    }

    pub fn info(&self, name: &str) -> Option<&StTensorInfo> {
        self.routes
            .get(name)
            .and_then(|&i| self.shards[i].tensor_info(name))
    }

    pub fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
        self.routes
            .get(name)
            .and_then(|&i| self.shards[i].tensor_slice(name))
    }

    /// Dequantize a whole tensor to f32 (routes to the carrying shard).
    pub fn dequant_tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let i = self
            .routes
            .get(name)
            .with_context(|| format!("tensor '{name}' not in the shard set"))?;
        self.shards[*i].dequant_tensor_f32(name)
    }

    /// All tensor names across the set.
    pub fn tensor_names(&self) -> impl Iterator<Item = &String> {
        self.routes.keys()
    }
}

impl StFile {
    /// Parse a safetensors file: 8-byte LE header length, the JSON
    /// header (name → {dtype, shape, data_offsets}), then the blob
    /// region. `__metadata__` is accepted and ignored.
    pub fn open(path: &Path) -> Result<Self> {
        let file =
            std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .with_context(|| format!("mmapping {}", path.display()))?;
        if mmap.len() < 8 {
            bail!("{}: too short for a safetensors header", path.display());
        }
        let header_len = u64::from_le_bytes(mmap[..8].try_into().expect("header len")) as usize;
        let header_end = 8 + header_len;
        ensure!(
            header_end <= mmap.len(),
            "{}: header length {} exceeds the file",
            path.display(),
            header_len
        );
        let header: HashMap<String, serde_json::Value> =
            serde_json::from_slice(&mmap[8..header_end])
                .with_context(|| format!("parsing the {} JSON header", path.display()))?;
        let mut tensor_map = HashMap::with_capacity(header.len());
        for (name, v) in header {
            if name == "__metadata__" {
                continue;
            }
            let dtype_s = v
                .get("dtype")
                .and_then(|d| d.as_str())
                .with_context(|| format!("tensor '{name}': missing dtype"))?;
            let dtype = StDtype::parse(dtype_s)?;
            let shape: Vec<usize> = v
                .get("shape")
                .and_then(|s| s.as_array())
                .map(|a| a.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect())
                .with_context(|| format!("tensor '{name}': missing shape"))?;
            let offs = v
                .get("data_offsets")
                .and_then(|o| o.as_array())
                .with_context(|| format!("tensor '{name}': missing data_offsets"))?;
            ensure!(
                offs.len() == 2,
                "tensor '{name}': data_offsets must be [begin, end]"
            );
            let begin = offs[0].as_u64().context("data_offsets[0]")?;
            let end = offs[1].as_u64().context("data_offsets[1]")?;
            ensure!(end >= begin, "tensor '{name}': inverted data_offsets");
            let n_elements: usize = shape.iter().product();
            ensure!(
                (end - begin) as usize == n_elements * dtype.elem_bytes(),
                "tensor '{name}': data length {} != {} elements × {} B",
                end - begin,
                n_elements,
                dtype.elem_bytes()
            );
            tensor_map.insert(
                name.clone(),
                StTensorInfo {
                    name,
                    dtype,
                    shape,
                    data_start: header_end as u64 + begin,
                    byte_len: end - begin,
                },
            );
        }
        ensure!(
            !tensor_map.is_empty(),
            "{}: no tensors in the header",
            path.display()
        );
        Ok(Self {
            mmap,
            tensor_map,
            path: path.to_path_buf(),
        })
    }

    pub fn tensor_info(&self, name: &str) -> Option<&StTensorInfo> {
        self.tensor_map.get(name)
    }

    /// The tensor's raw byte slice.
    pub fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
        let info = self.tensor_map.get(name)?;
        let start = info.data_start as usize;
        let end = start + info.byte_len as usize;
        self.mmap.get(start..end)
    }

    /// Dequantize a whole tensor to f32 (BF16/F16 exact upcasts; F32
    /// passthrough).
    pub fn dequant_tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let info = self
            .tensor_info(name)
            .with_context(|| format!("tensor '{name}' not found"))?;
        let slice = self
            .tensor_slice(name)
            .with_context(|| format!("tensor '{name}' data out of bounds"))?;
        let n = info.shape.iter().product::<usize>();
        ensure!(
            slice.len() == n * info.dtype.elem_bytes(),
            "tensor '{name}': length mismatch"
        );
        let mut out = Vec::with_capacity(n);
        match info.dtype {
            StDtype::F32 => {
                for chunk in slice.as_chunks::<4>().0 {
                    out.push(f32::from_le_bytes(*chunk));
                }
            }
            StDtype::F16 => {
                for chunk in slice.as_chunks::<2>().0 {
                    out.push(half::f16::from_bits(u16::from_le_bytes(*chunk)).to_f32());
                }
            }
            StDtype::Bf16 => {
                for chunk in slice.as_chunks::<2>().0 {
                    out.push(half::bf16::from_bits(u16::from_le_bytes(*chunk)).to_f32());
                }
            }
        }
        ensure!(out.len() == n, "tensor '{name}': dequant length mismatch");
        Ok(out)
    }
}

/// The HF safetensors vote source (llama-family naming; single-file
/// and sharded checkpoints both arrive as an [`StShardSet`]).
pub struct SafetensorsVoteSource<'a> {
    st: &'a StShardSet,
    path_digest: Option<[u8; 32]>,
    n_layers: usize,
    n_ff: usize,
    n_embd: usize,
    vocab: usize,
    gate: Vec<String>,
    up: Vec<String>,
    emb: String,
}

impl<'a> SafetensorsVoteSource<'a> {
    /// Open over every gate/up-carrying block (ascending layer index;
    /// contiguity asserted — a hole would silently shift the band).
    pub fn open(st: &'a StShardSet) -> Result<Self> {
        let mut layer_ids: Vec<usize> = Vec::new();
        for name in st.tensor_names() {
            if let Some(rest) = name.strip_prefix("model.layers.")
                && let Some((idx, suffix)) = rest.split_once('.')
                && suffix == "mlp.gate_proj.weight"
                && let Ok(i) = idx.parse::<usize>()
            {
                layer_ids.push(i);
            }
        }
        if layer_ids.is_empty() {
            bail!("no model.layers.N.mlp.gate_proj.weight tensors — not a llama-family checkpoint");
        }
        layer_ids.sort_unstable();
        layer_ids.dedup();
        let n_layers = layer_ids.len();
        for (pos, &id) in layer_ids.iter().enumerate() {
            if id != pos {
                bail!(
                    "ffn layers are not contiguous (layer {id} at position {pos}) — \
                     refusing the partial sweep"
                );
            }
        }
        let gate: Vec<String> = layer_ids
            .iter()
            .map(|i| format!("model.layers.{i}.mlp.gate_proj.weight"))
            .collect();
        let up: Vec<String> = layer_ids
            .iter()
            .map(|i| format!("model.layers.{i}.mlp.up_proj.weight"))
            .collect();
        for name in &up {
            ensure!(
                st.info(name).is_some(),
                "layer tensor '{name}' not found — refusing the partial sweep"
            );
        }
        // gate/up: shape [intermediate × hidden] — ne[0] = row length =
        // n_embd, ne[1] = rows = n_ff (the GGUF lane's convention).
        let gate_shape = st.info(&gate[0]).unwrap().shape.clone();
        let up_shape = st.info(&up[0]).unwrap().shape.clone();
        ensure!(
            gate_shape.len() == 2 && up_shape.len() == 2,
            "gate/up must be 2-D (gate {gate_shape:?}, up {up_shape:?})"
        );
        ensure!(
            gate_shape == up_shape,
            "gate shape {gate_shape:?} != up shape {up_shape:?} — refusing the asymmetric layer"
        );
        // HF stores [intermediate, hidden] — shape[0] = n_ff rows,
        // shape[1] = n_embd row length.
        let (n_ff, n_embd) = (gate_shape[0], gate_shape[1]);
        for (l, name) in gate.iter().enumerate().skip(1) {
            let s = st.info(name).unwrap().shape.clone();
            ensure!(
                s == gate_shape,
                "layer {l} gate shape {s:?} != layer 0 {gate_shape:?}"
            );
        }
        let emb = "model.embed_tokens.weight".to_string();
        let emb_info = st
            .info(&emb)
            .context("no model.embed_tokens.weight — the vote reads input embeddings")?;
        ensure!(
            emb_info.shape.len() == 2 && emb_info.shape[1] == n_embd,
            "embedding shape {:?} disagrees with n_embd {n_embd} (want [vocab × n_embd])",
            emb_info.shape
        );
        let vocab = emb_info.shape[0];
        Ok(Self {
            st,
            path_digest: st.digest,
            n_layers,
            n_ff,
            n_embd,
            vocab,
            gate,
            up,
            emb,
        })
    }
}

impl VoteTensors for SafetensorsVoteSource<'_> {
    fn n_layers(&self) -> usize {
        self.n_layers
    }
    fn n_ff(&self) -> usize {
        self.n_ff
    }
    fn n_embd(&self) -> usize {
        self.n_embd
    }
    fn vocab(&self) -> usize {
        self.vocab
    }
    fn gate_rows(&self, layer: usize) -> Result<Vec<f32>> {
        self.st
            .dequant_tensor_f32(&self.gate[layer])
            .with_context(|| format!("dequant {}", self.gate[layer]))
    }
    fn up_rows(&self, layer: usize) -> Result<Vec<f32>> {
        self.st
            .dequant_tensor_f32(&self.up[layer])
            .with_context(|| format!("dequant {}", self.up[layer]))
    }
    fn emb_rows(&self) -> Result<Vec<f32>> {
        self.st
            .dequant_tensor_f32(&self.emb)
            .with_context(|| format!("dequant {}", self.emb))
    }
    fn label(&self) -> String {
        "safetensors:model.embed_tokens.weight".to_string()
    }
    fn file_digest(&self) -> Option<[u8; 32]> {
        self.path_digest
    }
}
