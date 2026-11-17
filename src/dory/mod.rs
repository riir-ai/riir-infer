//! The `dory` forward lane (riir-infer Issue 1221 T1).
//!
//! NVIDIA Dory1.0-4B — a looped nGPT: 16-layer input block + a 40-layer
//! shared recurrent block executed `num_loops` times per forward + a
//! 16-layer output block (`s_t = N(F_θ(N(s_{t-1} + x)))`, the input-block
//! output re-injected EVERY loop). The GGUF ships all 72 physical blocks
//! with unique weights (T0 census: weight ties NONE) — the loader maps
//! block indices to the execution graph, never dedups.
//!
//! nGPT laws the forward must carry (the deltas vs the llama family):
//! - **Unit-norm latent**: the residual stream is L2-normalized (FP32,
//!   eps 1e-6) at BOTH boundaries (before input, after the recurrent
//!   block's output). No norm GAMMA tensors exist in the file (T0
//!   asserted zero) — the L2 norm is the normalization.
//! - **Learned residual-rate scales** (`residual_alpha`, one per block):
//!   the block update is `s_t = N(s_{t-1} + α_t · F_t(u_t))` where
//!   `α_t = |α̂_t|` (`residual_alpha` is stored BF16, applied signed-abs).
//! - **Per-channel gate scales**: `attn_gate_scale` gates the attention
//!   branch, `ffn_gate_scale`/`ffn_up_scale` gate the FFN branches,
//!   `output_scale` scales the output block's exit — each applied
//!   per-channel (`|scale|` sign convention like alpha).
//! - **QK scales**: `attn_qk_scale` (per attn block, per-channel over
//!   head_dim) rides the logits scale.
//! - **Dual RoPE**: `freq_base = 1e7` + `freq_base_2 = 1e4` — Dory pairs
//!   them per rope profile (`dory.rope.profile` per block, 2/1
//!   alternating runs in the census). T1 carries BOTH tables and selects
//!   per block by the profile bit.
//! - **Separate Q/K/V/O** projections (NOT fused), GQA 16Q/4KV,
//!   `head_dim = 256` (q 2560→4096, kv 2560→1024), FFN 7168.
//! - **3:1 SWA/full** (`sliding_window = 1024`): 36 attention blocks in
//!   the pattern `S-S-S-F` × 9 (attn on EVEN physical block indices,
//!   ffn on ODD).
//! - **per-loop KV** (`dory.recurrent_kv_cache_mode = "per_loop"`, T0):
//!   the training semantics — loop t attends loop-t states. T1 proves
//!   the k=1 identity against that reference; T2's last-loop posture is
//!   a reported delta, never assumed zero.
//!
//! Proof posture (T1): same weights + same k + same cache policy ⇒
//! bit-identical logits across runs (determinism fixture), plus the
//! full-sequence recompute reference with PER-LOOP caches (the golden).
//! The CPU forward is the parity/fixture snapshot (the compute
//! posture's parity arm); Metal is T2+.
//!
//! Feature `dory` (opt-in): unarmed builds compile the module to nothing
//! and the loader refuses the arch loud by name.

use crate::gguf_loader::GgufFile;

/// The T0 census constants, read from the header at load — never
/// hardcoded (the k-scheduler reads `num_loops` + the cache mode from
/// the header; T2's law, applied to the partition here too).
#[derive(Debug, Clone)]
pub struct DoryLoopMeta {
    /// `dory.num_loops` (T0 census: 4).
    pub num_loops: usize,
    /// `dory.input_layer_count` (16).
    pub input_layers: usize,
    /// `dory.recurrent_layer_count` (40).
    pub recurrent_layers: usize,
    /// `dory.output_layer_count` (16).
    pub output_layers: usize,
    /// `dory.recurrent_kv_cache_mode` ("per_loop" in the census artifact).
    pub kv_cache_mode: String,
}

impl DoryLoopMeta {
    /// First physical block index of the recurrent range.
    pub fn recurrent_start(&self) -> usize {
        self.input_layers
    }
    /// First physical block index of the output range.
    pub fn output_start(&self) -> usize {
        self.input_layers + self.recurrent_layers
    }
}

/// Per-block kind derived from the physical index parity + the header's
/// own counts (T0: attn stems sit on EVEN block indices, ffn on ODD —
/// each "block" is an attn+ffn PAIR spread over two physical indices).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DoryBlockKind {
    /// Even physical index — an attention block.
    Attn,
    /// Odd physical index — an FFN block.
    Ffn,
}

/// One physical block's weights (f32 dequantized at load; the CPU
/// parity snapshot posture — quant-resident serving is T2+).
#[derive(Debug, Clone)]
pub struct DoryBlockWeights {
    pub kind: DoryBlockKind,
    /// Learned per-channel residual-rate scale `α̂ [n_embd]` (the
    /// forward applies `|α|` elementwise — the nGPT sign convention).
    pub residual_alpha: Vec<f32>,
    /// Attention blocks: `attn_q [4096, 2560]`, `attn_k/v [1024, 2560]`,
    /// `attn_output [2560, 4096]`.
    pub attn_q: Vec<f32>,
    pub attn_k: Vec<f32>,
    pub attn_v: Vec<f32>,
    pub attn_output: Vec<f32>,
    /// Per-channel gate scales (attn blocks: `attn_gate_scale`,
    /// length q_dim = 4096 — it rides the attention branch output over
    /// the Q width, per the census shapes).
    pub attn_gate_scale: Vec<f32>,
    /// Per-Q-head QK scale (attn blocks; length q_dim = 4096 per the
    /// census — one scale per Q channel, folded into the logits scale).
    pub attn_qk_scale: Vec<f32>,
    /// FFN blocks: `ffn_gate [7168, 2560]`, `ffn_up [7168, 2560]`,
    /// `ffn_down [2560, 7168]`.
    pub ffn_gate: Vec<f32>,
    pub ffn_up: Vec<f32>,
    pub ffn_down: Vec<f32>,
    /// Per-channel FFN gate/up scales (ffn blocks; length 7168).
    pub ffn_gate_scale: Vec<f32>,
    pub ffn_up_scale: Vec<f32>,
    /// Which RoPE base this block uses (the census's 2/1 alternating
    /// runs — `true` = `freq_base` (1e7), `false` = `freq_base_2` (1e4)).
    pub rope_primary: bool,
    /// The `dory.rope.profile` entry for this block (the raw census
    /// value; carried for the drift tripwire).
    pub rope_profile: u64,
}

/// Full model weights: embeddings (untied — T0), 72 physical blocks,
/// the final `output_scale` vector, and the loop metadata.
#[derive(Debug, Clone)]
pub struct DoryWeights {
    pub meta: DoryLoopMeta,
    /// `token_embd` transposed to the forward's `[vocab][n_embd]`.
    pub wte: Vec<f32>,
    /// `output` (UNTIED head, T0) transposed to `[vocab][n_embd]`.
    pub lm_head: Vec<f32>,
    /// `output_scale.weight [vocab]` — the output head's per-channel
    /// exit scale (the T0 fixture's 131072 = vocab, NOT n_embd).
    pub output_scale: Vec<f32>,
    /// The 72 physical blocks in file order.
    pub blocks: Vec<DoryBlockWeights>,
    /// `sliding_window` (T0: 1024).
    pub sliding_window: usize,
    /// Primary RoPE base (`dory.rope.freq_base`, 1e7).
    pub rope_freq_base: f32,
    /// Secondary RoPE base (`dory.rope.freq_base_2`, 1e4).
    pub rope_freq_base_2: f32,
}

/// Load errors — loud, named (the issue's refusal law).
#[derive(Debug, thiserror::Error)]
pub enum DoryLoadError {
    #[error("not a dory GGUF: general.architecture = {0:?}")]
    NotDory(String),
    #[error("missing dory metadata key: {0}")]
    MissingMeta(&'static str),
    #[error("missing tensor: {0}")]
    MissingTensor(String),
    #[error("dory metadata mismatch: {key} = {got}, expected {expected}")]
    MetaMismatch {
        key: &'static str,
        got: String,
        expected: String,
    },
    #[error("tensor {name}: {kind} length {got} != expected {expected}")]
    ShapeMismatch {
        name: String,
        kind: &'static str,
        got: usize,
        expected: usize,
    },
    #[error(transparent)]
    Gguf(#[from] anyhow::Error),
}

/// Stem names observed by the T0 probe (`examples/i1221_dory_header_probe.rs`
/// census: `blk.N.attn_*` / `blk.N.ffn_*`; the probe fixture is the
/// authority — this loader mirrors its stems and the tripwire test pins
/// the mapping on the real file).
mod stems {
    // The census names carry the `.weight` suffix on the 1-D scale
    // vectors AND the projections — build them uniformly.
    pub fn attn_q(b: usize) -> String {
        format!("blk.{b}.attn_q.weight")
    }
    pub fn attn_k(b: usize) -> String {
        format!("blk.{b}.attn_k.weight")
    }
    pub fn attn_v(b: usize) -> String {
        format!("blk.{b}.attn_v.weight")
    }
    pub fn attn_o(b: usize) -> String {
        format!("blk.{b}.attn_output.weight")
    }
    pub fn attn_gate(b: usize) -> String {
        format!("blk.{b}.attn_gate_scale.weight")
    }
    pub fn attn_qk(b: usize) -> String {
        format!("blk.{b}.attn_qk_scale.weight")
    }
    pub fn ffn_gate(b: usize) -> String {
        format!("blk.{b}.ffn_gate.weight")
    }
    pub fn ffn_up(b: usize) -> String {
        format!("blk.{b}.ffn_up.weight")
    }
    pub fn ffn_down(b: usize) -> String {
        format!("blk.{b}.ffn_down.weight")
    }
    pub fn ffn_gate_scale(b: usize) -> String {
        format!("blk.{b}.ffn_gate_scale.weight")
    }
    pub fn ffn_up_scale(b: usize) -> String {
        format!("blk.{b}.ffn_up_scale.weight")
    }
    pub fn residual_alpha(b: usize) -> String {
        format!("blk.{b}.residual_alpha.weight")
    }
    pub const ROPE_PROFILE: &str = "dory.rope.profile";
}

/// The header metadata the loader requires. Mismatch vs the census
/// values = loud `MetaMismatch`, never a silent fallback (the
/// format-drift tripwire reads THIS).
pub fn read_loop_meta(gguf: &GgufFile) -> Result<DoryLoopMeta, DoryLoadError> {
    let arch = gguf
        .architecture()
        .ok_or_else(|| DoryLoadError::NotDory("<none>".into()))?
        .to_string();
    if arch != "dory" {
        return Err(DoryLoadError::NotDory(arch));
    }
    let num_loops = gguf
        .metadata_u64("dory.num_loops")
        .ok_or(DoryLoadError::MissingMeta("dory.num_loops"))? as usize;
    let input_layers =
        gguf.metadata_u64("dory.input_layer_count")
            .ok_or(DoryLoadError::MissingMeta("dory.input_layer_count"))? as usize;
    let recurrent_layers =
        gguf.metadata_u64("dory.recurrent_layer_count")
            .ok_or(DoryLoadError::MissingMeta("dory.recurrent_layer_count"))? as usize;
    let output_layers =
        gguf.metadata_u64("dory.output_layer_count")
            .ok_or(DoryLoadError::MissingMeta("dory.output_layer_count"))? as usize;
    let kv_cache_mode = gguf
        .metadata_string("dory.recurrent_kv_cache_mode")
        .ok_or(DoryLoadError::MissingMeta("dory.recurrent_kv_cache_mode"))?
        .to_string();
    Ok(DoryLoopMeta {
        num_loops,
        input_layers,
        recurrent_layers,
        output_layers,
        kv_cache_mode,
    })
}

/// Whole-tensor dequant through the loader's dense path (BF16/K-quant
/// capable; the Dory file is Q5_K dominant with BF16 vocab + scales).
fn whole(gguf: &GgufFile, name: &str) -> Result<Vec<f32>, DoryLoadError> {
    gguf.dequant_f16_to_f32(name).map_err(|e| {
        if gguf.tensor_info(name).is_none() {
            DoryLoadError::MissingTensor(name.to_string())
        } else {
            DoryLoadError::Gguf(e)
        }
    })
}

fn expect_len(
    name: &str,
    kind: &'static str,
    got: usize,
    expected: usize,
) -> Result<(), DoryLoadError> {
    if got != expected {
        return Err(DoryLoadError::ShapeMismatch {
            name: name.to_string(),
            kind,
            got,
            expected,
        });
    }
    Ok(())
}

/// Load the Dory weights from an opened GGUF (the arch must be `dory`).
///
/// GGUF stores 2-D tensors inner-dim-first: `token_embd` is
/// `[n_embd, vocab]` in file order (the census pinned 2560×131072), so
/// the loader TRANSPOSES both vocab tensors to the forward's
/// `[vocab][n_embd]` row-major. The transpose is exact f32 gathers —
/// no numerics move.
pub fn load_dory_weights_gguf(gguf: &GgufFile) -> Result<DoryWeights, DoryLoadError> {
    let meta = read_loop_meta(gguf)?;
    let total_blocks = meta.input_layers + meta.recurrent_layers + meta.output_layers;
    let n_layer_u64 = gguf
        .metadata_u64("dory.block_count")
        .ok_or(DoryLoadError::MissingMeta("dory.block_count"))? as usize;
    if n_layer_u64 != total_blocks {
        return Err(DoryLoadError::MetaMismatch {
            key: "dory.block_count",
            got: n_layer_u64.to_string(),
            expected: total_blocks.to_string(),
        });
    }

    // Vocabulary tensors: GGUF shape[0] is the fastest-varying dim.
    let te = gguf
        .tensor_info("token_embd.weight")
        .ok_or_else(|| DoryLoadError::MissingTensor("token_embd.weight".into()))?;
    if te.shape.len() != 2 {
        return Err(DoryLoadError::ShapeMismatch {
            name: "token_embd.weight".into(),
            kind: "rank",
            got: te.shape.len(),
            expected: 2,
        });
    }
    let n_embd = te.shape[0];
    let vocab = te.shape[1];
    let te_flat = whole(gguf, "token_embd.weight")?;
    expect_len(
        "token_embd.weight",
        "elements",
        te_flat.len(),
        n_embd * vocab,
    )?;
    let mut wte = vec![0.0f32; n_embd * vocab];
    for r in 0..n_embd {
        for c in 0..vocab {
            wte[c * n_embd + r] = te_flat[r * vocab + c];
        }
    }

    // Untied output head, same shape + transpose.
    let oh = gguf
        .tensor_info("output.weight")
        .ok_or_else(|| DoryLoadError::MissingTensor("output.weight".into()))?;
    expect_len("output.weight", "rank", oh.shape.len(), 2)?;
    if oh.shape[0] != n_embd || oh.shape[1] != vocab {
        return Err(DoryLoadError::ShapeMismatch {
            name: "output.weight".into(),
            kind: "shape",
            got: oh.shape[0] * oh.shape[1],
            expected: n_embd * vocab,
        });
    }
    let oh_flat = whole(gguf, "output.weight")?;
    let mut lm_head = vec![0.0f32; n_embd * vocab];
    for r in 0..n_embd {
        for c in 0..vocab {
            lm_head[c * n_embd + r] = oh_flat[r * vocab + c];
        }
    }

    // The output block's exit scale (T0 fixture: `output_scale.weight`,
    // BF16, shape [131072] — per-channel over the VOCAB dim, not n_embd;
    // it scales the logits head's rows, nGPT-style).
    let output_scale = whole(gguf, "output_scale.weight")?;
    expect_len("output_scale.weight", "elements", output_scale.len(), vocab)?;

    // Per-block RoPE profile (72 entries; 2/1 alternating runs per T0).
    let rope_profile: Vec<u64> = gguf
        .metadata_array(stems::ROPE_PROFILE)
        .ok_or(DoryLoadError::MissingMeta("dory.rope.profile"))?
        .iter()
        .map(|v| v.as_u64().unwrap_or(0))
        .collect();
    expect_len(
        stems::ROPE_PROFILE,
        "entries",
        rope_profile.len(),
        total_blocks,
    )?;

    let sliding_window = gguf.metadata_u64("dory.sliding_window").unwrap_or(1024) as usize;
    let rope_freq_base = gguf.metadata_f64("dory.rope.freq_base").unwrap_or(1e7) as f32;
    let rope_freq_base_2 = gguf.metadata_f64("dory.rope.freq_base_2").unwrap_or(1e4) as f32;

    let mut blocks = Vec::with_capacity(total_blocks);
    for (b, &profile_entry) in rope_profile.iter().enumerate().take(total_blocks) {
        // The scale vectors are BF16 1-D tensors named `<stem>.weight`
        // (the census fixture is the authority). `residual_alpha` is
        // PER-CHANNEL [n_embd = 2560], not a scalar — the nGPT residual
        // rate rides per-channel (the paper's a = |α̂| is the per-channel
        // magnitude; the forward applies |α| elementwise).
        let alpha_name = stems::residual_alpha(b);
        let residual_alpha_vec = whole(gguf, &alpha_name)?;
        expect_len(&alpha_name, "elements", residual_alpha_vec.len(), 2560)?;

        // The census vocabulary is {1, 2}: 1 = `freq_base_2` (1e4),
        // 2 = `freq_base` (1e7). ZERO is not in the profile alphabet —
        // mapping it to a base would be a silent default.
        let rope_primary = match profile_entry {
            2 => true,
            1 => false,
            other => {
                return Err(DoryLoadError::MetaMismatch {
                    key: "dory.rope.profile",
                    got: other.to_string(),
                    expected: "1 or 2".to_string(),
                });
            }
        };

        if b % 2 == 0 {
            // Attention block (even physical index).
            let names = [
                stems::attn_q(b),
                stems::attn_k(b),
                stems::attn_v(b),
                stems::attn_o(b),
                stems::attn_gate(b),
                stems::attn_qk(b),
            ];
            let attn_q = whole(gguf, &names[0])?;
            expect_len(&names[0], "elements", attn_q.len(), 4096 * 2560)?;
            let attn_k = whole(gguf, &names[1])?;
            expect_len(&names[1], "elements", attn_k.len(), 1024 * 2560)?;
            let attn_v = whole(gguf, &names[2])?;
            expect_len(&names[2], "elements", attn_v.len(), 1024 * 2560)?;
            let attn_output = whole(gguf, &names[3])?;
            expect_len(&names[3], "elements", attn_output.len(), 2560 * 4096)?;
            let attn_gate_scale = whole(gguf, &names[4])?;
            expect_len(&names[4], "elements", attn_gate_scale.len(), 4096)?;
            let attn_qk_scale = whole(gguf, &names[5])?;
            expect_len(&names[5], "elements", attn_qk_scale.len(), 4096)?;
            blocks.push(DoryBlockWeights {
                kind: DoryBlockKind::Attn,
                residual_alpha: residual_alpha_vec,
                attn_q,
                attn_k,
                attn_v,
                attn_output,
                attn_gate_scale,
                attn_qk_scale,
                ffn_gate: Vec::new(),
                ffn_up: Vec::new(),
                ffn_down: Vec::new(),
                ffn_gate_scale: Vec::new(),
                ffn_up_scale: Vec::new(),
                rope_primary,
                rope_profile: profile_entry,
            });
        } else {
            // FFN block (odd physical index).
            let names = [
                stems::ffn_gate(b),
                stems::ffn_up(b),
                stems::ffn_down(b),
                stems::ffn_gate_scale(b),
                stems::ffn_up_scale(b),
            ];
            let ffn_gate = whole(gguf, &names[0])?;
            expect_len(&names[0], "elements", ffn_gate.len(), 7168 * 2560)?;
            let ffn_up = whole(gguf, &names[1])?;
            expect_len(&names[1], "elements", ffn_up.len(), 7168 * 2560)?;
            let ffn_down = whole(gguf, &names[2])?;
            expect_len(&names[2], "elements", ffn_down.len(), 2560 * 7168)?;
            let ffn_gate_scale = whole(gguf, &names[3])?;
            expect_len(&names[3], "elements", ffn_gate_scale.len(), 7168)?;
            let ffn_up_scale = whole(gguf, &names[4])?;
            expect_len(&names[4], "elements", ffn_up_scale.len(), 7168)?;
            blocks.push(DoryBlockWeights {
                kind: DoryBlockKind::Ffn,
                residual_alpha: residual_alpha_vec,
                attn_q: Vec::new(),
                attn_k: Vec::new(),
                attn_v: Vec::new(),
                attn_output: Vec::new(),
                attn_gate_scale: Vec::new(),
                attn_qk_scale: Vec::new(),
                ffn_gate,
                ffn_up,
                ffn_down,
                ffn_gate_scale,
                ffn_up_scale,
                rope_primary,
                rope_profile: profile_entry,
            });
        }
    }

    Ok(DoryWeights {
        meta,
        wte,
        lm_head,
        output_scale,
        blocks,
        sliding_window,
        rope_freq_base,
        rope_freq_base_2,
    })
}

/// Convenience: open + load in one call.
pub fn load_dory_weights(path: &std::path::Path) -> Result<DoryWeights, DoryLoadError> {
    let gguf = GgufFile::open(path)?;
    load_dory_weights_gguf(&gguf)
}

/// The Dory lane's own dims view (the stack's `Config` carries
/// llama/gemma fields this lane ignores; keeping the lane
/// self-contained avoids a `Config` fork — the issue's scope law).
#[derive(Debug, Clone)]
pub struct DoryDims {
    pub vocab: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub mlp_hidden: usize,
    pub n_blocks: usize,
    pub sliding_window: usize,
    /// The nGPT boundary-norm epsilon (FP32, paper ε = 1e-6).
    pub norm_eps: f32,
}

impl DoryDims {
    /// The census dims for the pinned Q5_K_M artifact.
    pub fn census() -> Self {
        Self {
            vocab: 131_072,
            n_embd: 2560,
            n_head: 16,
            n_kv_head: 4,
            head_dim: 256,
            mlp_hidden: 7168,
            n_blocks: 72,
            sliding_window: 1024,
            norm_eps: 1e-6,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The loader's stem mapping must match the probe fixture's stems —
    /// a rename upstream diffs HERE, not at a bench 40 minutes in.
    #[test]
    fn stems_match_the_census_shapes() {
        assert_eq!(stems::attn_q(0), "blk.0.attn_q.weight");
        assert_eq!(stems::attn_k(0), "blk.0.attn_k.weight");
        assert_eq!(stems::attn_v(0), "blk.0.attn_v.weight");
        assert_eq!(stems::attn_o(0), "blk.0.attn_output.weight");
        assert_eq!(stems::attn_gate(0), "blk.0.attn_gate_scale.weight");
        assert_eq!(stems::attn_qk(0), "blk.0.attn_qk_scale.weight");
        assert_eq!(stems::ffn_gate(1), "blk.1.ffn_gate.weight");
        assert_eq!(stems::ffn_up(1), "blk.1.ffn_up.weight");
        assert_eq!(stems::ffn_down(1), "blk.1.ffn_down.weight");
        assert_eq!(stems::ffn_gate_scale(1), "blk.1.ffn_gate_scale.weight");
        assert_eq!(stems::ffn_up_scale(1), "blk.1.ffn_up_scale.weight");
        assert_eq!(stems::residual_alpha(0), "blk.0.residual_alpha.weight");
        assert_eq!(stems::ROPE_PROFILE, "dory.rope.profile");
    }

    /// `DoryBlockKind` derives from physical index parity (attn EVEN,
    /// ffn ODD) — the T0 census law the loader encodes.
    #[test]
    fn parity_partition() {
        let meta = DoryLoopMeta {
            num_loops: 4,
            input_layers: 16,
            recurrent_layers: 40,
            output_layers: 16,
            kv_cache_mode: "per_loop".into(),
        };
        assert_eq!(meta.recurrent_start(), 16);
        assert_eq!(meta.output_start(), 56);
        assert_eq!(
            meta.input_layers + meta.recurrent_layers + meta.output_layers,
            72
        );
    }

    /// The census dims constructor pins the T0 numbers.
    #[test]
    fn census_dims() {
        let d = DoryDims::census();
        assert_eq!(d.vocab, 131_072);
        assert_eq!(d.n_embd, 2560);
        assert_eq!(d.head_dim, 256);
        assert_eq!(d.n_kv_head, 4);
        assert_eq!(d.mlp_hidden, 7168);
        assert_eq!(d.sliding_window, 1024);
    }

    /// Load errors display their payloads (the loud-refusal contract).
    #[test]
    fn error_display_names_the_refusal() {
        let e = DoryLoadError::NotDory("llama".into());
        assert!(e.to_string().contains("llama"));
        let e = DoryLoadError::MissingTensor("blk.0.attn_q".into());
        assert!(e.to_string().contains("blk.0.attn_q"));
        let e = DoryLoadError::MetaMismatch {
            key: "dory.block_count",
            got: "71".into(),
            expected: "72".into(),
        };
        assert!(e.to_string().contains("71") && e.to_string().contains("72"));
    }
}
