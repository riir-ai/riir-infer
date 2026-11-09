//! Issue 1183 — the DSpine b16 draft model (tencent/Qwen3.8-27B-DSpine-b16):
//! the CPU oracle — loader + block forward + code bridge + serial refinement
//! tail, op-for-op the vendor reference at `.raw/dspine` pin
//! `6450af726375cb3c6bac4de95d98ebc8df99fbd4` (the `dspine_mlx/` MLX tree is
//! the forward template; the sglang Triton kernels carry the numerics
//! contract). The PROGRESS block in `.issues/1183_dspine_b16_draft_model_support.md`
//! is the ground-truth record this module implements.
//!
//! # The contract (anchored to the vendor source)
//!
//! **Checkpoint**: single `model.safetensors` 4,051,831,880 bytes, 71 BF16
//! tensors (+4 I64 version scalars), blake3
//! `53d185164e20b6ea51fd72a3f093492aff1596a7dda0693d45c7d057dc1cc15c` —
//! verified by the harness bin at load (the A1/A8 weights-from-disk law;
//! never committed).
//!
//! **Config** (HF `config.json`): 5 layers, hidden 5120, 32 Q / 8 KV heads,
//! head_dim 128, FFN 17408 (SwiGLU), vocab 248320, rms_eps 1e-6, RoPE NEOX
//! half-split @ theta 1e7, block_size 16, **all 5 draft layers
//! sliding_attention @ W=8192** (`validate_native_attention_config` refuses
//! anything else — the native posture, not optional), `is_causal` +
//! `draft_block_causal` true (causal INSIDE the block, every context row
//! visible), bridge {code_size 1024, rank 512, eps 1e-6, kappa 10.0},
//! head_topk 16, mask_token_id 248070, target_layer_ids [1, 16, 31, 46, 61],
//! terminal DISABLED (`selected_prefix_terminal: false` — the loader refuses
//! a nonzero `out.weight` and any terminal-residual rank).
//!
//! **Forward per draft round** (`dspine_mlx/model.py::DSpineDraftModel`):
//! block tokens `[anchor, MASK × (T-1)]` embedded from the TARGET's embed
//! table; per layer: causal block attention over `[context KV ∪ block KV]`
//! (SWA `q − k >= 8192` masks stale context, scale 1/√128, per-head q/k
//! RMSNorm, NEOX RoPE at absolute positions) → residual → MLP(SwiGLU) →
//! residual; then the **code-bridge write after EVERY layer** (fused next
//! norm): `raw = hidden` (the layer already summed its residual), `rms`,
//! `normalized = raw/(rms+eps)` (gamma-free), `projected` via the layer's
//! read weight (`early_read_weights[0..3]`, `read_weight` for the last),
//! **codes**: the anchor row (row 0 of each block) takes
//! `sqrt(C)·codebook[anchor]`, rows 1..T-1 take `sqrt(C)·projected/‖proj‖`
//! (continuous L2-normalized projections, NOT codebook rows); `packed =
//! codes @ [gate; predecessor_gate; prev]ᵀ` → q|s|v halves; the gated
//! adjacent injection `message = sigmoid(q + s_prev + b)·v_prev`, zeroed at
//! the anchor row; `delta = message @ write_weights[layer]ᵀ`;
//! `out = bf16(f + rms·delta·(non-anchor))` — the write lives in RMS units;
//! the next layer's input norm (or the model norm at the last layer) is
//! applied INSIDE the write. The LAST write stashes `{raw, rms, packed_q}`
//! for the tail.
//!
//! **Serial refinement tail** (`worker.py::DSpineDraftSampler.propose` +
//! `kernels/dspine.py::_serial_table/_serial_chain` — `exact_sender`,
//! `code_bridge_second_pass` serial_table, `first_exact = 4`): candidates =
//! top-16 of `rows @ lm_headᵀ` (stable: value desc, lower id wins ties,
//! ids ASCENDING); offline sender tables `TG[v] = bf16((lm_head[v]·norm_w))
//! @ W_last` `[V,512]` and `TSV[v] = [G_p; A] @ (sqrt(C)·codebook[v])`
//! `[V,1024]`; `m[n,j] = sigmoid(q[n] + TSV[sid,:512] + gate_b)·TSV[sid,512:]`
//! with `sid[n,j] = cand[n−1,j]` (row 0: the anchor token); score
//! `base[n,j'] + rms[n]·(m·TG[cand[j']]ᵀ)` — the CANDIDATE lane's base
//! (both vendor references broadcast `base[None,:]` over the sender axis);
//! per-lane argmax (first-index
//! ties) → the greedy chain walk (each row's chosen lane conditions the next
//! row's table read — predecessor-conditioned decoding); confidence
//! `1/Σexp((s−best)/max(rms,1e-6))` at the previous chosen lane.
//!
//! **Numerics** (the vendor runs the draft in BF16 storage): every matmul
//! output and norm/activation boundary is bf16-rounded (`bf16r`); matmuls
//! accumulate f32 over bf16-rounded inputs. The Triton kernels run
//! `enable_fp_fusion=False` and the sampler refuses TF32 — this oracle pins
//! its own fold orders (sequential, ascending index) per the Issue-940 law.
//! Cross-precision disclosure: the vendor target ran BF16; OURS is the
//! Q4_K_M GGUF league fixture (the lm_head readout dequantizes Q6_K rows) —
//! the same cross-posture class the 1179 T1 cell disclosed.
//!
//! **ReplaySSM is NOT here**: the draft is pure attention+MLP (no SSM
//! tensors in the checkpoint); the target GDN state replay at the accept
//! boundary is the 1179 verify loop's half. Feature `qwen38_dspine`
//! (implies `qwen38_specdec`); feature-off compiles to nothing.

use std::collections::VecDeque;
use std::path::Path;

use half::bf16;
use rayon::prelude::*;

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────

/// The DSpine b16 draft configuration (the 27B checkpoint's values; every
/// field validated by [`parse_config_json`] before any weight is read).
#[derive(Debug, Clone)]
pub struct DspineConfig {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    /// The TRAINED block size (16); serve via `set_block_size` (2..=this).
    pub trained_block_size: usize,
    pub sliding_window: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
    pub code_size: usize,
    pub rank: usize,
    pub bridge_eps: f32,
    pub vocab_size: usize,
    pub mask_token_id: u32,
    pub head_topk: usize,
    /// The TARGET layers whose hidden states concat into the fc input
    /// (ascending, duplicates refused); the caller supplies rows in this
    /// order.
    pub target_layer_ids: Vec<usize>,
    pub num_target_layers: usize,
}

impl Default for DspineConfig {
    fn default() -> Self {
        Self {
            n_layer: 5,
            n_embd: 5120,
            n_head: 32,
            n_kv_head: 8,
            head_dim: 128,
            n_ff: 17408,
            trained_block_size: 16,
            sliding_window: 8192,
            rope_theta: 1.0e7,
            rms_eps: 1.0e-6,
            code_size: 1024,
            rank: 512,
            bridge_eps: 1.0e-6,
            vocab_size: 248_320,
            mask_token_id: 248_070,
            head_topk: 16,
            target_layer_ids: vec![1, 16, 31, 46, 61],
            num_target_layers: 64,
        }
    }
}

/// Parse + validate the checkpoint's `config.json` against the DSpine
/// contract (the vendor loader guards, mirrored: unsupported keys REFUSE
/// loud — the anti-silent-fallback law).
pub fn parse_config_json(raw: &str) -> Result<DspineConfig, String> {
    let v: serde_json::Value =
        serde_json::from_str(raw).map_err(|e| format!("config json: {e}"))?;
    let get = |path: &[&str]| -> Option<&serde_json::Value> {
        let mut cur = &v;
        for p in path {
            cur = cur.get(p)?;
        }
        Some(cur)
    };
    let as_usize = |x: &serde_json::Value, what: &str| -> Result<usize, String> {
        x.as_u64()
            .map(|u| u as usize)
            .ok_or_else(|| format!("config: {what} not a u64"))
    };
    let as_f32 = |x: &serde_json::Value, what: &str| -> Result<f32, String> {
        x.as_f64()
            .map(|f| f as f32)
            .ok_or_else(|| format!("config: {what} not an f64"))
    };
    let dflash = get(&["dflash_config"]).ok_or("config: missing dflash_config")?;

    let n_layer = as_usize(
        v.get("num_hidden_layers")
            .ok_or("config: num_hidden_layers")?,
        "num_hidden_layers",
    )?;
    let n_embd = as_usize(
        v.get("hidden_size").ok_or("config: hidden_size")?,
        "hidden_size",
    )?;
    let n_head = as_usize(
        v.get("num_attention_heads").ok_or("config: heads")?,
        "heads",
    )?;
    let n_kv_head = as_usize(
        v.get("num_key_value_heads").ok_or("config: kv heads")?,
        "kv heads",
    )?;
    let head_dim = as_usize(v.get("head_dim").ok_or("config: head_dim")?, "head_dim")?;
    let n_ff = as_usize(
        v.get("intermediate_size")
            .ok_or("config: intermediate_size")?,
        "intermediate",
    )?;
    let vocab = as_usize(v.get("vocab_size").ok_or("config: vocab_size")?, "vocab")?;
    if n_head % n_kv_head != 0 {
        return Err(format!(
            "config: heads {n_head} not divisible by kv {n_kv_head}"
        ));
    }

    // The native SWA posture (validate_native_attention_config, mirrored):
    // exactly `n_layer` sliding_attention entries, W == 8192, explicit causal.
    let layer_types = v
        .get("layer_types")
        .and_then(|x| x.as_array())
        .ok_or("config: layer_types")?;
    if layer_types.len() != n_layer
        || !layer_types
            .iter()
            .all(|t| t.as_str() == Some("sliding_attention"))
    {
        return Err(format!(
            "config: native DSpine requires {n_layer} sliding_attention layers, got {layer_types:?}"
        ));
    }
    let sw = as_usize(
        v.get("sliding_window").ok_or("config: sliding_window")?,
        "sliding_window",
    )?;
    let use_swa = v
        .get("use_sliding_window")
        .and_then(|x| x.as_bool())
        .unwrap_or(false);
    if !use_swa {
        return Err("config: use_sliding_window must be true (native posture)".into());
    }
    if v.get("is_causal").and_then(|x| x.as_bool()) != Some(true) {
        return Err("config: is_causal must be true (draft_block_causal posture)".into());
    }

    let block = as_usize(
        v.get("block_size").ok_or("config: block_size")?,
        "block_size",
    )?;
    let rope_theta = if let Some(x) = v.get("rope_theta") {
        as_f32(x, "rope_theta")?
    } else {
        let rp = v
            .get("rope_parameters")
            .ok_or("config: rope_theta missing and no rope_parameters")?;
        as_f32(
            rp.get("rope_theta")
                .ok_or("config: rope_parameters.rope_theta")?,
            "rope_theta",
        )?
    };
    let rms_eps = as_f32(
        v.get("rms_norm_eps").ok_or("config: rms_norm_eps")?,
        "rms_norm_eps",
    )?;
    let num_target_layers = as_usize(
        v.get("num_target_layers")
            .ok_or("config: num_target_layers")?,
        "num_target_layers",
    )?;

    // The dflash_config contract (DSpineCodeBridge.__init__ + the MLX
    // config.py requires, mirrored).
    let expect_flag = |key: &str, want: bool| -> Result<(), String> {
        let got = dflash
            .get(key)
            .and_then(|x| x.as_bool())
            .ok_or_else(|| format!("config: dflash_config.{key} missing"))?;
        if got != want {
            return Err(format!(
                "config: dflash_config.{key} = {got}, expected {want}"
            ));
        }
        Ok(())
    };
    expect_flag("selected_prefix_terminal", false)?;
    expect_flag("selected_prefix_feature_writes", true)?;
    expect_flag("draft_block_causal", true)?;
    if dflash
        .get("selected_prefix_feature_write_mode")
        .and_then(|x| x.as_str())
        != Some("independent")
    {
        return Err("config: selected_prefix_feature_write_mode must be independent".into());
    }
    if dflash
        .get("selected_prefix_decode_mode")
        .and_then(|x| x.as_str())
        != Some("serial")
    {
        return Err("config: selected_prefix_decode_mode must be serial".into());
    }
    for banned in [
        "conv_kernel_size",
        "conv_group_size",
        "selector_rank",
        "selector_top_k",
        "projector_type",
        "attention_sink_bias",
        "attention_value_scale",
        "selected_prefix_feature_reliability",
        "selected_prefix_feature_next_token_rows",
    ] {
        if dflash.get(banned).is_some_and(|x| !x.is_null()) {
            return Err(format!(
                "config: dflash_config.{banned} unsupported (DSpine carries none)"
            ));
        }
    }
    if v.get("has_embed_tokens")
        .is_some_and(|x| x.as_bool().unwrap_or(false))
    {
        return Err("config: has_embed_tokens unsupported".into());
    }

    let bridge = dflash
        .get("selected_prefix_code_bridge")
        .ok_or("config: missing selected_prefix_code_bridge")?;
    let code_size = as_usize(
        bridge.get("code_size").ok_or("config: bridge code_size")?,
        "code_size",
    )?;
    let rank = as_usize(bridge.get("rank").ok_or("config: bridge rank")?, "rank")?;
    let bridge_eps = as_f32(bridge.get("eps").ok_or("config: bridge eps")?, "bridge eps")?;

    let feature_layers: Vec<usize> = dflash
        .get("selected_prefix_feature_layers")
        .and_then(|x| x.as_array())
        .ok_or("config: selected_prefix_feature_layers")?
        .iter()
        .map(|x| as_usize(x, "feature layer").map(|u| u - 1)) // HF is 1-based; 0-based here
        .collect::<Result<_, _>>()?;
    if feature_layers != (0..n_layer).collect::<Vec<_>>() {
        return Err(format!(
            "config: every draft layer must write (feature_layers {feature_layers:?} != 0..{n_layer})"
        ));
    }

    // The refinement width: BOTH config surfaces must agree (the vendor
    // resolve_dspine_head_topk discipline).
    let mut found: Vec<usize> = Vec::new();
    let refine = dflash
        .get("selected_prefix_last_write_refine")
        .ok_or("config: missing selected_prefix_last_write_refine")?;
    if refine.get("kind").and_then(|x| x.as_str()) != Some("exact_sender") {
        return Err("config: refine kind must be exact_sender".into());
    }
    let rv = refine
        .get("version")
        .map(|x| x.as_i64().unwrap_or(0))
        .unwrap_or(1);
    if rv != 1 {
        return Err(format!("config: refine version {rv} unsupported"));
    }
    found.push(as_usize(
        refine.get("head_topk").ok_or("config: refine head_topk")?,
        "head_topk",
    )?);
    if let Some(second) = v.get("code_bridge_second_pass") {
        if second.get("mode").and_then(|x| x.as_str()) != Some("serial_table") {
            return Err("config: second_pass mode must be serial_table".into());
        }
        let first_exact = as_usize(
            second.get("first_exact").ok_or("config: first_exact")?,
            "first_exact",
        )?;
        if first_exact + 1 != n_layer {
            return Err(format!(
                "config: serial_table redoes the LAST write only (first_exact {first_exact} != {})",
                n_layer - 1
            ));
        }
        found.push(as_usize(
            second.get("head_topk").ok_or("config: second head_topk")?,
            "head_topk",
        )?);
    }
    let uniq: std::collections::BTreeSet<usize> = found.iter().copied().collect();
    if uniq.len() != 1 {
        return Err(format!("config: refinement widths disagree: {found:?}"));
    }
    let head_topk = found[0];
    if !head_topk.is_power_of_two() || !(2..=64).contains(&head_topk) {
        return Err(format!(
            "config: head_topk {head_topk} not a power of two in [2, 64]"
        ));
    }

    let mask = as_usize(
        dflash
            .get("mask_token_id")
            .ok_or("config: dflash mask_token_id")?,
        "mask_token_id",
    )?;
    let target_layer_ids: Vec<usize> = dflash
        .get("target_layer_ids")
        .and_then(|x| x.as_array())
        .ok_or("config: target_layer_ids")?
        .iter()
        .map(|x| as_usize(x, "target_layer_ids entry"))
        .collect::<Result<_, _>>()?;
    if target_layer_ids.is_empty() || !target_layer_ids.iter().all(|&l| l < num_target_layers) || {
        let mut s = target_layer_ids.clone();
        s.sort_unstable();
        s.dedup();
        s.len() != target_layer_ids.len()
    } {
        return Err(format!(
            "config: target_layer_ids {target_layer_ids:?} invalid for {num_target_layers} layers"
        ));
    }

    let cfg = DspineConfig {
        n_layer,
        n_embd,
        n_head,
        n_kv_head,
        head_dim,
        n_ff,
        trained_block_size: block,
        sliding_window: sw,
        rope_theta,
        rms_eps,
        code_size,
        rank,
        bridge_eps,
        vocab_size: vocab,
        mask_token_id: mask as u32,
        head_topk,
        target_layer_ids,
        num_target_layers,
    };
    if n_embd % 512 != 0 || rank % 128 != 0 {
        // The vendor serial_table size contract (H % 512, RK % 128).
        return Err(format!(
            "config: H {n_embd} or rank {rank} off the serial-table grid"
        ));
    }
    if cfg.sliding_window == 0 {
        return Err("config: sliding_window 0".into());
    }
    Ok(cfg)
}

// ─────────────────────────────────────────────────────────────────────────────
// Weights + loader
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Default, Clone)]
pub struct DspineLayer {
    pub input_norm: Vec<f32>,
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub o_proj: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub gate_proj: Vec<f32>,
    pub up_proj: Vec<f32>,
    pub down_proj: Vec<f32>,
}

#[derive(Debug, Default, Clone)]
pub struct DspineBridge {
    /// `[V, code_size]` — near the target's output-embedding space (the
    /// unified transfer space, the paper's term).
    pub codebook: Vec<f32>,
    /// Layers 0..n-1 read weights: `early_read_weights[0..n-1]` then
    /// `read_weight`.
    pub early_read_weights: Vec<Vec<f32>>,
    pub read_weight: Vec<f32>,
    /// One write per layer (`[H, rank]` row-major).
    pub feature_write_weights: Vec<Vec<f32>>,
    pub gate_w: Vec<f32>,
    pub gate_b: Vec<f32>,
    pub pred_gate_w: Vec<f32>,
    pub prev_w: Vec<f32>,
    /// Terminal projection — must load all-zero (terminal-disabled).
    pub out_w: Vec<f32>,
}

#[derive(Debug, Default, Clone)]
pub struct DspineWeights {
    /// `[5120, 5×5120]` — the target-context fusion.
    pub fc: Vec<f32>,
    pub hidden_norm: Vec<f32>,
    pub model_norm: Vec<f32>,
    pub layers: Vec<DspineLayer>,
    pub bridge: DspineBridge,
}

impl DspineWeights {
    /// `packed_weight` rows: `[gate_w; pred_gate_w; prev_w]` — the vendor
    /// `prepare()` concat. `[3·rank, code_size]`.
    pub fn packed_weight(&self) -> Vec<f32> {
        let mut w = Vec::with_capacity(3 * self.bridge.gate_w.len());
        w.extend_from_slice(&self.bridge.gate_w);
        w.extend_from_slice(&self.bridge.pred_gate_w);
        w.extend_from_slice(&self.bridge.prev_w);
        w
    }

    /// `[2·rank, code_size]` — the `[G_p; A]` half (the TSV builder's pv_w).
    pub fn pv_weight(&self) -> Vec<f32> {
        let mut w = Vec::with_capacity(2 * self.bridge.pred_gate_w.len());
        w.extend_from_slice(&self.bridge.pred_gate_w);
        w.extend_from_slice(&self.bridge.prev_w);
        w
    }

    /// The read weight for layer `l` (early for 0..n-1, the last-layer
    /// `read_weight` for n-1).
    pub fn read_weight_for(&self, l: usize) -> &Vec<f32> {
        let b = &self.bridge;
        if l + 1 == self.layers.len() {
            &b.read_weight
        } else {
            &b.early_read_weights[l]
        }
    }
}

// ── minimal streaming safetensors reader (the dflash2 oracle's pattern —
//    per-tensor seek + read, no whole-file blob) ──

struct StFile {
    file: std::fs::File,
    index: std::collections::HashMap<String, (String, Vec<usize>, u64)>,
}

impl StFile {
    fn open(path: &Path) -> Result<Self, String> {
        use std::io::Read;
        let mut file = std::fs::File::open(path).map_err(|e| format!("open {path:?}: {e}"))?;
        let mut len_buf = [0u8; 8];
        file.read_exact(&mut len_buf)
            .map_err(|e| format!("header len: {e}"))?;
        let hlen = u64::from_le_bytes(len_buf) as usize;
        let mut hdr = vec![0u8; hlen];
        file.read_exact(&mut hdr)
            .map_err(|e| format!("header: {e}"))?;
        let parsed: serde_json::Value =
            serde_json::from_slice(&hdr).map_err(|e| format!("header json: {e}"))?;
        let obj = parsed
            .as_object()
            .ok_or_else(|| "header not an object".to_string())?;
        let data_base = 8 + hlen as u64;
        let mut index = std::collections::HashMap::new();
        for (name, meta) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype = meta
                .get("dtype")
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("{name}: no dtype"))?
                .to_string();
            let shape: Vec<usize> = meta
                .get("shape")
                .and_then(|v| v.as_array())
                .ok_or_else(|| format!("{name}: no shape"))?
                .iter()
                .map(|x| x.as_u64().unwrap_or(0) as usize)
                .collect();
            let beg = meta
                .get("data_offsets")
                .and_then(|v| v.as_array())
                .and_then(|a| a.first())
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            index.insert(name.clone(), (dtype, shape, data_base + beg));
        }
        Ok(Self { file, index })
    }

    fn expect(&self, name: &str, want: &[usize]) -> Result<(), String> {
        let got = self
            .index
            .get(name)
            .map(|(_, s, _)| s.clone())
            .ok_or_else(|| format!("tensor {name} not in file"))?;
        if got != want {
            return Err(format!("{name}: shape {got:?} != expected {want:?}"));
        }
        Ok(())
    }

    fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    fn read_f32(&mut self, name: &str) -> Result<Vec<f32>, String> {
        use std::io::{Read, Seek, SeekFrom};
        let (dtype, shape, off) = self
            .index
            .get(name)
            .cloned()
            .ok_or_else(|| format!("tensor {name} not in file"))?;
        let n: usize = shape.iter().product();
        self.file
            .seek(SeekFrom::Start(off))
            .map_err(|e| format!("seek {name}: {e}"))?;
        match dtype.as_str() {
            "BF16" => {
                let mut raw = vec![0u8; n * 2];
                self.file
                    .read_exact(&mut raw)
                    .map_err(|e| format!("read {name}: {e}"))?;
                Ok(raw
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| f32::from(bf16::from_le_bytes([c[0], c[1]])))
                    .collect())
            }
            "F32" => {
                let mut raw = vec![0u8; n * 4];
                self.file
                    .read_exact(&mut raw)
                    .map_err(|e| format!("read {name}: {e}"))?;
                Ok(raw
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]])))
                    .collect())
            }
            "I64" => Err(format!("{name}: I64 (read via read_i64)")),
            other => Err(format!("{name}: unsupported dtype {other}")),
        }
    }

    fn read_i64(&mut self, name: &str) -> Result<i64, String> {
        use std::io::{Read, Seek, SeekFrom};
        let (dtype, _, off) = self
            .index
            .get(name)
            .cloned()
            .ok_or_else(|| format!("tensor {name} not in file"))?;
        if dtype != "I64" {
            return Err(format!("{name}: expected I64, got {dtype}"));
        }
        self.file
            .seek(SeekFrom::Start(off))
            .map_err(|e| format!("seek {name}: {e}"))?;
        let mut raw = [0u8; 8];
        self.file
            .read_exact(&mut raw)
            .map_err(|e| format!("read {name}: {e}"))?;
        Ok(i64::from_le_bytes(raw))
    }
}

/// Load the DSpine draft from `model.safetensors` (every shape validated
/// against the config BEFORE reading; version scalars must all be 1; the
/// terminal projection must be all-zero).
pub fn load_dspine_safetensors(path: &Path, cfg: &DspineConfig) -> Result<DspineWeights, String> {
    let mut st = StFile::open(path)?;
    let (e, ff, hd) = (cfg.n_embd, cfg.n_ff, cfg.head_dim);
    let (qd, kvd) = (cfg.n_head * hd, cfg.n_kv_head * hd);
    let (c, r) = (cfg.code_size, cfg.rank);
    let v = cfg.vocab_size;
    let n = cfg.n_layer;

    // Version scalars first (the vendor refuses ≠1 loud).
    for name in [
        "draft_block_causal_version",
        "last_write_refine_version",
        "selected_prefix_bridge.independent_write_version",
        "selected_prefix_bridge.terminal_disabled_version",
    ] {
        let got = st.read_i64(name)?;
        if got != 1 {
            return Err(format!("{name}: version {got} unsupported (expect 1)"));
        }
    }
    // No DFlash2-class leftovers (the vendor refuses them loud).
    for prefix in ["prefix_gru.", "embed_proj.", "candidate_selector."] {
        if st.index.keys().any(|k| k.starts_with(prefix)) {
            return Err(format!(
                "checkpoint carries {prefix}* tensors (not a DSpine draft)"
            ));
        }
    }
    if st.contains("selected_prefix_bridge.terminal_residual_embedding") {
        return Err("checkpoint carries a terminal residual (terminal-disabled only)".into());
    }

    st.expect("fc.weight", &[e, 5 * e])?;
    st.expect("hidden_norm.weight", &[e])?;
    st.expect("norm.weight", &[e])?;
    st.expect("selected_prefix_bridge.codebook", &[v, c])?;
    st.expect("selected_prefix_bridge.early_read_weights", &[n - 1, c, e])?;
    st.expect("selected_prefix_bridge.read_weight", &[c, e])?;
    st.expect("selected_prefix_bridge.feature_write_weights", &[n, e, r])?;
    st.expect("selected_prefix_bridge.gate.weight", &[r, c])?;
    st.expect("selected_prefix_bridge.gate.bias", &[r])?;
    st.expect("selected_prefix_bridge.predecessor_gate.weight", &[r, c])?;
    st.expect("selected_prefix_bridge.prev.weight", &[r, c])?;
    st.expect("selected_prefix_bridge.out.weight", &[c, r])?;

    let mut layers = Vec::with_capacity(n);
    for i in 0..n {
        let p = |s: &str| format!("layers.{i}.{s}");
        st.expect(&p("input_layernorm.weight"), &[e])?;
        st.expect(&p("self_attn.q_proj.weight"), &[qd, e])?;
        st.expect(&p("self_attn.k_proj.weight"), &[kvd, e])?;
        st.expect(&p("self_attn.v_proj.weight"), &[kvd, e])?;
        st.expect(&p("self_attn.o_proj.weight"), &[e, qd])?;
        st.expect(&p("self_attn.q_norm.weight"), &[hd])?;
        st.expect(&p("self_attn.k_norm.weight"), &[hd])?;
        st.expect(&p("post_attention_layernorm.weight"), &[e])?;
        st.expect(&p("mlp.gate_proj.weight"), &[ff, e])?;
        st.expect(&p("mlp.up_proj.weight"), &[ff, e])?;
        st.expect(&p("mlp.down_proj.weight"), &[e, ff])?;
        layers.push(DspineLayer {
            input_norm: st.read_f32(&p("input_layernorm.weight"))?,
            q_proj: st.read_f32(&p("self_attn.q_proj.weight"))?,
            k_proj: st.read_f32(&p("self_attn.k_proj.weight"))?,
            v_proj: st.read_f32(&p("self_attn.v_proj.weight"))?,
            o_proj: st.read_f32(&p("self_attn.o_proj.weight"))?,
            q_norm: st.read_f32(&p("self_attn.q_norm.weight"))?,
            k_norm: st.read_f32(&p("self_attn.k_norm.weight"))?,
            ffn_norm: st.read_f32(&p("post_attention_layernorm.weight"))?,
            gate_proj: st.read_f32(&p("mlp.gate_proj.weight"))?,
            up_proj: st.read_f32(&p("mlp.up_proj.weight"))?,
            down_proj: st.read_f32(&p("mlp.down_proj.weight"))?,
        });
    }

    let out_w = st.read_f32("selected_prefix_bridge.out.weight")?;
    if out_w.iter().any(|&x| x != 0.0) {
        return Err(
            "terminal-disabled checkpoint contains a nonzero terminal projection (out.weight)"
                .into(),
        );
    }

    let early_raw = st.read_f32("selected_prefix_bridge.early_read_weights")?;
    let mut early_read_weights = Vec::with_capacity(n - 1);
    for i in 0..n - 1 {
        early_read_weights.push(early_raw[i * c * e..(i + 1) * c * e].to_vec());
    }
    let writes_raw = st.read_f32("selected_prefix_bridge.feature_write_weights")?;
    let feature_write_weights = (0..n)
        .map(|i| writes_raw[i * e * r..(i + 1) * e * r].to_vec())
        .collect();

    Ok(DspineWeights {
        fc: st.read_f32("fc.weight")?,
        hidden_norm: st.read_f32("hidden_norm.weight")?,
        model_norm: st.read_f32("norm.weight")?,
        layers,
        bridge: DspineBridge {
            codebook: st.read_f32("selected_prefix_bridge.codebook")?,
            early_read_weights,
            read_weight: st.read_f32("selected_prefix_bridge.read_weight")?,
            feature_write_weights,
            gate_w: st.read_f32("selected_prefix_bridge.gate.weight")?,
            gate_b: st.read_f32("selected_prefix_bridge.gate.bias")?,
            pred_gate_w: st.read_f32("selected_prefix_bridge.predecessor_gate.weight")?,
            prev_w: st.read_f32("selected_prefix_bridge.prev.weight")?,
            out_w,
        },
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Numerics helpers (the vendor bf16 pipeline, made explicit)
// ─────────────────────────────────────────────────────────────────────────────

/// One bf16 round-trip (RNE) — every vendor bf16 store/load boundary.
#[inline]
fn bf16r(v: f32) -> f32 {
    f32::from(bf16::from_f32(v))
}

/// Pub bf16 round for the fixture harness (bf16-valued synthetics).
pub fn bf16_round(v: f32) -> f32 {
    bf16r(v)
}

/// Sequential f32 dot (the pinned fold order — ascending index).
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

/// bf16 GEMV: `dst[o] = bf16(Σ_i w[o·in+i]·x[i])` — inputs are ALREADY
/// bf16-valued (callers round first); f32 accumulation, bf16-rounded output
/// (the torch/MLX bf16 matmul contract).
fn gemv_bf16(w: &[f32], x: &[f32], out_len: usize) -> Vec<f32> {
    let inn = x.len();
    debug_assert_eq!(w.len(), out_len * inn);
    let mut dst = vec![0.0f32; out_len];
    for (o, d) in dst.iter_mut().enumerate() {
        let row = &w[o * inn..(o + 1) * inn];
        let mut acc = 0.0f32;
        for (i, &v) in x.iter().enumerate() {
            acc += row[i] * v;
        }
        *d = bf16r(acc);
    }
    dst
}

/// Standard RMSNorm with gamma (`x · rsqrt(mean(x²)+eps) · weight`), output
/// bf16-rounded (the MLX nn.RMSNorm form; the q/k head norms, layer norms,
/// hidden_norm and model norm all use it — the finish norm too).
fn rmsnorm_std(x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mut ss = 0.0f32;
    for &v in x {
        ss += v * v;
    }
    let inv = 1.0 / (ss / n as f32 + eps).sqrt();
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        out[i] = bf16r(x[i] * inv * gamma[i]);
    }
    out
}

/// The bridge read norm — gamma-free `x/(rms+eps)` with the vendor's
/// F32-tiny floor on the rms, output bf16-rounded. Returns (rms, normalized).
fn bridge_read_norm(x: &[f32], eps: f32) -> (f32, Vec<f32>) {
    let n = x.len();
    let mut ss = 0.0f32;
    for &v in x {
        ss += v * v;
    }
    // The Triton kernel's floor: f32::MIN_POSITIVE (1.1754944e-38, the
    // shortest repr of the vendor's 1.1754943508222875e-38 literal).
    let rms = (ss / n as f32).max(f32::MIN_POSITIVE).sqrt();
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        out[i] = bf16r(x[i] / (rms + eps));
    }
    (rms, out)
}

/// NEOX half-split RoPE in place (pairs `(i, i + np)`; angle = pos · freqs).
fn apply_rope(v: &mut [f32], pos: usize, freqs: &[f32]) {
    let np = freqs.len();
    for (i, &f) in freqs.iter().enumerate() {
        let ang = pos as f32 * f;
        let (s, c) = (ang.sin(), ang.cos());
        let a = v[i];
        let b = v[i + np];
        v[i] = a * c - b * s;
        v[i + np] = a * s + b * c;
    }
}

/// Top-`k` token ids by logit, STABLE (value desc, lower id wins ties),
/// returned ASCENDING (the vendor's chunked-stable contract — its docstring
/// pins equality with the full stable argsort's selected ids).
fn top_k_ids_ascending(logits: &[f32], k: usize) -> Vec<u32> {
    let mut best_val: Vec<f32> = Vec::with_capacity(k);
    let mut best_id: Vec<u32> = Vec::with_capacity(k);
    for (i, &v) in logits.iter().enumerate() {
        let insert = best_val.len() < k
            || v > *best_val.last().unwrap()
            || (v == *best_val.last().unwrap() && i < *best_id.last().unwrap() as usize);
        if insert {
            // position after all strictly-better entries; ties keep the
            // earlier id first (partition on (value desc, id asc)).
            let mut p = 0usize;
            while p < best_val.len()
                && (best_val[p] > v || (best_val[p] == v && best_id[p] < i as u32))
            {
                p += 1;
            }
            best_val.insert(p, v);
            best_id.insert(p, i as u32);
            if best_val.len() > k {
                best_val.pop();
                best_id.pop();
            }
        }
    }
    best_id.sort_unstable();
    best_id
}

// ─────────────────────────────────────────────────────────────────────────────
// Serial sender tables (the offline TG/TSV build)
// ─────────────────────────────────────────────────────────────────────────────

/// The offline sender tables: `TG [V, rank]`, `TSV [V, 2·rank]` —
/// `TG[v]  = bf16((lm_head[v] · norm_w)) @ W_last`
/// `TSV[v] = [G_p; A] @ bf16(sqrt(C) · codebook[v])`
/// (the vendor `build_serial_tables`, bf16 store + f32 accumulate).
#[derive(Debug, Clone)]
pub struct SerialTables {
    pub tg: Vec<f32>,
    pub tsv: Vec<f32>,
}

pub fn build_serial_tables(
    lm_head: &[f32],
    norm_w: &[f32],
    write_w_last: &[f32],
    pv_w: &[f32],
    codebook: &[f32],
    cfg: &DspineConfig,
) -> SerialTables {
    let (h, r, c) = (cfg.n_embd, cfg.rank, cfg.code_size);
    // V derives from the lm_head length — the FULL vocab at production, a
    // slice for the fixture piece (the vendor build is the full 248320; a
    // slice build is only valid when every sid/candidate id stays inside).
    assert!(
        lm_head.len().is_multiple_of(h),
        "lm_head not a row multiple"
    );
    let v = lm_head.len() / h;
    assert_eq!(write_w_last.len(), h * r, "write_w_last");
    assert_eq!(pv_w.len(), 2 * r * c, "pv_w");
    assert!(
        codebook.len() >= v * c,
        "codebook smaller than the lm_head slice"
    );

    // Write the LAST layer's write in bf16 (the vendor stores it bf16).
    let wb: Vec<f32> = write_w_last.iter().map(|&x| bf16r(x)).collect();
    let scale = (c as f32).sqrt();

    let mut tg = vec![0.0f32; v * r];
    let mut tsv = vec![0.0f32; v * 2 * r];
    // Row-parallel (the dflash2 oracle's gemv_into_par pattern): V rows ×
    // (H·R + C·2R) MACs each — ~430 s serial → seconds on the rig. Each row
    // folds sequentially in the pinned order, so results are identical to
    // the serial build.
    let tg_rows: Vec<&mut [f32]> = tg.chunks_exact_mut(r).collect();
    let lm_rows: Vec<&[f32]> = lm_head.chunks_exact(h).collect();
    tg_rows
        .into_par_iter()
        .zip(lm_rows.into_par_iter())
        .for_each(|(tg_row, lm_row)| {
            let mut u = vec![0.0f32; h];
            for i in 0..h {
                u[i] = bf16r(lm_row[i] * norm_w[i]);
            }
            for o in 0..r {
                let mut acc = 0.0f32;
                // write_w is [H, R] row-major (the vendor matmul is
                // [V,H] @ [H,R]); column o walks the H stride, found by
                // the bench-959 fixture replay (the transposed read
                // produced TG entries beyond the ±H bound — impossible
                // for the true contract).
                for i in 0..h {
                    acc += wb[i * r + o] * u[i];
                }
                tg_row[o] = bf16r(acc);
            }
        });
    let tsv_rows: Vec<&mut [f32]> = tsv.chunks_exact_mut(2 * r).collect();
    let cb_rows: Vec<&[f32]> = codebook.chunks_exact(c).collect();
    tsv_rows
        .into_par_iter()
        .zip(cb_rows.into_par_iter())
        .for_each(|(tsv_row, cb_row)| {
            for o in 0..2 * r {
                let mut acc = 0.0f32;
                for i in 0..c {
                    acc += pv_w[o * c + i] * bf16r(scale * cb_row[i]);
                }
                tsv_row[o] = bf16r(acc);
            }
        });
    SerialTables { tg, tsv }
}

// ─────────────────────────────────────────────────────────────────────────────
// The drafter
// ─────────────────────────────────────────────────────────────────────────────

/// One greedy draft result: the chain tokens (block rows 1..T-1), their
/// confidences (the chosen-sender-lane confidence per row — the vendor
/// `serial_chain` layout), and the candidate matrix `[T-1][K]` (ascending
/// ids).
#[derive(Debug, Clone)]
pub struct DspineDraft {
    pub tokens: Vec<u32>,
    pub confidences: Vec<f32>,
    pub candidates: Vec<Vec<u32>>,
}

/// The stateful drafter: per-layer context KV rings (SW-8192) + the block
/// forward + the serial tail. Single-request (bs=1) — batching is a serving
/// concern outside the oracle.
pub struct DspineDrafter {
    pub cfg: DspineConfig,
    w: DspineWeights,
    /// Served block size (2..=trained).
    block_size: usize,
    /// Per layer: K/V rings `[W × n_kv_head·head_dim]`; occupancy shared
    /// (every layer injects at the same positions, ascending).
    ring_k: Vec<Vec<f32>>,
    ring_v: Vec<Vec<f32>>,
    ring_pos: VecDeque<usize>,
    rope_freqs: Vec<f32>,
    /// TRAINING-ONLY unrounded mode (plan 492 T1): the block forward +
    /// tail skip every bf16 boundary round — the smooth surrogate the
    /// distill loss/gradFD optimize. The vendor-exact path (default) is
    /// pinned bit-identical by test; the eval harness NEVER constructs
    /// this mode.
    pub plain_numerics: bool,
}

/// The last write's stash (the tail's inputs): `raw [T, H]` (bf16-valued),
/// `rms [T]`, `packed_q [T, rank]` (the q half of the last write's packed).
#[derive(Debug, Clone)]
pub struct LastWriteStash {
    pub raw: Vec<f32>,
    pub rms: Vec<f32>,
    pub packed_q: Vec<f32>,
}

// ─────────────────────────────────────────────────────────────────────────────
// The trainer's forward trace (plan 492 T1 — riir-train's hand-written
// backward consumes these intermediates; the FORWARD stays this module's
// one implementation, never a trainer-side fork).
// ─────────────────────────────────────────────────────────────────────────────

/// One draft layer's block-forward intermediates. Array shapes are the
/// draft geometry: `t` = served block rows, `e` = n_embd, `nh·hd`/`n_kv`
/// = the q/k/v widths, `n_ff` = intermediate, `c` = code_size, `r` = rank,
/// `ctx` = occupied context rows at draft time.
#[derive(Debug, Clone, Default)]
pub struct DspineLayerTrace {
    /// `[t·e]` the layer's normed block input (the q/k/v + o_proj-mlp
    /// projection input — for layer 0 the input-norm output, for deeper
    /// layers the previous write's fused norm).
    pub x_in: Vec<f32>,
    /// `[t·e]` the layer's residual input (post-write rows of the previous
    /// layer; the anchor/mask embeddings at layer 0).
    pub x: Vec<f32>,
    /// `[t·nh·hd]` post-rope block queries.
    pub q: Vec<f32>,
    /// `[t·n_kv]` post-rope block keys.
    pub k: Vec<f32>,
    /// `[t·n_kv]` block values (pre-attention; bf16 boundary values as used).
    pub v: Vec<f32>,
    /// `[ctx·n_kv]` the context-ring keys in `ctx_pos` order (frozen
    /// functions of the teacher taps — gradients die here).
    pub ctx_k: Vec<f32>,
    /// `[ctx·n_kv]` the context-ring values in `ctx_pos` order.
    pub ctx_v: Vec<f32>,
    /// `[ctx]` the ring positions (ascending — the attention visibility
    /// rule replays from these + `sliding_window`).
    pub ctx_pos: Vec<usize>,
    /// `[t·e]` post-attention residual (`bf16(x + o_proj(ctx))`).
    pub h: Vec<f32>,
    /// `[t·n_ff]` the SwiGLU gate projection (bf16 boundary values).
    pub ffn_g: Vec<f32>,
    /// `[t·n_ff]` the SwiGLU up projection (bf16 boundary values).
    pub ffn_u: Vec<f32>,
    /// `[t·e]` the post-MLP residual — the write pass's `raw`.
    pub out: Vec<f32>,
    /// `[t]` the bridge read rms per row.
    pub rms: Vec<f32>,
    /// `[t·e]` the read-norm output (the read gemv's input, e-wide).
    pub normed_in: Vec<f32>,
    /// `[t·c]` the read gemv output (the L2-normalization's input).
    pub projected: Vec<f32>,
    /// `[t·c]` the codes (anchor row = codebook, else L2(projected)).
    pub codes: Vec<f32>,
    /// `[t·3r]` the packed `[gate_w; pred_gate_w; prev_w] @ codes` rows.
    pub packed: Vec<f32>,
    /// `[t·r]` the gated adjacent-injection message.
    pub message: Vec<f32>,
    /// `[t·e]` the write projection output.
    pub delta: Vec<f32>,
    /// `[t·e]` the post-write rows (`bf16(out + rms·delta)` — the next
    /// layer's residual `x`).
    pub out_row: Vec<f32>,
    /// `[t·e]` the fused next norm output (the next layer's `x_in`).
    pub next_x_in: Vec<f32>,
}

/// The serial refinement tail's intermediates (the loss surface).
#[derive(Debug, Clone, Default)]
pub struct DspineTailTrace {
    /// `[γ·e]` the proposal rows (`γ` = t−1): `bf16(bf16(raw·inv)·model_norm)`
    /// — the candidate-selection logits' input.
    pub proposal_rows: Vec<f32>,
    /// `[γ·k]` the walk's base scores (`lm_head[cand] · (raw ⊙ model_norm)`,
    /// the UNROUNDED walk product — kept f32 exactly as the runtime keeps it).
    pub base: Vec<f32>,
    /// `[γ]` the tail rows' bridge read rms (`stash.rms[1..]`).
    pub rms: Vec<f32>,
    /// `[γ·r]` the tail's `q` (`stash.packed_q[1..]` — the gate_w half of
    /// the last write's packed).
    pub packed_q: Vec<f32>,
    /// `[γ][k]` the candidate matrix (ascending ids — duplicated from the
    /// draft so the trace alone is self-contained for the trainer).
    pub candidates: Vec<Vec<u32>>,
    /// `[γ][k]` first-index argmax lanes (the runtime walk's routing).
    pub idx: Vec<Vec<usize>>,
    /// `[γ][k]` the confidence masses.
    pub probs: Vec<Vec<f32>>,
}

/// Everything [`DspineDrafter::draft_block_trace`] captured from one draft
/// round — the trainer's backward substrate (plan 492 T1).
#[derive(Debug, Clone, Default)]
pub struct DspineBlockTrace {
    pub anchor_token: u32,
    /// The draft's absolute anchor position (== the context length at
    /// draft time; the rope base and the attention visibility rule derive
    /// from it).
    pub anchor_pos: usize,
    /// One entry per draft layer (n_layer of them, draft order).
    pub layers: Vec<DspineLayerTrace>,
    pub tail: DspineTailTrace,
}

impl DspineDrafter {
    pub fn new(cfg: DspineConfig, w: DspineWeights) -> Self {
        let hd = cfg.head_dim;
        let mut rope_freqs = vec![0.0f32; hd / 2];
        for (i, f) in rope_freqs.iter_mut().enumerate() {
            *f = cfg.rope_theta.powf(-(2.0 * i as f32) / hd as f32);
        }
        let n_kv = cfg.n_kv_head * hd;
        let sw = cfg.sliding_window;
        let ring_k = vec![vec![0.0f32; sw * n_kv]; cfg.n_layer];
        let ring_v = vec![vec![0.0f32; sw * n_kv]; cfg.n_layer];
        Self {
            block_size: cfg.trained_block_size,
            cfg,
            w,
            ring_k,
            ring_v,
            ring_pos: VecDeque::with_capacity(sw + 8),
            rope_freqs,
            plain_numerics: false,
        }
    }

    /// The training-only unrounded posture (see the field's docs).
    pub fn with_plain_numerics(mut self) -> Self {
        self.plain_numerics = true;
        self
    }

    /// Serve `2..=trained` rows per block (the vendor `set_block_size`; the
    /// DSpine draft carries no convolutions, so no tap bound applies).
    pub fn set_block_size(&mut self, t: usize) -> Result<(), String> {
        if !(2..=self.cfg.trained_block_size).contains(&t) {
            return Err(format!(
                "block size {t} outside 2..={}",
                self.cfg.trained_block_size
            ));
        }
        self.block_size = t;
        Ok(())
    }

    /// Reset the context rings (occupancy-only discipline — slots are
    /// rewritten on append).
    pub fn reset(&mut self) {
        self.ring_pos.clear();
    }

    /// Rows the drafter currently serves per block (the `set_block_size` value).
    pub fn served_block_size(&self) -> usize {
        self.block_size
    }

    /// Context rows currently in the rings (the next anchor position).
    pub fn context_length(&self) -> usize {
        self.ring_pos.len()
    }

    /// Append context rows: `feats_rows [S × 5·H]` are the TARGET's captured
    /// hidden rows CONCATENATED in `target_layer_ids` order at absolute
    /// positions `base_pos..base_pos+S`. Computes `ctx =
    /// rmsnorm_std(bf16(fc(bf16(feats))))` once, then per layer
    /// `k_norm(rope(k_proj(ctx)))` / `v_proj(ctx)` into the rings.
    /// Positions must ascend beyond the ring back (contiguous prefix).
    pub fn append_context(&mut self, feats_rows: &[f32], base_pos: usize) -> Result<(), String> {
        let cfg = &self.cfg;
        let (e, nf) = (cfg.n_embd, cfg.target_layer_ids.len());
        let inp = nf * e;
        if !feats_rows.len().is_multiple_of(inp) {
            return Err(format!(
                "feats rows {} not a multiple of the concat width {inp}",
                feats_rows.len()
            ));
        }
        let s = feats_rows.len() / inp;
        if s == 0 {
            return Ok(());
        }
        if self.ring_pos.back().is_some_and(|&p| base_pos <= p) {
            return Err(format!(
                "append: base_pos {base_pos} not beyond ring back {:?}",
                self.ring_pos.back()
            ));
        }

        // ctx = rmsnorm_std(bf16(fc(bf16(feats)))) — computed once.
        let mut ctx_rows = vec![0.0f32; s * e];
        let mut row_buf = vec![0.0f32; inp];
        for t in 0..s {
            let src = &feats_rows[t * inp..(t + 1) * inp];
            for (i, &v) in src.iter().enumerate() {
                row_buf[i] = bf16r(v);
            }
            let fc_out = gemv_bf16(&self.w.fc, &row_buf, e);
            let normed = rmsnorm_std(&fc_out, &self.w.hidden_norm, cfg.rms_eps);
            ctx_rows[t * e..(t + 1) * e].copy_from_slice(&normed);
        }

        // Per-layer K/V: k_norm per head + rope at the absolute position.
        let hd = cfg.head_dim;
        let kvh = cfg.n_kv_head;
        let n_kv = kvh * hd;
        let (mut k, mut v) = (vec![0.0f32; n_kv], vec![0.0f32; n_kv]);
        let mut normed = vec![0.0f32; hd];
        for (li, layer) in self.w.layers.iter().enumerate() {
            for t in 0..s {
                let ctx = &ctx_rows[t * e..(t + 1) * e];
                k.copy_from_slice(&gemv_bf16(&layer.k_proj, ctx, n_kv));
                v.copy_from_slice(&gemv_bf16(&layer.v_proj, ctx, n_kv));
                for h in 0..kvh {
                    let seg = &mut k[h * hd..(h + 1) * hd];
                    normed.copy_from_slice(&rmsnorm_std(seg, &layer.k_norm, cfg.rms_eps));
                    apply_rope(&mut normed, base_pos + t, &self.rope_freqs);
                    seg.copy_from_slice(&normed);
                }
                let pos = base_pos + t;
                let slot = pos % cfg.sliding_window;
                self.ring_k[li][slot * n_kv..(slot + 1) * n_kv].copy_from_slice(&k);
                self.ring_v[li][slot * n_kv..(slot + 1) * n_kv].copy_from_slice(&v);
            }
        }
        for t in 0..s {
            let pos = base_pos + t;
            if self.ring_pos.len() == cfg.sliding_window {
                self.ring_pos.pop_front();
            }
            self.ring_pos.push_back(pos);
        }
        Ok(())
    }

    /// One draft round: the `[anchor, MASK × (T-1)]` block forward through
    /// the 5 layers + bridge writes, then the serial refinement tail.
    ///
    /// `embed_row(token)` supplies the TARGET's token-embedding row (f32 —
    /// rounded bf16 here, the vendor's storage dtype); `lm_head` is OUR
    /// dequantized output head `[V × H]` (the cross-precision disclosure:
    /// the vendor's was bf16, ours is the Q6_K GGUF head — the T1-cell
    /// class).
    pub fn draft_block(
        &self,
        anchor_token: u32,
        anchor_pos: usize,
        embed_row: &dyn Fn(u32) -> Vec<f32>,
        lm_head: &[f32],
        tables: &SerialTables,
    ) -> Result<DspineDraft, String> {
        self.draft_block_impl(anchor_token, anchor_pos, embed_row, lm_head, tables, None)
    }

    /// [`Self::draft_block`] + the trainer's forward trace. ONE forward
    /// implementation: this is the same body with the intermediates captured
    /// — the returned [`DspineDraft`] is bit-identical to `draft_block`'s
    /// (pinned by test), so training optimizes the exact deployment graph.
    pub fn draft_block_trace(
        &self,
        anchor_token: u32,
        anchor_pos: usize,
        embed_row: &dyn Fn(u32) -> Vec<f32>,
        lm_head: &[f32],
        tables: &SerialTables,
    ) -> Result<(DspineDraft, DspineBlockTrace), String> {
        let mut tr = DspineBlockTrace::default();
        let draft = self.draft_block_impl(
            anchor_token,
            anchor_pos,
            embed_row,
            lm_head,
            tables,
            Some(&mut tr),
        )?;
        Ok((draft, tr))
    }

    #[allow(clippy::too_many_lines)]
    fn draft_block_impl(
        &self,
        anchor_token: u32,
        anchor_pos: usize,
        embed_row: &dyn Fn(u32) -> Vec<f32>,
        lm_head: &[f32],
        tables: &SerialTables,
        mut tr: Option<&mut DspineBlockTrace>,
    ) -> Result<DspineDraft, String> {
        // The rounding mode: the training-only plain posture skips every
        // bf16 boundary (the smooth surrogate); the default is the
        // vendor-exact path, pinned bit-identical by test.
        let rnd: fn(f32) -> f32 = if self.plain_numerics { |v| v } else { bf16r };
        let gv = |w: &[f32], x: &[f32], n: usize| -> Vec<f32> {
            let inn = x.len();
            let mut dst = vec![0.0f32; n];
            for (o, d) in dst.iter_mut().enumerate() {
                let row = &w[o * inn..(o + 1) * inn];
                let mut acc = 0.0f32;
                for (i, &v) in x.iter().enumerate() {
                    acc += row[i] * v;
                }
                *d = rnd(acc);
            }
            dst
        };
        let rn = |x: &[f32], gamma: &[f32], eps: f32| -> Vec<f32> {
            let n = x.len();
            let mut ss = 0.0f32;
            for &v in x {
                ss += v * v;
            }
            let inv = 1.0 / (ss / n as f32 + eps).sqrt();
            (0..n).map(|i| rnd(x[i] * inv * gamma[i])).collect()
        };
        let cfg = &self.cfg;
        let t = self.block_size;
        let e = cfg.n_embd;
        let hd = cfg.head_dim;
        let nh = cfg.n_head;
        let kvh = cfg.n_kv_head;
        let grp = nh / kvh;
        let sw = cfg.sliding_window;
        let n_kv = kvh * hd;

        // ── the block input: anchor + masks, TARGET embeddings ──
        // (the draft binds the target's embed_tokens; the checkpoint ships
        // none.) bf16-rounded: the vendor's table was bf16.
        let anchor_emb: Vec<f32> = embed_row(anchor_token).iter().map(|&x| rnd(x)).collect();
        let mask_emb: Vec<f32> = embed_row(cfg.mask_token_id)
            .iter()
            .map(|&x| rnd(x))
            .collect();
        let mut x = vec![0.0f32; t * e];
        x[..e].copy_from_slice(&anchor_emb);
        for i in 1..t {
            x[i * e..(i + 1) * e].copy_from_slice(&mask_emb);
        }

        // The block's rope base: positions continue after the context rows.
        let ctx_len = self.context_length();
        if anchor_pos != ctx_len {
            return Err(format!(
                "anchor_pos {anchor_pos} != context length {ctx_len} — append context first"
            ));
        }
        if self.ring_pos.is_empty() {
            return Err("no context appended — draft needs the target's context KV".into());
        }

        // Layer 0 input norm (every row); residual stream = x.
        let mut x_in = vec![0.0f32; t * e];
        for i in 0..t {
            let normed = rn(
                &x[i * e..(i + 1) * e],
                &self.w.layers[0].input_norm,
                cfg.rms_eps,
            );
            x_in[i * e..(i + 1) * e].copy_from_slice(&normed);
        }
        if let Some(tr) = tr.as_deref_mut() {
            tr.anchor_token = anchor_token;
            tr.anchor_pos = anchor_pos;
        }
        let mut stash = LastWriteStash {
            raw: vec![0.0f32; t * e],
            rms: vec![0.0f32; t],
            packed_q: vec![0.0f32; t * cfg.rank],
        };

        let scale = 1.0f32 / (hd as f32).sqrt();
        let packed_w = self.w.packed_weight();
        let sqrt_c = (cfg.code_size as f32).sqrt();

        for (li, layer) in self.w.layers.iter().enumerate() {
            if let Some(tr) = tr.as_deref_mut() {
                tr.layers.push(DspineLayerTrace {
                    x_in: x_in.clone(),
                    x: x.clone(),
                    ..Default::default()
                });
            }
            // ── attention: q/k/v block projections + context-ring K/V ──
            let mut q = vec![0.0f32; t * nh * hd];
            let mut k = vec![0.0f32; t * n_kv];
            let mut vv = vec![0.0f32; t * n_kv];
            for i in 0..t {
                let seg = &x_in[i * e..(i + 1) * e];
                let q_o = gv(&layer.q_proj, seg, nh * hd);
                q[i * nh * hd..(i + 1) * nh * hd].copy_from_slice(&q_o);
                k[i * n_kv..(i + 1) * n_kv].copy_from_slice(&gv(&layer.k_proj, seg, n_kv));
                vv[i * n_kv..(i + 1) * n_kv].copy_from_slice(&gv(&layer.v_proj, seg, n_kv));
            }
            // Per-head q/k norms + rope at absolute positions.
            let mut normed = vec![0.0f32; hd];
            for i in 0..t {
                let p = anchor_pos + i;
                for h in 0..nh {
                    let base = i * nh * hd + h * hd;
                    normed.copy_from_slice(&rn(
                        &q[base..base + hd],
                        &layer.q_norm,
                        cfg.rms_eps,
                    ));
                    apply_rope(&mut normed, p, &self.rope_freqs);
                    q[base..base + hd].copy_from_slice(&normed);
                }
                for h in 0..kvh {
                    let base = i * n_kv + h * hd;
                    normed.copy_from_slice(&rn(
                        &k[base..base + hd],
                        &layer.k_norm,
                        cfg.rms_eps,
                    ));
                    apply_rope(&mut normed, p, &self.rope_freqs);
                    k[base..base + hd].copy_from_slice(&normed);
                }
            }
            if let Some(tr) = tr.as_mut() {
                let lt = tr.layers.last_mut().unwrap();
                lt.q = q.clone();
                lt.k = k.clone();
                lt.v = vv.clone();
                // The occupied ring rows, gathered contiguous in ring_pos
                // order (the attention visibility rule replays from these).
                let ctx = self.ring_pos.len();
                lt.ctx_k.clear();
                lt.ctx_v.clear();
                lt.ctx_pos.clear();
                lt.ctx_k.reserve(ctx * n_kv);
                lt.ctx_v.reserve(ctx * n_kv);
                for &rp in self.ring_pos.iter() {
                    let slot = rp % sw;
                    lt.ctx_k
                        .extend_from_slice(&self.ring_k[li][slot * n_kv..(slot + 1) * n_kv]);
                    lt.ctx_v
                        .extend_from_slice(&self.ring_v[li][slot * n_kv..(slot + 1) * n_kv]);
                    lt.ctx_pos.push(rp);
                }
            }

            // Attention over [context ring ∪ block KV]: causal inside the
            // block, every context row visible (SWA q−k >= W masks stale
            // context). Fold order pinned: context keys in ascending
            // ring_pos order, then block keys ascending j (the Issue-940
            // law; 2-pass softmax — max pass, then weight pass).
            let mut attn_out = vec![0.0f32; t * nh * hd];
            let mut ovec = vec![0.0f32; hd];
            let ring_pos: Vec<usize> = self.ring_pos.iter().copied().collect();
            for i in 0..t {
                let qp = anchor_pos + i;
                let lo = qp.saturating_sub(sw - 1);
                for h in 0..nh {
                    let kvh_idx = h / grp;
                    let qb = i * nh * hd + h * hd;
                    let qseg = &q[qb..qb + hd];

                    // Pass 1: max unscaled score over visible keys.
                    let mut max_s = f32::NEG_INFINITY;
                    for &rp in &ring_pos {
                        if rp < lo {
                            continue;
                        }
                        let base = (rp % sw) * n_kv + kvh_idx * hd;
                        let mut dotv = 0.0f32;
                        for (d, qv) in qseg.iter().enumerate() {
                            dotv += qv * self.ring_k[li][base + d];
                        }
                        max_s = max_s.max(dotv);
                    }
                    for bj in 0..=i {
                        let base = bj * n_kv + kvh_idx * hd;
                        let mut dotv = 0.0f32;
                        for (d, qv) in qseg.iter().enumerate() {
                            dotv += qv * k[base + d];
                        }
                        max_s = max_s.max(dotv);
                    }

                    // Pass 2: weights + V accumulation.
                    let mut sum = 0.0f32;
                    ovec.iter_mut().for_each(|v| *v = 0.0);
                    for &rp in &ring_pos {
                        if rp < lo {
                            continue;
                        }
                        let slot = rp % sw;
                        let base = slot * n_kv + kvh_idx * hd;
                        let mut dotv = 0.0f32;
                        for (d, qv) in qseg.iter().enumerate() {
                            dotv += qv * self.ring_k[li][base + d];
                        }
                        let wgt = ((dotv - max_s) * scale).exp();
                        sum += wgt;
                        for (d, ov) in ovec.iter_mut().enumerate() {
                            *ov += wgt * self.ring_v[li][base + d];
                        }
                    }
                    for bj in 0..=i {
                        let base = bj * n_kv + kvh_idx * hd;
                        let mut dotv = 0.0f32;
                        for d in 0..hd {
                            dotv += qseg[d] * k[base + d];
                        }
                        let wgt = ((dotv - max_s) * scale).exp();
                        sum += wgt;
                        for d in 0..hd {
                            ovec[d] += wgt * vv[base + d];
                        }
                    }
                    let ob = i * nh * hd + h * hd;
                    for d in 0..hd {
                        attn_out[ob + d] = ovec[d] / sum;
                    }
                }
            }
            // o_proj (bf16 storage boundary — attn context rounds first).
            let mut ao = vec![0.0f32; t * e];
            for i in 0..t {
                let row_in: Vec<f32> = attn_out[i * nh * hd..(i + 1) * nh * hd]
                    .iter()
                    .map(|&v| rnd(v))
                    .collect();
                let o = gv(&layer.o_proj, &row_in, e);
                ao[i * e..(i + 1) * e].copy_from_slice(&o);
            }

            // Residual: h = bf16(x + attn_out).
            let mut h = vec![0.0f32; t * e];
            for j in 0..t * e {
                h[j] = rnd(x[j] + ao[j]);
            }

            // MLP: norm → SwiGLU → down, bf16 at every boundary.
            let mut mlp_out = vec![0.0f32; t * e];
            let mut mid = vec![0.0f32; cfg.n_ff];
            let mut ffn_g_all = vec![0.0f32; t * cfg.n_ff];
            let mut ffn_u_all = vec![0.0f32; t * cfg.n_ff];
            for i in 0..t {
                let hf = rn(&h[i * e..(i + 1) * e], &layer.ffn_norm, cfg.rms_eps);
                let g = gemv_bf16(&layer.gate_proj, &hf, cfg.n_ff);
                let u = gemv_bf16(&layer.up_proj, &hf, cfg.n_ff);
                ffn_g_all[i * cfg.n_ff..(i + 1) * cfg.n_ff].copy_from_slice(&g);
                ffn_u_all[i * cfg.n_ff..(i + 1) * cfg.n_ff].copy_from_slice(&u);
                for (o_i, (m_o, mg)) in mid.iter_mut().zip(g.iter()).enumerate() {
                    let s = rnd(mg / (1.0 + (-mg).exp()));
                    *m_o = rnd(s * u[o_i]);
                }
                let d = gv(&layer.down_proj, &mid, e);
                mlp_out[i * e..(i + 1) * e].copy_from_slice(&d);
            }

            // out = bf16(h + mlp_out) — the new residual stream.
            let mut out = vec![0.0f32; t * e];
            for j in 0..t * e {
                out[j] = rnd(h[j] + mlp_out[j]);
            }
            if let Some(tr) = tr.as_deref_mut() {
                let lt = tr.layers.last_mut().unwrap();
                lt.h = h.clone();
                lt.ffn_g = ffn_g_all;
                lt.ffn_u = ffn_u_all;
                lt.out = out.clone();
            }

            // ── the code-bridge write (fused next norm) ──
            // Two passes: (1) read-norm + codes + packed per row; (2) the
            // gated message — the sender/value halves come from the
            // PREVIOUS ROW's packed (the vendor `_serial_table`/bridge
            // indexing) — then write + fused norm.
            let read_w = self.w.read_weight_for(li);
            let write_w = &self.w.bridge.feature_write_weights[li];
            let next_norm_w = if li + 1 == cfg.n_layer {
                &self.w.model_norm
            } else {
                &self.w.layers[li + 1].input_norm
            };
            let mut next_x_in = vec![0.0f32; t * e];
            let mut next_x = vec![0.0f32; t * e];
            let mut packed_all = vec![0.0f32; t * 3 * cfg.rank];
            let mut rms_all = vec![0.0f32; t];
            let mut normed_all = vec![0.0f32; t * e];
            let mut projected_all = vec![0.0f32; t * cfg.code_size];
            let mut codes_all = vec![0.0f32; t * cfg.code_size];
            let mut message_all = vec![0.0f32; t * cfg.rank];
            let mut delta_all = vec![0.0f32; t * e];
            for i in 0..t {
                let raw_row = &out[i * e..(i + 1) * e];
                // raw is already bf16-valued (out is). The read norm.
                let (rms, normed_in) = if self.plain_numerics {
                    let nn = raw_row.len();
                    let mut ss = 0.0f32;
                    for &v in raw_row { ss += v * v; }
                    let rms = (ss / nn as f32).max(f32::MIN_POSITIVE).sqrt();
                    let normed: Vec<f32> = (0..nn).map(|i2| raw_row[i2] / (rms + cfg.bridge_eps)).collect();
                    (rms, normed)
                } else {
                    bridge_read_norm(raw_row, cfg.bridge_eps)
                };
                rms_all[i] = rms;
                normed_all[i * e..(i + 1) * e].copy_from_slice(&normed_in);
                let projected = gv(read_w, &normed_in, cfg.code_size);
                projected_all[i * cfg.code_size..(i + 1) * cfg.code_size]
                    .copy_from_slice(&projected);
                // codes: anchor row ← sqrt(C)·codebook[anchor]; else the
                // L2-normalized projection.
                let mut codes = vec![0.0f32; cfg.code_size];
                if i % t == 0 {
                    let cb = anchor_token as usize * cfg.code_size;
                    for (j, cv) in codes.iter_mut().enumerate() {
                        *cv = rnd(sqrt_c * self.w.bridge.codebook[cb + j]);
                    }
                } else {
                    let mut nrm = 0.0f32;
                    for &z in &projected {
                        nrm += z * z;
                    }
                    let nrm = nrm.sqrt();
                    for (j, cv) in codes.iter_mut().enumerate() {
                        *cv = rnd(sqrt_c * projected[j] / (nrm + cfg.bridge_eps));
                    }
                }
                codes_all[i * cfg.code_size..(i + 1) * cfg.code_size].copy_from_slice(&codes);
                let packed = gv(&packed_w, &codes, 3 * cfg.rank);
                packed_all[i * 3 * cfg.rank..(i + 1) * 3 * cfg.rank].copy_from_slice(&packed);
                if li + 1 == cfg.n_layer {
                    stash.raw[i * e..(i + 1) * e].copy_from_slice(raw_row);
                    stash.rms[i] = rms;
                    stash.packed_q[i * cfg.rank..(i + 1) * cfg.rank]
                        .copy_from_slice(&packed[..cfg.rank]);
                }
            }
            for i in 0..t {
                let raw_row = &out[i * e..(i + 1) * e];
                let rms = rms_all[i];
                let packed = &packed_all[i * 3 * cfg.rank..(i + 1) * 3 * cfg.rank];
                // message: gate over the PREDECESSOR row's sender/value
                // halves (zero at the anchor row).
                let prev = i.saturating_sub(1);
                let mut message = vec![0.0f32; cfg.rank];
                let mut delta = vec![0.0f32; e];
                if i % t != 0 {
                    let pbase = prev * 3 * cfg.rank;
                    for j in 0..cfg.rank {
                        let qv = packed[j];
                        let sv = packed_all[pbase + cfg.rank + j];
                        let vv_ = packed_all[pbase + 2 * cfg.rank + j];
                        let gate = 1.0 / (1.0 + (-(qv + sv + self.w.bridge.gate_b[j]).exp()));
                        message[j] = rnd(gate * vv_);
                    }
                }
                delta = gv(write_w, &message, e);
                message_all[i * cfg.rank..(i + 1) * cfg.rank].copy_from_slice(&message);
                delta_all[i * e..(i + 1) * e].copy_from_slice(&delta);
                // out_row = bf16(f + rms·delta) at non-anchor rows — the
                // updated residual stream (the next layer's residual).
                let mut out_row = vec![0.0f32; e];
                for (j, ov) in out_row.iter_mut().enumerate() {
                    let upd = if i % t != 0 { rms * delta[j] } else { 0.0 };
                    *ov = rnd(raw_row[j] + upd);
                }
                // The fused next norm (finish form: variance + eps).
                let mut ss = 0.0f32;
                for &xv in &out_row {
                    ss += xv * xv;
                }
                let inv = 1.0 / (ss / e as f32 + cfg.rms_eps).sqrt();
                for (j, ov) in next_x_in[i * e..(i + 1) * e].iter_mut().enumerate() {
                    let y = rnd(out_row[j] * inv);
                    *ov = rnd(y * next_norm_w[j]);
                }
                next_x[i * e..(i + 1) * e].copy_from_slice(&out_row);
            }
            if let Some(tr) = tr.as_deref_mut() {
                let lt = tr.layers.last_mut().unwrap();
                lt.rms = rms_all;
                lt.normed_in = normed_all;
                lt.projected = projected_all;
                lt.codes = codes_all;
                lt.packed = packed_all;
                lt.message = message_all;
                lt.delta = delta_all;
                lt.out_row = next_x.clone();
                lt.next_x_in = next_x_in.clone();
            }
            x_in = next_x_in;
            x = next_x;
        }

        // ── the serial refinement tail (greedy) ──
        self.serial_tail_impl(
            anchor_token,
            &stash,
            lm_head,
            tables,
            tr.as_mut().map(|t| &mut t.tail),
        )
    }

    /// The `exact_sender` serial_table tail: candidates, the predecessor-
    /// conditioned score tables, the greedy chain walk, the confidences.
    /// The tail's inputs (the vendor stash contract): `raw` = the last
    /// write's PRE-write rows, `rms` its read rms, `packed_q` the q half of
    /// its packed codes. The proposal rows for CANDIDATES are the write's
    /// normed outputs (the model norm applied to raw — recomputed here,
    /// deterministic, the same fold the write used). Rows 1..T-1 propose;
    /// row 0 is the anchor slot. Delegates the walk to
    /// [`serial_walk_core`] (the fixture piece (d) shares it). `tr = Some`
    /// captures the loss-surface trace (plan 492 T1) — the draft result is
    /// bit-identical either way (pinned by test).
    #[allow(clippy::too_many_arguments)]
    fn serial_tail_impl(
        &self,
        anchor_token: u32,
        stash: &LastWriteStash,
        lm_head: &[f32],
        tables: &SerialTables,
        mut tr: Option<&mut DspineTailTrace>,
    ) -> Result<DspineDraft, String> {
        let rnd: fn(f32) -> f32 = if self.plain_numerics { |v| v } else { bf16r };
        let cfg = &self.cfg;
        let t = self.block_size;
        let e = cfg.n_embd;
        let k = cfg.head_topk;
        let gamma = t - 1;

        // Proposal rows 1..T-1: normed = bf16(bf16(raw·inv) · norm_w).
        let mut rows = vec![0.0f32; gamma * e];
        for i in 0..gamma {
            let raw_row = &stash.raw[(i + 1) * e..(i + 2) * e];
            let mut ss = 0.0f32;
            for &xv in raw_row {
                ss += xv * xv;
            }
            let inv = 1.0 / (ss / e as f32 + cfg.rms_eps).sqrt();
            for (j, rv) in rows[i * e..(i + 1) * e].iter_mut().enumerate() {
                let y = rnd(raw_row[j] * inv);
                *rv = rnd(y * self.w.model_norm[j]);
            }
        }

        // Candidates per row: top-k of rows @ lm_headᵀ, ascending ids.
        let mut candidates: Vec<Vec<u32>> = Vec::with_capacity(gamma);
        for i in 0..gamma {
            let row = &rows[i * e..(i + 1) * e];
            let mut logits = vec![0.0f32; cfg.vocab_size];
            for (vi, lrow) in lm_head.chunks_exact(e).enumerate() {
                logits[vi] = dot(lrow, row);
            }
            candidates.push(top_k_ids_ascending(&logits, k));
        }

        let (idx, probs, extras) = serial_walk_core_ext(
            &stash.raw,
            &stash.rms,
            &stash.packed_q,
            &candidates,
            anchor_token,
            lm_head,
            &self.w.model_norm,
            &self.w.bridge.gate_b,
            tables,
            cfg,
        );
        if let Some(tr) = tr.as_mut() {
            tr.proposal_rows = rows;
            tr.base = extras.0;
            tr.rms = stash.rms[1..].to_vec();
            tr.packed_q = stash.packed_q[cfg.rank..].to_vec();
            tr.candidates = candidates.clone();
            tr.idx = idx.clone();
            tr.probs = probs.clone();
        }
        // The greedy chain walk + confidences (the vendor serial_chain
        // layout: row 0 reads lane 0; `previous` updates AFTER the read).
        let (tokens, confidences) = chain_walk(&candidates, &idx, &probs);
        Ok(DspineDraft {
            tokens,
            confidences,
            candidates,
        })
    }
}

/// The per-row lane tables + argmax + confidence mass (fixture piece (d)'s
/// core — [`DspineDrafter::serial_tail`] shares it).
/// score[n][j][jp] = base[n][jp] + rms[n] · Σ_rr m[n][j][rr] ·
/// TG[cand[n][jp]][rr], with m[n][j][rr] = sigmoid(q[n][rr] +
/// TSV[sid[n,j],rr] + gate_b[rr]) · TSV[sid[n,j], R+rr] and
/// sid[n][j] = cand[n−1][j] (row 0: the anchor token). Returns
/// (idx [γ×K] first-index argmax, probs [γ×K] the confidence mass).
#[allow(clippy::too_many_arguments)]
pub fn serial_walk_core(
    raw: &[f32],
    rms: &[f32],
    packed_q: &[f32],
    candidates: &[Vec<u32>],
    anchor_token: u32,
    lm_head: &[f32],
    norm_w: &[f32],
    gate_b: &[f32],
    tables: &SerialTables,
    cfg: &DspineConfig,
) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
    let (idx, probs, _) = serial_walk_core_ext(
        raw,
        rms,
        packed_q,
        candidates,
        anchor_token,
        lm_head,
        norm_w,
        gate_b,
        tables,
        cfg,
    );
    (idx, probs)
}

/// The trainer-facing extras bundle: the walk's `base` scores `[γ·K]`
/// (flattened row-major) and the per-row/per-lane gated message
/// `m [γ][K][R]` at the RUNTIME sids (the trainer recomputes rows for its
/// own sid policy — plan 492 T1).
pub type SerialWalkExtras = (Vec<f32>, Vec<Vec<Vec<f32>>>);

/// [`serial_walk_core`] + the trainer's loss-surface extras ([`SerialWalkExtras`]).
#[allow(clippy::too_many_arguments)]
pub fn serial_walk_core_ext(
    raw: &[f32],
    rms: &[f32],
    packed_q: &[f32],
    candidates: &[Vec<u32>],
    anchor_token: u32,
    lm_head: &[f32],
    norm_w: &[f32],
    gate_b: &[f32],
    tables: &SerialTables,
    cfg: &DspineConfig,
) -> (Vec<Vec<usize>>, Vec<Vec<f32>>, SerialWalkExtras) {
    let (e, r, k) = (cfg.n_embd, cfg.rank, cfg.head_topk);
    let gamma = candidates.len();
    // The lm_head may be a SLICE (the fixture piece (d)); every candidate id
    // must be covered.
    let max_cand = candidates.iter().flatten().copied().max().unwrap_or(0) as usize;
    assert!(
        max_cand * e + e <= lm_head.len(),
        "lm_head slice lacks candidate row {max_cand}"
    );
    let mut idx = vec![vec![0usize; k]; gamma];
    let mut probs = vec![vec![0.0f32; k]; gamma];
    let mut base_all = vec![0.0f32; gamma * k];
    let mut m_all: Vec<Vec<Vec<f32>>> = Vec::with_capacity(gamma);
    for nn in 0..gamma {
        let raw_row = &raw[(nn + 1) * e..(nn + 2) * e];
        let rms_n = rms[nn + 1];
        let q = &packed_q[(nn + 1) * r..(nn + 2) * r];
        let cand_n = &candidates[nn];
        // u = raw · norm_w, kept UNROUNDED f32 — both vendor references
        // keep the walk's base product in f32 (the Triton _serial_table
        // loads bf16 → f32 and multiplies; only the TG TABLE build rounds
        // u to bf16 before its matmul).
        let mut u = vec![0.0f32; e];
        for (i2, uv) in u.iter_mut().enumerate() {
            *uv = raw_row[i2] * norm_w[i2];
        }
        let base: Vec<f32> = cand_n
            .iter()
            .map(|&tok| {
                let lrow = &lm_head[tok as usize * e..(tok as usize + 1) * e];
                dot(lrow, &u)
            })
            .collect();
        let sid_tokens: Vec<u32> = if nn == 0 {
            vec![anchor_token; k]
        } else {
            candidates[nn - 1].clone()
        };
        base_all[nn * k..(nn + 1) * k].copy_from_slice(&base);
        let mut m_rows: Vec<Vec<f32>> = Vec::with_capacity(k);
        let mut scores = vec![0.0f32; k * k];
        for (j, &sid) in sid_tokens.iter().enumerate() {
            let tsv_base = sid as usize * 2 * r;
            let mut m_msg = vec![0.0f32; r];
            for rr in 0..r {
                let z = q[rr] + tables.tsv[tsv_base + rr] + gate_b[rr];
                let gate = 1.0 / (1.0 + (-z).exp());
                m_msg[rr] = gate * tables.tsv[tsv_base + r + rr];
            }
            m_rows.push(m_msg.clone());
            for jp in 0..k {
                let tg_base = cand_n[jp] as usize * r;
                let mut s = 0.0f32;
                for (rr, mv) in m_msg.iter().enumerate() {
                    s += mv * tables.tg[tg_base + rr];
                }
                // The CANDIDATE's base joins the score (the vendor
                // broadcasts base[None, :] over the sender axis in both
                // references); the sender's own base never does.
                scores[j * k + jp] = base[jp] + rms_n * s;
            }
        }
        for (j, srow) in scores.chunks_exact(k).enumerate() {
            // First-index argmax over the lane dimension.
            let mut best_j = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for (jp, &sv) in srow.iter().enumerate() {
                if sv > best_v {
                    best_v = sv;
                    best_j = jp;
                }
            }
            idx[nn][j] = best_j;
            // confidence mass: 1 / Σ exp((s − best)/max(rms, 1e-6)).
            let scale = rms_n.max(1.0e-6);
            let mut mass = 0.0f32;
            for sv in srow {
                mass += ((sv - best_v) / scale).exp();
            }
            probs[nn][j] = 1.0 / mass;
        }
        m_all.push(m_rows);
    }
    (idx, probs, (base_all, m_all))
}

/// The greedy chain walk: lane₀ = idx[0][0]; token[n] = cand[n][lane];
/// lane ← idx[n+1][lane]. Confidence per row = probs[r][previous] (the
/// vendor serial_chain layout: row 0 reads lane 0; `previous` updates to
/// the lane just used AFTER the read). Returns (tokens, confidences).
pub fn chain_walk(
    candidates: &[Vec<u32>],
    idx: &[Vec<usize>],
    probs: &[Vec<f32>],
) -> (Vec<u32>, Vec<f32>) {
    let gamma = candidates.len();
    let mut tokens = Vec::with_capacity(gamma);
    let mut confidences = Vec::with_capacity(gamma);
    let mut lane = idx[0][0];
    let mut previous = 0usize;
    for nn in 0..gamma {
        confidences.push(probs[nn][previous]);
        tokens.push(candidates[nn][lane]);
        previous = lane;
        if nn + 1 < gamma {
            lane = idx[nn + 1][lane];
        }
    }
    (tokens, confidences)
}

// ── fixture pieces (the G1 golden-generator contract; the harness bin
//    drives these with synthetic bf16 inputs and the vendor MLX side
//    replays the same bytes) ──

/// Piece (a): the bridge read+codes for `t` whole rows. Returns
/// (rms [t], codes [t × code_size]).
pub fn piece_read_codes(
    read_w: &[f32],
    raw: &[f32],
    eps: f32,
    codebook: &[f32],
    anchor_token: u32,
    code_size: usize,
    t: usize,
) -> (Vec<f32>, Vec<f32>) {
    let e = raw.len() / t;
    let mut rms_out = vec![0.0f32; t];
    let mut codes = vec![0.0f32; t * code_size];
    let sqrt_c = (code_size as f32).sqrt();
    for i in 0..t {
        let row = &raw[i * e..(i + 1) * e];
        let (rms, normed_in) = bridge_read_norm(row, eps);
        rms_out[i] = rms;
        let projected = gemv_bf16(read_w, &normed_in, code_size);
        if i % t == 0 {
            let cb = anchor_token as usize * code_size;
            for (j, cv) in codes[i * code_size..(i + 1) * code_size]
                .iter_mut()
                .enumerate()
            {
                *cv = bf16r(sqrt_c * codebook[cb + j]);
            }
        } else {
            let mut nrm = 0.0f32;
            for &z in &projected {
                nrm += z * z;
            }
            let nrm = nrm.sqrt();
            for (j, cv) in codes[i * code_size..(i + 1) * code_size]
                .iter_mut()
                .enumerate()
            {
                *cv = bf16r(sqrt_c * projected[j] / (nrm + eps));
            }
        }
    }
    (rms_out, codes)
}

/// Piece (b): message + write + fused norm for one row. Returns
/// (message [rank], out_row [e], normed [e]).
pub fn piece_message_write(
    packed: &[f32],
    prev_packed: &[f32],
    gate_b: &[f32],
    write_w: &[f32],
    rms: f32,
    raw: &[f32],
    is_anchor: bool,
    norm_w: &[f32],
    eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let r = gate_b.len();
    let e = raw.len();
    let mut message = vec![0.0f32; r];
    if !is_anchor {
        for j in 0..r {
            let z = packed[j] + prev_packed[r + j] + gate_b[j];
            let gate = 1.0 / (1.0 + (-z).exp());
            message[j] = bf16r(gate * prev_packed[2 * r + j]);
        }
    }
    let delta = gemv_bf16(write_w, &message, e);
    let mut out_row = vec![0.0f32; e];
    for (j, ov) in out_row.iter_mut().enumerate() {
        let upd = if is_anchor { 0.0 } else { rms * delta[j] };
        *ov = bf16r(raw[j] + upd);
    }
    let mut ss = 0.0f32;
    for &xv in &out_row {
        ss += xv * xv;
    }
    let inv = 1.0 / (ss / e as f32 + eps).sqrt();
    let mut normed = vec![0.0f32; e];
    for (j, nv) in normed.iter_mut().enumerate() {
        let y = bf16r(out_row[j] * inv);
        *nv = bf16r(y * norm_w[j]);
    }
    (message, out_row, normed)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests — synthetic invariants only (no vendor bytes; the G1 golden diff is
// the harness bin's fixture lane).
// ─────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_round_is_idempotent_and_rne_correct() {
        // A bf16-representable value round-trips exactly; a midpoint rounds
        // to even.
        let v = 0.8125f32; // exactly representable in bf16
        assert_eq!(bf16r(v), v);
        // Any f32 → bf16 → f32 changes only the low mantissa bits: the
        // result re-rounds to itself.
        let x = 0.123_456_79_f32;
        assert_eq!(bf16r(bf16r(x)), bf16r(x));
    }

    #[test]
    fn bridge_read_norm_matches_the_vendor_form() {
        // rms = sqrt(max(mean(x²), F32_TINY)); normalized = bf16(x/(rms+eps))
        // — gamma-free, unlike the engine's rmsnorm_into.
        let x = vec![3.0f32, -4.0]; // mean(x²) = 12.5
        let eps = 1.0e-6;
        let (rms, normed) = bridge_read_norm(&x, eps);
        assert!((rms - 12.5f32.sqrt()).abs() < 1e-6);
        let inv = 1.0 / (12.5f32.sqrt() + eps);
        assert_eq!(normed[0], bf16r(3.0 * inv));
        assert_eq!(normed[1], bf16r(-4.0 * inv));
        // The zero row hits the F32-tiny floor, never a division by zero.
        let (rms0, _) = bridge_read_norm(&[0.0; 4], eps);
        assert!(rms0 > 0.0);
    }

    #[test]
    fn piece_codes_anchor_takes_the_codebook_row_normalized() {
        // Row 0 (anchor) = bf16(sqrt(C)·codebook[anchor]); rows i != 0 =
        // bf16(sqrt(C)·projected/‖projected‖) — norm sqrt(C) by construction.
        let cfg = DspineConfig::default();
        let (c, e) = (cfg.code_size, cfg.n_embd);
        let anchor = 3u32;
        let mut codebook = vec![0.0f32; 16 * c];
        for (i, v) in codebook.iter_mut().enumerate() {
            *v = bf16r(((i % 13) as f32 - 6.0) * 0.05);
        }
        let read_w = vec![0.0f32; c * e]; // zero read weight → projected zero
        // Row 0 = anchor; rows 1..t-1 zero-hidden → projected zero → codes
        // zero (0/(0+eps) = 0).
        let raw = vec![0.0f32; 4 * e];
        let (rms, codes) = piece_read_codes(&read_w, &raw, cfg.bridge_eps, &codebook, anchor, c, 4);
        let sqrt_c = (c as f32).sqrt();
        for j in 0..c {
            assert_eq!(codes[j], bf16r(sqrt_c * codebook[anchor as usize * c + j]));
        }
        for i in 1..4 {
            for j in 0..c {
                assert_eq!(codes[i * c + j], 0.0);
            }
        }
        // Anchor rms over a zero row = the f32-tiny sqrt floor.
        assert!(rms[0] > 0.0 && rms[0] < 1e-18);
    }

    #[test]
    fn piece_message_write_anchor_message_is_zero() {
        let (r, e) = (8usize, 16usize);
        let packed: Vec<f32> = (0..3 * r).map(|i| bf16r(i as f32 * 0.1)).collect();
        let gate_b = vec![0.0f32; r];
        let write_w = vec![1.0f32; e * r];
        let raw = vec![0.5f32; e];
        let norm_w = vec![1.0f32; e];
        let (m_anchor, out_anchor, _) = piece_message_write(
            &packed, &packed, &gate_b, &write_w, 1.0, &raw, true, &norm_w, 1e-6,
        );
        assert!(m_anchor.iter().all(|&v| v == 0.0));
        assert!(out_anchor.iter().all(|&v| v == bf16r(0.5))); // no delta at the anchor
        let (m, out, normed) = piece_message_write(
            &packed, &packed, &gate_b, &write_w, 1.0, &raw, false, &norm_w, 1e-6,
        );
        assert!(m.iter().any(|&v| v != 0.0));
        // write_w all-ones: delta = Σ message → out = raw + rms·delta·1.
        let delta_sum: f32 = m.iter().copied().sum();
        assert_eq!(out[0], bf16r(0.5 + delta_sum));
        // normed is the standard-form RMSNorm of out.
        let ss: f32 = out.iter().map(|&v| v * v).sum();
        let inv = 1.0 / (ss / e as f32 + 1e-6).sqrt();
        assert_eq!(normed[0], bf16r(bf16r(out[0] * inv))); // norm_w=1 → second round no-op
    }

    #[test]
    fn chain_walk_follows_the_predecessor_lanes() {
        // γ=3 rows, K=2 lanes: idx[0]=[_ ,1], idx[1]=[0,_], idx[2]=[1,_] —
        // lane₀ = idx[0][0]; token[n] = cand[n][lane]; lane ← idx[n+1][lane].
        let candidates = vec![vec![10, 11], vec![20, 21], vec![30, 31]];
        let idx = vec![
            vec![0usize, 1usize],
            vec![0usize, 1usize],
            vec![1usize, 0usize],
        ];
        let probs = vec![
            vec![0.9f32, 0.1f32],
            vec![0.8f32, 0.2f32],
            vec![0.7f32, 0.3f32],
        ];
        let (tokens, confs) = chain_walk(&candidates, &idx, &probs);
        // lane₀ = idx[0][0] = 0 → token[0] = 10; lane₁ = idx[1][0] = 0 →
        // token[1] = 20; lane₂ = idx[2][0] = 1 → token[2] = 31.
        assert_eq!(tokens, vec![10, 20, 31]);
        // conf[n] = probs[n][previous]: prev starts 0 → conf[0]=probs[0][0];
        // prev = lane used at 0 (=0) → conf[1]=probs[1][0]; prev = lane at 1
        // (=0) → conf[2]=probs[2][0].
        assert_eq!(confs, vec![0.9, 0.8, 0.7]);
    }

    #[test]
    fn serial_walk_core_sid_chain_and_first_index_ties() {
        // A tiny geometry (E=4, R=2, K=2, γ=2): build tables by hand so the
        // winner lane is derivable. Row 0's sid = the anchor for every lane;
        // a TIE between lanes must resolve to the FIRST index.
        let cfg = DspineConfig {
            n_layer: 5,
            n_embd: 4,
            n_head: 2,
            n_kv_head: 1,
            head_dim: 2,
            n_ff: 4,
            trained_block_size: 4,
            sliding_window: 8,
            rope_theta: 1.0e7,
            rms_eps: 1e-6,
            code_size: 4,
            rank: 2,
            bridge_eps: 1e-6,
            vocab_size: 6,
            mask_token_id: 5,
            head_topk: 2,
            target_layer_ids: vec![1, 16, 31, 46, 61],
            num_target_layers: 64,
        };
        let e = cfg.n_embd;
        let r = cfg.rank;
        let v = cfg.vocab_size;
        // raw rows: row 0 unused by the walk; rows 1..γ the stash rows.
        let raw = vec![0.0f32; 3 * e];
        let mut rms = vec![0.0f32; 3];
        rms[1] = 2.0;
        rms[2] = 2.0;
        let packed_q = vec![0.0f32; 3 * r];
        // cand rows: identical so sid (anchor at row 0, cand[0] at row 1)
        // makes both rows' math identical.
        let candidates = vec![vec![1u32, 2u32], vec![1u32, 2u32]];
        let mut lm_head = vec![0.0f32; v * e];
        for i in 0..e {
            lm_head[1 * e + i] = 1.0; // token 1 row
            lm_head[2 * e + i] = 1.0; // token 2 row (SAME → base ties)
        }
        let norm_w = vec![0.0f32; e]; // u = 0 → base = 0 for every lane
        let gate_b = vec![0.0f32; r];
        // TG rows equal → scores tie → idx first-index (0).
        let tables = SerialTables {
            tg: vec![1.0f32; v * r],
            tsv: vec![0.5f32; v * 2 * r],
        };
        let (idx, probs) = serial_walk_core(
            &raw,
            &rms,
            &packed_q,
            &candidates,
            0,
            &lm_head,
            &norm_w,
            &gate_b,
            &tables,
            &cfg,
        );
        for row in &idx {
            assert_eq!(row, &[0usize, 0usize], "perfect ties → first-index lanes");
        }
        // Confidence at a perfect tie: mass = Σ exp(0/scale) = K → 1/K.
        for row in &probs {
            for p in row {
                assert!((p - 0.5).abs() < 1e-5, "tie mass = 1/K, got {p}");
            }
        }
    }

    #[test]
    fn serial_walk_core_scores_the_candidate_base() {
        // The G1 golden-diff regression: the CANDIDATE's base enters the
        // score (the vendor broadcasts base[None,:] over the sender axis in
        // both references). With TG = 0 the message half vanishes, so the
        // winner lane is pure argmax over base — it must follow the
        // CANDIDATE lane, not stay at lane 0 because the SENDER's base is
        // a per-row constant the argmax cannot see.
        let cfg = DspineConfig {
            n_layer: 5,
            n_embd: 4,
            n_head: 2,
            n_kv_head: 1,
            head_dim: 2,
            n_ff: 4,
            trained_block_size: 4,
            sliding_window: 8,
            rope_theta: 1.0e7,
            rms_eps: 1e-6,
            code_size: 4,
            rank: 2,
            bridge_eps: 1e-6,
            vocab_size: 6,
            mask_token_id: 5,
            head_topk: 2,
            target_layer_ids: vec![1, 16, 31, 46, 61],
            num_target_layers: 64,
        };
        let e = cfg.n_embd;
        let v = cfg.vocab_size;
        // Stash rows all ones; norm_w ones → u = 1 per element.
        let raw = vec![1.0f32; 3 * e];
        let rms = vec![0.0f32, 2.0, 2.0];
        let packed_q = vec![0.0f32; 3 * cfg.rank];
        let candidates = vec![vec![1u32, 2u32], vec![1u32, 2u32]];
        let mut lm_head = vec![0.0f32; v * e];
        for i in 0..e {
            lm_head[1 * e + i] = 1.0; // base = 4
            lm_head[2 * e + i] = 2.0; // base = 8 — must WIN lane 1
        }
        let norm_w = vec![1.0f32; e];
        let gate_b = vec![0.0f32; cfg.rank];
        let tables = SerialTables {
            tg: vec![0.0f32; v * cfg.rank], // s ≡ 0 → score = base[jp]
            tsv: vec![0.5f32; v * 2 * cfg.rank],
        };
        let (idx, probs) = serial_walk_core(
            &raw,
            &rms,
            &packed_q,
            &candidates,
            0,
            &lm_head,
            &norm_w,
            &gate_b,
            &tables,
            &cfg,
        );
        for row in &idx {
            assert_eq!(row, &[1usize, 1usize], "candidate base 8 must win lane 1");
        }
        // mass = exp((4−8)/2) + exp(0) = 1 + e⁻² → EVERY sender lane's
        // confidence = 1/mass (probs is per-SENDER lane; the mass is over
        // the candidate axis).
        let want = 1.0 / (1.0 + (-2.0f32).exp());
        for row in &probs {
            for p in row {
                assert!((p - want).abs() < 1e-5, "confidence = {p}");
            }
        }
        // And the chain token is the high-base candidate.
        let (tokens, _) = chain_walk(&candidates, &idx, &probs);
        assert_eq!(tokens, vec![2u32, 2u32]);
    }

    #[test]
    fn top_k_ids_ascending_stable_lower_id_wins() {
        // Descending values with a tie: the lower id must win the tie slot,
        // and the output must be ASCENDING ids.
        let logits = vec![5.0f32, 1.0, 5.0, 3.0, 0.5];
        assert_eq!(top_k_ids_ascending(&logits, 3), vec![0, 2, 3]);
        // k over the full set → every id, ascending.
        assert_eq!(top_k_ids_ascending(&logits, 5), vec![0, 1, 2, 3, 4]);
        // A tie at the boundary: values [9,9,9,1], k=2 → ids {0,1} not {0,2}.
        let l2 = vec![9.0f32, 9.0, 9.0, 1.0];
        assert_eq!(top_k_ids_ascending(&l2, 2), vec![0, 1]);
    }

    #[test]
    fn parse_config_json_refuses_the_wrong_postures() {
        let good = r#"{
            "architectures": ["DFlashDraftModel"],
            "block_size": 16,
            "code_bridge_second_pass": {"first_exact": 4, "head_topk": 16, "mode": "serial_table"},
            "dflash_config": {
                "draft_block_causal": true,
                "mask_token_id": 248070,
                "selected_prefix_code_bridge": {"code_size": 1024, "eps": 1e-06, "rank": 512},
                "selected_prefix_decode_mode": "serial",
                "selected_prefix_feature_layers": [1, 2, 3, 4, 5],
                "selected_prefix_feature_write_mode": "independent",
                "selected_prefix_feature_writes": true,
                "selected_prefix_last_write_refine": {"head_topk": 16, "kind": "exact_sender", "version": 1, "weight": 1.0},
                "selected_prefix_terminal": false,
                "target_layer_ids": [1, 16, 31, 46, 61]
            },
            "hidden_size": 5120,
            "head_dim": 128,
            "intermediate_size": 17408,
            "is_causal": true,
            "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
            "num_attention_heads": 32,
            "num_hidden_layers": 5,
            "num_key_value_heads": 8,
            "num_target_layers": 64,
            "rms_norm_eps": 1e-06,
            "rope_theta": 10000000.0,
            "sliding_window": 8192,
            "use_sliding_window": true,
            "vocab_size": 248320
        }"#;
        let cfg = parse_config_json(good).unwrap_or_else(|e| panic!("good config refused: {e}"));
        assert_eq!(cfg.head_topk, 16);
        assert_eq!(cfg.mask_token_id, 248_070);
        // A DFlash2 conv present → refuse.
        let bad = good.replace(
            "\"draft_block_causal\": true,",
            "\"draft_block_causal\": true,\n        \"conv_kernel_size\": 2,",
        );
        assert!(
            parse_config_json(&bad).is_err(),
            "conv checkpoint must refuse"
        );
        // Refinement widths disagree → refuse.
        let bad2 = good.replace(
            "\"head_topk\": 16, \"mode\": \"serial_table\"",
            "\"head_topk\": 8, \"mode\": \"serial_table\"",
        );
        assert!(
            parse_config_json(&bad2).is_err(),
            "width mismatch must refuse"
        );
        // Non-sliding layers → refuse.
        let bad3 = good.replace("sliding_attention", "full_attention");
        assert!(
            parse_config_json(&bad3).is_err(),
            "non-native posture must refuse"
        );
    }

    /// The tiny draft world the trace/FD tests share: bf16-representable
    /// synthetic weights at E=16, so the bf16 boundary roundings are exact
    /// identities and finite differences measure the true gradient (the
    /// trainer's validation geometry, plan 492 T1).
    fn test_tiny_cfg() -> DspineConfig {
        DspineConfig {
            n_layer: 5,
            n_embd: 16,
            n_head: 2,
            n_kv_head: 1,
            head_dim: 8,
            n_ff: 24,
            trained_block_size: 4,
            sliding_window: 32,
            rope_theta: 1.0e7,
            rms_eps: 1e-6,
            code_size: 8,
            rank: 4,
            bridge_eps: 1e-6,
            vocab_size: 32,
            mask_token_id: 31,
            head_topk: 4,
            target_layer_ids: vec![1, 2, 3, 4, 5],
            num_target_layers: 64,
        }
    }

    /// (weights, drafter-ready world, lm_head, tables, embed closure). Every
    /// value a multiple of 1/8 — bf16-exact, so roundings never perturb a
    /// finite difference.
    fn test_tiny_world(
        cfg: &DspineConfig,
    ) -> (
        DspineWeights,
        DspineDrafter,
        Vec<f32>,
        SerialTables,
        impl Fn(u32) -> Vec<f32> + '_,
    ) {
        let (e, c, r, nf, v) = (
            cfg.n_embd,
            cfg.code_size,
            cfg.rank,
            cfg.target_layer_ids.len(),
            cfg.vocab_size,
        );
        let q8 = |seed: u64, n: usize| -> Vec<f32> {
            let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (0..n)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    bf16r((((s >> 33) % 17) as i64 - 8) as f32 * 0.125)
                })
                .collect()
        };
        let mut w = DspineWeights {
            fc: q8(1, e * nf * e),
            hidden_norm: q8(2, e),
            model_norm: q8(3, e),
            layers: Vec::new(),
            bridge: DspineBridge {
                codebook: q8(4, v * c),
                early_read_weights: (0..cfg.n_layer - 1)
                    .map(|i| q8(100 + i as u64, c * e))
                    .collect(),
                read_weight: q8(200, c * e),
                feature_write_weights: (0..cfg.n_layer)
                    .map(|i| q8(300 + i as u64, e * r))
                    .collect(),
                gate_w: q8(400, r * c),
                gate_b: q8(401, r),
                pred_gate_w: q8(402, r * c),
                prev_w: q8(403, r * c),
                out_w: vec![0.0; e * e],
            },
        };
        for i in 0..cfg.n_layer {
            w.layers.push(DspineLayer {
                input_norm: q8(500 + i as u64, e),
                q_proj: q8(600 + i as u64, e * e),
                k_proj: q8(700 + i as u64, e * (cfg.n_kv_head * cfg.head_dim)),
                v_proj: q8(800 + i as u64, e * (cfg.n_kv_head * cfg.head_dim)),
                o_proj: q8(900 + i as u64, e * e),
                q_norm: q8(1000 + i as u64, cfg.head_dim),
                k_norm: q8(1100 + i as u64, cfg.head_dim),
                ffn_norm: q8(1200 + i as u64, e),
                gate_proj: q8(1300 + i as u64, cfg.n_ff * e),
                up_proj: q8(1400 + i as u64, cfg.n_ff * e),
                down_proj: q8(1500 + i as u64, e * cfg.n_ff),
            });
        }
        let lm_head = q8(1600, v * e);
        let tables = build_serial_tables(
            &lm_head,
            &w.model_norm,
            &w.bridge.feature_write_weights[cfg.n_layer - 1],
            &w.pv_weight(),
            &w.bridge.codebook,
            cfg,
        );
        let drafter = DspineDrafter::new(cfg.clone(), w.clone());
        let embed = move |tok: u32| -> Vec<f32> {
            let s0 = 1700 + tok as u64 * 13;
            q8(s0, e)
        };
        (w, drafter, lm_head, tables, embed)
    }

    /// The trace twin's parity pin: `draft_block_trace` returns the SAME
    /// draft as `draft_block` (bit-exact tokens/confidences) — the trainer
    /// optimizes the exact deployment graph, never a parallel forward.
    #[test]
    fn draft_block_trace_is_bit_identical_to_draft_block() {
        let cfg = test_tiny_cfg();
        let (w, _drafter, lm_head, tables, embed) = test_tiny_world(&cfg);
        let anchor = 3u32;
        let nf = cfg.target_layer_ids.len();
        let e = cfg.n_embd;
        let ctx_rows = 8usize;
        // bf16-representable synthetic taps (the test world's convention).
        let feats: Vec<f32> = (0..ctx_rows * nf * e)
            .map(|i| bf16r((((i * 7) % 11) as f32 - 5.0) * 0.125))
            .collect();
        let mut a = DspineDrafter::new(cfg.clone(), w.clone());
        a.append_context(&feats, 0).unwrap();
        let mut b = DspineDrafter::new(cfg.clone(), w);
        b.append_context(&feats, 0).unwrap();
        let draft_a = a
            .draft_block(anchor, ctx_rows, &embed, &lm_head, &tables)
            .unwrap();
        let (draft_b, tr) = b
            .draft_block_trace(anchor, ctx_rows, &embed, &lm_head, &tables)
            .unwrap();
        assert_eq!(draft_a.tokens, draft_b.tokens, "tokens diverged");
        assert_eq!(
            draft_a.confidences, draft_b.confidences,
            "confidences diverged"
        );
        assert_eq!(tr.layers.len(), cfg.n_layer);
        assert_eq!(tr.tail.idx.len(), cfg.trained_block_size - 1);
        // The tail base rows carry γ·k entries; every candidate is covered.
        assert_eq!(tr.tail.base.len(), tr.tail.idx.len() * cfg.head_topk);
    }
}
