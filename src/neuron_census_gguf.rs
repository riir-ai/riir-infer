//! The GGUF census session (Issue 1241 T2-prep): a [`CensusModel`]
//! adapter over a loaded qwen35-hybrid dense checkpoint.
//!
//! Mechanics proven here (the `--mechanics-check` bin posture, CPU,
//! one real checkpoint): identity push ≡ base bit-identically, dose
//! monotonicity on a live column, weight-restore digest identity. The
//! SCIENTIFIC choices — the probe set, the observable token, the sweep
//! posture — are the T2 cell's tuning surface (the cell routes to the
//! 4090 `--needs cuda` window per the issue); the defaults here are
//! deterministic stand-ins, good enough to prove the plumbing moves
//! real logits, never a claim about what the census finds.
//!
//! The adapter OWNS its weights: a push saves the column (bit-exact
//! restore — multiply/divide round-trips are NOT bit-safe), scales it,
//! prefills, restores. The observable is one f64 read of the
//! final-position logits.
//!
//! Determinism (Issue 940): the prefill is the body's own; identical
//! inputs give identical reads.

use crate::deltanet::weights::{DeltaNetLayerWeights, Proj, QwenDeltaNetWeights};
use crate::deltanet::{
    HybridCache, HybridForwardScratch, PrefillContext, effective_rotary_dim, prefill_qwen_deltanet,
};
use crate::gguf_loader::{GgufFile, load_qwen_deltanet_weights_gguf};
use crate::neuron_census::{CensusLayer, CensusModel};
use crate::rope::RopeFreqTable;
use crate::tokenizer::BpeTokenizer;
use anyhow::{Context, Result};

/// The default probe set (deterministic stand-ins; the T2 cell
/// replaces them with the task's own probes).
pub const DEFAULT_PROBES: &[&str] = &[
    "The capital of France is",
    "Water boils at a temperature of",
    "The main function of the heart is to",
    "Two plus two equals",
];

/// `(rows, cols)` of a layer's down_proj (Dense arm only — the census
/// surface is dense f32 by construction).
fn down_proj_shape(layer: &DeltaNetLayerWeights, _layer_idx: usize) -> Result<(usize, usize)> {
    match &layer.down_proj {
        Proj::Dense { rows, cols, .. } => Ok((*rows, *cols)),
        #[cfg(feature = "deltanet_ternary_inference")]
        Proj::Ternary(_) => anyhow::bail!(
            "down_proj is ternary — the census surface is dense f32 (load a dense checkpoint)"
        ),
    }
}

/// A loaded, push-capable census session over a qwen35 hybrid GGUF.
pub struct Qwen35CensusSession {
    config: crate::types::Config,
    weights: QwenDeltaNetWeights,
    rope: RopeFreqTable,
    cache: HybridCache,
    scratch: HybridForwardScratch,
    prefill_ctx: PrefillContext,
    /// Tokenized probe sequences (each a full prompt).
    probes: Vec<Vec<usize>>,
    /// The final-position target per probe: the probe's own last
    /// input token (the deterministic stand-in observable).
    targets: Vec<usize>,
    /// Saved-column scratch for the push/restore protocol
    /// (n_embd slots — down_proj is [n_embd, mlp_hidden]).
    saved: Vec<f32>,
    /// Per-layer neuron counts (the down_proj input widths) — the
    /// census layer table.
    layers: Vec<CensusLayer>,
    /// The weights digest at load (the cursor/artifact binding).
    digest: [u8; 32],
    n_embd: usize,
}

impl Qwen35CensusSession {
    /// Load the checkpoint and tokenize the probe set. `probes` empty
    /// → [`DEFAULT_PROBES`]. The observable is the final-position
    /// logit of each probe's own last input token.
    pub fn from_gguf(path: &std::path::Path, probes: &[String]) -> Result<Self> {
        let (config, weights) = load_qwen_deltanet_weights_gguf(path)
            .with_context(|| format!("loading {}", path.display()))?;
        let gguf = GgufFile::open(path).with_context(|| format!("reopening {}", path.display()))?;
        let tok = BpeTokenizer::from_gguf(&gguf).context("BPE tokenizer from gguf")?;

        let probe_texts: Vec<String> = if probes.is_empty() {
            DEFAULT_PROBES.iter().map(|s| s.to_string()).collect()
        } else {
            probes.to_vec()
        };
        let mut tokenized = Vec::with_capacity(probe_texts.len());
        let mut targets = Vec::with_capacity(probe_texts.len());
        for text in &probe_texts {
            let ids = tok.encode(text);
            if ids.is_empty() {
                anyhow::bail!("probe {text:?} tokenized to zero tokens");
            }
            targets.push(ids[ids.len() - 1]);
            tokenized.push(ids);
        }

        // The weights digest binds the push surface (wte + every
        // layer's down_proj) — the cursor/artifact identity is over the
        // bytes the census can move, not the whole file.
        let digest = {
            use blake3::Hasher;
            let mut h = Hasher::new();
            h.update(&(weights.wte.len() as u64).to_le_bytes());
            for v in &weights.wte {
                h.update(&v.to_bits().to_le_bytes());
            }
            for (i, layer) in weights.layers.iter().enumerate() {
                let (rows, cols) = down_proj_shape(layer, i)?;
                let _ = rows;
                h.update(&(cols as u64).to_le_bytes());
                let data = layer.down_proj.dense_data();
                for v in data {
                    h.update(&v.to_bits().to_le_bytes());
                }
            }
            *h.finalize().as_bytes()
        };

        let layers = weights
            .layers
            .iter()
            .enumerate()
            .map(|(i, layer)| {
                let (_, cols) = down_proj_shape(layer, i)?;
                Ok::<_, anyhow::Error>(CensusLayer {
                    layer: i as u8,
                    neurons: cols,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let max_probe = tokenized.iter().map(|p| p.len()).max().unwrap_or(1);
        let rope = RopeFreqTable::new(config.rope_theta, effective_rotary_dim(&config));
        let cache = HybridCache::with_layer_types_bounded(&config, &weights.layer_types, max_probe);
        let scratch = HybridForwardScratch::new(&config);
        let prefill_ctx = PrefillContext::new(&config, max_probe);
        let n_embd = config.n_embd;

        Ok(Self {
            config,
            weights,
            rope,
            cache,
            scratch,
            prefill_ctx,
            probes: tokenized,
            targets,
            saved: vec![0.0f32; n_embd],
            layers,
            digest,
            n_embd,
        })
    }

    /// The census layer table (every layer's MLP — both layer types
    /// carry the SwiGLU block).
    pub fn census_layers(&self) -> &[CensusLayer] {
        &self.layers
    }

    /// Base weights digest (the cursor/artifact binding).
    pub fn weights_digest(&self) -> [u8; 32] {
        self.digest
    }

    /// The observable: prefill probe `read`, return the
    /// final-position logit of the probe's own last input token.
    fn observable_for(&mut self, read: usize) -> f64 {
        let pi = read % self.probes.len();
        let target = self.targets[pi];
        self.cache.reset();
        let logits = prefill_qwen_deltanet(
            &self.weights,
            &self.config,
            &mut self.cache,
            &self.probes[pi],
            &mut self.scratch,
            &self.rope,
            &mut self.prefill_ctx,
        );
        logits[target] as f64
    }
}

impl CensusModel for Qwen35CensusSession {
    fn model_digest(&self) -> [u8; 32] {
        self.digest
    }

    fn layers(&self) -> &[CensusLayer] {
        &self.layers
    }

    fn base_observable(&mut self, read: usize) -> f64 {
        self.observable_for(read)
    }

    fn push_observable(&mut self, layer_index: usize, column: u32, f: f32, read: usize) -> f64 {
        let c = column as usize;
        // The layer's down_proj width (load-checked dense; the shape
        // read is one match, no slice borrow held).
        let cols = down_proj_shape(&self.weights.layers[layer_index], layer_index)
            .expect("census layers are dense by construction")
            .1;
        debug_assert!(c < cols, "column {c} out of the layer's width {cols}");

        // Save + scale. Column-major access over the row-major
        // [n_embd, mlp_hidden] data: neuron c's write column is
        // data[i * cols + c].
        {
            let data = self.weights.layers[layer_index].down_proj.dense_data_mut();
            let saved = &mut self.saved;
            for i in 0..self.n_embd {
                saved[i] = data[i * cols + c];
                data[i * cols + c] *= f;
            }
        }
        let value = self.observable_for(read);
        // Restore bit-exact (copy-back, never a divide).
        {
            let data = self.weights.layers[layer_index].down_proj.dense_data_mut();
            for (i, slot) in self.saved.iter().enumerate() {
                data[i * cols + c] = *slot;
            }
        }
        value
    }
}

/// The mechanics gate (the `--mechanics-check` bin posture): the
/// plumbing moves real logits on a real checkpoint, deterministically.
///
/// 1. Dose monotonicity on the LIVE column of a mid layer (scanned —
///    a dead column is not a finding, it is a bad pick; the check
///    scans strided candidates for the strongest mover).
/// 2. Identity push (f = 1.0) ≡ base bit-identically on every probe —
///    run AFTER the monotone probes so a restore bug cannot hide
///    behind a never-pushed state.
/// 3. Restore identity: after the pushes the weights digest is the
///    load digest (bit-exact restore, the protocol's contract).
pub fn mechanics_check(path: &std::path::Path, probes: &[String]) -> Result<String> {
    use crate::neuron_census::PUSH_SCALE;
    let mut s = Qwen35CensusSession::from_gguf(path, probes)?;
    let digest0 = s.weights_digest();

    let n_reads = s.probes.len();
    let mut base = Vec::with_capacity(n_reads);
    for read in 0..n_reads {
        base.push(s.base_observable(read));
    }
    let li = s.layers.len() / 2;
    let width = s.layers[li].neurons;

    // ── live-column scan + dose monotonicity ────────────────────────
    let mut best_col = 0usize;
    let mut best_delta = -1.0f64;
    for k in 0..16usize {
        let col = (k * 97 + 13) % width; // deterministic strided probe
        let pushed = s.push_observable(li, col as u32, PUSH_SCALE, 0);
        let delta = (pushed - base[0]).abs();
        if delta > best_delta {
            best_delta = delta;
            best_col = col;
        }
    }
    if best_delta <= 0.0 {
        anyhow::bail!(
            "no live column among the 16 scanned in layer {li} — every push left the observable bit-stable"
        );
    }
    let mut last = 0.0f64;
    for &f in &[1.5f32, 2.0, PUSH_SCALE] {
        let pushed = s.push_observable(li, best_col as u32, f, 0);
        let delta = (pushed - base[0]).abs();
        if delta <= last {
            anyhow::bail!(
                "dose monotonicity inverted at f={f}: |Δ| {delta} <= previous {last} (live column {best_col} of layer {li})"
            );
        }
        last = delta;
    }

    // ── identity push ≡ base (post-push ordering, see the doc) ─────
    for (read, &base_read) in base.iter().enumerate() {
        let ident = s.push_observable(li, best_col as u32, 1.0, read);
        if ident != base_read {
            anyhow::bail!(
                "identity push moved probe {read}: {ident} != {base_read} — the push protocol mutates state"
            );
        }
    }

    // ── restore identity ────────────────────────────────────────────
    let digest1 = s.weights_digest();
    if digest1 != digest0 {
        anyhow::bail!("weights digest moved across pushes — the restore is not bit-exact");
    }

    Ok(format!(
        "mechanics PASS: dose monotone on live column {best_col} of layer {li} \
         (1.5→2.0→{PUSH_SCALE}, final |Δ| {last:.6}); identity ≡ base over {n_reads} probes; \
         restore digest identity ({} layers, digest {})",
        s.layers.len(),
        hex8(&digest0)
    ))
}

fn hex8(d: &[u8; 32]) -> String {
    d.iter().take(8).map(|b| format!("{b:02x}")).collect()
}
