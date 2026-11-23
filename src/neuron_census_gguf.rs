//! The GGUF census session (Issue 1241 T2): a [`CensusModel`] adapter
//! over a loaded qwen35-hybrid dense checkpoint, built for the real
//! sweep cell.
//!
//! Observable (the T2 science, replacing the T2-prep stand-in): **mean
//! KL(base ‖ pushed) over the probe set** in f64 — the distributional
//! move a push causes, not one token's logit. The base softmax + LSE
//! are memoized per probe at construction (identity push is therefore
//! exactly `0.0` by the bit-identity law; dose monotonicity and the
//! dominance bars read KL directly). Probes: eight deterministic
//! short prompts spanning recall/arithmetic/semantic/narrative/symbolic
//! domains (`DEFAULT_PROBES`); `MIN_READS` = 8 = the probe count.
//!
//! Push (thread-safe by construction): the census arms
//! `prefill_ctx.census_column_scale = (layer, column, f)` — the
//! forward's `batched_mlp` hook scales the SwiGLU activation column
//! before the down-projection. Mathematically the down_proj
//! column-scale edit; the weights NEVER move, so parallel workers
//! share one `Arc<CensusShared>` read-only (the T2-prep save/scale/
//! restore protocol is gone — there is nothing to restore).
//!
//! Band cache (the sweep's cost lever): every candidate of layer L
//! shares the prefix (layers `0..L`) — the session snapshots each
//! probe's residual at the band boundary once per layer transition
//! and runs only `L..N` per push (`HybridCache::reset_deltanet_range`
//! re-seeds the suffix recurrence; attention KV is overwrite-safe).
//! Bit-identity band ≡ full is the mechanics gate's first assertion
//! (same ops, same order — the cached prefix is the full pass's own
//! boundary state).
//!
//! Determinism (Issue 940): identical inputs give identical reads on
//! both postures; folds are sequential f64.

use crate::deltanet::weights::{DeltaNetLayerWeights, Proj, QwenDeltaNetWeights};
use crate::deltanet::{
    HybridCache, HybridForwardScratch, PrefillContext, effective_rotary_dim,
    prefill_qwen_deltanet_range_into,
};
use crate::gguf_loader::{GgufFile, load_qwen_deltanet_weights_gguf};
use crate::neuron_census::{CensusLayer, CensusModel};
use crate::rope::RopeFreqTable;
use crate::tokenizer::BpeTokenizer;
use crate::types::rmsnorm_with_gamma_eps;
use anyhow::{Context, Result};
use std::sync::Arc;

/// The cell's probe set — eight deterministic short prompts spanning
/// domains (recall, science fact, arithmetic, semantic relation,
/// narrative, procedural, commonsense, symbolic). Short by design:
/// band-prefix memory and the per-forward cost scale with length.
pub const DEFAULT_PROBES: &[&str] = &[
    "The capital of France is",
    "Water boils at a temperature of",
    "Two plus two equals",
    "The opposite of hot is",
    "Once upon a time there was a",
    "To bake a cake you first",
    "The weather today is",
    "E = mc",
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

/// The immutable, shareable census payload: weights + config + rope +
/// the tokenized probes + the memoized base distribution per probe +
/// the sweep's identity and layer table. One `Arc` per process; every
/// worker session reads it read-only (the activation-scale push never
/// touches these bytes).
pub struct CensusShared {
    config: crate::types::Config,
    weights: QwenDeltaNetWeights,
    rope: RopeFreqTable,
    probes: Vec<Vec<usize>>,
    /// Per probe: the base softmax in f64 (`exp(lb − lse)`) — the KL
    /// weight vector, computed once.
    base_p: Vec<Vec<f64>>,
    /// Per probe: the base logits (f32, the forward's own output) —
    /// the `lb` side of `Σ p·(lb − lp)`.
    base_logits: Vec<Vec<f32>>,
    /// Per probe: the base log-sum-exp.
    base_lse: Vec<f64>,
    digest: [u8; 32],
    layers: Vec<CensusLayer>,
    n_embd: usize,
    vocab: usize,
    /// Longest probe (the cache/ctx capacity).
    max_probe: usize,
    /// Layers carrying a DeltaNet recurrence (the band reset targets).
    n_layers: usize,
}

impl CensusShared {
    /// Load the checkpoint, tokenize the probes, memoize the base
    /// distribution (one full prefill per probe), and derive the
    /// identity + layer table.
    pub fn from_gguf(path: &std::path::Path, probes: &[String]) -> Result<Arc<Self>> {
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
        for text in &probe_texts {
            let ids = tok.encode(text);
            if ids.is_empty() {
                anyhow::bail!("probe {text:?} tokenized to zero tokens");
            }
            tokenized.push(ids);
        }
        let max_probe = tokenized.iter().map(|p| p.len()).max().unwrap_or(1);

        // The weights digest binds the census surface (wte + every
        // layer's down_proj) — the cursor/artifact identity is over the
        // bytes the census READS (the push never writes them; the
        // digest is now a pure canary, checked after every gate run).
        let digest = {
            use blake3::Hasher;
            let mut h = Hasher::new();
            h.update(&(weights.wte.len() as u64).to_le_bytes());
            for v in &weights.wte {
                h.update(&v.to_bits().to_le_bytes());
            }
            for (i, layer) in weights.layers.iter().enumerate() {
                let (_, cols) = down_proj_shape(layer, i)?;
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
        let n_layers = weights.layers.len();

        let rope = RopeFreqTable::new(config.rope_theta, effective_rotary_dim(&config));
        let n_embd = config.n_embd;
        let vocab = config.vocab_size;

        // The base distribution per probe — one full prefill each
        // (embed → 0..N → norm → head, the band path's own code at the
        // full range).
        let mut base_logits = Vec::with_capacity(tokenized.len());
        let mut base_p = Vec::with_capacity(tokenized.len());
        let mut base_lse = Vec::with_capacity(tokenized.len());
        {
            let mut cache =
                HybridCache::with_layer_types_bounded(&config, &weights.layer_types, max_probe);
            let mut scratch = HybridForwardScratch::new(&config);
            let mut ctx = PrefillContext::new(&config, max_probe);
            let mut logits = vec![0.0f32; vocab];
            for toks in &tokenized {
                cache.reset_deltanet_range(0..n_layers);
                embed_rows(&weights, &mut ctx, toks, n_embd);
                prefill_qwen_deltanet_range_into(
                    &weights,
                    &config,
                    &mut cache,
                    &mut scratch,
                    &rope,
                    &mut ctx,
                    toks.len(),
                    0..n_layers,
                );
                norm_head_into(&weights, &config, &mut ctx, toks.len(), &mut logits);
                let (p, lse) = softmax_f64(&logits);
                if !lse.is_finite() {
                    anyhow::bail!("base log-sum-exp diverged on a probe — refusing to sweep");
                }
                base_logits.push(logits.clone());
                base_p.push(p);
                base_lse.push(lse);
            }
        }

        Ok(Arc::new(Self {
            config,
            weights,
            rope,
            probes: tokenized,
            base_p,
            base_logits,
            base_lse,
            digest,
            layers,
            n_embd,
            vocab,
            max_probe,
            n_layers,
        }))
    }

    /// The full census layer table.
    pub fn census_layers(&self) -> &[CensusLayer] {
        &self.layers
    }

    /// Base weights digest (the cursor/artifact binding).
    pub fn weights_digest(&self) -> [u8; 32] {
        self.digest
    }

    /// The probe count (the sweep's `reads`).
    pub fn n_probes(&self) -> usize {
        self.probes.len()
    }

    /// Build a worker session over the shared payload. `layer_filter`
    /// selects a subset of the table (the `--layers` posture — sweep a
    /// band into its own state dir); the session maps table positions
    /// back to real layer ids internally. An empty filter refuses loud.
    pub fn session(
        self: &Arc<Self>,
        layer_filter: Option<std::ops::Range<usize>>,
    ) -> Result<Qwen35CensusSession> {
        let selected: Vec<usize> = match layer_filter {
            Some(r) => {
                let end = r.end.min(self.layers.len());
                anyhow::ensure!(
                    r.start < end,
                    "layer filter {}..{} selects no layer of {}",
                    r.start,
                    r.end,
                    self.layers.len()
                );
                (r.start..end).collect()
            }
            None => (0..self.layers.len()).collect(),
        };
        let layer_rows: Vec<CensusLayer> = selected.iter().map(|&i| self.layers[i]).collect();
        let n_probes = self.probes.len();
        Ok(Qwen35CensusSession {
            shared: Arc::clone(self),
            cache: HybridCache::with_layer_types_bounded(
                &self.config,
                &self.weights.layer_types,
                self.max_probe,
            ),
            scratch: HybridForwardScratch::new(&self.config),
            prefill_ctx: PrefillContext::new(&self.config, self.max_probe),
            logits: vec![0.0f32; self.vocab],
            band: None,
            band_prefix: (0..n_probes).map(|_| Vec::new()).collect(),
            band_seq: vec![0; n_probes],
            selected,
            layer_rows,
        })
    }
}

/// Embed `toks` into the ctx's hidden rows (the range prefill's step 1,
/// verbatim).
fn embed_rows(weights: &QwenDeltaNetWeights, ctx: &mut PrefillContext, toks: &[usize], n: usize) {
    let hidden = ctx.hidden_rows_mut();
    for (p, &t) in toks.iter().enumerate() {
        let off = t * n;
        hidden[p * n..(p + 1) * n].copy_from_slice(&weights.wte[off..off + n]);
    }
}

/// Final RMSNorm on the last row + LM head into `logits` (the range
/// prefill's caller-side steps 3–4, verbatim).
fn norm_head_into(
    weights: &QwenDeltaNetWeights,
    config: &crate::types::Config,
    ctx: &mut PrefillContext,
    seq_len: usize,
    logits: &mut [f32],
) {
    let n = config.n_embd;
    let v = config.vocab_size;
    let last = (seq_len - 1) * n;
    rmsnorm_with_gamma_eps(
        &mut ctx.hidden_rows_mut()[last..last + n],
        &weights.final_norm,
        config.rms_norm_eps,
    );
    debug_assert!(
        logits.len() >= v,
        "logits buffer shorter than the vocabulary"
    );
    weights
        .lm_head
        .matvec(&ctx.hidden_rows()[last..last + n], &mut logits[..v]);
}

/// f64 softmax + log-sum-exp over f32 logits (no max-shift — the f64
/// range absorbs any realistic logit; a divergent LSE is refused by
/// the caller).
fn softmax_f64(logits: &[f32]) -> (Vec<f64>, f64) {
    let mut acc = 0.0f64;
    let mut out = Vec::with_capacity(logits.len());
    for &l in logits {
        let e = (l as f64).exp();
        acc += e;
        out.push(e);
    }
    let lse = acc.ln();
    for v in &mut out {
        *v /= acc;
    }
    (out, lse)
}

/// KL(base ‖ pushed) in one pass over the vocabulary:
/// `Σ p_b·(lb − lp) + lse_p − lse_b`. The base softmax/LSE are
/// memoized (the only exp is the pushed side's LSE sum).
fn kl_vs_base(logits: &[f32], base_logits: &[f32], base_p: &[f64], base_lse: f64) -> f64 {
    let mut s = 0.0f64;
    let mut acc = 0.0f64;
    for (i, &l) in logits.iter().enumerate() {
        let lf = l as f64;
        s += lf.exp();
        acc += base_p[i] * (base_logits[i] as f64 - lf);
    }
    acc + s.ln() - base_lse
}

/// A worker session: per-thread cache/scratch/ctx + the band cache.
/// `Send` by construction (owned data + `Arc<CensusShared>`, which is
/// `Send + Sync`); the push never writes the shared weights.
pub struct Qwen35CensusSession {
    shared: Arc<CensusShared>,
    cache: HybridCache,
    scratch: HybridForwardScratch,
    prefill_ctx: PrefillContext,
    /// Reused logits buffer (the head's output, one push at a time).
    logits: Vec<f32>,
    /// The band cache: the table layer index currently snapshotted.
    band: Option<usize>,
    /// Per probe: the residual at the band boundary `[seq · n_embd]`.
    band_prefix: Vec<Vec<f32>>,
    /// Per probe: the snapshotted sequence length.
    band_seq: Vec<usize>,
    /// Table position → real layer index (the `--layers` filter).
    selected: Vec<usize>,
    /// The filtered census layer table (what `layers()` reports).
    layer_rows: Vec<CensusLayer>,
}

impl Qwen35CensusSession {
    /// Snapshot the prefix (layers `0..L`) per probe — once per layer
    /// transition. The prefix deltanet state stays pristine through the
    /// whole band (candidates only re-run `L..N`).
    fn band_ensure(&mut self, tli: usize) {
        if self.band == Some(tli) {
            return;
        }
        let rl = self.selected[tli];
        let n = self.shared.n_embd;
        for (pi, toks) in self.shared.probes.iter().enumerate() {
            self.cache.reset_deltanet_range(0..rl);
            embed_rows(&self.shared.weights, &mut self.prefill_ctx, toks, n);
            prefill_qwen_deltanet_range_into(
                &self.shared.weights,
                &self.shared.config,
                &mut self.cache,
                &mut self.scratch,
                &self.shared.rope,
                &mut self.prefill_ctx,
                toks.len(),
                0..rl,
            );
            let bytes = toks.len() * n;
            let buf = &mut self.band_prefix[pi];
            buf.clear();
            buf.extend_from_slice(&self.prefill_ctx.hidden_rows()[..bytes]);
            self.band_seq[pi] = toks.len();
        }
        self.band = Some(tli);
    }

    /// The observable read for probe `pi` under the ARMED push: run the
    /// posture's layer range, then norm + head + KL vs the memoized
    /// base. Non-finite KL refuses loud (a corrupted read must never
    /// land in a row as a number).
    fn read_kl(&mut self, pi: usize, rl: usize, band: bool) -> f64 {
        let n = self.shared.n_embd;
        let seq = if band {
            self.cache.reset_deltanet_range(rl..self.shared.n_layers);
            let seq = self.band_seq[pi];
            self.prefill_ctx.hidden_rows_mut()[..seq * n]
                .copy_from_slice(&self.band_prefix[pi][..seq * n]);
            prefill_qwen_deltanet_range_into(
                &self.shared.weights,
                &self.shared.config,
                &mut self.cache,
                &mut self.scratch,
                &self.shared.rope,
                &mut self.prefill_ctx,
                seq,
                rl..self.shared.n_layers,
            );
            seq
        } else {
            self.cache.reset_deltanet_range(0..self.shared.n_layers);
            let toks = self.shared.probes[pi].as_slice();
            embed_rows(&self.shared.weights, &mut self.prefill_ctx, toks, n);
            prefill_qwen_deltanet_range_into(
                &self.shared.weights,
                &self.shared.config,
                &mut self.cache,
                &mut self.scratch,
                &self.shared.rope,
                &mut self.prefill_ctx,
                toks.len(),
                0..self.shared.n_layers,
            );
            toks.len()
        };
        norm_head_into(
            &self.shared.weights,
            &self.shared.config,
            &mut self.prefill_ctx,
            seq,
            &mut self.logits,
        );
        let v = kl_vs_base(
            &self.logits,
            &self.shared.base_logits[pi],
            &self.shared.base_p[pi],
            self.shared.base_lse[pi],
        );
        assert!(
            v.is_finite(),
            "census read is non-finite (probe {pi}, layer {rl}) — a corrupt read refuses loud"
        );
        v
    }

    /// The FULL-path push (the band gate's reference arm; `--no-band`):
    /// same observable, no prefix cache — embed + run the whole stack
    /// per read. The session carries no band snapshot in this mode.
    pub fn push_full(&mut self, tli: usize, column: u32, f: f32, read: usize) -> f64 {
        let rl = self.selected[tli];
        let pi = read % self.shared.probes.len();
        self.prefill_ctx.census_column_scale = Some((rl, column, f));
        let v = self.read_kl(pi, rl, false);
        self.prefill_ctx.census_column_scale = None;
        v
    }
}

impl CensusModel for Qwen35CensusSession {
    fn model_digest(&self) -> [u8; 32] {
        self.shared.digest
    }

    fn layers(&self) -> &[CensusLayer] {
        &self.layer_rows
    }

    /// KL base is exactly zero for every read (the memoized softmax is
    /// the zero-delta reference) — the memoization IS this constant.
    fn base_observable(&mut self, _read: usize) -> f64 {
        0.0
    }

    fn push_observable(&mut self, layer_index: usize, column: u32, f: f32, read: usize) -> f64 {
        let rl = self.selected[layer_index];
        let pi = read % self.shared.probes.len();
        self.band_ensure(layer_index);
        self.prefill_ctx.census_column_scale = Some((rl, column, f));
        let v = self.read_kl(pi, rl, true);
        self.prefill_ctx.census_column_scale = None;
        v
    }
}

/// The mechanics gate v2 (the `--mechanics-check` bin posture) on a
/// real checkpoint:
///
/// 1. **Band ≡ full bit-identity** — the same push measured through the
///    band cache and through a fresh full-stack read reads the SAME
///    f64 bits, at the first, a middle and the last layer, at f = 1
///    and the pinned push scale (this is the prefix-cache soundness
///    proof: the cached boundary is the full pass's own state).
/// 2. **Identity push = exactly 0.0** (both postures) — `f = 1.0`
///    multiplies by exact 1.0, the logits are bit-stable, KL vanishes.
/// 3. **Dose monotonicity** on a LIVE column of a middle layer (KL at
///    1.5 → 2.0 → the pinned scale strictly grows; the column is
///    scanned — a dead column is a bad pick, not a finding).
/// 4. **Weights-digest canary** — the push is activation-side; the
///    digest must be the load digest after every arm (trivially true
///    by construction, kept as the loud tripwire it was in T2-prep).
pub fn mechanics_check(path: &std::path::Path, probes: &[String]) -> Result<String> {
    use crate::neuron_census::PUSH_SCALE;
    let shared = CensusShared::from_gguf(path, probes)?;
    let digest0 = shared.weights_digest();
    let n_layers = shared.n_layers;
    let n_reads = shared.n_probes();
    let mut banded = shared.session(Some(0..n_layers))?;
    let mut full = shared.session(None)?;

    // ── 1. band ≡ full, bit-exact, three layers × two factors ─────
    for &rl in &[0usize, n_layers / 2, n_layers - 1] {
        let width = shared.layers[rl].neurons;
        let col = (13 + 97 * (rl + 1)) % width;
        for &f in &[1.0f32, PUSH_SCALE] {
            // Table position of real layer rl under the banded
            // session's FULL table (it spans everything, so 0..N): the
            // filter here is the whole model — positions equal ids.
            let band_v = banded.push_observable(rl, col as u32, f, 0);
            let full_v = full.push_full(rl, col as u32, f, 0);
            if band_v.to_bits() != full_v.to_bits() {
                anyhow::bail!(
                    "band ≡ full bit-identity FAILED at layer {rl} col {col} f={f}: \
                     band {band_v:?} vs full {full_v:?}"
                );
            }
        }
    }

    // ── 2. identity push = 0.0 exactly, both postures ──────────────
    for read in 0..n_reads {
        let v = banded.push_observable(0, 0, 1.0, read);
        if v.to_bits() != 0u64 {
            anyhow::bail!("identity push moved probe {read}: KL {v} != 0 exactly");
        }
    }
    for read in 0..n_reads.min(2) {
        let v = full.push_full(0, 0, 1.0, read);
        if v.to_bits() != 0u64 {
            anyhow::bail!("identity push (full posture) moved probe {read}: KL {v} != 0");
        }
    }

    // ── 3. dose monotonicity on a live middle-layer column ────────
    let li = n_layers / 2;
    let width = shared.layers[li].neurons;
    let mut best_col = 0usize;
    let mut best_delta = -1.0f64;
    for k in 0..16usize {
        let col = (k * 97 + 13) % width;
        let pushed = banded.push_observable(li, col as u32, PUSH_SCALE, 0);
        if pushed > best_delta {
            best_delta = pushed;
            best_col = col;
        }
    }
    if best_delta <= 0.0 {
        anyhow::bail!(
            "no live column among the 16 scanned in layer {li} — every push left the KL at zero"
        );
    }
    let mut last = 0.0f64;
    for &f in &[1.5f32, 2.0, PUSH_SCALE] {
        let pushed = banded.push_observable(li, best_col as u32, f, 0);
        if pushed <= last {
            anyhow::bail!(
                "dose monotonicity inverted at f={f}: KL {pushed} <= previous {last} \
                 (live column {best_col} of layer {li})"
            );
        }
        last = pushed;
    }

    // ── 4. digest canary ──────────────────────────────────────────
    let digest1 = shared.weights_digest();
    if digest1 != digest0 {
        anyhow::bail!("weights digest moved across pushes — the push must never write weights");
    }

    Ok(format!(
        "mechanics PASS: band ≡ full bit-identical (layers 0/{}/{}, f=1 & {PUSH_SCALE}); \
         identity KL ≡ 0 exactly over {n_reads} probes (both postures); \
         dose monotone on live column {best_col} of layer {li} \
         (1.5→2.0→{PUSH_SCALE}, final KL {last:.6}); \
         weights digest canary ({n_layers} layers, {})",
        n_layers / 2,
        n_layers - 1,
        hex8(&digest0)
    ))
}

fn hex8(d: &[u8; 32]) -> String {
    d.iter().take(8).map(|b| format!("{b:02x}")).collect()
}
