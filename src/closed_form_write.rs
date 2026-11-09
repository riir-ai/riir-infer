//! The closed-form fact-write lane (riir-infer Issue 1189) — overlay-at-dequant
//! apply + the teacher-forced capture sink + the [`FactWriteModel`] adapter.
//!
//! Posture (the issue's boundary, verbatim in law):
//!
//! - **The base artifact never moves.** A write is a rank-r FP overlay pair
//!   ([`katgpt_core::quant_error_lora::QuantErrorLora`], reused — never forked)
//!   folded AFTER the down-proj matvec/matmat at the MLP site. Undo is exact
//!   (drop the overlay table), provenance stays clean (the sealed envelope
//!   pins the base BLAKE3 — [`mount`] verifies it against the loaded weights
//!   and refuses a mismatch loud).
//! - **Demote is byte-identical.** `closed_form_overlays: None` (and the
//!   feature off entirely) leaves every forward untouched; the armed path is
//!   one `Option::is_some` branch per layer per MLP block.
//! - **The solve lives katgpt-core-side** ([`katgpt_core::closed_form_weight_write`],
//!   feature-gated there): this module only implements the [`FactWriteModel`]
//!   seam over the loaded dense weights + captures keys/residual through the
//!   batched prefill (the capture-sink pattern of Issue 981's `sink_audit` —
//!   the sink rides [`crate::deltanet::forward::PrefillContext`], cfg-gated,
//!   `None` = never read or written).
//!
//! Cell A = `Qwen3.5-0.8B-Base-Q8_0.gguf` (the qwen35-hybrid dense lane —
//! the `needle_eviction_gate` posture); Cell B = `qwen3.8-27b-Q4_K_M.gguf`
//! (Metal, a later slice — the GPU arm's overlay apply is its own rung).
//! Both postures apply at dequant on structural grounds (R648 §6); the
//! in-lattice pair on Cell A is T4's one honest fragility measurement.

use crate::deltanet::weights::QwenDeltaNetWeights;
use katgpt_core::quant_error_lora::QuantErrorLora;

/// The mounted overlay table: per-EDITED-layer rank-r pairs in the sealed
/// envelope's canonical order, plus the shared rank-wide scratch (one buffer
/// reused across pairs — `apply_correction_into` needs `[rank]`).
///
/// Mounted once at load ([`mount`]); the forward reads `Option<&WriteOverlays>`
/// off the weights struct. Empty per-layer pair lists are legal (a no-op
/// layer) — the envelope records which layers carry pairs.
pub struct WriteOverlays {
    /// `(edited_layer, pairs)` ascending by layer. Pairs in canonical
    /// (fact-asc) order — the envelope's `pairs_all` order.
    layers: Vec<(usize, Vec<QuantErrorLora>)>,
    /// The edited layer ids (ascending) — the stable slice the
    /// `FactWriteModel::edited_layers` seam hands the constructor.
    edited: Vec<usize>,
    /// Max rank across all mounted pairs. The fold's `[rank]` scratch is
    /// CALLER-owned (the forward's reused buffers — the G4 scratch
    /// discipline); this bound pins the debug check.
    max_rank: usize,
}

impl WriteOverlays {
    /// The edited layers this table carries (ascending).
    pub fn edited_layers(&self) -> &[usize] {
        &self.edited
    }

    /// Consume the table back into raw `(layer, pair)` rows — the
    /// re-mount path (`set_layer_overlays`'s replace-this-layer arm).
    pub fn into_rows(self) -> Vec<(usize, QuantErrorLora)> {
        let mut rows = Vec::new();
        for (l, pairs) in self.layers {
            for p in pairs {
                rows.push((l, p));
            }
        }
        rows
    }

    /// Widest mounted rank — the caller sizes the fold scratch to this.
    pub fn max_rank(&self) -> usize {
        self.max_rank
    }

    /// Fold every pair mounted for `layer` into `y`: `y += B·(A·key)` per
    /// pair, canonical order, caller scratch. `layer` not mounted = no-op.
    ///
    /// Deterministic fold order (ascending pair index) per the Issue-940
    /// reduction-order law — the sum crosses no sync boundary today, but the
    /// sidecar/e2e cells compare bit-for-bit, so the order is pinned anyway.
    /// `scratch` must be `>= pair.rank` wide (the forward's reused buffer;
    /// zero-allocation hot path).
    #[inline]
    pub fn apply(&self, layer: usize, key: &[f32], y: &mut [f32], scratch: &mut [f32]) {
        for (l, pairs) in &self.layers {
            if *l != layer {
                continue;
            }
            for pair in pairs.iter() {
                debug_assert!(pair.rank <= self.max_rank, "pair rank past the mount bound");
                debug_assert!(scratch.len() >= pair.rank, "fold scratch too narrow");
                pair.apply_correction_into(key, y, scratch);
            }
            break;
        }
    }
}

/// Build the overlay table from the sealed envelope's pairs, verifying the
/// envelope's base-weights digest against `weights` (the envelope pins the
/// BLAKE3 the solve was constructed over — a mismatched base refuses loud).
///
/// `pairs` is the envelope's [`pairs_all`](katgpt_core::closed_form_weight_write::ClosedFormWriteEnvelope::pairs_all)
/// output: `(edited_layer, pair)` in canonical (fact-asc, layer-asc) order.
#[cfg(feature = "katgpt-core/closed_form_weight_write")]
pub fn mount_from_envelope(
    weights: &QwenDeltaNetWeights,
    envelope: &katgpt_core::closed_form_weight_write::ClosedFormWriteEnvelope,
) -> Result<WriteOverlays, String> {
    let digest = weights_digest(weights);
    let expected = envelope.base_digest();
    if digest != expected {
        return Err(format!(
            "closed_form_write: envelope base digest {:02x?} != loaded weights {:02x?} — refusing to mount",
            &expected[..4],
            &digest[..4]
        ));
    }
    let pairs = envelope.pairs_all();
    Ok(mount(pairs))
}

/// Build the overlay table from raw `(edited_layer, pair)` rows (the T2
/// harness path; the envelope-verified path is [`mount_from_envelope`]).
/// Refuses an empty table (mounting nothing is `None`, not an empty table —
/// the demote posture is explicit at the call site).
pub fn mount(pairs: Vec<(usize, QuantErrorLora)>) -> WriteOverlays {
    assert!(!pairs.is_empty(), "closed_form_write: empty mount — use None");
    let mut layers: Vec<(usize, Vec<QuantErrorLora>)> = Vec::new();
    let mut max_rank = 0usize;
    for (layer, pair) in pairs {
        max_rank = max_rank.max(pair.rank);
        match layers.last_mut() {
            Some((l, ps)) if *l == layer => ps.push(pair),
            _ => layers.push((layer, vec![pair])),
        }
    }
    debug_assert!(
        layers.windows(2).all(|w| w[0].0 < w[1].0),
        "mount: layer rows must ascend (canonical envelope order)"
    );
    WriteOverlays {
        layers,
        edited: layers.iter().map(|(l, _)| *l).collect(),
        max_rank,
    }
}

/// BLAKE3 over the loaded dense weights — the SAME canonical fold the GGUF
/// artifact digest uses is not available here (the weights are dequantized
/// in-memory), so this digest is defined as: BLAKE3 over the streamed
/// f32 little-endian bytes of (wte, final_norm, lm_head, per-layer weights
/// in field order). The envelope must be constructed over the SAME function's
/// output (`FactWriteModel::base_weights_digest` below) — the constructor
/// reads the digest through the seam, so the two sides cannot disagree.
pub fn weights_digest(weights: &QwenDeltaNetWeights) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    let feed = |h: &mut blake3::Hasher, s: &[f32]| {
        for v in s {
            h.update(&v.to_le_bytes());
        }
    };
    feed(&mut h, &weights.wte);
    feed(&mut h, &weights.final_norm);
    // lm_head is tied to wte for Qwen 3.5 — feed it anyway so a future untie
    // is caught by the digest, not by a silent provenance hole.
    if let crate::deltanet::weights::Proj::Dense { data, .. } = &weights.lm_head {
        feed(&mut h, data);
    }
    for layer in &weights.layers {
        feed_layer(h.by_ref(), layer);
    }
    *h.finalize().as_bytes()
}

fn feed_layer(h: &mut blake3::Hasher, layer: &crate::deltanet::weights::DeltaNetLayerWeights) {
    let feed = |h: &mut blake3::Hasher, p: &crate::deltanet::weights::Proj| {
        if let crate::deltanet::weights::Proj::Dense { data, .. } = p {
            for v in data.iter() {
                h.update(&v.to_le_bytes());
            }
        }
    };
    feed(h, &layer.in_proj_qkv);
    feed(h, &layer.in_proj_z);
    feed(h, &layer.in_proj_a);
    feed(h, &layer.in_proj_b);
    feed(h, &layer.out_proj);
    feed(h, &layer.gate_proj);
    feed(h, &layer.up_proj);
    feed(h, &layer.down_proj);
    for v in layer.conv1d_weight.iter() {
        h.update(&v.to_le_bytes());
    }
    // GDN gate + norm parameters — every learned per-layer vector feeds the
    // digest so an added gate parameter cannot silently escape provenance.
    for v in layer.a_log.iter() {
        h.update(&v.to_le_bytes());
    }
    for v in layer.dt_bias.iter() {
        h.update(&v.to_le_bytes());
    }
    for v in layer.linear_norm.iter() {
        h.update(&v.to_le_bytes());
    }
    for v in layer.attn_q_norm.iter() {
        h.update(&v.to_le_bytes());
    }
    for v in layer.attn_k_norm.iter() {
        h.update(&v.to_le_bytes());
    }
    feed(h, &layer.attn_wq);
    feed(h, &layer.attn_wk);
    feed(h, &layer.attn_wv);
    feed(h, &layer.attn_wo);
    for v in layer.input_norm.iter() {
        h.update(&v.to_le_bytes());
    }
    for v in layer.post_attn_norm.iter() {
        h.update(&v.to_le_bytes());
    }
}

/// Fold the mounted overlays for `layer` into a just-computed down-proj output.
/// The ONE helper every MLP call site rides; `None` = zero work.
///
/// Sites (all dense decode bodies + the batched prefill MLP):
/// - `forward_qwen_deltanet_body` / `_pruned_body` / `_evictable_body`
/// - `batched_mlp` (per-position slab rows)
#[inline]
pub fn apply_down_proj(
    overlays: Option<&WriteOverlays>,
    layer: usize,
    key: &[f32],
    y: &mut [f32],
    scratch: &mut [f32],
) {
    if let Some(t) = overlays {
        t.apply(layer, key, y, scratch);
    }
}

// ─── T1: the teacher-forced capture sink (the raw-row tap) ─────────────────

/// Raw-row capture of the MLP tap depth during a whole-prompt prefill — the
/// T1 seam feeding [`katgpt_core::closed_form_weight_write::LayerCapture`].
///
/// The sink rides [`crate::deltanet::forward::PrefillContext`] (the
/// `sink_audit` pattern: `None` = never read or written; the serving path
/// constructs it `None`). While mounted, `batched_mlp` appends, for every
/// EDITED layer, the down-proj input rows (the "key", `[seq_len × mlp]`)
/// and the layer-output residual rows (`[seq_len × n]`) — the exact tensors
/// the solve reads. Only the whole-prompt prefill path captures (chunked
/// prefill never mounts the sink — `capture_at` is a one-shot teacher-forced
/// pass).
#[derive(Default)]
pub struct CaptureSink {
    /// Edited layers to capture (ascending; the adapter's `edited_layers`).
    pub layers: Vec<usize>,
    /// Per captured layer, in `layers` order: keys `[rows × mlp]` row-major
    /// (rows appended in position order).
    pub keys: Vec<Vec<f32>>,
    /// Per captured layer, in `layers` order: layer-output residual rows
    /// `[rows × n]` row-major.
    pub resids: Vec<Vec<f32>>,
    /// Positions captured so far (the harvest's row count).
    pub rows: usize,
    /// mlp width (the key row width) — pinned by the first observe.
    pub mlp: usize,
    /// Residual width — pinned by the first observe.
    pub n: usize,
}

impl CaptureSink {
    /// A sink for `layers` (ascending). Panics loud on an empty set —
    /// capturing nothing is a caller bug, not a demote posture.
    pub fn new(mut layers: Vec<usize>) -> Self {
        assert!(!layers.is_empty(), "capture sink: empty layer set");
        layers.sort_unstable();
        layers.dedup();
        Self {
            layers,
            keys: Vec::new(),
            resids: Vec::new(),
            rows: 0,
            mlp: 0,
            n: 0,
        }
    }

    /// Observe one layer's MLP slab (called from `batched_mlp`): keys from
    /// the down-proj input slab, resids from the post-residual-add slab.
    /// Rows append in position order across layers — the sink holds ALL
    /// captured layers' slabs for THIS prefill; the harvest slices positions.
    pub fn observe_layer(
        &mut self,
        layer: usize,
        keys: &[f32],
        resids: &[f32],
        seq_len: usize,
        mlp: usize,
        n: usize,
    ) {
        if !self.layers.contains(&layer) {
            return;
        }
        if self.mlp == 0 {
            self.mlp = mlp;
            self.n = n;
        }
        debug_assert_eq!(self.mlp, mlp, "capture sink: mlp width changed mid-prefill");
        debug_assert_eq!(self.n, n, "capture sink: residual width changed mid-prefill");
        let idx = self.layers.iter().position(|&l| l == layer).unwrap();
        // First observed layer sizes the row bookkeeping; every layer observes
        // the same seq_len rows in the same prefill.
        while self.keys.len() < self.layers.len() {
            self.keys.push(Vec::new());
            self.resids.push(Vec::new());
        }
        self.keys[idx].extend_from_slice(keys);
        self.resids[idx].extend_from_slice(resids);
        self.rows = seq_len;
    }

    /// Harvest `positions` (ascending, in-bounds) into the constructor's
    /// `LayerCapture` rows — one capture per edited layer, in `layers` order
    /// (`FactWriteModel::edited_layers` order).
    #[cfg(feature = "katgpt-core/closed_form_weight_write")]
    pub fn harvest(
        &self,
        positions: &[usize],
    ) -> Vec<katgpt_core::closed_form_weight_write::LayerCapture> {
        assert_eq!(
            self.keys.len(),
            self.layers.len(),
            "capture sink: harvest before every edited layer observed"
        );
        positions
            .iter()
            .copied()
            .reduce(|a, b| if b > a { b } else { a })
            .map(|max_p| assert!(max_p < self.rows, "capture harvest: position past the prefill"))
            .unwrap_or(());
        self.layers
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let mut keys = Vec::with_capacity(positions.len() * self.mlp);
                let mut resid = Vec::with_capacity(positions.len() * self.n);
                for &p in positions {
                    keys.extend_from_slice(&self.keys[i][p * self.mlp..(p + 1) * self.mlp]);
                    resid.extend_from_slice(&self.resids[i][p * self.n..(p + 1) * self.n]);
                }
                katgpt_core::closed_form_weight_write::LayerCapture {
                    keys,
                    resid,
                    n_pos: positions.len(),
                }
            })
            .collect()
    }
}

// ─── T3: the FactWriteModel adapter (the constructor's seam, riir-infer half)

/// The adapter that lets katgpt-core's closed-form constructor drive THIS
/// repo's dense qwen35-hybrid forward. Owns the model state for the
/// construction session (weights stay shared — the base artifact never
/// moves; overlays mount/demount through [`QwenDeltaNetWeights::
closed_form_overlays`]).
///
/// Gated with the constructor itself (`closed_form_write` ⇒
/// `katgpt-core/closed_form_weight_write`) — without the feature this type
/// does not exist and nothing here can arm.
#[cfg(feature = "katgpt-core/closed_form_weight_write")]
pub struct QwenFactWriteModel<'a> {
    pub weights: &'a mut QwenDeltaNetWeights,
    pub config: &'a Config,
    pub cache: crate::deltanet::forward::HybridCache,
    pub scratch: crate::deltanet::forward::HybridForwardScratch,
    pub prefill_ctx: crate::deltanet::forward::PrefillContext,
    pub rope: crate::rope::RopeFreqTable,
    /// Edited layers (ascending) — the solve's layer spread.
    pub edited: Vec<usize>,
}

#[cfg(feature = "katgpt-core/closed_form_weight_write")]
impl<'a> QwenFactWriteModel<'a> {
    /// Bind the adapter: fresh session state (cache/scratch/prefill ctx),
    /// the final-norm scale measured once (`norm(ones)` per the seam doc).
    pub fn bind(
        weights: &'a mut QwenDeltaNetWeights,
        config: &'a Config,
        edited: Vec<usize>,
    ) -> Self {
        assert!(!edited.is_empty(), "fact-write: empty edited-layer set");
        assert!(
            edited.windows(2).all(|w| w[0] < w[1]),
            "fact-write: edited layers must ascend"
        );
        assert!(
            *edited.last().unwrap() < config.n_layer,
            "fact-write: edited layer past the model"
        );
        Self {
            cache: crate::deltanet::forward::HybridCache::with_layer_types(
                config,
                &weights.layer_types,
            ),
            scratch: crate::deltanet::forward::HybridForwardScratch::new(config),
            prefill_ctx: crate::deltanet::forward::PrefillContext::new(config, 4096),
            rope: crate::rope::RopeFreqTable::new(
                config.rope_theta,
                crate::deltanet::effective_rotary_dim(config),
            ),
            weights,
            config,
            edited,
        }
    }

    /// Teacher-forced prefill over `seq`; returns the last-position logits.
    fn prefill_logits(&mut self, seq: &[u32]) -> Vec<f32> {
        let tokens: Vec<usize> = seq.iter().map(|&t| t as usize).collect();
        let v = self.config.vocab_size;
        let mut logits = vec![0.0f32; v];
        self.prefill_ctx.closed_form_capture = None;
        crate::deltanet::forward::prefill_qwen_deltanet_into(
            self.weights,
            self.config,
            &mut self.cache,
            &tokens,
            &mut self.scratch,
            &self.rope,
            &mut self.prefill_ctx,
            &mut logits,
            None,
        );
        logits
    }
}

#[cfg(feature = "katgpt-core/closed_form_weight_write")]
impl katgpt_core::closed_form_weight_write::FactWriteModel for QwenFactWriteModel<'_> {
    fn dim(&self) -> usize {
        self.config.n_embd
    }

    fn key_dim(&self) -> usize {
        self.config.mlp_hidden
    }

    fn edited_layers(&self) -> &[usize] {
        &self.edited
    }

    fn base_weights_digest(&self) -> [u8; 32] {
        weights_digest(self.weights)
    }

    fn base_layer_norm(&self, layer: usize) -> f64 {
        let w = &self.weights.layers[layer].down_proj;
        // Frobenius norm of the base down_proj — the 25% delta cap's reference.
        match w {
            crate::deltanet::weights::Proj::Dense { data, .. } => {
                // Sequential f64 fold (Issue-940 order pin).
                let mut acc = 0.0f64;
                for v in data.iter() {
                    acc += (*v as f64) * (*v as f64);
                }
                acc.sqrt()
            }
            _ => panic!("fact-write: the dense lane requires dense down_proj"),
        }
    }

    fn final_norm_scale(&self) -> Vec<f32> {
        // `norm(ones)` per dim: the RMSNorm row scale a ones-vector sees.
        let n = self.config.n_embd;
        let eps = self.config.rms_norm_eps as f32;
        let inv = (n as f32 + eps).sqrt();
        self.weights.final_norm.iter().map(|g| g / inv).collect()
    }

    fn token_out_embedding(&self, token: u32) -> Vec<f32> {
        let n = self.config.n_embd;
        let off = token as usize * n;
        self.weights.wte[off..off + n].to_vec()
    }

    fn capture_at(
        &mut self,
        seq: &[u32],
        positions: &[usize],
    ) -> Vec<katgpt_core::closed_form_weight_write::LayerCapture> {
        assert!(
            positions.windows(2).all(|w| w[0] < w[1]),
            "capture_at: positions must ascend"
        );
        assert!(
            positions.last().copied().unwrap_or(0) < seq.len(),
            "capture_at: position past the sequence"
        );
        let tokens: Vec<usize> = seq.iter().map(|&t| t as usize).collect();
        let v = self.config.vocab_size;
        let mut logits = vec![0.0f32; v];
        let edited = self.edited.clone();
        self.prefill_ctx.closed_form_capture = Some(CaptureSink::new(edited));
        crate::deltanet::forward::prefill_qwen_deltanet_into(
            self.weights,
            self.config,
            &mut self.cache,
            &tokens,
            &mut self.scratch,
            &self.rope,
            &mut self.prefill_ctx,
            &mut logits,
            None,
        );
        let sink = self.prefill_ctx.closed_form_capture.take().unwrap();
        sink.harvest(positions)
    }

    fn token_logprob(&mut self, seq: &[u32], pos: usize) -> f64 {
        assert!(pos >= 1, "token_logprob: pos must be >= 1");
        assert!(pos < seq.len(), "token_logprob: pos past the sequence");
        // ln p(seq[pos] | seq[..pos]): teacher-forced prefill over the
        // prefix, f64 log-softmax of the last-position logits, sequential
        // fold (Issue-940 order pin).
        let logits = self.prefill_logits(&seq[..pos]);
        let target = seq[pos] as usize;
        assert!(target < logits.len(), "token_logprob: token past vocab");
        let mut max = f32::NEG_INFINITY;
        for l in logits.iter() {
            if *l > max {
                max = *l;
            }
        }
        let mut lse = 0.0f64;
        for l in logits.iter() {
            lse += ((*l - max) as f64).exp();
        }
        let lse = lse.ln() + max as f64;
        logits[target] as f64 - lse
    }

    fn free_generate(
        &mut self,
        prefix: &[u32],
        max_new: usize,
        stop: Option<u32>,
    ) -> Vec<u32> {
        assert!(!prefix.is_empty(), "free_generate: empty prefix");
        let v = self.config.vocab_size;
        // Prefill bootstraps the caches; the last position's logits pick the
        // first generated token (first-index argmax — the tie law).
        let mut logits = self.prefill_logits(prefix);
        let mut generated = Vec::with_capacity(max_new);
        let mut next_tok = argmax_first(&logits);
        let mut pos = prefix.len();
        loop {
            // "Stopping BEFORE stop": the stop token is excluded from the
            // suffix, even when it is the first pick.
            if Some(next_tok) == stop {
                break;
            }
            generated.push(next_tok);
            if generated.len() >= max_new {
                break;
            }
            let mut x = vec![0.0f32; v.max(self.config.n_embd)];
            let out = crate::deltanet::forward::forward_qwen_deltanet(
                &mut x,
                self.weights,
                &mut self.cache,
                next_tok as usize,
                pos,
                self.config,
                &mut self.scratch,
                &self.rope,
            );
            next_tok = argmax_first(out);
            pos += 1;
        }
        generated
    }

    fn set_layer_overlays(&mut self, layer: usize, pairs: &[QuantErrorLora]) {
        if pairs.is_empty() {
            // Demount THIS layer's pairs: rebuild the table without it.
            match self.weights.closed_form_overlays.take() {
                None => {}
                Some(t) => {
                    let rows: Vec<(usize, QuantErrorLora)> = t.into_rows();
                    let kept: Vec<_> = rows.into_iter().filter(|(l, _)| *l != layer).collect();
                    self.weights.closed_form_overlays = (!kept.is_empty())
                        .then(|| mount(kept));
                }
            }
            return;
        }
        // Arm/extend: mount pairs for this layer over any existing table
        // (replace this layer's rows, keep the others — the constructor's
        // wholesale-per-layer contract).
        let existing: Vec<(usize, QuantErrorLora)> = self
            .weights
            .closed_form_overlays
            .take()
            .map(|t| t.into_rows())
            .unwrap_or_default();
        let mut rows: Vec<(usize, QuantErrorLora)> = existing
            .into_iter()
            .filter(|(l, _)| *l != layer)
            .collect();
        rows.extend(pairs.iter().map(|p| (layer, p.clone())));
        rows.sort_by_key(|(l, _)| *l);
        self.weights.closed_form_overlays = Some(mount(rows));
    }
}

/// Greedy first-index argmax over `logits` (the tie law: the FIRST maximal
/// index wins — `free_generate`'s pick and every decode consumer's).
fn argmax_first(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, l) in logits.iter().enumerate() {
        if *l > best_v {
            best_v = *l;
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(rank: usize, out_dim: usize, in_dim: usize, seed: u64) -> QuantErrorLora {
        // A deterministic rank-r pair: A rows unit-norm-ish, B rows seeded.
        let mut a = Vec::with_capacity(rank * in_dim);
        let mut b = Vec::with_capacity(out_dim * rank);
        let mut s = seed;
        let mut next = move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (u32::MAX as f32)) - 0.5
        };
        for _ in 0..rank * in_dim {
            a.push(next() * 0.1);
        }
        for _ in 0..out_dim * rank {
            b.push(next() * 0.1);
        }
        QuantErrorLora {
            a,
            b,
            alpha: 1.0,
            rank,
            in_dim,
            out_dim,
        }
    }

    #[test]
    fn mount_groups_by_layer_and_orders_canonically() {
        let t = mount(vec![
            (3, pair(2, 8, 16, 1)),
            (1, pair(4, 8, 16, 2)),
            (3, pair(2, 8, 16, 3)),
        ]);
        let ls = t.edited_layers();
        assert_eq!(ls, [1, 3], "layer rows ascend, per-layer pairs grouped");
        assert_eq!(t.max_rank, 4, "scratch sized to the widest rank");
    }

    #[test]
    fn apply_folds_pairs_in_order_and_other_layers_noop() {
        let in_dim = 16usize;
        let out_dim = 8usize;
        let p1 = pair(2, out_dim, in_dim, 11);
        let p2 = pair(2, out_dim, in_dim, 22);
        let t = mount(vec![(2, p1.clone()), (2, p2.clone())]);

        let key: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.05 - 0.4).collect();
        let mut y = vec![1.0f32; out_dim];
        let mut want = y.clone();
        // The pinned order: pair 1 then pair 2 (canonical envelope order).
        let mut scr = vec![0f32; 2];
        p1.apply_correction_into(&key, &mut want, &mut scr);
        p2.apply_correction_into(&key, &mut want, &mut scr);

        let mut scratch = vec![0f32; t.max_rank()];
        t.apply(2, &key, &mut y, &mut scratch);
        assert_eq!(y, want, "fold order + arithmetic pinned");

        // A different layer is a no-op.
        let mut y2 = vec![7.0f32; out_dim];
        t.apply(5, &key, &mut y2, &mut scratch);
        assert_eq!(y2, vec![7.0f32; out_dim]);
    }

    #[test]
    fn apply_down_proj_none_is_zero_work() {
        let mut y = vec![3.5f32; 4];
        apply_down_proj(None, 0, &[0.0; 8], &mut y, &mut []);
        assert_eq!(y, vec![3.5f32; 4], "demote = byte-identical");
    }

    #[test]
    #[should_panic(expected = "empty mount")]
    fn mount_refuses_empty() {
        let _ = mount(Vec::new());
    }

    /// The forward-level G3: with the overlay table MOUNTED but every pair
    /// mounted for a layer the prompt never reaches (or after demount), the
    /// full prefill + decode path is byte-identical to the never-mounted
    /// run; with a pair mounted on a REACHED layer, the logits move. This is
    /// the demote-identity + engagement-proof pair on the real forward
    /// (zeros weights — shapes and plumbing, not numerics).
    #[test]
    fn forward_demote_byte_identical_and_apply_engages() {
        use crate::deltanet::forward::{
            HybridCache, HybridForwardScratch, PrefillContext, forward_qwen_deltanet,
            prefill_qwen_deltanet_into,
        };
        use crate::deltanet::weights::QwenDeltaNetWeights;

        // The unit-test config (the Issue-47 lesson: the real qwen_deltanet
        // preset allocates multi-GiB wte TWICE — shrink vocab/mlp/block only;
        // n_embd and the head geometry stay real).
        let cfg = crate::deltanet::unit_test_config(
            4,
            vec![
                crate::types::DeltaNetLayerType::DeltaNet,
                crate::types::DeltaNetLayerType::Attention,
                crate::types::DeltaNetLayerType::DeltaNet,
                crate::types::DeltaNetLayerType::Attention,
            ],
        );
        let weights = QwenDeltaNetWeights::zeros(&cfg);
        let rope = crate::rope::RopeFreqTable::new(cfg.rope_theta, crate::deltanet::effective_rotary_dim(&cfg));
        let prompt: Vec<usize> = vec![3, 1, 4, 1, 5];
        let v = cfg.vocab_size;

        let run = |w: &QwenDeltaNetWeights| -> (Vec<f32>, Vec<u32>) {
            let mut cache = HybridCache::with_layer_types(&cfg, &w.layer_types);
            let mut scratch = HybridForwardScratch::new(&cfg);
            let mut ctx = PrefillContext::new(&cfg, 64);
            let mut logits = vec![0.0f32; v];
            prefill_qwen_deltanet_into(
                w,
                &cfg,
                &mut cache,
                &prompt,
                &mut scratch,
                &rope,
                &mut ctx,
                &mut logits,
                None,
            );
            let mut toks = Vec::new();
            let mut next = argmax_first(&logits);
            for pos in prompt.len()..prompt.len() + 3 {
                toks.push(next);
                let mut x = vec![0.0f32; v.max(cfg.n_embd)];
                let out = forward_qwen_deltanet(
                    &mut x,
                    w,
                    &mut cache,
                    next as usize,
                    pos,
                    &cfg,
                    &mut scratch,
                    &rope,
                );
                next = argmax_first(out);
            }
            (logits.clone(), toks)
        };

        // Baseline: no table mounted. Each arm rebuilds via zeros() — the
        // struct is GB-class for real models and deliberately NOT Clone.
        let mut base_w = QwenDeltaNetWeights::zeros(&cfg);
        let (base_logits, base_toks) = run(&base_w);

        // Demote posture: mount then demount — the exact-undo law (the table
        // is gone; the forward must be the never-mounted run, bit-for-bit).
        let mut demount_w = QwenDeltaNetWeights::zeros(&cfg);
        let pair_off = pair(2, cfg.n_embd, cfg.mlp_hidden, 7);
        demount_w.closed_form_overlays = Some(mount(vec![(1, pair_off.clone())]));
        demount_w.closed_form_overlays = None; // demount
        let (d_logits, d_toks) = run(&demount_w);
        assert_eq!(d_logits, base_logits, "demount = byte-identical logits");
        assert_eq!(d_toks, base_toks, "demount = byte-identical tokens");

        // Engage: mount a real pair on layer 1 (reached) — logits MUST move
        // (engagement proof; the Issue-1124 lesson).
        let mut armed_w = QwenDeltaNetWeights::zeros(&cfg);
        armed_w.closed_form_overlays = Some(mount(vec![(1, pair_off)]));
        let (a_logits, a_toks) = run(&armed_w);
        let moved = a_logits
            .iter()
            .zip(base_logits.iter())
            .any(|(a, b)| a.to_bits() != b.to_bits());
        assert!(moved, "armed overlay must move the logits (engagement)");
        assert_ne!(a_toks, base_toks, "armed overlay shifts the greedy walk");
    }
}
