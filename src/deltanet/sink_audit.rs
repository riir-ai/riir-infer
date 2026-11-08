//! The sink-audit census lane (katgpt-rs Issue 981 T1/T2, `sink_audit`).
//!
//! One pass per checkpoint, offline: run the hybrid decode forward over the
//! first N tokens, capture each FULL-ATTENTION layer's post-RoPE Q rows at
//! the moment the layer runs, then harvest the per-head softmax mass those
//! queries put on the first N positions — through
//! [`katgpt_core::kv_sink_window`]'s audit core, which owns the profile, the
//! chosen cut points, and the policy selector.
//!
//! # Scope
//!
//! Full-attention layers ONLY ([`DeltaNetLayerType::Attention`]). GDN layers
//! have no softmax sink; a blended measurement dilutes the signal the verdict
//! reads.
//!
//! # Determinism (G1)
//!
//! The census drives the incumbent decode forward token-by-token (pos 0..N,
//! layers ascending), then harvests (layers ascending, heads ascending,
//! queries ascending) with sequential f32 rows accumulating into f64 — one
//! fixed order, no RNG, no wall clock. Same checkpoint + corpus + N → the
//! same profile bits (the bin pins this with a BLAKE3 sidecar).
//!
//! # Sync boundary
//!
//! None — an offline diagnostic. Nothing here is synced, and nothing here
//! runs in a serving loop (the capture rides an `Option` that is `None`
//! everywhere else, cfg-gated so a flag-off build compiles the field itself
//! out).
//!
//! # What the audit is NOT
//!
//! It does not gate the forward (Plan 287's classifier stays the in-engine
//! lane), and it does not touch the token-gate reserve (a correctness margin,
//! not an eviction budget — Issue 981 T2's exclusion, recorded with reason).

use crate::deltanet::forward::{HybridCache, HybridForwardScratch, forward_qwen_deltanet};
use crate::deltanet::ternary_forward::forward_qwen_deltanet_ternary;
use crate::deltanet::ternary_weights::QwenDeltaNetTernaryWeights;
use crate::deltanet::weights::QwenDeltaNetWeights;
use crate::rope::RopeFreqTable;
use crate::types::{Config, DeltaNetLayerType};
use katgpt_core::kv_sink_window::{SinkAuditProfile, accumulate_head_mass_into, finalize_profile};

/// The Q-row capture riding [`HybridForwardScratch`] (the `scale_invariant`
/// precedent: session state beside the buffers the kernels already take).
///
/// Armed ONLY by the census pass; every other construction leaves it `None`,
/// and the field is cfg-gated so a flag-off build never carries it.
pub struct SinkAuditQCapture {
    /// Slot `s` holds `[N × q_dim]` post-RoPE Q rows for full-attention
    /// layer `attn_layers[s]` (parallel arrays — `q_rows.len() ==
    /// attn_layers.len()`).
    pub q_rows: Vec<Vec<f32>>,
    /// The `layer_idx` values that are full-attention layers, ascending.
    pub attn_layers: Vec<usize>,
    /// `layer_idx → slot` (None for GDN layers), built once.
    slot_of_layer: Vec<Option<usize>>,
    /// The position the caller is forwarding this step; the body's capture
    /// writes row `pos` of the matching layer.
    pub pos: usize,
    /// `N` — the capture width every row buffer was sized for.
    pub n_positions: usize,
}

impl std::fmt::Debug for SinkAuditQCapture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Shapes only — the rows are megabyte-scale.
        f.debug_struct("SinkAuditQCapture")
            .field("attn_layers", &self.attn_layers)
            .field("slots", &self.q_rows.len())
            .field("n_positions", &self.n_positions)
            .field("pos", &self.pos)
            .finish()
    }
}

impl SinkAuditQCapture {
    /// Size the capture for one census pass: `n_positions` rows per
    /// full-attention layer, `q_dim = n_head * head_dim` wide.
    pub fn new(config: &Config, layer_types: &[DeltaNetLayerType], n_positions: usize) -> Self {
        let q_dim = config.n_head * config.head_dim;
        let mut attn_layers = Vec::new();
        let mut slot_of_layer = vec![None; layer_types.len()];
        for (idx, lt) in layer_types.iter().enumerate() {
            if *lt == DeltaNetLayerType::Attention {
                slot_of_layer[idx] = Some(attn_layers.len());
                attn_layers.push(idx);
            }
        }
        let q_rows = attn_layers
            .iter()
            .map(|_| vec![0.0f32; n_positions * q_dim])
            .collect();
        Self {
            q_rows,
            attn_layers,
            slot_of_layer,
            pos: 0,
            n_positions,
        }
    }

    /// Capture one layer's Q rows for the current position. Called from the
    /// hybrid body AFTER [`forward_attention_layer`-equivalent] ran — the
    /// attention step reads `q_buf` last at step 5 and never writes it again,
    /// so the buffer still holds post-RoPE Q here.
    pub(crate) fn capture(&mut self, layer_idx: usize, q_buf: &[f32]) {
        if let Some(slot) = self.slot_of_layer[layer_idx] {
            let row_off = self.pos * q_buf.len();
            let dst = &mut self.q_rows[slot][row_off..row_off + q_buf.len()];
            dst.copy_from_slice(q_buf);
        }
    }
}

/// The census result: the audit profile plus the surface the claims are
/// scoped to.
#[derive(Debug)]
pub struct SinkCensus {
    pub profile: SinkAuditProfile,
    /// The census window N (== `profile.positions_scanned`).
    pub n_positions: usize,
    /// The tokens actually consumed (the first N of the encoded corpus).
    pub tokens_used: usize,
    /// The census-window K/V the profile was scored against (issue 1185 T1,
    /// the width-census lane's input). Full-attention layers only, slot order
    /// matching the profile's layer scope; an offline diagnostic that rides
    /// the gated module — never a serving surface.
    pub kv: SinkAuditKVCapture,
    /// The Q capture the census armed, kept for the offline lanes that score
    /// Q against the captured K/V (the width census). `None` on the dense
    /// twin — its Q rows are consumed inside the probe's arithmetic and never
    /// materialized per layer.
    pub q: Option<SinkAuditQCapture>,
}

/// The K/V capture for the width-census lane (issue 1185 T1): the census
/// window's post-RoPE K and V rows for every full-attention layer, extracted
/// FROM THE KV CACHE at census-finish time.
///
/// Why extraction, not a forward-body capture: the census cache is sized to
/// exactly the window (`census_posture`), each position's row is written once
/// during the decode pass, and the harvest itself scores the profile against
/// these very rows — so the cache holds the forward-time K/V bit-exactly, and
/// a body-level capture would duplicate the write in exactly the aliasing
/// class Issue 594 paid for. One extraction, shared by the hybrid and dense
/// twin finishes; zero forward-body edits; the capture exists only under the
/// `sink_audit` feature (the whole module is gated).
pub struct SinkAuditKVCapture {
    /// Slot `s` holds the `[n_positions × kv_dim]` row-major K rows of
    /// full-attention layer `attn_layers[s]` (parallel arrays — same slot
    /// order as the Q capture and the profile's layer scope).
    pub key_rows: Vec<Vec<f32>>,
    /// The matching V rows, same layout.
    pub value_rows: Vec<Vec<f32>>,
    /// The `layer_idx` values that are full-attention layers, ascending —
    /// identical to the Q capture's list.
    pub attn_layers: Vec<usize>,
    /// The census window N every row buffer holds.
    pub n_positions: usize,
    /// GQA geometry the width-census math (T2) splits heads by.
    pub kv_dim: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
}

impl SinkAuditKVCapture {
    /// Extract the window rows for `attn_layers` from a KV-cache-like
    /// surface: `key_value_of_layer(slot)` returns the layer's full K and V
    /// buffers (`[capacity × kv_dim]`; capacity may exceed the window —
    /// exactly the first `n_positions` rows are copied). Sequential copies,
    /// no arithmetic: the bytes are the cache's own, so the capture is as
    /// deterministic as the forward that filled it.
    pub(crate) fn extract<'a>(
        attn_layers: &[usize],
        mut key_value_of_layer: impl FnMut(usize) -> (&'a [f32], &'a [f32]),
        n_positions: usize,
        kv_dim: usize,
        n_kv_head: usize,
        head_dim: usize,
    ) -> Self {
        let want = n_positions * kv_dim;
        let mut key_rows = Vec::with_capacity(attn_layers.len());
        let mut value_rows = Vec::with_capacity(attn_layers.len());
        for slot in 0..attn_layers.len() {
            let (key, value) = key_value_of_layer(slot);
            debug_assert!(
                key.len() >= want && value.len() >= want,
                "kv cache rows shorter than the census window: key {} value {} want {want}",
                key.len(),
                value.len()
            );
            key_rows.push(key[..want.min(key.len())].to_vec());
            value_rows.push(value[..want.min(value.len())].to_vec());
        }
        Self {
            key_rows,
            value_rows,
            attn_layers: attn_layers.to_vec(),
            n_positions,
            kv_dim,
            n_kv_head,
            head_dim,
        }
    }

    /// Gather one KV head's rows into a contiguous `[n_positions ×
    /// head_dim]` block in `dst` (the cache/capture layout strides heads by
    /// `kv_dim`; the width-census math wants the per-head block contiguous).
    /// Reuse the buffer: `clear()` + extend, allocated once at the caller.
    pub fn gather_kv_head(&self, slot: usize, kvh: usize, dst: &mut Vec<f32>) {
        assert!(slot < self.key_rows.len(), "slot {slot} out of range");
        assert!(kvh < self.n_kv_head, "kvh {kvh} out of range");
        let rows = &self.key_rows[slot];
        let head_off = kvh * self.head_dim;
        dst.clear();
        dst.reserve(self.n_positions * self.head_dim);
        for pos in 0..self.n_positions {
            let off = pos * self.kv_dim + head_off;
            dst.extend_from_slice(&rows[off..off + self.head_dim]);
        }
    }

    /// Shapes-only debug — never the row contents (megabyte-scale).
    fn fmt_shapes(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SinkAuditKVCapture")
            .field("attn_layers", &self.attn_layers)
            .field("n_positions", &self.n_positions)
            .field("kv_dim", &self.kv_dim)
            .field("n_kv_head", &self.n_kv_head)
            .field("head_dim", &self.head_dim)
            .field("key_rows_slots", &self.key_rows.len())
            .field("value_rows_slots", &self.value_rows.len())
            .finish()
    }
}

impl std::fmt::Debug for SinkAuditKVCapture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.fmt_shapes(f)
    }
}

/// Run the census over the first `n_positions` tokens of `tokens`.
///
/// Drives the incumbent decode forward (no census-specific arithmetic in the
/// model), harvests the softmax mass per (full-attention layer, head, query),
/// and finalizes the audit profile. `m_act` is the mean per-layer max V-row
/// L2 norm over the window (the paper's secondary signal; disclosed, never
/// verdict-bearing — the fitted_v_gate norm class).
pub fn run_census(
    weights: &QwenDeltaNetWeights,
    config: &Config,
    tokens: &[usize],
    rope_freq: &RopeFreqTable,
    n_positions: usize,
) -> Result<SinkCensus, String> {
    if tokens.len() < n_positions {
        return Err(format!(
            "corpus encodes {} tokens < the census window {n_positions}",
            tokens.len()
        ));
    }
    let census_config = census_posture(config, n_positions);
    let mut cache =
        HybridCache::with_layer_types_bounded(&census_config, &weights.layer_types, n_positions);
    let mut scratch = HybridForwardScratch::new(&census_config);
    arm_capture(&mut scratch, config, &weights.layer_types, n_positions);

    let mut logits = vec![0.0f32; config.vocab_size];
    let tokens_used = n_positions;
    for (pos, &token) in tokens.iter().take(n_positions).enumerate() {
        set_capture_pos(&mut scratch, pos);
        forward_qwen_deltanet(
            &mut logits,
            weights,
            &mut cache,
            token,
            pos,
            &census_config,
            &mut scratch,
            rope_freq,
        );
    }
    finish_census(
        &mut scratch,
        &cache,
        &census_config,
        n_positions,
        tokens_used,
    )
}

/// The TERNARY census (Issue 981 follow-up (b)): the same pass over the
/// flagship's serving body — `QwenDeltaNetTernaryWeights` +
/// [`forward_qwen_deltanet_ternary`]. The dense loader mis-shapes the PQ2_0
/// ternary file (in_proj_qkv row overrun), so the Bonsai-2-27B census MUST
/// run here. The capture seam, the harvest, and the profile arithmetic are
/// the shared code below — the ternary body writes f32 post-RoPE Q/K/V into
/// the same buffers the dense body does (the Issue-594 gate applies at the
/// output, after the attention read, exactly as the audit premise requires).
///
/// A folded (Hadamard) file needs the `bonsai2_hadamard` feature at the
/// RUN's build (the loader refuses loud otherwise); this lane itself is
/// rotation-agnostic.
pub fn run_census_ternary(
    weights: &QwenDeltaNetTernaryWeights,
    config: &Config,
    tokens: &[usize],
    rope_freq: &RopeFreqTable,
    n_positions: usize,
) -> Result<SinkCensus, String> {
    if tokens.len() < n_positions {
        return Err(format!(
            "corpus encodes {} tokens < the census window {n_positions}",
            tokens.len()
        ));
    }
    let census_config = census_posture(config, n_positions);
    let mut cache =
        HybridCache::with_layer_types_bounded(&census_config, &weights.layer_types, n_positions);
    let mut scratch = HybridForwardScratch::new(&census_config);
    arm_capture(&mut scratch, config, &weights.layer_types, n_positions);

    let mut logits = vec![0.0f32; config.vocab_size];
    let tokens_used = n_positions;
    for (pos, &token) in tokens.iter().take(n_positions).enumerate() {
        set_capture_pos(&mut scratch, pos);
        forward_qwen_deltanet_ternary(
            &mut logits,
            weights,
            &mut cache,
            token,
            pos,
            &census_config,
            &mut scratch,
            rope_freq,
        );
    }
    finish_census(
        &mut scratch,
        &cache,
        &census_config,
        n_positions,
        tokens_used,
    )
}

/// The census posture shared by both bodies: the cache is sized to exactly
/// the census window (the Issue-929 lesson: never the model's 128k-class
/// block_size — Bonsai-2 would demand >100 GiB).
fn census_posture(config: &Config, n_positions: usize) -> Config {
    let mut census_config = config.clone();
    census_config.block_size = n_positions;
    census_config
}

/// Arm the Q capture on the scratch (the census owns the only armed copy).
fn arm_capture(
    scratch: &mut HybridForwardScratch,
    config: &Config,
    layer_types: &[DeltaNetLayerType],
    n_positions: usize,
) {
    scratch.sink_audit_q = Some(SinkAuditQCapture::new(config, layer_types, n_positions));
}

fn set_capture_pos(scratch: &mut HybridForwardScratch, pos: usize) {
    if let Some(cap) = scratch.sink_audit_q.as_mut() {
        cap.pos = pos;
    }
}

/// Take the capture back out, run the shared harvest, and assemble the
/// [`SinkCensus`].
fn finish_census(
    scratch: &mut HybridForwardScratch,
    cache: &HybridCache,
    config: &Config,
    n_positions: usize,
    tokens_used: usize,
) -> Result<SinkCensus, String> {
    let capture = scratch
        .sink_audit_q
        .take()
        .ok_or_else(|| "census capture vanished".to_string())?;
    let profile = harvest_profile(cache, &capture, config, n_positions);
    let kvd = crate::types::kv_dim(config);
    let kv = SinkAuditKVCapture::extract(
        &capture.attn_layers,
        |slot| {
            let layer = &cache.kv_cache.layers[capture.attn_layers[slot]];
            (layer.key.as_slice(), layer.value.as_slice())
        },
        n_positions,
        kvd,
        config.n_kv_head,
        config.head_dim,
    );
    Ok(SinkCensus {
        profile,
        n_positions,
        tokens_used,
        kv,
        // Issue 1185 T2: the Q rows ride out too — the width census scores
        // them against the captured K/V. Everything the offline lanes need
        // is now on one census return.
        q: Some(capture),
    })
}

/// The shared harvest — identical for the dense and ternary bodies (both
/// write f32 post-RoPE K/V into the same `KVCache` layout). Sequential f32
/// rows accumulating into f64, one fixed order (layers ascending, heads
/// ascending, queries ascending); the audit core owns the arithmetic.
fn harvest_profile(
    cache: &HybridCache,
    capture: &SinkAuditQCapture,
    config: &Config,
    n_positions: usize,
) -> SinkAuditProfile {
    let q_dim = config.n_head * config.head_dim;
    let kvd = crate::types::kv_dim(config);
    let n_head = config.n_head;
    let n_kv = config.n_kv_head;
    let head_dim = config.head_dim;
    let kv_per_head = n_head / n_kv;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // ── Harvest: (layers ascending, heads ascending, queries ascending) ──
    let mut acc = vec![0.0f64; n_positions];
    let mut row: Vec<f32> = Vec::new();
    for (slot, &layer_idx) in capture.attn_layers.iter().enumerate() {
        let layer_cache = &cache.kv_cache.layers[layer_idx];
        let q_rows = &capture.q_rows[slot];
        for qi in 0..n_positions {
            for h in 0..n_head {
                let kvh = h / kv_per_head;
                let q_row = &q_rows[qi * q_dim..(qi + 1) * q_dim];
                let q_head = &q_row[h * head_dim..(h + 1) * head_dim];
                // This head's K rows for positions 0..=qi, strided by kvd.
                let k_base = layer_cache.key.as_slice();
                let mut head_keys = Vec::with_capacity((qi + 1) * head_dim);
                for ki in 0..=qi {
                    let off = ki * kvd + kvh * head_dim;
                    head_keys.extend_from_slice(&k_base[off..off + head_dim]);
                }
                accumulate_head_mass_into(
                    q_head,
                    &head_keys,
                    qi + 1,
                    head_dim,
                    scale,
                    &mut row,
                    &mut acc,
                );
            }
        }
    }
    let first_n_profile = finalize_profile(&acc, capture.attn_layers.len(), n_head, n_positions);

    // ── M-Act: mean per-layer max V-row L2 norm (disclosed, not verdict-bearing)
    let m_act = mean_layer_max_v_norm(
        capture
            .attn_layers
            .iter()
            .map(|&l| cache.kv_cache.layers[l].value.as_slice()),
        n_positions,
        kvd,
    );

    SinkAuditProfile {
        f_attn_token0: first_n_profile[0],
        first_n_profile,
        m_act,
        layers_scanned: capture.attn_layers.len(),
        heads_scanned: n_head,
        positions_scanned: n_positions,
    }
}

/// M-Act: the mean over layers of each layer's max V-row L2 norm across the
/// first `n_positions` rows (f64, sequential, layers in the given order). The
/// paper's secondary signal — disclosed, never verdict-bearing. Shared by the
/// hybrid census above and the dense-transformer census
/// (`transformer::sink_audit_dense`).
pub(crate) fn mean_layer_max_v_norm<'a>(
    layers: impl Iterator<Item = &'a [f32]>,
    n_positions: usize,
    kvd: usize,
) -> f64 {
    let (mut sum, mut count) = (0.0f64, 0usize);
    for v in layers {
        let mut max_sq = 0.0f64;
        for row_v in v[..n_positions * kvd].chunks_exact(kvd) {
            let sq: f64 = row_v.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
            if sq > max_sq {
                max_sq = sq;
            }
        }
        sum += max_sq.sqrt();
        count += 1;
    }
    match count {
        0 => 0.0,
        c => sum / c as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deltanet::weights::QwenDeltaNetWeights;
    use crate::types::DeltaNetLayerType;
    use katgpt_core::kv_sink_window::{SinkAuditVerdict, SinkWindowPolicy};

    /// A tiny hybrid config (4 layers: GDN, ATTN, GDN, ATTN) with dims small
    /// enough to run on a test box.
    fn tiny_config() -> Config {
        Config {
            n_layer: 4,
            n_embd: 64,
            n_head: 4,
            n_kv_head: 2,
            head_dim: 16,
            mlp_hidden: 128,
            vocab_size: 128,
            block_size: 32,
            layer_types: vec![
                DeltaNetLayerType::DeltaNet,
                DeltaNetLayerType::Attention,
                DeltaNetLayerType::DeltaNet,
                DeltaNetLayerType::Attention,
            ],
            deltanet_linear_head_dim: 8,
            deltanet_linear_n_heads: 2,
            deltanet_linear_n_value_heads: 2,
            deltanet_conv_kernel_size: 4,
            ..Config::default()
        }
    }

    /// ALL-ZERO weights ⇒ every logit 0 ⇒ every softmax row uniform ⇒ the
    /// analytic profile: position k's mean mass is
    /// `(1/(N-k)) * Σ_{qi=k..N-1} 1/(qi+1)`, and m_act = 0 (zero V rows).
    #[test]
    fn census_on_zero_weights_matches_the_analytic_uniform_profile() {
        let config = tiny_config();
        let weights = QwenDeltaNetWeights::zeros(&config);
        let rope = RopeFreqTable::new(
            config.rope_theta,
            crate::deltanet::effective_rotary_dim(&config),
        );
        let n = 8usize;
        let tokens: Vec<usize> = (0..n).collect();

        let census = run_census(&weights, &config, &tokens, &rope, n).expect("census runs");
        assert_eq!(census.n_positions, n);
        assert_eq!(census.profile.positions_scanned, n);
        // Scope: only the two ATTENTION layers, 4 heads.
        assert_eq!(census.profile.layers_scanned, 2);
        assert_eq!(census.profile.heads_scanned, 4);

        for (k, &mass) in census.profile.first_n_profile.iter().enumerate() {
            let want: f64 = (k..n).map(|qi| 1.0 / (qi + 1) as f64).sum::<f64>() / (n - k) as f64;
            assert!(
                (mass - want).abs() < 1e-6,
                "profile[{k}] = {mass} want {want}"
            );
        }
        assert!((census.profile.f_attn_token0 - census.profile.first_n_profile[0]).abs() < 1e-15);
        assert_eq!(census.profile.m_act, 0.0, "zero V rows carry zero norms");

        // A TINY window reads SinkHeavy by construction: uniform softmax
        // concentrates on early positions because every query sees few of
        // them — profile[0] = H(N)/N ≈ 0.34 at N=8, above the heavy cut. The
        // classifier is correct; the census window is what scales this away
        // (H(512)/512 ≈ 0.012 → SinkFree on a real window).
        assert_eq!(census.profile.verdict(), SinkAuditVerdict::SinkHeavy);
        let incumbent = SinkWindowPolicy::new(4, 256);
        assert_eq!(
            SinkWindowPolicy::from_audit(SinkAuditVerdict::SinkHeavy, &census.profile, incumbent),
            katgpt_core::kv_sink_window::AuditedSinkPolicy::KeepIncumbentSinkHeavy,
            "heavy keeps the incumbent pin"
        );
    }

    /// G1 at the lane level: the same census twice is bit-identical.
    #[test]
    fn census_is_deterministic() {
        let config = tiny_config();
        let weights = QwenDeltaNetWeights::zeros(&config);
        let rope = RopeFreqTable::new(
            config.rope_theta,
            crate::deltanet::effective_rotary_dim(&config),
        );
        let n = 8usize;
        let tokens: Vec<usize> = (0..n).map(|i| (i * 7 + 3) % config.vocab_size).collect();

        let a = run_census(&weights, &config, &tokens, &rope, n).expect("census runs");
        let b = run_census(&weights, &config, &tokens, &rope, n).expect("census runs");
        assert_eq!(a.profile.first_n_profile, b.profile.first_n_profile);
        assert_eq!(a.profile.f_attn_token0, b.profile.f_attn_token0);
        assert_eq!(a.profile.m_act, b.profile.m_act);
    }

    /// A corpus shorter than the window refuses loud, never silently scans.
    #[test]
    fn short_corpus_refuses() {
        let config = tiny_config();
        let weights = QwenDeltaNetWeights::zeros(&config);
        let rope = RopeFreqTable::new(
            config.rope_theta,
            crate::deltanet::effective_rotary_dim(&config),
        );
        let err = run_census(&weights, &config, &[1, 2, 3], &rope, 8).unwrap_err();
        assert!(err.contains("census window"), "{err}");
    }

    /// The TERNARY census on all-zero ternary weights meets the SAME analytic
    /// uniform profile as the dense census — the shared-harvest identity. The
    /// config is all-attention (GDN ternary projections demand 128-multiple
    /// group dims the tiny dense config cannot satisfy) with 128-multiple
    /// ternary dims (`QwenDeltaNetTernaryWeights::zeros`' contract).
    #[cfg(feature = "deltanet_ternary_inference")]
    fn tiny_ternary_config() -> Config {
        Config {
            n_layer: 4,
            n_embd: 128,
            n_head: 2,
            n_kv_head: 1,
            head_dim: 64,
            mlp_hidden: 128,
            vocab_size: 128,
            block_size: 32,
            layer_types: vec![DeltaNetLayerType::Attention; 4],
            ..Config::default()
        }
    }

    #[cfg(feature = "deltanet_ternary_inference")]
    #[test]
    fn ternary_census_on_zero_weights_matches_the_analytic_uniform_profile() {
        let config = tiny_ternary_config();
        let weights = QwenDeltaNetTernaryWeights::zeros(&config);
        let rope = RopeFreqTable::new(
            config.rope_theta,
            crate::deltanet::effective_rotary_dim(&config),
        );
        let n = 8usize;
        let tokens: Vec<usize> = (0..n).collect();

        let census = run_census_ternary(&weights, &config, &tokens, &rope, n).expect("census runs");
        assert_eq!(census.n_positions, n);
        assert_eq!(census.profile.positions_scanned, n);
        // Scope: every layer is full-attention here — all four scanned.
        assert_eq!(census.profile.layers_scanned, 4);
        assert_eq!(census.profile.heads_scanned, 2);

        for (k, &mass) in census.profile.first_n_profile.iter().enumerate() {
            let want: f64 = (k..n).map(|qi| 1.0 / (qi + 1) as f64).sum::<f64>() / (n - k) as f64;
            assert!(
                (mass - want).abs() < 1e-6,
                "profile[{k}] = {mass} want {want}"
            );
        }
        assert!((census.profile.f_attn_token0 - census.profile.first_n_profile[0]).abs() < 1e-15);
        assert_eq!(census.profile.m_act, 0.0, "zero V rows carry zero norms");
        assert_eq!(census.profile.verdict(), SinkAuditVerdict::SinkHeavy);
    }

    /// G1 at the ternary lane level: the same census twice is bit-identical.
    #[cfg(feature = "deltanet_ternary_inference")]
    #[test]
    fn ternary_census_is_deterministic() {
        let config = tiny_ternary_config();
        let weights = QwenDeltaNetTernaryWeights::zeros(&config);
        let rope = RopeFreqTable::new(
            config.rope_theta,
            crate::deltanet::effective_rotary_dim(&config),
        );
        let n = 8usize;
        let tokens: Vec<usize> = (0..n).map(|i| (i * 7 + 3) % config.vocab_size).collect();

        let a = run_census_ternary(&weights, &config, &tokens, &rope, n).expect("census runs");
        let b = run_census_ternary(&weights, &config, &tokens, &rope, n).expect("census runs");
        assert_eq!(a.profile.first_n_profile, b.profile.first_n_profile);
        assert_eq!(a.profile.f_attn_token0, b.profile.f_attn_token0);
        assert_eq!(a.profile.m_act, b.profile.m_act);
    }

    // ── Issue 1185 T1: the K/V capture fixtures ─────────────────────────

    /// The capture is the harvest's own input: re-deriving the m_act signal
    /// from the captured V rows must reproduce the profile bit-exactly (same
    /// bytes, same sequential f64 folds, same layer order). This is the pin
    /// that the extraction-from-cache design carries the FORWARD-time values,
    /// not post-hoc noise.
    #[test]
    fn kv_capture_reproduces_the_m_act_signal_bit_exactly() {
        let config = tiny_config();
        let weights = QwenDeltaNetWeights::zeros(&config);
        let rope = RopeFreqTable::new(
            config.rope_theta,
            crate::deltanet::effective_rotary_dim(&config),
        );
        let n = 8usize;
        let tokens: Vec<usize> = (0..n).collect();

        let census = run_census(&weights, &config, &tokens, &rope, n).expect("census runs");
        let kvd = crate::types::kv_dim(&config);
        let m_from_capture =
            mean_layer_max_v_norm(census.kv.value_rows.iter().map(|v| v.as_slice()), n, kvd);
        assert_eq!(m_from_capture, census.profile.m_act);
    }

    /// Slot scope: the capture covers exactly the full-attention layers, in
    /// ascending order, shaped [N × kvd] per slot (zero-padding identity is
    /// covered by the m_act pin above — zero weights ⇒ zero V rows).
    #[test]
    fn kv_capture_covers_only_full_attention_layers() {
        let config = tiny_config();
        let weights = QwenDeltaNetWeights::zeros(&config);
        let rope = RopeFreqTable::new(
            config.rope_theta,
            crate::deltanet::effective_rotary_dim(&config),
        );
        let n = 8usize;
        let tokens: Vec<usize> = (0..n).collect();

        let census = run_census(&weights, &config, &tokens, &rope, n).expect("census runs");
        let kvd = crate::types::kv_dim(&config);
        assert_eq!(census.kv.attn_layers, vec![1, 3], "only the ATTN layers");
        assert_eq!(census.kv.key_rows.len(), 2);
        assert_eq!(census.kv.value_rows.len(), 2);
        for (key, value) in census.kv.key_rows.iter().zip(&census.kv.value_rows) {
            assert_eq!(key.len(), n * kvd);
            assert_eq!(value.len(), n * kvd);
        }
        assert_eq!(census.kv.n_positions, n);
        assert_eq!(census.kv.kv_dim, kvd);
        assert_eq!(census.kv.n_kv_head, config.n_kv_head);
        assert_eq!(census.kv.head_dim, config.head_dim);
    }

    /// G1 on the capture itself: the same census twice yields byte-identical
    /// K/V rows (no RNG, no wall clock — the extraction is pure copies).
    #[test]
    fn kv_capture_is_deterministic() {
        let config = tiny_config();
        let weights = QwenDeltaNetWeights::zeros(&config);
        let rope = RopeFreqTable::new(
            config.rope_theta,
            crate::deltanet::effective_rotary_dim(&config),
        );
        let n = 8usize;
        let tokens: Vec<usize> = (0..n).map(|i| (i * 7 + 3) % config.vocab_size).collect();

        let a = run_census(&weights, &config, &tokens, &rope, n).expect("census runs");
        let b = run_census(&weights, &config, &tokens, &rope, n).expect("census runs");
        assert_eq!(a.kv.key_rows, b.kv.key_rows);
        assert_eq!(a.kv.value_rows, b.kv.value_rows);
        assert_eq!(a.kv.attn_layers, b.kv.attn_layers);
    }

    /// The per-KV-head gather (the T2 width-census lane's input shape):
    /// gathering head (slot, kvh) must equal the manual strided read of the
    /// captured rows — contiguous [N × head_dim], heads in cache order.
    #[test]
    fn gather_kv_head_matches_the_strided_read() {
        let config = tiny_config();
        let weights = QwenDeltaNetWeights::zeros(&config);
        let rope = RopeFreqTable::new(
            config.rope_theta,
            crate::deltanet::effective_rotary_dim(&config),
        );
        let n = 8usize;
        let tokens: Vec<usize> = (0..n).collect();

        let census = run_census(&weights, &config, &tokens, &rope, n).expect("census runs");
        let mut gathered = Vec::new();
        census.kv.gather_kv_head(0, 1, &mut gathered);
        assert_eq!(gathered.len(), n * config.head_dim);
        let rows = &census.kv.key_rows[0];
        for (pos, chunk) in gathered.chunks_exact(config.head_dim).enumerate() {
            let off = pos * census.kv.kv_dim + config.head_dim;
            assert_eq!(chunk, &rows[off..off + config.head_dim], "pos {pos}");
        }
    }
}
