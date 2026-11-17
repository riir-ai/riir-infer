//! Loopability predictor — dense-GGUF arm (i1221 T3(d), plan 636 read-order leg 1).
//!
//! The Research-672 §3 instrument (the No-GD advocate's variant): chunk a
//! DENSE checkpoint's middle layer range, iterate the loop recurrence over
//! it, and read whether the iteration CONTRACTS (`λ_t`) and what the readout
//! costs at depth k (`KL(full ‖ loop-k)`). Report-only — the only asserts in
//! this lane are the composition/identity fixtures below; the emitted rows
//! are measurements, never gates.
//!
//! # Recurrence (the formula verbatim)
//!
//! ```text
//! s_0 = x                        // raw input-half boundary state (the anchor)
//! u_t = N(s_{t-1} + x)           // anchor re-injected every loop
//! s_t = N(F_chunk(u_t))          // F_chunk = the chunk layers, run verbatim
//! ```
//!
//! `N` is the UNWEIGHTED RMS normalize (`v / √(mean(v²) + ε)`, ε =
//! [`LOOPABILITY_NORM_EPSILON`], sequential f32 square-sum fold, division
//! form) — the dense checkpoint has no learned loop-boundary norm, so the
//! neutral scale projection is the instrument's own definition (the dory
//! forward's `rms_denom` form, mirrored value; `deltanet` does not depend on
//! the `dory` family). Δs₁'s baseline is the RAW anchor because `s_0 = x` is
//! raw by the formula; Δs_t for t ≥ 2 compares two N-normalized states.
//!
//! # Hybrid-stack loop semantics
//!
//! Loop passes are FULL-PROMPT passes over the chunk (the Dory prefill
//! shape, and the same semantics the Issue-929 `loop_probe` documents for
//! the ternary body): each pass advances the chunk's GDN recurrent+conv
//! states once and overwrites the chunk attention layers' KV rows
//! `[0..seq_len]` before scoring — cross-position mixing happens within a
//! pass, per-position loop state carries across passes through `s_t`. The
//! chunk MUST span the stack's middle (`mid_start >= 1`, `mid_start +
//! chunk_len < n_layer`) so a nonempty input half feeds the anchor and a
//! nonempty output half reads the final state.
//!
//! # Readouts per k
//!
//! - `λ_t = ‖Δs_t‖ / ‖Δs_{t-1}‖` for t = 2..=k — Frobenius-pooled over all
//!   positions AND the last position alone (the decode-relevant row). An
//!   exact-zero denominator reads as full contraction (λ := 0), never NaN.
//! - `KL(full ‖ loop-k)` at the last position (f64 log-softmax, sequential
//!   folds) — plus `KL(loop-1 ‖ loop-k)` (does looping beyond 1 change
//!   anything at all).
//! - argmax + top1−top2 margins both sides, the flip flag.
//!
//! Box/state lineage, JSONL rows, and resume live in the runner example
//! (`examples/i1221_t3d_loopability.rs`, plan 636 resumable format); this
//! module is the pure instrument.

/// The loop-boundary normalize ε — the dory forward's `NORM_EPSILON` value
/// mirrored (both instruments normalize with the same form; a shared const
/// would couple `deltanet` to the `dory` feature for one number).
pub const LOOPABILITY_NORM_EPSILON: f32 = 1e-6;

use super::forward::{
    HybridCache, HybridForwardScratch, PrefillContext, prefill_qwen_deltanet_into,
    prefill_qwen_deltanet_range_into,
};
use super::weights::QwenDeltaNetWeights;
use crate::rope::RopeFreqTable;
use crate::types::{Config, rmsnorm_with_gamma_eps};

/// The chunk coordinates: the looped middle range is
/// `[mid_start, mid_start + chunk_len)`.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct LoopabilityChunk {
    pub mid_start: usize,
    pub chunk_len: usize,
}

impl LoopabilityChunk {
    /// Validate the chunk against the stack (loud, not silent clamps): the
    /// input half and the output half must both be nonempty.
    pub fn validated(self, n_layers: usize) -> Self {
        assert!(
            self.mid_start >= 1,
            "mid_start must leave a nonempty input half"
        );
        assert!(
            self.mid_start + self.chunk_len < n_layers,
            "chunk must leave a nonempty output half: {}+{} vs {n_layers} layers",
            self.mid_start,
            self.chunk_len
        );
        self
    }
}

/// The full-depth reference readout (the unlooped model's own answer).
pub struct LoopReference {
    /// Last-position logits from the incumbent full-stack prefill.
    pub logits: Vec<f32>,
    pub argmax: usize,
    /// top1 − top2 logit margin.
    pub margin: f32,
}

/// One loop depth's measurement row.
#[derive(Clone, Debug, serde::Serialize)]
pub struct LoopKRow {
    pub k: usize,
    /// `λ_t` t = 2..=k, Frobenius-pooled over all positions (`k − 1` values).
    pub lambdas_pooled: Vec<f64>,
    /// `λ_t` measured on the last position alone.
    pub lambdas_last: Vec<f64>,
    /// Mean of `lambdas_pooled` (`None` at k = 1).
    pub lambda_mean_pooled: Option<f64>,
    /// `KL(full ‖ loop-k)` at the last position.
    pub kl_full_vs_loop: f64,
    /// `KL(loop-1 ‖ loop-k)` — `None` at k = 1 (the baseline itself).
    pub kl_loop1_vs_loopk: Option<f64>,
    pub argmax_full: usize,
    pub argmax_loop: usize,
    pub argmax_flip: bool,
    pub margin_full: f32,
    pub margin_loop: f32,
}

/// Embed the prompt tokens into a fresh [`PrefillContext`]'s hidden rows —
/// the manual half of the incumbent prefill's step 1 (the range entry takes
/// pre-loaded rows; see `prefill_qwen_deltanet_range_into`).
pub fn embed_prompt(
    weights: &QwenDeltaNetWeights,
    config: &Config,
    ctx: &mut PrefillContext,
    prompt_tokens: &[usize],
) {
    let n = config.n_embd;
    assert!(
        !prompt_tokens.is_empty(),
        "prompt requires at least one token"
    );
    let hidden = ctx.hidden_rows_mut();
    for (p, &tok) in prompt_tokens.iter().enumerate() {
        assert!(
            tok < config.vocab_size,
            "token {tok} out of vocab ({})",
            config.vocab_size
        );
        let off = tok * n;
        hidden[p * n..(p + 1) * n].copy_from_slice(&weights.wte[off..off + n]);
    }
}

/// The unweighted RMS normalize — `v / √(mean(v²) + ε)`, sequential f32
/// square-sum fold, division form (the dory `rms_denom` shape; see the
/// module doc for why the const is local).
fn rms_unweighted(row: &mut [f32]) {
    let mut acc = 0.0f32;
    for &x in row.iter() {
        acc += x * x;
    }
    let denom = (acc / row.len() as f32 + LOOPABILITY_NORM_EPSILON).sqrt();
    for x in row.iter_mut() {
        *x /= denom;
    }
}

/// f64 log-softmax with a max-subtract guard and sequential fold (report
/// space; the logits themselves stay f32).
fn log_softmax_f64(logits: &[f32]) -> Vec<f64> {
    let mut max = f64::NEG_INFINITY;
    for &l in logits {
        let l = l as f64;
        if l > max {
            max = l;
        }
    }
    let mut acc = 0.0f64;
    for &l in logits {
        acc += (l as f64 - max).exp();
    }
    let lse = max + acc.ln();
    logits.iter().map(|&l| l as f64 - lse).collect()
}

/// `KL(P ‖ Q)` over two logit vectors at the same position. Terms where the
/// target probability underflowed to zero are skipped; a term where the
/// APPROXIMATION assigns zero mass to target mass diverges to +inf — that is
/// a legitimate readout, never clamped.
fn kl_divergence(p_logits: &[f32], q_logits: &[f32]) -> f64 {
    debug_assert_eq!(p_logits.len(), q_logits.len());
    let lp = log_softmax_f64(p_logits);
    let lq = log_softmax_f64(q_logits);
    let mut kl = 0.0f64;
    for i in 0..lp.len() {
        let p = lp[i].exp();
        if p > 0.0 {
            kl += p * (lp[i] - lq[i]);
        }
    }
    kl
}

/// Strict-`>` argmax → lowest index on ties (the house rule), with the
/// top1 − top2 margin. (Local: the crate keeps small local argmax helpers —
/// `loop_probe::argmax_of` lives behind a different feature and the twt bins
/// carry their own; dedup across feature gates is out of this lane's scope.)
fn argmax_margin(logits: &[f32]) -> (usize, f32) {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    let mut second = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            second = best_v;
            best_v = v;
            best = i;
        } else if v > second {
            second = v;
        }
    }
    (best, best_v - second)
}

/// The full-depth reference: the incumbent prefill, last-position logits +
/// argmax/margin. Own cache/context; `scratch` is shared (fully rewritten
/// per layer, safe across calls).
pub fn full_depth_reference(
    weights: &QwenDeltaNetWeights,
    config: &Config,
    prompt_tokens: &[usize],
    scratch: &mut HybridForwardScratch,
    rope_freq: &RopeFreqTable,
) -> LoopReference {
    let seq = prompt_tokens.len();
    let mut cache = HybridCache::with_layer_types_bounded(config, &weights.layer_types, seq);
    let mut ctx = PrefillContext::new(config, seq);
    let mut logits = vec![0.0f32; config.vocab_size];
    prefill_qwen_deltanet_into(
        weights,
        config,
        &mut cache,
        prompt_tokens,
        scratch,
        rope_freq,
        &mut ctx,
        &mut logits,
        None,
    );
    let (argmax, margin) = argmax_margin(&logits);
    LoopReference {
        logits,
        argmax,
        margin,
    }
}

/// The loop variant at depth k: the recurrence, the output half from `s_k`,
/// the head — and the Δs norms alongside (they live in the same pass).
/// Callers pass k ≥ 1 — k = 0 would skip the chunk entirely and is NOT the
/// unlooped model (that is [`full_depth_reference`]).
#[allow(clippy::too_many_arguments)]
fn loop_variant_state(
    weights: &QwenDeltaNetWeights,
    config: &Config,
    prompt_tokens: &[usize],
    chunk: LoopabilityChunk,
    k: usize,
    scratch: &mut HybridForwardScratch,
    rope_freq: &RopeFreqTable,
) -> (Vec<f32>, Vec<f64>, Vec<f64>) {
    chunk.validated(weights.layers.len());
    let n = config.n_embd;
    let seq = prompt_tokens.len();
    let mid = chunk.mid_start;
    let chunk_end = mid + chunk.chunk_len;
    let n_layers = weights.layers.len();

    let mut cache = HybridCache::with_layer_types_bounded(config, &weights.layer_types, seq);
    let mut ctx = PrefillContext::new(config, seq);
    embed_prompt(weights, config, &mut ctx, prompt_tokens);
    prefill_qwen_deltanet_range_into(
        weights,
        config,
        &mut cache,
        scratch,
        rope_freq,
        &mut ctx,
        seq,
        0..mid,
    );
    // anchor = raw input-half boundary; s_0 = anchor (the formula's x)
    let mut anchor = vec![0.0f32; seq * n];
    anchor.copy_from_slice(&ctx.hidden_rows()[..seq * n]);
    let mut s_prev = anchor.clone();

    let mut deltas_pool = Vec::with_capacity(k);
    let mut deltas_last = Vec::with_capacity(k);
    for _t in 0..k {
        // u_t = N(s_{t-1} + x), written straight into the boundary rows
        {
            let hidden = ctx.hidden_rows_mut();
            for p in 0..seq {
                let (s, a, row) = (
                    &s_prev[p * n..(p + 1) * n],
                    &anchor[p * n..(p + 1) * n],
                    &mut hidden[p * n..(p + 1) * n],
                );
                for (r, (&sv, &av)) in row.iter_mut().zip(s.iter().zip(a.iter())) {
                    *r = sv + av;
                }
                rms_unweighted(row);
            }
        }
        // F_chunk over the whole prompt (the pass advances GDN states and
        // overwrites the chunk attention KV rows)
        prefill_qwen_deltanet_range_into(
            weights,
            config,
            &mut cache,
            scratch,
            rope_freq,
            &mut ctx,
            seq,
            mid..chunk_end,
        );
        // s_t = N(raw chunk output); Δs_t vs s_prev
        let mut ds_pool = 0.0f64;
        let mut ds_last = 0.0f64;
        {
            let hidden = ctx.hidden_rows_mut();
            for p in 0..seq {
                let row = &mut hidden[p * n..(p + 1) * n];
                rms_unweighted(row);
                let prev = &s_prev[p * n..(p + 1) * n];
                let mut sq = 0.0f64;
                for (r, &s) in row.iter().zip(prev.iter()) {
                    let d = r - s;
                    sq += (d * d) as f64;
                }
                ds_pool += sq;
                if p + 1 == seq {
                    ds_last = sq;
                }
            }
        }
        deltas_pool.push(ds_pool);
        deltas_last.push(ds_last);
        s_prev.copy_from_slice(&ctx.hidden_rows()[..seq * n]);
    }

    // output half from s_k (already normalized, already in the rows)
    prefill_qwen_deltanet_range_into(
        weights,
        config,
        &mut cache,
        scratch,
        rope_freq,
        &mut ctx,
        seq,
        chunk_end..n_layers,
    );
    // final norm + head on the last position (the incumbent's steps 3-4)
    let last_off = (seq - 1) * n;
    let mut logits = vec![0.0f32; config.vocab_size];
    {
        let hidden = ctx.hidden_rows_mut();
        rmsnorm_with_gamma_eps(
            &mut hidden[last_off..last_off + n],
            &weights.final_norm,
            config.rms_norm_eps,
        );
        weights
            .lm_head
            .matvec(&hidden[last_off..last_off + n], &mut logits);
    }
    (logits, deltas_pool, deltas_last)
}

/// `λ_t` ladder from a Δ² run: `sqrt(Δ²_t / Δ²_{t-1})` for t = 2..=k. An
/// exact-zero denominator reads as full contraction (0.0), never NaN.
fn lambdas_of(deltas_sq: &[f64]) -> Vec<f64> {
    (1..deltas_sq.len())
        .map(|i| {
            let (num, den) = (deltas_sq[i], deltas_sq[i - 1]);
            if den > 0.0 { (num / den).sqrt() } else { 0.0 }
        })
        .collect()
}

/// One loop depth's full row. The k = 1 baseline (for `kl_loop1_vs_loopk`)
/// is computed internally per call — an offline instrument, the extra pass
/// is noise next to the model load, and it keeps every row self-contained
/// (resume-safe: rows never depend on a prior call's state).
#[allow(clippy::too_many_arguments)]
pub fn loop_variant_row(
    weights: &QwenDeltaNetWeights,
    config: &Config,
    prompt_tokens: &[usize],
    chunk: LoopabilityChunk,
    k: usize,
    full: &LoopReference,
    scratch: &mut HybridForwardScratch,
    rope_freq: &RopeFreqTable,
) -> LoopKRow {
    assert!(k >= 1, "loop depth k must be >= 1");
    let (logits_loop1, _, _) =
        loop_variant_state(weights, config, prompt_tokens, chunk, 1, scratch, rope_freq);
    let (logits_k, deltas_pool, deltas_last) =
        loop_variant_state(weights, config, prompt_tokens, chunk, k, scratch, rope_freq);

    let lambdas_pooled = lambdas_of(&deltas_pool);
    let lambdas_last = lambdas_of(&deltas_last);
    let lambda_mean_pooled = if lambdas_pooled.is_empty() {
        None
    } else {
        Some(lambdas_pooled.iter().sum::<f64>() / lambdas_pooled.len() as f64)
    };
    let (argmax_loop, margin_loop) = argmax_margin(&logits_k);

    LoopKRow {
        k,
        lambda_mean_pooled,
        lambdas_pooled,
        lambdas_last,
        kl_full_vs_loop: kl_divergence(&full.logits, &logits_k),
        kl_loop1_vs_loopk: if k == 1 {
            None
        } else {
            Some(kl_divergence(&logits_loop1, &logits_k))
        },
        argmax_full: full.argmax,
        argmax_loop,
        argmax_flip: argmax_loop != full.argmax,
        margin_full: full.margin,
        margin_loop,
    }
}

/// Convenience: the reference once, then one row per requested depth (rows
/// in `ks` order, duplicates allowed). The runner example calls
/// [`loop_variant_row`] per k for durable per-row writes; this wrapper is
/// the lib/test surface.
pub fn run_loopability(
    weights: &QwenDeltaNetWeights,
    config: &Config,
    prompt_tokens: &[usize],
    chunk: LoopabilityChunk,
    ks: &[usize],
    scratch: &mut HybridForwardScratch,
    rope_freq: &RopeFreqTable,
) -> Vec<LoopKRow> {
    assert!(!ks.is_empty(), "loopability needs at least one depth");
    let full = full_depth_reference(weights, config, prompt_tokens, scratch, rope_freq);
    ks.iter()
        .map(|&k| {
            loop_variant_row(
                weights,
                config,
                prompt_tokens,
                chunk,
                k,
                &full,
                scratch,
                rope_freq,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deltanet::forward::effective_rotary_dim;

    /// A 6-layer hybrid in the qwen3.5-0.8B cadence (3 DeltaNet + 1 Attention),
    /// shrunk surfaces (the [`crate::deltanet::unit_test_config`] law), random
    /// weights — zero weights make every forward degenerate (all-zero logits
    /// through the zero gammas) and would pass any loop pin vacuously.
    fn small_model(seed: u64) -> (Config, QwenDeltaNetWeights) {
        use crate::types::DeltaNetLayerType::*;
        let layer_types = vec![DeltaNet, DeltaNet, DeltaNet, Attention, DeltaNet, Attention];
        let config = crate::deltanet::unit_test_config(6, layer_types);
        let mut weights = QwenDeltaNetWeights::zeros(&config);
        crate::deltanet::fill_random_weights(&mut weights, seed);
        (config, weights)
    }

    fn rope(config: &Config) -> RopeFreqTable {
        RopeFreqTable::new(config.rope_theta, effective_rotary_dim(config))
    }

    const CHUNK: LoopabilityChunk = LoopabilityChunk {
        mid_start: 2,
        chunk_len: 2,
    };

    #[test]
    fn loop_rows_are_wellformed() {
        let (config, weights) = small_model(11);
        let rope_freq = rope(&config);
        let mut scratch = HybridForwardScratch::new(&config);
        let prompt = vec![config.bos_token, 100, 200, 300];

        let rows = run_loopability(
            &weights,
            &config,
            &prompt,
            CHUNK,
            &[1, 2, 4],
            &mut scratch,
            &rope_freq,
        );
        assert_eq!(rows.len(), 3);
        for (i, row) in rows.iter().enumerate() {
            let k = [1, 2, 4][i];
            assert_eq!(row.k, k);
            assert_eq!(row.lambdas_pooled.len(), k - 1, "pooled λ ladder at k={k}");
            assert_eq!(
                row.lambdas_last.len(),
                k - 1,
                "last-position λ ladder at k={k}"
            );
            assert_eq!(row.lambda_mean_pooled.is_some(), k > 1);
            if let (Some(mean), true) = (row.lambda_mean_pooled, k > 1) {
                let expect =
                    row.lambdas_pooled.iter().sum::<f64>() / row.lambdas_pooled.len() as f64;
                assert!((mean - expect).abs() < 1e-12, "λ mean is the pooled mean");
            }
            assert_eq!(row.kl_loop1_vs_loopk.is_some(), k > 1);
            assert!(row.kl_full_vs_loop >= 0.0, "KL is non-negative at k={k}");
            if let Some(kl) = row.kl_loop1_vs_loopk {
                assert!(kl >= 0.0, "KL(loop-1 ‖ loop-k) is non-negative at k={k}");
            }
            assert!(row.argmax_loop < config.vocab_size);
            assert!(row.argmax_full < config.vocab_size);
            assert_eq!(row.argmax_flip, row.argmax_loop != row.argmax_full);
            assert!(row.margin_loop.is_finite(), "margin finite at k={k}");
            assert!(row.margin_full.is_finite());
            for λ in row.lambdas_pooled.iter().chain(&row.lambdas_last) {
                assert!(λ.is_finite(), "λ finite at k={k}");
            }
        }
    }

    #[test]
    fn loop_variant_is_not_inert() {
        let (config, weights) = small_model(23);
        let rope_freq = rope(&config);
        let mut scratch = HybridForwardScratch::new(&config);
        let prompt = vec![config.bos_token, 7, 41];
        let full = full_depth_reference(&weights, &config, &prompt, &mut scratch, &rope_freq);
        let row1 = loop_variant_row(
            &weights,
            &config,
            &prompt,
            CHUNK,
            1,
            &full,
            &mut scratch,
            &rope_freq,
        );
        // One extra pass through the chunk's random layers moves the readout
        // — the instrument's premise (a loop pass is not a no-op).
        assert!(
            row1.kl_full_vs_loop > 1e-6,
            "k=1 KL(full‖loop) = {}",
            row1.kl_full_vs_loop
        );
        let row2 = loop_variant_row(
            &weights,
            &config,
            &prompt,
            CHUNK,
            2,
            &full,
            &mut scratch,
            &rope_freq,
        );
        let kl12 = row2
            .kl_loop1_vs_loopk
            .expect("k=2 carries the loop-1 baseline KL");
        assert!(
            kl12 > 0.0,
            "looping beyond 1 changes the readout (KL = {kl12})"
        );
    }

    #[test]
    fn loop_rows_are_deterministic_across_construction() {
        let prompt = |config: &Config| vec![config.bos_token, 13, 251, 4];
        let run = |seed: u64| {
            let (config, weights) = small_model(seed);
            let rope_freq = rope(&config);
            let mut scratch = HybridForwardScratch::new(&config);
            let full = full_depth_reference(
                &weights,
                &config,
                &prompt(&config),
                &mut scratch,
                &rope_freq,
            );
            loop_variant_row(
                &weights,
                &config,
                &prompt(&config),
                CHUNK,
                2,
                &full,
                &mut scratch,
                &rope_freq,
            )
        };
        let a = run(31);
        let b = run(31);
        let bits = |v: &[f64]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&a.lambdas_pooled), bits(&b.lambdas_pooled));
        assert_eq!(bits(&a.lambdas_last), bits(&b.lambdas_last));
        assert_eq!(a.kl_full_vs_loop.to_bits(), b.kl_full_vs_loop.to_bits());
        assert_eq!(
            a.kl_loop1_vs_loopk.unwrap().to_bits(),
            b.kl_loop1_vs_loopk.unwrap().to_bits()
        );
        assert_eq!(a.argmax_loop, b.argmax_loop);
        assert_eq!(a.margin_loop.to_bits(), b.margin_loop.to_bits());
        // Different weights (different seed) are NOT required to agree — a
        // different model may read differently; this is the negative control.
        let c = run(97);
        let same_weights_shape = a.lambdas_pooled.len() == c.lambdas_pooled.len();
        assert!(same_weights_shape);
    }

    #[test]
    fn kl_zero_for_identical_logits() {
        let v: Vec<f32> = (0..64)
            .map(|i| ((i * 37 % 29) as f32 - 14.0) * 0.21)
            .collect();
        assert_eq!(kl_divergence(&v, &v), 0.0);
    }

    #[test]
    fn lambdas_zero_denominator_reads_full_contraction() {
        // Δ² = [4, 0, 1]: t=1 is a real step (0/4 → 0), t=2's denominator is
        // an exact zero → full contraction (0.0), never NaN.
        assert_eq!(lambdas_of(&[4.0, 0.0, 1.0]), vec![0.0, 0.0]);
        assert_eq!(lambdas_of(&[1.0]), Vec::<f64>::new());
    }

    #[test]
    #[should_panic(expected = "mid_start must leave a nonempty input half")]
    fn chunk_validation_refuses_empty_input_half() {
        LoopabilityChunk {
            mid_start: 0,
            chunk_len: 2,
        }
        .validated(6);
    }

    #[test]
    #[should_panic(expected = "chunk must leave a nonempty output half")]
    fn chunk_validation_refuses_empty_output_half() {
        LoopabilityChunk {
            mid_start: 2,
            chunk_len: 4,
        }
        .validated(6);
    }

    #[test]
    fn chunk_validation_accepts_a_middle_chunk() {
        let c = LoopabilityChunk {
            mid_start: 1,
            chunk_len: 4,
        }
        .validated(6);
        assert_eq!((c.mid_start, c.chunk_len), (1, 4));
    }

    #[test]
    #[should_panic(expected = "out of vocab")]
    fn embed_prompt_refuses_out_of_vocab_token() {
        let (config, weights) = small_model(5);
        let mut ctx = PrefillContext::new(&config, 2);
        embed_prompt(&weights, &config, &mut ctx, &[config.vocab_size]);
    }

    #[test]
    #[should_panic(expected = "at least one token")]
    fn embed_prompt_refuses_empty_prompt() {
        let (config, weights) = small_model(5);
        let mut ctx = PrefillContext::new(&config, 2);
        embed_prompt(&weights, &config, &mut ctx, &[]);
    }

    #[test]
    fn rms_unweighted_is_idempotent_up_to_epsilon() {
        let mut v: Vec<f32> = (0..128).map(|i| ((i % 17) as f32 - 8.0) * 0.37).collect();
        rms_unweighted(&mut v);
        let first = v.clone();
        rms_unweighted(&mut v);
        for (a, b) in first.iter().zip(&v) {
            assert!((a - b).abs() < 1e-5, "N(N(x)) ≈ N(x): {a} vs {b}");
        }
    }
}
