//! `si_zero_shot_ppl` — Plan 622 Phase 7 G3: the Bonsai zero-shot split.
//!
//! The scale-invariant schedule (`transformer::attention_si`, opt-in
//! `scale_invariant_attn`) measured at the MODEL level on the packed
//! Bonsai ternary decode path — zero weight edits, one loaded checkpoint
//! shared by every arm, the si session swapped per arm on the forward
//! scratch:
//!
//! | arm | session | pre-recorded expectation |
//!|---|---|---|
//!| `base` | `None` (the plain path) | the reference curve |
//!| `si` | `Armed` (uncalibrated) | near-unit-variance logits (QK-norm) → close to base |
//!| `si-cal` | probe → frozen σ table → `Armed` (calibrated) | the schedule's premise, measured |
//!| `si-pprope` | `Armed` + the pp-RoPE table | **EXPECTED FAIL** (G3 (iii), recorded in advance — a full-RoPE checkpoint loses its trained position representation) |
//!
//! Protocol (per arm): 16 log-spaced length rungs from `--min-length`
//! (128) to `--max-length` (2048); each rung resets the cache, feeds the
//! sequence prefix `0..=L`, and accumulates the NLL of the last
//! `--score-window` (128) targets — the same scored span under a growing
//! context, so the curve reads length-scaling, not text difficulty.
//! Per-rung ppl + the pooled ppl + per-rung Δ vs the `base` arm; the
//! verdict bar is `--tolerance` (default 5%) on EVERY rung.
//!
//! The `si-cal` arm first runs the load-time σ probe (`--probe-tokens`,
//! default 512) — plain math, per-`(layer, head)` raw-logit Welford, sink
//! dropped — and prints the σ diagnostics beside the frozen table digest.
//!
//! `--retrieval` adds the needle probe (G3's "retrieval probe"): a fact
//! at the head of the prompt, filler to the rung length, the question at
//! the end, 6 greedy decode tokens, substring match — at 1024 and 2048.
//!
//! The `si-pprope` arm on an UNMARKED checkpoint requires
//! `--pp-train-len` (loud measurement bypass naming the law); a MARKED
//! checkpoint (Plan-449 marks) goes through `pp_rope_table_from_marks`
//! (the serving path). Either way the row is reported, never asserted.
//!
//! Usage:
//! ```text
//! cargo run --release --features scale_invariant_attn,bonsai2_hadamard \
//!   --bin si_zero_shot_ppl -- \
//!   --gguf ../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf \
//!   --corpus <text> [--arms base,si,si-cal,si-pprope] [--max-length 2048]
//!   [--score-window 128] [--tolerance 0.05] [--probe-tokens 512]
//!   [--tau 10.0] [--retrieval] [--pp-train-len 1024]
//!   [--save-cal <path>] [--probe-only]   (Plan 620 — freeze the σ table
//!   for the GPU G3 lane; e.g. --arms si-cal,base --probe-only --save-cal c.bin)
//! ```

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use katgpt_core::scale_invariant::{HeadCalibration, ScaleInvariantLut, SpreadAccumulator};
#[cfg(feature = "golden_nll")]
use katgpt_core::softmax_fix::{FloatFormat, precision_cliff};
#[cfg(feature = "golden_nll")]
use riir_infer_core::golden_nll::{GoldenNllScratch, GoldenReading};

use riir_infer_core::corpus_text::load_corpus_text;
use riir_infer_core::deltanet::forward::{HybridCache, HybridForwardScratch, effective_rotary_dim};
use riir_infer_core::deltanet::ternary_forward::forward_qwen_deltanet_ternary;
use riir_infer_core::deltanet::ternary_weights::QwenDeltaNetTernaryWeights;
use riir_infer_core::gguf_loader::{
    GgufFile, load_qwen_deltanet_ternary_weights_gguf, pp_rope_table_from_marks,
};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_core::transformer::attention_si::{SiAttentionPlan, SiAttentionSession};
use riir_infer_core::types::{Config, DeltaNetLayerType};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arm {
    Base,
    Si,
    SiCal,
    SiPprope,
}

impl Arm {
    fn parse(s: &str) -> Result<Self> {
        match s {
            "base" => Ok(Arm::Base),
            "si" => Ok(Arm::Si),
            "si-cal" => Ok(Arm::SiCal),
            "si-pprope" => Ok(Arm::SiPprope),
            other => bail!("unknown arm {other} (base|si|si-cal|si-pprope)"),
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Arm::Base => "base",
            Arm::Si => "si",
            Arm::SiCal => "si-cal",
            Arm::SiPprope => "si-pprope",
        }
    }
}

/// One length rung's scoring result.
#[derive(Debug, Clone)]
struct Rung {
    length: usize,
    nll_sum: f64,
    targets: usize,
    /// Issue 959 T6 golden readout (cfg'd `golden_nll`): the same
    /// scored positions through the argmax-excluded-LSE f64 oracle.
    #[cfg(feature = "golden_nll")]
    golden_sum: f64,
    /// Positions in the confident band (top-2 margin ≥ the fp32 cliff
    /// 16.64 nats — where an fp32-consumer readout reads false zeros).
    #[cfg(feature = "golden_nll")]
    confident: usize,
    #[cfg(feature = "golden_nll")]
    golden_confident_sum: f64,
    /// Positions in the deep band (margin ≥ the f64 cliff 36.73 — where
    /// even an f64 max-subtract LSE absorbs the tail).
    #[cfg(feature = "golden_nll")]
    deep: usize,
    #[cfg(feature = "golden_nll")]
    golden_deep_sum: f64,
    /// Positions where the golden NLL and the incumbent f64 form disagree
    /// AT ALL (bit-wise) — the incumbent's resolution-loss census.
    #[cfg(feature = "golden_nll")]
    disagrees: usize,
    #[cfg(feature = "golden_nll")]
    max_abs_delta: f64,
}

impl Rung {
    fn ppl(&self) -> f64 {
        (self.nll_sum / self.targets.max(1) as f64).exp()
    }
}

/// The golden-NLL accumulation state for one rung (Issue 959 T6).
/// Under `cfg(not(feature = "golden_nll"))` this is a zero-sized inert
/// placeholder — the armed path is compile-time unreachable and the
/// runtime flag refuses loud (the harness law: never a silent skip).
#[cfg(feature = "golden_nll")]
#[derive(Debug)]
struct GoldenAccum {
    scratch: GoldenNllScratch,
    sum: f64,
    confident: usize,
    confident_sum: f64,
    deep: usize,
    deep_sum: f64,
    disagrees: usize,
    max_abs_delta: f64,
}

#[cfg(feature = "golden_nll")]
impl GoldenAccum {
    fn new(vocab: usize) -> Self {
        Self {
            scratch: GoldenNllScratch::with_vocab(vocab),
            sum: 0.0,
            confident: 0,
            confident_sum: 0.0,
            deep: 0,
            deep_sum: 0.0,
            disagrees: 0,
            max_abs_delta: 0.0,
        }
    }

    fn record(
        &mut self,
        logits: &[f32],
        target: usize,
        incumbent_nll: f64,
        fp32_cliff: f64,
        f64_cliff: f64,
    ) {
        let GoldenReading { nll, top2_margin } = self.scratch.read_target(logits, target);
        self.sum += nll;
        if top2_margin >= fp32_cliff {
            self.confident += 1;
            self.confident_sum += nll;
        }
        if top2_margin >= f64_cliff {
            self.deep += 1;
            self.deep_sum += nll;
        }
        let delta = (nll - incumbent_nll).abs();
        if nll.to_bits() != incumbent_nll.to_bits() {
            self.disagrees += 1;
        }
        if delta > self.max_abs_delta {
            self.max_abs_delta = delta;
        }
    }
}

#[cfg(not(feature = "golden_nll"))]
#[derive(Debug, Default)]
struct GoldenAccum;

#[cfg(not(feature = "golden_nll"))]
impl GoldenAccum {
    fn inert() -> Self {
        Self
    }
}

/// The state one arm runs its rungs with (cache + scratch are shared and
/// reset per rung; the session is the arm's identity).
struct Rig<'a> {
    config: Config,
    weights: &'a QwenDeltaNetTernaryWeights,
    cache: HybridCache,
    scratch: HybridForwardScratch,
    rope_freq: &'a RopeFreqTable,
}

impl<'a> Rig<'a> {
    fn new(
        config: &Config,
        weights: &'a QwenDeltaNetTernaryWeights,
        rope_freq: &'a RopeFreqTable,
    ) -> Self {
        let layer_types = weights.layer_types.clone();
        Self {
            config: config.clone(),
            weights,
            cache: HybridCache::with_layer_types(config, &layer_types),
            scratch: HybridForwardScratch::new(config),
            rope_freq,
        }
    }

    /// Set the arm's session on the scratch (the G3 A/B swap).
    fn set_session(&mut self, session: Option<SiAttentionSession>) {
        self.scratch.attention.scale_invariant = session;
    }

    /// Feed `seq[from..=to]`; accumulate the NLL of `seq[p+1]` for scored
    /// positions `p >= score_from`. Zero-alloc hot loop (the buffers ride
    /// the rig). `golden` accumulates the Issue-959-T6 oracle readout at
    /// the same scored positions (inert under cfg-off).
    #[allow(clippy::too_many_arguments)]
    fn feed(
        &mut self,
        seq: &[usize],
        from: usize,
        to: usize,
        score_from: usize,
        x: &mut [f32],
        nll: &mut f64,
        targets: &mut usize,
        golden: &mut GoldenAccum,
    ) {
        let n = self.config.n_embd;
        let v = self.config.vocab_size;
        #[cfg(feature = "golden_nll")]
        let fp32_cliff = precision_cliff(FloatFormat::Fp32);
        #[cfg(feature = "golden_nll")]
        let f64_cliff = precision_cliff(FloatFormat::Fp64);
        #[cfg(not(feature = "golden_nll"))]
        let _ = golden; // inert ZST under cfg-off (the armed path is compile-time unreachable)
        for pos in from..=to {
            let logits = forward_qwen_deltanet_ternary(
                x,
                self.weights,
                &mut self.cache,
                seq[pos],
                pos,
                &self.config,
                &mut self.scratch,
                self.rope_freq,
            );
            if pos >= score_from && pos + 1 < seq.len() {
                let target = seq[pos + 1];
                // f64 logsumexp over the raw logits (stable, exact enough
                // for the ppl rows; the exp pass is once per SCORED
                // position, never per fed token).
                let mut max_l = f32::NEG_INFINITY;
                for &l in logits.iter() {
                    max_l = max_l.max(l);
                }
                let mut sum = 0.0f64;
                for &l in logits.iter() {
                    sum += (f64::from(l) - f64::from(max_l)).exp();
                }
                let lse = f64::from(max_l) + sum.ln();
                let inc_nll = lse - f64::from(logits[target]);
                *nll += inc_nll;
                *targets += 1;
                #[cfg(feature = "golden_nll")]
                golden.record(logits, target, inc_nll, fp32_cliff, f64_cliff);
            }
        }
        let _ = (n, v);
    }

    /// Greedy-decode `steps` tokens from the fed prompt (argmax each step).
    fn greedy(
        &mut self,
        next_token: usize,
        next_pos: usize,
        steps: usize,
        x: &mut [f32],
    ) -> Vec<usize> {
        let mut out = Vec::with_capacity(steps);
        let mut token = next_token;
        for pos in next_pos..next_pos + steps {
            let logits = forward_qwen_deltanet_ternary(
                x,
                self.weights,
                &mut self.cache,
                token,
                pos,
                &self.config,
                &mut self.scratch,
                self.rope_freq,
            );
            let mut best = 0usize;
            let mut best_l = f32::NEG_INFINITY;
            for (i, &l) in logits.iter().enumerate() {
                if l > best_l {
                    best_l = l;
                    best = i;
                }
            }
            out.push(best);
            token = best;
        }
        out
    }

    /// Issue 1188 — the paired-ablation scoring pass: same forward + same
    /// scored-span contract as [`Rig::feed`], but every scored position's
    /// reading is COLLECTED (the paired per-position Δ needs both passes'
    /// rows). The incumbent NLL fold is the same LSE shape as `feed`'s;
    /// the golden readout rides the same oracle scratch (cfg-gated — the
    /// ablate entry point refuses loud without the feature).
    fn feed_collect(
        &mut self,
        seq: &[usize],
        from: usize,
        to: usize,
        score_from: usize,
        x: &mut [f32],
    ) -> Vec<ScoredReading> {
        let mut out = Vec::new();
        #[cfg(feature = "golden_nll")]
        let mut scratch = GoldenNllScratch::with_vocab(self.config.vocab_size);
        for pos in from..=to {
            let logits = forward_qwen_deltanet_ternary(
                x,
                self.weights,
                &mut self.cache,
                seq[pos],
                pos,
                &self.config,
                &mut self.scratch,
                self.rope_freq,
            );
            if pos >= score_from && pos + 1 < seq.len() {
                let target = seq[pos + 1];
                let mut max_l = f32::NEG_INFINITY;
                for &l in logits.iter() {
                    max_l = max_l.max(l);
                }
                let mut sum = 0.0f64;
                for &l in logits.iter() {
                    sum += (f64::from(l) - f64::from(max_l)).exp();
                }
                let lse = f64::from(max_l) + sum.ln();
                let inc_nll = lse - f64::from(logits[target]);
                let mut argmax = 0usize;
                let mut best_l = f32::NEG_INFINITY;
                for (i, &l) in logits.iter().enumerate() {
                    if l > best_l {
                        best_l = l;
                        argmax = i;
                    }
                }
                #[cfg(feature = "golden_nll")]
                let GoldenReading { nll, top2_margin } = scratch.read_target(logits, target);
                out.push(ScoredReading {
                    inc_nll,
                    argmax,
                    #[cfg(feature = "golden_nll")]
                    golden_nll: nll,
                    #[cfg(feature = "golden_nll")]
                    top2_margin,
                });
            }
        }
        out
    }
}

/// One scored position's paired reading ([`Rig::feed_collect`]).
#[derive(Debug, Clone)]
struct ScoredReading {
    /// The incumbent f64-LSE NLL (the same fold `feed` accumulates).
    inc_nll: f64,
    /// First-index strict-greater argmax (the same rule `greedy` walks).
    argmax: usize,
    /// The Issue-959-T6 golden-oracle NLL (cfg-gated: the oracle is the
    /// `golden_nll` feature's payload).
    #[cfg(feature = "golden_nll")]
    golden_nll: f64,
    /// Top-2 logit margin in nats (the band coordinate).
    #[cfg(feature = "golden_nll")]
    top2_margin: f64,
}

/// The load-time σ probe: plain math over `n_tokens`, per-`(layer, head)`
/// raw-logit Welford, sink dropped (the session does the contract).
fn run_sigma_probe(
    rig: &mut Rig,
    seq: &[usize],
    n_tokens: usize,
    x: &mut [f32],
) -> Result<SpreadAccumulator> {
    let slots = rig.weights.layers.len() * rig.config.n_head;
    rig.cache.reset();
    rig.set_session(Some(SiAttentionSession::Probe(SpreadAccumulator::new(
        slots,
    ))));
    let to = n_tokens.min(seq.len()) - 1;
    let mut nll = 0.0;
    let mut targets = 0usize;
    #[cfg(feature = "golden_nll")]
    let mut probe_golden = GoldenAccum::new(0);
    #[cfg(not(feature = "golden_nll"))]
    let mut probe_golden = GoldenAccum::inert();
    rig.feed(
        seq,
        0,
        to,
        usize::MAX,
        x,
        &mut nll,
        &mut targets,
        &mut probe_golden,
    );
    match rig.scratch.attention.scale_invariant.take() {
        Some(SiAttentionSession::Probe(acc)) => Ok(acc),
        _ => bail!("sigma probe: session vanished — internal wiring bug"),
    }
}

/// Extract the frozen table from the probe. `expected_observed` is the
/// attention-layer slot count (`attention_layers × n_head`): every one of
/// those MUST have been observed (loud on shortfall — a probe that measured
/// nothing is a broken probe), while the GDN slots have no per-key logits
/// (the Plan-622 audited discard) and ride the NEUTRAL σ = 1.0 — a value
/// the armed path never reads (the dispatch fires only in attention
/// layers). Degenerate σ on an OBSERVED slot stays loud — a zero-σ head is
/// a wrong population, never a divide.
fn calibration_from_probe(
    acc: &SpreadAccumulator,
    slots: usize,
    expected_observed: usize,
) -> Result<HeadCalibration> {
    // `heads()` is the PRE-SIZED slot count — the observed set is the slots
    // carrying a std estimate. Every attention-layer slot MUST have one
    // (loud on shortfall — a probe that measured nothing is a broken probe,
    // not a calibration); the GDN slots have no per-key logits (the
    // Plan-622 audited discard) and ride the NEUTRAL σ = 1.0, a value the
    // armed path never reads (the dispatch fires only in attention layers).
    let mut neutral = 0usize;
    let mut scales = Vec::with_capacity(slots);
    for slot in 0..slots {
        match acc.std_estimate(slot) {
            Some(sigma) => {
                if !(sigma.is_finite() && sigma > 0.0) {
                    bail!("sigma probe: slot {slot} σ = {sigma} — degenerate population");
                }
                scales.push(sigma as f32);
            }
            None => {
                scales.push(1.0);
                neutral += 1;
            }
        }
    }
    let observed = slots - neutral;
    ensure!(
        observed == expected_observed,
        "sigma probe: observed {observed} of {expected_observed} attention-layer slots — \
         a shortfall is a broken probe, not a calibration"
    );
    println!(
        "# calibration table: {observed} measured + {neutral} neutral (GDN, never read) = {slots} slots",
    );
    HeadCalibration::build(scales).map_err(anyhow::Error::msg)
}

// ── Issue 1188 — the family-ablation cell ───────────────────────────

/// One ablated tensor's row (the per-tensor disclosure table).
struct AblationRow {
    name: String,
    class: &'static str,
    elements: usize,
    bytes_before: usize,
    bytes_after: usize,
    /// Vector-class tensors whose trailing `len % 32` elements were copied
    /// verbatim (Q8 blocks need 32-multiple spans; the tail is disclosed,
    /// never silently dropped).
    tail_copied: usize,
}

/// The family ablation result.
struct AblationReport {
    tensors: usize,
    elements: u64,
    bytes_before: u64,
    bytes_after: u64,
    rows: Vec<AblationRow>,
}

/// Re-quantize ONE family's tensors in place to the candidate posture
/// (v1: `Q8_0` blocks — quant-resident for Proj/GateProj classes, a
/// quantize→dequant round-trip for the Vector classes). Walk order is
/// layer-ascending and field-fixed — deterministic, no RNG.
fn apply_family_ablation(
    weights: &mut QwenDeltaNetTernaryWeights,
    family: riir_infer_core::family_census::Family,
) -> Result<AblationReport> {
    use riir_infer_core::deltanet::ternary_weights::{GateProjWeights, ProjWeights};
    use riir_infer_core::family_census::{Family as F, StorageClass};

    // v1 scope: the GLOBAL tensors (token_embd / output) are bare
    // `TernaryGroupWeights` fields — the ternary forward's embedding
    // gather + lm_head matvec read them as such, and the Q8_0 candidate
    // cannot materialize in that container without a serving-path type
    // change (out of the tool's scope). Loud refusal, never a silent skip;
    // the per-layer families (16 of 18) carry the ranking substance.
    if matches!(family, F::TokenEmbd | F::Output) {
        bail!(
            "family-ablate {}: v1 prices the PER-LAYER families only — the global tensors are \
             bare TernaryGroupWeights and cannot hold the Q8_0 candidate without a \
             serving-path type change (QwenDeltaNetTernaryWeights::wte/lm_head); \
             a v2 extension switches the globals to ProjWeights",
            family.slug()
        );
    }

    let mut rows: Vec<AblationRow> = Vec::new();
    fn push_row(
        rows: &mut Vec<AblationRow>,
        name: String,
        class: &'static str,
        elements: usize,
        before: usize,
        after: usize,
    ) {
        rows.push(AblationRow {
            name,
            class,
            elements,
            bytes_before: before,
            bytes_after: after,
            tail_copied: 0,
        });
    }

    match family.storage_class() {
        StorageClass::Proj => {
            for (i, layer) in weights.layers.iter_mut().enumerate() {
                let mut pick: [&mut ProjWeights; 1] = match family {
                    F::AttnQ => [&mut layer.attn_wq],
                    F::AttnK => [&mut layer.attn_wk],
                    F::AttnV => [&mut layer.attn_wv],
                    F::AttnOut => [&mut layer.attn_wo],
                    F::AttnGate => [&mut layer.in_proj_z],
                    F::GdnQkv => [&mut layer.in_proj_qkv],
                    F::GdnOut => [&mut layer.out_proj],
                    F::FfnGate => [&mut layer.gate_proj],
                    F::FfnUp => [&mut layer.up_proj],
                    F::FfnDown => [&mut layer.down_proj],
                    _ => continue,
                };
                let w = &mut pick[0];
                if w.rows() == 0 {
                    continue; // the layer kind's unused field (empty placeholder)
                }
                let dense = w.dequant_to_dense();
                let (r, c) = (w.rows(), w.cols());
                let before = w.bytes();
                **w = ProjWeights::from_dense_q8_0(&dense, r, c);
                push_row(
                    &mut rows,
                    format!("blk.{i}.{}", family.slug()),
                    "proj",
                    r * c,
                    before,
                    w.bytes(),
                );
            }
        }
        StorageClass::GateProj => {
            for (i, layer) in weights.layers.iter_mut().enumerate() {
                let gate = match family {
                    F::GdnAlpha => &mut layer.in_proj_a,
                    F::GdnBeta => &mut layer.in_proj_b,
                    _ => continue,
                };
                if gate.rows() == 0 {
                    continue;
                }
                let dense = gate.dequant_to_dense();
                let (r, c) = (gate.rows(), gate.cols());
                let before = gate.bytes();
                *gate = GateProjWeights::from_dense_q8_0(&dense, r, c);
                push_row(
                    &mut rows,
                    format!("blk.{i}.{}", family.slug()),
                    "gate_proj",
                    r * c,
                    before,
                    gate.bytes(),
                );
            }
        }
        StorageClass::Vector => {
            // Quantize→dequant round-trip in place: the storage-precision
            // sensitivity read for the runtime-f32 vectors; the forward
            // code is untouched. Trailing len % 32 elements copy verbatim.
            fn round_trip(v: &mut [f32], name: String, rows: &mut Vec<AblationRow>) {
                use bytemuck::Zeroable;
                use riir_infer_core::quant::q8kv::{
                    BlockQ8_0, dequantize_row_q8_0, quantize_row_q8_0,
                };
                let before = std::mem::size_of_val(v);
                let elements = v.len();
                let n_blocks = elements / 32;
                let tail = elements % 32;
                if n_blocks > 0 {
                    let mut blocks = vec![BlockQ8_0::zeroed(); n_blocks];
                    quantize_row_q8_0(&v[..n_blocks * 32], &mut blocks);
                    dequantize_row_q8_0(&blocks, &mut v[..n_blocks * 32]);
                }
                rows.push(AblationRow {
                    name,
                    class: "vector",
                    elements,
                    bytes_before: before,
                    bytes_after: before, // in place — same bytes, perturbed values
                    tail_copied: tail,
                });
            }
            if family == F::Norms {
                round_trip(
                    &mut weights.final_norm,
                    "output_norm.weight".into(),
                    &mut rows,
                );
                for (i, layer) in weights.layers.iter_mut().enumerate() {
                    round_trip(
                        &mut layer.input_norm,
                        format!("blk.{i}.attn_norm.weight"),
                        &mut rows,
                    );
                    round_trip(
                        &mut layer.post_attn_norm,
                        format!("blk.{i}.post_attention_norm.weight"),
                        &mut rows,
                    );
                    if !layer.linear_norm.is_empty() {
                        round_trip(
                            &mut layer.linear_norm,
                            format!("blk.{i}.ssm_norm.weight"),
                            &mut rows,
                        );
                    }
                    if !layer.attn_q_norm.is_empty() {
                        round_trip(
                            &mut layer.attn_q_norm,
                            format!("blk.{i}.attn_q_norm.weight"),
                            &mut rows,
                        );
                    }
                    if !layer.attn_k_norm.is_empty() {
                        round_trip(
                            &mut layer.attn_k_norm,
                            format!("blk.{i}.attn_k_norm.weight"),
                            &mut rows,
                        );
                    }
                }
            }
            for (i, layer) in weights.layers.iter_mut().enumerate() {
                match family {
                    F::GdnConv if !layer.conv1d_weight.is_empty() => {
                        round_trip(
                            &mut layer.conv1d_weight,
                            format!("blk.{i}.ssm_conv1d.weight"),
                            &mut rows,
                        );
                    }
                    F::GdnDt if !layer.dt_bias.is_empty() => {
                        round_trip(
                            &mut layer.dt_bias,
                            format!("blk.{i}.ssm_dt.bias"),
                            &mut rows,
                        );
                    }
                    F::GdnA if !layer.a_log.is_empty() => {
                        round_trip(&mut layer.a_log, format!("blk.{i}.ssm_a"), &mut rows);
                    }
                    _ => {}
                }
            }
        }
    }

    let report = AblationReport {
        tensors: rows.len(),
        elements: rows.iter().map(|r| r.elements as u64).sum(),
        bytes_before: rows.iter().map(|r| r.bytes_before as u64).sum(),
        bytes_after: rows.iter().map(|r| r.bytes_after as u64).sum(),
        rows,
    };
    ensure!(
        report.tensors > 0,
        "family-ablate {}: no tensors of this family in the checkpoint — wrong family or wrong file",
        family.slug()
    );
    Ok(report)
}

/// The paired ablation cell (Issue 1188 T1): score the SAME scored span
/// through the incumbent weights and the family-ablated weights (ONE
/// frozen read — the corpus prefix is fixed, both passes walk it
/// identically), then report the paired golden Δ-NLL + the disagreement
/// census. Report-only: the tool RANKS, the posture decision is
/// owner-gated per the lossy-surface promotion rule.
#[allow(clippy::too_many_arguments)]
fn run_family_ablation(
    weights: &mut QwenDeltaNetTernaryWeights,
    config: &Config,
    rope_freq: &RopeFreqTable,
    seq: &[usize],
    family: riir_infer_core::family_census::Family,
    posture: &str,
    prefix: usize,
    score_window: usize,
) -> Result<()> {
    if !cfg!(feature = "golden_nll") {
        bail!(
            "--family-ablate needs the golden_nll feature (loud refusal, never a silent skip): \
             cargo build --release --features scale_invariant_attn,bonsai2_hadamard,golden_nll \
             --bin si_zero_shot_ppl"
        );
    }
    if posture != "q8" {
        bail!(
            "--family-posture '{posture}' unsupported in v1: the tool prices the Q8 candidate \
             (Q8_0 blocks) on this lane; 'pq2' would apply to a NON-PQ2 incumbent and the \
             in-scope Bonsai incumbent IS PQ2_0"
        );
    }
    ensure!(
        seq.len() > prefix,
        "corpus too short: {} tokens < prefix {} + 1",
        seq.len(),
        prefix
    );
    ensure!(
        prefix > score_window,
        "prefix {prefix} must exceed the scored span {score_window}"
    );

    let feed_to = prefix - 1;
    let score_from = prefix - score_window;
    let mut x = vec![0.0f32; config.n_embd.max(config.vocab_size)];

    // Pass 1 — the incumbent (every family at its loaded posture).
    let t = Instant::now();
    let mut rig = Rig::new(config, weights, rope_freq);
    let incumbent = rig.feed_collect(seq, 0, feed_to, score_from, &mut x);
    drop(rig);
    println!(
        "# pass 1 (incumbent): {} scored positions in {:.1}s",
        incumbent.len(),
        t.elapsed().as_secs_f32()
    );

    // The ablation (deterministic walk; the report is the engagement proof).
    let t = Instant::now();
    let report = apply_family_ablation(weights, family)?;
    println!(
        "\n# family-ablate {} -> q8 | {} tensor(s) requantized in {:.1}s",
        family.slug(),
        report.tensors,
        t.elapsed().as_secs_f32()
    );
    println!(
        "#   totals: {} elements | bytes {} -> {} ({:.2} -> {:.2} bpw)",
        report.elements,
        report.bytes_before,
        report.bytes_after,
        (report.bytes_before as f64 * 8.0) / report.elements.max(1) as f64,
        (report.bytes_after as f64 * 8.0) / report.elements.max(1) as f64,
    );
    for r in &report.rows {
        println!(
            "#   {} [{}] {} elems, {} -> {} bytes{}",
            r.name,
            r.class,
            r.elements,
            r.bytes_before,
            r.bytes_after,
            if r.tail_copied > 0 {
                format!(", tail {} copied verbatim (len % 32)", r.tail_copied)
            } else {
                String::new()
            },
        );
    }

    // Pass 2 — the ablated weights, the SAME scored span.
    let t = Instant::now();
    let mut rig = Rig::new(config, weights, rope_freq);
    let ablated = rig.feed_collect(seq, 0, feed_to, score_from, &mut x);
    drop(rig);
    println!(
        "# pass 2 (ablated): {} scored positions in {:.1}s",
        ablated.len(),
        t.elapsed().as_secs_f32()
    );

    // ── the paired report ──
    ensure!(
        incumbent.len() == ablated.len() && !incumbent.is_empty(),
        "paired passes scored different position counts ({} vs {}) — internal wiring bug",
        incumbent.len(),
        ablated.len()
    );
    let n = incumbent.len();
    #[cfg(feature = "golden_nll")]
    {
        use katgpt_core::softmax_fix::{FloatFormat, precision_cliff};
        let fp32_cliff = precision_cliff(FloatFormat::Fp32);
        let f64_cliff = precision_cliff(FloatFormat::Fp64);
        let mut sum_inc = 0.0f64;
        let mut sum_ab = 0.0f64;
        let mut max_abs = 0.0f64;
        let mut flips = 0usize;
        let mut band_n = 0usize;
        let mut band_delta = 0.0f64;
        let mut deep_n = 0usize;
        let mut deep_delta = 0.0f64;
        let mut worst = Vec::<(f64, usize)>::new();
        for (i, (a, b)) in incumbent.iter().zip(ablated.iter()).enumerate() {
            sum_inc += a.golden_nll;
            sum_ab += b.golden_nll;
            let d = b.golden_nll - a.golden_nll;
            max_abs = max_abs.max(d.abs());
            if a.argmax != b.argmax {
                flips += 1;
            }
            if a.top2_margin >= fp32_cliff {
                band_n += 1;
                band_delta += d;
            }
            if a.top2_margin >= f64_cliff {
                deep_n += 1;
                deep_delta += d;
            }
            worst.push((d.abs(), i));
        }
        worst.sort_by(|x, y| y.0.partial_cmp(&x.0).expect("finite"));
        let pooled = (sum_ab - sum_inc) / n as f64;
        let mut sum_inc_lse = 0.0f64;
        let mut sum_ab_lse = 0.0f64;
        for (a, b) in incumbent.iter().zip(ablated.iter()) {
            sum_inc_lse += a.inc_nll;
            sum_ab_lse += b.inc_nll;
        }
        println!(
            "\n# paired golden readout (Issue 1188 T1, one frozen read, {} positions):",
            n
        );
        println!(
            "#   pooled Δ-NLL {pooled:+.3e} nats/token | mean |Δ| {:.3e} | max |Δ| {max_abs:.3e}",
            (sum_ab - sum_inc).abs() / n as f64,
        );
        println!(
            "#   incumbent-LSE readout cross-check: pooled Δ {:+.3e} nats/token (the f64-LSE view of the same delta)",
            (sum_ab_lse - sum_inc_lse) / n as f64,
        );
        println!(
            "#   golden ppl: incumbent {:.6} -> ablated {:.6}",
            (sum_inc / n as f64).exp(),
            (sum_ab / n as f64).exp()
        );
        println!("#   argmax flips: {flips}/{n}");
        println!(
            "#   fp32-cliff band (margin ≥ {:.2} nats): {band_n}/{n} positions, pooled Δ {:+.3e}",
            fp32_cliff, band_delta,
        );
        println!(
            "#   f64-cliff deep band (margin ≥ {:.2} nats): {deep_n}/{n} positions, pooled Δ {:+.3e}",
            f64_cliff, deep_delta,
        );
        print!("#   top |Δ| positions:");
        for (d, i) in worst.iter().take(3) {
            let i = *i;
            let delta = ablated[i].golden_nll - incumbent[i].golden_nll;
            let _ = d;
            print!(" pos{} {delta:+.3e}", score_from + i);
        }
        println!();
        println!(
            "# tool verdict: RANK ONLY — the posture decision is owner-gated (the lossy-surface promotion rule)"
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let mut gguf_path: Option<PathBuf> = None;
    let mut corpus: Option<PathBuf> = None;
    let mut arms = vec![Arm::Base, Arm::Si, Arm::SiCal, Arm::SiPprope];
    let mut max_length = 2048usize;
    let mut min_length = 128usize;
    let mut rungs_n = 16usize;
    let mut score_window = 128usize;
    let mut tolerance = 0.05f64;
    let mut probe_tokens = 512usize;
    let mut tau = 10.0f32;
    let mut retrieval = false;
    let mut pp_train_len: Option<usize> = None;
    // Plan 620 — the GPU G3 lane consumes the frozen σ table: `--save-cal`
    // writes the HeadCalibration sidecar right after the probe (BLAKE3-checked
    // format, `HeadCalibration::to_sidecar`), `--probe-only` stops there.
    let mut save_cal: Option<PathBuf> = None;
    let mut probe_only = false;
    // Issue 959 T6 — the f64 golden-NLL readout over the scored spans
    // (reference-oracle mode: base arm only, one forward, two readouts).
    let mut golden_nll = false;
    // Issue 1188 — the family sensitivity tooling. `--family-census` is a
    // header-only early-exit (no corpus, no forward); `--family-ablate`
    // runs the paired requant cell (golden_nll-gated).
    let mut family_census = false;
    let mut family_ablate: Option<riir_infer_core::family_census::Family> = None;
    let mut family_posture = String::from("q8");
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--gguf" => gguf_path = Some(PathBuf::from(args.next().expect("path"))),
            "--corpus" => corpus = Some(PathBuf::from(args.next().expect("path"))),
            "--arms" => {
                arms = args
                    .next()
                    .expect("list")
                    .split(',')
                    .map(Arm::parse)
                    .collect::<Result<Vec<_>>>()?;
                if !arms.contains(&Arm::Base) {
                    bail!("the base arm is the reference side — always included");
                }
            }
            "--max-length" => max_length = args.next().expect("n").parse()?,
            "--min-length" => min_length = args.next().expect("n").parse()?,
            "--rungs" => rungs_n = args.next().expect("n").parse()?,
            "--score-window" => score_window = args.next().expect("n").parse()?,
            "--tolerance" => tolerance = args.next().expect("x").parse()?,
            "--probe-tokens" => probe_tokens = args.next().expect("n").parse()?,
            "--tau" => tau = args.next().expect("x").parse()?,
            "--retrieval" => retrieval = true,
            "--pp-train-len" => pp_train_len = Some(args.next().expect("n").parse()?),
            "--save-cal" => save_cal = Some(PathBuf::from(args.next().expect("path"))),
            "--probe-only" => probe_only = true,
            "--golden-nll" => golden_nll = true,
            "--family-census" => family_census = true,
            "--family-ablate" => {
                let slug = args.next().expect("family slug");
                family_ablate = Some(
                    riir_infer_core::family_census::Family::parse(&slug)
                        .unwrap_or_else(|| panic!(
                            "unknown family '{slug}' (see --family-census rollup for the vocabulary)"
                        )),
                );
            }
            "--family-posture" => family_posture = args.next().expect("posture"),
            other => bail!("unknown arg {other}"),
        }
    }
    let gguf_path = gguf_path.context("--gguf is required")?;

    // ── Issue 1188 census mode: header-only, no corpus, no forward ──
    if family_census {
        let file = GgufFile::open(&gguf_path).context("open for census")?;
        let census = riir_infer_core::family_census::census_of(&file);
        println!(
            "# family census | file {} | arch {:?} | tensors {} | unclassified {}",
            gguf_path.display(),
            file.architecture(),
            census.rows.len(),
            census.unclassified,
        );
        println!("# name\tfamily\tclass\ttype\tshape(ne0xne1x...)\telements\tbytes");
        for row in &census.rows {
            let (fam, cls) = match row.classified {
                Some(c) => (
                    c.family.slug().to_string(),
                    c.family.storage_class().to_string(),
                ),
                None => ("UNCLASSIFIED".to_string(), "-".to_string()),
            };
            let shape = row
                .shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join("x");
            println!(
                "{}\t{fam}\t{cls}\t{:?}\t{shape}\t{}\t{}",
                row.name,
                row.ggml_type,
                row.shape.iter().product::<usize>(),
                row.byte_len,
            );
        }
        println!("# rollup: family\ttensors\t elements\tbytes\ttypes");
        for r in &census.rollups {
            println!(
                "#   {}\t{}\t{}\t{}\t{}",
                r.family.slug(),
                r.tensors,
                r.elements,
                r.bytes,
                r.types.join(","),
            );
        }
        if census.unclassified > 0 {
            bail!(
                "family census: {} unclassified tensor(s) — extend the taxonomy before \
                 any ablation on this file (a silent miss is a skipped family)",
                census.unclassified
            );
        }
        return Ok(());
    }

    let corpus = corpus.context("--corpus is required")?;
    if golden_nll && !cfg!(feature = "golden_nll") {
        bail!(
            "--golden-nll needs the golden_nll feature (loud refusal, never a silent skip): \
             cargo build --release --features scale_invariant_attn,bonsai2_hadamard,golden_nll --bin si_zero_shot_ppl"
        );
    }
    if golden_nll {
        if arms.len() != 1 || arms[0] != Arm::Base {
            println!(
                "# golden-nll: reference-oracle mode — the readout rides the BASE arm's forward \
                 (one forward, two readouts); overriding --arms to base"
            );
            arms = vec![Arm::Base];
        }
        println!(
            "# golden-nll: Issue 959 T6 — per-token NLL via the argmax-excluded-LSE f64 oracle \
             (katgpt-core fixed_log_softmax_target_f64); confidence bands at the fp32 cliff \
             16.64 / f64 cliff 36.73 nats"
        );
    }
    assert!(min_length >= 4 && max_length > min_length, "4 <= min < max");
    assert!(
        score_window >= 8 && score_window <= min_length,
        "8 <= score-window <= min-length"
    );

    // ── load once, share across arms ──
    let t0 = Instant::now();
    let (mut config, mut weights) = load_qwen_deltanet_ternary_weights_gguf(&gguf_path)
        .with_context(|| format!("load {}", gguf_path.display()))?;
    println!(
        "# si_zero_shot_ppl: {} | layers={} heads={}/{} rot={} | load {:.1}s",
        gguf_path.display(),
        config.n_layer,
        config.n_head,
        config.n_kv_head,
        weights.rotation.is_some(),
        t0.elapsed().as_secs_f32()
    );

    // The row_logit_floor law: cap the cache window at what the run needs,
    // BEFORE any cache construction.
    config.block_size = max_length + 32;

    let gguf = GgufFile::open(&gguf_path).context("re-open for tokenizer + marks")?;
    let tok = BpeTokenizer::from_gguf(&gguf).context("gpt2 BPE tokenizer from gguf")?;

    // The pp-RoPE table for the si-pprope arm: marks first (the serving
    // path), else the loud measurement bypass.
    let pp_table = match pp_rope_table_from_marks(&gguf, &config) {
        Ok(t) => {
            println!(
                "# pp-RoPE: Plan-449 marks present — kept {} of {} pairs (base-marked)",
                t.kept(),
                effective_rotary_dim(&config) / 2
            );
            Some(t)
        }
        Err(e) => match pp_train_len {
            Some(len) => {
                println!(
                    "# pp-RoPE: UNMARKED CHECKPOINT — measurement bypass at --pp-train-len {len} \
                     (Plan 622 G3 (iii): the zero-shot pairing is the pre-recorded expected FAIL; \
                     loader verdict was: {e})"
                );
                Some(RopeFreqTable::new_pp(
                    config.rope_theta,
                    effective_rotary_dim(&config),
                    len,
                ))
            }
            None => {
                println!(
                    "# pp-RoPE: unmarked checkpoint and no --pp-train-len — the si-pprope arm \
                     will refuse if requested ({e})"
                );
                None
            }
        },
    };
    drop(gguf);

    // ── the frozen sequence (byte-identical for every arm) ──
    let text = load_corpus_text(&corpus)?;
    let mut seq = tok.encode_with_bos(&text);
    seq.truncate(max_length + 1);
    if seq.len() < min_length + 1 {
        bail!(
            "corpus too short: {} tokens (BOS incl.) < min-length {} + 1",
            seq.len(),
            min_length
        );
    }
    println!(
        "# corpus: {} tokens (BOS + text) | seq[0]=BOS={} sink carve-out armed | τ = {tau}",
        seq.len(),
        tok.bos_id()
    );

    // The 16 log-spaced rungs (the schedule is logarithmic in t; the curve
    // reads its shape, and the feed cost stays under the linear ladder's).
    let rungs: Vec<usize> = (0..rungs_n)
        .map(|k| {
            let l = min_length as f64
                * (max_length as f64 / min_length as f64).powf(k as f64 / (rungs_n - 1) as f64);
            l.round() as usize
        })
        .collect();
    println!(
        "# rungs ({}): {:?} | scored span {score_window} tokens per rung",
        rungs.len(),
        rungs
    );

    let plain_table = RopeFreqTable::new(config.rope_theta, effective_rotary_dim(&config));

    // ── Issue 1188: the paired family-ablation cell (standalone mode) ──
    if let Some(family) = family_ablate {
        if retrieval || save_cal.is_some() || probe_only {
            bail!(
                "--family-ablate is a standalone paired cell — incompatible with \
                 --retrieval / --save-cal / --probe-only"
            );
        }
        run_family_ablation(
            &mut weights,
            &config,
            &plain_table,
            &seq,
            family,
            &family_posture,
            max_length,
            score_window,
        )?;
        return Ok(());
    }

    // The session LUT is arm-independent — one build at the block size.
    let lut = Arc::new(ScaleInvariantLut::build(tau, config.block_size));
    println!(
        "# LUT: max_distance {} (block {}) | ατ·ln(1+T/τ) at T={max_length} ≈ {:.1}",
        lut.max_distance(),
        config.block_size,
        (config.block_size as f64 / tau as f64).ln() * core::f64::consts::E.sqrt() * tau as f64
    );

    let mut rig = Rig::new(&config, &weights, &plain_table);
    let mut x = vec![0.0f32; config.n_embd.max(config.vocab_size)];

    // σ table built once (the si-cal arm), the diagnostics beside it.
    let mut calibration: Option<HeadCalibration> = None;
    if arms.contains(&Arm::SiCal) {
        let t = Instant::now();
        let acc = run_sigma_probe(&mut rig, &seq, probe_tokens, &mut x)?;
        let slots = weights.layers.len() * config.n_head;
        let attention_slots = weights
            .layer_types
            .iter()
            .filter(|t| **t == DeltaNetLayerType::Attention)
            .count()
            * config.n_head;
        let cal = calibration_from_probe(&acc, slots, attention_slots)?;
        // Diagnostics: the σ spread across slots is the premise check —
        // near-unity σ (QK-norm) says the uncalibrated arm should already
        // track the schedule's unit-variance premise.
        let mut sig: Vec<f64> = (0..slots).map(|s| f64::from(cal.scale_for(s))).collect();
        sig.sort_by(|a, b| a.partial_cmp(b).expect("finite σ"));
        let q = |p: f64| sig[((sig.len() - 1) as f64 * p) as usize];
        println!(
            "# sigma probe: {probe_tokens} tokens, {slots} slots in {:.1}s | σ min {:.4} p25 {:.4} \
             median {:.4} p75 {:.4} max {:.4} | mean {:.4}",
            t.elapsed().as_secs_f32(),
            sig[0],
            q(0.25),
            q(0.5),
            q(0.75),
            sig[sig.len() - 1],
            sig.iter().sum::<f64>() / sig.len() as f64
        );
        println!(
            "# calibration table: digest {:02x}… over {slots} slots",
            cal.digest()[..4]
                .iter()
                .fold(0u64, |a, b| (a << 8) | u64::from(*b))
                & 0xffff
        );
        // Plan 620 — freeze the table for the GPU G3 lane (the sidecar is
        // BLAKE3-checked on load; quote host + digest beside every row).
        if let Some(path) = &save_cal {
            std::fs::write(path, cal.to_sidecar())
                .with_context(|| format!("write cal sidecar {}", path.display()))?;
            println!("# cal sidecar written: {}", path.display());
        }
        if probe_only {
            println!("# --probe-only: stopping after the probe (the GPU lane owns the arms)");
            return Ok(());
        }
        calibration = Some(cal);
    }

    // ── arms ──
    let mut results: Vec<(Arm, Vec<Rung>)> = Vec::new();
    for arm in &arms {
        let t = Instant::now();
        match *arm {
            Arm::Base => rig.set_session(None),
            Arm::Si => rig.set_session(Some(SiAttentionSession::Armed(Arc::new(
                SiAttentionPlan::uncalibrated(lut.clone(), config.n_head),
            )))),
            Arm::SiCal => {
                let cal = calibration
                    .as_ref()
                    .expect("calibration built above")
                    .clone();
                let plan = SiAttentionPlan::calibrated(
                    lut.clone(),
                    cal,
                    config.n_head,
                    weights.layers.len(),
                )
                .map_err(anyhow::Error::msg)?;
                rig.set_session(Some(SiAttentionSession::Armed(Arc::new(plan))));
            }
            Arm::SiPprope => {
                let Some(table) = pp_table.as_ref() else {
                    bail!(
                        "si-pprope requested but no pp table: the checkpoint is unmarked and \
                         --pp-train-len was not given (the serving path loud-fails too)"
                    );
                };
                rig.rope_freq = table;
                rig.set_session(Some(SiAttentionSession::Armed(Arc::new(
                    SiAttentionPlan::uncalibrated(lut.clone(), config.n_head),
                ))));
            }
        }
        let mut rung_results = Vec::with_capacity(rungs.len());
        #[cfg(feature = "golden_nll")]
        let mut golden = if golden_nll {
            GoldenAccum::new(config.vocab_size)
        } else {
            GoldenAccum::new(0)
        };
        #[cfg(not(feature = "golden_nll"))]
        let mut golden = GoldenAccum::inert();
        for &l in &rungs {
            rig.cache.reset();
            let mut nll = 0.0f64;
            let mut targets = 0usize;
            #[cfg(feature = "golden_nll")]
            {
                golden.sum = 0.0;
                golden.confident = 0;
                golden.confident_sum = 0.0;
                golden.deep = 0;
                golden.deep_sum = 0.0;
                golden.disagrees = 0;
                golden.max_abs_delta = 0.0;
            }
            let score_from = l.saturating_sub(score_window);
            rig.feed(
                &seq,
                0,
                l,
                score_from,
                &mut x,
                &mut nll,
                &mut targets,
                &mut golden,
            );
            rung_results.push(Rung {
                length: l,
                nll_sum: nll,
                targets,
                #[cfg(feature = "golden_nll")]
                golden_sum: golden.sum,
                #[cfg(feature = "golden_nll")]
                confident: golden.confident,
                #[cfg(feature = "golden_nll")]
                golden_confident_sum: golden.confident_sum,
                #[cfg(feature = "golden_nll")]
                deep: golden.deep,
                #[cfg(feature = "golden_nll")]
                golden_deep_sum: golden.deep_sum,
                #[cfg(feature = "golden_nll")]
                disagrees: golden.disagrees,
                #[cfg(feature = "golden_nll")]
                max_abs_delta: golden.max_abs_delta,
            });
        }
        // Restore the plain rope table for the next arm (the pprope arm's
        // swap must not leak).
        if *arm == Arm::SiPprope {
            rig.rope_freq = &plain_table;
        }
        println!(
            "# arm {} done in {:.1}s",
            arm.name(),
            t.elapsed().as_secs_f32()
        );
        results.push((*arm, rung_results));
    }

    // ── the G3 table ──
    println!("\n# G3 zero-shot split — pooled ppl over {score_window}-token scored spans");
    println!(
        "# {:>8} {:>14}",
        "length",
        rungs
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join(" ")
            .truncate_to_width(14)
    );
    let base = &results[0].1;
    for (arm, rung_list) in &results {
        let total_nll: f64 = rung_list.iter().map(|r| r.nll_sum).sum();
        let total_t: usize = rung_list.iter().map(|r| r.targets).sum();
        let pooled = (total_nll / total_t.max(1) as f64).exp();
        let worst_delta: f64 = rung_list
            .iter()
            .zip(base.iter())
            .map(|(r, b)| (r.ppl() - b.ppl()) / b.ppl())
            .fold(0.0f64, |a, d| a.max(d.abs()));
        println!(
            "# {:>8} pooled_ppl {:.4} worst |Δppl/ppl_base| {:.4} ({})",
            arm.name(),
            pooled,
            worst_delta,
            match *arm {
                Arm::Base => "reference".to_string(),
                Arm::Si | Arm::SiCal => {
                    if worst_delta <= tolerance {
                        "PASS".to_string()
                    } else {
                        "FAIL".to_string()
                    }
                }
                Arm::SiPprope => format!(
                    "{} (EXPECTED FAIL pre-recorded — G3 (iii))",
                    if worst_delta > tolerance {
                        "FAIL"
                    } else {
                        "unexpected PASS"
                    }
                ),
            }
        );
        for (r, b) in rung_list.iter().zip(base.iter()) {
            println!(
                "#   L={:>5} ppl {:.4} Δ {:+.4} ({} tokens)",
                r.length,
                r.ppl(),
                (r.ppl() - b.ppl()) / b.ppl(),
                r.targets
            );
        }
    }

    // ── the golden-NLL report (Issue 959 T6) ──
    #[cfg(feature = "golden_nll")]
    if golden_nll {
        let (base_arm, rung_list) = &results[0];
        debug_assert_eq!(base_arm, &Arm::Base);
        let total_t: usize = rung_list.iter().map(|r| r.targets).sum();
        let total_inc: f64 = rung_list.iter().map(|r| r.nll_sum).sum();
        let total_golden: f64 = rung_list.iter().map(|r| r.golden_sum).sum();
        let conf_t: usize = rung_list.iter().map(|r| r.confident).sum();
        let conf_golden: f64 = rung_list.iter().map(|r| r.golden_confident_sum).sum();
        let deep_t: usize = rung_list.iter().map(|r| r.deep).sum();
        let deep_golden: f64 = rung_list.iter().map(|r| r.golden_deep_sum).sum();
        let disagrees: usize = rung_list.iter().map(|r| r.disagrees).sum();
        let max_delta: f64 = rung_list
            .iter()
            .map(|r| r.max_abs_delta)
            .fold(0.0f64, f64::max);
        println!(
            "\n# golden-NLL readout (Issue 959 T6, arXiv:2610.31642 Eq. 5) — the tail-faithful \
oracle over the same scored positions; the lossy-quant gates diff THEIR variant \
checkpoints through this readout"
        );
        println!(
            "#   pooled: incumbent f64-lse ppl {:.6} · golden f64-oracle ppl {:.6} · \
Δ {:+.3e} nats/token (mean over {total_t} scored)",
            (total_inc / total_t.max(1) as f64).exp(),
            (total_golden / total_t.max(1) as f64).exp(),
            (total_golden - total_inc) / total_t.max(1) as f64
        );
        println!(
            "#   confident band (margin ≥ 16.64 nats, the fp32 cliff): {conf_t}/{total_t} positions, \
pooled golden ppl {} — the band where an fp32-consumer readout reads false zeros",
            if conf_t > 0 {
                format!("{:.6}", (conf_golden / conf_t as f64).exp())
            } else {
                "n/a (empty band)".to_string()
            }
        );
        println!(
            "#   deep band (margin ≥ 36.73 nats, the f64 cliff): {deep_t}/{total_t} positions, \
pooled golden ppl {} — the band where even an f64 max-subtract LSE absorbs the tail",
            if deep_t > 0 {
                format!("{:.6}", (deep_golden / deep_t as f64).exp())
            } else {
                "n/a (empty band)".to_string()
            }
        );
        println!(
            "#   incumbent-vs-golden: {disagrees}/{total_t} positions disagree AT ALL, \
max |Δnll| {max_delta:.3e} nats — on a healthy model this is f64-composition noise; \
a lossy variant shows its true deltas here"
        );
        for r in rung_list {
            println!(
                "#   L={:>5} golden_ppl {:.4} confident {}/{}, deep {}/{}",
                r.length,
                (r.golden_sum / r.targets.max(1) as f64).exp(),
                r.confident,
                r.targets,
                r.deep,
                r.targets
            );
        }
    }
    #[cfg(not(feature = "golden_nll"))]
    let _ = golden_nll; // the runtime flag already refused above

    // ── the retrieval probe ──
    if retrieval {
        println!("\n# retrieval probe (needle at the head, question at the tail, 6 greedy tokens)");
        const FACTS: [(&str, &str, &str); 2] = [
            (
                "The codeword of the amber lantern is ZEBRA.",
                "zebra",
                "Question: What is the codeword of the amber lantern? Answer:",
            ),
            (
                "The password of the blue gate is RIVER.",
                "river",
                "Question: What is the password of the blue gate? Answer:",
            ),
        ];
        const FILLER: &str =
            " The merchant walked along the road and counted his coins under the afternoon sun.";
        const RETRIEVAL_RUNGS: [usize; 2] = [1024, 2048];
        for &l in &RETRIEVAL_RUNGS {
            let filler_tokens = tok.encode(FILLER);
            for (arm, _) in &results {
                if *arm != Arm::Base && !matches!(arm, Arm::Si | Arm::SiCal | Arm::SiPprope) {
                    continue;
                }
                // Re-arm the session exactly as the ppl loop did.
                match *arm {
                    Arm::Base => rig.set_session(None),
                    Arm::Si => rig.set_session(Some(SiAttentionSession::Armed(Arc::new(
                        SiAttentionPlan::uncalibrated(lut.clone(), config.n_head),
                    )))),
                    Arm::SiCal => {
                        let cal = calibration.as_ref().expect("built").clone();
                        let plan = SiAttentionPlan::calibrated(
                            lut.clone(),
                            cal,
                            config.n_head,
                            weights.layers.len(),
                        )
                        .map_err(anyhow::Error::msg)?;
                        rig.set_session(Some(SiAttentionSession::Armed(Arc::new(plan))));
                    }
                    Arm::SiPprope => {
                        let table = pp_table.as_ref().expect("checked above");
                        rig.rope_freq = table;
                        rig.set_session(Some(SiAttentionSession::Armed(Arc::new(
                            SiAttentionPlan::uncalibrated(lut.clone(), config.n_head),
                        ))));
                    }
                }
                if *arm == Arm::SiPprope && pp_table.is_none() {
                    continue;
                }
                let mut row = format!("#   L={l:>5} {}", arm.name());
                for (fact, want, question) in FACTS {
                    let fact_tokens = tok.encode(fact);
                    let question_tokens = tok.encode(question);
                    let budget = l.saturating_sub(fact_tokens.len() + question_tokens.len() + 1);
                    let k = budget / filler_tokens.len().max(1);
                    let mut prompt = vec![tok.bos_id()];
                    prompt.extend(fact_tokens.iter());
                    for _ in 0..k {
                        prompt.extend(filler_tokens.iter());
                    }
                    prompt.extend(question_tokens.iter());
                    if prompt.len() > config.block_size - 8 {
                        row.push_str("  SKIP(prompt>block)");
                        continue;
                    }
                    rig.cache.reset();
                    let feed_to = prompt.len() - 1;
                    let mut nll = 0.0;
                    let mut targets = 0usize;
                    #[cfg(feature = "golden_nll")]
                    let mut ret_golden = GoldenAccum::new(0);
                    #[cfg(not(feature = "golden_nll"))]
                    let mut ret_golden = GoldenAccum::inert();
                    rig.feed(
                        &prompt,
                        0,
                        feed_to,
                        usize::MAX,
                        &mut x,
                        &mut nll,
                        &mut targets,
                        &mut ret_golden,
                    );
                    let generated = rig.greedy(prompt[feed_to], feed_to, 6, &mut x);
                    let text_out = tok.decode(&generated).to_lowercase();
                    let hit = text_out.contains(want);
                    row.push_str(&format!(
                        "  [{}: {}]",
                        want,
                        if hit { "HIT" } else { "MISS" }
                    ));
                }
                if *arm == Arm::SiPprope {
                    rig.rope_freq = &plain_table;
                }
                println!("{row}");
            }
        }
    }

    println!("\n# verdict law: si / si-cal PASS iff every rung's |Δppl|/ppl_base ≤ {tolerance}");
    println!("# si-pprope: the FAIL is the PRE-RECORDED expectation (Plan 622 G3 (iii)) —");
    println!("#   the quality claim lives in riir-train Plan 449, not in zero-shot pairing.");

    Ok(())
}

/// Tiny width clamp for the rung header row.
trait TruncateWidth {
    fn truncate_to_width(self, w: usize) -> String;
}
impl TruncateWidth for String {
    fn truncate_to_width(self, w: usize) -> String {
        if self.len() <= w {
            self
        } else {
            self.chars().take(w).collect()
        }
    }
}
