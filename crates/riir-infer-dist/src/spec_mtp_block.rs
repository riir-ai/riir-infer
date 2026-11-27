//! Issue 1253 T5.3 (plan 639) — the GGUF-embedded MTP draft block's
//! DRAFTER SEAT over the speculative-fill machinery
//! ([`crate::spec_fill::DraftSource`]).
//!
//! The seat chains [`riir_infer_core::mtp_draft::mtp_draft_step`] (the
//! T5.2 CPU reference, the oracle the reference-keeps-the-job law pins)
//! into a greedy K-token draft:
//!
//! ```text
//! step 0: compose(emb(last_landed), h_final)   @ cursor     → pred_0 (disclosure:
//!          the block's own read of the pending slot; the runner's echo
//!          d_0 == pending free-rides on this slot instead)
//! step j: compose(emb(pred_{j-1}), h_{j-1})    @ cursor + j  → pred_j = d_j
//! ```
//!
//! Each step consumes the evolved `h` in place (the fork's chain
//! semantics — `h_prev` left in the caller's buffer) and advances the
//! block's OWN KV cache at the draft positions. The cache discipline
//! mirrors the fork's: rows [0..next_pos) hold the chains that produced
//! the now-committed prefix; a new round's chain restarts at `next_pos`
//! and overwrites its own rows, and the step's attention reads only
//! `[0..pos + 1]` — no explicit truncation is needed, rejected tails are
//! simply rewritten.
//!
//! ## Postures (the honest-basis law, the BMTH precedent's disclosure
//! shape)
//!
//! The block consumes the post-final-norm PRIMAL hidden. In the 2-stage
//! spec runner that hidden lives TAIL-side (the head stage's rows are
//! mid-pipeline boundary activations), so the runner can never feed it:
//! seated in [`crate::spec_fill::run_head_spec`] the drafter degrades to
//! the LOSSLESS echo and names it through
//! [`MtpDraftBlockDrafter::take_error`] (never a silent wrong-basis
//! draft). The lawful postures are the FULL-STACK ones — the T5.4
//! teacher-forced acceptance harness and the T5.5 Metal serving cell —
//! which call [`MtpDraftBlockDrafter::seed`] per round with the real
//! `h_final` and drive [`MtpDraftBlockDrafter::draft_chain`] (or let the
//! round loop call `propose` on the freshly seeded state).
//!
//! ## Card presets
//!
//! The card presets name the vendor postures the measured cells sweep:
//! NVIDIA = 3 draft tokens with a 0.10 min-prob chain floor; Apple =
//! 1..=8 draft tokens, no floor. The min-prob gate reads the greedy
//! draft's own top-token mass `1/Z` (`Z = Σ exp(l_j − l_max)`, the
//! distribution's normalizer) — a scalar threshold read, never a
//! softmax weighting (the sigmoid-not-softmax law); the first step
//! below the floor ends the chain, its prediction NOT proposed. The
//! gate applies uniformly to every chain step INCLUDING `pred_0` (a
//! floor-flat distribution starves the chain to the echo — the T5.4
//! sweep owns the tuning, this seat only carries the axis).
//!
//! Greedy only (argmax, first-index tie rule) — the acceptance
//! harness's teacher posture; no sampling knob (plan 639 T5.3).
//!
//! Determinism: the chain is a pure function of the seeded state +
//! cache; the f32 mass is a sequential ascending fold (Issue 940); the
//! tie rule is first-index strict `>`.

use std::sync::Arc;

use riir_infer_core::deltanet::forward::effective_rotary_dim;
use riir_infer_core::deltanet::ternary_weights::QwenDeltaNetTernaryWeights;
use riir_infer_core::gguf_loader::MtpDraftBlock;
use riir_infer_core::mtp_draft::{MtpDraftStepScratch, mtp_draft_step};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::transformer::KVCache;
use riir_infer_core::types::Config;

use crate::spec_fill::{DraftCtx, DraftSource};

/// The card domain: the presets sweep 1..=8 draft tokens (the Apple
/// card's ceiling; the NVIDIA card's 3 sits inside).
pub const MAX_DRAFT_K: usize = 8;

/// The K + min-prob pair the measured cells sweep. `k` sits in
/// `1..=MAX_DRAFT_K`; `min_prob` (when armed) sits in `(0.0, 1.0]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MtpDraftPreset {
    pub k: usize,
    pub min_prob: Option<f32>,
}

impl MtpDraftPreset {
    /// The NVIDIA card posture: 3 draft tokens, min-prob 0.10.
    pub fn nvidia_draft3() -> Self {
        Self {
            k: 3,
            min_prob: Some(0.10),
        }
    }

    /// The Apple card posture: `k` draft tokens, no chain floor.
    pub fn apple(k: usize) -> Result<Self, String> {
        Self::new(k, None)
    }

    /// Raw construction with the domain validation both cards share.
    pub fn new(k: usize, min_prob: Option<f32>) -> Result<Self, String> {
        if k == 0 || k > MAX_DRAFT_K {
            return Err(format!("draft k {k} outside 1..={MAX_DRAFT_K}"));
        }
        if let Some(mp) = min_prob
            && !(mp > 0.0 && mp <= 1.0)
        {
            return Err(format!("min_prob {mp} outside (0.0, 1.0]"));
        }
        Ok(Self { k, min_prob })
    }
}

/// The GGUF-embedded draft block, seated (T5.3). Construction is the
/// allocation boundary: the block's KV cache, the step scratch, the
/// logits buffer and the chain's `h` grow ONCE here — the chain steps
/// reuse them (the step's own alloc-free pin lives in the root crate's
/// `mtp_draft_step_g4_alloc`).
pub struct MtpDraftBlockDrafter {
    block: Arc<MtpDraftBlock>,
    weights: Arc<QwenDeltaNetTernaryWeights>,
    config: Config,
    rope: RopeFreqTable,
    cache: KVCache,
    scratch: MtpDraftStepScratch,
    logits: Vec<f32>,
    /// The chain's evolving hidden — PRIMAL (the basis contract; the
    /// step consumes post-final-norm primal and leaves `h_draft` here).
    h: Vec<f32>,
    last_token: u32,
    next_pos: usize,
    /// The runner cursor this seat was seeded for (`Some(next_pos)` at
    /// seed time; `propose` compares it against `committed.len() - 1`).
    seeded_for: Option<usize>,
    preset: MtpDraftPreset,
    error: Option<String>,
}

impl MtpDraftBlockDrafter {
    /// Seat the block against a main stack. Geometry disagreements
    /// refuse at the seat — never mid-chain.
    pub fn new(
        block: Arc<MtpDraftBlock>,
        weights: Arc<QwenDeltaNetTernaryWeights>,
        config: Config,
        preset: MtpDraftPreset,
    ) -> Result<Self, String> {
        if block.block_index != config.n_layer {
            return Err(format!(
                "draft block index {} != config n_layer {} — wrong block for this stack",
                block.block_index, config.n_layer
            ));
        }
        let rotary_dim = effective_rotary_dim(&config);
        let rope = RopeFreqTable::new(config.rope_theta, rotary_dim);
        Ok(Self {
            cache: KVCache::new(&config),
            scratch: MtpDraftStepScratch::new(&config),
            logits: vec![0.0; config.vocab_size],
            h: vec![0.0; config.n_embd],
            block,
            weights,
            config,
            rope,
            last_token: 0,
            next_pos: 0,
            seeded_for: None,
            preset,
            error: None,
        })
    }

    /// Feed one round's full-stack state: the post-final-norm PRIMAL
    /// hidden at the last landed position (`h_final`), that landed
    /// token, and the first DRAFT position (= the runner's cursor, the
    /// position the pending prediction occupies). The T5.4 harness and
    /// the T5.5 serving cell call this per round.
    pub fn seed(
        &mut self,
        h_final_primal: &[f32],
        last_landed_token: u32,
        next_pos: usize,
    ) -> Result<(), String> {
        let n = self.config.n_embd;
        if h_final_primal.len() < n {
            return Err(format!(
                "seed hidden {} < n_embd {n} — refusing",
                h_final_primal.len()
            ));
        }
        self.h[..n].copy_from_slice(&h_final_primal[..n]);
        self.last_token = last_landed_token;
        self.next_pos = next_pos;
        self.seeded_for = Some(next_pos);
        Ok(())
    }

    /// The greedy chain: up to `preset.k` (token, top-token mass) pairs,
    /// gated by `preset.min_prob` (the first step below the floor ends
    /// the chain, its prediction NOT included). Bit-identical across
    /// re-seeds of the same state.
    pub fn draft_chain(&mut self) -> Vec<(u32, f32)> {
        let mut out = Vec::with_capacity(self.preset.k);
        for _ in 0..self.preset.k {
            let res = mtp_draft_step(
                &self.block,
                &self.weights,
                &mut self.cache,
                self.next_pos,
                &mut self.h,
                self.last_token as usize,
                &self.config,
                &self.rope,
                &mut self.scratch,
                &mut self.logits,
            );
            if let Err(e) = res {
                if self.error.is_none() {
                    self.error = Some(format!("mtp draft step: {e}"));
                }
                break;
            }
            let (tok, mass) = top_token_mass(&self.logits[..self.config.vocab_size]);
            if !mass.is_finite() {
                // A non-finite mass (inf/NaN logits class) gates the
                // chain out — never proposed, never compared.
                if self.error.is_none() {
                    self.error = Some("mtp draft chain: non-finite top-token mass".into());
                }
                break;
            }
            if let Some(mp) = self.preset.min_prob
                && mass < mp
            {
                break;
            }
            out.push((tok, mass));
            self.last_token = tok;
            self.next_pos += 1;
        }
        out
    }

    /// The first error the drafter hit (unseeded posture, step refusal,
    /// non-finite logits class), if any. Surfaced by the caller — never
    /// swallowed (the [`crate::spec_mtp::MtpHeadDrafter`] precedent).
    pub fn take_error(&mut self) -> Option<String> {
        self.error.take()
    }

    pub fn preset(&self) -> MtpDraftPreset {
        self.preset
    }

    /// Swap the swept cell (the T5.4 harness re-seats per posture).
    pub fn set_preset(&mut self, preset: MtpDraftPreset) {
        self.preset = preset;
    }

    /// The next draft position (advanced by every chain step).
    pub fn next_pos(&self) -> usize {
        self.next_pos
    }

    /// Fresh-window state (the T5.4 harness's per-window posture): clears
    /// the block's KV cache, the seed cursor and any carried error. The
    /// chain `h` is left as-is — [`Self::seed`] overwrites it before the
    /// next [`Self::draft_chain`]. Without this, a new window's early
    /// chain steps would attend over the previous window's tail rows
    /// (rows below the new seed position are never rewritten by the new
    /// chain — the no-explicit-truncation discipline only covers rows at
    /// or above the cursor).
    pub fn reset(&mut self) {
        self.cache.reset();
        self.seeded_for = None;
        self.error = None;
    }

    /// The runner cursor the seat was seeded for, if any.
    pub fn seeded_for(&self) -> Option<usize> {
        self.seeded_for
    }
}

impl DraftSource for MtpDraftBlockDrafter {
    /// `d_0` = the echo (the accepted-prefix law), `d_j` = chain step
    /// `j`'s prediction. Shorter than `ctx.k` when the min-prob gate (or
    /// a step refusal) starved the chain — a shorter proposal verifies
    /// fewer slots, never a wrong one. Outside the seeded posture the
    /// seat degrades to the lossless echo and names it through
    /// [`MtpDraftBlockDrafter::take_error`] (the postures section).
    fn propose(&mut self, ctx: &DraftCtx<'_>) -> Vec<u32> {
        let pending = ctx.committed.last().copied().unwrap_or(0);
        let cursor = ctx.committed.len().saturating_sub(1);
        if self.seeded_for != Some(cursor) {
            if self.error.is_none() {
                self.error = Some(format!(
                    "mtp drafter not seeded for cursor {cursor} (seeded for {:?}) — the \
                     2-stage spec runner cannot produce the post-final-norm hidden the \
                     block consumes; degrading to the lossless echo",
                    self.seeded_for
                ));
            }
            return vec![pending; ctx.k];
        }
        let chain = self.draft_chain();
        let mut out = Vec::with_capacity(ctx.k);
        out.push(pending);
        for (tok, _) in chain.into_iter().skip(1).take(ctx.k.saturating_sub(1)) {
            out.push(tok);
        }
        out.shrink_to_fit();
        out
    }
}

/// The greedy draft's read: the argmax token (first-index tie rule) and
/// its top-token mass `1/Z`, `Z = Σ_j exp(l_j − l_max)` — the
/// distribution's normalizer, WITHOUT materializing a softmax (the
/// sigmoid-not-softmax law: a scalar threshold read gating chain
/// continuation, never a weighting). Sequential ascending fold (Issue
/// 940). `Z ≥ 1` by construction (the max term contributes exp(0) = 1),
/// so the mass sits in `(0, 1]`.
fn top_token_mass(logits: &[f32]) -> (u32, f32) {
    debug_assert!(!logits.is_empty(), "top_token_mass: empty logits");
    let mut best = 0usize;
    for (i, &l) in logits.iter().enumerate() {
        if l > logits[best] {
            best = i;
        }
    }
    let lmax = logits[best];
    let mut z = 0.0f32;
    for &l in logits {
        z += (l - lmax).exp();
    }
    (best as u32, 1.0 / z)
}
