//! The Dory CPU forward (riir-infer Issue 1221 T1b).
//!
//! Recurrence (paper §3.2): `s₀ = x; u_t = N(s_{t-1} + x); s_t = N(F_θ(u_t))`
//! for `t = 1..=k` — the input block's output `x` is re-injected EVERY
//! loop, with normalization at both boundaries and no learned adapter.
//! Each sublayer inside `F` follows the nGPT blend rule (fig 2e):
//! `s' = N((1−a)⊙u + a⊙N(f(u)))` with `u = N(s)` and `a = |α̂|`.
//!
//! Cache policy (T1 golden + T2 serving posture): **per-loop** is the
//! training semantics — loop `t` of position `p` attends loop-`t`
//! keys/values of positions `≤ p`; every recurrent attention block
//! carries `k` cache slots, the input/output blocks one. **Last-loop**
//! ([`DoryCachePolicy::LastLoop`], the `CacheStrategy::Last` serving
//! shape) is the 1/k-memory posture: ONE slot per recurrent block; the
//! in-flight positions' rows are REWRITTEN by every pass (the last
//! write is the canonical final-loop state), and a decode step's query
//! attends the cached prefix's final-loop entries plus its own
//! current-loop row (paper §6.1). Two laws pin the posture:
//! a from-scratch prefill is BIT-IDENTICAL under both policies at every
//! k (within a pass, every position is in-flight and attends its own
//! pass's states — the cache policy changes only what is RETAINED),
//! while the incremental decode diverges from per-loop at k ≥ 2 —
//! that divergence is the policy's approximation, REPORTED (the
//! k=4 delta cell), never assumed zero.
//!
//! The serving loop count lives on the CACHE (`DoryKvCache::new(w, k)`
//! — the request posture) while [`DoryWeights::meta`] carries the
//! checkpoint's `num_loops` default (the k-scheduler law: read from the
//! header, never hardcoded; the elastic override is a cache built at a
//! different k).
//!
//! Bit-identity law (the T1 proof): the batched prefill and the
//! incremental decode share every per-row primitive — projections loop
//! over activation rows calling the same `matvec_row`, attention reads
//! the cache in both postures, and every norm/softmax/LM-head fold is a
//! fixed sequential f32 reduction. Same weights + same k + same cache
//! policy ⇒ bit-identical logits, tested at k=1 and k=2.
//!
//! Numerics pins (the determinism law): sequential folds everywhere (no
//! platform-lane reassociation), `N(v) = v / √(mean(v²) + ε)` in DIVISION
//! form (nGPT Eq. 5 — the radius-√d hypersphere the paper calls "L2
//! norm"), softmax with a max-subtract then a sequential exp-sum, and
//! `logit_v = (z·W_out[v])·output_scale[v] / σ₀` with σ₀ from
//! [`super::LOGIT_SIGMA_ZERO`].
//!
//! Named reading decisions (no published oracle exists — the paper's
//! figures are the authority; each is documented where it bites):
//! - `ŝ_qk²` scales Q only (the artifact's qk vector is q-width; no
//!   k-width scale tensor exists in the T0 census).
//! - The rope pairing is the house rotate-half convention
//!   ([`crate::rope`]) over ALL head dims (`dory.rope.dimension_count ==
//!   head_dim` is enforced at load). The community GGUF targets an
//!   UNMERGED llama.cpp PR whose converter may permute Q/K rows — if the
//!   T2/T3 directional cells or the smoke generation look incoherent,
//!   the row order is the first suspect, and the fix is a load-time
//!   `unpermute_interleaved_rows` pass, never a forward change.
//! - The gate reads `attn ⊙ σ(g ⊙ ŝ_g²)` (scale, then sigmoid — fig 2c
//!   order), with `g` the gate projection's own output row.

use super::{DoryBlockKind, DoryWeights, LOGIT_SIGMA_ZERO, NORM_EPSILON};
use crate::rope::{RopeFreqTable, apply_rope_with_freq};
use std::fmt;

/// The forward's loud refusals (bad input token ids — the only data-
/// dependent failure; cache misuse is a programmer error and asserts).
#[derive(Debug, PartialEq)]
pub enum DoryForwardError {
    TokenOutOfRange { token: u32, vocab: usize },
}

impl fmt::Display for DoryForwardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TokenOutOfRange { token, vocab } => {
                write!(f, "token id {token} out of range (vocab {vocab})")
            }
        }
    }
}

impl std::error::Error for DoryForwardError {}

/// The KV retention policy (T2): which loop's states the cache keeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DoryCachePolicy {
    /// The training semantics (the T1 golden): loop `t` keeps its own
    /// slot — `k` slots per recurrent attention block.
    #[default]
    PerLoop,
    /// The serving posture (`CacheStrategy::Last` on the loop hosts):
    /// ONE slot per recurrent block, rewritten by every pass — the
    /// canonical rows are the final loop's. 1/k the KV memory; the
    /// incremental decode's prefix reads are the reported approximation.
    LastLoop,
}

/// One attention block's per-loop KV rows. Every slot holds the same
/// `[len][kv_dim]` table — loop `t` reads/writes slot `t` (per-loop) or
/// slot 0, overwritten in place (last-loop).
struct BlockCache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

/// KV cache over [`DoryCachePolicy`]. Built against specific
/// weights at a specific loop count; `prefill` fills it, `decode_step`
/// appends to it.
pub struct DoryKvCache {
    k_loops: usize,
    policy: DoryCachePolicy,
    len: usize,
    /// One entry per ATTENTION physical block, ordinal = physical / 2
    /// (attention sits on even indices — the T0 parity law).
    blocks: Vec<BlockCache>,
}

impl BlockCache {
    /// Write one row of K/V at `pos` — append when the row is genuinely
    /// new (slot len == the row's start offset), replace in place when it
    /// already exists (the last-loop pass rewrite: the in-flight
    /// position's row is rewritten by every recurrent pass, the prefix's
    /// rows never touched). The no-hole invariant (`slot.len() == start`
    /// or `>= end`) is the programmer-error guard for BOTH policies.
    fn write_row(&mut self, slot: usize, pos: usize, kr: &[f32], vr: &[f32]) {
        let (start, end) = (pos * kr.len(), pos * kr.len() + kr.len());
        if self.k[slot].len() == start {
            self.k[slot].extend_from_slice(kr);
            self.v[slot].extend_from_slice(vr);
        } else {
            assert!(
                self.k[slot].len() >= end,
                "cache row hole at pos {pos} (k len {} < {end})",
                self.k[slot].len()
            );
            self.k[slot][start..end].copy_from_slice(kr);
            self.v[slot][start..end].copy_from_slice(vr);
        }
    }
}

impl DoryKvCache {
    /// A fresh empty PER-LOOP cache for `weights` at loop count `k`
    /// (the T1 golden default; the serving posture is
    /// [`Self::with_policy`]).
    pub fn new(weights: &DoryWeights, k: usize) -> Self {
        Self::with_policy(weights, k, DoryCachePolicy::PerLoop)
    }

    /// A fresh empty cache for `weights` at loop count `k` under
    /// `policy` — the elastic-k + cache-policy serving surface (both
    /// live on the request, never the weights).
    pub fn with_policy(weights: &DoryWeights, k: usize, policy: DoryCachePolicy) -> Self {
        assert!(k >= 1, "loop count k must be >= 1");
        let mut blocks = Vec::new();
        for (phys, b) in weights.blocks.iter().enumerate() {
            if b.kind == DoryBlockKind::Attn {
                let in_recurrent =
                    weights.meta.recurrent_start() <= phys && phys < weights.meta.output_start();
                let slots = if in_recurrent {
                    match policy {
                        DoryCachePolicy::PerLoop => k,
                        DoryCachePolicy::LastLoop => 1,
                    }
                } else {
                    1
                };
                blocks.push(BlockCache {
                    k: vec![Vec::new(); slots],
                    v: vec![Vec::new(); slots],
                });
            }
        }
        Self {
            k_loops: k,
            policy,
            len: 0,
            blocks,
        }
    }

    /// Cached positions (every slot holds the same length).
    pub fn len(&self) -> usize {
        self.len
    }

    /// True when empty (the `len() == 0` clippy form).
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The loop count this cache was built for.
    pub fn k_loops(&self) -> usize {
        self.k_loops
    }

    /// The retention policy this cache was built with.
    pub fn policy(&self) -> DoryCachePolicy {
        self.policy
    }

    /// Total stored K+V elements across every block and slot. The
    /// per-loop footprint is k-proportional; the last-loop footprint is
    /// k-INDEPENDENT (one slot per recurrent block — the byte-count
    /// asserts read this counter).
    pub fn kv_elements(&self) -> usize {
        self.blocks
            .iter()
            .map(|b| b.k.iter().chain(b.v.iter()).map(Vec::len).sum::<usize>())
            .sum()
    }

    /// The KV slot a recurrent pass writes: its own loop index
    /// (per-loop) or the shared slot 0 (last-loop).
    fn recurrent_slot(&self, pass: usize) -> usize {
        match self.policy {
            DoryCachePolicy::PerLoop => pass,
            DoryCachePolicy::LastLoop => 0,
        }
    }

    fn clear(&mut self) {
        for b in &mut self.blocks {
            for slot in b.k.iter_mut().chain(b.v.iter_mut()) {
                slot.clear();
            }
        }
        self.len = 0;
    }

    fn ordinal(physical: usize) -> usize {
        physical / 2
    }
}

/// The Dory CPU forward over borrowed weights.
pub struct DoryForward<'w> {
    w: &'w DoryWeights,
    /// Full-attention rope table (`rope.freq_base`, 1e7 in the census).
    freq_full: RopeFreqTable,
    /// Sliding-window rope table (`rope.freq_base_2`, 1e4 in the census).
    freq_swa: RopeFreqTable,
}

/// One block-range pass context: the row count, the KV slot the pass
/// writes, the first row's absolute position, and the cache. Grouped
/// because every block pass shares them.
struct PassCtx<'c> {
    t: usize,
    slot: usize,
    pos_base: usize,
    cache: &'c mut DoryKvCache,
}

impl<'c> PassCtx<'c> {
    fn new(t: usize, slot: usize, pos_base: usize, cache: &'c mut DoryKvCache) -> Self {
        Self {
            t,
            slot,
            pos_base,
            cache,
        }
    }
}

impl<'w> DoryForward<'w> {
    pub fn new(weights: &'w DoryWeights) -> Self {
        let d = &weights.dims;
        assert!(
            d.n_head.is_multiple_of(d.n_kv_head),
            "GQA requires n_kv_head | n_head ({} % {} != 0)",
            d.n_head,
            d.n_kv_head
        );
        Self {
            w: weights,
            freq_full: RopeFreqTable::new(weights.rope_freq_base, d.head_dim),
            freq_swa: RopeFreqTable::new(weights.rope_freq_base_2, d.head_dim),
        }
    }

    /// The loop count the CHECKPOINT defaults to — the header's
    /// `num_loops` (never hardcoded). The serving posture picks k at
    /// [`DoryKvCache::new`].
    pub fn k_loops(&self) -> usize {
        self.w.meta.num_loops
    }

    /// Full-sequence prefill: the whole pipeline batched, sharing every
    /// per-row primitive with [`Self::decode_step`], so the two postures
    /// are bit-identical by construction (the T1 golden tests pin it).
    /// Returns the logits row for EVERY position (`[seq][vocab]`).
    /// Clears `cache` first (a fresh-context posture; chunked prefill is
    /// T2+).
    pub fn prefill(
        &self,
        tokens: &[u32],
        cache: &mut DoryKvCache,
    ) -> Result<Vec<Vec<f32>>, DoryForwardError> {
        cache.clear();
        let d = &self.w.dims;
        let t = tokens.len();
        // Embed: exact f32 row gathers ([vocab][n_embd] file order).
        let mut s = vec![0.0f32; t * d.n_embd];
        for (i, &tok) in tokens.iter().enumerate() {
            self.embed_into(tok, &mut s[i * d.n_embd..(i + 1) * d.n_embd])?;
        }

        // Input block: one pass, slot 0.
        let in_end = self.w.meta.recurrent_start();
        self.run_block_range(0, in_end, &mut s, &mut PassCtx::new(t, 0, 0, cache));
        let x = s.clone();

        // Recurrent block: k passes with input re-injection + boundary
        // norms (the paper's loop-boundary transforms, no adapter).
        let r0 = in_end;
        let r1 = r0 + self.w.meta.recurrent_layers;
        let k = cache.k_loops;
        let mut u = vec![0.0f32; t * d.n_embd];
        let mut s_next = vec![0.0f32; t * d.n_embd];
        for lp in 0..k {
            add_rows(&mut u, &s, &x);
            norm_rows(&mut u, t, d.n_embd);
            s_next.copy_from_slice(&u);
            let slot = cache.recurrent_slot(lp);
            self.run_block_range(r0, r1, &mut s_next, &mut PassCtx::new(t, slot, 0, cache));
            norm_rows(&mut s_next, t, d.n_embd);
            std::mem::swap(&mut s, &mut s_next);
        }

        // Output block: one pass, slot 0.
        self.run_block_range(
            r1,
            self.w.blocks.len(),
            &mut s,
            &mut PassCtx::new(t, 0, 0, cache),
        );
        cache.len = t;

        // LM head per position.
        let mut logits = Vec::with_capacity(t);
        for i in 0..t {
            logits.push(self.logits_row(&s[i * d.n_embd..(i + 1) * d.n_embd]));
        }
        Ok(logits)
    }

    /// One incremental decode step at `pos = cache.len()`. Appends the
    /// token's keys/values to every block's slots and returns the
    /// position's logits — bit-identical to the same token's row in a
    /// prefill over the same prefix (the T1 identity).
    pub fn decode_step(
        &self,
        token: u32,
        cache: &mut DoryKvCache,
    ) -> Result<Vec<f32>, DoryForwardError> {
        let d = &self.w.dims;
        let t = 1;
        let pos = cache.len;

        let mut s = vec![0.0f32; d.n_embd];
        self.embed_into(token, &mut s)?;

        let in_end = self.w.meta.recurrent_start();
        self.run_block_range(0, in_end, &mut s, &mut PassCtx::new(t, 0, pos, cache));
        let x = s.clone();

        let r0 = in_end;
        let r1 = r0 + self.w.meta.recurrent_layers;
        let k = cache.k_loops;
        let mut u = vec![0.0f32; d.n_embd];
        let mut s_next = vec![0.0f32; d.n_embd];
        for lp in 0..k {
            add_rows(&mut u, &s, &x);
            norm_rows(&mut u, t, d.n_embd);
            s_next.copy_from_slice(&u);
            let slot = cache.recurrent_slot(lp);
            self.run_block_range(r0, r1, &mut s_next, &mut PassCtx::new(t, slot, pos, cache));
            norm_rows(&mut s_next, t, d.n_embd);
            std::mem::swap(&mut s, &mut s_next);
        }

        self.run_block_range(
            r1,
            self.w.blocks.len(),
            &mut s,
            &mut PassCtx::new(t, 0, pos, cache),
        );
        cache.len += 1;
        Ok(self.logits_row(&s))
    }

    fn embed_into(&self, token: u32, out: &mut [f32]) -> Result<(), DoryForwardError> {
        let d = &self.w.dims;
        if token as usize >= d.vocab {
            return Err(DoryForwardError::TokenOutOfRange {
                token,
                vocab: d.vocab,
            });
        }
        let src = token as usize * d.n_embd;
        out.copy_from_slice(&self.w.wte[src..src + d.n_embd]);
        Ok(())
    }

    /// Run physical blocks `[from, to)` over `ctx.t` rows (`s:
    /// [t][n_embd]`, in place), writing KV into `ctx.slot` at positions
    /// `ctx.pos_base..`.
    fn run_block_range(&self, from: usize, to: usize, s: &mut [f32], ctx: &mut PassCtx<'_>) {
        let (t, n_embd) = (ctx.t, self.w.dims.n_embd);
        for (offset, b) in self.w.blocks[from..to].iter().enumerate() {
            let phys = from + offset;
            // u = N(s) — the block's normalized input (fig 2e, norm 1).
            let mut u = s.to_vec();
            norm_rows(&mut u, t, n_embd);
            let mut o = vec![0.0f32; t * n_embd];
            match b.kind {
                DoryBlockKind::Attn => {
                    self.attn_block(b, &u, &mut o, ctx, phys);
                }
                DoryBlockKind::Ffn => self.ffn_block(b, &u, &mut o, t),
            }
            // Blend + closing norm (fig 2e): s' = N((1−a)⊙u + a⊙N(o)).
            blend_norm_rows(s, &u, &o, &b.residual_alpha, t, n_embd);
        }
    }

    /// The attention sublayer (fig 2c): projections from the normalized
    /// input (Q/K/V and the gate's OWN projection) → rope at the
    /// absolute position → per-head norm → `ŝ_qk²` on Q → causal GQA
    /// scored `d_h^{-1/2}` over the cache → sigmoid gate → all-heads
    /// norm → output projection.
    fn attn_block(
        &self,
        b: &super::DoryBlockWeights,
        u: &[f32],
        o: &mut [f32],
        ctx: &mut PassCtx<'_>,
        phys: usize,
    ) {
        let (t, slot, pos_base) = (ctx.t, ctx.slot, ctx.pos_base);
        let d = &self.w.dims;
        let (n_embd, n_head, n_kv_head, head_dim) = (d.n_embd, d.n_head, d.n_kv_head, d.head_dim);
        let q_dim = n_head * head_dim;
        let kv_dim = n_kv_head * head_dim;
        let heads_per_kv = n_head / n_kv_head;
        let score_scale = 1.0 / (head_dim as f32).sqrt();

        let mut q = vec![0.0f32; t * q_dim];
        let mut k = vec![0.0f32; t * kv_dim];
        let mut v = vec![0.0f32; t * kv_dim];
        let mut g = vec![0.0f32; t * q_dim];
        for i in 0..t {
            let u_row = &u[i * n_embd..(i + 1) * n_embd];
            matvec_row(
                &b.attn_q,
                q_dim,
                n_embd,
                u_row,
                &mut q[i * q_dim..(i + 1) * q_dim],
            );
            matvec_row(
                &b.attn_k,
                kv_dim,
                n_embd,
                u_row,
                &mut k[i * kv_dim..(i + 1) * kv_dim],
            );
            matvec_row(
                &b.attn_v,
                kv_dim,
                n_embd,
                u_row,
                &mut v[i * kv_dim..(i + 1) * kv_dim],
            );
            matvec_row(
                &b.attn_gate,
                q_dim,
                n_embd,
                u_row,
                &mut g[i * q_dim..(i + 1) * q_dim],
            );
        }

        // RoPE + per-head norm; Q additionally takes ŝ_qk². The CACHE
        // stores the final k̃ and the raw v — both postures then read the
        // cache, so attention sees identical values by construction.
        let table = if b.rope_primary {
            &self.freq_full
        } else {
            &self.freq_swa
        };
        let bc = &mut ctx.cache.blocks[DoryKvCache::ordinal(phys)];
        for i in 0..t {
            let pos = pos_base + i;
            let qr = &mut q[i * q_dim..(i + 1) * q_dim];
            let kr = &mut k[i * kv_dim..(i + 1) * kv_dim];
            apply_rope_with_freq(qr, kr, pos, head_dim, table.as_slice());
            norm_heads(qr, head_dim);
            norm_heads(kr, head_dim);
            for (qc, &sc) in qr.iter_mut().zip(b.attn_qk_scale.iter()) {
                *qc *= sc * sc;
            }
            bc.write_row(slot, pos, kr, &v[i * kv_dim..(i + 1) * kv_dim]);
        }

        // Attention + gate + all-heads norm + output projection, per row.
        let mut scores = vec![0.0f32; pos_base + t];
        let mut attn_out = vec![0.0f32; q_dim];
        let mut gated = vec![0.0f32; q_dim];
        let mut normed = vec![0.0f32; q_dim];
        let bc = &ctx.cache.blocks[DoryKvCache::ordinal(phys)];
        for i in 0..t {
            let pos = pos_base + i;
            let qr = &q[i * q_dim..(i + 1) * q_dim];
            let gr = &g[i * q_dim..(i + 1) * q_dim];
            let j_start = if b.rope_primary {
                0
            } else {
                pos.saturating_sub(self.w.dims.sliding_window - 1)
            };
            attn_out.fill(0.0);
            for hq in 0..n_head {
                let grp = hq / heads_per_kv;
                let qh = &qr[hq * head_dim..(hq + 1) * head_dim];
                let mut n = 0usize;
                for j in j_start..=pos {
                    let krow = &bc.k[slot][j * kv_dim + grp * head_dim..][..head_dim];
                    scores[n] = dot(qh, krow) * score_scale;
                    n += 1;
                }
                let p = &mut scores[..n];
                softmax(p);
                let oh = &mut attn_out[hq * head_dim..(hq + 1) * head_dim];
                for (c, &pj) in p.iter().enumerate() {
                    let j = j_start + c;
                    let vrow = &bc.v[slot][j * kv_dim + grp * head_dim..][..head_dim];
                    for dd in 0..head_dim {
                        oh[dd] += pj * vrow[dd];
                    }
                }
            }
            // Output gate: attn ⊙ σ(g ⊙ ŝ_g²) — scale, sigmoid, multiply.
            for c in 0..q_dim {
                let sg = b.attn_gate_scale[c];
                let act = gr[c] * sg * sg;
                gated[c] = attn_out[c] / (1.0 + (-act).exp());
            }
            norm_row_into(&mut normed, &gated);
            matvec_row(
                &b.attn_output,
                n_embd,
                q_dim,
                &normed,
                &mut o[i * n_embd..(i + 1) * n_embd],
            );
        }
    }

    /// The SwiGLU mixer (fig 2d): scaled gate/up branches → silu⊙ →
    /// MLP-channel norm → down projection.
    fn ffn_block(&self, b: &super::DoryBlockWeights, u: &[f32], o: &mut [f32], t: usize) {
        let n_embd = self.w.dims.n_embd;
        let mlp = self.w.dims.mlp_hidden;
        let mut h = vec![0.0f32; mlp];
        let mut normed = vec![0.0f32; mlp];
        for i in 0..t {
            let u_row = &u[i * n_embd..(i + 1) * n_embd];
            let mut gproj = vec![0.0f32; mlp];
            let mut uproj = vec![0.0f32; mlp];
            matvec_row(&b.ffn_gate, mlp, n_embd, u_row, &mut gproj);
            matvec_row(&b.ffn_up, mlp, n_embd, u_row, &mut uproj);
            for c in 0..mlp {
                let gs = b.ffn_gate_scale[c];
                let us = b.ffn_up_scale[c];
                let gated = gproj[c] * gs * gs;
                let uped = uproj[c] * us * us;
                h[c] = (gated / (1.0 + (-gated).exp())) * uped;
            }
            norm_row_into(&mut normed, &h);
            matvec_row(
                &b.ffn_down,
                n_embd,
                mlp,
                &normed,
                &mut o[i * n_embd..(i + 1) * n_embd],
            );
        }
    }

    /// `logit_v = (z·W_out[v])·output_scale[v] / σ₀` — one sequential
    /// dot per vocab row (the fixed reduction order; the /σ₀ division is
    /// the paper's fig-2a logit scale).
    fn logits_row(&self, z: &[f32]) -> Vec<f32> {
        let vocab = self.w.dims.vocab;
        let n_embd = self.w.dims.n_embd;
        let mut out = Vec::with_capacity(vocab);
        for v in 0..vocab {
            let w = &self.w.lm_head[v * n_embd..(v + 1) * n_embd];
            out.push(dot(w, z) * self.w.output_scale[v] / LOGIT_SIGMA_ZERO);
        }
        out
    }
}

// ── shared per-row primitives (the bit-identity contract) ───────────

/// Row-wise add: `out = a + b`.
fn add_rows(out: &mut [f32], a: &[f32], b: &[f32]) {
    for (o, (x, y)) in out.iter_mut().zip(a.iter().zip(b.iter())) {
        *o = x + y;
    }
}

/// Norm every `width` row of an in-place `[rows][width]` buffer.
fn norm_rows(buf: &mut [f32], rows: usize, width: usize) {
    for r in 0..rows {
        let s = r * width;
        let row = &mut buf[s..s + width];
        let r = rms_denom(row);
        for x in row.iter_mut() {
            *x /= r;
        }
    }
}

/// `N(v) = v / √(mean(v²) + ε)` into `out` — the nGPT normalize
/// (arXiv:2410.01131 Eq. 5), sequential square-sum fold, DIVISION form
/// (the pinned rounding).
fn norm_row_into(out: &mut [f32], v: &[f32]) {
    let r = rms_denom(v);
    for (o, &x) in out.iter_mut().zip(v.iter()) {
        *o = x / r;
    }
}

/// The normalize denominator: `√(mean(v²) + ε)` — sequential f32 fold.
fn rms_denom(v: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for &x in v {
        acc += x * x;
    }
    (acc / v.len() as f32 + NORM_EPSILON).sqrt()
}

/// Blend + closing norm over a batch (fig 2e):
/// `out_row = N((1−a)⊙u_row + a⊙N(o_row))` with `a = |α̂|` per channel.
fn blend_norm_rows(
    out: &mut [f32],
    u: &[f32],
    o: &[f32],
    alpha: &[f32],
    rows: usize,
    width: usize,
) {
    let mut on = vec![0.0f32; width];
    let mut mix = vec![0.0f32; width];
    for i in 0..rows {
        let s = i * width;
        norm_row_into(&mut on, &o[s..s + width]);
        for c in 0..width {
            let a = alpha[c].abs();
            mix[c] = (1.0 - a) * u[s + c] + a * on[c];
        }
        norm_row_into(&mut out[s..s + width], &mix);
    }
}

/// Sequential dot — THE fixed-order reduction every score and logit
/// uses.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

/// `out = W·x` for one activation row; `W` is `[rows][cols]` row-major
/// (the GGUF ne[0] = in_features law). Every batched projection calls
/// this per row, which is what makes the batched prefill and the
/// incremental decode bit-identical.
fn matvec_row(w: &[f32], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) {
    for r in 0..rows {
        out[r] = dot(&w[r * cols..(r + 1) * cols], x);
    }
}

/// Per-head RMS norm over every `head_dim` slice of a concat-heads row.
fn norm_heads(row: &mut [f32], head_dim: usize) {
    for h in 0..row.len() / head_dim {
        let s = h * head_dim;
        let head = &mut row[s..s + head_dim];
        let r = rms_denom(head);
        for x in head.iter_mut() {
            *x /= r;
        }
    }
}

/// Max-subtract softmax with a sequential exp-sum — fixed order. The
/// max element contributes exp(0) = 1, so the sum is never zero.
fn softmax(p: &mut [f32]) {
    let mut max = f32::NEG_INFINITY;
    for &x in p.iter() {
        if x > max {
            max = x;
        }
    }
    let mut sum = 0.0f32;
    for x in p.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    for x in p.iter_mut() {
        *x /= sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dory::{DoryBlockWeights, DoryDims, DoryLoopMeta};

    /// A deterministic pseudo-random f32 stream (LCG — no unseeded
    /// global RNG; the global-RNG gate class). Values in [-mag, mag).
    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self, mag: f32) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let unit = ((self.0 >> 33) as f32) / ((1u64 << 31) as f32) - 1.0;
            unit * mag
        }
    }

    const VOCAB: usize = 13;
    const N_EMBD: usize = 16;
    const N_HEAD: usize = 4;
    const N_KV: usize = 2;
    const HEAD_DIM: usize = 4;
    const MLP: usize = 8;
    const WINDOW: usize = 3;
    /// 6 physical blocks = 3 pairs: input pair, recurrent pair, output
    /// pair; attention on even (b0 full, b2/b4 SWA), FFN on odd.
    const N_BLOCKS: usize = 6;
    const K: usize = 2;
    const SEQ: usize = 5;

    fn rand_vec(rng: &mut Lcg, n: usize, mag: f32) -> Vec<f32> {
        (0..n).map(|_| rng.next_f32(mag)).collect()
    }

    fn synth_block(rng: &mut Lcg, kind: DoryBlockKind, rope_primary: bool) -> DoryBlockWeights {
        DoryBlockWeights {
            kind,
            // 0.05..0.35 — meaningful blends, not identity.
            residual_alpha: (0..N_EMBD)
                .map(|_| 0.05 + 0.30 * rng.next_f32(1.0).abs())
                .collect(),
            attn_q: rand_vec(rng, N_HEAD * HEAD_DIM * N_EMBD, 0.3),
            attn_k: rand_vec(rng, N_KV * HEAD_DIM * N_EMBD, 0.3),
            attn_v: rand_vec(rng, N_KV * HEAD_DIM * N_EMBD, 0.3),
            attn_output: rand_vec(rng, N_EMBD * N_HEAD * HEAD_DIM, 0.3),
            attn_gate: rand_vec(rng, N_HEAD * HEAD_DIM * N_EMBD, 0.3),
            attn_gate_scale: rand_vec(rng, N_HEAD * HEAD_DIM, 0.6),
            attn_qk_scale: rand_vec(rng, N_HEAD * HEAD_DIM, 0.6),
            ffn_gate: rand_vec(rng, MLP * N_EMBD, 0.3),
            ffn_up: rand_vec(rng, MLP * N_EMBD, 0.3),
            ffn_down: rand_vec(rng, N_EMBD * MLP, 0.3),
            ffn_gate_scale: rand_vec(rng, MLP, 0.6),
            ffn_up_scale: rand_vec(rng, MLP, 0.6),
            rope_primary,
            rope_profile: if rope_primary { 2 } else { 1 },
        }
    }

    fn synth_weights(seed: u64, sliding_window: usize) -> DoryWeights {
        let mut rng = Lcg(seed);
        let blocks = vec![
            synth_block(&mut rng, DoryBlockKind::Attn, true),
            synth_block(&mut rng, DoryBlockKind::Ffn, false),
            synth_block(&mut rng, DoryBlockKind::Attn, false),
            synth_block(&mut rng, DoryBlockKind::Ffn, false),
            synth_block(&mut rng, DoryBlockKind::Attn, false),
            synth_block(&mut rng, DoryBlockKind::Ffn, false),
        ];
        DoryWeights {
            meta: DoryLoopMeta {
                num_loops: K,
                input_layers: 2,
                recurrent_layers: 2,
                output_layers: 2,
                kv_cache_mode: "per_loop".into(),
            },
            wte: rand_vec(&mut rng, VOCAB * N_EMBD, 0.5),
            lm_head: rand_vec(&mut rng, VOCAB * N_EMBD, 0.5),
            output_scale: (0..VOCAB)
                .map(|_| 0.5 + 0.5 * rng.next_f32(1.0).abs())
                .collect(),
            blocks,
            sliding_window,
            rope_freq_base: 1e4,
            rope_freq_base_2: 1e2,
            dims: DoryDims {
                vocab: VOCAB,
                n_embd: N_EMBD,
                n_head: N_HEAD,
                n_kv_head: N_KV,
                head_dim: HEAD_DIM,
                mlp_hidden: MLP,
                n_blocks: N_BLOCKS,
                sliding_window,
                norm_eps: NORM_EPSILON,
            },
        }
    }

    fn tokens() -> Vec<u32> {
        (0..SEQ as u32).map(|i| i * 3 % VOCAB as u32).collect()
    }

    fn prefill_logits(w: &DoryWeights, toks: &[u32], k: usize) -> Vec<Vec<f32>> {
        let f = DoryForward::new(w);
        let mut cache = DoryKvCache::new(w, k);
        f.prefill(toks, &mut cache).expect("prefill")
    }

    fn incremental_logits(w: &DoryWeights, toks: &[u32], k: usize) -> Vec<Vec<f32>> {
        let f = DoryForward::new(w);
        let mut cache = DoryKvCache::new(w, k);
        let mut out = Vec::with_capacity(toks.len());
        for &tok in toks {
            out.push(f.decode_step(tok, &mut cache).expect("decode"));
        }
        out
    }

    fn assert_bit_eq(a: &[Vec<f32>], b: &[Vec<f32>], what: &str) {
        assert_eq!(a.len(), b.len(), "{what}: row count");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(x.len(), y.len(), "{what}: row {i} width");
            for (j, (p, q)) in x.iter().zip(y.iter()).enumerate() {
                assert_eq!(
                    p.to_bits(),
                    q.to_bits(),
                    "{what}: logits[{i}][{j}] diverged ({p} vs {q})"
                );
            }
        }
    }

    /// T1's named golden: the full-sequence recompute (batched prefill,
    /// per-loop caches) vs incremental decode — BIT-identical at k=1.
    #[test]
    fn k1_identity_prefill_equals_incremental() {
        let w = synth_weights(7, WINDOW);
        let toks = tokens();
        assert_bit_eq(
            &prefill_logits(&w, &toks, 1),
            &incremental_logits(&w, &toks, 1),
            "k=1 identity",
        );
    }

    /// The same golden at k=2 over per-loop caches (loop t attends
    /// loop-t states in BOTH postures).
    #[test]
    fn k2_identity_per_loop_caches() {
        let w = synth_weights(7, WINDOW);
        let toks = tokens();
        assert_bit_eq(
            &prefill_logits(&w, &toks, K),
            &incremental_logits(&w, &toks, K),
            "k=2 identity",
        );
    }

    /// The determinism fixture: same weights + same k + same policy ⇒
    /// bit-identical logits across runs, both postures.
    #[test]
    fn deterministic_two_passes_bit_identical() {
        let w = synth_weights(11, WINDOW);
        let toks = tokens();
        assert_bit_eq(
            &prefill_logits(&w, &toks, K),
            &prefill_logits(&w, &toks, K),
            "prefill determinism",
        );
        assert_bit_eq(
            &incremental_logits(&w, &toks, K),
            &incremental_logits(&w, &toks, K),
            "decode determinism",
        );
    }

    /// The loop actually deepens: k=2 is not a k=1 replay.
    #[test]
    fn deeper_loops_change_logits() {
        let w = synth_weights(3, WINDOW);
        let toks = tokens();
        let k1 = prefill_logits(&w, &toks, 1);
        let k2 = prefill_logits(&w, &toks, K);
        let differing = k1
            .iter()
            .zip(k2.iter())
            .flat_map(|(a, b)| a.iter().zip(b.iter()))
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        assert!(
            differing > 0,
            "k=2 logits are bit-identical to k=1 — the recurrent block is not deepening"
        );
    }

    /// The SWA window actually gates: with window 3, positions ≤ 2 see
    /// the same prefix in the windowed and full postures (bit-equal),
    /// positions 3+ attend different prefixes (at least one element
    /// differs).
    #[test]
    fn swa_window_gates_attention() {
        let w_swa = synth_weights(5, WINDOW);
        let w_full = synth_weights(5, usize::MAX);
        let toks = tokens();
        let l_swa = prefill_logits(&w_swa, &toks, 1);
        let l_full = prefill_logits(&w_full, &toks, 1);
        for i in 0..=2 {
            assert_eq!(
                l_swa[i].iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                l_full[i].iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "position {i} is inside the window — the postures must be bit-equal"
            );
        }
        let diverged = (3..SEQ).any(|i| {
            l_swa[i]
                .iter()
                .zip(l_full[i].iter())
                .any(|(x, y)| x.to_bits() != y.to_bits())
        });
        assert!(diverged, "positions 3+ ignore the sliding window");
    }

    /// The normalized-latent law shows up at the logits: bounded norms
    /// through the full depth keep the head outputs finite and modest
    /// even with random weights.
    #[test]
    fn logits_finite_and_bounded() {
        let w = synth_weights(13, WINDOW);
        let toks = tokens();
        for row in prefill_logits(&w, &toks, K) {
            for v in row {
                assert!(v.is_finite(), "non-finite logit");
                assert!(
                    v.abs() < 1e5,
                    "logit magnitude {v} — the stream is unbounded"
                );
            }
        }
    }

    /// Per-loop footprint: the recurrent slots are k-proportional, the
    /// I/O slots are not (T2's byte-count assert reads the same
    /// counter).
    #[test]
    fn cache_footprint_is_k_proportional() {
        let w = synth_weights(17, WINDOW);
        let toks = tokens();
        let f = DoryForward::new(&w);
        let mut c1 = DoryKvCache::new(&w, 1);
        f.prefill(&toks, &mut c1).expect("prefill k=1");
        let mut c2 = DoryKvCache::new(&w, K);
        f.prefill(&toks, &mut c2).expect("prefill k=2");
        // 3 attention blocks: input (1 slot) + recurrent (k) + output
        // (1 slot); each row stores kv_dim K + V.
        let kv_dim = N_KV * HEAD_DIM;
        let want1 = 3 * SEQ * kv_dim * 2;
        let want2 = (2 + K) * SEQ * kv_dim * 2;
        assert_eq!(c1.kv_elements(), want1, "k=1 footprint");
        assert_eq!(c2.kv_elements(), want2, "k=2 footprint");
        assert_eq!(c2.len(), SEQ);
        assert!(c2.kv_elements() > c1.kv_elements());
    }

    // ── T2: the last-loop serving posture ─────────────────────────

    /// THE last-loop invariant: a from-scratch prefill is bit-identical
    /// under both policies at every k — within a pass every position is
    /// in-flight and attends its own pass's states, so the retention
    /// policy changes only what the cache KEEPS, never what a pass sees.
    /// Pinned at k=1 and k=4 (the paper's serving depth).
    #[test]
    fn last_loop_prefill_bit_identical_to_per_loop() {
        let w = synth_weights(7, WINDOW);
        let toks = tokens();
        for k in [1usize, 4] {
            let f = DoryForward::new(&w);
            let mut per = DoryKvCache::with_policy(&w, k, DoryCachePolicy::PerLoop);
            let mut last = DoryKvCache::with_policy(&w, k, DoryCachePolicy::LastLoop);
            let a = f.prefill(&toks, &mut per).expect("per-loop prefill");
            let b = f.prefill(&toks, &mut last).expect("last-loop prefill");
            assert_bit_eq(&a, &b, "prefill policy identity at k={k}");
        }
    }

    /// k=1 degenerate: pass 1 IS the final pass, so both policies are
    /// the same computation in both postures (prefill AND incremental).
    #[test]
    fn last_loop_k1_equals_per_loop_everywhere() {
        let w = synth_weights(23, WINDOW);
        let toks = tokens();
        let f = DoryForward::new(&w);
        for build in [
            |w: &DoryWeights| DoryKvCache::with_policy(w, 1, DoryCachePolicy::PerLoop),
            |w: &DoryWeights| DoryKvCache::with_policy(w, 1, DoryCachePolicy::LastLoop),
        ] {
            let mut pre = build(&w);
            let a = f.prefill(&toks, &mut pre).expect("prefill");
            let mut inc = build(&w);
            let mut b = Vec::with_capacity(toks.len());
            for &tok in &toks {
                b.push(f.decode_step(tok, &mut inc).expect("decode"));
            }
            assert_bit_eq(&a, &b, "k=1 identity under one policy");
        }
    }

    /// The serving payoff: the last-loop footprint is k-INDEPENDENT and
    /// equals the k=1 per-loop footprint (one slot per recurrent block).
    #[test]
    fn last_loop_footprint_is_k_independent() {
        let w = synth_weights(17, WINDOW);
        let toks = tokens();
        let f = DoryForward::new(&w);
        let mut l1 = DoryKvCache::with_policy(&w, 1, DoryCachePolicy::LastLoop);
        f.prefill(&toks, &mut l1).expect("prefill last k=1");
        let mut l4 = DoryKvCache::with_policy(&w, 4, DoryCachePolicy::LastLoop);
        f.prefill(&toks, &mut l4).expect("prefill last k=4");
        let mut p1 = DoryKvCache::with_policy(&w, 1, DoryCachePolicy::PerLoop);
        f.prefill(&toks, &mut p1).expect("prefill per k=1");
        assert_eq!(l4.kv_elements(), l1.kv_elements(), "last-loop k=1 vs k=4");
        assert_eq!(
            l4.kv_elements(),
            p1.kv_elements(),
            "last-loop k=4 vs per-loop k=1"
        );
        assert_eq!(l4.len(), SEQ);
    }

    /// The approximation is REAL: at k ≥ 2 the last-loop incremental
    /// decode diverges from per-loop (prefix reads are final-loop states
    /// instead of per-pass states). Engagement proof — never asserted
    /// zero; the real-checkpoint magnitude is the k=4 reported delta.
    #[test]
    fn last_loop_incremental_diverges_from_per_loop_at_k2() {
        let w = synth_weights(29, WINDOW);
        let toks = tokens();
        let f = DoryForward::new(&w);
        let run = |policy| {
            let mut c = DoryKvCache::with_policy(&w, K, policy);
            let mut out = Vec::with_capacity(toks.len());
            for &tok in &toks {
                out.push(f.decode_step(tok, &mut c).expect("decode"));
            }
            out
        };
        let per = run(DoryCachePolicy::PerLoop);
        let last = run(DoryCachePolicy::LastLoop);
        let differing = per
            .iter()
            .zip(last.iter())
            .flat_map(|(a, b)| a.iter().zip(b.iter()))
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        assert!(
            differing > 0,
            "k=2 last-loop decode is bit-identical to per-loop — the policy is not reached"
        );
    }

    /// The last-loop decode is self-consistent: same weights + same k +
    /// same policy ⇒ bit-identical across runs (the determinism law at
    /// the serving posture).
    #[test]
    fn last_loop_incremental_deterministic() {
        let w = synth_weights(31, WINDOW);
        let toks = tokens();
        let f = DoryForward::new(&w);
        let run = || {
            let mut c = DoryKvCache::with_policy(&w, K, DoryCachePolicy::LastLoop);
            let mut out = Vec::with_capacity(toks.len());
            for &tok in &toks {
                out.push(f.decode_step(tok, &mut c).expect("decode"));
            }
            out
        };
        assert_bit_eq(&run(), &run(), "last-loop determinism");
    }

    /// Elastic k is a per-REQUEST posture: two caches at different k on
    /// the same weights coexist without interference (the serving law —
    /// k lives on the cache, never the weights).
    #[test]
    fn elastic_k_caches_coexist_on_shared_weights() {
        let w = synth_weights(37, WINDOW);
        let toks = tokens();
        let f = DoryForward::new(&w);
        let mut c1 = DoryKvCache::with_policy(&w, 1, DoryCachePolicy::PerLoop);
        let mut c4 = DoryKvCache::with_policy(&w, 4, DoryCachePolicy::LastLoop);
        let l1 = f.prefill(&toks, &mut c1).expect("k=1 prefill");
        let l4 = f.prefill(&toks, &mut c4).expect("k=4 prefill");
        // The k=1 and k=4 prefills differ (depth), the caches are
        // independent, and a further decode on either reflects its own k.
        let differing = l1
            .iter()
            .zip(l4.iter())
            .flat_map(|(a, b)| a.iter().zip(b.iter()))
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        assert!(differing > 0, "k=4 prefill collapsed onto k=1");
        let n1 = f
            .decode_step(toks[0], &mut c1)
            .expect("decode on k=1 cache");
        let n4 = f
            .decode_step(toks[0], &mut c4)
            .expect("decode on k=4 cache");
        assert!(n1.iter().zip(n4.iter()).any(|(x, y)| x != y));
        assert_eq!(c1.len(), SEQ + 1);
        assert_eq!(c4.len(), SEQ + 1);
    }

    /// A token id ≥ vocab refuses loud and named (the lane's refusal
    /// law), on both entry points.
    #[test]
    fn token_out_of_range_refuses_loud() {
        let w = synth_weights(19, WINDOW);
        let f = DoryForward::new(&w);
        let mut cache = DoryKvCache::new(&w, 1);
        let err = f
            .decode_step(VOCAB as u32, &mut cache)
            .expect_err("oob token must refuse");
        assert_eq!(
            err,
            DoryForwardError::TokenOutOfRange {
                token: VOCAB as u32,
                vocab: VOCAB
            }
        );
        let err = f
            .prefill(&[0, VOCAB as u32], &mut cache)
            .expect_err("oob token must refuse");
        assert!(matches!(err, DoryForwardError::TokenOutOfRange { .. }));
    }
}
