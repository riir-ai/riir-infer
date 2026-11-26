//! LLaMA-family forward pass.
//!
//! Supports `LLaMA`, Mistral, `MiniCPM`, and other LLaMA-architecture models.
//! Differences from Gemma 2:
//! - No embedding scaling (no `sqrt(n_embd`))
//! - No post-attention / post-MLP norms
//! - `SwiGLU` MLP (`SiLU` activation) instead of `GeGLU`
//! - Separate `lm_head` (not tied to wte)
//! - No logit softcapping

use super::*;
use crate::llama_layer::LlamaTransformerWeights;
use crate::transformer::gemma2::PostLayerHook;

/// LLaMA-family forward pass (decode, single token).
pub fn forward_llama<'a>(
    ctx: &'a mut ForwardContext,
    weights: &LlamaTransformerWeights,
    cache: &mut MultiLayerKVCache,
    token: usize,
    pos: usize,
    config: &Config,
) -> &'a mut [f32] {
    let n = config.n_embd;
    let hd = config.head_dim;
    let q_dim = config.n_head * config.head_dim;
    let kvd = types::kv_dim(config);
    let n_kv = config.n_kv_head;

    // 1. Embedding: x = wte[token] (no scaling, unlike Gemma 2)
    let tok_off = token * n;
    unsafe {
        load_embed(&mut ctx.x, &weights.wte, tok_off, n);
    }

    // 2. Layer loop
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        let layer_cache = &mut cache.layers[layer_idx];

        // a. Save residual
        ctx.xr[..n].copy_from_slice(&ctx.x[..n]);

        // b. Pre-attention RMSNorm (no offset)
        types::rmsnorm_with_gamma_eps(&mut ctx.x, &layer_weights.input_norm, config.rms_norm_eps);

        // c. QKV projections — parallel (Issue 053: serial for small models)
        let x_in = &ctx.x[..n];
        let wq = &layer_weights.attn_wq;
        let wk = &layer_weights.attn_wk;
        let wv = &layer_weights.attn_wv;
        let q_buf = &mut ctx.q;
        let k_buf = &mut ctx.k;
        let v_buf = &mut ctx.v;
        if n >= RAYON_QKV_THRESHOLD {
            rayon::scope(|s| {
                s.spawn(move |_| matmul(q_buf, wq, x_in, q_dim, n));
                s.spawn(move |_| matmul(k_buf, wk, x_in, kvd, n));
                s.spawn(move |_| matmul(v_buf, wv, x_in, kvd, n));
            });
        } else {
            matmul(q_buf, wq, x_in, q_dim, n);
            matmul(k_buf, wk, x_in, kvd, n);
            matmul(v_buf, wv, x_in, kvd, n);
        }

        // d. Apply RoPE to Q and K
        crate::rope::apply_rope_with_freq(
            &mut ctx.q,
            &mut ctx.k,
            pos,
            hd,
            ctx.rope_freq_table.as_slice(),
        );

        // e. Store K,V in per-layer cache
        let pos_off = pos * kvd;
        unsafe {
            std::ptr::copy_nonoverlapping(
                ctx.k.as_ptr(),
                layer_cache.key.as_mut_ptr().add(pos_off),
                kvd,
            );
            std::ptr::copy_nonoverlapping(
                ctx.v.as_ptr(),
                layer_cache.value.as_mut_ptr().add(pos_off),
                kvd,
            );
        }

        // f. Multi-head attention with GQA (no softcapping for LLaMA), with
        // the shared m_Y probe / row-logit-floor hooks (riir-infer Issue 011).
        unsafe {
            super::attend::attend_row(
                ctx,
                &layer_cache.key,
                &layer_cache.value,
                layer_idx,
                super::attend::AttnShape {
                    n_head: config.n_head,
                    n_kv_head: n_kv,
                    kv_dim: kvd,
                    head_dim: hd,
                    t_n: pos + 1,
                    scale: 1.0 / (hd as f32).sqrt(),
                    softcap: 0.0, // no attention logit softcapping
                    block_size: config.block_size,
                },
            );
        }

        // g. Output projection
        matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, q_dim);

        // h. Residual add (no post-norm in LLaMA)
        for i in 0..n {
            unsafe {
                *ctx.x.get_unchecked_mut(i) += *ctx.xr.get_unchecked(i);
            }
        }

        // i. Save residual2
        ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);

        // j. Pre-MLP RMSNorm
        types::rmsnorm_with_gamma_eps(
            &mut ctx.x,
            &layer_weights.post_attn_norm,
            config.rms_norm_eps,
        );

        // k. SwiGLU (Issue 053: threshold-gated parallelism)
        let x_in = &ctx.x[..n];
        let wg = &layer_weights.gate_proj;
        let wu = &layer_weights.up_proj;
        let gate_buf = &mut ctx.gate;
        let up_buf = &mut ctx.up;
        let mlp_hidden = config.mlp_hidden;
        if n >= RAYON_MLP_THRESHOLD {
            rayon::scope(|s| {
                s.spawn(move |_| matmul(gate_buf, wg, x_in, mlp_hidden, n));
                s.spawn(move |_| matmul(up_buf, wu, x_in, mlp_hidden, n));
            });
        } else {
            matmul(gate_buf, wg, x_in, mlp_hidden, n);
            matmul(up_buf, wu, x_in, mlp_hidden, n);
        }
        types::swiglu(&mut ctx.hidden, &ctx.gate, &ctx.up);

        // l. Down projection
        types::matmul_parallel(
            &mut ctx.x,
            &layer_weights.down_proj,
            &ctx.hidden,
            n,
            config.mlp_hidden,
        );

        // m. Residual add (no post-norm in LLaMA)
        for i in 0..n {
            unsafe {
                *ctx.x.get_unchecked_mut(i) += *ctx.xr2.get_unchecked(i);
            }
        }

        // Delta routing: reuse existing infrastructure if enabled
        #[cfg(feature = "delta_routing")]
        apply_delta_routing_step(ctx, layer_idx, n, None);
    }

    // Snapshot hidden state
    ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);

    // 3. Final RMSNorm
    types::rmsnorm_with_gamma_eps(&mut ctx.x, &weights.final_norm, config.rms_norm_eps);

    // 4. Separate lm_head (not tied)
    types::matmul_parallel(
        &mut ctx.logits,
        &weights.lm_head,
        &ctx.x,
        config.vocab_size,
        n,
    );

    &mut ctx.logits
}

/// LLaMA-family forward with the Issue-395 `PostLayerHook` seam (Issue 1254 —
/// the Plato's-Key cell's recipient-side replace/capture point).
///
/// The layer stack is [`forward_llama`]'s steps 1–4 **verbatim** (the
/// correspondence law is pinned bit-exactly by the `g3` test module below:
/// unarmed, the two forwards agree to the last bit over multi-position
/// runs), with the hook call inserted after the post-MLP residual add and
/// the cfg `delta_routing` arm — the gemma seam's exact placement
/// ("after the MLP residual add and after delta-routing (if enabled)").
/// The loop is duplicated, not delegated: `forward_llama` stays the
/// production path byte-identical BY CONSTRUCTION, matching this file's
/// existing fork convention (`forward_llama_attn_capture`).
///
/// The harness arms the hook per forward CALL — the swap/capture applies
/// only at the subject position (`pos ==` subject index), so an armed call
/// and an unarmed call share one hook type and the zero-cost unarmed
/// posture is the caller passing a no-op impl (`NoHook` monomorphizes
/// away, as on the gemma path).
///
/// ⛔ The ctx RoPE table must come from
/// [`LlamaTransformerWeights::build_rope_freq_table`] when the checkpoint
/// carries llama3 scaling factors (`rope_freqs.weight`) — the forward reads
/// the table it is given, never the weights' factors directly.
pub fn forward_llama_hooked<'a, H: PostLayerHook + ?Sized>(
    ctx: &'a mut ForwardContext,
    weights: &LlamaTransformerWeights,
    cache: &mut MultiLayerKVCache,
    token: usize,
    pos: usize,
    config: &Config,
    hook: &mut H,
) -> &'a mut [f32] {
    let n = config.n_embd;
    let hd = config.head_dim;
    let q_dim = config.n_head * config.head_dim;
    let kvd = types::kv_dim(config);
    let n_kv = config.n_kv_head;

    // 1. Embedding: x = wte[token] (no scaling, unlike Gemma 2)
    let tok_off = token * n;
    unsafe {
        load_embed(&mut ctx.x, &weights.wte, tok_off, n);
    }

    // 2. Layer loop
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        let layer_cache = &mut cache.layers[layer_idx];

        // a. Save residual
        ctx.xr[..n].copy_from_slice(&ctx.x[..n]);

        // b. Pre-attention RMSNorm (no offset)
        types::rmsnorm_with_gamma_eps(&mut ctx.x, &layer_weights.input_norm, config.rms_norm_eps);

        // c. QKV projections — parallel (Issue 053: serial for small models)
        let x_in = &ctx.x[..n];
        let wq = &layer_weights.attn_wq;
        let wk = &layer_weights.attn_wk;
        let wv = &layer_weights.attn_wv;
        let q_buf = &mut ctx.q;
        let k_buf = &mut ctx.k;
        let v_buf = &mut ctx.v;
        if n >= RAYON_QKV_THRESHOLD {
            rayon::scope(|s| {
                s.spawn(move |_| matmul(q_buf, wq, x_in, q_dim, n));
                s.spawn(move |_| matmul(k_buf, wk, x_in, kvd, n));
                s.spawn(move |_| matmul(v_buf, wv, x_in, kvd, n));
            });
        } else {
            matmul(q_buf, wq, x_in, q_dim, n);
            matmul(k_buf, wk, x_in, kvd, n);
            matmul(v_buf, wv, x_in, kvd, n);
        }

        // d. Apply RoPE to Q and K
        crate::rope::apply_rope_with_freq(
            &mut ctx.q,
            &mut ctx.k,
            pos,
            hd,
            ctx.rope_freq_table.as_slice(),
        );

        // e. Store K,V in per-layer cache
        let pos_off = pos * kvd;
        unsafe {
            std::ptr::copy_nonoverlapping(
                ctx.k.as_ptr(),
                layer_cache.key.as_mut_ptr().add(pos_off),
                kvd,
            );
            std::ptr::copy_nonoverlapping(
                ctx.v.as_ptr(),
                layer_cache.value.as_mut_ptr().add(pos_off),
                kvd,
            );
        }

        // f. Multi-head attention with GQA (no softcapping for LLaMA)
        unsafe {
            super::attend::attend_row(
                ctx,
                &layer_cache.key,
                &layer_cache.value,
                layer_idx,
                super::attend::AttnShape {
                    n_head: config.n_head,
                    n_kv_head: n_kv,
                    kv_dim: kvd,
                    head_dim: hd,
                    t_n: pos + 1,
                    scale: 1.0 / (hd as f32).sqrt(),
                    softcap: 0.0, // no attention logit softcapping
                    block_size: config.block_size,
                },
            );
        }

        // g. Output projection
        matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, q_dim);

        // h. Residual add (no post-norm in LLaMA)
        for i in 0..n {
            unsafe {
                *ctx.x.get_unchecked_mut(i) += *ctx.xr.get_unchecked(i);
            }
        }

        // i. Save residual2
        ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);

        // j. Pre-MLP RMSNorm
        types::rmsnorm_with_gamma_eps(
            &mut ctx.x,
            &layer_weights.post_attn_norm,
            config.rms_norm_eps,
        );

        // k. SwiGLU (Issue 053: threshold-gated parallelism)
        let x_in = &ctx.x[..n];
        let wg = &layer_weights.gate_proj;
        let wu = &layer_weights.up_proj;
        let gate_buf = &mut ctx.gate;
        let up_buf = &mut ctx.up;
        let mlp_hidden = config.mlp_hidden;
        if n >= RAYON_MLP_THRESHOLD {
            rayon::scope(|s| {
                s.spawn(move |_| matmul(gate_buf, wg, x_in, mlp_hidden, n));
                s.spawn(move |_| matmul(up_buf, wu, x_in, mlp_hidden, n));
            });
        } else {
            matmul(gate_buf, wg, x_in, mlp_hidden, n);
            matmul(up_buf, wu, x_in, mlp_hidden, n);
        }
        types::swiglu(&mut ctx.hidden, &ctx.gate, &ctx.up);

        // l. Down projection
        types::matmul_parallel(
            &mut ctx.x,
            &layer_weights.down_proj,
            &ctx.hidden,
            n,
            config.mlp_hidden,
        );

        // m. Residual add (no post-norm in LLaMA)
        for i in 0..n {
            unsafe {
                *ctx.x.get_unchecked_mut(i) += *ctx.xr2.get_unchecked(i);
            }
        }

        // Delta routing: reuse existing infrastructure if enabled
        #[cfg(feature = "delta_routing")]
        apply_delta_routing_step(ctx, layer_idx, n, None);

        // The Issue-395 post-layer seam: mutate the post-layer residual
        // in-place (the swap's replace point / the key-capture read point).
        hook.after_layer(layer_idx, &mut ctx.x[..n]);
    }

    // Snapshot hidden state
    ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);

    // 3. Final RMSNorm
    types::rmsnorm_with_gamma_eps(&mut ctx.x, &weights.final_norm, config.rms_norm_eps);

    // 4. Separate lm_head (not tied)
    types::matmul_parallel(
        &mut ctx.logits,
        &weights.lm_head,
        &ctx.x,
        config.vocab_size,
        n,
    );

    &mut ctx.logits
}

/// Generate tokens using `LLaMA` weights with temperature sampling.
pub fn generate_llama(
    weights: &LlamaTransformerWeights,
    config: &Config,
    rng: &mut Rng,
    prompt_tokens: &[usize],
    max_tokens: usize,
) -> Vec<usize> {
    let mut cache = MultiLayerKVCache::new(config);
    let mut ctx = ForwardContext::new(config);
    // Issue 1254: a checkpoint carrying llama3 rope-scaling factors
    // (`rope_freqs.weight`) must run the WEIGHTED frequency table — build
    // it through the weights' own constructor (a plain table on a scaled
    // checkpoint runs wrong frequencies silently).
    ctx.rope_freq_table = weights.build_rope_freq_table(config);
    // Pre-allocate for prompt + worst-case generated tokens so the autoregressive
    // decode loop never reallocates.
    let mut tokens: Vec<usize> = Vec::with_capacity(prompt_tokens.len() + max_tokens);
    tokens.extend_from_slice(prompt_tokens);

    // Prefill: process all prompt tokens
    for (pos, &token) in prompt_tokens.iter().enumerate() {
        forward_llama(&mut ctx, weights, &mut cache, token, pos, config);
    }

    // Generate: autoregressive decode
    for _ in 0..max_tokens {
        // Temperature scaling + softmax on last forward pass logits
        softmax_scaled(&mut ctx.logits, 1.0 / config.temperature);

        // Reuse pre-allocated CDF buffer -- avoids vocab_size allocation per token
        let next = crate::types::sample_token_into(&ctx.logits, rng, &mut ctx.cdf_buf);
        if next == 1 {
            break;
        } // EOS
        tokens.push(next);

        // Forward pass for next position
        let pos = tokens.len() - 1;
        forward_llama(&mut ctx, weights, &mut cache, tokens[pos], pos, config);
    }

    tokens.shrink_to_fit();
    tokens
}

/// LLaMA-family forward pass that captures attention pattern features at specified layers.
///
/// This is a **cold-path diagnostic function** (Issue 380 Path A) — it duplicates
/// the layer loop from [`forward_llama`] to compute attention entropy at the
/// last token position for each layer in `capture_layers`. The entropy is
/// computed from `softmax(Q_h · K^T)` over all positions `[0..t_n]` for each
/// head `h`, after `RoPE` is applied to Q/K and after K is stored in the cache.
///
/// **Do NOT use in hot paths** — the entropy computation adds `O(n_head * t_n)`
/// work per captured layer, plus a `t_n`-sized allocation. Use [`forward_llama`]
/// for production decode.
///
/// The rest of the forward pass is identical to [`forward_llama`]: no embedding
/// scaling, `SwiGLU` MLP, RMSNorm-with-gamma pre-norms (no post-norms), separate
/// `lm_head`, and no logit softcapping.
///
/// Returns `(&mut ctx.logits, Vec<AttnLayerFeatures>)`. The features vector
/// has one entry per layer in `capture_layers` (in ascending layer order, since
/// the layer loop is sequential). Layers not in `capture_layers` are skipped.
///
/// # Contract
///
/// - `capture_layers` entries must be `< config.n_layer`.
/// - `token < config.vocab_size`.
/// - `pos` is the 0-based position of `token` in the sequence; `t_n = pos + 1`.
/// - The KV cache for positions `[0..pos]` must already be populated.
#[allow(clippy::too_many_arguments)]
pub fn forward_llama_attn_capture<'a>(
    ctx: &'a mut ForwardContext,
    weights: &LlamaTransformerWeights,
    cache: &mut MultiLayerKVCache,
    token: usize,
    pos: usize,
    config: &Config,
    capture_layers: &[usize],
) -> (&'a mut [f32], Vec<AttnLayerFeatures>) {
    let n = config.n_embd;
    let hd = config.head_dim;
    let q_dim = config.n_head * config.head_dim;
    let kvd = types::kv_dim(config);
    let n_kv = config.n_kv_head;

    // 1. Embedding: x = wte[token] (no scaling, unlike Gemma 2)
    let tok_off = token * n;
    unsafe {
        load_embed(&mut ctx.x, &weights.wte, tok_off, n);
    }

    // Loop-invariant attention scale and token count.
    let scale = 1.0 / (hd as f32).sqrt();
    let t_n = pos + 1;
    let ln_tn = (t_n as f32).ln().max(1e-12);

    let mut features: Vec<AttnLayerFeatures> = Vec::with_capacity(capture_layers.len());

    // 2. Layer loop (inlined from forward_llama — no hooks, no LoRA).
    for (layer_idx, layer_weights) in weights.layers.iter().enumerate() {
        let layer_cache = &mut cache.layers[layer_idx];

        // a. Save residual
        ctx.xr[..n].copy_from_slice(&ctx.x[..n]);

        // b. Pre-attention RMSNorm (no offset)
        types::rmsnorm_with_gamma_eps(&mut ctx.x, &layer_weights.input_norm, config.rms_norm_eps);

        // c. QKV projections — parallel (Issue 053: serial for small models)
        let x_in = &ctx.x[..n];
        let wq = &layer_weights.attn_wq;
        let wk = &layer_weights.attn_wk;
        let wv = &layer_weights.attn_wv;
        let q_buf = &mut ctx.q;
        let k_buf = &mut ctx.k;
        let v_buf = &mut ctx.v;
        if n >= RAYON_QKV_THRESHOLD {
            rayon::scope(|s| {
                s.spawn(move |_| matmul(q_buf, wq, x_in, q_dim, n));
                s.spawn(move |_| matmul(k_buf, wk, x_in, kvd, n));
                s.spawn(move |_| matmul(v_buf, wv, x_in, kvd, n));
            });
        } else {
            matmul(q_buf, wq, x_in, q_dim, n);
            matmul(k_buf, wk, x_in, kvd, n);
            matmul(v_buf, wv, x_in, kvd, n);
        }

        // d. Apply RoPE to Q and K
        crate::rope::apply_rope_with_freq(
            &mut ctx.q,
            &mut ctx.k,
            pos,
            hd,
            ctx.rope_freq_table.as_slice(),
        );

        // e. Store K,V in per-layer cache
        let pos_off = pos * kvd;
        unsafe {
            std::ptr::copy_nonoverlapping(
                ctx.k.as_ptr(),
                layer_cache.key.as_mut_ptr().add(pos_off),
                kvd,
            );
            std::ptr::copy_nonoverlapping(
                ctx.v.as_ptr(),
                layer_cache.value.as_mut_ptr().add(pos_off),
                kvd,
            );
        }

        // ── Attention capture (Issue 380 Path A) ──
        // After RoPE+cache-store, before the fused attention call: compute
        // per-head softmax(Q·K^T) at the last token position to extract
        // entropy / max-weight / self-weight. LLaMA has no attention logit
        // softcapping, so the score is just dot * scale.
        if capture_layers.contains(&layer_idx) {
            let mut entropy_sum = 0.0_f32;
            let mut max_weight_sum = 0.0_f32;
            let mut self_weight_sum = 0.0_f32;

            let scores_buf = &mut ctx.head_scores[..t_n];

            for h in 0..config.n_head {
                let q_off = h * hd;
                let kv_group = h * n_kv / config.n_head;
                let kv_group_off = kv_group * hd;

                // Compute Q·K scores for all positions [0..t_n].
                let mut max_score = f32::NEG_INFINITY;
                for (t, slot) in scores_buf.iter_mut().enumerate().take(t_n) {
                    let k_off = t * kvd + kv_group_off;
                    let dot = crate::simd::simd_dot_f32(
                        &ctx.q[q_off..q_off + hd],
                        &layer_cache.key[k_off..k_off + hd],
                        hd,
                    );
                    let score = dot * scale; // LLaMA: no softcapping
                    *slot = score;
                    if score > max_score {
                        max_score = score;
                    }
                }

                // Softmax (stable: subtract max).
                let mut sum = 0.0_f32;
                for slot in scores_buf.iter_mut().take(t_n) {
                    let e = (*slot - max_score).exp();
                    *slot = e;
                    sum += e;
                }
                let inv_sum = 1.0 / sum;

                // Entropy + max weight + self weight (last position).
                let mut entropy = 0.0_f32;
                let mut max_w = 0.0_f32;
                for &e in scores_buf.iter().take(t_n) {
                    let w = e * inv_sum;
                    if w > 1e-12 {
                        entropy -= w * w.ln();
                    }
                    if w > max_w {
                        max_w = w;
                    }
                }
                let self_w = scores_buf[t_n - 1] * inv_sum;

                entropy_sum += entropy / ln_tn;
                max_weight_sum += max_w;
                self_weight_sum += self_w;
            }

            let n_head_f = config.n_head as f32;
            features.push(AttnLayerFeatures {
                layer_idx,
                entropy_normalized: entropy_sum / n_head_f,
                mean_max_weight: max_weight_sum / n_head_f,
                mean_self_weight: self_weight_sum / n_head_f,
            });
        }

        // f. Multi-head attention with GQA (no softcapping for LLaMA)
        ctx.attn_out[..q_dim].fill(0.0);
        unsafe {
            attention_heads_parallel(
                &ctx.q,
                &layer_cache.key,
                &layer_cache.value,
                &mut ctx.attn_out,
                &mut ctx.head_scores,
                config.n_head,
                n_kv,
                kvd,
                hd,
                t_n,
                scale,
                0.0, // no attention logit softcapping
                config.block_size,
            );
        }

        // g. Output projection
        matmul(&mut ctx.x, &layer_weights.attn_wo, &ctx.attn_out, n, q_dim);

        // h. Residual add (no post-norm in LLaMA)
        for i in 0..n {
            unsafe {
                *ctx.x.get_unchecked_mut(i) += *ctx.xr.get_unchecked(i);
            }
        }

        // i. Save residual2
        ctx.xr2[..n].copy_from_slice(&ctx.x[..n]);

        // j. Pre-MLP RMSNorm
        types::rmsnorm_with_gamma_eps(
            &mut ctx.x,
            &layer_weights.post_attn_norm,
            config.rms_norm_eps,
        );

        // k. SwiGLU (Issue 053: threshold-gated parallelism)
        let x_in = &ctx.x[..n];
        let wg = &layer_weights.gate_proj;
        let wu = &layer_weights.up_proj;
        let gate_buf = &mut ctx.gate;
        let up_buf = &mut ctx.up;
        let mlp_hidden = config.mlp_hidden;
        if n >= RAYON_MLP_THRESHOLD {
            rayon::scope(|s| {
                s.spawn(move |_| matmul(gate_buf, wg, x_in, mlp_hidden, n));
                s.spawn(move |_| matmul(up_buf, wu, x_in, mlp_hidden, n));
            });
        } else {
            matmul(gate_buf, wg, x_in, mlp_hidden, n);
            matmul(up_buf, wu, x_in, mlp_hidden, n);
        }
        types::swiglu(&mut ctx.hidden, &ctx.gate, &ctx.up);

        // l. Down projection
        types::matmul_parallel(
            &mut ctx.x,
            &layer_weights.down_proj,
            &ctx.hidden,
            n,
            config.mlp_hidden,
        );

        // m. Residual add (no post-norm in LLaMA)
        for i in 0..n {
            unsafe {
                *ctx.x.get_unchecked_mut(i) += *ctx.xr2.get_unchecked(i);
            }
        }

        // Delta routing: reuse existing infrastructure if enabled
        #[cfg(feature = "delta_routing")]
        apply_delta_routing_step(ctx, layer_idx, n, None);
    }

    // Snapshot hidden state
    ctx.hidden_state[..n].copy_from_slice(&ctx.x[..n]);

    // 3. Final RMSNorm
    types::rmsnorm_with_gamma_eps(&mut ctx.x, &weights.final_norm, config.rms_norm_eps);

    // 4. Separate lm_head (not tied)
    types::matmul_parallel(
        &mut ctx.logits,
        &weights.lm_head,
        &ctx.x,
        config.vocab_size,
        n,
    );

    (&mut ctx.logits, features)
}

// ── Issue-1254 G3: the Issue-395 llama seam ─────────────────────────────

#[cfg(test)]
mod issue1254_llama_hook_g3 {
    use super::*;
    use crate::llama_layer::{LlamaLayerWeights, LlamaTransformerWeights};
    use crate::types::ModelArchitecture;

    /// Deterministic LCG — no `rand` dep (the ternary tests.rs convention).
    struct Lcg(u64);

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0
        }
        fn next_f32(&mut self) -> f32 {
            ((self.next_u64() >> 33) as f32) / (1u64 << 31) as f32 - 1.0
        }
    }

    fn tiny_llama_config() -> Config {
        Config {
            vocab_size: 64,
            block_size: 32,
            n_embd: 64,
            n_head: 4,
            n_kv_head: 2,
            head_dim: 16,
            n_layer: 3,
            mlp_hidden: 128,
            rms_norm_eps: 1e-6,
            model_arch: ModelArchitecture::Llama,
            tied_embeddings: false,
            use_rope: true,
            post_norm: false,
            ..Default::default()
        }
    }

    fn tiny_llama(seed: u64) -> (Config, LlamaTransformerWeights) {
        let config = tiny_llama_config();
        let n = config.n_embd;
        let q_dim = config.n_head * config.head_dim;
        let kvd = config.n_kv_head * config.head_dim;
        let mlp = config.mlp_hidden;
        let vocab = config.vocab_size;
        let mut rng = Lcg::new(seed);

        let mat = |rng: &mut Lcg, rows: usize, cols: usize| -> Vec<f32> {
            (0..rows * cols).map(|_| rng.next_f32() * 0.2).collect()
        };
        let norm = |rng: &mut Lcg, len: usize| -> Vec<f32> {
            (0..len)
                .map(|_| 0.5 + 0.5 * (rng.next_f32() + 1.0))
                .collect()
        };

        let mut layers = Vec::with_capacity(config.n_layer);
        for _ in 0..config.n_layer {
            layers.push(LlamaLayerWeights {
                attn_wq: mat(&mut rng, q_dim, n),
                attn_wk: mat(&mut rng, kvd, n),
                attn_wv: mat(&mut rng, kvd, n),
                attn_wo: mat(&mut rng, n, q_dim),
                gate_proj: mat(&mut rng, mlp, n),
                up_proj: mat(&mut rng, mlp, n),
                down_proj: mat(&mut rng, n, mlp),
                input_norm: norm(&mut rng, n),
                post_attn_norm: norm(&mut rng, n),
            });
        }

        let weights = LlamaTransformerWeights {
            wte: mat(&mut rng, vocab, n),
            lm_head: mat(&mut rng, vocab, n),
            final_norm: norm(&mut rng, n),
            layers,
            rope_freq_weights: None,
        };
        (config, weights)
    }

    /// G3 (unarmed identity): `forward_llama` ≡ `forward_llama_hooked(NoHook)`
    /// bit-exactly over a multi-position run — the armed seam compiles to a
    /// no-op, so the cell's unarmed posture IS the production forward.
    #[test]
    fn g3_unhooked_seam_is_bit_identical_to_plain_forward() {
        let (config, weights) = tiny_llama(0xB0C);
        let plain_cache = &mut MultiLayerKVCache::new(&config);
        let hook_cache = &mut MultiLayerKVCache::new(&config);
        let mut plain_ctx = ForwardContext::new(&config);
        let mut hook_ctx = ForwardContext::new(&config);
        let mut hook = NoHook;

        for pos in 0..6 {
            let token = (pos * 7 + 3) % config.vocab_size;
            let plain =
                forward_llama(&mut plain_ctx, &weights, plain_cache, token, pos, &config).to_vec();
            let hooked = forward_llama_hooked(
                &mut hook_ctx,
                &weights,
                hook_cache,
                token,
                pos,
                &config,
                &mut hook,
            )
            .to_vec();
            assert_eq!(
                plain.len(),
                hooked.len(),
                "logit length divergence at pos {pos}"
            );
            for (i, (a, b)) in plain.iter().zip(hooked.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "unarmed seam diverged at pos {pos} logit {i}: {a} != {b}"
                );
            }
            // The KV caches must stay identical too — the hook must not
            // perturb state the next position attends over.
            for (l, (pk, hk)) in plain_cache
                .layers
                .iter()
                .zip(hook_cache.layers.iter())
                .enumerate()
            {
                assert_eq!(
                    pk.key.len(),
                    hk.key.len(),
                    "KV cache length divergence at pos {pos} layer {l}"
                );
                for (kv_a, kv_b) in pk.key.iter().zip(hk.key.iter()) {
                    assert_eq!(
                        kv_a.to_bits(),
                        kv_b.to_bits(),
                        "KV cache divergence at pos {pos} layer {l}"
                    );
                }
            }
        }
    }

    /// Engagement proof (the Issue-1124 T1.5 law): a mutating hook is
    /// REACHED — zeroing the post-layer residual at layer 0 must move the
    /// logits. A seam that compiles but never fires is the silent no-op this
    /// cell cannot afford.
    #[test]
    fn g3_armed_hook_is_reached_and_changes_logits() {
        struct ZeroAtLayer0;
        impl PostLayerHook for ZeroAtLayer0 {
            fn after_layer(&mut self, layer_idx: usize, residual: &mut [f32]) {
                if layer_idx == 0 {
                    residual.fill(0.0);
                }
            }
        }

        let (config, weights) = tiny_llama(0xB0D);
        let cache = &mut MultiLayerKVCache::new(&config);
        let mut ctx = ForwardContext::new(&config);
        let plain = forward_llama(&mut ctx, &weights, cache, 5, 0, &config).to_vec();

        let cache2 = &mut MultiLayerKVCache::new(&config);
        let mut ctx2 = ForwardContext::new(&config);
        let mut hook = ZeroAtLayer0;
        let armed =
            forward_llama_hooked(&mut ctx2, &weights, cache2, 5, 0, &config, &mut hook).to_vec();
        assert!(
            plain
                .iter()
                .zip(armed.iter())
                .any(|(a, b)| a.to_bits() != b.to_bits()),
            "zero-at-layer-0 hook did not move the logits — the seam is not reached"
        );
    }

    /// Capture proof: the same seam reads the post-layer residual — the
    /// donor-side capture protocol without any second mechanism. The captured
    /// value equals the state the plain forward carries at that point,
    /// verified indirectly: capturing-without-mutating must stay bit-identical
    /// to the plain forward while recording a vector of the right shape.
    #[test]
    fn g3_capture_only_hook_preserves_forward_and_records() {
        struct CaptureLayer1 {
            captured: Option<Vec<f32>>,
        }
        impl PostLayerHook for CaptureLayer1 {
            fn after_layer(&mut self, layer_idx: usize, residual: &mut [f32]) {
                if layer_idx == 1 {
                    self.captured = Some(residual.to_vec());
                }
            }
        }

        let (config, weights) = tiny_llama(0xB0E);
        let cache = &mut MultiLayerKVCache::new(&config);
        let mut ctx = ForwardContext::new(&config);
        let plain = forward_llama(&mut ctx, &weights, cache, 9, 2, &config).to_vec();

        let cache2 = &mut MultiLayerKVCache::new(&config);
        let mut ctx2 = ForwardContext::new(&config);
        let mut hook = CaptureLayer1 { captured: None };
        let captured_out =
            forward_llama_hooked(&mut ctx2, &weights, cache2, 9, 2, &config, &mut hook).to_vec();
        assert!(
            plain
                .iter()
                .zip(captured_out.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "capture-only hook perturbed the forward"
        );
        let captured = hook.captured.expect("layer-1 capture never fired");
        assert_eq!(captured.len(), config.n_embd, "captured residual width");
    }

    /// The weighted RoPE table: all-ones factors build a table bit-identical
    /// to the plain one; scaled factors scale per-dim; a wrong-shaped factor
    /// vector refuses loud.
    #[test]
    fn g3_rope_freq_weighted_table_laws() {
        let theta = 500_000.0f32;
        let hd = 16usize;

        let plain: Vec<f32> = crate::rope::RopeFreqTable::new(theta, hd)
            .as_slice()
            .to_vec();

        let ones = vec![1.0f32; hd / 2];
        let weighted_ones = crate::rope::RopeFreqTable::new_weighted(theta, hd, &ones);
        let weighted_ones = weighted_ones.as_slice();
        assert_eq!(plain.len(), weighted_ones.len());
        for (a, b) in plain.iter().zip(weighted_ones) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "all-ones factors must be the plain table"
            );
        }

        let halves = vec![0.5f32; hd / 2];
        let weighted_half = crate::rope::RopeFreqTable::new_weighted(theta, hd, &halves);
        for (p, w) in plain.iter().zip(weighted_half.as_slice()) {
            assert_eq!(
                (p * 0.5).to_bits(),
                w.to_bits(),
                "factor must scale per-dim"
            );
        }

        let result = std::panic::catch_unwind(|| {
            let _ = crate::rope::RopeFreqTable::new_weighted(theta, hd, &[1.0; 3]);
        });
        assert!(result.is_err(), "wrong-shaped factors must refuse loud");
    }

    /// The weights' ONE table constructor: None ⇒ the plain table (bit-exact);
    /// Some ⇒ the weighted table.
    #[test]
    fn g3_build_rope_freq_table_routes_by_factors() {
        let (config, mut weights) = tiny_llama(0xB0F);
        let theta = config.rope_theta;
        let hd = config.head_dim;

        let plain: Vec<f32> = weights.build_rope_freq_table(&config).as_slice().to_vec();
        let expected: Vec<f32> = crate::rope::RopeFreqTable::new(theta, hd)
            .as_slice()
            .to_vec();
        assert_eq!(plain.len(), expected.len());
        for (a, b) in plain.iter().zip(expected.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }

        weights.rope_freq_weights = Some(vec![0.5f32; hd / 2]);
        let weighted: Vec<f32> = weights.build_rope_freq_table(&config).as_slice().to_vec();
        for (p, w) in plain.iter().zip(weighted.iter()) {
            assert_eq!((p * 0.5).to_bits(), w.to_bits());
        }
    }
}
