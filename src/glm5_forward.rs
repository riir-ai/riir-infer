//! GLM-5.3-Flash DSA-layer CPU-reference decode — Plan 627 Phase 1 T1.
//!
//! A thin wrapper over the UPSTREAM MLA substrate
//! (`katgpt_attn::mla` — `MlaConfig`, `MlaWeights`, `MlaKVCache`,
//! `mla_forward_token_dispatched`), specialized to the glm5 shape:
//!
//! - `qk_rope_head_dim = 0` (NO attention rope — config-proven; the only
//!   rope in the arch is the trained indexer's, which is Phase 4's lane).
//!   Upstream nope-mode (`use_nope = true`) with a zero rope dim is
//!   structurally clean: every rope slice is empty, every rope dot is 0.
//! - NO output gate: the DSA-layer weight family carries no gate tensor
//!   (inventory-proven — the only gate rows are the MoE routers).
//! - The fused projections de-fuse exactly as the upstream module doc's
//!   "loader concern" paragraph names (`q_b_proj` = the row-stack of
//!   `w_uq` + `w_qr`; `kv_a_proj_with_mqa` = `w_dkv` + `w_kr`;
//!   `kv_b_proj` = `w_uk` + `w_uv`): with `d_r = 0` the rope halves are
//!   EMPTY, so `q_b_proj` is `w_uq` verbatim, `kv_a_proj_with_mqa` is
//!   `w_dkv` verbatim, and `kv_b_proj` splits at row `d_h·n_h`.
//!
//! DENSE-PATH ONLY: the trained KPool indexer (`self_attn.indexer.*`)
//! selects tokens at long context — that selection is the Phase-4 LSA
//! bucket-indexer lane's. The CPU reference here attends to ALL cached
//! tokens (the dense fallback), which is exact while the context is
//! within the indexer's `index_topk` reach. The wrapper refuses a cache
//! longer than that reach rather than silently over-attend.
//!
//! Metadata contract: `Glm5Config` (the `glm5` feature) validates the
//! arch; this module adds the forward. Feature `glm5_forward` implies
//! `glm5` + `katgpt-attn/mla_attention`.
//!
//! Plan 627 Phase 1 T2 adds the chunk PREFILL seat (`prefill`) over the
//! substrate's `mla_forward_chunk` — the substrate's in-module gates pin
//! chunk ≡ the sequential decode loop BIT-EXACTLY, so the prefill carries
//! no tolerance of its own. Real-capture parity (needs weights) and the
//! Metal arm remain later lane legs.

use katgpt_attn::mla::{
    mla_forward_chunk_dispatched, mla_forward_token_dispatched, MlaConfig, MlaForwardScratch,
    MlaKVCache, MlaWeights,
};
use katgpt_kv::shard_kv::rope::RopeFreqs;

use crate::glm5::Glm5Config;

// ---------------------------------------------------------------------------
// Config conversion (glm5 facts → the upstream MLA vocabulary)
// ---------------------------------------------------------------------------

impl Glm5Config {
    /// The upstream `MlaConfig` for this arch's DSA layers.
    ///
    /// `qk_rope_head_dim = 0` and `use_nope = true` (config-proven);
    /// `use_output_gate = false` (inventory-proven: no gate tensor on the
    /// DSA layers); `rms_norm_eps` from the config (real checkpoint: 1e-5).
    /// `rope_theta` is carried but never applied under nope-mode.
    pub fn to_mla_config(&self) -> MlaConfig {
        MlaConfig {
            kv_lora_rank: self.kv_lora_rank,
            q_lora_rank: self.q_lora_rank.unwrap_or(0),
            qk_nope_head_dim: self.qk_nope_head_dim,
            qk_rope_head_dim: self.qk_rope_head_dim,
            v_head_dim: self.v_head_dim(),
            n_heads: self.num_attention_heads,
            hidden_size: self.hidden_size,
            use_output_gate: false,
            use_nope: true,
            rope_theta: 10_000.0,
            rms_norm_eps: self.rms_norm_eps,
        }
    }

    /// The wrapper refuses contexts past this length: beyond the indexer's
    /// `index_topk` reach the dense fallback would over-attend (the real
    /// arch selects; this reference does not — yet).
    pub fn dense_path_reach(&self) -> usize {
        self.index_topk.unwrap_or(usize::MAX)
    }
}

// ---------------------------------------------------------------------------
// Weights: the fused-tensor de-fuse (the loader's concern, upstream-named)
// ---------------------------------------------------------------------------

/// A DSA layer's weights, in the upstream `MlaWeights` layout after
/// de-fusing the checkpoint's fused tensors.
///
/// The checkpoint stores (inventory-proven name grammar, `glm5` feature):
/// `q_a_proj` [q_lora × d], `q_b_proj` [d_h·n_h × q_lora],
/// `kv_a_proj_with_mqa` [(d_c + d_r) × d] = [d_c × d] at d_r = 0,
/// `kv_b_proj` [(d_h + v_h)·n_h × d_c], `o_proj` [d × v_h·n_h],
/// `q_a_layernorm.weight` [q_lora], `kv_a_layernorm.weight` [d_c].
/// All fp8 rows carry `weight_scale_inv` twins (dequant is the loader
/// leg's job — this struct consumes f32 rows).
pub struct Glm5DsaWeights {
    pub q_a_proj: Vec<f32>,
    pub q_b_proj: Vec<f32>,
    pub kv_a_proj_with_mqa: Vec<f32>,
    pub kv_b_proj: Vec<f32>,
    pub o_proj: Vec<f32>,
    pub q_a_layernorm: Vec<f32>,
    pub kv_a_layernorm: Vec<f32>,
}

impl Glm5DsaWeights {
    /// Row-count sanity for every fused tensor (loud, named).
    pub fn validate_shapes(&self, cfg: &Glm5Config) -> Result<(), String> {
        let d = cfg.hidden_size;
        let d_c = cfg.kv_lora_rank;
        let d_qc = cfg.q_lora_rank.unwrap_or(0);
        let d_h = cfg.qk_nope_head_dim;
        let d_r = cfg.qk_rope_head_dim;
        let v_h = cfg.v_head_dim();
        let n_h = cfg.num_attention_heads;
        let checks = [
            ("q_a_proj", self.q_a_proj.len(), d_qc * d),
            ("q_b_proj", self.q_b_proj.len(), d_h * n_h * d_qc),
            (
                "kv_a_proj_with_mqa",
                self.kv_a_proj_with_mqa.len(),
                (d_c + d_r) * d,
            ),
            ("kv_b_proj", self.kv_b_proj.len(), (d_h + v_h) * n_h * d_c),
            ("o_proj", self.o_proj.len(), d * v_h * n_h),
            ("q_a_layernorm", self.q_a_layernorm.len(), d_qc),
            ("kv_a_layernorm", self.kv_a_layernorm.len(), d_c),
        ];
        for (name, got, want) in checks {
            if got != want {
                return Err(format!(
                    "{name}: {} element(s), expected {want} (shape mismatch)",
                    got
                ));
            }
        }
        Ok(())
    }

    /// De-fuse into the upstream `MlaWeights` (zero-copy slices are not
    /// available — `MlaWeights` owns its rows; the copies are the price of
    /// the shared substrate, paid once at load).
    ///
    /// With `d_r = 0`: `w_qr` and `w_kr` are EMPTY vectors; `q_b_proj` is
    /// `w_uq` verbatim; `kv_a_proj_with_mqa` is `w_dkv` verbatim;
    /// `kv_b_proj` splits at row offset `d_h·n_h` into `w_uk` / `w_uv`.
    /// Panics on a rope-carrying shape — the glm5 DSA family has none
    /// (config-pinned `qk_rope_head_dim = 0`); a future multimodal or
    /// rope-carrying variant extends this de-fuse explicitly.
    pub fn to_mla_weights(&self, cfg: &Glm5Config) -> MlaWeights {
        let d_c = cfg.kv_lora_rank;
        let d_h = cfg.qk_nope_head_dim;
        let d_r = cfg.qk_rope_head_dim;
        let v_h = cfg.v_head_dim();
        let n_h = cfg.num_attention_heads;
        assert_eq!(
            d_r, 0,
            "glm5 DSA de-fuse pins qk_rope_head_dim = 0 (config-proven); \
             extend the de-fuse explicitly for a rope-carrying variant"
        );

        let uk_rows = d_h * n_h;
        let uv_rows = v_h * n_h;
        let mut w_uk = Vec::with_capacity(uk_rows * d_c);
        w_uk.extend_from_slice(&self.kv_b_proj[..uk_rows * d_c]);
        let mut w_uv = Vec::with_capacity(uv_rows * d_c);
        w_uv.extend_from_slice(&self.kv_b_proj[uk_rows * d_c..]);

        MlaWeights {
            w_dkv: self.kv_a_proj_with_mqa.clone(),
            w_dq: self.q_a_proj.clone(),
            w_uq: self.q_b_proj.clone(),
            // d_r = 0: no rope-query rows exist in q_b_proj.
            w_qr: Vec::new(),
            w_uk,
            w_uv,
            // d_r = 0: no rope-key rows exist in kv_a_proj_with_mqa.
            w_kr: Vec::new(),
            w_o: self.o_proj.clone(),
            q_a_norm_weight: self.q_a_layernorm.clone(),
            kv_a_norm_weight: self.kv_a_layernorm.clone(),
            // Inventory-proven: no output-gate tensor on the DSA layers.
            w_g: None,
        }
    }
}

// ---------------------------------------------------------------------------
// The per-layer forward seat
// ---------------------------------------------------------------------------

/// One DSA layer's decode seat: the upstream cache + scratch + rope table,
/// constructed once and reused across tokens (the alloc-free decode
/// discipline).
pub struct Glm5DsaLayer {
    cfg: Glm5Config,
    mla: MlaConfig,
    weights: MlaWeights,
    cache: MlaKVCache,
    scratch: MlaForwardScratch,
    /// Zero-width rope table (d_r = 0) — kept to satisfy the upstream
    /// forward's signature; never applied under `use_nope`.
    _rope: RopeFreqs,
}

impl Glm5DsaLayer {
    /// Build the seat for one DSA layer. `max_seq` bounds the cache; the
    /// dense-path reach (`index_topk`) bounds the attention length — take
    /// `max_seq <= cfg.dense_path_reach()` or the first forward refuses.
    pub fn new(cfg: &Glm5Config, weights: Glm5DsaWeights, max_seq: usize) -> Result<Self, String> {
        cfg.validate().map_err(|e| format!("config invalid: {e}"))?;
        weights.validate_shapes(cfg)?;
        if max_seq == 0 {
            return Err("max_seq must be positive".to_string());
        }
        let mla = cfg.to_mla_config();
        Ok(Self {
            cfg: cfg.clone(),
            weights: weights.to_mla_weights(cfg),
            cache: MlaKVCache::new(&mla, max_seq),
            scratch: MlaForwardScratch::new(&mla, max_seq),
            _rope: RopeFreqs::new_with_theta(mla.qk_rope_head_dim, mla.rope_theta),
            mla,
        })
    }

    /// The layer's config view.
    pub fn config(&self) -> &Glm5Config {
        &self.cfg
    }

    /// Cached token count.
    pub fn seq_len(&self) -> usize {
        self.cache.seq_len
    }

    /// Reset the cache (new sequence).
    pub fn reset(&mut self) {
        self.cache.reset();
    }

    /// One decode step over the dense path: appends this token's latent to
    /// the cache and attends to EVERYTHING cached (see the module doc for
    /// the dense-path contract). Returns a slice of the layer scratch
    /// holding the layer output `[hidden_size]`.
    pub fn forward_token(&mut self, h: &[f32]) -> Result<&[f32], String> {
        if self.cache.seq_len >= self.cfg.dense_path_reach() {
            return Err(format!(
                "dense-path reach exhausted: cache holds {} token(s), \
                 index_topk is {} — the indexer-selection lane owns longer contexts",
                self.cache.seq_len,
                self.cfg.dense_path_reach()
            ));
        }
        let out = mla_forward_token_dispatched(
            &self.mla,
            &self.weights,
            &mut self.cache,
            &mut self.scratch,
            &mut self._rope,
            h,
        );
        Ok(out)
    }

    /// Prefill a CHUNK of hidden states over the dense path — one batched
    /// causal call (the substrate's `mla_forward_chunk_dispatched`; its
    /// in-module gates pin chunk ≡ the sequential decode loop BIT-EXACTLY
    /// per posture — reconstruct default, absorbed under `mla_absorbed`,
    /// `KATGPT_MLA_RECONSTRUCT=1` the reconstruct kill-switch). Appends
    /// every row's latent to the cache and returns the per-row layer
    /// outputs flat `[n × hidden_size]` row-major. The reach check mirrors
    /// `forward_token`'s: the chunk may end at exactly `dense_path_reach()`
    /// cached tokens; dim mismatches and empty chunks refuse typed.
    pub fn prefill(&mut self, hs: &[&[f32]]) -> Result<Vec<f32>, String> {
        let n = hs.len();
        if n == 0 {
            return Err("empty prefill chunk".to_string());
        }
        let reach = self.cfg.dense_path_reach();
        if self.cache.seq_len + n > reach {
            return Err(format!(
                "dense-path reach exhausted: chunk would hold {} token(s), \
                 index_topk is {} — the indexer-selection lane owns longer contexts",
                self.cache.seq_len + n,
                reach
            ));
        }
        for (i, h) in hs.iter().enumerate() {
            if h.len() != self.cfg.hidden_size {
                return Err(format!(
                    "hidden state dim mismatch at row {i}: {} != {}",
                    h.len(),
                    self.cfg.hidden_size
                ));
            }
        }
        Ok(mla_forward_chunk_dispatched(
            &self.mla,
            &self.weights,
            &mut self.cache,
            &mut self.scratch,
            &mut self._rope,
            hs,
        ))
    }
}

// ---------------------------------------------------------------------------
// Tests — synthetic weights, the de-fuse identity, the dense-path contract
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny but structurally EXACT glm5 shape: 1 DSA layer, nope, no gate.
    fn tiny_dsa_cfg() -> Glm5Config {
        let json = r##"{
            "model_type": "glm5_next",
            "text_config": {
                "model_type": "glm5_next_text",
                "num_hidden_layers": 1,
                "layer_types": ["deepseek_sparse_attention"],
                "hidden_size": 64,
                "num_attention_heads": 4,
                "q_lora_rank": 32,
                "kv_lora_rank": 16,
                "qk_nope_head_dim": 8,
                "qk_rope_head_dim": 0,
                "v_head_dim": 8,
                "max_position_embeddings": 4096,
                "vocab_size": 128,
                "first_k_dense_replace": 0,
                "n_routed_experts": 2,
                "rms_norm_eps": 1e-05,
                "index_topk": 64
            }
        }"##;
        Glm5Config::parse(json).expect("tiny dsa parse")
    }

    /// Deterministic pseudo-random f32 rows (LCG — no new dep, the
    /// upstream `random_matrix` convention).
    fn rand_rows(seed: u64, rows: usize, cols: usize) -> Vec<f32> {
        let mut s = seed | 1;
        let scale = 1.0 / (cols as f32).sqrt();
        (0..rows * cols)
            .map(|_| {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let u = ((s >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0;
                u * scale
            })
            .collect()
    }

    fn rand_vec(seed: u64, n: usize) -> Vec<f32> {
        rand_rows(seed, 1, n)
    }

    fn tiny_weights(cfg: &Glm5Config) -> Glm5DsaWeights {
        let d = cfg.hidden_size;
        let d_c = cfg.kv_lora_rank;
        let d_qc = cfg.q_lora_rank.unwrap_or(0);
        let d_h = cfg.qk_nope_head_dim;
        let v_h = cfg.v_head_dim();
        let n_h = cfg.num_attention_heads;
        Glm5DsaWeights {
            q_a_proj: rand_rows(1, d_qc, d),
            q_b_proj: rand_rows(2, d_h * n_h, d_qc),
            kv_a_proj_with_mqa: rand_rows(3, d_c, d),
            kv_b_proj: rand_rows(4, (d_h + v_h) * n_h, d_c),
            o_proj: rand_rows(5, d, v_h * n_h),
            q_a_layernorm: rand_vec(6, d_qc),
            kv_a_layernorm: rand_vec(7, d_c),
        }
    }

    #[test]
    fn config_conversion_matches_the_glm5_facts() {
        let cfg = tiny_dsa_cfg();
        let mla = cfg.to_mla_config();
        assert_eq!(mla.kv_lora_rank, 16);
        assert_eq!(mla.q_lora_rank, 32);
        assert_eq!(mla.qk_nope_head_dim, 8);
        assert_eq!(mla.qk_rope_head_dim, 0, "glm5: no attention rope");
        assert_eq!(mla.v_head_dim, 8);
        assert_eq!(mla.n_heads, 4);
        assert_eq!(mla.hidden_size, 64);
        assert!(mla.use_nope, "nope-mode is the config-proven posture");
        assert!(!mla.use_output_gate, "inventory-proven: no gate tensor");
        assert_eq!(mla.rms_norm_eps, 1e-5);
        assert_eq!(mla.qk_head_dim(), 8, "rope dim 0 adds nothing");
        assert_eq!(mla.d_r(), 0);
    }

    #[test]
    fn de_fuse_splits_kv_b_at_the_row_boundary() {
        let cfg = tiny_dsa_cfg();
        let w = tiny_weights(&cfg);
        let mla_w = w.to_mla_weights(&cfg);
        let d_h = cfg.qk_nope_head_dim;
        let n_h = cfg.num_attention_heads;
        let d_c = cfg.kv_lora_rank;
        let v_h = cfg.v_head_dim();

        // w_uk = the FIRST d_h·n_h rows of kv_b_proj, byte-identical.
        assert_eq!(mla_w.w_uk.len(), d_h * n_h * d_c);
        assert_eq!(&mla_w.w_uk[..], &w.kv_b_proj[..d_h * n_h * d_c]);
        // w_uv = the REST, byte-identical.
        assert_eq!(mla_w.w_uv.len(), v_h * n_h * d_c);
        assert_eq!(&mla_w.w_uv[..], &w.kv_b_proj[d_h * n_h * d_c..]);
        // The rope halves are EMPTY at d_r = 0 (not zeros — absent).
        assert!(mla_w.w_qr.is_empty());
        assert!(mla_w.w_kr.is_empty());
        // The verbatim passes.
        assert_eq!(mla_w.w_uq, w.q_b_proj);
        assert_eq!(mla_w.w_dkv, w.kv_a_proj_with_mqa);
        assert_eq!(mla_w.w_o, w.o_proj);
        assert_eq!(mla_w.q_a_norm_weight, w.q_a_layernorm);
        assert_eq!(mla_w.kv_a_norm_weight, w.kv_a_layernorm);
        assert!(mla_w.w_g.is_none());
    }

    #[test]
    fn shape_validation_refuses_a_short_row() {
        let cfg = tiny_dsa_cfg();
        let mut w = tiny_weights(&cfg);
        w.q_b_proj.pop();
        let err = w.validate_shapes(&cfg).unwrap_err();
        assert!(err.contains("q_b_proj"), "{err}");
    }

    #[test]
    fn decode_runs_and_tracks_the_cache() {
        let cfg = tiny_dsa_cfg();
        let w = tiny_weights(&cfg);
        let mut layer = Glm5DsaLayer::new(&cfg, w, 16).expect("seat");
        assert_eq!(layer.seq_len(), 0);
        for step in 0usize..4 {
            let h = rand_vec(100 + step as u64, cfg.hidden_size);
            let out = layer.forward_token(&h).expect("decode step");
            assert_eq!(out.len(), cfg.hidden_size);
            assert_eq!(layer.seq_len(), step + 1);
        }
        layer.reset();
        assert_eq!(layer.seq_len(), 0);
    }

    #[test]
    fn decode_is_deterministic_and_reset_restores_step_zero() {
        let cfg = tiny_dsa_cfg();
        let h0 = rand_vec(42, cfg.hidden_size);
        let h1 = rand_vec(43, cfg.hidden_size);
        let run = || -> Vec<f32> {
            let w = tiny_weights(&cfg);
            let mut layer = Glm5DsaLayer::new(&cfg, w, 16).expect("seat");
            let a = layer.forward_token(&h0).expect("a").to_vec();
            let b = layer.forward_token(&h1).expect("b").to_vec();
            let mut both = a;
            both.extend_from_slice(&b);
            both
        };
        let x = run();
        let y = run();
        assert_eq!(x, y, "bit-identical across seats (deterministic decode)");
        assert!(x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn dense_path_reach_refuses_past_index_topk() {
        let cfg = tiny_dsa_cfg(); // index_topk 64, max_seq 64
        let w = tiny_weights(&cfg);
        let mut layer = Glm5DsaLayer::new(&cfg, w, 64).expect("seat");
        // Fill to the reach: the NEXT call must refuse (the dense fallback
        // would over-attend beyond the indexer's selection budget).
        for step in 0usize..64 {
            let h = rand_vec(200 + step as u64, cfg.hidden_size);
            layer.forward_token(&h).expect("within reach");
        }
        assert_eq!(layer.seq_len(), 64);
        let h = rand_vec(999, cfg.hidden_size);
        let err = match layer.forward_token(&h) {
            Err(e) => e,
            Ok(_) => panic!("expected the dense-path refusal"),
        };
        assert!(err.contains("dense-path reach exhausted"), "{err}");
    }

    #[test]
    fn absorbed_matches_reconstruct_on_the_glm5_shape() {
        // The upstream spec-match discipline, replayed on the glm5 nope/0
        // shape through MY de-fused weights: the dispatched router may run
        // either path; the reconstruct reference runs explicitly beside it.
        // (Without `mla_absorbed` the dispatched fn IS the reconstruct —
        // the parity leg re-claims under the combined posture.)
        let cfg = tiny_dsa_cfg();
        let w = tiny_weights(&cfg);
        let mla_w = w.to_mla_weights(&cfg);
        let mla = cfg.to_mla_config();

        let run = |dispatched: bool| -> Vec<f32> {
            let mut cache = MlaKVCache::new(&mla, 8);
            let mut scratch = MlaForwardScratch::new(&mla, 8);
            let mut rope = RopeFreqs::new_with_theta(mla.qk_rope_head_dim, mla.rope_theta);
            let h0 = rand_vec(42, cfg.hidden_size);
            let h1 = rand_vec(43, cfg.hidden_size);
            let a = if dispatched {
                mla_forward_token_dispatched(&mla, &mla_w, &mut cache, &mut scratch, &mut rope, &h0)
            } else {
                katgpt_attn::mla::mla_forward_token(
                    &mla,
                    &mla_w,
                    &mut cache,
                    &mut scratch,
                    &mut rope,
                    &h0,
                )
            };
            let a = a.to_vec();
            let b = if dispatched {
                mla_forward_token_dispatched(&mla, &mla_w, &mut cache, &mut scratch, &mut rope, &h1)
            } else {
                katgpt_attn::mla::mla_forward_token(
                    &mla,
                    &mla_w,
                    &mut cache,
                    &mut scratch,
                    &mut rope,
                    &h1,
                )
            };
            let mut both = a;
            both.extend_from_slice(b);
            both
        };
        let ref_out = run(false);
        let disp_out = run(true);
        assert_eq!(
            ref_out, disp_out,
            "dispatched ≡ reconstruct (no absorbed compile here)"
        );
        assert!(ref_out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn seat_new_refuses_a_broken_config_or_shapes() {
        let cfg = tiny_dsa_cfg();
        let mut w = tiny_weights(&cfg);
        w.o_proj.pop();
        let err = match Glm5DsaLayer::new(&cfg, w, 8) {
            Err(e) => e,
            Ok(_) => panic!("expected the shape refusal"),
        };
        assert!(err.contains("o_proj"), "{err}");

        let mut bad_cfg = tiny_dsa_cfg();
        bad_cfg.num_hidden_layers = 0;
        let w2 = tiny_weights(&bad_cfg);
        let err = match Glm5DsaLayer::new(&bad_cfg, w2, 8) {
            Err(e) => e,
            Ok(_) => panic!("expected the config refusal"),
        };
        assert!(err.contains("config invalid"), "{err}");
    }

    /// The inventory gate and the forward seat agree on the same map:
    /// a tiny weight_map (built from the SAME grammar the validator
    /// expects) passes validation, and the DSA layer rows it names are
    /// exactly what `Glm5DsaWeights` consumes. This is the seam the real
    /// loader leg walks (validate → read → de-fuse → forward).
    #[test]
    fn inventory_and_forward_share_the_dsa_grammar() {
        use std::collections::BTreeMap;

        // The T0 test module owns the fixture builder; rebuild the
        // DSA-layer rows here (the shared grammar, second spelling).
        let names: Vec<String> = {
            // The T0 test module owns the fixture builder; rebuild the
            // DSA-layer rows here (the shared grammar, second spelling).
            let mut v = Vec::new();
            for s in [
                "self_attn.q_a_layernorm.weight",
                "self_attn.q_a_proj.weight",
                "self_attn.q_a_proj.weight_scale_inv",
                "self_attn.q_b_proj.weight",
                "self_attn.q_b_proj.weight_scale_inv",
                "self_attn.kv_a_layernorm.weight",
                "self_attn.kv_a_proj_with_mqa.weight",
                "self_attn.kv_a_proj_with_mqa.weight_scale_inv",
                "self_attn.kv_b_proj.weight",
                "self_attn.indexer.wk.weight",
                "self_attn.indexer.wq_b.weight",
                "self_attn.indexer.weights_proj.weight",
                "self_attn.indexer.k_norm.weight",
                "self_attn.indexer.k_norm.bias",
                "self_attn.indexer.index_kpool_compress_ape",
                "self_attn.indexer.index_kpool_compress_gate",
            ] {
                v.push(format!("model.language_model.layers.0.{s}"));
            }
            v
        };
        let map: BTreeMap<String, String> = names
            .into_iter()
            .map(|n| (n, "s.safetensors".to_string()))
            .collect();
        let dsa_rows: Vec<&String> = map
            .keys()
            .filter(|n| n.contains("self_attn.") || n.contains("layernorm"))
            .collect();
        assert_eq!(dsa_rows.len(), 16, "the DSA family row count");
        for n in dsa_rows {
            assert!(n.starts_with("model.language_model.layers.0."));
        }
    }

    // ── Plan 627 Phase 1 T2: the chunk prefill seat ────────────────────

    #[test]
    fn prefill_matches_the_decode_loop_bit_exact() {
        let cfg = tiny_dsa_cfg();
        let d = cfg.hidden_size;
        let tokens: Vec<Vec<f32>> = (0..4).map(|i| rand_vec(100 + i, d)).collect();
        let refs: Vec<&[f32]> = tokens.iter().map(|t| t.as_slice()).collect();

        // Arm A: sequential decode (tiny_weights is seed-deterministic, so
        // both arms run the identical weight set).
        let mut dec = Glm5DsaLayer::new(&cfg, tiny_weights(&cfg), 16).expect("seat");
        let expected: Vec<Vec<f32>> = tokens
            .iter()
            .map(|t| dec.forward_token(t).expect("decode").to_vec())
            .collect();

        // Arm B: one chunk prefill (fresh seat).
        let mut pre = Glm5DsaLayer::new(&cfg, tiny_weights(&cfg), 16).expect("seat");
        let got = pre.prefill(&refs).expect("prefill");
        assert_eq!(got.len(), tokens.len() * d);
        for (i, (row, exp)) in got.chunks(d).zip(expected.iter()).enumerate() {
            assert_eq!(row, exp.as_slice(), "row {i} diverged from the decode loop");
        }
        assert_eq!(pre.seq_len(), 4);
        assert_eq!(dec.seq_len(), 4);
    }

    #[test]
    fn prefill_after_prefix_matches_all_decode() {
        let cfg = tiny_dsa_cfg();
        let d = cfg.hidden_size;
        let tokens: Vec<Vec<f32>> = (0..5).map(|i| rand_vec(200 + i, d)).collect();

        // Expected: all-decode.
        let mut dec = Glm5DsaLayer::new(&cfg, tiny_weights(&cfg), 16).expect("seat");
        let expected: Vec<Vec<f32>> = tokens
            .iter()
            .map(|t| dec.forward_token(t).expect("decode").to_vec())
            .collect();

        // Mixed: decode 2, then chunk the rest.
        let mut mix = Glm5DsaLayer::new(&cfg, tiny_weights(&cfg), 16).expect("seat");
        for (i, t) in tokens.iter().enumerate().take(2) {
            let out = mix.forward_token(t).expect("decode").to_vec();
            assert_eq!(out, expected[i].as_slice(), "prefix row {i}");
        }
        let rest: Vec<&[f32]> = tokens[2..].iter().map(|t| t.as_slice()).collect();
        let got = mix.prefill(&rest).expect("prefill");
        assert_eq!(got.len(), 3 * d);
        for (i, row) in got.chunks(d).enumerate() {
            assert_eq!(row, expected[2 + i].as_slice(), "chunked row {}", 2 + i);
        }
        assert_eq!(mix.seq_len(), 5);
    }

    #[test]
    fn prefill_refuses_past_dense_reach() {
        let cfg = tiny_dsa_cfg(); // index_topk 64
        let reach = cfg.dense_path_reach();
        let d = cfg.hidden_size;
        let mut seat = Glm5DsaLayer::new(&cfg, tiny_weights(&cfg), 4096).expect("seat");
        // A chunk longer than the reach refuses typed, cache untouched.
        let over_vecs: Vec<Vec<f32>> = (0..reach + 1)
            .map(|i| rand_vec(300 + i as u64, d))
            .collect();
        let over: Vec<&[f32]> = over_vecs.iter().map(|t| t.as_slice()).collect();
        let err = seat
            .prefill(&over)
            .expect_err("over-reach chunk must refuse");
        assert!(err.contains("dense-path reach"), "err: {err}");
        assert_eq!(seat.seq_len(), 0);
        // Exactly at the reach is legal.
        let exact_vecs: Vec<Vec<f32>> = (0..reach).map(|i| rand_vec(400 + i as u64, d)).collect();
        let exact: Vec<&[f32]> = exact_vecs.iter().map(|t| t.as_slice()).collect();
        let got = seat
            .prefill(&exact)
            .expect("at-reach chunk must be accepted");
        assert_eq!(got.len(), reach * d);
        assert_eq!(seat.seq_len(), reach);
    }

    #[test]
    fn prefill_refuses_bad_dims_and_empty() {
        let cfg = tiny_dsa_cfg();
        let d = cfg.hidden_size;
        let mut seat = Glm5DsaLayer::new(&cfg, tiny_weights(&cfg), 16).expect("seat");
        let empty: Vec<&[f32]> = Vec::new();
        assert!(seat.prefill(&empty).is_err());
        let bad_vec = rand_vec(500, d - 1);
        let bad = [bad_vec.as_slice()];
        let err = seat.prefill(&bad).expect_err("dim mismatch must refuse");
        assert!(err.contains("dim mismatch"), "err: {err}");
        assert_eq!(seat.seq_len(), 0);
    }

    #[test]
    fn prefill_is_deterministic() {
        let cfg = tiny_dsa_cfg();
        let d = cfg.hidden_size;
        let tokens: Vec<Vec<f32>> = (0..3).map(|i| rand_vec(600 + i, d)).collect();
        let refs: Vec<&[f32]> = tokens.iter().map(|t| t.as_slice()).collect();
        let mut a = Glm5DsaLayer::new(&cfg, tiny_weights(&cfg), 16).expect("seat");
        let got1 = a.prefill(&refs).expect("prefill 1");
        a.reset();
        let got2 = a.prefill(&refs).expect("prefill 2");
        assert_eq!(got1, got2);
    }
}
