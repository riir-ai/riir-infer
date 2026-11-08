//! Issue 1183 T2 — the DSpine CPU-oracle harness.
//!
//! Two modes:
//!
//! 1. `--fixtures <out-dir>` — synthetic PIECEWISE fixture dumps (the G1
//!    discipline: piecewise fixtures FIRST — the Bench-952 lesson: e2e
//!    parity alone hid the uzu qkvg divergence for a full session). Writes
//!    bf16-valued f32 `.bin` inputs AND our oracle's outputs for three
//!    pieces: (a) bridge read+codes, (b) message+write+fused-norm, (d) the
//!    serial walk (tables→idx/probs→chain). The Python/MLX side replays the
//!    same inputs through `dspine_mlx` and diffs inside the stated
//!    tolerances (bit-exact on the scalar chains; ~1e-5 relative where the
//!    vendor's matmul reduction order is opaque).
//!
//! 2. default — the real-checkpoint shape walk: load
//!    `tencent/Qwen3.8-27B-DSpine-b16` (blake3-pinned at load, the A1/A8
//!    law), build the sender tables over a synthetic bf16 lm_head stand-in
//!    (the REAL head dequant is the T5 harness's seam), run one block
//!    forward, and assert the invariants (chain tokens ∈ candidates;
//!    confidences ∈ (0, 1]; anchor message zero).
//!
//! Run: cargo run --release -p riir-infer-gpu --features qwen38_dspine \
//!        --bin qwen38_dspine_t2 -- [--fixtures /tmp/dspine_fixtures] \
//!        [--checkpoint /Users/katopz/uzu27b/models/Qwen3.8-27B-DSpine-b16]

#![cfg(feature = "qwen38_dspine")]

use std::path::Path;

use riir_infer_gpu::qwen38_dspine::{
    DspineConfig, DspineDrafter, bf16_round, build_serial_tables, chain_walk,
    load_dspine_safetensors, parse_config_json, piece_message_write, piece_read_codes,
    serial_walk_core,
};

/// The Issue-1183 pin (T1; b3sum + python-blake3 agree).
const EXPECTED_CKPT_BLAKE3_PREFIX: &str = "53d18516";
const DEFAULT_CKPT_DIR: &str = "/Users/katopz/uzu27b/models/Qwen3.8-27B-DSpine-b16";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut fixtures_dir = String::new();
    let mut ckpt_dir = DEFAULT_CKPT_DIR.to_string();
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--fixtures" => {
                i += 1;
                fixtures_dir = args[i].clone();
            }
            "--checkpoint" => {
                i += 1;
                ckpt_dir = args[i].clone();
            }
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(2);
            }
        }
        i += 1;
    }

    if !fixtures_dir.is_empty() {
        write_fixtures(Path::new(&fixtures_dir));
    } else {
        shape_walk(Path::new(&ckpt_dir));
    }
}

// ── deterministic synthetic data (a fixed LCG — no external RNG dep) ──

struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        // Numerical Recipes LCG; map to [-1, 1).
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (self.0 >> 33) as u32;
        (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
    /// bf16-representable values only — bit-exact fixture comparisons are
    /// then possible on both sides.
    fn next_bf16(&mut self) -> f32 {
        bf16_round(self.next_f32())
    }
}

fn dump_bin(out: &Path, name: &str, v: &[f32]) {
    std::fs::write(out.join(format!("{name}.f32.bin")), bytemuck::cast_slice(v))
        .unwrap_or_else(|e| panic!("write {name}: {e}"));
}

// ── mode 1: the piecewise fixture dumps (inputs + OUR outputs) ──

#[allow(clippy::too_many_lines)]
fn write_fixtures(out: &Path) {
    std::fs::create_dir_all(out).unwrap_or_else(|e| panic!("mkdir {out:?}: {e}"));
    let cfg = DspineConfig::default(); // the REAL shapes; tensors synthetic
    let (e, c, r) = (cfg.n_embd, cfg.code_size, cfg.rank);
    let k = cfg.head_topk;
    let t = 8usize; // a small served block
    let gamma = t - 1;
    // A synthetic anchor INSIDE the codebook slice (piece (a)/(d) index it;
    // the real 151644-id anchor is the shape walk's, over the full tables).
    let anchor = 3u32;
    let mut rng = Lcg(0x1183_0001);

    // Shared synthetic weights (bf16-valued, like the checkpoint's).
    let read_w: Vec<f32> = (0..c * e).map(|_| rng.next_bf16()).collect();
    let write_w: Vec<f32> = (0..e * r).map(|_| rng.next_bf16()).collect();
    let gate_b: Vec<f32> = (0..r).map(|_| rng.next_bf16()).collect();
    let norm_w: Vec<f32> = (0..e).map(|_| rng.next_bf16()).collect();
    let cb_rows = 4096usize; // a codebook slice (anchor + a few candidates)
    let codebook: Vec<f32> = (0..cb_rows * c).map(|_| rng.next_bf16()).collect();

    // (a) read+codes over t rows (row 0 = anchor).
    let raw: Vec<f32> = (0..t * e).map(|_| rng.next_bf16()).collect();
    let (rms_a, codes) = piece_read_codes(&read_w, &raw, cfg.bridge_eps, &codebook, anchor, c, t);
    dump_bin(out, "a_raw", &raw);
    dump_bin(out, "a_read_w", &read_w);
    dump_bin(out, "a_codebook", &codebook);
    dump_bin(out, "a_rms_out", &rms_a);
    dump_bin(out, "a_codes_out", &codes);

    // (b) message+write: packed halves per row (q|s|v = 3r) with the
    //     predecessor chain row i-1 (row 0 anchored).
    let packed: Vec<f32> = (0..t * 3 * r).map(|_| rng.next_bf16()).collect();
    let mut message_all = Vec::with_capacity(t * r);
    let mut out_all = Vec::with_capacity(t * e);
    let mut normed_all = Vec::with_capacity(t * e);
    for i in 0..t {
        let prow = &packed[i * 3 * r..(i + 1) * 3 * r];
        let prev_prow = if i == 0 {
            &packed[..3 * r]
        } else {
            &packed[(i - 1) * 3 * r..i * 3 * r]
        };
        let (m, o, n) = piece_message_write(
            prow,
            prev_prow,
            &gate_b,
            &write_w,
            rms_a[i],
            &raw[i * e..(i + 1) * e],
            i % t == 0,
            &norm_w,
            cfg.rms_eps,
        );
        message_all.extend_from_slice(&m);
        out_all.extend_from_slice(&o);
        normed_all.extend_from_slice(&n);
    }
    dump_bin(out, "b_packed_in", &packed);
    dump_bin(out, "b_write_w", &write_w);
    dump_bin(out, "b_gate_b", &gate_b);
    dump_bin(out, "b_raw_in", &raw);
    dump_bin(out, "b_norm_w", &norm_w);
    dump_bin(out, "b_message_out", &message_all);
    dump_bin(out, "b_out_out", &out_all);
    dump_bin(out, "b_normed_out", &normed_all);

    // (d) the serial walk: tables over the synthetic codebook slice + a
    //     synthetic lm_head slice (candidates PRE-COMPUTED so the top-k tie
    //     rule stays under its own unit test, not the golden diff).
    let v_slice = cb_rows; // candidate ids index the codebook slice
    let lm_head: Vec<f32> = (0..v_slice * e).map(|_| rng.next_bf16()).collect();
    // pv_w = [G_p; A] [2r, c]; TG needs write_w_last (= write_w here).
    let pred_gate: Vec<f32> = (0..r * c).map(|_| rng.next_bf16()).collect();
    let prev_w: Vec<f32> = (0..r * c).map(|_| rng.next_bf16()).collect();
    let mut pv_w = Vec::with_capacity(2 * r * c);
    pv_w.extend_from_slice(&pred_gate);
    pv_w.extend_from_slice(&prev_w);
    let tables = build_serial_tables(&lm_head, &norm_w, &write_w, &pv_w, &codebook, &cfg);
    let candidates: Vec<Vec<u32>> = (0..gamma)
        .map(|row| {
            let base = (row * 7 + 3) as u32;
            (0..k as u32)
                .map(|j| (base + j * 13) % v_slice as u32)
                .collect::<Vec<_>>()
        })
        .collect();
    // The stash-shaped inputs for rows 1..t (the walk reads (nn+1)).
    let mut stash_raw = vec![0.0f32; t * e];
    let mut stash_rms = vec![0.0f32; t];
    let mut stash_q = vec![0.0f32; t * r];
    for i in 0..t {
        stash_raw[i * e..(i + 1) * e].copy_from_slice(&raw[i * e..(i + 1) * e]);
        stash_rms[i] = rms_a[i];
        stash_q[i * r..(i + 1) * r].copy_from_slice(&packed[i * 3 * r..i * 3 * r + r]);
    }
    let (idx, probs) = serial_walk_core(
        &stash_raw,
        &stash_rms,
        &stash_q,
        &candidates,
        anchor,
        &lm_head,
        &norm_w,
        &gate_b,
        &tables,
        &cfg,
    );
    let (tokens, confs) = chain_walk(&candidates, &idx, &probs);
    dump_bin(out, "d_raw_in", &stash_raw);
    dump_bin(out, "d_rms_in", &stash_rms);
    dump_bin(out, "d_q_in", &stash_q);
    dump_bin(out, "d_lm_head_in", &lm_head);
    dump_bin(out, "d_pred_gate_in", &pred_gate);
    dump_bin(out, "d_prev_w_in", &prev_w);
    dump_bin(out, "d_norm_w_in", &norm_w);
    dump_bin(out, "d_write_w_in", &write_w);
    dump_bin(out, "d_gate_b_in", &gate_b);
    dump_bin(out, "d_tg_out", &tables.tg);
    dump_bin(out, "d_tsv_out", &tables.tsv);
    let cand_flat: Vec<u32> = candidates.iter().flatten().copied().collect();
    dump_u32_bin(out, "d_cand_in", &cand_flat);
    let idx_flat: Vec<f32> = idx.iter().flatten().map(|&x| x as f32).collect();
    dump_bin(out, "d_idx_out", &idx_flat);
    dump_bin(
        out,
        "d_probs_out",
        &probs.iter().flatten().copied().collect::<Vec<_>>(),
    );
    dump_bin(
        out,
        "d_tokens_out",
        &tokens.iter().map(|&x| x as f32).collect::<Vec<_>>(),
    );
    dump_bin(out, "d_confs_out", &confs);

    std::fs::write(
        out.join("MANIFEST.md"),
        format!(
            "# DSpine piecewise fixtures (ours-side inputs + outputs; the\n\
             # vendor MLX side replays the inputs through dspine_mlx and diffs\n\
             # against the `_out` files).\n\n\
             - geometry: E={e} C={c} R={r} K={k} T={t} V_SLICE={v_slice} ANCHOR={anchor}\n\
             - (a) piece_read_codes: in a_raw a_read_w a_codebook → a_rms_out a_codes_out (bit-exact)\n\
             - (b) piece_message_write: in b_packed_in b_write_w b_gate_b b_raw_in b_norm_w →\n\
             -   b_message_out b_out_out b_normed_out (bit-exact)\n\
             - (d) serial_walk_core+chain_walk: in d_* → d_idx_out d_probs_out d_tokens_out\n\
             -   d_confs_out (bit-exact on the bf16-valued chain; the TSV/TG matmuls ≤1e-5 rel)\n\
             - all f32 .bin little-endian; idx/tokens are f32-CAST integers\n\
             - fold order: sequential ascending index everywhere (the 940 law)\n"
        ),
    )
    .unwrap_or_else(|e| panic!("manifest: {e}"));
    println!(
        "[t2.fixtures] wrote {} (inputs + our outputs)",
        out.display()
    );
}

fn dump_u32_bin(out: &Path, name: &str, v: &[u32]) {
    std::fs::write(out.join(format!("{name}.u32.bin")), bytemuck::cast_slice(v))
        .unwrap_or_else(|e| panic!("write {name}: {e}"));
}

// ── mode 2: the real-checkpoint shape walk ──

fn shape_walk(ckpt_dir: &Path) {
    let t0 = std::time::Instant::now();
    let st_path = ckpt_dir.join("model.safetensors");
    let cfg_json = std::fs::read_to_string(ckpt_dir.join("config.json"))
        .unwrap_or_else(|e| panic!("config.json: {e}"));
    let cfg = parse_config_json(&cfg_json).unwrap_or_else(|e| panic!("config: {e}"));
    println!(
        "[t2] config ok: layers {} block {} bridge C{} R{} topk {} targets {:?}",
        cfg.n_layer,
        cfg.trained_block_size,
        cfg.code_size,
        cfg.rank,
        cfg.head_topk,
        cfg.target_layer_ids
    );

    // The artifact pin (A1/A8: verify BEFORE any weight is consumed).
    let digest = blake3_file(&st_path);
    println!(
        "[t2] checkpoint blake3 {digest} ({:.1}s)",
        t0.elapsed().as_secs_f32()
    );
    if !digest.starts_with(EXPECTED_CKPT_BLAKE3_PREFIX) {
        eprintln!(
            "[t2] ⛔ blake3 {digest} does not start with the 1183 pin {EXPECTED_CKPT_BLAKE3_PREFIX} — refusing"
        );
        std::process::exit(2);
    }

    let w = load_dspine_safetensors(&st_path, &cfg).unwrap_or_else(|e| panic!("load: {e}"));
    println!("[t2] weights loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // The sender tables over a synthetic bf16 lm_head stand-in (the REAL
    // head dequant is the T5 harness's seam; the builder only needs shapes
    // + folds).
    let mut rng = Lcg(777);
    let lm_head: Vec<f32> = (0..cfg.vocab_size * cfg.n_embd)
        .map(|_| rng.next_bf16())
        .collect();
    let tables = build_serial_tables(
        &lm_head,
        &w.model_norm,
        &w.bridge.feature_write_weights[cfg.n_layer - 1],
        &w.pv_weight(),
        &w.bridge.codebook,
        &cfg,
    );
    println!(
        "[t2] tables built in {:.1}s (TG {} MB, TSV {} MB)",
        t0.elapsed().as_secs_f32(),
        tables.tg.len() * 4 / (1 << 20),
        tables.tsv.len() * 4 / (1 << 20)
    );

    let mut drafter = DspineDrafter::new(cfg.clone(), w);
    drafter
        .set_block_size(8)
        .unwrap_or_else(|e| panic!("block size: {e}"));

    // Synthetic captured target features (T5 wires the REAL captures); the
    // walk validates plumbing + invariants, not acceptance. Amplitude 0.02:
    // realistic hidden-scale magnitudes — unbounded LCG noise overflows to
    // inf through 5 layers and poisons the tail's score mass with NaNs.
    let nf = cfg.target_layer_ids.len();
    let ctx_rows = 32usize;
    let feats: Vec<f32> = (0..ctx_rows * nf * cfg.n_embd)
        .map(|_| rng.next_f32() * 0.02)
        .collect();
    drafter
        .append_context(&feats, 0)
        .unwrap_or_else(|e| panic!("append: {e}"));
    assert_eq!(drafter.context_length(), ctx_rows, "context length");

    // A deterministic stand-in embed table (the real one is the target's
    // Q4_K table — T5's seam).
    let embed_row = |tok: u32| -> Vec<f32> {
        let mut h = blake3::Hasher::new();
        h.update(&tok.to_le_bytes());
        let hex = h.finalize().to_hex();
        (0..cfg.n_embd)
            .map(|i| {
                let b = u32::from_str_radix(&hex[(i % 32)..(i % 32) + 1], 16).unwrap_or(0);
                ((b as f32 / 15.0) * 2.0 - 1.0) * 0.02
            })
            .collect()
    };

    let draft = drafter
        .draft_block(151_644, ctx_rows, &embed_row, &lm_head, &tables)
        .unwrap_or_else(|e| panic!("draft: {e}"));
    let gamma = drafter.served_block_size() - 1;
    assert_eq!(draft.tokens.len(), gamma, "chain length");
    for (i, tok) in draft.tokens.iter().enumerate() {
        assert!(
            draft.candidates[i].contains(tok),
            "chain token {tok} not in candidates at row {i}"
        );
    }
    for (i, &cf) in draft.confidences.iter().enumerate() {
        assert!(
            cf > 0.0 && cf <= 1.0,
            "confidence {cf} out of (0,1] at row {i}"
        );
    }
    let names: Vec<String> = draft.tokens.iter().take(3).map(|t| t.to_string()).collect();
    println!(
        "[t2] shape walk PASS — chain [{}, …] conf {:.3}/{:.3}/{:.3} ({} rows, {:.1}s total)",
        names.join(", "),
        draft.confidences[0],
        draft.confidences[1],
        draft.confidences[2.min(draft.confidences.len() - 1)],
        draft.tokens.len(),
        t0.elapsed().as_secs_f32()
    );
}

fn blake3_file(path: &Path) -> String {
    use blake3::Hasher;
    let mut f = std::fs::File::open(path).unwrap_or_else(|e| panic!("blake3 open {path:?}: {e}"));
    let mut hasher = Hasher::new();
    std::io::copy(&mut f, &mut hasher).expect("blake3 read");
    hasher.finalize().to_hex().to_string()
}
