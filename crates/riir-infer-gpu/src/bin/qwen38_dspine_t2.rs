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
    let mut e2e_dir = String::new();
    let mut ckpt_dir = DEFAULT_CKPT_DIR.to_string();
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--fixtures" => {
                i += 1;
                fixtures_dir = args[i].clone();
            }
            "--e2e" => {
                i += 1;
                e2e_dir = args[i].clone();
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
    } else if !e2e_dir.is_empty() {
        e2e_fixture(Path::new(&e2e_dir), Path::new(&ckpt_dir));
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
    //
    //     DISCRIMINATING SCALES (bench 959): at natural synthetic magnitudes
    //     the walk degenerates — the message half's heavy tails drive score
    //     gaps ~1e6× the confidence scale, the mass underflows to exactly 1.0
    //     everywhere, and the fixture cannot tell the two score formulas
    //     apart (the golden diff would pass trivially — a green zero). Two
    //     knobs keep it live (probe-tuned): pv_w ×2e-7 puts the rms·s term
    //     at ~0.4× base scale so the CANDIDATE base is visible to the argmax
    //     while the message half still perturbs winners; rms ×30 raises the
    //     confidence scale so the mass stays strictly between 1/K and 1
    //     (probs land in ~[0.10, 0.16], top-2 gaps ≤ ~5.3). pv entries are
    //     RE-ROUNDED to bf16 after scaling (inputs stay bf16-valued — the
    //     vendor build rounds pv_w to bf16 before its matmul, so both sides
    //     must see the same bf16 values); rms is an fp32 stash scalar on the
    //     vendor path (no bf16 store), so the scaled rms stays f32.
    let v_slice = cb_rows; // candidate ids index the codebook slice
    let lm_head: Vec<f32> = (0..v_slice * e).map(|_| rng.next_bf16()).collect();
    // pv_w = [G_p; A] [2r, c]; TG needs write_w_last (= write_w here).
    let pv_scale = 2.0e-7f32;
    let pred_gate: Vec<f32> = (0..r * c)
        .map(|_| bf16_round(rng.next_f32() * pv_scale))
        .collect();
    let prev_w: Vec<f32> = (0..r * c)
        .map(|_| bf16_round(rng.next_f32() * pv_scale))
        .collect();
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
    // The stash-shaped inputs for rows 1..t (the walk reads (nn+1)). The
    // rms scale ↑ (see the block comment) keeps the confidence mass live.
    const RMS_SCALE: f32 = 30.0;
    let mut stash_raw = vec![0.0f32; t * e];
    let mut stash_rms = vec![0.0f32; t];
    let mut stash_q = vec![0.0f32; t * r];
    for i in 0..t {
        stash_raw[i * e..(i + 1) * e].copy_from_slice(&raw[i * e..(i + 1) * e]);
        stash_rms[i] = rms_a[i] * RMS_SCALE;
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
    // The degeneracy guard — a fixture whose mass collapsed (all probs 1.0)
    // or whose argmax never moves proves nothing; refuse to write it.
    let distinct_idx: std::collections::HashSet<usize> = idx.iter().flatten().copied().collect();
    let min_prob = probs
        .iter()
        .flatten()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let max_prob = probs
        .iter()
        .flatten()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    if distinct_idx.len() < 4 || !(0.05..0.995).contains(&min_prob) || max_prob > 0.995 {
        panic!(
            "[t2.fixtures] piece (d) degenerate: distinct_idx={} min_prob={min_prob} max_prob={max_prob} — rescale",
            distinct_idx.len()
        );
    }
    println!(
        "[t2.fixtures] piece (d) discriminating: distinct_idx={} probs [{min_prob:.4}, {max_prob:.4}]",
        distinct_idx.len()
    );
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
             - piece (d) scales: pv_w \u{d7}2e-7 (bf16 re-rounded), stash rms \u{d7}30 \u{2014} the\n\
             -   confidence mass must stay non-degenerate or the diff proves nothing\n\
             - (a) piece_read_codes: in a_raw a_read_w a_codebook \u{2192} a_rms_out a_codes_out\n\
             - (b) piece_message_write: in b_packed_in b_write_w b_gate_b b_raw_in b_norm_w \u{2192}\n\
             -   b_message_out b_out_out b_normed_out\n\
             - (d) serial_walk_core+chain_walk: in d_* \u{2192} d_idx_out d_probs_out d_tokens_out\n\
             -   d_confs_out\n\
             - all f32 .bin little-endian; idx/tokens are f32-CAST integers\n\
             - fold order: sequential ascending index everywhere (the 940 law)\n\
             - tolerances (measured, bench 959): idx/tokens bit-exact; bf16-valued\n\
             -   outputs \u{2264}0.2% flips at \u{2264}2 bf16 ULP (the MLX Metal matmul fold vs our\n\
             -   pinned sequential fold); tg/tsv \u{2264}0.05% flips; rms \u{2264}1e-5 rel; probs/confs\n\
             -   \u{2264}1e-3 abs (score-ULP class); MLX argmax ties = first index (probed)\n"
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

// ── mode 3: the E2E main-path golden fixture (plan 492 T0-residue (b)) ──
//
// The Bench-959 lesson: the piecewise replay drove the PIECE functions and
// never the main inline loop — the message-gate hyperbola shipped at the
// oracle birth and survived every piecewise pin. This mode drives the REAL
// `draft_block` main path (5 layers + bridge writes + the serial tail, one
// call, the vendor-exact rounded posture) on the shape-walk's synthetic
// posture and dumps EVERYTHING the vendor mirror needs to rebuild the same
// inputs in MLX: the geometry + seeds (MANIFEST.txt), the two embed rows the
// block consumes (anchor + mask, bf16-rounded — the stand-in table the vendor
// side must inject), the raw f32 context captures (pre-bf16-round), and the
// draft's outputs (tokens / candidates / confidences) plus the tail-trace
// diagnostics (proposal rows + per-row candidates + idx + probs + the last-
// write stash pieces) so a mismatch localizes without a second run.
//
// The vendor arm (`dspine_golden_replay.py --e2e`) regenerates lm_head and
// feats from the SAME LCG (seed 777, same draw order: head first, feats
// second — bit-identical by construction, no 1.25 GB head dump), loads the
// SAME checkpoint through the vendor's own loader, injects the embed rows +
// captures, and drives DSpineCodeBridge.__call__ + RefinementTail.greedy
// end-to-end. The diff bar: chain TOKENS + per-row candidates vs the dump.
// Confidences are disclosed but excluded from the bar (the vendor greedy
// path returns no probs; the score→prob walk is pinned piecewise in (d)).
#[allow(clippy::too_many_lines)]
fn e2e_fixture(out: &Path, ckpt_dir: &Path) {
    let t0 = std::time::Instant::now();
    let st_path = ckpt_dir.join("model.safetensors");
    let cfg_json = std::fs::read_to_string(ckpt_dir.join("config.json"))
        .unwrap_or_else(|e| panic!("config.json: {e}"));
    let cfg = parse_config_json(&cfg_json).unwrap_or_else(|e| panic!("config: {e}"));
    let digest = blake3_file(&st_path);
    println!("[t2.e2e] checkpoint blake3 {digest}");
    if !digest.starts_with(EXPECTED_CKPT_BLAKE3_PREFIX) {
        eprintln!("[t2.e2e] ⛔ blake3 {digest} does not start with the 1183 pin — refusing");
        std::process::exit(2);
    }
    let w = load_dspine_safetensors(&st_path, &cfg).unwrap_or_else(|e| panic!("load: {e}"));

    // The shape-walk synthetic posture, verbatim (seed, draw order, scales):
    // lm_head first, feats second.
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
    let mut drafter = DspineDrafter::new(cfg.clone(), w);
    drafter
        .set_block_size(8)
        .unwrap_or_else(|e| panic!("block size: {e}"));
    let nf = cfg.target_layer_ids.len();
    let ctx_rows = 32usize;
    let feats: Vec<f32> = (0..ctx_rows * nf * cfg.n_embd)
        .map(|_| rng.next_f32() * 0.02)
        .collect();
    drafter
        .append_context(&feats, 0)
        .unwrap_or_else(|e| panic!("append: {e}"));
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

    let anchor_token = 151_644u32;
    let anchor_pos = ctx_rows;
    let (draft, tr) = drafter
        .draft_block_trace(anchor_token, anchor_pos, &embed_row, &lm_head, &tables)
        .unwrap_or_else(|e| panic!("draft: {e}"));
    let t = drafter.served_block_size();
    let gamma = t - 1;
    assert_eq!(draft.tokens.len(), gamma);

    std::fs::create_dir_all(out).unwrap_or_else(|e| panic!("mkdir {out:?}: {e}"));
    dump_bin(out, "anchor_emb", &embed_row(anchor_token));
    dump_bin(out, "mask_emb", &embed_row(cfg.mask_token_id));
    dump_bin(out, "feats", &feats);
    // The synthetic bf16-valued head, dumped whole (~1.25 GB at the b16
    // geometry): the Python LCG replay of 312M sequential draws would be
    // minutes of pure-Python u64 wrapping and an entire avoidable
    // bit-exactness risk — the dump makes the vendor arm a PURE replay
    // (zero input reconstruction).
    dump_bin(out, "lm_head", &lm_head);
    dump_u32_bin(out, "tokens", &draft.tokens);
    dump_u32_bin(
        out,
        "candidates",
        &draft
            .candidates
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>(),
    );
    dump_bin(out, "confs", &draft.confidences);
    // Layer-0 bisect dumps: the trace snapshots each layer's input normed
    // rows (x_in), the pre-layer residual (x), the post-norm+rope q/k/v,
    // the post-attention residual add (h), and the post-MLP stream (out).
    // The vendor arm hooks layer 0's attention + MLP returns and diffs the
    // same points — a 33-magnitude e2e miss localizes to embed/ctx,
    // attention, or MLP in ONE extra run.
    let l0 = &tr.layers[0];
    dump_bin(out, "l0_x", &l0.x);
    dump_bin(out, "l0_x_in", &l0.x_in);
    dump_bin(out, "l0_q", &l0.q);
    dump_bin(out, "l0_k", &l0.k);
    dump_bin(out, "l0_h", &l0.h);
    dump_bin(out, "l0_out", &l0.out);
    // Tail-trace diagnostics (the vendor mirror diffs these to localize a
    // mismatch without a second run): the proposal rows (the pinned
    // normed-row fold over the stash raw), the tail rms, and the walk's
    // idx/probs tables. The stash raw itself is internal — the proposal
    // rows + tail_rms pin it well enough to localize.
    dump_bin(out, "proposal_rows", &tr.tail.proposal_rows);
    dump_bin(out, "tail_rms", &tr.tail.rms);
    dump_u32_bin(
        out,
        "tail_idx",
        &tr.tail
            .idx
            .iter()
            .flatten()
            .map(|&v| v as u32)
            .collect::<Vec<_>>(),
    );
    dump_bin(
        out,
        "tail_probs",
        &tr.tail.probs.iter().flatten().copied().collect::<Vec<_>>(),
    );
    let mut manifest = String::new();
    manifest.push_str("# E2E main-path golden fixture (plan 492 T0-residue (b))\n");
    manifest.push_str(&format!(
        "- geometry: T={t} gamma={gamma} V={} H={} nf={nf} ctx_rows={ctx_rows} k={}\n",
        cfg.vocab_size, cfg.n_embd, cfg.head_topk
    ));
    manifest.push_str(&format!(
        "- anchor_token={anchor_token} anchor_pos={anchor_pos} mask_token_id={}\n",
        cfg.mask_token_id
    ));
    manifest.push_str("- lcg_seed=777 draw_order=lm_head,feats lm_head_scale=bf16 feats_scale=0.02 (provenance only — lm_head/feats are DUMPED; the vendor arm reconstructs nothing)\n");
    manifest.push_str("- checkpoint_blake3=");
    manifest.push_str(&digest);
    manifest.push('\n');
    manifest.push_str("- embed_rows: anchor_emb/mask_emb bf16-ROUNDED (the vendor stand-in table returns these verbatim)\n");
    manifest.push_str(
        "- feats: RAW f32 (the Rust side rounds bf16 before fc — the vendor arm casts bf16 too)\n",
    );
    manifest.push_str("- diff bar: tokens + candidates (the vendor greedy returns no probs; confs/tail_probs are diagnostics)\n");
    std::fs::write(out.join("MANIFEST.txt"), manifest).unwrap_or_else(|e| panic!("manifest: {e}"));
    println!(
        "[t2.e2e] wrote {} — chain [{}, …] conf {:.3} ({}, {:.1}s)",
        out.display(),
        draft
            .tokens
            .first()
            .map(|t| t.to_string())
            .unwrap_or_default(),
        draft.confidences.first().copied().unwrap_or(0.0),
        draft.tokens.len(),
        t0.elapsed().as_secs_f32()
    );
}
