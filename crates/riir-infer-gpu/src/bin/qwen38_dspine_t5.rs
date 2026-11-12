//! Issue 1183 T5 — the head-only acceptance price cell (the decision gate).
//!
//! Measures the DSpine b16 drafter's acceptance against OUR Q4_K_M GGUF
//! target, reusing the Bench-954 rig shape (the 1179 T1 cell): the CPU
//! oracle drafts a 15-token chain per cycle (`DspineDrafter::draft_block` —
//! block forward + gated injection + the exact_sender serial tail), the
//! target verifies it in one keyed pass (`qwen38_spec::keyed_accept`), and
//! the cell reports mean accepted tokens per forward pass. Same box law as
//! 954: acceptance is a discrete, load-invariant measurement — speed is
//! irrelevant here (this is not a perf cell).
//!
//! **Decision gate (the issue's T5, pre-registered):** head-only mean_j
//! ≥ the 1179 beat-arithmetic ~3.2-class on OUR target → this becomes the
//! beat lane's preferred head (and the first candidate above the riir-ai
//! plan-614 Phase-3 L_eff floor); below the uzu 2.5-class bar → record +
//! Path B (riir-train distillation) takes the GPU run. Between 2.5 and
//! 3.2 → owner verdict.
//!
//! Contexts (each a disclosure, carried in the JSON row):
//! - the draft consumes OUR Q6_K-dequant `lm_head` (the vendor's was bf16)
//!   and OUR captured hidden states (the league Metal posture's f16/f32
//!   activations, not the vendor runtime's bf16) — the Bench-954
//!   target-numerics class;
//! - the TG/TSV serial tables derive from that same dequant head (one
//!   build per run, ~100 s over the full 248320 vocab);
//! - the draft runs f32 CPU (the T2 oracle, Bench-959 golden-diff'd
//!   op-for-op vs the vendor MLX reference — idx/tokens bit-exact).
//!
//! Correctness gate (G1-T5, greedy): the harness first walks `--sanity-steps`
//! tokens PURELY serially, then runs the specdec cycles from the same
//! state — the two token streams must be IDENTICAL (the draft only
//! proposes; the target's own keyed samples decide).

#![cfg(feature = "qwen38_dspine")]

use std::path::Path;

use riir_infer_core::gguf_loader::GgufFile;
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_gpu::qwen38_dense_cubecl::Qwen38WgpuForward;
use riir_infer_gpu::qwen38_dspine::{
    DspineConfig, DspineDrafter, build_serial_tables, load_dspine_safetensors, parse_config_json,
};
use riir_infer_gpu::qwen38_spec::{KeyedVerifyPosture, keyed_accept, serial_keyed_step};

const DEFAULT_GGUF: &str = "/Users/katopz/uzu27b/models/qwen3.8-27b-Q4_K_M.gguf";
const DEFAULT_DSPINE_DIR: &str = "/Users/katopz/uzu27b/models/Qwen3.8-27B-DSpine-b16";
/// The T1 pin (blake3 53d18516…, b3sum + python-blake3 agree).
const EXPECTED_DSPINE_BLAKE3_PREFIX: &str = "53d18516";
/// The Bench-954 league GGUF pin (the Q4_K_M target row).
const EXPECTED_GGUF_BLAKE3_PREFIX: &str = "ce14c955";

/// The vendor-method filler (Bench 1167 R2's FILLER, as the t1 cell).
const VENDOR_FILLER: &str = "Analyze the following system design and enumerate every failure mode you can find, ordered by severity, with the detection signal for each. Be exhaustive: cover consensus, storage, networking, scheduling, and the operator surface. ";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut gguf_path = DEFAULT_GGUF.to_string();
    let mut dspine_dir = DEFAULT_DSPINE_DIR.to_string();
    let mut mode = "greedy".to_string(); // greedy | stochastic
    let mut cycles = 200usize;
    let mut ctx = 8192usize;
    let mut chunk = 512usize;
    let mut prompt_kind = "fixture".to_string(); // fixture | chat | vendor
    let mut fixture_path = String::new();
    let mut seed = 0u64;
    let mut sanity_steps = 32usize;
    let mut out_path = String::new();
    let mut block_size = 16usize;
    let mut backend = "wgpu".to_string();
    let mut expected_dspine_pin = EXPECTED_DSPINE_BLAKE3_PREFIX.to_string();

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--gguf" => {
                i += 1;
                gguf_path = args[i].clone();
            }
            "--dspine" => {
                i += 1;
                dspine_dir = args[i].clone();
            }
            "--mode" => {
                i += 1;
                mode = args[i].clone();
            }
            "--cycles" => {
                i += 1;
                cycles = args[i].parse().expect("cycles");
            }
            "--ctx" => {
                i += 1;
                ctx = args[i].parse().expect("ctx");
            }
            "--chunk" => {
                i += 1;
                chunk = args[i].parse().expect("chunk");
            }
            "--prompt" => {
                i += 1;
                prompt_kind = args[i].clone();
            }
            "--fixture" => {
                i += 1;
                fixture_path = args[i].clone();
            }
            "--seed" => {
                i += 1;
                seed = args[i].parse().expect("seed");
            }
            "--sanity-steps" => {
                i += 1;
                sanity_steps = args[i].parse().expect("sanity-steps");
            }
            "--out" => {
                i += 1;
                out_path = args[i].clone();
            }
            "--block-size" => {
                i += 1;
                block_size = args[i].parse().expect("block-size");
            }
            "--backend" => {
                i += 1;
                backend = args[i].clone();
            }
            "--expect-dspine-pin" => {
                // The trained-artifact lane (plan 492 T4): override the
                // vendor blake3 prefix pin with the caller's pin (a trained
                // checkpoint is new evidence, never a vendor forgery — the
                // law stays, the pin moves with the evidence).
                i += 1;
                expected_dspine_pin = args[i].clone();
            }
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(2);
            }
        }
        i += 1;
    }

    let posture = if mode == "greedy" {
        KeyedVerifyPosture {
            temperature: 0.0,
            top_k: None,
            top_p: None,
            seed,
        }
    } else if mode == "stochastic" {
        // Bench-1167 R2 vendor-method convention: T=0.95 top_k=20 top_p=1.0.
        KeyedVerifyPosture {
            temperature: 0.95,
            top_k: Some(20),
            top_p: Some(1.0),
            seed,
        }
    } else {
        eprintln!("--mode must be greedy|stochastic");
        std::process::exit(2);
    };

    println!("[t5] gguf {gguf_path}");
    println!("[t5] dspine {dspine_dir} mode {mode} cycles {cycles} prompt {prompt_kind}");

    // ── artifact pins (A1/A8: verify BEFORE any weight is consumed) ──
    let ds_file = Path::new(&dspine_dir).join("model.safetensors");
    let t0 = std::time::Instant::now();
    let ds_digest = blake3_file(&ds_file);
    println!(
        "[t5] dspine blake3 {ds_digest} ({:.1}s)",
        t0.elapsed().as_secs_f32()
    );
    if !ds_digest.starts_with(expected_dspine_pin.as_str()) {
        eprintln!(
            "[t5] ⛔ dspine blake3 {ds_digest} does not start with the expected pin {expected_dspine_pin} — refusing (weights-from-disk pin law; a trained artifact passes --expect-dspine-pin <prefix>)"
        );
        std::process::exit(2);
    }
    let gguf_digest = blake3_file(Path::new(&gguf_path));
    println!("[t5] gguf blake3 {gguf_digest}");
    if !gguf_digest.starts_with(EXPECTED_GGUF_BLAKE3_PREFIX) {
        eprintln!(
            "[t5] ⛔ gguf blake3 {gguf_digest} does not start with the Bench-954 pin {EXPECTED_GGUF_BLAKE3_PREFIX} — refusing"
        );
        std::process::exit(2);
    }

    // ── load the target (the backend seam: wgpu everywhere; cuda on the
    //    cudarc-capable boxes — the 24 GB bar cannot hold the cubecl
    //    forward's workspace for this checkpoint, while the cudarc forward
    //    is the serving runtime the plan's gate names) ──
    let mut fwd: Box<dyn T5Target> = match backend.as_str() {
        "wgpu" => Box::new(
            WgpuTarget::new(Path::new(&gguf_path), ctx, chunk).unwrap_or_else(|e| panic!("{e}")),
        ),
        #[cfg(all(feature = "ternary_gemv_cuda_raw", not(target_os = "macos")))]
        "cuda" => {
            Box::new(CudaTarget::new(Path::new(&gguf_path), ctx).unwrap_or_else(|e| panic!("{e}")))
        }
        #[cfg(any(not(feature = "ternary_gemv_cuda_raw"), target_os = "macos"))]
        "cuda" => {
            eprintln!("--backend cuda needs the ternary_gemv_cuda_raw feature (non-macOS)");
            std::process::exit(2);
        }
        other => {
            eprintln!("--backend must be wgpu|cuda (got {other})");
            std::process::exit(2);
        }
    };
    let vocab_size = fwd.vocab_size();
    println!("[t5] target loaded ({backend})");
    let gg = GgufFile::open(Path::new(&gguf_path)).unwrap_or_else(|e| panic!("[t5] reopen: {e}"));
    let tok = BpeTokenizer::from_gguf(&gg).unwrap_or_else(|e| panic!("[t5] tokenizer: {e}"));

    // ── load the DSpine drafter (the T2 oracle; Bench-959 golden-diff'd) ──
    let t0 = std::time::Instant::now();
    let cfg_json = std::fs::read_to_string(Path::new(&dspine_dir).join("config.json"))
        .unwrap_or_else(|e| panic!("[t5] config.json: {e}"));
    let cfg: DspineConfig =
        parse_config_json(&cfg_json).unwrap_or_else(|e| panic!("[t5] config: {e}"));
    let w = load_dspine_safetensors(&ds_file, &cfg).unwrap_or_else(|e| panic!("[t5] load: {e}"));
    let capture_layers = cfg.target_layer_ids.clone();
    let d = cfg.n_embd;
    let captured = capture_layers.len();
    let vocab = cfg.vocab_size;
    assert_eq!(vocab, vocab_size, "vocab mismatch");
    println!(
        "[t5] dspine loaded in {:.1}s (layers {} block {} capture {:?})",
        t0.elapsed().as_secs_f32(),
        cfg.n_layer,
        cfg.trained_block_size,
        capture_layers
    );

    // ── lm_head f32 table (the draft's readout = the TARGET's output head;
    //    the cross-precision disclosure — the vendor's was bf16) ──
    let t0 = std::time::Instant::now();
    let lm_head = dequant_lm_head(&gg, vocab, d);
    println!(
        "[t5] lm_head dequant {vocab} rows in {:.1}s",
        t0.elapsed().as_secs_f32()
    );

    // ── the serial refinement tables over the FULL vocab (one build; the
    //    TG/TSV readout of the exact_sender tail) ──
    let t0 = std::time::Instant::now();
    let tables = build_serial_tables(
        &lm_head,
        &w.model_norm,
        &w.bridge.feature_write_weights[cfg.n_layer - 1],
        &w.pv_weight(),
        &w.bridge.codebook,
        &cfg,
    );
    println!(
        "[t5] serial tables built in {:.1}s (TG {} MB, TSV {} MB)",
        t0.elapsed().as_secs_f32(),
        tables.tg.len() * 4 / (1 << 20),
        tables.tsv.len() * 4 / (1 << 20)
    );

    // ── the drafter ──
    let mut drafter = DspineDrafter::new(cfg.clone(), w);
    drafter
        .set_block_size(block_size)
        .unwrap_or_else(|e| panic!("[t5] block size: {e}"));
    let block = drafter.served_block_size();

    // ── the prompt ──
    let prompt_text = match prompt_kind.as_str() {
        "fixture" => {
            if fixture_path.is_empty() {
                eprintln!("--prompt fixture needs --fixture <path>");
                std::process::exit(2);
            }
            std::fs::read_to_string(&fixture_path).unwrap_or_else(|e| panic!("[t5] fixture: {e}"))
        }
        // The vendor regime (the Bench-952 lesson — a raw fixture makes the
        // target's continuation start at <|im_end|>; the chat wrapper keeps
        // the head in-distribution).
        "chat" => {
            if fixture_path.is_empty() {
                eprintln!("--prompt chat needs --fixture <path>");
                std::process::exit(2);
            }
            let fixture = std::fs::read_to_string(&fixture_path)
                .unwrap_or_else(|e| panic!("[t5] fixture: {e}"));
            format!("<|im_start|>user\n{fixture}<|im_end|>\n<|im_start|>assistant\n<think>\n")
        }
        "vendor" => {
            let per = tok.encode(VENDOR_FILLER).len().max(1);
            let reps = ((1357 + per / 2) / per).max(1);
            VENDOR_FILLER.repeat(reps)
        }
        other => {
            eprintln!("--prompt must be fixture|chat|vendor (got {other})");
            std::process::exit(2);
        }
    };
    let prompt: Vec<u32> = tok
        .encode(&prompt_text)
        .into_iter()
        .map(|t| t as u32)
        .collect();
    println!(
        "[t5] prompt: {} chars -> {} tokens",
        prompt_text.len(),
        prompt.len()
    );
    assert!(
        prompt.len() + cycles * (block + 4) < ctx,
        "prompt+cycles exceed ctx {ctx}"
    );

    // ── prefill + DSpine context append (every prompt position's captured
    //    features feed the context fusion — the vendor's append_context) ──
    let t0 = std::time::Instant::now();
    let (feats, last_logits) = fwd
        .ingest_capture(&prompt, 0, &capture_layers)
        .unwrap_or_else(|e| panic!("[t5] prefill: {e}"));
    println!("[t5] prefill done in {:.1}s", t0.elapsed().as_secs_f32());
    let mut packed = vec![0.0f32; prompt.len() * d * captured];
    for (li, f) in feats.iter().enumerate() {
        assert_eq!(f.len(), prompt.len() * d, "capture layer {li} length");
        for t in 0..prompt.len() {
            packed[t * d * captured + li * d..t * d * captured + (li + 1) * d]
                .copy_from_slice(&f[t * d..(t + 1) * d]);
        }
    }
    drafter
        .append_context(&packed, 0)
        .unwrap_or_else(|e| panic!("[t5] draft prefill append: {e}"));

    // The first emitted token: the target's sample at the last prompt position.
    let mut pending = serial_keyed_step(&posture, &last_logits, prompt.len());
    let mut emitted_total = 1usize;

    // ── G1-T5 sanity (greedy only): a pure serial walk must equal the
    //    specdec stream. Run the serial walk FIRST from this state, then
    //    rewind and run the specdec cycles, comparing prefixes.
    let mut serial_stream: Vec<u32> = Vec::new();
    if mode == "greedy" && sanity_steps > 0 {
        fwd.snapshot()
            .unwrap_or_else(|e| panic!("[t5] serial snapshot: {e}"));
        let mut tok_i = pending;
        for pos in (prompt.len()..).take(sanity_steps) {
            let (next, _, _) = fwd
                .forward_token_capture(tok_i, pos, &[])
                .unwrap_or_else(|e| panic!("[t5] serial sanity: {e}"));
            serial_stream.push(next);
            tok_i = next;
        }
        fwd.restore()
            .unwrap_or_else(|e| panic!("[t5] serial restore: {e}"));
    }

    // ── the specdec cycle loop ──
    // The embed closure: the TARGET's Q4_K-dequant token rows (the draft
    // binds the target's embed_tokens — no incoherence transform; the
    // DSpine checkpoint ships none).
    let embed_row = |tok: u32| -> Vec<f32> {
        gg.dequant_tensor_row("token_embd.weight", tok as usize, d)
            .unwrap_or_else(|e| panic!("[t5] token emb {tok}: {e}"))
    };
    let mut js: Vec<usize> = Vec::new();
    // Per-position rank-0 (the Bench-954 definition): the draft's chain[p-1]
    // vs the TARGET's own keyed argmax at verify row p, unconditional.
    let mut rank0_hits = vec![0usize; block - 1];
    let mut specdec_stream: Vec<u32> = Vec::new();
    let mut pos = prompt.len(); // the pending token's position (unforwarded)
    let positions_bound = ctx - block - 2;
    let t_loop = std::time::Instant::now();
    for cycle in 0..cycles {
        if pos >= positions_bound {
            println!("[t5] position bound {positions_bound} reached at cycle {cycle}");
            break;
        }
        // Draft: block−1 candidates from one block forward + serial tail.
        let t_draft = std::time::Instant::now();
        let draft = drafter
            .draft_block(pending, pos, &embed_row, &lm_head, &tables)
            .unwrap_or_else(|e| panic!("[t5] draft: {e}"));
        let chain = draft.tokens;
        assert_eq!(chain.len(), block - 1, "chain length");

        // Verify: [pending, chain...] in one keyed pass, capturing the
        // DSpine context features.
        let rows = chain.len() + 1;
        fwd.snapshot()
            .unwrap_or_else(|e| panic!("[t5] verify snapshot: {e}"));
        let mut logits_flat = vec![0.0f32; rows * vocab];
        let mut caps_j: Vec<Vec<Vec<f32>>> = Vec::with_capacity(rows);
        for (r, tok_i) in std::iter::once(pending)
            .chain(chain.iter().copied())
            .enumerate()
        {
            let (_, logits, caps) = fwd
                .forward_token_capture(tok_i, pos + r, &capture_layers)
                .unwrap_or_else(|e| panic!("[t5] verify row {r}: {e}"));
            logits_flat[r * vocab..(r + 1) * vocab].copy_from_slice(&logits);
            caps_j.push(caps);
        }
        let draft_secs = t_draft.elapsed().as_secs_f32();
        let _ = draft_secs;

        // rank-0 census (unconditional over all cycles).
        for (r, trow) in logits_flat.chunks_exact(vocab).enumerate().skip(1) {
            let s = posture.sample_row(trow, pos + r);
            if chain[r - 1] == s {
                rank0_hits[r - 1] += 1;
            }
        }
        let acc = keyed_accept(&posture, &logits_flat, vocab, &chain, pos);
        let j = acc.j;
        js.push(j);
        emitted_total += j;
        specdec_stream.extend_from_slice(&chain[..j.saturating_sub(1).min(chain.len())]);
        specdec_stream.push(acc.bonus);

        // DSpine context append: rows 0..j-1 (the Bench-954 vendor
        // approximation — the LAST row's capture serves the bonus token's
        // feature), at absolute positions pos..pos+j.
        {
            let mut packed_rows = vec![0.0f32; j * d * captured];
            for r in 0..j {
                for (li, cap) in caps_j[r].iter().enumerate() {
                    packed_rows[r * d * captured + li * d..r * d * captured + (li + 1) * d]
                        .copy_from_slice(cap);
                }
            }
            drafter
                .append_context(&packed_rows, pos)
                .unwrap_or_else(|e| panic!("[t5] draft append: {e}"));
        }

        // State discipline: full accept leaves the forwarded set exactly
        // the true prefix; a partial accept restores and re-forwards.
        if j < rows {
            fwd.restore()
                .unwrap_or_else(|e| panic!("[t5] verify rollback: {e}"));
            for (r, tok_i) in std::iter::once(pending)
                .chain(chain.iter().take(j - 1).copied())
                .enumerate()
            {
                fwd.forward_token_capture(tok_i, pos + r, &[])
                    .unwrap_or_else(|e| panic!("[t5] re-forward {r}: {e}"));
            }
        }
        pending = acc.bonus;
        pos += j;

        if cycle % 10 == 0 || cycle == cycles - 1 {
            let mean = js.iter().sum::<usize>() as f64 / js.len() as f64;
            println!(
                "[t5] cycle {cycle}: j={j} mean_j={mean:.3} emitted={emitted_total} pos={pos} ctx_len={} ({:.1}s)",
                drafter.context_length(),
                t_loop.elapsed().as_secs_f32(),
            );
        }
    }

    // ── G1-T5 verdict: the specdec greedy stream must equal the serial walk
    //    token for token over the compared prefix.
    if mode == "greedy" && sanity_steps > 0 {
        let n = specdec_stream.len().min(serial_stream.len());
        let mismatches: Vec<usize> = (0..n)
            .filter(|&i| specdec_stream[i] != serial_stream[i])
            .collect();
        if !mismatches.is_empty() {
            let i = mismatches[0];
            eprintln!(
                "[t5] ⛔ G1-T5 FAIL at stream {i}: specdec {} vs serial {} ({} mismatches of {n})",
                specdec_stream[i],
                serial_stream[i],
                mismatches.len()
            );
            std::process::exit(3);
        }
        println!("[t5] G1-T5 PASS — serial == specdec over {n} committed tokens");
    }

    let mean = js.iter().sum::<usize>() as f64 / js.len().max(1) as f64;
    let mut hist = [0usize; 17];
    for &j in &js {
        hist[j.min(16)] += 1;
    }
    let n_cycles = js.len();
    let survival: Vec<f64> = (0..=block - 1)
        .map(|p| js.iter().filter(|&&j| j > p).count() as f64 / n_cycles.max(1) as f64)
        .collect();
    let rank0_rate: Vec<f64> = rank0_hits
        .iter()
        .map(|&h| h as f64 / n_cycles.max(1) as f64)
        .collect();
    let row = serde_json::json!({
        "cell": format!("t5_head_only_{}_{}", prompt_kind, mode),
        "mode": mode,
        "prompt_kind": prompt_kind,
        "prompt_tokens": prompt.len(),
        "cycles": js.len(),
        "mean_j": mean,
        "j_histogram": hist,
        "rank0_hits_by_position": rank0_hits,
        "rank0_rate_by_position": rank0_rate,
        "survival_by_position": survival,
        "note_position_stats": "rank0[p] = DSpine chain[p-1] == target keyed sample at verify row p, unconditional over all cycles; survival[p] = P(j > p)",
        "emitted_total": emitted_total,
        "dspine_blake3": ds_digest,
        "gguf_blake3": gguf_digest,
        "seed": seed,
        "sampling": if mode == "greedy" { "greedy (keyed argmax, first-index ties)" } else { "T=0.95 top_k=20 top_p=1.0 (keyed gumbel-max; penalty-free)" },
        "capture_layers": capture_layers,
        "block_size": block,
        "note": "DSpine b16 serial-tail chain (exact_sender, Bench-959 oracle) over OUR Q4_K_M target; CPU f32 draft; Q6_K-dequant lm_head in the tail + TG/TSV tables; target-capture posture per the run's --backend (wgpu/cubecl or cuda/cudarc — the Bench-954 target-numerics disclosure class, backend disclosed by the row's backend field)",
        "backend": backend,
        "ts": chrono_now(),
    });
    let line = serde_json::to_string(&row).expect("json");
    println!("[t5] ROW {line}");
    if !out_path.is_empty() {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&out_path)
            .expect("out open");
        writeln!(f, "{line}").expect("out write");
    }
    let _ = &serial_stream;
    let _ = &specdec_stream;
}

/// The T5 target seam — the four calls the harness loop makes, so the
/// cell runs on the wgpu forward (every box) or the cudarc serving
/// forward (CUDA boxes; the 24 GB bar cannot hold the cubecl forward's
/// workspace for the 27B Q4_K_M checkpoint). The gate compares ckpts ON
/// ONE backend, so cross-backend bit-identity is not required here —
/// internal consistency (G1-T5 serial == specdec) is asserted per run.
trait T5Target {
    fn vocab_size(&self) -> usize;
    /// Prefill + per-position captures: (feats[layer][pos·d..], last logits).
    fn ingest_capture(
        &mut self,
        tokens: &[u32],
        base_pos: usize,
        capture_layers: &[usize],
    ) -> Result<(Vec<Vec<f32>>, Vec<f32>), String>;
    /// One token forward: (argmax, logits, captures in `capture_layers` order).
    fn forward_token_capture(
        &mut self,
        token: u32,
        pos: usize,
        capture_layers: &[usize],
    ) -> Result<(u32, Vec<f32>, Vec<Vec<f32>>), String>;
    /// The specdec state discipline: snapshot before a verify run, restore
    /// on a partial accept (KV rows past the rewind are overwritten before
    /// read by any re-feed — the T1.8 lesson — so only the recurrent state
    /// needs the snapshot).
    fn snapshot(&mut self) -> Result<(), String>;
    fn restore(&mut self) -> Result<(), String>;
}

struct WgpuTarget(
    Qwen38WgpuForward,
    Option<riir_infer_gpu::qwen38_dense_cubecl::StateCheckpoint>,
);

impl WgpuTarget {
    fn new(path: &Path, ctx: usize, chunk: usize) -> Result<Self, String> {
        let t0 = std::time::Instant::now();
        let fwd = Qwen38WgpuForward::new(path, ctx, chunk)
            .map_err(|e| format!("[t5] wgpu target load failed: {e}"))?;
        println!(
            "[t5] wgpu forward constructed in {:.1}s",
            t0.elapsed().as_secs_f32()
        );
        Ok(Self(fwd, None))
    }
}

impl T5Target for WgpuTarget {
    fn vocab_size(&self) -> usize {
        self.0.config.vocab_size
    }
    fn ingest_capture(
        &mut self,
        tokens: &[u32],
        base_pos: usize,
        capture_layers: &[usize],
    ) -> Result<(Vec<Vec<f32>>, Vec<f32>), String> {
        self.0.ingest_capture(tokens, base_pos, capture_layers)
    }
    fn forward_token_capture(
        &mut self,
        token: u32,
        pos: usize,
        capture_layers: &[usize],
    ) -> Result<(u32, Vec<f32>, Vec<Vec<f32>>), String> {
        self.0.forward_token_capture(token, pos, capture_layers)
    }
    fn snapshot(&mut self) -> Result<(), String> {
        let cp = self.0.checkpoint_state();
        self.1 = Some(cp);
        Ok(())
    }
    fn restore(&mut self) -> Result<(), String> {
        if let Some(cp) = self.1.take() {
            self.0.restore_state(&cp);
        }
        Ok(())
    }
}

#[cfg(all(feature = "ternary_gemv_cuda_raw", not(target_os = "macos")))]
struct CudaTarget(riir_infer_gpu::qwen38_dense_cudarc::Qwen38DenseForward);

#[cfg(all(feature = "ternary_gemv_cuda_raw", not(target_os = "macos")))]
impl CudaTarget {
    fn new(path: &Path, ctx: usize) -> Result<Self, String> {
        let t0 = std::time::Instant::now();
        let fwd = riir_infer_gpu::qwen38_dense_cudarc::Qwen38DenseForward::new(path, ctx)
            .map_err(|e| format!("[t5] cuda target load failed: {e}"))?;
        println!(
            "[t5] cuda forward constructed in {:.1}s",
            t0.elapsed().as_secs_f32()
        );
        Ok(Self(fwd))
    }
}

#[cfg(all(feature = "ternary_gemv_cuda_raw", not(target_os = "macos")))]
impl T5Target for CudaTarget {
    fn vocab_size(&self) -> usize {
        self.0.cfg.vocab_size
    }
    fn ingest_capture(
        &mut self,
        tokens: &[u32],
        base_pos: usize,
        capture_layers: &[usize],
    ) -> Result<(Vec<Vec<f32>>, Vec<f32>), String> {
        // The cudarc capture contract: capture_out buffers are PRE-SIZED to
        // n_embd and OVERWRITTEN per call (never appended). Scratch per
        // layer, then extend the per-position feature rows.
        let n = self.0.cfg.n_embd;
        let mut scratch: Vec<Vec<f32>> = capture_layers.iter().map(|_| vec![0.0f32; n]).collect();
        let mut feats: Vec<Vec<f32>> = capture_layers.iter().map(|_| Vec::new()).collect();
        let mut last_logits = Vec::new();
        for (i, &t) in tokens.iter().enumerate() {
            let (next, logits) = self.0.forward_token_capture_logits(
                t,
                base_pos + i,
                capture_layers,
                &mut scratch,
            )?;
            for (l, cap) in scratch.iter().enumerate() {
                feats[l].extend_from_slice(cap);
            }
            last_logits = logits;
            let _ = next;
        }
        Ok((feats, last_logits))
    }
    fn forward_token_capture(
        &mut self,
        token: u32,
        pos: usize,
        capture_layers: &[usize],
    ) -> Result<(u32, Vec<f32>, Vec<Vec<f32>>), String> {
        let mut caps: Vec<Vec<f32>> = capture_layers
            .iter()
            .map(|_| vec![0.0f32; self.0.cfg.n_embd])
            .collect();
        let (next, logits) =
            self.0
                .forward_token_capture_logits(token, pos, capture_layers, &mut caps)?;
        Ok((next, logits, caps))
    }
    fn snapshot(&mut self) -> Result<(), String> {
        self.0.verify_snapshot_gdn()
    }
    fn restore(&mut self) -> Result<(), String> {
        self.0.verify_rollback_gdn()
    }
}

fn dequant_lm_head(gg: &GgufFile, vocab: usize, e: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; vocab * e];
    let t0 = std::time::Instant::now();
    // Per-row through the loader's typed dispatcher (Q6_K for this
    // checkpoint; any supported type works — loud error otherwise).
    for v in 0..vocab {
        let row = gg
            .dequant_tensor_row("output.weight", v, e)
            .unwrap_or_else(|err| panic!("lm_head row {v}: {err}"));
        out[v * e..(v + 1) * e].copy_from_slice(&row);
        if v % 50000 == 0 && v > 0 {
            eprintln!(
                "[t5] lm_head {v}/{vocab} ({:.1}s)",
                t0.elapsed().as_secs_f32()
            );
        }
    }
    out
}

fn blake3_file(path: &Path) -> String {
    use blake3::Hasher;
    let mut f = std::fs::File::open(path).unwrap_or_else(|e| panic!("blake3 open {path:?}: {e}"));
    let mut hasher = Hasher::new();
    std::io::copy(&mut f, &mut hasher).expect("blake3 read");
    hasher.finalize().to_hex().to_string()
}

fn chrono_now() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("epoch-{now}")
}
