//! Issue 1203 E3 — the Issue-761 dispatch-decomposition counters probe.
//!
//! One warm-up + one measured pp512 prefill through the SAME forward the E1
//! bench drives (`qwen38_wgpu_bench`), then dumps the vendored fork's
//! Issue-761 always-on counters (`cubecl_wgpu::compute`, [patch] → the fork):
//! launch / enqueue-exec / enqueue-write / flush / submit / regulate /
//! write-poll / read, counts + nanos. The decomposition answers, for m3's
//! armed dense path, whether the host dispatch ENCODING is the binding cost
//! (ENQUEUE_* dominates) or the host is merely GPU-paced (REGULATE /
//! WRITE_POLL / READ dominate) — the E3 attribution the 1203 issue names.
//!
//! Posture: build with `--features qwen38_wgpu,metal_tensor_gemm` so the
//! ARMED path is what the counters attribute (the env arms live inside the
//! mtg-gated code). The forward's own load banner (`[Issue 1150] armed` /
//! `[Issue 1176] DEMOTED`) rides stderr — the engagement adjudicator; a
//! feature-less build would measure the incumbent and adjudicate nothing.
//!
//! Counter nesting (a lower row is contained in the row above, fork doc):
//! `LAUNCH_NANOS ⊇ ENQUEUE_EXEC_NANOS ⊇ FLUSH_NANOS ⊇ {REGULATE_NANOS,
//! WRITE_POLL_NANOS}`; `READ_NANOS` is the read drain, outside launch.
//!
//! Usage (the E1–E5 runner invokes exactly this shape):
//!   i1203_counters_probe --gguf <model> [--pp 512] [--decode 0]
//!
//! Whole-file cfg keeps non-`qwen38_wgpu` builds an empty binary; the Cargo
//! [[bin]] required-features row is the reader protection.

#![cfg(feature = "qwen38_wgpu")]

use cubecl_wgpu::{
    ENQUEUE_EXEC_COUNT, ENQUEUE_EXEC_NANOS, ENQUEUE_WRITE_COUNT, ENQUEUE_WRITE_NANOS, FLUSH_COUNT,
    FLUSH_NANOS, LAUNCH_COUNT, LAUNCH_NANOS, PIPELINE_NANOS, PREPARE_NANOS, READ_COUNT, READ_NANOS,
    REGISTER_NANOS, REGULATE_COUNT, REGULATE_NANOS, SUBMIT_COUNT, WRITE_COUNT, WRITE_POLL_COUNT,
    WRITE_POLL_NANOS, reset_dispatch_profile,
};
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_gpu::qwen38_dense_cubecl::Qwen38WgpuForward;

const DEFAULT_GGUF: &str = "/Users/katopz/uzu27b/models/qwen3.8-27b-Q4_K_M.gguf";

fn load(counter: &std::sync::atomic::AtomicU64) -> u64 {
    counter.load(std::sync::atomic::Ordering::Relaxed)
}

/// The dense-tensor GEMV engagement delta (the bench's helper, verbatim
/// semantics): prefill gemms are outside every delta — capture after ingest.
#[cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]
fn dense_dispatch_count() -> u64 {
    riir_infer_gpu::gemm_dense_f16_metal_zero_copy::dense_mm()
        .map(|mm| mm.dispatch_count())
        .unwrap_or(0)
}

#[cfg(not(all(feature = "metal_tensor_gemm", target_os = "macos")))]
fn dense_dispatch_count() -> u64 {
    0
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut gguf = DEFAULT_GGUF.to_string();
    let mut pp = 512usize;
    let mut decode_steps = 0usize;
    let mut ctx = 24_576usize;
    let mut chunk = 512usize;
    let mut prompt = "The harbor town woke slowly under a grey sky,".to_string();
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--gguf" => {
                i += 1;
                gguf = args[i].clone();
            }
            "--pp" => {
                i += 1;
                pp = args[i].parse().expect("pp len");
            }
            "--decode" => {
                i += 1;
                decode_steps = args[i].parse().expect("decode n");
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
                prompt = args[i].clone();
            }
            other => {
                eprintln!(
                    "[probe] unknown arg {other} (flags: --gguf --pp --decode --ctx --chunk --prompt)"
                );
                std::process::exit(2);
            }
        }
        i += 1;
    }

    println!("[probe] loading {gguf} (ctx {ctx}, chunk {chunk}) ...");
    let t0 = std::time::Instant::now();
    let mut fwd = Qwen38WgpuForward::new(std::path::Path::new(&gguf), ctx, chunk)
        .unwrap_or_else(|e| panic!("[probe] load failed: {e}"));
    println!("[probe] loaded in {:.1}s", t0.elapsed().as_secs_f32());

    let gg = riir_infer_core::gguf_loader::GgufFile::open(std::path::Path::new(&gguf))
        .unwrap_or_else(|e| panic!("[probe] gguf open: {e}"));
    let tok = BpeTokenizer::from_gguf(&gg).unwrap_or_else(|e| panic!("[probe] tokenizer: {e}"));
    let seed: Vec<u32> = tok.encode(&prompt).into_iter().map(|t| t as u32).collect();
    let padded = pp.div_ceil(chunk) * chunk;
    let mut ids = seed.clone();
    while ids.len() < padded {
        ids.extend_from_slice(&seed);
    }
    ids.truncate(padded);
    println!(
        "[probe] prompt: {} chars -> padded to {padded} tokens",
        prompt.len()
    );

    // Warm-up pass (untimed): pipelines compiled, caches warm — then the
    // counters reset so the measured window is the whole story.
    fwd.zero_state();
    fwd.ingest(&ids, 0)
        .unwrap_or_else(|e| panic!("[probe] warm-up ingest: {e}"));
    if decode_steps > 0 {
        let mut next = 0u32;
        for s in 0..decode_steps {
            next = fwd
                .forward_token(if s == 0 { *ids.last().unwrap() } else { next }, padded + s)
                .unwrap_or_else(|e| panic!("[probe] warm-up decode: {e}"));
        }
    }
    reset_dispatch_profile();
    let dd0 = dense_dispatch_count();

    let t = std::time::Instant::now();
    fwd.zero_state();
    fwd.ingest(&ids, 0)
        .unwrap_or_else(|e| panic!("[probe] ingest: {e}"));
    if decode_steps > 0 {
        let mut next = 0u32;
        for s in 0..decode_steps {
            next = fwd
                .forward_token(if s == 0 { *ids.last().unwrap() } else { next }, padded + s)
                .unwrap_or_else(|e| panic!("[probe] decode: {e}"));
        }
    }
    let wall = t.elapsed();
    let wall_nanos = wall.as_nanos() as u64;
    let dd_delta = dense_dispatch_count().saturating_sub(dd0);

    let launch_n = load(&LAUNCH_NANOS);
    let exec_n = load(&ENQUEUE_EXEC_NANOS);
    let flush_n = load(&FLUSH_NANOS);
    let regulate_n = load(&REGULATE_NANOS);
    let write_poll_n = load(&WRITE_POLL_NANOS);
    let read_n = load(&READ_NANOS);
    let launch_c = load(&LAUNCH_COUNT);
    let exec_c = load(&ENQUEUE_EXEC_COUNT);

    let pct = |nanos: u64| -> f64 {
        if wall_nanos == 0 {
            0.0
        } else {
            100.0 * nanos as f64 / wall_nanos as f64
        }
    };

    println!(
        "[probe] wall {:.3}s ({} ns)",
        wall.as_secs_f64(),
        wall_nanos
    );
    println!(
        "[probe] LAUNCH        count={:>7}  ms={:>9.1}  wall%={:5.1}  per-launch-us={:8.1}",
        launch_c,
        launch_n as f64 / 1e6,
        pct(launch_n),
        if launch_c == 0 {
            0.0
        } else {
            launch_n as f64 / launch_c as f64 / 1e3
        },
    );
    println!(
        "[probe] ENQUEUE_EXEC  count={:>7}  ms={:>9.1}  wall%={:5.1}  per-enqueue-us={:8.1}",
        exec_c,
        exec_n as f64 / 1e6,
        pct(exec_n),
        if exec_c == 0 {
            0.0
        } else {
            exec_n as f64 / exec_c as f64 / 1e3
        },
    );
    println!(
        "[probe]   PIPELINE    ms={:>9.1}",
        load(&PIPELINE_NANOS) as f64 / 1e6
    );
    println!(
        "[probe]   PREPARE     ms={:>9.1}",
        load(&PREPARE_NANOS) as f64 / 1e6
    );
    println!(
        "[probe]   REGISTER    ms={:>9.1}",
        load(&REGISTER_NANOS) as f64 / 1e6
    );
    println!(
        "[probe] ENQUEUE_WRITE count={:>7}  ms={:>9.1}  wall%={:5.1}",
        load(&ENQUEUE_WRITE_COUNT),
        load(&ENQUEUE_WRITE_NANOS) as f64 / 1e6,
        pct(load(&ENQUEUE_WRITE_NANOS)),
    );
    println!(
        "[probe] FLUSH         count={:>7}  ms={:>9.1}  wall%={:5.1}  submit={}",
        load(&FLUSH_COUNT),
        flush_n as f64 / 1e6,
        pct(flush_n),
        load(&SUBMIT_COUNT),
    );
    println!(
        "[probe]   REGULATE    count={:>7}  ms={:>9.1}  wall%={:5.1}  per-block-us={:8.1}",
        load(&REGULATE_COUNT),
        regulate_n as f64 / 1e6,
        pct(regulate_n),
        {
            let c = load(&REGULATE_COUNT);
            if c == 0 {
                0.0
            } else {
                regulate_n as f64 / c as f64 / 1e3
            }
        },
    );
    println!(
        "[probe]   WRITE_POLL  count={:>7}  ms={:>9.1}  wall%={:5.1}  (writes={})",
        load(&WRITE_POLL_COUNT),
        write_poll_n as f64 / 1e6,
        pct(write_poll_n),
        load(&WRITE_COUNT),
    );
    println!(
        "[probe] READ          count={:>7}  ms={:>9.1}  wall%={:5.1}  per-read-ms={:8.2}",
        load(&READ_COUNT),
        read_n as f64 / 1e6,
        pct(read_n),
        {
            let c = load(&READ_COUNT);
            if c == 0 {
                0.0
            } else {
                read_n as f64 / c as f64 / 1e6
            }
        },
    );

    // The E5 attribution rows: the launch machinery ABOVE the stream
    // (server.rs: the LAUNCH − ENQUEUE_EXEC delta) and the two verdict
    // shares — CPU-encode-bound vs GPU-paced.
    let above_stream = launch_n.saturating_sub(exec_n);
    println!(
        "[probe] LAUNCH−ENQUEUE_EXEC (launch machinery above the stream) ms={:.1} wall%={:.1}",
        above_stream as f64 / 1e6,
        pct(above_stream)
    );
    let gpu_paced = regulate_n + write_poll_n + read_n;
    println!(
        "[probe] GPU-paced share (REGULATE+WRITE_POLL+READ) ms={:.1} wall%={:.1}",
        gpu_paced as f64 / 1e6,
        pct(gpu_paced)
    );
    println!("[probe] [gemv-engagement] dense dispatches Δ+{dd_delta} over the measured pass");
    println!(
        "[probe] verdict input: encode-bound wall%={:.1} vs GPU-paced wall%={:.1} — ENCODE-bound = dispatch-serialization class (b), GPU-paced = throughput class (a)",
        pct(exec_n.saturating_sub(flush_n)),
        pct(gpu_paced)
    );
    println!(
        "[probe] posture reminder: the stderr load banner ([Issue 1150] armed / [Issue 1176] DEMOTED) adjudicates what this measured — a feature-less build measures the incumbent, not the armed path."
    );
}
