//! Issue 1245 bisect — which chunk-prefill arm poisons the tail logits?
//!
//! The filed reproduction ran every "posture" with the LEAGUE env defaults,
//! which on macOS+`metal_tensor_gemm` means the matmul2d tensor-flash
//! attention (`RIIR_QWEN38_ATTN_TENSOR`, demote-only) was ARMED in all of
//! them — and the decode control never enters `prefill_attn_layer` at all.
//! This example isolates the arms with readbacks:
//!
//! - default (league env untouched): reproduce the garbage + scratch reads
//! - `--no-attn-tensor`: demote the tensor-flash arm (tiled CubeCL attention)
//! - `--no-f16act`: additionally demote the half-B GEMM context (mixed path)
//! - `--prefix 8`: decode-only control (< chunk ⇒ no whole-chunk ingest)
//!
//! Reads per arm: `last_logits` stats (the garbage), `tmp_b[0..e]` +
//! `x_b` last row stats (the tail norm's input/output), and ONE decode
//! continuation step (do the chunk-seeded recurrent states survive?).
//!
//! ```sh
//! cargo run --release -p riir-infer-gpu --features qwen38_wgpu \
//!   --example i1245_tail_bisect -- \
//!   --gguf ../riir-train/data/Qwen3.5-0.8B-Base-Q8_0.gguf \
//!   --corpus tests/fixtures/i1188_family_census_bonsai2_27b_pq2.txt \
//!   --prefix 512 [--no-attn-tensor] [--no-f16act] [--no-dense]
//! ```
#![cfg(feature = "qwen38_wgpu")]

use riir_infer_core::gguf_loader::GgufFile;
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_gpu::qwen38_dense_cubecl::Qwen38WgpuForward;

fn stats(name: &str, v: &[f32]) {
    let nonfinite = v.iter().filter(|x| !x.is_finite()).count();
    let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
    let (mut am, mut ami) = (0.0f32, 0usize);
    for (i, &x) in v.iter().enumerate() {
        if x < mn {
            mn = x;
        }
        if x > mx {
            mx = x;
        }
        if x.abs() > am {
            am = x.abs();
            ami = i;
        }
    }
    let head: Vec<String> = v.iter().take(6).map(|x| format!("{x:.4}")).collect();
    println!(
        "[bisect] {name}: n={} nonfinite={} min={mn:.4} max={mx:.4} absmax={am:.4}@{ami} head=[{}]",
        v.len(),
        nonfinite,
        head.join(", ")
    );
}

fn main() {
    let mut gguf = String::from("../riir-train/data/Qwen3.5-0.8B-Base-Q8_0.gguf");
    let mut corpus = String::from("tests/fixtures/i1188_family_census_bonsai2_27b_pq2.txt");
    let mut prefix = 512usize;
    let mut chunk = 512usize;
    let mut no_attn_tensor = false;
    let mut no_f16act = false;
    let mut no_dense = false;
    let mut read_scratch = true;
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--gguf" => {
                gguf = args[i + 1].clone();
                i += 1;
            }
            "--corpus" => {
                corpus = args[i + 1].clone();
                i += 1;
            }
            "--prefix" => {
                prefix = args[i + 1].parse().expect("prefix usize");
                i += 1;
            }
            "--chunk" => {
                chunk = args[i + 1].parse().expect("chunk usize");
                i += 1;
            }
            "--no-attn-tensor" => no_attn_tensor = true,
            "--no-f16act" => no_f16act = true,
            "--no-dense" => no_dense = true,
            "--no-read-scratch" => read_scratch = false,
            other => panic!("unknown arg {other}"),
        }
        i += 1;
    }

    // Demote BEFORE construction (the forward reads env once at boot).
    // SAFETY: single-threaded main, before any thread spawn — no concurrent
    // env readers can race the writes.
    unsafe {
        if no_attn_tensor {
            std::env::set_var("RIIR_QWEN38_ATTN_TENSOR", "0");
        }
        if no_f16act {
            std::env::set_var("RIIR_QWEN38_DENSE_MM2D_F16ACT", "0");
        }
        if no_dense {
            std::env::set_var("RIIR_QWEN38_DENSE_MM2D", "0");
        }
    }
    println!(
        "[bisect] arms: no_attn_tensor={no_attn_tensor} no_f16act={no_f16act} \
         no_dense={no_dense} prefix={prefix} chunk={chunk}"
    );

    let gg = GgufFile::open(std::path::Path::new(&gguf)).expect("gguf open");
    let tok = BpeTokenizer::from_gguf(&gg).expect("tokenizer");
    let text = std::fs::read_to_string(&corpus).expect("corpus read");
    let ids: Vec<u32> = tok.encode(&text).into_iter().map(|t| t as u32).collect();
    println!("[bisect] corpus -> {} tokens", ids.len());
    assert!(ids.len() >= prefix, "corpus shorter than prefix");

    let ctx = prefix + 16;
    let mut fwd =
        Qwen38WgpuForward::new(std::path::Path::new(&gguf), ctx, chunk).expect("session boot");

    fwd.ingest(&ids[..prefix], 0).expect("ingest");
    let logits = fwd.last_logits().expect("last_logits");
    stats("logits_b (chunk tail)", &logits);

    if read_scratch {
        let gg2 = GgufFile::open(std::path::Path::new(&gguf)).expect("gguf reopen");
        let e = gg2
            .metadata_u64("qwen35.embedding_length")
            .or_else(|| gg2.metadata_u64("embedding_length"))
            .unwrap_or(0) as usize;
        assert!(e > 0, "embedding_length not found in the GGUF header");
        let xrow = prefix.min(chunk) - 1; // the row the tail norm reads
        let (tmp, xb) = fwd.debug_read_scratch(e, Some(xrow)).expect("scratch read");
        stats("tmp_b[0..e] (tail-norm output)", &tmp);
        if let Some(xr) = xb {
            stats("x_b last residual row", &xr);
        }
    }

    // Decode continuation probe: the chunk seeded the recurrent states —
    // one teacher-forced decode step; garbage here ⇒ the states (layer
    // loop) are poisoned, sane here ⇒ the defect is the tail stage alone.
    let next = ids[prefix];
    let (am, dlogits, _) = fwd
        .forward_token_capture(next, prefix, &[])
        .expect("decode continuation");
    stats("decode continuation logits", &dlogits);
    println!("[bisect] decode argmax={am} target={}", ids[prefix + 1]);
}
