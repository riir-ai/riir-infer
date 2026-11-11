//! Issue 1199 T3 — the loopback alternation-tax cell (PROVISIONAL posture;
//! the exclusive-window re-run is the verdict half). Four arms over the
//! synth Bonsai-2, each a fresh head+tail pair joined before the next:
//!
//! 1. `serial`     — 1 stream, per-frame path. The baseline.
//! 2. `rr`         — 2 streams, pair OFF (the per-frame round-robin — the
//!    taxed arm 1198's T1 evidence measured).
//! 3. `batch`      — 2 streams, pair ON, interleave OFF (the T2.4 fused
//!    batch: whole-step chains back to back).
//! 4. `batch+il`   — 2 streams, pair ON, interleave ON (1199 rung b —
//!    layer-granular interleave).
//!
//! `RIIR_DIST_STEP_TIMING=1` prints the per-64-step phase decomposition
//! per role per arm on stderr; the example also prints a one-line
//! aggregate wall per arm. Run RELEASE (`--release`), box state recorded
//! beside the numbers. The interleave arm is read per batch frame, so the
//! example flips it between arms in-process.

#[path = "../../../tests/common/synth_bonsai2.rs"]
mod synth;

use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use riir_infer_core::deltanet::{QwenDeltaNetTernaryWeights, WeightScope};
use riir_infer_core::multi_host::LayerRange;
use riir_infer_core::types::Config;
use riir_infer_dist::multi::{MultiRequest, run_head_multi, run_tail_multi};
use riir_infer_dist::runner::{HeadOptions, HeadReport};
use riir_infer_dist::stage::{ArtifactIdentity, StageCompute, TernaryCpuStage};
use riir_infer_dist::transport::{Direction, PeerKey, TcpTransport};

const PROMPT: [u32; 8] = [7, 13, 42, 100, 5, 200, 33, 90];
const MAX_NEW: usize = 192; // 3 instrumented windows per stream at s=2

fn set_env(k: &str, v: &str) {
    // Edition 2024. Safety: called on the main thread between arms; the
    // arm's readers (the tail thread of the CURRENT arm) were joined.
    unsafe { std::env::set_var(k, v) };
}

/// The instrument's per-window accounting is keyed on `steps`, which the
/// fused-batch arm counts per ROW at receive time — so its fwd window
/// flushes on a misaligned grid and the batch path never advances the
/// per-frame reply counter. Direct `run_tail_multi` timing (non-batch
/// arms) reads clean; the batched arms are adjudicated on the whole-arm
/// wall + the head's recv window instead (also printed by the
/// instrument).
fn load() -> (Config, Arc<QwenDeltaNetTernaryWeights>, ArtifactIdentity) {
    let bytes = synth::build_gguf(
        &synth::synth_metadata(false),
        &synth::synth_tensors_bonsai2(),
    );
    let path = synth::write_tmp("riir_i1199_cell", &bytes);
    let (c, w) =
        riir_infer_core::gguf_loader::load_qwen_deltanet_ternary_weights_gguf_scoped(&path, |c| {
            WeightScope::full(c.n_layer)
        })
        .expect("load synth");
    let id = ArtifactIdentity::of_gguf(&path).expect("identity");
    let _ = std::fs::remove_file(&path);
    (c, Arc::new(w), id)
}

/// The real-geometry arm (the T3 deciding cell): a real GGUF checkpoint
/// loaded FULL-scope once and Arc-shared by every stage — `require_coverage`
/// lets a full view serve any stage range, and the stage reads only its own
/// window's tensors. One resident copy (~14 GB for the PQ2_0 27B), the
/// same checkpoint class the 2-box CPU cells ran.
fn load_real(gguf: &str) -> (Config, Arc<QwenDeltaNetTernaryWeights>, ArtifactIdentity) {
    let path = std::path::PathBuf::from(gguf);
    let id = ArtifactIdentity::of_gguf(&path).expect("identity");
    let (c, w) = riir_infer_core::gguf_loader::load_qwen_deltanet_ternary_weights_gguf_scoped(
        &path,
        |cfg| WeightScope::full(cfg.n_layer),
    )
    .expect("load real");
    eprintln!(
        "[i1199-cell] real model: n_layer={} n_embd={} block={}",
        c.n_layer, c.n_embd, c.block_size
    );
    (c, Arc::new(w), id)
}

#[allow(clippy::too_many_arguments)]
fn run_arm(
    name: &str,
    config: &Config,
    w: &Arc<QwenDeltaNetTernaryWeights>,
    id: ArtifactIdentity,
    streams: usize,
    pair_decode: bool,
    max_new: usize,
) -> Vec<HeadReport> {
    let n = config.n_layer as u32;
    let k = n / 2;
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let (tc, tw) = (config.clone(), w.clone());
    let key = [9u8; 32];
    let tail = thread::spawn(move || {
        let mut stages: Vec<Box<dyn StageCompute>> = (0..streams)
            .map(|_| {
                Box::new(
                    TernaryCpuStage::new(tc.clone(), tw.clone(), LayerRange::new(k, n), id)
                        .unwrap(),
                ) as Box<dyn StageCompute>
            })
            .collect();
        let (sock, _) = listener.accept().unwrap();
        let mut link = TcpTransport::new(sock, PeerKey(key), Direction::Upstream).unwrap();
        let mut refs: Vec<&mut dyn StageCompute> = stages
            .iter_mut()
            .map(|b| b.as_mut() as &mut dyn StageCompute)
            .collect();
        run_tail_multi(&mut refs, &mut link, |_, _, _| {}).expect("tail run")
    });
    let mut head_stages: Vec<Box<dyn StageCompute>> = (0..streams)
        .map(|_| {
            Box::new(
                TernaryCpuStage::new(config.clone(), w.clone(), LayerRange::new(0, k), id).unwrap(),
            ) as Box<dyn StageCompute>
        })
        .collect();
    let sock = TcpStream::connect(addr).unwrap();
    let mut link = TcpTransport::new(sock, PeerKey(key), Direction::Downstream).unwrap();
    let reqs: Vec<MultiRequest> = (0..streams)
        .map(|_| MultiRequest::new(PROMPT.to_vec(), max_new))
        .collect();
    let opts = HeadOptions {
        pair_decode,
        // No EOS refusal noise on the synth fixture: decode the full window.
        eos: None,
        ..HeadOptions::default()
    };
    let mut refs: Vec<&mut dyn StageCompute> = head_stages
        .iter_mut()
        .map(|b| b.as_mut() as &mut dyn StageCompute)
        .collect();
    let t0 = Instant::now();
    let reports = run_head_multi(&mut refs, &mut link, &reqs, opts).expect("head run");
    let wall = t0.elapsed();
    let tail_steps = tail.join().expect("tail join");
    let tokens: usize = reports.iter().map(|r| r.generated.len()).sum();
    eprintln!(
        "[i1199-cell] arm={name} streams={streams} pair={pair_decode} wall={wall:?} tail_steps={tail_steps} tokens={tokens}"
    );
    reports
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let warm = args.iter().any(|a| a == "--warm");
    let gguf = args
        .iter()
        .position(|a| a == "--gguf")
        .and_then(|i| args.get(i + 1))
        .cloned();
    // Real-geometry arms decode fewer tokens (each step is ~1000× the
    // synth step): 3 instrumented windows per stream at s=2.
    let max_new: usize = if gguf.is_some() { 32 } else { MAX_NEW };
    set_env("RIIR_DIST_STEP_TIMING", "1");
    let (config, w, id) = match &gguf {
        Some(p) => load_real(p),
        None => load(),
    };

    if warm {
        run_arm("warmup", &config, &w, id, 1, false, max_new);
    }

    // Arm 1: serial.
    run_arm("serial", &config, &w, id, 1, false, max_new);
    // Arm 2: 2-stream round-robin, per-frame path (the taxed arm).
    run_arm("rr", &config, &w, id, 2, false, max_new);
    // Arm 3: fused batch, interleave OFF.
    set_env("RIIR_TAIL_LAYER_INTERLEAVE", "0");
    run_arm("batch", &config, &w, id, 2, true, max_new);
    // Arm 4: fused batch, interleave ON (1199 rung b).
    set_env("RIIR_TAIL_LAYER_INTERLEAVE", "1");
    run_arm("batch+il", &config, &w, id, 2, true, max_new);
    set_env("RIIR_TAIL_LAYER_INTERLEAVE", "0");
    eprintln!("[i1199-cell] done — see step-timing[head|tail] lines per arm above");
}
