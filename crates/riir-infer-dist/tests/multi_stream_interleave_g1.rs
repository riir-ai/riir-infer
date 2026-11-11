//! Issue 1199 rung (b) — the tail's fused-batch LAYER interleave
//! (`RIIR_TAIL_LAYER_INTERLEAVE=1`, promote-only): per-stream tokens
//! BIT-IDENTICAL to the 1-box oracle through the interleave path, the
//! split's step composition (`begin` → ascending layers → `end`)
//! bit-equal to the whole-step call at the stage level, and split
//! mis-use refusing loud. This file is its own test binary on purpose:
//! it arms the env once at startup and every test in it runs the
//! interleave path (the incumbent batch path is pinned by
//! `multi_stream_batch.rs`, which never arms).

#[path = "../../../tests/common/synth_bonsai2.rs"]
mod synth;

use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::thread;

use riir_infer_core::deltanet::{QwenDeltaNetTernaryWeights, WeightScope};
use riir_infer_core::multi_host::LayerRange;
use riir_infer_core::types::Config;
use riir_infer_dist::multi::{MultiRequest, run_head_multi, run_tail_multi};
use riir_infer_dist::runner::{HeadOptions, HeadReport, run_local_with};
use riir_infer_dist::stage::{ArtifactIdentity, StageCompute, TernaryCpuStage};
use riir_infer_dist::transport::{Direction, PeerKey, TcpTransport};

// Edition 2024: env mutation is unsafe. Safety: this runs before any
// thread that reads the arm spawns, and the value never changes for the
// life of this test binary.
fn arm_interleave() {
    unsafe { std::env::set_var("RIIR_TAIL_LAYER_INTERLEAVE", "1") };
}

const PROMPT_A: [u32; 5] = [7, 13, 42, 100, 5];
const MAX_NEW: usize = 6;

fn load(name: &str) -> (Config, Arc<QwenDeltaNetTernaryWeights>, ArtifactIdentity) {
    let bytes = synth::build_gguf(
        &synth::synth_metadata(false),
        &synth::synth_tensors_bonsai2(),
    );
    let path = synth::write_tmp(name, &bytes);
    let (c, w) =
        riir_infer_core::gguf_loader::load_qwen_deltanet_ternary_weights_gguf_scoped(&path, |c| {
            WeightScope::full(c.n_layer)
        })
        .expect("load synth");
    let id = ArtifactIdentity::of_gguf(&path).expect("identity");
    let _ = std::fs::remove_file(&path);
    (c, Arc::new(w), id)
}

fn oracle(
    config: &Config,
    w: &Arc<QwenDeltaNetTernaryWeights>,
    id: ArtifactIdentity,
    prompt: &[u32],
    max_new: usize,
) -> Vec<u32> {
    let n = config.n_layer as u32;
    let mut s = TernaryCpuStage::new(config.clone(), w.clone(), LayerRange::full(n), id).unwrap();
    run_local_with(&mut s, prompt, max_new, HeadOptions::default())
}

/// One s-stream paired run over loopback TCP at split `k`. The interleave
/// arm is read per batch frame, so this process (armed at startup) always
/// interleaves its fused batches.
#[allow(clippy::too_many_arguments)]
fn run_multi(
    config: &Config,
    w: &Arc<QwenDeltaNetTernaryWeights>,
    id: ArtifactIdentity,
    k: u32,
    key: [u8; 32],
    prompts: &[&[u32]],
    max_new: usize,
) -> (std::io::Result<Vec<HeadReport>>, std::io::Result<usize>) {
    let n = config.n_layer as u32;
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let (tc, tw) = (config.clone(), w.clone());
    let n_streams = prompts.len();
    let tail = thread::spawn(move || {
        let mut stages: Vec<Box<dyn StageCompute>> = (0..n_streams)
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
        run_tail_multi(&mut refs, &mut link, |_, _, _| {})
    });
    let mut head_stages: Vec<Box<dyn StageCompute>> = (0..n_streams)
        .map(|_| {
            Box::new(
                TernaryCpuStage::new(config.clone(), w.clone(), LayerRange::new(0, k), id).unwrap(),
            ) as Box<dyn StageCompute>
        })
        .collect();
    let sock = TcpStream::connect(addr).unwrap();
    let mut link = TcpTransport::new(sock, PeerKey(key), Direction::Downstream).unwrap();
    let reqs: Vec<MultiRequest> = prompts
        .iter()
        .map(|p| MultiRequest::new(p.to_vec(), max_new))
        .collect();
    let opts = HeadOptions {
        // Pair the decode feeds so the head hints BatchN — the interleave
        // rides the fused-batch arm.
        pair_decode: true,
        ..HeadOptions::default()
    };
    let mut refs: Vec<&mut dyn StageCompute> = head_stages
        .iter_mut()
        .map(|b| b.as_mut() as &mut dyn StageCompute)
        .collect();
    let head = run_head_multi(&mut refs, &mut link, &reqs, opts);
    let tail = tail.join().unwrap();
    (head, tail)
}

/// THE gate: 2 streams through the interleave path land exactly the
/// 1-box oracle's tokens (per stream) — the split's begin→layers→end
/// composition is the same computation as the whole-step call.
#[test]
fn interleave_tokens_bit_identical_to_one_box_oracle() {
    arm_interleave();
    let (config, w, id) = load("riir_dist_interleave_g1");
    let prompt_b: [u32; 8] = [3, 1, 4, 1, 5, 9, 2, 6];
    let prompts: [&[u32]; 2] = [&PROMPT_A, &prompt_b];
    let (head, tail) = run_multi(&config, &w, id, 2, [31u8; 32], &prompts, MAX_NEW);
    let reports = head.expect("interleave head");
    let steps = tail.expect("interleave tail");
    for (i, r) in reports.iter().enumerate() {
        assert_eq!(
            r.generated,
            oracle(&config, &w, id, prompts[i], MAX_NEW),
            "interleaved stream {i} diverged from the 1-box oracle"
        );
    }
    assert!(steps > 0, "no steps served");
}

/// Stage-level composition law: `begin` → every layer ascending → `end`
/// returns EXACTLY what `step` returns (bit-equal), on the full range
/// (logits out) and on a sub range fed a real head-half hidden row
/// (hidden out) — the whole-step call is its own split, walked by the
/// same code.
#[test]
fn split_composition_bit_equals_whole_step() {
    arm_interleave();
    let (config, w, id) = load("riir_dist_interleave_split_law");
    let n = config.n_layer as u32;
    let empty: [f32; 0] = [];

    let probe = TernaryCpuStage::new(config.clone(), w.clone(), LayerRange::full(n), id).unwrap();
    assert!(
        probe.layer_split_supported(),
        "cpu stage must support the split"
    );
    let (lo, hi) = probe.layer_split_range();
    assert_eq!((lo, hi), (0, n as usize), "full-range split window");

    for pos in [0usize, 1, 5, 17] {
        let mut a =
            TernaryCpuStage::new(config.clone(), w.clone(), LayerRange::full(n), id).unwrap();
        let mut b =
            TernaryCpuStage::new(config.clone(), w.clone(), LayerRange::full(n), id).unwrap();
        let la = a.step(PROMPT_A[0], pos, &empty).to_vec();
        b.step_begin(PROMPT_A[0], pos, &empty);
        for layer in lo..hi {
            b.step_layer(layer);
        }
        let lb = b.step_end().to_vec();
        assert_eq!(
            la, lb,
            "split composition diverged from step() at pos {pos} (bitwise)"
        );
    }

    // A sub range (the tail stage posture): hidden out, fed a real
    // upstream hidden row — a non-first stage's step copies `input`.
    let k = n / 2;
    for pos in [0usize, 3, 9] {
        let hidden_in = {
            let mut h =
                TernaryCpuStage::new(config.clone(), w.clone(), LayerRange::new(0, k), id).unwrap();
            h.step(PROMPT_A[0], pos, &empty).to_vec()
        };
        let mut a =
            TernaryCpuStage::new(config.clone(), w.clone(), LayerRange::new(k, n), id).unwrap();
        let mut b =
            TernaryCpuStage::new(config.clone(), w.clone(), LayerRange::new(k, n), id).unwrap();
        let la = a.step(PROMPT_A[0], pos, &hidden_in).to_vec();
        let (slo, shi) = b.layer_split_range();
        assert_eq!(
            (slo, shi),
            (k as usize, n as usize),
            "sub-range split window"
        );
        b.step_begin(PROMPT_A[0], pos, &hidden_in);
        for layer in slo..shi {
            b.step_layer(layer);
        }
        let lb = b.step_end().to_vec();
        assert_eq!(la, lb, "sub-range split diverged from step() at pos {pos}");
    }
}

/// Split mis-use refuses loud: a layer skip, a repeat, a double begin,
/// an early end, and calls outside a split each panic — a mis-ordered
/// interleave is a protocol bug, never a silent skip.
#[test]
fn split_misuse_refuses_loud() {
    arm_interleave();
    let (config, w, id) = load("riir_dist_interleave_misuse");
    let n = config.n_layer as u32;
    let empty: [f32; 0] = [];
    let expect_panic = |name: &str, f: &dyn Fn(&mut TernaryCpuStage)| {
        let mut s =
            TernaryCpuStage::new(config.clone(), w.clone(), LayerRange::full(n), id).unwrap();
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(&mut s)));
        assert!(caught.is_err(), "{name}: must refuse loud");
    };
    expect_panic("layer skip", &|s| {
        s.step_begin(PROMPT_A[0], 0, &empty);
        s.step_layer(1); // skips 0
    });
    expect_panic("layer repeat", &|s| {
        s.step_begin(PROMPT_A[0], 0, &empty);
        s.step_layer(0);
        s.step_layer(0); // repeats 0
    });
    expect_panic("double begin", &|s| {
        s.step_begin(PROMPT_A[0], 0, &empty);
        s.step_begin(PROMPT_A[0], 1, &empty);
    });
    expect_panic("early end", &|s| {
        s.step_begin(PROMPT_A[0], 0, &empty);
        s.step_layer(0);
        let _ = s.step_end(); // layers 1.. unrun
    });
    expect_panic("end outside split", &|s| {
        let _ = s.step_end();
    });
    expect_panic("layer outside split", &|s| {
        s.step_layer(0);
    });
}
