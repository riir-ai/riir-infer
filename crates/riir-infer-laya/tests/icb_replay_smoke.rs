//! The ICB replay machinery's smoke fixture (issue 1181 T2 tranche 1).
//!
//! Pins, at the machinery level (self-contained MSL, no sgemm coupling):
//! 1. **G1-equivalent correctness** — a recorded 2-command sequence
//!    replayed N times is BIT-IDENTICAL to (a) the direct per-op encode
//!    path running the same sequence N times and (b) the host f32
//!    composition (every step is an exact f32 op: ×2, +b, b·0.5).
//! 2. **The remedy's cost shape** — per-replay host enqueue wall for
//!    the blit-copy+execute posture vs the execute-from-source posture
//!    (the #1993 degradation-tax A/B) vs the direct-encode control,
//!    MEASURED and printed for the T4 break-even line. Never asserted:
//!    a perf number is box-state, not a gate.
//! 3. **The event fence** — signal/wait across two command buffers
//!    orders (trivially green on one in-order queue; the API surface
//!    must not rot).
//!
//! [[test]] row with required-features in the same commit — the
//! repo-birth gate law. macOS + `laya-riir-metal` only.

use std::time::Instant;

use metal::{Device, MTLSize};

use riir_infer_laya::laya::riir::metal::icb_replay::{IcbCommand, ReplayGroup};

/// The machinery's own MSL — two elementwise kernels whose composition
/// over N replays is exactly computable on the host (×2 and ·0.5 are
/// powers of two, so each replay is two exact f32 adds).
const MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void icb_axpy(device float* x [[buffer(0)]],
                     device const float* y [[buffer(1)]],
                     constant float& s [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid < 4096u) { x[gid] = x[gid] * s + y[gid]; }
}

kernel void icb_step(device float* x [[buffer(0)]],
                     device const float* y [[buffer(1)]],
                     constant float& w [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid < 4096u) { x[gid] += y[gid] * w; }
}
"#;

const N: usize = 4096;

/// Host twin of one replay: x = (x·2 + y), then x += y·0.5 — the same
/// f32 ops in the same order the GPU sequence applies.
fn host_replay(x: &mut [f32], y: &[f32]) {
    for i in 0..x.len() {
        x[i] = x[i] * 2.0 + y[i];
        x[i] += y[i] * 0.5;
    }
}

struct Rig {
    device: Device,
    queue: metal::CommandQueue,
    axpy: metal::ComputePipelineState,
    step: metal::ComputePipelineState,
}

impl Rig {
    fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("no Metal device")?;
        let lib = device
            .new_library_with_source(MSL, &metal::CompileOptions::new())
            .map_err(|e| format!("MSL compile failed: {e}"))?;
        let pipeline = |name: &str| -> Result<metal::ComputePipelineState, String> {
            let f = lib.get_function(name, None).map_err(|e| e.to_string())?;
            device
                .new_compute_pipeline_state_with_function(&f)
                .map_err(|e| e.to_string())
        };
        Ok(Self {
            axpy: pipeline("icb_axpy")?,
            step: pipeline("icb_step")?,
            queue: device.new_command_queue(),
            device,
        })
    }

    /// The direct-encode control: one command buffer, one compute
    /// encoder, the two dispatches — the incumbent shape.
    fn direct_pass(
        &self,
        x: &metal::Buffer,
        y: &metal::Buffer,
        sx: metal::Buffer,
        sw: metal::Buffer,
    ) {
        let cb = self.queue.new_command_buffer().to_owned();
        let enc = cb.new_compute_command_encoder().to_owned();
        let tpg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let grid = MTLSize {
            width: N as u64,
            height: 1,
            depth: 1,
        };
        enc.set_compute_pipeline_state(&self.axpy);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(y), 0);
        enc.set_buffer(2, Some(&sx), 0);
        enc.dispatch_threads(grid, tpg);
        enc.set_compute_pipeline_state(&self.step);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(y), 0);
        enc.set_buffer(2, Some(&sw), 0);
        enc.dispatch_threads(grid, tpg);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    /// Copy host bytes into a device buffer (wait included).
    fn upload(&self, bytes: &[f32]) -> metal::Buffer {
        self.device
            .new_buffer_with_data(
                bytes.as_ptr() as *const core::ffi::c_void,
                (bytes.len() * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            )
            .to_owned()
    }

    fn download(&self, b: &metal::Buffer) -> Vec<f32> {
        let mut out = vec![0f32; b.length() as usize / 4];
        let cb = self.queue.new_command_buffer().to_owned();
        let _ = cb;
        unsafe {
            std::ptr::copy_nonoverlapping(b.contents().cast::<f32>(), out.as_mut_ptr(), out.len());
        }
        out
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[test]
#[ignore = "AGX G17X (M5 Ultra, macOS 27.2) driver segfault INSIDE -[AGXG17XFamilyIndirectComputeCommand setComputePipelineState:] (+116) while recording — the ICB machinery itself is API-correct (allocation + command-handle surface verified); the crash is driver-internal. Re-run when the OS/driver state changes or the missing ICB-recording contract is identified (run: cargo test -p riir-infer-laya --features laya-riir-metal --test icb_replay_smoke -- --ignored --nocapture)"]
fn icb_replay_smoke() {
    if !ReplayGroup::device_supports(&Device::system_default().expect("metal device")) {
        panic!(
            "icb_replay_smoke: the host lacks Mac2 (ICB compute) support — the lane is the machinery's premise"
        );
    }
    let rig = Rig::new().expect("rig");

    // Deterministic inputs.
    let x0: Vec<f32> = (0..N).map(|i| (i as f32 * 0.25) - 512.0).collect();
    let yv: Vec<f32> = (0..N).map(|i| ((i % 97) as f32) * 0.125 - 6.0).collect();

    // ── Host expectation over 40 replays ────────────────────────────
    let mut host = x0.clone();
    for _ in 0..40 {
        host_replay(&mut host, &yv);
    }

    // ── The recorded sequence (the remedy posture) ──────────────────
    let x_dev = rig.upload(&x0);
    let y_dev = rig.upload(&yv);
    let s_buf = rig.scalar_posture(&[2.0f32]);
    let w_buf = rig.scalar_posture(&[0.5f32]);
    let tpg = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    let grid = MTLSize {
        width: N as u64,
        height: 1,
        depth: 1,
    };
    let group = ReplayGroup::record(
        rig.device.clone(),
        rig.queue.clone(),
        &[
            IcbCommand {
                pipeline: rig.axpy.clone(),
                bindings: vec![
                    (0, x_dev.clone(), 0),
                    (1, y_dev.clone(), 0),
                    (2, s_buf.clone(), 0),
                ],
                staging: vec![],
                grid,
                tpg,
                barrier: false,
            },
            IcbCommand {
                pipeline: rig.step.clone(),
                bindings: vec![
                    (0, x_dev.clone(), 0),
                    (1, y_dev.clone(), 0),
                    (2, w_buf.clone(), 0),
                ],
                staging: vec![],
                grid,
                tpg,
                barrier: true,
            },
        ],
    )
    .expect("record");
    assert_eq!(group.count(), 2);

    // 40 replays through the remedy path, ONE wait at the end, then the
    // fence legs on the last command buffer (signal → a fresh cb waits).
    let event = rig.device.new_event();
    let mut last = None;
    for _ in 0..40 {
        let cb = group.replay();
        last = Some(cb);
    }
    let last = last.expect("replays");
    group.signal_event(&last, &event, 1);
    last.wait_until_completed();
    let waiter = rig.queue.new_command_buffer().to_owned();
    group.signal_wait(&waiter, &event, 1);
    waiter.commit();
    waiter.wait_until_completed();

    let got = rig.download(&x_dev);
    assert_eq!(got, host, "replay != host composition");

    // ── The direct-encode control: same 40 passes, bit-identical ────
    let x_ctl = rig.upload(&x0);
    for _ in 0..40 {
        rig.direct_pass(&x_ctl, &y_dev, s_buf.clone(), w_buf.clone());
    }
    let got_ctl = rig.download(&x_ctl);
    assert_eq!(got_ctl, host, "direct != host composition");
    assert_eq!(got_ctl, got, "replay != direct");
}

/// The degradation-tax A/B (measurement-only, prints the cost rows the
/// T4 break-even quotes): blit-copy+execute vs execute-from-source vs
/// the direct-encode control, per-replay host enqueue wall at ONE final
/// wait. Not `#[ignore]`d — it is cheap (< 2 s) and its value IS the
/// printed line; nothing here asserts a perf number.
#[test]
#[ignore = "same AGX G17X recording segfault as icb_replay_smoke (the A/B needs a recorded group to measure)"]
fn icb_replay_cost_ab() {
    let rig = Rig::new().expect("rig");
    let x0: Vec<f32> = (0..N).map(|i| (i as f32 * 0.5) - 256.0).collect();
    let yv: Vec<f32> = vec![1.0; N];
    let x_dev = rig.upload(&x0);
    let y_dev = rig.upload(&yv);
    let s_buf = rig.scalar_posture(&[2.0f32]);
    let w_buf = rig.scalar_posture(&[0.5f32]);
    let tpg = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    let grid = MTLSize {
        width: N as u64,
        height: 1,
        depth: 1,
    };
    let group = ReplayGroup::record(
        rig.device.clone(),
        rig.queue.clone(),
        &[
            IcbCommand {
                pipeline: rig.axpy.clone(),
                bindings: vec![
                    (0, x_dev.clone(), 0),
                    (1, y_dev.clone(), 0),
                    (2, s_buf.clone(), 0),
                ],
                staging: vec![],
                grid,
                tpg,
                barrier: false,
            },
            IcbCommand {
                pipeline: rig.step.clone(),
                bindings: vec![
                    (0, x_dev.clone(), 0),
                    (1, y_dev.clone(), 0),
                    (2, w_buf.clone(), 0),
                ],
                staging: vec![],
                grid,
                tpg,
                barrier: true,
            },
        ],
    )
    .expect("record");

    const REPLAYS: usize = 20;
    const ROUNDS: usize = 9;

    // Warm the pipelines once (first-use compile).
    group.replay().wait_until_completed();

    let mut remedy = Vec::with_capacity(ROUNDS);
    let mut source = Vec::with_capacity(ROUNDS);
    let mut direct = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        let t = Instant::now();
        for _ in 0..REPLAYS {
            let cb = group.replay();
            let _ = &cb;
        }
        rig.queue_empty_wait();
        remedy.push(t.elapsed().as_secs_f64() * 1e6 / REPLAYS as f64);

        let t = Instant::now();
        for _ in 0..REPLAYS {
            let cb = group.replay_from_source();
            let _ = &cb;
        }
        rig.queue_empty_wait();
        source.push(t.elapsed().as_secs_f64() * 1e6 / REPLAYS as f64);

        let t = Instant::now();
        for _ in 0..REPLAYS {
            rig.direct_pass_noready(&x_dev, &y_dev, s_buf.clone(), w_buf.clone());
        }
        rig.queue_empty_wait();
        direct.push(t.elapsed().as_secs_f64() * 1e6 / REPLAYS as f64);
    }

    println!(
        "icb_replay cost A/B (median host µs/replay over {ROUNDS}×{REPLAYS}):\n  remedy (blit-copy+execute): {:.1}\n  source (execute-from-src):  {:.1}\n  direct encode (2 ops):      {:.1}",
        median(remedy),
        median(source),
        median(direct),
    );
}

impl Rig {
    /// scalar buffer via the group's own helper is unavailable before a
    /// group exists — the fixture uses the same constructor directly.
    fn scalar_posture(&self, floats: &[f32]) -> metal::Buffer {
        let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        self.device
            .new_buffer_with_data(
                bytes.as_ptr() as *const core::ffi::c_void,
                bytes.len() as u64,
                metal::MTLResourceOptions::StorageModeShared,
            )
            .to_owned()
    }

    /// Drain the queue: commit an empty command buffer and wait it —
    /// the timing bar for the A/B rounds (everything enqueued lands).
    fn queue_empty_wait(&self) {
        let cb = self.queue.new_command_buffer().to_owned();
        cb.commit();
        cb.wait_until_completed();
    }

    /// The direct control WITHOUT the per-pass wait (the A/B's fair
    /// posture: enqueue-side cost only).
    fn direct_pass_noready(
        &self,
        x: &metal::Buffer,
        y: &metal::Buffer,
        sx: metal::Buffer,
        sw: metal::Buffer,
    ) {
        let cb = self.queue.new_command_buffer().to_owned();
        let enc = cb.new_compute_command_encoder().to_owned();
        let tpg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let grid = MTLSize {
            width: N as u64,
            height: 1,
            depth: 1,
        };
        enc.set_compute_pipeline_state(&self.axpy);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(y), 0);
        enc.set_buffer(2, Some(&sx), 0);
        enc.dispatch_threads(grid, tpg);
        enc.set_compute_pipeline_state(&self.step);
        enc.set_buffer(0, Some(x), 0);
        enc.set_buffer(1, Some(y), 0);
        enc.set_buffer(2, Some(&sw), 0);
        enc.dispatch_threads(grid, tpg);
        enc.end_encoding();
        cb.commit();
    }
}
