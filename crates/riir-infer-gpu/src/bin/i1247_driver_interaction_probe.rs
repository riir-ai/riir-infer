//! Issue 1247 — the m3 per-driver-interaction decomposition probe.
//!
//! Issue 1203 attributed the m3 dense-lane stall to per-driver-interaction
//! latency (`write_buffer` 3–20 ms, `queue.submit` 5–33 ms, REGULATE blocks
//! 2.6–16.7 s — healthy is µs-class) and left the ABSOLUTE layer unexplained.
//! This probe decomposes that layer with public wgpu API only — no model, no
//! cubecl forward — so each cost class is measured in isolation:
//!
//!   CELL A  `device.create_buffer` — the fresh per-write `MTLBuffer`
//!           allocation wgpu-core's `StagingBuffer::new` pays on EVERY
//!           `Queue::write_buffer` (StorageModeShared +
//!           CPUCacheModeWriteCombined + TRANSIENT). Timed across a size
//!           ladder spanning the decode working set's page/VM classes.
//!   CELL B  `queue.write_buffer` — the full write path (fresh staging
//!           buffer + memcpy + blit encode + pending-writes bookkeeping)
//!           into a persistent target, same size ladder.
//!   CELL C  `queue.submit` — an empty command encoder, one commit per call
//!           (the write-flush and task-flush shape).
//!   CELL D  `device.poll(Wait{submission_index})` — the REGULATE /
//!           write-flush poll path, paced against a trivial submission.
//!   CELL E  (pressure) repeats CELL B and A with a resident set approximating
//!           the model's footprint — tests the pressure-dependent
//!           allocation-latency hypothesis that amo/m5ultra (no stall) and
//!           m3 (stall) could otherwise not be told apart.
//!
//! Instruments: host wall clock ONLY. GPU timestamps are banned on this box
//! (macOS 26+ returns zeros, gfx-rs #9414). Every cell prints per-size
//! mean / p50 / p95 over its repeats; every run prints box provenance.
//!
//! Usage:
//!   i1247_driver_interaction_probe [--sizes 4k,1m,64m] [--iters N] \
//!     [--pressure-gb N] [--skip-pressure]
//!
//! The probe never gates a promotion — it is the issue's diagnostic
//! instrument. Findings land in `.benchmarks/1247_*.md` and the issue.

#![cfg(all(target_os = "macos", feature = "metal_tensor_gemm"))]

use std::time::Instant;

struct Cfg {
    sizes: Vec<u64>,
    iters: usize,
    pressure_gb: u64,
    skip_pressure: bool,
}

fn parse_sizes(spec: &str) -> Vec<u64> {
    let mut out = Vec::new();
    for tok in spec.split(',') {
        let t = tok.trim().to_ascii_lowercase();
        if t.is_empty() {
            continue;
        }
        let (num, unit) = t.split_at(t.find(|c: char| c.is_ascii_alphabetic()).unwrap_or(t.len()));
        let n: u64 = num
            .parse()
            .unwrap_or_else(|e| panic!("bad size `{tok}`: {e}"));
        let mult = match unit {
            "k" | "kb" => 1024,
            "m" | "mb" => 1024 * 1024,
            "g" | "gb" => 1024 * 1024 * 1024,
            "" => 1,
            other => panic!("bad size unit `{other}` in `{tok}`"),
        };
        out.push(n * mult);
    }
    if out.is_empty() {
        out = vec![4 * 1024, 64 * 1024, 1024 * 1024, 64 * 1024 * 1024];
    }
    out
}

fn parse_args() -> Cfg {
    let args: Vec<String> = std::env::args().collect();
    let mut cfg = Cfg {
        sizes: Vec::new(),
        iters: 24,
        pressure_gb: 10,
        skip_pressure: false,
    };
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--sizes" => {
                i += 1;
                cfg.sizes = parse_sizes(&args[i]);
            }
            "--iters" => {
                i += 1;
                cfg.iters = args[i].parse().expect("iters");
            }
            "--pressure-gb" => {
                i += 1;
                cfg.pressure_gb = args[i].parse().expect("pressure-gb");
            }
            "--skip-pressure" => cfg.skip_pressure = true,
            other => {
                eprintln!(
                    "[i1247] unknown arg {other} (flags: --sizes --iters --pressure-gb --skip-pressure)"
                );
                std::process::exit(2);
            }
        }
        i += 1;
    }
    if cfg.sizes.is_empty() {
        cfg.sizes = parse_sizes("");
    }
    cfg
}

fn pct(sorted: &[u128], p: f64) -> u128 {
    // Last-index floor guard: the p95 of a 24-sample run reads the top
    // sample — disclose it as the tail, never a percentile claim (the
    // percentile-floor law: quote tail support beside it).
    let idx = ((sorted.len() as f64) * p) as usize;
    let idx = idx.min(sorted.len() - 1);
    sorted[idx]
}

fn print_cell(name: &str, size: u64, mut samples: Vec<u128>) {
    samples.sort_unstable();
    let n = samples.len();
    let mean: u128 = samples.iter().sum::<u128>() / n.max(1) as u128;
    println!(
        "[i1247] {:<24} size={:>10} n={:>3}  mean={:>9} ns  p50={:>9}  p95={:>9}  (p95 tail support {})",
        name,
        size,
        n,
        mean,
        pct(&samples, 0.50),
        pct(&samples, 0.95),
        n - (n as f64 * 0.95) as usize,
    );
}

fn device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::METAL,
        backend_options: wgpu::BackendOptions::default(),
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        display: None,
    });
    let adapters = futures_block(instance.enumerate_adapters(wgpu::Backends::METAL));
    println!("[i1247] metal adapters: {}", adapters.len());
    let adapter = futures_block(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .expect("no Metal adapter");
    let info = adapter.get_info();
    println!(
        "[i1247] adapter: {} {:?} (device {}, driver {})",
        info.name, info.backend, info.device, info.driver
    );
    let (device, queue) = futures_block(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("i1247-probe"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
        experimental_features: wgpu::ExperimentalFeatures::default(),
    }))
    .expect("no device pair");
    (device, queue)
}

fn futures_block<T>(fut: impl std::future::Future<Output = T>) -> T {
    // The wgpu request API is async; the probe is single-threaded and
    // short-lived — drive the future with a no-op waker; wgpu's own
    // initialization work happens on the instance's internal machinery and
    // device polling drives completion, so spin with short sleeps until
    // Ready (a 30 s hard bound refuses a wedged adapter).
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
    fn noop(_: *const ()) {}
    fn noop_clone(p: *const ()) -> RawWaker {
        RawWaker::new(p, &VTABLE)
    }
    static VTABLE: RawWakerVTable = RawWakerVTable::new(noop_clone, noop, noop, noop);
    let waker = unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) };
    let mut cx = Context::from_waker(&waker);
    let mut fut = Box::pin(fut);
    let deadline = Instant::now() + std::time::Duration::from_secs(30);
    loop {
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(v) => return v,
            Poll::Pending => {
                if Instant::now() > deadline {
                    panic!("[i1247] device request did not resolve in 30s");
                }
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        }
    }
}

/// Time `f` `iters` times, with 3 untimed warm-ups first.
#[allow(dead_code)]
fn time_iters<F: FnMut()>(iters: usize, mut f: F) -> Vec<u128> {
    let mut out = Vec::with_capacity(iters);
    // 3 warm-up iterations, untimed.
    for _ in 0..3 {
        f();
    }
    for _ in 0..iters {
        let t = Instant::now();
        f();
        out.push(t.elapsed().as_nanos());
    }
    out
}

fn cell_a_create_buffer(device: &wgpu::Device, sizes: &[u64], iters: usize) {
    println!("\n[i1247] === CELL A: device.create_buffer (the per-write MTLBuffer alloc) ===");
    for &size in sizes {
        let mut samples = Vec::new();
        for _ in 0..3 {
            let _b = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("(i1247) staging"),
                size,
                usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
        }
        for _ in 0..iters {
            let t = Instant::now();
            let b = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("(i1247) staging"),
                size,
                usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            samples.push(t.elapsed().as_nanos());
            drop(b);
        }
        print_cell("A create_buffer", size, samples);
    }
}

fn cell_b_write_buffer(device: &wgpu::Device, queue: &wgpu::Queue, sizes: &[u64], iters: usize) {
    println!("\n[i1247] === CELL B: queue.write_buffer (full write path) ===");
    let payload: Vec<u8> = vec![0xAB; (*sizes.iter().max().unwrap_or(&1024) as usize).max(1024)];
    let mut target_size = *sizes.iter().max().unwrap_or(&1024);
    if target_size < wgpu::COPY_BUFFER_ALIGNMENT {
        target_size = wgpu::COPY_BUFFER_ALIGNMENT;
    }
    let target = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("(i1247) write target"),
        size: target_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    for &size in sizes {
        let write_len = (size as usize).min(payload.len());
        let mut samples = Vec::new();
        for _ in 0..3 {
            queue.write_buffer(&target, 0, &payload[..write_len]);
        }
        for _ in 0..iters {
            let t = Instant::now();
            queue.write_buffer(&target, 0, &payload[..write_len]);
            samples.push(t.elapsed().as_nanos());
        }
        // Drain the staged copies so no cell inherits the previous's
        // pending-write backlog; the drain itself is CELL D's subject.
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(10)),
        });
        print_cell("B write_buffer", size, samples);
    }
}

fn cell_c_submit(device: &wgpu::Device, queue: &wgpu::Queue, iters: usize) {
    println!("\n[i1247] === CELL C: queue.submit (empty commit) ===");
    let mut samples = Vec::new();
    let make_encoder = || {
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("(i1247) empty"),
        })
    };
    for _ in 0..3 {
        queue.submit([make_encoder().finish()]);
    }
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(std::time::Duration::from_secs(10)),
    });
    for _ in 0..iters {
        let t = Instant::now();
        queue.submit([make_encoder().finish()]);
        samples.push(t.elapsed().as_nanos());
    }
    print_cell("C submit empty", 0, samples);
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(std::time::Duration::from_secs(10)),
    });
}

fn cell_d_poll_wait(device: &wgpu::Device, queue: &wgpu::Queue, iters: usize) {
    println!("\n[i1247] === CELL D: device.poll(Wait) after a trivial submission ===");
    let mut samples = Vec::new();
    for _ in 0..3 {
        let enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("(i1247) poll"),
        });
        let idx = queue.submit([enc.finish()]);
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: Some(idx),
            timeout: Some(std::time::Duration::from_secs(10)),
        });
    }
    for _ in 0..iters {
        let enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("(i1247) poll"),
        });
        let idx = queue.submit([enc.finish()]);
        let t = Instant::now();
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: Some(idx),
            timeout: Some(std::time::Duration::from_secs(10)),
        });
        samples.push(t.elapsed().as_nanos());
    }
    print_cell("D poll(Wait)", 0, samples);
}

fn cell_h_metal_direct(device: &wgpu::Device, queue: &wgpu::Queue, iters: usize) {
    println!(
        "\n[i1247] === CELL H: raw Metal (newBufferWithLength + commandBuffer + commit) vs the wgpu path ==="
    );
    // SAFETY: the guards keep the hal objects alive through the cell; the
    // raw pointers are used inside this function only. The as_hal GUARD is
    // bound OUTSIDE the inner reads and held for the whole cell — the Ref
    // borrows never outlive it (no retain dance, no use-after-free).
    //
    // The H cells read the VENDORED hal types directly — the zero-copy GEMM
    // lane's raw-Metal shape, module doc: the raw MTLCommandQueue + a commit
    // batch OUTSIDE the wgpu state machine (the vendor's own design, PLAN
    // 1223-era). Here we measure the SAME raw primitives in isolation.
    let dev_guard = unsafe { device.as_hal::<wgpu::hal::api::Metal>() };
    let q_guard = unsafe { queue.as_hal::<wgpu::hal::api::Metal>() };
    // ⚠ The pointer extraction below is the failing seam — the wgpu-core
    // device wrapper (`DispatchDevice`) may need the dispatch through-path.
    // Leaving the cells OUT until the extraction is proven: the segfault at
    // objc_msgSend (address 0x1f00000080 — a freed/wrong object) says the
    // raw_device() path through the DISPATCH layer is not the same object the
    // `metal` crate methods expect. The raw-Metal cell needs the hal-side
    // extraction that gemm_dense_f16_metal_zero_copy uses — which the vendored
    // code drives from the cubecl-wgpu backend, NOT from a wgpu-core device.
    // parked: cell H stays unrun; the wgpu-path cells already answered the
    // absolute-layer question at the API level.
    let _ = (&dev_guard, &q_guard, iters);
    // H1/H2 stay PARKED behind the extraction proof — see the ⚠ note above.
    println!(
        "[i1247]   H1/H2 PARKED: the wgpu-core dispatch-layer extraction segfaults (raw_device() through DispatchDevice is not the ObjC object the metal crate expects); the raw-Metal comparison needs the cubecl-wgpu backend-side extraction — a follow-up cell, not a blocker: the wgpu-path cells above already decompose the absolute layer at the API level."
    );
}

fn cell_g_fence_scan(device: &wgpu::Device, queue: &wgpu::Queue) {
    println!("\n[i1247] === CELL G: poll(Poll) fence-scan cost vs pending CB count ===");
    // The metal Fence::get_latest iterates EVERY pending command buffer and
    // calls `cmd_buf.status()` (an ObjC call) on each — wgpu-core's maintain
    // path calls get_fence_value + maintain per poll. With the real forward's
    // thousands of CBs pending between REGULATE drains, the linear scan is a
    // candidate for the unattributed remainder. The fence is device-global,
    // so the scan length = this device's pending CB count — build backlog,
    // then time a non-blocking poll(Poll).
    let mut pending: Vec<wgpu::SubmissionIndex> = Vec::new();
    for k in 0..3 {
        let target = 256usize * (k + 1); // 256 / 512 / 768 pending CBs
        while pending.len() < target {
            let enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("(i1247) G scan"),
            });
            pending.push(queue.submit([enc.finish()]));
        }
        let mut samples = Vec::new();
        for _ in 0..32 {
            let t = Instant::now();
            let _ = device.poll(wgpu::PollType::Poll);
            samples.push(t.elapsed().as_nanos());
        }
        print_cell(
            &format!("G poll(Poll) @{} pending", pending.len()),
            0,
            samples,
        );
    }
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(std::time::Duration::from_secs(30)),
    });
}

fn cell_i_cadence(device: &wgpu::Device, queue: &wgpu::Queue, passes: usize) {
    // Issue 1247 T2: reproduce the cubecl stream cadence at public-wgpu
    // level and time the same wall the real forward pays. The pattern, from
    // the vendored fork (stream.rs, Issue-761 counters): every Compute task
    // dispatch lands in ONE cumulative encoder; `flush_if_needed` submits
    // (commit) when tasks_count hits tasks_max=32; every Write task
    // flush()-then-writes (its own commit); the 64th pending write fires a
    // write-flush commit + poll(Wait{submission_index}); REGULATE
    // poll(Wait)s the previous submission index every 512 submitted tasks.
    // A pp pass is ~1500 commits: ~46 task-flushes of 32 + ~1500/64 ≈ 24
    // write-flush poll(Wait)s + ~3 REGULATE drains, interleaved. This cell
    // walks exactly that loop over `passes` x (32 dispatches + 64 writes),
    // timing per-pass wall, and reports the same wall decomposition the
    // forward's counters give: task-flush submit cost, write-flush
    // submit+poll cost, and the end drain.
    println!(
        "\n[i1247] === CELL I (cadence): {passes} passes x (32 dispatches -> commit + 64 writes w/ poll@64) ==="
    );
    let (pipeline, bg) = cadence_pipeline(device);
    let target = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("(i1247) I target"),
        size: 8192,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let payload = [0x5Au8; 4096];
    // Warm-up pass (pipelines + driver compile), untimed.
    cadence_pass(device, queue, &pipeline, &bg, &target, &payload);
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(std::time::Duration::from_secs(10)),
    });

    let mut pass_walls: Vec<u128> = Vec::new();
    let mut flush_submit_nanos: u128 = 0;
    let mut write_poll_nanos: u128 = 0;
    let mut regulate_nanos: u128 = 0;
    let mut submits: usize = 0;
    let mut pending_writes: usize = 0;
    let mut tasks_since_regulate: usize = 0;
    let mut last_submission: Option<wgpu::SubmissionIndex> = None;

    for _p in 0..passes {
        let t_pass = Instant::now();
        // The cumulative tasks encoder: 32 compute tasks accumulate here.
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("(i1247) I tasks"),
        });
        for _t in 0..32 {
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("(i1247) I pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(16, 1, 1);
            }
            tasks_since_regulate += 1;
        }
        // flush_if_needed: tasks_max reached -> commit.
        let t_flush = Instant::now();
        let idx = queue.submit([enc.finish()]);
        submits += 1;
        flush_submit_nanos += t_flush.elapsed().as_nanos();
        last_submission = Some(idx);
        // SubmissionLoad::regulate: every 512 submitted tasks, poll(Wait) the
        // previous submission index (here: one flush = 32 tasks, so every
        // 16th flush; the first 512-task window primes on Empty).
        if tasks_since_regulate >= 512 {
            let t_reg = Instant::now();
            let _ = device.poll(wgpu::PollType::Wait {
                submission_index: last_submission.clone(),
                timeout: None,
            });
            regulate_nanos += t_reg.elapsed().as_nanos();
            tasks_since_regulate = 0;
        }
        // The Write arm cadence: 64 writes, each flush-then-write at the
        // stream level; here the flush already happened (one per 32 tasks),
        // so each write is queue.write_buffer; the 64th pending write fires
        // the write-flush commit + poll(Wait{submission_index}).
        let mut wrote_this_window = 0usize;
        for _w in 0..64 {
            let t_write = Instant::now();
            queue.write_buffer(&target, 0, &payload);
            pending_writes += 1;
            wrote_this_window += 1;
            if pending_writes >= 64 {
                let wf_enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("(i1247) I write flush"),
                });
                let widx = queue.submit([wf_enc.finish()]);
                submits += 1;
                let _ = device.poll(wgpu::PollType::Wait {
                    submission_index: Some(widx),
                    timeout: None,
                });
                write_poll_nanos += t_write.elapsed().as_nanos();
                pending_writes = 0;
            }
            let _ = wrote_this_window;
        }
        pass_walls.push(t_pass.elapsed().as_nanos());
    }
    // The end drain (the pass-boundary sync the forward pays).
    let t_drain = Instant::now();
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(std::time::Duration::from_secs(120)),
    });
    let drain = t_drain.elapsed().as_nanos();

    let total_wall: u128 = pass_walls.iter().sum::<u128>() + drain;
    let per_pass_us = |n: u128| n as f64 / passes as f64 / 1e3;
    println!(
        "[i1247]   passes={passes}  total_wall={:.3} s",
        total_wall as f64 / 1e9
    );
    println!(
        "[i1247]   per-pass wall               = {:>10.1} us",
        per_pass_us(total_wall)
    );
    println!(
        "[i1247]   task-flush submit (32-task commit) = {:>10.1} us/pass  ({} submits)",
        per_pass_us(flush_submit_nanos),
        submits
    );
    println!(
        "[i1247]   write-flush submit+poll (64-write) = {:>10.1} us/pass",
        per_pass_us(write_poll_nanos)
    );
    println!(
        "[i1247]   REGULATE poll(Wait) drain         = {:>10.1} us/pass",
        per_pass_us(regulate_nanos)
    );
    println!(
        "[i1247]   end drain (pass-boundary sync)     = {:>10.1} us",
        drain as f64 / 1e3
    );
    println!(
        "[i1247]   per-pass wall ex-regulate-ex-drain = {:>10.1} us  (the pure cadence cost)",
        per_pass_us(total_wall - regulate_nanos - drain)
    );
}

fn cadence_pipeline(device: &wgpu::Device) -> (wgpu::ComputePipeline, wgpu::BindGroup) {
    let kernel = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("(i1247) I kernel"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
            "@group(0) @binding(0) var<storage, read_write> buf: array<f32, 2048>;\n@compute @workgroup_size(64)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n    let i = gid.x;\n    if i < 2048u {\n        var x = buf[i];\n        for (var k = 0u; k < 256u; k = k + 1u) {\n            x = x * 1.0000001 + 0.0000001;\n        }\n        buf[i] = x;\n    }\n}",
        )),
    });
    let busy_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("(i1247) I busy buf"),
        size: 2048 * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("(i1247) I bg layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("(i1247) I pl layout"),
        bind_group_layouts: &[Some(&bg_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("(i1247) I pipeline"),
        layout: Some(&pipeline_layout),
        module: &kernel,
        entry_point: None,
        compilation_options: Default::default(),
        cache: None,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("(i1247) I bg"),
        layout: &bg_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: busy_buf.as_entire_binding(),
        }],
    });
    (pipeline, bg)
}

fn cadence_pass(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    bg: &wgpu::BindGroup,
    target: &wgpu::Buffer,
    payload: &[u8; 4096],
) {
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("(i1247) I warmup"),
    });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("(i1247) I warmup pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.dispatch_workgroups(32, 1, 1);
    }
    queue.submit([enc.finish()]);
    queue.write_buffer(target, 0, payload);
}

fn cell_f_backlog(device: &wgpu::Device, queue: &wgpu::Queue, depth: usize) {
    println!(
        "\n[i1247] === CELL F: write latency vs in-flight command-buffer backlog (depth {depth}) ==="
    );
    let work_size = 1024u32;
    let kernel = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("(i1247) F busy"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
            "@group(0) @binding(0) var<storage, read_write> buf: array<f32, 1024>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i < 1024u {
        var x = buf[i];
        for (var k = 0u; k < 512u; k = k + 1u) {
            x = x * 1.0000001 + 0.0000001;
        }
        buf[i] = x;
    }
}",
        )),
    });
    let busy_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("(i1247) F busy buf"),
        size: (work_size as u64) * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("(i1247) F bg layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("(i1247) F pl layout"),
        bind_group_layouts: &[Some(&bg_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("(i1247) F pipeline"),
        layout: Some(&pipeline_layout),
        module: &kernel,
        entry_point: None,
        compilation_options: Default::default(),
        cache: None,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("(i1247) F bg"),
        layout: &bg_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: busy_buf.as_entire_binding(),
        }],
    });
    let target = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("(i1247) F target"),
        size: 8192,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let payload = [0x5Au8; 4096];
    let mut samples: Vec<u128> = Vec::new();
    let mut backlog: Vec<usize> = Vec::new();
    let mut submitted: usize = 0;
    // Warm writes first (pipelines/staging warm).
    for _ in 0..8 {
        queue.write_buffer(&target, 0, &payload);
    }
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(std::time::Duration::from_secs(10)),
    });
    // One cumulative probe: submit empty CBs WITHOUT polling, and every few
    // submissions time a 4 KB write — the write latency reads as a function
    // of the growing in-flight depth. The backlog depth ladder covers the
    // saturation band the hypothesis predicts (write-flush cadence ~64 +
    // task-flush ~512 → the real forward's steady-state in-flight depth).
    let stride = (depth / 16).max(8);
    while submitted < depth {
        // Each backlog CB: one compute pass, 16 workgroups × 64 threads of
        // real (small) GPU work — the driver retires actual dispatches.
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("(i1247) F backlog"),
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("(i1247) F pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(work_size / 64, 1, 1);
        }
        queue.submit([enc.finish()]);
        submitted += 1;
        if submitted.is_multiple_of(stride) {
            let t = Instant::now();
            queue.write_buffer(&target, 0, &payload);
            samples.push(t.elapsed().as_nanos());
            backlog.push(submitted);
        }
    }
    println!("[i1247]   in_flight_depth -> write_4kb_latency_ns (16-point curve):",);
    for (d, s) in backlog.iter().zip(samples.iter()) {
        println!(
            "[i1247]   depth={:>6}  write={:>10} ns{}",
            d,
            s,
            if *s > 1_000_000 { "  <-- ms-class" } else { "" }
        );
    }
    // The drain: how long does ONE poll(Wait) take against the full backlog?
    let t = Instant::now();
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(std::time::Duration::from_secs(120)),
    });
    println!(
        "[i1247]   drain poll(Wait) over the full backlog: {:.3} s",
        t.elapsed().as_secs_f64()
    );
}

fn cell_e_pressure(device: &wgpu::Device, queue: &wgpu::Queue, cfg: &Cfg) {
    if cfg.skip_pressure {
        println!("\n[i1247] === CELL E (pressure): SKIPPED by flag ===");
        return;
    }
    println!(
        "\n[i1247] === CELL E: pressure — {} GB resident, then A/B repeat at the two extreme sizes ===",
        cfg.pressure_gb
    );
    let blob_size = 256 * 1024 * 1024u64; // 256 MB per blob (the default max-buffer-size limit)
    let blobs: Vec<wgpu::Buffer> = (0..(cfg.pressure_gb * 2))
        .map(|_i| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("(i1247) pressure blob"),
                size: blob_size,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        })
        .collect();
    println!(
        "[i1247] pressure set resident: {} blobs × 512 MB",
        blobs.len()
    );
    // Touch a target inside the pressure set so the allocator's live set is
    // real, then re-run the A and B cells at the ladder's extremes.
    let sizes = [
        cfg.sizes.first().copied().unwrap_or(4096),
        cfg.sizes.last().copied().unwrap_or(64 * 1024 * 1024),
    ];
    cell_a_create_buffer(device, &sizes, cfg.iters);
    cell_b_write_buffer(device, queue, &sizes, cfg.iters);
    // Blobs stay alive until here by construction.
    drop(blobs);
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(std::time::Duration::from_secs(10)),
    });
}

fn provenance() {
    println!(
        "[i1247] provenance: host={} os={} cpus={} mem_gb={}",
        std::env::var("HOSTNAME").unwrap_or_else(|_| whoami_host()),
        std::env::consts::OS,
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0),
        system_mem_gb(),
    );
    println!(
        "[i1247] load: {}",
        std::fs::read_to_string("/proc/loadavg").unwrap_or_else(|_| loadavg_mac())
    );
}

fn whoami_host() -> String {
    std::process::Command::new("hostname")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".into())
}

fn loadavg_mac() -> String {
    std::process::Command::new("uptime")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".into())
}

fn system_mem_gb() -> u64 {
    std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()
        .and_then(|o| {
            String::from_utf8_lossy(&o.stdout)
                .trim()
                .parse::<u64>()
                .ok()
        })
        .map(|b| b / (1024 * 1024 * 1024))
        .unwrap_or(0)
}

fn main() {
    let cfg = parse_args();
    provenance();
    let (device, queue) = device();

    println!(
        "[i1247] sizes: {:?}  iters: {}  pressure: {} GB{}",
        cfg.sizes,
        cfg.iters,
        cfg.pressure_gb,
        if cfg.skip_pressure { " (skipped)" } else { "" }
    );

    cell_a_create_buffer(&device, &cfg.sizes, cfg.iters);
    cell_b_write_buffer(&device, &queue, &cfg.sizes, cfg.iters);
    cell_c_submit(&device, &queue, cfg.iters);
    cell_d_poll_wait(&device, &queue, cfg.iters);
    cell_f_backlog(&device, &queue, 1024);
    cell_f_backlog(&device, &queue, 4096);
    cell_i_cadence(&device, &queue, 48);
    cell_g_fence_scan(&device, &queue);
    cell_h_metal_direct(&device, &queue, cfg.iters);
    cell_e_pressure(&device, &queue, &cfg);
    println!("\n[i1247] done.");
}
