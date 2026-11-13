//! Issue 1211 — the FOSCAR fold A/B cell (T3's latency half).
//!
//! Incumbent vs folded Q8KV decode-attention kernel latency on the Metal
//! posture, interleaved per iteration (pair (inc, fold), median across
//! pairs — the sequential-A/B scheduler-tax law), same buffers, same
//! shapes, one long-context cell. Parity is NOT measured here — it lives
//! in `attention_q8kv_folded`'s `#[test]`s (the falsification arm).
//!
//! The pre-registered gate (Issue 1211): **≥5% kernel latency gain** at the
//! measured decode shapes AND parity holds → promote into the q8kv dispatch
//! path; otherwise demote = keep the incumbent and record the number in
//! Research 026 §3. This cell prints the per-shape medians + the geomean
//! ratio; the verdict is recorded in the bench doc, never auto-promoted.
//!
//! Box state: run `../riir-reflex/scripts/bench_preflight.sh` before the
//! cell and quote its PROVENANCE line beside every number (the league law).
//!
//! Run (release, quiet box):
//!
//! ```sh
//! cargo run --release -p riir-infer-gpu --features q8kv_fold_cell \
//!   --bin i1211_fold_ab_cell -- --iters 40 --warmup 5
//! ```

#[cfg(feature = "cubecl_runtime")]
mod imp {
    use std::time::Instant;

    use cubecl::prelude::*;
    use cubecl::server::Handle;

    use riir_infer_gpu::attention_cubecl::AttentionParams;
    use riir_infer_gpu::attention_q8kv_cubecl::{AttentionQ8KVCubeCL, Q8KVBuffers};
    use riir_infer_gpu::attention_q8kv_folded::AttentionQ8KVFolded;
    use riir_infer_gpu::cubecl_runtime::{ActiveRuntime, CubeCLContext};

    const N_HEAD: usize = 8;
    const N_KV_HEAD: usize = 4;
    const HEAD_DIM: usize = 256;
    const SOFTCAP: f32 = 50.0;
    const SCALE: f32 = 0.0625; // 1/√256

    /// Deterministic sin/cos fixture (the parity tests' data — same family so
    /// the latency cell and the parity arm measure the same lane shape).
    fn make_test_data(n_positions: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let q_len = N_HEAD * HEAD_DIM;
        let kv_len = n_positions * N_KV_HEAD * HEAD_DIM;

        let query: Vec<f32> = (0..q_len)
            .map(|i| ((i as f32) * 0.01).sin() * 0.5)
            .collect();
        let keys: Vec<f32> = (0..kv_len)
            .map(|i| ((i as f32) * 0.02).cos() * 0.3)
            .collect();
        let values: Vec<f32> = (0..kv_len)
            .map(|i| ((i as f32) * 0.03).sin() * 0.2)
            .collect();

        (query, keys, values)
    }

    /// Uploaded buffer set for one shape — quantize + upload ONCE, then the
    /// timed loop only launches and reads.
    struct ShapeBuffers {
        params: AttentionParams,
        query_handle: Handle,
        kv_qs_handle: Handle,
        kv_scales_handle: Handle,
        out_handle_inc: Handle,
        out_handle_fold: Handle,
    }

    fn build_shape(client: &ComputeClient<ActiveRuntime>, n_positions: usize) -> ShapeBuffers {
        let (query, keys, values) = make_test_data(n_positions);
        let q8_bufs = Q8KVBuffers::quantize_kv(&keys, &values, N_KV_HEAD, HEAD_DIM);
        let q_len = N_HEAD * HEAD_DIM;

        let query_handle = client.create_from_slice(f32::as_bytes(&query));
        let kv_qs_handle =
            client.create_from_slice(bytemuck::cast_slice::<u32, u8>(&q8_bufs.kv_qs));
        let kv_scales_handle = client.create_from_slice(f32::as_bytes(&q8_bufs.kv_scales));

        ShapeBuffers {
            params: AttentionParams {
                n_head: N_HEAD,
                n_kv_head: N_KV_HEAD,
                head_dim: HEAD_DIM,
                n_positions,
                softcap: SOFTCAP,
                scale: SCALE,
            },
            query_handle,
            kv_qs_handle,
            kv_scales_handle,
            out_handle_inc: client.empty(q_len * core::mem::size_of::<f32>()),
            out_handle_fold: client.empty(q_len * core::mem::size_of::<f32>()),
        }
    }

    fn run_incumbent(client: &ComputeClient<ActiveRuntime>, s: &ShapeBuffers) {
        AttentionQ8KVCubeCL::launch::<ActiveRuntime>(
            client,
            s.query_handle.clone(),
            s.kv_qs_handle.clone(),
            s.kv_scales_handle.clone(),
            s.out_handle_inc.clone(),
            &s.params,
        );
    }

    fn run_folded(client: &ComputeClient<ActiveRuntime>, s: &ShapeBuffers) {
        AttentionQ8KVFolded::launch::<ActiveRuntime>(
            client,
            s.query_handle.clone(),
            s.kv_qs_handle.clone(),
            s.kv_scales_handle.clone(),
            s.out_handle_fold.clone(),
            &s.params,
        );
    }

    /// Median of a sorted slice (even n: lower-middle — pinned, first-index
    /// tie rule per the determinism law).
    fn median(sorted: &[f64]) -> f64 {
        sorted[sorted.len() / 2]
    }

    /// p90 with the tail-support line (the n−1-max rule: a percentile index
    /// landing on n−1 is the MAX — print the support beside it).
    fn p90_with_support(sorted: &[f64]) -> (f64, usize) {
        let n = sorted.len();
        let idx = (((n as f64) * 0.9).floor() as usize).min(n - 1);
        (sorted[idx], n - idx)
    }

    fn run() {
        let mut shapes: Vec<usize> = vec![512, 4096, 32768];
        let mut iters: usize = 40;
        let mut warmup: usize = 5;

        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--positions" => {
                    let v = args.next().expect("--positions needs a value");
                    shapes = v
                        .split(',')
                        .map(|s| s.trim().parse().expect("positions must be usize"))
                        .collect();
                }
                "--iters" => {
                    iters = args
                        .next()
                        .expect("--iters needs a value")
                        .parse()
                        .expect("iters must be usize");
                }
                "--warmup" => {
                    warmup = args
                        .next()
                        .expect("--warmup needs a value")
                        .parse()
                        .expect("warmup must be usize");
                }
                other => {
                    eprintln!("unknown arg: {other}");
                    std::process::exit(2);
                }
            }
        }

        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let backend = std::any::type_name::<ActiveRuntime>();
        println!("i1211 fold A/B cell — backend: {backend}");
        println!(
            "box: run ../riir-reflex/scripts/bench_preflight.sh and quote its PROVENANCE line beside these numbers"
        );
        println!("shapes={shapes:?} iters(pairs)={iters} warmup(pairs)={warmup}");

        let mut ratios: Vec<f64> = Vec::new();

        for &n_positions in &shapes {
            let s = build_shape(&client, n_positions);

            // Warmup pairs — both kernels compile (Metal shader pipelines)
            // and caches settle BEFORE anything is timed.
            for _ in 0..warmup {
                run_incumbent(&client, &s);
                let _ = client.read_one(s.out_handle_inc.clone());
                run_folded(&client, &s);
                let _ = client.read_one(s.out_handle_fold.clone());
            }

            // Timed interleaved pairs: per pair, one incumbent launch+sync,
            // one folded launch+sync. Interleaving is the scheduler-tax
            // control; launch+readback is the honest per-kernel serving cost
            // and cancels in the ratio (both arms pay it).
            let mut inc_times: Vec<f64> = Vec::with_capacity(iters);
            let mut fold_times: Vec<f64> = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t0 = Instant::now();
                run_incumbent(&client, &s);
                let _ = client.read_one(s.out_handle_inc.clone());
                inc_times.push(t0.elapsed().as_secs_f64() * 1e3);

                let t1 = Instant::now();
                run_folded(&client, &s);
                let _ = client.read_one(s.out_handle_fold.clone());
                fold_times.push(t1.elapsed().as_secs_f64() * 1e3);
            }

            inc_times.sort_by(|a, b| a.partial_cmp(b).expect("no NaN timings"));
            fold_times.sort_by(|a, b| a.partial_cmp(b).expect("no NaN timings"));

            let inc_med = median(&inc_times);
            let fold_med = median(&fold_times);
            let (inc_p90, inc_sup) = p90_with_support(&inc_times);
            let (fold_p90, fold_sup) = p90_with_support(&fold_times);
            let ratio = fold_med / inc_med; // < 1.0 = folded faster

            ratios.push(ratio);

            println!(
                "n_pos={n_positions:>6}: incumbent med {inc_med:.4} ms (p90 {inc_p90:.4}, support {inc_sup}/{iters}) | \
                 folded med {fold_med:.4} ms (p90 {fold_p90:.4}, support {fold_sup}/{iters}) | \
                 ratio {ratio:.4}x"
            );
        }

        // Geomean ratio across shapes (log-mean), the gate's aggregate read.
        let geo = (ratios.iter().map(|r| r.ln()).sum::<f64>() / ratios.len() as f64).exp();
        println!("geomean ratio: {geo:.4}x (folded/incumbent; <= 0.95 ≈ ≥5% gain)");
        if geo <= 0.95 {
            println!(
                "VERDICT (cell-only): latency bar MET — parity still gates promotion (see the parity tests)"
            );
        } else {
            println!(
                "VERDICT (cell-only): latency bar MISSED — pre-registered demote posture (keep the incumbent)"
            );
        }
    }

    pub fn main() {
        run();
    }
}

#[cfg(feature = "cubecl_runtime")]
fn main() {
    imp::main();
}

#[cfg(not(feature = "cubecl_runtime"))]
fn main() {
    eprintln!("compile with --features q8kv_fold_cell");
    std::process::exit(2);
}
