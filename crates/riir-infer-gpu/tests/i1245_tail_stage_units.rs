//! Issue 1245 bisect leg 3 — the full chunk-tail stage in miniature:
//! batched-norm a [p × e] fixture, then the EXACT tail sequence
//! (`RmsNormCubeCL` on an offset row view → 32768-row GEMV chunks into an
//! offset `logits_b`) at the 0.8B fixture's shapes, with a CPU reference
//! for every row. Determinism cell: N identical runs must agree bitwise.
#![cfg(all(test, feature = "qwen38_wgpu"))]

use cubecl::client::ComputeClient;
use riir_infer_gpu::cubecl_runtime::{ActiveRuntime, CubeCLContext};

fn stats(name: &str, v: &[f32]) -> (usize, usize) {
    let nonfinite = v.iter().filter(|x| !x.is_finite()).count();
    let zeros = v.iter().filter(|&&x| x == 0.0).count();
    println!(
        "[i1245-units] {name}: n={} nonfinite={} zeros={}",
        v.len(),
        nonfinite,
        zeros
    );
    (nonfinite, zeros)
}

fn gemv_ref(w: &[f32], x: &[f32], m: usize, e: usize) -> Vec<f32> {
    (0..m)
        .map(|r| {
            let row = &w[r * e..(r + 1) * e];
            row.iter().zip(x.iter()).map(|(&w, &x)| w * x).sum::<f32>()
        })
        .collect()
}

const P: usize = 512;
const E: usize = 1024;
const VOCAB: usize = 248320;
const LM_CHUNK: usize = 32768;
const EPS: f32 = 1e-5;

/// Isolation: does a fresh-handle m=1 plane GEMV pass when the pool is
/// virgin, and fail after big allocations reshape the pool? (the ladder's
/// failure signature, split from the tail stage)
#[test]
fn gemv_fresh_vs_warm_pool() {
    let ctx = CubeCLContext::new().expect("cubecl context");
    let cl: ComputeClient<ActiveRuntime> = ctx.client();
    let e = 1024usize;

    let w: Vec<f32> = (0..e).map(|i| ((i as f32 * 0.618).sin()) * 0.5).collect();
    let tiny: Vec<f32> = (0..e).map(|i| ((i as f32 * 0.37).sin()) * 2.0).collect();
    let want0: f32 = w.iter().zip(tiny.iter()).map(|(&a, &b)| a * b).sum();

    let run1 = |cl: &ComputeClient<ActiveRuntime>| -> f32 {
        let w_h = cl.create_from_slice(bytemuck::cast_slice(&w));
        let in_h = cl.create_from_slice(bytemuck::cast_slice(&tiny));
        let out_h = cl.empty(4);
        unsafe {
            riir_infer_gpu::gemv_cubecl::GemvCubeCL::launch::<ActiveRuntime>(
                cl,
                w_h,
                in_h,
                out_h.clone(),
                1,
                e,
            );
        }
        cl.read_one(out_h)
            .map(|b| bytemuck::cast_slice(&b)[0])
            .expect("readback")
    };
    let got_virgin = run1(&cl);
    println!(
        "[i1245-units] VIRGIN pool m=1: out[0]={got_virgin:.4} want {want0:.4} match={}",
        (got_virgin - want0).abs() < 1e-3
    );

    // Warm the pool with the model-shaped allocations (128 MiB w + 2 MiB
    // scratch + 1 MB logits), drop them, then re-run the same fresh m=1.
    {
        let big: Vec<f32> = vec![0.5f32; 32768 * e];
        let _w2 = cl.create_from_slice(bytemuck::cast_slice(&big));
        let _t = cl.empty(512 * e * 4);
        let _l = cl.empty(248320 * 4);
    }
    let got_warm = run1(&cl);
    println!(
        "[i1245-units] WARM pool m=1: out[0]={got_warm:.4} want {want0:.4} match={}",
        (got_warm - want0).abs() < 1e-3
    );
    assert!(
        (got_virgin - want0).abs() < 1e-3,
        "fresh m=1 GEMV wrong on a VIRGIN pool — the kernel itself is broken"
    );

    // Arm matrix, all fresh no-offset handles:
    //  A m=1, input 4 KiB (passed above)
    //  B m=1, input 2 MiB
    //  C m=32768, input 4 KiB (row 0 repeated), out 993280 B
    //  D m=32768, input 2 MiB, out 993280 B  (the tail shape)
    for &in_rows in &[1usize, 8, 32, 64, 128, 256, 512] {
        let wbig: Vec<f32> = (0..1024 * e)
            .map(|i| ((i as f32 * 0.618).sin()) * 0.5)
            .collect();
        let xin: Vec<f32> = (0..in_rows * e)
            .map(|i| ((i as f32 * 0.37).sin()) * 2.0)
            .collect();
        let want0: f32 = wbig[0..e]
            .iter()
            .zip(xin[0..e].iter())
            .map(|(&a, &b)| a * b)
            .sum();
        let w_h = cl.create_from_slice(bytemuck::cast_slice(&wbig));
        let in_h = cl.create_from_slice(bytemuck::cast_slice(&xin));
        let out_h = cl.empty(4);
        unsafe {
            riir_infer_gpu::gemv_cubecl::GemvCubeCL::launch::<ActiveRuntime>(
                &cl,
                w_h,
                in_h,
                out_h.clone(),
                1,
                e,
            );
        }
        let got: Vec<f32> = cl
            .read_one(out_h)
            .map(|b| bytemuck::cast_slice(&b)[..1].to_vec())
            .expect("readback");
        println!(
            "[i1245-units] LADDER in_rows={in_rows} ({} KiB): out[0]={:.4} want {want0:.4} match={}",
            in_rows * 4,
            got[0],
            (got[0] - want0).abs() < 1e-2
        );
    }
}

#[test]
fn tail_stage_full_miniature() {
    let ctx = CubeCLContext::new().expect("cubecl context");
    let cl: ComputeClient<ActiveRuntime> = ctx.client();

    // Fixture rows — plausible model-scale activations.
    let xb: Vec<f32> = (0..P * E)
        .map(|i| ((i as f32 * 0.1713).sin()) * (1.0 + (i % 97) as f32 * 0.01))
        .collect();
    // The tail-norm γ: ones + 1 (the loader convention: γ = 1 + w).
    let gamma: Vec<f32> = vec![1.0; E];

    // CPU reference of the tail row's norm (γ = 1, so the γ multiply is a
    // no-op in the reference; the kernel applies `x · inv_rms · γ`).
    let last_row = &xb[(P - 1) * E..P * E];
    let mean_sq = last_row.iter().map(|v| v * v).sum::<f32>() / E as f32;
    let inv_rms = 1.0 / (mean_sq + EPS).sqrt();
    let normed_ref: Vec<f32> = last_row.iter().map(|v| v * inv_rms).collect();

    // One lm chunk's weights (the FIRST chunk — where the bisect saw
    // garbage indices 1619..1803).
    let m = LM_CHUNK;
    let w: Vec<f32> = (0..m * E)
        .map(|i| ((i as f32 * 0.618).sin()) * 0.5)
        .collect();

    let mut runs: Vec<Vec<f32>> = Vec::new();
    for run in 0..2 {
        let xb_h = cl.create_from_slice(bytemuck::cast_slice(&xb));
        let gamma_h = cl.create_from_slice(bytemuck::cast_slice(&gamma));
        let w_h = cl.create_from_slice(bytemuck::cast_slice(&w));
        let tmp_h = cl.empty(P * E * 4);
        let logits_h = cl.empty(VOCAB * 4);

        if run == 0 {
            // POOL-OFFSET PROBE: what slice did the pool hand tmp_h and
            // logits_h? (the wgpu resource walk; the zero-copy GEMM's
            // extract_raw_buffer path)
            if let Ok(managed) = cl.get_resource(tmp_h.clone()) {
                let r = managed.resource();
                println!(
                    "[i1245-units] run{run}: tmp_h pool offset={} size={}",
                    r.offset, r.size
                );
            }
            if let Ok(managed) = cl.get_resource(logits_h.clone()) {
                let r = managed.resource();
                println!(
                    "[i1245-units] run{run}: logits_h pool offset={} size={}",
                    r.offset, r.size
                );
            }
        }
        unsafe {
            // The EXACT tail sequence: batched norm (the loop's norm launch,
            // row p-1 fresh in tmp_b) → offset row view → single-row norm →
            // per-chunk GEMVs into offset logits (chunk 0 only in the cell).
            riir_infer_gpu::norms_cubecl::RmsNormBatchedCubeCL::launch::<ActiveRuntime>(
                &cl,
                xb_h.clone(),
                gamma_h.clone(),
                tmp_h.clone(),
                P,
                E,
                EPS,
            );
            let last = xb_h.clone().offset_start(((P - 1) * E * 4) as u64);
            riir_infer_gpu::norms_cubecl::RmsNormCubeCL::launch::<ActiveRuntime>(
                &cl,
                last,
                gamma_h.clone(),
                tmp_h.clone(),
                E,
                EPS,
            );
            let out = logits_h.clone().offset_start(0u64);
            riir_infer_gpu::gemv_cubecl::GemvCubeCL::launch::<ActiveRuntime>(
                &cl,
                w_h.clone(),
                tmp_h.clone(),
                out,
                m,
                E,
            );
        }

        let normed: Vec<f32> = cl
            .read_one(tmp_h.clone())
            .map(|b| bytemuck::cast_slice(&b)[..E].to_vec())
            .expect("norm readback");
        let mut norm_mismatch = 0usize;
        for (g, r) in normed.iter().zip(normed_ref.iter()) {
            if (g - r).abs() > 1e-3 * (1.0 + r.abs()) {
                norm_mismatch += 1;
            }
        }
        stats(&format!("run{run} tmp normed"), &normed);
        assert_eq!(norm_mismatch, 0, "tail norm wrong at run {run}");

        // CPU reference for the logits with the GPU's ACTUAL normed row
        // (not my host ref — the single-row tail norm may read a different
        // row if the stage itself is the defect; comparing against the
        // actual input isolates the GEMV alone).
        // Reconstruct the batched-norm rows host-side (the alternate
        // reference + the which-row scan's input).
        let mut tmp_host = vec![0.0f32; P * E];
        for r in 0..P {
            let row = &xb[r * E..(r + 1) * E];
            let ms = row.iter().map(|v| v * v).sum::<f32>() / E as f32;
            let inv = 1.0 / (ms + EPS).sqrt();
            for (j, v) in row.iter().enumerate() {
                tmp_host[r * E + j] = v * inv;
            }
        }
        let logits_ref_actual = gemv_ref(&w, &normed, m, E);
        let logits_ref_fulltmp = gemv_ref(&w, &tmp_host, m, E);
        let logits: Vec<f32> = cl
            .read_one(logits_h)
            .map(|b| bytemuck::cast_slice(&b)[..m].to_vec())
            .expect("logits readback");
        stats(&format!("run{run} logits chunk0"), &logits);

        // Same kernel, fresh output handle, SAME input bytes — the control:
        // if this matches while the offset-output pass above does not, the
        // offset-OUTPUT binding is the defect. If both mismatch identically,
        // it's the offset-INPUT (tmp) binding or the kernel itself.
        let logits_offset_input: Vec<f32> = {
            let out_h = cl.empty(m * 4);
            unsafe {
                riir_infer_gpu::gemv_cubecl::GemvCubeCL::launch::<ActiveRuntime>(
                    &cl,
                    w_h.clone(),
                    tmp_h.clone().offset_start((E * 4) as u64),
                    out_h.clone(),
                    m,
                    E,
                );
            }
            cl.read_one(out_h)
                .map(|b| bytemuck::cast_slice(&b).to_vec())
                .expect("offset-input readback")
        };
        let ref_row1 = gemv_ref(&w, &tmp_host[E..2 * E], 1, E)[0];
        println!(
            "[i1245-units] run{run}: offset-input out[0]={:.4} vs row-1 dot {:.4} vs row-0 dot {:.4}",
            logits_offset_input[0], ref_row1, logits_ref_actual[0]
        );
        // Which host row does the offset-input out[0] match? Scan.
        {
            let g0 = logits_offset_input[0];
            let mut best = (f32::INFINITY, i64::MIN);
            for start in 0..=(P * E - E) {
                let win = &tmp_host[start..start + E];
                let mut dot = 0.0f32;
                for (k, &xv) in win.iter().enumerate() {
                    dot += w[k] * xv;
                }
                let d = (dot - g0).abs();
                if d < best.0 {
                    best = (d, start as i64);
                }
            }
            println!(
                "[i1245-units] run{run}: offset-input scan: best start={} (row {} col {}) delta {:.4}",
                best.1,
                best.1 / E as i64,
                best.1 % E as i64,
                best.0
            );
            // SLAB-BASE HYPOTHESIS: the input binding's SLICE OFFSET is
            // ignored, so the kernel reads the SLAB BASE — which holds the
            // first-allocated buffer (xb). Scan the RAW xb fixture rows.
            {
                let g0 = logits_offset_input[0];
                let mut best2 = (f32::INFINITY, i64::MIN);
                for start in 0..=(P * E - E) {
                    let win = &xb[start..start + E];
                    let mut dot = 0.0f32;
                    for (k, &xv) in win.iter().enumerate() {
                        dot += w[k] * xv;
                    }
                    let d = (dot - g0).abs();
                    if d < best2.0 {
                        best2 = (d, start as i64);
                    }
                }
                println!(
                    "[i1245-units] run{run}: offset-input vs RAW-XB scan: best start={} (row {} col {}) delta {:.4}",
                    best2.1,
                    best2.1 / E as i64,
                    best2.1 % E as i64,
                    best2.0
                );
            }
        }
        // THE CONTROL PAIR: fresh tiny input (4 KiB) at the pool's small
        // bucket — the leg-1 shape that PASSED — and the RAW xb buffer as
        // input (2 MiB, big bucket).
        let tiny: Vec<f32> = (0..E).map(|i| ((i as f32 * 0.37).sin()) * 2.0).collect();
        let tiny_ref0 = {
            let mut dot = 0.0f32;
            for k in 0..E {
                dot += w[k] * tiny[k];
            }
            dot
        };
        let tiny_out: Vec<f32> = {
            let in_h = cl.create_from_slice(bytemuck::cast_slice(&tiny));
            let out_h = cl.empty(4);
            unsafe {
                riir_infer_gpu::gemv_cubecl::GemvCubeCL::launch::<ActiveRuntime>(
                    &cl,
                    w_h.clone(),
                    in_h,
                    out_h.clone(),
                    1,
                    E,
                );
            }
            cl.read_one(out_h)
                .map(|b| bytemuck::cast_slice(&b).to_vec())
                .expect("tiny readback")
        };
        println!(
            "[i1245-units] run{run}: TINY input: out[0]={:.4} want {tiny_ref0:.4} match={}",
            tiny_out[0],
            (tiny_out[0] - tiny_ref0).abs() < 1e-3
        );
        // SIZE LADDER: 4 KiB (passed), 256 KiB, 1 MiB, 2 MiB — find the
        // failing bucket boundary.
        for &sz in &[262144usize, 1048576, 2097152] {
            let mid: Vec<f32> = (0..sz).map(|i| ((i as f32 * 0.037).sin()) * 1.5).collect();
            let want0 = {
                let mut dot = 0.0f32;
                for k in 0..E {
                    dot += w[k] * mid[k];
                }
                dot
            };
            let in_h = cl.create_from_slice(bytemuck::cast_slice(&mid));
            let (poff, psize) = cl
                .get_resource(in_h.clone())
                .map(|m| {
                    let r = m.resource();
                    (r.offset, r.size)
                })
                .unwrap_or((0, 0));
            let out_h = cl.empty(4);
            unsafe {
                riir_infer_gpu::gemv_cubecl::GemvCubeCL::launch::<ActiveRuntime>(
                    &cl,
                    w_h.clone(),
                    in_h,
                    out_h.clone(),
                    1,
                    E,
                );
            }
            let got: Vec<f32> = cl
                .read_one(out_h)
                .map(|b| bytemuck::cast_slice(&b).to_vec())
                .expect("mid readback");
            println!(
                "[i1245-units] run{run}: LADDER sz={sz} pool_off={poff} pool_sz={psize}: out[0]={:.4} want {want0:.4} match={}",
                got[0],
                (got[0] - want0).abs() < 1e-2
            );
        }
        let logits_direct: Vec<f32> = {
            let out_h = cl.empty(m * 4);
            unsafe {
                riir_infer_gpu::gemv_cubecl::GemvCubeCL::launch::<ActiveRuntime>(
                    &cl,
                    w_h.clone(),
                    tmp_h.clone(),
                    out_h.clone(),
                    m,
                    E,
                );
            }
            cl.read_one(out_h)
                .map(|b| bytemuck::cast_slice(&b).to_vec())
                .expect("direct readback")
        };
        let mut mismatch_direct = 0usize;
        for (&g, &r) in logits_direct.iter().zip(logits_ref_actual.iter()) {
            if (g - r).abs() > 1e-2 * (1.0 + r.abs()) {
                mismatch_direct += 1;
            }
        }
        println!("[i1245-units] run{run}: direct-output mismatches={mismatch_direct}/{m}");

        let mut mismatch = 0usize;
        for (i, (&g, &r)) in logits.iter().zip(logits_ref_actual.iter()).enumerate() {
            if (g - r).abs() > 1e-2 * (1.0 + r.abs()) {
                mismatch += 1;
                if mismatch <= 5 {
                    println!("[i1245-units]   row {i}: got {g} want {r}");
                }
            }
        }
        println!("[i1245-units] run{run}: mismatches={mismatch}/{m}");
        // Which of the 512 normed rows does the GPU output match? A
        // row-correlation scan: for the first 64 outputs, find the host
        // normed row that best explains it.
        {
            let mut hits = std::collections::HashMap::new();
            #[allow(clippy::needless_range_loop)]
            for out_i in 0..8usize.min(m) {
                let g = logits[out_i];
                // What INPUT VECTOR does row `out_i` of the weights see, if
                // each output element k of row `out_i` is a lane-strided dot?
                // Invert: assume the kernel read input[k] with STRIDE:
                //   partial[l] = sum_j w[out_i*E + l + j*32] * in[l + j*32]
                // Solve for in[] lane by lane using host-side w + the true
                // normed row structure? Instead: directly test the STRIDE
                // hypothesis — compute the dot with input indexed as
                // tmp[(j/32)*E + (j%32)*1 + lane] ... too speculative.
                // CONCRETE: which single tmp row position p* gives the
                // closest dot for in = &tmp_host[p*E..(p+1)*E].
                let mut best = (f32::INFINITY, usize::MAX);
                for r in 0..P {
                    let row = &tmp_host[r * E..(r + 1) * E];
                    let mut dot = 0.0f32;
                    let wi = out_i * E;
                    for (k, &xv) in row.iter().enumerate() {
                        dot += w[wi + k] * xv;
                    }
                    let d = (dot - g).abs();
                    if d < best.0 {
                        best = (d, r);
                    }
                }
                *hits.entry(best.1).or_insert(0usize) += 1;
            }
            // ALSO: does output row out_i match ANY row's dot when the GEMV
            // read input as the BATCHED layout of a DIFFERENT buffer? Print
            // the raw first-4 outputs vs all candidate dots for row 0.
            {
                let g0 = logits[0];
                let mut best = (f32::INFINITY, String::new());
                for r in 0..P {
                    let row = &tmp_host[r * E..(r + 1) * E];
                    let mut dot = 0.0f32;
                    for (k, &xv) in row.iter().enumerate() {
                        dot += w[k] * xv;
                    }
                    let d = (dot - g0).abs();
                    if d < best.0 {
                        best = (d, format!("normed row {r} (dot {dot:.4})"));
                    }
                }
                println!(
                    "[i1245-units] run{run}: out[0]={g0:.4}; best host match: {} delta {:.4}",
                    best.1, best.0
                );
                // THE STRIDED-VIEW HYPOTHESIS: the GEMV's N binding is e
                // elements, but if the input slice start is NOT 0 — e.g.
                // the binding resolved to the scratch slice with a stride —
                // the effective input is some contiguous 1024 window of the
                // [P × E] scratch. Scan ALL P*E-1024+1 windows for a dot
                // matching out[0].
                let mut best_win = (f32::INFINITY, i64::MIN);
                let max_start = P * E - E;
                for start in 0..=max_start {
                    let win = &tmp_host[start..start + E];
                    #[allow(clippy::needless_range_loop)]
                    let mut dot = 0.0f32;
                    for (k, &xv) in win.iter().enumerate() {
                        dot += w[k] * xv;
                    }
                    let d = (dot - g0).abs();
                    if d < best_win.0 {
                        best_win = (d, start as i64);
                    }
                }
                println!(
                    "[i1245-units] run{run}: window scan: best start={} delta {:.4} \u{2192} row {} col {}",
                    best_win.1,
                    best_win.0,
                    best_win.1 / E as i64,
                    best_win.1 % E as i64
                );
                // Is the match a FLUKE (many windows within tolerance)? Count
                // windows within 1e-3 of out[0].
                {
                    let mut near = 0usize;
                    for start in 0..=max_start {
                        let win = &tmp_host[start..start + E];
                        let mut dot = 0.0f32;
                        for (k, &xv) in win.iter().enumerate() {
                            dot += w[k] * xv;
                        }
                        if (dot - g0).abs() < 1e-3 {
                            near += 1;
                        }
                    }
                    println!("[i1245-units] run{run}: windows within 1e-3 of out[0]: {near}");
                }
            }
            let mut rows: Vec<(usize, usize)> = hits.into_iter().collect();
            rows.sort_by_key(|a| std::cmp::Reverse(a.1));
            println!(
                "[i1245-units] run{run}: which-row histogram (row=count, top 8): {:?}",
                &rows[..rows.len().min(8)]
            );
        }

        let mut mismatch_full = 0usize;
        for (&g, &r) in logits.iter().zip(logits_ref_fulltmp.iter()) {
            if (g - r).abs() > 1e-2 * (1.0 + r.abs()) {
                mismatch_full += 1;
            }
        }
        println!("[i1245-units] run{run}: vs-FULL-scratch ref mismatches={mismatch_full}/{m}");
        assert!(mismatch * 100 < m, "chunk-tail GEMV wrong at run {run}");
        runs.push(logits);
    }

    assert_eq!(
        bytemuck::cast_slice::<f32, u8>(&runs[0]),
        bytemuck::cast_slice::<f32, u8>(&runs[1]),
        "the tail stage is NOT deterministic run-to-run"
    );
    println!("[i1245-units] determinism: two runs bitwise identical");
}
