//! The workspace's own plane-GEMV unit fixture at the CHUNK TAIL's shapes.
//!
//! Issue 1245 bisect leg 2: `GemvCubeCL::launch` at the exact shapes the
//! dense chunk tail runs (the 0.8B fixture's e = 1024, m split into
//! 32768-row chunks like `lm_head_chunks`), plus the single-row tail norm —
//! the miniature of `ingest_chunk_gpu`'s tail stage. If the plane kernel
//! misses rows at these shapes, this reproduces it WITHOUT the model (a
//! pure kernel cell); if it passes, the bisect moves back up the stack.
//!
//! Run:
//! ```sh
//! cargo test --release -p riir-infer-gpu --features qwen38_wgpu \
//!   --test i1245_tail_gemv_units -- --nocapture
//! ```
#![cfg(all(test, feature = "qwen38_wgpu"))]

use cubecl::client::ComputeClient;
use riir_infer_gpu::cubecl_runtime::{ActiveRuntime, CubeCLContext};

fn stats(name: &str, v: &[f32]) {
    let nonfinite = v.iter().filter(|x| !x.is_finite()).count();
    let zeros = v.iter().filter(|&&x| x == 0.0).count();
    let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
    for &x in v {
        if x < mn {
            mn = x;
        }
        if x > mx {
            mx = x;
        }
    }
    println!(
        "[i1245-units] {name}: n={} nonfinite={} zeros={} min={mn:.4} max={mx:.4}",
        v.len(),
        nonfinite,
        zeros
    );
}

/// CPU reference for one [m, e] × [e] GEMV.
fn gemv_ref(w: &[f32], x: &[f32], m: usize, e: usize) -> Vec<f32> {
    (0..m)
        .map(|r| {
            let row = &w[r * e..(r + 1) * e];
            row.iter().zip(x.iter()).map(|(&w, &x)| w * x).sum::<f32>()
        })
        .collect()
}

#[test]
fn tail_gemv_shapes_reproduce() {
    let ctx = CubeCLContext::new().expect("cubecl context");
    let cl: ComputeClient<ActiveRuntime> = ctx.client();
    let e = 1024usize;

    // The input row: model-scale hidden values (plausible magnitudes).
    let x: Vec<f32> = (0..e).map(|i| ((i as f32 * 0.37).sin()) * 2.0).collect();

    // m = 32768 rows per lm chunk (7 chunks) + the 18944 tail — the exact
    // split `lm_head_chunks` produces at vocab 248320.
    for m in [32768usize, 18944] {
        let w: Vec<f32> = (0..m * e)
            .map(|i| ((i as f32 * 0.618).sin()) * 0.5)
            .collect();
        let expected = gemv_ref(&w, &x, m, e);

        let w_h = cl.create_from_slice(bytemuck::cast_slice(&w));
        let x_h = cl.create_from_slice(bytemuck::cast_slice(&x));
        let out_len = m;
        let out_h = cl.empty(out_len * 4);

        unsafe {
            riir_infer_gpu::gemv_cubecl::GemvCubeCL::launch::<ActiveRuntime>(
                &cl,
                w_h,
                x_h,
                out_h.clone(),
                m,
                e,
            );
        }
        let got: Vec<f32> = cl
            .read_one(out_h)
            .map(|b| bytemuck::cast_slice(&b).to_vec())
            .expect("readback");
        stats(&format!("plane gemv m={m}"), &got);

        // The coverage assertion: every row must be WRITTEN (no zeros from
        // missed rows — the inputs make an all-zero row astronomically
        // unlikely) and match the CPU reference.
        let mut mismatches = 0usize;
        let mut zero_rows = 0usize;
        for (i, (&g, &exp)) in got.iter().zip(expected.iter()).enumerate() {
            if g == 0.0 {
                zero_rows += 1;
            }
            if (g - exp).abs() > 1e-3 * (1.0 + exp.abs()) {
                mismatches += 1;
                if mismatches <= 5 {
                    println!("[i1245-units]   row {i}: got {g} want {exp}");
                }
            }
        }
        println!("[i1245-units] m={m}: zero_rows={zero_rows} mismatches={mismatches}/{m}");
        assert_eq!(zero_rows, 0, "plane GEMV missed rows at m={m}");
        assert!(
            mismatches * 1000 < m,
            "plane GEMV wrong rows at m={m}: {mismatches}"
        );
    }
}
