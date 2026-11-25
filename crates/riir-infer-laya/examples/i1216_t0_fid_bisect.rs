//! Issue 1216 T0 fidelity bisect (plan 634 addendum) — the drill proved the
//! python f64 replay matches the reference at 5.25e-7 mean while the
//! ops::layer_norm + ops::matmul_w host replay sits at 2.30e-3. This
//! example isolates WHICH stage breaks: it computes, for the first 3 cache
//! rows, (a) LN rows via ops::layer_norm, (b) the frozen logits THROUGH
//! matmul_w, (c) the frozen logits via an in-row f64 loop (no gemm), and
//! prints the three deltas against the cache's reference logits. Whichever
//! arm diverges names the broken op.
//!
//! Instrument-grade, `--release --example`, reads the fid LENC cache.

use std::io::Read;

use riir_infer_laya::laya::config::Checkpoint;
use riir_infer_laya::laya::riir::agent::{DeviceKind, RiirAgent};
use riir_infer_laya::laya::riir::ops;

const MAGIC: &[u8; 4] = b"LENC";
const N_ROWS: usize = 3;

fn main() -> Result<(), String> {
    let root = riir_infer_laya::laya::weights::weights_root();
    let agent = RiirAgent::load_with_device(&root, Checkpoint::TypedDecisions, DeviceKind::Cpu)
        .map_err(|e| e.to_string())?;
    let st = agent.scorer_tensors();
    let _d = st.d;

    let mut f =
        std::fs::File::open("/tmp/i1216_t0/fid_cpu_cache.bin").map_err(|e| e.to_string())?;
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).map_err(|e| e.to_string())?;
    if &magic != MAGIC {
        return Err("not LENC".into());
    }
    let u32v = |f: &mut std::fs::File| -> Result<u32, String> {
        let mut b = [0u8; 4];
        f.read_exact(&mut b).map_err(|e| e.to_string())?;
        Ok(u32::from_le_bytes(b))
    };
    let _ver = u32v(&mut f)?;
    let n_rows = u32v(&mut f)? as usize;
    let take = n_rows.min(N_ROWS);

    for _row in 0..take {
        let id_len = u32v(&mut f)? as usize;
        let mut idb = vec![0u8; id_len];
        f.read_exact(&mut idb).map_err(|e| e.to_string())?;
        let id = String::from_utf8(idb).map_err(|e| e.to_string())?;
        let seq_len = u32v(&mut f)? as usize;
        let row_d = u32v(&mut f)? as usize;
        let _qtype = u32v(&mut f)?;
        let k = u32v(&mut f)? as usize;
        drop_n(&mut f, k * 4)?;
        let mut markers = vec![0f32; k * row_d];
        read_f32(&mut f, &mut markers)?;
        drop_n(&mut f, seq_len * row_d * 4)?;
        let mut ref_logits = vec![0f32; k];
        read_f32(&mut f, &mut ref_logits)?;

        // (a) LN via ops::layer_norm
        let x = ops::layer_norm(&markers, &st.s0w, &st.s0b, st.eps, row_d);
        // (b) frozen logits through ops::matmul_w (h = gelu(x@W1+b1))
        let s1 = ops::matmul_w(&x, k, row_d, &st.s1w, row_d);
        let mut h = s1.clone();
        for (r, b) in h.chunks_exact_mut(row_d).zip(st.s1b.chunks_exact(row_d)) {
            for (v, bi) in r.iter_mut().zip(b.iter()) {
                *v += bi;
            }
        }
        ops::gelu_erf_inplace(&mut h);
        let z_gemm = {
            let mut z = ops::matmul_w(&h, k, row_d, &st.s3w, 1);
            for v in z.iter_mut() {
                *v += st.s3b[0];
            }
            z
        };
        // (c) frozen logits via a plain f64 loop over the SAME h
        // (index form deliberate — the pointer form reads worse here)
        #[allow(clippy::needless_range_loop)]
        let z_loop: Vec<f32> = (0..k)
            .map(|i| {
                let h_i = &h[i * row_d..(i + 1) * row_d];
                let mut acc = st.s3b[0] as f64;
                for j in 0..row_d {
                    acc += h_i[j] as f64 * st.s3w[j] as f64;
                }
                acc as f32
            })
            .collect();
        // (d) h via the in-row f64 loop (no gemm for W1 either) then z via f64
        let z_f64: Vec<f32> = (0..k)
            .map(|i| {
                let x_i = &x[i * row_d..(i + 1) * row_d];
                let mut acc = st.s3b[0] as f64;
                for o in 0..row_d {
                    let wrow = &st.s1w[o * row_d..(o + 1) * row_d];
                    let mut a = st.s1b[o] as f64;
                    for l in 0..row_d {
                        a += x_i[l] as f64 * wrow[l] as f64;
                    }
                    // erf gelu in f64 then cast
                    let g = 0.5 * a * (1.0 + erf64(a / std::f64::consts::SQRT_2));
                    acc += (g as f32) as f64 * st.s3w[o] as f64;
                }
                acc as f32
            })
            .collect();

        // (e) the s1 gemm vs the f64 s1 — element diff (the decisive read)
        let mut s1_f64 = vec![0f64; k * row_d];
        for i in 0..k {
            let x_i = &x[i * row_d..(i + 1) * row_d];
            for o in 0..row_d {
                let wrow = &st.s1w[o * row_d..(o + 1) * row_d];
                let mut a = st.s1b[o] as f64;
                for l in 0..row_d {
                    a += x_i[l] as f64 * wrow[l] as f64;
                }
                s1_f64[i * row_d + o] = a;
            }
        }
        let mut worst = (0usize, 0f32, 0f64);
        let mut n_big = 0usize;
        let mut max_abs_a1 = 0f32;
        for i in 0..k * row_d {
            let dlt = (h_gelu_of(s1[i]) - h_gelu_of(s1_f64[i] as f32)).abs();
            if dlt > 1e-5 {
                n_big += 1;
            }
            let a = s1[i].abs();
            if a > max_abs_a1 {
                max_abs_a1 = a;
                worst = (i, s1[i], s1_f64[i]);
            }
        }
        println!(
            "  s1 diff: n(|Δh|>1e-5)={n_big}/{} max|a1|={max_abs_a1:.3} worst-elt a1_gemm={:+.6} a1_f64={:+.6}",
            k * row_d,
            worst.1,
            worst.2
        );

        println!("row {id}:");
        for oi in 0..k {
            println!(
                "  opt {oi}: ref {:+.6} gemm {:+.6} (Δ{:.2e}) loop-on-h {:+.6} (Δ{:.2e}) f64 {:+.6} (Δ{:.2e})",
                ref_logits[oi],
                z_gemm[oi],
                (z_gemm[oi] - ref_logits[oi]).abs(),
                z_loop[oi],
                (z_loop[oi] - ref_logits[oi]).abs(),
                z_f64[oi],
                (z_f64[oi] - ref_logits[oi]).abs()
            );
        }
    }
    Ok(())
}

fn erf64(x: f64) -> f64 {
    // python-math-erf-equivalent via the Abramowitz–Stegun 7.1.26 (|err|<1.5e-7
    // — good enough to separate a 2e-3-class break)
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    sign * (1.0 - poly * (-ax * ax).exp())
}

fn h_gelu_of(a: f32) -> f32 {
    // the crate's gelu scalar law (erf(v/√2)+1)·0.5·v — via libm on f32
    // is not exported here; the f64 erf cast matches to <1e-6, fine for a
    // >1e-5-counting bisect
    let g = 0.5 * (a as f64) * (1.0 + erf64((a as f64) / std::f64::consts::SQRT_2));
    g as f32
}

fn read_f32(f: &mut std::fs::File, out: &mut [f32]) -> Result<(), String> {
    let mut bytes = vec![0u8; out.len() * 4];
    f.read_exact(&mut bytes).map_err(|e| e.to_string())?;
    for (i, v) in out.iter_mut().enumerate() {
        *v = f32::from_le_bytes([
            bytes[i * 4],
            bytes[i * 4 + 1],
            bytes[i * 4 + 2],
            bytes[i * 4 + 3],
        ]);
    }
    Ok(())
}

fn drop_n(f: &mut std::fs::File, n_bytes: usize) -> Result<(), String> {
    let mut remaining = n_bytes;
    let mut buf = [0u8; 1 << 16];
    while remaining > 0 {
        let take = remaining.min(buf.len());
        f.read_exact(&mut buf[..take]).map_err(|e| e.to_string())?;
        remaining -= take;
    }
    Ok(())
}
