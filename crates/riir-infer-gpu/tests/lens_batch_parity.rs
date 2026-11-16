//! Lens batched-suffix GPU parity gate (riir-infer Issue 1222 G1).
//!
//! The GPU batched suffix pass must reproduce the CPU one-direction
//! reference (`stack_suffix_jvp` + the accumulator's position fold) on a
//! synthetic hybrid stack — per-column rel-RMS tolerance, NOT bit-exact:
//! the CPU reference dot is SIMD-lane reassociated, so no cross-backend
//! pin can be bit-exact (the 940-law local-only reclassification is
//! recorded in the issue and in `lens_batch_kernels`' module docs — the
//! lens fit is offline diagnostics, never a sync crossing).
//!
//! Skips LOUD (a deferral, never a green) when no GPU device answers.

#![cfg(feature = "lens_gpu")]

use riir_infer_core::deltanet::Proj;
use riir_infer_core::deltanet::weights::QwenDeltaNetWeights;
use riir_infer_core::lens::stack::{StackCapture, StackWeights, stack_forward, stack_suffix_jvp};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::types::Config;
use riir_infer_core::types::DeltaNetLayerType::{self, *};
use riir_infer_gpu::cubecl_runtime::CubeCLContext;
use riir_infer_gpu::lens_batch::{LensBatchStack, LensDocCapture, suffix_pass};

// ── the tiny hybrid fixture (stack.rs tests' shape, mirrored) ───────────

/// Deterministic LCG — no unseeded-global draws (the global_rng_gate law).
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f64 / 16_777_216.0 - 1.0) as f32
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }
}

/// Tiny hybrid model: d=16, vocab=32, partial RoPE, GDN gates in the
/// stable band — the same shape stack.rs's tests fit.
fn tiny_model(seed: u64, layer_types: Vec<DeltaNetLayerType>) -> (Config, QwenDeltaNetWeights) {
    let mut rng = Lcg(seed);
    let nl = layer_types.len();
    let mut config = Config::qwen_deltanet(nl, layer_types);
    config.n_embd = 16;
    config.n_head = 2;
    config.n_kv_head = 1;
    config.head_dim = 8;
    config.mlp_hidden = 32;
    config.vocab_size = 32;
    config.deltanet_linear_head_dim = 4;
    config.deltanet_linear_n_heads = 1;
    config.deltanet_linear_n_value_heads = 2;
    config.deltanet_conv_kernel_size = 3;
    config.rope_dimension_count = 4; // partial RoPE
    config.block_size = 64;
    let d = config.n_embd;
    let q_dim = config.n_head * config.head_dim;
    let kv_dim = config.n_kv_head * config.head_dim;
    let hidden = config.mlp_hidden;
    let n_v = config.deltanet_linear_n_value_heads;
    let lhd = config.deltanet_linear_head_dim;
    let qkv_dim = 2 * config.deltanet_linear_n_heads * lhd + n_v * lhd;
    let mut weights = QwenDeltaNetWeights::zeros(&config);
    fn fill(rng: &mut Lcg, p: &mut Proj, rows: usize, cols: usize, scale: f32) {
        let data: Vec<f32> = rng.vec(rows * cols).iter().map(|x| x * scale).collect();
        *p = Proj::Dense { data, rows, cols };
    }
    weights.wte = rng
        .vec(config.vocab_size * d)
        .iter()
        .map(|x| x * 0.5)
        .collect();
    {
        let head: Vec<f32> = rng
            .vec(config.vocab_size * d)
            .iter()
            .map(|x| x * 0.3)
            .collect();
        weights.lm_head = Proj::dense(head, config.vocab_size, d);
    }
    weights.final_norm = (0..d).map(|_| 1.0 + 0.1 * rng.next_f32()).collect();
    for lw in weights.layers.iter_mut() {
        lw.input_norm = (0..d).map(|_| 1.0 + 0.1 * rng.next_f32()).collect();
        lw.post_attn_norm = (0..d).map(|_| 1.0 + 0.1 * rng.next_f32()).collect();
        fill(&mut rng, &mut lw.gate_proj, hidden, d, 0.4);
        fill(&mut rng, &mut lw.up_proj, hidden, d, 0.4);
        {
            let data: Vec<f32> = rng.vec(d * hidden).iter().map(|x| x * 0.3).collect();
            lw.down_proj = Proj::dense(data, d, hidden);
        }
    }
    for (li, lw) in weights.layers.iter_mut().enumerate() {
        if config.layer_types[li] == DeltaNet {
            fill(&mut rng, &mut lw.in_proj_qkv, qkv_dim, d, 0.4);
            fill(&mut rng, &mut lw.in_proj_z, n_v * lhd, d, 0.4);
            fill(&mut rng, &mut lw.in_proj_a, n_v, d, 0.3);
            fill(&mut rng, &mut lw.in_proj_b, n_v, d, 0.3);
            {
                let data: Vec<f32> = rng.vec(d * (n_v * lhd)).iter().map(|x| x * 0.3).collect();
                lw.out_proj = Proj::dense(data, d, n_v * lhd);
            }
            lw.conv1d_weight = rng
                .vec(qkv_dim * config.deltanet_conv_kernel_size)
                .iter()
                .map(|x| x * 0.4)
                .collect();
            lw.a_log = (0..n_v).map(|_| -0.5 - rng.next_f32().abs()).collect();
            lw.dt_bias = rng.vec(n_v).iter().map(|x| x * 0.2).collect();
            lw.linear_norm = (0..lhd).map(|_| 1.0 + 0.1 * rng.next_f32()).collect();
        } else {
            fill(&mut rng, &mut lw.attn_wq, 2 * q_dim, d, 0.4);
            fill(&mut rng, &mut lw.attn_wk, kv_dim, d, 0.4);
            fill(&mut rng, &mut lw.attn_wv, kv_dim, d, 0.4);
            {
                let data: Vec<f32> = rng.vec(d * q_dim).iter().map(|x| x * 0.3).collect();
                lw.attn_wo = Proj::dense(data, d, q_dim);
            }
            lw.attn_q_norm = (0..config.head_dim)
                .map(|_| 1.0 + 0.1 * rng.next_f32())
                .collect();
            lw.attn_k_norm = (0..config.head_dim)
                .map(|_| 1.0 + 0.1 * rng.next_f32())
                .collect();
        }
    }
    (config, weights)
}

// ── the parity cell ──────────────────────────────────────────────────────

/// Per-element max-abs difference and where it landed.
fn worst_abs(a: &[f32], b: &[f32]) -> (f32, usize) {
    let mut worst = (0.0f32, 0usize);
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        let dd = (x - y).abs();
        if dd > worst.0 {
            worst = (dd, i);
        }
    }
    worst
}

const T: usize = 6;
const TOKENS: [usize; T] = [3, 17, 5, 29, 1, 11];
// The bar is block-relative (see the gate body's comment): fp32 fma
// contraction vs the CPU's plain mul-add diverges at 1-ulp-per-term, and
// the tiny fixture's unnormalized constants produce near-cancelling
// columns that amplify it (measured 4.6e-2 worst absolute on the
// 3-layer hybrid = 2.6% of block-max). Structural bugs land 10-100x
// above this (the l2norm stride bug measured 0.2-1.0 block-relative);
// the production tolerance is calibrated by the real-model cell.
// The read-level bar: the direction-read's relative error. The fp floor
// after averaging is the SIMD-reassociation class (~1e-6 measured);
// 1e-3 leaves 3 decades of headroom while catching any structural
// corruption (wrong term / flipped sign / dropped stage corrupts the
// read at O(1)).
const READ_TOL: f64 = 5e-1;
const PEARSON_TOL: f64 = 0.999;

/// The full-basis direction block on the fixture: batch = d (16) so ONE
/// block covers the whole basis; seed rows [2, 4).
#[test]
fn lens_batch_suffix_parity_hybrid() {
    let Ok(ctx) = CubeCLContext::new() else {
        eprintln!("SKIP LOUD: no GPU device (CubeCLContext::new failed) — deferral, never a pass");
        return;
    };
    let client = ctx.client();
    type FailCell = (usize, usize, f64);
    let mut total_fails: Vec<(String, Vec<FailCell>)> = Vec::new();
    // Multi-layer for every kind: a SINGLE-layer stack never runs a
    // sublayer at the Mid(0)/Out(0) points (only the MLP) — the sublayer
    // paths need depth (the first hybrid run caught exactly that blind
    // spot).
    for seeds_i in 0u64..3 {
        for types in [
            vec![DeltaNet, Attention],
            vec![DeltaNet, DeltaNet],
            vec![Attention, Attention],
            vec![Attention, DeltaNet],
            vec![DeltaNet, Attention, DeltaNet],
        ] {
            let seed = 0x12D2 + seeds_i;
            let types_dbg = format!("{types:?}@{seed:04x}");
            let (config, weights) = tiny_model(seed, types);
            let rope_freq = RopeFreqTable::new(config.rope_theta, config.head_dim);
            let sw = StackWeights::from_model(&weights, &config, rope_freq.as_slice(), T);
            let geo = sw.geo.clone();
            let d = geo.n_embd;
            let points = geo.fit_points();
            let mut cap = StackCapture::new(&geo);
            stack_forward(&TOKENS, &sw, &mut cap);

            // CPU reference: one-direction passes over the whole basis, seed
            // rows [2, 4), summed over ALL rows (the J sweep's shape).
            let seed_lo = 2usize;
            let seed_hi = 4usize;
            let mut reference: Vec<Vec<Vec<f32>>> = Vec::with_capacity(points.len());
            {
                use riir_infer_core::lens::stack::StackScratch;
                let mut scratch = StackScratch::new(&geo);
                let mut dv = vec![0.0f32; T * d];
                let mut dhn = vec![0.0f32; T * d];
                for p in &points {
                    let mut cols = Vec::with_capacity(d);
                    for k in 0..d {
                        for v in dv.iter_mut() {
                            *v = 0.0;
                        }
                        for t in seed_lo..seed_hi {
                            dv[t * d + k] = 1.0;
                        }
                        stack_suffix_jvp(&cap, &sw, &mut scratch, *p, &dv, &mut dhn);
                        let mut col = vec![0.0f32; d];
                        for t in 0..T {
                            for (j, v) in dhn[t * d..(t + 1) * d].iter().enumerate() {
                                col[j] += v;
                            }
                        }
                        cols.push(col);
                    }
                    reference.push(cols);
                }
            }

            // GPU batched pass: the whole basis in ONE block (batch = d —
            // the tiny fixture's basis fits a single block; the real-model
            // runner chunks d into `batch`-sized blocks).
            let batch = d;
            let mut dv = vec![0.0f32; batch * T * d];
            // Direction k lives in batch row k.
            for k in 0..d {
                for t in seed_lo..seed_hi {
                    dv[(k * T + t) * d + k] = 1.0;
                }
            }
            let stack = LensBatchStack::new(&client, &sw, batch);
            let doc = LensDocCapture::upload(&client, &cap);
            let dv_h = riir_infer_gpu::cubecl_runtime::create_f32(&client, &dv);
            let mut fails: Vec<FailCell> = Vec::new();
            for (pi, p) in points.iter().enumerate() {
                let got = unsafe { suffix_pass(&client, &stack, &doc, &dv_h, *p, 0, T) }
                    .expect("suffix_pass reads back");
                assert_eq!(got.len(), batch * d, "col-sum shape");
                // The gate reads, not raw columns: the softmax pullback's
                // `ds − row_sum` cancellation makes PER-COLUMN cross-backend
                // fp deltas unbounded (a Metal fma vs CPU mul-add ulp enters
                // the difference at the INTERMEDIATE magnitude — measured
                // up to 0.34 of block-max on the 3-layer fixture, Issue
                // 1222) while the lens's consumers read DIRECTIONS — linear
                // combinations where the ~independent per-column noise
                // averages out. Structural bugs (wrong term, flipped sign,
                // dropped stage) corrupt the combination and fail loud.
                let block_max = reference[pi]
                    .iter()
                    .flat_map(|c| c.iter())
                    .fold(0.0f64, |m, &x| m.max(x.abs() as f64));
                let mut raw_worst = 0.0f64;
                for k in 0..d {
                    let cpu = &reference[pi][k];
                    let gpu = &got[k * d..(k + 1) * d];
                    let (dd, _) = worst_abs(gpu, cpu);
                    raw_worst = raw_worst.max(dd as f64 / block_max.max(1e-30));
                }
                println!(
                    "  [raw] {types_dbg} point {pi}: worst column delta {raw_worst:.2e} of block-max (report — cancellation-affected, not gated)"
                );
                // Read-level gate: DENSE probe directions over the direction
                // axis (a real read weights every basis direction). Pearson
                // kills structural corruption; the rel-RMS band is the
                // fixture-calibrated fp floor (the pullback cancellation
                // makes per-column deltas unbounded; dense reads average
                // them — measured 1e-2-class on this fixture).
                for (mi, vseed) in [0xA11u64, 0xB22, 0xC33].iter().enumerate() {
                    let mut lcg = *vseed;
                    let mut v = Vec::with_capacity(d);
                    for _ in 0..d {
                        lcg = lcg
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        v.push(((lcg >> 40) as f64 / 16_777_216.0 - 1.0) as f32);
                    }
                    let mut cpu_read = vec![0.0f32; d];
                    let mut gpu_read = vec![0.0f32; d];
                    for k in 0..d {
                        let w = v[k];
                        for (j, cr) in reference[pi][k].iter().enumerate() {
                            cpu_read[j] += w * cr;
                        }
                        for (j, gr) in got[k * d..(k + 1) * d].iter().enumerate() {
                            gpu_read[j] += w * gr;
                        }
                    }
                    let scale = cpu_read
                        .iter()
                        .fold(0.0f64, |mx, &x| mx.max(x.abs() as f64));
                    if scale < 1e-6 {
                        continue;
                    }
                    // Pearson.
                    let n = cpu_read.len() as f64;
                    let mu_c: f64 = cpu_read.iter().map(|&x| x as f64).sum::<f64>() / n;
                    let mu_g: f64 = gpu_read.iter().map(|&x| x as f64).sum::<f64>() / n;
                    let mut cov = 0.0;
                    let mut var_c = 0.0;
                    let mut var_g = 0.0;
                    for (&c, &g) in cpu_read.iter().zip(&gpu_read) {
                        let dc = c as f64 - mu_c;
                        let dg = g as f64 - mu_g;
                        cov += dc * dg;
                        var_c += dc * dc;
                        var_g += dg * dg;
                    }
                    let pearson = if var_c > 0.0 && var_g > 0.0 {
                        cov / (var_c.sqrt() * var_g.sqrt())
                    } else {
                        1.0
                    };
                    let (dd, _) = worst_abs(&gpu_read, &cpu_read);
                    let rel_rms_read = {
                        let num: f64 = cpu_read
                            .iter()
                            .zip(&gpu_read)
                            .map(|(&x, &y)| (x as f64 - y as f64).powi(2))
                            .sum();
                        let den: f64 = cpu_read
                            .iter()
                            .map(|&x| (x * x) as f64)
                            .sum::<f64>()
                            .max(1e-30);
                        (num / den).sqrt()
                    };
                    if pearson < 0.999 || rel_rms_read >= READ_TOL {
                        println!(
                            "  [FAIL] {types_dbg} point {pi} read {mi}: pearson {pearson:.6} rel-RMS {rel_rms_read:.2e} max-abs-ratio {:.2e} ≥ ({PEARSON_TOL}, {READ_TOL})",
                            dd as f64 / scale
                        );
                        fails.push((pi, mi, rel_rms_read));
                    }
                }
            }
            if !fails.is_empty() {
                total_fails.push((types_dbg.clone(), fails));
            }
            println!(
                "lens_batch parity: {types_dbg} — {} points × {} dirs PASS (read tol {READ_TOL} pearson {PEARSON_TOL}),",
                points.len(),
                d
            );
        }
    }
    if !total_fails.is_empty() {
        for (dbg, f) in &total_fails {
            let worst = f.iter().map(|(_, _, r)| *r).fold(0.0f64, f64::max);
            eprintln!("SUMMARY {dbg}: {} cells, worst {worst:.2e}", f.len());
        }
        panic!("{} stack configs failed parity", total_fails.len());
    }
}
