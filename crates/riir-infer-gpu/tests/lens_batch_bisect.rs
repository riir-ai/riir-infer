//! Isolated per-layer bisect for the [DeltaNet, Attention, DeltaNet]
//! parity divergence (riir-infer Issue 1222 debug cell — NOT a gate).
//! Seeds an IDENTICAL random tangent into the GPU driver's sublayer step
//! and a hand-composed CPU twin (public ops only), per layer — no prefix,
//! fully isolated. Temporary instrumentation; removed after the finding.

#![cfg(feature = "lens_gpu")]

use riir_infer_core::deltanet::Proj;
use riir_infer_core::deltanet::weights::QwenDeltaNetWeights;
use riir_infer_core::lens::ops::{diag_apply, mlp_half_rule_jvp};
use riir_infer_core::lens::stack::FitPoint;
use riir_infer_core::lens::stack::{StackCapture, StackWeights, stack_forward};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::types::DeltaNetLayerType::*;
use riir_infer_core::types::{Config, matmul};
use riir_infer_gpu::cubecl_runtime::{CubeCLContext, create_f32, read_f32};
use riir_infer_gpu::lens_batch::{
    LensBatchStack, LensDocCapture, debug_attn_stage, debug_gdn_stage, debug_sublayer_readback,
};
use riir_infer_gpu::lens_batch_kernels::{LensAttnJvpCL, LensGdnScanCL, LensQkNormCL};

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

fn tiny_model(
    seed: u64,
    layer_types: Vec<riir_infer_core::types::DeltaNetLayerType>,
) -> (Config, QwenDeltaNetWeights) {
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
    config.rope_dimension_count = 4;
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

const T: usize = 6;
const TOKENS: [usize; T] = [3, 17, 5, 29, 1, 11];
const BATCH: usize = 16;

fn worst_diff(cpu: &[f32], gpu: &[f32]) -> (f32, usize) {
    let mut worst = (0.0f32, 0usize);
    for (i, (&a, &b)) in cpu.iter().zip(gpu).enumerate() {
        let dd = (a - b).abs();
        if dd > worst.0 {
            worst = (dd, i);
        }
    }
    worst
}

fn worst_rel(cpu: &[f32], gpu: &[f32]) -> f64 {
    let num: f64 = cpu
        .iter()
        .zip(gpu)
        .map(|(&a, &b)| ((a - b) as f64).powi(2))
        .sum();
    let den: f64 = cpu.iter().map(|&a| (a * a) as f64).sum::<f64>().max(1e-30);
    (num / den).sqrt()
}

#[test]
fn bisect_dad_layers_isolated() {
    let Ok(ctx) = CubeCLContext::new() else {
        eprintln!("SKIP LOUD: no GPU device");
        return;
    };
    let client = ctx.client();
    let seed = 0x12D3;
    let (config, weights) = tiny_model(seed, vec![DeltaNet, Attention, DeltaNet]);
    let rope_freq = RopeFreqTable::new(config.rope_theta, config.head_dim);
    let sw = StackWeights::from_model(&weights, &config, rope_freq.as_slice(), T);
    let geo = sw.geo.clone();
    let d = geo.n_embd;
    let mut cap = StackCapture::new(&geo);
    stack_forward(&TOKENS, &sw, &mut cap);

    let stack = LensBatchStack::new(&client, &sw, BATCH);
    let doc = LensDocCapture::upload(&client, &cap);

    // One random batched tangent, same for every layer probe.
    let mut rng = Lcg(0xB155);
    let dx: Vec<f32> = rng.vec(BATCH * T * d);
    let _t_h = create_f32(&client, &dx);

    for li in 0..geo.n_layers {
        // Run the driver's sublayer half on a PRE-DIAGNOSTICED input (the
        // CPU twin applies the input-norm diag itself, so both sides start
        // from the same post-diag tangent).
        let mut cpu_in = dx.clone();
        let lc0 = &cap.layers[li];
        for t in 0..T {
            diag_apply(
                &lc0.input_scale[t * d..(t + 1) * d],
                &mut cpu_in[t * d..(t + 1) * d],
            );
        }
        let cpu_in_h = create_f32(&client, &cpu_in);
        let gpu_sub = unsafe { debug_sublayer_readback(&client, &stack, &doc, li, &cpu_in_h) }
            .expect("sub readback");

        // CPU twin: hand-compose the sublayer step on cpu_in.
        let lw = &sw.layers[li];
        let lc = &cap.layers[li];
        let td = T * d;
        let mut cpu_t2 = vec![0.0f32; td];
        let gdn_geo = geo.gdn_geo();
        let attn_geo = geo.attn_geo();
        match (&lc.sub, &lw.sub) {
            (
                riir_infer_core::lens::stack::SubCapture::Gdn(gc),
                riir_infer_core::lens::stack::SubWeights::Gdn(gw),
            ) => {
                let (n_k, n_v, key, val) = (
                    gdn_geo.n_k_heads,
                    gdn_geo.n_v_heads,
                    gdn_geo.key_dim,
                    gdn_geo.val_dim,
                );
                let conv_dim = gdn_geo.conv_dim();
                let q_dim = gdn_geo.q_dim();
                let v_dim = gdn_geo.v_dim();
                use riir_infer_core::lens::gdn::{
                    GdnRecurrenceConsts, L2NormConsts, expand_tiled_jvp, gdn_recurrence_jvp,
                    l2_norm_jvp,
                };
                use riir_infer_core::lens::ops::{conv1d_causal_jvp, half_rule_jvp};
                // 1. projections.
                let mut dqkv_raw = vec![0.0f32; T * conv_dim];
                let mut dz = vec![0.0f32; T * v_dim];
                let mut da = vec![0.0f32; T * n_v];
                let mut db = vec![0.0f32; T * n_v];
                for t in 0..T {
                    let x_t = &cpu_in[t * d..(t + 1) * d];
                    let mut y = vec![0.0f32; conv_dim];
                    matmul(&mut y, gw.w_qkv, x_t, conv_dim, d);
                    dqkv_raw[t * conv_dim..(t + 1) * conv_dim].copy_from_slice(&y);
                    let mut z = vec![0.0f32; v_dim];
                    matmul(&mut z, gw.w_z, x_t, v_dim, d);
                    dz[t * v_dim..(t + 1) * v_dim].copy_from_slice(&z);
                    let mut a = vec![0.0f32; n_v];
                    matmul(&mut a, gw.w_a, x_t, n_v, d);
                    da[t * n_v..(t + 1) * n_v].copy_from_slice(&a);
                    let mut b = vec![0.0f32; n_v];
                    matmul(&mut b, gw.w_b, x_t, n_v, d);
                    db[t * n_v..(t + 1) * n_v].copy_from_slice(&b);
                }
                // 2. conv + silu diag.
                let mut dconv = vec![0.0f32; T * conv_dim];
                conv1d_causal_jvp(
                    gw.conv_w,
                    gdn_geo.kernel_size,
                    &dqkv_raw,
                    conv_dim,
                    &mut dconv,
                );
                let mut dqkv = vec![0.0f32; T * conv_dim];
                for i in 0..T * conv_dim {
                    dqkv[i] = gc.conv_silu_grad[i] * dconv[i];
                }
                let mut dq = vec![0.0f32; T * q_dim];
                let mut dk = vec![0.0f32; T * q_dim];
                let mut dvv = vec![0.0f32; T * v_dim];
                for t in 0..T {
                    dq[t * q_dim..(t + 1) * q_dim]
                        .copy_from_slice(&dqkv[t * conv_dim..t * conv_dim + q_dim]);
                    dk[t * q_dim..(t + 1) * q_dim]
                        .copy_from_slice(&dqkv[t * conv_dim + q_dim..t * conv_dim + 2 * q_dim]);
                    dvv[t * v_dim..(t + 1) * v_dim]
                        .copy_from_slice(&dqkv[t * conv_dim + 2 * q_dim..(t + 1) * conv_dim]);
                }
                // 3. gate coefs.
                let mut dbeta = vec![0.0f32; T * n_v];
                let mut ddecay = vec![0.0f32; T * n_v];
                for i in 0..T * n_v {
                    dbeta[i] = gc.dbeta_coef[i] * db[i];
                    ddecay[i] = gc.ddecay_coef[i] * da[i];
                }
                // 4. l2norm + expand (q, k).
                let block = n_k * key;
                let mut dqn = vec![0.0f32; T * n_v * key];
                let mut dkn = vec![0.0f32; T * n_v * key];
                for t in 0..T {
                    let qc = L2NormConsts {
                        y: &gc.qn[t * n_v * key..t * n_v * key + block],
                        inv: &gc.qn_inv[t * n_k..(t + 1) * n_k],
                    };
                    let mut src = dq[t * q_dim..(t + 1) * q_dim].to_vec();
                    l2_norm_jvp(&qc, key, &mut src);
                    expand_tiled_jvp(
                        &src,
                        n_k,
                        key,
                        n_v / n_k,
                        &mut dqn[t * n_v * key..(t + 1) * n_v * key],
                    );
                    let kc = L2NormConsts {
                        y: &gc.kn[t * n_v * key..t * n_v * key + block],
                        inv: &gc.kn_inv[t * n_k..(t + 1) * n_k],
                    };
                    let mut srck = dk[t * q_dim..(t + 1) * q_dim].to_vec();
                    l2_norm_jvp(&kc, key, &mut srck);
                    expand_tiled_jvp(
                        &srck,
                        n_k,
                        key,
                        n_v / n_k,
                        &mut dkn[t * n_v * key..(t + 1) * n_v * key],
                    );
                }
                // 5. scan.
                let rc = GdnRecurrenceConsts {
                    t_len: T,
                    n_v_heads: n_v,
                    key_dim: key,
                    val_dim: val,
                    scale: gdn_geo.scale(),
                    q: &gc.qn,
                    k: &gc.kn,
                    v: &gc.v,
                    beta: &gc.beta,
                    decay: &gc.decay,
                    states: &gc.states,
                };
                let mut ds = vec![0.0f32; n_v * val * key];
                let mut spre = vec![0.0f32; n_v * val * key];
                let mut dy_pre = vec![0.0f32; T * v_dim];
                gdn_recurrence_jvp(
                    &rc,
                    &dqn,
                    &dkn,
                    &dvv,
                    &dbeta,
                    &ddecay,
                    &mut ds,
                    &mut spre,
                    &mut dy_pre,
                );
                // 6. ssm norm (separate buffer — the w_out writes below
                // would clobber a shared [T·d] scratch).
                let mut dnormed = vec![0.0f32; T * v_dim];
                for t in 0..T {
                    use riir_infer_core::lens::attn::{QkNormConsts, qk_norm_true_jvp};
                    let qkc = QkNormConsts {
                        x: &gc.y_pre[t * v_dim..(t + 1) * v_dim],
                        r: &gc.norm_r[t * n_v..(t + 1) * n_v],
                    };
                    qk_norm_true_jvp(
                        &qkc,
                        gw.linear_norm,
                        val,
                        &dy_pre[t * v_dim..(t + 1) * v_dim],
                        &mut dnormed[t * v_dim..(t + 1) * v_dim],
                    );
                }
                // 7-8. gate half rule (dg=dz, du=dnormed).
                for t in 0..T {
                    use riir_infer_core::lens::ops::MlpConsts;
                    let mc = MlpConsts {
                        s: &gc.gate_s[t * v_dim..(t + 1) * v_dim],
                        u: &gc.normed[t * v_dim..(t + 1) * v_dim],
                        sp: &gc.gate_sp[t * v_dim..(t + 1) * v_dim],
                    };
                    let mut dout = vec![0.0f32; v_dim];
                    half_rule_jvp(
                        &mc,
                        &dz[t * v_dim..(t + 1) * v_dim],
                        &dnormed[t * v_dim..(t + 1) * v_dim],
                        &mut dout,
                    );
                    let mut y = vec![0.0f32; d];
                    matmul(&mut y, gw.w_out, &dout, d, v_dim);
                    cpu_t2[t * d..(t + 1) * d].copy_from_slice(&y);
                }
            }
            (
                riir_infer_core::lens::stack::SubCapture::Attn(ac),
                riir_infer_core::lens::stack::SubWeights::Attn(aw),
            ) => {
                use riir_infer_core::lens::attn::{
                    AttnLayerConsts, AttnLayerScratch, attn_layer_jvp,
                };
                let consts = AttnLayerConsts {
                    w: *aw,
                    geo: attn_geo,
                    cap: ac,
                };
                let mut asc = AttnLayerScratch::new(&attn_geo);
                // The twin composes the PROBE ROW's slice (cpu_in is the
                // full batched tangent).
                let mut row_out = vec![0.0f32; td];
                attn_layer_jvp(&consts, &cpu_in[0..td], &mut asc, &mut row_out);
                cpu_t2.copy_from_slice(&row_out);
            }
            _ => panic!("sub kind mismatch"),
        }

        // Compare the (p=1) batch row of the sublayer output.
        let gpu_row = &gpu_sub[..td];
        let cpu_row = &cpu_t2;
        let rel = worst_rel(cpu_row, gpu_row);
        let is_gdn = geo.layer_is_gdn[li];
        println!(
            "layer {li} ({}): sublayer rel-RMS = {rel:.3e} | cpu_norm {:.3e} gpu_norm {:.3e}",
            if is_gdn { "GDN" } else { "ATTN" },
            cpu_row.iter().map(|&a| (a * a) as f64).sum::<f64>().sqrt(),
            gpu_row.iter().map(|&a| (a * a) as f64).sum::<f64>().sqrt()
        );
        // Per-position first-dim values — the divergence growth pattern.
        for t in 0..T {
            println!(
                "  t{t}: cpu={:+.4} gpu={:+.4}",
                cpu_row[t * d],
                gpu_row[t * d]
            );
        }
        if li == 0 {
            // Stage bisect: GPU driver stage buffers vs the CPU twin.
            // (The scan alone is exact — the inputs are the suspects.)
            let gc0 = match &cap.layers[0].sub {
                riir_infer_core::lens::stack::SubCapture::Gdn(g) => g,
                _ => panic!("gdn"),
            };
            let gw0 = match &sw.layers[0].sub {
                riir_infer_core::lens::stack::SubWeights::Gdn(g) => *g,
                _ => panic!("gdn"),
            };
            let gdn0 = geo.gdn_geo();
            let (n_k, n_v, key, val) = (gdn0.n_k_heads, gdn0.n_v_heads, gdn0.key_dim, gdn0.val_dim);
            let conv_dim = gdn0.conv_dim();
            let q_dim = gdn0.q_dim();
            let v_dim = gdn0.v_dim();
            // CPU twin stage values (batch row 1 = the probe row).
            let probe_row = 0usize; // row 0 = the diag-applied input (the diverging posture)
            let row_off = probe_row * td;
            let mut cpu_dz = vec![0.0f32; T * v_dim];
            let mut cpu_da = vec![0.0f32; T * n_v];
            let mut cpu_dqkv = vec![0.0f32; T * conv_dim];
            let mut cpu_dq = vec![0.0f32; T * q_dim];
            let mut cpu_dk = vec![0.0f32; T * q_dim];
            let mut cpu_dvv = vec![0.0f32; T * v_dim];
            let mut cpu_dbeta = vec![0.0f32; T * n_v];
            let mut cpu_ddecay = vec![0.0f32; T * n_v];
            for t in 0..T {
                let x_t = &cpu_in[row_off + t * d..row_off + (t + 1) * d];
                let mut z = vec![0.0f32; v_dim];
                matmul(&mut z, gw0.w_z, x_t, v_dim, d);
                cpu_dz[t * v_dim..(t + 1) * v_dim].copy_from_slice(&z);
                let mut a = vec![0.0f32; n_v];
                matmul(&mut a, gw0.w_a, x_t, n_v, d);
                cpu_da[t * n_v..(t + 1) * n_v].copy_from_slice(&a);
            }
            {
                use riir_infer_core::lens::ops::conv1d_causal_jvp;
                // Recompute dqkv_raw exactly like the driver: GEMM on the
                // diag-applied input.
                let mut dqkv_raw = vec![0.0f32; T * conv_dim];
                for t in 0..T {
                    let mut y = vec![0.0f32; conv_dim];
                    matmul(
                        &mut y,
                        gw0.w_qkv,
                        &cpu_in[row_off + t * d..row_off + (t + 1) * d],
                        conv_dim,
                        d,
                    );
                    dqkv_raw[t * conv_dim..(t + 1) * conv_dim].copy_from_slice(&y);
                }
                let mut dconv = vec![0.0f32; T * conv_dim];
                conv1d_causal_jvp(
                    gw0.conv_w,
                    gdn0.kernel_size,
                    &dqkv_raw,
                    conv_dim,
                    &mut dconv,
                );
                for i in 0..T * conv_dim {
                    cpu_dqkv[i] = gc0.conv_silu_grad[i] * dconv[i];
                }
                for t in 0..T {
                    cpu_dq[t * q_dim..(t + 1) * q_dim]
                        .copy_from_slice(&cpu_dqkv[t * conv_dim..t * conv_dim + q_dim]);
                    cpu_dk[t * q_dim..(t + 1) * q_dim]
                        .copy_from_slice(&cpu_dqkv[t * conv_dim + q_dim..t * conv_dim + 2 * q_dim]);
                    cpu_dvv[t * v_dim..(t + 1) * v_dim]
                        .copy_from_slice(&cpu_dqkv[t * conv_dim + 2 * q_dim..(t + 1) * conv_dim]);
                    cpu_dbeta[t * n_v..(t + 1) * n_v].copy_from_slice(&{
                        let mut b = vec![0.0f32; n_v];
                        // dbeta = dbeta_coef · db; recompute db.
                        let mut dbv = vec![0.0f32; n_v];
                        matmul(
                            &mut dbv,
                            gw0.w_b,
                            &cpu_in[row_off + t * d..row_off + (t + 1) * d],
                            n_v,
                            d,
                        );
                        for h in 0..n_v {
                            b[h] = gc0.dbeta_coef[t * n_v + h] * dbv[h];
                        }
                        b
                    });
                    cpu_ddecay[t * n_v..(t + 1) * n_v].copy_from_slice(&{
                        let mut dc = vec![0.0f32; n_v];
                        matmul(
                            &mut dc,
                            gw0.w_a,
                            &cpu_in[row_off + t * d..row_off + (t + 1) * d],
                            n_v,
                            d,
                        );
                        for (hi, slot) in dc.iter_mut().enumerate() {
                            *slot *= gc0.ddecay_coef[t * n_v + hi];
                        }
                        dc
                    });
                }
            }
            use riir_infer_core::lens::gdn::{
                GdnRecurrenceConsts as Grc, gdn_recurrence_jvp as scan_jvp,
            };
            // CPU twin dqn/dkn for the comparison.
            let mut cpu_dqn = vec![0.0f32; T * n_v * key];
            let mut cpu_dkn = vec![0.0f32; T * n_v * key];
            {
                let block = n_k * key;
                let mut dq_src = vec![0.0f32; T * q_dim];
                let mut dk_src = vec![0.0f32; T * q_dim];
                for t in 0..T {
                    dq_src[t * q_dim..(t + 1) * q_dim]
                        .copy_from_slice(&cpu_dqkv[t * conv_dim..t * conv_dim + q_dim]);
                    dk_src[t * q_dim..(t + 1) * q_dim]
                        .copy_from_slice(&cpu_dqkv[t * conv_dim + q_dim..t * conv_dim + 2 * q_dim]);
                }
                for t in 0..T {
                    use riir_infer_core::lens::gdn::{L2NormConsts, expand_tiled_jvp, l2_norm_jvp};
                    let qc = L2NormConsts {
                        y: &gc0.qn[t * n_v * key..t * n_v * key + block],
                        inv: &gc0.qn_inv[t * n_k..(t + 1) * n_k],
                    };
                    l2_norm_jvp(&qc, key, &mut dq_src[t * q_dim..(t + 1) * q_dim]);
                    expand_tiled_jvp(
                        &dq_src[t * q_dim..(t + 1) * q_dim],
                        n_k,
                        key,
                        n_v / n_k,
                        &mut cpu_dqn[t * n_v * key..(t + 1) * n_v * key],
                    );
                    let kc = L2NormConsts {
                        y: &gc0.kn[t * n_v * key..t * n_v * key + block],
                        inv: &gc0.kn_inv[t * n_k..(t + 1) * n_k],
                    };
                    l2_norm_jvp(&kc, key, &mut dk_src[t * q_dim..(t + 1) * q_dim]);
                    expand_tiled_jvp(
                        &dk_src[t * q_dim..(t + 1) * q_dim],
                        n_k,
                        key,
                        n_v / n_k,
                        &mut cpu_dkn[t * n_v * key..(t + 1) * n_v * key],
                    );
                }
            }
            // CPU twin dy_pre: the scan on the verified CPU stage values.
            let mut cpu_dy_pre = vec![0.0f32; T * v_dim];
            {
                let rc = Grc {
                    t_len: T,
                    n_v_heads: n_v,
                    key_dim: key,
                    val_dim: val,
                    scale: gdn0.scale(),
                    q: &gc0.qn,
                    k: &gc0.kn,
                    v: &gc0.v,
                    beta: &gc0.beta,
                    decay: &gc0.decay,
                    states: &gc0.states,
                };
                let mut dbeta_f = vec![0.0f32; T * n_v];
                let mut ddecay_f = vec![0.0f32; T * n_v];
                for t in 0..T {
                    let mut dbv = vec![0.0f32; n_v];
                    matmul(
                        &mut dbv,
                        gw0.w_b,
                        &cpu_in[row_off + t * d..row_off + (t + 1) * d],
                        n_v,
                        d,
                    );
                    let mut dav = vec![0.0f32; n_v];
                    matmul(
                        &mut dav,
                        gw0.w_a,
                        &cpu_in[row_off + t * d..row_off + (t + 1) * d],
                        n_v,
                        d,
                    );
                    for h in 0..n_v {
                        dbeta_f[t * n_v + h] = gc0.dbeta_coef[t * n_v + h] * dbv[h];
                        ddecay_f[t * n_v + h] = gc0.ddecay_coef[t * n_v + h] * dav[h];
                    }
                }
                // dv from the conv split.
                let mut dv_f = vec![0.0f32; T * v_dim];
                for t in 0..T {
                    dv_f[t * v_dim..(t + 1) * v_dim]
                        .copy_from_slice(&cpu_dqkv[t * conv_dim + 2 * q_dim..(t + 1) * conv_dim]);
                }
                let mut ds_f = vec![0.0f32; n_v * val * key];
                let mut spre_f = vec![0.0f32; n_v * val * key];
                scan_jvp(
                    &rc,
                    &cpu_dqn,
                    &cpu_dkn,
                    &dv_f,
                    &dbeta_f,
                    &ddecay_f,
                    &mut ds_f,
                    &mut spre_f,
                    &mut cpu_dy_pre,
                );
            }
            let t_h4 = create_f32(&client, &cpu_in);
            let stage_defs: Vec<(usize, &str, usize, &Vec<f32>)> = vec![
                (0, "dz", v_dim, &cpu_dz),
                (1, "da", n_v, &cpu_da),
                (2, "dqkv", conv_dim, &cpu_dqkv),
                (3, "dq", q_dim, &cpu_dq),
                (4, "dk", q_dim, &cpu_dk),
                (5, "dv", v_dim, &cpu_dvv),
                (6, "dbeta", n_v, &cpu_dbeta),
                (7, "ddecay", n_v, &cpu_ddecay),
                (8, "dqn", n_v * key, &cpu_dqn),
                (9, "dkn", n_v * key, &cpu_dkn),
                (10, "dy_pre", v_dim, &cpu_dy_pre),
            ];
            for (si, name, width, cpu) in stage_defs {
                if cpu.is_empty() && si < 8 {
                    continue;
                }
                let got = unsafe { debug_gdn_stage(&client, &stack, &doc, 0, &t_h4, si) }
                    .expect("gdn stage readback");
                let row = &got[probe_row * T * width..(probe_row + 1) * T * width];
                let (dd, at) = worst_diff(cpu, row);
                println!("  gdn stage {si} ({name}): worst |cpu-gpu| = {dd:.6} at {at}");
                if si == 10 {
                    // Stages 11-13: ssm qk_norm → gate half rule → w_out.
                    let mut dnormed = vec![0.0f32; T * v_dim];
                    for t in 0..T {
                        use riir_infer_core::lens::attn::{QkNormConsts, qk_norm_true_jvp};
                        let qkc = QkNormConsts {
                            x: &gc0.y_pre[t * v_dim..(t + 1) * v_dim],
                            r: &gc0.norm_r[t * n_v..(t + 1) * n_v],
                        };
                        qk_norm_true_jvp(
                            &qkc,
                            gw0.linear_norm,
                            val,
                            &cpu_dy_pre[t * v_dim..(t + 1) * v_dim],
                            &mut dnormed[t * v_dim..(t + 1) * v_dim],
                        );
                    }
                    // The GPU dy_pre ROW — reuse `row` (stage 10's readback).
                    let gpu_dn_h = create_f32(&client, &{
                        let mut full = vec![0.0f32; BATCH * T * v_dim];
                        full[probe_row * T * v_dim..(probe_row + 1) * T * v_dim]
                            .copy_from_slice(&cpu_dy_pre);
                        full
                    });
                    let dn_h = client.empty(BATCH * T * v_dim * 4);
                    unsafe {
                        LensQkNormCL::launch(
                            &client,
                            create_f32(&client, &gc0.y_pre),
                            create_f32(&client, &gc0.norm_r),
                            create_f32(&client, gw0.linear_norm),
                            gpu_dn_h.clone(),
                            dn_h.clone(),
                            BATCH,
                            T,
                            n_v,
                            val,
                        )
                    };
                    let gpu_dn = read_f32(&client, dn_h).expect("dn read");
                    let gpu_dn_row = &gpu_dn[probe_row * T * v_dim..(probe_row + 1) * T * v_dim];
                    let (ddn, atn) = worst_diff(&dnormed, gpu_dn_row);
                    println!("  gdn stage 11 (ssm qk_norm on GPU dy_pre): worst {ddn:.6} at {atn}");
                    let _ = atn;
                    // Gate half rule on the EXACT dnormed.
                    let mut cpu_dout = vec![0.0f32; T * v_dim];
                    for t in 0..T {
                        use riir_infer_core::lens::ops::{MlpConsts, half_rule_jvp};
                        let mc = MlpConsts {
                            s: &gc0.gate_s[t * v_dim..(t + 1) * v_dim],
                            u: &gc0.normed[t * v_dim..(t + 1) * v_dim],
                            sp: &gc0.gate_sp[t * v_dim..(t + 1) * v_dim],
                        };
                        let mut dz_row = vec![0.0f32; v_dim];
                        matmul(
                            &mut dz_row,
                            gw0.w_z,
                            &cpu_in[row_off + t * d..row_off + (t + 1) * d],
                            v_dim,
                            d,
                        );
                        half_rule_jvp(
                            &mc,
                            &dz_row,
                            &dnormed[t * v_dim..(t + 1) * v_dim],
                            &mut cpu_dout[t * v_dim..(t + 1) * v_dim],
                        );
                    }
                    // CPU w_out on the exact dout.
                    let mut cpu_t2 = vec![0.0f32; T * d];
                    for t in 0..T {
                        let mut y = vec![0.0f32; d];
                        matmul(
                            &mut y,
                            gw0.w_out,
                            &cpu_dout[t * v_dim..(t + 1) * v_dim],
                            d,
                            v_dim,
                        );
                        cpu_t2[t * d..(t + 1) * d].copy_from_slice(&y);
                    }
                    // GPU: the sublayer probe output (t2) from the driver.
                    let t_h5 = create_f32(&client, &cpu_in);
                    let gpu_sub =
                        unsafe { debug_sublayer_readback(&client, &stack, &doc, 0, &t_h5) }
                            .expect("sub");
                    let gpu_sub_row = &gpu_sub[probe_row * td..(probe_row + 1) * td];
                    let (dt2, att2) = worst_diff(&cpu_t2, gpu_sub_row);
                    println!(
                        "  gdn stage 13 (t2 via driver sub): worst {dt2:.6} at {att2} (cpu {:+.4} gpu {:+.4})",
                        cpu_t2[att2], gpu_sub_row[att2]
                    );
                }
            }

            // Isolated scan probe: the REAL layer-0 constants, dense
            // random tangents, t_len swept 1..=T — the accumulation curve.
            use riir_infer_core::lens::gdn::{GdnRecurrenceConsts, gdn_recurrence_jvp};
            let gc0 = match &cap.layers[0].sub {
                riir_infer_core::lens::stack::SubCapture::Gdn(g) => g,
                _ => panic!("gdn"),
            };
            let _gw0 = match &sw.layers[0].sub {
                riir_infer_core::lens::stack::SubWeights::Gdn(g) => *g,
                _ => panic!("gdn"),
            };
            let gdn0 = geo.gdn_geo();
            let (n_v, key, val) = (gdn0.n_v_heads, gdn0.key_dim, gdn0.val_dim);
            let scale = gdn0.scale();
            // Dense random per-position tangents (the scan's direct inputs).
            let mut rng2 = Lcg(0xCAFE);
            let dqn = rng2.vec(T * n_v * key);
            let dkn = rng2.vec(T * n_v * key);
            let dvv = rng2.vec(T * n_v * val);
            let dbeta = rng2.vec(T * n_v);
            let ddecay = rng2.vec(T * n_v);
            for tl in 1..=T {
                let rc = GdnRecurrenceConsts {
                    t_len: tl,
                    n_v_heads: n_v,
                    key_dim: key,
                    val_dim: val,
                    scale,
                    q: &gc0.qn,
                    k: &gc0.kn,
                    v: &gc0.v,
                    beta: &gc0.beta,
                    decay: &gc0.decay,
                    states: &gc0.states,
                };
                let mut want = vec![0.0f32; tl * n_v * val];
                let mut ds = vec![0.0f32; n_v * val * key];
                let mut spre = vec![0.0f32; n_v * val * key];
                gdn_recurrence_jvp(
                    &rc,
                    &dqn[..tl * n_v * key],
                    &dkn[..tl * n_v * key],
                    &dvv[..tl * n_v * val],
                    &dbeta[..tl * n_v],
                    &ddecay[..tl * n_v],
                    &mut ds,
                    &mut spre,
                    &mut want,
                );
                let ds_h = client.empty(n_v * val * key * 4);
                let spre_h = client.empty(n_v * val * key * 4);
                let dy_h = client.empty(tl * n_v * val * 4);
                unsafe {
                    LensGdnScanCL::launch(
                        &client,
                        create_f32(&client, &gc0.qn),
                        create_f32(&client, &gc0.kn),
                        create_f32(&client, &gc0.v),
                        create_f32(&client, &gc0.beta),
                        create_f32(&client, &gc0.decay),
                        create_f32(&client, &dbeta),
                        create_f32(&client, &ddecay),
                        create_f32(&client, &dqn),
                        create_f32(&client, &dkn),
                        create_f32(&client, &dvv),
                        create_f32(&client, &gc0.states),
                        ds_h,
                        spre_h,
                        dy_h.clone(),
                        1,
                        tl,
                        n_v,
                        key,
                        val,
                        scale,
                    )
                };
                let got = read_f32(&client, dy_h).expect("scan read");
                let rel_scan = worst_rel(&want, &got);
                println!(
                    "  scan t_len={tl}: rel {rel_scan:.3e} | last-dy cpu={:+.4} gpu={:+.4}",
                    want[(tl - 1) * n_v * val],
                    got[(tl - 1) * n_v * val]
                );
            }
        }
    }
}

/// The parity composite's [D,A] Mid(0) path, stage by stage: seed dv,
/// run mlp(0) then layer(1) on the GPU driver and compare each stage's
/// working tangent against the CPU composition (public ops).
#[test]
fn bisect_da_mid0_stages() {
    let Ok(ctx) = CubeCLContext::new() else {
        eprintln!("SKIP LOUD: no GPU device");
        return;
    };
    let client = ctx.client();
    let (config, weights) = tiny_model(0x12D2, vec![DeltaNet, Attention]);
    let rope_freq = RopeFreqTable::new(config.rope_theta, config.head_dim);
    let sw = StackWeights::from_model(&weights, &config, rope_freq.as_slice(), T);
    let geo = sw.geo.clone();
    let d = geo.n_embd;
    let hidden = geo.mlp_hidden;
    let mut cap = StackCapture::new(&geo);
    stack_forward(&TOKENS, &sw, &mut cap);

    let batch = d;
    let mut dv = vec![0.0f32; batch * T * d];
    for k in 0..d {
        for t in 2..4 {
            dv[(k * T + t) * d + k] = 1.0;
        }
    }
    let stack = LensBatchStack::new(&client, &sw, batch);
    let doc = LensDocCapture::upload(&client, &cap);
    let dv_h = create_f32(&client, &dv);

    // GPU stage A: mlp_step(0) on a copy of dv row 1.
    // (debug mlp: run via layer-less composition — reuse debug_attn? No —
    // add the mlp readback through the existing debug fns is not available;
    // compare the FULL suffix stages instead: t after mlp(0) is not
    // exposed, so compare after layer_step(1) — the CPU side composes
    // mlp(0) + layer(1) and we diff the end-to-end driver path.)
    let t_h = create_f32(&client, &dv);
    // GPU stage A: mlp(0) on the working copy, read back.
    let gpu_mlp0 =
        unsafe { riir_infer_gpu::lens_batch::debug_mlp_readback(&client, &stack, &doc, 0, &t_h) }
            .expect("mlp0 readback");
    let gpu_mlp0_row = &gpu_mlp0[T * d..(1 + 1) * T * d];
    // GPU stage B: layer(1) — debug_layer_step_readback on the SAME handle
    // (the working t carries mlp(0)'s output).
    let gpu_l1 = unsafe {
        riir_infer_gpu::lens_batch::debug_layer_step_readback(&client, &stack, &doc, 1, &t_h)
    }
    .expect("layer1 readback");
    let gpu_l1_row = &gpu_l1[T * d..(1 + 1) * T * d];
    // GPU stage C: the full suffix (for the end-to-end column).
    let got = unsafe {
        riir_infer_gpu::lens_batch::suffix_pass(
            &client,
            &stack,
            &doc,
            &dv_h,
            FitPoint::Mid(0),
            0,
            T,
        )
    }
    .expect("suffix");
    let gpu_col = &got[d..(1 + 1) * d];

    // CPU: mlp(0) + layer(1) + final diag on dir 1.
    let mut t = vec![0.0f32; T * d];
    for tt in 2..4 {
        t[tt * d + 1] = 1.0;
    }
    // mlp(0) (stack_suffix_jvp's Mid arm).
    {
        let lc = &cap.layers[0];
        let lw = &sw.layers[0];
        let mut t2 = vec![0.0f32; T * d];
        let mut dg = vec![0.0f32; T * hidden];
        let mut du = vec![0.0f32; T * hidden];
        let mut dact = vec![0.0f32; hidden];
        let mut dmlp = vec![0.0f32; T * d];
        for tt in 0..T {
            t2[tt * d..(tt + 1) * d].copy_from_slice(&t[tt * d..(tt + 1) * d]);
            diag_apply(
                &lc.post_scale[tt * d..(tt + 1) * d],
                &mut t2[tt * d..(tt + 1) * d],
            );
            matmul(
                &mut dg[tt * hidden..(tt + 1) * hidden],
                lw.w_gate,
                &t2[tt * d..(tt + 1) * d],
                hidden,
                d,
            );
            matmul(
                &mut du[tt * hidden..(tt + 1) * hidden],
                lw.w_up,
                &t2[tt * d..(tt + 1) * d],
                hidden,
                d,
            );
        }
        for tt in 0..T {
            use riir_infer_core::lens::ops::MlpConsts;
            let mc = MlpConsts {
                s: &lc.mlp_s[tt * hidden..(tt + 1) * hidden],
                u: &lc.mlp_u[tt * hidden..(tt + 1) * hidden],
                sp: &lc.mlp_sp[tt * hidden..(tt + 1) * hidden],
            };
            mlp_half_rule_jvp(
                &mc,
                &dg[tt * hidden..(tt + 1) * hidden],
                &du[tt * hidden..(tt + 1) * hidden],
                lw.w_down,
                d,
                &mut dact,
                &mut dmlp[tt * d..(tt + 1) * d],
            );
        }
        for i in 0..T * d {
            t[i] += dmlp[i];
        }
    }
    // layer(1): input diag → attn jvp → residual → mlp → residual.
    {
        let lc = &cap.layers[1];
        let lw = &sw.layers[1];
        for tt in 0..T {
            diag_apply(
                &lc.input_scale[tt * d..(tt + 1) * d],
                &mut t[tt * d..(tt + 1) * d],
            );
        }
        {
            use riir_infer_core::lens::attn::{AttnLayerConsts, AttnLayerScratch, attn_layer_jvp};
            let aw = match &lw.sub {
                riir_infer_core::lens::stack::SubWeights::Attn(aw) => *aw,
                _ => panic!("attn"),
            };
            let ac = match &lc.sub {
                riir_infer_core::lens::stack::SubCapture::Attn(ac) => ac,
                _ => panic!("attn cap"),
            };
            let consts = AttnLayerConsts {
                w: aw,
                geo: geo.attn_geo(),
                cap: ac,
            };
            let mut asc = AttnLayerScratch::new(&geo.attn_geo());
            let mut t2 = vec![0.0f32; T * d];
            attn_layer_jvp(&consts, &t, &mut asc, &mut t2);
            for i in 0..T * d {
                t[i] += t2[i];
            }
        }
        let mut t2 = vec![0.0f32; T * d];
        let mut dg = vec![0.0f32; T * hidden];
        let mut du = vec![0.0f32; T * hidden];
        let mut dact = vec![0.0f32; hidden];
        let mut dmlp = vec![0.0f32; T * d];
        for tt in 0..T {
            t2[tt * d..(tt + 1) * d].copy_from_slice(&t[tt * d..(tt + 1) * d]);
            diag_apply(
                &lc.post_scale[tt * d..(tt + 1) * d],
                &mut t2[tt * d..(tt + 1) * d],
            );
            matmul(
                &mut dg[tt * hidden..(tt + 1) * hidden],
                lw.w_gate,
                &t2[tt * d..(tt + 1) * d],
                hidden,
                d,
            );
            matmul(
                &mut du[tt * hidden..(tt + 1) * hidden],
                lw.w_up,
                &t2[tt * d..(tt + 1) * d],
                hidden,
                d,
            );
        }
        for tt in 0..T {
            use riir_infer_core::lens::ops::MlpConsts;
            let mc = MlpConsts {
                s: &lc.mlp_s[tt * hidden..(tt + 1) * hidden],
                u: &lc.mlp_u[tt * hidden..(tt + 1) * hidden],
                sp: &lc.mlp_sp[tt * hidden..(tt + 1) * hidden],
            };
            mlp_half_rule_jvp(
                &mc,
                &dg[tt * hidden..(tt + 1) * hidden],
                &du[tt * hidden..(tt + 1) * hidden],
                lw.w_down,
                d,
                &mut dact,
                &mut dmlp[tt * d..(tt + 1) * d],
            );
        }
        for i in 0..T * d {
            t[i] += dmlp[i];
        }
    }
    // Final norm diag + col-sum over all rows.
    for tt in 0..T {
        diag_apply(
            &cap.final_scale[tt * d..(tt + 1) * d],
            &mut t[tt * d..(tt + 1) * d],
        );
    }
    let mut cpu_col = vec![0.0f32; d];
    for tt in 0..T {
        for (j, v) in t[tt * d..(tt + 1) * d].iter().enumerate() {
            cpu_col[j] += v;
        }
    }
    let (dd, at) = worst_diff(&cpu_col, gpu_col);
    println!(
        "DA Mid(0) dir 1 end-to-end: worst |cpu-gpu| = {dd:.6} at dim {at} (cpu {:+.6} gpu {:+.6})",
        cpu_col[at], gpu_col[at]
    );

    // Stage diffs against the CPU composition.
    {
        // CPU: recompute the post-mlp(0) tangent (t after the first block).
        let mut t_a = vec![0.0f32; T * d];
        for tt in 2..4 {
            t_a[tt * d + 1] = 1.0;
        }
        {
            let lc = &cap.layers[0];
            let lw = &sw.layers[0];
            let mut t2 = vec![0.0f32; T * d];
            let mut dg = vec![0.0f32; T * hidden];
            let mut du = vec![0.0f32; T * hidden];
            let mut dact = vec![0.0f32; hidden];
            let mut dmlp = vec![0.0f32; T * d];
            for tt in 0..T {
                t2[tt * d..(tt + 1) * d].copy_from_slice(&t_a[tt * d..(tt + 1) * d]);
                diag_apply(
                    &lc.post_scale[tt * d..(tt + 1) * d],
                    &mut t2[tt * d..(tt + 1) * d],
                );
                matmul(
                    &mut dg[tt * hidden..(tt + 1) * hidden],
                    lw.w_gate,
                    &t2[tt * d..(tt + 1) * d],
                    hidden,
                    d,
                );
                matmul(
                    &mut du[tt * hidden..(tt + 1) * hidden],
                    lw.w_up,
                    &t2[tt * d..(tt + 1) * d],
                    hidden,
                    d,
                );
            }
            for tt in 0..T {
                use riir_infer_core::lens::ops::MlpConsts;
                let mc = MlpConsts {
                    s: &lc.mlp_s[tt * hidden..(tt + 1) * hidden],
                    u: &lc.mlp_u[tt * hidden..(tt + 1) * hidden],
                    sp: &lc.mlp_sp[tt * hidden..(tt + 1) * hidden],
                };
                mlp_half_rule_jvp(
                    &mc,
                    &dg[tt * hidden..(tt + 1) * hidden],
                    &du[tt * hidden..(tt + 1) * hidden],
                    lw.w_down,
                    d,
                    &mut dact,
                    &mut dmlp[tt * d..(tt + 1) * d],
                );
            }
            for i in 0..T * d {
                t_a[i] += dmlp[i];
            }
        }
        let (da, ata) = worst_diff(&t_a, gpu_mlp0_row);
        println!("stage A mlp(0): worst |cpu-gpu| = {da:.6} at {ata}");
        // CPU layer(1) on t_a.
        let lc = &cap.layers[1];
        let lw = &sw.layers[1];
        let mut t_b = t_a.clone();
        for tt in 0..T {
            diag_apply(
                &lc.input_scale[tt * d..(tt + 1) * d],
                &mut t_b[tt * d..(tt + 1) * d],
            );
        }
        {
            use riir_infer_core::lens::attn::{AttnLayerConsts, AttnLayerScratch, attn_layer_jvp};
            let aw = match &lw.sub {
                riir_infer_core::lens::stack::SubWeights::Attn(aw) => *aw,
                _ => panic!("attn"),
            };
            let ac = match &lc.sub {
                riir_infer_core::lens::stack::SubCapture::Attn(ac) => ac,
                _ => panic!("attn cap"),
            };
            let consts = AttnLayerConsts {
                w: aw,
                geo: geo.attn_geo(),
                cap: ac,
            };
            let mut asc = AttnLayerScratch::new(&geo.attn_geo());
            let mut t2 = vec![0.0f32; T * d];
            attn_layer_jvp(&consts, &t_b, &mut asc, &mut t2);
            for i in 0..T * d {
                t_b[i] += t2[i];
            }
        }
        {
            let mut t2 = vec![0.0f32; T * d];
            let mut dg = vec![0.0f32; T * hidden];
            let mut du = vec![0.0f32; T * hidden];
            let mut dact = vec![0.0f32; hidden];
            let mut dmlp = vec![0.0f32; T * d];
            for tt in 0..T {
                t2[tt * d..(tt + 1) * d].copy_from_slice(&t_b[tt * d..(tt + 1) * d]);
                diag_apply(
                    &lc.post_scale[tt * d..(tt + 1) * d],
                    &mut t2[tt * d..(tt + 1) * d],
                );
                matmul(
                    &mut dg[tt * hidden..(tt + 1) * hidden],
                    lw.w_gate,
                    &t2[tt * d..(tt + 1) * d],
                    hidden,
                    d,
                );
                matmul(
                    &mut du[tt * hidden..(tt + 1) * hidden],
                    lw.w_up,
                    &t2[tt * d..(tt + 1) * d],
                    hidden,
                    d,
                );
            }
            for tt in 0..T {
                use riir_infer_core::lens::ops::MlpConsts;
                let mc = MlpConsts {
                    s: &lc.mlp_s[tt * hidden..(tt + 1) * hidden],
                    u: &lc.mlp_u[tt * hidden..(tt + 1) * hidden],
                    sp: &lc.mlp_sp[tt * hidden..(tt + 1) * hidden],
                };
                mlp_half_rule_jvp(
                    &mc,
                    &dg[tt * hidden..(tt + 1) * hidden],
                    &du[tt * hidden..(tt + 1) * hidden],
                    lw.w_down,
                    d,
                    &mut dact,
                    &mut dmlp[tt * d..(tt + 1) * d],
                );
            }
            for i in 0..T * d {
                t_b[i] += dmlp[i];
            }
        }
        let (db, atb) = worst_diff(&t_b, gpu_l1_row);
        println!("stage B layer(1): worst |cpu-gpu| = {db:.6} at {atb}");

        // Attn sub-bisect ON THE STAGE-B INPUT: upload t_b's PRE-ATTN state
        // (post input-diag — t_b currently has the attn output folded in;
        // recompute the post-diag state), run the GPU attn stages on it,
        // and diff dgated.
        let mut t_pre = t_a.clone();
        for tt in 0..T {
            diag_apply(
                &lc.input_scale[tt * d..(tt + 1) * d],
                &mut t_pre[tt * d..(tt + 1) * d],
            );
        }
        // CPU: attn jvp on t_pre → the expected dgated.
        {
            use riir_infer_core::lens::attn::{
                AttentionConsts, QkNormConsts, attention_jvp, qg_deinterleave, qk_norm_true_jvp,
            };
            use riir_infer_core::lens::ops::{GateConsts, gate_half_rule_jvp, rope_jvp};
            let attn_geo = geo.attn_geo();
            let (q_dim, kv_dim, qg_dim, hd) = (
                attn_geo.q_dim(),
                attn_geo.kv_dim(),
                attn_geo.qg_dim(),
                attn_geo.head_dim,
            );
            let aw = match &lw.sub {
                riir_infer_core::lens::stack::SubWeights::Attn(aw) => *aw,
                _ => panic!("attn"),
            };
            let ac = match &lc.sub {
                riir_infer_core::lens::stack::SubCapture::Attn(ac) => ac,
                _ => panic!("attn cap"),
            };
            let mut cpu_dqg = vec![0.0f32; T * qg_dim];
            let mut cpu_dq = vec![0.0f32; T * q_dim];
            let mut cpu_dgate = vec![0.0f32; T * q_dim];
            let mut cpu_dk = vec![0.0f32; T * kv_dim];
            let mut cpu_dv = vec![0.0f32; T * kv_dim];
            let mut cpu_dq2 = vec![0.0f32; T * q_dim];

            let mut cpu_dk2 = vec![0.0f32; T * kv_dim];
            let mut cpu_dgated = vec![0.0f32; T * q_dim];
            let mut ds_scratch = vec![0.0f32; T];
            for tt in 0..T {
                matmul(
                    &mut cpu_dqg[tt * qg_dim..(tt + 1) * qg_dim],
                    aw.w_qg,
                    &t_pre[tt * d..(tt + 1) * d],
                    qg_dim,
                    d,
                );
                matmul(
                    &mut cpu_dv[tt * kv_dim..(tt + 1) * kv_dim],
                    aw.w_v,
                    &t_pre[tt * d..(tt + 1) * d],
                    kv_dim,
                    d,
                );
                qg_deinterleave(
                    &cpu_dqg[tt * qg_dim..(tt + 1) * qg_dim],
                    attn_geo.n_head,
                    hd,
                    &mut cpu_dq[tt * q_dim..(tt + 1) * q_dim],
                    &mut cpu_dgate[tt * q_dim..(tt + 1) * q_dim],
                );
                matmul(
                    &mut cpu_dk[tt * kv_dim..(tt + 1) * kv_dim],
                    aw.w_k,
                    &t_pre[tt * d..(tt + 1) * d],
                    kv_dim,
                    d,
                );
                let qc = QkNormConsts {
                    x: &ac.q_pre[tt * q_dim..(tt + 1) * q_dim],
                    r: &ac.q_r[tt * attn_geo.n_head..(tt + 1) * attn_geo.n_head],
                };
                qk_norm_true_jvp(
                    &qc,
                    aw.q_norm,
                    hd,
                    &cpu_dq[tt * q_dim..(tt + 1) * q_dim],
                    &mut cpu_dq2[tt * q_dim..(tt + 1) * q_dim],
                );
                let kc = QkNormConsts {
                    x: &ac.k_pre[tt * kv_dim..(tt + 1) * kv_dim],
                    r: &ac.k_r[tt * attn_geo.n_kv..(tt + 1) * attn_geo.n_kv],
                };
                qk_norm_true_jvp(
                    &kc,
                    aw.k_norm,
                    hd,
                    &cpu_dk[tt * kv_dim..(tt + 1) * kv_dim],
                    &mut cpu_dk2[tt * kv_dim..(tt + 1) * kv_dim],
                );
            }
            let cpu_dq2_pre_rope = cpu_dq2.clone();
            for tt in 0..T {
                rope_jvp(
                    &mut cpu_dq2[tt * q_dim..(tt + 1) * q_dim],
                    &mut cpu_dk2[tt * kv_dim..(tt + 1) * kv_dim],
                    tt,
                    hd,
                    attn_geo.rotary_dim,
                    aw.rope_freq,
                );
            }
            attention_jvp(
                &AttentionConsts {
                    t_len: T,
                    n_head: attn_geo.n_head,
                    n_kv: attn_geo.n_kv,
                    head_dim: hd,
                    scale: attn_geo.scale(),
                    attn: &ac.attn,
                },
                &ac.q,
                &ac.k,
                &ac.v,
                &cpu_dq2,
                &cpu_dk2,
                &cpu_dv,
                &mut ds_scratch,
                &mut cpu_dgated,
            );
            let gc = GateConsts {
                sg: &ac.gate_sg,
                z: &ac.att,
                spg: &ac.gate_spg,
            };
            let mut cpu_dgated2 = vec![0.0f32; T * q_dim];
            gate_half_rule_jvp(&gc, &cpu_dgate, &cpu_dgated, &mut cpu_dgated2);

            // GPU: the same pipeline via debug_attn_stage on t_pre.
            let mut full = vec![0.0f32; BATCH * T * d];
            full[T * d..2 * T * d].copy_from_slice(&t_pre);
            let t_pre_h = create_f32(&client, &full);
            {
                // Mini qk_norm at BATCH=16 on the real captured values:
                // row 1 = cpu_dq, all other rows zero.
                let mut full_dq = vec![0.0f32; BATCH * T * q_dim];
                full_dq[T * q_dim..2 * T * q_dim].copy_from_slice(&cpu_dq);
                let xh = create_f32(&client, &ac.q_pre);
                let rh = create_f32(&client, &ac.q_r);
                let gh = create_f32(&client, aw.q_norm);
                let dxh = create_f32(&client, &full_dq);
                let dyh = client.empty(full_dq.len() * 4);
                unsafe {
                    LensQkNormCL::launch(
                        &client,
                        xh,
                        rh,
                        gh,
                        dxh,
                        dyh.clone(),
                        BATCH,
                        T,
                        attn_geo.n_head,
                        hd,
                    )
                };
                let got_mini = read_f32(&client, dyh).expect("mini read");
                let gpu_mini_row = &got_mini[T * q_dim..2 * T * q_dim];
                let (dm, atm) = worst_diff(&cpu_dq2, gpu_mini_row);
                println!("  MINI qk_norm batch=16 row 1: worst {dm:.6} at {atm}");
                // Hand-compute the flipped element: [50] = t=3, head 0, dim 2.
                let (ft, fh, fi) = (atm / q_dim, (atm % q_dim) / hd, atm % hd);
                let x_base = ft * q_dim + fh * hd;
                let dx_base = T * q_dim + ft * q_dim + fh * hd;
                let mut dot = 0.0f32;
                for i in 0..hd {
                    dot += ac.q_pre[x_base + i] * cpu_dq[dx_base - T * q_dim + i];
                }
                let r = ac.q_r[ft * attn_geo.n_head + fh];
                let coef = r * r * r * dot / hd as f32;
                let hand = aw.q_norm[fi]
                    * (r * cpu_dq[dx_base - T * q_dim + fi] - ac.q_pre[x_base + fi] * coef);
                println!(
                    "    [{atm}] t={ft} h={fh} i={fi}: cpu={:+.6} gpu={:+.6} hand={:+.6} | dot={dot:.6} r={r:.4}",
                    cpu_dq2[atm], gpu_mini_row[atm], hand
                );
                println!(
                    "    q_pre[x_base..+8]={:+.4?} dq_row[:8]={:+.4?}",
                    &ac.q_pre[x_base..x_base + 8],
                    &cpu_dq[ft * q_dim..ft * q_dim + 8]
                );
            }
            {
                // K-side mini: qk_norm with heads=n_kv=1 on the real values,
                // row 1 = cpu_dk (the twin's split output).
                let mut full_dk = vec![0.0f32; BATCH * T * kv_dim];
                full_dk[T * kv_dim..2 * T * kv_dim].copy_from_slice(&cpu_dk);
                let xh = create_f32(&client, &ac.k_pre);
                let rh = create_f32(&client, &ac.k_r);
                let gh = create_f32(&client, aw.k_norm);
                let dxh = create_f32(&client, &full_dk);
                let dyh = client.empty(full_dk.len() * 4);
                unsafe {
                    LensQkNormCL::launch(
                        &client,
                        xh,
                        rh,
                        gh,
                        dxh,
                        dyh.clone(),
                        BATCH,
                        T,
                        attn_geo.n_kv,
                        hd,
                    )
                };
                let got_k = read_f32(&client, dyh).expect("k read");
                let gpu_k_row = &got_k[T * kv_dim..2 * T * kv_dim];
                let mut want_k = vec![0.0f32; T * kv_dim];
                for tt in 0..T {
                    let kc = QkNormConsts {
                        x: &ac.k_pre[tt * kv_dim..(tt + 1) * kv_dim],
                        r: &ac.k_r[tt * attn_geo.n_kv..(tt + 1) * attn_geo.n_kv],
                    };
                    qk_norm_true_jvp(
                        &kc,
                        aw.k_norm,
                        hd,
                        &cpu_dk[tt * kv_dim..(tt + 1) * kv_dim],
                        &mut want_k[tt * kv_dim..(tt + 1) * kv_dim],
                    );
                }
                let (dk2d, atk) = worst_diff(&want_k, gpu_k_row);
                println!("  K mini qk_norm heads=1 batch=16: worst {dk2d:.6} at {atk}");
                // Also diff the twin's OWN dk2 against want_k (is the twin
                // the wrong side?).
                let (dtw, attw) = worst_diff(&want_k, &cpu_dk2);
                println!("  K twin cpu_dk2 vs want_k: worst {dtw:.6} at {attw}");
            }
            // K-side qk_norm expectation (pre-rope) for the stage-3 compare.
            let mut k_want = vec![0.0f32; T * kv_dim];
            for tt in 0..T {
                let kc = QkNormConsts {
                    x: &ac.k_pre[tt * kv_dim..(tt + 1) * kv_dim],
                    r: &ac.k_r[tt * attn_geo.n_kv..(tt + 1) * attn_geo.n_kv],
                };
                qk_norm_true_jvp(
                    &kc,
                    aw.k_norm,
                    hd,
                    &cpu_dk[tt * kv_dim..(tt + 1) * kv_dim],
                    &mut k_want[tt * kv_dim..(tt + 1) * kv_dim],
                );
            }
            // Attention-jvp mini cell: fresh uploads of the VERIFIED
            // stage outputs + the real capture, batch=16, compare row 1.
            {
                let mut full_dq2 = vec![0.0f32; BATCH * T * q_dim];
                full_dq2[T * q_dim..2 * T * q_dim].copy_from_slice(&cpu_dq2);
                let mut full_dk2 = vec![0.0f32; BATCH * T * kv_dim];
                full_dk2[T * kv_dim..2 * T * kv_dim].copy_from_slice(&cpu_dk2);
                let mut full_dv = vec![0.0f32; BATCH * T * kv_dim];
                full_dv[T * kv_dim..2 * T * kv_dim].copy_from_slice(&cpu_dv);
                let qh = create_f32(&client, &ac.q);
                let kh = create_f32(&client, &ac.k);
                let vh = create_f32(&client, &ac.v);
                let ah = create_f32(&client, &ac.attn);
                let dqh = create_f32(&client, &full_dq2);
                let dkh = create_f32(&client, &full_dk2);
                let dvh = create_f32(&client, &full_dv);
                let outh = client.empty(BATCH * T * q_dim * 4);
                unsafe {
                    LensAttnJvpCL::launch(
                        &client,
                        qh,
                        kh,
                        vh,
                        ah,
                        dqh,
                        dkh,
                        dvh,
                        outh.clone(),
                        BATCH,
                        T,
                        attn_geo.n_head,
                        attn_geo.n_kv,
                        hd,
                        attn_geo.scale(),
                    )
                };
                let got_j = read_f32(&client, outh).expect("jvp read");
                let gpu_j_row = &got_j[T * q_dim..2 * T * q_dim];
                let (dj, atj) = worst_diff(&cpu_dgated, gpu_j_row);
                println!("  JVP mini batch=16 row 1: worst {dj:.6} at {atj}");
                for tt in 0..T {
                    println!(
                        "    t{tt} cpu_h0={:+.4?} h1={:+.4?}",
                        &cpu_dgated[tt * q_dim..tt * q_dim + 4],
                        &cpu_dgated[tt * q_dim + 8..tt * q_dim + 12]
                    );
                    println!(
                        "    t{tt} gpu_h0={:+.4?} h1={:+.4?}",
                        &gpu_j_row[tt * q_dim..tt * q_dim + 4],
                        &gpu_j_row[tt * q_dim + 8..tt * q_dim + 12]
                    );
                }
            }
            for stage in [1, 2, 3, 4, 5, 6, 7] {
                let got = unsafe { debug_attn_stage(&client, &stack, &doc, 1, &t_pre_h, stage) }
                    .expect("stage");
                let width = match stage {
                    3 | 7 => kv_dim,
                    _ => q_dim,
                };
                let gpu_row = &got[T * width..2 * T * width];
                let cpu = match stage {
                    1 => &cpu_dq,
                    2 => &cpu_dq2_pre_rope,
                    3 => &k_want,
                    4 => &cpu_dq2,
                    5 => &cpu_dgated,
                    6 => &cpu_dgated2,
                    7 => &cpu_dk,
                    _ => &cpu_dq,
                };
                if stage == 7 {
                    let at77 = 29usize;
                    println!(
                        "    dk[29] driver={:+.6} cpu={:+.6} | cpu_dv[29]={:+.6} cpu_dq[29]={:+.6} cpu_dgate[29]={:+.6}",
                        gpu_row[at77], cpu[at77], cpu_dv[at77], cpu_dq[at77], cpu_dgate[at77]
                    );
                    let (dv_mix, _) = worst_diff(&cpu_dv, gpu_row);
                    let (dq_mix, _) = worst_diff(&cpu_dq, gpu_row);
                    let (dg_mix, _) = worst_diff(&cpu_dgate, gpu_row);
                    println!(
                        "    row-mix: vs dv {dv_mix:.3e} vs dq {dq_mix:.3e} vs dgate {dg_mix:.3e}"
                    );
                }
                let (d5, at5) = worst_diff(cpu, gpu_row);
                println!("  attn stage {stage}: worst |cpu-gpu| = {d5:.6} at {at5}");
            }
        }
    }
}

/// The [DA-D]@12d3 Mid(0) composite, stage by stage: GPU layer readbacks
/// vs the CPU composition (public ops) after each stage. The failing
/// parity configuration.
#[test]
fn bisect_dad12d3_mid0_stages() {
    let Ok(ctx) = CubeCLContext::new() else {
        eprintln!("SKIP LOUD: no GPU device");
        return;
    };
    let client = ctx.client();
    let (config, weights) = tiny_model(0x12D3, vec![DeltaNet, Attention, DeltaNet]);
    let rope_freq = RopeFreqTable::new(config.rope_theta, config.head_dim);
    let sw = StackWeights::from_model(&weights, &config, rope_freq.as_slice(), T);
    let geo = sw.geo.clone();
    let d = geo.n_embd;
    let hidden = geo.mlp_hidden;
    let mut cap = StackCapture::new(&geo);
    stack_forward(&TOKENS, &sw, &mut cap);

    let batch = d;
    let mut dv = vec![0.0f32; batch * T * d];
    for k in 0..d {
        for t in 2..4 {
            dv[(k * T + t) * d + k] = 1.0;
        }
    }
    let stack = LensBatchStack::new(&client, &sw, batch);
    let doc = LensDocCapture::upload(&client, &cap);

    // CPU composition, stage by stage, on dir 1.
    let mut t = vec![0.0f32; T * d];
    for tt in 2..4 {
        t[tt * d + 1] = 1.0;
    }
    // Stage A: mlp(0).
    {
        let lc = &cap.layers[0];
        let lw = &sw.layers[0];
        let mut t2 = vec![0.0f32; T * d];
        let mut dg = vec![0.0f32; T * hidden];
        let mut du = vec![0.0f32; T * hidden];
        let mut dact = vec![0.0f32; hidden];
        let mut dmlp = vec![0.0f32; T * d];
        for tt in 0..T {
            t2[tt * d..(tt + 1) * d].copy_from_slice(&t[tt * d..(tt + 1) * d]);
            diag_apply(
                &lc.post_scale[tt * d..(tt + 1) * d],
                &mut t2[tt * d..(tt + 1) * d],
            );
            matmul(
                &mut dg[tt * hidden..(tt + 1) * hidden],
                lw.w_gate,
                &t2[tt * d..(tt + 1) * d],
                hidden,
                d,
            );
            matmul(
                &mut du[tt * hidden..(tt + 1) * hidden],
                lw.w_up,
                &t2[tt * d..(tt + 1) * d],
                hidden,
                d,
            );
        }
        for tt in 0..T {
            use riir_infer_core::lens::ops::MlpConsts;
            let mc = MlpConsts {
                s: &lc.mlp_s[tt * hidden..(tt + 1) * hidden],
                u: &lc.mlp_u[tt * hidden..(tt + 1) * hidden],
                sp: &lc.mlp_sp[tt * hidden..(tt + 1) * hidden],
            };
            mlp_half_rule_jvp(
                &mc,
                &dg[tt * hidden..(tt + 1) * hidden],
                &du[tt * hidden..(tt + 1) * hidden],
                lw.w_down,
                d,
                &mut dact,
                &mut dmlp[tt * d..(tt + 1) * d],
            );
        }
        for i in 0..T * d {
            t[i] += dmlp[i];
        }
    }
    let cpu_post_mlp0 = t.clone();
    // Stage B: layer(1) — attn.
    {
        let lc = &cap.layers[1];
        let lw = &sw.layers[1];
        for tt in 0..T {
            diag_apply(
                &lc.input_scale[tt * d..(tt + 1) * d],
                &mut t[tt * d..(tt + 1) * d],
            );
        }
        {
            use riir_infer_core::lens::attn::{AttnLayerConsts, AttnLayerScratch, attn_layer_jvp};
            let aw = match &lw.sub {
                riir_infer_core::lens::stack::SubWeights::Attn(aw) => *aw,
                _ => panic!("attn"),
            };
            let ac = match &lc.sub {
                riir_infer_core::lens::stack::SubCapture::Attn(ac) => ac,
                _ => panic!("attn cap"),
            };
            let consts = AttnLayerConsts {
                w: aw,
                geo: geo.attn_geo(),
                cap: ac,
            };
            let mut asc = AttnLayerScratch::new(&geo.attn_geo());
            let mut t2 = vec![0.0f32; T * d];
            attn_layer_jvp(&consts, &t, &mut asc, &mut t2);
            for i in 0..T * d {
                t[i] += t2[i];
            }
        }
        {
            let mut t2 = vec![0.0f32; T * d];
            let mut dg = vec![0.0f32; T * hidden];
            let mut du = vec![0.0f32; T * hidden];
            let mut dact = vec![0.0f32; hidden];
            let mut dmlp = vec![0.0f32; T * d];
            for tt in 0..T {
                t2[tt * d..(tt + 1) * d].copy_from_slice(&t[tt * d..(tt + 1) * d]);
                diag_apply(
                    &lc.post_scale[tt * d..(tt + 1) * d],
                    &mut t2[tt * d..(tt + 1) * d],
                );
                matmul(
                    &mut dg[tt * hidden..(tt + 1) * hidden],
                    lw.w_gate,
                    &t2[tt * d..(tt + 1) * d],
                    hidden,
                    d,
                );
                matmul(
                    &mut du[tt * hidden..(tt + 1) * hidden],
                    lw.w_up,
                    &t2[tt * d..(tt + 1) * d],
                    hidden,
                    d,
                );
            }
            for tt in 0..T {
                use riir_infer_core::lens::ops::MlpConsts;
                let mc = MlpConsts {
                    s: &lc.mlp_s[tt * hidden..(tt + 1) * hidden],
                    u: &lc.mlp_u[tt * hidden..(tt + 1) * hidden],
                    sp: &lc.mlp_sp[tt * hidden..(tt + 1) * hidden],
                };
                mlp_half_rule_jvp(
                    &mc,
                    &dg[tt * hidden..(tt + 1) * hidden],
                    &du[tt * hidden..(tt + 1) * hidden],
                    lw.w_down,
                    d,
                    &mut dact,
                    &mut dmlp[tt * d..(tt + 1) * d],
                );
            }
            for i in 0..T * d {
                t[i] += dmlp[i];
            }
        }
    }
    let cpu_post_l1 = t.clone();
    // Stage C: layer(2) — GDN.
    {
        let lc = &cap.layers[2];
        let lw = &sw.layers[2];
        for tt in 0..T {
            diag_apply(
                &lc.input_scale[tt * d..(tt + 1) * d],
                &mut t[tt * d..(tt + 1) * d],
            );
        }
        {
            use riir_infer_core::lens::gdn::{GdnLayerConsts, gdn_layer_jvp};
            let gw = match &lw.sub {
                riir_infer_core::lens::stack::SubWeights::Gdn(gw) => *gw,
                _ => panic!("gdn"),
            };
            let gc = match &lc.sub {
                riir_infer_core::lens::stack::SubCapture::Gdn(gc) => gc,
                _ => panic!("gdn cap"),
            };
            let consts = GdnLayerConsts {
                w: gw,
                geo: geo.gdn_geo(),
                cap: gc,
            };
            use riir_infer_core::lens::gdn::GdnLayerScratch;
            let mut gsc = GdnLayerScratch::new(&geo.gdn_geo());
            let mut t2 = vec![0.0f32; T * d];
            gdn_layer_jvp(&consts, &t, &mut gsc, &mut t2);
            for i in 0..T * d {
                t[i] += t2[i];
            }
        }
        {
            let mut t2 = vec![0.0f32; T * d];
            let mut dg = vec![0.0f32; T * hidden];
            let mut du = vec![0.0f32; T * hidden];
            let mut dact = vec![0.0f32; hidden];
            let mut dmlp = vec![0.0f32; T * d];
            for tt in 0..T {
                t2[tt * d..(tt + 1) * d].copy_from_slice(&t[tt * d..(tt + 1) * d]);
                diag_apply(
                    &lc.post_scale[tt * d..(tt + 1) * d],
                    &mut t2[tt * d..(tt + 1) * d],
                );
                matmul(
                    &mut dg[tt * hidden..(tt + 1) * hidden],
                    lw.w_gate,
                    &t2[tt * d..(tt + 1) * d],
                    hidden,
                    d,
                );
                matmul(
                    &mut du[tt * hidden..(tt + 1) * hidden],
                    lw.w_up,
                    &t2[tt * d..(tt + 1) * d],
                    hidden,
                    d,
                );
            }
            for tt in 0..T {
                use riir_infer_core::lens::ops::MlpConsts;
                let mc = MlpConsts {
                    s: &lc.mlp_s[tt * hidden..(tt + 1) * hidden],
                    u: &lc.mlp_u[tt * hidden..(tt + 1) * hidden],
                    sp: &lc.mlp_sp[tt * hidden..(tt + 1) * hidden],
                };
                mlp_half_rule_jvp(
                    &mc,
                    &dg[tt * hidden..(tt + 1) * hidden],
                    &du[tt * hidden..(tt + 1) * hidden],
                    lw.w_down,
                    d,
                    &mut dact,
                    &mut dmlp[tt * d..(tt + 1) * d],
                );
            }
            for i in 0..T * d {
                t[i] += dmlp[i];
            }
        }
    }
    for tt in 0..T {
        diag_apply(
            &cap.final_scale[tt * d..(tt + 1) * d],
            &mut t[tt * d..(tt + 1) * d],
        );
    }
    let mut cpu_col = vec![0.0f32; d];
    for tt in 0..T {
        for (j, v) in t[tt * d..(tt + 1) * d].iter().enumerate() {
            cpu_col[j] += v;
        }
    }

    // GPU: the driver's stages, with readbacks between.
    let t_h = create_f32(&client, &dv);
    let gpu_a =
        unsafe { riir_infer_gpu::lens_batch::debug_mlp_readback(&client, &stack, &doc, 0, &t_h) }
            .expect("a");
    let gpu_b = unsafe {
        riir_infer_gpu::lens_batch::debug_layer_step_readback(&client, &stack, &doc, 1, &t_h)
    }
    .expect("b");
    let gpu_c = unsafe {
        riir_infer_gpu::lens_batch::debug_layer_step_readback(&client, &stack, &doc, 2, &t_h)
    }
    .expect("c");
    let gpu_a_row = &gpu_a[T * d..2 * T * d];
    let gpu_b_row = &gpu_b[T * d..2 * T * d];
    let gpu_c_row = &gpu_c[T * d..2 * T * d];
    let (da, _) = worst_diff(&cpu_post_mlp0, gpu_a_row);
    let (db, _) = worst_diff(&cpu_post_l1, gpu_b_row);
    let (dc, _) = worst_diff(&t, gpu_c_row);
    println!(
        "DAD@12d3 Mid(0) dir 1 stage diffs: mlp0 {da:.3e} | layer1 {db:.3e} | layer2 {dc:.3e}"
    );
    // CONDITIONING PROBE: perturb the injected tangent by 1 ulp-scale and
    // measure how much the CPU layer-1 output moves. If the composition is
    // ill-conditioned, the GPU's 2.9e-2 divergence IS the fp noise floor.
    {
        let eps = 1.0e-4; // ~1000 ulps of a unit-scale value
        let mut t_pert = vec![0.0f32; T * d];
        for tt in 2..4 {
            t_pert[tt * d + 1] = 1.0 + eps;
        }
        // Propagate mlp(0) + layer(1) on the perturbed input (CPU only).
        let mut tp = t_pert;
        {
            let lc = &cap.layers[0];
            let lw = &sw.layers[0];
            let mut t2 = vec![0.0f32; T * d];
            let mut dg = vec![0.0f32; T * hidden];
            let mut du = vec![0.0f32; T * hidden];
            let mut dact = vec![0.0f32; hidden];
            let mut dmlp = vec![0.0f32; T * d];
            for tt in 0..T {
                t2[tt * d..(tt + 1) * d].copy_from_slice(&tp[tt * d..(tt + 1) * d]);
                diag_apply(
                    &lc.post_scale[tt * d..(tt + 1) * d],
                    &mut t2[tt * d..(tt + 1) * d],
                );
                matmul(
                    &mut dg[tt * hidden..(tt + 1) * hidden],
                    lw.w_gate,
                    &t2[tt * d..(tt + 1) * d],
                    hidden,
                    d,
                );
                matmul(
                    &mut du[tt * hidden..(tt + 1) * hidden],
                    lw.w_up,
                    &t2[tt * d..(tt + 1) * d],
                    hidden,
                    d,
                );
            }
            for tt in 0..T {
                use riir_infer_core::lens::ops::MlpConsts;
                let mc = MlpConsts {
                    s: &lc.mlp_s[tt * hidden..(tt + 1) * hidden],
                    u: &lc.mlp_u[tt * hidden..(tt + 1) * hidden],
                    sp: &lc.mlp_sp[tt * hidden..(tt + 1) * hidden],
                };
                mlp_half_rule_jvp(
                    &mc,
                    &dg[tt * hidden..(tt + 1) * hidden],
                    &du[tt * hidden..(tt + 1) * hidden],
                    lw.w_down,
                    d,
                    &mut dact,
                    &mut dmlp[tt * d..(tt + 1) * d],
                );
            }
            for i in 0..T * d {
                tp[i] += dmlp[i];
            }
        }
        let t_post_mlp0_pert = tp.clone();
        {
            let lc = &cap.layers[1];
            let lw = &sw.layers[1];
            for tt in 0..T {
                diag_apply(
                    &lc.input_scale[tt * d..(tt + 1) * d],
                    &mut tp[tt * d..(tt + 1) * d],
                );
            }
            {
                use riir_infer_core::lens::attn::{
                    AttnLayerConsts, AttnLayerScratch, attn_layer_jvp,
                };
                let aw = match &lw.sub {
                    riir_infer_core::lens::stack::SubWeights::Attn(aw) => *aw,
                    _ => panic!("attn"),
                };
                let ac = match &lc.sub {
                    riir_infer_core::lens::stack::SubCapture::Attn(ac) => ac,
                    _ => panic!("attn cap"),
                };
                let consts = AttnLayerConsts {
                    w: aw,
                    geo: geo.attn_geo(),
                    cap: ac,
                };
                let mut asc = AttnLayerScratch::new(&geo.attn_geo());
                let mut t2 = vec![0.0f32; T * d];
                attn_layer_jvp(&consts, &tp, &mut asc, &mut t2);
                for i in 0..T * d {
                    tp[i] += t2[i];
                }
            }
            {
                let mut t2 = vec![0.0f32; T * d];
                let mut dg = vec![0.0f32; T * hidden];
                let mut du = vec![0.0f32; T * hidden];
                let mut dact = vec![0.0f32; hidden];
                let mut dmlp = vec![0.0f32; T * d];
                for tt in 0..T {
                    t2[tt * d..(tt + 1) * d].copy_from_slice(&tp[tt * d..(tt + 1) * d]);
                    diag_apply(
                        &lc.post_scale[tt * d..(tt + 1) * d],
                        &mut t2[tt * d..(tt + 1) * d],
                    );
                    matmul(
                        &mut dg[tt * hidden..(tt + 1) * hidden],
                        lw.w_gate,
                        &t2[tt * d..(tt + 1) * d],
                        hidden,
                        d,
                    );
                    matmul(
                        &mut du[tt * hidden..(tt + 1) * hidden],
                        lw.w_up,
                        &t2[tt * d..(tt + 1) * d],
                        hidden,
                        d,
                    );
                }
                for tt in 0..T {
                    use riir_infer_core::lens::ops::MlpConsts;
                    let mc = MlpConsts {
                        s: &lc.mlp_s[tt * hidden..(tt + 1) * hidden],
                        u: &lc.mlp_u[tt * hidden..(tt + 1) * hidden],
                        sp: &lc.mlp_sp[tt * hidden..(tt + 1) * hidden],
                    };
                    mlp_half_rule_jvp(
                        &mc,
                        &dg[tt * hidden..(tt + 1) * hidden],
                        &du[tt * hidden..(tt + 1) * hidden],
                        lw.w_down,
                        d,
                        &mut dact,
                        &mut dmlp[tt * d..(tt + 1) * d],
                    );
                }
                for i in 0..T * d {
                    tp[i] += dmlp[i];
                }
            }
        }
        // The unperturbed CPU layer-1 output = cpu_post_l1; measure the delta.
        let (dcond, _) = worst_diff(&cpu_post_l1, &tp);
        let rel_in = eps; // relative input perturbation
        println!(
            "conditioning: input +{eps:.0e} → layer1 output moves {dcond:.3e} (gain ≈ {:.2e})",
            dcond / rel_in
        );
        let _ = t_post_mlp0_pert;
    }

    // The end-to-end column via suffix_pass for cross-check.
    let dv_h = create_f32(&client, &dv);
    let got = unsafe {
        riir_infer_gpu::lens_batch::suffix_pass(
            &client,
            &stack,
            &doc,
            &dv_h,
            FitPoint::Mid(0),
            0,
            T,
        )
    }
    .expect("suffix");
    let gpu_col = &got[d..2 * d];
    let (de, ate) = worst_diff(&cpu_col, gpu_col);
    println!(
        "  end-to-end: worst {de:.3e} at dim {ate} (cpu {:+.4} gpu {:+.4})",
        cpu_col[ate], gpu_col[ate]
    );
    // No assert — reporting cell.
}
