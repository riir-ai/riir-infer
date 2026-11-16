//! Per-kernel unit parity for the batched lens lane (riir-infer Issue
//! 1222). Each GPU kernel runs against its CPU twin (`riir_infer_core`
//! `lens::{ops, gdn, attn}`) on a batch-of-1 fixture — the composite
//! suffix gate (`lens_batch_parity`) localizes a failure to a layer type;
//! this gate names the KERNEL. Skips loud without a GPU.

#![cfg(feature = "lens_gpu")]

use riir_infer_gpu::cubecl_runtime::{ActiveComputeClient, CubeCLContext, create_f32, read_f32};
use riir_infer_gpu::lens_batch_kernels::*;

/// Deterministic LCG (the global_rng_gate law).
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

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn ctx_or_skip() -> Option<ActiveComputeClient> {
    let Ok(ctx) = CubeCLContext::new() else {
        eprintln!("SKIP LOUD: no GPU device — deferral, never a pass");
        return None;
    };
    Some(ctx.client())
}

/// GEMM vs `katgpt_types`-style `y = W @ x` per row (the crate matmul the
/// CPU lens calls — `riir_infer_core::types::matmul`).
#[test]
fn unit_gemm() {
    let Some(client) = ctx_or_skip() else { return };
    let mut rng = Lcg(0xD201);
    let (p_rows, rows, cols) = (3usize, 7usize, 5usize);
    let x: Vec<f32> = rng.vec(p_rows * cols);
    let w: Vec<f32> = rng.vec(rows * cols);
    let mut want = vec![0.0f32; p_rows * rows];
    for p in 0..p_rows {
        let mut y = vec![0.0f32; rows];
        riir_infer_core::types::matmul(&mut y, &w, &x[p * cols..(p + 1) * cols], rows, cols);
        want[p * rows..(p + 1) * rows].copy_from_slice(&y);
    }
    let xh = create_f32(&client, &x);
    let wh = create_f32(&client, &w);
    let yh = client.empty(p_rows * rows * 4);
    unsafe { LensGemmCL::launch(&client, xh, wh, yh.clone(), p_rows, rows, cols) };
    let got = read_f32(&client, yh).expect("read");
    assert_eq!(got.len(), want.len());
    let d = max_abs_diff(&got, &want);
    assert!(d < 2e-4, "gemm max abs diff {d}");
}

/// Scale-by-table vs the LN-rule `diag_apply` on one position block.
#[test]
fn unit_scale_table() {
    let Some(client) = ctx_or_skip() else { return };
    let mut rng = Lcg(0xD202);
    let (batch, t_len, d) = (2usize, 4usize, 6usize);
    let a: Vec<f32> = rng.vec(batch * t_len * d);
    let table: Vec<f32> = rng.vec(t_len * d);
    let mut want = a.clone();
    for (i, v) in want.iter_mut().enumerate() {
        *v *= table[i % (t_len * d)];
    }
    let ah = create_f32(&client, &a);
    let th = create_f32(&client, &table);
    let oh = client.empty(a.len() * 4);
    unsafe { LensScaleTableCL::launch(&client, ah, th, oh.clone(), a.len(), t_len * d) };
    let got = read_f32(&client, oh).expect("read");
    let dd = max_abs_diff(&got, &want);
    assert!(dd < 1e-5, "scale_table max abs diff {dd}");
}

/// Causal conv1d vs `ops::conv1d_causal_jvp` (batch 1 slice).
#[test]
fn unit_conv1d() {
    let Some(client) = ctx_or_skip() else { return };
    let mut rng = Lcg(0xD203);
    let (t_len, channels, kw) = (5usize, 4usize, 3usize);
    let dx: Vec<f32> = rng.vec(t_len * channels);
    let kernel: Vec<f32> = rng.vec(channels * kw);
    let mut want = vec![0.0f32; t_len * channels];
    riir_infer_core::lens::ops::conv1d_causal_jvp(&kernel, kw, &dx, channels, &mut want);
    let xh = create_f32(&client, &dx);
    let kh = create_f32(&client, &kernel);
    let yh = client.empty(dx.len() * 4);
    unsafe { LensConv1dCL::launch(&client, xh, kh, yh.clone(), 1, t_len, channels, kw) };
    let got = read_f32(&client, yh).expect("read");
    let dd = max_abs_diff(&got, &want);
    assert!(dd < 1e-5, "conv1d max abs diff {dd}");
}

/// GDN recurrence scan vs `gdn::gdn_recurrence_jvp` (batch 1).
#[test]
fn unit_gdn_scan() {
    let Some(client) = ctx_or_skip() else { return };
    let mut rng = Lcg(0xD204);
    let (t_len, n_v, key, val) = (4usize, 2usize, 4usize, 4usize);
    let qn: Vec<f32> = rng.vec(t_len * n_v * key);
    let kn: Vec<f32> = rng.vec(t_len * n_v * key);
    let v: Vec<f32> = rng.vec(t_len * n_v * val);
    let beta: Vec<f32> = rng.vec(t_len * n_v);
    let decay: Vec<f32> = rng.vec(t_len * n_v).iter().map(|x| x.abs()).collect();
    let dbeta: Vec<f32> = rng.vec(t_len * n_v);
    let ddecay: Vec<f32> = rng.vec(t_len * n_v);
    let dq: Vec<f32> = rng.vec(t_len * n_v * key);
    let dk: Vec<f32> = rng.vec(t_len * n_v * key);
    let dv: Vec<f32> = rng.vec(t_len * n_v * val);
    let states: Vec<f32> = rng.vec(t_len * n_v * val * key);
    let scale = 0.25f32;
    // CPU reference.
    use riir_infer_core::lens::gdn::{GdnRecurrenceConsts, gdn_recurrence_jvp};
    let c = GdnRecurrenceConsts {
        t_len,
        n_v_heads: n_v,
        key_dim: key,
        val_dim: val,
        scale,
        q: &qn,
        k: &kn,
        v: &v,
        beta: &beta,
        decay: &decay,
        states: &states,
    };
    let mut want = vec![0.0f32; t_len * n_v * val];
    {
        let mut ds = vec![0.0f32; n_v * val * key];
        let mut spre = vec![0.0f32; n_v * val * key];
        gdn_recurrence_jvp(
            &c, &dq, &dk, &dv, &dbeta, &ddecay, &mut ds, &mut spre, &mut want,
        );
    }
    // GPU.
    let (qn_h, kn_h, v_h) = (
        create_f32(&client, &qn),
        create_f32(&client, &kn),
        create_f32(&client, &v),
    );
    let (beta_h, decay_h) = (create_f32(&client, &beta), create_f32(&client, &decay));
    let (dbeta_h, ddecay_h) = (create_f32(&client, &dbeta), create_f32(&client, &ddecay));
    let (dq_h, dk_h, dv_h) = (
        create_f32(&client, &dq),
        create_f32(&client, &dk),
        create_f32(&client, &dv),
    );
    let states_h = create_f32(&client, &states);
    let ds_h = client.empty(n_v * val * key * 4);
    let spre_h = client.empty(n_v * val * key * 4);
    let dy_h = client.empty(t_len * n_v * val * 4);
    unsafe {
        LensGdnScanCL::launch(
            &client,
            qn_h,
            kn_h,
            v_h,
            beta_h,
            decay_h,
            dbeta_h,
            ddecay_h,
            dq_h,
            dk_h,
            dv_h,
            states_h,
            ds_h,
            spre_h,
            dy_h.clone(),
            1,
            t_len,
            n_v,
            key,
            val,
            scale,
        )
    };
    let got = read_f32(&client, dy_h).expect("read");
    let dd = max_abs_diff(&got, &want);
    assert!(dd < 1e-4, "gdn scan max abs diff {dd}");
}

/// QK-norm vs `attn::qk_norm_true_jvp` (batch 1).
#[test]
fn unit_qk_norm() {
    let Some(client) = ctx_or_skip() else { return };
    let mut rng = Lcg(0xD205);
    let (t_len, heads, hd) = (3usize, 2usize, 4usize);
    let x: Vec<f32> = rng.vec(t_len * heads * hd);
    let r: Vec<f32> = rng
        .vec(t_len * heads)
        .iter()
        .map(|g| g.abs() + 0.1)
        .collect();
    let gamma: Vec<f32> = rng.vec(hd);
    let dx: Vec<f32> = rng.vec(t_len * heads * hd);
    let mut want = vec![0.0f32; dx.len()];
    for t in 0..t_len {
        use riir_infer_core::lens::attn::{QkNormConsts, qk_norm_true_jvp};
        let c = QkNormConsts {
            x: &x[t * heads * hd..(t + 1) * heads * hd],
            r: &r[t * heads..(t + 1) * heads],
        };
        qk_norm_true_jvp(
            &c,
            &gamma,
            hd,
            &dx[t * heads * hd..(t + 1) * heads * hd],
            &mut want[t * heads * hd..(t + 1) * heads * hd],
        );
    }
    let (xh, rh, gh, dxh) = (
        create_f32(&client, &x),
        create_f32(&client, &r),
        create_f32(&client, &gamma),
        create_f32(&client, &dx),
    );
    let dyh = client.empty(dx.len() * 4);
    unsafe { LensQkNormCL::launch(&client, xh, rh, gh, dxh, dyh.clone(), 1, t_len, heads, hd) };
    let got = read_f32(&client, dyh).expect("read");
    let dd = max_abs_diff(&got, &want);
    assert!(dd < 1e-5, "qk_norm max abs diff {dd}");
}

/// Softmax-attention jvp vs `attn::attention_jvp` (batch 1, GQA 2/1).
#[test]
fn unit_attn_jvp() {
    let Some(client) = ctx_or_skip() else { return };
    let mut rng = Lcg(0xD206);
    let (t_len, n_head, n_kv, hd) = (4usize, 2usize, 1usize, 4usize);
    let q_dim = n_head * hd;
    let kv_dim = n_kv * hd;
    let q: Vec<f32> = rng.vec(t_len * q_dim);
    let k: Vec<f32> = rng.vec(t_len * kv_dim);
    let v: Vec<f32> = rng.vec(t_len * kv_dim);
    let attn: Vec<f32> = (0..t_len * n_head * t_len)
        .map(|i| {
            let t = i / (n_head * t_len);
            let j = i % t_len;
            if j <= t { 1.0 / (t + 1) as f32 } else { 0.0 }
        })
        .collect();
    let dq: Vec<f32> = rng.vec(t_len * q_dim);
    let dk: Vec<f32> = rng.vec(t_len * kv_dim);
    let dv: Vec<f32> = rng.vec(t_len * kv_dim);
    let scale = (hd as f32).sqrt().recip();
    use riir_infer_core::lens::attn::{AttentionConsts, attention_jvp};
    let c = AttentionConsts {
        t_len,
        n_head,
        n_kv,
        head_dim: hd,
        scale,
        attn: &attn,
    };
    let mut want = vec![0.0f32; t_len * q_dim];
    {
        let mut ds_scratch = vec![0.0f32; t_len];
        attention_jvp(&c, &q, &k, &v, &dq, &dk, &dv, &mut ds_scratch, &mut want);
    }
    let (qh, kh, vh, ah) = (
        create_f32(&client, &q),
        create_f32(&client, &k),
        create_f32(&client, &v),
        create_f32(&client, &attn),
    );
    let (dqh, dkh, dvh) = (
        create_f32(&client, &dq),
        create_f32(&client, &dk),
        create_f32(&client, &dv),
    );
    let outh = client.empty(t_len * q_dim * 4);
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
            1,
            t_len,
            n_head,
            n_kv,
            hd,
            scale,
        )
    };
    let got = read_f32(&client, outh).expect("read");
    let dd = max_abs_diff(&got, &want);
    assert!(dd < 1e-4, "attn jvp max abs diff {dd}");
}

/// L2-norm + expand vs `gdn::l2_norm_jvp` + `gdn::expand_tiled_jvp`.
#[test]
fn unit_l2norm_expand() {
    let Some(client) = ctx_or_skip() else { return };
    let mut rng = Lcg(0xD207);
    let (t_len, n_k, n_v, key) = (3usize, 1usize, 2usize, 4usize);
    let dx: Vec<f32> = rng.vec(t_len * n_k * key);
    let qn: Vec<f32> = rng.vec(t_len * n_v * key);
    let inv: Vec<f32> = rng.vec(t_len * n_k).iter().map(|g| g.abs() + 0.1).collect();
    // CPU: per position, l2 map in place then tile (the y consts slice
    // starts at the FULL expanded row — stride n_v·key).
    let mut want = vec![0.0f32; t_len * n_v * key];
    for t in 0..t_len {
        use riir_infer_core::lens::gdn::{L2NormConsts, expand_tiled_jvp, l2_norm_jvp};
        let mut src = dx[t * n_k * key..(t + 1) * n_k * key].to_vec();
        let qc = L2NormConsts {
            y: &qn[t * n_v * key..t * n_v * key + n_k * key],
            inv: &inv[t * n_k..(t + 1) * n_k],
        };
        l2_norm_jvp(&qc, key, &mut src);
        expand_tiled_jvp(
            &src,
            n_k,
            key,
            n_v / n_k,
            &mut want[t * n_v * key..(t + 1) * n_v * key],
        );
    }
    let (dxh, qnh, invh) = (
        create_f32(&client, &dx),
        create_f32(&client, &qn),
        create_f32(&client, &inv),
    );
    let dsth = client.empty(t_len * n_v * key * 4);
    unsafe {
        LensL2NormExpandCL::launch(
            &client,
            dxh,
            qnh,
            invh,
            dsth.clone(),
            1,
            t_len,
            n_k,
            n_v / n_k,
            key,
            n_v * key,
        )
    };
    let got = read_f32(&client, dsth).expect("read");
    let dd = max_abs_diff(&got, &want);
    assert!(dd < 1e-5, "l2norm+expand max abs diff {dd}");
}

/// RoPE jvp vs `ops::rope_jvp` (partial rope, GQA 2/1).
#[test]
fn unit_rope() {
    let Some(client) = ctx_or_skip() else { return };
    let mut rng = Lcg(0xD208);
    let (t_len, n_head, n_kv, head_dim, rotary_dim) = (3usize, 2usize, 1usize, 4usize, 4usize);
    let half = rotary_dim / 2;
    let freq: Vec<f32> = rng
        .vec(head_dim / 2)
        .iter()
        .map(|x| x.abs() + 0.05)
        .collect();
    // Host cs table (the driver's rule).
    let mut cs = vec![0.0f32; t_len * 2 * half];
    for t in 0..t_len {
        for i in 0..half {
            let (s, c) = (t as f32 * freq[i]).sin_cos();
            cs[t * 2 * half + i] = c;
            cs[t * 2 * half + half + i] = s;
        }
    }
    let q_dim = n_head * head_dim;
    let kv_dim = n_kv * head_dim;
    let mut dq = rng.vec(t_len * q_dim);
    let mut dk = rng.vec(t_len * kv_dim);
    let in_q = dq.clone();
    let in_k = dk.clone();
    for t in 0..t_len {
        use riir_infer_core::lens::ops::rope_jvp;
        rope_jvp(
            &mut dq[t * q_dim..(t + 1) * q_dim],
            &mut dk[t * kv_dim..(t + 1) * kv_dim],
            t,
            head_dim,
            rotary_dim,
            &freq,
        );
    }
    let dqh = create_f32(&client, &in_q);
    let dkh = create_f32(&client, &in_k);
    let csh = create_f32(&client, &cs);
    unsafe {
        LensRopeJvpCL::launch(
            &client,
            dqh.clone(),
            dkh.clone(),
            csh,
            1,
            t_len,
            n_head,
            n_kv,
            head_dim,
            rotary_dim,
        )
    };
    let got_q = read_f32(&client, dqh).expect("read");
    let got_k = read_f32(&client, dkh).expect("read");
    let ddq = max_abs_diff(&got_q, &dq);
    let ddk = max_abs_diff(&got_k, &dk);
    if ddq >= 1e-5 || ddk >= 1e-5 {
        for i in 0..got_q.len() {
            println!(
                "q[{i}] cpu={:.6} gpu={:.6} in={:.6}",
                dq[i], got_q[i], in_q[i]
            );
        }
        println!("cs={cs:?} freq={freq:?} half={half}");
    }
    assert!(
        ddq < 1e-5 && ddk < 1e-5,
        "rope max abs diff q {ddq} k {ddk}"
    );
}

/// Col-sum vs the accumulator's position fold.
#[test]
fn unit_col_sum() {
    let Some(client) = ctx_or_skip() else { return };
    let mut rng = Lcg(0xD209);
    let (batch, t_len, d) = (2usize, 5usize, 4usize);
    let buf: Vec<f32> = rng.vec(batch * t_len * d);
    let mut want = vec![0.0f32; batch * d];
    for p in 0..batch {
        for t in 0..t_len {
            for (j, w) in want[p * d..(p + 1) * d].iter_mut().enumerate() {
                *w += buf[(p * t_len + t) * d + j];
            }
        }
    }
    let bh = create_f32(&client, &buf);
    let oh = client.empty(batch * d * 4);
    unsafe { LensColSumCL::launch(&client, bh, oh.clone(), batch, t_len, d, 0, t_len) };
    let got = read_f32(&client, oh).expect("read");
    let dd = max_abs_diff(&got, &want);
    assert!(dd < 1e-4, "col_sum max abs diff {dd}");
}

/// Half rule vs `ops::half_rule_jvp` + `ops::gate_half_rule_jvp` shapes.
#[test]
fn unit_half_rule2() {
    let Some(client) = ctx_or_skip() else { return };
    let mut rng = Lcg(0xD20A);
    let (batch, t_len, n) = (2usize, 3usize, 6usize);
    let a: Vec<f32> = rng.vec(t_len * n);
    let b: Vec<f32> = rng.vec(t_len * n);
    let c: Vec<f32> = rng.vec(t_len * n);
    let x: Vec<f32> = rng.vec(batch * t_len * n);
    let y: Vec<f32> = rng.vec(batch * t_len * n);
    let mut want = vec![0.0f32; x.len()];
    for i in 0..x.len() {
        let m = i % (t_len * n);
        want[i] = 0.5 * (a[m] * x[i] + b[m] * c[m] * y[i]);
    }
    let (ah, bh, ch, xh, yh) = (
        create_f32(&client, &a),
        create_f32(&client, &b),
        create_f32(&client, &c),
        create_f32(&client, &x),
        create_f32(&client, &y),
    );
    let oh = client.empty(x.len() * 4);
    unsafe { LensHalfRule2CL::launch(&client, ah, bh, ch, xh, yh, oh.clone(), x.len(), t_len * n) };
    let got = read_f32(&client, oh).expect("read");
    let dd = max_abs_diff(&got, &want);
    assert!(dd < 1e-5, "half_rule2 max abs diff {dd}");
}
