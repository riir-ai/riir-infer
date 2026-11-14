//! Issue 1216 T0 fidelity drill-down (plan 634 addendum) — WHY does the
//! host scorer replay disagree with the forward path's own logits by
//! ~2e-3 mean? The LENC reference logits are the RAW scorer outputs
//! (head.rs `reads_of`); the harness answer path applies
//! softmax(logits/T) with T = the per-(qtype,k) bucket temperature
//! (temps.rs). This example compares host-frozen-logits against
//! softmax-recomputed probs joined to the forward path's answer triple —
//! the answers ride the dump bin's own join (markers == logits order).
//!
//! Hypotheses, one read each:
//!   H1 — the reference logits in the cache are already TEMPERATURE-scaled
//!        (z/T) rather than raw (then the replay Δ would be a per-question
//!        scale artifact, |Δ| ∝ |z|·|1/T − 1|, T ≤ 1 ⇒ Δ up to ~|z|·(1/T−1)).
//!   H2 — a real transcription break (op-order or weight-layout mismatch).
//! H1 predicts: dividing the replay z by T (resolved per qtype/k) collapses
//! the delta to ~1e-6; H2 predicts it survives.

use std::io::Read;

use riir_infer_laya::laya::config::Checkpoint;
use riir_infer_laya::laya::riir::agent::{DeviceKind, RiirAgent};
use riir_infer_laya::laya::riir::ops;
use riir_infer_laya::laya::temps::softmax32;

const MAGIC: &[u8; 4] = b"LENC";

fn main() -> Result<(), String> {
    let root = riir_infer_laya::laya::weights::weights_root();
    let agent = RiirAgent::load_with_device(&root, Checkpoint::TypedDecisions, DeviceKind::Cpu)
        .map_err(|e| e.to_string())?;
    let st = agent.scorer_tensors();

    // temperature table (the same config the forward tail reads)
    let (agent_cfg, _enc_cfg) =
        riir_infer_laya::laya::config::load_checkpoint_configs(&root.join("typed"), "typed")
            .map_err(|e| e.to_string())?;
    let temps = riir_infer_laya::laya::temps::Temperatures::from_config(&agent_cfg);

    // read the fid cache rows (id, qtype, k, marker rows, ref logits)
    let mut f =
        std::fs::File::open("/tmp/i1216_t0/fid_cpu_cache.bin").map_err(|e| e.to_string())?;
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).map_err(|e| e.to_string())?;
    if &magic != MAGIC {
        return Err("not LENC".into());
    }
    let mut u32v = |f: &mut std::fs::File| -> Result<u32, String> {
        let mut b = [0u8; 4];
        f.read_exact(&mut b).map_err(|e| e.to_string())?;
        Ok(u32::from_le_bytes(b))
    };
    let _version = u32v(&mut f)?;
    let n_rows = u32v(&mut f)? as usize;
    let d = st.d;

    let mut raw_worst = 0f32;
    let mut raw_sum = 0f64;
    let mut scaled_maxdel = 0f64; // max prob-space delta after the T correction
    let mut n_opts = 0usize;
    let mut h = Vec::new();
    for _ in 0..n_rows {
        let id_len = u32v(&mut f)? as usize;
        let mut idb = vec![0u8; id_len];
        f.read_exact(&mut idb).map_err(|e| e.to_string())?;
        let id = String::from_utf8(idb).map_err(|e| e.to_string())?;
        let seq_len = u32v(&mut f)? as usize;
        let row_d = u32v(&mut f)? as usize;
        let qtype = u32v(&mut f)? as usize;
        let k = u32v(&mut f)? as usize;
        drop_n(&mut f, k * 4)?;
        let mut rows_f32 = vec![0f32; k * row_d];
        read_f32(&mut f, &mut rows_f32)?;
        drop_n(&mut f, seq_len * row_d * 4)?;
        let mut ref_logits = vec![0f32; k];
        read_f32(&mut f, &mut ref_logits)?;

        // host frozen replay (same code path as the T0 example)
        let x = ops::layer_norm(&rows_f32, &st.s0w, &st.s0b, st.eps, d);
        let s1 = ops::matmul_w(&x, k, d, &st.s1w, d);
        let mut hh = s1;
        for (row, b) in hh.chunks_exact_mut(d).zip(st.s1b.chunks_exact(d)) {
            for (v, bi) in row.iter_mut().zip(b.iter()) {
                *v += bi;
            }
        }
        ops::gelu_erf_inplace(&mut hh);
        h = hh;
        let z = {
            let mut z = ops::matmul_w(&h, k, d, &st.s3w, 1);
            for v in z.iter_mut() {
                *v += st.s3b[0];
            }
            z
        };
        for (a, b) in z.iter().zip(ref_logits.iter()) {
            let del = (a - b).abs();
            raw_worst = raw_worst.max(del);
            raw_sum += del as f64;
            n_opts += 1;
        }
        // H1 check: ref_logits/T vs z/T in prob space
        let t = temps.for_question(qtype, k);
        let probs_host = softmax32(&(z.iter().map(|v| *v / t as f32).collect::<Vec<_>>()));
        let probs_ref = softmax32(&ref_logits);
        for (a, b) in probs_host.iter().zip(probs_ref.iter()) {
            scaled_maxdel = scaled_maxdel.max((a - b).abs() as f64);
        }
        let _ = id;
    }
    println!(
        "raw logits delta: mean {:.3e} max {:.3e} ({} options)",
        raw_sum / n_opts as f64,
        raw_worst,
        n_opts
    );
    println!(
        "prob-space delta with T correction: max {:.3e}",
        scaled_maxdel
    );
    println!(
        "H1 (temperature-scaled reference) {} — H2 (transcription break) {}",
        if scaled_maxdel < 1e-6 {
            "CONFIRMED"
        } else {
            "rejected"
        },
        if scaled_maxdel >= 1e-6 {
            "CONFIRMED"
        } else {
            "rejected"
        }
    );
    Ok(())
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
