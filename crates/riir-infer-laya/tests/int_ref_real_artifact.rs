//! Issue 1214 T1c — the REAL-ARTIFACT receipt: the calibrated artifact
//! (`scripts/i1214_prepare.py --out <dir>`, Qwen3-0.6B int8) loaded
//! through `artifact::load_dir` and driven through the integer forward.
//!
//! NOT part of the default gate surface: needs the artifact on disk —
//! `I1214_ARTIFACT_DIR=<dir>` — and is skipped LOUD without it (a
//! deferral, never a green zero). Release run recommended (the head GEMM
//! over a 151,936 vocab is the cost center; the lane is checked-op
//! integer, so debug only makes it slower, never different).
//!
//! What it receipts (the first cross-artifact G1 arms on the real model):
//!   1. the loader accepts the real container — `IntRefModel::new` runs
//!      every load-time shape + overflow law against REAL calibrated
//!      scales (the loader gates so far used synthetic fixtures only);
//!   2. prefill/decode split == full forward, EXACT (G1(b) — the cache
//!      path must reproduce the no-cache path bit for bit);
//!   3. batch composition with a junk sibling row == solo row, EXACT
//!      (G1(a) — one row must not smell another);
//!   4. greedy decode through `generate` exercises the real cache
//!      growth path (capacity-stride slots, grow-by-doubling);
//!   5. the printed receipt line: blake3 of the final-position logits
//!      row + the artifact digests + the greedy token ids. Determinism
//!      receipts quote THIS line — same bytes on every rig is the
//!      lane's whole point.
//!
//! Run:
//!   I1214_ARTIFACT_DIR=artifacts/qwen3-0.6b-int8 \
//!     cargo test -p riir-infer-laya --release --features int_ref \
//!     --test int_ref_real_artifact -- --nocapture

#![cfg(feature = "int_ref")]

use riir_infer_laya::laya::riir::int_ref::{IntRefModel, KvCache, load_dir};

fn blake3_hex(bytes: &[u8]) -> String {
    use blake3::Hasher;
    let mut h = Hasher::new();
    h.update(bytes);
    h.finalize().to_hex().to_string()
}

fn logits_row_digest(row: &[i32]) -> String {
    // LE bytes of the i32 values — the byte-level identity the receipt
    // is about (endianness pinned so a big-endian host quoting the same
    // line still agrees).
    let mut buf = Vec::with_capacity(row.len() * 4);
    for &v in row {
        buf.extend_from_slice(&(v as u32).to_le_bytes());
    }
    blake3_hex(&buf)
}

fn prompt_ids(vocab: usize, t: usize, salt: u64) -> Vec<u32> {
    (0..t)
        .map(|i| ((i as u64 + salt) * 7_919 % (vocab as u64 - 1)) as u32)
        .collect()
}

#[test]
fn real_artifact_receipt() {
    let Some(dir) = std::env::var_os("I1214_ARTIFACT_DIR") else {
        eprintln!(
            "SKIP LOUD: I1214_ARTIFACT_DIR not set — the receipt needs the \
             calibrated artifact (scripts/i1214_prepare.py), never a green zero"
        );
        std::process::exit(0);
    };
    let dir = std::path::PathBuf::from(dir);
    let (a, digests) =
        load_dir(&dir).unwrap_or_else(|e| panic!("load_dir({}): {e}", dir.display()));
    assert_eq!(digests.model.len(), 64, "model digest is blake3 hex");
    assert_eq!(digests.config.len(), 64, "config digest is blake3 hex");
    let model = IntRefModel::new(a);
    let c = model.config();
    let t = 8;
    let ids = prompt_ids(c.vocab, t, 1);
    let junk = prompt_ids(c.vocab, t, 9_999);

    // (1) full forward, no cache
    let positions: Vec<usize> = (0..t).collect();
    let (logits, _xm, _xk) = model.forward(&ids, &positions, None, None, 1, t);
    assert_eq!(logits.len(), t * c.vocab, "logits rows");
    let full_last = &logits[(t - 1) * c.vocab..t * c.vocab];

    // (2) prefill/decode split must reproduce the same rows EXACTLY
    let mut cache = KvCache::new(1, c.n_layers, c.n_kv, c.head_dim);
    let split_at = 6;
    let (pre, _xm, _xk) = model.forward(
        &ids[..split_at],
        &positions[..split_at],
        None,
        Some(&mut cache),
        1,
        split_at,
    );
    assert_eq!(
        &pre[..split_at * c.vocab],
        &logits[..split_at * c.vocab],
        "prefill rows diverge from the full forward"
    );
    for step in split_at..t {
        let (lg, _xm, _xk) = model.forward(&[ids[step]], &[step], None, Some(&mut cache), 1, 1);
        let want = &logits[step * c.vocab..(step + 1) * c.vocab];
        assert_eq!(&lg[..c.vocab], want, "decode step {step} diverges (G1b)");
    }

    // (3) batch composition: the target row inside a junk batch must be
    // byte-identical to its solo forward (G1a)
    let mut b_ids = ids.clone();
    b_ids.extend_from_slice(&junk);
    let mut b_pos = positions.clone();
    b_pos.extend_from_slice(&positions);
    let (blogits, _xm, _xk) = model.forward(&b_ids, &b_pos, None, None, 2, t);
    let b_last = &blogits[(t - 1) * c.vocab..t * c.vocab];
    assert_eq!(b_last, full_last, "batched row diverges (G1a)");

    // (4) greedy decode through the real cache-growth path
    let gens = model.generate(&ids, 1, t, 4);

    // (5) the receipt line
    eprintln!(
        "i1214 receipt: model={} config={} logits_final={} greedy={:?}",
        &digests.model[..16],
        &digests.config[..16],
        logits_row_digest(full_last),
        gens[0]
    );
}

// ---------------------------------------------------------------------------
// G3 — the ΔPPL eval on the frozen slice (env I1214_EVAL_IDS, default
// `<artifact dir>/eval_ids.json` from scripts/i1214_eval_ids.py; skipped
// loud without the file).
//
// Protocol (pre-registered in `.issues/1214_integer_exact_reference_lane.md`):
// wikitext-2-raw-v1 TEST, first 32,768 tokens, chunks of 1024,
// non-overlapping; HF next-token convention — logits row t scores the
// gold id at t+1, so each chunk yields chunk-1 predictions and the last
// row's prediction (past the chunk edge) is DROPPED. The identical rule
// runs in the Python baseline (`scripts/i1214_eval_baseline.py`), so
// ΔPPL is clean. The eval reader is allowed float — it is a measurement
// instrument beside the lane, never a lane surface; the lane itself
// stays integer.
// ---------------------------------------------------------------------------

#[test]
fn real_artifact_g3_ppl() {
    let Some(dir) = std::env::var_os("I1214_ARTIFACT_DIR") else {
        eprintln!(
            "SKIP LOUD: I1214_ARTIFACT_DIR not set — G3 needs the calibrated \
             artifact, never a green zero"
        );
        std::process::exit(0);
    };
    let dir = std::path::PathBuf::from(dir);
    let ids_path = std::env::var_os("I1214_EVAL_IDS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| dir.join("eval_ids.json"));
    if !ids_path.exists() {
        eprintln!(
            "SKIP LOUD: {} missing — run scripts/i1214_eval_ids.py first, \
             never a green zero",
            ids_path.display()
        );
        std::process::exit(0);
    }

    let (a, _digests) = load_dir(&dir).unwrap_or_else(|e| panic!("load_dir: {e}"));
    let model = std::sync::Arc::new(IntRefModel::new(a));
    let c = model.config();
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&ids_path).expect("read eval ids"))
            .expect("parse eval ids");
    let chunk: usize = meta["chunk"].as_u64().expect("chunk") as usize;
    let ids: Vec<u32> = meta["ids"]
        .as_array()
        .expect("ids array")
        .iter()
        .map(|v| v.as_u64().expect("id") as u32)
        .collect();
    assert_eq!(ids.len() % chunk, 0, "eval ids must tile whole chunks");
    let n_chunks = ids.len() / chunk;

    // per-vocab column scale head_m[v]/2^head_k (the LM head's grid)
    let head_k = model.head_k();
    let hm_f: Vec<f64> = model
        .head_m()
        .iter()
        .map(|&m| m as f64 / (1u64 << head_k) as f64)
        .collect();

    let started = std::time::Instant::now();
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(8);
    let next_chunk = std::sync::atomic::AtomicUsize::new(0);
    let results = std::sync::Mutex::new(Vec::with_capacity(n_chunks));
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                loop {
                    let ci = next_chunk.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if ci >= n_chunks {
                        break;
                    }
                    let model = &model;
                    let slice = &ids[ci * chunk..(ci + 1) * chunk];
                    let positions: Vec<usize> = (0..chunk).collect();
                    let (logits, xm, xk) = model.forward(slice, &positions, None, None, 1, chunk);
                    // per-token quant: xm/xk carry ONE scale PER ROW — row t's
                    // logits live under row t's scale (xm[0] would be wrong
                    // for every other position)
                    let xm_f: Vec<f64> = xm
                        .iter()
                        .zip(xk.iter())
                        .map(|(&m, &k)| m as f64 / (1u64 << k) as f64)
                        .collect();
                    // HF convention: logits row t predicts slice[t+1] — score
                    // tokens 1..chunk-1 with rows 0..chunk-2 (the last row
                    // predicts past the chunk edge and is dropped; the
                    // Python baseline applies the same rule)
                    let mut nll_sum = 0.0f64;
                    let mut nll_fixed = 0u64;
                    for t in 0..chunk - 1 {
                        let row = &logits[t * c.vocab..(t + 1) * c.vocab];
                        // real-domain logits (f64; exact products: i32 x
                        // 16-bit mantissa scales fit the 53-bit mantissa)
                        let mut max = f64::NEG_INFINITY;
                        let mut real: Vec<f64> = Vec::with_capacity(c.vocab);
                        for (v, &l) in row.iter().enumerate() {
                            let r = l as f64 * xm_f[t] * hm_f[v];
                            real.push(r);
                            if r > max {
                                max = r;
                            }
                        }
                        let mut lse = 0.0f64;
                        for &r in &real {
                            lse += (r - max).exp();
                        }
                        let lse = max + lse.ln();
                        let nll = lse - real[slice[t + 1] as usize];
                        nll_sum += nll;
                        // fixed-point receipt: nll * 2^20 rounded — the
                        // deterministic eval-side digest input (per-position
                        // rounding, sequential fold; the lane's own Issue-940
                        // tie law does not reach eval arithmetic, but the
                        // receipt must still be reproducible per platform)
                        nll_fixed =
                            nll_fixed.wrapping_add(((nll * (1u64 << 20) as f64).round()) as u64);
                    }
                    results
                        .lock()
                        .expect("results lock")
                        .push((ci, nll_sum, nll_fixed));
                }
            });
        }
    });
    let mut results = results.into_inner().expect("results lock");
    results.sort_by_key(|r| r.0);
    let total_nll: f64 = results.iter().map(|r| r.1).sum();
    let total_fixed: u64 = results.iter().fold(0u64, |a, r| a.wrapping_add(r.2));
    let n_pred = ids.len() - n_chunks; // chunk-count * (chunk-1)
    let ppl = (total_nll / n_pred as f64).exp();
    let elapsed = started.elapsed();

    eprintln!(
        "i1214 G3 int8: ppl={:.6} n_pred={} chunks={} threads={} elapsed={:.1}s \
         nll_fixed={:016x}",
        ppl,
        n_pred,
        n_chunks,
        threads,
        elapsed.as_secs_f32(),
        total_fixed
    );

    // G2 (informational, no bar): sampled decode at b1 on chunk 0's tail
    let mut cache = KvCache::new(1, c.n_layers, c.n_kv, c.head_dim);
    let prefill: Vec<u32> = ids[..64].to_vec();
    let positions: Vec<usize> = (0..64).collect();
    let (_lg, _xm, _xk) = model.forward(&prefill, &positions, None, Some(&mut cache), 1, 64);
    let mut cur = prefill[63];
    let g_started = std::time::Instant::now();
    let steps = 10;
    let mut toks = Vec::with_capacity(steps);
    for step in 0..steps {
        let (lg, _xm, _xk) = model.forward(&[cur], &[64 + step], None, Some(&mut cache), 1, 1);
        let tok = riir_infer_laya::laya::riir::int_ref::greedy_pick(&lg[..c.vocab], model.head_m());
        toks.push(tok);
        cur = tok as u32;
    }
    let g_elapsed = g_started.elapsed();
    eprintln!(
        "i1214 G2 decode (b1, PROVISIONAL): {:.3} tok/s over {} steps \
         ({:.3}s) tokens={:?} — quote only with box state",
        steps as f64 / g_elapsed.as_secs_f64(),
        steps,
        g_elapsed.as_secs_f64(),
        toks
    );
}
