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
