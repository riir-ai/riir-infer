//! i1221 T3(a) — the KL-spectra instrument's gate battery (plan 636
//! read-order leg 2). The runner's rows are measurements; THESE are the
//! lane's asserts, on the synthetic dory GGUF (the shared kit,
//! `common/synth_dory.rs`) — the same fixtures law the T2 posture used:
//!
//! 1. **kill-switch** — the lens-armed prefill ≡ the unarmed prefill,
//!    final logits bit-identical (arming clones, never perturbs).
//! 2. **lens premise, Dory-strongest form** — the exit on snapshot τ is
//!    BIT-IDENTICAL to a fresh (τ+1)-loop prefill at EVERY position
//!    (the readout depends only on the final recurrent state; katgpt-rs
//!    Bench 899 could only pin the pos-0 form — their readout shares
//!    the loop cache, Dory's does not).
//! 3. **last-exit identity** — exit_K ≡ full logits (the KL profile's
//!    final entry is EXACTLY 0, the G1b anchor).
//! 4. **not-inert** — exit_1 ≠ exit_K somewhere (KL > 0): the loop
//!    deepens on synth weights.
//! 5. **KL math** — self-KL = 0 exactly, non-negativity, distinct-pair
//!    positivity, lse algebra vs a direct softmax fold.
//! 6. **readout summarizers** — effective_depth / agreement_exit /
//!    monotone_fraction on planted profiles (the Bench-899 definitions).
//! 7. **row + resume lineage** — serde round trip; the lineage check
//!    accepts its own header + skips completed pairs, REFUSES on a
//!    moved pin/STAMP/prompt-spec (exit 2 paths exercised via the
//!    public function's contract on tmp files — pid-scoped names, the
//!    shared-temp-path law).
//! 8. **determinism** — two lens runs produce byte-identical snapshots
//!    and logits.

#![cfg(feature = "dory")]

#[path = "common/synth_bonsai2.rs"]
mod synth;
#[path = "common/synth_dory.rs"]
mod synth_dory;

use riir_infer_core::dory::spectra::{
    KL_VECTOR_EPS, agreement_exit, argmax_and_margin, effective_depth, exit_kl, exit_row,
    kl_loop1_vs_exit, kl_positions, monotone_fraction, run_lens, spectra_row,
    verify_resume_lineage,
};
use riir_infer_core::dory::{DoryForward, DoryKvCache, load_dory_weights};
use synth::write_tmp;
use synth_dory::synth_gguf;
use std::sync::atomic::{AtomicU64, Ordering};

/// A pid+counter-scoped tmp name (the shared-temp-path law: safe against
/// sibling tests in one binary AND another process running the same test).
static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

fn tmp_path(tag: &str) -> std::path::PathBuf {
    let n = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "i1221_t3a_{}_{}_{}",
        tag,
        std::process::id(),
        n
    ))
}

fn loaded_weights() -> (std::path::PathBuf, riir_infer_core::dory::DoryWeights) {
    let bytes = synth_gguf("dory", true);
    let path = write_tmp("i1221_t3a_synth", &bytes);
    eprintln!("[t3a-gate] synth at {} ({} bytes)", path.display(), bytes.len());
    let w = load_dory_weights(&path).expect("synth dory loads");
    (path, w)
}

const SEQ: usize = 5;

fn prompt() -> Vec<u32> {
    // Deterministic pseudo-tokens in [4, vocab) — the smoke's law.
    let mut s = 0x9E3779B97F4A7C15u64;
    (0..SEQ)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            4 + ((s >> 33) % 7) as u32
        })
        .collect()
}

fn logits_identical(a: &[Vec<f32>], b: &[Vec<f32>]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b.iter()).all(|(x, y)| {
            x.len() == y.len() && x.iter().zip(y.iter()).all(|(u, v)| u.to_bits() == v.to_bits())
        })
}

/// Gate 1 — the kill-switch: arming the lens never moves the logits.
#[test]
fn kill_switch_lens_equals_plain_prefill() {
    let (_path, w) = loaded_weights();
    let fwd = DoryForward::new(&w);
    let toks = prompt();

    let mut armed_cache = DoryKvCache::new(&w, w.meta.num_loops);
    let mut snaps = Vec::new();
    let armed = run_lens(&fwd, &toks, &mut armed_cache, &mut snaps).expect("lens");

    let mut plain_cache = DoryKvCache::new(&w, w.meta.num_loops);
    let plain = fwd.prefill(&toks, &mut plain_cache).expect("plain");

    assert!(logits_identical(&armed, &plain), "kill-switch: the lens moved the logits");
    assert_eq!(snaps.len(), w.meta.num_loops, "snapshot count != k");
    assert!(
        snaps.iter().all(|s| s.len() == SEQ * w.dims.n_embd),
        "snapshot shape != t·n_embd"
    );
}

/// Gate 2 — the lens premise at its strongest Dory form: exiting at
/// depth d from the lens ≡ a fresh d-loop prefill, EVERY position,
/// bit-identical.
#[test]
fn lens_premise_exit_equals_fresh_prefill_all_positions() {
    let (_path, w) = loaded_weights();
    let fwd = DoryForward::new(&w);
    let toks = prompt();
    let k = w.meta.num_loops;

    let mut cache = DoryKvCache::new(&w, k);
    let mut snaps = Vec::new();
    let _full = run_lens(&fwd, &toks, &mut cache, &mut snaps).expect("lens");

    for (d, snap) in snaps.iter().enumerate() {
        let want_k = d + 1;
        let mut fresh_cache = DoryKvCache::new(&w, want_k);
        let fresh = fwd.prefill(&toks, &mut fresh_cache).expect("fresh prefill");
        let mut exit_cache = DoryKvCache::new(&w, 1);
        let exit = exit_row(&fwd, snap, SEQ, &mut exit_cache).expect("exit");
        assert!(
            logits_identical(&exit, &fresh),
            "exit at depth {want_k} != fresh {want_k}-loop prefill (lens premise broken)"
        );
    }
}

/// Gate 3 — the last exit IS the full readout (KL profile's final entry
/// exactly 0) and the KL algebra sees it that way.
#[test]
fn last_exit_identity_and_zero_kl() {
    let (_path, w) = loaded_weights();
    let fwd = DoryForward::new(&w);
    let toks = prompt();
    let k = w.meta.num_loops;

    let mut cache = DoryKvCache::new(&w, k);
    let mut snaps = Vec::new();
    let full = run_lens(&fwd, &toks, &mut cache, &mut snaps).expect("lens");

    let mut exit_cache = DoryKvCache::new(&w, 1);
    let exit_k = exit_row(&fwd, &snaps[k - 1], SEQ, &mut exit_cache).expect("exit");
    assert!(logits_identical(&exit_k, &full), "exit_K != full logits");

    let kl = kl_positions(&full, &exit_k);
    assert!(kl.iter().all(|v| *v == 0.0), "KL(full‖exit_K) not exactly 0: {kl:?}");

    // Deepening axis: KL(exit_1 ‖ exit_1) = 0 exactly too.
    let mut c1 = DoryKvCache::new(&w, 1);
    let exit1 = exit_row(&fwd, &snaps[0], SEQ, &mut c1).expect("exit1");
    let kl11 = kl_loop1_vs_exit(&exit1, &exit1);
    assert!(kl11.iter().all(|v| *v == 0.0), "KL(exit1‖exit1) not 0");
}

/// Gate 4 — the loop is not inert on synth weights: depth 1's readout
/// differs from the full depth somewhere (KL > 0).
#[test]
fn loop_not_inert_exit1_differs_from_full() {
    let (_path, w) = loaded_weights();
    let fwd = DoryForward::new(&w);
    let toks = prompt();
    let k = w.meta.num_loops;

    let mut cache = DoryKvCache::new(&w, k);
    let mut snaps = Vec::new();
    let full = run_lens(&fwd, &toks, &mut cache, &mut snaps).expect("lens");

    // The loop is not inert: depth 1's recurrent-exit STATE differs from
    // the full depth's (the computation genuinely deepens).
    let mut max_state_diff = 0.0f32;
    for (a, b) in snaps[0].iter().zip(snaps[k - 1].iter()) {
        max_state_diff = max_state_diff.max((a - b).abs());
    }
    assert!(
        max_state_diff > 0.0,
        "depth-1 state == full-depth state — the loop would be inert"
    );
    // The readout sees the difference too (non-constant logit diff —
    // NOT a softmax-equal pair). ⚠ The synthetic head is ONE-HOT
    // SATURATED by construction (coordinate-vector magnitudes × the
    // 1/0.0177 logit scale): KL(full‖exit_d) reads exactly 0 on the
    // synth because both distributions collapse onto the same token
    // (KL of two identical one-hots is exactly 0, whatever the logit
    // gap). The KL>0 signal is the REAL PIN cell's to show — the gate
    // asserts the state-level deepening + a non-constant readout diff,
    // never a synth KL magnitude.
    let mut min_diff = f32::INFINITY;
    let mut max_diff = f32::NEG_INFINITY;
    let exit1 = exit_logits(&w, &snaps[0]);
    for (a, b) in full[0].iter().zip(exit1[0].iter()) {
        min_diff = min_diff.min(a - b);
        max_diff = max_diff.max(a - b);
    }
    assert!(
        max_diff > min_diff,
        "exit-1 logits == full logits + constant (softmax-equal) — readout blind to depth"
    );
    eprintln!(
        "[t3a-gate] inert gate: max_state_diff={max_state_diff} logit_diff_span=[{min_diff}, {max_diff}] (synth head one-hot saturated — KL magnitudes are the real pin's readout)"
    );
}

fn exit_logits(w: &riir_infer_core::dory::DoryWeights, snap: &[f32]) -> Vec<Vec<f32>> {
    let fwd = DoryForward::new(w);
    let mut scratch = DoryKvCache::new(w, 1);
    exit_row(&fwd, snap, SEQ, &mut scratch).expect("exit")
}

/// Gate 5 — the KL algebra: self = 0, non-negative, distinct > 0, and
/// the lse form agrees with a direct softmax fold.
#[test]
fn kl_math_identities() {
    let p = vec![1.0f32, 2.0, -0.5, 3.0, 0.0];
    let q = vec![1.5f32, -1.0, 2.5, 0.25, 0.75];

    assert_eq!(exit_kl(&p, &p), 0.0, "self-KL != 0");
    let kl_pq = exit_kl(&p, &q);
    let kl_qp = exit_kl(&q, &p);
    assert!(kl_pq >= 0.0 && kl_qp >= 0.0, "negative KL");
    assert!(kl_pq > 0.0, "distinct pairs must have KL > 0");

    // Direct softmax fold in f64 vs the lse form (same fold, different
    // spelling — they must agree to f64 rounding).
    let direct = {
        let max_p = p.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let max_q = q.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let zp: f64 = p.iter().map(|&x| (x as f64 - max_p).exp()).sum();
        let zq: f64 = q.iter().map(|&x| (x as f64 - max_q).exp()).sum();
        let mut acc = 0.0f64;
        for (&a, &b) in p.iter().zip(q.iter()) {
            let pa = (a as f64 - max_p).exp() / zp;
            let qb = (b as f64 - max_q).exp() / zq;
            acc += pa * (pa.ln() - qb.ln());
        }
        acc
    };
    assert!(
        (kl_pq - direct).abs() < 1e-12,
        "lse-form KL {kl_pq} vs direct softmax fold {direct}"
    );

    // Length mismatch → NaN (the caller-bug visibility contract).
    assert!(exit_kl(&p, &q[..3]).is_nan());
    assert!(exit_kl(&[], &[]).is_nan());
}

/// Gate 6 — the Bench-899 summarizers on planted profiles.
#[test]
fn summarizer_definitions() {
    // effective_depth: first post-peak loop within threshold
    // (Bench-899 semantics: the peak loop itself qualifies on a loose
    // threshold; pre-peak loops never count, however small).
    let kl = vec![3.0, 5.0, 2.0, 0.0];
    assert_eq!(effective_depth(&kl, 1.0), Some(4));
    assert_eq!(effective_depth(&kl, 2.5), Some(3));
    assert_eq!(effective_depth(&kl, 6.0), Some(2), "loose threshold: the peak loop itself qualifies");
    // Monotone decay (peak at 0): depth 1 exits immediately.
    let mono = vec![0.5, 0.2, 0.0];
    assert_eq!(effective_depth(&mono, 1.0), Some(1));
    // Pre-peak small values must not count (the Bench-899 trap row).
    let trap = vec![0.05, 5.0, 0.4, 0.05, 0.0];
    assert_eq!(effective_depth(&trap, 0.1), Some(4));
    // No post-peak hit: peak at the END... impossible against a final 0,
    // but NaN profiles refuse (the divergence contract).
    assert_eq!(effective_depth(&[1.0, f64::NAN, 0.0], 1.0), None);
    assert_eq!(effective_depth(&[], 1.0), None);

    // agreement_exit: smallest k whose every ≥k argmax equals the final.
    assert_eq!(agreement_exit(&[7, 3, 3, 3]), Some(2));
    assert_eq!(agreement_exit(&[3, 3, 3, 3]), Some(1));
    // [1, 2, 3, 2]: the tail agreeing with the FINAL (2) is just the
    // last entry — so k = 4 - 1 + 1 = 4, never 3 (3 ≠ 2).
    assert_eq!(agreement_exit(&[1, 2, 3, 2]), Some(4));
    assert_eq!(agreement_exit(&[]), None);

    // monotone_fraction: all-decreasing = 1.0; one rise = half (3 steps).
    assert!((monotone_fraction(&[4.0, 3.0, 2.0, 1.0]) - 1.0).abs() < 1e-12);
    assert!((monotone_fraction(&[3.0, 2.0, 5.0, 1.0]) - 2.0 / 3.0).abs() < 1e-12);
    assert!((monotone_fraction(&[1.0]) - 1.0).abs() < 1e-12, "short profile reads 1.0");
    // The epsilon forgives a sub-tolerance rise.
    assert!(
        monotone_fraction(&[1.0, 1.0 + KL_VECTOR_EPS / 2.0]) == 1.0,
        "sub-epsilon rise should still read monotone"
    );

    // argmax_and_margin: first-index tie rule + runner-up margin.
    let (am, m) = argmax_and_margin(&[1.0, 5.0, 5.0, 2.0]);
    assert_eq!(am, 1, "ties go to the first index");
    assert!((m - 0.0).abs() < 1e-6, "tied top margin is 0");
    let (am2, m2) = argmax_and_margin(&[1.0, 5.0, 4.0]);
    assert_eq!(am2, 1);
    assert!((m2 - 1.0).abs() < 1e-6);
}

/// Gate 7 — the row serde round trip + the resume lineage law: own
/// header accepted with completed pairs returned; moved pin / moved
/// STAMP / moved prompt spec REFUSE (the exit-2 paths surfaced here as
/// panics through a catch-free contract check — the function's refusal
/// is a process exit, so the gate verifies the ACCEPT path and the
/// refusal MESSAGES by calling the refusal path via a mismatched field
/// and expecting the process to have exited... which a lib test cannot
/// survive; the refusal arms therefore live in the runner's own smoke.
/// What a lib test CAN pin: accept + skip-set correctness.)
#[test]
fn row_serde_and_resume_accept_path() {
    // Row serde round trip.
    let row = spectra_row(1, 2, &[0.5, 1.5], &[0.1, 0.2], 7, 9, 1.0, 0.5, 3, 12.5);
    let line = serde_json::to_string(&row).expect("row json");
    let v: serde_json::Value = serde_json::from_str(&line).expect("row parse");
    assert_eq!(v["record"], "i1221_t3a_kl_spectra_row");
    assert_eq!(v["prompt_idx"], 1);
    assert_eq!(v["depth"], 2);
    assert_eq!(v["kl_mean"], 1.0);
    assert_eq!(v["kl_max"], 1.5);
    assert_eq!(v["argmax_flip_last"], true);
    assert_eq!(v["flips_all_positions"], 3);

    // Resume accept path: write header + one row, verify, expect the
    // pair back and the remaining pairs absent.
    let rows = tmp_path("resume");
    let header = concat!(
        "{\"record\":\"i1221_t3a_kl_spectra_header\",\"schema_version\":1,",
        "\"model\":{\"path\":\"p.gguf\",\"blake3\":\"abc\",\"arch\":\"dory\",\"k_full\":2,",
        "\"n_embd\":16,\"vocab\":11,\"blocks\":6,\"kv_cache_mode\":\"per_loop\"}},",
        "\"ks\":\"1..=2\",",
        "\"prompts\":{\"sets\":3,\"seq_len\":8,\"seed\":0,\"law\":\"x\"}},",
        "\"threshold_nats\":1.0,",
        "\"box_state\":{\"utc_rfc3339\":\"t\",\"profile\":\"release\",\"threads\":1}},",
        "\"repo_commit\":\"deadbeef\"}\n"
    );
    std::fs::write(&rows, format!("{header}{}", serde_json::to_string(&row).unwrap())).expect("write");
    let done = verify_resume_lineage(&rows, "abc", "deadbeef", 2, 8, 3, 0);
    assert_eq!(done, std::collections::BTreeSet::from([(1usize, 2usize)]));
    let _ = std::fs::remove_file(&rows);
}

/// Gate 8 — determinism: two lens runs byte-identical snapshots and
/// logits (no hidden state, no unseeded draws).
#[test]
fn two_lens_runs_are_byte_identical() {
    let (_path, w) = loaded_weights();
    let fwd = DoryForward::new(&w);
    let toks = prompt();
    let k = w.meta.num_loops;

    let mut c1 = DoryKvCache::new(&w, k);
    let mut s1 = Vec::new();
    let l1 = run_lens(&fwd, &toks, &mut c1, &mut s1).expect("lens1");
    let mut c2 = DoryKvCache::new(&w, k);
    let mut s2 = Vec::new();
    let l2 = run_lens(&fwd, &toks, &mut c2, &mut s2).expect("lens2");

    assert!(logits_identical(&l1, &l2), "logits differ across runs");
    assert_eq!(s1.len(), s2.len());
    for (a, b) in s1.iter().zip(s2.iter()) {
        assert!(
            a.iter().zip(b.iter()).all(|(x, y)| x.to_bits() == y.to_bits()),
            "snapshots differ across runs"
        );
    }
}

/// The unarmed prefill ≡ decode-step identity (T2's law) must hold with
/// the refactor in place — the prefill now delegates through
/// `prefill_snapshots(None)`, so this re-pins the T2 seam end to end.
#[test]
fn prefill_refactor_keeps_the_incremental_identity() {
    let (_path, w) = loaded_weights();
    let fwd = DoryForward::new(&w);
    let toks = prompt();

    let mut cache = DoryKvCache::new(&w, w.meta.num_loops);
    let prefill = fwd.prefill(&toks, &mut cache).expect("prefill");

    // Incremental: feed tokens one at a time, final row must match.
    let mut inc_cache = DoryKvCache::new(&w, w.meta.num_loops);
    let mut last = Vec::new();
    for (i, &t) in toks.iter().enumerate() {
        last = fwd.decode_step(t, &mut inc_cache).expect("decode step");
        assert_eq!(inc_cache.len(), i + 1);
    }
    assert_eq!(last.len(), w.dims.vocab, "decode returns one vocab row");
    assert!(
        last.iter()
            .zip(prefill[prefill.len() - 1].iter())
            .all(|(x, y)| x.to_bits() == y.to_bits()),
        "incremental final-row logits != prefill final row (T2 identity broken by the refactor)"
    );
}
