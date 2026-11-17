//! i1221 T4 gates — the depth self-spec rider (k=1 draft / k_full
//! verify / greedy EXACT rejection sampling).
//!
//! The load-bearing gate is the pre-registered one: the self-spec
//! emitted stream is TOKEN-IDENTICAL to the k_full AR greedy reference
//! (per-loop caches both arms). Supporting gates pin the repair seam
//! (truncate + decode continuity), the stats invariants (the reject
//! path exercised), cache alignment at exit, and the loud refusals
//! (non-PerLoop or partial-depth verify, non-k=1 draft).
//!
//! All synth (6-block, num_loops=2, tiny vocab) — milliseconds; the
//! real-pin cell is the resumable example, never a lib test.

#![cfg(feature = "dory")]

#[path = "common/synth_bonsai2.rs"]
mod synth;
#[path = "common/synth_dory.rs"]
mod synth_dory;

use riir_infer_core::dory::self_spec::{ar_greedy, self_spec_greedy};
use riir_infer_core::dory::{DoryCachePolicy, DoryForward, DoryKvCache, load_dory_weights};
use std::sync::atomic::{AtomicU64, Ordering};

/// A pid+counter-scoped tmp name (the shared-temp-path law).
static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

fn tmp_path(tag: &str) -> std::path::PathBuf {
    let n = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("i1221_t4_{}_{}_{}", tag, std::process::id(), n))
}

fn loaded_weights() -> (std::path::PathBuf, riir_infer_core::dory::DoryWeights) {
    let bytes = synth_dory::synth_gguf("dory", true);
    let path = tmp_path("synth");
    std::fs::write(&path, &bytes).expect("write synth gguf");
    let w = load_dory_weights(&path).expect("synth dory loads");
    (path, w)
}

/// Deterministic pseudo-token prompt in [4, vocab) — the lane's law.
fn prompt(seed: u64, len: usize, vocab: usize) -> Vec<u32> {
    let mut s = seed | 1;
    (0..len)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            4 + ((s >> 33) % (vocab as u64 - 4)) as u32
        })
        .collect()
}

/// THE gate (pre-registered): the self-spec emitted stream is
/// token-identical to the k_full AR greedy reference, across prompts
/// and gamma values — and the caches exit aligned with the emitted
/// prefix.
#[test]
fn self_spec_stream_is_token_identical_to_the_ar_reference() {
    let (_p, w) = loaded_weights();
    let fwd = DoryForward::new(&w);
    let k = w.meta.num_loops;
    let vocab = w.dims.vocab;
    let n = 24usize;
    for (pi, seed) in [7u64, 0x5EED, 0xD00D].iter().enumerate() {
        for gamma in [1usize, 3, 5] {
            let toks = prompt(*seed, 6, vocab);
            let mut ref_cache = DoryKvCache::new(&w, k);
            let reference = ar_greedy(&fwd, &toks, n, &mut ref_cache).expect("ar greedy");

            let mut verify_cache = DoryKvCache::new(&w, k);
            let mut draft_cache = DoryKvCache::new(&w, 1);
            let (stream, stats) =
                self_spec_greedy(&fwd, &toks, n, gamma, &mut verify_cache, &mut draft_cache)
                    .expect("self spec");

            assert_eq!(
                stream, reference,
                "p{pi} gamma={gamma}: self-spec stream diverged from the k={k} AR reference"
            );
            assert_eq!(stats.emitted, n, "emitted cap");
            assert_eq!(
                stats.drafted,
                stats.rounds * gamma,
                "drafted == rounds*gamma"
            );
            // Cache alignment at exit: both caches carry exactly the
            // prompt + the emitted prefix (the repair law).
            assert_eq!(
                verify_cache.len(),
                toks.len() + n,
                "verify cache not aligned with the emitted prefix"
            );
            assert_eq!(
                draft_cache.len(),
                toks.len() + n,
                "draft cache not aligned with the emitted prefix"
            );
            // Per-loop footprints: verify is k-proportional, draft is
            // the k=1 posture.
            assert_eq!(verify_cache.k_loops(), k);
            assert_eq!(draft_cache.k_loops(), 1);
        }
    }
}

/// The repair seam is load-bearing: truncation drops whole positions,
/// and a fresh decode at a dropped position reproduces the clean-path
/// row for that token bit-exactly — including the self-spec repair
/// shape (junk phantom token fully replaced by the emitted target).
#[test]
fn draft_cache_truncate_keeps_decode_continuity() {
    let (_p, w) = loaded_weights();
    let fwd = DoryForward::new(&w);
    let vocab = w.dims.vocab;
    let toks = prompt(0xABCD, 8, vocab);

    // Clean path: prefill 5, decode t5 then t6; keep both rows.
    let mut plain = DoryKvCache::new(&w, 2);
    let _ = fwd.prefill(&toks[..5], &mut plain).expect("prefill");
    let plain_t5 = fwd.decode_step(toks[5], &mut plain).expect("decode");
    let plain_t6 = fwd.decode_step(toks[6], &mut plain).expect("decode");

    // Repair shape: the same prefix, a JUNK phantom at position 5,
    // then truncate (dropping the phantom wholesale) and compute the
    // real t5 fresh — every lane must match the clean row.
    let junk = (vocab - 1) as u32; // a token the clean path never used
    let mut rew = DoryKvCache::new(&w, 2);
    let _ = fwd.prefill(&toks[..5], &mut rew).expect("prefill");
    let _ = fwd.decode_step(junk, &mut rew).expect("phantom decode");
    let _ = fwd.decode_step(junk, &mut rew).expect("phantom decode");
    rew.truncate(5);
    assert_eq!(rew.len(), 5, "truncate len");
    let rew_t5 = fwd.decode_step(toks[5], &mut rew).expect("repair decode");
    let rew_t6 = fwd.decode_step(toks[6], &mut rew).expect("repair decode");
    for (tag, a, b) in [("t5", &plain_t5, &rew_t5), ("t6", &plain_t6, &rew_t6)] {
        assert_eq!(a.len(), b.len(), "{tag} row width");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{tag}: truncate+decode != clean decode at vocab lane {i}"
            );
        }
    }
}

/// The reject path must be EXERCISEABLE and the stats must add up.
/// Synth-weights caveat the census surfaces: if the k=1 draft's argmax
/// equals the k=2 target argmax at every position (loop near-no-op on
/// random small weights), every draft is accepted — the sweep proves
/// which posture the synth fixture is in, and the exact-sampling
/// reject arm stays proven by the real-pin cell (where Bench 1236's
/// spectra guarantee deep divergence).
#[test]
fn self_spec_identity_across_seeds_with_stats_census() {
    let (_p, w) = loaded_weights();
    let fwd = DoryForward::new(&w);
    let k = w.meta.num_loops;
    let vocab = w.dims.vocab;
    let n = 24usize;
    let mut any_reject = false;
    for seed in [
        7u64,
        0x5EED,
        0xD00D,
        0x1234_5678,
        99,
        123_456,
        0xDEAD_BEEF,
        42,
    ]
    .iter()
    {
        let toks = prompt(*seed, 6, vocab);
        let mut ref_cache = DoryKvCache::new(&w, k);
        let reference = ar_greedy(&fwd, &toks, n, &mut ref_cache).expect("ar");
        let mut verify_cache = DoryKvCache::new(&w, k);
        let mut draft_cache = DoryKvCache::new(&w, 1);
        let (stream, st) =
            self_spec_greedy(&fwd, &toks, n, 2, &mut verify_cache, &mut draft_cache).expect("spec");
        assert_eq!(stream, reference, "seed {seed}: identity broke");
        assert_eq!(st.emitted, n, "seed {seed}: cap");
        assert_eq!(st.drafted, st.rounds * 2, "seed {seed}: drafted");
        assert_eq!(
            verify_cache.len(),
            toks.len() + n,
            "seed {seed}: verify cache aligned"
        );
        assert_eq!(
            draft_cache.len(),
            toks.len() + n,
            "seed {seed}: draft cache aligned"
        );
        eprintln!(
            "[t4-census] seed {seed:#x}: rounds {} accepted {}/{} bonus {}",
            st.rounds, st.accepted, st.drafted, st.bonus
        );
        any_reject |= st.accepted < st.drafted;
    }
    eprintln!(
        "[t4-census] any_reject={any_reject} (false = the synth draft never misses — the reject arm is exercised by the real-pin cell)"
    );
}

/// The loud refusals: a LastLoop verify, a partial-depth verify, and a
/// non-k=1 draft all refuse before any decode work.
#[test]
fn self_spec_refuses_wrong_postures_loud() {
    let (_p, w) = loaded_weights();
    let fwd = DoryForward::new(&w);
    let k = w.meta.num_loops;
    let vocab = w.dims.vocab;
    let toks = prompt(9, 4, vocab);

    // LastLoop verify.
    let mut last_loop = DoryKvCache::with_policy(&w, k, DoryCachePolicy::LastLoop);
    let mut draft = DoryKvCache::new(&w, 1);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = self_spec_greedy(&fwd, &toks, 2, 2, &mut last_loop, &mut draft);
    }));
    assert!(r.is_err(), "LastLoop verify must refuse loud");

    // Partial-depth verify (k=1 verify on a k=2 checkpoint).
    let mut k1_verify = DoryKvCache::new(&w, 1);
    let mut draft2 = DoryKvCache::new(&w, 1);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = self_spec_greedy(&fwd, &toks, 2, 2, &mut k1_verify, &mut draft2);
    }));
    assert!(r.is_err(), "partial-depth verify must refuse loud");

    // Non-k=1 draft.
    let mut verify = DoryKvCache::new(&w, k);
    let mut k2_draft = DoryKvCache::new(&w, k);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = self_spec_greedy(&fwd, &toks, 2, 2, &mut verify, &mut k2_draft);
    }));
    assert!(r.is_err(), "non-k=1 draft must refuse loud");
}
