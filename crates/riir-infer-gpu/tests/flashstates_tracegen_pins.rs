#![cfg(feature = "flashstates_tracegen")]
//! Plan 630 / Issue 1201 T1 — the trace generator's pins: determinism
//! (same seed → byte-identical trace; different seed → different), the
//! structure contract (shared seeded preamble, growing transcripts,
//! round-robin request order, disjoint vocab bands, suffix band), the
//! A0/A0+ accounting the pre-registered bars read, and the digest pins
//! that freeze the cell ladder's inputs (a moved digest is a LOUD cell
//! invalidation, never a silent re-baseline).

use riir_infer_gpu::flashstates_tracegen::{
    PREAMBLE_TOKENS, TURN_SUFFIX_MAX, TURN_SUFFIX_MIN, generate,
};

/// The pinned cell ladder (the plan's scale axis) at seed 0 — the
/// digests freeze the generator's output; a change here is a LOUD cell
/// invalidation (re-freezing requires a plan edit naming why). Probe
/// recipe: a one-off `--example` run printing `Trace::to_string()` per
/// ladder point (the probes are ephemeral, the pins are not).
#[test]
fn digest_pins_freeze_the_cell_ladder() {
    let pins: &[(usize, usize, u64, &str)] = &[
        (
            8,
            4096,
            0,
            "blake3-a90fdcf074e25fd6b44e95eadc9982b2cd18307e0b9af52ab9a8671820183660",
        ),
        (
            16,
            8192,
            0,
            "blake3-6dcce1c26cc8aba5bc75a5c66b268ce9e68ef41f641fd549fbec6d65635264f3",
        ),
        (
            32,
            16384,
            0,
            "blake3-a956160cea90dc734589e12b2792f2692d693a0cfd4d010b77701e45c4d8c6f1",
        ),
        (
            16,
            32768,
            0,
            "blake3-4f6c995f42d4e965cf4194ca357632c25ff477bc71fbafffeeaf11133ef9e1a5",
        ),
    ];
    for &(n, cap, seed, expected) in pins {
        let t = generate(n, cap, seed);
        assert!(
            t.digest.ends_with(expected),
            "trace digest moved at n={n} cap={cap} seed={seed} — a cell input changed"
        );
    }
}

#[test]
fn determinism_same_seed_byte_identical() {
    let a = generate(8, 4096, 7);
    let b = generate(8, 4096, 7);
    assert_eq!(a, b, "same seed must reproduce the trace exactly");
    assert_eq!(a.digest, b.digest);
}

#[test]
fn different_seed_different_trace() {
    let a = generate(4, 2048, 1);
    let b = generate(4, 2048, 2);
    assert_ne!(a.digest, b.digest);
    assert_ne!(a.preamble, b.preamble, "the preamble is seeded, not fixed");
}

#[test]
fn structure_shared_preamble_growing_transcripts() {
    let t = generate(8, 4096, 0);
    assert_eq!(t.preamble.len(), PREAMBLE_TOKENS);
    assert_eq!(t.sessions.len(), 8);
    for s in &t.sessions {
        assert!(!s.turns.is_empty());
        // Growing transcript: every turn's prompt strictly extends the
        // previous turn's prompt + continuation.
        for w in s.turns.windows(2) {
            let (prev, cur) = (&w[0], &w[1]);
            assert!(cur.prompt.len() > prev.prompt.len());
            assert_eq!(&cur.prompt[..prev.prompt.len()], &prev.prompt[..]);
            let cont_start = prev.prompt.len();
            assert_eq!(
                &cur.prompt[cont_start..cont_start + prev.continuation.len()],
                &prev.continuation[..]
            );
        }
        // Every prompt opens with the shared preamble and the preamble
        // band is disjoint from the body band (the shared-prefix
        // structure is visible in the rows themselves).
        for turn in &s.turns {
            assert_eq!(&turn.prompt[..PREAMBLE_TOKENS], &t.preamble[..]);
            assert!(t.preamble.iter().all(|&tok| tok < 30_000));
            assert!(turn.continuation.iter().all(|&tok| tok >= 30_000));
        }
        // The transcript reaches the cap (a session closes only when
        // capped) — recomputed from the prompts, not trusted to the
        // generator's bookkeeping.
        let last = s.turns.last().unwrap();
        let transcript_len = last.prompt.len() - PREAMBLE_TOKENS + last.continuation.len();
        assert!(transcript_len >= t.transcript_cap_tokens);
    }
}

#[test]
fn round_robin_request_order() {
    let n = 8;
    let t = generate(n, 4096, 0);
    let min_turns = t.sessions.iter().map(|s| s.turns.len()).min().unwrap();
    assert!(min_turns >= 1);
    // The first min_turns * n requests walk sessions 0..n in order,
    // n times (full rounds before any session drops).
    let reqs: Vec<(usize, usize)> = t.requests().iter().map(|r| (r.session, r.turn)).collect();
    for round in 0..min_turns {
        for i in 0..n {
            assert_eq!(
                reqs[round * n + i],
                (i, round),
                "round-robin order broken at round {round} slot {i}"
            );
        }
    }
}

#[test]
fn accounting_a0_full_prompt_a0plus_suffix_band() {
    let t = generate(8, 4096, 0);
    let reqs = t.requests();
    let a0 = t.a0_uncached_lens();
    let a0plus = t.a0plus_uncached_lens();
    assert_eq!(a0.len(), reqs.len());
    assert_eq!(a0plus.len(), reqs.len());
    for ((r, &a0_len), &ap_len) in reqs.iter().zip(a0.iter()).zip(a0plus.iter()) {
        // A0 floor: the full prompt (fresh stream per request).
        assert_eq!(a0_len, r.prompt.len());
        // A0+ control: the fresh suffix only, inside the pinned band.
        assert!(
            (TURN_SUFFIX_MIN..=TURN_SUFFIX_MAX).contains(&ap_len),
            "A0+ uncached {ap_len} outside the pinned suffix band"
        );
        assert_eq!(ap_len, r.suffix_len);
        // And the suffix is a real suffix of the prompt (the head is
        // preamble + transcript, all cacheable).
        assert!(r.prompt.len() > ap_len);
    }
}
