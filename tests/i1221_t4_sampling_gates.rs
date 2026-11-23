//! i1221 T4 sampling-arm gates (plan 638 — the Bench-634 ExactQ law on
//! the depth self-spec rider).
//!
//! The claim under test is a LAW, not a statistic: the sampled self-spec
//! stream is distributed exactly as the k_full target's own sampler.
//! The battery, pre-registered in `.plans/638` before this file existed:
//!
//! - G1 policy-level TV on a toy target (the Bench-634 form: 40k rounds,
//!   pinned seed, TV < 0.02) with a FAIL-REQUIRED greedy-verify control;
//! - G2 the induced law equals the target — deterministic, on random
//!   (p, q) pairs and on every pair a forward rollout captures;
//! - G3 E2E identity THROUGH THE REAL FORWARD — `#[ignore]`, release:
//!   N is sized for the bar, and the temperature is pinned by a rule
//!   (`ladder_temperature`), never by a search for a passing statistic;
//! - G4 determinism (a pinned seed replays bit-identically), the counter
//!   identities, cache alignment (the cap's overshoot included), and the
//!   reject/repair path exercised by a LONG STREAM with a cache-state
//!   receipt;
//! - G5 the loud posture refusals.
//!
//! Fixture honesty (plan 638's adjudication): the stock synth's target
//! is a POINT MASS at every ladder temperature, and its loop is a
//! near-no-op (p ≈ q, KL ≈ 8e-4 nats on the entropy-bearing variant) —
//! no post-load rescale makes the two depths diverge. So here G3 proves
//! the full path emits the TARGET law (not a point mass, not a broken
//! repair) and G4 proves the repair machinery under real rejects; the
//! divergent-(p, q) E2E identity is G1 + G2 + the real-pin receipt.
//! Every forward cell prints its KL(p‖q) so the record states what it
//! could see.

#![cfg(feature = "dory")]

#[path = "common/synth_bonsai2.rs"]
mod synth;
#[path = "common/synth_dory.rs"]
mod synth_dory;

use riir_infer_core::dory::self_spec::{
    SampleCapture, SampleOpts, SelfSpecSampleStats, SpecRng, ar_sample, induced_into,
    sample_categorical, self_spec_greedy_verify_control, self_spec_sample, softmax_f64_into,
    spec_step,
};
use riir_infer_core::dory::{
    DoryCachePolicy, DoryForward, DoryKvCache, DoryWeights, load_dory_weights,
};
use std::sync::atomic::{AtomicU64, Ordering};

/// A pid+counter-scoped tmp name (the shared-temp-path law).
static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

fn tmp_path(tag: &str) -> std::path::PathBuf {
    let n = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("i1221_t4s_{}_{}_{}", tag, std::process::id(), n))
}

fn loaded_weights() -> DoryWeights {
    let bytes = synth_dory::synth_gguf("dory", true);
    let path = tmp_path("synth");
    std::fs::write(&path, &bytes).expect("write synth gguf");
    let w = load_dory_weights(&path).expect("synth dory loads");
    let _ = std::fs::remove_file(&path);
    w
}

/// The entropy-bearing variant (plan 638's adjudication): the stock
/// fixture with `lm_head × 2^-16` after load — the smallest power of two
/// bringing max|logit| under 100 (the stock head saturates: logits
/// 3.9e5…5.8e6, a point-mass target). The forward is untouched; the
/// identity is weight-agnostic.
fn entropy_bearing(mut w: DoryWeights) -> DoryWeights {
    for x in w.lm_head.iter_mut() {
        *x *= 1.0 / 65536.0;
    }
    w
}

/// The pre-registered temperature rule: the smallest T in
/// {1, 2, 4, 8, 16} at which the first-position target row has
/// p_max ≤ 0.75 (the identity is temperature-invariant; the control's
/// TV = 1 − p_max is then ≥ 0.25). `None` = no qualifying T ⇒ UNDECIDABLE
/// on this fixture (reported, never a silent pass — the ladder is never
/// widened to make a fixture qualify).
fn ladder_temperature(vrow: &[f32]) -> Option<f64> {
    let mut p = vec![0.0f64; vrow.len()];
    [1.0f64, 2.0, 4.0, 8.0, 16.0].into_iter().find(|&t| {
        softmax_f64_into(vrow, 1.0 / t, &mut p);
        p.iter().copied().fold(0.0, f64::max) <= 0.75
    })
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

/// f64 softmax of f64 logits (sequential, ascending — the instrument's
/// fold order).
fn softmax_of(z: &[f64]) -> Vec<f64> {
    let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut out: Vec<f64> = z.iter().map(|&v| (v - m).exp()).collect();
    let s = out.iter().sum::<f64>();
    for v in &mut out {
        *v /= s;
    }
    out
}

/// Total variation between an empirical histogram and a law.
fn tv(hist: &[u64], p: &[f64]) -> f64 {
    let n = hist.iter().sum::<u64>() as f64;
    hist.iter()
        .zip(p.iter())
        .map(|(&c, &pi)| (c as f64 / n - pi).abs())
        .sum::<f64>()
        * 0.5
}

/// KL(p ‖ q) in nats, sequential f64.
fn kl(p: &[f64], q: &[f64]) -> f64 {
    p.iter()
        .zip(q.iter())
        .filter(|&(a, _)| *a > 0.0)
        .map(|(&a, &b)| {
            if b > 0.0 {
                a * (a / b).ln()
            } else {
                f64::INFINITY
            }
        })
        .sum()
}

/// The G2 bar, derived from the row length (the sequential-fold noise
/// scale), never from a reading.
fn induced_bar(vocab: usize) -> f64 {
    f64::max(1e-12, 4.0 * vocab as f64 * f64::EPSILON)
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0, f64::max)
}

// ───────────────────────────────── G1 ─────────────────────────────────

const TOY_V: usize = 32;
const G1_ROUNDS: usize = 40_000;
const G1_SEED: u64 = 0x634_BEC4;

/// The toy target: one strong token over a flat tail (p_max ≈ 0.516 —
/// entropy-bearing, so the control's point mass reads TV ≈ 0.48).
fn toy_target() -> Vec<f64> {
    softmax_of(
        &(0..TOY_V)
            .map(|i| if i == 0 { 3.5 } else { 0.0 })
            .collect::<Vec<_>>(),
    )
}

/// Run the pure judge step `rounds` times with drafts drawn from `q`.
fn g1_arm(p: &[f64], q: &[f64], rounds: usize, seed: u64) -> (Vec<u64>, usize) {
    let mut rng = SpecRng::new(seed);
    let mut hist = vec![0u64; p.len()];
    let mut scratch = vec![0.0f64; p.len()];
    let mut accepted = 0usize;
    for _ in 0..rounds {
        let t = sample_categorical(q, rng.next_f64());
        let (x, acc) = spec_step(p, q, t, rng.next_f64(), rng.next_f64(), &mut scratch);
        hist[x] += 1;
        accepted += usize::from(acc);
    }
    (hist, accepted)
}

/// G1 — the Bench-634 policy-level bar: the pure step's emitted
/// histogram is the target (TV < 0.02 at 40k rounds, pinned seed), the
/// observed accept rate is the law's Σ min(p, q), and the greedy-verify
/// control FAILS exactness by construction (TV = 1 − p_max > 0.30).
#[test]
fn g1_policy_level_tv_matches_the_bench_634_bar() {
    let p = toy_target();
    let uniform = 1.0 / TOY_V as f64;
    let mixture: Vec<f64> = p.iter().map(|&a| 0.3 * a + 0.7 * uniform).collect();
    let displaced = softmax_of(
        &(0..TOY_V)
            .map(|i| if i == 7 { 3.5 } else { 0.0 })
            .collect::<Vec<_>>(),
    );
    assert!(
        (0.50..=0.55).contains(&p[0]),
        "toy p_max {} (want ≈ 0.52)",
        p[0]
    );

    for (name, q, seed) in [
        ("mixture", mixture.as_slice(), G1_SEED),
        ("displaced", displaced.as_slice(), G1_SEED ^ 0x11),
    ] {
        let (hist, accepted) = g1_arm(&p, q, G1_ROUNDS, seed);
        let tv_spec = tv(&hist, &p);
        let law_accept: f64 = p.iter().zip(q.iter()).map(|(&a, &b)| a.min(b)).sum();
        let observed = accepted as f64 / G1_ROUNDS as f64;
        eprintln!(
            "[g1] {name}: TV(spec, p) = {tv_spec:.5} · accept {observed:.4} (law Σmin(p,q) = {law_accept:.4}) · residual rounds {:.3}",
            1.0 - observed
        );
        assert!(
            tv_spec < 0.02,
            "{name}: spec TV {tv_spec:.5} ≥ 0.02 — THE LAW FAILED"
        );
        assert!(
            (observed - law_accept).abs() < 0.03,
            "{name}: accept rate {observed:.4} vs the law's {law_accept:.4}"
        );
        assert!(
            (0.4..0.9).contains(&law_accept),
            "{name}: the toy proposal must exercise the reject path (law accept {law_accept:.3})"
        );
    }

    // The floor: a DIRECT sampler at the same N sits equally low (the
    // bar is reachable — it is not an N too small to pass).
    let mut rng = SpecRng::new(G1_SEED);
    let mut direct = vec![0u64; TOY_V];
    for _ in 0..G1_ROUNDS {
        direct[sample_categorical(&p, rng.next_f64())] += 1;
    }
    let tv_direct = tv(&direct, &p);
    eprintln!("[g1] direct sampler: TV = {tv_direct:.5} (the floor)");
    assert!(
        tv_direct < 0.02,
        "direct sampler TV {tv_direct:.5} — the toy cannot meet its own bar"
    );

    // The fail-required control: a greedy judge of sampled drafts emits
    // the target argmax every round — a point mass.
    let mut control = vec![0u64; TOY_V];
    control[0] = G1_ROUNDS as u64; // argmax(p) — whatever was drafted
    let tv_control = tv(&control, &p);
    eprintln!("[g1] greedy-verify control: TV = {tv_control:.5} (= 1 − p_max)");
    assert!(
        tv_control > 0.30,
        "the control did NOT fail exactness (TV {tv_control:.5}) — the battery has no teeth"
    );
}

// ───────────────────────────────── G2 ─────────────────────────────────

/// G2 — the induced law IS the target, deterministically: on seeded
/// random pairs (plus the identical and one-hot-proposal edges) and on
/// every (p, q) pair a real forward rollout captures.
#[test]
fn g2_induced_law_is_exact_on_random_and_forward_rows() {
    // Random pairs: softmaxes of seeded logits (V = 64).
    let v = 64usize;
    let mut worst = 0.0f64;
    let bar = induced_bar(v);
    let mut rng = SpecRng::new(0x1D_CE_D5);
    let mut scratch = vec![0.0f64; v];
    let mut induced = vec![0.0f64; v];
    for pair in 0..200 {
        let p = softmax_of(
            &(0..v)
                .map(|_| (rng.next_f64() - 0.5) * 8.0)
                .collect::<Vec<_>>(),
        );
        let mut q_logits: Vec<f64> = (0..v).map(|_| (rng.next_f64() - 0.5) * 8.0).collect();
        if pair % 2 == 0 {
            // Half the proposals cannot reach a quarter of the tokens —
            // the residual carries everything the proposal cannot.
            for (i, zi) in q_logits.iter_mut().enumerate() {
                if i % 4 == pair % 8 {
                    *zi = -60.0;
                }
            }
        }
        let q = softmax_of(&q_logits);
        induced_into(&p, &q, &mut induced, &mut scratch);
        worst = worst.max(max_abs_diff(&induced, &p));
    }
    // Edges: an identical proposal (reject mass exactly 0) and a one-hot
    // proposal (every other token reaches the output only by residual).
    let mut p = vec![0.0f64; v];
    p[5] = 0.25;
    p[9] = 0.75;
    induced_into(&p, &p, &mut induced, &mut scratch);
    worst = worst.max(max_abs_diff(&induced, &p));
    let mut one_hot = vec![0.0f64; v];
    one_hot[3] = 1.0;
    induced_into(&p, &one_hot, &mut induced, &mut scratch);
    worst = worst.max(max_abs_diff(&induced, &p));
    eprintln!("[g2] random + edge pairs (V={v}): max |induced − p| = {worst:.3e} (bar {bar:.1e})");
    assert!(
        worst <= bar,
        "induced law deviates from the target by {worst:.3e} > {bar:.1e}"
    );

    // Forward rows: every pair the real forward's rollouts capture, on
    // the stock fixture and the entropy-bearing variant (the latter at
    // the ladder's T = 8, where the rows carry mass).
    let vocab = synth_dory::VOCAB as usize;
    let bar = induced_bar(vocab);
    let mut scratch = vec![0.0f64; vocab];
    let mut induced = vec![0.0f64; vocab];
    for (name, w, t) in [
        ("stock", loaded_weights(), 1.0),
        ("entropy", entropy_bearing(loaded_weights()), 8.0),
    ] {
        let fwd = DoryForward::new(&w);
        let k = w.meta.num_loops;
        let mut worst = 0.0f64;
        let mut pairs = 0usize;
        let mut kl_sum = 0.0f64;
        for seed in [7u64, 0x5EED, 0xD00D] {
            for gamma in [1usize, 3, 5] {
                let toks = prompt(seed, 5, vocab);
                let mut verify_cache = DoryKvCache::new(&w, k);
                let mut draft_cache = DoryKvCache::new(&w, 1);
                let mut cap = SampleCapture::default();
                let mut rng = SpecRng::new(seed.wrapping_mul(31).wrapping_add(gamma as u64));
                let mut opts = SampleOpts {
                    temperature: t,
                    rng: &mut rng,
                    capture: Some(&mut cap),
                };
                self_spec_sample(
                    &fwd,
                    &toks,
                    24,
                    gamma,
                    &mut verify_cache,
                    &mut draft_cache,
                    &mut opts,
                )
                .expect("self spec sample");
                for (pp, qq) in cap.pairs() {
                    assert_eq!(pp.len(), vocab);
                    induced_into(pp, qq, &mut induced, &mut scratch);
                    worst = worst.max(max_abs_diff(&induced, pp));
                    kl_sum += kl(pp, qq);
                    pairs += 1;
                }
            }
        }
        eprintln!(
            "[g2:{name}] forward rows: {pairs} pairs, max |induced − p| = {worst:.3e} (bar {bar:.1e}) · mean KL(p‖q) = {:.5} nats",
            kl_sum / pairs as f64
        );
        assert!(
            pairs >= 3 * 3 * 12,
            "captured {pairs} judge pairs (want ≥ 3 seeds × 3 γ × 12)"
        );
        assert!(
            worst <= bar,
            "induced law on forward rows deviates by {worst:.3e} > {bar:.1e}"
        );
    }
}

// ─────────────────────────── diagnostics ───────────────────────────

/// The target row's mass structure at the ladder's temperatures — the
/// input G3's rule reads (report-only; asserts only that the rows are
/// distributions). Both fixtures, so the record states what each can
/// see (the stock head saturates; the entropy-bearing variant carries
/// mass from T = 8; neither depth pair diverges — the loop's near-no-op).
#[test]
fn synth_target_entropy_census() {
    for (name, w) in [
        ("stock", loaded_weights()),
        ("entropy", entropy_bearing(loaded_weights())),
    ] {
        let fwd = DoryForward::new(&w);
        let k = w.meta.num_loops;
        let vocab = w.dims.vocab;
        let toks = prompt(0xCE455, 3, vocab);
        let mut verify_cache = DoryKvCache::new(&w, k);
        let mut draft_cache = DoryKvCache::new(&w, 1);
        let vrow = fwd
            .prefill(&toks, &mut verify_cache)
            .expect("prefill")
            .pop()
            .expect("row");
        let drow = fwd
            .prefill(&toks, &mut draft_cache)
            .expect("prefill")
            .pop()
            .expect("row");
        for t in [1.0f64, 2.0, 4.0, 8.0, 16.0] {
            let mut p = vec![0.0f64; vocab];
            let mut q = vec![0.0f64; vocab];
            softmax_f64_into(&vrow, 1.0 / t, &mut p);
            softmax_f64_into(&drow, 1.0 / t, &mut q);
            eprintln!(
                "[census:{name}] T={t:>4}: p_max = {:.4} · p_max(q) = {:.4} · KL(p‖q) = {:.6} nats",
                p.iter().copied().fold(0.0, f64::max),
                q.iter().copied().fold(0.0, f64::max),
                kl(&p, &q)
            );
            assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            assert!((q.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        }
        eprintln!("[census:{name}] logits (target) = {vrow:?}");
    }
}

// ───────────────────────────────── G3 ─────────────────────────────────

/// The entropy-bearing fixture's first-position target law at the
/// rule's temperature: (weights, T, p_true, prompt).
fn g3_posture() -> (DoryWeights, f64, Vec<f64>, Vec<u32>) {
    let w = entropy_bearing(loaded_weights());
    let fwd = DoryForward::new(&w);
    let k = w.meta.num_loops;
    let vocab = w.dims.vocab;
    let toks = prompt(0xE2E, 3, vocab);
    let mut probe = DoryKvCache::new(&w, k);
    let vrow = fwd
        .prefill(&toks, &mut probe)
        .expect("prefill")
        .pop()
        .expect("row");
    let Some(t) = ladder_temperature(&vrow) else {
        panic!(
            "G3 UNDECIDABLE: no T in the ladder brings the entropy-bearing target's p_max to ≤ 0.75 (p_max(T=16) = {})",
            {
                let mut p = vec![0.0f64; vocab];
                softmax_f64_into(&vrow, 1.0 / 16.0, &mut p);
                p.iter().copied().fold(0.0, f64::max)
            }
        );
    };
    let p_true = {
        let mut p = vec![0.0f64; vocab];
        softmax_f64_into(&vrow, 1.0 / t, &mut p);
        p
    };
    (w, t, p_true, toks)
}

/// G3 — E2E identity THROUGH THE REAL FORWARD (synth 6-block, k=2, the
/// entropy-bearing variant at the ladder's T): the first emitted token's
/// histogram over N full-path draws (fresh caches every draw) is the
/// target row's law. `#[ignore]`: N is sized for the bar — run in
/// release:
///
/// ```text
/// cargo test --release --features dory --test i1221_t4_sampling_gates \
///   g3_e2e -- --ignored --nocapture
/// ```
#[test]
#[ignore = "release-only cell: N = 200,000 full-path draws per arm (plan 638 G3)"]
fn g3_e2e_identity_through_the_real_forward() {
    const N: usize = 200_000;
    let (w, t, p_true, toks) = g3_posture();
    let fwd = DoryForward::new(&w);
    let k = w.meta.num_loops;
    let vocab = w.dims.vocab;
    let p_max = p_true.iter().copied().fold(0.0, f64::max);
    eprintln!(
        "[g3] posture: T={t} (the ladder rule) · p_max = {p_max:.4} · the control's TV by construction = {:.4}",
        1.0 - p_max
    );

    // Spec arms: fresh caches per draw, one emitted token, the full
    // round machinery (draft → lazy judge → bonus/repair).
    for gamma in [1usize, 4] {
        let mut rng = SpecRng::new(0xC0FFEE);
        let mut hist = vec![0u64; vocab];
        let mut rejected_total = 0usize;
        for _ in 0..N {
            let mut verify_cache = DoryKvCache::new(&w, k);
            let mut draft_cache = DoryKvCache::new(&w, 1);
            let mut opts = SampleOpts {
                temperature: t,
                rng: &mut rng,
                capture: None,
            };
            let (stream, st) = self_spec_sample(
                &fwd,
                &toks,
                1,
                gamma,
                &mut verify_cache,
                &mut draft_cache,
                &mut opts,
            )
            .expect("self spec sample");
            hist[stream[0] as usize] += 1;
            rejected_total += st.rejected;
        }
        let tv_spec = tv(&hist, &p_true);
        eprintln!(
            "[g3] spec γ={gamma}: TV(spec, p_true) = {tv_spec:.5} · rejected {rejected_total}/{N} first-token draws",
        );
        assert!(
            tv_spec < 0.02,
            "G3 spec γ={gamma}: TV {tv_spec:.5} ≥ 0.02 — THE E2E LAW FAILED"
        );
    }

    // The reference, full path too: `ar_sample` on fresh caches.
    let mut rng = SpecRng::new(0xC0FFEE ^ 0x55);
    let mut hist = vec![0u64; vocab];
    for _ in 0..N {
        let mut cache = DoryKvCache::new(&w, k);
        let stream = ar_sample(&fwd, &toks, 1, t, &mut cache, &mut rng).expect("ar sample");
        hist[stream[0] as usize] += 1;
    }
    let tv_ref = tv(&hist, &p_true);
    eprintln!("[g3] reference (ar_sample, full path): TV(ref, p_true) = {tv_ref:.5} (the floor)");
    assert!(
        tv_ref < 0.02,
        "G3 reference TV {tv_ref:.5} — N cannot meet its own bar"
    );

    // The fail-required control through the real machinery. The control
    // judges by the target row's ARGMAX, which the temperature does not
    // move — its TV is 1 − p_max(T) by construction.
    let mut rng = SpecRng::new(0xC0FFEE);
    let mut hist = vec![0u64; vocab];
    for _ in 0..N {
        let mut verify_cache = DoryKvCache::new(&w, k);
        let mut draft_cache = DoryKvCache::new(&w, 1);
        let (stream, _st) = self_spec_greedy_verify_control(
            &fwd,
            &toks,
            1,
            4,
            &mut verify_cache,
            &mut draft_cache,
            &mut rng,
        )
        .expect("control");
        hist[stream[0] as usize] += 1;
    }
    let tv_control = tv(&hist, &p_true);
    eprintln!(
        "[g3] greedy-verify control: TV = {tv_control:.5} (1 − p_max = {:.4})",
        1.0 - p_max
    );
    assert!(
        tv_control > 0.10,
        "G3: the control did NOT fail (TV {tv_control:.5}) — the battery has no teeth"
    );
}

// ───────────────────────────────── G4 ─────────────────────────────────

type Run = (Vec<u32>, SelfSpecSampleStats, usize, usize);

fn sample_run(
    fwd: &DoryForward<'_>,
    w: &DoryWeights,
    toks: &[u32],
    n: usize,
    gamma: usize,
    t: f64,
    seed: u64,
) -> Run {
    let mut verify_cache = DoryKvCache::new(w, fwd.k_loops());
    let mut draft_cache = DoryKvCache::new(w, 1);
    let mut rng = SpecRng::new(seed);
    let mut opts = SampleOpts {
        temperature: t,
        rng: &mut rng,
        capture: None,
    };
    let (stream, st) = self_spec_sample(
        fwd,
        toks,
        n,
        gamma,
        &mut verify_cache,
        &mut draft_cache,
        &mut opts,
    )
    .expect("self spec sample");
    (stream, st, verify_cache.len(), draft_cache.len())
}

/// The counter identities of one run (the sampling arm's cost law) and
/// its exit alignment.
fn assert_identities(
    st: &SelfSpecSampleStats,
    gamma: usize,
    n: usize,
    prompt_len: usize,
    verify_len: usize,
    draft_len: usize,
    tag: &str,
) {
    let b = &st.base;
    assert_eq!(b.emitted, n, "{tag}: emitted cap");
    assert_eq!(b.drafted, b.rounds * gamma, "{tag}: drafted == rounds·γ");
    assert_eq!(
        b.accepted + st.rejected + b.bonus,
        st.emitted_uncapped,
        "{tag}: accepted + rejected + bonus == emitted_uncapped"
    );
    assert_eq!(
        b.verify_steps,
        b.accepted + b.rounds,
        "{tag}: verify_steps == accepted + rounds"
    );
    assert_eq!(
        b.draft_steps,
        b.rounds * gamma + b.bonus,
        "{tag}: draft_steps == rounds·γ + bonus"
    );
    assert_eq!(
        st.rejects_by_pos.iter().sum::<usize>(),
        st.rejected,
        "{tag}: rejects_by_pos sums"
    );
    assert_eq!(
        st.rejects_by_pos.len(),
        gamma,
        "{tag}: rejects_by_pos width"
    );
    // γ draft draws + 2 judge draws per position actually JUDGED (a
    // reject ends the judge loop early), + 1 per bonus.
    assert_eq!(
        st.uniforms,
        b.drafted + 2 * (b.accepted + st.rejected) + b.bonus,
        "{tag}: uniforms contract"
    );
    assert!(st.emitted_uncapped >= n, "{tag}: a run reaches the cap");
    assert_eq!(
        verify_len,
        prompt_len + n,
        "{tag}: verify cache exits at prompt + n"
    );
    assert_eq!(
        draft_len,
        prompt_len + n,
        "{tag}: draft cache exits at prompt + n"
    );
}

/// G4 — determinism, the counter identities, cache alignment (the cap's
/// overshoot included), different seeds differing. Milliseconds.
#[test]
fn g4_sampling_arm_is_deterministic_and_counters_add_up() {
    let w = entropy_bearing(loaded_weights());
    let fwd = DoryForward::new(&w);
    let k = w.meta.num_loops;
    let vocab = w.dims.vocab;
    let t = 8.0; // the ladder rule's pick for this fixture (the census pins it)
    let mut any_overshoot = false;
    let mut rejects_in_sweep = 0usize;

    for seed in [
        7u64,
        0x5EED,
        0xD00D,
        0x1234_5678,
        99,
        123_456,
        0xDEAD_BEEF,
        42,
    ] {
        for gamma in [1usize, 3, 5] {
            // n deliberately NOT a multiple of γ+1: rounds overshoot.
            let n = 10usize;
            let toks = prompt(seed, 5, vocab);
            let s = seed.wrapping_mul(131).wrapping_add(gamma as u64);
            let r1 = sample_run(&fwd, &w, &toks, n, gamma, t, s);
            let r2 = sample_run(&fwd, &w, &toks, n, gamma, t, s);
            assert_eq!(
                r1.0, r2.0,
                "seed {seed:#x} γ={gamma}: streams differ across passes at one seed"
            );
            assert_eq!(
                format!("{:?}", r1.1),
                format!("{:?}", r2.1),
                "counters differ across passes"
            );
            let tag = format!("seed {seed:#x} γ={gamma}");
            assert_identities(&r1.1, gamma, n, toks.len(), r1.2, r1.3, &tag);
            any_overshoot |= r1.1.emitted_uncapped > n;
            rejects_in_sweep += r1.1.rejected;
        }
    }
    assert!(
        any_overshoot,
        "the cap overshoot was never exercised across the sweep"
    );
    eprintln!(
        "[g4] sweep: {rejects_in_sweep} reject(s) realized across 24 runs (pinned seeds — deterministic)"
    );
    // Deterministic at the pinned seeds (3 realized on the first run);
    // the robust exercise with its receipt is the release-only G4b.
    assert!(
        rejects_in_sweep >= 1,
        "the sweep never rejected — the repair path is unexercised at these seeds"
    );

    // Different seeds give different streams.
    let toks = prompt(0xD1FF, 5, vocab);
    let stream_at = |seed: u64| sample_run(&fwd, &w, &toks, 24, 2, t, seed).0;
    assert_ne!(
        stream_at(1),
        stream_at(2),
        "different seeds must give different streams"
    );

    // The reference sampler replays at one seed and differs across seeds.
    let ref_at = |seed: u64| {
        let mut cache = DoryKvCache::new(&w, k);
        let mut rng = SpecRng::new(seed);
        ar_sample(&fwd, &toks, 16, t, &mut cache, &mut rng).expect("ar sample")
    };
    assert_eq!(
        ref_at(3),
        ref_at(3),
        "ar_sample must replay bit-identically at one seed"
    );
    assert_ne!(ref_at(3), ref_at(4), "ar_sample must differ across seeds");
}

/// G4b — the reject/repair path REALLY exercised: a 20,000-token stream
/// at γ = 3 (≈ 1.3e-2 rejects per judged token measured ⇒ ≈ 200
/// rejects, P(0) ≈ 1e-87), the counter identities held across it, and
/// the cache-state receipt. `#[ignore]`: the stream is sized for the
/// exercise, not for a debug run — release:
///
/// ```text
/// cargo test --release --features dory --test i1221_t4_sampling_gates \
///   g4b_long_stream -- --ignored --nocapture
/// ```
#[test]
#[ignore = "release-only cell: a 20,000-token stream (plan 638, the long-stream exercise)"]
fn g4b_long_stream_reject_repair_and_cache_receipt() {
    let w = entropy_bearing(loaded_weights());
    let fwd = DoryForward::new(&w);
    let k = w.meta.num_loops;
    let vocab = w.dims.vocab;
    let t = 8.0;

    let n = 20_000usize;
    let gamma = 3usize;
    let toks = prompt(0x10E6, 4, vocab);
    let seed = 0x10E6_5EED;
    let (stream, st, v_len, d_len) = sample_run(&fwd, &w, &toks, n, gamma, t, seed);
    eprintln!(
        "[g4b] long stream: n={n} γ={gamma} T={t}: rounds {} accepted {} REJECTED {} (by pos {:?}) bonus {} · rejects per judged token = {:.2e}",
        st.base.rounds,
        st.base.accepted,
        st.rejected,
        st.rejects_by_pos,
        st.base.bonus,
        st.rejected as f64 / st.base.drafted as f64
    );
    assert!(
        st.rejected >= 100,
        "the long stream rejected only {} — the repair path is under-exercised",
        st.rejected
    );
    assert_identities(&st, gamma, n, toks.len(), v_len, d_len, "long stream");

    // The cache-state receipt: the verify cache the stream leaves behind
    // is, token for token, the reference's. Decode one more token on the
    // spec arm's cache and on a fresh k_full prefill of the same
    // sequence — the two rows must be bit-identical (the lazy verify +
    // repair left exactly the reference's state).
    let mut after = DoryKvCache::new(&w, k);
    let mut draft_after = DoryKvCache::new(&w, 1);
    let mut rng = SpecRng::new(seed);
    let mut opts = SampleOpts {
        temperature: t,
        rng: &mut rng,
        capture: None,
    };
    let (stream_again, _) = self_spec_sample(
        &fwd,
        &toks,
        n,
        gamma,
        &mut after,
        &mut draft_after,
        &mut opts,
    )
    .expect("self spec sample (receipt rerun)");
    assert_eq!(
        stream_again, stream,
        "the receipt rerun must reproduce the stream"
    );
    let probe_token = (vocab / 2) as u32;
    let spec_next = fwd
        .decode_step(probe_token, &mut after)
        .expect("decode after the spec stream");
    let mut reference = DoryKvCache::new(&w, k);
    let mut full: Vec<u32> = toks.clone();
    full.extend_from_slice(&stream);
    let _ = fwd
        .prefill(&full, &mut reference)
        .expect("reference prefill");
    let reference_next = fwd
        .decode_step(probe_token, &mut reference)
        .expect("reference decode");
    assert_eq!(spec_next.len(), reference_next.len());
    for (i, (a, b)) in spec_next.iter().zip(reference_next.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "cache-state receipt: the verify cache after the stream differs from the reference at vocab lane {i}"
        );
    }
}

// ───────────────────────────────── G5 ─────────────────────────────────

/// G5 — the loud refusals: a LastLoop verify, a partial-depth verify, a
/// non-k=1 draft, and a non-positive temperature all refuse before any
/// decode work.
#[test]
fn g5_sampling_arm_refuses_wrong_postures_loud() {
    let w = loaded_weights();
    let fwd = DoryForward::new(&w);
    let k = w.meta.num_loops;
    let vocab = w.dims.vocab;
    let toks = prompt(9, 4, vocab);
    let refuses = |f: &dyn Fn()| std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).is_err();

    // LastLoop verify.
    assert!(
        refuses(&|| {
            let mut last_loop = DoryKvCache::with_policy(&w, k, DoryCachePolicy::LastLoop);
            let mut draft = DoryKvCache::new(&w, 1);
            let mut rng = SpecRng::new(1);
            let mut opts = SampleOpts {
                temperature: 1.0,
                rng: &mut rng,
                capture: None,
            };
            let _ = self_spec_sample(&fwd, &toks, 2, 2, &mut last_loop, &mut draft, &mut opts);
        }),
        "LastLoop verify must refuse loud"
    );
    // Partial-depth verify.
    assert!(
        refuses(&|| {
            let mut k1_verify = DoryKvCache::new(&w, 1);
            let mut draft = DoryKvCache::new(&w, 1);
            let mut rng = SpecRng::new(1);
            let mut opts = SampleOpts {
                temperature: 1.0,
                rng: &mut rng,
                capture: None,
            };
            let _ = self_spec_sample(&fwd, &toks, 2, 2, &mut k1_verify, &mut draft, &mut opts);
        }),
        "partial-depth verify must refuse loud"
    );
    // Non-k=1 draft.
    assert!(
        refuses(&|| {
            let mut verify = DoryKvCache::new(&w, k);
            let mut k2_draft = DoryKvCache::new(&w, k);
            let mut rng = SpecRng::new(1);
            let mut opts = SampleOpts {
                temperature: 1.0,
                rng: &mut rng,
                capture: None,
            };
            let _ = self_spec_sample(&fwd, &toks, 2, 2, &mut verify, &mut k2_draft, &mut opts);
        }),
        "non-k=1 draft must refuse loud"
    );
    // Non-positive temperature, both arms.
    assert!(
        refuses(&|| {
            let mut verify = DoryKvCache::new(&w, k);
            let mut draft = DoryKvCache::new(&w, 1);
            let mut rng = SpecRng::new(1);
            let mut opts = SampleOpts {
                temperature: 0.0,
                rng: &mut rng,
                capture: None,
            };
            let _ = self_spec_sample(&fwd, &toks, 2, 2, &mut verify, &mut draft, &mut opts);
        }),
        "temperature 0 must refuse loud"
    );
    assert!(
        refuses(&|| {
            let mut cache = DoryKvCache::new(&w, k);
            let mut rng = SpecRng::new(1);
            let _ = ar_sample(&fwd, &toks, 2, f64::NAN, &mut cache, &mut rng);
        }),
        "a NaN temperature must refuse loud"
    );
}
