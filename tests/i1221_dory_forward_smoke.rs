//! Issue 1221 T1 — the k-loop smoke generation on the REAL pinned
//! checkpoint (the T1 proof line's "k=1 smoke generation").
//!
//! Assertions are STRUCTURAL, never quality claims (no tokenizer is
//! wired for the lane yet; the paper's benchmark numbers are too coarse
//! for byte-parity — the issue's oracle-gap risk): determinism (two
//! passes bit-identical), the k=1 identity on loaded weights, the
//! per-loop cache footprint growing with k, finite/bounded logits, and
//! k=2 differing from k=1 (the loop deepens on the real weights).
//! Coherence/quality adjudication waits for the T2/T3 directional
//! cells; an incoherent stream here would point at the named reading
//! decisions in `src/dory/forward.rs` (rope row order first).
//!
//! Heavy by design: ~17 GB RSS (the f32 dequant snapshot posture) and
//! minutes of CPU forward — `#[ignore]`d so no default gate run picks
//! it up. Run explicitly:
//!
//! ```sh
//! cargo test --release --features dory --test i1221_dory_forward_smoke -- --ignored --nocapture
//! ```
//!
//! Skip-LOUD posture still applies when the pin is absent
//! (`I1221_DORY_GGUF` env, else the riir-train data pin).

#![cfg(feature = "dory")]

use riir_infer_core::dory::{DoryCachePolicy, DoryForward, DoryKvCache, load_dory_weights};

const PIN_FILE: &str = "NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf";

fn pin_path() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("I1221_DORY_GGUF") {
        let p = std::path::PathBuf::from(p);
        return if p.exists() { Some(p) } else { None };
    }
    for base in ["../riir-train/data", "../../riir-train/data"] {
        let p = std::path::PathBuf::from(base).join(PIN_FILE);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

#[test]
#[ignore = "heavy cell: ~17 GB RSS + minutes of CPU forward — run with -- --ignored"]
fn smoke_generation_on_the_pinned_checkpoint() {
    let Some(path) = pin_path() else {
        panic!(
            "SKIP LOUD: the Dory pin is absent (I1221_DORY_GGUF or \
             ../riir-train/data/{PIN_FILE}); restore via the riir-train \
             data_backup lane"
        );
    };
    let t0 = std::time::Instant::now();
    let w = load_dory_weights(&path).expect("the pin loads (arch dory)");
    eprintln!(
        "[smoke] loaded in {:.1}s — {:?}",
        t0.elapsed().as_secs_f32(),
        w.dims
    );

    let k_meta = w.meta.num_loops;
    eprintln!(
        "[smoke] meta: num_loops={k_meta} kv_mode={} recurrent={}",
        w.meta.kv_cache_mode, w.meta.recurrent_layers
    );

    // A fixed prompt of in-range token ids (no tokenizer wired — the
    // structural assertions below are id-agnostic).
    let toks: Vec<u32> = (0..8)
        .map(|i| (i * 15485863 + 11) % w.dims.vocab as u32)
        .collect();

    // k=1: prefill vs incremental identity on the REAL weights.
    let f = DoryForward::new(&w);
    let mut c = DoryKvCache::new(&w, 1);
    let t1 = std::time::Instant::now();
    let prefill = f.prefill(&toks, &mut c).expect("prefill k=1");
    eprintln!(
        "[smoke] k=1 prefill {} tok in {:.1}s ({:.2} s/tok)",
        toks.len(),
        t1.elapsed().as_secs_f32(),
        t1.elapsed().as_secs_f32() / toks.len() as f32
    );

    let mut c2 = DoryKvCache::new(&w, 1);
    let t2 = std::time::Instant::now();
    let mut incremental = Vec::with_capacity(toks.len());
    for &tok in &toks {
        incremental.push(f.decode_step(tok, &mut c2).expect("decode k=1"));
    }
    eprintln!(
        "[smoke] k=1 incremental {} steps in {:.1}s ({:.2} s/tok)",
        toks.len(),
        t2.elapsed().as_secs_f32(),
        t2.elapsed().as_secs_f32() / toks.len() as f32
    );
    for (i, (a, b)) in prefill.iter().zip(incremental.iter()).enumerate() {
        for (j, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "k=1 identity FAILED on the real checkpoint at logits[{i}][{j}]"
            );
        }
    }
    eprintln!(
        "[smoke] k=1 identity: BIT-EXACT over {} positions",
        toks.len()
    );

    // Determinism: a second prefill pass replays bit-identically.
    let mut c3 = DoryKvCache::new(&w, 1);
    let replay = f.prefill(&toks, &mut c3).expect("prefill replay");
    for (i, (a, b)) in prefill.iter().zip(replay.iter()).enumerate() {
        for (j, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "determinism at logits[{i}][{j}]");
        }
    }

    // The loop deepens on the real weights: k=meta differs from k=1.
    let mut c4 = DoryKvCache::new(&w, k_meta);
    let deep = f.prefill(&toks, &mut c4).expect("prefill k=meta");
    let differing = prefill
        .iter()
        .zip(deep.iter())
        .flat_map(|(a, b)| a.iter().zip(b.iter()))
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    assert!(differing > 0, "k={k_meta} is a bit-identical replay of k=1");
    eprintln!(
        "[smoke] k={k_meta} vs k=1: {differing} differing logits — the recurrent block deepens"
    );

    // Footprint: the per-loop cache grows with k (the T2 counter).
    assert!(
        c4.kv_elements() > c.kv_elements(),
        "k={k_meta} cache footprint must exceed k=1"
    );

    // Bounded + finite through the real depth (the normalized-latent
    // law at 192 effective layers for k=4).
    for row in prefill.iter().chain(deep.iter()) {
        for (j, v) in row.iter().enumerate() {
            assert!(v.is_finite(), "non-finite logit at [{j}] (vocab {j})");
            assert!(
                v.abs() < 1e6,
                "logit magnitude {v} at vocab {j} — unbounded stream"
            );
        }
    }

    // Greedy smoke generation: 2 argmax steps continuing the prefill's
    // own cache (a real generate loop over the live context).
    let argmax = |row: &[f32]| -> u32 {
        let mut best = 0usize;
        for (i, v) in row.iter().enumerate() {
            if *v > row[best] {
                best = i;
            }
        }
        best as u32
    };
    let mut chain = toks.clone();
    let mut next = argmax(prefill.last().expect("prefill nonempty"));
    for _ in 0..2 {
        chain.push(next);
        let row = f.decode_step(next, &mut c).expect("gen decode");
        next = argmax(&row);
    }
    eprintln!("[smoke] greedy chain: {:?}", chain);

    eprintln!(
        "[smoke] ALL STRUCTURAL CELLS GREEN in {:.1}s total",
        t0.elapsed().as_secs_f32()
    );
}

/// Issue 1221 T2 — the k=4 last-loop serving cell on the REAL pin: the
/// policy-invariant receipt (prefill bit-identical to per-loop), the
/// k-independent footprint receipt, and the REPORTED decode divergence
/// table (first-divergence step + per-step max-abs logit delta + argmax
/// agreement up to divergence — the structural arm of the paper's
/// Table-6 shape; the suite-level accuracy delta needs T3's probe
/// suites). The divergence is an approximation, NEVER asserted zero —
/// the cell only asserts the invariant and the engagement (the table
/// prints; a zero divergence would be reported as such, not failed).
/// Heavy like the smoke above (~17 GB RSS + minutes of CPU forward):
///
/// ```sh
/// cargo test --release --features dory --test i1221_dory_forward_smoke -- --ignored --nocapture last_loop
/// ```
#[test]
#[ignore = "heavy cell: ~17 GB RSS + minutes of CPU forward — run with -- --ignored"]
fn last_loop_serving_delta_report_k4() {
    let Some(path) = pin_path() else {
        panic!(
            "SKIP LOUD: the Dory pin is absent (I1221_DORY_GGUF or \
             ../riir-train/data/{PIN_FILE}); restore via the riir-train \
             data_backup lane"
        );
    };
    let t0 = std::time::Instant::now();
    let w = load_dory_weights(&path).expect("the pin loads (arch dory)");
    let k = w.meta.num_loops; // the header's own serving depth (4 in the census)
    let f = DoryForward::new(&w);
    let toks: Vec<u32> = (0..8)
        .map(|i| (i * 15485863 + 11) % w.dims.vocab as u32)
        .collect();

    // Cell 1 — the invariant on the real weights: k={k} prefill is
    // bit-identical under both policies.
    let t1 = std::time::Instant::now();
    let mut c_per = DoryKvCache::with_policy(&w, k, DoryCachePolicy::PerLoop);
    let per = f.prefill(&toks, &mut c_per).expect("per-loop prefill");
    let mut c_last = DoryKvCache::with_policy(&w, k, DoryCachePolicy::LastLoop);
    let last = f.prefill(&toks, &mut c_last).expect("last-loop prefill");
    let prefill_diverged = per
        .iter()
        .zip(last.iter())
        .flat_map(|(a, b)| a.iter().zip(b.iter()))
        .any(|(x, y)| x.to_bits() != y.to_bits());
    assert!(
        !prefill_diverged,
        "k={k} prefill diverged between policies — the retention invariant broke"
    );
    eprintln!(
        "[delta] prefill invariant: k={k} BIT-EXACT across policies over {} positions ({:.1}s)",
        toks.len(),
        t1.elapsed().as_secs_f32()
    );

    // Cell 2 — the footprint receipt. Block counts derived from the
    // loaded weights (attn sits on even indices per range: input 8,
    // recurrent 20, output 8 in the census — never hardcoded).
    let attn_in_range = |from: usize, to: usize| {
        w.blocks[from..to]
            .iter()
            .filter(|b| b.kind == riir_infer_core::dory::DoryBlockKind::Attn)
            .count()
    };
    let n_in = attn_in_range(0, w.meta.recurrent_start());
    let n_rec = attn_in_range(w.meta.recurrent_start(), w.meta.output_start());
    let n_out = attn_in_range(w.meta.output_start(), w.blocks.len());
    let kv_dim = w.dims.n_kv_head * w.dims.head_dim;
    let want_per = (n_in + n_out + n_rec * k) * toks.len() * kv_dim * 2;
    let want_last = (n_in + n_out + n_rec) * toks.len() * kv_dim * 2;
    assert_eq!(c_per.kv_elements(), want_per, "per-loop k={k} footprint");
    assert_eq!(c_last.kv_elements(), want_last, "last-loop k={k} footprint");
    eprintln!(
        "[delta] kv_elements k={k}: per_loop={} last_loop={} ({:.2}x total; \
         recurrent-only = {k}x by construction: attn blocks in/rec/out = {n_in}/{n_rec}/{n_out})",
        c_per.kv_elements(),
        c_last.kv_elements(),
        c_per.kv_elements() as f64 / c_last.kv_elements() as f64,
    );

    // Cell 3 — the decode divergence table: greedy chains from each
    // policy's own prefill cache (a serving posture: the cache the
    // policy kept is the cache the chain reads).
    let argmax = |row: &[f32]| -> u32 {
        let mut best = 0usize;
        for (i, v) in row.iter().enumerate() {
            if *v > row[best] {
                best = i;
            }
        }
        best as u32
    };
    const STEPS: usize = 16;
    let mut chain_per = Vec::new();
    let mut chain_last = Vec::new();
    let mut deltas: Vec<(usize, f32, bool)> = Vec::new(); // (step, max|Δlogit|, argmax agree)
    let mut next_per = argmax(per.last().expect("prefill nonempty"));
    let mut next_last = argmax(last.last().expect("prefill nonempty"));
    let t2 = std::time::Instant::now();
    for step in 0..STEPS {
        chain_per.push(next_per);
        chain_last.push(next_last);
        if next_per != next_last {
            eprintln!(
                "[delta] FIRST TOKEN DIVERGENCE at generated step {step}: \
                 per_loop={} last_loop={}",
                next_per, next_last
            );
            break;
        }
        let row_per = f.decode_step(next_per, &mut c_per).expect("per decode");
        let row_last = f.decode_step(next_last, &mut c_last).expect("last decode");
        let max_delta = row_per
            .iter()
            .zip(row_last.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let agree = argmax(&row_per) == argmax(&row_last);
        deltas.push((step, max_delta, agree));
        next_per = argmax(&row_per);
        next_last = argmax(&row_last);
    }
    eprintln!(
        "[delta] decode table k={k} ({:.1}s for the run steps):",
        t2.elapsed().as_secs_f32()
    );
    eprintln!("[delta] step | max|dlogit| | argmax_agree");
    for (step, d, agree) in &deltas {
        eprintln!("[delta] {:>4} | {:>10.6} | {}", step, d, agree);
    }
    if deltas.len() == STEPS {
        eprintln!(
            "[delta] chains never diverged in {STEPS} greedy steps \
             (max step delta {:.6}) — the approximation is smaller than \
             one greedy flip over this horizon",
            deltas.last().map(|(_, d, _)| *d).unwrap_or(0.0)
        );
    }
    eprintln!(
        "[delta] chains: per={:?} last={:?}",
        chain_per.iter().take(8).collect::<Vec<_>>(),
        chain_last.iter().take(8).collect::<Vec<_>>()
    );
    eprintln!(
        "[delta] CELL COMPLETE in {:.1}s total",
        t0.elapsed().as_secs_f32()
    );
}
