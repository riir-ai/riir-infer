//! Issue 1226 — the pp32768 placebo-VOID replay probe (code + math only).
//!
//! The F1 relaxed cell hit the pre-registered hard gate on 2026-11-16:
//! `placebo math@32767: KL 5.428e-5 → FAIL` while the code-family placebo in
//! the SAME process was exact-zero (evidence:
//! `.benchmarks/i1226_f16b_pp32k_void_evidence.log`). The mechanism is
//! UNPROVEN; this probe replays the walk's exact posture + sequence for the
//! first two families and adds the discriminating reads the walk lacks:
//!
//! 1. **The fresh-forward third reference** — `ref_math` measured again on a
//!    brand-new forward after the full code+math replay. Nonzero vs the
//!    walk-order `ref_math` ⇒ `fwd`'s reference drifted across the family
//!    boundary (the state class); exact ⇒ the reference is stable and the
//!    defect sits in the placebo half.
//! 2. **The tap-off arm** (`I1226_REPRO_TAP_OFF=1`) — the retention tap is
//!    armed INSIDE the walk's placebo pass loop (its final frame), and
//!    `f16b_tap_site` commits raw-Metal `dispatch_scratch` work on the SAME
//!    shared queue mid-frame. This arm reruns the whole replay with the tap
//!    never armed; the placebo reading with vs without the tap separates the
//!    interleave class from the state class in one pair of runs.
//! 3. **The family-order arm** (`I1226_REPRO_FAM=math`) — math alone, run
//!    FIRST on a fresh forward. Exact ⇒ math is not intrinsically drift-y;
//!    the walk's failure required the code family to have run before it.
//!
//! Every comparison prints KL max / flips / worst_rel; the placebo bars are
//! the walk's own pre-registration (KL == 0, flips == 0, worst_rel == 0 —
//! ANY nonzero is a defect read, never a tolerance).
//!
//! ```sh
//! CARGO_TARGET_DIR=/tmp/i1225f16b cargo test -p riir-infer-gpu --release \
//!   --features prefill_f16b,production,cubecl_runtime,multi_host,riir-infer-core/bonsai2_hadamard \
//!   --test i1226_pp32k_placebo_repro -- --ignored --nocapture --test-threads=1
//! # bisect arms: I1226_REPRO_TAP_OFF=1 · I1226_REPRO_FAM=code|math
//! # model:      BONSAI2_G1_MODEL overrides the sibling-layout default
//! ```

#![cfg(all(
    feature = "prefill_f16b",
    feature = "metal_tensor_gemm",
    feature = "cubecl_runtime",
    feature = "multi_host",
    feature = "ternary_gemm_batched",
    target_os = "macos"
))]

use std::path::PathBuf;

use riir_infer_core::gguf_loader::{GgufFile, load_qwen_deltanet_ternary_weights_gguf};
use riir_infer_core::multi_host::LayerRange;
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_gpu::TernaryDeltanetGpuForward;
use riir_infer_gpu::cubecl_runtime::CubeCLContext;
use riir_infer_gpu::gemm_ternary_metal_tensor::set_metal_tensor_relaxed_precision;
use riir_infer_gpu::prefill_f16b::{
    F16B_SITE_MASK_ALL, f16b_tap_reset, f16b_tap_snapshot, set_f16b_placebo, set_f16b_route,
    set_f16b_site_mask, set_f16b_tap,
};
use riir_infer_gpu::{set_prefill_attn_metal_tensor, set_prefill_use_metal_tensor_zerocopy};

/// The CURRENT league lane model (the Bonsai-2 league pin; the walk's own
/// default).
const FOLDED_MODEL: &str = "../../../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf";

/// The replay's families — the walk's frozen corpus, code + math VERBATIM
/// (a changed text changes every number).
const FAMILIES: [(&str, &str); 2] = [
    (
        "code",
        r#"use std::collections::HashMap;

/// Count how often each word appears in the input, ignoring case.
pub fn word_counts(text: &str) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for word in text.split_whitespace() {
        let key = word
            .trim_matches(|c: char| !c.is_alphanumeric())
            .to_lowercase();
        if key.is_empty() {
            continue;
        }
        *counts.entry(key).or_insert(0) += 1;
    }
    counts
}

/// Return the `n` most frequent words, ties broken alphabetically.
pub fn top_words(counts: &HashMap<String, usize>, n: usize) -> Vec<(String, usize)> {
    let mut pairs: Vec<(String, usize)> = counts
        .iter()
        .map(|(word, &count)| (word.clone(), count))
        .collect();
    pairs.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    pairs.truncate(n);
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_are_case_insensitive() {
        let counts = word_counts("The cat saw the other cat.");
        assert_eq!(counts["the"], 2);
        assert_eq!(counts["cat"], 2);
        assert_eq!(counts["saw"], 1);
    }

    #[test]
    fn top_words_breaks_ties_alphabetically() {
        let counts = word_counts("b a c b a");
        assert_eq!(top_words(counts, 2), vec![("a".to_string(), 2), ("b".to_string(), 2)]);
    }
}
"#,
    ),
    (
        "math",
        "Problem. A rectangle has a perimeter of 54 centimetres, and its length is \
         three centimetres more than twice its width. Find the area of the rectangle.\n\n\
         Solution. Let the width be w centimetres. Then the length is 2w + 3. The \
         perimeter of a rectangle is twice the sum of its length and width, so \
         2(w + 2w + 3) = 54. Dividing both sides by 2 gives 3w + 3 = 27, hence \
         3w = 24 and w = 8. The length is therefore 2(8 + 19) = 2(27) = 54, which matches \
         the given perimeter. The area is length times width, 19 × 8 = 152 square centimetres.\n\n\
         Problem. Solve the equation x^2 - 5x + 6 = 0.\n\n\
         Solution. We look for two numbers whose product is 6 and whose sum is 5. \
         These are 2 and 3, so the quadratic factors as (x - 2)(x - 3) = 0. A product \
         is zero exactly when one of its factors is zero, so x = 2 or x = 3. \
         Substituting x = 2 gives 4 - 10 + 6 = 0, and x = 3 gives 9 - 15 + 6 = 0, \
         so both roots are correct.\n\n\
         Problem. What is the sum of the first 100 positive integers?\n\n\
         Solution. Write the sum forwards and backwards: S = 1 + 2 + ... + 100 and \
         S = 100 + 99 + ... + 1. Adding the two lines term by term gives 100 pairs, \
         each summing to 101, so 2S = 100 × 101 = 10100 and S = 5050.",
    ),
];

/// The serving frame width (the walk's constant).
const FRAME: usize = 128;

/// One scored length per family at pp32768 (the walk's k rule: p >= 16384 → k = 1,
/// so the single scored prefix is 32767).
fn scored_length(p: usize) -> usize {
    p - 1
}

/// The walk's frame loop, verbatim shape: reset_state, FRAME chunks,
/// `is_final` from `tok_base + chunk.len() == prompt_len`, logits read every
/// frame. `tap` arms the retention tap on the final frame exactly as the
/// walk's armed pass does (and as its placebo pass — involuntarily — does).
fn pass_logits(
    fwd: &mut TernaryDeltanetGpuForward,
    prefix: &[usize],
    n_layer: usize,
    tap: bool,
) -> (Vec<f32>, u64, f64) {
    fwd.reset_state();
    let prompt_len = prefix.len();
    let mut logits = Vec::new();
    let mut flips = 0u64;
    let mut worst_rel = 0.0f64;
    for (fi, chunk) in prefix.chunks(FRAME).enumerate() {
        let tok_base = fi * FRAME;
        let is_final = tok_base + chunk.len() == prompt_len;
        if is_final && tap {
            f16b_tap_reset();
            set_f16b_tap(true);
        }
        let _ = fwd.prefill_range_chunk(
            chunk,
            None,
            tok_base,
            prompt_len,
            LayerRange::new(0, n_layer as u32),
            is_final,
        );
        logits = fwd.read_logits();
        if is_final {
            if tap {
                set_f16b_tap(false);
                for row in f16b_tap_snapshot() {
                    flips += row.flips;
                    worst_rel = worst_rel.max(row.worst_rel);
                }
            }
            break; // the walk reads the final frame's logits — same read here
        }
    }
    (logits, flips, worst_rel)
}

fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let m = logits.iter().fold(f32::NEG_INFINITY, |a: f32, &b| a.max(b));
    let m = m as f64;
    let lse: f64 = logits
        .iter()
        .map(|&x| ((x as f64) - m).exp())
        .sum::<f64>()
        .ln();
    logits.iter().map(|&x| x as f64 - m - lse).collect()
}

/// `KL(p ‖ q)` from log-probabilities, nats.
fn kl(lp: &[f64], lq: &[f64]) -> f64 {
    lp.iter()
        .zip(lq)
        .map(|(&a, &b)| a.exp() * (a - b))
        .sum::<f64>()
        .max(0.0)
}

fn kl_between(a: &[f32], b: &[f32]) -> f64 {
    kl(&log_softmax(a), &log_softmax(b))
}

fn print_row(label: &str, kl: f64, flips: u64, worst_rel: f64) {
    let ok = kl == 0.0 && flips == 0 && worst_rel == 0.0;
    println!(
        "# {label}: KL {kl:.3e} · flips {flips} · worst_rel {worst_rel:.3e} → {}",
        if ok { "EXACT" } else { "NONZERO (defect read)" }
    );
}

#[test]
#[ignore = "requires the Metal GPU + the Bonsai-2 PQ2_0 model; run explicitly with --release"]
fn pp32k_placebo_repro() {
    let load = std::fs::read_to_string("/proc/loadavg")
        .ok()
        .map(|s| s.lines().next().map(String::from).unwrap_or(s))
        .or_else(|| {
            std::process::Command::new("uptime")
                .output()
                .ok()
                .map(|o| String::from_utf8_lossy(o.stdout.as_slice()).into_owned())
                .map(|s| s.lines().next().unwrap_or(&s).to_string())
        });
    println!(
        "[provenance start] {} · correctness cell (bit comparisons — CPU load does not move bits; GPU must be compute-exclusive)",
        load.unwrap_or_else(|| "load n/a".to_string())
    );
    let tap_off = std::env::var("I1226_REPRO_TAP_OFF").as_deref() == Ok("1");
    let fam_filter = std::env::var("I1226_REPRO_FAM").ok();
    println!(
        "# i1226 pp32k placebo repro · tap {} · families {}",
        if tap_off {
            "OFF"
        } else {
            "ON (the walk's default)"
        },
        fam_filter.as_deref().unwrap_or("code,math")
    );

    let model = std::env::var_os("BONSAI2_G1_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(FOLDED_MODEL));
    assert!(
        model.exists(),
        "the Bonsai-2 league pin is the walk's premise — not found at {}",
        model.display()
    );

    // The walk's posture, verbatim (relaxed) — with ONE deliberate
    // divergence knob: the walk's relaxed posture sets
    // `set_metal_tensor_relaxed_precision(!strict)` = TRUE (walk L401),
    // while this repro previously pinned it FALSE (a strict-precision
    // “relaxed” — the harness delta candidate after the third failure
    // reproduced the walk's digit-class at quiet load; the prior repro
    // arms produced a DIFFERENT fingerprint, 4.141e-5 vs the walk's
    // 5.428e-5, consistent with a posture delta moving the stale bytes).
    // `I1226_REPRO_RELAXED=1` matches the walk exactly; the default stays
    // the historical strict-pinned posture so the prior arms reproduce.
    let relaxed_precision = std::env::var("I1226_REPRO_RELAXED").as_deref() == Ok("1");
    set_prefill_use_metal_tensor_zerocopy(true);
    set_prefill_attn_metal_tensor(true);
    set_metal_tensor_relaxed_precision(relaxed_precision);
    println!(
        "# posture: metal_tensor_relaxed_precision={} {}",
        relaxed_precision,
        if relaxed_precision {
            "(the walk's relaxed posture)"
        } else {
            "(the historical strict-pinned repro posture)"
        }
    );
    set_f16b_placebo(false);
    set_f16b_route(true);
    set_f16b_site_mask(F16B_SITE_MASK_ALL);
    let tap_armed_default = !tap_off;

    let tok = {
        let gguf = GgufFile::open(&model).expect("open gguf for tokenizer");
        BpeTokenizer::from_gguf(&gguf).expect("bpe tokenizer from gguf")
    };
    let (mut config, weights) =
        load_qwen_deltanet_ternary_weights_gguf(&model).expect("folded model load");
    let n_layer = config.n_layer;
    let p_max = 32768usize;
    config.block_size = p_max + 8;

    let ctx = CubeCLContext::new().expect("GPU init");

    // ── fwd: the walk's armed-posture forward ──────────────────────────────
    let mut fwd = TernaryDeltanetGpuForward::new(&ctx, &config, &weights);

    // Pre-tokenize the requested families (cycled to p_max + 2 — the walk's
    // corpus amendment).
    let wanted: Vec<&str> = FAMILIES
        .iter()
        .map(|(f, _)| *f)
        .filter(|f| fam_filter.as_deref().is_none_or(|w| w == "both" || *f == w))
        .collect();
    assert!(
        !wanted.is_empty(),
        "I1226_REPRO_FAM must name code, math, or both"
    );
    let toks: Vec<(&str, Vec<usize>)> = FAMILIES
        .iter()
        .filter(|(f, _)| wanted.contains(f))
        .map(|(f, text)| {
            let base = tok.encode(text);
            let cycled: Vec<usize> = base.iter().cycle().take(p_max + 2).copied().collect();
            assert!(cycled.len() > p_max, "{f}: cycled short");
            (*f, cycled)
        })
        .collect();

    let mut results: Vec<(String, f64, u64, f64)> = Vec::new();
    let mut in_loop_refs: Vec<(&str, Vec<f32>)> = Vec::new();

    // ── The walk's exact sequence, per family ─────────────────────────────
    for (fam, toks) in &toks {
        let len = scored_length(p_max);
        let prefix = &toks[..len];

        // REF — the incumbent route.
        set_f16b_route(false);
        let (ref_logits, _, _) = pass_logits(&mut fwd, prefix, n_layer, false);
        set_f16b_route(true);
        in_loop_refs.push((fam, ref_logits.clone()));
        println!("# {fam}@{len}: ref done");

        // ARMED — the twin serves, tap on the final frame (the walk's armed
        // pass; skipped entirely under the tap-off arm).
        let tap_here = tap_armed_default;
        let (armed_logits, a_flips, a_worst) = pass_logits(&mut fwd, prefix, n_layer, tap_here);
        let armed_kl = kl_between(&ref_logits, &armed_logits);
        println!("# {fam}@{len}: armed done (tap {tap_here})");
        results.push((
            format!("armed {fam} vs ref {fam} (informational)"),
            armed_kl,
            a_flips,
            a_worst,
        ));

        // PLACEBO — the walk's shape exactly: placebo pinned BEFORE the new
        // forward's construction (the unpatched twin body is a construction
        // posture), the flag stays on through the pass, off after. The pass
        // arms the tap on its final frame — the walk's placebo does this
        // INVOLUNTARILY; the tap-off arm also disarms it here (that IS the
        // bisect).
        let placebo_tap = tap_armed_default;
        set_f16b_placebo(true);
        let mut pfwd = TernaryDeltanetGpuForward::new(&ctx, &config, &weights);
        let (placebo_logits, p_flips, p_worst) =
            pass_logits(&mut pfwd, prefix, n_layer, placebo_tap);
        drop(pfwd);
        set_f16b_placebo(false);
        let placebo_kl = kl_between(&ref_logits, &placebo_logits);
        println!("# {fam}@{len}: placebo done (tap {placebo_tap})");
        results.push((
            format!("placebo {fam} vs ref {fam} (the walk's HARD gate)"),
            placebo_kl,
            p_flips,
            p_worst,
        ));

        // I1226_REPRO_REFREPLAY=1 — the third discriminator: a SECOND
        // route-off ref pass on fwd RIGHT NOW (after this family's armed
        // pass and pfwd's construction+pass+drop), compared against the
        // family's ORIGINAL in-loop reference. Nonzero ⇒ fwd's incumbent
        // pass is history-dependent (each pass shifts the next — the state
        // class lives inside fwd). Then one more ref pass: a third
        // comparison shows whether the drift settles after one more pass
        // (the base arm's fresh-vs-now EXACT hints it does).
        if std::env::var("I1226_REPRO_REFREPLAY").as_deref() == Ok("1") {
            set_f16b_route(false);
            let (ref2, _, _) = pass_logits(&mut fwd, prefix, n_layer, false);
            let (ref3, _, _) = pass_logits(&mut fwd, prefix, n_layer, false);
            set_f16b_route(true);
            let ref2_vs_orig = kl_between(&ref2, &ref_logits);
            let ref3_vs_ref2 = kl_between(&ref3, &ref2);
            print_row(
                &format!("REFREPLAY ref#2 vs orig ref ({fam}) — history discriminator"),
                ref2_vs_orig,
                0,
                0.0,
            );
            print_row(
                &format!("REFREPLAY ref#3 vs ref#2 ({fam}) — settle check"),
                ref3_vs_ref2,
                0,
                0.0,
            );
            results.push((
                format!("refreplay #2-vs-orig ({fam}, informational)"),
                ref2_vs_orig,
                0,
                0.0,
            ));
            results.push((
                format!("refreplay #3-vs-#2 ({fam}, informational)"),
                ref3_vs_ref2,
                0,
                0.0,
            ));
        }
    }

    // ── The fresh-forward third reference (the discriminator) ─────────────
    // Constructed AFTER the full replay, same posture as fwd, its reference
    // pass taken with the route OFF (the incumbent — what every reference
    // should be). Two comparisons against the LAST family:
    //   fresh-fwd ref vs fwd's ORIGINAL in-loop reference → nonzero means
    //   fwd's reference moved across the replay (the state class); exact
    //   means the reference was stable and the defect sits in the placebo
    //   half.
    //   fresh-fwd ref vs fwd's reference re-measured NOW → whether fwd's
    //   current output still equals the fresh forward's (no decay) or has
    //   settled somewhere else.
    set_f16b_route(true); // the walk's armed posture at construction time
    let mut fwd2 = TernaryDeltanetGpuForward::new(&ctx, &config, &weights);
    {
        let (last_fam, last_toks) = toks.last().expect("at least one family");
        let len = scored_length(p_max);
        let prefix = &last_toks[..len];
        let orig_ref = in_loop_refs
            .iter()
            .rev()
            .find(|(f, _)| f == last_fam)
            .map(|(_, l)| l.clone())
            .expect("the in-loop reference for the last family");
        set_f16b_route(false);
        let (fresh_logits, _, _) = pass_logits(&mut fwd2, prefix, n_layer, false);
        let (fwd_now_logits, _, _) = pass_logits(&mut fwd, prefix, n_layer, false);
        set_f16b_route(true);
        let fresh_vs_orig = kl_between(&fresh_logits, &orig_ref);
        let fresh_vs_now = kl_between(&fresh_logits, &fwd_now_logits);
        print_row(
            &format!("fresh-fwd ref vs ORIG in-loop ref ({last_fam}) — state discriminator"),
            fresh_vs_orig,
            0,
            0.0,
        );
        print_row(
            &format!("fresh-fwd ref vs fwd-NOW ref ({last_fam})"),
            fresh_vs_now,
            0,
            0.0,
        );
        results.push((
            format!("fresh-vs-orig ({last_fam}, informational)"),
            fresh_vs_orig,
            0,
            0.0,
        ));
        results.push((
            format!("fresh-vs-now ({last_fam}, informational)"),
            fresh_vs_now,
            0,
            0.0,
        ));
    }

    println!("\n── i1226 repro verdict ──");
    let mut defects = 0usize;
    for (label, k, f, w) in &results {
        let hard = label.contains("HARD gate");
        let bad = if hard {
            *k != 0.0 || *f != 0 || *w != 0.0
        } else {
            false
        };
        if bad {
            defects += 1;
        }
        println!(
            "  {:<52} KL {k:.3e} · flips {f} · worst_rel {w:.3e}{}",
            label,
            if bad { "  ← VOID-class" } else { "" }
        );
    }
    if defects > 0 {
        println!(
            "\nVERDICT: {defects} placebo defect(s) reproduced — bisect: rerun with \
             I1226_REPRO_TAP_OFF=1 (tap-interleave class) then I1226_REPRO_FAM=math \
             (order/state class); keep this log as the evidence row."
        );
        panic!(
            "i1226 repro: the walk's placebo defect reproduced ({defects} HARD-gate reads nonzero)"
        );
    }
    println!(
        "\nVERDICT: all placebo reads EXACT under this arm — the defect did not reproduce; compare against the arm that shows it."
    );
}
