//! Issue 1225 T1 — the f16-B per-family retention walk (the pre-registered
//! numerics decider, frozen in `.issues/1225` BEFORE this twin landed).
//!
//! The lossy-surface promotion rule (Bench 776 T3, katgpt-rs AGENTS.md): a
//! bit-changing surface gates on **deployed-path behavior — per-family,
//! conditional retention**, never bit-identity (the input rounds by
//! construction) and never an aggregate. The instrument mirrors the repo's
//! own lossy-surface precedent (`issue1146_relaxed_kl.rs`) widened per GEMM
//! family:
//!
//! - **End-to-end gate:** per frozen text family, final-token logits under
//!   the twin (whole trajectory on f16-B) vs the f32-B reference pass —
//!   KL mean/max, top-1, ΔNLL.
//! - **Per-GEMM-family gate (the demote axis):** the matched-input tap — at
//!   every twin-routed site of the tapped frame, the SAME input dispatches
//!   through BOTH bodies; per token row, the argmax pick must survive. A
//!   family that fails its floor demotes THAT family (site mask) and the
//!   walk re-runs on the survivor set.
//! - **Placebo control (HARD):** the twin compiled unpatched must read
//!   KL == 0.0 exactly and zero flips — any nonzero is an instrument defect
//!   and VOIDS the posture's armed numbers.
//! - **Engagement (the Issue-1124 T1.5 lesson):** every admitted site's tap
//!   instance count must be > 0; a demoted/compile-failed twin prints one
//!   loud counted line and reads here as zeros.
//!
//! The floors below are PRE-REGISTERED (issue 1225, commit `5776cf99`) —
//! never re-typed after a run:
//!   placebo: KL == 0, top1 == n, flips == 0, worst_rel == 0
//!   armed per family: flip rate ≤ 1e-3, worst_rel ≤ 1e-2
//!   armed end-to-end: KL max ≤ 1e-3 nats, top1 ≥ 15/16, |ΔNLL| ≤ 0.05
//!
//! One forward per selector posture (block_size 4104; the P=512 and P=4096
//! protocols share it — positions differ, capacity doesn't). Frame width
//! 128 (the serving frame). GPU-exclusive (the AGENTS rule — no sibling
//! compute workload); box state printed.
//!
//! ```sh
//! CARGO_TARGET_DIR=/tmp/i1225f16b cargo test -p riir-infer-gpu --release \
//!   --features prefill_f16b,metal_tensor_gemm,cubecl_runtime,ternary_gemv,multi_host,ternary_gemm_batched,ternary_gemm_simdgroup,ternary_deltanet_chunked_prefill,ternary_attention_batched_prefill,riir-infer-core/bonsai2_hadamard \
//!   --test issue1225_f16b_retention_walk -- --ignored --nocapture --test-threads=1
//! # posture axis: I1225_POSTURE=relaxed (default) | strict
//! # P axis:       I1225_PS=512,4096 (default 512,4096)
//! # demote axis:  I1225_FAMILIES=gdn_out_proj,ffn_down (comma site names — the T1
//! #               re-run after a family fails its floor; default = all armed)
//! # model:        BONSAI2_G1_MODEL overrides the sibling-layout default
//! ```

#![cfg(all(
    feature = "prefill_f16b",
    feature = "metal_tensor_gemm",
    feature = "cubecl_runtime",
    feature = "ternary_gemv",
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
    F16B_SITE_MASK_ALL, f16b_flip_samples, f16b_tap_reset, f16b_tap_snapshot, set_f16b_placebo,
    set_f16b_route, set_f16b_site_mask, set_f16b_tap,
};
use riir_infer_gpu::ternary_deltanet_gpu_forward::PrefillSite;
use riir_infer_gpu::{set_prefill_attn_metal_tensor, set_prefill_use_metal_tensor_zerocopy};

/// The CURRENT league lane model (the Bonsai-2 league pin; the same
/// sibling-layout default every lane here reads).
const FOLDED_MODEL: &str = "../../../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf";

/// The frozen four-family text corpus — `issue1146_relaxed_kl.rs` verbatim
/// (the repo's lossy-surface precedent; a changed text changes every
/// number, so it moves only together with a re-measured control row).
const FAMILIES: [(&str, &str); 4] = [
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
    (
        "prose",
        "The lighthouse keeper had kept the same routine for eleven years. Each \
         evening, an hour before sunset, he climbed the one hundred and twelve steps \
         to the lamp room, wiped the salt from the great lens, and checked the oil. \
         The island was small enough to walk around in twenty minutes, and in winter \
         the supply boat came only twice a month, so he had learned to read the \
         weather the way other people read the newspaper. A grey band low on the \
         horizon meant rain by morning. A sudden stillness in the gulls meant wind.\n\n\
         That autumn a letter arrived with the boat, addressed in handwriting he did \
         not recognise. It was from a schoolteacher on the mainland who wanted to \
         bring her class to see the light. He read it three times at the kitchen \
         table before he understood that he was nervous. He had not spoken to more \
         than one person at a time in years. Still, he wrote back the same night and \
         said yes, and then he spent the following week repainting the railings, \
         sweeping the stairs, and practising, out loud and to nobody, an explanation \
         of how a lens can bend a small flame into a beam that sailors see twenty \
         miles away.\n\n\
         The children arrived on a calm morning in October. There were nineteen of \
         them, wrapped in bright coats, and they asked more questions in the first \
         ten minutes than anyone had asked him in a decade. Why was the lamp so \
         large? What happened if the oil ran out? Had he ever seen a shipwreck? He \
         answered every question slowly and carefully, and by the time they reached \
         the top of the tower he had forgotten to be nervous at all. When the boat \
         carried them away that afternoon, they waved until the island was only a \
         dark line on the water, and he stood on the rocks and waved back.",
    ),
    (
        "zh",
        "春天的早晨，小镇上的市场总是最热闹的。天刚亮，卖菜的农民就推着车子来了，车上装满了新鲜的青菜、萝卜和西红柿。\
         卖早点的小店门口排起了长队，包子和豆浆的香味飘满了整条街。老人们喜欢在这个时候出来散步，\
         一边走一边和熟人打招呼，问问家里的情况，聊聊最近的天气。\n\n\
         李阿姨在市场里开了一家小小的面馆，已经二十多年了。她每天四点钟起床和面，六点钟准时开门。\
         她说，做面最重要的是耐心，面要揉得够久，汤要熬得够长，客人吃得出来。很多年轻人离开小镇去城里工作，\
         可是每次回家，第一件事就是来她的店里吃一碗面。他们说，城里的面再好吃，也没有家乡的味道。\n\n\
         到了中午，市场慢慢安静下来。农民们收拾好剩下的菜，准备回家吃饭，下午还要去地里干活。\
         李阿姨也终于可以坐下来休息一会儿，喝一杯热茶，看着街上来来往往的人，觉得日子虽然平淡，却很踏实。\n\n\
         傍晚的时候，镇上的孩子们放学回家，路过面馆总要停下来看一看。有的孩子帮李阿姨搬椅子，\
         有的孩子坐在门口写作业。李阿姨常常给他们煮一小碗清汤面，看着他们吃得干干净净，心里很高兴。\
         她说，等这些孩子长大了，走得再远，也会记得小镇上有一碗热乎乎的面在等着他们。",
    ),
];

/// The serving frame width (the dist lane's `--prefill-chunk` default).
const FRAME: usize = 128;

/// Pre-registered floors (issue 1225, commit `5776cf99` — frozen BEFORE the
/// twin landed; never re-type after a run).
const FAM_FLIP_RATE_MAX: f64 = 1e-3;
const FAM_WORST_REL_MAX: f64 = 1e-2;
const E2E_KL_MAX: f64 = 1e-3;
const E2E_TOP1_MIN_FRAC: f64 = 15.0 / 16.0;
const E2E_DNLL_BAND: f64 = 0.05;

/// Scored prefix lengths: `k` points spread evenly over `[p/2, p-1]` (the
/// 1146 shape). Pre-registered: K=8 at P=512, K=2 at P=4096.
fn scored_lengths(p: usize, k: usize) -> Vec<usize> {
    let lo = (p / 2).max(1);
    let span = p - 1 - lo;
    let mut v: Vec<usize> = (0..k)
        .map(|i| {
            lo + match k {
                1 => span,
                _ => i * span / (k - 1),
            }
        })
        .collect();
    v.dedup();
    v
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv { (i, x) } else { (bi, bv) }
        })
        .0
}

/// Log-softmax in f64, max-shifted (sequential fold — a measurement, so the
/// reduction order is pinned and repeatable).
fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let m = logits
        .iter()
        .fold(f64::NEG_INFINITY, |a, &x| a.max(x as f64));
    let lse = m + logits
        .iter()
        .map(|&x| ((x as f64) - m).exp())
        .sum::<f64>()
        .ln();
    logits.iter().map(|&x| x as f64 - lse).collect()
}

/// `KL(p ‖ q)` from log-probabilities, nats.
fn kl(lp: &[f64], lq: &[f64]) -> f64 {
    lp.iter()
        .zip(lq)
        .map(|(&a, &b)| a.exp() * (a - b))
        .sum::<f64>()
        .max(0.0)
}

/// End-to-end aggregate over the scored positions (the 1146 FamilyRow).
#[derive(Default)]
struct FamilyRow {
    n: usize,
    kl_sum: f64,
    kl_max: f64,
    top1_agree: usize,
    dnll_sum: f64,
    nll_ref_sum: f64,
}

impl FamilyRow {
    fn push(&mut self, ref_logits: &[f32], cand_logits: &[f32], target: usize) {
        let (lr, lc) = (log_softmax(ref_logits), log_softmax(cand_logits));
        let d = kl(&lr, &lc);
        self.n += 1;
        self.kl_sum += d;
        self.kl_max = self.kl_max.max(d);
        self.top1_agree += usize::from(argmax(ref_logits) == argmax(cand_logits));
        self.dnll_sum += lr[target] - lc[target];
        self.nll_ref_sum += -lr[target];
    }
}

/// Per-GEMM-family retention aggregate over the tapped instances.
#[derive(Default, Clone, Copy)]
struct FamGemmRow {
    instances: u64,
    rows: u64,
    flips: u64,
    worst_rel: f64,
}

/// One prefill pass over `prefix` in serving frames; returns the FINAL
/// frame's logits (the vocab readback — the committed-pick distribution).
fn pass_logits(fwd: &mut TernaryDeltanetGpuForward, prefix: &[usize], n_layer: usize) -> Vec<f32> {
    fwd.reset_state();
    let prompt_len = prefix.len();
    let mut logits = Vec::new();
    for (fi, chunk) in prefix.chunks(FRAME).enumerate() {
        let tok_base = fi * FRAME;
        let is_final = tok_base + chunk.len() == prompt_len;
        let _ = fwd.prefill_range_chunk(
            chunk,
            None,
            tok_base,
            prompt_len,
            LayerRange::new(0, n_layer as u32),
            is_final,
        );
        // The sync: every frame's readback drains the queue (the
        // bench_1146 pattern — the same cost every arm pays).
        logits = fwd.read_logits();
    }
    logits
}

fn print_provenance(when: &str) {
    let load = std::fs::read_to_string("/proc/loadavg")
        .ok()
        .map(|s| s.lines().next().map(String::from).unwrap_or(s))
        .or_else(|| {
            let out = std::process::Command::new("uptime")
                .output()
                .ok()
                .map(|o| String::from_utf8_lossy(o.stdout.as_slice()).into_owned())?;
            out.lines().next().map(String::from)
        });
    println!(
        "[provenance {when}] {} · GPU-exclusive per the AGENTS rule (no sibling compute workload)",
        load.unwrap_or_else(|| "load n/a".to_string())
    );
}

/// Issue 1226: the load tripwire. The pp32768 VOID record's cross-arm table
/// (see `.issues/1226_f16b_lane_followups.md`) shows the placebo HARD gate
/// fires ONLY under heavy CPU load (load 17.8 and 34.2 reproduced; load
/// 1.9–5.8 never did, across four quiet runs) — the reference pass is
/// load-sensitively nondeterministic (the Issue-670 submission-visibility
/// class, environmental). The walk therefore REFUSES above the ceiling
/// instead of spending ~1 h producing void-prone numbers; the refusal is
/// the pre-registration's own posture (an instrument that cannot hold its
/// measurement premises says so, never a green-zero). `I1225_LOAD_OK=1`
/// overrides for diagnosis — the run then stamps itself DIAGNOSTIC and its
/// numbers are void-by-default in the record.
fn load_tripwire(phase: &str) -> f64 {
    const LOAD_CEILING: f64 = 16.0;
    let one_min: f64 = std::fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|s| s.split_whitespace().next().and_then(|v| v.parse().ok()))
        .or_else(|| {
            let out = std::process::Command::new("uptime")
                .output()
                .ok()
                .map(|o| String::from_utf8_lossy(o.stdout.as_slice()).into_owned())?;
            // "load averages: a b c" — the FIRST number is the 1-minute mean.
            out.split("load averages:")
                .nth(1)
                .and_then(|rest| rest.split_whitespace().next().and_then(|v| v.parse().ok()))
        })
        .unwrap_or(0.0);
    let override_ok = std::env::var("I1225_LOAD_OK").as_deref() == Ok("1");
    if one_min > LOAD_CEILING && !override_ok {
        panic!(
            "i1226 load tripwire [{phase}]: 1-min load {one_min} > {LOAD_CEILING} — the reference \
             pass is load-sensitively nondeterministic at pp32768 (the 1226 VOID \
             record's cross-arm table); rerun on a quiet box, or \
             I1225_LOAD_OK=1 for a DIAGNOSTIC (void-by-default) run"
        );
    }
    if override_ok {
        println!("# I1225_LOAD_OK=1 — DIAGNOSTIC run: numbers void-by-default in the record");
    }
    one_min
}

#[test]
#[ignore = "requires the Metal GPU + the Bonsai-2 PQ2_0 model; run explicitly with --release"]
fn f16b_retention_walk() {
    print_provenance("start");
    let load = load_tripwire("start");
    println!("[provenance load-1min] {load}");
    let model = std::env::var_os("BONSAI2_G1_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(FOLDED_MODEL));
    if !model.exists() {
        panic!(
            "the Bonsai-2 league pin is the walk's premise — not found at {} \
             (BONSAI2_G1_MODEL overrides); a fixture cannot carry the deployed-path read",
            model.display()
        );
    }
    let posture = std::env::var("I1225_POSTURE").unwrap_or_else(|_| "relaxed".to_string());
    assert!(
        matches!(posture.as_str(), "relaxed" | "strict"),
        "I1225_POSTURE must be relaxed|strict, got {posture}"
    );
    let strict = posture == "strict";
    let ps: Vec<usize> = std::env::var("I1225_PS")
        .ok()
        .map(|v| v.split(',').map(|s| s.trim().parse().unwrap()).collect())
        .unwrap_or_else(|| vec![512, 4096]);

    // The serving posture: both promoted Metal arms ON (the twin rides the
    // shipped selector, it does not change it).
    set_prefill_use_metal_tensor_zerocopy(true);
    set_prefill_attn_metal_tensor(true);
    set_metal_tensor_relaxed_precision(!strict);

    // The demote axis (the T1 re-run after a family fails): comma site
    // names → the site mask; default = every family armed.
    match std::env::var("I1225_FAMILIES") {
        Ok(list) => {
            let mut mask = 0u64;
            for name in list.split(',') {
                let name = name.trim();
                let site = [
                    PrefillSite::AttnQ,
                    PrefillSite::AttnKv,
                    PrefillSite::AttnOutProj,
                    PrefillSite::GdnQkv,
                    PrefillSite::GdnZ,
                    PrefillSite::GdnA,
                    PrefillSite::GdnB,
                    PrefillSite::GdnOutProj,
                    PrefillSite::FfnGate,
                    PrefillSite::FfnUp,
                    PrefillSite::FfnDown,
                ]
                .into_iter()
                .find(|s| s.name() == name)
                .unwrap_or_else(|| panic!("I1225_FAMILIES: unknown site name {name:?}"));
                mask |= site.bit();
            }
            set_f16b_site_mask(mask);
            println!("# demote axis armed: {list} (mask {mask:#x})");
        }
        Err(_) => set_f16b_site_mask(F16B_SITE_MASK_ALL),
    }
    // The twin is a CONSTRUCTION-TIME posture: route + placebo pinned before
    // the forward exists. Placebo OFF for the armed+ref passes; the placebo
    // control runs in its own forward below.
    set_f16b_placebo(false);
    set_f16b_route(true);

    let tok = {
        let gguf = GgufFile::open(&model).expect("open gguf for tokenizer");
        BpeTokenizer::from_gguf(&gguf).expect("bpe tokenizer from gguf")
    };
    let (mut config, weights) =
        load_qwen_deltanet_ternary_weights_gguf(&model).expect("folded model load");
    let n_layer = config.n_layer;
    let p_max = *ps.iter().max().unwrap();
    config.block_size = p_max + 8;

    let ctx = CubeCLContext::new().expect("GPU init");
    println!(
        "# issue1225 f16-B retention walk · posture {} · P={ps:?} · frame {FRAME} · families {}",
        posture,
        FAMILIES.len()
    );

    // ── The armed posture's forward ────────────────────────────────────────
    let mut fwd = TernaryDeltanetGpuForward::new(&ctx, &config, &weights);

    let mut e2e: Vec<(&str, FamilyRow)> = FAMILIES
        .iter()
        .map(|(f, _)| (*f, FamilyRow::default()))
        .collect();
    let mut gemm_rows: Vec<FamGemmRow> = vec![FamGemmRow::default(); 12];
    let mut failures: Vec<String> = Vec::new();
    // Engagement accounting: the twin's dispatch total must equal the
    // incumbent's served-site total at the same mask (deltas per pass type —
    // the tap's dispatch_scratch also moves the incumbent counter, so the
    // ref delta is taken before any tap runs and the twin delta only ever
    // moves in armed passes).
    let mask_all = std::env::var("I1225_FAMILIES").is_err();
    let mut inc_total = 0u64;
    let mut twin_total = 0u64;

    for (fi, (family, text)) in FAMILIES.iter().enumerate() {
        // Issue 1226 (second failure, 19:19Z): the launch tripwire cannot see
        // a MID-RUN ramp — the relaxed run launched at load 2.78 and hit the
        // placebo HARD gate under an evening ramp to 59.71 (the strict twin
        // refused at launch). One check per family boundary: the run dies
        // loudly at the ramp instead of spending ~24 min producing
        // void-prone numbers; the per-family line also gives the record a
        // mid-run load trace (the box-state law).
        let mid_load = load_tripwire(&format!("mid-run family {family}"));
        println!("[provenance load-1min {family}] {mid_load}");
        let base_toks = tok.encode(text);
        // The corpus at p_max: the frozen text's tokens CYCLED to the scored
        // length (deterministic; the pre-registration's corpus amendment —
        // the 1146 texts encode ~412-700 tokens, short of P=4096; the axis
        // under test is numerics, not task skill).
        let toks: Vec<usize> = base_toks.iter().cycle().take(p_max + 2).copied().collect();
        assert!(
            toks.len() > p_max,
            "{family}: cycled to {} tokens, need > P={p_max}",
            toks.len()
        );
        let k = if p_max >= 16384 {
            1
        } else if p_max >= 4096 {
            2
        } else {
            8
        };
        let lengths = scored_lengths(p_max, k);
        let mut placebos_left = 1usize; // the control at the FIRST length of each family
        for &len in &lengths {
            let prefix = &toks[..len];
            let target = toks[len];

            // REF — the incumbent route (route off; twin unused).
            set_f16b_route(false);
            let (inc0, _) = fwd.f16b_route_dispatch_counts();
            let reference = pass_logits(&mut fwd, prefix, n_layer);
            let (inc1, _) = fwd.f16b_route_dispatch_counts();
            set_f16b_route(true);
            inc_total += inc1 - inc0;
            eprintln!("# {family}@{len}: ref done (route back on)");

            // ARMED — the twin feeds the trajectory; the tap arms ONLY on the
            // final frame (the deepest-context frame of the scored pass).
            let mut snap = None;
            fwd.reset_state();
            let (_, twin0) = fwd.f16b_route_dispatch_counts();
            for (fi, chunk) in prefix.chunks(FRAME).enumerate() {
                let tok_base = fi * FRAME;
                let is_final = tok_base + chunk.len() == prefix.len();
                if is_final {
                    f16b_tap_reset();
                    set_f16b_tap(true);
                }
                let _ = fwd.prefill_range_chunk(
                    chunk,
                    None,
                    tok_base,
                    prefix.len(),
                    LayerRange::new(0, n_layer as u32),
                    is_final,
                );
                let armed_logits = fwd.read_logits();
                if is_final {
                    set_f16b_tap(false);
                    snap = Some((f16b_tap_snapshot(), f16b_flip_samples(), armed_logits));
                }
            }
            let (snap, samples, armed_logits) = snap.expect("the final frame always taps");
            let (_, twin1) = fwd.f16b_route_dispatch_counts();
            twin_total += twin1 - twin0;

            // End-to-end row.
            assert!(
                reference.iter().chain(&armed_logits).all(|x| x.is_finite()),
                "{family}@{len}: non-finite logits"
            );
            e2e[fi].1.push(&reference, &armed_logits, target);

            // Per-family retention rows (accumulate over the pass's lengths).
            for (slot, row) in snap.iter().enumerate() {
                let acc = &mut gemm_rows[slot];
                acc.instances += row.instances;
                acc.rows += row.rows;
                acc.flips += row.flips;
                acc.worst_rel = acc.worst_rel.max(row.worst_rel);
            }
            if !samples.is_empty() {
                for s in &samples {
                    println!(
                        "  flip {family}@{len} {} occ={} row={} inc={} twin={} margin={:.3e}",
                        s.site, s.occurrence, s.row, s.inc_pick, s.twin_pick, s.inc_top1_margin
                    );
                }
            }

            // PLACEBO — its own forward (the twin compiled unpatched), the
            // same ref as the comparison base. One length per family is the
            // instrument-exactness control.
            if placebos_left > 0 {
                placebos_left -= 1;
                set_f16b_placebo(true);
                let mut pfwd = TernaryDeltanetGpuForward::new(&ctx, &config, &weights);
                let placebo = {
                    let mut logits = Vec::new();
                    pfwd.reset_state();
                    for (fi, chunk) in prefix.chunks(FRAME).enumerate() {
                        let tok_base = fi * FRAME;
                        let is_final = tok_base + chunk.len() == prefix.len();
                        if is_final {
                            f16b_tap_reset();
                            set_f16b_tap(true);
                        }
                        let _ = pfwd.prefill_range_chunk(
                            chunk,
                            None,
                            tok_base,
                            prefix.len(),
                            LayerRange::new(0, n_layer as u32),
                            is_final,
                        );
                        logits = pfwd.read_logits();
                        if is_final {
                            set_f16b_tap(false);
                        }
                    }
                    logits
                };
                let psnap = f16b_tap_snapshot();
                let mut pr = FamilyRow::default();
                pr.push(&reference, &placebo, target);
                let placebo_ok = pr.kl_max == 0.0
                    && pr.top1_agree == pr.n
                    && psnap.iter().all(|r| r.flips == 0 && r.worst_rel == 0.0);
                drop(pfwd);
                set_f16b_placebo(false);
                println!(
                    "# placebo {family}@{len}: KL max {:.3e} · top1 {}/{} · flips {} · worst_rel {:.3e} → {}",
                    pr.kl_max,
                    pr.top1_agree,
                    pr.n,
                    psnap.iter().map(|r| r.flips).sum::<u64>(),
                    psnap.iter().fold(0.0f64, |a, r| a.max(r.worst_rel)),
                    if placebo_ok {
                        "PASS"
                    } else {
                        "FAIL — instrument defect, posture VOID"
                    }
                );
                if !placebo_ok {
                    failures.push(format!(
                        "{posture} {family}@{len}: PLACEBO control failed (KL {:.3e}, flips {}) — the armed numbers are VOID",
                        pr.kl_max,
                        psnap.iter().map(|r| r.flips).sum::<u64>()
                    ));
                }
            }
        }
        let (name, row) = (&e2e[fi].0, &e2e[fi].1);
        let n = row.n as f64;
        println!(
            "{name:<6} n={:>2} · KL mean {:.3e} max {:.3e} nats · top1 {}/{} · ΔNLL mean {:+.3e} · NLL(ref) mean {:.3}",
            row.n,
            row.kl_sum / n,
            row.kl_max,
            row.top1_agree,
            row.n,
            row.dnll_sum / n,
            row.nll_ref_sum / n
        );
        if row.kl_max > E2E_KL_MAX {
            failures.push(format!(
                "{posture} {name}: end-to-end KL max {:.3e} > {E2E_KL_MAX:e}",
                row.kl_max
            ));
        }
        if (row.top1_agree as f64) < E2E_TOP1_MIN_FRAC * row.n as f64 {
            failures.push(format!(
                "{posture} {name}: top1 {}/{} under the {E2E_TOP1_MIN_FRAC} floor",
                row.top1_agree, row.n
            ));
        }
        if row.dnll_sum / n > E2E_DNLL_BAND || row.dnll_sum / n < -E2E_DNLL_BAND {
            failures.push(format!(
                "{posture} {name}: ΔNLL mean {:+.3e} outside ±{E2E_DNLL_BAND}",
                row.dnll_sum / n
            ));
        }
    }

    // ── The per-GEMM-family table + its floors ─────────────────────────────
    println!("\n── per-GEMM-family matched-input retention ({posture}) ──");
    println!(
        "  {:<14} {:>9} {:>10} {:>7} {:>9}  {:<8}",
        "family", "instances", "rows", "flips", "flip_rate", "worst_rel"
    );
    for (slot, acc) in gemm_rows.iter().enumerate() {
        let name = match slot {
            0 => "attn_q",
            1 => "attn_kv",
            2 => "attn_out_proj",
            3 => "gdn_in_proj_qkv",
            4 => "gdn_in_proj_z",
            5 => "gdn_in_proj_a",
            6 => "gdn_in_proj_b",
            7 => "gdn_out_proj",
            8 => "ffn_gate",
            9 => "ffn_up",
            10 => "ffn_down",
            _ => "other",
        };
        println!(
            "  {name:<14} {:>9} {:>10} {:>7} {:>9.2e}  {:<8.3e}",
            acc.instances,
            acc.rows,
            acc.flips,
            if acc.rows == 0 {
                f64::NAN
            } else {
                acc.flips as f64 / acc.rows as f64
            },
            acc.worst_rel,
        );
        // Engagement floor (the Issue-1124 T1.5 lesson, enforced by counter
        // equality): at the same mask, the twin's dispatch total must equal
        // the incumbent's served-site total — a twin fallthrough (compile
        // failure, dispatch error, a mask mismatch) reads as T < R. A
        // per-site ZERO row is NOT a failure by itself: with T == R it means
        // the incumbent never dispatched that site either (this model stack
        // routes the GDN a/b projections through the deltanet prework lane,
        // not prefill_project; `Other` is never named) — printed, never
        // failed.
        if acc.instances == 0 {
            let site = SITES_BY_SLOT[slot];
            let admitted = riir_infer_gpu::prefill_f16b::f16b_site_admits(site);
            if admitted {
                println!(
                    "  (engagement note: {name} admitted but zero twin instances — the \
                     incumbent never served this site in this stack either, T==R holds)"
                );
            }
        } else {
            let rate = acc.flips as f64 / acc.rows as f64;
            if rate > FAM_FLIP_RATE_MAX {
                failures.push(format!(
                    "{posture} {name}: flip rate {rate:.3e} > {FAM_FLIP_RATE_MAX:e} — DEMOTE this family (site mask) and re-run"
                ));
            }
            if acc.worst_rel > FAM_WORST_REL_MAX {
                failures.push(format!(
                    "{posture} {name}: worst_rel {:.3e} > {FAM_WORST_REL_MAX:e} — DEMOTE this family and re-run",
                    acc.worst_rel
                ));
            }
        }
    }

    // The engagement totals gate (mask ALL only — a narrowed twin mask
    // intentionally serves fewer sites than the incumbent tensor mask).
    println!(
        "\nengagement totals: incumbent {inc_total} · twin {twin_total} → {}",
        if inc_total == twin_total {
            "EQUAL"
        } else if mask_all {
            "MISMATCH"
        } else {
            "narrowed mask (equality not expected)"
        }
    );
    if mask_all && inc_total != twin_total {
        failures.push(format!(
            "{posture}: ENGAGEMENT totals mismatch — incumbent served {inc_total} sites, \
             twin routed {twin_total}: a fallthrough served the incumbent under the twin label"
        ));
    }

    if failures.is_empty() {
        println!(
            "\nWALK VERDICT {posture}: PASS — every pre-registered floor held; the twin route \n\
             graduates to the paired frame A/B (issue 1225 T3)"
        );
    } else {
        println!(
            "\nWALK VERDICT {posture}: FAIL — {} floor finding(s):",
            failures.len()
        );
        for f in &failures {
            println!("  - {f}");
        }
        panic!(
            "issue1225 walk floors failed at {posture}: {}",
            failures.len()
        );
    }
    print_provenance("end");
}

/// The site slots in [`PrefillSite`] discriminant order (the census index →
/// site mapping, mirrored from the module's own `SITES`).
const SITES_BY_SLOT: [PrefillSite; 12] = [
    PrefillSite::AttnQ,
    PrefillSite::AttnKv,
    PrefillSite::AttnOutProj,
    PrefillSite::GdnQkv,
    PrefillSite::GdnZ,
    PrefillSite::GdnA,
    PrefillSite::GdnB,
    PrefillSite::GdnOutProj,
    PrefillSite::FfnGate,
    PrefillSite::FfnUp,
    PrefillSite::FfnDown,
    PrefillSite::Other,
];

#[test]
fn scored_lengths_span_second_half() {
    assert_eq!(scored_lengths(512, 1), vec![511]);
    let v = scored_lengths(512, 8);
    assert_eq!(v.first(), Some(&256));
    assert_eq!(v.last(), Some(&511));
    assert_eq!(v.len(), 8);
    let v2 = scored_lengths(4096, 2);
    assert_eq!(v2, vec![2048, 4095]);
}
