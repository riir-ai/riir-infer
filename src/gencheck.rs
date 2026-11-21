//! Generation-check instruments (riir-infer Issue 1242 T2 — the QDA distill's
//! `gencheck`, riir-refine Research 427 / arXiv:2611.23388): the
//! **truncation-at-limit count** and the **suffix-cycle / repetition census**
//! over long generations.
//!
//! QDA's biggest search win came from a candidate the quality proxy LIKED but
//! the generations caught looping — the failure class a per-token NLL readout
//! cannot see. This module is the generation-behavior axis the frozen-read
//! census (golden Δ-NLL + argmax flips) lacks: given a batch of long
//! generations and the sampling length limit, it reports
//!
//! - `truncated_at_limit`: the generation hit the limit (truncation census),
//! - the longest **suffix cycle**: the strict period `p` of the maximal
//!   repeating tail (the loop shape — `p=1` is a single-token stutter, `p=8`
//!   an 8-token phrase loop), absent → `None`,
//! - `tail_repeat_ratio`: repetitions of that cycle in the tail window,
//! - `distinct_ngram_ratio` at the pinned window (n=8): distinct 8-grams over
//!   total — the density axis (a looping generation collapses toward ~0).
//!
//! Thresholds are **pinned in-source and fixture-demonstrated** (the issue's
//! law): `LOOP_RATIO_MIN` (a tail is a loop when the cycle repeats this many
//! times) and `DENSITY_LOOP_MAX` (distinct-8-gram ratio at or below this reads
//! looping-dense). The `flagged()` verdict ORs the two axes — QDA's class was
//! "proxy liked it, generations caught the loop", so both the explicit cycle
//! and the density collapse must fire independently.
//!
//! Pure text analysis: zero forward, zero deps, deterministic (integer
//! suffix mechanics + ordered folds — no floats beyond the two ratios).
//! Instrument-grade only.

/// The sampling length limit the truncation census reads (the harness sets
/// its own; this is the fixture/default posture).
pub const DEFAULT_LEN_LIMIT: usize = 1024;

/// A tail is a LOOP when the detected cycle repeats at least this many times.
pub const LOOP_RATIO_MIN: usize = 3;

/// Distinct-8-gram ratio at or below this reads looping-dense.
pub const DENSITY_LOOP_MAX: f64 = 0.35;

/// The n-gram width for the density axis (pinned; QDA's repetition class is
/// phrase-scale, 8 tokens catches both token stutters and phrase loops).
pub const DENSITY_NGRAM: usize = 8;

/// Minimum tokens for ANY verdict (below this the census abstains — short
/// generations cannot demonstrate a loop).
pub const MIN_TOKENS: usize = 64;

/// One generation's census reading.
#[derive(Debug, Clone, PartialEq)]
pub struct GencheckReading {
    /// Token count of the generation (the raw input length).
    pub tokens: usize,
    /// The generation hit the length limit (truncation census).
    pub truncated_at_limit: bool,
    /// Strict period of the maximal repeating suffix cycle, if any.
    pub suffix_cycle_period: Option<usize>,
    /// How many times the suffix cycle repeats in the tail.
    pub tail_repeat_count: usize,
    /// Distinct `DENSITY_NGRAM`-grams / total, in `(0,1]`.
    pub distinct_ngram_ratio: f64,
}

impl GencheckReading {
    /// The gencheck verdict: either loop axis fires. `None` tokens (below
    /// [`MIN_TOKENS`]) never flag.
    pub fn flagged(&self) -> bool {
        if self.tokens < MIN_TOKENS {
            return false;
        }
        if self.distinct_ngram_ratio <= DENSITY_LOOP_MAX {
            return true;
        }
        match self.suffix_cycle_period {
            Some(_) => self.tail_repeat_count >= LOOP_RATIO_MIN,
            None => false,
        }
    }
}

/// Tokenize on whitespace+counter boundaries for the TEXT entry point: the
/// census is vocabulary-agnostic (it reads repetition shape, not identity),
/// so a stable word split suffices for prose generations. Rust byte-token
/// lanes pass their own token slice to [`census_tokens`] directly.
pub fn census_text(generation: &str, len_limit: usize) -> GencheckReading {
    let tokens: Vec<&str> = generation.split_whitespace().collect();
    census_tokens(&tokens, len_limit)
}

/// The census over a token slice (the canonical form).
pub fn census_tokens(tokens: &[&str], len_limit: usize) -> GencheckReading {
    let tokens_n = tokens.len();
    let truncated_at_limit = tokens_n >= len_limit;

    let (suffix_cycle_period, tail_repeat_count) = suffix_cycle(tokens);
    let distinct_ngram_ratio = distinct_ngram_ratio(tokens, DENSITY_NGRAM);

    GencheckReading {
        tokens: tokens_n,
        truncated_at_limit,
        suffix_cycle_period,
        tail_repeat_count,
        distinct_ngram_ratio,
    }
}

/// The maximal repeating suffix cycle: the largest `p` (smallest period at
/// the tail) such that the last `k*p` tokens are `k` repetitions of the last
/// `p` tokens, with `k >= 2` and the cycle content itself non-constant at
/// `p == 1` excluded only when the unit is NOT repeated (a single trailing
/// token is not a cycle). Returns `(period, repeat_count)`; `None` when no
/// tail cycle exists.
///
/// Determinism: scans `p` ascending from 1, keeps the FIRST period whose
/// repeat count is maximal-equivalent (the smallest period wins ties — a
/// `p=1` stutter is preferred over a `p=2` description of the same tail).
fn suffix_cycle(tokens: &[&str]) -> (Option<usize>, usize) {
    let n = tokens.len();
    if n < 4 {
        return (None, 0);
    }
    let mut best: Option<(usize, usize)> = None;
    // Cap the period at n/2 (a cycle needs >= 2 repeats to exist).
    for p in 1..=(n / 2) {
        // Count full repeats of tokens[n-p..] backwards.
        let unit = &tokens[n - p..n];
        let mut k = 1usize;
        while n >= (k + 1) * p && tokens[n - (k + 1) * p..n - k * p] == *unit {
            k += 1;
        }
        if k >= 2 {
            match best {
                None => best = Some((p, k)),
                Some((_bp, bk)) => {
                    // Prefer the smallest period at maximal repeats: a p=1
                    // stutter subsumes its multiples. Keep the FIRST (smallest
                    // p) whose k >= 2 — but if a larger p repeats MORE times
                    // in the tail window, the tail ratio uses the max k.
                    if k > bk {
                        best = Some((p, k));
                    }
                }
            }
        }
    }
    match best {
        Some((p, k)) => (Some(p), k),
        None => (None, 0),
    }
}

/// Distinct n-grams / total n-grams, sequential ascending fold into a sorted
/// vec (deterministic; no hash iteration order).
fn distinct_ngram_ratio(tokens: &[&str], n: usize) -> f64 {
    if tokens.len() < n {
        return 1.0; // cannot form any n-gram: vacuously dense-free
    }
    let total = tokens.len() - n + 1;
    let mut grams: Vec<&[&str]> = Vec::with_capacity(total);
    for i in 0..total {
        grams.push(&tokens[i..i + n]);
    }
    grams.sort_unstable();
    let mut distinct = 1usize;
    for w in grams.windows(2) {
        if w[0] != w[1] {
            distinct += 1;
        }
    }
    distinct as f64 / total as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_prose_does_not_flag() {
        // Varied prose: no cycle, healthy density.
        // Genuinely varied: every token unique (mod-numbering would invent a
        // cycle — the first fixture run proved the census catches it, which
        // is exactly the mechanism working).
        let words: Vec<String> = (0..200).map(|i| format!("word{i:04}")).collect();
        let toks: Vec<&str> = words.iter().map(|s| s.as_str()).collect();
        let r = census_tokens(&toks, DEFAULT_LEN_LIMIT);
        assert!(!r.flagged(), "varied prose must not flag: {r:?}");
        assert!(r.distinct_ngram_ratio > 0.5);
    }

    #[test]
    fn token_stutter_flags() {
        // A single-token loop to the horizon: the QDA proxy-liked class.
        let words: Vec<String> = (0..300)
            .map(|i| {
                if i < 20 {
                    format!("seed{}", i)
                } else {
                    "the".to_string()
                }
            })
            .collect();
        let toks: Vec<&str> = words.iter().map(|s| s.as_str()).collect();
        let r = census_tokens(&toks, DEFAULT_LEN_LIMIT);
        assert_eq!(r.suffix_cycle_period, Some(1));
        assert!(r.tail_repeat_count >= LOOP_RATIO_MIN);
        assert!(r.flagged());
    }

    #[test]
    fn phrase_loop_flags() {
        // An 8-token phrase loop — the phrase-scale repetition class.
        let phrase = [
            "the", "king", "of", "the", "golden", "hall", "spoke", "thus",
        ];
        let mut words: Vec<String> = (0..20).map(|i| format!("w{i}")).collect();
        for _ in 0..40 {
            for w in phrase {
                words.push(w.to_string());
            }
        }
        let toks: Vec<&str> = words.iter().map(|s| s.as_str()).collect();
        let r = census_tokens(&toks, DEFAULT_LEN_LIMIT);
        assert_eq!(r.suffix_cycle_period, Some(8));
        assert!(r.flagged(), "phrase loop must flag: {r:?}");
    }

    #[test]
    fn truncation_census_reads_the_limit() {
        let words: Vec<String> = (0..DEFAULT_LEN_LIMIT)
            .map(|i| format!("word{i:04}"))
            .collect();
        let toks: Vec<&str> = words.iter().map(|s| s.as_str()).collect();
        let r = census_tokens(&toks, DEFAULT_LEN_LIMIT);
        assert!(r.truncated_at_limit);
        assert!(!r.flagged(), "truncation alone is not a loop");

        let short: Vec<&str> = toks[..100].to_vec();
        let r2 = census_tokens(&short, DEFAULT_LEN_LIMIT);
        assert!(!r2.truncated_at_limit);
    }

    #[test]
    fn short_generations_abstain() {
        let toks = vec!["a"; 10];
        let r = census_tokens(&toks, DEFAULT_LEN_LIMIT);
        assert!(!r.flagged(), "below MIN_TOKENS never flags");
    }

    #[test]
    fn density_collapse_flags_without_an_explicit_tail_cycle() {
        // Dense repetition spread through the body (not a clean suffix):
        // distinct-8-gram ratio collapses, the suffix cycle may be absent —
        // the density axis must fire alone (QDA's body-collapse class).
        let mut words: Vec<String> = Vec::new();
        let block = [
            "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
        ];
        for i in 0..40 {
            for w in block {
                let _ = i;
                words.push(w.to_string());
            }
        }
        let toks: Vec<&str> = words.iter().map(|s| s.as_str()).collect();
        let r = census_tokens(&toks, DEFAULT_LEN_LIMIT);
        assert!(
            r.distinct_ngram_ratio <= DENSITY_LOOP_MAX,
            "{:?}",
            r.distinct_ngram_ratio
        );
        assert!(r.flagged());
    }

    #[test]
    fn thresholds_are_pinned_constants() {
        // The issue law: pinned in-source. The values are load-bearing for
        // the T2 gate's cross-run comparability — a change here is a
        // threshold change, never a silent drift.
        assert_eq!(LOOP_RATIO_MIN, 3);
        assert_eq!(DENSITY_LOOP_MAX, 0.35);
        assert_eq!(DENSITY_NGRAM, 8);
        assert_eq!(MIN_TOKENS, 64);
    }

    #[test]
    fn text_entry_matches_token_entry() {
        let prose = "the quick brown fox jumps over the lazy dog. ";
        let long: String = prose.repeat(30);
        let a = census_text(&long, DEFAULT_LEN_LIMIT);
        let toks: Vec<&str> = long.split_whitespace().collect();
        let b = census_tokens(&toks, DEFAULT_LEN_LIMIT);
        assert_eq!(a, b);
    }
}
