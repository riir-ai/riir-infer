//! Issue 1214 T0 — the integer-exact reference lane's arithmetic core.
//!
//! Transcribed decision-content from detllm (`nathanrs/detllm` @
//! `9c3f1c9cdd42294087cc0507992c21dfde944bea`, NO LICENSE upstream —
//! decision-content-only posture; citations: `.research/027` §quotes,
//! upstream `intmath.py:7-17`, `dyadic.py:3-6,166-205`). The rounding
//! policy is FROZEN here exactly as upstream froze it after their
//! milestone 1: every power-of-two rescale goes through
//! [`rshift_round`] (ties toward +inf); [`round_half_away_div`] exists
//! only where a non-power-of-two divisor appears (softmax
//! normalization). Changing either law breaks every golden below and
//! every bit-identity claim the lane will ever make — this module is
//! the lane's semantics root.
//!
//! Everything here is integer-only by construction (the type-level
//! no-float rule); the source-scan test in this module's test tree
//! refuses float tokens so the guard holds by test, not by hope.

mod dyadic;
mod num;

pub use dyadic::Dyadic;
pub use num::{argmax_first, ilog2_floor, isqrt, round_half_away_div, rshift_round};

#[cfg(test)]
mod no_float_guard {
    /// The type-level no-float rule, enforced as a source scan: the lane's
    /// files must not contain float type tokens or float literals outside
    /// comments. Stronger than a runtime mode switch — a stray `0.5` or
    /// `f32` in the numerical path cannot reach a build without this test
    /// reding first. Line comments are stripped before scanning (prose
    /// mentions of "f32" in a doc comment are not code).
    #[test]
    fn lane_sources_carry_no_float_tokens() {
        let files = [
            ("int_ref/mod.rs", include_str!("mod.rs")),
            ("int_ref/num.rs", include_str!("num.rs")),
            ("int_ref/dyadic.rs", include_str!("dyadic.rs")),
        ];
        // the token list is assembled through concat! so THIS source file
        // never contains its own tokens (the guard must not self-match).
        let tokens = [
            concat!("f", "32"),
            concat!("f", "64"),
            concat!(".", "5"),
            concat!(" as ", "f"),
            concat!(".exp", "("),
            concat!(".powf", "("),
            concat!(".sqrt", "("),
            concat!(".ln", "("),
            concat!(".log2", "("),
        ];
        for (name, src) in files {
            for (i, line) in src.lines().enumerate() {
                // strip line comments before scanning; block comments in
                // these files are doc comments on `//!`/`///` lines, which
                // the `//` strip covers too.
                let code = match line.find("//") {
                    Some(c) => &line[..c],
                    None => line,
                };
                for tok in tokens {
                    assert!(
                        !code.contains(tok),
                        "no-float guard: {name}:{line}: float token in code: {code}",
                        name = name,
                        line = i + 1,
                        code = code.trim(),
                    );
                }
            }
        }
    }
}
