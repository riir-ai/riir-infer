//! Issue 1214 — the integer-exact reference lane's arithmetic core (T0)
//! + tensor bookkeeping and DI op layer (T1a).
//!
//! Transcribed decision-content from detllm (`nathanrs/detllm` @
//! `9c3f1c9cdd42294087cc0507992c21dfde944bea`, NO LICENSE upstream —
//! decision-content-only posture; citations: `.research/027` §quotes +
//! the per-file def lines pinned in each module doc). The rounding
//! policy is FROZEN exactly as upstream froze it after their milestone
//! 1: every power-of-two rescale goes through [`rshift_round`] (ties
//! toward +inf); [`round_half_away_div`] exists only where a
//! non-power-of-two divisor appears (softmax normalization, the fit
//! path). Changing either law breaks every golden below and every
//! bit-identity claim the lane will ever make — this module is the
//! lane's semantics root.
//!
//! Everything here is integer-only by construction (the type-level
//! no-float rule); the source-scan test in this module's test tree
//! refuses float tokens so the guard holds by test, not by hope.

mod artifact;
mod dyadic;
mod model;
mod num;
mod ops;
mod tensor;

pub use artifact::{ArtifactDigests, StMap, assemble_artifact, load_dir, parse_container};
pub use dyadic::Dyadic;
pub use model::{
    IntRefArtifact, IntRefConfig, IntRefModel, KvCache, LayerWeights, LinearWeights, greedy_pick,
};
pub use num::{argmax_first, ilog2_floor, isqrt, round_half_away_div, rshift_round};
pub use ops::{
    OUT_FRAC_BITS, P_OUT_BITS, PROB_ONE, ROPE_FRAC_BITS, RopeTables, SIG_BITS, SIG_ONE,
    apply_int_rope, di_exp, di_rmsnorm, di_rmsnorm_gamma, di_softmax, di_softmax_row, di_swiglu,
    int_gemm, int_gemm_u8i8, int_gemm_u16i8,
};
pub use tensor::{
    Mat, RES_MAX_BITS, RES_MIN_BITS, fit_dyadic_ratio, norm_scale, quant_i8_pertoken,
    requant_i8_rowcol, requant_i8_static, requant_i32_common, residual_add, residual_from_delta,
    residual_renorm, scaled_round_div,
};

#[cfg(test)]
mod no_float_guard {
    /// The type-level no-float rule, enforced as a source scan: the lane's
    /// files must not contain float type tokens or float literals outside
    /// comments. Stronger than a runtime mode switch — a stray `0.5` or
    /// float type token in the numerical path cannot reach a build
    /// without this test reding first. Line comments are stripped before
    /// scanning (prose mentions in a doc comment are not code).
    const LANE_FILES: &[(&str, &str)] = &[
        ("int_ref/mod.rs", include_str!("mod.rs")),
        ("int_ref/num.rs", include_str!("num.rs")),
        ("int_ref/dyadic.rs", include_str!("dyadic.rs")),
        ("int_ref/tensor.rs", include_str!("tensor.rs")),
        ("int_ref/model.rs", include_str!("model.rs")),
        ("int_ref/artifact.rs", include_str!("artifact.rs")),
        ("int_ref/ops/mod.rs", include_str!("ops/mod.rs")),
        ("int_ref/ops/di_exp.rs", include_str!("ops/di_exp.rs")),
        (
            "int_ref/ops/di_softmax.rs",
            include_str!("ops/di_softmax.rs"),
        ),
        (
            "int_ref/ops/di_rmsnorm.rs",
            include_str!("ops/di_rmsnorm.rs"),
        ),
        ("int_ref/ops/di_swiglu.rs", include_str!("ops/di_swiglu.rs")),
        ("int_ref/ops/rope.rs", include_str!("ops/rope.rs")),
        ("int_ref/ops/matmul.rs", include_str!("ops/matmul.rs")),
    ];

    /// The banned token list, assembled through concat! so THIS source
    /// file never contains its own tokens (the guard must not
    /// self-match).
    const TOKENS: &[&str] = &[
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

    /// The detector: returns every (line, token, code) hit in `src`.
    /// Split out of the real-file scan so the falsification arms can
    /// prove it FIRES (the gate working is part of the claim —
    /// `.issues/1214` G1-falsification arms).
    fn float_token_hits(name: &str, src: &str) -> Vec<String> {
        let mut hits = Vec::new();
        for (i, line) in src.lines().enumerate() {
            // strip line comments before scanning; block comments in
            // these files are doc comments on `//!`/`///` lines, which
            // the `//` strip covers too.
            let code = match line.find("//") {
                Some(c) => &line[..c],
                None => line,
            };
            for tok in TOKENS {
                if code.contains(tok) {
                    hits.push(format!(
                        "{name}:{}: float token in code: {}",
                        i + 1,
                        code.trim()
                    ));
                }
            }
        }
        hits
    }

    #[test]
    fn lane_sources_carry_no_float_tokens() {
        for (name, src) in LANE_FILES {
            let hits = float_token_hits(name, src);
            assert!(
                hits.is_empty(),
                "no-float guard tripped:\n{}",
                hits.join("\n")
            );
        }
    }

    #[test]
    fn guard_detector_fires_on_planted_float() {
        // falsification arm 1: each banned token, planted in a snippet,
        // must be CAUGHT (the detector is live, not a green zero). Every
        // string piece below is SMALLER than the token it assembles —
        // the same discipline as the TOKENS list — so this source never
        // self-matches.
        let planted = [
            ("float type", concat!("let x: ", "f", "32", " = 0;")),
            ("double type", concat!("let y = 1 ", "as ", "f", "64", ";")),
            ("float literal", concat!("let z = 0", ".", "5", " * w;")),
            ("cast form", concat!("let w = b ", "as ", "f", "32", ";")),
            ("exp call", concat!("let e = v", ".ex", "p", "();")),
            ("powf call", concat!("let p = v", ".pow", "f", "(2);")),
            ("sqrt call", concat!("let s = v", ".sq", "rt", "();")),
            ("ln call", concat!("let l = v", ".l", "n", "();")),
            ("log2 call", concat!("let l = v", ".lo", "g2", "();")),
        ];
        for (what, snippet) in planted {
            let hits = float_token_hits("planted.rs", snippet);
            assert!(!hits.is_empty(), "guard missed a planted {what}: {snippet}");
        }
    }

    #[test]
    fn guard_ignores_float_words_in_comments() {
        // falsification arm 2: prose mentions of the banned shapes in a
        // comment must NOT fire (the `//` strip is the mechanism)
        let snippet = concat!(
            "// returns the ",
            "f",
            "32",
            " lanes; x",
            ".sq",
            "rt",
            "() below\n",
            "let v = 1_i64;"
        );
        assert!(float_token_hits("comments.rs", snippet).is_empty());
    }

    #[test]
    fn lane_root_reachable_for_consumers() {
        // a wiring canary, not a numerics check: the lane's public
        // surface re-exports the ops layer (the forward in T1b consumes
        // exactly these paths)
        let _ = super::P_OUT_BITS;
        let _ = super::OUT_FRAC_BITS;
        let _ = super::ROPE_FRAC_BITS;
        let _ = super::SIG_ONE;
    }
}
