//! The DI op layer (Issue 1214 T1a) — the integer non-linear ops of the
//! DetLLM port, each a decision-content transcription of one upstream
//! def (pinned sha `9c3f1c9cdd42294087cc0507992c21dfde944bea`, NO
//! LICENSE — decision-content-only posture; overview `.research/027`).
//!
//! - [`di_exp`] — `ops/di_exp.py:21`, Algorithm 1 (shift decomposition,
//!   the `m_f = m + (m>>1) - (m>>4)` constant; the implicit non-dyadic
//!   output scale is always consumed by a ratio where it cancels);
//! - [`di_softmax`] — `ops/di_softmax.py:33`, Algorithm 2 (sentinel
//!   masked row max, `P_OUT_BITS = 15` probs, no float anywhere);
//!   upstream's optional Eq. 10 clipping knob is deliberately NOT
//!   ported (an ablation arm their NOTES.md §4 shows strictly dominated
//!   by DI-Exp's own underflow);
//! - [`di_rmsnorm`] / [`di_rmsnorm_gamma`] — `ops/di_rmsnorm.py:33,55`,
//!   Algorithm 4 (scale-free pre-shift + bitwise isqrt);
//! - [`di_swiglu`] — `ops/di_swiglu.py:39`, Algorithm 3 (the stable
//!   two-branch sigmoid their NOTES.md mandates over the paper's row-max
//!   form; `SIG_BITS = 8`);
//! - [`apply_int_rope`] — `ops/rope.py:38` (tables are OFFLINE artifacts;
//!   upstream `ops/rope.py:25` is the float builder and stays out of this
//!   module tree by the type-level no-float rule);
//! - [`int_gemm`] family — `backends.py:84,138,152` (exact int8×int8→int32
//!   product-sum; the hi/lo int8 split for 15-bit probabilities; the −64
//!   unsigned trick). `GUARD_ROWS` is a CUDA cuBLASLt over-read defense —
//!   irrelevant to this CPU reference and noted, not ported.

pub mod di_exp;
pub mod di_rmsnorm;
pub mod di_softmax;
pub mod di_swiglu;
pub mod matmul;
pub mod rope;

pub use di_exp::di_exp;
pub use di_rmsnorm::{OUT_FRAC_BITS, di_rmsnorm, di_rmsnorm_gamma};
pub use di_softmax::{P_OUT_BITS, PROB_ONE, di_softmax, di_softmax_row};
pub use di_swiglu::{SIG_BITS, SIG_ONE, di_swiglu};
pub use matmul::{int_gemm, int_gemm_u8i8, int_gemm_u16i8};
pub use rope::{ROPE_FRAC_BITS, RopeTables, apply_int_rope};
