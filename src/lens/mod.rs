//! Op-level RelP lens maps (Research 666, katgpt-rs Issue 999).
//!
//! Provenance: `stdlogits/neuron-lens` @ `849bfe16` (MIT) — the R/S Jacobian
//! lenses computed under RelP hand-propagation rules instead of autograd or
//! finite differences. This module ships the OP-LEVEL forward (`jvp`) and
//! adjoint (`vjp`) pairs that the layer-stack lens fit (T2d) composes into a
//! `linearize`/`jvp`/`vjp` triple over a whole checkpoint.
//!
//! # The rules (mirroring the upstream estimator)
//!
//! - **LN rule** — decoder RMSNorms hold their `rsqrt` constant: the tangent
//!   map is the diagonal `(r ⊙ γ)`, which is self-adjoint.
//! - **Identity rule** — `silu(g)` linearized at its local derivative. Exact
//!   for a scalar smooth function: the local linearization IS the derivative.
//! - **Half rule** — a two-factor product passes half of each factor's change
//!   (`β = 0.5`): `d(s·u) = ½(s·du + u·ds)`. Applied to the SwiGLU gate·up
//!   product and the GDN output gate.
//! - **True derivative** — RoPE (an exact per-pair rotation; `vjp` = inverse
//!   rotation) and every purely linear map (projections, causal depthwise
//!   `conv1d`) admit exact adjoints, no rule needed.
//!
//! Empirically the approximation READS better than the true derivative (the
//! R-lens leads the J-lens at predicting steering L4–L23; RelP halves the
//! S-lens's input-side noise) — the rules are the point, not a shortcut.
//!
//! # Discipline
//!
//! - All maps are free functions over slices; caller-provided scratch;
//!   zero allocation. The fit is offline (once per checkpoint), never a
//!   serving path.
//! - Weight layout is the crate's convention everywhere: row-major
//!   `[rows, cols]`, `y = W @ x` (see `crate::types::matmul`).
//! - `jvp` fns reuse the FORWARD's exact math where the op is exact
//!   (`crate::rope`, `crate::types::matmul`) so the fit inherits bit-parity
//!   with the serving forward; `vjp` fns are the exact adjoints of those
//!   maps, pinned per-op by adjointness tests
//!   (`⟨J·dx, dy⟩ == ⟨dx, Jᵀ·dy⟩`) and a RoPE round-trip identity.
//!
//! PoC gates G1–G4 live in `../katgpt-rs/.issues/999_rs_lens_fit_validate_poc.md`.

pub mod ops;
