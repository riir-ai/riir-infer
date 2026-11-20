# Bench 1249 — plan 503 T0.1/T0.2 lattice-census gate run (compile + fixtures + clippy + real-checkpoint smoke + determinism)

**Date:** 2026-11-20 (zed-989683dc@4090). Plan: [riir-train 503](../../../riir-train/.plans/503_lattice_conditioning_xhc_gdn.md) Phase 0; authoring receipt riir-infer `d112204`; this bench = the compile-gated residue discharged.

## Cells

| cell | posture | verdict |
|---|---|---|
| fixtures | `cargo test --lib --features lattice_census lattice_census` (dev, isolated `CARGO_TARGET_DIR=F:/tmp/i503-t0`) | **9/9 PASS** |
| fixtures default posture | same minus the feature | **0 passed, exit 0** — the green-zero law honored (gated module compiles to nothing) |
| clippy armed | `cargo clippy --lib --features lattice_census -- -D warnings` | **exit 0** (after 2 manual mechanicals: unused `frob2` removed with its orphaned `DiagonalFold` trait; collapsible-if → let-chain, edition 2024) |
| bin release build + real-model smoke | `--release --features lattice_census,bonsai2_hadamard` on `Ternary-Bonsai-2-27B-PQ2_0.gguf`, `--n 512 --cap-every 64` | **PASS**: 64 layers / 48 GDN / n_embd 5120 / folded=true (loud disclosure fired as designed); census 673.4 s CPU ternary; report 1 099 242 B |
| determinism G1 | re-run, byte `cmp` | **BYTE-IDENTICAL** — digest `255a239584efd4958069c14d4abbcf4b896fb73e5897d5185551c461972910bb` both runs |
| rustfmt | `rustfmt --edition 2024` both files | clean (edition 2024 — let-chains legal) |

## The baseline row (T0.1's deliverable)

**mean r_ent = 3.3442** (48 GDN layers × 8 capture points × 16 heads, unweighted mean over layers) — the untrained-baseline geometry the Phase-2 lattice arm is judged against. Report: `.benchmarks/i503_lattice_census_baseline.json` (1.1 MB), BLAKE3 `255a2395…` over the canonical fixed-precision payload. b-twin artifact deleted post-verify (digest is the record).

## The gate-caught defect (the fixtures are load-bearing)

The first fixture run caught a **real numeric defect** in the authored `FilmFilter::fixed_coeff`: `1.0 + (-((ug - 0.5) * 4.0).exp())` parses in Rust as `1.0 + (-(((ug-0.5)*4.0).exp()))` — **unary minus binds LOOSER than the method call**, so the "sigmoid" compiled as `1/(1 − exp(x))`, which is unbounded and sign-flipped (measured sg = 1.185, −0.248, 6.63 on valid u ∈ [0,1); γ spilled to 3.618 vs the ±5 % bound). The Python replication model had used the correct `exp(−x)`, so the defect was invisible at authoring. Fix: bind `xg`/`xb` first, `(-xg).exp()` — with the precedence trap documented in-rule. The `query_gain_oracle` failure was a fixture bug (q mutated across cases); fixed with a fresh `q2`.

Bin-side compile residue also discharged: wrong import path (`transformer` → `deltanet`), `parse_args` → `Result<Args>`, edition-2024 binding modes (`&t` patterns).

## Box state

RAM 13.0 GB free post-sweep, AC power; target dir isolated `F:/tmp/i503-t0` (1.9 GB, volume hygiene per the E:-flagged standing rule — E: exFAT still Warning/Full-Repair-Needed); no other compute consumers (the i677 join on C: scratch, negligible).

## Posture notes

- T0.2's FiLM probe: the filter math is fixture-verified; the injection seam on the FOLDED 27B still needs the matvec-wrap landing (the input-proj hook is bypassed on rotated layers — the disclosure fired in the smoke's stdout as designed). The unarmed baseline above is unaffected.
- The query-gain capture arm (transient `q_normed`) remains the scratch-field + 2 write-sites step — reports carry `g1=na` until it lands (self-describing).
- `CARGO_TARGET_DIR=F:/tmp/i503-t0` teardown: owed at the plan's close (the F: scratch class, per the cleanup law this is a per-plan dir — 1.9 GB, removed when 503 Phase 0 closes).
