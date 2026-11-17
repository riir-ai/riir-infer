# Bench 1236 — i1221 T3(a): the per-loop KL spectra on the looped-trained pin

**Status:** COMPLETE — verdict **LOOP-TRAINED INVARIANCE CONFIRMED (the dense arm's mirror image)** — the Dory pin's per-loop KL collapses toward 0 with depth (KL(full‖exit) means 13.8–24.1 → 5.05/0.25/2.99 → 0.78/0.031/0.52 → **exactly 0** at the trained k=4), the KL profiles are monotone-decaying at 100% of positions on all three prompts, and the last-exit identity holds bit-exactly — the trained recurrent block does what the untrained dense chunk refused to do (Bench 1235: KL floor 3–4 nats, no convergence). Report-only; the only gates are the lib fixtures (9/9) + the runner's two real-pin receipts, both TRUE on every prompt.
**Date:** 2026-11-17 (cell 12:29–12:33Z amo; instrument + fixtures + cell + record one session under the T3(a) claim `zed-84603cf7@amo`, epoch 1795905000).
**Scope:** riir-infer only. Lane: `.issues/1221` (T3(a) fold rides this record); pre-registration: `.plans/636` (read-order leg 2, after (d)'s calibration).

## The cell

```
cargo run --release --features dory \
  --example i1221_t3a_kl_spectra -- \
  --gguf ../riir-train/data/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf \
  --out .benchmarks/i1221_t3a_kl_spectra_cell
```

The lens: ONE k=4 prefill with per-pass state snapshots armed (`prefill_snapshots`), then the readout replayed on each snapshot (`readout_rows` — output block + LM head over the mid-loop state). "Exit at depth d" is thus a d-loop prefill **by construction** — the fixtures pin the lens premise at EVERY position (exit_τ ≡ fresh (τ+1)-loop prefill, bit-identical; the Dory readout depends only on the final recurrent state, so the Dory form is strictly stronger than katgpt-rs Bench 899's pos-0-only lens premise). Readouts per (prompt, depth): KL(full‖exit_d) per position in f64 (the Bench-899 lse algebra), KL(exit_1‖exit_d) (the deepening axis), last-position argmax/margins + the all-position flip count; per-position summaries: effective depth (first post-peak loop within 1.0 nat, Bench-899 definition), agreement exit, monotone fraction.

- Pin: `NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf` blake3 `f946e530236859ffc4e9a9ceeb3041c8485f6a4d51a56c31153732a50b37f393`, arch dory, k_full=4 (the header's own `dory.num_loops` — the plan's k=1..4 IS the trained range), 72 blocks = I16/R40/O16, vocab 131072, n_embd 2560, cache per_loop.
- Prompts: the smoke's pseudo-token LCG (uniform [4, vocab), per-set seed = seed + idx), seq 8 × 3 sets — the (d) law verbatim, so the two arms share a prompt shape.
- STAMP: `67ef17db` at cell time (the F2 G1-fix landing; my instrument files were the only other working-tree changes — dory-side, disjoint from the gpu lane).

## Provenance

- `PROVENANCE: power=AC Power load=4.01-5.39 swap=n/a canary=skipped powermode=0(auto)` (bench_preflight PASS; amo, 30 threads, release). **Shared-window structural read disclosed per the `96b44752` law** (the routing: zed-78e08ea6's lane holds bmo exclusive; this cell ran amo-side, no timing rows — the readouts are exact-determinism quantities, wall_ms context only).
- Cell wall: 103.9 s of lens runs, 280.3 s total (12 exits ≈ one output-block pass + 8×131072 lm-head rows each; model load 2.0 s).
- **Resume receipt:** a second run verified the lineage (12 rows present) and skipped all pairs — `resume: 12 row(s) present, lineage verified`, 0.0 s, exit 0. The (d) session's resume-scan fix class is covered: this runner's scanner reads the FLAT row spelling (`prompt_idx`/`depth` at top level).

## The rows (13 lines = header + 12; resumable lineage intact)

**KL(full ‖ exit_d), mean/max per prompt (nats, f64):**

| prompt | d=1 | d=2 | d=3 | d=4 (= trained) |
|---|---|---|---|---|
| 0 | 13.794 / 51.562 | 5.051 / 21.289 | 0.777 / 4.705 | **0.0000 / 0.0000** |
| 1 | 14.728 / 52.282 | 0.248 / 1.290 | 0.031 / 0.183 | **0.0000 / 0.0000** |
| 2 | 24.097 / 81.459 | 2.994 / 23.257 | 0.519 / 4.038 | **0.0000 / 0.0000** |

**The shape the lane was built to see:** monotone decay at **24/24 positions** (all prompts, all profiles, tolerance 1e-9), effective depth 1–4 (position-mixed: most positions read 1–2, a few need the full 4), agreement exit 1–2 (the last-position argmax settles by depth 2 everywhere — the d=1 flips on last position are 1/3 prompts, all-position flips 3–4/8 settle to 0 by d=3–4), and the deepening axis KL(exit_1‖exit_d) GROWS with depth (8.4→13.1→15.9 on p0; 7.6→16.9→20.3 on p1; 12.5→22.6→27.4 on p2) — the loop deepens monotonically while the readout converges home, exactly the Dory k-scaling premise's measured form on a real looped-trained checkpoint.

**The dense-arm calibration (plan 636's purpose for (d)):** Bench 1235's untrained dense chunk sat at a KL FLOOR of 2.9–4.6 nats with λ_t rising toward 0.9 — contraction without convergence. The trained pin sits at the opposite pole: KL collapses ~14–24 → 0 exactly at the trained depth, monotone 100%. The (d) baseline reads as the "no loop training" anchor it was built to be; the trained invariance is orders of magnitude below it (0 nats vs 3–4 nats floor at the trained depth). Effective depth spreading 1–4 by position is the first measured signal for the elastic-k serving posture (T2's cache-at-k): most positions could exit at 1–2 loops under a per-position halting rule — T3(b)'s halter traces inherit exactly this fixture.

## Identity receipts (the real-pin gates)

1. **Kill-switch — TRUE ×3:** the lens-armed k=4 prefill's final logits are BIT-IDENTICAL to a plain unarmed prefill on every prompt (arming clones snapshots; it never perturbs the computation).
2. **Lens premise (strongest form) — TRUE ×3:** exit_4 replayed from the snapshot ≡ the full logits at EVERY position, bit-exact (not just Bench 899's pos-0 — the Dory readout's independence from the loop cache makes the lens a bit-exact d-loop prefill oracle at every depth; fixture-pinned for all four depths on the synth, receipt-pinned at d=4 on the real pin).
3. **Lib fixtures 9/9** (`tests/i1221_t3a_spectra_gates.rs`, required-features dory): kill-switch, lens-premise-all-positions, last-exit identity (KL exactly 0), not-inert (state-level deepening — the synth head is one-hot saturated by construction, so synth KL magnitudes are meaningless and the gate asserts the state diff + non-constant logit diff instead, disclosed in-file), KL algebra vs a direct softmax fold (f64 agreement to 1e-12), the Bench-899 summarizer definitions (incl. the pre-peak trap row), row serde + resume accept path, two-run byte-identity, and the prefill-refactor re-pin of T2's prefill≡decode identity.

## Files

- `src/dory/forward.rs` — verbatim-seam extraction: `prefill` → `prefill_snapshots` (armed/unarmed; unarmed delegates, bit-identical by construction), `readout_rows` (the output-block + LM-head exit), + the snapshot push after each pass's boundary norm.
- `src/dory/spectra.rs` (new) — the instrument core: `run_lens`/`exit_row`/`kl_positions`/`exit_kl` (the Bench-899 lse algebra at f64)/`argmax_and_margin`/`effective_depth`/`agreement_exit`/`monotone_fraction`/`SpectraRow`/`verify_resume_lineage`.
- `tests/common/synth_dory.rs` (new) — the dory synth kit EXTRACTED from `tests/i1221_dory_loader_synth.rs` (the loader test now consumes it; one writer, two consumers).
- `tests/i1221_t3a_spectra_gates.rs` (new) + the `[[test]]`/`[[example]]` required-features rows (`dory` only — the lane gate).
- `examples/i1221_t3a_kl_spectra.rs` (new) — the resumable cell runner.
- Postures: clippy −D warnings clean at default / no-default / dory / dory+deltanet (all-targets); lib 367 green at `dory,deltanet_inference` (T2's 354 floor intact — the floor fires downward only); the workspace `--all-targets` clippy has ONE pre-existing failure in `riir-infer-laya`'s `i1216_t0_headroom` example (E0433 at HEAD `67ef17db`, verified pre-existing in a detached worktree — NOT this lane's; the example lacks a required-features row for the feature-gated `laya` module, a P43-class row for the laya lane to fix, noted in what-remains).

## Session

zed-84603cf7@amo (amo, M5 Ultra, AC) — routing receipt from zed-78e08ea6 (T3(a) unowned, structural-read window). **Lease stays live to 2026-11-19T12:30Z** for the record's follow-through (the fold rides the next i1221 commit if the lane pauses).
