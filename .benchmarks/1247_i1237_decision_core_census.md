# Bench 1247 — Issue 1237: decision-core census over laya (ACE at the R1 scorer-input readout)

**Status:** RECORDED 2026-11-19 (katgpt-rs-93@amo). Pre-registration in `.issues/1237` (`15954292`), set before any number. Shared rig job `901d28cb` on amo (Metal forwards). Accuracy tables only; no latency is claimed, so no box state is needed.

## Setup (as pre-registered)

- **Readout R1:** the head's scorer-input marker rows. `f` = the scorer logit. The Jacobian is analytic, in f64 (`Scorer64`), and unit-pinned against central finite differences.
- **Fidelity gate:** `Scorer64` against the lane's own raw logits on the first 32 forwards. Max |Δ| was 7.6e-7 (english), 8.7e-7 (typed) and 7.2e-7 (multilingual), all under the 1e-3 gate.
- **Extraction:** katgpt-core `relevance_core` (consumed, not forked). The PCA control is the same extract with `S = I`, which relevance_core documents as its positive control. The random control is `subspace_intervention::random_basis_into`, seed `0x1237ACE0`.
- **Suite:** riir-reflex `tetris_oracle_laya_en_v4.jsonl`, the first 200 records, 4604 `noul` forwards. Records 0–99 fit (4604 marker rows, Cov rows/d = 4.50 at d = 1024 and 5.99 at d = 768). Records 100–199 are the held-out eval (2302 forwards).
- **Decision:** `sign(l_true − l_false)`. Keep = `μ + QQᵀ(h−μ)`, remove = `h − QQᵀ(h−μ)`, applied to both marker rows.

## (1) Rank and causal reads

| ckpt | d | eval true-rate | E0.90 rank | energy share r = 1 | participation ratio |
|---|---:|---:|---:|---:|---:|
| english | 1024 | 0.671 | **1** | 0.993 | 1.01 |
| typed | 1024 | 0.186 | **1** | 0.994 | 1.01 |
| multilingual | 768 | 0.005 | **1** | 0.999 | 1.00 |

Keep-retention and remove-flip on the held-out half, as core / PCA / random:

| ckpt | r | keep | remove-flip | SUFFICIENT | NECESSARY |
|---|---:|---|---|---|---|
| english | 1 | 0.994 / 0.995 / 0.702 | 0.644 / 0.639 / 0.000 | ✗ (ties PCA) | ✓ |
| english | 2 | 0.997 / 0.997 / 0.992 | 0.294 / 0.295 / 0.000 | ✗ | ✓ |
| english | 4–64 | 1.000 / 1.000 / 0.865–0.976 | 0.42–0.82 / 0.57–0.71 / ≤0.001 | ✗ (ties PCA) | ✓ |
| typed | 1 | **0.999 / 0.197 / 0.358** | 0.119 / 0.008 / 0.000 | **✓** | ✗ (< 0.25) |
| typed | 2 | 0.998 / 0.998 / 0.940 | 0.159 / 0.155 / 0.000 | ✗ | ✗ |
| typed | 4–64 | 1.000 / 1.000 / 0.896–0.999 | 0.23–0.55 / 0.32–0.42 / ≤0.003 | ✗ (ties PCA) | ✓ except r = 32 |
| multilingual | 1–64 | ≥ 0.997 for all three arms | 0.41–0.88 / 0.44–0.86 / 0.000 | ✗ | ✓ |

Full per-rank rows are in `*_report.json`.

**Pre-registered verdicts:**
- **english:** LOW_DIM_CORE **NOT CLAIMED**. The core is sufficient in substance (0.994 at r = 1 against a 0.671 majority) and necessary at every rank. But the rule requires beating PCA, and at this readout the rank-1 relevance direction is the top-variance direction: keep ties PCA to the third decimal. That is a definitional tie, recorded exactly as the rule reads.
- **typed:** LOW_DIM_CORE **CLAIMED at r = 1**. Keep is 0.999 against 0.197 for PCA and 0.358 for random. Here the top-variance direction is NOT the decision direction, which is the case ACE exists to separate.
- **multilingual:** **UNINFORMATIVE**. On this English-phrased suite it answers false on 99.5% of the eval forwards, so every keep arm (random included) retains ≥ 0.995. The necessity column stands; sufficiency needs a balanced multilingual suite.

**Reading the remove-flip column:** mean-ablation pushes both marker rows toward μ, so a removed forward takes the mean's decision. Flip is therefore capped by the class balance, and that cap is why it is not monotone in r. The random control's remove-flip is ≤ 0.003 at every rank on every checkpoint, so the necessity reads are not projection artifacts.

## (2) english vs typed core agreement

`compare` = principal-angle `basis_similarity`; the random baseline is r/d.

| r | similarity | baseline | ratio |
|---:|---:|---:|---:|
| **1 (pre-registered: english E0.90 rank)** | **0.951** | 0.00098 | **974×** |
| 2 | 0.972 | 0.00195 | 498× |
| 4 | 0.908 | 0.00391 | 232× |
| 8 | 0.716 | 0.00781 | 92× |
| 16 | 0.644 | 0.0156 | 41× |
| 32 | 0.548 | 0.0313 | 18× |

**AGREEMENT CLAIMED** (≥ 5× at the pre-registered rank, and at every rank). The disclosed caveat stands: the two checkpoints are separately fine-tuned on one ModernBERT-large init, so their coordinates are shared through that init. This is agreement within one family, not the paper's cross-family claim.

## (3) Transport

Deferred as pre-registered: it needs a second paired family.

## What it says

At the scorer's input the laya decision rides an essentially rank-1 relevance direction (≥ 99.3% of the energy), and that direction is shared across the english and typed fine-tunes (cos² 0.95). In english that direction is also the dominant variance direction. In typed it is not: PCA-1 there retains only 0.197 of decisions, below the 0.814 majority floor.

This is the R1 readout only, the last d-dim layer before a scalar. A near-rank-1 relevance there is expected structurally: one LN→MLP→scalar head over a binary decision. The deeper read is the named **R0 follow-up**: encoder output through the two head layers, finite differences.

No serving posture follows from this. R1 is not a dist-lane activation, so the issue's downstream lossy-projection idea does not arm.

## Code

The measured code is `code_at_measurement.patch`: `riir-infer-laya` feature `decision_core_census`, `laya/riir/decision_core.rs` + the `laya-decision-core-census` bin. Landing it is gated on this box's `katgpt-rs` working tree reaching `relevance_core`; see the issue. Cargo validates `katgpt-core/relevance_core` at resolution, so committing it earlier would break every riir-infer build on a stale checkout. The run used an isolated worktree pair: riir-infer @ `15954292` + katgpt-rs @ `19fa122b7`.

## Addendum — T-ml: multilingual on a balanced suite (2026-11-20)

The pre-registration is in `.issues/1237` (`2c9d6cc0`), committed before this number. The suite is the tetris v4 fixture rendered into es / th / ja from its closed template vocabulary (`ml_gen_suite.py`, hand-written phrase tables, no machine translation; deterministic from the fixture, suite blake3 `3d3064ce…8198`). Records are interleaved round-robin across the languages.

The question was chosen by a disjoint pilot (records 200–239): Q1 "is the stack clean" scored true-rate 0.323 and Q2 "holes under it" scored 0.671. Q1 was first to pass the [0.2, 0.8] gate. Census on the first 200 records, multilingual (d = 768): eval true-rate 0.356, Cov rows/d 5.99, fidelity max |Δlogit| 1.0e-6. Run locally on amo while it was quiet (~30 s of forwards); no latency is claimed.

| r | keep core / PCA / random | remove-flip core / PCA / random | SUFFICIENT | NECESSARY |
|---:|---|---|---|---|
| 1 | **0.795 / 0.374 / 0.652** | 0.262 / 0.212 / 0.001 | ✗ (< 0.95) | ✓ |
| 2 | 0.999 / 0.999 / 0.672 | 0.449 / 0.448 / 0.001 | ✗ (ties PCA) | ✓ |
| 4–64 | 1.000 / 1.000 / 0.790–0.999 | 0.48–0.69 / 0.54–0.69 / ≤0.001 | ✗ (ties PCA) | ✓ |

Energy share: r = 1 0.920, r = 2 0.9999996. Participation ratio 1.17.

**Verdict:** multilingual is now DECIDABLE, and LOW_DIM_CORE is NOT CLAIMED by the rule. Its relevance is rank-2. The rank-1 core already beats both controls by a wide margin (0.795 against 0.374 / 0.652) but is not sufficient. The rank-2 core is sufficient (0.999) but coincides with the top-2 PCA span, the same tie english showed at r = 1. It is necessary at every rank.

Across the three checkpoints the R1 picture is one shape: a 1–2-dim relevance core that is causally necessary. In english and multilingual it lies inside the dominant variance directions, so PCA finds it too. In typed it does not, and only there does ACE beat PCA.
