# Issue 1242 — QDA-derived eval instruments: KL-vs-anchor ranked selection, gencheck (truncation/repetition census), cached-anchor scoring, proxy break-even

**Status:** open — filed from the humanfia/QDA distill (riir-refine Research 427); unclaimed.
**Claimed:** zed-989683dc@4090, 1795264982, expires 2026-11-21T14:43:00+00:00
(renewal 2 - T2 gencheck LANDED 759bc14; the build continues: T1 anchor-KL (the golden_nll cached-anchor shape) + its first consumer run, plan 508's pre-GPU KL gate.)
(renewal 1 - lease lapsed overnight; the build starts this window: T2 gencheck first (pure-text fixtures), then T1 anchor-KL (the golden_nll cached-anchor shape); FIRST CONSUMER: plan 508's pre-GPU KL minimum-effect gate (the teacher-vs-student measurement is the same code).)
(scope: T1 anchor-KL census + T2 gencheck + T3 break-even + T4 composition note — the instrument BUILD on this box (opt-in feature `anchor_kl`, offline, zero serving-path cost; fixtures + canaries in-repo). The instrument's FIRST consumer is plan 508 T2's gate (the SAME truncation/repetition census + anchor-KL on the KD cell), so the definitions land 508-compatible. Heavy RUNS route per the m3 box law.)
**Routing:** riir-infer quant-eval lane, beside `golden_nll` / `si_zero_shot_ppl`; heavy runs on amo/bmo per the m3 box law. Claim with a lease line before implementing (katgpt-rs Issue 943 discipline).

## Why (the evidence)

- QDA (arXiv:2611.23388) scores candidate quantized checkpoints by KL-vs-anchor on held-out traces in minutes (350–990× faster than full eval) and picks the more accurate checkpoint **82%** of the time over 44 pairs; layer-wise error gets 73% and ranks AWQ last of three where AWQ is second — per-layer metrics are blind to cross-layer error accumulation.
- Its failure catch (`gencheck`) watches truncation counts from repetition loops; its biggest search win came from a candidate the proxy liked but generations caught looping.
- Our frozen-read census (golden Δ-NLL + per-family argmax flips) IS the audit form of this instrument. The deltas: ranking purpose, a generation-behavior axis, anchor caching. Grep 2026-11-21: **zero truncation/repetition census in the riir-infer/katgpt-rs trees** (the `truncat|repetition loop|degenerate` sweep returned only unrelated belief-gate and vocab-truncation hits).

## Deliverables (each opt-in, offline, zero serving-path cost)

- [x] T1 **KL-vs-anchor ranked-selection census** — LANDED 2026-11-21T13:0xZ (zed-989683dc@4090 lease, riir-infer `0d575fc`): `src/anchor_kl.rs` (feature `anchor_kl`, OPT-IN, offline report-only, zero deps) — per-token forward-KL from logits slices (log-domain LSE, f64 sequential ascending folds per the Issue-940 law), the top-k-union truncated variant (TOP_K 64 pinned; restricted LSEs; union < vocab by construction at vocab > 2k), the trace-set mean fold; 7 fixtures incl. the Bernoulli oracle (f32 tolerance 1e-6) + the disjoint-peaks union test + the refuse paths; clippy armed exit 0, rustfmt(2024) clean, default posture green-zero honored. The content-addressed anchor CACHE (per model bytes + trace set + dtype) rides the census bin wiring (the golden_nll scratch pattern) — the math half is the landed instrument. Original row:
- [x] T2 **gencheck instrument** — LANDED 2026-11-21T10:1xZ (zed-989683dc@4090 lease, riir-infer `759bc14`): `src/gencheck.rs` (ungated, pure text, zero deps) — truncation-at-limit census + maximal-suffix-cycle detection (smallest-period tie rule) + repeat count + distinct-8-gram density; thresholds PINNED in-source (LOOP_RATIO_MIN 3 / DENSITY_LOOP_MAX 0.35 / DENSITY_NGRAM 8 / MIN_TOKENS 64 abstain floor) with the pinned-constants fixture gate; 8 fixtures incl. axes-independence (density collapse flags WITHOUT an explicit tail cycle); the fixture run caught its own first draft (mod-numbered prose invented a period-37 cycle — the mechanism verified itself); clippy + rustfmt(2024) clean, 8/8. Original row:; heavy-quant looping positive control + healthy zero-flag control (the Bench-007 positive-control shape); report-only at serve until a wired consumer. Slot beside `RIIR_QUANTIZE_CENSUS` in the requant/EXL3 gate harnesses; natural fit in the sink-audit/needle long-context lanes.
- [ ] T3 **proxy break-even estimator** (folded from the no-GD advocate; generic math): `ship proxy ⟺ V·c_p + F·c_g + (1−f̄)·L < V·c_g` with f̄ the measured proxy fidelity → typed verdict SHIP-PROXY / SHIP-DIRECT / UNDECIDABLE; golden unit tests. Candidate home katgpt-core/`estimators` (keyless opt-in) only if it finds a second consumer; else stays a formula in this lane's docs.
- [ ] T4 **composition-algebra note** — checkpoint→checkpoint transforms: memoizable content-addressed prefixes, typed non-commuting ordering, idempotence collapse. Design note in this lane's docs; implementation waits for the second consumer (two-instance bar).

## Proof paths (pre-registered)

Planted fixtures per instrument (canaries prove the floors fire); fidelity bar pre-registered; every latency/cost number with PROVENANCE + GPU exclusivity; the census lanes are offline so no alloc-free serving claim applies — the T1 cache's loud stale refusal IS its gate (fixture).

## Refs

riir-refine Research 427 (distill + Path 0 table) and Plan 393; qda @ `dc39a02b` (Apache-2.0); prior art: Fireworks 2024 + arXiv:2505.15353 (KL-as-quant-quality is a published class — the fidelity census + failure catch are our deltas).
