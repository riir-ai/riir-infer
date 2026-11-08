# Issue 1185 — sink_audit K/V capture extension + per-layer ε-width census (Cactus "How Wide Is Attention", katgpt-rs issue 987 lead A2)

**Status:** OPEN — intake issue, filed from katgpt-rs `.issues/987_cactus_neurips2026_follow_ups.md` (Research 657 §A2, 2026-11-08). Issue-first per the ungoat discipline: no GOAT plan opens until this issue names its fixture. Execution routes 4090/bmo (offline census over Bonsai-2-27B — a heavy-model lane, not a workstation cell).

## Named plan-time check — DONE at filing (2026-11-08, amo)

The seam does NOT expose K/V today. `HybridForwardScratch` (`src/deltanet/forward.rs`) carries exactly one census capture:

```rust
#[cfg(feature = "sink_audit")]
pub sink_audit_q: Option<crate::deltanet::sink_audit::SinkAuditQCapture>,
```

— a Q-only capture (`SinkAuditQCapture`), harvested in the same seam the dense and ternary bodies share (`run_census` / `run_census_ternary`, plus `run_census_dense` for the dense lane; consumers: the `sink_audit` bin and `needle_eviction_gate --sink-audit N`). The extension is real work: new cfg-gated K/V capture fields beside `sink_audit_q` + harvest in `forward_attention_layer{,_ternary}` (and the dense transformer twin), never a rewrite of the serving path.

## What the lead asks (from 987)

Extend the capture from Q-only to **K/V**, then run the weighted block-pivoted-QR ε-width census OFFLINE on **Bonsai-2-27B** (`Ternary-Bonsai-2-27B-PQ2_0.gguf`), emitting per-KV-head ε-window / sink budgets that calibrate the SHIPPED `kv_sink_window` (katgpt-rs 841 §B-3). Vendor numbers are self-reported — every gate needs OUR fixtures (the x86_64-matrix lesson: a benchmark IS its fixture).

**NOT licensed by the paper:** runtime head-skipping (the authors' method is an offline measurement, not a runtime algorithm). The output is a census tool + budget numbers, nothing more.

- [ ] T1 — K/V capture extension: cfg-gated `SinkAuditKVCapture` beside `sink_audit_q` in `HybridForwardScratch` + harvest in both attention bodies (ternary + dense twins); feature-off compiles the capture to nothing (the Plan-287 G3 posture); `cargo test --features sink_audit` census cells stay green (947/948 records unchanged — the Q census is not re-litigated).
- [ ] T2 — the census: weighted block-pivoted-QR ε-width over the captured K/V, per layer × KV head, on Bonsai-2-27B (4090/bmo window, GPU-exclusivity per the correctness-gate law; record box state beside every number).
- [ ] T3 — the GATE (two mandatory arms, both pre-registered):
  - **Needle-retention cell** — the paper's own width budgets FAIL needle tests 10/10 (a loss-only gate is forbidden): every candidate `kv_sink_window` budget must hold needle retention at the incumbent's level on the needle_eviction_gate lane, or the budget is refused. Per-family conditional retention per the lossy-surface promotion rule.
  - **Hop-spanning percentile sanity arm** — a `n·0.99`-style index lands on the MAX for n ≤ 100, not p99 (the katgpt-rs `percentile_index_audit` class): the census's percentile math ships with tail-support reporting, and the fixture pins the small-n hop behavior before any budget number is quoted.
- [ ] T4 — budget readout: per-KV-head ε-window/sink budgets vs the shipped `kv_sink_window` pin (841 §B-3); if the census disagrees with the shipped pin, that is a VERDICT (owner-gated re-pin), never a silent edit.

## Constraints

- Offline-only (`sink_audit` feature posture — zero serving-path cost, the 947/948 precedent); modelless; no new deps; BLAKE3-pinned G1 bit-identity on the existing Q-census cells when the capture fields are added.
- Ternary body AND dense body both get the capture (the 981 lesson: dense AND ternary share the seam).
- Heavy-model runs are 4090/bmo lanes: file the run as a rig job, never a workstation cell; `SSH inside sshd hangs` — schtasks route for the 4090.

## Refs

- katgpt-rs `.issues/987_cactus_neurips2026_follow_ups.md` lead A2 · Research 657 §A2 · Cactus "How Wide Is Attention? Measuring Its Effective Width in Tokens" (NeurIPS 2026, Mosoyan et al. — title/venue verified on cactuscompute.com/research 2026-11-08; full text not independently re-read at filing — the paper-claim numbers above ride Research 657's distillation)
- Producer precedent: this repo `.benchmarks/947_issue981_sink_audit_census.md` + `948_bonsai2_27b_sink_audit_census.md` (the Q-capture census, UNDECIDED verdict that motivates measuring K/V); katgpt-rs 841 §B-3 `kv_sink_window` (the consumer being calibrated)
- katgpt-rs `scripts/percentile_index_audit.py` (the percentile-hop class the sanity arm pins)
