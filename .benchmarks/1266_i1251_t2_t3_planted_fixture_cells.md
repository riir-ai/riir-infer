# Bench 1266 — Issue 1251 T2/T3: the planted-fixture cells (trigger-vote screen)

**Verdict: T2b PASS — both pre-registered bars cleared on real bytes. The T4
real-organism arm is ARMED.** T2a SanityPass (recovery 3/3, but bar-2
structurally unavailable in the circular regime — see the contrast note).

- Issue: riir-infer 1251 (T2 fixture battery + T3 recovery cell; T1 record:
  riir-infer `4f129d43` + `075b8ca6`, the vote + fixture gates + bin).
- Lane: `readout_screen` (opt-in) — `src/trigger_vote_cell.rs` (the planter +
  the in-place Q8_0 GGUF patcher + the PLNT1 manifest),
  `src/trigger_vote_cell_score.rs` (the baselines + the pinned scorer),
  `src/trigger_vote_cell_fixture.rs` (7 hermetic gates on tiny GGUFs the
  fixture writes itself), the `trigger-vote` bin (`--plant`, `--cell`).
- Date: 2026-11-26 · box amo (M5 Ultra, AC) · release build (`cargo build
  --release --features readout_screen --bin trigger-vote),
  CARGO_TARGET_DIR=/tmp/i1251_tgt.
- Base checkpoint: `../riir-train/data/Qwen3.5-0.8B-Base-Q8_0.gguf`
  (blake3 `70fd9c83d63454ae72e07a29ebdb78f863cfad102d8939f6ec49b2aea32104e2`;
  24 layers, n_ff 3584, n_embd 1024, vocab 248320).

## The two planters (T2)

- **T2b (`--plant upstream`, non-circular, the deciding cell)** — core
  directions = base down-proj columns (donor neurons 0/5/8 of layer 12,
  near-orthogonal, fan-survival-screened); trigger tokens EMERGE from
  geometry (`t = argmax over the normal-token domain of ĉ·e_s` → tokens
  **29, 1282, 29252**); gate/up rows += γ·normalize(ĉ + 0.15·r̂) over 24-fan
  neurons per trigger (fan0 = the payload neuron: its down column += β·ê_u);
  512 noise-field rows at γ_noise = γ/10 (the fine-tune floor).
  γ = 0.1605, β = 0.1274 (0.5× the layer's median row/column norms).
  Planted file blake3 `4d885d9ea6f34006742606c6fbc530a8e8a6f2f011f9ae6ef8ef2ebd4cec95bc`,
  byte-identical across two plantings (plan = pure function of base+config).
- **T2a (`--plant circular`, disclosed circular, sanity only)** — cores =
  ê_t of the lowest-id domain tokens that pass the quorum + margin screens →
  tokens **61, 90, 107**; same fan/noise shape. The screens exist because the
  naive lowest-id picks (0/1/2) are mean-conditioned special-token rows whose
  fans cross-vote the massive-activation crowd (see the finding below).

## The T3 cell results (the pinned bar: ≥⅔ of ≥3 planted in the vote's
top-10 AND each at ≥2× the better-ranked baseline)

| cell | planted | vote ranks (votes) | summed-change ranks | emb-shift | bar1 | bar2 | verdict |
|---|---|---|---|---|---|---|---|
| **T2b upstream** | 29, 1282, 29252 | **2, 3, 7** (24 each) | 54, 351, 916 | no signal (0 ranked) | **3/3 PASS** | **PASS** (11×–131×) | **T3 GO — T4 ARMED** |
| T2a circular | 61, 90, 107 | 8, 10, 4 (24/24/25) | 1, 2, 3 | no signal | 3/3 | fail (structural) | SanityPass |

- The vote's pooled band = layer 12 alone (single-layer band, legal).
- Baseline scope = the band's active rows (584 condition neurons), the SAME
  dots the vote reads, summed over all neurons — the direct ablation of the
  vote's one mechanism (per-neuron top-15 sparsification + counting vs
  summing). Emb-shift reads ZERO (weights-only edits) — honestly reported as
  no-signal/unranked; the binding baseline is summed-change.
- **The contrast IS the finding:** in the circular regime the delta writes
  the read statistic directly, so summed-change tracks it perfectly (ranks
  1/2/3) and the 2× bar is structurally unavailable — exactly why the issue
  forbids T2a from being the deciding cell. In the non-circular regime the
  noise field drowns the sums (Σ|Δ·e_s| ≈ Σ γ_noise·|e_s|·E|cos| — a
  token-norm mass the planted signal cannot outrank), while the vote's
  per-neuron top-15 sparsification keeps the trigger votes concentrated —
  the paper's mechanism, reproduced on our lanes.

## Wall times (release, quiet box, operational only)

plant T2b 26.6 s (812 MB copy + requant of 584 gate/up rows + 3 down
columns) · plant T2a 64 s (the bench scan) · cell run 14.4 s (detector +
both baselines) · self-test 17 gates PASS (10 T1 + 7 new).

## The real-geometry findings recorded for T4 (the reader's ablation priority)

1. **Massive-activation rows ride every top-15** — tokens 11/12/13/14/220
   (the outlier-norm family; 220 is the table's max-norm row) collected
   72/72 fan memberships in the naive T2a sim. The pinned raw-e_t proxy is
   outlier-row-biased: the vote's crowd floor is set by rows the planter
   cannot even select as triggers (the domain excludes them by norm, but
   they still COLLECT votes). The rmsnorm-γ / outlier-gated reader variant
   (pinned T4+ ablation) should be evaluated BEFORE the real-organism run —
   on a trained backdoor the trigger competes against this same crowd.
2. **Donor qualification is load-bearing**: the naive donor scan (ortho
   only) picked donors whose domain-argmax margin was thinner than the fan
   noise (ρ=0.15) — the planted token fell OUT of its own fans' top-15s
   (fan_rank 0). The fan-survival quorum screen (8-fan samples, ≥70% at
   rank ≤ 5) fixed acceptance; the assembled sim stays the authority.
3. **The noise field's vote crowd is real at 512 neurons** (runner 26 votes
   vs planted 24) — the vote wins on RANK concentration, not count margin.
   A larger noise field would erode bar-1; a smaller one weakens the
   summed-change drowning that carries bar-2. 512/24 at γ/10 is the pinned
   operating point.

## Gates + determinism

`trigger-vote --self-test` 17/17 (the 7 new cell gates: t2b_upstream_recovery,
t2a_circular_sanity, plant_determinism, row_census_exact, requant_roundtrip,
manifest_tamper_refuses, planter_refuses_degenerate — the whole chain runs
on tiny GGUFs the fixture writes itself, per-process unique scratch dirs).
`cargo test --features readout_screen --lib` 328/0 (305 default posture);
clippy `-D warnings` clean at default / --no-default-features /
readout_screen; rustfmt clean (per-file, this lane's files).
PLNT1 manifest = canonical binary + trailing BLAKE3; tamper refuses loud.
Two plantings byte-identical (blake3 pinned above). Cleanup: the planted
fixtures are regenerable from the base + pinned constants (seed 0x1251,
layer 12) — the /tmp copies are deleted at lane close, only digests kept.
