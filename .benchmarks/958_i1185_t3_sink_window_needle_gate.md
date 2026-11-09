# Bench 958 — i1185 T3: the sink-window budget needle-retention gate (the Cactus width budgets vs the shipped `kv_sink_window`, qwen35-hybrid lane)

**Status:** DONE — the 16K gate cell refused every width-derived candidate (8/8 comparison rows); the incumbent stands; the re-pin verdict routes to the owner (T4).

## What T3 pre-registered (issue 1185) and what this cell is

The width census (Bench 957) measured per-KV-head ε-widths on Bonsai-2-27B (medians 376/256/168 at ε 0.1/0.2/0.3) and explicitly refused to license a `kv_sink_window` re-pin from the table alone: *"changing it needs T3's needle-retention gate + the lossy-surface per-family rule"*. Research 657's distillation of the Cactus paper records the shape of that failure: the paper's own width budgets fail needle tests 10/10 — a loss-only gate is forbidden.

So T3 = apply the width-derived budgets as what a `kv_sink_window` budget actually is — a **ceiling** (`live_slots <= n_sink + window`, the katgpt-core module invariant) — on the model-bound needle lane, and compare per-family retention against the serving incumbent's level. A candidate that drops below the incumbent at any tested budget is **REFUSED**; the re-pin stays owner-gated even on an all-HOLD table.

## Machinery landed (this bench's code)

- **The window binds now** (`src/deltanet/kv_evict.rs`): selection runs through `kv_sink_window::select_evict_windowed` — stale rows first regardless of score, sinks never. An unbounded window reduces index-for-index to the old Sink-only pin path (fixture `unbounded_window_reduces_to_the_pin_path_selection`), so every incumbent consumer is byte-identical (the two armed T3 bit-identity tests still pass); previously a bounded-window policy's window was silently inert in this lane.
- **Candidate arms** (`needle_eviction_gate --sink-window W1,W2,…`): one `sinkwin{W}` arm per candidate, budget = `n_sink + W` (the policy's own ceiling), max-recent policy, sink pin `(4, W)`. The stale-first selector drives the cache down to the ceiling — the paper's application of a width budget as a cache window.
- **The verdict gate** (run-end section): each candidate vs EVERY `maxrecent@{frac}` arm (the serving posture: sinks pinned, unbounded window, budget fracs) on **needle-row survival** (the family readout; the lossy-surface per-family rule) AND **NLL retrieval** (`EPS_NATS = 1.0`). Any missing arm = NO VERDICT, never folded.
- **Percentile sanity arm** (issue 1185 T3 arm 2, `src/deltanet/width_census.rs`): `percentile_read` + `width_summary` — nearest-rank `ceil(p·n) − 1` with `tail_support = n − index` attached to every read; the `sink_audit --width-eps` stdout line and `width_census_report.json` gained the disclosed `summary` block (`width_summary_v1`). Fixtures pin the hop: p99 on 64 heads lands ON the max with tail_support 1 (the Bonsai census's own n), p99 at n=100 lands on index 98 (not the naive `(0.99·100) as usize = 99` truncation), p50 is the lower median (index 31 of 64, tail 33 — the disclosed replacement for the old bare `widths[n/2]` upper-median).
- Fixtures: 5 in `width_census.rs`, 2 in `kv_evict.rs`; clippy `-D warnings` clean at `sink_audit` (lib) and both bins; rustfmt clean.

## The smoke cell (machinery validation, 2026-11-09 ~06:2x +07, amo)

`--context 2048 --budget-frac 0.5 --sink-window 256,8192 --arms maxrecent,sinkwin` (full reference fresh; qwen35-hybrid 0.8B Q8_0, `chat_probe`):

| arm | needle surv | retrieval | verdict vs maxrecent@50pct (surv 0.182, retr 0.750) |
|---|---|---|---|
| full | 1.000 | 8/8 | reference |
| sinkwin256 (ceiling 260) | **0.000** | 4/8 | **REFUSED** |
| sinkwin8192 (ceiling > live) | 1.000 | 8/8 | **HOLD** (no eviction fires; byte-identical to full) |

The gate discriminates: the ceiling form of a narrow window kills every needle row (surv 0.000 — the paper's 10/10 shape, reproduced on our stack), while a ceiling wider than the context never evicts and holds trivially. Not a constant refuser.

## The 16K cell (the recorded gate run)

- **Host:** amo (M5 Ultra 96GB, macOS 27.2, AC plugged); box state header in `run.log`. Wall 6371.8s (~1 h 46 m); prefill ~555 s/arm (~30 tok/s, ~3.4× the 4090 box's CPU rate — cross-box wall times are not comparable, verdicts are deterministic).
- **Model:** `Qwen3.5-0.8B-Base-Q8_0.gguf` (774 MB, pulled from the 4090 box — the Bench 947 lane's checkpoint); **corpus:** `chat_probe` (21 pages, the 947 lane's corpus, pulled from the same box).
- **Rig:** context 16,384 · needles 8 (depths ≈ 1.7K…12.5K, all planted) · budget fracs 0.25/0.5 (twin budgets 4,096/8,192) · candidates W ∈ {168, 256, 376, 8192} = the Bench 957 ε-grid medians (0.3/0.2/0.1) + a context-half window to prove the gate can say yes · cadence 512 · defer 1 · sinks 4 · sinks-row survival 1.000 on every arm (the pin held).
- Arms executed: full + maxrecent@{25,50}pct + sinkwin{168,256,376,8192} (diff/usage/random built but filtered — the gate compares one policy family).
- **Run disclosure:** the executed binary carried the candidate arms INSIDE the per-frac loop, so each `sinkwin{W}` ran TWICE — a construction bug fixed in the same change (the loop is now frac-independent + a duplicate-arm-name guard bails loud). The duplicates are a free determinism cross-check: every pair's NLL vector is **bit-identical** (all four candidates × 8 deltas, compared from the log); the report keeps one row per name.

## Results (the gate verdict)

| arm | budget | needle-row surv | retrieval | genNLL | verdict vs twins |
|---|---|---|---|---|---|
| full | MAX | 1.000 | 8/8 | 3.199 | reference |
| maxrecent@25pct | 4,096 | **0.652** | 7/8 | 3.176 | incumbent twin |
| maxrecent@50pct | 8,192 | **0.815** | 6/8 | 3.179 | incumbent twin |
| sinkwin168 (ceiling 172) | 172 | **0.000** | 4/8 | 3.442 | **REFUSED ×2** |
| sinkwin256 (ceiling 260) | 260 | **0.000** | 4/8 | 3.438 | **REFUSED ×2** |
| sinkwin376 (ceiling 380) | 380 | **0.000** | 4/8 | 3.440 | **REFUSED ×2** |
| sinkwin8192 (ceiling 8196) | 8,196 | 0.375 | 6/8 | 3.330 | **REFUSED ×2** |

Gate output: `8 comparison row(s) REFUSED — the width-derived kv_sink_window budget family may NOT re-pin (the incumbent stands)`.

### The readings

1. **The ε-grid medians die completely as cache windows.** At ceiling 172–380 on a 16K prompt, the first eviction event evicts every stale row — every needle row (depths ≥ 1.7K) — needle survival 0.000, and retrieval collapses to 4/8, exactly the residual the code-progression leak supports (codes 1000+977·i; the model reconstructs some from the visible pattern — the pre-958 lane property; needle-ROW survival is the honest family readout and it is zero).
2. **The paper's 10/10 needle failure, reproduced on our stack with our budgets**: the width census's own medians (Bench 957), applied as what a `kv_sink_window` budget is (a ceiling), lose 100% of needle rows. A loss-only gate would have passed them on aggregate NLL — genNLL 3.44 vs full's 3.20 is a barely-visible drift; the family readout is where the loss lives.
3. **Even a context-half ceiling loses**: W=8192 keeps the deep needles bit-exactly (needle 7's Δ = 9.5e-6 nats — the ceiling preserves what it covers) but drops 5 of 8 rows and fails both twin comparisons. A width budget large enough to hold this rig's needles would have to be ~context-sized — at which point it is not a budget.
4. **The incumbent twins hold**: unbounded-window sink pins at frac budgets keep 0.652/0.815 of needle rows and 7/8, 6/8 retrieval. The serving posture (sinks pinned, no window ceiling) is the posture the needle lane validates; the window ceiling posture is where budgets go to kill needles.
5. **Sinks are inviolate everywhere** (survival 1.000 on every arm, including all ceiling arms) — the pin half of the shipped pin works; it is the WINDOW half that is a loss surface at long context.

## T4 verdict this evidences

**No re-pin.** The census (957) said the width axis and the sink axis are different quantities; this cell says applying the width budgets as `kv_sink_window` ceilings destroys needle retention at every tested size, including a context-half window, while the incumbent-window form (the unbounded serving posture the pin actually ships as) holds. The re-pin decision is OWNER-GATED and the measured recommendation is: keep the pin; the width budgets are a census TABLE, not a policy.

## What this does NOT license (unchanged from 957)

- No `kv_sink_window` re-pin from an agent: the gate's verdict routes to the OWNER (T4), and a REFUSED row vetoes the budget family outright.
- No runtime head-skipping / windowed routing (offline measurement only, per the source).
- No cross-family comparison against Cactus's own numbers (different lossy surfaces).

## Observed pre-existing wart (not this bench's defect)

`upsert_report`'s incremental merge cannot re-read its own file mid-run: `delta_nll` is written NaN→null before the delta-fill loop and `Vec<f32>` has no null-tolerant deserializer, so each intermediate merge logs "rewriting WITHOUT the old rows" and the file holds only the newest arm until the final (post-fill) write. The final report is complete; a CRASH mid-run loses all completed arms — the opposite of the function doc's claim. Filed as issue 1187 (one-line serde fix, out of T3's scope).
