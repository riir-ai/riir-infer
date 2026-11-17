# Bench 1232 — i1221 T2: the last-loop serving posture (elastic k + last-loop KV) on the real pin

**Status:** DONE 2026-11-17T02:29Z (zed-b2e60a87@amo, amo) — T2's structural half GREEN; the k-scaling direction cell parks to T3 (labeled probe suites — see `.issues/1221`).

## What landed (fix `3523a01e` lineage, this bench's own commit)

`DoryCachePolicy { PerLoop (default/T1 golden), LastLoop }` on `DoryKvCache` (`src/dory/forward.rs`):

- **PerLoop** — the training semantics: k slots per recurrent attention block, loop t keeps its own states (T1's proven golden; default constructor byte-identical).
- **LastLoop** — the `CacheStrategy::Last` serving shape (katgpt-rs `crates/katgpt-types/src/looping.rs`): ONE slot per recurrent block; the in-flight row is rewritten by every pass (`BlockCache::write_row` — append-when-new, replace-when-existing, no-hole assert); a decode step's query reads the prefix's final-loop states plus its own current-loop row (paper §6.1, arXiv:2611.03998).
- **Elastic k** stays the per-request posture (cache-at-k; the weights never carry it) — pinned by `elastic_k_caches_coexist_on_shared_weights`.

**The designed invariant (now a law, lib test + real pin):** a from-scratch prefill is BIT-IDENTICAL under both policies at every k — within a pass every position is in-flight and attends its own pass's states; the retention policy changes only what the cache KEEPS. The policy's approximation lives entirely in the incremental decode (prefix reads are final-loop instead of per-pass states).

## Lib tests (13 dory tests green, `--features dory`)

- `last_loop_prefill_bit_identical_to_per_loop` (k=1 + k=4) — the invariant.
- `last_loop_k1_equals_per_loop_everywhere` — the k=1 degenerate (pass 1 IS the final pass).
- `last_loop_footprint_is_k_independent` — byte-count: k=1 ≡ k=4 ≡ per-loop k=1.
- `last_loop_incremental_diverges_from_per_loop_at_k2` — engagement proof (the approximation is REAL on synth weights; never asserted zero).
- `last_loop_incremental_deterministic` — the determinism law at the serving posture.
- `elastic_k_caches_coexist_on_shared_weights` — per-request k, no shared state.
- Pre-existing per-loop battery unchanged (18 dory lib tests total, 0.00s).

## Real-pin cell — `tests/i1221_dory_forward_smoke.rs::last_loop_serving_delta_report_k4` (`-- --ignored --nocapture last_loop`)

Pin: `../riir-train/data/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf` (T0 pin, census num_loops=4). Fixed pseudo-token prompt (8 ids, no tokenizer — the structural posture; the smoke's id-agnostic law). Release build.

| cell | result |
|---|---|
| prefill invariant k=4 | **BIT-EXACT across policies** over 8 positions × 131072 vocab (70.4s for the two prefills) |
| footprint `kv_elements` k=4 | per-loop 1,572,864 vs last-loop 589,824 — **2.67× total; recurrent-only = 4× = k** (attn blocks in/rec/out = 8/20/8, derived from the loaded weights, never hardcoded) |
| decode divergence k=4 | REAL: max\|Δlogit\| grows monotonically 4.63 → 6.99 over 16 greedy steps (prefix reads are final-loop states) |
| greedy flips | **0 of 16 steps — argmax agreement 16/16**; the chains are token-identical over the horizon |

Decode table (step \| max\|Δlogit\| \| argmax agree): 0\|4.625820\|true · 1\|5.000336\|true · 2\|5.304909\|true · 3\|5.562302\|true · 4\|5.776123\|true · 5\|5.953552\|true · 6\|6.099991\|true · 7\|6.222366\|true · 8\|6.326881\|true · 9\|6.416695\|true · 10\|6.517929\|true · 11\|6.629166\|true · 12\|6.730347\|true · 13\|6.822525\|true · 14\|6.908401\|true · 15\|6.987305\|true. Chains never diverged in 16 steps.

**Reading (honest):** the last-loop approximation on THIS checkpoint, at THIS prompt, is a bounded logit drift that never crosses a greedy decision boundary in 16 steps — the 4× recurrent-KV memory win is free over this horizon. NOT a quality claim: one pseudo-token prompt, no labels, no suites (the paper's Table-6 per-suite accuracy delta + output-length ratio need T3's probe payload). The monotone drift growth (≈ +0.1/step, not saturating) is the number to watch on longer horizons — T3's resumable cells own it. Greedy chains degenerating into a repetition loop is expected without a tokenizer/prompt and does not affect the structural verdict.

**Box state (disclosed):** amo, 2026-11-17 ~02:20Z, AC power; the i1226 walk v2 was CONCURRENTLY running on the box (~18% of one core, load ≈ 2.8/30) — a STRUCTURAL cell (bit-identity, element counts, divergence table), no latency claim; the elapsed times are wall-clock cost disclosures only. RSS ~17 GB (the f32 dequant snapshot posture).

## Gate posture

- `cargo test --features dory --lib dory::` — 18/18 green.
- `cargo test --features dory --test i1221_dory_loader_synth` — 5/5 green.
- `cargo clippy --features dory --all-targets` / `--all-targets` / `--no-default-features --all-targets` — clean.
- The smoke target keeps its required-features row; both heavy cells `#[ignore]`d (0 passed, 2 ignored at the gate posture — the P43 law holds).
