# Plan 630 — FlashStates-class hybrid state cell: the T1 pre-registration (Issue 1201)

> **Status:** IN PROGRESS — T1 ✅; T2 ✅ CLOSED 2026-11-12 (code `8e64a036`+4 fix commits incl. the Metal port + harness bin; amo mechanics runs a0 ✓ a0p ✓ a1 ✓ seeds 0/1/2; **the G1 deep-compares LANDED in `4_2048_s2`: a1 `g1_checked=4`, run COMPLETE — `compare_walks` passes are the only path that increments the counter and drift aborts the run, so 4/4 resumed-vs-full state-identity comparisons held; runs on disk at `.benchmarks/flashstates_cell/`, untracked by design, digests in the manifests). T3–T5 open (T3 claimed: amo-idle-i1201t3@amo). The bar-bearing ladder (N ∈ {8,16,32}, 32K cap, bars 1–7) is the T3+ cell work — the 4-session mechanics runs are never bar rows.

## What this plan is

The FROZEN cell design for Issue 1201 (FlashStates-class multi-session hybrid state serving,
`../.research/023_FlashStates_Hybrid_State_Prefix_Cache.md`). Everything the measured cell runs —
trace shape, seeds, arms, metrics, bars, falsification gates, compute posture — is pinned HERE
before any arm runs. The bars are pre-registered: a run that misses them DEMOTES the arm, never
retunes the bar. Substrate seams verified against source at pre-registration time (2026-11-11):
`Qwen38PrefixCache` (`crates/riir-infer-gpu/src/qwen38_prefix_cache.rs`: `new/insert/restore/
note_write/len/pruned`), `katgpt_kv::radix_prefix::RadixPrefixTree` (`match_prefix/insert/
path_pages_into/evict_to_budget`, katgpt-kv feature `radix_prefix_cache`),
`PagedKVCache::{chunk_page_tables, retain_chunk_pages, release_chunk_pages, adopt_chunk_pages}`
(katgpt-rs `crates/katgpt-transformer/src/kv_cache.rs:738-776`).

## The trace (deterministic, token-exact — the one shared input)

Synthetic multi-session GROWING-CONTEXT replay, generator `tracegen` (this plan's T1 deliverable,
pure CPU, seeded, no device):

- **Session shape:** N sessions run "concurrently" (interleaved round-robin). Every session
  opens with the SAME seeded **preamble** (2048 tokens — the agentic system/tool preamble;
  identical across sessions, distinct per seed). Session i's turn t prompt = preamble + the
  session's full prior transcript (every earlier turn's suffix AND synthetic output) + a fresh
  turn suffix — the agentic growing-context shape (each turn's prompt contains all prior
  turns; the cacheable prefix grows monotonically per session, and the shared preamble gives
  the radix arm its cross-session structural role).
- **Token-exactness:** the generator emits the TOKEN stream directly (`Vec<Vec<u32>>` per turn:
  a prompt row + a continuation row). Arms consume the SAME rows — no re-encoding, no
  tokenizer drift between arms. Outputs are SYNTHETIC (seeded continuation tokens), not model
  samples: the cell measures cache/reuse mechanics, not decode quality; the decode-quality
  axis is G1's bit-identity gate, not the trace.
- **Scale ladder:** N ∈ {8, 16, 32}; per-session transcript cap 32K tokens (context_length
  honest-scale — the B200 1M-ctx traces do not transfer); turn suffix 128-512 tokens (seeded).
- **Seeds:** the ladder runs at seeds {0, 1, 2}; the trace file carries a BLAKE3 digest per
  (N, seed) — every arm's run record names the digest it replayed. A changed generator output
  at a pinned seed is a LOUD cell invalidation, never a silent re-baseline.
- **Interleaving:** strict round-robin over sessions, one turn each per round; rounds continue
  until every session hits its cap. Session i's cache key namespace is disjoint (per-session
  prefix identity by construction — no cross-session shared prefix in the base trace; a
  shared-system-prompt arm variant may join later as its own pre-registered cell, never mixed
  into this one).

## Arms

| arm | definition | substrate |
|---|---|---|
| A0 — incumbent (the bar denominator) | the AS-SHIPPED serving posture under concurrent load: a FRESH stream per request — no cross-request reuse of any kind. No serving lane today composes per-session caches; this is what the stack does today and the honest floor. Median uncached = the full prompt (preamble + transcript + suffix) by construction — asserted by the generator's accounting test | the serving forward, no cache |
| A0+ — per-session single-stream control (reported beside, never the bar denominator) | one `Qwen38PrefixCache` instance PER SESSION, caller-chosen anchors at every turn end: within-session reuse works (restore + suffix refill), cross-session sharing and shared-budget retention do not. UNWIRED machinery — composing it per-session is itself lane work, which is why it is the control, not the floor (the Report-the-Floor law: the floor is the naive SHIPPED posture) | `qwen38_prefix_cache.rs` (CUDA today; Metal = the T2 port work item) |
| A1 — +multi-session radix | one SHARED tree across sessions; end-of-request automatic frontier capture (742 T4's caller-chosen anchors stay as the manual override); resume = `restore` + suffix re-fill | `RadixPrefixTree` + the chunk-page seam, opt-in `radix_prefix_cache` — NEVER a private tree |
| A2 — A1 + GDS retention | cost-density (GreedyDual-Size) priority replacing recency: (estimated prefill seconds saved if evicted) ÷ (capacity actually freed — shared pages a surviving state needs count as NOT freed) + last-use watermark; leaf-last (a session's newest checkpoint evicts only after its inner checkpoints); `note_write` lineage prune stays as the correctness floor. Runs at a PINNED shared state budget (the resource constraint that makes eviction real — A0+/A1 at the same budget are its comparison rows) | A1 + the retention policy in the seam consumer |

## Metrics (defined once, here)

- **median uncached prompt tokens** — per request: prompt tokens − restored prefix length
  (the suffix actually re-prefilled); median over all requests in the run. A0's median is
  ~the full transcript (minus its own-session anchors) by construction — sanity floor.
- **p50/p90 TTFT** — per request: wall time from request start to first decode token emitted,
  measured at the serving seam (restore + suffix prefill + first decode step). Percentile
  tail support printed beside every p90 (the n−1-max rule: support < 10 rows is reported,
  never a green bar).
- **decode tok/s regression** — steady-state decode rate per arm vs A0, same trace, same
  box state.
- **eviction fidelity (A2 only)** — the falsification fixture: two states sharing KV pages +
  forced eviction → the shared page is not double-counted as freed and the LEAF survives the
  inner (T3's fixture, pre-named here).

## Pre-registered bars (frozen 2026-11-11 — a miss DEMOTES, never retunes)

1. **Reuse (primary):** A1/A2 median uncached prompt tokens ≤ 20% of **A0**'s at N ≥ 16
   (≥ 80% reduction).
2. **TTFT:** A1/A2 p90 TTFT ≤ 0.5× A0's at N ≥ 16 (≥ 2× improvement).
3. **Decode regression:** A1/A2 steady-state decode ≥ 98% of A0's (≤ 2% regression).
4. **Attribution row (mandatory disclosure, A0+):** the A0+ control runs beside every A1/A2
   cell. If bar 1 passes but A1/A2 ≈ A0+ on uncached tokens (within 10%), the record discloses
   "the measured gain is the per-session-cache composition, not the multi-session lane" — the
   lane's residual value claim then rests on bar 6 alone.
5. **Retention (A2's own axis):** at the pinned shared budget (budget = 25% of the trace's
   total demand — pinned with the first measured cell, before A2 runs), A2's median uncached
   ≤ A1's at the SAME budget; A1-vs-A0+ at that budget is reported beside. A2 failing bar 6's
   fidelity fixture refuses the arm regardless of the rate numbers.
6. **G1 (correctness, binds every arm):** resumed decode bit-identical to full re-prefill —
   deterministic replay walk, BLAKE3 walk pins at the resume boundary, cache hits under the
   CUDA-graph arm re-proven (the 742 T4 gate shape). Issue-940 determinism law: any synced
   reduction stays ordered.
7. **GOAT verdict:** A1 and A2 are judged SEPARATELY against A0 on bars 1-6. The paper's own
   ablation is the scoping evidence — ASC placement alone with LRU is NEGATIVE at 32
   sessions; both retention halves ship together or neither is honest. Either arm missing
   its bar stays opt-in (never default); promotion rides the owner gate on a PASS.

## Compute posture (the compute law, verbatim duties)

- The measured cells run on GPU: the 4090 CUDA arm (A0 today) and the m3/amo Metal arm (a PORT
  work item — 742 T4 is CUDA-only today; the port is T2's responsibility, parity-gated against
  the CPU short snapshot before any measured cell).
- CPU posture = SHORT parity snapshot only, never the long measured cell.
- Resumable by construction (every arm > 20 min): durable state dir under
  `.benchmarks/<cell-id>/` (NEVER /tmp), per-arm checkpoint flush (the `flashmemory_decode_gate`
  arm-bin pattern: complete-flag + bucket rewrite + temp+rename), a resume manifest (trace
  BLAKE3 + per-arm cursor + box state), idempotent `--resume`.
- GPU exclusivity attested beside every quoted number (the perf-rematch preflight posture;
  box state: free RAM, load, power source/mode).
- Cells run ALONE (whole-box compiler load skews walls); a cell sharing the box with any
  cargo/rig job is PROVISIONAL at best.

## Tasks

- [x] **T1 — the pre-registration + the trace generator (this session):** this plan = the frozen
  cell; `tracegen` (pure CPU, seeded `Vec<Vec<u32>>` rows, per-(N,seed) BLAKE3 digest,
  determinism-pinned by test) lands in `crates/riir-infer-gpu/src/` (ungated module — the
  generator is device-free; the harness bin joins at T2). The incumbent sanity row: A0's
  median uncached ≈ transcript length is asserted by the generator's own accounting test.
- [x] **T2 — the multi-session extension on the named radix path** — **RADIX HALF LANDED 2026-11-11 (`crates/riir-infer-gpu/src/flashstates_multisession.rs` + `tests/flashstates_multisession_g1.rs` 5/5, feature `flashstates_multi` = ["dep:katgpt-kv", "katgpt-kv/radix_prefix_cache"]; both deps workspace-table rows, BOUNDARY-allowlisted):** `FlashStatesManager` — the ledger is the SUBSTRATE'S OWN (one retain per fresh capture = the tree's hold; one per adoption = the session's; session end releases adopted only; eviction releases the tree's), the KV half rides `PagedKVCache` literally (`ensure_pages` into a dedicated per-capture slot as the fresh-page vehicle — `prompt_tokens − 1` arity so a chunk-aligned capture allocates exactly `chunks` pages; `write_kv` capture writes; `retain/release_chunk_pages` holds), the GDN half rides `GdnStatePool` (the page shape the paged cache cannot hold — a pool, never a tree), and the CONCURRENT-FILL conflict (two sessions capturing the shared preamble — the trace's base shape) recovers first-writer-wins through the substrate's own `PathMismatch` refusal: the loser discards its fresh duplicates, adopts the winner's pages, retries once. Gates: clippy `-D` clean (feature posture all-targets + default lib), fmt clean, 5/5 (ledger balance incl. the cross-session dedup — B's capture allocates only its divergent suffix, cold-session byte round-trip, first-writer-wins recovery, eviction dropping the tree hold while an adopting session's survives, continuous-session no-re-adoption — the cursor bug the gates caught, fixed in the same landing). Graph safety by construction: the manager is device-free, the harness copies pages INTO live buffers (data move, never a re-home). **REMAINDER LANDED 2026-11-11 `8e64a036`:** the A0 Metal port (`qwen38_prefix_cache_cubecl.rs` — the 742-T4 contract over the wgpu forward's checkpoint/restore, the full-state divergence disclosed: snapshots carry GDN+conv+KV because the Metal forward is weights-bound; A0+ stays the control, never a bar denominator) + `copy_range_cubecl.rs` (the f32 range-copy kernel — KV capture staging + GDN boundary-arena slots; restore rides the existing KV fill kernel at base_pos=0) + the harness bin (`flashstates_cell`: arms a0/a0p/a1, the canonical partition drive making G1 a state-identity claim, A1 restoring to the deepest chunk boundary P_r, GDN boundary captures launched in-drive via a pre-cloned-handles sink into a fresh-span arena — zero walk syncs, T4's in-chunk export the named successor — resumable rows+cursor, manifest digest check, league posture via the `flashstates_cell` feature, a2 refuses loud) + the state-io pub surface on the forward + `session_held_chunks`. Gates: clippy `-D` clean both postures, fmt clean, 5/5 + 6/6 green. **CLOSED 2026-11-12 (amo idle lane):** the G1-on-model evidence landed — `4_2048_s2` (seed 2) a1 reads `g1_checked=4` with `a1.COMPLETE=ok`: the counter increments only after `compare_walks` Ok (decode argmax identity + first-decode logits BLAKE3 identity, resumed-vs-full), drift aborts the run, so the completed run pins 4/4 PASS. T2 CLOSED; the bar-bearing N ∈ {8,16,32} ladder is T3+'s cell work.
- [ ] **T3 — cost-density retention (the load-bearing half)** + the falsification fixture.
- [ ] **T4 — in-chunk GDN export** (Metal + CUDA arms; CPU reference parity).
- [ ] **T5 — persistence verdict** against riir-infer Issue 1108 T2's design (likely DEMOTE at
  our scale), recorded with the T1-T3 numbers.
- [-] **T6 — league-facing cell wiring into `perf_rematch.sh`** — DEFERRED until T1-T3 produce
  a measured arm (owner-gated wiring; unchanged).

## Non-goals

- No decode-quality claims from the synthetic trace (G1's bit-identity is the quality gate).
- No shared-system-prompt mixing in the base cell (its own pre-registered cell if ever).
- No default-on posture for A1/A2 regardless of the bars (opt-in until the owner gate).
