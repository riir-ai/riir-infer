## 2026-11-24 — Issue 1248: f16 bandwidth-transfer census — smaller=faster does NOT cross to ndb / chain / riir-net / seal-escrow
**Status:** LANDED (advisory census, no code work owed) — delegated Claude verdict round 1 **AGREE** (session `9b75bcf7-6a02-47ca-83fb-01114da2e439`, 2026-11-24); its one correction APPLIED — the ndb dense scorer is routed to the EXISTING ndb Plan 338 lane, no parallel re-open trigger.

- [x] f32 census (counts + per-file profile) over riir-neuron-db, riir-chain, riir-ai/crates/riir-net, seal-remake/crates/seal-escrow (read-only greps, m3, 2026-11-24, riir-infer HEAD 963825a4)
- [x] verdict drafted: NO dtype transfer — each repo fails ≥1 precondition of the f16-B shape; "smaller" already landed in each repo's own currency
- [x] delegated Claude verdict (AGREE / REVISE) recorded in the Verdict record section
- [x] verdict correction applied: ndb's one real bulk dense scorer is COVERED by ndb Plan 338 (`embed_quant`) — f16 recorded there as the UNMEASURED middle arm; this issue carries NO parallel ndb re-open trigger (substrate-first, one tracking home)
- [x] instrument note: shape-grep vocabulary registered for the re-run trigger

### The question (owner, 2026-11-24)

The f16-B serving twin (Issue 1225, Benches 1223→1226) reclaimed ~65% of the prefill GEMM wall by halving the B-operand bytes. Owner ask: can the same gain be had in ndb / chain / riir-net / seal-escrow ("eggshell" ≈ escrow) — "smaller = faster too? should we grep f32 around that?"

### The shape law (why f16-B won, and what must hold for it to transfer)

Four preconditions, ALL required:

1. **Bandwidth-bound bulk loop** — a GEMM/GEMV/dot shape moving large f32 operand blocks with few flops per byte (the Bench-1223 finding). Latency/ALU-bound loops pay a conversion tax instead of a bandwidth dividend.
2. **Locally consumed data** — never crosses sync / quorum / replay (the Issue-940 determinism law: sync-crossing f32 reductions pin fold order for bit-identity; f16 narrows further and breaks replay/anti-cheat exactness).
3. **Lossy acceptable** — the lossy-surface promotion rule applies: per-family, conditional retention on the DEPLOYED path, never bit-identity or aggregate perplexity alone.
4. **Volume >> compute** — the convert tax (f32→f16 per element) must be covered by bytes saved.

Corroborating lesson from the other side: the Q8KV family (Benches 1211/1212) is **bandwidth-governed** — load-shape moves it, ALU/dtype tricks alone don't.

### The census (measured 2026-11-24, m3, read-only)

| repo | f32 census | profile | verdict |
|---|---|---|---|
| seal-remake/crates/seal-escrow | **0 hits** | no float math at all | nothing to convert |
| riir-ai/crates/riir-net | **13 hits** | types vocabulary only (ws.rs 4, selector.rs 1) | wire is bytes; raw physicals (positions) MUST stay f32 for replay bit-identity (anti-cheat). Frame-size wins already landed as BATCHING (AVTB, plan 632) — this repo's own currency of "smaller = faster" |
| riir-chain | ~1,702 total; **wallet/money half ≈ zero** (money = `[u64; 8]`, by law) | residue: `latcal_fixed.rs` 78 + `latcal.rs` 62 (the Lean-PROVEN fixed-point encoding, Plans 004 EPSILON/SCALE), forensic/texture (offline), aoi/occlusion, programs' scoring floats (consensus-adjacent) | the chain ALREADY made its precision move — fixed-point, proof-carrying. The consensus-adjacent f32 is determinism-bound: f16 there BREAKS quorum, does not speed it. F16 = regression of the exact axis fixed-point won |
| riir-neuron-db | ~2,944 hits | vocabulary/geometry/audits (`substrate_geometry_audits` 164, `transition_error_taxonomy` 66, `shard_compactor` 66, `zone_geometry` 56) — NOT dense hot loops | hot paths are KV/index walks (LocalKv write 826 ns, BM25, latent KNN 123 µs p50). The one real bulk dense scorer — `src/dense_embed/index.rs::rerank_into` (scores a `max_candidates` shortlist, never the whole corpus) — is **COVERED by ndb Plan 338** (`embed_quant`, `src/dense_embed/quant.rs` + recall fixture): its two 2-bit arms ALREADY FAILED the pre-registered bar (recall@10 ≥ 0.95 of f32 — arm A 0.938, rotated arm B 0.812, at 768 dims; arm B opt-in permanently; the T3 real-corpus half deferred until an embedder checkpoint exists in any lane). **f16 is the UNMEASURED middle option** (near-lossless for retrieval vs 2-bit): it joins Plan 338's arm table when that cell reopens — one tracking home, never a new lane here.

### Instrument note

`grep f32` finds type vocabulary, not hot loops — counts alone are a false-positive surface (ndb's 2.9k is ~90% type/audit text). The right instrument is a SHAPE grep: `chunks_exact|dot|matvec|gemv|simd|blocks` over slices + "is this loop bytes-dominated and allocation-free". Run over the four repos here it yields ≈ zero dense-bandwidth loops: escrow/net have none at all; chain's are fixed-point-bound; ndb's hot loops are index walks.

### Where smaller = faster still pays (all infer-side; NOT this issue's work)

1. `ffn_gate_up` 28.1% — the named next decode lever (riir-ai league row).
2. `pp512` config-class rung — −6.7% measured (Bench 1227), a serving-config call.
3. producer-writes-f16 (Issue 1226) — kills the per-launch cast fallback (0.15–0.19 ms/launch); pure win, no new lossiness.
4. pp32768 retention check (Issue 1226) — the tail the promotion didn't cover.

### Re-open triggers

- A new bandwidth-bound dense f32 loop lands in any of the four repos → re-run the shape grep + this census.
- A serving consumer appears whose hot loop is bytes-dominated over locally-consumed f32 (e.g. a ndb-side dense scorer at corpus scale) → it joins **ndb Plan 338's arm table** (f16 the unmeasured middle arm; pre-register recall/retention bars THERE — `embed_quant` recall fixture), never a new lane.
- riir-net: only if a frame-budget bind re-appears that AVTB batching cannot cover — the answer there stays batching/compression, never dtype (raw-physical law).

### Verdict record

**Round 1 — AGREE** (delegated reviewer `claude_code`, session `9b75bcf7-6a02-47ca-83fb-01114da2e439`, 2026-11-24): NO-transfer conclusion upheld; the reviewer independently re-ran the shape grep across all four repos (escrow/net: zero matches; chain: offline-forensic + consensus/cold-tier feeds only — lossy there breaks bit-identical replay; ndb: small fixed-size kernels + the dense-embed index) and confirmed chain/net as written, the instrument note, and the µs-scale rerank-volume call. **One correction, applied:** the ndb dense scorer was missed as already-owned — ndb Plan 338 (`embed_quant`, 2-bit arms A/B FAILED the 0.95-recall bar, T3 deferred on no embedder checkpoint) is the existing lane; f16 recorded as its unmeasured middle arm; the parallel ndb re-open trigger in this file's original draft was removed (substrate-first: one tracking home).
