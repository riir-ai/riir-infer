# Issue 1231 — the resident wall's pipelined snapshot-future harvest: overlap the fence with the next burst's enqueue phase (the 1230 re-aim)

**Status:** Open — T1 LANDED 2026-11-17 (riir-infer `f881fb87` + riir-ai `387cb84be`; the cell is the remaining work). Filed from Issue 1230's close (the counter-drain mechanism refuted-by-analysis — fence-equivalent in an in-order queue) and 1218 T2's measured-negative ([Bench 1229](../.benchmarks/1229_i1218_t2_cadence_wall_cell.md)).
**Lease closed:** amo-i1231, epoch 1794874761, expired 2026-11-18T23:59:43Z unworked past T1 — folded per the release law.
**Claimed:** amo-idle-hk, 1795062768, expires 2026-11-20T10:32:48+00:00 — T2: the wall cell + stop cell at the pre-registered bars (quiet exclusive on amo, rig status QUIET load 1.48 at claim time; the 1226-walk collision window checked empty).

## The mechanism (verified against the vendored fork's source, not assumed)

The burst-fenced resident wall (16.0 ms/tok, Bench 1218 posture) sits ~0.68 ms/tok above the never-fenced deep-queue floor (15.32, Issue 1168 P2). The residual is the burst-boundary serialization: `resident_harvest`'s blocking `read_one` is issued only AFTER the burst's enqueue phase, so the host idles through [wake + detokenize + next enqueue phase] while the GPU has nothing queued. The lever: issue the ring's staging copy at the burst BOUNDARY (FIFO after that burst's argmax — a snapshot of exactly that burst's window, before any later overwrite), keep enqueueing the next burst, and block the future only then. The fence then overlaps the next enqueue phase; the GPU never drains.

Three load-bearing facts verified in the vendored fork (the create-now-block-later legality):

1. `ComputeClient::read_async` encodes the staging copy EAGERLY at call time (`do_read` → `submit_blocking` runs `server.read` on the device thread immediately: staging reserve + copy encode + flush + `map_async` — the Issue-761 synchronous prologue). The future resolves when the map callback fires (channel recv), not at queue drain.
2. The copy sits FIFO between burst j's argmax and burst j+1's first dispatch — it snapshots exactly burst j's window regardless of when the host awaits. Ring overwrite by later bursts cannot tear an already-executed copy.
3. The dedicated `WgpuPoll` thread (held alive by the future's `start_polling` handle) pumps the device while the host does other work — the callback fires without the host polling. `block_on` later wakes on the buffered result.

The vendor patch that made it expressible: `read_async`'s RPIT pinned `+ 'static` (edition-2024 capture tied the future to `&self` for no soundness reason — the body returns a `DynFut`, 'static by the dyn default). Patched in OUR fork (riir-infer `vendor/cubecl-runtime-0.11.0-pre.2`, the #1359 patch home) and synced byte-identical to riir-ai's vendor copy.

## T1 — LANDED (riir-infer `f881fb87` + riir-ai `387cb84be`)

- Substrate: `resident_harvest_fut(count)` (the snapshot future; bounds asserted at call time — the count must refuse before the ids exist), `extract_window` (the ONE ring-window extractor, shared with the synchronous `resident_harvest` — the slot arithmetic has a single definition site), `resident_block_on` (the blocking half under its own name; `cubecl_runtime`-gated, NOT multi_host — the league cell posture must reach it).
- Consumer: `ResidentStream::harvest_fut()` (`+ 'static` pinned in the 2024-edition consumer too — the caller holds the future ACROSS `enqueue_token()` calls) + the two cells:
  - `bench_831_decode_long_resident_pipeline` — the wall cell: cadence vs pipelined ABBA×pairs, walk identity pairwise + across pairs, the WITHIN-CELL delta as the instrument of record (the Bench-1229 drift lesson: never a cross-run pair-mean).
  - `bench_831_resident_stop_identity_pipeline` — the stop contract under the one-burst-late detok: visible prefix truncation-exact vs the incumbent, overshoot bounded (≤ cadence overshoot + one burst) and REPORTED.
- The promoted cells (`resident_ab`, `resident_stop_identity`) are untouched — verbatim sibling copies carry the probe/incumbent/cadence arms (the battery reads as a pair).

## T2 — the wall cell (pre-registered bars, BEFORE any run)

- **Quiet exclusive is a HARD premise** (the 1229 lesson): `rig run --exclusive` after `rig status` shows QUIET, provenance (load, AC, powermode) recorded beside every number. bmo preferred (amo carries concurrent lanes).
- Posture: the bench_831 battery env (ctx=2048 n=512 pairs=3, census uniform-cache ON), burst K = the cadence resolver (unset = 8), ring depth 2×K.
- **Adopt iff the pipelined arm improves ≥ 0.3 ms/tok vs the same-cell cadence arm** (the within-cell delta, median of pairs) AND G1 walk identity holds (FNV pin + pairwise asserts) AND the stop cell's visible prefix stays truncation-exact with the overshoot bounded ≤ 2K.
- TTFT recorded (the lead costs one burst of first-visible latency — the streaming-shape axis, reported not gated).
- Regression or parity drift = measured-negative: the lane stays a bench-only instrument, no serving change, no default change.
- Adoption path (only on a PASS): the serving loop wires the pipelined schedule behind a PROMOTE-ONLY env (the resolver pattern), then the GOAT gates + the 1165-style owner verdict. The env does NOT exist yet by design — a knob arming nothing is config rot (P22).

## Tasks

- [x] **T1 — substrate + consumer + cells** — LANDED `f881fb87` + `387cb84be` (see above; clippy clean at the resident + full bench postures, the bench target gated at RELEASE after the dev run green-zeroed it, rustfmt edition 2024, grammar pins pass).
- [x] **T2 — the wall cell + stop cell at the pre-registered bars** — **PASS, ADOPT BAR MET** (Bench 1243, 2026-11-19, amo-idle-hk lease; quiet-exclusive amo, load 1.43–1.74): pipelined −0.351 ms/tok vs the same-cell cadence arm (14.98→14.63, pairs 1–3 identical, ratio 0.977×; ≥ the 0.3 bar), G1 FNV `121982e1b7878d05` identical to the cadence ref, stop prefixes truncation-exact both cells, overshoot 7 ≤ 2K, TTFT 222.2 vs 142.7 ms reported. **The cell's first catch (fixed same session, riir-ai):** the wall cell's pipelined loop lacked the EOS drain — the 7 tail tokens after the last snapshot were never harvested, the walk ended 505/512 (G1 pair assert red); fix = the post-future EOS `harvest()` in `resident_chain_pipelined` (the stop cell's tail-path contract); re-run green, the numbers above are the fixed run. T3 (serving wiring + GOAT + owner verdict) unblocked, unclaimed.
- [ ] **T3 — GOAT + promotion lane** (only on a T2 PASS; the serving loop + env + owner verdict).

## Boundary

riir-infer substrate (the resident decode lane's fence mechanics; plan-626 surface) + the vendor fork we already own. Consumer-side routing stays riir-ai. No default posture changes without the GOAT gates + the 1165-style owner verdict.
