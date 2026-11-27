# Bench 1273 — i1244 T3: HiSA rotation census verdict (merged 4-shard census + T3.1 classify + T2.3 agreement)

**Issue:** riir-infer Issue 1244 (HiSA rotation census + header-only planner PoC; report-only instruments, RECORD-tier per Research 040 §4).
**Landing session:** zed-bd6cf660@amo, 2026-11-27T11:1xZ (the sixth inbox of the landing chain; wake-executed at the 11:10Z armed wake after shards completed 10:31–10:50Z).

## What ran

The T2.2 census sweep — 4 Shared rig shards (b7795e3f / b11b8955 / aca86ac8 / 4ce1c735), 192 cells each = **768 cells** over the union grid: 16 attention layers × 4 rotating chunks × 12 rotation-frequency bands `[(0,1), (1,2), (2,4), (4,8), (8,16), (16,32), (32,64), (64,128), (128,256), (256,512), (512,1024), (1024,2048)]`. Each cell = a dose-response pair (replace vs remove pass) over a frozen ctx-2048 read of `Ternary-Bonsai-2.5-27B-PQ2_0.gguf` (read_len 2048, null_offset 2048, steps 2048). Per-cell walls 407–857 s loaded; per-shard report digests `cdc9ae93…` / `3a235366…` / `e43f9242…` / `928917e5…`.

Merge + classify by the T3 landing tooling (`hisa_merge`, `d89f6352`): parse ×4 → duplicate fold → INCOMPLETE-union refusal gate (passed) → ONE canonical `HISACEN1` report over the union grid → T3.1 classify → T2.3 planner agreement. Artifacts:

- merged report: `~/riir-bench/i1244_merged.HISACEN1.txt` — source blake3 `6a9879f4cffd251181c501349cbb4b3b92d7d28539ec6198c633bf43561c6775`
- verdict: `~/riir-bench/i1244_verdict.HISAVERDICT1.txt` — blake3 `c67435025b8c847282311cff7fe1980edd2ab3da7430febc10bf1059bcd482b6` (binds the source digest)
- classify slack: `supp_abs = 1.000e-3` nats (pre-registered, disclosed in every verdict line): band suppressed iff `d_nll_rem ≤ |d_nll_rep| + 1e-3`
- model header at classify time: `freq_base=1.000e7 key_length=256 d_rot=64 p_rot=32 phi_deg=90` (the T0 pins, re-read from the checkpoint)

**PROVENANCE (bench_preflight at the merge/classify read):** `power=AC Power load=3.84 swap=5153.38M canary=skipped powermode=0(auto) ane=ok` — PASSED. (The census is a behavioral/decision-content read, not a latency claim; the provenance line still rides per the box-state law.)

## T3.1 — the typed verdict (64 groups × 12 bands)

| class | groups | meaning |
|---|---|---|
| `partial_rope_sufficient` | **16** | suppressed at ALL 12 bands — position-free behavior across the whole rotation spectrum |
| `hisa_windows_required` | **13** | distance-conditional: near bands not suppressed + far bands suppressed |
| `no_suppression` | **0** | no group carries mass everywhere |
| `mixed_unclassified` | **35** | typed loud, never force-fitted — scattered near/low bands fail the suppression test inside the group (positive margins at isolated near bands, e.g. layer 63's (0,1) band reads +5e-4…+1e-3 across its chunks) |

Per-(layer, chunk) classes + per-band margins + the suppression bit-strings are in the verdict file verbatim. The margin field is `d_nll_rem − (|d_nll_rep| + supp_abs)` per band: negative = suppressed, positive = carried.

## The T3.2 demote discipline — applied

The backbone is **partial-RoPE** (T0: `rope.dimension_count = 64` of key_length 256 — 75% of key dims position-free, 4-layer full-attention interval). The verdict contains `hisa_windows_required` groups (13) — **distance-conditional suppression EXISTS on this backbone** — so T3.2's binding clause fires:

> "a `no_suppression` verdict, OR a `hisa_windows_required` verdict on a partial-RoPE backbone (where the lever arithmetic is 10–14.5% × 1/4 layers and cannot clear the Issue-1211 ≥5% e2e bar), CLOSES the kernel lane loud — no follow-up issue."

**THE DECODE RING-MODE KERNEL LANE IS CLOSED, LOUD.** The closure is a measured one: the census's suppression is real (no_suppression = 0 — every group suppresses at some bands) but the only class that could ever justify a kernel arm is distance-conditional, and on this backbone the lever arithmetic (10–14.5% K bytes on the rotating quarter of the 1-in-4 attention layers) is below the Issue-1211 ≥5% e2e bar by construction. A kernel-arm follow-up issue files ONLY on `hisa_windows_required` at material effect size AND a full-rotary serving surface in scope — today none exists outside the paused DFlash2 drafter, whose lane owns any such surface; nothing files from here.

## T2.3 — header-only planner agreement (POST-tables, per the ordering law)

The planner's exact per-pair windows predict: suppressed 80 / not-suppressed 672 / partial (kept out of the binary agreement) 16. Measured confusion over the binary subset: **tp=66, fp=604, fn=14, tn=68 → agreement = 0.1782** — the floor pins here, with the tables present, as the ordering law requires. Reading: the header-only planner's coarse per-pair windows substantially over-predict suppression (fp 604) — the real suppression map is per-(layer, chunk, band) specific in a way the planner's per-pair summary cannot see. The planner stays the PoC it was scoped as (REPORT-ONLY, Research 040 §4); no serving surface consumes it.

## Notes

- The four shard reports' row LINES are byte-identical inside the merged report (format∘parse idempotence at the fixed .9e precision — the tooling's own test).
- The 35 `mixed_unclassified` groups are the honest outcome of the pre-registered classify rule at slack 1e-3: suppression is band-scattered, not window-clean, in over half the groups. A future re-classification would need a different pre-registered rule (e.g. majority-band or margin-magnitude tiers) — that is a NEW decision, not this record's to make; the typed-loud class exists precisely so this over-claim cannot happen silently.
- The landing chain: six inbox hand-offs (fc40d4ff → efbee190 → 3ec64b7c → f3590021 → 4944b3ae → bd6cf660), each death re-stamping the lease and re-arming the wake on the live inbox; the shards and their RETRY (the dropped `--resume`-vs-fresh-dir flag) survived every hand-off.
- Teardown at close: lease folded, `/tmp/i1244_merge` removed, the `/Users/katopz/git/i1244-merge-wt` landing worktree removed (built because the shared checkout sat detached at an 11-15-era commit under the i067-T2 ArmC lane's parked state — the landing tooling was absent at that tip).
