# Issue 1203: the qwen38 dense wgpu lane stalls on m3 — per-box qualification gap (never booted here before T2)

**Status:** Open — the lane is qualified on amo (m5ultra) and the CUDA twin on 4090; m3 (M3 Max, the third Metal box) has never run it until the 1201 T2 G1 attempt. Found while fetching the league model to m3 for the FlashStates cell; reproduces CACHE-FREE through the league bench, so it is the forward/runtime, never the harness.

**Lease closed:** zed-m3-1201t2b@m3, 1794369600, expired 2026-11-13T04:00:00+00:00 — holder gone from m3's live threads (evidence stand recorded above; no post-expiry landing found).

**Claimed:** zed-638b5482@m3, 1795476079, expires 2026-11-25T18:00:00+00:00
Scope: E1 engagement proof first (isolated-target build + two pp512 runs), then E3 counters probe; E2/E4/E5 as the quiet window allows. The lane is m3-NATIVE (per-box M3 Max Metal behavior — amo/bmo cannot reproduce it); quiet-window + bench_preflight gates apply, no sibling compute while cells run.
(scope: the m3 stall DIAGNOSIS — repro + sample + readback-pathology triage per the suspicions below — gated on the 1191 v2 128K run completing: m3 is caffeinate-held by that run until ~2026-11-14 05:20 +0700 and the v2 walls require a quiet box, so NO GPU work on m3 before then; DIAGNOSIS READ-ONLY — the vendored cubecl tree is dirty from a sibling lane, any vendor-side fix waits for that lane)

## The evidence (2026-11-11, m3, release builds, sha256-verified model `e00082f7`)

1. **League bench reproduction, cache-free** (`qwen38_wgpu_bench --pp 512 --decode 8`, commit `8e64a036`+):
   - League-armed posture (MM2D+half-B+ATTN_TENSOR+Q6K, the issue-1176 default): `[pp 512→512] warm-up: 54.32s = 9 tok/s`, `pass 1: 106.43s = 5 tok/s`.
   - Full demote (all five envs `=0`, the incumbent packed-load route): `warm-up: 368.11s = 1 tok/s`.
   - Expected: ~966 tok/s class (bench 946's league row class; the box is AC, quiet, load < 2).
2. **Where the wall goes** (`sample(1)` on the stalled process, 1862/1862 samples): the main thread is parked in `read_logits → ComputeClient::read_one → block_on → parking::park` — the CPU is not the bottleneck; the process waits on GPU completion/round-trips that take ~0.5–1s per kernel round (a 512-token chunk = ~100 launches + readbacks ≈ 200–370s ✓ consistent with both postures' tok/s).
3. **Both postures stall equally** — the armed 1146 tensor stack AND the pre-1176 incumbent — so it is below the arm layer (command submission / readback round-trip / device clock), not a mis-armed kernel.
4. **The model was never on m3 before this session** (no qwen35-arch GGUF on the box; `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` here is `dflash`-arch, refused by `Qwen38WgpuConfig::from_gguf`). The qwen38 wgpu lane has zero m3 history — the AGENTS.md league row's "m3 rows watch-only" was the tell.
5. **The ternary Metal lane is known-good on this box** (bench 1011/1116 era) — wgpu<msl> itself works; the gap is dense-lane-specific.

## The counters cell (2026-11-24, m3, zed-m3-i1203-e1 — E1+E3 verdict evidence)

Model `qwen3.8-27b-Q4_K_M.gguf` sha256 `e00082f7…` verified (the `/Users/kat/uzu27b` copy, the flashstates-bridge bit-identical twin). Release build, isolated `/tmp/i1203`, features `qwen38_wgpu,metal_tensor_gemm`. Raw logs + full stdout/stderr: `.benchmarks/i1203_m3_stall_cells/`. Preflight PASSED before every quoted cell (`power=AC Power load=3.1–5.4 powermode=2(high) ane=ok`; HIGH POWER disclosed).

| cell | posture | pp512 pass 1 | tok/s |
|---|---|---|---|
| run1 | envs unset — `[Issue 1150] armed` + `[Issue 1171]` + `[Issue 1170]` + Q6_K banners, `relaxed_precision=true` | 116.2 s | **4** |
| run2 | all-five `=0` — `[Issue 1176] DEMOTED … the incumbent packed-load route` | 213.1 s | **2** |
| run3 | armed + `CUBECL_WGPU_MAX_TOTAL_TASKS=4096` | 113.2 s | 5 |
| run4 | armed + `CUBECL_WGPU_MAX_TOTAL_TASKS=16384` | 140.0 s | 4 |

Counters (armed, pp512, wall 139.06 s — default caps): LAUNCH 1785 × 1.2 µs (0.0% wall) · ENQUEUE_EXEC 1785 × 1.4 µs (0.0%) · **ENQUEUE_WRITE 1530 → 19.4 s (13.9%)** · **FLUSH 4636/1548 submits → 50.3 s (36.2%), REGULATE 3 blocks = 50.2 s (36.1%, 16.7 s per block) · WRITE_POLL 1.9 s (1.4%)** · READ 1 × 0.05 ms; GPU-paced share 37.5%, encode-bound 0.0%. With `…TASKS=16384` (wall 141.96 s): **REGULATE 0 blocks / 0.0 ms — the cap lift engages, ENQUEUE_WRITE 45.4 s (32.0%) + FLUSH 32.7 s (23.1%), GPU-paced share collapses to 1.1% — and the WALL IS UNMOVED.**

Verdict: the stall is (1) NOT launch/encode serialization — (b) refuted at 0.0% wall; (2) NOT the Issue-1217 in-flight cap — lifted to 16384, REGULATE zeroed, wall unchanged (amo's "fence-bound not cap-bound" cross-confirms; m3's mass sits one layer lower); (3) the mass is the **per-buffer write + submit path** (1530 writes / 1785 dispatches against ~140 s wall ⇒ ~77–92 ms per write/submit unit class on this box) **plus a ~45–50% unattributed remainder** (instrumented sections cover ~50% of wall — the un-instrumented host/GPU wait needs the E4 ablation or a per-kernel GPU timer, both blocked this window). E4's harness died silently ×3 (see the task line) — memory-pressure OOM suspect, needs a smaller-model retry at a quiet window. Lease stays live to 2026-11-25T18:00Z for that retry + the fix-arm adjudication.

## Suspicions (ranked, none proven)

- A wgpu/Metal round-trip pathology on M3 Max in the vendored cubecl-wgpu 0.11.0-pre.2 stack (the only Metal box in the fleet running a different GPU family for this lane).

## T1 prep — source-level adjudication (2026-11-13, zed-d565d322@m3; no GPU, no builds — the 1191 v2 run holds the box)

Read the vendored stack + the dense lane end to end. Findings:

1. **The wgpu 1ms-poll class is REFUTED for our binary.** vendored `wgpu-hal-30.0.0/src/metal/device.rs:2023-2047` already carries the #9328 CondVar wait (`wait_while_for` on the fence value) — gfx-rs/wgpu #7712 (1.2 ms min per poll-Wait on M2) was fixed by PR #9328 (merged 2026-06-05, in this vendor's version); #9531/#9532 fixed a >1h-CB deadlock in the same path (not our shape).
2. **`read_one` mechanics pinned** (`vendor/cubecl-wgpu…/compute/server.rs` + `stream.rs` + `poll.rs`): prologue = staging reserve + copy-encode + flush + `map_async` initiation (READ_NANOS counts ONLY this) → the caller's `block_on` waits on the map callback, delivered by the **WgpuPoll THREAD**'s `device.poll(Wait)` loop. The sampled `parking::park` in the main thread is mechanics — the question is what the wait waited FOR.
3. **Three host-block stations, all counter-instrumented**: `SubmissionLoad::regulate` (≥512 in-flight tasks → poll(Wait) prev submission — REGULATE_NANOS), the `write_to_buffer` 64-write flush (WRITE_POLL_NANOS), and the read drain (READ_NANOS + caller wall). The Issue-761 **always-on dispatch counters (LAUNCH/FLUSH/SUBMIT/REGULATE/WRITE_POLL/READ counts+nanos) exist in the vendored fork with ZERO consumers in this repo** — tomorrow's probe reads the pub statics directly (a tiny bin + `cubecl-wgpu` resolves to the fork via `[patch.crates-io]`).
4. **Metal binding economics are favorable**: on MSL, addresses bind IMMEDIATE (`ParamsTransfer::Uniform` is the Vulkan-only arm of `into_resources` — no per-launch address uniform on Metal), and the metadata uniform rides the content cache (DEFAULT ON, `RIIR_UNIFORM_CACHE=0` demotes; Bench 771 promoted 762's GOAT). The 64-write-flush storms are already mitigated; WRITE_POLL fires on cache misses/bulk writes only.
5. **Cross-queue hazard REFUTED**: the dense tensor context extracts the raw `MTLDevice`/`MTLCommandQueue` FROM the wgpu device/queue (`as_hal::<Metal>` — `gemm_dense_f16_metal_zero_copy.rs:312-327`); every dispatch commits to the SAME in-order queue, WITHOUT `wait_until_completed` (the async-pipeline law), output copied back GPU→GPU via a blit INSIDE the same CB. No cross-queue sync per GEMM.
6. **Engagement-proof trap FOUND in the evidence class** (the Issue-1124 lesson, verbatim here): the armed posture prints `[Issue 1150] dense-f16 matmul2d arm armed … relaxed_precision=<bool>` at load and demoted prints `[Issue 1176] … DEMOTED`; a build WITHOUT `metal_tensor_gemm` silently runs the INCUMBENT even with league envs unset (on non-mtg builds `dense_armed = env=="1"`, promote-only). The old evidence's 9-vs-1 tok/s split suggests the feature WAS in (a no-mtg build would read both cells as the same incumbent, modulo the Q6K=0 f32-lm_head load delta) — but **the logs adjudicate: first act in the GPU window is to re-run pp512 on an mtg build and grep BOTH banner lines from stderr.** Old runs without the armed banner = the whole "stall" was the feature-less-build trap.
7. **The anomaly REFRAMED by existing numbers — armed-path-specific, not a runtime stall:** amo's own incumbent measured ~736 s / 5120-token pass (bench 946) ≈ **0.73 s per launch-round — the same magnitude as m3's "0.5–1 s per kernel round"**. The incumbent packed route is UNIVERSALLY slow; amo escaped via the armed tensor stack (≈966 tok/s). m3 incumbent ≈ 2–5× amo incumbent (box variance + the f32-lm_head load) — unremarkable. **m3 armed ≈ 100× amo armed is the ONLY real anomaly.** If m3's wait machinery were broken, the incumbent would be equally hit; it is not. Remaining candidates, all armed-path-local: (a) matmul2d/tensor-op throughput on M3 Max's GPU family vs M5 Ultra (expect several×, not 100×); (b) a per-dispatch serialization at `cmd_buffer.commit()` from the raw-Metal side on M3 Max (this box has FORM: the dense kernel's `useResource` hazard declarations exist precisely because of "the memory-less hazard-tracking race the ternary module measured on M3 Max"); (c) `relaxed_precision` — default TRUE, demote = bit-exact costs ~2× on the ternary sibling (bench 923) — real but small; the banner prints the actual posture.

### GPU-window experiment list (post-v2, quiet box; entry point per item)

- [x] **E1 — engagement proof first — DONE 2026-11-24 (zed-m3-i1203-e1@m3), verdict: REAL STALL, both postures.** Armed banners present (relaxed_precision=true), all-five-`=0` demote banner present; armed 4 tok/s / demoted 2 tok/s at pp512 (pass-1 116.2 s / 213.1 s) — the ~240× gap reproduces WITH the armed banner, so it is not the 1124 feature-less-build trap. Evidence: `.benchmarks/i1203_m3_stall_cells/` (raw logs) + the counters section below.
- [ ] **E2 — per-launch microbench**: time `dispatch` × N (single weight, one chunk shape) on m3 vs the same N on amo (or reuse bench 923's harness shape); separates per-commit serialization (b) from kernel throughput (a). Also A/B `RIIR_QWEN38_DENSE_MM2D_RELAXED=0|unset` (~2× expected).
- [ ] **E2 — per-launch microbench — PARTIAL via the counters (2026-11-24): launch machinery EXONERATED on m3** (LAUNCH 1785 × 1.2 µs = 2.2 ms = 0.0% wall; ENQUEUE_EXEC 1.4 µs each) — per-commit serialization (b) is not the mass; the dedicated m3-vs-amo microbench cell stays unrun. The RELAXED A/B also unrun (the armed banner prints relaxed_precision=true; demote-of-relaxed alone not measured).
- [x] **E3 — the 761 counters probe — RAN 2026-11-24**, two postures (default caps + `CUBECL_WGPU_MAX_TOTAL_TASKS=16384`): verdict below — the mass is ENQUEUE_WRITE + FLUSH, NOT REGULATE once the in-flight cap lifts, and the WALL DOES NOT MOVE (1217's caps lever is m3-null, consistent with amo's fence-bound-not-cap-bound finding).
- [ ] **E4 — decode-ablate residual wall — BLOCKED 2026-11-24; death site LOCALIZED + harness hardened (2026-11-24, zed-638b5482):** the harness DIED SILENTLY on m3 ×3 (13×3 pairs, 13×1 pair, narrowed 4-classes×1) — the LAST stdout line is the matrix header in EVERY run, so the death sits inside `zero_state` + `ingest` (a full 512-chunk prefill ≈ 116 s stalled) + the 32 warm-up decode steps, never reaching any cell. Lead cause unchanged (memory-pressure OOM under sibling contention — the box reached 57.6/64 GiB; run1–4 prove the 27B posture itself completes). Driver flaw found: run5/5b carry NO .err file — stderr was unobserved, so even a Rust panic would have been invisible; the retry captures BOTH streams. **The Qwen3.5-0.8B pin is a dead end as-is:** the lane's arch gate accepts `qwen35` (the 0.8B passes it) but the weight loader accepts Q4_K/Q6_K only — `Qwen3.5-0.8B-Base-Q8_0` refuses at the first quant tensor ("expected Q4_K or Q6_K"); adding Q8_0 dequant is loader work, NOT taken (the quiet-window 27B retry is the primary). **Harness hardened** (`qwen38_wgpu_bench.rs`: warm-up-start + cell-start + per-arm-start lines; stdout is line-buffered, so a repeat death's last line names the exact stage).
- [ ] **E4 retry recipe (the quiet-window run — execute gates FIRST):** `agent-mail rig status m3` QUIET + `scripts/bench_preflight.sh` PASSED + `vm_stat` free ≥ ~40 GiB, then:
  `CARGO_TARGET_DIR=/tmp/i1203 cargo build --release -p riir-infer-gpu --features qwen38_wgpu,metal_tensor_gemm --bin qwen38_wgpu_bench`
  `caffeinate -dis /tmp/i1203/release/qwen38_wgpu_bench --gguf /Users/kat/uzu27b/models/qwen3.8-27b-Q4_K_M.gguf --ctx 4096 --decode 16 --decode-ablate-matrix --decode-ablate-pairs 3 > .benchmarks/i1203_m3_stall_cells/run6_ablate.out 2> .benchmarks/i1203_m3_stall_cells/run6_ablate.err` — BOTH streams to files (the run5 lesson); envs unset = the league-armed posture (run1's posture); ctx trimmed 24576→4096 (KV headroom under pressure; the per-class deltas are internally comparable at any ctx) and 16 steps/cell halves the stalled wall (expect ~15–30 min); per-pair lines stream progress — a repeat death's last line names its site. E5 adjudication reads run6's coarse shares + residue against the E3 counters (ENQUEUE_WRITE + FLUSH mass vs the decode wall).
- [ ] **E5 — verdict + fix — 2026-11-24 update: arm (b) per-commit serialization REFUTED by the counters** (encode 0.0% wall); the named suspect is the per-buffer ENQUEUE_WRITE + FLUSH/submit path (13.9% + 36.2% of wall at default caps; 32.0% + 23.1% with caps lifted, wall unmoved) plus a ~45–50% unattributed remainder — fix arms: batch/fewer buffer writes (the scalar-arg write shape), the ICB replay lane (1181, blocked on the macOS AGX crash), or a Metal-specific enqueue path; matmul2d throughput (a) stays possible for part of the remainder (no per-kernel GPU timing yet). 1217's caps env is NOT m3's lever.
- A device-limit stall (the Issue-730 grid guard caps wg-x at 65535, but y/z dims or threadgroup memory may hit M3-specific limits a kernel trips).
- Power-state/clock throttling on an idle GPU executing tiny kernels — would NOT explain the league bench's own warm-up being 100× off, but the parked-readback shape is consistent with sub-ms kernels each paying a ~0.5s round trip.

## The next lane (unblocks 1201's m3 arm)

1. Reproduce minimal: a two-launch probe (one trivial kernel + one readback) timing the round-trip on m3 vs amo — separates per-launch stall from kernel cost.
2. Compare `system_profiler SPMetalDataType` / power state between boxes; try `MTL_SHADER_VALIDATION=0/1`, wgpu trace capture.
3. If per-launch round-trip: look at the vendored cubecl-wgpu command-buffer flush policy (maybe it waits per dispatch on M3'spresent-vs-deferred scheduling).

Not blocking 1201 T2: the cell runs on the qualified Metal boxes (amo queued via the fleet); this issue is the m3-arm unblock.

## Upstream prior-art census — DONE (2026-11-13 07:05 +0700, zed-fa3b5839@m3; web + vendor grep, no GPU — inside the T1 read-only posture, diagnosis scope untouched)

Corrects the earlier datum: **gfx-rs/wgpu #9660 was RETRACTED, never fixed** — its own
author closed it next-day ("Fable 5 created this issue without consulting me"), zero
maintainer triage, no fix commit exists; the CHECKPOINT_PLAN's "closed with a fix
~wgpu 22" line was FALSE and is corrected there. Verified "M3 Max" census in
gfx-rs/wgpu = 4 issues, all unrelated/retracted → **zero open M3-Max perf/stall
reports upstream**; the box-family prior rests on first-party measurement only.

E5 interpretation aids (verdict class, applicability adjudicated):

1. **#8474 (wgpu MEMBER-filed, OPEN)** — MSL compiler miscompiles
   `unpack_unorm2x16/snorm2x16_to_float` on M3+ (Chrome crbug.com/dawn/449576833;
   WebKit polyfill `4ffaa038`). The credible "M3-family Metal misbehaves" instance —
   correctness, not perf. Our shape? NO: zero `unpack_*2x16` uses in riir-infer-gpu
   or the vendored cubecl lanes (grep clean). Prior only.
2. **#9414 (OPEN) — instrumentation trap for E4/E5:** Metal timestamp queries return
   ALL ZEROS on macOS 26+/Metal 4 (Apple killed `MTLCounterSampleBuffer` reads;
   `MTL4CounterHeap` unadopted in wgpu-hal 29–30). m3 is macOS 27.2: do NOT add GPU
   timestamp probes to the diagnosis; wall clock + the Issue-761 CPU-side counters
   are the instruments (lane already passes `timestamp_writes: None`).
3. **#8768 (closed dup; Dawn-confirmed)** — Apple's Metal driver allocates in
   `AGX::Compiler::compileProgram` per render pass — per-submission driver-side
   compile overhead is a real Apple-driver class; one prior for E2's
   per-commit-serialization arm (b). Compute-pipeline analog unverified.
4. Vendor freshness (re-verified): #9328 `wait_while_for` IS in
   `vendor/wgpu-hal-30.0.0/src/metal/device.rs:2042` (T1 finding stands). #9100,
   #9215/#9645 (ray-tracing sync), #10013/#10780 (sRGB presentation), #10639/#10640
   (CPU encode cost, M3 Pro) — all non-shapes for a headless compute lane; recorded
   so the next dig does not re-read them.

## References

- `8e64a036` — the 1201 T2 landing whose G1 attempt surfaced this.
- `../.benchmarks/` league records (946/1150/1160) — the qualified boxes' numbers.
- riir-ai AGENTS.md M3 league row — "m3 rows watch-only".
