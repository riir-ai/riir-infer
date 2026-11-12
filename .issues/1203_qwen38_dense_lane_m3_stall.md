# Issue 1203: the qwen38 dense wgpu lane stalls on m3 — per-box qualification gap (never booted here before T2)

**Status:** Open — the lane is qualified on amo (m5ultra) and the CUDA twin on 4090; m3 (M3 Max, the third Metal box) has never run it until the 1201 T2 G1 attempt. Found while fetching the league model to m3 for the FlashStates cell; reproduces CACHE-FREE through the league bench, so it is the forward/runtime, never the harness.

**Lease closed:** zed-m3-1201t2b@m3, 1794369600, expired 2026-11-13T04:00:00+00:00 — holder gone from m3's live threads (evidence stand recorded above; no post-expiry landing found).

**Claimed:** zed-d565d322@m3, 1794522000, expires 2026-11-14T18:00:00+00:00
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

- [ ] **E1 — engagement proof first**: rebuild `qwen38_wgpu_bench` WITH `--features qwen38_wgpu,metal_tensor_gemm` (isolated target dir), run `--pp 512` twice (envs unset; then all-five `=0`), grep stderr for the `[Issue 1150] armed` / `[Issue 1176] DEMOTED` banners + `relaxed_precision=`. ~100× reproduced WITH the armed banner → real; armed banner absent in the old evidence → issue closes as the 1124 feature-less-build trap.
- [ ] **E2 — per-launch microbench**: time `dispatch` × N (single weight, one chunk shape) on m3 vs the same N on amo (or reuse bench 923's harness shape); separates per-commit serialization (b) from kernel throughput (a). Also A/B `RIIR_QWEN38_DENSE_MM2D_RELAXED=0|unset` (~2× expected).
- [ ] **E3 — the 761 counters probe**: tiny bin reading the vendored fork's pub statics (reset_dispatch_profile → run one pp512 → dump LAUNCH/FLUSH/REGULATE/WRITE_POLL/READ counts+nanos). Attributes the mass: encode vs REGULATE vs WRITE_POLL vs read-drain.
- [ ] **E4 — decode-ablate residual wall**: `--decode-ablate-matrix --decode 32` — the ALL cell's residual (sync/host floor) vs per-class deltas; if the residual dominates, it is the read/sync path; if per-class deltas dominate, it is kernel cost.
- [ ] **E5 — verdict + fix**: per the winning arm — m3-specific driver quirk (b) → consider `wait_until_completed`-free batched commits or a commit-amortization knob behind a feature flag (GOAT-gated); matmul2d throughput (a) → geometry/MSL adjudication; 1124-trap → close 1203, re-run the 1201 T2 G1 on m3 properly.
- A device-limit stall (the Issue-730 grid guard caps wg-x at 65535, but y/z dims or threadgroup memory may hit M3-specific limits a kernel trips).
- Power-state/clock throttling on an idle GPU executing tiny kernels — would NOT explain the league bench's own warm-up being 100× off, but the parked-readback shape is consistent with sub-ms kernels each paying a ~0.5s round trip.

## The next lane (unblocks 1201's m3 arm)

1. Reproduce minimal: a two-launch probe (one trivial kernel + one readback) timing the round-trip on m3 vs amo — separates per-launch stall from kernel cost.
2. Compare `system_profiler SPMetalDataType` / power state between boxes; try `MTL_SHADER_VALIDATION=0/1`, wgpu trace capture.
3. If per-launch round-trip: look at the vendored cubecl-wgpu command-buffer flush policy (maybe it waits per dispatch on M3'spresent-vs-deferred scheduling).

Not blocking 1201 T2: the cell runs on the qualified Metal boxes (amo queued via the fleet); this issue is the m3-arm unblock.

## References

- `8e64a036` — the 1201 T2 landing whose G1 attempt surfaced this.
- `../.benchmarks/` league records (946/1150/1160) — the qualified boxes' numbers.
- riir-ai AGENTS.md M3 league row — "m3 rows watch-only".
