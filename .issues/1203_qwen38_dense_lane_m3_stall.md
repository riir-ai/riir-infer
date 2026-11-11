# Issue 1203: the qwen38 dense wgpu lane stalls on m3 — per-box qualification gap (never booted here before T2)

**Status:** Open — the lane is qualified on amo (m5ultra) and the CUDA twin on 4090; m3 (M3 Max, the third Metal box) has never run it until the 1201 T2 G1 attempt. Found while fetching the league model to m3 for the FlashStates cell; reproduces CACHE-FREE through the league bench, so it is the forward/runtime, never the harness.

**Claimed:** zed-m3-1201t2b@m3, 1794369600, expires 2026-11-13T04:00:00+00:00

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
