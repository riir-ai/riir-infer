# Issue 1191 — Metal FlashMemory budget decode arm + the resumable 128K gate (issue 1117 T4 vehicle)

**Status:** OPEN — `- [ ]` tasks below. Claimed lane; forward-only per the compute law (2026-11-09): the T4 measured cell runs on METAL, never a long CPU cell. CPU stays as the short parity snapshot only.

**Claimed:** amo-zed-1191metal@amo, 1794220517, expires 2026-11-11T10:35:17+00:00

## Why

Issue 1117's T3 quality gate (Bench 893) ran ~12.6 h of teacher-forced CPU decode (M3 Max, 3.6 tok/s dense declining to 2.8 under load) while the Metal stack sat idle — and T4 (the 128K axis) is parked at "~16 h-class on CPU", which is exactly why it has not run. Owner compute law (2026-11-09, repeated 10×): GPU first, every run; a missing GPU arm is a work item, not an excuse; long runs are resumable by construction; the result must profit riir-rethink / riir-refine / riir-infer. This issue IS that work item.

## Tasks

- [ ] T1 — Metal budget arm in `riir-infer-gpu` behind a new feature `flashmemory_budget_metal` (opt-in; env PROMOTE-ONLY per the env-switch law until the GOAT, then demote-only): the three missing kernels — (a) block-centroid scoring (GEMV over per-block mean keys; the Metal dense-pair GEMV arm is the base), (b) top-ρ selection (small top-k, first-index strict tie rule per the determinism law), (c) block gather out of the existing KV cache layout the Metal stack already serves. The GDN scan / GLU FFN / ktile64 attention arms are NOT touched.
- [ ] T2 — Parity gate vs the CPU reference lane (`forward_qwen_deltanet_ternary_flashmemory`, `51f52f4`): SHORT CPU snapshot, then the Metal arm at the same ρ grid; G5-class bars (top-1 agreement + NLL drift bound) recorded before any measured cell quotes numbers. Bit-exactness is not required across backends; the CPU record stays the fidelity reference.
- [ ] T3 — Resumable gate runner: `flashmemory_decode_gate` gains a durable state dir (default `~/.cache/i1117/`, NEVER /tmp) + `--resume`: per-arm/per-segment checkpoint flush (corpus BLAKE3 + step cursor + arm table + sampled metrics so far), idempotent relaunch continuing from the last flushed cursor. The Bench 893 v1 failure class (killed + corpus recipe lost with /tmp) must be impossible by construction. CPU cells keep the same runner.
- [ ] T4 — The measured cell: 32K re-cell + the 128K axis ON METAL (M3, quiet rig; `scripts/bench_preflight.sh` posture, GPU exclusivity noted). Deliver: speedup/retention curve vs Metal dense, the crossover ρ re-measured (the ρ ≈ 0.2 crossover was a CPU-internal ratio — Metal may move it), per-bucket ΔNLL/top-1 at the ρ grid. This is the 1117 T4 confirmation that gates `flashmemory_budget` promotion.
- [ ] T5 — Record + verdict: bench record in `.benchmarks/`, issue 1117 T4 closed on PASS, promotion recommendation (ρ ≤ 0.15 operating point) to the consumers (riir-infer serving posture; Rethink/reflex decision lanes are the downstream profit).

## Non-goals

- No CPU long cells — the CPU posture is the parity snapshot only.
- No CUDA twin here (4090 gets it AFTER Metal passes parity; separate work item).
- No change to the dense decode path numerics.
