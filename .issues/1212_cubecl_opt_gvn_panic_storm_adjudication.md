# Issue 1212 — the cubecl-opt GVN panic storm adjudication (riir-train Issue 687 finding 1: upstream disposition + the standing posture)

**Date:** 2026-11-12
**Status:** CLOSED — adjudicated same day (zed-0729eb7f@amo, amo). Pin stands; no backportable upstream fix exists; two re-opens named with triggers ((c) the opt-off A/B, 4090-window-gated; the stack-bump evaluation, its own GOAT). Neither is claimable today.

## The finding (inbound facts, riir-train Issue 687 finding 1)

**2,996** worker-thread panics during the plan-373 P0.5e bs-2048 arm's kernel compilation on the 4090/Vulkan:

```
thread 'DSD-4-0' (544) panicked at .../cubecl-opt-0.11.0-pre.2/src/gvn/convert.rs:21:39:
no entry found for key
```

- The run COMPLETED exit 0 — each panic is the GVN pass aborting, the compile worker recovering, the kernel serving through a fallback path. Nothing silently wrong with the numbers (the P0.5d parity gate: all 70 grad groups ≤ 9.6e-5).
- Clean-box v4 re-run reproduced the storm (riir-train Bench 707): co-tenant-independent, substrate-side.
- Reach CONFIRMED same day (zed-fe93a6a1@amo, read-only census): `cubecl-opt 0.11.0-pre.2` is IN-GRAPH on every lane compiling cubecl kernels — `cargo tree -p riir-infer-gpu --features cubecl_runtime -i cubecl-opt` is 4 hops via cubecl-wgpu → cubecl-cpp (crates.io, unvendored) → cubecl-opt; `cuda_backend` rides the same `cubecl_runtime` umbrella (same exposure class on the 4090). **The census's pending HISTORY note lands with this adjudication.**

## Panic site (pinned source read, this adjudication)

`cubecl-opt-0.11.0-pre.2/src/gvn/convert.rs:21:39` is the `leaders[val]` index inside `Expression::Instruction` conversion — `instruction.args.iter().map(|val| leaders[val])`, the GVN value→leader map lacking an operand entry during the convert-back pass; the `IndexMap` Index impl panics "no entry found for key". Classic incomplete-coverage defect in the pre-Pliron GVN implementation.

## (b) Upstream disposition — RESOLVED: no backportable fix exists; the fix path is a migration

- **The pin:** `cubecl = "=0.11.0-pre.2"` (exact, both `wgpu-msl` and `wgpu-spirv` arms) + `cubecl-environment = "=0.11.0-pre.2"`; the `[patch.crates-io]` vendor set covers ONLY cubecl-runtime / cubecl-wgpu / cubecl-common / wgpu-hal — cubecl-opt resolves RAW from crates.io (0.11.0-pre.2, checksum `d0904a49…`), matching the 687 census.
- **The upstream state:** the entire optimizer was rewritten to the Pliron IR in tracel-ai/cubecl PR #1402 ("refactor: Make everything Pliron", commit `12c49762`, 2026-08-11 — ONE DAY after pre.2's 2026-08-10 crates.io publish). That refactor is the LAST commit ever to touch `crates/cubecl-opt/src/gvn/convert.rs`; the path's full commit history contains NO panic-fix commit — the defective implementation no longer exists upstream. Crates.io corroborates the rewrite arc: 0.11.0-pre.3 (2026-08-25) collapses to 1,454 code lines / 12 files (mid-rewrite), the line matures back to 7,464 / 31 by 0.11.0 (2026-10-06); stable 0.11.4 (2026-11-09) carries only the rewritten optimizer. (The older GVN post-dominator panic, PR #1051, merged 2025-11-20 — already inside our pin; a different defect.)
- **No upstream issue filed, deliberately:** the defective code is dead upstream — their rewrite IS the fix; a report against the pre-rewrite lineage would be noise.
- **The bump path (pre.2 → 0.11.4 stable) is a FULL-STACK MIGRATION, not a patch:** `cubecl` is exact-pinned at the riir-infer-gpu dep level; the vendored forks are pre.2-pinned (the runtime fork carries our drop-queue policy fix, upstream #1359); `cubecl_runtime.rs` compiles against the pre.2 API; reflex's `laya-riir-cubecl` lane resolves through this crate. The Pliron migration changed cubecl-ir's API surface. **Re-open trigger:** a stack-bump evaluation issue (its own GOAT: serving-lane compiles + the parity gates + a league A/B), filed when a lane needs post-pre.2 cubecl features or when (c) below measures the storm's cost as material.

## (a) Shape-dependence — recorded; the minimal-repro chase CLOSED on economics

- Fires only at bs-2048 kernel specializations (2,996 panics), never at bs-512: kernel specialization IR differs by shape constants, and only the bs-2048 IRs hit the missing-leader path. Firing surface is compile-time, in compile worker threads, recovery by fallback.
- A minimal-repro chase is closed LOW-VALUE: the defective implementation is dead upstream (no patch would be authored from a repro), and the standing defenses (reach census + watch-signature) already cover exposure. **Re-open trigger: none** — a repro buys nothing the census does not.
- **amo Metal-log sweep (2026-11-12):** no `gvn/convert` / `no entry found for key` hits in riir-infer `.benchmarks/` or riir-ai `.benchmarks/` on this box — consistent with never-fired-on-Metal locally (absence is weak evidence; the watch-signature stays armed).

## (c) Fallback cost — bounded; the isolating A/B named with its prerequisite

- Both 4090 cells (contaminated 449.941 s/step, clean 10.633 s/step) ran WITH the storm — the measured numbers already include whatever runtime cost the un-GVN'd fallback kernels carry. Parity and LINEAR scaling (4× FLOPs → 4.12× time, Bench 707) hold through it: no correctness risk, no cliff.
- The cost cannot be isolated from existing cells (bounded below by 0, above by the whole measured step). pre.2 exposes NO opt toggle — `KernelOptions` (vendored `cubecl-runtime/src/kernel.rs`) carries only kernel_name/debug_symbols/cluster_dim, and the pass list is fixed inside the compiler — so an opt-off arm needs a one-line vendor patch to cubecl-opt (skip the GVN pass) or the stack bump. **Re-open trigger (the (c) closer):** 4090-window-gated, AFTER Phase 2's campaign — vendor-patch the GVN pass off, re-run the bs-2048 P0.5e arm, read the Δ s/step. Not a Phase-2 blocker (P0.5 complete, Phase 2 viable — Bench 707).

## Standing posture (the verdict)

1. **KEEP the `=0.11.0-pre.2` pin** — vendored stack + API + consumers all pre.2-pinned; the bump is its own migration issue.
2. **KEEP the watch-signature armed on every lane:** `gvn/convert.rs:21:39: no entry found for key` in compile-worker output = this defect (compile-time noise + an un-isolated runtime tax on the affected kernels; correctness unaffected).
3. The two re-opens above are named, triggered, and unclaimable today.
