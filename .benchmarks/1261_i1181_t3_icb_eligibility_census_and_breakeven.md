# Bench 1261 — Issue 1181 T3: the ICB-replay eligibility census (resident decode bind tables) + the T2-remainder break-even

**Date:** 2026-11-25 · **Box:** m3 (M3 Max, static code census + one 0.1 s fixture re-run) · **Session:** zed-m3-i1181-t3@m3
**Lease:** `.issues/1181` (`**Claimed:** zed-m3-i1181-t3@m3`, 1795593157, exp 2026-11-26T12:00Z)
**Method:** source census of the resident per-token dispatch sequence (`resident_enqueue` → `forward_dispatch_only` → `forward_from_x_body_range` → the per-layer kernels), every launch classified against the `icb_replay.rs` contracts; the T4 break-even priced from the fixture's measured per-replay host-enqueue rows (the 2026-11-25 m3 quiet-box rows of record: remedy 69.0 / source 35.3 / direct 33.0 µs; this session's re-run under a load ~5–16 sibling window read 106.4 / 37.2 / 33.8 — direction unchanged, quoted as the box-state caveat only).

## The question (T3, pre-registered in the issue)

Which subgraphs of the resident decode per-token dispatch sequence are **fully static** — every command with a frozen pipeline (`supportIndirectCommandBuffers = YES`), frozen grid/tpg, explicit fixed bindings, and NO patchable field or scalar binding — and therefore representable as one `ReplayGroup::record`-able group per the B223/B319/metal-graph eligibility vocabulary? Multi-node groups fall back to direct encode until hazard-safe grouping is proven.

## The census — one resident enqueue, launch by launch

The resident path runs on **CubeCL/wgpu** (Metal via the vendored wgpu backend), NOT raw metal-rs — the census classifies the dispatch sequence in `IcbCommand` vocabulary and treats the runtime bridge as a named integration surface (see §Bridge, below). Launch geometry below is for the 27B folded census posture (Bonsai-2-27B, n_layer 64, vocab 248320, n=5120).

| # | launch (per token) | grid (static?) | bindings | per-token-varying scalars | eligibility |
|---|---|---|---|---|---|
| 1 | `DequantWteRowIndirectCubeCL` (indirect embed) | Static(n/256) — **static** | wte planes ×2, scales, **id ring**, x | `id_slot` = `(t+k−1)%k` — **cyclic period k** | **ELIGIBLE via the k-phase group set** (scalar slot + the `id_buf` bind LENGTH are both phase-indexed) |
| 2 | `RotationCubeCL::launch_inverse` (folded wte) | static | x, signs | none | **ELIGIBLE** |
| 3.. | per layer ×64: input norm (fused residual+norm+rotate per posture), GDN: in-proj concat GEMV + dense a/b pair GEMV, conv1d, beta/decay, expand+L2, recurrence (rowpar, state RW), zgate, [permute-rotate], out GEMV; FFN: gate|up GEMV, swiglu(+rotate fused), down GEMV(+residual fused); tail: final norm, rotate stage, lm_head GEMV | all shape-derived, **static** | weights (persistent), activations/scratch/states (persistent handles) | none found | **ELIGIBLE — the 1127 "shape-only" dispatches** (the resident doc's own count; confirmed: no `pos`, no `t` scalar reaches any GDN/FFN launch; state/conv/KV advance is device-resident dataflow) |
| A | attention layer (one per layer at `global_attn_every_n_layers`): Wqkv GEMV, split-qg, qk-norm, **RoPE partial**, **KV append**, **flash decode** (+split-K pair over min-pos 512), [rotate], wo GEMV | grid **varies with n_positions** (decode kernel one cube per head is static; the split-K arms and grid guards are length-driven) | KV caches (persistent), rope tables (persistent) | `pos` (RoPE + KV append params + cache bind LENGTH `(pos+1)·kvd`), `n_positions` (flash params + split geometry + grid guards) | **INELIGIBLE at record time** — `pos` advances every token, unboundedly; NOT periodic, no k-phase trick |
| Z | `ArgmaxLastCubeCL` (ring write) | static (vocab reduction) | logits, **ring slot sub-handle** | slot offset `(t%k)·32B` — **cyclic period k** | **ELIGIBLE via the k-phase group set** |
| − | `memory_cleanup` (cleanup_interval only; **defaults 0 = never** in the decode walks) | — | — | — | absent by default |

**Classification verdict: the sequence splits into a large fully-static class and one hard wall.**

- **Static class (≈93% of the launch count on the 64-layer hybrid: 1127 GDN/FFN + 2 ring + 1 embed of the ~1215/token count at the Issue-1168 P1 posture):** every GDN and FFN launch, the embed, the rotation, the argmax. Bindings are persistent pool handles; the only per-token variation is the ring phase (cyclic, period k).
- **The k-phase resolution (the census's constructive finding):** record **k groups** (k = the resident ring depth, 8 at the shipping cadence), each at one phase; token t replays group `t mod k`. Within a phase, `id_slot`, the argmax slot offset, and the `id_buf` bind length are CONSTANTS. The groups share every buffer; total ICB capacity = k × 1129 ≈ 9032 commands ≤ the metal `MTLIndirectCommandBufferDescriptor` window (1024 per descriptor ⇒ 9 descriptors or per-layer grouping; capacity is a real but bounded cost, not a blocker).
- **The wall (attention):** `pos` enters the RoPE and KV-append param buffers AND the KV-cache bind length every token, strictly increasing — INELIGIBLE per the discipline (a patchable field with no device-side advance). The metal-graph `mg_icb_dispatch_eligible` shape refuses it. Two honest remediations, both OUT of T3's scope: T4's declared patch contracts (re-declare the param buffers + bind length per replay — the patch slot vocabulary exists in the machinery's scalar-buffer design), or the graph-unsafe alternative of moving pos device-side (a new kernel, never pre-registered here).

## The T2-remainder break-even (T4's formula, priced now)

`break_even_launches = instantiate_cost / per_launch_savings`, savings ≤ 0 ⇒ LOUD no-break-even.

**Host-side economics (the fixture's rows of record, m3 quiet):**

- instantiate = `ReplayGroup::record` over the group (1129 commands): unmeasured directly, but the machinery's per-command work is the same class as the fixture's 2-command record; the direct per-command encode datapoint bounds it: the fixture's direct-encode control is 33.0 µs for 2 commands ⇒ ~16.5 µs/command ⇒ record ≈ **18.6 ms per 1129-command group** (per phase; ×k groups = ~149 ms once per stream — amortized to irrelevance over a generation, and prefill-adjacent).
- per-launch (per-replay) savings, remedy posture: **direct 33.0 µs (2 commands) vs remedy 69.0 µs (2 commands)** — savings NEGATIVE (−36 µs) per replay of a 2-command group. The remedy's blit-copy doubles the enqueue; the source posture (35.3 µs) is ≈ parity (+2.3 µs).
- **LOUD VERDICT (T4's law, applied now with existing data): at the measured per-command costs, ICB replay does NOT break even on host enqueue for the resident group — per-command replay enqueue (≈34.5 µs/2 = 17.25 µs rem./cmd vs 16.5 direct) is parity-to-negative at the source posture and 2× negative at the remedy posture.** `break_even_launches` is INFINITE at both postures on this evidence; the demote per the pre-registered savings ≤ 0 clause.

**What would flip the verdict (the honest conditions):**

1. **The group's GPU-side win is not the host enqueue at all — it is the inter-launch gap.** The measured target class is the Issue-734 finding's CUDA twin: GPU-side inter-kernel gaps dominate deep dispatch streams (their datum: ~34 µs × P dispatches). An ICB execute dispatches 1129 commands with ONE host enqueue and hardware-side replay — if it also compresses the GPU-side gaps, the win is wall-side, and the host A/B (which excludes GPU time by construction) cannot see it. THE deciding cell is a WALL A/B of replay-vs-direct over the real static group, never another host-enqueue round. This census prices the host half; the wall half needs the bridge first.
2. **The remedy posture's blit cost is per-replay fixed (≈34 µs over the source), while its value scales with group SIZE** — at 1129 commands the blit amortizes to ~30 ns/command; the crossover vs the source posture sits where per-replay degradation (the #1993 10→23 ms class over 20 replays) exceeds ~34 µs/replay. The degradation tax was never measured on this driver (the fixture's source arm ran clean over 20 replays at 4096 elements); a longer-replay degradation census is the missing half of the remedy-vs-source pick.
3. **The wall A/B's precondition is the harvest-fence cadence lever** (the T1/T2 record's sequence: encode is TODAY overlapped inside the fence wait — 1217's caps lift measured 0.05 ms/tok, the wait conserved at the fence; until the fence lever lands, ANY host-encode saving reads ~0 wall).

## The bridge finding (named, load-bearing for any follow-up)

The resident path's runtime is CubeCL/wgpu; the machinery is raw metal-rs. The tree's ONLY existing raw-Metal-over-CubeCL-buffers seam is **`gemv_ternary_msl_field::dispatch_zero_copy`** (the Issue-1138 field lane): it extracts raw `id<MTLBuffer>` pointers from CubeCL pool handles and encodes raw Metal dispatches on the SAME queue (raw shared device/queue pointers held valid by the wgpu Arcs; ordering via the shared in-order queue; hazard declaration per buffer + output aliasing handled with a dedicated output + blit-back — the exact pattern an ICB execute declaration needs, already measured in the field lane).

A replay arm over the resident static class would therefore compose: (a) the field lane's zero-copy extraction + queue-sharing for the buffers and the execute encoder; (b) pipelines rebuilt with `icb_pipeline` (`supportIndirectCommandBuffers` — wgpu-created pipelines are NOT known ICB-capable; the static class's kernels must exist as raw-MSL twins the way the field lane carries its own `gemv_ternary_msl_field.metal` — THE dominant build cost, ~30 kernel translations for the 1129-command class); (c) scalar materialization via `ReplayGroup::scalar_buffer` for the static params buffers (they are already persistent handles — the field lane's `params_handle` cache makes them natural scalar buffers); (d) the k-phase group set for the ring phase.

**Bridge cost verdict: the census does NOT greenlight a build.** The k-phase shape is real and the contracts are now named, but the wall (attention) excludes ~7% of launches INCLUDING the model's only position-carrying kernels, the wall-side value is unproven (host A/B is negative-to-parity), and the kernel-twin build cost is large. The lane's honest next unit — IF a holder picks it up — is the deciding wall A/B in the cheapest representable posture (one GDN layer's ~18 static launches as a 2-pipeline group through a field-lane-style bridge), not the full 1129-command group.

## What this does NOT claim

- No production posture change: the resident default, the incumbent encode path, and every shipped lane are untouched. The machinery stays built-not-linked; the census is a report.
- No CUDA claim: the prefill graphs arm is a different engine (its lane-map row in the league record stands).
- No claim that the attention kernels are permanently ineligible — T4's declared patch contracts could admit them; that is a new pre-registration, not this census.

## Provenance

- Census read at riir-infer develop `ce4922b7` (the lease commit); vendor pins: cubecl-runtime/cubecl-wgpu `0.11.0-pre.2` (vendored, the fork carries the Issue-1217 env caps + the Issue-761 counters), metal `0.31.0`.
- Cost rows: `icb_replay_cost_ab` m3 release quiet 2026-11-25 (69.0/35.3/33.0 µs — the T2 row of record); this session's re-run (106.4/37.2/33.8) under m3 load 5–16 — the load window's host-enqueue inflation is the remedy arm's (blit allocations), direction stable. The smoke fixture re-ran GREEN this session (2/2 passed; bit-identity unchanged).
- Launch-count cross-check: the resident doc's "1127 shape-only layer dispatches" + embed 1 + rotation 1 (folded) + argmax 1 = 1130 static-class commands/token at the census posture; the T1 counters row measured n=108384 launches / 96 tokens ≈ 1129/token — AGREES within the posture's attention-layer count (the probe's 64-layer config carries 0 attention layers in the pure-GDN census posture; production hybrid adds the wall class).
- Fence sequence of record: Bench 1218 §"Issue-1217 caps A/B" (the wait conserved at the harvest fence; the fence cadence is the named next lever).
