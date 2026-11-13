# Research 028 — oMLX 0.8.0rc1 LAN Prefill Accelerator — intel + defer

**Status:** RECORD — deferred intel, no novelty claim, no plan filed (owner directive 2026-11-13: "note this as defer until we get 4x M5 Ultra 128GB ... we do have 5090 btw so this is good intel to note at riir-infer").

**Source (verified 2026-11-13 via release-page fetch):** [`jundot/omlx` release `v0.8.0rc1`](https://github.com/jundot/omlx/releases/tag/v0.8.0rc1) @ `454c90d`, released 2026-11-12, signed (Jun Kim). Release candidate — stable line is 0.7.3.

## What shipped (as-reported)

**The Prefill Accelerator** — heterogeneous prefill offload over plain LAN:

> "A PC with NVIDIA GPUs, a DGX Spark or a second Mac with a faster GPU runs the first layers of each prompt chunk, while your Mac runs the rest, decodes and keeps the KV cache. Auto picks how many layers the helper takes from the measured speed of both machines and the link, and keeps the prompt on the Mac when a split would be slower."

- **The Mac must hold the whole model; the helper does not.** The GPU "speeds up as many layers as it can hold" — capacity-fitting layer assignment, not full pipeline-parallel. During decode the Mac runs everything; decode speed stays about the same. The helper is PREFILL-ONLY.
- **Auto split with a refuse-slow-split gate:** layer count from measured speeds of both machines + the link; prompt kept entirely on the Mac when a split would be slower (#5065, #5163, #5195). Helper recovery arms: OOM step-down, layers handed back for long prompts, offline retries (#5180).
- **Plain LAN, no special fabric:** 1 GbE works, 2.5 GbE recommended. One `docker run` on the NVIDIA side (`jundot/omlx-cuda-helper`, `--gpus all --network host`), no SSH or oMLX install there. Driver 580+, Ampere-or-newer GPU or GB10; not yet tested on Windows. Helper is checked per model family; both Macs need the same oMLX version.
- **Release-reported numbers** (prefill tok/s, 64K prompt, oQ4e models, M3 Ultra 512 GB + CUDA helper over 2.5 GbE; decode ~unchanged):

| Model | M3 Ultra alone | + DGX Spark | + 2× RTX A6000 |
|---|---|---|---|
| Qwen3.8 27B | 352 | 1,062 (+202%) | 1,879 (+434%) |
| Qwen3.8 Flash-Next | 1,370 | 2,715 (+98%) | 3,928 (+187%) |
| DeepSeek V4.1 Flash | 1,085 | 2,412 (+122%) | 1,914 (+76%) |
| GLM-5.3 Flash | 555 | 1,076 (+94%) | 978 (+76%) |
| MiniMax M3.1 Flash | 312 | 504 (+62%) | 468 (+50%) |

- **Provenance continuity — both kernel upstreams are ALREADY mined in riir-refine.** The CUDA helper's kernels are adapted from **TensorFold** (@ashhart) and **mlx-serve** (@ddalcu), licenses vendored in `third_party/`. TensorFold: entries_gpu_144–146 (pinned PR shas — `cef366c9`, `19c491ce`, `0afc366c`, `acb21860`, `33b443da`, `7fce15ce`, `618625fd`, `297a182d`). mlx-serve: **Batch 242** (2026-10-15, `ddalcu/mlx-serve` @ pin `1015f785`, MIT — +31 own rules entries_gpu_97–99, corpus 816→847, verdict YES B+; [Research 274](../../riir-refine/.research/274_mlx_serve_distill.md)). The helper's adapted kernels are downstream of that mined pin — the only distill debt is a delta re-pin since `1015f785`, owed only if the adaptations diverge materially.
- Also in the release (recorded for the league lanes): DSpark-drafter batched draft verification, 8-request decode 73.2 → 107.9 tok/s (+47%) on DeepSeek V4.1; TP/pipeline cluster rows over Thunderbolt RDMA; Concurrent Request Priority (Balanced keeps ≥10 tok/s streams, cuts new-prompt TTFT 26%; a short request arriving mid-prefill starts after the running chunk: 13.4 s → 4.4 s TTFT); Affine4/Affine8 KV compression decode up to +58% at long context.

⚠ **Affine KV honesty note:** their Affine4/Affine8 KV decode is **not token-identical to 0.7.3 — greedy output can differ** (#5095). Vocabulary cousin of our FOSCAR affine-folding lane (Research 026), opposite concern: they fold a *lossy* KV compression into the attention path; 026 folded *exact* affine transforms into kernels with parity held. Different axis; do not conflate when grepping "affine".

## Positioning vs our lanes (why this is intel, not a plan)

1. **Not the first sighting — the 2026-11-04 distill verdict recorded the two-Mac precursor.** riir-refine's mining queue (`.distill/001_mining_queue_snapshot.md`, oMLX v0.7.4.dev1 @ `a208eca6`, verdict MARGINAL C+ corpus / LANE INTEL HIGH) already carried Prefill Accelerator #5065 as the two-Mac disaggregated prefill: the fast Mac runs the FIRST layers of each prompt chunk, the big Mac the rest + decode + the KV host-side (DeepSeek V4.1 Flash @64K 1,060→3,123 tok/s, **+195%**; "decode unchanged by design"; "Thunderbolt or 5GbE is enough, no RDMA needed"), mapped to plan-623 + Issue 028, with the follow-up NAMED not filed (a riir-infer prefill-accelerator A/B issue, owner-gated). **What v0.8.0rc1 adds is the NVIDIA/DGX helper**: docker-run deployment, the published CUDA-helper table, RC status. That row also corroborates the prefill-only reading below (it read "decode unchanged by design" and "KV host-side" from #5065); the not-source-verified caveat still applies to the v0.8 helper itself.
2. **It is Research 015's beat-lever 5, shipped.** Lever 5: "Cost-model split, not RAM-proportional: probe per-host per-layer latency and link RTT/bandwidth at boot, split to equalise stage time subject to memory. exo has no probing despite its README." oMLX ships exactly that (measured speeds + link → layer count), plus a gate we had not spelled out: **refuse the split when it would be slower** (prompt stays on the Mac). Consumer-product validation of our own derived lever — cite 015 §3 lever 5 when the lane re-opens.
3. **Different design point than our plan-623 lane.** Our `multi_host` head/tail split (riir-infer-dist, plan 623) crosses the wire **per token** during decode — the latency-bound case that produced the RDMA/UC wire laws (UC wire shape, RC refused, recv rings ≤3 MB, Bench 928/929). oMLX's accelerator crosses the wire **only during prefill chunks** — the bandwidth case (015 §2 physics) — and decode never leaves the Mac. Same conclusion our Issue-1165 triage reached for the other half: prefill migrates, **decode stays resident** (the `RIIR_DECODE_RESIDENT` macOS-greedy default is the same instinct productized on their side as "decode speed stays about the same").
4. **Transport implication:** a prefill-only offload arm needs no RDMA — plain TCP at 2.5 GbE carried their best cells. That is lever 7's ladder rung 1 (std TCP over the bridge) shown sufficient for the prefill-only variant. If we ever add a prefill-offload arm beside the decode-resident lane, it does not inherit the UC/RDMA wire laws; it needs only the determinism posture (host-endian frame cast, strict decode) our `multi_host` wire already pins.
5. **League context:** oMLX is an MLX-format serving stack (oQ4e/oQ3e quant families) — not a bench target for our GGUF PQ2_0 ternary Bonsai lane or the qwen3.8 dense Metal posture, and it already carries the league's `watch_repo` row (`omlx: mlx python`, the Issue-846 coverage sweep) — this release moves that opponent class' capability, not its pin. It defines the **heterogeneous prefill-offload opponent class**: their best qwen3.8-27B cell is 1,879 tok/s prefill (M3 Ultra 512 GB + 2×A6000, 2.5 GbE) — a different box and stack from our M5 Ultra league rows; recorded, not comparable.
6. **Hardware posture (current):** both our CUDA boxes meet the helper's Ampere+ floor — the 4090 (Ada, the ledger's Windows box) and the owner-stated 5090 (Blackwell; the repo docs still name the 4090 — reconcile the hardware ledger when this unblocks). Driver-580+ and Windows-helper status unverified — check at unblock time.

## The defer (owner directive, verbatim condition)

**Trigger:** "4× M5 Ultra 128GB" class hardware. Fleet today: amo + bmo (M5 Ultra 96 GB), the 4090 box, owner-stated 5090. Not fired.

**What unblocks when it fires** (candidate cell sketch, nothing committed):

- A/B cell: a prefill-offload arm for the **Bonsai PQ2_0 ternary league lane** (the owner's standing rule — bonsai/ternary first; also the 11-04 row's own choice), riding OUR OWN `multi_host`/`riir-infer-dist` lane — the `LayerRange`/`StageHandshake`/`geometry_fingerprint` handshake plane, a helper process holding the first-N `LayerRange` layers. NOT via oMLX itself: **oMLX cannot load PQ2_0 GGUF**, so a cell against oMLX's published table is format-crossed, non-scoring (the landed `mlx_bonsai2_companion_bench.py` instrument + the prism-ml MLX-2bit pack cover the MLX leg per the 11-04 row). The **GDN asymmetry favors the offload shape on our lane**: the cross-node handoff unit is the FIXED-SIZE GDN state (the splash B236 capture family + Issue 028), not a growing KV cache.
- The qwen3.8 dense Metal league lane is the only oMLX-comparable posture — parked as the secondary option if oMLX itself is ever benched as opponent (owner call, perf-rematch charter).
- Decode resident (1165 posture); league shapes pp2048/pp16k/pp32k vs the single-host bars; FNV/argmax tripwires on every arm.
- G1-correctness row: the refuse-slow-split gate (split only when measured faster) + the auto layer-count vs a pinned static split.
- Engagement proof first (the Issue-1124 T1.5 law): prove the helper arm is actually reached in the measured posture before quoting any ratio.
- Optional: a delta re-pin of the oMLX helper's `third_party/` kernels vs the mined mlx-serve pin `1015f785` (Batch 242) — only if the adaptations diverge materially; the `distill` skill owns that decision.

No code, no lane, no AGENTS.md lane-table row until the trigger fires.

## Closest cousins

| Cousin | Relation |
|---|---|
| [`.research/015_Multi_Host_Inference_exo_llamacpp_MLX_Distill.md`](015_Multi_Host_Inference_exo_llamacpp_MLX_Distill.md) | The frame. oMLX = shipped consumer-product instance of levers 5 + 7-TCP-rung, in the prefill-only design point. |
| riir-refine 11-04 distill verdict row (oMLX v0.7.4.dev1 @ `a208eca6`, `.distill/001_mining_queue_snapshot.md`) | The prior record of the same feature — two-Mac #5065 (+195% DeepSeek V4.1 @64K, decode-unchanged-by-design, GDN-asymmetry-favors-us, follow-up named-not-filed). v0.8.0rc1 = the CUDA/DGX helper + docker deployment + published table on top. |
| plan 623 `multi_host` + `crates/riir-infer-dist` | Our per-token PP lane — decode crosses the wire; different (harder) point in the design space; keeps the RDMA wire laws. |
| Research 026 (FOSCAR affine folding) | Vocabulary cousin via "Affine KV"; opposite exactness axis (they lossy, we parity-held). |
| Issue 028 disaggregated handoff + `disaggregated::geometry_fingerprint` | Handshake identity plane; oMLX approximates it with per-model-family helper checks + same-version Macs. |
| Issue 1165 resident-router decode | The decode-stays-local conclusion, ours measured; oMLX ships it as a product constant. |
