# riir-infer Issue 1177 — dense-lane residues: the cross-backend bit-parity G1 cell (4090) + the parked dense-f16 decode GEMV lever

**Status:** Open — residue pointer filed 2026-11-05 when the parent lanes
closed (1150 + 1170; Claude verdict round 1 AGREE on the closure batch).
Two items the lane still owes, both priced/deferred, neither blocking a
landed surface. **Item 2's owed cells ALL LANDED 2026-11-08 ([Bench 956](../.benchmarks/956_i1177_owed_cells_attribution_ceiling_engagement_instability.md))** — the 6× was
stale-posture artifact (real packed-vs-f16 ctx gap 1.32×, stable n=5);
ceiling measured 700–750 GB/s (87–91% of the assumed 819); engagement
Δ+15,872/cell; walk byte-identical across 4 postures. Lever stays retired
(1184 finding + measured confirmation). Item 1 (4090 parity cell) remains.
**Lease closed:** zed-754da340@amo, 1794174540, was expires 2026-11-10T15:00:00+00:00 — holder session gone from amo's live-thread list (probe `01a1beaa` parked, no wake); under it Item 2's owed cells landed (Bench 956) and the lever closed — nothing left under that lease. **NOT re-claimed (2026-11-09, zed-15be441a@amo):** Item 1 is unblocked and ready — **the 4090 gate is OPEN** (verified 2026-11-09 ~05:1x +07 via Tailscale: 1% util, 731 MiB desktop-only, zero CUDA compute consumers; the filing-time "3 contenders" state is gone). Next executor claims per the lease law and runs the cell: build the cudarc forward at the league env set on the 4090, fixed walk on both backends, FNV/argmax pins registered here — **pre-register the f16-activation band from the f32 kernel-order precedent (`katgpt-rs closure/bridge_certified`) BEFORE the run, never post-hoc**.

Lease note (prior): Item 2 price cell on amo Metal; Item 1 stays 4090-queue-gated.
**Claimed:** zed-15be441a@amo, 1794201411, expires 2026-11-11T05:30:00Z
(scope: Item 1 — the cudarc walk-parity bin, the band pre-registration, the Metal+CUDA 48-token prose walk at ctx 2048, FNV/argmax pins registered here; the model is the bench-946 Q4_K_M, expected present on the 4090 at the probe bin's own F:/models default — sha256 verified against the amo copy before any run)

### Item 1 execution — the pre-registered band (written 2026-11-09, BEFORE any run; zed-15be441a@amo)

The parity cell: the SAME qwen3.8-27b-Q4_K_M GGUF on both backends, the SAME fixed walk — the prose family (the harness default prompt, "The harbor town woke slowly under a grey sky,"), 48 greedy tokens, ctx 2048 on BOTH sides (the 4090's 24 GB cannot hold f32 KV at the Metal-default ctx 24576 beside the 16.8 GB weights; ctx is allocation-only, never numerics — and both sides run the SAME value so the harnesses are symmetric). CUDA side = the cudarc twin at its DEFAULT env posture (all-fused arms — the twin has no league env set; the RIIR_QWEN38_* league envs are Metal-side wgpu arms); Metal side = the bench-946 walk posture (`--features qwen38_wgpu,metal_tensor_gemm`, bare default = the promoted 1176 league posture).

1. **Primary pin: EXACT.** The greedy-argmax walk id sequence matches byte-for-byte across backends (registered as the FNV-1a of the id sequence + the per-step argmax agreement). Precedent for expecting exactness: the bench-946 Metal retention cell read the lossy league arms 272/272 walk-identical vs the bit-exact incumbent at pp ≤ 675 — the lossy-arm numerics delta sits BELOW the argmax decision floor on this model class; both backends read the SAME Q4_K_M weights (and the same Q6_K lm_head tail), so the remaining cross-backend delta is the reassociation/fma-contraction class.
2. **The pre-registered divergence band (exercised ONLY on a flip):** a walk divergence is adjudicated in-band iff, at the flip step, the two backends' top-1 logit gap is within **|logit₁ − logit₂| ≤ 1e-2 (f32)** — the near-tie band, one order above the reassociation-class logit noise this twin itself accepts (the T1.8 note at the rows-B kernel: differing summation order is the accepted tolerance class, measured near-tie argmax flips ~1/512 on the Bonsai chunked path; Q4_K dequant + cross-backend fma contraction live in the same class). A flip OUTSIDE that band = **G1 FAIL** (real numerics break, not tie luck) — the cell records the failure and the lane files the follow-up, never a post-hoc band.
3. **On an in-band flip:** both continuations are recorded (each backend's own tail FNV from the flip position), the step position + both top-2 gaps are printed, and the pin registers the exact prefix length + the divergent-tie step as the verdict. The gap read uses each backend's capture-logits variant (cudarc `forward_token_capture_logits`; the Metal side's equivalent instrument) — the primary walk still runs `forward_token` on both sides, never the capture path.
4. **Instrument (new, both-output-format-preserving):** a small cudarc walk bin printing the SAME `[walk] ids: [...]` line as the Metal bench (its first run adds no Metal-bin changes — Metal's existing harness is used verbatim at `--ctx 2048 --walk-tokens 48`). FNV-1a over the ids is computed offline from the printed lines on both sides, registered in this issue.

## Item 1 — the cross-backend bit-parity G1 cell (from Issue 1150's acceptance gates)

⟲ **PROGRESS 2026-11-09 13:4x +07 (zed-15be441a@amo) — transfer healthy, dbirks shortcut KILLED with byte-level proof:**
- **dbirks file is NOT a valid parity arm** (probe 13:2x +07): GGUF header parse both files — same 866 tensors, same dims/types/offsets, tokenizer arrays hash-identical, BUT **438 of 866 tensor digests differ** (per-tensor SHA256 over the full data region, both sides; ranges from the shared offsets, 16.8 GB hashed per side). The 512-B delta is metadata only (7 extra provenance kvs in dbirks: base_model/license/tags/finetune/languages; header 530 B longer; `add_bos_token` present=False in amo, absent in dbirks — unread by riir-infer either way). Verdict: dbirks Q4_K_M is a **different quantization run** — the walk on it would have produced a non-comparable (and if ids matched, falsely reassuring) result. Digest tables `/tmp/i1177_dbirks/` (transient; the verdict line is the record). The exact-copy transfer was already the right call — now proven, not assumed.
- **Transfer health** (13:26 +07): chunks part_00+part_01 landed (2/16), part_02 shipping; measured rate ~1 GiB/15.4 min → **ETA ~17:00 +07**, inside the job's 9 h deadline; 18:30 timed wake armed as insurance. LAN 192.168.1.33 is stale — the 4090 now answers ARP at **192.168.1.36** but port 22 is firewall-dead on the LAN side ("Host is down") — Tailscale remains the only transfer route. Stale 677 MB `amo-copy.gguf` leftover from the naive-scp era sits on F: (delete failed while locked at script start; the final `copy /b` concat overwrites it — no action).
- **bmo outbox (delivered by hand 13:3x +07 — bmo's tailscale down, mail parked):** the i1167 rungc A/B (job 5ea19300, Nov 8) — variant 560.53 vs incumbent 565.20 ms/frame mean (−0.83%, bit-identity anchors present) addressed to the retired zed-754da340 session; 1167's lease folded complete in the same window (`bb40729`) — no open claim. i1117 128k run (e5922317) started 12:37 +07, still running.

⟲ **PROGRESS 2026-11-09 (zed-15be441a@amo, claim epoch 1794201411) — Metal leg RUN, CUDA leg staged:**
- **Band pre-registered** (`c9603b9`, before any run — the block above).
- **CUDA walk bin landed** (`ff5355a`): `qwen38_cudarc_walk` — the Metal bench's walk verbatim (same default prompt, same tokenizer, sequential per-token prefill (route ≡ multi-token per the 1150 route G1, bench 938), same walk loop, same `[walk] ids` line). Built clean on the 4090 (`cargo build --release --features ternary_gemv_cuda_raw`, 2m38s).
- **Metal leg DONE 12:2x +07** (amo, quiet, preflight PASSED — `power=AC Power load=3.80 powermode=0(auto)`; `/tmp/i1179_target` release, `--features qwen38_wgpu,metal_tensor_gemm`, bare default = the 1176 league posture — banner confirmed all four arms + relaxed; ctx 2048, 48 tokens, default prose prompt 45→10 tok):
  `ids: [279, 2993, 421, 18378, 10849, 694, 12238, 1132, 264, 38936, 35101, 13, 561, 3594, 82738, 314, 11679, 11, 31694, 11, 321, 279, 36334, 61306, 314, 888, 49371, 1016, 13834, 494, 279, 2981, 71072, 13, 1049, 557, 279, 2993, 314, 6353, 421, 1801, 1208, 3166, 440, 7193, 11, 430]` — **FNV-1a (LE-u32) = `0x814125045f90b9`**; text: " the kind that promised rain but delivered only a damp chill. The air smelled of salt, diesel, and the faint sweetness of overripe peaches from the market stalls. It was the kind of morning that made people move with purpose, as"
- **Model pin (amo copy)**: sha256 `e00082f779fa385cee8c68a3ec8833a75778cc87272240b942f74e0b8243e520` (16,810,714,336 B). The 4090's own dbirks file DIFFERS (16,810,714,848 B — 512 B metadata delta, different bytes = different file) → the amo copy ships to the 4090: chunked scp (16 × 1 GiB, per-chunk skip-resume, Tailscale ~1 MB/s — LAN-direct amo↔4090 is ARP-dead this posture, both sides reach the router but not each other), ETA ~5-6 h; concat + size-check on arrival, then Get-FileHash vs the amo pin.
- **CUDA leg queued**: 4090 rig job `01a1bed7-47f8-7df0-b54d-cac7fd8e199a` (`scripts/i1177_4090/run_walk_leg.ps1` — waits for the file size, hashes, runs the walk at ctx 2048 × 48, notify armed, 9 h deadline).
- **Remaining**: the CUDA leg's ids → compare vs the Metal pin → register BOTH FNVs + the verdict here (+ the bench record, next free number at write time). On a flip: the pre-registered band adjudicates (capture-logits re-run both sides at the flip step).

Issue 1150's G1 gate: Metal forward vs `qwen38_dense_cudarc.rs` (the
~7k-line CUDA twin, the bit-parity oracle) on the SAME qwen3.8-27B
Q4_K_M GGUF — FNV/argmax pins over a fixed token walk, tolerance band
set BEFORE the run (f32 kernel-order classes get the certified-precedent
treatment, never post-hoc bands). **Never run**: the lane's identity
evidence to date is Metal-vs-Metal (benches 938–946 walks digit-exact
within one backend) + the league demote byte-identity (bench 946). The
4090 is the only cudarc host and was BUSY (3 contenders) at filing.
Execute when the 4090 queue frees: build the cudarc forward at the
league env set, run the fixed walk on both backends, register the pins
(FNV + argmax) in this issue. Refusals/classes: the f16-activation
rounding order differs by construction across backends — pre-register
the band from the f32-class kernel-order precedent
(`katgpt-rs closure/bridge_certified`), never post-hoc.

## Item 2 — the parked dense-f16 decode GEMV lever (from Issue 1170 lever 2)

Decode on the dense-f16 posture reads ~1.9–22 tok/s class vs a ~15 tok/s
bandwidth bound per the 1150 audit framing (54 GB f16 at ~800 GB/s); the
`gemv_dense_f16` kernel is one simdgroup-per-row with strided reads,
while the Q6K tail's packed GEMV (bench 939) already proved the
saturating pattern (per-lane partials + one reduction). **Explicitly
parked non-serving in 1170**: the packed posture serves decode; this
lever matters only if the dense posture ever becomes the serving decode
posture or for parity rows. Price before any kernel work: the expected
ceiling is bandwidth-class (~15 tok/s), so the lever buys ~7× on a
non-serving surface — lead with a price cell, never the kernel.

### Item 2 price cell — FIRST READ + verdict round 1 REVISE (2026-11-05, zed-83e79a9e@amo)

**Status: PARKED pending attribution — the measurements below are real
but the code-vs-box split, the ceiling, and correctness proof are OWED
(Claude verdict round 1 REVISE, session 10bde1fe). Neither "~1.3×" nor
"79% of bound" is claimed.**

Cell ran on amo (M5 Ultra 96 GB, quiet, AC, exclusive GPU, co-tenant
guard passed), bench `qwen38_wgpu_bench --decode 32 --decode-ctx 2048`,
HEAD `d85ee80`+, 2 process runs per arm:

| cell | A: dense-f16 league default (`gemv_dense_f16`) | B: packed incumbent (`DENSE_MM2D=0 ATTN_TENSOR=0 F16ACT=0`) |
|---|---|---|
| raw decode 32-step | 11.67 / 11.67 tok/s | 8.76 / 8.82 tok/s |
| decode-ctx 2048 | 10.02 / 10.03 tok/s | **5.00 / 1.23 tok/s (UNADJUDICATED, n=2)** |
| load | 33.6 s (54 GB f16 materialization) | 9.9–11.1 s (16.8 GB packed) |

Witnesses: `[Issue 1150] dense-f16 matmul2d arm armed` + `[Issue 1170]
half-B armed` + `Q6_K packed tail armed (~2.1 GB resident)` (A) /
`[Issue 1176] DEMOTED` (B). Full logs `/tmp/i1177_price/` (transient).

**Corrections the verdict AGREED are recordable now:**
1. **The i1150 record's "~52 tok/s incumbent (issue-1153 cells)" is a
   mis-citation**: issue 1153 = Plan 623 Phase 3 (Bench 929) = the
   Bonsai ternary 2-box DIST decode row ("ours 2-box s1 52.28–52.59
   tok/s BEATS llama.cpp-RPC 2-box 48.6", amo⇄bmo TB wire, ~7 GB/token
   PQ2_0) — never the qwen3.8 packed single-box Metal decode (16.8
   GB/token). This cell's packed control reads 8.8 raw on amo.
2. **"The f16 posture must never serve decode" is stale post-1176**: the
   league default IS the dense posture (the demote interlocks prove it —
   `DENSE_MM2D=0` alone refuses: ATTN_TENSOR and F16ACT host on the
   dense context), so `gemv_dense_f16` decode is the serving Metal
   decode posture at HEAD.
3. **The 1.86/1.23 baseline is in doubt**: the decode path changed
   post-record (`27edb75`/`5ab9a51` 11-04 — the lm_head 10.7 GB f32
   tail → Q6_K packed GEMV ~2.1 GB + the `forward_token` decode loop;
   `d19223c` 11-05 made lmhead+f16act league default), and `5923713`
   resolved the 10-29 bimodality as a POSTURE split — the 1.86/1.23
   pair may not even be one posture. Co-tenancy cannot yet be credited
   with the 6×.

**OWED before any verdict or lever lane (verdict round 1 REVISE) — ALL LANDED 2026-11-08, [Bench 956](../.benchmarks/956_i1177_owed_cells_attribution_ceiling_engagement_instability.md):**
- [x] Attribution cell: A2 (`RIIR_QWEN38_LMHEAD_Q6K=0`) + A3 (also
  `F16ACT=0`) × 2 legs each — raw decode arm-insensitive (±0.4);
  ctx-2048: Q6K tail +1.2 tok/s, F16ACT ~0 at decode (prefill-only
  variant; engagement Δ identical). A1 baseline re-run at today's HEAD:
  12.06/12.11 raw, 10.33/10.37 ctx.
- [x] Measured ceiling: `examples/gemv_read_bw.rs` — the ENGAGED decode
  kernel over a 2 GiB shape, 3 processes: 715.5 / 748.5 / 699.4 GB/s
  median = **700–750 GB/s measured**, 87–91% of the assumed 819; the
  1184 finding's derived ~720 lands inside the band → f16 decode runs
  AT its kernel-class ceiling, no craft headroom.
- [x] Correctness proof: `[gemv-engagement]` instrument in the bench bin
  (dispatch_count delta per cell) — **Δ+15,872 per 32-step cell =
  496 GEMVs/step**, 10/10 legs; greedy-walk ids BYTE-IDENTICAL across
  A1/A2/A3/B1 (64 tokens, 4 postures).
- [x] Packed ctx-2048 instability: REFUTED at n=5 — 7.85/7.92/7.86/7.85/
  7.86 (spread 0.9%); the 11-05 "5.00/1.23" pair is stale-posture class
  (predates `d19223c`'s demote interlocks), never bimodality. Real
  packed-vs-f16 ctx gap = 1.32×; B raw 9.0 corroborates the 1184
  discriminator band (packedOFF ≈ 8.6-9).
- The re-priced lever lane stays CLOSED (the 1184 posture finding retired
  it at the byte-class level; §5 of the bench confirms at the measured
  level). Residue = Item 1 only (4090-queue-gated).

## Provenance

- Parent lanes closed 2026-11-05: riir-infer Issue 1150 (the dense
  forward arc, benches i1150/938–946), riir-infer Issue 1170 (lever 1 =
  half-B, bench 941).
- Claude verdict round 1 AGREE (2026-11-05): one residue issue covering
  both, always cited as "riir-infer Issue 1177" (riir-ai allocated its
  own 1177 the same day — always disambiguate cross-repo).
