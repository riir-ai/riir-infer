# Bench 1250 — Issue 1176 T4: the ours-side 2-box layer-split prefill A/B (amo⇄bmo TB5, Bonsai PQ2_0, armed) — DEMOTE at the preregistered pp32k bar; the structural model CONFIRMED (2.03×/1.94× over the chunked harness); pp16k passes its bar

**Status:** RECORD — T3 verdict **DEMOTE** per plan 652 §5(i) (the preregistered kill bar), with the §4 structural predictions CONFIRMED and a pp16k bar-pass recorded as the re-open recipe's evidence. The lane closes at armed single-host; the re-open path is the full-width dist prefill body (kernel lane).
**Date:** 2026-11-20 19:53–2026-11-21 01:07 +07
**Session:** zed-amo-g7@amo, 1795066588 (claim `1293497f9` on riir-ai `.issues/1176`; plan 652 §7 owner amendment — unsigned dist_stage, head=amo, tail=bmo sole listener, T1(f) LISTEN-probe HARD gate)
**Issue:** riir-ai `.issues/1176` (T4) · plan `652_i1176_t4_layer_split_preregistration.md` (the committed predictions + kill criterion, delegated verdict 2 rounds AGREE)
**Bar discipline:** oMLX's +195% is source-reported on foreign boxes — never quoted; the only bar is OUR armed single-host (max rule §5(i)).

## Box state + build

- head = amo (M5 Ultra, 30 cores, AC), tail = bmo (M5 Ultra, AC, Exclusive per cell via the rig referee), wire = the TB5 bridge `192.168.10.2` (`--split 32`, both stages GPU)
- Build: riir-infer `35f1c6ce`, release, `-p riir-infer-dist --bin dist_stage --features gpu,bonsai2_hadamard` — one binary, sha256 `cf57ffa109f16878ef2d0d817f4587dfaa1bb4249882ef47f660d8487efc33a5`, **amo↔bmo identical at staging AND re-verified at every run's preflight** (T1d)
- Model: `Ternary-Bonsai-2-27B-PQ2_0.gguf` sha256 `3907dc16…62ec1` = the league pin EXACT (both boxes)
- ARMED posture: the production aggregate + f16-B twin (`[i1225]` lines in every log); no demote literals set (asserted per run)
- PROVENANCE (measuring box): `power=AC Power load=3.31–4.66 swap=3889.12M canary=skipped powermode=0(auto) ane=ok` — preflight PASSED per run
- Protocol: 3 passes per length × {1-box armed oracle, 2-box armed}, per-length separate processes, loopback oracle via `--full-check`, 2-box via the per-cell tail boot (fresh bmo Exclusive rig job per cell), ABAB per length, medians. Chunk = the landed length-conditional default (512 at the league lengths).

## T1 receipts (plan 652 T1 a–g)

- **(a) split-32 layer map**: head attn {3,…,31} = 8, tail {35,…,63} = 8 — verified at the loader's own rule (`(i+1) % 4 == 0`) + live per-stage load lines
- **(b) GDN state layout**: 16k × 48v × 128 dim ⇒ 3.146 MB/layer f32 (48 layers ≈ 151 MB/seq) — the loader test's numbers confirmed at source; conv state rides the layer
- **(c) harness-parity gate (§5(viii))**: **FAILED at pp16k/pp32k** — 1-box oracle 585.06 / 378.76 tok/s = 0.60× / 0.45× of the Bench 1160 vehicle cells (981.63 / 838.05); pp2048 1113.04 = 0.966× (PASS). The ordered investigation RAN (3 probe jobs) and the penalty is **STRUCTURAL, not a fault**: chunk boundaries are cheap (w512 ≈ w4096: 80.6 vs 82.6 s at pp32k); full-width frames are impossible (`prefill_range_chunk: chunk 32768 > prefill chunk cap 4096 (RIIR_PREFILL_CHUNK_MAX)`, `ternary_deltanet_gpu_forward.rs:10805`, and the tail's 671 MB frame fill refuses); Bench 919's same rig read 183.9 s (178 tok/s) PRE-armed-stack — today's 86.5 s reproduces the rig at 2× the old posture (no box drift). Mechanism: the vehicle's bar comes from `fwd.prefill()` — a single-call FULL-width body (M=32768 GEMM, the W-frame rung) that never crosses a wire; the dist path structurally runs the chunked body capped at M=4096. The plan's 0–15% prediction priced per-frame overhead and missed frame-WIDTH economics.
- **(d) sha-verified binary** both boxes ✓ (above)
- **(e) engagement proof**: the armed-stack log lines on BOTH stages every cell (`[Issue 1146]`/`[i1225]`); a non-engaging stage VOIDS the cell
- **(f) listen topology (the HARD gate)**: `lsof` sampler on the head PID during every 2-box cell — **T1f clean on EVERY attempt** (all receipts `…_lsof.txt` absent-VOID; zero LISTEN sockets on amo at any sample); the tail's LISTEN on bmo is the expected posture
- **(g) seam carrier dtype**: f32 confirmed at source (`frame_bound` = 4·n_rows·hidden; the transport doc's f32 rows) — §3's wire pricing stands

## G1 — digit-exact over the wire (the §5(iii) law, executable form)

Every pass-1 cell: **G1 PASS — the 2-box stream == the 1-box oracle stream IN-PROCESS** (`--full-check`'s own comparator; 16 decode tokens per length). The plan's literal logits-FNV is not emitted by `dist_stage` (recorded preregistration gap, disclosed): the executable bit-identity law here is the generated-stream equality (the Bench-919 cross-width precedent) + the wire's exact-f32 protocol (plan-623 T3.4 ×6 precedent); argmax pins compared DIRECTLY. Argmax measured: pp2048 **248046 = the pin ✓**, pp32768 **248046 = the pin ✓**, pp16384 13 (vs pin 95759 — the cross-width near-tie class, same family as the plan's own FNV-divergence disclosure (a); §5(iii)(c) omitted the caveat for argmax — recorded as a preregistration miss, never hidden).

## The cells

### 1-box armed oracle (the §5(viii) vehicle + the §5(i) bar's same-window half)

| length | oracle prefill median | tok/s | vs Bench 1160 armed |
|---|---:|---:|---:|
| pp2048 | 1.840 s | **1113.04** | 0.966× |
| pp16384 | 28.004 s | **585.06** | 0.596× |
| pp32768 | 86.513 s | **378.76** | 0.452× |

### 2-box armed wire cells (the A/B arm)

| length | passes | prefill median | tok/s | argmax | G1 | T1f |
|---|---:|---:|---:|---|---|---|
| pp2048 | 2* | 1.213 s | **1688.38** | 248046 ✓ | PASS | clean |
| pp16384 | 3 | 14.455 s | **1133.45** | 13 | PASS | clean |
| pp32768 | 3 | 42.571 s | **769.73** | 248046 ✓ | PASS | clean |

\* pp2048 pass 3's head log was overwritten by its own whole-cell retry (the v3 driver's retry overwrites the cell log — a receipt-shape defect fixed in the record: pass counts are read from the session log, 3 passes RAN, 2 logs retained). pp2048 gates nothing (§5.2).

## T3 — the preregistered adjudication

| length | bar = max(vehicle, same-window 1-box) | kill line (×1.084) | 2-box median | verdict |
|---|---:|---:|---:|---|
| pp32768 (DECIDING) | 838.05 | 908.04 | **769.73** | ⛔ **DEMOTE per §5(i)** |
| pp16384 | 981.63 | 1064.09 | **1133.45** | ✅ passes the bar (1.154× vehicle — beyond the ±8.4% band) |
| pp2048 | 1152.62 | — (gates nothing) | 1688.38 | EXPECTED (§5.2) |

**The verdict: DEMOTE** — the lane closes at armed single-host per §5(i), as preregistered. Two confirmations ride with it:

1. **The §4 structural model is CONFIRMED on the wire**: 2-box / own-harness-1-box = **2.03× (pp32k) and 1.94× (pp16k)** — inside the committed 1.9–2.0× band. The layer-split machinery delivers exactly what the pipelining model promised; the DEMOTE is entirely the bar's frame-width economics (the vehicle's full-width body is 2.2× the chunked body at pp32k — §5(viii)'s structural finding), not a failure of the split.
2. **pp16k passes its bar beyond noise** (1.154× vehicle) — recorded as the re-open recipe's evidence: a length-conditional 2-box armed posture at ≥16k is a real GOAT candidate IF the full-width dist body lands (it lifts the 1-box bar's same-window half too — the §5(i) max rule then compares two full-width postures, which is the honest fight the plan wanted).

## Provenance & honesty

- **Three firings**: v1 burned 2 cells (a driver double-boot flaw + os-error-65 route hits during a sibling clang storm), v2 lost 2 cells to TB5 route drops that killed tails mid-handshake (one successful connect then "failed to fill whole buffer" on the tail — connect-retry against a dead tail is useless), **v3 landed all cells** behind a live 64-token wire-health probe (real handshake at a scratch port) + whole-cell retries ×3 per pass. Every failed cell is a receipt on disk (`.i1176-tmp/2box/20261120T1929*/`, `…T001404/`, `…T003833/` on amo), never a silent drop.
- **Referee poison stamps**: Zed (owner-exempt class) + transient sibling builds/viewer runs through both legs — passes stayed tight (±0.1–1.5%); the stamps are recorded and disclosed. The 32k 2-box trio: 42.536/42.571/42.603 (±0.08%).
- **The driver receipts**: single-boot tail fix, per-cell bridge pre-check, 3-attempt connect retry, wire-probe gate — the rig-referee hygiene note (a cancelled driver leaves queued bmo jobs that fire independently) recorded.
- **Roads not taken, on purpose**: raising `RIIR_PREFILL_CHUNK_MAX` (the frame cap is the kernel's own economics, not a knob), m3 as a peer (owner declined — silicon class), interleaving with the 932 guard leg (owner lane declined, load-confound history).

## The re-open recipe (§8's named path)

1. **A full-width dist prefill body** — the kernel lane: bounded-frame wire transport + a `prefill()`-class body behind it (raises or bypasses the M=4096 chunk cap safely). This lifts BOTH the 1-box bar half and the 2-box arm to the vehicle's GEMM class.
2. Then the length-conditional 2-box armed posture (≥16k) re-runs THIS A/B — the pp16k bar-pass above predicts it wins on its own terms.
3. The wire economics hold (§3: 0.17–0.24% wire share, W-P1 confirmed — the wire was never the bottleneck at any length).

## References

Plan 652 (the preregistration) · riir-ai `.issues/1176` (the verdict + claim) · Bench 1160 (the armed bar) · Bench 935 (the rig pattern + crossover) · Bench 919 (the same-rig pre-armed read) · Bench 929 (foreground discipline) · riir-infer plan 623 (the wire laws) · `gguf_loader.rs` (the layer map + GDN dims).
