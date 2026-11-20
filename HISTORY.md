## 2026-11-20 — Issue 1237 CLOSED: decision-core census over laya (ACE) — Bench 1247 (katgpt-rs-93@amo)

Consumes katgpt-core `relevance_core` (katgpt-rs Plan 640).
Multilingual is CLAIMED at r = 2 on the full 2302-forward eval (a small, sign-consistent edge). Side finding: Issue 1239 (`EncodedQuestion.hidden` was the post-head residual), fixed `954b27be`.…

## 2026-11-20 — Issue 1239 CLOSED: `EncodedQuestion.hidden` is the encoder output again (katgpt-rs-93@amo)

Found by the Issue 1237 R0 census: `encode_question` downloaded `h` after the head's in-place `layers_forward_rows`, so `hidden` was the post-head residual (max |Δ| 1281.6 english / 1311.8 typed / 899.3 multilingual vs the encoder-only seam)…

## 2026-11-19 — Issue 1238 CLOSED: frame-size-keyed GPU caches bucketed (katgpt-rs-93@amo)

The i1226 F2 verdict's non-blocking condition (d).

## 2026-11-19 — Issue 1226 CLOSED (file removed; this row is the note): the f16-B lane follow-ups — F1 VALID (Bench 1234), F2 PROMOTED default-ON in the dist lane (`07ad293b`), F3 done (katgpt-rs-ec@amo, 1795103700)

F2 — producer-writes-f16. The shadow lane and G1 landed 2026-11-17 (zed-ee646c82: `c0054df2` + `67ef17db`, G1 3/3 bit-identical per kernel).
The pp32k half was flagged poisoned (a busy editor process) and kept by the rule recorded at poison time. Tap retention: satisfied by construction, since C's logits equal B's. Delegated-owner…

## 2026-11-19 — Issue 1234 CLOSED (file removed; this row is the note): the shared-clone `reset --hard` that wiped the F2 WIP — F2 re-authored past it and landed

The incident (2026-11-17, zed-b2e60a87): a `git reset --hard origin/develop` in the shared clone wiped ~800 lines of the F2 lane's unstaged edits. There was no git-object or snapshot recovery.

## 2026-11-19 — Issue 1231 CLOSED (file removed; this row is the note): the pipelined resident snapshot-future harvest — T1+T2+T3 all landed, the delegated verdict AGREE (amo-idle-hk, 1795081500)

The lane arc: T1 substrate+consumer+cells (riir-infer `f881fb87` + riir-ai `387cb84be`); T2 the wall cell at the pre-registered bars (riir-infer Bench 1243, `9663ce79` + logs) — ADOPT BAR MET: within-cell ABBA -0.351 ms/tok (14.98 to 14.63, >= the…

## 2026-11-17 — the lens lane (Research 666 / katgpt-rs Issue 999) CLOSED DEMOTED: the pre-registered T5 gates FAILED on the real-corpus fit — `lens` stays opt-in forever, promotion dead, the laya "headful lens variant" is REFUTED-BY-VERDICT (zed-amo-i999verdict@amo, 1794930920)

The lane arc: T2a–T2d op/layer/stack maps landed `16d50c46`/`11fb57de`/`c85210fa`/`d4cf53c8` (the T2d parity pin found + fixed T2c's conv-tap-order flip), T3 harness `0f796e6c` + the 40-doc real-corpus fit (Qwen3.5-0.8B f32-dequant, 50 734 s…

## 2026-11-17 — Issue 1235 CLOSED CONFIRMATORY: the rec kernel's q-defer fusion arm is UNDER the bar at every shape (Bench 1239; 0.977× at the serving p=512) — the compiler already schedules the q-butterfly off the chain; planes=8 stands, both named kernel-work classes now measured out (amo-zed, 1794904830)

The lane executed its own pre-registration end to end in one session: T1 landed (`d5f7e6b3` — the comptime q_defer specialization on the MR kernel, PROMOTE-ONLY `RIIR_REC_QDEFER`, the arm's own vacuous counter, the G1 sweep extended to every ladder…
En-route: the F2 incident got a NEW verified finding (`8766bfcb` — HEAD's no-prefill_f16b lib posture compile-red at develop, 4×E0599, the wipe's +125-line loss class; the recovery's re-application…

## 2026-11-17 — Issue 1233 (gdn_rec chunked-form prefill kernel) CLOSED CONFIRMATORY at T1: the chunked class is measured-NOT-BIT-IDENTICAL — the issue's own G1 bar excludes it (amo-zed, 1794904830)

The lane closed in its own read-only design phase, on the repo's own records — no kernel was built, and none could clear the bar as scoped:; The premise error: "a Metal twin would be a NEW kernel family" — it would not.
The Metal serving posture's promoted prefill lane is the bit-exact multi-token dispatch (default-on 2026-08-20) under the FNV walk pin — swapping the numerics class on the DEFAULT serving path is an…

## 2026-11-17 — Issue 1214 + 1217 files removed (the hygiene pass): 1214's close claimed the removal but never executed it; 1217's resolution had no HISTORY row

1214 (the int_ref lane) closed 2026-11-17 in `395407ea` — G1 FULL PASS all four arms, G3 ΔPPL +0.067, the fusion promotion (plan 632), T4/T5 routed external-blocked — with the status line saying "Issue file removed this close" while the file stayed…

## 2026-11-17 — i1221 lane: T0–T3(b) swept (the Dory looped checkpoint consumed end to end) — the trained/dense loopability mirror pair + katgpt-rs Plan 304's cost_floor answered on real data (`4e6d5161` + `bc9fa3e0` + `c5de8e46` + `613916c4` + `0571def5`)

The consumer-lane sweep, one session per leg under the plan-636 pre-registration (read order d→a→b, cheapest-signal-first).
Verdict: loop-trained invariance confirmed — KL(full‖exit) collapses 13.8–24.1 → exactly 0 at the trained k=4, monotone 24/24 positions, KL(exit₁‖exit_d) grows 8→27 (the loop deepens while the…

## 2026-11-17 — Issue 1230 CLOSED (refuted-by-analysis) + Issue 1231 filed+T1 landed: the counter drain is fence-equivalent — the real lever is the pipelined snapshot-future harvest (`f881fb87` + `387cb84be`)

The 1230 mechanism died the good death: stress-testing the completion-counter premise against the queue semantics before writing any code showed it fence-equivalent in an in-order queue — a counter read's staging copy sits at the SAME FIFO position…
T2 (the wall cell at the pre-registered bars — adopt iff ≥ 0.3 ms/tok vs the same-cell cadence arm, quiet exclusive HARD premise) queued next; the serving env deliberately does not exist until a…

## 2026-11-17 — Issue 1218 CLOSED + removed: the fence-cadence lever adjudicated measured-negative (Bench 1229) — the ring depth is not the serializer, 1181's pool-bound finding confirmed; the completion-counter drain filed as Issue 1230

The T2 wall cell ran all five arms (K8 re-anchor + K32 armed, ABBA×3 each, stop-identity both postures, the mid-burst STOP_AT=65) via `.i1218cell/run_cell.sh` on amo — G1 walk identity PASS (identical FNV `121982e1b7878d05` both cadences), stop…
Rig incident disclosed: the job wrapper `b5247eb6` poison-looped ×131 while the script completed beside it (no job log ever written; the Exclusive slot freed itself, the sibling's 1226 walk was…

## 2026-11-17 — Issue 1227 CLOSED + removed: the fence F2 member-exemption drift (the 144-finding cry-wolf class)

`scripts/fence_gate.py`'s hand-typed F2 exemption (`infer_core|infer_gpu|infer_laya`) predated four member crates (ane 10-15, clef, audio 10-23, dist 10-27) — every develop run printed 144 UNDEFENDED findings for the repo's OWN leaves, invisible to…

## 2026-11-16 — Issue 1199 CLOSED + removed: the alternation tax adjudicated at real Bonsai-2 27B (Bench 1221) — rung (b) declined, the loopback-CPU overlap thesis refuted at this geometry

The T3 deciding cell ran on bmo Exclusive (job `01a1e210-aeb6`, release, PROVENANCE `power=AC load=2.20 powermode=0(auto)`; the first attempt `01a1e1cb` was the GGUF loader's fail-closed `prism.hadamard` refusal — the real arm requires…

## 2026-11-12 — Issue 1210 CLOSED + removed: the `rope.rs` module-header doc corrected to the rotate-half pairing the code actually uses

The header described the ADJACENT pairing (`q[2i]`, `q[2i+1]`) — a different rotation subgroup — while every rotation path in the file (`apply_rope_heads_precomputed`, `apply_inverse_rope_heads_precomputed`, `apply_rope_values`,…

## 2026-11-11 — Issue 1205 CLOSED + removed: the Issue-022 T1.4 residual PAID — the REAL parent-vs-quantized ΔS pair ran (Bench 969); the 7c5b5ef removal note's debt closed

Both prerequisites finally coexisted on one box: the laya capture half (`Encoder::forward_capture` + `examples/twt_laya_profile`) and a quantized laya-class checkpoint — the english q8 STORAGE artifact (converted this run by `laya-quant8`,…

## 2026-11-11 — Issue 1178 CLOSED + removed (hygiene close-out): the uzu decode-path distill lane is substantively complete — every task landed and recorded; the 8.1 GB `.raw/uzu` research clone finally removed per T0's own close-out promise

All five task rows were already DONE with landed records — nothing new was executed in this run: T0 Research 378 (`50b43d71`, riir-refine) + the T0(e) source-read resolution (zed-15be441a@amo, 2026-11-09, pin `1866d189`); T1 Bench 950 (instrument…

## 2026-11-11 — Issue 1201 T2 radix half LANDED (`00902067`): the FlashStates multi-session manager over the named substrate — the ledger is the substrate's own, concurrent-fill recovers first-writer-wins, graph safety by construction

`flashstates_multisession.rs` (feature `flashstates_multi` = ["dep:katgpt-kv", "katgpt-kv/radix_prefix_cache"]; both deps workspace-table rows, BOUNDARY-allowlisted — katgpt-kv measurement-lane-capability posture): `FlashStatesManager` composes…

## 2026-11-11 — Issue 1200 T3 LANDED (`ef18a7c0` + en-route lint repairs `ef2865b4`): the fail-closed refusal half's real-boot confirmation on m3 Metal — over-envelope boots refuse at t=0, in-envelope boots decode clean

The league-GGUF boot pair through `flashmemory_decode_gate --metal --max-steps 1` (`Ternary-Bonsai-2-27B-PQ2_0.gguf`, ctx 32768, block 64): (a) ρ0.30 — the fm_state ctor refuses before any budget-arm decode step (max_seq = the GGUF context_length…

## 2026-11-11 — Issue 1201 T1 LANDED (`bab043b4`): the FlashStates multi-session cell pre-registered — plan 630 freezes trace/arms/metrics/bars before any run + the deterministic trace generator (token rows, BLAKE3-digested, 6 pins)

Plan [630](.plans/630_flashstates_multisession_cell_pre_registration.md) is the frozen cell: A0 = the AS-SHIPPED fresh-stream-per-request floor (the honest denominator — no lane composes per-session caches today); A0+ = per-session…
`.plans/.highwater` 628 → 630 (stale by one; 629 was landed uncounted).

## 2026-11-10 — Issue 1184 fold (hygiene): the delegated T2b price-cell run CLOSED 2026-11-08 but its file was left on disk — record folded into HISTORY, file removed (Bench 955; parent 1178's T2b closed, lease folded in-batch)

The i1178 T2b lanesliced-Q4 decode price cell executed by zed-amo-1184@amo on bmo (rig jobs `01a1bb89` VOID per the preregistered engagement gate — the `env $ENVSET` zsh non-splitting demote-literal defect, disclosed in-record — then `01a1bbbe` ALL…

## 2026-11-10 — Issue 1198 CLOSED + removed: the multi-stream overlap bar re-scoped to a pathology tripwire — Bench 908's ratio<0.85 measured the DEBUG posture; the release posture carries the CPU-stage alternation tax (fix `9b472d3`, debug completion `8452a8f`; the tax itself = issue 1199's optimization rung)

`multi_stream_g1::two_streams_overlap_serial_vs_interleaved` reds on amo at ratio 2.68–3.28 (bar wanted < 0.85), pre-existing at base `83d491b`, correctness asserts green everywhere.
Session: zed-34912c3b@amo, 1794237218 (finding + structural analysis + T1); re-scope landed `9b472d3` + `8452a8f`; closed zed-95e45495@amo, 1794259085 (bookkeeping + the verified re-run).

## 2026-11-09 — Issue 1197 CLOSED RESOLVED: the plain synth fixture's Metal-vs-CPU divergence is a token-determined spike train (nonlinearity cliffs on untrained weights) — no position growth, state map contractive, attention exonerated

The 1196 residue's premise ("divergence grows with position") was a sampling artifact — two points, one noise step and one spike step.

## 2026-11-09 — Issue 1196 CLOSED RESOLVED: the synth-kit ssm_a convention violation — the "Metal hd-128 NaN" was a fixture bug, every kernel exonerated (fix `c59e57f9`, T1 gate `a097a95d`)

The synth Bonsai-2 fixture's `ssm_a = +1.0` violated the Issue-594 loader convention (`a_log = −exp(A_log_raw)`, NEGATIVE): decay = exp(+softplus) ≥ 1 on EVERY lane, the recurrence state grew unboundedly (m3 census: decay_buf [1.0, 1.0, 1.01, 1.06,…

## 2026-11-09 — Issue 1192 CLOSED FIXED: the i1179 probe examples' manifest rows (the --all-targets green-zero/required-features trap class)

The ten `crates/riir-infer-gpu/examples/*_i1179.rs` probes had NO manifest rows — cargo auto-discovered and compiled them at EVERY posture, so `cargo clippy -p riir-infer-gpu --all-targets` failed E0432/E0433/E0601 (23 errors) at any posture…

# HISTORY.md — riir-infer

Compacted 2026-11-20: every heading kept verbatim, bodies <=2 lines of durable facts; full pre-compaction text in git history (pre-compaction HEAD 5a27ee2). Operational rules live in `AGENTS.md`.

## 2026-11-10 — Issue 1195 CLOSED RESOLVED: the root manifest's dev-profile debuginfo budget landed (`fb1c1f0`) — line-tables-only + deps debug=false; the A/B measured the daily-loop half; opt-3 dep codegen judged and declined

`[profile.dev] debug = "line-tables-only"` + `[profile.dev.package."*"] debug = false` + the `[profile.dev-full]` debugger escape hatch — the proven katgpt-rs shape (riir-ai .issues/723 Phase 1, incl. the reproduced counterintuitive trap:…

## 2026-11-09 — Bench 965: the 1183 T5 re-measure on the FIXED forward — untrained mean_j 1.000 → 2.565; the "worse than uzu" closure premise overturned (verdict revised in place, delegated-owner AGREE)

The EXACT 1183 T5 posture re-run on the sigmoid-fixed forward (G1-T5 PASS, blake3 pins on dspine `53d18516…` + gguf `ce14c955…`): mean_j 2.565 over 200 cycles (emitted 514; survival p1/p2/p3 = 0.68/0.38/0.205) — 2.0× uzu's trained 1.27 (Bench 954),…

## 2026-11-09 — Issue 1196 CLOSED + removed: the plain-numerics posture kept a bf16 staircase in the MLP gate/up gemvs (the training surrogate's "skips every bf16 boundary" contract violated at two ops)

`draft_block_impl`'s MLP called `gemv_bf16` (unconditional `bf16r(acc)`, no plain branch) while every other draft projection rides the plain-aware `gv` closure — the trainer's FD surface carried a staircase at exactly the two ops the distill…

## 2026-11-09 — Issue 1193 CLOSED + removed: the DSpine main-forward message gate shipped 1/(1 − e^z) (a hyperbola) since the oracle birth — unary-minus precedence, `-(z).exp()` binds as −(exp(z)) — fixed to the vendor's `mx.sigmoid` spelling; the piecewise golden contract's blind spot recorded

The bug: the main `draft_block_impl` inline loop (born `c8de365`, 1183 T2) computed `1/(1 + (-(qv+sv+b).exp()))` = `1/(1 − e^z)` — pole at z=0, negative for z>0 — while the vendor pin computes `mx.sigmoid` (`.raw/dspine` bridge.py:143).
NOT yet re-run (needs the checkpoint + the MLX lane): the 1183 T5 cells, the Bench 959 shape-walk re-measure, the main-path golden piece. Session zed-15be441a@amo, 1794240000.

## 2026-11-09 — Issue 1180 CLOSED + removed: the sink-audit census follow-ups complete — T1 ungated arm (Bench 953), T2a SinkFree neutral half PASS (Bench 963), T2b SinkHeavy protection NEUTRAL under maxrecent / load-bearing under random (Bench 961); the pin stays 4, no promotion, no demotion

T1 (Bench 953) — the real ungated G2 arm via the dense-body census through the gemma-2 f16 CPU forward: `gemma-2-2b-it` f16 (no attention output gate) reads 0.4071 → SinkHeavy (inside the paper's ungated 0.297–0.467 cluster) while the gated…

## 2026-11-09 — Issue 1177 CLOSED + removed: the dense-lane residues complete — Item 2's owed cells (Bench 956) + Item 1's cross-backend bit-parity G1 EXACT (Bench 964)

Item 2 (the dense-f16 decode GEMV lever) — the verdict-round-1 OWED cells landed 2026-11-08 ([Bench 956](.benchmarks/956_i1177_owed_cells_attribution_ceiling_engagement_instability.md), zed-754da340@amo): the "6×" was a stale-posture artifact (real…

## 2026-11-09 — Issue 1187 CLOSED + removed: the needle-gate mid-run report merge works again — de_nan_vec on the delta fields + mean_delta de_nan_f32, fix `f3385a8`

The Bench-958 smoke's per-arm "merge failed to parse" lines were this pre-existing defect firing: `run_arm` returns rows with `delta_nll`/`mean_delta` still NaN (the run-end fill happens later), serde_json writes NaN as `null`, and the next…

## 2026-11-09 — Issue 1185 CLOSED + removed: the K/V width-census arc complete — capture (T1) + Bonsai census (T2, Bench 957) + the sink-window needle gate (T3, Bench 958: every width-derived ceiling REFUSED 8/8, needle surv 0.000 at the ε-grid medians) + the readout (T4: keep the pin; the width budgets are a census TABLE, not a policy); the percentile-hop disclosure arm shipped with it (tail_support on every read); feat commit `2830499`

## 2026-11-06 — Issues 1168 + 1108 CLOSED + removed: the decode host-sync triage (arc delivered via plan 626) + the Locals/STCON inventory (all tasks done, Track A parked on its consumer trigger)

1168 (decode host-sync lever triage) — T1/T1b probes DONE 11-03 (`tests/issue1168_readback_floor_probe.rs` + riir-ai `issue1168_decode_loop_decomposition.rs`): readback itself 0.17–0.25 ms, the mass is enqueue-front + GPU-serialized read under the…
- 1108 (Locals/STCON distill inventory, Research 014) — T0/T2/T3/T4 DONE (T2 persistence delta folded into `qwen38_prefix_cache.rs`'s serving-lane trigger; T3 riir-refine Research 315 +…

## 2026-11-05 — Issue 1164 CLOSED + removed: the T5 mixed-mode mask PROVEN at model level — Cell D re-cell reads fnv 0/6 (was 6/6 vacuous), anchor intact

The closure bar the fix `a9aeedd` (any → all predicate, 2026-11-04) named: re-run riir-ai bench 1157's Cell D through the same harness (`bench_1156_t6_league_cells`, name-filtered, one process, release, amo QUIET, preflight PROVENANCE quoted, runtime 3902…

## 2026-11-05 — Issues 1170 + 1171 + 1172 CLOSED + removed: the dense-lane prefill campaign's three landed lanes — half-B past the fork (bench 941), tensor-flash attention (942), the GDN-bucket census (943)

## 2026-11-05 — Issue 1150 CLOSED + removed: the Metal dense-qwen3.8 forward arc complete — fork-parity delivered, the league posture is the default, G2 table recorded (Benches i1150 + 938–946)

The arc's stale "IN PROGRESS" status line stopped at the bimodality question; every task was already done.

## 2026-11-05 — Issues 1170 + 1171 + 1172 CLOSED + removed: the dense-lane prefill campaign's three landed lanes — half-B past the fork (bench 941), tensor-flash attention (942), the GDN-bucket census (943)

1170 (dense-lane GEMM engine to MLX-class) — lever 1 LANDED (bench 941, zed-amo-i1170-halfb@amo): the staged-activation half×half matmul2d over f16 twins (`RIIR_QWEN38_DENSE_MM2D_F16ACT`, promote-only then) answered the G2 physics question YES at…
Bench 945's ablation matrix later REFUTED the per-section floor reading (marker-boundary instrumentation cost; free-running small kernels 0.0–0.1 ms/layer) with dated ⚠ rows — the record's census…

## 2026-11-05 — Issue 1176 CLOSED + removed: the league env set promoted to the METAL DEFAULT — bare boot 966 tok/s (was ~7), retention clean 272/272 (Bench 946)

The five-env league posture (`DENSE_MM2D` / `LMHEAD_Q6K` / `DENSE_MM2D_RELAXED` / `DENSE_MM2D_F16ACT` / `ATTN_TENSOR`) flipped DEMOTE-ONLY on macOS+metal_tensor_gemm (exact `=0` each restores the incumbent; promote-only `=1` elsewhere; dependent…
Record: `.benchmarks/946_issue1176_default_posture_promotion.md`.

## 2026-11-05 — Issue 1165 CLOSED + removed: the tau=0.85 envelope-sparse prefill promotion LANDED (Metal/wgpu, >=16K) — P1-P6 all done, riir-ai bench 1158 PASS

Every task was already `[x]` (the Status line was stale — the promotion landed 2026-11-02: P1+P2+P3 `657b9caf` (default-on >=16384, demote-only `RIIR_PREFILL_SPARSE=0` byte-identical, default tau 0.9->0.85, the embedded digest-pinned v2 manifest…

## 2026-11-04 — Issue 1175 CLOSED + removed: the zero-length-binding launcher audit — 2 live sites fixed, the fused-twin offset bug found, 1 cross-repo site fixed in riir-ai

Audit: 1102 `BufferArg::from_raw_parts` sites; exactly 2 literal-zero binds (both `RmsNormQkFusedCubeCL`'s unused slots — safe by accident, the kernel routes by `params[3]` and never length-reads).
Near-miss recorded: the brace surgery initially dropped two `r0 += rc;` loop advances — clippy's `while_immutable_condition` caught both before commit (the compile gate earns its keep).

## 2026-11-04 — Issue 1174 CLOSED + removed: the free-running ablation instrument landed — the wall is GEMM-owned (~61%), the census non-GEMM share was marker-boundary cost (Bench 945)

Instrument (`e59d59d`): `qwen38_ablate.rs` (27 launch-site classes, census-section aligned; all/csv parse, refuse-loud unknowns; 4 unit tests incl. the census-mapping drift pin) + 26 geometry-preserving skip gates in `qwen38_dense_cubecl.rs`…
Claude verdict round 1 REVISE (4 blocks) → all applied + confirmation run (`scan` 55.66 vs 54.41 reproduced; base identical) → round 2 AGREE, `0de2cc2`.

## 2026-11-04 — Issue 1173 CLOSED-NEGATIVE + removed: the fused norm+gating family measured sub-bar — the launch-floor lever REFUTED at e2e (Bench 944); the census's per-section walls are marker-boundary artifacts

Three batched fused kernels (F1 norm_ssm+z_gate, F2 norm+f16-twin, F3 residual+norm, incl. the F3 cross-layer arm) landed BIT-IDENTICAL (G1 `tests/qwen38_fused_norm_gating_g1.rs` 3/3 GPU-vs-GPU bitwise f32+f16+twin-less; e2e greedy walks…
Claude verdict round 1 REVISE → all four revisions applied → round 2 AGREE, closed `c9245db`. Record: `.benchmarks/944_issue1173_fused_norm_gating_ab.md`.

## 2026-11-04 — Issue 1169 CLOSED-NEGATIVE + removed: the weight-delta transport PoC's Phase-0 kill branch FIRED — real checkpoints are dense in byte space, no protocol gets built

T0.2 measured 3/3 real plan-452 checkpoint pairs: flipped-byte 92.7–94.4%, RLE/varint mask payload 1.58–1.60× zstd(full artifact) at pinned levels 3+19 — the mask is LARGER than the whole artifact; 0/3 on the pre-registered ≥2-sparse bar.

## 2026-11-02 — Issue 1166 CLOSED + removed: the tile-geometry residual answered NEGATIVE — the promoted ktile64 is the composed-neighborhood optimum (Bench 936)

The second-generation sweep (`tests/issue1166_tile_geometry_sweep.rs`, `d9f8d9d`/`6941f77`): 1113's winner crossed with the axes 1113 varied (ktile128; 64×256 and 128×128 AT K64; 8sg/2sg; K32 floor) — G1 worst_rel=0.000e0 all 7 configs; G2…
Record: Bench 936 + `i1166_tiles/sweep_run.log`. Session zed-amo-1166-tiles, epoch 1793712000 (claim `f97dcbc`).

## 2026-11-02 — Issue 1146 CLOSED + removed: the 1-box prefill throughput gap — 0.13–0.16× → 0.76×/0.93×/1.00× of llama.cpp (parity at pp32k) in one day-arc of measured, GOAT-gated rungs

Arc (every rung measured before promotion, per-flag discipline): root cause the dist `gpu` row omitted the prefill-lane features (`412c887`, dispatch arithmetic to the digit, Bench 914) → width re-curve 128→512 1.39–1.42× 1-box (919) → tensor…

## 2026-11-02 — Issue 1163 CLOSED + removed: the T4.4 regime capture wrote 0 rows at every posture — the post-warmup setter was a swallowed OnceLock no-op (found STATICALLY; the filed read_one race never ran)

Root cause: the dump runner warms up DISARMED → the first `regime_capture_armed()` seeds `CAPTURE_PATH` (`OnceLock<Option<PathBuf>>`) with None from the unset env → `set_regime_capture_path` fails forever (already initialized, `let _ =` swallowed)…

## 2026-11-01 — Issue 1162 CLOSED + removed: the lone `ternary_gemv` combo failed to compile (4 errors, pre-existing) — an ungated method whose targets + only caller are `ternary_gemm_batched`-gated

Repair `2c694cb`: `prefill_project_pair` gated `ternary_gemm_batched` (the cfg its targets `PrefillSite`/`prefill_project` and its only caller `prefill_tokens_chunk_once` already carry; the module gate in lib.rs stays untouched — gating the whole…

## 2026-11-01 — Issue 1161 CLOSED + removed: the 7 dead-code warnings at the cubecl_runtime-only posture (reflex's laya-riir-cubecl composition) — root cause: the callers' WHOLE MODULE is cfg'd out

`ternary_deltanet_gpu_forward` is gated `all(cubecl_runtime, ternary_gemv)` (lib.rs:696), so at cubecl_runtime-only every call site (REACH marks, dense_pair_wg_size, regime_capture_append) compiles to nothing while the callee modules compile —…

## 2026-10-31 — riir-ai Plan 607 follow-up: the promoted `RIIR_GDN_REC_MR` re-pinned through the formal league harness + the share-model corrected — 0.2714×/0.2792×/0.2732× (pp2048/16k/32k), FNV pins EXACT ×9, +7.7% window question closed (drift), rec share at the league chunk 5.09%/3.35% vs the serving-chunk census's 2.16%

Formal bout on bmo at tuple riir-ai `ded79a301` + this repo `6116c47` + katgpt-rs `2bbaf2640`, fork `73bfe6cc48`, verified-idle, canaries in-band: pp2048 289.69/1067.25, pp16k 254.04/909.87, pp32k 210.42/770.33 — the armed class (D4:…

## 2026-10-31 — Issue 1160 CLOSED + removed: soft-token feedback decode (LIFT/MoI class) — G0 COLLAPSE on Bonsai-2 PQ2_0

Pre-registered G0 fired: pooled median decode top-1 mass 0.976 ≥ 0.9 over 512 greedy Metal decode steps (arith/G1b headline class 0.999; the pilot's top-128 τ=1.5 soft token puts 0.985 of its weight on the argmax) ⇒ `E^T s` ≈ the argmax embedding…

## 2026-10-31 — riir-ai Plan 607 E1 LANDED:  promoted DEFAULT-ON at planes=8 (Metal/CubeCL prefill; the CUDA lane default-on since Bench 896) — riir-ai Plan 607 D4's armed-posture paired-window A/B on bmo at the m5ultra league shapes: +1.24%/+1.37%/+1.20% (pp2048/16k/32k, n=6/arm/shape), 18/18 armed passes above incumbent, manifest FNV birth pins EXACT under the armed kernel; delegated-owner verdict AGREE; env now DEMOTE-ONLY (0/off/false = incumbent byte-exact; garbage fails open to the DEFAULT arm); caveats recorded (e2e exceeds the +0.38% pre-registered ceiling — the A2 census share under-counts; M3-box e2e unmeasured per the box-state binding)

## 2026-10-30 — Issue 1157 CLOSED + removed: regime_gate inline sigmoid delegated to the substrate — bit-preserving, pinned

`crates/riir-infer-gpu/src/regime_gate.rs` `fn sigmoid` is now a thin delegation to `katgpt_core::exact_sigmoid` (the bit-stable two-branch reference form, body-identical — NOT the Cephes `sigmoid`, whose ±40 clamp / ~1.2e-7 drift a gate decision…

## 2026-10-30 — AGENTS.md compaction: the Role section's lane-landing narratives moved here (switches + postures stay in AGENTS.md's lane table)

GDN L8 eager prefill (plan 621): PROMOTED DEFAULT-ON 2026-10-24 — riir-ai Bench 993 +3.22% / Bench 997 +2.76% pp2048, pp4096 +2.62% post-promotion, G1 bytewise; the ±8.4% band clause superseded by measurement. GEMV WG occupancy (riir-ai Issue…
Prework-fused `e2edf61`+`2581fdd` (riir-ai `d25c7c460`), CUDA N/A proof `69e7d8e`.

## 2026-10-29 — Issue 654 promotion e2e EXECUTED + the gate REFUSED: the CUDA-placed head measured, the roll-chain loss priced structural (Bench 931)

The owner-GO'd promotion rung ("654 lgtm", claimed zed-m3-i654-promo) executed on the 4090 primary platform per the pinned posture-B gate — promotion REFUSED, the arm stays opt-in default-off: every replicated cell loses to plain decode (k=2:…

## 2026-10-29 — Bench 929 follow-up: the plan-623 launch-context discriminator RE-TESTED — class refuted, harness spawn-retry landed

The Bench-929 PROVENANCE discriminator ("detached-matrix spawns EHOSTUNREACH 6/6 while foreground connects, root cause not found") was re-tested on demand (zed-m3-idle1029b): the launch-context class is REFUTED as a deterministic posture property —…

## 2026-10-29 — Issue 1155 CLOSED + removed: the exact slot-addressable GDN state PoC — hypothesis CONFIRMED (claim `89b4880`, tests `36c71f6`, addendum `5552964`, hash re-pin `074fc8f`)

The Qwenputer identity PROVED on our kernels (the divagr18/qwenputer distill, Research 018): one-hot keys address one state column through the REAL `gated_deltanet_step_inplace` bit-exactly when payloads are bounded integers in the ±2^24 grid +…

## 2026-10-29 — Issue 1150 follow-up LANDED: the dense-f16 matmul2d prefill arm — the Metal scalar-GEMM wall broken on the dense lane (A/B 50.22× kernel-level, claim `2c6dfaa`+`049d3ce`, this commit)

The Issue-1150 session record's named follow-up (all in-crate CubeCL GEMMs at ~110 GFLOP/s on Metal — the wall the Bonsai league's 0.249× prefill row shares): `gemm_dense_f16_metal_tensor.metal` + `gemm_dense_f16_metal_zero_copy.rs` — the dense…

## 2026-10-29 — Plan 623 Phase 3 COMPLETE: T3.5 scoreboard LANDED + the T3.4 arm wire-refuted and demoted (Bench 929, issue 1153 `ed10f14`+record, issue 1154 `6a1c524`)

The Phase-3 scoreboard on the amo⇄bmo TB wire (zed-m3-i971-housekeeping, foreground-chunked cells, bmo carrying the disclosed mammoth-capture co-tenant): decode row — ours 2-box s1 52.28–52.59 tok/s BEATS llama.cpp-RPC 2-box 48.6 (+7.6%; the…

## 2026-10-29 — Plan 623 Phase 3 T3.3 CLOSED + T3.2's rkey half CLOSED: the MTU ladder is neutral, RDMA WRITE refused-by-silence (Bench 928, T3.2/T3.3 `570ee79`+`5304716`, bench `f9c4803`)

`verbs_write` + `verbs_metal` (feature `rdma_metal`, metal 0.31 — one workspace version, the BOUNDARY.md metal row) + the verbs-module surface they needed: `create_qp_typed`/`create_rc_qp`, `post_rdma_write`/`post_rdma_read` (the WR opcode +…

## 2026-10-29 — Plan 623 Phase 3 T3.4 LANDED: the decode-feed async-finish arm (issue 1151, T3.4 `9f3c0f7`)

> ⚠ 1151 is DOUBLY HELD — this is the SECOND holder (`1151_plan623_t34_gpu_comm_sync`, filed `148f22d` reading the counter's value 1151 instead of value+1 — the counter already stood at 1151 after `559be00` — removed `726cec7`).

## 2026-10-29 — Plan 623 Phase 3 T3.1 LANDED: the verbs FFI lane — first RDMA traffic on the amo⇄bmo TB bridge (Bench 927, T3.1 `226cace`+`16fb64a`, bench `322e843`)

`crates/riir-infer-dist/src/verbs/{ffi,mod}.rs` behind the opt-in `rdma` feature (macOS-only by contract — the `ibv_context_ops` vtable layout is Apple's librdma.dylib header's; rdma-core's evolved vtable is not claimed): the SDK header mirrored…

## 2026-10-28 — Issue 1151 CLOSED + removed: the promoted serving posture has a single home — the `production` aggregate + `PRODUCTION_RUNGS` + the dist posture floor (fix `80ff665`, close `b5acf92`)

> ⚠ 1151 is DOUBLY HELD in this repo — this is the FIRST holder (`1151_production_posture_aggregate_single_home`, filed `559be00`, removed `f6b3b6a`), and it KEEPS the number (first allocation; katgpt-rs `citation_weight.py` awards it 11–5).

## 2026-10-28 — Issue 1129 CLOSED + removed: the cuda_backend CubeCL decode A/B validates the dense-pair default (riir-ai Bench 1014, the 4090 lane)

The remaining scope (the CudaRuntime arm of the DEFAULT-ON dense-pair fusion — the unmeasured-default class; the scored 4090 decode lane is cudarc, whose K2 fuses the pair) measured + closed from the amo idle lane over SSH while amo held zero-cargo…

## 2026-10-28 — Plan 623 T2.5: the boot-probe planner is fully wire-automatic (the probe EXCHANGE + its residue)

The EXCHANGE (zed-amo-i623-exchange, `22eb298`): the head fetches the tail's boot-probe record over the wire and composes `plan_split` live — no operator hand-carries JSON between hosts. Commits: `a260b62`.

## 2026-10-28 — the grid-reselection drafter arm LANDED (katgpt-rs 966/plan-632; the arc closed NEGATIVE on the lane read, the arm stays opt-in)

`lattice_walk_reselect` (feature `grid_reselect`, `c04e3fa`): the keyed-walk body with the pick replaced by the prefix-first scored law `argmax_b [ln softmax(s/T) + λ·ln P(b|a)]` over each slot's own selector grid — re-scored along the re-selected…

## 2026-10-27 — Issues 1145 + 1139 CLOSED + removed: the M5 Ultra decode-gap campaign closes at 1.04× vs llama.cpp (Bench 907: multi-cube norm+rotate PROMOTED DEFAULT-ON, +2.83%)

Issue 1145 (the campaign's last routed rung, filed from the 1139 T4 census): one launch of `dim/hblock` cubes (5 at Bonsai-2's 5120/1024) replaces the mlp4-norm + r4-staging PAIR at each decode norm site — every cube redundantly runs the…

## 2026-10-27 — Plan 623 Phase 4 (TP): T4.1/T4.2 cell built + correct, gate REFUTED by measurement (Bench 906); Issue 1109 CLOSED + removed

First real TP cell over jaccl (`scripts/plan623_mlx_tp2_bench.py`): column/row sharding of the Bonsai-2 pack runtime, both ranks run all 64 layers, `all_sum` after row mats + vocab-split lm_head `all_gather`, greedy lockstep with NO token exchange.

## 2026-10-27 — Issue 1127 CLOSED + removed: the decode-hold escalation ladder — T1–T4 all executed with records; no owner gate remains (verdict, this session)

All four tasks closed with measured records: T1 attention-chain rung opt-in +0.77% (Bench 900); T2 census-negative (Bench 902); T3 measured-negative at the bit-identical boundary (Bench 903); T4 league re-cell EXECUTED (Bench 1008, riir-ai…

## 2026-10-27 — Issue 1133: GDN conv+beta/decay+expand merge rung MEASURED-NULL (Bench 905: G1 exact everywhere, G2 2/4 pairs, pooled ≈0% ± 0.5%) — shipped opt-in, NOT promoted (`a4da7be` + the rung commit)

The last unmerged launch-bound boundary in the armed GDN decode chain: `conv_beta_decay_expand_fused_f32` folds conv1d + the Bench-899 beta/decay+expand pair into ONE launch (48 layers × 1 node ≈ 62 µs/token dispatch predicted ≈ +0.57%).

## 2026-10-26 — Issue 1122: history-fed n-gram drafter probe CLOSED-NEGATIVE (Bench 895: re-roll α 0.063 vs breakeven 0.58 — T1 wiring refused) — file removed

The NgramRL class (draft decode call N from the model's own same-prompt outputs) probed BEFORE wiring per the Issue-665 honest-assessment pattern.

## 2026-10-26 — Issue 1121: StabilityTracker first-ordinal pin CLOSED — landed `c34b954` — file removed

Hardening pin (Issue 1106's `stability.rs`): the first observed stage is now `debug_assert_eq!(ordinal, 0)` — a partial/offset capture feed (e.g. starting at `AfterLayer(30)`) passed both shipped guards (width + strictly-increasing) and reported…

## 2026-10-25 — Issue 1118 Finding 1: BPE merge ranks re-keyed to packed token-id pairs — LANDED (Bench 894: G1 bit-identical, G2 2.03× encode, G4 zero-alloc probes) — `ad7b5a0`+`54a6616`+`ddc7105`+`c5032f6`

The merge loop allocated two heap Strings per adjacent pair per merge iteration (its own comment admitted it, deferring "a future optimization"). Commits: `7b2fed9`, `addd0b4`, `0e9cc4ce`.

## 2026-10-24 — Issue 1112: arm8 whole-prefill CUDA crash on unfolded (pre-rotation) synth shapes CLOSED — root cause + fix + regression GREEN (`0522312` + `441b78f` + `63ab2d2`) — file removed

The crasher was never a kernel: the synth Bonsai-2-style file is UNFOLDED with DENSE a/b, `upload_layer_weights` stores the dense f32 mirrors beside EMPTY dummy `TernaryGroupWeights::new(0, 0)` handles, and the whole-prefill PLAIN lane…

## 2026-10-13 — Issue 042: cubecl #1806 idle-server park fix vendored CLOSED — landed `fa86871` (mirrors: riir-ai `4b86d2003`, riir-refine `f0718269`) — file removed

The idle-queue's "cubecl pin-bump re-gate" lane (queued cycles 38–41, blocked on the 038 sibling's WIP — cleared when their lane landed in full at `4db9e15`). Commits: `6bba5fdf0`, `58e7689`, `0c5ed0fe7`.

## 2026-10-10 — Issue 037: wgpu full-suite device-loss cascade CLOSED — root cause + VRAM admission gate landed — `3564a72` (+ upstream draft `dad3e7c`) — file removed

The "which test kills the device" question was a category error: no single killer test exists — the full `cubecl_runtime` suite needs ~16 GiB contiguous commit and the loss fired at the box's GLOBAL VRAM commit position (trainer 7.6 GiB resident;…

## 2026-10-09 — the elementwise params[0]=n root fix + the vendored metal adapter E0308 — `02e7a65`

Retired katgpt-rs Issue 928's two len_derived stability rows: sigmoid_f32 / silu_f32 bound by an EXPLICIT params[0]=n (the copy_f32_grid2d idiom), never input.len(); the new oversized-binding sentinel test proves [n, n+pad) untouched.] Commits:…

## 2026-10-09 — Issue 036: LittleBit init-only sub-1-bit PTQ CLOSED — NO-GO for the format+kernel lane (Bench 034; T4 closed-by-verdict) — file removed

The paper's unmeasured point (Dual-SVID init-only, no QAT) measured end-to-end on gemma-2-2b-it (the T0 host decision `8d3fd7e`: local PPL bin over existing substrate — `forward_gemma2_f16` + `nll` + SentencePiece; wikitext-2-raw test sha256…

## 2026-10-07 — Issue 035: FA-constrained dLLM decoding POC CLOSED — NOT PROMOTED (G2 FAIL, sequential lane) — `7d90dfd` (+ file removed this commit)

P3 executed on the 4090 box (quiet: desktop-only GPU consumers, 26.6/33.3 GB RAM free; real-vocab arm = gemma-2-2b 256k, `../riir-train/data/gemma-2-2b-it-f16.gguf`). Commits: `f4ffa93`, `7bda765a`.

## 2026-10-06 — Issue 028 T4 S2: the q4 weight arm — the container loads and runs the dual-format pair — (this commit)

`ProjWeights { Ternary(TernaryGroupWeights), Q4K(Vec<BlockQ4K>, rows, cols) }` at the 10 per-layer projection fields; `bitlinear` dispatches the Q4_K arm to fused `gemv_q4_k_row` (rayon across 16-row chunks, bit-stable), GPU hooks refuse q4 LOUD,…

## 2026-10-06 — Issue 028 T4 S1: the Q4_K prefill pack exists — plus the science premise settled — `94a4e7c`

`Ternary-Bonsai-2-27B-Q4_K.pf.gguf`, 14.43 GB, blake3 `0a7a5605fadc37b482c0fddf6c28334244eea818b1716461bddb412f1a28309c`; the science premise: the bonsai is TRAINED TERNARY, so the PQ2_0 wire is exact up to f16 group scales — requant carries only…

## 2026-10-06 — the GPU crate is clippy `-D` clean at every posture — `e993eff`

Pre-existing warnings gone; the rotation quantize pair (`quantize_rotate_q8`/`_permute_rotate_q8` + `*_warp` fields) is test-consumed only — allow gate `not(all(test, feature = "prefill_q8_act"))` (the first `not(any(test, …))` was wrong in exactly…

## 2026-10-06 — Issue 035 P2.6 LANDED: real-vocabulary compile + the row-fold — the honest G2 regime — `ddcaeb0`

Trie-stepped subset construction in `fa_schema.rs` (256k vocab compiles the fn-call grammar in 60s; toy 0.13s, object 1.8s) + one vocab-wide exp row per position in `fa_posterior.rs` (fn-call draw 11.79s → 1.98s at L=256).

## 2026-10-06 — Issue 035 P2.5 LANDED: DFA minimization — the G2 ratio drops ~4.4× — `9e94684`+`672e2b0`

`src/fa_minimize.rs` (UNGATED, 7 tests): trim + block-granularity partition refinement applied ALWAYS inside `fa_schema::compile` (language-preserving by construction); empty-language schemas now fail at COMPILE time (`SchemaError::InvalidSchema`).

## 2026-10-06 — Issue 035 P2 LANDED: the schema front end + the honest overhead table — `1254264`+`5b5c462`

`src/fa_schema.rs` (UNGATED, 12 tests): JSON-schema subset (six types · properties/required · items · enum · anyOf) → Thompson NFA → on-demand subset graph → token `Automaton`; three grammar-soundness laws pinned (fresh ws wrappers,…

## 2026-10-06 — Issue 022 T5.4 CLOSED: the equal-FLOP skip-class gate — the lane wins, honestly — `8bd5c2d`+this commit

Record `.benchmarks/022_t54_skip_arms.md` (instrument `src/bin/twt_skip_amputation.rs` at `a7b080c`): twt 0.8919 > hydra 0.8642 > random floor 0.8437 >> ShortGPT 0.0320 at the equal 3-layer cut — selection disagreement total, ShortGPT's BI greedy…

## 2026-10-06 — Issue 035 P1 LANDED: the `fa_constraint` decode wiring — exact joint x→0 per denoising step — `304876f`+`f4ffa93`

P0's sampler wired into the `gemma2_d2f` denoising loop behind opt-in `fa_constraint` (implies `gemma2_d2f` ⇒ `dllm`): ONE exact joint draw per step; commit axes `RawTop1` / `ConstrainedMarginal` (marginal log rides `SamplerFeatures::marginal_log`,…

## 2026-10-01 — riir-ai Issue 1004 R2 LANDED: the fused GDN prework (conv1d + SiLU + q/k L2-norm + head expansion, ONE dispatch) — `929d31e`

Feature `deltanet_prework_fused` (toggle `set_prefill_prework_fused`): 2 dispatches/layer vs the shipping chain's ~69; stage-isolated 7.29× @2048 / 9.13× @4096; G1 bit-identical (logits FNV `0ec2396fd4627f29` @2048 / `f35280cb95f5d306` @4096).

## 2026-10-01 — Issue 026 CLOSED (hygiene): the DQ phase-matrix instrument landed and ran — every axis INADMISSIBLE at the frozen corpora

Plan 614's instrument (`dq_fakequant` + the `dq_phase_matrix` runner; record `.benchmarks/023_dq_phase_matrix.md`) ran to EXIT0: base arith 0.9583, NIAH pooled 0.9896 — both OUTSIDE the pre-registered 0.25–0.95 window, so T3's damage-ratio…

## 2026-10-01 — Issue 032 DONE: Metal CubeCL prefill computes Hadamard-folded (Bonsai-2) models (`5d0592f`)

Batched twins of all 8 decode rotation sites in the CubeCL prefill body + `GemvBatchedCubeCL::launch_token_grid`; G1 vs folded decode-eager top-1 equal, worst rel err ≤ 1.5e-5; pre-rotation Q2_0 pin `99a0733c45a0e663` @2048 exact; first Bonsai-2 M3…

## 2026-09-30 — reflex Issue 054 Part 2 REFUTED for this lane: the prefix-state handoff lead — laya is ModernBERT (bidirectional), not GDN; the coupling gate landed

The premise named the wrong architecture: laya checkpoints are ModernBERT-large / mmBERT-base (no causal mask; the state also sits after the per-question head span — different RoPE offset per question).

## 2026-09-30 — Issue 013 T2 EXECUTED: the K=V+ λ ladder on gemma-2 — G-A PASS vs a catastrophic baseline; K=V+ not viable standalone; G-D overfit; NIAH build crash (029)

Record `.benchmarks/012_t2_kv_ladder_gate.md` (+ `012_t2_run.log`, table `012_kv_table_residual.bin` sha `3a0f5333d6bafd63…`, reused by T3): f16 6.0907 (== T1, cross-run determinism PASS) vs k-0.00 catastrophic (tax +30,948,056%); k-1.00 λ*=1 G-A…

## 2026-09-30 — Issue 013 T3 EXECUTED: P3 V-cache reconstruction — G1+G3 PASS at every λ; the 50.0% bytes/token law holds; G2 +7.9–8.8% recorded (the promotable P-half)

Chained run rc 0 (waited 399 min for T2, BLAKE3-verified its table artifact).

## 2026-10-05 — Issue 013 T4 scaling half EXECUTED: the naive read-path cost explodes with context (1.08→1.79→3.82×); the pre-registered rule fired HOLD — katgpt-core `fitted_v_reconstruct` stays opt-in

Bench 016 (`.benchmarks/016_t4_g2_scaling_hold.md`, cells `016a`/`016b`, log `016_t4_g2_run.log`; cache-sizing fix `2300457`): 1.079× @129 → 1.789× @1025 → 3.821× @4097 window edge ≫ the ≤1.20 promotion bar → katgpt-rs promotion HELD (`682936a1c`…

## 2026-10-05 — Issue 013 COMPLETE (T4 kernel half + the promotion EXECUTED): both levers landed as one read path, the window-edge bar met — `fitted_v_reconstruct` PROMOTED to katgpt-core default

Bench 017 (`.benchmarks/017_t4_deferred_lever.md`, cells `017a`/`017b`, log `017_t4_deferred_run.log`): the fused deferred-restore attention kernel + the bitwise per-position (sin,cos) table landed as ONE read path (`fe75497`, extended by katgpt-rs…

## 2026-10-05 — Issue 033 COMPLETE: the KV-axis extension arm landed and ran — the KV-store axis is NULL (a4-class KV storage accuracy-free), the activation axis dominates; instrument extended with the KV arm + hardened corpora

Bench 033 (`.benchmarks/033_kv_axis_extension/`, exe at `ad683a6`): the runner gained the KV-STORE axis (`DQ_KV_GRIDS` — every stored K/V row rounded at the CUDA `kv_fill` prefill + CubeCL `forward_attention_layer_gpu` decode sites) +…

## 2026-09-27 — Issue 014 RESOLVED: activation-aware ternary scale fit — SPLIT BY LANE (born-ternary clean negative; dense-parent mechanism transfer)

From katgpt-rs Issue 886 (substrate `0fb2254d9`, katgpt-rs Bench 896).

## 2026-09-27 — Issue 020 RESOLVED: the len_derived CAPACITY finding was the classifier's, not ours — and both fixed temp paths are gone

Finding 1: CAPACITY at `encoder_lane_cubecl.rs` was a classifier false positive, closed in the katgpt-rs instrument (`43c14d754`, the shared `length_from_handle_size_method` rule) — no riir-infer code changed. Commits: `20ea589`, `337da41`.

## 2026-09-26 — Issue 019 RESOLVED: the c0dfa06 Metal barrier REVERTED — A alone was the wobble, and serial dispatches never needed barriers

Necessity conceded on the timing re-check (the "clean ×3 with barriers" triplet was confounded by `2a34bd3` in the working tree); premise refuted on the code: the fork's `begin_compute_pass` never sets `MTLDispatchType` — Metal defaults to SERIAL,…

## 2026-09-26 — Plan 611 DONE: the T7 op-layer unification verdict (Bench 006) — riir-reflex Issue 008 closed

One portable CubeCL `Backend` over this repo's op layer (S1–S4), A/B'd against the hand lanes (S5; rules pre-registered at `fac0dfd`): CubeCL 5.1–8.4× slower than Metal, 0/12 wins (Bench 006, results `ac85c8e`) — the hand Metal lane stays the macOS…

## 2026-09-26 — Issue 018 CLOSED: the Cubecl-posture drift wobble was a softmax write-after-read race (fix `2a34bd3`)

Both CubeCL softmax kernels (`softmax_f32`, `softmax_rows_inplace_f32`) read the max from smem slot 0 then wrote the sum over it with NO barrier; one `sync_cube()` after the read (+ chunked-launch fix: rows past `MAX_WG_X` re-normalized row 0 — the…

## 2026-09-26 — the riir-ai carve docs adopted into this repo (Issues 998 + 1003, Plan 610, Proposal 041, Benches 870 + 871, doc 002)

Moved home with numbers verbatim (every "riir-ai Issue 998 / Plan 610 / Proposal 041" citation resolves to the same document): `.issues/998_riir_infer_repo_promotion.md` + `.issues/1003_riir_infer_carve_remains_4090.md` (OPEN),…

## 2026-09-25 — Issue 001 CLOSED: EXL3 (trellis-coded) weight format — reader complete, fused-GEMV lane closed at this tier

Full record (§1–§17 numbers unchanged) in `.docs/001_exl3_trellis_format_support.md`.

## 2026-09-25 — T10 rung 2 (Metal): the attn_rope hoist pre-pass — default-off, promotion probe pending

`attn_rope` derives Q/K rope ONCE per layer into packed `[2, seq, d]` device scratch (`5ef7442`); opt-in `LAYA_METAL_ROPE_HOIST=1`, the in-kernel rope arm stays default and bit-identical; promotion via quiet-box position-balanced A/B (instrument…

## 2026-09-25 — two silent MiniCPM5 (llama-arch GGUF) defects, found bringing it up as Issue 011's long-context fixture

RoPE pairing `0b26b9a`: llama.cpp stores llama Q/K rows interleaved; this crate's rotate-half RoPE never un-permuted — `rope::unpermute_interleaved_rows` in `load_llama_weights_gguf` (ppl 259.93 → 69.0547 == HF fp32; gemma2/qwen2 are NEOX,…

## 2026-09-25 — Issue 010 CLOSED (REFUTED): Gemma-2 has no QK-norm; the 288-tensor fixture is upstream-faithful

The premise was false: upstream `Gemma2Attention` defines no `q_norm`/`k_norm` — it bounds scores with `attn_logit_softcapping` (tanh, cap 50), already applied here (`attention_head_softcap`); per-head QK-norm arrived in Gemma-3. 288 = 11…

## 2026-09-25 — Issue 009: software-pipelined narrow staging CLOSED — NEGATIVE at the gate, and the probe's tiny-op floor exposed

`sgemm_narrow_pipe` built, measured, reverted: gate NO-GO (probe −1.5%; forward-level paired env-flip −0.8..−1.6%, reproducible 3/3, under the ≥3% bar; nvcc SINKS register loads — the `asm volatile("" ::: "memory")` fence is load-bearing).

## 2026-09-25 — Issue 008: the narrow reg4 rung CLOSED — NEGATIVE, the narrow zone is TLP-bound, not bandwidth-bound

`sgemm_narrow_reg4` (4×4 fragments, 32×16 warp tile → 128 threads): every true narrow-served row regressed +41..+177% (`sgemm_shape_timing`, the `LAYA_CUDA_REG4` axis; the wide/xwide rows re-measured the 007 rung at −2.6..−20% both runs — the…

## 2026-09-25 — Issue 007: the register-blocking sgemm rung CLOSED — 4×4 fragments, −9..−21 % kernel on every wide/xwide shape

`sgemm_wide_reg4` + `sgemm_xwide_reg4` (16 accumulators, 2 B smem/FMA vs 3, ceiling 50%; same grids so block-fit cliffs carry over; k-ascending per-output accumulation).

## 2026-09-25 — Issue 006: the float4 sgemm rung CLOSED — every instance's B loads collapsed, −7..−17 % on every suite

The packed zone's problem was LOAD-ISSUE THROUGHPUT (12-16.5 TF/s = 15-20% of peak; 6 smem loads per 8 FMAs with the four B loads contiguous).

## 2026-09-25 — Issue 005: CUDA graphs CLOSED — NEGATIVE, the lane is GPU-bound (submit fully hidden)

Instrument ships (`LAYA_CUDA_STATS=1`, zero cost unset): `gpu == wall ±0.3%` on every row of every checkpoint — the whole CPU submit path (launches ~15 µs/call, uploads, ~0.2 ms allocs) executes inside the GPU window; a graph replay removes work…

## 2026-09-25 — Issue 004: the sgemm tile ladder CLOSED — block-fit floors, the narrow single-question win

Three instances (`sgemm_narrow` 32×64×64, `sgemm_wide`, `sgemm_xwide` 64×128×32) picked by BLOCK-FIT on the SM count — the M3 `m < 256` rule does NOT transfer to the 128-SM 4090 (measured cliff: narrow −15.7% at n=2048 exactly 128 blocks, +46% at…

## 2026-09-25 — Issue 003 CLOSED: CUDA flash attention — the packed-path zeros defect fixed + the fused rung

v1's trait-default attention sliced host memory at non-zero offsets → `(ptr,len)` chain MISS → stale-zeros upload (typed_decisions 0.7445→0.2690; the evidence was the published bench, reflex `.benchmarks/026_4090windows_cuda` vs…

## 2026-09-25 — CalibrationTables promoted substrate-side (883 P0 Kimi fixture rider)

`katgpt_core::fitted_anchor_table::LayeredVkCalibration` (+ `VkLayerTables`) is now the ONE builder for both 883 P0 fixtures — this repo's gemma-2 dashboard and katgpt-rs's new Kimi-K3 dashboard (katgpt-rs Bench 889: ρ(V)≈ρ(K)≈ρ(V−K) per MLA layer,…

## 2026-09-24 — Issue 002 CLOSED: the CUDA backend for the laya lane (the 4090 bench row, 17–60× the CPU posture)

Landed `9b52cb1`/`99f156e`→`e99d767`: `laya-riir-cuda` — cudarc 0.19 (`driver`+`nvrtc`, `cuda-13030`+`fallback-dynamic-loading`, target-scoped `not(macos)`), CUDA C → PTX at construction (NVRTC, sm_89); the Metal architecture ported verbatim…

## 2026-09-23 — crates.io publication: keep `publish = false` until the vendor patches upstream (owner-gates menu v2 row 2)

HARD blocker: both `vendor/` forks are load-bearing via `[patch.crates-io]` (`cubecl-runtime` — the #1359 drop-queue fix; `wgpu-hal` — the VRAM accessors) and a `[patch.crates-io]` section does not survive publication.

## 2026-09-25 — Issue 020 T6 CLOSED NEGATIVE: the encoder's host side is 1–1.6% of forward wall (measured before building)

Instrument `crates/riir-infer-laya/tests/metal_host_gpu_split.rs` (`#[ignore]`, `required-features = ["laya-riir-metal"]`, measurement-only) splits `Encoder::forward_packed` into `enq` (the whole host side) vs `sync` (`download_into`): seq188…

## 2026-09-25 — the narrow sgemm's shape is the measured local optimum (T7 occupancy axis refuted, both arms)

Two challengers behind a temporary `LAYA_METAL_SGEMM_VAR` flag (from the consumer's `sgemm_shape_timing` probe, reflex `ebe667e`): bk32 lost 17–33%, bn32 15–45% — the barrier cost beats the 2–3× co-residency gain; at BK 64 a 2-TG fit breaks the…

## 2026-09-25 — the sgemm MMA-roofline probe: the narrow instance is staging-bandwidth-bound (reflex Issue 020 T7 follow-up)

`crates/riir-infer-laya/examples/sgemm_roofline.rs` (measurement-only, `[[example]]` required-features row per the T1.1e law): the shipped narrow kernel verbatim vs an MMA-only twin (staged once, no re-staging, position-balanced).

## 2026-09-25 — f16-B staging REFUTED at kernel level (the roofline probe's own follow-up arm)

The `sgemm_hb` arm (`device const half*` B, RNE `f32_to_f16` seed): flat ±2% on every cell — B (12.6 MB f32) FITS L2 (~32 MB), so the binding cost is the L1/threadgroup-issue path, which halving bytes does not relieve; the f16-B backend rung is…

## 2026-09-27 — Issue 021 CLOSED: the cuda CLS-row corruption — a chain-cache prefix-match

## alias, not a kernel race (the fused-GLU temp's slot vs the hidden that reused its address)

Fixed at the root: `chain_buf`/`chain_slot_for` now EVICT same-pointer different-length entries on bind; plus `download_into`'s `memcpy_dtoh` is `cuMemcpyDtoHAsync` (stream-ordered but ASYNC) — a trailing sync now guards the host read.

## 2026-09-27 — Issue 011 CLOSED: `row_logit_floor` model-bound G1 complete — needle@64K PASS at every arm

T3c (MiniCPM5-1B at 64K real dilution, log `/tmp/ri011run/t6_minicpm64k.log`): PASS at every arm b8/b6/b6s0/b4 — 3/3 seq-exact, 0.00% top-1 flips over 12 scored tokens, m_Y preserved within 0.0007 (same top head L15H7), base ppl 1.0362.

## 2026-09-28 — Issue 017 (gpu_transpose dead module) + Issue 023 (fence F2 self-alias FP): both closed

017: `gpu_transpose.rs` + `src/kernels/transpose.wgsl` deleted (zero callers then known), `transpose_cubecl.rs` (Issue 572) covers the reachable use, BOUNDARY.md repointed, the riir-ai re-export dropped (`6c6bf169b`).

## 2026-09-28 — Issue 025 (owner-gate pickup) closed: D7/D8 executed, D9/D10 recorded

D7: gpu_transpose deleted + BOUNDARY.md repointed (record above).

## 2026-09-29 — Issue 022 Phase 3 complete (audition + zero-training surrogate); Issue 024 closed measured-N/A both mechanisms

022 P3 (`27c9f86`): `src/twt/audition.rs` (mean + LaCo RDSC merge operators, the per-channel α/β branch-correction fit, BLAKE3-pinned selection) + `examples/twt_laya_audition.rs`; the apply path proven BIT-IDENTICAL against the parent forward every…

## 2026-09-29 — Issue 022 Phase 4 LANDED (re-ternarization arms + κ budget + collapsed-GGUF writer)

`baeb686` behind `twt_collapse`: `src/twt/ternarize.rs` (three deterministic arms; PRE-REGISTERED budget κ = 2.0, τ_code = 1; arm B = integer CODE vote, arm C divides by the f16-ROUNDED scale and is BIT-EXACT on ternary input) +…

## 2026-09-28 — Issue 015 audio PoC DEFERRED (owner: until M5 Ultra) + 023's missed hunk landed

Issue 015 stays OPEN, T1–T6 deferred (`- [-]`), the turnkey recon recorded in-issue (exact bundle `silero-vad-unified-256ms-v6.2.1.mlmodelc`, the wire contract from FluidAudio's `VadManager` @ `20d4f0bd`; the audio lane bypasses the laya…

## 2026-09-29 — Issue 022 T5.0 LANDED: the passthrough-collapsed checkpoint + the first real GOAT numbers (coarse grid FAIL; fine end measured)

Loader prerequisites fail-closed (`318b9fa`/`ceb96c3`): the writer REFUSES a qwen35 collapse without `twt.layer_types` (U8 `DeltaNetLayerType`) and refuses a stale `qwen35.nextn_predict_layers`; the loader…

## 2026-09-29 — Issue 022 T5.0 COMPLETE: the agreement cliff mapped — 4.7% depth cut PASSES the GOAT bar, quality parity holds to 11%

Bench `.benchmarks/022_t5_collapsed_goat_agreement.md`: ε=0.01 (61/64 blocks, 4.7% cut) agreement 0.9486 — the FIRST pass of the pre-registered ≥0.9 bar; ε=0.015 → 0.8955; ε=0.02 → 0.5247; ε=0.03 → 0.0301; the grid points all FAIL.

## 2026-09-29 — Issue 022 T5.1 lane (1) COMPLETE: gemma-2 f16 control — clean negative at every real depth cut; the fine-end bracket is structurally empty

Instrument `0e0436c` (`PostLayerHook` capture seam on `forward_gemma2_f16_tapped`; `examples/twt_gemma2_profile` behind `twt_gemma2`; arch arms in `twt_collapse_emit` + `twt_goat_agreement`), verdict `55fa831`; bench…

## 2026-09-29 — two Windows batch traps hit by the T2/T3/T5.1 schtask runners (recorded; the deleted wait-loop script's durable note)

(1) A `Start-Process`-launched `.cmd` inherits the MSYS PATH — `tasklist | find` resolves `/usr/bin/find` and the wait-loop wedges forever; the schtask launchers are immune (clean system env → `C:\Windows\System32\find.exe`).

## 2026-09-29 — Research 004: Disaggregated Quantization distilled (arXiv:26.26333) — three issues filed, riir-train Plan 430 routed

Distill of arXiv:2609.26333 (DQ/QADD), deepening the riir-clippy arxiv-walk-217 row: Issues 026 (phase-isolated quant sensitivity bench), 027 (T0 encoder-only asymmetric Q2_0 + Lloyd-Max grids), 028 (dual-PTQ resident disaggregated serving) filed;…

## 2026-10-01 — Plan 614 LANDED: the DQ phase matrix ran to EXIT0; every axis INADMISSIBLE at the frozen corpora; the instrument (and its defect chain) is the deliverable

v2 `23bbff5`+`3cf9ae9`, EXIT0 (log `F:/wt/dq614-matrix2.log`, report `F:/wt/dq614-matrix2/dq_phase_matrix.md`, record `.benchmarks/023_dq_phase_matrix.md`, `.highwater` 15→23): all four axes INADMISSIBLE (base arith 0.9583 — the `a07fffd`…

## 2026-10-01 — Issue 027 CLOSED measured-negative + removed: the offline LUT grid lane (Plan 615) — weight-space wins were real, the 2-bit class is function-destroying on dense artifacts

`2cb807d`: T0's asymmetric encoder (`quantize_row_q2_0_asymmetric`) +4.9 dB/family and Q2_0A's Lloyd-Max grids +0.5–1.2 dB/family — but MiniCPM5-1B ppl 6.8e5–7.0e6 vs base 49.8 at EVERY reachable granularity; the pre-registered class gate fired.

## 2026-10-01 — Issues 998 + 1004 CLOSED (hygiene): the D4 re-narrowing dissolved under the ratified no-migration architecture; the corpus-follows triggers resolved negative

998: the D4 re-narrowing is resolved-dissolved — P3/T7 landed as the ENCODER-lane unification under its pre-registered NO-DELETION verdict and D10 (owner-gate, riir-ai 1016) ratified the training families stay riir-gpu-side BY DESIGN; every…

## 2026-10-05 — Issue 031 CLOSED measured-non-repro + removed: the dq614 FATAL teardown hang did not reproduce across 6 arms; the A/B instrument is landed as durable

Instrument: `crates/riir-infer-gpu/src/bin/dq614_teardown_repro.rs` (`required-features = ["ternary_gemv_cuda_raw"]`, whole-file `#![cfg]`, dev-profile refusal) with arms `--exit plain|graceful|hard` × `--sticky` × `--alloc-mb` × `--iters` ×…

## 2026-10-10 — Issue 040 CLOSED landed + removed: the weight-histogram MDL description-length audit (GGUF lane); the EXL3 rows ride the loader lane

`483f2ec`: `quant::desc_len` (feature `desc_len`) + the `desc_len_audit` report bin — per-tensor entropy / floor / slack sorted by HONEST slack, with the two-part honesty law (`N·H(p)` prices the weight VALUES only; the codebook + block-format…

## 2026-10-11 — Issue 1003 CLOSED (owner-gate record superseded): the D10 disposition stands — the 69 training families stay riir-gpu-side BY DESIGN; the live homes are the riir-ai BOUNDARY.md D4 row + riir-ai HISTORY:360

This closure SUPERSEDES the 2026-10-01 sentence in the Issue-998 closure row ("Issue 1003 stays OPEN deliberately") — that sentence predates the owner's 2026-10-05 rule ("rm owner-gate issues; ask Claude to judge; defer→HISTORY+close", riir-ai…

## 2026-10-11 — Issue 028 CLOSED (PoC gate fired NULL, container shelved): the dual-PTQ disaggregated serving record; T5 stays deferred with its trigger

T1–T3 landed `09d0dd5`: the `DisaggregatedTernaryWeights` container (`load_pair` / `.pf` one-file / `from_single`), the ownership-preserving in-process `PhaseHandoff`, the escape-set law (`ssm_a`/`ssm_dt.bias`/`ssm_alpha`/`ssm_beta` bit-shared…

## 2026-10-11 — katgpt-rs Plan 622 Phase 7 engine lane LANDED (doc-sync row): the scale-invariant attention vehicle on the packed Bonsai ternary decode path; GPU legs open, GPU-window-gated

Landed `d1b9bd5` (session katopz-m3-622-continue) + follow-ups `71a0845` (loop_probe stale layer_idx repair) / `47f8f6f` (quiet the posture-dependent unused layer_idx warning) / `9f3b27b` (the harness compiles at its OWN required-features posture —…

## 2026-10-13 — Issue 041 CLOSED (T1–T5 all landed): the prefix-cache snapshot submit burst — 96 submits → 2 launches per direction, the serving-lane TTFT lane

Landed `971b842` (T1+T2: the pointer-table copy path — module-local `TableCopyEngine` in `qwen38_prefix_cache.rs`, dual device pointer tables (live-state freshness-checked via a host address compare + per-entry tables), ONE `prefix_table_copy`…

## 2026-10-13 — Issue 1005 CLOSED (T1–T8 all landed; defers carry their triggers): the eDLM/Drex inference lane — GGUF arch, block-causal mask, pointer head, marker encode, real-artifact parity, GPU forward, KV reuse, CUDA graphs

Phase 1–3 complete across four sessions: T1–T3 (the `edlm` GGUF arch + segment block-causal mask + row-form fallback), T4–T6 (marker encode/pack + the pointer head + T6 parity PASSED on the real `drex-dlm-Q8_0.gguf` — 87 tokens EXACT vs their…

## 2026-10-13 — Issue 039 CLOSED measured-non-repro + removed: the wgpu `--lib` device-loss tail did NOT reproduce at HEAD on a clean exclusive box (green ×2 parallel + ×1 serial); the CUDA-posture leg surfaced and fixed a posture-biased TEST instead

The free window opened (verified idle: 5% util, 376 MiB, no compute apps) and the full recipe ran from a fresh isolated worktree at the then-tip `fbd6e6c` (`E:\git\wt039infer`, isolated `CARGO_TARGET_DIR`): wgpu posture (parallel) 258/0 in 520.27s;…

## 2026-10-22 — Issue 1107 CLOSED negative-audit + removed: NO riir-infer path right-pads rows into a shared recurrent (GDN) prefill — the TokenGate pads-to-front law has nothing to repair here; the one multi-sequence surface (laya `forward_packed`) is varlen-packed by construction, and the riir-ai HLA note does not fire either

Audit per the issue's task 3 (a negative result is a valid close; not left open as a maybe).

## 2026-10-23 — Issue 1111 CLOSED + removed: `forward_qwen_deltanet_ternary_hidden` returns the head basis (`R·h` on a Hadamard-folded file) — KEPT and documented, width fixed, pinned on the synthetic folded fixture

Filed from riir-train Plan 461 T2.0 (P3 measured the returned vector = the FORWARD-ROTATED final-norm state on `Ternary-Bonsai-2-27B-PQ2_0.gguf`; the doc claimed the primal state and D = 12288). Commits: `4ede8e7b4`.

## 2026-10-25 — Issue 1113 CLOSED + removed: the mma `kv_f16_to_f32` +112 re-bias — fix + both pins EXECUTED on the 4090 (the compile-in-context window the M3 could not reach)

The one-line re-bias (`(exp + 112u) << 23`) and the two device-free pins landed 2026-10-24 (zed-m3-docsync-956-close) from the M3, where the module compiles to nothing (`#[cfg(all(feature = "ternary_gemv_cuda_raw", not(target_os = "macos")))]`);…

## 2026-10-25 — Issue 1114 CLOSED + removed: the Issue-994 KV guard default is backend-keyed (Metal measured clean to 128K)

The single-buffer KV guard's 208 MiB default is the 4090/`wgpu<spirv>` bisect (riir-ai Issue 994).

## 2026-10-25 — Issue 1110 CLOSED + removed: the β-reopen probe — ROW DIES AT PROBE COST (Bench 890, renumbered from a colliding 888 — see the record's numbering note): the only real gemma2 checkpoint is a plain IT model, the d2f decode is a structural 1-step `<eos>` collapse, the drift census has zero substrate

T2 executed on the 4090 (CUDA posture `gemma2_d2f,cuda_backend`, detached runs, GPU 44 °C / 5 % util, no compute consumers). Commits: `b2fe307`.

## 2026-10-26 — Issue 1119 CLOSED + removed: the zero-copy cache-split handle gap — fix `7b2fed9`, acceptance = the i1122 gate1 cell re-run

The riir-ai Issue 1122 gate1 cell (`gate1_site_isolated_tax`) ran bit-identical simdgroup code with 0 site-timing rows and n=0 site dispatches — the G1-instrument illusion class. Commits: `7b2fed9`.

## 2026-10-25 — Issue 1115 CLOSED + removed: the GEMV occupancy wg rule default-on EVERYWHERE (the 4090 CudaRuntime A/B measured the platform split away)

Bench 887: prefill-tail lm_head A/B on the 4090 CudaRuntime — G1 FNV-identical ×6+6 rounds (pp512 + pp4096, Bonsai-2 vocab 248320 → the 128-wg bucket), reach exact (+1/±0 across 12/12 rounds via the counter), G3 1.0013×/1.0000× sub-noise → per the…

## 2026-10-26 — Issue 015 CLOSED + removed: the silero-vad audio lane PoC — T6 verdict finalized (quiet-ish-box latency re-run, replicated ×2)

The PoC itself LANDED 2026-10-23 (T0–T3: `vad-ane` feature on `riir-infer-audio`, digest-walled loads both postures via `riir-infer-ane`'s `load_model_precompiled` + `predict_f32_multi`, 5-test gate suite green, the ANE finding — all-CPU placement…

## 2026-10-26 — Issue 1123 CLOSED + removed: the issue028 dual-format q4-prefill gate red on develop — the TEST was stale, not the loader (the TernaryQ2 promotion outran the battery)

Filed 2026-10-26 from the plan-623 Phase 1 validation sweep: `dual_format_q4_prefill_pair_loads_and_runs` tripped `matches!(dec.layers[0].in_proj_qkv, ProjWeights::Ternary(_))` at pristine `7a01f05` (filer's detached-worktree run verified it was…

## 2026-10-26 — Issue 1125 CLOSED negative-already-shipped + removed: the Bench-1004 fused-quantize rung — the #395 mechanism IS the shipped K1–K5 posture; the census falsifier discharges it dynamically (0 standalone quantize launches on the 4090)

Filed from riir-ai Bench 1004 (the 0.9730× decode RED, first sub-parity decode cell of the scored era; the opponent's +3.84% jump rode prism b10826's #395/#401/#402 FWHT/q8_1 fusion quartet). Commits: `a2ebb37`, `2e3889e`, `1320dda`.

## 2026-10-26 — Issue 1126 CLOSED measured-negative + removed: the GDN core fusion rung (Bench 898 −3.5%/−2.1%, G1 exact everywhere, opt-in `RIIR_GDN_CORE_FUSED`) and its named 4→3 follow-up (Bench 899 direction-positive 3/3 pairs, opt-in `RIIR_GDN_BETA_EXPAND_FUSED` — NOT promoted, league re-cell owner-gated)

The #398 conv-chain twin implemented as `recurrence_f32_fused_expand_hd{64,128,256}` — ONE launch replacing beta_decay + expand_and_l2_normalize + recurrence_fused, the Issue-742 body shared verbatim via…

## 2026-10-26 — Issue 1127 T1 LANDED (Bench 900): the attention-chain fusion rung — 5→1 launches, G1 bit-identical incl. the graph lane, G2 4/5 paired windows positive (armed median +0.77%), opt-in `RIIR_ATTN_CHAIN_FUSED` — + Issue 1131 filed: the GDN rotation FWHT geometry rung (the census rows outside T3's family)

The T1 rung (`attn_chain_fused_f32` + the Issue-618 devpos twin for the graph-capture lane, `8d51f21`+`1d0291d`) merges split_qg + q/k-norm + RoPE + kv-append into ONE launch per attention layer — every body verbatim, ordering confined to…

## 2026-10-28 — Issue 1147 CLOSED + removed: the T2.4 kernel micro-batching rung — fused-batch v0 LANDED (`03f0fe6`, Bench 912) then DEMOTED same-day on the 2-box wire refutation (Bench 913: armed 46.9–52.6 vs incumbent 99.2–108.5 aggregate decode window, −52% — the BatchN wave forfeits the head⇄tail GPU overlap that IS the 2.37× bubble fill)

`pair_decode` ships default-OFF (`RIIR_DIST_PAIR=1` arms it for loopback rigs only, where the +39% was real and stands).

## 2026-10-28 — Issue 1130 CLOSED G2-FAIL + removed: the Metal concat-epilogue split rung (folded GDN qkv|z Split2 into the ternary concat GEMV epilogue)

Rung `b635a43` (`gemv_ternary_plane_rowtiled8_f16scale_split` + `GemvTernaryCubeCL::launch_split`); G1 to_bits EXACT 13/13 on amo + m3 (`9f1b2ba`). Commits: `445cc4e5a`, `760265d`.

## 2026-10-28 — Issue 1131 CLOSED + removed: the T6 GDN rotation-FWHT geometry rung (CUDA) re-priced sub-bar, re-pin adjudicated NO-GO

Filed `86e419d` (Bench 902 rows `gdn_input_rotabq` + `gdn_out_norm_gate_permute_rotq`, 564 µs/token). Commits: `8d204ba`, `194b952`.

## 2026-10-28 — Issue 1132 CENSUS COMPLETE + removed: the Metal per-dispatch floor census (whole-graph launch attribution on the folded decode)

Instrument `3d95ff9`: `DECODE_STAGE_MASK` 11→22 bits (per-launch-site refinement classes + the previously-ungated decode tail; mask=ALL bit-identical) + riir-ai `b864dc936` Phase E — the census rides the stage-instrument plumbing, no xctrace.

## 2026-10-28 — Issue 1133 CLOSED measured-null + removed: GDN conv+beta/decay+expand merge rung (row above, 2026-10-27)

Rung `0eaebad`, record Bench 905 (`.benchmarks/905_i1133_conv_merge_ab.md`): G1 exact everywhere, G2 2/4 pairs, pooled ≈0% ± 0.5%; ships opt-in `RIIR_GDN_CONV_MERGE=1`, default OFF. Commits: `537ca18`.

## 2026-10-28 — Issue 1134 CLOSED G2-FAIL + removed: the norm-pair epilogue fold rung (input_norm + mid_norm)

Donated kernel `ResidualAddRmsNormRotateCubeCL` (`cbb5bf1`; G1 to_bits 3/3 + REACH PASS) is an e2e LOSS in 7/7 A/B runs (riir-ai `bench_1137_normrot_e2e_ab`): −2.96/−2.23 … −8.90/−9.48%. Commits: `b02f379`, `4aed461`, `f44c740`, `ffb07bf`.

## 2026-10-28 — Issue 1135 CLOSED ceiling-gate NULL + removed: the attention proj+decode launch-cluster rung

Three verified-idle interleaved windows on amo (riir-ai `bench_1138_attn_proj_decode_strip_ceiling` + raw log `.benchmarks/1138_issue1135_attn_cluster_strip_ceiling_amo.raw.log`): proj 32 la/tok × 13.4-20.5 µs (pool 0.43-0.66 ms/tok); the only…

## 2026-10-28 — Issue 1136 CLOSED + removed: the gdn_out_gemv rung — ceiling real, G1 structurally dead; GDN chain launch budget CLOSED BY MEASUREMENT

Ceiling (riir-ai `bench_1139_gdn_outproj_strip_ceiling`, raw log `.benchmarks/1139_issue1136_gdn_outproj_strip_ceiling_amo.raw.log`): +0.524/+0.592 ms/tok (+2.16/+2.46%) @ 48 la/tok. Commits: `afc1f43`.

## 2026-10-28 — Issue 1137 CLOSED NULL + removed: the FFN decode-GEMV ALU-chain rung (and its hand-MSL lane i1138, Bench 1149)

Fork mapping study (`0cc47fa`: `kernel_mul_mv_tq2_0_f32`, PR-26980 algebra) found its in-tree twin `gemv_ternary_fma_cubecl` (Plan 563 / Bench 847, `RIIR_GEMV_FMA` default-OFF); riir-ai `bench_1141_fma_aspect_probe` on M5 Ultra: FMA loses ~3× on…

## 2026-11-02 — Issue 1152 CLOSED + removed: verbs_metal fw gate + the amo RTR EINVAL — the MTLBuffer leg measured CLOSED on macOS 27.2 (`ebe0b11`+`429e085`)

The T3.2 MTLBuffer leg (Plan 623) was code-complete since `570ee79` but gated twice: amo's Application Firewall silently dropped TCP SYNs to unsigned listeners (no GUI session to accept the prompt), and an amo-side provider regression EINVAL'd…

## 2026-10-31 — Issue 1159 CLOSED + removed: non-aligned prefill chunk slow path — the chunked conv1d's p % 64 gate (`6a889df`)

Located by the 1146 stage census at one wide frame (tensor-zerocopy, incumbent prework): deltanet Δ 7370 ms at p=3000 vs 3524 at p=4096; projections linear in p.

## 2026-10-31 — Issue 1158 CLOSED + removed: mask-input drafter refinement audit (JetSpec++ pass-1 collapse class)

Inventory of every `mask_token_id` reader, two classes, never pooled.
Commits: riir-infer `1f85c99`, riir-ai `e6a2cd1e0`, riir-train `bcd9e30e`.

## 2026-11-02 — Issue 1156 CLOSED + removed: the sparse-prefill league lane — T1–T6 all DONE, τ=0.85 promoted-by-verdict (bench 1157; promotion lane = issue 1165)

Full arc: T2 block-sparse arm (τ=1.0 bit-identity contract, riir-ai bench 1151); T3 the ΩMSR per-family sweep — blanket promotion REFUSED on the mean scorer (riir-ai bench 1152), T3-v2 the envelope scorer re-open CONFIRMED — 7/7 families floored (riir-ai bench 1153,…

## 2026-11-05 — Plan 485 P2 landed in riir-train: fixed-workspace `_into` dispatch variants (`f7b0e2d`) + the pre-existing gate red fixed (`5142561`)

P2 (`f7b0e2d`): nine `GpuGemmaCubeCL` dispatchers gained `_into` twins that launch the SAME kernel into a caller-owned output handle — gemv_tiled, gemm, add, gemv_batched_backward (the backward stretch's kernel), geglu, rmsnorm_batched,…

## 2026-11-11 — Issue 1204 CLOSED + removed: the T3 `--resume` paths decoded from a FRESH cache at the cursor (the documented state-rebuild replay loop was missing) — resumed rows were silently corrupt (`be7fa79a`)

T3 (`5f187da7`) promised the replay in its own comments; the loops were never written — both arms resumed from `cursor` with zeroed KV/GDN/flashmemory state, so every resumed row decoded as if the context were empty (the loader's digest pins…

## 2026-11-14 — Issue 1214 T1c LANDED END-TO-END: the int_ref lane ran its first REAL artifact — G3 PASSES (int8 ΔPPL +0.067 vs bf16), G1(a)/(b) green on the real model (`ea1eb27` + `6494992` + `5d2abb2`)

The DetLLM port (decision-content-only, upstream @ `9c3f1c9c`, NO LICENSE) reached its first end-to-end receipts: `scripts/i1214_prepare.py` calibrated Qwen3-0.6B on amo (transformers 4.57.6/datasets 4.8.5/hub 0.36.2 pinned in `.venv` — the v5-era…
Full record: `.benchmarks/1213_i1214_t1c_receipt_g3_ppl_cell.md`.

## 2026-11-12 — Issue 1206 T2b LANDED: the real-activation cell sealed G1 PASS, +26.75pp (`e46a310`)

The katgpt-rs issue-988 grafts T2b cell executes HERE per its host rule (riir-infer path-deps katgpt-core).
CLOSED + removed 2026-11-16 (zed-lens999amo@amo): the lane completed end-to-end after this row was written — T4 folded to katgpt-rs 988 (GOAT `bench_988_grafts_goat` G1/G2/G3(a)/G4 PASS, G3(b)…

## 2026-11-12 — Issue 1212 CLOSED — the cubecl-opt GVN panic storm adjudicated: pin stands, no backportable fix (upstream Pliron rewrite #1402 is the fix), re-opens named

The riir-train Issue 687 finding 1 disposition (issue file removed at close; verdict recorded here): 2,996 compile-worker panics at `cubecl-opt-0.11.0-pre.2/src/gvn/convert.rs:21:39` (`leaders[val]` — the GVN value→leader map missing an operand…
Watch-signature (every lane): `gvn/convert.rs:21:39: no entry found for key` in compile-worker output = this defect.

## 2026-11-13 — Issue 1213 FOUND+FIXED: the FM_MAX_SEL raise left the selection row stride at 128 — the 128K Metal ρ arms ran a collided cross-head selector (VOID; fix `5425494e` + `19c12a3c`)

Found by the issue 1191 T4 trio's sel-frac anomaly (ρ0.15(M) recorded 0.0504 where the bmo CPU trio recorded 0.1488): `66782122` (the 1200 capacity raise, 128→1024) moved the const, the envelope, and the k clamp, but NOT the selection row's stride…
Fix: the four literals → FM_MAX_SEL with lockstep comments + docs; regression lock `fm_budget_metal_row_stride_regression` (n_blocks 1000, ρ0.15, count 150 — fails pre-fix at every kv ≥ 1).

## 2026-11-13 — Issues 1211 + 1209 CLOSED+REMOVED: the FOSCAR Q8KV fold promoted (T3b/T6 negatives) and the weight-structure audit lane landed (G1–G5)

Issue 1211 (FOSCAR affine folding, Research 026): T1–T3 landed + PROMOTED into the q8kv dispatch path (`c40e598`, claim `1d67e95`, T3 by `zed-71cd4872@amo`) — parity PASS at every shape/posture (folded-vs-f64 ≤ 2× incumbent-vs-f64 everywhere),…
Record `.benchmarks/1209_weight_audit_zoo_baseline.md`; T5 (consumer: riir-train mint-time scan + `--diff-base`) stays open downstream.

## Lessons

Check the reference implementation, not a remembered standard, before filing a fixture-integrity finding (Issue 010). A G1 that stops below the production chunk size certifies a size nobody ships (Issue 032). A dead-module grep must cover every…

## 2026-11-14 — Issues 1190 + 1215 + the upstream EchoConsistency fix: the ranked cell lands, the count gate widens, the 3-day E0004 breaks

Issue 1190 (the i1188 T3 ranked-sensitivity cell) LANDED: the first requeue (`720d48c7`) fast-died in ~1 s per pass — the driver's pre-registered flag triple (`--min-length 128 --score-window 512`) violated the bin's own…
Issue 1215 T2 (the count gate widens): four CPU-only suites joined `scripts/test_gate.py` with measured floors — issue879 quant certification (2, release), bonsai2 rotation load (11), issue046…

## 2026-11-12/14 — Issue 1208: the deletion-mask adapter lane (T1–T4 code halves; the released-mask ingest lands census-clean)

The Deletion Tuning adapter lane (riir-train Research 493 track (a); arXiv:2611.10866, NeurIPS 2026) — a trained BINARY mask over a frozen model's gate projections, zero new weight values, deployed GD-free.
T5 code half (2026-11-12): `src/adapter/wanda.rs` — the Wanda scoring substrate (per-column Σx² capture, order-pinned f64 folds, the self-describing `IWANDA1` artifact) + `i1208_wanda_score`.

## 2026-11-16 — katgpt-rs 999 T4 prep: the read-orientation FIX + the gates cell (riir-infer `2ec0245e`)

The T3 real-corpus fit kept running (detached, doc 2/40 at this session's start); this session landed the T4 cells that consume it, and found a real read-layer defect on the way: `s_scores_into`/`r_read_delta_into` were TRANSPOSED against the…
Same commit: `stack_forward_write` — the steering-write injection (additive at a fit point's seed rows; alpha=0 bit-identical, causal + point-local, both tested) — G2's measured arm — and…

## 2026-11-15 — Issue 1188 CLOSED: the ranked-sensitivity tool complete end to end (T0/T1 `45778ee` + T2/T3 via bench 1215 `d825892a`)

The mixed-precision sensitivity-ranking tool (katgpt-rs 987 lead CQ) is complete across all three tasks: T0 family taxonomy + harness fixture (851-tensor census, 0 unclassified; ffn_gate fixture cell proven end to end) and T1 the `--family-ablate`…

## 2026-11-14 — Issue 1181 T0+T1: the encode-share demote gate fires LANE OPEN (Bench 1218) + the pool-bound queue finding (Issue 1217)

The command-replay watch lane's arming trigger completed (the 1165 flip landed 11-12; T1 measured above the bar): the `issue1181_encode_share_probe` cell (`2a744b5`, per-call `resident_enqueue()` distribution + the incumbent front/read P1…
The bmo cross-box confirmation cell is queued on that box's exclusive lane (job `01a1daaa-fa45`) and appends its row to Bench 1218 when it runs.

## 2026-11-16 — Issue 1225 CLOSED: the f16-B serving lane end to end (T1–T4: the decider PASSES, the twin lands, the frame A/B is OVER, the route PROMOTED in the dist lane)

The Bench-1223 f16-B rung graduated and the whole decider lane landed in one engagement (lease amo-idle-i1225f16b@amo, claim `a904584b`, epoch 1794835595, was expires 2026-11-18T13:26:35Z — scope complete same lease): T0′ pre-registration…
T4 the promotion [Bench 1226](.benchmarks/1226_i1225_t4_flip.md): delegated-owner Claude verdict AGREE round 1 (session `4a934b4c-82ab-45fa-8de5-830077072543`) — G1–G4 hold, modelless-clean; all…

## 2026-11-16 — Issue 1117 CLOSED: the FlashMemory budget lane end to end (T1–T3 + the T4 verdict PASS + the T5 NOT-promoted recommendation; bench 1214 the record)

Every task landed; the file's own checkboxes were stale (T4 `- [ ]` while the record existed), and the last lease (zed-d565d322@m3, epoch 1794522000, was expires 2026-11-14T12:00Z) expired unfolded with its scope complete — this is the post-expiry…
The bmo trio's host legs completed valid earlier under bench 1209's host-identity record (collector ALL INTEGRITY PASSED); the issue-1200 envelope class died fail-closed (`fe830fc` + the…

## 2026-11-16 — Issue 1215 CLOSED: the riir-infer standing test-count gate (T1–T4 landed; T3 recorded workstation-owned)

All four tasks are done; the Status line was stale ("T2/T3 open" while T2/T4 landed 2026-11-14) and the lease (zed-41cd0608@amo, epoch 1794597600, was expires 2026-11-16T12:00Z) expired unfolded with the holder session gone from amo's thread list —…
Issue file removed this close; AGENTS.md's Build Commands section already documents the gate (`python3 scripts/test_gate.py`, the floors, `--canary`, `--receipt`).

## 2026-11-17 — Issue 1226 F1 CLOSED VALID: the f16-B pp32768 retention walk (Bench 1234; no family demote)

The i1225 follow-up's first caveat cell (the pp32768 retention row the T4 verdict carried forward) landed over a multi-lane arc (leases amo-idle-i1225f16b + zed-b2e60a87, both folded at the close; the final adjudication executed per the 03:17Z…
Verdict: all 8 placebo rows EXACT 0.000e0 both postures (the v1 failure shape gone — branch (a) confirmed end to end); armed rows STRICT 9.279e-8/9.302e-8 (code/math — the repro-pinned twin drift,…

## 2026-11-17 — hygiene closes + the gdn_rec chunked-form lane filed (amo-idle-1226-f1-arc)

The F1 arc's follow-on housekeeping pass, three closes/folds and one filing: Issue 1214 CLOSED — the int_ref DetLLM port's in-repo engineering is complete per its own gates (G1 FULL PASS all four arms zero-tolerance incl. the 4090 cross-vendor…
Arc-close cleanup from the F1 pass completed the same session: `/tmp/i1225f16b` (7.8 GiB walk vehicle) + 15 log/script fragments removed, evidence committed (Bench 1234 + the void/repro/fail2 logs).

## 2026-11-14 — Issue 1216 DEAD at T0: the decision-head overlay PoC loses in-sample (Bench 1217)

Backfilled doc-sync row (the 11-17 hygiene pass enumerated closes and missed this one; the close predates it by three days).
Re-open trigger: a different objective class, never a re-tune.
