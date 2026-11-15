# Issue 1219: the ANE sync-inference deadlock — tvspeech clip-6 request never completes via ANE_ProgramSendRequest (IOKit mach_msg2 wedge)

**Status:** Open — filed 2026-11-15 from plan 633's P2.3 tvspeech cell (two independent reproductions, deterministic at the same clip; the deadlock stack PERSISTED at `.raw/asr/p633_p23_anestall_20261115_120406.txt`)
**Claimed:** zed-84db451f@amo, 1794778106, expires 2026-11-17T05:30:00Z
(claim scope: the filing + evidence preservation + the 633 plan fold; the FIX (an ANE-substrate change in `riir-infer-gpu`/`riir-infer-ane` or an OS-level workaround) is open for the ANE-lane owner — this issue is the evidence record + the repro recipe.)

## The defect

The plan-633 tvspeech n=100 cell (NeMo OpenThai2.0-ASR fp16 encoder on ANE via coremltools/MLModel, `.raw/asr/p633_p23.py --family tvspeech --per-clip-log`) wedged **twice, deterministically, at the same clip**:

- **Attempt 1** (job 2b7decbe / 01a1dcdb, pid 38793): clips 1-5 logged (per-clip walls 0.4-1.06 s each), then ~2 h 30 m of ZERO progress inside clip 6. Sampled twice in-session (stacks printed to the transcript, not persisted — the sampling lesson is why attempt 3's wrapper carries sample-before-kill).
- **Attempt 3** (job 01a1dd7f / short ref 118363fb, pid 17629): clips 1-5 logged identically (11:39:05 local), stall detected by the fixed watchdog at +25 min (the 1500 s grace + 600 s no-line rule), **stack persisted** before kill: `.raw/asr/p633_p23_anestall_20261115_120406.txt` (254 KB).
- (Attempt 2 was a harness-side watchdog bug — stale mtime baseline killed the process at +6 s during imports; no evidence either way. Recorded in the 633 plan `f14232c4`.)

**Clip 6 of the tvspeech slice is the deterministic wedge point** (audio ~29.4 s — among the slice's longest; attempts 1 and 3 died inside it from independent process lifetimes ~2 h apart on different box states). The fleurs-th family ran GREEN through four n=100 cells the same morning on the same harness/env/box — the trigger is tvspeech-slice-content-specific (likely a clip whose mel/padding shape drives an ANE compiler or driver path that never answers), not a harness or env defect.

## The persisted stack (the evidence)

`MLModel predictionFromFeatures` → `MLE5Engine _predictionFromFeatures:stream:` → Espresso `E5RT::ExecutionStreamImpl::ExecuteStreamSync` → `AneInferenceOperationImplUsingAnefAPIs::EvaluateANERequest` → `[_ANEClient doEvaluateDirectWithModel:…]` → `ANEServicesProgramProcessRequestDirect` → `ANE::ANEServicesDevice::ANE_ProgramSendRequest` → `IOConnectCallAsyncMethod` → **`mach_msg2_internal`** — **2247/2700 samples** of the main thread sit at the kernel IPC send, waiting on an async IOKit reply from the ANE driver that never arrives. Weight distribution in the sample: the entire main-thread call chain is `IOKit send`-blocked; ~45 worker threads idle in `__psynch_cvwait`. This is a **kernel-service-side non-answer** (the ANE driver accepted or dropped the async program request and never called back), not a userspace spin: CPU held ~56-62 % is the sampling + Python overhead, the wedge frame itself is a blocking mach message.

Provenance: process was quiet-box exclusive (PROVENANCE AC, load 5.16 at preflight pass, powermode 0 auto); the two reproductions ran 2 h apart under different poison regimes, same clip.

## Repro recipe (for the ANE-lane owner)

```
cd ~/git/riir-infer
PY=.raw/asr-env/bin/python
$PY .raw/asr/p633_p23.py --family tvspeech --per-clip-log   # wedges at clip 6 within ~6-25 min
sample <pid> 5 -file /tmp/anestall.txt                       # the mach_msg2 wedge stack
```

Minimum facts to narrow: (a) does a solo run of clip 6 alone (a `--family tvspeech --max-clips`-style slice starting at clip 6) wedge WITHOUT running clips 1-5 first — separates "this clip's shape wedges the ANE compile/eval" from "request N after a fixed request count deadlocks" (the counts-then-stops shape would point at an ANEServices request-slot leak); (b) does the fp32 encoder (the P1-era reference graph) wedge on the same clip; (c) `sudo powermetrics` or the ANE telemetry during the wedge (driver-side queue state). The stack names `ANE_ProgramSendRequest(..., bool, ...)` — the bool is plausibly the "isDirect/async" flag; a non-async send variant is the obvious OS-interpretation difference to probe.

## Fix directions (owner/ANE-lane adjudication — none attempted here)

1. **Timeout + retry at the call site** (the 633-lane-bandage shape): run the encoder inference with a watchdog (macOS has no per-call timeout on `MLModel prediction`; wrap in a subprocess with SIGKILL + retry, or an async dispatch with deadline) — unblocks the lane, records the wedge rate, does not fix the driver wedge.
2. **Per-clip process isolation**: one encoder process per clip (or per K clips) — the deadlock seems request-count/session correlated (clips 1-5 always fine, 6 always dead); a fresh MLModel per N requests would sidestep it at process-spawn cost (~4 min NeMo import per process — need the import amortized, e.g. fork after load).
3. **The ONNX/CPU fallback for the wedge class**: the P1.2 ONNX fp lane already runs the same model on CPU at RTFx ~32× — the ANE encoder is a speed lane, not a correctness lane; the fallback posture (probe-detect the wedge, fall back per-clip) is the prod-safe shape.
4. **Radar/feedback to Apple** with the persisted stack (the `ANE_ProgramSendRequest` + `mach_msg2_internal` signature, clip-6 determinism) — owner act.

## Non-goals

- No fix implementation from the filing lane (the ANE substrate belongs to the ANE-lane owner; the 633 plan's remaining tvspeech cell is BLOCKED on this defect and recorded as such).
- The plan-633 tvspeech cell does NOT wait on this fix silently: its next requeue would carry fix direction 1 or 2 as the lane-bandage if the owner wants the cell completed before the deadlock is root-caused.

**Related:** plan 633 (the bench lane; the disposition + reproduction history: `d810c9bd`, `f14232c4`), the fleurs-green control rows (`.raw/asr/p633_p23_cells.jsonl`, 4 green n=100 fleurs cells same morning), Research 029 §7 (the ANE lane's stage record), riir-reflex `scripts/bench_preflight.sh` (the box-state gate that kept the timing rows honest throughout).

**Session:** zed-84db451f, 1794778106
