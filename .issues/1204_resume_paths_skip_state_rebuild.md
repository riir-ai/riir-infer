# Issue 1204 — the T3 `--resume` paths decode from a FRESH cache at the cursor (the documented state-rebuild replay loop is missing) — resumed rows are silently corrupt

**Status:** OPEN — `- [ ]` tasks below. Found statically 2026-11-11 ~15:20 +0700 by the i1117 128k restart lane (`zed-95d14db9@m3`) while planning the post-timeout re-queue of the killed `rho=0.30` arm (126,784/131,072, job `e5922317`). Monitor-role discipline held on the lane: the 128k re-run was queued as a FRESH recompute precisely because of this finding — `--resume` was not used.

**Claimed:** zed-m3-i1204@m3, 1794407379, expires 2026-11-12T14:29:40+00:00

## Why

T3 (`5f187da7`, "the durable state dir + --resume") promises in its own commit message and code comments that resume "replays 0..cursor through a fresh cache (deterministic — the state rebuild costs the skipped steps' compute, never their wall)". **The replay loop is not in the code — in either arm.**

**Dense arm** (`crates/riir-infer-gpu/src/bin/flashmemory_decode_gate.rs`, the resume branch at ~477-545): the branch creates a fresh `HybridCache::with_layer_types_bounded` + `HybridForwardScratch::new`, then immediately runs `for pos in cursor..steps` calling `forward_qwen_deltanet_ternary(.., tokens[pos], pos, ..)`. No loop ever feeds positions `0..cursor`; the cache slots `0..cursor` are empty when the first resumed forward attends over them.

**Sparse arm** (the resume branch at ~834-849): `start_pos = cursor` with a fresh `cache` (~815) and fresh `FlashMemoryDecodeState` (~818), then `for pos in start_pos..steps` calls `forward_qwen_deltanet_ternary_flashmemory(.., tokens[pos], pos, ..)` into that empty state. The only "replay" in the function is the comment on line ~835.

**Consequence:** every row at `pos ≥ cursor` in a resumed arm is decoded with zeroed KV/GDN/flashmemory state — the arm believes it is at position `cursor` with an empty context. The rows are wrong relative to the arm's own prefix and wrong relative to the same arm computed fresh. The loader accepts them silently (digest pins inputs, not state); the T4 bars (`|ΔΔNLL| ≤ 0.245`, pruning top-1 gap ≤ 0.131) would be computed over a corrupt tail. The stale in-code comment at ~884 ("a restart recomputes the arm — resume covers completed bins only") states the SAFE pre-T3 behavior and contradicts the shipped T3 code path directly above it.

**Why T3's hermetic cell missed it:** the 600-step cell verified "RESUME from cursor 320/600 (320 rows recovered)', both arms completed, report written" — completion, never row-parity against a fresh full run.

**Blast radius today:** nothing consumed a resumed bin yet — this box's killed ρ=0.30 prefix (97%) was never resumed (the restart recompute is queued instead; prefix preserved at `rho_0.30.prefix97.bin` on bmo). Cross-repo: `riir-infer` Issue 1200's "Resume anchor for the post-fix re-run … `--resume` continues from it once T1+T2 land" recommendation rides this path — the post-fix Metal re-run would resume from the 03:34 snapshot with the same corrupt-tail shape. That anchor is unusable until this issue lands.

**Cost note (honest):** a CORRECT resume saves no wall time — the state rebuild replay costs `cursor` forwards either way, so `replay + tail == full recompute` (131,072 forwards). Resume's only legitimate value is preserving recovered rows across repeated kills and the wall_secs bookkeeping. Any fix note should carry that, so nobody fixes the bug expecting a speedup.

## Tasks

- [ ] T1 — hermetic row-parity reproducer (small `--ctx`, e.g. 600): fresh full run → bins; truncate `rho_0.30.bin` to a `complete=0`, `n=100` prefix (header edit: byte 44 = 0, bytes 45..53 = 100); `--resume` run; assert rows `100..600` vs the fresh run's rows — expected MISMATCH proves the bug; then add the replay loop (dense + sparse: `for pos in 0..cursor { forward, discard rows }`) and assert rows byte-identical post-fix
- [ ] T2 — fix the stale comment at ~881-888 ("resume covers completed bins only") to describe the fixed behavior, and the `wall_secs` accumulation note
- [ ] T3 — re-anchor Issue 1200's "resume anchor" paragraph to this issue (the Metal post-fix re-run must not resume until T1 lands)
- [ ] T4 — 128k lane note: the bmo restart job (`01a1c9ac`, exclusive) runs the fresh recompute; `rho_0.30.prefix97.bin` is post-hoc collection value only, never loader input
