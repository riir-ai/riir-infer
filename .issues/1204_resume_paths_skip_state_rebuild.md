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
  - REFINED 2026-11-11 (zed-m3-i1204): the replay loops are IN (see Lane state) — the remaining T1 half is running the cell with the PRE-FIX binary first (mismatch proof), then the fixed binary (parity proof). The pre-fix binary must be built from a tree WITHOUT the fix (stash the edit, build, `cp target/release/flashmemory_decode_gate /tmp/i1204cell/gate_prefix`, pop, rebuild → `gate_fixed`). Cell recipe:
    ```sh
    # corpus: any plain text ≥ 601 tokens (e.g. head -c 16384 of a repo .md concat)
    G=/Users/kat/git/riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf
    C=/tmp/i1204cell/corpus.txt
    # 1. fresh posture (both binaries give identical fresh rows — same digest):
    $GATE --gguf $G --corpus-file $C --ctx 600 --max-steps 600 --rho-list 0.30 \
      --bucket 100 --block 32 --refresh 32 --out /tmp/i1204cell/fresh
    # 2. truncate fresh bins to cursor=100 prefixes (byte 44 = 0; bytes 45..53 = 100 LE;
    #    truncate file to 69 + 100*12 bytes) into /tmp/i1204cell/res/
    # 3. resume run into /tmp/i1204cell/res — pre-fix: rows 100..600 MUST mismatch fresh;
    #    post-fix: rows byte-identical AND rho arm's mean_selected_fraction identical
    #    (the replay re-runs the retain_every sampling, so the stat is recomputed, never estimated)
    ```
  - CPU posture only (no `--metal`) — the cell must not contend with the GPU measurement lanes.
- [x] T2 — fix the stale comment at ~881-888 ("resume covers completed bins only") to describe the fixed behavior, and the `wall_secs` accumulation note
- [x] T3 — re-anchor Issue 1200's "resume anchor" paragraph to this issue (the Metal post-fix re-run must not resume until T1 lands) — REFINED: the re-anchor now states the STRONGER truth (resume is CPU-posture-only, saves no wall time, and the Metal re-run runs FRESH — the anchor snapshot is post-hoc collection value only)
- [x] T4 — 128k lane note: the bmo restart job (`01a1c9ac`, exclusive) runs the fresh recompute; `rho_0.30.prefix97.bin` is post-hoc collection value only, never loader input — recorded below in Lane state

## Lane state (zed-m3-i1204@m3, 2026-11-11 ~15:00Z)

- **The fix is WRITTEN but not yet compile-gated or cell-validated** — it sits in the working tree of the shared m3 checkout (commit `wip:` checkpoint pushed):
  - Dense CPU resume branch (`flashmemory_decode_gate.rs`, the `else if args.resume && !args.metal …` arm): replay loop `for pos in 0..cursor` through the fresh cache, rows recomputed + VERIFIED against the prefix (loud `bail!` on mismatch — the prefix came from a different build/posture), then the existing tail loop.
  - Sparse CPU resume branch: same replay through fresh `cache` + `FlashMemoryDecodeState`, rows recomputed + verified, and the `retain_every` selection sampling RE-RUN over the replayed window (the inline sampling block became the shared `sample_selection` closure — single home, both callers) so a resumed arm's `mean_selected_fraction` matches a fresh run's exactly — the old `selected_sum = prefix.mean_selected_fraction × cursor` estimate is GONE.
  - **Metal-posture finding (beyond the filed gap):** the dense resume branch sat BEFORE the Metal branch in the else-if chain — a `--metal --resume` run would HIJACK the dense arm onto the CPU forward (mixed posture vs the Metal rho arms). Fixed: the resume branches now gate `!args.metal`; `--resume` under `--metal` prints one disclosure line and every arm recomputes from 0. The Metal sparse arm never consulted `resumed` (it always recomputed from 0 — safe, just not resumable); unchanged.
  - Module doc + the per-bucket-prefix comment + the `wall_secs` accumulation comment updated (T2) to the fixed behavior: resume buys recovered rows across repeated kills, never wall time.
- **Deferred with reason:** the T1 cell + compile gate — m3 went to 19% battery discharging under load 120 (the i1117 128k Metal trio mid-flight, ~10-12 h lanes); a release build + 4× 5 GB model loads + ~600 forwards × 4 runs do not fit the battery window and would race the sibling measurement for the box's last hour. Run the cell on power or on a quiet box (the recipe above; lease live to 2026-11-12T14:29Z). The `rustfmt --check` parse gate is clean; the compile gate RIDES the cell's release build (do not claim the fix until both cell arms pass).
