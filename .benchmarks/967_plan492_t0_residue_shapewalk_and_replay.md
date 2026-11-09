# Bench 967 — plan 492 T0-residue (a)+(b): the 959 shape-walk re-run on the FULLY fixed forward + the fixture-level vendor replay re-run (both PASS; the chain moved again)

**Status:** DONE — plan 492's T0-residue landed. (a) the shape-walk re-run on
the fully fixed forward (riir-infer `a844514d`, post-683/1196): chain
`[35446, 11380, 107231, …]` conf `1.000/1.000/0.587` (7 rows) — the numbers
moved again off 959's `201633/143447/149688` conf `1.000/0.891/1.000`, exactly
as the residue expected (959's re-measure rode the 1193 gate fix only; 683's
three backward fixes + 1196's gemv staircase landed after). (b) HALF-LANDED:
the fixture-level vendor-MLX replay re-ran GREEN end-to-end (12/12 PASS, rc=0)
on the CURRENT oracle's regenerated fixtures — the fixed oracle and the vendor
reference still agree inside every stated tolerance class. The main-path
golden piece (`draft_block` vs `DSpineCodeBridge.__call__` END-TO-END — the
1193-lesson pin that would have caught the gate at birth) is NOT in this
record: it is a replay-script extension (the piecewise script has no
main-path arm), filed as the residue's open half in plan 492.

## Cell (a) — the shape walk (real checkpoint, the fixed forward)

- **Host:** amo (M5 Ultra 96 GB, macOS, quiet — load 3.8/30; the walk is the
  CPU oracle lane; no GPU consumer). 2026-11-10.
- **Bin:** `qwen38_dspine_t2` (default mode) at a detached worktree of
  origin/develop `a844514d`; the four forward commits between 959's re-run
  and this cell: `17f5d9f2` (the trace refactor — parity-pinned bit-exact),
  `d8ee5be8`, `1f0ec285` (the 1193 gate fix — ALREADY in 959's re-run),
  `eaf7eee9` (the 1196 plain-aware gemv staircase).
- **Checkpoint:** `tencent/Qwen3.8-27B-DSpine-b16`, blake3
  `53d185164e20b6ea51fd72a3f093492aff1596a7dda0693d45c7d057dc1cc15c` —
  verified at load (the pin held).
- **Posture:** the harness's own synthetic-capture stand-in (block_size 8 →
  γ = 7 rows; amplitude-0.02 captures; the blake3 stand-in embed table) —
  the SAME posture 959 ran; the walk validates plumbing + invariants on the
  REAL head weights, not acceptance.
- **Verdict: PASS** — chain tokens ∈ candidates (7/7), confidences ∈ (0, 1]
  (7/7), timing: tables 54.1 s (TG 485 MB, TSV 970 MB), walk 67.0 s total.

**The chain (this is the correction-of-record row):**

| record | forward | chain (first 3) | conf (first 3) |
|---|---|---|---|
| Bench 959 (pre-959-fix) | hyperbola gate | `233648/25157/149688` | `1.000/1.000/1.000` |
| Bench 959 (the 1193 fix) | sigmoid gate, pre-683 | `201633/143447/149688` | `1.000/0.891/1.000` |
| **this bench** | **fully fixed (683 + 1196)** | **`35446/11380/107231`** | **`1.000/1.000/0.587`** |

Honest note: the walk is a synthetic-capture plumbing cell (the harness's
own disclosure — "the walk validates plumbing + invariants, not
acceptance"); the moving chain numbers prove the forward changed and the
invariants hold — they are NOT a quality metric. The quality row of record
stays Bench 965's mean_j 2.565 (the real-capture posture).

## Cell (b, half) — the fixture-level vendor replay re-run

- **Fixtures regenerated** by the CURRENT oracle
  (`qwen38_dspine_t2 --fixtures /tmp/dspine_fixtures_v2`; the degeneracy
  guard alive: `distinct_idx=7 probs [0.2447, 0.6932]` — discriminating).
- **Replay:** `scripts/dspine_golden_replay.py` (the vendor pin `6450af7` at
  `~/git/riir-infer/.raw/dspine`, its `.venv`, MLX 0.32.4) on amo.
- **Verdict: 12/12 PASS, rc = 0** — piece (a) rms max-abs 3.1e-6 (the fp32
  reduction-order class), codes 11/8192 ≤ 2 ULP; piece (b) message 0/4096
  bit-exact, out/normed ≤ 1 ULP; piece (d) TG 322/2097152 ≤ 1 ULP, TSV
  282/4194304 ≤ 1 ULP, **idx 0/112 mismatches, tokens 0/7 bit-exact**,
  probs/confs max-abs ≤ 1.9e-4 (the score-ULP class); the argmax-tie
  sentinel PASS. **The current oracle and the vendor reference still agree
  inside every stated class after 683/1196** — the backward fixes did not
  move the FORWARD's vendor parity (they never should; the forward's gate
  and gemv fixes are IN both of these runs' classes and stayed inside).

## What remains (the residue's open half — stays in plan 492)

The **main-path golden piece**: `draft_block` (the real forward, one call)
vs `DSpineCodeBridge.__call__` end-to-end. The piecewise replay is
structurally blind to ops implemented BOTH as a piece and inline (the 1193
lesson — the gate was inline in the main path and mirror-only in the
piece). The replay script needs a new arm (drive the piece inputs through
`draft_block`'s single-block path and diff the vendor `__call__`'s
end-to-end outputs). The chain numbers above ARE a main-path execution
(the t2 bin drives `draft_block`), but no VENDOR end-to-end diff rode them —
the parity of the main path vs the vendor is pinned only piecewise plus the
T5 G1 serial==specdec gate (Bench 965). Filed in plan 492's T0-residue row
as the remaining `- [ ]`.

## Provenance

- Cells ran over SSH on amo; box state amo (load 3.8/30, quiet, AC per rig
  status); the runs are CPU-lane correctness cells — no timing claim rides
  them (the s/step numbers quoted are the walk's own wall-clock disclosure,
  not a benchmark bar).
- The worktree mirror was removed after the runs (the sister path-dep
  symlink `/tmp/katgpt-rs` with it); the fixtures
  (`/tmp/dspine_fixtures_v2`) stay on amo until plan 492's main-path piece
  consumes them.
