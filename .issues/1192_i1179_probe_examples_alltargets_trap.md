# Issue 1192 — the i1179 probe examples break `--all-targets` at the DEFAULT posture (the green-zero/required-features trap class)

**Status:** CLOSED FIXED (2026-11-09) — 23 verified compile errors at `cargo clippy -p riir-infer-gpu --all-targets` (default features), found by the 1191 T1 lane (amo-zed-1191metal@amo, `ce10a19`), independently re-verified on m3 (`/tmp/plan1191`, 2026-11-09). PRE-EXISTING (not the 1191 lane's; the lane's own postures are clean). Docs-law: riir-infer AGENTS "a `#![cfg]`-gated file trap" + the required-features row discipline.

**Claimed:** amo-zed-1191metal@amo, 1794221900, expires 2026-11-11T11:20:00+00:00

## Why

Cargo auto-discovers `examples/*.rs` with no `[[example]]` row and compiles them at whatever features the invoking command carries. Ten `crates/riir-infer-gpu/examples/*_i1179.rs` probe/dump files have NO manifest rows, so every `--all-targets` run at a posture lacking their features fails:

- **E0432/E0433 (9 files):** `probe_{realctx,draft,trace,myl0,mycpu,attnmirror}_i1179.rs`, `dump_l0_pieces_i1179.rs`, `probe_feat_i1179.rs` import `riir_infer_gpu::qwen38_specdec` / `qwen38_dense_cubecl` / the `half` crate — gated modules and a dev-dep that the default posture does not compile.
- **E0601 (1 file):** `probe_realctx_i1179.rs` has no `main` (a fragment).

The verdict consequence is the repo's own trap: a gate that names `--all-targets` at the default posture cannot be GREEN on this package today, and nobody notices because no pinned suite runs it that way.

## Tasks

- [x] T1 — Add `[[example]]` rows with `required-features` for all ten `*_i1179.rs` files (the gated-import nine: their real feature sets, e.g. `qwen38_specdec` / `qwen38_wgpu`-family / the `half`-bearing set; the no-main fragment: either its true feature row or convert to a `#![cfg]`-gated doc/example). Verify: `cargo clippy -p riir-infer-gpu --all-targets` GREEN at default AND at the armed postures of every row added. **DONE: 10 rows (7× qwen38_specdec: dump_l0_pieces, probe_attnmirror, probe_draft, probe_mycpu, probe_myl0, probe_trace, probe_realctx; 1× qwen38_wgpu: probe_feat; 1× cubecl_runtime: make_mini_target; 1× qwen38_specdec: dump_myrope — the sweep's own correction: its specdec call is mid-file, not a top-level use, and the first enumeration missed it). Acceptance MEASURED: --all-targets 0 errors + 0 warnings at default, qwen38_specdec, qwen38_wgpu, cubecl_runtime. EN-ROUTE FIXES (the rows exposed files never compiled at their armed posture): probe_attnmirror's two closures both mut-captured the File (E0499 — RefCell interior mutability, 6 read sites through the cell) + a file-level #![allow(dead_code)] (the offline instrument's geometry consts/fns are the recorded mirror); probe_draft gained its own file-level allow + as_chunks::<5120> (the chunks_exact lint); probe_feat's unused dot dropped; dump_l0_pieces' unneeded mut dropped.**
- [x] T2 — Run `scripts/required_features_build_audit.py ../riir-infer --grep i1179` (the row-must-build half): a row that names a set the example cannot build with is worse than none. `--batch` the exact sets. **DONE: 10/10 rows BUILD at their own sets (0 FAIL / 0 NO-FEAT / 0 UNSEEN), run under uv python 3.12 (the box's system py 3.9 lacks tomllib).**
- [x] T3 — Sweep for SIBLINGS of the class: any other tracked `examples/*.rs` / `benches/*.rs` with no manifest row whose imports are feature-gated (`grep -L 'name = ' is not enough — diff the directory listing against the manifest's target rows`). Fix or pin, with a reason per row. **DONE: the sweep (directory-vs-manifest diff, every example/bench file across riir-infer-gpu + the sibling repos' example/bench dirs) found ZERO rowless files left in this crate (13/13 declared) and ZERO TRAPS in the siblings — riir-ai (19 rowless files), riir-train (14), riir-neuron-db (1), riir-chain (1) all import only UNGATED modules (harmless auto-discovery; the gated-import-vs-declared-rows check per file). No cross-repo filings needed — the class was unique to the i1179 probes.**
- [x] T4 — Record + close: HISTORY row; if the sweep finds the class live in sibling repos, file there per boundary (do not fix across repos silently). **DONE — CLOSED FIXED: the HISTORY row carries the arc; the sibling sweep found the class empty (recorded above), no cross-repo filings.**

## Non-goals

- No behavior change to the probes themselves (they are offline i1179 instruments).
- No new gate wiring in this issue — if a `--all-targets`-at-default lane SHOULD exist, that is a separate decision.
