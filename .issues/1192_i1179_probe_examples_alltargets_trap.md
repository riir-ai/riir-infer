# Issue 1192 — the i1179 probe examples break `--all-targets` at the DEFAULT posture (the green-zero/required-features trap class)

**Status:** OPEN — 23 verified compile errors at `cargo clippy -p riir-infer-gpu --all-targets` (default features), found by the 1191 T1 lane (amo-zed-1191metal@amo, `ce10a19`), independently re-verified on m3 (`/tmp/plan1191`, 2026-11-09). PRE-EXISTING (not the 1191 lane's; the lane's own postures are clean). Docs-law: riir-infer AGENTS "a `#![cfg]`-gated file trap" + the required-features row discipline.

**Claimed:** amo-zed-1191metal@amo, 1794221900, expires 2026-11-11T11:20:00+00:00

## Why

Cargo auto-discovers `examples/*.rs` with no `[[example]]` row and compiles them at whatever features the invoking command carries. Ten `crates/riir-infer-gpu/examples/*_i1179.rs` probe/dump files have NO manifest rows, so every `--all-targets` run at a posture lacking their features fails:

- **E0432/E0433 (9 files):** `probe_{realctx,draft,trace,myl0,mycpu,attnmirror}_i1179.rs`, `dump_l0_pieces_i1179.rs`, `probe_feat_i1179.rs` import `riir_infer_gpu::qwen38_specdec` / `qwen38_dense_cubecl` / the `half` crate — gated modules and a dev-dep that the default posture does not compile.
- **E0601 (1 file):** `probe_realctx_i1179.rs` has no `main` (a fragment).

The verdict consequence is the repo's own trap: a gate that names `--all-targets` at the default posture cannot be GREEN on this package today, and nobody notices because no pinned suite runs it that way.

## Tasks

- [ ] T1 — Add `[[example]]` rows with `required-features` for all ten `*_i1179.rs` files (the gated-import nine: their real feature sets, e.g. `qwen38_specdec` / `qwen38_wgpu`-family / the `half`-bearing set; the no-main fragment: either its true feature row or convert to a `#![cfg]`-gated doc/example). Verify: `cargo clippy -p riir-infer-gpu --all-targets` GREEN at default AND at the armed postures of every row added.
- [ ] T2 — Run `scripts/required_features_build_audit.py ../riir-infer --grep i1179` (the row-must-build half): a row that names a set the example cannot build with is worse than none. `--batch` the exact sets.
- [ ] T3 — Sweep for SIBLINGS of the class: any other tracked `examples/*.rs` / `benches/*.rs` with no manifest row whose imports are feature-gated (`grep -L 'name = ' is not enough — diff the directory listing against the manifest's target rows`). Fix or pin, with a reason per row.
- [ ] T4 — Record + close: HISTORY row; if the sweep finds the class live in sibling repos, file there per boundary (do not fix across repos silently).

## Non-goals

- No behavior change to the probes themselves (they are offline i1179 instruments).
- No new gate wiring in this issue — if a `--all-targets`-at-default lane SHOULD exist, that is a separate decision.
