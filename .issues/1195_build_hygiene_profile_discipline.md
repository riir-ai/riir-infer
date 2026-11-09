# Issue 1195 — build hygiene: the root manifest carries no `[profile]` sections (dev full-debuginfo since birth, no dist posture)

**Status:** OPEN — found by the Protocol-H build-hygiene quick pass (housekeeping cycle 2026-11-09, zed-dae2c913@amo; report-only, filed not fixed per the step-5 discipline). Pre-existing since repo birth (`git log -S '[profile' -- Cargo.toml` is EMPTY — profiles never landed here), while the workspace standing discipline (build-hygiene: dev line-tables-only + opt-3 deps, dist fat-LTO strip) is implemented in sibling roots (katgpt-rs root carries 4 `[profile]` sections).
**Claimed:** amo-zed-1191metal@amo, 1794260544, expires 2026-11-11T21:45:00+00:00

## Why

`/Users/katopz/git/riir-infer/Cargo.toml` (the root-package workspace manifest) has zero `[profile.*]` rows — the only "profile" hits are comments about the unrelated `twt_profile` feature. Consequences, measured on the boxes that build this repo many times a day:

- **Dev builds emit FULL debuginfo by default.** Live specimen at filing time: the 1191 T2 debug test binary (`/tmp/i1191/debug/deps/flashmemory_budget_metal_t2-*`) holding 8.6 GB RSS on amo; the standing incremental-cache bloat class (~30 GB when dirty) that prior boundary rows keep LIST-ing. Every `cargo test --lib` / T2-prep debug compile+link on amo/bmo pays the full cost.
- **No dist posture.** The league/bench lanes run `--release` with default release settings; nothing pins a strip/fat-LTO shipping posture the way riir-reflex's `[profile.dist]` does. Not urgent (no shipped binaries yet), but the lane is missing where the convention expects it.

## Constraints (read before T1)

- ⛔ **The RELEASE posture is the LEAGUE posture** — `[profile.release]` opt levels / LTO / codegen-units are what every Bench 8xx–9xx number was measured under. NEVER touch the release half without a pre-registered A/B and the GOAT/owner gate; this issue scopes T1 to the DEV half only.
- Feature-unification note: `cargo` profiles are workspace-root-scoped; a `[profile.dev]` change here also applies to path-dep sibling builds driven from THIS root (`../katgpt-rs` crates compile at whatever profile the invoking root picks). Dev-only debuginfo reduction is behavior-neutral (line tables, no panic/assert change) — but assert it, don't assume it (T3).

## Tasks

- [ ] T1 — Adopt dev line-tables-only in the root manifest: `[profile.dev] debug = "line-tables-only"` (+ `[profile.dev.package."*"]` opt-level 3 for deps per the build-hygiene law, judged here since this root drives the whole closure). Verify default `cargo test -p riir-infer-core --lib` still passes and the debug binary size / target growth drops.
- [ ] T2 — Pre-registered A/B for T1 (the build-hygiene bar): on a quiet box state (quote the rig status line), time one clean `cargo build -p riir-infer-gpu --lib` + one `cargo test --lib` link, before vs after the profile row, same `CARGO_TARGET_DIR`. Record the delta in the closing HISTORY row.
- [ ] T3 — Decide the dist posture (report/owner-gated, not implemented by default): whether riir-infer needs a `[profile.dist]` mirror of the reflex convention before any binary ships. File the verdict either way; do not silently skip.
- [ ] T4 — Record + close: HISTORY row with the A/B numbers; fold this file.

## Non-goals

- No release-profile change (league posture is frozen until a bench says otherwise).
- No CI wiring in this issue.
