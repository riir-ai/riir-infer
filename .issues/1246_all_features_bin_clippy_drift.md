# Issue 1246: all-features BIN drift — three bins fail clippy at `--all-features` on develop

**Status:** open — lint-drift cleanup, foreign-bin clippy findings (mechanical)

**Claimed:** zed-b7e21c90@amo, 1795459058, expires 2026-11-26T18:24:18Z

## Finding

Found 2026-11-24 during i1241 T2-prep validation (`05b919d2`): the
`--all-features` posture's LIB compiles green (after the same session's
courtesy fix — `ternary_weights.rs` ternary→dense initializer omitted the
cfg(closed_form_write) `closed_form_overlays` field, E0063, fixed
`05b919d2`), but three BINS fail clippy `-D warnings` at all-features,
verified PRE-EXISTING (none of the three touched by the i1241 lane):

| bin | findings |
|---|---|
| `src/bin/i508_kl_gate.rs` | `cmp_owned` (:57 `teacher == PathBuf::from(DEFAULT_TEACHER)`) + `bool_comparison` (:57 `teacher.exists() == false`) |
| `src/bin/i508_teacher_logits.rs` | `chunks_exact_to_as_chunks` (:173, `chunks_exact(4)`) + 2 more warnings in the same bin |
| `src/bin/issue929_bonsai2_loop_probe.rs` | `useless_borrows_in_formatting` (:742) + `doc_overindented_list_items` |

Repro: `CARGO_TARGET_DIR=/tmp/x cargo clippy --lib --bins --all-features -- -D warnings`
(amo, M5 Ultra, `katgpt-rs` sibling at `e53dbc767`, 2026-11-24 ~01:15 local).

## Why it matters

The all-features bins posture is part of the repo's whole-claim surface
(katgpt-rs AGENTS.md §"The full gate" — check vs clippy is an independent
axis; the same logic binds here). A green lib at all-features does not
cover the bins; these three break the posture and mask any NEW all-features
bin regression behind them (the floor is already red, so one more red is
invisible).

## Fix shape

All mechanical (the listed lints, one bin each; `cargo refine fix` class —
but the rulewright's baseline refused on a contention race during the
discovery run, so manual or retry under a quiet box). No behavior change
beyond the lint-level rewrites; the `i508_*` pair belong to the plan-508
lane's instruments (m3-side lane — courtesy ping before reshaping).

## Acceptance

`cargo clippy --lib --bins --all-features -- -D warnings` green on
develop; no pinned count moves (the bins are not in `scripts/test_gate.py`
floors — verify before claiming otherwise).
