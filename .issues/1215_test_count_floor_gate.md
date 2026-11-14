# Issue 1215 — the riir-infer standing test-count gate (the EXECUTE half; test_gate.py)

**Status:** OPEN — T1 LANDED (`42cc99b`), T2/T3 open. This issue converts the count-floor-infra deferral (noted in plan 632 T2's adjudication) into the standing lane.

**Claimed:** zed-41cd0608@amo, 1794597600, expires 2026-11-16T12:00:00Z

## Why

katgpt-rs landed the repo's own EXECUTE-vs-COMPILE split (its `test_gate.sh`, Issue 718 T3(b); the durable record is katgpt-rs `.docs/10_audits/ci_compile_vs_execute_axis.md`) after measuring that its whole CI surface compiled everything and EXECUTED almost nothing — every uninvoked assertion was *unknown*, not passing. riir-infer has the same shape and no gate: nothing here pins a passed-COUNT. The concrete green-zero class is measured in-repo tonight: `cargo test -p riir-infer-laya --features laya-riir --lib int_ref` prints `ok. 0 passed` exit 0 because the `int_ref` FEATURE row is missing (the whole `#![cfg(feature = "int_ref")]` module compiles to nothing) — byte-identical to a pass. Plan 632 T2 formalized the int_ref determinism suite but had no standing instrument to keep its count honest; the repo-birth discipline (a `#![cfg]` target lands with its `required-features` row the same commit) is per-target, and nothing catches the suite-level count afterwards.

## Landed shape (T1)

`scripts/test_gate.py` — stdlib-only Python (3.9-parseable), ASCII output, encoding-safe (bytes + explicit utf-8 decode; no `text=True`), no network. The katgpt-rs/riir-train law set, scoped:

- **Floors fire DOWNWARD only** — adding tests never reds; deleting tests, a feature change that compiles a suite to nothing, or a broken build reds.
- **One `test result:` line per invocation** — more or fewer is a blind parse (covers the SKIP-silent and green-zero shapes; a receipt-row SKIP kills the binary before any result line, which the law reds).
- **`--canary`** runs the first row at floor 100000 and MUST fail (the floors-live proof; verified firing tonight).
- **Metal row skips LOUD off macOS** (a deferral, never a green).
- **`--receipt`** opts into the slow row (~31 min release posture; the G3 eval + the receipt-hash prefix `22d4844a` as the loud-zero defence — a passed-but-no-hash run reds).

Standing rows (floors measured 2026-11-14, amo AC, load 7.8–9.6, develop `086a2bb`):

| row | pkg | features | target | filter | floor |
|---|---|---|---|---|---|
| int_ref CPU core | riir-infer-laya | `laya-riir,int_ref` | --lib | int_ref | 67 |
| int_ref G1(c) GPU twin | riir-infer-laya | `laya-riir,int_ref,laya-riir-metal` | --lib | int_ref | 70 |
| int_ref python mirror | riir-infer-laya | `int_ref` | --test int_ref_python_mirror | — | 8 |

Slow row (--receipt): `int_ref_real_artifact` at `--release`, floor 2 + the receipt-hash prefix (green 2026-11-14, 1892.89 s, bit-identical to bench 1213's pins).

## Tasks

- [x] T1 — the gate script + the int_ref rows + canary + receipt row; all four postures verified live tonight (canary fires, 3 standing rows green, receipt row green at release). LANDED `42cc99b`.
- [x] T2 — row-widening DONE (2026-11-14, same session): four CPU-only suites joined with measured floors — issue879 quant certification (2, release; the two reduced-geometry arms — the other two are `#[ignore]`d CPU-heavy/production-artifact arms), bonsai2 rotation load (11, 0.12 s), issue046 spectral class spec (4, 0.05 s), exl3 lib lane (308, 0.58 s). Full-gate 7 rows green + canary re-verified firing. **The entropy_bounded_commit row is deliberately ABSENT** (this is a recorded deferral, never a silent skip): the upstream `07acd887b` `decoding` umbrella (Plan 635 T2) + the cross-crate cfg mismatch made the lane non-compiling for 3 days (fixed upstream katgpt-rs `14435d80e` 2026-11-14 — the enum-variant cfg vs the match-arm cfg disagree across katgpt-core/katgpt-speculative; the fix ungates the VARIANT + the arm, the ECHO signal stays gated at its producer), and after the fix the suite still cannot floor on this shared box — the heavy gemma2_2b tests hold ~8.5 GB host + ~12 GB device per instance (the issue-712 gate serializes them) and the process SIGKILLs under the box's memory pressure (10.9 GB swap in use with Zed + Docker co-resident). Its 6-test EB battery passes standalone (176 s release). The row joins when a dedicated GPU box runs the gate (bmo/4090 posture) or the heavy tests get a host-memory guard symmetric to the VRAM one.
- [ ] T3 — CI wiring decision: the katgpt-rs precedent runs its gate weekly (schedule) + on push (main-only). riir-infer's CI is main-only/dispatch (spending posture); either wire the standing rows into a dispatch-only workflow or record the workstation-owned decision here (the gate refusing off-macOS for the Metal row keeps a runner honest).
- [ ] T4 — the green-zero repair observed en route is its own mini-audit: grep the repo's Cargo.toml `[[test]]`/`[[bin]]` rows against the `mod`-gated surfaces they name (the class tonight: a feature-gated module whose suite needs BOTH the module feature and the row) — file any additional missing rows found.

## Non-goals

- No `cargo hack --each-feature` sweep here (the riir-ai feature matrix lesson — priced, not run).
- No timing bars: this gate pins COUNTS only (the latency bar class is the benches' job).
