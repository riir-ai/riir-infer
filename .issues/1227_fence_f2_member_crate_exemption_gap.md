# Issue 1227: fence F2 flags the workspace's own member crates — the exemption list predates them

**Status:** Open — the gate is red on every workstation run; CI is main-only so `develop` never sees it (the spending-limit posture that hides it).
**Claimed:** amo-i1227, 1794870995, expires 2026-11-18T23:16:24Z

## The defect

`scripts/fence_gate.py` F2 (the import fence) reports **144 UNDEFENDED findings on develop** — every run since the member crates landed:

| token | findings | where |
|---|---|---|
| `riir_infer_dist` | 120 | the Plan-623 dist lane (`c4d8f40c`, 2026-10-27) |
| `riir_infer_ane` | 17 | plan 619 T1.0 (`2ca7f303`, 2026-10-15) |
| `riir_infer_audio` | 4 | issue 015 T1-T3 (`7da24f85`, 2026-10-23) |
| `riir_infer_clef` | 3 | plan 619 (`63cb9b6c` era) |

The gate's own exemption regex predates every one of them:

```python
FOREIGN_RIIR = r"riir_(?!infer_core\b|infer_gpu\b|infer_laya\b)[a-z0-9_]+"
```

`riir-infer-laya` was added when the encoder lane landed (`c6716a46`, 2026-09-24); `riir-infer-ane` (2026-10-15), `riir-infer-clef`, `riir-infer-audio` (2026-10-23) and `riir-infer-dist` (2026-10-27) followed — each landed its crate into the root `[workspace] members` (all six are members today) without widening this regex. The gate file last changed functionally 2026-09-24; the workspace membership list moved four times since.

## Why this matters (the fence's own law)

- F2's purpose: the repo is **upstream of every engine by design** — no code position may reference an ENGINE crate (`riir_engine`, `riir_gpu`, `riir_games*`, `riir_router`, `riir_chain`, `riir_neuron_db`, ...). The module headers legitimately SAY "NO cognition" in prose; the gate enforces the no-dep direction.
- The member crates are this repo's OWN leaves (same class as `riir_infer_core`/`riir_infer_gpu` — self-references are the point of the repo). Their internal references are not boundary crossings; flagging them is exactly the "fence that cries wolf is a fence nobody runs" failure the doc comment names.
- `fence_expected.txt` is **deliberately EMPTY by design** ("a row reading 'not written yet' would be a backlog wearing a pin") — so 144 pins would be 144 lies, not a fix. The regex is the fix.
- CI evidence it was GREEN once: Issue 023 closed 2026-09-28 with `✓ PASSED — 0 undefended, 0 pinned` (`61ebf9ce`). At `origin/main` (2026-10-03, the last CI-armed tip) the gate still reads green on that tree. Every defector crate landed AFTER the last main push — `develop` is 1,670 commits ahead and CI is main-only (spending-limit posture, `74850b5f`), so the red has been invisible to CI the whole time. This is the rot-between-two-lanes class: workstation-only gates drift while the CI arm is parked.

## The fix (one line + the floors)

```python
FOREIGN_RIIR = r"riir_(?!infer_core\b|infer_gpu\b|infer_laya\b|infer_ane\b|infer_clef\b|infer_audio\b|infer_dist\b)[a-z0-9_]+"
```

Better (drift-proof): derive the exemption from the root `[workspace] members` — parse the member dirs' `[[package]] name` fields and build the negative lookahead from the measured set, so the next member crate cannot re-red the gate silently (the Issue-837 lesson: registering a repo is a multi-file operation and one list always rots). The regex-derived-from-members form keeps the gate honest by construction; the hand list is the fallback if the parse floor fails loud.

Floors to add while here: the findings count at a green posture is 0 — pin `max_undefended = 0` (already implied by exit code) and a walk floor bump: MIN_RS_FILES 20 → stays (467 measured), MIN_TOMLS 1 → stays (16 measured). No floor changes needed; the fix is the exemption set only.

## Verification

- `py scripts/fence_gate.py` → `✓ PASSED — 0 undefended, 0 pinned` (selftest still fires its 4 planted violations — the engine tokens `riir_engine`/`riir_gpu` stay foreign).
- A planted `use riir_infer_dist::...` in a NON-member context is impossible (dist IS a member) — the F1 path-dep fence still catches real escapes (the gate's other half, untouched).
- Negative control: `riir_trainer`/`riir_engine` must still fire (the selftest asserts this).
