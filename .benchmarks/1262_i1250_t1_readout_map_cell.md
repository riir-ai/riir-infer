# Bench 1262 — Issue 1250 T1: the static readout map cell (Qwen3.5-0.8B-Base-Q8_0)

**Status:** T1 REAL CELL LANDED 2026-11-25 (zed-m3-i1250, m3 idle lane). Weights-only
logit-lens readout map over every down_proj column, sign-split — the
strictly-cheaper tier in front of `census_screen` (forward-pass-based) and the
1241 interventional census (557 CPU-h): v0 costs ONE pass, once per checkpoint.
Issue: riir-infer `.issues/1250` T1. Lane module: `src/readout_screen.rs`
(feature `readout_screen`, bin `readout-screen`). Source: katgpt-rs Research 693
(the Anthropic Fellows pair, alignment.anthropic.com/2026/neuron-basis-theory).

## PROVENANCE

- Box: m3 (Apple M3 Max, 16 cores, 64 GiB), AC power, battery 100% charged.
- Contention: the lane's own rayon run + one interactive editor (Zed, exempt);
  the map_grammar_lozo2 compute that overlapped the FIRST run finished before
  it; the G4 re-run overlapped nothing heavier than the editor.
- Build: `--release`, isolated `CARGO_TARGET_DIR=/tmp/i1250-target`.
- Runtime (first pass, report rendered with token strings): **1584.5 s**
  (~26.4 min) for 172,032 units × 248,320-vocab dot — one dequant of the tied
  table (1.016 GB f32) + per-layer column materialization (14.7 MB, dropped per
  layer). Runtime is a disclosure, not a gate — T4 carries the GOAT legs.

## The cell

```
readout-screen --gguf ../riir-train/data/Qwen3.5-0.8B-Base-Q8_0.gguf \
  --out /tmp/i1250_readout_map.rdmap
```

- Model: qwen35-hybrid, 24 layers, n_ff 3584, n_embd 1024, vocab 248,320,
  TIED unembedding (`token_embd.weight`; no `output.weight` — the source label
  records it), rms_eps 9.999999974752427e-7 (read from the file's metadata).
- Units: 24 × 3584 × 2 signs = **172,032** rows; artifact 15,310,919 bytes
  (71-byte header + 172,032 × 89-byte rows, exact).
- Artifact: `/tmp/i1250_readout_map.rdmap` — **not committed** (14.6 MB binary;
  the digest + report are the record; regenerate with the one command above).
- **file_digest_blake3 = 70fd9c83d63454ae72e07a29ebdb78f863cfad102d8939f6ec49b2aea32104e2**
- **artifact_digest_blake3 = c508d9170875cd60747fa622818c269e565339c665f72c388ecb48f76ae0e56f**
- G4 byte-identity: the re-run digest recorded below (same command,
  `--no-strings` — the flag affects the REPORT TEXT only, never the artifact
  bytes).

## Per-band rollups (the T4/T5 discrimination input)

| band | n | sharp med | sharp mad | sharp max | margin med | margin max | readable |
|---|---|---|---|---|---|---|---|
| early | 28,672 | 12.130 | 0.949 | 39.62 | 0.445 | 8.71 | 11,340 |
| middle | 28,672 | 12.353 | 1.089 | 58.67 | 0.470 | 21.02 | 13,231 |
| late | 28,672 | 13.499 | 2.191 | 91.25 | 0.686 | 50.44 | 18,438 |

- Sharpness = top-1 lens logit; margin = top1 − top2; readable = units at/above
  the checkpoint-wide sharpness median (12.5128).
- **The gradient reads** (the pair's own finding, visible in OUR checkpoint):
  the late band is the strongest — highest median sharpness, widest margins,
  and 18,438/28,672 units above the global median vs 11,340 early. Middle-band
  readouts sit between — T5's upgrade trigger ("middle-band readouts too weak
  rather than the screen failing") is NOT met by v0: no band collapsed.
- bucket v1 = UNASSIGNED for every row — T2's J-share NNLS pass fills the
  high/middle/low classification and the per-band bucket rollup.

## Readability spot-check (the lane's liveness evidence)

Top units read as coherent token families (the paper's readout behavior):

- l23 c414: ` such / such / Such / Such / 这样的` — the "such" family.
- l23 c1116: `<|endoftext|> / <|im_end|> / <|file_sep|>` — the control/EOT
  family (the massive-activation class the paper calls out).
- l22 c1318 + l23 c1548: `ing / ING / ting / inga / ging` — the -ing family.
- l23 c896: `part / parts / Part` · c518: `like / Like / 像` ·
  c1100: `themselves / itself / himself / herself / yourself` (the reflexive
  family) · c569: `... / ..." / ...”` (the ellipsis family).
- l20 c80: `的 / ه / سя / 의 / 的传播` — the cross-lingual function-morpheme
  family.

These are display evidence only — the T4 GOAT (ordering legs on the census
sweep) is the acceptance gate for the screen as a ROUTING prior; this cell
claims the artifact exists, is deterministic, and reads.

## Gates

- G1 fixture battery 9/9 (`cargo test --features readout_screen --lib
  readout_screen`): running-selection vs brute force, the engineered-unit
  liveness proof, the bit-exact sign trick, first-index ties, pinned band
  formula, artifact byte-stability + header round-trip + truncation/foreign
  refusal, the loud refusal battery, lens-norm f64 reference + positive-scale
  invariance, report determinism. The bin's `--self-test` drives the same
  source (`readout_screen_fixture`, the neuron_census_fixture pattern).
- Clippy `-D warnings` at the feature posture (lib+bins+tests) + the default
  lib posture: clean. (The rulewright's fix lane refused fail-closed —
  `baseline_broken`: the PRE-EXISTING default-posture `--all-targets` red in
  `examples/i701_precomp_fit.rs` (missing `required-features` row, P30 class;
  proven pre-existing at HEAD `85dcb746` in a detached worktree before any of
  this branch's changes). Mechanical findings fixed by hand per the documented
  blockage path.)
- G3: the feature is opt-in and zero-dep; unarmed builds compile the module to
  nothing; serving bins never link it (same offline posture as `weight_audit`).
- **Pre-existing red NOT touched** (not this lane's): the i701 example above;
  `.docs/006_dory_looped_lane.md` referenced by AGENTS.md is absent from the
  tree (highwater = 5).

## Session

Session: zed-m3-i1250, 1795616209. Claim lease in `.issues/1250` (reshape to
`Lease closed` at this landing).
