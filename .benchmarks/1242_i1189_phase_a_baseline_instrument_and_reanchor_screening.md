# Bench 1242 — the plan-635 phase-A baseline instrument + the re-anchor screening (i1189 recipe-2)

**Cell class:** correctness instrument (baseline screening), NOT a perf cell — no
perf claim leaves this record. Deciding cells (T2/T4) carry their own twice-run
G2 law per the plan.

## What ran

1. **The T3 gate amendment LANDED in the harness** (`27211f8`, this repo): the
   plan-635 T3 receipt's amendment (i)-(iv) — baseline-delta recall scoring, the
   authored HELD-OUT paraphrase probe set (`heldpara[f0]#k`, disjointness +
   no-answer-leak test-pinned), the first-k + distinct-token-ratio hit rule
   (`RECALL_FIRST_K=8`, `RECALL_DISTINCT_RATIO=0.5` — the T3 attractor measured
   3/8 = 0.375, below the floor), the repetition gate beside G1b, and
   `--phase-a-only` (the baseline instrument). Tests 4 -> 7, all green; clippy
   `-D warnings` silent; rustfmt clean; cargo refine fix 0 applicable edits
   (28 advisory spans).
2. **Phase-A baseline run #1 — the DEFAULT doc (France -> Paris) at
   `--paraphrases`** (held-out set + gates). exit 0, 271.1 s probe wall, 5.5 s
   load.
3. **Phase-A screening run #2 — five long-tail capital candidates at
   `--facts`** (one run, 5 facts x 5 held-out probes + 3 controls). exit 0,
   1171.4 s probe wall (28 probes: 25 held-out + 3 controls).

## Box state

4090 rig (shikuwa, i7-13700K 16 cores), 2026-11-18T21:1x-21:3xZ, AC power, CPU
load ~1% (the crash-looping i1223 exclusive job cancelled 21:05Z first —
poison x956, zero docs done; rig freed), 15.9/32 GB RAM free, GPU idle 3%.
Concurrent: one sibling agent's fleet-sweep script work (intermittent, scripts
lane). CPU-bound correctness posture; no exclusivity claim made.

## Result 1 — the default fact FAILS the amendment-(ii) gate

Baseline recall on the held-out paraphrase set: **2/5** (probes #3 Elysee-Palace
and #4 name-frame hit at baseline). `baseline-gate: 2/5 > 1/5 -> FAIL — the
fact is too well-known to test recall gains; re-anchor branch (plan-635 T3)`.
In-sample paraphrases 4/5 at baseline (matches the T3 receipt's baseline
reading — cross-check); heldout wording 0/1 (the "France's capital city is"
form teacher-forced lp was already measured in T3); repetition gate GREEN
(0/11 collapsed). **The re-anchor branch of the amendment FIRES.**

## Result 2 — the candidate screening (the anchor selection evidence)

| fact | answer | baseline heldpara | answer-token lp range | profile |
|---|---|---|---|---|
| f0 | Astana (Kazakhstan) | 2/5 | -0.97..-13.1 | partly known, sometimes argmax — too known |
| f1 | Yamoussoukro (Cote d'Ivoire) | 1/5 | -4.9..-11.7 | AT the floor (one Q-form probe generated the answer at baseline) |
| **f2** | **Gitega (Burundi)** | **0/5** | **-3.98..-10.25** | **at floor + IN-MODEL mis-ranked** |
| f3 | Ngerulmud (Palau) | 0/5 | -10.9..-18.1 | at floor but deep-negative lp — likely NOT in-model |
| **f4** | **Dodoma (Tanzania)** | **0/5** | **-5.41..-12.0** | **at floor + in-model mis-ranked (reserve)** |

Repetition gate GREEN on all five (0/5 collapsed each). The lp discriminator:
shallow-negative answer lp (−4..−10) = the answer is a live alternative the
generation did not pick (the stale-capital mis-ranking profile — Burundi's
2019 capital change Gitega-vs-Bujumbura, Tanzania's Dodoma-vs-Dar-es-Salaam);
deep-negative (Palau, −11..−18) = the answer may not be in-model at all, and a
re-ranking write cannot recall what is not there.

## The pre-registered re-anchor (pending the delegated verdict)

- **Anchor: Burundi -> Gitega** (f2) — 0/5 baseline on all five held-out
  surface forms, answer in-model (shallowest clean lp −3.98..−10.25).
- **Reserve: Tanzania -> Dodoma** (f4) — same profile, deeper lp.
- Both carried as authored docs (train asks x3 + contrast family + the
  screened held-out probe set verbatim) — no probe re-tuning after the
  pre-registration (the T5 law's spirit, applied at the anchor level).
- Falsifier unchanged: if the T2 cells on the new anchor show the fit pinned
  at 0.500 AND no amended-metric recall gain, recipe-2 is measured-negative
  and the ladder records STOP (the T4-NO-GO law generalizes) — the anchor
  change cannot rescue a saturated direction family.

Artifacts: `E:/tmp/i635/phase_a_baseline.log`, `E:/tmp/i635/screen_baseline.log`,
`E:/tmp/i635/screen_candidates.json` (the screening doc, reproduced inline in
the plan's anchor row). Harness commit: `27211f8`.

Session: zed-4090-house@4090 (shikuwa), 1795035423

## Amendment (2026-11-18T22:0xZ — the delegated-verdict folds, measured)

Delegated verdict round 1: **AGREE** (reviewer independently checked the plan
row at `aeb7e1f`, the screening table, the per-probe f2 rows in the log, the
anchor doc, and the harness's lp-sum semantics). Four fold-ins, all landed:

1. **Same-subject locality probes** (beside G1b): the anchor doc
   (`examples/facts/i1189_gitega_anchor.json`) carries
   `lookalike_asks: ["The largest city of Burundi is", "Bujumbura is a city in"]`
   — under the write, Bujumbura must stay unchanged. These ride the G1b
   look-alike zero-change class in every T2/T4 cell.
2. **Reserve-use condition pre-registered** (in the plan row): Dodoma is
   invoked ONLY on anchor-as-instrument failure (a Gitega D/F identity fails,
   or the Gitega baseline shifts under an identity re-run) — never on a
   negative recall result, which is the experiment's answer, not the anchor's
   defect.
3. **Baseline generations MEASURED** (the instrument now prints them;
   re-run `screen_baseline_r2.log`): f2#4 emits `**Bujumbura**` verbatim —
   the stale-capital reading is measured, no longer an assumption; f4 says
   `Dar es Salaam` on three probes; f3's neighbors surface (Rarotonga,
   Palikir) while Ngerulmud never does.
4. **Per-token mean-lp measured** (instrument column; the verdict's
   summed-lp token-count confound CONFIRMED real): Gitega f2 mean
   −1.33..−3.42/token (best in-model profile), Dodoma f4 −2.71..−5.98,
   Ngerulmud f3 −2.18..−3.62 — per-token the Palau answer is shallower than
   the summed lp suggested, so the Palau exclusion now rests on the measured
   generations (the answer never surfaces; neighbors do) + the joint floor,
   not on summed lp alone. The pick is unchanged.

**Determinism cross-check:** the re-run (richer read-out, same binary posture,
same probes verbatim) reproduced EVERY hit/ratio/lp value of the committed
run #1 table exactly — diff shows only the appended mean_lp columns. Two
independent executions, identical measurements.

The anchor's five held-out probes and all other doc content are UNCHANGED
from the registered screening — this amendment adds read-out, never retunes.
