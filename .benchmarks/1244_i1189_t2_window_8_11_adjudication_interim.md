# Bench 1244 — plan-635 T2 interim: window {8,9,10,11} on the Gitega anchor (REJECTED)

**Status:** INTERIM — arm 1+2 of the 8-arm sweep. The reference arm
({10,11,12,13} on Gitega) is in flight; final T2 adjudication waits on it.
Cell class: correctness/quality read-out (the amended G1 metric), not perf.

## Posture

4090 box (shikuwa), windows {8,9,10,11} runs a+b sequential, rank 16,
rounds 6, Gitega anchor doc (`examples/facts/i1189_gitega_anchor.json`),
`--out E:/tmp/i635/t2/w8_9_10_11_{a,b}`. Both runs exit 0; wall ~3.5 h/arm
(the 55-min estimate was bmo-derived; this box's construct runs ~4x slower
under its own RAM pressure — the sweep's remaining schedule re-arms from
measured walls).

## Twice-run G2 (the determinism law): PASS

- envelope digest `270c2c21…d20148` BYTE-IDENTICAL across a/b (manifest
  compare);
- probes_baseline + probes_armed tables byte-identical (manifest compare);
- fit 0.000 reproduced exactly.

## The read-out (amended metric): REJECTED on every bar

| bar | result | verdict |
|---|---|---|
| fit-gain >= +0.05 over reference | fit = 0.000 (T1's France reference class: 0.500) | FAIL — negative |
| amended-metric recall gain | 0/5 -> 0/5, delta +0/-0 | FAIL — zero |
| G1b look-alikes zero-change | VIOLATED — " Kigali, which has…" -> " K G Bumbi, located in Burundi, with" | FAIL |
| controls byte-identical | VIOLATED — control#2 generation drifted ("…I'm late" -> "…but I can") | FAIL |
| D/F identities | byte-identical both runs | PASS |
| repetition-gate (amendment iv) | GREEN (0/5 collapsed, both postures) | PASS |
| baseline-gate (amendment ii) | 0/5 <= 1/5 PASS | PASS |

**The mechanism reading (honest, diagnostic):** fit 0.000 with
dropped_energy 1.8e-13 and miss=false — the solve CONVERGED on a
zero-explained-share solution, and the overlay still applied its 0.50
boost along that garbage direction: heldpara answer logprobs collapsed
~-60 nats (-5.69 -> -66.11 class on every probe) and armed generations
degrade into near-gibberish ("Dar esitegaga", "G G Bumbi"). The write is
not merely failing to recall — it is ACTIVELY DESTRUCTIVE at this window
on this anchor: it damages neighboring-geography tokens (the G1b class)
and reaches even a control. Window {8,9,10,11} on the Gitega anchor is
REJECTED; the T2 record keeps the arm (a pre-registered cell, both runs,
recorded not hidden).

## Schedule note (compute law)

Arms 3-8 were sized 55 min each from T1's bmo wall; measured wall is
~205 min/arm. The reference arm (10-13 a+b) is the next adjudication
point (~16:00 local); arms 5-8 ({12,13,14,15} + {14,15,16,17}) re-decide
THEN — if the reference shows the France-class fit (~0.5) with no recall
gain, coverage is REFUTED per the pre-registered bar and the remaining
arms cancel as dead compute; if the reference is flat at 0.000-class,
no window can clear +0.05 and the T4-NO-GO law generalizes (STOP).

Artifacts: `E:/tmp/i635/t2/` (logs + manifests a/b). Determinism compare
ran via manifest json equality (digest + both probe tables).

Session: zed-d102f326@4090 (shikuwa), 1795053600
