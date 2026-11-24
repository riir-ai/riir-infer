# Bench 1219 — issue 1189 T4: the full-default closed-form write cell on Cell A (Qwen3.5-0.8B-Base-Q8_0)

**Issue:** `1189` (T4 — the deciding cell)
**Ran:** bmo exclusive, job `01a1db54-eb41…aafe697` (re-queue 2 after the cmake fix), started 2026-11-14T18:37:50Z, ended 2026-11-14T20:02:21Z (85 min, exit 0). Adjudicated 2026-11-15 by zed-a0615aba@amo against the pre-registered T4 gates. **POISONED-WINDOW DISCLOSURE (zero timing claims):** macOS `mediaanalysisd` (pid 1125) started during the exclusive window ~30 min in — the rig flagged the run results-suspect. The adjudication KEEPS the run on the T3-precedent posture (zed-07a26ed5@amo, concurred): this is a correctness/determinism cell — byte-compare + identities + generation recall — contention moves wall time, not arithmetic, and both runs rode the same window so the compare stays apples-to-apples. No perf number is claimed anywhere in this bench; a red gate under poison would have been the re-run trigger (none red).

## Posture

- Model: `Qwen3.5-0.8B-Base-Q8_0.gguf` (the T0-named qwen35-hybrid dense decode lane) at `/Users/katopz/i1208_t5/` on bmo.
- Harness: `examples/closed_form_write_cell.rs` at FULL defaults (4 layers [10,11,12,13], rank 16, 6 refinement rounds, boost calibration), run TWICE (RUN A → `/tmp/i1189_t4_a`, RUN B → `/tmp/i1189_t4_b`) + the per-file byte-compare — the G2 same-inputs evidence.
- Construct wall: 2450 s (bmo, exclusive — clean; the T3 reduced smoke's 612 s was at half posture on a LOADED amo).

## Verdict: G1 FAIL — the write moves the activation, it does not RECALL the fact

| gate | result | evidence |
|---|---|---|
| G1 held-out wording recall | **FAIL 0/1** | `recall-pre[f0]: "Paris" in armed generations: 0/1` in BOTH runs; held-out logprob DID improve (−2.7278 → −2.3827, Δ +0.3452) but the free generation answers "the capital of the country." — activation moved, fact not produced |
| G1b protected look-alikes | **GREEN (city)** | look-alike#0 stays "Rome." in both postures; ⚠ one digit moved inside the look-alike text ("The 2017-20" → "The 2019-20") — a string-level change in a protected row that kept its semantics; recorded, not gated (the gate's subject is the look-alike answer, which held) |
| controls | GREEN | control#0/#1/#2 generations byte-identical baseline → armed |
| G2 determinism (same inputs → same sidecar) | **PASS** | RUN A and RUN B sealed the SAME envelope digest `9b85b57f68f01b003eec990270d2864212a423b90c74f80a1ccc30a8556c195d`; the per-file byte-compare printed `BYTE-IDENTICAL` across the two runs (4 pair files + manifest) |
| G3 empty-overlay demote identity | **PASS** | `D demote identity: byte-identical ✓ (5 probes)` in both runs; `F reload identity: byte-identical ✓` (sidecar reload + remount reproduces the armed pass exactly) |
| G4 refinement monotone-or-decline | **PASS (terminal)** | fit=0.500 at rounds=6, `miss=false`, dropped_energy 1.9e-13 — the refinement converged at its floor without diverging |
| fit quality | — | gain=0.50 (the solve's own fit floor), rank 16, layers [10,11,12,13] |

**T5 falsification arms: VACUOUS — skipped with the 990 law's reason** (they count only when G1 > baseline; G1 = 0/1 here, below the wsp-scaled 2/5 floor the issue pinned).

## Reading

The T3 reduced-smoke pre-read (0/1 at `--layers 11,12 --rank 8 --rounds 2`) is CONFIRMED at full defaults: the boost-calibration + 4-layer + rank-16 posture did not unlock generation-level recall. The write's activation effect is real and deterministic (same digest twice, held-out logprob up), but on this model class the closed-form solve at rank 16 does not reach a generation-decisive direction for the fact. Cell A's honest outcome: **overlay-at-dequant machinery green end-to-end (determinism + demote + reload identities), fact-write efficacy NEGATIVE at this posture**. The in-lattice-vs-overlay pair (T4's second mandate) is MOOT while G1 = 0 — an in-lattice write cannot beat a posture that does not recall.

## Consequences

1. Issue 1189's ladder: **T4 verdict = NO-GO for Cell A at the current recipe** — T5 skipped vacuous, T6 recorded here + in katgpt-rs Research 659 §7. Cell B (qwen3.8-27b-Q4_K_M) is NOT automatically run: the recipe's only Measured outcome is negative on the smaller class; a Cell B spend needs a recipe change first (rank/layer sweep or teacher-forced recall in the calibration set), not a bigger box.
2. The substrate lanes stay: the T1 tap seam, T2 overlay-at-dequant, T3 harness + sidecar (deterministic, sealed, reload-identical) are standing, reusable machinery with green identity gates — the negative is the RECIPE's, not the pipeline's.
3. No serving-posture anything (T6's owner gate untouched).

## Provenance

- Log: bmo `~/.agent-mail/jobs/logs/01a1db54-eb41-70ae-a650-ad9a7aafe697.log` (bmo-local; the notify copy rode `zed-07a26ed5@amo`'s inbox)
- Sidecar digests: `9b85b57f68f01b003eec990270d2864212a423b90c74f80a1ccc30a8556c195d` × 2 runs
- Box state: bmo exclusive, AC, quiet lane (job runner's exclusive gate), load ≤ 2.7/30 throughout (probe cache); POISONED ×1 (`mediaanalysisd`, disclosed above — zero timing claims)
