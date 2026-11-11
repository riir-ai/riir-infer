# Plan 631 — i1190 T3 ranked-sensitivity rig job: claim, corpus regen path, queue

**Status:** IN FLIGHT — claim on `.issues/1190` pushed; corpus recipe reconstructed + verified (sha256 byte-match); the rig job parks on bmo behind i1117. This plan carries the job's exact inputs; execution/verdict live in the issue + the bench record when the run lands.

Session: zed-c21902cf@amo, epoch 1765484700 (2026-11-12). Supersedes nothing; the protocol is `.issues/1190`'s pre-registration, unchanged.

## Why a plan

The rig job's corpus recipe (`/tmp/i1188_corpus.txt`, "the market prose paragraph ×28") existed only as a /tmp file on amo — uncommitted, tmp-wiped on reboot. The issue says "regenerate verbatim from the committed recipe if missing"; no committed recipe existed. This plan commits it (the recipe AND the unit paragraph), so the ONE-frozen-corpus law holds across reruns on any box.

## Corpus recipe (reconstructed 2026-11-12, byte-exact)

- `/tmp/i1188_corpus.txt` = **unit × 28**, where unit = the 1 037-byte market-prose paragraph
  ("The morning market opened under a wide awning, and the first customers arrived…"),
  NO trailing newline beyond the unit's own, concatenated 28× = 29 036 bytes.
- **unit sha256** `9f95700f8d447da6c9390df6fe5c6d5bd2c17e6274b71db9d4f83a44d4ae5934`
  → **×28 sha256** `a6998c740ab43d21eaa127e699d1aff042cba41b4df8aea46e7f05b40712fcbc`
  (= the issue's pinned digest, byte-verified on amo 2026-11-12).
- Committed bytes: `scripts/i1188_corpus_unit.txt` (the unit only; the job script cats it 28×).
- Regenerate anywhere: `for i in $(seq 28); do cat scripts/i1188_corpus_unit.txt; done > /tmp/i1188_corpus.txt`
  then verify the ×28 sha256 BEFORE the run (refuse on mismatch).

## Job shape (bmo, `--exclusive`, parked behind i1117)

- Model: `~/git/riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf` (bmo symlink → `~/models/`,
  file present 7 206 168 928 B). sha256 **`3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1`**
  (read in full over the bridge 2026-11-12) — the job verifies this before any pass; mismatch = refuse.
- Features: `scale_invariant_attn,bonsai2_hadamard,golden_nll`; `--bin si_zero_shot_ppl`.
- 16 family passes + the ffn_down determinism re-run, exactly per the issue's pre-registered
  repro (flags `--max-length 2048 --min-length 128 --score-window 512`).
- Out: `~/git/riir-infer/.i1190_out/` (logs + the ranked table rows); results mail lands on
  amo (`zed-c21902cf@amo`) at end; T2's ranked table + the keep/demote lines are written from
  those logs — the posture line stays owner-gated (the issue's own law).
- Est. 1.5–2.5 h on bmo-class silicon at prefix 2048; `caffeinate -is` per pass; PROVENANCE
  line beside every row (the gates section of the issue).

- [x] Claim line on 1190 pushed before any authoring.
- [x] Corpus recipe reconstructed + unit committed; ×28 digest byte-matches the pin.
- [x] bmo model presence + full sha256 verified over the bridge.
- [x] Plan committed (this file) with the job body; highwater 630→631.
- [x] Rig job queued (`agent-mail rig run bmo --exclusive`, notify amo) — same commit as the queueing act.
- [ ] Run lands: 16/16 engagement + determinism + sanity gates; T2 ranked table written; bench record; issue rows closed.
