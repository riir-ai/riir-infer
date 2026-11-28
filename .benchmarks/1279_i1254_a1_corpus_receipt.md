# Bench 1279 — i1254 A1 fact-corpus receipt (the data plane of record)

**Unit:** riir-infer Issue 1254 (Plato's Key cross-model key transfer) — the A1
fact corpus landed before any forward runs. Landing hash: `895bd60` (branch
`i1254-p0a-p0b`) carries `src/i1254_corpus.rs` + the tracked
`src/i1254_a1_corpus.jsonl` + the `i1254_corpus_audit` bin; disclosure: the
parallel sweep lane committed this session's dirty worktree state mid-turn
(same lease session `zed-2611a378@4090`, same content — the record commit below
adds ONLY this receipt, the highwater, and the issue-status line).

**Provenance:** amo/4090 box, CPU-only authoring + audit run (GPU busy on the
train census job `tr714-census` — the donor-forward cells stay GPU-queue-gated
per the issue's rig discipline; nothing below is a timing claim).

## Corpus

- File: `src/i1254_a1_corpus.jsonl` (tracked, include_str!-loaded), 1387 lines.
- **Corpus digest (blake3):** `34b18a6a1c61e42fe25ad2f7eefcd14e90ac9bc85456b07f7038c7b44c297f13`
- 16 relations (the issue's pre-registered set), per-relation counts:

| rel | name | n | rel | name | n |
|---|---|---|---|---|---|
| c | capital_country | 188 | an | author_novel | 93 |
| sc | us_state_capital | 50 | pp | painter_painting | 61 |
| sa | us_state_abbrev | 50 | cw | composer_work | 83 |
| cu | currency_country | 127 | fc | founder_company | 83 |
| lg | language_country | 97 | iv | inventor_invention | 39 |
| ct | city_country | 153 | cs | chemical_symbol | 118 |
| co | country_continent | 87 | lc | landmark_city | 87 |
| cc | character_creator | 50 | pm | planet_moon | 21 |

- Prompt = `capitalize_first(before + subj + after)` per the relation registry
  (`src/i1254_corpus.rs::RELATIONS`); the subject span ends the prefix, so the
  subject's last token is computable exactly for both the donor capture and the
  recipient REPLACE hook. Code-symbol relations (sa/cs) match CASE-SENSITIVELY
  ("IN" never matches "in"; "Au" never matches prose); everything else matches
  accent-folded word-boundary substring over the first 16 generated tokens.
- Any-of object arrays carry the honest alternative surfaces (co-officials,
  transliterations, renamed capitals: Astana/Nur-Sultan, Kyiv/Kiev,
  Dodoma/Dar es Salaam, …). Known-limitation disclosures:
  the pool is ~1.4k raw facts (below the 2k-pair TARGET, which is a target, not
  a gate — the paired-recall filter will cut further; the fit runs on whatever
  the paired pool is and reports its size); facts were authored from model
  knowledge, not scraped — spot-checked, not externally verified (the paired
  filter is itself the defense: a wrong fact fails recall for both models and
  drops out).

## Split (subject-wise, deterministic)

- Key: blake3(subject bytes)[0..8] LE → SplitMix64 finalizer → `%100`:
  `<70` Train, `<80` Val, else Test. Same subject ⇒ same bucket forever.
- **Split-assignment digest (blake3):** `19573a87ce2100078dc346dd16b6c6ea4ffa948e2eae6e239a7c1dc5b8f44fdc`
- Counts: **train=991 val=107 test=289** of 1387 (71.4 / 7.7 / 20.8).
- Pinned buckets (guard against hash/finalizer drift): France→Test, Io→Train,
  Sherlock Holmes→Train (exact pins in the module tests; debug and release
  agree).

## Gates (this landing)

- `cargo test --lib i1254_corpus`: **7/7 PASS** (load + well-formed, prompt
  render + subject-span split, split determinism + digest stability, split
  distribution bounds, matching word-boundary/accent, case-sensitive code
  matching, pinned buckets).
- `cargo clippy --lib --bins`: clean (default build).
- `i1254_corpus_audit` bin: prints the receipt above; refusal battery GREEN
  (unknown relation code refuses loud).
- rustfmt clean on touched .rs (bare rustfmt per file).

## Manifest repair riding this landing

The P0a commit (`ea8c88b`) landed `Cargo.toml` with UNRESOLVED conflict markers
(`<<<<<<< HEAD … >>>>>>> e0715b1`) in the `[dependencies]` table + absolute
`E:/git/katgpt-rs/...` dep paths — a state that fails TOML parse for anyone
checking the branch out fresh (verified: `tomllib` refuses at line 49). This
landing's tree carries the repaired manifest: relative sibling paths, one row
per dep, audit-bin `[[bin]]` row added, and it parses clean. Noted here because
a manifest fix with no reader-visible record is how the next session re-finds
the hole.

## Next units (unchanged routing)

Donor band capture (4090 census-window-gated) → per-side PCA r=256 (train-split
only — the leak law) → per-(ℓA,ℓB) Procrustes in PC space → VAL layer selection
→ REPLACE-hook test arms. The recipient-side (llama-3.2-3B) recall sweep is the
first cell once a compute window opens; the donor 27B forwards stay
`--est --exclusive` rig-queued behind the census job.
