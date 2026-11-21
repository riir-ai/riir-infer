# Bench 1251 — Issue 1243(a) T1: the QDA Q4 frozen-read census (pins + sanity + pooled Δ-NLL deviation arm; the si-lane scoring census refuses loud)

**Status:** PARTIAL — every step the existing tooling can run, ran clean on bmo; the census's si-lane scoring arm is LOUD-BLOCKED by the ternary loader's globals contract (verbatim refusal recorded below), so the pooled Δ-NLL rides the prism `llama-perplexity` deviation arm (same lane both files — apples-to-apples). **No adoption verdict is claimed**: pooled Δ sits inside the noise band at this corpus scale, and per-family conditional retention (the decision rule) was never computable cross-file without the ablate machinery. T1 stays OPEN on the scoring instrument; the rest of T1's deliverables (pins, sanity assert, header census) are DONE.

**Date:** 2026-11-21, bmo (the quiet runner), Exclusive rig job `01a1fbe7-9109-7281-b467-774a91dbf921` (attempt 4; attempt 1 died on a missing script — pre-pull staging gap; attempt 2 on the missing `hf` CLI — driver now curls the pinned resolve URL; attempt 3 on a driver header-parser misalignment — fixed + locally smoke-tested before attempt 4).

PROVENANCE: `PROVENANCE: bmo (M5 Ultra, dedicated runner), job exclusive, caffeinate -is per pass, load at start: 1.07 1.08 1.11, AC: AC Power` (driver `scripts/i1243a_census.sh` at `440ed432`; box state + `uptime` + `df` lines in the job log head).

## Pins (the T1 pin action — recorded for every consumer after this)

| piece | sha256 | note |
|---|---|---|
| incumbent `~/uzu27b/models/qwen3.8-27b-Q4_K_M.gguf` (bmo) | `e00082f779fa385cee8c68a3ec8833a75778cc87272240b942f74e0b8243e520` | 16,810,714,336 B; the bench-946 league-serving path; first sha256 record of this file = this T1 pin |
| QDA `QDA-Qwen3.8-27B-Q4.gguf` | `30cc845fe0df2cc4d2533763478ac65e799ce00eb57d8702a3743c7e34bb6693` | 17,556,575,840 B; fetched from `qda-lab/QDA-Qwen3.8-27B-Q4-GGUF` @ rev `2038e5f1502bdd09e15a4015491ff95b8b523e37` (curl, resume-safe); lives on bmo `~/riir-bench/i1243a/qda/` |
| Bonsai control `Ternary-Bonsai-2-27B-PQ2_0.gguf` | `3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1` | the plan-631 pin, verified at job start |
| corpus `i1188_corpus.txt` | `a6998c740ab43d21eaa127e699d1aff042cba41b4df8aea46e7f05b40712fcbc` | the committed recipe (unit `9f95700f…` ×28 = 29,036 B), ONE frozen corpus shared by every arm |

## The issue's arch/size sanity assert (GGUF header read): PASS

`header_census.txt` — both files: arch `qwen35`, block_count 65 (64 main + 1 nextn), 866 tensors, **params 27,320,697,856 — identical** (also matches the HF repo's published `total`). The comparator is the same base; the false-green trap (the 1.14 GB DFlash2 drafter head) is nowhere near this pair.

| file | Q6_K | Q4_K | Q5_K | Q8_0 | type-23 (IQ-class id) | type-11 | F32 |
|---|---|---|---|---|---|---|---|
| incumbent Q4_K_M | 67 | 439 | — | — | — | — | 360 |
| QDA Q4 | 52 | 196 | 140 | 35 | 82 | 1 | 360 |

The QDA recipe's storage mix is materially richer (Q5_K/Q8_0/IQ-class where the incumbent is uniform Q4_K) — same param count, ~745 MB larger file. This is the recipe's per-tensor precision allocation; the header census is the cheap structural form of it (a full per-tensor allocation table rides the follow-up instrument).

## The loud refusal (the census's si-lane scoring arm — T1's named posture)

`si_zero_shot_ppl` (release, features `scale_invariant_attn,bonsai2_hadamard,golden_nll`) on the incumbent, rc=1, verbatim:

```
Error: load /Users/katopz/uzu27b/models/qwen3.8-27b-Q4_K_M.gguf

Caused by:
    tensor 'token_embd.weight' is Q4_K — expected Q2_0 (id 42/142) or PTQ1_0 (id 143) for ternary repack
                 (the globals token_embd/output stay ternary by contract; a q4-class prefill pack quantizes only
                 the per-layer projections — Issue 028 T4 S2)
```

This falsifies T1's "zero new code" premise as filed: the ternary loader's contract keeps the GLOBALS ternary (`gguf_loader.rs` `load_ternary_proj`), and both W4 files carry K-quant globals (incumbent `token_embd` Q4_K, `output` Q6_K). A dense f32 load is not an alternative lane either: 27.3 B params × 4 B ≈ 109 GB — over bmo's 96 GB. The missing instrument is a **dense-lane golden census** (wire `load_qwen_deltanet_weights_gguf` + `forward_qwen_deltanet` into the existing `GoldenNllScratch` oracle, quant-resident or GPU-lane for residency) — that is new code and rides a separate claim, NOT this cell. Consequently T1's **per-family argmax flips and fp32/f64 cliff-band columns are UNDELIVERED** (family attribution cross-file needs the ablate machinery anyway; the filing session's column list was written by analogy to bench 1215, never a runnable two-file procedure).

## Deviation arm (disclosed): prism `llama-perplexity`, same lane both files

`PrismML-Eng/llama.cpp` @ tag `prism-b12128-d7b2896` (the league opponent-arm fork), Metal/CPU build on bmo, `llama-perplexity -m <file> -f i1188_corpus.txt` — the ONLY lane that loads BOTH files today. Same corpus, same chunking (11 chunks, n_ctx 512), same tokenizer family (both GGUFs carry the qwen35 tokenizer).

| file | PPL (11 chunks) |
|---|---|
| incumbent Q4_K_M | **1.0139 ± 0.00275** |
| QDA-Q4 | **1.0134 ± 0.00260** |

Δ-PPL = **−0.0005** (Δ-NLL ≈ −4.9e-4 nats/token, QDA marginally lower). Reading discipline:

1. **The delta is inside the error bars** — no pooled regression VISIBLE, and no pooled gain PROVEN. Aggregate never decides (the lossy-surface law); per-family retention is the adoption rule and it did not run (no instrument).
2. **The corpus is easy** — the i1188 unit ×28 is repetitive by design (sensitivity attribution, not difficulty); both files read near the ppl floor, which compresses delta resolution. A harder held-out corpus (the 1242 T1 anchor traces, when they exist) is the right re-read.
3. Lane caveat: prism-kernel numbers, not house-lane numbers; the pairing (same lane, same corpus, same tokenizer) is what makes the COMPARISON valid, not the kernel's absolute scale.

## Bonsai control (its lane, zero code): ran clean

`si_zero_shot_ppl` base arm on the plan-631 pin: 16 rungs 512→2048, 512 scored/rung, pooled ppl **1.2438**, per-rung curve 2.1464 (L=512) → 1.1848 (L=2048), arm wall 1006.7 s, load 2.9 s. Its role stays what the issue says: cross-architecture pivot visibility, never a like-for-like QDA verdict. The golden-oracle family census on this same file is bench 1215 (unchanged by this record).

## Artifacts

- bmo `~/riir-bench/i1243a/`: `summary.txt`, `header_census.txt`, `incumbent_attempt.log` (the refusal), `bonsai_golden.log`, `pp_incumbent.log`, `pp_qda_q4.log`, `load_trace.log` (2-min cadence), `i1188_corpus.txt`, `qda/QDA-Qwen3.8-27B-Q4.gguf` (the pinned fetch), `llamacpp/` (the prism build).
- Driver: `scripts/i1243a_census.sh` at `440ed432` (curl fetch + header census + loud-attempt capture + prism arm).
- Job log: `agent-mail rig log bmo 01a1fbe7-9109-7281-b467-774a91dbf921`.

## What T1 still owes (the honest remainder)

1. The dense-lane golden census instrument (new, small, named above) — then: per-position Δ-NLL, argmax flips, cliff bands, ONE frozen read, both files + the Q6 row.
2. The Q6 row (`qda-lab/QDA-Qwen3.8-27B-Q6-GGUF` — verified to exist on HF, fetch + pin + same arms).
3. A harder-corpus re-read of the pooled pair once 1242's anchor traces exist.
