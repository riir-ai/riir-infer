# Bench 955 — Issue 1178 T2b price cell: the lane-sliced Q4 decode GEMV on bmo (delegated run)

**Verdict: KEEP OPT-IN — the lanesliced twin loses the e2e decode bar (11.66 vs 11.77 packedOFF,
−0.9%) while adding a 4.5× prefill cliff; G1 token identity PASSES bit-identical; the Bench-951
validity question is RESOLVED (its verdict was NOT an unreachable-arm artifact — the packed route
runs 8.8 today, it stood then).** The lanesliced law does not transfer to Metal at the league
posture. The f16/packed weight-byte-class decomposition (issue 1184) is CONFIRMED — decode
e2e ≈ equal across byte classes (T2's 12.0→11.8 plateau), so ALL remaining decode headroom lives
in the packed GEMV kernel itself, exactly the lanesliced law's target — and the lanesliced form
did not deliver it on this stack.

- **Run:** bmo rig job `01a1bbbe-…` (Exclusive, metal), 2026-11-08 14:52→15:57 UTC (22:52→22:57 +07),
  worktree `~/git/riir-infer-t2bwt` @ `8b94ee4` (impl `5674cef4` + instrument `fd8ecac7` in-tree),
  model `/Users/katopz/uzu27b/models/qwen3.8-27b-Q4_K_M.gguf` (16.81 GB, the Bench-1167 pin).
- **Box:** load 0.51/0.81/1.07, AC power, up 6 days; rig QUIET; no sibling engine, mds 0% (guards
  clean); load avg peaked 5.69 during run 1's teardown, 0.5-1.1 across all measured legs.
- **PROVENANCE:** bmo = bmo (quiet runner, M-series Metal); no GPU exclusivity claim is made or
  needed — decode legs are 43-58 s wall and the two drift legs of every arm agree to ≤0.3%.

## Run 1 (`01a1bb89`) — VOID, disclosed

All three arms measured the league f16 posture: `packedON engagement: demoted=0 armed=1
ls_twin=0`, decode 12.02/12.03/12.04, load ~21 s everywhere. Root cause: the issue's script was
`#!/bin/zsh` but delivered the demote env via `env $ENVSET` — **zsh does not word-split unquoted
variables**, so `env` received one `NAME=VALUE` pair with `RIIR_QWEN38_DENSE_MM2D` =
`"0 RIIR_QWEN38_ATTN_TENSOR=0 …"` ≠ the exact demote literal `"0"` (and the other two envs were
never set). The preregistered engagement gate voids the run; the fix is env plumbing only
(`.issues/t2b_job_v2.sh`, export/unset), arms + gates byte-identical. The league-f16 half is
incidentally valid and kept as a same-window cross-check: decode 12.02/12.03, pp median
905/909 — bracketing the G3 band below.

## Arms (run 2, all gates enforced)

| arm | engagement | load | pp1536 median | decode512 | ctx1344 | drift (leg2) |
|---|---|---|---|---|---|---|
| packedOFF (packed route, twin OFF) | `DEMOTED by RIIR_QWEN38_DENSE_MM2D=0` ×1, ls_twin=0 | **9.4 s** | **10 tok/s** (158.3 s) | **8.77** | **7.70** | 8.81 |
| packedON (+`q4k_lanesliced_decode_gemv`) | DEMOTED ×1, **ls_twin=432** (8 shapes: 128×[17408,5120], 96×[48,5120], 64×[5120,6144], 48×[6144,5120], 32×[5120,17408], 24×[10240,5120], 24×[1024,5120], 16×[12288,5120] — ~6.65/tensor avg) | 15.6 s | 10 tok/s (158.3 s) | **11.66** | 10.59 | 11.68 |
| leagueOFF (f16 league posture) | `[Issue 1150] … arm armed` ×1 | 20.0 s | **918 tok/s** | **12.03** | 10.86 | 12.03 |

### G1 — token identity: PASS, bit-identical

`[walk] ids` md5 `94e5afc27140abc5abccac5dc0848c9c` for BOTH packedOFF and packedON (64 tokens,
default prompt). The lanesliced re-association (`a·nib·x − b·x` fused vs canonical
`(A·nib − B)·x`) did not flip a single token at the league fixture. No 940 disclosure owed.

### The Bench-951 discriminator — RESOLVED: 951 stands

packedOFF decode = **8.77** (≈8.6-9 band). **Bench 951 measured the real packed route** — the
packed path is FASTER today (post-Oct-29 decode work) than its pre-1176 control yet still ~27%
under league f16; nothing about 951's arms was unreachable. **No correction is filed to 951**;
the T2a routed-factored price cell remains honestly measured at 8.6 (its own record stands, and
this cell re-derives its control at 8.77 — same class).

### T2b price — the lanesliced law does NOT transfer (G2 FAIL, G3 ~tied, G4 met)

- **G2 decode e2e:** packedON 11.66/10.59 vs packedOFF 8.77/7.70 → **+33% decode** (the
  lanesliced GEMV IS ~4.3× faster than the plain packed GEMV — the kernel-level law holds) — but
  against the LEAGUE f16 12.03 it is **−3.1% short-context** (11.66 < 12.03×0.985 = 11.85 bar)
  and **−2.5% ctx1344** (10.59 vs 10.86×0.985 = 10.70 bar, also short). T3's ≥1.5% bar fails on
  both cells. **No posture flip is proposed**; the production question is answered NO.
- **G3:** leagueOFF reproduces 12.03/12.03 (league record 12.05, drift band ✓) — the same-window
  f16 datum the record needs. Run 1's 12.02-12.04 agrees.
- **G4 load:** packedOFF 9.4 s (the packed upload is the CHEAPEST load on record — no f16
  conversion, no repack), packedON 15.6 s (+6.2 s = the single-thread repack over ~10 GB of Q4_K
  at ~1.6 s/GB — cheaper than the 0.3-0.5 s/100 MB worst-case line), leagueOFF 20.0 s (f16
  convert). Memory +~4.3 GB for the twins (the ~1.44× quant-weight line: 10.0 GB quant → +14.4
  GB sliced+AB words) — viable on 96 GB bmo, a real cost on 24 GB-class machines.
- **The prefill cliff (the decisive fact):** demoted-dense prefill = **10 tok/s** in BOTH packed
  arms vs league 918 — **~90×**. The issue's arithmetic ("Q4 GEMM should land ~9-15× under f16
  at m=5120·n=17408") is REFUTED — the plain packed Q4 GEMM is ~90× under, not ~9-15×. A
  league-posture flip to packed+lanesliced would need a packed prefill GEMM ~2 orders of
  magnitude better than what exists; T2a's routed-factored GEMM (951, 8.6 decode class) is not
  that. The decode route is the ONLY lane this price cell opens, and its decode number loses.

### The roofline decomposition — CONFIRMED, with the plateau sharpened

The issue's central claim holds in the e2e column: decode tok/s is ~FLAT across weight-byte
classes (packed 8.8 → lanesliced 11.7 → f16 12.0) while bytes/step differ 3.6×. Effective BW:
packedOFF ~0.19 TB/s (9.6 GB / 51.7 ms FFN wall), packedON ~0.30 TB/s (14.0 GB / 46.5 ms),
league f16 ~0.80 TB/s (34.2 GB / 42.6 ms). The f16 kernel sits at ~97% of its own achievable
bandwidth, and the lanesliced packed kernel reached only ~0.30 TB/s — the ALU bound was NOT
lifted to the bandwidth bound. The two remaining decode levers are unchanged from 1178 T2's
record (both engine-side): (a) the GEMM-first decode reshape (decode GEMVs as 1-row GEMMs
through the f16 tensor path — dequant in-kernel, ~1-2% expected), (b) the GDN `ffn_gate_up`
decode fold (28.1% of decode time, riir-infer 1139→1145 named lever). The lanesliced corpus
rule stands as fixture-tested decision content (its Metal transfer failed; the CUDA twin is the
named re-gate).

## Discipline

- Lease: `.issues/1184` claimed `zed-amo-1184@amo` epoch 1794147545 — folded in the same batch.
- Deposit: this record numbers 955 by allocation-time `.highwater` read (954) + `dual_allocation_gate`
  clean; issue 1184 + plan 1178 edits reference it.
