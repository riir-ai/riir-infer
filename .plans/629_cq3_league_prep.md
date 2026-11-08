# Plan 629 — CQ3 league prep cycle (issue 1186 T2b: matched-calibration harness + opponent arm)

**Status:** IN PROGRESS — T2b opponent half queued; ours half (capture seam + quantize driver) next cycle.

Executes the league-prep window of `.issues/1186_cq3_qwen38_dense_league_cell.md` (T2b). The
pre-registered gate there is law: **the matched-calibration arm is the validity condition for
quoting any CQ3 margin** — T2a measured the GPTQ-class residual's gain as a
calibration-objective gain (+23% held-out rel-RMS at n_cal/d_in = 2:1), so every margin this
cell produces comes from matched budgets or is marked record-only.

## Pre-registered protocol (frozen before any artifact lands)

### Sources + boxes

| field | pin |
|---|---|
| model source | `unsloth/Qwen3.8-27B-GGUF` `BF16/` split (2 shards), LFS sha256 `b9966e82b7a4d87028b5eae061d578ee826305ebf8baea5bfc6e09bad0ba191f` + `92e3943c4f9bd6292a7bef82369f65fed9bfed088b9df0fb2fa2ce17c9edfa02`; upstream weights = `Qwen/Qwen3.8-27B` |
| opponent build | `PrismML-Eng/llama.cpp` @ tag `prism-b12128-d7b2896` (the Bench 1155 scored tag) |
| prep box | **amo** (m5 ultra, 96 GB unified — the 54.66 GB BF16 fits; the 4090's 32 GB RAM cannot hold it without disk thrash, measured rig: i7-13700K / 32 GB / 24 GB VRAM) |
| perf cell box (T3, later) | 4090, run ALONE, Bout law, `PROVENANCE:` line |
| box-crossing disclosure | prep artifacts are BLAKE3-shipped, never measured on the prep box; the FairnessManifest pins artifact hashes, not prep-box clocks |

### Calibration corpora (text lane; equal modality mix = both text by construction)

| corpus | bytes | sha256 | budget |
|---|---|---|---|
| v1 | 262,144 | `a6fb9133e3a675b42cca96f726fa4b386c07ca93eca80d18689504b715c3d316` | the league's pinned `bpb_corpus_v1.txt` (Moby-Dick prefix) — ≈65K tokens ≈128×512-token windows ≈ the vendor's own 128-sample posture |
| v2 | 1,256,465 | `a3df73ada8719d77323a2f2e81d5592af407941194288c86845e1514d0ef51c7` | v1's exact bytes + a continuation from the 2026-11-09 Gutenberg `pg2701.txt` fetch, aligned at v1's final sentence (the 09-15 fetch this box no longer has is unreproducible byte-exactly — Gutenberg re-typesets; divergence measured at byte 23,724). Disclosed derivation; v1 stays byte-exact. |

Chunking: `llama-imatrix -c 512` windows; ours-side capture consumes the same 512-token
chunks in the same order (tokenizers differ slightly across engines — recorded in the row,
not hidden). Retention/perplexity eval NEVER reads the calibration corpora — a held-out
corpus gets pinned when the T3 cell lands.

### Arms

| arm | ours budget | opp budget | status |
|---|---|---|---|
| **b1** (matched, min) | v1 | v1 | **the transferable row** |
| **b2** (matched, ~5×) | v2 | v2 | transferable; directly tests T2a's overfit finding at higher n_cal |
| a (asymmetric, record-only) | v2 | v1 | the vendor's flattering shape — never quoted as a margin |
| a′ (reverse, record-only) | v1 | v2 | same two artifacts, zero extra runs |
| rtn (record-only) | v1 | none (plain quantize) | the uncalibrated opponent floor |

### bpw match

Ours: g=32, b=3 → `b_eff = b + 16/g = 3.5` vs opponent `Q3_K` (uniform class — ours is
uniform too; no Q3_K_M mixed tensors). Q3_K ≈ 3.44 bpw → ours +1.7% HEAVIER: conservative
direction, disclosed in every row. A win at equal-or-hearser bpw is a real win.

### Ledger

Every row carries: corpus file + sha256, window count (from the imatrix log, not estimated),
model source shards + sha256, opponent build tag + commit, quantize log bpw, artifact
BLAKE3, box name. Lands as `.benchmarks/<N>_issue1186_t2b_opp_arm.md` when the job returns.

## Tasks

- [x] Claim re-pinned in the parseable `**Claimed:**` shape (the prose-only claim was unregistered) — `412ef2a7`
- [x] Rig probe: 4090 = i7-13700K/32 GB/24 GB VRAM, fork at scored tag, imatrix/quantize/bench built; BF16 absent-to-fit → prep routes amo
- [x] Corpus v2 derived (v1-prefix property held) + pinned; HF source identified + shard shas pinned
- [x] Plan pre-registered (this file, before any artifact exists)
- [ ] amo prep job (queued via `agent-mail rig run`): clone fork@tag → build Metal Release → download BF16 (sha256-verify both shards) → `llama-imatrix` @v1 + @v2 → `llama-quantize` Q3_K(v1), Q3_K(v2), Q3_K(rtn) → BLAKE3 all artifacts + ledger JSON — script `scripts/cq3_league/prep_opponent_arm.sh`
- [ ] Ledger bench record landed from the job output (window counts, bpw, hashes)
- [ ] Ours-side capture seam (per-tensor Gram accumulation, sequential layer-local) + `cq3_quantize` GGUF→CQ3 driver — next cycle
- [ ] Ours artifacts at b1 + b2 budgets (same corpora, same chunk order)
- [ ] T3 cells: perf (pp2048/pp4096 + decode vs Q3_K(b1), 4090 ALONE) + per-family retention (held-out corpus, pinned then) — gated on the capture seam + a CUDA CQ3 GEMV arm
- [ ] T4 verdict per the league law (GOAT gate; demote-only env on win; loss → recorded verdict + corpus note)

## Constraints carried from the issue

- Opt-in until T3 passes (never in the release set before a GOAT); linear mixes only, no softmax.
- League model naming law: ours keep full lineage names, never a bare colliding stem.
- T2a's finding stands: no margin is quoted from arm (a)/(a′)/rtn alone — b1/b2 only.
