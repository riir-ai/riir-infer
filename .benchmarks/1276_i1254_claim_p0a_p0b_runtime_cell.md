# Bench 1276 — i1254 claim + P0a/P0b runtime cell (rope_freqs.weight loader arm)

**Verdict: P0a/P0b GREEN** — the recipient-side runtime verification over the REAL
checkpoints passes end-to-end; the loader-arm's missing tensor read landed; the
scaling-factor shape disposition is measured, not assumed. A1/A2 forward cells
are the next units (see the issue's Claim line).

**Status:** LANDED 2026-11-28, branch `i1254-p0a-p0b` (riir-infer `ea8c88b` + `19e1391`),
claim lease `zed-2611a378@4090` exp 2026-11-29T06:00Z (issue 1254).

## What ran

| Read | Result |
|---|---|
| Recipient GGUF probe (`plan402_gguf_probe.py`) | `llama` arch, 28 blocks, d=3072, GQA 24/8, head_dim 128, ctx 131072 (clamped at run), rope freq_base 5e5, `llama.rope.dimension_count` 128 |
| Donor GGUF probe | `qwen35` arch, 64 blocks, d=5120, PQ2_0 ternary (Bonsai-2.5-27B) |
| `rope_freqs.weight` tensor-info walk (python GGUF parse, `E:/tmp/i1254_tensor_flex.py`) | **dims=[64], F32** — the PAIRED spelling (one factor per rotary pair), NOT head_dim/2=128 |
| KV metadata walk | NO `rope_scaling`/`rope_freq_base_scaled` keys — the scaling rides ONLY the tensor (llama3-type, factor 32 confirmed by the measured max freq ratio) |
| P0 runtime probe (release, `--features deltanet_ternary_inference,bonsai2_hadamard`) | P0b PRESENT len=64 → scaled table built, `identical-to-plain=false`, **max(freq_ratio)=32.0000** (matches llama3 rope_scaling factor 32) · P0e recipient load OK (d=3072/28 blocks/8 kv) · P0e donor ternary load OK · P0d donor tap = the ternary forward's `layer_capture` argument (compile-proven API surface) |

## Findings

1. **The paired spelling** (found by the probe's refusal, confirmed by the
   tensor-info walk): `rope_freqs.weight` carries `head_dim/4` factors
   (64 for head_dim 128), one per rotary pair, applied to BOTH dims —
   llama.cpp semantics. `new_weighted` now accepts both spellings
   (per-dim `half` or paired `half/2`) and still refuses everything else loud.
2. The upstream comment's "expected head_dim/2 = 128 entries" was wrong for
   the REAL checkpoint — the loader would have refused it loud at load.
   Caught before any silent wrong-frequency run: the probe is the cell's own
   baseline-recall-gate tripwire working as designed (construction time, not serve time).
3. All-ones factors build the plain table bit-exactly (both spellings) —
   g3-test-pinned (`g3_rope_freq_weighted_table_laws` paired arms added).

## Gates

- lib 310/0 default (dev AND release) · clippy -D warnings default + ternary
  posture · rustfmt clean on touched files · the probe bin ungated
  (donor leg SKIP-LOUD without the feature — never a silent half-run).

## Artifacts on disk

- `riir-train/data/Llama-3.2-3B-Base.Q8_0.gguf` — 3,421,895,136 B (size-exact vs
  the HF HEAD), sha256 `3a9d5b61ee410ad957b341d3cacf05c40d3e7f3b127fdbc49ec3437270af2c6d`,
  source `RichardErkhov/1024m_-_Llama-3.2-3B-Base-gguf` @ main
  (`Llama-3.2-3B-Base.Q8_0.gguf`), fetched 2026-11-28T04:0xZ. License: Llama 3.2
  Community License — LOCAL USE ONLY, never a shipped artifact.
- `riir-train/data/MiniCPM5-1B-F16.gguf` — verified size-exact
  (2,166,551,936 B) vs `BACKUP_MANIFEST.tsv` row 610 (the AGENTS LOCAL-KEEP row
  is NOT stale on this box — the issue's "not on disk" observation was
  box-relative). Probe: `llama` arch, 24 blocks, d=1536 (A3 smoke pair usable).
- `riir-train/.env` — created on THIS box (the HF read token carried from amo
  via the Tailscale channel; token VALID vs whoami-v2, user katopz).
- `E:/tmp/wt-i1254` — the detached worktree (branch `i1254-p0a-p0b`).
- Probe scripts: `E:/tmp/i1254_tensor_flex.py` + `i1254_keys_v3final2.py`
  (the GGUF kv/tensor walk; the v3 u64-string-length layout caught two
  misparse classes en route).

## Next units (the cell's remaining arms)

- A1 paired-fact corpus + donor capture (`layer_capture` at the 22–38% band) +
  the r=256 PC-space Procrustes fit + the recipient REPLACE hook cells.
- The 27B donor forwards route to the rig queue (`--exclusive --est`) per the
  issue's scale discipline; m3 authors only.

Session: zed-2611a378@4090, epoch 1795902000
