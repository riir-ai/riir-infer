# Bench 1228 — Issue 1208 T7's first real-model mint cells (the four solver arms, the 0.8B hybrid, amo CPU)

Date: 2026-11-16, window ~21:05–21:20Z (load1 4–7 co-tenant-normal, 30 cores, AC, post-walk window — the F1 walk was killed 20:59Z per `.issues/1226` HEAD; these cells are NOT load-sensitive evidence, the sweep is a 26–50 s CPU forward pass).

Fixture: `riir-train/data/Qwen3.5-0.8B-Base-Q8_0.gguf` (qwen35 hybrid dense, 24 layers, gate 3584×1024 — 88,539,648 gate weights = the 88,080,384-bit mask, the bench-1208 census exactly). Corpus: `riir-train/data/chat_probe` (the T5/T6 posture), 512 tokens cap, corpus digest `72ecce464f551867ee6e9a545c4afb80fb49f601aa45c2116cc580335fffbc8a`. Build: release, `deletion_mask,deltanet_inference`, the increment-2 bin (`d4948cb3`).

## The cells

| solver | rate | artifact bytes | rimsk blake3 |
|---|---|---|---|
| magnitude | 0.001 | 137,060 | `22b30aa41e0f9030af81d22795808861e3ec25412124908fa9031ca1247adb3a` |
| wanda | 0.001 | **1,066** | `25cb0309c76f5d5cae2bbda127003e59cdb679445115b786adb0bcddb602d40b` |
| sparsegpt | 0.001 | 137,060 | `22b30aa4…adb3a` — **IDENTICAL to magnitude (see finding 1)** |
| magnitude | 0.01 | 1,239,487 | `5f5c546179059f59b1d1524995add822bd1b2dcccdcd12b32cfc62b8f62356c1` |
| sparsegpt | 0.01 | 1,244,834 | `e58e6fdf931812ba45de5a6bf253ea4d54f4e0ba1952f303cd96c2019f72c6b9` |

Sweep wall: 26.0 s (magnitude/wanda posture), 50.3 s (sparsegpt posture — the d²-Gram rank-1 adds). Sparsegpt sweep digest `a85a8bc064744062d9e231a7b28d33eba17a5268838d4ce0145b720d7d7d5435` (24 layers, 512 tokens). Compensated sidecar (ii-b, rate 0.001): 24 tensors, blake3 `9ae9d9a548a0fc5b6b0a72ca75d7538b0a095987522fc19004ae485b0df3ca83`.

## Findings

1. **At rate 0.001 the sparsegpt SELECTION is bit-identical to magnitude's mask** (same artifact digest across all 24 layers). Verified NOT a bug: at rate 0.01 the solvers DIVERGE (different masks, different sizes) — the OBS metric axis is real. The identity at 0.1% is a whitening artifact: the gate input rides post-RMSNorm activations, whose Gram over 512 chat_probe tokens is near-isotropic → near-uniform Hinv diag → `w²/d²_j` ranking ≈ `w²` ranking = magnitude, with no re-ordering anywhere in the bottom 0.1%. **Consequence for T7's pre-registration: at paper-class rates (≤0.1%) on RMSNorm-fed gates, the sparsegpt SELECTION arm carries no metric edge — its value proposition at this class is the COMPENSATION (the ii-b sidecar), exactly what the FKL audit must adjudicate against (ii-a).** The floor arms stay mandatory.
2. **Artifact-size spread validates the codec design**: wanda's 0.1% mask compresses to 1,066 bytes (99.9% ones + RLE + zstd — the T1 premise verbatim), magnitude/sparsegpt-at-0.001 to 137 KB, the 1% masks to ~1.2 MB. The wanda selection's deletions cluster (a few columns) → near-perfect RLE; magnitude's scatter → 137× worse.
3. The per-layer loss print at 0.1% shows 0.000000 — print precision (3670 deleted weights of the bottom-0.1% tiny-|w| tail); at 1% it reads 1.9e-5 (blk.23). Not a zero-defect: the loss path is exercised and nonzero in the fixtures.

## The FKL/geometry audit rows

NOT in this bench — the produced masks' FKL (the T6 `--fk-lab` arm) + geometry rows are the NEXT cell (the audit consumes these artifacts on disk); the T7 pre-registration's floor comparison (ours vs magnitude at matched rate, one frozen read) rides it. This bench records: the mint chain works end-to-end on the real fixture, the artifacts exist, the digests are the pins.

## Reproduce

```sh
cargo run --release --features deletion_mask,deltanet_inference --bin i1208_mask_mint -- \
  riir-train/data/Qwen3.5-0.8B-Base-Q8_0.gguf riir-train/data/chat_probe /tmp/out.rimsk \
  --solver <wanda|sparsegpt|magnitude|random> --rate 0.001 --max-tokens 512
```
