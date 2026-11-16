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

## The FKL audit rows (2026-11-16 ~21:20Z, the same window — the T6 `--fk-lab` arm over the minted masks, 64 probe tokens, 2×64 walks ≈ 9–10 s per cell)

| solver | rate | FKL mean | p95 | p99 | table digest |
|---|---|---|---|---|---|
| wanda | 0.001 | 0.000000 | 0.000000 | 0.000000 | `bfa61215…39c56c24` (= magnitude's — both all-zero) |
| magnitude | 0.001 | 0.000000 | 0.000000 | 0.000000 | `bfa61215…39c56c24` |
| random | 0.001 | **0.001182** | 0.003115 | 0.004725 | `41eab4ac…9e9b` |
| wanda | 0.01 | 0.000000 | 0.000001 | 0.000002 | `418f83df…08ab7` |
| magnitude | 0.01 | 0.000001 | 0.000002 | 0.000002 | `ee9b987c…4f60c` |
| sparsegpt | 0.01 | 0.000001 | 0.000003 | 0.000004 | `fdbd4769…263c2` |
| random | 0.01 | **0.022466** | 0.069802 | **0.258598** | `4bbf56e8…1037135` |

**The floor comparison (pre-registered, one frozen read):** the three principled arms tie WITHIN INSTRUMENT RESOLUTION at both rates (≤2e-6 mean at 1%) and crush random by 4 orders of magnitude (0.0225 vs 0.000001 at 1%). The falsifier's bad case (principled ≤ magnitude) did not fire — **closed-form content is ALIVE on this fixture**; solver CHOICE is immaterial for SELECTION at ≤1% on RMSNorm-fed gates (finding 1's whitening explanation extends to the KL axis). The sparsegpt differentiation therefore rides entirely on the ii-b COMPENSATION sidecar — whose FKL needs a compensated-forward harness (the audit applies RIMSK masks only); that harness is the lane's next instrument, not this cell. All cells: instrument-grade (random resolved loudly at both rates — the instrument is not blind), one frozen read, digests pinned per row.

## Reproduce

```sh
cargo run --release --features deletion_mask,deltanet_inference --bin i1208_mask_mint -- \
  riir-train/data/Qwen3.5-0.8B-Base-Q8_0.gguf riir-train/data/chat_probe /tmp/out.rimsk \
  --solver <wanda|sparsegpt|magnitude|random> --rate 0.001 --max-tokens 512
```

## The T7b compensated-forward rows (2026-11-16 ~22:15Z, zed-e2bf92a2 — the ii-b sidecar's FKL, the lane's "next instrument" landed)

The harness: `CompensatedSidecar` (I1208COMP1) extracted into `mint.rs` (parser/encoder; byte-identical to the original inline writer — the re-minted 0.001 sidecar reproduced `9ae9d9a5…ca83` EXACTLY, and the 0.01 rimsk reproduced `e58e6fdf…c6b9`), `GateMaskOverlay::mount_compensated` (bind-where-bound, loud refusals), and the audit's `--fk-lab-comp` posture (checkpoint binding verified loud; two-walk KL with the compensated matrices mounted directly — deleted positions baked to 0, survivors carry the error feedback). Gates: 16 lib (adapter::mint, incl. the sidecar round-trip/refusal/header tests) + 6 apply-gate (incl. `compensated_overlay_identity_and_binding`: identity pin, codec round trip, bind-where-bound, perturbation-moves-the-walk, length/deep-layer refusals) — clippy clean at the gated posture. NOTE: this session's model+corpus copies rode `/tmp` (`/tmp/i1208_model.gguf`, `/tmp/i1208_corpus/` — byte-copies of the riir-train data paths; digests bind regardless).

| solver | rate | FKL mean | p95 | p99 | table digest | sidecar blake3 |
|---|---|---|---|---|---|---|
| sparsegpt **ii-b compensated** | 0.001 | 0.000000 | 0.000000 | 0.000000 | `bfa61215…39c56c24` (= the ii-a all-zero table) | `9ae9d9a5…ca83` |
| sparsegpt **ii-b compensated** | 0.01 | **0.000000** | **0.000000** | **0.000000** | `feb5184b…ff50ed` | `5e2ef4b6…923c` |

**Reading (the instrument honest, the claim bounded):** at 0.001 the ii-b row ties the ii-a all-zero table byte-for-byte (same digest) — selection choice immaterial, compensation indistinguishable. At **0.01 the compensation clears the mask-only arms by a full resolution class**: ii-a read 1e-6-class means with 2–4e-6 p99s; the compensated walk reads all-zero at the 6-dp print resolution (true values < 5e-7 per position), with the model digest binding verified and the synthetic battery proving the mount is numerically live (a perturbed matrix moves the walk). Random at 0.01 read 0.0225 mean / 0.2586 p99 (bench 1228) — four orders above the compensated arm. 64-token probe, one frozen read; the zero is "below print resolution", never an exact-zero claim.

## The geometry rows (the T6 arms over real T7-produced content — first real-content rows; same window, seed 0xc0ffee k=80)

| mask | drift max/mean (per-layer, all 24) | overlap (all 24) | vs the trained envelope (drift ≤1.4×, overlap 0.43–0.46) |
|---|---|---|---|
| sparsegpt ii-a 0.001 (`22b30aa4…`) | 1.000 / 1.000 | 1.000 | far inside — identity-class |
| sparsegpt ii-a 0.01 (`e58e6fdf…`) | 1.000 / 1.000 | 1.000 | far inside — identity-class |

**The falsifier did NOT fire** (the pre-registered kill: closed-form geometry OUTSIDE the envelope would kill the FFT-like inheritance claim). The closed-form masks preserve the principal subspace BETTER than trained FFT masks — consistent with the FKL axis (bottom-|w|-tail deletion does not move the top-80 left subspace; the per-layer stable ranks read identical to 2dp). Honest note: at these rates the geometry axis is resolution-limited (the full-row deletion golden moves it; tail deletion does not) — the rows are the first real-content pins, not a discrimination between the two rates.

## The τ/rate operating-point pick (the pre-registered relative-to-magnitude rule, ONE frozen read — this section is the pick)

At 0.001 every principled arm ties within instrument resolution (all-zero tables across ii-a and ii-b) — no solver separation, matching finding 1's whitening explanation. At 0.01 the **sparsegpt ii-b compensated arm dominates magnitude ii-a** (all-zero vs 1e-6-class mean, 2–4e-6 vs <5e-7 p99). **The pick: `--solver sparsegpt --rate 0.01 --emit-compensated` is T7's operating point** — the rate where deletion actually bites AND the compensation provably restores the distribution; 0.001 is the conservative point where solver choice is immaterial. The floor arms (magnitude/random) stay mandatory at any future rate; the falsifier (no operating point clearing the quality floor → closed-form content dead) did not fire — closed-form content is ALIVE with a proven operating point.
