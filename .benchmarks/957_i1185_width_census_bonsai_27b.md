# Bench 957 — i1185 T2-run: the Bonsai-2-27B per-layer ε-width census (the Cactus "How Wide Is Attention?" estimator, Research 657 §A2)

**Status:** DONE — the tool lane ran clean on the real checkpoint; the verdict is a TABLE, not a budget (no promotion, no pin edit — T3's needle gate comes first).

## Cell posture (the box state IS the claim)

- **Host:** 4090 (Windows, rig job `01a1bbeb-0311-7ca2-9da5-f1788063140b`, shared lane, 2026-11-08 23:08–23:29 +07). CPU-side lanes (the ternary census + the width census are CPU); GPU idle (1–3%) — GPU exclusivity not required by construction. Rig-serialized; no concurrent heavy CPU job (the queue's serialization is the guarantee).
- **HEAD:** riir-infer `496e739` (width_census.rs `463e2ce` + the bin's `--width-eps` `4e5f471`; script `678bb9d`+BOM fix `496e739`).
- **Checkpoint:** `Ternary-Bonsai-2-27B-PQ2_0.gguf` (ternary body, folded=true, 64 layers, **16 full-attention layers × 4 KV heads = 64 heads**).
- **Corpus:** `chat_probe` (2,713,703 tokens; first 512 consumed) — the 947/948 census lane's corpus, unchanged.
- **Window:** n=512. **Build:** release, features `sink_audit,deltanet_ternary_inference,bonsai2_hadamard`.
- **Engagement:** the v1 sink report's BLAKE3 is **identical across all three ε cells** (`82a11b47afd6…`) — the width flag provably did not touch the canonical payload; the width pass ran after it in the same process (`width census: … heads=64 censored=0` lines in each stdout.log).
- f_attn_token0 = **0.131020** → verdict **Undecided** → incumbent pin n_sink=4 stands (reproduces 948's cell shape; the sink verdict itself is NOT re-litigated here).

## The pre-registered ε grid results (per-KV-head width M̂_ε, window 512)

| ε | min | p25 | **median** | p75 | max | r_ent median | r_ent < 0.9 | r_ent > 1.1 |
|---|---|---|---|---|---|---|---|---|
| 0.1 | 80 | 296 | **376** | 456 | 512 | 1.043 | 18/64 | 28/64 |
| 0.2 | 32 | 192 | **256** | 312 | 496 | 0.918 | 30/64 | 16/64 |
| 0.3 | 8 | 128 | **168** | 216 | 480 | 0.804 | 40/64 | 16/64 |

Raw per-head tables: `i1185_width_census_eps{0.1,0.2,0.3}.json` (this directory, pulled from the 4090 cells byte-as-written).

## The readings (honest, per the 657 sketch's own caveats)

1. **The global pin 4 is far below every head's width at every ε tested.** Zero heads reach width ≤ 4 at ε = 0.2 (the narrowest is 32, layer 59/kvh 1). The shipped `kv_sink_window` pin 4 is a SINK pin (leading-mass concentration), not a WIDTH budget — the census says the two axes are different quantities on this checkpoint: heads carry wide effective support AND concentrated leading mass simultaneously. No budget re-pin is licensed by this table alone.
2. **Sublinear ε-growth, matching the paper's shape qualitatively**: median 376 → 256 → 168 as ε goes 0.1 → 0.2 → 0.3 (the paper reports 168–1,032 medians at ε=0.2 on its models; our 256 sits inside that family — a coincidence of scale, NOT a validation: different models, different window (512 vs 16k), our estimator choices are pre-registered OURS).
3. **Specialization is present and ε-growing**: r_ent < 1 heads go 18 → 30 → 40 of 64 as ε loosens (median r_ent 1.04 → 0.92 → 0.80). At ε=0.2–0.3 a majority of KV heads have real-query widths BELOW the entropy-matched isotropic control — the query distribution concentrates on dictionary subsets. This is the signal a per-head windowed-vs-full routing cell (the only league cell the source licenses) would read — parked until that cell is named.
4. **A narrow-head population exists at loose ε but is sparse**: the narrowest 12 heads at ε=0.2 cluster in late layers (59, 63, 3, 7, 31, 39). NOT the paper's dense-model sink-dumping population (their widths reach single digits); ours bottom out at 32. The hybrid's GDN layers carry the recycling load the paper's dense models push into narrow attention heads — consistent with 657's "may not exist in hybrids" caveat, now with numbers.
5. **Upper-bound estimator, unsaturated at this window**: 3 heads hit width > 480 at ε=0.1 (window 512 itself) — the paper's "unsaturated at 128k" caveat applies a fortiori; the 512-window medians are FLOORS on wider contexts, direction unknown without a wider-window cell.

## What this does NOT license

- No `kv_sink_window` re-pin (the incumbent 4 stands; changing it needs T3's needle-retention gate + the lossy-surface per-family rule).
- No runtime head-skipping / windowed routing (offline measurement only, per the source).
- No cross-family comparison against Cactus's own numbers (different lossy surfaces; the lineage naming law applies).

## Next (the gate, pre-registered in 1185)

- T3's needle-retention cell: ANY future budget derived from this table must hold needle retention at the incumbent's level on the needle_eviction_gate lane or be refused. The percentile-hop sanity arm rides the same cell (this bench's medians use index n/2 — no 0.99-percentile claims were made; the audit class is armed for any future percentile quote).
- The wider-window cell (1024/2048) is the cheap follow-up if the budget question stays live; the estimator's cost law (issue 1185 module doc) prices it before any run.

Session: zed-amo (amo), epoch 1794150461. Refs: riir-infer `.issues/1185` · katgpt-rs `.issues/987` lead A2 · Research 657 §A2 · `.benchmarks/947/948` (the incumbent census cells).
