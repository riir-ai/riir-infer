# Bench 1246 — Issue 1238: the frame-size-keyed GPU caches, bucketed by row count

**Status:** RECORDED 2026-11-19 (katgpt-rs-93@amo) — T1 census + T2 identity + T3 G4.

## T1 — the growth (analytic key census)

Every per-frame Metal/CubeCL scratch on the prefill GEMM lanes was keyed by its EXACT byte length and never
evicted: the zero-copy GEMM's dedicated output (`output_cache`, incumbent `dispatch` + the twin's
`dispatch_f16b*`/`dispatch_scratch`), the f16-B twin's cast input (`input_cache`), the i1226 F2 shadow pool
(`f16b_shadow_pool`), and the same class in `gemm_dense_f16_metal_zero_copy` + `attention_prefill_metal_tensor`.
The keys are pure arithmetic in the frame row count `p`, so the census is computed from the key formulas rather
than sampled: `census.py` in the logs dir, Bonsai-2-27B widths read from the GGUF header (hidden 5120,
attn/GDN 6144, mlp 17408; GEMM out dims 12288 / 1024 / 5120 / 10240 / 6144 / 48 / 17408). The mix is every
tail `p = 1..=chunk` (one serving process sees each final-frame length eventually). Upper bound: every twin GEMM
site counted; the dense/attention siblings are excluded from the numbers.

| chunk | key | shadow keys / GiB | twin output keys / GiB | total GiB | fixed-width floor GiB |
|---:|---|---:|---:|---:|---:|
| 128 (dist default) | exact (pre) | 350 / 0.42 | 665 / 1.30 | **1.72** | 0.032 |
| 128 | bucketed (landed) | 24 / 0.01 | 56 / 0.05 | **0.06** | 0.032 |
| 512 (Bench 1245) | exact (pre) | 1396 / 6.65 | 2656 / 20.74 | **27.40** | 0.127 |
| 512 | bucketed | 30 / 0.05 | 70 / 0.20 | **0.25** | 0.127 |
| 4096 (single-node cap) | exact (pre) | 11166 / 425 | 21239 / 1326 | 1751 (OOM long before) | 1.016 |
| 4096 | bucketed | 39 / 0.44 | 91 / 1.60 | **2.03** | 1.016 |

Verdict: the growth matters (27 GiB of process-lifetime buffers at the Bench-1245 chunk), so T2 ran. The issue
named the shadow pool + `input_cache`; the dominant term is the twin's `output_cache` (≈ 3× the shadow), and
under F2's default `input_cache` takes no new keys at all (every served site has a shadow).

## T2 — the bound

`riir-infer-gpu/src/frame_bucket_cache.rs` (`FrameBucketCache<B>`): key `(row_bytes, rows.next_power_of_two())`,
buffer allocated at the bucket size. Per row width at most `log2(max_rows)+1` live buffers summing to
`< 2 × bucket(max_rows)` rows, so resident bytes are always `< 4 ×` the full-frame floor. One mechanism for all
six cache sites. Every consumer binds the exact extent (`from_raw_parts(.., total/2)`, the blit's
`output_bytes`, the tap's `p * m` read, the kernels' `p`/`m`/`n` uniforms), so the bucket's slack is never
read back. `row_bytes` stays in the key, so within one frame two widths never share a buffer, which keeps the
aliasing classes the exact key had and drops its accidental `p1·w1 == p2·w2` collisions.

### Identity (pre vs post, end to end)

Shared job `01a1f454` on amo (correctness only, no timing claim). Identity-only mode of
`i1226_f2_price_cell` (B C C B + one A pass). PRE is the parent commit plus only the digest-print
test edit (detached worktree); POST is the landed tree. `--release`, Bonsai-2-27B PQ2_0, relaxed posture.

| shape | PRE digest B/C | POST digest B/C | PRE digest A (f32) | POST digest A (f32) | C == B |
|---|---|---|---|---|---|
| chunk 128 × 32 + tail 37 (dist default) | `b16d8f8397008d06` | `b16d8f8397008d06` | `7c4aabbeb2aabb72` | `7c4aabbeb2aabb72` | × 4, 12672 casts saved, misaligned 0 |
| chunk 96 × 8 + tail 37 (non-pow2 frame: slack exercised) | `b6bf3c9a5375b3e3` | `b6bf3c9a5375b3e3` | `7c68bbe2dab357bb` | `7c68bbe2dab357bb` | × 4, 3456 saved, 0 |
| chunk 512 × 8 + tail 2 | `f19e1c9279557c00` | `f19e1c9279557c00` | `cbdf9103745a0026` | `cbdf9103745a0026` | × 4, 3456 saved, 0 |

Byte-identical pre vs post on every arm and shape. The A arm covers the incumbent `dispatch` output cache,
and B/C cover the twin, the cast input and the shadow pool. Not exercised by this cell: the
dense-f16 GEMM and attention-prefill siblings; their consumers were read for exact-extent binds
(blit `output_bytes` / `out_bytes`). Log: `identity_pre_post.log`.

## T3 — G4

`frame_bucket_cache::tests::fixed_width_allocates_once`: 100 dispatches at a fixed width → 1 allocation.
`tail_sweep_is_bounded` pins the key-count formula and the `< 4 × floor` bound at chunk 128/384/512/4096.
