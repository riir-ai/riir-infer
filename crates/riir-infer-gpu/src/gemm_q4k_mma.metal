// gemm_q4k_mma.metal
// Plan 614 Phase-3 session-2 — the simdgroup-mma Q4_K verify kernel (the
// 2026-11-12 verdict's GO lane): llama.cpp kernel_mul_mm's class — Q4_K
// dequant → threadgroup f32 tiles → simdgroup fragment loads →
// register-resident simdgroup_multiply_accumulate — replacing the mpp
// matmul2d cooperative machinery whose serial staging→barrier→matmul2d→
// barrier structure Bench 1164 pinned as the ~1.2 ms per-call floor.
//
// Adaptations vs the session-1 design record (recorded, not silent):
// 1. The MSL spec leaves "the mapping of matrix elements to threads ...
//    unspecified" (§2.3.2), so the direct-register A-fragment fill the
//    record sketched is not spec-implementable. The fragment operands come
//    from simdgroup_load instead, and the performance core survives:
//    ONE threadgroup_barrier per K-tile (A and B tiles double-buffered —
//    the fill of tile k+1 overlaps the mma of tile k), no mpp cooperative
//    staging, the mma work register-resident.
// 2. The G1 byte contract carries verbatim: tile values are
//    `d * sc * q - dmin * mn` in f32 with the SAME expression order as
//    dequantize_row_q4_k (src/quant/q4k.rs), rounded to half ONCE (the
//    reference CPU GEMM consumes f16-rounded dequanted weights; the half
//    value is exact in the f32 fragment). get_scale_min_k4_msl ports
//    verbatim. Fragments are float (not half) because the mma has no
//    float←half×float operand combination and B (activations) must stay
//    f32 unrounded for the contract.
// 3. Geometry: tile 64×64, 4 simdgroups × 16 C-fragments (each simdgroup
//    owns a [32m × 32p] quadrant), K_TILE = 32 = SZ_SIMD 16 × NK_CHUNK 2 —
//    the staged kernel's (row, 16-k) dequant work-unit shape survives: one
//    unit per thread per K-tile.
//
// Block layout (llama.cpp-mul_mm-proven simdgroup_load/store mapping,
// transpose=false: fragment(col c, row r) = mem[c*epr + r]):
//   A block (kb, m8): element (k8, m8r) at 64*(kb*8 + m8) + 8*k8 + m8r
//     — a-fragment (col=k, row=m). Unit writes are stride-8 f32 scalars,
//     conflict-free (each 8-thread half covers 8 distinct banks).
//   B block (kb, p8): element (k8, p8r) at 64*(kb*8 + p8) + 8*k8 + p8r
//     — b-fragment (col=p, row=k).
//   C store: y[(rb + p)*M + (ra + m)] via dst[c*M + r] per fragment.
//
// Threadgroup budget: 32 KB exactly — A 2×8 KB + B 2×8 KB (the double
// buffers). The ragged-tile store path aliases the dead tile memory as
// scratch after the K-loop.
#ifndef MMA_Q4K_RELAXED
#define MMA_Q4K_RELAXED false
#endif

#include <metal_stdlib>
using namespace metal;

constant constexpr int K_TILE      = 32;
constant constexpr int TILE_M      = 64;
constant constexpr int TILE_P      = 64;
constant constexpr int NSG         = 4;    // simdgroups per threadgroup
constant constexpr int NUM_THREADS = 32 * NSG;
constant constexpr int QK_K        = 256;
constant constexpr int BLOCK_BYTES = 144;  // d(2)+dmin(2)+scales(12)+qs(128)
constant constexpr int TILE_ELEMS  = TILE_M * K_TILE;  // 2048 f32 per tile

// get_scale_min_k4 — the verbatim port of src/quant/q4k.rs (as in
// gemm_q4k_metal_tensor.metal).
static inline void get_scale_min_k4_msl(
    uint j, device const uchar* s, thread uint& sc, thread uint& mn) {
    if (j < 4u) {
        sc = s[j] & 63u;
        mn = s[j + 4u] & 63u;
    } else {
        sc = (s[j + 4u] & 0x0Fu) | ((s[j - 4u] >> 6) << 4);
        mn = (s[j + 4u] >> 4) | ((s[j] >> 6) << 4);
    }
}

// ---------------------------------------------------------------------------
// Y[P × M] = dequant_q4k(W[M × K]) @ X[P × K](f32)^T
//
// Buffers (identical bindings to gemm_q4k_tensor):
//   0: weight_q4k — packed Q4_K payloads [M × bpr × 144 B], byte-addressed
//   1: d_dmin — [M × bpr × 2] f32 (the handle's pre-converted block scales)
//   2: input_batch  X [P × K] row-major f32
//   3: output_batch Y [P × M] row-major f32 (DEDICATED buffer + blit)
//   4: k_input_dim, 5: m_features, 6: p_tokens
//
// Dispatch: grid = ((P + TILE_P - 1)/TILE_P, (M + TILE_M - 1)/TILE_M, 1),
// 128 threads/workgroup, threadgroup memory 32768 B.
// Shape laws: n (K) a multiple of 256 (the Q4_K block → K_TILE never
// ragged); m and p guarded (any size ≥ 1).
// ---------------------------------------------------------------------------
kernel void gemm_q4k_mma(
    device const uchar* weight_q4k    [[buffer(0)]],
    device const float* d_dmin        [[buffer(1)]],
    device const float* input_batch   [[buffer(2)]],
    device float*       output_batch  [[buffer(3)]],
    constant uint&      k_input_dim   [[buffer(4)]],
    constant uint&      m_features    [[buffer(5)]],
    constant uint&      p_tokens      [[buffer(6)]],
    threadgroup char*   shmem         [[threadgroup(0)]],
    uint3               tgpig         [[threadgroup_position_in_grid]],
    ushort              tiitg         [[thread_index_in_threadgroup]],
    ushort              sgitg         [[simdgroup_index_in_threadgroup]]
) {
    const int ra = tgpig.y * TILE_M;
    const int rb = tgpig.x * TILE_P;
    const uint bpr = k_input_dim / QK_K;

    threadgroup float* tiles = (threadgroup float*)(shmem);
    // Per K-tile i, buffer phase buf = i & 1:
    //   A tile: tiles[buf * TILE_ELEMS]
    //   B tile: tiles[TILE_ELEMS + buf * TILE_ELEMS]

    simdgroup_float8x8 mc[16];
    #pragma clang loop unroll(full)
    for (ushort i = 0; i < 16u; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    const ushort sgm = sgitg & 1u;   // m half [0, 32)
    const ushort sgp = sgitg >> 1u;  // p half [0, 32)

    const int n_tiles = (int)k_input_dim / K_TILE;
    for (int it = 0; it < n_tiles; ++it) {
        const int loop_k = it * K_TILE;
        const uint buf = (uint)(it & 1);
        threadgroup float* sa = tiles + buf * TILE_ELEMS;
        threadgroup float* sb = tiles + TILE_ELEMS + buf * TILE_ELEMS;

        // === A fill: one (row, 16-k) dequant unit per thread ===
        {
            const int row_local  = tiitg >> 1;              // [0, 64)
            const int kchunk     = tiitg & 1;               // [0, 2)
            const int k_pos      = loop_k + kchunk * 16;
            const int row_global = ra + row_local;
            if (row_global < (int)m_features) {
                const uint rg = (uint)row_global;
                const uint e     = (uint)k_pos % QK_K;
                const uint b     = (uint)k_pos / QK_K;
                const uint blk   = rg * bpr + b;
                device const uchar* sc_p = weight_q4k + blk * BLOCK_BYTES + 4u;
                device const uchar* qs_p = weight_q4k + blk * BLOCK_BYTES + 16u;

                const uint j = e >> 5;                      // sub-block [0, 8)
                uint sc, mn;
                get_scale_min_k4_msl(j, sc_p, sc, mn);
                const uint pair  = j >> 1;
                const bool low   = (e & 32u) == 0u;
                const uint byte0 = 32u * pair + (low ? (e & 63u) : ((e & 63u) - 32u));

                const float d    = d_dmin[blk * 2u];
                const float dmin = d_dmin[blk * 2u + 1u];
                const float d_sc = d * (float)sc;
                const float m_t  = dmin * (float)mn;

                const uint kb0 = (uint)(kchunk * 2);
                const uint m8  = (uint)row_local >> 3;
                const uint m8r = (uint)row_local & 7u;
                #pragma clang loop unroll(full)
                for (uint t = 0; t < 16u; ++t) {
                    const uint byte_v = qs_p[byte0 + t];
                    const uint nib = low ? (byte_v & 0x0Fu) : (byte_v >> 4);
                    const float w = d_sc * (float)nib - m_t;
                    // The ONE half-round (the byte contract), written to the
                    // A block as a stride-8 f32 scalar: element (k8, m8r) at
                    // 64*((kchunk*2 + t>>3)*8 + m8) + 8*(t&7) + m8r.
                    sa[64u * ((kb0 + (t >> 3)) * 8u + m8) + 8u * (t & 7u) + m8r]
                        = (float)(half)w;
                }
            }
        }

        // === B fill: one (token, 16-k) load unit per thread ===
        {
            const int p_local  = tiitg >> 1;
            const int kchunk   = tiitg & 1;
            const int p_global = rb + p_local;
            if (p_global < (int)p_tokens) {
                device const float* x = input_batch
                    + (uint)p_global * k_input_dim
                    + (uint)(loop_k + kchunk * 16);
                const uint p8  = (uint)p_local >> 3;
                const uint p8r = (uint)p_local & 7u;
                #pragma clang loop unroll(full)
                for (uint t = 0; t < 16u; ++t) {
                    const uint kb = (uint)(kchunk * 2 + (int)(t >> 3));
                    // b-fragment layout: [8p at stride x 8k contiguous] —
                    // element (p8r, k8) at 64*(kb*8 + p8) + 8*p8r + (t&7)
                    // (the a operand is [8k at stride x 8m contiguous]; the
                    // two differ — the identity lens is invariant to getting
                    // this wrong, the dot cases are not).
                    sb[64u * (kb * 8u + p8) + 8u * p8r + (t & 7u)] = x[t];
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === mma: this simdgroup's [32m × 32p] quadrant, 16 C-fragments ===
        #pragma clang loop unroll(full)
        for (ushort ik = 0; ik < 4u; ++ik) {
            simdgroup_float8x8 ma[4];
            simdgroup_float8x8 mb[4];
            #pragma clang loop unroll(full)
            for (ushort i = 0; i < 4u; ++i) {
                simdgroup_load(ma[i], sa + 64u * (ik * 8u + (4u * sgm + i)), 8u, ulong2(0u), false);
            }
            #pragma clang loop unroll(full)
            for (ushort i = 0; i < 4u; ++i) {
                simdgroup_load(mb[i], sb + 64u * (ik * 8u + (4u * sgp + i)), 8u, ulong2(0u), false);
            }
            #pragma clang loop unroll(full)
            for (ushort i = 0; i < 16u; ++i) {
                // llama.cpp-mul_mm operand order (mb first) — the fragment
                // role the hardware mma expects for this tile orientation.
                simdgroup_multiply_accumulate(mc[i], mb[i >> 2u], ma[i & 3u], mc[i]);
            }
        }
    }

    // === Store ===
    const bool full_m = (ra + TILE_M) <= (int)m_features;
    const bool full_p = (rb + TILE_P) <= (int)p_tokens;
    if (full_m && full_p) {
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < 16u; ++i) {
            const ushort mb = i & 3u;
            const ushort pb = i >> 2u;
            device float* dst = output_batch
                + (uint)(rb + 32 * (int)sgp + 8 * (int)pb) * m_features
                + (uint)(ra + 32 * (int)sgm + 8 * (int)mb);
            simdgroup_store(mc[i], dst, m_features, ulong2(0u), false);
        }
    } else {
        // Ragged tile: stage through a threadgroup scratch (the dead tile
        // memory aliases), then guarded scalar writes.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* scratch = tiles;
        #pragma clang loop unroll(full)
        for (ushort i = 0; i < 16u; ++i) {
            const ushort mb = i & 3u;
            const ushort pb = i >> 2u;
            simdgroup_store(mc[i],
                scratch + (uint)(32 * (int)sgp + 8 * (int)pb) * TILE_M
                        + (uint)(32 * (int)sgm + 8 * (int)mb),
                (uint)TILE_M, ulong2(0u), false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tiitg; idx < (uint)(TILE_M * TILE_P); idx += NUM_THREADS) {
            const uint pl = idx / TILE_M;
            const uint ml = idx % TILE_M;
            const int pg = rb + (int)pl;
            const int mg = ra + (int)ml;
            if (pg < (int)p_tokens && mg < (int)m_features) {
                output_batch[(uint)pg * m_features + (uint)mg] = scratch[idx];
            }
        }
    }
}
