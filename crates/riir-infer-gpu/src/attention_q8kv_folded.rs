//! FOSCAR affine-folded Q8KV decode attention (Issue 1211, arXiv:2611.09722).
//!
//! Exact-arithmetic sibling of [`crate::attention_q8kv_cubecl`]: the same
//! online-softmax flash decode over the same Q8_0 cache bytes, with the
//! per-element dequant FMUL folded into the matmul operands (FOSCAR §4.1;
//! Q8_0 is symmetric, z = 0, so only scale-folding exists here):
//!
//! - **T1 — QK acc-fold.** The key scale varies along the reduction axis AND
//!   per position (token-wise groups), so the fold is per (pos, block): a raw
//!   32-wide `query·i8` partial (no per-element scale FMUL), then ONE multiply
//!   of the partial by `block_scale(pos, block)`. The Gemma-2 softcap applies
//!   after the fold produces the exact score — ordering safe.
//! - **T2 — PV operand-fold into P.** The value scale varies along the PV
//!   reduction axis (positions), so the scale rides the probabilities:
//!   `p'(pos_i, block) = exp_i · v_scale(pos_i, block)` is computed ONCE per
//!   (position, block) into shared memory (2048 f32 = 8 KB on top of the
//!   256-f32 score/exp scratch = 9 KB total), and each thread FMAs the shared
//!   `p'` against its raw converted i8 — no per-element scale load or FMUL.
//!   The online-softmax denominator division stays after accumulation,
//!   unchanged.
//!
//! The fold changes f32 ROUNDING (real-exact, not bit-exact); the parity gate
//! is the pre-registered falsification arm of Issue 1211: folded-vs-f64 error
//! ≤ 2 × incumbent-vs-f64 error on the fixture (a measured class, not an
//! assumed tolerance). The incumbent kernel in `attention_q8kv_cubecl.rs` is
//! deliberately untouched — it is the A/B control.
//!
//! # Sink guard (Issue 716) — fold-exempt
//!
//! The first `sink_rows` positions ride the lossless f32 sidecar in both
//! phases, bypassing the fold exactly as they bypass the quantized path in
//! the incumbent: QK reads the sidecar row directly (identical to the
//! incumbent — nothing to fold there), and PV accumulates the UNfolded
//! weight (`exp_i`, from the score/exp scratch) against the sidecar f32.
//!
//! # Demote path
//!
//! PROMOTED into the q8kv dispatch path (Issue 1211 T3: the pre-registered
//! gate PASSED — geomean 0.9360× folded/incumbent at 512/4096/32768 on
//! Metal ≥ the 5% bar, parity held at every shape: folded-vs-f64 ≤ 2×
//! incumbent-vs-f64, both postures). The three `gemma2_cubecl/dispatch.rs`
//! launch sites route through [`q8kv_fold_enabled`]: unset = folded (the
//! promoted arm), `RIIR_Q8KV_FOLD=0` (the exact literal; "false"/"off" the
//! documented aliases) restores the incumbent byte-identically. The
//! incumbent kernel and its launchers are never edited.
//!
//! # CubeCL constraints
//!
//! Same workarounds as the incumbent module: no conditional expressions as
//! values (NativeExpand), unrolled reductions with `sync_cube()` between
//! steps, hardcoded Gemma 2 2B constants, `#[cube(launch_unchecked)]`.

//! Local copies of the decode kernel's byte-extraction pair
//! (`attention_q8kv_cubecl.rs` `get_byte_q8`/`sign_extend_i8` — 3-liners,
//! deliberately NOT imported: cross-module `#[cube]` fn calls go through
//! macro codegen this crate has never exercised; the duplication is cheaper
//! than the risk — the same posture `qwen_prefill_q8kv_cubecl.rs` records).

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

#[cfg(feature = "cubecl_runtime")]
use crate::attention_cubecl::AttentionParams;

#[cfg(all(test, feature = "cubecl_runtime"))]
use crate::attention_q8kv_cubecl::Q8KVBuffers;

/// The promoted arm's demote latch (the env-switch law): unset = folded
/// (the Issue-1211 promoted default); the exact literal `0` ("false"/
/// "off" documented aliases) restores the incumbent decode path
/// byte-identically — the incumbent kernel/launchers are the demote
/// target and are never edited. Read once per process.
#[cfg(feature = "cubecl_runtime")]
pub fn q8kv_fold_enabled() -> bool {
    static FOLD: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FOLD.get_or_init(|| {
        std::env::var("RIIR_Q8KV_FOLD").map_or(true, |s| !matches!(s.trim(), "0" | "false" | "off"))
    })
}

// ---------------------------------------------------------------------------
// CubeCL helper functions (local copies — see the module-level note)
// ---------------------------------------------------------------------------

/// Extract byte at index (0–3) from a u32 word (little-endian).
#[cfg(feature = "cubecl_runtime")]
#[cube]
fn get_byte_q8(word: u32, idx: u32) -> u32 {
    (word >> (idx * 8u32)) & 0xFFu32
}

/// Sign-extend a u8 byte value to f32.
///
/// Two if-blocks (no conditional expressions as values) — the CubeCL
/// NativeExpand macro bug workaround the decode kernel documents.
#[cfg(feature = "cubecl_runtime")]
#[cube]
fn sign_extend_i8(byte_val: u32) -> f32 {
    let mut result = byte_val as f32;
    if byte_val >= 128u32 {
        result = byte_val as f32 - f32::new(256.0f32);
    }
    result
}

// ---------------------------------------------------------------------------
// Flash attention decode kernel — Q8_0 with FOSCAR affine folding
// ---------------------------------------------------------------------------

/// CubeCL flash attention decode with online softmax, softcapping, and Q8_0
/// KV consumed through FOSCAR affine folding (Issue 1211).
///
/// Identical algorithm, buffers, dispatch geometry, and sink-guard semantics
/// as [`crate::attention_q8kv_cubecl::attention_decode_q8kv`] — read that
/// fn's doc for the buffer contract (caller-owns-correctness, Issue 710 H11,
/// the Bench 642 allocation-size law all apply verbatim). Differences:
///
/// 1. QK phase: per (pos, block) raw 32-wide partial, one scale FMUL per
///    block (acc-fold).
/// 2. PV phase: a 2048-f32 shared `p'` table built once per tile
///    (`exp_i · v_scale(pos_i, block)`), each thread then FMAs `p' · i8_f32`
///    with no scale load or FMUL (operand-fold into P). Sink rows bypass the
///    fold: they accumulate the unfolded weight against the f32 sidecar.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn attention_decode_q8kv_folded(
    query: &[f32],
    kv_qs: &[u32],
    kv_scales: &[f32],
    sink_kv: &[f32],
    attn_out: &mut [f32],
) {
    // ── Gemma 2 2B constants (identical to the incumbent) ──
    let head_dim = 256u32;
    let n_head = 8u32;
    let n_kv_head = 4u32;
    let n_blocks = 8u32; // head_dim / Q8_BLOCK_SIZE = 256 / 32
    let u32_per_block = 8u32; // Q8_BLOCK_SIZE / 4 = 32 / 4
    let u32_per_head = n_blocks * u32_per_block; // 64 u32s per (pos, kv_head)
    let kv_stride_q8 = n_kv_head * u32_per_head; // 256 u32s per position
    let kv_scale_stride = n_kv_head * n_blocks; // 32 f32s per position
    let softcap = f32::new(50.0f32);
    let scale = f32::new(0.0625f32); // 1/√256
    let cube_size = 256u32;

    // ── Derive dimensions from array lengths ──
    let kv_qs_half = kv_qs.len() as u32 / 2u32;
    let kv_scales_half = kv_scales.len() as u32 / 2u32;
    let n_positions = kv_scales_half / kv_scale_stride;

    // ── Issue 716 sink sidecar geometry ──
    let kv_f32_stride = n_kv_head * head_dim; // 1024 f32s per position
    let sink_half = sink_kv.len() as u32 / 2u32;
    let sink_rows = sink_half / kv_f32_stride;

    // ── Workgroup/thread assignment ──
    let head_idx = ABSOLUTE_POS as u32 / cube_size;
    let tid = UNIT_POS;

    // Guard: skip if overdispatched beyond n_head
    if head_idx >= n_head {
        terminate!();
    }

    let head_off = head_idx * head_dim;

    // Guard: no positions — write zeros and exit
    if n_positions == 0u32 {
        if tid < head_dim {
            attn_out[(head_off + tid) as usize] = f32::new(0.0f32);
        }
        terminate!();
    }

    // GQA: compute which KV head this query head attends to
    let kv_group = head_idx * n_kv_head / n_head;

    // Whether this thread owns a valid output dimension
    let valid_dim = tid < head_dim;

    // ── Shared memory: score/exp scratch (1 KB, incumbent) ──
    let mut smem = Shared::<[f32]>::new_slice(256usize);

    // ── Issue 1211 T2: the folded-P table (8 KB) ──
    // p_prime[i * n_blocks + b] = exp_i · v_scale(tile_base + i, b), rebuilt
    // once per tile. Layout is position-major so the builder's scale loads
    // are coalesced (consecutive slots share the position row) and the
    // consumer's reads broadcast (all 32 threads of a block read the same
    // slot per position).
    let mut p_prime = Shared::<[f32]>::new_slice(2048usize);

    // ── Online softmax running state (per-thread) ──
    let mut running_max = f32::new(-1e30f32);
    let mut running_sum = f32::new(0.0f32);
    let mut running_out = f32::new(0.0f32);

    // ── Process KV positions in tiles of 256 ──
    let n_tiles = n_positions.div_ceil(cube_size);

    let mut tile = 0u32;
    while tile < n_tiles {
        let tile_base = tile * cube_size;
        let pos = tile_base + tid;
        let valid_pos = pos < n_positions;

        // ── Phase 1: Softcapped Q·K score — T1 acc-fold ──
        let mut my_score = f32::new(-1e30f32);
        if valid_pos {
            let mut dot = f32::new(0.0f32);
            if pos < sink_rows {
                // Issue 716 sink guard: lossless f32 sidecar row (keys half).
                // Fold-exempt by construction — the sidecar is already exact.
                let k_base = pos * kv_f32_stride + kv_group * head_dim;
                let mut dim = 0u32;
                while dim < head_dim {
                    dot += query[(head_off + dim) as usize] * sink_kv[(k_base + dim) as usize];
                    dim += 1u32;
                }
            } else {
                // T1 acc-fold: per (pos, block) a raw 32-wide query·i8 partial
                // (no per-element scale FMUL), then ONE partial multiply by
                // the block scale. Same traversal order as the incumbent
                // (blocks ascending, elements ascending within a block) —
                // the fold regroups the reduction, it does not reorder it.
                let mut block_idx = 0u32;
                while block_idx < n_blocks {
                    let scale_idx = pos * kv_scale_stride + kv_group * n_blocks + block_idx;
                    let block_scale = kv_scales[scale_idx as usize];

                    let qs_block_base =
                        pos * kv_stride_q8 + kv_group * u32_per_head + block_idx * u32_per_block;

                    let mut partial = f32::new(0.0f32);
                    let mut e = 0u32;
                    while e < 32u32 {
                        let word_idx = qs_block_base + e / 4u32;
                        let byte_in_word = e % 4u32;
                        let word = kv_qs[word_idx as usize];
                        let byte_val = get_byte_q8(word, byte_in_word);
                        let i8_val = sign_extend_i8(byte_val);

                        let dim = block_idx * 32u32 + e;
                        partial += query[(head_off + dim) as usize] * i8_val;
                        e += 1u32;
                    }
                    dot += block_scale * partial;
                    block_idx += 1u32;
                }
            }
            let raw = dot * scale;
            my_score = softcap * f32::tanh(raw / softcap);
        }
        smem[tid as usize] = my_score;
        sync_cube();

        // ── Phase 2: Unrolled parallel max reduction → tile_max ──
        if tid < 128u32 && smem[(tid + 128u32) as usize] > smem[tid as usize] {
            smem[tid as usize] = smem[(tid + 128u32) as usize];
        }
        sync_cube();
        if tid < 64u32 && smem[(tid + 64u32) as usize] > smem[tid as usize] {
            smem[tid as usize] = smem[(tid + 64u32) as usize];
        }
        sync_cube();
        if tid < 32u32 && smem[(tid + 32u32) as usize] > smem[tid as usize] {
            smem[tid as usize] = smem[(tid + 32u32) as usize];
        }
        sync_cube();
        if tid < 16u32 && smem[(tid + 16u32) as usize] > smem[tid as usize] {
            smem[tid as usize] = smem[(tid + 16u32) as usize];
        }
        sync_cube();
        if tid < 8u32 && smem[(tid + 8u32) as usize] > smem[tid as usize] {
            smem[tid as usize] = smem[(tid + 8u32) as usize];
        }
        sync_cube();
        if tid < 4u32 && smem[(tid + 4u32) as usize] > smem[tid as usize] {
            smem[tid as usize] = smem[(tid + 4u32) as usize];
        }
        sync_cube();
        if tid < 2u32 && smem[(tid + 2u32) as usize] > smem[tid as usize] {
            smem[tid as usize] = smem[(tid + 2u32) as usize];
        }
        sync_cube();
        if tid < 1u32 && smem[1usize] > smem[0usize] {
            smem[0usize] = smem[1usize];
        }
        sync_cube();
        let tile_max = smem[0usize];
        sync_cube();

        // ── Phase 3: Compute exp(score - tile_max) ──
        let mut my_exp = f32::new(0.0f32);
        if valid_pos {
            my_exp = f32::exp(my_score - tile_max);
        }
        smem[tid as usize] = my_exp;
        sync_cube();

        // ── Phase 4a (Issue 1211 T2): build the folded-P table ──
        // 2048 slots, 8 per thread: slot = tid + k·cube_size, position-major
        // (i = slot / n_blocks, b = slot % n_blocks). The v_scale load is
        // guarded on pos_i (an invalid tile position must never touch the
        // global buffer); the exp factor smem[i_p] is 0 there anyway.
        let mut k = 0u32;
        while k < 8u32 {
            let slot = tid + k * cube_size;
            let i_p = slot / n_blocks;
            let b_p = slot % n_blocks;
            let pos_i = tile_base + i_p;
            let mut pp = f32::new(0.0f32);
            if pos_i < n_positions {
                let v_scale_idx =
                    kv_scales_half + pos_i * kv_scale_stride + kv_group * n_blocks + b_p;
                pp = smem[i_p as usize] * kv_scales[v_scale_idx as usize];
            }
            p_prime[slot as usize] = pp;
            k += 1u32;
        }
        sync_cube();

        // ── Phase 4b: Weighted value accumulation against the folded operands ──
        let mut tile_val = f32::new(0.0f32);
        if valid_dim {
            // Precompute block/element for this thread's output dimension
            let v_block_idx = tid / 32u32;
            let v_element = tid % 32u32;
            let v_word_offset = v_element / 4u32;
            let v_byte_in_word = v_element % 4u32;

            let mut i = 0u32;
            while i < cube_size {
                let pos_i = tile_base + i;
                if pos_i < n_positions {
                    let v_qs_word_idx = kv_qs_half
                        + pos_i * kv_stride_q8
                        + kv_group * u32_per_head
                        + v_block_idx * u32_per_block
                        + v_word_offset;
                    let v_qs_word = kv_qs[v_qs_word_idx as usize];
                    let v_byte = get_byte_q8(v_qs_word, v_byte_in_word);
                    let v_i8 = sign_extend_i8(v_byte);

                    if pos_i < sink_rows {
                        // Issue 716 sink guard (fold-exempt): the f32 sidecar
                        // row rides the UNfolded weight — smem[i], not p'.
                        tile_val += smem[i as usize]
                            * sink_kv[(sink_half
                                + pos_i * kv_f32_stride
                                + kv_group * head_dim
                                + tid) as usize];
                    } else {
                        // T2: the scale rides p' — no per-element scale load
                        // or FMUL. The shared read broadcasts across the 32
                        // threads of this output block.
                        tile_val += p_prime[(i * n_blocks + v_block_idx) as usize] * v_i8;
                    }
                }
                i += 1u32;
            }
        }

        // ── Phase 5: Unrolled parallel sum reduction → tile_sum ──
        sync_cube();
        if tid < 128u32 {
            smem[tid as usize] = smem[tid as usize] + smem[(tid + 128u32) as usize];
        }
        sync_cube();
        if tid < 64u32 {
            smem[tid as usize] = smem[tid as usize] + smem[(tid + 64u32) as usize];
        }
        sync_cube();
        if tid < 32u32 {
            smem[tid as usize] = smem[tid as usize] + smem[(tid + 32u32) as usize];
        }
        sync_cube();
        if tid < 16u32 {
            smem[tid as usize] = smem[tid as usize] + smem[(tid + 16u32) as usize];
        }
        sync_cube();
        if tid < 8u32 {
            smem[tid as usize] = smem[tid as usize] + smem[(tid + 8u32) as usize];
        }
        sync_cube();
        if tid < 4u32 {
            smem[tid as usize] = smem[tid as usize] + smem[(tid + 4u32) as usize];
        }
        sync_cube();
        if tid < 2u32 {
            smem[tid as usize] = smem[tid as usize] + smem[(tid + 2u32) as usize];
        }
        sync_cube();
        if tid < 1u32 {
            smem[0usize] = smem[0usize] + smem[1usize];
        }
        sync_cube();
        let tile_sum = smem[0usize];

        // ── Phase 6: Online softmax update ──
        let mut new_max = running_max;
        if tile_max > new_max {
            new_max = tile_max;
        }
        let prev_corr = f32::exp(running_max - new_max);
        let curr_corr = f32::exp(tile_max - new_max);

        running_sum = running_sum * prev_corr + tile_sum * curr_corr;
        running_out = running_out * prev_corr + tile_val * curr_corr;
        running_max = new_max;

        sync_cube();
        tile += 1u32;
    }

    // ── Final: Normalize and write output ──
    if valid_dim {
        attn_out[(head_off + tid) as usize] = running_out / running_sum;
    }
}

// ---------------------------------------------------------------------------
// Launchers — mirrors of the incumbent pair, dispatching the folded kernel
// ---------------------------------------------------------------------------

/// FOSCAR-folded launch surface over the Q8KV buffer pair (Issue 1211).
///
/// Mirror of [`crate::attention_q8kv_cubecl::AttentionQ8KVCubeCL`] with the
/// same guard discipline (asserted Gemma 2 2B constants, the Issue 515 T2
/// binding-derives-units guards, the Bench 642 allocation-size law), calling
/// the folded kernel. No serving path calls this — see the module doc for
/// the promotion/demotion law.
#[cfg(feature = "cubecl_runtime")]
pub struct AttentionQ8KVFolded;

#[cfg(feature = "cubecl_runtime")]
impl AttentionQ8KVFolded {
    /// Launch the folded flash attention decode kernel (no sink sidecar).
    ///
    /// Buffer contract identical to
    /// [`crate::attention_q8kv_cubecl::AttentionQ8KVCubeCL::launch`] — the
    /// folded kernel reads the same bytes with the same derived dimensions.
    ///
    /// # Panics
    ///
    /// Panics if params don't match Gemma 2 2B constants (n_head=8, n_kv_head=4,
    /// head_dim=256).
    pub fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        query_handle: Handle,
        kv_qs_handle: Handle,
        kv_scales_handle: Handle,
        attn_out_handle: Handle,
        params: &AttentionParams,
    ) {
        assert_eq!(params.n_head, 8, "Only Gemma 2 2B n_head=8 supported");
        assert_eq!(params.n_kv_head, 4, "Only Gemma 2 2B n_kv_head=4 supported");
        assert_eq!(
            params.head_dim, 256,
            "Only Gemma 2 2B head_dim=256 supported"
        );

        let n_head = params.n_head as u32;
        let q_len = params.n_head * params.head_dim;

        let kv_stride_q8 = params.n_kv_head * 8 * 8; // 256 u32s
        let kv_scale_stride = params.n_kv_head * 8; // 32 f32s
        let total_qs_len = 2 * params.n_positions * kv_stride_q8;
        let total_scales_len = 2 * params.n_positions * kv_scale_stride;

        crate::cubecl_runtime::assert_binding_derives_units(
            &kv_qs_handle,
            2 * kv_stride_q8,
            params.n_positions,
            "AttentionQ8KVFolded::launch kv_qs",
        );
        crate::cubecl_runtime::assert_binding_derives_units(
            &kv_scales_handle,
            2 * kv_scale_stride,
            params.n_positions,
            "AttentionQ8KVFolded::launch kv_scales",
        );

        // Issue 716 dummy: 1 f32 allocation → sink_rows derives 0 → the sink
        // branches are never taken (Bench 642 allocation-size law; the
        // 515 T2 loud assert pins it).
        let no_sink_handle = client.empty(core::mem::size_of::<f32>());
        crate::cubecl_runtime::assert_binding_derives_units(
            &no_sink_handle,
            2 * params.n_kv_head * params.head_dim,
            0,
            "AttentionQ8KVFolded::launch no_sink dummy",
        );
        // SAFETY: Caller guarantees correct buffer sizes.
        unsafe {
            attention_decode_q8kv_folded::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_head, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(query_handle, q_len),
                BufferArg::from_raw_parts(kv_qs_handle, total_qs_len),
                BufferArg::from_raw_parts(kv_scales_handle, total_scales_len),
                BufferArg::from_raw_parts(no_sink_handle, 1),
                BufferArg::from_raw_parts(attn_out_handle, q_len),
            );
        }
    }

    /// Launch the folded kernel with the Issue 716 sink guard.
    ///
    /// Buffer contract identical to
    /// [`crate::attention_q8kv_cubecl::AttentionQ8KVCubeCL::launch_with_sink`].
    ///
    /// # Panics
    ///
    /// Panics if params don't match Gemma 2 2B constants (n_head=8,
    /// n_kv_head=4, head_dim=256) or `sink_rows > n_positions`.
    #[cfg(feature = "q8kv_sink_guard")]
    pub fn launch_with_sink<R: Runtime>(
        client: &ComputeClient<R>,
        query_handle: Handle,
        kv_qs_handle: Handle,
        kv_scales_handle: Handle,
        sink_kv_handle: Handle,
        attn_out_handle: Handle,
        sink_rows: usize,
        params: &AttentionParams,
    ) {
        assert_eq!(params.n_head, 8, "Only Gemma 2 2B n_head=8 supported");
        assert_eq!(params.n_kv_head, 4, "Only Gemma 2 2B n_kv_head=4 supported");
        assert_eq!(
            params.head_dim, 256,
            "Only Gemma 2 2B head_dim=256 supported"
        );
        assert!(
            sink_rows <= params.n_positions,
            "sink_rows ({sink_rows}) must be <= n_positions ({})",
            params.n_positions
        );

        let n_head = params.n_head as u32;
        let q_len = params.n_head * params.head_dim;

        let kv_stride_q8 = params.n_kv_head * 8 * 8;
        let kv_scale_stride = params.n_kv_head * 8;
        let total_qs_len = 2 * params.n_positions * kv_stride_q8;
        let total_scales_len = 2 * params.n_positions * kv_scale_stride;
        let total_sink_len = 2 * sink_rows * params.n_kv_head * params.head_dim;

        crate::cubecl_runtime::assert_binding_derives_units(
            &kv_qs_handle,
            2 * kv_stride_q8,
            params.n_positions,
            "AttentionQ8KVFolded::launch_with_sink kv_qs",
        );
        crate::cubecl_runtime::assert_binding_derives_units(
            &kv_scales_handle,
            2 * kv_scale_stride,
            params.n_positions,
            "AttentionQ8KVFolded::launch_with_sink kv_scales",
        );
        crate::cubecl_runtime::assert_binding_derives_units(
            &sink_kv_handle,
            2 * params.n_kv_head * params.head_dim,
            sink_rows,
            "AttentionQ8KVFolded::launch_with_sink sink_kv",
        );

        // SAFETY: Caller guarantees correct buffer sizes.
        unsafe {
            attention_decode_q8kv_folded::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_head, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(query_handle, q_len),
                BufferArg::from_raw_parts(kv_qs_handle, total_qs_len),
                BufferArg::from_raw_parts(kv_scales_handle, total_scales_len),
                BufferArg::from_raw_parts(sink_kv_handle, total_sink_len),
                BufferArg::from_raw_parts(attn_out_handle, q_len),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests — the Issue 1211 falsification arm (parity) on the GPU fixture
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "cubecl_runtime"))]
mod tests {
    use super::*;
    use crate::cubecl_runtime::ActiveRuntime;
    use crate::cubecl_runtime::CubeCLContext;

    /// Gemma 2 2B test constants (mirror of the incumbent module's).
    const N_HEAD: usize = 8;
    const N_KV_HEAD: usize = 4;
    const HEAD_DIM: usize = 256;
    const SOFTCAP: f32 = 50.0;
    const SCALE: f32 = 0.0625; // 1/√256

    /// Deterministic sin/cos fixture — the incumbent module's test data.
    fn make_test_data(n_positions: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let q_len = N_HEAD * HEAD_DIM;
        let kv_len = n_positions * N_KV_HEAD * HEAD_DIM;

        let query: Vec<f32> = (0..q_len)
            .map(|i| ((i as f32) * 0.01).sin() * 0.5)
            .collect();
        let keys: Vec<f32> = (0..kv_len)
            .map(|i| ((i as f32) * 0.02).cos() * 0.3)
            .collect();
        let values: Vec<f32> = (0..kv_len)
            .map(|i| ((i as f32) * 0.03).sin() * 0.2)
            .collect();

        (query, keys, values)
    }

    /// Run the INCUMBENT Q8_0 attention kernel (the A/B control) and return
    /// the output. Local copy of the incumbent module's test runner — its
    /// tests module is private, and sharing would couple the fixture lanes.
    fn run_incumbent(
        client: &ComputeClient<ActiveRuntime>,
        query: &[f32],
        keys: &[f32],
        values: &[f32],
        n_positions: usize,
    ) -> Vec<f32> {
        let params = AttentionParams {
            n_head: N_HEAD,
            n_kv_head: N_KV_HEAD,
            head_dim: HEAD_DIM,
            n_positions,
            softcap: SOFTCAP,
            scale: SCALE,
        };

        let q8_bufs = Q8KVBuffers::quantize_kv(keys, values, N_KV_HEAD, HEAD_DIM);

        let q_len = N_HEAD * HEAD_DIM;
        let query_handle = client.create_from_slice(f32::as_bytes(query));
        let kv_qs_handle =
            client.create_from_slice(bytemuck::cast_slice::<u32, u8>(&q8_bufs.kv_qs));
        let kv_scales_handle = client.create_from_slice(f32::as_bytes(&q8_bufs.kv_scales));
        let out_handle = client.empty(q_len * core::mem::size_of::<f32>());

        crate::attention_q8kv_cubecl::AttentionQ8KVCubeCL::launch::<ActiveRuntime>(
            client,
            query_handle,
            kv_qs_handle,
            kv_scales_handle,
            out_handle.clone(),
            &params,
        );

        let bytes = client
            .read_one(out_handle)
            .expect("should read incumbent q8kv output");
        f32::from_bytes(&bytes).to_vec()
    }

    /// Run the INCUMBENT kernel with the sink guard (the A/B control).
    #[cfg(feature = "q8kv_sink_guard")]
    fn run_incumbent_with_sink(
        client: &ComputeClient<ActiveRuntime>,
        query: &[f32],
        keys: &[f32],
        values: &[f32],
        n_positions: usize,
        sink_rows: usize,
    ) -> Vec<f32> {
        let params = AttentionParams {
            n_head: N_HEAD,
            n_kv_head: N_KV_HEAD,
            head_dim: HEAD_DIM,
            n_positions,
            softcap: SOFTCAP,
            scale: SCALE,
        };

        let q8_bufs =
            Q8KVBuffers::quantize_kv_with_sink(keys, values, N_KV_HEAD, HEAD_DIM, sink_rows);

        let q_len = N_HEAD * HEAD_DIM;
        let query_handle = client.create_from_slice(f32::as_bytes(query));
        let kv_qs_handle =
            client.create_from_slice(bytemuck::cast_slice::<u32, u8>(&q8_bufs.kv_qs));
        let kv_scales_handle = client.create_from_slice(f32::as_bytes(&q8_bufs.kv_scales));
        let sink_handle = if q8_bufs.sink_kv.is_empty() {
            client.empty(core::mem::size_of::<f32>())
        } else {
            client.create_from_slice(f32::as_bytes(&q8_bufs.sink_kv))
        };
        let out_handle = client.empty(q_len * core::mem::size_of::<f32>());

        crate::attention_q8kv_cubecl::AttentionQ8KVCubeCL::launch_with_sink::<ActiveRuntime>(
            client,
            query_handle,
            kv_qs_handle,
            kv_scales_handle,
            sink_handle,
            out_handle.clone(),
            sink_rows,
            &params,
        );

        let bytes = client
            .read_one(out_handle)
            .expect("should read incumbent q8kv sink output");
        f32::from_bytes(&bytes).to_vec()
    }

    /// Run the folded Q8_0 attention kernel and return the output.
    fn run_folded(
        client: &ComputeClient<ActiveRuntime>,
        query: &[f32],
        keys: &[f32],
        values: &[f32],
        n_positions: usize,
    ) -> Vec<f32> {
        let params = AttentionParams {
            n_head: N_HEAD,
            n_kv_head: N_KV_HEAD,
            head_dim: HEAD_DIM,
            n_positions,
            softcap: SOFTCAP,
            scale: SCALE,
        };

        let q8_bufs = Q8KVBuffers::quantize_kv(keys, values, N_KV_HEAD, HEAD_DIM);

        let q_len = N_HEAD * HEAD_DIM;
        let query_handle = client.create_from_slice(f32::as_bytes(query));
        let kv_qs_handle =
            client.create_from_slice(bytemuck::cast_slice::<u32, u8>(&q8_bufs.kv_qs));
        let kv_scales_handle = client.create_from_slice(f32::as_bytes(&q8_bufs.kv_scales));
        let out_handle = client.empty(q_len * core::mem::size_of::<f32>());

        AttentionQ8KVFolded::launch::<ActiveRuntime>(
            client,
            query_handle,
            kv_qs_handle,
            kv_scales_handle,
            out_handle.clone(),
            &params,
        );

        let bytes = client
            .read_one(out_handle)
            .expect("should read folded q8kv output");
        f32::from_bytes(&bytes).to_vec()
    }

    /// Run the folded kernel with the sink guard and return the output.
    #[cfg(feature = "q8kv_sink_guard")]
    fn run_folded_with_sink(
        client: &ComputeClient<ActiveRuntime>,
        query: &[f32],
        keys: &[f32],
        values: &[f32],
        n_positions: usize,
        sink_rows: usize,
    ) -> Vec<f32> {
        let params = AttentionParams {
            n_head: N_HEAD,
            n_kv_head: N_KV_HEAD,
            head_dim: HEAD_DIM,
            n_positions,
            softcap: SOFTCAP,
            scale: SCALE,
        };

        let q8_bufs =
            Q8KVBuffers::quantize_kv_with_sink(keys, values, N_KV_HEAD, HEAD_DIM, sink_rows);

        let q_len = N_HEAD * HEAD_DIM;
        let query_handle = client.create_from_slice(f32::as_bytes(query));
        let kv_qs_handle =
            client.create_from_slice(bytemuck::cast_slice::<u32, u8>(&q8_bufs.kv_qs));
        let kv_scales_handle = client.create_from_slice(f32::as_bytes(&q8_bufs.kv_scales));
        let sink_handle = if q8_bufs.sink_kv.is_empty() {
            client.empty(core::mem::size_of::<f32>())
        } else {
            client.create_from_slice(f32::as_bytes(&q8_bufs.sink_kv))
        };
        let out_handle = client.empty(q_len * core::mem::size_of::<f32>());

        AttentionQ8KVFolded::launch_with_sink::<ActiveRuntime>(
            client,
            query_handle,
            kv_qs_handle,
            kv_scales_handle,
            sink_handle,
            out_handle.clone(),
            sink_rows,
            &params,
        );

        let bytes = client
            .read_one(out_handle)
            .expect("should read folded q8kv sink output");
        f32::from_bytes(&bytes).to_vec()
    }

    /// f64 attention reference over the SAME Q8_0 bytes the kernels read:
    /// exact dequantization (`scale_f64 · i8_f64`), f64 dot/softcap/softmax,
    /// sink rows served from the f32 sidecar. This is the fixture oracle the
    /// falsification arm measures both kernels against — the incumbent's own
    /// error defines the folded kernel's bound (a measured class).
    fn attention_f64_reference(
        query: &[f32],
        bufs: &Q8KVBuffers,
        n_positions: usize,
        sink_rows: usize,
    ) -> Vec<f64> {
        let kv_stride_q8 = N_KV_HEAD * 64usize; // u32s per (pos, kv_head)
        let kv_scale_stride = N_KV_HEAD * 8usize;
        let kv_qs_half = bufs.kv_qs.len() / 2;
        let kv_scales_half = bufs.kv_scales.len() / 2;
        let sink_half = bufs.sink_kv.len() / 2;
        let kv_f32_stride = N_KV_HEAD * HEAD_DIM;
        let have_sink = sink_rows > 0 && !bufs.sink_kv.is_empty();

        let mut out = vec![0.0f64; N_HEAD * HEAD_DIM];
        for head in 0..N_HEAD {
            let kv_group = head * N_KV_HEAD / N_HEAD;

            // Scores: f64 dot with exact dequant, then scale + softcap.
            let mut scores = vec![0.0f64; n_positions];
            for (pos, score) in scores.iter_mut().enumerate() {
                let mut dot = 0.0f64;
                if have_sink && pos < sink_rows {
                    let base = pos * kv_f32_stride + kv_group * HEAD_DIM;
                    for dim in 0..HEAD_DIM {
                        dot +=
                            query[head * HEAD_DIM + dim] as f64 * bufs.sink_kv[base + dim] as f64;
                    }
                } else {
                    for b in 0..8usize {
                        let scale = bufs.kv_scales[pos * kv_scale_stride + kv_group * 8 + b] as f64;
                        for e in 0..32usize {
                            let word_idx = pos * kv_stride_q8 + kv_group * 64 + b * 8 + e / 4;
                            let word = bufs.kv_qs[word_idx];
                            let byte = (word >> (((e % 4) as u32) * 8)) & 0xFF;
                            let i8v = if byte >= 128 {
                                byte as f64 - 256.0
                            } else {
                                byte as f64
                            };
                            let dim = b * 32 + e;
                            dot += query[head * HEAD_DIM + dim] as f64 * (scale * i8v);
                        }
                    }
                }
                let raw = dot * SCALE as f64;
                *score = SOFTCAP as f64 * (raw / SOFTCAP as f64).tanh();
            }

            // Plain softmax (mathematically identical to the online variant).
            let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut exps: Vec<f64> = scores.iter().map(|s| (s - max_s).exp()).collect();
            let sum: f64 = exps.iter().sum();
            for e in exps.iter_mut() {
                *e /= sum;
            }

            // Weighted values with exact dequant; sink rows from the sidecar.
            for dim in 0..HEAD_DIM {
                let mut acc = 0.0f64;
                for (pos, exp_p) in exps.iter().enumerate() {
                    let v = if have_sink && pos < sink_rows {
                        bufs.sink_kv[sink_half + pos * kv_f32_stride + kv_group * HEAD_DIM + dim]
                            as f64
                    } else {
                        let b = dim / 32;
                        let e = dim % 32;
                        let scale = bufs.kv_scales
                            [kv_scales_half + pos * kv_scale_stride + kv_group * 8 + b]
                            as f64;
                        let word = bufs.kv_qs
                            [kv_qs_half + pos * kv_stride_q8 + kv_group * 64 + b * 8 + e / 4];
                        let byte = (word >> (((e % 4) as u32) * 8)) & 0xFF;
                        let i8v = if byte >= 128 {
                            byte as f64 - 256.0
                        } else {
                            byte as f64
                        };
                        scale * i8v
                    };
                    acc += exp_p * v;
                }
                out[head * HEAD_DIM + dim] = acc;
            }
        }
        out
    }

    /// Max absolute deviation between a GPU output (f32) and the f64 oracle.
    fn max_abs_dev(gpu: &[f32], oracle: &[f64]) -> f64 {
        gpu.iter()
            .zip(oracle.iter())
            .map(|(&g, &o)| (g as f64 - o).abs())
            .fold(0.0f64, f64::max)
    }

    /// Issue 1211 falsification arm, no-sink posture: on the fixture, the
    /// folded kernel's f64 deviation must stay within 2× the incumbent's own
    /// (measured first, printed always). Shapes cover single position, one
    /// full tile, a partial tail tile (300 = 256 + 44), and two full tiles.
    #[test]
    fn test_q8kv_folded_parity_vs_f64() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        for &n_positions in &[1usize, 32, 300, 512] {
            let (query, keys, values) = make_test_data(n_positions);
            let bufs = Q8KVBuffers::quantize_kv(&keys, &values, N_KV_HEAD, HEAD_DIM);
            let oracle = attention_f64_reference(&query, &bufs, n_positions, 0);

            let incumbent = run_incumbent(&client, &query, &keys, &values, n_positions);
            let folded = run_folded(&client, &query, &keys, &values, n_positions);

            let err_inc = max_abs_dev(&incumbent, &oracle);
            let err_fold = max_abs_dev(&folded, &oracle);
            println!(
                "folded parity n_pos={n_positions}: incumbent vs f64 {err_inc:.3e}, \
                 folded vs f64 {err_fold:.3e}, bound {:.3e}",
                2.0 * err_inc
            );
            assert!(
                err_fold <= 2.0 * err_inc,
                "folded parity bound FAILED at n_pos={n_positions}: \
                 folded {err_fold:.3e} > 2 × incumbent {err_inc:.3e}"
            );
        }
    }

    /// Issue 1211 falsification arm, sink-guard posture: the fold-exempt
    /// sidecar path must hold the same measured bound. 300 positions spans a
    /// partial tail tile; `sink_rows = 4` puts the sink span inside the first
    /// tile for both phases.
    #[cfg(feature = "q8kv_sink_guard")]
    #[test]
    fn test_q8kv_folded_sink_parity_vs_f64() {
        let ctx = CubeCLContext::new().expect("CubeCL should initialize");
        let client = ctx.client();

        let n_positions = 300;
        let sink_rows = 4;
        let (query, keys, values) = make_test_data(n_positions);
        let bufs =
            Q8KVBuffers::quantize_kv_with_sink(&keys, &values, N_KV_HEAD, HEAD_DIM, sink_rows);
        let oracle = attention_f64_reference(&query, &bufs, n_positions, sink_rows);

        let incumbent =
            run_incumbent_with_sink(&client, &query, &keys, &values, n_positions, sink_rows);
        let folded = run_folded_with_sink(&client, &query, &keys, &values, n_positions, sink_rows);

        let err_inc = max_abs_dev(&incumbent, &oracle);
        let err_fold = max_abs_dev(&folded, &oracle);
        println!(
            "folded sink parity n_pos={n_positions} sink_rows={sink_rows}: \
             incumbent vs f64 {err_inc:.3e}, folded vs f64 {err_fold:.3e}, bound {:.3e}",
            2.0 * err_inc
        );
        assert!(
            err_fold <= 2.0 * err_inc,
            "folded sink parity bound FAILED: folded {err_fold:.3e} > 2 × incumbent {err_inc:.3e}"
        );
    }
}
