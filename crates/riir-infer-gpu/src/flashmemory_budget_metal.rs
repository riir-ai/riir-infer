//! Metal FlashMemory budget decode arm — the three kernels (Issue 1191 T1,
//! Issue 1117 T4 vehicle).
//!
//! # What this is
//!
//! The FlashMemory budget decode lane (issue 1117 T2's CPU reference
//! `forward_qwen_deltanet_ternary_flashmemory`) runs CPU-only today; the
//! compute law (2026-11-09) says the measured cells run on METAL. This
//! module is the GPU half: the three missing kernels behind the opt-in
//! feature `flashmemory_budget_metal` (PROMOTE-ONLY env per the env-switch
//! law until the T4 GOAT — exact `1`/`true`/`on`, never demote-only):
//!
//! 1. **(a) block-centroid scoring** — [`FmBlockScoreMetal`]: per
//!    (kv_head, block) GEMV `dot(query[kv], centroid[block][kv]) × scale`
//!    over the per-block mean keys. One thread per score — the
//!    `GqaFlashMemorySelector::select` scoring loop's device twin.
//! 2. **(b) top-ρ select** — [`FmTopRhoSelectMetal`]: one cube per kv
//!    head; iterative shared-memory argmax over untaken blocks (strictly-
//!    greater compare → FIRST index wins ties — the determinism law,
//!    matching the CPU `select_nth_unstable_by` tie rule exactly), then an
//!    in-shared insertion sort to ascending, then the current block. The
//!    CPU `RowRule::Budget` shape: top `k = max(round(ρ·n), 1) − 1`
//!    complete blocks + the current block, ascending output.
//! 3. **(c) block gather** — [`FmBlockGatherMetal`]: the gated decode
//!    attention restricted to the selected blocks — per query head, tiles
//!    over its kv group's selected blocks' token ranges, online max/sum
//!    softmax, `σ(gate)` epilogue. The `sparse_attention_accumulate`
//!    semantics (selected-token softmax, weighted V) in the
//!    `QwenAttentionDecodeGatedCubeCL` tile shape. TOLERANCE-class vs the
//!    CPU two-pass form (online vs two-pass max — T2's parity gate is the
//!    correctness authority, bit-exactness is not claimed across
//!    backends).
//!
//! # Substrate-first (what this consumes, never duplicates)
//!
//! Kernel (a) is the dense f32 GEMV shape (`GemvCubeCL` family) at kv-head
//! granularity; kernel (c) is `QwenAttentionDecodeGatedCubeCL`'s tile loop
//! restricted to a block list; the prefill block-sparse arm
//! (`qwen_attention_prefill_sparse_cubecl`: pool → score → select → mask)
//! is the structural cousin and stays prefill-only. The GDN scan, the GLU
//! FFN, the ktile64 attention arm, and the dense decode numerics are NOT
//! touched (issue 1191's non-goal).
//!
//! # Selection transport
//!
//! Kernel (b) writes `[n_kv * MAX_SEL]` u32 indices (ascending, `u32::MAX`
//! padding) + `[n_kv]` counts; kernel (c) reads its kv group's slice. The
//! host keeps the buffers device-resident across the step — no readback on
//! the serving path.
//!
//! # Raw/latent posture
//!
//! Decode numerics are raw physical state (sync boundary downstream
//! unchanged). Nothing here gates raw-sync correctness; the arm is opt-in
//! and inert by default.

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;
#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

/// Selection-list capacity per kv head (the budget rule keeps
/// `round(ρ·n) − 1` complete blocks + the current block). The row is the
/// hard ceiling: at ρ=0.15 it admits n_blocks ≤ 857 (block-64 ≈ 54.8K
/// context), at ρ=0.3 n_blocks ≤ 428 — a config needing more REFUSES
/// LOUD (issue 1200: the overflow wrote past the kv row — device clobber
/// + a rig-hour-sized GPU wedge — never a silent truncation).
pub const FM_MAX_SEL: usize = 128;

/// The selection kernel's shared-memory slice ceiling in blocks (2048
/// score slots + 512 pair slots). Issue 1200 T2: the late per-refresh
/// assert fired 99.94% into a multi-hour run — the same refusal now runs
/// at construction (t=0) and at every launch, naming the ceiling. The
/// slice RAISE is the 1191 lane's capacity call, never snuck in here.
pub const FM_MAX_BLOCKS_SMEM: usize = 2048;

/// The selection count the budget rule computes for `(n_blocks, rho)` —
/// the kernel's arithmetic mirrored host-side, operation for operation in
/// f32 (same `round`, same clamps), so the envelope refusal can fire
/// before any device work and can never disagree with what the kernel
/// would do. `top-k + 1`: k complete blocks plus the current block.
pub fn fm_budget_select_count(n_blocks: usize, rho: f32) -> u32 {
    let k_f = (rho * n_blocks as f32).round();
    let k_int = if k_f < 1.0f32 { 1u32 } else { k_f as u32 };
    let k = (k_int.saturating_sub(1u32)).min((n_blocks as u32).saturating_sub(1u32));
    k + 1
}

/// The largest `n_blocks` whose budget count still fits [`FM_MAX_SEL`] at
/// `rho` — analytic threshold verified with the SAME arithmetic the
/// kernel runs (the walk is ≤ 2 steps: the candidate lands within one
/// count step of the boundary).
pub fn fm_budget_max_n_blocks(rho: f32) -> usize {
    let mut n = ((FM_MAX_SEL as f32 + 0.5) / rho) as usize;
    while n > 1 && fm_budget_select_count(n, rho) as usize > FM_MAX_SEL {
        n -= 1;
    }
    n
}

/// Issue 1200 T1 (fail-closed half) — the slot-capacity refusal. The
/// selection row is `[FM_MAX_SEL]` u32 per kv head; a budget rule needing
/// more picks writes past its own row (clobbering the NEXT kv group; the
/// last group past the buffer) and the host fraction reader then walks
/// off the buffer. The CPU `RowRule::Budget` clamp (`.min(n-1)`) is the
/// CPU envelope; the GPU port adds the SLOT-capacity refusal on top —
/// a config needing more than the row holds refuses naming the ceiling
/// and the largest n_blocks the rho admits, never silently truncates.
#[track_caller]
pub fn fm_budget_envelope_check(site: &str, n_blocks: usize, rho: f32) {
    let count = fm_budget_select_count(n_blocks, rho);
    if count as usize > FM_MAX_SEL {
        panic!(
            "[issue 1200] {site}: the budget rule needs {count} selection slots \
             (rho {rho} x {n_blocks} blocks) but FM_MAX_SEL = {FM_MAX_SEL} — refusing loud. \
             Largest supported n_blocks at this rho = {} (or lower rho; raising \
             FM_MAX_SEL is the 1191 lane's pre-registered capacity call).",
            fm_budget_max_n_blocks(rho),
        );
    }
}

/// Issue 1200 T2 (refuse half) — the shared-memory ceiling, moved to
/// construction + launch. The kernel's smem slice covers 2048 score
/// slots + 512 pair slots; a longer context must pair a larger slice
/// with the capacity raise (both the 1191 lane's call) — until then the
/// config refuses at t=0 instead of the last block-open refresh 99.94%
/// into a multi-hour run.
#[track_caller]
pub fn fm_budget_blocks_check(site: &str, n_blocks: usize) {
    assert!(
        n_blocks <= FM_MAX_BLOCKS_SMEM,
        "[issue 1200] {site}: {n_blocks} blocks > the {FM_MAX_BLOCKS_SMEM} \
         shared-memory ceiling — refusing (raise the smem slice together with \
         FM_MAX_SEL; both raises are the 1191 lane's capacity call)"
    );
}

/// The promote-only env gate — exact `1`/`true`/`on` arms the Metal
/// budget arm (the env-switch law: an opt-in arm's env is PROMOTE-ONLY;
/// unset or any other value = off). Read once per process, cached.
/// [`set_flashmemory_budget_metal`] is authoritative over it afterwards
/// (the A/B-harness pattern — `set_attn_split_decode`'s precedent).
#[cfg(feature = "cubecl_runtime")]
static FM_BUDGET_METAL_OVERRIDE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0); // 0 = unset, 1 = off, 2 = on
#[cfg(feature = "cubecl_runtime")]
static FM_BUDGET_METAL_LATCH: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0); // 0 = unset, 1 = off, 2 = on

#[cfg(feature = "cubecl_runtime")]
pub fn flashmemory_budget_metal_enabled() -> bool {
    use std::sync::atomic::Ordering;
    // The AUTHORITATIVE override (the setter's seat): 0 = unset, 1 = off,
    // 2 = on — consulted BEFORE the env latch in every read.
    match FM_BUDGET_METAL_OVERRIDE.load(Ordering::Relaxed) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    match FM_BUDGET_METAL_LATCH.load(Ordering::Relaxed) {
        0 => {}
        1 => return false,
        _ => return true,
    }
    let on = matches!(
        std::env::var("RIIR_FM_BUDGET_METAL").as_deref(),
        Ok("1") | Ok("true") | Ok("on")
    );
    FM_BUDGET_METAL_LATCH.store(if on { 2 } else { 1 }, Ordering::Relaxed);
    on
}

/// Authoritative arm posture — beats the env latch (the A/B-harness
/// pattern; the env-switch law keeps the ENV promote-only, this setter is
/// the harness seat). Also re-arms the rho/block/refresh reads: the
/// forward's constructor reads those envs fresh per construction, so a
/// setter flip re-arms the NEXT construction, never the running one.
#[cfg(feature = "cubecl_runtime")]
pub fn set_flashmemory_budget_metal(on: bool) {
    use std::sync::atomic::Ordering;
    FM_BUDGET_METAL_OVERRIDE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
}

// ---------------------------------------------------------------------------
// Kernel (a): block-centroid scoring — one thread per (kv_head, block)
// ---------------------------------------------------------------------------

/// Scores `dot(query[kv], centroid[block][kv]) × scale` for every
/// (kv_head, block).
///
/// - `query`: `[n_kv * head_dim]` — the KV-resolution query (the host's
///   `build_kv_resolution_query` output, already averaged per GQA group).
/// - `centroids`: `[n_blocks * n_kv * head_dim]` — per-block mean keys,
///   the CPU `GqaFlashMemoryBlockCache::key_centroids` layout verbatim
///   (block-major, kv-head-major within a block).
/// - `scores_out`: `[n_kv * max_blocks]` — kernel (b)'s input.
#[cfg(feature = "cubecl_runtime")]
#[allow(clippy::needless_range_loop, reason = "CubeCL macro expansion")]
#[cube(launch_unchecked)]
fn fm_block_score_f32(query: &[f32], centroids: &[f32], scores_out: &mut [f32], params: &[f32]) {
    let n_kv = params[0usize] as u32;
    let head_dim = params[1usize] as u32;
    let n_blocks = params[2usize] as u32;
    let scale = params[3usize];
    let i = ABSOLUTE_POS as u32;

    let kv = i / n_blocks;
    let block = i % n_blocks;
    if kv < n_blocks && block < n_blocks {
        let q_off = (kv * head_dim) as usize;
        let c_off = (block * n_kv * head_dim + kv * head_dim) as usize;
        let mut dot = f32::new(0.0f32);
        let mut d = 0u32;
        while d < head_dim {
            dot += query[q_off + d as usize] * centroids[c_off + d as usize];
            d += 1u32;
        }
        scores_out[(kv * n_blocks + block) as usize] = dot * scale;
    }
}

#[cfg(feature = "cubecl_runtime")]
pub struct FmBlockScoreMetal;

#[cfg(feature = "cubecl_runtime")]
impl FmBlockScoreMetal {
    /// Launch the centroid-scoring GEMV. One thread per (kv, block).
    ///
    /// # Safety
    /// - `query`: `[n_kv * head_dim]` f32.
    /// - `centroids`: `[n_blocks * n_kv * head_dim]` f32 (block-major).
    /// - `scores_out`: `[n_kv * n_blocks]` f32.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        query: Handle,
        centroids: Handle,
        scores_out: Handle,
        n_kv: usize,
        head_dim: usize,
        n_blocks: usize,
        scale: f32,
    ) {
        let params: [f32; 4] = [n_kv as f32, head_dim as f32, n_blocks as f32, scale];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
        unsafe {
            fm_block_score_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static((n_kv * n_blocks) as u32, 1, 1),
                CubeDim::new_1d(64),
                BufferArg::from_raw_parts(query, n_kv * head_dim),
                BufferArg::from_raw_parts(centroids, n_blocks * n_kv * head_dim),
                BufferArg::from_raw_parts(scores_out, n_kv * n_blocks),
                BufferArg::from_raw_parts(params_handle, 4),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel (b): top-ρ select — one cube per kv head, iterative argmax,
// first-index strict ties, ascending output + the current block
// ---------------------------------------------------------------------------

/// The budget selection, matching the CPU `RowRule::Budget` exactly:
/// top `k = max(round(ρ·n), 1) − 1` complete blocks by score (ties →
/// lower index) plus the current block (index `n−1`), output ascending.
///
/// One cube per kv head; every thread scans the row cooperatively per
/// selection round (`taken` mask in shared memory); thread 0 insertion-
/// sorts the k picks ascending (k ≤ 128 — trivial), appends `n−1`, writes
/// the count.
#[cfg(feature = "cubecl_runtime")]
#[allow(clippy::needless_range_loop, reason = "CubeCL macro expansion")]
#[cube(launch_unchecked)]
fn fm_top_rho_select_f32(
    scores: &[f32],
    sel_out: &mut [u32],
    sel_counts: &mut [u32],
    params: &[f32],
) {
    let kv = CUBE_POS as u32;
    let n_blocks = params[1usize] as u32;
    let rho = params[2usize];
    let tid = UNIT_POS;
    let n_threads = CUBE_DIM_X;

    // The row's scores + the argmax-reduce (score, idx) pairs, in ONE
    // shared slice: n_blocks slots for scores + 2*n_threads for the pairs
    // (the launcher's slice covers n_blocks ≤ 2048: 2048 + 512 = 2560).
    let mut smem = Shared::<[f32]>::new_slice(2560usize);
    let mut taken = Shared::<[u32]>::new_slice(512usize); // u32 bit-words: 2048 blocks

    let row_off = kv * n_blocks;
    let stride = tid;
    let mut b = stride;
    while b < n_blocks {
        smem[b as usize] = scores[(row_off + b) as usize];
        b += n_threads;
    }
    if tid == 0u32 {
        let words = n_blocks.div_ceil(32u32);
        let mut w = 0u32;
        while w < words {
            taken[w as usize] = 0u32;
            w += 1u32;
        }
    }
    sync_cube();

    // k = max(round(ρ·n), 1) − 1, clamped to n−1 — the CPU rule verbatim.
    let k_f = (rho * n_blocks as f32).round();
    let k_int = if k_f < 1.0f32 { 1u32 } else { k_f as u32 };
    let k = (k_int.saturating_sub(1u32)).min(n_blocks.saturating_sub(1u32));
    // Issue 1200 — defense-in-depth: the launcher refuses an
    // out-of-envelope config BEFORE this kernel runs, so the clamp is
    // unreachable through `FmTopRhoSelectMetal::launch`; a future direct
    // caller gets a truncated (honestly counted via sel_counts ≤ 128)
    // selection instead of writes past the kv row (device clobber).
    let k = k.min(127u32);

    // Iterative argmax over untaken blocks — strictly-greater compare so
    // ties keep the FIRST index (the determinism law; the CPU
    // `select_nth_unstable_by` orders equal scores by ascending index and
    // the host sort is stable to the same effect).
    let mut round = 0u32;
    while round < k {
        // Cooperative argmax.
        let mut my_best = f32::new(-f32::MAX);
        let mut my_idx = n_blocks;
        let mut b = stride;
        while b < n_blocks {
            // The CURRENT block is never a pick candidate — the CPU rule's
            // pool is `0..current` (it is appended once, unconditionally,
            // after the picks; a pick-and-reappend would double-count its
            // tokens in the gather's softmax).
            let word = b / 32u32;
            let bit = b % 32u32;
            let is_taken = (taken[word as usize] >> bit) & 1u32;
            let is_current = b == n_blocks - 1u32;
            if is_taken == 0u32 && !is_current {
                let s = smem[b as usize];
                if s > my_best {
                    my_best = s;
                    my_idx = b;
                }
            }
            b += n_threads;
        }
        // Cube-wide reduce (idx ties resolved toward the smaller index):
        // tree over shared memory pairs (score, idx).
        // Reuse `taken`'s tail? No — pack (score, idx) into the smem via a
        // second slice is not available; do a bounded warp-local reduce
        // through the `smem` scratch's tail region beyond n_blocks.
        //
        // Simplest deterministic form: serialize the reduce through
        // shared pairs stored at [n_blocks + 2*tid].
        let pair_base = n_blocks + 2u32 * tid;
        smem[pair_base as usize] = my_best;
        smem[(pair_base + 1u32) as usize] = my_idx as f32;
        sync_cube();

        if tid == 0u32 {
            let mut best_s = f32::new(-f32::MAX);
            let mut best_i = n_blocks;
            let mut t = 0u32;
            while t < n_threads {
                let s = smem[(n_blocks + 2u32 * t) as usize];
                let i = smem[(n_blocks + 2u32 * t + 1u32) as usize] as u32;
                if i < n_blocks && (s > best_s || (s == best_s && i < best_i)) {
                    best_s = s;
                    best_i = i;
                }
                t += 1u32;
            }
            // Mark taken + record the pick (unordered; sorted below).
            let word = best_i / 32u32;
            let bit = best_i % 32u32;
            taken[word as usize] |= 1u32 << bit;
            // Stash picks at the sel_out row's head (k slots).
            sel_out[(kv * 128u32 + round) as usize] = best_i;
        }
        sync_cube();
        round += 1u32;
    }

    if tid == 0u32 {
        // Insertion sort the k picks ascending (k ≤ 128 — trivial), then
        // the current block.
        let base = kv * 128u32;
        let mut i = 1u32;
        while i < k {
            let v = sel_out[(base + i) as usize];
            let mut j = i;
            while j > 0u32 {
                let prev = sel_out[(base + j - 1u32) as usize];
                if prev > v {
                    sel_out[(base + j) as usize] = prev;
                    j -= 1u32;
                } else {
                    break;
                }
            }
            sel_out[(base + j) as usize] = v;
            i += 1u32;
        }
        // Append the current block (n−1) — the CPU rule keeps it always.
        let count = k + 1u32;
        sel_out[(base + k) as usize] = n_blocks - 1u32;
        sel_counts[kv as usize] = count;
        // Pad the tail with u32::MAX (kernel (c) reads exactly `count`).
        let mut p = count;
        while p < 128u32 {
            sel_out[(base + p) as usize] = u32::MAX;
            p += 1u32;
        }
    }
}

#[cfg(feature = "cubecl_runtime")]
pub struct FmTopRhoSelectMetal;

#[cfg(feature = "cubecl_runtime")]
impl FmTopRhoSelectMetal {
    /// Launch the budget selection. One cube per kv head, 256 threads.
    ///
    /// # Safety
    /// - `scores`: `[n_kv * n_blocks]` f32 (kernel (a)'s output).
    /// - `sel_out`: `[n_kv * 128]` u32 (indices ascending, `u32::MAX` pad).
    /// - `sel_counts`: `[n_kv]` u32.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        scores: Handle,
        sel_out: Handle,
        sel_counts: Handle,
        n_kv: usize,
        n_blocks: usize,
        rho: f32,
    ) {
        assert!(n_blocks >= 1, "fm_top_rho_select: empty block row");
        // Issue 1200 T1+T2 — the fail-closed envelope at the authoritative
        // launch site (the constructor refuses the max-context config at
        // t=0; this is the per-launch gate at the exact arithmetic).
        fm_budget_blocks_check("fm_top_rho_select", n_blocks);
        fm_budget_envelope_check("fm_top_rho_select", n_blocks, rho);
        let params: [f32; 3] = [n_kv as f32, n_blocks as f32, rho];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
        unsafe {
            fm_top_rho_select_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_kv as u32, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(scores, n_kv * n_blocks),
                BufferArg::from_raw_parts(sel_out, n_kv * FM_MAX_SEL),
                BufferArg::from_raw_parts(sel_counts, n_kv),
                BufferArg::from_raw_parts(params_handle, 3),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel (c): block gather — the gated decode attention over selected
// blocks
// ---------------------------------------------------------------------------

/// The gated decode attention restricted to the selected blocks — one
/// cube per query head (head_dim threads), the
/// `QwenAttentionDecodeGatedCubeCL` tile shape walked over the SELECTION
/// list instead of the dense token range.
///
/// Semantics per query head: online max/sum softmax over exactly the
/// selected blocks' tokens (each block's `[start, end)` from the counts
/// and the fixed block size), scores as `dot(q, k) × scale`, the
/// weighted V accumulation in the same tile order, `σ(gate)` in the final
/// write. The empty-selection fallback matches the CPU
/// `sparse_attention_accumulate` (the current block).
#[cfg(feature = "cubecl_runtime")]
#[allow(clippy::needless_range_loop, reason = "CubeCL macro expansion")]
#[cube(launch_unchecked)]
fn fm_block_gather_f32(
    query: &[f32],
    key_cache: &[f32],
    value_cache: &[f32],
    gate: &[f32],
    sel: &[u32],
    sel_counts: &[u32],
    attn_out: &mut [f32],
    params: &[f32],
) {
    let head_dim = params[0usize] as u32;
    let n_head = params[1usize] as u32;
    let n_kv_head = params[2usize] as u32;
    let seq_len = params[3usize] as u32;
    let scale = params[4usize];
    let block_size = params[5usize] as u32;

    let cube_size = head_dim;
    let head_idx = CUBE_POS as u32;
    let tid = UNIT_POS;

    if head_idx >= n_head {
        terminate!();
    }

    let head_off = head_idx * head_dim;
    let kv_stride = n_kv_head * head_dim;
    let kv_group = head_idx * n_kv_head / n_head;
    let kv_off = kv_group * head_dim;
    let valid_dim = tid < head_dim;

    let n_sel = sel_counts[kv_group as usize];
    let sel_base = kv_group * 128u32;

    let mut running_max = f32::new(-1e30f32);
    let mut running_sum = f32::new(0.0f32);
    let mut running_out = f32::new(0.0f32);

    let mut smem = Shared::<[f32]>::new_slice(256usize);

    // Empty selection fallback: attend the current block (the CPU
    // sparse_attention_accumulate's single-block fallback).
    let mut s = 0u32;
    while s < n_sel {
        let block_idx = sel[(sel_base + s) as usize];
        if block_idx != u32::MAX {
            let tok_start = block_idx * block_size;
            let tok_end_min = tok_start + block_size;
            let tok_end = if tok_end_min < seq_len {
                tok_end_min
            } else {
                seq_len
            };

            let n_tiles = (tok_end - tok_start).div_ceil(cube_size);
            let mut tile = 0u32;
            while tile < n_tiles {
                let tile_base = tok_start + tile * cube_size;
                let pos = tile_base + tid;
                let valid_pos = pos < tok_end;

                let mut my_score = f32::new(-1e30f32);
                if valid_pos {
                    let k_base = pos * kv_stride + kv_off;
                    let mut dot = f32::new(0.0f32);
                    let mut d = 0u32;
                    while d < head_dim {
                        dot += query[(head_off + d) as usize] * key_cache[(k_base + d) as usize];
                        d += 1u32;
                    }
                    my_score = dot * scale;
                }
                smem[tid as usize] = my_score;
                sync_cube();

                // Tile max — the decode kernel's tree (Issue 715's full-tree form).
                let mut stride = 128u32;
                while stride > 0u32 {
                    if tid < stride && smem[(tid + stride) as usize] > smem[tid as usize] {
                        smem[tid as usize] = smem[(tid + stride) as usize];
                    }
                    sync_cube();
                    stride /= 2u32;
                }
                let tile_max = smem[0usize];
                sync_cube();

                let new_max = if tile_max > running_max {
                    tile_max
                } else {
                    running_max
                };
                let exp_prev = running_max - new_max;
                let exp_tile = tile_max - new_max;

                running_sum = running_sum * exp_prev.exp();
                running_out = running_out * exp_prev.exp();
                running_max = new_max;

                if valid_pos {
                    smem[tid as usize] = exp_tile.exp() * (my_score - tile_max).exp();
                } else {
                    smem[tid as usize] = f32::new(0.0f32);
                }
                sync_cube();

                let mut tile_sum = f32::new(0.0f32);
                let mut acc = f32::new(0.0f32);
                let mut p = 0u32;
                while p < cube_size {
                    let kv_pos = tile_base + p;
                    if kv_pos < tok_end {
                        let weight = smem[p as usize];
                        tile_sum = tile_sum + weight;
                        if valid_dim {
                            let v_idx = kv_pos * kv_stride + kv_off + tid;
                            acc += weight * value_cache[v_idx as usize];
                        }
                    }
                    p += 1u32;
                }
                if valid_dim {
                    running_out += acc;
                }
                running_sum = running_sum + tile_sum;

                // The back-edge barrier (the decode kernel's Issue-715 race cure).
                sync_cube();
                tile += 1u32;
            }
        }
        s += 1u32;
    }

    // Final normalize + fused gate (the decode kernel's epilogue).
    if valid_dim {
        let inv_sum = f32::new(1.0f32) / running_sum;
        let raw = running_out * inv_sum;
        let g = gate[(head_off + tid) as usize];
        let neg_g = f32::new(0.0f32) - g;
        let sig = f32::new(1.0f32) / (f32::new(1.0f32) + neg_g.exp());
        attn_out[(head_off + tid) as usize] = raw * sig;
    }
}

#[cfg(feature = "cubecl_runtime")]
pub struct FmBlockGatherMetal;

#[cfg(feature = "cubecl_runtime")]
impl FmBlockGatherMetal {
    /// Launch the block-gather attention. One cube per query head.
    ///
    /// # Safety
    /// - `query`: `[n_head * head_dim]` f32 (post-norm post-RoPE).
    /// - `key_cache`/`value_cache`: `[max_seq * n_kv * head_dim]` f32 (the
    ///   existing layout the Metal stack serves).
    /// - `gate`: `[n_head * head_dim]` f32.
    /// - `sel`: `[n_kv * 128]` u32; `sel_counts`: `[n_kv]` u32.
    /// - `attn_out`: `[n_head * head_dim]` f32.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        query: Handle,
        key_cache: Handle,
        value_cache: Handle,
        gate: Handle,
        sel: Handle,
        sel_counts: Handle,
        attn_out: Handle,
        n_head: usize,
        n_kv: usize,
        head_dim: usize,
        seq_len: usize,
        block_size: usize,
    ) {
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let params: [f32; 6] = [
            head_dim as f32,
            n_head as f32,
            n_kv as f32,
            seq_len as f32,
            scale,
            block_size as f32,
        ];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
        let q_len = n_head * head_dim;
        let kv_len = seq_len * n_kv * head_dim;
        unsafe {
            fm_block_gather_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static(n_head as u32, 1, 1),
                CubeDim::new_1d(head_dim as u32),
                BufferArg::from_raw_parts(query, q_len),
                BufferArg::from_raw_parts(key_cache, kv_len),
                BufferArg::from_raw_parts(value_cache, kv_len),
                BufferArg::from_raw_parts(gate, q_len),
                BufferArg::from_raw_parts(sel, n_kv * FM_MAX_SEL),
                BufferArg::from_raw_parts(sel_counts, n_kv),
                BufferArg::from_raw_parts(attn_out, q_len),
                BufferArg::from_raw_parts(params_handle, 6),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Integration kernels (Issue 1191 T2): the query averaging + the centroid
// rebuild — the two dense-side support pieces the full decode forward needs
// before the three T1 kernels can run device-resident.
// ---------------------------------------------------------------------------

/// The KV-resolution query — `build_kv_resolution_query`'s device twin:
/// average the query heads within each GQA group (the BLOCK-CONTIGUOUS
/// mapping, Bench 457's regression law — never the interleaved set).
///
/// One thread per (kv_head, dim).
#[cfg(feature = "cubecl_runtime")]
#[allow(clippy::needless_range_loop, reason = "CubeCL macro expansion")]
#[cube(launch_unchecked)]
fn fm_query_kv_avg_f32(q_heads: &[f32], kv_query: &mut [f32], params: &[f32]) {
    let n_head = params[0usize] as u32;
    let n_kv = params[1usize] as u32;
    let head_dim = params[2usize] as u32;
    let i = ABSOLUTE_POS as u32;
    let kv = i / head_dim;
    let d = i % head_dim;
    if kv >= n_kv {
        terminate!();
    }
    let group = n_head / n_kv;
    let mut acc = f32::new(0.0f32);
    let mut g = 0u32;
    while g < group {
        acc += q_heads[((kv * group + g) * head_dim + d) as usize];
        g += 1u32;
    }
    kv_query[i as usize] = acc / group as f32;
}

#[cfg(feature = "cubecl_runtime")]
pub struct FmQueryKvAvgMetal;

#[cfg(feature = "cubecl_runtime")]
impl FmQueryKvAvgMetal {
    /// Launch the KV-resolution query average.
    ///
    /// # Safety
    /// - `q_heads`: `[n_head * head_dim]` f32 (post-norm post-RoPE).
    /// - `kv_query`: `[n_kv * head_dim]` f32.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        q_heads: Handle,
        kv_query: Handle,
        n_head: usize,
        n_kv: usize,
        head_dim: usize,
    ) {
        let params: [f32; 3] = [n_head as f32, n_kv as f32, head_dim as f32];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
        unsafe {
            fm_query_kv_avg_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static((n_kv * head_dim) as u32, 1, 1),
                CubeDim::new_1d(64),
                BufferArg::from_raw_parts(q_heads, n_head * head_dim),
                BufferArg::from_raw_parts(kv_query, n_kv * head_dim),
                BufferArg::from_raw_parts(params_handle, 3),
            );
        }
    }
}

/// The block-centroid REBUILD — `GqaFlashMemoryBlockCache::rebuild_from_keys`'
/// device twin: per (block, kv_head) the mean of the block's keys. One cube
/// per (block, kv_head), head_dim threads; each thread sums its dim lane
/// across the block's tokens IN TICK ORDER (the CPU's ascending
/// accumulation — the reduction order the determinism law pins) and
/// scales by the count.
///
/// Blocks past the last token contribute nothing (their lanes stay
/// zero-initialized); `n_blocks` names the ACTIVE count (ceil(seq/blocks)).
#[cfg(feature = "cubecl_runtime")]
#[allow(clippy::needless_range_loop, reason = "CubeCL macro expansion")]
#[cube(launch_unchecked)]
fn fm_centroid_rebuild_f32(key_cache: &[f32], centroids_out: &mut [f32], params: &[f32]) {
    let n_kv = params[0usize] as u32;
    let head_dim = params[1usize] as u32;
    let block_size = params[2usize] as u32;
    let seq_len = params[3usize] as u32;
    let kvd = n_kv * head_dim;

    let cube_idx = CUBE_POS as u32;
    let block = cube_idx / n_kv;
    let kv = cube_idx % n_kv;
    let tid = UNIT_POS;
    if block * block_size >= seq_len {
        terminate!();
    }

    let tok_start = block * block_size;
    let tok_end = if tok_start + block_size < seq_len {
        tok_start + block_size
    } else {
        seq_len
    };
    let count = tok_end - tok_start;

    let mut acc = f32::new(0.0f32);
    let mut tok = tok_start;
    while tok < tok_end {
        acc += key_cache[(tok * kvd + kv * head_dim + tid) as usize];
        tok += 1u32;
    }
    centroids_out[(block * kvd + kv * head_dim + tid) as usize] = acc / count as f32;
}

#[cfg(feature = "cubecl_runtime")]
pub struct FmCentroidRebuildMetal;

#[cfg(feature = "cubecl_runtime")]
impl FmCentroidRebuildMetal {
    /// Launch the centroid rebuild over the ACTIVE blocks.
    ///
    /// # Safety
    /// - `key_cache`: `[seq_len * n_kv * head_dim]` f32.
    /// - `centroids_out`: `[n_blocks * n_kv * head_dim]` f32.
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        key_cache: Handle,
        centroids_out: Handle,
        n_kv: usize,
        head_dim: usize,
        block_size: usize,
        seq_len: usize,
        n_blocks: usize,
    ) {
        let params: [f32; 4] = [
            n_kv as f32,
            head_dim as f32,
            block_size as f32,
            seq_len as f32,
        ];
        let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
        unsafe {
            fm_centroid_rebuild_f32::launch_unchecked::<R>(
                client,
                CubeCount::Static((n_blocks * n_kv) as u32, 1, 1),
                CubeDim::new_1d(head_dim as u32),
                BufferArg::from_raw_parts(key_cache, seq_len * n_kv * head_dim),
                BufferArg::from_raw_parts(centroids_out, n_blocks * n_kv * head_dim),
                BufferArg::from_raw_parts(params_handle, 4),
            );
        }
    }
}

/// The arm's per-layer host config + device state (allocated at forward
/// construction when [`flashmemory_budget_metal_enabled`] is armed; the
/// unarmed forward carries no state and dispatches the incumbent dense
/// path bit-identically).
#[cfg(feature = "cubecl_runtime")]
pub struct FmLayerStateGpu {
    /// Per-block mean keys: `[max_blocks * n_kv * head_dim]`.
    pub centroids: Handle,
    /// Kernel (a)'s scores: `[n_kv * max_blocks]`.
    pub scores: Handle,
    /// Kernel (b)'s selection: `[n_kv * FM_MAX_SEL]` u32.
    pub sel: Handle,
    /// Kernel (b)'s selection counts: `[n_kv]` u32.
    pub sel_counts: Handle,
    /// The KV-resolution query scratch: `[n_kv * head_dim]`.
    pub kv_query: Handle,
    /// The FM block size (the CPU lane's FlashMemoryConfig.block_size —
    /// default 64; NOT the model context).
    pub fm_block_size: usize,
    /// The refresh period (the CPU lane's default 64).
    pub refresh_period: usize,
    /// ρ — the budget fraction.
    pub rho: f32,
    /// Blocks live at the last refresh (block-open detection is
    /// host-side: `n_blocks != last_n_blocks` → force refresh). `Cell`:
    /// the decode path holds `&self` — interior mutability, single-threaded
    /// per forward.
    pub last_n_blocks: std::cell::Cell<usize>,
    /// The pos the last refresh ran at.
    pub last_refresh_pos: std::cell::Cell<usize>,
}
