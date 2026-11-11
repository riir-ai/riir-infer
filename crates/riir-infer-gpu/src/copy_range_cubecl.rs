//! The FlashStates lane's device range-copy (Plan 630 / Issue 1201 T2).
//!
//! One generic elementwise f32 range copy: copy `len` elements from `src`
//! to `dst`, both addressed by launch-time byte-offset views
//! ([`Handle::offset_start`] — u64-exact, never f32 params, the KV window
//! offsets reach ~75M elements where f32 loses exactness). Covers the two
//! directions the cell needs and nothing else:
//!
//! - KV capture: cache rows `[held·kvd, total·kvd)` → a staging buffer
//!   (then one host readback per turn — never a whole-window read).
//! - GDN boundary capture: one layer's state slot → one arena slot (batched
//!   at capture end into a single readback; zero syncs during the walk).
//!
//! The restore directions need no kernel: KV rows ride the existing
//! [`crate::qwen_attention_cubecl::QwenKvCacheFillSplitBatchedCubeCL`]
//! (staging → cache at `base_pos = 0`), GDN slots ride plain full-buffer
//! writes (small, one per layer).
//!
//! `len` rides the f32 params lane (exact to 2²⁴ — the largest copy the
//! cell issues is one turn's fresh KV rows, ≤ ~640 × kvd ≪ 2²⁴; the launch
//! asserts it). Posture: feature `qwen38_wgpu`, same family as the forward
//! it serves.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::cubecl_runtime::ActiveRuntime;

#[cfg(feature = "qwen38_wgpu")]
#[cube(launch_unchecked)]
fn copy_range_f32(src: &[f32], dst: &mut [f32], params: &[f32]) {
    let len = params[0usize] as usize;
    let idx = ABSOLUTE_POS;
    if idx >= len {
        terminate!();
    }
    dst[idx] = src[idx];
}

/// Launch the range copy. `src`/`dst` are byte-offset views over the
/// caller's handles (`handle.offset_start(byte_off)` sized to the copy);
/// `len` is the f32 element count.
///
/// # Safety
/// - `src`/`dst` views cover ≥ `len` f32 each and do not overlap.
#[cfg(feature = "qwen38_wgpu")]
pub unsafe fn copy_range_f32_launch(
    client: &ComputeClient<ActiveRuntime>,
    src: Handle,
    dst: Handle,
    len_elems: usize,
) {
    assert!(
        len_elems < (1usize << 24),
        "copy_range: len {len_elems} exceeds the f32-params exactness bound"
    );
    if len_elems == 0 {
        return;
    }
    let params: [f32; 1] = [len_elems as f32];
    let params_handle = crate::params_cache::params_handle(client, f32::as_bytes(&params));
    let wg_size = 128u32;
    let num_wg = (len_elems as u32).div_ceil(wg_size).max(1);
    unsafe {
        copy_range_f32::launch_unchecked::<ActiveRuntime>(
            client,
            CubeCount::Static(num_wg, 1, 1),
            CubeDim::new_1d(wg_size),
            BufferArg::from_raw_parts(src, len_elems),
            BufferArg::from_raw_parts(dst, len_elems),
            BufferArg::from_raw_parts(params_handle, 1),
        );
    }
}
