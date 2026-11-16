//! Batched-tangent CubeCL kernels for the lens R-sweep (riir-infer Issue
//! 1222; katgpt-rs Issue 999 T3 follow-up).
//!
//! The CPU lens maps (`riir_infer_core::lens`) propagate ONE tangent
//! `[T·d]` per pass; the full-basis R sweep needs `points · d` passes per
//! doc, which is GPU-class cost (the issue's option 1). Every map is
//! LINEAR in the tangent input over shared per-position capture
//! constants, so batching `B` basis directions is exact: every activation
//! buffer widens to `[B·T, X]` (batch-major) and every per-direction
//! matvec becomes a GEMM over the same weight rows.
//!
//! # Layouts
//!
//! - Batched activation buffer `[B·T, X]`: element `(p, t, i)` at
//!   `(p·T + t)·X + i`.
//! - Capture table `[T·X]` (shared across the batch): `(t, i)` at
//!   `t·X + i`; a batched element's table index is `idx % (T·X)` (the
//!   `[B, T, X]` block size is the modulus).
//! - GDN tangent state scratch `[B, n_v, val·key]`: the `(p, h)` block at
//!   `(p·n_v + h)·val·key`.
//!
//! # Semantics
//!
//! Every kernel mirrors its CPU twin's loop order exactly
//! (`riir_infer_core::lens::{ops, gdn, attn, stack}`): sequential
//! k-ascending sums, the same tap convention (`conv1d_causal_jvp`'s
//! flipped taps), the same GQA group rule (`kv_group(h) = h·n_kv/n_head`),
//! the forward's RoPE pairing (`(i, i + rotary_dim/2)`,
//! `(x·c − y·s, x·s + y·c)`). Cross-backend f32 is tolerance-equal, not
//! bit-equal — the CPU reference dot is SIMD-lane reassociated, so no
//! cross-backend pin can be bit-exact (the parity gate is per-column
//! rel-RMS; recorded in Issue 1222 as the 940-law local-only
//! reclassification — the lens fit is offline diagnostics, never synced).
//!
//! # CubeCL v0.11 constraints (the crate's house rules)
//!
//! - No conditional expressions as values — `if { }` statements.
//! - Composed absolute ids fold `CUBE_POS_Y` (the launcher caps
//!   workgroups per x-dim at 65 535, so y is load-bearing above the cap).
//! - `f32::new(literal)` for constants in `#[cube]` context.
//! - `terminate!()` instead of `return`.
//! - `while` loops for runtime bounds.
//!
//! Never a serving path — the lens fit is offline (once per checkpoint).

#[cfg(feature = "cubecl_runtime")]
use cubecl::prelude::*;

#[cfg(feature = "cubecl_runtime")]
use cubecl::server::Handle;

#[cfg(feature = "cubecl_runtime")]
use crate::cubecl_runtime::{debug_assert_binding_at_least, f32_exact};

#[cfg(feature = "cubecl_runtime")]
const WG: u32 = 256;

#[cfg(feature = "cubecl_runtime")]
const MAX_WG_PER_DIM: u64 = 65_535;

/// Composed workgroup-capable launch count (x-dim capped at 65 535,
/// remainder in y; the kernels fold y back in).
#[cfg(feature = "cubecl_runtime")]
fn capped_cube_count(n_wg: u64) -> CubeCount {
    let n_wg = n_wg.max(1);
    if n_wg <= MAX_WG_PER_DIM {
        CubeCount::Static(n_wg as u32, 1, 1)
    } else {
        let y = n_wg.div_ceil(MAX_WG_PER_DIM) as u32;
        CubeCount::Static(MAX_WG_PER_DIM as u32, y, 1)
    }
}

// ---------------------------------------------------------------------------
// Batched GEMM: y[p·rows + r] = Σ_k w[r·cols + k] · x[p·cols + k]
// ---------------------------------------------------------------------------

/// Batched matvec-over-rows — the batched twin of `katgpt_types::matmul`
/// (`y = W @ x` per direction). Layouts: `x [P, cols]`, `w [rows, cols]`
/// row-major, `y [P, rows]`. One thread per output; k ascending.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn lens_gemm_f32(x: &[f32], w: &[f32], y: &mut [f32], params: &[u32]) {
    let p_rows = params[0usize] as usize;
    let rows = params[1usize] as usize;
    let cols = params[2usize] as usize;
    let tid = ((CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * CUBE_DIM_X + UNIT_POS_X) as usize;
    if tid < p_rows * rows {
        let p = tid / rows;
        let r = tid % rows;
        let mut acc = f32::new(0.0f32);
        let w_row = r * cols;
        let x_row = p * cols;
        let mut k = 0usize;
        while k < cols {
            acc += w[w_row + k] * x[x_row + k];
            k += 1;
        }
        y[tid] = acc;
    }
}

/// [`lens_gemm_f32`] launcher.
///
/// # Safety
///
/// - `x_handle`: `p_rows · cols` f32; `w_handle`: `rows · cols` f32;
///   `y_handle`: `p_rows · rows` f32.
#[cfg(feature = "cubecl_runtime")]
pub struct LensGemmCL;

#[cfg(feature = "cubecl_runtime")]
impl LensGemmCL {
    /// # Safety
    ///
    /// Buffer bindings must satisfy the struct's documented contract (the
    /// per-handle element counts in the struct-level `# Safety` block).
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        x: Handle,
        w: Handle,
        y: Handle,
        p_rows: usize,
        rows: usize,
        cols: usize,
    ) {
        debug_assert_binding_at_least(&x, p_rows * cols, "lens_gemm x");
        debug_assert_binding_at_least(&w, rows * cols, "lens_gemm w");
        debug_assert_binding_at_least(&y, p_rows * rows, "lens_gemm y");
        let params: [u32; 3] = [
            f32_exact(p_rows) as u32,
            f32_exact(rows) as u32,
            f32_exact(cols) as u32,
        ];
        let ph = crate::params_cache::params_handle(client, u32::as_bytes(&params));
        let n_wg = (p_rows * rows).div_ceil(WG as usize) as u64;
        unsafe {
            lens_gemm_f32::launch_unchecked::<R>(
                client,
                capped_cube_count(n_wg),
                CubeDim::new_1d(WG),
                BufferArg::from_raw_parts(x, p_rows * cols),
                BufferArg::from_raw_parts(w, rows * cols),
                BufferArg::from_raw_parts(y, p_rows * rows),
                BufferArg::from_raw_parts(ph, 3),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Elementwise: scale-by-table, add, half rule
// ---------------------------------------------------------------------------

/// `out[i] = a[i] · table[i % table_len]` — the batched twin of the LN-rule
/// `diag_apply` and the gate-coefficient multiplies (the table is the
/// per-position capture; the modulus is the `[T·X]` block size).
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn lens_scale_table_f32(a: &[f32], table: &[f32], out: &mut [f32], params: &[u32]) {
    let rows = params[0usize] as usize;
    let row_len = params[1usize] as usize;
    let tid = ((CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * CUBE_DIM_X + UNIT_POS_X) as usize;
    // Factors, never the monolithic count: f32-exact params cap at 2^24,
    // and `batch·t_len·X` exceeds it at real geometry. `tid / row_len < rows`
    // ⇔ `tid < rows·row_len` exactly (integer division), and the modulus is
    // unchanged (row_len == table_len).
    if tid / row_len < rows {
        out[tid] = a[tid] * table[tid % row_len];
    }
}

/// [`lens_scale_table_f32`] launcher.
///
/// # Safety
///
/// - `a_handle` / `out_handle`: `n` f32; `table_handle`: `table_len` f32.
#[cfg(feature = "cubecl_runtime")]
pub struct LensScaleTableCL;

#[cfg(feature = "cubecl_runtime")]
impl LensScaleTableCL {
    /// # Safety
    ///
    /// Buffer bindings must satisfy the struct's documented contract (the
    /// per-handle element counts in the struct-level `# Safety` block).
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        a: Handle,
        table: Handle,
        out: Handle,
        n: usize,
        table_len: usize,
    ) {
        debug_assert_binding_at_least(&a, n, "lens_scale a");
        debug_assert_binding_at_least(&table, table_len, "lens_scale table");
        debug_assert_binding_at_least(&out, n, "lens_scale out");
        // f32-exact params discipline (plan 611 S2/S3): the total `n` =
        // batch·t_len·X exceeds 2^24 at real geometry, so the kernel takes
        // the FACTORS. table_len is the modulus row length, so n = rows ×
        // table_len always — a non-multiple refuses loud instead of rounding.
        assert!(
            table_len > 0 && n.is_multiple_of(table_len),
            "LensScaleTableCL: n ({n}) must be a positive multiple of table_len ({table_len})"
        );
        let rows = n / table_len;
        let params: [u32; 2] = [f32_exact(rows) as u32, f32_exact(table_len) as u32];
        let ph = crate::params_cache::params_handle(client, u32::as_bytes(&params));
        let n_wg = n.div_ceil(WG as usize) as u64;
        unsafe {
            lens_scale_table_f32::launch_unchecked::<R>(
                client,
                capped_cube_count(n_wg),
                CubeDim::new_1d(WG),
                BufferArg::from_raw_parts(a, n),
                BufferArg::from_raw_parts(table, table_len),
                BufferArg::from_raw_parts(out, n),
                BufferArg::from_raw_parts(ph, 2),
            );
        }
    }
}

/// In-place residual add: `acc[i] += add[i]` — the batched twin of the
/// stack's `s.t[i] += s.t2[i]` residual plumbing.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn lens_add_inplace_f32(acc: &mut [f32], add: &[f32], params: &[u32]) {
    let rows = params[0usize] as usize;
    let row_len = params[1usize] as usize;
    let tid = ((CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * CUBE_DIM_X + UNIT_POS_X) as usize;
    if tid / row_len < rows {
        acc[tid] += add[tid];
    }
}

/// [`lens_add_inplace_f32`] launcher.
///
/// # Safety
///
/// - `acc_handle` / `add_handle`: `n` f32 each.
#[cfg(feature = "cubecl_runtime")]
pub struct LensAddInplaceCL;

#[cfg(feature = "cubecl_runtime")]
impl LensAddInplaceCL {
    /// # Safety
    ///
    /// Buffer bindings must satisfy the struct's documented contract (the
    /// per-handle element counts in the struct-level `# Safety` block).
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        acc: Handle,
        add: Handle,
        n: usize,
        row_len: usize,
    ) {
        debug_assert_binding_at_least(&acc, n, "lens_add acc");
        debug_assert_binding_at_least(&add, n, "lens_add add");
        // f32-exact params discipline: n (batch·t_len·d) exceeds 2^24 at real
        // geometry — take the factors, range by `tid / row_len < rows`.
        assert!(
            row_len > 0 && n.is_multiple_of(row_len),
            "LensAddInplaceCL: n ({n}) must be a positive multiple of row_len ({row_len})"
        );
        let rows = n / row_len;
        let params: [u32; 2] = [f32_exact(rows) as u32, f32_exact(row_len) as u32];
        let ph = crate::params_cache::params_handle(client, u32::as_bytes(&params));
        let n_wg = n.div_ceil(WG as usize) as u64;
        unsafe {
            lens_add_inplace_f32::launch_unchecked::<R>(
                client,
                capped_cube_count(n_wg),
                CubeDim::new_1d(WG),
                BufferArg::from_raw_parts(acc, n),
                BufferArg::from_raw_parts(add, n),
                BufferArg::from_raw_parts(ph, 2),
            );
        }
    }
}

/// The half rule's batched elementwise core: `out[i] = ½(a[m]·x[i] +
/// b[m]·c[m]·y[i])` with `m = i % table_len`. Serves BOTH the MLP half rule
/// (`a=s, x=du, b=u, c=sp, y=dg`) and the GDN/attention gate half rule
/// (`a=sg, x=dz, b=z, c=spg, y=dg`) — the two CPU forms are the same shape
/// (`ops::half_rule_jvp` / `ops::gate_half_rule_jvp`).
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn lens_half_rule2_f32(
    a_tab: &[f32],
    b_tab: &[f32],
    c_tab: &[f32],
    x: &[f32],
    y: &[f32],
    out: &mut [f32],
    params: &[u32],
) {
    let rows = params[0usize] as usize;
    let row_len = params[1usize] as usize;
    let tid = ((CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * CUBE_DIM_X + UNIT_POS_X) as usize;
    if tid / row_len < rows {
        let m = tid % row_len;
        out[tid] = f32::new(0.5f32) * (a_tab[m] * x[tid] + b_tab[m] * c_tab[m] * y[tid]);
    }
}

/// [`lens_half_rule2_f32`] launcher.
///
/// # Safety
///
/// - table handles: `table_len` f32 each; `x`/`y`/`out`: `n` f32 each.
#[cfg(feature = "cubecl_runtime")]
pub struct LensHalfRule2CL;

#[cfg(feature = "cubecl_runtime")]
impl LensHalfRule2CL {
    #[allow(clippy::too_many_arguments)]
    /// # Safety
    ///
    /// Buffer bindings must satisfy the struct's documented contract (the
    /// per-handle element counts in the struct-level `# Safety` block).
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        a_tab: Handle,
        b_tab: Handle,
        c_tab: Handle,
        x: Handle,
        y: Handle,
        out: Handle,
        n: usize,
        table_len: usize,
    ) {
        debug_assert_binding_at_least(&a_tab, table_len, "half2 a");
        debug_assert_binding_at_least(&b_tab, table_len, "half2 b");
        debug_assert_binding_at_least(&c_tab, table_len, "half2 c");
        debug_assert_binding_at_least(&x, n, "half2 x");
        debug_assert_binding_at_least(&y, n, "half2 y");
        debug_assert_binding_at_least(&out, n, "half2 out");
        // f32-exact params discipline: n (batch·t_len·X) exceeds 2^24 at real
        // geometry — the kernel takes the factors; table_len is the modulus
        // row length, so the multiple relation always holds.
        assert!(
            table_len > 0 && n.is_multiple_of(table_len),
            "LensHalfRule2CL: n ({n}) must be a positive multiple of table_len ({table_len})"
        );
        let rows = n / table_len;
        let params: [u32; 2] = [f32_exact(rows) as u32, f32_exact(table_len) as u32];
        let ph = crate::params_cache::params_handle(client, u32::as_bytes(&params));
        let n_wg = n.div_ceil(WG as usize) as u64;
        unsafe {
            lens_half_rule2_f32::launch_unchecked::<R>(
                client,
                capped_cube_count(n_wg),
                CubeDim::new_1d(WG),
                BufferArg::from_raw_parts(a_tab, table_len),
                BufferArg::from_raw_parts(b_tab, table_len),
                BufferArg::from_raw_parts(c_tab, table_len),
                BufferArg::from_raw_parts(x, n),
                BufferArg::from_raw_parts(y, n),
                BufferArg::from_raw_parts(out, n),
                BufferArg::from_raw_parts(ph, 2),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// GDN: causal conv, row-slice split, l2norm+expand, scan
// ---------------------------------------------------------------------------

/// Causal depthwise conv1d `jvp`, batched — the twin of
/// `ops::conv1d_causal_jvp` with the SERVING KERNEL's tap order:
/// `dy[p,t,c] = Σ_k K[c, kw−1−k] · dx[p, t−k, c]` (`x[t−k] = 0` for
/// `t < k`).
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn lens_conv1d_causal_f32(dx: &[f32], kernel: &[f32], dy: &mut [f32], params: &[u32]) {
    let batch = params[0usize] as usize;
    let t_len = params[1usize] as usize;
    let channels = params[2usize] as usize;
    let kw = params[3usize] as usize;
    let tid = ((CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * CUBE_DIM_X + UNIT_POS_X) as usize;
    let total = batch * t_len * channels;
    if tid < total {
        let p = tid / (t_len * channels);
        let rem = tid % (t_len * channels);
        let t = rem / channels;
        let c = rem % channels;
        let k_max = if kw < t + 1 { kw } else { t + 1 };
        let mut acc = f32::new(0.0f32);
        let mut k = 0usize;
        while k < k_max {
            acc += kernel[c * kw + (kw - 1 - k)] * dx[(p * t_len + t - k) * channels + c];
            k += 1;
        }
        dy[tid] = acc;
    }
}

/// [`lens_conv1d_causal_f32`] launcher.
///
/// # Safety
///
/// - `dx`/`dy`: `batch · t_len · channels` f32; `kernel`: `channels · kw`
///   f32.
#[cfg(feature = "cubecl_runtime")]
pub struct LensConv1dCL;

#[cfg(feature = "cubecl_runtime")]
impl LensConv1dCL {
    /// # Safety
    ///
    /// Buffer bindings must satisfy the struct's documented contract (the
    /// per-handle element counts in the struct-level `# Safety` block).
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        dx: Handle,
        kernel: Handle,
        dy: Handle,
        batch: usize,
        t_len: usize,
        channels: usize,
        kw: usize,
    ) {
        let n = batch * t_len * channels;
        debug_assert_binding_at_least(&dx, n, "lens_conv dx");
        debug_assert_binding_at_least(&kernel, channels * kw, "lens_conv kernel");
        debug_assert_binding_at_least(&dy, n, "lens_conv dy");
        let params: [u32; 4] = [
            f32_exact(batch) as u32,
            f32_exact(t_len) as u32,
            f32_exact(channels) as u32,
            f32_exact(kw) as u32,
        ];
        let ph = crate::params_cache::params_handle(client, u32::as_bytes(&params));
        let n_wg = n.div_ceil(WG as usize) as u64;
        unsafe {
            lens_conv1d_causal_f32::launch_unchecked::<R>(
                client,
                capped_cube_count(n_wg),
                CubeDim::new_1d(WG),
                BufferArg::from_raw_parts(dx, n),
                BufferArg::from_raw_parts(kernel, channels * kw),
                BufferArg::from_raw_parts(dy, n),
                BufferArg::from_raw_parts(ph, 4),
            );
        }
    }
}

/// Row-slice copy: `dst[p,t,j] = src[p, t, off + j]` — the dqkv tangent's
/// contiguous split into `(dq, dk, dv)` rows (`lens::gdn` step 2's
/// `copy_from_slice` splits).
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn lens_copy_row_slice_f32(src: &[f32], dst: &mut [f32], params: &[u32]) {
    let batch = params[0usize] as usize;
    let t_len = params[1usize] as usize;
    let src_c = params[2usize] as usize;
    let dst_c = params[3usize] as usize;
    let off = params[4usize] as usize;
    let tid = ((CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * CUBE_DIM_X + UNIT_POS_X) as usize;
    let total = batch * t_len * dst_c;
    if tid < total {
        let p = tid / (t_len * dst_c);
        let rem = tid % (t_len * dst_c);
        let t = rem / dst_c;
        let j = rem % dst_c;
        dst[tid] = src[(p * t_len + t) * src_c + off + j];
    }
}

/// [`lens_copy_row_slice_f32`] launcher.
///
/// # Safety
///
/// - `src`: `batch · t_len · src_c` f32; `dst`: `batch · t_len · dst_c`
///   f32; `off + dst_c ≤ src_c`.
#[cfg(feature = "cubecl_runtime")]
pub struct LensCopyRowSliceCL;

#[cfg(feature = "cubecl_runtime")]
impl LensCopyRowSliceCL {
    /// # Safety
    ///
    /// Buffer bindings must satisfy the struct's documented contract (the
    /// per-handle element counts in the struct-level `# Safety` block).
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        src: Handle,
        dst: Handle,
        batch: usize,
        t_len: usize,
        src_c: usize,
        dst_c: usize,
        off: usize,
    ) {
        debug_assert!(off + dst_c <= src_c, "lens_copy_row_slice: slice past row");
        debug_assert_binding_at_least(&src, batch * t_len * src_c, "lens_slice src");
        debug_assert_binding_at_least(&dst, batch * t_len * dst_c, "lens_slice dst");
        let params: [u32; 5] = [
            f32_exact(batch) as u32,
            f32_exact(t_len) as u32,
            f32_exact(src_c) as u32,
            f32_exact(dst_c) as u32,
            f32_exact(off) as u32,
        ];
        let ph = crate::params_cache::params_handle(client, u32::as_bytes(&params));
        let n = batch * t_len * dst_c;
        let n_wg = n.div_ceil(WG as usize) as u64;
        unsafe {
            lens_copy_row_slice_f32::launch_unchecked::<R>(
                client,
                capped_cube_count(n_wg),
                CubeDim::new_1d(WG),
                BufferArg::from_raw_parts(src, batch * t_len * src_c),
                BufferArg::from_raw_parts(dst, n),
                BufferArg::from_raw_parts(ph, 5),
            );
        }
    }
}

/// L2-norm tangent map + tiled expand, fused, batched — the twin of
/// `gdn::l2_norm_jvp` + `gdn::expand_tiled_jvp` (one launch instead of a
/// separate expand pass). Thread per `(p, t, src head h)`: the in-place
/// norm map `inv·(v − y·⟨y, v⟩)` is applied to the SOURCE head's tangent,
/// then the result is tiled `repeat` times into the expanded layout.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn lens_l2norm_expand_f32(
    dx: &[f32],
    y_tab: &[f32],
    inv_tab: &[f32],
    ddst: &mut [f32],
    params: &[u32],
) {
    let batch = params[0usize] as usize;
    let t_len = params[1usize] as usize;
    let n_src = params[2usize] as usize;
    let repeat = params[3usize] as usize;
    let head_dim = params[4usize] as usize;
    // The captured y table is the FULL expanded buffer `[T, n_v·key]` (the
    // CPU consts slice `cap.qn[t * n_v * key ..]`): its per-position stride
    // is wider than the source heads.
    let y_stride = params[5usize] as usize;
    let tid = ((CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * CUBE_DIM_X + UNIT_POS_X) as usize;
    let total = batch * t_len * n_src;
    if tid < total {
        let p = tid / (t_len * n_src);
        let rem = tid % (t_len * n_src);
        let t = rem / n_src;
        let h = rem % n_src;
        let inv = inv_tab[t * n_src + h];
        let row_base = (p * t_len + t) * n_src;
        let dst_base = (p * t_len + t) * (n_src * repeat * head_dim);
        if inv == f32::new(0.0f32) {
            // The forward's degenerate guard: the tangent map is the
            // identity — tile the untouched tangent.
            let mut r = 0usize;
            while r < repeat {
                let mut i = 0usize;
                while i < head_dim {
                    ddst[dst_base + (r * n_src + h) * head_dim + i] =
                        dx[row_base * head_dim + h * head_dim + i];
                    i += 1;
                }
                r += 1;
            }
        } else {
            let y_base = t * y_stride + h * head_dim;
            let mut dot = f32::new(0.0f32);
            let mut i = 0usize;
            while i < head_dim {
                dot += y_tab[y_base + i] * dx[row_base * head_dim + h * head_dim + i];
                i += 1;
            }
            let mut r = 0usize;
            while r < repeat {
                let mut i = 0usize;
                while i < head_dim {
                    let v = dx[row_base * head_dim + h * head_dim + i];
                    ddst[dst_base + (r * n_src + h) * head_dim + i] =
                        inv * (v - y_tab[y_base + i] * dot);
                    i += 1;
                }
                r += 1;
            }
        }
    }
}

/// [`lens_l2norm_expand_f32`] launcher.
///
/// # Safety
///
/// - `dx`: `batch·t_len·n_src·head_dim`; `y_tab`: the FULL expanded table
///   `t_len·y_stride` (`y_stride = n_v·head_dim` for the stack's qn/kn);
///   `inv_tab`: `t_len·n_src`; `ddst`: `batch·t_len·n_src·repeat·head_dim`.
#[cfg(feature = "cubecl_runtime")]
pub struct LensL2NormExpandCL;

#[cfg(feature = "cubecl_runtime")]
impl LensL2NormExpandCL {
    /// # Safety
    ///
    /// Buffer bindings must satisfy the struct's documented contract (the
    /// per-handle element counts in the struct-level `# Safety` block).
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        dx: Handle,
        y_tab: Handle,
        inv_tab: Handle,
        ddst: Handle,
        batch: usize,
        t_len: usize,
        n_src: usize,
        repeat: usize,
        head_dim: usize,
        y_stride: usize,
    ) {
        debug_assert!(y_stride >= n_src * head_dim, "l2x y_stride");
        debug_assert_binding_at_least(&dx, batch * t_len * n_src * head_dim, "l2x dx");
        debug_assert_binding_at_least(&y_tab, t_len * y_stride, "l2x y");
        debug_assert_binding_at_least(&inv_tab, t_len * n_src, "l2x inv");
        debug_assert_binding_at_least(&ddst, batch * t_len * n_src * repeat * head_dim, "l2x dst");
        let params: [u32; 6] = [
            f32_exact(batch) as u32,
            f32_exact(t_len) as u32,
            f32_exact(n_src) as u32,
            f32_exact(repeat) as u32,
            f32_exact(head_dim) as u32,
            f32_exact(y_stride) as u32,
        ];
        let ph = crate::params_cache::params_handle(client, u32::as_bytes(&params));
        let n = batch * t_len * n_src;
        let n_wg = n.div_ceil(WG as usize) as u64;
        unsafe {
            lens_l2norm_expand_f32::launch_unchecked::<R>(
                client,
                capped_cube_count(n_wg),
                CubeDim::new_1d(WG),
                BufferArg::from_raw_parts(dx, batch * t_len * n_src * head_dim),
                BufferArg::from_raw_parts(y_tab, t_len * y_stride),
                BufferArg::from_raw_parts(inv_tab, t_len * n_src),
                BufferArg::from_raw_parts(ddst, batch * t_len * n_src * repeat * head_dim),
                BufferArg::from_raw_parts(ph, 6),
            );
        }
    }
}

/// GDN recurrence scan `jvp`, batched — the twin of
/// `gdn::gdn_recurrence_jvp`, thread per `(p, h, r)` (batch direction ×
/// value head × output row), sequential over `t` with the CPU's exact row
/// math and loop order. The running tangent state `ds` lives in global
/// memory `[batch, n_v, val·key]` and SELF-INITIALIZES: at `t == 0` the
/// CPU's zero-on-entry makes the first step write zero, so no fill launch
/// is needed and an uninitialized pooled buffer is safe.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn lens_gdn_scan_f32(
    qn: &[f32],
    kn: &[f32],
    v: &[f32],
    beta: &[f32],
    decay: &[f32],
    dbeta: &[f32],
    ddecay: &[f32],
    dq: &[f32],
    dk: &[f32],
    dv: &[f32],
    states: &[f32],
    ds: &mut [f32],
    spre: &mut [f32],
    dy_out: &mut [f32],
    params: &[u32],
    params_f: &[f32],
) {
    let batch = params[0usize] as usize;
    let t_len = params[1usize] as usize;
    let n_v = params[2usize] as usize;
    let key_dim = params[3usize] as usize;
    let val_dim = params[4usize] as usize;
    let scale = params_f[0usize];
    let per_head = val_dim * key_dim;
    let tid = ((CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * CUBE_DIM_X + UNIT_POS_X) as usize;
    let total = batch * n_v * val_dim;
    if tid < total {
        let p = tid / (n_v * val_dim);
        let rem = tid % (n_v * val_dim);
        let h = rem / val_dim;
        let r = rem % val_dim;
        let ds_base = (p * n_v + h) * per_head + r * key_dim;
        let t_n_v = n_v;
        let mut t = 0usize;
        while t < t_len {
            let gate_idx = t * t_n_v + h;
            let g = decay[gate_idx];
            let b = beta[gate_idx];
            let bt_idx = (p * t_len + t) * t_n_v + h;
            let dg = ddecay[bt_idx];
            let db = dbeta[bt_idx];
            // Tangent state: dS̃ = g·dS_prev + dg·S_prev; S̃ = g·S_prev
            // (S_0 = 0). PER-HEAD blocks — heads never share the state.
            let prev_off = if t == 0 {
                0usize
            } else {
                ((t - 1) * t_n_v + h) * per_head
            };
            let mut i = 0usize;
            while i < key_dim {
                if t == 0 {
                    // The CPU twin zeroes ds on entry; sp = 0 at t = 0, so
                    // the first step degenerates to writing zero — the
                    // buffer self-initializes (no fill launch, no
                    // uninitialized-pool read).
                    ds[ds_base + i] = f32::new(0.0f32);
                    spre[ds_base + i] = f32::new(0.0f32);
                } else {
                    let sp = states[prev_off + r * key_dim + i];
                    ds[ds_base + i] = g * ds[ds_base + i] + dg * sp;
                    spre[ds_base + i] = g * sp;
                }
                i += 1;
            }
            let cur_off = gate_idx * per_head;
            let q_base = t * t_n_v * key_dim + h * key_dim;
            let v_base = t * t_n_v * val_dim + h * val_dim;
            let dv_base = (p * t_len + t) * t_n_v * val_dim + h * val_dim;
            let dq_base = (p * t_len + t) * t_n_v * key_dim + h * key_dim;
            // Row math — the CPU scan's exact order (single accumulators,
            // ci ascending).
            let mut sk = f32::new(0.0f32);
            let mut kv_dot = f32::new(0.0f32);
            let mut ci = 0usize;
            while ci < key_dim {
                sk += spre[ds_base + ci] * kn[q_base + ci];
                kv_dot +=
                    ds[ds_base + ci] * kn[q_base + ci] + spre[ds_base + ci] * dk[dq_base + ci];
                ci += 1;
            }
            let vm = v[v_base + r] - sk;
            let d_r = b * vm;
            let dd = b * (dv[dv_base + r] - kv_dot) + db * vm;
            let mut ci2 = 0usize;
            while ci2 < key_dim {
                ds[ds_base + ci2] += dd * kn[q_base + ci2] + d_r * dk[dq_base + ci2];
                ci2 += 1;
            }
            let mut dy_read = f32::new(0.0f32);
            let mut dy_state = f32::new(0.0f32);
            let mut ci3 = 0usize;
            while ci3 < key_dim {
                dy_read += ds[ds_base + ci3] * qn[q_base + ci3];
                dy_state += states[cur_off + r * key_dim + ci3] * dq[dq_base + ci3];
                ci3 += 1;
            }
            dy_out[dv_base + r] = (dy_read + dy_state) * scale;
            t += 1;
        }
    }
}

/// [`lens_gdn_scan_f32`] launcher.
///
/// # Safety
///
/// - shared tables: `qn`/`kn` `t_len·n_v·key_dim`, `v` `t_len·n_v·val_dim`,
///   `beta`/`decay` `t_len·n_v`, `states` `t_len·n_v·val_dim·key_dim`;
/// - batched: `dbeta`/`ddecay` `batch·t_len·n_v`, `dq`/`dk`
///   `batch·t_len·n_v·key_dim`, `dv`/`dy_out` `batch·t_len·n_v·val_dim`;
/// - scratch: `ds`/`spre` `batch·n_v·val_dim·key_dim` (any contents — the
///   scan self-initializes at `t == 0`).
#[cfg(feature = "cubecl_runtime")]
pub struct LensGdnScanCL;

#[cfg(feature = "cubecl_runtime")]
impl LensGdnScanCL {
    #[allow(clippy::too_many_arguments)]
    /// # Safety
    ///
    /// Buffer bindings must satisfy the struct's documented contract (the
    /// per-handle element counts in the struct-level `# Safety` block).
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        qn: Handle,
        kn: Handle,
        v: Handle,
        beta: Handle,
        decay: Handle,
        dbeta: Handle,
        ddecay: Handle,
        dq: Handle,
        dk: Handle,
        dv: Handle,
        states: Handle,
        ds: Handle,
        spre: Handle,
        dy_out: Handle,
        batch: usize,
        t_len: usize,
        n_v: usize,
        key_dim: usize,
        val_dim: usize,
        scale: f32,
    ) {
        let per_head = val_dim * key_dim;
        debug_assert_binding_at_least(&qn, t_len * n_v * key_dim, "scan qn");
        debug_assert_binding_at_least(&kn, t_len * n_v * key_dim, "scan kn");
        debug_assert_binding_at_least(&v, t_len * n_v * val_dim, "scan v");
        debug_assert_binding_at_least(&beta, t_len * n_v, "scan beta");
        debug_assert_binding_at_least(&decay, t_len * n_v, "scan decay");
        debug_assert_binding_at_least(&dbeta, batch * t_len * n_v, "scan dbeta");
        debug_assert_binding_at_least(&ddecay, batch * t_len * n_v, "scan ddecay");
        debug_assert_binding_at_least(&states, t_len * n_v * per_head, "scan states");
        debug_assert_binding_at_least(&ds, batch * n_v * per_head, "scan ds");
        debug_assert_binding_at_least(&spre, batch * n_v * per_head, "scan spre");
        debug_assert_binding_at_least(&dq, batch * t_len * n_v * key_dim, "scan dq");
        debug_assert_binding_at_least(&dk, batch * t_len * n_v * key_dim, "scan dk");
        debug_assert_binding_at_least(&dv, batch * t_len * n_v * val_dim, "scan dv");
        debug_assert_binding_at_least(&dy_out, batch * t_len * n_v * val_dim, "scan dy");
        let params: [u32; 5] = [
            f32_exact(batch) as u32,
            f32_exact(t_len) as u32,
            f32_exact(n_v) as u32,
            f32_exact(key_dim) as u32,
            f32_exact(val_dim) as u32,
        ];
        let ph = crate::params_cache::params_handle(client, u32::as_bytes(&params));
        let pf = crate::params_cache::params_handle(client, f32::as_bytes(&[scale]));
        let n = batch * n_v * val_dim;
        let n_wg = n.div_ceil(WG as usize) as u64;
        unsafe {
            lens_gdn_scan_f32::launch_unchecked::<R>(
                client,
                capped_cube_count(n_wg),
                CubeDim::new_1d(WG),
                BufferArg::from_raw_parts(qn, t_len * n_v * key_dim),
                BufferArg::from_raw_parts(kn, t_len * n_v * key_dim),
                BufferArg::from_raw_parts(v, t_len * n_v * val_dim),
                BufferArg::from_raw_parts(beta, t_len * n_v),
                BufferArg::from_raw_parts(decay, t_len * n_v),
                BufferArg::from_raw_parts(dbeta, batch * t_len * n_v),
                BufferArg::from_raw_parts(ddecay, batch * t_len * n_v),
                BufferArg::from_raw_parts(dq, batch * t_len * n_v * key_dim),
                BufferArg::from_raw_parts(dk, batch * t_len * n_v * key_dim),
                BufferArg::from_raw_parts(dv, batch * t_len * n_v * val_dim),
                BufferArg::from_raw_parts(states, t_len * n_v * per_head),
                BufferArg::from_raw_parts(ds, batch * n_v * per_head),
                BufferArg::from_raw_parts(spre, batch * n_v * per_head),
                BufferArg::from_raw_parts(dy_out, batch * t_len * n_v * val_dim),
                BufferArg::from_raw_parts(ph, 5),
                BufferArg::from_raw_parts(pf, 1),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Norms: qk/ssm true-derivative jvp (batched)
// ---------------------------------------------------------------------------

/// QK-norm (and the GDN ssm_norm — same shape) tangent map, batched — the
/// twin of `attn::qk_norm_true_jvp`: per `(p, t, head)`,
/// `dy = gamma·(r·dx − x·(r³·⟨x, dx⟩/hd))`, the true RMSNorm derivative
/// with the captured rsqrt `r`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn lens_qk_norm_true_f32(
    x_tab: &[f32],
    r_tab: &[f32],
    gamma: &[f32],
    dx: &[f32],
    dy: &mut [f32],
    params: &[u32],
) {
    let batch = params[0usize] as usize;
    let t_len = params[1usize] as usize;
    let heads = params[2usize] as usize;
    let hd = params[3usize] as usize;
    let tid = ((CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * CUBE_DIM_X + UNIT_POS_X) as usize;
    let total = batch * t_len * heads;
    if tid < total {
        let p = tid / (t_len * heads);
        let rem = tid % (t_len * heads);
        let t = rem / heads;
        let h = rem % heads;
        let x_base = t * heads * hd + h * hd;
        let dx_base = (p * t_len + t) * heads * hd + h * hd;
        let r = r_tab[t * heads + h];
        let r3 = r * r * r;
        let mut dot_xdx = f32::new(0.0f32);
        let mut i = 0usize;
        while i < hd {
            dot_xdx += x_tab[x_base + i] * dx[dx_base + i];
            i += 1;
        }
        let coef = r3 * dot_xdx / (params[3usize] as f32);
        let mut i2 = 0usize;
        while i2 < hd {
            dy[dx_base + i2] = gamma[i2] * (r * dx[dx_base + i2] - x_tab[x_base + i2] * coef);
            i2 += 1;
        }
    }
}

/// [`lens_qk_norm_true_f32`] launcher.
///
/// # Safety
///
/// - `x_tab`: `t_len·heads·hd`; `r_tab`: `t_len·heads`; `gamma`: `hd`;
///   `dx`/`dy`: `batch·t_len·heads·hd`.
#[cfg(feature = "cubecl_runtime")]
pub struct LensQkNormCL;

#[cfg(feature = "cubecl_runtime")]
impl LensQkNormCL {
    /// # Safety
    ///
    /// Buffer bindings must satisfy the struct's documented contract (the
    /// per-handle element counts in the struct-level `# Safety` block).
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        x_tab: Handle,
        r_tab: Handle,
        gamma: Handle,
        dx: Handle,
        dy: Handle,
        batch: usize,
        t_len: usize,
        heads: usize,
        hd: usize,
    ) {
        debug_assert_binding_at_least(&x_tab, t_len * heads * hd, "qkn x");
        debug_assert_binding_at_least(&r_tab, t_len * heads, "qkn r");
        debug_assert_binding_at_least(&gamma, hd, "qkn gamma");
        debug_assert_binding_at_least(&dx, batch * t_len * heads * hd, "qkn dx");
        debug_assert_binding_at_least(&dy, batch * t_len * heads * hd, "qkn dy");
        let params: [u32; 4] = [
            f32_exact(batch) as u32,
            f32_exact(t_len) as u32,
            f32_exact(heads) as u32,
            f32_exact(hd) as u32,
        ];
        let ph = crate::params_cache::params_handle(client, u32::as_bytes(&params));
        let n = batch * t_len * heads;
        let n_wg = n.div_ceil(WG as usize) as u64;
        unsafe {
            lens_qk_norm_true_f32::launch_unchecked::<R>(
                client,
                capped_cube_count(n_wg),
                CubeDim::new_1d(WG),
                BufferArg::from_raw_parts(x_tab, t_len * heads * hd),
                BufferArg::from_raw_parts(r_tab, t_len * heads),
                BufferArg::from_raw_parts(gamma, hd),
                BufferArg::from_raw_parts(dx, batch * t_len * heads * hd),
                BufferArg::from_raw_parts(dy, batch * t_len * heads * hd),
                BufferArg::from_raw_parts(ph, 4),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Attention: jvp (true derivative), RoPE, gated-q split
// ---------------------------------------------------------------------------

/// Softmax-attention `jvp`, batched — the twin of `attn::attention_jvp`.
/// Thread per `(p, t, h)`: the first pass computes `row_sum = ⟨a⊙ds⟩`
/// (ds per key `j`: `(⟨dq,k_j⟩ + ⟨q,dk_j⟩)·scale`); the output pass
/// RECOMPUTES each `ds_j` (the per-thread `ds` row would need t·f32 of
/// shared memory — the recompute doubles the small attention-core dot cost
/// instead of a shared-memory t·256 tile) and accumulates
/// `dout += da·v_j + a_j·dv_j`, `da = a_j·(ds_j − row_sum)`.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn lens_attn_jvp_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    attn: &[f32],
    dq: &[f32],
    dk: &[f32],
    dv: &[f32],
    dout: &mut [f32],
    params: &[u32],
    params_f: &[f32],
) {
    let batch = params[0usize] as usize;
    let t_len = params[1usize] as usize;
    let n_head = params[2usize] as usize;
    let n_kv = params[3usize] as usize;
    let hd = params[4usize] as usize;
    let scale = params_f[0usize];
    let q_dim = n_head * hd;
    let kv_dim = n_kv * hd;
    let tid = ((CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * CUBE_DIM_X + UNIT_POS_X) as usize;
    let total = batch * t_len * n_head;
    if tid < total {
        let p = tid / (t_len * n_head);
        let rem = tid % (t_len * n_head);
        let t = rem / n_head;
        let h = rem % n_head;
        let kv = h * n_kv / n_head;
        let t_n = t + 1;
        let q_base = (p * t_len + t) * q_dim + h * hd;
        let q_s_base = t * q_dim + h * hd;
        let attn_base = (t * n_head + h) * t_len;
        // Pass 1: row_sum over the causal keys.
        let mut row_sum = f32::new(0.0f32);
        let mut j = 0usize;
        while j < t_n {
            let k_off = j * kv_dim + kv * hd;
            let mut ds = f32::new(0.0f32);
            let mut d = 0usize;
            while d < hd {
                ds += dq[q_base + d] * k[k_off + d] + q[q_s_base + d] * dk[k_off + d];
                d += 1;
            }
            ds *= scale;
            row_sum += attn[attn_base + j] * ds;
            j += 1;
        }
        // Pass 2: recompute ds_j, accumulate the output row.
        let mut d = 0usize;
        while d < hd {
            dout[q_base + d] = f32::new(0.0f32);
            d += 1;
        }
        let mut j2 = 0usize;
        while j2 < t_n {
            let k_off = j2 * kv_dim + kv * hd;
            let v_off = j2 * kv_dim + kv * hd;
            let mut ds2 = f32::new(0.0f32);
            let mut d2 = 0usize;
            while d2 < hd {
                ds2 += dq[q_base + d2] * k[k_off + d2] + q[q_s_base + d2] * dk[k_off + d2];
                d2 += 1;
            }
            ds2 *= scale;
            let a = attn[attn_base + j2];
            let da = a * (ds2 - row_sum);
            let mut d3 = 0usize;
            while d3 < hd {
                dout[q_base + d3] +=
                    da * v[v_off + d3] + a * dv[(p * t_len + j2) * kv_dim + kv * hd + d3];
                d3 += 1;
            }
            j2 += 1;
        }
    }
}

/// [`lens_attn_jvp_f32`] launcher.
///
/// # Safety
///
/// - shared: `q` `t_len·q_dim`, `k`/`v` `t_len·kv_dim`, `attn`
///   `t_len·n_head·t_len`;
/// - batched: `dq` `batch·t_len·q_dim`, `dk`/`dv` `batch·t_len·kv_dim`,
///   `dout` `batch·t_len·q_dim`.
#[cfg(feature = "cubecl_runtime")]
pub struct LensAttnJvpCL;

#[cfg(feature = "cubecl_runtime")]
impl LensAttnJvpCL {
    #[allow(clippy::too_many_arguments)]
    /// # Safety
    ///
    /// Buffer bindings must satisfy the struct's documented contract (the
    /// per-handle element counts in the struct-level `# Safety` block).
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        q: Handle,
        k: Handle,
        v: Handle,
        attn: Handle,
        dq: Handle,
        dk: Handle,
        dv: Handle,
        dout: Handle,
        batch: usize,
        t_len: usize,
        n_head: usize,
        n_kv: usize,
        head_dim: usize,
        scale: f32,
    ) {
        let q_dim = n_head * head_dim;
        let kv_dim = n_kv * head_dim;
        debug_assert_binding_at_least(&q, t_len * q_dim, "ajvp q");
        debug_assert_binding_at_least(&k, t_len * kv_dim, "ajvp k");
        debug_assert_binding_at_least(&v, t_len * kv_dim, "ajvp v");
        debug_assert_binding_at_least(&attn, t_len * n_head * t_len, "ajvp attn");
        debug_assert_binding_at_least(&dq, batch * t_len * q_dim, "ajvp dq");
        debug_assert_binding_at_least(&dk, batch * t_len * kv_dim, "ajvp dk");
        debug_assert_binding_at_least(&dv, batch * t_len * kv_dim, "ajvp dv");
        debug_assert_binding_at_least(&dout, batch * t_len * q_dim, "ajvp dout");
        let params: [u32; 5] = [
            f32_exact(batch) as u32,
            f32_exact(t_len) as u32,
            f32_exact(n_head) as u32,
            f32_exact(n_kv) as u32,
            f32_exact(head_dim) as u32,
        ];
        let ph = crate::params_cache::params_handle(client, u32::as_bytes(&params));
        let pf = crate::params_cache::params_handle(client, f32::as_bytes(&[scale]));
        let n = batch * t_len * n_head;
        let n_wg = n.div_ceil(WG as usize) as u64;
        unsafe {
            lens_attn_jvp_f32::launch_unchecked::<R>(
                client,
                capped_cube_count(n_wg),
                CubeDim::new_1d(WG),
                BufferArg::from_raw_parts(q, t_len * q_dim),
                BufferArg::from_raw_parts(k, t_len * kv_dim),
                BufferArg::from_raw_parts(v, t_len * kv_dim),
                BufferArg::from_raw_parts(attn, t_len * n_head * t_len),
                BufferArg::from_raw_parts(dq, batch * t_len * q_dim),
                BufferArg::from_raw_parts(dk, batch * t_len * kv_dim),
                BufferArg::from_raw_parts(dv, batch * t_len * kv_dim),
                BufferArg::from_raw_parts(dout, batch * t_len * q_dim),
                BufferArg::from_raw_parts(ph, 5),
                BufferArg::from_raw_parts(pf, 1),
            );
        }
    }
}

/// RoPE `jvp`, batched — the twin of `ops::rope_jvp` (the forward's exact
/// rotation). Thread per `(p, t, i < rotary_dim/2)`: one `(cos, sin)` pair
/// from the HOST-precomputed `cs` table (`[T, 2·half]` — the same host-side
/// `sin_cos` values the CPU reference computes, so parity is inherited,
/// not re-derived), then every q head and every k head's `(i, i + half)`
/// pair rotates in place with the forward's convention
/// `(x·c − y·s, x·s + y·c)`. `t == 0` is the identity (sin 0, cos 1).
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn lens_rope_jvp_f32(dq: &mut [f32], dk: &mut [f32], cs: &[f32], params: &[u32]) {
    let batch = params[0usize] as usize;
    let t_len = params[1usize] as usize;
    let n_head = params[2usize] as usize;
    let n_kv = params[3usize] as usize;
    let head_dim = params[4usize] as usize;
    let rotary_dim = params[5usize] as usize;
    let half = rotary_dim / 2;
    let q_dim = n_head * head_dim;
    let kv_dim = n_kv * head_dim;
    let tid = ((CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * CUBE_DIM_X + UNIT_POS_X) as usize;
    let total = batch * t_len * half;
    if tid < total {
        let p = tid / (t_len * half);
        let rem = tid % (t_len * half);
        let t = rem / half;
        let i = rem % half;
        let cs_base = t * 2 * half;
        let c = cs[cs_base + i];
        let s = cs[cs_base + half + i];
        // Q heads.
        let q_base = (p * t_len + t) * q_dim;
        let mut h = 0usize;
        while h < n_head {
            let base = q_base + h * head_dim + i;
            let x = dq[base];
            let y = dq[base + half];
            dq[base] = x * c - y * s;
            dq[base + half] = x * s + y * c;
            h += 1;
        }
        // K heads.
        let k_base = (p * t_len + t) * kv_dim;
        let mut hk = 0usize;
        while hk < n_kv {
            let base = k_base + hk * head_dim + i;
            let x = dk[base];
            let y = dk[base + half];
            dk[base] = x * c - y * s;
            dk[base + half] = x * s + y * c;
            hk += 1;
        }
    }
}

/// [`lens_rope_jvp_f32`] launcher.
///
/// # Safety
///
/// - `dq`: `batch·t_len·n_head·head_dim`; `dk`:
///   `batch·t_len·n_kv·head_dim`; `freq`: `rotary_dim/2` f32.
#[cfg(feature = "cubecl_runtime")]
pub struct LensRopeJvpCL;

#[cfg(feature = "cubecl_runtime")]
impl LensRopeJvpCL {
    #[allow(clippy::too_many_arguments)]
    /// # Safety
    ///
    /// Buffer bindings must satisfy the struct's documented contract (the
    /// per-handle element counts in the struct-level `# Safety` block).
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        dq: Handle,
        dk: Handle,
        cs: Handle,
        batch: usize,
        t_len: usize,
        n_head: usize,
        n_kv: usize,
        head_dim: usize,
        rotary_dim: usize,
    ) {
        let q_dim = n_head * head_dim;
        let kv_dim = n_kv * head_dim;
        debug_assert_binding_at_least(&dq, batch * t_len * q_dim, "rope dq");
        debug_assert_binding_at_least(&dk, batch * t_len * kv_dim, "rope dk");
        debug_assert_binding_at_least(&cs, t_len * 2 * (rotary_dim / 2), "rope cs");
        let params: [u32; 6] = [
            f32_exact(batch) as u32,
            f32_exact(t_len) as u32,
            f32_exact(n_head) as u32,
            f32_exact(n_kv) as u32,
            f32_exact(head_dim) as u32,
            f32_exact(rotary_dim) as u32,
        ];
        let ph = crate::params_cache::params_handle(client, u32::as_bytes(&params));
        let n = batch * t_len * (rotary_dim / 2);
        let n_wg = n.div_ceil(WG as usize) as u64;
        unsafe {
            lens_rope_jvp_f32::launch_unchecked::<R>(
                client,
                capped_cube_count(n_wg),
                CubeDim::new_1d(WG),
                BufferArg::from_raw_parts(dq, batch * t_len * q_dim),
                BufferArg::from_raw_parts(dk, batch * t_len * kv_dim),
                BufferArg::from_raw_parts(cs, t_len * 2 * (rotary_dim / 2)),
                BufferArg::from_raw_parts(ph, 6),
            );
        }
    }
}

/// Gated-Q split, batched — the twin of `attn::qg_deinterleave`:
/// `attn_wq`'s tangent row is `[q_h(hd), gate_h(hd)]` per head; split into
/// the `(dq, dgate)` buffers.
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn lens_qg_split_f32(dqg: &[f32], dq: &mut [f32], dgate: &mut [f32], params: &[u32]) {
    let batch = params[0usize] as usize;
    let t_len = params[1usize] as usize;
    let n_head = params[2usize] as usize;
    let hd = params[3usize] as usize;
    let q_dim = n_head * hd;
    let qg_dim = 2 * q_dim;
    let tid = ((CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * CUBE_DIM_X + UNIT_POS_X) as usize;
    let total = batch * t_len * n_head * hd;
    if tid < total {
        let p = tid / (t_len * n_head * hd);
        let rem = tid % (t_len * n_head * hd);
        let t = rem / (n_head * hd);
        let rem2 = rem % (n_head * hd);
        let h = rem2 / hd;
        let i = rem2 % hd;
        let src_base = (p * t_len + t) * qg_dim + h * 2 * hd;
        let dst_base = (p * t_len + t) * q_dim + h * hd;
        dq[dst_base + i] = dqg[src_base + i];
        dgate[dst_base + i] = dqg[src_base + hd + i];
    }
}

/// [`lens_qg_split_f32`] launcher.
///
/// # Safety
///
/// - `dqg`: `batch·t_len·2·n_head·hd`; `dq`/`dgate`:
///   `batch·t_len·n_head·hd`.
#[cfg(feature = "cubecl_runtime")]
pub struct LensQgSplitCL;

#[cfg(feature = "cubecl_runtime")]
impl LensQgSplitCL {
    /// # Safety
    ///
    /// Buffer bindings must satisfy the struct's documented contract (the
    /// per-handle element counts in the struct-level `# Safety` block).
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        dqg: Handle,
        dq: Handle,
        dgate: Handle,
        batch: usize,
        t_len: usize,
        n_head: usize,
        head_dim: usize,
    ) {
        let q_dim = n_head * head_dim;
        debug_assert_binding_at_least(&dqg, batch * t_len * 2 * q_dim, "qg src");
        debug_assert_binding_at_least(&dq, batch * t_len * q_dim, "qg dq");
        debug_assert_binding_at_least(&dgate, batch * t_len * q_dim, "qg dgate");
        let params: [u32; 4] = [
            f32_exact(batch) as u32,
            f32_exact(t_len) as u32,
            f32_exact(n_head) as u32,
            f32_exact(head_dim) as u32,
        ];
        let ph = crate::params_cache::params_handle(client, u32::as_bytes(&params));
        let n = batch * t_len * q_dim;
        let n_wg = n.div_ceil(WG as usize) as u64;
        unsafe {
            lens_qg_split_f32::launch_unchecked::<R>(
                client,
                capped_cube_count(n_wg),
                CubeDim::new_1d(WG),
                BufferArg::from_raw_parts(dqg, batch * t_len * 2 * q_dim),
                BufferArg::from_raw_parts(dq, n),
                BufferArg::from_raw_parts(dgate, n),
                BufferArg::from_raw_parts(ph, 4),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Reduction: per-direction column sum (the accumulator's add_doc_column)
// ---------------------------------------------------------------------------

/// Row-sum reduction over the final tangent: `out[p·d + j] =
/// Σ_{t ∈ [row_lo, row_hi)} buf[p, t, j]` — the batched twin of
/// `fit::ColAccum::add_doc_column(_window)`'s position fold (t ascending;
/// `row_hi = T` is the J sweep's `sum_all_rows`, the seed window is the K
/// sweep's shape).
#[cfg(feature = "cubecl_runtime")]
#[cube(launch_unchecked)]
fn lens_col_sum_f32(buf: &[f32], out: &mut [f32], params: &[u32]) {
    let batch = params[0usize] as usize;
    let t_len = params[1usize] as usize;
    let d = params[2usize] as usize;
    let row_lo = params[3usize] as usize;
    let row_hi = params[4usize] as usize;
    let tid = ((CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * CUBE_DIM_X + UNIT_POS_X) as usize;
    let total = batch * d;
    if tid < total {
        let p = tid / d;
        let j = tid % d;
        let mut acc = f32::new(0.0f32);
        let mut t = row_lo;
        while t < row_hi {
            acc += buf[(p * t_len + t) * d + j];
            t += 1;
        }
        out[tid] = acc;
    }
}

/// [`lens_col_sum_f32`] launcher.
///
/// # Safety
///
/// - `buf`: `batch·t_len·d` f32; `out`: `batch·d` f32; `row_lo ≤ row_hi ≤
///   t_len`.
#[cfg(feature = "cubecl_runtime")]
pub struct LensColSumCL;

#[cfg(feature = "cubecl_runtime")]
impl LensColSumCL {
    /// # Safety
    ///
    /// Buffer bindings must satisfy the struct's documented contract (the
    /// per-handle element counts in the struct-level `# Safety` block).
    pub unsafe fn launch<R: Runtime>(
        client: &ComputeClient<R>,
        buf: Handle,
        out: Handle,
        batch: usize,
        t_len: usize,
        d: usize,
        row_lo: usize,
        row_hi: usize,
    ) {
        debug_assert!(row_lo <= row_hi && row_hi <= t_len, "col_sum window");
        debug_assert_binding_at_least(&buf, batch * t_len * d, "col_sum buf");
        debug_assert_binding_at_least(&out, batch * d, "col_sum out");
        let params: [u32; 5] = [
            f32_exact(batch) as u32,
            f32_exact(t_len) as u32,
            f32_exact(d) as u32,
            f32_exact(row_lo) as u32,
            f32_exact(row_hi) as u32,
        ];
        let ph = crate::params_cache::params_handle(client, u32::as_bytes(&params));
        let n = batch * d;
        let n_wg = n.div_ceil(WG as usize) as u64;
        unsafe {
            lens_col_sum_f32::launch_unchecked::<R>(
                client,
                capped_cube_count(n_wg),
                CubeDim::new_1d(WG),
                BufferArg::from_raw_parts(buf, batch * t_len * d),
                BufferArg::from_raw_parts(out, n),
                BufferArg::from_raw_parts(ph, 5),
            );
        }
    }
}
