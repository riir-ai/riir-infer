//! Issue 1186 item (d) — the CQ3 cudarc twin: the raw-CUDA kernel mirror of
//! the CubeCL serving arm (`qwen38_cq3_gpu`, landed `154159ef`), living in
//! THIS repo beside `qwen38_dense_cudarc` (the module re-exported by
//! riir-ai). The 4090 lane's decode/GEMV path for the league cell.
//!
//! # Kernel set (mirrors the CubeCL module's contract)
//!
//! - `cq3_act_prep` — SmoothQuant divide → P gather → S₁ signs → the
//!   normalized FWHT butterfly → S₂ signs, one WORKGROUP per (group,
//!   token), thread 0 replays the host butterfly's exact serial op order
//!   (the shared-scratch shape the CubeCL kernel pins; the host order is
//!   `qwen38_cq3::Rotation::apply_block` verbatim).
//! - `cq3_quant_x8` — per-group absmax int8 staging of the ROTATED
//!   activation (the `qwen38_quant_x_q8` op order at g-sized groups;
//!   absmax/127, round-and-clamp ±127, zero scale pins x8 to 0). Staged
//!   ONCE per (token, tensor) — it does not depend on the output row.
//! - `cq3_gemv_tiled` — the decode-shape exactness witness: one thread per
//!   (row, token); per-group int32 dot over the b-bit packed indices +
//!   pre-rounded int8 codebook (`c8`); the de-scale rounds through shared
//!   memory (the contraction-proof ×1.0 spelling — `fma(y,1,c) =
//!   round(y+c)`, exactly the host's two-op rounding; nvrtc's default
//!   contraction fuses the bare `y += product` the same way Metal's does,
//!   the measured 1–2 ulp drift) and the final fold is the host's
//!   sequential group order.
//!
//! The PLANE (perf-tier) kernel is the named follow-up — decode first
//! (the league's decode row + the G1 exactness witness), prefill rides
//! the batched launch (p tokens in one grid, the option-(i) posture).
//!
//! # Numerics law (the same tiered bar the CubeCL arm measured)
//!
//! The TILED kernel is the `to_bits` parity witness vs
//! `PackedCq3Tensor::gemv_i8_sdot` on the DEVICE-vs-DEVICE side and the
//! ≤ 1e-5 abs witness vs the host on staged inputs (nvrtc compiles with
//! default contraction too — the shared-memory round trip is what makes
//! the de-scale contract-proof, identical to the CubeCL finding; the f32
//! max/round/clamp staging ops are exact). The int32 dot itself is EXACT
//! everywhere — the scalar per-element i32 chain has no fp freedom; only
//! the final f32 fold has spelling freedom, and the shared-memory store
//! pins it.
//!
//! # Device form
//!
//! Weights ride the u32-word packed form (`pack_indices_b32` —
//! LSB-first, `ceil(d_in·b/32)` words per row, byte-identical to the
//! CubeCL handle's upload), scales pre-expanded f32 host-side (the
//! `f16::from_bits` conversion — the Issue-593 class never reaches a
//! kernel), `c8` as i32.
//!
//! Correctness bar: the G1 battery (`qwen38_cq3_cudarc_g1`, 4090-gated)
//! pins tiled `to_bits` vs the HOST reference + the engagement counters.
//! Opt-in `cq3_cuda` (implies `cq3` + `ternary_gemv_cuda_raw`); never in
//! `production` before the T3 GOAT.
#![cfg(all(feature = "cq3_cuda", not(target_os = "macos")))]

use std::sync::Arc;

use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream};

/// Words per row on the device form: `ceil(d_in·b/32)` (`d_in` is a
/// multiple of g ≥ 16, b ≤ 8 — exact).
#[must_use]
pub fn words_row_len(d_in: usize, b: u32) -> usize {
    (d_in as u64 * u64::from(b)).div_ceil(32) as usize
}

// The CUDA source lives in the CQ3_CUDA_SRC string below (the kernels
// are documented in the string's comments; keeping one copy — the wire —
// avoids a drifting twin).

/// The compiled CQ3 kernel set (nvrtc sm_89). One per context; the
/// launchers are the module's three pub methods.
pub struct Cq3CudarcKernels {
    act_prep: CudaFunction,
    quant_x8: CudaFunction,
    gemv_tiled: CudaFunction,
    /// The functions borrow the module's code image — retained so the
    /// functions stay valid for the struct's whole life. `load_module`
    /// hands back the `Arc` (cudarc 0.19) and the functions hold clones
    /// of it, so this field is retention only.
    _module: Arc<CudaModule>,
}

impl Cq3CudarcKernels {
    /// Compile + load (nvrtc, sm_89 — the 4090's arch, the same option
    /// string the dense module pins).
    ///
    /// # Errors
    /// nvrtc compile / module load / function lookup — named loud.
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, String> {
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            CQ3_CUDA_SRC,
            cudarc::nvrtc::CompileOptions {
                arch: Some("sm_89"),
                ..Default::default()
            },
        )
        .map_err(|e| format!("nvrtc: {e}"))?;
        let module = ctx.load_module(ptx).map_err(|e| e.to_string())?;
        let f = |name: &str| {
            module
                .load_function(name)
                .map_err(|e| format!("{name}: {e}"))
        };
        Ok(Self {
            act_prep: f("cq3_act_prep")?,
            quant_x8: f("cq3_quant_x8")?,
            gemv_tiled: f("cq3_gemv_tiled")?,
            _module: module,
        })
    }

    /// Launch the activation prep (rotation). One workgroup per (group,
    /// token); shared scratch = `g` f32.
    ///
    /// # Safety
    ///
    /// Caller guarantees `x`/`out` cover `tokens·d_in` f32, `perm`/`s1`/
    /// `s2` cover `g`, and `smooth` covers `smooth_len > 0 ? tokens·d_in`
    /// f32 (else it is a non-empty dummy — a size-0 buffer arg corrupts
    /// the launch, the measured trap).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_act_prep(
        &self,
        stream: &CudaStream,
        x: &CudaSlice<f32>,
        smooth: &CudaSlice<f32>,
        perm: &CudaSlice<u32>,
        s1: &CudaSlice<f32>,
        s2: &CudaSlice<f32>,
        out: &CudaSlice<f32>,
        g: usize,
        smooth_len: usize,
        tokens: usize,
        groups: usize,
    ) -> Result<(), String> {
        let inv = 1.0f32 / (g as f32).sqrt();
        // Fixed-size array, never a Vec: `[T; N]` is the DeviceRepr shape
        // the launcher accepts (a Vec does not implement it) and the hot
        // path stays alloc-free.
        let params: [f32; 4] = [g as f32, smooth_len as f32, inv, tokens as f32];
        let shared = (g * std::mem::size_of::<f32>()) as u32;
        // SAFETY: caller contract above.
        unsafe {
            stream
                .launch_builder(&self.act_prep)
                .arg(x)
                .arg(smooth)
                .arg(perm)
                .arg(s1)
                .arg(s2)
                .arg(out)
                .arg(&params)
                .launch(LaunchConfig {
                    grid_dim: (groups as u32, tokens as u32, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: shared,
                })
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Launch the per-group absmax int8 staging.
    ///
    /// # Safety
    ///
    /// Caller guarantees `xr`/`x8` cover `tokens·d_in`, `s_x` covers
    /// `tokens·groups`.
    pub unsafe fn launch_quant_x8(
        &self,
        stream: &CudaStream,
        xr: &CudaSlice<f32>,
        x8: &CudaSlice<i32>,
        s_x: &CudaSlice<f32>,
        g: usize,
        tokens: usize,
        groups: usize,
    ) -> Result<(), String> {
        let params: [f32; 2] = [g as f32, tokens as f32];
        // SAFETY: caller contract above.
        unsafe {
            stream
                .launch_builder(&self.quant_x8)
                .arg(xr)
                .arg(x8)
                .arg(s_x)
                .arg(&params)
                .launch(LaunchConfig {
                    grid_dim: (groups as u32, tokens as u32, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Launch the tiled GEMV (one thread per (row, token)).
    ///
    /// # Safety
    ///
    /// Caller guarantees `words` covers `rows·words_row_len(d_in, b)`,
    /// `scales` covers `rows·groups`, `c8` covers `1<<b`, `x8` covers
    /// `tokens·d_in`, `s_x` covers `tokens·groups`, `out` covers
    /// `tokens·rows`, and the shared budget is `groups·4` bytes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn launch_gemv_tiled(
        &self,
        stream: &CudaStream,
        words: &CudaSlice<u32>,
        scales: &CudaSlice<f32>,
        c8: &CudaSlice<i32>,
        x8: &CudaSlice<i32>,
        s_x: &CudaSlice<f32>,
        out: &CudaSlice<f32>,
        g: usize,
        b: u32,
        s_c: f32,
        tokens: usize,
        groups: usize,
        rows: usize,
    ) -> Result<(), String> {
        // `f32::from(u32)` does not exist — a widening `as` cast, exact
        // for the b ≤ 8 domain.
        let params: [f32; 6] = [
            g as f32,
            b as f32,
            s_c,
            tokens as f32,
            groups as f32,
            rows as f32,
        ];
        let shared = (groups * std::mem::size_of::<f32>()) as u32;
        // SAFETY: caller contract above.
        unsafe {
            stream
                .launch_builder(&self.gemv_tiled)
                .arg(words)
                .arg(scales)
                .arg(c8)
                .arg(x8)
                .arg(s_x)
                .arg(out)
                .arg(&params)
                .launch(LaunchConfig {
                    grid_dim: (rows as u32, tokens as u32, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: shared,
                })
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

/// The CUDA source (the three kernels above, verbatim).
pub const CQ3_CUDA_SRC: &str = r#"
inline __device__ void cq3_fwht_block(float* s, const int g) {
    for (int h = 1; h < g; h <<= 1) {
        for (int c = 0; c < g; c += (h << 1)) {
            for (int i = 0; i < h; i++) {
                const float a = s[c + i];
                const float b = s[c + i + h];
                s[c + i] = a + b;
                s[c + i + h] = a - b;
            }
        }
    }
}

extern "C" __global__ void cq3_act_prep(
    const float* __restrict__ x,
    const float* __restrict__ smooth,
    const unsigned int* __restrict__ perm,
    const float* __restrict__ s1,
    const float* __restrict__ s2,
    float* __restrict__ out,
    const float* __restrict__ params)
{
    const int g = (int)params[0];
    const int smooth_len = (int)params[1];
    const float inv = params[2];
    const int tokens = (int)params[3];
    const int grp = blockIdx.x;
    const int pos = blockIdx.y;
    const int d_in = gridDim.x * g;
    if (grp >= gridDim.x || pos >= tokens) return;
    const int base = pos * d_in + grp * g;

    extern __shared__ float scratch[];

    if (threadIdx.x == 0) {
        for (int m = 0; m < g; m++) {
            const int src_slot = (int)perm[m];
            const float src = x[base + src_slot];
            float v = src;
            if (smooth_len > 0) v = src / smooth[base + src_slot];
            v *= s1[m];
            scratch[m] = v;
        }
        cq3_fwht_block(scratch, g);
        for (int m = 0; m < g; m++) {
            out[base + m] = scratch[m] * inv * s2[m];
        }
    }
}

extern "C" __global__ void cq3_quant_x8(
    const float* __restrict__ xr,
    int* __restrict__ x8,
    float* __restrict__ s_x,
    const float* __restrict__ params)
{
    const int g = (int)params[0];
    const int tokens = (int)params[1];
    const int grp = blockIdx.x;
    const int pos = blockIdx.y;
    const int groups = gridDim.x;
    const int d_in = groups * g;
    if (grp >= groups || pos >= tokens) return;
    const int base = pos * d_in + grp * g;

    const int lane = threadIdx.x & 31;
    float m = 0.f;
    for (int i = lane; i < g; i += 32) {
        m = fmaxf(m, fabsf(xr[base + i]));
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        m = fmaxf(m, __shfl_down_sync(0xFFFFFFFFu, m, off));
    }
    const float s = m / 127.0f;
    if (lane == 0) s_x[pos * groups + grp] = s;
    if (s > 0.f) {
        for (int i = lane; i < g; i += 32) {
            const float q = roundf(xr[base + i] / s);
            x8[base + i] = (int)fminf(127.f, fmaxf(-127.f, q));
        }
    } else {
        for (int i = lane; i < g; i += 32) {
            x8[base + i] = 0;
        }
    }
}

extern "C" __global__ void cq3_gemv_tiled(
    const unsigned int* __restrict__ words,
    const float* __restrict__ scales,
    const int* __restrict__ c8,
    const int* __restrict__ x8,
    const float* __restrict__ s_x,
    float* __restrict__ out,
    const float* __restrict__ params)
{
    const int g = (int)params[0];
    const int b = (int)params[1];
    const float s_c = params[2];
    const int tokens = (int)params[3];
    const int groups_in = (int)params[4];
    const int rows = (int)params[5];
    const int row = blockIdx.x;
    const int pos = blockIdx.y;
    if (row >= rows || pos >= tokens) return;
    const int d_in = groups_in * g;
    const int tok_base = pos * d_in;

    extern __shared__ float contrib[];

    const unsigned int mask = ((unsigned int)b >= 32u)
        ? 0xFFFFFFFFu
        : ((1u << (unsigned int)b) - 1u);
    const size_t row_word0 = (size_t)row * (size_t)((d_in * b + 31) / 32);

    for (int grp = 0; grp < groups_in; grp++) {
        int acc = 0;
        for (int m = 0; m < g; m++) {
            const size_t pos_w = row_word0 + (size_t)(grp * g + m) * (size_t)b;
            const size_t w = pos_w >> 5;
            const unsigned int off = (unsigned int)(pos_w & 31u);
            unsigned int pv = words[w] >> off;
            if (off + (unsigned int)b > 32u) {
                pv |= words[w + 1] << (32u - off);
            }
            const int idx = (int)(pv & mask);
            acc += c8[idx] * x8[tok_base + grp * g + m];
        }
        const float gf = scales[(size_t)row * groups_in + grp];
        contrib[grp] = (float)acc * s_c * s_x[pos * groups_in + grp] * gf;
    }
    __syncthreads();
    float total = 0.f;
    for (int grp = 0; grp < groups_in; grp++) {
        total = total * 1.0f + contrib[grp];
    }
    out[(size_t)pos * rows + row] = total;
}
"#;
