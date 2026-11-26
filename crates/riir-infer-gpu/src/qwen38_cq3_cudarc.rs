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
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

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
        // Scalar-arg shape (the dense module's proven one): ONE scalar per
        // `.arg`, `&i32`/`&f32` push their own bytes. An ARRAY reference
        // would push its 8-byte HOST address — pointer-sized, so the
        // launch's size check passes and the kernel dereferences a stack
        // address on the device (the measured A2v2c illegal-access class).
        let (g_i, smooth_len_i, tokens_i) = (g as i32, smooth_len as i32, tokens as i32);
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
                .arg(&g_i)
                .arg(&smooth_len_i)
                .arg(&inv)
                .arg(&tokens_i)
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
        let (g_i, tokens_i) = (g as i32, tokens as i32);
        // SAFETY: caller contract above.
        unsafe {
            stream
                .launch_builder(&self.quant_x8)
                .arg(xr)
                .arg(x8)
                .arg(s_x)
                .arg(&g_i)
                .arg(&tokens_i)
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
        // Scalar-arg shape (see launch_act_prep) — b rides as i32, so the
        // f32 round trip the old params array needed is gone entirely.
        let (g_i, b_i, tokens_i, groups_i, rows_i) = (
            g as i32,
            b as i32,
            tokens as i32,
            groups as i32,
            rows as i32,
        );
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
                .arg(&g_i)
                .arg(&b_i)
                .arg(&s_c)
                .arg(&tokens_i)
                .arg(&groups_i)
                .arg(&rows_i)
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
    const int g, const int smooth_len, const float inv, const int tokens)
{
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
    const int g, const int tokens)
{
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
    // Broadcast lane 0's reduced max: the down-shuffle ladder leaves the
    // full max ONLY on lane 0 (lane k holds max of [k..31]), so without
    // this every lane quantized its own elements with its own partial
    // scale (the A2v2d read: elem 16 clamped to 127 where the host
    // staged 114 — the global max lived in elems 1..15, visible to
    // lanes 1..15, missed by lane 16's window).
    m = __shfl_sync(0xFFFFFFFFu, m, 0);
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
    const int g, const int b, const float s_c, const int tokens, const int groups_in, const int rows)
{
    const int row = blockIdx.x;
    const int pos = blockIdx.y;
    if (row >= rows || pos >= tokens) return;
    const int d_in = groups_in * g;
    const int tok_base = pos * d_in;

    extern __shared__ float contrib[];

    const unsigned int mask = ((unsigned int)b >= 32u)
        ? 0xFFFFFFFFu
        : ((1u << (unsigned int)b) - 1u);
    // The row's FIRST BIT position: row·d_in·b — the row offset is in
    // BITS (the per-row WORD count mixed into this bit-space shift read
    // row 1+ from the wrong words: in-bounds, so parity-only — the
    // A2v2c read).
    const size_t row_bit0 = (size_t)row * (size_t)(d_in * b);

    for (int grp = 0; grp < groups_in; grp++) {
        int acc = 0;
        for (int m = 0; m < g; m++) {
            const size_t pos_b = row_bit0 + (size_t)(grp * g + m) * (size_t)b;
            const size_t w = pos_b >> 5;
            const unsigned int off = (unsigned int)(pos_b & 31u);
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

// ─────────────────────────────────────────────────────────────────────────────
// Issue 1186 T3-prep (a2) — the cudarc serving handle (the `154159ef`
// CubeCL `Cq3Handle` mirror, cudarc idiom).
//
// One handle per routed tensor: device buffers (words u32 / scales f32 /
// c8 i32 / perm u32 / s1 s2 smooth f32 — scales PRE-EXPANDED host-side,
// the Issue-593 subnormal class never reaches a kernel; smooth rides a
// zeroed d_in dummy when unarmed — a size-0 buffer arg corrupts the
// launch, the measured trap), the kernel set, consts, engagement
// counters (the Issue-1124 T1.5 class), and the grown-once decode /
// cached-p batched scratch. The three-launch chain per dispatch:
// act_prep (rotate) → quant_x8 (stage) → gemv_tiled (never-unpack dot).
// Unarmed = the field is `None` in the dense module's QuantW — the
// incumbent route, byte-identical (G3).
// ─────────────────────────────────────────────────────────────────────────────

/// Decode scratch: rotated x (f32) + staged x8 (i32) + s_x (f32).
struct Cq3DecodeScratch {
    x_rot: CudaSlice<f32>,
    x8: CudaSlice<i32>,
    s_x: CudaSlice<f32>,
}

/// The cudarc CQ3 serving handle (one per routed tensor).
pub struct Cq3HandleCuda {
    kernels: Arc<Cq3CudarcKernels>,
    words: CudaSlice<u32>,
    scales: CudaSlice<f32>,
    c8: CudaSlice<i32>,
    perm: CudaSlice<u32>,
    s1: CudaSlice<f32>,
    s2: CudaSlice<f32>,
    /// The SmoothQuant fold (f32, d_in); a zeroed dummy when unarmed
    /// (`smooth_len == 0` is the only gate the kernel consults).
    smooth: CudaSlice<f32>,
    smooth_len: usize,
    /// Per-group rotation consts.
    g: usize,
    b: u32,
    s_c: f32,
    pub d_out: usize,
    pub d_in: usize,
    /// Engagement counters (act-prep, gemv launches) — the construction
    /// line names the lane, the smoke asserts these move.
    prep_launches: AtomicUsize,
    gemv_launches: AtomicUsize,
    /// Grown once at the decode shape.
    decode_scratch: Mutex<Option<Cq3DecodeScratch>>,
    /// Lazily grown batched scratch: `(p, ...)` — one entry (the league's
    /// tile count is fixed per run), the Metal twin's cached-p shape.
    batch_scratch: Mutex<Option<(usize, Cq3DecodeScratch)>>,
}

impl Cq3HandleCuda {
    /// Build from a packed artifact tensor: the battery's proven host path
    /// (`device_form`, `qwen38_cq3_cudarc_g1`) — unpack → `pack_indices_b32`
    /// → pre-expand f16 scales → i32 codebook → rotation `device_params` →
    /// dummy smooth when unarmed.
    ///
    /// # Errors
    /// Upload failures name the tensor stage loud.
    pub fn from_packed_cuda(
        kernels: Arc<Cq3CudarcKernels>,
        stream: &Arc<CudaStream>,
        t: &crate::qwen38_cq3_serving::PackedCq3Tensor,
    ) -> Result<Self, String> {
        let carrier = t.unpack_indices();
        let words = crate::qwen38_cq3::pack_indices_b32(&carrier, t.b);
        let scales: Vec<f32> = t
            .scales
            .iter()
            .map(|&bits| half::f16::from_bits(bits).to_f32())
            .collect();
        let cb = t.codebook();
        let c8: Vec<i32> = cb.c8.iter().map(|&v| i32::from(v)).collect();
        let rot = t.rotation();
        let (rot_s1, rot_s2, rot_perm) = rot.device_params();
        let rot_perm: Vec<u32> = rot_perm.to_vec();
        let rot_s1: Vec<f32> = rot_s1.to_vec();
        let rot_s2: Vec<f32> = rot_s2.to_vec();
        let g = t.g;
        let groups = t.d_in / g;
        assert!(
            groups <= 1024,
            "cq3 serving: groups per tensor {groups} exceeds the tiled kernel's shared slice (1024) — \
             the recipe pins g=32 (27B max 432); a wider model needs the cap raised"
        );
        // Dummy d_in fold when unarmed (size-0 = the corrupt-launch trap).
        let smooth_host = t.smooth.clone().unwrap_or_else(|| vec![0.0_f32; t.d_in]);
        let words_dev = stream
            .clone_htod(&words)
            .map_err(|e| format!("cq3 upload words: {e}"))?;
        let scales_dev = stream
            .clone_htod(&scales)
            .map_err(|e| format!("cq3 upload scales: {e}"))?;
        let c8_dev = stream
            .clone_htod(&c8)
            .map_err(|e| format!("cq3 upload c8: {e}"))?;
        let perm_dev = stream
            .clone_htod(&rot_perm)
            .map_err(|e| format!("cq3 upload perm: {e}"))?;
        let s1_dev = stream
            .clone_htod(&rot_s1)
            .map_err(|e| format!("cq3 upload s1: {e}"))?;
        let s2_dev = stream
            .clone_htod(&rot_s2)
            .map_err(|e| format!("cq3 upload s2: {e}"))?;
        let smooth_dev = stream
            .clone_htod(&smooth_host)
            .map_err(|e| format!("cq3 upload smooth: {e}"))?;
        let smooth_len = if t.smooth.is_some() { t.d_in } else { 0 };
        eprintln!(
            "[cq3-serving-cuda] armed: {}x{} g={} b={} b_eff={:.3} words={}KiB smooth={} \
             (launch counters on; opt-in cq3_cuda, never production before the T3 GOAT)",
            t.d_out,
            t.d_in,
            g,
            t.b,
            t.bits_per_weight(),
            words.len() * 4 / 1024,
            if smooth_len > 0 { "on" } else { "off" },
        );
        Ok(Self {
            kernels,
            words: words_dev,
            scales: scales_dev,
            c8: c8_dev,
            perm: perm_dev,
            s1: s1_dev,
            s2: s2_dev,
            smooth: smooth_dev,
            smooth_len,
            g,
            b: t.b,
            s_c: cb.s_c,
            d_out: t.d_out,
            d_in: t.d_in,
            prep_launches: AtomicUsize::new(0),
            gemv_launches: AtomicUsize::new(0),
            decode_scratch: Mutex::new(None),
            batch_scratch: Mutex::new(None),
        })
    }

    /// Dims for the dispatch site's shape assertions.
    pub fn dims(&self) -> (usize, usize) {
        (self.d_out, self.d_in)
    }

    /// Launch counters (the engagement-proof readout).
    pub fn launch_counts(&self) -> (usize, usize) {
        (
            self.prep_launches.load(Ordering::Relaxed),
            self.gemv_launches.load(Ordering::Relaxed),
        )
    }

    /// Groups per token (= d_in / g, asserted ≤ 1024 at construction).
    fn groups(&self) -> usize {
        self.d_in / self.g
    }

    /// The three-launch decode chain: rotate → stage → never-unpack dot.
    /// `x` covers d_in f32 (the UNROTATED normed activation); `y` covers
    /// d_out f32 (row-major [1, d_out] — the kernel's out[pos·rows+row]).
    ///
    /// # Errors
    /// Launch failures name the stage loud.
    pub fn gemv_decode(
        &self,
        stream: &CudaStream,
        x: &CudaSlice<f32>,
        y: &CudaSlice<f32>,
    ) -> Result<(), String> {
        let mut slot = self.decode_scratch.lock().expect("cq3 scratch lock");
        let s = match slot.as_ref() {
            Some(s) => s,
            None => {
                let grown = self.grow_scratch(stream, 1)?;
                slot.insert(grown)
            }
        };
        // SAFETY: scratch covers 1·d_in / 1·groups; x covers d_in; y
        // covers d_out; buffers are handle-carried (sizes fixed at
        // construction).
        unsafe {
            self.kernels.launch_act_prep(
                stream,
                x,
                &self.smooth,
                &self.perm,
                &self.s1,
                &self.s2,
                &s.x_rot,
                self.g,
                self.smooth_len,
                1,
                self.groups(),
            )
        }
        .map_err(|e| format!("cq3 act_prep: {e}"))?;
        // SAFETY: covers 1·d_in / 1·groups.
        unsafe {
            self.kernels
                .launch_quant_x8(stream, &s.x_rot, &s.x8, &s.s_x, self.g, 1, self.groups())
        }
        .map_err(|e| format!("cq3 quant_x8: {e}"))?;
        // SAFETY: covers d_out·wpr / d_out·groups / 1<<b / 1·d_in / 1·groups /
        // 1·d_out; shared = groups·4 B.
        unsafe {
            self.kernels.launch_gemv_tiled(
                stream,
                &self.words,
                &self.scales,
                &self.c8,
                &s.x8,
                &s.s_x,
                y,
                self.g,
                self.b,
                self.s_c,
                1,
                self.groups(),
                self.d_out,
            )
        }
        .map_err(|e| format!("cq3 gemv_tiled: {e}"))?;
        self.prep_launches.fetch_add(1, Ordering::Relaxed);
        self.gemv_launches.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// The batched (prefill option-(i)) chain: `p` tokens, `x` covers
    /// p·d_in, `y` covers p·d_out row-major [p, d_out]. Scratch cached per
    /// p (one entry — the league's tile count is fixed per run); a NEW p
    /// grows the cache (the league runs one tile per run, disclosed).
    ///
    /// # Errors
    /// Launch / grow failures name the stage loud.
    pub fn gemv_batched(
        &self,
        stream: &CudaStream,
        x: &CudaSlice<f32>,
        y: &CudaSlice<f32>,
        p: usize,
    ) -> Result<(), String> {
        if p == 1 {
            return self.gemv_decode(stream, x, y);
        }
        let mut slot = self.batch_scratch.lock().expect("cq3 scratch lock");
        let s = match slot.as_ref() {
            Some((cached_p, s)) if *cached_p == p => s,
            _ => {
                let grown = self.grow_scratch(stream, p)?;
                slot.insert((p, grown))
            }
        };
        // SAFETY: mirror of [`Self::gemv_decode`] at p tokens — the
        // scratch covers p·d_in / p·groups; x covers p·d_in; y covers
        // p·d_out row-major.
        unsafe {
            self.kernels.launch_act_prep(
                stream,
                x,
                &self.smooth,
                &self.perm,
                &self.s1,
                &self.s2,
                &s.x_rot,
                self.g,
                self.smooth_len,
                p,
                self.groups(),
            )
        }
        .map_err(|e| format!("cq3 act_prep(p={p}): {e}"))?;
        // SAFETY: covers p·d_in / p·groups.
        unsafe {
            self.kernels
                .launch_quant_x8(stream, &s.x_rot, &s.x8, &s.s_x, self.g, p, self.groups())
        }
        .map_err(|e| format!("cq3 quant_x8(p={p}): {e}"))?;
        // SAFETY: covers p·d_out row-major; shared = groups·4 B.
        unsafe {
            self.kernels.launch_gemv_tiled(
                stream,
                &self.words,
                &self.scales,
                &self.c8,
                &s.x8,
                &s.s_x,
                y,
                self.g,
                self.b,
                self.s_c,
                p,
                self.groups(),
                self.d_out,
            )
        }
        .map_err(|e| format!("cq3 gemv_tiled(p={p}): {e}"))?;
        self.prep_launches.fetch_add(1, Ordering::Relaxed);
        self.gemv_launches.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Grow a p-token scratch (rotated f32 + staged i32 + scales f32).
    fn grow_scratch(&self, stream: &CudaStream, p: usize) -> Result<Cq3DecodeScratch, String> {
        let groups = self.groups();
        let x_rot = stream
            .alloc_zeros::<f32>(p * self.d_in)
            .map_err(|e| format!("cq3 scratch x_rot: {e}"))?;
        let x8 = stream
            .alloc_zeros::<i32>(p * self.d_in)
            .map_err(|e| format!("cq3 scratch x8: {e}"))?;
        let s_x = stream
            .alloc_zeros::<f32>(p * groups)
            .map_err(|e| format!("cq3 scratch s_x: {e}"))?;
        Ok(Cq3DecodeScratch { x_rot, x8, s_x })
    }
}
