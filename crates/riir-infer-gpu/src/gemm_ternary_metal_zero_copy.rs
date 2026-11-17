//! Issue 663 T5 — Zero-copy raw-Metal ternary GEMM via shared CubeCL device.
//!
//! This is the **third attempt** at capturing the 3.72× kernel-level matmul2d
//! speedup (Bench 645) end-to-end. The first two were refuted by host
//! round-trip / staging-copy overhead:
//!
//! - Issue 656 (0.87×): raw Metal kernel on a **separate** `Device::system_default()`
//!   — each of ~448 prefill projections forced 2 blocking syncs (CubeCL read →
//!   Metal upload → Metal run → Metal read → CubeCL write).
//! - Issue 657 (0.95×): same kernel dispatched via wgpu MSL passthrough on the
//!   shared CubeCL queue — per-dispatch staging copies + bind group rebuild
//!   negated the kernel gain.
//!
//! This module eliminates **both** overhead sources by using the raw Metal
//! dispatch path validated in the T4 test (`bench_663_zero_copy_interop.rs`):
//!
//! - **Same device + queue as CubeCL** — extracted via `wgpu::Device::as_hal::<Metal>`
//!   + `raw_device()` / `as_raw()` (no cross-queue sync, no separate context).
//! - **Zero-copy weight cache** — reuses the CubeCL-managed weight buffers
//!   directly (extracts raw `id<MTLBuffer>` via `get_resource` →
//!   `as_hal::<Metal>` → `raw_handle()`). No re-upload, no staging copy.
//! - **Zero-copy input/output** — each dispatch reads + writes the same GPU
//!   memory CubeCL manages. No staging buffers, no host round-trip.
//! - **Raw `metal::ComputePipelineState` dispatch** — no wgpu compute pass,
//!   no bind group, no command encoder allocations per dispatch.
//!
//! # Architecture
//!
//! ```text
//! Boot (once per weight matrix):
//!   CubeCL weight handles → extract raw id<MTLBuffer> → cache in ZeroCopyWeightCache
//!
//! Per dispatch (zero copy, zero round-trip):
//!   1. Extract raw id<MTLBuffer> for input + output CubeCL handles
//!   2. queue_ref.new_command_buffer() on the shared Metal queue
//!   3. encoder.set_buffer(0..4, raw handles) + set_bytes(5..9, scalars)
//!   4. encoder.dispatch_threads(grid, threadgroup)
//!   5. cmd_buffer.commit() — ordering with CubeCL is automatic (same queue)
//! ```
//!
//! # Feature gate
//!
//! `metal_tensor_gemm` + macOS-only. Requires the vendored wgpu-hal fork that
//! exposes `metal::Buffer::raw_handle()` (Issue 663 T1.5).

#![cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use metal::foreign_types::ForeignTypeRef;
use metal::{
    Buffer, BufferRef, CommandQueueRef, CompileOptions, ComputePipelineState, DeviceRef,
    MTLResourceOptions, MTLResourceUsage, MTLSize,
};
use wgpu::{Device as WgpuDevice, Queue as WgpuQueue};

use cubecl::client::ComputeClient;
use cubecl::server::Handle;

use crate::cubecl_runtime::ActiveRuntime;
use crate::gemv_ternary_cubecl::TernaryHandle;

/// The raw MSL source — identical to Issue 656's matmul2d kernel (scalar
/// buffer bindings 5-9, not the packed-struct wgpu variant). Dispatched
/// directly via `metal::ComputePipelineState`.
const MSL_SOURCE: &str = include_str!("gemm_ternary_metal_tensor.metal");

/// Issue 1225 — the f16-B twin's two anchored patches (the Bench-1223 G5
/// lever, now canonical HERE; the price-cell test consumes this const).
/// The kernel signature's input pointer and the single downcast site —
/// `tensor(input_rw, …)` infers its element type from the pointer, so the
/// tensor op takes a half B with no further edits; the accumulator stays
/// float. LOSSY by construction (the input rounds f32→f16, ≤ 2^-11 relative
/// per element); the tolerance class + the Issue-1225 per-family retention
/// decider are the quality gates. If mpp::tensor_ops::matmul2d rejects
/// mixed half/half→float operands the twin compile FAILS LOUD at
/// [`MetalTensorZeroCopyGemm::new_f16b`] and the route demotes counted —
/// never a silently-mis-patched dispatch.
///
/// ⛔ Occurrence counts are asserted at twin compile ([`apply_patches_checked`])
/// — a serving-source drift fails LOUD before any dispatch, the same guard
/// the Bench-1223 instrument ran under.
pub const F16_BODY_PATCHES: &[(&str, &str, usize)] = &[
    (
        "device const float* input_batch      [[buffer(3)]],   // X[P × K] row-major",
        "device const half*   input_batch      [[buffer(3)]],   // X[P × K] row-major (i1225 f16-B twin)",
        1,
    ),
    (
        "device float* input_rw = (device float*)(input_batch);",
        "device half* input_rw = (device half*)(input_batch);",
        1,
    ),
];

/// The f16-B cast kernel (appended to the twin's MSL library): one
/// f32→f16 elementwise pass over the input, dispatched as the FIRST encoder
/// of the twin's command buffer so the gemm encoder reads the rounded copy
/// in the same commit (encoder order = guaranteed visibility within one
/// command buffer). This is the per-launch cast fallback — the bounded
/// worst case (measured 0.15–0.19 ms/launch, Bench 1223); a
/// producer-writes-f16 serving shape is the tax-free upper bound.
#[cfg(feature = "prefill_f16b")]
const CAST_F16_MSL: &str = r#"
kernel void cast_f32_f16_zc(device const float* src [[buffer(0)]],
                            device half* dst [[buffer(1)]],
                            constant uint& n [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid < n) { dst[gid] = half(src[gid]); }
}
"#;

/// Apply the twin patches with the occurrence-count drift guard: a find that
/// matches a different number of times than pinned means the serving .metal
/// moved — refuse LOUD, never emit a silently-mis-patched source.
#[cfg(feature = "prefill_f16b")]
fn apply_patches_checked(source: &str, patches: &[(&str, &str, usize)]) -> Result<String, String> {
    let mut out = source.to_string();
    for (find, replace, want) in patches {
        let got = out.matches(find).count();
        if got != *want {
            return Err(format!(
                "i1225 twin patch drift: find occurs {got}x, pin says {want}x — \
                 the serving .metal moved; re-anchor F16_BODY_PATCHES"
            ));
        }
        out = out.replacen(find, replace, *want);
    }
    Ok(out)
}

/// Cached raw weight buffer info for one ternary weight matrix.
///
/// Stores the CubeCL handles + lazily-extracted raw buffer info. The raw
/// pointers are re-extracted on each dispatch (not cached across dispatches)
/// because CubeCL's memory pool can reallocate. However, the CubeCL handles
/// themselves are stable (reference-counted) and cheap to clone.
#[derive(Clone)]
pub struct ZeroCopyWeightCache {
    /// CubeCL handle for `pos_bits_u32`.
    pos: Handle,
    /// CubeCL handle for `neg_bits_u32`.
    neg: Handle,
    /// CubeCL handle for `group_scale_f32`.
    scale: Handle,
}

// SAFETY: `Handle` is `Send + Sync` (it wraps an Arc to CubeCL's server-side
// resource management). The cache can be shared across threads.
unsafe impl Send for ZeroCopyWeightCache {}
unsafe impl Sync for ZeroCopyWeightCache {}

/// Zero-copy raw-Metal ternary GEMM dispatch context (Issue 663 T5).
///
/// Created once at model load time. Holds the raw `id<MTLDevice>` +
/// `id<MTLCommandQueue>` extracted from the shared CubeCL wgpu device + queue,
/// plus the compiled `metal::tensor` matmul2d `ComputePipelineState`.
///
/// # Why this beats Issue 657
///
/// Issue 657 (`MetalTensorWgpuGemm`) used wgpu's compute pass API with
/// staging copies to work around wgpu's usage-scope validation. This module
/// skips wgpu entirely for the dispatch — it goes straight to Metal's
/// `MTLComputeCommandEncoder`, which has no usage-scope validation and can
/// bind the same buffer as both read + write within one dispatch (with
/// `use_resource(..., Read | Write)`).
///
/// # Why this beats Issue 656
///
/// Issue 656 (`MetalTensorGemm`) used a separate `Device::system_default()`
/// Metal context, forcing host round-trips for data exchange with CubeCL.
/// This module shares CubeCL's device + queue — the buffers ARE the same GPU
/// memory, no round-trip is possible.
pub struct MetalTensorZeroCopyGemm {
    /// Raw `id<MTLDevice>` extracted from the shared CubeCL wgpu device.
    /// Borrowed for the lifetime of the `WgpuDevice` Arc (held by `CubeCLContext`).
    device: *mut metal::MTLDevice,
    /// Raw `id<MTLCommandQueue>` extracted from the shared CubeCL wgpu queue.
    /// Same queue CubeCL submits to — ordering is automatic.
    queue: *mut metal::MTLCommandQueue,
    /// The compiled matmul2d pipeline. Owns the library + function internally.
    pipeline: ComputePipelineState,
    /// Cached dedicated output buffers, keyed by byte size. These are
    /// REQUIRED because CubeCL sub-allocates all handles within one pool
    /// buffer — when the output shares the same `id<MTLBuffer>` as the
    /// weights/input, Metal's hazard tracking silently fails the kernel's
    /// cooperative-tensor store. The dedicated output buffer breaks the
    /// aliasing; after the kernel writes, we copy back to the CubeCL handle.
    /// This is ONE GPU→GPU copy (not a host round-trip).
    output_cache: Mutex<HashMap<u64, Buffer>>,
    /// Issue 1146: CubeCL output ranges wgpu has already marked initialized,
    /// per pool buffer, as SORTED, MERGED `[start, end)` intervals — see
    /// [`Self::ensure_wgpu_initialized`]. Containment (not exact-key match):
    /// a new handle whose range lies inside an initialized region costs
    /// nothing, so a drifting pool layout (long contexts) does not re-fill.
    /// Holding the `wgpu::Buffer` clone keeps its identity from being reused
    /// by a later allocation while it is remembered here.
    wgpu_initialized: Mutex<HashMap<wgpu::Buffer, Vec<(u64, u64)>>>,
    /// Issue 1146: first-use fills issued (`RIIR_ZC_FILL_TRACE=1` prints).
    init_fills: std::sync::atomic::AtomicU64,
    /// Issue 663 T5 profiling: total dispatch count (for overhead analysis).
    dispatch_count: std::sync::atomic::AtomicU64,
    /// Issue 1122 gate 2: the tile geometry this pipeline was compiled at.
    /// The launch grid, threadgroup width, and threadgroup-memory length all
    /// derive from it — a dispatch that disagrees with the compiled kernel
    /// reads stale tiles (the constants here MUST mirror the `SWEEP_*`
    /// defines the MSL was compiled with).
    config: crate::gemm_ternary_metal_tensor::MetalTileConfig,
    /// Whether THIS pipeline was compiled with the double-buffer staging
    /// variant — captured at construction so a launch can never disagree
    /// with its own smem need (the global pin may flip after this context
    /// exists; the compiled variant cannot).
    double_buffered: bool,
    /// The most recent committed command buffer, retained for
    /// [`Self::wait_last`](Self::wait_last). The raw dispatches bypass
    /// wgpu's usage tracking, so a CubeCL-side read of the output can
    /// otherwise RACE the pending raw work (raw-Metal submissions are
    /// invisible to wgpu's hazard tracking) — instrument-grade sync for
    /// tests and timing cells. One retained buffer at a time; replacing it
    /// releases the previous.
    last_cmd_buffer: std::sync::Mutex<Option<metal::CommandBuffer>>,
    /// Hold the wgpu device + queue Arcs alive — the raw Metal handles point
    /// into their internals. Dropping these would invalidate `device` + `queue`.
    _wgpu_device: Arc<WgpuDevice>,
    _wgpu_queue: Arc<WgpuQueue>,
    /// Issue 1225: the f16-B twin state (the patched pipeline replaces
    /// `pipeline`; the cast pipeline + the dedicated f16 input cache live
    /// here). `None` unless constructed via [`Self::new_f16b`] — and `None`
    /// in the placebo posture (the control arm dispatches through the
    /// unpatched [`Self::dispatch`] path).
    #[cfg(feature = "prefill_f16b")]
    f16b: Option<F16bState>,
}

/// Issue 1225: the f16-B twin's extra state (cast pipeline + the dedicated
/// f16 input buffers, keyed by byte length — same shape as `output_cache`).
#[cfg(feature = "prefill_f16b")]
struct F16bState {
    cast_pipeline: ComputePipelineState,
    input_cache: Mutex<HashMap<u64, Buffer>>,
}

// SAFETY: The raw `id` pointers are Objective-C object pointers that can be
// shared across threads. Metal's command queue submission is thread-safe.
// The `ComputePipelineState` (metal-rs) is `Send + Sync` (it wraps an `id`
// via foreign-types). The wgpu Arcs are `Send + Sync`.
unsafe impl Send for MetalTensorZeroCopyGemm {}
unsafe impl Sync for MetalTensorZeroCopyGemm {}

impl MetalTensorZeroCopyGemm {
    /// Create a new zero-copy GEMM context on the shared CubeCL Metal device.
    ///
    /// Takes the wgpu `Device` + `Queue` Arcs from `CubeCLContext`. Extracts
    /// the raw Metal handles (via the T4 pattern), compiles the matmul2d
    /// kernel, and returns a dispatch context. Returns `Err` if:
    /// - wgpu isn't using the Metal backend (non-macOS without the feature)
    /// - MSL compilation fails (missing `metal::tensor` support — Metal 3.1+)
    pub fn new(wgpu_device: Arc<WgpuDevice>, wgpu_queue: Arc<WgpuQueue>) -> Result<Self, String> {
        // Issue 1122 gate 2: compile at the production-geometry selector (the
        // explicit setter or the `RIIR_METAL_TENSOR_GEOM` env arm), so an e2e
        // cell can seat the Bench-1113 swept winner without a code change.
        Self::new_with_config(
            crate::gemm_ternary_metal_tensor::production_geometry(),
            wgpu_device,
            wgpu_queue,
        )
    }

    /// Compile the zero-copy context at an explicit [`MetalTileConfig`] (the
    /// calibration-sweep entry point; [`new`](Self::new) is this at the
    /// production-geometry selector).
    pub fn new_with_config(
        config: crate::gemm_ternary_metal_tensor::MetalTileConfig,
        wgpu_device: Arc<WgpuDevice>,
        wgpu_queue: Arc<WgpuQueue>,
    ) -> Result<Self, String> {
        // Extract the raw Metal device + queue (T4 pattern).
        let device_ptr: *mut metal::MTLDevice = unsafe {
            let guard = wgpu_device.as_hal::<wgpu::hal::api::Metal>();
            let dev = guard
                .as_ref()
                .ok_or("wgpu device is not Metal — zero-copy GEMM requires Metal backend")?;
            let proto = dev.raw_device(); // &Retained<ProtocolObject<dyn MTLDevice>>
            (&**proto) as *const _ as *mut metal::MTLDevice
        };

        let queue_ptr: *mut metal::MTLCommandQueue = unsafe {
            let guard = wgpu_queue.as_hal::<wgpu::hal::api::Metal>();
            let q = guard
                .as_ref()
                .ok_or("wgpu queue is not Metal — zero-copy GEMM requires Metal backend")?;
            let proto = q.as_raw(); // &ProtocolObject<dyn MTLCommandQueue>
            proto as *const _ as *mut metal::MTLCommandQueue
        };

        // Compile the matmul2d kernel on this device — at the config's
        // `SWEEP_*` defines (the same guard block `MetalTensorGemm` uses; a
        // bare compile rides the kernel's built-in K32 defaults).
        let device_ref: &DeviceRef = unsafe { DeviceRef::from_ptr(device_ptr) };
        let options = CompileOptions::new();
        let source = crate::gemm_ternary_metal_tensor::compose_msl_source(config, MSL_SOURCE);
        let library = device_ref
            .new_library_with_source(source.as_str(), &options)
            .map_err(|e| format!("MSL compilation failed ({config:?}): {e}"))?;
        let function = library
            .get_function("gemm_ternary_tensor", None)
            .map_err(|e| format!("Function 'gemm_ternary_tensor' not found: {e}"))?;
        let pipeline = device_ref
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("Pipeline creation failed: {e}"))?;

        // Drop the library — the pipeline holds its own reference.
        drop(library);

        Ok(Self {
            device: device_ptr,
            queue: queue_ptr,
            pipeline,
            output_cache: Mutex::new(HashMap::new()),
            wgpu_initialized: Mutex::new(HashMap::new()),
            init_fills: std::sync::atomic::AtomicU64::new(0),
            dispatch_count: std::sync::atomic::AtomicU64::new(0),
            config,
            double_buffered: crate::gemm_ternary_metal_tensor::production_double_buffer(),
            last_cmd_buffer: std::sync::Mutex::new(None),
            _wgpu_device: wgpu_device,
            _wgpu_queue: wgpu_queue,
            #[cfg(feature = "prefill_f16b")]
            f16b: None,
        })
    }

    /// Total dispatches since creation (profiling).
    pub fn dispatch_count(&self) -> u64 {
        self.dispatch_count
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Cache CubeCL weight handles for zero-copy dispatch.
    ///
    /// Called once per weight matrix at model load time. Stores references to
    /// the CubeCL weight handles — the raw `id<MTLBuffer>` pointers are
    /// extracted fresh on every dispatch (not cached) because CubeCL's memory
    /// pool can reallocate/defragment between dispatches, invalidating cached
    /// raw pointers.
    pub fn cache_weights(
        &self,
        _client: &ComputeClient<ActiveRuntime>,
        w: &TernaryHandle,
    ) -> Result<ZeroCopyWeightCache, String> {
        Ok(ZeroCopyWeightCache {
            pos: w.pos_bits_u32.clone(),
            neg: w.neg_bits_u32.clone(),
            scale: w.group_scale_f32.clone(),
        })
    }

    /// Extract `(raw id<MTLBuffer>, offset)` from a CubeCL handle.
    ///
    /// Returns a raw pointer into the CubeCL-managed GPU memory. The pointer
    /// is valid for as long as the CubeCL handle is alive.
    pub(crate) fn extract_raw_buffer(
        client: &ComputeClient<ActiveRuntime>,
        handle: &Handle,
    ) -> Result<(*mut metal::MTLBuffer, u64), String> {
        // cuda_backend twin: the CubeCL resource is CUDA-shaped (no wgpu
        // buffer to extract a Metal handle from) — report unavailable
        // instead of failing to compile. Callers treat Err as "path
        // disabled" and fall through.
        #[cfg(all(feature = "cuda_backend", not(target_os = "macos")))]
        {
            let _ = (client, handle);
            Err(
                "zero-copy Metal extraction unavailable: cuda_backend replaces the wgpu CubeCL runtime"
                    .to_string(),
            )
        }
        #[cfg(any(not(feature = "cuda_backend"), target_os = "macos"))]
        {
            let managed = client
                .get_resource(handle.clone())
                .map_err(|e| format!("get_resource failed: {e:?}"))?;
            let resource = managed.resource();
            let offset = resource.offset;
            let wgpu_buffer = &resource.buffer;

            // SAFETY: `as_hal` is unsafe because it exposes the raw hal buffer.
            // We only read the raw pointer (via `raw_handle()`) — no mutation.
            let id_ptr = unsafe {
                let guard = wgpu_buffer.as_hal::<wgpu::hal::api::Metal>();
                let hal_buf = guard
                    .as_ref()
                    .ok_or("CubeCL buffer is not Metal — zero-copy requires Metal backend")?;
                let proto = hal_buf.raw_handle(); // &ProtocolObject<dyn MTLBuffer>
                proto as *const _ as *mut metal::MTLBuffer
            };
            Ok((id_ptr, offset))
        }
    }

    /// Issue 1146 — the first-use zero-init hazard (Issue 670 T5's
    /// "first-prefill pool-state" residual, root-caused).
    ///
    /// wgpu zero-initializes buffer memory LAZILY: the first wgpu pass that
    /// binds a never-initialized range gets a clear inserted ahead of it. The
    /// raw-Metal blit below writes the CubeCL output range BEHIND wgpu's back,
    /// so when that range is fresh pool memory, the next CubeCL kernel that
    /// reads it triggers the clear and the GEMM result becomes zeros. A warm
    /// pool hid it (every later prefill reuses initialized pages), which is why
    /// warmed A/Bs (riir-ai Bench 1009) read exact while a cold first prefill
    /// (every dist stage) did not.
    ///
    /// The cure: the first time an output range is seen, have CubeCL write it
    /// (one fill, submitted ahead of the raw command buffer on the same
    /// queue), so wgpu marks it initialized before the raw write lands. Steady
    /// state pays nothing: a seen range is a set lookup.
    pub(crate) fn ensure_wgpu_initialized(
        &self,
        client: &ComputeClient<ActiveRuntime>,
        output: &Handle,
        len: usize,
    ) -> Result<(), String> {
        let managed = client
            .get_resource(output.clone())
            .map_err(|e| format!("get_resource failed: {e:?}"))?;
        let resource = managed.resource();
        let start = resource.offset;
        let end = start + (len * std::mem::size_of::<f32>()) as u64;
        let mut seen = self.wgpu_initialized.lock().unwrap();
        let ranges = seen.entry(resource.buffer.clone()).or_default();
        if covered(ranges, start, end) {
            return Ok(());
        }
        unsafe {
            crate::elementwise_cubecl::FillZerosCubeCL::launch::<ActiveRuntime>(
                client,
                output.clone(),
                len,
            );
        }
        let _ = client.flush();
        insert_merged(ranges, start, end);
        let n = self
            .init_fills
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;
        if n.is_power_of_two() && std::env::var("RIIR_ZC_FILL_TRACE").as_deref() == Ok("1") {
            eprintln!("[Issue 1146] zero-copy first-use fills: {n}");
        }
        Ok(())
    }

    /// First-use fills issued so far (the zero-init cure's cost meter).
    pub fn init_fill_count(&self) -> u64 {
        self.init_fills.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Dispatch the ternary GEMM via raw Metal on the shared CubeCL queue.
    ///
    /// **Zero copy, zero host round-trip.** Reads input from the CubeCL input
    /// handle, writes output to the CubeCL output handle, dispatches the
    /// matmul2d kernel on the same Metal command queue CubeCL uses. No
    /// staging buffers, no bind groups, no host round-trip.
    ///
    /// # Arguments
    ///
    /// - `client`: the CubeCL compute client (for raw buffer extraction)
    /// - `weights`: the cached raw weight buffers (from `cache_weights`)
    /// - `w`: the ternary weight handle (for `m`, `n`, `blocks64`, etc.)
    /// - `input`: CubeCL handle holding `[P × N]` f32 input activations
    /// - `output`: CubeCL handle for `[P × M]` f32 output (will be written)
    /// - `p`: number of tokens (P dimension)
    pub fn dispatch(
        &self,
        client: &ComputeClient<ActiveRuntime>,
        weights: &ZeroCopyWeightCache,
        w: &TernaryHandle,
        input: &Handle,
        output: &Handle,
        p: usize,
    ) -> Result<(), String> {
        // Issue 670 T4: force CubeCL to submit any pending command buffers to
        // the shared Metal queue BEFORE we commit our raw command buffer.
        // This dispatch binds CubeCL pool buffers directly; if the producer
        // kernels writing `input` are still sitting in CubeCL's server
        // channel, our raw commit would be enqueued AHEAD of them and the
        // kernel would read stale input — deterministic-per-shape garbage
        // (the `needless_return` G1 failure in the production-scale re-bench).
        // The isolated Issue 663 bench masked this hazard by calling
        // `pollster::block_on(client.sync())` before extraction;
        // `flush()` gets the same submission ordering without a CPU-GPU sync
        // (it submits, it does not wait). Flush FIRST — also settles any
        // pending pool bookkeeping before `extract_raw_buffer` walks it.
        //
        // Issue 670 T5 (2026-08-14): a first-dispatch `client.sync()` variant
        // was tried and does NOT cure the residual first-prefill corruption
        // (case-order A/B: only the FIRST zerocopy prefill after init reads
        // wrong state; later prefills are bit-identical to simdgroup). The
        // corruption is not a submission-visibility hazard — ROOT-CAUSED in
        // Issue 1146 as wgpu's lazy zero-init clearing the raw-written output
        // on its first wgpu use; cured by `ensure_wgpu_initialized`.
        let _ = client.flush();
        self.ensure_wgpu_initialized(client, output, p * w.m)?;

        // Extract raw weight + input buffer handles fresh on every dispatch.
        let (pos_ptr, pos_off) = Self::extract_raw_buffer(client, &weights.pos)?;
        let (neg_ptr, neg_off) = Self::extract_raw_buffer(client, &weights.neg)?;
        let (scale_ptr, scale_off) = Self::extract_raw_buffer(client, &weights.scale)?;
        let (input_ptr, input_off) = Self::extract_raw_buffer(client, input)?;
        let (output_ptr, output_off) = Self::extract_raw_buffer(client, output)?;

        let input_ref: &BufferRef = unsafe { BufferRef::from_ptr(input_ptr) };
        let pos_ref: &BufferRef = unsafe { BufferRef::from_ptr(pos_ptr) };
        let neg_ref: &BufferRef = unsafe { BufferRef::from_ptr(neg_ptr) };
        let scale_ref: &BufferRef = unsafe { BufferRef::from_ptr(scale_ptr) };
        let cubecl_output_ref: &BufferRef = unsafe { BufferRef::from_ptr(output_ptr) };

        // Allocate or reuse a DEDICATED output buffer for the kernel write.
        //
        // CubeCL sub-allocates all handles within one `id<MTLBuffer>` pool
        // buffer. When the output handle shares the same pool buffer as the
        // weights/input (which it ALWAYS does in practice), Metal's hazard
        // tracking silently fails the cooperative-tensor store — the kernel
        // runs but writes nothing. The fix: write to a dedicated Metal
        // buffer (not from the CubeCL pool), then copy back to the CubeCL
        // output handle via `encoder.copy_from_buffer` (GPU→GPU, same
        // command buffer, zero host round-trip).
        //
        // This is the same pattern Issue 657 used for its staging buffers,
        // but ONLY for the output (not the input + weights) — reducing the
        // copy count from 2 (Issue 657) to 1.
        let output_bytes = (p * w.m * std::mem::size_of::<f32>()) as u64;
        let dedicated_output = {
            let mut cache = self.output_cache.lock().unwrap();
            cache
                .entry(output_bytes)
                .or_insert_with(|| {
                    let device_ref: &DeviceRef = unsafe { DeviceRef::from_ptr(self.device) };
                    device_ref.new_buffer(output_bytes, MTLResourceOptions::StorageModeShared)
                })
                .clone()
        };

        // matmul2d tile constants — from THIS context's compiled config
        // (Issue 1122 gate 2; must mirror the `SWEEP_*` defines above). The
        // smem length is THIS instance's compile-time posture, never the
        // live global — a double-buffer-compiled pipeline launched with
        // single-tile smem would write past the buffer.
        let nra = u64::from(self.config.nra);
        let nrb = u64::from(self.config.nrb);
        let nsg = u64::from(self.config.nsg);
        let threads_per_sg: u64 = 32;
        let num_threads = threads_per_sg * nsg;
        let threadgroup_mem_bytes = if self.double_buffered {
            self.config.threadgroup_mem_bytes() * 2
        } else {
            self.config.threadgroup_mem_bytes()
        };

        // Grid: one workgroup per output tile.
        // NOTE: use `dispatch_thread_groups` (not `dispatch_threads`). The
        // matmul2d kernel indexes its output tile via `tgpig` (threadgroup
        // position in grid), so the grid must be in thread GROUPS, not
        // threads. `dispatch_threads` interprets the grid as thread count,
        // which dispatches far too many workgroups + causes out-of-bounds writes.
        // (Issue 656's `MetalTensorGemm::launch` uses `dispatch_thread_groups`.)
        let grid = MTLSize {
            width: (p as u64).div_ceil(nrb),    // token tiles
            height: (w.m as u64).div_ceil(nra), // feature tiles
            depth: 1,
        };
        let threadgroup = MTLSize {
            width: num_threads,
            height: 1,
            depth: 1,
        };

        // Scalar params (must match the MSL kernel's `constant uint&` bindings).
        let blocks64: u32 = w.blocks64 as u32;
        let groups_per_row: u32 = w.groups_per_row as u32;
        let n: u32 = w.n as u32;
        let m: u32 = w.m as u32;
        let p_tokens: u32 = p as u32;

        // Dispatch on the shared Metal queue.
        let queue_ref: &CommandQueueRef = unsafe { CommandQueueRef::from_ptr(self.queue) };
        let cmd_buffer = queue_ref.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);

        // Buffer bindings (match gemm_ternary_metal_tensor.metal):
        //   0: pos_bits_u32 (read, CubeCL pool)
        //   1: neg_bits_u32 (read, CubeCL pool)
        //   2: group_scale_f32 (read, CubeCL pool)
        //   3: input_batch (read, CubeCL pool)
        //   4: output_batch (write, DEDICATED Metal buffer)
        encoder.set_buffer(0, Some(pos_ref), pos_off);
        encoder.set_buffer(1, Some(neg_ref), neg_off);
        encoder.set_buffer(2, Some(scale_ref), scale_off);
        encoder.set_buffer(3, Some(input_ref), input_off);
        encoder.set_buffer(4, Some(&dedicated_output), 0); // offset 0 — dedicated

        // Scalar bindings (5..9 match the MSL `constant uint&` declarations).
        encoder.set_bytes(5, 4, &blocks64 as *const u32 as *const _);
        encoder.set_bytes(6, 4, &groups_per_row as *const u32 as *const _);
        encoder.set_bytes(7, 4, &n as *const u32 as *const _);
        encoder.set_bytes(8, 4, &m as *const u32 as *const _);
        encoder.set_bytes(9, 4, &p_tokens as *const u32 as *const _);

        // Threadgroup memory for the dequanted weight tile (index 0).
        encoder.set_threadgroup_memory_length(0, threadgroup_mem_bytes);

        // Declare resource usage so Metal's hazard tracking is correct. The
        // weights + input are read; the output is written. Without this,
        // memory-less hazard tracking on M3 Max can race with CubeCL's
        // in-flight writes to the same buffers.
        encoder.use_resource(pos_ref, MTLResourceUsage::Read);
        encoder.use_resource(neg_ref, MTLResourceUsage::Read);
        encoder.use_resource(scale_ref, MTLResourceUsage::Read);
        encoder.use_resource(input_ref, MTLResourceUsage::Read);
        encoder.use_resource(&dedicated_output, MTLResourceUsage::Write);

        encoder.dispatch_thread_groups(grid, threadgroup);
        encoder.end_encoding();

        // Copy the kernel's output from the dedicated buffer back to the
        // CubeCL output handle. This is a GPU→GPU copy within the same
        // command buffer — zero host round-trip. The copy is ordered after
        // the compute pass (same encoder sequence).
        let blit_encoder = cmd_buffer.new_blit_command_encoder();
        blit_encoder.copy_from_buffer(
            &dedicated_output,
            0,
            cubecl_output_ref,
            output_off,
            output_bytes,
        );
        blit_encoder.end_encoding();

        cmd_buffer.commit();
        // Retain for wait_last (the one caller class that needs an explicit
        // wait — see the comment above: non-queued reads of the output).
        *self.last_cmd_buffer.lock().expect("last_cmd_buffer lock") = Some(cmd_buffer.to_owned());

        // NOTE: we do NOT call `cmd_buffer.wait_until_completed()`. The command
        // buffer runs on the shared Metal queue — CubeCL's subsequent
        // dispatches (including the next prefill_project that reads this
        // output) are automatically ordered after it (Metal guarantees
        // in-order execution within one queue). Blocking here would serialize
        // the async pipeline, reintroducing the Issue 656 host-round-trip cost.
        //
        // The `client.flush()` at the TOP of this function ensures the input
        // buffer's producers are already SUBMITTED, so queue ordering makes
        // their writes visible to this kernel; the queue ordering likewise
        // makes the OUTPUT visible to the next consumer. The only caller that
        // needs explicit `wait_until_completed` is a test that reads the
        // buffer via a non-queued path (e.g., `client.read_one` which may map
        // without a GPU sync). Production code never does this within the
        // tick loop.

        self.dispatch_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Issue 1225 — the f16-B serving twin (feature `prefill_f16b`)
    // -----------------------------------------------------------------------

    /// Compile the f16-B twin: the serving zero-copy context at the
    /// production geometry with [`F16_BODY_PATCHES`] applied to the MSL body
    /// (B-side activations read `half`; accumulator stays float), plus the
    /// cast kernel in the same library. The placebo posture
    /// ([`crate::prefill_f16b::placebo_pinned`]) builds the UNPATCHED body
    /// with no f16 state — the control arm dispatches through the standard
    /// [`Self::dispatch`]. The flags (placebo included) must be pinned BEFORE
    /// construction — the compiled body is a construction-time posture.
    ///
    /// The extra compile cost (the unpatched pipeline `new_with_config`
    /// builds first) is accepted: one-time, opt-in lane, keeps the ctor
    /// refactor out of the incumbent path.
    #[cfg(feature = "prefill_f16b")]
    pub fn new_f16b(
        wgpu_device: Arc<WgpuDevice>,
        wgpu_queue: Arc<WgpuQueue>,
    ) -> Result<Self, String> {
        let mut me = Self::new_with_config(
            crate::gemm_ternary_metal_tensor::production_geometry(),
            wgpu_device,
            wgpu_queue,
        )?;
        if crate::prefill_f16b::placebo_pinned() {
            eprintln!("[i1225] f16b twin built PLACEBO (unpatched body — the control arm)");
            return Ok(me);
        }
        let patched = apply_patches_checked(MSL_SOURCE, F16_BODY_PATCHES)?;
        let mut source = crate::gemm_ternary_metal_tensor::compose_msl_source(me.config, &patched);
        source.push_str(CAST_F16_MSL);
        let device_ref: &DeviceRef = unsafe { DeviceRef::from_ptr(me.device) };
        let library = device_ref
            .new_library_with_source(source.as_str(), &CompileOptions::new())
            .map_err(|e| format!("f16b twin MSL compilation failed: {e}"))?;
        let gemm_fn = library
            .get_function("gemm_ternary_tensor", None)
            .map_err(|e| format!("f16b twin: function not found: {e}"))?;
        me.pipeline = device_ref
            .new_compute_pipeline_state_with_function(&gemm_fn)
            .map_err(|e| format!("f16b twin pipeline failed: {e}"))?;
        let cast_fn = library
            .get_function("cast_f32_f16_zc", None)
            .map_err(|e| format!("f16b twin cast kernel not found: {e}"))?;
        let cast_pipeline = device_ref
            .new_compute_pipeline_state_with_function(&cast_fn)
            .map_err(|e| format!("f16b cast pipeline failed: {e}"))?;
        me.f16b = Some(F16bState {
            cast_pipeline,
            input_cache: Mutex::new(HashMap::new()),
        });
        eprintln!("[i1225] f16b twin compiled (patched B-side half; cast kernel armed)");
        Ok(me)
    }

    /// The f16-B twin dispatch: ONE command buffer carrying the cast encoder
    /// (f32 input → dedicated f16 copy) → the patched gemm encoder (reads the
    /// f16 copy) → the blit back to the CubeCL output handle. Same queue, so
    /// CubeCL ordering and the in-buffer encoder ordering make the rounded
    /// copy visible to the gemm and the result visible to the next consumer.
    /// The per-launch cast is the bounded worst case (Bench 1223: 0.15–0.19
    /// ms/launch); a producer-writes-f16 shape removes it from the frame.
    #[cfg(feature = "prefill_f16b")]
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_f16b(
        &self,
        client: &ComputeClient<ActiveRuntime>,
        weights: &ZeroCopyWeightCache,
        w: &TernaryHandle,
        input: &Handle,
        output: &Handle,
        p: usize,
    ) -> Result<(), String> {
        self.dispatch_f16b_producer(client, weights, w, input, None, output, p)
    }

    /// i1226 F2 — the producer-writes-f16 form: `input_f16` is `Some(shadow)`
    /// when the rotate that last wrote `input` also emitted the packed-u32
    /// f16 pairs (same RNE rounding as the cast encoder — the G1 shadow cell
    /// pins it), and the cast encoder is SKIPPED — the gemm binds the shadow
    /// at its extracted offset directly. Misaligned shadow offsets (a pool
    /// slice under the vector loads' 16-byte assumption) fall back to the
    /// cast path LOUD and counted — never a silent wrong bind. `input` stays
    /// the f32 handle and `input_cache` stays for the fallback — never freed.
    #[cfg(feature = "prefill_f16b")]
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_f16b_producer(
        &self,
        client: &ComputeClient<ActiveRuntime>,
        weights: &ZeroCopyWeightCache,
        w: &TernaryHandle,
        input: &Handle,
        input_f16: Option<&Handle>,
        output: &Handle,
        p: usize,
    ) -> Result<(), String> {
        let Some(f16b) = self.f16b.as_ref() else {
            return Err(
                "f16b twin has no f16 state (built in the placebo posture) — pin the \
                 posture BEFORE construction"
                    .to_string(),
            );
        };
        // Same submission-ordering + zero-init law as `dispatch` (Issue 670
        // T4 / Issue 1146 lazy-clear).
        let _ = client.flush();
        self.ensure_wgpu_initialized(client, output, p * w.m)?;

        let (pos_ptr, pos_off) = Self::extract_raw_buffer(client, &weights.pos)?;
        let (neg_ptr, neg_off) = Self::extract_raw_buffer(client, &weights.neg)?;
        let (scale_ptr, scale_off) = Self::extract_raw_buffer(client, &weights.scale)?;
        let (input_ptr, input_off) = Self::extract_raw_buffer(client, input)?;
        let (output_ptr, output_off) = Self::extract_raw_buffer(client, output)?;

        let input_ref: &BufferRef = unsafe { BufferRef::from_ptr(input_ptr) };
        let pos_ref: &BufferRef = unsafe { BufferRef::from_ptr(pos_ptr) };
        let neg_ref: &BufferRef = unsafe { BufferRef::from_ptr(neg_ptr) };
        let scale_ref: &BufferRef = unsafe { BufferRef::from_ptr(scale_ptr) };
        let cubecl_output_ref: &BufferRef = unsafe { BufferRef::from_ptr(output_ptr) };

        // The producer shadow, extracted for the gemm's bind. A pool slice
        // arrives with a nonzero offset — the patched GEMM's vector/simdgroup
        // loads assume 16-byte+ alignment, so a misaligned shadow declines to
        // the cast path (counted, loud once) instead of binding wrong.
        let mut shadow: Option<(&BufferRef, u64)> = None;
        if let Some(h) = input_f16 {
            let (ptr, off) = Self::extract_raw_buffer(client, h)?;
            if off % 16 == 0 {
                // Safety: same shape as the extracted input/output refs above
                // — the raw pointer is the CubeCL pool allocation, kept alive
                // by the caller's shadow handle for the dispatch's lifetime.
                shadow = Some((unsafe { BufferRef::from_ptr(ptr) }, off));
                crate::prefill_f16b::count_cast_skipped();
            } else {
                crate::prefill_f16b::count_shadow_misaligned();
            }
        }
        // Dedicated f16 input copy + dedicated output (the same
        // pool-aliasing law as `dispatch`'s output buffer) — the copy is
        // only allocated when the CAST path is actually taken.
        let cast_input: Option<Buffer> = if shadow.is_none() {
            crate::prefill_f16b::count_cast_encoder();
            let f16_bytes = (p * w.n * std::mem::size_of::<u16>()) as u64;
            let f16_input = {
                let mut cache = f16b.input_cache.lock().unwrap();
                cache
                    .entry(f16_bytes)
                    .or_insert_with(|| {
                        let device_ref: &DeviceRef = unsafe { DeviceRef::from_ptr(self.device) };
                        device_ref.new_buffer(f16_bytes, MTLResourceOptions::StorageModeShared)
                    })
                    .clone()
            };
            Some(f16_input)
        } else {
            None
        };
        let output_bytes = (p * w.m * std::mem::size_of::<f32>()) as u64;
        let dedicated_output = self.dedicated_output(output_bytes);

        let nra = u64::from(self.config.nra);
        let nrb = u64::from(self.config.nrb);
        let num_threads = u64::from(32 * self.config.nsg);
        let threadgroup_mem_bytes = if self.double_buffered {
            self.config.threadgroup_mem_bytes() * 2
        } else {
            self.config.threadgroup_mem_bytes()
        };
        let grid = MTLSize {
            width: (p as u64).div_ceil(nrb),
            height: (w.m as u64).div_ceil(nra),
            depth: 1,
        };
        let threadgroup = MTLSize {
            width: num_threads,
            height: 1,
            depth: 1,
        };

        let blocks64: u32 = w.blocks64 as u32;
        let groups_per_row: u32 = w.groups_per_row as u32;
        let n: u32 = w.n as u32;
        let m: u32 = w.m as u32;
        let p_tokens: u32 = p as u32;

        let queue_ref: &CommandQueueRef = unsafe { CommandQueueRef::from_ptr(self.queue) };
        let cmd_buffer = queue_ref.new_command_buffer();

        // Encoder 1: the cast (f32 → f16 dedicated copy) — SKIPPED when the
        // producer shadow is bound (its per-launch cost is the tax F2
        // removes from the frame).
        if let Some(f16_input) = cast_input.as_ref() {
            let cast_n: u32 = p_tokens * n;
            let enc = cmd_buffer.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&f16b.cast_pipeline);
            enc.set_buffer(0, Some(input_ref), input_off);
            enc.set_buffer(1, Some(f16_input), 0);
            enc.set_bytes(2, 4, &cast_n as *const u32 as *const _);
            enc.use_resource(input_ref, MTLResourceUsage::Read);
            enc.use_resource(f16_input, MTLResourceUsage::Write);
            let cast_grid = MTLSize {
                width: (u64::from(p_tokens) * u64::from(n)).div_ceil(256),
                height: 1,
                depth: 1,
            };
            let cast_tg = MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            };
            enc.dispatch_thread_groups(cast_grid, cast_tg);
            enc.end_encoding();
        }

        // Encoder 2: the patched gemm — input binding is the f16 copy (cast
        // path, offset 0) or the producer shadow (its extracted offset).
        {
            let encoder = cmd_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.pipeline);
            encoder.set_buffer(0, Some(pos_ref), pos_off);
            encoder.set_buffer(1, Some(neg_ref), neg_off);
            encoder.set_buffer(2, Some(scale_ref), scale_off);
            if let Some(f16_input) = cast_input.as_ref() {
                encoder.set_buffer(3, Some(f16_input), 0);
                encoder.use_resource(f16_input, MTLResourceUsage::Read);
            } else {
                let (shadow_ref, shadow_off) = shadow.expect("cast path provides the f16 input");
                encoder.set_buffer(3, Some(shadow_ref), shadow_off);
                encoder.use_resource(shadow_ref, MTLResourceUsage::Read);
            }
            encoder.set_buffer(4, Some(&dedicated_output), 0);
            encoder.set_bytes(5, 4, &blocks64 as *const u32 as *const _);
            encoder.set_bytes(6, 4, &groups_per_row as *const u32 as *const _);
            encoder.set_bytes(7, 4, &n as *const u32 as *const _);
            encoder.set_bytes(8, 4, &m as *const u32 as *const _);
            encoder.set_bytes(9, 4, &p_tokens as *const u32 as *const _);
            encoder.set_threadgroup_memory_length(0, threadgroup_mem_bytes);
            encoder.use_resource(pos_ref, MTLResourceUsage::Read);
            encoder.use_resource(neg_ref, MTLResourceUsage::Read);
            encoder.use_resource(scale_ref, MTLResourceUsage::Read);
            encoder.use_resource(&dedicated_output, MTLResourceUsage::Write);
            encoder.dispatch_thread_groups(grid, threadgroup);
            encoder.end_encoding();
        }

        // Blit back to the CubeCL output handle (same-command-buffer ordering).
        {
            let blit = cmd_buffer.new_blit_command_encoder();
            blit.copy_from_buffer(
                &dedicated_output,
                0,
                cubecl_output_ref,
                output_off,
                output_bytes,
            );
            blit.end_encoding();
        }

        cmd_buffer.commit();
        *self.last_cmd_buffer.lock().expect("last_cmd_buffer lock") = Some(cmd_buffer.to_owned());
        self.dispatch_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// i1226 F2 — run THE cast pipeline (`f16b.cast_pipeline`, the encoder
    /// the producer shadow replaces) over `input` and read the rounded f16
    /// bytes back. INSTRUMENT-ONLY (the G1 shadow cell's authority — the
    /// comparison the armed posture is gated on; the integration test is the
    /// only caller) — never a serving path. `input`: `n` f32 elements;
    /// returns `n` u16s.
    #[cfg(feature = "prefill_f16b")]
    pub fn cast_f16_reference(
        &self,
        client: &ComputeClient<ActiveRuntime>,
        input: &Handle,
        n: usize,
    ) -> Result<Vec<u16>, String> {
        let f16b = self
            .f16b
            .as_ref()
            .ok_or_else(|| "f16b twin built in the placebo posture".to_string())?;
        let _ = client.flush();
        let (input_ptr, input_off) = Self::extract_raw_buffer(client, input)?;
        let input_ref: &BufferRef = unsafe { BufferRef::from_ptr(input_ptr) };
        let device_ref: &DeviceRef = unsafe { DeviceRef::from_ptr(self.device) };
        let dst = device_ref.new_buffer(
            (n * std::mem::size_of::<u16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let queue_ref: &CommandQueueRef = unsafe { CommandQueueRef::from_ptr(self.queue) };
        let cmd_buffer = queue_ref.new_command_buffer();
        let enc = cmd_buffer.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&f16b.cast_pipeline);
        enc.set_buffer(0, Some(input_ref), input_off);
        enc.set_buffer(1, Some(&dst), 0);
        let cast_n: u32 = n as u32;
        enc.set_bytes(2, 4, &cast_n as *const u32 as *const _);
        enc.use_resource(input_ref, MTLResourceUsage::Read);
        enc.use_resource(&dst, MTLResourceUsage::Write);
        let grid = MTLSize {
            width: (n as u64).div_ceil(256),
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        enc.dispatch_thread_groups(grid, tg);
        enc.end_encoding();
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
        let ptr = dst.contents() as *const u16;
        Ok(unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec())
    }

    /// Tap-only dispatch (the Issue-1225 retention walk): the incumbent body
    /// into THIS instance's own dedicated output — NO blit, so the real
    /// output handle (already carrying the twin's result) is untouched.
    /// Returns the dedicated buffer; the caller waits
    /// ([`Self::wait_last`]) and reads it before the next dispatch reuses
    /// it. The tap is instrument-grade: it serializes per site by design.
    #[cfg(feature = "prefill_f16b")]
    pub fn dispatch_scratch(
        &self,
        client: &ComputeClient<ActiveRuntime>,
        weights: &ZeroCopyWeightCache,
        w: &TernaryHandle,
        input: &Handle,
        p: usize,
    ) -> Result<Buffer, String> {
        let _ = client.flush();
        let (pos_ptr, pos_off) = Self::extract_raw_buffer(client, &weights.pos)?;
        let (neg_ptr, neg_off) = Self::extract_raw_buffer(client, &weights.neg)?;
        let (scale_ptr, scale_off) = Self::extract_raw_buffer(client, &weights.scale)?;
        let (input_ptr, input_off) = Self::extract_raw_buffer(client, input)?;
        let input_ref: &BufferRef = unsafe { BufferRef::from_ptr(input_ptr) };
        let pos_ref: &BufferRef = unsafe { BufferRef::from_ptr(pos_ptr) };
        let neg_ref: &BufferRef = unsafe { BufferRef::from_ptr(neg_ptr) };
        let scale_ref: &BufferRef = unsafe { BufferRef::from_ptr(scale_ptr) };

        let output_bytes = (p * w.m * std::mem::size_of::<f32>()) as u64;
        let dedicated_output = self.dedicated_output(output_bytes);

        let nra = u64::from(self.config.nra);
        let nrb = u64::from(self.config.nrb);
        let num_threads = u64::from(32 * self.config.nsg);
        let threadgroup_mem_bytes = if self.double_buffered {
            self.config.threadgroup_mem_bytes() * 2
        } else {
            self.config.threadgroup_mem_bytes()
        };
        let grid = MTLSize {
            width: (p as u64).div_ceil(nrb),
            height: (w.m as u64).div_ceil(nra),
            depth: 1,
        };
        let threadgroup = MTLSize {
            width: num_threads,
            height: 1,
            depth: 1,
        };

        let queue_ref: &CommandQueueRef = unsafe { CommandQueueRef::from_ptr(self.queue) };
        let cmd_buffer = queue_ref.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(pos_ref), pos_off);
        encoder.set_buffer(1, Some(neg_ref), neg_off);
        encoder.set_buffer(2, Some(scale_ref), scale_off);
        encoder.set_buffer(3, Some(input_ref), input_off);
        encoder.set_buffer(4, Some(&dedicated_output), 0);
        encoder.set_bytes(5, 4, &(w.blocks64 as u32) as *const u32 as *const _);
        encoder.set_bytes(6, 4, &(w.groups_per_row as u32) as *const u32 as *const _);
        encoder.set_bytes(7, 4, &(w.n as u32) as *const u32 as *const _);
        encoder.set_bytes(8, 4, &(w.m as u32) as *const u32 as *const _);
        encoder.set_bytes(9, 4, &(p as u32) as *const u32 as *const _);
        encoder.set_threadgroup_memory_length(0, threadgroup_mem_bytes);
        encoder.use_resource(pos_ref, MTLResourceUsage::Read);
        encoder.use_resource(neg_ref, MTLResourceUsage::Read);
        encoder.use_resource(scale_ref, MTLResourceUsage::Read);
        encoder.use_resource(input_ref, MTLResourceUsage::Read);
        encoder.use_resource(&dedicated_output, MTLResourceUsage::Write);
        encoder.dispatch_thread_groups(grid, threadgroup);
        encoder.end_encoding();
        cmd_buffer.commit();
        *self.last_cmd_buffer.lock().expect("last_cmd_buffer lock") = Some(cmd_buffer.to_owned());
        self.dispatch_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(dedicated_output)
    }

    /// The dedicated output buffer last written for `byte_len` (the tap reads
    /// the twin's result from here after [`Self::wait_last`]).
    #[cfg(feature = "prefill_f16b")]
    pub fn dedicated_output_for(&self, byte_len: u64) -> Option<Buffer> {
        self.output_cache.lock().unwrap().get(&byte_len).cloned()
    }

    /// Allocate-or-reuse a dedicated output buffer of `byte_len` (the shared
    /// body of `dispatch_f16b`/`dispatch_scratch`; plain `dispatch` keeps its
    /// inline copy — the incumbent path is never refactored under an opt-in
    /// lane).
    #[cfg(feature = "prefill_f16b")]
    fn dedicated_output(&self, byte_len: u64) -> Buffer {
        let mut cache = self.output_cache.lock().unwrap();
        cache
            .entry(byte_len)
            .or_insert_with(|| {
                let device_ref: &DeviceRef = unsafe { DeviceRef::from_ptr(self.device) };
                device_ref.new_buffer(byte_len, MTLResourceOptions::StorageModeShared)
            })
            .clone()
    }

    /// Convenience: get the device pointer (for T4b-style device-equality checks).
    pub fn device_ptr(&self) -> usize {
        self.device as usize
    }

    /// Convenience: get the queue pointer (for diagnostics).
    pub fn queue_ptr(&self) -> usize {
        self.queue as usize
    }

    /// Wait for the most recent raw dispatch's command buffer to complete
    /// (instrument-grade sync: tests and timing cells that read the output
    /// via a non-queued path — `client.read_one` — race the pending raw
    /// work otherwise, because the raw submissions are invisible to wgpu's
    /// tracking). Production dispatch loops never call this: the next
    /// dispatch's `client.flush()` + queue ordering make the pipeline
    /// self-serializing (the dispatch comment block above).
    pub fn wait_last(&self) {
        if let Some(cb) = self
            .last_cmd_buffer
            .lock()
            .expect("last_cmd_buffer lock")
            .as_ref()
        {
            cb.wait_until_completed();
        }
    }
}

impl Drop for MetalTensorZeroCopyGemm {
    fn drop(&mut self) {
        // The raw device + queue pointers are borrowed from the wgpu Arcs
        // (`_wgpu_device` + `_wgpu_queue`). Dropping those Arcs would
        // invalidate the pointers — but Drop runs before the Arcs drop, so
        // the borrow is still valid here. We don't need to release anything
        // (foreign-types doesn't own these borrowed refs).
    }
}

/// Whether `[start, end)` lies inside one of the sorted, merged `ranges`.
pub(crate) fn covered(ranges: &[(u64, u64)], start: u64, end: u64) -> bool {
    let i = ranges.partition_point(|&(s, _)| s <= start);
    i > 0 && ranges[i - 1].1 >= end
}

/// Insert `[start, end)` into sorted, merged `ranges`, merging overlaps and
/// adjacent intervals.
pub(crate) fn insert_merged(ranges: &mut Vec<(u64, u64)>, start: u64, end: u64) {
    let (mut s, mut e) = (start, end);
    let lo = ranges.partition_point(|&(_, re)| re < s);
    let mut hi = lo;
    while hi < ranges.len() && ranges[hi].0 <= e {
        s = s.min(ranges[hi].0);
        e = e.max(ranges[hi].1);
        hi += 1;
    }
    ranges.splice(lo..hi, std::iter::once((s, e)));
}

#[cfg(test)]
mod init_range_tests {
    use super::{covered, insert_merged};

    #[test]
    fn containment_merge_and_gaps() {
        let mut r = Vec::new();
        assert!(!covered(&r, 0, 1));
        insert_merged(&mut r, 100, 200);
        assert!(covered(&r, 100, 200) && covered(&r, 120, 180));
        assert!(!covered(&r, 90, 150) && !covered(&r, 150, 210));
        // Disjoint insert keeps order; a gap stays uncovered.
        insert_merged(&mut r, 300, 400);
        assert_eq!(r, vec![(100, 200), (300, 400)]);
        assert!(!covered(&r, 150, 350));
        // Adjacent intervals merge (end == start).
        insert_merged(&mut r, 200, 300);
        assert_eq!(r, vec![(100, 400)]);
        assert!(covered(&r, 150, 350));
        // An insert spanning several intervals collapses them.
        insert_merged(&mut r, 500, 600);
        insert_merged(&mut r, 700, 800);
        insert_merged(&mut r, 50, 750);
        assert_eq!(r, vec![(50, 800)]);
        // Inserting inside an existing interval is a no-op.
        insert_merged(&mut r, 60, 70);
        assert_eq!(r, vec![(50, 800)]);
    }
}
