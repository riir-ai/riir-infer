//! Issue 1167 rung C — the H5 GEMM-staging price cell (the price BEFORE any
//! kernel is written).
//!
//! The promoted prefill GEMM (`gemm_ternary_metal_tensor`, KTILE64 posture)
//! runs per K-tile: Phase-1 dequant into threadgroup `half` staging →
//! barrier → `matmul2d` (threadgroup-staged A) → barrier. There is no
//! double-buffering, so the matmul idles during dequant+barrier and the ALU
//! idles during the matmul — the rung C prize is exactly that serialized
//! slice. This instrument prices it WITHOUT writing a serving kernel:
//!
//! - **Arm `incumbent`** — the real `MetalTensorGemm` at
//!   [`MetalTileConfig::KTILE64`], full Phase-1 + matmul2d pipeline.
//! - **Arm `dequant_probe`** — a measurement-only pipeline (compiled in THIS
//!   file, never referenced by any serving path) that reproduces the
//!   incumbent's Phase-1 dequant body VERBATIM (V2 hoisted body + the
//!   ragged-edge fallback), both per-K-tile barriers, the same grid /
//!   threadgroup size / smem length / buffer bindings — and replaces the
//!   matmul with a structurally-live consume (one staged slot per thread per
//!   tile, folded through a threadgroup epilogue into one stored float per
//!   workgroup), so no store can be dead-code eliminated (the timed-region
//!   loud-zero lesson). Probe share = probe wall / incumbent wall is the
//!   dequant+barrier+stage-write ceiling rung C could reclaim.
//!
//! Methodology: interleaved rotating-start arms, median over rounds,
//! per-launch wall with `wait_until_completed` fencing BOTH arms (a timed
//! region cannot exclude the work). Buffers are uploaded ONCE per (shape, P)
//! and reused across launches — deliberately unlike the 1166 sweep's
//! per-launch upload, whose host-side memcpy swamped per-launch kernel time;
//! this cell wants kernel-only walls and the upload cancels in a ratio
//! anyway. The incumbent arm's G1 is scale-relative 1e-5 against a CPU
//! decode of the same bit-plane fixture (config-independent truth); the
//! probe arm's liveness is the every-tile-written sentinel check.
//!
//! The instrument prints per row: incumbent ms, probe ms, probe share, the
//! matmul+store residual, its TFLOPS, and the H5-device-f16 traffic delta
//! (the A-side bytes a no-staging variant adds over the packed-bit read at
//! the 800 GB/s DRAM class) — the verdict arithmetic lives in the bench
//! record, not here.
//!
//! ```sh
//! CARGO_TARGET_DIR=/tmp/i1167 cargo test -p riir-infer-gpu --release \
//!   --features metal_tensor_gemm \
//!   --test issue1167_rung_c_staging_price -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Env: `BENCH1167_SHAPES` (`MxN,…`, default the three Bonsai-2 prefill
//! bottlenecks; G3/G4 add the GDN projections), `BENCH1167_PS` (default
//! `512,2048,4096` for G2, `512` for G3, `512` for G4), `BENCH1167_ROUNDS` (4),
//! `BENCH1167_LAUNCHES` (2), `BENCH1167_REPS` (G3 repetition blocks for the
//! delta IQR), `BENCH1167_FRAME_REF_MS` (G4 projection reference, 500.86).
//!
//! G4 (Issue 1167 rung B's packed-byte dequant ground) adds the PackedBody
//! twin: the serving kernel with the A-side read+decode swapped for the
//! packed-2-bit body (ONE u32 load per 16-element work item, decode algebra
//! `(code&1) - ((code>>1)&1)` — mathematically identical to
//! `(pos_bit - neg_bit)`). The twin is CORRECT, so baseline − packed is the
//! ADDRESSABLE packed-format price in situ (same stores, barriers, grid),
//! and the per-launch bit-identity assert is the correctness anchor a
//! future packed serving kernel inherits.

#![cfg(all(feature = "metal_tensor_gemm", target_os = "macos"))]

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions,
    MTLResourceUsage, MTLSize,
};
use riir_infer_gpu::gemm_ternary_metal_tensor::{MetalTensorGemm, MetalTileConfig};

/// Sentinel written into probe output buffers before launch — a tile that
/// fails to store its consume is caught, never inherited.
const SENTINEL: f32 = -98765.4;

/// G1 scale-relative bound — the bench_1113/1166 bar.
const G1_REL_LIMIT: f64 = 1e-5;

/// The three Bonsai-2 prefill bottlenecks (Bench 925/937 census): the GEMM
/// is `Y[P×M] = dequant(W[M×K]) @ X[K×P]`, one launch per projection.
const DEFAULT_SHAPES: &[(&str, usize, usize)] = &[
    ("ffn_gate/up", 17408, 5120),
    ("ffn_down", 5120, 17408),
    ("attn_q", 12288, 5120),
];

/// The GDN projection shapes (48 of 64 layers carry them; the rung-C verdict
/// must cover every ternary GEMM class, not only FFN). Derived from the
/// model: state 128, k-heads 16, v-heads 48 → qkv [14336×5120], z [6144×5120],
/// a/b [16×5120], out [5120×6144] (the loader's own probe read).
const GDN_SHAPES: &[(&str, usize, usize)] = &[
    ("gdn_qkv", 14336, 5120),
    ("gdn_z", 6144, 5120),
    ("gdn_ab", 16, 5120),
    ("gdn_out", 5120, 6144),
];

/// P = 512 (the dist w512 frame) + the league rows.
const DEFAULT_PS: &[usize] = &[512, 2048, 4096];

// ---------------------------------------------------------------------------
// Fixtures — raw bit-plane wire format, no cross-crate dependency
// ---------------------------------------------------------------------------

struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
}

/// The exact wire format the kernel reads: pos/neg u32 words
/// `[rows × (cols/32)]`, one f32 scale per 128-element group per row
/// (`words_per_row = blocks64 * 2` at the kernel).
struct TernaryFixture {
    rows: usize,
    cols: usize,
    pos_u32: Vec<u32>,
    neg_u32: Vec<u32>,
    scale_f32: Vec<f32>,
    blocks64: usize,
    groups_per_row: usize,
    /// Packed-2-bit twin of the bit-planes (G4): 16 codes per u32, code lane
    /// = pos_bit | neg_bit << 1 — the pure-function source format a packed
    /// loader would hand the kernel. Synthesized, never a second fixture RNG.
    packed_u32: Vec<u32>,
}

impl TernaryFixture {
    /// Realistic ternary fill matching the measured Bonsai distribution
    /// (35.1% −1 / 29.9% 0 / 35.0% +1 — the bench_1113 fill). `cols % 64`
    /// is asserted whole-word so the timing shapes stay on the V2 fast path.
    fn build(rows: usize, cols: usize, seed: u64) -> Self {
        assert_eq!(
            cols % 64,
            0,
            "whole-word columns keep the body on the V2 fast path"
        );
        let words_per_row = cols / 32;
        let blocks64 = cols / 64;
        let groups_per_row = cols.div_ceil(128);
        let mut rng = Lcg(seed);
        let mut pos_u32 = vec![0u32; rows * words_per_row];
        let mut neg_u32 = vec![0u32; rows * words_per_row];
        for idx in 0..pos_u32.len() {
            let mut pw = 0u32;
            let mut nw = 0u32;
            for lane in 0..32u32 {
                match rng.next_u64() % 1000 {
                    0..=350 => pw |= 1 << lane,
                    351..=649 => {}
                    _ => nw |= 1 << lane,
                }
            }
            pos_u32[idx] = pw;
            neg_u32[idx] = nw;
        }
        let mut scale_f32 = Vec::with_capacity(rows * groups_per_row);
        for _ in 0..rows * groups_per_row {
            // f16-exact by construction — the real wire stores scales as f16
            // (`TernaryGroupWeights.group_scale`), so `sign * scale` is
            // exactly representable in half and the kernel's staged tile is
            // exact. A full-f32 scale here would inject a ~5e-4 half-rounding
            // error the production path never carries (G1 caught exactly
            // that on the first run).
            let m = (rng.next_u64() % 1024) as f32;
            let e = -6i32 + (rng.next_u64() % 5) as i32;
            scale_f32.push((1.0 + m / 1024.0) * 2f32.powi(e));
        }
        // The packed synthesis is a PURE FUNCTION of the bit-planes: the G4
        // twin's decode algebra `(code&1) - ((code>>1)&1)` reproduces
        // `(pos_bit - neg_bit)` exactly for every code 0..4, so its staged
        // tile is bit-identical and the price arm doubles as the correctness
        // anchor for a future packed serving kernel.
        let packed_words_per_row = words_per_row * 2; // 32 columns per 2 bit-plane words → 16 codes per word
        let mut packed_u32 = vec![0u32; rows * packed_words_per_row];
        for r in 0..rows {
            for wp in 0..packed_words_per_row {
                let mut word = 0u32;
                for lane in 0..16u32 {
                    let col = wp * 16 + lane as usize;
                    let pw = pos_u32[r * words_per_row + col / 32];
                    let nw = neg_u32[r * words_per_row + col / 32];
                    let bit = 1u32 << (col % 32);
                    let code = (pw & bit != 0) as u32 | ((nw & bit != 0) as u32) << 1;
                    word |= code << (lane * 2);
                }
                packed_u32[r * packed_words_per_row + wp] = word;
            }
        }
        Self {
            rows,
            cols,
            pos_u32,
            neg_u32,
            scale_f32,
            blocks64,
            groups_per_row,
            packed_u32,
        }
    }

    /// CPU dequant-GEMV — decoded straight from the bit-plane layout,
    /// config-independent truth with no shared GPU path. Scale index is the
    /// kernel's hard-coded 128-wide group (`col / 128`, array sized
    /// `rows × ceil(cols/128)`) — the Q2_0_g128 wire contract, which is NOT
    /// `cols / groups_per_row` on ragged cols (33×192: 96-wide arithmetic
    /// groups vs the wire's 128-wide indexed ones).
    fn cpu_reference(&self, x: &[f32], p: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; p * self.rows];
        for r in 0..self.rows {
            let pos = &self.pos_u32[r * 2 * self.blocks64..(r + 1) * 2 * self.blocks64];
            let neg = &self.neg_u32[r * 2 * self.blocks64..(r + 1) * 2 * self.blocks64];
            for t in 0..p {
                let x_row = &x[t * self.cols..(t + 1) * self.cols];
                let mut acc = 0.0f32;
                for b in 0..self.blocks64 {
                    for half_word in 0..2usize {
                        let pw = pos[b * 2 + half_word];
                        let nw = neg[b * 2 + half_word];
                        if pw == 0 && nw == 0 {
                            continue;
                        }
                        for lane in 0..32u32 {
                            let e = b * 64 + half_word * 32 + lane as usize;
                            let bit = 1u32 << lane;
                            let sign = if pw & bit != 0 {
                                1.0f32
                            } else if nw & bit != 0 {
                                -1.0f32
                            } else {
                                continue;
                            };
                            acc +=
                                sign * self.scale_f32[r * self.groups_per_row + e / 128] * x_row[e];
                        }
                    }
                }
                out[t * self.rows + r] = acc;
            }
        }
        out
    }
}

fn random_input(len: usize, seed: u64) -> Vec<f32> {
    let mut rng = Lcg(seed);
    (0..len)
        .map(|_| ((rng.next_u64() >> 40) as f32 / 8_388_608.0) - 1.0)
        .collect()
}

// ---------------------------------------------------------------------------
// The dequant-only probe pipeline (measurement-only, compiled in this file)
// ---------------------------------------------------------------------------

/// The probe MSL: the incumbent's Phase-1 dequant body VERBATIM (V2 hoisted
/// fast path + the ragged-edge per-element fallback), both per-K-tile
/// barriers, and a structurally-live consume in place of the matmul —
/// per-thread one staged slot per tile, folded through a threadgroup
/// epilogue into one stored float per workgroup.
fn probe_source(cfg: MetalTileConfig) -> String {
    let sz_simd = 16u32;
    let k_tile = sz_simd * cfg.nk_chunk;
    let num_threads = 32 * cfg.nsg;
    let a_work_items = cfg.nra * cfg.nk_chunk;
    assert!(
        (k_tile as usize * cfg.nra as usize) >= num_threads as usize,
        "consume indexes sa[tiitg]; the tile must hold at least one slot per thread"
    );
    format!(
        r#"// Issue 1167 rung C price probe — dequant-only twin of gemm_ternary_tensor.
// Measurement-only: NO serving path references this kernel. Phase 1 is the
// incumbent body verbatim; the matmul is replaced by a liveness consume.
#include <metal_stdlib>
using namespace metal;

constant constexpr int SZ_SIMD     = {sz_simd};
constant constexpr int NK_CHUNK    = {nk_chunk};
constant constexpr int K_TILE      = SZ_SIMD * NK_CHUNK;
constant constexpr int NRA         = {nra};
constant constexpr int NRB         = {nrb};
constant constexpr int NUM_THREADS = {num_threads};
constant constexpr int A_WORK_ITEMS = {a_work_items};

kernel void ternary_dequant_probe(
    device const uint*  pos_bits_u32     [[buffer(0)]],
    device const uint*  neg_bits_u32     [[buffer(1)]],
    device const float* group_scale_f32  [[buffer(2)]],
    device const float* input_batch      [[buffer(3)]],
    device float*       output_batch     [[buffer(4)]],
    constant uint&      blocks64         [[buffer(5)]],
    constant uint&      groups_per_row   [[buffer(6)]],
    constant uint&      k_input_dim      [[buffer(7)]],
    constant uint&      m_features       [[buffer(8)]],
    constant uint&      p_tokens         [[buffer(9)]],
    constant uint&      tiles_x          [[buffer(10)]],
    threadgroup char*   shmem            [[threadgroup(0)]],
    uint3               tgpig            [[threadgroup_position_in_grid]],
    ushort              tiitg            [[thread_index_in_threadgroup]]
) {{
    const int ra = tgpig.y * NRA;
    threadgroup half* sa = (threadgroup half*)(shmem);
    const uint words_per_row = blocks64 * 2u;
    half acc = 0;

    for (int loop_k = 0; loop_k < (int)k_input_dim; loop_k += K_TILE) {{
        for (int work = tiitg; work < A_WORK_ITEMS; work += NUM_THREADS) {{
            const int row_local = work / NK_CHUNK;
            const int k_chunk   = work % NK_CHUNK;
            const int k_base    = k_chunk * SZ_SIMD;
            const int k_pos     = loop_k + k_base;
            const int row_global = ra + row_local;

            if (row_global < (int)m_features && k_pos + SZ_SIMD <= (int)k_input_dim) {{
                const uint word_idx = (uint)row_global * words_per_row + (uint)(k_pos / 32);
                const uint shift    = (uint)(k_pos % 32);
                const uint pw       = pos_bits_u32[word_idx] >> shift;
                const uint nw       = neg_bits_u32[word_idx] >> shift;
                const float scale   = group_scale_f32[(uint)row_global * groups_per_row + (uint)(k_pos / 128)];
                threadgroup half4* dst =
                    (threadgroup half4*)(sa + (size_t)k_base + (size_t)row_local * K_TILE);
                #pragma clang loop unroll(full)
                for (short q = 0; q < SZ_SIMD / 4; ++q) {{
                    half4 v;
                    #pragma clang loop unroll(full)
                    for (short j = 0; j < 4; ++j) {{
                        const uint b = (uint)(q * 4 + j);
                        const float sign = (float)((pw >> b) & 1u) - (float)((nw >> b) & 1u);
                        v[j] = (half)(sign * scale);
                    }}
                    dst[q] = v;
                }}
            }} else {{
                #pragma clang loop unroll(full)
                for (short i = 0; i < SZ_SIMD; ++i) {{
                    const int col = k_pos + i;
                    half val = 0.0h;
                    if (row_global < (int)m_features && col < (int)k_input_dim) {{
                        const uint word_idx = (uint)row_global * words_per_row + (uint)(col / 32);
                        const uint bit_pos  = (uint)(col % 32);
                        const uint pos_bit  = (pos_bits_u32[word_idx] >> bit_pos) & 1u;
                        const uint neg_bit  = (neg_bits_u32[word_idx] >> bit_pos) & 1u;
                        const float sign    = (float)pos_bit - (float)neg_bit;
                        const float scale   = group_scale_f32[(uint)row_global * groups_per_row + (uint)(col / 128)];
                        val = (half)(sign * scale);
                    }}
                    sa[(size_t)(k_base + i) + (size_t)row_local * K_TILE] = val;
                }}
            }}
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Liveness consume — one staged slot per thread per tile. Cheap
        // (128 smem reads), but structurally live: every acc feeds the
        // epilogue store, so no Phase-1 store can be eliminated.
        acc += sa[tiitg];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    threadgroup half* red = (threadgroup half*)(shmem);
    red[tiitg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiitg == 0) {{
        half total = 0;
        for (ushort i = 0; i < NUM_THREADS; ++i) total += red[i];
        output_batch[(uint)tgpig.y * tiles_x + tgpig.x] = (float)total;
    }}
}}
"#,
        sz_simd = sz_simd,
        nk_chunk = cfg.nk_chunk,
        nra = cfg.nra,
        nrb = cfg.nrb,
        num_threads = num_threads,
        a_work_items = a_work_items,
    )
}

/// The five buffer bindings both arms share, uploaded once per (shape, P)
/// and reused across launches.
struct GemmIo<'a> {
    pos: &'a Buffer,
    neg: &'a Buffer,
    scale: &'a Buffer,
    input: &'a Buffer,
    output: &'a Buffer,
}

/// The measurement-only pipeline. Buffer bindings, scalar bindings,
/// threadgroup memory, grid and threadgroup size all mirror
/// `MetalTensorGemm::launch` so the launch overhead is shared.
struct DequantProbe {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    config: MetalTileConfig,
}

impl DequantProbe {
    fn new(config: MetalTileConfig) -> Result<Self, String> {
        let device = Device::system_default().ok_or("No Metal device found")?;
        let options = CompileOptions::new();
        let source = probe_source(config);
        let library = device
            .new_library_with_source(source.as_str(), &options)
            .map_err(|e| format!("probe MSL compilation failed: {e}"))?;
        let function = library
            .get_function("ternary_dequant_probe", None)
            .map_err(|e| format!("probe function not found: {e}"))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("probe pipeline creation failed: {e}"))?;
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            pipeline,
            config,
        })
    }

    fn upload_weights(&self, fx: &TernaryFixture) -> (Buffer, Buffer, Buffer) {
        let mk = |data: &[u32]| {
            self.device.new_buffer_with_data(
                data.as_ptr() as *const core::ffi::c_void,
                (data.len() * 4) as u64,
                MTLResourceOptions::StorageModeManaged,
            )
        };
        let scale = self.device.new_buffer_with_data(
            fx.scale_f32.as_ptr() as *const core::ffi::c_void,
            (fx.scale_f32.len() * 4) as u64,
            MTLResourceOptions::StorageModeManaged,
        );
        (mk(&fx.pos_u32), mk(&fx.neg_u32), scale)
    }

    fn sentinel_output(&self, len: usize) -> Buffer {
        self.device.new_buffer_with_data(
            vec![SENTINEL; len].as_ptr() as *const core::ffi::c_void,
            (len * 4) as u64,
            MTLResourceOptions::StorageModeManaged,
        )
    }

    fn upload_f32(&self, data: &[f32]) -> Buffer {
        self.device.new_buffer_with_data(
            data.as_ptr() as *const core::ffi::c_void,
            (data.len() * 4) as u64,
            MTLResourceOptions::StorageModeManaged,
        )
    }

    fn read_f32(&self, buf: &Buffer, len: usize) -> Vec<f32> {
        let ptr = buf.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
    }

    /// Mirror of the SERVING launch path (the zero-copy branch's raw
    /// encoder work): buffer bindings, scalar bindings, threadgroup memory,
    /// grid and threadgroup size — but with `wait_until_completed` fencing
    /// so a timed region cannot exclude the work. Serves BOTH the
    /// in-situ arms and their fenced-wall comparison baseline.
    fn launch(&self, io: &GemmIo, fx: &TernaryFixture, p: usize) {
        let cfg = self.config;
        let tiles_x = p.div_ceil(cfg.nrb as usize) as u32;
        let grid = MTLSize {
            width: tiles_x as u64,
            height: fx.rows.div_ceil(cfg.nra as usize) as u64,
            depth: 1,
        };
        let threadgroup = MTLSize {
            width: u64::from(32 * cfg.nsg),
            height: 1,
            depth: 1,
        };
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipeline);
        enc.set_buffer(0, Some(io.pos), 0);
        enc.set_buffer(1, Some(io.neg), 0);
        enc.set_buffer(2, Some(io.scale), 0);
        enc.set_buffer(3, Some(io.input), 0);
        enc.set_buffer(4, Some(io.output), 0);
        enc.set_bytes(
            5,
            4,
            (fx.blocks64 as u32).to_ne_bytes().as_ptr() as *const _,
        );
        enc.set_bytes(
            6,
            4,
            (fx.groups_per_row as u32).to_ne_bytes().as_ptr() as *const _,
        );
        enc.set_bytes(7, 4, (fx.cols as u32).to_ne_bytes().as_ptr() as *const _);
        enc.set_bytes(8, 4, (fx.rows as u32).to_ne_bytes().as_ptr() as *const _);
        enc.set_bytes(9, 4, (p as u32).to_ne_bytes().as_ptr() as *const _);
        enc.set_bytes(10, 4, tiles_x.to_ne_bytes().as_ptr() as *const _);
        enc.set_threadgroup_memory_length(0, cfg.threadgroup_mem_bytes());
        for b in [io.pos, io.neg, io.scale, io.input] {
            enc.use_resource(b, MTLResourceUsage::Read);
        }
        enc.use_resource(io.output, MTLResourceUsage::Write);
        enc.dispatch_thread_groups(grid, threadgroup);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
}

// ---------------------------------------------------------------------------
// In-situ ablation twins — the REAL serving source with one element removed
// ---------------------------------------------------------------------------

/// Which single element the twin removes (the result is WRONG and that is
/// fine: correctness is irrelevant when profiling — the wall is still valid,
/// and no twin result is ever read).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TwinKind {
    /// The exact serving kernel (both barriers, real dequant) — the fenced
    /// baseline the exposed-share deltas price against.
    Baseline,
    /// (a) both per-K-tile barriers removed — prices the barrier+
    /// serialization exposure in situ.
    NoBarriers,
    /// (b) the A-side dequant sources replaced by constants (no packed-bit
    /// loads, the sign math stays) — prices the dequant LOAD+math exposure.
    ConstDequant,
    /// (c) the staged-tile stores removed (matmul reads stale threadgroup
    /// memory) — prices the stage-WRITE exposure.
    NoStage,
    /// (d) G4 — the A-side read+decode replaced by the packed-2-bit body:
    /// ONE u32 load per 16-element work item (16× 2-bit codes), the decode
    /// algebra `(code&1) - ((code>>1)&1)` — mathematically identical to
    /// `(pos_bit - neg_bit)`, so the twin is CORRECT and its delta vs the
    /// baseline is the ADDRESSABLE packed-format price (loads+ALU), same
    /// stores, same barriers, same grid. Buffer 0 carries the packed words
    /// (half the bit-plane bytes); buffer 1 stays bound (its fast path never
    /// reads it; the ragged fallback would misread it and is DEAD at the
    /// whole-word timing shapes — `cols % 64 == 0` keeps every work item on
    /// the V2 fast path, asserted by the fixture builder).
    PackedBody,
}

/// The exact string each twin patches, with the occurrence count the patch
/// must consume — a drift in the serving source fails LOUD at twin
/// construction, never as a silently-mis-patched timing row.
///
/// ⛔ Every find is anchored with its leading `\n` + exact indentation so it
/// matches ONLY the incumbent loop's occurrence — the Issue-1167 rung-C
/// double-buffer lambda (`#if SWEEP_DOUBLE_BUFFER`, preprocessor-dead at the
/// default 0) duplicates the same statements one level deeper, and the
/// un-anchored finds drifted red the day `b490a9f` landed. The lambda's
/// copies are compiled out at the serving posture; never patch them.
const PATCH_INCUMBENT_FASTPATH_LOADS: &[(&str, &str, usize)] = &[
    (
        "\n                const uint pw       = pos_bits_u32[word_idx] >> shift;",
        "\n                const uint pw       = 0xAAAAAAAAu;",
        1,
    ),
    (
        "\n                const uint nw       = neg_bits_u32[word_idx] >> shift;",
        "\n                const uint nw       = 0u;",
        1,
    ),
    (
        "\n                const float scale   = group_scale_f32[(uint)row_global * groups_per_row + (uint)(k_pos / 128)];",
        "\n                const float scale   = 1.0f;",
        1,
    ),
];
const PATCH_PACKED_BODY: &[(&str, &str, usize)] = &[
    (
        "const uint words_per_row = blocks64 * 2u;  // u32 words per weight row",
        "const uint words_per_row = blocks64 * 2u;  // u32 words per weight row\n    const uint packed_words_per_row = words_per_row * 2u;  // packed codes: 16 per u32",
        1,
    ),
    (
        "\n                const uint word_idx = (uint)row_global * words_per_row + (uint)(k_pos / 32);\n                const uint shift    = (uint)(k_pos % 32);\n                const uint pw       = pos_bits_u32[word_idx] >> shift;\n                const uint nw       = neg_bits_u32[word_idx] >> shift;\n                const float scale   = group_scale_f32[(uint)row_global * groups_per_row + (uint)(k_pos / 128)];",
        "\n                const uint packed_idx = (uint)row_global * packed_words_per_row + (uint)(k_pos / 16);\n                const uint packed_w   = pos_bits_u32[packed_idx];\n                const float scale     = group_scale_f32[(uint)row_global * groups_per_row + (uint)(k_pos / 128)];",
        1,
    ),
    (
        "\n                        const float sign = (float)((pw >> b) & 1u) - (float)((nw >> b) & 1u);",
        "\n                        const uint code     = (packed_w >> (b * 2u)) & 3u;\n                        const float sign = (float)(code & 1u) - (float)((code >> 1u) & 1u);",
        1,
    ),
];
const PATCH_NO_BARRIERS: &[(&str, &str, usize)] = &[
    (
        "\n        threadgroup_barrier(mem_flags::mem_threadgroup);\n\n        // === PHASE 2: Cooperative tensor matmul ===",
        "\n\n        // === PHASE 2: Cooperative tensor matmul ===",
        1,
    ),
    (
        "mm.run(mB, mA, cT);\n\n        threadgroup_barrier(mem_flags::mem_threadgroup);\n    }\n#endif // SWEEP_DOUBLE_BUFFER",
        "mm.run(mB, mA, cT);\n    }\n#endif // SWEEP_DOUBLE_BUFFER",
        1,
    ),
];
const PATCH_NO_STAGE: &[(&str, &str, usize)] = &[
    ("\n                    dst[q] = v;", "", 1),
    (
        "sa[(size_t)(k_base + i) + (size_t)row_local * K_TILE] = val;",
        "",
        1,
    ),
];

fn patches_for(kind: TwinKind) -> &'static [(&'static str, &'static str, usize)] {
    match kind {
        TwinKind::Baseline => &[],
        TwinKind::NoBarriers => PATCH_NO_BARRIERS,
        TwinKind::ConstDequant => PATCH_INCUMBENT_FASTPATH_LOADS,
        TwinKind::NoStage => PATCH_NO_STAGE,
        TwinKind::PackedBody => PATCH_PACKED_BODY,
    }
}

/// A compiled in-situ twin: the serving `gemm_ternary_tensor` pipeline at a
/// patched source. Buffer bindings, scalars, threadgroup memory, grid and
/// threadgroup size mirror `MetalTensorGemm::launch` exactly.
struct ServingTwin {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    config: MetalTileConfig,
}

impl ServingTwin {
    fn new(config: MetalTileConfig, kind: TwinKind) -> Result<Self, String> {
        let serving = include_str!("../src/gemm_ternary_metal_tensor.metal");
        let mut source = crate_defines(config);
        source.push_str(&apply_patches(serving, patches_for(kind)));
        let device = Device::system_default().ok_or("No Metal device found")?;
        let library = device
            .new_library_with_source(source.as_str(), &CompileOptions::new())
            .map_err(|e| format!("twin {kind:?} MSL compilation failed: {e}"))?;
        let function = library
            .get_function("gemm_ternary_tensor", None)
            .map_err(|e| format!("twin {kind:?}: serving function not found: {e}"))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("twin {kind:?} pipeline creation failed: {e}"))?;
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            pipeline,
            config,
        })
    }

    fn upload_weights(&self, fx: &TernaryFixture) -> (Buffer, Buffer, Buffer) {
        self.upload_weights_for(fx, TwinKind::Baseline)
    }

    /// The PackedBody twin's buffer 0 is the packed code array (half the
    /// bit-plane bytes); its fast path never reads buffer 1, which stays
    /// bound for the dead ragged fallback and the use_resource set.
    fn upload_weights_for(&self, fx: &TernaryFixture, kind: TwinKind) -> (Buffer, Buffer, Buffer) {
        let mk = |data: &[u32]| {
            self.device.new_buffer_with_data(
                data.as_ptr() as *const core::ffi::c_void,
                (data.len() * 4) as u64,
                MTLResourceOptions::StorageModeManaged,
            )
        };
        let scale = self.device.new_buffer_with_data(
            fx.scale_f32.as_ptr() as *const core::ffi::c_void,
            (fx.scale_f32.len() * 4) as u64,
            MTLResourceOptions::StorageModeManaged,
        );
        let pos = if kind == TwinKind::PackedBody {
            mk(&fx.packed_u32)
        } else {
            mk(&fx.pos_u32)
        };
        (pos, mk(&fx.neg_u32), scale)
    }

    fn read_f32(&self, buf: &Buffer, len: usize) -> Vec<f32> {
        let ptr = buf.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
    }

    fn upload_f32(&self, data: &[f32]) -> Buffer {
        self.device.new_buffer_with_data(
            data.as_ptr() as *const core::ffi::c_void,
            (data.len() * 4) as u64,
            MTLResourceOptions::StorageModeManaged,
        )
    }

    fn launch(&self, io: &GemmIo, fx: &TernaryFixture, p: usize) {
        let cfg = self.config;
        let grid = MTLSize {
            width: p.div_ceil(cfg.nrb as usize) as u64,
            height: fx.rows.div_ceil(cfg.nra as usize) as u64,
            depth: 1,
        };
        let threadgroup = MTLSize {
            width: u64::from(32 * cfg.nsg),
            height: 1,
            depth: 1,
        };
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipeline);
        enc.set_buffer(0, Some(io.pos), 0);
        enc.set_buffer(1, Some(io.neg), 0);
        enc.set_buffer(2, Some(io.scale), 0);
        enc.set_buffer(3, Some(io.input), 0);
        enc.set_buffer(4, Some(io.output), 0);
        enc.set_bytes(
            5,
            4,
            (fx.blocks64 as u32).to_ne_bytes().as_ptr() as *const _,
        );
        enc.set_bytes(
            6,
            4,
            (fx.groups_per_row as u32).to_ne_bytes().as_ptr() as *const _,
        );
        enc.set_bytes(7, 4, (fx.cols as u32).to_ne_bytes().as_ptr() as *const _);
        enc.set_bytes(8, 4, (fx.rows as u32).to_ne_bytes().as_ptr() as *const _);
        enc.set_bytes(9, 4, (p as u32).to_ne_bytes().as_ptr() as *const _);
        enc.set_threadgroup_memory_length(0, cfg.threadgroup_mem_bytes());
        for b in [io.pos, io.neg, io.scale, io.input] {
            enc.use_resource(b, MTLResourceUsage::Read);
        }
        enc.use_resource(io.output, MTLResourceUsage::Write);
        enc.dispatch_thread_groups(grid, threadgroup);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }
}

/// The geometry + posture defines the serving contexts compile with — the
/// SAME composition `compose_msl_source` produces at the promoted posture
/// (relaxed is read from the module's own selector so the twin's matmul
/// precision class matches serving; TWIN_RELAXED env overrides for the
/// demote arm).
fn crate_defines(config: MetalTileConfig) -> String {
    let relaxed = match std::env::var("TWIN_RELAXED").as_deref() {
        Ok("0") => false,
        _ => riir_infer_gpu::gemm_ternary_metal_tensor::production_relaxed_precision(),
    };
    let relaxed_line = if relaxed {
        "#define SWEEP_RELAXED true\n"
    } else {
        ""
    };
    format!("{}{relaxed_line}", config.defines())
}

/// Apply the named patches to the serving source. Patches are (find,
/// replace, expected-count) — an unexpected count fails loud.
fn apply_patches(source: &str, patches: &[(&str, &str, usize)]) -> String {
    let mut out = source.to_string();
    for (find, replace, want) in patches.iter() {
        let got = out.matches(find).count();
        assert_eq!(
            got, *want,
            "twin patch drifted: {find:?} occurs {got}x, the patch expects {want} — the \
             serving source moved and this instrument must be re-read"
        );
        out = out.replace(find, replace);
    }
    out
}

// ---------------------------------------------------------------------------
// PROVENANCE (the box-state law — print, never gate)
// ---------------------------------------------------------------------------

fn loadavg_1m() -> Option<f64> {
    let out = std::process::Command::new("sysctl")
        .arg("-n")
        .arg("vm.loadavg")
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    s.split_whitespace().nth(1)?.parse().ok()
}

fn print_provenance(when: &str) {
    match loadavg_1m() {
        Some(load) => {
            println!("PROVENANCE ({when}): loadavg1m={load:.2}");
            if load > 6.0 {
                println!(
                    "⚠ LOADED BOX (load {load:.2} > 6.0): the TIMING rows below are NOT \
                     quotable — G1 rows remain valid."
                );
            }
        }
        None => println!("PROVENANCE ({when}): loadavg unavailable"),
    }
}

// ---------------------------------------------------------------------------
// G1 — the incumbent pipeline must match the CPU decode; the probe must write
// every tile (liveness, the loud-zero defence)
// ---------------------------------------------------------------------------

/// (rows, cols, p) — ragged coverage on every axis (edge tiles exercise the
/// probe's fallback body), plus one mid projection.
const G1_CASES: &[(usize, usize, usize)] =
    &[(130, 384, 9), (33, 192, 3), (256, 640, 16), (1024, 5120, 8)];

#[test]
#[ignore = "requires macOS Metal + a GPU; run explicitly with --release"]
fn g1_incumbent_and_probe_fixture_gates() {
    // The G1 bar is defined at the BIT-EXACT posture (relaxed off — the
    // 1e-5 scale-relative bar). relaxed_precision=true shortens matmul2d's
    // accumulation (fp16-class) — a ~5e-4-class DISTRIBUTION shift, exactly
    // what Bench 940's 2× precision cell measured on the dense lane. G1
    // pins the fixture+pipeline correctness, so it compiles the demote
    // posture explicitly — the module's explicit PIN (`set_metal_tensor_
    // relaxed_precision(false)`, the public setter, wins over the env for
    // contexts constructed after). G1 pins the fixture+pipeline correctness;
    // the G2/G3 timing cells pin their own compiled selector via
    // crate_defines and quote distribution-class tolerance, not G1 exactness.
    // (The pin is process-global — G1 and G2/G3 run as separate test
    // binaries/processes under --test-threads=1, and each G-run constructs
    // its own pipelines.) The double-buffer pin is ALSO pinned OFF here
    // explicitly: this test's arm label is "incumbent", and g1b (which pins
    // ON) may run first in the same process — the explicit pin makes the
    // label true under any construction order.
    riir_infer_gpu::gemm_ternary_metal_tensor::set_metal_tensor_relaxed_precision(false);
    riir_infer_gpu::gemm_ternary_metal_tensor::set_metal_tensor_double_buffer(false);
    let gemm = MetalTensorGemm::new_with_config(MetalTileConfig::KTILE64)
        .expect("incumbent pipeline must compile at the promoted posture");
    let probe = DequantProbe::new(MetalTileConfig::KTILE64)
        .expect("probe pipeline must compile at the promoted posture");

    for &(rows, cols, p) in G1_CASES {
        let fx = TernaryFixture::build(rows, cols, 0x1167 ^ rows as u64 ^ (cols as u64) << 8);
        let input = random_input(p * cols, 0xBEEF ^ p as u64);
        let want = fx.cpu_reference(&input, p);

        // Incumbent arm — scale-relative 1e-5 against the CPU decode.
        let (pos, neg, scale) = gemm.upload_weights(&fx.pos_u32, &fx.neg_u32, &fx.scale_f32);
        let input_buf = gemm.upload_f32(&input);
        let output_buf = gemm.create_output(p * rows);
        gemm.launch(
            &pos,
            &neg,
            &scale,
            &input_buf,
            &output_buf,
            fx.blocks64 as u32,
            fx.groups_per_row as u32,
            fx.cols as u32,
            fx.rows as u32,
            p as u32,
        );
        let got = gemm.read_f32(&output_buf, p * rows);
        let ref_scale = want.iter().fold(0.0f32, |a, b| a.max(b.abs())).max(1e-6);
        let mut worst = (0.0f64, 0usize);
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let rel = ((g - w).abs() / ref_scale) as f64;
            if rel > worst.0 {
                worst = (rel, i);
            }
        }
        println!(
            "  G1 incumbent {rows}x{cols} p={p}: worst_rel={:.3e} at [{}] {}",
            worst.0,
            worst.1,
            if worst.0 < G1_REL_LIMIT {
                "PASS"
            } else {
                "FAIL"
            }
        );
        assert!(
            worst.0 < G1_REL_LIMIT,
            "incumbent arm G1 FAILED at {rows}x{cols} p={p} (worst_rel={:.3e} at [{}] >= {G1_REL_LIMIT})",
            worst.0,
            worst.1
        );

        // Probe arm — every tile must store its consume (sentinel check).
        let tiles = (p.div_ceil(128)) * (rows.div_ceil(64));
        let (ppos, pneg, pscale) = probe.upload_weights(&fx);
        let pinput = probe.upload_f32(&input);
        let poutput = probe.sentinel_output(tiles);
        let pio = GemmIo {
            pos: &ppos,
            neg: &pneg,
            scale: &pscale,
            input: &pinput,
            output: &poutput,
        };
        probe.launch(&pio, &fx, p);
        let pgot = probe.read_f32(&poutput, tiles);
        for (i, g) in pgot.iter().enumerate() {
            assert_ne!(
                *g, SENTINEL,
                "probe tile {i} at {rows}x{cols} p={p} never wrote — the consume is dead \
                 and the probe prices nothing"
            );
            assert!(g.is_finite(), "probe tile {i} wrote a non-finite consume");
        }
        println!("  G1 probe {rows}x{cols} p={p}: {tiles}/{tiles} tiles live PASS");
    }
}

// ---------------------------------------------------------------------------
// G1b — the double-buffer staging variant (Issue 1167 rung C decider, gate 1):
// engagement + correctness + bit-identity, at the bit-exact posture
// ---------------------------------------------------------------------------

/// The decider's G1 gate for the PROMOTE-ONLY `RIIR_METAL_TENSOR_DOUBLE_BUFFER`
/// variant (the kernel's `SWEEP_DOUBLE_BUFFER` block, landed `c1bbbeb`):
///
/// 1. **Engagement** — the composed MSL carries the define, the context
///    reports itself double-buffered, and the context's smem sizing is the
///    doubled tile (the announce alone is not a proof the compiled source
///    carried the define — the Issue-1124 T1.5 lesson).
/// 2. **Correctness** — the variant at the bit-exact posture (relaxed OFF)
///    meets the same 1e-5 scale-relative bar vs the CPU decode on the SAME
///    ragged cases as [`g1_incumbent_and_probe_fixture_gates`].
/// 3. **Bit-identity** — variant output == incumbent output, exactly, per
///    element (same dequant arithmetic, same tile contents, same matmul
///    sequence — only the staging is re-timed). This is the variant's G1
///    form the kernel doc names; a tolerance here would be a lossy-surface
///    claim, which rung C is NOT.
///
/// Pin-phase discipline: ALL variant work (construction + launches) runs
/// while the pin is ON, then the incumbent phase runs with the pin OFF —
/// each context captures its own posture at construction, so a launch can
/// never disagree with its pipeline's smem need (the under-allocation
/// footgun the capture exists to close).
#[test]
#[ignore = "requires macOS Metal + a GPU; run explicitly with --release"]
fn g1b_double_buffer_variant_gate() {
    use riir_infer_gpu::gemm_ternary_metal_tensor::compose_msl_source;

    let cfg = MetalTileConfig::KTILE64;

    // ── Phase A: the variant (pin ON, bit-exact relaxed OFF) ──
    riir_infer_gpu::gemm_ternary_metal_tensor::set_metal_tensor_relaxed_precision(false);
    riir_infer_gpu::gemm_ternary_metal_tensor::set_metal_tensor_double_buffer(true);

    // Engagement at the source level: the one composition every compile site
    // rides must carry the define.
    let composed = compose_msl_source(cfg, "");
    assert!(
        composed.contains("#define SWEEP_DOUBLE_BUFFER 1"),
        "engagement: compose_msl_source did not carry SWEEP_DOUBLE_BUFFER 1 — \
         the variant pipeline would silently compile the incumbent body"
    );
    let single_tile = cfg.threadgroup_mem_bytes();

    let variant = MetalTensorGemm::new_with_config(cfg)
        .expect("double-buffer variant pipeline must compile at the promoted posture");
    assert!(
        variant.double_buffered(),
        "engagement: context built with the pin ON must report double_buffered"
    );
    assert_eq!(
        variant.threadgroup_mem_bytes(),
        single_tile * 2,
        "engagement: a double-buffer context sizes smem for TWO tiles"
    );

    let mut variant_outputs: Vec<Vec<f32>> = Vec::new();
    for &(rows, cols, p) in G1_CASES {
        let fx = TernaryFixture::build(rows, cols, 0x1167 ^ rows as u64 ^ (cols as u64) << 8);
        let input = random_input(p * cols, 0xBEEF ^ p as u64);
        let want = fx.cpu_reference(&input, p);

        let (pos, neg, scale) = variant.upload_weights(&fx.pos_u32, &fx.neg_u32, &fx.scale_f32);
        let input_buf = variant.upload_f32(&input);
        let output_buf = variant.create_output(p * rows);
        variant.launch(
            &pos,
            &neg,
            &scale,
            &input_buf,
            &output_buf,
            fx.blocks64 as u32,
            fx.groups_per_row as u32,
            fx.cols as u32,
            fx.rows as u32,
            p as u32,
        );
        let got = variant.read_f32(&output_buf, p * rows);

        let ref_scale = want.iter().fold(0.0f32, |a, b| a.max(b.abs())).max(1e-6);
        let mut worst = (0.0f64, 0usize);
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let rel = ((g - w).abs() / ref_scale) as f64;
            if rel > worst.0 {
                worst = (rel, i);
            }
        }
        println!(
            "  G1b variant {rows}x{cols} p={p}: worst_rel={:.3e} at [{}] {}",
            worst.0,
            worst.1,
            if worst.0 < G1_REL_LIMIT {
                "PASS"
            } else {
                "FAIL"
            }
        );
        assert!(
            worst.0 < G1_REL_LIMIT,
            "variant G1 FAILED at {rows}x{cols} p={p} (worst_rel={:.3e} at [{}] >= {G1_REL_LIMIT})",
            worst.0,
            worst.1
        );
        variant_outputs.push(got);
    }

    // ── Phase B: the incumbent (pin OFF) — bit-identity vs the variant ──
    riir_infer_gpu::gemm_ternary_metal_tensor::set_metal_tensor_double_buffer(false);
    let incumbent = MetalTensorGemm::new_with_config(cfg)
        .expect("incumbent pipeline must compile at the promoted posture");
    assert!(
        !incumbent.double_buffered(),
        "the phase-B context must report the incumbent posture"
    );
    assert_eq!(incumbent.threadgroup_mem_bytes(), single_tile);

    for (case, &(rows, cols, p)) in G1_CASES.iter().enumerate() {
        let fx = TernaryFixture::build(rows, cols, 0x1167 ^ rows as u64 ^ (cols as u64) << 8);
        let input = random_input(p * cols, 0xBEEF ^ p as u64);

        let (pos, neg, scale) = incumbent.upload_weights(&fx.pos_u32, &fx.neg_u32, &fx.scale_f32);
        let input_buf = incumbent.upload_f32(&input);
        let output_buf = incumbent.create_output(p * rows);
        incumbent.launch(
            &pos,
            &neg,
            &scale,
            &input_buf,
            &output_buf,
            fx.blocks64 as u32,
            fx.groups_per_row as u32,
            fx.cols as u32,
            fx.rows as u32,
            p as u32,
        );
        let got = incumbent.read_f32(&output_buf, p * rows);

        let variant_out = &variant_outputs[case];
        let first_diff = got
            .iter()
            .zip(variant_out.iter())
            .position(|(a, b)| a.to_bits() != b.to_bits());
        assert!(
            first_diff.is_none(),
            "variant != incumbent at {rows}x{cols} p={p}: first differing element [{}] \
             (got {:?}, want {:?}) — the staging re-time moved a bit, which rung C \\
             does not own (lossy-surface walk would be required, never silent)",
            first_diff.unwrap_or(0),
            first_diff.map(|i| got[i]),
            first_diff.map(|i| variant_out[i]),
        );
        println!(
            "  G1b bit-identity {rows}x{cols} p={p}: variant == incumbent, {} elements EXACT",
            got.len()
        );
    }
}

// ---------------------------------------------------------------------------
// G2 — the interleaved price cell: incumbent vs dequant-only probe
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires macOS Metal + a GPU; run explicitly with --release"]
fn g2_staging_price_incumbent_vs_dequant_probe() {
    print_provenance("start");

    let shapes: Vec<(String, usize, usize)> = std::env::var("BENCH1167_SHAPES")
        .ok()
        .map(|v| {
            v.split(',')
                .map(|s| {
                    let (m, n) = s.split_once('x').expect("shape as MxN");
                    (format!("{m}x{n}"), m.parse().unwrap(), n.parse().unwrap())
                })
                .collect()
        })
        .unwrap_or_else(|| {
            DEFAULT_SHAPES
                .iter()
                .map(|&(n, m, k)| (n.to_string(), m, k))
                .collect()
        });
    let ps: Vec<usize> = std::env::var("BENCH1167_PS")
        .ok()
        .map(|v| v.split(',').map(|s| s.parse().unwrap()).collect())
        .unwrap_or_else(|| DEFAULT_PS.to_vec());
    let rounds: usize = std::env::var("BENCH1167_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let launches: usize = std::env::var("BENCH1167_LAUNCHES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);

    let cfg = MetalTileConfig::KTILE64;
    let gemm = MetalTensorGemm::new_with_config(cfg)
        .expect("incumbent pipeline must compile at the promoted posture");
    let probe = DequantProbe::new(cfg).expect("probe pipeline must compile");

    println!(
        "\nG2 price cell: posture {cfg:?} · rounds={rounds} launches={launches} · \
         buffers uploaded once per (shape, P), reused across launches"
    );

    for shape_ref in shapes.iter() {
        let (shape_name, m, n) = (shape_ref.0.as_str(), shape_ref.1, shape_ref.2);
        let fx = TernaryFixture::build(m, n, 0x1167 ^ m as u64 ^ (n as u64) << 8);
        let (pos, neg, scale) = gemm.upload_weights(&fx.pos_u32, &fx.neg_u32, &fx.scale_f32);
        let (ppos, pneg, pscale) = probe.upload_weights(&fx);

        for &p in &ps {
            let input = random_input(p * n, 0xBEEF);
            let input_buf = gemm.upload_f32(&input);
            let pinput = probe.upload_f32(&input);
            let output_buf = gemm.create_output(p * m);
            let tiles = p.div_ceil(cfg.nrb as usize) * m.div_ceil(cfg.nra as usize);
            let poutput = probe.sentinel_output(tiles);

            // Warm both arms once (JIT/pool growth lands here).
            gemm.launch(
                &pos,
                &neg,
                &scale,
                &input_buf,
                &output_buf,
                fx.blocks64 as u32,
                fx.groups_per_row as u32,
                fx.cols as u32,
                fx.rows as u32,
                p as u32,
            );
            probe.launch(
                &GemmIo {
                    pos: &ppos,
                    neg: &pneg,
                    scale: &pscale,
                    input: &pinput,
                    output: &poutput,
                },
                &fx,
                p,
            );

            // Probe liveness AFTER warm — the loud-zero defence on the arm
            // whose price is about to be quoted.
            for g in probe.read_f32(&poutput, tiles) {
                assert_ne!(g, SENTINEL, "probe tile never wrote at {shape_name} P={p}");
                assert!(g.is_finite(), "probe tile non-finite at {shape_name} P={p}");
            }

            // Interleaved rotating-start rounds over the two arms. Each
            // round launches the arms in an order rotated by the round
            // index; the result lands in the arm's own slot.
            let mut times = [[0.0f64; 2]; 8];
            let times = &mut times[..rounds.min(8)];
            for (r, row) in times.iter_mut().enumerate() {
                for first_arm in 0..2 {
                    let slot = (first_arm + r) % 2;
                    let t0 = std::time::Instant::now();
                    for _ in 0..launches {
                        if slot == 0 {
                            gemm.launch(
                                &pos,
                                &neg,
                                &scale,
                                &input_buf,
                                &output_buf,
                                fx.blocks64 as u32,
                                fx.groups_per_row as u32,
                                fx.cols as u32,
                                fx.rows as u32,
                                p as u32,
                            );
                        } else {
                            probe.launch(
                                &GemmIo {
                                    pos: &ppos,
                                    neg: &pneg,
                                    scale: &pscale,
                                    input: &pinput,
                                    output: &poutput,
                                },
                                &fx,
                                p,
                            );
                        }
                    }
                    let dt = t0.elapsed().as_secs_f64() / launches as f64;
                    assert!(
                        dt > 0.0,
                        "{shape_name} P={p} arm {slot}: zero-length timing"
                    );
                    row[slot] = dt;
                }
            }

            let median = |mut v: Vec<f64>| {
                v.sort_by(|a, b| a.total_cmp(b));
                v[v.len() / 2]
            };
            let inc = median(times.iter().map(|row| row[0]).collect());
            let probe_ms = median(times.iter().map(|row| row[1]).collect());
            let share = probe_ms / inc;
            let matmul_only = inc - probe_ms;
            let flops = 2.0 * (m as f64) * (n as f64) * (p as f64);
            let tflops = flops / matmul_only / 1e12;

            // H5-device-f16 traffic delta: the A-side bytes a no-staging
            // variant reads over the packed-bit read, per launch, at the
            // M5-class ~800 GB/s DRAM figure (a CLASS, not a measurement).
            let p_tiles = p.div_ceil(cfg.nrb as usize);
            let f16_bytes = (p_tiles * m * n * 2) as f64;
            let bits_bytes = (p_tiles * (m * n / 4 + m * fx.groups_per_row * 4)) as f64;
            let delta_gb = (f16_bytes - bits_bytes) / 1e9;
            let delta_ms = delta_gb / 0.8;

            println!(
                "\n── {shape_name} (m={m} k={n}) P={p} ──\n  incumbent   {:>9.3} ms/launch\n  \
                 dequant_probe {:>7.3} ms/launch  (share {:.1}%)\n  matmul+store {:>8.3} ms  \
                 (matmul-only {:.1} TFLOPS)\n  H5 traffic delta +{:.3} GB/launch ≈ +{:.3} ms \
                 at the 800 GB/s class",
                inc * 1e3,
                probe_ms * 1e3,
                share * 100.0,
                matmul_only * 1e3,
                tflops,
                delta_gb,
                delta_ms,
            );
        }
    }

    print_provenance("end");
}

// ---------------------------------------------------------------------------
// G3 — the in-situ ablation twins at the SERVING posture (default P=512):
// incumbent − twin = the EXPOSED share (what other workgroups do NOT hide)
// ---------------------------------------------------------------------------

/// P = 512 IS the serving shape: the census frame is `I1146C_FRAMES`×512
/// chunks and every GEMM launch inside it dispatches at P=512
/// (`chunk.div_ceil(nrb)` tiles). `BENCH1167_PS` widens for the record.
#[test]
#[ignore = "requires macOS Metal + a GPU; run explicitly with --release"]
fn g3_insitu_ablation_twins_at_serving_posture() {
    print_provenance("start");

    let shapes: Vec<(String, usize, usize)> = std::env::var("BENCH1167_SHAPES")
        .ok()
        .map(|v| {
            v.split(',')
                .map(|s| {
                    let (m, n) = s.split_once('x').expect("shape as MxN");
                    (format!("{m}x{n}"), m.parse().unwrap(), n.parse().unwrap())
                })
                .collect()
        })
        .unwrap_or_else(|| {
            DEFAULT_SHAPES
                .iter()
                .chain(GDN_SHAPES.iter())
                .map(|&(n, m, k)| (n.to_string(), m, k))
                .collect()
        });
    let ps: Vec<usize> = std::env::var("BENCH1167_PS")
        .ok()
        .map(|v| v.split(',').map(|s| s.parse().unwrap()).collect())
        .unwrap_or_else(|| vec![512]);
    let rounds: usize = std::env::var("BENCH1167_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let launches: usize = std::env::var("BENCH1167_LAUNCHES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);

    let cfg = MetalTileConfig::KTILE64;
    let twins = [
        (TwinKind::Baseline, "baseline"),
        (TwinKind::NoBarriers, "no_barriers"),
        (TwinKind::ConstDequant, "const_dequant"),
        (TwinKind::NoStage, "no_stage"),
    ];
    let rounds_rep: usize = std::env::var("BENCH1167_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let compiled: Vec<(TwinKind, &str, ServingTwin)> = twins
        .iter()
        .filter_map(|&(kind, name)| match ServingTwin::new(cfg, kind) {
            Ok(t) => Some((kind, name, t)),
            Err(e) => {
                println!("  TWIN {name} COMPILE_FAIL — {e}");
                None
            }
        })
        .collect();
    assert!(
        !compiled.is_empty(),
        "every twin failed to compile — the instrument calibrated nothing"
    );
    let relaxed = std::env::var("TWIN_RELAXED").as_deref() != Ok("0");
    println!(
        "\nG3 in-situ twins: {} compiled · relaxed_precision={relaxed} (serving selector) \
         · rounds={rounds} launches={launches} · buffers uploaded once per (shape, P)",
        compiled.len()
    );

    for shape_ref in shapes.iter() {
        let (shape_name, m, n) = (shape_ref.0.as_str(), shape_ref.1, shape_ref.2);
        let fx = TernaryFixture::build(m, n, 0x1167 ^ m as u64 ^ (n as u64) << 8);
        let weights: Vec<(Buffer, Buffer, Buffer)> = compiled
            .iter()
            .map(|(_, _, t)| t.upload_weights(&fx))
            .collect();

        for &p in &ps {
            let input = random_input(p * n, 0xBEEF);
            let inputs: Vec<Buffer> = compiled
                .iter()
                .map(|(_, _, t)| t.upload_f32(&input))
                .collect();
            let outputs: Vec<Buffer> = compiled
                .iter()
                .map(|(_, _, t)| {
                    t.device
                        .new_buffer((p * m * 4) as u64, MTLResourceOptions::StorageModeManaged)
                })
                .collect();

            // Warm every arm once.
            for (ai, (_, _, t)) in compiled.iter().enumerate() {
                let io = GemmIo {
                    pos: &weights[ai].0,
                    neg: &weights[ai].1,
                    scale: &weights[ai].2,
                    input: &inputs[ai],
                    output: &outputs[ai],
                };
                t.launch(&io, &fx, p);
            }

            let n_arms = compiled.len();
            let mut times = vec![vec![0.0f64; n_arms]; rounds * rounds_rep];
            for (r, row) in times.iter_mut().enumerate() {
                for k in 0..n_arms {
                    let ai = (r + k) % n_arms;
                    let (_, _, t) = &compiled[ai];
                    let io = GemmIo {
                        pos: &weights[ai].0,
                        neg: &weights[ai].1,
                        scale: &weights[ai].2,
                        input: &inputs[ai],
                        output: &outputs[ai],
                    };
                    let t0 = std::time::Instant::now();
                    for _ in 0..launches {
                        t.launch(&io, &fx, p);
                    }
                    let dt = t0.elapsed().as_secs_f64() / launches as f64;
                    assert!(dt > 0.0, "{shape_name} P={p} twin {ai}: zero-length timing");
                    row[ai] = dt;
                }
            }

            let median = |mut v: Vec<f64>| {
                v.sort_by(|a, b| a.total_cmp(b));
                v[v.len() / 2]
            };
            // Wilson-style spread proxy: min/max of the per-round medians of
            // each repetition block (BENCH1167_REPS>1 splits the rounds into
            // blocks and medians each — the delta CI class the bar needs).
            let meds: Vec<f64> = (0..n_arms)
                .map(|ai| median(times.iter().map(|row| row[ai]).collect()))
                .collect();
            let base = meds[0];
            let lo_hi = |ai: usize| -> (f64, f64) {
                let vals: Vec<f64> = times.iter().map(|row| row[ai]).collect();
                let mut v = vals.clone();
                v.sort_by(|a, b| a.total_cmp(b));
                let q = |f: f64| -> f64 {
                    let idx = ((v.len() as f64) * f) as usize;
                    v[idx.min(v.len() - 1)]
                };
                (q(0.25), q(0.75))
            };
            println!(
                "\n── {shape_name} (m={m} k={n}) P={p} — serving twin baseline {:.3} ms/launch \
                 (n={} rounds; IQR in brackets) ──",
                base * 1e3,
                times.len()
            );
            for (ai, (_, name, _)) in compiled.iter().enumerate() {
                let med = meds[ai];
                let exposed = base - med;
                let (b_lo, b_hi) = lo_hi(0);
                let (t_lo, t_hi) = lo_hi(ai);
                // the delta's spread: conservative pairwise bounds
                let d_lo = b_lo - t_hi;
                let d_hi = b_hi - t_lo;
                println!(
                    "  {name:<13} {:>9.3} ms/launch   exposed (baseline − twin) {:>7.3} ms  \
                     ({:>5.1}% of wall)  Δ-IQR [{:.3}, {:.3}] ms",
                    med * 1e3,
                    exposed * 1e3,
                    exposed / base * 100.0,
                    d_lo * 1e3,
                    d_hi * 1e3
                );
            }
        }
    }

    print_provenance("end");
}

// ---------------------------------------------------------------------------
// G4 — the packed-byte dequant price at the serving posture (Issue 1167
// rung B's named surviving ground). The PackedBody twin is CORRECT — its
// decode algebra reproduces `(pos_bit - neg_bit)` exactly — so
// baseline − packed is the ADDRESSABLE load+ALU delta in situ, not an
// ablation bound, and the bit-identity check is the correctness anchor a
// future packed serving kernel inherits for free.
// ---------------------------------------------------------------------------

/// Per-chunk GEMM launches the frame projection weights (the Bench-937
/// census posture: `I1146C_CHUNK=512` chunks, `I1146C_FRAMES=8` per frame —
/// every launch inside a chunk dispatches at P=512, so launches/frame =
/// 8 × the per-chunk set below; the attn_o projection 5120×4096 is NOT in
/// the shape set and is disclosed uncovered, same class as gdn_out).
const FRAME_LAUNCHES_PER_CHUNK: &[(&str, usize)] = &[
    ("ffn_gate/up", 128), // 2 × 64 layers
    ("ffn_down", 64),
    ("attn_q", 16), // the fused qkv projection, 16 attn layers
    ("gdn_qkv", 48),
    ("gdn_z", 48),
    ("gdn_ab", 96), // a and b, two 16×5120 projections × 48 layers
    ("gdn_out", 48),
];

#[test]
#[ignore = "requires macOS Metal + a GPU; run explicitly with --release"]
fn g4_packed_body_price_at_serving_posture() {
    print_provenance("start");

    let shapes: Vec<(String, usize, usize)> = std::env::var("BENCH1167_SHAPES")
        .ok()
        .map(|v| {
            v.split(',')
                .map(|s| {
                    let (m, n) = s.split_once('x').expect("shape as MxN");
                    (format!("{m}x{n}"), m.parse().unwrap(), n.parse().unwrap())
                })
                .collect()
        })
        .unwrap_or_else(|| {
            DEFAULT_SHAPES
                .iter()
                .chain(GDN_SHAPES.iter())
                .map(|&(n, m, k)| (n.to_string(), m, k))
                .collect()
        });
    // Default P=512 — the serving chunk posture the projection weights.
    let ps: Vec<usize> = std::env::var("BENCH1167_PS")
        .ok()
        .map(|v| v.split(',').map(|s| s.parse().unwrap()).collect())
        .unwrap_or_else(|| vec![512]);
    let rounds: usize = std::env::var("BENCH1167_ROUNDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let launches: usize = std::env::var("BENCH1167_LAUNCHES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    let frame_ref_ms: f64 = std::env::var("BENCH1167_FRAME_REF_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(500.86);

    let cfg = MetalTileConfig::KTILE64;
    let twins = [(TwinKind::Baseline, "baseline"), (TwinKind::PackedBody, "packed")];
    let compiled: Vec<(TwinKind, &str, ServingTwin)> = twins
        .iter()
        .filter_map(|&(kind, name)| match ServingTwin::new(cfg, kind) {
            Ok(t) => Some((kind, name, t)),
            Err(e) => {
                println!("  TWIN {name} COMPILE_FAIL — {e}");
                None
            }
        })
        .collect();
    assert_eq!(
        compiled.len(),
        2,
        "both arms must compile — the packed price needs its baseline"
    );
    let relaxed = std::env::var("TWIN_RELAXED").as_deref() != Ok("0");
    println!(
        "\nG4 packed-body price: 2 arms · relaxed_precision={relaxed} (serving selector) \
         · rounds={rounds} launches={launches} · buffers uploaded once per (shape, P)"
    );

    // Per-P projection accumulators (ms/frame): Σ delta × launches/chunk × 8.
    let mut projection = vec![0.0f64; ps.len()];

    for shape_ref in shapes.iter() {
        let (shape_name, m, n) = (shape_ref.0.as_str(), shape_ref.1, shape_ref.2);
        let fx = TernaryFixture::build(m, n, 0x1167 ^ m as u64 ^ (n as u64) << 8);
        let weights: Vec<(Buffer, Buffer, Buffer)> = compiled
            .iter()
            .map(|(kind, _, t)| t.upload_weights_for(&fx, *kind))
            .collect();

        for (pi, &p) in ps.iter().enumerate() {
            let input = random_input(p * n, 0xBEEF);
            let inputs: Vec<Buffer> = compiled
                .iter()
                .map(|(_, _, t)| t.upload_f32(&input))
                .collect();
            let outputs: Vec<Buffer> = compiled
                .iter()
                .map(|(_, _, t)| {
                    t.device
                        .new_buffer((p * m * 4) as u64, MTLResourceOptions::StorageModeManaged)
                })
                .collect();

            // Warm every arm once.
            for (ai, (_, _, t)) in compiled.iter().enumerate() {
                let io = GemmIo {
                    pos: &weights[ai].0,
                    neg: &weights[ai].1,
                    scale: &weights[ai].2,
                    input: &inputs[ai],
                    output: &outputs[ai],
                };
                t.launch(&io, &fx, p);
            }

            // The packed twin is CORRECT by algebra — assert it against the
            // serving baseline bit-for-bit. A mismatch is a packed-decode
            // defect, never a timing row.
            {
                let base = compiled[0].2.read_f32(&outputs[0], p * m);
                let packed = compiled[1].2.read_f32(&outputs[1], p * m);
                let flip = base
                    .iter()
                    .zip(packed.iter())
                    .position(|(a, b)| a.to_bits() != b.to_bits());
                assert!(
                    flip.is_none(),
                    "{shape_name} P={p}: packed twin diverged from the serving baseline \
                     at element {:?} (packed decode must be bit-identical)",
                    flip
                );
            }

            let n_arms = compiled.len();
            let mut times = vec![vec![0.0f64; n_arms]; rounds];
            for (r, row) in times.iter_mut().enumerate() {
                for k in 0..n_arms {
                    let ai = (r + k) % n_arms;
                    let (_, _, t) = &compiled[ai];
                    let io = GemmIo {
                        pos: &weights[ai].0,
                        neg: &weights[ai].1,
                        scale: &weights[ai].2,
                        input: &inputs[ai],
                        output: &outputs[ai],
                    };
                    let t0 = std::time::Instant::now();
                    for _ in 0..launches {
                        t.launch(&io, &fx, p);
                    }
                    let dt = t0.elapsed().as_secs_f64() / launches as f64;
                    assert!(dt > 0.0, "{shape_name} P={p} arm {ai}: zero-length timing");
                    row[ai] = dt;
                }
            }

            let median = |mut v: Vec<f64>| {
                v.sort_by(|a, b| a.total_cmp(b));
                v[v.len() / 2]
            };
            let meds: Vec<f64> = (0..n_arms)
                .map(|ai| median(times.iter().map(|row| row[ai]).collect()))
                .collect();
            let (b_lo, b_hi) = {
                let mut v = times.iter().map(|row| row[0]).collect::<Vec<_>>();
                v.sort_by(|a, b| a.total_cmp(b));
                (v[0], v[v.len() - 1])
            };
            let (t_lo, t_hi) = {
                let mut v = times.iter().map(|row| row[1]).collect::<Vec<_>>();
                v.sort_by(|a, b| a.total_cmp(b));
                (v[0], v[v.len() - 1])
            };
            let delta = meds[0] - meds[1];
            println!(
                "\n── {shape_name} (m={m} k={n}) P={p} — baseline {:.3} ms/launch · packed {:.3} ms/launch \
                 · Δ {:>7.3} ms ({:>5.2}% of wall)  Δ-range [{:.3}, {:.3}] ms",
                meds[0] * 1e3,
                meds[1] * 1e3,
                delta * 1e3,
                delta / meds[0] * 100.0,
                (b_lo - t_hi) * 1e3,
                (b_hi - t_lo) * 1e3
            );

            // Frame projection weight: the shape's per-chunk launch count.
            let weight = FRAME_LAUNCHES_PER_CHUNK
                .iter()
                .find(|(name, _)| *name == shape_name)
                .map(|(_, w)| *w)
                .unwrap_or(0);
            if weight == 0 {
                println!("  (not in the frame projection weight table — raw row only)");
            } else {
                projection[pi] += delta * 1e3 * (weight as f64) * 8.0;
            }
        }
    }

    for (pi, &p) in ps.iter().enumerate() {
        let share = projection[pi] / frame_ref_ms * 100.0;
        println!(
            "\nPACKED PRICE projection P={p}: {:.2} ms/frame vs the {:.2} ms Bench-937 frame \
             = {:.2}% e2e (bar 1.5% = {:.2} ms) — {}",
            projection[pi],
            frame_ref_ms,
            share,
            frame_ref_ms * 0.015,
            if share >= 1.5 { "OVER — a G1 serving twin + decider A/B is justified" } else { "UNDER — the packed rung closes NEGATIVE on the measured price" }
        );
    }

    print_provenance("end");
}
