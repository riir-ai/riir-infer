//! Issue 1214 G1(c) — the GPU twin of [`int_gemm`]: ONE MSL kernel, one
//! thread per output element, `long` (i64) accumulation in FIXED
//! ascending-k order, host-side i32 range check on readback. The exact
//! [`int_gemm`](super) semantic — i64 checked accumulation +
//! `i32::try_from` — never a tolerance, never a fallback.
//!
//! This is a CORRECTNESS arm (G1(c): same-bits logits across the CPU int
//! reference vs the GPU int GEMM), not a perf lane: buffers upload per
//! call (no weight cache — the pointer-stability contract machinery is
//! the house Metal backend's job), the kernel is a naive triple loop,
//! and every failure path is a loud refusal (no device → Err, MSL
//! compile → Err, overflow → Err). Determinism is by construction:
//! integer ops only, one thread per output, no atomics, no floating
//! point anywhere.
//!
//! Compiles to nothing off macOS (the house `laya-riir-metal` posture).

use std::ffi::c_void;

use metal::{Buffer, CommandQueue, ComputePipelineState, Device, MTLResourceOptions, MTLSize};
use objc2::rc::autoreleasepool;

/// `StorageModeShared` + tracked hazards — the house Metal options; on
/// Apple silicon Shared reads back without a `didModifyRange` dance.
const RESOURCE_OPTIONS: MTLResourceOptions =
    MTLResourceOptions::StorageModeShared.union(MTLResourceOptions::HazardTrackingModeTracked);

const MSL_IGEMM: &str = r#"
// i1214 G1(c): the exact twin of the CPU int_gemm — a[i,k] @ b[k,n] with
// b the PRE-TRANSPOSED weight ([k, n] row-major, the lane's w8t layout).
// One thread per output element; i64 accumulation in ascending-k order
// (the same order the CPU folds); integer only — bit-identical by
// construction. The host checks every result against i32 on readback,
// exactly the CPU's i64-checked + i32::try_from law.
kernel void igemm_i8(
    device const char* a [[buffer(0)]],
    device const char* b [[buffer(1)]],
    device long* out     [[buffer(2)]],
    constant uint& m     [[buffer(3)]],
    constant uint& n     [[buffer(4)]],
    constant uint& k     [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= n || gid.y >= m) { return; }
    long acc = 0;
    for (uint l = 0u; l < k; ++l) {
        acc += (long)a[gid.y * k + l] * (long)b[l * n + gid.x];
    }
    out[gid.y * n + gid.x] = acc;
}
"#;

const TG: usize = 16;

/// The GPU GEMM device: one Metal device + queue + the compiled-once
/// `igemm_i8` pipeline. Share by `Arc` — never clone.
pub struct IntGemmGpu {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
}

impl IntGemmGpu {
    /// Build the twin — fails loud when no Metal device exists or the
    /// kernel refuses to compile (never a silent CPU fallback).
    pub fn new() -> Result<Self, String> {
        let Some(device) = Device::system_default() else {
            return Err("IntGemmGpu: no Metal device on this host".into());
        };
        let queue = device.new_command_queue();
        let lib = device
            .new_library_with_source(MSL_IGEMM, &metal::CompileOptions::new())
            .map_err(|e| format!("IntGemmGpu: MSL compile failed: {e}"))?;
        let f = lib
            .get_function("igemm_i8", None)
            .map_err(|e| format!("IntGemmGpu: kernel igemm_i8: {e}"))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&f)
            .map_err(|e| format!("IntGemmGpu: pipeline: {e}"))?;
        Ok(Self {
            device,
            queue,
            pipeline,
        })
    }

    fn upload_i8(&self, data: &[i8]) -> Buffer {
        self.device.new_buffer_with_data(
            data.as_ptr().cast::<c_void>(),
            data.len() as u64,
            RESOURCE_OPTIONS,
        )
    }

    fn upload_u32(&self, v: u32) -> Buffer {
        self.device.new_buffer_with_data(
            std::ptr::from_ref(&v).cast::<c_void>(),
            4,
            RESOURCE_OPTIONS,
        )
    }

    fn alloc_out(&self, len: usize) -> Buffer {
        self.device.new_buffer(len as u64 * 8, RESOURCE_OPTIONS)
    }

    /// The exact twin: `a [m, k]` @ `b [k, n]` → `Vec<i32>` — bit-for-bit
    /// what [`int_gemm`] returns, or a loud refusal (shape mismatch,
    /// geometry past the kernel's u32 indexing, an output outside i32 —
    /// the CPU's `i32::try_from` law, checked on readback).
    pub fn gemm(
        &self,
        a: &[i8],
        m: usize,
        k: usize,
        b: &[i8],
        n: usize,
    ) -> Result<Vec<i32>, String> {
        if a.len() != m * k {
            return Err(format!("IntGemmGpu: a len {} != {m}·{k}", a.len()));
        }
        if b.len() != k * n {
            return Err(format!("IntGemmGpu: b len {} != {k}·{n}", b.len()));
        }
        if m == 0 || n == 0 || k == 0 {
            return Err("IntGemmGpu: empty geometry".into());
        }
        // the kernel indexes with u32 (gid·k + l and l·n + gid) — refuse
        // any product that could wrap it, loudly, host-side
        for (name, len) in [("m·k", m * k), ("k·n", k * n), ("m·n", m * n)] {
            if len > u32::MAX as usize {
                return Err(format!("IntGemmGpu: {name} = {len} past u32 indexing"));
            }
        }
        let (m32, n32, k32) = (m as u32, n as u32, k as u32);
        autoreleasepool(|_| {
            let a_buf = self.upload_i8(a);
            let b_buf = self.upload_i8(b);
            let out_buf = self.alloc_out(m * n);
            let (mb, nb, kb) = (
                self.upload_u32(m32),
                self.upload_u32(n32),
                self.upload_u32(k32),
            );
            let cb = self.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pipeline);
            enc.set_buffer(0, Some(&a_buf), 0);
            enc.set_buffer(1, Some(&b_buf), 0);
            enc.set_buffer(2, Some(&out_buf), 0);
            enc.set_buffer(3, Some(&mb), 0);
            enc.set_buffer(4, Some(&nb), 0);
            enc.set_buffer(5, Some(&kb), 0);
            let tpg = MTLSize {
                width: TG as u64,
                height: TG as u64,
                depth: 1,
            };
            let groups = MTLSize {
                width: (n as u64).div_ceil(TG as u64),
                height: (m as u64).div_ceil(TG as u64),
                depth: 1,
            };
            enc.dispatch_thread_groups(groups, tpg);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            let ptr = out_buf.contents();
            assert!(!ptr.is_null(), "IntGemmGpu: out buffer contents null");
            let len = m * n;
            let mut out = Vec::with_capacity(len);
            for i in 0..len {
                let v = unsafe { *(ptr.cast::<i64>().add(i)) };
                let v32 = i32::try_from(v).map_err(|_| {
                    format!("IntGemmGpu: out[{i}] = {v} outside i32 (the CPU i32::try_from law)")
                })?;
                out.push(v32);
            }
            Ok(out)
        })
    }
}

impl std::fmt::Debug for IntGemmGpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("IntGemmGpu(metal/igemm_i8)")
    }
}

#[cfg(test)]
mod goldens {
    use super::super::int_gemm;
    use super::*;

    /// xorshift64* — the lane's seeded, dependency-free generator (the
    /// toy's exact step order; no unseeded draws, the workspace
    /// global-RNG gate class).
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Rng(seed | 1)
        }
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn i8s(&mut self, span: i64) -> i8 {
            ((self.next() % (2 * span as u64 + 1)) as i64 - span) as i8
        }
    }

    fn assert_gpu_matches_cpu(a: &[i8], m: usize, k: usize, b: &[i8], n: usize) {
        let cpu = int_gemm(a, m, k, b, n);
        let gpu = IntGemmGpu::new()
            .expect("IntGemmGpu build")
            .gemm(a, m, k, b, n)
            .expect("IntGemmGpu gemm");
        assert_eq!(cpu.len(), gpu.len(), "out length");
        for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
            assert_eq!(c, g, "gemm[{i}] m={m} k={k} n={n}: cpu {c} vs gpu {g}");
        }
    }

    #[test]
    fn gpu_matches_cpu_fixture_shapes() {
        let mut rng = Rng::new(0x1214);
        // toy + ragged-edge shapes (the 16×16 tile edges) + the real
        // lane geometries (hidden 1024, inter 3072, q/kv dims)
        for (m, k, n) in [
            (1, 1, 1),
            (4, 8, 16),
            (33, 64, 97),
            (17, 129, 31),
            (8, 1024, 1024),  // q
            (8, 1024, 512),   // k/v (kv_dim)
            (8, 2048, 1024),  // o
            (8, 1024, 3072),  // gate/up
            (8, 3072, 1024),  // down
            (64, 1024, 4096), // a bigger batch row
        ] {
            let a: Vec<i8> = (0..m * k).map(|_| rng.i8s(127)).collect();
            let b: Vec<i8> = (0..k * n).map(|_| rng.i8s(127)).collect();
            assert_gpu_matches_cpu(&a, m, k, &b, n);
        }
    }

    #[test]
    fn gpu_matches_cpu_extreme_values() {
        // all-±127: the worst-case accumulator exercise
        let m = 8;
        let k = 1024;
        let n = 64;
        let a = vec![127_i8; m * k];
        let mut b = vec![-127_i8; k * n];
        for (i, v) in b.iter_mut().enumerate() {
            if i % 3 == 0 {
                *v = 127;
            }
        }
        assert_gpu_matches_cpu(&a, m, k, &b, n);
    }

    #[test]
    fn gpu_refuses_loud() {
        let gpu = IntGemmGpu::new().expect("IntGemmGpu build");
        let a = [1_i8; 8];
        let b = [1_i8; 8];
        // a len 8 vs declared 1·3
        assert!(gpu.gemm(&a, 1, 3, &b, 2).is_err(), "a shape");
        // b len 8 vs declared 2·2 (a len 4 == 1·2 is well-shaped)
        assert!(gpu.gemm(&a[..4], 1, 2, &b, 2).is_err(), "b shape");
        assert!(gpu.gemm(&a[..4], 0, 4, &b[..4], 1).is_err(), "empty m");
        assert!(
            gpu.gemm(&a[..1], 1, 1, &b[..1], (u32::MAX as usize) + 1)
                .is_err(),
            "m·n past u32 indexing"
        );
    }
}
