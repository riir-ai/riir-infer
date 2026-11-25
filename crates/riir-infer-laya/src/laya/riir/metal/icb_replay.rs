//! The Metal ICB replay machinery (issue 1181 T2, tranche 1) — the B319
//! remedy's primitives, built not linked.
//!
//! The cubecl #1993 finding (riir-refine kernel_opt `metal-icb-replay-
//! rewrite-private-copy`, upstream @ `c1d0125`): replaying an unchanged
//! `MTLIndirectCommandBuffer` degrades on Apple drivers (10 → 23 ms over
//! 20 replays) — the remedy is a per-replay BLIT-COPY of the commands
//! alone into a private ICB before every execute, plus the cross-stream
//! last-replay event fence. riir-infer's vendored cubecl carries no ICB
//! path and `cubecl-metal` is absent from the lockfile, so the remedy
//! lives here — the machinery in OUR binding set, measured on OUR stack.
//!
//! metal-rs 0.31.0 covers every primitive except ONE: `execute_commands_
//! in_buffer` is bound only on the RENDER command encoder; the compute
//! twin (`MTLComputeCommandEncoder.executeCommandsInBuffer:withRange:`,
//! macOS 11+) is the same ObjC selector, sent directly — the
//! `GPUStartTime` probe precedent ([`super::close_encode`]).
//!
//! Shape (v0):
//! - [`ReplayGroup::record`] freezes a dispatch sequence into the SOURCE
//!   ICB. Scalars are NOT expressible in an indirect compute command
//!   (`set_bytes` has no ICB twin) — the caller materializes every
//!   scalar into its own [`ReplayGroup::scalar_buffer`] and binds it at
//!   a kernel buffer index. That IS the eligibility discipline (1181
//!   T3's declared-resource contract): a command whose scalars change
//!   per replay is INELIGIBLE unless the change routes through a buffer
//!   whose CONTENT advances device-side.
//! - [`ReplayGroup::replay`] blit-copies the commands alone into the
//!   private ICB and executes the private range — the remedy's exact
//!   shape — on a fresh command buffer the caller commits/waits.
//!   [`ReplayGroup::replay_from_source`] executes the source range
//!   directly: the A/B arm that measures the degradation tax this
//!   driver actually applies.
//! - The event-fence helpers ([`ReplayGroup::signal_event`],
//!   [`signal_wait`]) expose the cross-stream leg; v1 is single-queue,
//!   where Metal's in-order queue already orders replays and the fence
//!   is not load-bearing (documented, exercised by the smoke fixture,
//!   never silently absent).
//!
//! No production caller yet — the decode-lane integration is T3's
//! eligibility census, decided WITH this module's measured per-replay
//! cost (the fixture prints it; the break-even prices it). Never a
//! default: the incumbent per-op encode path is untouched, and any
//! default flip follows the env-switch law + the GOAT gates.
//!
//! ## The three ICB recording/execution contracts (found 2026-11-25, m3)
//!
//! The 11-15/11-17/11-23 "AGX G17X driver segfault" class was NOT driver
//! state — it was three lane-logic contract gaps, reproducing on BOTH AGX
//! families (M5 Ultra AND M3 Max, macOS 27.2.1):
//!
//! 1. **Pipeline flag** — a pipeline entering an ICB MUST be created with
//!    `supportIndirectCommandBuffers = YES` ([`ReplayGroup::icb_pipeline`
//!    is the one home]). Without it the debug wrapper asserts
//!    ("Setting a pipeline that does not have
//!    supportIndirectCommandBuffers = YES is invalid") and the release
//!    driver segfaults inside the AGX `setComputePipelineState:` setter
//!    (the recorded crash class).
//! 2. **metal-rs 0.31's `MTLIndirectCommandType` enum values are WRONG**
//!    against the SDK header (MTLIndirectCommandBuffer.h): the header
//!    binds `ConcurrentDispatch = 1<<5` and `ConcurrentDispatchThreads =
//!    1<<6`, the crate binds `1<<4/1<<5` — bit 5 collides with
//!    ConcurrentDispatch, so the crate's spelling leaves the
//!    concurrentDispatchThreads command type UNENABLED (debug:
//!    "not enabled on this indirect command buffer"; release: the same
//!    segfault class). [`ReplayGroup::record`] uses
//!    `from_bits_retain(1 << 6)` with the citation comment.
//! 3. **The execute-declaration contract** — an encoder that executes an
//!    ICB sees none of the buffers inside it (the cubecl capture.rs
//!    law): every bound buffer must be DECLARED to the executing encoder
//!    (`useResources`, Read|Write) plus the ICB itself (Read), or the
//!    commands run against nothing — the silent no-op class (recording
//!    validates clean, execution does nothing, no readback changes).
//!    `replay`/`replay_from_source` declare `self.resident` + the target
//!    ICB.
//!
//! With all three, the fixture is GREEN unignored: 40 replays
//! bit-identical to the host composition AND the direct-encode control;
//! the cost A/B measures (m3 release posture 2026-11-25: remedy 69.0 µs
//! / source 35.3 µs / direct 33.0 µs per-replay host enqueue).

use std::ops::Deref;

use metal::{
    ComputeCommandEncoderRef, ComputePipelineState, Device, IndirectCommandBuffer,
    IndirectCommandBufferDescriptor, MTLGPUFamily, MTLIndirectCommandType, MTLResourceOptions,
    MTLSize, NSRange,
};
use objc2::rc::autoreleasepool;

use super::{Result, rt};

/// metal-rs 0.31.0 binds `execute_commands_in_buffer` on the RENDER
/// encoder only; the compute twin is the same selector.
fn execute_commands_in_buffer_compute(
    enc: &ComputeCommandEncoderRef,
    icb: &metal::IndirectCommandBufferRef,
    range: NSRange,
) {
    use metal::objc::{msg_send, sel, sel_impl};
    // The legacy `objc` macro probes `feature = "cargo-clippy"` (the
    // close_encode precedent).
    #[allow(unexpected_cfgs)]
    unsafe {
        msg_send![enc, executeCommandsInBuffer: icb withRange: range]
    }
}

/// One frozen dispatch in the recorded sequence.
///
/// `barrier` orders THIS command against the previous one inside the
/// ICB (a data dependency the recorder cannot infer). `grid` is the
/// THREADS-per-grid (the dispatch_threads geometry the backend's `run`
/// path uses); `tpg` the threads per threadgroup.
pub struct IcbCommand {
    pub pipeline: ComputePipelineState,
    /// `(kernel buffer index, buffer, byte offset)` — the command's
    /// complete binding set, indices explicit (an ICB command does not
    /// inherit the encoder's bindings).
    pub bindings: Vec<(usize, metal::Buffer, u64)>,
    /// `(threadgroup memory index, byte length)`.
    pub staging: Vec<(usize, u64)>,
    pub grid: MTLSize,
    pub tpg: MTLSize,
    pub barrier: bool,
}

/// A recorded sequence + its private replay copy, on one device/queue.
pub struct ReplayGroup {
    device: Device,
    queue: metal::CommandQueue,
    /// The frozen template — recorded once, never executed from in the
    /// remedy posture (the degradation-tax arm does, deliberately).
    source: IndirectCommandBuffer,
    /// The per-replay blit destination (the B319 private copy).
    private: IndirectCommandBuffer,
    /// Every buffer the recorded commands bind, deduped — DECLARED to the
    /// executing encoder at replay time (`useResources`). The contract the
    /// 2026-11-25 m3 no-op cell found: an encoder that executes an ICB
    /// sees none of the buffers inside it (the cubecl capture.rs law —
    /// without the declaration the commands run against nothing, a
    /// silent no-op: recording validates clean, execution does nothing,
    /// no readback changes).
    resident: Vec<metal::Buffer>,
    count: usize,
}

impl ReplayGroup {
    /// Capability gate (loud, never a silent fallback — the lane's
    /// convention): ICB compute execution is macOS 11+ / Apple Silicon
    /// (`Mac2`).
    pub fn device_supports(device: &Device) -> bool {
        device.supports_family(MTLGPUFamily::Mac2)
    }

    /// Materialize a scalar block (the frozen `constant` args of one
    /// command) into its own device buffer — the patchable-slot shape:
    /// the CONTENT is a declared resource the replay never re-encodes.
    pub fn scalar_buffer(&self, bytes: &[u8]) -> metal::Buffer {
        self.device
            .new_buffer_with_data(
                bytes.as_ptr().cast::<core::ffi::c_void>(),
                bytes.len() as u64,
                MTLResourceOptions::StorageModeShared,
            )
            .to_owned()
    }

    /// An ICB-compatible compute pipeline from a function: the recording
    /// contract requires `supportIndirectCommandBuffers = YES` (the
    /// 2026-11-25 m3 finding — without it the debug wrapper asserts
    /// "Setting a pipeline that does not have
    /// supportIndirectCommandBuffers = YES is invalid" and the release
    /// driver segfaults inside the AGX setComputePipelineState setter).
    /// Every pipeline handed to [`ReplayGroup::record`] MUST come
    /// through this constructor (or set the flag itself).
    pub fn icb_pipeline(
        device: &Device,
        function: &metal::FunctionRef,
    ) -> Result<ComputePipelineState> {
        let desc = metal::ComputePipelineDescriptor::new();
        let d: &metal::ComputePipelineDescriptorRef = desc.deref();
        d.set_compute_function(Some(function));
        d.set_support_indirect_command_buffers(true);
        device.new_compute_pipeline_state(d).map_err(rt)
    }

    /// Freeze `commands` into the source ICB. `max_count` is the ICB's
    /// capacity — the recorded count can never grow past it (a re-record
    /// of a longer sequence allocates a new group; the capacity is part
    /// of the frozen contract).
    pub fn record(
        device: Device,
        queue: metal::CommandQueue,
        commands: &[IcbCommand],
    ) -> Result<Self> {
        if commands.is_empty() {
            return Err(rt("icb_replay: an empty sequence records nothing"));
        }
        if !Self::device_supports(&device) {
            return Err(rt(
                "icb_replay: the device lacks ICB compute support (Mac2)",
            ));
        }
        let desc = IndirectCommandBufferDescriptor::new();
        let d: &metal::IndirectCommandBufferDescriptorRef = desc.deref();
        // metal-rs 0.31's MTLIndirectCommandType values are WRONG against
        // the SDK header (MTLIndirectCommandBuffer.h): the header binds
        // ConcurrentDispatch = 1<<5 and ConcurrentDispatchThreads = 1<<6,
        // the crate binds 1<<4/1<<5 — so the crate's ConcurrentDispatch
        // Threads bit-5 collides with ConcurrentDispatch and leaves the
        // concurrentDispatchThreads command type UNENABLED. Recording a
        // concurrentDispatchThreads command against such a descriptor is
        // the 11-15/11-17/11-23 crash class (release drivers segfault;
        // the debug wrapper asserts "not enabled on this indirect
        // command buffer"). from_bits_retain(1 << 6) IS the header's
        // ConcurrentDispatchThreads.
        d.set_command_types(MTLIndirectCommandType::from_bits_retain(1 << 6));
        d.set_inherit_buffers(false);
        d.set_inherit_pipeline_state(false);
        // Metal's per-command compute buffer-bind ceiling.
        d.set_max_kernel_buffer_bind_count(31);
        let count = commands.len();
        let opts = MTLResourceOptions::StorageModeShared;
        let source = device
            .new_indirect_command_buffer_with_descriptor(d, count as u64, opts)
            .to_owned();
        let private = device
            .new_indirect_command_buffer_with_descriptor(d, count as u64, opts)
            .to_owned();
        let resident = autoreleasepool(|_| {
            let mut resident: Vec<metal::Buffer> = Vec::new();
            for (i, c) in commands.iter().enumerate() {
                let cmd = source.indirect_compute_command_at_index(i as u64);
                if c.barrier {
                    cmd.set_barrier();
                } else {
                    cmd.clear_barrier();
                }
                cmd.set_compute_pipeline_state(&c.pipeline);
                for (idx, buf, off) in &c.bindings {
                    cmd.set_kernel_buffer(*idx as u64, Some(buf), *off);
                    // Dedupe by the ObjC object pointer (the Ref wrappers
                    // are transparent over it; metal-rs's ForeignType trait
                    // is private).
                    let p = buf.as_ref() as *const metal::BufferRef as *const u8;
                    if !resident
                        .iter()
                        .any(|r| r.as_ref() as *const metal::BufferRef as *const u8 == p)
                    {
                        resident.push(buf.clone());
                    }
                }
                for (idx, len) in &c.staging {
                    cmd.set_threadgroup_memory_length(*idx as u64, *len);
                }
                cmd.concurrent_dispatch_threads(c.grid, c.tpg);
            }
            resident
        });
        Ok(Self {
            device,
            queue,
            source,
            private,
            resident,
            count,
        })
    }

    /// The B319 remedy: blit-copy the commands alone into the private
    /// ICB, then execute the PRIVATE range on a fresh command buffer.
    /// The returned buffer is committed; the caller waits it (or
    /// composes it into its own stream). The source ICB is never
    /// executed from, so it cannot accumulate the driver's replay
    /// degradation.
    pub fn replay(&self) -> metal::CommandBuffer {
        autoreleasepool(|_| {
            let cb = self.queue.new_command_buffer().to_owned();
            {
                let blit = cb.new_blit_command_encoder().to_owned();
                blit.copy_indirect_command_buffer(
                    &self.source,
                    NSRange::new(0, self.count as u64),
                    &self.private,
                    0,
                );
                blit.end_encoding();
            }
            {
                let enc = cb.new_compute_command_encoder().to_owned();
                // The execute-declaration contract (the 2026-11-25 m3
                // no-op cell): an encoder executing an ICB sees none of
                // the buffers inside it — declare them, plus the ICB
                // itself (read: the blit above rewrote it).
                let refs: Vec<&metal::ResourceRef> = self
                    .resident
                    .iter()
                    .map(|b| {
                        let r: &metal::ResourceRef = b.deref();
                        r
                    })
                    .collect();
                enc.use_resources(
                    &refs,
                    metal::MTLResourceUsage::Read | metal::MTLResourceUsage::Write,
                );
                enc.use_resource(self.private.deref(), metal::MTLResourceUsage::Read);
                execute_commands_in_buffer_compute(
                    &enc,
                    self.private.deref(),
                    NSRange::new(0, self.count as u64),
                );
                enc.end_encoding();
            }
            cb.commit();
            cb
        })
    }

    /// The A/B arm: execute the SOURCE range directly — no private
    /// copy. This is the posture the #1993 finding measured degrading
    /// over repeated replays on Apple drivers; the fixture measures it
    /// on this driver and the break-even prices the difference.
    pub fn replay_from_source(&self) -> metal::CommandBuffer {
        autoreleasepool(|_| {
            let cb = self.queue.new_command_buffer().to_owned();
            let enc = cb.new_compute_command_encoder().to_owned();
            // The execute-declaration contract, source arm: same
            // declaration, the source ICB read.
            let refs: Vec<&metal::ResourceRef> = self
                .resident
                .iter()
                .map(|b| {
                    let r: &metal::ResourceRef = b.deref();
                    r
                })
                .collect();
            enc.use_resources(
                &refs,
                metal::MTLResourceUsage::Read | metal::MTLResourceUsage::Write,
            );
            enc.use_resource(self.source.deref(), metal::MTLResourceUsage::Read);
            execute_commands_in_buffer_compute(
                &enc,
                self.source.deref(),
                NSRange::new(0, self.count as u64),
            );
            enc.end_encoding();
            cb.commit();
            cb
        })
    }

    /// The cross-stream fence's signal leg: `cb` signals `event` at
    /// `value` once its (blit + execute) work completes. Not
    /// load-bearing at v1 (single-queue in-order ordering); the multi-
    /// stream consumer composes this before its next-stream replay.
    pub fn signal_event(&self, cb: &metal::CommandBufferRef, event: &metal::EventRef, value: u64) {
        cb.encode_signal_event(event, value);
    }

    /// The fence's wait leg: `cb` blocks (device-side) until `event`
    /// reaches `value`.
    pub fn signal_wait(&self, cb: &metal::CommandBufferRef, event: &metal::EventRef, value: u64) {
        cb.encode_wait_for_event(event, value);
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn count(&self) -> usize {
        self.count
    }
}
