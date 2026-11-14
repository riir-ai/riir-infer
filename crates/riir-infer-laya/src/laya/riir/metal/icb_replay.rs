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
        d.set_command_types(MTLIndirectCommandType::ConcurrentDispatchThreads);
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
        autoreleasepool(|_| {
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
                }
                for (idx, len) in &c.staging {
                    cmd.set_threadgroup_memory_length(*idx as u64, *len);
                }
                cmd.concurrent_dispatch_threads(c.grid, c.tpg);
            }
        });
        Ok(Self {
            device,
            queue,
            source,
            private,
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
