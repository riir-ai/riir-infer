//! Issue 1256 T1 rung 2 — the residual-mount walk's alloc census (the
//! `implants_g4_alloc` shape: a per-binary counting allocator, ONE test
//! so the count is unpolluted). The paired law: a residual-action
//! implant mounted but never fired must add ZERO allocations to the
//! walk vs the bias-only mount on the same rows — the pending arm/take
//! and the cache lookups are alloc-free by construction. The FIRE path
//! is measured and disclosed (the table clone per fire is the priced
//! cost — fires are rare by design), never gated as zero.
//!
//! Run: `cargo test -p riir-infer-core --features implants_cell --test
//! implants_residual_g4 -- --nocapture`

#![cfg(feature = "implants_cell")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

use riir_infer_core::implants_cell::{
    ArmsRig, FixtureCodec, LogitsSource, MountPosture, ResidualRungMode, ResidualSiteDims,
    SYNTH_SITE_IN, SyntheticCodec, SyntheticSource, WalkBufs, author_lanes, decode_walk,
    residual_variant_lanes,
};

const RANK: u32 = 2;
const SCALE: f32 = 0.05;
const SEED: u64 = 0x1256_9880_5EED_00A2;

fn fixture_dims() -> ResidualSiteDims {
    ResidualSiteDims {
        in_dim: SYNTH_SITE_IN,
        out_dim: 48,
        layer: 0,
    }
}

#[test]
fn residual_mount_silent_walk_alloc_free() {
    let lanes = author_lanes(&SyntheticCodec).expect("author");
    let res = residual_variant_lanes(
        &lanes,
        &fixture_dims(),
        RANK,
        SEED,
        SCALE,
        ResidualRungMode::ResidualOnly,
    )
    .expect("variant");
    let bias_rig = ArmsRig::mint(&lanes);
    let res_rig = ArmsRig::mint(&res);
    let toks = SyntheticCodec
        .encode("The quick brown fox.")
        .expect("encode");

    let mut src = SyntheticSource::new();
    // Warm everything once (construction is one-time, out of scope):
    // both rigs, the walk buffers, the source buffers, the token path.
    let bufs = &mut WalkBufs::new(src.vocab());
    let mut warm = |rig: &ArmsRig| {
        let mut rt = rig.new_runtime(MountPosture::Armed).expect("mount");
        let _ = decode_walk(&mut src, &toks, 4, &mut rt, bufs);
    };
    warm(&bias_rig);
    warm(&res_rig);

    // The paired census: identical walks (same rows, same steps), only
    // the mount's ACTION KIND differs. The mount itself is one-time
    // construction (bigger residual artifact ⟹ more decode allocs) and
    // sits OUTSIDE the measured region — the census covers the walk.
    let mut measure = |rig: &ArmsRig| {
        let mut rt = rig.new_runtime(MountPosture::Armed).expect("mount");
        let before = ALLOCS.load(Ordering::Relaxed);
        let out = decode_walk(&mut src, &toks, 4, &mut rt, bufs);
        let after = ALLOCS.load(Ordering::Relaxed);
        (after - before, out)
    };
    let (bias_allocs, bias_out) = measure(&bias_rig);
    let (res_allocs, res_out) = measure(&res_rig);
    // Both walks silent: same tokens, same digests (the G1 law — also
    // the non-vacuous witness: the walks really ran).
    assert_eq!(
        bias_out.tokens, res_out.tokens,
        "both mounts silent: identical rows"
    );
    assert_eq!(bias_out.fires.len(), 0, "the prose row must not fire");
    assert_eq!(
        res_allocs, bias_allocs,
        "the residual mount's silent path must add zero allocations (bias {bias_allocs} vs residual {res_allocs})"
    );
}
