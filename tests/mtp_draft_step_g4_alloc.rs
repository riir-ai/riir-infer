//! Issue 1253 T5.2 G4 — the MTP draft step is allocation-free at the
//! reference posture (plan 639 bar (c)'s precursor; the verdict defect —
//! a per-call `to_vec` for the attention residual anchor — fixed in the
//! scratch, and this gate exists so it cannot come back).
//!
//! Its OWN test target, deliberately: the counting allocator is
//! process-global and `cargo test` runs a target's tests in parallel
//! threads in ONE process, so sibling tests' allocations leak into any
//! count taken beside them (the twt_g4_alloc lesson; the gdn_delta_roll
//! precedent). ONE measured test + the liveness canary.
//!
//! Runs: `cargo test --features mtp_draft_block,deltanet_ternary_inference,bonsai2_hadamard
//! --test mtp_draft_step_g4_alloc`
#![cfg(all(
    feature = "mtp_draft_block",
    feature = "deltanet_ternary_inference",
    feature = "bonsai2_hadamard"
))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

#[path = "common/mtp_draft_fixture.rs"]
mod fixture;

use riir_infer_core::deltanet::forward::{HybridCache, HybridForwardScratch, effective_rotary_dim};
use riir_infer_core::deltanet::rotation::rotate_inverse_inplace;
use riir_infer_core::deltanet::ternary_forward::forward_qwen_deltanet_ternary;
use riir_infer_core::gguf_loader::{
    GgufFile, load_mtp_draft_block, load_qwen_deltanet_ternary_from_gguf,
};
use riir_infer_core::mtp_draft::{MtpDraftStepScratch, mtp_draft_step};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::transformer::KVCache;

use std::io::Write as _;

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

/// The measured test: warm the step once (rayon globals + any lazy init),
/// zero the counter, run the step, assert ZERO allocations.
#[test]
fn draft_step_is_alloc_free() {
    let path = std::env::temp_dir().join(format!("riir_1253_t52_g4_{}.gguf", std::process::id()));
    {
        let mut f = std::fs::File::create(&path).expect("create temp gguf");
        f.write_all(&fixture::declared_twin_bytes()).expect("write");
    }
    let gguf = GgufFile::open(&path).expect("open twin");
    let (config, weights) =
        load_qwen_deltanet_ternary_from_gguf(&gguf, "").expect("main stack loads");
    let block = load_mtp_draft_block(&gguf, &config).expect("draft block admits");

    let n = config.n_embd;
    let v = config.vocab_size;
    let rope = RopeFreqTable::new(config.rope_theta, effective_rotary_dim(&config));

    // Land two tokens on the main stack, recover the primal hidden.
    let mut cache = HybridCache::with_layer_types(&config, &weights.layer_types);
    let mut scratch = HybridForwardScratch::new(&config);
    let mut x = vec![0.0f32; v.max(n)];
    for (pos, tok) in [3usize, 17].iter().enumerate() {
        forward_qwen_deltanet_ternary(
            &mut x,
            &weights,
            &mut cache,
            *tok,
            pos,
            &config,
            &mut scratch,
            &rope,
        );
    }
    let mut h = scratch.hidden_copy[..n].to_vec();
    if let Some(rot) = &weights.rotation {
        let signs = rot.signs_for_width(n).map(|s| s.to_vec());
        if let Some(s) = &signs {
            rotate_inverse_inplace(&mut h[..n], Some(s), rot.block_size);
        }
    }

    let mut ds = MtpDraftStepScratch::new(&config);
    let mut lg = vec![0.0f32; v];
    let mut draft_cache = KVCache::new(&config);
    let mut h_warm = h.clone();
    mtp_draft_step(
        &block,
        &weights,
        &mut draft_cache,
        2,
        &mut h_warm,
        17,
        &config,
        &rope,
        &mut ds,
        &mut lg,
    )
    .expect("warm step");

    // Measured section: one step, counter zeroed first (the input clone
    // lands BEFORE the zeroing — the measurement covers the step alone).
    let mut h_m = h.clone();
    ALLOCS.store(0, Ordering::Relaxed);
    mtp_draft_step(
        &block,
        &weights,
        &mut draft_cache,
        3,
        &mut h_m,
        17,
        &config,
        &rope,
        &mut ds,
        &mut lg,
    )
    .expect("measured step");
    let measured = ALLOCS.load(Ordering::Relaxed);
    assert_eq!(
        measured, 0,
        "the draft step allocated {measured} times — the scratch is grow-once by contract \
         (the verdict defect class; bar (c)'s precursor)"
    );
    let _ = std::fs::remove_file(&path);
}

/// The allocator liveness canary: the counter itself must be live — a
/// counting allocator that counts nothing proves nothing (the green-zero
/// law at the instrument layer).
#[test]
fn alloc_counter_is_live() {
    let before = ALLOCS.load(Ordering::Relaxed);
    let probe: Vec<u8> = vec![0u8; 1024];
    assert!(probe.len() == 1024);
    let after = ALLOCS.load(Ordering::Relaxed);
    assert!(
        after > before,
        "the counting allocator counted nothing — the gate is blind"
    );
}
