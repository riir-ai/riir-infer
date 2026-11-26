//! Issue 1256 T1 — the implant observe path's G4 alloc census (the
//! repo-birth gate discipline: the row rides the SAME commit as the
//! gated target). A global counting allocator wraps every test in this
//! binary: construction may allocate (one-time), the steady-state
//! `observe_into` loop must allocate ZERO over the mounted-and-firing
//! posture.
//!
//! Run: `cargo test -p riir-infer-core --features implants --test
//! implants_g4_alloc -- --nocapture`

#![cfg(feature = "implants")]

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

use riir_infer_core::implants::{
    FireRef, ImplantAction, ImplantArtifact, ImplantRuntime, ImplantSpec, TokenPat, TriggerRule,
    WINDOW_MAX,
};

#[test]
fn observe_path_zero_alloc_steady_state() {
    // Two implants, patterns at mixed depths, one AnyOf — the composition
    // shape. Construction allocates (out of scope: one-time).
    let spec_a = ImplantSpec::new(
        "a",
        vec![TriggerRule {
            pattern: vec![TokenPat::Exact(0), TokenPat::Exact(1)],
            action: 0,
        }],
        vec![ImplantAction {
            biases: vec![(10, 8.0)],
        }],
        None,
    )
    .expect("spec a");
    let spec_b = ImplantSpec::new(
        "b",
        vec![TriggerRule {
            pattern: vec![
                TokenPat::AnyOf(vec![(0..WINDOW_MAX as u32).sum(), 2]),
                TokenPat::Exact(3),
                TokenPat::Exact(4),
            ],
            action: 0,
        }],
        vec![ImplantAction {
            biases: vec![(20, 8.0), (21, 1.0)],
        }],
        None,
    )
    .expect("spec b");

    let mut rt = ImplantRuntime::new(2);
    rt.mount(0, &ImplantArtifact::mint(&spec_a, 1))
        .expect("mount a");
    rt.mount(1, &ImplantArtifact::mint(&spec_b, 1))
        .expect("mount b");

    // Warm the window to its full ring (so the steady state is the
    // copy_within path, not the grow path) and warm the fire buffer.
    let mut buf: Vec<FireRef> = Vec::with_capacity(4);
    for t in 0..WINDOW_MAX as u32 {
        buf.clear();
        rt.observe_into(t, &mut buf);
    }

    // The census: 4096 observe steps over a firing stream. ZERO allocs.
    let before = ALLOCS.load(Ordering::Relaxed);
    for step in 0..4096u32 {
        buf.clear();
        let t = step % WINDOW_MAX as u32;
        rt.observe_into(t, &mut buf);
        if !buf.is_empty() {
            let act = rt.action(&buf[0]);
            std::hint::black_box(act.biases.len());
        }
    }
    let after = ALLOCS.load(Ordering::Relaxed);
    assert_eq!(
        after - before,
        0,
        "implant observe path allocated {}/4096 steps",
        after - before
    );
    // The stream fired (the census ran the armed path, not a vacuous
    // zero): "0 1" recurs every WINDOW_MAX steps; implant B's
    // AnyOf(0+1+..+15, 2) head with tail 3 4 also recurs on the cyclic
    // stream.
    let fires_a = rt.fires()[0];
    let fires_b = rt.fires()[1];
    assert!(fires_a > 0, "census must exercise the fire path");
    assert!(fires_b > 0, "the AnyOf composition arm must fire too");
}
