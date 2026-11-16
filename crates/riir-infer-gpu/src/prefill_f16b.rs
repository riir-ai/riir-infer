//! Issue 1225 — the f16-B serving twin lane: route latches, per-family
//! admission, and the matched-input retention tap census.
//!
//! The lever (Bench 1223): the ternary prefill GEMM's B-side activations
//! read f16 instead of f32 — the serving GEMM at prefill shapes is
//! B-side-bandwidth-bound, and the win is posture-independent. This module
//! carries the ROUTE half of the twin ([`crate::gemm_ternary_metal_zero_copy`]
//! carries the kernel half):
//!
//! - **Route** ([`set_f16b_route`]): the substrate latch stays opt-in —
//!   the default build never routes the twin (every bit-exact gate keeps
//!   the incumbent bytes, the Issue-1146 lossy-surface scoping law). The
//!   PROMOTED default-ON lives in the dist serving lane (`stage_gpu`'s
//!   arm: `RIIR_PREFILL_F16B=0` demotes byte-identically; unset = ON —
//!   T4's delegated-owner verdict AGREE 2026-11-16, G1 Bench 1224 + G2
//!   Bench 1225 + G3/G4 by construction, modelless-clean). The twin's
//!   pipeline is a CONSTRUCTION-TIME posture: pin the flags BEFORE
//!   `TernaryDeltanetGpuForward::new` (the same ordering law the relaxed
//!   pin carries).
//! - **Per-family admission** ([`set_f16b_site_mask`]): the demote axis of
//!   the T1 decider — a family whose retention fails the pre-registered
//!   floor drops out of the route individually (the
//!   `PREFILL_TENSOR_SITE_MASK` pattern), the rest stay armed.
//! - **Tap** ([`set_f16b_tap`]): the instrument half — at every twin-routed
//!   site, the SAME input also dispatches through the incumbent into a
//!   scratch (never the real output), and the two outputs' per-row argmax
//!   picks, worst scale-relative drift and bounded flip samples tally into
//!   the census ([`f16b_tap_snapshot`]). Matched-input by construction, so
//!   trajectory divergence never smears family attribution. The tap
//!   serializes per site (wait + shared-memory reads) — instrument-grade
//!   only, never armed in a timing run.
//! - **Placebo** ([`set_f16b_placebo`]): the control arm — the twin
//!   compiles UNPATCHED (numerics-identical) and dispatches through the
//!   standard f32 path. The decider's HARD gate: the placebo must read
//!   KL == 0 and zero flips; any nonzero is an instrument defect and voids
//!   the posture's armed numbers.
//!
//! `embed` is a dequant lookup and the lm_head/tail is a separate dispatch
//! path — neither is a twin-routed GEMM site; both are covered end-to-end by
//! the walk's trajectory read, never per-family.

#![cfg(feature = "prefill_f16b")]

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::ternary_deltanet_gpu_forward::PrefillSite;

static ROUTE_ARMED: AtomicBool = AtomicBool::new(false);
static TAP_ARMED: AtomicBool = AtomicBool::new(false);
static PLACEBO_PINNED: AtomicBool = AtomicBool::new(false);

/// Per-family admission (bit `i` = [`PrefillSite`] slot `i`); default ALL —
/// a family is demoted individually by the T1 decider's verdict.
static SITE_MASK: AtomicU64 = AtomicU64::new(u64::MAX);

/// Pin the twin route for every forward constructed after (and consulted at
/// every dispatch). SUBSTRATE posture: PROMOTE-ONLY — the default here stays
/// OFF so every bit-exact gate keeps the incumbent bytes (the Issue-1146
/// lossy-surface scoping law); the PROMOTED default-ON lives in the dist
/// serving lane (`stage_gpu`'s arm: `RIIR_PREFILL_F16B=0` demotes, unset =
/// ON — T4's delegated-owner verdict, 2026-11-16), the lane where the
/// quality read (Bench 1224) was taken. `false` keeps the incumbent f32-B
/// route byte-identically (the walk's A arm).
pub fn set_f16b_route(on: bool) {
    ROUTE_ARMED.store(on, Ordering::Relaxed);
}

/// Whether the twin route is armed.
pub fn f16b_route_armed() -> bool {
    ROUTE_ARMED.load(Ordering::Relaxed)
}

/// Arm the matched-input retention tap (instrument-grade; implies nothing
/// about the route — the walk arms both).
pub fn set_f16b_tap(on: bool) {
    TAP_ARMED.store(on, Ordering::Relaxed);
}

/// Whether the tap is armed.
pub fn f16b_tap_armed() -> bool {
    TAP_ARMED.load(Ordering::Relaxed)
}

/// Pin the placebo posture — the twin compiles UNPATCHED. Construction-time:
/// must be pinned before the twin's pipeline compiles.
pub fn set_f16b_placebo(on: bool) {
    PLACEBO_PINNED.store(on, Ordering::Relaxed);
}

/// Whether the twin should compile the unpatched (control) body.
pub fn placebo_pinned() -> bool {
    PLACEBO_PINNED.load(Ordering::Relaxed)
}

static TWIN_MISSING_WARNED: std::sync::OnceLock<()> = std::sync::OnceLock::new();

/// One loud line per process: the twin route is armed but the twin is
/// absent (compile failed, or the flags were pinned after construction) —
/// the incumbent serves every site, counted. The walk's engagement gate
/// reads this as a failure via the zero instance counts.
pub fn warn_twin_missing_once() {
    TWIN_MISSING_WARNED.get_or_init(|| {
        eprintln!(
            "[i1225] f16b route armed but the twin is absent (compile failed or \
             pinned after construction) — the incumbent serves, counted"
        );
    });
}

/// Restrict the twin route to `mask` sites (bit `i` = [`PrefillSite`] slot
/// `i`). [`f16b_site_mask_all`] restores every family.
pub fn set_f16b_site_mask(mask: u64) {
    SITE_MASK.store(mask, Ordering::Relaxed);
}

/// Every family admitted — the default posture.
pub const F16B_SITE_MASK_ALL: u64 = u64::MAX;

/// Whether `site` is admitted to the twin route.
pub fn f16b_site_admits(site: PrefillSite) -> bool {
    let mask = SITE_MASK.load(Ordering::Relaxed);
    mask == F16B_SITE_MASK_ALL || mask & site.bit() != 0
}

// ---------------------------------------------------------------------------
// The matched-input retention tap census
// ---------------------------------------------------------------------------

/// Per-family tally (one slot per [`PrefillSite`]).
struct F16bTapSiteStats {
    /// Twin dispatches tapped (one per site instance — a site's occurrence k
    /// in a forward is layer k for uniform stacks).
    instances: AtomicU64,
    /// Token rows compared.
    rows: AtomicU64,
    /// Rows whose argmax pick flipped (twin vs incumbent, same input).
    flips: AtomicU64,
    /// Worst scale-relative drift, f64 bits (the value is ≥ 0, so the bit
    /// pattern orders like the value and a CAS-max is exact).
    worst_rel_bits: AtomicU64,
}

impl F16bTapSiteStats {
    const fn new() -> Self {
        Self {
            instances: AtomicU64::new(0),
            rows: AtomicU64::new(0),
            flips: AtomicU64::new(0),
            worst_rel_bits: AtomicU64::new(0),
        }
    }
}

static TAP_CENSUS: [F16bTapSiteStats; 12] = [
    F16bTapSiteStats::new(),
    F16bTapSiteStats::new(),
    F16bTapSiteStats::new(),
    F16bTapSiteStats::new(),
    F16bTapSiteStats::new(),
    F16bTapSiteStats::new(),
    F16bTapSiteStats::new(),
    F16bTapSiteStats::new(),
    F16bTapSiteStats::new(),
    F16bTapSiteStats::new(),
    F16bTapSiteStats::new(),
    F16bTapSiteStats::new(),
];

/// A bounded flip sample (the record's adjudication detail — which layer
/// occurrence, which row, which pick, what the incumbent's top-1 margin was).
#[derive(Clone, Debug)]
pub struct F16bFlipSample {
    pub site: &'static str,
    pub occurrence: u64,
    pub row: usize,
    pub inc_pick: usize,
    pub twin_pick: usize,
    pub inc_top1_margin: f32,
}

static TAP_SAMPLES: Mutex<Vec<F16bFlipSample>> = Mutex::new(Vec::new());
const TAP_SAMPLES_CAP: usize = 8;

/// The site slots in [`PrefillSite`] discriminant order (the census index →
/// site mapping; `PrefillSite` has no `from_slot`).
const SITES: [PrefillSite; 12] = [
    PrefillSite::AttnQ,
    PrefillSite::AttnKv,
    PrefillSite::AttnOutProj,
    PrefillSite::GdnQkv,
    PrefillSite::GdnZ,
    PrefillSite::GdnA,
    PrefillSite::GdnB,
    PrefillSite::GdnOutProj,
    PrefillSite::FfnGate,
    PrefillSite::FfnUp,
    PrefillSite::FfnDown,
    PrefillSite::Other,
];

/// Reset the census (once per armed pass, before the tapped frames).
pub fn f16b_tap_reset() {
    for s in TAP_CENSUS.iter() {
        s.instances.store(0, Ordering::Relaxed);
        s.rows.store(0, Ordering::Relaxed);
        s.flips.store(0, Ordering::Relaxed);
        s.worst_rel_bits.store(0, Ordering::Relaxed);
    }
    TAP_SAMPLES.lock().expect("tap samples lock").clear();
}

/// Tally one tapped site instance: `inc` and `twin` are the SAME input's
/// incumbent / twin outputs, each `[p × m]` row-major. Per row: the argmax
/// pick over `m`; per instance: worst scale-relative drift over a strided
/// subsample (every 16th element — the drift class is already pinned by the
/// G5 full-element cell; the stride keeps the tap's host loop bounded).
pub fn f16b_tap_tally(
    site: PrefillSite,
    occurrence: u64,
    inc: &[f32],
    twin: &[f32],
    p: usize,
    m: usize,
) {
    let stats = &TAP_CENSUS[site.slot()];
    stats.instances.fetch_add(1, Ordering::Relaxed);
    stats.rows.fetch_add(p as u64, Ordering::Relaxed);

    // Worst scale-relative drift, strided (every 16th element).
    let ref_scale = inc.iter().fold(0.0f32, |a, &b| a.max(b.abs())).max(1e-6);
    let mut worst = 0.0f64;
    for i in (0..inc.len()).step_by(16) {
        let rel = ((inc[i] - twin[i]).abs() / ref_scale) as f64;
        if rel > worst {
            worst = rel;
        }
    }
    let mut cur = stats.worst_rel_bits.load(Ordering::Relaxed);
    while worst.to_bits() > cur {
        match stats.worst_rel_bits.compare_exchange_weak(
            cur,
            worst.to_bits(),
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(seen) => cur = seen,
        }
    }

    // Per-row argmax picks (sequential fold, first-index tie rule — the
    // determinism law's measurement shape).
    let mut flips = 0u64;
    for row in 0..p {
        let base = row * m;
        let inc_row = &inc[base..base + m];
        let twin_row = &twin[base..base + m];
        let (inc_pick, inc_best) =
            inc_row
                .iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
                    if x > bv { (i, x) } else { (bi, bv) }
                });
        let twin_pick = twin_row
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
                if x > bv { (i, x) } else { (bi, bv) }
            })
            .0;
        if twin_pick != inc_pick {
            flips += 1;
            let mut samples = TAP_SAMPLES.lock().expect("tap samples lock");
            if samples.len() < TAP_SAMPLES_CAP {
                let second_best = inc_row
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != inc_pick)
                    .fold(f32::NEG_INFINITY, |a, (_, &x)| a.max(x));
                samples.push(F16bFlipSample {
                    site: site.name(),
                    occurrence,
                    row,
                    inc_pick,
                    twin_pick,
                    inc_top1_margin: inc_best - second_best,
                });
            }
        }
    }
    stats.flips.fetch_add(flips, Ordering::Relaxed);
}

/// One site's snapshot row.
#[derive(Clone, Debug)]
pub struct F16bTapRow {
    pub site: &'static str,
    pub instances: u64,
    pub rows: u64,
    pub flips: u64,
    pub worst_rel: f64,
}

/// The instance count recorded so far for `site` (the tap's occurrence
/// number — a site's k-th tap in a forward is layer k for uniform stacks).
pub fn f16b_tap_instances(site: PrefillSite) -> u64 {
    TAP_CENSUS[site.slot()].instances.load(Ordering::Relaxed)
}

/// Snapshot the census (after a tapped pass). One row per [`PrefillSite`]
/// slot, in discriminant order — including zero-row slots, so a missing
/// family is a LOUD zero the engagement gate reads, never a blind absence.
pub fn f16b_tap_snapshot() -> Vec<F16bTapRow> {
    SITES
        .iter()
        .zip(TAP_CENSUS.iter())
        .map(|(site, s)| F16bTapRow {
            site: site.name(),
            instances: s.instances.load(Ordering::Relaxed),
            rows: s.rows.load(Ordering::Relaxed),
            flips: s.flips.load(Ordering::Relaxed),
            worst_rel: f64::from_bits(s.worst_rel_bits.load(Ordering::Relaxed)),
        })
        .collect()
}

/// The bounded flip samples collected since the last reset.
pub fn f16b_flip_samples() -> Vec<F16bFlipSample> {
    TAP_SAMPLES.lock().expect("tap samples lock").clone()
}
