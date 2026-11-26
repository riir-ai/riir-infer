//! The Issue-1256 T1 fixture-tier cell gates: G1 silent identity, G1b
//! false fire, G2 fired exactness, G3 composition — all through the lib
//! module's [`SyntheticSource`] walk (no checkpoint, deterministic).
//! The real tier runs the SAME cells through the SAME code in the
//! `implants_cell` bin over the league checkpoint (CPU, same-device).
//!
//! These are the CI-runnable halves of the pre-registered bars; the
//! real-tier run lands its own benchmark record.

use riir_infer_core::implants_cell::{
    ADD_DEMOTE_ENV, ArmsRig, FixtureCodec, SUB_DEMOTE_ENV, SyntheticCodec, SyntheticSource,
    add_pairs, author_lanes, cell_g1_silent_identity, cell_g1b_false_fire, cell_g2_fired_exactness,
    cell_g3_composition, sub_pairs,
};

/// Serializes EVERY mounting test (any mount reads the demote env at
/// mount — a parallel test inside the demote arm's set-to-"0" window
/// would mount demoted; the lock keeps the env clean per test).
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The fixture families are authored as pinned (the contamination law):
/// 15 addition rows, 14 subtraction rows, all single-digit answers.
#[test]
fn authored_fixture_families_pinned() {
    let add = add_pairs();
    let sub = sub_pairs();
    assert_eq!(add.len(), 15, "addition family pinned at 15 rows");
    assert_eq!(sub.len(), 14, "subtraction family pinned at 14 rows");
    for (a, b) in &add {
        assert!(*a >= 2 && *a <= 5 && *b >= 2 && *b <= 5);
        assert!(a + b <= 9, "addition answers stay single-digit");
    }
    for (a, b) in &sub {
        assert!(*a >= 4 && *a <= 7 && *b >= 2 && b < a);
        assert!(a - b >= 1, "subtraction answers stay positive");
    }
}

/// The lanes author through the synthetic codec with the full shape
/// validation (every prompt ≤ WINDOW_MAX tokens, `=`-tailed, unsplit
/// operands), and the demote envs are the cell lane's own.
#[test]
fn author_lanes_synthetic() {
    let lanes = author_lanes(&SyntheticCodec).expect("synthetic authoring cannot fail");
    assert_eq!(lanes.add.name, "i1256_add10");
    assert_eq!(lanes.sub.name, "i1256_sub10");
    assert_eq!(lanes.add.demote_env.as_deref(), Some(ADD_DEMOTE_ENV));
    assert_eq!(lanes.sub.demote_env.as_deref(), Some(SUB_DEMOTE_ENV));
    assert_eq!(lanes.add.rules.len(), 15);
    assert_eq!(lanes.sub.rules.len(), 14);
    assert_eq!(lanes.table.len(), 29);
    assert_eq!(
        lanes.table.iter().filter(|r| r.implant == 0).count(),
        15,
        "slot 0 carries every addition row"
    );
}

/// G1 — armed-but-silent ≡ base; demoted-on-a-firing-row ≡ base.
#[test]
fn g1_silent_identity() {
    let _env = ENV_LOCK.lock().expect("env lock");
    let lanes = author_lanes(&SyntheticCodec).unwrap();
    let rig = ArmsRig::mint(&lanes);
    let mut source = SyntheticSource::new();
    let cell = cell_g1_silent_identity(&mut source, &SyntheticCodec, &rig, 8)
        .expect("cell machinery runs");
    assert!(cell.pass, "G1 failed: {:?}", cell.notes);
    assert_eq!(
        cell.counts
            .iter()
            .find(|(k, _)| *k == "silent_rows")
            .map(|(_, v)| *v),
        Some(14),
        "prose 4 + look-alikes 10"
    );
    assert_eq!(
        cell.counts
            .iter()
            .find(|(k, _)| *k == "silent_fires")
            .map(|(_, v)| *v),
        Some(0),
    );
}

/// G1b — zero fires on every look-alike + prose row.
#[test]
fn g1b_false_fire() {
    let _env = ENV_LOCK.lock().expect("env lock");
    let lanes = author_lanes(&SyntheticCodec).unwrap();
    let rig = ArmsRig::mint(&lanes);
    let cell = cell_g1b_false_fire(&SyntheticCodec, &rig).expect("cell machinery runs");
    assert!(cell.pass, "G1b failed: {:?}", cell.notes);
    assert_eq!(
        cell.counts
            .iter()
            .find(|(k, _)| *k == "total_fires")
            .map(|(_, v)| *v),
        Some(0),
    );
    assert_eq!(
        cell.counts
            .iter()
            .find(|(k, _)| *k == "lookalike_prose_rows")
            .map(|(_, v)| *v),
        Some(14),
    );
}

/// G2 — every fired span emits its forced token exactly; prompt phase
/// bit-identical; look-alike rows walk-identical (no regression).
#[test]
fn g2_fired_exactness() {
    let _env = ENV_LOCK.lock().expect("env lock");
    let lanes = author_lanes(&SyntheticCodec).unwrap();
    let rig = ArmsRig::mint(&lanes);
    let mut source = SyntheticSource::new();
    let cell = cell_g2_fired_exactness(&mut source, &SyntheticCodec, &rig, 8)
        .expect("cell machinery runs");
    assert!(cell.pass, "G2 failed: {:?}", cell.notes);
    assert_eq!(
        cell.counts
            .iter()
            .find(|(k, _)| *k == "exact_forced")
            .map(|(_, v)| *v),
        Some(29),
        "every tabled row exact"
    );
    assert_eq!(
        cell.counts
            .iter()
            .find(|(k, _)| *k == "lookalike_identical")
            .map(|(_, v)| *v),
        Some(10),
    );
}

/// G3 — co-mounted: solo fires, no interference, combined exactness ≥
/// each single, silent rows stay silent.
#[test]
fn g3_composition() {
    let _env = ENV_LOCK.lock().expect("env lock");
    let lanes = author_lanes(&SyntheticCodec).unwrap();
    let rig = ArmsRig::mint(&lanes);
    let mut source = SyntheticSource::new();
    let cell =
        cell_g3_composition(&mut source, &SyntheticCodec, &rig, 8).expect("cell machinery runs");
    assert!(cell.pass, "G3 failed: {:?}", cell.notes);
    assert_eq!(
        cell.counts
            .iter()
            .find(|(k, _)| *k == "combined_exact")
            .map(|(_, v)| *v),
        Some(29),
    );
    assert_eq!(
        cell.counts
            .iter()
            .find(|(k, _)| *k == "no_interference")
            .map(|(_, v)| *v),
        Some(29),
    );
    assert_eq!(
        cell.counts
            .iter()
            .find(|(k, _)| *k == "silent_rows_ok")
            .map(|(_, v)| *v),
        Some(14),
    );
}

/// The demote posture is the exact-literal-`0` env at mount: a non-`0`
/// value arms the implant (the drift_guard posture law, exercised here
/// so the env seam cannot silently invert).
#[test]
fn demote_env_nonzero_arms() {
    let _env = ENV_LOCK.lock().expect("env lock");
    // Prove the runtime's demote resolution is exact-literal: mount with
    // the env set to "1" (armed), then "0" (demoted). The cells' own
    // demote arms cover the "0" side; this covers the non-`0` side.
    let lanes = author_lanes(&SyntheticCodec).unwrap();
    let artifact = riir_infer_core::implants::ImplantArtifact::mint(&lanes.add, 1);
    // SAFETY: the test holds ENV_LOCK and never spawns threads; no other
    // thread reads env concurrently.
    unsafe {
        std::env::set_var(ADD_DEMOTE_ENV, "1");
    }
    let mut rt = riir_infer_core::implants::ImplantRuntime::new(1);
    let res = rt.mount(0, &artifact);
    // SAFETY: same contract.
    unsafe {
        std::env::remove_var(ADD_DEMOTE_ENV);
    }
    assert!(res.is_ok(), "non-zero env must arm: {res:?}");
    assert!(!rt.is_demoted(0));
    // And the firing behavior is live: feed "2+2=" → one fire.
    let toks = SyntheticCodec.encode("2+2=").unwrap();
    let mut fire_buf = Vec::new();
    let mut fires = 0usize;
    for t in toks {
        fire_buf.clear();
        rt.observe_into(t as u32, &mut fire_buf);
        fires += fire_buf.len();
    }
    assert_eq!(fires, 1, "armed implant fires exactly once on its row");
}
