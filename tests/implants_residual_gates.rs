//! The Issue-1256 T1 rung-2 gates: the residual-overlay write (the
//! `Gr` cell at the fixture tier) + the exact-delta assertion (the
//! fixture fold equals the independently recomputed `B·(A·key)`) + the
//! residual artifact roundtrip (canonical bytes, the v2 digest law).
//! The real tier runs the SAME `Gr` cell through the SAME code in the
//! `implants_cell` bin over the league checkpoint. The alloc census is
//! its own binary (`implants_residual_g4` — a per-binary counting
//! allocator, the `implants_g4_alloc` shape).

use katgpt_core::quant_error_lora::QuantErrorLora;
use riir_infer_core::closed_form_write::{WriteOverlays, mount};
use riir_infer_core::implants::{
    ImplantAction, ImplantArtifact, ImplantError, ResidualOverlay, ResidualPair, TokenPat,
    TriggerRule,
};
use riir_infer_core::implants_cell::{
    ArmsRig, FixtureCodec, LogitsSource, MountPosture, ResidualRungMode, ResidualSiteDims,
    SYNTH_SITE_IN, SyntheticCodec, SyntheticSource, WalkBufs, author_lanes, cell_gr_residual_write,
    residual_overlay, residual_variant_lanes,
};

const RANK: u32 = 2;
const SCALE: f32 = 0.05;
const SEED: u64 = 0x1256_9880_5EED_00A2;
const SY_VOCAB: usize = 48;

fn fixture_dims() -> ResidualSiteDims {
    ResidualSiteDims {
        in_dim: SYNTH_SITE_IN,
        out_dim: SY_VOCAB,
        layer: 0,
    }
}

/// The overlay's forward-facing table (the cell-tier construction: the
/// spec's raw pairs become `QuantErrorLora`s, `mount`ed per layer — the
/// same `build_residual_table` path the walk uses).
fn build_table(ov: &ResidualOverlay) -> WriteOverlays {
    let rows: Vec<(usize, QuantErrorLora)> = ov
        .pairs
        .iter()
        .map(|p| {
            (
                ov.layer as usize,
                QuantErrorLora {
                    a: p.a.clone(),
                    b: p.b.clone(),
                    alpha: p.alpha,
                    rank: p.rank as usize,
                    in_dim: p.in_dim(),
                    out_dim: p.out_dim(),
                },
            )
        })
        .collect();
    mount(rows)
}

/// The residual overlay validates its shape (rank-divisible rows, shared
/// widths) and the variant lanes re-validate through the spec
/// constructor (a bias-less action is legal only with a residual).
#[test]
fn residual_shapes_validate() {
    let dims = fixture_dims();
    let ov = residual_overlay(&dims, RANK, SEED, SCALE).expect("overlay authors");
    assert_eq!(ov.layer, 0);
    assert_eq!(ov.pairs.len(), 1);
    assert_eq!(ov.in_dim(), SYNTH_SITE_IN);
    assert_eq!(ov.out_dim(), SY_VOCAB);
    // Refusals.
    assert!(ResidualPair::new(0, 1.0, vec![1.0; 4], vec![1.0; 4]).is_err());
    assert!(ResidualPair::new(2, 1.0, vec![1.0; 5], vec![1.0; 4]).is_err());
    assert!(ResidualPair::new(2, f32::NAN, vec![1.0; 4], vec![1.0; 4]).is_err());
    assert!(ResidualOverlay::new(0, vec![]).is_err());
    let p1 = ResidualPair::new(2, 1.0, vec![1.0; 8], vec![1.0; 48]).unwrap();
    let p2 = ResidualPair::new(2, 1.0, vec![1.0; 4], vec![1.0; 48]).unwrap();
    assert!(
        ResidualOverlay::new(0, vec![p1, p2]).is_err(),
        "mixed widths refuse"
    );
    // A write-less action refuses; a bias-less action with a residual is
    // the legal pure-residual shape.
    let lanes = author_lanes(&SyntheticCodec).expect("author");
    let mut rules = lanes.add.rules.clone();
    rules.truncate(1);
    assert!(
        riir_infer_core::implants::ImplantSpec::new(
            "w",
            rules.clone(),
            vec![ImplantAction {
                biases: vec![],
                residual: None
            }],
            None
        )
        .is_err(),
        "a write-less action refuses"
    );
    assert!(
        riir_infer_core::implants::ImplantSpec::new(
            "w",
            rules,
            vec![ImplantAction {
                biases: vec![],
                residual: Some(ov),
            }],
            None,
        )
        .is_ok(),
        "the pure-residual action is legal"
    );
}

/// The residual artifact roundtrips canonically (v2): decode ⟹ re-mint
/// byte-identical (determinism); the bias-less actions survive the
/// roundtrip; a tampered payload refuses at the digest (before parse).
#[test]
fn residual_artifact_roundtrip_canonical() {
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
    let art = ImplantArtifact::mint(&res.add, 1);
    assert_eq!(art.format_version, 2, "the residual rung rides format v2");
    let decoded = art.decode().expect("canonical decode");
    assert!(decoded.actions.iter().all(|a| a.biases.is_empty()));
    assert!(decoded.actions.iter().all(|a| a.residual.is_some()));
    let remint = ImplantArtifact::mint(&decoded, 1);
    assert_eq!(remint.canonical, art.canonical, "re-mint byte-identical");
    // Tamper: flip one payload byte after the header — the digest check
    // fires BEFORE any parse (the corruption-is-stronger law).
    let mut tampered = art.clone();
    tampered.canonical[art.canonical.len() - 1] ^= 0xFF;
    assert!(
        tampered.decode().is_err(),
        "a tampered residual artifact refuses at the digest"
    );
}

/// **The exact-delta assertion (fixture tier):** arming the overlay for
/// one step changes that step's logits by EXACTLY `B·(A·key)` —
/// recomputed independently in this test from the site key and the
/// authored pair rows (bit-for-bit, per-component), and the one-shot
/// consumption is byte-verified (the step after arms `None` ≡ base).
#[test]
fn residual_fold_delta_exact() {
    let dims = fixture_dims();
    let ov = residual_overlay(&dims, RANK, SEED, SCALE).expect("overlay authors");
    let table = build_table(&ov);
    let mut src = SyntheticSource::new();
    let vocab = src.vocab();
    let (token, pos) = (12usize, 3usize); // '=' at a mid-prompt position

    // Base step.
    let mut base = vec![0.0_f32; vocab];
    src.arm_residual(None);
    src.step(token, pos, &mut base);
    // Armed step (the fold lands).
    let mut armed = vec![0.0_f32; vocab];
    src.arm_residual(Some(table.clone()));
    src.step(token, pos, &mut armed);
    // Consumed (one-shot): identical to base.
    let mut consumed = vec![0.0_f32; vocab];
    src.arm_residual(None);
    src.step(token, pos, &mut consumed);
    assert_eq!(
        base, consumed,
        "the one-shot overlay is consumed by its step"
    );

    // The independent recomputation: key = site_key_for(token, pos),
    // delta[v] = alpha · Σ_r b[v][r] · (A·key)[r] — mirroring the
    // CONSUMED implementation's exact accumulation order (the 4-lane
    // unrolled dot in `apply_correction_into`; the consumed impl is the
    // authority, the recomputation mirrors it bit-for-bit).
    let key = src.site_key_for(token, pos);
    let pair: &ResidualPair = &ov.pairs[0];
    let rank = pair.rank as usize;
    let in_dim = pair.in_dim();
    assert_eq!(in_dim, SYNTH_SITE_IN);
    let mut scratch = vec![0.0_f32; rank];
    for (k, sslot) in scratch.iter_mut().enumerate() {
        let row = &pair.a[k * in_dim..(k + 1) * in_dim];
        let mut acc = 0.0_f32;
        let mut i = 0;
        while i + 4 <= in_dim {
            acc += row[i] * key[i]
                + row[i + 1] * key[i + 1]
                + row[i + 2] * key[i + 2]
                + row[i + 3] * key[i + 3];
            i += 4;
        }
        while i < in_dim {
            acc += row[i] * key[i];
            i += 1;
        }
        *sslot = acc;
    }
    let mut expected = [0.0_f32; SY_VOCAB];
    for (v, slot) in expected.iter_mut().enumerate() {
        let mut acc = 0.0_f32;
        for (r, s) in scratch.iter().enumerate() {
            acc += pair.b[v * rank + r] * s;
        }
        *slot = pair.alpha * acc;
    }
    for v in 0..SY_VOCAB {
        // The exact claim under the impl's own rounding: the armed logit
        // equals base + fold computed as ONE f32 add (`y[o] += scale·acc`).
        // (Recovering the fold as `armed − base` is inexact in float
        // arithmetic — the add/subtract pair rounds twice — so the
        // comparison is on the armed sum, never the subtracted delta.)
        let want = base[v] + expected[v];
        assert_eq!(
            armed[v].to_bits(),
            want.to_bits(),
            "component {v}: armed logit != base + fold (independent recomputation)"
        );
    }
}

/// **The stateless-tier re-convergence (fixture-only):** the synthetic
/// source is a pure function of (token, pos) — after the one-shot fold
/// is consumed, the next silent step is byte-identical to base again.
/// (A stateful forward persists the fold's effect in its KV — the real
/// tier's post-fire divergence is disclosed by the cell, never failed.)
#[test]
fn gr_stateless_reconvergence_fixture() {
    use riir_infer_core::implants_cell::decode_walk;
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
    let base_rig = ArmsRig::mint(&lanes);
    let res_rig = ArmsRig::mint(&res);
    let toks = SyntheticCodec
        .encode(&rig_first_row())
        .expect("encode row0");
    let mut src = SyntheticSource::new();
    let bufs = &mut WalkBufs::new(src.vocab());
    let mut rt_base = base_rig.new_runtime(MountPosture::None).expect("mount");
    let base = decode_walk(&mut src, &toks, 8, &mut rt_base, bufs);
    let mut rt_res = res_rig.new_runtime(MountPosture::Armed).expect("mount");
    let armed = decode_walk(&mut src, &toks, 8, &mut rt_res, bufs);
    let f = toks.len() - 1;
    assert_eq!(
        armed.residual_arms,
        armed.fires.len(),
        "per-fire arming holds"
    );
    assert_eq!(armed.residual_arms, 1, "one fire on this row");
    assert_eq!(
        armed.step_digests[..=f],
        base.step_digests[..=f],
        "fire-step identity"
    );
    assert_ne!(
        armed.step_digests[f + 1],
        base.step_digests[f + 1],
        "the fold landed"
    );
    assert_eq!(armed.tokens, base.tokens, "the nudge held the token path");
    assert_eq!(
        armed.step_digests[f + 2..],
        base.step_digests[f + 2..],
        "stateless source: the walk re-converges after the consumed one-shot"
    );
}

fn rig_first_row() -> String {
    let (a, b) = riir_infer_core::implants_cell::add_pairs()[0];
    format!("{a}+{b}=")
}

/// The Gr cell passes at the fixture tier (silent identity, fired
/// one-shot, co-write, demoted arm, no stale arming).
#[test]
fn gr_residual_cell_fixture() {
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _env = ENV_LOCK.lock().expect("env lock");
    let lanes = author_lanes(&SyntheticCodec).expect("author");
    let mut src = SyntheticSource::new();
    let cell = cell_gr_residual_write(
        &mut src,
        &SyntheticCodec,
        &lanes,
        &fixture_dims(),
        RANK,
        SCALE,
        8,
    )
    .expect("cell runs");
    assert!(cell.pass, "Gr cell must pass: {:?}", cell.notes);
    assert_eq!(
        cell.counts
            .iter()
            .find(|(k, _)| *k == "gr2_argmax_flip_disclosed")
            .map(|(_, v)| *v),
        Some(0),
        "the nudge posture holds at the fixture (no flip)"
    );
}

/// **The 046 view-enumeration certificate (fixture tier):** the fire
/// gate over the authored lanes certifies exhaustively — the fire set
/// fires exactly, the FULL distance-1 boundary shell is enumerated with
/// margins (shell views whose firing suffix survives fire LEGALLY —
/// the suffix-anchor design, disclosed as a counter, never folded),
/// the authored intent surface stays silent, and the runtime AGREES
/// with the certificate over every enumerated view (the exhaustive
/// G1b/G2 upgrade on live code).
#[test]
fn cert_view_enumeration_fixture() {
    use riir_infer_core::implants::FireRef;
    use riir_infer_core::implants_cert::{CertOrigin, SpecRef, enumerate_views};
    use std::sync::Mutex;
    let lanes = author_lanes(&SyntheticCodec).expect("author");
    let rig = ArmsRig::mint(&lanes);
    let specs = [
        SpecRef {
            slot: 0,
            name: lanes.add.name.clone(),
            digest: rig.add.digest,
        },
        SpecRef {
            slot: 1,
            name: lanes.sub.name.clone(),
            digest: rig.sub.digest,
        },
    ];
    let impls = [&lanes.add, &lanes.sub];
    // Authored views: the look-alikes + prose rows (the intent surface).
    let authored: Vec<Vec<u32>> = riir_infer_core::implants_cell::LOOKALIKE_PROMPTS
        .iter()
        .chain(riir_infer_core::implants_cell::PROSE_PROMPTS.iter())
        .map(|s| SyntheticCodec.encode(s).expect("encode authored view"))
        .map(|toks| toks.iter().map(|&t| t as u32).collect())
        .collect();
    let enum_ = enumerate_views(&specs, &impls, &authored).expect("the fixture gate certifies");
    assert_eq!(enum_.missed_fires.len(), 0);
    assert_eq!(enum_.authored_fires.len(), 0);
    assert_eq!(enum_.structural.len(), 0);
    let shell_fires = enum_.shell_fires();
    let mut cert = enum_.cert;
    cert.canonicalize().expect("canonicalize");
    let census = cert.census();
    // The fire set = every authored rule pattern (one per row: 29) —
    // plus the shell's legal suffix-anchor fires (disclosed, not
    // counted here).
    let fireset_rows = cert
        .views
        .iter()
        .filter(|v| v.origin == CertOrigin::FireSet)
        .count();
    assert_eq!(fireset_rows, 29, "the fire set is the 29 authored patterns");
    assert!(census.firing >= 29);
    // The shell is exhaustive and non-empty (the boundary is real).
    assert!(census.shell > 0, "the boundary shell must be enumerated");
    // Boundary-silent rows exist (margin-1 shell views) — recorded
    // honestly; certified-silent rows (margin >= 2) too.
    assert!(census.boundary_silent > 0, "margin-1 shell rows exist");
    assert!(census.certified_silent > 0, "certified-silent rows exist");
    // The suffix-anchor surface is disclosed: shell views whose firing
    // suffix survived (or landed on another pattern) — legal fires,
    // counted from the rows, never folded into the silence classes.
    assert!(shell_fires > 0, "the suffix-anchor surface is non-empty");
    // The silence classes exclude them (census sanity).
    assert_eq!(
        census.firing,
        29 + shell_fires,
        "firing = fire-set members + legal shell fires"
    );
    // Every authored view stayed silent (the intent surface).
    assert!(
        cert.views
            .iter()
            .filter(|v| v.origin == CertOrigin::Authored)
            .all(|v| v.decision.is_none())
    );
    // The digest binds the artifacts.
    let digest1 = cert.digest;
    let specs2 = [
        SpecRef {
            slot: 0,
            name: lanes.add.name.clone(),
            digest: [0xAA; 32],
        },
        SpecRef {
            slot: 1,
            name: lanes.sub.name.clone(),
            digest: rig.sub.digest,
        },
    ];
    let mut enum2 = enumerate_views(&specs2, &impls, &authored).expect("enumerate 2");
    enum2.cert.canonicalize().expect("canonicalize 2");
    assert_ne!(
        digest1, enum2.cert.digest,
        "the digest binds the artifact digests"
    );
    // The weakest-tenth list is non-empty and silent.
    let weak = cert.weakest_tenth();
    assert!(!weak.is_empty());
    assert!(
        weak.iter().all(|v| v.decision.is_none()),
        "the weakest rows are silent views"
    );
    // The sentinel sits outside the alphabet.
    assert!(!cert.alphabet.contains(&cert.sentinel()));

    // **The runtime agreement check:** every enumerated view through
    // the REAL gate (a fresh runtime per view — serial; the census is
    // thousands of views, each a handful of observe steps).
    let mismatches = {
        let probe = Mutex::new((rig, SyntheticSource::new()));
        enum2.cert.views.len(); // keep the moved enum alive for the closure shape
        cert_view_runtime_check(&cert, |view| {
            let (rig, _src) = &mut *probe.lock().expect("probe");
            let mut rt = rig.new_runtime(MountPosture::Armed).expect("mount");
            let mut buf: Vec<FireRef> = Vec::new();
            for &t in view {
                buf.clear();
                rt.observe_into(t, &mut buf);
            }
            buf.iter()
                .map(|f| (f.implant, f.action))
                .collect::<Vec<_>>()
        })
    };
    assert_eq!(
        mismatches, 0,
        "the runtime must agree with the certificate over every enumerated view"
    );
}

/// Drive the real runtime per cert view (extracted so the borrow of
/// the runtime factory stays simple).
fn cert_view_runtime_check<F>(
    cert: &riir_infer_core::implants_cert::ImplantCert,
    mut observe: F,
) -> usize
where
    F: FnMut(&[u32]) -> Vec<(u16, u16)>,
{
    let mut mismatches = 0usize;
    for v in &cert.views {
        let fired = observe(&v.tokens);
        let runtime_decision = fired.first().copied();
        match (v.decision, runtime_decision) {
            (None, None) => {}
            (Some(d), Some(r)) if d == r => {}
            _ => mismatches += 1,
        }
    }
    mismatches
}

/// The refusal law (defense in depth): the bounded-window mount law
/// refuses an over-WINDOW pattern at the CONSTRUCTOR (the gate can
/// never exist), so the cert's own re-assertion is unreachable by
/// construction — this test pins the constructor half (the law's
/// load-bearing half) and documents why the cert half cannot fire.
#[test]
fn cert_refusal_law_constructor_half() {
    let long: Vec<TokenPat> = std::iter::repeat_n(
        TokenPat::Exact(1),
        riir_infer_core::implants::WINDOW_MAX + 1,
    )
    .collect();
    let err = riir_infer_core::implants::ImplantSpec::new(
        "unbounded",
        vec![TriggerRule {
            pattern: long,
            action: 0,
        }],
        vec![ImplantAction {
            biases: vec![(2, 1.0)],
            residual: None,
        }],
        None,
    )
    .expect_err("over-WINDOW patterns refuse at the constructor");
    assert!(
        matches!(err, ImplantError::Usage(_)),
        "the refusal is the typed Usage class"
    );
}

/// A u32::MAX closure alphabet member collides with the sentinel —
/// the enumeration refuses loud.
#[test]
fn cert_refuses_sentinel_collision() {
    use riir_infer_core::implants_cert::{SpecRef, enumerate_views};
    let spec = riir_infer_core::implants::ImplantSpec::new(
        "max",
        vec![TriggerRule {
            pattern: vec![TokenPat::Exact(u32::MAX)],
            action: 0,
        }],
        vec![ImplantAction {
            biases: vec![(1, 1.0)],
            residual: None,
        }],
        None,
    )
    .expect("spec");
    let specs = [SpecRef {
        slot: 0,
        name: "max".into(),
        digest: [0; 32],
    }];
    let err = enumerate_views(&specs, &[&spec], &[]).expect_err("sentinel collision refuses");
    assert!(
        err.contains("sentinel"),
        "refusal names the collision: {err}"
    );
}
