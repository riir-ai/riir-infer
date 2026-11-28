#![cfg(feature = "mtp_draft_block")]
//! Issue 1253 T5.3 (plan 639) — the GGUF-embedded draft block's
//! drafter-seat gates (synthetic, pure-CPU; the real twin's measured
//! cells are T5.4/T5.5, quiet-window-only, NOT gates).
//!
//! Pins the seat's mechanics on the synthetic folded twin (the T5.1
//! fixture's 15-tensor draft block + the kit's folded ternary main
//! stack):
//! 1. THE CARD PRESETS — NVIDIA 3×0.10 maps; Apple k∈1..=8 maps; the
//!    domain refusals (k=0/9, min_prob 0/1.5) refuse at construction.
//! 2. THE GREEDY CHAIN — deterministic (bit-identical across re-seeds
//!    of the same state), advances next_pos by k, pairs finite with the
//!    mass in (0, 1].
//! 3. THE SEAT STATE — the min-prob gate reads the preset live (a swap
//!    starves the very next chain), no sticky gate state, reseeds are
//!    clean and bit-identical (the conditioning premise itself is a
//!    T5.2 pin — this fixture's synthetic head carries an
//!    input-invariant top tie class; disclosed at the arm).
//! 4. THE PROPOSE COMPOSITION — d_0 = the echo, d_j = chain step j;
//!    truncates to ctx.k; the unseeded/stale posture degrades to the
//!    lossless echo with take_error set (the 2-stage runner's posture).
//! 5. THE MIN-PROB GATE — min_prob → 1 starves the chain (no point
//!    masses on a real-valued fixture); a mid floor cuts the chain at
//!    the first step under it; None runs it full.
//! 6. THE FRESH-WINDOW RESET — reset() clears the seed cursor + carried
//!    error and the post-reset chain reproduces the fresh-drafter chain
//!    bit-identically (the T5.4 economics harness's per-window posture).

#[path = "../../../tests/common/mtp_draft_fixture.rs"]
mod fixture;

use std::sync::Arc;

use riir_infer_core::deltanet::forward::{HybridCache, HybridForwardScratch, effective_rotary_dim};
use riir_infer_core::deltanet::rotation::rotate_inverse_inplace;
use riir_infer_core::deltanet::ternary_forward::forward_qwen_deltanet_ternary;
use riir_infer_core::deltanet::ternary_weights::QwenDeltaNetTernaryWeights;
use riir_infer_core::gguf_loader::{
    GgufFile, load_mtp_draft_block, load_qwen_deltanet_ternary_from_gguf, mtp_twin_census,
};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::types::Config;
use riir_infer_dist::spec_fill::{DraftCtx, DraftSource};
use riir_infer_dist::spec_mtp_block::{MAX_DRAFT_K, MtpDraftBlockDrafter, MtpDraftPreset};

use std::io::Write as _;

fn write_tmp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("{name}_{}.gguf", std::process::id()));
    let mut f = std::fs::File::create(&path).expect("create temp gguf");
    f.write_all(bytes).expect("write temp gguf");
    path
}

fn rope_table(config: &Config) -> RopeFreqTable {
    let rotary_dim = effective_rotary_dim(config);
    RopeFreqTable::new(config.rope_theta, rotary_dim)
}

/// Land `toks` on the synthetic main stack; return the primal hidden of
/// the LAST position (the seed feed the full-stack posture owes the
/// seat — the Issue-1111 recovery recipe, the module basis contract).
fn land_main_stack(
    config: &Config,
    weights: &QwenDeltaNetTernaryWeights,
    toks: &[usize],
) -> Vec<f32> {
    let n = config.n_embd;
    let rope = rope_table(config);
    let mut cache = HybridCache::with_layer_types(config, &weights.layer_types);
    let mut scratch = HybridForwardScratch::new(config);
    let mut x = vec![0.0f32; config.vocab_size.max(n)];
    for (pos, tok) in toks.iter().enumerate() {
        forward_qwen_deltanet_ternary(
            &mut x,
            weights,
            &mut cache,
            *tok,
            pos,
            config,
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
    h
}

fn load_twin(
    unique: &str,
) -> (
    std::path::PathBuf,
    Config,
    Arc<QwenDeltaNetTernaryWeights>,
    Arc<riir_infer_core::gguf_loader::MtpDraftBlock>,
) {
    let path = write_tmp(unique, &fixture::declared_twin_bytes());
    let gguf = GgufFile::open(&path).expect("open twin");
    let census = mtp_twin_census(&gguf);
    assert!(census.draft_admittable());
    let (config, weights): (Config, QwenDeltaNetTernaryWeights) =
        load_qwen_deltanet_ternary_from_gguf(&gguf, "").expect("main stack loads");
    let block = load_mtp_draft_block(&gguf, &config).expect("draft block admits");
    (path, config, Arc::new(weights), Arc::new(block))
}

/// Arm 1 — the card presets map + the domain refusals.
#[test]
fn presets_map_the_card_cells() {
    let nv = MtpDraftPreset::nvidia_draft3();
    assert_eq!(nv.k, 3, "NVIDIA card = 3 draft tokens");
    assert_eq!(nv.min_prob, Some(0.10), "NVIDIA card min-prob 0.10");
    for k in 1..=MAX_DRAFT_K {
        let p = MtpDraftPreset::apple(k).expect("apple card admits 1..=8");
        assert_eq!(p.k, k);
        assert_eq!(p.min_prob, None, "Apple card carries no chain floor");
    }
    assert!(MtpDraftPreset::apple(0).is_err(), "k=0 refuses");
    assert!(
        MtpDraftPreset::apple(MAX_DRAFT_K + 1).is_err(),
        "k>8 refuses"
    );
    assert!(
        MtpDraftPreset::new(3, Some(0.0)).is_err(),
        "min_prob 0 refuses"
    );
    assert!(
        MtpDraftPreset::new(3, Some(1.5)).is_err(),
        "min_prob above 1 refuses"
    );
    let ok = MtpDraftPreset::new(4, Some(1.0)).expect("min_prob 1.0 sits in the domain");
    assert_eq!(ok.min_prob, Some(1.0));
}

/// Arm 2 — the greedy chain: deterministic, advancing, finite.
#[test]
fn chain_deterministic_and_advances() {
    let (path, config, weights, block) = load_twin("riir_1253_t53_det");
    let h = land_main_stack(&config, &weights, &[7, 11, 23]);
    let k = 4;
    let mut d = MtpDraftBlockDrafter::new(
        block,
        weights,
        config.clone(),
        MtpDraftPreset::apple(k).unwrap(),
    )
    .expect("seat");
    d.seed(&h, 23, 3).expect("seed");
    let c1 = d.draft_chain();
    assert_eq!(c1.len(), k, "ungated chain runs full k");
    assert_eq!(d.next_pos(), 3 + k, "each step advances the draft position");
    for (tok, mass) in &c1 {
        assert!(
            mass.is_finite() && *mass > 0.0 && *mass <= 1.0,
            "mass in (0,1]"
        );
        assert!((*tok as usize) < config.vocab_size, "token in vocab");
    }
    // Re-seed the SAME state → bit-identical chain (the step's own
    // determinism pin composed into the seat; the cache rows the first
    // chain wrote at 3..3+k are rewritten identically before read).
    d.seed(&h, 23, 3).expect("reseed");
    let c2 = d.draft_chain();
    assert_eq!(c1, c2, "bit-identical across re-seeds");
    drop(path); // the twin file lives to the end of the arm's scope
}

/// Arm 2b — the T5.5 G4 build item (plan 639): `draft_chain_into`
/// (scratch reuse) is byte-identical to `draft_chain`, clears stale
/// contents first, and reuses the caller's allocation across rounds.
#[test]
fn draft_chain_into_matches_draft_chain_and_reuses() {
    let (path, config, weights, block) = load_twin("riir_1253_t53_chain_into");
    let h = land_main_stack(&config, &weights, &[7, 11, 23]);
    let k = 4;
    let mut d = MtpDraftBlockDrafter::new(
        block,
        weights,
        config.clone(),
        MtpDraftPreset::apple(k).unwrap(),
    )
    .expect("seat");
    d.seed(&h, 23, 3).expect("seed");
    let want = d.draft_chain();
    // Stale contents are cleared, never appended to.
    let mut scratch_buf = vec![(u32::MAX, f32::NAN); 9];
    d.seed(&h, 23, 3).expect("reseed");
    d.draft_chain_into(&mut scratch_buf);
    assert_eq!(scratch_buf, want, "into == draft_chain (stale cleared)");
    // The allocation is REUSED across rounds: capacity never shrinks,
    // and the second round lands in the same buffer byte-identically.
    let cap_after_first = scratch_buf.capacity();
    d.seed(&h, 23, 3).expect("reseed 2");
    d.draft_chain_into(&mut scratch_buf);
    assert_eq!(scratch_buf, want, "bit-identical across into-rounds");
    assert_eq!(scratch_buf.capacity(), cap_after_first, "no realloc churn");
    drop(path);
}

/// Arm 3 — seat state: live preset reads, no sticky gate state, clean
/// reseeds (the chain-conditioning premise itself is NOT expressible on
/// this fixture — its synthetic draft head carries an input-invariant
/// tie class at the top (mass 1/12 at every step, invariant to the
/// compose embed); the step-level conditioning is the T5.2
/// compose-order probe's pin, the seat passes the token through
/// verbatim).
#[test]
fn seat_state_preset_swaps_and_clean_reseeds() {
    let (path, config, weights, block) = load_twin("riir_1253_t53_seed");
    let h = land_main_stack(&config, &weights, &[7, 11, 23]);
    let mut d =
        MtpDraftBlockDrafter::new(block, weights, config, MtpDraftPreset::apple(4).unwrap())
            .expect("seat");
    d.seed(&h, 23, 3).expect("seed");
    let full = d.draft_chain();
    assert_eq!(full.len(), 4, "ungated chain runs full k");

    // The gate reads the preset LIVE — swap to the starved posture
    // without reseeding and the very next chain is gated out.
    d.set_preset(MtpDraftPreset::new(4, Some(1.0)).unwrap());
    assert_eq!(d.draft_chain().len(), 0, "live preset: 1.0 floor starves");

    // No sticky gate state: back to the Apple posture + reseed → full k
    // again, and bit-identical to the original chain (the cache rows the
    // earlier chains wrote at 3..7 are rewritten identically before
    // read — the same determinism arm 2 pins).
    d.set_preset(MtpDraftPreset::apple(4).unwrap());
    d.seed(&h, 23, 3).expect("reseed");
    let again = d.draft_chain();
    assert_eq!(again.len(), 4, "no sticky gate state");
    assert_eq!(again, full, "bit-identical across preset round-trips");

    // A different seed state is accepted (no refusal) and chains
    // deterministically of its own.
    let mut h2 = h.clone();
    h2[0] = -h2[0];
    d.seed(&h2, 23, 3).expect("seed h2");
    let c2a = d.draft_chain();
    d.seed(&h2, 23, 3).expect("reseed h2");
    let c2b = d.draft_chain();
    assert_eq!(c2a, c2b, "the h2 chain is deterministic of its own");
    drop(path);
}

/// Arm 4 — propose: echo + chain, truncation, the stale fallback.
#[test]
fn propose_echo_plus_chain_and_stale_fallback() {
    let (path, config, weights, block) = load_twin("riir_1253_t53_propose");
    let h = land_main_stack(&config, &weights, &[7, 11, 23]);
    let mut d =
        MtpDraftBlockDrafter::new(block, weights, config, MtpDraftPreset::apple(3).unwrap())
            .expect("seat");
    // Unseeded (the 2-stage runner's first posture): lossless echo +
    // take_error set.
    let ctx = DraftCtx {
        committed: &[7, 11, 23, 99],
        k: 3,
    };
    let out = d.propose(&ctx);
    assert_eq!(out, vec![99, 99, 99], "unseeded → pure echo");
    let err = d.take_error().expect("the unseeded posture names itself");
    assert!(err.contains("not seeded"), "error names the posture: {err}");

    // Seeded for cursor 3: d_0 = the echo, d_1.. = the chain.
    d.seed(&h, 23, 3).expect("seed");
    let seeded_err = d.take_error();
    assert!(
        seeded_err.is_none(),
        "seed does not clear, but nothing pre-seeded remains: {seeded_err:?}"
    );
    let out = d.propose(&ctx);
    assert_eq!(out.len(), 3, "k=3 proposal");
    assert_eq!(out[0], 99, "d_0 is the echo (the accepted-prefix law)");
    // The chain the propose consumed: re-seed and read it directly —
    // d_j must be chain step j's token.
    d.seed(&h, 23, 3).expect("reseed");
    let chain = d.draft_chain();
    assert_eq!(chain.len(), 3);
    assert_eq!(out[1], chain[1].0, "d_1 = chain step 1");
    assert_eq!(out[2], chain[2].0, "d_2 = chain step 2");

    // ctx.k = 1: the echo alone (chain step 0 is the disclosure read).
    d.seed(&h, 23, 3).expect("reseed");
    let ctx1 = DraftCtx {
        committed: &[7, 11, 23, 99],
        k: 1,
    };
    let out1 = d.propose(&ctx1);
    assert_eq!(out1, vec![99], "k=1 → the echo alone");

    // Stale cursor: seeded for 3, the runner moved on → echo + error.
    d.seed(&h, 23, 3).expect("reseed");
    let _ = d.draft_chain(); // the chain advanced next_pos; seeded_for unchanged
    let ctx_stale = DraftCtx {
        committed: &[7, 11, 23, 99, 55],
        k: 2,
    };
    let out_stale = d.propose(&ctx_stale);
    assert_eq!(out_stale, vec![55, 55], "stale cursor → pure echo");
    assert!(d.take_error().is_some(), "the stale posture names itself");
    drop(path);
}

/// Arm 5 — the min-prob gate starves/cuts the chain.
#[test]
fn min_prob_gate_stops_the_chain() {
    let (path, config, weights, block) = load_twin("riir_1253_t53_gate");
    let h = land_main_stack(&config, &weights, &[7, 11, 23]);
    let mk = |preset| {
        let mut d =
            MtpDraftBlockDrafter::new(block.clone(), weights.clone(), config.clone(), preset)
                .expect("seat");
        d.seed(&h, 23, 3).expect("seed");
        d
    };
    // min_prob 1.0: no point masses on a real-valued fixture → starved.
    let mut d = mk(MtpDraftPreset::new(4, Some(1.0)).unwrap());
    assert_eq!(
        d.draft_chain().len(),
        0,
        "the 1.0 floor starves a real-valued chain"
    );
    // None: full k.
    let mut d = mk(MtpDraftPreset::apple(4).unwrap());
    let full = d.draft_chain();
    assert_eq!(full.len(), 4);
    // A mid floor: just above the ARGMIN step's mass cuts the chain at
    // that step (every earlier mass exceeds the floor — the argmin is
    // unique on a real-valued fixture; ties would cut earlier, never
    // later, so the assert stays sound).
    let j = (0..full.len())
        .min_by(|a, b| full[*a].1.total_cmp(&full[*b].1))
        .unwrap();
    let mp = full[j].1 * 1.000001f32;
    let mut d = mk(MtpDraftPreset::new(4, Some(mp)).unwrap());
    let cut = d.draft_chain();
    assert_eq!(
        cut.len(),
        j,
        "the chain ends before the step under the floor"
    );
    assert_eq!(
        &cut[..],
        &full[..j],
        "the surviving steps are bit-identical"
    );
    drop(path);
}

/// Arm 6 — the fresh-window reset (the T5.4 economics harness's
/// per-window posture): a window boundary must clear the seed cursor,
/// any carried error and the block's KV rows, and the next seed must
/// reproduce the fresh-drafter chain bit-identically. (The stale-row
/// READ difference itself is not expressible on this fixture — its
/// synthetic attention is zeroed — so the pin is the seat state + the
/// bit-identity; the cache-clearing correctness rests on
/// `KVCache::reset` and the harness smoke.)
#[test]
fn reset_clears_window_state_and_matches_fresh() {
    let (path, config, weights, block) = load_twin("riir_1253_t53_reset");
    let h = land_main_stack(&config, &weights, &[7, 11, 23]);
    let mk = || {
        MtpDraftBlockDrafter::new(
            block.clone(),
            weights.clone(),
            config.clone(),
            MtpDraftPreset::apple(4).unwrap(),
        )
        .expect("seat")
    };
    // The fresh reference chain at a LATER seed position (rows below it
    // were never written by this drafter).
    let mut fresh = mk();
    fresh.seed(&h, 23, 5).expect("fresh seed");
    let reference = fresh.draft_chain();
    assert_eq!(reference.len(), 4);

    // A used drafter: full chain at 3 (writes rows 3..7), then a stale
    // propose (the unseeded fallback arms the error path), then RESET.
    let mut used = mk();
    used.seed(&h, 23, 3).expect("used seed");
    assert_eq!(used.draft_chain().len(), 4);
    let ctx = DraftCtx {
        committed: &[1, 2, 3],
        k: 2,
    };
    let _ = used.propose(&ctx); // stale cursor → arms the error
    assert!(used.take_error().is_some(), "the fallback armed the error");
    used.reset();
    assert_eq!(used.seeded_for(), None, "reset clears the seed cursor");
    assert_eq!(used.take_error(), None, "reset clears the carried error");

    // Post-reset chain at the later position == the fresh reference
    // (bit-identical — the reset cleared the rows the earlier chain
    // wrote, so the later seed sees the fresh cache).
    used.seed(&h, 23, 5).expect("post-reset seed");
    let again = used.draft_chain();
    assert_eq!(again, reference, "post-reset chain == fresh-drafter chain");
    drop(path);
}
