#![cfg(feature = "flashmemory_budget_metal")]
//! Issue 1200 — the FlashMemory budget arm's envelope: the selection
//! count the kernel computes must never exceed the `[FM_MAX_SEL]` row it
//! writes into, and the block count must never exceed the shared-memory
//! slice. Pure arithmetic (no device): the host mirror
//! [`fm_budget_select_count`] is pinned operation-for-operation against
//! the kernel's f32 rule, and the refusal thresholds are pinned to the
//! issue's MEASURED failure points — the 32k ρ0.3 arm panicked
//! host-side at n_blocks 429 (`index 512, len 512`), the 128k ρ0.15 arm
//! WEDGED the device uninterruptible at n_blocks 857.
//!
//! Posture: `--features flashmemory_budget_metal` (the module is
//! feature-gated in lib.rs; this row is the reader+count protection —
//! the whole-file `#![cfg]` compiles to nothing without it, and the
//! `[[test]]` required-features row in Cargo.toml is its pair).

use riir_infer_gpu::flashmemory_budget_metal::{
    FM_MAX_BLOCKS_SMEM, FM_MAX_SEL, fm_budget_blocks_check, fm_budget_envelope_check,
    fm_budget_max_n_blocks, fm_budget_select_count,
};

/// The CPU twin — verbatim from katgpt-attn's `RowRule::Budget`
/// (`select_scored_row`'s count arithmetic): `max(round(ρ·n), 1) − 1`
/// clamped to `n − 1`, plus the current block. The host mirror must
/// equal this at every (n, rho) — twin-faithfulness is the gate.
fn cpu_budget_count(n: usize, rho: f32) -> usize {
    let k = (((rho * n as f32).round() as usize).max(1) - 1).min(n.saturating_sub(1));
    k + 1
}

#[test]
fn select_count_matches_the_cpu_rule_across_the_grid() {
    for n in 1usize..=2100 {
        for rho in [0.05f32, 0.15, 0.3, 0.5, 0.75, 1.0] {
            assert_eq!(
                fm_budget_select_count(n, rho) as usize,
                cpu_budget_count(n, rho),
                "mirror diverged from the CPU rule at n={n} rho={rho}"
            );
        }
    }
}

#[test]
fn select_count_pins_the_issue_measured_thresholds() {
    // The 32k panic config: rho 0.3 x 512 blocks -> 154 picks (the issue's
    // measured overflow; the panic was the host reader walking 129..154
    // past the 128-wide row).
    assert_eq!(fm_budget_select_count(512, 0.3), 154);
    // The bankable clean arm: rho 0.15 x 512 -> 77 (no overflow).
    assert_eq!(fm_budget_select_count(512, 0.15), 77);
    // The exact boundaries: LAST passing / FIRST overflowing block count.
    assert_eq!(fm_budget_select_count(428, 0.3) as usize, FM_MAX_SEL);
    assert_eq!(fm_budget_select_count(429, 0.3) as usize, FM_MAX_SEL + 1);
    // The 128k wedge threshold: the frozen log line was pos 54,784
    // (n_blocks 856, count 128 — clean); the NEXT token opened block 857
    // (count 129) and the device wedged.
    assert_eq!(fm_budget_select_count(856, 0.15) as usize, FM_MAX_SEL);
    assert_eq!(fm_budget_select_count(857, 0.15) as usize, FM_MAX_SEL + 1);
    // Degenerate rows: n=1 keeps only the current block; small n clamps
    // to the whole row.
    assert_eq!(fm_budget_select_count(1, 0.9), 1);
    assert_eq!(fm_budget_select_count(2, 0.9), 2);
}

#[test]
#[should_panic(expected = "budget rule needs 154 selection slots")]
fn envelope_check_refuses_the_32k_panic_config() {
    fm_budget_envelope_check("test", 512, 0.3);
}

#[test]
#[should_panic(expected = "budget rule needs 129 selection slots")]
fn envelope_check_refuses_the_128k_wedge_config() {
    fm_budget_envelope_check("test", 857, 0.15);
}

#[test]
fn envelope_check_passes_the_in_envelope_configs() {
    // The pre-registered operating point and the bankable clean arms —
    // these must NEVER refuse (the refusal is byte-neutral below the cap).
    fm_budget_envelope_check("test", 512, 0.15);
    fm_budget_envelope_check("test", 428, 0.3);
    fm_budget_envelope_check("test", 856, 0.15);
    fm_budget_envelope_check("test", 1, 0.15);
}

#[test]
fn max_n_blocks_is_maximal_for_the_operating_rhos() {
    for rho in [0.05f32, 0.15, 0.3, 0.5, 0.9] {
        let max = fm_budget_max_n_blocks(rho);
        assert!(max > 1, "degenerate max at rho={rho}");
        assert!(
            fm_budget_select_count(max, rho) as usize <= FM_MAX_SEL,
            "max not fitting at rho={rho}: n={max}"
        );
        assert!(
            fm_budget_select_count(max + 1, rho) as usize > FM_MAX_SEL,
            "max not maximal at rho={rho}: n={max} (n+1 also fits)"
        );
    }
    // The named operating points, pinned: block-64 envelopes 54.8K at
    // rho 0.15 and 27.4K at rho 0.3 (the issue's honest envelope finding).
    assert_eq!(fm_budget_max_n_blocks(0.15), 856);
    assert_eq!(fm_budget_max_n_blocks(0.3), 428);
}

#[test]
#[should_panic(expected = "2049 blocks > the 2048 shared-memory ceiling")]
fn blocks_check_refuses_past_the_smem_slice() {
    fm_budget_blocks_check("test", FM_MAX_BLOCKS_SMEM + 1);
}

#[test]
fn blocks_check_passes_the_ceiling_and_below() {
    fm_budget_blocks_check("test", FM_MAX_BLOCKS_SMEM);
    fm_budget_blocks_check("test", 1);
}
