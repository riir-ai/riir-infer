//! Issue 1196 T1 — the stage attribution for the synth Bonsai-2 hd-128
//! Metal dense decode NaN (NaN logits from step 16 on the incumbent dense
//! path; the CPU dense lane is clean throughout).
//!
//! Method: walk the synth fixture through the Metal decode forward 40 steps
//! teacher-forced (the repro leg asserts the finding's shape), then re-walk
//! with each coarse decode stage region masked OFF via
//! `set_decode_stage_mask` (the Issue-1132 census — masking a stage SKIPS
//! its dispatch: the residual stream keeps whatever the earlier stages
//! left, which is finite garbage, never NaN). The region whose removal
//! kills the step-16 NaN is the producer.
//!
//! Requires Metal (the compute law: GPU runs on GPU). SKIPs loud off-Metal.
#![cfg(feature = "flashmemory_budget_metal_t2")]

#[path = "../../../tests/common/synth_bonsai2.rs"]
mod synth;

use riir_infer_core::gguf_loader::load_qwen_deltanet_ternary_weights_gguf;
use riir_infer_core::types::Config;
use riir_infer_gpu::ternary_deltanet_gpu_forward::{
    DECODE_STAGE_ALL, TernaryDeltanetGpuForward, set_decode_stage_mask,
};

/// The finding's window: NaNs from step 16 on. 40 steps = 24 NaN steps of
/// evidence — short by the compute law (Metal decode on a 4-layer synth is
/// sub-second per walk).
const STEPS: usize = 40;
const NAN_FROM: usize = 16;

/// The GPU forward's KV working-set clamp (the 1191 T2 pattern).
const GPU_CTX_CLAMP: usize = 8192;

/// Coarse region masks: each leg leaves ONE region off, everything else on.
/// Bits: 0 input_norm · 1 deltanet · 2 attention · 3 mid_norm · 4 ffn ·
/// 20 tail_norm · 21 tail_lm_head (the sub-bits refine parents and are not
/// needed at coarse resolution).
const REGIONS: [(&str, u32, u32); 5] = [
    ("deltanet", 1, 1u32 << 1),
    ("attention", 2, 1u32 << 2),
    ("ffn", 4, 1u32 << 4),
    ("tail_norm", 20, 1u32 << 20),
    ("tail_lm_head", 21, 1u32 << 21),
];

/// The DeltaNet region's OWN census bits (5-10, 15-19 refine parent bit 1 —
/// consulted only when bit 1 is on). The T1 verdict names the coarse
/// producer; this pass splits it: mask each sub-bit off (parent stays ON)
/// and see which removal kills the NaN. Bits not listed run in every leg.
const GDN_SUBS: [(&str, u32, u32); 11] = [
    ("gdn_proj", 5, 1u32 << 5),
    ("gdn_prerec", 6, 1u32 << 6),
    ("gdn_recurrence", 7, 1u32 << 7),
    ("gdn_outproj", 8, 1u32 << 8),
    ("gdn_proj_concat", 9, 1u32 << 9),
    ("gdn_proj_dense", 10, 1u32 << 10),
    ("gdn_pre_conv", 15, 1u32 << 15),
    ("gdn_pre_betal2", 16, 1u32 << 16),
    ("gdn_out_zgate", 17, 1u32 << 17),
    ("gdn_out_rot", 18, 1u32 << 18),
    ("gdn_out_gemv", 19, 1u32 << 19),
];

fn load_synth() -> (
    Config,
    riir_infer_core::deltanet::QwenDeltaNetTernaryWeights,
) {
    let metadata = synth::synth_metadata(true);
    let tensors = synth::synth_tensors_bonsai2();
    let bytes = synth::build_gguf(&metadata, &tensors);
    let path = synth::write_tmp("i1196_synth_folded.gguf", &bytes);
    load_qwen_deltanet_ternary_weights_gguf(&path).expect("synth folded fixture load")
}

fn token_stream(len: usize, vocab: usize) -> Vec<usize> {
    let mut state = 0x1196_5A17u64;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as usize % vocab
        })
        .collect()
}

/// One Metal dense walk. The budget arm stays DISARMED (the incumbent dense
/// path; the env latch is disarmed by default in a fresh process).
fn metal_walk(
    config: &Config,
    weights: &riir_infer_core::deltanet::QwenDeltaNetTernaryWeights,
    tokens: &[usize],
    stage_mask: u32,
) -> Vec<Vec<f32>> {
    set_decode_stage_mask(stage_mask);
    let ctx = riir_infer_gpu::cubecl_runtime::CubeCLContext::new().expect("Metal ctx");
    assert!(
        ctx.is_metal_adapter(),
        "the cell REQUIRES Metal (compute law: GPU runs on GPU)"
    );
    let mut gpu_config = config.clone();
    gpu_config.block_size = GPU_CTX_CLAMP;
    let mut fwd = TernaryDeltanetGpuForward::new(&ctx, &gpu_config, weights);
    let mut logits = Vec::with_capacity(tokens.len());
    for &tok in tokens {
        fwd.set_input_token(weights, tok);
        logits.push(fwd.forward_token());
    }
    set_decode_stage_mask(DECODE_STAGE_ALL);
    logits
}

fn first_nan_step(logits: &[Vec<f32>]) -> Option<usize> {
    logits.iter().position(|row| row.iter().any(|v| v.is_nan()))
}

#[test]
fn i1196_stage_attribution_hd128_metal_dense_nan() {
    let (config, weights) = load_synth();
    let tokens = token_stream(STEPS, config.vocab_size);

    // ── Repro leg: the finding's SHAPE on the unmasked pipeline — a
    //    finite prefix, then NaNs from a small step onward to the end. The
    //    exact onset is seed-dependent (the T2 session's probe saw 16; the
    //    recorded shape is "clean prefix, then NaN forever"), so the gate
    //    pins the shape, not the literal step.
    let clean_run = metal_walk(&config, &weights, &tokens, DECODE_STAGE_ALL);
    let first_nan = first_nan_step(&clean_run);
    println!("[i1196] repro: first NaN step = {first_nan:?} (the T2 session's seed saw 16)");
    let onset = first_nan
        .expect("repro: the unmasked synth walk must produce NaN logits (the 1196 finding)");
    assert!(
        onset >= 2 && onset < STEPS,
        "repro: the NaN onset must sit inside the walk (got {onset})"
    );
    assert!(
        clean_run[onset..]
            .iter()
            .all(|row| row.iter().any(|v| v.is_nan())),
        "repro: once NaN, every later step stays NaN (the finding's shape)"
    );
    assert!(
        clean_run[..onset]
            .iter()
            .all(|row| row.iter().all(|v| v.is_finite())),
        "repro: steps 0..{onset} must be finite"
    );

    // ── Attribution legs: mask one coarse region off per walk; the region
    //    whose removal kills the NaN is the producer. ──
    let mut culprits = Vec::new();
    for (name, bit, mask_bit) in REGIONS {
        let mask = DECODE_STAGE_ALL & !mask_bit;
        let run = metal_walk(&config, &weights, &tokens, mask);
        let nan = first_nan_step(&run);
        println!("[i1196] mask off {name:<12} (bit {bit}): first NaN step = {nan:?}");
        if nan.is_none() {
            culprits.push(name);
        }
    }

    println!(
        "[i1196] VERDICT: regions whose removal kills the NaN = {culprits:?} \
         (empty = the NaN survives every coarse mask — refine with sub-bits)"
    );
    assert!(
        !culprits.is_empty(),
        "attribution: some coarse region must produce the NaN (masking it off \
         leaves finite garbage); an empty culprit set means the NaN comes from \
         a stage combination, not one region"
    );

    // ── DeltaNet refinement: the coarse producer was the deltanet region
    //    (tail_lm_head is the disclosure point, not the producer — with the
    //    head skipped the walk emits the finite residual stream). Split the
    //    region: parent bit 1 stays ON, each sub-bit masked off in turn.
    //    NOTE: masking a sub-bit changes the layer's MATH downstream too
    //    (e.g. skipping the recurrence feeds zeros forward), so a sub-bit
    //    removal can also kill the NaN indirectly. The leg prints every
    //    row; the root-cause read weights a sub whose removal kills the NaN
    //    AND whose neighborhood stays NaN — T2 names the kernel from the
    //    hd-constant hunt, this pass only bounds it.
    if culprits.contains(&"deltanet") {
        println!("[i1196] deltanet sub-bit split (parent ON, sub OFF):");
        for (name, bit, mask_bit) in GDN_SUBS {
            let mask = DECODE_STAGE_ALL & !mask_bit;
            let run = metal_walk(&config, &weights, &tokens, mask);
            let nan = first_nan_step(&run);
            println!("[i1196]   gdn sub off {name:<16} (bit {bit}): first NaN = {nan:?}");
        }
    }
}
