//! Issue 1253 T2 — the MTP-twin load-census gates (synthetic; the real-file
//! census lives in `examples/mtp_twin_census.rs`).
//!
//! Pins the two loader classes the census introduces:
//! 1. DECLARED trailing nextn block — the load succeeds and the disclosure
//!    fires (the draft head stays for the MTP lane; the main stack is
//!    complete — never a silent partial load).
//! 2. UNDECLARED trailing nextn block — the load REFUSES loud naming the
//!    trailing block class (the Issue-742 garbage class: without the
//!    declaration the block would be typed and consumed as a main layer).
//! 3. A clean file (no trailing-block tensors) loads with an empty census —
//!    the no-false-positive arm.

#[path = "common/synth_bonsai2.rs"]
mod synth;

use synth::*;

use riir_infer_core::gguf_loader::{load_qwen_deltanet_ternary_from_gguf, GgufFile};

use std::io::Write as _;

/// The synthetic main model (4 layers, folded contract metadata so the load
/// takes the same path as the league files).
fn main_model_bytes() -> Vec<u8> {
    build_gguf(&synth_metadata(true), &synth_tensors_bonsai2())
}

/// One draft-head block appended at `blk.{N_LAYER}`: the main-layer tensor
/// set the loader would read as a stack layer + the `nextn.*` component —
/// the undeclared arm must refuse BEFORE any of these are read. Types ride
/// the same F32/Q4_0-class mix the real 2.5 twin carries (F32 here; the
/// census reads block INDICES, never the sub-names).
fn mtp_block_tensors() -> Vec<TensorSpec> {
    let i = N_LAYER; // the trailing block index == main layer count
    let mut out = vec![
        TensorSpec {
            name: format!("blk.{i}.nextn.eh_proj.weight"),
            ne: vec![N_EMBD * 2, N_EMBD],
            ggml_type: 0, // F32
            data: f32_payload(&vec![0.01f32; (N_EMBD * 2 * N_EMBD) as usize]),
        },
        TensorSpec {
            name: format!("blk.{i}.nextn.norm.weight"),
            ne: vec![N_EMBD],
            ggml_type: 0,
            data: f32_payload(&vec![1.0f32; N_EMBD as usize]),
        },
    ];
    // The main-layer tensors the undeclared arm would read as a bogus stack
    // layer — a COMPLETE legal ternary DeltaNet block (block 4 is DeltaNet
    // per the synthetic interval: (4+1) % 4 != 0) so the refusal cannot be
    // confused with a missing-tensor or type/decode error: the undeclared
    // class must fire on the census alone, before any per-tensor read.
    let key_dim = N_K * STATE;
    let value_dim = N_V * STATE;
    let ternary = |name: String, ne0: u64, ne1: u64| TensorSpec {
        ggml_type: 142, // the synthetic main stack's ternary id
        data: q2_0_payload(ne0 * ne1, 9),
        ne: vec![ne0, ne1],
        name,
    };
    out.push(ternary(
        format!("blk.{i}.attn_qkv.weight"),
        N_EMBD,
        key_dim * 2 + value_dim,
    ));
    out.push(ternary(
        format!("blk.{i}.attn_gate.weight"),
        N_EMBD,
        value_dim,
    ));
    out.push(ternary(
        format!("blk.{i}.ssm_out.weight"),
        value_dim,
        N_EMBD,
    ));
    out.push(ternary(format!("blk.{i}.ffn_gate.weight"), N_EMBD, FFN));
    out.push(ternary(format!("blk.{i}.ffn_up.weight"), N_EMBD, FFN));
    out.push(ternary(format!("blk.{i}.ffn_down.weight"), FFN, N_EMBD));
    out.push(TensorSpec {
        name: format!("blk.{i}.ssm_conv1d.weight"),
        ne: vec![CONV_K, key_dim * 2 + value_dim],
        ggml_type: 0,
        data: f32_payload(&vec![0.1f32; (CONV_K * (key_dim * 2 + value_dim)) as usize]),
    });
    out.push(TensorSpec {
        name: format!("blk.{i}.ssm_dt.bias"),
        ne: vec![N_V],
        ggml_type: 0,
        data: f32_payload(&vec![0.01f32; N_V as usize]),
    });
    out.push(TensorSpec {
        name: format!("blk.{i}.ssm_a"),
        ne: vec![N_V],
        ggml_type: 0,
        data: f32_payload(&vec![-2.718_281_7f32; N_V as usize]),
    });
    out.push(TensorSpec {
        name: format!("blk.{i}.ssm_norm.weight"),
        ne: vec![STATE],
        ggml_type: 0,
        data: f32_payload(&vec![1.0f32; STATE as usize]),
    });
    // a/b: the BF16 dense arm (the synthetic kit's escape set — block 4
    // loads them through load_gate_proj's Dense path).
    let a_vals: Vec<f32> = (0..(N_EMBD * N_V))
        .map(|j| ((j % 11) as f32 - 5.0) * 0.1)
        .collect();
    let b_vals: Vec<f32> = (0..(N_EMBD * N_V))
        .map(|j| ((j % 7) as f32 - 3.0) * 0.1)
        .collect();
    out.push(TensorSpec {
        name: format!("blk.{i}.ssm_alpha.weight"),
        ne: vec![N_EMBD, N_V],
        ggml_type: 30, // BF16
        data: bf16_payload(&a_vals),
    });
    out.push(TensorSpec {
        name: format!("blk.{i}.ssm_beta.weight"),
        ne: vec![N_EMBD, N_V],
        ggml_type: 30,
        data: bf16_payload(&b_vals),
    });
    for name in [
        format!("blk.{i}.attn_norm.weight"),
        format!("blk.{i}.post_attention_norm.weight"),
    ] {
        out.push(TensorSpec {
            name,
            ne: vec![N_EMBD],
            ggml_type: 0,
            data: f32_payload(&vec![1.0f32; N_EMBD as usize]),
        });
    }
    out
}

fn write_tmp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("{name}_{}.gguf", std::process::id()));
    let mut f = std::fs::File::create(&path).expect("create temp gguf");
    f.write_all(bytes).expect("write temp gguf");
    path
}

/// Arm 1 — declared: `block_count` includes the MTP block AND
/// `qwen35.nextn_predict_layers = 1` covers it. The load SUCCEEDS with
/// `n_layer == N_LAYER` (the draft block never enters the main stack) and
/// the file's trailing tensors are exactly the MTP class (the disclosure
/// fires; asserted via the census helper the loader itself uses).
#[test]
fn mtp_declared_loads_with_main_stack_excluded() {
    let mut meta = synth_metadata(true);
    meta.push(("qwen35.block_count".into(), Val::U64(N_LAYER + 1)));
    meta.push(("qwen35.nextn_predict_layers".into(), Val::U64(1)));
    let mut tensors = synth_tensors_bonsai2();
    tensors.extend(mtp_block_tensors());
    let bytes = build_gguf(&meta, &tensors);
    let path = write_tmp("riir_1253_mtp_declared", &bytes);

    let gguf = GgufFile::open(&path).expect("open declared-mtp synthetic");
    let census = riir_infer_core::gguf_loader::mtp_twin_census(&gguf);
    assert_eq!(census.declared_nextn, 1);
    assert_eq!(census.n_main, N_LAYER as usize);
    assert_eq!(census.trailing_tensors, mtp_block_tensors().len());
    assert_eq!(
        census.nextn_names, 2,
        "eh_proj + norm carry the nextn component"
    );
    assert_eq!(census.highest_block, Some(N_LAYER as usize));
    assert!(
        census.covered(),
        "declared nextn=1 covers the trailing block"
    );

    // The full load path: succeeds, main stack intact, n_main excludes the
    // draft block. (The one-line disclosure prints to stderr — the typed
    // surface above is the assertable half.)
    let (config, weights) =
        load_qwen_deltanet_ternary_from_gguf(&gguf, "").expect("declared-mtp load");
    assert_eq!(config.n_layer, N_LAYER as usize);
    assert_eq!(weights.layers.len(), N_LAYER as usize);
    assert!(weights.invariants_hold());
    let _ = std::fs::remove_file(&path);
}

/// Arm 2 — undeclared: the trailing block tensors are present but the file
/// does NOT declare `nextn_predict_layers`. The load REFUSES loud, naming
/// the trailing-block class and the declare-or-refuse law.
#[test]
fn mtp_undeclared_refuses_loud() {
    let mut meta = synth_metadata(true);
    // block_count includes the MTP block, but NO nextn metadata: n_main
    // would become N_LAYER + 1 and the draft block would be typed as a
    // main layer — the exact garbage class the census refuses.
    meta.push(("qwen35.block_count".into(), Val::U64(N_LAYER + 1)));
    let mut tensors = synth_tensors_bonsai2();
    tensors.extend(mtp_block_tensors());
    let bytes = build_gguf(&meta, &tensors);
    let path = write_tmp("riir_1253_mtp_undeclared", &bytes);

    let gguf = GgufFile::open(&path).expect("open undeclared-mtp synthetic");
    let census = riir_infer_core::gguf_loader::mtp_twin_census(&gguf);
    assert!(
        !census.covered(),
        "undeclared draft head must read uncovered"
    );
    let err = match load_qwen_deltanet_ternary_from_gguf(&gguf, "") {
        Ok(_) => panic!("undeclared trailing block must refuse"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("nextn.*"),
        "refusal must name the undeclared draft-head class, got: {msg}"
    );
    assert!(
        msg.contains("nextn_predict_layers"),
        "refusal must name the declaration, got: {msg}"
    );
    let _ = std::fs::remove_file(&path);
}

/// Arm 3 — clean file: no trailing-block tensors, census empty, load
/// unchanged (the no-false-positive arm; the existing suite's posture).
#[test]
fn clean_file_census_is_empty() {
    let bytes = main_model_bytes();
    let path = write_tmp("riir_1253_clean", &bytes);
    let gguf = GgufFile::open(&path).expect("open clean synthetic");
    let census = riir_infer_core::gguf_loader::mtp_twin_census(&gguf);
    assert_eq!(census.trailing_tensors, 0);
    assert_eq!(census.highest_block, None);
    assert_eq!(census.declared_nextn, 0);
    assert_eq!(census.nextn_names, 0);
    assert_eq!(census.n_main, N_LAYER as usize);
    assert!(census.covered());
    let (config, _) = load_qwen_deltanet_ternary_from_gguf(&gguf, "").expect("clean load");
    assert_eq!(config.n_layer, N_LAYER as usize);
    let _ = std::fs::remove_file(&path);
}
