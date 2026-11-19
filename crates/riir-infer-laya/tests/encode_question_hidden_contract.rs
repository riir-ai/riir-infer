//! Issue 1239 — the `EncodedQuestion` contract pin.
//!
//! `RiirAgent::encode_question` documents `hidden` as the ENCODER output
//! (pre-type-emb). The head writes its residual stream in place, so a
//! download after the head returned the post-head state (measured max |Δ|
//! 1281.6 on english). This pins both halves, CPU posture (bit-exact):
//!
//! - `hidden` == the encoder-only seam (`encoder_hidden`), bit for bit;
//! - `marker_rows` == the head re-run from that `hidden`
//!   (`head_rows_from_hidden`), bit for bit — `hidden` really is the
//!   head's INPUT.
//!
//! Skip-loud without weights; `LAYA_REQUIRE_DATA=1` turns the skip into a
//! failure.
#![cfg(feature = "laya-riir")]

use riir_infer_laya::laya::config::Checkpoint;
use riir_infer_laya::laya::riir::agent::{DeviceKind, RiirAgent};
use riir_infer_laya::laya::weights::weights_root;

#[test]
fn encode_question_hidden_is_the_encoder_output() {
    let root = weights_root();
    let file = root.join("english").join("model.safetensors");
    if !file.is_file() {
        assert!(
            std::env::var("LAYA_REQUIRE_DATA").as_deref() != Ok("1"),
            "LAYA_REQUIRE_DATA=1 but no english checkpoint at {}",
            file.display()
        );
        eprintln!("SKIP (loud): no english checkpoint at {}", file.display());
        return;
    }
    let agent = RiirAgent::load_with_device(&root, Checkpoint::English, DeviceKind::Cpu)
        .expect("english loads on the CPU posture");
    let state = serde_json::json!("The stack stands low. A long straight piece is falling.");
    let cases = [
        serde_json::json!({"type": "noul", "instructions": "Does the stack look clean?", "criteria": null}),
        serde_json::json!({
            "type": "choice",
            "instructions": "Pick the best placement.",
            "criteria": ["left_edge", "middle", "right_edge"]
        }),
    ];
    for qdef in &cases {
        let enc = agent
            .encode_question(&state, qdef)
            .expect("encode_question");
        let (hidden, markers, qtype) = agent.encoder_hidden(&state, qdef).expect("encoder_hidden");
        assert_eq!(markers, enc.markers, "same rendered sequence");
        assert_eq!(qtype, enc.qtype);
        assert_eq!(enc.hidden.len(), hidden.len());
        let worst = enc
            .hidden
            .iter()
            .zip(&hidden)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(
            worst, 0.0,
            "EncodedQuestion.hidden must be the encoder output (max |Δ| {worst})"
        );
        let rows = agent
            .head_rows_from_hidden(&enc.hidden, enc.qtype, &enc.markers)
            .expect("head_rows_from_hidden");
        assert_eq!(
            rows, enc.marker_rows,
            "marker_rows must be the head over hidden"
        );
    }
}

/// The Metal posture: the fix adds a mid-pass `download_into` of the
/// encoder output before the head enqueues — pin that the device stream
/// still yields the encoder output and the head over it, bit for bit
/// against the same posture's seams.
#[cfg(all(target_os = "macos", feature = "laya-riir-metal"))]
#[test]
fn encode_question_hidden_is_the_encoder_output_metal() {
    let root = weights_root();
    if !root.join("english").join("model.safetensors").is_file() {
        eprintln!("SKIP (loud): no english checkpoint");
        return;
    }
    let agent = RiirAgent::load_with_device(&root, Checkpoint::English, DeviceKind::Metal)
        .expect("english loads on the Metal posture");
    let state = serde_json::json!("The stack stands low. A long straight piece is falling.");
    let qdef = serde_json::json!({"type": "noul", "instructions": "Does the stack look clean?", "criteria": null});
    let enc = agent
        .encode_question(&state, &qdef)
        .expect("encode_question");
    let (hidden, _, _) = agent.encoder_hidden(&state, &qdef).expect("encoder_hidden");
    assert_eq!(
        enc.hidden, hidden,
        "Metal: hidden must be the encoder output"
    );
    let rows = agent
        .head_rows_from_hidden(&enc.hidden, enc.qtype, &enc.markers)
        .expect("head_rows_from_hidden");
    assert_eq!(
        rows, enc.marker_rows,
        "Metal: marker_rows must be the head over hidden"
    );
}
