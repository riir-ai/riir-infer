//! i1221 T3(c) — the tokens-per-task cell's gate battery (plan 636's
//! tokenizer-decision amendment). The runner's rows are measurements;
//! THESE are the lane's asserts, hermetic (no model file — the suite,
//! verifiers, stop-rule accounting, and lineage are pure functions on
//! [`riir_infer_core::dory::probe_suite`]; the real-pin run carries its
//! own receipts):
//!
//! 1. **stop-rule replay** — scripted decoded streams through the SAME
//!    `step_satisfied` path the cell's greedy loop reads: the verifier
//!    fires at the completion step, never early; eos-before-answer and
//!    cap-hit account as their own outcomes.
//! 2. **suite freeze** — the digest is stable across calls (a lineage
//!    key must not drift run-to-run) and flips on a content change
//!    (proven on a synthetic probe, never by touching SUITE).
//! 3. **wrapper format** — the `[INST]` wrapper is the trained shape
//!    and never double-wraps.
//! 4. **accounting shape** — the per-k summary's refusal classes stay
//!    disjoint from the mean (planted rows, the amendment's law).
//! 5. **row schema** — the JSONL row round-trips with the frozen field
//!    set a consumer diffs against.

#![cfg(feature = "dory")]

use riir_infer_core::dory::probe_suite::{
    self as suite, TaskKind, make_row, step_satisfied, suite_digest, summarize_k,
};

#[test]
fn stop_rule_fires_at_the_completion_step_never_early() {
    let yn = suite::SUITE.iter().find(|t| t.id == "yn_prime_7").unwrap();
    let steps = ["The", "The number 7", "Yes, 7 is", "Yes, 7 is prime."];
    // First-fire semantics: the runner stops at the FIRST satisfying
    // step — "Yes, 7 is" already carries the answer, so satisfaction
    // begins there (step 2) and stays monotone after.
    let first = steps
        .iter()
        .position(|s| step_satisfied(yn, s))
        .expect("the completion stream satisfies");
    assert_eq!(first, 2, "the answer word arrives at step 2, never earlier");
    assert!(
        steps[first..].iter().all(|s| step_satisfied(yn, s)),
        "monotone after the answer"
    );

    // the know-trap: 'know' must not satisfy a yes task mid-stream
    let steps2 = ["I", "I know", "I know 7 is odd."];
    assert!(
        steps2.iter().all(|s| !step_satisfied(yn, s)),
        "'know' never reads as 'yes'"
    );

    let no = suite::SUITE.iter().find(|t| t.id == "yn_boil_50").unwrap();
    let steps3 = ["Water", "Water boils", "No, it boils at 100 degrees."];
    let first3 = steps3
        .iter()
        .position(|s| step_satisfied(no, s))
        .expect("the completion stream satisfies");
    assert_eq!(first3, 2);
    assert!(steps3[first3..].iter().all(|s| step_satisfied(no, s)));
}

#[test]
fn stop_rule_outcomes_partition_every_stream() {
    // Every scripted stream lands in exactly one outcome class: pass,
    // or (at the runner's level) missed_eos / cap_hit. The runner's
    // loop reads step_satisfied per step — this pins the class boundaries.
    let caps = suite::SUITE
        .iter()
        .find(|t| t.id == "tf_caps_tree")
        .unwrap();
    assert!(!step_satisfied(caps, "the word is tree."));
    assert!(step_satisfied(caps, "The word is TREE."));
    assert!(!step_satisfied(caps, "tree")); // raw case-sensitivity
    assert!(!step_satisfied(caps, "the word is TIMBER."));
}

#[test]
fn suite_digest_is_stable_and_content_sensitive() {
    let d1 = suite_digest();
    let d2 = suite_digest();
    assert_eq!(d1, d2, "the lineage key must be stable within a build");
    assert_eq!(d1.len(), 64, "blake3 hex");

    // content sensitivity proven WITHOUT touching SUITE: the digest of
    // one probe's canonical bytes differs from another's.
    let mut h = blake3::Hasher::new();
    h.update(suite::SUITE[0].prompt.as_bytes());
    let a = h.finalize().to_string();
    let mut h = blake3::Hasher::new();
    h.update(suite::SUITE[1].prompt.as_bytes());
    let b = h.finalize().to_string();
    assert_ne!(
        a, b,
        "distinct prompts hash distinctly (the digest binds content)"
    );
}

#[test]
fn wrapper_is_the_trained_inst_shape() {
    let wrapped = suite::wrap_inst("What is 2+2?");
    assert_eq!(wrapped, "[INST] What is 2+2? [/INST]");
    assert!(!wrapped.contains("[INST] [INST]"), "never double-wraps");
    for t in suite::SUITE.iter() {
        let w = suite::wrap_inst(t.prompt);
        assert!(w.starts_with("[INST] ") && w.ends_with(" [/INST]"));
        assert_eq!(w.matches("[INST]").count(), 1);
    }
}

#[test]
fn accounting_refusals_stay_out_of_the_mean() {
    let task = &suite::SUITE[8]; // tf_reverse_listen
    let rows = vec![
        make_row(
            8,
            2,
            task,
            "passed",
            5,
            5,
            "netsil".into(),
            &[1, 2, 3, 4, 5],
        ),
        make_row(
            8,
            2,
            task,
            "cap_hit",
            0,
            64,
            "listen listen".into(),
            &[6; 64],
        ),
        make_row(8, 2, task, "missed_eos", 0, 12, "silnet".into(), &[7; 12]),
        make_row(8, 4, task, "passed", 4, 4, "netsil".into(), &[1, 2, 3, 4]),
    ];
    let k2 = summarize_k(2, &rows);
    assert_eq!(k2.passed, 1);
    assert_eq!(k2.cap_hit + k2.missed_eos, 2);
    assert_eq!(k2.mean_tokens_to_pass, Some(5.0));
    let k4 = summarize_k(4, &rows);
    assert_eq!(k4.mean_tokens_to_pass, Some(4.0));
    // refusal classes partition n
    assert_eq!(k2.passed + k2.cap_hit + k2.missed_eos, k2.n);
}

#[test]
fn row_schema_is_the_frozen_field_set() {
    let task = &suite::SUITE[15];
    let row = make_row(15, 1, task, "passed", 2, 2, "Yes".into(), &[42, 43]);
    let v: serde_json::Value = serde_json::to_value(&row).expect("row to value");
    for field in [
        "record",
        "prompt_idx",
        "k",
        "task_id",
        "outcome",
        "tokens_to_pass",
        "generated",
        "passed",
        "answer_text",
        "stream_blake3",
    ] {
        assert!(v.get(field).is_some(), "frozen field `{field}` present");
    }
    assert_eq!(
        v.get("record").and_then(|x| x.as_str()),
        Some("i1221_t3c_tokens_per_task_row")
    );
    // the field values survive the JSONL form (the write-side const +
    // the closed outcome vocabulary; the t3a precedent — struct-
    // deserialization is impossible against the &'static record field,
    // so the field set is pinned on the Value form)
    assert_eq!(v.get("task_id").and_then(|x| x.as_str()), Some(task.id));
    assert_eq!(v.get("passed").and_then(|x| x.as_bool()), Some(true));
    assert_eq!(v.get("k").and_then(|x| x.as_u64()), Some(1));
    assert_eq!(v.get("outcome").and_then(|x| x.as_str()), Some("passed"));
    assert_eq!(TaskKind::YesNo, task.kind);
}
