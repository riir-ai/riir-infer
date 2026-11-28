//! i1221 T3(c) — the labeled micro-task probe suite (plan 636's
//! tokenizer-decision amendment, 2026-11-28). The "tokenizer lane" the
//! T3(c) park named has been in-stack since `8ecbc7f4` (2026-09-24) —
//! [`riir_infer_core::tokenizer::BpeTokenizer::from_gguf`] — and the
//! Dory pin carries its own embedded gpt2 BPE, so the park's blocker is
//! discharged and the suite is the remaining authoring half.
//!
//! 16 labeled micro-tasks (4 each: extract / count / transform / yes-no),
//! every task carrying a deterministic verifier over the DECODED
//! generation text. Labeled, authored, never benchmark-derived (the
//! grammar-not-answers law: no external test rows anywhere in this
//! suite). The suite is FROZEN data: [`suite_digest`] binds its
//! canonical bytes into every cell's resume lineage, so a suite edit
//! starts a fresh lineage instead of silently mixing rows.
//!
//! Verifier rules (pre-registered, fixture-tested):
//! - extract / count / yes-no: word-boundary match on the NORMALIZED
//!   (lowercased) decoded text — `"3"` matches a standalone `3`, never
//!   the `3` inside `13`; `"no"` matches the word `no`, never `know`.
//! - count tasks carry a single-word alt (digit OR its word form).
//! - transform: per-task matching rule; the caps task matches the RAW
//!   text case-sensitively (normalization would erase the thing being
//!   verified), the others normalize.

use blake3::Hasher;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::io::Read;
use std::path::Path;

pub const ROW_RECORD: &str = "i1221_t3c_tokens_per_task_row";
pub const HEADER_RECORD: &str = "i1221_t3c_tokens_per_task_header";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskKind {
    Extract,
    Count,
    Transform,
    YesNo,
}

#[derive(Clone, Copy, Debug)]
pub struct ProbeTask {
    pub id: &'static str,
    pub kind: TaskKind,
    /// The `[INST]` BODY — the cell wraps it via [`wrap_inst`].
    pub prompt: &'static str,
    /// The verifier's target (matched per the kind's rule).
    pub answer: &'static str,
    /// A single-word accepted alternative (count tasks: the word form of
    /// the digit). `None` = the answer alone.
    pub alt_answer: Option<&'static str>,
}

/// The suite, in FIXED order (the row's `prompt_idx` keys this slice;
/// reordering is a lineage break via the digest, never a silent remap).
pub const SUITE: [ProbeTask; 16] = [
    // ── extract ──
    ProbeTask {
        id: "ex_list_third",
        kind: TaskKind::Extract,
        prompt: "In the list 'apple, banana, cherry, date', which fruit comes third? Answer with the fruit name only.",
        answer: "cherry",
        alt_answer: None,
    },
    ProbeTask {
        id: "ex_sentence_name",
        kind: TaskKind::Extract,
        prompt: "The sentence is 'Mara adopted a gray cat named Pepper in 2021.' What is the cat's name? Answer with the name only.",
        answer: "pepper",
        alt_answer: None,
    },
    ProbeTask {
        id: "ex_smallest",
        kind: TaskKind::Extract,
        prompt: "From the numbers 17, 42, 3, 99, pick the smallest one. Answer with that number only.",
        answer: "3",
        alt_answer: Some("three"),
    },
    ProbeTask {
        id: "ex_capital_france",
        kind: TaskKind::Extract,
        prompt: "What city is the capital of France? Answer with the city name only.",
        answer: "paris",
        alt_answer: None,
    },
    // ── count ──
    ProbeTask {
        id: "ct_letters_seven",
        kind: TaskKind::Count,
        prompt: "Count the letters in the word 'seven'. Reply with just the number.",
        answer: "5",
        alt_answer: Some("five"),
    },
    ProbeTask {
        id: "ct_r_strawberry",
        kind: TaskKind::Count,
        prompt: "How many times does the letter 'r' appear in the word 'strawberry'? Reply with just the number.",
        answer: "3",
        alt_answer: Some("three"),
    },
    ProbeTask {
        id: "ct_dog_list",
        kind: TaskKind::Count,
        prompt: "The list is 'dog, cat, dog, bird, dog'. How many times does 'dog' appear in it? Reply with just the number.",
        answer: "3",
        alt_answer: Some("three"),
    },
    ProbeTask {
        id: "ct_add_23_19",
        kind: TaskKind::Count,
        prompt: "Add 23 and 19. Reply with the sum only.",
        answer: "42",
        alt_answer: None,
    },
    // ── transform ──
    ProbeTask {
        id: "tf_reverse_listen",
        kind: TaskKind::Transform,
        prompt: "Reverse the word 'listen'. Reply with the reversed word only.",
        answer: "netsil",
        alt_answer: None,
    },
    ProbeTask {
        id: "tf_caps_tree",
        kind: TaskKind::Transform,
        prompt: "Write the word 'tree' in all capital letters. Reply with the capitalized word only.",
        answer: "TREE",
        alt_answer: None,
    },
    ProbeTask {
        id: "tf_drop_ly",
        kind: TaskKind::Transform,
        prompt: "What is the word 'rapidly' without its last two letters? Reply with the remaining word only.",
        answer: "rapid",
        alt_answer: None,
    },
    ProbeTask {
        id: "tf_sort_first",
        kind: TaskKind::Transform,
        prompt: "Sort these words alphabetically: 'pear, apple, plum'. Reply with the first word of the sorted list only.",
        answer: "apple",
        alt_answer: None,
    },
    // ── yes-no ──
    ProbeTask {
        id: "yn_prime_7",
        kind: TaskKind::YesNo,
        prompt: "Is 7 a prime number? Reply with yes or no only.",
        answer: "yes",
        alt_answer: None,
    },
    ProbeTask {
        id: "yn_boil_50",
        kind: TaskKind::YesNo,
        prompt: "Does water boil at 50 degrees Celsius at sea level? Reply with yes or no only.",
        answer: "no",
        alt_answer: None,
    },
    ProbeTask {
        id: "yn_triangle_2right",
        kind: TaskKind::YesNo,
        prompt: "Can a triangle have two right angles? Reply with yes or no only.",
        answer: "no",
        alt_answer: None,
    },
    ProbeTask {
        id: "yn_run_verb",
        kind: TaskKind::YesNo,
        prompt: "Is the word 'run' a verb? Reply with yes or no only.",
        answer: "yes",
        alt_answer: None,
    },
];

/// The `[INST]` wrapper (the pin's trained instruct format — the vocab
/// carries `[INST]`/`[/INST]` as special tokens, matched by the
/// tokenizer's special-token pass).
pub fn wrap_inst(body: &str) -> String {
    format!("[INST] {body} [/INST]")
}

/// Lowercase + trim (the normalization every word-boundary rule runs on).
pub fn normalize(text: &str) -> String {
    text.trim().to_lowercase()
}

/// Word-boundary match: `needle` equals one whole alphanumeric run of
/// `haystack` (already normalized by the caller when the rule wants
/// case-insensitivity). `"3"` never matches inside `"13"`; `"no"` never
/// matches inside `"know"`.
pub fn contains_word(haystack: &str, needle: &str) -> bool {
    haystack
        .split(|c: char| !c.is_alphanumeric())
        .any(|run| run == needle)
}

/// The per-task verifier over the RAW decoded generation text. Total
/// function: no task can panic on arbitrary model output.
pub fn verify(task: &ProbeTask, decoded: &str) -> bool {
    match task.kind {
        // The caps task IS a case check — match the raw text.
        TaskKind::Transform if task.id == "tf_caps_tree" => contains_word(decoded, task.answer),
        TaskKind::YesNo => {
            let hay = normalize(decoded);
            contains_word(&hay, task.answer) && !contains_word(&hay, opposite_of(task.answer))
        }
        _ => {
            let hay = normalize(decoded);
            contains_word(&hay, task.answer)
                || task.alt_answer.is_some_and(|alt| contains_word(&hay, alt))
        }
    }
}

fn opposite_of(answer: &str) -> &'static str {
    match answer {
        "yes" => "no",
        "no" => "yes",
        _ => "",
    }
}

/// Per-step stop-rule read: has the generation SO FAR satisfied the
/// task? The cell calls this after each pushed token; the gates replay
/// scripted streams through the same function (no model needed).
pub fn step_satisfied(task: &ProbeTask, decoded_so_far: &str) -> bool {
    verify(task, decoded_so_far)
}

/// One (task, k) outcome row — the resumable JSONL payload. Both tag
/// fields are owned `String`s so the row is fully serde-general (the
/// round trip is a REAL deserialization — the t3a family's `&'static
/// str` record field cannot target `Deserialize` and their gates fell
/// back to Value projection; this row does not).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TokensPerTaskRow {
    pub record: String,
    pub prompt_idx: usize,
    pub k: usize,
    pub task_id: String,
    /// "passed" | "missed_eos" | "cap_hit" (the closed vocabulary,
    /// written from the const set, validated on read by the consumer)
    pub outcome: String,
    /// Generation length at the verifier's first pass (Passed rows
    /// only; 0 otherwise — cap/missed rows carry `generated`).
    pub tokens_to_pass: usize,
    /// Total generated tokens this arm produced.
    pub generated: usize,
    pub passed: bool,
    /// The decoded generation (the inspect-ability receipt; a miss is
    /// never opaque).
    pub answer_text: String,
    /// blake3 over the generated token ids, little-endian u32s in order
    /// (the determinism receipt — a resumed/re-run arm must match).
    pub stream_blake3: String,
}

#[allow(clippy::too_many_arguments)] // the row's full field set is the contract
pub fn make_row(
    prompt_idx: usize,
    k: usize,
    task: &ProbeTask,
    outcome: &'static str,
    tokens_to_pass: usize,
    generated: usize,
    answer_text: String,
    stream: &[u32],
) -> TokensPerTaskRow {
    let mut h = Hasher::new();
    for t in stream {
        h.update(&t.to_le_bytes());
    }
    TokensPerTaskRow {
        record: ROW_RECORD.to_string(),
        prompt_idx,
        k,
        task_id: task.id.to_string(),
        outcome: outcome.to_string(),
        tokens_to_pass,
        generated,
        passed: outcome == "passed",
        answer_text,
        stream_blake3: h.finalize().to_hex().to_string(),
    }
}

/// Per-k summary — the direction-check table. Cap-hit and missed rows
/// NEVER enter the mean (the pre-registered accounting: ties/refusals
/// count as their cap and are reported separately, never folded).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KSummary {
    pub k: usize,
    pub n: usize,
    pub passed: usize,
    pub missed_eos: usize,
    pub cap_hit: usize,
    pub mean_tokens_to_pass: Option<f64>,
    pub min_tokens_to_pass: Option<usize>,
    pub max_tokens_to_pass: Option<usize>,
}

pub fn summarize_k(k: usize, rows: &[TokensPerTaskRow]) -> KSummary {
    let mine: Vec<&TokensPerTaskRow> = rows.iter().filter(|r| r.k == k).collect();
    let mut passed_at: Vec<usize> = mine
        .iter()
        .filter(|r| r.outcome == "passed")
        .map(|r| r.tokens_to_pass)
        .collect();
    passed_at.sort_unstable();
    let n = passed_at.len();
    KSummary {
        k,
        n: mine.len(),
        passed: n,
        missed_eos: mine.iter().filter(|r| r.outcome == "missed_eos").count(),
        cap_hit: mine.iter().filter(|r| r.outcome == "cap_hit").count(),
        mean_tokens_to_pass: if n == 0 {
            None
        } else {
            Some(passed_at.iter().sum::<usize>() as f64 / n as f64)
        },
        min_tokens_to_pass: passed_at.first().copied(),
        max_tokens_to_pass: passed_at.last().copied(),
    }
}

/// blake3 over the suite's canonical bytes (prompts + answers + kinds in
/// fixed order) — folded into every header so a suite edit breaks
/// lineage LOUD instead of mixing rows across suites.
pub fn suite_digest() -> String {
    let mut h = Hasher::new();
    for t in SUITE.iter() {
        h.update(t.id.as_bytes());
        h.update(b"\x00");
        h.update(kind_tag(t.kind).as_bytes());
        h.update(t.prompt.as_bytes());
        h.update(b"\x00");
        h.update(t.answer.as_bytes());
        h.update(b"\x00");
        if let Some(alt) = t.alt_answer {
            h.update(alt.as_bytes());
        }
        h.update(b"\x01");
    }
    h.finalize().to_hex().to_string()
}

fn kind_tag(kind: TaskKind) -> &'static str {
    match kind {
        TaskKind::Extract => "extract",
        TaskKind::Count => "count",
        TaskKind::Transform => "transform",
        TaskKind::YesNo => "yesno",
    }
}

/// Resume lineage (plan 636's law, this cell's field set): the header
/// must match on pin blake3, repo STAMP, ks, max_gen, and the suite
/// digest — a mismatch is a typed `Err` (the CALLER exits 2 — the
/// gates exercise every refusal arm in-process; never append across
/// lineages). Returns the `(prompt_idx, k)` pairs already done. A
/// missing/empty file resumes nothing. Pid-scoped tmp names are the
/// CALLER's duty (the shared-temp-path law).
pub fn verify_resume_lineage_c(
    rows_path: &Path,
    pin_blake3: &str,
    repo_commit: &str,
    ks: &[usize],
    max_gen: usize,
) -> Result<BTreeSet<(usize, usize)>, String> {
    let refuse = |msg: &str| -> String {
        format!(
            "header lineage mismatch: {msg} — never append across lineages; \
             rerun with --fresh or a new --out"
        )
    };
    let content = match std::fs::read_to_string(rows_path) {
        Ok(c) => c,
        Err(_) => return Ok(Default::default()),
    };
    let mut lines = content.lines();
    let header = lines.next().unwrap_or("");
    let v: serde_json::Value =
        serde_json::from_str(header).map_err(|_| refuse("header line is not JSON"))?;
    let model = v
        .get("model")
        .ok_or_else(|| refuse("header has no model"))?;
    if model.get("blake3").and_then(|b| b.as_str()) != Some(pin_blake3) {
        return Err(refuse("pin blake3 differs (moved/replaced checkpoint)"));
    }
    if v.get("repo_commit").and_then(|s| s.as_str()) != Some(repo_commit) {
        return Err(refuse("repo STAMP differs (different code lineage)"));
    }
    let probe = v
        .get("probe")
        .ok_or_else(|| refuse("header has no probe spec"))?;
    let got_ks: Vec<usize> = probe
        .get("ks")
        .and_then(|x| x.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|u| u.as_u64())
                .map(|u| u as usize)
                .collect()
        })
        .unwrap_or_default();
    if got_ks != ks {
        return Err(refuse("k set differs (different arms)"));
    }
    if probe.get("max_gen").and_then(|x| x.as_u64()) != Some(max_gen as u64) {
        return Err(refuse("max_gen differs (different stopping rule)"));
    }
    if probe.get("suite_blake3").and_then(|x| x.as_str()) != Some(suite_digest().as_str()) {
        return Err(refuse("suite digest differs (the probe suite was edited)"));
    }
    let mut done = BTreeSet::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let r: serde_json::Value =
            serde_json::from_str(line).map_err(|_| refuse("row line is not JSON (torn write?)"))?;
        if r.get("record").and_then(|x| x.as_str()) != Some(ROW_RECORD) {
            continue;
        }
        let p = r.get("prompt_idx").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
        let k = r.get("k").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
        done.insert((p, k));
    }
    Ok(done)
}

/// blake3 of a whole file (the pin digest; the t3a posture, copied —
/// std-only read loop, 1 MiB chunks).
pub fn blake3_file(path: &Path) -> String {
    let mut file = std::fs::File::open(path).unwrap_or_else(|e| panic!("open {path:?}: {e}"));
    let mut hasher = Hasher::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf).expect("read model file");
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    hasher.finalize().to_hex().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suite_shape_is_fixed() {
        assert_eq!(SUITE.len(), 16, "4 kinds x 4 tasks");
        for kind in [
            TaskKind::Extract,
            TaskKind::Count,
            TaskKind::Transform,
            TaskKind::YesNo,
        ] {
            assert_eq!(
                SUITE.iter().filter(|t| t.kind == kind).count(),
                4,
                "{kind:?} needs exactly 4"
            );
        }
        let mut ids: Vec<&str> = SUITE.iter().map(|t| t.id).collect();
        ids.sort_unstable();
        let n = ids.len();
        ids.dedup();
        assert_eq!(ids.len(), n, "task ids unique");
        for t in SUITE.iter() {
            assert!(!t.prompt.is_empty());
            assert!(!t.answer.is_empty());
        }
    }

    #[test]
    fn verifiers_pass_and_fail_on_canned_text() {
        let find = |id: &str| SUITE.iter().find(|t| t.id == id).unwrap();
        // extract
        assert!(verify(find("ex_list_third"), "The fruit is cherry."));
        assert!(!verify(find("ex_list_third"), "The fruit is banana."));
        // the containment trap: 13 contains 3, the word-boundary rule must not
        assert!(!verify(find("ex_smallest"), "The smallest is 13."));
        assert!(verify(find("ex_smallest"), "The smallest is 3."));
        assert!(verify(find("ex_smallest"), "It is three."));
        // count
        assert!(verify(find("ct_letters_seven"), "The word has 5 letters."));
        assert!(verify(find("ct_letters_seven"), "five"));
        assert!(!verify(find("ct_letters_seven"), "The word has 4 letters."));
        // transform — the caps task is case-sensitive on the RAW text
        assert!(verify(find("tf_caps_tree"), "The word is TREE."));
        assert!(!verify(find("tf_caps_tree"), "The word is tree."));
        assert!(verify(find("tf_reverse_listen"), "netsil"));
        assert!(!verify(find("tf_reverse_listen"), "silten"));
        // yes-no — the word-boundary law: know != no
        assert!(verify(find("yn_prime_7"), "Yes, 7 is prime."));
        assert!(!verify(find("yn_prime_7"), "I know 7 is odd."));
        assert!(verify(find("yn_boil_50"), "No, it boils at 100."));
        assert!(!verify(find("yn_boil_50"), "Yes."));
        // no panic on arbitrary output
        assert!(!verify(find("ex_capital_france"), "\u{0}\u{1} ???"));
    }

    #[test]
    fn summarize_k_never_folds_refusals_into_the_mean() {
        let task = &SUITE[0];
        let rows = vec![
            make_row(0, 1, task, "passed", 4, 4, "cherry".into(), &[1, 2, 3, 4]),
            make_row(
                1,
                1,
                task,
                "passed",
                7,
                7,
                "the answer is cherry".into(),
                &[5, 6, 7],
            ),
            make_row(
                2,
                1,
                task,
                "cap_hit",
                0,
                64,
                "banana banana".into(),
                &[8; 64],
            ),
            make_row(
                3,
                1,
                task,
                "missed_eos",
                0,
                9,
                "banana".into(),
                &[9, 1, 2, 3, 4, 5, 6, 7, 8],
            ),
        ];
        let s = summarize_k(1, &rows);
        assert_eq!(s.n, 4);
        assert_eq!(s.passed, 2);
        assert_eq!(s.cap_hit, 1);
        assert_eq!(s.missed_eos, 1);
        assert_eq!(s.mean_tokens_to_pass, Some(5.5)); // (4 + 7) / 2, cap row excluded
        assert_eq!(s.min_tokens_to_pass, Some(4));
        assert_eq!(s.max_tokens_to_pass, Some(7));
        assert_eq!(summarize_k(2, &rows).n, 0);
        assert_eq!(summarize_k(2, &rows).mean_tokens_to_pass, None);
    }

    #[test]
    fn row_round_trips_through_serde() {
        let task = &SUITE[3];
        let row = make_row(3, 4, task, "passed", 6, 6, "Paris".into(), &[7, 8, 9]);
        let line = serde_json::to_string(&row).expect("row json");
        let back: TokensPerTaskRow = serde_json::from_str(&line).expect("row parse");
        assert_eq!(back.record, ROW_RECORD);
        assert_eq!(back.k, 4);
        assert!(back.passed);
        assert_eq!(back.stream_blake3, row.stream_blake3);
    }

    #[test]
    fn lineage_accepts_own_header_and_refuses_drift() {
        let dir = std::env::temp_dir().join(format!("i1221_t3c_gate_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("tmp dir");
        let rows_path = dir.join("rows.jsonl");
        let header = format!(
            "{{\"record\":\"{HEADER_RECORD}\",\"model\":{{\"path\":\"p\",\"blake3\":\"PIN\",\"arch\":\"dory\",\"k_full\":4}},\"repo_commit\":\"STAMP\",\"probe\":{{\"ks\":[1,2,4],\"max_gen\":64,\"suite_blake3\":\"{}\"}}}}\n",
            suite_digest()
        );
        let task = &SUITE[0];
        let row = serde_json::to_string(&make_row(
            0,
            1,
            task,
            "passed",
            3,
            3,
            "cherry".into(),
            &[1, 2, 3],
        ))
        .expect("row");
        std::fs::write(&rows_path, format!("{header}{row}\n")).expect("write rows");

        let done = verify_resume_lineage_c(&rows_path, "PIN", "STAMP", &[1, 2, 4], 64)
            .expect("own header accepts");
        assert_eq!(done, BTreeSet::from([(0, 1)]));

        // every drift arm refuses with a NAMED reason (typed Err — the
        // caller exits 2; the message names the drifted field)
        for (pin, stamp, ks, gen_cap, msg) in [
            ("OTHER", "STAMP", &[1usize, 2, 4][..], 64usize, "pin blake3"),
            ("PIN", "OTHER", &[1, 2, 4], 64, "repo STAMP"),
            ("PIN", "STAMP", &[1, 2], 64, "k set"),
            ("PIN", "STAMP", &[1, 2, 4], 32, "max_gen"),
        ] {
            let err = verify_resume_lineage_c(&rows_path, pin, stamp, ks, gen_cap)
                .expect_err(&format!("{msg} drift must refuse"));
            assert!(err.contains(msg), "refusal names the field: {err}");
        }

        // a suite edit breaks lineage
        let header2 = header.replace(&suite_digest(), "deadbeef");
        std::fs::write(&rows_path, header2).expect("rewrite");
        let err = verify_resume_lineage_c(&rows_path, "PIN", "STAMP", &[1, 2, 4], 64)
            .expect_err("suite digest drift must refuse");
        assert!(err.contains("suite digest"), "{err}");

        // a torn row line refuses
        std::fs::write(&rows_path, format!("{header}{{torn\n")).expect("rewrite torn");
        let err = verify_resume_lineage_c(&rows_path, "PIN", "STAMP", &[1, 2, 4], 64)
            .expect_err("torn row must refuse");
        assert!(err.contains("torn write"), "{err}");

        let _ = std::fs::remove_file(&rows_path);
        let _ = std::fs::remove_dir(&dir);
    }
}
