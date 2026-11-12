//! grafts_export — the katgpt-rs issue-988 T4 GOAT bench's frozen-read
//! fixture exporter (riir-infer Issue 1206 lane).
//!
//! CPU-only, header-only: no weights load, no GPU, no forward passes.
//! Loads the T2b sample cache (`--samples`, the grafts_cell
//! `--save-samples` JSON), re-runs the EXACT fit the sealed cell ran
//! (`fit_parts` on the solved split at the pinned λ), and writes the
//! fixture pair the GOAT bench replays:
//!
//!   `<out>/grafts_t2b_artifact.bin` — the conditional graft's artifact
//!        envelope, `GRFZ` container: magic + format_version u32 + digest
//!        32B + epoch u64 + canonical_len u32 + canonical bytes.
//!   `<out>/grafts_t2b_rows.bin` — the frozen decision rows, `GRFW`
//!        container v2 (998 T2): magic + version u32 + n u32, then per
//!        row {instance u32, d0 u16, d1 u16, d2 u16, answer u8,
//!        base_wrong u8, armed u8, class u8, split u8} (class: 0 task /
//!        1 off-task easy / 2 hard-A / 3 hard-B; split: 0 solved, 1 dev,
//!        2 off-task fit, 3 off-task held-out). v1 parsers refuse v2
//!        loud and vice versa.
//!
//! Both files are BLAKE3-hashed into the printed manifest JSON, which
//! also carries the provenance chain (the sealed report's checkpoint +
//! payload digests, the fit pins) and the armed × base-wrong crosstab the
//! bench's G3 selectivity cell reads.
//!
//! ```bash
//! cargo run --release --features grafts_cell --bin grafts_export -- \
//!     --model ../riir-train/data/Qwen3.5-0.8B-Base-Q8_0.gguf \
//!     --samples .benchmarks/grafts_cell_t2b/slots_early.json \
//!     --offtask .benchmarks/grafts_cell_t2b/offtask.json \
//!     --offtask-test .benchmarks/grafts_cell_t2b/offtask_test.json \
//!     --report .benchmarks/grafts_cell_t2b/report.json \
//!     --out .benchmarks/grafts_cell_t2b
//! ```
//!
//! Posture 3 (katgpt-rs issue 998, verdict session `40a812a8`): the
//! threshold rule is the CALIBRATION split's max raw ridge score × 2.0
//! (M frozen in the posture-3 pre-registration; RAW SCORE SPACE — the
//! margin never applies in sigmoid space, asserted below). `--tail-
//! diagnostics` runs the same fit + score path READ-ONLY and prints the
//! pooled untrained-tail evidence the pre-registration quotes (verdict
//! amendments 2+3+4): top scores, per-class maxima, a GPD tail fit, and
//! expected exceedances at candidate margins. No writes, no placement.
//!
//! The bench in katgpt-rs reconstructs activations FROM the artifact's
//! own read/switch weights to hit each frozen row's recorded triple and
//! arming bit (a 2-functional Gram solve per slot, margins absorbing the
//! f32 error) — so the fixture carries the ~60 KB artifact + ~8 KB of
//! decision rows, never the 22 MB of activations. The fit's
//! reproducibility is pinned by the artifact digest: any solver or
//! calibration drift in `fit_parts` changes the digest and the bench
//! refuses loud.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};

use riir_infer_core::gguf_loader::GgufFile;
use riir_infer_core::grafts_cell::{
    PromptFormat, RowReader, TaskKind, digit_tokens, fit_parts, load_offtask, load_samples,
    probe_layers,
};
use riir_infer_core::tokenizer::BpeTokenizer;

use katgpt_core::graft::{Graft, GraftSwitch, GraftWrite, WriteEntry, fit_switch};
use katgpt_core::{GRAFT_ARTIFACT_FORMAT_VERSION, GraftArtifact, exact_sigmoid};

// ── Pinned (identical to the cell's a-priori pins — never re-tuned here) ──

const RIDGE: f32 = 0.1;
const BIAS_DELTA: f32 = 8.0;
/// The reference prompt's operands for the digit map (the cell's own
/// `digit_map` constants — the map must be the SEALED cell's surface).
const REF_OPERANDS: &[u8] = &[73, 34, 51];
const TASK: TaskKind = TaskKind::ThreeOpAdd10;
const FORMAT: PromptFormat = PromptFormat::FewShot;
const ARTIFACT_EPOCH: u64 = 1;

const ARTIFACT_MAGIC: &[u8; 4] = b"GRFZ";
const ROWS_MAGIC: &[u8; 4] = b"GRFW";
/// v2 (998 T2): adds the per-row class byte — 0 task, 1 off-task easy,
/// 2 off-task hard-A (digits non-arithmetic), 3 off-task hard-B
/// (arithmetic words). v1 parsers refuse v2 loud and vice versa.
const ROWS_VERSION: u32 = 2;
/// Row record width: 4 + 2 + 2 + 2 + 1 + 1 + 1 + 1 + 1.
const ROW_BYTES: usize = 15;
/// Split ids in the rows file: 0 solved (task), 1 dev (task), 2
/// off-task fit, 3 off-task held-out.
const SPLIT_SOLVED: u8 = 0;
const SPLIT_DEV: u8 = 1;
const SPLIT_OFFTASK_FIT: u8 = 2;
/// The posture-2 FRESH test corpus — the ONLY off-task gate surface.
const SPLIT_OFFTASK_TEST: u8 = 3;
/// The posture-1 held-out segment, re-labeled CALIBRATION by the ruling
/// (the threshold's source; reported in the rows for diagnostics, never
/// gated).
const SPLIT_OFFTASK_CALIBRATION: u8 = 4;
/// Posture 3 (katgpt-rs issue 998 posture-3 pre-registration, verdict
/// session 40a812a8): the threshold = the CALIBRATION split's max off-
/// task score × this margin, in RAW RIDGE SCORE SPACE (amendment 1 —
/// the margin NEVER applies in sigmoid space; the placement asserts the
/// space below). M frozen a priori from the pre-registration's measured
/// tail evidence (expected exceedance 0.0357/600 at M = 2.0, far under
/// the 0.5/600 decision line). Calibration-ONLY (amendment 3).
const DETECTOR_TAIL_MARGIN: f64 = 2.0;

struct Args {
    model: PathBuf,
    samples: PathBuf,
    /// The off-task capture cache — posture 2: the DETECTOR-FIT segment
    /// (first 400) + the CALIBRATION segment (last 600; the posture-1
    /// held-out rows, re-labeled by the ruling — the threshold's ONLY
    /// source, never a gate).
    offtask: PathBuf,
    /// The posture-2 FRESH test corpus (single-segment cache, seed
    /// `OFFTASK_SEED_TEST`) — the ONLY off-task gate surface. Required:
    /// gating on calibration/fit data is what posture 1's refusal
    /// forbids. Posture 3: the FRESH P3 corpus (seed
    /// `OFFTASK_SEED_TEST_P3`) replaces it as the gate surface; the
    /// spent posture-2 cache feeds the tail diagnostics only.
    offtask_test: PathBuf,
    report: Option<PathBuf>,
    out: Option<PathBuf>,
    /// Read-only tail-diagnostics mode (posture-3 pre-registration
    /// evidence): fit + score, print the pooled untrained tail, exit
    /// before placement. Writes nothing.
    tail_diagnostics: bool,
}

fn parse_args() -> Result<Args> {
    let mut model = None;
    let mut samples = None;
    let mut offtask = None;
    let mut offtask_test = None;
    let mut report = None;
    let mut out = None;
    let mut tail_diagnostics = false;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--model" => model = Some(PathBuf::from(it.next().context("--model needs a value")?)),
            "--samples" => {
                samples = Some(PathBuf::from(it.next().context("--samples needs a value")?))
            }
            "--offtask" => {
                offtask = Some(PathBuf::from(it.next().context("--offtask needs a value")?))
            }
            "--offtask-test" => {
                offtask_test = Some(PathBuf::from(
                    it.next().context("--offtask-test needs a value")?,
                ))
            }
            "--report" => {
                report = Some(PathBuf::from(it.next().context("--report needs a value")?))
            }
            "--out" => out = Some(PathBuf::from(it.next().context("--out needs a value")?)),
            "--tail-diagnostics" => tail_diagnostics = true,
            other => bail!("unknown arg {other} (usage in the module doc)"),
        }
    }
    let tail_diagnostics = tail_diagnostics;
    Ok(Args {
        model: model.context("--model is required")?,
        samples: samples.context("--samples is required")?,
        offtask: offtask.context("--offtask is required (detector fit + calibration segments)")?,
        offtask_test: offtask_test
            .context("--offtask-test is required (the fresh gate corpus — posture-2 ruling)")?,
        report: if tail_diagnostics {
            report
        } else {
            Some(report.context("--report is required (export mode)")?)
        },
        out: if tail_diagnostics {
            out
        } else {
            Some(out.context("--out is required (export mode)")?)
        },
        tail_diagnostics,
    })
}

fn put_u16(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn main() -> Result<()> {
    let args = parse_args()?;
    std::fs::create_dir_all(
        args.out
            .as_ref()
            .context("--out is required (export mode)")?,
    )
    .with_context(|| format!("mkdir {}", args.out.as_ref().unwrap().display()))?;

    // ── Header-only model facts (n_embd, n_layer → the probe plan) ──
    let t0 = std::time::Instant::now();
    let gguf =
        GgufFile::open(&args.model).with_context(|| format!("open {}", args.model.display()))?;
    let (n_embd, n_layer) = qwen35_dims(&gguf)?;
    let layers = probe_layers(n_layer);
    println!(
        "# header: n_embd={n_embd} n_layer={n_layer} probe layers {layers:?} ({:.1}s)",
        t0.elapsed().as_secs_f32()
    );

    // ── Tokenizer (header-only) → the sealed cell's digit surface ──
    let tok = BpeTokenizer::from_gguf(&gguf).context("BPE tokenizer from gguf")?;
    let reference = TASK.prompt(REF_OPERANDS, FORMAT);
    let digit_tok = digit_tokens(&tok, &reference).context("digit tokens")?;

    // ── Sample cache → the fit surface (answers re-derived: the u8 law) ──
    let (mut solved, mut dev) = load_samples(&args.samples)?;
    for s in solved.iter_mut().chain(dev.iter_mut()) {
        s.answer = TASK.answer(&s.operands);
    }
    println!("# samples: {} solved + {} dev", solved.len(), dev.len());

    // ── Off-task caches: the detector-fit + CALIBRATION segments (the
    // posture-1 corpus, re-labeled by the ruling) + the posture-2 FRESH
    // test corpus (the only gate surface).
    let (offtask_fit, offtask_calibration) = load_offtask(&args.offtask)?;
    println!(
        "# offtask: {} detector-fit + {} CALIBRATION from {} (the calibration segment is the threshold's only source, never a gate)",
        offtask_fit.len(),
        offtask_calibration.len(),
        args.offtask.display()
    );
    if offtask_fit.len() != riir_infer_core::grafts_cell::OFFTASK_FIT_N
        || offtask_calibration.len() != riir_infer_core::grafts_cell::OFFTASK_HELDOUT_N
    {
        bail!(
            "offtask cache must carry the pinned corpus ({} fit + {} calibration)",
            riir_infer_core::grafts_cell::OFFTASK_FIT_N,
            riir_infer_core::grafts_cell::OFFTASK_HELDOUT_N
        );
    }
    let offtask_test: Vec<riir_infer_core::grafts_cell::OfftaskRow> = serde_json::from_reader(
        std::fs::File::open(&args.offtask_test)
            .with_context(|| format!("open {}", args.offtask_test.display()))?,
    )
    .context("load offtask test")?;
    println!(
        "# offtask TEST: {} fresh rows from {} (seed OFFTASK_SEED_TEST — the ONLY gate surface)",
        offtask_test.len(),
        args.offtask_test.display()
    );
    if offtask_test.len() != riir_infer_core::grafts_cell::OFFTASK_TEST_N {
        bail!(
            "offtask test cache must carry the pinned corpus ({} rows)",
            riir_infer_core::grafts_cell::OFFTASK_TEST_N
        );
    }
    let offtask_bytes =
        std::fs::read(&args.offtask).with_context(|| format!("read {}", args.offtask.display()))?;
    let offtask_blake3 = blake3::hash(&offtask_bytes).to_hex().to_string();
    let offtask_test_bytes = std::fs::read(&args.offtask_test)
        .with_context(|| format!("read {}", args.offtask_test.display()))?;
    let offtask_test_blake3 = blake3::hash(&offtask_test_bytes).to_hex().to_string();

    // ── The EXACT sealed-cell fit (reads + table; the SWITCH below is the detector) ──
    let parts = fit_parts(&solved, &layers, n_embd, RIDGE)
        .map_err(anyhow::Error::msg)
        .context("graft fit (solved split)")?;

    // ── The task-DETECTOR switch (998 T2): labels task=1 / off-task=0
    // over solved + offtask-fit; threshold PLACED (never a default pin):
    // min(off-task-fit 99.5% quantile, min task-fit score) — task-side
    // zero-disarm wins by construction on the fit split; held-out
    // evidence is a measured verdict, never a moved threshold.
    let k = layers.len();
    let d = k * n_embd;
    let n_reads = solved[0].operands.len();
    let det_d = d * n_reads;
    let all_ids: Vec<u32> = (0..n_reads)
        .flat_map(|slot| {
            layers
                .iter()
                .map(move |&l| riir_infer_core::grafts_cell::read_id(slot, l))
        })
        .collect();
    let mut det_x = Vec::with_capacity((solved.len() + offtask_fit.len()) * det_d);
    let mut det_labels: Vec<bool> = Vec::with_capacity(solved.len() + offtask_fit.len());
    for s in &solved {
        det_x.extend_from_slice(&s.rows);
        det_labels.push(true);
    }
    for r in &offtask_fit {
        det_x.extend_from_slice(&r.rows);
        det_labels.push(false);
    }
    let mut detector = fit_switch(
        &det_x,
        det_labels.len(),
        det_d,
        &all_ids,
        &vec![n_embd; k * n_reads],
        &det_labels,
        RIDGE,
        0.5,
        true,
    )
    .map_err(anyhow::Error::msg)
    .context("detector fit (solved + offtask-fit)")?;

    // Scores over every surface (the placement + diagnostics currency).
    // The switch's segments are slot-major then layer-major — the rows
    // layout's own order — so the fold walks them IN ORDER with a
    // running offset (never the raw layer id; that is the host's
    // address space, not a row offset).
    let score = |sw: &GraftSwitch, rows: &[f32]| {
        let mut s = sw.bias;
        let mut off = 0usize;
        for seg in &sw.segments {
            s += seg.score(&rows[off..off + n_embd]);
            off += n_embd;
        }
        f64::from(s)
    };
    let task_fit_scores: Vec<f64> = solved.iter().map(|s| score(&detector, &s.rows)).collect();
    let off_fit_scores: Vec<f64> = offtask_fit
        .iter()
        .map(|r| score(&detector, &r.rows))
        .collect();
    let task_dev_scores: Vec<f64> = dev.iter().map(|s| score(&detector, &s.rows)).collect();
    let cal_scores: Vec<f64> = offtask_calibration
        .iter()
        .map(|r| score(&detector, &r.rows))
        .collect();
    let test_scores: Vec<f64> = offtask_test
        .iter()
        .map(|r| score(&detector, &r.rows))
        .collect();

    // ── Posture-3 tail diagnostics (katgpt-rs issue 998 posture-3
    // pre-registration, verdict session 40a812a8 amendments 2+3+4): the
    // pooled untrained-tail evidence the pre-registration quotes, from
    // the SAME fit + score path the placement uses. READ-ONLY: nothing
    // is written, the run exits before threshold placement. Pooling is
    // a diagnostics decision only — the THRESHOLD stays calibration-only
    // (amendment 3, frozen).
    if args.tail_diagnostics {
        print_tail_diagnostics(
            &cal_scores,
            &test_scores,
            &offtask_calibration,
            &offtask_test,
        );
        return Ok(());
    }

    // Threshold placement — POSTURE 3 (pre-registered in katgpt-rs
    // issue 998 BEFORE the fresh capture exists, commit ebe59f611; the
    // verdict session 40a812a8 AGREE): the CALIBRATION split's max
    // off-task score × M = 2.0, in RAW RIDGE SCORE SPACE. The margin is
    // a multiplier on the SCORE, never on the sigmoid — asserted here:
    // a sigmoid-space 2× of a >0.5 threshold would exceed 1 and disarm
    // everything, so the placed sigmoid must land in (0.5, 1) and the
    // placed score must be exactly M × cal_max. No automatic fallback —
    // a degenerate placement is a FAIL, recorded.
    let mut cal_sorted = cal_scores.clone();
    cal_sorted.sort_by(f64::total_cmp);
    let cal_max = cal_sorted[cal_sorted.len() - 1];
    let th_score = DETECTOR_TAIL_MARGIN * cal_max;
    let th_source = format!(
        "calibration max x M={DETECTOR_TAIL_MARGIN} (RAW score space, n={}), posture-3 pre-registration commit ebe59f611",
        cal_sorted.len()
    );
    let th_sigmoid = exact_sigmoid(th_score as f32);
    assert!(
        th_score > cal_max && (0.5..1.0).contains(&th_sigmoid),
        "posture-3 placement space assertion FAILED: score {th_score} vs cal_max {cal_max}, sigmoid {th_sigmoid} — the margin must multiply the RAW score"
    );
    detector.threshold = th_sigmoid;
    // Tail support at a max-based threshold = the calibration rows AT the
    // max (1 unless ties) — disclosed on every verdict line per the
    // posture-2 ruling's tail-support law.
    let tail_support = cal_sorted
        .iter()
        .rev()
        .take_while(|&&v| v >= cal_max)
        .count();

    // Diagnostics: arm rates per surface at the PLACED threshold + AUC
    // (fit and held-out separately — the T1 law: in-sample proves
    // nothing) + the fragility gaps.
    let auc = |task_s: &[f64], off_s: &[f64]| -> f64 {
        // Rank-based AUC of score vs (task=1): pooled, stable sort.
        let mut pooled: Vec<(f64, bool)> = task_s
            .iter()
            .map(|&v| (v, true))
            .chain(off_s.iter().map(|&v| (v, false)))
            .collect();
        pooled.sort_by(|a, b| a.0.total_cmp(&b.0));
        let n1 = task_s.len();
        let n0 = off_s.len();
        let mut rank_sum = 0.0f64;
        for (r, (_, is_task)) in pooled.iter().enumerate() {
            if *is_task {
                rank_sum += (r + 1) as f64;
            }
        }
        (rank_sum - n1 as f64 * (n1 + 1) as f64 / 2.0) / (n0 * n1) as f64
    };
    let armed_frac = |s: &[f64]| {
        s.iter()
            .filter(|&&v| exact_sigmoid(v as f32) >= th_sigmoid)
            .count()
    };
    let fit_auc = auc(&task_fit_scores, &off_fit_scores);
    let cal_auc = auc(&task_dev_scores, &cal_scores);
    let test_auc = auc(&task_dev_scores, &test_scores);
    let gap_cal = task_dev_scores
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min)
        - cal_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let gap_test = task_dev_scores
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min)
        - test_scores
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
    let contract = |task_s: &[f64], name: &str| -> Result<()> {
        let disarmed = task_s
            .iter()
            .filter(|&&v| exact_sigmoid(v as f32) < th_sigmoid)
            .count();
        if disarmed > 0 {
            bail!(
                "detector contract FAILS at placement: {disarmed} disarmed task row(s) on {name} — lane-A GOAT FAIL (998 T2: the threshold never moves on held-out evidence)"
            );
        }
        Ok(())
    };
    // The CONTRACT arms (dev-200 held-out task rows: any disarm = FAIL).
    // The fit split is reported, never gated (in-sample proves nothing).
    contract(&task_dev_scores, "dev (held-out task)")?;

    println!(
        "# detector: threshold score {th_score:.6} (sigmoid {th_sigmoid:.6}, source: {th_source}, cal max {cal_max:.6}, tail support {tail_support}) | AUC fit {fit_auc:.4} cal {cal_auc:.4} TEST {test_auc:.4}"
    );
    println!(
        "# detector rates: task fit {}/{} armed (reported) | task dev {}/{} armed (gated contract) | calibration {}/{} armed (placement data, reported) | TEST {}/{} armed (GATED)",
        armed_frac(&task_fit_scores),
        task_fit_scores.len(),
        armed_frac(&task_dev_scores),
        task_dev_scores.len(),
        armed_frac(&cal_scores),
        cal_scores.len(),
        armed_frac(&test_scores),
        test_scores.len(),
    );
    println!("# detector gaps: calibration {gap_cal:.4} | TEST {gap_test:.4} (the fragility read)");
    // Score-distribution percentiles per surface (min/p5/p50/p95/max) —
    // the placement's evidence, printed so a pathological asymmetry is
    // visible instead of inferred.
    let pct = |name: &str, s: &[f64]| {
        let mut v = s.to_vec();
        v.sort_by(f64::total_cmp);
        let q =
            |p: f64| v[(((p * v.len() as f64).ceil() as usize).saturating_sub(1)).min(v.len() - 1)];
        println!(
            "# scores {name}: min {:.4} p5 {:.4} p50 {:.4} p95 {:.4} max {:.4}",
            v[0],
            q(0.05),
            q(0.50),
            q(0.95),
            v[v.len() - 1]
        );
    };
    pct("task-fit", &task_fit_scores);
    pct("offtask-fit", &off_fit_scores);
    pct("task-dev", &task_dev_scores);
    pct("calibration", &cal_scores);
    pct("TEST", &test_scores);
    for c in [1u8, 2, 3] {
        let mut n = 0usize;
        let mut armed = 0usize;
        for (k, r) in offtask_test.iter().enumerate() {
            if r.class == c {
                n += 1;
                if exact_sigmoid(test_scores[k] as f32) >= th_sigmoid {
                    armed += 1;
                }
            }
        }
        println!("# offtask TEST class {c}: {armed}/{n} armed");
    }

    // ── Frozen decision rows v2: the REAL read/switch path over each row
    // (task rows: solved/dev from the task cache; off-task rows: fit and
    // held-out from the off-task cache; armed bits from the PLACED
    // detector switch).
    let out = args.out.as_ref().context("--out is required (export mode)")?;
    let mut rows = Vec::with_capacity(
        (solved.len()
            + dev.len()
            + offtask_fit.len()
            + offtask_test.len()
            + offtask_calibration.len())
            * ROW_BYTES
            + 12,
    );
    rows.extend_from_slice(ROWS_MAGIC);
    put_u32(&mut rows, ROWS_VERSION);
    put_u32(
        &mut rows,
        (solved.len()
            + dev.len()
            + offtask_fit.len()
            + offtask_test.len()
            + offtask_calibration.len()) as u32,
    );
    for s in &solved {
        let reader = RowReader::new(&layers, n_embd, &s.rows);
        let mut triple = [0u16; 3];
        for (j, read) in parts.reads.iter().enumerate() {
            triple[j] = read.value_at(&reader);
        }
        let armed = detector.armed_at(&reader);
        put_u32(
            &mut rows,
            u32::try_from(s.instance).context("instance > u32")?,
        );
        put_u16(&mut rows, triple[0]);
        put_u16(&mut rows, triple[1]);
        put_u16(&mut rows, triple[2]);
        rows.push(s.answer);
        rows.push(u8::from(s.base_wrong));
        rows.push(u8::from(armed));
        rows.push(0); // class: task
        rows.push(SPLIT_SOLVED);
    }
    for s in &dev {
        let reader = RowReader::new(&layers, n_embd, &s.rows);
        let mut triple = [0u16; 3];
        for (j, read) in parts.reads.iter().enumerate() {
            triple[j] = read.value_at(&reader);
        }
        let armed = detector.armed_at(&reader);
        put_u32(
            &mut rows,
            u32::try_from(s.instance).context("instance > u32")?,
        );
        put_u16(&mut rows, triple[0]);
        put_u16(&mut rows, triple[1]);
        put_u16(&mut rows, triple[2]);
        rows.push(s.answer);
        rows.push(u8::from(s.base_wrong));
        rows.push(u8::from(armed));
        rows.push(0); // class: task
        rows.push(SPLIT_DEV);
    }
    for (split_id, surface) in [
        (SPLIT_OFFTASK_FIT, &offtask_fit),
        (SPLIT_OFFTASK_TEST, &offtask_test),
        (SPLIT_OFFTASK_CALIBRATION, &offtask_calibration),
    ] {
        for r in surface {
            let reader = RowReader::new(&layers, n_embd, &r.rows);
            let mut triple = [0u16; 3];
            for (j, read) in parts.reads.iter().enumerate() {
                triple[j] = read.value_at(&reader);
            }
            let armed = detector.armed_at(&reader);
            put_u32(&mut rows, u32::try_from(r.index).context("instance > u32")?);
            put_u16(&mut rows, triple[0]);
            put_u16(&mut rows, triple[1]);
            put_u16(&mut rows, triple[2]);
            rows.push(0); // no answer on off-task rows
            rows.push(0); // base_wrong is a task-side label
            rows.push(u8::from(armed));
            rows.push(r.class);
            rows.push(split_id);
        }
    }

    // Sanity: the fit's own triples must equal the recorded solved rows
    // (one read path, one truth).
    for (i, t) in parts.triples.iter().enumerate() {
        let off = 12 + i * ROW_BYTES;
        let rec = [
            u16::from_le_bytes([rows[off + 4], rows[off + 5]]),
            u16::from_le_bytes([rows[off + 6], rows[off + 7]]),
            u16::from_le_bytes([rows[off + 8], rows[off + 9]]),
        ];
        if rec.as_slice() != t.as_slice() {
            bail!("triple mismatch at solved row {i}: recorded {rec:?} vs fit {t:?}");
        }
    }

    // ── Writes: the sealed cell's surface (decided digit → +BIAS_DELTA) ──
    let writes = GraftWrite::new(
        0,
        (0..10u16)
            .map(|v| WriteEntry {
                value: v,
                biases: vec![(digit_tok[v as usize] as u32, BIAS_DELTA)],
            })
            .collect::<Vec<_>>(),
    )
    .map_err(anyhow::Error::msg)?;
    let graft = Graft::new(
        parts.reads.clone(),
        parts.table.clone(),
        vec![writes],
        vec![detector.clone()],
    )
    .map_err(anyhow::Error::msg)?;

    // ── Artifact envelope (GRFZ) ──
    let artifact = GraftArtifact::mint(&graft, ARTIFACT_EPOCH);
    let mut bin = Vec::with_capacity(48 + artifact.canonical.len());
    bin.extend_from_slice(ARTIFACT_MAGIC);
    put_u32(&mut bin, GRAFT_ARTIFACT_FORMAT_VERSION);
    bin.extend_from_slice(&artifact.graft_digest);
    put_u64(&mut bin, artifact.epoch);
    put_u32(&mut bin, artifact.canonical.len() as u32);
    bin.extend_from_slice(&artifact.canonical);

    let artifact_path = out.join("grafts_t2b_artifact.bin");
    let rows_path = out.join("grafts_t2b_rows.bin");
    std::fs::write(&artifact_path, &bin)
        .with_context(|| format!("write {}", artifact_path.display()))?;
    std::fs::write(&rows_path, &rows).with_context(|| format!("write {}", rows_path.display()))?;

    let artifact_hash = blake3::hash(&bin).to_hex().to_string();
    let rows_hash = blake3::hash(&rows).to_hex().to_string();
    println!(
        "# artifact {} ({} bytes, graft digest {}) blake3 {}",
        artifact_path.display(),
        bin.len(),
        artifact.graft_digest_hex(),
        &artifact_hash[..16]
    );
    println!(
        "# rows {} ({} bytes, {} rows) blake3 {}",
        rows_path.display(),
        rows.len(),
        rows.len().saturating_sub(12) / ROW_BYTES,
        &rows_hash[..16]
    );

    // ── Manifest (provenance chain + the crosstab the G3 cell reads) ──
    let report_path = args
        .report
        .as_ref()
        .context("--report is required (export mode)")?;
    let report: serde_json::Value = serde_json::from_reader(
        std::fs::File::open(report_path)
            .with_context(|| format!("open {}", report_path.display()))?,
    )
    .context("parse report")?;
    let manifest = serde_json::json!({
        "issue": "katgpt-rs issue 988 T4 — the GOAT bench's frozen-read fixture (riir-infer issue 1206 lane)",
        "producer": {
            "checkpoint_blake3": report["checkpoint_blake3"],
            "sealed_report_payload_blake3": report["payload_blake3"],
            "split_seed": report["split_seed"],
            "split_digest": report["split_digest"],
            "solved_triple_coverage": report["solved_triple_coverage"],
            "sealed_rows": report["sealed_rows"],
        },
        "fit_pins": {
            "task": "ThreeOpAdd10",
            "prompt_format": "FewShot",
            "ridge": RIDGE,
            "bias_delta": BIAS_DELTA,
            "detector_threshold_score": th_score,
            "detector_threshold_sigmoid": th_sigmoid,
            "detector_threshold_source": th_source,
            "posture": "3 — calibration-max x M=2.0 (raw score space; posture-1 fit-quantile FAILED 234/600, posture-2 calibration-quantile FAILED 1/600 — both recorded, never re-thresholded)",
            "n_embd": n_embd,
            "probe_layers": layers,
            "artifact_epoch": ARTIFACT_EPOCH,
        },
        "offtask_corpus": {
            "fit_calibration_cache_blake3": offtask_blake3,
            "detector_fit_n": offtask_fit.len(),
            "calibration_n": offtask_calibration.len(),
            "test_cache_blake3": offtask_test_blake3,
            "test_n": offtask_test.len(),
            "test_provenance": "OFFTASK_SEED_TEST_P3 fresh capture (posture 3, seed 0x998_0FF3), taken AFTER the posture-3 pre-registration commit ebe59f611 (the ruling's contamination ban); the spent posture-2 corpus feeds the tail diagnostics only",
            "classes": "1 easy / 2 hard-A digits-non-arithmetic / 3 hard-B arithmetic-words (998 T2 pre-registration)",
        },
        "fixtures": {
            "grafts_t2b_artifact.bin": {
                "bytes": bin.len(),
                "blake3": artifact_hash,
                "graft_digest_hex": artifact.graft_digest_hex(),
                "canonical_bytes": artifact.canonical.len(),
                "rules": graft.table.rules.len(),
            },
            "grafts_t2b_rows.bin": {
                "bytes": rows.len(),
                "blake3": rows_hash,
                "rows": solved.len() + dev.len() + offtask_fit.len() + offtask_test.len() + offtask_calibration.len(),
                "row_bytes": ROW_BYTES,
                "format_version": ROWS_VERSION,
            },
        },
        "detector": {
            "law": "998 T2 posture 3 — gated legs: dev-200 task disarm contract + sealed-400 capture-only arm check (cell run) + off-task TEST 600 CP-UB gate (0 false ARMS, the durable unit) + hard sub-leg; fit/calibration rates reported, never gated; FIRES reported beside, never gating (arms are table-independent)",
            "fit_auc": fit_auc,
            "cal_auc": cal_auc,
            "test_auc": test_auc,
            "threshold_score": th_score,
            "threshold_sigmoid": th_sigmoid,
            "tail_margin": DETECTOR_TAIL_MARGIN,
            "cal_max_score": cal_max,
            "tail_support": tail_support,
            "class_maxima_test": {
                "easy": test_scores
                    .iter()
                    .zip(offtask_test.iter())
                    .filter(|(_, r)| r.class == 1)
                    .map(|(s, _)| *s)
                    .fold(f64::NEG_INFINITY, f64::max),
                "hard_a": test_scores
                    .iter()
                    .zip(offtask_test.iter())
                    .filter(|(_, r)| r.class == 2)
                    .map(|(s, _)| *s)
                    .fold(f64::NEG_INFINITY, f64::max),
                "hard_b": test_scores
                    .iter()
                    .zip(offtask_test.iter())
                    .filter(|(_, r)| r.class == 3)
                    .map(|(s, _)| *s)
                    .fold(f64::NEG_INFINITY, f64::max),
            },
            "task_fit_armed": armed_frac(&task_fit_scores),
            "task_fit_n": task_fit_scores.len(),
            "task_dev_armed": armed_frac(&task_dev_scores),
            "task_dev_n": task_dev_scores.len(),
            "offtask_fit_armed": armed_frac(&off_fit_scores),
            "offtask_fit_n": off_fit_scores.len(),
            "calibration_armed": armed_frac(&cal_scores),
            "calibration_n": cal_scores.len(),
            "test_armed": armed_frac(&test_scores),
            "test_n": test_scores.len(),
            "gap_calibration": gap_cal,
            "gap_test": gap_test,
            "kl_labeling": "Zipf-surface KL(top-40) — never a real-model no-harm proof",
        },
    });
    let manifest_path = out.join("grafts_t2b_manifest.json");
    std::fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)
        .with_context(|| format!("write {}", manifest_path.display()))?;
    println!("# manifest {}", manifest_path.display());
    Ok(())
}

/// The `qwen35.` header facts the export needs (header-only — no
/// weights, no GPU; the fit consumes the sample cache, never the model).
fn qwen35_dims(gguf: &GgufFile) -> Result<(usize, usize)> {
    let n_embd = gguf
        .metadata_u64("qwen35.embedding_length")
        .context("qwen35.embedding_length missing from the GGUF header")? as usize;
    let n_layer = gguf
        .metadata_u64("qwen35.block_count")
        .context("qwen35.block_count missing from the GGUF header")? as usize;
    Ok((n_embd, n_layer))
}

/// The posture-3 pre-registration's tail evidence (katgpt-rs issue 998,
/// verdict session 40a812a8 amendments 2+3+4), printed READ-ONLY from
/// the same fit + score path the placement uses:
/// - amendment 2: the top scores of the POOLED 1,200 untrained rows
///   (calibration + the spent posture-2 test) plus a tail estimate with
///   expected exceedances at candidate margins — the number the 0-arms
///   gate's power rests on, quoted BEFORE the fresh capture exists;
/// - amendment 3 disclosed here: the threshold stays CALIBRATION-ONLY
///   (the posture-2 test corpus was a gate surface; admitting it into
///   placement would blur the placement/gate separation the rulings
///   built — the margin carries the tail uncertainty instead);
/// - amendment 4: per-class maxima, hard-A named (the posture-2 false
///   arm was a hard-A row).
fn print_tail_diagnostics(
    cal_scores: &[f64],
    test_scores: &[f64],
    cal_rows: &[riir_infer_core::grafts_cell::OfftaskRow],
    test_rows: &[riir_infer_core::grafts_cell::OfftaskRow],
) {
    println!(
        "# tail-diagnostics: posture-3 pre-registration inputs (verdict 40a812a8 amendments 2+3+4) — read-only, no placement"
    );

    // Posture-2 reproduction check: the SAME quantile rule must land on
    // the recorded threshold — proof this run reads the same fit. A
    // mismatch is a hard stop, never a footnote.
    let mut cal_sorted = cal_scores.to_vec();
    cal_sorted.sort_by(f64::total_cmp);
    let q_idx = ((0.995 * cal_sorted.len() as f64).ceil() as usize)
        .saturating_sub(1)
        .min(cal_sorted.len() - 1);
    let reproduced = cal_sorted[q_idx];
    const POSTURE2_THRESHOLD: f64 = 0.05397161468863487;
    let matches = (reproduced - POSTURE2_THRESHOLD).abs() < 1e-12;
    println!(
        "# posture-2 reproduction: cal quantile(0.995) = {reproduced:.17} vs recorded {POSTURE2_THRESHOLD:.17} — {}",
        if matches {
            "MATCH (same fit surface)"
        } else {
            "MISMATCH — STOP: this run is NOT reading the posture-2 fit surface"
        }
    );
    if !matches {
        println!("# tail-diagnostics ABORTED on the reproduction mismatch (the numbers below would describe a different detector)");
        return;
    }

    let mut pooled: Vec<f64> = cal_scores.iter().chain(test_scores.iter()).copied().collect();
    pooled.sort_by(f64::total_cmp);
    let n = pooled.len();
    let cal_max = cal_sorted[cal_sorted.len() - 1];
    let mut test_sorted = test_scores.to_vec();
    test_sorted.sort_by(f64::total_cmp);
    let test_max = test_sorted[test_sorted.len() - 1];
    let top = |k: usize, v: &[f64]| -> Vec<String> {
        let k = k.min(v.len());
        v[v.len() - k..]
            .iter()
            .rev()
            .map(|x| format!("{x:.6}"))
            .collect()
    };
    println!(
        "# pooled untrained rows: {n} (calibration {} + spent posture-2 test {})",
        cal_scores.len(),
        test_scores.len()
    );
    println!("# pooled top-10 (desc): {}", top(10, &pooled).join(" "));
    println!(
        "# cal top-5: {} | test top-5: {}",
        top(5, &cal_sorted).join(" "),
        top(5, &test_sorted).join(" ")
    );
    println!(
        "# maxima: cal {cal_max:.6} test {test_max:.6} pooled {:.6} | fresh/cal ratio {:.4}",
        cal_max.max(test_max),
        test_max / cal_max
    );

    // Amendment 4: per-class maxima — hard-A named (its posture-2 false
    // arm sat at 0.1043).
    for c in [1u8, 2, 3] {
        let cm = cal_rows
            .iter()
            .zip(cal_scores)
            .filter(|(r, _)| r.class == c)
            .map(|(_, s)| *s)
            .fold(f64::NEG_INFINITY, f64::max);
        let tm = test_rows
            .iter()
            .zip(test_scores)
            .filter(|(r, _)| r.class == c)
            .map(|(_, s)| *s)
            .fold(f64::NEG_INFINITY, f64::max);
        let name = match c {
            1 => "easy",
            2 => "hard-A",
            _ => "hard-B",
        };
        println!("# class {c} ({name}) max: cal {cm:.6} test {tm:.6}");
    }

    // Amendment 2: the tail estimate. GPD over the pooled exceedances
    // above the pooled p90 (method-of-moments; diagnostics grade, so a
    // validity failure is printed, never hidden), expected exceedances
    // per 600 fresh rows at each candidate margin, and the raw top-k
    // spacing as the nonparametric cross-check.
    let u = pooled[((0.90 * (n as f64 - 1.0)) as usize).min(n - 1)];
    let ex: Vec<f64> = pooled
        .iter()
        .copied()
        .filter(|&x| x > u)
        .map(|x| x - u)
        .collect();
    let n_ex = ex.len();
    if n_ex >= 10 {
        let m = ex.iter().sum::<f64>() / n_ex as f64;
        let var = ex
            .iter()
            .map(|e| (e - m) * (e - m))
            .sum::<f64>()
            / (n_ex as f64 - 1.0);
        let xi = 0.5 * (1.0 - m * m / var);
        let sigma = m * (1.0 - xi);
        let p_u = n_ex as f64 / n as f64;
        println!(
            "# GPD(u=p90={u:.6}, n_ex={n_ex}): xi {xi:.4} sigma {sigma:.6} p_u {p_u:.5} (moments){}",
            if (0.0..0.5).contains(&xi) {
                ""
            } else {
                " [INVALID xi — fit not usable, read the spacing line instead]"
            }
        );
        if (0.0..0.5).contains(&xi) {
            for &margin in &[1.5f64, 2.0, 2.5, 3.0] {
                let t = margin * cal_max;
                if t <= u {
                    println!(
                        "# expected exceedances @ threshold {t:.6} (M={margin}): BELOW the fit threshold u — GPD unusable there"
                    );
                } else {
                    let tail = p_u * (1.0 + xi * (t - u) / sigma).powf(-1.0 / xi);
                    println!(
                        "# expected exceedances @ threshold {t:.6} (M={margin}): {:.4} per 600 fresh rows (P per row {:.3e})",
                        600.0 * tail,
                        tail
                    );
                }
            }
        }
    } else {
        println!("# GPD skipped: only {n_ex} exceedances above the pooled p90 {u:.6}");
    }
    let top5: Vec<f64> = pooled[pooled.len().saturating_sub(5)..].to_vec();
    let gaps: Vec<String> = top5
        .windows(2)
        .map(|w| format!("{:.6}", w[1] - w[0]))
        .collect();
    println!(
        "# top-5 spacing (asc, gaps): {} → gaps {}",
        top5.iter().map(|x| format!("{x:.6}")).collect::<Vec<_>>().join(" "),
        gaps.join(" ")
    );
    let above_p2 = pooled
        .iter()
        .filter(|&&x| x > POSTURE2_THRESHOLD)
        .count();
    println!(
        "# context: {above_p2}/{n} pooled rows exceed the posture-2 threshold {POSTURE2_THRESHOLD:.5}"
    );
}
