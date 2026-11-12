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
//!     --report .benchmarks/grafts_cell_t2b/report.json \
//!     --out .benchmarks/grafts_cell_t2b
//! ```
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
/// The detector's false-arm target (the G3(b) flip bar): threshold
/// placed at the off-task-fit score quantile 0.995 — 998 T2
/// pre-registration.
const DETECTOR_QUANTILE: f64 = 0.995;

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
    /// forbids.
    offtask_test: PathBuf,
    report: PathBuf,
    out: PathBuf,
}

fn parse_args() -> Result<Args> {
    let mut model = None;
    let mut samples = None;
    let mut offtask = None;
    let mut offtask_test = None;
    let mut report = None;
    let mut out = None;
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
            other => bail!("unknown arg {other} (usage in the module doc)"),
        }
    }
    Ok(Args {
        model: model.context("--model is required")?,
        samples: samples.context("--samples is required")?,
        offtask: offtask.context("--offtask is required (detector fit + calibration segments)")?,
        offtask_test: offtask_test
            .context("--offtask-test is required (the fresh gate corpus — posture-2 ruling)")?,
        report: report.context("--report is required")?,
        out: out.context("--out is required")?,
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
    std::fs::create_dir_all(&args.out).with_context(|| format!("mkdir {}", args.out.display()))?;

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

    // Threshold placement — POSTURE 2 (pre-registered BEFORE the fresh
    // capture existed): the CALIBRATION split's off-task score quantile
    // at α = 0.995 (split-conformal shape; the calibration rows are
    // off-task rows the detector never trained on — the posture-1
    // held-out segment, re-labeled by the ruling). No automatic
    // fallback — a degenerate placement is a FAIL, recorded.
    let mut cal_sorted = cal_scores.clone();
    cal_sorted.sort_by(f64::total_cmp);
    let q_idx = ((DETECTOR_QUANTILE * cal_sorted.len() as f64).ceil() as usize)
        .saturating_sub(1)
        .min(cal_sorted.len() - 1);
    let q_tail_support = cal_sorted.len() - q_idx;
    let q_value = cal_sorted[q_idx];
    let th_score = q_value;
    let th_source = format!(
        "calibration-split quantile (alpha={DETECTOR_QUANTILE}, n={})",
        cal_sorted.len()
    );
    let th_sigmoid = exact_sigmoid(th_score as f32);
    detector.threshold = th_sigmoid;

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
        "# detector: threshold score {th_score:.4} (sigmoid {th_sigmoid:.4}, source: {th_source}, tail support {q_tail_support}) | AUC fit {fit_auc:.4} cal {cal_auc:.4} TEST {test_auc:.4}"
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

    let artifact_path = args.out.join("grafts_t2b_artifact.bin");
    let rows_path = args.out.join("grafts_t2b_rows.bin");
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
    let report: serde_json::Value = serde_json::from_reader(
        std::fs::File::open(&args.report)
            .with_context(|| format!("open {}", args.report.display()))?,
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
            "posture": "2 — calibration-split placement; posture-1 (fit-quantile placement) FAILED the off-task gate 234/600 and was recorded, never re-thresholded",
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
            "test_provenance": "OFFTASK_SEED_TEST fresh capture, taken AFTER the posture-2 rule commit (the ruling's contamination ban)",
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
            "law": "998 T2 posture 2 — gated legs: dev-200 task disarm contract + sealed-400 capture-only arm check (cell run) + off-task TEST 600 CP-UB gate + hard sub-leg; fit/calibration rates reported, never gated",
            "fit_auc": fit_auc,
            "cal_auc": cal_auc,
            "test_auc": test_auc,
            "threshold_score": th_score,
            "threshold_sigmoid": th_sigmoid,
            "quantile": DETECTOR_QUANTILE,
            "quantile_tail_support": q_tail_support,
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
    let manifest_path = args.out.join("grafts_t2b_manifest.json");
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
