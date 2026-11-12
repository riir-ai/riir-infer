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
//!        container: magic + version u32 + n u32, then per row
//!        {instance u32, d0 u16, d1 u16, d2 u16, answer u8, base_wrong u8,
//!        armed u8, split u8} (split: 0 = solved, 1 = dev).
//!
//! Both files are BLAKE3-hashed into the printed manifest JSON, which
//! also carries the provenance chain (the sealed report's checkpoint +
//! payload digests, the fit pins) and the armed × base-wrong crosstab the
//! bench's G3 selectivity cell reads.
//!
//! ```bash
//! cargo run --release --features grafts_cell --bin grafts_export -- \
//!     --model ../riir-train/data/Qwen3.5-0.8B-Base-Q8_0.gguf \
//!     --samples .benchmarks/grafts_cell_t2b/samples.json \
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
    PromptFormat, RowReader, TaskKind, digit_tokens, fit_parts, load_samples, probe_layers,
};
use riir_infer_core::tokenizer::BpeTokenizer;

use katgpt_core::graft::{Graft, GraftWrite, WriteEntry};
use katgpt_core::{GRAFT_ARTIFACT_FORMAT_VERSION, GraftArtifact};

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
const ROWS_VERSION: u32 = 1;
/// Row record width: 4 + 2 + 2 + 2 + 1 + 1 + 1 + 1.
const ROW_BYTES: usize = 14;

struct Args {
    model: PathBuf,
    samples: PathBuf,
    report: PathBuf,
    out: PathBuf,
}

fn parse_args() -> Result<Args> {
    let mut model = None;
    let mut samples = None;
    let mut report = None;
    let mut out = None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--model" => model = Some(PathBuf::from(it.next().context("--model needs a value")?)),
            "--samples" => {
                samples = Some(PathBuf::from(it.next().context("--samples needs a value")?))
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
    let gguf = GgufFile::open(&args.model)
        .with_context(|| format!("open {}", args.model.display()))?;
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

    // ── The EXACT sealed-cell fit (solved split, pinned λ) ──
    let parts = fit_parts(&solved, &layers, n_embd, RIDGE)
        .map_err(anyhow::Error::msg)
        .context("graft fit (solved split)")?;

    // ── Frozen decision rows: the REAL read/switch path over each row ──
    let mut rows = Vec::with_capacity((solved.len() + dev.len()) * ROW_BYTES + 12);
    rows.extend_from_slice(ROWS_MAGIC);
    put_u32(&mut rows, ROWS_VERSION);
    put_u32(&mut rows, (solved.len() + dev.len()) as u32);
    let mut stats = Crosstab::default();
    for (split, samples) in [(0u8, &solved), (1u8, &dev)] {
        for s in samples {
            let reader = RowReader::new(&layers, n_embd, &s.rows);
            let mut triple = [0u16; 3];
            for (j, read) in parts.reads.iter().enumerate() {
                triple[j] = read.value_at(&reader);
            }
            let armed = parts.switch.armed_at(&reader);
            stats.count(split, s.base_wrong, armed);
            put_u32(&mut rows, u32::try_from(s.instance).context("instance > u32")?);
            put_u16(&mut rows, triple[0]);
            put_u16(&mut rows, triple[1]);
            put_u16(&mut rows, triple[2]);
            rows.push(s.answer);
            rows.push(u8::from(s.base_wrong));
            rows.push(u8::from(armed));
            rows.push(split);
        }
    }
    stats.print();

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
        vec![parts.switch.clone()],
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
            "switch_threshold": 0.5,
            "n_embd": n_embd,
            "probe_layers": layers,
            "artifact_epoch": ARTIFACT_EPOCH,
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
                "rows": solved.len() + dev.len(),
                "row_bytes": ROW_BYTES,
            },
        },
        "crosstab": stats.to_json(),
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
        .context("qwen35.embedding_length missing from the GGUF header")?
        as usize;
    let n_layer = gguf
        .metadata_u64("qwen35.block_count")
        .context("qwen35.block_count missing from the GGUF header")?
        as usize;
    Ok((n_embd, n_layer))
}

/// armed × base-wrong counts per split — the G3 selectivity cell's
/// expected-shape record (the bench measures the same corpus itself;
/// this is the provenance record of what the fitted artifact does).
#[derive(Default)]
struct Crosstab {
    solved: [usize; 4],  // [right+disarmed, right+armed, wrong+disarmed, wrong+armed]
    dev: [usize; 4],
}

impl Crosstab {
    fn count(&mut self, split: u8, base_wrong: bool, armed: bool) {
        let cell = usize::from(base_wrong) * 2 + usize::from(armed);
        if split == 0 {
            self.solved[cell] += 1;
        } else {
            self.dev[cell] += 1;
        }
    }

    fn print(&self) {
        println!(
            "# crosstab solved [right+dis {}, right+armed {}, wrong+dis {}, wrong+armed {}] dev [{} {} {} {}]",
            self.solved[0], self.solved[1], self.solved[2], self.solved[3],
            self.dev[0], self.dev[1], self.dev[2], self.dev[3]
        );
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "solved": {"base_right_disarmed": self.solved[0], "base_right_armed": self.solved[1],
                       "base_wrong_disarmed": self.solved[2], "base_wrong_armed": self.solved[3]},
            "dev": {"base_right_disarmed": self.dev[0], "base_right_armed": self.dev[1],
                    "base_wrong_disarmed": self.dev[2], "base_wrong_armed": self.dev[3]},
        })
    }
}
