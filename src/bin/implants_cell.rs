//! implants_cell — the Issue-1256 T1 decode-seam cell bin: runs the four
//! pre-registered cells (G1 silent identity, G1b false fire, G2 fired
//! exactness, G3 composition) at either tier, ending in a BLAKE3-pinned
//! report JSON.
//!
//! Tiers (the SAME cells through the SAME walk code):
//! - `--tier fixture` (default): the lib module's `SyntheticSource` —
//!   deterministic hash logits, no checkpoint. CI-runnable; the report
//!   is byte-deterministic.
//! - `--tier real`: the league checkpoint through the dense hybrid body
//!   (`load_qwen_deltanet_weights_gguf` + `forward_qwen_deltanet`), CPU,
//!   the bench-971 host shape. Same-device scope: the byte-identity
//!   claim is per-device by construction (bf16-class forwards differ
//!   across GPU types — the WSM repo's own disclosure); this box's
//!   device here is the CPU. A ternary checkpoint refuses loud (the
//!   grafts_cell posture).
//!
//! ```bash
//! cargo run --release --features implants_cell --bin implants_cell -- \
//!     --tier fixture --out .benchmarks/i1256_t1_cells
//! cargo run --release --features implants_cell --bin implants_cell -- \
//!     --tier real --model ../riir-train/data/Qwen3.5-0.8B-Base-Q8_0.gguf \
//!     --out .benchmarks/i1256_t1_cells
//! ```
//!
//! Determinism (the grafts G1 law): the report's `payload_blake3` is
//! taken over the COMPACT canonical JSON with the digest field AND the
//! `box_state` line blanked (the box snapshot is live provenance, never
//! a payload fact); the pretty form is written beside it. A re-run over
//! the same checkpoint + fixture set must reproduce the payload digest
//! byte-for-byte.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result, bail};

use riir_infer_core::deltanet::{
    HybridCache, HybridForwardScratch, effective_rotary_dim, forward_qwen_deltanet,
};
use riir_infer_core::gguf_loader::{GgmlType, GgufFile, load_qwen_deltanet_weights_gguf};
use riir_infer_core::implants_cell::{
    ArmsRig, FixtureCodec, LogitsSource, MountPosture, SyntheticCodec, WalkBufs, author_lanes,
    cell_g1_silent_identity, cell_g1b_false_fire, cell_g2_fired_exactness, cell_g3_composition,
    decode_walk,
};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::tokenizer::BpeTokenizer;

const DEFAULT_MODEL: &str = "../riir-train/data/Qwen3.5-0.8B-Base-Q8_0.gguf";
const DEFAULT_OUT: &str = ".benchmarks/i1256_t1_cells";
const ISSUE_TAG: &str = "riir-infer issue 1256 T1";
/// The cache bound (prompts ≤ ~16 tokens by the WINDOW_MAX law + the
/// generation headroom; a fixed bound keeps one reusable cache honest).
const MAX_POSITIONS: usize = 64;
const MAX_NEW: usize = 8;

struct Args {
    tier: String,
    model: PathBuf,
    out: PathBuf,
}

fn parse_args() -> Result<Args> {
    let mut tier = "fixture".to_string();
    let mut model = PathBuf::from(DEFAULT_MODEL);
    let mut out = PathBuf::from(DEFAULT_OUT);
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--tier" => tier = it.next().context("--tier needs a value")?,
            "--model" => model = PathBuf::from(it.next().context("--model needs a value")?),
            "--out" => out = PathBuf::from(it.next().context("--out needs a value")?),
            other => {
                bail!("unknown arg {other} (usage: --tier fixture|real --model <gguf> --out <dir>)")
            }
        }
    }
    if tier != "fixture" && tier != "real" {
        bail!("--tier must be fixture|real, got {tier}");
    }
    Ok(Args { tier, model, out })
}

// ── The real-tier host ───────────────────────────────────────────────────

struct Host {
    config: riir_infer_core::types::Config,
    weights: riir_infer_core::deltanet::QwenDeltaNetWeights,
    rope_freq: RopeFreqTable,
    tok: BpeTokenizer,
    vocab_size: usize,
}

/// Load the host (dense-hybrid only; a ternary file refuses loud — the
/// grafts_cell posture).
fn load_host(model: &Path) -> Result<Host> {
    let probe = GgufFile::open(model).with_context(|| format!("open {}", model.display()))?;
    let is_ternary = probe
        .tensor_info("token_embd.weight")
        .map(|info| matches!(info.ggml_type, GgmlType::Q2_0 | GgmlType::PTQ1_0))
        .unwrap_or(false);
    if is_ternary {
        bail!("implants_cell: ternary checkpoint refused (the cell is the dense hybrid body)");
    }
    let tok = BpeTokenizer::from_gguf(&probe).context("BPE tokenizer from gguf")?;
    let (config, weights) = load_qwen_deltanet_weights_gguf(model)
        .with_context(|| format!("load {}", model.display()))?;
    let rope_freq = RopeFreqTable::new(config.rope_theta, effective_rotary_dim(&config));
    let vocab_size = config.vocab_size;
    Ok(Host {
        config,
        weights,
        rope_freq,
        tok,
        vocab_size,
    })
}

/// The real-tier logits source: one forward per step into its own
/// buffer, copied out (the grafts Runner shape; the walk owns the fold).
struct RealSource<'a> {
    host: &'a Host,
    cache: HybridCache,
    scratch: HybridForwardScratch,
    logits: Vec<f32>,
}

impl<'a> RealSource<'a> {
    fn new(host: &'a Host) -> Self {
        let cache = HybridCache::with_layer_types_bounded(
            &host.config,
            &host.weights.layer_types,
            MAX_POSITIONS,
        );
        let scratch = HybridForwardScratch::new(&host.config);
        Self {
            host,
            cache,
            scratch,
            logits: vec![0.0; host.vocab_size],
        }
    }
}

impl LogitsSource for RealSource<'_> {
    fn vocab(&self) -> usize {
        self.logits.len()
    }

    fn reset(&mut self) {
        self.cache.reset();
    }

    fn step(&mut self, token: usize, pos: usize, out: &mut [f32]) {
        forward_qwen_deltanet(
            &mut self.logits,
            &self.host.weights,
            &mut self.cache,
            token,
            pos,
            &self.host.config,
            &mut self.scratch,
            &self.host.rope_freq,
        );
        out.copy_from_slice(&self.logits);
    }
}

/// The real-tier codec: the checkpoint's own BPE tokenizer (the
/// context-derived digit surface — the bench-971 continuation trick).
struct RealCodec<'a> {
    tok: &'a BpeTokenizer,
}

impl FixtureCodec for RealCodec<'_> {
    fn encode(&self, s: &str) -> Result<Vec<usize>, String> {
        let toks = self.tok.encode(s);
        if toks.is_empty() {
            return Err(format!("encode {s:?} produced no tokens"));
        }
        Ok(toks)
    }

    fn decode_one(&self, t: usize) -> String {
        self.tok.decode(&[t])
    }

    fn digit_continuation(&self, prompt: &str, d: u8) -> Result<usize, String> {
        let base = self.tok.encode(prompt);
        let with = self.tok.encode(&format!("{prompt}{d}"));
        let suffix = with
            .len()
            .checked_sub(base.len())
            .ok_or_else(|| format!("digit {d}: continuation shorter than base"))?;
        if suffix != 1 {
            return Err(format!(
                "digit {d}: continues {prompt:?} with {suffix} tokens (need exactly 1)"
            ));
        }
        Ok(with[with.len() - 1])
    }
}

// ── The report ───────────────────────────────────────────────────────────

#[derive(Clone, serde::Serialize)]
struct CellRow {
    name: &'static str,
    pass: bool,
    counts: Vec<(&'static str, usize)>,
    notes: Vec<String>,
}

impl From<&riir_infer_core::implants_cell::CellResult> for CellRow {
    fn from(c: &riir_infer_core::implants_cell::CellResult) -> Self {
        Self {
            name: c.name,
            pass: c.pass,
            counts: c.counts.clone(),
            notes: c.notes.clone(),
        }
    }
}

#[derive(Clone, serde::Serialize)]
struct Report {
    issue: &'static str,
    tier: String,
    model_path: String,
    model_blake3: String,
    device_scope: String,
    box_state: String,
    max_new: usize,
    cells: Vec<CellRow>,
    payload_blake3: String,
}

fn blake3_file(path: &Path) -> Result<String> {
    let mut f =
        std::fs::File::open(path).with_context(|| format!("open {} for digest", path.display()))?;
    let mut hasher = blake3::Hasher::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        use std::io::Read;
        let n = f.read(&mut buf).context("read for digest")?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex(&hasher.finalize().as_bytes()[..]))
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Best-effort box-state line (the shared-box disclosure posture).
fn box_state_line() -> String {
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);
    let mut line = format!("cpus={cpus} tier-posture=cpu");
    if let Ok(out) = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader",
        ])
        .output()
    {
        let txt = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if !txt.is_empty() {
            line.push_str(&format!(" gpu(busy,other-lane-context)={txt}"));
        }
    } else {
        line.push_str(" gpu=n/a");
    }
    line
}

fn write_report(report: &Report, out_dir: &Path) -> Result<String> {
    std::fs::create_dir_all(out_dir).with_context(|| format!("mkdir {}", out_dir.display()))?;
    // The blanked-field digest law: hash the COMPACT canonical payload
    // with the digest field empty AND the box_state line blanked — the
    // box snapshot is live provenance (GPU util/memory move between runs
    // under sibling load), never a payload fact. The pretty form is
    // written with both filled.
    let mut blanked = report.clone();
    blanked.payload_blake3 = String::new();
    blanked.box_state = String::new();
    let compact = serde_json::to_string(&blanked).context("serialize blanked")?;
    let digest = hex(&blake3::hash(compact.as_bytes()).as_bytes()[..]);
    let mut final_report = report.clone();
    final_report.payload_blake3 = digest.clone();
    let pretty = serde_json::to_string_pretty(&final_report).context("serialize final")?;
    let path = out_dir.join("report.json");
    std::fs::write(&path, pretty).with_context(|| format!("write {}", path.display()))?;
    Ok(digest)
}

// ── Main ─────────────────────────────────────────────────────────────────

fn run() -> Result<ExitCode> {
    let args = parse_args()?;
    let started = std::time::Instant::now();

    let (model_blake3, device_scope) = if args.tier == "real" {
        (
            blake3_file(&args.model).context("digest the checkpoint")?,
            "cpu (dense-hybrid body forward; the byte-identity scope is this device)".to_string(),
        )
    } else {
        ("none".to_string(), "synthetic (no checkpoint)".to_string())
    };
    if args.tier == "real" {
        println!("# model blake3 {model_blake3}");
    }

    // The cells.
    let cells: Vec<riir_infer_core::implants_cell::CellResult> = if args.tier == "real" {
        let host = load_host(&args.model)?;
        println!(
            "# host: qwen35 hybrid dense | layers={} n_embd={} vocab={}",
            host.config.n_layer, host.config.n_embd, host.vocab_size
        );
        let codec = RealCodec { tok: &host.tok };
        let lanes = author_lanes(&codec).map_err(anyhow::Error::msg)?;
        println!(
            "# lanes authored: add {} rules, sub {} rules, table {} rows",
            lanes.add.rules.len(),
            lanes.sub.rules.len(),
            lanes.table.len()
        );
        let rig = ArmsRig::mint(&lanes);
        let mut source = RealSource::new(&host);
        let mut armed_probe = rig
            .new_runtime(MountPosture::Armed)
            .map_err(anyhow::Error::msg)?;
        let probe_toks = codec.encode("2+2=").map_err(anyhow::Error::msg)?;
        let probe_fires = {
            let bufs = &mut WalkBufs::new(source.vocab());
            let out = decode_walk(&mut source, &probe_toks, MAX_NEW, &mut armed_probe, bufs);
            out.fires.len()
        };
        println!("# armed probe on '2+2=': {probe_fires} fire(s)");
        vec![
            cell_g1_silent_identity(&mut source, &codec, &rig, MAX_NEW)
                .map_err(anyhow::Error::msg)?,
            cell_g1b_false_fire(&codec, &rig).map_err(anyhow::Error::msg)?,
            cell_g2_fired_exactness(&mut source, &codec, &rig, MAX_NEW)
                .map_err(anyhow::Error::msg)?,
            cell_g3_composition(&mut source, &codec, &rig, MAX_NEW).map_err(anyhow::Error::msg)?,
        ]
    } else {
        let lanes = author_lanes(&SyntheticCodec).map_err(anyhow::Error::msg)?;
        let rig = ArmsRig::mint(&lanes);
        let mut source = riir_infer_core::implants_cell::SyntheticSource::new();
        vec![
            cell_g1_silent_identity(&mut source, &SyntheticCodec, &rig, MAX_NEW)
                .map_err(anyhow::Error::msg)?,
            cell_g1b_false_fire(&SyntheticCodec, &rig).map_err(anyhow::Error::msg)?,
            cell_g2_fired_exactness(&mut source, &SyntheticCodec, &rig, MAX_NEW)
                .map_err(anyhow::Error::msg)?,
            cell_g3_composition(&mut source, &SyntheticCodec, &rig, MAX_NEW)
                .map_err(anyhow::Error::msg)?,
        ]
    };

    // Verdicts.
    let mut all_pass = true;
    for c in &cells {
        println!(
            "# {} {} | {}",
            if c.pass { "PASS" } else { "FAIL" },
            c.name,
            c.counts
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
        for n in &c.notes {
            println!("#   note: {n}");
        }
        all_pass &= c.pass;
    }

    let report = Report {
        issue: ISSUE_TAG,
        tier: args.tier.clone(),
        model_path: args.model.display().to_string(),
        model_blake3,
        device_scope,
        box_state: box_state_line(),
        max_new: MAX_NEW,
        cells: cells.iter().map(CellRow::from).collect(),
        payload_blake3: String::new(),
    };
    let digest = write_report(&report, &args.out)?;
    println!(
        "# report {} (payload blake3 {digest}) in {:.1}s",
        args.out.display(),
        started.elapsed().as_secs_f64()
    );
    Ok(if all_pass {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(e) => {
            eprintln!("FATAL: {e:#}");
            ExitCode::FAILURE
        }
    }
}
