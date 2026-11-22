//! `qwen38_cq3_capture` — the CQ3 calibration capture sinks (issue 1186 T2b
//! ours half, plan 629). Pure instrumentation state: per-(layer, site)
//! append sinks + the matched-budget ledger. The read-back plumbing lives
//! on the dense forward (`Qwen38WgpuForward::cq3_tap`, the census pattern)
//! so this module never touches the compute client.
//!
//! Files: `<dir>/blk{li:03}_{site}.cq3cap` — fixed 16-byte header (magic
//! `CQ3CAP1`, format v1, d_in) then raw LE f32 rows (`p × d_in` per chunk
//! tap). Row counts are recoverable from file length; the budget is a PIN
//! (overrun fails loud — the matched-budget law, never a silent widening).
//! `<dir>/ledger.json` stamps the run: corpus sha, tokens, chunking,
//! per-file rows, capture config — the T2 gate reads it, not estimates.
//!
//! Disarmed posture: `CaptureSession::arm_from_env` returns `None` without
//! `RIIR_CQ3_CAPTURE=<dir>`; `arm` exists regardless of feature posture
//! (the census pattern — the armed body compiles under `cq3_calibrate`
//! only; the unarmed stub keeps call sites uniform and compiles to
//! nothing behind the always-`None` constructor).

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// Header magic.
pub const CQ3CAP_MAGIC: &[u8; 7] = b"CQ3CAP1";
/// Format version.
pub const CQ3CAP_FORMAT_V1: u32 = 1;

/// The capture site classes — the league cell's scored set, keyed by
/// matrix ROLE (attn q/k/v share their input row; gate/up share theirs).
pub const CAPTURE_SITES: &[&str] = &[
    "attn_qkv_in",
    "attn_o_in",
    "gdn_qkvz_in",
    "gdn_out_in",
    "mlp_gateup_in",
    "mlp_down_in",
];

/// Which capture site a weight matrix reads, by its GGUF name's tensor
/// suffix (the component BEFORE the trailing `.weight` —
/// `blk.{i}.{suffix}.weight`). `None` = not a capture-fed matrix (norms,
/// biases, embeddings, lm_head stay out — the scored set is the 11
/// matrix classes).
pub fn site_for_matrix(name: &str) -> Option<&'static str> {
    let mut parts = name.rsplit('.');
    let _weight = parts.next()?;
    let suffix = parts.next()?;
    match suffix {
        "attn_q" | "attn_k" | "attn_v" => Some("attn_qkv_in"),
        "attn_qkv" | "attn_gate" | "ssm_alpha" | "ssm_beta" => Some("gdn_qkvz_in"),
        "attn_output" => Some("attn_o_in"),
        "ssm_out" => Some("gdn_out_in"),
        "ffn_gate" | "ffn_up" => Some("mlp_gateup_in"),
        "ffn_down" => Some("mlp_down_in"),
        _ => None,
    }
}

/// One per-(layer, site) sink: buffered f32 row appends behind a fixed
/// header, armed lazily on the first tap.
pub struct CaptureSink {
    name: String,
    d_in: usize,
    path: PathBuf,
    writer: Option<BufWriter<File>>,
    rows_written: usize,
    budget_rows: usize,
}

impl CaptureSink {
    pub fn new(name: String, d_in: usize, dir: &Path, budget_rows: usize) -> Self {
        let path = dir.join(format!("{name}.cq3cap"));
        Self {
            name,
            d_in,
            path,
            writer: None,
            rows_written: 0,
            budget_rows,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn rows_written(&self) -> usize {
        self.rows_written
    }

    /// Append `rows.len() / d_in` rows. Loud on shape mismatch or budget
    /// overrun — the ledger's n_cal is a pin.
    pub fn append(&mut self, rows: &[f32]) -> Result<usize, String> {
        assert_eq!(
            rows.len() % self.d_in,
            0,
            "cq3cap {}: rows len {} not a multiple of d_in {}",
            self.name,
            rows.len(),
            self.d_in
        );
        let n_rows = rows.len() / self.d_in;
        if self.rows_written + n_rows > self.budget_rows {
            return Err(format!(
                "cq3cap {}: budget overrun ({cur}+{n_rows} > {budget} rows) — narrow the \
                 capture tokens; never widen silently",
                self.name,
                cur = self.rows_written,
                n_rows = n_rows,
                budget = self.budget_rows,
            ));
        }
        if self.writer.is_none() {
            if let Some(parent) = self.path.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("cq3cap {}: mkdir: {e}", self.name))?;
            }
            let f = File::create(&self.path)
                .map_err(|e| format!("cq3cap {}: create: {e}", self.name))?;
            let mut w = BufWriter::with_capacity(1 << 20, f);
            let mut header = [0u8; 16];
            header[..7].copy_from_slice(CQ3CAP_MAGIC);
            header[8..12].copy_from_slice(&CQ3CAP_FORMAT_V1.to_le_bytes());
            header[12..16].copy_from_slice(&(self.d_in as u32).to_le_bytes());
            w.write_all(&header)
                .map_err(|e| format!("cq3cap {}: header: {e}", self.name))?;
            self.writer = Some(w);
        }
        let w = self.writer.as_mut().expect("writer just armed");
        w.write_all(bytemuck::cast_slice(rows))
            .map_err(|e| format!("cq3cap {}: write: {e}", self.name))?;
        self.rows_written += n_rows;
        Ok(n_rows)
    }

    /// Flush + fsync; returns the file's final row count.
    pub fn finish(&mut self) -> Result<usize, String> {
        let rows = self.rows_written;
        if let Some(mut w) = self.writer.take() {
            w.flush()
                .map_err(|e| format!("cq3cap {}: flush: {e}", self.name))?;
            w.get_ref()
                .sync_all()
                .map_err(|e| format!("cq3cap {}: fsync: {e}", self.name))?;
        }
        Ok(rows)
    }
}

/// The armed capture session: sinks per (layer, site) + the budget. Held
/// on the dense forward as `Option<CaptureSession>`.
pub struct CaptureSession {
    dir: PathBuf,
    budget_rows_per_file: usize,
    /// `[li * CAPTURE_SITES.len() + site_idx]`.
    sinks: Vec<CaptureSink>,
    total_rows: usize,
}

impl CaptureSession {
    /// The constructor: explicit dir + per-file row budget + n_layers +
    /// per-site d_in (in [`CAPTURE_SITES`] order). The driver passes its
    /// own `--dir`/`--tokens` — no env latch anywhere on the capture path
    /// (the forward's taps read only the installed session; unarmed boots
    /// never construct one and stay byte-identical). Sinks are created
    /// unarmed (file opened lazily on first append).
    pub fn with_config(
        dir: PathBuf,
        budget_rows_per_file: usize,
        n_layers: usize,
        d_in_by_site: &[(&'static str, usize)],
    ) -> Self {
        std::fs::create_dir_all(&dir).expect("cq3 capture dir");
        let mut sinks = Vec::with_capacity(n_layers * CAPTURE_SITES.len());
        for li in 0..n_layers {
            for &(site, d_in) in d_in_by_site {
                sinks.push(CaptureSink::new(
                    format!("blk{li:03}_{site}"),
                    d_in,
                    &dir,
                    budget_rows_per_file,
                ));
            }
        }
        Self {
            dir,
            budget_rows_per_file,
            sinks,
            total_rows: 0,
        }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn budget_rows_per_file(&self) -> usize {
        self.budget_rows_per_file
    }

    pub fn total_rows(&self) -> usize {
        self.total_rows
    }

    fn sink_idx(li: usize, site: &str) -> Result<usize, String> {
        let s = CAPTURE_SITES
            .iter()
            .position(|&c| c == site)
            .ok_or_else(|| format!("cq3 capture: unknown site {site}"))?;
        Ok(li * CAPTURE_SITES.len() + s)
    }

    /// Append one site's chunk rows. Caller synced before the read.
    pub fn tap_rows(
        &mut self,
        li: usize,
        site: &'static str,
        rows: usize,
        d_in: usize,
        vals: &[f32],
    ) -> Result<(), String> {
        let idx = Self::sink_idx(li, site)?;
        let expect = rows * d_in;
        if vals.len() != expect {
            return Err(format!(
                "cq3 capture blk{li:03}_{site}: read {} floats, expected {expect} ({rows}×{d_in})",
                vals.len()
            ));
        }
        let n = self.sinks[idx].append(vals)?;
        self.total_rows += n;
        Ok(())
    }

    /// Flush + close every touched sink, write `ledger.json`, return the
    /// per-file (name, rows) summary.
    pub fn finish(&mut self, meta: &CaptureLedgerMeta) -> Result<Vec<(String, usize)>, String> {
        let mut summary = Vec::new();
        for s in self.sinks.iter_mut() {
            let rows = s.finish()?;
            if rows > 0 {
                summary.push((s.name().to_string(), rows));
            }
        }
        let ledger = serde_json::json!({
            "format": CQ3CAP_FORMAT_V1,
            "corpus_blake3": meta.corpus_blake3,
            "corpus_bytes": meta.corpus_bytes,
            "tokens_captured": meta.tokens_captured,
            "chunk": meta.chunk,
            "state_per_chunk": meta.state_per_chunk,
            "window": meta.window,
            "budget_rows_per_file": self.budget_rows_per_file,
            "sites": CAPTURE_SITES,
            "files": summary.iter().map(|(n, r)| serde_json::json!({"name": n, "rows": r})).collect::<Vec<_>>(),
        });
        let path = self.dir.join("ledger.json");
        let mut f = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .map_err(|e| format!("cq3 ledger: open: {e}"))?;
        writeln!(
            f,
            "{}",
            serde_json::to_string_pretty(&ledger).expect("json")
        )
        .map_err(|e| format!("cq3 ledger: write: {e}"))?;
        f.sync_all()
            .map_err(|e| format!("cq3 ledger: fsync: {e}"))?;
        Ok(summary)
    }
}

/// The matched-budget ledger's provenance block (the driver fills it).
/// Hash law: this repo pins with BLAKE3 — the corpora carry plan-629
/// sha256 pins verified by the opponent prep script; the ours-side
/// ledger's own identity field is the corpus BLAKE3.
pub struct CaptureLedgerMeta {
    pub corpus_blake3: String,
    pub corpus_bytes: u64,
    pub tokens_captured: usize,
    pub chunk: usize,
    /// "fresh" | "carry" — pinned per run, never implied.
    pub state_per_chunk: &'static str,
    /// "head" | "tail" — WHICH slice of the corpus the budget read.
    /// The calibration-integrity law (issue 1186, 2026-11-22): corpora
    /// with a shared head make every head-window capture byte-identical —
    /// equal sample counts are not equal information; the window is part
    /// of the ledger's identity, never implied.
    pub window: &'static str,
}

/// The budget window's token slice: head = `[0, budget)`, tail =
/// `[total − budget, total)` (the continuation region of a
/// prefix-extended corpus). Refuses a window larger than the corpus and
/// any spelling but head/tail — the driver's `--window` maps here.
pub fn window_slice(total: usize, budget: usize, window: &str) -> Result<(usize, usize), String> {
    match window {
        "head" => Ok((0, budget)),
        "tail" => {
            if budget > total {
                return Err(format!(
                    "cq3 window: budget {budget} exceeds the corpus's {total} tokens"
                ));
            }
            Ok((total - budget, total))
        }
        other => Err(format!("cq3 window: unknown {other:?} (head | tail)")),
    }
}
