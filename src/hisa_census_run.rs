//! The HiSA census sweep driver (riir-infer Issue 1244 T2.1b) — the
//! machinery Issue 1244 T2.2 runs through. REPORT-ONLY instrument: an
//! offline diagnostic beside [`crate::hisa_plan`]'s closed-form prior and
//! [`crate::hisa_census`]'s score-path arm. Zero serving-path code; the
//! bin front is `src/bin/hisa_census.rs` (`required-features =
//! ["hisa_census"]` — a default build skips it loudly, the green-zero
//! law).
//!
//! # The experiment (per attention layer)
//!
//! Over ONE frozen teacher-forced token read (positions `0..N`, gold =
//! the next corpus token, `N` scored steps):
//!
//! 1. **Base pass** — no arm, incumbent logits. Per step: f64 NLL of the
//!    gold + the f32 top-1. These are the shared reference every cell's
//!    Δ-NLL and flip count reads against.
//! 2. **Observe pass per layer** — [`crate::hisa_census::HisaMode::Observe`]
//!    (incumbent-identical scores): the E_real band-mass base for that
//!    layer, every band of the posture in one pass.
//! 3. **Null snapshot pass** — the null document (a distinct same-length
//!    corpus slice) through the incumbent forward; each probed layer's
//!    `cache.key[..N·kvd]` post-RoPE f32 rows become the
//!    [`crate::hisa_census::HisaMode::Replace`] arm's null keys
//!    (`t_null = N` by construction — the arm's `t_null >= t_n` law).
//!    ONE pass covers every layer.
//! 4. **Cell passes** — per (layer, 16-feature chunk, dyadic log-Δ band):
//!    a [`crate::hisa_census::HisaMode::Remove`] pass (the suppression
//!    posture) and a [`crate::hisa_census::HisaMode::Replace`] pass (the
//!    E_null noise floor). The arm is armed ONLY for its probed layer;
//!    every other layer runs the incumbent path bit-identically.
//!
//! Cell reads: pooled golden Δ-NLL + argmax flips vs the base (f64
//! positive-summand folds, Issue 940 — never an f32 softmax), the arm's
//! band energies (`E_remove/E_observe`, `E_replace/E_observe` — the
//! paper's suppression-vs-noise read), in-band sample counts, and
//! `swapped_skipped` (a nonzero count is a driver bug and is disclosed
//! LOUD in the report, never folded).
//!
//! # Grids
//!
//! Chunks tile the head's rotary range in 16-feature units (the T2.1
//! "16-feature pair chunk" law; a non-multiple rotary leaves a disclosed
//! tail chunk). Bands are the dyadic log-Δ family `{0},{1},[2,4),…`
//! capped at the read length — distances only reach `N−1`, so bands
//! beyond the cap are structurally absent (the report discloses the cap;
//! nothing vacuous is ever run or folded).
//!
//! # Ledger + resume (the i1241 sweep pattern)
//!
//! State dir: `manifest.json` (a blake3 digest over the model identity +
//! config signature + read tokens + grids + the cell table) and one row
//! file per committed cell (`rows/<layer>_<chunk>_<band>.json`, written
//! tmp+rename — the rename is the commit point). `--resume` verifies the
//! manifest digest (a foreign digest refuses loud), skips committed rows,
//! and continues at the cursor; kill/resume is bit-identical by
//! construction (per-cell rows carry the full result).
//!
//! # Sync boundary
//!
//! None — an offline diagnostic. Deterministic: fixed grids, ascending
//! folds, f64 NLL math, no RNG, no wall clock in any stored value (the
//! per-pass wall times are stdout progress only, never report content).

use std::io::Write as _;
use std::path::{Path, PathBuf};

use crate::deltanet::forward::HybridCache;
use crate::deltanet::forward::HybridForwardScratch;
use crate::deltanet::ternary_forward::forward_qwen_deltanet_ternary;
use crate::deltanet::ternary_weights::QwenDeltaNetTernaryWeights;
use crate::hisa_census::HisaCensusArm;
use crate::rope::RopeFreqTable;
use crate::types::Config;

/// The report file's first bytes — a reader refuses anything else (the
/// sink_audit payload-version law).
pub const REPORT_MAGIC: &[u8] = b"HISACEN1\x01";

/// Default frozen-read length (tokens; the read's max distance is N−1).
pub const DEFAULT_READ_LEN: usize = 256;

/// The chunk law: 16-feature chunks tile the rotary range (T2.1).
pub const CHUNK_FEATURES: usize = 16;

// ── grids ──────────────────────────────────────────────────────────────

/// The dyadic log-Δ distance band family `{0},{1},[2,4),[4,8),…` capped
/// at `read_len` (distances run `0..=read_len−1`; bands are half-open).
/// The boundary sequence is `0,1,2,4,8,…` — band k covers
/// `[2^k − 1, 2^(k+1) − 1)` (the T2.1 band text, computed — never a
/// second table).
pub fn band_grid(read_len: usize) -> Vec<(usize, usize)> {
    let cap = read_len.max(1);
    let mut bands = Vec::new();
    let mut lo = 0usize;
    let mut next = 1usize;
    while lo < cap {
        let hi = next.min(cap);
        bands.push((lo, hi));
        lo = next;
        next = if next == 1 { 2 } else { next * 2 };
    }
    bands
}

/// 16-feature chunks tiling the head's rotary range `[0, rotary_dim)`.
/// A non-multiple rotary leaves a disclosed tail chunk (never hardcoded
/// geometry).
pub fn chunk_grid(rotary_dim: usize) -> Vec<(usize, usize)> {
    let mut chunks = Vec::new();
    let mut lo = 0usize;
    while lo < rotary_dim {
        let hi = (lo + CHUNK_FEATURES).min(rotary_dim);
        chunks.push((lo, hi));
        lo = hi;
    }
    chunks
}

/// One census cell: probed layer (model layer index), the chunk's feature
/// range within one head, the target distance band.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct CellSpec {
    pub layer: usize,
    pub chunk: (usize, usize),
    pub band: (usize, usize),
}

/// The full cell table: `attn_layers × chunks × bands`, in the canonical
/// sorted order the report and ledger key on.
pub fn build_cell_table(
    attn_layers: &[usize],
    chunks: &[(usize, usize)],
    bands: &[(usize, usize)],
) -> Vec<CellSpec> {
    let mut layers: Vec<usize> = attn_layers.to_vec();
    layers.sort_unstable();
    let mut table = Vec::with_capacity(layers.len() * chunks.len() * bands.len());
    for &layer in &layers {
        for &chunk in chunks {
            for &band in bands {
                table.push(CellSpec { layer, chunk, band });
            }
        }
    }
    table
}

/// Apply the CLI filters and the `limit` cap. `layer_filter` holds MODEL
/// layer indices; `chunk_filter`/`band_filter` hold POSITIONS in the
/// respective grids (stable, grid-defined); `None` = all. A filter value
/// outside its grid refuses loud — a silently-empty run is a green zero.
pub fn filter_cell_table(
    table: &[CellSpec],
    layer_filter: Option<&[usize]>,
    chunk_filter: Option<&[usize]>,
    band_filter: Option<&[usize]>,
    limit: Option<usize>,
    chunks: &[(usize, usize)],
    bands: &[(usize, usize)],
) -> Result<Vec<CellSpec>, String> {
    // The filters are validated UP FRONT against their grids — an
    // out-of-range filter value must refuse loud, never match nothing
    // and shrink to a silent (green-zero) empty run.
    if let Some(sel) = chunk_filter {
        for &i in sel {
            if i >= chunks.len() {
                return Err(format!(
                    "chunk filter index {i} out of grid ({} chunks)",
                    chunks.len()
                ));
            }
        }
    }
    if let Some(sel) = band_filter {
        for &i in sel {
            if i >= bands.len() {
                return Err(format!(
                    "band filter index {i} out of grid ({} bands)",
                    bands.len()
                ));
            }
        }
    }
    if let Some(sel) = layer_filter {
        let have: std::collections::HashSet<usize> = table.iter().map(|c| c.layer).collect();
        for &l in sel {
            if !have.contains(&l) {
                return Err(format!(
                    "layer filter {l} has no cells in the table (layers {have:?})"
                ));
            }
        }
    }
    let mut out = Vec::new();
    for c in table {
        if let Some(sel) = layer_filter
            && !sel.contains(&c.layer)
        {
            continue;
        }
        if let Some(sel) = chunk_filter {
            let idx = chunks
                .iter()
                .position(|&x| x == c.chunk)
                .ok_or_else(|| format!("chunk {:?} not in the grid", c.chunk))?;
            if !sel.contains(&idx) {
                continue;
            }
        }
        if let Some(sel) = band_filter {
            let idx = bands
                .iter()
                .position(|&x| x == c.band)
                .ok_or_else(|| format!("band {:?} not in the grid", c.band))?;
            if !sel.contains(&idx) {
                continue;
            }
        }
        out.push(*c);
    }
    if let Some(l) = limit {
        out.truncate(l);
    }
    Ok(out)
}

// ── the frozen read's per-step statistics ──────────────────────────────

/// f64 positive-summand NLL of `gold` under `logits` (Issue 940: the
/// softmax runs entirely in f64, max-shifted, ascending index fold) and
/// the f32 top-1 (FIRST index on ties — the strict-`>` scan pins it).
pub fn step_stats(logits: &[f32], gold: usize) -> (f64, usize) {
    let mut max = f64::NEG_INFINITY;
    for &x in logits {
        let x = x as f64;
        if x > max {
            max = x;
        }
    }
    let mut sum = 0.0f64;
    for &x in logits {
        sum += ((x as f64) - max).exp();
    }
    let lse = max + sum.ln();
    let gold_x = logits
        .get(gold)
        .copied()
        .unwrap_or_else(|| panic!("gold index {gold} outside the logits window"));
    let nll = lse - (gold_x as f64);
    // f32 top-1, FIRST index on ties (strict `>` scan — `max_by` returns
    // the LAST maximal element, which would pin the wrong tie rule).
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in logits.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    (nll, best)
}

/// One read pass's outcome: per-step f64 NLLs, per-step f32 top-1, and
/// the (possibly updated) census arm handed back.
pub type ReadPassOutcome = (Vec<f64>, Vec<usize>, Option<HisaCensusArm>);

/// One pass over the frozen read. `arm_slot` arms the census arm (the
/// dispatch takes it ONLY for `arm.layer`); the (possibly updated) arm is
/// handed back so the driver can read its accumulators. `cache` is reset
/// in place per pass (the Issue-838 H1 reset — no per-pass reallocation).
#[allow(clippy::too_many_arguments)]
pub fn run_read_pass(
    weights: &QwenDeltaNetTernaryWeights,
    config: &Config,
    read: &[usize],
    golds: &[usize],
    rope_freq: &RopeFreqTable,
    cache: &mut HybridCache,
    scratch: &mut HybridForwardScratch,
    arm_slot: Option<HisaCensusArm>,
) -> Result<ReadPassOutcome, String> {
    if read.len() != golds.len() {
        return Err(format!(
            "read pass: read len {} != golds len {} — the driver sliced them wrong",
            read.len(),
            golds.len()
        ));
    }
    #[cfg(feature = "scale_invariant_attn")]
    if scratch.attention.scale_invariant.is_some() {
        return Err(
            "read pass: a scale-invariant session is armed — undefined under the HiSA census"
                .into(),
        );
    }
    scratch.attention.hisa_census = arm_slot;
    cache.reset();
    let mut logits = vec![0.0f32; config.vocab_size];
    let mut nlls = Vec::with_capacity(read.len());
    let mut top1 = Vec::with_capacity(read.len());
    for (pos, (&token, &gold)) in read.iter().zip(golds.iter()).enumerate() {
        forward_qwen_deltanet_ternary(
            &mut logits,
            weights,
            cache,
            token,
            pos,
            config,
            scratch,
            rope_freq,
        );
        let (nll, am) = step_stats(&logits, gold);
        nlls.push(nll);
        top1.push(am);
    }
    let arm = scratch.attention.hisa_census.take();
    Ok((nlls, top1, arm))
}

/// The per-step mean of an NLL vec (empty refuses — a zero-step read is a
/// driver bug, not a zero effect).
pub fn mean(nlls: &[f64]) -> f64 {
    assert!(!nlls.is_empty(), "mean over an empty NLL vec");
    nlls.iter().sum::<f64>() / nlls.len() as f64
}

/// Flip count: positions where the perturbed top-1 differs from the base
/// top-1. THE pinned definition (the T2.1 row).
pub fn flip_count(base_top1: &[usize], perturbed_top1: &[usize]) -> u64 {
    assert_eq!(base_top1.len(), perturbed_top1.len(), "flip arity");
    base_top1
        .iter()
        .zip(perturbed_top1)
        .filter(|(b, p)| b != p)
        .count() as u64
}

// ── the cell result + canonical report ─────────────────────────────────

/// One committed census cell — the report's row unit.
#[derive(Clone, Debug)]
pub struct CellResult {
    pub spec: CellSpec,
    /// E_real base mass for the cell's band (from the layer's Observe pass).
    pub e_observe: f64,
    pub count_observe: u64,
    /// Remove (suppression) posture: in-band pooled mass + samples.
    pub e_remove: f64,
    pub count_remove: u64,
    /// Replace (E_null noise floor) posture.
    pub e_replace: f64,
    pub count_replace: u64,
    /// Replace-only tripwire — expect 0 (the driver guarantees
    /// `t_null >= t_n`); nonzero is a driver bug, disclosed LOUD.
    pub swapped_skipped: u64,
    /// Pooled per-step-mean NLLs (f64 folds).
    pub base_nll_mean: f64,
    pub nll_remove_mean: f64,
    pub nll_replace_mean: f64,
    /// `nll_x_mean − base_nll_mean` (the effect size; positive = worse).
    pub d_nll_remove: f64,
    pub d_nll_replace: f64,
    /// Top-1 flips vs the base over the read.
    pub flips_remove: u64,
    pub flips_replace: u64,
    /// Scored steps (the read length).
    pub steps: u64,
}

fn fmt_f(x: f64) -> String {
    if x.is_finite() {
        format!("{x:.9e}")
    } else {
        format!("{x}")
    }
}

/// The canonical report bytes (fixed precision, sorted rows) — the digest
/// is taken over EXACTLY these bytes; the blake3 line is appended after.
pub fn canonical_report_bytes(id: &SweepIdentity, rows: &[CellResult]) -> Vec<u8> {
    let mut sorted: Vec<&CellResult> = rows.iter().collect();
    sorted.sort_by_key(|r| r.spec);
    let mut s = String::new();
    s.push_str(&format!(
        "model={} read_len={} null_offset={} steps={}\n",
        id.model, id.read_len, id.null_offset, id.read_len
    ));
    s.push_str(&format!(
        "attn_layers={:?}\nchunks={:?}\nbands={:?}\n",
        id.attn_layers, id.chunks, id.bands
    ));
    s.push_str(&format!(
        "cells={} swapped_skipped_total={}\n",
        sorted.len(),
        sorted.iter().map(|r| r.swapped_skipped).sum::<u64>()
    ));
    for r in sorted {
        s.push_str(&format!(
            "cell layer={} chunk=({},{}) band=({},{}) e_obs={} e_rem={} e_rep={} c_obs={} c_rem={} c_rep={} d_nll_rem={} d_nll_rep={} nll_base={} flips_rem={} flips_rep={} steps={} skipped={}\n",
            r.spec.layer,
            r.spec.chunk.0,
            r.spec.chunk.1,
            r.spec.band.0,
            r.spec.band.1,
            fmt_f(r.e_observe),
            fmt_f(r.e_remove),
            fmt_f(r.e_replace),
            r.count_observe,
            r.count_remove,
            r.count_replace,
            fmt_f(r.d_nll_remove),
            fmt_f(r.d_nll_replace),
            fmt_f(r.base_nll_mean),
            r.flips_remove,
            r.flips_replace,
            r.steps,
            r.swapped_skipped,
        ));
    }
    let mut bytes = REPORT_MAGIC.to_vec();
    bytes.extend_from_slice(s.as_bytes());
    bytes
}

/// The full report: canonical bytes + the blake3 line over them.
pub fn render_report(canonical: &[u8]) -> Vec<u8> {
    let mut out = canonical.to_vec();
    let hex = blake3::hash(canonical).to_hex();
    out.extend_from_slice(format!("blake3={hex}\n").as_bytes());
    out
}

// ── the ledger (the i1241 sweep pattern) ───────────────────────────────

/// The identity of one sweep — the fields both the resume manifest and
/// the canonical report key on. The bin builds ONE and hands it to both.
#[derive(Clone)]
pub struct SweepIdentity<'a> {
    pub model: &'a str,
    pub model_len: u64,
    pub config_sig: &'a str,
    pub tokens_digest: &'a str,
    pub read_len: usize,
    pub null_offset: usize,
    pub attn_layers: &'a [usize],
    pub chunks: &'a [(usize, usize)],
    pub bands: &'a [(usize, usize)],
}

/// The resume manifest digest: blake3 over the canonical identity string.
/// A foreign digest refuses loud — never silently re-run someone else's
/// sweep.
pub fn manifest_digest(id: &SweepIdentity, table: &[CellSpec]) -> String {
    let mut s = String::new();
    s.push_str(&format!("model={} bytes={}\n", id.model, id.model_len));
    s.push_str(&format!("config={}\n", id.config_sig));
    s.push_str(&format!("tokens={}\n", id.tokens_digest));
    s.push_str(&format!(
        "read_len={} null_offset={}\n",
        id.read_len, id.null_offset
    ));
    s.push_str(&format!(
        "attn_layers={:?}\nchunks={:?}\nbands={:?}\n",
        id.attn_layers, id.chunks, id.bands
    ));
    for c in table {
        s.push_str(&format!(
            "cell {} ({},{}),({},{})\n",
            c.layer, c.chunk.0, c.chunk.1, c.band.0, c.band.1
        ));
    }
    blake3::hash(s.as_bytes()).to_hex().to_string()
}

/// The resumable sweep ledger: `manifest.json` + one row file per
/// committed cell (`rows/<layer>_<chunk>_<band>.json`, tmp+rename — the
/// rename is the commit point).
pub struct CensusLedger {
    dir: PathBuf,
    rows_dir: PathBuf,
    total: usize,
}

fn row_file_name(spec: &CellSpec) -> String {
    format!("{}_{}_{}.json", spec.layer, spec.chunk.0, spec.band.0)
}

/// Recover a cell spec from a committed row's canonical line (the name
/// carries only the lo ends; the LINE carries both range ends — the
/// resume cursor matches on the full spec).
pub fn spec_from_row_line(line: &str) -> Option<CellSpec> {
    let mut layer = None;
    let mut chunk = None;
    let mut band = None;
    for tok in line.split_whitespace() {
        if let Some(v) = tok.strip_prefix("layer=") {
            layer = v.parse().ok();
        } else if let Some(v) = tok.strip_prefix("chunk=") {
            let v = v.trim_matches(|c| c == '(' || c == ')');
            let mut it = v.split(',');
            let lo = it.next()?.parse().ok()?;
            let hi = it.next()?.parse().ok()?;
            chunk = Some((lo, hi));
        } else if let Some(v) = tok.strip_prefix("band=") {
            let v = v.trim_matches(|c| c == '(' || c == ')');
            let mut it = v.split(',');
            let lo = it.next()?.parse().ok()?;
            let hi = it.next()?.parse().ok()?;
            band = Some((lo, hi));
        }
    }
    Some(CellSpec {
        layer: layer?,
        chunk: chunk?,
        band: band?,
    })
}

/// The ledger's committed cell specs (the resume cursor): every row file
/// parsed through its canonical line. An unreadable row is LOUD — a
/// silent skip would re-run or worse, silently drop cells from the
/// report.
pub fn committed_specs(ledger: &CensusLedger) -> Result<Vec<CellSpec>, String> {
    let mut out = Vec::new();
    let it = std::fs::read_dir(ledger.rows_dir())
        .map_err(|e| format!("scan {}: {e}", ledger.rows_dir().display()))?;
    for e in it.flatten() {
        let path = e.path();
        if path.extension().map(|x| x == "json").unwrap_or(false) {
            let bytes =
                std::fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
            let text = String::from_utf8_lossy(&bytes);
            let line = text
                .lines()
                .find(|l| l.starts_with("cell "))
                .ok_or_else(|| format!("row {} has no cell line", path.display()))?;
            let spec = spec_from_row_line(line)
                .ok_or_else(|| format!("row {} has an UNPARSEABLE cell line", path.display()))?;
            out.push(spec);
        }
    }
    Ok(out)
}

impl CensusLedger {
    /// Create a FRESH ledger; refuses a non-empty state dir loud (name
    /// `--resume` or a fresh dir — never silently re-run or overwrite).
    pub fn create(dir: &Path, digest: &str, total: usize) -> Result<Self, String> {
        let rows_dir = dir.join("rows");
        if dir.exists() {
            let occupied = dir
                .read_dir()
                .map(|it| it.filter_map(|e| e.ok()).count())
                .unwrap_or(0);
            if occupied > 0 {
                return Err(format!(
                    "state dir {} is not empty ({} entries) — pass --resume or use a fresh dir",
                    dir.display(),
                    occupied
                ));
            }
        }
        std::fs::create_dir_all(&rows_dir)
            .map_err(|e| format!("create {}: {e}", rows_dir.display()))?;
        let manifest = format!("digest={digest}\ntotal={total}\n");
        write_atomic(&dir.join("manifest.json"), manifest.as_bytes())
            .map_err(|e| format!("write manifest: {e}"))?;
        Ok(Self {
            dir: dir.to_path_buf(),
            rows_dir,
            total,
        })
    }

    /// Resume: verify the manifest digest (foreign refuses loud), then
    /// count committed rows. Returns (ledger, committed count).
    pub fn open(dir: &Path, digest: &str, total: usize) -> Result<Self, String> {
        let manifest_path = dir.join("manifest.json");
        let bytes = std::fs::read(&manifest_path)
            .map_err(|e| format!("read {}: {e}", manifest_path.display()))?;
        let text = String::from_utf8_lossy(&bytes);
        let stored = text
            .lines()
            .find_map(|l| l.strip_prefix("digest="))
            .map(str::trim)
            .ok_or_else(|| format!("manifest {} has no digest line", manifest_path.display()))?;
        if stored != digest {
            return Err(format!(
                "state dir {} holds a FOREIGN sweep (digest {stored} != {digest}) — refusing loud",
                dir.display()
            ));
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            rows_dir: dir.join("rows"),
            total,
        })
    }

    pub fn rows_dir(&self) -> &Path {
        &self.rows_dir
    }

    pub fn total(&self) -> usize {
        self.total
    }

    /// Commit one cell result (tmp+rename — the rename is the commit
    /// point). A row that already exists is an idempotent no-op only if
    /// its bytes match; a mismatch refuses loud.
    pub fn commit(&self, result: &CellResult) -> Result<(), String> {
        let name = row_file_name(&result.spec);
        let path = self.rows_dir.join(&name);
        let bytes = row_bytes(result);
        if path.exists() {
            let old = std::fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
            if old == bytes {
                return Ok(());
            }
            return Err(format!(
                "row {} already committed with DIFFERENT bytes — refusing loud",
                path.display()
            ));
        }
        write_atomic(&path, &bytes).map_err(|e| format!("commit {}: {e}", path.display()))
    }

    /// How many cells are committed (the resume cursor).
    pub fn committed(&self) -> usize {
        self.rows_dir
            .read_dir()
            .map(|it| {
                it.filter_map(|e| e.ok())
                    .filter(|e| e.path().extension().map(|x| x == "json").unwrap_or(false))
                    .count()
            })
            .unwrap_or(0)
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

/// One row's canonical bytes (the ledger AND the report draw from the
/// same fixed-precision form).
fn row_bytes(r: &CellResult) -> Vec<u8> {
    let line = format!(
        "cell layer={} chunk=({},{}) band=({},{}) e_obs={} e_rem={} e_rep={} c_obs={} c_rem={} c_rep={} d_nll_rem={} d_nll_rep={} nll_base={} flips_rem={} flips_rep={} steps={} skipped={}\n",
        r.spec.layer,
        r.spec.chunk.0,
        r.spec.chunk.1,
        r.spec.band.0,
        r.spec.band.1,
        fmt_f(r.e_observe),
        fmt_f(r.e_remove),
        fmt_f(r.e_replace),
        r.count_observe,
        r.count_remove,
        r.count_replace,
        fmt_f(r.d_nll_remove),
        fmt_f(r.d_nll_replace),
        fmt_f(r.base_nll_mean),
        r.flips_remove,
        r.flips_replace,
        r.steps,
        r.swapped_skipped,
    );
    line.into_bytes()
}

fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("json.tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)
}

// ── tests (hermetic — no model file, no forward) ───────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn band_grid_is_the_dyadic_issue_family() {
        // The T2.1 band text: {0},{1},[2,4),[4,8),… capped at the read.
        // Boundaries 0,1,2,4,8 — band k = [2^k − 1, 2^(k+1) − 1).
        assert_eq!(band_grid(8), vec![(0, 1), (1, 2), (2, 4), (4, 8)]);
        assert_eq!(band_grid(8), vec![(0, 1), (1, 2), (2, 4), (4, 8)]);
        assert_eq!(
            band_grid(256),
            vec![
                (0, 1),
                (1, 2),
                (2, 4),
                (4, 8),
                (8, 16),
                (16, 32),
                (32, 64),
                (64, 128),
                (128, 256)
            ]
        );
        // A non-power-of-two read caps the last band at the read length.
        assert_eq!(band_grid(24).last(), Some(&(16, 24)));
        // Degenerate read: one band covering the single distance.
        assert_eq!(band_grid(1), vec![(0, 1)]);
    }

    #[test]
    fn chunk_grid_tiles_rotary_in_16feat_units() {
        assert_eq!(chunk_grid(64), vec![(0, 16), (16, 32), (32, 48), (48, 64)]);
        // A non-multiple rotary leaves a disclosed tail chunk.
        assert_eq!(chunk_grid(20), vec![(0, 16), (16, 20)]);
        assert_eq!(chunk_grid(16), vec![(0, 16)]);
    }

    #[test]
    fn cell_table_is_the_sorted_cross_product() {
        let layers = [7, 3];
        let chunks = chunk_grid(32);
        let bands = band_grid(4); // [(0,1),(1,2),(2,4)] — 3 dyadic bands
        let table = build_cell_table(&layers, &chunks, &bands);
        assert_eq!(table.len(), 2 * 2 * 3);
        // Canonical order: layers ascending, then chunks, then bands.
        assert_eq!(table[0].layer, 3);
        assert_eq!(table[0].chunk, (0, 16));
        assert_eq!(table[0].band, (0, 1));
        assert_eq!(table.last().unwrap().layer, 7);
        // Already-sorted input is a fixpoint (the Ord derive agrees).
        let mut sorted = table.clone();
        sorted.sort();
        assert_eq!(table, sorted);
    }

    #[test]
    fn step_stats_matches_a_hand_computed_value() {
        // logits [ln 2, ln 4, 0]: softmax = (2,4,1)/7. gold 1 → NLL = ln(7/4).
        let logits = [2.0f32.ln() as f32, 4.0f32.ln() as f32, 0.0];
        let (nll, am) = step_stats(&logits, 1);
        let want = (7.0f64 / 4.0).ln();
        // The logits are f32 STORAGE (ln 2 / ln 4 rounded) — the f64 hand
        // value is exact, the fixture is not; 1e-6 is the storage floor.
        assert!((nll - want).abs() < 1e-6, "nll {nll} vs {want}");
        assert_eq!(am, 1);
        // Tie on top-1 → FIRST index (the pinned tie rule).
        let tie = [1.0f32, 1.0, 0.5];
        let (_, am) = step_stats(&tie, 0);
        assert_eq!(am, 0);
    }

    #[test]
    fn flip_count_counts_top1_divergence_only() {
        assert_eq!(flip_count(&[1, 2, 3], &[1, 2, 3]), 0);
        assert_eq!(flip_count(&[1, 2, 3], &[1, 9, 3]), 1);
        assert_eq!(flip_count(&[5, 5], &[7, 5]), 1);
    }

    #[test]
    fn filter_cell_table_filters_and_refuses() {
        let chunks = chunk_grid(32);
        let bands = band_grid(4);
        let table = build_cell_table(&[3, 7], &chunks, &bands);
        // Layer filter only.
        let sel = filter_cell_table(&table, Some(&[7]), None, None, None, &chunks, &bands)
            .expect("layer filter");
        assert_eq!(sel.len(), table.len() / 2);
        assert!(sel.iter().all(|c| c.layer == 7));
        // Band-position filter + limit.
        let sel = filter_cell_table(&table, None, None, Some(&[1]), Some(3), &chunks, &bands)
            .expect("band filter");
        assert!(sel.len() <= 3);
        assert!(sel.iter().all(|c| c.band == (1, 2)));
        // A grid position out of range refuses (never a silent empty run).
        assert!(
            filter_cell_table(&table, Some(&[3]), None, Some(&[99]), None, &chunks, &bands)
                .is_err()
        );
    }

    #[test]
    fn report_digest_is_stable_and_magic_first() {
        let mk = |band: (usize, usize), dnl: f64| CellResult {
            spec: CellSpec {
                layer: 3,
                chunk: (0, 16),
                band,
            },
            e_observe: 0.5,
            count_observe: 100,
            e_remove: 0.1,
            count_remove: 100,
            e_replace: 0.4,
            count_replace: 100,
            swapped_skipped: 0,
            base_nll_mean: 2.0,
            nll_remove_mean: 2.0 + dnl,
            nll_replace_mean: 2.0 + dnl * 0.5,
            d_nll_remove: dnl,
            d_nll_replace: dnl * 0.5,
            flips_remove: 3,
            flips_replace: 1,
            steps: 256,
        };
        let rows = vec![mk((2, 4), 0.125), mk((4, 8), 0.0)];
        let id = SweepIdentity {
            model: "m.gguf",
            model_len: 0,
            config_sig: "",
            tokens_digest: "",
            read_len: 256,
            null_offset: 256,
            attn_layers: &[3],
            chunks: &[(0, 16)],
            bands: &[(2, 4), (4, 8)],
        };
        let a = canonical_report_bytes(&id, &rows);
        let b = canonical_report_bytes(&id, &rows);
        assert_eq!(a, b);
        assert!(a.starts_with(REPORT_MAGIC));
        let rep = render_report(&a);
        assert!(rep.starts_with(REPORT_MAGIC));
        let hex = blake3::hash(&a).to_hex().to_string();
        let tail = String::from_utf8_lossy(&rep);
        assert!(tail.contains(&format!("blake3={hex}")));
        // Row ORDER does not move the digest (canonicalization sorts by
        // the distinct specs).
        let rows_rev: Vec<CellResult> = rows.iter().rev().cloned().collect();
        let c = canonical_report_bytes(&id, &rows_rev);
        assert_eq!(a, c);
    }

    #[test]
    fn manifest_digest_binds_every_identity_input() {
        let t1 = vec![CellSpec {
            layer: 3,
            chunk: (0, 16),
            band: (0, 1),
        }];
        let t2 = vec![CellSpec {
            layer: 3,
            chunk: (0, 16),
            band: (1, 2),
        }];
        let id = SweepIdentity {
            model: "m.gguf",
            model_len: 123,
            config_sig: "sig",
            tokens_digest: "tok",
            read_len: 256,
            null_offset: 512,
            attn_layers: &[3],
            chunks: &[(0, 16)],
            bands: &[(0, 1)],
        };
        let mut id_other_model = id.clone();
        id_other_model.model = "other.gguf";
        assert_eq!(manifest_digest(&id, &t1), manifest_digest(&id, &t1));
        assert_ne!(
            manifest_digest(&id, &t1),
            manifest_digest(&id, &t2),
            "a different cell table must re-digest"
        );
        assert_ne!(
            manifest_digest(&id, &t1),
            manifest_digest(&id_other_model, &t1)
        );
    }

    #[test]
    fn ledger_commit_resume_roundtrip_is_bit_identical() {
        let dir = std::env::temp_dir().join(format!("hisa_ledger_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let table = vec![
            CellSpec {
                layer: 3,
                chunk: (0, 16),
                band: (0, 1),
            },
            CellSpec {
                layer: 3,
                chunk: (0, 16),
                band: (1, 2),
            },
        ];
        let digest = manifest_digest(
            &SweepIdentity {
                model: "m.gguf",
                model_len: 1,
                config_sig: "s",
                tokens_digest: "t",
                read_len: 4,
                null_offset: 8,
                attn_layers: &[3],
                chunks: &[(0, 16)],
                bands: &[(0, 1), (1, 2)],
            },
            &table,
        );
        let mk = |band: (usize, usize)| CellResult {
            spec: CellSpec {
                layer: 3,
                chunk: (0, 16),
                band,
            },
            e_observe: 0.5,
            count_observe: 8,
            e_remove: 0.2,
            count_remove: 8,
            e_replace: 0.45,
            count_replace: 8,
            swapped_skipped: 0,
            base_nll_mean: 2.0,
            nll_remove_mean: 2.1,
            nll_replace_mean: 2.05,
            d_nll_remove: 0.1,
            d_nll_replace: 0.05,
            flips_remove: 1,
            flips_replace: 0,
            steps: 4,
        };
        // Fresh create refuses a second create over the same dir.
        let led = CensusLedger::create(&dir, &digest, 2).expect("create");
        assert_eq!(led.committed(), 0);
        assert!(
            CensusLedger::create(&dir, &digest, 2).is_err(),
            "non-empty dir refuses"
        );
        led.commit(&mk((0, 1))).expect("commit 1");
        assert_eq!(led.committed(), 1);
        // Idempotent byte-identical re-commit is a no-op.
        led.commit(&mk((0, 1))).expect("idempotent re-commit");
        assert_eq!(led.committed(), 1);
        drop(led);
        // Resume: foreign digest refuses; matching digest counts rows.
        assert!(CensusLedger::open(&dir, "foreign", 2).is_err());
        let led2 = CensusLedger::open(&dir, &digest, 2).expect("resume");
        assert_eq!(led2.committed(), 1);
        led2.commit(&mk((1, 2))).expect("commit 2");
        assert_eq!(led2.committed(), 2);
        // A DIFFERENT result for a committed cell refuses loud.
        let mut bad = mk((0, 1));
        bad.e_remove = 0.9;
        assert!(led2.commit(&bad).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn row_spec_roundtrips_through_the_canonical_line() {
        let spec = CellSpec {
            layer: 9,
            chunk: (32, 48),
            band: (4, 8),
        };
        let r = CellResult {
            spec,
            e_observe: 1.0,
            count_observe: 16,
            e_remove: 0.5,
            count_remove: 16,
            e_replace: 0.9,
            count_replace: 16,
            swapped_skipped: 0,
            base_nll_mean: 3.0,
            nll_remove_mean: 3.25,
            nll_replace_mean: 3.1,
            d_nll_remove: 0.25,
            d_nll_replace: 0.1,
            flips_remove: 2,
            flips_replace: 1,
            steps: 64,
        };
        let line = String::from_utf8(row_bytes(&r)).unwrap();
        assert_eq!(spec_from_row_line(&line), Some(spec));
        // The ledger row file name is derived from the same spec.
        assert_eq!(row_file_name(&spec), "9_32_4.json");
        // A foreign line parses to None, never to a wrong spec.
        assert_eq!(spec_from_row_line("cell layer=x chunk=(1)"), None);
    }

    #[test]
    fn row_bytes_match_the_report_cell_line() {
        let r = CellResult {
            spec: CellSpec {
                layer: 9,
                chunk: (32, 48),
                band: (4, 8),
            },
            e_observe: 1.0,
            count_observe: 16,
            e_remove: 0.5,
            count_remove: 16,
            e_replace: 0.9,
            count_replace: 16,
            swapped_skipped: 0,
            base_nll_mean: 3.0,
            nll_remove_mean: 3.25,
            nll_replace_mean: 3.1,
            d_nll_remove: 0.25,
            d_nll_replace: 0.1,
            flips_remove: 2,
            flips_replace: 1,
            steps: 64,
        };
        let row = String::from_utf8(row_bytes(&r)).unwrap();
        let rows = vec![r];
        let report = String::from_utf8(canonical_report_bytes(
            &SweepIdentity {
                model: "m",
                model_len: 0,
                config_sig: "",
                tokens_digest: "",
                read_len: 64,
                null_offset: 64,
                attn_layers: &[9],
                chunks: &[(32, 48)],
                bands: &[(4, 8)],
            },
            &rows,
        ))
        .unwrap();
        // The report embeds the SAME line the ledger committed.
        assert!(report.contains(row.trim_end()));
    }
}
