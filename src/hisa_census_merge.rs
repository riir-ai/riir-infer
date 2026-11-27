//! Issue-1244 T3 landing tooling: merge the sharded census runs into ONE
//! canonical `HISACEN1` report, then classify per (layer, rotating chunk)
//! — the T3.1 typed verdict — with the T2.3 predicted-vs-measured
//! agreement columns against the [`crate::hisa_plan`] closed-form windows.
//!
//! The merge never re-measures: it parses committed census artifacts
//! (report files or ledger state dirs), verifies every digest, folds
//! identical duplicates, refuses conflicting ones, and refuses an
//! INCOMPLETE union loud (the wake's completeness check is this function).
//! The classifier's suppression rule is pre-registered at the tool level
//! and disclosed in every verdict it emits; the T2.3 agreement floor still
//! pins POST-tables (never here) — the tool only prints the confusion.
//!
//! REPORT-ONLY: nothing here runs a forward pass or touches a serving
//! path. Deterministic: fixed formats, ascending folds, no wall clock in
//! any stored value.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use crate::hisa_census_run::{
    CellResult, CellSpec, REPORT_MAGIC, canonical_report_bytes_parts, render_report, row_bytes,
};
use crate::hisa_plan::{HisaHeader, WindowTable};

/// The verdict file's first bytes — a reader refuses anything else.
pub const VERDICT_MAGIC: &[u8] = b"HISAVERDICT1\x01";

/// Pre-registered suppression slack (nats per scored step, mean over the
/// read): the Remove effect counts as suppressed when it is no worse than
/// the Replace noise floor by more than this. Disclosed in every verdict.
pub const DEFAULT_SUPP_ABS: f64 = 1e-3;

/// The planner's angle budget the T2.3 prediction reads (the paper's
/// default; the planner prints the others).
pub const DEFAULT_PHI_DEG: f64 = 90.0;

// ── parsing ────────────────────────────────────────────────────────────

/// One parsed census artifact's identity fields (the cross-input check).
#[derive(Clone, Debug)]
pub struct ReportMeta {
    pub model: String,
    pub read_len: u64,
    pub null_offset: u64,
    pub steps: u64,
    pub blake3: String,
}

/// Parse a rendered report (canonical bytes + the trailing blake3 line).
/// The digest is verified over EXACTLY the canonical prefix; the header's
/// `cells=` count must equal the parsed row count; the grid lines are NOT
/// parsed — every cell line carries its full ranges, and the union grids
/// derive from the cells (never a second table).
pub fn parse_report(bytes: &[u8]) -> Result<(ReportMeta, Vec<CellResult>), String> {
    if !bytes.starts_with(REPORT_MAGIC) {
        return Err("not a census report (bad magic)".to_string());
    }
    let text = std::str::from_utf8(&bytes[REPORT_MAGIC.len()..])
        .map_err(|e| format!("report body is not UTF-8: {e}"))?;
    // The blake3 line is LAST; everything before it is the canonical bytes
    // the digest was taken over — MAGIC included (the digest's input is
    // the exact file prefix, never the post-magic text alone).
    let split = text
        .rfind("\nblake3=")
        .ok_or_else(|| "report has no blake3 line".to_string())?;
    let canonical = &bytes[..REPORT_MAGIC.len() + split + 1];
    let digest_line = &text[split + 1..];
    let digest = digest_line
        .lines()
        .next()
        .and_then(|l| l.strip_prefix("blake3="))
        .map(str::trim)
        .ok_or_else(|| "report blake3 line unparseable".to_string())?;
    let actual = blake3::hash(canonical).to_hex();
    if actual.as_str() != digest {
        return Err(format!(
            "report digest MISMATCH (stored {digest} != computed {actual}) — refusing loud"
        ));
    }

    let mut meta = ReportMeta {
        model: String::new(),
        read_len: 0,
        null_offset: 0,
        steps: 0,
        blake3: digest.to_string(),
    };
    let mut declared_cells: Option<u64> = None;
    let mut rows = Vec::new();
    for line in text[..split + 1].lines() {
        if line.starts_with("model=") {
            for tok in line.split_whitespace() {
                if let Some(v) = tok.strip_prefix("model=") {
                    meta.model = v.to_string();
                } else if let Some(v) = tok.strip_prefix("read_len=") {
                    meta.read_len = v.parse().map_err(|_| "bad read_len".to_string())?;
                } else if let Some(v) = tok.strip_prefix("null_offset=") {
                    meta.null_offset = v.parse().map_err(|_| "bad null_offset".to_string())?;
                } else if let Some(v) = tok.strip_prefix("steps=") {
                    meta.steps = v.parse().map_err(|_| "bad steps".to_string())?;
                }
            }
            if meta.model.is_empty() {
                return Err("report header has no model field".to_string());
            }
        } else if line.starts_with("cells=") {
            for tok in line.split_whitespace() {
                if let Some(v) = tok.strip_prefix("cells=") {
                    declared_cells = Some(v.parse().map_err(|_| "bad cells count".to_string())?);
                }
            }
        } else if line.starts_with("cell ") {
            rows.push(parse_cell_line(line)?);
        }
    }
    if meta.read_len == 0 {
        return Err("report header has no read_len".to_string());
    }
    if let Some(n) = declared_cells
        && n != rows.len() as u64
    {
        return Err(format!(
            "report header declares {n} cells but {} rows parsed",
            rows.len()
        ));
    }
    Ok((meta, rows))
}

/// Parse one canonical cell line (the ledger row file and the report row
/// share the form `row_bytes` writes). Every field is REQUIRED — a
/// missing one refuses loud naming the field, never a defaulted cell.
/// PRECISION: the canonical line carries 10-significant-digit decimals;
/// parsing one lands within an ulp of the measured value, and re-rendering
/// the parsed value is BYTE-IDENTICAL to the source line (format∘parse is
/// idempotent at fixed precision) — the DECIMAL is the artifact of record,
/// never a bit-exact echo of the in-memory measurement. The absolute
/// posture means are not stored on the line; they reconstruct as
/// base+delta (a derived value, disclosed — the classifier and the
/// canonical re-render read only the deltas, so round-trips stay
/// byte-exact).
pub fn parse_cell_line(line: &str) -> Result<CellResult, String> {
    let mut layer: Option<usize> = None;
    let mut chunk: Option<(usize, usize)> = None;
    let mut band: Option<(usize, usize)> = None;
    let mut floats: BTreeMap<&str, f64> = BTreeMap::new();
    let mut ints: BTreeMap<&str, u64> = BTreeMap::new();
    for tok in line.split_whitespace() {
        if tok == "cell" {
            continue; // the row prefix (both media write it)
        }
        if let Some(v) = tok.strip_prefix("layer=") {
            layer = Some(v.parse().map_err(|_| "bad layer".to_string())?);
        } else if let Some(v) = tok.strip_prefix("chunk=") {
            chunk = Some(parse_pair(v)?);
        } else if let Some(v) = tok.strip_prefix("band=") {
            band = Some(parse_pair(v)?);
        } else {
            let (k, v) = tok
                .split_once('=')
                .ok_or_else(|| format!("cell token `{tok}` has no `=`"))?;
            if matches!(
                k,
                "c_obs" | "c_rem" | "c_rep" | "flips_rem" | "flips_rep" | "steps" | "skipped"
            ) {
                let n: u64 = v
                    .parse()
                    .map_err(|_| format!("cell field {k}={v} is not an integer"))?;
                ints.insert(k, n);
            } else {
                let x: f64 = v
                    .parse()
                    .map_err(|_| format!("cell field {k}={v} is not a number"))?;
                floats.insert(k, x);
            }
        }
    }
    let layer = layer.ok_or_else(|| "cell line has no layer".to_string())?;
    let chunk = chunk.ok_or_else(|| "cell line has no chunk".to_string())?;
    let band = band.ok_or_else(|| "cell line has no band".to_string())?;
    let get_f = |k: &str| -> Result<f64, String> {
        floats
            .get(k)
            .copied()
            .ok_or_else(|| format!("cell line missing `{k}`"))
    };
    let get_u = |k: &str| -> Result<u64, String> {
        ints.get(k)
            .copied()
            .ok_or_else(|| format!("cell line missing `{k}`"))
    };
    let base = get_f("nll_base")?;
    let d_rem = get_f("d_nll_rem")?;
    let d_rep = get_f("d_nll_rep")?;
    Ok(CellResult {
        spec: CellSpec { layer, chunk, band },
        e_observe: get_f("e_obs")?,
        e_remove: get_f("e_rem")?,
        e_replace: get_f("e_rep")?,
        count_observe: get_u("c_obs")?,
        count_remove: get_u("c_rem")?,
        count_replace: get_u("c_rep")?,
        swapped_skipped: get_u("skipped")?,
        base_nll_mean: base,
        nll_remove_mean: base + d_rem,
        nll_replace_mean: base + d_rep,
        d_nll_remove: d_rem,
        d_nll_replace: d_rep,
        flips_remove: get_u("flips_rem")?,
        flips_replace: get_u("flips_rep")?,
        steps: get_u("steps")?,
    })
}

fn parse_pair(v: &str) -> Result<(usize, usize), String> {
    let v = v.trim_matches(|c| c == '(' || c == ')');
    let mut it = v.split(',');
    let lo: usize = it
        .next()
        .ok_or_else(|| "pair missing lo".to_string())?
        .trim()
        .parse()
        .map_err(|_| "pair lo not a number".to_string())?;
    let hi: usize = it
        .next()
        .ok_or_else(|| "pair missing hi".to_string())?
        .trim()
        .parse()
        .map_err(|_| "pair hi not a number".to_string())?;
    Ok((lo, hi))
}

// ── the merge ──────────────────────────────────────────────────────────

/// The merged census: identity + every cell, in canonical (sorted) order.
#[derive(Debug)]
pub struct MergedCensus {
    pub model: String,
    pub read_len: u64,
    pub null_offset: u64,
    pub attn_layers: Vec<usize>,
    pub chunks: Vec<(usize, usize)>,
    pub bands: Vec<(usize, usize)>,
    pub rows: Vec<CellResult>,
}

/// Merge census artifacts. Each entry is `(label, meta, rows)`; report
/// inputs carry `Some(meta)` (the header identity), state-dir inputs
/// `None` (rows only — at least ONE report must be present to source the
/// identity). Identity fields must agree across ALL inputs; duplicate
/// specs must carry identical payloads (fold) — a conflicting duplicate
/// refuses loud; the union grids derive from the cells; an INCOMPLETE
/// union refuses with the missing-spec list.
pub fn merge_census(
    inputs: &[(String, Option<ReportMeta>, Vec<CellResult>)],
) -> Result<MergedCensus, String> {
    if inputs.is_empty() {
        return Err("no census inputs".to_string());
    }
    let identity = inputs
        .iter()
        .find_map(|(_, m, _)| m.clone())
        .ok_or_else(|| {
            "no input carries a report header — pass at least one --from REPORT".to_string()
        })?;
    for (label, meta, _) in inputs.iter() {
        let Some(meta) = meta else { continue };
        if meta.model != identity.model {
            return Err(format!(
                "model mismatch: `{}` vs `{}` ({})",
                identity.model, meta.model, label
            ));
        }
        if meta.read_len != identity.read_len {
            return Err(format!(
                "read_len mismatch: {} vs {} ({})",
                identity.read_len, meta.read_len, label
            ));
        }
        if meta.null_offset != identity.null_offset {
            return Err(format!(
                "null_offset mismatch: {} vs {} ({})",
                identity.null_offset, meta.null_offset, label
            ));
        }
    }
    let mut by_spec: BTreeMap<CellSpec, CellResult> = BTreeMap::new();
    for (label, _, rows) in inputs {
        for r in rows {
            if r.steps != identity.steps {
                return Err(format!(
                    "row steps {} != header steps {} ({} {:?})",
                    r.steps, identity.steps, label, r.spec
                ));
            }
            match by_spec.get(&r.spec) {
                None => {
                    by_spec.insert(r.spec, r.clone());
                }
                Some(old) => {
                    if row_bytes(old) != row_bytes(r) {
                        return Err(format!(
                            "duplicate cell {:?} with DIFFERENT payloads ({} vs a prior input) — refusing loud",
                            r.spec, label
                        ));
                    }
                }
            }
        }
    }
    let mut layers: Vec<usize> = Vec::new();
    let mut chunks: Vec<(usize, usize)> = Vec::new();
    let mut bands: Vec<(usize, usize)> = Vec::new();
    for spec in by_spec.keys() {
        if !layers.contains(&spec.layer) {
            layers.push(spec.layer);
        }
        if !chunks.contains(&spec.chunk) {
            chunks.push(spec.chunk);
        }
        if !bands.contains(&spec.band) {
            bands.push(spec.band);
        }
    }
    layers.sort_unstable();
    chunks.sort_unstable();
    bands.sort_unstable();
    // Completeness: every (layer, chunk, band) of the union grid present —
    // the shards partitioned BANDS, so a hole means a shard did not finish.
    let mut missing = Vec::new();
    for &l in &layers {
        for &c in &chunks {
            for &b in &bands {
                let spec = CellSpec {
                    layer: l,
                    chunk: c,
                    band: b,
                };
                if !by_spec.contains_key(&spec) {
                    missing.push(spec);
                }
            }
        }
    }
    if !missing.is_empty() {
        let head: Vec<String> = missing
            .iter()
            .take(8)
            .map(|s| {
                format!(
                    "layer={} chunk=({},{}) band=({},{})",
                    s.layer, s.chunk.0, s.chunk.1, s.band.0, s.band.1
                )
            })
            .collect();
        return Err(format!(
            "INCOMPLETE census union: {} of {} cells missing (first: {}) — shards unfinished, re-check before landing",
            missing.len(),
            layers.len() * chunks.len() * bands.len(),
            head.join("; ")
        ));
    }
    Ok(MergedCensus {
        model: identity.model.clone(),
        read_len: identity.read_len,
        null_offset: identity.null_offset,
        attn_layers: layers,
        chunks,
        bands,
        rows: by_spec.into_values().collect(),
    })
}

/// The merged report bytes (canonical + blake3 line) — the ONE artifact
/// the whole census digests to.
pub fn merged_report_bytes(m: &MergedCensus) -> Vec<u8> {
    let canonical = canonical_report_bytes_parts(
        &m.model,
        m.read_len as usize,
        m.null_offset as usize,
        &m.attn_layers,
        &m.chunks,
        &m.bands,
        &m.rows,
    );
    render_report(&canonical)
}

// ── T3.1 — the typed per-(layer, chunk) classification ─────────────────

/// The T3.1 classes (the issue's own vocabulary; the fourth class is the
/// honest catch-all — a pattern the three named classes do not cover is
/// TYPED, never force-fitted).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupClass {
    /// Suppressed at ALL bands — position-free behavior.
    PartialRopeSufficient,
    /// Distance-conditional suppression — matters near, suppressible far.
    HisaWindowsRequired,
    /// Mass carried at every distance.
    NoSuppression,
    /// Anything else — typed loud, adjudicated in the record text.
    MixedUnclassified,
}

impl GroupClass {
    pub fn as_str(self) -> &'static str {
        match self {
            GroupClass::PartialRopeSufficient => "partial_rope_sufficient",
            GroupClass::HisaWindowsRequired => "hisa_windows_required",
            GroupClass::NoSuppression => "no_suppression",
            GroupClass::MixedUnclassified => "mixed_unclassified",
        }
    }
    /// Parse a class name (the record's reader). Deliberately not the
    /// `FromStr` trait — the error type would be a whole trait impl for a
    /// four-arm match.
    pub fn parse_class(s: &str) -> Option<Self> {
        Some(match s {
            "partial_rope_sufficient" => GroupClass::PartialRopeSufficient,
            "hisa_windows_required" => GroupClass::HisaWindowsRequired,
            "no_suppression" => GroupClass::NoSuppression,
            "mixed_unclassified" => GroupClass::MixedUnclassified,
            _ => return None,
        })
    }
}

/// One group's verdict: the per-band suppression vector (ascending band
/// order) + the margins that decided it.
#[derive(Clone, Debug)]
pub struct GroupVerdict {
    pub layer: usize,
    pub chunk: (usize, usize),
    pub class: GroupClass,
    /// `true` = suppressed at that band (ascending band order).
    pub suppressed: Vec<bool>,
    /// `d_nll_rem − (|d_nll_rep| + supp_abs)` per band — negative = the
    /// suppression slack the band cleared.
    pub margins: Vec<f64>,
}

/// The T3.1 classification over the merged rows. Suppression rule
/// (pre-registered, disclosed): band b is suppressed iff
/// `d_nll_rem(b) <= |d_nll_rep(b)| + supp_abs` — the Remove posture does
/// no more harm than the Replace noise floor, within the slack.
pub fn classify(
    rows: &[CellResult],
    bands: &[(usize, usize)],
    supp_abs: f64,
) -> Result<Vec<GroupVerdict>, String> {
    if bands.is_empty() {
        return Err("no bands to classify over".to_string());
    }
    let mut groups: BTreeMap<(usize, (usize, usize)), Vec<&CellResult>> = BTreeMap::new();
    for r in rows {
        groups
            .entry((r.spec.layer, r.spec.chunk))
            .or_default()
            .push(r);
    }
    let mut out = Vec::new();
    for ((layer, chunk), mut cells) in groups {
        // Align to the ASCENDING band grid; a group missing a band is a
        // caller bug post-completeness-check — refuse, never misalign.
        cells.sort_by_key(|c| c.spec.band);
        let mut suppressed = Vec::with_capacity(bands.len());
        let mut margins = Vec::with_capacity(bands.len());
        for band in bands {
            let cell = cells.iter().find(|c| c.spec.band == *band).ok_or_else(|| {
                format!(
                    "group layer={layer} chunk=({},{}) missing band ({},{})",
                    chunk.0, chunk.1, band.0, band.1
                )
            })?;
            let threshold = cell.d_nll_replace.abs() + supp_abs;
            let margin = cell.d_nll_remove - threshold;
            suppressed.push(margin <= 0.0);
            margins.push(margin);
        }
        let n = suppressed.len();
        let all = suppressed.iter().all(|&s| s);
        let none = suppressed.iter().all(|&s| !s);
        let class = if all {
            GroupClass::PartialRopeSufficient
        } else if !suppressed[0] && suppressed[n - 1] {
            GroupClass::HisaWindowsRequired
        } else if none {
            GroupClass::NoSuppression
        } else {
            GroupClass::MixedUnclassified
        };
        out.push(GroupVerdict {
            layer,
            chunk,
            class,
            suppressed,
            margins,
        });
    }
    Ok(out)
}

// ── T2.3 — the closed-form prior's predicted labels ────────────────────

/// The closed-form prediction for one (chunk, band) cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Predicted {
    /// Every pair in the chunk is past its window across the whole band
    /// (`max window <= band.lo`) — the prior says suppressible.
    Suppressed,
    /// Some pair still carries through the whole band (`max window >=
    /// band.hi`) — the prior says load-bearing.
    NotSuppressed,
    /// The band straddles the chunk's window frontier — disclosed, kept
    /// OUT of the binary agreement (counted separately).
    Partial,
}

/// The T2.3 predicted-label table: (chunk, band) -> prediction.
pub type PredictedMap = BTreeMap<((usize, usize), (usize, usize)), Predicted>;

/// Predicted labels over every (chunk, band) from the planner's exact
/// per-pair windows at `phi_deg` (T2.3's prior: a pair is suppressible at
/// distance d iff its window is well below d).
pub fn predicted_labels(
    header: &HisaHeader,
    phi_deg: f64,
    chunks: &[(usize, usize)],
    bands: &[(usize, usize)],
) -> Result<PredictedMap, String> {
    let table = WindowTable::new(header, phi_deg);
    let windows = &table.windows;
    let p_rot = windows.len();
    let mut out = BTreeMap::new();
    for chunk in chunks {
        let l_lo = chunk.0 / 2;
        let l_hi = chunk.1 / 2;
        if l_lo >= l_hi || l_hi > p_rot {
            return Err(format!(
                "chunk ({},{}) maps to pairs [{},{}) outside the header's {} rotating pairs",
                chunk.0, chunk.1, l_lo, l_hi, p_rot
            ));
        }
        let max_window = windows[l_lo..l_hi]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        for band in bands {
            let pred = if max_window <= band.0 as f64 {
                Predicted::Suppressed
            } else if max_window >= band.1 as f64 {
                Predicted::NotSuppressed
            } else {
                Predicted::Partial
            };
            out.insert((*chunk, *band), pred);
        }
    }
    Ok(out)
}

/// The T2.3 confusion over the cells whose prediction is binary; partial
/// predictions are counted separately and never folded into the rate.
#[derive(Clone, Copy, Debug, Default)]
pub struct Agreement {
    pub n_pred_suppressed: u64,
    pub n_pred_not: u64,
    pub n_pred_partial: u64,
    pub tp: u64,
    pub fp: u64,
    pub fn_: u64,
    pub tn: u64,
}

impl Agreement {
    /// Agreement rate over the binary-predicted cells; None when none.
    pub fn rate(&self) -> Option<f64> {
        let n = self.tp + self.fp + self.fn_ + self.tn;
        (n > 0).then(|| (self.tp + self.tn) as f64 / n as f64)
    }
}

pub fn agreement(
    verdicts: &[GroupVerdict],
    bands: &[(usize, usize)],
    predicted: &PredictedMap,
) -> Agreement {
    let mut a = Agreement::default();
    for v in verdicts {
        for (i, band) in bands.iter().enumerate() {
            match predicted.get(&(v.chunk, *band)) {
                Some(Predicted::Partial) => a.n_pred_partial += 1,
                Some(Predicted::Suppressed) => {
                    a.n_pred_suppressed += 1;
                    if v.suppressed[i] {
                        a.tp += 1;
                    } else {
                        a.fn_ += 1;
                    }
                }
                Some(Predicted::NotSuppressed) => {
                    a.n_pred_not += 1;
                    if v.suppressed[i] {
                        a.fp += 1;
                    } else {
                        a.tn += 1;
                    }
                }
                None => {}
            }
        }
    }
    a
}

// ── the typed verdict file (HISAVERDICT1) ──────────────────────────────

/// The canonical verdict bytes: header + one line per group + counts +
/// the optional T2.3 line + the blake3 line over everything before it.
pub fn render_verdict(
    source_blake3: &str,
    supp_abs: f64,
    phi_line: Option<&str>,
    verdicts: &[GroupVerdict],
    bands: &[(usize, usize)],
    agreement: Option<&Agreement>,
) -> Vec<u8> {
    let mut s = String::new();
    let _ = writeln!(s, "source_blake3={source_blake3}");
    let _ = writeln!(s, "supp_abs={supp_abs:.3e} bands={bands:?}");
    let mut counts: BTreeMap<&'static str, u64> = BTreeMap::new();
    for v in verdicts {
        let supp: Vec<&str> = v
            .suppressed
            .iter()
            .map(|&b| if b { "1" } else { "0" })
            .collect();
        let margins: Vec<String> = v.margins.iter().map(|m| format!("{m:+.2e}")).collect();
        let _ = writeln!(
            s,
            "group layer={} chunk=({},{}) class={} supp={} margins=[{}]",
            v.layer,
            v.chunk.0,
            v.chunk.1,
            v.class.as_str(),
            supp.join(""),
            margins.join(",")
        );
        *counts.entry(v.class.as_str()).or_insert(0) += 1;
    }
    let c = |k: &'static str| counts.get(k).copied().unwrap_or(0);
    let _ = writeln!(
        s,
        "counts partial_rope_sufficient={} hisa_windows_required={} no_suppression={} mixed_unclassified={}",
        c("partial_rope_sufficient"),
        c("hisa_windows_required"),
        c("no_suppression"),
        c("mixed_unclassified")
    );
    if let Some(line) = phi_line {
        let _ = writeln!(s, "model_header={line}");
    }
    if let Some(a) = agreement {
        let rate = a
            .rate()
            .map(|r| format!("{r:.4}"))
            .unwrap_or_else(|| "n/a".to_string());
        let _ = writeln!(
            s,
            "t23 predicted_suppressed={} predicted_not={} predicted_partial={} tp={} fp={} fn={} tn={} agreement={rate} (floor pins POST-tables, in the record)",
            a.n_pred_suppressed, a.n_pred_not, a.n_pred_partial, a.tp, a.fp, a.fn_, a.tn
        );
    }
    let mut bytes = VERDICT_MAGIC.to_vec();
    bytes.extend_from_slice(s.as_bytes());
    let hex = blake3::hash(&bytes).to_hex();
    bytes.extend_from_slice(format!("blake3={hex}\n").as_bytes());
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell(
        layer: usize,
        chunk: (usize, usize),
        band: (usize, usize),
        d_rem: f64,
        d_rep: f64,
    ) -> CellResult {
        CellResult {
            spec: CellSpec { layer, chunk, band },
            e_observe: 1.0,
            e_remove: 0.5,
            e_replace: 0.9,
            count_observe: 8,
            count_remove: 8,
            count_replace: 8,
            swapped_skipped: 0,
            base_nll_mean: 3.0,
            nll_remove_mean: 3.0 + d_rem,
            nll_replace_mean: 3.0 + d_rep,
            d_nll_remove: d_rem,
            d_nll_replace: d_rep,
            flips_remove: 0,
            flips_replace: 0,
            steps: 64,
        }
    }

    const BANDS_4: [(usize, usize); 4] = [(0, 1), (1, 2), (2, 4), (4, 8)];

    #[test]
    fn cell_line_roundtrips_through_the_canonical_form() {
        let r = cell(9, (32, 48), (2, 4), 8.44e-3, -7.92e-3);
        let line = String::from_utf8(row_bytes(&r)).unwrap();
        let parsed = parse_cell_line(line.trim_end()).unwrap();
        assert_eq!(parsed.spec, r.spec);
        // Exact fields: the integers and the representable exponents.
        assert_eq!(parsed.e_observe.to_bits(), r.e_observe.to_bits());
        assert_eq!(parsed.count_observe, r.count_observe);
        assert_eq!(parsed.flips_replace, r.flips_replace);
        // Fixed-precision floats: within an ulp of the measured value and
        // re-render BYTE-IDENTICALLY (the decimal is the artifact).
        let rel = |a: f64, b: f64| ((a / b) - 1.0).abs();
        assert!(rel(parsed.d_nll_remove, r.d_nll_remove) < 1e-9);
        assert!(rel(parsed.d_nll_replace, r.d_nll_replace) < 1e-9);
        // Re-render is byte-identical (the fixed-precision loop is closed).
        assert_eq!(row_bytes(&parsed), line.into_bytes());
    }

    #[test]
    fn cell_line_refuses_missing_fields_loud() {
        let full = String::from_utf8(row_bytes(&cell(1, (0, 16), (0, 1), 1e-4, 2e-4))).unwrap();
        for drop in ["e_rep", "d_nll_rep", "skipped", "steps"] {
            let broken: String = full
                .split_whitespace()
                .filter(|t| !t.starts_with(&format!("{drop}=")))
                .collect::<Vec<_>>()
                .join(" ");
            assert!(
                parse_cell_line(&broken).is_err(),
                "dropping {drop} must refuse"
            );
        }
    }

    #[test]
    fn report_parse_verifies_digest_and_refuses_tamper() {
        let rows = vec![
            cell(3, (0, 16), (0, 1), 1e-3, 5e-4),
            cell(3, (0, 16), (1, 2), -1e-4, 2e-4),
        ];
        // Build the canonical bytes through the shared renderer.
        let canonical = canonical_report_bytes_parts(
            "m.gguf",
            64,
            64,
            &[3],
            &[(0, 16), (16, 32)],
            &BANDS_4,
            &rows,
        );
        let rendered = render_report(&canonical);
        let (parsed_meta, parsed_rows) = parse_report(&rendered).unwrap();
        assert_eq!(parsed_meta.model, "m.gguf");
        assert_eq!(parsed_rows.len(), 2);
        assert_eq!(parsed_rows[0].spec.band, (0, 1));
        // Tamper one byte of the payload — the digest must refuse it.
        let mut tampered = rendered.clone();
        let at = rendered.len() - 12;
        tampered[at] = tampered[at].wrapping_add(1);
        assert!(parse_report(&tampered).is_err(), "tampered report refuses");
        // Not-a-report refuses.
        assert!(parse_report(b"nonsense").is_err());
    }

    #[test]
    fn merge_folds_identical_duplicates_and_refuses_conflicts() {
        let meta = ReportMeta {
            model: "m".into(),
            read_len: 64,
            null_offset: 64,
            steps: 64,
            blake3: "x".into(),
        };
        let r1 = cell(3, (0, 16), (0, 1), 1e-3, 5e-4);
        let mut r_conflict = r1.clone();
        r_conflict.d_nll_remove = 9e-3;
        // Identical duplicate folds.
        let ok = merge_census(&[
            ("a".into(), Some(meta.clone()), vec![r1.clone()]),
            ("b".into(), Some(meta.clone()), vec![r1.clone()]),
        ])
        .unwrap();
        assert_eq!(ok.rows.len(), 1);
        // Conflicting duplicate refuses, naming the label.
        let err = merge_census(&[
            ("a".into(), Some(meta.clone()), vec![r1]),
            ("b".into(), Some(meta.clone()), vec![r_conflict]),
        ])
        .unwrap_err();
        assert!(
            err.contains("DIFFERENT payloads") && err.contains('b'),
            "{err}"
        );
        // Identity mismatch refuses.
        let mut meta2 = meta.clone();
        meta2.read_len = 16;
        assert!(
            merge_census(&[
                ("a".into(), Some(meta), vec![]),
                ("b".into(), Some(meta2), vec![])
            ])
            .is_err()
        );
    }

    #[test]
    fn merge_refuses_incomplete_union_loud() {
        let meta = ReportMeta {
            model: "m".into(),
            read_len: 64,
            null_offset: 64,
            steps: 64,
            blake3: "x".into(),
        };
        let rows = vec![
            cell(3, (0, 16), (0, 1), 0.0, 0.0),
            cell(3, (16, 32), (4, 8), 0.0, 0.0),
        ];
        let err = merge_census(&[("a".into(), Some(meta), rows)]).unwrap_err();
        assert!(err.contains("INCOMPLETE"), "{err}");
        assert!(err.contains("missing"), "{err}");
    }

    fn group_of(chunk: (usize, usize), d_rems: [f64; 4], d_reps: [f64; 4]) -> Vec<CellResult> {
        BANDS_4
            .iter()
            .zip(d_rems)
            .zip(d_reps)
            .map(|((&band, dr), dp)| cell(7, chunk, band, dr, dp))
            .collect()
    }

    #[test]
    fn classifier_types_the_three_classes_and_the_mixed_rest() {
        let supp_abs = DEFAULT_SUPP_ABS;
        // All bands suppressed -> partial_rope_sufficient.
        let all_supp = group_of((0, 16), [0.0, 0.0, 0.0, 0.0], [5e-4; 4]);
        // Near load-bearing, far suppressed -> hisa_windows_required.
        let windows = group_of((16, 32), [5e-2, 5e-2, 5e-2, 0.0], [5e-4; 4]);
        // Never suppressed -> no_suppression.
        let never = group_of((32, 48), [5e-2; 4], [5e-4; 4]);
        // Scramble -> mixed_unclassified (suppressed near AND far, hole in
        // the middle — no named class covers that).
        let mixed = group_of((48, 64), [0.0, 5e-2, 0.0, 0.0], [5e-4; 4]);
        let mut rows = Vec::new();
        rows.extend(all_supp.clone());
        rows.extend(windows.clone());
        rows.extend(never.clone());
        rows.extend(mixed.clone());
        let v = classify(&rows, &BANDS_4, supp_abs).unwrap();
        assert_eq!(v.len(), 4);
        let class_of = |c: (usize, usize)| v.iter().find(|g| g.chunk == c).unwrap().class;
        assert_eq!(class_of((0, 16)), GroupClass::PartialRopeSufficient);
        assert_eq!(class_of((16, 32)), GroupClass::HisaWindowsRequired);
        assert_eq!(class_of((32, 48)), GroupClass::NoSuppression);
        assert_eq!(class_of((48, 64)), GroupClass::MixedUnclassified);
    }

    #[test]
    fn suppression_rule_tracks_the_pre_registered_slack() {
        // d_rem at |d_rep| + slack -> suppressed (<=); just above -> not.
        let rows = group_of((0, 16), [6e-4, 1.61e-3, 0.0, 0.0], [5e-4; 4]);
        let v = classify(&rows[..2], &BANDS_4[..2], DEFAULT_SUPP_ABS).unwrap();
        assert_eq!(v[0].suppressed, [true, false]);
        assert!(v[0].margins[0] <= 0.0 && v[0].margins[1] > 0.0);
    }

    fn test_header() -> HisaHeader {
        HisaHeader {
            freq_base: 1e7,
            key_length: 64,
            dimension_count: Some(64),
            dimension_sections: None,
            full_attention_interval: Some(4),
            block_count: 64,
            context_length: 262_144,
            arch: "qwen35".to_string(),
        }
    }

    #[test]
    fn predicted_labels_follow_the_exact_windows() {
        let header = test_header();
        let chunks = vec![(0, 16), (48, 64)];
        let bands = vec![(2, 4), (64, 128), (1024, 2048)];
        let pred = predicted_labels(&header, DEFAULT_PHI_DEG, &chunks, &bands).unwrap();
        let wt = WindowTable::new(&header, DEFAULT_PHI_DEG);
        let w_low = wt.windows[0..8]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let w_high = wt.windows[24..32]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        // The low chunk's frontier: below it Suppressed, above NotSuppressed.
        if w_low <= 64.0 {
            assert_eq!(pred[&((0, 16), (64, 128))], Predicted::Suppressed);
        }
        if w_low >= 4.0 {
            assert_eq!(pred[&((0, 16), (2, 4))], Predicted::NotSuppressed);
        }
        // The high chunk's slowest pair reaches past any readable distance
        // (R = Φ·b^((d_rot−2)/d_rot) ≈ 9.5e6) — NotSuppressed everywhere.
        assert!(w_high >= 2048.0);
        assert_eq!(pred[&((48, 64), (1024, 2048))], Predicted::NotSuppressed);
    }

    #[test]
    fn agreement_counts_binary_cells_and_keeps_partial_out() {
        let bands = BANDS_4.to_vec();
        let mk = |chunk: (usize, usize), supp: [bool; 4]| GroupVerdict {
            layer: 0,
            chunk,
            class: GroupClass::MixedUnclassified,
            suppressed: supp.to_vec(),
            margins: vec![0.0; 4],
        };
        let header = test_header();
        let pred = predicted_labels(&header, DEFAULT_PHI_DEG, &[(0, 16)], &bands).unwrap();
        let verdicts = vec![mk((0, 16), [false, false, false, false])];
        let a = agreement(&verdicts, &bands, &pred);
        // Whatever the windows say, partial predictions never enter the
        // binary counts, and tp+fp+fn+tn+partials == bands.
        assert_eq!(
            a.tp + a.fp + a.fn_ + a.tn + a.n_pred_partial,
            bands.len() as u64
        );
    }

    #[test]
    fn verdict_bytes_are_canonical_and_digest_pinned() {
        let bands = BANDS_4.to_vec();
        let v = vec![GroupVerdict {
            layer: 3,
            chunk: (0, 16),
            class: GroupClass::HisaWindowsRequired,
            suppressed: vec![false, false, true, true],
            margins: vec![1e-3, 2e-3, -1e-3, -2e-3],
        }];
        let bytes = render_verdict("abc", DEFAULT_SUPP_ABS, None, &v, &bands, None);
        assert!(bytes.starts_with(VERDICT_MAGIC));
        let text = std::str::from_utf8(&bytes).unwrap();
        let split = text.rfind("\nblake3=").unwrap();
        let digest = text[split + 1..].lines().next().unwrap();
        let actual = blake3::hash(&bytes[..split + 1]).to_hex();
        assert_eq!(format!("blake3={actual}"), digest);
        assert!(text.contains("class=hisa_windows_required"));
        assert!(text.contains("counts partial_rope_sufficient=0 hisa_windows_required=1 no_suppression=0 mixed_unclassified=0"));
        // Class vocabulary round-trips (the record's reader).
        assert_eq!(
            GroupClass::parse_class("hisa_windows_required"),
            Some(GroupClass::HisaWindowsRequired)
        );
        assert_eq!(GroupClass::parse_class("nonsense"), None);
    }
}
