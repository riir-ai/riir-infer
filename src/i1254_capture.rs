//! Issue 1254 (Plato's Key cell) — the capture DATA PLANE
//! (`i1254_capture`).
//!
//! The A1 adjudication cell (recipient Llama-3.2-3B BASE × donor
//! Ternary-Bonsai-2.5-27B PQ2_0) needs three planes; this module is the
//! middle one (the fit plane is the LANDED [`crate::i1254_fit`], the run
//! plane is the `i1254_capture` bin):
//!
//! - **The llama-side hooks** — capture (recipient band keys) and replace
//!   (the swap's apply point), both `PostLayerHook` impls over
//!   `forward_llama_hooked`. Armed only at the caller's chosen token
//!   position (the harness drives the token loop); `NoHook`
//!   monomorphizes away, so an unarmed forward is byte-identical (G3).
//! - **The corpus data plane** — the canonical corpus
//!   ([`crate::i1254_corpus`]) bound to capture: subject-token span law
//!   (BPE locality, never a `find`), greedy recall decision (corpus
//!   matching + the distinct-token guard), the 22–38% depth band, and
//!   the resumable per-fact capture store (append-only JSONL rows +
//!   fixed-header binary key planes) with BLAKE3 model provenance in a
//!   per-side meta receipt.
//! - **The paired join** — the two per-side stores joined into
//!   train/val/test paired-key sets for the fit plane, the split tag
//!   RECOMPUTED from the corpus at join time (never trusted from the
//!   rows), plus the recipient per-split baseline recall (the verdict
//!   round-1 gate's data source).
//!
//! Laws carried here (issue text is the record):
//! - **Leak law:** PCs and training means fit on the TRAIN split of the
//!   subject-wise corpus split ONLY — the join buckets pairs per
//!   [`crate::i1254_corpus::Split`], never index-strided, and the fit
//!   plane consumes exactly the bucket it is given.
//! - **Protocol law:** the paired-recall filter (both models answer the
//!   direct prompt) applies WITHIN each split bucket.
//! - **Long-run law (owner compute decree):** the capture store is
//!   append-only + resumable by construction — a killed/rebooted capture
//!   resumes from the rows + key files already on disk, never restarts.
//! - **Boundary law:** a fact whose subject tokenization breaks the
//!   BPE-locality law is skipped and COUNTED, never silently
//!   approximated.
//!
//! Mined from the duplicate-continuation thread's untracked drafts
//! (zed-f6919b20@4090, 2026-11-28 — its CaptureStore/join/recall design
//! carried nearly verbatim; its f64 fit core replaced by the landed
//! closed-form [`crate::i1254_fit`]).

use std::collections::HashSet;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::i1254_corpus::{answer_matches, Fact, Relation, MATCH_WINDOW_TOKENS};
use crate::tokenizer::BpeTokenizer;
use crate::transformer::PostLayerHook;

// ── llama-side hooks (capture + replace) ────────────────────────────────

/// The Issue-395 hook, capture shape: snapshots the post-layer residual at
/// the armed band layers (the recipient-side key capture).
pub struct CaptureHook {
    pub capture_layers: &'static [usize],
    pub captured: Vec<Option<Vec<f32>>>,
}
impl CaptureHook {
    pub fn new(capture_layers: &'static [usize]) -> Self {
        Self {
            capture_layers,
            captured: vec![None; capture_layers.len()],
        }
    }
}
impl PostLayerHook for CaptureHook {
    fn after_layer(&mut self, layer_idx: usize, residual: &mut [f32]) {
        if let Some(slot) = self.capture_layers.iter().position(|&l| l == layer_idx) {
            self.captured[slot] = Some(residual.to_vec());
        }
    }
}

/// The Issue-395 hook, replace shape: writes a prepared state into the
/// post-layer residual at the armed layer (the swap's replace point; the
/// P5 eval arms' seam).
pub struct ReplaceHook {
    pub layer: usize,
    pub state: Vec<f32>,
}
impl PostLayerHook for ReplaceHook {
    fn after_layer(&mut self, layer_idx: usize, residual: &mut [f32]) {
        if layer_idx == self.layer {
            let n = residual.len().min(self.state.len());
            residual[..n].copy_from_slice(&self.state[..n]);
        }
    }
}

/// Greedy argmax with the first-index tie rule (the deterministic pick).
pub fn argmax_first(logits: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

// ── the corpus data plane: subject span + recall decision ───────────────

/// Locate the subject's last token index for a corpus-rendered prompt with
/// the BPE-locality check. `prefix` is [`crate::i1254_corpus::prompt_parts`]'s
/// first element — the prompt up to and including the subject — so the span
/// comes from the prompt's own structure, never a `find` that could match an
/// earlier occurrence. Returns Err when the boundary law fails
/// (the fact is skipped and counted, never silently).
pub fn subject_tokens(
    tok: &BpeTokenizer,
    prompt: &str,
    prefix: &str,
) -> std::result::Result<(Vec<usize>, usize), String> {
    debug_assert!(prompt.starts_with(prefix), "prefix must end at the subject");
    let full = tok.encode_with_bos(prompt);
    let ps = tok.encode_with_bos(prefix);
    if ps.len() < 2 || full.len() < ps.len() + 1 {
        return Err(format!(
            "degenerate tokenization (prefix {ps} tok, full {full} tok)",
            ps = ps.len(),
            full = full.len()
        ));
    }
    // BPE locality: the full prompt's token prefix must equal the prefix's
    // own tokenization (a merge across the boundary breaks the index law).
    if full[..ps.len()] != ps[..] {
        return Err(
            "BPE locality: full prompt's token prefix != prefix's own tokenization \
             (a merge crosses the subject boundary)"
                .into(),
        );
    }
    Ok((full, ps.len() - 1))
}

/// Distinct-token floor inside the greedy window (a repetition run is not
/// recall — the example's guard, carried).
pub const RECALL_DISTINCT_RATIO: f64 = 0.4;

pub fn distinct_token_ratio(window: &[usize]) -> f64 {
    if window.is_empty() {
        return 0.0;
    }
    let mut seen = HashSet::new();
    for t in window {
        seen.insert(*t);
    }
    seen.len() as f64 / window.len() as f64
}

/// The corpus recall decision: the greedy window (≤
/// [`crate::i1254_corpus::MATCH_WINDOW_TOKENS`] tokens) must clear the
/// distinct-token guard AND contain an acceptable object under the
/// relation's matching semantics (accent-folded, per-relation case
/// sensitivity).
pub fn recall_decision(
    gen_toks: &[usize],
    tok: &BpeTokenizer,
    fact: &Fact,
    rel: &Relation,
) -> bool {
    let k = gen_toks.len().min(MATCH_WINDOW_TOKENS);
    if k == 0 {
        return false;
    }
    let window = &gen_toks[..k];
    if distinct_token_ratio(window) < RECALL_DISTINCT_RATIO {
        return false;
    }
    let decoded = tok.decode(window);
    answer_matches(&decoded, &fact.obj, rel.case_sensitive)
}

/// The depth band 22–38% — layer INDICES, never absolute numbers in prose
/// (28-block and 64-block recipients share one law).
pub fn depth_band(n_layer: usize) -> Vec<usize> {
    let lo = ((n_layer as f64 * 0.22).round() as usize).max(1);
    let hi = ((n_layer as f64 * 0.38).round() as usize).min(n_layer.saturating_sub(1));
    (lo..=hi).collect()
}

// ── the capture store (resumable, append-only) ──────────────────────────

/// Magic + layout of a per-fact key file: `I1254KS1` (8 bytes) ‖ u32 LE
/// `n_planes` ‖ u32 LE `dim` ‖ `n_planes × dim` f32 LE planes. Fixed 16-byte
/// header so a torn or foreign file refuses on size/shape at read time.
pub const KEYS_MAGIC: &[u8; 8] = b"I1254KS1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureRow {
    /// The fact's index in canonical corpus order.
    pub fact: usize,
    /// Relation code (corpus registry).
    pub rel: String,
    /// Split bucket char (`T`/`V`/`S` per the corpus digest convention).
    pub split: char,
    /// Subject's last-token position in the tokenized prompt.
    pub subj_pos: usize,
    /// Full prompt token count.
    pub n_tokens: usize,
    /// Greedy-window recall decision.
    pub recall: bool,
    /// The boundary/span law passed (a false row carries no keys).
    pub boundary_ok: bool,
    /// Skip reason when the fact did not produce keys ("" otherwise).
    pub note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureMeta {
    pub side: String,
    pub model_path: String,
    pub model_blake3: String,
    pub n_layer: usize,
    pub band: Vec<usize>,
    pub dim: usize,
    pub ctx: usize,
    pub facts_attempted: usize,
    pub recall_hits: usize,
    pub skips_boundary: usize,
    pub skips_ctx: usize,
}

/// One side's capture state: append-only `rows_<side>.jsonl` +
/// `keys/<side>/<fact>.bin` + a `meta_<side>.json` receipt written at phase
/// end. Resume = the rows file's fact set (each row's keys file is
/// validated before the fact is skipped).
pub struct CaptureStore {
    dir: PathBuf,
    side: String,
}

impl CaptureStore {
    pub fn open(dir: &Path, side: &str) -> std::io::Result<Self> {
        let keys_dir = dir.join("keys").join(side);
        std::fs::create_dir_all(&keys_dir)?;
        Ok(Self {
            dir: dir.to_path_buf(),
            side: side.to_string(),
        })
    }

    pub fn rows_path(&self) -> PathBuf {
        self.dir.join(format!("rows_{}.jsonl", self.side))
    }

    pub fn keys_path(&self, fact: usize) -> PathBuf {
        self.dir
            .join("keys")
            .join(&self.side)
            .join(format!("{fact}.bin"))
    }

    pub fn meta_path(&self) -> PathBuf {
        self.dir.join(format!("meta_{}.json", self.side))
    }

    /// The fact indices already captured (rows present). A malformed
    /// TRAILING line (a torn append) is dropped with a warning; malformed
    /// interior lines refuse — an interior hole means the store was
    /// tampered with, never silently patched.
    pub fn load_done(&self) -> std::io::Result<HashSet<usize>> {
        let mut done = HashSet::new();
        let path = self.rows_path();
        if !path.exists() {
            return Ok(done);
        }
        let file = std::fs::File::open(&path)?;
        let mut lines: Vec<String> = BufReader::new(file).lines().collect::<Result<_, _>>()?;
        while let Some(last) = lines.last() {
            if last.trim().is_empty() || serde_json::from_str::<CaptureRow>(last).is_ok() {
                break;
            }
            lines.pop();
            eprintln!(
                "[store] dropped torn trailing row line (crash tail) — will re-run that fact"
            );
        }
        for (i, line) in lines.iter().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let row: CaptureRow = serde_json::from_str(line).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("{} line {}: malformed row: {e}", path.display(), i + 1),
                )
            })?;
            done.insert(row.fact);
        }
        Ok(done)
    }

    pub fn append_row(&self, row: &CaptureRow) -> std::io::Result<()> {
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.rows_path())?;
        let mut line = serde_json::to_string(row)?;
        line.push('\n');
        f.write_all(line.as_bytes())?;
        f.sync_data()?;
        Ok(())
    }

    pub fn write_keys(&self, fact: usize, planes: &[Vec<f32>]) -> std::io::Result<()> {
        let dim = planes.first().map(|p| p.len()).unwrap_or(0);
        let mut bytes: Vec<u8> = Vec::with_capacity(16 + planes.len() * dim * 4);
        bytes.extend_from_slice(KEYS_MAGIC);
        bytes.extend_from_slice(&(planes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&(dim as u32).to_le_bytes());
        for p in planes {
            for v in p {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        let tmp = self.keys_path(fact).with_extension("bin.tmp");
        {
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(&bytes)?;
            f.sync_data()?;
        }
        std::fs::rename(&tmp, self.keys_path(fact))?;
        Ok(())
    }

    /// Read + validate a key file. Size/magic mismatches refuse loud.
    pub fn read_keys(&self, fact: usize) -> Result<Vec<Vec<f32>>> {
        read_keys_file(&self.keys_path(fact))
    }

    pub fn write_meta(&self, meta: &CaptureMeta) -> std::io::Result<()> {
        let tmp = self.meta_path().with_extension("json.tmp");
        {
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(serde_json::to_string_pretty(meta)?.as_bytes())?;
            f.write_all(b"\n")?;
            f.sync_data()?;
        }
        std::fs::rename(&tmp, self.meta_path())?;
        Ok(())
    }

    pub fn read_meta(&self) -> Result<CaptureMeta> {
        let raw = std::fs::read_to_string(self.meta_path())
            .with_context(|| format!("meta {}", self.meta_path().display()))?;
        serde_json::from_str(&raw).with_context(|| format!("meta parse {}", self.side))
    }

    /// All rows, in append order.
    pub fn read_rows(&self) -> Result<Vec<CaptureRow>> {
        let path = self.rows_path();
        if !path.exists() {
            bail!(
                "rows file {} missing — run the capture phase first",
                path.display()
            );
        }
        let file = std::fs::File::open(&path)?;
        let mut rows = Vec::new();
        for (i, line) in BufReader::new(file).lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let row: CaptureRow = serde_json::from_str(&line).map_err(|e| {
                anyhow::anyhow!("{} line {}: malformed row: {e}", path.display(), i + 1)
            })?;
            rows.push(row);
        }
        Ok(rows)
    }
}

fn read_keys_file(path: &Path) -> Result<Vec<Vec<f32>>> {
    let raw = std::fs::read(path).with_context(|| format!("keys {}", path.display()))?;
    if raw.len() < 16 || raw[0..8] != *KEYS_MAGIC {
        bail!(
            "keys {}: bad magic/size (len {})",
            path.display(),
            raw.len()
        );
    }
    let n_planes = u32::from_le_bytes(raw[8..12].try_into().unwrap()) as usize;
    let dim = u32::from_le_bytes(raw[12..16].try_into().unwrap()) as usize;
    let want = 16 + n_planes * dim * 4;
    if raw.len() != want {
        bail!(
            "keys {}: size {} != header want {want} (n_planes {n_planes} × dim {dim}) — torn or foreign file",
            path.display(),
            raw.len()
        );
    }
    let mut planes = Vec::with_capacity(n_planes);
    let mut off = 16;
    for _ in 0..n_planes {
        let mut plane = Vec::with_capacity(dim);
        for _ in 0..dim {
            let b: [u8; 4] = raw[off..off + 4].try_into().unwrap();
            plane.push(f32::from_le_bytes(b));
            off += 4;
        }
        planes.push(plane);
    }
    Ok(planes)
}

// ── the paired join (the fit plane's input) ─────────────────────────────

/// A paired fact with BOTH sides' band keys (f32 as captured). `don_keys` /
/// `rec_keys` are the band planes in band order — plane `i` is the hidden
/// state at band layer `i` (the fit driver flattens one (ℓA, ℓB) combo at a
/// time into the [`crate::i1254_fit`] row-major key matrices).
#[derive(Clone)]
pub struct PairedKeys {
    pub fact: usize,
    pub don_keys: Vec<Vec<f32>>,
    pub rec_keys: Vec<Vec<f32>>,
}

/// The paired sets per split, joined from the two capture stores. The split
/// tag is RECOMPUTED from the corpus (never trusted from the rows) and
/// asserted equal to the row's char. Both stores' band metas must agree
/// (the layers have to correspond — a band mismatch refuses LOUD). Facts
/// join only when both sides recalled AND both boundary laws passed AND
/// both key files read.
pub fn join_paired(
    donor: &CaptureStore,
    recipient: &CaptureStore,
    facts: &[Fact],
) -> Result<(Vec<PairedKeys>, Vec<PairedKeys>, Vec<PairedKeys>)> {
    let don_meta = donor.read_meta().context("donor meta")?;
    let rec_meta = recipient.read_meta().context("recipient meta")?;
    if don_meta.band != rec_meta.band {
        bail!(
            "band mismatch: donor {:?} != recipient {:?} — the two captures ran different depth bands; the layers do not correspond",
            don_meta.band,
            rec_meta.band
        );
    }
    let don_rows = donor.read_rows()?;
    let rec_rows = recipient.read_rows()?;
    let rec_by_fact: std::collections::HashMap<usize, &CaptureRow> =
        rec_rows.iter().map(|r| (r.fact, r)).collect();
    let mut sets: [Vec<PairedKeys>; 3] = Default::default();
    let mut joined_skips: Vec<String> = Vec::new();
    for dr in &don_rows {
        let Some(rr) = rec_by_fact.get(&dr.fact) else {
            continue;
        };
        if !dr.boundary_ok || !rr.boundary_ok || !dr.recall || !rr.recall {
            continue;
        }
        let fact = &facts[dr.fact];
        let split = crate::i1254_corpus::split_of(&fact.subj);
        let tag = match split {
            crate::i1254_corpus::Split::Train => 'T',
            crate::i1254_corpus::Split::Val => 'V',
            crate::i1254_corpus::Split::Test => 'S',
        };
        if dr.split != tag {
            bail!(
                "fact {}: row split '{}' != corpus split '{tag}' — the store is stale against the corpus",
                dr.fact,
                dr.split
            );
        }
        let don_keys = match donor.read_keys(dr.fact) {
            Ok(k) => k,
            Err(e) => {
                joined_skips.push(format!("fact {}: donor keys: {e}", dr.fact));
                continue;
            }
        };
        let rec_keys = match recipient.read_keys(rr.fact) {
            Ok(k) => k,
            Err(e) => {
                joined_skips.push(format!("fact {}: recipient keys: {e}", rr.fact));
                continue;
            }
        };
        let slot = match split {
            crate::i1254_corpus::Split::Train => 0,
            crate::i1254_corpus::Split::Val => 1,
            crate::i1254_corpus::Split::Test => 2,
        };
        sets[slot].push(PairedKeys {
            fact: dr.fact,
            don_keys,
            rec_keys,
        });
    }
    if !joined_skips.is_empty() {
        eprintln!(
            "[join] {} paired fact(s) skipped on unreadable keys (first: {})",
            joined_skips.len(),
            joined_skips[0]
        );
    }
    let [train, val, test] = sets;
    Ok((train, val, test))
}

/// Per-split recipient baseline (recall rate over the split's boundary-ok
/// rows) — the verdict round-1 gate's data source (a recipient whose
/// baseline recall is broken is a REFUSAL naming the phase, never a
/// transfer table).
pub fn baseline_per_split(recipient: &CaptureStore, facts: &[Fact]) -> Result<[(usize, usize); 3]> {
    let rows = recipient.read_rows()?;
    let mut counts = [(0usize, 0usize); 3]; // (hits, total)
    for r in &rows {
        if !r.boundary_ok {
            continue;
        }
        let split = crate::i1254_corpus::split_of(&facts[r.fact].subj);
        let slot = match split {
            crate::i1254_corpus::Split::Train => 0,
            crate::i1254_corpus::Split::Val => 1,
            crate::i1254_corpus::Split::Test => 2,
        };
        counts[slot].1 += 1;
        if r.recall {
            counts[slot].0 += 1;
        }
    }
    Ok(counts)
}

/// BLAKE3 of a model file (streamed, 1 MiB chunks) — the capture's
/// provenance receipt.
pub fn file_blake3(path: &Path) -> Result<String> {
    let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = blake3::Hasher::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut reader = std::io::BufReader::with_capacity(1 << 20, file);
    loop {
        use std::io::Read;
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::i1254_corpus::relation_of;

    // The shared-temp-path law: a unique per-process scratch root.
    fn scratch_root() -> PathBuf {
        std::env::temp_dir().join(format!("i1254_capture_test_{}", std::process::id()))
    }

    #[test]
    fn keys_codec_round_trip() {
        let dir = scratch_root().join("keys_codec");
        let store = CaptureStore::open(&dir, "donor").unwrap();
        let planes = vec![vec![1.0f32, -2.5, 3.25, 0.0], vec![-1e-8, 7.5, 0.125, 4.0]];
        store.write_keys(7, &planes).unwrap();
        let back = store.read_keys(7).unwrap();
        assert_eq!(back.len(), 2);
        for (a, b) in planes.iter().zip(&back) {
            assert_eq!(a.len(), b.len());
            for (x, y) in a.iter().zip(b) {
                assert_eq!(x.to_bits(), y.to_bits());
            }
        }
        // Shape mismatch refuses loud.
        let mut raw = std::fs::read(store.keys_path(7)).unwrap();
        raw.truncate(raw.len() - 4);
        std::fs::write(store.keys_path(7), &raw).unwrap();
        assert!(store.read_keys(7).is_err(), "torn file must refuse");
    }

    #[test]
    fn store_resume_and_rows() {
        let dir = scratch_root().join("resume");
        let store = CaptureStore::open(&dir, "recipient").unwrap();
        assert!(store.load_done().unwrap().is_empty());
        let row = CaptureRow {
            fact: 3,
            rel: "c".into(),
            split: 'T',
            subj_pos: 6,
            n_tokens: 12,
            recall: true,
            boundary_ok: true,
            note: String::new(),
        };
        store.append_row(&row).unwrap();
        // A torn trailing line (crash mid-append) is dropped on resume.
        {
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(store.rows_path())
                .unwrap();
            use std::io::Write;
            f.write_all(b"{\"fact\":99,\"rel\":").unwrap();
        }
        let done = store.load_done().unwrap();
        assert!(done.contains(&3));
        assert!(!done.contains(&99), "torn tail must not count as done");
        let rows = store.read_rows();
        // read_rows refuses the torn tail LOUD — load_done already warned
        // and the fit path re-reads clean stores; the torn tail is the
        // honest refusal, never a silently patched store.
        assert!(rows.is_err(), "torn tail must refuse read_rows");
    }

    #[test]
    fn recall_decision_semantics() {
        // Matching rides the corpus module (accent fold, case rules); this
        // pins the decision guards on real data shapes via
        // distinct_token_ratio — the decode-side semantics are already
        // unit-pinned in i1254_corpus and the end-to-end path runs in the
        // capture bins' smoke cell.
        let facts = crate::i1254_corpus::load_corpus().unwrap();
        let window_ok = vec![1usize, 2, 3, 4];
        assert!(distinct_token_ratio(&window_ok) >= RECALL_DISTINCT_RATIO);
        let degenerate = vec![5usize, 5, 5, 5];
        assert!(distinct_token_ratio(&degenerate) < RECALL_DISTINCT_RATIO);
        assert!(!facts.is_empty());
        let rel = relation_of(facts[0].rel).unwrap();
        assert!(!rel.case_sensitive || facts[0].rel == "sa" || facts[0].rel == "cs");
    }

    #[test]
    fn depth_band_shapes() {
        assert_eq!(depth_band(28), vec![6, 7, 8, 9, 10, 11]);
        assert_eq!(depth_band(64), (14..=24).collect::<Vec<_>>());
        // Degenerate depth collapses to an EMPTY band (never panics, never
        // wraps through 0 — the saturating_sub law).
        assert!(depth_band(1).is_empty());
        assert!(depth_band(2).is_empty() || depth_band(2) == vec![1]);
    }

    #[test]
    fn join_pairs_respects_split_and_recall() {
        let dir = scratch_root().join("join");
        let facts = crate::i1254_corpus::load_corpus().unwrap();
        // Pick facts from each split.
        let mut picks = [None::<usize>; 3];
        for (i, f) in facts.iter().enumerate() {
            let slot = match crate::i1254_corpus::split_of(&f.subj) {
                crate::i1254_corpus::Split::Train => 0,
                crate::i1254_corpus::Split::Val => 1,
                crate::i1254_corpus::Split::Test => 2,
            };
            if picks[slot].is_none() {
                picks[slot] = Some(i);
            }
        }
        let donor = CaptureStore::open(&dir, "donor").unwrap();
        let recipient = CaptureStore::open(&dir, "recipient").unwrap();
        let band = depth_band(28);
        for (slot, fi) in picks.iter().enumerate() {
            let fi = fi.unwrap();
            let split_char = ['T', 'V', 'S'][slot];
            let planes = vec![vec![slot as f32 + 1.0; 4]; band.len()];
            donor.write_keys(fi, &planes).unwrap();
            recipient.write_keys(fi, &planes).unwrap();
            for (store, recall) in [(&donor, true), (&recipient, slot != 2)] {
                store
                    .append_row(&CaptureRow {
                        fact: fi,
                        rel: facts[fi].rel.into(),
                        split: split_char,
                        subj_pos: 4,
                        n_tokens: 9,
                        recall,
                        boundary_ok: true,
                        note: String::new(),
                    })
                    .unwrap();
            }
        }
        // The join cross-checks the stores' band metas (both sides must have
        // run the SAME depth band).
        let band_meta = |side: &str, hits: usize| CaptureMeta {
            side: side.into(),
            model_path: "synthetic".into(),
            model_blake3: "synthetic".into(),
            n_layer: 28,
            band: band.clone(),
            dim: 4,
            ctx: 512,
            facts_attempted: 3,
            recall_hits: hits,
            skips_boundary: 0,
            skips_ctx: 0,
        };
        donor.write_meta(&band_meta("donor", 3)).unwrap();
        recipient.write_meta(&band_meta("recipient", 2)).unwrap();
        let (train, val, test) = join_paired(&donor, &recipient, &facts).unwrap();
        assert_eq!(train.len(), 1, "train pick joins");
        assert_eq!(val.len(), 1, "val pick joins");
        assert!(test.is_empty(), "recipient-missed test pick must NOT join");
        assert_eq!(train[0].fact, picks[0].unwrap());
        assert_eq!(train[0].don_keys[0][0], 1.0);

        // A band mismatch refuses LOUD.
        let mut other = band_meta("donor", 3);
        other.band = vec![1, 2];
        donor.write_meta(&other).unwrap();
        assert!(join_paired(&donor, &recipient, &facts).is_err());
    }
}
