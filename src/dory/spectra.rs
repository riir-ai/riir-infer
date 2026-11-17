//! i1221 T3(a) — the per-loop KL spectra instrument core (plan 636
//! read-order leg 2; the Issue-898 `kl_depth_probe` semantics on the
//! Dory T2 posture).
//!
//! Everything here is a pure function of the weights + prompt + depth:
//! the lens captures snapshots, exits replay the readout, KLs are f64
//! sequential folds. Report-only — the only asserts live in this
//! module's `#[cfg(test)]` fixtures and the runner's real-pin receipts.

use super::forward::{DoryForward, DoryForwardError, DoryKvCache};

/// One lens run: a k=K prefill with per-pass snapshots armed. Returns
/// the full logits (`[pos][vocab]`) and fills `snapshots` with the
/// recurrent-exit state after each completed pass (`snapshots[τ]` =
/// τ+1 loops done, `t·n_embd` f32 rows each). Arming is the unarmed
/// computation verbatim (the fixtures pin final-logits bit-identity).
pub fn run_lens(
    fwd: &DoryForward<'_>,
    tokens: &[u32],
    cache: &mut DoryKvCache,
    snapshots: &mut Vec<Vec<f32>>,
) -> Result<Vec<Vec<f32>>, DoryForwardError> {
    snapshots.clear();
    fwd.prefill_snapshots(tokens, cache, Some(snapshots))
}

/// The readout exit: output block + LM head on a snapshot state.
/// `scratch` is the exit's own output-block cache (fresh per call is
/// the runner's posture; re-running on the same cache is idempotent —
/// the block writes its KV rows before reading them).
pub fn exit_row(
    fwd: &DoryForward<'_>,
    snapshot: &[f32],
    t: usize,
    scratch: &mut DoryKvCache,
) -> Result<Vec<Vec<f32>>, DoryForwardError> {
    let mut s = snapshot.to_vec();
    fwd.readout_rows(&mut s, t, scratch)
}

/// KL(reference ‖ candidate) per position, f64 over the softmax the
/// logits imply (the Bench-899 `kl_from_logits` algebra at f64,
/// clamped at 0 — the true value is non-negative and a tiny negative
/// is f64 cancellation).
pub fn kl_positions(reference: &[Vec<f32>], candidate: &[Vec<f32>]) -> Vec<f64> {
    reference
        .iter()
        .zip(candidate.iter())
        .map(|(p, q)| exit_kl(p, q))
        .collect()
}

/// KL(reference ‖ candidate) for ONE position's logits. f64 sequential
/// fold, clamped at 0, NaN on length mismatch/emptiness (the caller's
/// contract is same-vocab rows, so NaN means a caller bug and the rows
/// carry it visibly rather than laundering it).
pub fn exit_kl(reference: &[f32], candidate: &[f32]) -> f64 {
    if reference.len() != candidate.len() || reference.is_empty() {
        return f64::NAN;
    }
    let lse_p = log_sum_exp_f64(reference);
    let lse_q = log_sum_exp_f64(candidate);
    if !lse_p.is_finite() || !lse_q.is_finite() {
        return f64::NAN;
    }
    let mut acc = 0.0f64;
    for (&a, &b) in reference.iter().zip(candidate.iter()) {
        let a = f64::from(a);
        acc += (a - lse_p).exp() * (a - f64::from(b));
    }
    (acc - lse_p + lse_q).max(0.0)
}

/// KL(exit_1 ‖ exit_d) per position — the deepening axis (how far depth
/// d's readout has moved from depth 1's, the loop's own trajectory).
pub fn kl_loop1_vs_exit(exit1: &[Vec<f32>], exit_d: &[Vec<f32>]) -> Vec<f64> {
    kl_positions(exit1, exit_d)
}

fn log_sum_exp_f64(logits: &[f32]) -> f64 {
    let mut max = f64::NEG_INFINITY;
    for &v in logits {
        let v = f64::from(v);
        if v > max {
            max = v;
        }
    }
    if !max.is_finite() {
        return max;
    }
    let mut acc = 0.0f64;
    for &v in logits {
        acc += (f64::from(v) - max).exp();
    }
    max + acc.ln()
}

/// Last-position argmax (first-index tie rule, the determinism law) and
/// its margin over the runner-up.
pub fn argmax_and_margin(logits: &[f32]) -> (usize, f32) {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    let mut second = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            second = best_v;
            best_v = v;
            best = i;
        } else if v > second {
            second = v;
        }
    }
    (best, best_v - second)
}

/// Position argmax only (first-index tie rule).
pub fn argmax_of(logits: &[f32]) -> usize {
    argmax_and_margin(logits).0
}

/// KL effective depth, 1-based (the Bench-899 definition): the first
/// loop at or after the per-position KL peak whose readout is within
/// `threshold` nats of the final readout. `None` on empty/NaN/no
/// post-peak hit (against the final readout the last entry is 0, so a
/// post-peak hit is the normal exit).
pub fn effective_depth(kl: &[f64], threshold: f64) -> Option<usize> {
    if kl.is_empty() || kl.iter().any(|v| v.is_nan()) {
        return None;
    }
    let peak = first_argmax(kl)?;
    for (i, &v) in kl.iter().enumerate().skip(peak) {
        if v <= threshold {
            return Some(i + 1);
        }
    }
    None
}

/// Smallest k (1-based) such that every loop ≥ k agrees with the final
/// argmax. `argmaxes[τ]` is depth τ+1's readout argmax; the final loop
/// always agrees with itself.
pub fn agreement_exit(argmaxes: &[usize]) -> Option<usize> {
    let last = *argmaxes.last()?;
    let agreeing = argmaxes.iter().rev().take_while(|&&a| a == last).count();
    Some(argmaxes.len() - agreeing + 1)
}

/// Adjacent-step tolerance for the monotone signature (f64 identity
/// holds exactly for the last step; the epsilon only forgives the f32
/// state's rounding at intermediate steps — same shape as Bench 899's
/// tolerance, absolute here because the profiles are already in nats).
pub const KL_VECTOR_EPS: f64 = 1e-9;

/// Fraction of adjacent KL steps that are non-increasing within
/// [`KL_VECTOR_EPS`] (the Bench-899 monotone signature; a profile of
/// length < 2 has no steps and reads 1.0 by convention).
pub fn monotone_fraction(kl: &[f64]) -> f64 {
    if kl.len() < 2 {
        return 1.0;
    }
    let mut non_increasing = 0usize;
    for w in kl.windows(2) {
        if w[1] <= w[0] + KL_VECTOR_EPS {
            non_increasing += 1;
        }
    }
    f64::from(non_increasing as u32) / f64::from(kl.len() as u32 - 1)
}

fn first_argmax(xs: &[f64]) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (i, &v) in xs.iter().enumerate() {
        if v.is_nan() {
            continue;
        }
        match best {
            Some((_, b)) if v <= b => {}
            _ => best = Some((i, v)),
        }
    }
    best.map(|(i, _)| i)
}

// ── resumable rows (plan 636's durable format) ──────────────────────

/// One `(prompt, depth)` spectra row (serde; flushed per row by the
/// runner). Every field is defined at every depth — unlike (d)'s k=1
/// rows, the exit readout exists at depth 1, so no nulls.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SpectraRow {
    pub record: &'static str,
    pub prompt_idx: usize,
    pub depth: usize,
    /// KL(full ‖ exit_d) per position, nats.
    pub kl_full_vs_exit: Vec<f64>,
    pub kl_mean: f64,
    pub kl_max: f64,
    /// KL(exit_1 ‖ exit_d) per position (the deepening axis).
    pub kl_loop1_vs_exit: Vec<f64>,
    pub kl_loop1_mean: f64,
    /// Last-position argmax, full vs exit.
    pub argmax_full_last: usize,
    pub argmax_exit_last: usize,
    pub argmax_flip_last: bool,
    pub margin_full_last: f32,
    pub margin_exit_last: f32,
    /// Positions whose argmax flipped vs full at this depth.
    pub flips_all_positions: usize,
    pub wall_ms: f64,
}

/// Build a row (pure; the runner owns I/O and the per-position argmax
/// census it passes in).
#[allow(clippy::too_many_arguments)]
pub fn spectra_row(
    prompt_idx: usize,
    depth: usize,
    kl_full: &[f64],
    kl_loop1: &[f64],
    argmax_full_last: usize,
    argmax_exit_last: usize,
    margin_full_last: f32,
    margin_exit_last: f32,
    flips_all_positions: usize,
    wall_ms: f64,
) -> SpectraRow {
    let sum: f64 = kl_full.iter().sum();
    let sum1: f64 = kl_loop1.iter().sum();
    SpectraRow {
        record: "i1221_t3a_kl_spectra_row",
        prompt_idx,
        depth,
        kl_full_vs_exit: kl_full.to_vec(),
        kl_mean: sum / kl_full.len().max(1) as f64,
        kl_max: kl_full.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        kl_loop1_vs_exit: kl_loop1.to_vec(),
        kl_loop1_mean: sum1 / kl_loop1.len().max(1) as f64,
        argmax_full_last,
        argmax_exit_last,
        argmax_flip_last: argmax_full_last != argmax_exit_last,
        margin_full_last,
        margin_exit_last,
        flips_all_positions,
        wall_ms,
    }
}

/// Resume lineage check (plan 636's law): the rows file's header must
/// match this run on every lineage field — pin blake3, repo STAMP,
/// k_full, seq_len, prompt_sets, seed. Mismatch REFUSES (exit 2, never
/// append across lineages). Returns the `(prompt_idx, depth)` pairs
/// already completed; a missing/empty file resumes nothing.
#[allow(clippy::too_many_arguments)]
pub fn verify_resume_lineage(
    rows_path: &std::path::Path,
    pin_blake3: &str,
    repo_commit: &str,
    k_full: usize,
    seq_len: usize,
    prompt_sets: usize,
    seed: u64,
) -> std::collections::BTreeSet<(usize, usize)> {
    let content = match std::fs::read_to_string(rows_path) {
        Ok(c) => c,
        Err(_) => return Default::default(),
    };
    let mut lines = content.lines();
    let header = lines.next().unwrap_or("");
    let v: serde_json::Value = match serde_json::from_str(header) {
        Ok(v) => v,
        Err(_) => lineage_bad(rows_path, "header line is not JSON"),
    };
    let model = match v.get("model") {
        Some(m) => m,
        None => lineage_bad(rows_path, "header has no model"),
    };
    if model.get("blake3").and_then(|b| b.as_str()) != Some(pin_blake3) {
        lineage_bad(rows_path, "pin blake3 differs (moved/replaced checkpoint)");
    }
    if model.get("k_full").and_then(|k| k.as_u64()) != Some(k_full as u64) {
        lineage_bad(rows_path, "k_full differs (different checkpoint lineage)");
    }
    if v.get("repo_commit").and_then(|s| s.as_str()) != Some(repo_commit) {
        lineage_bad(rows_path, "repo STAMP differs (different code lineage)");
    }
    let prompts = match v.get("prompts") {
        Some(p) => p,
        None => lineage_bad(rows_path, "header has no prompts"),
    };
    if prompts.get("seq_len").and_then(|s| s.as_u64()) != Some(seq_len as u64)
        || prompts.get("sets").and_then(|s| s.as_u64()) != Some(prompt_sets as u64)
        || prompts.get("seed").and_then(|s| s.as_u64()) != Some(seed)
    {
        lineage_bad(rows_path, "prompt spec differs (seq_len/sets/seed)");
    }
    let mut done = std::collections::BTreeSet::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let r: serde_json::Value = match serde_json::from_str(line) {
            Ok(r) => r,
            Err(_) => lineage_bad(rows_path, "row line is not JSON (torn write?)"),
        };
        if r.get("record").and_then(|x| x.as_str()) != Some("i1221_t3a_kl_spectra_row") {
            continue;
        }
        let p = r.get("prompt_idx").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
        let d = r.get("depth").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
        done.insert((p, d));
    }
    done
}

fn lineage_bad(path: &std::path::Path, msg: &str) -> ! {
    eprintln!(
        "REFUSE: {:?} header lineage mismatch: {msg}\n  never append across \
         lineages — rerun with --fresh or a new --out",
        path,
    );
    std::process::exit(2);
}
