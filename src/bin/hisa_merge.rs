//! hisa_merge — the Issue-1244 T3 landing front: merge the sharded
//! census runs into ONE canonical `HISACEN1` report, then classify per
//! (layer, rotating chunk) — the T3.1 typed verdict — with the T2.3
//! predicted-vs-measured columns against the `hisa_plan` closed-form
//! windows. REPORT-ONLY: no forward pass, no serving path; every refusal
//! is loud and typed.
//!
//! ```bash
//! # The landing pass: four shard reports -> one merged report + verdict:
//! cargo run --release --features hisa_census --bin hisa_merge -- \
//!     --from ~/riir-bench/i1244_b012_report.txt \
//!     --from ~/riir-bench/i1244_b345_report.txt \
//!     --from ~/riir-bench/i1244_b678_report.txt \
//!     --from ~/riir-bench/i1244_b91011_report.txt \
//!     --out  ~/riir-bench/i1244_merged_report.txt \
//!     --classify-out ~/riir-bench/i1244_verdict.txt \
//!     --model ../riir-train/data/Ternary-Bonsai-2.5-27B-PQ2_0.gguf
//! # A shard that finished its rows but died before the report: its STATE
//! DIR works too (auto-detected; at least one report must be present for
//! the identity header):
//! #   --from ~/riir-bench/i1244_b678_state
//! ```
//!
//! INCOMPLETE union -> refusal naming the missing cells (the wake's
//! completeness check is this refusal). Duplicate cells fold when
//! byte-identical and refuse otherwise. The suppression rule is the
//! pre-registered `d_nll_rem <= |d_nll_rep| + supp_abs` (default slack
//! 1e-3 nats, disclosed in the verdict); the T2.3 agreement floor is NOT
//! pinned here — it pins POST-tables in the benchmark record.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use riir_infer_core::gguf_loader::GgufFile;
use riir_infer_core::hisa_census_merge::{
    self, DEFAULT_PHI_DEG, DEFAULT_SUPP_ABS, GroupClass, ReportMeta, parse_cell_line,
};
use riir_infer_core::hisa_census_run::CellResult;
use riir_infer_core::hisa_plan::HisaHeader;

struct Args {
    from: Vec<PathBuf>,
    out: Option<PathBuf>,
    classify_out: Option<PathBuf>,
    supp_abs: f64,
    model: Option<PathBuf>,
    phi_deg: f64,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        from: Vec::new(),
        out: None,
        classify_out: None,
        supp_abs: DEFAULT_SUPP_ABS,
        model: None,
        phi_deg: DEFAULT_PHI_DEG,
    };
    let mut it = std::env::args().skip(1);
    while let Some(f) = it.next() {
        let mut val = |name: &str| -> String {
            it.next()
                .unwrap_or_else(|| panic!("--{name} needs a value"))
        };
        match f.as_str() {
            "--from" => a.from.push(val("from").into()),
            "--out" => a.out = Some(val("out").into()),
            "--classify-out" => a.classify_out = Some(val("classify-out").into()),
            "--supp-abs" => a.supp_abs = val("supp-abs").parse().context("bad --supp-abs")?,
            "--model" => a.model = Some(val("model").into()),
            "--phi-deg" => a.phi_deg = val("phi-deg").parse().context("bad --phi-deg")?,
            other => bail!("unknown arg {other} (see the module docs for the usage)"),
        }
    }
    if a.from.is_empty() {
        bail!("no --from inputs — pass one REPORT file or STATE dir per shard");
    }
    Ok(a)
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes).with_context(|| format!("write {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("rename -> {}", path.display()))?;
    Ok(())
}

/// One `--from` input: a report FILE (parsed with its identity header) or
/// a ledger STATE DIR (its `rows/*.json`, no header).
fn load_from(path: &Path) -> Result<(Option<ReportMeta>, Vec<CellResult>)> {
    if path.is_dir() {
        let rows_dir = path.join("rows");
        let mut files: Vec<PathBuf> = std::fs::read_dir(&rows_dir)
            .with_context(|| format!("scan {}", rows_dir.display()))?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().map(|x| x == "json").unwrap_or(false))
            .collect();
        files.sort();
        let mut rows = Vec::new();
        for f in files {
            let bytes = std::fs::read(&f).with_context(|| format!("read {}", f.display()))?;
            let text = String::from_utf8(bytes).context("row file not UTF-8")?;
            for line in text.lines().filter(|l| !l.trim().is_empty()) {
                rows.push(parse_cell_line(line).map_err(|e| anyhow::anyhow!("{f:?}: {e}"))?);
            }
        }
        println!("from {} (state dir): {} row(s)", path.display(), rows.len());
        Ok((None, rows))
    } else {
        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let (meta, rows) = hisa_census_merge::parse_report(&bytes)
            .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
        println!(
            "from {} (report): {} row(s), blake3 {}",
            path.display(),
            rows.len(),
            &meta.blake3[..12.min(meta.blake3.len())]
        );
        Ok((Some(meta), rows))
    }
}

fn main() -> Result<()> {
    let args = parse_args()?;

    // Load every input first (all refusals land before any write).
    let mut inputs: Vec<(String, Option<ReportMeta>, Vec<CellResult>)> = Vec::new();
    for p in &args.from {
        let (meta, rows) = load_from(p)?;
        inputs.push((p.display().to_string(), meta, rows));
    }

    let merged = hisa_census_merge::merge_census(&inputs)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .context("merge refused")?;
    println!(
        "merged: {} cells over {} layers x {} chunks x {} bands (read_len {}, model `{}`)",
        merged.rows.len(),
        merged.attn_layers.len(),
        merged.chunks.len(),
        merged.bands.len(),
        merged.read_len,
        merged.model
    );

    let out = args
        .out
        .context("--out is required (the merged report path)")?;
    if out.exists() {
        bail!(
            "--out {} already exists — refusing to clobber (remove it deliberately to re-merge)",
            out.display()
        );
    }
    let report_bytes = hisa_census_merge::merged_report_bytes(&merged);
    write_atomic(&out, &report_bytes)?;
    // The merged report's own digest — the verdict names it as its source.
    let text = String::from_utf8_lossy(&report_bytes);
    let source_digest = text
        .lines()
        .rev()
        .find_map(|l| l.strip_prefix("blake3="))
        .unwrap_or_default()
        .to_string();
    println!(
        "merged report written: {} (blake3 {})",
        out.display(),
        source_digest
    );

    // T3.1 classification over the merged table.
    let verdicts = hisa_census_merge::classify(&merged.rows, &merged.bands, args.supp_abs)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    for v in &verdicts {
        println!(
            "group layer={:>2} chunk=({:>2},{:>2}) class={:<24} supp={}",
            v.layer,
            v.chunk.0,
            v.chunk.1,
            v.class.as_str(),
            v.suppressed
                .iter()
                .map(|&b| if b { '1' } else { '0' })
                .collect::<String>()
        );
    }
    let count = |c: GroupClass| verdicts.iter().filter(|v| v.class == c).count();
    println!(
        "counts: partial_rope_sufficient={} hisa_windows_required={} no_suppression={} mixed_unclassified={}",
        count(GroupClass::PartialRopeSufficient),
        count(GroupClass::HisaWindowsRequired),
        count(GroupClass::NoSuppression),
        count(GroupClass::MixedUnclassified)
    );

    // T2.3 predicted-vs-measured columns (only with the model header; the
    // agreement FLOOR is not pinned here — it pins in the record text).
    let mut phi_line = None;
    let mut agreement = None;
    if let Some(model) = &args.model {
        let gguf = GgufFile::open(model).with_context(|| format!("open {}", model.display()))?;
        let header = HisaHeader::from_gguf(&gguf).context("header reads")?;
        if !merged.model.contains(
            &model
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default(),
        ) {
            println!(
                "WARN: --model `{}` is not a suffix of the census model string `{}` — the T2.3 prior rides the passed header",
                model.display(),
                merged.model
            );
        }
        let predicted = hisa_census_merge::predicted_labels(
            &header,
            args.phi_deg,
            &merged.chunks,
            &merged.bands,
        )
        .map_err(|e| anyhow::anyhow!("{e}"))?;
        let a = hisa_census_merge::agreement(&verdicts, &merged.bands, &predicted);
        let rate = a
            .rate()
            .map(|r| format!("{r:.4}"))
            .unwrap_or_else(|| "n/a".to_string());
        println!(
            "t23: predicted_suppressed={} predicted_not={} predicted_partial={} tp={} fp={} fn={} tn={} agreement={rate} (floor pins POST-tables)",
            a.n_pred_suppressed, a.n_pred_not, a.n_pred_partial, a.tp, a.fp, a.fn_, a.tn
        );
        phi_line = Some(format!(
            "freq_base={:.3e} key_length={} d_rot={} p_rot={} phi_deg={}",
            header.freq_base,
            header.key_length,
            header.rotary_dim(),
            header.rotating_pairs(),
            args.phi_deg
        ));
        agreement = Some(a);
    } else {
        println!("t23: skipped (no --model header passed; the prior needs freq_base + key_length)");
    }

    if let Some(vout) = &args.classify_out {
        if vout.exists() {
            bail!(
                "--classify-out {} already exists — refusing to clobber",
                vout.display()
            );
        }
        let bytes = hisa_census_merge::render_verdict(
            &source_digest,
            args.supp_abs,
            phi_line.as_deref(),
            &verdicts,
            &merged.bands,
            agreement.as_ref(),
        );
        write_atomic(vout, &bytes)?;
        println!("verdict written: {}", vout.display());
    } else {
        println!("verdict: not written (no --classify-out)");
    }

    // The demote discipline is NOT decided here — it is T3.2's record-text
    // call, applied by the landing session from the printed tables.
    Ok(())
}
