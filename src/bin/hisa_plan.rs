//! Issue 1244 T1 — the header-only HiSA planner front (`hisa_plan`).
//!
//! GGUF header before weights (the `plan402_gguf_probe.py` law): no model
//! load, no forward, ~0.2 s against a 5 GB file. REQUIRED header reads
//! before ANY window arithmetic (Research 040 §7): `rope.freq_base`,
//! `attention.key_length`, `rope.dimension_count` (+ `dimension_sections`
//! when present), `full_attention_interval` (hybrids), `block_count`,
//! context length. Absent `rope.dimension_count` = FULL rotary (the
//! header's own statement — the dflash drafter shape), never a default
//! guess; missing required keys refuse loud.
//!
//! REPORT-ONLY: exit 0 on every successful read; the restore pointer on a
//! missing model file (the Issue-617 loud-fail shape).
//!
//! Usage:
//! ```text
//! cargo run --release --bin hisa_plan -- --gguf ../riir-train/data/Ternary-Bonsai-2.5-27B-PQ2_0.gguf
//! cargo run --release --bin hisa_plan -- --gguf <model> --budgets 90,180,360,720 --contexts 512,2048,4096,16384,32768
//! ```

use anyhow::{Context, Result, bail};
use riir_infer_core::gguf_loader::GgufFile;
use riir_infer_core::hisa_plan::{
    DEFAULT_BUDGETS_DEG, DEFAULT_CONTEXTS, HisaHeader, WindowTable, context_cells, ring_table,
};
use std::path::{Path, PathBuf};

const RESTORE_POINTER: &str = "re-download: `hf download prism-ml/Ternary-Bonsai-2.5-27B-GGUF Ternary-Bonsai-2.5-27B-PQ2_0.gguf --local-dir data/` from the riir-train root (the league pin; LOCAL-KEEP row in data/BASE_MODELS.md)";

fn parse_csv_num(s: &str) -> Result<Vec<f64>> {
    s.split(',')
        .map(|p| {
            p.trim()
                .parse::<f64>()
                .with_context(|| format!("hisa_plan: bad csv number `{p}`"))
        })
        .collect()
}

fn parse_csv_u64(s: &str) -> Result<Vec<u64>> {
    s.split(',')
        .map(|p| {
            p.trim()
                .parse::<u64>()
                .with_context(|| format!("hisa_plan: bad csv number `{p}`"))
        })
        .collect()
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut gguf: Option<PathBuf> = None;
    let mut budgets = DEFAULT_BUDGETS_DEG.to_vec();
    let mut contexts = DEFAULT_CONTEXTS.to_vec();
    let mut json = false;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--gguf" => {
                gguf = Some(PathBuf::from(
                    args.next().unwrap_or_else(|| die("--gguf needs a path")),
                ));
            }
            "--budgets" => {
                budgets =
                    parse_csv_num(&args.next().unwrap_or_else(|| die("--budgets needs a csv")))?;
            }
            "--contexts" => {
                contexts =
                    parse_csv_u64(&args.next().unwrap_or_else(|| die("--contexts needs a csv")))?;
            }
            "--json" => json = true,
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => die(&format!("unknown arg `{other}` (--help for usage)")),
        }
    }
    let path = gguf.ok_or_else(|| anyhow::anyhow!("hisa_plan: --gguf <path> required"))?;
    if !path.exists() {
        bail!(
            "hisa_plan: model file missing: {}\n  {RESTORE_POINTER}",
            path.display()
        );
    }
    run(&path, &budgets, &contexts, json)
}

fn die(msg: &str) -> ! {
    eprintln!("hisa_plan: {msg}");
    std::process::exit(2);
}

fn print_help() {
    println!(
        "hisa_plan — the header-only HiSA planner (issue 1244 T1, report-only)\n\
         \n\
         USAGE: hisa_plan --gguf <model.gguf> [--budgets 90,180,360,720] [--contexts 512,2048,4096,16384,32768] [--json]\n\
         \n\
         REQUIRED header reads before any window arithmetic: rope.freq_base,\n\
         attention.key_length, rope.dimension_count (+ dimension_sections),\n\
         full_attention_interval (hybrids), block_count, context_length.\n\
         Absent dimension_count = FULL rotary (the header's own statement)."
    );
}

fn run(path: &Path, budgets: &[f64], contexts: &[u64], json: bool) -> Result<()> {
    let t0 = std::time::Instant::now();
    let gguf = GgufFile::open(path)
        .with_context(|| format!("hisa_plan: cannot open {}", path.display()))?;
    let header = HisaHeader::from_gguf(&gguf)?;
    let read_ms = t0.elapsed().as_millis();

    if json {
        print_json(path, &header, budgets, contexts)?;
        return Ok(());
    }

    println!("== hisa_plan (issue 1244 T1, report-only) ==");
    println!("model: {}", path.display());
    println!(
        "header: arch={} blocks={} ctx={} d(key)={} b={:.3e} read {} ms",
        header.arch,
        header.block_count,
        header.context_length,
        header.key_length,
        header.freq_base,
        read_ms
    );
    match header.dimension_count {
        Some(r) => println!(
            "rope: PARTIAL — dimension_count={r} of key_length={} ({} of {} pairs rotate; {} ALWAYS-KEPT)",
            header.key_length,
            r / 2,
            header.key_length / 2,
            header.key_length / 2 - r / 2
        ),
        None => println!(
            "rope: FULL — no `rope.dimension_count` key ({} of {} pairs rotate)",
            header.rotary_dim() / 2,
            header.key_length / 2
        ),
    }
    if let Some(s) = &header.dimension_sections {
        println!(
            "mrope sections: {:?} — provenance only: text serving collapses to standard partial RoPE (T=H=W=pos, 4th=0; read from riir_infer_core::rope, Issue-594 note), never assumed here",
            s
        );
    }
    match header.full_attention_interval {
        Some(n) => println!(
            "hybrid: full_attention_interval={n} → {} of {} blocks are full attention",
            header.block_count / n,
            header.block_count
        ),
        None => println!("hybrid: none (every block attends)"),
    }

    for &phi in budgets {
        let table = WindowTable::new(&header, phi);
        println!(
            "\n--- Φ={}° ({} rotating pairs, {} always-kept, reach R={:.3e}) ---",
            phi,
            table.windows.len(),
            table.always_kept_pairs,
            table.reach
        );
        println!(
            "{:>6} {:>10} {:>10} {:>8}   per-pair windows over the ROTARY dim",
            "t", "ρ_dec", "ρ_prefill", "κ"
        );
        for cell in context_cells(&table, contexts) {
            println!(
                "{:>6} {:>9.2}% {:>9.2}% {:>7.2}%",
                cell.t,
                cell.decode_kept * 100.0,
                cell.prefill_kept * 100.0,
                cell.cache * 100.0
            );
        }
        // the ring tables at both granularities, at the planner's largest
        // printed context (the ring is a decode-structure; the longest
        // context is the binding posture for allocation)
        let tmax = contexts.iter().copied().max().unwrap_or(2048);
        for (pairs, label) in [(8usize, "16-feature"), (16usize, "32-feature")] {
            let rows = ring_table(&table, pairs, 32);
            println!(
                "  ring table ({} chunks, t={}): chunk l-range → exact window → q8-aligned ring (Q8_BLOCK_SIZE=32)",
                label, tmax
            );
            for r in &rows {
                println!(
                    "    l{:>3}..{:<3} → {:>12.1} → {:>12}",
                    r.l_first,
                    r.l_first + r.chunk_pairs - 1,
                    r.window_exact,
                    r.ring_q8_aligned
                );
            }
        }
    }

    println!("\nreport-only: exit 0 (nothing armed, nothing written)");
    Ok(())
}

fn print_json(path: &Path, header: &HisaHeader, budgets: &[f64], contexts: &[u64]) -> Result<()> {
    use serde_json::json;
    let mut cells = Vec::new();
    for &phi in budgets {
        let table = WindowTable::new(header, phi);
        for cell in context_cells(&table, contexts) {
            cells.push(json!({
                "phi_deg": phi,
                "t": cell.t,
                "decode_kept": cell.decode_kept,
                "prefill_kept": cell.prefill_kept,
                "cache_share": cell.cache,
            }));
        }
    }
    let doc = json!({
        "model": path.display().to_string(),
        "header": {
            "arch": header.arch,
            "block_count": header.block_count,
            "context_length": header.context_length,
            "key_length": header.key_length,
            "freq_base": header.freq_base,
            "dimension_count": header.dimension_count,
            "dimension_sections": header.dimension_sections,
            "full_attention_interval": header.full_attention_interval,
            "rotary_dim": header.rotary_dim(),
            "total_pairs": header.total_pairs(),
            "rotating_pairs": header.rotating_pairs(),
        },
        "cells": cells,
    });
    println!("{}", serde_json::to_string_pretty(&doc)?);
    Ok(())
}
