//! i1221_dory_header_probe — the Issue-1221 T0 instrument: the Dory GGUF
//! header census + the loop/I-R/O partition verdict, BEFORE any loader
//! work (T1+). Report-only: header parse only, never a forward path.
//!
//! Reads: `general.architecture` + every metadata key (the loop-count key
//! hunt), then the full tensor inventory grouped by name stem (the
//! `blk.N.` layer index stripped) so per-loop duplication vs a shared
//! recurrent block is visible as a stem COUNT, and the layer-index
//! histogram that exposes the I/R/O block layout.
//!
//! Proof row (T0): every tensor named + the arch key + the partition
//! verdict (explicit metadata vs naming convention). `--full` dumps every
//! tensor row (the committed tripwire fixture is generated with it; a
//! later upstream format event re-runs the probe and diffs against it).
//!
//! ```bash
//! cargo run --release --example i1221_dory_header_probe -- \
//!     /Users/katopz/git/riir-train/data/NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf [--full]
//! ```

use std::collections::BTreeMap;
use std::path::PathBuf;

use riir_infer_core::gguf_loader::GgufFile;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let full = args.iter().any(|a| a == "--full");
    let path = args
        .iter()
        .skip(1)
        .find(|a| !a.starts_with('-'))
        .map(PathBuf::from);
    let path = match path {
        Some(p) => p,
        None => {
            eprintln!("usage: i1221_dory_header_probe <model.gguf> [--full]");
            std::process::exit(2);
        }
    };

    let file = GgufFile::open(&path).unwrap_or_else(|e| {
        eprintln!("header parse refused: {e}");
        std::process::exit(1);
    });

    // ── metadata: arch key first, then the writer's ordered KV list ──
    let arch = file
        .metadata
        .get("general.architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("<absent>");
    println!("# dory header probe | arch {arch} | tensors {} | metadata kvs {}",
        file.tensor_infos.len(), file.metadata_order.len());

    println!("\n## general.* keys");
    for (k, v) in &file.metadata_order {
        if k.starts_with("general.") {
            println!("  {k} = {}", render(v));
        }
    }

    println!("\n## non-tokenizer, non-general keys (the loop-count hunt)");
    let mut loopish = 0usize;
    for (k, v) in &file.metadata_order {
        if k.starts_with("tokenizer.") || k.starts_with("general.") {
            continue;
        }
        let lower = k.to_lowercase();
        let hits_loop = ["loop", "recur", "ngpt", "dory", "shared", "cycle"]
            .iter()
            .any(|needle| lower.contains(needle));
        println!("  {}{} = {}", k, if hits_loop { "  [LOOPISH]" } else { "" }, render(v));
        if hits_loop {
            loopish += 1;
        }
    }
    println!("  loopish keys: {loopish}");
    let tokenizer_kvs = file
        .metadata_order
        .iter()
        .filter(|(k, _)| k.starts_with("tokenizer."))
        .count();
    println!("  tokenizer.* kvs: {tokenizer_kvs} (not dumped)");

    if full {
        println!("\n## full tensor inventory (file order)");
        for info in &file.tensor_infos {
            println!(
                "  {} | {:?} | shape {:?} | {} B",
                info.name, info.ggml_type, info.shape, info.byte_len
            );
        }
        if let Some(profile) = file.metadata.get("dory.rope.profile") {
            println!("\n## dory.rope.profile (per-block)");
            if let Some(arr) = profile.as_array() {
                let rendered: Vec<String> = arr.iter().map(render).collect();
                println!("  {}", rendered.join(" "));
            }
        }
        println!("\n## full metadata (file order, tokenizer included)");
        for (k, v) in &file.metadata_order {
            println!("  {k} = {}", render(v));
        }
    }

    // ── tensor inventory: stem grouping (the per-loop vs shared tell) ──
    // A shared recurrent block ships each weight ONCE (stem count == the
    // block's layer span); per-loop weights would repeat per loop k.
    struct Stem {
        count: usize,
        shapes: Vec<String>,
        types: Vec<String>,
        layers: Vec<usize>,
    }
    let mut stems: BTreeMap<String, Stem> = BTreeMap::new();
    let mut layer_idxs: Vec<usize> = Vec::new();
    let mut unlayered: usize = 0usize;
    let mut total_bytes: u64 = 0;

    for info in &file.tensor_infos {
        total_bytes += info.byte_len;
        let (layer, stem) = match strip_layer(&info.name) {
            Some((idx, rest)) => (Some(idx), rest.to_string()),
            None => (None, info.name.clone()),
        };
        match layer {
            Some(idx) => layer_idxs.push(idx),
            None => unlayered += 1,
        }
        let entry = stems.entry(stem).or_insert(Stem {
            count: 0,
            shapes: Vec::new(),
            types: Vec::new(),
            layers: Vec::new(),
        });
        entry.count += 1;
        let shape = format!(
            "{} [{:?}]",
            info.shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(","),
            info.ggml_type
        );
        if !entry.shapes.contains(&shape) {
            entry.shapes.push(shape);
        }
        let ty = format!("{:?}", info.ggml_type);
        if !entry.types.contains(&ty) {
            entry.types.push(ty);
        }
        if let Some(idx) = layer {
            entry.layers.push(idx);
        }
    }

    layer_idxs.sort_unstable();
    let (min_layer, max_layer) = match (layer_idxs.first(), layer_idxs.last()) {
        (Some(a), Some(b)) => (*a, *b),
        _ => (0, 0),
    };
    println!("\n## layer histogram");
    println!("  layered tensors: {} (unlayered {unlayered}), layer span {min_layer}..={max_layer}",
        layer_idxs.len());
    // Density per layer index: gaps expose block boundaries.
    let mut hist: BTreeMap<usize, usize> = BTreeMap::new();
    for idx in &layer_idxs {
        *hist.entry(*idx).or_default() += 1;
    }
    let mut runs: Vec<(usize, usize, usize)> = Vec::new(); // (start, end, tensor count per layer)
    for (idx, n) in &hist {
        match runs.last_mut() {
            Some((_, end, per)) if *end + 1 == *idx && *per == *n => *end = *idx,
            _ => runs.push((*idx, *idx, *n)),
        }
    }
    for (start, end, per) in &runs {
        println!("  layers {start}..={end}: {per} tensors/layer");
    }

    println!("\n## tensor stems (count = per-loop duplication tell; total bytes {total_bytes})");
    for (stem, s) in &stems {
        let layer_note = if s.layers.is_empty() {
            String::from("-")
        } else {
            format!("layers {}..={}", s.layers[0], s.layers[s.layers.len() - 1])
        };
        println!(
            "  {stem}  x{}  {layer_note}  types {:?}  shapes {}",
            s.count,
            s.types,
            s.shapes.join(" | ")
        );
    }

    // ── the tie check: repeated stems byte-identical or divergent? ──
    // THE per-loop-vs-shared decision: a shared recurrent block shipped as
    // duplicated per-slot rows reads byte-identical; a genuinely untied
    // (per-slot) checkpoint diverges. First-mismatch byte offset named.
    println!("\n## tie check (repeated stems, full byte compare)");
    for (stem, s) in &stems {
        if s.count <= 1 || s.layers.is_empty() {
            continue;
        }
        let names: Vec<String> = s
            .layers
            .iter()
            .map(|idx| format!("blk.{idx}.{stem}"))
            .collect();
        let slices: Vec<Option<&[u8]>> =
            names.iter().map(|n| file.tensor_slice(n)).collect();
        if slices.iter().any(|s| s.is_none()) {
            println!("  {stem}: slice read FAILED (byte identity undecidable)");
            continue;
        }
        let base = slices[0].unwrap();
        let mut verdict = String::from("IDENTICAL");
        for (row, other) in slices.iter().enumerate().skip(1) {
            let other = other.unwrap();
            if base.len() != other.len() {
                verdict = format!("DIVERGES (row {row} len {} != {})", other.len(), base.len());
                break;
            }
            if base != other {
                let off = base
                    .iter()
                    .zip(other.iter())
                    .position(|(a, b)| a != b)
                    .unwrap_or(0);
                verdict = format!("DIVERGES (first mismatch at byte {off}, row {row})");
                break;
            }
        }
        println!("  {stem}  x{}  {}  ({} B/row)", s.count, verdict, base.len());
    }

    // ── the T0 verdict lines ──
    println!("\n## T0 verdict inputs");
    println!("  arch key: {arch}");
    println!(
        "  I/R/O partition: explicit-metadata {} / naming-convention only",
        if loopish > 0 { "YES" } else { "NO" }
    );
    let duplicated: Vec<&String> = stems
        .iter()
        .filter(|(_, s)| s.count > 1 && !s.layers.is_empty())
        .map(|(k, _)| k)
        .collect();
    if duplicated.is_empty() {
        println!("  per-loop weights: NONE — every layered stem appears once (shared-block shape)");
    } else {
        println!("  per-loop weights: {} stem(s) repeat: {:?}", duplicated.len(), duplicated);
    }
}

/// Strip the `blk.N.` prefix, returning (layer index, remainder).
fn strip_layer(name: &str) -> Option<(usize, &str)> {
    let after = name.strip_prefix("blk.")?;
    let dot = after.find('.')?;
    let idx: usize = after[..dot].parse().ok()?;
    Some((idx, &after[dot + 1..]))
}

fn render(v: &riir_infer_core::gguf_loader::GgufValue) -> String {
    use riir_infer_core::gguf_loader::GgufValue;
    match v {
        GgufValue::Array(arr) => {
            if arr.len() > 8 {
                format!("[{} items]", arr.len())
            } else {
                let items: Vec<String> = arr.iter().map(render).collect();
                format!("[{}]", items.join(", "))
            }
        }
        GgufValue::String(s) => {
            if s.len() > 80 {
                format!("\"{}…\" ({} B)", &s[..80], s.len())
            } else {
                format!("\"{s}\"")
            }
        }
        other => format!("{other:?}"),
    }
}
