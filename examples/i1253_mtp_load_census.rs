//! Issue 1253 T2 — the `-mtp` GGUF load census.
//!
//! Classifies our loader's behavior on the MTP-variant GGUFs (the
//! `-mtp` twins carry PrismML's MTP draft-head tensors, which the Bonsai-2
//! format never had). Two refusal classes are in play:
//! 1. `GgufFile::open` validates EVERY tensor's ggml type id eagerly during
//!    the header walk (`unknown GGML type {id} for tensor '{name}'`) — a new
//!    MTP-only type id refuses here, loudly, naming the tensor;
//! 2. the weight load is pull-based (`load_qwen_deltanet_ternary_weights_gguf`
//!    requests known names only, never iterates the directory) — unknown
//!    tensor NAMES are structurally skipped, never refused.
//!
//! Outcome is typed either way: the census prints the draft-head tensor
//! inventory (names/types/shapes of every tensor OUTSIDE the known
//! `blk.*`/embedding/norm families) and the full ternary weight load's
//! verdict. Never a silent partial: the run ends `CENSUS OK` or fails loud.
//!
//! ```sh
//! cargo run --release --features bonsai2_hadamard \
//!   --example i1253_mtp_load_census -- ../riir-train/data/Ternary-Bonsai-2.5-27B-PQ2_0-mtp.gguf
//! ```

use std::path::PathBuf;

use riir_infer_core::gguf_loader::{load_qwen_deltanet_ternary_weights_gguf, GgufFile};

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("../riir-train/data/Ternary-Bonsai-2.5-27B-PQ2_0-mtp.gguf"));

    // ── Half 1: the eager header walk (ggml-type refusal class) ──────────
    let gguf = GgufFile::open(&path).map_err(|e| {
        anyhow::anyhow!("CENSUS REFUSAL (open, typed): {e:#}")
    })?;
    println!(
        "open OK: gguf v{} · {} tensors · {} metadata keys",
        gguf.version,
        gguf.tensor_infos.len(),
        gguf.metadata_order.len(),
    );

    // ── Half 2: the draft-head inventory (everything outside the known
    //    families the pull-based loader can request) ─────────────────────
    let known = |n: &str| {
        n.starts_with("blk.")
            || n == "token_embd.weight"
            || n == "token_embd_norm.weight"
            || n == "output.weight"
            || n == "output_norm.weight"
            || n.starts_with("v.") // vision tower (mmproj-class files only)
    };
    let mut foreign = Vec::new();
    let arch = gguf.metadata_string("general.architecture").unwrap_or("qwen35").to_owned();
    // The MTP head rides the nextn convention (Issue 742 T9.2 precedent):
    // block_count INCLUDES the draft blocks (65 = 64 main + 1 nextn here),
    // and the draft tensors are the trailing `blk.{n_main}.*` (F16, never
    // ternary) + `blk.{n_main}.nextn.{eh_proj,enorm,hnorm,shared_head_norm}`.
    // Foreign-class = any block at or beyond n_main, or outside the standard
    // families (vision tower, exotic prefixes).
    let n_main = config_block_count(&gguf)
        - gguf
            .metadata_u64(&format!("{arch}.nextn_predict_layers"))
            .unwrap_or(0) as usize;
    for t in &gguf.tensor_infos {
        let beyond_main = t.name.starts_with("blk.")
            && t.name[4..]
                .split_once('.')
                .and_then(|(idx, _)| idx.parse::<usize>().ok())
                .is_some_and(|i| i >= n_main);
        if !known(&t.name) || beyond_main {
            foreign.push(t);
        }
    }
    println!(
        "draft-head-class tensors: {} (types: {:?}) — main stack = {n_main} blocks, the trailing draft block is EXCLUDED from the load by nextn (Issue 742 T9.2)",
        foreign.len(),
        {
            let mut types: Vec<String> = foreign.iter().map(|t| format!("{:?}", t.ggml_type)).collect();
            types.sort();
            types.dedup();
            types
        }
    );
    for t in &foreign {
        println!("  {} · {:?} · shape {:?}", t.name, t.ggml_type, t.shape);
    }

    // ── Half 3: the full ternary weight load (pull-based; unknown names
    //    are skipped by construction — this proves it empirically) ───────
    let (config, _weights) = load_qwen_deltanet_ternary_weights_gguf(&path)
        .map_err(|e| anyhow::anyhow!("CENSUS REFUSAL (weight load, typed): {e:#}"))?;
    let nextn_layers = gguf
        .metadata_u64(&format!(
            "{}.nextn_predict_layers",
            gguf.metadata_string("general.architecture").unwrap_or("qwen35")
        ))
        .unwrap_or(0);
    println!(
        "weight load OK: arch {} · blocks {} (nextn_predict_layers {nextn_layers}) · emb {} · vocab {} · rope-dim {:?} · head_dim {}",
        gguf
            .metadata_string("general.architecture")
            .unwrap_or("?"),
        config.n_layer,
        config.n_embd,
        config.vocab_size,
        gguf.metadata_u64(&format!(
            "{}.rope.dimension_count",
            gguf.metadata_string("general.architecture").unwrap_or("qwen35")
        )),
        config.head_dim,
    );
    println!("CENSUS OK");
    Ok(())
}

fn config_block_count(gguf: &GgufFile) -> usize {
    gguf.metadata_u64(&format!(
        "{}.block_count",
        gguf.metadata_string("general.architecture").unwrap_or("qwen35")
    ))
    .unwrap_or(0) as usize
}
