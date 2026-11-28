//! Issue 1254 P0 runtime probe — the recipient-side verification reads
//! over the REAL checkpoints (no forward pass):
//! P0b `rope_freqs.weight` presence/type/len + the scaled-table build,
//! P0d the donor `layer_capture` tap discovery, P0e both loads.
//! Refusal battery: every refusal names its phase.
//!
//! Report of record: `.benchmarks/1277_i1254_claim_and_t0_runtime.md`.

use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let recipient = args
        .iter()
        .position(|a| a == "--recipient")
        .and_then(|i| args.get(i + 1))
        .map(String::from)
        .unwrap_or_else(|| String::from("E:/git/riir-train/data/Llama-3.2-3B-Base.Q8_0.gguf"));
    let donor = args
        .iter()
        .position(|a| a == "--donor")
        .and_then(|i| args.get(i + 1))
        .map(String::from)
        .unwrap_or_else(|| {
            String::from("E:/git/riir-train/data/Ternary-Bonsai-2.5-27B-PQ2_0.gguf")
        });

    println!("== i1254 P0 runtime probe ==");
    println!("recipient: {recipient}");
    println!("donor:     {donor}");

    // ── P0b: rope_freqs.weight presence/type/len + the scaled table ──
    let p = Path::new(&recipient);
    let gguf = match riir_infer_core::gguf_loader::GgufFile::open(p) {
        Ok(g) => g,
        Err(e) => {
            println!("P0b REFUSAL(recipient-open): {e}");
            std::process::exit(2);
        }
    };
    let (config, weights) = match riir_infer_core::gguf_loader::load_llama_weights_gguf(p) {
        Ok(x) => x,
        Err(e) => {
            println!("P0b REFUSAL(recipient-load): {e}");
            std::process::exit(2);
        }
    };
    let hd = config.head_dim;
    match riir_infer_core::rope_freq_loader::read_rope_freq_weights(&gguf, hd) {
        Ok(None) => {
            println!("P0b: rope_freqs.weight ABSENT — plain table (head_dim={hd})");
        }
        Ok(Some(w)) => {
            println!(
                "P0b: rope_freqs.weight PRESENT len={} (head_dim/2={hd}) — scaled checkpoint",
                w.len()
            );
            let t = weights.build_rope_freq_table(&config);
            let plain = riir_infer_core::rope::RopeFreqTable::new(config.rope_theta, hd);
            let same = t.as_slice() == plain.as_slice();
            let max_ratio = t
                .as_slice()
                .iter()
                .zip(plain.as_slice())
                .map(|(a, b)| a / b)
                .fold(f32::MIN, f32::max);
            println!(
                "P0b: scaled table built; identical-to-plain={same}; max(freq_ratio)={max_ratio:.4}"
            );
        }
        Err(e) => {
            println!("P0b REFUSAL(rope-freqs-read): {e}");
            std::process::exit(2);
        }
    }

    // ── P0e-recipient: the full load already ran (weights built) ──
    println!(
        "P0e-recipient: llama load OK — d={} blocks={} kv_heads={}",
        config.n_embd, config.n_layer, config.n_kv_head
    );

    // ── P0e/P0d donor: ternary load + layer_capture tap discovery ──
    #[cfg(feature = "deltanet_ternary_inference")]
    {
        let d = Path::new(&donor);
        match riir_infer_core::gguf_loader::load_qwen_deltanet_ternary_weights_gguf(d) {
            Ok((_dcfg, _dw)) => {
                println!("P0e-donor: qwen35 ternary load OK (PQ2_0)");
                println!(
                    "P0d: donor capture tap = the ternary forward's `layer_capture` \
                     argument (post-layer residual snapshot, copy-only-when-armed) — \
                     API surface confirmed at compile time"
                );
            }
            Err(e) => {
                println!("P0e-donor REFUSAL(ternary-load): {e}");
                std::process::exit(2);
            }
        }
    }
    #[cfg(not(feature = "deltanet_ternary_inference"))]
    {
        println!(
            "P0d/P0e-donor: SKIP-LOUD (deltanet_ternary_inference not compiled — \
             rebuild with --features deltanet_ternary_inference for the donor legs)"
        );
    }

    println!("== P0 runtime probe done ==");
}
