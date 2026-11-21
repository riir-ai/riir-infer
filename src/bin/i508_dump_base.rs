//! i508_dump_base — plan 508 T1's base prep: load the Q4_K_M student GGUF
//! through riir-infer's native dequant (`dequant_tensor_row`), write every
//! tensor to an HF-loadable `model.safetensors` (hand-rolled writer —
//! 8-byte header length + JSON header + raw little-endian F32 bytes; zero
//! new deps) with the GGUF→HF name mapping for the qwen35 architecture.
//!
//! The torch-side LoRA then trains on this EXACT base (the student's true
//! quantized values, dequantized) — never the HF repo's fp16, whose weights
//! are the teacher's. The delta the KD teaches must live on the student's
//! own bytes.
//!
//! Totality contract: every tensor must map; an unmapped name REFUSES (a
//! missing layer in the LoRA base would silently train a partial model).

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};

use riir_infer_core::gguf_loader::GgufFile;

const DEFAULT_STUDENT: &str = "E:/tmp/i508/Qwen3.5-0.8B-Base-Q4_K_M.gguf";
const DEFAULT_OUT: &str = "E:/tmp/i508/lora_base";

/// GGUF → HF name map for the qwen35 (llama.cpp naming) hybrid family.
/// Exact mappings only; anything outside REFUSES.
fn hf_name(gguf_name: &str) -> Option<String> {
    if gguf_name == "token_embd.weight" {
        return Some("model.embed_tokens.weight".to_string());
    }
    if gguf_name == "output_norm.weight" {
        return Some("model.norm.weight".to_string());
    }
    if gguf_name == "output.weight" {
        return Some("lm_head.weight".to_string());
    }
    let rest = gguf_name.strip_prefix("blk.")?;
    let (idx, tail) = rest.split_once('.')?;
    let i: usize = idx.parse().ok()?;
    let hf = match tail {
        "attn_q.weight" => format!("model.layers.{i}.self_attn.q_proj.weight"),
        "attn_k.weight" => format!("model.layers.{i}.self_attn.k_proj.weight"),
        "attn_v.weight" => format!("model.layers.{i}.self_attn.v_proj.weight"),
        "attn_output.weight" => format!("model.layers.{i}.self_attn.o_proj.weight"),
        "attn_norm.weight" => format!("model.layers.{i}.input_layernorm.weight"),
        "ffn_norm.weight" => format!("model.layers.{i}.post_attention_layernorm.weight"),
        "ffn_gate.weight" => format!("model.layers.{i}.mlp.gate_proj.weight"),
        "ffn_up.weight" => format!("model.layers.{i}.mlp.up_proj.weight"),
        "ffn_down.weight" => format!("model.layers.{i}.mlp.down_proj.weight"),
        // The GDN/linear-attention layers (the hybrid family): mapped so the
        // dump is total; the LoRA targets stay the projections per the plan.
        "ssm_alpha.weight" => format!("model.layers.{i}.linear_attn.ssm_alpha.weight"),
        "ssm_beta.weight" => format!("model.layers.{i}.linear_attn.ssm_beta.weight"),
        "ssm_conv.weight" => format!("model.layers.{i}.linear_attn.conv1d.weight"),
        "ssm_dt.weight" => format!("model.layers.{i}.linear_attn.dt.weight"),
        "ssm_dt.bias" => format!("model.layers.{i}.linear_attn.dt_bias.weight"),
        // The full hybrid vocabulary (the first dump run enumerated the
        // missing names — the totality contract at work):
        "ssm_out.weight" => format!("model.layers.{i}.linear_attn.out_proj.weight"),
        "ssm_norm.weight" => format!("model.layers.{i}.linear_attn.norm.weight"),
        "ssm_conv1d.weight" => format!("model.layers.{i}.linear_attn.conv1d.weight"),
        "ssm_a.weight" => format!("model.layers.{i}.linear_attn.ssm_a.weight"),
        "attn_gate.weight" => format!("model.layers.{i}.self_attn.gate.weight"),
        "attn_qkv.weight" => format!("model.layers.{i}.self_attn.qkv_proj.weight"),
        "post_attention_norm.weight" => {
            format!("model.layers.{i}.post_attention_norm.weight")
        }
        "attn_q_norm.weight" => format!("model.layers.{i}.self_attn.q_norm.weight"),
        "attn_k_norm.weight" => format!("model.layers.{i}.self_attn.k_norm.weight"),
        // The bare A-parameter vector (F32 [16], no .weight suffix in the
        // GGUF naming): kept under its own name.
        "ssm_a" => format!("model.layers.{i}.linear_attn.ssm_a"),
        _ => return None,
    };
    Some(hf)
}

/// Hand-rolled safetensors writer: header JSON (name → dtype/shape/offsets),
/// padded to 8 bytes, then raw F32 LE in offset order.
fn write_safetensors(
    path: &std::path::Path,
    tensors: &BTreeMap<String, (Vec<u64>, Vec<u8>)>,
) -> Result<()> {
    let mut header: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    let mut offset: u64 = 0;
    let mut ordered: Vec<&String> = tensors.keys().collect();
    ordered.sort();
    for name in &ordered {
        let (shape, bytes) = &tensors[*name];
        header.insert(
            (*name).clone(),
            serde_json::json!({
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [offset, offset + bytes.len() as u64],
            }),
        );
        offset += bytes.len() as u64;
    }
    header.insert(
        "__metadata__".to_string(),
        serde_json::json!({"format": "pt"}),
    );
    let mut header_bytes = serde_json::to_vec(&header)?;
    while header_bytes.len() % 8 != 0 {
        header_bytes.push(b' ');
    }
    let mut f = std::fs::File::create(path)?;
    f.write_all(&(header_bytes.len() as u64).to_le_bytes())?;
    f.write_all(&header_bytes)?;
    for name in &ordered {
        f.write_all(&tensors[*name].1)?;
    }
    f.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    let mut student = PathBuf::from(DEFAULT_STUDENT);
    let mut out_dir = PathBuf::from(DEFAULT_OUT);
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--out" => out_dir = PathBuf::from(it.next().expect("--out value")),
            other => student = PathBuf::from(other),
        }
    }
    std::fs::create_dir_all(&out_dir)?;

    let file = GgufFile::open(&student).context("open student GGUF")?;
    let infos = &file.tensor_infos;
    println!("# student: {} tensors", infos.len());

    let mut dumped = 0usize;
    let mut refused: Vec<String> = Vec::new();
    let mut out_tensors: BTreeMap<String, (Vec<u64>, Vec<u8>)> = BTreeMap::new();

    for info in infos {
        // The MTP head (blk.24.nextn.*) rides the quantized file but is
        // structurally excluded from the KD forward (verdict duty (i)) —
        // the LoRA base drops it explicitly, never silently.
        if info.name.starts_with("blk.24.") {
            println!("# skipped MTP-head tensor: {}", info.name);
            continue;
        }
        let hf = match hf_name(&info.name) {
            Some(h) => h,
            None => {
                refused.push(info.name.clone());
                continue;
            }
        };
        // Dequant on the FLATTENED tensor: k-quant blocks span 256
        // CONTIGUOUS elements, and some hybrid tensors (ssm_alpha/beta at
        // [1024, 16]) have last-dims narrower than a block — the row walk
        // would demand a 256-multiple of 16 and panic (measured). GGUF
        // stores the tensor contiguously, so the whole tensor is one
        // "row" of its total element count; shape comes from tensor_info.
        let dims = &info.shape;
        let total: usize = dims.iter().product();
        let (rows, row_len) = (1usize, total);
        let mut flat = Vec::with_capacity(rows * row_len);
        for r in 0..rows {
            let row = file.dequant_tensor_row(&info.name, r, row_len).with_context(|| {
                format!(
                    "dequant {} ({:?}, type {:?}) row {r}/{rows} row_len {row_len}",
                    info.name, dims, info.ggml_type
                )
            })?;
            flat.extend_from_slice(&row);
        }
        let expected = rows * row_len;
        if flat.len() != expected {
            bail!(
                "{}: dequant produced {} elements, expected {expected}",
                info.name,
                flat.len()
            );
        }
        let shape: Vec<u64> = info.shape.iter().map(|&v| v as u64).collect();
        let bytes: Vec<u8> = flat.iter().flat_map(|v| v.to_le_bytes()).collect();
        out_tensors.insert(hf, (shape, bytes));
        dumped += 1;
        if dumped % 50 == 0 {
            println!("# dumped {dumped}/{}", infos.len());
        }
    }
    println!(
        "# dumped: {dumped} | refused (unmapped names): {}",
        refused.len()
    );
    for r in refused.iter().take(12) {
        println!("#   unmapped: {r}");
    }
    if !refused.is_empty() {
        bail!(
            "{} tensor names unmapped — the dump is total by contract; extend the map",
            refused.len()
        );
    }

    let out_path = out_dir.join("model.safetensors");
    write_safetensors(&out_path, &out_tensors).context("write safetensors")?;
    let meta = std::fs::metadata(&out_path)?;
    println!(
        "# wrote {} ({} bytes) — the LoRA base is the student's own dequantized weights",
        out_path.display(),
        meta.len()
    );
    Ok(())
}
