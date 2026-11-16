//! `cq3_quantize` — the CQ3 quantize driver (issue 1186 T2b ours half,
//! plan 629). Two subcommands, the matched-calibration harness's both
//! halves:
//!
//! - `capture` — boot the league dense forward, ingest the calibration
//!   corpus in the league's 512-token windows, tap the 11 scored matrix
//!   classes' input rows into per-(layer, site) `.cq3cap` files + the
//!   `ledger.json` matched-budget row.
//! - `quantize` — source GGUF + capture dir → per tensor:
//!   host-dequant → `fit_smooth` (α = 0.25, the T2a-pinned fold) →
//!   `quantize_residual` (g = 32, b = 3 — the pre-registered bpw match
//!   vs Q3_K ~3.44, ours b_eff 3.5) → packed `.cq3` container v1 +
//!   BLAKE3 manifest sidecar.
//!
//! Determinism (the g1f class): same model + same capture dir →
//! bit-identical artifact — sequential folds, fixed iteration order, the
//! codebook reconstructed from (g, b) (its seeded derivation is pinned by
//! the g1f fixture), no parallelism in the sweep.
//!
//! Hash law: this repo pins with BLAKE3 (the workspace rule); the corpora
//! carry plan-629 sha256 pins verified by the opponent prep script — the
//! ours-side ledger records the corpus BLAKE3 beside the run's own
//! config, and the artifact carries its BLAKE3 manifest sidecar.
//!
//! Posture: feature `cq3_calibrate` (implies `cq3`); never in the release
//! set before the T3 GOAT (the opt-in law). The capture pass runs the
//! REAL league forward (the promoted posture) so the captured rows are
//! the serving distribution — the Issue-1124 T1.5 engagement law, at the
//! data side.
//!
//! Heavy-phase cost note (the plan's "post-walk quiet window"): the
//! per-tensor Hessian accumulation is O(n_cal · d_in²) sequential f32
//! (the Issue-940 determinism law forbids a parallel fold) — the
//! mlp_down tensors dominate (~minutes each at n_cal 2048). Run the
//! quantize pass on a quiet box, serialized behind any benchmark.

#![cfg(feature = "cq3_calibrate")]

use std::collections::BTreeMap;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use riir_infer_core::gguf_loader::GgufFile;
use riir_infer_gpu::qwen38_cq3::{fit_smooth, quantize_residual};
use riir_infer_gpu::qwen38_cq3_capture::{
    CQ3CAP_FORMAT_V1, CQ3CAP_MAGIC, CaptureLedgerMeta, CaptureSession,
};

const MAGIC: &[u8; 7] = b"CQ3ART1";
const FORMAT_V1: u32 = 1;
/// The pre-registered recipe pins (plan 629): g = 32, b = 3 → b_eff 3.5
/// vs Q3_K ≈ 3.44 (+1.7% heaviest, the conservative direction).
const DEFAULT_G: usize = 32;
const DEFAULT_B: u32 = 3;
/// The T2a-pinned SmoothQuant fold strength.
const DEFAULT_ALPHA: f32 = 0.25;
/// The league's ingest window (the plan's chunking law).
const CHUNK: usize = 512;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let Some(mode) = args.get(1).cloned() else {
        usage();
        std::process::exit(2);
    };
    let r = match mode.as_str() {
        "capture" => cmd_capture(&args[2..]),
        "quantize" => cmd_quantize(&args[2..]),
        _ => {
            usage();
            std::process::exit(2);
        }
    };
    if let Err(e) = r {
        eprintln!("[cq3_quantize] FAIL: {e}");
        std::process::exit(1);
    }
}

fn usage() {
    eprintln!(
        "usage:\n  cq3_quantize capture <model.gguf> <corpus.txt> --dir <out-dir> [--tokens 2048] [--carry]\n  cq3_quantize quantize <model.gguf> <capture-dir> --out <artifact.cq3> [--g 32] [--b 3] [--alpha 0.25]"
    );
}

struct Flags {
    positional: Vec<String>,
    named: BTreeMap<String, String>,
}

/// Tiny arg walker: `--key value` pairs + bare positionals; `--carry` is
/// a valueless flag. Unknown keys fail loud (the never-a-silent-pass law).
fn parse_args(argv: &[String], keys: &[&str]) -> Result<Flags, String> {
    let mut positional = Vec::new();
    let mut named = BTreeMap::new();
    let mut i = 0;
    while i < argv.len() {
        let a = &argv[i];
        if a == "--carry" {
            named.insert("carry".into(), "1".into());
            i += 1;
        } else if let Some(k) = a.strip_prefix("--") {
            let key = keys
                .iter()
                .copied()
                .find(|c| *c == k)
                .ok_or_else(|| format!("unknown flag {a}"))?;
            let v = argv
                .get(i + 1)
                .filter(|v| !v.starts_with("--"))
                .ok_or_else(|| format!("flag {a} needs a value"))?;
            named.insert(key.to_string(), v.clone());
            i += 2;
        } else {
            positional.push(a.clone());
            i += 1;
        }
    }
    Ok(Flags { positional, named })
}

fn opt_u64(f: &Flags, key: &str, default: u64) -> Result<u64, String> {
    match f.named.get(key) {
        Some(v) => v
            .parse()
            .map_err(|_| format!("--{key}: {v} is not a number")),
        None => Ok(default),
    }
}

fn opt_f64(f: &Flags, key: &str, default: f64) -> Result<f64, String> {
    match f.named.get(key) {
        Some(v) => v
            .parse()
            .map_err(|_| format!("--{key}: {v} is not a number")),
        None => Ok(default),
    }
}

fn blake3_of(path: &Path) -> Result<String, String> {
    let mut f =
        std::fs::File::open(path).map_err(|e| format!("blake3 {}: open: {e}", path.display()))?;
    let mut hasher = blake3::Hasher::new();
    std::io::copy(&mut f, &mut hasher)
        .map_err(|e| format!("blake3 {}: read: {e}", path.display()))?;
    Ok(hex_encode(hasher.finalize().as_bytes()))
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// The per-site input dims in [`CAPTURE_SITES`] order, from the forward's
/// config — the same geometry the tap sites read.
fn site_dims(
    cfg: &riir_infer_gpu::qwen38_dense_cubecl::Qwen38WgpuConfig,
) -> Vec<(&'static str, usize)> {
    let e = cfg.n_embd;
    let o_in = cfg.n_head * cfg.head_dim;
    let gdn_out_in = cfg.n_v_heads * cfg.head_v_dim;
    let mlp = cfg.mlp_hidden;
    vec![
        ("attn_qkv_in", e),
        ("attn_o_in", o_in),
        ("gdn_qkvz_in", e),
        ("gdn_out_in", gdn_out_in),
        ("mlp_gateup_in", e),
        ("mlp_down_in", mlp),
    ]
}

// ────────────────────────────── capture ──────────────────────────────

fn cmd_capture(argv: &[String]) -> Result<(), String> {
    let f = parse_args(argv, &["dir", "tokens"])?;
    let [model, corpus] = f.positional.as_slice() else {
        return Err("capture needs <model.gguf> <corpus.txt>".into());
    };
    let dir = f
        .named
        .get("dir")
        .cloned()
        .ok_or("capture needs --dir <out-dir>")?;
    let budget_tokens = opt_u64(&f, "tokens", 2048)? as usize;
    // State posture: fresh per chunk by default (the imatrix-matched
    // window posture); `--carry` runs one continuous stream. Pinned in
    // the ledger either way — never implied.
    let carry = f.named.contains_key("carry");

    let corpus_path = Path::new(corpus);
    let corpus_bytes_len = std::fs::metadata(corpus_path)
        .map_err(|e| format!("corpus: {e}"))?
        .len();
    let corpus_blake3 = blake3_of(corpus_path)?;

    let boot = std::time::Instant::now();
    println!("[cq3-capture] booting {} (chunk {CHUNK})", model);
    let mut fwd = riir_infer_gpu::qwen38_dense_cubecl::Qwen38WgpuForward::new(
        Path::new(model),
        CHUNK * 2, // ctx headroom: only one window is resident at a time
        CHUNK,
    )
    .map_err(|e| format!("boot: {e}"))?;
    let cfg = fwd.config.clone();
    println!(
        "[cq3-capture] booted {:.1}s (layers {}, e {}, mlp {})",
        boot.elapsed().as_secs_f32(),
        cfg.n_layer,
        cfg.n_embd,
        cfg.mlp_hidden,
    );

    // Tokenize with the model's own tokenizer (the league posture).
    let gg = GgufFile::open(Path::new(model)).map_err(|e| format!("reopen: {e}"))?;
    let tok = riir_infer_core::tokenizer::BpeTokenizer::from_gguf(&gg)
        .map_err(|e| format!("tokenizer: {e}"))?;
    drop(gg);
    let text = std::fs::read_to_string(corpus_path).map_err(|e| format!("corpus read: {e}"))?;
    let all: Vec<u32> = tok.encode(&text).into_iter().map(|t| t as u32).collect();
    println!(
        "[cq3-capture] corpus {corpus_bytes_len} B -> {} tokens",
        all.len()
    );
    let n_chunks = budget_tokens / CHUNK;
    if all.len() < n_chunks * CHUNK {
        return Err(format!(
            "corpus tokenizes to {} tokens, need {} for --tokens {budget_tokens}",
            all.len(),
            n_chunks * CHUNK
        ));
    }

    fwd.set_cq3_capture(Some(CaptureSession::with_config(
        PathBuf::from(&dir),
        budget_tokens,
        cfg.n_layer,
        &site_dims(&cfg),
    )));

    let mut pos = 0usize;
    let mut captured = 0usize;
    let t0 = std::time::Instant::now();
    for c in 0..n_chunks {
        let window = &all[c * CHUNK..(c + 1) * CHUNK];
        if !carry && c > 0 {
            fwd.zero_state();
            pos = 0;
        }
        fwd.ingest(window, pos)
            .map_err(|e| format!("chunk {c}: {e}"))?;
        if carry {
            pos += CHUNK;
        }
        captured += CHUNK;
        if (c + 1) % 4 == 0 || c + 1 == n_chunks {
            println!(
                "[cq3-capture] chunk {}/{} ({} tokens, {:.1}s)",
                c + 1,
                n_chunks,
                captured,
                t0.elapsed().as_secs_f32()
            );
        }
    }

    let mut session = fwd.take_cq3_capture().expect("session installed");
    let meta = CaptureLedgerMeta {
        corpus_blake3,
        corpus_bytes: corpus_bytes_len,
        tokens_captured: captured,
        chunk: CHUNK,
        state_per_chunk: if carry { "carry" } else { "fresh" },
    };
    let summary = session.finish(&meta)?;
    let total: usize = summary.iter().map(|(_, r)| r).sum();
    println!(
        "[cq3-capture] DONE: {} files, {} rows total, {:.1}s",
        summary.len(),
        total,
        t0.elapsed().as_secs_f32()
    );
    Ok(())
}

// ────────────────────────────── quantize ──────────────────────────────

/// One artifact tensor record (the serialized subset of `CqTensor` — the
/// rotation + codebook reconstruct from (g, b), the g1f determinism pin).
struct TensorRecord {
    name: String,
    d_out: usize,
    d_in: usize,
    smooth: Option<Vec<f32>>,
    indices: Vec<u8>,
    scales: Vec<u16>,
}

/// The artifact's tensor set: every capture-fed matrix, in layer order
/// (deterministic iteration → bit-identical bytes). Returns
/// (tensor name, d_in, d_out, site).
fn tensor_plan(
    gguf: &GgufFile,
    n_layer: usize,
) -> Result<Vec<(String, usize, usize, &'static str)>, String> {
    let interval = gguf
        .metadata_u64("qwen35.full_attention_interval")
        .unwrap_or(4) as usize;
    let mut plan = Vec::with_capacity(n_layer);
    for i in 0..n_layer {
        let is_attn = (i + 1) % interval == 0;
        let suffixes: &[&str] = if is_attn {
            &[
                "attn_q",
                "attn_k",
                "attn_v",
                "attn_output",
                "ffn_gate",
                "ffn_up",
                "ffn_down",
            ]
        } else {
            &[
                "attn_qkv",
                "attn_gate",
                "ssm_alpha",
                "ssm_beta",
                "ssm_out",
                "ffn_gate",
                "ffn_up",
                "ffn_down",
            ]
        };
        for &suffix in suffixes {
            let name = format!("blk.{i}.{suffix}.weight");
            let Some(info) = gguf.tensor_info(&name) else {
                return Err(format!(
                    "{name}: missing from GGUF (the league lane needs it)"
                ));
            };
            let d_in = info.shape[0];
            let d_out = info.shape.iter().product::<usize>() / d_in;
            // q/k/v share the attention-layer qkv input; gdn qkv/z/alpha/
            // beta share the gdn-layer one; the rest map 1:1.
            let site = match suffix {
                "attn_q" | "attn_k" | "attn_v" => "attn_qkv_in",
                "attn_qkv" | "attn_gate" | "ssm_alpha" | "ssm_beta" => "gdn_qkvz_in",
                "attn_output" => "attn_o_in",
                "ssm_out" => "gdn_out_in",
                "ffn_gate" | "ffn_up" => "mlp_gateup_in",
                "ffn_down" => "mlp_down_in",
                other => return Err(format!("{other}: no capture site")),
            };
            plan.push((name, d_in, d_out, site));
        }
    }
    if plan.is_empty() {
        return Err("tensor plan is empty — no capture-fed matrices found".into());
    }
    Ok(plan)
}

/// Read one `.cq3cap` file (header + rows); asserts the weight's d_in.
fn load_capture(path: &Path, d_in: usize) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() < 16 {
        return Err(format!("{}: truncated header", path.display()));
    }
    if &bytes[..7] != CQ3CAP_MAGIC {
        return Err(format!("{}: bad magic", path.display()));
    }
    let fmt = u32::from_le_bytes(bytes[8..12].try_into().expect("16-byte header"));
    if fmt != CQ3CAP_FORMAT_V1 {
        return Err(format!("{}: format {fmt} != v1", path.display()));
    }
    let file_d_in = u32::from_le_bytes(bytes[12..16].try_into().expect("16-byte header")) as usize;
    if file_d_in != d_in {
        return Err(format!(
            "{}: d_in {file_d_in} != weight's {d_in}",
            path.display()
        ));
    }
    let mut vals: Vec<f32> = bytemuck::cast_slice(&bytes[16..]).to_vec();
    if !vals.len().is_multiple_of(d_in) {
        // A torn final write truncates to the last whole row — the file
        // length is row-pinned by the sink's budget law, so this is
        // defensive only.
        vals.truncate(vals.len() / d_in * d_in);
    }
    Ok(vals)
}

fn cmd_quantize(argv: &[String]) -> Result<(), String> {
    let f = parse_args(argv, &["out", "g", "b", "alpha"])?;
    let [model, capdir] = f.positional.as_slice() else {
        return Err("quantize needs <model.gguf> <capture-dir>".into());
    };
    let out = f.named.get("out").cloned().ok_or("quantize needs --out")?;
    let g = opt_u64(&f, "g", DEFAULT_G as u64)? as usize;
    let b = opt_u64(&f, "b", DEFAULT_B as u64)? as u32;
    let alpha = opt_f64(&f, "alpha", DEFAULT_ALPHA as f64)? as f32;

    let gguf = GgufFile::open(Path::new(model)).map_err(|e| format!("open: {e}"))?;
    let block_count = gguf
        .metadata_u64("qwen35.block_count")
        .ok_or("missing qwen35.block_count")? as usize;
    let nextn = gguf
        .metadata_u64("qwen35.nextn_predict_layers")
        .unwrap_or(0) as usize;
    let n_layer = block_count - nextn;

    let plan = tensor_plan(&gguf, n_layer)?;
    println!(
        "[cq3-quantize] {} tensors, g={g} b={b} alpha={alpha} (b_eff {:.3})",
        plan.len(),
        b as f64 + 16.0 / g as f64
    );

    let mut records = Vec::with_capacity(plan.len());
    let t0 = std::time::Instant::now();
    for (idx, (name, d_in, d_out, site)) in plan.iter().enumerate() {
        let li: usize = name
            .split('.')
            .nth(1)
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| format!("{name}: no layer index"))?;
        let cap_path = PathBuf::from(capdir).join(format!("blk{li:03}_{site}.cq3cap"));
        if !cap_path.exists() {
            return Err(format!("{}: capture file missing", cap_path.display()));
        }
        let cal = load_capture(&cap_path, *d_in)?;
        let n_cal = cal.len() / d_in;
        if n_cal == 0 {
            return Err(format!("{}: empty capture", cap_path.display()));
        }
        let w = gguf
            .dequant_f16_to_f32(name)
            .map_err(|e| format!("{name}: dequant: {e}"))?;
        if w.len() != d_out * d_in {
            return Err(format!(
                "{name}: dequant {} floats, expected {} ({d_out}×{d_in})",
                w.len(),
                d_out * d_in
            ));
        }
        let smooth = fit_smooth(&w, *d_out, *d_in, &cal, n_cal, alpha);
        let tensor = quantize_residual(&w, *d_out, *d_in, g, b, Some(&smooth), &cal, n_cal);
        records.push(TensorRecord {
            name: name.clone(),
            d_out: *d_out,
            d_in: *d_in,
            smooth: tensor.smooth,
            indices: tensor.indices,
            scales: tensor.scales,
        });
        if (idx + 1) % 8 == 0 || idx + 1 == plan.len() {
            println!(
                "[cq3-quantize] {}/{} {name} ({d_out}×{d_in}, n_cal {n_cal}, {:.1}s cum)",
                idx + 1,
                plan.len(),
                t0.elapsed().as_secs_f32()
            );
        }
    }

    // Serialize container v1: prologue + per-tensor records. Field order
    // is the wire — never reordered without a format bump.
    let path = PathBuf::from(&out);
    let file = std::fs::File::create(&path).map_err(|e| format!("out: {e}"))?;
    let mut w = BufWriter::with_capacity(1 << 22, file);
    let emit = |w: &mut BufWriter<std::fs::File>| -> std::io::Result<()> {
        w.write_all(MAGIC)?;
        w.write_all(&[0u8; 1])?;
        w.write_all(&FORMAT_V1.to_le_bytes())?;
        w.write_all(&(g as u32).to_le_bytes())?;
        w.write_all(&b.to_le_bytes())?;
        w.write_all(&alpha.to_bits().to_le_bytes())?;
        w.write_all(&0u32.to_le_bytes())?; // reserved
        w.write_all(&(records.len() as u32).to_le_bytes())?;
        for r in &records {
            w.write_all(&(r.name.len() as u32).to_le_bytes())?;
            w.write_all(r.name.as_bytes())?;
            w.write_all(&(r.d_out as u32).to_le_bytes())?;
            w.write_all(&(r.d_in as u32).to_le_bytes())?;
            match &r.smooth {
                Some(s) => {
                    w.write_all(&[1u8])?;
                    for v in s {
                        w.write_all(&v.to_le_bytes())?;
                    }
                }
                None => w.write_all(&[0u8])?,
            }
            w.write_all(&(r.indices.len() as u64).to_le_bytes())?;
            w.write_all(&r.indices)?;
            w.write_all(&(r.scales.len() as u64).to_le_bytes())?;
            for s in &r.scales {
                w.write_all(&s.to_le_bytes())?;
            }
        }
        Ok(())
    };
    emit(&mut w).map_err(|e| format!("write: {e}"))?;
    w.flush().map_err(|e| format!("flush: {e}"))?;
    let file = w.into_inner().map_err(|e| format!("into_inner: {e}"))?;
    file.sync_all().map_err(|e| format!("fsync: {e}"))?;

    let digest = blake3_of(&path)?;
    let sidecar = PathBuf::from(format!("{out}.blake3"));
    std::fs::write(&sidecar, format!("{digest}  {}\n", path.display()))
        .map_err(|e| format!("sidecar: {e}"))?;

    println!(
        "[cq3-quantize] DONE: {} ({:.2} GiB), blake3 {digest}, {:.1}s",
        path.display(),
        (path.metadata().map(|m| m.len()).unwrap_or(0) as f64) / (1 << 30) as f64,
        t0.elapsed().as_secs_f32()
    );
    Ok(())
}
