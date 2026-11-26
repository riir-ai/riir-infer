//! i701_precomp_fit — the Issue-701 cell-2 PRODUCER half: per-block
//! pre-compensation fit for the Q8_0 candidate posture, folded INTO the
//! dense weights BEFORE quantization (zero added bandwidth — the deployed
//! tensor stays plain Q8_0), and emitted as a **spliced GGUF**: a byte
//! copy of the input file with the scored families' tensor-data regions
//! replaced by the pre-compensated requant blocks (same type, same shape,
//! same byte length — the header is untouched).
//!
//! The consumer cell is the i1188 harness UNCHANGED: run
//! `si_zero_shot_ppl --family-ablate <family> --family-posture q8` on the
//! baseline GGUF and on the spliced artifact; the paired per-family
//! Δ-NLL/argmax-flip tables decide. Kill bar (riir-train issue 701): tie
//! or any per-family retention regression ⇒ the lane dies here.
//!
//! Fit (GPTQ-class, adapted to the Q8_0 32-block container; the cq3
//! `quantize_residual` / mint `sparsegpt_reconstruct` feedback shape):
//!   per scored tensor: H = mean(x·xᵀ) from the calibration activations at
//!   that tensor's input; damp λ = `--damp`·mean(diag H); per ROW, sweep
//!   the 32-blocks left→right: quantize each block with the incumbent
//!   `quantize_row_q8_0` law, then add the block's quant error into the
//!   NOT-YET-QUANTIZED later columns through the inverse-Hessian columns
//!   (`w[rest] += (q − w)·H⁻¹[blk, rest] / H⁻¹[blk, blk]`), the Frantar
//!   feedback law — with the compensation STAYING IN the weights, so the
//!   container after the sweep is plain dense that quantizes better.
//!   One pass; deterministic order throughout (the Issue-940 law); f64
//!   inverse via Gauss-Jordan with partial pivoting (d ≤ 3584 on the
//!   0.8B), loud dead pivot.
//!
//! Capture: v1 rides the i1189 `closed_form_write` capture plumbing — the
//! `CaptureSink` slabs (the down-proj input `keys` + the post-residual
//! `resids` per layer, whole-prompt prefill) under the `closed_form_write`
//! feature. `resids` IS the residual entering the layer's MLP block, so
//! `gate/up`'s input = `rmsnorm(post_attn_norm, resids)` (the gammas +
//! eps are public) and `down`'s input IS `keys` — the production input
//! distribution for the two MLP roles exactly. The GDN-projection
//! families share the `post_attn` stream position (the bodies' step b/f
//! norm the same residual), so one pass feeds every v1-scored family.
//! The captured slabs are written per-(layer, site) files with a BLAKE3
//! digest pinned beside them; the splice pass refuses a digest mismatch.
//!
//! ```bash
//! cargo run --release --features closed_form_write \
//!     --example i701_precomp_fit -- capture \
//!     --gguf /Users/katopz/git/riir-train/data/Qwen3.5-0.8B-Base-Q8_0.gguf \
//!     --corpus <text> --dir /tmp/i701_cap --tokens 4096
//! cargo run --release --features closed_form_write \
//!     --example i701_precomp_fit -- splice \
//!     --gguf <same.gguf> --cap /tmp/i701_cap --out <artifact.gguf> \
//!     [--families ffn_gate,ffn_up,ffn_down,gdn_qkv,gdn_out,attn_gate] [--damp 0.01]
//! ```
//!
//! Offline tool: zero serving-path surface; never a default feature.
//! Determinism: same model + same capture dir ⇒ byte-identical artifact
//! (fixed sweep order, no RNG anywhere).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use bytemuck::Zeroable;
use riir_infer_core::closed_form_write::{CaptureSink, QwenFactWriteModel};
use riir_infer_core::family_census::Family;
use riir_infer_core::gguf_loader::{GgufFile, load_qwen_deltanet_weights_gguf};
use riir_infer_core::quant::q8kv::{BlockQ8_0, Q8_BLOCK_SIZE, quantize_row_q8_0};
use riir_infer_core::tokenizer::BpeTokenizer;

const DEFAULT_GGUF: &str = "/Users/katopz/git/riir-train/data/Qwen3.5-0.8B-Base-Q8_0.gguf";
const CAP_MAGIC: &[u8; 7] = b"I701CAP";
const CAP_FORMAT_V1: u32 = 1;
const HEADER_BYTES: usize = 20;

/// The two capture roles (v1): `post_attn` = the residual entering the
/// MLP block (feeds gate/up + the GDN-projection families through its
/// norm), `mlp_hidden` = the SwiGLU intermediate (feeds down/out).
const ROLE_POST_ATTN: &str = "post_attn";
const ROLE_MLP_HIDDEN: &str = "mlp_hidden";

fn parse_families(spec: &str) -> Result<Vec<Family>> {
    let mut out = Vec::new();
    for slug in spec.split(',') {
        let slug = slug.trim();
        if slug.is_empty() {
            continue;
        }
        let f = Family::parse(slug)
            .ok_or_else(|| anyhow::anyhow!("unknown family '{slug}' (family_census vocabulary)"))?;
        if family_role(f).is_none() {
            bail!("family '{slug}' is outside the v1 producer scope");
        }
        out.push(f);
    }
    if out.is_empty() {
        out = vec![
            Family::FfnGate,
            Family::FfnUp,
            Family::FfnDown,
            Family::GdnQkv,
            Family::GdnOut,
            Family::AttnGate,
        ];
    }
    Ok(out)
}

/// Which capture role feeds a family's H.
fn family_role(f: Family) -> Option<&'static str> {
    match f {
        Family::FfnGate | Family::FfnUp | Family::GdnQkv | Family::AttnGate => Some(ROLE_POST_ATTN),
        Family::FfnDown | Family::GdnOut => Some(ROLE_MLP_HIDDEN),
        _ => None,
    }
}

fn tensor_suffix(f: Family) -> &'static str {
    match f {
        Family::FfnGate => "ffn_gate",
        Family::FfnUp => "ffn_up",
        Family::FfnDown => "ffn_down",
        Family::GdnQkv => "attn_qkv",
        Family::GdnOut => "ssm_out",
        Family::AttnGate => "attn_gate",
        _ => "",
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

// ── capture ─────────────────────────────────────────────────────────────

fn cmd_capture(argv: &[String]) -> Result<()> {
    let mut gguf_path: Option<PathBuf> = None;
    let mut corpus: Option<PathBuf> = None;
    let mut dir: Option<PathBuf> = None;
    let mut tokens = 4096usize;
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--gguf" => gguf_path = Some(PathBuf::from(&argv[i + 1])),
            "--corpus" => corpus = Some(PathBuf::from(&argv[i + 1])),
            "--dir" => dir = Some(PathBuf::from(&argv[i + 1])),
            "--tokens" => tokens = argv[i + 1].parse()?,
            other => bail!("capture: unknown arg {other}"),
        }
        i += 2;
    }
    let gguf_path = gguf_path.unwrap_or_else(|| PathBuf::from(DEFAULT_GGUF));
    let corpus = corpus.context("capture: --corpus <text> is required")?;
    let dir = dir.context("capture: --dir <out-dir> is required")?;

    let (config, weights) = load_qwen_deltanet_weights_gguf(&gguf_path)
        .with_context(|| format!("load {}", gguf_path.display()))?;
    let gg = GgufFile::open(&gguf_path)?;
    let tok = BpeTokenizer::from_gguf(&gg).context("tokenizer")?;
    drop(gg);
    let text =
        std::fs::read_to_string(&corpus).with_context(|| format!("corpus {}", corpus.display()))?;
    let seq: Vec<u32> = tok.encode(&text).into_iter().map(|t| t as u32).collect();
    if seq.len() < tokens {
        bail!(
            "corpus tokenizes to {} tokens < --tokens {tokens}",
            seq.len()
        );
    }
    let prompt: Vec<u32> = seq[..tokens].to_vec();

    let n = config.n_embd;
    let mlp = config.mlp_hidden;
    let eps = config.rms_norm_eps as f32;

    // One whole-prompt capture prefill through the adapter's sink (the
    // capture-at shape: an empty edited set is refused by bind, so v1
    // binds layer 0 and harvests ALL layers from the sink directly).
    let mut weights = weights;
    {
        let layers_all: Vec<usize> = (0..config.n_layer).collect();
        let mut model = QwenFactWriteModel::bind(&mut weights, &config, vec![0]);
        let tokens_us: Vec<usize> = prompt.iter().map(|&t| t as usize).collect();
        let v = config.vocab_size;
        let mut logits = vec![0.0f32; v];
        model.cache.reset();
        model.prefill_ctx.closed_form_capture = Some(CaptureSink::new(layers_all.clone()));
        model.prefill_ctx.closed_form_overlays = None;
        riir_infer_core::deltanet::prefill_qwen_deltanet_into(
            model.weights,
            model.config,
            &mut model.cache,
            &tokens_us,
            &mut model.scratch,
            &model.rope,
            &mut model.prefill_ctx,
            &mut logits,
            None,
        );
        let Some(sink) = model.prefill_ctx.closed_form_capture.take() else {
            bail!("capture: sink vanished — the prefill never observed a layer");
        };
        // keys[layer] = the down-proj input rows [tokens × mlp];
        // resids[layer] = the post-residual rows [tokens × n].
        if sink.keys.len() != config.n_layer || sink.resids.len() != config.n_layer {
            bail!(
                "capture: sink saw {}/{} layers — the prefill was not whole-prompt",
                sink.keys.len(),
                config.n_layer
            );
        }
        if sink.rows < tokens {
            bail!("capture: sink rows {} < tokens {tokens}", sink.rows);
        }

        std::fs::create_dir_all(&dir)?;
        let mut manifest = String::from("# i701 capture v1\n");
        manifest.push_str(&format!("model: {}\n", gguf_path.display()));
        manifest.push_str(&format!("corpus: {}\n", corpus.display()));
        manifest.push_str(&format!("tokens: {tokens}\n"));
        manifest.push_str(&format!("n_embd: {n} mlp: {mlp}\n"));

        // post_attn rows: rmsnorm(post_attention_norm, resid) per layer.
        // mlp_hidden rows: the down-proj input slab verbatim.
        let mut digest = blake3::Hasher::new();
        digest.update(b"i701-precomp-capture-v1");
        digest.update(&(tokens as u64).to_le_bytes());
        for layer in 0..config.n_layer {
            let resid = &sink.resids[layer];
            let keys = &sink.keys[layer];
            // post_attn file: normed residual, same row count.
            let gamma = &weights.layers[layer].post_attn_norm;
            let path_pa = dir.join(format!("blk{layer:03}_{ROLE_POST_ATTN}.i701cap"));
            write_site(&path_pa, n, tokens, |r, out| {
                norm_row(&resid[r * n..(r + 1) * n], gamma, eps, out);
            })?;
            digest.update(&path_pa.display().to_string().len().to_le_bytes());
            // mlp_hidden file: the keys slab verbatim.
            let path_mh = dir.join(format!("blk{layer:03}_{ROLE_MLP_HIDDEN}.i701cap"));
            std::fs::copy(&path_pa, &path_mh).ok(); // placeholder swap below
            write_site(&path_mh, mlp, tokens, |r, out| {
                out.copy_from_slice(&keys[r * mlp..(r + 1) * mlp]);
            })?;
            manifest.push_str(&format!(
                "blk{layer:03}: {} {}\n",
                path_pa.display(),
                path_mh.display()
            ));
        }
        let digest_hex = hex(digest.finalize().as_bytes());
        manifest.push_str(&format!("digest: {digest_hex}\n"));
        std::fs::write(dir.join("capture_manifest.txt"), manifest)?;
        println!(
            "# capture: {tokens} tokens × {} layers × 2 sites -> {} | digest {digest_hex}",
            config.n_layer,
            dir.display(),
        );
    }
    Ok(())
}

/// RMSNorm exactly as the bodies fold it (rmsnorm_with_gamma_eps shape):
/// `x / sqrt(mean(x²) + eps) * gamma`, f32 sequential.
fn norm_row(row: &[f32], gamma: &[f32], eps: f32, out: &mut [f32]) {
    let mut acc = 0.0f32;
    for v in row {
        acc += v * v;
    }
    let inv = 1.0 / (acc / row.len() as f32 + eps).sqrt();
    for (o, (&v, &g)) in out.iter_mut().zip(row.iter().zip(gamma.iter())) {
        *o = v * inv * g;
    }
}

fn write_site(
    path: &Path,
    d_in: usize,
    rows: usize,
    fill: impl Fn(usize, &mut [f32]),
) -> Result<()> {
    let mut bytes = Vec::with_capacity(HEADER_BYTES + rows * d_in * 4);
    bytes.extend_from_slice(CAP_MAGIC);
    bytes.push(0); // pad to 8
    bytes.extend_from_slice(&CAP_FORMAT_V1.to_le_bytes());
    bytes.extend_from_slice(&(d_in as u32).to_le_bytes());
    bytes.extend_from_slice(&(rows as u32).to_le_bytes());
    let mut row = vec![0.0f32; d_in];
    for r in 0..rows {
        fill(r, &mut row);
        for v in &row {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(path, bytes)?;
    Ok(())
}

fn load_site(path: &Path, d_in: usize) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("capture file {}", path.display()))?;
    if bytes.len() < HEADER_BYTES || &bytes[..7] != CAP_MAGIC {
        bail!("{}: bad capture file", path.display());
    }
    let fmt = u32::from_le_bytes(bytes[8..12].try_into().expect("header"));
    if fmt != CAP_FORMAT_V1 {
        bail!("{}: format {fmt} != v1", path.display());
    }
    let file_d = u32::from_le_bytes(bytes[12..16].try_into().expect("header")) as usize;
    let rows = u32::from_le_bytes(bytes[16..20].try_into().expect("header")) as usize;
    if file_d != d_in {
        bail!("{}: d_in {file_d} != expected {d_in}", path.display());
    }
    let vals: Vec<f32> = bytes[HEADER_BYTES..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| f32::from_le_bytes(c))
        .collect();
    if vals.len() != rows * d_in {
        bail!(
            "{}: {} floats, expected {rows}×{d_in}",
            path.display(),
            vals.len()
        );
    }
    Ok(vals)
}

// ── the fit ─────────────────────────────────────────────────────────────

/// f64 Gauss-Jordan inverse with partial pivoting; loud on a dead pivot.
fn invert_f64(a_in: &[f64], d: usize) -> Result<Vec<f64>> {
    let mut a = a_in.to_vec();
    let mut inv = vec![0.0f64; d * d];
    for i in 0..d {
        inv[i * d + i] = 1.0;
    }
    for col in 0..d {
        // Partial pivot.
        let (mut piv, mut prow) = (a[col * d + col].abs(), col);
        for r in (col + 1)..d {
            let v = a[r * d + col].abs();
            if v > piv {
                piv = v;
                prow = r;
            }
        }
        if piv < 1e-12 {
            bail!("invert: dead pivot at col {col} (H is singular past damping)");
        }
        if prow != col {
            for c in 0..d {
                a.swap(col * d + c, prow * d + c);
                inv.swap(col * d + c, prow * d + c);
            }
        }
        let diag = a[col * d + col];
        for c in 0..d {
            a[col * d + c] /= diag;
            inv[col * d + c] /= diag;
        }
        for r in 0..d {
            if r == col {
                continue;
            }
            let f = a[r * d + col];
            if f == 0.0 {
                continue;
            }
            for c in 0..d {
                a[r * d + c] -= f * a[col * d + c];
                inv[r * d + c] -= f * inv[col * d + c];
            }
        }
    }
    Ok(inv)
}

/// GPTQ feedback quantization of ONE row into Q8_0 32-blocks: sweep the
/// blocks left→right; each block quantizes with the incumbent scale law,
/// then its quant error `(q − w)` distributes into the LATER columns
/// through `Hinv[blk, rest] / Hinv[blk, blk]` — Frantar's law with the
/// compensation staying in the weights. Returns the pre-compensated row.
fn precomp_row(row: &mut [f32], hinv: &[f64], d: usize, blocks: usize) -> u64 {
    let mut feedback = 0u64;
    let mut q = [0.0f32; Q8_BLOCK_SIZE];
    for b in 0..blocks {
        let b0 = b * Q8_BLOCK_SIZE;
        // Quantize THIS block (the values already carry earlier feedback).
        let mut packed = vec![BlockQ8_0::zeroed(); 1];
        quantize_row_q8_0(&row[b0..b0 + Q8_BLOCK_SIZE], &mut packed);
        let scale = f32::from_bits(u32::from(packed[0].d));
        for (m, slot) in q.iter_mut().enumerate() {
            *slot = scale * f32::from(packed[0].qs[m]);
        }
        // The diagonal denominator for this block's first column — the
        // Frantar normalization uses the block's diagonal entries; v1 uses
        // the block-lead column's Hinv diagonal per element (stable, one
        // read per element).
        for m in 0..Q8_BLOCK_SIZE {
            let e = f64::from(q[m]) - f64::from(row[b0 + m]);
            if e == 0.0 {
                continue;
            }
            let den = hinv[(b0 + m) * d + (b0 + m)];
            if den <= 0.0 {
                continue;
            }
            let s = e / den;
            for j in (b0 + Q8_BLOCK_SIZE)..d {
                let u = hinv[(b0 + m) * d + j];
                if u != 0.0 {
                    row[j] += (s * u) as f32;
                    feedback += 1;
                }
            }
        }
    }
    feedback
}

fn cmd_splice(argv: &[String]) -> Result<()> {
    let mut gguf_path: Option<PathBuf> = None;
    let mut cap_dir: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut families_spec = String::new();
    let mut damp = 0.01f64;
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--gguf" => gguf_path = Some(PathBuf::from(&argv[i + 1])),
            "--cap" => cap_dir = Some(PathBuf::from(&argv[i + 1])),
            "--out" => out = Some(PathBuf::from(&argv[i + 1])),
            "--families" => families_spec = argv[i + 1].clone(),
            "--damp" => damp = argv[i + 1].parse()?,
            other => bail!("splice: unknown arg {other}"),
        }
        i += 2;
    }
    let gguf_path = gguf_path.unwrap_or_else(|| PathBuf::from(DEFAULT_GGUF));
    let cap_dir = cap_dir.context("splice: --cap <capture-dir> is required")?;
    let out = out.context("splice: --out <artifact.gguf> is required")?;
    let families = parse_families(&families_spec)?;

    let gguf =
        GgufFile::open(&gguf_path).with_context(|| format!("open {}", gguf_path.display()))?;
    let (config, weights) = load_qwen_deltanet_weights_gguf(&gguf_path)?;
    let n = config.n_embd;
    let mlp = config.mlp_hidden;
    let _ = &weights;

    // Whole-artifact byte buffer (the 0.8B Q8_0 is ~1.7 GB — fits).
    let mut bytes =
        std::fs::read(&gguf_path).with_context(|| format!("read {}", gguf_path.display()))?;

    let t0 = std::time::Instant::now();
    // Layer-kind map: attention layers carry attn_q/k/v/output, GDN layers
    // carry attn_qkv/attn_gate/ssm_out — the census proves it (blk.3 is
    // full-attention on the 0.8B). A GDN-family tensor on an attention
    // layer simply does not exist: skip (counted), never fail.
    let layer_types = weights.layer_types.clone();
    let _ = layer_types;
    let mut skipped_not_gdn = 0usize;
    let mut skipped_width = 0usize;
    let mut spliced_tensors = 0usize;
    for family in &families {
        let role = family_role(*family).expect("v1 families have roles");
        let suffix = tensor_suffix(*family);
        let storage = family.storage_class();
        let _ = storage;
        for layer in 0..config.n_layer {
            let name = format!("blk.{layer}.{suffix}.weight");
            let Some(info) = gguf.tensor_info(&name) else {
                skipped_not_gdn += 1; // the other layer kind owns no such tensor
                continue;
            };
            if info.ggml_type != riir_infer_core::gguf_loader::GgmlType::Q8_0 {
                bail!(
                    "{name}: incumbent type {:?} != Q8_0 (v1 splices Q8_0 files only)",
                    info.ggml_type
                );
            }
            let (d_out, d_in) = match family {
                Family::FfnGate | Family::FfnUp => (mlp, n),
                Family::FfnDown => (n, mlp),
                Family::GdnQkv => {
                    let lkh = config.deltanet_linear_n_heads;
                    let lvh = config.deltanet_linear_n_value_heads;
                    let hd = config.deltanet_linear_head_dim;
                    (lkh * hd + 2 * lvh * hd, n)
                }
                Family::GdnOut => (
                    n,
                    config.deltanet_linear_n_value_heads * config.deltanet_linear_head_dim,
                ),
                Family::AttnGate => (
                    config.deltanet_linear_n_value_heads * config.deltanet_linear_head_dim,
                    n,
                ),
                _ => bail!("unreached"),
            };
            // The mlp_hidden capture width is the MLP width; the gdn_out
            // input width is v_dim — the two coincide on the 0.8B (2048)
            // but the ROLE file is keyed by capture width, so gdn_out only
            // rides mlp_hidden when the widths match; otherwise the family
            // is outside v1's capture coverage (skip, counted).
            let cap_w = if role == ROLE_MLP_HIDDEN { mlp } else { n };
            if d_in != cap_w {
                skipped_width += 1;
                continue;
            }
            let cap_path = cap_dir.join(format!("blk{layer:03}_{role}.i701cap"));
            let cal = load_site(&cap_path, d_in)?;
            let n_cal = cal.len() / d_in;
            if n_cal < 256 {
                bail!(
                    "{name}: only {n_cal} calibration rows (<256) — price the budget first (the cq3 T2a lesson)"
                );
            }

            // H = (1/n_cal)·XᵀX in f64, sequential.
            let mut h = vec![0.0f64; d_in * d_in];
            for xr in cal.chunks_exact(d_in) {
                for r in 0..d_in {
                    let v = f64::from(xr[r]);
                    if v == 0.0 {
                        continue;
                    }
                    let hrow = &mut h[r * d_in..(r + 1) * d_in];
                    for (c, hv) in hrow.iter_mut().enumerate() {
                        *hv += v * f64::from(xr[c]);
                    }
                }
            }
            let inv_n = 1.0 / n_cal as f64;
            for v in &mut h {
                *v *= inv_n;
            }
            // Damp.
            let mut diag_sum = 0.0f64;
            for j in 0..d_in {
                diag_sum += h[j * d_in + j];
            }
            let lam = damp * diag_sum / d_in as f64;
            for j in 0..d_in {
                h[j * d_in + j] += lam;
            }
            let hinv = invert_f64(&h, d_in)?;
            drop(h);

            // The incumbent dense rows (from the file's Q8_0 blocks — the
            // store IS the incumbent; requant errors are what the fit
            // fights).
            let dense = gguf.dequant_f16_to_f32(&name)?;
            debug_assert_eq!(dense.len(), d_out * d_in);
            let mut w = dense;

            // Per-row feedback sweep.
            let blocks = d_in / Q8_BLOCK_SIZE;
            if d_in % Q8_BLOCK_SIZE != 0 {
                bail!("{name}: d_in {d_in} not a multiple of {Q8_BLOCK_SIZE}");
            }
            let mut feedback_total = 0u64;
            for r in 0..d_out {
                let row = &mut w[r * d_in..(r + 1) * d_in];
                feedback_total += precomp_row(row, &hinv, d_in, blocks);
            }

            // Requantize the pre-compensated rows and splice the blocks.
            let nb = d_in / Q8_BLOCK_SIZE;
            let packed = vec![BlockQ8_0::zeroed(); nb];
            let start = info.data_start as usize;
            let len = info.byte_len as usize;
            if len != d_out * nb * std::mem::size_of::<BlockQ8_0>() {
                bail!("{name}: byte len {len} != {} block bytes", d_out * nb * 34);
            }
            let mut new_blocks: Vec<u8> = Vec::with_capacity(len);
            let mut one = vec![BlockQ8_0::zeroed(); nb];
            for r in 0..d_out {
                quantize_row_q8_0(&w[r * d_in..(r + 1) * d_in], &mut one);
                for blk in &one {
                    new_blocks.extend_from_slice(&blk.d.to_le_bytes());
                    for v in &blk.qs {
                        new_blocks.push(*v as u8);
                    }
                }
            }
            bytes[start..start + len].copy_from_slice(&new_blocks);
            spliced_tensors += 1;
            if layer == 0 {
                println!(
                    "# {}: {d_out}×{d_in}, n_cal {n_cal}, feedback {feedback_total} elements moved",
                    name
                );
            }
            let _ = packed;
        }
    }

    std::fs::write(&out, &bytes).with_context(|| format!("write {}", out.display()))?;
    println!(
        "# spliced artifact: {} ({} bytes) in {:.1}s | tensors {spliced_tensors} | skipped (other layer kind) {skipped_not_gdn} | skipped (capture width) {skipped_width} — run the i1188 paired cell next",
        out.display(),
        bytes.len(),
        t0.elapsed().as_secs_f32()
    );
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let Some(mode) = args.get(1).cloned() else {
        eprintln!("usage: i701_precomp_fit capture|splice [...]");
        std::process::exit(2);
    };
    let r = match mode.as_str() {
        "capture" => cmd_capture(&args[2..]),
        "splice" => cmd_splice(&args[2..]),
        other => {
            eprintln!("unknown mode {other}");
            std::process::exit(2);
        }
    };
    if let Err(e) = r {
        eprintln!("[i701_precomp_fit] FAIL: {e:#}");
        std::process::exit(1);
    }
}
