//! Issue 1243 T1 remainder — the DENSE-lane golden-NLL census instrument.
//!
//! Wires the existing quant-resident dense Metal lane
//! ([`riir_infer_gpu::qwen38_dense_cubecl::Qwen38WgpuForward`]) into the
//! existing f64 golden oracle ([`riir_infer_core::golden_nll`]) so a
//! dense-quantized qwen35-arch checkpoint (e.g. the QDA-Q4/Q6 released
//! GGUFs) gets the same tail-faithful pooled reading the ternary
//! `si_zero_shot_ppl --golden-nll` lane produces — the lossy-surface
//! law's instrument for the dense family (per-family conditional
//! retention decides adoption; an aggregate never does).
//!
//! **Posture.** Teacher-forced over a corpus: the scored position's
//! target is the corpus's own next token. Default stride = the session
//! chunk (score at every chunk boundary through the fast chunked
//! prefill — logits are read once per chunk by the lane); `--stride 1`
//! decodes per token for per-position scoring. Both postures are
//! deterministic (greedy lane, no RNG; the readings' digest pins the
//! run).
//!
//! **Instrument-grade only** — never a serving path; the forward is
//! untouched, the logits are read.
//!
//! Pairing: run once per checkpoint with `--save`, diff the two JSONs
//! at the record level (pooled Δ-NLL + per-position flips) — the bin
//! stays single-checkpoint (the cell, not the diff, is this bin's job).
//!
//! Whole-file cfg keeps non-feature builds an empty binary; the Cargo
//! [[bin]] required-features row is the reader protection (the P43 law).

#![cfg(feature = "i1243_dense_golden")]

use std::path::PathBuf;

use riir_infer_core::corpus_text::load_corpus_text;
use riir_infer_core::golden_nll::GoldenNllScratch;
use riir_infer_core::tokenizer::BpeTokenizer;
use riir_infer_gpu::qwen38_dense_cubecl::Qwen38WgpuForward;

const DEFAULT_GGUF: &str = "/Users/katopz/uzu27b/models/qwen3.8-27b-Q4_K_M.gguf";
const DEFAULT_CTX: usize = 8192;
const DEFAULT_CHUNK: usize = 512;
const DEFAULT_MAX_POSITIONS: usize = 8192;

struct Args {
    gguf: String,
    corpus: String,
    label: String,
    ctx_len: usize,
    chunk: usize,
    stride: usize,
    max_positions: usize,
    save: Option<PathBuf>,
}

fn parse_args() -> Args {
    let mut a = Args {
        gguf: DEFAULT_GGUF.to_string(),
        corpus: String::new(),
        label: String::new(),
        ctx_len: DEFAULT_CTX,
        chunk: DEFAULT_CHUNK,
        stride: DEFAULT_CHUNK,
        max_positions: DEFAULT_MAX_POSITIONS,
        save: None,
    };
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        let flag = argv[i].clone();
        let mut next = || {
            i += 1;
            argv.get(i)
                .expect("missing value after flag")
                .to_string()
        };
        match flag.as_str() {
            "--gguf" => a.gguf = next(),
            "--corpus" => a.corpus = next(),
            "--label" => a.label = next(),
            "--ctx-len" => a.ctx_len = next().parse().expect("--ctx-len usize"),
            "--chunk" => a.chunk = next().parse().expect("--chunk usize"),
            "--stride" => a.stride = next().parse().expect("--stride usize"),
            "--max-positions" => a.max_positions = next().parse().expect("--max-positions usize"),
            "--save" => a.save = Some(PathBuf::from(next())),
            other => panic!("unknown flag {other}"),
        }
    }
    assert!(!a.corpus.is_empty(), "--corpus is required");
    if a.label.is_empty() {
        a.label = PathBuf::from(&a.gguf)
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| a.gguf.clone());
    }
    a
}

fn main() {
    let args = parse_args();
    println!(
        "[i1243] dense golden census: gguf={} label={} corpus={} ctx={} chunk={} stride={} max_pos={}",
        args.gguf, args.label, args.corpus, args.ctx_len, args.chunk, args.stride, args.max_positions
    );

    let gg = riir_infer_core::gguf_loader::GgufFile::open(PathBuf::from(&args.gguf).as_path())
        .unwrap_or_else(|e| panic!("[i1243] gguf open: {e}"));
    let tok = BpeTokenizer::from_gguf(&gg)
        .unwrap_or_else(|e| panic!("[i1243] tokenizer: {e}"));
    let vocab = tok.vocab_size();
    let text = load_corpus_text(PathBuf::from(&args.corpus).as_path())
        .unwrap_or_else(|e| panic!("[i1243] corpus: {e}"));
    let ids: Vec<u32> = tok
        .encode(&text)
        .into_iter()
        .map(|t| t as u32)
        .collect();
    println!(
        "[i1243] corpus {} bytes -> {} tokens (vocab {vocab})",
        text.len(),
        ids.len()
    );
    assert!(ids.len() > 1, "corpus tokenizes to < 2 tokens");

    // Band thresholds — the same cliff coordinates the si-lane discloses
    // (the fp32 cliff where an fp32 consumer reads false zeros; the f64
    // cliff where even a max-subtract LSE absorbs the tail).
    use katgpt_core::softmax_fix::{FloatFormat, precision_cliff};
    let fp32_cliff = precision_cliff(FloatFormat::Fp32);
    let f64_cliff = precision_cliff(FloatFormat::Fp64);

    let mut session = Qwen38WgpuForward::new(PathBuf::from(&args.gguf).as_path(), args.ctx_len, args.chunk)
        .unwrap_or_else(|e| panic!("[i1243] session: {e}"));

    let mut scratch = GoldenNllScratch::with_vocab(vocab);

    // The scored reading at one teacher-forced position.
    struct Row {
        pos: usize,
        nll: f64,
        inc_nll: f64,
        top2_margin: f64,
    }

    let mut rows: Vec<Row> = Vec::new();
    let mut scored = 0usize;

    if args.stride > 1 {
        // Chunk-boundary posture: ingest whole chunks (the fast chunked
        // prefill reads logits once per chunk), score the chunk's last
        // position against the corpus's next token.
        let mut p = 0usize;
        while p + args.chunk < ids.len() && scored < args.max_positions {
            let base = p;
            session
                .ingest(&ids[p..p + args.chunk], base)
                .unwrap_or_else(|e| panic!("[i1243] ingest @{base}: {e}"));
            let logits = session
                .last_logits()
                .unwrap_or_else(|e| panic!("[i1243] logits @{base}: {e}"));
            let target = ids[p + args.chunk] as usize;
            let g = scratch.read_target(&logits, target);
            // The incumbent f64 max-subtract form — the same comparator
            // the si-lane's disagreement census carries.
            let mut max_l = f32::NEG_INFINITY;
            for &l in &logits {
                max_l = max_l.max(l);
            }
            let mut sum = 0.0f64;
            for &l in &logits {
                sum += (f64::from(l) - f64::from(max_l)).exp();
            }
            let inc_nll = f64::from(max_l) + sum.ln() - f64::from(logits[target]);
            rows.push(Row { pos: p + args.chunk - 1, nll: g.nll, inc_nll, top2_margin: g.top2_margin });
            scored += 1;
            p += args.chunk;
            if args.stride > args.chunk {
                // Stride > chunk: skip ahead without scoring (the session
                // state is cumulative; ingest advances it regardless).
                let skip = args.stride - args.chunk;
                if p + skip < ids.len() {
                    session
                        .ingest(&ids[p..p + skip], p)
                        .unwrap_or_else(|e| panic!("[i1243] skip-ingest @{p}: {e}"));
                    p += skip;
                }
            }
        }
    } else {
        // Per-position posture: decode one token at a time, score every
        // position against the corpus's next token.
        session.ingest(&ids[..1], 0).unwrap_or_else(|e| panic!("[i1243] seed: {e}"));
        for t in 1..ids.len().min(args.max_positions + 1) {
            let (_argmax, logits, _caps) = session
                .forward_token_capture(ids[t - 1], t - 1, &[])
                .unwrap_or_else(|e| panic!("[i1243] decode @{t}: {e}"));
            let target = ids[t] as usize;
            let g = scratch.read_target(&logits, target);
            let mut max_l = f32::NEG_INFINITY;
            for &l in &logits {
                max_l = max_l.max(l);
            }
            let mut sum = 0.0f64;
            for &l in &logits {
                sum += (f64::from(l) - f64::from(max_l)).exp();
            }
            let inc_nll = f64::from(max_l) + sum.ln() - f64::from(logits[target]);
            rows.push(Row { pos: t - 1, nll: g.nll, inc_nll, top2_margin: g.top2_margin });
            scored += 1;
        }
    }

    assert!(scored > 0, "no positions scored (corpus too short for the stride)");

    // The census table — pooled golden reading + the confidence-band
    // split at the fp32/f64 cliffs (the shapes the si-lane prints, so
    // dense-vs-ternary records stay comparable) + the digest identity.
    let mut golden_sum = 0.0f64;
    let mut inc_sum = 0.0f64;
    let mut margin_sum = 0.0f64;
    let mut confident = 0usize;
    let mut golden_confident_sum = 0.0f64;
    let mut deep = 0usize;
    let mut golden_deep_sum = 0.0f64;
    let mut disagrees = 0usize;
    let mut max_abs_delta = 0.0f64;
    for r in &rows {
        golden_sum += r.nll;
        inc_sum += r.inc_nll;
        margin_sum += r.top2_margin;
        if r.top2_margin >= fp32_cliff {
            confident += 1;
            golden_confident_sum += r.nll;
        }
        if r.top2_margin >= f64_cliff {
            deep += 1;
            golden_deep_sum += r.nll;
        }
        let d = (r.nll - r.inc_nll).abs();
        if d > 0.0 {
            disagrees += 1;
        }
        if d > max_abs_delta {
            max_abs_delta = d;
        }
    }
    let n = rows.len() as f64;
    // The digest identity: an FNV-1a fold over the f64 bit patterns of
    // the per-position golden NLLs (std-only; the corpus path + position
    // count + pooled value carry the rest of the identity).
    let mut digest: u64 = 0xcbf2_9ce4_8422_2325;
    for r in &rows {
        for b in r.nll.to_bits().to_le_bytes() {
            digest ^= b as u64;
            digest = digest.wrapping_mul(0x1000_001b_0000_0001);
        }
    }
    println!(
        "[i1243] pooled golden NLL/pos {:.12} (incumbent form {:.12}, Δ oracle-vs-incumbent {:.3e})",
        golden_sum / n,
        inc_sum / n,
        (golden_sum - inc_sum) / n
    );
    println!(
        "[i1243] mean top-2 margin {:.4} nats | confident(≥fp32 cliff {fp32_cliff:.2}) {confident}/{} sum {:.6} | deep(≥f64 cliff {f64_cliff:.2}) {deep}/{deep_total} sum {golden_deep_sum:.6}",
        margin_sum / n,
        rows.len(),
        golden_confident_sum,
        deep_total = rows.len(),
    );
    println!(
        "[i1243] oracle-vs-incumbent disagreement census: {disagrees}/{} positions differ, max|Δ| {:.3e}",
        rows.len(),
        max_abs_delta
    );
    println!(
        "[i1243] digest nll_fold={digest:016x} scored={scored} corpus_bytes={} label={}",
        text.len(),
        args.label
    );

    if let Some(save) = &args.save {
        let mut out = String::from("{\"label\":\"");
        out.push_str(&args.label);
        out.push_str("\",\"gguf\":\"");
        out.push_str(&args.gguf);
        out.push_str("\",\"corpus\":\"");
        out.push_str(&args.corpus);
        out.push_str(&format!(
            "\",\"stride\":{},\"chunk\":{},\"scored\":{},\"digest\":\"{digest:016x}\",\"rows\":[",
            args.stride, args.chunk, scored
        ));
        for (i, r) in rows.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&format!(
                "{{\"pos\":{},\"nll\":{:.17e},\"inc\":{:.17e},\"margin\":{:.17e}}}",
                r.pos, r.nll, r.inc_nll, r.top2_margin
            ));
        }
        out.push_str("]}\n");
        std::fs::write(save, out)
            .unwrap_or_else(|e| panic!("[i1243] save {}: {e}", save.display()));
        println!("[i1243] readings saved to {}", save.display());
    }
}
