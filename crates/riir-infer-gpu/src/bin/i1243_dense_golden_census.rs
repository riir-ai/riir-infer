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
        i += 1;
        let mut next = || {
            let v = argv
                .get(i)
                .map(|s| s.to_string())
                .expect("missing value after flag");
            i += 1;
            v
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
        args.gguf,
        args.label,
        args.corpus,
        args.ctx_len,
        args.chunk,
        args.stride,
        args.max_positions
    );

    let gg = riir_infer_core::gguf_loader::GgufFile::open(PathBuf::from(&args.gguf).as_path())
        .unwrap_or_else(|e| panic!("[i1243] gguf open: {e}"));
    let tok = BpeTokenizer::from_gguf(&gg).unwrap_or_else(|e| panic!("[i1243] tokenizer: {e}"));
    let vocab = tok.vocab_size();
    let text = load_corpus_text(PathBuf::from(&args.corpus).as_path())
        .unwrap_or_else(|e| panic!("[i1243] corpus: {e}"));
    let ids: Vec<u32> = tok.encode(&text).into_iter().map(|t| t as u32).collect();
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

    let mut session = Qwen38WgpuForward::new(
        PathBuf::from(&args.gguf).as_path(),
        args.ctx_len,
        args.chunk,
    )
    .unwrap_or_else(|e| panic!("[i1243] session: {e}"));

    let mut scratch = GoldenNllScratch::with_vocab(vocab);

    // The scored reading at one teacher-forced position.
    // `argmax` rides the row (Issue 1186 T3-prep (c)): paired runs — one
    // unarmed, one CQ3-armed — diff their argmax ids into the retention
    // gate's flip census (overall + per-family via the family mask), and
    // their pooled golden NLLs into Δ-NLL. The bin stays single-checkpoint;
    // the paired diff is the cell, not this bin.
    struct Row {
        pos: usize,
        nll: f64,
        inc_nll: f64,
        top2_margin: f64,
        argmax: u32,
    }

    let mut rows: Vec<Row> = Vec::new();
    let mut scored = 0usize;

    if args.stride > 1 {
        // Issue 1245: the dense lane's chunked-prefill tail-logits
        // materialization is GARBAGE on every posture/chunk size (filed
        // with the reproduction) — the chunk posture reads it and would
        // silently quote garbage readings. Refuse loud until the fix
        // lands; the census runs the decode posture (--stride 1).
        panic!(
            "[i1243] the chunk posture is refused: the dense lane's prefill tail-logits \
             read is garbage (riir-infer issue 1245) — run --stride 1 (decode path, sane)"
        );
    }

    if args.stride == 0 {
        panic!("[i1243] --stride 0 is invalid");
    }

    {
        // Decode posture (stride 1): per-token, per-position scoring —
        // the posture the i1245 control proved sane.
        session
            .ingest(&ids[..1], 0)
            .unwrap_or_else(|e| panic!("[i1243] seed: {e}"));
        let decode_limit = ids
            .len()
            .min(args.max_positions + 1)
            .min(args.ctx_len.saturating_sub(1));
        for t in 1..decode_limit {
            let (argmax, logits, _caps) = session
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
            rows.push(Row {
                pos: t - 1,
                nll: g.nll,
                inc_nll,
                top2_margin: g.top2_margin,
                argmax,
            });
            scored += 1;
        }
    }

    assert!(
        scored > 0,
        "no positions scored (corpus too short for the stride)"
    );

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
        // Issue 1186 T3-prep (c) — the retention run's provenance rides the
        // saved JSON: the CQ3 arming (if any) and the family mask (if any)
        // are named beside the readings, so a paired diff can verify BOTH
        // runs' postures came from the same build + corpus.
        let cq3_env = std::env::var("RIIR_QWEN38_CQ3").unwrap_or_default();
        let cq3_families_env = std::env::var("RIIR_QWEN38_CQ3_FAMILIES").unwrap_or_default();
        out.push_str(&format!(
            ",\"cq3_artifact\":\"{}\",\"cq3_families\":\"{}\"",
            cq3_env, cq3_families_env
        ));
        out.push_str(&format!(
            "\",\"stride\":{},\"chunk\":{},\"scored\":{},\"digest\":\"{digest:016x}\",\"\
             argmax_ids_saved\":true,\"rows\":[",
            args.stride, args.chunk, scored
        ));
        for (i, r) in rows.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&format!(
                "{{\"pos\":{},\"nll\":{:.17e},\"inc\":{:.17e},\"margin\":{:.17e},\"argmax\":{}}}",
                r.pos, r.nll, r.inc_nll, r.top2_margin, r.argmax
            ));
        }
        out.push_str("]}\n");
        std::fs::write(save, out)
            .unwrap_or_else(|e| panic!("[i1243] save {}: {e}", save.display()));
        println!(
            "[i1243] readings saved to {} (argmax ids per position — the paired flip census reads them)",
            save.display()
        );
    }
}
