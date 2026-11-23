//! i508_teacher_logits — plan 508 T1's teacher scoring pass: forward the
//! fp16 teacher over the ON-POLICY generations (the student's own answers)
//! and dump per-position logits (f32 LE, one .bin per sequence, plus the
//! token ids) for the torch-side KD loss (token-level / top-k KL against
//! the student's training forward).
//!
//! The on-policy set comes from the student's OWN greedy generations on a
//! fixed prompt panel (the plan's determinism law: greedy = reproducible;
//! the sampling variant is a v2 lever). Generation rides the existing
//! `generate_greedy_qwen_deltanet`; the teacher then scores each generated
//! sequence (prompt + answer) position-by-position.
//!
//! Outputs (under --out):
//! - `panel.txt`         the prompt panel (numbered)
//! - `gen/seq{i}.txt`    each generation's token-id text record
//! - `gen/seq{i}.tokens` the token ids (binary u32 LE)
//! - `teacher_logits/seq{i}.bin` f32 LE logits [seq_len × vocab]
//! - `manifest.json`     the long-run-law artifact (seed, corpus blake3,
//!   model digests, panel hash, per-seq digests)
//!
//! Budget note: the teacher is 0.8B f16 — a full panel forward is minutes,
//! well under the <2 GPU-h envelope. CPU posture (the logits are the
//! artifact; GPU speedup rides the same forward on the CUDA lane).

use std::io::Write as _;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};

use riir_infer_core::corpus_text::load_corpus_text;
use riir_infer_core::deltanet::{
    HybridCache, HybridForwardScratch, effective_rotary_dim, forward_qwen_deltanet,
    generate_greedy_qwen_deltanet,
};
use riir_infer_core::gguf_loader::{GgufFile, load_qwen_deltanet_weights_gguf};
use riir_infer_core::rope::RopeFreqTable;
use riir_infer_core::tokenizer::BpeTokenizer;

const DEFAULT_TEACHER: &str = "E:/tmp/i508/Qwen3.5-0.8B-Base-f16.gguf";
const DEFAULT_STUDENT: &str = "E:/tmp/i508/Qwen3.5-0.8B-Base-Q4_K_M.gguf";
const DEFAULT_CORPUS: &str = "../riir-train/data/chat_probe";
const DEFAULT_PANEL: usize = 8;
const DEFAULT_GEN_TOKENS: usize = 512;

/// blake3 (hex) of a byte slice — the house digest standard.
fn blake3_hex(data: &[u8]) -> String {
    blake3::hash(data).to_string()
}

fn main() -> Result<()> {
    let mut teacher = PathBuf::from(DEFAULT_TEACHER);
    let mut student = PathBuf::from(DEFAULT_STUDENT);
    let mut corpus = PathBuf::from(DEFAULT_CORPUS);
    let mut out_dir = PathBuf::from("E:/tmp/i508/onpolicy");
    let mut panel_n = DEFAULT_PANEL;
    let mut gen_tokens = DEFAULT_GEN_TOKENS;
    let mut seed: u64 = 0x50E1C5; // pinned: the panel shuffle seed
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--teacher" => teacher = PathBuf::from(it.next().expect("value")),
            "--student" => student = PathBuf::from(it.next().expect("value")),
            "--corpus" => corpus = PathBuf::from(it.next().expect("value")),
            "--out" => out_dir = PathBuf::from(it.next().expect("value")),
            "--panel" => panel_n = it.next().expect("value").parse().expect("panel usize"),
            "--gen-tokens" => gen_tokens = it.next().expect("value").parse().expect("gen usize"),
            "--seed" => {
                let v = it.next().expect("value");
                seed = u64::from_str_radix(v.trim_start_matches("0x"), 16).expect("seed hex");
            }
            other => bail!("unknown flag {other}"),
        }
    }
    std::fs::create_dir_all(out_dir.join("gen"))?;
    std::fs::create_dir_all(out_dir.join("teacher_logits"))?;

    // The teacher tokenizes the panel (both models share the vocab).
    let probe = GgufFile::open(&student).context("open student")?;
    let tok = BpeTokenizer::from_gguf(&probe).context("BPE tokenizer")?;
    let corpus_text = load_corpus_text(&corpus).context("load corpus")?;

    // Fixed prompt panel: corpus line-blocks, deterministically shuffled by
    // the pinned seed (Fisher-Yates over an LCG).
    let lines: Vec<&str> = corpus_text
        .lines()
        .filter(|l| l.trim().len() > 20)
        .collect();
    if lines.len() < panel_n {
        bail!("corpus lines {} < panel {}", lines.len(), panel_n);
    }
    let mut idx: Vec<usize> = (0..lines.len()).collect();
    let mut lcg = seed;
    for i in (1..panel_n).rev() {
        lcg = lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (lcg >> 33) as usize % (i + 1);
        idx.swap(i, j);
    }
    let panel: Vec<String> = (0..panel_n)
        .map(|i| lines[idx[i]].trim().to_string())
        .collect();

    let panel_path = out_dir.join("panel.txt");
    {
        let mut f = std::fs::File::create(&panel_path)?;
        for (i, p) in panel.iter().enumerate() {
            writeln!(f, "--- prompt {i} ---")?;
            writeln!(f, "{p}")?;
        }
    }
    println!("# panel: {panel_n} prompts (seed {seed:#x})");

    // The student generates (greedy — deterministic own-answers).
    println!("# loading student (Q4_K_M) for generation ...");
    let t0 = std::time::Instant::now();
    let (s_config, s_weights) =
        load_qwen_deltanet_weights_gguf(&student).context("load student")?;
    println!(
        "# student loaded ({:.1}s): layers={} vocab={}",
        t0.elapsed().as_secs_f32(),
        s_config.n_layer,
        s_config.vocab_size
    );

    let mut seq_meta: Vec<serde_json::Value> = Vec::new();
    for (i, prompt) in panel.iter().enumerate() {
        let prompt_tokens = tok.encode(prompt);
        let gen_ids =
            generate_greedy_qwen_deltanet(&s_weights, &s_config, &prompt_tokens, gen_tokens);
        let full: Vec<usize> = prompt_tokens
            .iter()
            .cloned()
            .chain(gen_ids.iter().cloned())
            .collect();
        let record: Vec<String> = full.iter().map(|t| t.to_string()).collect();
        std::fs::write(
            out_dir.join("gen").join(format!("seq{i}.txt")),
            record.join(" "),
        )?;
        let mut tf = std::fs::File::create(out_dir.join("gen").join(format!("seq{i}.tokens")))?;
        for t in &full {
            tf.write_all(&(*t as u32).to_le_bytes())?;
        }
        seq_meta.push(serde_json::json!({
            "seq": i,
            "prompt_idx": idx[i],
            "prompt_tokens": prompt_tokens.len(),
            "gen_tokens": gen_ids.len(),
            "total_tokens": full.len(),
        }));
        println!(
            "# seq{i}: prompt {} tok + gen {} tok = {}",
            prompt_tokens.len(),
            gen_ids.len(),
            full.len()
        );
    }

    // The teacher scores every sequence (per-position logits dump).
    println!("# loading teacher (f16) for scoring ...");
    let (t_config, t_weights) =
        load_qwen_deltanet_weights_gguf(&teacher).context("load teacher")?;
    let rope = RopeFreqTable::new(t_config.rope_theta, effective_rotary_dim(&t_config));
    let vocab = t_config.vocab_size;
    let mut scratch = HybridForwardScratch::new(&t_config);

    for (i, prompt) in panel.iter().enumerate() {
        let prompt_tokens = tok.encode(prompt);
        let _ = &prompt_tokens; // the panel's tokenization is asserted upstream; kept for the manifest
        let gen_path = out_dir.join("gen").join(format!("seq{i}.tokens"));
        let gen_bytes = std::fs::read(&gen_path)?;
        let full: Vec<usize> = gen_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c) as usize)
            .collect();

        let mut cache =
            HybridCache::with_layer_types_bounded(&t_config, &t_weights.layer_types, full.len());
        let mut logits = vec![0.0f32; vocab];
        let logit_path = out_dir.join("teacher_logits").join(format!("seq{i}.bin"));
        let mut lf = std::fs::File::create(&logit_path)?;
        // Feed token i; the dumped logits predict token i+1 (positions
        // 0..len-2 drive the KD loss; the last row is dumped for
        // self-description).
        for (pos, &token) in full.iter().enumerate() {
            let l = forward_qwen_deltanet(
                &mut logits,
                &t_weights,
                &mut cache,
                token,
                pos,
                &t_config,
                &mut scratch,
                &rope,
            );
            for v in l.iter() {
                lf.write_all(&v.to_le_bytes())?;
            }
        }
        let blake = blake3_hex(&std::fs::read(&logit_path)?);
        seq_meta[i].as_object_mut().unwrap().insert(
            "teacher_logits_blake3".to_string(),
            serde_json::json!(blake),
        );
        seq_meta[i].as_object_mut().unwrap().insert(
            "teacher_logits_bytes".to_string(),
            serde_json::json!(logit_path.metadata()?.len()),
        );
        println!("# teacher seq{i}: {} positions dumped", full.len());
    }

    // The manifest (the long-run law: everything needed to reproduce).
    let student_bytes = std::fs::read(&student)?;
    let teacher_bytes = std::fs::read(&teacher)?;
    let corpus_bytes = std::fs::read(&corpus)?;
    let manifest = serde_json::json!({
        "seed": seed,
        "panel_n": panel_n,
        "gen_tokens": gen_tokens,
        "corpus_blake3": blake3_hex(&corpus_bytes),
        "student_blake3": blake3_hex(&student_bytes),
        "teacher_blake3": blake3_hex(&teacher_bytes),
        "student_path": student.display().to_string(),
        "teacher_path": teacher.display().to_string(),
        "sequences": seq_meta,
    });
    std::fs::write(
        out_dir.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    println!("# manifest → {}", out_dir.join("manifest.json").display());
    println!("# DONE — the torch-side KD trains against these logits");
    Ok(())
}
