//! The hermetic fixture gates (Issue 1251 T2/T3 G1) — the
//! [`trigger_vote_fixture`] shape for the PLANTED-fixture lane:
//! feature-gated (not test-gated) so the `trigger-vote --self-test` bin
//! posture drives the SAME gate source the cargo tests assert; an
//! unarmed build compiles it to nothing.
//!
//! The gates run the WHOLE chain on tiny deterministic GGUF files the
//! fixture writes itself (the minimal writer below): plan → materialize
//! (real Q8_0 in-place requant) → detector → baselines → pinned bar.
//! SplitMix64 weights; no external data; every temp file lives under a
//! per-process temp dir (the shared-temp law) and is removed at the
//! end of the battery.

use crate::quant::q8kv::{BlockQ8_0, Q8_BLOCK_SIZE, quantize_row_q8_0};
use crate::trigger_vote::{GgufVoteSource, VoteReader, trigger_vote_with_reader};
use crate::trigger_vote_cell::{
    PlantConfig, PlantManifest, PlantMode, manifest_path, materialize, patch_file, plan,
};
use crate::trigger_vote_cell_score::{CellOutcome, baseline_rankings, score_cell};
use anyhow::{Context, Result, bail, ensure};
use std::path::PathBuf;

/// SplitMix64 (the house deterministic fixture RNG; local copy).
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64, tag: u64) -> Self {
        Self(seed ^ tag.wrapping_mul(0xD1B54A32D192ED03))
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn f32v(&mut self) -> f32 {
        let v = (self.next() >> 11) as f64 / (1u64 << 53) as f64;
        (v * 2.0 - 1.0) as f32
    }
}

/// Fixture dims (small but above every vote minimum; n_embd/n_ff
/// multiples of 32 — the Q8_0 block law).
const LAYERS: usize = 2;
const FF: usize = 128;
const EMBD: usize = 32;
const VOCAB: usize = 1024;
/// The friendly-donor tokens (the fixture builds three down columns as
/// near-copies of these embedding rows, so the Upstream donor scan finds
/// them deterministically).
const FRIENDLY_TOKENS: [usize; 3] = [50, 90, 130];
/// Block width per friendly token (EMBD / 3 = the disjoint dim blocks
/// that make the friendly rows exactly orthogonal — the T1 fixture's
/// block-source construction: cross-terms are exactly zero and the vote
/// concentration is analytic, never seed luck).
const BLOCK: usize = EMBD / 3;

struct TinyTensor {
    name: String,
    shape: Vec<usize>,
    data: Vec<u8>,
}

/// Quantize f32 values into Q8_0 wire bytes (the house quantizer).
fn q8_vals(vals: &[f32]) -> Vec<u8> {
    let nb = vals.len() / Q8_BLOCK_SIZE;
    let mut blocks = vec![
        BlockQ8_0 {
            d: 0,
            qs: [0i8; Q8_BLOCK_SIZE]
        };
        nb
    ];
    quantize_row_q8_0(vals, &mut blocks);
    let mut out = Vec::with_capacity(nb * 34);
    for b in &blocks {
        out.extend_from_slice(&b.d.to_le_bytes());
        out.extend_from_slice(bytemuck::bytes_of(&b.qs));
    }
    out
}

/// Write a minimal GGUF v3 file (no metadata KVs; 32-byte data
/// alignment — the loader's default). Q8_0 tensors only.
fn write_tiny_gguf(path: &std::path::Path, tensors: &[TinyTensor]) -> Result<()> {
    let mut out: Vec<u8> = Vec::with_capacity(4096);
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes()); // no metadata KVs
    let mut offsets = Vec::with_capacity(tensors.len());
    let mut off: u64 = 0;
    for t in tensors {
        offsets.push(off);
        // name: u64 len + bytes (no NUL — the GGUF v3 convention).
        out.extend_from_slice(&(t.name.len() as u64).to_le_bytes());
        out.extend_from_slice(t.name.as_bytes());
        out.extend_from_slice(&(t.shape.len() as u32).to_le_bytes());
        for d in &t.shape {
            out.extend_from_slice(&(*d as u64).to_le_bytes());
        }
        out.extend_from_slice(&crate::gguf_loader::GgmlType::Q8_0.id().to_le_bytes());
        out.extend_from_slice(&off.to_le_bytes());
        let bytes = t.data.len() as u64;
        off += bytes;
        // Align every tensor offset to 32 B (llama.cpp behavior).
        off = off.div_ceil(32) * 32;
    }
    // Pad to the data section (32-byte alignment from the header end).
    while !out.len().is_multiple_of(32) {
        out.push(0);
    }
    for (i, t) in tensors.iter().enumerate() {
        while !out.len().is_multiple_of(32) {
            out.push(0);
        }
        debug_assert_eq!(out.len() as u64 % 32, offsets[i] % 32);
        out.extend_from_slice(&t.data);
    }
    std::fs::write(path, &out).with_context(|| format!("writing {}", path.display()))
}

/// The deterministic tiny base checkpoint (owns the file; Drop
/// removes it).
struct TinyBase(std::path::PathBuf);

impl Drop for TinyBase {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
        let _ = std::fs::remove_file(manifest_path(&self.0));
    }
}

/// Build the tiny base: random weights + three friendly down columns
/// aligned with the friendly tokens' embedding rows.
fn tiny_base(dir: &std::path::Path, tag: u64) -> Result<TinyBase> {
    let mut rng = SplitMix64::new(0xC10DD00D, tag);
    let mut emb = vec![0f32; VOCAB * EMBD];
    for v in emb.iter_mut() {
        *v = rng.f32v();
    }
    // The friendly rows: each concentrated on its OWN dim block (value
    // tuned to the typical random-row norm, so they sit INSIDE the
    // normal-token domain — the domain argmax must find them). Exactly
    // orthogonal by construction; every direction built from one of
    // them reads the others as exactly zero.
    for (j, &tok) in FRIENDLY_TOKENS.iter().enumerate() {
        let row = &mut emb[tok * EMBD..(tok + 1) * EMBD];
        for v in row.iter_mut() {
            *v = 0.0;
        }
        for v in row[j * BLOCK..(j + 1) * BLOCK].iter_mut() {
            *v = 1.0; // norm √10 ≈ the random-row norm family
        }
    }
    let mut gate = Vec::with_capacity(LAYERS);
    let mut up = Vec::with_capacity(LAYERS);
    for _ in 0..LAYERS {
        let mut g = vec![0f32; FF * EMBD];
        for v in g.iter_mut() {
            *v = rng.f32v();
        }
        let mut u = vec![0f32; FF * EMBD];
        for v in u.iter_mut() {
            *v = rng.f32v();
        }
        gate.push(g);
        up.push(u);
    }
    // Down matrices: layer 0 random; layer 1 carries the friendly
    // columns at neurons 0..3.
    let mut down0 = vec![0f32; EMBD * FF];
    for v in down0.iter_mut() {
        *v = rng.f32v();
    }
    let mut down1 = vec![0f32; EMBD * FF];
    for v in down1.iter_mut() {
        *v = rng.f32v();
    }
    for (j, &tok) in FRIENDLY_TOKENS.iter().enumerate() {
        // unit(e_tok) scaled to a plausible column norm — the friendly
        // block rows are already exactly orthogonal to each other, so
        // the donor scan accepts all three before any random column.
        let row = &emb[tok * EMBD..(tok + 1) * EMBD];
        let mut s: f64 = 0.0;
        for &x in row {
            s += (x as f64) * (x as f64);
        }
        let n = s.sqrt() as f32;
        let scale = 2.0; // ~the random columns' norm scale
        for r in 0..EMBD {
            down1[r * FF + j] = row[r] / n * scale;
        }
    }
    // Q8_0 tensors ([n_embd × n_ff]: ne[0] = row length, ne[1] = rows).
    let mut tensors = Vec::with_capacity(3 + 4 * LAYERS);
    tensors.push(TinyTensor {
        name: "token_embd.weight".into(),
        shape: vec![EMBD, VOCAB],
        data: q8_vals(&emb),
    });
    for l in 0..LAYERS {
        tensors.push(TinyTensor {
            name: format!("blk.{l}.ffn_gate.weight"),
            shape: vec![EMBD, FF],
            data: q8_vals(&gate[l]),
        });
        tensors.push(TinyTensor {
            name: format!("blk.{l}.ffn_up.weight"),
            shape: vec![EMBD, FF],
            data: q8_vals(&up[l]),
        });
        let d = if l == 1 { &down1 } else { &down0 };
        tensors.push(TinyTensor {
            name: format!("blk.{l}.ffn_down.weight"),
            // GGUF down convention: ne[0] = n_ff (row length = the
            // input/neuron dim; EMBD rows of length FF) — what the
            // planter's `down[r * n_ff + c]` indexing reads.
            shape: vec![FF, EMBD],
            data: q8_vals(d),
        });
    }
    let path = dir.join(format!("tiny_base_{tag}.gguf"));
    write_tiny_gguf(&path, &tensors)?;
    Ok(TinyBase(path))
}

/// The per-process, per-call scratch dir (the shared-temp law: unique
/// per invocation — the lib-test and the fixture's own test run this
/// battery CONCURRENTLY in one binary, so a pid-scoped dir would race
/// on the shared file names).
fn scratch() -> PathBuf {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let d = std::env::temp_dir().join(format!("t1251_cell_{}_{n}", std::process::id()));
    std::fs::create_dir_all(&d).expect("scratch dir");
    d
}

fn planted_path(dir: &std::path::Path, name: &str) -> PathBuf {
    dir.join(format!("{name}_{}.gguf", std::process::id()))
}

fn remove_artifacts(p: &std::path::Path) {
    let _ = std::fs::remove_file(p);
    let _ = std::fs::remove_file(manifest_path(p));
}

/// Open a planted pair as vote sources.
fn open_pair(
    base: &std::path::Path,
    planted: &std::path::Path,
) -> Result<(GgufVoteSource<'static>, GgufVoteSource<'static>)> {
    // Leak the mmaps for the battery's lifetime (a fixture-lane
    // convenience; the real cell path borrows properly). The files are
    // removed in Drop — the maps stay valid because the OS keeps the
    // mapping alive after unlink on macOS/Linux.
    let b: &'static crate::gguf_loader::GgufFile =
        Box::leak(Box::new(crate::gguf_loader::GgufFile::open(base)?));
    let f: &'static crate::gguf_loader::GgufFile =
        Box::leak(Box::new(crate::gguf_loader::GgufFile::open(planted)?));
    let bs = GgufVoteSource::open(b, base)?;
    let fs = GgufVoteSource::open(f, planted)?;
    // Transmute the lifetime away for the fixture harness only.
    let bs: GgufVoteSource<'static> = unsafe { std::mem::transmute(bs) };
    let fs: GgufVoteSource<'static> = unsafe { std::mem::transmute(fs) };
    Ok((bs, fs))
}

/// Run an Upstream-mode cell end to end on the tiny base under an
/// explicit vote reader; returns the cell report + the manifest.
fn run_upstream_cell(
    base: &TinyBase,
    dir: &std::path::Path,
    reader: VoteReader,
    cfg_overrides: impl FnOnce(&mut PlantConfig),
) -> Result<(crate::trigger_vote_cell_score::CellReport, PlantManifest)> {
    let gguf = crate::gguf_loader::GgufFile::open(&base.0)?;
    let mut cfg = PlantConfig {
        mode: PlantMode::Upstream,
        layer: 1,
        triggers: 3,
        fan_per_trigger: 8,
        noise_neurons: 4,
        seed: 0x1251,
    };
    cfg_overrides(&mut cfg);
    let plan = plan(&gguf, &cfg)?;
    let dst = planted_path(dir, "t2b");
    let manifest = materialize(&base.0, &dst, &plan)?;
    let (bs, fs) = open_pair(&base.0, &dst)?;
    let vote = trigger_vote_with_reader(&bs, &fs, reader)?;
    let bl = baseline_rankings(&bs, &fs, &vote)?;
    let rep = score_cell(
        PlantMode::Upstream,
        &manifest.planted_tokens(),
        &vote,
        &bl,
        Some(true),
    );
    let manifest = PlantManifest::load(&manifest_path(&dst))?;
    remove_artifacts(&dst);
    Ok((rep, manifest))
}

// ── Gates ───────────────────────────────────────────────────────────

/// T2b upstream: the whole chain recovers the geometry-chosen tokens —
/// every planted token lands in the detector's top-10 (bar 1). The 2×
/// baseline bar is NOT asserted here: on tiny dims summed-change tracks
/// the vote too closely for the ratio to be meaningful (the bar
/// arithmetic is pinned by the score_cell unit gates; the REAL-bytes
/// cell carries the verdict).
fn gate_t2b_upstream_recovery() -> Result<()> {
    let dir = scratch();
    let base = tiny_base(&dir, 1)?;
    let (rep, _m) = run_upstream_cell(&base, &dir, VoteReader::Raw, |_| {})?;
    ensure!(
        rep.rows.iter().all(|r| r.in_top10),
        "t2b recovery failed (a planted token outside the top-10):\n{}",
        rep.text
    );
    ensure!(
        matches!(rep.outcome, CellOutcome::Pass | CellOutcome::Unproven),
        "t2b cell outcome {}: the chain must recover (Pass) or close honestly \
         (Unproven), never Negative on a well-formed fixture:\n{}",
        "{:?}",
        rep.text
    );
    Ok(())
}

/// T2a circular: the same chain with the disclosed-circular cores —
/// SanityPass (harness sanity, never the deciding cell).
fn gate_t2a_circular_sanity() -> Result<()> {
    let dir = scratch();
    let base = tiny_base(&dir, 2)?;
    let gguf = crate::gguf_loader::GgufFile::open(&base.0)?;
    let cfg = PlantConfig {
        mode: PlantMode::Circular,
        layer: 1,
        triggers: 3,
        fan_per_trigger: 8,
        noise_neurons: 4,
        seed: 0x1251,
    };
    let plan = plan(&gguf, &cfg)?;
    let dst = planted_path(&dir, "t2a");
    let manifest = materialize(&base.0, &dst, &plan)?;
    let (bs, fs) = open_pair(&base.0, &dst)?;
    let vote = trigger_vote_with_reader(&bs, &fs, VoteReader::Raw)?;
    let bl = baseline_rankings(&bs, &fs, &vote)?;
    let rep = score_cell(
        PlantMode::Circular,
        &manifest.planted_tokens(),
        &vote,
        &bl,
        None,
    );
    ensure!(
        rep.outcome == CellOutcome::SanityPass,
        "t2a sanity did not pass:\n{}",
        rep.text
    );
    remove_artifacts(&dst);
    Ok(())
}

/// Determinism: two materializations of the same plan → identical
/// planted bytes + identical manifest bytes.
fn gate_plant_determinism() -> Result<()> {
    let dir = scratch();
    let base = tiny_base(&dir, 3)?;
    let gguf = crate::gguf_loader::GgufFile::open(&base.0)?;
    let cfg = PlantConfig {
        mode: PlantMode::Upstream,
        layer: 1,
        triggers: 3,
        fan_per_trigger: 4,
        noise_neurons: 4,
        seed: 0xABCD,
    };
    let plan = plan(&gguf, &cfg)?;
    let a = planted_path(&dir, "det_a");
    let b = planted_path(&dir, "det_b");
    let ma = materialize(&base.0, &a, &plan)?;
    let mb = materialize(&base.0, &b, &plan)?;
    let ba = std::fs::read(&a)?;
    let bb = std::fs::read(&b)?;
    ensure!(ba == bb, "planted files diverged under the same plan");
    ensure!(
        ma.encode() == mb.encode(),
        "manifests diverged under the same plan"
    );
    remove_artifacts(&a);
    remove_artifacts(&b);
    Ok(())
}

/// Untouched bytes stay identical: the detector's active-row census is
/// EXACTLY the planted condition neurons (fans + noise; the payload
/// edits down only, which the vote never reads).
fn gate_row_census_exact() -> Result<()> {
    let dir = scratch();
    let base = tiny_base(&dir, 4)?;
    let gguf = crate::gguf_loader::GgufFile::open(&base.0)?;
    let cfg = PlantConfig {
        mode: PlantMode::Upstream,
        layer: 1,
        triggers: 3,
        fan_per_trigger: 8,
        noise_neurons: 4,
        seed: 0x1251,
    };
    let plan = plan(&gguf, &cfg)?;
    let dst = planted_path(&dir, "census");
    let manifest = materialize(&base.0, &dst, &plan)?;
    // Every non-edited tensor is byte-identical (empty-patch copy).
    let scratch_copy = planted_path(&dir, "plain_copy");
    patch_file(&base.0, &scratch_copy, &[])?;
    ensure!(
        std::fs::read(&base.0)? == std::fs::read(&scratch_copy)?,
        "the patcher mutated untouched bytes"
    );
    remove_artifacts(&scratch_copy);
    let (bs, fs) = open_pair(&base.0, &dst)?;
    let vote = trigger_vote_with_reader(&bs, &fs, VoteReader::Raw)?;
    let fan = manifest.triggers[0].fan_len as u64;
    let want = manifest.noise_neurons as u64 + manifest.triggers.len() as u64 * fan;
    ensure!(
        vote.n_active_rows == want,
        "active rows {} != the condition census {want}",
        vote.n_active_rows
    );
    ensure!(
        vote.band.unwrap_or((0, 0)).0 == 1,
        "the band did not admit the planted layer"
    );
    remove_artifacts(&dst);
    Ok(())
}

/// Requant roundtrip: the house quantizer's decode error stays inside
/// one half-step of the block scale, and requantizing is deterministic.
fn gate_requant_roundtrip() -> Result<()> {
    let mut rng = SplitMix64::new(0x0F0F, 7);
    let vals: Vec<f32> = (0..Q8_BLOCK_SIZE * 4).map(|_| rng.f32v() * 3.0).collect();
    let bytes = q8_vals(&vals);
    // Deterministic.
    let bytes2 = q8_vals(&vals);
    ensure!(bytes == bytes2, "quantization is not deterministic");
    // Decode error per block: the house quantizer stores the scale as
    // f16, so the dequant scale d_f16 can sit up to half an f16 ulp
    // (2^-11 relative) away from the f32 d the codes were computed
    // against — the envelope is d/2 + 127·d·2^-11, never a bare d/2.
    for (b, blk) in bytes.as_chunks::<34>().0.iter().enumerate() {
        let d = half::f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])).to_f32();
        for j in 0..Q8_BLOCK_SIZE {
            let want = vals[b * Q8_BLOCK_SIZE + j];
            let got = d * (blk[2 + j] as i8 as f32);
            let bound = d * 0.5 + 127.0 * d * (1.0 / 2048.0) + 1e-6;
            ensure!(
                (want - got).abs() <= bound,
                "requant error {want} → {got} exceeds the f16-scale envelope {bound}"
            );
        }
    }
    Ok(())
}

/// Manifest tamper refuses: flipping any body byte breaks the digest.
fn gate_manifest_tamper_refuses() -> Result<()> {
    let dir = scratch();
    let base = tiny_base(&dir, 5)?;
    let gguf = crate::gguf_loader::GgufFile::open(&base.0)?;
    let cfg = PlantConfig {
        mode: PlantMode::Upstream,
        layer: 1,
        triggers: 3,
        fan_per_trigger: 4,
        noise_neurons: 4,
        seed: 0x7777,
    };
    let plan = plan(&gguf, &cfg)?;
    let dst = planted_path(&dir, "tamper");
    let manifest = materialize(&base.0, &dst, &plan)?;
    let mut bytes = manifest.encode();
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0x01;
    ensure!(
        PlantManifest::decode(&bytes).is_err(),
        "a tampered manifest decoded clean"
    );
    remove_artifacts(&dst);
    Ok(())
}

/// Planter self-falsifier: a degenerate embedding table (all rows
/// equal — no geometry to pick tokens from) refuses loud.
fn gate_planter_refuses_degenerate() -> Result<()> {
    let dir = scratch();
    // Build a base whose embedding rows are all IDENTICAL (norm in
    // range): the domain argmax degenerates to id order and the fan
    // margin analysis collapses.
    let mut rng = SplitMix64::new(0xD3AD, 9);
    let mut emb = vec![0f32; VOCAB * EMBD];
    let proto: Vec<f32> = (0..EMBD).map(|_| rng.f32v()).collect();
    for t in 0..VOCAB {
        emb[t * EMBD..(t + 1) * EMBD].copy_from_slice(&proto);
    }
    let mut gate = Vec::with_capacity(LAYERS);
    let mut up = Vec::with_capacity(LAYERS);
    let mut downs = Vec::with_capacity(LAYERS);
    for _ in 0..LAYERS {
        let mut g = vec![0f32; FF * EMBD];
        for v in g.iter_mut() {
            *v = rng.f32v();
        }
        let mut u = vec![0f32; FF * EMBD];
        for v in u.iter_mut() {
            *v = rng.f32v();
        }
        let mut d = vec![0f32; EMBD * FF];
        for v in d.iter_mut() {
            *v = rng.f32v();
        }
        gate.push(g);
        up.push(u);
        downs.push(d);
    }
    let mut tensors = Vec::with_capacity(LAYERS);
    tensors.push(TinyTensor {
        name: "token_embd.weight".into(),
        shape: vec![EMBD, VOCAB],
        data: q8_vals(&emb),
    });
    for l in 0..LAYERS {
        tensors.push(TinyTensor {
            name: format!("blk.{l}.ffn_gate.weight"),
            shape: vec![EMBD, FF],
            data: q8_vals(&gate[l]),
        });
        tensors.push(TinyTensor {
            name: format!("blk.{l}.ffn_up.weight"),
            shape: vec![EMBD, FF],
            data: q8_vals(&up[l]),
        });
        tensors.push(TinyTensor {
            name: format!("blk.{l}.ffn_down.weight"),
            shape: vec![EMBD, FF],
            data: q8_vals(&downs[l]),
        });
    }
    let path = dir.join(format!("degenerate_{}.gguf", std::process::id()));
    write_tiny_gguf(&path, &tensors)?;
    struct D(PathBuf);
    impl Drop for D {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    let _guard = D(path.clone());
    let gguf = crate::gguf_loader::GgufFile::open(&path)?;
    let cfg = PlantConfig {
        mode: PlantMode::Upstream,
        layer: 1,
        triggers: 3,
        fan_per_trigger: 8,
        noise_neurons: 4,
        seed: 0x1251,
    };
    match plan(&gguf, &cfg) {
        Ok(_) => bail!("the degenerate fixture planned clean — the falsifier is dead"),
        Err(_) => Ok(()),
    }
}

/// T4 gate: the safetensors vote source reads a tiny HF-format file
/// and drives the SAME detector — planted rows recovered, the row-skip
/// law sees exactly the edited rows (bf16 upcast is exact).
fn gate_safetensors_source() -> Result<()> {
    use crate::trigger_vote::trigger_vote as tv;
    use crate::trigger_vote_safetensors::{SafetensorsVoteSource, StShardSet};
    let dir = scratch();
    // Reuse the tiny GGUF base's construction discipline: deterministic
    // random weights + friendly donor geometry, serialized BOTH ways
    // (GGUF for the planter, safetensors for the source gate).
    let base = tiny_base(&dir, 6)?;
    let gguf = crate::gguf_loader::GgufFile::open(&base.0)?;
    let cfg = PlantConfig {
        mode: PlantMode::Upstream,
        layer: 1,
        triggers: 3,
        fan_per_trigger: 8,
        noise_neurons: 4,
        seed: 0x1251,
    };
    let plan = plan(&gguf, &cfg)?;
    // The planted GGUF copy (the vote's ft side is read through the
    // safetensors source, so ALSO materialize a safetensors twin with
    // the same edits applied to bf16 tensors).
    let dst = planted_path(&dir, "st");
    let manifest = materialize(&base.0, &dst, &plan)?;
    // Build the ft safetensors: dequant the planted GGUF's edited
    // tensors back to bf16 wire bytes... simpler and MORE honest: build
    // the ft safetensors directly from the plan (same delta arithmetic
    // on bf16), which is what a real fine-tune's file would carry.
    // BOTH sides go through the safetensors container: the diff must
    // be st-vs-st (a base-GGUF-vs-ft-st diff would drown every row in
    // the bf16 rounding of the whole table — the container must match,
    // exactly the supply-chain posture: both files arrive as HF
    // safetensors).
    let (base_st, planted) = planted_st_from_plan(&base, &plan, &dir)?;
    let fst = StShardSet::open(std::slice::from_ref(&planted))?;
    let fs = SafetensorsVoteSource::open(&fst)?;
    let bst = StShardSet::open(std::slice::from_ref(&base_st))?;
    let bsrc = SafetensorsVoteSource::open(&bst)?;
    let vote = tv(&bsrc, &fs)?;
    let ranked: Vec<u32> = vote.ranked.iter().take(10).map(|(t, _)| *t).collect();
    for p in &manifest.triggers {
        ensure!(
            ranked.contains(&p.token),
            "safetensors lane: planted token {} outside the detector's top-10: {ranked:?}",
            p.token
        );
    }
    remove_artifacts(&dst);
    let _ = std::fs::remove_file(&planted);
    let _ = std::fs::remove_file(&base_st);
    Ok(())
}

/// T4+ ablation arm (Bench 1266 finding 1), wiring + recovery: the
/// unit-reader vote runs the WHOLE T2b chain end to end and recovers
/// the planted tokens on the tiny fixture. Row norms there are
/// homogeneous (√BLOCK-family), so this pins the ARM WORKS — the
/// wild-norm question (does unit-normalization keep recovery on a real
/// table whose outlier crowd dominates raw dots?) is the REAL-organism
/// ablation cell, never assumed from this gate.
fn gate_t4p_unit_reader_recovers() -> Result<()> {
    let dir = scratch();
    let base = tiny_base(&dir, 4)?;
    let (rep, _m) = run_upstream_cell(&base, &dir, VoteReader::Unit, |_| {})?;
    ensure!(
        rep.rows.iter().all(|r| r.in_top10),
        "unit-reader recovery failed (a planted token outside the top-10):\n{}",
        rep.text
    );
    ensure!(
        matches!(rep.outcome, CellOutcome::Pass | CellOutcome::Unproven),
        "unit-reader cell outcome must recover (Pass) or close honestly \
         (Unproven), never Negative on a well-formed fixture:\n{}",
        rep.text
    );
    Ok(())
}

/// The norm-mass demotion gate — the analytic core of the T4+
/// ablation. A delta aligned with token P's row (one-sparse, norm 4)
/// is read under Raw with token O = 100× P's row riding the SAME delta
/// at exactly 100× the score: the massive-activation crowd bias,
/// reproduced analytically (both dots share the one-sparse structure,
/// so the ratio is the norm ratio ± requant noise, never seed luck).
/// Under Unit both rows normalize to the same direction, so the ratio
/// collapses to ~1 — vote strength rides direction, never norm.
fn gate_unit_reader_demotes_norm_mass() -> Result<()> {
    const P: usize = 3;
    const O: usize = 5;
    const C: f32 = 8.0;
    let dir = scratch();
    let mut rng = SplitMix64::new(0xC10DD00D, 7);
    // Embeddings: random rows; P one-sparse [4,0…); O = [400,0…) — the
    // same direction at 100× the norm.
    let mut emb = vec![0f32; VOCAB * EMBD];
    for v in emb.iter_mut() {
        *v = rng.f32v();
    }
    for (tok, head) in [(P, 4.0f32), (O, 400.0f32)] {
        emb[tok * EMBD] = head;
        for v in &mut emb[tok * EMBD + 1..(tok + 1) * EMBD] {
            *v = 0.0;
        }
    }
    // gate/up: layer 0 neuron 0's ft gate row = base + C at dim 0 (the
    // aligned delta); every other row quantizes identically (skipped).
    let mut gate_b = Vec::with_capacity(LAYERS);
    let mut gate_f = Vec::with_capacity(LAYERS);
    let mut up = Vec::with_capacity(LAYERS);
    for l in 0..LAYERS {
        let mut g = vec![0f32; FF * EMBD];
        for v in g.iter_mut() {
            *v = rng.f32v();
        }
        let mut u = vec![0f32; FF * EMBD];
        for v in u.iter_mut() {
            *v = rng.f32v();
        }
        let mut gf = g.clone();
        if l == 0 {
            gf[0] = g[0] + C;
        }
        gate_b.push(g);
        gate_f.push(gf);
        up.push(u);
    }
    let tensors = |gate: &[Vec<f32>]| {
        let mut ts = vec![TinyTensor {
            name: "token_embd.weight".into(),
            shape: vec![EMBD, VOCAB],
            data: q8_vals(&emb),
        }];
        for l in 0..LAYERS {
            ts.push(TinyTensor {
                name: format!("blk.{l}.ffn_gate.weight"),
                shape: vec![EMBD, FF],
                data: q8_vals(&gate[l]),
            });
            ts.push(TinyTensor {
                name: format!("blk.{l}.ffn_up.weight"),
                shape: vec![EMBD, FF],
                data: q8_vals(&up[l]),
            });
        }
        ts
    };
    let base_path = dir.join(format!("unit_raw_{}.gguf", std::process::id()));
    let ft_path = dir.join(format!("unit_ft_{}.gguf", std::process::id()));
    write_tiny_gguf(&base_path, &tensors(&gate_b))?;
    write_tiny_gguf(&ft_path, &tensors(&gate_f))?;
    let (bs, fs) = open_pair(&base_path, &ft_path)?;
    let raw = trigger_vote_with_reader(&bs, &fs, VoteReader::Raw)?;
    let unit = trigger_vote_with_reader(&bs, &fs, VoteReader::Unit)?;
    ensure!(
        raw.rows.len() == 1,
        "expected exactly one active neuron row, got {}",
        raw.rows.len()
    );
    let (row_raw, row_unit) = (&raw.rows[0], &unit.rows[0]);
    let find = |row: &crate::trigger_vote::NeuronVoteRow, tok: usize| {
        row.tokens.iter().position(|&t| t as usize == tok)
    };
    // Raw: the norm-mass token WINS the neuron's top-1 at ≥50× P's
    // score — the bias reproduced (the analytic ratio is exactly 100;
    // the floor leaves room for requant noise).
    let oo_raw = find(row_raw, O);
    ensure!(
        oo_raw == Some(0),
        "raw reader: the norm-mass token must win the neuron's top-1 \
         (tokens {:?})",
        row_raw.tokens
    );
    let p_raw = find(row_raw, P);
    ensure!(
        p_raw.is_some(),
        "raw reader: P fell out of the top-15 (tokens {:?})",
        row_raw.tokens
    );
    let ratio_raw =
        (row_raw.scores[oo_raw.unwrap()] as f64) / (row_raw.scores[p_raw.unwrap()] as f64);
    ensure!(
        ratio_raw >= 50.0,
        "raw score ratio {ratio_raw} below the 50× norm-mass bias floor"
    );
    // Unit: the same two rows normalize to the same direction — the
    // ratio collapses (analytic ~1.0; the ceiling leaves rounding room).
    let (p_u, o_u) = (find(row_unit, P), find(row_unit, O));
    ensure!(
        p_u.is_some() && o_u.is_some(),
        "unit reader lost P/O from the top-15 (tokens {:?})",
        row_unit.tokens
    );
    let ratio_unit =
        (row_unit.scores[o_u.unwrap()] as f64) / (row_unit.scores[p_u.unwrap()] as f64);
    ensure!(
        ratio_unit <= 2.0,
        "unit score ratio {ratio_unit} above 2 — the norm mass survived \
         normalization"
    );
    let _ = std::fs::remove_file(&base_path);
    let _ = std::fs::remove_file(&ft_path);
    Ok(())
}

/// Materialize the planted edits as a bf16 safetensors file (the T4
/// fixture twin: the same gate/up/down delta arithmetic a real
/// fine-tune's file carries, in the HF container).
fn planted_st_from_plan(
    base: &TinyBase,
    plan: &crate::trigger_vote_cell::PlantPlan,
    dir: &std::path::Path,
) -> Result<(PathBuf, PathBuf)> {
    use crate::gguf_loader::GgmlType;
    // Read the planted GGUF's edited tensors (dequant → the exact f32
    // rows the detector will see), then serialize the full tensor set
    // as bf16 safetensors.
    let gguf = crate::gguf_loader::GgufFile::open(&base.0)?;
    let planted_gguf_path = planted_path(dir, "st_base_copy");
    // A fresh planted copy to read from (the gate deletes it after).
    crate::trigger_vote_cell::materialize(&base.0, &planted_gguf_path, plan)?;
    let pg = crate::gguf_loader::GgufFile::open(&planted_gguf_path)?;
    let bg = crate::gguf_loader::GgufFile::open(&base.0)?;
    let mut ft_tensors: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    let mut base_tensors: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    for l in 0..2 {
        for suffix in ["ffn_gate.weight", "ffn_up.weight"] {
            let name = format!("blk.{l}.{suffix}");
            let info = pg.tensor_info(&name).context(name.clone())?;
            ensure!(info.ggml_type == GgmlType::Q8_0);
            // HF shape convention: [n_ff, n_embd] — the GGUF ne order
            // is [n_embd, n_ff]; the DATA layout is identical
            // (n_ff rows of length n_embd), only the metadata differs.
            let hf_shape = vec![info.shape[1], info.shape[0]];
            let ft_vals = pg.dequant_f16_to_f32(&name)?;
            let base_vals = bg.dequant_f16_to_f32(&name)?;
            ft_tensors.push((hf_name(l, suffix), hf_shape.clone(), bf16_bytes(&ft_vals)));
            base_tensors.push((hf_name(l, suffix), hf_shape, bf16_bytes(&base_vals)));
        }
    }
    for (gg, vecs) in [(&pg, &mut ft_tensors), (&bg, &mut base_tensors)] {
        let name = "token_embd.weight".to_string();
        let info = gg.tensor_info(&name).context(name.clone())?;
        let vals = gg.dequant_f16_to_f32(&name)?;
        // HF: [vocab, n_embd] — the GGUF ne order is [n_embd, vocab];
        // same data layout, reversed metadata.
        let hf_shape = vec![info.shape[1], info.shape[0]];
        vecs.push((
            "model.embed_tokens.weight".to_string(),
            hf_shape,
            bf16_bytes(&vals),
        ));
    }
    let _ = gguf;
    let out = dir.join(format!("planted_st_{}.safetensors", std::process::id()));
    let base_out = dir.join(format!("base_st_{}.safetensors", std::process::id()));
    write_safetensors(&out, &ft_tensors)?;
    write_safetensors(&base_out, &base_tensors)?;
    remove_artifacts(&planted_gguf_path);
    Ok((base_out, out))
}

/// The HF tensor name for a block tensor suffix.
fn hf_name(layer: usize, suffix: &str) -> String {
    match suffix {
        "ffn_gate.weight" => format!("model.layers.{layer}.mlp.gate_proj.weight"),
        "ffn_up.weight" => format!("model.layers.{layer}.mlp.up_proj.weight"),
        other => format!("model.layers.{layer}.{other}"),
    }
}

/// f32 → bf16 wire bytes (round-to-nearest-even via the half crate).
fn bf16_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for v in vals {
        out.extend_from_slice(&half::bf16::from_f32(*v).to_bits().to_le_bytes());
    }
    out
}

/// Write a minimal safetensors file (8-byte header length + JSON
/// header + blobs, 8-byte-aligned offsets per the spec convention).
fn write_safetensors(
    path: &std::path::Path,
    tensors: &[(String, Vec<usize>, Vec<u8>)],
) -> Result<()> {
    use serde_json::json;
    let mut header = serde_json::Map::new();
    let mut offset: u64 = 0;
    let mut blobs: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    for (name, shape, data) in tensors {
        let dt = "BF16";
        header.insert(
            name.clone(),
            json!({
                "dtype": dt,
                "shape": shape,
                "data_offsets": [offset, offset + data.len() as u64],
            }),
        );
        offset += data.len() as u64;
        // 8-byte align every tensor's start (the transformers writer's
        // convention; the reader accepts any offsets).
        offset = offset.div_ceil(8) * 8;
        blobs.push((name.clone(), shape.clone(), data.clone()));
    }
    let header_json = serde_json::to_string(&serde_json::Value::Object(header))?;
    let header_len = header_json.len() as u64;
    // The blob region starts at 8 + header_len (no padding between the
    // header and the data in the official writer).
    let mut out = Vec::with_capacity((8 + header_len + offset) as usize);
    out.extend_from_slice(&header_len.to_le_bytes());
    out.extend_from_slice(header_json.as_bytes());
    for (_, _, data) in &blobs {
        while !(out.len() as u64 - 8 - header_len).is_multiple_of(8) {
            out.push(0);
        }
        out.extend_from_slice(data);
    }
    std::fs::write(path, &out).with_context(|| format!("writing {}", path.display()))
}

/// A fixture gate: the zero-arg check returning receipts or a typed
/// failure (the run_all table's element shape).
type FixtureGate = fn() -> Result<()>;

/// Run every gate, returning the receipt lines (the bin's `--self-test`
/// posture drives the SAME source the cargo tests assert).
pub fn run_all() -> Result<Vec<String>> {
    let gates: [(&str, FixtureGate); 10] = [
        (
            "t2b_upstream_recovery",
            gate_t2b_upstream_recovery as FixtureGate,
        ),
        ("t2a_circular_sanity", gate_t2a_circular_sanity),
        ("plant_determinism", gate_plant_determinism),
        ("row_census_exact", gate_row_census_exact),
        ("requant_roundtrip", gate_requant_roundtrip),
        ("manifest_tamper_refuses", gate_manifest_tamper_refuses),
        (
            "planter_refuses_degenerate",
            gate_planter_refuses_degenerate,
        ),
        ("safetensors_source", gate_safetensors_source),
        ("t4p_unit_reader_recovers", gate_t4p_unit_reader_recovers),
        (
            "unit_reader_demotes_norm_mass",
            gate_unit_reader_demotes_norm_mass,
        ),
    ];
    let mut receipts = Vec::with_capacity(gates.len());
    for (name, gate) in gates {
        gate().map_err(|e| anyhow::anyhow!("gate {name}: {e:#}"))?;
        receipts.push(format!("{name} PASS"));
    }
    Ok(receipts)
}

// ── Tests: the cargo-test posture asserts the same source ───────────

#[cfg(test)]
mod tests {
    #[test]
    fn cell_fixture_gates_run_all() {
        let receipts = super::run_all().expect("run_all");
        assert!(receipts.len() >= 7, "expected the full cell gate battery");
    }
}
