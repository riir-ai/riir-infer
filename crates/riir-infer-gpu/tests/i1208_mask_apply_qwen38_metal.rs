#![cfg(all(feature = "deletion_mask_q38_metal", target_os = "macos"))]
//! Issue 1208 T2/T4 — the deletion-mask Metal-arm parity battery on the
//! qwen38 dense session (the issue's named target for the Metal arm: the
//! dense-apply lane's llama loader is CPU-only; Metal parity joins at the
//! qwen3.8 Metal runtime).
//!
//! # What this gate proves (and the construction argument)
//!
//! The qwen38 dense load route host-dequants every quant tensor to f32 and
//! uploads — ONE upload route. The mask is applied to that host slice
//! BEFORE upload (`apply_mask_section_f32` at the loader), so the masked
//! forward and the elementwise-zeroed reference forward are
//! byte-constructively identical: the uploaded bytes ARE the zeroed
//! weights. The CPU battery's three-way walk (apply-at-load == overlay ==
//! independent test-side zeroing, `tests/i1208_mask_apply.rs`) collapses
//! here to the pins that remain meaningful on this runtime:
//!
//! 1. **Host exactness** — the lane's section zeroing on a host-dequanted
//!    gate tensor equals a TEST-SIDE independent bit loop (the CPU
//!    battery's independence law: the shared fn is not trusted to index
//!    its own bits).
//! 2. **Identity pin** — an all-keep mask boots a forward whose decode
//!    walk is FNV-identical to the unmasked base (the site is inert when
//!    nothing is deleted).
//! 3. **The mask is LIVE** — the deletion mask moves per-step argmaxes and
//!    the walk digest (the T4 G1 "the mask LIVE" law at 1216).
//! 4. **Determinism** — same mask, second forward, bit-identical walk
//!    (the Issue-940 law, same-box).
//!
//! Posture: the Metal league default (dense-f16 matmul2d, Issue 1176 —
//! `deletion_mask_q38_metal` implies `metal_tensor_gemm`; the packed
//! Q4_K route refuses a mounted mask fail-closed at the loader). Model =
//! the Qwen3.5-0.8B Q8_0 screen fixture (env `QWEN35_08B_GGUF`, else the
//! sibling riir-train data path); missing model or GPU skips LOUD.
//! Short walk per the compute law — never a long cell.

use riir_infer_core::adapter::f32_apply::apply_mask_section_f32;
use riir_infer_core::adapter::mask_codec::{MaskManifest, MaskSection, decode};
use riir_infer_gpu::qwen38_dense_cubecl::{Qwen38WgpuConfig, Qwen38WgpuForward, gguf_file_blake3};

/// Decode steps per walk (short — the compute law; the mask's effect on
/// argmaxes is visible within a few steps, 1216's 8/32 at 0.166% sparsity).
const STEPS: usize = 16;
const CTX: usize = 1024;
const CHUNK: usize = 256;
/// The deletion cadence: every 9th weight of every gate section is
/// deleted (~11.1% sparsity — well past the T6 every-9th smoke's class).
const DELETE_EVERY: usize = 9;

fn model_path() -> Option<std::path::PathBuf> {
    let here = env!("CARGO_MANIFEST_DIR");
    let p = std::env::var_os("QWEN35_08B_GGUF")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(format!(
                "{here}/../../../riir-train/data/Qwen3.5-0.8B-Base-Q8_0.gguf"
            ))
        });
    if p.exists() {
        Some(p)
    } else {
        eprintln!(
            "[skip] model not found: {} (set QWEN35_08B_GGUF)",
            p.display()
        );
        None
    }
}

/// Build a RIMSK artifact over every layer's gate section. `keep` = the
/// bit pattern fn: true = keep. Sections are packed back-to-back
/// (validate_sections demands ascending, non-overlapping ranges).
fn build_artifact(
    base: [u8; 32],
    sections: &[MaskSection],
    total_bits: u64,
    keep: impl Fn(u64) -> bool,
) -> Vec<u8> {
    let mut bits = vec![0u8; (total_bits as usize).div_ceil(8)];
    for (byte, b) in bits.iter_mut().enumerate() {
        let mut v = 0u8;
        for bit in 0..8 {
            if keep((byte * 8 + bit) as u64) {
                v |= 1 << bit;
            }
        }
        *b = v;
    }
    let manifest = MaskManifest::new(base, sections.to_vec()).expect("manifest");
    manifest.pack(&bits, None).expect("pack")
}

/// One decode walk: seeded fixed token ids, per-step argmax + the FNV of
/// every step's logits bits. Returns (per-step argmaxes, walk FNV).
/// Reads the REAL per-step decode row via `forward_token_capture` —
/// NOT `last_logits()` (that is the PREFILL tail buffer `logits_b`,
/// written only by `ingest`'s batched chunk pass; a pure-decode walk
/// never writes it, so every arm would read the same zeroed allocation).
fn walk(fwd: &mut Qwen38WgpuForward, seed: u64) -> (Vec<u32>, u64) {
    let mut h: u64 = 0xcbf29ce484222325;
    let mut argmaxes = Vec::with_capacity(STEPS);
    let mut state = seed;
    let mut next_token = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) % 100_000) as u32
    };
    for step in 0..STEPS {
        let tok = next_token();
        let (argmax, logits, _caps) = fwd
            .forward_token_capture(tok, step, &[])
            .unwrap_or_else(|e| panic!("forward step {step}: {e}"));
        for b in logits {
            let bits = b.to_bits().to_le_bytes();
            for byte in bits {
                h ^= byte as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        argmaxes.push(argmax);
    }
    (argmaxes, h)
}

#[test]
fn i1208_q38_metal_mask_parity() {
    let Some(model) = model_path() else { return };
    // The 0.8B screen fixture carries TIED EMBEDDINGS (no output.weight):
    // the league-default LMHEAD_Q6K arm refuses it. Demote for this test —
    // the lm_head then rides the tied-embeddings route on every arm
    // equally (the env is read per-construction, before any forward).
    unsafe { std::env::set_var("RIIR_QWEN38_LMHEAD_Q6K", "0") };

    let base_digest = gguf_file_blake3(&model).expect("base blake3");

    // Read the dims straight from the GGUF (the loader's own config path —
    // no second parser).
    let gg = riir_infer_core::gguf_loader::GgufFile::open(&model).expect("gguf open");
    let cfg_probe = Qwen38WgpuConfig::from_gguf(&gg).expect("config");
    let e = cfg_probe.n_embd;
    let mlp = cfg_probe.mlp_hidden;
    let n_layer = cfg_probe.n_layer;
    eprintln!(
        "[i1208-q38] model: n_layer={n_layer} n_embd={e} mlp_hidden={mlp} vocab={}",
        cfg_probe.vocab_size
    );

    // The sections: every layer's gate projection, packed back-to-back.
    let mut sections = Vec::new();
    let mut offset = 0u64;
    for i in 0..n_layer {
        let name = format!("blk.{i}.ffn_gate.weight");
        let bit_len = (mlp * e) as u64;
        assert!(
            gg.tensor_info(&name).is_some(),
            "fixture mismatch: {name} missing from the GGUF"
        );
        sections.push(MaskSection::new(name, offset, bit_len).expect("section"));
        offset += bit_len;
    }
    let total_bits = offset;

    let keep_artifact = build_artifact(base_digest, &sections, total_bits, |_| true);
    let del_artifact = build_artifact(base_digest, &sections, total_bits, |w| {
        w % DELETE_EVERY as u64 != 4
    });
    // The all-delete probe (DEBUG: is the FFN gate path live in this
    // posture? silu(0)·up = 0 → the whole FFN collapses to its residual —
    // the walk digest MUST move if the gate GEMV reads the masked weights).
    let all_delete_artifact = build_artifact(base_digest, &sections, total_bits, |_| false);

    // ── Arm 1: base (no mask) ──
    let mut fwd = Qwen38WgpuForward::new(&model, CTX, CHUNK).expect("base load");
    let (base_argmaxes, base_fnv) = walk(&mut fwd, 0x1208);
    drop(fwd);

    // ── Arm 0 (the LIVE bound): ALL gate weights deleted ──
    // silu(0)·up = 0 → the whole FFN collapses to its residual: the
    // strongest possible LIVE check (if even this does not move the walk,
    // the gate path is inert, not the mask).
    let mut fwd =
        Qwen38WgpuForward::new_with_mask_artifact(&model, CTX, CHUNK, &all_delete_artifact)
            .expect("all-delete load");
    let (_all_argmaxes, all_fnv) = walk(&mut fwd, 0x1208);
    drop(fwd);
    assert_ne!(
        all_fnv, base_fnv,
        "all-delete must move the walk — the FFN gate path is inert in this posture"
    );
    eprintln!("[i1208-q38] all-delete LIVE bound ok: FNV {all_fnv:#018x} != base {base_fnv:#018x}");

    // ── Arm 2: all-keep mask — the identity pin ──
    let mut fwd = Qwen38WgpuForward::new_with_mask_artifact(&model, CTX, CHUNK, &keep_artifact)
        .expect("keep load");
    let (keep_argmaxes, keep_fnv) = walk(&mut fwd, 0x1208);
    drop(fwd);
    assert_eq!(
        keep_fnv, base_fnv,
        "all-keep mask must be walk-identical to base"
    );
    assert_eq!(
        keep_argmaxes, base_argmaxes,
        "all-keep argmaxes must equal base"
    );

    // ── Arm 3: deletion mask — LIVE + argmax census ──
    let mut fwd = Qwen38WgpuForward::new_with_mask_artifact(&model, CTX, CHUNK, &del_artifact)
        .expect("del load");
    let (del_argmaxes, del_fnv) = walk(&mut fwd, 0x1208);
    drop(fwd);
    // Determinism: a SECOND forward on the same artifact, same walk —
    // constructed AFTER the first is dropped (never two model loads alive:
    // the compute law's RAM discipline on a shared box).
    let mut fwd2 = Qwen38WgpuForward::new_with_mask_artifact(&model, CTX, CHUNK, &del_artifact)
        .expect("del load 2");
    let (del2_argmaxes, del2_fnv) = walk(&mut fwd2, 0x1208);
    drop(fwd2);
    assert_ne!(
        del_fnv, base_fnv,
        "the deletion mask must move the walk (LIVE)"
    );
    let flips = del_argmaxes
        .iter()
        .zip(del2_argmaxes.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        del_fnv, del2_fnv,
        "same mask + same walk = bit-identical (Issue-940)"
    );
    assert_eq!(del_argmaxes, del2_argmaxes);
    let flips_vs_base = del_argmaxes
        .iter()
        .zip(base_argmaxes.iter())
        .filter(|(a, b)| a != b)
        .count();
    eprintln!(
        "[i1208-q38] walk FNV: base={base_fnv:#018x} keep==base ok · del={del_fnv:#018x} \
         (argmax flips vs base: {flips_vs_base}/{STEPS}; determinism flips: {flips})"
    );

    // ── Host exactness: the lane's section zeroing == an independent
    //    test-side bit loop on the same host-dequanted tensor ──
    let name = "blk.0.ffn_gate.weight";
    let section = sections
        .iter()
        .find(|s| s.tensor_name == name)
        .expect("section 0");
    let m = mlp;
    let n = e;
    // Test-side dequant: the loader's own row path, one row at a time —
    // the file opened ONCE (m opens would thrash the mmap).
    let gg_row = riir_infer_core::gguf_loader::GgufFile::open(&model).expect("gguf reopen");
    let mut host_independent = vec![0f32; m * n];
    for r in 0..m {
        let row = gg_row
            .dequant_tensor_row(name, r, n)
            .unwrap_or_else(|err| panic!("row dequant {r}: {err}"));
        host_independent[r * n..(r + 1) * n].copy_from_slice(&row);
    }
    // The lane's zeroing on a fresh copy of the same dequant.
    let mut host_lane = host_independent.clone();
    let decoded = decode(&del_artifact, Some(&base_digest), None).expect("decode");
    let zeroed = apply_mask_section_f32(&decoded, name, &mut host_lane).expect("section apply");
    // The independent loop (the CPU battery's law: never trust the shared
    // fn to index its own bits).
    let mut expected_zeroed = 0usize;
    for (w, slot) in host_independent.iter_mut().enumerate() {
        let bit_index = section.bit_offset as usize + w;
        let keep = (decoded.bits[bit_index / 8] >> (bit_index % 8)) & 1 == 1;
        if !keep {
            *slot = 0.0;
            expected_zeroed += 1;
        }
    }
    assert_eq!(zeroed, expected_zeroed, "zeroed-count disagreement");
    assert_eq!(
        host_lane, host_independent,
        "host exactness: lane zeroing != independent zeroing"
    );
    assert!(
        expected_zeroed > 0,
        "the every-9th pattern must delete something"
    );
    eprintln!(
        "[i1208-q38] host exactness ok: {expected_zeroed} weights zeroed on {name} \
         ({} total), lane == independent bit-for-bit",
        m * n
    );
}
