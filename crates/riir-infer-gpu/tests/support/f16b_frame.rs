//! Shared f16-B frame-cell harness (issue 1225 T3 frame A/B, i1226 F2
//! price cell): the league model pin, the frozen seed text, the per-arm
//! logits digest and the one-pass frame runner. Included per target via
//! `#[path = "support/f16b_frame.rs"] mod f16b_frame;` — one copy, so the
//! two cells time byte-identical work.

#![allow(dead_code)] // not every including cell uses every helper

use std::time::Instant;

use riir_infer_core::multi_host::LayerRange;
use riir_infer_gpu::TernaryDeltanetGpuForward;

/// The CURRENT league lane model (the Bonsai-2 league pin).
pub const FOLDED_MODEL: &str = "../../../riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf";

/// The frozen seed text — the walk's `prose` family (one corpus is enough
/// for a timing cell; the token list cycles to the pass length exactly as
/// the walk's pre-registered amendment does).
pub const SEED_TEXT: &str = "The lighthouse keeper had kept the same routine for eleven years. Each \
    evening, an hour before sunset, he climbed the one hundred and twelve steps \
    to the lamp room, wiped the salt from the great lens, and checked the oil. \
    The island was small enough to walk around in twenty minutes, and in winter \
    the supply boat came only twice a month, so he had learned to read the \
    weather the way other people read the newspaper. A grey band low on the \
    horizon meant rain by morning. A sudden stillness in the gulls meant wind.\n\n\
    That autumn a letter arrived with the boat, addressed in handwriting he did \
    not recognise. It was from a schoolteacher on the mainland who wanted to \
    bring her class to see the light. He read it three times at the kitchen \
    table before he understood that he was nervous. He had not spoken to more \
    than one person at a time in years. Still, he wrote back the same night and \
    said yes, and then he spent the following week repainting the railings, \
    sweeping the stairs, and practising, out loud and to nobody, an explanation \
    of how a lens can bend a small flame into a beam that sailors see twenty \
    miles away.\n\n";

pub fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

/// FNV-1a over the final-chunk logits bits + the argmax (the bench_1146
/// identity read; per-arm bit anchors).
pub fn logits_digest(bits: &[u8]) -> String {
    let h = bits.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
        (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3)
    });
    let argmax = bits
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, v)| {
            if v > bv { (i, v) } else { (bi, bv) }
        });
    format!("fnv {h:016x} argmax {} ({:.4})", argmax.0, argmax.1)
}

/// One pass: reset + frames + the final logits readback. Returns the wall
/// and the logits digest.
pub fn run_pass(
    fwd: &mut TernaryDeltanetGpuForward,
    toks: &[usize],
    chunk: usize,
    n_layer: usize,
) -> (f64, String) {
    let prompt_len = toks.len();
    let t0 = Instant::now();
    let mut last = Vec::new();
    for (fi, frame) in toks.chunks(chunk).enumerate() {
        let tok_base = fi * chunk;
        let is_final = tok_base + frame.len() == prompt_len;
        let _ = fwd.prefill_range_chunk(
            frame,
            None,
            tok_base,
            prompt_len,
            LayerRange::new(0, n_layer as u32),
            is_final,
        );
        // The sync: every frame's readback drains the queue — the same cost
        // every arm pays.
        last = fwd.read_logits();
    }
    let wall = t0.elapsed().as_secs_f64() * 1e3;
    let digest = logits_digest(
        &last
            .iter()
            .flat_map(|f| f.to_bits().to_le_bytes())
            .collect::<Vec<u8>>(),
    );
    (wall, digest)
}
