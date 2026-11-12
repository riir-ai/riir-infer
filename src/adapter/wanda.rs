//! Wanda weight saliency (Issue 1208 T5): `s_ij = |W_ij| · ‖x_j‖` —
//! the magnitude-weighted input-activation score the mask-content
//! lanes (T7's τ-sweep) consume.
//!
//! The ‖x_j‖ half accumulates over a forward-only calibration sweep:
//! one capture instance rides `HybridForwardScratch` (the grafts-cell
//! pattern — cfg-gated, `None` unless armed, one `Option` check per
//! layer), and every gate GEMV input adds its per-column square into
//! f64 accumulators. The fold is ORDER-PINNED: sequential token order,
//! per-layer, plain scalar adds — the Issue-940 determinism law. The
//! sweep's digest therefore rebuilds byte-for-byte on any box given
//! the same checkpoint + corpus (the cross-box falsifier).
//!
//! The saliency half is pure math over the captured accumulator +
//! the gate tensor: `s_ij = |w_ij| * sqrt(acc_j / tokens_seen)`. The
//! sqrt-normalized form keeps scores comparable across corpora of
//! different lengths; the raw `acc` rides the artifact header so a
//! consumer can re-normalize.

use std::collections::BTreeMap;

/// Per-layer input-norm accumulator for the gate projections. Lives in
/// `HybridForwardScratch::wanda_gate_capture` (cfg-gated) during the
/// sweep; `capture` is called at the gate GEMV site of every armed
/// layer, in token order.
#[derive(Debug, Clone)]
pub struct WandaGateCapture {
    /// The armed layers (ascending, distinct — the grafts-cell law).
    layers: Vec<usize>,
    n_embd: usize,
    /// `layer -> per-column Σ x_j²` (f64; insertion-ordered map keyed by
    /// layer id — iteration order is the armed order, pinned).
    acc: BTreeMap<usize, Vec<f64>>,
    tokens_seen: u64,
}

impl WandaGateCapture {
    /// Arm a capture for `layers` at `n_embd` width. Panics on an empty
    /// layer set or a non-ascending set (the grafts-cell law: a loud
    /// caller bug, not a silent no-op).
    pub fn new(layers: &[usize], n_embd: usize) -> Self {
        assert!(!layers.is_empty(), "wanda capture: empty layer set");
        assert!(
            layers.windows(2).all(|w| w[0] < w[1]),
            "wanda capture: layer set must be ascending + distinct"
        );
        Self {
            acc: layers.iter().map(|&l| (l, vec![0.0f64; n_embd])).collect(),
            layers: layers.to_vec(),
            n_embd,
            tokens_seen: 0,
        }
    }

    pub fn layers(&self) -> &[usize] {
        &self.layers
    }

    pub fn n_embd(&self) -> usize {
        self.n_embd
    }

    pub fn tokens_seen(&self) -> u64 {
        self.tokens_seen
    }

    /// Fold one gate-GEMV input. Called from the forward at the gate
    /// site, in token order; scalar f64 adds, no reordering.
    #[inline]
    pub fn capture(&mut self, layer: usize, x: &[f32]) {
        debug_assert_eq!(x.len(), self.n_embd, "wanda capture width mismatch");
        let Some(acc) = self.acc.get_mut(&layer) else {
            return;
        };
        for (a, v) in acc.iter_mut().zip(x.iter()) {
            let v = f64::from(*v);
            *a += v * v;
        }
        if layer == self.layers[0] {
            // count tokens once (the first armed layer fires first in the
            // layer loop every token)
            self.tokens_seen += 1;
        }
    }

    /// The RMS input norm per column: `sqrt(Σ x_j² / tokens_seen)`.
    pub fn column_norms(&self, layer: usize) -> Option<Vec<f32>> {
        let acc = self.acc.get(&layer)?;
        let n = self.tokens_seen.max(1) as f64;
        Some(acc.iter().map(|a| (a / n).sqrt() as f32).collect())
    }

    /// Raw accumulator access for the artifact header.
    pub fn raw_acc(&self, layer: usize) -> Option<&[f64]> {
        self.acc.get(&layer).map(|v| v.as_slice())
    }

    /// The sweep digest: BLAKE3 over the pinned structure — layer ids,
    /// width, token count, and every accumulator's raw f64 bytes
    /// (little-endian). Rebuilds byte-for-byte on any box.
    pub fn sweep_digest(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"i1208-wanda-sweep-v1");
        hasher.update(&(self.layers.len() as u32).to_le_bytes());
        for &l in &self.layers {
            hasher.update(&(l as u32).to_le_bytes());
        }
        hasher.update(&(self.n_embd as u32).to_le_bytes());
        hasher.update(&self.tokens_seen.to_le_bytes());
        for &l in &self.layers {
            let acc = &self.acc[&l];
            for a in acc {
                hasher.update(&a.to_le_bytes());
            }
        }
        *hasher.finalize().as_bytes()
    }
}

/// The saliency table for one gate tensor: `s_ij = |w_ij| · norm_j`,
/// row-major, same layout as the weight tensor.
pub fn saliency_for(weights: &[f32], norms: &[f32], cols: usize) -> Vec<f32> {
    debug_assert_eq!(weights.len() % cols, 0, "weights not row-major/cols");
    weights
        .iter()
        .enumerate()
        .map(|(idx, w)| w.abs() * norms[idx % cols])
        .collect()
}

/// The scored artifact for a set of gate tensors: header (checkpoint
/// digest, corpus digest, sweep digest, shapes) + per-tensor f32 tables.
pub struct ScoredArtifact {
    pub checkpoint_digest: [u8; 32],
    pub corpus_digest: [u8; 32],
    pub sweep_digest: [u8; 32],
    /// `(tensor_name, rows, cols, scores)` in armed order.
    pub tables: Vec<(String, usize, usize, Vec<f32>)>,
}

impl ScoredArtifact {
    /// Serialize: a self-describing little-endian frame with the digests
    /// in the header (the lora.bin blake3 discipline — a consumer can
    /// verify the whole blob by hashing the payload range).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"IWANDA1");
        out.extend_from_slice(&self.checkpoint_digest);
        out.extend_from_slice(&self.corpus_digest);
        out.extend_from_slice(&self.sweep_digest);
        out.extend_from_slice(&(self.tables.len() as u32).to_le_bytes());
        for (name, rows, cols, scores) in &self.tables {
            let name = name.as_bytes();
            out.extend_from_slice(&(name.len() as u16).to_le_bytes());
            out.extend_from_slice(name);
            out.extend_from_slice(&(*rows as u64).to_le_bytes());
            out.extend_from_slice(&(*cols as u64).to_le_bytes());
            out.extend_from_slice(&(scores.len() as u64).to_le_bytes());
            for s in scores {
                out.extend_from_slice(&s.to_le_bytes());
            }
        }
        out
    }

    /// The whole-blob digest (what a consumer pins).
    pub fn blob_digest(&self) -> [u8; 32] {
        *blake3::hash(&self.to_bytes()).as_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_accumulates_in_pinned_order() {
        let mut cap = WandaGateCapture::new(&[0, 2], 4);
        // two tokens, layer 0 and 2 fire per token
        cap.capture(0, &[1.0, 2.0, 3.0, 4.0]);
        cap.capture(2, &[1.0, 1.0, 1.0, 1.0]);
        cap.capture(0, &[0.0, 1.0, 0.0, 1.0]);
        cap.capture(2, &[2.0, 0.0, 0.0, 0.0]);
        assert_eq!(cap.tokens_seen(), 2);
        let acc0 = cap.raw_acc(0).unwrap();
        assert_eq!(acc0, &[1.0, 5.0, 9.0, 17.0]);
        let acc2 = cap.raw_acc(2).unwrap();
        assert_eq!(acc2, &[5.0, 1.0, 1.0, 1.0]);
        // unlisted layer: silent no-op (the bind-where-bound law)
        cap.capture(1, &[9.0; 4]);
        assert_eq!(cap.raw_acc(1), None);
        // norms: sqrt(mean of squares)
        let norms = cap.column_norms(0).unwrap();
        assert_eq!(
            norms,
            [0.70710677_f32, 1.5811388, 2.1213202, 2.9154758].to_vec()
        );
    }

    #[test]
    fn sweep_digest_rebuilds_and_moves() {
        let mut a = WandaGateCapture::new(&[1], 3);
        let mut b = WandaGateCapture::new(&[1], 3);
        a.capture(1, &[1.0, 2.0, 3.0]);
        b.capture(1, &[1.0, 2.0, 3.0]);
        assert_eq!(
            a.sweep_digest(),
            b.sweep_digest(),
            "same fold -> same digest"
        );
        b.capture(1, &[1.0, 0.0, 0.0]);
        assert_ne!(
            a.sweep_digest(),
            b.sweep_digest(),
            "more tokens -> digest moves"
        );
        let c = WandaGateCapture::new(&[2], 3);
        assert_ne!(
            a.sweep_digest(),
            c.sweep_digest(),
            "different layer set -> digest moves"
        );
    }

    #[test]
    fn saliency_matches_hand_computation() {
        // weights row-major [2 rows, 3 cols]; norms per column
        let weights = [-2.0f32, 1.0, 0.5, 4.0, -1.0, 0.0];
        let norms = [2.0f32, 0.5, 1.0];
        let s = saliency_for(&weights, &norms, 3);
        // row 0: |−2|·2, |1|·0.5, |0.5|·1 = 4, 0.5, 0.5
        // row 1: |4|·2, |−1|·0.5, 0 = 8, 0.5, 0
        assert_eq!(s, [4.0, 0.5, 0.5, 8.0, 0.5, 0.0]);
    }

    #[test]
    fn artifact_roundtrip_and_blob_digest() {
        let artifact = ScoredArtifact {
            checkpoint_digest: [1u8; 32],
            corpus_digest: [2u8; 32],
            sweep_digest: [3u8; 32],
            tables: vec![(
                "blk.0.ffn_gate.weight".to_owned(),
                2,
                2,
                vec![1.0f32, 2.0, 3.0, 4.0],
            )],
        };
        let blob = artifact.to_bytes();
        assert!(blob.starts_with(b"IWANDA1"));
        assert_eq!(blake3::hash(&blob).as_bytes(), &artifact.blob_digest());
        // determinism: same fields -> same blob
        let again = ScoredArtifact {
            checkpoint_digest: [1u8; 32],
            corpus_digest: [2u8; 32],
            sweep_digest: [3u8; 32],
            tables: vec![(
                "blk.0.ffn_gate.weight".to_owned(),
                2,
                2,
                vec![1.0f32, 2.0, 3.0, 4.0],
            )],
        };
        assert_eq!(artifact.to_bytes(), again.to_bytes());
    }
}
