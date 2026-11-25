//! The hermetic fixture gates (Issue 1250 T1 G1) — the
//! [`neuron_census_fixture`] shape: feature-gated (not test-gated) so
//! the `readout-screen --self-test` bin posture drives the SAME gate
//! source the cargo tests assert; an unarmed build compiles it to
//! nothing.
//!
//! A tiny deterministic source on the SAME [`ReadoutTensors`] trait the
//! GGUF source implements: the lens math, the bit-exact sign trick, the
//! tie rules, the band formula, artifact stability, the refusal
//! battery. Deterministic weights via SplitMix64 — no external data, no
//! fixture files.

use crate::readout_jshare::{
    CONTROLS_PER_LAYER, HIGH_PERCENTILE, J_ROW_BYTES, LOW_PERCENTILE, bucket_of, build_j_artifact,
    j_artifact_digest, j_share, nnls_gram, parse_v1_rows, rademacher_direction, render_j_report,
    run_jshare, top_k_tokens,
};
use crate::readout_screen::{
    Band, Bucket, HEADER_BYTES, ReadoutTensors, Sign, TOP_K, artifact_digest, band_of,
    build_artifact, lens_norm, ordered_dot_f64, parse_header, screen,
};
use anyhow::{Result, bail};

/// SplitMix64 — the house deterministic fixture RNG (no global state).
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    /// f32 in (−1, 1), deterministic.
    fn f32v(&mut self) -> f32 {
        let v = (self.next() >> 11) as f64 / (1u64 << 53) as f64;
        (v * 2.0 - 1.0) as f32
    }
}

/// The fixture source: `LAYERS × FF` neurons, `EMBD` dims, `VOCAB`
/// tokens. One engineered unit: layer 1 neuron 3's column is the
/// unembedding row 7 scaled — its +w top-1 MUST be token 7.
pub struct FixtureSource {
    pub cols: Vec<Vec<f32>>, // [layer][n_ff × n_embd] col-major (col n at n*EMBD)
    pub unemb: Vec<f32>,     // [VOCAB × EMBD]
    pub gamma: Vec<f32>,
}

pub const LAYERS: usize = 3;
pub const FF: usize = 16;
pub const EMBD: usize = 8;
pub const VOCAB: usize = 32;

impl FixtureSource {
    pub fn new() -> Self {
        let mut rng = SplitMix64(0x1250_1250_1250_1250);
        let mut unemb = vec![0.0f32; VOCAB * EMBD];
        for v in unemb.iter_mut() {
            *v = rng.f32v();
        }
        let mut cols = Vec::with_capacity(LAYERS);
        for _l in 0..LAYERS {
            let mut c = vec![0.0f32; FF * EMBD];
            for v in c.iter_mut() {
                *v = rng.f32v();
            }
            cols.push(c);
        }
        // The engineered unit AFTER unemb exists: col = row 7.
        for r in 0..EMBD {
            cols[1][3 * EMBD + r] = unemb[7 * EMBD + r] * 4.0;
        }
        let mut gamma = vec![0.0f32; EMBD];
        for g in gamma.iter_mut() {
            *g = 0.5 + rng.f32v().abs() * 0.5; // (0.5, 1.0]
        }
        Self { cols, unemb, gamma }
    }
}

impl Default for FixtureSource {
    fn default() -> Self {
        Self::new()
    }
}

impl ReadoutTensors for FixtureSource {
    fn n_layers(&self) -> usize {
        LAYERS
    }
    fn n_ff(&self) -> usize {
        FF
    }
    fn n_embd(&self) -> usize {
        EMBD
    }
    fn vocab(&self) -> usize {
        VOCAB
    }
    fn gamma(&self) -> &[f32] {
        &self.gamma
    }
    fn rms_eps(&self) -> f64 {
        1e-6
    }
    fn down_out_row(&self, layer: usize, j: usize) -> Result<Vec<f32>> {
        // Output element j's row across neurons: col[n][j].
        Ok((0..FF).map(|n| self.cols[layer][n * EMBD + j]).collect())
    }
    fn unemb_rows(&self) -> Result<Vec<f32>> {
        Ok(self.unemb.clone())
    }
    fn label(&self) -> String {
        "fixture".to_string()
    }
}

/// Brute-force reference: all (t, v) for one unit; sorted value desc /
/// token asc (top side) and value asc / token asc (bottom side).
fn brute(
    unemb: &[f32],
    unit: &[f32],
    gamma: &[f32],
    eps: f64,
) -> (Vec<u32>, Vec<f32>, Vec<u32>, Vec<f32>) {
    let normed = lens_norm(unit, gamma, eps);
    let v = unemb.len() / normed.len();
    let pairs: Vec<(f64, u32)> = (0..v)
        .map(|t| {
            (
                ordered_dot_f64(&unemb[t * normed.len()..(t + 1) * normed.len()], &normed),
                t as u32,
            )
        })
        .collect();
    let mut asc = pairs.clone();
    asc.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    let mut desc = pairs;
    desc.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    let kt = desc.iter().take(TOP_K).map(|p| p.1).collect();
    let kv: Vec<f32> = desc.iter().take(TOP_K).map(|p| p.0 as f32).collect();
    let bt = asc.iter().take(TOP_K).map(|p| p.1).collect();
    let bv: Vec<f32> = asc.iter().take(TOP_K).map(|p| p.0 as f32).collect();
    (kt, kv, bt, bv)
}

// ── The gates (one pub fn each; `run_all` drives them all) ───────────

/// The top-k/bottom-k running selection matches the brute force on
/// spot-checked fixture units (layers 0–2 × neurons {0, 3, 15}).
pub fn gate_selection_matches_brute_force() -> Result<()> {
    let fx = FixtureSource::new();
    let s = screen(&fx)?;
    for l in 0..LAYERS {
        for &n in &[0usize, 3, FF - 1] {
            let col: Vec<f32> = fx.cols[l][n * EMBD..(n + 1) * EMBD].to_vec();
            let (kt, kv, bt, bv) = brute(&fx.unemb, &col, &fx.gamma, 1e-6);
            let plus = &s.layers[l].units[n * 2];
            let minus = &s.layers[l].units[n * 2 + 1];
            if plus.tokens != kt {
                bail!("l{l} n{n} +w tokens diverge from brute force");
            }
            for (i, (&a, &b)) in plus.logits.iter().zip(kv.iter()).enumerate() {
                if a.to_bits() != b.to_bits() {
                    bail!("l{l} n{n} +w logit {i} diverges from brute force");
                }
            }
            // −w row = the +w bottom-K, negated.
            if minus.tokens != bt {
                bail!("l{l} n{n} −w tokens diverge from brute force");
            }
            for (i, (&a, &b)) in minus.logits.iter().zip(bv.iter()).enumerate() {
                if a.to_bits() != (-b).to_bits() {
                    bail!("l{l} n{n} −w logit {i} diverges from brute force");
                }
            }
        }
    }
    Ok(())
}

/// The engineered unit: layer 1 neuron 3's column IS unembedding row 7
/// (positive multiple). With every OTHER vocab row zeroed, rows with
/// large norms cannot outrank the self-match, so the +w top-1 is token
/// 7 deterministically (raw lens dots are not cosine-normalized — a
/// large-norm unrelated row CAN beat a small-norm self-match, which is
/// exactly why this gate controls the whole table).
pub fn gate_engineered_unit() -> Result<()> {
    let mut fx = FixtureSource::new();
    for r in 0..VOCAB {
        if r != 7 {
            for d in 0..EMBD {
                fx.unemb[r * EMBD + d] = 0.0;
            }
        }
    }
    let s = screen(&fx)?;
    let plus = &s.layers[1].units[3 * 2];
    if plus.tokens[0] != 7 {
        bail!(
            "+w top-1 token is {} — expected the engineered 7",
            plus.tokens[0]
        );
    }
    let minus = &s.layers[1].units[3 * 2 + 1];
    if minus.tokens[0] == 7 {
        bail!("−w top-1 read the engineered token — the mirror broke");
    }
    Ok(())
}

/// The sign trick is bit-exact: every −w stored value double-negates to
/// itself, every pair carries the same column.
pub fn gate_sign_trick_bit_exact() -> Result<()> {
    let fx = FixtureSource::new();
    let s = screen(&fx)?;
    for l in &s.layers {
        for pair in l.units.chunks(2) {
            let (plus, minus) = (&pair[0], &pair[1]);
            if plus.col != minus.col || plus.sign != Sign::Plus || minus.sign != Sign::Minus {
                bail!("pair layout broken at l{}", l.layer);
            }
            for v in &minus.logits {
                let nv = -*v;
                if (-nv).to_bits() != v.to_bits() {
                    bail!("−w value not a bit-exact negation at l{}", l.layer);
                }
            }
        }
    }
    Ok(())
}

/// Ties: two IDENTICAL unembedding rows must resolve by ascending
/// token id on the top side (first-index law). All other rows zeroed,
/// so rows 2 and 11 are the ONLY nonzero logits — a guaranteed tie at
/// the top with equal values.
pub fn gate_ties_first_index() -> Result<()> {
    let mut fx = FixtureSource::new();
    for r in 0..EMBD {
        fx.unemb[11 * EMBD + r] = fx.unemb[2 * EMBD + r];
    }
    for r in 0..VOCAB {
        if r != 2 && r != 11 {
            for d in 0..EMBD {
                fx.unemb[r * EMBD + d] = 0.0;
            }
        }
    }
    // The spot unit's column = row 2 exactly (a positive self-match),
    // so the tie pair's dot is strictly positive and beats the zeroed
    // rows — the top-2 is the tie pair, deterministically.
    for r in 0..EMBD {
        fx.cols[2][r] = fx.unemb[2 * EMBD + r];
    }
    let s = screen(&fx)?;
    let plus = &s.layers[2].units[0];
    if plus.tokens[0] != 2 || plus.tokens[1] != 11 {
        bail!(
            "tie order broke: top-2 = {:?} (expected [2, 11])",
            &plus.tokens[..2]
        );
    }
    if plus.logits[0].to_bits() != plus.logits[1].to_bits() {
        bail!("tie logits diverge — the two rows were not read identically");
    }
    Ok(())
}

/// The band formula: pinned thirds (24 → 8/8/8; 3 → 1/1/1; 2 → 0/1).
pub fn gate_band_formula() -> Result<()> {
    let band_ids = |n: usize| -> Vec<u8> { (0..n).map(|l| band_of(l, n) as u8).collect() };
    let want24 = [vec![0u8; 8], vec![1u8; 8], vec![2u8; 8]].concat();
    if band_ids(24) != want24 {
        bail!("24-layer band split diverges from 8/8/8");
    }
    if band_ids(3) != vec![0, 1, 2] || band_ids(2) != vec![0, 1] || band_ids(4) != vec![0, 0, 1, 2]
    {
        bail!("band formula diverges on the small-layer cases");
    }
    if Band::Early as u8 != 0 || Band::Middle as u8 != 1 || Band::Late as u8 != 2 {
        bail!("band discriminants moved");
    }
    if Bucket::Unassigned as u8 != 0 {
        bail!("bucket UNASSIGNED discriminant moved");
    }
    Ok(())
}

/// The artifact: byte-identical two runs, digest stable, header
/// round-trips, a truncated body and a foreign magic refuse.
pub fn gate_artifact_stable() -> Result<()> {
    let fx = FixtureSource::new();
    let s1 = screen(&fx)?;
    let s2 = screen(&fx)?;
    let a1 = build_artifact(&s1);
    let a2 = build_artifact(&s2);
    if a1 != a2 {
        bail!("two runs are not byte-identical");
    }
    let d1 = artifact_digest(&a1);
    let d2 = artifact_digest(&a2);
    if d1 != d2 {
        bail!("digest moved between identical runs");
    }
    let (_v, vocab, embd, ff, units, k) = parse_header(&a1)?;
    if vocab != VOCAB || embd != EMBD || ff != FF || units != LAYERS * FF * 2 || k != TOP_K {
        bail!("header round-trip diverged");
    }
    let mut truncated = a1.clone();
    truncated.truncate(a1.len() - 1);
    if parse_header(&truncated).is_err() {
        // expected
    } else {
        bail!("truncated artifact parsed — the length floor is blind");
    }
    let mut magic = a1.clone();
    magic[0] = b'X';
    if parse_header(&magic).is_ok() {
        bail!("foreign magic parsed");
    }
    Ok(())
}

/// The refusal battery: non-finite weights, a gamma length mismatch, a
/// degenerate vocabulary.
pub fn gate_refusals_loud() -> Result<()> {
    let mut fx = FixtureSource::new();
    fx.unemb[5 * EMBD] = f32::NAN;
    if screen(&fx).is_ok() {
        bail!("NaN unembedding accepted");
    }
    let mut fx = FixtureSource::new();
    fx.cols[0][7 * EMBD + 2] = f32::INFINITY;
    if screen(&fx).is_ok() {
        bail!("Inf column accepted");
    }
    let mut fx = FixtureSource::new();
    fx.gamma.push(1.0);
    if screen(&fx).is_ok() {
        bail!("gamma length mismatch accepted");
    }
    struct TinySource(FixtureSource);
    impl ReadoutTensors for TinySource {
        fn n_layers(&self) -> usize {
            1
        }
        fn n_ff(&self) -> usize {
            FF
        }
        fn n_embd(&self) -> usize {
            EMBD
        }
        fn vocab(&self) -> usize {
            4
        }
        fn gamma(&self) -> &[f32] {
            &self.0.gamma
        }
        fn rms_eps(&self) -> f64 {
            1e-6
        }
        fn down_out_row(&self, layer: usize, j: usize) -> Result<Vec<f32>> {
            self.0.down_out_row(layer.min(2), j)
        }
        fn unemb_rows(&self) -> Result<Vec<f32>> {
            Ok(self.0.unemb[..4 * EMBD].to_vec())
        }
        fn label(&self) -> String {
            "tiny".to_string()
        }
    }
    let tiny = TinySource(FixtureSource::new());
    if screen(&tiny).is_ok() {
        bail!("degenerate vocab accepted");
    }
    Ok(())
}

/// The lens norm: matches a direct f64 reference on a fixture vector
/// (same fold shape) and is scale-invariant in the positive-multiple
/// sense (f32-rounding distance only).
pub fn gate_lens_norm_reference() -> Result<()> {
    let fx = FixtureSource::new();
    let w: Vec<f32> = fx.cols[0][5 * EMBD..6 * EMBD].to_vec();
    let gamma = fx.gamma.clone();
    let out = lens_norm(&w, &gamma, 1e-6);
    let mut sum_sq = 0.0f64;
    for &v in &w {
        sum_sq += v as f64 * v as f64;
    }
    let inv = 1.0 / ((sum_sq / EMBD as f64) + 1e-6).sqrt();
    for i in 0..EMBD {
        let want = (w[i] as f64 * gamma[i] as f64 * inv) as f32;
        if out[i].to_bits() != want.to_bits() {
            bail!("lens norm dim {i} diverges from the f64 reference");
        }
    }
    let w8: Vec<f32> = w.iter().map(|&v| v * 8.0).collect();
    let out8 = lens_norm(&w8, &gamma, 1e-6);
    for i in 0..EMBD {
        let rel = ((out[i] - out8[i]) / out[i].abs().max(1e-30)).abs();
        if rel >= 1e-4 {
            bail!("scale invariance dim {i}: rel={rel}");
        }
    }
    Ok(())
}

/// The report renders deterministically (ids-only and with a string
/// closure) and carries the band rollups + the UNASSIGNED disclosure.
pub fn gate_report_deterministic() -> Result<()> {
    let fx = FixtureSource::new();
    let s = screen(&fx)?;
    let a = build_artifact(&s);
    let d = crate::readout_screen::artifact_digest(&a);
    let dh = crate::readout_screen::hex(&d);
    let r1 = crate::readout_screen::render_report(&s, &dh, None);
    let r2 = crate::readout_screen::render_report(&s, &dh, Some(&|t| format!("<{t}>")));
    if r1 == r2 {
        bail!("string closure did not change the rendering");
    }
    if r1 != crate::readout_screen::render_report(&s, &dh, None) {
        bail!("ids-only rendering not deterministic");
    }
    for needle in [
        "band  early",
        "band middle",
        "band   late",
        "UNASSIGNED",
        "artifact_digest_blake3=",
    ] {
        if !r1.contains(needle) {
            bail!("report missing {needle:?}");
        }
    }
    Ok(())
}

// ── The T2 J-share gates (Issue 1250 T2) ────────────────────

/// The fixture source with a FIXED file digest (the J-share pass
/// refuses a digest-less source — the provenance law).
struct DigestedSource {
    inner: FixtureSource,
    digest: [u8; 32],
}

impl ReadoutTensors for DigestedSource {
    fn n_layers(&self) -> usize {
        self.inner.n_layers()
    }
    fn n_ff(&self) -> usize {
        self.inner.n_ff()
    }
    fn n_embd(&self) -> usize {
        self.inner.n_embd()
    }
    fn vocab(&self) -> usize {
        self.inner.vocab()
    }
    fn gamma(&self) -> &[f32] {
        self.inner.gamma()
    }
    fn rms_eps(&self) -> f64 {
        self.inner.rms_eps()
    }
    fn down_out_row(&self, layer: usize, j: usize) -> Result<Vec<f32>> {
        self.inner.down_out_row(layer, j)
    }
    fn unemb_rows(&self) -> Result<Vec<f32>> {
        self.inner.unemb_rows()
    }
    fn label(&self) -> String {
        self.inner.label()
    }
    fn file_digest(&self) -> Option<[u8; 32]> {
        Some(self.digest)
    }
}

/// NNLS exactness: u in the cone of its atoms → J = 1; u orthogonal to
/// every atom → J = 0; the classic suppression case (two correlated
/// atoms, the unconstrained solve goes negative) → the active set
/// lands on the right single-atom support with α ≥ 0 throughout.
pub fn gate_jshare_nnls_exactness() -> Result<()> {
    // Cone case: u = 2·e0 + 3·e1 → NNLS recovers (2, 3), J = 1.
    let mut unemb = vec![0.0f32; 4 * 4];
    unemb[0] = 1.0;
    unemb[5] = 1.0;
    let u = [2.0f32, 3.0, 0.0, 0.0];
    let j = j_share(&u, &unemb, &[0, 1], 4, 4).expect("cone J");
    // 1e-9, not 1e-12: the sub-solve carries a 1e-12×maxdiag Tikhonov
    // damping (the rank-deficiency guard) — exact recovery reads
    // 1 − ~2e-12, and the assertion must sit above the damping floor.
    if (j - 1.0).abs() > 1e-9 {
        bail!("cone case: J {j} != 1.0");
    }
    // Orthogonal case: u = e2, atoms {e0, e1} → J = 0.
    let u2 = [0.0f32, 0.0, 1.0, 0.0];
    let j0 = j_share(&u2, &unemb, &[0, 1], 4, 4).expect("orthogonal J");
    if j0.abs() > 1e-12 {
        bail!("orthogonal case: J {j0} != 0.0");
    }
    // Suppression case (the NNLS textbook example): c = (1, 0.9) on
    // Gram [[1, 0.9], [0.9, 1]] — the unconstrained solve is
    // (5.26, −4.74); NNLS must land on the single atom 0 with α = 1.
    let g = [1.0f64, 0.9, 0.9, 1.0];
    let c = [1.0f64, 0.9];
    let alpha = nnls_gram(&g, &c, 2);
    if (alpha[0] - 1.0).abs() > 1e-9 || alpha[1].abs() > 1e-9 {
        bail!("suppression case: α = {:?} (want [1, 0])", alpha);
    }
    Ok(())
}

/// NNLS degeneracy + tie law.
pub fn gate_jshare_ties() -> Result<()> {
    // Symmetric atoms (identical gradient): both enter, α = (1, 1).
    let g = [1.0f64, 0.0, 0.0, 1.0];
    let c = [1.0f64, 1.0];
    let alpha = nnls_gram(&g, &c, 2);
    if (alpha[0] - 1.0).abs() > 1e-9 || (alpha[1] - 1.0).abs() > 1e-9 {
        bail!("symmetric case: α = {:?} (want [1, 1])", alpha);
    }
    // Degenerate atom (zero diagonal): never selected, never divides.
    let g2 = [1.0f64, 0.0, 0.0, 0.0];
    let c2 = [1.0f64, 5.0];
    let alpha2 = nnls_gram(&g2, &c2, 2);
    if (alpha2[0] - 1.0).abs() > 1e-9 || alpha2[1].abs() > 1e-9 {
        bail!("degenerate atom: α = {:?} (want [1, 0])", alpha2);
    }
    Ok(())
}

/// The Rademacher control determinism + the top-k comparator.
pub fn gate_jshare_controls() -> Result<()> {
    let a = rademacher_direction(42, 64);
    let b = rademacher_direction(42, 64);
    let c = rademacher_direction(43, 64);
    if a != b {
        bail!("rademacher: same seed differs");
    }
    if a == c {
        bail!("rademacher: different seed identical (64 dims)");
    }
    if a.iter().any(|&x| x != 1.0 && x != -1.0) {
        bail!("rademacher: non-±1 entry");
    }
    // top_k_tokens matches the brute-force reference (the engineered
    // unit — the col IS unemb row 7 × 4).
    let fx = FixtureSource::new();
    let unit = &fx.cols[1][3 * EMBD..4 * EMBD];
    let normed = lens_norm(unit, &fx.gamma, 1e-6);
    let (toks, vals) = top_k_tokens(&normed, &fx.unemb, VOCAB, EMBD, TOP_K);
    let (bt, bv, _, _) = brute(&fx.unemb, unit, &fx.gamma, 1e-6);
    if toks != bt || vals != bv {
        bail!("top_k_tokens != brute force");
    }
    Ok(())
}

/// The T2 pass end-to-end on the fixture: the v1 artifact built from
/// `screen()`, the J-share pass over the SAME digested source, bucket
/// coverage, the engineered unit's J = 1, the v2 artifact byte
/// stability + shape, report determinism, and the digest-mismatch
/// refusal.
pub fn gate_jshare_e2e() -> Result<()> {
    let digest = [0xABu8; 32];
    let src = DigestedSource {
        inner: FixtureSource::new(),
        digest,
    };
    let scr = screen(&src)?;
    let v1 = build_artifact(&scr);
    let (_, _, _, _, n_units, _) = parse_header(&v1)?;
    if n_units != LAYERS * FF * 2 {
        bail!("fixture v1 units {n_units} != {}", LAYERS * FF * 2);
    }
    let js = run_jshare(&v1, &src)?;
    let total: usize = js.layers.iter().map(|l| l.units.len()).sum();
    if total != n_units {
        bail!("T2 units {total} != v1 {n_units}");
    }
    for l in &js.layers {
        if l.controls.len() != CONTROLS_PER_LAYER {
            bail!("layer {} controls {}", l.layer, l.controls.len());
        }
        for u in &l.units {
            if u.bucket == Bucket::Unassigned {
                bail!("unit {}:{} left Unassigned", u.layer, u.col);
            }
            if !(0.0..=1.0).contains(&u.jshare) {
                bail!("unit {}:{} J {} out of [0,1]", u.layer, u.col, u.jshare);
            }
            if u.tokens.len() != TOP_K {
                bail!("unit {}:{} tokens {}", u.layer, u.col, u.tokens.len());
            }
        }
    }
    // The engineered unit (layer 1 col 3 +): its column IS unemb row 7
    // × 4 — its own top atom spans the unit fully → J = 1.
    let eng = js.layers[1]
        .units
        .iter()
        .find(|u| u.col == 3 && u.sign == Sign::Plus)
        .expect("engineered + unit");
    if eng.tokens[0] != 7 {
        bail!("engineered unit top token {} != 7", eng.tokens[0]);
    }
    if (eng.jshare - 1.0).abs() > 1e-6 {
        bail!("engineered unit J {} != 1.0", eng.jshare);
    }
    // Bucket census sums to the unit total.
    let buckets: usize = js.rollups.iter().map(|r| r.high + r.middle + r.low).sum();
    if buckets != total {
        bail!("rollup buckets {buckets} != {total}");
    }
    // The v2 artifact: byte-stable across two runs + shape-pinned.
    let bytes = build_j_artifact(&js);
    let bytes2 = build_j_artifact(&run_jshare(&v1, &src)?);
    if bytes != bytes2 {
        bail!("v2 artifact not byte-stable across runs");
    }
    let want = HEADER_BYTES + total * J_ROW_BYTES + (4 + 4) + LAYERS * CONTROLS_PER_LAYER * 4;
    if bytes.len() != want {
        bail!("v2 artifact length {} != {want}", bytes.len());
    }
    let _ = j_artifact_digest(&bytes);
    // Report determinism.
    let hex = "ab".repeat(32);
    let r1 = render_j_report(&js, &hex);
    let r2 = render_j_report(&js, &hex);
    if r1 != r2 {
        bail!("T2 report not deterministic");
    }
    // The digest-mismatch refusal names its cause.
    let bad = DigestedSource {
        inner: FixtureSource::new(),
        digest: [0xCDu8; 32],
    };
    match run_jshare(&v1, &bad) {
        Err(e) => {
            if !e.to_string().contains("digest mismatch") {
                bail!("digest refusal named the wrong cause: {e}");
            }
        }
        Ok(_) => bail!("digest mismatch accepted"),
    }
    // The v1 row parser round-trips the supports.
    let rows = parse_v1_rows(&v1)?;
    if rows.len() != n_units {
        bail!("parse_v1_rows {} != {n_units}", rows.len());
    }
    // The v2 parser round-trips: the artifact parses back to the same
    // layer/unit structure, and the census renders.
    let (_, _, _, nl2, _, layers2) = crate::readout_jshare::parse_v2_rows(&bytes)?;
    if nl2 != LAYERS || layers2.len() != LAYERS {
        bail!("v2 parse layers {nl2}/{}, want {LAYERS}", layers2.len());
    }
    let total2: usize = layers2.iter().map(|l| l.units.len()).sum();
    if total2 != total {
        bail!("v2 parse units {total2} != {total}");
    }
    for (a, b) in js.layers.iter().zip(layers2.iter()) {
        if a.controls != b.controls {
            bail!("v2 parse controls drift at layer {}", a.layer);
        }
        for (ua, ub) in a.units.iter().zip(b.units.iter()) {
            if ua.bucket != ub.bucket
                || ua.jshare.to_bits() != ub.jshare.to_bits()
                || ua.tokens != ub.tokens
            {
                bail!("v2 parse unit drift at {}:{}", ua.layer, ua.col);
            }
        }
    }
    let census = crate::readout_jshare::signs_census(&js);
    if !census.contains("both-signs-High") || !census.contains("GATE:") {
        bail!("signs census missing its verdict lines");
    }
    Ok(())
}

/// The percentile → bucket edges (the pinned bars).
pub fn gate_jshare_bucket_edges() -> Result<()> {
    if bucket_of(HIGH_PERCENTILE) != Bucket::High {
        bail!("p = HIGH_PERCENTILE must read High");
    }
    if bucket_of(HIGH_PERCENTILE - 1e-9) == Bucket::High {
        bail!("just below the High bar must not read High");
    }
    if bucket_of(LOW_PERCENTILE - 1e-9) != Bucket::Low {
        bail!("below the median must read Low");
    }
    if bucket_of(LOW_PERCENTILE) == Bucket::Low {
        bail!("exactly the median is Middle (the Low bar is open)");
    }
    Ok(())
}

/// Run every gate; returns the receipt lines (the bin's `--self-test`
/// and the cargo tests share this exact source).
pub fn run_all() -> Result<Vec<String>> {
    type Gate = fn() -> Result<()>;
    let gates: [(&str, Gate); 8] = [
        ("selection=brute-force", gate_selection_matches_brute_force),
        ("engineered-unit", gate_engineered_unit),
        ("sign-trick=bit-exact", gate_sign_trick_bit_exact),
        ("ties=first-index", gate_ties_first_index),
        ("bands=pinned", gate_band_formula),
        ("artifact=stable", gate_artifact_stable),
        ("refusals=loud", gate_refusals_loud),
        ("lens-norm=f64-reference", gate_lens_norm_reference),
    ];
    let mut receipt = Vec::with_capacity(gates.len() + 6);
    for (name, gate) in gates {
        gate()?;
        receipt.push(format!("  PASS {name}"));
    }
    gate_report_deterministic()?;
    receipt.push("  PASS report=deterministic".to_string());
    // T2 (Issue 1250): the J-share pass.
    gate_jshare_nnls_exactness()?;
    receipt.push("  PASS jshare-nnls=exact".to_string());
    gate_jshare_ties()?;
    receipt.push("  PASS jshare-degenerate+ties".to_string());
    gate_jshare_controls()?;
    receipt.push("  PASS jshare-controls=deterministic".to_string());
    gate_jshare_bucket_edges()?;
    receipt.push("  PASS jshare-bucket-edges".to_string());
    gate_jshare_e2e()?;
    receipt.push("  PASS jshare-e2e".to_string());
    Ok(receipt)
}
