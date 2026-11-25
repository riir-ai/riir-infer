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

use crate::readout_screen::{
    Band, Bucket, ReadoutTensors, Sign, TOP_K, artifact_digest, band_of, build_artifact, lens_norm,
    ordered_dot_f64, parse_header, screen,
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
    let mut receipt = Vec::with_capacity(gates.len() + 1);
    for (name, gate) in gates {
        gate()?;
        receipt.push(format!("  PASS {name}"));
    }
    gate_report_deterministic()?;
    receipt.push("  PASS report=deterministic".to_string());
    Ok(receipt)
}
