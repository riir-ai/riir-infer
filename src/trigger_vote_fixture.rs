//! The hermetic fixture gates (Issue 1251 T1 G1) — the
//! [`readout_screen_fixture`] shape: feature-gated (not test-gated) so
//! the `trigger-vote --self-test` bin posture drives the SAME gate
//! source the cargo tests assert; an unarmed build compiles it to
//! nothing.
//!
//! A tiny deterministic source on the SAME [`VoteTensors`] trait the
//! GGUF source implements: the planted-trigger recovery, the band
//! chaining rule, the tie law, the identity fast path, the zero-score
//! abstention, the row-skip accounting, artifact byte-stability, the
//! geometry refusal battery. Deterministic weights via SplitMix64 (the
//! house fixture RNG, local copy — the 1250 fixture's is private by
//! the same convention). No external data, no fixture files.

use crate::trigger_vote::{
    ARTIFACT_MAGIC, ARTIFACT_VERSION, BAND_WIDTH, HEADER_BYTES, RANKED_N, ROW_BYTES,
    VOTES_PER_NEURON, Verdict, VoteTensors, artifact_digest, build_artifact, parse_header,
    trigger_vote,
};
use anyhow::{Result, ensure};

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

pub const LAYERS: usize = 3;
pub const FF: usize = 16;
pub const EMBD: usize = 8;
pub const VOCAB: usize = 32;

/// The fixture source: `LAYERS` layers of `FF` neurons over `EMBD`
/// dims, `VOCAB` tokens. Base weights and the embedding table are
/// builder-controlled; `plant_gate_rows` writes finetune deltas.
#[derive(Clone)]
pub struct FixtureVoteSource {
    pub gate: Vec<Vec<f32>>, // [layer][FF × EMBD]
    pub up: Vec<Vec<f32>>,   // [layer][FF × EMBD]
    pub emb: Vec<f32>,       // [VOCAB × EMBD]
    pub label: String,
}

impl FixtureVoteSource {
    /// All-zero source (the base of every planted fixture — deltas
    /// equal the planted rows exactly, so the active census is exact).
    pub fn zero() -> Self {
        Self {
            gate: vec![vec![0.0; FF * EMBD]; LAYERS],
            up: vec![vec![0.0; FF * EMBD]; LAYERS],
            emb: vec![0.0; VOCAB * EMBD],
            label: "fixture:zero".into(),
        }
    }

    /// Random source (SplitMix64 `seed`); `boost` scales one embedding
    /// row (the planted trigger's dominance lever).
    pub fn random(seed: u64, boost: Option<(usize, f32)>) -> Self {
        let mut rng = SplitMix64(seed);
        let mut emb = vec![0.0f32; VOCAB * EMBD];
        for v in emb.iter_mut() {
            *v = rng.f32v();
        }
        if let Some((row, scale)) = boost {
            for v in &mut emb[row * EMBD..(row + 1) * EMBD] {
                *v *= scale;
            }
        }
        let mut gate = Vec::with_capacity(LAYERS);
        let mut up = Vec::with_capacity(LAYERS);
        for _ in 0..LAYERS {
            let mut g = vec![0.0f32; FF * EMBD];
            for v in g.iter_mut() {
                *v = rng.f32v();
            }
            let mut u = vec![0.0f32; FF * EMBD];
            for v in u.iter_mut() {
                *v = rng.f32v();
            }
            gate.push(g);
            up.push(u);
        }
        Self {
            gate,
            up,
            emb,
            label: format!("fixture:random({seed:#x})"),
        }
    }

    /// The block source: all-zero weights + one-hot block embeddings
    /// (`e_t = 1.0` at dim `t % EMBD`). Dots become EXACT: a delta row
    /// supported on dims D votes for exactly the tokens hitting D, all
    /// scores equal within a dim — the tie law and the disjoint-band
    /// constructions are pinned by hand, never by seed luck.
    pub fn block(embedding_dims: usize) -> Self {
        let mut emb = vec![0.0f32; VOCAB * EMBD];
        for t in 0..VOCAB {
            emb[t * EMBD + (t % embedding_dims)] = 1.0;
        }
        Self {
            gate: vec![vec![0.0; FF * EMBD]; LAYERS],
            up: vec![vec![0.0; FF * EMBD]; LAYERS],
            emb,
            label: "fixture:block".into(),
        }
    }

    /// Write `dg` into the gate rows `rows` of `layer` (the delta IS
    /// the finetune row — the base stays whatever it was).
    pub fn plant_gate_rows(&mut self, layer: usize, rows: &[usize], dg: &[f32]) {
        assert_eq!(dg.len(), EMBD, "delta length must be EMBD");
        assert!(layer < LAYERS, "layer out of range");
        for &i in rows {
            assert!(i < FF, "row out of range");
            self.gate[layer][i * EMBD..(i + 1) * EMBD].copy_from_slice(dg);
        }
    }

    /// The delta row `scale × e_token` (the planted trigger direction).
    pub fn trigger_dir(&self, token: usize, scale: f32) -> Vec<f32> {
        self.emb[token * EMBD..(token + 1) * EMBD]
            .iter()
            .map(|&v| scale * v)
            .collect()
    }
}

impl VoteTensors for FixtureVoteSource {
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
    fn gate_rows(&self, layer: usize) -> Result<Vec<f32>> {
        self.gate
            .get(layer)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("fixture layer {layer} out of range"))
    }
    fn up_rows(&self, layer: usize) -> Result<Vec<f32>> {
        self.up
            .get(layer)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("fixture layer {layer} out of range"))
    }
    fn emb_rows(&self) -> Result<Vec<f32>> {
        Ok(self.emb.clone())
    }
    fn label(&self) -> String {
        self.label.clone()
    }
}

// ── The gates ────────────────────────────────────────────────────────

/// The planted trigger: layer 1's every gate row moves `10 × e_7`
/// (the boosted embedding makes the direction dominate every other
/// token's dot). Asserts: verdict Voted, the whole layer active, band
/// exactly `[1..=1]`, token 7 ranked first with a full vote count.
pub fn gate_planted_trigger_ranked_first() -> Result<()> {
    // The BLOCK source keeps the math exact: e_t = one-hot(t % EMBD)
    // makes the cross-terms exactly zero, so the planted direction
    // votes for its own token group and nothing else — no seed luck,
    // no statistical near-ties. The planted rows carry token 7's
    // direction (dim 7): exactly tokens {7, 15, 23, 31} score > 0
    // (10 per planted neuron), the other 28 abstain (score 0 → no
    // vote), and the first-index law ranks 7 first among the four
    // full-vote tokens.
    let base = FixtureVoteSource::block(EMBD);
    let mut ft = base.clone();
    let planted: Vec<usize> = (0..FF / 2).collect();
    let dg = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0];
    for &row in &planted {
        ft.plant_gate_rows(1, &[row], &dg);
    }
    let v = trigger_vote(&base, &ft)?;
    ensure!(
        v.verdict == Verdict::Voted,
        "verdict {:?} != Voted",
        v.verdict
    );
    ensure!(
        v.n_active_rows == FF as u64 / 2,
        "n_active_rows {}",
        v.n_active_rows
    );
    ensure!(v.band == Some((1, 1)), "band {:?} != [1..=1]", v.band);
    ensure!(
        v.layers[1].n_active == FF / 2,
        "layer 1 n_active {} != {}",
        v.layers[1].n_active,
        FF / 2
    );
    ensure!(
        v.layers[0].n_active == 0 && v.layers[2].n_active == 0,
        "untouched layers active"
    );
    ensure!(
        v.layers[1].votes[7] == (FF / 2) as u32,
        "trigger votes {} != {}",
        v.layers[1].votes[7],
        FF / 2
    );
    ensure!(
        !v.ranked.is_empty() && v.ranked[0] == (7, (FF / 2) as u32),
        "ranked[0] {:?} != (7, {})",
        v.ranked.first(),
        FF / 2
    );
    // Exactly the four tokens hitting dim 7 vote; everything else
    // abstains (score exactly 0).
    let voted: Vec<u32> = (0..VOCAB as u32)
        .filter(|&t| v.layers[1].votes[t as usize] > 0)
        .collect();
    ensure!(voted == [7, 15, 23, 31], "voted {voted:?} != [7,15,23,31]");
    ensure!(v.ranked.len() == 4, "ranked len {}", v.ranked.len());
    Ok(())
}

/// The band chaining rule: layers 0 and 1 carry the SAME delta
/// (identical vote profiles → overlap 12 ≥ 10 → linked); layer 2
/// carries a delta on disjoint dims (overlap 0 → break). Asserts the
/// band is exactly `[0..=1]` and the ranking pools both layers.
pub fn gate_band_chaining_contiguous() -> Result<()> {
    let base = FixtureVoteSource::block(EMBD);
    let mut ft = base.clone();
    let mut front = vec![0.0f32; EMBD];
    front[0] = 3.0;
    front[1] = 2.0;
    front[2] = 1.0;
    let mut back = vec![0.0f32; EMBD];
    back[5] = 3.0;
    back[6] = 2.0;
    back[7] = 1.0;
    let all: Vec<usize> = (0..FF).collect();
    ft.plant_gate_rows(0, &all, &front);
    ft.plant_gate_rows(1, &all, &front);
    ft.plant_gate_rows(2, &all, &back);
    let v = trigger_vote(&base, &ft)?;
    ensure!(v.verdict == Verdict::Voted, "verdict {:?}", v.verdict);
    // Every layer: exactly the 12 tokens hitting the planted dims vote,
    // each with the full FF votes.
    for l in &v.layers {
        let voted: Vec<usize> = (0..VOCAB).filter(|&t| l.votes[t] > 0).collect();
        ensure!(voted.len() == 12, "layer {} voted {}", l.layer, voted.len());
        ensure!(
            voted.iter().all(|&t| l.votes[t] == FF as u32),
            "layer {} vote counts wrong",
            l.layer
        );
    }
    ensure!(
        v.layers[1].overlap_prev == Some(12),
        "overlap(1) {:?} != 12",
        v.layers[1].overlap_prev
    );
    ensure!(
        v.layers[2].overlap_prev == Some(0),
        "overlap(2) {:?} != 0",
        v.layers[2].overlap_prev
    );
    ensure!(v.band == Some((0, 1)), "band {:?} != [0..=1]", v.band);
    // Pooled ranking: both front layers' 12 tokens at 2×FF votes each,
    // token-asc among ties.
    ensure!(v.ranked.len() == 12, "ranked len {}", v.ranked.len());
    ensure!(
        v.ranked[0] == (0, (2 * FF) as u32),
        "ranked[0] {:?}",
        v.ranked[0]
    );
    let expected: Vec<u32> = (0..VOCAB as u32)
        .filter(|t| (t % EMBD as u32) < 3)
        .collect();
    let got: Vec<u32> = v.ranked.iter().map(|(t, _)| *t).collect();
    ensure!(got == expected, "ranked order {got:?} != {expected:?}");
    Ok(())
}

/// The tie law: a delta supported equally on dims 3 and 4 gives eight
/// tokens the same exact score — the neuron's selection must order
/// them by ascending token id (first-index rule).
pub fn gate_ties_first_index() -> Result<()> {
    let base = FixtureVoteSource::block(EMBD);
    let mut ft = base.clone();
    let mut dg = vec![0.0f32; EMBD];
    dg[3] = 5.0;
    dg[4] = 5.0;
    ft.plant_gate_rows(0, &[0], &dg);
    let v = trigger_vote(&base, &ft)?;
    ensure!(v.rows.len() == 1, "rows {}", v.rows.len());
    let toks = &v.rows[0].tokens;
    ensure!(toks.len() == 8, "selected {}", toks.len());
    ensure!(
        toks[..4] == [3, 4, 11, 12],
        "tie order {toks:?} does not start [3, 4, 11, 12]"
    );
    ensure!(
        v.layers[0].top[..4] == [3, 4, 11, 12],
        "layer top tie order {:?}",
        v.layers[0].top
    );
    Ok(())
}

/// The identity control: base == finetune (same weights, fixture
/// digests `None` so the FULL pass runs) → every row skipped,
/// `ZeroDelta`, no band, no ranking.
pub fn gate_identity_zero_delta() -> Result<()> {
    let base = FixtureVoteSource::random(0x1251_0002, None);
    let ft = base.clone();
    let v = trigger_vote(&base, &ft)?;
    ensure!(
        v.verdict == Verdict::ZeroDelta,
        "verdict {:?} != ZeroDelta",
        v.verdict
    );
    ensure!(v.n_active_rows == 0, "n_active_rows {}", v.n_active_rows);
    ensure!(v.rows.is_empty(), "rows nonempty");
    ensure!(v.band.is_none(), "band {:?}", v.band);
    ensure!(v.ranked.is_empty(), "ranked nonempty");
    ensure!(v.layers.iter().all(|l| l.n_active == 0), "layer actives");
    ensure!(v.layers.iter().all(|l| l.top.is_empty()), "layer tops");
    Ok(())
}

/// The zero-score abstention: embeddings supported on dims 0..=3
/// only, delta rows supported on dim 6 — every dot lands exactly 0.0,
/// nothing votes, and the verdict is `ZeroVotes` (rows CHANGED —
/// distinct from `ZeroDelta`).
pub fn gate_zero_scores_no_votes() -> Result<()> {
    let base = FixtureVoteSource::block(4);
    let mut ft = base.clone();
    let mut dg = vec![0.0f32; EMBD];
    dg[6] = 4.0;
    let all: Vec<usize> = (0..FF).collect();
    ft.plant_gate_rows(1, &all, &dg);
    let v = trigger_vote(&base, &ft)?;
    ensure!(
        v.verdict == Verdict::ZeroVotes,
        "verdict {:?} != ZeroVotes",
        v.verdict
    );
    ensure!(
        v.n_active_rows == FF as u64,
        "n_active_rows {}",
        v.n_active_rows
    );
    ensure!(v.band.is_none(), "band {:?}", v.band);
    ensure!(v.ranked.is_empty(), "ranked nonempty");
    ensure!(
        v.rows.is_empty(),
        "rows nonempty (zero scores must not vote)"
    );
    Ok(())
}

/// The row-skip accounting: only the planted rows are active, and only
/// they vote — the untouched rows contribute nothing to any count.
pub fn gate_row_skip_only_active_votes() -> Result<()> {
    let base = FixtureVoteSource::block(EMBD);
    let mut ft = base.clone();
    let mut dg = vec![0.0f32; EMBD];
    dg[0] = 3.0;
    dg[1] = 2.0;
    dg[2] = 1.0;
    ft.plant_gate_rows(1, &[2, 9], &dg);
    let v = trigger_vote(&base, &ft)?;
    ensure!(v.n_active_rows == 2, "n_active_rows {}", v.n_active_rows);
    ensure!(
        v.layers[1].n_active == 2 && v.layers[0].n_active == 0 && v.layers[2].n_active == 0,
        "per-layer actives {:?}",
        v.layers.iter().map(|l| l.n_active).collect::<Vec<_>>()
    );
    ensure!(v.rows.len() == 2, "rows {}", v.rows.len());
    ensure!(
        v.rows.iter().all(|r| r.layer == 1),
        "rows from the wrong layer"
    );
    // Both planted rows carry the same delta → the same 12 tokens, 2
    // votes each.
    ensure!(v.ranked.len() == 12, "ranked len {}", v.ranked.len());
    ensure!(
        v.ranked.iter().all(|&(_, votes)| votes == 2),
        "vote counts {:?}",
        v.ranked
    );
    Ok(())
}

/// Artifact byte-stability: two independently constructed identical
/// pairs vote to byte-identical artifacts (no hidden state, no
/// scheduling dependence).
pub fn gate_artifact_stable() -> Result<()> {
    let mk = || -> Result<crate::trigger_vote::TriggerVote> {
        let base = FixtureVoteSource::random(0x1251_0003, Some((11, 3.0)));
        let mut ft = base.clone();
        let dg = base.trigger_dir(11, 10.0);
        ft.plant_gate_rows(1, &[3, 4, 5], &dg);
        trigger_vote(&base, &ft)
    };
    let a = build_artifact(&mk()?);
    let b = build_artifact(&mk()?);
    ensure!(a == b, "artifact bytes differ across constructions");
    ensure!(artifact_digest(&a) == artifact_digest(&b), "digests differ");
    ensure!(&a[..6] == ARTIFACT_MAGIC, "magic prefix wrong");
    Ok(())
}

/// The geometry refusal battery: a pair disagreeing on `n_ff` refuses
/// loud naming the field; layer-census and vocab mismatches likewise.
pub fn gate_dims_refusal() -> Result<()> {
    // n_ff mismatch: a 12-row finetune against a 16-row base.
    let base = FixtureVoteSource::block(EMBD);
    let mut ft = FixtureVoteSource::block(EMBD);
    ft.gate = vec![vec![0.0; 12 * EMBD]; LAYERS];
    ft.up = vec![vec![0.0; 12 * EMBD]; LAYERS];
    // Shrink via a slimmed trait impl is overkill — instead hand-roll a
    // wrapper source with n_ff = 12.
    struct Slim(FixtureVoteSource);
    impl VoteTensors for Slim {
        fn n_layers(&self) -> usize {
            LAYERS
        }
        fn n_ff(&self) -> usize {
            12
        }
        fn n_embd(&self) -> usize {
            EMBD
        }
        fn vocab(&self) -> usize {
            VOCAB
        }
        fn gate_rows(&self, layer: usize) -> Result<Vec<f32>> {
            Ok(self.0.gate[layer].clone())
        }
        fn up_rows(&self, layer: usize) -> Result<Vec<f32>> {
            Ok(self.0.up[layer].clone())
        }
        fn emb_rows(&self) -> Result<Vec<f32>> {
            Ok(self.0.emb.clone())
        }
        fn label(&self) -> String {
            "fixture:slim".into()
        }
    }
    let slim = Slim(ft);
    let err = trigger_vote(&base, &slim)
        .err()
        .ok_or_else(|| anyhow::anyhow!("n_ff mismatch did not refuse"))?;
    ensure!(
        err.to_string().contains("n_ff mismatch"),
        "refusal text {err:#} does not name n_ff"
    );
    // vocab mismatch via a shrunk embedding table.
    struct SlimVocab(FixtureVoteSource);
    impl VoteTensors for SlimVocab {
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
            24
        }
        fn gate_rows(&self, layer: usize) -> Result<Vec<f32>> {
            Ok(self.0.gate[layer].clone())
        }
        fn up_rows(&self, layer: usize) -> Result<Vec<f32>> {
            Ok(self.0.up[layer].clone())
        }
        fn emb_rows(&self) -> Result<Vec<f32>> {
            Ok(self.0.emb[..24 * EMBD].to_vec())
        }
        fn label(&self) -> String {
            "fixture:slim-vocab".into()
        }
    }
    let err = trigger_vote(&base, &SlimVocab(base.clone()))
        .err()
        .ok_or_else(|| anyhow::anyhow!("vocab mismatch did not refuse"))?;
    ensure!(
        err.to_string().contains("vocab mismatch"),
        "refusal text {err:#} does not name vocab"
    );
    Ok(())
}

/// The artifact layout is pinned by const: header bytes, row bytes,
/// and the total length formula all hold on a real (fixture-sized)
/// artifact.
pub fn gate_layout_pinned() -> Result<()> {
    ensure!(HEADER_BYTES == 126, "HEADER_BYTES {HEADER_BYTES}");
    ensure!(ROW_BYTES == 128, "ROW_BYTES {ROW_BYTES}");
    ensure!(VOTES_PER_NEURON == 15, "VOTES_PER_NEURON");
    ensure!(BAND_WIDTH == 100, "BAND_WIDTH");
    ensure!(RANKED_N == 4096, "RANKED_N");
    let base = FixtureVoteSource::block(EMBD);
    let mut ft = base.clone();
    let mut dg = vec![0.0f32; EMBD];
    dg[2] = 1.0;
    ft.plant_gate_rows(0, &[1], &dg);
    let v = trigger_vote(&base, &ft)?;
    let bytes = build_artifact(&v);
    let want = HEADER_BYTES + v.n_layers * 4 + v.n_layers * v.n_ff * ROW_BYTES + v.ranked.len() * 8;
    ensure!(bytes.len() == want, "len {} != {want}", bytes.len());
    Ok(())
}

/// Parse roundtrip: the built artifact's header decodes to the vote's
/// own scalars (verdict, band sentinels, active count, digests).
pub fn gate_parse_roundtrip() -> Result<()> {
    let base = FixtureVoteSource::random(0x1251_0004, None);
    let mut ft = base.clone();
    let dg = base.trigger_dir(5, 8.0);
    ft.plant_gate_rows(2, &[0, 7], &dg);
    let v = trigger_vote(&base, &ft)?;
    let bytes = build_artifact(&v);
    let h = parse_header(&bytes)?;
    ensure!(h.version == ARTIFACT_VERSION, "version");
    ensure!(h.votes_per_neuron == VOTES_PER_NEURON, "votes_per_neuron");
    ensure!(h.n_layers == v.n_layers, "n_layers");
    ensure!(h.n_ff == v.n_ff, "n_ff");
    ensure!(h.n_embd == v.n_embd, "n_embd");
    ensure!(h.vocab == v.vocab, "vocab");
    ensure!(h.n_active_rows == v.n_active_rows, "n_active_rows");
    let (bs, be) = v.band.unwrap_or((u32::MAX as usize, 0));
    ensure!(h.band_start == bs as u32 && h.band_end == be as u32, "band");
    ensure!(h.verdict == v.verdict, "verdict");
    ensure!(h.base_digest == [0u8; 32], "fixture digests must ship zero");
    // A truncated artifact refuses loud.
    ensure!(
        parse_header(&bytes[..HEADER_BYTES - 1]).is_err(),
        "truncated artifact parsed"
    );
    // A foreign magic refuses loud.
    let mut bad = bytes.clone();
    bad[0] = b'X';
    ensure!(parse_header(&bad).is_err(), "foreign magic parsed");
    Ok(())
}

/// A fixture gate: the zero-arg check returning receipts or a typed
/// failure (the run_all table's element shape).
type FixtureGate = fn() -> Result<()>;

/// Run every gate, returning the receipt lines (the bin's `--self-test`
/// posture drives the SAME source the cargo tests assert).
pub fn run_all() -> Result<Vec<String>> {
    let gates: [(&str, FixtureGate); 10] = [
        (
            "planted_trigger_ranked_first",
            gate_planted_trigger_ranked_first,
        ),
        ("band_chaining_contiguous", gate_band_chaining_contiguous),
        ("ties_first_index", gate_ties_first_index),
        ("identity_zero_delta", gate_identity_zero_delta),
        ("zero_scores_no_votes", gate_zero_scores_no_votes),
        (
            "row_skip_only_active_votes",
            gate_row_skip_only_active_votes,
        ),
        ("artifact_stable", gate_artifact_stable),
        ("dims_refusal", gate_dims_refusal),
        ("layout_pinned", gate_layout_pinned),
        ("parse_roundtrip", gate_parse_roundtrip),
    ];
    let mut receipts = Vec::with_capacity(gates.len());
    for (name, gate) in gates {
        gate().map_err(|e| anyhow::anyhow!("gate {name}: {e:#}"))?;
        receipts.push(format!("{name} PASS"));
    }
    Ok(receipts)
}
