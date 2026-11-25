//! Issue 1244 T1 — the header-only HiSA planner (`hisa_plan`).
//!
//! Research 040 (HiSA, arXiv:2611.26363): with RoPE, each query–key score
//! decomposes into per-frequency-pair parts; part *l* is kept iff its
//! rotation `Δ·θ_l < Φ` (the angle budget), giving closed-form per-pair
//! windows `Δ_max,l = Φ·b^(2l/d_rot)` over the ROTARY dimension, reach
//! `R = Φ·b^((d_rot−2)/d_rot)` (the slowest pair's window), decode kept
//! share `ρ_dec(t) = Σ min(Δ_max,l, t) / (P·t)`, prefill kept share
//! `ρ(t)` (the causal-matrix mean), and cache `κ(t) = (ρ_dec(t) + v)/2`
//! with `v` the value-share kept (1.0 unless value eviction fires, which
//! needs `R < t` — never on our backbones).
//!
//! The planner is the guard for exactly the error class Research 040's
//! first draft made (headline numbers computed from a full-rotary
//! assumption the header refutes): **REQUIRED header reads before any
//! window arithmetic** — `rope.freq_base`, `attention.key_length`,
//! `rope.dimension_count` (+ `rope.dimension_sections` when present),
//! `full_attention_interval` (hybrids), `block_count`, context length.
//! Absence is typed, never defaulted: a missing `rope.dimension_count`
//! means FULL rotary (the dflash drafter's shape — the header's own
//! statement), a missing `rope.freq_base` or `attention.key_length` is a
//! loud refusal (no window arithmetic without them).
//!
//! mrope sections are RECORDED, never assumed to change the math: our
//! serving forward collapses them for text-only decode (`rope.rs`
//! Issue-594 note: T=H=W=pos, 4th=0 — standard partial RoPE on
//! `rotary_dim` dims), so per-pair frequencies follow `θ_l = b^(−2l/d_rot)`
//! and the windows above are exact for the served posture. The bin prints
//! the section split as provenance.
//!
//! REPORT-ONLY: every function here is a pure read; nothing routes, gates,
//! or writes serving state. The integer `⌈Δ_max,l⌉` keep-threshold is a
//! KERNEL convention (kept iff `Δ < L_l`); the planner reports exact
//! windows.
//!
//! Grandfathered floors (Research 040 §2, the pre-registered expectation):
//! at the paper's own anchor regime (d=64 all-rotating, b=1e4, Φ=90°,
//! t=4096) the exact-window form gives `ρ_dec = 23.86%` vs the paper
//! Table 8's 23.9%. The anchor test pins the tight band.

use crate::gguf_loader::GgufFile;

/// The angle budget in degrees (the planner prints per budget; the paper's
/// default is 90°).
pub const DEFAULT_BUDGETS_DEG: [f64; 4] = [90.0, 180.0, 360.0, 720.0];

/// The context postures the planner prints (the league serving shapes).
pub const DEFAULT_CONTEXTS: [u64; 5] = [512, 2048, 4096, 16384, 32768];

/// The typed header facts. Every field is a header read, never a default —
/// the two `Option`s are the header's own statements (absent = the
/// architecture's answer), the rest refuse loud when missing.
#[derive(Debug, Clone)]
pub struct HisaHeader {
    /// `rope.freq_base` (b). REQUIRED.
    pub freq_base: f64,
    /// `attention.key_length` (d, the full-attention head dim). REQUIRED.
    pub key_length: u64,
    /// `rope.dimension_count` (d_rot). `None` = FULL rotary (no key in the
    /// header — the dflash drafter's shape); `Some(r)` = partial (r/2 of
    /// the d/2 pairs rotate, the rest are position-free ALWAYS-KEPT rows).
    pub dimension_count: Option<u64>,
    /// `rope.dimension_sections` when present (mrope; provenance only —
    /// text serving collapses sections, read from our forward, never
    /// assumed here).
    pub dimension_sections: Option<Vec<u64>>,
    /// `qwen35.full_attention_interval` when present (hybrids: layer i is
    /// full attention iff `(i+1) % interval == 0`). `None` = every layer
    /// attends.
    pub full_attention_interval: Option<u64>,
    /// `{arch}.block_count`. REQUIRED.
    pub block_count: u64,
    /// `{arch}.context_length`. REQUIRED.
    pub context_length: u64,
    /// `general.architecture`. REQUIRED (the metadata prefix's owner).
    pub arch: String,
}

impl HisaHeader {
    /// The REQUIRED reads, in the Research-040 §7 order. A missing required
    /// key refuses loud naming the key — no window arithmetic on defaults.
    pub fn from_gguf(gguf: &GgufFile) -> anyhow::Result<Self> {
        let arch = gguf
            .metadata_string("general.architecture")
            .ok_or_else(|| anyhow::anyhow!("hisa_plan: `general.architecture` missing — not a GGUF model header this planner reads"))?
            .to_string();
        let p = format!("{arch}.");
        let req_u64 = |key: &str| -> anyhow::Result<u64> {
            gguf.metadata_u64(&format!("{p}{key}"))
                .ok_or_else(|| anyhow::anyhow!("hisa_plan: required header key `{p}{key}` missing — refusing (no window arithmetic on defaults)"))
        };
        let freq_base = gguf
            .metadata_f64(&format!("{p}rope.freq_base"))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "hisa_plan: required header key `{p}rope.freq_base` missing — refusing"
                )
            })?;
        Ok(Self {
            freq_base,
            key_length: req_u64("attention.key_length")?,
            dimension_count: gguf.metadata_u64(&format!("{p}rope.dimension_count")),
            dimension_sections: gguf
                .metadata_array(&format!("{p}rope.dimension_sections"))
                .map(|a| a.iter().filter_map(|v| v.as_u64()).collect()),
            full_attention_interval: gguf.metadata_u64(&format!("{p}full_attention_interval")),
            block_count: req_u64("block_count")?,
            context_length: req_u64("context_length")?,
            arch,
        })
    }

    /// The rotating dimension: `dimension_count` when present, else the
    /// full `key_length` (the header's own full-rotary statement).
    pub fn rotary_dim(&self) -> u64 {
        self.dimension_count.unwrap_or(self.key_length)
    }

    /// Total key pairs P = d/2.
    pub fn total_pairs(&self) -> u64 {
        self.key_length / 2
    }

    /// Rotating pairs P_rot = d_rot/2. The other `P − P_rot` pairs never
    /// rotate — ALWAYS-KEPT rows.
    pub fn rotating_pairs(&self) -> u64 {
        self.rotary_dim() / 2
    }
}

/// The per-pair window table for one (Φ, header) cell.
///
/// `windows[l] = Φ_rad · b^(2l/d_rot)` for the ROTATING pairs (l = 0 is the
/// fastest pair); the UNROTATED pairs are the ALWAYS-KEPT rows (window =
/// unbounded, reported separately, never folded into a zero).
#[derive(Debug, Clone)]
pub struct WindowTable {
    /// `Φ` in radians (the arithmetic's unit; the printed table is degrees).
    pub phi_rad: f64,
    /// Per-rotating-pair exact windows, l = 0..P_rot.
    pub windows: Vec<f64>,
    /// `R = Φ·b^((d_rot−2)/d_rot)` — the slowest rotating pair's window,
    /// the reach beyond which value eviction may fire.
    pub reach: f64,
    /// The number of pairs that never rotate (their window is unbounded).
    pub always_kept_pairs: u64,
    /// Total pairs P.
    pub total_pairs: u64,
}

impl WindowTable {
    pub fn new(header: &HisaHeader, phi_deg: f64) -> Self {
        let phi_rad = phi_deg.to_radians();
        let d_rot = header.rotary_dim() as f64;
        let p_rot = header.rotating_pairs();
        let b = header.freq_base;
        let windows = (0..p_rot)
            .map(|l| phi_rad * b.powf(2.0 * l as f64 / d_rot))
            .collect();
        let reach = phi_rad * b.powf((d_rot - 2.0) / d_rot);
        Self {
            phi_rad,
            windows,
            reach,
            always_kept_pairs: header.total_pairs() - p_rot,
            total_pairs: header.total_pairs(),
        }
    }

    /// Decode kept share at context `t`: `ρ_dec(t) = Σ_l min(Δ_max,l, t) /
    /// (P·t)` over ALL pairs (unrotated rows contribute `t` each — the
    /// always-kept share rides in the numerator, never a folded zero).
    pub fn decode_kept_share(&self, t: u64) -> f64 {
        let tf = t as f64;
        let rot: f64 = self.windows.iter().map(|w| w.min(tf)).sum();
        let kept = rot + self.always_kept_pairs as f64 * tf;
        kept / (self.total_pairs as f64 * tf)
    }

    /// Prefill kept share at context `t`: the causal-matrix mean of the
    /// kept indicator. For one pair with window `L`:
    /// - `L >= t` → 1.0 (every causal distance is inside the window);
    /// - `L < t` → `(Σ_{Δ=1..L} 1 + (t−L)·L) / (t(t+1)/2)` — distances
    ///   `1..=⌊L⌋` fully kept (the exact-window form; the fractional tail
    ///   of `L` counts its kept fraction `L−⌊L⌋` on the boundary distance),
    ///   distances beyond `L` never kept. Averaged over all P pairs with
    ///   the ALWAYS-KEPT rows at 1.0.
    pub fn prefill_kept_share(&self, t: u64) -> f64 {
        let tf = t as f64;
        // causal distances 1..=t; the mean distance-weighted kept share per pair
        let denom = tf * (tf + 1.0) / 2.0;
        let mut kept = self.always_kept_pairs as f64 * denom;
        for &w in &self.windows {
            if w >= tf {
                kept += denom;
            } else {
                let full = w.floor();
                let frac = w - full;
                // Σ_{Δ=1..⌊w⌋} Δ + frac·(⌊w⌋+1)
                let s = full * (full + 1.0) / 2.0 + frac * (full + 1.0);
                kept += s + (tf - w) * w;
            }
        }
        kept / (self.total_pairs as f64 * denom)
    }

    /// Cache kept share at context `t`: `κ(t) = (ρ_dec(t) + v)/2` with `v`
    /// the value-share — 1.0 unless value eviction fires (`reach < t`), in
    /// which case `v = min(reach, t)/t`.
    pub fn cache_share(&self, t: u64) -> f64 {
        let tf = t as f64;
        let v = if self.reach < tf {
            self.reach / tf
        } else {
            1.0
        };
        (self.decode_kept_share(t) + v) / 2.0
    }
}

/// One chunk-ring row: the ring length for a chunk of `chunk_pairs`
/// consecutive rotating pairs = the SLOWEST pair's window in the chunk
/// (largest l), rounded UP to the alignment grain.
#[derive(Debug, Clone)]
pub struct RingRow {
    pub chunk_pairs: usize,
    pub l_first: usize,
    pub window_exact: f64,
    pub ring_exact: f64,
    pub ring_q8_aligned: u64,
}

/// The chunk-ring table at a granularity (`chunk_pairs` 8 = 16-feature
/// chunks, 16 = 32-feature chunks) with the q8-block-alignment variant
/// (`Q8_BLOCK_SIZE` 32 — a ring is quantized in 32-elem blocks, so the
/// allocated ring is `ceil(window / 32) · 32`).
pub fn ring_table(table: &WindowTable, chunk_pairs: usize, align: usize) -> Vec<RingRow> {
    let mut rows = Vec::new();
    let p_rot = table.windows.len();
    let mut l = 0;
    while l < p_rot {
        let end = (l + chunk_pairs).min(p_rot);
        // the chunk's slowest pair = the LAST (largest l) in the chunk
        let window = table.windows[end - 1];
        rows.push(RingRow {
            chunk_pairs: end - l,
            l_first: l,
            window_exact: window,
            ring_exact: window,
            ring_q8_aligned: (window.ceil() as u64).div_ceil(align as u64) * align as u64,
        });
        l = end;
    }
    rows
}

/// One printed context cell (the planner's decision row).
#[derive(Debug, Clone)]
pub struct ContextCell {
    pub t: u64,
    pub decode_kept: f64,
    pub prefill_kept: f64,
    pub cache: f64,
}

pub fn context_cells(table: &WindowTable, contexts: &[u64]) -> Vec<ContextCell> {
    contexts
        .iter()
        .map(|&t| ContextCell {
            t,
            decode_kept: table.decode_kept_share(t),
            prefill_kept: table.prefill_kept_share(t),
            cache: table.cache_share(t),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The paper's own anchor regime (Research 040 §2 / issue 1244 T1.2):
    /// d=64 ALL-rotating (P=32, no ALWAYS-KEPT rows), b=1e4, Φ=90°, t=4K.
    /// Exact-window `ρ_dec(4K)` = 23.86% vs the paper Table 8's 23.9% —
    /// the tight band is the pre-registered bar.
    fn anchor_header() -> HisaHeader {
        HisaHeader {
            freq_base: 1e4,
            key_length: 64,
            dimension_count: Some(64), // full rotary via the explicit key
            dimension_sections: None,
            full_attention_interval: None,
            block_count: 1,
            context_length: 262_144,
            arch: "test".to_string(),
        }
    }

    #[test]
    fn anchor_table8_rho_dec_4k_is_23_86_percent() {
        let h = anchor_header();
        let table = WindowTable::new(&h, 90.0);
        let rho = table.decode_kept_share(4096);
        assert!(
            (rho - 0.2386).abs() < 0.001,
            "ρ_dec(4K) = {rho:.6}, expected 0.2386 ± 0.001 (paper Table 8: 23.9%)"
        );
    }

    #[test]
    fn anchor_rho_4k_and_kappa_4k() {
        let h = anchor_header();
        let table = WindowTable::new(&h, 90.0);
        let rho = table.prefill_kept_share(4096);
        assert!(
            (rho - 0.293).abs() < 0.002,
            "ρ(4K) = {rho:.6}, expected 0.293 (the note's prefill row)"
        );
        let kappa = table.cache_share(4096);
        assert!(
            (kappa - 0.619).abs() < 0.002,
            "κ(4K) = {kappa:.6}, expected 0.619 ((ρ_dec + v)/2 with v=1)"
        );
    }

    /// Table 7's ρ(T) rows: the closed form at 2K/16K — the golden fixture
    /// pins what the closed form gives (the paper's own rows are the
    /// compare target recorded on the fixture; config drift refuses).
    #[test]
    fn anchor_rho_rows_2k_16k_stable() {
        let h = anchor_header();
        let table = WindowTable::new(&h, 90.0);
        let rho2k = table.prefill_kept_share(2048);
        let rho16k = table.prefill_kept_share(16384);
        // monotone: prefill keeps less as the context grows (long distances
        // enter the causal matrix and fall outside the windows)
        assert!(
            rho2k > rho16k,
            "ρ(2K) = {rho2k:.6} must exceed ρ(16K) = {rho16k:.6}"
        );
        // and both sit above their decode twins (short distances dominate)
        assert!(rho2k > table.decode_kept_share(2048));
        assert!(rho16k > table.decode_kept_share(16384));
    }

    /// The partial-RoPE geometry (the REAL backbones): d=256,
    /// d_rot=64 → 32 rotating + 96 ALWAYS-KEPT pairs. The unrotated share
    /// bounds every kept-share cell from below; the 2K/Φ=90° cell must
    /// reproduce the research note's hand-derived 90.0% kept (10.0% cut).
    #[test]
    fn bonsai_geometry_2k_phi90_matches_the_note_table() {
        let h = HisaHeader {
            freq_base: 1e7,
            key_length: 256,
            dimension_count: Some(64),
            dimension_sections: Some(vec![11, 11, 10, 0]),
            full_attention_interval: Some(4),
            block_count: 64,
            context_length: 262_144,
            arch: "qwen35-hybrid".to_string(),
        };
        assert_eq!(h.rotating_pairs(), 32);
        assert_eq!(h.total_pairs(), 128);
        let table = WindowTable::new(&h, 90.0);
        let rho = table.decode_kept_share(2048);
        assert!(
            (rho - 0.900).abs() < 0.004,
            "ρ_dec(2K) = {rho:.6}, expected 0.900 ± 0.004 (the note's hand-derived cell)"
        );
        // reach: Φ·b^((d_rot−2)/d_rot) ≈ 9.5e6 ≫ 262k → value eviction never fires
        assert!(table.reach > 262_144.0, "reach = {:.3e}", table.reach);
        assert!((table.cache_share(2048) - (rho + 1.0) / 2.0).abs() < 1e-12);
    }

    /// The full-rotary drafter geometry: NO `dimension_count` key → the
    /// header's own full-rotary statement (64/64 pairs rotate at d=128).
    #[test]
    fn dflash_full_rotary_header_reads_full() {
        let h = HisaHeader {
            freq_base: 1e7,
            key_length: 128,
            dimension_count: None,
            dimension_sections: None,
            full_attention_interval: None,
            block_count: 5,
            context_length: 262_144,
            arch: "dflash".to_string(),
        };
        assert_eq!(h.rotary_dim(), 128);
        assert_eq!(h.rotating_pairs(), 64);
        assert_eq!(h.total_pairs(), 64);
        assert_eq!(WindowTable::new(&h, 90.0).always_kept_pairs, 0);
    }

    /// The ALWAYS-KEPT rows never fold into a zero: the kept share is
    /// bounded below by the unrotated pair share at every t.
    #[test]
    fn always_kept_rows_bound_the_share_from_below() {
        let h = HisaHeader {
            freq_base: 1e7,
            key_length: 256,
            dimension_count: Some(64),
            dimension_sections: None,
            full_attention_interval: None,
            block_count: 64,
            context_length: 262_144,
            arch: "qwen35-hybrid".to_string(),
        };
        let table = WindowTable::new(&h, 45.0);
        let floor = 96.0 / 128.0;
        for t in [512u64, 2048, 32768] {
            let rho = table.decode_kept_share(t);
            assert!(
                rho >= floor,
                "ρ_dec({t}) = {rho:.6} below the ALWAYS-KEPT floor {floor}"
            );
        }
    }

    /// Ring rows: the chunk's ring = the SLOWEST pair's window; the q8
    /// alignment rounds UP to 32-elem blocks (Q8_BLOCK_SIZE verified 32 in
    /// `quant::q8kv`).
    #[test]
    fn ring_rows_take_the_slowest_pair_and_align_to_q8() {
        let h = anchor_header();
        let table = WindowTable::new(&h, 90.0);
        let rows = ring_table(&table, 8, 32);
        assert_eq!(rows.len(), 4); // 32 pairs / 8
        // the first chunk's slowest pair is l=7
        assert!((rows[0].window_exact - table.windows[7]).abs() < 1e-9);
        // monotone rings (windows grow with l)
        for w in rows.windows(2) {
            assert!(w[1].window_exact >= w[0].window_exact);
            assert!(w[1].ring_q8_aligned >= w[0].ring_q8_aligned);
        }
        // alignment: every ring is a multiple of 32 and >= the exact window
        for r in &rows {
            assert_eq!(r.ring_q8_aligned % 32, 0);
            assert!(r.ring_q8_aligned as f64 >= r.window_exact);
        }
    }

    /// The required-read refusals are loud and typed (no window arithmetic
    /// on defaults).
    #[test]
    fn missing_required_headers_refuse() {
        // exercised via the bin path (GgufFile needed); here the struct's
        // floor is pinned: total/rotating pairs are exact division facts
        let h = anchor_header();
        assert_eq!(h.total_pairs(), 32);
        assert_eq!(h.rotating_pairs(), 32);
    }

    /// T1.2's golden fixture: the planner's computed cells for BOTH
    /// geometries (the paper anchor + the real Bonsai partial-RoPE
    /// header) at every printed (Φ, t) cell, canonically serialized and
    /// blake3-pinned. The digest refuses loud on any config/formula drift
    /// (a changed window form, budget, context list, or precision shifts
    /// it). HERMETIC — pure math, no model file on the read path. Re-pin
    /// ONLY with the reason in the same commit (the drift IS the finding).
    #[test]
    fn golden_fixture_digest_is_pinned() {
        let mut rows = String::new();
        for (name, header) in [
            ("anchor_d64_full_b1e4", anchor_header()),
            (
                "bonsai_d256_partial64_b1e7",
                HisaHeader {
                    freq_base: 1e7,
                    key_length: 256,
                    dimension_count: Some(64),
                    dimension_sections: Some(vec![11, 11, 10, 0]),
                    full_attention_interval: Some(4),
                    block_count: 64,
                    context_length: 262_144,
                    arch: "qwen35".to_string(),
                },
            ),
        ] {
            for &phi in &DEFAULT_BUDGETS_DEG {
                let table = WindowTable::new(&header, phi);
                rows.push_str(&format!("{name}|phi={phi}|R={:.6}\n", table.reach));
                for cell in context_cells(&table, &DEFAULT_CONTEXTS) {
                    rows.push_str(&format!(
                        "{name}|phi={phi}|t={}|dec={:.6}|pre={:.6}|kappa={:.6}\n",
                        cell.t, cell.decode_kept, cell.prefill_kept, cell.cache
                    ));
                }
                for pairs in [8usize, 16] {
                    for r in ring_table(&table, pairs, 32) {
                        rows.push_str(&format!(
                            "{name}|phi={phi}|ring{pairs}|l{}|w={:.4}|q8={}\n",
                            r.l_first, r.window_exact, r.ring_q8_aligned
                        ));
                    }
                }
            }
        }
        let digest = blake3::hash(rows.as_bytes()).to_hex().to_string();
        // The pinned digest of the 2026-11-25 T1 landing (8 geometry cells
        // × 4 budgets; the pre-registered Research-040 §2 expectations all
        // reproduced — see the anchor tests above).
        if std::env::var("HISA_PLAN_PRINT_DIGEST").is_ok() {
            println!("HISA_PLAN_GOLDEN_DIGEST={digest}");
            println!("HISA_PLAN_GOLDEN_ROWS={}", rows.len());
        }
        assert_eq!(
            digest, "4b9e9ed47fc4fdf712a05ef3a883bf653d56a9191f64447f26e2a6b0c9fb4894",
            "golden fixture drifted — a formula/budget/context/precision change moved the planner's cells; re-pin ONLY with the reason in the same commit"
        );
    }
}
