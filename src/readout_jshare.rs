//! Issue-1250 T2 — the J-share verbalizability pass over the T1 readout
//! map.
//!
//! Definition pinned from katgpt-rs Research 693 §1.2: the J-share of a
//! unit vector w is the **fraction of w explained by a ≤25-token
//! nonnegative combination of lens directions**, percentile-ranked
//! against same-layer random directions. This module:
//!
//! 1. Reads the T1 `RDMAP1` artifact (each unit's top-`TOP_K` lens
//!    tokens are the atom support; T2 reads the artifact, never
//!    re-derives the scan) plus the SAME GGUF checkpoint (the unit
//!    vectors and the unembedding atom rows; the artifact's whole-file
//!    BLAKE3 must match or the pass refuses before any compute).
//! 2. Solves, per unit, the nonnegative least-squares fit of the unit
//!    onto its own atom rows (Lawson–Hanson active-set NNLS on the
//!    precomputed Gram, f64 sequential folds per the Issue-940 law,
//!    first-index tie rules) and reports `J = ‖Eα*‖² / ‖u‖²` — the
//!    squared-norm fraction in `[0, 1]`. Scale-invariant: the atoms
//!    enter through their span (α absorbs any per-atom scale); the unit
//!    posture is PINNED to the T1 lens unit (`lens_norm(col, γ, ε)`),
//!    the same vector whose lens logits ranked the atoms.
//! 3. Builds, per layer, a deterministic random-direction control
//!    population (Rademacher ±1 vectors from a SplitMix64 stream keyed
//!    on the file digest — transcendental-free, bit-reproducible) run
//!    through the SAME pipeline (lens norm → top-`TOP_K` scan → NNLS
//!    J), and buckets every unit High/Middle/Low by its percentile
//!    against that control distribution: High at p ≥ 0.95, Low at
//!    p < 0.50, Middle between (ties count as ≤ — the conservative
//!    direction for High). The bars are THIS lane's pin; the source's
//!    19–29%-high census is provenance, never a bar.
//! 4. Emits the canonical `RDMAP2` artifact (v2 rows carry the bucket +
//!    the J-share bits beside the token support; the trailer carries
//!    the control count and each layer's ascending control
//!    distribution for audit — the control seed base is the header's
//!    whole-file BLAKE3 field) + the per-band rollup (the T1 shape
//!    extended with the bucket census and J-share median/MAD).
//!
//! Posture: opt-in under `readout_screen`, offline by construction —
//! weights only, no forward pass, no serving path, zero new deps.
//! Deterministic: same inputs → bit-identical artifact + report (the
//! parallelism partitions whole units per rayon task; every unit's
//! NNLS is computed by exactly one task; every fold is sequential f64).

use crate::readout_screen::{
    Band, Bucket, HEADER_BYTES, ROW_BYTES, ReadoutTensors, Sign, TOP_K, band_of, lens_norm,
    ordered_dot_f64, parse_header,
};
use anyhow::{Result, bail};
use rayon::prelude::*;
use std::fmt::Write as _;

/// The v2 artifact magic (v1 stays readable; the pass refuses a v1
/// artifact only by digest, never by version — it CONSUMES v1).
pub const J_MAGIC: &[u8; 7] = b"RDMAP2\x01";
/// v2 artifact format version.
pub const J_VERSION: u32 = 2;
/// v2 packed row: `[layer u8][col u32][sign u8][bucket u8][jshare bits
/// u32]` — 11 bytes — plus `TOP_K × [token u32][bits u32]` (the v1
/// bits slot kept zero — the support rides the tokens, never
/// re-derived).
pub const J_ROW_BYTES: usize = 11 + 8 * TOP_K;
/// Random-direction controls per layer (this lane's pin — disclosed in
/// every record; the percentile bars read this population).
pub const CONTROLS_PER_LAYER: usize = 64;
/// The High bucket bar: a unit is High when ≥ 95% of its layer's
/// random controls score at or below its J-share.
pub const HIGH_PERCENTILE: f64 = 0.95;
/// The Low bucket bar: below the control median a unit is
/// indistinguishable from a random direction.
pub const LOW_PERCENTILE: f64 = 0.50;
/// The artifact header's file-digest offset (`HEADER_BYTES - 32`): the
/// v1/v2 header shape puts the whole-file BLAKE3 last.
pub const FILE_DIGEST_OFFSET: usize = HEADER_BYTES - 32;
/// Trailer: `[n_controls u32][n_layers u32]` + per layer
/// `[CONTROLS_PER_LAYER × f32 bits]` (ascending control J-shares). The
/// control seed base is NOT duplicated here — it IS the header's
/// whole-file BLAKE3 field (one copy, one truth; the control streams
/// key on the same provenance bytes they audit).
pub const J_TRAILER_FIXED: usize = 4 + 4;

// ── Deterministic NNLS (Lawson–Hanson on the precomputed Gram) ───────

/// Solve `min ‖Gα − c‖² s.t. α ≥ 0` from the precomputed `G = EᵀE`
/// (k×k row-major symmetric Gram) and `c = Eᵀu`. Lawson–Hanson active
/// set with first-index tie rules everywhere: the entering index is the
/// FIRST strictly-maximal positive gradient (ascending scan), the
/// pivot is the first maximal-magnitude row, and the inner
/// feasible-point step takes the FIRST minimal crossing ratio, evicting
/// every atom achieving it (ascending). All arithmetic f64 sequential.
pub fn nnls_gram(g: &[f64], c: &[f64], k: usize) -> Vec<f64> {
    let mut alpha = vec![0.0f64; k];
    let mut passive: Vec<usize> = Vec::with_capacity(k); // ascending
    // Never-select set: (a) a zero-diagonal atom (zero Gram column)
    // cannot produce a nonzero LS coefficient — admitting it makes the
    // passive solve singular, the eviction path returns it to ACTIVE,
    // and the outer loop re-enters it forever; (b) an atom evicted on
    // a ZERO feasible-point step (λ = 0) has a positive residual
    // gradient only through suppression that vanishes when it enters —
    // its unconstrained coefficient is negative, so the NNLS optimum
    // genuinely excludes it; re-admitting it would cycle.
    let mut banned = vec![false; k];
    for i in 0..k {
        banned[i] |= g[i * k + i].abs() < 1e-300;
    }
    let mut active: Vec<usize> = (0..k).filter(|&i| !banned[i]).collect();
    loop {
        // Gradient w = c − Gα; entering = first argmax over ACTIVE
        // (strictly greater keeps the first index on ties).
        let mut enter: Option<usize> = None;
        let mut best = 0.0f64;
        for &i in &active {
            if banned[i] {
                continue;
            }
            let mut wi = c[i];
            for &j in &passive {
                wi -= g[i * k + j] * alpha[j];
            }
            if wi > best {
                best = wi;
                enter = Some(i);
            }
        }
        let Some(e) = enter else {
            break;
        };
        let pos = passive.partition_point(|&j| j < e);
        passive.insert(pos, e);
        active.retain(|&j| j != e);
        // Inner loop: solve the passive LS; while any passive α ≤ 0,
        // step to the first zero-crossing and evict its atoms.
        loop {
            let alpha_prev = alpha.clone(); // the current feasible point
            solve_sub_ls(g, c, k, &passive, &mut alpha);
            if passive.iter().all(|&j| alpha[j] > 0.0) {
                break;
            }
            // Lawson–Hanson feasible-point step from α_prev toward z:
            // t = min over offending atoms of α_i/(α_i − z_i); the step
            // lands every atom achieving the min exactly at 0.
            let z = std::mem::take(&mut alpha);
            alpha = alpha_prev;
            let mut t = f64::INFINITY;
            for &j in &passive {
                if z[j] <= 0.0 {
                    let denom = alpha[j] - z[j];
                    if denom > 1e-300 {
                        t = t.min(alpha[j] / denom);
                    }
                }
            }
            if t.is_finite() && t > 0.0 {
                for &j in &passive {
                    alpha[j] += t * (z[j] - alpha[j]);
                }
            }
            // Evict the offending atoms. A ZERO step (t == 0, or no
            // viable t at all) bans them permanently: their residual
            // gradient is positive only through suppression that
            // vanishes on entry — readmitting would cycle.
            let zero_step = !t.is_finite() || t <= 0.0;
            let evict: Vec<usize> = passive
                .iter()
                .copied()
                .filter(|&j| z[j] <= 0.0 && (zero_step || alpha[j].abs() < 1e-9))
                .collect();
            let evict = if evict.is_empty() {
                passive
                    .iter()
                    .copied()
                    .filter(|&j| z[j] <= 0.0)
                    .collect::<Vec<_>>()
            } else {
                evict
            };
            for j in evict {
                alpha[j] = 0.0;
                passive.retain(|&x| x != j);
                active.push(j);
                if zero_step {
                    banned[j] = true;
                }
            }
            active.sort_unstable();
        }
    }
    alpha
}

/// Solve `G_PP x = c_P` for the passive set (ascending), zeroing α
/// outside it. Gaussian elimination with partial pivoting; the pivot
/// is the first maximal-magnitude row (first-index ties). Pure f64
/// sequential. On a singular passive Gram the solve leaves α
/// untouched (the caller's eviction path guarantees progress).
fn solve_sub_ls(g: &[f64], c: &[f64], k: usize, passive: &[usize], alpha: &mut [f64]) {
    for a in alpha.iter_mut() {
        *a = 0.0;
    }
    let m = passive.len();
    if m == 0 {
        return;
    }
    let mut a = vec![0.0f64; m * m];
    let mut b = vec![0.0f64; m];
    // Tikhonov damping (δ = 1e-12 × the max diagonal): the atom sets
    // this lane feeds are frequently rank-deficient (TOP_K atoms in
    // n_embd dims can exceed the rank — the fixture's 8 dims vs 10
    // atoms is degenerate by construction), and an undamped singular
    // passive solve bails to all-zero α, feeding the eviction churn.
    // The damping is deterministic and far below the fold precision.
    let mut max_diag = 0.0f64;
    for (r, &i) in passive.iter().enumerate() {
        for (cc, &j) in passive.iter().enumerate() {
            a[r * m + cc] = g[i * k + j];
        }
        b[r] = c[i];
        max_diag = max_diag.max(g[i * k + i]);
    }
    let delta = 1e-12 * max_diag;
    for r in 0..m {
        a[r * m + r] += delta;
    }
    for col in 0..m {
        let mut piv = col;
        for r in (col + 1)..m {
            if a[r * m + col].abs() > a[piv * m + col].abs() {
                piv = r;
            }
        }
        if a[piv * m + col].abs() < 1e-300 {
            return; // singular — leave α zero (the caller re-evicts)
        }
        if piv != col {
            for cc in col..m {
                a.swap(col * m + cc, piv * m + cc);
            }
            b.swap(col, piv);
        }
        let d = a[col * m + col];
        for r in (col + 1)..m {
            let f = a[r * m + col] / d;
            if f == 0.0 {
                continue;
            }
            for cc in col..m {
                a[r * m + cc] -= f * a[col * m + cc];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = vec![0.0f64; m];
    for r in (0..m).rev() {
        let mut s = b[r];
        for cc in (r + 1)..m {
            s -= a[r * m + cc] * x[cc];
        }
        x[r] = s / a[r * m + r];
    }
    for (r, &j) in passive.iter().enumerate() {
        alpha[j] = x[r];
    }
}

// ── The J-share over one unit ─────────────────────────────────────────

/// The J-share of unit `u` against the raw unembedding rows of
/// `tokens` (the unit's own T1 atom support). Returns `None` when `u`
/// is degenerate (zero/non-finite norm). `unemb` is the vocab×n_embd
/// row-major table; all folds sequential f64; `J = αᵀGα / ‖u‖²` with α
/// the NNLS solution, clamped to `[0, 1]`.
pub fn j_share(
    u: &[f32],
    unemb: &[f32],
    tokens: &[u32],
    n_embd: usize,
    vocab: usize,
) -> Option<f64> {
    let mut ussq = 0.0f64;
    for &x in u {
        ussq += x as f64 * x as f64;
    }
    if !ussq.is_finite() || ussq <= 0.0 {
        return None;
    }
    let atoms: Vec<&[f32]> = tokens
        .iter()
        .filter(|&&t| (t as usize) < vocab)
        .map(|&t| &unemb[t as usize * n_embd..(t as usize + 1) * n_embd])
        .collect();
    let kk = atoms.len();
    if kk == 0 {
        return Some(0.0);
    }
    let mut g = vec![0.0f64; kk * kk];
    let mut c = vec![0.0f64; kk];
    for (a, ea) in atoms.iter().enumerate() {
        c[a] = ordered_dot_f64(ea, u);
        for b in a..kk {
            let v = ordered_dot_f64(ea, atoms[b]);
            g[a * kk + b] = v;
            g[b * kk + a] = v;
        }
    }
    let alpha = nnls_gram(&g, &c, kk);
    // J = αᵀGα / ‖u‖² — sequential fold in α-ascending order.
    let mut num = 0.0f64;
    for a in 0..kk {
        for b in 0..kk {
            num += alpha[a] * g[a * kk + b] * alpha[b];
        }
    }
    Some((num / ussq).clamp(0.0, 1.0))
}

// ── Deterministic random controls ────────────────────────────────────

/// SplitMix64 (the house stateless PRNG).
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// A ±1 Rademacher direction from a SplitMix64 stream
/// (transcendental-free → bit-reproducible across boxes).
pub fn rademacher_direction(seed: u64, n_embd: usize) -> Vec<f32> {
    let mut st = seed;
    (0..n_embd)
        .map(|_| {
            if splitmix64(&mut st) & 1 == 0 {
                1.0f32
            } else {
                -1.0f32
            }
        })
        .collect()
}

/// The control seed for `(layer, k)` — folds the file digest so two
/// checkpoints never share a control stream.
fn control_seed(file_digest: &[u8; 32], layer: usize, k: usize) -> u64 {
    let mut acc = 0x1250_C0DEu64;
    for (i, &b) in file_digest.iter().enumerate() {
        acc = acc.rotate_left(1) ^ (b as u64).wrapping_mul(i as u64 | 1);
    }
    acc ^ ((layer as u64) << 32) ^ (k as u64).wrapping_mul(0x9E3779B97F4A7C15)
}

/// Top-`cap` tokens by lens logit (value desc, token-asc ties) — the
/// T1 comparator as a free function for the control pipeline (the
/// fixture cross-checks it against brute force).
pub fn top_k_tokens(
    normed: &[f32],
    unemb: &[f32],
    vocab: usize,
    n_embd: usize,
    cap: usize,
) -> (Vec<u32>, Vec<f32>) {
    let mut tokens: Vec<u32> = Vec::with_capacity(cap);
    let mut vals: Vec<f32> = Vec::with_capacity(cap);
    for t in 0..vocab {
        let row = &unemb[t * n_embd..(t + 1) * n_embd];
        let val = ordered_dot_f64(row, normed) as f32;
        let worse = vals.len() == cap
            && (val < vals[cap - 1] || (val == vals[cap - 1] && t as u32 > tokens[cap - 1]));
        if worse {
            continue;
        }
        let pos = vals
            .binary_search_by(|&v| val.partial_cmp(&v).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or_else(|p| p);
        let mut pos = pos;
        while pos < vals.len() && vals[pos] == val {
            pos += 1;
        }
        if vals.len() < cap {
            vals.insert(pos, val);
            tokens.insert(pos, t as u32);
        } else if pos < cap {
            vals.insert(pos, val);
            tokens.insert(pos, t as u32);
            vals.pop();
            tokens.pop();
        }
    }
    (tokens, vals)
}

// ── The pass ──────────────────────────────────────────────────────────

/// One T2 unit row (the v1 support carried forward + the J outcome).
#[derive(Debug, Clone)]
pub struct JRow {
    pub layer: usize,
    pub col: usize,
    pub sign: Sign,
    pub bucket: Bucket,
    pub tokens: Vec<u32>,
    /// The J-share (f32 bits stored canonically).
    pub jshare: f32,
}

/// Per-band rollup (the T1 shape extended with the bucket census).
#[derive(Debug, Clone)]
pub struct JBandRollup {
    pub band: Band,
    pub n_units: usize,
    pub high: usize,
    pub middle: usize,
    pub low: usize,
    /// J-share median + MAD (the pinned lower-middle order statistic —
    /// never an interpolated percentile).
    pub j_median: f32,
    pub j_mad: f32,
}

/// One layer's T2 result.
#[derive(Debug, Clone)]
pub struct JLayer {
    pub layer: usize,
    pub band: Band,
    pub units: Vec<JRow>,
    /// The layer's control J-shares, ascending.
    pub controls: Vec<f32>,
}

/// The full T2 result.
#[derive(Debug, Clone)]
pub struct JShareScreen {
    pub n_layers: usize,
    pub n_ff: usize,
    pub n_embd: usize,
    pub vocab: usize,
    pub controls_per_layer: usize,
    pub layers: Vec<JLayer>,
    pub rollups: Vec<JBandRollup>,
    /// The file digest the control streams keyed on (the trailer's
    /// seed base).
    pub seed_base: [u8; 32],
}

/// Parse a v1 artifact into `(layer, col, sign, tokens)` rows (the
/// header is validated first; the bits slot is the lens logits,
/// unused by T2 — the tokens are the atom support).
/// One v1 support row: `(layer, col, sign, tokens)`.
pub type V1SupportRow = (usize, usize, Sign, Vec<u32>);

pub fn parse_v1_rows(bytes: &[u8]) -> Result<Vec<V1SupportRow>> {
    let (_, _, _, _, n_units, _) = parse_header(bytes)?;
    let body = &bytes[HEADER_BYTES..];
    if body.len() != n_units * ROW_BYTES {
        bail!(
            "v1 body {} bytes != {} units × {ROW_BYTES}",
            body.len(),
            n_units
        );
    }
    let mut rows = Vec::with_capacity(n_units);
    for chunk in body.as_chunks::<ROW_BYTES>().0 {
        let layer = chunk[0] as usize;
        let col = u32::from_le_bytes([chunk[1], chunk[2], chunk[3], chunk[4]]) as usize;
        let sign = match chunk[5] {
            0 => Sign::Plus,
            1 => Sign::Minus,
            other => bail!("v1 row sign byte {other}"),
        };
        let mut tokens = Vec::with_capacity(TOP_K);
        for i in 0..TOP_K {
            let off = 9 + i * 8;
            let tok =
                u32::from_le_bytes([chunk[off], chunk[off + 1], chunk[off + 2], chunk[off + 3]]);
            if tok != u32::MAX {
                tokens.push(tok);
            }
        }
        rows.push((layer, col, sign, tokens));
    }
    Ok(rows)
}

/// Run the J-share pass. `artifact` is the v1 bytes (the atom
/// supports); `src` the open readout source over the SAME checkpoint —
/// the artifact's whole-file BLAKE3 must match or the pass refuses
/// before any compute.
pub fn run_jshare(artifact: &[u8], src: &dyn ReadoutTensors) -> Result<JShareScreen> {
    let (_, _, n_embd_hdr, n_ff_hdr, n_units, _) = parse_header(artifact)?;
    let header_fd: [u8; 32] = artifact[FILE_DIGEST_OFFSET..HEADER_BYTES]
        .try_into()
        .expect("32-byte slice");
    let live_fd = src
        .file_digest()
        .ok_or_else(|| anyhow::anyhow!("no file digest available for the source"))?;
    if header_fd != live_fd {
        bail!(
            "artifact file-digest mismatch: the RDMAP1 was built from a different \
             checkpoint — refusing the J-share pass"
        );
    }
    let n_layers = src.n_layers();
    let n_ff = src.n_ff();
    let d = src.n_embd();
    let v = src.vocab();
    if n_embd_hdr != d || n_ff_hdr != n_ff {
        bail!("artifact shape ({n_embd_hdr},{n_ff_hdr}) disagrees with the source ({d},{n_ff})");
    }
    let rows = parse_v1_rows(artifact)?;
    if rows.len() != n_units {
        bail!("v1 body rows {} != header units {n_units}", rows.len());
    }
    let gamma = src.gamma().to_vec();
    let eps = src.rms_eps();
    let unemb = src.unemb_rows()?;
    if unemb.len() != v * d {
        bail!(
            "unembedding length {} != vocab {v} × n_embd {d}",
            unemb.len()
        );
    }
    for (i, &x) in unemb.iter().enumerate() {
        if !x.is_finite() {
            bail!("non-finite unembedding weight at [{i}] — refusing the J-share pass");
        }
    }
    let seed_base = live_fd;

    // Group the v1 rows by layer (they arrive layer-ascending; a short
    // or overfull layer refuses — a partial sweep would skew the
    // percentile bars).
    let mut by_layer: Vec<Vec<V1SupportRow>> = vec![Vec::new(); n_layers];
    for r in rows {
        let (layer, col, sign, tokens) = r;
        if layer >= n_layers {
            bail!("artifact row layer {layer} >= n_layers {n_layers}");
        }
        by_layer[layer].push((layer, col, sign, tokens));
    }
    for (layer, lr) in by_layer.iter().enumerate() {
        if lr.len() != n_ff * 2 {
            bail!(
                "layer {layer} carries {} artifact rows, expected {} (n_ff × 2 signs)",
                lr.len(),
                n_ff * 2
            );
        }
    }

    let mut layers: Vec<JLayer> = Vec::with_capacity(n_layers);
    for (layer, layer_rows) in by_layer.iter().enumerate() {
        // Materialize this layer's write columns exactly as T1 does
        // (col[n][r] = down_out_row(r)[n]).
        let mut cols = vec![0.0f32; n_ff * d];
        for r in 0..d {
            let row = src.down_out_row(layer, r)?;
            if row.len() != n_ff {
                bail!(
                    "layer {layer} ffn_down row {r} length {} != n_ff {n_ff}",
                    row.len()
                );
            }
            for (n, &val) in row.iter().enumerate() {
                if !val.is_finite() {
                    bail!("non-finite ffn_down weight at layer {layer} row {r} col {n}");
                }
                cols[n * d + r] = val;
            }
        }
        // Controls first (they feed this layer's percentile bars).
        let mut controls: Vec<f32> = Vec::with_capacity(CONTROLS_PER_LAYER);
        for k in 0..CONTROLS_PER_LAYER {
            let r = rademacher_direction(control_seed(&seed_base, layer, k), d);
            let normed = lens_norm(&r, &gamma, eps);
            let (toks, _) = top_k_tokens(&normed, &unemb, v, d, TOP_K);
            let j = j_share(&normed, &unemb, &toks, d, v).unwrap_or(0.0);
            controls.push(j as f32);
        }
        controls.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let specs: &[V1SupportRow] = layer_rows;
        let units: Vec<JRow> = specs
            .par_iter()
            .map(|(layer, col, sign, tokens)| {
                // The unit vector at this sign: the raw column for
                // Plus, its negation for Minus (the T1 sign trick's
                // convention — the minus unit IS −w).
                let raw = &cols[*col * d..(*col + 1) * d];
                let normed = if *sign == Sign::Plus {
                    lens_norm(raw, &gamma, eps)
                } else {
                    let mut neg = raw.to_vec();
                    for x in &mut neg {
                        *x = -*x;
                    }
                    lens_norm(&neg, &gamma, eps)
                };
                let j = j_share(&normed, &unemb, tokens, d, v).unwrap_or(0.0) as f32;
                let bucket = bucket_of(percentile(&controls, j));
                JRow {
                    layer: *layer,
                    col: *col,
                    sign: *sign,
                    bucket,
                    tokens: tokens.clone(),
                    jshare: j,
                }
            })
            .collect();
        // Canonical order: col asc, Plus before Minus (the v1 order).
        let mut units = units;
        units.sort_by_key(|u| (u.col, u.sign as u8));
        let band = band_of(layer, n_layers);
        layers.push(JLayer {
            layer,
            band,
            units,
            controls,
        });
    }

    // Per-band rollups.
    let mut rollups: Vec<JBandRollup> = Vec::with_capacity(3);
    for band in [Band::Early, Band::Middle, Band::Late] {
        let js: Vec<f32> = layers
            .iter()
            .filter(|l| l.band == band)
            .flat_map(|l| l.units.iter().map(|u| u.jshare))
            .collect();
        if js.is_empty() {
            continue;
        }
        let mut s = js.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let med = s[(s.len() - 1) / 2];
        let mut dev: Vec<f32> = js.iter().map(|&x| (x - med).abs()).collect();
        dev.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mad = dev[(dev.len() - 1) / 2];
        let (mut high, mut middle, mut low) = (0usize, 0usize, 0usize);
        for l in layers.iter().filter(|l| l.band == band) {
            for u in &l.units {
                match u.bucket {
                    Bucket::High => high += 1,
                    Bucket::Middle => middle += 1,
                    Bucket::Low => low += 1,
                    Bucket::Unassigned => {}
                }
            }
        }
        rollups.push(JBandRollup {
            band,
            n_units: js.len(),
            high,
            middle,
            low,
            j_median: med,
            j_mad: mad,
        });
    }

    Ok(JShareScreen {
        n_layers,
        n_ff,
        n_embd: d,
        vocab: v,
        controls_per_layer: CONTROLS_PER_LAYER,
        layers,
        rollups,
        seed_base,
    })
}

/// Fraction of ascending `sorted` ≤ `v` (ties count as ≤ — the
/// conservative direction for High).
fn percentile(sorted: &[f32], v: f32) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted.partition_point(|&x| x <= v) as f64 / sorted.len() as f64
}

/// The pinned percentile → bucket classification (one site — the pass
/// and the fixture edge gate share it): High at p ≥ [`HIGH_PERCENTILE`],
/// Low at p < [`LOW_PERCENTILE`], Middle between.
pub fn bucket_of(p: f64) -> Bucket {
    if p >= HIGH_PERCENTILE {
        Bucket::High
    } else if p < LOW_PERCENTILE {
        Bucket::Low
    } else {
        Bucket::Middle
    }
}

// ── The v2 artifact ───────────────────────────────────────────────────

/// Serialize the v2 artifact: the v1-shaped header (new magic, v2) +
/// v2 rows + the control trailer.
pub fn build_j_artifact(s: &JShareScreen) -> Vec<u8> {
    let n_units: usize = s.layers.iter().map(|l| l.units.len()).sum();
    let mut out = Vec::with_capacity(
        HEADER_BYTES
            + n_units * J_ROW_BYTES
            + J_TRAILER_FIXED
            + s.layers.len() * s.controls_per_layer * 4,
    );
    out.extend_from_slice(J_MAGIC);
    out.extend_from_slice(&J_VERSION.to_le_bytes());
    out.extend_from_slice(&(s.vocab as u32).to_le_bytes());
    out.extend_from_slice(&(s.n_embd as u32).to_le_bytes());
    out.extend_from_slice(&(s.n_ff as u32).to_le_bytes());
    out.extend_from_slice(&(s.n_layers as u32).to_le_bytes());
    out.extend_from_slice(&(TOP_K as u32).to_le_bytes());
    // Shape parity with v1: the eps and unemb-source fields keep their
    // offsets (the pass reads the source live; neither re-enters the
    // artifact's arithmetic).
    out.extend_from_slice(&0.0f32.to_bits().to_le_bytes());
    out.push(0u8);
    out.extend_from_slice(&[0u8; 3]);
    out.extend_from_slice(&s.seed_base);
    for l in &s.layers {
        for u in &l.units {
            out.push(u.layer as u8);
            out.extend_from_slice(&(u.col as u32).to_le_bytes());
            out.push(u.sign as u8);
            out.push(u.bucket as u8);
            out.extend_from_slice(&u.jshare.to_bits().to_le_bytes());
            for i in 0..TOP_K {
                let tok = u.tokens.get(i).copied().unwrap_or(u32::MAX);
                out.extend_from_slice(&tok.to_le_bytes());
                out.extend_from_slice(&0u32.to_le_bytes());
            }
        }
    }
    out.extend_from_slice(&(s.controls_per_layer as u32).to_le_bytes());
    out.extend_from_slice(&(s.layers.len() as u32).to_le_bytes());
    for l in &s.layers {
        for &c in &l.controls {
            out.extend_from_slice(&c.to_bits().to_le_bytes());
        }
    }
    out
}

/// BLAKE3 digest of the v2 artifact bytes.
pub fn j_artifact_digest(bytes: &[u8]) -> [u8; 32] {
    *blake3::hash(bytes).as_bytes()
}

/// Render the deterministic T2 text report (byte-identical across
/// runs; `{:?}` floats are shortest round-trip).
pub fn render_j_report(s: &JShareScreen, artifact_hex: &str) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "== i1250 T2 J-share verbalizability (RDMAP2 v{}) ==",
        J_VERSION
    );
    let _ = writeln!(
        out,
        "units {} layers × {} cols × 2 signs · controls/layer {} · bars High p≥{:.2} / Low p<{:.2}",
        s.layers.len(),
        s.n_ff,
        s.controls_per_layer,
        HIGH_PERCENTILE,
        LOW_PERCENTILE
    );
    let _ = writeln!(out, "artifact blake3 = {}", artifact_hex);
    let _ = writeln!(out, "-- per-band rollups --");
    let _ = writeln!(
        out,
        "  band   units    high  middle     low   j_med   j_mad"
    );
    for r in &s.rollups {
        let _ = writeln!(
            out,
            "  {:>5?} {:>7} {:>7} {:>7} {:>7} {:>6.3} {:>6.3}",
            r.band, r.n_units, r.high, r.middle, r.low, r.j_median, r.j_mad
        );
    }
    let (mut hi, mut mid, mut lo) = (0usize, 0usize, 0usize);
    for l in &s.layers {
        for u in &l.units {
            match u.bucket {
                Bucket::High => hi += 1,
                Bucket::Middle => mid += 1,
                Bucket::Low => lo += 1,
                Bucket::Unassigned => {}
            }
        }
    }
    let total = hi + mid + lo;
    let _ = writeln!(out, "-- global bucket census --");
    let _ = writeln!(
        out,
        "  high {} ({:.1}%) · middle {} ({:.1}%) · low {} ({:.1}%) of {} units",
        hi,
        100.0 * hi as f64 / total.max(1) as f64,
        mid,
        100.0 * mid as f64 / total.max(1) as f64,
        lo,
        100.0 * lo as f64 / total.max(1) as f64,
        total
    );
    let _ = writeln!(out, "-- control envelope (per band of layers) --");
    for band in [Band::Early, Band::Middle, Band::Late] {
        let mut cs: Vec<f32> = Vec::new();
        for l in &s.layers {
            if l.band == band {
                cs.extend_from_slice(&l.controls);
            }
        }
        if cs.is_empty() {
            continue;
        }
        cs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let _ = writeln!(
            out,
            "  {:>5?}: controls min {:.3} · median {:.3} · max {:.3}",
            band,
            cs[0],
            cs[(cs.len() - 1) / 2],
            cs[cs.len() - 1]
        );
    }
    out
}
