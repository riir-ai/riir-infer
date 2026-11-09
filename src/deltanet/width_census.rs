//! The per-layer ε-width census (issue 1185 T2 — the Cactus "How Wide Is
//! Attention?" estimator, Research 657 §A2, `sink_audit`-gated).
//!
//! Reads an attention head as a runtime 2-layer MLP (W₁ = the cached keys,
//! softmax across T, W₂ = the values) and measures its **atomic effective
//! width** M̂_ε: the smallest number of REAL token columns whose span admits
//! a least-squares refit reproducing the head's output within relative
//! Frobenius ε. Greedy weighted block-pivoted-QR selection; the refit is
//! never formed — for a fixed token set S, the LS residual of Y against
//! inputs A_S equals, per output dimension j, the distance of Y[:, j] (an
//! n-vector over queries) to span(A_S's columns). That identity (pinned by
//! `closed_form_matches_the_explicit_ls_refit` against an explicitly solved
//! refit) replaces the solve with an orthonormal basis.
//!
//! # Spaces (read this before touching the indexing)
//!
//! The causal triangle means query `pos` only attends tokens `0..=pos`, so
//! activation column t is structurally zero for queries with `pos < t`.
//! Columns are stored as FULL query-length vectors with those zeros — a
//! rectangular space — so inner products across columns are well defined.
//! Two bases are maintained per head run:
//!
//! - **train basis** (over train queries): deflates column novelty for the
//!   greedy scoring. Selection never sees held-out queries.
//! - **held-out basis** (over held-out queries): Gram–Schmidt over the
//!   held-out components of the chosen columns; the closed-form residual
//!   projects the held-out Y **columns** (n_ho-vectors) onto it.
//!
//! # Pre-registered estimator choices (the 657 sketch is underdetermined)
//!
//! Every choice below is OURS, made at implementation time and pinned by
//! fixture — the paper's numbers are never quoted from this lane:
//!
//! - **Queries per KV head**: every Q head in the head's GQA group, at
//!   every position in the window; enumerated qh-major then pos ascending.
//! - **Held-out split**: the sketch's "every 4th" — enumeration index
//!   `i % held_out_every == 0` is held out.
//! - **Residual semantics**: held-out components of the chosen columns fit
//!   the held-out Y (an upper-bound reading of "refitted on held-out" —
//!   the source itself declares the estimator an upper bound; the stricter
//!   train-fit-evaluated-on-held-out form needs an explicit solve and is
//!   NOT what the sketch's closed form computes).
//! - **Scale**: the forward's own `1/√head_dim`.
//! - **Stop rule**: r ≤ ε after a block ⇒ width = |S| (block granularity —
//!   the paper's estimator is block-greedy). Cap or column exhaustion with
//!   `r > ε` ⇒ right-censored (`width = None`, `censored = true`).
//! - **Degenerate Y**: `‖Y_ho‖_F = 0` ⇒ width 0 (the empty set reproduces
//!   the zero output exactly); the entropy ratio is `None` there.
//! - **Entropy-matched control**: each real query replaced by a seeded
//!   random direction `β·u`, β found by 30-step geometric bisection so the
//!   control query's attention entropy (nats) matches the real query's;
//!   `R_ε^ent = width / control_width` separates dictionary-simplicity
//!   (≈ 1) from query-distribution specialization (< 1 here: the real
//!   distribution concentrates where isotropic queries cannot).
//! - **Determinism**: sequential f32 folds for logits/softmax, f64 for
//!   basis/residual arithmetic, one fixed order (slots ascending, KV heads
//!   ascending, queries in enumeration order, columns ascending with ties
//!   to the lower column index); control directions from blake3-seeded
//!   splitmix64 — no unseeded RNG anywhere.
//!
//! # Cost law (the 4090 run lane's budget)
//!
//! One head at a time; the transient is the column store plus the held-out
//! rows: ≈ `T·(n_train·8 + n_ho·4 + n_ho·d·8)` bytes, and the greedy
//! deflation is `atoms·T·n_train` f64 ops. A 16k window at the sketch's
//! n = 16,384 is GB-scale and CPU-hours — the RUN lane decides the window
//! and the host; this module is correct-first.
//!
//! # Scope
//!
//! Full-attention layers only (the slots of [`SinkAuditKVCapture`]), offline
//! only, behind `sink_audit`. It emits measurements — never a runtime
//! routing decision (head-skipping is NOT licensed by the source; the
//! needle-retention gate on any derived budget is mandatory and lives in
//! the run lane, issue 1185 T3).

use super::sink_audit::{SinkAuditKVCapture, SinkAuditQCapture};

/// The estimator's knobs (pre-registered defaults from the 657 sketch).
#[derive(Debug, Clone)]
pub struct WidthCensusConfig {
    /// Relative-Frobenius budget ε. The one hyperparameter.
    pub eps: f64,
    /// Atoms per greedy block (the sketch's block-pivoted step).
    pub block: usize,
    /// Right-censor cap on |S|.
    pub max_atoms: usize,
    /// Every k-th enumerated query is held out (`i % k == 0`).
    pub held_out_every: usize,
}

impl Default for WidthCensusConfig {
    fn default() -> Self {
        Self {
            eps: 0.2,
            block: 8,
            max_atoms: 6144,
            held_out_every: 4,
        }
    }
}

/// One KV head's census row.
#[derive(Debug, Clone)]
pub struct HeadWidthReport {
    pub layer_idx: usize,
    pub kvh: usize,
    /// `Some(w)` = the greedy stop landed on `r ≤ ε` at `w` atoms;
    /// `None` = right-censored (cap or column exhaustion with `r > ε`).
    /// 0 = the degenerate zero-output head (empty set suffices).
    pub width: Option<usize>,
    pub censored: bool,
    /// The relative-Frobenius residual at the stop point.
    pub residual: f64,
    /// The chosen token columns (ascending at block granularity — the
    /// selection order the run lane's needle gate replays).
    pub atoms: Vec<usize>,
    /// The entropy-matched isotropic control's width (same semantics).
    pub control_width: Option<usize>,
    pub control_censored: bool,
    pub control_residual: f64,
    /// `width / control_width` — `None` when either side is degenerate (0)
    /// or censored (a ratio of two censored stops is not a measurement).
    pub r_ent: Option<f64>,
}

/// The census result over every (full-attention layer, KV head).
#[derive(Debug, Clone)]
pub struct WidthCensus {
    pub per_head: Vec<HeadWidthReport>,
    pub config: WidthCensusConfig,
    /// The census window (== both captures' `n_positions`).
    pub n_positions: usize,
    /// Layers scanned (== `kv.attn_layers.len()`).
    pub layers_scanned: usize,
    /// KV heads per layer.
    pub n_kv_head: usize,
}

/// One percentile read with its TAIL SUPPORT disclosed (issue 1185 T3 arm 2;
/// the katgpt-rs `percentile_index_audit` class). The nearest-rank index is
/// `ceil(p·n) − 1` — never the `(n·p) as usize` truncation, whose "p99"
/// silently reads the maximum for every n ≤ 100. At n ≤ 1/(1−p) even the
/// correct index lands on (or beside) the maximum, so every read carries
/// `tail_support = n − index`: how many samples the read actually stands
/// on. A read with tail support < 10 is weak and must be quoted as weak —
/// disclosed, never folded into a bare percentile label.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PercentileRead {
    /// The requested percentile in (0, 1].
    pub p: f64,
    /// The sorted value at the read's index.
    pub value: usize,
    /// The index into the sorted slice the read landed on.
    pub index: usize,
    /// `n − index` — the number of samples at or above `value`.
    pub tail_support: usize,
}

/// Nearest-rank percentile over a SORTED (ascending) slice, tail support
/// attached. `None` on an empty slice or a p outside (0, 1] — refused,
/// never clamped into a plausible-looking read.
pub fn percentile_read(sorted: &[usize], p: f64) -> Option<PercentileRead> {
    if sorted.is_empty() || !(p > 0.0 && p <= 1.0) {
        return None;
    }
    let n = sorted.len();
    let rank = (p * n as f64).ceil(); // 1-based nearest rank, ≥ 1
    let index = (rank as usize).saturating_sub(1).min(n - 1);
    Some(PercentileRead {
        p,
        value: sorted[index],
        index,
        tail_support: n - index,
    })
}

/// The census's summary block (issue 1185 T3 arm 2): the per-head width
/// distribution as disclosed percentile reads. On the 64-head Bonsai
/// census, `p99` reads the maximum with `tail_support = 1` — that is the
/// hop the fixture pins, visible in the output instead of hidden behind a
/// confident label.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WidthSummary {
    /// Number of widths summarized (censored heads excluded by the caller).
    pub n: usize,
    pub min: usize,
    pub max: usize,
    pub p25: Option<PercentileRead>,
    pub p50: Option<PercentileRead>,
    pub p75: Option<PercentileRead>,
    pub p99: Option<PercentileRead>,
}

/// Summarize per-head widths (any order; sorted internally). All
/// percentiles go through [`percentile_read`] — one rank rule, one
/// disclosure shape.
pub fn width_summary(widths: &[usize]) -> WidthSummary {
    let mut sorted = widths.to_vec();
    sorted.sort_unstable();
    let min = sorted.first().copied().unwrap_or(0);
    let max = sorted.last().copied().unwrap_or(0);
    WidthSummary {
        n: widths.len(),
        min,
        max,
        p25: percentile_read(&sorted, 0.25),
        p50: percentile_read(&sorted, 0.50),
        p75: percentile_read(&sorted, 0.75),
        p99: percentile_read(&sorted, 0.99),
    }
}

/// Run the ε-width census over a capture pair (the `q` + `kv` of one
/// [`super::sink_audit::SinkCensus`] with `q: Some`).
///
/// Order: slots ascending (== `kv.attn_layers`), KV heads ascending,
/// queries in enumeration order, columns ascending. Errors loud on a
/// capture-pair mismatch — never silently scores a half-armed pair.
pub fn run_width_census(
    q: &SinkAuditQCapture,
    kv: &SinkAuditKVCapture,
    config: &WidthCensusConfig,
) -> Result<WidthCensus, String> {
    if q.n_positions != kv.n_positions {
        return Err(format!(
            "capture mismatch: Q window {} != KV window {}",
            q.n_positions, kv.n_positions
        ));
    }
    if q.attn_layers != kv.attn_layers {
        return Err("capture mismatch: Q and KV slot layer lists differ".into());
    }
    if q.q_rows.len() != kv.attn_layers.len() {
        return Err("capture mismatch: Q slot count != layer list".into());
    }
    if config.block == 0 || config.held_out_every == 0 {
        return Err("width census config: block and held_out_every must be positive".into());
    }
    let n = kv.n_positions;
    let head_dim = kv.head_dim;
    let kvd = kv.kv_dim;
    let n_kv = kv.n_kv_head;
    let q_dim = q.q_rows.first().map_or(0, |r| r.len() / n.max(1));
    if q_dim == 0 || !q_dim.is_multiple_of(head_dim) {
        return Err(format!(
            "Q capture width {q_dim} not a multiple of head_dim"
        ));
    }
    let n_head = q_dim / head_dim;
    if n_kv == 0 || !n_head.is_multiple_of(n_kv) {
        return Err(format!(
            "GQA geometry: {n_head} Q heads not divisible by {n_kv} KV heads"
        ));
    }
    let kv_per_head = n_head / n_kv;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut per_head = Vec::with_capacity(n_kv);
    for (slot, &layer_idx) in kv.attn_layers.iter().enumerate() {
        let q_rows = &q.q_rows[slot];
        let keys = &kv.key_rows[slot];
        let values = &kv.value_rows[slot];
        for kvh in 0..n_kv {
            let head = HeadSlice {
                keys,
                kv_head_off: kvh * head_dim,
                kvd,
                head_dim,
                scale,
            };
            let report = census_one_head(
                layer_idx,
                kvh,
                q_rows,
                values,
                &head,
                n,
                q_dim,
                head_dim,
                kv_per_head,
                config,
            );
            per_head.push(report);
        }
    }
    Ok(WidthCensus {
        per_head,
        config: config.clone(),
        n_positions: n,
        layers_scanned: kv.attn_layers.len(),
        n_kv_head: n_kv,
    })
}

/// One (layer, KV head) census. Sequential and fixed-order; the transient
/// lives and dies with this call.
#[allow(clippy::too_many_arguments)]
fn census_one_head(
    layer_idx: usize,
    kvh: usize,
    q_rows: &[f32],
    values: &[f32],
    head: &HeadSlice<'_>,
    n: usize,
    q_dim: usize,
    head_dim: usize,
    kv_per_head: usize,
    config: &WidthCensusConfig,
) -> HeadWidthReport {
    // The GQA group of this KV head: Q heads h with h / kv_per_head == kvh.
    let group: Vec<usize> = (kvh * kv_per_head..(kvh + 1) * kv_per_head).collect();

    // Per-token payload ‖v_t‖² (the weighting half of the pivoted score).
    let mut payloads = vec![0.0f64; n];
    for (t, payload) in payloads.iter_mut().enumerate() {
        let off = t * head.kvd + head.kv_head_off;
        let v_row = &values[off..off + head.head_dim];
        *payload = v_row.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    }

    let n_queries = group.len() * n;
    let mut real = HeadRun::new(n, n_queries, config.held_out_every);
    let mut ctrl = HeadRun::new(n, n_queries, config.held_out_every);
    let mut logits = vec![0.0f32; n];
    let mut a_row = vec![0.0f32; n];

    for (gi, (qh, pos)) in group
        .iter()
        .flat_map(|&h| (0..n).map(move |p| (h, p)))
        .enumerate()
    {
        let prefix = pos + 1;
        let q_off = pos * q_dim + qh * head_dim;
        let q_vec = &q_rows[q_off..q_off + head_dim];
        head.softmax_row(q_vec, prefix, &mut logits, &mut a_row);
        let entropy = entropy_of(&a_row[..prefix]);
        real.push_query(gi, prefix, &a_row);

        // The entropy-matched control query: a seeded random direction whose
        // β-tuned softmax over the SAME dictionary matches the real query's
        // attention entropy.
        let u = control_direction(layer_idx, kvh, gi, head_dim);
        let beta = bisect_entropy_beta(&u, head, prefix, entropy);
        let mut c_logits = vec![0.0f32; n];
        let mut c_row = vec![0.0f32; n];
        // The direction enters the logit fold scaled by β.
        let mut u_scaled = vec![0.0f32; head_dim];
        for (j, us) in u_scaled.iter_mut().enumerate() {
            *us = (u[j] * beta) as f32;
        }
        head.softmax_row(&u_scaled, prefix, &mut c_logits, &mut c_row);
        ctrl.push_query(gi, prefix, &c_row);
    }

    // Held-out Y rows (the head's true outputs) — computed only for held-out
    // queries; train Y is never read by the estimator.
    real.compute_held_out_y(values, head);
    ctrl.compute_held_out_y(values, head);

    let real_stop = real.run(config, &payloads);
    let ctrl_stop = ctrl.run(config, &payloads);
    let r_ent = match (real_stop.width, ctrl_stop.width) {
        (Some(w), Some(cw)) if w > 0 && cw > 0 => Some(w as f64 / cw as f64),
        _ => None,
    };
    HeadWidthReport {
        layer_idx,
        kvh,
        width: real_stop.width,
        censored: real_stop.censored,
        residual: real_stop.residual,
        atoms: real_stop.atoms,
        control_width: ctrl_stop.width,
        control_censored: ctrl_stop.censored,
        control_residual: ctrl_stop.residual,
        r_ent,
    }
}

/// One KV head's slice of the captured tensors — the parameter block every
/// per-head scorer shares (storage strides + the softmax scale).
struct HeadSlice<'a> {
    keys: &'a [f32],
    kv_head_off: usize,
    kvd: usize,
    head_dim: usize,
    scale: f32,
}

impl HeadSlice<'_> {
    /// Stable softmax over one query's prefix against this KV head's key
    /// rows: sequential f32 logit folds (determinism law), max-subtracted,
    /// normalized into `a_row[..prefix]` (entries past the prefix are
    /// untouched garbage — callers only read the prefix).
    fn softmax_row(&self, q_vec: &[f32], prefix: usize, logits: &mut [f32], a_row: &mut [f32]) {
        let mut lmax = f32::NEG_INFINITY;
        for (t, lt) in logits.iter_mut().enumerate().take(prefix) {
            let k_off = t * self.kvd + self.kv_head_off;
            let k_row = &self.keys[k_off..k_off + self.head_dim];
            let mut dot = 0.0f32;
            for (q, k) in q_vec.iter().zip(k_row.iter()) {
                dot += q * k;
            }
            let l = dot * self.scale;
            *lt = l;
            if l > lmax {
                lmax = l;
            }
        }
        let mut z = 0.0f64;
        for (lt, a) in logits.iter().zip(a_row.iter_mut()).take(prefix) {
            let e = (f64::from(*lt - lmax)).exp();
            *a = e as f32;
            z += e;
        }
        for a in a_row.iter_mut().take(prefix) {
            *a = (f64::from(*a) / z) as f32;
        }
    }

    /// The f64 attention entropy (nats) of a β-scaled direction over the
    /// prefix — the control arm's probe (deterministic, allocation-free
    /// beyond the caller-owned logit buffer).
    fn direction_entropy(&self, u: &[f64], beta: f64, prefix: usize, logits: &mut [f64]) -> f64 {
        let mut lmax = f64::NEG_INFINITY;
        for (t, lt) in logits.iter_mut().enumerate().take(prefix) {
            let k_off = t * self.kvd + self.kv_head_off;
            let k_row = &self.keys[k_off..k_off + self.head_dim];
            let mut dot = 0.0f64;
            for (uj, k) in u.iter().zip(k_row.iter()) {
                dot += uj * f64::from(*k);
            }
            let l = dot * f64::from(self.scale) * beta;
            *lt = l;
            if l > lmax {
                lmax = l;
            }
        }
        let mut z = 0.0f64;
        for lt in logits[..prefix].iter_mut() {
            *lt = (*lt - lmax).exp();
            z += *lt;
        }
        let mut h = 0.0f64;
        for lt in &logits[..prefix] {
            let a = lt / z;
            if a > 0.0 {
                h -= a * a.ln();
            }
        }
        h
    }
}

/// Shannon entropy (nats, f64) of a softmax row.
fn entropy_of(a_row: &[f32]) -> f64 {
    let mut h = 0.0f64;
    for &a in a_row {
        let a = f64::from(a);
        if a > 0.0 {
            h -= a * a.ln();
        }
    }
    h
}

/// blake3-seeded splitmix64 unit direction (determinism law: no unseeded
/// RNG; the seed binds layer/KV-head/query so reruns are bit-identical).
fn control_direction(layer_idx: usize, kvh: usize, query_idx: usize, head_dim: usize) -> Vec<f64> {
    let mut seed_material = Vec::with_capacity(8 + 24);
    seed_material.extend_from_slice(b"riir-infer width_census ctrl");
    seed_material.extend_from_slice(&(layer_idx as u64).to_le_bytes());
    seed_material.extend_from_slice(&(kvh as u64).to_le_bytes());
    seed_material.extend_from_slice(&(query_idx as u64).to_le_bytes());
    let digest = blake3::hash(&seed_material);
    let mut state = u64::from_le_bytes(digest.as_bytes()[0..8].try_into().expect("8 bytes"));
    let mut v = vec![0.0f64; head_dim];
    let mut sq = 0.0f64;
    for slot in v.iter_mut() {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // Uniform on (0, 1), mapped to a symmetric draw.
        let unit = ((z >> 11) as f64) / ((1u64 << 53) as f64);
        *slot = unit * 2.0 - 1.0;
        sq += *slot * *slot;
    }
    let norm = sq.sqrt();
    if norm > 0.0 {
        for slot in v.iter_mut() {
            *slot /= norm;
        }
    } else {
        v[0] = 1.0;
    }
    v
}

/// 30-step geometric bisection for the control sharpness β matching the
/// real query's attention entropy (nats). Entropy decreases monotonically
/// in β (sharper ⇒ less spread); [1e-6, 1e6] brackets every achievable
/// target in [0, ln(prefix)].
fn bisect_entropy_beta(u: &[f64], head: &HeadSlice<'_>, prefix: usize, target_entropy: f64) -> f64 {
    let mut logits = vec![0.0f64; prefix];
    let (mut lo, mut hi) = (1e-6f64, 1e6f64);
    for _ in 0..30 {
        let mid = (lo * hi).sqrt();
        if head.direction_entropy(u, mid, prefix, &mut logits) < target_entropy {
            // Too sharp — widen.
            hi = mid;
        } else {
            lo = mid;
        }
    }
    (lo * hi).sqrt()
}

/// The greedy stop verdict for one head run (real or control).
#[derive(Debug, Clone)]
struct HeadStop {
    width: Option<usize>,
    censored: bool,
    residual: f64,
    atoms: Vec<usize>,
}

/// One head's greedy machinery over its query set. Two spaces, both
/// zero-masked to rectangular (see the module doc): the TRAIN columns the
/// greedy scores and deflates, and the HELD-OUT rows whose components of
/// the chosen columns fit the held-out Y.
struct HeadRun {
    /// Total enumerated queries; the split key is `gi % held_out_every`.
    n_queries: usize,
    held_out_every: usize,
    /// Train-query activation columns: `train_cols[t][i]` = the activation
    /// of train query i on token t (0.0 where causally masked, and in the
    /// tail past `n_train`). Deflated in place by the train basis as atoms
    /// land (a chosen column is consumed into its basis vector).
    train_cols: Vec<Vec<f64>>,
    /// Held-out query activation rows: `ho_rows[i][t]` (0.0 past the
    /// prefix). Never enters selection.
    ho_rows: Vec<Vec<f32>>,
    /// Held-out enumeration indices (for the Y pull-through).
    ho_query_gis: Vec<usize>,
    /// Each enumerated query's prefix (both spaces need it).
    prefixes: Vec<usize>,
    /// Held-out Y rows, filled by [`Self::compute_held_out_y`].
    y_ho: Vec<Vec<f64>>,
    /// Train query count (the train space's dimension).
    n_train: usize,
    n_ho: usize,
    y_ho_norm_sq: f64,
}

impl HeadRun {
    /// `n_columns` = the window (one column per token); `n_queries_expected`
    /// sizes the column buffers up front (the tail past `n_train` stays 0.0
    /// and is invisible to every inner product).
    fn new(n_columns: usize, n_queries_expected: usize, held_out_every: usize) -> Self {
        Self {
            n_queries: 0,
            held_out_every,
            train_cols: (0..n_columns)
                .map(|_| vec![0.0f64; n_queries_expected])
                .collect(),
            ho_rows: Vec::new(),
            ho_query_gis: Vec::new(),
            prefixes: Vec::new(),
            y_ho: Vec::new(),
            n_train: 0,
            n_ho: 0,
            y_ho_norm_sq: 0.0,
        }
    }

    /// File one query's activation row (only the prefix is meaningful).
    fn push_query(&mut self, gi: usize, prefix: usize, a_row: &[f32]) {
        self.n_queries += 1;
        self.prefixes.push(prefix);
        if gi.is_multiple_of(self.held_out_every) {
            // Held-out: activations stored for the residual basis, never
            // for selection.
            let mut row = vec![0.0f32; self.train_cols.len()];
            row[..prefix].copy_from_slice(&a_row[..prefix]);
            self.ho_rows.push(row);
            self.ho_query_gis.push(gi);
            self.n_ho += 1;
        } else {
            let train_i = self.n_train;
            self.n_train += 1;
            for (t, &a) in a_row.iter().enumerate().take(prefix) {
                self.train_cols[t][train_i] = f64::from(a);
            }
        }
    }

    /// Pull the held-out Y rows through the captured V rows: Y row = the
    /// activation row dotted with this head's V rows. (The caller supplies
    /// the value rows; each held-out query's own softmax is the weights.)
    fn compute_held_out_y(&mut self, values: &[f32], head: &HeadSlice<'_>) {
        self.y_ho.clear();
        self.y_ho_norm_sq = 0.0;
        for (row_i, &gi) in self.ho_query_gis.iter().enumerate() {
            let prefix = self.prefixes[gi];
            let a = &self.ho_rows[row_i];
            let mut y = vec![0.0f64; head.head_dim];
            for (t, &w32) in a.iter().enumerate().take(prefix) {
                let w = f64::from(w32);
                if w == 0.0 {
                    continue;
                }
                let v_off = t * head.kvd + head.kv_head_off;
                for (j, yj) in y.iter_mut().enumerate() {
                    *yj += w * f64::from(values[v_off + j]);
                }
            }
            for x in &y {
                self.y_ho_norm_sq += x * x;
            }
            self.y_ho.push(y);
        }
    }

    fn run(&mut self, config: &WidthCensusConfig, payloads: &[f64]) -> HeadStop {
        let n_columns = self.train_cols.len();
        // Degenerate Y: the empty set reproduces the zero output exactly.
        if self.y_ho_norm_sq == 0.0 {
            return HeadStop {
                width: Some(0),
                censored: false,
                residual: 0.0,
                atoms: Vec::new(),
            };
        }
        let mut chosen_mask = vec![false; n_columns];
        let mut atoms: Vec<usize> = Vec::new();
        let mut train_basis: Vec<Vec<f64>> = Vec::new();
        let mut ho_basis: Vec<Vec<f64>> = Vec::new();
        // r over the empty basis is 1.0 (Y_ho nonzero, nothing covered).
        let mut residual = 1.0f64;
        loop {
            // Stop check FIRST: the first block with r ≤ ε is the width.
            if residual <= config.eps {
                return HeadStop {
                    width: Some(atoms.len()),
                    censored: false,
                    residual,
                    atoms,
                };
            }
            if atoms.len() >= config.max_atoms || atoms.len() >= n_columns {
                return HeadStop {
                    width: None,
                    censored: true,
                    residual,
                    atoms,
                };
            }
            // Score: novelty × payload over unchosen columns; ties → the
            // lower column index (stable sort on (score desc, t asc)).
            let mut scored: Vec<(usize, f64)> = Vec::with_capacity(n_columns - atoms.len());
            for (t, col) in self.train_cols.iter().enumerate() {
                if chosen_mask[t] {
                    continue;
                }
                let mut nov = 0.0f64;
                for &r in col {
                    nov += r * r;
                }
                scored.push((t, nov * payloads[t]));
            }
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let budget = config.block.min(config.max_atoms - atoms.len());
            let take = budget.min(scored.len());
            if take == 0 || scored[0].1 <= 0.0 {
                // No column carries novelty left while r > ε: the span
                // cannot grow — a censor, never a fake width.
                return HeadStop {
                    width: None,
                    censored: true,
                    residual,
                    atoms,
                };
            }
            for &(t, _) in scored.iter().take(take) {
                chosen_mask[t] = true;
                atoms.push(t);
                // ── train space: deflate the column into the basis ──
                let mut u = std::mem::take(&mut self.train_cols[t]);
                // Twice-for-stability Gram–Schmidt against the basis.
                for _pass in 0..2 {
                    for b in &train_basis {
                        let mut dot = 0.0f64;
                        for i in 0..u.len() {
                            dot += b[i] * u[i];
                        }
                        for i in 0..u.len() {
                            u[i] -= dot * b[i];
                        }
                    }
                }
                let mut norm = 0.0f64;
                for &x in &u {
                    norm += x * x;
                }
                let norm = norm.sqrt();
                if norm > 0.0 {
                    for x in u.iter_mut() {
                        *x /= norm;
                    }
                    for col in self.train_cols.iter_mut() {
                        let mut dot = 0.0f64;
                        for i in 0..col.len() {
                            dot += u[i] * col[i];
                        }
                        for (i, c) in col.iter_mut().enumerate() {
                            *c -= dot * u[i];
                        }
                    }
                    train_basis.push(u);
                }
                // ── held-out space: extend the residual basis ──
                let mut hv = vec![0.0f64; self.n_ho];
                for (i, row) in self.ho_rows.iter().enumerate() {
                    hv[i] = f64::from(row[t]);
                }
                for _pass in 0..2 {
                    for b in &ho_basis {
                        let mut dot = 0.0f64;
                        for i in 0..hv.len() {
                            dot += b[i] * hv[i];
                        }
                        for i in 0..hv.len() {
                            hv[i] -= dot * b[i];
                        }
                    }
                }
                let mut hnorm = 0.0f64;
                for &x in &hv {
                    hnorm += x * x;
                }
                let hnorm = hnorm.sqrt();
                if hnorm > 0.0 {
                    for x in hv.iter_mut() {
                        *x /= hnorm;
                    }
                    ho_basis.push(hv);
                }
            }
            // Re-measure the closed-form residual on held-out Y.
            residual = self.closed_form_residual(&ho_basis);
        }
    }

    /// r_m = √(‖Y_ho‖²_F − ‖UᵀY_ho‖²_F) / ‖Y_ho‖_F over the held-out basis:
    /// exact because each Y_ho column's LS residual against the chosen
    /// columns' held-out components is its distance to their span.
    fn closed_form_residual(&self, ho_basis: &[Vec<f64>]) -> f64 {
        if self.y_ho_norm_sq == 0.0 {
            return 0.0;
        }
        let mut covered = 0.0f64;
        for y_col in 0..self.y_ho[0].len() {
            // y_j = the held-out queries' j-th output component (n_ho-vector).
            let mut proj_sq = 0.0f64;
            for b in ho_basis {
                let mut dot = 0.0f64;
                for (i, bi) in b.iter().enumerate() {
                    dot += bi * self.y_ho[i][y_col];
                }
                proj_sq += dot * dot;
            }
            covered += proj_sq;
        }
        let r_sq = (self.y_ho_norm_sq - covered).max(0.0) / self.y_ho_norm_sq;
        r_sq.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deltanet::sink_audit::SinkAuditKVCapture;
    use crate::deltanet::sink_audit::SinkAuditQCapture;
    use crate::types::{Config, DeltaNetLayerType};

    /// One attention layer, 2 Q heads over 1 KV head, head_dim 4 — the
    /// smallest GQA shape the estimator's group logic can express.
    fn tiny_config() -> Config {
        Config {
            n_layer: 1,
            n_embd: 8,
            n_head: 2,
            n_kv_head: 1,
            head_dim: 4,
            mlp_hidden: 16,
            vocab_size: 32,
            block_size: 16,
            layer_types: vec![DeltaNetLayerType::Attention],
            ..Config::default()
        }
    }

    /// Deterministic pseudo-content: an LCG-shaped f32 in [-1, 1).
    fn pseudo(i: usize, salt: u64) -> f32 {
        let mut z = (i as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(salt);
        z ^= z >> 27;
        z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z ^= z >> 31;
        ((z % 2000) as f32 / 1000.0) - 1.0
    }

    /// A synthetic capture pair: every buffer filled with pseudo content
    /// (real forward semantics irrelevant — the estimator reads the captures
    /// as plain matrices).
    fn synthetic_captures(
        config: &Config,
        n: usize,
        salt: u64,
    ) -> (SinkAuditQCapture, SinkAuditKVCapture) {
        let layer_types = &[DeltaNetLayerType::Attention];
        let mut q = SinkAuditQCapture::new(config, layer_types, n);
        for slot in q.q_rows.iter_mut() {
            for (i, v) in slot.iter_mut().enumerate() {
                *v = pseudo(i, salt);
            }
        }
        let kvd = crate::types::kv_dim(config);
        let want = n * kvd;
        let key: Vec<f32> = (0..want).map(|i| pseudo(i, salt + 1)).collect();
        let value: Vec<f32> = (0..want).map(|i| pseudo(i, salt + 2)).collect();
        let kv = SinkAuditKVCapture::extract(
            &[0],
            |_slot| (key.as_slice(), value.as_slice()),
            n,
            kvd,
            config.n_kv_head,
            config.head_dim,
        );
        (q, kv)
    }

    /// THE method pin: the closed-form residual is the EXPLICIT least-squares
    /// refit residual. Recompute A_S,ho and Y_ho from first principles for a
    /// tiny head, solve min ‖A_S,ho·B − Y_ho‖_F by hand (Gram–Schmidt on the
    /// columns, per-output-dimension projections), and compare to the
    /// census's reported relative residual.
    #[test]
    fn closed_form_matches_the_explicit_ls_refit() {
        let config = tiny_config();
        let n = 8usize;
        let (q, kv) = synthetic_captures(&config, n, 11);
        let cfg = WidthCensusConfig {
            eps: 0.0, // never stops early — force the full greedy walk
            block: 2,
            max_atoms: n,
            held_out_every: 4,
        };
        let census = run_width_census(&q, &kv, &cfg).expect("census");
        // One layer, one KV head; with eps = 0 the walk runs until the
        // held-out basis reproduces Y_ho EXACTLY (r = 0 ≤ 0 — a clean stop,
        // never a fake width). The atom count is data-dependent: the held-out
        // space saturates at rank n_ho, so the walk stops at or below 8.
        let report = &census.per_head[0];
        assert!(!report.censored, "eps=0 stops clean, never censors");
        let width = report.width.expect("clean stop carries a width");
        assert_eq!(report.atoms.len(), width);
        assert!(width <= n);
        assert!(
            report.residual < 1e-9,
            "stopped span must reproduce Y exactly"
        );

        // ── First-principles reconstruction (independent of the module's
        // internals): enumerate the GQA group (2 Q heads × 8 positions),
        // held-out = gi % 4 == 0 → 4 held-out queries. ──
        let head_dim = config.head_dim;
        let kvd = crate::types::kv_dim(&config);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let keys = &kv.key_rows[0];
        let values = &kv.value_rows[0];
        let q_rows = &q.q_rows[0];
        let q_dim = config.n_head * head_dim;
        let mut ho_a: Vec<Vec<f64>> = Vec::new();
        let mut ho_y: Vec<Vec<f64>> = Vec::new();
        for gi in 0..(config.n_head * n) {
            if gi % 4 != 0 {
                continue;
            }
            let qh = gi / n;
            let pos = gi % n;
            let prefix = pos + 1;
            let q_off = pos * q_dim + qh * head_dim;
            let q_vec = &q_rows[q_off..q_off + head_dim];
            let mut logits = vec![0.0f64; prefix];
            let mut lmax = f64::NEG_INFINITY;
            for (t, lt) in logits.iter_mut().enumerate() {
                let k_off = t * kvd;
                let mut dot = 0.0f64;
                for j in 0..head_dim {
                    dot += f64::from(q_vec[j]) * f64::from(keys[k_off + j]);
                }
                let l = dot * f64::from(scale);
                *lt = l;
                if l > lmax {
                    lmax = l;
                }
            }
            let mut z = 0.0f64;
            let mut a = vec![0.0f64; n]; // zero-masked to full width
            for t in 0..prefix {
                let e = (logits[t] - lmax).exp();
                a[t] = e;
                z += e;
            }
            for x in a.iter_mut().take(prefix) {
                *x /= z;
            }
            let mut y = vec![0.0f64; head_dim];
            for (t, &w) in a.iter().enumerate() {
                if w == 0.0 {
                    continue;
                }
                let v_off = t * kvd;
                for (j, yj) in y.iter_mut().enumerate() {
                    *yj += w * f64::from(values[v_off + j]);
                }
            }
            ho_a.push(a);
            ho_y.push(y);
        }
        let n_ho = ho_a.len();
        assert_eq!(n_ho, 4);

        // Explicit LS: for each Y column j, min_b ‖A b − y_j‖ — Gram–Schmidt
        // the A rows (n_ho-dim activation columns of the chosen tokens;
        // every token is chosen here), project each Y column, accumulate.
        let mut cols: Vec<Vec<f64>> = Vec::new();
        for t in report.atoms.clone() {
            let col: Vec<f64> = ho_a.iter().map(|row| row[t]).collect();
            let mut v = col.clone();
            for b in &cols {
                let dot: f64 = b.iter().zip(&v).map(|(x, y)| x * y).sum();
                for (vi, &bi) in v.iter_mut().zip(b) {
                    *vi -= dot * bi;
                }
            }
            let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-12 {
                for x in v.iter_mut() {
                    *x /= norm;
                }
                cols.push(v);
            }
        }
        let mut total = 0.0f64;
        let mut ynorm = 0.0f64;
        for j in 0..head_dim {
            let yj: Vec<f64> = ho_y.iter().map(|y| y[j]).collect();
            let n2: f64 = yj.iter().map(|x| x * x).sum();
            ynorm += n2;
            let mut proj = 0.0f64;
            for b in &cols {
                let dot: f64 = b.iter().zip(&yj).map(|(x, y)| x * y).sum();
                proj += dot * dot;
            }
            total += n2 - proj;
        }
        let explicit_r = (total.max(0.0) / ynorm).sqrt();
        let close = (explicit_r - report.residual).abs() < 1e-9
            || (explicit_r < 1e-7 && report.residual < 1e-7);
        assert!(
            close,
            "closed form {} != explicit LS refit {}",
            report.residual, explicit_r
        );
    }

    /// Zero V rows ⇒ zero Y ⇒ the empty set reproduces the output exactly:
    /// width 0, residual 0, no censor, and the entropy ratio is undefined.
    #[test]
    fn zero_value_rows_degenerate_to_width_zero() {
        let config = tiny_config();
        let n = 8usize;
        let (q, kv_zero_v) = {
            let (q, mut kv) = synthetic_captures(&config, n, 5);
            for row in kv.value_rows.iter_mut() {
                row.fill(0.0);
            }
            (q, kv)
        };
        let census =
            run_width_census(&q, &kv_zero_v, &WidthCensusConfig::default()).expect("census");
        let report = &census.per_head[0];
        assert_eq!(report.width, Some(0));
        assert_eq!(report.residual, 0.0);
        assert!(!report.censored);
        assert_eq!(report.control_width, Some(0));
        assert!(report.r_ent.is_none());
    }

    /// G1 on the estimator: the same capture pair twice is bit-identical —
    /// every fixture in the module, every field.
    #[test]
    fn width_census_is_deterministic() {
        let config = tiny_config();
        let n = 8usize;
        let (q, kv) = synthetic_captures(&config, n, 7);
        let a = run_width_census(&q, &kv, &WidthCensusConfig::default()).expect("census");
        let b = run_width_census(&q, &kv, &WidthCensusConfig::default()).expect("census");
        let fa = format!("{:?}", a.per_head);
        let fb = format!("{:?}", b.per_head);
        assert_eq!(fa, fb);
    }

    /// The specialization semantics of R_ε^ent: every real query in the
    /// group pinned to ONE token (a constant peaked query) keeps the real
    /// width at the first block, while the entropy-matched control queries
    /// each peak on their OWN random token — spreading the control's span.
    /// Real width < control width ⇒ r_ent < 1 = query-distribution
    /// specialization, the reading the run lane's verdicts use.
    #[test]
    fn peaked_real_queries_specialize_against_the_control() {
        let config = tiny_config();
        let n = 12usize;
        let head_dim = config.head_dim;
        let (mut q, kv) = synthetic_captures(&config, n, 3);
        // Rewrite every real query as KV head 0's key row at token 2 (all
        // heads, all positions) — attention collapses onto token 2.
        let kvd = kv.kv_dim;
        let peak: Vec<f32> = kv.key_rows[0][2 * kvd..2 * kvd + head_dim].to_vec();
        let q_dim = config.n_head * head_dim;
        for slot in q.q_rows.iter_mut() {
            for pos in 0..n {
                for h in 0..config.n_head {
                    let off = pos * q_dim + h * head_dim;
                    slot[off..off + head_dim].copy_from_slice(&peak);
                }
            }
        }
        let census = run_width_census(
            &q,
            &kv,
            &WidthCensusConfig {
                // block 2 — the default 8 would consume the whole 12-token
                // window in one block and mask the separation.
                block: 2,
                ..WidthCensusConfig::default()
            },
        )
        .expect("census");
        let report = &census.per_head[0];
        let real_w = report.width.expect("peaked queries must not censor");
        let ctrl_w = report
            .control_width
            .expect("the control must not censor on a 12-token window");
        assert!(
            real_w < ctrl_w,
            "specialized real width {real_w} must beat the spread control {ctrl_w}"
        );
        assert!(report.r_ent.expect("ratio on two clean stops") < 1.0);
    }

    /// A tiny ε forces the walk past the cap: right-censored, width None,
    /// and the residual is honest (> ε).
    #[test]
    fn tiny_eps_censors_at_the_cap() {
        let config = tiny_config();
        let n = 8usize;
        let (q, kv) = synthetic_captures(&config, n, 9);
        let cfg = WidthCensusConfig {
            eps: 1e-9,
            block: 2,
            max_atoms: 3,
            held_out_every: 4,
        };
        let census = run_width_census(&q, &kv, &cfg).expect("census");
        let report = &census.per_head[0];
        assert!(report.censored);
        assert!(report.width.is_none());
        assert!(report.residual > cfg.eps);
        assert_eq!(report.atoms.len(), 3);
    }

    /// Mismatched capture pairs refuse loud — never score a half-armed pair.
    #[test]
    fn capture_mismatch_refuses() {
        let config = tiny_config();
        let n = 8usize;
        let (q, kv) = synthetic_captures(&config, n, 13);
        let mut q_wrong_window =
            SinkAuditQCapture::new(&config, &[DeltaNetLayerType::Attention], n + 1);
        q_wrong_window.q_rows = q.q_rows.clone();
        q_wrong_window.n_positions = n + 1;
        assert!(run_width_census(&q_wrong_window, &kv, &WidthCensusConfig::default()).is_err());

        let mut q_wrong_layers =
            SinkAuditQCapture::new(&config, &[DeltaNetLayerType::Attention], n);
        q_wrong_layers.q_rows = q.q_rows.clone();
        q_wrong_layers.attn_layers = vec![0, 1];
        assert!(run_width_census(&q_wrong_layers, &kv, &WidthCensusConfig::default()).is_err());
    }

    // ── Issue 1185 T3 arm 2 — the percentile-hop fixtures ────────────────
    // The katgpt-rs `percentile_index_audit` class: a percentile index must
    // be nearest-rank WITH tail support, because at small n every tail
    // percentile silently degrades to the maximum. These fixtures pin the
    // hop BEFORE any budget number is quoted from the lane.

    /// The 64-head Bonsai census at p99: the correct nearest-rank index is
    /// `ceil(0.99·64) − 1 = 63` — the MAXIMUM — with tail_support 1. The
    /// read is honest only because the output names it: value 63, index 63,
    /// tail 1. This is the hop, pinned as deliberate behavior.
    #[test]
    fn p99_on_a_64_head_census_lands_on_the_max_and_discloses_it() {
        let sorted: Vec<usize> = (0..64).collect();
        let r = percentile_read(&sorted, 0.99).expect("read");
        assert_eq!(r.index, 63);
        assert_eq!(r.value, 63); // the maximum
        assert_eq!(r.tail_support, 1);
    }

    /// Just past the hop: at n = 100 the correct p99 index is 98, NOT the
    /// naive `(0.99·100) as usize = 99` truncation — which lands on the
    /// max. Tail support 2 says the read stands on two samples.
    #[test]
    fn p99_at_n_100_is_not_the_max() {
        let sorted: Vec<usize> = (0..100).collect();
        let r = percentile_read(&sorted, 0.99).expect("read");
        assert_eq!(r.index, 98);
        assert_eq!(r.value, 98);
        assert_ne!(r.value, sorted[99]); // not the maximum
        assert_eq!(r.tail_support, 2);
        // And the hazard the helper exists for, executable: the truncated
        // form's index IS the max's index at this n.
        assert_eq!((0.99f64 * 100.0) as usize, 99);
    }

    /// p50 on 64 heads is the lower median (index 31), tail 33 — the
    /// disclosed replacement for the old bare `widths[n/2]` upper-median.
    #[test]
    fn p50_is_the_lower_median_with_its_tail_named() {
        let sorted: Vec<usize> = (0..64).collect();
        let r = percentile_read(&sorted, 0.50).expect("read");
        assert_eq!(r.index, 31);
        assert_eq!(r.value, 31);
        assert_eq!(r.tail_support, 33);
    }

    /// Refusals and edges: empty → None; p outside (0, 1] → None; p = 1.0
    /// is a legal read of the max with tail 1.
    #[test]
    fn percentile_read_refuses_and_edges() {
        let sorted: Vec<usize> = (0..64).collect();
        assert!(percentile_read(&[], 0.5).is_none());
        assert!(percentile_read(&sorted, 0.0).is_none());
        assert!(percentile_read(&sorted, -0.1).is_none());
        assert!(percentile_read(&sorted, 1.001).is_none());
        let r = percentile_read(&sorted, 1.0).expect("p=1.0 is legal");
        assert_eq!(r.value, 63);
        assert_eq!(r.tail_support, 1);
    }

    /// The summary sorts any-order input and delegates every percentile to
    /// the one rank rule.
    #[test]
    fn width_summary_sorts_and_delegates() {
        let mut widths: Vec<usize> = (0..64).map(|i| 63 - i).collect(); // descending
        widths[0] = 700; // shuffle the extremes, keep n = 64
        let s = width_summary(&widths);
        assert_eq!(s.n, 64);
        assert_eq!(s.min, 0);
        assert_eq!(s.max, 700);
        let mut sorted = widths.clone();
        sorted.sort_unstable();
        assert_eq!(s.p50, percentile_read(&sorted, 0.50));
        assert_eq!(s.p99, percentile_read(&sorted, 0.99));
        assert_eq!(s.p99.as_ref().expect("read").tail_support, 1);
        assert!(width_summary(&[]).p50.is_none());
    }
}
