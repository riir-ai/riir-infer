//! Lattice-conditioning geometry census (riir-train plan 503 T0.1; the
//! sink_audit / weight_audit family shape).
//!
//! A **reference-free, offline, report-only** census over the per-head GDN
//! recurrent states `S ∈ R^{d_k × d_v}` of a frozen checkpoint: after each
//! pinned capture position's state update, the per-head state matrix is
//! snapshotted and reduced to the plan's three closed forms (paper §4.4):
//!
//! - entropy effective rank `r_ent = exp(−Σ p_i ln p_i)` over the singular
//!   spectrum `p_i = σᵢ²/Σσⱼ²` of `S`,
//! - stable rank `r_stable = ‖S‖²_F / σ1²`,
//! - query-alignment gain `G1 = d_k (u1ᵀq)² / ‖q‖²` against the layer's
//!   L2-normed query at the same position (`q_normed` head slice).
//!
//! The singular spectrum comes from a **cyclic Jacobi eigen-solve of the
//! d_k × d_k symmetric Gram matrix `A = S Sᵀ`** in f64 (σᵢ² = λᵢ): one
//! algorithm serves both rank features, λ_max doubles as σ1². The census
//! answers the plan's Phase-0 question — what does UNTRAINED-style GDN state
//! geometry look like on OUR checkpoint, so the Phase-2 lattice arm is
//! judged against a measured baseline — and T0.2's probe re-runs it under a
//! deterministic FiLM injection to report the shift.
//!
//! Posture (family law): opt-in `lattice_census`, report-only (never a
//! serving path), zero new deps, one deterministic report. Determinism per
//! the katgpt-rs Issue-940 law: f64 sequential ascending folds everywhere,
//! cyclic Jacobi in a fixed sweep order with a fixed iteration cap, pinned
//! first-index tie rules, fixed-precision payload text — two runs over the
//! same checkpoint + corpus + window produce the identical digest. The
//! report is the artifact; the census never gates anything by itself.
//!
//! Capture design (the sink_audit extraction lesson): the recurrent state is
//! DURABLY held in `HybridCache.deltanet_state` between position steps, so
//! the driver snapshots it directly at the pinned positions — **zero
//! forward-body edits** for the state half. The query-alignment feature
//! needs the transient `DeltaNetLayerScratch.q_normed` (it dies at the end
//! of each layer step), so its capture arm rides the scratch the way
//! `sink_audit_q` does: the driver arms [`LatticeQueryGainCapture`] with the
//! SNAPSHOT positions (the state's pos+1 labels); the two cfg write-sites
//! (ternary + dense main body, the census's own postures) copy the layer's
//! `q_normed` into the slot for `pos+1` right after the GDN layer returns.
//! Unarmed callers pay a None check; the report carries `query_armed=false`
//! + `g1=na` rows until the arm is wired.

use super::ternary_weights::ternary_matvec_parallel;
use crate::types::{Config, DeltaNetLayerType};
use anyhow::{Result, bail};
use std::fmt::Write as _;

/// Snapshot budget guard (the family cost-bound law): a capture position
/// holds `[n_gdn_layers × n_v_heads × d_k × d_v]` f32 — on a 24-layer hybrid
/// that is ~16 MiB per position (16 GDN layers × 1 MiB). The census refuses
/// loud above [`MAX_CAPTURE_POSITIONS`], naming the budget, instead of
/// silently ballooning.
pub const MAX_CAPTURE_POSITIONS: usize = 64;

/// Jacobi sweep cap: 128×128 cyclic Jacobi converges far below this on
/// Gram matrices of real recurrent states; the cap only pins determinism
/// (a fixed worst-case iteration count, never a convergence race).
const JACOBI_MAX_SWEEPS: usize = 30;

/// Convergence threshold relative to the diagonal scale: stop sweeping when
/// the largest off-diagonal magnitude falls below this fraction of the
/// largest diagonal element.
const JACOBI_REL_EPS: f64 = 1e-12;

/// Geometry of one head's state matrix at one capture point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeadGeometry {
    /// Entropy effective rank `exp(−Σ p ln p)`, p over the eigenvalues of
    /// S Sᵀ (σᵢ²). A zero state reads 0.0 (the zero-spectrum row is
    /// structural, counted in the report, never flagged).
    pub r_ent: f64,
    /// Stable rank `‖S‖²_F / σ1²` (= Σλ / λ_max); 0.0 on a zero state.
    pub r_stable: f64,
    /// Query-alignment gain `d_k (u1ᵀq)² / ‖q‖²` when the capture arm fed a
    /// query; `None` until the scratch capture arm lands (plan 503 T0.1
    /// compile-window note).
    pub query_gain: Option<f64>,
}

/// One capture point: every GDN layer's per-head geometry, in layer order.
#[derive(Clone, Debug)]
pub struct CapturePoint {
    /// The forward position the snapshot was taken AFTER (the state at this
    /// point includes position `pos`'s update).
    pub pos: usize,
    /// Per GDN layer (ascending layer index), per head (ascending), the
    /// head geometry. `geometry[layer_slot][head]`.
    pub geometry: Vec<Vec<HeadGeometry>>,
}

/// The query-gain capture arm (plan 503 T0.1(b), the `sink_audit_q` mirror):
/// the driver arms it with the SNAPSHOT positions (the state's pos+1 labels,
/// ascending — the same schedule the state snapshots use); the two cfg
/// write-sites (ternary + dense main body) copy each GDN layer's transient
/// `q_normed` into the snapshot's slot right after the layer returns — the
/// step at position `p` writes slot `p+1`, exactly the state the snapshot at
/// `p+1` reads. G1 pairs the top singular vector of that state with THIS
/// query. Slots are addressed by layer (the census's `layer_slot` order =
/// ascending GDN layer) and by row (the snapshot position's ordinal in the
/// armed list). Offline instrument: sized `n_gdn × snaps × n_v_heads×d_k`
/// f32 (27B: 48 × 8 × 2048 × 4 B ≈ 3.1 MB).
pub struct LatticeQueryGainCapture {
    /// Slot `s` holds `[snaps × n_v_heads × d_k]` q_normed rows for GDN
    /// layer `gdn_layers[s]` (parallel arrays).
    pub q_rows: Vec<Vec<f32>>,
    /// The `layer_idx` values that are GDN layers, ascending.
    pub gdn_layers: Vec<usize>,
    /// `layer_idx → slot` (None for attention layers), built once.
    slot_of_layer: Vec<Option<usize>>,
    /// The armed snapshot positions, ascending (binary-search keyed).
    snaps: Vec<usize>,
    /// Row length per layer: `n_v_heads × d_k`.
    row_len: usize,
}

impl LatticeQueryGainCapture {
    /// Size the capture for one census pass: `snaps` rows per GDN layer.
    pub fn new(config: &Config, layer_types: &[DeltaNetLayerType], snaps: &[usize]) -> Self {
        let row_len = config.deltanet_linear_n_value_heads * config.deltanet_linear_head_dim;
        let mut gdn_layers = Vec::new();
        let mut slot_of_layer = vec![None; layer_types.len()];
        for (idx, lt) in layer_types.iter().enumerate() {
            if *lt == DeltaNetLayerType::DeltaNet {
                slot_of_layer[idx] = Some(gdn_layers.len());
                gdn_layers.push(idx);
            }
        }
        let q_rows = gdn_layers
            .iter()
            .map(|_| vec![0.0f32; snaps.len() * row_len])
            .collect();
        Self {
            q_rows,
            gdn_layers,
            slot_of_layer,
            snaps: snaps.to_vec(),
            row_len,
        }
    }

    /// Capture one GDN layer's `q_normed` for the step at position `pos` —
    /// written into the snapshot slot for `pos+1` (the pairing contract:
    /// the snapshot taken AFTER pos's update is the state at pos+1). A
    /// position outside the armed snapshot schedule is a no-op (the arm
    /// captures only the pinned pairs; every other step pays one search).
    pub(crate) fn capture(&mut self, layer_idx: usize, pos: usize, q_normed: &[f32]) {
        let Some(slot) = self.slot_of_layer[layer_idx] else {
            return;
        };
        let Ok(row) = self.snaps.binary_search(&(pos.wrapping_add(1))) else {
            return;
        };
        let off = row * self.row_len;
        let n = q_normed.len().min(self.row_len);
        self.q_rows[slot][off..off + n].copy_from_slice(&q_normed[..n]);
    }

    /// The captured query row for `(layer_slot, snapshot ordinal)` — the
    /// census driver pairs it with the state snapshot at the same ordinal.
    pub fn q_row(&self, layer_slot: usize, snap_ordinal: usize) -> &[f32] {
        let off = snap_ordinal * self.row_len;
        &self.q_rows[layer_slot][off..off + self.row_len]
    }

    /// The armed snapshot positions (the driver pairs rows by ordinal).
    pub fn snaps(&self) -> &[usize] {
        &self.snaps
    }
}

/// The census report (deterministic payload text is the artifact).
#[derive(Clone, Debug)]
pub struct LatticeCensusReport {
    /// Pinned capture positions, ascending — mirrors the driver's snapshot
    /// schedule exactly.
    pub positions: Vec<usize>,
    /// One row per capture point.
    pub points: Vec<CapturePoint>,
    /// Config echo: `(n_gdn_layers, n_v_heads, d_k, d_v)`.
    pub shape: (usize, usize, usize, usize),
    /// Whether the query capture arm was armed (mirrors the wiring state so
    /// a report without `query_gain` rows is self-describing, never silent).
    pub query_armed: bool,
}

impl LatticeCensusReport {
    /// The canonical payload: fixed-precision deterministic text (the BLAKE3
    /// input). Two runs over the same checkpoint + corpus + window produce
    /// byte-identical text.
    pub fn canonical_payload(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(
            s,
            "lattice_census v1 shape={},{},{},{} query_armed={}",
            self.shape.0, self.shape.1, self.shape.2, self.shape.3, self.query_armed
        );
        let _ = writeln!(s, "positions={:?}", self.positions);
        for p in &self.points {
            let _ = writeln!(s, "# pos={}", p.pos);
            for (layer_slot, heads) in p.geometry.iter().enumerate() {
                for (head, g) in heads.iter().enumerate() {
                    let _ = writeln!(
                        s,
                        "row layer={} head={} r_ent={:.6} r_stable={:.6} g1={}",
                        layer_slot,
                        head,
                        g.r_ent,
                        g.r_stable,
                        match g.query_gain {
                            Some(v) => format!("{v:.6}"),
                            None => "na".to_string(),
                        }
                    );
                }
            }
        }
        s
    }

    /// BLAKE3 digest over [`Self::canonical_payload`] — the report pin.
    pub fn digest(&self) -> String {
        blake3::hash(self.canonical_payload().as_bytes()).to_string()
    }

    /// Per-layer mean r_ent over all capture points (ascending layer slot),
    /// sequential ascending fold (the Issue-940 law).
    pub fn mean_r_ent_per_layer(&self) -> Vec<f64> {
        let n_layers = self.shape.0;
        let mut out = vec![0.0f64; n_layers];
        if self.points.is_empty() {
            return out;
        }
        let mut count = 0u64;
        for p in &self.points {
            for (layer_slot, heads) in p.geometry.iter().enumerate() {
                for g in heads {
                    out[layer_slot] += g.r_ent;
                }
            }
            count += 1;
        }
        let heads = self.shape.1 as f64;
        for v in out.iter_mut() {
            *v /= (count as f64) * heads;
        }
        out
    }
}

/// Geometry of every head of one state snapshot.
///
/// `state` = the layer's recurrent state `[n_v_heads × d_k × d_v]`
/// (row-major, head-major — the `DeltaNetState` layout). `queries` (when the
/// capture arm is wired) = the layer's `q_normed` at the same step
/// `[n_v_heads × d_k]`; head h's query is `queries[h × d_k ..]`.
pub fn census_state_snapshot(
    state: &[f32],
    queries: Option<&[f32]>,
    n_v_heads: usize,
    d_k: usize,
    d_v: usize,
) -> Result<Vec<HeadGeometry>> {
    let per_head = d_k * d_v;
    if state.len() != n_v_heads * per_head {
        bail!(
            "state len {} != n_v_heads({n_v_heads}) × d_k×d_v({per_head})",
            state.len()
        );
    }
    if let Some(q) = queries
        && q.len() != n_v_heads * d_k
    {
        bail!(
            "queries len {} != n_v_heads({n_v_heads}) × d_k({d_k})",
            q.len()
        );
    }
    let mut out = Vec::with_capacity(n_v_heads);
    for h in 0..n_v_heads {
        let s = &state[h * per_head..(h + 1) * per_head];
        let q = queries.map(|q| &q[h * d_k..(h + 1) * d_k]);
        out.push(head_geometry(s, q, d_k, d_v)?);
    }
    Ok(out)
}

/// The three closed forms for one head.
fn head_geometry(s: &[f32], q: Option<&[f32]>, d_k: usize, d_v: usize) -> Result<HeadGeometry> {
    // A = S Sᵀ in f64, sequential ascending fold over S's rows (Issue-940).
    let mut a = vec![0.0f64; d_k * d_k];
    for (i, row) in s.chunks_exact(d_v).enumerate() {
        for (j, col) in s.chunks_exact(d_v).enumerate() {
            if j < i {
                continue; // symmetric fill after
            }
            let mut acc = 0.0f64;
            for t in 0..d_v {
                acc += (row[t] as f64) * (col[t] as f64);
            }
            a[i * d_k + j] = acc;
            a[j * d_k + i] = acc;
        }
    }
    let (eigenvalues, top_vec) = jacobi_eigen(&mut a, d_k)?;
    let lambda_max = eigenvalues[0]; // descending order from jacobi_eigen

    let (r_ent, r_stable) = if lambda_max <= 0.0 {
        (0.0, 0.0) // zero state: structural row
    } else {
        let total = eigenvalues.iter().sum::<f64>().max(0.0);
        if total <= 0.0 {
            (0.0, 0.0)
        } else {
            let mut ent = 0.0f64;
            for &l in &eigenvalues {
                if l > 0.0 {
                    let p = l / total;
                    ent -= p * p.ln();
                }
            }
            (ent.exp(), total / lambda_max)
        }
    };

    let query_gain = q.map(|q| query_alignment_gain(top_vec.as_slice(), q, d_k));
    Ok(HeadGeometry {
        r_ent,
        r_stable,
        query_gain,
    })
}

/// `G1 = d_k (u1ᵀ q)² / ‖q‖²` — pure function, fixture-testable without a
/// checkpoint. `u1` = the top left singular vector (Jacobi's top
/// eigenvector of S Sᵀ, length d_k); `q` = the head's L2-normed query
/// (length d_k). A zero query reads 0.0.
pub fn query_alignment_gain(u1: &[f64], q: &[f32], d_k: usize) -> f64 {
    debug_assert_eq!(u1.len(), d_k);
    debug_assert_eq!(q.len(), d_k);
    let mut qn2 = 0.0f64;
    for &v in q {
        qn2 += (v as f64) * (v as f64);
    }
    if qn2 <= 0.0 {
        return 0.0;
    }
    let mut dot = 0.0f64;
    for i in 0..d_k {
        dot += u1[i] * (q[i] as f64);
    }
    (d_k as f64) * dot * dot / qn2
}

/// Cyclic Jacobi eigen-decomposition of a symmetric d×d matrix (in place),
/// descending eigenvalues, returning `(eigenvalues, top_eigenvector_column)`.
///
/// Determinism: fixed cyclic rotation order (i ascending, j > i), the
/// standard stable rotation formulas in f64, sequential ascending folds in
/// the eigenvector accumulation, a fixed sweep cap. The eigenvectors
/// accumulate in a per-column V matrix; column 0 (after the descending
/// sort) is u1.
fn jacobi_eigen(a: &mut [f64], d: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    if a.len() != d * d {
        bail!("jacobi: matrix len {} != d×d ({d}×{d})", a.len());
    }
    // V starts as identity; rotations accumulate.
    let mut v = vec![0.0f64; d * d];
    for i in 0..d {
        v[i * d + i] = 1.0;
    }
    let diag_scale = (0..d)
        .map(|i| a[i * d + i].abs())
        .fold(0.0f64, |m, x| if x > m { x } else { m });
    if diag_scale == 0.0 {
        // Zero matrix: all eigenvalues 0; u1 = e0 (first-index tie rule).
        return Ok((vec![0.0; d], {
            let mut u = vec![0.0; d];
            u[0] = 1.0;
            u
        }));
    }
    let threshold = diag_scale * JACOBI_REL_EPS;
    for _ in 0..JACOBI_MAX_SWEEPS {
        // Largest |off-diagonal| this sweep.
        let mut max_off = 0.0f64;
        for i in 0..d {
            for j in (i + 1)..d {
                let x = a[i * d + j].abs();
                if x > max_off {
                    max_off = x;
                }
            }
        }
        if max_off <= threshold {
            break;
        }
        for i in 0..d {
            for j in (i + 1)..d {
                let aij = a[i * d + j];
                if aij.abs() <= threshold {
                    continue;
                }
                let theta = (a[j * d + j] - a[i * d + i]) / (2.0 * aij);
                // t = sign(theta) / (|theta| + sqrt(theta² + 1)) — the stable form.
                let t = if theta >= 0.0 {
                    1.0 / (theta + (theta * theta + 1.0).sqrt())
                } else {
                    -1.0 / (-theta + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // Rotate rows/cols i,j of A.
                for k in 0..d {
                    let aik = a[i * d + k];
                    let ajk = a[j * d + k];
                    a[i * d + k] = c * aik - s * ajk;
                    a[j * d + k] = s * aik + c * ajk;
                }
                for k in 0..d {
                    let aki = a[k * d + i];
                    let akj = a[k * d + j];
                    a[k * d + i] = c * aki - s * akj;
                    a[k * d + j] = s * aki + c * akj;
                }
                // Accumulate into V (columns i,j rotate).
                for k in 0..d {
                    let vki = v[k * d + i];
                    let vkj = v[k * d + j];
                    v[k * d + i] = c * vki - s * vkj;
                    v[k * d + j] = s * vki + c * vkj;
                }
            }
        }
    }
    // Extract eigenvalues (diagonal), sort DESCENDING with the first-index
    // tie rule (stable sort by reversed comparator on index).
    let mut order: Vec<usize> = (0..d).collect();
    order.sort_by(|&x, &y| {
        let ax = a[x * d + x];
        let ay = a[y * d + y];
        ay.partial_cmp(&ax).unwrap_or(std::cmp::Ordering::Equal)
    });
    let eigenvalues: Vec<f64> = order.iter().map(|&i| a[i * d + i]).collect();
    let top: Vec<f64> = (0..d).map(|k| v[k * d + order[0]]).collect();
    Ok((eigenvalues, top))
}

// ── T0.2 — the fixed-coeff FiLM probe filter (modelless extraction test) ──

/// A deterministic fixed-coefficient FiLM affine filter over the GDN layer
/// input: `x' = γ ⊙ x + β` with γ, β derived ONCE from (layer, seed) by
/// sigmoid-bounded EMA-style arithmetic — never softmax, never trained,
/// never random at run time (the evolve_belief family shape).
///
/// Scale law: γ ∈ (1−γ_amp, 1+γ_amp), β ∈ (−β_amp, +β_amp) with the amplitudes
/// pinned small (the probe perturbs, it does not drive); a γ amplitude of 0
/// with β amplitude 0 is the identity probe (the census must read UNCHANGED —
/// the probe's own negative control, fixture-pinned).
///
/// Injection seam (T0.2, compile-window note): the ternary GDN body accepts
/// `input_proj_hook: Option<&dyn TernaryInputProjHook>` — the hook receives
/// the RMSNorm'd layer input `x` and computes the four projections, so a
/// hook impl applying this filter to a LOCAL copy of x injects FiLM with
/// zero body edits ON UNFOLDED layers. ⛔ The folded path (Bonsai-2-27B
/// PQ2_0 carries `bonsai2_hadamard` rotation) BYPASSES the fused
/// input-proj hook (the Issue-980 rotated-QKVZ memoization cannot apply the
/// transform) — on folded layers the probe must instead wrap the matvec
/// hook (`hook: Option<&dyn TernaryMatvecHook>`, which the rotated path
/// still consults for qkv/z) or inject pre-rotation into `rotation_buf`'s
/// feed. The seam choice is recorded in the run report (`injection_seam`);
/// a probe that silently no-ops on a folded checkpoint is the failure the
/// F1 engagement-proof lesson (Bench-1124 class) exists to catch — the
/// driver verifies injection ENGAGED (a non-identity filter must move at
/// least one census row) before any shift is reported.
#[derive(Clone, Debug)]
pub struct FilmFilter {
    /// Per-channel multiplicative coefficients, length n_embd.
    pub gamma: Vec<f32>,
    /// Per-channel additive coefficients, length n_embd.
    pub beta: Vec<f32>,
}

impl FilmFilter {
    /// Derive the filter for `layer` from a fixed integer seed: deterministic
    /// LCG → u32 channel samples → sigmoid-bounded amplitudes. Sequential
    /// ascending construction; no RNG dependency (the corpus-wide determinism
    /// law — an unseeded-global draw is banned).
    pub fn fixed_coeff(n_embd: usize, layer: usize, seed: u64) -> Self {
        // 64-bit LCG (Numerical Recipes g), fixed constants.
        let mut state = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407)
            .wrapping_add((layer as u64) << 32);
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };
        const GAMMA_AMP: f32 = 0.05; // ±5% multiplicative
        const BETA_AMP: f32 = 0.01; // ±0.01 additive on an RMSNorm'd input
        let mut gamma = Vec::with_capacity(n_embd);
        let mut beta = Vec::with_capacity(n_embd);
        for _ in 0..n_embd {
            let g = next();
            let b = next();
            // u ∈ (0,1): the u32 → unit-interval map.
            let ug = ((g >> 11) as f64) / ((1u64 << 53) as f64);
            let ub = ((b >> 11) as f64) / ((1u64 << 53) as f64);
            // EMA-style sigmoid-bounded map: sigmoid centered at 0.
            // ⛔ unary minus binds LOOSER than the method call in Rust:
            // `-(x).exp()` parses as `-(x.exp())` = −exp(x), not exp(−x)
            // — the sg=1.185/−0.248 fixture failure was exactly this.
            let xg = (ug - 0.5) * 4.0;
            let xb = (ub - 0.5) * 4.0;
            let sg = 1.0f64 / (1.0 + (-xg).exp());
            let sb = 1.0f64 / (1.0 + (-xb).exp());
            gamma.push((1.0 + (2.0 * sg - 1.0) * (GAMMA_AMP as f64)) as f32);
            beta.push(((2.0 * sb - 1.0) * (BETA_AMP as f64)) as f32);
        }
        Self { gamma, beta }
    }

    /// Apply in place: `x'[i] = γ[i]·x[i] + β[i]` (ascending, in-place —
    /// zero-alloc).
    pub fn apply(&self, x: &mut [f32]) {
        for (i, xi) in x.iter_mut().enumerate() {
            *xi = self.gamma[i] * *xi + self.beta[i];
        }
    }

    /// The identity probe (γ=1, β=0): the census MUST read unchanged under
    /// it — the filter's own negative control.
    pub fn identity(n_embd: usize) -> Self {
        Self {
            gamma: vec![1.0; n_embd],
            beta: vec![0.0; n_embd],
        }
    }

    /// Whether this filter is exactly the identity (the driver skips the
    /// engagement check only for this shape).
    pub fn is_identity(&self) -> bool {
        self.gamma.iter().all(|&g| g == 1.0) && self.beta.iter().all(|&b| b == 0.0)
    }
}

// ── T0.2 ─ the FiLM matvec-wrap hook (the folded-path injection seam) ──

/// The single-layer FiLM hook (the probe posture — ONE GDN layer armed per
/// run). The forward passes ONE hook to every ternary matvec, so the armed
/// layer's matvecs are keyed by the RESIDENT-PLANE ADDRESS of its
/// `in_proj_qkv` / `in_proj_z` (`as_ternary` is buffer-stable — Issue 043);
/// every other matvec dispatches the incumbent kernel unchanged, so the
/// census reads the injected layer's downstream state shift and nothing else.
/// Addresses are stored as `usize` (the trait is Send+Sync; raw pointers are
/// not — the address-keyed upload-cache pattern is this trait's own
/// convention). The weights outlive the hook: the census bin holds the
/// `QwenDeltaNetTernaryWeights` for the whole run.
///
/// The equivariance (plan 503): γ⊙(Rx) = R(γ⊙x) — the diagonal gamma
/// commutes with the ±1 rotation — but β does not, so the hook carries
/// the PRE-ROTATED beta (Rβ, computed once from the layer's signs) and
/// applies γ⊙x_rot + Rβ, which equals FiLM-then-rotate exactly. On an
/// unfolded layer the signs are all `+1`, so Rβ = β and the same form is
/// the plain FiLM. The plain beta NEVER appears on the armed path — `apply`
/// (γx + β) would double-add it on a rotated layer.
pub struct FilmMatvecHook {
    pub filter: FilmFilter,
    pub rotated_beta: Vec<f32>,
    pub armed_qkv_addr: usize,
    pub armed_z_addr: usize,
}

impl FilmMatvecHook {
    /// Arm ONE GDN layer: `qkv`/`z` are the layer's `in_proj_qkv`/`in_proj_z`
    /// (ternary or Q2-resident — `as_ternary` answers with the resident
    /// planes, the same view the forward hands the hook); `signs` is the
    /// rotation's sign vector for width n_embd (all `+1` when unfolded).
    /// Refuses loud on a non-ternary armed projection (fail-closed — never a
    /// silent no-op hook).
    pub fn arm_layer(
        filter: FilmFilter,
        qkv: &super::ternary_weights::ProjWeights,
        z: &super::ternary_weights::ProjWeights,
        signs: &[i8],
    ) -> Self {
        let rotated_beta = filter.rotated_beta(signs);
        let qkv_addr = qkv
            .as_ternary()
            .expect("armed layer in_proj_qkv must be ternary (T0.2 probe posture)")
            as *const _ as usize;
        let z_addr = z
            .as_ternary()
            .expect("armed layer in_proj_z must be ternary (T0.2 probe posture)")
            as *const _ as usize;
        Self {
            filter,
            rotated_beta,
            armed_qkv_addr: qkv_addr,
            armed_z_addr: z_addr,
        }
    }
}

impl katgpt_core::TernaryMatvecHook for FilmMatvecHook {
    fn matvec(&self, w: &katgpt_core::TernaryGroupWeights, x: &[f32], y: &mut [f32]) {
        let addr = w as *const _ as usize;
        if addr != self.armed_qkv_addr && addr != self.armed_z_addr {
            // Not an armed projection: the exact incumbent path.
            ternary_matvec_parallel(w, &x[..w.cols], &mut y[..w.rows]);
            return;
        }
        // Folded-path FiLM exactness: the input here is ALREADY rotated
        // (x_rot = Rx on the folded layers), so the exact FiLM-then-rotate
        // form is γ⊙x_rot + Rβ — no plain beta (it would double-add).
        let mut xf = x.to_vec();
        for ((xi, g), rb) in xf[..w.cols]
            .iter_mut()
            .zip(self.filter.gamma.iter())
            .zip(self.rotated_beta.iter())
        {
            *xi = g * *xi + rb;
        }
        ternary_matvec_parallel(w, &xf[..w.cols], &mut y[..w.rows]);
    }
}

impl FilmFilter {
    /// Rβ — the beta with the layer's sign pattern applied (the rotation's
    /// ±1 signs), computable once per layer: γ commutes with the rotation,
    /// β does not, so the rotated-path FiLM needs the rotated beta.
    pub fn rotated_beta(&self, signs: &[i8]) -> Vec<f32> {
        self.beta
            .iter()
            .zip(signs.iter())
            .map(|(b, s)| b * (*s as f32))
            .collect()
    }
}

#[cfg(test)]
mod film_hook_tests {
    use super::super::ternary_weights::ProjWeights;
    use super::*;

    fn group(rows: usize, cols: usize) -> katgpt_core::TernaryGroupWeights {
        let src: Vec<f32> = (0..rows * cols)
            .map(|i| if i % 3 == 0 { 0.7 } else { -0.4 })
            .collect();
        katgpt_core::TernaryGroupWeights::quantize_from_f32(&src, rows, cols)
    }

    #[test]
    fn rotated_beta_flips_by_signs() {
        let mut f = FilmFilter::identity(4);
        f.beta = vec![0.1, -0.2, 0.3, -0.4];
        let signs = [1i8, -1, 1, -1];
        let rb = f.rotated_beta(&signs);
        assert_eq!(rb, vec![0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn query_capture_pairs_step_pos_with_next_snapshot_slot() {
        // The pairing contract: the step at pos p writes the snapshot slot
        // for p+1; positions outside the armed schedule no-op; attention
        // layers no-op.
        let mut config = Config::qwen_deltanet(2, vec![]);
        config.deltanet_linear_head_dim = 8;
        config.deltanet_linear_n_value_heads = 2;
        let layer_types = vec![DeltaNetLayerType::DeltaNet, DeltaNetLayerType::Attention];
        let mut cap = LatticeQueryGainCapture::new(&config, &layer_types, &[1, 65]);
        assert_eq!(cap.gdn_layers, vec![0]);
        let q: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect(); // 2 heads × d_k 8

        cap.capture(0, 0, &q); // → snapshot slot 1 (row 0)
        cap.capture(0, 64, &q); // → snapshot slot 65 (row 1)
        cap.capture(0, 7, &q); // pos 8 not armed → no-op
        cap.capture(1, 0, &q); // attention layer → no-op

        assert_eq!(cap.q_row(0, 0), q.as_slice());
        assert_eq!(cap.q_row(0, 1), q.as_slice());
        // The row the schedule never armed stays zero-filled.
        let untouched = LatticeQueryGainCapture::new(&config, &layer_types, &[3]);
        assert!(untouched.q_row(0, 0).iter().all(|&v| v == 0.0));
    }

    #[test]
    fn armed_matvec_films_and_non_armed_dispatches_incumbent() {
        let (rows, cols) = (384usize, 256usize);
        let armed = group(rows, cols);
        let other = group(rows, cols);
        let x: Vec<f32> = (0..cols).map(|i| (i % 7) as f32 * 0.25 - 0.75).collect();
        let mut f = FilmFilter::identity(cols);
        // Non-identity by hand: γ = 1.05, β = 0.01 — the armed matvec must
        // DIFFER from the plain kernel; a non-armed one must match it EXACTLY
        // (the address keying never perturbs bystander matvecs).
        f.gamma = vec![1.05; cols];
        f.beta = vec![0.01; cols];
        let signs = vec![1i8; cols];
        let qkv = ProjWeights::Ternary(armed);
        let z = ProjWeights::Ternary(group(rows, cols));
        let hook = FilmMatvecHook::arm_layer(f, &qkv, &z, &signs);

        let mut y_armed = vec![0.0f32; rows];
        let mut y_plain = vec![0.0f32; rows];
        let qkv_w = qkv.as_ternary().unwrap();
        katgpt_core::TernaryMatvecHook::matvec(&hook, qkv_w, &x, &mut y_armed);
        ternary_matvec_parallel(qkv_w, &x, &mut y_plain);
        assert!(
            y_armed != y_plain,
            "the armed projection's matvec must move under a non-identity filter"
        );

        let mut y_bystander = vec![0.0f32; rows];
        let mut y_bystander_plain = vec![0.0f32; rows];
        katgpt_core::TernaryMatvecHook::matvec(&hook, &other, &x, &mut y_bystander);
        ternary_matvec_parallel(&other, &x, &mut y_bystander_plain);
        assert_eq!(
            y_bystander, y_bystander_plain,
            "a non-armed projection must dispatch the incumbent kernel unchanged"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * b.abs().max(1e-12)
    }

    #[test]
    fn identity_state_is_full_rank() {
        // S = I (d_k = d_v = 8): every singular value 1 → r_ent = exp(ln 8) = 8,
        // r_stable = 8.
        let mut s = vec![0.0f32; 64];
        for i in 0..8 {
            s[i * 8 + i] = 1.0;
        }
        let g = census_state_snapshot(&s, None, 1, 8, 8).expect("census");
        assert!(approx(g[0].r_ent, 8.0, 1e-6), "r_ent={}", g[0].r_ent);
        assert!(
            approx(g[0].r_stable, 8.0, 1e-6),
            "r_stable={}",
            g[0].r_stable
        );
    }

    #[test]
    fn rank_one_state_reads_one() {
        // S = outer(u, u) with unit u: one singular value √8, rest 0 →
        // r_ent = 1, r_stable = 1. Entropy of a point mass is 0 → exp(0)=1.
        let u = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut s = vec![0.0f32; 64];
        for i in 0..8 {
            for j in 0..8 {
                s[i * 8 + j] = u[i] * u[j];
            }
        }
        let g = census_state_snapshot(&s, None, 1, 8, 8).expect("census");
        assert!(approx(g[0].r_ent, 1.0, 1e-6), "r_ent={}", g[0].r_ent);
        assert!(
            approx(g[0].r_stable, 1.0, 1e-6),
            "r_stable={}",
            g[0].r_stable
        );
    }

    #[test]
    fn zero_state_reads_structural_zero() {
        let s = vec![0.0f32; 64];
        let g = census_state_snapshot(&s, None, 1, 8, 8).expect("census");
        assert_eq!(g[0].r_ent, 0.0);
        assert_eq!(g[0].r_stable, 0.0);
    }

    #[test]
    fn query_gain_oracle() {
        // u1 = e0, q = e0: G1 = d_k · 1 / 1 = d_k (perfect alignment).
        let mut u1 = vec![0.0f64; 8];
        u1[0] = 1.0;
        let mut q = vec![0.0f32; 8];
        q[0] = 1.0;
        assert!(approx(query_alignment_gain(&u1, &q, 8), 8.0, 1e-12));
        // Orthogonal query: G1 = 0 (fresh q — the previous case's e0 must not leak in).
        let mut q2 = vec![0.0f32; 8];
        q2[1] = 1.0;
        assert!(approx(query_alignment_gain(&u1, &q2, 8), 0.0, 1e-12));
        // Zero query: 0.0 (structural).
        let qz = vec![0.0f32; 8];
        assert_eq!(query_alignment_gain(&u1, &qz, 8), 0.0);
    }

    #[test]
    fn report_payload_deterministic() {
        let s = vec![0.5f32; 2 * 8 * 8]; // 2 heads, d_k = d_v = 8
        let g = census_state_snapshot(&s, None, 2, 8, 8).expect("census");
        let report = LatticeCensusReport {
            positions: vec![3],
            points: vec![CapturePoint {
                pos: 3,
                geometry: vec![g],
            }],
            shape: (1, 2, 8, 8),
            query_armed: false,
        };
        let a = report.canonical_payload();
        let b = report.canonical_payload();
        assert_eq!(a, b);
        assert_eq!(report.digest(), report.digest());
        assert!(a.contains("g1=na"), "unarmed reports carry g1=na rows");
    }

    #[test]
    fn jacobi_recovers_known_spectrum() {
        // A = diag(4, 1, 0.25) in a rotated basis — Jacobi must recover the
        // descending spectrum regardless of basis.
        let mut m = vec![0.0f64; 9];
        // m = R A Rᵀ for the 30° Givens rotation R on axes (0,1).
        let a00 = 4.0;
        let a11 = 1.0;
        let c = (30.0f64).to_radians().cos();
        let s = (30.0f64).to_radians().sin();
        m[0] = c * c * a00 + s * s * a11;
        m[1] = c * s * (a00 - a11);
        m[3] = m[1];
        m[4] = s * s * a00 + c * c * a11;
        m[8] = 0.25;
        let (vals, top) = jacobi_eigen(&mut m, 3).expect("jacobi");
        assert!(approx(vals[0], 4.0, 1e-9), "λ0={}", vals[0]);
        assert!(approx(vals[1], 1.0, 1e-9), "λ1={}", vals[1]);
        assert!(approx(vals[2], 0.25, 1e-9), "λ2={}", vals[2]);
        // Top eigenvector is the rotated e0 axis: (c, s) in the 0/1 plane.
        assert!((top[0].abs() - c).abs() < 1e-9 || (top[0].abs() - s).abs() < 1e-9);
    }

    #[test]
    fn capture_budget_refuses_above_cap() {
        // The budget law is enforced in the bin driver, but the constant
        // contract is pinned here: the cap is finite and small.
        const {
            assert!(MAX_CAPTURE_POSITIONS >= 15);
        } // the paper's per-seq posture fits
        const {
            assert!(MAX_CAPTURE_POSITIONS <= 512);
        }
    }

    #[test]
    fn film_filter_deterministic_and_bounded() {
        let a = FilmFilter::fixed_coeff(64, 3, 0x50E1C5);
        let b = FilmFilter::fixed_coeff(64, 3, 0x50E1C5);
        assert_eq!(a.gamma, b.gamma);
        assert_eq!(a.beta, b.beta);
        // Same seed, different layer → different filter (layer-keyed).
        let c = FilmFilter::fixed_coeff(64, 4, 0x50E1C5);
        assert_ne!(a.gamma, c.gamma);
        // Bounds: γ ∈ (0.95, 1.05), β ∈ (−0.01, 0.01).
        for g in &a.gamma {
            assert!((0.95..=1.05).contains(g), "γ={g}");
        }
        for b in &a.beta {
            assert!((-0.01..=0.01).contains(b), "β={b}");
        }
    }

    #[test]
    fn film_identity_reads_unchanged() {
        let mut x: Vec<f32> = (0..32).map(|i| (i as f32) * 0.25 - 4.0).collect();
        let before = x.clone();
        let id = FilmFilter::identity(32);
        assert!(id.is_identity());
        id.apply(&mut x);
        assert_eq!(x, before, "the identity probe must be byte-identical");
        // And a non-identity filter must MOVE the vector (engagement).
        let f = FilmFilter::fixed_coeff(32, 0, 0x50E1C5);
        assert!(!f.is_identity());
        f.apply(&mut x);
        assert_ne!(x, before, "a non-identity filter must engage");
    }
}
