//! `implants` — the exact-implant runtime lane (riir-infer Issue 1256 T1).
//!
//! N **silent modules** mounted beside the decode path: each implant is an
//! exact read table over token ids → a fire gate (a pure function of the
//! bounded token window) → a sampling-seam write (logit biases; the
//! forced-token shape is the dominant-bias entry, the grafts_cell
//! BIAS_DELTA precedent). Silent until fired: a runtime that never fires
//! returns zero writes and the decode is byte-identical — the caller pays
//! one `Option::is_some` branch per step plus the table scan (the
//! `closed_form_write` demote shape).
//!
//! Substrate census (substrate-first gate, 2026-11-26): the grafts
//! primitive (`katgpt-core/grafts`, issue 988) owns the activation-probe
//! read modality, the fitted rule tables, the logit-bias write seam, and
//! the `GraftArtifact`/`GraftSlot` mount. Implants are a DIFFERENT read
//! modality — exact token-id tables, no probes, no capture, no fit — so
//! this module is riir-infer-native and does NOT force its tables through
//! f32 probe cells. The **A2 mount law** is re-stated here, not forked:
//! `surgical_read_patch` set the precedent (its own `SurgicalPatchSlot`
//! cites the GraftSlot law); every refusal class below is fixture-pinned
//! against the same law matrix (corrupt-before-lineage, fork, downgrade,
//! advance, idempotent).
//!
//! Bounded-read law (the 046 refusal): the window is a FIXED-capacity
//! ring (`WINDOW_MAX`); an implant whose pattern needs more than the
//! window refuses at mount. A gate that reads unbounded history cannot be
//! view-enumerated (the Sketchpad upgrade) — it is refused, never
//! silently sampled.
//!
//! Zero-allocation hot path: the window ring and the fire buffer are
//! allocated ONCE at construction/mount; [`ImplantRuntime::observe_into`]
//! allocates nothing (fixture-pinned G4). Deterministic: rules are
//! evaluated in canonical (mount) order, first-match-wins per implant;
//! the artifact codec is canonical-bytes + BLAKE3 (re-encode == bytes, or
//! the decode refuses [`ImplantError::NonCanonical`]).
//!
//! Telemetry is LEVEL-only: per-implant fire counters, never a decision
//! input. Per-implant demote envs follow the drift_guard posture: the
//! exact literal `0` at mount time demotes that implant (a demoted
//! implant never fires; the window still advances — the window is shared
//! stream state, not the implant's).
//!
//! OPT-IN behind the `implants` feature; a default build compiles the
//! module to nothing (the byte-identical-off law).

use std::collections::BTreeSet;

/// Artifact format version this build reads/writes. v2 adds the
/// residual-overlay write action (the T1 rung 2); v1 artifacts refuse
/// loud at the version check (nothing shipped in the wild — the module
/// landed the same day v2 did).
pub const IMPLANT_ARTIFACT_FORMAT_VERSION: u32 = 2;

/// Magic prefix of the canonical byte encoding.
const IMPLANT_MAGIC: &[u8] = b"RIMPL1\x01";

/// The bounded window capacity (the 046 read-surface bound). A pattern
/// longer than this refuses at mount — unbounded-history gates are not
/// representable.
pub const WINDOW_MAX: usize = 16;

/// A token-window predicate over one slot: the exact id, or any of a
/// sorted, deduped id set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenPat {
    /// Exactly this token id.
    Exact(u32),
    /// Any of these token ids (canonical: sorted ascending, deduped).
    AnyOf(Vec<u32>),
}

impl TokenPat {
    /// Whether `token` satisfies this slot. (Crate-visible: the cert
    /// module's decision replication reads it; not part of the public
    /// runtime surface.)
    pub(crate) fn matches(&self, token: u32) -> bool {
        match self {
            Self::Exact(t) => *t == token,
            Self::AnyOf(set) => set.binary_search(&token).is_ok(),
        }
    }

    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            Self::Exact(t) => {
                out.push(0);
                out.extend_from_slice(&t.to_le_bytes());
            }
            Self::AnyOf(set) => {
                out.push(1);
                out.extend_from_slice(&(set.len() as u32).to_le_bytes());
                for t in set {
                    out.extend_from_slice(&t.to_le_bytes());
                }
            }
        }
    }

    fn decode(bytes: &[u8], off: &mut usize) -> Result<Self, ImplantError> {
        let need = |off: &mut usize, n: usize| -> Result<(), ImplantError> {
            if *off + n > bytes.len() {
                Err(ImplantError::Encoding("truncated pattern"))
            } else {
                Ok(())
            }
        };
        need(off, 1)?;
        let tag = bytes[*off];
        *off += 1;
        match tag {
            0 => {
                need(off, 4)?;
                let mut b = [0u8; 4];
                b.copy_from_slice(&bytes[*off..*off + 4]);
                *off += 4;
                Ok(Self::Exact(u32::from_le_bytes(b)))
            }
            1 => {
                need(off, 4)?;
                let n = u32::from_le_bytes(bytes[*off..*off + 4].try_into().unwrap()) as usize;
                *off += 4;
                need(off, n * 4)?;
                let mut set = Vec::with_capacity(n);
                for _ in 0..n {
                    let mut b = [0u8; 4];
                    b.copy_from_slice(&bytes[*off..*off + 4]);
                    *off += 4;
                    set.push(u32::from_le_bytes(b));
                }
                Ok(Self::AnyOf(set))
            }
            _ => Err(ImplantError::Encoding("unknown pattern tag")),
        }
    }
}

/// One authored trigger rule: a pattern aligned to the window END (the
/// most recent `pattern.len()` tokens, oldest→newest) mapping to an
/// action index. A pattern longer than [`WINDOW_MAX`] is refused at
/// construction.
#[derive(Debug, Clone, PartialEq)]
pub struct TriggerRule {
    /// The exact token pattern (oldest→newest, matched against the window
    /// tail). Length 1..=`WINDOW_MAX`.
    pub pattern: Vec<TokenPat>,
    /// The [`ImplantSpec::actions`] index this rule fires.
    pub action: u16,
}

/// One rank-r overlay pair of a residual write (the T1 rung 2): the
/// same shape the `closed_form_write::WriteOverlays` fold applies —
/// `y += alpha · B·(A·key)` at the chosen site. Raw f32 rows, carried
/// spec-side so the `implants` feature stays pure-local; the
/// forward-facing `QuantErrorLora`/`WriteOverlays` construction happens
/// in the consumer tier (`implants_cell`).
///
/// Layout matches `QuantErrorLora`: `a` is `[rank × in_dim]` row-major,
/// `b` is `[out_dim × rank]` row-major; `in_dim`/`out_dim` derive from
/// the row lengths (`a.len()/rank`, `b.len()/rank`) and are validated
/// non-zero at construction.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidualPair {
    /// The fold's rank.
    pub rank: u32,
    /// Scaling factor (1.0 = full correction).
    pub alpha: f32,
    /// `[rank × in_dim]` row-major.
    pub a: Vec<f32>,
    /// `[out_dim × rank]` row-major.
    pub b: Vec<f32>,
}

impl ResidualPair {
    /// Validate + construct: rank ≥ 1, row widths divide exactly, both
    /// dims ≥ 1, every value finite. Shape errors are construction-time
    /// refusals — a malformed pair never reaches an artifact.
    ///
    /// # Errors
    /// [`ImplantError::Usage`] on any malformed shape or a non-finite
    /// value.
    pub fn new(rank: u32, alpha: f32, a: Vec<f32>, b: Vec<f32>) -> Result<Self, ImplantError> {
        if rank == 0 {
            return Err(ImplantError::Usage("residual pair rank 0"));
        }
        let rank = rank as usize;
        if a.is_empty() || !a.len().is_multiple_of(rank) {
            return Err(ImplantError::Usage(
                "residual pair a rows not rank-divisible",
            ));
        }
        if b.is_empty() || !b.len().is_multiple_of(rank) {
            return Err(ImplantError::Usage(
                "residual pair b rows not rank-divisible",
            ));
        }
        if !alpha.is_finite()
            || a.iter().any(|v| !v.is_finite())
            || b.iter().any(|v| !v.is_finite())
        {
            return Err(ImplantError::Usage("residual pair non-finite value"));
        }
        Ok(Self {
            rank: rank as u32,
            alpha,
            a,
            b,
        })
    }

    /// The fold's input width (`a.len() / rank`).
    #[must_use]
    pub fn in_dim(&self) -> usize {
        self.a.len() / self.rank as usize
    }

    /// The fold's output width (`b.len() / rank`).
    #[must_use]
    pub fn out_dim(&self) -> usize {
        self.b.len() / self.rank as usize
    }

    fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.rank.to_le_bytes());
        out.extend_from_slice(&self.alpha.to_le_bytes());
        out.extend_from_slice(&(self.a.len() as u32).to_le_bytes());
        for v in &self.a {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&(self.b.len() as u32).to_le_bytes());
        for v in &self.b {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }

    fn decode(bytes: &[u8], off: &mut usize) -> Result<Self, ImplantError> {
        let take = |off: &mut usize, n: usize| -> Result<(), ImplantError> {
            if *off + n > bytes.len() {
                Err(ImplantError::Encoding("truncated residual pair"))
            } else {
                Ok(())
            }
        };
        take(off, 4)?;
        let rank = u32::from_le_bytes(bytes[*off..*off + 4].try_into().unwrap());
        *off += 4;
        take(off, 4)?;
        let alpha = f32::from_le_bytes(bytes[*off..*off + 4].try_into().unwrap());
        *off += 4;
        let row = |bytes: &[u8], off: &mut usize| -> Result<Vec<f32>, ImplantError> {
            take(off, 4)?;
            let n = u32::from_le_bytes(bytes[*off..*off + 4].try_into().unwrap()) as usize;
            *off += 4;
            take(
                off,
                n.checked_mul(4)
                    .ok_or(ImplantError::Encoding("residual row length overflow"))?,
            )?;
            let mut v = Vec::with_capacity(n.min(1 << 20));
            for _ in 0..n {
                v.push(f32::from_le_bytes(
                    bytes[*off..*off + 4].try_into().unwrap(),
                ));
                *off += 4;
            }
            Ok(v)
        };
        let a = row(bytes, off)?;
        let b = row(bytes, off)?;
        // Shape validation rides the constructor (Usage on a malformed
        // row is the same loud refusal the mint path gives).
        Self::new(rank, alpha, a, b).map_err(|_| ImplantError::Encoding("residual pair shape"))
    }
}

/// The residual-overlay write of one action (the T1 rung 2): the overlay
/// pairs to fold at `layer`'s site when the implant fires. One-shot and
/// fire-gated — the consumer tier arms the fold for the step AFTER the
/// fire (the write lands one step past the trigger completing, the same
/// lag the bias write's effect has on the sampled token) and the next
/// silent step is byte-identical again. The site is named by layer
/// index; the site KIND (the MLP down-proj output — the
/// `closed_form_write` posture) is the consumer tier's contract.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidualOverlay {
    /// The site layer index (consumer-tier meaning; the fold targets
    /// this layer's down-proj output).
    pub layer: u32,
    /// The pairs, applied in order (the Issue-940 fold-order law: the
    /// order is the write's identity).
    pub pairs: Vec<ResidualPair>,
}

impl ResidualOverlay {
    /// Validate + construct: at least one pair, all pairs sharing one
    /// input width and one output width (a mixed-width overlay would
    /// fold different key widths at the same site — refuse at authoring).
    ///
    /// # Errors
    /// [`ImplantError::Usage`] on an empty pair list or a width mismatch.
    pub fn new(layer: u32, pairs: Vec<ResidualPair>) -> Result<Self, ImplantError> {
        if pairs.is_empty() {
            return Err(ImplantError::Usage("residual overlay with no pairs"));
        }
        let (in0, out0) = (pairs[0].in_dim(), pairs[0].out_dim());
        for p in &pairs {
            if p.in_dim() != in0 || p.out_dim() != out0 {
                return Err(ImplantError::Usage("residual overlay mixed pair widths"));
            }
        }
        Ok(Self { layer, pairs })
    }

    /// The shared fold input width.
    #[must_use]
    pub fn in_dim(&self) -> usize {
        self.pairs[0].in_dim()
    }

    /// The shared fold output width.
    #[must_use]
    pub fn out_dim(&self) -> usize {
        self.pairs[0].out_dim()
    }

    fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.layer.to_le_bytes());
        out.extend_from_slice(&(self.pairs.len() as u32).to_le_bytes());
        for p in &self.pairs {
            p.encode(out);
        }
    }

    fn decode(bytes: &[u8], off: &mut usize) -> Result<Self, ImplantError> {
        if *off + 8 > bytes.len() {
            return Err(ImplantError::Encoding("truncated residual overlay"));
        }
        let layer = u32::from_le_bytes(bytes[*off..*off + 4].try_into().unwrap());
        *off += 4;
        let n = u32::from_le_bytes(bytes[*off..*off + 4].try_into().unwrap()) as usize;
        *off += 4;
        let mut pairs = Vec::with_capacity(n.min(256));
        for _ in 0..n {
            pairs.push(ResidualPair::decode(bytes, off)?);
        }
        Self::new(layer, pairs).map_err(|_| ImplantError::Encoding("residual overlay shape"))
    }
}

/// The sampling-seam write: per-token logit biases applied in order when
/// the implant fires. The forced-token shape is a dominant positive bias
/// on the forced id (the grafts_cell BIAS_DELTA 8.0 precedent clears the
/// base top-1 gap). A bias-less action with a residual overlay is the
/// pure-residual write shape (the T1 rung 2); at least one of the two
/// must be present.
#[derive(Debug, Clone, PartialEq)]
pub struct ImplantAction {
    /// `(token, bias)` pairs, applied in order.
    pub biases: Vec<(u32, f32)>,
    /// The fire-gated residual overlay (the T1 rung 2): folded at the
    /// chosen site on the step after the fire. `None` = bias-only.
    pub residual: Option<ResidualOverlay>,
}

/// The implant spec: name, rules, actions, optional per-implant demote
/// env. Canonical order: rules sorted by (pattern bytes, action).
#[derive(Debug, Clone, PartialEq)]
pub struct ImplantSpec {
    /// Stable implant name (telemetry identity; non-empty, ≤ 64 bytes).
    pub name: String,
    /// The trigger rules (canonical: sorted by pattern encoding then
    /// action).
    pub rules: Vec<TriggerRule>,
    /// The write actions (each rule's `action` indexes this).
    pub actions: Vec<ImplantAction>,
    /// Environment variable that demotes this implant at mount when set
    /// to the exact literal `0` (the drift_guard posture). `None` = no
    /// demote seam.
    pub demote_env: Option<String>,
}

impl ImplantSpec {
    /// Validate + construct: canonicalizes rule order, refuses empty
    /// rules/actions, over-long patterns, unknown action indexes, bias-less
    /// actions, and an empty/oversized name.
    ///
    /// # Errors
    /// [`ImplantError::Usage`] on any malformed spec.
    pub fn new(
        name: impl Into<String>,
        mut rules: Vec<TriggerRule>,
        actions: Vec<ImplantAction>,
        demote_env: Option<String>,
    ) -> Result<Self, ImplantError> {
        let name = name.into();
        if name.is_empty() || name.len() > 64 {
            return Err(ImplantError::Usage("implant name empty or over 64 bytes"));
        }
        if rules.is_empty() {
            return Err(ImplantError::Usage("implant with no rules"));
        }
        if actions.is_empty() {
            return Err(ImplantError::Usage("implant with no actions"));
        }
        for rule in &rules {
            if rule.pattern.is_empty() || rule.pattern.len() > WINDOW_MAX {
                return Err(ImplantError::Usage("rule pattern empty or over WINDOW_MAX"));
            }
            if usize::from(rule.action) >= actions.len() {
                return Err(ImplantError::Usage("rule action index out of range"));
            }
            for pat in &rule.pattern {
                if matches!(pat, TokenPat::AnyOf(set) if set.is_empty()) {
                    return Err(ImplantError::Usage("AnyOf pattern with empty set"));
                }
            }
        }
        for act in &actions {
            if act.biases.is_empty() && act.residual.is_none() {
                return Err(ImplantError::Usage("action with no writes"));
            }
        }
        rules.sort_by(|a, b| {
            pattern_rank(&a.pattern)
                .cmp(&pattern_rank(&b.pattern))
                .then(a.action.cmp(&b.action))
        });
        Ok(Self {
            name,
            rules,
            actions,
            demote_env,
        })
    }

    fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.name.len() as u32).to_le_bytes());
        out.extend_from_slice(self.name.as_bytes());
        out.extend_from_slice(&(self.rules.len() as u32).to_le_bytes());
        for rule in &self.rules {
            out.extend_from_slice(&(rule.pattern.len() as u32).to_le_bytes());
            for pat in &rule.pattern {
                pat.encode(out);
            }
            out.extend_from_slice(&rule.action.to_le_bytes());
        }
        out.extend_from_slice(&(self.actions.len() as u32).to_le_bytes());
        for act in &self.actions {
            out.extend_from_slice(&(act.biases.len() as u32).to_le_bytes());
            for (t, b) in &act.biases {
                out.extend_from_slice(&t.to_le_bytes());
                out.extend_from_slice(&b.to_le_bytes());
            }
            match &act.residual {
                None => out.push(0),
                Some(res) => {
                    out.push(1);
                    res.encode(out);
                }
            }
        }
        match &self.demote_env {
            None => out.push(0),
            Some(env) => {
                out.push(1);
                out.extend_from_slice(&(env.len() as u32).to_le_bytes());
                out.extend_from_slice(env.as_bytes());
            }
        }
    }

    fn decode(bytes: &[u8], off: &mut usize) -> Result<Self, ImplantError> {
        let take = |off: &mut usize, n: usize| -> Result<(), ImplantError> {
            if *off + n > bytes.len() {
                Err(ImplantError::Encoding("truncated spec"))
            } else {
                Ok(())
            }
        };
        take(off, 4)?;
        let name_len = u32::from_le_bytes(bytes[*off..*off + 4].try_into().unwrap()) as usize;
        *off += 4;
        take(off, name_len)?;
        let name = String::from_utf8(bytes[*off..*off + name_len].to_vec())
            .map_err(|_| ImplantError::Encoding("name not UTF-8"))?;
        *off += name_len;
        take(off, 4)?;
        let n_rules = u32::from_le_bytes(bytes[*off..*off + 4].try_into().unwrap()) as usize;
        *off += 4;
        let mut rules = Vec::with_capacity(n_rules.min(1024));
        for _ in 0..n_rules {
            take(off, 4)?;
            let pat_len = u32::from_le_bytes(bytes[*off..*off + 4].try_into().unwrap()) as usize;
            *off += 4;
            let mut pattern = Vec::with_capacity(pat_len.min(WINDOW_MAX * 4));
            for _ in 0..pat_len {
                pattern.push(TokenPat::decode(bytes, off)?);
            }
            take(off, 2)?;
            let action = u16::from_le_bytes(bytes[*off..*off + 2].try_into().unwrap());
            *off += 2;
            rules.push(TriggerRule { pattern, action });
        }
        take(off, 4)?;
        let n_actions = u32::from_le_bytes(bytes[*off..*off + 4].try_into().unwrap()) as usize;
        *off += 4;
        let mut actions = Vec::with_capacity(n_actions.min(1024));
        for _ in 0..n_actions {
            take(off, 4)?;
            let n_bias = u32::from_le_bytes(bytes[*off..*off + 4].try_into().unwrap()) as usize;
            *off += 4;
            let mut biases = Vec::with_capacity(n_bias.min(1024));
            for _ in 0..n_bias {
                take(off, 8)?;
                let t = u32::from_le_bytes(bytes[*off..*off + 4].try_into().unwrap());
                let b = f32::from_le_bytes(bytes[*off + 4..*off + 8].try_into().unwrap());
                *off += 8;
                biases.push((t, b));
            }
            take(off, 1)?;
            let residual = match bytes[*off] {
                0 => {
                    *off += 1;
                    None
                }
                1 => {
                    *off += 1;
                    Some(ResidualOverlay::decode(bytes, off)?)
                }
                _ => return Err(ImplantError::Encoding("unknown residual tag")),
            };
            actions.push(ImplantAction { biases, residual });
        }
        take(off, 1)?;
        let demote_env = match bytes[*off] {
            0 => {
                *off += 1;
                None
            }
            1 => {
                *off += 1;
                take(off, 4)?;
                let env_len =
                    u32::from_le_bytes(bytes[*off..*off + 4].try_into().unwrap()) as usize;
                *off += 4;
                take(off, env_len)?;
                let env = String::from_utf8(bytes[*off..*off + env_len].to_vec())
                    .map_err(|_| ImplantError::Encoding("demote env not UTF-8"))?;
                *off += env_len;
                Some(env)
            }
            _ => return Err(ImplantError::Encoding("unknown demote-env tag")),
        };
        // Canonical form is enforced by the re-encode check in
        // `ImplantArtifact::decode` (sorted rules, sorted AnyOf sets are
        // the constructor's job — decode feeds `ImplantSpec::new`).
        Self::new(name, rules, actions, demote_env)
    }
}

/// Stable sort key for a pattern: its canonical encoding.
fn pattern_rank(pattern: &[TokenPat]) -> Vec<u8> {
    let mut buf = Vec::new();
    for pat in pattern {
        pat.encode(&mut buf);
    }
    buf
}

// ── Errors ───────────────────────────────────────────────────────────────

/// Implant errors — typed, loud, never permissive (the graft error
/// taxonomy's shape).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImplantError {
    /// API/precondition misuse (empty rules, over-long patterns, ...).
    Usage(&'static str),
    /// Malformed canonical bytes (bad magic, version, truncation,
    /// trailing bytes, non-finite bias).
    Encoding(&'static str),
    /// The artifact's stored digest does not match BLAKE3 over its
    /// canonical bytes — corrupted after mint.
    DigestMismatch {
        /// The digest the artifact carries.
        stored: [u8; 32],
        /// The digest computed over the canonical bytes.
        computed: [u8; 32],
    },
    /// The decoded spec re-encodes to different bytes — the payload is
    /// not in canonical form (unsorted rules, unsorted AnyOf set).
    NonCanonical,
    /// Same epoch as mounted, different content digest — a lineage fork,
    /// refused loud. The mounted artifact stays.
    Fork {
        /// The epoch already mounted.
        mounted_epoch: u64,
        /// The incoming artifact's content digest.
        incoming_digest: [u8; 32],
    },
    /// An epoch strictly below the mounted one — a downgrade, refused
    /// loud. The mounted artifact stays.
    Downgrade {
        /// The epoch already mounted.
        mounted_epoch: u64,
        /// The refused artifact's epoch.
        incoming_epoch: u64,
    },
    /// A mount with an environment-demote var set to the exact literal
    /// `0` installs metadata only (recorded, never a fire participant).
    /// This is the typed return, not an error — listed here so the
    /// install result vocabulary stays in one enum.
    DemotedAtMount,
}

impl core::fmt::Display for ImplantError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Usage(m) => write!(f, "implant usage: {m}"),
            Self::Encoding(m) => write!(f, "implant artifact encoding: {m}"),
            Self::DigestMismatch { stored, computed } => write!(
                f,
                "implant artifact digest mismatch: stored {} computed {}",
                hex32(stored),
                hex32(computed)
            ),
            Self::NonCanonical => write!(f, "implant artifact not in canonical form"),
            Self::Fork {
                mounted_epoch,
                incoming_digest,
            } => write!(
                f,
                "implant fork: epoch {mounted_epoch} mounted, incoming digest {}",
                hex32(incoming_digest)
            ),
            Self::Downgrade {
                mounted_epoch,
                incoming_epoch,
            } => write!(
                f,
                "implant downgrade: epoch {mounted_epoch} mounted, incoming {incoming_epoch}"
            ),
            Self::DemotedAtMount => write!(f, "implant demoted at mount"),
        }
    }
}

impl std::error::Error for ImplantError {}

/// Lowercase hex for a 32-byte digest (logs, issue records).
#[must_use]
pub fn hex32(d: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in d {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

// ── Artifact (canonical bytes + BLAKE3, the A2 shape) ───────────────────

/// A versioned, digest-pinned implant artifact: the spec's canonical
/// bytes + BLAKE3 + the caller-supplied monotonic epoch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImplantArtifact {
    /// The encoding format version.
    pub format_version: u32,
    /// The caller-supplied monotonic epoch (the mount lineage).
    pub epoch: u64,
    /// BLAKE3 over [`Self::canonical`].
    pub digest: [u8; 32],
    /// The spec's canonical encoding.
    pub canonical: Vec<u8>,
}

impl ImplantArtifact {
    /// Mint: canonicalize the spec, stamp version + epoch + digest.
    /// Deterministic — the same spec + epoch mint byte-identical
    /// artifacts.
    #[must_use]
    pub fn mint(spec: &ImplantSpec, epoch: u64) -> Self {
        let mut canonical = Vec::new();
        canonical.extend_from_slice(IMPLANT_MAGIC);
        canonical.extend_from_slice(&IMPLANT_ARTIFACT_FORMAT_VERSION.to_le_bytes());
        spec.encode(&mut canonical);
        let digest: [u8; 32] = blake3::hash(&canonical).into();
        Self {
            format_version: IMPLANT_ARTIFACT_FORMAT_VERSION,
            epoch,
            digest,
            canonical,
        }
    }

    /// The content digest, lowercase hex.
    #[must_use]
    pub fn digest_hex(&self) -> String {
        hex32(&self.digest)
    }

    /// Verify the stored digest against BLAKE3 over the canonical bytes.
    ///
    /// # Errors
    /// [`ImplantError::DigestMismatch`] on tampered content.
    pub fn verify_digest(&self) -> Result<(), ImplantError> {
        let computed: [u8; 32] = blake3::hash(&self.canonical).into();
        if computed == self.digest {
            Ok(())
        } else {
            Err(ImplantError::DigestMismatch {
                stored: self.digest,
                computed,
            })
        }
    }

    /// Decode the artifact's spec: version check → digest verify (before
    /// ANY parse — corruption is the stronger verdict) → parse →
    /// canonicality (re-encode == bytes, else [`ImplantError::NonCanonical`]).
    ///
    /// # Errors
    /// [`ImplantError::Encoding`] / [`ImplantError::DigestMismatch`] /
    /// [`ImplantError::NonCanonical`], or the spec constructor's
    /// [`ImplantError::Usage`].
    pub fn decode(&self) -> Result<ImplantSpec, ImplantError> {
        if self.format_version != IMPLANT_ARTIFACT_FORMAT_VERSION {
            return Err(ImplantError::Encoding(
                "unsupported artifact format version",
            ));
        }
        self.verify_digest()?;
        let payload = self
            .canonical
            .strip_prefix(IMPLANT_MAGIC)
            .ok_or(ImplantError::Encoding("bad magic"))?;
        let payload = match payload.strip_prefix(&IMPLANT_ARTIFACT_FORMAT_VERSION.to_le_bytes()) {
            Some(p) => p,
            None => return Err(ImplantError::Encoding("bad version prefix")),
        };
        let mut off = 0usize;
        let spec = ImplantSpec::decode(payload, &mut off)?;
        if off != payload.len() {
            return Err(ImplantError::Encoding("trailing bytes after spec"));
        }
        let mut re = Vec::new();
        spec.encode(&mut re);
        let mut full = Vec::with_capacity(self.canonical.len());
        full.extend_from_slice(IMPLANT_MAGIC);
        full.extend_from_slice(&IMPLANT_ARTIFACT_FORMAT_VERSION.to_le_bytes());
        full.extend_from_slice(&re);
        if full != self.canonical {
            return Err(ImplantError::NonCanonical);
        }
        Ok(spec)
    }
}

// ── Slot (the A2 monotonic mount law) ───────────────────────────────────

/// What one [`ImplantSlot::install`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImplantInstall {
    /// First artifact in the slot.
    Installed,
    /// Same epoch, same digest — the idempotent re-install; the slot is
    /// unchanged.
    Idempotent,
    /// The mounted epoch advanced; the old artifact is replaced.
    Advanced {
        /// The epoch that was mounted before.
        from: u64,
    },
}

/// The per-implant monotonic mount. Refusals leave the mounted artifact
/// untouched (the refusal-leaves-the-mount law).
#[derive(Debug, Default)]
pub struct ImplantSlot {
    mounted: Option<(u64, [u8; 32], ImplantSpec)>,
}

impl ImplantSlot {
    /// An empty slot.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The mounted epoch, if any.
    #[must_use]
    pub fn epoch(&self) -> Option<u64> {
        self.mounted.as_ref().map(|(e, _, _)| *e)
    }

    /// The mounted content digest, if any.
    #[must_use]
    pub fn digest(&self) -> Option<[u8; 32]> {
        self.mounted.as_ref().map(|(_, d, _)| *d)
    }

    /// The mounted spec, if any.
    #[must_use]
    pub fn spec(&self) -> Option<&ImplantSpec> {
        self.mounted.as_ref().map(|(_, _, s)| s)
    }

    /// Install an artifact: integrity BEFORE lineage (corruption refuses
    /// before any epoch comparison), then the A2 law — idempotent on
    /// (same epoch, same digest), fork on (same epoch, different digest),
    /// downgrade refused, advance replaces.
    ///
    /// # Errors
    /// See [`ImplantError`]; every refusal leaves the mount untouched.
    pub fn install(&mut self, artifact: &ImplantArtifact) -> Result<ImplantInstall, ImplantError> {
        artifact.verify_digest()?;
        let incoming = artifact.decode()?;
        match self.mounted {
            None => {
                self.mounted = Some((artifact.epoch, artifact.digest, incoming));
                Ok(ImplantInstall::Installed)
            }
            Some((epoch, digest, _)) => {
                if artifact.epoch == epoch {
                    if artifact.digest == digest {
                        Ok(ImplantInstall::Idempotent)
                    } else {
                        Err(ImplantError::Fork {
                            mounted_epoch: epoch,
                            incoming_digest: artifact.digest,
                        })
                    }
                } else if artifact.epoch < epoch {
                    Err(ImplantError::Downgrade {
                        mounted_epoch: epoch,
                        incoming_epoch: artifact.epoch,
                    })
                } else {
                    let from = epoch;
                    self.mounted = Some((artifact.epoch, artifact.digest, incoming));
                    Ok(ImplantInstall::Advanced { from })
                }
            }
        }
    }
}

// ── The composition runtime ─────────────────────────────────────────────

/// A fired rule: which implant, which action. Copy — the fire buffer is
/// caller-owned and reused (the G4 scratch discipline).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FireRef {
    /// The implant index (mount order).
    pub implant: u16,
    /// The fired action index within that implant.
    pub action: u16,
}

/// The N-module composition runtime: one shared token window, per-implant
/// slots + demote posture + fire counters.
///
/// Construction allocates once (window ring + slot vectors); the observe
/// path is zero-alloc. A runtime with zero mounted implants is legal and
/// observes nothing (the caller's `Option::is_some` branch still skips
/// its work — the silent-identity posture).
pub struct ImplantRuntime {
    slots: Vec<ImplantSlot>,
    demoted: Vec<bool>,
    fires: Vec<u64>,
    window: Vec<u32>,
    window_len: usize,
}

impl ImplantRuntime {
    /// An empty runtime over `n` slots (one per planned implant; extra
    /// slots are harmless — they mount later and start silent).
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self {
            slots: (0..n).map(|_| ImplantSlot::new()).collect(),
            demoted: vec![false; n],
            fires: vec![0; n],
            window: Vec::with_capacity(WINDOW_MAX),
            window_len: 0,
        }
    }

    /// Mount artifact `i` into slot `i`: resolve the demote env (exact
    /// literal `0` → demoted), then the A2 install. A demoted implant
    /// records the mount (its slot + counters exist) but never fires.
    ///
    /// # Errors
    /// Slot index out of range, or the install's refusal taxonomy.
    pub fn mount(
        &mut self,
        i: usize,
        artifact: &ImplantArtifact,
    ) -> Result<ImplantInstall, ImplantError> {
        if i >= self.slots.len() {
            return Err(ImplantError::Usage("slot index out of range"));
        }
        let spec = artifact.decode()?;
        let demoted_by_env = spec
            .demote_env
            .as_deref()
            .map(|env| std::env::var(env).as_deref() == Ok("0"))
            .unwrap_or(false);
        if demoted_by_env {
            // Record the mount so telemetry + slot laws stay coherent,
            // then mark demoted. The A2 install semantics hold.
            let install = self.slots[i].install(artifact)?;
            self.demoted[i] = true;
            let _ = install;
            return Err(ImplantError::DemotedAtMount);
        }
        self.demoted[i] = false;
        self.slots[i].install(artifact)
    }

    /// Advance the window by one token, then evaluate every mounted,
    /// undemoted implant's rules against the window tail. Fired rules
    /// append `(implant, action)` refs to `out` (caller-owned buffer —
    /// clear + reuse, never reallocate in steady state) and bump the
    /// per-implant fire counters (LEVEL telemetry). Returns the fire
    /// count. Zero allocation.
    ///
    /// First-match-wins per implant per step: at most one fire per
    /// implant per token (the deterministic bound that keeps G3
    /// composition decidable — an implant never double-writes a step).
    pub fn observe_into(&mut self, token: u32, out: &mut Vec<FireRef>) -> usize {
        debug_assert!(out.is_empty(), "fire buffer must be cleared by the caller");
        // Window advance: fixed-capacity ring semantics via a shifted
        // Vec — WINDOW_MAX ≤ 16, the memmove is bounded and cheaper than
        // ring-index arithmetic on the hot read path below.
        if self.window.len() < WINDOW_MAX {
            self.window.push(token);
        } else {
            self.window.copy_within(1.., 0);
            let last = self.window.len() - 1;
            self.window[last] = token;
        }
        self.window_len = self.window.len();
        let start = self.window.len();
        for (i, slot) in self.slots.iter().enumerate() {
            if self.demoted[i] {
                continue;
            }
            let Some(spec) = slot.spec() else {
                continue;
            };
            for rule in &spec.rules {
                let plen = rule.pattern.len();
                if plen > start {
                    continue;
                }
                let tail = &self.window[start - plen..];
                if rule
                    .pattern
                    .iter()
                    .zip(tail.iter())
                    .all(|(pat, &t)| pat.matches(t))
                {
                    self.fires[i] += 1;
                    out.push(FireRef {
                        implant: i as u16,
                        action: rule.action,
                    });
                    break; // first-match-wins per implant per step
                }
            }
        }
        out.len()
    }

    /// The fired action's biases (resolve a [`FireRef`] against the
    /// mount). Panics on an out-of-range ref — a caller bug, loud.
    #[must_use]
    pub fn action(&self, fire: &FireRef) -> &ImplantAction {
        let spec = self.slots[usize::from(fire.implant)]
            .spec()
            .expect("fire ref names an unmounted implant");
        &spec.actions[usize::from(fire.action)]
    }

    /// Per-implant fire counters (LEVEL telemetry — read-only, never a
    /// decision input).
    #[must_use]
    pub fn fires(&self) -> &[u64] {
        &self.fires
    }

    /// The number of slots.
    #[must_use]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// True when no slot is occupied.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.iter().all(|s| s.spec().is_none())
    }

    /// The demote posture of slot `i`.
    #[must_use]
    pub fn is_demoted(&self, i: usize) -> bool {
        self.demoted.get(i).copied().unwrap_or(false)
    }
}

impl core::fmt::Debug for ImplantRuntime {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ImplantRuntime")
            .field("slots", &self.slots.len())
            .field(
                "mounted",
                &self.slots.iter().filter(|s| s.spec().is_some()).count(),
            )
            .field("demoted", &self.demoted.iter().filter(|&&d| d).count())
            .field("fires", &self.fires)
            .field("window_len", &self.window_len)
            .finish()
    }
}

/// Canonicalize an `AnyOf` set: sorted + deduped. The constructor path
/// for authored specs.
#[must_use]
pub fn any_of(ids: &[u32]) -> TokenPat {
    let set: BTreeSet<u32> = ids.iter().copied().collect();
    TokenPat::AnyOf(set.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec_two_plus_two() -> ImplantSpec {
        // "2 + 2" (exact ids 0, 1, 2) → dominant bias forcing id 7 ("4").
        ImplantSpec::new(
            "two_plus_two",
            vec![TriggerRule {
                pattern: vec![TokenPat::Exact(0), TokenPat::Exact(1), TokenPat::Exact(2)],
                action: 0,
            }],
            vec![ImplantAction {
                biases: vec![(7, 8.0)],
                residual: None,
            }],
            None,
        )
        .expect("fixture spec")
    }

    #[test]
    fn artifact_roundtrip_and_determinism() {
        let spec = spec_two_plus_two();
        let a = ImplantArtifact::mint(&spec, 1);
        let b = ImplantArtifact::mint(&spec, 1);
        assert_eq!(a, b, "same spec + epoch mints byte-identical artifacts");
        assert_ne!(a.digest, [0u8; 32]);
        let decoded = a.decode().expect("roundtrip");
        assert_eq!(decoded, spec);
    }

    #[test]
    fn canonical_refusal_on_tampered_bytes() {
        let spec = spec_two_plus_two();
        let mut a = ImplantArtifact::mint(&spec, 1);
        // Flip a payload byte AFTER mint (keep the header intact) —
        // digest mismatch fires.
        let last = a.canonical.len() - 1;
        a.canonical[last] ^= 0x80;
        assert!(matches!(
            a.decode(),
            Err(ImplantError::DigestMismatch { .. })
        ));
        // Restore the byte, then corrupt the stored digest — mismatch
        // again (verify runs before parse).
        a.canonical[last] ^= 0x80;
        a.digest[0] ^= 0xFF;
        assert!(matches!(
            a.decode(),
            Err(ImplantError::DigestMismatch { .. })
        ));
    }

    #[test]
    fn noncanonical_refusal_on_unsorted_rules() {
        // Hand-build a spec whose rules are NOT in canonical order: the
        // raw constructor sorts them, so build the artifact bytes from an
        // UNSORTED spec by encoding manually through a minted artifact of
        // the sorted spec, then swapping rule order in the payload.
        let r1 = TriggerRule {
            pattern: vec![TokenPat::Exact(9)],
            action: 0,
        };
        let r2 = TriggerRule {
            pattern: vec![TokenPat::Exact(3)],
            action: 0,
        };
        // Construct via new() would sort; build the artifact from an
        // UNSORTED clone bypassing the sort by encoding directly.
        let spec = ImplantSpec {
            name: "unsorted".into(),
            rules: vec![r1.clone(), r2.clone()],
            actions: vec![ImplantAction {
                biases: vec![(1, 1.0)],
                residual: None,
            }],
            demote_env: None,
        };
        let mut canonical = Vec::new();
        canonical.extend_from_slice(IMPLANT_MAGIC);
        canonical.extend_from_slice(&IMPLANT_ARTIFACT_FORMAT_VERSION.to_le_bytes());
        spec.encode(&mut canonical);
        let digest: [u8; 32] = blake3::hash(&canonical).into();
        let artifact = ImplantArtifact {
            format_version: IMPLANT_ARTIFACT_FORMAT_VERSION,
            epoch: 1,
            digest,
            canonical,
        };
        // The digest is honest (minted over the unsorted bytes) but the
        // re-encode check refuses: decode → sort → different bytes.
        assert_eq!(artifact.decode(), Err(ImplantError::NonCanonical));
    }

    #[test]
    fn slot_a2_law_matrix() {
        let spec = spec_two_plus_two();
        let a1 = ImplantArtifact::mint(&spec, 1);
        let a2 = ImplantArtifact::mint(&spec, 2);
        // Fork: same epoch, different content.
        let mut other_spec = spec_two_plus_two();
        other_spec.name = "fork".into();
        let fork = ImplantArtifact::mint(&other_spec, 1);

        let mut slot = ImplantSlot::new();
        assert_eq!(slot.epoch(), None);
        assert_eq!(slot.install(&a1), Ok(ImplantInstall::Installed));
        assert_eq!(slot.install(&a1), Ok(ImplantInstall::Idempotent));
        assert_eq!(
            slot.install(&fork),
            Err(ImplantError::Fork {
                mounted_epoch: 1,
                incoming_digest: fork.digest
            })
        );
        assert_eq!(slot.epoch(), Some(1), "fork refusal leaves the mount");
        assert_eq!(slot.install(&a2), Ok(ImplantInstall::Advanced { from: 1 }));
        assert_eq!(slot.epoch(), Some(2));
        // Downgrade refuses and leaves the advance in place.
        assert_eq!(
            slot.install(&a1),
            Err(ImplantError::Downgrade {
                mounted_epoch: 2,
                incoming_epoch: 1
            })
        );
        assert_eq!(slot.epoch(), Some(2));
        // Corruption refuses BEFORE lineage: a corrupt artifact at a
        // HIGHER epoch still refuses, and the mount stays.
        let mut corrupt = ImplantArtifact::mint(&spec, 3);
        let last = corrupt.canonical.len() - 1;
        corrupt.canonical[last] ^= 0x01;
        assert!(matches!(
            slot.install(&corrupt),
            Err(ImplantError::DigestMismatch { .. })
        ));
        assert_eq!(slot.epoch(), Some(2));
    }

    #[test]
    fn fire_and_silent_identity() {
        let artifact = ImplantArtifact::mint(&spec_two_plus_two(), 1);
        let mut rt = ImplantRuntime::new(1);
        rt.mount(0, &artifact).expect("mount");

        let mut buf: Vec<FireRef> = Vec::new();
        // "1 2 2 ..." — the "2 + 2" triple never completes: silent.
        for t in [5u32, 1, 2, 5, 1, 2, 6] {
            buf.clear();
            let n = rt.observe_into(t, &mut buf);
            assert_eq!(n, 0, "no fire on non-matching stream at token {t}");
        }
        assert_eq!(rt.fires(), [0]);

        // "... 0 1 2" — fires once, exactly at the third token.
        buf.clear();
        let n = rt.observe_into(0, &mut buf);
        assert_eq!(n, 0);
        buf.clear();
        let n = rt.observe_into(1, &mut buf);
        assert_eq!(n, 0);
        buf.clear();
        let n = rt.observe_into(2, &mut buf);
        assert_eq!(n, 1);
        assert_eq!(
            buf,
            [FireRef {
                implant: 0,
                action: 0
            }]
        );
        assert_eq!(rt.fires(), [1]);
        let act = rt.action(&buf[0]);
        assert_eq!(act.biases, [(7, 8.0)]);

        // Repeating the same triple fires again (first-match-wins per
        // step, one fire per implant per step).
        buf.clear();
        let n = rt.observe_into(1, &mut buf);
        assert_eq!(n, 0);
        buf.clear();
        let n = rt.observe_into(2, &mut buf);
        assert_eq!(n, 0);
        buf.clear();
        let n = rt.observe_into(0, &mut buf);
        assert_eq!(n, 0);
        buf.clear();
        let n = rt.observe_into(1, &mut buf);
        assert_eq!(n, 0);
        buf.clear();
        let n = rt.observe_into(2, &mut buf);
        assert_eq!(n, 1);
        assert_eq!(rt.fires(), [2]);
    }

    #[test]
    fn window_ring_bounds_and_long_pattern_refusal() {
        // A pattern at WINDOW_MAX matches only when the ring is full.
        let pattern: Vec<TokenPat> = (0..WINDOW_MAX as u32).map(TokenPat::Exact).collect();
        let spec = ImplantSpec::new(
            "full_window",
            vec![TriggerRule { pattern, action: 0 }],
            vec![ImplantAction {
                biases: vec![(1, 1.0)],
                residual: None,
            }],
            None,
        )
        .expect("max-window spec");
        let artifact = ImplantArtifact::mint(&spec, 1);
        let mut rt = ImplantRuntime::new(1);
        rt.mount(0, &artifact).expect("mount");

        let mut buf: Vec<FireRef> = Vec::new();
        // One short of the window: no fire yet (the first token has been
        // evicted by the time the WINDOW_MAX-th lands).
        for t in 0..(WINDOW_MAX - 1) as u32 {
            buf.clear();
            assert_eq!(rt.observe_into(t, &mut buf), 0);
        }
        buf.clear();
        assert_eq!(rt.observe_into(WINDOW_MAX as u32 - 1, &mut buf), 1);

        // Over-window patterns refuse at construction.
        let mut long: Vec<TokenPat> = (0..WINDOW_MAX as u32).map(TokenPat::Exact).collect();
        long.push(TokenPat::Exact(99));
        assert!(
            ImplantSpec::new(
                "too_long",
                vec![TriggerRule {
                    pattern: long,
                    action: 0,
                }],
                vec![ImplantAction {
                    biases: vec![(1, 1.0)],
                    residual: None,
                }],
                None,
            )
            .is_err()
        );
    }

    #[test]
    fn composition_two_implants_no_interference() {
        // Implant A: "0 1" → bias token 10. Implant B: "2 3" → bias
        // token 20. Each stays silent on the other's domain; both fire on
        // their own; interleaved streams never cross-fire.
        let spec_a = ImplantSpec::new(
            "a",
            vec![TriggerRule {
                pattern: vec![TokenPat::Exact(0), TokenPat::Exact(1)],
                action: 0,
            }],
            vec![ImplantAction {
                biases: vec![(10, 8.0)],
                residual: None,
            }],
            None,
        )
        .expect("spec a");
        let spec_b = ImplantSpec::new(
            "b",
            vec![TriggerRule {
                pattern: vec![TokenPat::Exact(2), TokenPat::Exact(3)],
                action: 0,
            }],
            vec![ImplantAction {
                biases: vec![(20, 8.0)],
                residual: None,
            }],
            None,
        )
        .expect("spec b");
        let mut rt = ImplantRuntime::new(2);
        rt.mount(0, &ImplantArtifact::mint(&spec_a, 1))
            .expect("mount a");
        rt.mount(1, &ImplantArtifact::mint(&spec_b, 1))
            .expect("mount b");

        let mut buf: Vec<FireRef> = Vec::new();
        // Stream: 0 1 2 3 — A fires at "1", B fires at "3"; A stays
        // silent on "2 3" and B on "0 1".
        buf.clear();
        assert_eq!(rt.observe_into(0, &mut buf), 0);
        buf.clear();
        assert_eq!(rt.observe_into(1, &mut buf), 1);
        assert_eq!(
            buf,
            [FireRef {
                implant: 0,
                action: 0
            }]
        );
        buf.clear();
        assert_eq!(rt.observe_into(2, &mut buf), 0);
        buf.clear();
        assert_eq!(rt.observe_into(3, &mut buf), 1);
        assert_eq!(
            buf,
            [FireRef {
                implant: 1,
                action: 0
            }]
        );
        assert_eq!(rt.fires(), [1, 1]);

        // The G3 interleaving hazard: "2 1" (B's head, A's tail) must not
        // fire either — patterns are anchored to the window tail.
        buf.clear();
        assert_eq!(rt.observe_into(2, &mut buf), 0);
        buf.clear();
        assert_eq!(
            rt.observe_into(1, &mut buf),
            0,
            "cross-implant look-alike must not fire"
        );
        assert_eq!(rt.fires(), [1, 1]);
    }

    #[test]
    fn anyof_canonical_and_first_match_wins() {
        let pat = any_of(&[5, 3, 5, 1]);
        assert_eq!(pat, TokenPat::AnyOf(vec![1, 3, 5]), "sorted + deduped");

        // Two rules, both match the same window: the canonical-first rule
        // (lower pattern rank) wins.
        let r_later = TriggerRule {
            pattern: vec![TokenPat::Exact(4)],
            action: 1,
        };
        let r_earlier = TriggerRule {
            pattern: vec![TokenPat::Exact(4)],
            action: 0,
        };
        let spec = ImplantSpec::new(
            "tiebreak",
            vec![r_later, r_earlier],
            vec![
                ImplantAction {
                    biases: vec![(11, 1.0)],
                    residual: None,
                },
                ImplantAction {
                    biases: vec![(12, 1.0)],
                    residual: None,
                },
            ],
            None,
        )
        .expect("tiebreak spec");
        assert_eq!(
            spec.rules[0].action, 0,
            "canonical order sorts equal patterns by action"
        );

        let artifact = ImplantArtifact::mint(&spec, 1);
        let mut rt = ImplantRuntime::new(1);
        rt.mount(0, &artifact).expect("mount");
        let mut buf: Vec<FireRef> = Vec::new();
        rt.observe_into(4, &mut buf);
        assert_eq!(
            buf,
            [FireRef {
                implant: 0,
                action: 0
            }]
        );
    }

    #[test]
    fn demote_env_disarms_implant_but_window_advances() {
        // demote_env set to "0" in-process is not testable portably
        // (env mutation races), so this fixture exercises the demoted
        // path directly through the runtime's internal posture.
        let spec = spec_two_plus_two();
        let mut rt = ImplantRuntime::new(1);
        let artifact = ImplantArtifact::mint(&spec, 1);
        rt.mount(0, &artifact).expect("mount");
        rt.demoted[0] = true; // the mount-time env arm's effect
        let mut buf: Vec<FireRef> = Vec::new();
        for t in [0u32, 1, 2] {
            buf.clear();
            assert_eq!(
                rt.observe_into(t, &mut buf),
                0,
                "demoted implants never fire"
            );
        }
        // The window still advanced — a later undemote (re-mount at a
        // higher epoch) sees the real stream tail.
        assert_eq!(rt.window_len, 3);
        assert_eq!(rt.fires(), [0]);
        assert!(rt.is_demoted(0));
    }
}
