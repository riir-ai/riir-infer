//! `implants_cert` — the Issue-1256 T1 proof-path upgrade (Research 046
//! F1, the Sketchpad distill): the fire gate's **view-enumeration
//! certificate**. The gate is a pure function of a bounded token window
//! (patterns ≤ [`WINDOW_MAX`], refused beyond at mount), so its
//! decision surface is FINITE and exhaustively certifiable:
//!
//! - **The fire set** is the product of per-slot member expansions
//!   (`Exact(t)` → `{t}`, `AnyOf(set)` → the set) — every view the gate
//!   ever fires on, exactly.
//! - **The boundary shell** is every single-token perturbation of every
//!   fire-set member (substitution, insertion, deletion over the
//!   closure alphabet), enumerated EXHAUSTIVELY — the entire
//!   decision boundary at radius 1. The G1b false-fire bar upgrades
//!   from sampled look-alikes to this shell: no shell view outside the
//!   fire set fires, per-implant.
//! - **Margins** are exact integer Levenshtein distances to the nearest
//!   fire-set member — device-independent by construction (the token
//!   channel carries no perturbation; the Sketchpad τ discipline applies
//!   to the F2 logit-margin form, not this one — recorded as a
//!   disclosure, never claimed as the same certificate class).
//! - **The certificate artifact** is canonical bytes + BLAKE3 bound to
//!   the implant artifacts' digests (the mount law's hash pinning,
//!   extended with the per-view margin table) + the weakest-tenth rows
//!   (lowest margins — the recheck list a new posture reads first).
//!
//! Refusal law (the 046 boundary): a gate reading UNBOUNDED history is
//! not enumerable. The bounded-window mount already refuses long
//! patterns; this module re-asserts the bound and refuses a sentinel
//! collision — the certificate never quietly falls back to sampling.

use crate::implants::{ImplantSpec, TokenPat, WINDOW_MAX};
use std::collections::BTreeSet;

/// Certificate format version.
pub const IMPLANT_CERT_FORMAT_VERSION: u32 = 1;

/// Magic prefix of the canonical certificate encoding.
const CERT_MAGIC: &[u8] = b"RICRT1\x01";

/// The certificate's per-view row.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct CertView {
    /// The window-tail candidate (oldest→newest).
    pub tokens: Vec<u32>,
    /// The decision: `None` = silent, `Some((implant, action))` = the
    /// fired ref (first-match-wins per implant, mount order — the
    /// runtime's own gate order).
    pub decision: Option<(u16, u16)>,
    /// The margin: exact Levenshtein distance to the nearest fire-set
    /// member (0 = the view IS a fire-set member).
    pub margin: u32,
    /// The view's origin class (the audit column, never a decision).
    pub origin: CertOrigin,
}

/// Where an enumerated view came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CertOrigin {
    /// A fire-set member.
    FireSet,
    /// A single-token perturbation of a fire-set member (the boundary
    /// shell — the exhaustive G1b surface).
    Shell,
    /// A caller-supplied authored view (the look-alike/prose rows).
    Authored,
}

/// The view-enumeration certificate over one mounted spec set.
#[derive(Debug, Clone)]
pub struct ImplantCert {
    /// The certifying specs in mount order (decisions are resolved
    /// against exactly these).
    pub specs: Vec<SpecRef>,
    /// The per-view rows, canonical order (sorted by tokens then origin).
    pub views: Vec<CertView>,
    /// The closure alphabet (sorted, deduped): every id named in any
    /// pattern — the perturbation alphabet. The sentinel (never a
    /// member) is [`ImplantCert::sentinel`].
    pub alphabet: Vec<u32>,
    /// The perturbation sentinel id (`max(alphabet) + 1`); it matches no
    /// pattern slot and no AnyOf set. `u32::MAX` alphabet refuses loud.
    pub sentinel: u32,
    /// The BLAKE3 over the canonical encoding (set by `canonicalize`).
    pub digest: [u8; 32],
}

/// The per-spec bind (the mount law's digest pinning).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct SpecRef {
    /// The implant's mount slot.
    pub slot: u16,
    /// The spec's name (telemetry identity in the cert).
    pub name: String,
    /// The implant artifact's content digest (BLAKE3 over canonical
    /// bytes — the pin that binds this cert to mounted bytes).
    pub digest: [u8; 32],
}

impl ImplantCert {
    /// The perturbation sentinel.
    #[must_use]
    pub fn sentinel(&self) -> u32 {
        self.sentinel
    }

    /// The weakest-tenth recheck list **over the SILENT surface**: the
    /// lowest-margin non-firing views, capped at one tenth of the
    /// census (minimum 1) — the rows a new posture rechecks first (the
    /// Sketchpad weakest-tenth discipline; here the margins are exact
    /// integers, so the recheck is a re-read, not a re-measurement).
    /// Firing rows (margin 0) are excluded — they are the certified
    /// surface, not the risk surface.
    #[must_use]
    pub fn weakest_tenth(&self) -> Vec<&CertView> {
        let mut rows: Vec<&CertView> = self.views.iter().filter(|v| v.decision.is_none()).collect();
        rows.sort_by_key(|v| (v.margin, v.tokens.clone()));
        let cap = (rows.len() / 10).max(1);
        rows.into_iter().take(cap).collect()
    }

    /// The census verdict counters.
    #[must_use]
    pub fn census(&self) -> CertCensus {
        let mut c = CertCensus::default();
        for v in &self.views {
            match v.decision {
                Some(_) => c.firing += 1,
                None if v.margin >= 2 => c.certified_silent += 1,
                None => c.boundary_silent += 1,
            }
            if v.origin == CertOrigin::Shell {
                c.shell += 1;
            }
            if v.origin == CertOrigin::Authored {
                c.authored += 1;
            }
        }
        c
    }

    /// Canonical bytes + digest (deterministic: sorted views, sorted
    /// alphabet, fixed-width numeric fields).
    ///
    /// # Errors
    /// [`ImplantError::Usage`]-shaped text on an oversized field (the
    /// cert is offline — a length past u32 refuses loud).
    pub fn canonicalize(&mut self) -> Result<(), String> {
        self.views
            .sort_by(|a, b| a.tokens.cmp(&b.tokens).then(a.origin.cmp(&b.origin)));
        let mut out = Vec::new();
        out.extend_from_slice(CERT_MAGIC);
        out.extend_from_slice(&IMPLANT_CERT_FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&(self.specs.len() as u32).to_le_bytes());
        for s in &self.specs {
            out.extend_from_slice(&s.slot.to_le_bytes());
            out.extend_from_slice(&(s.name.len() as u32).to_le_bytes());
            out.extend_from_slice(s.name.as_bytes());
            out.extend_from_slice(&s.digest);
        }
        out.extend_from_slice(&self.sentinel.to_le_bytes());
        out.extend_from_slice(&(self.alphabet.len() as u32).to_le_bytes());
        for &a in &self.alphabet {
            out.extend_from_slice(&a.to_le_bytes());
        }
        if self.views.len() > u32::MAX as usize {
            return Err("cert view census over u32".to_string());
        }
        out.extend_from_slice(&(self.views.len() as u32).to_le_bytes());
        for v in &self.views {
            if v.tokens.len() > u32::MAX as usize {
                return Err("cert view length over u32".to_string());
            }
            out.extend_from_slice(&(v.tokens.len() as u32).to_le_bytes());
            for &t in &v.tokens {
                out.extend_from_slice(&t.to_le_bytes());
            }
            match v.decision {
                None => out.push(0),
                Some((i, a)) => {
                    out.push(1);
                    out.extend_from_slice(&i.to_le_bytes());
                    out.extend_from_slice(&a.to_le_bytes());
                }
            }
            out.extend_from_slice(&v.margin.to_le_bytes());
            out.push(match v.origin {
                CertOrigin::FireSet => 0,
                CertOrigin::Shell => 1,
                CertOrigin::Authored => 2,
            });
        }
        self.digest = blake3::hash(&out).into();
        Ok(())
    }
}

/// The census counters (typed, serializable).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct CertCensus {
    /// Views the gate fires on.
    pub firing: usize,
    /// Silent views at margin ≥ 2 (silence survives any single-token
    /// perturbation).
    pub certified_silent: usize,
    /// Silent views at margin 1 (exact-channel-safe only — one
    /// substitution from firing; recorded honestly, never folded).
    pub boundary_silent: usize,
    /// Shell-origin rows (the exhaustive boundary).
    pub shell: usize,
    /// Authored-origin rows.
    pub authored: usize,
}

/// The enumerate result: the cert + the assertion ledger.
#[derive(Debug)]
pub struct Enumeration {
    /// The certificate (digest set by [`ImplantCert::canonicalize`]).
    pub cert: ImplantCert,
    /// Fire-set members that did NOT fire — must be empty for a
    /// certificate to issue (the exhaustive G2 gate half).
    pub missed_fires: Vec<CertView>,
    /// AUTHORED views that fired — must be empty (the intent surface;
    /// the sampled G1b upgrade).
    pub authored_fires: Vec<CertView>,
    /// Views whose decision disagrees with the independent suffix
    /// predicate — must be empty (a broken slot predicate cannot
    /// certify itself).
    pub structural: Vec<CertView>,
}

impl Enumeration {
    /// The shell views that fire legally (suffix preservation or a
    /// perturbation landing on another tabled pattern) — the disclosed
    /// suffix-anchor surface, never a refusal.
    #[must_use]
    pub fn shell_fires(&self) -> usize {
        self.cert
            .views
            .iter()
            .filter(|v| v.origin == CertOrigin::Shell && v.decision.is_some())
            .count()
    }

    /// **The runtime agreement check (the exhaustive G1b/G2 upgrade on
    /// LIVE code):** every enumerated view is driven through the REAL
    /// gate via `observe` (the caller feeds tokens and returns the fire
    /// refs of the last step — a fresh runtime per view), and every
    /// decision must match the cert row. Returns the mismatch count
    /// (0 = the runtime agrees with the certificate over the whole
    /// enumerated surface).
    #[must_use]
    pub fn verify_against_runtime<F>(&self, mut observe: F) -> usize
    where
        F: FnMut(&[u32]) -> Vec<(u16, u16)>,
    {
        let mut mismatches = 0usize;
        for v in &self.cert.views {
            let fired = observe(&v.tokens);
            let runtime_decision = fired.first().copied();
            match (v.decision, runtime_decision) {
                (None, None) => {}
                (Some(d), Some(r)) if d == r => {}
                _ => mismatches += 1,
            }
        }
        mismatches
    }
}

/// Enumerate the certificate over `specs` (mount order) + the authored
/// views (token sequences; the look-alike/prose rows). The fire decision
/// replicates the runtime's gate exactly: first-match-wins per implant,
/// canonical rule order, window tail of the view's own length.
///
/// # Errors
/// Text refusals (the 046 law — never a silent fallback): a pattern
/// over [`WINDOW_MAX`], a `u32::MAX` alphabet member (the sentinel
/// collides), or an over-census shell.
pub fn enumerate_views(
    specs: &[SpecRef],
    specs_impl: &[&ImplantSpec],
    authored: &[Vec<u32>],
) -> Result<Enumeration, String> {
    if specs.len() != specs_impl.len() {
        return Err("spec refs and specs length mismatch".to_string());
    }
    // ── Refusal law: every pattern within the bounded window ──
    for s in specs_impl {
        for rule in &s.rules {
            if rule.pattern.len() > WINDOW_MAX {
                return Err(format!(
                    "implant {}: rule pattern {} over WINDOW_MAX {WINDOW_MAX} — unbounded-history gates are not certifiable",
                    s.name,
                    rule.pattern.len()
                ));
            }
        }
    }
    // ── The closure alphabet + the sentinel ──
    let mut alpha: BTreeSet<u32> = BTreeSet::new();
    for s in specs_impl {
        for rule in &s.rules {
            for pat in &rule.pattern {
                match pat {
                    TokenPat::Exact(t) => {
                        alpha.insert(*t);
                    }
                    TokenPat::AnyOf(set) => {
                        for t in set {
                            alpha.insert(*t);
                        }
                    }
                }
            }
        }
    }
    let sentinel = alpha
        .last()
        .copied()
        .unwrap_or(0)
        .checked_add(1)
        .ok_or("closure alphabet hits u32::MAX — the sentinel collides")?;
    let alphabet: Vec<u32> = alpha.into_iter().collect();

    // ── The fire set (per-rule slot expansion, deduped) ──
    let mut fire_set: BTreeSet<Vec<u32>> = BTreeSet::new();
    for s in specs_impl {
        for rule in &s.rules {
            let mut expansions: Vec<Vec<u32>> = vec![Vec::new()];
            for pat in &rule.pattern {
                let members: Vec<u32> = match pat {
                    TokenPat::Exact(t) => vec![*t],
                    TokenPat::AnyOf(set) => set.clone(),
                };
                let mut next = Vec::with_capacity(expansions.len() * members.len());
                for prefix in &expansions {
                    for m in &members {
                        let mut v = prefix.clone();
                        v.push(*m);
                        next.push(v);
                    }
                }
                expansions = next;
            }
            for v in expansions {
                fire_set.insert(v);
            }
        }
    }

    // ── The gate decision (the runtime's exact order) ──
    let decision = |view: &[u32]| -> Option<(u16, u16)> {
        for (i, s) in specs_impl.iter().enumerate() {
            for rule in &s.rules {
                let plen = rule.pattern.len();
                if plen > view.len() {
                    continue;
                }
                let tail = &view[view.len() - plen..];
                if rule
                    .pattern
                    .iter()
                    .zip(tail.iter())
                    .all(|(pat, &t)| pat.matches(t))
                {
                    return Some((specs[i].slot, rule.action));
                }
            }
        }
        None
    };
    // The INDEPENDENT replication of the suffix predicate (explicit
    // membership, never `TokenPat::matches`) — the two-implementation
    // agreement the certificate asserts (a broken slot predicate cannot
    // certify itself).
    let independent = |view: &[u32]| -> Option<(u16, u16)> {
        for (i, s) in specs_impl.iter().enumerate() {
            for rule in &s.rules {
                let plen = rule.pattern.len();
                if plen > view.len() {
                    continue;
                }
                let tail = &view[view.len() - plen..];
                let ok = rule
                    .pattern
                    .iter()
                    .zip(tail.iter())
                    .all(|(pat, &t)| match pat {
                        TokenPat::Exact(x) => *x == t,
                        TokenPat::AnyOf(set) => set.contains(&t),
                    });
                if ok {
                    return Some((specs[i].slot, rule.action));
                }
            }
        }
        None
    };

    // ── Rows ──
    // The gate is SUFFIX-anchored by design (patterns anchor to the
    // window tail): a shell view whose firing suffix is a complete
    // pattern fires LEGALLY — front insertions preserve the match and a
    // substitution can land the view ON another tabled pattern. Those
    // rows are recorded (`shell_fires`, disclosed), never refused. The
    // refusal classes are: a fire-set member that does not fire (the
    // exhaustive G2 gate half), an AUTHORED view that fires (the intent
    // surface — the sampled G1b upgrade), and a decision that
    // disagrees with the independent suffix predicate (a broken slot
    // predicate cannot certify itself).
    let mut views: Vec<CertView> = Vec::new();
    let mut missed_fires: Vec<CertView> = Vec::new();
    let mut authored_fires: Vec<CertView> = Vec::new();
    let mut structural: Vec<CertView> = Vec::new();

    let push_row = |tokens: Vec<u32>, origin: CertOrigin| -> CertView {
        let d = decision(&tokens);
        let margin = fire_set
            .iter()
            .map(|f| levenshtein(&tokens, f))
            .min()
            .unwrap_or(u32::MAX);
        CertView {
            tokens,
            decision: d,
            margin,
            origin,
        }
    };

    // Fire-set members: every one MUST fire (exhaustive G2 gate half).
    let fire_members: Vec<Vec<u32>> = fire_set.iter().cloned().collect();
    for tokens in &fire_members {
        let row = push_row(tokens.clone(), CertOrigin::FireSet);
        if row.decision.is_none() {
            missed_fires.push(row);
        } else {
            views.push(row);
        }
    }

    // The boundary shell: every single-token perturbation of every
    // fire-set member (substitution over the closure alphabet + the
    // sentinel, insertion at every gap, deletion) — deduped, fire-set
    // members excluded (already rowed above).
    let mut shell: BTreeSet<Vec<u32>> = BTreeSet::new();
    let mut perturb_alphabet = alphabet.clone();
    perturb_alphabet.push(sentinel);
    for tokens in &fire_members {
        // Substitutions.
        for pos in 0..tokens.len() {
            for &a in &perturb_alphabet {
                if a != tokens[pos] {
                    let mut v = tokens.clone();
                    v[pos] = a;
                    shell.insert(v);
                }
            }
        }
        // Insertions.
        for pos in 0..=tokens.len() {
            for &a in &perturb_alphabet {
                let mut v = Vec::with_capacity(tokens.len() + 1);
                v.extend_from_slice(&tokens[..pos]);
                v.push(a);
                v.extend_from_slice(&tokens[pos..]);
                shell.insert(v);
            }
        }
        // Deletions.
        if tokens.len() > 1 {
            for pos in 0..tokens.len() {
                let mut v = Vec::with_capacity(tokens.len() - 1);
                v.extend_from_slice(&tokens[..pos]);
                v.extend_from_slice(&tokens[pos + 1..]);
                shell.insert(v);
            }
        }
    }
    for tokens in shell {
        if fire_set.contains(&tokens) {
            continue;
        }
        let row = push_row(tokens, CertOrigin::Shell);
        if row.decision != independent(&row.tokens) {
            structural.push(row);
        } else {
            views.push(row);
        }
    }

    // The authored views (the look-alike/prose rows — the INTENT
    // surface): any fire here is a false fire by authored intent — the
    // certificate refuses.
    for tokens in authored {
        if fire_set.contains(tokens) {
            continue;
        }
        let row = push_row(tokens.clone(), CertOrigin::Authored);
        if row.decision.is_some() {
            authored_fires.push(row);
        } else if row.decision != independent(&row.tokens) {
            structural.push(row);
        } else {
            views.push(row);
        }
    }

    if !missed_fires.is_empty() || !authored_fires.is_empty() || !structural.is_empty() {
        return Err(format!(
            "certificate REFUSED: {} missed fire(s), {} authored-intent fire(s), {} structural disagreement(s)",
            missed_fires.len(),
            authored_fires.len(),
            structural.len()
        ));
    }

    Ok(Enumeration {
        cert: ImplantCert {
            specs: specs.to_vec(),
            views,
            alphabet,
            sentinel,
            digest: [0; 32],
        },
        missed_fires,
        authored_fires,
        structural,
    })
}

/// Exact Levenshtein distance (token sequences; bounded by WINDOW_MAX+1
/// lengths — the cert is offline, the O(n·m) table is ≤ 17×17).
#[must_use]
fn levenshtein(a: &[u32], b: &[u32]) -> u32 {
    let (n, m) = (a.len(), b.len());
    let mut prev: Vec<u32> = (0..=m as u32).collect();
    let mut cur = vec![0_u32; m + 1];
    for i in 1..=n {
        cur[0] = i as u32;
        for j in 1..=m {
            let cost = u32::from(a[i - 1] != b[j - 1]);
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[m]
}
