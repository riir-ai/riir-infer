//! Issue 1201 T2 — the Metal twin of `qwen38_prefix_cache` (742 T4): the
//! single-stream prefix state cache over the wgpu/Metal forward, same
//! CONTRACT (insert at a filled anchor / restore the longest exact prefix /
//! `note_write` lineage prune / `len` / `pruned`), different payload.
//!
//! # The disclosed divergence (the honest half of the port)
//!
//! 742 T4 snapshots GDN-ONLY and relies on the live stream's KV rows
//! `[0..P)` surviving between requests (`note_write` prunes a checkpoint
//! whose KV tail a younger write could clobber). This lane has no
//! live-stream shape to rely on: the Metal forward is weights-bound (one
//! instance serves every session), so a "session stream" is expressed as
//! checkpoint-restore, and the snapshot carries the FULL state — GDN S,
//! conv windows, and the KV window — via
//! [`Qwen38WgpuForward::checkpoint_state`]. Consequences, all intended:
//!
//! - restore is complete (GDN + KV) — no `note_write` hazard can corrupt
//!   a restore, because the restored bytes are the anchor-time bytes, not
//!   live-cache survival; the prune stays as the contract floor (T3's
//!   retention consumes it) and because stale checkpoints still cost
//!   memory.
//! - insert/restore move the whole KV window (readback + upload of
//!   `max_ctx × kvd` per attention side) — this lane's restore cost is
//!   strictly heavier than the CUDA lane's. Plan 630 scopes A0+ as the
//!   control reported beside, never a TTFT-bar denominator; the port
//!   exists so the control EXISTS on Metal, not to win bars.
//! - the stale tail beyond a checkpoint's fill is never read (causal
//!   attention reads `[0..pos)`), so whole-buffer snapshots are sound for
//!   the growing-context trace.
//!
//! # Match discipline
//!
//! Same as 742 T4: blake3-class fast filter replaced by a u64 FNV-1a fold
//! (no new dep on this lane) + EXACT token compare — the compare is the
//! authority, the filter only skips. Longest exact prefix wins; entries
//! are per-anchor (`insert` appends; T3's retention policy later owns
//! eviction — this module never evicts on its own).
//!
//! Posture: feature `qwen38_wgpu` (same family as the forward it
//! checkpoints). The harness composition (per-session instances over the
//! plan-630 trace) is the bin's A0+ arm.

use std::sync::Arc;

use crate::qwen38_dense_cubecl::{Qwen38WgpuForward, StateCheckpoint};

/// One whole-prefix checkpoint: the token prefix, its u64 filter key, and
/// the full state snapshot at the anchor.
struct PrefixEntry {
    len: usize,
    key: u64,
    tokens: Arc<[u32]>,
    cp: StateCheckpoint,
}

fn filter_key(tokens: &[u32]) -> u64 {
    // FNV-1a over the LE token bytes — a skip filter only; the exact
    // compare below is the match authority.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for t in tokens {
        h ^= u64::from(*t);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The per-session single-stream prefix cache (the A0+ arm's substrate on
/// Metal — the port of `Qwen38PrefixCache`'s role, not its cudarc
/// machinery: 742 T4's pointer tables / table-copy kernel / arena pool are
/// driver-submit optimizations for the CUDA dtod path; this lane's copies
/// are unified-memory uploads through cubecl, where those micro-lanes
/// have no counterpart).
#[derive(Default)]
pub struct PrefixCacheCubecl {
    entries: Vec<PrefixEntry>,
    pruned: usize,
}

impl PrefixCacheCubecl {
    pub fn new() -> Self {
        Self::default()
    }

    /// Entries currently held.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Checkpoints dropped by [`Self::note_write`] since construction.
    pub fn pruned(&self) -> usize {
        self.pruned
    }

    /// Anchor a checkpoint: the caller must have filled EXACTLY `tokens`
    /// into `fwd` (in order, from the state the last restore left) — the
    /// snapshot is taken now, at the anchor. Anchors at every turn end is
    /// the cell's policy; caller-chosen anchors elsewhere.
    pub fn insert(&mut self, fwd: &Qwen38WgpuForward, tokens: &[u32]) {
        assert!(!tokens.is_empty(), "prefix cache: empty anchor");
        let cp = fwd.checkpoint_state();
        self.entries.push(PrefixEntry {
            len: tokens.len(),
            key: filter_key(tokens),
            tokens: Arc::from(tokens.to_vec().into_boxed_slice()),
            cp,
        });
    }

    /// Restore the longest exact prefix of `tokens` among the anchors and
    /// leave it in `fwd` (state bytes written back; nothing else is
    /// touched — the caller drives from the returned length). Returns the
    /// matched prefix length (0 = no anchor matched; the caller prefills
    /// from zero over whatever state `fwd` currently holds, which must be
    /// caller-reset — a fresh stream).
    pub fn restore(&self, fwd: &mut Qwen38WgpuForward, tokens: &[u32]) -> usize {
        let mut best: Option<&PrefixEntry> = None;
        for e in &self.entries {
            if e.len > tokens.len() || e.key != filter_key(&tokens[..e.len]) {
                continue;
            }
            if tokens[..e.len] == e.tokens[..] {
                let better = match best {
                    None => true,
                    Some(b) => e.len > b.len,
                };
                if better {
                    best = Some(e);
                }
            }
        }
        match best {
            None => 0,
            Some(e) => {
                fwd.restore_state(&e.cp);
                e.len
            }
        }
    }

    /// The lineage prune (the contract floor): drop every checkpoint
    /// whose KV tail a write reaching `pos` could have clobbered
    /// (`len > pos`). Under the cell's monotone growing-context drive no
    /// entry is ever pruned; the rule exists for callers whose writes
    /// restart lower.
    pub fn note_write(&mut self, pos: usize) {
        let before = self.entries.len();
        self.entries.retain(|e| e.len <= pos);
        self.pruned += before - self.entries.len();
    }
}
