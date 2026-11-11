//! Plan 630 / Issue 1201 T2 — the FlashStates multi-session manager: the
//! radix-indexed shared state pool for concurrent hybrid (GDN+attention)
//! sessions, consuming the NAMED substrate end to end — never a private
//! tree, never a private refcount protocol:
//!
//! - [`katgpt_kv::radix_prefix::RadixPrefixTree`] — the prefix index
//!   (Bench 762: 2.45× the flat shape's hit-rate at equal budget; the
//!   742 T4 substrate note forbids re-deriving the tree).
//! - [`katgpt_transformer::PagedKVCache`] — the KV-row page pool for the
//!   ATTENTION layers: fresh pages via `ensure_pages`, capture writes via
//!   `write_kv`, holds via `retain_chunk_pages`, tree-eviction drops via
//!   `release_chunk_pages`.
//! - [`GdnStatePool`] (this module) — the GDN recurrent+conv state pages.
//!   A pool, NOT a tree: the page shape (per-layer fixed-size recurrence
//!   state) is one `PagedKVCache` cannot hold, and the 742-T4 note forbids
//!   re-deriving the TREE, not pool mechanics. Append-only allocation
//!   (the address-stability contract the paged cache documents for
//!   captured graphs), identical refcount discipline.
//!
//! # The refcount ledger (the substrate's own, applied to N sessions)
//!
//! The substrate's radix tests fix the discipline: one retain per fresh
//! capture IS the tree's hold (insert's contract); one retain per
//! adoption IS the adopting session's hold; tree eviction releases the
//! tree's. Applied here:
//!
//! - fresh capture chunk: `retain ×1` (tree) — the creating session
//!   takes NO pool hold on its own fresh pages (its live buffers carry
//!   the data; pool-side it is tree-only).
//! - adopted chunk (a session serving a prefix another session indexed):
//!   `retain ×1` (the adopting session).
//! - session end: release ONLY the session's adopted holds — its fresh
//!   captures stay tree-held and keep serving other sessions.
//! - tree eviction: release the tree's hold. A page frees exactly when
//!   its last holder drops. Eviction can forget a prefix a live session
//!   still adopts (leaf-first): the page survives on the session's hold,
//!   the tree loses shareability for it, and the next capture re-indexes
//!   it (a tree-hold retain on an already-adopted page).
//!
//! # The concurrent-fill recovery (the interleaved trace's base shape)
//!
//! Two sessions capturing the same shared prefix between one another's
//! serve and commit each hold their OWN fresh pages for those chunks —
//! the substrate's insert refuses the second committer (`PathMismatch`).
//! The manager recovers the first conflict automatically (first writer
//! wins): the loser DISCARDS its own fresh pages for the tree's matched
//! span, adopts the winner's pages (the session hold), and retries the
//! insert once — its genuinely-divergent suffix then indexes as new
//! chunks. A second failure propagates (real corruption). This is the
//! vLLM/FlashStates first-writer-wins dedup shape, expressed through the
//! substrate's refusal instead of a private one. Disclosed assumption:
//! adopting another session's pages for the same token prefix is
//! byte-correct — same model, deterministic capture (the cell's G1
//! contract pins it).
//!
//! # The session bridge (graph safety by construction)
//!
//! The manager is DEVICE-FREE: a serve returns a [`ResumePlan`] of pool
//! ids, and the harness copies the referenced pages INTO the session's
//! existing live buffers (data move, never a buffer re-home) before the
//! forward runs as-shipped. No captured graph address changes. The KV
//! pool's per-sequence tables are used ONLY as the fresh-page allocation
//! vehicle (`ensure_pages` into a dedicated per-capture slot — never
//! reused, never rolled back; the refcounts are the sole lifetime
//! authority, and the slot's table entries are write-only bookkeeping
//! after the capture). `prompt_tokens - 1` is the ensure arity so a
//! chunk-aligned capture allocates exactly `chunks` pages (the +1
//! decode slack of `pos / PAGE_SIZE + 1` would leak one page per batch).
//!
//! # Posture
//!
//! Feature `flashstates_multi` (opt-in; katgpt-kv's `radix_prefix_cache`
//! rides by name; BOUNDARY lists katgpt-kv as measurement-lane
//! capability, never a default-feature dep). CPU-testable end to end —
//! no device anywhere; the GPU harness composition (real forwards) is
//! the plan-630 T2 remainder along with the A0 Metal port.

use std::collections::HashMap;

use katgpt_kv::radix_prefix::{InsertOutcome, RadixPrefixTree, TreeError};
use katgpt_transformer::PagedKVCache;

/// Page tokens per chunk — the tree's `page_tokens` and the KV pool's
/// `PAGE_SIZE` must agree; the manager asserts it at construction.
pub const PAGE_TOKENS: usize = katgpt_transformer::PAGE_SIZE;

/// The hybrid layer plan: how the manager routes a chunk-major page id
/// to its pool. Attention layers own the first `n_attn` slots (their
/// pages live in the `PagedKVCache`), the GDN layers own the rest
/// (their pages live in the [`GdnStatePool`]). The per-layer fixed
/// state shapes are caller-supplied — the manager is shape-agnostic.
#[derive(Clone, Debug)]
pub struct LayerPlan {
    pub n_attn: usize,
    pub n_gdn: usize,
    /// f32 elements of one GDN layer's recurrent state page.
    pub gdn_recurrent_len: usize,
    /// f32 elements of one GDN layer's conv-window page.
    pub gdn_conv_len: usize,
}

impl LayerPlan {
    /// Total layers per chunk — the tree's `n_layers`.
    pub fn n_layers(&self) -> usize {
        self.n_attn + self.n_gdn
    }
}

/// One GDN layer's captured state at a chunk boundary: the recurrent
/// matrix page + the conv-window page (742 T4's snapshot shape, cut to
/// one layer so a chunk's pages index per-layer like the KV rows do).
#[derive(Clone, Debug, PartialEq)]
pub struct GdnStatePage {
    pub recurrent: Vec<f32>,
    pub conv: Vec<f32>,
}

/// The GDN state pool: refcounted, append-only (a page id never moves or
/// is reordered — the paged cache's address-stability contract), free-list
/// reuse on the zero-ref path. The pool NEVER interprets state bytes.
#[derive(Default)]
pub struct GdnStatePool {
    pages: Vec<GdnStatePage>,
    refcounts: Vec<usize>,
    free: Vec<usize>,
}

impl GdnStatePool {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.pages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }

    /// Pages currently held (allocated minus free-listed).
    pub fn held(&self) -> usize {
        self.pages.len() - self.free.len()
    }

    /// Allocate a page pre-shaped to the plan (zeroed). Refcount starts
    /// at 0 — every hold arrives through [`Self::retain`].
    pub fn alloc(&mut self, plan: &LayerPlan) -> usize {
        let page = GdnStatePage {
            recurrent: vec![0.0; plan.gdn_recurrent_len],
            conv: vec![0.0; plan.gdn_conv_len],
        };
        if let Some(id) = self.free.pop() {
            self.pages[id] = page;
            id
        } else {
            self.pages.push(page);
            self.refcounts.push(0);
            self.pages.len() - 1
        }
    }

    pub fn retain(&mut self, ids: &[usize]) {
        for &id in ids {
            self.refcounts[id] += 1;
        }
    }

    /// Drop one hold per id; a page reaching zero returns to the free
    /// list (its bytes are dropped — by the ledger, no tree node and no
    /// session references it anymore).
    pub fn release(&mut self, ids: &[usize]) {
        for &id in ids {
            let rc = &mut self.refcounts[id];
            assert!(*rc > 0, "gdn state pool: release without hold (id {id})");
            *rc -= 1;
            if *rc == 0 {
                self.free.push(id);
            }
        }
    }

    /// Return NEVER-HELD pages (refcount 0 — the concurrent-fill
    /// recovery's discarded fresh allocs) to the free list. A held page
    /// through this path is a ledger bug and asserts.
    pub fn discard(&mut self, ids: &[usize]) {
        for &id in ids {
            assert_eq!(
                self.refcounts[id], 0,
                "gdn state pool: discard of held page {id}"
            );
            self.free.push(id);
        }
    }

    /// Read a page's held state (a live page — some holder guarantees
    /// reachability).
    pub fn page(&self, id: usize) -> &GdnStatePage {
        assert!(
            self.refcounts[id] > 0,
            "gdn state pool: read of unheld page {id}"
        );
        &self.pages[id]
    }

    /// Write a page's state (capture path — the creator's fresh alloc).
    pub fn write(&mut self, id: usize, recurrent: &[f32], conv: &[f32]) {
        let page = &mut self.pages[id];
        assert_eq!(page.recurrent.len(), recurrent.len(), "recurrent len");
        assert_eq!(page.conv.len(), conv.len(), "conv len");
        page.recurrent.copy_from_slice(recurrent);
        page.conv.copy_from_slice(conv);
    }

    pub fn refcount(&self, id: usize) -> usize {
        self.refcounts[id]
    }
}

/// Per-session bookkeeping. `tables` is the session's full chunk-major
/// per-layer page layout (its own prefix) — the commit supply and the
/// restore source. `adopted_kv`/`adopted_gdn` are ONLY the holds the
/// session took at serve/recovery time (never its fresh captures) — the
/// session-end release set.
#[derive(Clone, Debug)]
struct SessionRecord {
    tables: Vec<Vec<usize>>,
    held_chunks: usize,
    adopted_kv: Vec<Vec<usize>>,
    adopted_gdn: Vec<usize>,
}

/// What a serve hands the harness: which chunks of the prompt are
/// pool-resolvable and which pool pages carry them, per layer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResumePlan {
    /// Whole chunks resolvable for this session (its held prefix, plus
    /// tree-matched chunks beyond it).
    pub restore_chunks: usize,
    /// `[attn_layer][chunk] -> kv pool id` for chunks [0..restore_chunks).
    pub kv_pages: Vec<Vec<usize>>,
    /// `[gdn_layer][chunk] -> state pool id` for chunks [0..restore_chunks).
    pub gdn_pages: Vec<Vec<usize>>,
}

impl ResumePlan {
    /// Tokens the harness need not prefill (restore_chunks × page_tokens).
    pub fn restored_tokens(&self) -> usize {
        self.restore_chunks * PAGE_TOKENS
    }
}

/// The capture half's fresh-page grant: the ids the caller writes the
/// new chunks' states into, per layer. KV pages are written through
/// `PagedKVCache::write_kv(layer, capture_slot, pos, k, v)` with
/// `capture_slot` from this plan; GDN pages through [`GdnStatePool::write`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapturePlan {
    /// The KV pool's sequence slot for this capture batch (fresh, never
    /// reused — see the module doc on the allocation vehicle).
    pub capture_slot: usize,
    /// The prompt's total whole chunks (what the capture will index).
    pub total_chunks: usize,
    /// Chunks the session already holds (the capture cursor).
    pub held_chunks: usize,
    /// `[attn_layer] -> fresh kv pool ids` for chunks [held..total).
    pub fresh_kv_ids: Vec<Vec<usize>>,
    /// `[gdn_layer] -> fresh state pool ids` for chunks [held..total).
    pub fresh_gdn_ids: Vec<Vec<usize>>,
}

/// The multi-session manager: one tree, both pools, the session ledger,
/// and the serve/capture/end/evict flows.
pub struct FlashStatesManager {
    tree: RadixPrefixTree,
    kv: PagedKVCache,
    gdn: GdnStatePool,
    plan: LayerPlan,
    sessions: HashMap<u64, SessionRecord>,
    /// The KV pool's slot allocator for capture batches (monotonic —
    /// a slot is used by exactly one capture, never reused).
    next_capture_slot: usize,
}

/// The manager's chunk-major view of one session's tables: flat
/// [chunk * n_layers + layer].
fn flatten(tables: &[Vec<usize>], total: usize, nl: usize) -> Vec<usize> {
    (0..total)
        .flat_map(|c| (0..nl).map(move |l| tables[l][c]))
        .collect()
}

impl FlashStatesManager {
    /// Assemble from caller-built pools. Asserts the tree's chunk shape
    /// matches the layer plan and the KV pool's page size matches
    /// [`PAGE_TOKENS`] — a mismatched assembly would corrupt the
    /// chunk-major id routing silently.
    pub fn new(
        tree: RadixPrefixTree,
        kv: PagedKVCache,
        gdn: GdnStatePool,
        plan: LayerPlan,
    ) -> Self {
        assert_eq!(
            tree.n_layers(),
            plan.n_layers(),
            "tree n_layers must equal the layer plan's total"
        );
        assert_eq!(
            tree.page_tokens(),
            PAGE_TOKENS,
            "tree page_tokens must equal the KV pool's PAGE_SIZE"
        );
        Self {
            tree,
            kv,
            gdn,
            plan,
            sessions: HashMap::new(),
            next_capture_slot: 0,
        }
    }

    pub fn tree(&self) -> &RadixPrefixTree {
        &self.tree
    }

    pub fn kv_pool(&self) -> &PagedKVCache {
        &self.kv
    }

    pub fn gdn_pool(&self) -> &GdnStatePool {
        &self.gdn
    }

    /// Mutable pool access for the capture writer (the harness writes
    /// fresh pages' bytes between [`Self::plan_capture`] and
    /// [`Self::commit_capture`] — the designed flow, not a test seam).
    pub fn kv_pool_mut(&mut self) -> &mut PagedKVCache {
        &mut self.kv
    }

    pub fn gdn_pool_mut(&mut self) -> &mut GdnStatePool {
        &mut self.gdn
    }

    /// Register a session. Ids are caller-owned (the serving host's
    /// session identity); re-registration of a live id is a loud caller
    /// bug.
    pub fn session_start(&mut self, session_id: u64) {
        assert!(
            !self.sessions.contains_key(&session_id),
            "session {session_id} already live"
        );
        self.sessions.insert(
            session_id,
            SessionRecord {
                tables: (0..self.plan.n_layers()).map(|_| Vec::new()).collect(),
                held_chunks: 0,
                adopted_kv: (0..self.plan.n_attn).map(|_| Vec::new()).collect(),
                adopted_gdn: Vec::new(),
            },
        );
    }

    /// The serve flow (the substrate's serve shape, split for the
    /// request span): longest-prefix match → session holds on the
    /// newly-adopted chunks → unlock (the refcount hold suffices; the
    /// lock only protects the in-flight match). Returns the restore
    /// plan: a CONTINUOUS session (live cursor == its held chunks)
    /// restores nothing new; a COLD session (live cursor 0, or an
    /// evicted live cache) restores everything resolvable.
    pub fn serve(&mut self, session_id: u64, tokens: &[u32]) -> ResumePlan {
        let nl = self.plan.n_layers();
        let hit = self.tree.match_prefix(tokens);
        let matched = hit.matched_chunks;
        let held = self.sessions[&session_id].held_chunks;
        let restore_end = matched.max(held);

        let mut path = Vec::with_capacity(matched * nl);
        self.tree.path_pages_into(&hit, &mut path);
        self.tree.unlock(&hit);

        // Adopt [held..matched): take the session's hold and extend its
        // tables. (When the tree forgot part of the session's own prefix,
        // matched < held: nothing new to adopt — restore rides tables.)
        let new_from = held.min(matched);
        let new_to = matched;
        if new_to > new_from {
            let mut kv_new = vec![Vec::with_capacity(new_to - new_from); self.plan.n_attn];
            let mut gdn_new: Vec<usize> = Vec::with_capacity((new_to - new_from) * self.plan.n_gdn);
            {
                let record = self.sessions.get_mut(&session_id).expect("checked above");
                for l in 0..nl {
                    for c in new_from..new_to {
                        let id = path[c * nl + l];
                        record.tables[l].push(id);
                        if l < self.plan.n_attn {
                            kv_new[l].push(id);
                        } else {
                            gdn_new.push(id);
                        }
                    }
                }
            }
            self.kv.retain_chunk_pages(&kv_new);
            self.gdn.retain(&gdn_new);
            let record = self.sessions.get_mut(&session_id).expect("checked above");
            for (row, src) in record.adopted_kv.iter_mut().zip(&kv_new) {
                row.extend_from_slice(src);
            }
            record.adopted_gdn.extend_from_slice(&gdn_new);
            // The cursor advances to the adopted span — a repeat serve
            // of the same prompt must not re-adopt (a double hold).
            record.held_chunks = matched;
        }

        let record = self.sessions.get(&session_id).expect("checked above");
        let kv_rows: Vec<Vec<usize>> = record
            .tables
            .iter()
            .take(self.plan.n_attn)
            .map(|row| row[..restore_end].to_vec())
            .collect();
        let gdn_rows: Vec<Vec<usize>> = record
            .tables
            .iter()
            .skip(self.plan.n_attn)
            .map(|row| row[..restore_end].to_vec())
            .collect();
        ResumePlan {
            restore_chunks: restore_end,
            kv_pages: kv_rows,
            gdn_pages: gdn_rows,
        }
    }

    /// The capture half's allocation grant: fresh pages for the chunks
    /// the session does not yet hold, per layer. The caller writes the
    /// new chunks' KV rows (`write_kv` positions [held*PAGE_TOKENS..
    /// total*PAGE_TOKENS)) and GDN states into these pages, then commits
    /// with [`Self::commit_capture`]. `prompt_tokens` is the FULL prompt
    /// the capture will index (growing-context: it extends the served
    /// prompt) and must be chunk-aligned.
    pub fn plan_capture(&mut self, session_id: u64, prompt_tokens: usize) -> CapturePlan {
        assert!(
            prompt_tokens.is_multiple_of(PAGE_TOKENS),
            "capture prompts are chunk-aligned (got {prompt_tokens})"
        );
        let total = prompt_tokens / PAGE_TOKENS;
        let held = self.sessions[&session_id].held_chunks;
        assert!(
            total >= held,
            "capture prompt shorter than the session's held prefix"
        );
        let fresh = total - held;
        let capture_slot = self.next_capture_slot;
        self.next_capture_slot += 1;
        // The allocation vehicle: a fresh slot's ensure arity
        // `prompt_tokens - 1` allocates exactly `total` pages (see the
        // module doc). The slot's table entries are never consulted
        // after the read below.
        if fresh > 0 {
            self.kv.ensure_pages(capture_slot, prompt_tokens - 1);
        }
        let fresh_kv_ids: Vec<Vec<usize>> = if fresh > 0 {
            self.kv.chunk_page_tables(capture_slot, total)
        } else {
            (0..self.plan.n_attn).map(|_| Vec::new()).collect()
        };
        let mut fresh_gdn_ids = vec![Vec::with_capacity(fresh); self.plan.n_gdn];
        for _ in 0..fresh {
            for row in fresh_gdn_ids.iter_mut() {
                let id = self.gdn.alloc(&self.plan);
                row.push(id);
            }
        }
        CapturePlan {
            capture_slot,
            total_chunks: total,
            held_chunks: held,
            fresh_kv_ids,
            fresh_gdn_ids,
        }
    }

    /// Commit a capture: assemble the full chunk-major layout (the
    /// session's held prefix + the fresh pages), index it in the tree,
    /// and take the TREE's hold on the newly-indexed chunks (the insert
    /// contract's retain — the creating session takes none). Advances
    /// the session's held cursor to the capture total. On the
    /// concurrent-fill conflict (another session indexed the same prefix
    /// with its own pages between this session's serve and commit) the
    /// manager performs the adopt-recovery once and retries — see the
    /// module doc.
    pub fn commit_capture(
        &mut self,
        session_id: u64,
        tokens: &[u32],
        plan: &CapturePlan,
    ) -> Result<InsertOutcome, TreeError> {
        assert_eq!(
            tokens.len() / PAGE_TOKENS,
            plan.total_chunks,
            "capture plan and commit tokens disagree"
        );
        assert_eq!(
            plan.held_chunks, self.sessions[&session_id].held_chunks,
            "no intervening serve/capture between plan and commit"
        );
        let nl = self.plan.n_layers();
        let record = &self.sessions[&session_id];
        let held = record.held_chunks;
        let total = plan.total_chunks;

        // Assemble: [0..held) from the record, [held..total) interleaving
        // the fresh rows by layer.
        let mut tables = record.tables.clone();
        for c in held..total {
            let fresh_idx = c - held;
            for (l, row) in tables.iter_mut().enumerate().take(nl) {
                let id = if l < self.plan.n_attn {
                    plan.fresh_kv_ids[l][fresh_idx]
                } else {
                    plan.fresh_gdn_ids[l - self.plan.n_attn][fresh_idx]
                };
                row.push(id);
            }
        }

        match self.tree.insert(tokens, &flatten(&tables, total, nl)) {
            Ok(outcome) => {
                self.retain_tree_hold(session_id, &tables, total, outcome.new_chunks);
                let record = self.sessions.get_mut(&session_id).expect("checked above");
                record.tables = tables;
                record.held_chunks = total;
                Ok(outcome)
            }
            Err(TreeError::PathMismatch { .. }) => {
                // Concurrent-fill: another session indexed a prefix of
                // this prompt with its own pages after our serve. Adopt
                // the winner's pages for the tree's matched span, discard
                // our fresh duplicates there, retry ONCE.
                self.recover_conflict(session_id, tokens, &mut tables, total);
                let outcome = self.tree.insert(tokens, &flatten(&tables, total, nl))?;
                self.retain_tree_hold(session_id, &tables, total, outcome.new_chunks);
                let record = self.sessions.get_mut(&session_id).expect("checked above");
                record.tables = tables;
                record.held_chunks = total;
                Ok(outcome)
            }
            Err(e) => Err(e),
        }
    }

    /// The insert contract's retain: the TREE's hold on the
    /// newly-indexed chunks [total-new .. total). The creating session
    /// takes no hold of its own (the ledger).
    fn retain_tree_hold(
        &mut self,
        session_id: u64,
        tables: &[Vec<usize>],
        total: usize,
        new_chunks: usize,
    ) {
        if new_chunks == 0 {
            return;
        }
        let nl = self.plan.n_layers();
        let fresh_start = total - new_chunks;
        let mut kv_new = vec![Vec::with_capacity(new_chunks); self.plan.n_attn];
        let mut gdn_new: Vec<usize> = Vec::with_capacity(new_chunks * self.plan.n_gdn);
        for c in fresh_start..total {
            for (l, row) in tables.iter().enumerate().take(nl) {
                let id = row[c];
                if l < self.plan.n_attn {
                    kv_new[l].push(id);
                } else {
                    gdn_new.push(id);
                }
            }
        }
        self.kv.retain_chunk_pages(&kv_new);
        self.gdn.retain(&gdn_new);
        let _ = session_id;
    }

    /// The concurrent-fill recovery: re-match the prompt, adopt the
    /// tree's pages for the span beyond our held cursor, discard our own
    /// fresh duplicates there (they were written but never held), and
    /// splice the tables so the retry insert sees the winner's ids under
    /// the matched span and our pages under the genuinely-divergent
    /// suffix.
    fn recover_conflict(
        &mut self,
        session_id: u64,
        tokens: &[u32],
        tables: &mut [Vec<usize>],
        total: usize,
    ) {
        let nl = self.plan.n_layers();
        let held = self.sessions[&session_id].held_chunks;
        let hit = self.tree.match_prefix(tokens);
        let winner_matched = hit
            .matched_chunks
            .max(self.sessions[&session_id].held_chunks);
        let mut path = Vec::with_capacity(hit.matched_chunks * nl);
        self.tree.path_pages_into(&hit, &mut path);
        self.tree.unlock(&hit);
        let adopt_to = winner_matched.min(total);

        // Our own fresh duplicates in [held..adopt_to): return the KV
        // pages to the pool's free list (rc 0 — never held) and discard
        // the gdn allocs.
        let mut kv_dead = vec![Vec::new(); self.plan.n_attn];
        let mut gdn_dead: Vec<usize> = Vec::new();
        for c in held..adopt_to {
            for (l, row) in tables.iter().enumerate().take(nl) {
                let id = row[c];
                if l < self.plan.n_attn {
                    kv_dead[l].push(id);
                } else {
                    gdn_dead.push(id);
                }
            }
        }
        if adopt_to > held {
            self.kv.free_pages.extend(kv_dead.iter().flatten());
            self.gdn.discard(&gdn_dead);
            // Adopt the winner's pages: session hold + table splice.
            let mut kv_adopt = vec![Vec::new(); self.plan.n_attn];
            let mut gdn_adopt: Vec<usize> = Vec::new();
            let record = self.sessions.get_mut(&session_id).expect("checked above");
            for c in held..adopt_to {
                for l in 0..nl {
                    let id = path[c * nl + l];
                    record.tables[l].push(id);
                    tables[l][c] = id;
                    if l < self.plan.n_attn {
                        kv_adopt[l].push(id);
                    } else {
                        gdn_adopt.push(id);
                    }
                }
            }
            self.kv.retain_chunk_pages(&kv_adopt);
            self.gdn.retain(&gdn_adopt);
            let record = self.sessions.get_mut(&session_id).expect("checked above");
            for (row, src) in record.adopted_kv.iter_mut().zip(&kv_adopt) {
                row.extend_from_slice(src);
            }
            record.adopted_gdn.extend_from_slice(&gdn_adopt);
        }
    }

    /// End a session: release its ADOPTED holds only (fresh captures
    /// stay tree-held — the ledger). The session's own captures remain
    /// shareable to every other session through the tree.
    pub fn session_end(&mut self, session_id: u64) {
        let record = self
            .sessions
            .remove(&session_id)
            .unwrap_or_else(|| panic!("session_end on unknown session {session_id}"));
        self.kv.release_chunk_pages(&record.adopted_kv);
        self.gdn.release(&record.adopted_gdn);
    }

    /// The session's capture cursor (held chunks) — the harness reads it
    /// to plan its capture-slot range before the walk.
    pub fn session_held_chunks(&self, session_id: u64) -> usize {
        self.sessions[&session_id].held_chunks
    }

    /// Tree eviction (leaf-preferential LRU): route each victim's
    /// chunk-major pages to their pools and drop the TREE's hold. Pages
    /// a session still adopts survive on the session's hold.
    pub fn evict_to_budget(&mut self) {
        let nl = self.plan.n_layers();
        let n_attn = self.plan.n_attn;
        let kv = &mut self.kv;
        let gdn = &mut self.gdn;
        self.tree.evict_to_budget(|flat| {
            let chunks = flat.len() / nl;
            let mut kv_rows = vec![Vec::with_capacity(chunks); n_attn];
            let mut gdn_flat: Vec<usize> = Vec::with_capacity(chunks * (nl - n_attn));
            for chunk in flat.chunks(nl) {
                for (l, &id) in chunk.iter().enumerate() {
                    if l < n_attn {
                        kv_rows[l].push(id);
                    } else {
                        gdn_flat.push(id);
                    }
                }
            }
            kv.release_chunk_pages(&kv_rows);
            gdn.release(&gdn_flat);
        });
    }
}
