#![cfg(feature = "flashstates_multi")]
//! Plan 630 / Issue 1201 T2 — the multi-session manager's gates over the
//! REAL substrate (RadixPrefixTree + PagedKVCache — no mocks): the
//! ledger (holds taken, released, balanced), the cross-session preamble
//! share, the concurrent-fill recovery (first writer wins), eviction
//! discipline (the tree's hold drops, an adopting session's survives),
//! and the cold-session byte round-trip.

use katgpt_core::Config;
use katgpt_kv::radix_prefix::RadixPrefixTree;
use katgpt_transformer::PagedKVCache;
use riir_infer_gpu::flashstates_multisession::{
    FlashStatesManager, GdnStatePool, LayerPlan, PAGE_TOKENS,
};

/// A tiny hybrid: 1 attention layer + 1 GDN layer. KV rows are
/// `kv_dim = 4 * 4 = 16` f32 (K|V halves per page); GDN state pages are
/// 8 f32 recurrent + 4 f32 conv — shapes are arbitrary (the manager
/// never interprets bytes).
const N_ATTN: usize = 1;
const N_GDN: usize = 1;

fn plan() -> LayerPlan {
    LayerPlan {
        n_attn: N_ATTN,
        n_gdn: N_GDN,
        gdn_recurrent_len: 8,
        gdn_conv_len: 4,
    }
}

fn kv_pool() -> PagedKVCache {
    let mut config = Config::micro();
    config.n_layer = N_ATTN;
    config.block_size = 4096;
    PagedKVCache::new(&config, 4)
}

fn tree(budget_chunks: usize) -> RadixPrefixTree {
    RadixPrefixTree::new(
        N_ATTN + N_GDN,
        PAGE_TOKENS,
        budget_chunks.saturating_mul(N_ATTN + N_GDN),
    )
}

fn mgr() -> FlashStatesManager {
    FlashStatesManager::new(tree(usize::MAX), kv_pool(), GdnStatePool::new(), plan())
}

/// Distinct token sequence: chunk c of a session-tagged stream.
fn tokens(tag: u32, chunks: usize) -> Vec<u32> {
    (0..chunks * PAGE_TOKENS)
        .map(|i| tag * 100_003 + i as u32)
        .collect()
}

/// Drive a full capture: plan → write deterministic bytes → commit.
/// KV rows carry the token value; GDN recurrent carries (tag + chunk·7).
fn capture(mgr: &mut FlashStatesManager, id: u64, tag: u32, prompt: &[u32]) {
    let cp = mgr.plan_capture(id, prompt.len());
    let kvd = 16;
    for (c, chunk_tokens) in prompt.chunks(PAGE_TOKENS).enumerate() {
        if c < cp.held_chunks {
            continue;
        }
        for (i, &tok) in chunk_tokens.iter().enumerate() {
            let p = c * PAGE_TOKENS + i;
            let k = vec![tok as f32; kvd];
            let v = vec![(tok as f32) * 0.5; kvd];
            for l in 0..N_ATTN {
                mgr.kv_pool_mut().write_kv(l, cp.capture_slot, p, &k, &v);
            }
        }
        for l in 0..N_GDN {
            let id = cp.fresh_gdn_ids[l][c - cp.held_chunks];
            let rec: Vec<f32> = (0..8)
                .map(|j| (tag as f32) + (c as f32) * 7.0 + j as f32)
                .collect();
            let conv: Vec<f32> = (0..4).map(|j| (tag as f32) * 2.0 + j as f32).collect();
            mgr.gdn_pool_mut().write(id, &rec, &conv);
        }
    }
    mgr.commit_capture(id, prompt, &cp).expect("capture commit");
}

#[test]
fn shared_prefix_adopted_across_sessions_ledger_balanced() {
    let mut m = mgr();
    let preamble = tokens(1, 2); // 2 chunks, shared
    let a_prompt = [preamble.clone(), tokens(2, 2)].concat();
    m.session_start(1);
    capture(&mut m, 1, 11, &a_prompt);

    // B serves preamble + its own suffix: the preamble is A's pages.
    let b_prompt = [preamble.clone(), tokens(3, 2)].concat();
    m.session_start(2);
    let plan_b = m.serve(2, &b_prompt);
    assert_eq!(plan_b.restore_chunks, 2, "B adopts the shared preamble");
    // B's restore ids are A's captured pages (readable, held).
    assert!(m.gdn_pool().refcount(plan_b.gdn_pages[0][0]) >= 2);

    capture(&mut m, 2, 22, &b_prompt);
    // Ledger after B's capture: the trunk pages = tree(1) + B(1) = 2.
    assert_eq!(m.gdn_pool().refcount(plan_b.gdn_pages[0][0]), 2);

    // A ends: releases only its ADOPTED holds (it adopted nothing —
    // its pages are its own captures). The trunk survives for B.
    m.session_end(1);
    assert_eq!(m.gdn_pool().refcount(plan_b.gdn_pages[0][0]), 2);

    // B ends: drops its adopted trunk hold → tree-only = 1.
    m.session_end(2);
    assert_eq!(m.gdn_pool().refcount(plan_b.gdn_pages[0][0]), 1);
    // 6 pages: A's own 2 + the shared preamble 2 (ONE copy — B's
    // capture allocated only its divergent suffix, the dedup being the
    // lane's point) + B's own 2. All tree-held.
    assert_eq!(m.gdn_pool().held(), 6, "all pages stay tree-held");
}

#[test]
fn cold_session_restores_writer_bytes() {
    let mut m = mgr();
    let a_prompt = tokens(5, 3);
    m.session_start(1);
    capture(&mut m, 1, 55, &a_prompt);

    let b_prompt = [a_prompt.clone(), tokens(6, 1)].concat();
    m.session_start(2);
    let plan_b = m.serve(2, &b_prompt);
    assert_eq!(plan_b.restore_chunks, 3);

    // The restore plan's pages carry A's written bytes (the cold
    // session's harness copy would land exactly these).
    for l in 0..N_GDN {
        for (c, &id) in plan_b.gdn_pages[l].iter().enumerate() {
            let page = m.gdn_pool().page(id);
            let expect_rec: Vec<f32> = (0..8).map(|j| 55.0 + (c as f32) * 7.0 + j as f32).collect();
            assert_eq!(page.recurrent, expect_rec, "gdn bytes round-trip");
        }
    }
    // KV rows read back through the pool's real seam.
    let (mut k, mut v) = (vec![0f32; 16], vec![0f32; 16]);
    let kv_id = plan_b.kv_pages[0][2];
    // The kv pool's per-seq tables are the capture vehicle's — read
    // through the PAGE directly via the pool's pub field layout.
    let page = &m.kv_pool().pages[kv_id];
    assert_eq!(&page[..16], &[a_prompt[2 * PAGE_TOKENS] as f32; 16][..]);
    let _ = (&mut k, &mut v);
}

#[test]
fn concurrent_fill_recovers_first_writer_wins() {
    let mut m = mgr();
    let preamble = tokens(9, 2);
    // A and B both serve the preamble while the tree is EMPTY (matched
    // 0 for both), then both capture it — the second commit hits the
    // conflict and must recover onto the first writer's pages.
    m.session_start(1);
    m.session_start(2);
    assert_eq!(m.serve(1, &preamble).restore_chunks, 0);
    assert_eq!(m.serve(2, &preamble).restore_chunks, 0);

    capture(&mut m, 1, 91, &preamble);
    // B's capture: same tokens, its OWN fresh pages → conflict.
    let b_id = 2;
    let cp = {
        let prompt = &preamble;
        m.plan_capture(b_id, prompt.len())
    };
    // Write DISTINCT garbage into B's fresh pages (the wrong-writer
    // shape) — the recovery must discard them wholesale.
    for l in 0..N_GDN {
        for (i, &id) in cp.fresh_gdn_ids[l].iter().enumerate() {
            m.gdn_pool_mut().write(id, &[99.0; 8], &[99.0; 4]);
            let _ = i;
        }
    }
    let outcome = m
        .commit_capture(b_id, &preamble, &cp)
        .expect("recovery must land the insert");
    // B now holds the FIRST writer's pages (adopted, refcount 2).
    let plan_b = m.serve(b_id, &preamble);
    assert_eq!(plan_b.restore_chunks, 2);
    assert_eq!(m.gdn_pool().refcount(plan_b.gdn_pages[0][0]), 2);
    // The surviving bytes are A's, not B's garbage.
    let page = m.gdn_pool().page(plan_b.gdn_pages[0][0]);
    assert_ne!(
        page.recurrent[0], 99.0,
        "B's discarded pages are not served"
    );
    let expect: Vec<f32> = (0..8).map(|j| 91.0 + j as f32).collect();
    assert_eq!(page.recurrent, expect);
    assert_eq!(outcome.new_chunks, 0, "B's commit indexed nothing new");
}

#[test]
fn eviction_drops_tree_hold_adopting_session_pages_survive() {
    let mut m = FlashStatesManager::new(
        tree(2), // 2 chunks of headroom — A's 3-chunk capture is over
        kv_pool(),
        GdnStatePool::new(),
        plan(),
    );
    let a_prompt = tokens(11, 3);
    m.session_start(1);
    capture(&mut m, 1, 111, &a_prompt);
    let held_before = m.gdn_pool().held();

    // B adopts A's full prefix (3 chunks) and captures nothing.
    let b_prompt = [a_prompt.clone(), tokens(12, 2)].concat();
    m.session_start(2);
    let plan_b = m.serve(2, &b_prompt);
    assert_eq!(plan_b.restore_chunks, 3);
    let adopted_gdn_id = plan_b.gdn_pages[0][0];

    // A ends (no adopted holds — its captures) → tree-only on A's pages.
    m.session_end(1);
    assert_eq!(m.gdn_pool().refcount(adopted_gdn_id), 2); // tree + B

    // Evict: 5 chunks held (tree) > 3 budget → A's leaf chunks drop.
    m.evict_to_budget();
    assert!(m.tree().held_pages() <= 3 * (N_ATTN + N_GDN));
    // B's ADOPTED pages survive on the session's hold even when the
    // tree forgot them; the tree-only tail freed.
    assert!(m.gdn_pool().held() <= held_before);
    assert_eq!(m.gdn_pool().refcount(adopted_gdn_id), 1, "B's hold");

    // B's restore still resolves its adopted span from its tables.
    let plan_b2 = m.serve(2, &b_prompt);
    assert!(plan_b2.restore_chunks >= 1);
    assert_eq!(plan_b2.gdn_pages[0][0], adopted_gdn_id);
}

#[test]
fn continuous_session_restores_nothing_new() {
    let mut m = mgr();
    let turn1 = tokens(21, 2);
    m.session_start(1);
    capture(&mut m, 1, 21, &turn1);
    let rc_before: Vec<usize> = (0..m.gdn_pool().len())
        .map(|id| m.gdn_pool().refcount(id))
        .collect();

    // Turn 2: the growing prompt re-matches its own prefix.
    let turn2 = [turn1.clone(), tokens(22, 2)].concat();
    let plan2 = m.serve(1, &turn2);
    assert_eq!(plan2.restore_chunks, 2, "its own prefix, nothing new");
    capture(&mut m, 1, 22, &turn2);

    // Refcounts unchanged on the old pages (the tree hold existed since
    // turn 1; turn 2's insert only retains the NEW chunks).
    for (id, &rc) in rc_before.iter().enumerate() {
        assert_eq!(m.gdn_pool().refcount(id), rc, "page {id} refcount drift");
    }
}

/// The KV pool's per-page refcount (pub field on the substrate).
fn kv_refcount(m: &FlashStatesManager, id: usize) -> usize {
    m.kv_pool().page_ref_counts[id]
}

#[test]
fn evicted_tail_under_live_session_truncates_the_cursor_never_dangles() {
    // The T2-completion floor: fresh captures are TREE-HOLD-ONLY (the
    // ledger), so tree eviction can free a LIVE session's own pages
    // (rc 1→0 → free-list → reuse-clobbered). evict_to_budget TRUNCATES
    // the affected session's cursor AT EVICTION TIME — riding the
    // dangling table ids would restore garbage (or another session's
    // state) at the next serve.
    let mut m = FlashStatesManager::new(tree(2), kv_pool(), GdnStatePool::new(), plan());
    let a_prompt = tokens(31, 3); // 3 chunks over a 2-chunk budget
    m.session_start(1);
    capture(&mut m, 1, 31, &a_prompt);
    let held_before = m.session_held_chunks(1);
    assert_eq!(held_before, 3);

    // Eviction fires while session 1 is LIVE: over budget → the oldest
    // leaf chunks drop, their pages free (tree-hold-only), and the
    // session's cursor truncates to the surviving prefix.
    m.evict_to_budget();
    let held_after = m.session_held_chunks(1);
    assert!(
        held_after < held_before,
        "eviction freed this session's own pages; the cursor must truncate"
    );

    // Next serve: the resolvable prefix only. Every planned id must
    // still be held by a pool (rc ≥ 1) — a freed id here is the defect.
    let resumed = m.serve(1, &a_prompt);
    assert_eq!(
        resumed.restore_chunks, held_after,
        "restore rides the truncated cursor"
    );
    for row in resumed.kv_pages.iter().chain(resumed.gdn_pages.iter()) {
        for &id in row.iter() {
            assert!(
                kv_refcount(&m, id) >= 1 || m.gdn_pool().refcount(id) >= 1,
                "plan carries a freed page id {id}"
            );
        }
    }

    // The truncated session captures forward again from the truncated
    // cursor (the next capture re-indexes the suffix).
    let turn2 = [a_prompt.clone(), tokens(32, 1)].concat();
    let served2 = m.serve(1, &turn2);
    assert_eq!(
        served2.restore_chunks, held_after,
        "turn2's prompt re-matches exactly the surviving prefix"
    );
    capture(&mut m, 1, 32, &turn2);
    assert_eq!(m.session_held_chunks(1), turn2.len() / PAGE_TOKENS);
}
