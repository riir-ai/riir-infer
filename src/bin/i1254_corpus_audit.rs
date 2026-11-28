//! Issue 1254 A1 — the fact-corpus audit (the determinism receipt).
//!
//! Report-only, CPU-only, ungated (default build): prints the corpus digest,
//! the split-assignment digest, per-relation counts, split counts, and one
//! sample prompt per relation. This is the receipt the cell record cites —
//! the corpus data plane is pinned by these two digests before any forward
//! runs against it.
//!
//! Run: `cargo run --release --bin i1254_corpus_audit`

use riir_infer_core::i1254_corpus::{
    RELATIONS, Split, build_prompt, corpus_digest, load_corpus, relation_of,
    split_assignment_digest, split_of,
};

fn main() {
    println!("== i1254 A1 fact-corpus audit ==");
    println!("corpus digest:  {}", corpus_digest());

    let facts = match load_corpus() {
        Ok(f) => f,
        Err(e) => {
            println!("REFUSAL(corpus-load): {e}");
            std::process::exit(2);
        }
    };
    println!("split digest:   {}", split_assignment_digest(&facts));
    println!("facts total:    {}", facts.len());

    let mut counts = std::collections::BTreeMap::new();
    let mut splits = [0usize; 3];
    for f in &facts {
        *counts.entry(f.rel).or_insert(0usize) += 1;
        match split_of(&f.subj) {
            Split::Train => splits[0] += 1,
            Split::Val => splits[1] += 1,
            Split::Test => splits[2] += 1,
        }
    }
    println!("-- per relation --");
    for r in RELATIONS {
        println!(
            "  {:>4} {:<24} {}",
            r.code,
            r.name,
            counts.get(r.code).copied().unwrap_or(0)
        );
    }
    println!(
        "-- split (subject-wise) --  train={} val={} test={} (of {})",
        splits[0],
        splits[1],
        splits[2],
        facts.len()
    );

    println!("-- sample prompts --");
    for r in RELATIONS {
        if let Some(f) = facts.iter().find(|f| f.rel == r.code) {
            println!("  [{}] {}", r.code, build_prompt(r, &f.subj));
        }
    }
    println!("-- split pins (determinism receipt) --");
    for subj in ["France", "Io", "Sherlock Holmes"] {
        println!("  {subj:?} -> {:?}", split_of(subj));
    }
    // A refusal-battery probe: unknown codes refuse loud (never silently).
    match relation_of("zz") {
        Err(_) => println!("refusal battery: unknown-relation code refused OK"),
        Ok(_) => {
            println!("REFUSAL BATTERY FAILED: unknown code resolved");
            std::process::exit(2);
        }
    }
    println!("== audit done ==");
}
