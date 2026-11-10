//! Plan 630 / Issue 1201 T1 — the FlashStates cell's deterministic trace
//! generator. The one shared input every arm of the pre-registered
//! multi-session cell replays: synthetic agentic growing-context sessions
//! (shared preamble + per-session monotone transcript + fresh turn
//! suffixes), emitted as TOKEN ROWS (no tokenizer in the loop — arms
//! consume the rows directly, so no re-encoding drift is possible), with
//! a BLAKE3 digest per (N, cap, seed) pinning the trace bytes.
//!
//! Device-free and deterministic by construction: the only entropy is the
//! caller's seed, run through a splitmix64-seeded xoshiro256** — no
//! platform RNG anywhere. A changed output at a pinned seed is a LOUD
//! cell invalidation (the determinism test pins the digests), never a
//! silent re-baseline.
//!
//! Posture: feature `flashstates_tracegen` (pulls only `dep:blake3` for
//! the digest). The module is device-free — it never joins a GPU feature
//! gate; the eventual harness bin (T2) consumes it alongside the serving
//! arms.

use std::fmt;

/// A turn's two token rows: the prompt a serving lane receives, and the
/// synthetic continuation it produces (seeded — the cell measures cache
/// mechanics, not decode quality; quality is G1's bit-identity gate).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Turn {
    pub prompt: Vec<u32>,
    pub continuation: Vec<u32>,
}

/// One session's turn sequence (the session owns its transcript).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Session {
    pub turns: Vec<Turn>,
}

/// The whole trace: the shared preamble + every session's turns, plus
/// the accounting arms A0/A0+ need (per-turn uncached lengths under the
/// no-reuse floor and the per-session-anchor control) and the BLAKE3
/// digest of the canonical encoding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Trace {
    /// Identical across sessions (the agentic system/tool preamble).
    pub preamble: Vec<u32>,
    pub n_sessions: usize,
    pub transcript_cap_tokens: usize,
    pub seed: u64,
    pub sessions: Vec<Session>,
    /// `blake3-<hex>` over the canonical encoding (lengths + LE bytes).
    pub digest: String,
}

/// Canonical byte encoding: header (u64 LE fields) then every row
/// (u64 length + u32 LE tokens). The digest is over these bytes only.
fn canonical_bytes(t: &Trace) -> Vec<u8> {
    let mut out = Vec::with_capacity(1 << 16);
    out.extend_from_slice(&t.n_sessions.to_le_bytes());
    out.extend_from_slice(&(t.transcript_cap_tokens as u64).to_le_bytes());
    out.extend_from_slice(&t.seed.to_le_bytes());
    let put_row = |out: &mut Vec<u8>, row: &[u32]| {
        out.extend_from_slice(&(row.len() as u64).to_le_bytes());
        for tok in row {
            out.extend_from_slice(&tok.to_le_bytes());
        }
    };
    put_row(&mut out, &t.preamble);
    for s in &t.sessions {
        for turn in &s.turns {
            put_row(&mut out, &turn.prompt);
            put_row(&mut out, &turn.continuation);
        }
    }
    out
}

impl fmt::Display for Trace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let total_prompt: u64 = self
            .sessions
            .iter()
            .map(|s| s.turns.iter().map(|t| t.prompt.len() as u64).sum::<u64>())
            .sum();
        write!(
            f,
            "flashstates-trace n_sessions={} cap={} seed={} preamble={} turns_total={} prompt_tokens_total={} digest={}",
            self.n_sessions,
            self.transcript_cap_tokens,
            self.seed,
            self.preamble.len(),
            self.sessions.iter().map(|s| s.turns.len()).sum::<usize>(),
            total_prompt,
            self.digest,
        )
    }
}

/// splitmix64 — the seed mixer (deterministic across platforms).
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// xoshiro256** — the stream generator, seeded via splitmix64 expansion.
#[derive(Clone)]
struct Rng([u64; 4]);

impl Rng {
    fn new(seed: u64) -> Self {
        let mut sm = seed;
        let s = [
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
        ];
        Self(s)
    }

    fn next_u64(&mut self) -> u64 {
        let result = self.0[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.0[1] << 17;
        self.0[2] ^= self.0[0];
        self.0[3] ^= self.0[1];
        self.0[1] ^= self.0[2];
        self.0[0] ^= self.0[3];
        self.0[2] ^= t;
        self.0[3] = self.0[3].rotate_left(45);
        result
    }

    /// Token id in [lo, hi) — the trace's vocabulary band (plausible ids,
    /// no real tokenizer mapping; arms consume rows directly).
    fn token(&mut self, lo: u32, hi: u32) -> u32 {
        lo + (self.next_u64() % u64::from(hi - lo)) as u32
    }

    /// Length in [lo, hi] inclusive.
    fn len_in(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next_u64() as usize % (hi - lo + 1))
    }
}

/// The pinned trace constants (the plan freezes them; a cell varying
/// these runs a NEW cell, never this one).
pub const PREAMBLE_TOKENS: usize = 2048;
pub const TURN_SUFFIX_MIN: usize = 128;
pub const TURN_SUFFIX_MAX: usize = 512;
pub const CONTINUATION_MIN: usize = 32;
pub const CONTINUATION_MAX: usize = 128;
/// Vocabulary band (plausible ids; disjoint preamble band so the
/// shared-prefix structure is visible in the rows themselves).
const PREAMBLE_VOCAB: (u32, u32) = (1000, 3000);
const BODY_VOCAB: (u32, u32) = (30_000, 50_000);

/// Generate the trace at (N sessions, per-session transcript cap, seed).
/// Rounds interleave strictly round-robin; a session's turn is emitted
/// while its transcript (prior suffixes + continuations, preamble
/// excluded) is still under the cap; the trace ends when every session
/// is capped.
pub fn generate(n_sessions: usize, transcript_cap_tokens: usize, seed: u64) -> Trace {
    assert!(n_sessions >= 1, "n_sessions must be >= 1");
    assert!(
        transcript_cap_tokens > TURN_SUFFIX_MAX + CONTINUATION_MAX,
        "cap must exceed one turn's footprint"
    );
    let mut rng = Rng::new(seed);
    let preamble: Vec<u32> = (0..PREAMBLE_TOKENS)
        .map(|_| rng.token(PREAMBLE_VOCAB.0, PREAMBLE_VOCAB.1))
        .collect();

    // Per-session transcript (the growing part) + the fresh-suffix stream.
    let mut transcripts: Vec<Vec<u32>> = vec![Vec::new(); n_sessions];
    let mut sessions: Vec<Session> = (0..n_sessions)
        .map(|_| Session { turns: Vec::new() })
        .collect();
    let mut open: Vec<bool> = vec![true; n_sessions];

    while open.iter().any(|&o| o) {
        for i in 0..n_sessions {
            if !open[i] {
                continue;
            }
            let suffix_len = rng.len_in(TURN_SUFFIX_MIN, TURN_SUFFIX_MAX);
            let suffix: Vec<u32> = (0..suffix_len)
                .map(|_| rng.token(BODY_VOCAB.0, BODY_VOCAB.1))
                .collect();
            // prompt = preamble + transcript-so-far + fresh suffix.
            let mut prompt = Vec::with_capacity(preamble.len() + transcripts[i].len() + suffix_len);
            prompt.extend_from_slice(&preamble);
            prompt.extend_from_slice(&transcripts[i]);
            prompt.extend_from_slice(&suffix);
            let cont_len = rng.len_in(CONTINUATION_MIN, CONTINUATION_MAX);
            let continuation: Vec<u32> = (0..cont_len)
                .map(|_| rng.token(BODY_VOCAB.0, BODY_VOCAB.1))
                .collect();
            // The transcript grows by the suffix + the synthetic output
            // (the growing-context contract: the next prompt contains
            // this whole turn).
            transcripts[i].extend_from_slice(&suffix);
            transcripts[i].extend_from_slice(&continuation);
            sessions[i].turns.push(Turn {
                prompt,
                continuation,
            });
            if transcripts[i].len() >= transcript_cap_tokens {
                open[i] = false;
            }
        }
    }

    let mut trace = Trace {
        preamble,
        n_sessions,
        transcript_cap_tokens,
        seed,
        sessions,
        digest: String::new(),
    };
    let digest = blake3::hash(&canonical_bytes(&trace)).to_hex().to_string();
    trace.digest = format!("blake3-{digest}");
    trace
}

/// One request (a turn) in strict round-robin serving order.
pub struct RequestRef<'a> {
    pub session: usize,
    pub turn: usize,
    pub prompt: &'a [u32],
    /// This turn's fresh suffix — its length is A0+'s uncached read.
    pub suffix_len: usize,
}

impl Trace {
    /// A0 (the floor: fresh stream per request) — uncached prompt tokens
    /// for every request = the full prompt. Returns one row per turn in
    /// round-robin request order (the serving order the harness drives).
    pub fn a0_uncached_lens(&self) -> Vec<usize> {
        self.requests()
            .into_iter()
            .map(|r| r.prompt.len())
            .collect()
    }

    /// A0+ (per-session single-stream caches, turn-end anchors) — the
    /// suffix refill only (the previous turn's anchor covered the whole
    /// prior prompt + continuation). Zero-entries eviction is the
    /// generous-instance control posture; budgeted variants join at T3.
    pub fn a0plus_uncached_lens(&self) -> Vec<usize> {
        self.requests().into_iter().map(|r| r.suffix_len).collect()
    }

    /// One request row per turn in strict round-robin serving order —
    /// the serving sequence the harness drives and the accounting reads.
    pub fn requests(&self) -> Vec<RequestRef<'_>> {
        let max_turns = self
            .sessions
            .iter()
            .map(|s| s.turns.len())
            .max()
            .unwrap_or(0);
        let mut out = Vec::new();
        for t in 0..max_turns {
            for (i, s) in self.sessions.iter().enumerate() {
                if let Some(turn) = s.turns.get(t) {
                    out.push(RequestRef {
                        session: i,
                        turn: t,
                        prompt: &turn.prompt,
                        suffix_len: suffix_of(self, i, t),
                    });
                }
            }
        }
        out
    }
}

// The suffix length needs the transcript deltas, not arithmetic on the
// prompt; keep the helper explicit and total.
fn suffix_of(t: &Trace, session: usize, turn: usize) -> usize {
    let s = &t.sessions[session];
    if turn == 0 {
        return s.turns[0].prompt.len() - t.preamble.len();
    }
    let prev = &s.turns[turn - 1];
    let cur = &s.turns[turn];
    cur.prompt.len() - prev.prompt.len() - prev.continuation.len()
}
