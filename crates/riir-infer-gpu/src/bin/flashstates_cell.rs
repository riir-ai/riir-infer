//! Plan 630 / Issue 1201 T2 — the FlashStates cell harness: the
//! pre-registered multi-session cell's arms over the REAL Metal forward,
//! the plan's deterministic tracegen, and the T2 radix manager.
//!
//! # Arms
//!
//! - `a0` — the incumbent floor: fresh stream per request
//!   ([`Qwen38WgpuForward::zero_state`] + full prefill + teacher-forced
//!   decode). Uncached = the full prompt, by construction.
//! - `a0p` — the per-session single-stream control: one
//!   [`PrefixCacheCubecl`] per session, anchor at every turn end;
//!   within-session the walk is restore + suffix refill. Full-state
//!   snapshots (the port's disclosed divergence — the module doc);
//!   reported beside, never a bar denominator.
//! - `a1` — the radix arm: [`FlashStatesManager`] shared across sessions,
//!   automatic end-of-request capture, restore = KV rows + GDN state
//!   copied INTO the live buffers (the manager is device-free; this
//!   harness is the composition).
//! - `a2` — T3's arm: a1 + cost-density retention — the manager armed
//!   with [`FlashStatesManager::with_gds_retention`], the same pinned
//!   shared-state budget (`--budget-frac`, default 0.25 of the trace's
//!   total demand; 0 = the T2 infinite posture, no eviction anywhere).
//!   Both budgeted arms are bounded on the same tree-index axis; the
//!   budgeted G1 sample covers the POST-EVICTION resume.
//!
//! # The drive partition (the G1 precondition)
//!
//! Every walk processes tokens through ONE canonical partition: sequential
//! decode steps to the next chunk boundary, full chunk-batch ingests, a
//! sequential ragged tail — batch boundaries at every `chunk` multiple. A
//! resumed walk and a full re-prefill of the same request therefore
//! execute the IDENTICAL op sequence over the shared suffix region, which
//! makes G1's bit-identity a state-identity claim (the restored bytes are
//! the only possible difference), never a segmentation artifact.
//!
//! # A1 restore granularity (the honest T2 posture)
//!
//! GDN state is sequential — intermediate chunk-boundary states inside a
//! batched prefill are not observable from outside the forward (the
//! kernels overwrite the state buffers as the batch advances). The T2
//! harness therefore captures TRUE GDN states only at chunk boundaries
//! (the drive's observable boundaries, via the device range-copy into a
//! per-turn arena — zero syncs during the walk; T4's in-chunk kernel
//! export is the named successor). A1 restores to the deepest
//! chunk-boundary `P_r` ≤ the tree match: KV rows `[0..P_r)` from the
//! pool, the GDN page at `P_r`, and the stretch `[P_r..)` re-prefilled
//! (its KV rows recomputed in-walk). Intra-boundary GDN pages carry their
//! containing boundary's state (slot = `(c+1)·PAGE_TOKENS / chunk`) —
//! never consumed under the floor-boundary rule, disclosed here because
//! T3's eviction WILL create partial matches that must either land on
//! boundaries or wait for T4.
//!
//! # Metrics (plan 630's definitions)
//!
//! median uncached prompt tokens (prompt − restored prefix), p50/p90 TTFT
//! (walk start → the prefill's first-token readback; tail support printed
//! beside every p90 — the n−1-max rule), teacher-forced decode tok/s.
//! Bars evaluate only from summaries of the same cell directory (same
//! trace digest enforced by the manifest).
//!
//! # Resumability (the compute law)
//!
//! Durable state under `.benchmarks/flashstates_cell/<n>_<cap>_s<seed>/`:
//! `manifest.json` (trace digest + args — a digest mismatch on `--resume`
//! refuses), per-arm `<arm>.rows.jsonl` + `<arm>.progress.json` (the
//! cursor is temp+rename-written after every request row; a row past the
//! cursor is ignored on resume). A resumed A1 arm rebuilds its manager
//! EMPTY (tree/pools are process state, never persisted) — the requests
//! past the cursor are all served cold for their sessions (restores read
//! 0 until captures re-accumulate). That is fine for the G1 posture
//! (correctness, never bars); a MEASURED cell runs its bars from an
//! unresumed arm, or discloses the resume boundary.
//! Idempotent: a completed arm re-prints its
//! summary from the rows. Long cells: run inside `caffeinate -dis`, GPU
//! exclusivity attested separately (the bin prints timing only, never
//! box-state claims — quote `bench_preflight`'s PROVENANCE line beside
//! any published number).
//!
//! # G1
//!
//! `--g1` is the short correctness posture: sampled requests (turn 0 of
//! every session + every 64th, capped by `--g1-sample`, default 8 in any
//! measured run) get a resumed-vs-full deep check — first-decode logits
//! BLAKE3 equality + the whole decode argmax sequence equal. A drift
//! aborts the run loud (exit 1). Reaching the summary with g1 checks
//! recorded IS the bar-6 verdict on the sampled set.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use cubecl::bytes::Bytes;
use cubecl::prelude::ComputeClient;
use cubecl::server::Handle;

use katgpt_core::Config as KatConfig;
use katgpt_kv::radix_prefix::RadixPrefixTree;
use katgpt_transformer::PagedKVCache;

use riir_infer_gpu::copy_range_cubecl::copy_range_f32_launch;
use riir_infer_gpu::cubecl_runtime::ActiveRuntime;
use riir_infer_gpu::flashstates_multisession::{
    FlashStatesManager, GdnStatePool, LayerPlan, PAGE_TOKENS, ResumePlan,
};
use riir_infer_gpu::flashstates_tracegen::{self, RequestRef, Trace};
use riir_infer_gpu::qwen_attention_cubecl::QwenKvCacheFillSplitBatchedCubeCL;
use riir_infer_gpu::qwen38_dense_cubecl::{Qwen38WgpuForward, Qwen38WgpuLayerType, StateKind};
use riir_infer_gpu::qwen38_prefix_cache_cubecl::PrefixCacheCubecl;

// ---------------------------------------------------------------------------
// rows + the per-arm resume log

#[derive(Clone, Debug)]
struct Row {
    req: usize,
    session: usize,
    turn: usize,
    prompt_len: usize,
    restored: usize,
    uncached: usize,
    ttft_ms: f64,
    decode_ms: f64,
    decode_tokens: usize,
    first_token: u32,
}

fn row_json(r: &Row) -> String {
    serde_json::json!({
        "req": r.req, "session": r.session, "turn": r.turn,
        "prompt_len": r.prompt_len, "restored": r.restored, "uncached": r.uncached,
        "ttft_ms": r.ttft_ms, "decode_ms": r.decode_ms,
        "decode_tokens": r.decode_tokens, "first_token": r.first_token,
    })
    .to_string()
}

fn row_from(v: &serde_json::Value) -> Option<Row> {
    let gi = |k: &str| -> usize { v.get(k).and_then(|x| x.as_u64()).unwrap_or(0) as usize };
    let gf = |k: &str| -> f64 { v.get(k).and_then(|x| x.as_f64()).unwrap_or(0.0) };
    Some(Row {
        req: gi("req"),
        session: gi("session"),
        turn: gi("turn"),
        prompt_len: gi("prompt_len"),
        restored: gi("restored"),
        uncached: gi("uncached"),
        ttft_ms: gf("ttft_ms"),
        decode_ms: gf("decode_ms"),
        decode_tokens: gi("decode_tokens"),
        first_token: v.get("first_token")?.as_u64()? as u32,
    })
}

/// Per-arm append log + cursor. `record` appends the row, then bumps the
/// cursor via temp+rename — a crash between the two writes leaves a row
/// past the cursor, ignored on resume.
struct ArmLog {
    path: PathBuf,
    progress_path: PathBuf,
    next_req: usize,
}

impl ArmLog {
    fn open(dir: &Path, arm: &str, resume: bool) -> Self {
        let path = dir.join(format!("{arm}.rows.jsonl"));
        let progress_path = dir.join(format!("{arm}.progress.json"));
        let next_req = if resume && progress_path.exists() {
            let txt = fs::read_to_string(&progress_path).expect("read progress");
            let v: serde_json::Value = serde_json::from_str(&txt).expect("parse progress");
            v["next_req"].as_u64().unwrap_or(0) as usize
        } else {
            if path.exists() {
                fs::remove_file(&path).ok();
            }
            0
        };
        Self {
            path,
            progress_path,
            next_req,
        }
    }

    fn record(&mut self, row: &Row) {
        let line = row_json(row);
        let mut f = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .expect("open rows");
        writeln!(f, "{line}").expect("append row");
        f.sync_all().ok();
        let tmp = self.progress_path.with_extension("json.tmp");
        fs::write(&tmp, format!("{{\"next_req\":{}}}", row.req + 1)).expect("write progress");
        fs::rename(&tmp, &self.progress_path).expect("rename progress");
        self.next_req = row.req + 1;
    }

    fn complete(&self, dir: &Path, arm: &str) {
        fs::write(dir.join(format!("{arm}.COMPLETE")), "ok\n").expect("complete flag");
    }
}

fn read_rows(path: &Path, upto: usize) -> Vec<Row> {
    let mut rows = Vec::new();
    let Ok(txt) = fs::read_to_string(path) else {
        return rows;
    };
    for line in txt.lines() {
        let parsed = serde_json::from_str::<serde_json::Value>(line).ok();
        let Some(row) = parsed.as_ref().and_then(row_from) else {
            continue;
        };
        if row.req < upto {
            rows.push(row);
        }
    }
    rows.shrink_to_fit();
    rows
}

// ---------------------------------------------------------------------------
// args

struct Args {
    n: usize,
    cap: usize,
    seed: u64,
    gguf: String,
    ctx: usize,
    chunk: usize,
    arms: Vec<String>,
    out_dir: Option<PathBuf>,
    resume: bool,
    max_requests: Option<usize>,
    g1_sample: Option<usize>,
    /// The pinned shared-state budget (plan 630 bar 5): budget_frac × the
    /// trace's total demand (N × cap tokens → chunks × n_layers pages).
    /// 0.0 = the infinite tree budget (the T2 mechanics posture — no
    /// eviction ever fires; byte-identical). Applies to a1/a2 only.
    budget_frac: f64,
}

fn default_gguf() -> String {
    std::env::var("QWEN38_CELL_GGUF")
        .unwrap_or_else(|_| "/Users/kat/git/riir-train/data/Qwen3.8-27B-DFlash2-Q4_K_M.gguf".into())
}

fn next_val(it: &mut impl Iterator<Item = String>, name: &str) -> String {
    it.next().unwrap_or_else(|| panic!("{name} needs a value"))
}

fn parse_args() -> Args {
    let mut a = Args {
        n: 8,
        cap: 8192,
        seed: 0,
        gguf: default_gguf(),
        ctx: 0,
        chunk: 512,
        arms: vec!["a0".into(), "a0p".into(), "a1".into()],
        out_dir: None,
        resume: false,
        max_requests: None,
        g1_sample: None,
        budget_frac: 0.25,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n" => a.n = next_val(&mut it, "--n").parse().expect("n"),
            "--cap" => a.cap = next_val(&mut it, "--cap").parse().expect("cap"),
            "--seed" => a.seed = next_val(&mut it, "--seed").parse().expect("seed"),
            "--gguf" => a.gguf = next_val(&mut it, "--gguf"),
            "--ctx" => a.ctx = next_val(&mut it, "--ctx").parse().expect("ctx"),
            "--chunk" => a.chunk = next_val(&mut it, "--chunk").parse().expect("chunk"),
            "--arm" => {
                a.arms = next_val(&mut it, "--arm")
                    .split(',')
                    .map(str::to_string)
                    .collect();
            }
            "--out-dir" => a.out_dir = Some(PathBuf::from(next_val(&mut it, "--out-dir"))),
            "--resume" => a.resume = true,
            "--max-requests" => {
                a.max_requests = Some(next_val(&mut it, "--max-requests").parse().expect("k"))
            }
            "--g1-sample" => {
                a.g1_sample = Some(next_val(&mut it, "--g1-sample").parse().expect("k"))
            }
            "--budget-frac" => {
                let v: f64 = next_val(&mut it, "--budget-frac").parse().expect("frac");
                assert!(
                    (0.0..=1.0).contains(&v),
                    "--budget-frac must be in [0, 1] (0 = no eviction)"
                );
                a.budget_frac = v;
            }
            "--g1" => {
                a.g1_sample = Some(usize::MAX);
                a.max_requests = Some(12);
                a.n = a.n.min(4);
                a.cap = a.cap.min(2048);
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => panic!("unknown arg {other} (--help)"),
        }
    }
    if a.ctx == 0 {
        let need = 2048 + a.cap + 1024;
        a.ctx = need.div_ceil(a.chunk) * a.chunk;
    }
    a
}

fn print_help() {
    println!(
        "flashstates_cell — plan 630 / issue 1201 T2/T3 harness\n\
         --n N --cap T --seed S      trace shape (default 8 / 8192 / 0)\n\
         --gguf PATH                 model (env QWEN38_CELL_GGUF, else the m3 league path)\n\
         --ctx T --chunk C           forward window + drive batch width (0 = derived)\n\
         --arm a0,a0p,a1,a2          arms (a2 = T3 cost-density retention, needs budget)\n\
         --budget-frac F             shared-state budget = F × trace demand (default 0.25;\
                                     0 = infinite tree budget, no eviction — T2 posture)\n\
         --out-dir DIR               durable state (default .benchmarks/flashstates_cell/<id>)\n\
         --resume                    continue partial arms\n\
         --max-requests K            truncate the request sequence\n\
         --g1-sample K               cap the deep resumed-vs-full sample (0 disables)\n\
         --g1                        the short correctness posture (sample=all, 12 requests)"
    );
}

// ---------------------------------------------------------------------------
// stats

fn percentile(sorted: &[f64], p: f64) -> (f64, usize) {
    assert!(!sorted.is_empty(), "percentile of empty");
    let idx = ((p * sorted.len() as f64).ceil() as usize).max(1) - 1;
    (sorted[idx], sorted.len() - idx)
}

fn median(sorted: &[f64]) -> f64 {
    assert!(!sorted.is_empty(), "median of empty");
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    }
}

struct Stats {
    requests: usize,
    median_uncached: f64,
    ttft_p50: f64,
    sup50: usize,
    ttft_p90: f64,
    sup90: usize,
    decode_med: f64,
    decode_agg: f64,
    g1_checked: usize,
}

fn stats_of(rows: &[Row], g1_checked: usize) -> Stats {
    assert!(!rows.is_empty(), "stats of empty rows");
    let mut uncached: Vec<f64> = rows.iter().map(|r| r.uncached as f64).collect();
    let mut ttft: Vec<f64> = rows.iter().map(|r| r.ttft_ms).collect();
    let mut drate: Vec<f64> = rows
        .iter()
        .map(|r| {
            if r.decode_ms > 0.0 {
                r.decode_tokens as f64 * 1000.0 / r.decode_ms
            } else {
                0.0
            }
        })
        .collect();
    uncached.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ttft.sort_by(|a, b| a.partial_cmp(b).unwrap());
    drate.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (ttft_p50, sup50) = percentile(&ttft, 0.5);
    let (ttft_p90, sup90) = percentile(&ttft, 0.9);
    let total_decode_ms: f64 = rows.iter().map(|r| r.decode_ms).sum();
    let total_decode_tokens: usize = rows.iter().map(|r| r.decode_tokens).sum();
    Stats {
        requests: rows.len(),
        median_uncached: median(&uncached),
        ttft_p50,
        sup50,
        ttft_p90,
        sup90,
        decode_med: median(&drate),
        decode_agg: total_decode_tokens as f64 * 1000.0 / total_decode_ms.max(1e-9),
        g1_checked,
    }
}

fn stats_value(arm: &str, s: &Stats) -> serde_json::Value {
    serde_json::json!({
        "arm": arm,
        "requests": s.requests,
        "median_uncached": s.median_uncached,
        "ttft_p50_ms": s.ttft_p50, "ttft_p50_support": s.sup50,
        "ttft_p90_ms": s.ttft_p90, "ttft_p90_support": s.sup90,
        "decode_median_tok_s": s.decode_med,
        "decode_aggregate_tok_s": s.decode_agg,
        "g1_checked": s.g1_checked,
    })
}

fn print_stats(arm: &str, s: &Stats) {
    println!(
        "[{arm}] requests={} median_uncached={:.0} ttft_p50={:.1}ms(sup {}) ttft_p90={:.1}ms(sup {}) decode_med={:.1} decode_agg={:.1}tok/s g1_checked={}",
        s.requests,
        s.median_uncached,
        s.ttft_p50,
        s.sup50,
        s.ttft_p90,
        s.sup90,
        s.decode_med,
        s.decode_agg,
        s.g1_checked,
    );
}

// ---------------------------------------------------------------------------
// the drive + the boundary capture sink

/// The A1 boundary-capture sink. Holds PRE-CLONED handles so the drive can
/// launch copies while the forward is `&mut` — launches are async (the
/// in-order queue), the arena is read once at capture finalize. Arena
/// layout: `[(slot - slot_lo) * n_gdn + layer] * elems` f32; only the
/// fresh span `[slot_lo, slot_hi]` is allocated.
struct DriveCapture {
    client: ComputeClient<ActiveRuntime>,
    gdn_handles: Vec<(Handle, Handle)>,
    arena_gdn: Handle,
    arena_conv: Handle,
    n_gdn: usize,
    gdn_elems: usize,
    conv_elems: usize,
    chunk: usize,
    slot_lo: usize,
    slot_hi: usize,
}

impl DriveCapture {
    fn boundary(&mut self, total_tokens: usize) {
        if total_tokens == 0 || !total_tokens.is_multiple_of(self.chunk) {
            return;
        }
        let slot = total_tokens / self.chunk;
        if slot < self.slot_lo || slot > self.slot_hi {
            return;
        }
        let idx = slot - self.slot_lo;
        for (gl, (gdn_h, conv_h)) in self.gdn_handles.iter().enumerate() {
            let off_g = ((idx * self.n_gdn + gl) * self.gdn_elems * 4) as u64;
            let off_c = ((idx * self.n_gdn + gl) * self.conv_elems * 4) as u64;
            unsafe {
                // SRC is the layer's own (small) state buffer — offset 0;
                // the arena slot offset belongs to the DST only.
                copy_range_f32_launch(
                    &self.client,
                    gdn_h.clone(),
                    self.arena_gdn.clone().offset_start(off_g),
                    self.gdn_elems,
                );
                copy_range_f32_launch(
                    &self.client,
                    conv_h.clone(),
                    self.arena_conv.clone().offset_start(off_c),
                    self.conv_elems,
                );
            }
        }
    }
}

/// The canonical partition drive: sequential steps to the next chunk
/// boundary, full chunk-batch ingests, sequential ragged tail. Returns the
/// prefill's last-token argmax (the first decode token under greedy
/// sampling). `sink` receives the boundary crossings (the state after
/// exactly `total_tokens` tokens).
fn drive(
    fwd: &mut Qwen38WgpuForward,
    tokens: &[u32],
    base: usize,
    chunk: usize,
    sink: &mut Option<DriveCapture>,
) -> Result<u32, String> {
    let mut arg = 0u32;
    let mut p = 0usize;
    while p < tokens.len() && !(base + p).is_multiple_of(chunk) {
        arg = fwd.forward_token(tokens[p], base + p)?;
        p += 1;
        if let Some(c) = sink.as_mut() {
            c.boundary(base + p);
        }
    }
    while tokens.len() - p >= chunk {
        arg = fwd.ingest(&tokens[p..p + chunk], base + p)?;
        p += chunk;
        if let Some(c) = sink.as_mut() {
            c.boundary(base + p);
        }
    }
    while p < tokens.len() {
        arg = fwd.forward_token(tokens[p], base + p)?;
        p += 1;
        if let Some(c) = sink.as_mut() {
            c.boundary(base + p);
        }
    }
    Ok(arg)
}

/// Teacher-forced decode of the trace's continuation, same boundary hooks.
fn drive_decode(
    fwd: &mut Qwen38WgpuForward,
    cont: &[u32],
    base: usize,
    sink: &mut Option<DriveCapture>,
) -> Result<(), String> {
    for (i, &t) in cont.iter().enumerate() {
        fwd.forward_token(t, base + i)?;
        if let Some(c) = sink.as_mut() {
            c.boundary(base + i + 1);
        }
    }
    Ok(())
}

fn logits_digest(logits: &[f32]) -> String {
    use blake3::Hasher;
    let mut h = Hasher::new();
    for v in logits {
        h.update(&v.to_le_bytes());
    }
    h.finalize().to_hex().to_string()[..16].to_string()
}

/// The G1 core: the resumed walk's state is already in `fwd` (the caller
/// applied its restore); drive `[resume_len..]`, capture the first-decode
/// logits, then run the FULL re-prefill from a fresh stream and demand
/// bit-identity (BLAKE3 over the logits row + the decode argmax sequence).
fn compare_walks(
    fwd: &mut Qwen38WgpuForward,
    prompt: &[u32],
    cont: &[u32],
    chunk: usize,
    resume_len: usize,
) -> Result<(), String> {
    let mut sink: Option<DriveCapture> = None;
    drive(fwd, &prompt[resume_len..], resume_len, chunk, &mut sink)?;
    let (arg_r, logits_r, _) = fwd.forward_token_capture(cont[0], prompt.len(), &[])?;
    let mut argmax_r = vec![arg_r];
    for (i, &t) in cont.iter().enumerate().skip(1) {
        argmax_r.push(fwd.forward_token(t, prompt.len() + i)?);
    }
    let digest_r = logits_digest(&logits_r);

    fwd.zero_state();
    drive(fwd, prompt, 0, chunk, &mut sink)?;
    let (arg_f, logits_f, _) = fwd.forward_token_capture(cont[0], prompt.len(), &[])?;
    let mut argmax_f = vec![arg_f];
    for (i, &t) in cont.iter().enumerate().skip(1) {
        argmax_f.push(fwd.forward_token(t, prompt.len() + i)?);
    }
    let digest_f = logits_digest(&logits_f);

    if argmax_r != argmax_f {
        return Err(format!(
            "G1 drift ({resume_len}-resumed vs full): decode argmax — resumed {argmax_r:?} vs full {argmax_f:?}"
        ));
    }
    if digest_r != digest_f {
        return Err(format!(
            "G1 drift ({resume_len}-resumed vs full): first-decode logits BLAKE3 — resumed {digest_r} vs full {digest_f}"
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// the cell (the forward + everything derived from it)

struct Cell {
    fwd: Qwen38WgpuForward,
    chunk: usize,
    kvd: usize,
    n_attn: usize,
    n_gdn: usize,
    /// Layer indices of the attention layers (kind order) — the layer
    /// kinds are INTERLEAVED (interval pattern), never two blocks; the
    /// manager's row counts (n_attn/n_gdn) translate through these.
    attn_ids: Vec<usize>,
    gdn_ids: Vec<usize>,
    gdn_elems: usize,
    conv_elems: usize,
    /// The cold-stream GDN/conv bytes (indexed by GDN layer, in order) —
    /// a fresh stream's recurrence starts at zero; A1 cold walks write
    /// these instead of paying `zero_state` per request. The KV cold bytes
    /// are never needed (a cold A1 walk prefills from 0, overwriting every
    /// row it will ever read).
    cold_gdn_bytes: Vec<Vec<u8>>,
    cold_conv_bytes: Vec<Vec<u8>>,
}

impl Cell {
    fn boot(gguf: &str, ctx: usize, chunk: usize) -> Result<Self, String> {
        assert!(
            chunk.is_multiple_of(PAGE_TOKENS),
            "drive chunk {chunk} must be a multiple of the tree page ({PAGE_TOKENS})"
        );
        let mut fwd = Qwen38WgpuForward::new(Path::new(gguf), ctx, chunk)?;
        let mut attn_ids = Vec::new();
        let mut gdn_ids = Vec::new();
        for (li, kind) in fwd.config.layer_types.iter().enumerate() {
            if *kind == Qwen38WgpuLayerType::Attention {
                attn_ids.push(li);
            } else {
                gdn_ids.push(li);
            }
        }
        let n_attn = attn_ids.len();
        let n_gdn = gdn_ids.len();
        let kvd = fwd.config.kvd();
        let gdn_elems = fwd.state_elems(StateKind::Gdn);
        let conv_elems = fwd.state_elems(StateKind::Conv);
        // Guarantee the boot state, then snapshot it as THE cold-stream
        // bytes (a fresh forward's buffers may be uninitialized
        // allocations; zero_state is definitive and runs once).
        fwd.zero_state();
        let mut cold_gdn_bytes = Vec::with_capacity(n_gdn);
        let mut cold_conv_bytes = Vec::with_capacity(n_gdn);
        for (li, kind) in fwd.config.layer_types.iter().enumerate() {
            if *kind != Qwen38WgpuLayerType::Deltanet {
                continue;
            }
            cold_gdn_bytes.push(
                fwd.layer_state_bytes(li, StateKind::Gdn)
                    .ok_or("gdn state slot missing")?,
            );
            cold_conv_bytes.push(
                fwd.layer_state_bytes(li, StateKind::Conv)
                    .ok_or("conv state slot missing")?,
            );
        }
        Ok(Self {
            fwd,
            chunk,
            kvd,
            n_attn,
            n_gdn,
            attn_ids,
            gdn_ids,
            gdn_elems,
            conv_elems,
            cold_gdn_bytes,
            cold_conv_bytes,
        })
    }

    /// The derived page economics (printed at boot so every run record
    /// carries its own memory posture): bytes per chunk of live shared
    /// state — the budget bounds exactly this.
    fn print_page_economics(&self) {
        let kv_bytes = self.n_attn * PAGE_TOKENS * self.kvd * 2 * 4;
        let gdn_bytes = self.n_gdn * (self.gdn_elems + self.conv_elems) * 4;
        println!(
            "[cell] page economics: layers={} (attn {} / gdn {}) kvd={} gdn_elems={} conv_elems={} — per chunk: kv {} B, gdn {} B, total {} B",
            self.n_attn + self.n_gdn,
            self.n_attn,
            self.n_gdn,
            self.kvd,
            self.gdn_elems,
            self.conv_elems,
            kv_bytes,
            gdn_bytes,
            kv_bytes + gdn_bytes,
        );
    }

    fn layer_plan(&self) -> LayerPlan {
        LayerPlan {
            n_attn: self.n_attn,
            n_gdn: self.n_gdn,
            gdn_recurrent_len: self.gdn_elems,
            gdn_conv_len: self.conv_elems,
        }
    }

    /// A1 restore: pool ids → live buffers. KV rows `[0..pr_tokens)` per
    /// attention layer (staging upload + the fill kernel at `base_pos =
    /// 0`); GDN from the floor-boundary page. `pr_tokens == 0` writes the
    /// cold-stream bytes (fresh stream). Returns `pr_tokens` — the resume
    /// point the caller drives from.
    fn a1_restore(
        &self,
        mgr: &FlashStatesManager,
        plan: &ResumePlan,
        prompt_tokens: usize,
    ) -> Result<usize, String> {
        let client = self.fwd.client().clone();
        let chunk_pages = self.chunk / PAGE_TOKENS;
        let restore_chunks = plan.restore_chunks.min(prompt_tokens / PAGE_TOKENS);
        let pr_tokens = (restore_chunks / chunk_pages) * self.chunk;
        if pr_tokens == 0 {
            for (gl, bytes) in self.cold_gdn_bytes.iter().enumerate() {
                self.fwd
                    .write_layer_state(self.gdn_ids[gl], StateKind::Gdn, bytes);
            }
            for (gl, bytes) in self.cold_conv_bytes.iter().enumerate() {
                self.fwd
                    .write_layer_state(self.gdn_ids[gl], StateKind::Conv, bytes);
            }
            pollster::block_on(client.sync()).ok();
            return Ok(0);
        }

        let mut host_k = vec![0.0f32; pr_tokens * self.kvd];
        let mut host_v = vec![0.0f32; pr_tokens * self.kvd];
        let staging_k = client.create_from_slice(&vec![0u8; pr_tokens * self.kvd * 4]);
        let staging_v = client.create_from_slice(&vec![0u8; pr_tokens * self.kvd * 4]);
        for (l, &li_a) in self.attn_ids.iter().enumerate() {
            let (kcache, vcache) = self
                .fwd
                .kv_cache_handles(li_a)
                .ok_or("attention layer without KV caches")?;
            host_k.fill(0.0);
            host_v.fill(0.0);
            for (c, &page_id) in plan.kv_pages[l]
                .iter()
                .take(pr_tokens / PAGE_TOKENS)
                .enumerate()
            {
                let page = &mgr.kv_pool().pages[page_id];
                for i in 0..PAGE_TOKENS {
                    let tok = c * PAGE_TOKENS + i;
                    let k_off = i * self.kvd;
                    let v_off = PAGE_TOKENS * self.kvd + i * self.kvd;
                    host_k[tok * self.kvd..(tok + 1) * self.kvd]
                        .copy_from_slice(&page[k_off..k_off + self.kvd]);
                    host_v[tok * self.kvd..(tok + 1) * self.kvd]
                        .copy_from_slice(&page[v_off..v_off + self.kvd]);
                }
            }
            // Owned-Bytes writes: the host buffers may be reused on the
            // next layer before the enqueue retires.
            client.write(
                &staging_k,
                Bytes::from_bytes_vec(bytemuck::cast_slice(&host_k).to_vec()),
            );
            client.write(
                &staging_v,
                Bytes::from_bytes_vec(bytemuck::cast_slice(&host_v).to_vec()),
            );
            unsafe {
                QwenKvCacheFillSplitBatchedCubeCL::launch(
                    &client,
                    staging_k.clone(),
                    staging_v.clone(),
                    kcache,
                    vcache,
                    self.kvd,
                    pr_tokens,
                    0,
                );
            }
        }
        // The stagings must outlive their launches: sync before they drop.
        pollster::block_on(client.sync()).ok();

        // GDN: the floor boundary's pages — the state after exactly
        // pr_tokens tokens = chunk pr_tokens/PAGE_TOKENS − 1.
        let boundary_chunk = pr_tokens / PAGE_TOKENS - 1;
        for (gl, row) in plan.gdn_pages.iter().enumerate() {
            let li = self.gdn_ids[gl];
            let page = mgr.gdn_pool().page(row[boundary_chunk]);
            self.fwd
                .write_layer_state(li, StateKind::Gdn, bytemuck::cast_slice(&page.recurrent));
            self.fwd
                .write_layer_state(li, StateKind::Conv, bytemuck::cast_slice(&page.conv));
        }
        pollster::block_on(client.sync()).ok();
        Ok(pr_tokens)
    }

    /// A1 capture finalize: sync the boundary arena, read it back, fill
    /// the fresh GDN pages (slot = `(c+1)·PAGE_TOKENS / chunk`; slot 0 is
    /// the cold-stream state — only reachable on a fresh walk's chunk 0),
    /// capture the fresh KV rows (range-copy → read → `write_kv`), and
    /// commit. Returns the aligned capture length.
    fn a1_capture_finalize(
        &self,
        mgr: &mut FlashStatesManager,
        session: u64,
        chain: &[u32],
        sink: &mut Option<DriveCapture>,
    ) -> Result<usize, String> {
        let client = self.fwd.client().clone();
        let aligned = chain.len() / PAGE_TOKENS * PAGE_TOKENS;
        let total_chunks = aligned / PAGE_TOKENS;
        let cp = mgr.plan_capture(session, aligned);
        let fresh = total_chunks - cp.held_chunks;
        if fresh == 0 {
            mgr.commit_capture(session, &chain[..aligned], &cp)
                .map_err(|e| format!("commit_capture: {e:?}"))?;
            return Ok(aligned);
        }
        let Some(cap) = sink.as_ref() else {
            return Err("a1 capture without a sink".into());
        };
        pollster::block_on(cap.client.sync()).ok();
        let raw_g = match cap.client.read_one(cap.arena_gdn.clone()) {
            Ok(b) => b.to_vec(),
            Err(_) => Vec::new(),
        };
        let raw_c = match cap.client.read_one(cap.arena_conv.clone()) {
            Ok(b) => b.to_vec(),
            Err(_) => Vec::new(),
        };
        if raw_g.is_empty() || raw_c.is_empty() {
            return Err("a1 capture: arena readback failed".into());
        }
        let arena_g: &[f32] = bytemuck::cast_slice(&raw_g);
        let arena_c: &[f32] = bytemuck::cast_slice(&raw_c);
        let lo = cap.slot_lo;

        // GDN pages. slot >= lo for every fresh chunk (slots are
        // (c+1)*PAGE_TOKENS/chunk over c >= held); slot 0 = the cold
        // state (only reachable when lo == 0).
        for c in cp.held_chunks..total_chunks {
            let slot = (c + 1) * PAGE_TOKENS / self.chunk;
            let idx = slot - lo;
            for gl in 0..self.n_gdn {
                let id = cp.fresh_gdn_ids[gl][c - cp.held_chunks];
                let (rec, conv): (Vec<f32>, Vec<f32>) = if slot == 0 {
                    (
                        bytemuck::cast_slice(&self.cold_gdn_bytes[gl]).to_vec(),
                        bytemuck::cast_slice(&self.cold_conv_bytes[gl]).to_vec(),
                    )
                } else {
                    let off_g = (idx * self.n_gdn + gl) * self.gdn_elems;
                    let off_c = (idx * self.n_gdn + gl) * self.conv_elems;
                    (
                        arena_g[off_g..off_g + self.gdn_elems].to_vec(),
                        arena_c[off_c..off_c + self.conv_elems].to_vec(),
                    )
                };
                mgr.gdn_pool_mut().write(id, &rec, &conv);
            }
        }

        // KV rows [held_row .. aligned).
        let fresh_rows = fresh * PAGE_TOKENS;
        let held_row = cp.held_chunks * PAGE_TOKENS;
        let elem = fresh_rows * self.kvd;
        let staging_k = client.create_from_slice(&vec![0u8; elem * 4]);
        let staging_v = client.create_from_slice(&vec![0u8; elem * 4]);
        let byte_off = held_row as u64 * self.kvd as u64 * 4;
        for (l, &li_a) in self.attn_ids.iter().enumerate() {
            let (kcache, vcache) = self
                .fwd
                .kv_cache_handles(li_a)
                .ok_or("attention layer without KV caches")?;
            unsafe {
                copy_range_f32_launch(
                    &client,
                    kcache.offset_start(byte_off),
                    staging_k.clone(),
                    elem,
                );
                copy_range_f32_launch(
                    &client,
                    vcache.offset_start(byte_off),
                    staging_v.clone(),
                    elem,
                );
            }
            pollster::block_on(client.sync()).ok();
            let kb = match client.read_one(staging_k.clone()) {
                Ok(b) => b.to_vec(),
                Err(_) => Vec::new(),
            };
            let vb = match client.read_one(staging_v.clone()) {
                Ok(b) => b.to_vec(),
                Err(_) => Vec::new(),
            };
            if kb.len() < elem * 4 || vb.len() < elem * 4 {
                return Err("a1 capture: KV staging readback failed".into());
            }
            let kf: &[f32] = bytemuck::cast_slice(&kb);
            let vf: &[f32] = bytemuck::cast_slice(&vb);
            // fresh_kv_ids is the capture slot's FULL table (total chunks);
            // the fresh rows start at held_chunks — skip the held prefix
            // (the staging buffer carries fresh_rows only).
            for (fi, _page_id) in cp.fresh_kv_ids[l].iter().skip(cp.held_chunks).enumerate() {
                for t in 0..PAGE_TOKENS {
                    let pos = held_row + fi * PAGE_TOKENS + t;
                    let local = fi * PAGE_TOKENS + t;
                    mgr.kv_pool_mut().write_kv(
                        l,
                        cp.capture_slot,
                        pos,
                        &kf[local * self.kvd..(local + 1) * self.kvd],
                        &vf[local * self.kvd..(local + 1) * self.kvd],
                    );
                }
            }
        }
        // The stagings must outlive their launches (each layer's readback
        // already synced; this covers the final pair before they drop).
        pollster::block_on(client.sync()).ok();

        mgr.commit_capture(session, &chain[..aligned], &cp)
            .map_err(|e| format!("commit_capture: {e:?}"))?;
        Ok(aligned)
    }
}

// ---------------------------------------------------------------------------
// arms

fn run_a0(
    cell: &mut Cell,
    trace: &Trace,
    requests: &[RequestRef<'_>],
    dir: &Path,
    resume: bool,
) -> Result<Stats, String> {
    let mut log = ArmLog::open(dir, "a0", resume);
    let mut rows = read_rows(&log.path, log.next_req);
    let chunk = cell.chunk;
    for (req_idx, r) in requests.iter().enumerate() {
        if req_idx < log.next_req {
            continue;
        }
        let cont = &trace.sessions[r.session].turns[r.turn].continuation;
        let t0 = Instant::now();
        cell.fwd.zero_state();
        let mut sink: Option<DriveCapture> = None;
        let arg = drive(&mut cell.fwd, r.prompt, 0, chunk, &mut sink)?;
        let ttft_ms = t0.elapsed().as_secs_f64() * 1e3;
        let t1 = Instant::now();
        drive_decode(&mut cell.fwd, cont, r.prompt.len(), &mut sink)?;
        let decode_ms = t1.elapsed().as_secs_f64() * 1e3;
        let row = Row {
            req: req_idx,
            session: r.session,
            turn: r.turn,
            prompt_len: r.prompt.len(),
            restored: 0,
            uncached: r.prompt.len(),
            ttft_ms,
            decode_ms,
            decode_tokens: cont.len(),
            first_token: arg,
        };
        println!(
            "[a0] req {req_idx} s{} t{} uncached={} ttft={ttft_ms:.0}ms",
            r.session, r.turn, row.uncached
        );
        log.record(&row);
        rows.push(row);
    }
    let s = stats_of(&rows, 0);
    log.complete(dir, "a0");
    Ok(s)
}

fn run_a0p(
    cell: &mut Cell,
    trace: &Trace,
    requests: &[RequestRef<'_>],
    dir: &Path,
    resume: bool,
    g1_set: &HashSet<usize>,
) -> Result<Stats, String> {
    let mut caches: Vec<PrefixCacheCubecl> = (0..trace.n_sessions)
        .map(|_| PrefixCacheCubecl::new())
        .collect();
    let mut log = ArmLog::open(dir, "a0p", resume);
    let mut rows = read_rows(&log.path, log.next_req);
    let chunk = cell.chunk;
    let mut g1_checked = 0usize;
    for (req_idx, r) in requests.iter().enumerate() {
        if req_idx < log.next_req {
            continue;
        }
        let cont = &trace.sessions[r.session].turns[r.turn].continuation;
        let t0 = Instant::now();
        let mut sink: Option<DriveCapture> = None;
        let restore_len = caches[r.session].restore(&mut cell.fwd, r.prompt);
        if restore_len == 0 {
            cell.fwd.zero_state();
        }
        let arg = drive(
            &mut cell.fwd,
            &r.prompt[restore_len..],
            restore_len,
            chunk,
            &mut sink,
        )?;
        let ttft_ms = t0.elapsed().as_secs_f64() * 1e3;
        let t1 = Instant::now();
        drive_decode(&mut cell.fwd, cont, r.prompt.len(), &mut sink)?;
        let decode_ms = t1.elapsed().as_secs_f64() * 1e3;
        // Anchor at the turn end (the cell's policy): the snapshot covers
        // exactly prompt + continuation — the next turn's prompt prefix.
        let mut chain = r.prompt.to_vec();
        chain.extend_from_slice(cont);
        caches[r.session].note_write(chain.len());
        caches[r.session].insert(&cell.fwd, &chain);
        if g1_set.contains(&req_idx) && restore_len > 0 {
            let rel = caches[r.session].restore(&mut cell.fwd, r.prompt);
            if rel == 0 {
                cell.fwd.zero_state();
            }
            compare_walks(&mut cell.fwd, r.prompt, cont, chunk, rel)?;
            g1_checked += 1;
        }
        let row = Row {
            req: req_idx,
            session: r.session,
            turn: r.turn,
            prompt_len: r.prompt.len(),
            restored: restore_len,
            uncached: r.prompt.len() - restore_len,
            ttft_ms,
            decode_ms,
            decode_tokens: cont.len(),
            first_token: arg,
        };
        println!(
            "[a0p] req {req_idx} s{} t{} restored={restore_len} uncached={} ttft={ttft_ms:.0}ms",
            r.session, r.turn, row.uncached
        );
        log.record(&row);
        rows.push(row);
    }
    let s = stats_of(&rows, g1_checked);
    log.complete(dir, "a0p");
    Ok(s)
}

/// The radix-arm posture (a1/a2 share the runner; the flag picks the
/// victim policy, the budget arms eviction).
struct RadixArmPosture {
    arm: &'static str,
    gds: bool,
    budget_pages: Option<usize>,
}

fn run_a1(
    cell: &mut Cell,
    trace: &Trace,
    requests: &[RequestRef<'_>],
    dir: &Path,
    resume: bool,
    g1_set: &HashSet<usize>,
    posture: RadixArmPosture,
) -> Result<Stats, String> {
    let RadixArmPosture {
        arm,
        gds,
        budget_pages,
    } = posture;
    let plan = cell.layer_plan();
    let mut kc = KatConfig::micro();
    kc.n_layer = plan.n_attn;
    kc.block_size = cell.fwd.max_ctx();
    kc.n_kv_head = cell.fwd.config.n_kv_head;
    kc.head_dim = cell.fwd.config.head_dim;
    let kv = PagedKVCache::new(&kc, 65536);
    let tree_budget = budget_pages.unwrap_or((usize::MAX / 4).saturating_mul(plan.n_layers()));
    let tree = RadixPrefixTree::new(plan.n_layers(), PAGE_TOKENS, tree_budget);
    let mut mgr = FlashStatesManager::new(tree, kv, GdnStatePool::new(), plan);
    if gds {
        mgr = mgr.with_gds_retention();
    }
    let mut log = ArmLog::open(dir, arm, resume);
    let mut rows = read_rows(&log.path, log.next_req);
    let chunk = cell.chunk;
    let turns_total: Vec<usize> = trace.sessions.iter().map(|s| s.turns.len()).collect();
    let mut g1_checked = 0usize;
    // Sessions register on their first SERVED request of this process —
    // a resumed run rebuilds the manager empty (the tree/pools are not
    // persisted), so a request past the cursor is a cold first serve for
    // its session even mid-trace.
    let mut started: HashSet<u64> = HashSet::new();
    for (req_idx, r) in requests.iter().enumerate() {
        if req_idx < log.next_req {
            continue;
        }
        let sid = r.session as u64;
        if started.insert(sid) {
            mgr.session_start(sid);
        }
        let cont = &trace.sessions[r.session].turns[r.turn].continuation;
        let chain_len = r.prompt.len() + cont.len();
        let t0 = Instant::now();
        let rp = mgr.serve(sid, r.prompt);
        let pr = cell.a1_restore(&mgr, &rp, r.prompt.len())?;
        // The fresh capture span gates the boundary captures: slots are
        // `(c+1)·PAGE_TOKENS / chunk` over fresh chunks `c`.
        let held = mgr.session_held_chunks(sid);
        let aligned = chain_len / PAGE_TOKENS * PAGE_TOKENS;
        let slot_lo = (held + 1) * PAGE_TOKENS / chunk;
        let slot_hi = aligned / chunk;
        let span = slot_hi - slot_lo + 1;
        let client = cell.fwd.client().clone();
        let mut gdn_handles = Vec::with_capacity(cell.n_gdn);
        for gl in 0..cell.n_gdn {
            let li = cell.gdn_ids[gl];
            let g = cell.fwd.gdn_state_handle(li);
            let c = cell.fwd.conv_state_handle(li);
            if g.is_none() || c.is_none() {
                return Err(format!(
                    "a1 sink gather: layer {li} (gl {gl}, n_attn {}, n_gdn {}, types {}) has no state slot",
                    cell.n_attn,
                    cell.n_gdn,
                    cell.fwd.config.layer_types.len()
                ));
            }
            gdn_handles.push((g.expect("checked"), c.expect("checked")));
        }
        let arena_elems = span * cell.n_gdn * cell.gdn_elems;
        let arena_conv_elems = span * cell.n_gdn * cell.conv_elems;
        let mut sink = Some(DriveCapture {
            arena_gdn: client.create_from_slice(&vec![0u8; arena_elems * 4]),
            arena_conv: client.create_from_slice(&vec![0u8; arena_conv_elems * 4]),
            client,
            gdn_handles,
            n_gdn: cell.n_gdn,
            gdn_elems: cell.gdn_elems,
            conv_elems: cell.conv_elems,
            chunk,
            slot_lo,
            slot_hi,
        });
        let arg = drive(&mut cell.fwd, &r.prompt[pr..], pr, chunk, &mut sink)?;
        let ttft_ms = t0.elapsed().as_secs_f64() * 1e3;
        let t1 = Instant::now();
        drive_decode(&mut cell.fwd, cont, r.prompt.len(), &mut sink)?;
        let decode_ms = t1.elapsed().as_secs_f64() * 1e3;
        let mut chain = r.prompt.to_vec();
        chain.extend_from_slice(cont);
        cell.a1_capture_finalize(&mut mgr, sid, &chain, &mut sink)?;
        // Eviction to the pinned shared-state budget (plan 630 bar 5):
        // fires after the capture commits, BEFORE the G1 sample — the deep
        // check then covers the POST-EVICTION resume (the budgeted serving
        // shape, truncation floor included). Infinite budget (frac 0) →
        // never fires → byte-identical to the T2 mechanics posture.
        mgr.evict_to_budget();
        if g1_set.contains(&req_idx) {
            let rg = mgr.serve(sid, r.prompt);
            let prg = cell.a1_restore(&mgr, &rg, r.prompt.len())?;
            compare_walks(&mut cell.fwd, r.prompt, cont, chunk, prg)?;
            g1_checked += 1;
        }
        if r.turn + 1 == turns_total[r.session] {
            mgr.session_end(sid);
        }
        let row = Row {
            req: req_idx,
            session: r.session,
            turn: r.turn,
            prompt_len: r.prompt.len(),
            restored: pr,
            uncached: r.prompt.len() - pr,
            ttft_ms,
            decode_ms,
            decode_tokens: cont.len(),
            first_token: arg,
        };
        println!(
            "[{arm}] req {req_idx} s{} t{} restored={pr} uncached={} ttft={ttft_ms:.0}ms",
            r.session, r.turn, row.uncached
        );
        log.record(&row);
        rows.push(row);
    }
    let s = stats_of(&rows, g1_checked);
    log.complete(dir, arm);
    Ok(s)
}

fn print_bars(dir: &Path, n_sessions: usize, sums: &HashMap<String, Stats>) {
    let (Some(a0), Some(a1)) = (sums.get("a0"), sums.get("a1")) else {
        println!("[bars] a0+a1 summaries required — skipped");
        return;
    };
    let applicable = n_sessions >= 16;
    let v = |pass: bool| {
        if !applicable {
            "N-A (N<16)"
        } else if pass {
            "PASS"
        } else {
            "MISS"
        }
    };
    let ratio = |arm: &Stats| arm.median_uncached / a0.median_uncached.max(1e-9);
    let ratio_unc = ratio(a1);
    let ratio_ttft = a1.ttft_p90 / a0.ttft_p90.max(1e-9);
    let ratio_dec = a1.decode_med / a0.decode_med.max(1e-9);
    let b1 = ratio_unc <= 0.20;
    let b2 = ratio_ttft <= 0.50;
    let b3 = ratio_dec >= 0.98;
    println!(
        "[bars] a1/a0 ratios — uncached={ratio_unc:.3} ttft_p90={ratio_ttft:.3} decode={ratio_dec:.3} (N={n_sessions})"
    );
    println!(
        "[bar 1] a1 uncached ≤ 20% of a0: {} (ratio {ratio_unc:.3})",
        v(b1)
    );
    println!(
        "[bar 2] a1 p90 TTFT ≤ 50% of a0: {} (ratio {ratio_ttft:.3})",
        v(b2)
    );
    println!(
        "[bar 3] a1 decode ≥ 98% of a0: {} (ratio {ratio_dec:.3})",
        v(b3)
    );
    if let Some(a0p) = sums.get("a0p") {
        println!(
            "[bar 4 attribution] a0p median_uncached={:.0} vs a1 {:.0} (a1/a0p {:.3}) — the control, never the denominator",
            a0p.median_uncached,
            a1.median_uncached,
            a1.median_uncached / a0p.median_uncached.max(1e-9)
        );
        if applicable && b1 && a1.median_uncached <= 1.10 * a0p.median_uncached {
            println!(
                "[bar 4 attribution] DISCLOSURE: a1 ≈ a0p on uncached tokens — the measured gain is the per-session-cache composition, not the multi-session lane; the lane's residual value claim rests on the TTFT/decode axes alone"
            );
        }
    }
    if a1.g1_checked > 0 {
        println!(
            "[bar 6] G1 resumed-vs-full bit-identity (a1): PASS on {} sampled requests (drift aborts the run)",
            a1.g1_checked
        );
    } else {
        println!("[bar 6] G1 (a1): no sampled checks ran (enable with --g1-sample)");
    }
    let mut bars = serde_json::json!({
        "n_sessions": n_sessions,
        "a1_ratio_uncached": ratio_unc,
        "a1_ratio_ttft_p90": ratio_ttft,
        "a1_ratio_decode": ratio_dec,
        "a1_bar1_pass": applicable && b1,
        "a1_bar2_pass": applicable && b2,
        "a1_bar3_pass": applicable && b3,
        "bar_applicable": applicable,
        "a1_g1_checked": a1.g1_checked,
    });
    if let Some(a2) = sums.get("a2") {
        let r_unc = ratio(a2);
        let r_ttft = a2.ttft_p90 / a0.ttft_p90.max(1e-9);
        let r_dec = a2.decode_med / a0.decode_med.max(1e-9);
        println!(
            "[bars] a2/a0 ratios — uncached={r_unc:.3} ttft_p90={r_ttft:.3} decode={r_dec:.3}"
        );
        println!(
            "[bar 1] a2 uncached ≤ 20% of a0: {} (ratio {r_unc:.3})",
            v(r_unc <= 0.20)
        );
        println!(
            "[bar 2] a2 p90 TTFT ≤ 50% of a0: {} (ratio {r_ttft:.3})",
            v(r_ttft <= 0.50)
        );
        println!(
            "[bar 3] a2 decode ≥ 98% of a0: {} (ratio {r_dec:.3})",
            v(r_dec >= 0.98)
        );
        // Bar 5 — A2's own axis (retention at the SAME pinned budget):
        let b5 = a2.median_uncached <= a1.median_uncached;
        println!(
            "[bar 5] a2 median_uncached ≤ a1 at the same budget: {} (a1 {:.0} vs a2 {:.0})",
            v(b5),
            a1.median_uncached,
            a2.median_uncached
        );
        if a2.g1_checked > 0 {
            println!(
                "[bar 6] G1 resumed-vs-full bit-identity (a2): PASS on {} sampled requests",
                a2.g1_checked
            );
        }
        bars["a2_ratio_uncached"] = serde_json::json!(r_unc);
        bars["a2_ratio_ttft_p90"] = serde_json::json!(r_ttft);
        bars["a2_ratio_decode"] = serde_json::json!(r_dec);
        bars["a2_bar1_pass"] = serde_json::json!(applicable && r_unc <= 0.20);
        bars["a2_bar2_pass"] = serde_json::json!(applicable && r_ttft <= 0.50);
        bars["a2_bar3_pass"] = serde_json::json!(applicable && r_dec >= 0.98);
        bars["a2_bar5_retention_pass"] = serde_json::json!(applicable && b5);
        bars["a2_g1_checked"] = serde_json::json!(a2.g1_checked);
    }
    fs::write(
        dir.join("bars.json"),
        serde_json::to_string_pretty(&bars).unwrap() + "\n",
    )
    .ok();
}

fn main() {
    let args = parse_args();
    for a in &args.arms {
        match a.as_str() {
            "a0" | "a0p" | "a1" | "a2" => {}
            other => {
                eprintln!("unknown arm {other} (--help)");
                std::process::exit(2);
            }
        }
    }
    let trace = flashstates_tracegen::generate(args.n, args.cap, args.seed);
    let mut requests = trace.requests();
    if let Some(k) = args.max_requests {
        requests.truncate(k);
    }
    let dir = args.out_dir.clone().unwrap_or_else(|| {
        PathBuf::from(format!(
            ".benchmarks/flashstates_cell/{}_{}_s{}",
            args.n, args.cap, args.seed
        ))
    });
    fs::create_dir_all(&dir).expect("out dir");
    let manifest_path = dir.join("manifest.json");
    if manifest_path.exists() {
        if !args.resume {
            eprintln!(
                "cell dir {} already exists — pass --resume to continue it",
                dir.display()
            );
            std::process::exit(2);
        }
        let m: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read manifest"))
                .expect("parse manifest");
        if m["trace_digest"].as_str() != Some(trace.digest.as_str()) {
            eprintln!(
                "trace digest mismatch: manifest {} vs generated {} — refusing (--resume)",
                m["trace_digest"].as_str().unwrap_or("?"),
                trace.digest
            );
            std::process::exit(2);
        }
    } else {
        let m = serde_json::json!({
            "trace_digest": trace.digest,
            "n_sessions": args.n,
            "transcript_cap_tokens": args.cap,
            "seed": args.seed,
            "gguf": args.gguf,
            "ctx": args.ctx,
            "chunk": args.chunk,
            "budget_frac": args.budget_frac,
        });
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&m).unwrap() + "\n",
        )
        .expect("write manifest");
        println!("cell: {} — {}", dir.display(), trace);
    }
    let mut cell = match Cell::boot(&args.gguf, args.ctx, args.chunk) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("boot failed: {e}");
            std::process::exit(1);
        }
    };
    cell.print_page_economics();
    // Warmup: compile the Metal pipelines once so arm 0 pays no compile
    // cost (an arm-order bias otherwise).
    {
        let mut warm = trace.preamble.clone();
        warm.truncate(cell.chunk);
        cell.fwd.zero_state();
        let mut sink: Option<DriveCapture> = None;
        drive(&mut cell.fwd, &warm, 0, cell.chunk, &mut sink).expect("warmup drive");
        println!("[warmup] {} tokens driven", warm.len());
    }
    // G1 sample set: turn 0 of every session + every 64th request.
    let mut g1_list: Vec<usize> = Vec::new();
    for (i, r) in requests.iter().enumerate() {
        if r.turn == 0 || i % 64 == 0 {
            g1_list.push(i);
        }
    }
    let sample_cap = args.g1_sample.unwrap_or(8);
    if sample_cap != usize::MAX {
        g1_list.truncate(sample_cap);
    }
    let g1_set: HashSet<usize> = g1_list.into_iter().collect();
    // The pinned shared-state budget (plan 630 bar 5): frac × the trace's
    // total demand — every session at its transcript cap → chunks ×
    // n_layers pages. Floor of one chunk so a nonzero frac always evicts.
    let n_layers = cell.n_attn + cell.n_gdn;
    let demand_pages = trace.n_sessions * (args.cap / PAGE_TOKENS) * n_layers;
    let budget_pages = if args.budget_frac > 0.0 {
        let pages = ((demand_pages as f64 * args.budget_frac) as usize).max(n_layers);
        println!(
            "[cell] shared-state budget: frac {} × demand {} pages → {} pages ({} chunks/tree-index axis; a1+a2 bounded alike)",
            args.budget_frac,
            demand_pages,
            pages,
            pages / n_layers
        );
        Some(pages)
    } else {
        println!(
            "[cell] shared-state budget: none (frac 0 — the T2 infinite posture, no eviction)"
        );
        None
    };
    let mut sums: HashMap<String, Stats> = HashMap::with_capacity(args.arms.len());
    for arm in &args.arms {
        let complete = dir.join(format!("{arm}.COMPLETE")).exists();
        if complete && !args.resume {
            eprintln!("{arm}: COMPLETE present without --resume — pass --resume");
            std::process::exit(2);
        }
        if complete {
            let rows = read_rows(&dir.join(format!("{arm}.rows.jsonl")), usize::MAX);
            if rows.is_empty() {
                eprintln!("{arm}: COMPLETE but no rows — refusing");
                std::process::exit(2);
            }
            let s = stats_of(&rows, 0);
            print_stats(arm, &s);
            sums.insert(arm.clone(), s);
            continue;
        }
        let s = match arm.as_str() {
            "a0" => run_a0(&mut cell, &trace, &requests, &dir, args.resume),
            "a0p" => run_a0p(&mut cell, &trace, &requests, &dir, args.resume, &g1_set),
            "a1" => run_a1(
                &mut cell,
                &trace,
                &requests,
                &dir,
                args.resume,
                &g1_set,
                RadixArmPosture {
                    arm: "a1",
                    gds: false,
                    budget_pages,
                },
            ),
            "a2" => run_a1(
                &mut cell,
                &trace,
                &requests,
                &dir,
                args.resume,
                &g1_set,
                RadixArmPosture {
                    arm: "a2",
                    gds: true,
                    budget_pages,
                },
            ),
            _ => unreachable!("validated above"),
        };
        let s = match s {
            Ok(s) => s,
            Err(e) => {
                eprintln!("arm {arm} failed: {e}");
                std::process::exit(1);
            }
        };
        print_stats(arm, &s);
        let v = stats_value(arm, &s);
        fs::write(
            dir.join(format!("{arm}.summary.json")),
            serde_json::to_string_pretty(&v).unwrap() + "\n",
        )
        .ok();
        sums.insert(arm.clone(), s);
    }
    print_bars(&dir, trace.n_sessions, &sums);
}
