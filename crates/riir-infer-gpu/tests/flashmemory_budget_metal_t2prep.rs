#![cfg(all(
    feature = "flashmemory_budget_metal",
    feature = "cubecl_runtime",
    target_os = "macos"
))]
//! Issue 1191 T2 (prep) — the three Metal FlashMemory budget kernels vs
//! the CPU reference forms, on a small synthetic fixture (the compute
//! law's SHORT snapshot; no model weights, no long CPU cell — the full
//! forward parity rides the T2 integration once the Metal decode forward
//! routes attention layers through this arm).
//!
//! Three parity legs, the CPU side replayed verbatim from
//! `flashmemory_gqa` + `katgpt-attn::flashmemory_sparse`:
//!
//! 1. **Scores** — kernel (a) vs the CPU `select` scoring loop
//!    (`dot(q_kv, centroid) × scale`): tolerance (device vs host fp32
//!    order) with a tight bound.
//! 2. **Selection** — kernel (b) vs the CPU `RowRule::Budget` top-ρ rule:
//!    the SELECTED SET must match exactly (ties: first index — the
//!    determinism law; the CPU's `select_nth_unstable_by` orders equal
//!    scores by ascending index, the kernel's strictly-greater argmax
//!    keeps the first index). Set equality is the gate, not order (the
//!    CPU output is ascending, the kernel's is ascending too, so sequence
//!    equality holds and is asserted).
//! 3. **Gather** — kernel (c) vs the CPU `sparse_attention_accumulate`:
//!    per-head max-abs tolerance (online-tile softmax vs the CPU two-pass
//!    form — tolerance-class across backends by design).

use cubecl::prelude::*;
use cubecl::server::Handle;
use riir_infer_gpu::cubecl_runtime::ActiveComputeClient;
use riir_infer_gpu::cubecl_runtime::ActiveRuntime;
use riir_infer_gpu::flashmemory_budget_metal::{
    FM_MAX_SEL, FmBlockGatherMetal, FmBlockScoreMetal, FmTopRhoSelectMetal,
};

/// Minimal deterministic PRNG (the crate's fixture discipline).
struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn read_f32(client: &ActiveComputeClient, h: &Handle, n: usize) -> Vec<f32> {
    let b = client
        .read_one(h.clone())
        .unwrap_or_else(|e| panic!("read_one failed: {e:?}"));
    bytemuck::cast_slice(&b)[..n].to_vec()
}

fn read_u32(client: &ActiveComputeClient, h: &Handle, n: usize) -> Vec<u32> {
    let b = client
        .read_one(h.clone())
        .unwrap_or_else(|e| panic!("read_u32 failed: {e:?}"));
    bytemuck::cast_slice(&b)[..n].to_vec()
}

/// The CPU budget rule — verbatim from katgpt-attn's `RowRule::Budget`
/// (`select_scored_row`): top `k = max(round(ρ·n), 1) − 1` complete
/// blocks by score (ties: lower index via the stable sort) plus the
/// current block, ascending.
fn cpu_budget_select(scores: &[f32], rho: f32) -> Vec<usize> {
    let n = scores.len();
    let current = n.checked_sub(1).expect("non-empty row");
    let k = (((rho * n as f32).round() as usize).max(1) - 1).min(current);
    let mut rank: Vec<usize> = (0..current).collect();
    if k < current {
        rank.select_nth_unstable_by(k, |&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
        rank.truncate(k);
    }
    rank.sort_unstable();
    rank.push(current);
    rank
}

/// The CPU scorer — verbatim from `GqaFlashMemorySelector::select`'s
/// loop: `dot(q_kv, centroid_block_kv) × scale`.
fn cpu_scores(
    query: &[f32],
    centroids: &[f32],
    n_kv: usize,
    head_dim: usize,
    n_blocks: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0f32; n_kv * n_blocks];
    for kv in 0..n_kv {
        let q = &query[kv * head_dim..(kv + 1) * head_dim];
        for b in 0..n_blocks {
            let c = &centroids
                [b * n_kv * head_dim + kv * head_dim..b * n_kv * head_dim + (kv + 1) * head_dim];
            let mut dot = 0f32;
            for d in 0..head_dim {
                dot += q[d] * c[d];
            }
            out[kv * n_blocks + b] = dot * scale;
        }
    }
    out
}

/// The CPU gather — verbatim from `flashmemory_gqa::sparse_attention_accumulate`
/// (two-pass softmax over the selected blocks' tokens, weighted V in the
/// same ascending token order).
#[allow(clippy::too_many_arguments)]
fn cpu_gather(
    query: &[f32],
    key_cache: &[f32],
    value_cache: &[f32],
    gate: &[f32],
    sel: &[Vec<usize>],
    n_head: usize,
    n_kv: usize,
    hd: usize,
    seq_len: usize,
    block_size: usize,
) -> Vec<f32> {
    let kvd = n_kv * hd;
    let q_dim = n_head * hd;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut attn_out = vec![0f32; q_dim];
    let mut scores = vec![0f32; seq_len.max(1)];
    for q_h in 0..n_head {
        let kv_group = q_h * n_kv / n_head;
        let q_slice = &query[q_h * hd..(q_h + 1) * hd];
        let out_slice = &mut attn_out[q_h * hd..(q_h + 1) * hd];
        out_slice.fill(0.0);

        let mut blocks_to_attend: Vec<usize> = sel[kv_group].clone();
        if blocks_to_attend.is_empty() {
            blocks_to_attend.push((seq_len - 1) / block_size);
        }

        let mut n_scored = 0usize;
        let mut max_score = f32::NEG_INFINITY;
        for &block_idx in &blocks_to_attend {
            let tok_start = block_idx * block_size;
            let tok_end = (tok_start + block_size).min(seq_len);
            for tok in tok_start..tok_end {
                let key_off = tok * kvd + kv_group * hd;
                let k_slice = &key_cache[key_off..key_off + hd];
                let score = k_slice
                    .iter()
                    .zip(q_slice)
                    .map(|(&k, &q)| k * q)
                    .sum::<f32>()
                    * scale;
                scores[n_scored] = score;
                n_scored += 1;
                if score > max_score {
                    max_score = score;
                }
            }
        }

        let mut sum_exp = 0f32;
        for s in scores.iter_mut().take(n_scored) {
            *s = (*s - max_score).exp();
            sum_exp += *s;
        }
        let inv_sum = 1.0 / sum_exp;

        let mut score_idx = 0usize;
        for &block_idx in &blocks_to_attend {
            let tok_start = block_idx * block_size;
            let tok_end = (tok_start + block_size).min(seq_len);
            for tok in tok_start..tok_end {
                let val_off = tok * kvd + kv_group * hd;
                let weight = scores[score_idx] * inv_sum;
                score_idx += 1;
                let v_slice = &value_cache[val_off..val_off + hd];
                for (o, &v) in out_slice.iter_mut().zip(v_slice.iter()) {
                    *o += weight * v;
                }
            }
        }
        // The gate epilogue (the kernel fuses it; the CPU forward applies
        // it after accumulate — same math, so the parity applies it here).
        for i in 0..hd {
            out_slice[i] *= gate_sigmoid(gate[q_h * hd + i]);
        }
    }
    attn_out
}

/// The sigmoid the CPU lane applies to the gate (the standard logistic —
/// the same form the kernel's epilogue computes).
fn gate_sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// The fixture bundle: (query_kv, centroids, key_cache, value_cache, gate, seq_len).
type Fixture = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, usize);

fn fixture(
    n_head: usize,
    n_kv: usize,
    hd: usize,
    n_blocks: usize,
    block_size: usize,
    seed: u64,
) -> Fixture {
    let mut rng = Rng(seed);
    let kvd = n_kv * hd;
    let seq_len = n_blocks * block_size;
    // The KV cache: full rows (every block complete — the budget rule's
    // "complete blocks" domain).
    let key_cache: Vec<f32> = (0..seq_len * kvd)
        .map(|_| (rng.next_f32() - 0.5) * 4.0)
        .collect();
    let value_cache: Vec<f32> = (0..seq_len * kvd)
        .map(|_| (rng.next_f32() - 0.5) * 4.0)
        .collect();
    // The KV-resolution query.
    let query: Vec<f32> = (0..kvd).map(|_| (rng.next_f32() - 0.5) * 2.0).collect();
    // Block centroids: block means (the CPU rebuild's output shape).
    let mut centroids = vec![0f32; n_blocks * kvd];
    for b in 0..n_blocks {
        for h in 0..n_kv {
            let c_off = b * kvd + h * hd;
            let k_off_head = h * hd;
            let mut mean = vec![0f32; hd];
            for tok in (b * block_size)..((b + 1) * block_size) {
                let row = tok * kvd + k_off_head;
                for d in 0..hd {
                    mean[d] += key_cache[row + d];
                }
            }
            for d in 0..hd {
                centroids[c_off + d] = mean[d] / block_size as f32;
            }
        }
    }
    // The gate: sigmoid-spread.
    let gate: Vec<f32> = (0..n_head * hd)
        .map(|_| (rng.next_f32() - 0.5) * 6.0)
        .collect();
    (query, centroids, key_cache, value_cache, gate, seq_len)
}

/// The three-leg parity gate.
#[test]
fn fm_budget_metal_kernels_parity_vs_cpu() {
    let ctx = match riir_infer_gpu::cubecl_runtime::CubeCLContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP: no CubeCL device on this box ({e})");
            return;
        }
    };
    if !ctx.is_metal_adapter() {
        eprintln!("SKIP: the active CubeCL backend is not Metal — the budget arm is Metal-scoped");
        return;
    }
    let client = &ctx.client();

    // Small fixture: 8 query heads / 2 kv groups / head_dim 256 (the
    // Bonsai geometry), 12 blocks of 64 (768 tokens — several refresh
    // periods' worth), ρ 0.25 → k = max(3,1)−1 = 2 complete + current.
    let n_head = 8usize;
    let n_kv = 2usize;
    let hd = 256usize;
    let n_blocks = 12usize;
    let block_size = 64usize;
    let rho = 0.25f32;
    let scale = 1.0 / (hd as f32).sqrt();
    let (query, centroids, key_cache, value_cache, gate, seq_len) =
        fixture(n_head, n_kv, hd, n_blocks, block_size, 0x1191_D00D);

    // ── Leg 1+2: scores + selection ──
    let q_h = client.create_from_slice(f32::as_bytes(&query));
    let c_h = client.create_from_slice(f32::as_bytes(&centroids));
    let s_out = client.empty(n_kv * n_blocks * core::mem::size_of::<f32>());
    unsafe {
        FmBlockScoreMetal::launch::<ActiveRuntime>(
            client,
            q_h.clone(),
            c_h.clone(),
            s_out.clone(),
            n_kv,
            hd,
            n_blocks,
            scale,
        );
    }
    let _ = client.flush();
    let got_scores = read_f32(client, &s_out, n_kv * n_blocks);
    let want_scores = cpu_scores(&query, &centroids, n_kv, hd, n_blocks, scale);
    let mut worst_score = 0f32;
    for (g, w) in got_scores.iter().zip(want_scores.iter()) {
        worst_score = worst_score.max((g - w).abs());
    }
    assert!(
        worst_score < 2e-4,
        "LEG1 score parity: worst |Δ| {worst_score} > 2e-4"
    );

    // Selection at ρ: device vs CPU.
    let sel_out = client.empty(n_kv * FM_MAX_SEL * core::mem::size_of::<u32>());
    let sel_counts = client.empty(n_kv * core::mem::size_of::<u32>());
    unsafe {
        FmTopRhoSelectMetal::launch::<ActiveRuntime>(
            client,
            s_out.clone(),
            sel_out.clone(),
            sel_counts.clone(),
            n_kv,
            n_blocks,
            rho,
        );
    }
    let _ = client.flush();
    let got_counts = read_u32(client, &sel_counts, n_kv);
    let got_sel = read_u32(client, &sel_out, n_kv * FM_MAX_SEL);
    let mut worst_rank_note = String::new();
    for kv in 0..n_kv {
        let row = &got_scores[kv * n_blocks..(kv + 1) * n_blocks];
        let want = cpu_budget_select(row, rho);
        let got_n = got_counts[kv] as usize;
        let got: Vec<usize> = got_sel[kv * FM_MAX_SEL..kv * FM_MAX_SEL + got_n]
            .iter()
            .map(|&x| x as usize)
            .collect();
        if got != want {
            worst_rank_note = format!("kv{kv}: got {got:?} want {want:?} (scores {row:?})");
        }
    }
    assert!(
        worst_rank_note.is_empty(),
        "LEG2 selection parity MISMATCH: {worst_rank_note}"
    );

    // Tie rule spot-check: duplicate scores must pick the FIRST index.
    {
        let tied = vec![0.5f32; n_blocks];
        let s_h = client.create_from_slice(f32::as_bytes(&tied));
        let so = client.empty(n_kv * FM_MAX_SEL * core::mem::size_of::<u32>());
        let sc = client.empty(n_kv * core::mem::size_of::<u32>());
        unsafe {
            FmTopRhoSelectMetal::launch::<ActiveRuntime>(
                client,
                s_h.clone(),
                so.clone(),
                sc.clone(),
                n_kv,
                n_blocks,
                rho,
            );
        }
        let _ = client.flush();
        let counts = read_u32(client, &sc, n_kv);
        let sel = read_u32(client, &so, n_kv * FM_MAX_SEL);
        // All-equal scores: the first (k) complete indices 0..k must win
        // (ties → first index), plus the current block.
        for kv in 0..n_kv {
            let n = counts[kv] as usize;
            let picked: Vec<usize> = sel[kv * FM_MAX_SEL..kv * FM_MAX_SEL + n]
                .iter()
                .map(|&x| x as usize)
                .collect();
            let mut want: Vec<usize> = (0..n - 1).collect();
            want.push(n_blocks - 1);
            assert_eq!(
                picked, want,
                "TIE RULE: all-equal scores must select the FIRST indices"
            );
        }
    }

    // ── Leg 3: gather ──
    // Device-side selection buffers already hold the ρ = 0.25 verdict;
    // run the gather on them and compare vs the CPU gather on the same
    // sets.
    let q_heads_h = client.empty(n_head * hd * core::mem::size_of::<f32>());
    // Expand the kv-resolution query per head? NO — the CPU lane's
    // gather reads the PER-HEAD queries (post-norm post-RoPE), not the
    // averaged kv-resolution one. Fixture: per-head queries derived
    // deterministically (the averaged query ± a per-head delta).
    let mut q_heads = vec![0f32; n_head * hd];
    for q_h in 0..n_head {
        let kv_group = q_h * n_kv / n_head;
        for d in 0..hd {
            let jitter = ((q_h * hd + d) % 7) as f32 * 1e-3;
            q_heads[q_h * hd + d] = query[kv_group * hd + d] + jitter;
        }
    }
    client.write(
        &q_heads_h,
        cubecl::bytes::Bytes::from_bytes_vec(f32::as_bytes(&q_heads).to_vec()),
    );
    let kc_h = client.create_from_slice(f32::as_bytes(&key_cache));
    let vc_h = client.create_from_slice(f32::as_bytes(&value_cache));
    let g_h = client.create_from_slice(f32::as_bytes(&gate));
    let ao_h = client.empty(n_head * hd * core::mem::size_of::<f32>());
    unsafe {
        FmBlockGatherMetal::launch::<ActiveRuntime>(
            client,
            q_heads_h.clone(),
            kc_h.clone(),
            vc_h.clone(),
            g_h.clone(),
            sel_out.clone(),
            sel_counts.clone(),
            ao_h.clone(),
            n_head,
            n_kv,
            hd,
            seq_len,
            block_size,
        );
    }
    let _ = client.flush();
    let got_gather = read_f32(client, &ao_h, n_head * hd);

    // The CPU gather over the SAME device-selected sets.
    let mut sel_sets: Vec<Vec<usize>> = Vec::with_capacity(n_kv);
    for kv in 0..n_kv {
        let n = got_counts[kv] as usize;
        sel_sets.push(
            got_sel[kv * FM_MAX_SEL..kv * FM_MAX_SEL + n]
                .iter()
                .map(|&x| x as usize)
                .collect(),
        );
    }
    let want_gather = cpu_gather(
        &q_heads,
        &key_cache,
        &value_cache,
        &gate,
        &sel_sets,
        n_head,
        n_kv,
        hd,
        seq_len,
        block_size,
    );
    let mut worst_gather = 0f32;
    for (g, w) in got_gather.iter().zip(want_gather.iter()) {
        worst_gather = worst_gather.max((g - w).abs());
    }
    // Tolerance class: online-tile softmax (device) vs the two-pass form
    // (CPU) over ~672 selected tokens — 5e-4 of the output scale.
    assert!(
        worst_gather < 5e-4,
        "LEG3 gather parity: worst |Δ| {worst_gather} > 5e-4 (tolerance class, online vs two-pass softmax)"
    );

    // Liveness: the lane is not a silent no-op — the sparse output must
    // differ from a dense (all-blocks) gather somewhere. Dense = select
    // every block via ρ = 1.0.
    let so2 = client.empty(n_kv * FM_MAX_SEL * core::mem::size_of::<u32>());
    let sc2 = client.empty(n_kv * core::mem::size_of::<u32>());
    unsafe {
        FmTopRhoSelectMetal::launch::<ActiveRuntime>(
            client,
            s_out.clone(),
            so2.clone(),
            sc2.clone(),
            n_kv,
            n_blocks,
            1.0,
        );
    }
    let _ = client.flush();
    let ao2 = client.empty(n_head * hd * core::mem::size_of::<f32>());
    unsafe {
        FmBlockGatherMetal::launch::<ActiveRuntime>(
            client,
            q_heads_h.clone(),
            kc_h.clone(),
            vc_h.clone(),
            g_h.clone(),
            so2.clone(),
            sc2.clone(),
            ao2.clone(),
            n_head,
            n_kv,
            hd,
            seq_len,
            block_size,
        );
    }
    let _ = client.flush();
    let dense_gather = read_f32(client, &ao2, n_head * hd);
    let max_delta = got_gather
        .iter()
        .zip(dense_gather.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_delta > 1e-5,
        "LIVENESS: the ρ=0.25 gather must differ from the all-blocks gather somewhere, max |Δ| {max_delta}"
    );
}
