//! Plan 633 P0.2 — the named prediction cell (NUMERIC, falsifiable,
//! computed BEFORE any conversion). Prints the roofline's own per-call
//! numbers for the OpenThai2.0-ASR decoder⊕joint call shapes and the
//! tier-sweep encoder call shapes, from `katgpt_core::ane_roofline`.
//!
//! Run: cargo run -p riir-infer --example p633_p02_predictions --release
//! (standalone; zero GPU, zero ANE — the roofline is a pure function).

fn main() {
    use katgpt_core::ane_roofline::{ane_estimate, AneOpShape, AnePeaks, Dtype};

    let peaks = AnePeaks::m5();
    println!(
        "AnePeaks::m5(): compute {} TFLOP/s fp16, bw {} GB/s, dispatch floor {} ms, working-set {} bytes",
        peaks.compute_tflops_fp16, peaks.bandwidth_gbs, peaks.dispatch_floor_ms, peaks.working_set_bytes
    );

    // ── P0.2's named denominator: the vocab matmul ─────────────────────
    // joint: [640 prednet + 1024 encoder] projected → 640 → vocab 14,989.
    // The plan's shape: 640 × 14,989 fp16 ≈ 19.2 MB (the joint's second
    // projection). Posture A = per-step (batch 1); Posture B = batched
    // over the chunk's frames (tier-dependent batch).
    println!("\n== P0.2 denominator: the joint vocab matmul (640 x 14989, fp16) ==");
    let vocab_rows = 14989u64;
    let joint_dim = 640u64;
    let w_bytes = joint_dim * vocab_rows * 2; // fp16 weights ≈ 19.2 MB
    println!("vocab weight bytes: {w_bytes} ({:.1} MB)", w_bytes as f64 / 1e6);
    for &batch in &[1u64, 8, 28, 56, 112] {
        // activations [batch, 640+1024] → [batch, 640] → [batch, 14989]
        let shp = AneOpShape::gemm(batch, vocab_rows, joint_dim, Dtype::F16);
        let c = ane_estimate(shp, Dtype::F16, &peaks);
        println!(
            "  batch {batch:4}: compute_ms {:.4} memory_ms {:.4} runtime_ms {:.4} bound {:?} (largest operand {:.1} MB)",
            shp.flops as f64 / (peaks.compute_tflops_fp16 * 1e9),
            shp.bytes_moved as f64 / (peaks.bandwidth_gbs * 1e6),
            c.runtime_ms,
            c.bound,
            shp.largest_operand_bytes as f64 / 1e6
        );
    }

    // ── Posture A prediction row: per-step, tier-INVARIANT call shape ──
    println!("\n== Posture A (per-step joint call: 1 encoder frame + 640 state) ==");
    println!("  prediction: per-call p50 FLAT across tiers [56,0]..[56,13] (+/-15%) and >= runtime_ms");
    let shp = AneOpShape::gemm(1, vocab_rows, joint_dim, Dtype::F16);
    let c = ane_estimate(shp, Dtype::F16, &peaks);
    println!("  roofline runtime_ms = {:.4} (dispatch floor dominates at {:.4})", c.runtime_ms, peaks.dispatch_floor_ms);

    // ── Posture B prediction row: batched over the chunk's frames ──────
    // chunk frames per tier: [56,0]=80ms→~7 frames, [56,3]=320ms→~27,
    // [56,6]=560ms→~47, [56,13]=1120ms→~94 (10ms frames, reduced by the
    // encoder's 8x stride to Tenc — the joint sees the pre-stride frame
    // count; the tier's chunk length in frames is the batch).
    println!("\n== Posture B (batched joint over the chunk's frames) ==");
    println!("  prediction: batched p50 >= 3x its own runtime_ms at short tiers, <= 1.5x by [56,13]");
    for (tier, frames) in [("56,0", 7u64), ("56,3", 27), ("56,6", 47), ("56,13", 94)] {
        let shp = AneOpShape::gemm(frames, vocab_rows, joint_dim, Dtype::F16);
        let c = ane_estimate(shp, Dtype::F16, &peaks);
        println!(
            "  tier [{tier}] frames {frames:3}: runtime_ms {:.4} bound {:?} largest {:.1} MB",
            c.runtime_ms,
            c.bound,
            shp.largest_operand_bytes as f64 / 1e6
        );
    }

    // ── the encoder per-tier context rows (the cache matmuls scale) ────
    // The FastConformer attention context grows the K/V cache length:
    // left 56 + right {0,3,6,13} + 1. The per-layer QK/AV matmuls scale
    // with (frames + context). d_model 1024, heads 8? (FastConformer
    // 609M: d 1024, 16 layers estimated from the census's 144 LayerNorm
    // / 72 MatMul groups). The context matmul at tier extremes:
    println!("\n== encoder attention context matmul (d=1024, per layer) ==");
    for (tier, ctx_len) in [("56,0", 57u64), ("56,13", 70)] {
        let shp = AneOpShape::gemm(ctx_len, 1024, 1024, Dtype::F16);
        let c = ane_estimate(shp, Dtype::F16, &peaks);
        println!(
            "  tier [{tier}] ctx {ctx_len}: runtime_ms {:.4} bound {:?}",
            c.runtime_ms, c.bound
        );
    }

    println!("\nP0.2 rows complete — the predictions stand as written in plan 633; the cells above are the denominators the measurements are read against.");
}
