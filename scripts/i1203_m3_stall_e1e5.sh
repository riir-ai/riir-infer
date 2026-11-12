#!/bin/sh
# Issue 1203 — the m3 qwen38-dense stall: the E1–E5 GPU-window runner.
#
# Runs ONLY after the 1191 v2 128K gate has exited (the box must be quiet —
# the v2 walls and these cells both need exclusive GPU). Everything isolates
# its target dir; every posture prints its engagement banner and the script
# greps it (the Issue-1124 engagement-proof law — the E1 adjudicator).
#
# Experiments (each tagged in the output, continue-on-fail, verdicts at end):
#   E1  engagement proof: pp512 armed vs demoted on a metal_tensor_gemm build.
#       The stderr banners ([Issue 1150] armed / [Issue 1176] DEMOTED +
#       relaxed_precision=<bool>) decide whether the old 9-vs-1 evidence even
#       measured the armed path. ~100x reproduced WITH the armed banner = real
#       anomaly; old evidence lacking the banner = the feature-less-build trap,
#       issue closes as such.
#   E2  per-launch microbench: NOT automated here (harness shape unpinned) —
#       SKIP-LOUD with the manual recipe.
#   E3  the 761 dispatch counters: needs the probe bin (skipped loud until it
#       lands — a tiny bin reading the vendored fork's pub statics).
#   E4  decode-ablate residual wall (the ALL cell's residual = sync/host floor).
#   E5  is the human verdict — the script only assembles the evidence rows.
#
# Usage: scripts/i1203_m3_stall_e1e5.sh [--skip-e4]
# Report: .benchmarks scratch at ~/riir-bench/i1203_m3_stall/ (never a repo
# write); the record lands in .benchmarks/ by the lane, not this script.
set -u
RIG_MARK="$HOME/riir-bench/i1203_m3_stall"
GGUF="$HOME/uzu27b/models/qwen3.8-27b-Q4_K_M.gguf"
REPO="$HOME/git/riir-infer"
TARGET="$RIG_MARK/target"
LOG="$RIG_MARK/e1e5.log"

mkdir -p "$RIG_MARK"
: > "$LOG"

say() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }

# ── quiet-box guard: refuse while the 1191 v2 gate is alive ──────────────
if pgrep -f "flashmemory_decode_gate" >/dev/null 2>&1; then
    say "REFUSE: flashmemory_decode_gate still running (the 1191 v2 trio holds the GPU) — re-run after it exits"
    exit 2
fi

# ── preflight ────────────────────────────────────────────────────────────
[ -f "$GGUF" ] || { say "REFUSE: model missing at $GGUF"; exit 2; }
GOT=$(shasum -a 256 "$GGUF" | cut -c1-8)
[ "$GOT" = "e00082f7" ] || { say "REFUSE: model sha mismatch ($GOT, expect e00082f7)"; exit 2; }
cd "$REPO" || exit 2
export CARGO_TARGET_DIR="$TARGET"

say "=== i1203 E1-E5 window start ==="
say "box: $(uptime)"

# ── E1: engagement proof (build once WITH metal_tensor_gemm) ─────────────
say "E1: building qwen38_wgpu_bench --features qwen38_wgpu,metal_tensor_gemm (release, isolated target)"
cargo build --release --bin qwen38_wgpu_bench \
    --features qwen38_wgpu,metal_tensor_gemm 2>>"$LOG" \
    && say "E1: build OK" || { say "E1: BUILD FAILED — stop (everything below needs this binary)"; exit 1; }

BENCH="$TARGET/release/qwen38_wgpu_bench"

say "E1a: ARMED posture (envs unset — league default)"
env -u RIIR_QWEN38_DENSE_MM2D -u RIIR_QWEN38_DENSE_MM2D_F16ACT -u RIIR_QWEN38_ATTN_TENSOR \
    -u RIIR_QWEN38_LMHEAD_Q6K -u RIIR_QWEN38_DENSE_MM2D_RELAXED \
    "$BENCH" --gguf "$GGUF" --pp 512 --decode 8 > "$RIG_MARK/e1a_armed.txt" 2>&1
grep -m1 "Issue 1150\|Issue 1176" "$RIG_MARK/e1a_armed.txt" | sed 's/^/E1a banner: /' | tee -a "$LOG"
grep -E "warm-up|pass 1" "$RIG_MARK/e1a_armed.txt" | sed 's/^/E1a timing: /' | tee -a "$LOG"

say "E1b: DEMOTED posture (all five envs =0)"
RIIR_QWEN38_DENSE_MM2D=0 RIIR_QWEN38_DENSE_MM2D_F16ACT=0 RIIR_QWEN38_ATTN_TENSOR=0 \
RIIR_QWEN38_LMHEAD_Q6K=0 RIIR_QWEN38_DENSE_MM2D_RELAXED=0 \
    "$BENCH" --gguf "$GGUF" --pp 512 --decode 8 > "$RIG_MARK/e1b_demoted.txt" 2>&1
grep -m1 "Issue 1150\|Issue 1176" "$RIG_MARK/e1b_demoted.txt" | sed 's/^/E1b banner: /' | tee -a "$LOG"
grep -E "warm-up|pass 1" "$RIG_MARK/e1b_demoted.txt" | sed 's/^/E1b timing: /' | tee -a "$LOG"

say "E1 verdict inputs: armed banner present? demoted banner = DEMOTED? compare e1a/e1b pass times against the 1203 evidence (9 vs 1 tok/s)"

# ── E2: manual recipe (SKIP-LOUD) ────────────────────────────────────────
say "E2: SKIP-LOUD — per-launch microbench needs the harness pinned first; manual recipe: time MetalDenseF16ZeroCopyGemm::dispatch xN on a single weight (bench 923's harness shape), m3 vs amo, relaxed 0|unset A/B"

# ── E3: 761 counters probe (SKIP-LOUD until the bin lands) ───────────────
if [ -x "$TARGET/release/i1203_counters_probe" ]; then
    say "E3: running the counters probe"
    "$TARGET/release/i1203_counters_probe" --gguf "$GGUF" 2>&1 | tee -a "$LOG"
else
    say "E3: SKIP-LOUD — the probe bin (vendored fork's Issue-761 pub statics reader) is not built yet; write scripts-side bin first (cubecl-wgpu resolves via [patch.crates-io])"
fi

# ── E4: decode-ablate residual wall ─────────────────────────────────────
if [ "${1:-}" != "--skip-e4" ]; then
    say "E4: decode-ablate matrix (32 steps, short ctx — the class census)"
    "$BENCH" --gguf "$GGUF" --decode 32 --decode-ablate-matrix > "$RIG_MARK/e4_ablate.txt" 2>&1
    grep -E "residual|wall|tok/s|class" "$RIG_MARK/e4_ablate.txt" | tail -20 | sed 's/^/E4: /' | tee -a "$LOG"
else
    say "E4: skipped by --skip-e4"
fi

say "=== window end — evidence in $RIG_MARK/, log $LOG; E5 verdict = the lane's read of these rows ==="
exit 0
