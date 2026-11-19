#!/bin/bash
# Issue 1231 T2 — the pipelined-harvest wall cell (amo exclusive, quiet).
# Arms: the cadence resident lane vs the SAME lane driven by the pipelined
# snapshot-future loop (resident_harvest_fut), ABBA within the cell test.
# The pre-registered bars live in .issues/1231 T2 (adopt iff the pipelined
# arm improves >= 0.3 ms/tok vs the same-cell cadence arm, within-cell
# delta median of pairs; G1 walk identity both arms; the stop cell's
# visible prefix truncation-exact, overshoot <= 2K).
set -uo pipefail
cd /Users/katopz/git/riir-ai
OUT=/Users/katopz/git/riir-infer/.i1231cell
mkdir -p "$OUT"

FEATURES="cubecl_runtime ternary_gemv deltanet_recurrence_rowpar params_handle_cache ternary_gemm_batched"
export CARGO_TARGET_DIR=/tmp/i1231cell

echo "════ [i1231] quiet-exclusivity re-check at cell start ════"
uptime
ps aux | grep -E "cargo|riir-gpu" | grep -v grep | head -3

echo "════ [i1231] release build (one binary, both postures) ════"
cargo test -p riir-gpu --features "$FEATURES" --release --test bench_831_decode_long --no-run || exit 2

echo "════ [i1231] cell 1: cadence resident (the same-cell cadence arm) ════"
env DECODE_CTX=2048 DECODE_TOKENS=512 DECODE_RUNS=3 \
  cargo test -p riir-gpu --features "$FEATURES" --release \
  --test bench_831_decode_long -- --nocapture --ignored --test-threads=1 \
  bench_831_decode_long_resident_ab 2>&1 | tee "$OUT/k8_cadence_ab.log" | tail -20

echo "════ [i1231] cell 2: pipelined-harvest wall cell (ABBA within) ════"
env DECODE_CTX=2048 DECODE_TOKENS=512 DECODE_RUNS=3 \
  cargo test -p riir-gpu --features "$FEATURES" --release \
  --test bench_831_decode_long -- --nocapture --ignored --test-threads=1 \
  bench_831_decode_long_resident_pipeline 2>&1 | tee "$OUT/k8_pipeline_ab.log" | tail -25

echo "════ [i1231] cell 3: cadence stop-identity ════"
env DECODE_CTX=2048 DECODE_TOKENS=512 DECODE_RUNS=3 \
  cargo test -p riir-gpu --features "$FEATURES" --release \
  --test bench_831_decode_long -- --nocapture --ignored --test-threads=1 \
  bench_831_resident_stop_identity 2>&1 | tee "$OUT/k8_cadence_stop.log" | tail -15

echo "════ [i1231] cell 4: pipelined stop-identity ════"
env DECODE_CTX=2048 DECODE_TOKENS=512 DECODE_RUNS=3 \
  cargo test -p riir-gpu --features "$FEATURES" --release \
  --test bench_831_decode_long -- --nocapture --ignored --test-threads=1 \
  bench_831_resident_stop_identity_pipeline 2>&1 | tee "$OUT/k8_pipeline_stop.log" | tail -15

echo "I1231-T2-CELLS-DONE"
