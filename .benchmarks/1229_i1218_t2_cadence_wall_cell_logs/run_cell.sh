#!/bin/bash
# Issue 1218 T2 — the fence-cadence wall cell (amo exclusive).
# Runs bench_831's wall A/B + stop-identity at the incumbent cadence (env
# unset) and the armed K=32 cadence, same binary, ABBA within each run.
# The pre-registered bars live in .issues/1218 T2 (adopt iff the resident
# wall improves >= 0.3 ms/tok vs the same-box resident-K8 arm; G1
# walk-identity both arms — the stop-identity cell asserts the visible
# prefixes identical + reports overshoot).
set -uo pipefail
cd /Users/katopz/git/riir-ai
OUT=/Users/katopz/git/riir-infer/.i1218cell
mkdir -p "$OUT"

FEATURES="cubecl_runtime ternary_gemv deltanet_recurrence_rowpar params_handle_cache ternary_gemm_batched"
export CARGO_TARGET_DIR=/tmp/i1218cell

echo "════ [i1218] release build (one binary, both postures) ════"
cargo test -p riir-gpu --features "$FEATURES" --release --test bench_831_decode_long --no-run || exit 2

run_cell() {
  local name="$1"; shift
  echo "════ [i1218] cell: $name ════"
  env "$@" cargo test -p riir-gpu --features "$FEATURES" --release \
    --test bench_831_decode_long -- --nocapture --ignored --test-threads=1 \
    bench_831_decode_long_resident_ab 2>&1 | tee "$OUT/${name}_ab.log" | tail -25
  echo "---- stop-identity ----"
  env "$@" cargo test -p riir-gpu --features "$FEATURES" --release \
    --test bench_831_decode_long -- --nocapture --ignored --test-threads=1 \
    bench_831_resident_stop_identity 2>&1 | tee "$OUT/${name}_stop.log" | tail -18
}

# Run 1 — the incumbent cadence (env unset): same-box incumbent + resident-K8 anchor.
run_cell k8_incumbent DECODE_CTX=2048 DECODE_TOKENS=512 DECODE_RUNS=3

# Run 2 — the ARMED cadence K=32 (the T2 arm).
run_cell k32_armed DECODE_CTX=2048 DECODE_TOKENS=512 DECODE_RUNS=3 RIIR_RESIDENT_HARVEST_K=32

# Run 3 — the mid-burst stop arm at the armed cadence (STOP_AT=65 lands one
# past a K=32 burst boundary: overshoot handling + identity at the unaligned stop).
echo "════ [i1218] cell: k32_armed_stop65 ════"
RIIR_RESIDENT_HARVEST_K=32 STOP_CTX=512 STOP_AT=65 \
  cargo test -p riir-gpu --features "$FEATURES" --release \
  --test bench_831_decode_long -- --nocapture --ignored --test-threads=1 \
  bench_831_resident_stop_identity 2>&1 | tee "$OUT/k32_armed_stop65_stop.log" | tail -18

echo "════ [i1218] ALL CELLS DONE — logs in $OUT ════"
