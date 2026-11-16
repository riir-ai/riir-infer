#!/bin/sh
cd /Users/katopz/git/riir-infer
export CARGO_TARGET_DIR=/tmp/i1225f16b
F="prefill_f16b,production,cubecl_runtime,multi_host,riir-infer-core/bonsai2_hadamard"
I1225_PS=32768 cargo test -p riir-infer-gpu --release --features "$F" \
  --test issue1225_f16b_retention_walk f16b_retention_walk -- --ignored --nocapture --test-threads=1 \
  > /tmp/i1225f16b_walk_pp32k_relaxed.log 2>&1
echo "RELAXED_EXIT=$?" >> /tmp/i1225f16b_walk_pp32k_relaxed.log
I1225_POSTURE=strict I1225_PS=32768 cargo test -p riir-infer-gpu --release --features "$F" \
  --test issue1225_f16b_retention_walk f16b_retention_walk -- --ignored --nocapture --test-threads=1 \
  > /tmp/i1225f16b_walk_pp32k_strict.log 2>&1
echo "STRICT_EXIT=$?" >> /tmp/i1225f16b_walk_pp32k_strict.log
echo PP32K_RUNS_DONE >> /tmp/i1225f16b_walk_pp32k_strict.log
