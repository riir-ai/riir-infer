#!/bin/sh
# Issue 1190 / plan 631 — the i1188 T3 ranked-sensitivity cell on bmo (rig job).
# 16 per-layer-family Q8_0 ablation passes + the ffn_down determinism re-run
# over the golden-NLL oracle, Bonsai-2-27B-PQ2_0, prefix 2048 / 512 scored.
# PRE-REGISTERED in .issues/1190 — flags and order are pins, never tunings.
#
# Staging: runs from bmo's OWN ~/git/riir-infer checkout (the sibling layout
# the i1213 Metal trio proved) — the a45099bc attempt died in the --worktree
# sandbox at katgpt-core resolution. Out on the DURABLE path ~/riir-bench/
# (the i1117 loss lessons), not a checkout dotdir.
set -eu

export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
REPO="$HOME/git/riir-infer"
OUT="$HOME/riir-bench/i1190_ranked_cell"
GGUF="$HOME/git/riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf"
GGUF_SHA256="3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1"
UNIT="$REPO/scripts/i1188_corpus_unit.txt"
UNIT_SHA256="9f95700f8d447da6c9390df6fe5c6d5bd2c17e6274b71db9d4f83a44d4ae5934"
CORPUS="$OUT/i1188_corpus.txt"
CORPUS_SHA256="a6998c740ab43d21eaa127e699d1aff042cba41b4df8aea46e7f05b40712fcbc"
BIN="$OUT/target/release/si_zero_shot_ppl"
FEATURES="scale_invariant_attn,bonsai2_hadamard,golden_nll"

cd "$REPO"
mkdir -p "$OUT"

echo "=== i1190 ranked-sensitivity cell — bmo box state at start ==="
date -u +%Y-%m-%dT%H:%M:%SZ
uptime
sysctl hw.memsize | tr -s ' \t' ' '
sysctl -n machdep.cpu.brand_string
pmset -g batt | head -2
df -h "$HOME" | tail -1
git -C "$REPO" rev-parse HEAD

# double-launch guard
if pgrep -f si_zero_shot_ppl >/dev/null 2>&1; then
  echo "REFUSE: a si_zero_shot_ppl is already running:"; pgrep -lf si_zero_shot_ppl; exit 2
fi

# model pin (the 1177 dbirks lesson: never measure a different quant run)
test -f "$GGUF" || { echo "REFUSE: gguf missing: $GGUF"; exit 2; }
GOT=$(shasum -a 256 "$GGUF" | cut -d' ' -f1)
[ "$GOT" = "$GGUF_SHA256" ] || { echo "REFUSE: model sha256 $GOT != pin"; exit 2; }
echo "model OK: sha256 $GOT"

# corpus: regenerate unit x28 from the committed recipe (plan 631), verify
test -f "$UNIT" || { echo "REFUSE: corpus unit missing: $UNIT"; exit 2; }
UGOT=$(shasum -a 256 "$UNIT" | cut -d' ' -f1)
[ "$UGOT" = "$UNIT_SHA256" ] || { echo "REFUSE: corpus unit sha256 $UGOT != pin"; exit 2; }
for i in $(seq 28); do cat "$UNIT"; done > "$CORPUS"
CGOT=$(shasum -a 256 "$CORPUS" | cut -d' ' -f1)
CBYTES=$(wc -c < "$CORPUS" | tr -d ' ')
[ "$CGOT" = "$CORPUS_SHA256" ] || { echo "REFUSE: corpus x28 sha256 $CGOT != pin"; exit 2; }
[ "$CBYTES" = "29036" ] || { echo "REFUSE: corpus byte count $CBYTES != 29036"; exit 2; }
echo "corpus OK: $CBYTES bytes, sha256 $CGOT"

echo "=== build si_zero_shot_ppl (release, $FEATURES) ==="
cargo build --release \
  --features "$FEATURES" \
  --bin si_zero_shot_ppl \
  --target-dir "$OUT/target"

echo "=== provenance ==="
echo "PROVENANCE: bmo (M5 Ultra, dedicated runner), job exclusive, caffeinate -is per pass, load at start: $(uptime | sed 's/.*load averages*: //'), RAM free: $(memory_pressure -Q 2>/dev/null | head -1 || echo n/a), AC: $(pmset -g batt | grep -o 'AC Power' || echo MISSING)"

# per-pass load trace (post-hoc contention attribution — the ratio law)
(
  while kill -0 "$$" 2>/dev/null; do
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) load=$(uptime | sed 's.*load averages*: //')" >> "$OUT/load_trace.log"
    sleep 120
  done
) &
TRACE_PID=$!

FAMILIES="ffn_gate ffn_up ffn_down gdn_qkv gdn_out attn_q attn_k attn_v attn_out attn_gate gdn_alpha gdn_beta gdn_conv gdn_dt gdn_a norms"

CELL_RC=0
for f in $FAMILIES; do
  echo "=== pass: $f ($(date -u +%H:%M:%SZ)) load=$(uptime | sed 's.*load averages*: //') ===" | tee -a "$OUT/pass_times.log"
  caffeinate -is "$BIN" \
    --gguf "$GGUF" \
    --corpus "$CORPUS" \
    --family-ablate "$f" --family-posture q8 \
    --max-length 2048 --min-length 128 --score-window 512 \
    > "$OUT/ablate_$f.log" 2>&1 || CELL_RC=$?
  grep -E "pooled|Δ|delta|tensors|bpw" "$OUT/ablate_$f.log" | tail -5 >> "$OUT/pass_times.log" || true
done

echo "=== determinism gate: ffn_down re-run ($(date -u +%H:%M:%SZ)) ===" | tee -a "$OUT/pass_times.log"
caffeinate -is "$BIN" \
  --gguf "$GGUF" \
  --corpus "$CORPUS" \
  --family-ablate ffn_down --family-posture q8 \
  --max-length 2048 --min-length 128 --score-window 512 \
  > "$OUT/ablate_ffn_down_rerun.log" 2>&1 || CELL_RC=$?

kill "$TRACE_PID" 2>/dev/null || true

echo "=== determinism compare (pooled delta-NLL lines must be to_bits-identical) ==="
grep -iE "pooled.*nll|delta.*nll" "$OUT/ablate_ffn_down.log" | sort > "$OUT/det_a.txt" || true
grep -iE "pooled.*nll|delta.*nll" "$OUT/ablate_ffn_down_rerun.log" | sort > "$OUT/det_b.txt" || true
if diff -q "$OUT/det_a.txt" "$OUT/det_b.txt" >/dev/null 2>&1 && [ -s "$OUT/det_a.txt" ]; then
  echo "DETERMINISM PASS: pooled delta-NLL lines to_bits-identical"
  echo "$(cat "$OUT/det_a.txt")"
else
  echo "DETERMINISM FAIL (or no pooled lines extracted) — diff:"
  diff "$OUT/det_a.txt" "$OUT/det_b.txt" | head -10 || true
  CELL_RC=1
fi

echo "=== summary: pooled delta-NLL per family (T2's ranked-table input) ==="
for f in $FAMILIES; do
  LINE=$(grep -iE "pooled.*delta|pooled.*Δ|pooled dNLL" "$OUT/ablate_$f.log" | tail -1)
  echo "$f | $LINE"
done | tee "$OUT/ranked_input.txt"

echo "cell exit rc=$CELL_RC"
date -u +%Y-%m-%dT%H:%M:%SZ
ls -la "$OUT"
echo "I1190 CELL DONE — logs in $OUT (16 passes + ffn_down rerun); T2 ranked table derives from ranked_input.txt"
exit "$CELL_RC"
