#!/bin/sh
# Issue 1213 / 1117 T4 — the 128K Metal rho-arms trio RE-RUN on bmo (M5 Ultra).
# The m3 v2 run died at 55% of rho0.15 (laptop closed 08:04 +0700 Nov 13; the
# wake topology died with it). Owner directive: re-run on bmo. The gate
# binary is the FIXED stride build (issue 1213 fix 5425494e + 19c12a3c; bmo
# checkout verified to carry it).
#
# Same loss-lesson posture as the bmo CPU trio (scripts/i1117_bmo_128k_run.sh):
# (1) artifacts on a DURABLE path (~/riir-bench), (2) per-arm bins written
# incrementally with digest auto-load on restart — a kill loses at most the
# in-flight arm and a re-queued job resumes the rest.
#
# SAME-BOX LAW (the reason dense is re-measured here): the T4 verdict is
# Metal-arm vs Metal-dense dNLL + the speedup denominator — both must be
# same-box. The m3 dense (2.4405 pooled / 45396.1s) stays the m3 record,
# never a bmo baseline. Corpus = the bmo CPU trio's own bytes (identical
# corpus to the host legs the ddNLL bars compare against).
set -eu

export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
REPO="$HOME/git/riir-infer"
OUT="$HOME/riir-bench/i1117_128k_metal_bmo"
GGUF="$HOME/git/riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf"
CORPUS="$HOME/riir-bench/i1117_128k/corpus_128k.txt"
CORPUS_SHA256="95816174fb3a5d92f2101b6d46bb77f466c36fc911123d42a13a0cd67543ff10"
CORPUS_BYTES=629060
GATE="$OUT/target/release/flashmemory_decode_gate"

cd "$REPO"
mkdir -p "$OUT"

echo "=== i1213/1117 128K Metal trio — bmo box state at start ==="
date -u +%Y-%m-%dT%H:%M:%SZ
uptime
sysctl hw.memsize | tr -s ' \t' ' '
sysctl -n machdep.cpu.brand_string
pmset -g batt | head -2
df -h "$HOME" | tail -1
git -C "$REPO" rev-parse HEAD

# double-launch guard: a second gate would contend the GPU (Bench 649 law).
if pgrep -f flashmemory_decode_gate >/dev/null 2>&1; then
  echo "REFUSE: a flashmemory_decode_gate is already running:"; pgrep -lf flashmemory_decode_gate; exit 2
fi

# corpus identity: the host legs' bytes, never regenerated.
test -f "$CORPUS" || { echo "REFUSE: corpus missing: $CORPUS"; exit 2; }
GOT_BYTES=$(wc -c < "$CORPUS" | tr -d ' ')
GOT_SHA=$(shasum -a 256 "$CORPUS" | cut -d' ' -f1)
[ "$GOT_BYTES" = "$CORPUS_BYTES" ] || { echo "REFUSE: corpus byte count $GOT_BYTES != $CORPUS_BYTES"; exit 2; }
[ "$GOT_SHA" = "$CORPUS_SHA256" ] || { echo "REFUSE: corpus sha256 mismatch (host-leg corpus drifted?)"; exit 2; }
echo "corpus OK: $GOT_BYTES bytes, sha256 $GOT_SHA"

test -f "$GGUF" || { echo "REFUSE: gguf missing: $GGUF"; exit 2; }
ls -la "$GGUF"

echo "=== build the gate (release, flashmemory_budget_gate — CPU + Metal arms) ==="
cargo build --release \
  -p riir-infer-gpu \
  --features flashmemory_budget_gate \
  --bin flashmemory_decode_gate \
  --target-dir "$OUT/target"

echo "=== launch the Metal trio (dense + rho 0.15 + rho 0.30 @ ctx 131072) ==="
(
  while kill -0 "$$" 2>/dev/null; do
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) load=$(uptime | sed 's/.*load averages*: //')" >> "$OUT/load_trace.log"
    sleep 300
  done
) &
TRACE_PID=$!
# caffeinate binds the sleep assertions to THIS script's lifetime
# (re-arm per run, never inherit — the i1117 sleep-proofing lesson; m3 v2
# died to exactly the loss this guards against).
GATE_RC=0
caffeinate -is "$GATE" \
  --gguf "$GGUF" \
  --corpus-file "$CORPUS" \
  --ctx 131072 \
  --rho-list 0.15,0.30 \
  --bucket 8192 \
  --metal \
  --out "$OUT" > "$OUT/run_metal_bmo.log" 2>&1 || GATE_RC=$?
kill "$TRACE_PID" 2>/dev/null || true
echo "gate exit rc=$GATE_RC"
date -u +%Y-%m-%dT%H:%M:%SZ
tail -20 "$OUT/run_metal_bmo.log"

echo "=== best-effort collection (bins are the durable record either way) ==="
if python3 "$REPO/scripts/i1117_collect_128k.py" --out "$OUT" 2>&1; then
  echo "collector wrote the stats tables"
else
  echo "collector skipped/failed (no 32K ref report here is EXPECTED on bmo —"
  echo "the cross-posture table is not verdict-bearing); the arm bins in"
  echo "$OUT are complete and durable; the record task runs the collector"
fi

echo "=== sel read (the 1213 fix verification input) ==="
python3 "$REPO/scripts/i1117_read_sel.py" "$OUT"/rho_0.15.bin "$OUT"/rho_0.30.bin 2>&1 || true

echo "=== artifacts ==="
ls -la "$OUT"
echo "METAL TRIO DONE — arm bins: dense.bin / rho_0.15.bin / rho_0.30.bin (complete=1 each expected)."
exit "$GATE_RC"
