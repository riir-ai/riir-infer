#!/bin/sh
# Issue 1243 T1(a) — the QDA Q4 frozen-read census driver (bmo rig job).
#
# Orchestration only — every instrument is an existing binary:
#   1. box state + PROVENANCE (the bench_preflight discipline)
#   2. pins: incumbent sha256 (the T1 pin — first record), Bonsai plan-631 pin
#   3. fetch + pin qda-lab/QDA-Qwen3.8-27B-Q4-GGUF @ rev 2038e5f1 (the Q6 row
#      is OUT of this job's scope per the work order — Q4 only)
#   4. the issue's arch/size sanity assert: arch + layer count + parameter
#      count (GGUF header read) equal between QDA-Q4 and the incumbent,
#      else REFUSE LOUD before any scoring
#   5. the census attempt per the issue posture (si_zero_shot_ppl golden-NLL
#      build): the incumbent attempt EXPECTS the loud loader refusal (its
#      token_embd.weight is Q4_K; the ternary loader's contract keeps globals
#      ternary — gguf_loader.rs load_ternary_proj) — the refusal text is the
#      evidence, captured not fatal. The Bonsai control golden read MUST run
#      (bench 1215 proved the lane) — its failure IS fatal.
#   6. disclosed deviation arm: prism llama-perplexity pooled NLL for both
#      W4 files (same lane, same corpus → apples-to-apples pooled Δ-NLL).
#      Time-boxed by the rig job timeout; failure is non-fatal (recorded).
#
# Corpus: the i1188 recipe regenerated from the committed unit (x28, pinned
# shas) — ONE frozen corpus shared by every arm (the grammar-not-answers law).
# Artifacts: ~/riir-bench/i1243a/ on the box; the .benchmarks doc quotes them.
set -eu

export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
REPO="$HOME/git/riir-infer"
OUT="$HOME/riir-bench/i1243a"
INC="$HOME/uzu27b/models/qwen3.8-27b-Q4_K_M.gguf"
BONSAI="$HOME/git/riir-train/data/Ternary-Bonsai-2.5-27B-PQ2_0.gguf"
BONSAI_SHA256="3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1"
UNIT="$REPO/scripts/i1188_corpus_unit.txt"
UNIT_SHA256="9f95700f8d447da6c9390df6fe5c6d5bd2c17e6274b71db9d4f83a44d4ae5934"
CORPUS="$OUT/i1188_corpus.txt"
CORPUS_SHA256="a6998c740ab43d21eaa127e699d1aff042cba41b4df8aea46e7f05b40712fcbc"
QDA_DIR="$OUT/qda"
QDA_FILE="$QDA_DIR/QDA-Qwen3.8-27B-Q4.gguf"
QDA_REV="2038e5f1502bdd09e15a4015491ff95b8b523e37"
FEATURES="scale_invariant_attn,bonsai2_hadamard,golden_nll"
BIN="$OUT/target/release/si_zero_shot_ppl"
FLAGS="--arms base --max-length 2048 --min-length 512 --score-window 512"

cd "$REPO"
mkdir -p "$OUT" "$QDA_DIR"

echo "=== i1243a census driver — bmo box state at start ==="
date -u +%Y-%m-%dT%H:%M:%SZ
uptime
sysctl hw.memsize | tr -s ' \t' ' '
sysctl -n machdep.cpu.brand_string
pmset -g batt | head -2
df -h "$HOME" | tail -1
git -C "$REPO" rev-parse HEAD

echo "PROVENANCE: bmo (M5 Ultra, dedicated runner), job exclusive, caffeinate -is per pass, load at start:$(uptime | awk -F'load averages:' '{print $2}'), AC: $(pmset -g batt | grep -o 'AC Power' || echo MISSING)"

# per-pass load trace (post-hoc contention attribution — the ratio law)
(
  while kill -0 "$$" 2>/dev/null; do
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) load=$(uptime | awk -F'load averages:' '{print $2}')" >> "$OUT/load_trace.log"
    sleep 120
  done
) &
TRACE_PID=$!
trap 'kill "$TRACE_PID" 2>/dev/null || true' EXIT

# double-launch guard
if pgrep -f 'si_zero_shot_ppl|llama-perplexity' >/dev/null 2>&1; then
  echo "REFUSE: a census binary is already running:"; pgrep -lf 'si_zero_shot_ppl|llama-perplexity'; exit 2
fi

# --- pins ---------------------------------------------------------------
test -f "$INC" || { echo "REFUSE: incumbent missing: $INC"; exit 2; }
INC_SHA=$(shasum -a 256 "$INC" | cut -d' ' -f1)
echo "INCUMBENT_PIN sha256 $INC_SHA  ($INC)"
test -f "$BONSAI" || { echo "REFUSE: bonsai missing: $BONSAI"; exit 2; }
BON_GOT=$(shasum -a 256 "$BONSAI" | cut -d' ' -f1)
[ "$BON_GOT" = "$BONSAI_SHA256" ] || { echo "REFUSE: bonsai sha256 $BON_GOT != plan-631 pin"; exit 2; }
echo "BONSAI_PIN OK sha256 $BON_GOT"

# --- corpus (regen from the committed recipe, pinned) -------------------
test -f "$UNIT" || { echo "REFUSE: corpus unit missing: $UNIT"; exit 2; }
UGOT=$(shasum -a 256 "$UNIT" | cut -d' ' -f1)
[ "$UGOT" = "$UNIT_SHA256" ] || { echo "REFUSE: corpus unit sha256 $UGOT != pin"; exit 2; }
for i in $(seq 28); do cat "$UNIT"; done > "$CORPUS"
CGOT=$(shasum -a 256 "$CORPUS" | cut -d' ' -f1)
CBYTES=$(wc -c < "$CORPUS" | tr -d ' ')
[ "$CGOT" = "$CORPUS_SHA256" ] || { echo "REFUSE: corpus x28 sha256 $CGOT != pin"; exit 2; }
[ "$CBYTES" = "29036" ] || { echo "REFUSE: corpus byte count $CBYTES != 29036"; exit 2; }
echo "corpus OK: $CBYTES bytes, sha256 $CGOT"

# --- fetch + pin the QDA Q4 GGUF ---------------------------------------
# bmo has no hf CLI (job 01a1fbc9 refused) - plain curl against the pinned
# resolve URL (HF LFS supports ranges: -C - resumes a partial fetch).
if [ ! -f "$QDA_FILE" ]; then
  command -v curl >/dev/null 2>&1 || { echo "REFUSE: no curl on this box"; exit 2; }
  URL="https://huggingface.co/qda-lab/QDA-Qwen3.8-27B-Q4-GGUF/resolve/$QDA_REV/QDA-Qwen3.8-27B-Q4.gguf"
  curl -L -C - --retry 3 --retry-delay 5 -o "$QDA_FILE" "$URL"
fi
test -f "$QDA_FILE" || { echo "REFUSE: QDA download did not land: $QDA_FILE"; exit 2; }
QDA_SHA=$(shasum -a 256 "$QDA_FILE" | cut -d' ' -f1)
echo "QDA_Q4_PIN sha256 $QDA_SHA  rev $QDA_REV  ($QDA_FILE)"

# --- the issue's arch/size sanity assert (GGUF header read) -------------
set +e
python3 - "$INC" "$QDA_FILE" > "$OUT/header_census.txt" <<'PYEOF'
import struct, sys

def header(path):
    f = open(path, "rb")
    def u(fmt):
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    def s():
        n = u("<Q"); return f.read(n).decode("utf-8", "replace")
    magic = f.read(4); ver = u("<I"); ntensors = u("<Q"); nkv = u("<Q")
    TYPES = {0:("<B",1),1:("<b",1),2:("<H",1),3:("<h",1),4:("<I",1),5:("<i",1),
             6:("<f",1),7:("<B",1),10:("<Q",1),11:("<q",1),12:("<d",1)}
    def val(t):
        if t == 8: return s()
        if t == 9:
            et = u("<I"); n = u("<Q")
            items = [val(et) for _ in range(min(n, 4))]
            rest = n - len(items)
            if et == 8:
                for _ in range(rest): s()
            else:
                f.seek(struct.calcsize(TYPES[et][0]) * rest, 1)
            return ["ARR", n, items]
        fmt, _ = TYPES[t]; return u(fmt)
    kvs = {}
    for _ in range(nkv):
        k = s(); t = u("<I"); kvs[k] = val(t)
    total_elems = 0; hist = {}
    for _ in range(ntensors):
        name = s(); nd = u("<I")
        dims = [u("<Q") for _ in range(nd)]
        ty = u("<I"); off = u("<Q")
        n = 1
        for d in dims: n *= d
        total_elems += n
        hist[ty] = hist.get(ty, 0) + 1
    return {"magic": magic.decode(), "ver": ver, "ntensors": ntensors,
            "arch": kvs.get("general.architecture"),
            "block_count": kvs.get("qwen35.block_count"),
            "nextn": kvs.get("qwen35.nextn_predict_layers"),
            "n_embd": kvs.get("qwen35.embedding_length"),
            "params": total_elems, "hist": hist}

inc = header(sys.argv[1]); qda = header(sys.argv[2])
for tag, h in (("incumbent", inc), ("qda_q4", qda)):
    print(f"{tag}: arch={h['arch']} block_count={h['block_count']} nextn={h['nextn']} "
          f"n_embd={h['n_embd']} tensors={h['ntensors']} params={h['params']} hist={h['hist']}")
fails = []
if inc["arch"] != qda["arch"]: fails.append(f"arch {inc['arch']} != {qda['arch']}")
main_i = (inc["block_count"] or 0) - (inc["nextn"] or 0)
main_q = (qda["block_count"] or 0) - (qda["nextn"] or 0)
if main_i != main_q: fails.append(f"main layer count {main_i} != {main_q}")
if inc["params"] != qda["params"]:
    fails.append(f"param count {inc['params']} != {qda['params']}")
if fails:
    print("SANITY ASSERT REFUSES LOUD: " + "; ".join(fails))
    sys.exit(3)
print("SANITY ASSERT PASS: arch + main layers + param count equal")
PYEOF
SAN_RC=$?
set -e
if [ "$SAN_RC" -ne 0 ]; then
  cat "$OUT/header_census.txt"
  if [ "$SAN_RC" -eq 3 ]; then
    echo "REFUSE: the issue's sanity assert failed — no scoring (exit 3)"
  else
    echo "REFUSE: header census CRASHED (tooling bug, rc=$SAN_RC) — no scoring (exit 4)"
    exit 4
  fi
  exit 3
fi
cat "$OUT/header_census.txt"

# --- build the census binary (release, golden_nll) ----------------------
echo "=== build si_zero_shot_ppl (release, $FEATURES) ==="
cargo build --release --features "$FEATURES" --bin si_zero_shot_ppl --target-dir "$OUT/target"

# --- the census attempt per the issue posture ---------------------------
echo "=== incumbent attempt (EXPECTS the loud ternary-globals refusal) ==="
set +e
"$BIN" --gguf "$INC" --corpus "$CORPUS" $FLAGS > "$OUT/incumbent_attempt.log" 2>&1
INC_RC=$?
set -e
echo "incumbent attempt rc=$INC_RC"
tail -6 "$OUT/incumbent_attempt.log"

echo "=== Bonsai control golden read (MUST run — bench 1215 lane) ==="
"$BIN" --gguf "$BONSAI" --corpus "$CORPUS" $FLAGS > "$OUT/bonsai_golden.log" 2>&1
tail -12 "$OUT/bonsai_golden.log"

# --- deviation arm: prism llama-perplexity pooled NLL (non-fatal) -------
PRISM="$OUT/llamacpp"
PP="$PRISM/build/bin/llama-perplexity"
echo "=== deviation arm: prism llama-perplexity (pooled NLL, both W4 files) ==="
if [ ! -x "$PP" ]; then
  if [ ! -d "$PRISM/.git" ]; then
    git clone --depth 1 --branch prism-b12128-d7b2896 \
      https://github.com/PrismML-Eng/llama.cpp.git "$PRISM" \
      || echo "PRISM SKIPPED: clone failed"
  fi
  if [ -d "$PRISM/.git" ] && [ ! -x "$PP" ]; then
    (cd "$PRISM" && cmake -B build -DCMAKE_BUILD_TYPE=Release \
      && cmake --build build --config Release -j 8 --target llama-perplexity) \
      || echo "PRISM SKIPPED: build failed"
  fi
fi
if [ -x "$PP" ]; then
  set +e
  "$PP" -m "$INC" -f "$CORPUS" > "$OUT/pp_incumbent.log" 2>&1
  echo "pp incumbent rc=$?"
  grep -iE 'NLL|perplexity|error' "$OUT/pp_incumbent.log" | tail -4
  "$PP" -m "$QDA_FILE" -f "$CORPUS" > "$OUT/pp_qda_q4.log" 2>&1
  echo "pp qda rc=$?"
  grep -iE 'NLL|perplexity|error' "$OUT/pp_qda_q4.log" | tail -4
  set -e
else
  echo "PRISM SKIPPED: no llama-perplexity binary"
fi

# --- summary ------------------------------------------------------------
echo "=== i1243a artifacts ==="
{
  echo "incumbent_pin_sha256 $INC_SHA"
  echo "bonsai_pin_sha256 $BON_GOT"
  echo "qda_q4_pin_sha256 $QDA_SHA rev $QDA_REV"
  echo "corpus_sha256 $CGOT bytes $CBYTES"
  echo "incumbent_attempt_rc $INC_RC"
  grep -E 'SANITY ASSERT' "$OUT/header_census.txt" || true
} | tee "$OUT/summary.txt"
ls -la "$OUT"
date -u +%Y-%m-%dT%H:%M:%SZ
echo "DONE i1243a"
