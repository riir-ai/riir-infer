#!/usr/bin/env bash
# prep_opponent_arm.sh — issue 1186 T2b opponent half (plan 629, riir-infer).
# Runs ON amo. Clone fork@scored-tag, build Metal, pull BF16 (sha-verify),
# imatrix @v1 + @v2, quantize Q3_K x3 (v1 / v2 / rtn), ledger JSON.
# Prep only — no measurement here; the perf cell runs on the 4090 later.
set -euo pipefail

# Non-interactive rig shells carry no homebrew PATH — cmake lives at /opt/homebrew/bin.
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
command -v cmake >/dev/null || { echo "FATAL: cmake not on PATH after export" >&2; exit 3; }

W=/Users/katopz/cq3-prep
TAG=prism-b12128-d7b2896
REPO=https://github.com/PrismML-Eng/llama.cpp.git
CORPUS_DIR=/Users/katopz/cq3-prep/corpora
MODELS=/Users/katopz/cq3-prep/models
V1_SHA=a6fb9133e3a675b42cca96f726fa4b386c07ca93eca80d18689504b715c3d316
V2_SHA=a3df73ada8719d77323a2f2e81d5592af407941194288c86845e1514d0ef51c7
SHARD1_SHA=b9966e82b7a4d87028b5eae061d578ee826305ebf8baea5bfc6e09bad0ba191f
SHARD2_SHA=92e3943c4f9bd6292a7bef82369f65fed9bfed088b9df0fb2fa2ce17c9edfa02
HF_BASE=https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/BF16

mkdir -p "$W" "$MODELS" "$CORPUS_DIR"
LOG="$W/prep_log.txt"
exec > >(tee -a "$LOG") 2>&1
echo "== prep_opponent_arm start $(date -u +%Y-%m-%dT%H:%M:%SZ) host $(hostname) =="

sha256_of() { shasum -a 256 "$1" | awk '{print $1}'; }
die() { echo "FATAL: $*" >&2; exit 3; }

# --- corpora (shipped from m3; verify against pins) ---
for c in bpb_corpus_v1.txt bpb_corpus_v2.txt; do
  [ -f "$CORPUS_DIR/$c" ] || die "missing corpus $CORPUS_DIR/$c (ship from m3 first)"
done
v1s=$(sha256_of "$CORPUS_DIR/bpb_corpus_v1.txt"); [ "$v1s" = "$V1_SHA" ] || die "v1 sha mismatch: $v1s"
v2s=$(sha256_of "$CORPUS_DIR/bpb_corpus_v2.txt"); [ "$v2s" = "$V2_SHA" ] || die "v2 sha mismatch: $v2s"
echo "corpora verified: v1 v2 shas match pins"

# --- clone + build ---
if [ ! -d "$W/llama.cpp/.git" ]; then
  git clone --depth 1 --branch "$TAG" "$REPO" "$W/llama.cpp" || die "clone of $REPO @$TAG failed"
fi
cd "$W/llama.cpp"
echo "fork pin: $(git log --oneline -1) tag=$(git describe --tags 2>/dev/null || echo '?')"
if [ ! -x build/bin/llama-imatrix ] || [ ! -x build/bin/llama-quantize ]; then
  cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=ON >/dev/null
  cmake --build build --config Release -j 8 --target llama-imatrix llama-quantize
fi
IMAT=build/bin/llama-imatrix
QUANT=build/bin/llama-quantize
[ -x "$IMAT" ] && [ -x "$QUANT" ] || die "imatrix/quantize binaries missing after build"

# --- BF16 shards (resume-friendly; verify each) ---
S1="$MODELS/Qwen3.8-27B-BF16-00001-of-00002.gguf"
S2="$MODELS/Qwen3.8-27B-BF16-00002-of-00002.gguf"
dl() { # url dest sha
  local url=$1 dest=$2 sha=$3
  if [ -f "$dest" ] && [ "$(sha256_of "$dest")" = "$sha" ]; then echo "have $dest (sha ok)"; return 0; fi
  curl -L --fail --retry 3 -C - -o "$dest.part" "$url"
  mv "$dest.part" "$dest"
  local got; got=$(sha256_of "$dest")
  [ "$got" = "$sha" ] || die "shard sha mismatch for $dest: $got"
}
dl "$HF_BASE/Qwen3.8-27B-BF16-00001-of-00002.gguf" "$S1" "$SHARD1_SHA"
dl "$HF_BASE/Qwen3.8-27B-BF16-00002-of-00002.gguf" "$S2" "$SHARD2_SHA"

# --- imatrix passes (window counts recorded from log) ---
run_imatrix() { # corpus outfile
  local corpus=$1 out=$2
  if [ -f "$out" ]; then echo "have $out"; return 0; fi
  "$IMAT" -m "$S1" -f "$corpus" -c 512 -o "$out" -noflush 2>&1 | tee "$W/$(basename "$out").imatrix_log.txt" | tail -3
  [ -f "$out" ] || die "imatrix produced no $out"
}
run_imatrix "$CORPUS_DIR/bpb_corpus_v1.txt" "$MODELS/imatrix_v1.gguf"
run_imatrix "$CORPUS_DIR/bpb_corpus_v2.txt" "$MODELS/imatrix_v2.gguf"

# --- quantize passes ---
quant() { # out qtype imatrix_or_empty tagname
  local out=$1 qtype=$2 imat=$3 tag=$4
  if [ -f "$out" ]; then echo "have $out"; return 0; fi
  if [ -n "$imat" ]; then
    "$QUANT" --imatrix "$imat" "$S1" "$out" "$qtype" 2>&1 | tee "$W/${tag}_quant_log.txt" | tail -5
  else
    "$QUANT" "$S1" "$out" "$qtype" 2>&1 | tee "$W/${tag}_quant_log.txt" | tail -5
  fi
  [ -f "$out" ] || die "quantize produced no $out"
}
quant "$MODELS/qwen38-Q3_K-cal-v1.gguf"        Q3_K "$MODELS/imatrix_v1.gguf" q3k_v1
quant "$MODELS/qwen38-Q3_K-cal-v2.gguf"        Q3_K "$MODELS/imatrix_v2.gguf" q3k_v2
quant "$MODELS/qwen38-Q3_K-rtn.gguf"           Q3_K ""                              q3k_rtn

# --- ledger ---
py3() { command -v python3 >/dev/null && python3 "$@"; }
py3 - "$W" <<'PYEOF'
import hashlib, json, subprocess, sys, os
w = sys.argv[1]
m = os.path.join(w, "models")
def blake3(p):
    try:
        import blake3
    except ImportError:
        return None
    h = blake3.blake3()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()
def windows(p):
    log = os.path.join(w, os.path.basename(p) + ".imatrix_log.txt")
    n = None
    if os.path.exists(log):
        import re
        txt = open(log, errors="replace").read()
        ms = re.findall(r"chunk\s+(\d+)", txt) or re.findall(r"Processing chunk\s+(\d+)", txt)
        if ms:
            n = max(int(x) for x in ms)
    return n
rows = {}
for name, f in [("imatrix_v1", "imatrix_v1.gguf"), ("imatrix_v2", "imatrix_v2.gguf"),
                ("q3k_cal_v1", "qwen38-Q3_K-cal-v1.gguf"), ("q3k_cal_v2", "qwen38-Q3_K-cal-v2.gguf"),
                ("q3k_rtn", "qwen38-Q3_K-rtn.gguf")]:
    p = os.path.join(m, f)
    if os.path.exists(p):
        rows[name] = {"file": p, "bytes": os.path.getsize(p), "sha256": sha256(p),
                      "blake3": blake3(p), "imatrix_windows": windows(p) if name.startswith("imatrix") else None}
commit = subprocess.run(["git", "-C", os.path.join(w, "llama.cpp"), "rev-parse", "HEAD"],
                        capture_output=True, text=True).stdout.strip()
ledger = {
    "schema": "cq3-prep-ledger-v1", "plan": 629, "issue": 1186,
    "date": subprocess.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], capture_output=True, text=True).stdout.strip(),
    "host": subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip(),
    "model_source": "unsloth/Qwen3.8-27B-GGUF BF16 split (upstream Qwen/Qwen3.8-27B)",
    "shard_sha256": ["b9966e82b7a4d87028b5eae061d578ee826305ebf8baea5bfc6e09bad0ba191f",
                     "92e3943c4f9bd6292a7bef82369f65fed9bfed088b9df0fb2fa2ce17c9edfa02"],
    "opponent": {"repo": "PrismML-Eng/llama.cpp", "tag": "prism-b12128-d7b2896", "commit": commit},
    "corpora": {"v1": {"sha256": "a6fb9133e3a675b42cca96f726fa4b386c07ca93eca80d18689504b715c3d316", "bytes": 262144},
                 "v2": {"sha256": "a3df73ada8719d77323a2f2e81d5592af407941194288c86845e1514d0ef51c7", "bytes": 1256465}},
    "chunk_ctx": 512, "quant_type": "Q3_K",
    "artifacts": rows,
}
with open(os.path.join(w, "ledger.json"), "w") as f:
    json.dump(ledger, f, indent=2)
print(json.dumps({k: {"sha256": v["sha256"], "bytes": v["bytes"], "windows": v["imatrix_windows"]} for k, v in rows.items()}, indent=1))
PYEOF

echo "== prep_opponent_arm DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
