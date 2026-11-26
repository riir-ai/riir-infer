#!/usr/bin/env bash
# bout_metal_cell1.sh — issue 1186 T3 cell 1, the Metal bout on amo.
#
# The pre-registration (riir-infer .issues/1186, "T3 pre-registration", the
# 2026-11-24 amendment) governs: quotable arms = the n2048 matched pairs only,
# window identity in every ledger row, decode = the only promotable axis,
# prefill record-only, engagement proof first, both sides fresh in ONE window.
#
# Ours: qwen38_wgpu_bench (the CubeCL dense forward, RIIR_QWEN38_CQ3 armed)
# on the BF16 base with the CQ3 artifact (the mixed posture: artifact tensors
# serve CQ3, the rest keep the incumbent route).
# Opponent: llama.cpp Metal (prism-b12128-d7b2896, the prep's pinned build) at
# the matched-calibration Q3_K artifacts.
#
# Resumable: per-leg done markers under the OUT dir (durable, never /tmp).
# The script produces RAW LOGS + report.json (the ledger rows); the VERDICT
# (ratios vs the pre-registered bars) is adjudicated OUT of band by the lane,
# never here — the runner measures, it does not judge.
set -euo pipefail
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
command -v cmake >/dev/null || { echo "FATAL: cmake not on PATH" >&2; exit 3; }

REPO=/Users/katopz/git/riir-infer
W=/Users/katopz/cq3-prep
OUT=$W/bout_metal_cell1
MODELS=$W/models
LC=$W/llama.cpp
BF16=$MODELS/Qwen3.8-27B-BF16-00001-of-00002.gguf
# Bash-3.2 constraint: macOS /bin/bash 3.2.57 (and sh posix mode) has no
# associative arrays and parses [b1] subscripts as arithmetic under set -u.
# Plain variables + a case dispatch only; no bash-4-isms anywhere.
OURS_b1=$MODELS/Qwen3.8-27B-CQ3-g32-b3-cal-v1.cq3
OURS_b2p=$MODELS/Qwen3.8-27B-CQ3-g32-b3-cal-v2tail.cq3
OPP_b1=$MODELS/qwen38-Q3_K-cal-v1-n2048.gguf
OPP_b2p=$MODELS/qwen38-Q3_K-cal-v2tail-n2048.gguf
PREFLIGHT=/Users/katopz/git/riir-reflex/scripts/bench_preflight.sh
DECODE_STEPS=128
WALK_N=32
PP_SHAPES=2048,4096
mkdir -p "$OUT"
cd "$REPO"
LOG="$OUT/bout_log.txt"
exec > >(tee -a "$LOG") 2>&1
echo "== bout_metal_cell1 start $(date -u +%Y-%m-%dT%H:%M:%SZ) host $(hostname) =="

leg() { # <name> — skip when done (resume), else run the body function
  local name=$1
  if [ -f "$OUT/L_$name.done" ]; then echo "[resume] L_$name done — skip"; return 0; fi
  echo "── L_$name ──"
  "body_$name"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$OUT/L_$name.done"
}
die() { echo "FATAL: $*" >&2; exit 3; }

body_provenance() {
  [ -f "$PREFLIGHT" ] || die "bench_preflight.sh missing at $PREFLIGHT"
  "$PREFLIGHT" > "$OUT/provenance.txt" 2>&1 || die "bench_preflight refused (box state) — see $OUT/provenance.txt"
  grep -E "PROVENANCE|powermode|load" "$OUT/provenance.txt" || true
  # GPU-exclusivity probe (the correctness-law probe, recorded beside the rows)
  { echo "gpu-consumers:"; ps aux | grep -iE "python.*aned|powermetrics|mps" | grep -v grep || echo "  none"; } > "$OUT/gpu_probe.txt"
  cat "$OUT/gpu_probe.txt"
  for f in "$BF16" "$OURS_b1" "$OURS_b2p" "$OPP_b1" "$OPP_b2p"; do
    [ -f "$f" ] || die "missing artifact $f"
  done
  echo "build: $(git rev-parse --short=12 HEAD)"
}

body_build_ours() {
  cargo build --release -p riir-infer-gpu --features qwen38_wgpu,cq3 --bin qwen38_wgpu_bench
  test -x target/release/qwen38_wgpu_bench
}

body_build_opponent() {
  cd "$LC"
  echo "fork pin: $(git log --oneline -1)"
  [ -x build/bin/llama-bench ] || cmake --build build --config Release -j 8 --target llama-bench
  test -x build/bin/llama-bench
  cd "$REPO"
}

# The behavioral engagement proof: the same walk prompt under the unarmed
# incumbent vs the armed arm — a b=3 lossy arm that produced IDENTICAL ids
# would prove the dispatch was never reached (the Issue-1124 T1.5 class).
body_engagement() {
  local BENCH=target/release/qwen38_wgpu_bench
  echo "[engagement] unarmed incumbent walk (baseline ids)"
  "$BENCH" --gguf "$BF16" --walk-tokens $WALK_N > "$OUT/walk_incumbent.txt" 2>&1 || die "incumbent walk refused"
  echo "[engagement] armed walk (b1)"
  RIIR_QWEN38_CQ3="$OURS_b1" "$BENCH" --gguf "$BF16" --walk-tokens $WALK_N > "$OUT/walk_b1.txt" 2>&1 || die "armed walk refused"
  grep -m1 "\[cq3-serving\] RIIR_QWEN38_CQ3 armed" "$OUT/walk_b1.txt" || die "no cq3 construction line — the arm did not route"
  local i1 a1
  i1=$(grep -m1 '\[walk\] ids:' "$OUT/walk_incumbent.txt" || die "no incumbent walk ids")
  a1=$(grep -m1 '\[walk\] ids:' "$OUT/walk_b1.txt" || die "no armed walk ids")
  echo "incumbent: $i1"
  echo "armed b1 : $a1"
  [ "$i1" != "$a1" ] || die "walk ids IDENTICAL armed vs incumbent — the CQ3 dispatch was never reached (engagement proof FAILED)"
  echo "[engagement] PASS — the arm is reached (ids diverge) and the construction line names the routing"
}

body_ours_cells() {
  local BENCH=target/release/qwen38_wgpu_bench
  for arm in b1 b2p; do
    case $arm in
      b1) OURS_PATH=$OURS_b1 ;;
      b2p) OURS_PATH=$OURS_b2p ;;
      *) die "unknown arm $arm" ;;
    esac
    for run in a b; do
      echo "[ours] arm=$arm run=$run (one load: pp + decode + decode-ctx + walk)"
      RIIR_QWEN38_CQ3="$OURS_PATH" "$BENCH" --gguf "$BF16" \
        --pp "$PP_SHAPES" --decode $DECODE_STEPS --decode-ctx 2048 \
        --walk-tokens $WALK_N \
        > "$OUT/ours_${arm}_${run}.txt" 2>&1 || die "ours cell arm=$arm run=$run refused"
      grep -E "MEDIAN|\[decode\]|\[decode-ctx|\[walk\] ids:|\[cq3-serving\]|gemv-engagement" "$OUT/ours_${arm}_${run}.txt" || die "no cell rows in ours_${arm}_${run}"
    done
    # the determinism leg: walk ids bit-identical across the two runs
    local ia ib
    ia=$(grep -m1 '\[walk\] ids:' "$OUT/ours_${arm}_a.txt")
    ib=$(grep -m1 '\[walk\] ids:' "$OUT/ours_${arm}_b.txt")
    [ "$ia" = "$ib" ] || die "determinism leg FAILED arm=$arm (walk ids differ across identical runs)"
    echo "[determinism] arm=$arm PASS (walk ids bit-identical across runs)"
  done
}

body_opponent_cells() {
  for arm in b1 b2p; do
    case $arm in
      b1) OPP_PATH=$OPP_b1 ;;
      b2p) OPP_PATH=$OPP_b2p ;;
      *) die "unknown arm $arm" ;;
    esac
    echo "[opponent] arm=$arm llama-bench (pp $PP_SHAPES, tg $DECODE_STEPS)"
    "$LC/build/bin/llama-bench" -m "$OPP_PATH" -p "$PP_SHAPES" -n "$DECODE_STEPS" \
      -ngl 99 -r 3 -t 8 > "$OUT/opponent_${arm}.txt" 2>&1 || die "llama-bench refused arm=$arm"
    tail -4 "$OUT/opponent_${arm}.txt"
  done
}

body_ledger() {
  python3 - "$OUT" "$REPO" <<'PYEOF'
import hashlib, json, os, re, subprocess, sys
out, repo = sys.argv[1], sys.argv[2]
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()
def read(p):
    return open(os.path.join(out, p), errors="replace").read()
def grab(pat, txt, cast=str):
    m = re.search(pat, txt)
    return cast(m.group(1)) if m else None

prov = read("provenance.txt")
prov_line = next((l for l in prov.splitlines() if "PROVENANCE" in l), None)
build = subprocess.run(["git", "-C", repo, "rev-parse", "--short=12", "HEAD"],
                       capture_output=True, text=True).stdout.strip()
models = "/Users/katopz/cq3-prep/models"
digests = {
    "ours_b1_blake3_sidecar": open(os.path.join(models, "Qwen3.8-27B-CQ3-g32-b3-cal-v1.cq3.blake3")).read().split()[0],
    "ours_b2p_blake3_sidecar": open(os.path.join(models, "Qwen3.8-27B-CQ3-g32-b3-cal-v2tail.cq3.blake3")).read().split()[0],
    "opponent_b1_sha256": sha256(os.path.join(models, "qwen38-Q3_K-cal-v1-n2048.gguf")),
    "opponent_b2p_sha256": sha256(os.path.join(models, "qwen38-Q3_K-cal-v2tail-n2048.gguf")),
    "base_bf16_shard1_sha256_head_1MB": None,  # the 47 GB shard is pinned by the prep ledger, not re-hashed here
}
rows = {}
for arm in ("b1", "b2p"):
    txt = read(f"ours_{arm}_a.txt")
    txt_b = read(f"ours_{arm}_b.txt")
    opp = read(f"opponent_{arm}.txt")
    row = {
        "window": "head-2048" if arm == "b1" else "tail-2048",
        "ours_build": build,
        "provenance": prov_line,
        "pp2048_median_tok_s": grab(r"\[pp 2048→\d+\] MEDIAN (\d+) tok/s", txt, int),
        "pp4096_median_tok_s": grab(r"\[pp 4096→\d+\] MEDIAN (\d+) tok/s", txt, int),
        "decode128_tok_s_run_a": grab(r"\[decode\] \d+ steps in [\d.]+s = ([\d.]+) tok/s", txt, float),
        "decode128_tok_s_run_b": grab(r"\[decode\] \d+ steps in [\d.]+s = ([\d.]+) tok/s", txt_b, float),
        "decode_ctx2048_tok_s_run_a": grab(r"\[decode-ctx 2048\] \d+ steps in [\d.]+s = ([\d.]+) tok/s", txt, float),
        "walk_ids": grab(r"\[walk\] ids: (\[.*\])", txt),
        "walk_ids_run_b": grab(r"\[walk\] ids: (\[.*\])", txt_b),
        "cq3_armed_line": grab(r"(\[cq3-serving\] RIIR_QWEN38_CQ3 armed[^\n]*)", txt),
        "opponent_llama_bench_raw": opp[-2000:],
    }
    missing = [k for k in ("pp2048_median_tok_s", "pp4096_median_tok_s", "decode128_tok_s_run_a",
                           "decode_ctx2048_tok_s_run_a", "walk_ids", "cq3_armed_line") if row[k] is None]
    row["unquotable_missing_fields"] = missing  # per the pre-registration: a row missing ANY field is UNQUOTABLE
    rows[arm] = row
report = {
    "schema": "cq3-t3-metal-bout-v1", "issue": 1186, "cell": 1,
    "pre_registration": "riir-infer .issues/1186 — T3 pre-registration (2026-11-24 amendment)",
    "date": subprocess.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], capture_output=True, text=True).stdout.strip(),
    "host": subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip(),
    "ours_engine": "qwen38_wgpu_bench (CubeCL dense, RIIR_QWEN38_CQ3 plane posture)",
    "opponent_engine": "llama.cpp prism-b12128-d7b2896 Metal (llama-bench, the prep's pinned build)",
    "digests": digests, "rows": rows,
}
with open(os.path.join(out, "report.json"), "w") as f:
    json.dump(report, f, indent=2)
bad = [a for a, r in rows.items() if r["unquotable_missing_fields"]]
print("LEDGER WRITTEN:", os.path.join(out, "report.json"))
for arm, r in rows.items():
    print(f"  {arm}: decode128 a/b = {r['decode128_tok_s_run_a']}/{r['decode128_tok_s_run_b']} tok/s, "
          f"ctx2048 = {r['decode_ctx2048_tok_s_run_a']}, pp2048 = {r['pp2048_median_tok_s']}, "
          f"pp4096 = {r['pp4096_median_tok_s']}")
if bad:
    print(f"REFUSE: unquotable rows (missing ledger fields): {bad}")
    sys.exit(4)
print("ALL ROWS QUOTABLE (ledger fields complete; the verdict math is the lane's, out of band)")
PYEOF
}

leg provenance
leg build_ours
leg build_opponent
leg engagement
leg ours_cells
leg opponent_cells
leg ledger
echo "== bout_metal_cell1 DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =="
