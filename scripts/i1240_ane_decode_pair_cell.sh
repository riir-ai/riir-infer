#!/usr/bin/env bash
# i1240_ane_decode_pair_cell.sh — Issue 1240 T1: the ANE↔decode contention
# pair cell (research GOAT leg; riir-infer .issues/1240).
#
# Measures resident Metal decode tok/s (riir-ai riir-gpu bench_831
# DECODE_LONG, the plan-626 routed lane) ALONE vs UNDER a concurrent ANE
# batch-encode load (riir-infer-laya examples/i1240_ane_encode_load), in
# INTERLEAVED A/B pairs (sequential arms measure the box — the
# sequential_ab_timing_audit law; katgpt-rs tests/common/ab_timing.rs
# class), plus an ANE-alone window so the ENCODER's own delta is
# recorded beside the decode delta (the issue's both-deltas rule).
#
# Pre-registered bars (the issue's, read verbatim at adjudication):
#   decode delta <5%   = co-residency GO
#   decode delta 5-20% = scheduling-lane class
#   decode delta >20%  = NO-GO (background core refuted on this silicon)
#
# Preconditions this script REFUSES on (loud, never a silent pass):
#   - the box not AC-powered / the reflex bench_preflight refusal
#   - contender processes at an A-arm start (the perf_rematch classifier,
#     sourced — never a re-typed threshold copy; Issue 866's blind-zero law)
#   - missing model file / missing ANE artifacts / missing binaries
#
# The B arms run the ANE load BY DESIGN — bench_preflight's ANE axis
# would refuse there; the preflight runs ONCE at T0 (alone-posture
# provenance) and the B arms carry the standing disclosure instead.
#
# Usage:
#   scripts/i1240_ane_decode_pair_cell.sh --out .benchmarks/i1240_cells \
#     [--rounds 3] [--ane-secs 30] [--artifacts ~/riir-bench/i1240_ane_artifacts] \
#     [--dry-run]
#
# Exit: 0 all rounds landed · 1 a refusal / arm failure (the partial
# results stay on disk under --out).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFER_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
AI_ROOT="${RIIR_AI_ROOT:-$INFER_ROOT/../riir-ai}"
REFLEX_ROOT="${REFLEX_ROOT:-$INFER_ROOT/../riir-reflex}"
MODEL="${RIIR_MODEL_PATH:-/Users/katopz/git/riir-train/data/Ternary-Bonsai-2-27B-PQ2_0.gguf}"
DECODE_TARGET="/tmp/i831m5"
LOAD_TARGET="${I1240_LOAD_TARGET:-/tmp/i1240-target}"
DECODE_FEATURES="cubecl_runtime,ternary_gemv,deltanet_recurrence_rowpar,params_handle_cache,ternary_gemm_batched"
DECODE_RE='DECODE_LONG_MEDIAN ctx=[0-9]+ n=[0-9]+ runs=[0-9]+: ([0-9.]+) tok/s'

OUT="$INFER_ROOT/.benchmarks/i1240_cells"
ROUNDS=3
ANE_SECS=30
ARTIFACTS="$HOME/riir-bench/i1240_ane_artifacts"
CORPUS="$REFLEX_ROOT/.raw/datasets_t20k/massive_intent_en"
DRY_RUN=0

while [ $# -gt 0 ]; do case "$1" in
  --out) OUT="$2"; shift 2;;
  --rounds) ROUNDS="$2"; shift 2;;
  --ane-secs) ANE_SECS="$2"; shift 2;;
  --artifacts) ARTIFACTS="$2"; shift 2;;
  --dry-run) DRY_RUN=1; shift;;
  -h|--help) sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
  *) echo "i1240: unknown arg: $1" >&2; exit 3;;
esac; done

mkdir -p "$OUT"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%SZ)" "$*" | tee -a "$OUT/cell.log"; }
die() { log "REFUSED: $*"; exit 1; }

# ── the quiet gate: the SOURCED classifier, never a copy ────────────────
# PERF_REMATCH_LIB=1 source (the league_retry_driver pattern) exposes
# contender_candidates | classify_contenders; a clean sample is ZERO pids
# on stdout + exit 0. classify exits 3 on a blind vocabulary — refuse,
# never a pass (Issue 866 T3).
PERF_REMATCH_ROOT="$AI_ROOT"
PERF_REMATCH_LIB=1
# shellcheck disable=SC1091
source "$AI_ROOT/scripts/perf_rematch.sh"

quiet_gate() { # $1 label
  local errf out rc
  errf="$(mktemp "${TMPDIR:-/tmp}/i1240_err.XXXXXX")"
  rc=0
  out="$(contender_candidates | classify_contenders 2>"$errf")" || rc=$?
  if [ "$rc" -ne 0 ]; then
    sed 's/^/[classify] /' "$errf" >&2
    rm -f "$errf"
    die "$1: the contender classifier refused (exit $rc) — never a quiet pass"
  fi
  rm -f "$errf"
  if [ -n "$out" ]; then
    die "$1: contender processes present: $(echo "$out" | tr '\n' ';' | cut -c1-200)"
  fi
  log "$1: quiet gate clean"
}

# ── T0: provenance + posture ────────────────────────────────────────────
[ "$(uname -s)" = "Darwin" ] || die "the pair cell is a macOS lane (Metal + ANE)"
[ -f "$MODEL" ] || die "model missing: $MODEL"
[ -d "$ARTIFACTS" ] && [ -f "$ARTIFACTS/manifest.json" ] || \
  die "ANE artifacts missing under $ARTIFACTS (convert via reflex scripts/ane_convert.py)"
[ -d "$CORPUS" ] || die "corpus missing: $CORPUS"

DECODE_BIN="$DECODE_TARGET/release/deps"  # cargo test runs from the workspace
LOAD_BIN="$LOAD_TARGET/release/examples/i1240_ane_encode_load"
[ -x "$LOAD_BIN" ] || die "ANE load bin missing: $LOAD_BIN (cargo build --release -p riir-infer-laya --features laya-riir-ane --example i1240_ane_encode_load)"

# AC power + the bench_preflight PROVENANCE line (alone posture).
PREFLIGHT="$REFLEX_ROOT/scripts/bench_preflight.sh"
[ -x "$PREFLIGHT" ] || PREFLIGHT="$REFLEX_ROOT/scripts/bench_preflight.sh"
if [ -x "$PREFLIGHT" ] && [ "$DRY_RUN" -eq 0 ]; then
  PROV="$("$PREFLIGHT" 2>&1)" || die "bench_preflight refused: $PROV"
  echo "$PROV" | tee "$OUT/preflight.txt"
  log "T0 preflight provenance captured (see preflight.txt)"
fi

if [ "$DRY_RUN" -eq 1 ]; then
  log "dry-run: refusing nothing further; would run $ROUNDS rounds (A: decode alone, E: ANE alone ${ANE_SECS}s, B: decode under ANE)"
  exit 0
fi

quiet_gate "T0 start"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS="$OUT/cell_$STAMP"
mkdir -p "$RESULTS"
echo "cell $STAMP rounds=$ROUNDS ane_secs=$ANE_SECS model=$(basename "$MODEL")" > "$RESULTS/meta.txt"
uname -m; sw_vers 2>/dev/null | head -3; sysctl -n hw.memsize >> "$RESULTS/meta.txt" 2>/dev/null || true
log "results dir: $RESULTS"

# ── the decode arm: bench_831 DECODE_LONG, prints DECODE_LONG_MEDIAN ────
run_decode() { # $1 log-file -> prints median tok/s on stdout
  local logfile="$1"
  (cd "$AI_ROOT" && \
   RIIR_MODEL_PATH="$MODEL" DECODE_CTX=2048 DECODE_TOKENS=512 DECODE_RUNS=3 \
   CARGO_TARGET_DIR="$DECODE_TARGET" \
   cargo test -p riir-gpu --features "$DECODE_FEATURES" --release \
     --test bench_831_decode_long -- --ignored --nocapture --test-threads=1 \
     > "$logfile" 2>&1) || { tail -20 "$logfile" >&2; die "decode arm failed (log $logfile)"; }
  local med
  med="$(sed -nE "s@.*$DECODE_RE.*@\1@p" "$logfile" | tail -1)"
  [ -n "$med" ] || { tail -20 "$logfile" >&2; die "no DECODE_LONG_MEDIAN line in $logfile"; }
  echo "$med"
}

# ── the ANE load arm: runs until killed; READY gates the decode start ───
start_ane_load() { # $1 log-file -> echoes the pid (kills via kill_ane_load)
  LAYA_ANE_ARTIFACTS_DIR="$ARTIFACTS" \
    "$LOAD_BIN" --checkpoint english --corpus "$CORPUS" \
      --secs 100000 --warmup 8 --tick-secs 10 > "$1" 2>&1 &
  echo $!
}

wait_ready() { # $1 log-file $2 pid
  for _ in $(seq 1 120); do
    if ! kill -0 "$2" 2>/dev/null; then
      tail -20 "$1" >&2; die "ANE load died before READY"
    fi
    grep -q '^ANE_LOAD_READY' "$1" && return 0
    sleep 1
  done
  tail -20 "$1" >&2
  die "ANE load never reached READY (120s)"
}

kill_ane_load() { # $1 pid $2 log-file — SIGTERM, then harvest the TOTAL
  kill "$1" 2>/dev/null || true
  for _ in $(seq 1 20); do kill -0 "$1" 2>/dev/null || break; sleep 0.5; done
  kill -9 "$1" 2>/dev/null || true
  grep '^ANE_LOAD_TOTAL' "$2" | tail -1 || true
}

# ── the interleaved pairs ────────────────────────────────────────────────
SUMMARY="$RESULTS/summary.md"
printf '| round | decode alone tok/s | decode under ANE tok/s | decode delta | ane alone (E) | ane under decode (B) |\n|---|---|---|---|---|---|\n' > "$SUMMARY"

for round in $(seq 1 "$ROUNDS"); do
  # A — decode ALONE (quiet gate first: the arm must not inherit a sibling
  # launched during the previous arm — the per-ARM pre-gate law, Issue 854 T4).
  quiet_gate "round $round A-arm pre"
  log "round $round: A arm (decode alone) starting"
  A_MED="$(run_decode "$RESULTS/round${round}_a_decode_alone.log")"
  log "round $round: A arm decode = $A_MED tok/s"

  # E — ANE ALONE window (the encoder's own baseline).
  quiet_gate "round $round E-arm pre"
  E_LOG="$RESULTS/round${round}_e_ane_alone.log"
  LOAD_PID="$(start_ane_load "$E_LOG")"
  wait_ready "$E_LOG" "$LOAD_PID" || true
  sleep "$ANE_SECS"
  E_TOTAL="$(kill_ane_load "$LOAD_PID" "$E_LOG")"
  log "round $round: E arm ane-alone: $E_TOTAL"

  # B — decode UNDER the ANE load (the pair leg).
  quiet_gate "round $round B-arm pre"
  B_LOG="$RESULTS/round${round}_b_pair.log"
  L_LOG="$RESULTS/round${round}_b_ane_load.log"
  LOAD_PID="$(start_ane_load "$L_LOG")"
  wait_ready "$L_LOG" "$LOAD_PID" || true
  sleep 2   # let the ANE reach steady state before the decode window opens
  log "round $round: B arm (decode under ANE) starting"
  B_MED="$(run_decode "$RESULTS/round${round}_b_decode.log")"
  B_ANE="$(kill_ane_load "$LOAD_PID" "$L_LOG")"
  log "round $round: B arm decode = $B_MED tok/s under ANE ($B_ANE)"

  # Interleave verdict row (delta as the A-vs-B ratio, 4dp).
  DELTA="$(awk -v a="$A_MED" -v b="$B_MED" 'BEGIN{printf "%.4f", b/a}')"
  printf '| %s | %s | %s | %s | %s | %s |\n' "$round" "$A_MED" "$B_MED" "$DELTA" \
    "$(echo "$E_TOTAL" | grep -oE 'tps=[0-9.]+' || echo n/a)" \
    "$(echo "$B_ANE" | grep -oE 'tps=[0-9.]+' || echo n/a)" >> "$SUMMARY"
done

log "cell complete — $ROUNDS rounds; summary: $SUMMARY"
cat "$SUMMARY"
