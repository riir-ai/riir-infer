#!/bin/bash
# i1201 bar-rung resume — run 16_2048_s0 on bmo (issue 1201 T3(g), plan 630).
# Third attempt of the (16, 2048, seed 0) @ frac 1/16 rung. Stop forensics
# (2026-11-20, zed-00f2c1ae@amo): stop 1 (11-17 13:23 +07) and stop 2
# (11-18 ~22:03 +07, 58 min after the claimant's resume) both killed the
# runner silently — no crash report, no jetsam, no sleep entry; the
# visible fingerprint is caffeinate releasing 51 s after launch (the ssh
# channel dying) and stderr never captured. This attempt runs UNDER THE
# RIG DAEMON (a death now mails), captures BOTH streams, and holds
# caffeinate for the whole run. bmo rebooted 11-20 01:18 +07 for the
# 27.2.1 update; softwareupdate --list is EMPTY (no pending restart).
set -uo pipefail
RUN_DIR="$HOME/git/riir-infer/.benchmarks/flashstates_cell/16_2048_s0"
REPO="$HOME/git/riir-infer"
MODEL="/Users/katopz/uzu27b/models/qwen3.8-27b-Q4_K_M.gguf"
LOG="$RUN_DIR/job_20261120T1100_resume.log"

mkdir -p "$RUN_DIR"
exec > >(tee -a "$LOG") 2>&1

echo "PROVENANCE i1201-rung1-resume3 start $(date '+%Y-%m-%dT%H:%M:%S%z') host $(hostname) os $(sw_vers -productVersion)-$(sw_vers -buildVersion)"
echo "single-instance guard:"
# Exact process-name match: the run dir path appears in this script's own
# tee argv, so a -f path regex would self-match (measured: job 01a1f705
# refused itself on exactly that). pgrep -x cannot match bash/tee/rustc.
if pgrep -x flashstates_cell > /dev/null 2>&1; then
  echo "REFUSED: a flashstates_cell process is already alive"
  exit 4
fi

cd "$REPO" || exit 3
git fetch origin develop 2>&1 | tail -1
git checkout --force develop 2>&1 | tail -1
# Untracked-source scrub ONLY — never a pathspec clean over .benchmarks:
# the run dirs are untracked, so `git clean -fd -- .benchmarks` DELETED the
# 16_2048_s0 run state (a0 rows + a0p cursor, measured 10:38 +07 today,
# job 01a1f708) and two sibling lanes' bench dirs. -e exclusions are the
# safe form; the runner's own manifest/rows live under .benchmarks.
git clean -qfd -e target -e .benchmarks -e .raw -e .deploy || true
HEAD_SHA="$(git rev-parse --short HEAD)"
echo "riir-infer @ $HEAD_SHA (binary rebuilt only if the Nov-18 binary's source moved)"

# The warm binary predates b9de7d8's tree only by docs commits; but a
# rebuild is cheap insurance against a stale-binary-silent-mismatch —
# the manifest digest check inside the bin is the real gate.
cargo build --release --bin flashstates_cell --features flashstates_cell 2>&1 | tail -3
BIN="$REPO/target/release/flashstates_cell"
[ -x "$BIN" ] || { echo "REFUSED: no flashstates_cell binary after build"; exit 3; }

echo "model: $(ls -la "$MODEL" | awk '{print $5, $9}')"
shasum -a 256 "$MODEL" | head -1

caffeinate -dis "$BIN" \
  --n 16 --cap 2048 --seed 0 --ctx 5120 --chunk 512 \
  --arm a0,a0p,a1,a2 \
  --budget-frac 0.0625 \
  --g1-sample 8 \
  --gguf "$MODEL" \
  --out-dir "$RUN_DIR" \
  --resume
RC=$?
echo "DONE-MARKER rc=$RC at $(date '+%Y-%m-%dT%H:%M:%S%z')"
ls -la "$RUN_DIR"
exit "$RC"
