#!/bin/zsh
# i1178 T2b price cell — lane-sliced Q4 decode GEMV, RUN 2 (v2).
# Run 1 (rig job 01a1bb89) is VOID: zsh does not word-split $ENVSET, so `env`
# received ONE arg (NAME=RIIR_QWEN38_DENSE_MM2D, VALUE="0 RIIR_QWEN38_ATTN_TENSOR=0 …")
# — the demote literal never matched, all three arms measured the league f16
# posture, ls_twin=0. v2 delivers the env via export/unset (shell-agnostic) and
# adds FAIL-FAST engagement checks after each packed leg1 (the preregistered
# engagement gate enforced EARLIER — a void run now costs one leg, not six).
# Arms + gates + decision rules are UNCHANGED from the issue's preregistration.
set -u
export PATH=/opt/homebrew/bin:$PATH
WT="$HOME/git/riir-infer-t2bwt"
LOGDIR=/tmp/i1178_t2b_v2
mkdir -p "$LOGDIR"
echo "== i1178 T2b job v2 start $(date -u +%FT%TZ) =="

# Advisory sibling-engine wait (splash cell may be draining). Up to 20 min, advisory only.
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
  if pgrep -if 'splash' >/dev/null 2>&1 || pgrep -f 'uzu-engine|uzu serve|uzu27b.*serve' >/dev/null 2>&1; then
    echo "[guard] sibling engine process alive (leg $i); waiting 60s"; sleep 60
  else
    echo "[guard] no sibling engine process (leg $i)"; break
  fi
done

# mds tripwire (Bench-950 lesson): drain up to 5 min, hard-refuse exit 3.
mds_cpu() { ps -A -o %cpu,comm | grep -E 'mds_stores|/[m]ds$' | awk '{s+=$1} END {print s+0}'; }
for i in 1 2 3 4 5; do
  c=$(mds_cpu)
  if awk "BEGIN{exit !($c > 5)}"; then echo "[mds] ${c}% > 5%; draining 60s (leg $i)"; sleep 60; else break; fi
done
c=$(mds_cpu)
if awk "BEGIN{exit !($c > 5)}"; then echo "[mds] REFUSE exit 3: mds_stores at ${c}% after drain"; exit 3; fi
echo "[mds] clean at ${c}%"

cd "$WT" || exit 20
git --no-pager log --oneline -1
export CARGO_TARGET_DIR="$WT/target"

echo "== build OFF (league feature set) =="
cargo build --release -p riir-infer-gpu --features qwen38_wgpu,production --bin qwen38_wgpu_bench 2>&1 | tail -3
[ -x target/release/qwen38_wgpu_bench ] || { echo "BUILD OFF FAILED"; exit 10; }
cp target/release/qwen38_wgpu_bench /tmp/t2b_bench_OFF
echo "== build ON (+q4k_lanesliced_decode_gemv) =="
cargo build --release -p riir-infer-gpu --features qwen38_wgpu,production,q4k_lanesliced_decode_gemv --bin qwen38_wgpu_bench 2>&1 | tail -3
[ -x target/release/qwen38_wgpu_bench ] || { echo "BUILD ON FAILED"; exit 11; }
cp target/release/qwen38_wgpu_bench /tmp/t2b_bench_ON

BOX="uptime: $(uptime) | ac: $(pmset -g ps | head -1 | tr -s ' ')"

packed_env() {
  if [ "$1" = "1" ]; then
    export RIIR_QWEN38_DENSE_MM2D=0 RIIR_QWEN38_ATTN_TENSOR=0 RIIR_QWEN38_DENSE_MM2D_F16ACT=0
  else
    unset RIIR_QWEN38_DENSE_MM2D RIIR_QWEN38_ATTN_TENSOR RIIR_QWEN38_DENSE_MM2D_F16ACT
  fi
}

run_arm() {
  tag=$1; bin=$2; packed=$3
  packed_env "$packed"
  echo "== arm $tag leg1 (pp1536 + decode512 + ctx1344 + walk64) env: RIIR_QWEN38_DENSE_MM2D=${RIIR_QWEN38_DENSE_MM2D:-unset} ATTN_TENSOR=${RIIR_QWEN38_ATTN_TENSOR:-unset} F16ACT=${RIIR_QWEN38_DENSE_MM2D_F16ACT:-unset} =="
  env "$bin" --pp 1536 --decode 512 --decode-ctx 1344 --walk > "$LOGDIR/${tag}_leg1.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then echo "ARM $tag LEG1 FAILED rc=$rc — log tail:"; tail -20 "$LOGDIR/${tag}_leg1.log"; exit 12; fi
  # FAIL-FAST engagement gate (the preregistered VOID rule, enforced after leg1):
  if [ "$packed" = "1" ]; then
    d=$(grep -c 'DEMOTED by RIIR_QWEN38_DENSE_MM2D=0' "$LOGDIR/${tag}_leg1.log")
    [ "$d" -ge 1 ] || { echo "ENGAGEMENT FAIL $tag: dense demote line missing (demoted=$d) — VOID, exit 14"; tail -5 "$LOGDIR/${tag}_leg1.log"; exit 14; }
  else
    a=$(grep -c 'dense-f16 matmul2d arm armed' "$LOGDIR/${tag}_leg1.log")
    [ "$a" -ge 1 ] || { echo "ENGAGEMENT FAIL $tag: f16 armed line missing (armed=$a) — VOID, exit 15"; tail -5 "$LOGDIR/${tag}_leg1.log"; exit 15; }
  fi
  echo "== arm $tag leg2 (decode512 drift leg) =="
  env "$bin" --decode 512 > "$LOGDIR/${tag}_leg2.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then echo "ARM $tag LEG2 FAILED rc=$rc — log tail:"; tail -20 "$LOGDIR/${tag}_leg2.log"; exit 13; fi
  echo "-- $tag posture lines:"
  grep -hE 'Issue 1150|Issue 1176|1178 T2b' "$LOGDIR/${tag}_leg1.log" | sort | uniq -c | sort -rn | head -6
  echo "-- $tag engagement: demoted=$(grep -c 'DEMOTED' "$LOGDIR/${tag}_leg1.log") armed=$(grep -c 'dense-f16 matmul2d arm armed' "$LOGDIR/${tag}_leg1.log") ls_twin=$(grep -c '1178 T2b' "$LOGDIR/${tag}_leg1.log")"
  echo "-- $tag cells:"
  grep -hE '\[decode|\[pp |\[decode-ctx|loaded in' "$LOGDIR/${tag}_leg1.log" "$LOGDIR/${tag}_leg2.log"
}

run_arm packedOFF /tmp/t2b_bench_OFF 1
# packedON: the twin must have armed — check INSIDE run_arm would need the
# feature; do it here after leg1 (fail fast before its leg2 + league arms).
lsn=$(grep -c '1178 T2b' "$LOGDIR/packedOFF_leg1.log")
[ "$lsn" -ge 1 ] || { echo "ENGAGEMENT FAIL: packedOFF built WITHOUT the twin? (ls_twin=$lsn — expected 0; continuing)"; }
run_arm packedON  /tmp/t2b_bench_ON  1
lsn=$(grep -c '1178 T2b' "$LOGDIR/packedON_leg1.log")
[ "$lsn" -ge 1 ] || { echo "ENGAGEMENT FAIL packedON: ls_twin=$lsn (bar: >=1) — VOID, exit 16"; tail -5 "$LOGDIR/packedON_leg1.log"; exit 16; }
run_arm leagueOFF /tmp/t2b_bench_OFF 0

echo "== G1 walk pins (md5 of the [walk] ids line) =="
for f in "$LOGDIR"/packedOFF_leg1.log "$LOGDIR"/packedON_leg1.log; do
  echo "$f : $(grep '\[walk\] ids' "$f" | md5)"
done
diff <(grep '\[walk\] ids' "$LOGDIR/packedOFF_leg1.log") <(grep '\[walk\] ids' "$LOGDIR/packedON_leg1.log") >/dev/null 2>&1 \
  && echo "G1: walk ids IDENTICAL packedOFF==packedON" \
  || echo "G1: walk ids DIVERGE packedOFF!=packedON — count + disclose per the 940 law"

echo "== box: $BOX"
echo "== i1178 T2b job v2 end $(date -u +%FT%TZ) =="
