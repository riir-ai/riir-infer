#!/bin/bash
# lens_t3 collection watcher — katgpt-rs 999 (amo, 2026-11-16).
# Waits for the fit (pid $1) to exit, then copies the artifacts to a
# reboot-safe location and writes a marker. Belt-and-braces beside the
# 15:30 agent-mail wake: if the wake lane doesn't run, the artifacts still
# leave /tmp. No mail from here (the bare-shell whoami trap) — just the
# copy + marker; the wake reads them.
PID="$1"
OUT=/tmp/lens_t3_run
SAFE="$HOME/lens_t3_run_backup"
LOG=/tmp/lens_t3_run.log

while kill -0 "$PID" 2>/dev/null; do
  sleep 60
done

echo "[watcher] fit pid $PID exited at $(date '+%F %T %z')" >> /tmp/lens_t3_watcher.log
if [ -f "$OUT/manifest.json" ]; then
  mkdir -p "$SAFE"
  cp "$OUT"/manifest.json "$OUT"/fit_all.k "$OUT"/fit_all.j \
     "$OUT"/fit_even.k "$OUT"/fit_even.j \
     "$OUT"/fit_odd.k "$OUT"/fit_odd.j \
     "$OUT"/ckpt.json "$SAFE"/ 2>>/tmp/lens_t3_watcher.log &&
    echo "[watcher] artifacts copied to $SAFE" >> /tmp/lens_t3_watcher.log
  cp "$LOG" "$SAFE"/run.log 2>>/tmp/lens_t3_watcher.log
  touch "$SAFE/COLLECTED"
  echo "[watcher] marker written; the 999 wake lane can collect" >> /tmp/lens_t3_watcher.log
else
  echo "[watcher] NO manifest.json — the fit died before completing; check $LOG and ckpt.sums (relaunch with --resume)" >> /tmp/lens_t3_watcher.log
fi
