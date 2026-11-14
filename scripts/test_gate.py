#!/usr/bin/env python3
"""riir-infer standing test gate (Issue 1215).

The EXECUTE half of this repo's test story: the floors below pin the
passed-COUNT of the standing suites so a target that compiles to
nothing, skips silently, or loses tests cannot report green.

Laws (the katgpt-rs / riir-train test_gate shape, scoped here):
- A floor fires DOWNWARD only: adding tests never reds the gate;
  deleting tests, a feature change that compiles a lib to nothing, or
  a broken build reds it.
- Exactly ONE ``test result:`` line is expected per invocation. More
  or fewer means the parse has gone blind -- fail loudly rather than
  sum or guess (the #![cfg]-gated-file green-zero trap: ``ok. 0
  passed`` with exit 0 is byte-for-byte a real pass; a skipped target
  produces no line at all).
- --canary runs the FIRST row with an ABSOLUTE floor of 100000 and
  asserts the gate FAILS -- the proof the floors are live. Dev-time.
- The Metal row SKIPs LOUD (a deferral, never a green) off macOS.
- The receipt row runs ONLY with --receipt (it costs ~31 min: the full
  G3 eval + the receipt hash at the release posture). It SKIPs LOUD
  without I1214_ARTIFACT_DIR set.

Floors measured 2026-11-14 on amo (M5 Ultra, AC, load 7.8-9.6) at
develop 086a2bb; the release-posture receipt cell green the same night
(G3 ppl 22.931885 + logits hash 22d4844a... bit-identical to bench
1213's pins). Raising a floor is a measurement act: re-run the suite,
land the new number with its box state in the commit message.

Usage:
    python3 scripts/test_gate.py              # the standing rows
    python3 scripts/test_gate.py --canary     # floors-must-be-live proof
    python3 scripts/test_gate.py --receipt    # + the real-artifact cell

Exit codes: 0 all pass, 1 any row failed, 2 blindness (no rows ran /
every row went blind).
"""

import argparse
import os
import platform
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Row: (name, package, features, target_args, filter_args, floor, timeout_s)
# - target_args names the cargo test target surface ("--lib" or "--test X").
# - filter_args is the trailing test-name filter ("" = none).
# - floor is a passed-count floor (downward-firing).
# Rows measured 2026-11-14 (amo, AC, load 7.8–9.6) — T2 batch (issue 1215):
# the four CPU-only integration/lib suites join with measured floors; the
# entropy_bounded_commit lane does NOT (see the issue: the heavy gemma2_2b
# GPU tests SIGKILL on this shared box's memory pressure — the lane
# verifies on dedicated GPU boxes; its 6-test EB battery passes standalone).
ROWS = [
    (
        "int_ref CPU core suite",
        "riir-infer-laya",
        "laya-riir,int_ref",
        "--lib",
        "int_ref",
        67,
        1200,
    ),
    (
        "int_ref G1(c) GPU twin (Metal)",
        "riir-infer-laya",
        "laya-riir,int_ref,laya-riir-metal",
        "--lib",
        "int_ref",
        70,
        1200,
    ),
    (
        "int_ref python mirror (double transcription)",
        "riir-infer-laya",
        "int_ref",
        "--test int_ref_python_mirror",
        "",
        8,
        1200,
    ),
    (
        "issue879 GDN quant certification (reduced arms)",
        "riir-infer-core",
        "deltanet_ternary_inference",
        "--release --test issue879_gdn_quant_certification",
        "",
        2,
        1800,
    ),
    (
        "bonsai2 rotation loader contract",
        "riir-infer-core",
        "bonsai2_hadamard",
        "--release --test bonsai2_rotation_load",
        "",
        11,
        600,
    ),
    (
        "issue046 GDN spectral class spec",
        "riir-infer-core",
        "deltanet_inference",
        "--release --test issue046_gdn_spectral_class_spec",
        "",
        4,
        600,
    ),
    (
        "exl3 trellis lane (lib)",
        "riir-infer-core",
        "exl3",
        "--release --lib",
        "",
        308,
        1200,
    ),
]

# The slow receipt row (--receipt only). Floor 2: G3 PPL + the receipt.
# Also asserts the receipt line's hash prefix -- the loud-zero defence:
# a SKIP (artifact unset / eval ids missing) kills the test binary
# BEFORE any `test result:` line, which the one-line law already reds.
RECEIPT_ROW = (
    "int_ref real-artifact receipt (release posture)",
    "riir-infer-laya",
    "int_ref",
    "--release --test int_ref_real_artifact",
    "",
    2,
    7200,
)

RECEIPT_HASH_PREFIX = "logits_final=22d4844a"


def run_row(name, pkg, features, target_args, filter_args, floor, timeout_s,
            canary_floor=None):
    """Run one row; return (ok, detail). Never raises on row failure."""
    cmd = ["cargo", "test", "-p", pkg, "--features", features]
    cmd.extend(target_args.split())
    cmd.append("--")
    cmd.append("--test-threads=2")
    if filter_args:
        cmd.append(filter_args)

    effective_floor = floor
    label = "%s [%s] (floor %d)" % (name, features, floor)
    if canary_floor is not None:
        effective_floor = canary_floor
        label = "CANARY %s (floor %d)" % (name, canary_floor)
    print("=== %s ===" % label)

    env = dict(os.environ)
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            timeout=timeout_s,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        out = proc.stdout.decode("utf-8", "replace")
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        print("FAIL %s: timed out after %ds" % (name, timeout_s))
        return False, "timeout"
    except OSError as exc:
        print("FAIL %s: cargo spawn failed: %s" % (name, exc))
        return False, "spawn-failed"

    lines = [
        ln for ln in out.splitlines() if ln.strip().startswith("test result:")
    ]
    if len(lines) != 1:
        print(
            "FAIL %s: expected exactly 1 'test result:' line, got %d "
            "-- the parse is blind, the target skipped, or the shape "
            "changed" % (name, len(lines))
        )
        tail = out.splitlines()[-12:]
        for ln in tail:
            print("    | %s" % ln)
        return False, "no-single-result-line"

    result = lines[0].split()
    # "test result: ok. N passed; ..." -> the number before "passed;"
    passed = None
    for i, tok in enumerate(result):
        if tok == "passed;":
            passed = int(result[i - 1])
            break
    if passed is None:
        print("FAIL %s: could not parse the passed count from: %s"
              % (name, lines[0]))
        return False, "parse-failed"

    verdict_ok = rc == 0 and passed >= effective_floor
    print("    cargo rc=%d passed=%d floor=%d -> %s"
          % (rc, passed, effective_floor, "OK" if verdict_ok else "FAIL"))
    if not verdict_ok:
        for ln in out.splitlines()[-12:]:
            print("    | %s" % ln)
        return False, "floor-or-rc"

    if RECEIPT_HASH_PREFIX and "--test int_ref_real_artifact" in target_args:
        if RECEIPT_HASH_PREFIX not in out:
            print(
                "FAIL %s: passed but the receipt line carries no %s "
                "-- the receipt posture drifted or the eval skipped"
                % (name, RECEIPT_HASH_PREFIX)
            )
            return False, "receipt-hash-missing"
        print("    receipt hash prefix %s present" % RECEIPT_HASH_PREFIX)
    return True, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canary", action="store_true",
                    help="run the first row at floor=100000; MUST fail")
    ap.add_argument("--receipt", action="store_true",
                    help="also run the slow real-artifact receipt row")
    args = ap.parse_args()

    rows = list(ROWS)
    if args.receipt:
        if not os.environ.get("I1214_ARTIFACT_DIR"):
            print(
                "SKIP LOUD (receipt row): I1214_ARTIFACT_DIR not set -- "
                "the receipt needs the calibrated artifact, never a "
                "green zero"
            )
        else:
            rows.append(RECEIPT_ROW)

    if platform.system() != "Darwin":
        print(
            "SKIP LOUD (Metal row): not macOS -- the G1(c) GPU-twin "
            "suite compiles to nothing here; this is a deferral, not a "
            "green"
        )
        rows = [r for r in rows if "laya-riir-metal" not in r[2]]

    if not rows:
        print("FAIL: no rows to run -- the population is empty")
        return 2

    failures = 0
    for i, row in enumerate(ROWS):
        name, pkg, features, target_args, filter_args, floor, timeout_s = row
        if i == 0 and args.canary:
            ok, _ = run_row(name, pkg, features, target_args, filter_args,
                            floor, timeout_s, canary_floor=100000)
            if ok:
                print("CANARY DID NOT FIRE -- the floors are not live")
                return 1
            print("canary fired: the floor comparison path is live")
            return 0
        ok, _ = run_row(name, pkg, features, target_args, filter_args,
                        floor, timeout_s)
        if not ok:
            failures += 1

    if args.receipt and os.environ.get("I1214_ARTIFACT_DIR"):
        name, pkg, features, target_args, filter_args, floor, timeout_s = \
            RECEIPT_ROW
        ok, _ = run_row(name, pkg, features, target_args, filter_args,
                        floor, timeout_s)
        if not ok:
            failures += 1

    print("")
    if failures:
        print("test_gate: FAIL (%d row(s) red)" % failures)
        return 1
    print("test_gate: PASS (%d row(s) green)" % len(
        [r for r in rows]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
