#!/usr/bin/env python3
"""Issue 1117/1213 - one-bin sel reader for the mid-run checkpoint wakes.

The full collector (i1117_collect_128k.py) needs ALL THREE bins + the 32K
reference, so it cannot answer "does the fixed build's rho arm read sel ~= rho?"
the moment rho_0.15.bin lands. This reads ONE bin and prints the sel fraction
plus the digest/n_written/wall header facts - the checkpoint's decision input.

Bin layout is write_arm_bin's (see the collector's docstring): FMG1 | digest32
| steps u64 | complete u8 | n u64 | sel f64 | wall f64 | n x (nll f32, argmax u64).

Usage:
  python3 scripts/i1117_read_sel.py <bin> [<bin>...]
  python3 scripts/i1117_read_sel.py --selftest <bin>   # + compare vs an expected sel

Exit 0 = every bin parsed with complete=1; exit 1 = any parse/complete failure.
The EXPECTED value check is the caller's (wake instruction: sel must read
~= 0.15/0.30; v1's VOID bins read 0.0504/0.1081 - the collision's second face).
"""
import struct
import sys
from pathlib import Path

HEADER = struct.Struct("<4s32sQBQdd")  # magic, digest, steps, complete, n, sel, wall
HEADER_SIZE = 69


def read_header(path):
    b = Path(path).read_bytes()
    if len(b) < HEADER_SIZE:
        raise ValueError("truncated header: %s" % path)
    magic, digest, steps, complete, n, sel, wall = HEADER.unpack(b[:HEADER_SIZE])
    if magic != b"FMG1":
        raise ValueError("bad magic: %s" % path)
    return {
        "path": str(path),
        "digest": digest.hex(),
        "steps": steps,
        "complete": bool(complete),
        "n": n,
        "sel": sel,
        "wall_secs": wall,
        "bytes": len(b),
    }


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    if not args:
        print(__doc__)
        return 2
    rc = 0
    for a in args:
        try:
            b = read_header(a)
        except ValueError as e:
            print("FAIL: %s" % e)
            rc = 1
            continue
        print(
            "%s: sel=%.4f complete=%s n=%d steps=%d wall=%.0fs bytes=%d digest=%s…"
            % (
                b["path"],
                b["sel"],
                b["complete"],
                b["n"],
                b["steps"],
                b["wall_secs"],
                b["bytes"],
                b["digest"][:12],
            )
        )
        if not b["complete"]:
            print("  ^ complete=0 - in-flight prefix (header sel is valid only at completion)")
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
