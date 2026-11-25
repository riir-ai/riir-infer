#!/usr/bin/env python3
"""Dump GGUF v3 tensor names + shapes + dtypes (header-only, Issue 1250 T1 recon)."""
import struct
import sys

GGML_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
              8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
              13: "Q5_K", 14: "Q6_K", 15: "IQ2_XXS", 16: "IQ2_XS",
              18: "Q2_0", 30: "BF16"}


def main(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"GGUF", magic
        version, = struct.unpack("<I", f.read(4))
        n_tensors, = struct.unpack("<Q", f.read(8))
        n_kv, = struct.unpack("<Q", f.read(8))

        def read_str():
            n, = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", "replace")

        def read_val(t):
            if t == 0:
                return struct.unpack("<B", f.read(1))[0]
            if t == 1:
                return struct.unpack("<b", f.read(1))[0]
            if t == 2:
                return struct.unpack("<H", f.read(2))[0]
            if t == 3:
                return struct.unpack("<h", f.read(2))[0]
            if t == 4:
                return struct.unpack("<I", f.read(4))[0]
            if t == 5:
                return struct.unpack("<i", f.read(4))[0]
            if t == 6:
                return struct.unpack("<f", f.read(4))[0]
            if t == 7:
                return struct.unpack("<B", f.read(1))[0]
            if t == 8:
                return read_str()
            if t == 9:
                et, = struct.unpack("<I", f.read(4))
                n, = struct.unpack("<Q", f.read(8))
                return [read_val(et) for _ in range(n)]
            if t == 10:
                return struct.unpack("<Q", f.read(8))[0]
            if t == 11:
                return struct.unpack("<q", f.read(8))[0]
            if t == 12:
                return struct.unpack("<d", f.read(8))[0]
            raise ValueError(f"kv type {t}")

        for _ in range(n_kv):
            k = read_str()
            t, = struct.unpack("<I", f.read(4))
            read_val(t)
        rows = []
        for _ in range(n_tensors):
            name = read_str()
            n_dim, = struct.unpack("<I", f.read(4))
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dim)]
            dtype, = struct.unpack("<I", f.read(4))
            off, = struct.unpack("<Q", f.read(8))
            rows.append((name, dims, GGML_TYPES.get(dtype, str(dtype))))
        for name, dims, dt in rows:
            print(f"{dt:>6} {'x'.join(map(str, dims)):>16}  {name}")


if __name__ == "__main__":
    main(sys.argv[1])
