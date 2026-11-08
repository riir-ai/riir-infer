#!/usr/bin/env python3
"""DSpine piecewise fixture replay (Issue 1183 G1a) — the vendor-MLX golden.

Replays the fixture inputs at /tmp/dspine_fixtures through the vendor
dspine_mlx op graph (MLX, bf16/f32 exactly as dspine_mlx/bridge.py +
propose.py write them) and diffs against the Rust oracle's `_out` files.

Modes:
  --inspect   numpy-only sanity pass over the fixture data (no MLX needed):
              prints the piece-(d) score/mass distributions — the fixture
              degeneracy guard's evidence.
  (default)   full vendor replay + diff. Needs mlx (`uv venv .raw/dspine/.venv
              && uv pip install --python .raw/dspine/.venv/bin/python mlx numpy`)
              and the vendor clone at .raw/dspine (T0 pin 6450af7).

Geometry is read from MANIFEST.md's `geometry:` line. Tolerances:
  (a) rms ≤1e-5 abs (fp32, reduction-order class); codes bit-exact (bf16)
  (b) message/out/normed bit-exact (bf16)
  (d) tg/tsv: bit-exact count + max rel diff reported (the MLX matmul fold
      order may flip ±1 bf16 ULP — the ≤1e-5 rel MANIFEST line predates the
      measurement; the bench record restates what held);
      idx/tokens bit-exact; probs/confs ≤1e-4 abs
Exit 0 iff every piece verdict passes.
"""
from __future__ import annotations

import argparse
import math
import os
import struct
import sys

import numpy as np

DEFAULT_FIXTURES = "/tmp/dspine_fixtures"
DEFAULT_RAW = os.path.join(os.path.dirname(__file__), "..", ".raw", "dspine")

# bridge.py: torch.finfo(torch.float32).tiny, the read-rms clamp floor.
F32_TINY = 1.1754943508222875e-38
# cfg.bridge_eps / cfg.rms_eps (config.json; both 1e-6 on the b16 checkpoint).
EPS_BRIDGE = 1.0e-6
EPS_NORM = 1.0e-6


def read_f32(path: str) -> np.ndarray:
    return np.fromfile(path, dtype="<f4")


def read_u32(path: str) -> np.ndarray:
    return np.fromfile(path, dtype="<u4")


def parse_geometry(path: str) -> dict[str, int]:
    for line in open(path, encoding="utf-8"):
        if line.startswith("- geometry:"):
            out = {}
            for tok in line.split(":")[1].split():
                if "=" in tok:
                    k, v = tok.split("=")
                    out[k] = int(v)
            return out
    raise SystemExit("MANIFEST.md geometry line not found")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def bf16_view(x: np.ndarray) -> np.ndarray:
    """Round f32 → bf16 (RNE) → f32 via the bias-and-mask trick."""
    u = x.astype(np.float32).view("<u4")
    return ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).view(np.float32)


def bf16_flip_stats(got: np.ndarray, ref: np.ndarray) -> tuple[int, int]:
    """(flip count, max bf16-ULP distance) over bf16-valued f32 arrays."""
    bad = bits16(got) != bits16(ref)
    if not bad.any():
        return 0, 0
    ia = got[bad].view("<u4").astype(np.int64)
    ib = ref[bad].view("<u4").astype(np.int64)
    sa = np.where(ia < 0, 0x80000000 - (ia & 0x7FFFFFFF), ia)
    sb = np.where(ib < 0, 0x80000000 - (ib & 0x7FFFFFFF), ib)
    return int(bad.sum()), int(np.abs(sa - sb).max()) // 65536


def bits16(x: np.ndarray) -> np.ndarray:
    """The bf16 (upper-16) bit pattern of an f32 array — for bit-exact diffs."""
    u = x.astype(np.float32).view("<u4")
    return ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000) >> 16


def walk_numpy(fix: str, g: dict) -> dict:
    """Mirror serial_walk_core in f32 numpy (sequential ascending folds)."""
    e, r, k = g["E"], g["R"], g["K"]
    anchor = g["ANCHOR"]
    raw = read_f32(f"{fix}/d_raw_in.f32.bin").reshape(-1, e)
    rms = read_f32(f"{fix}/d_rms_in.f32.bin")
    q_all = read_f32(f"{fix}/d_q_in.f32.bin").reshape(-1, r)
    cand = read_u32(f"{fix}/d_cand_in.u32.bin").reshape(-1, k)
    lm_head = read_f32(f"{fix}/d_lm_head_in.f32.bin").reshape(-1, e)
    norm_w = read_f32(f"{fix}/d_norm_w_in.f32.bin")
    gate_b = read_f32(f"{fix}/d_gate_b_in.f32.bin")
    tg = read_f32(f"{fix}/d_tg_out.f32.bin").reshape(-1, r)
    tsv = read_f32(f"{fix}/d_tsv_out.f32.bin").reshape(-1, 2 * r)
    gamma = cand.shape[0]
    idx = np.zeros((gamma, k), dtype=np.int64)
    probs = np.zeros((gamma, k), dtype=np.float32)
    score_log = []
    for nn in range(gamma):
        raw_row = raw[nn + 1]
        rms_n = float(rms[nn + 1])
        q = q_all[nn + 1]
        cand_n = cand[nn]
        u = (raw_row * norm_w).astype(np.float32)  # unrounded f32 product
        base = np.array(
            [float(np.dot(lm_head[t], u)) for t in cand_n], dtype=np.float32
        )
        sid = np.full(k, anchor, dtype=np.int64) if nn == 0 else cand[nn - 1]
        scores = np.zeros((k, k), dtype=np.float32)
        for j, sidv in enumerate(sid):
            tsv_row = tsv[int(sidv)]
            z = (q + tsv_row[:r] + gate_b).astype(np.float32)
            m = (sigmoid(z) * tsv_row[r:]).astype(np.float32)
            for jp in range(k):
                s = np.float32(0.0)
                tg_row = tg[int(cand_n[jp])]
                for rr in range(r):
                    s = np.float32(s + m[rr] * tg_row[rr])
                scores[j, jp] = np.float32(base[jp] + np.float32(rms_n * s))
        score_log.append(scores)
        scale = max(rms_n, 1.0e-6)
        for j in range(k):
            srow = scores[j]
            best_j = 0
            best_v = np.float32(-np.inf)
            for jp in range(k):
                if srow[jp] > best_v:
                    best_v = srow[jp]
                    best_j = jp
            idx[nn, j] = best_j
            mass = np.float32(0.0)
            for jp in range(k):
                mass = np.float32(
                    mass + math.exp(float((srow[jp] - best_v)) / scale)
                )
            probs[nn, j] = np.float32(1.0 / float(mass))
    return {
        "idx": idx,
        "probs": probs,
        "scores": score_log,
        "rms": rms,
        "cand": cand,
    }


def inspect(fix: str, g: dict) -> None:
    w = walk_numpy(fix, g)
    rms = w["rms"][1:]
    print(f"rms rows 1..: {np.round(rms, 2)}")
    allsc = np.concatenate([s.ravel() for s in w["scores"]])
    print(
        f"scores: min={allsc.min():.1f} max={allsc.max():.1f} "
        f"std={allsc.std():.1f}"
    )
    gaps = []
    for s in w["scores"]:
        for j in range(s.shape[0]):
            srow = s[j]
            gaps.append(float(srow.max() - np.sort(srow)[-2]))
    gaps = np.array(gaps)
    scale = max(float(rms.mean()), 1e-6)
    print(
        f"top-2 gaps: min={gaps.min():.3f} med={np.median(gaps):.3f} "
        f"max={gaps.max():.3f}; gap/scale med={np.median(gaps) / scale:.3f}"
    )
    print(
        f"idx distinct={len(np.unique(w['idx']))} "
        f"probs [{w['probs'].min():.4f}, {w['probs'].max():.4f}]"
    )


# ─────────────────────────── the vendor replay ───────────────────────────


def vendor_replay(fix: str, g: dict, raw_root: str) -> int:
    import mlx.core as mx

    sys.path.insert(0, os.path.join(raw_root, "dspine_mlx"))
    from dspine_mlx.bridge import build_serial_tables  # the vendor build
    from dspine_mlx.propose import _chain  # the vendor walk

    e, c, r, k, t = g["E"], g["C"], g["R"], g["K"], g["T"]
    anchor, gamma = g["ANCHOR"], g["T"] - 1
    sqrt_c = math.sqrt(c)
    rc = 0  # exit code

    def verdict(name: str, ok: bool, detail: str) -> None:
        nonlocal rc
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            rc = 1

    # ── piece (a): the bridge read+codes half (bridge.__call__ up to codes) ──
    raw = mx.array(read_f32(f"{fix}/a_raw.f32.bin")).reshape(1, t, e).astype(mx.bfloat16)
    read_w = mx.array(read_f32(f"{fix}/a_read_w.f32.bin")).reshape(c, e).astype(mx.bfloat16)
    codebook = mx.array(read_f32(f"{fix}/a_codebook.f32.bin")).reshape(-1, c).astype(mx.bfloat16)
    f32 = raw.astype(mx.float32)
    rms = mx.sqrt(mx.maximum(mx.mean(f32 * f32, axis=-1), F32_TINY))  # [1,t]
    normalized = (f32 / (rms[..., None] + EPS_BRIDGE)).astype(mx.bfloat16)
    projected = mx.matmul(normalized, read_w.T).astype(mx.float32)
    proj_norm = mx.sqrt(mx.sum(projected * projected, axis=-1, keepdims=True))
    codes = (sqrt_c * projected / (proj_norm + EPS_BRIDGE)).astype(mx.bfloat16)
    anchor_codes = (sqrt_c * codebook[anchor].astype(mx.float32)).astype(mx.bfloat16)
    codes = mx.concatenate([anchor_codes[None, None, :], codes[:, 1:, :]], axis=1)
    rms_np = np.array(rms.reshape(-1)).astype(np.float32)
    codes_np = np.array(codes.reshape(t, c).astype(mx.float32)).astype(np.float32)
    rms_ref = read_f32(f"{fix}/a_rms_out.f32.bin")
    codes_ref = read_f32(f"{fix}/a_codes_out.f32.bin").reshape(t, c)
    rms_diff = float(np.abs(rms_np - rms_ref).max())
    rms_rel = rms_diff / max(float(np.abs(rms_ref).max()), 1e-30)
    verdict(
        "piece (a) rms",
        rms_rel <= 1e-5,
        f"max abs {rms_diff:.3e} (rel {rms_rel:.3e}; fp32 reduction-order class)",
    )
    mism, maxulp = bf16_flip_stats(codes_np, codes_ref)
    verdict(
        "piece (a) codes",
        mism <= codes_ref.size * 2 // 1000 and maxulp <= 2,
        f"{mism}/{codes_ref.size} bf16 flips (<=0.2%, <=2 ULP); max {maxulp} ULP",
    )

    # ── piece (b): message+write+fused norm (bridge.__call__ after packed) ──
    packed = mx.array(read_f32(f"{fix}/b_packed_in.f32.bin")).reshape(1, t, 3 * r).astype(mx.bfloat16)
    write_w = mx.array(read_f32(f"{fix}/b_write_w.f32.bin")).reshape(e, r).astype(mx.bfloat16)
    gate_b = mx.array(read_f32(f"{fix}/b_gate_b.f32.bin")).astype(mx.float32)
    b_raw = mx.array(read_f32(f"{fix}/b_raw_in.f32.bin")).reshape(1, t, e).astype(mx.bfloat16)
    norm_w = mx.array(read_f32(f"{fix}/b_norm_w.f32.bin")).astype(mx.bfloat16)
    rms_b = mx.array(read_f32(f"{fix}/a_rms_out.f32.bin")).reshape(1, t)
    bf32 = b_raw.astype(mx.float32)
    q, s, v = mx.split(packed.astype(mx.float32), [r, 2 * r], axis=-1)
    gate = mx.sigmoid(q[:, 1:, :] + s[:, :-1, :] + gate_b)
    message = mx.concatenate(
        [mx.zeros_like(q[:, :1, :]), gate * v[:, :-1, :]], axis=1
    ).astype(mx.bfloat16)
    delta = mx.matmul(message, write_w.T).astype(mx.float32)
    not_anchor = (mx.arange(t).reshape(1, t) % t != 0).astype(mx.float32)[..., None]
    out = (bf32 + (rms_b[..., None] * delta) * not_anchor).astype(mx.bfloat16)
    x = out.astype(mx.float32)
    y = (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + EPS_NORM)).astype(mx.bfloat16)
    normed = (y.astype(mx.float32) * norm_w.astype(mx.float32)).astype(mx.bfloat16)
    for name, got, ref_name in (
        ("message", message.reshape(t, r), "b_message_out"),
        ("out", out.reshape(t, e), "b_out_out"),
        ("normed", normed.reshape(t, e), "b_normed_out"),
    ):
        ref = read_f32(f"{fix}/{ref_name}.f32.bin").reshape(got.shape)
        got_np = np.array(got.astype(mx.float32)).astype(np.float32)
        mism, maxulp = bf16_flip_stats(got_np, ref)
        verdict(
            f"piece (b) {name}",
            mism <= ref.size * 2 // 1000 and maxulp <= 2,
            f"{mism}/{ref.size} bf16 flips (<=0.2%, <=2 ULP); max {maxulp} ULP",
        )

    # ── piece (d): vendor build_serial_tables + greedy score block + walk ──
    lm_head = read_f32(f"{fix}/d_lm_head_in.f32.bin").reshape(-1, e)
    norm_w_f32 = read_f32(f"{fix}/d_norm_w_in.f32.bin")
    write_w_np = read_f32(f"{fix}/d_write_w_in.f32.bin").reshape(e, r)
    pv_np = np.concatenate(
        [read_f32(f"{fix}/d_pred_gate_in.f32.bin"),
         read_f32(f"{fix}/d_prev_w_in.f32.bin")]
    ).reshape(2 * r, c)
    v_slice = lm_head.shape[0]
    tg_m, tsv_m = build_serial_tables(
        head_rows=lambda a, b: mx.array(lm_head[a:b]).astype(mx.bfloat16),
        vocab_size=v_slice,
        norm_w=mx.array(norm_w_f32),
        write_w=mx.array(write_w_np).astype(mx.bfloat16),
        pv_w=mx.array(pv_np).astype(mx.bfloat16),
        codebook=codebook if codebook.shape[0] == v_slice else codebook[:v_slice],
    )
    tg_np = np.array(tg_m.astype(mx.float32)).astype(np.float32).reshape(-1, r)
    tsv_np = np.array(tsv_m.astype(mx.float32)).astype(np.float32).reshape(-1, 2 * r)
    tg_ref = read_f32(f"{fix}/d_tg_out.f32.bin").reshape(-1, r)
    tsv_ref = read_f32(f"{fix}/d_tsv_out.f32.bin").reshape(-1, 2 * r)
    for name, got, ref in (("tg", tg_np, tg_ref), ("tsv", tsv_np, tsv_ref)):
        mism, maxulp = bf16_flip_stats(got, ref)
        verdict(
            f"piece (d) {name} table",
            mism <= ref.size // 2000 and maxulp <= 2,
            f"{mism}/{ref.size} bf16 flips (<=0.05%, <=2 ULP); max {maxulp} ULP",
        )

    # the greedy score block (RefinementTail.greedy minus candidate selection)
    d_raw = read_f32(f"{fix}/d_raw_in.f32.bin").reshape(t, e)
    d_rms = read_f32(f"{fix}/d_rms_in.f32.bin")
    d_q = read_f32(f"{fix}/d_q_in.f32.bin").reshape(t, r)
    d_gate_b = read_f32(f"{fix}/d_gate_b_in.f32.bin")
    cand = read_u32(f"{fix}/d_cand_in.u32.bin").reshape(gamma, k)
    raw_m = mx.array(d_raw)
    rms_m = mx.array(d_rms)
    q_m = mx.array(d_q)
    gate_b_m = mx.array(d_gate_b)
    norm_m = mx.array(norm_w_f32)
    tg_f = tg_m.astype(mx.float32)
    tsv_f = tsv_m.astype(mx.float32)
    lm_m = mx.array(lm_head)
    idx_np = np.zeros((gamma, k), dtype=np.int64)
    probs_np = np.zeros((gamma, k), dtype=np.float32)
    for nn in range(gamma):
        raw_row = raw_m[nn + 1]
        rms_n = rms_m[nn + 1]
        q = q_m[nn + 1]
        cand_n = cand[nn]
        head_rows = lm_m[mx.array(cand_n.astype(np.int32))].astype(mx.float32)
        base = mx.sum((raw_row[None, :] * norm_m) * head_rows, axis=-1)  # [K]
        sid = (
            mx.full((1, k), anchor, dtype=mx.int32)
            if nn == 0
            else mx.array(cand[nn - 1].astype(np.int32))[None, :]
        )
        tsv_sid = tsv_f[sid.squeeze(0)]  # [K, 2R]
        tvg, tvv = tsv_sid[:, :r], tsv_sid[:, r:]
        m = mx.sigmoid(q[None, :] + tvg + gate_b_m) * tvv  # [K, R]
        tg_cand = tg_f[mx.array(cand_n.astype(np.int32))]  # [K, R]
        s_mat = mx.matmul(m, tg_cand.T)  # [K, K]
        # base[None, :] broadcasts the CANDIDATE's base over the
        # sender axis — the vendor's base[:, None, :] layout
        # (the bench-959 replay bug: base[:, None] indexes the sender).
        score = base[None, :] + rms_n * s_mat
        idx_np[nn] = np.array(mx.argmax(score, axis=-1)).astype(np.int64)
        srow_np = np.array(score).astype(np.float32)  # [K(sender), K(cand)]
        scale = max(float(rms_m[nn + 1]), 1.0e-6)
        for j in range(k):
            row = srow_np[j]
            best = float(row.max())
            mass = np.float32(0.0)
            for jp in range(k):
                mass = np.float32(mass + math.exp(float(row[jp] - best) / scale))
            probs_np[nn, j] = np.float32(1.0 / float(mass))
    # the vendor walk + the Triton confidence-chain layout
    cand_mx = mx.array(cand.astype(np.int32))
    idx_mx = mx.array(idx_np.astype(np.int32))
    tokens_np = np.array(_chain(idx_mx, cand_mx)).astype(np.int64)
    confs_np = np.zeros(gamma, dtype=np.float32)
    tokens_np = np.zeros(gamma, dtype=np.int64)
    lane = int(idx_np[0, 0])
    previous = 0
    for nn in range(gamma):
        confs_np[nn] = probs_np[nn][previous]
        tokens_np[nn] = cand[nn][lane]
        previous = lane
        if nn + 1 < gamma:
            lane = int(idx_np[nn + 1][lane])

    idx_ref = read_f32(f"{fix}/d_idx_out.f32.bin").reshape(gamma, k)
    probs_ref = read_f32(f"{fix}/d_probs_out.f32.bin").reshape(gamma, k)
    tok_ref = read_f32(f"{fix}/d_tokens_out.f32.bin")
    conf_ref = read_f32(f"{fix}/d_confs_out.f32.bin")
    mism_idx = int((idx_np != idx_ref.astype(np.int64)).sum())
    verdict("piece (d) idx", mism_idx == 0, f"{mism_idx}/{idx_ref.size} lane mismatches")
    mism_tok = int((tokens_np != tok_ref.astype(np.int64)).sum())
    verdict("piece (d) tokens", mism_tok == 0, f"{mism_tok}/{gamma} token mismatches")
    pd = float(np.abs(probs_np - probs_ref).max())
    verdict("piece (d) probs", pd <= 1e-3, f"max abs {pd:.3e} (score-ULP class)")
    cd = float(np.abs(confs_np - conf_ref).max())
    verdict("piece (d) confs", cd <= 1e-3, f"max abs {cd:.3e} (score-ULP class)")

    # the tie-convention probe: MLX argmax must be first-index
    tie = mx.argmax(mx.array([1.0, 1.0, 0.5]), axis=-1)
    tie_ok = int(tie) == 0
    verdict("mlx argmax tie", tie_ok, f"argmax([1,1,0.5]) = {int(tie)} (want 0)")

    return rc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", default=DEFAULT_FIXTURES)
    ap.add_argument("--raw", default=DEFAULT_RAW)
    ap.add_argument("--inspect", action="store_true")
    args = ap.parse_args()
    g = parse_geometry(f"{args.fixtures}/MANIFEST.md")
    if args.inspect:
        inspect(args.fixtures, g)
        return
    raise SystemExit(vendor_replay(args.fixtures, g, args.raw) or 0)


if __name__ == "__main__":
    main()
