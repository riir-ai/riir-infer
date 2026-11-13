#!/usr/bin/env python3
"""Issue 1214 T1a cross-language mirror — regenerates the Python-mirror
golden vectors for crates/riir-infer-laya/tests/int_ref_python_mirror.rs.

Transcribes detllm @ 9c3f1c9c (NO LICENSE upstream — decision-content-only
posture) directly from the pinned source text, in pure stdlib Python ints
(no torch needed: every op is integer arithmetic), and emits the Rust test
file asserting the Rust port matches on seeded pseudo-random inputs. Two
independent transcriptions agreeing (Python mirror vs the Rust port, both
read from the same upstream text) is the transcription evidence.

Regenerate: python3 scripts/i1214_python_mirror.py > tests/int_ref_python_mirror.rs
"""

import math
import random

# ---- intmath.py (frozen rounding laws) ----

def rshift_round(a, k):
    if k == 0:
        return a
    return (a + (1 << (k - 1))) >> k


def round_half_away_div(a, b):
    assert b > 0
    s = -1 if a < 0 else 1
    return s * ((abs(a) + b // 2) // b)


def ilog2_floor(a):
    assert a >= 1
    return a.bit_length() - 1


def isqrt(a):
    assert a >= 0
    x, c, d = a, 0, 1 << 62
    while d > a:
        d >>= 2
    while d:
        if x >= c + d:
            x -= c + d
            c = (c >> 1) + d
        else:
            c >>= 1
        d >>= 2
    return c


def clamp_i8(v):
    return max(-128, min(127, v))


# ---- dyadic.py (tensor bookkeeping) ----

def scaled_round_div(num, den, e):
    if e >= 0:
        return round_half_away_div(num * (1 << e), den)
    return round_half_away_div(num, den * (1 << -e))


def fit_dyadic_ratio(num, den, bits=16):
    assert den >= 1 and num >= 1
    L = ilog2_floor(num) - ilog2_floor(den)
    e = (bits - 1) - L
    m = scaled_round_div(num, den, e)
    adj = 0
    if m >= (1 << bits):
        adj -= 1
    if m < (1 << (bits - 1)):
        adj += 1
    if adj != 0:
        e += adj
        m = scaled_round_div(num, den, e)
    if m >= (1 << bits):
        m = rshift_round(m, 1)
        e -= 1
    return m, e


def norm_scale(m, k, bits=16):
    sh = (bits - 1) - ilog2_floor(max(m, 1))
    up = max(sh, 0)
    down = max(-sh, 0)
    m2 = rshift_round(m << up, down)
    if m2 >= (1 << bits):
        m2 = rshift_round(m2, 1)
        sh -= 1
    return m2, k + sh


def quant_i8_pertoken_row(x, m, k):
    m, k = norm_scale(m, k)
    a = max(max(abs(v) for v in x), 1)
    x8 = [clamp_i8(round_half_away_div(v * 127, a)) for v in x]
    fm, fe = fit_dyadic_ratio(a * m, 127)
    return x8, fm, k + fe


def requant_i32_common_row(row, row_m, row_k, col_m, col_k, target_bits):
    scaled = [v * col_m[c] for c, v in enumerate(row)]
    a = max(max(abs(v) for v in scaled), 1)
    sh = max(ilog2_floor(a) - (target_bits - 1), 0)
    y = [rshift_round(v, sh) for v in scaled]
    return y, row_m, row_k + col_k - sh


def requant_i8_rowcol_row(row, row_m, row_k, col_m, col_k):
    scaled = [v * col_m[c] for c, v in enumerate(row)]
    a0 = max(max(abs(v) for v in scaled), 1)
    sh = max(ilog2_floor(a0) - 45, 0)
    scaled = [rshift_round(v, sh) for v in scaled]
    a = max(max(abs(v) for v in scaled), 1)
    y8 = [clamp_i8(round_half_away_div(v * 127, a)) for v in scaled]
    fm, fe = fit_dyadic_ratio(a * row_m, 127)
    return y8, fm, row_k + col_k + fe - sh


def requant_i8_static_el(p, col_mv, row_mv, shift, s_m):
    num = p * col_mv * row_mv
    up = max(-shift, 0)
    down = max(shift, 0)
    num <<= up
    den = s_m << down
    return clamp_i8(round_half_away_div(num, den))


# residual stream

def residual_renorm_row(row, k_res):
    g = max(max(abs(v) for v in row), 0)
    lg = ilog2_floor(max(g, 1))
    sh_down = max(lg - (30 - 1), 0)
    sh_up = max((24 + 2) - lg, 0) if g > 0 else 0
    if sh_down > 0:
        sh_up = 0
    out = [rshift_round(v, sh_down) << sh_up for v in row]
    return out, k_res - sh_down + sh_up


def residual_add_row(h, k_res, d, dm, dk):
    rel = dk - k_res
    down = max(rel, 0)
    up = max(-rel, 0)
    d_aligned = [rshift_round(v * dm, down) << up for v in d]
    return residual_renorm_row([a + b for a, b in zip(h, d_aligned)], k_res)


# ---- ops ----

def di_exp_row(x, m, k):
    m_f = m + (m >> 1) - (m >> 4)
    tpos = max(round_half_away_div(1 << k, m_f), 1)
    t = -tpos
    out = []
    for xi in x:
        assert xi <= 0
        q = xi // t  # python floor division
        r = xi - q * t
        unshifted = (r >> 1) - t
        out.append(unshifted >> min(q, 62))
    return out, tpos


def di_softmax_row(scores, m, k, valid, p_out_bits=15):
    NEG = -(2 ** 63) + 1
    row_max = NEG
    any_valid = False
    for i, v in enumerate(valid):
        if v:
            any_valid = True
            row_max = max(row_max, scores[i])
    if not any_valid:
        row_max = 0
    xd = [(scores[i] - row_max) if valid[i] else 0 for i in range(len(scores))]
    exps, _ = di_exp_row(xd, m, k)
    exps = [e if valid[i] else 0 for i, e in enumerate(exps)]
    denom = max(sum(exps), 1)
    return [round_half_away_div(e << (p_out_bits - 1), denom) for e in exps]


def di_rmsnorm_row(row):
    n = len(row)
    g = max(max(abs(v) for v in row), 0)
    lg = ilog2_floor(max(g, 1))
    sh = 25 - lg
    up = max(sh, 0)
    down = max(-sh, 0)
    xp = [rshift_round(v, down) << up for v in row]
    ssum = sum(v * v for v in xp)
    mean = round_half_away_div(ssum, n)
    rms = max(isqrt(mean), 1)
    return [round_half_away_div(v << 20, rms) for v in xp]


def di_rmsnorm_gamma_row(row, gamma_m, gamma_k):
    bare = di_rmsnorm_row(row)
    return [v * gamma_m[i] for i, v in enumerate(bare)], 20 + gamma_k


def di_swiglu_row(gate, up, gm, gk, um, uk):
    neg_abs = [-abs(g) for g in gate]
    e, tpos = di_exp_row(neg_abs, gm, gk)
    out = []
    for i, g in enumerate(gate):
        denom = max(tpos + e[i], 1)
        num = tpos if g >= 0 else e[i]
        sig = round_half_away_div(num << (8 - 1), denom)
        out.append(g * sig * up[i])
    return out, gm * um, gk + uk + (8 - 1)


def apply_int_rope_row(x, cos_row, sin_row):
    d = len(x)
    half = d // 2
    out = []
    for j in range(d):
        partner = -x[j + half] if j < half else x[j - half]
        v = x[j] * cos_row[j] + partner * sin_row[j]
        out.append(rshift_round(v, 14))
    return out


# ---- seeded inputs + Rust emission ----

rng = random.Random(1214)

# residual_renorm_row under its upstream name for the from_delta case
ns_renorm = residual_renorm_row


def rint(lo, hi):
    return rng.randint(lo, hi)


def emit_mat(name, rows, cols, values):
    flat = ", ".join(str(v) for v in values)
    print(f"    let {name}_data: Vec<i64> = vec![{flat}];")
    print(f"    let {name} = Mat::from(({name}_data.as_slice(), {rows}, {cols}));")


def emit_scale(name, values):
    flat = ", ".join(str(v) for v in values)
    print(f"    let {name}: Vec<i64> = vec![{flat}];")


HEADER = """\
//! Issue 1214 T1a — the Python-mirror cross-check (GENERATED FILE).
//!
//! Every expected vector below was produced by
//! `scripts/i1214_python_mirror.py` — a second, independent
//! transcription of detllm @ `9c3f1c9c` (NO LICENSE upstream;
//! decision-content-only posture) in pure stdlib Python — and pins the
//! Rust port to it on seeded pseudo-random inputs. Two independent
//! transcriptions agreeing is the transcription evidence; a divergence
//! here is a port bug (or a mirror bug — adjudicate against the
//! upstream source text, never against the failing side).
//!
//! Regenerate:
//! `python3 scripts/i1214_python_mirror.py > tests/int_ref_python_mirror.rs`
//! then `cargo test -p riir-infer-laya --features int_ref --test
//! int_ref_python_mirror`.
//!
//! GENERATED — edit the generator, never this file.

#![cfg(feature = "int_ref")]

use riir_infer_laya::laya::riir::int_ref::{
    apply_int_rope, di_exp, di_rmsnorm, di_rmsnorm_gamma, di_softmax, di_swiglu,
    quant_i8_pertoken, residual_add, residual_from_delta, requant_i32_common,
    requant_i8_rowcol, requant_i8_static, Mat, RopeTables,
};

fn check(name: &str, got: &[i64], want: &[i64]) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(g, w, "{name}[{i}]: got {g}, mirror says {w}");
    }
}
"""

print(HEADER)


def case_di_exp():
    print("#[test]")
    print("fn mirror_di_exp() {")
    rows, cols = 3, 16
    for r in range(rows):
        x = [-rint(0, 1 << 12) for _ in range(cols)]
        m = rint(1, 1 << 12)
        k = rint(4, 20)
        exps, tpos = di_exp_row(x, m, k)
        emit_mat("x", 1, cols, x)
        emit_scale("m", [m])
        print(f"    let k: Vec<u32> = vec![{k}];")
        print(f"    let (res, tp) = di_exp(x, &m, &k);")
        print(f"    let want_r{r}: Vec<i64> = vec![{', '.join(str(v) for v in exps)}];")
        print(f"    check(\"di_exp r{r}\", &res, &want_r{r});")
        print(f"    assert_eq!(tp[0], {tpos});")
    print("}\n")


def case_di_softmax():
    print("#[test]")
    print("fn mirror_di_softmax() {")
    rows, cols = 3, 16
    for r in range(rows):
        scores = [rint(-(1 << 12), 1 << 12) for _ in range(cols)]
        valid = [rng.random() < 0.75 for _ in range(cols)]
        if r == rows - 1:
            valid = [False] * cols  # the all-masked row
        m = rint(1, 1 << 10)
        k = rint(4, 16)
        probs = di_softmax_row(scores, m, k, valid)
        emit_mat("scores", 1, cols, scores)
        emit_scale("m", [m])
        print(f"    let k: Vec<u32> = vec![{k}];")
        vstr = ", ".join("true" if v else "false" for v in valid)
        print(f"    let valid = vec![{vstr}];")
        print("    let p = di_softmax(scores, &m, &k, &valid, 15);")
        print(f"    let want_p{r}: Vec<i32> = vec![{', '.join(str(v) for v in probs)}];")
        print(f"    assert_eq!(&p, &want_p{r});")
    print("}\n")


def case_di_rmsnorm():
    print("#[test]")
    print("fn mirror_di_rmsnorm() {")
    rows, cols = 3, 128
    x = [rint(-(1 << 20), 1 << 20) for _ in range(rows * cols)]
    emit_mat("x", rows, cols, x)
    y = []
    for r in range(rows):
        y.extend(di_rmsnorm_row(x[r * cols:(r + 1) * cols]))
    print(f"    let want_y: Vec<i32> = vec![{', '.join(str(v) for v in y)}];")
    print("    assert_eq!(di_rmsnorm(x), want_y);")
    # gamma variant
    rows_g, cols_g = 2, 32
    xg = [rint(-(1 << 20), 1 << 20) for _ in range(rows_g * cols_g)]
    gamma_m = [rint(-(1 << 10), 1 << 10) for _ in range(cols_g)]
    gamma_k = 10
    emit_mat("xg", rows_g, cols_g, xg)
    emit_scale("gamma_m", gamma_m)
    print(f"    let gamma_k: u32 = {gamma_k};")
    yg = []
    kouts = []
    for r in range(rows_g):
        row, ko = di_rmsnorm_gamma_row(xg[r * cols_g:(r + 1) * cols_g], gamma_m, gamma_k)
        yg.extend(row)
        kouts.append(ko)
    print(f"    let want_yg: Vec<i64> = vec![{', '.join(str(v) for v in yg)}];")
    print("    let (got_yg, got_k) = di_rmsnorm_gamma(xg, &gamma_m, gamma_k);")
    print("    check(\"di_rmsnorm_gamma\", &got_yg, &want_yg);")
    print(f"    assert_eq!(got_k, {kouts[0]});")
    print("}\n")


def case_di_swiglu():
    print("#[test]")
    print("fn mirror_di_swiglu() {")
    rows, cols = 2, 16
    gate = [rint(-(1 << 12), 1 << 12) for _ in range(rows * cols)]
    up = [rint(-(1 << 12), 1 << 12) for _ in range(rows * cols)]
    gm = [rint(1, 1 << 8) for _ in range(rows)]
    gk = [rint(4, 12) for _ in range(rows)]
    um = [rint(1, 1 << 8) for _ in range(rows)]
    uk = [rint(4, 12) for _ in range(rows)]
    emit_mat("gate", rows, cols, gate)
    emit_mat("up", rows, cols, up)
    emit_scale("gm", gm)
    emit_scale("um", um)
    print(f"    let gk: Vec<u32> = vec![{', '.join(str(v) for v in gk)}];")
    print(f"    let uk: Vec<u32> = vec![{', '.join(str(v) for v in uk)}];")
    y, m_out, k_out = [], [], []
    for r in range(rows):
        yr, mo, ko = di_swiglu_row(
            gate[r * cols:(r + 1) * cols], up[r * cols:(r + 1) * cols],
            gm[r], gk[r], um[r], uk[r],
        )
        y.extend(yr)
        m_out.append(mo)
        k_out.append(ko)
    print(f"    let want_y: Vec<i64> = vec![{', '.join(str(v) for v in y)}];")
    print(f"    let want_m: Vec<i64> = vec![{', '.join(str(v) for v in m_out)}];")
    print(f"    let want_k: Vec<u32> = vec![{', '.join(str(v) for v in k_out)}];")
    print("    let (gy, gm_out, gk_out) = di_swiglu(gate, up, &gm, &gk, &um, &uk);")
    print("    check(\"di_swiglu y\", &gy, &want_y);")
    print("    assert_eq!(gm_out, want_m);")
    print("    assert_eq!(gk_out, want_k);")
    print("}\n")


def case_quant():
    print("#[test]")
    print("fn mirror_quant_i8_pertoken() {")
    rows, cols = 3, 16
    x = [rint(-(1 << 20), 1 << 20) for _ in range(rows * cols)]
    m = [rint(1, 1 << 12) for _ in range(rows)]
    k = [rint(4, 20) for _ in range(rows)]
    emit_mat("x", rows, cols, x)
    emit_scale("m", m)
    emit_scale("k", k)
    x8, m_out, k_out = [], [], []
    for r in range(rows):
        q8, fm, ko = quant_i8_pertoken_row(x[r * cols:(r + 1) * cols], m[r], k[r])
        x8.extend(q8)
        m_out.append(fm)
        k_out.append(ko)
    print(f"    let want_x8: Vec<i8> = vec![{', '.join(str(v) for v in x8)}];")
    print(f"    let want_m: Vec<i64> = vec![{', '.join(str(v) for v in m_out)}];")
    print(f"    let want_k: Vec<i64> = vec![{', '.join(str(v) for v in k_out)}];")
    print("    let (qx8, qm, qk) = quant_i8_pertoken(x, &m, &k);")
    print("    assert_eq!(qx8, want_x8);")
    print("    assert_eq!(qm, want_m);")
    print("    assert_eq!(qk, want_k);")
    print("}\n")


def case_requant():
    print("#[test]")
    print("fn mirror_requant_family() {")
    rows, cols = 2, 16
    p = [rint(-(1 << 22), 1 << 22) for _ in range(rows * cols)]
    row_m = [rint(1, 1 << 8) for _ in range(rows)]
    row_k = [rint(4, 20) for _ in range(rows)]
    col_m = [rint(1, 1 << 12) for _ in range(cols)]
    col_k = 6
    emit_mat("p", rows, cols, p)
    emit_scale("row_m", row_m)
    emit_scale("row_k", row_k)
    emit_scale("col_m", col_m)
    print(f"    let col_k: i64 = {col_k};")
    y32, k32 = [], []
    for r in range(rows):
        yr, _, ko = requant_i32_common_row(p[r * cols:(r + 1) * cols], row_m[r], row_k[r], col_m, col_k, 12)
        y32.extend(yr)
        k32.append(ko)
    print(f"    let want_y32: Vec<i32> = vec![{', '.join(str(v) for v in y32)}];")
    print(f"    let want_k32: Vec<i64> = vec![{', '.join(str(v) for v in k32)}];")
    print("    let (gy32, gm32, gk32) = requant_i32_common(p, &row_m, &row_k, &col_m, col_k, 12);")
    print("    assert_eq!(gy32, want_y32);")
    print("    assert_eq!(gm32, row_m);")
    print("    assert_eq!(gk32, want_k32);")
    # rowcol
    y8, m8, k8 = [], [], []
    for r in range(rows):
        q8, fm, ko = requant_i8_rowcol_row(p[r * cols:(r + 1) * cols], row_m[r], row_k[r], col_m, col_k)
        y8.extend(q8)
        m8.append(fm)
        k8.append(ko)
    print(f"    let want_y8: Vec<i8> = vec![{', '.join(str(v) for v in y8)}];")
    print(f"    let want_m8: Vec<i64> = vec![{', '.join(str(v) for v in m8)}];")
    print(f"    let want_k8: Vec<i64> = vec![{', '.join(str(v) for v in k8)}];")
    print("    let (gy8, gm8, gk8) = requant_i8_rowcol(p, &row_m, &row_k, &col_m, col_k);")
    print("    assert_eq!(gy8, want_y8);")
    print("    assert_eq!(gm8, want_m8);")
    print("    assert_eq!(gk8, want_k8);")
    # static: per-element s_m/s_k (the caller pre-broadcasts shapes)
    sm = [rint(1, 1 << 10) for _ in range(rows * cols)]
    sk = [rint(2, 12) for _ in range(rows * cols)]
    print(f"    let sm: Vec<i64> = vec![{', '.join(str(v) for v in sm)}];")
    print(f"    let sk: Vec<i64> = vec![{', '.join(str(v) for v in sk)}];")
    ys = []
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            ys.append(requant_i8_static_el(
                p[idx], col_m[c], row_m[r], row_k[r] + col_k - sk[idx], sm[idx],
            ))
    print(f"    let want_ys: Vec<i8> = vec![{', '.join(str(v) for v in ys)}];")
    print("    let gy8s = requant_i8_static(p, &row_m, &row_k, &col_m, col_k, &sm, &sk);")
    print("    assert_eq!(gy8s, want_ys);")
    print("}\n")


def case_residual():
    print("#[test]")
    print("fn mirror_residual_ops() {")
    rows, cols = 2, 16
    h = [rint(-(1 << 28), 1 << 28) for _ in range(rows * cols)]
    k_res = [rint(2, 12) for _ in range(rows)]
    d = [rint(-(1 << 18), 1 << 18) for _ in range(rows * cols)]
    dm = [rint(1, 1 << 8) for _ in range(rows)]
    dk = [rint(2, 24) for _ in range(rows)]
    emit_mat("h", rows, cols, h)
    emit_scale("k_res", k_res)
    emit_mat("d", rows, cols, d)
    emit_scale("dm", dm)
    emit_scale("dk", dk)
    # residual_add
    h2, k2 = [], []
    for r in range(rows):
        hr, kr = residual_add_row(
            h[r * cols:(r + 1) * cols], k_res[r],
            d[r * cols:(r + 1) * cols], dm[r], dk[r],
        )
        h2.extend(hr)
        k2.append(kr)
    print(f"    let want_h: Vec<i32> = vec![{', '.join(str(v) for v in h2)}];")
    print(f"    let want_k: Vec<i64> = vec![{', '.join(str(v) for v in k2)}];")
    print("    let (gh, gk) = residual_add(h, &k_res, d, &dm, &dk);")
    print("    assert_eq!(gh, want_h);")
    print("    assert_eq!(gk, want_k);")
    # residual_from_delta: upstream renorms d·dm DIRECTLY at scale dk —
    # no alignment through residual_add (dyadic.py:187-191)
    h3, k3 = [], []
    for r in range(rows):
        hr, kr = ns_renorm([v * dm[r] for v in d[r * cols:(r + 1) * cols]], dk[r])
        h3.extend(hr)
        k3.append(kr)
    print(f"    let want_h3: Vec<i32> = vec![{', '.join(str(v) for v in h3)}];")
    print(f"    let want_k3: Vec<i64> = vec![{', '.join(str(v) for v in k3)}];")
    print("    let (gh3, gk3) = residual_from_delta(d, &dm, &dk);")
    print("    assert_eq!(gh3, want_h3);")
    print("    assert_eq!(gk3, want_k3);")
    print("}\n")


def case_rope():
    print("#[test]")
    print("fn mirror_rope() {")
    max_pos, d = 4, 8
    half = d // 2
    # e as the base stands in for any model base (2.718281828459045);
    # the mirror only needs ONE self-consistent table
    inv_freq = [2.718281828459045 ** (-2.0 * j / d) for j in range(half)]
    cos_t, sin_t = [], []
    for pos in range(max_pos):
        ang = [pos * f for f in inv_freq]
        ang = ang + ang  # tile, not interleave (HF convention)
        cos_t.extend(int(round(math.cos(a) * (1 << 14))) for a in ang)
        sin_t.extend(int(round(math.sin(a) * (1 << 14))) for a in ang)
    rows = 3
    x = [rint(-(1 << 16), 1 << 16) for _ in range(rows * d)]
    positions = [rint(0, max_pos - 1) for _ in range(rows)]
    print(f"    let cos_data: Vec<i16> = vec![{', '.join(str(v) for v in cos_t)}];")
    print(f"    let sin_data: Vec<i16> = vec![{', '.join(str(v) for v in sin_t)}];")
    print(f"    let tables = RopeTables {{ cos: cos_data, sin: sin_data, max_pos: {max_pos}, head_dim: {d} }};")
    emit_mat("x", rows, d, x)
    print(f"    let positions: Vec<usize> = vec![{', '.join(str(v) for v in positions)}];")
    out = []
    for r in range(rows):
        pos = positions[r]
        out.extend(apply_int_rope_row(
            x[r * d:(r + 1) * d],
            cos_t[pos * d:(pos + 1) * d],
            sin_t[pos * d:(pos + 1) * d],
        ))
    print(f"    let want_out: Vec<i64> = vec![{', '.join(str(v) for v in out)}];")
    print("    let got = apply_int_rope(x, &tables, &positions);")
    print("    check(\"rope\", &got, &want_out);")
    print("}\n")


case_di_exp()
case_di_softmax()
case_di_rmsnorm()
case_di_swiglu()
case_quant()
case_requant()
case_residual()
case_rope()
print("// mirror end")
