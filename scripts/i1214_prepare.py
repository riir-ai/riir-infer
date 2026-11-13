#!/usr/bin/env python3
"""i1214 T1c — offline preparation for the int_ref lane (the float stage).

Decision-content transcription of detllm `src/detllm/prepare.py` @
`9c3f1c9cdd42294087cc0507992c21dfde944bea` (NO LICENSE upstream —
decision-content-only posture; overview `.research/027`; consumption
contract `.issues/1214_integer_exact_reference_lane.md`). Emits the
artifact the Rust loader `crates/riir-infer-laya/src/laya/riir/int_ref/
artifact.rs` consumes: `model.safetensors` (int tensors only) +
`config.json` (scalar constants) — upstream prepare.py's own layout,
byte-compatible.

The ONLY float stage of the lane (issue 1214 scope guard): SmoothQuant-
style analytic factors, alpha=0.5 for the linear-input sites, alpha=0.3
for the QK smoothing (the canonical Makefile build), training-free.
Preparation is never required to be bit-reproducible (upstream's own
rule); the artifact is pinned by digest at load (blake3, Rust side).

Model: dense Qwen3-0.6B bf16 from HF (the G3 target — NOT Qwen3.5, see
the issue's G3 scope guard: GDN-hybrid is tranche 2, and pre-quantized
GGUF fails the never-pre-quantized rule).

Run (from the repo root; needs `torch`, `transformers`, `safetensors`,
`datasets` — use `uv`):
    uv run python scripts/i1214_prepare.py --out artifacts/qwen3-0.6b-int8

Then load + verify in Rust:
    IntRefModel::new(artifact::load_dir(dir)?.0)

Upstream citations below are `prepare.py:<line>` at the pinned sha.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import torch

# ---------------------------------------------------------------------------
# Dyadic fitting (float allowed: offline only) — prepare.py:46-97
# ---------------------------------------------------------------------------


def fit_dyadic_float(s: float, bits: int = 16) -> tuple[int, int]:
    """Fit m/2^k ≈ s with m in [2^(bits-1), 2^bits). Returns (m, k).
    (prepare.py:49-55)"""
    assert s > 0
    k = (bits - 1) - math.floor(math.log2(s))
    m = round(s * 2.0 ** k)
    if m >= 1 << bits:
        m = (m + 1) // 2
        k -= 1
    assert (1 << (bits - 1)) <= m < (1 << bits), (s, m, k)
    return m, k


def quant_weight_per_channel(W: torch.Tensor, bits: int = 16,
                             weight_bits: int = 8):
    """W float [out, in] -> (w8 [out, in] int8, m int64 [out], k int).

    Per-output-channel symmetric; mantissas share ONE k per tensor so the
    requant epilogue keeps a scalar col_k (prepare.py:57-79; the
    weight_bits > 8 diagnostic path is not ported — the lane is int8).
    """
    W = W.double()
    lim = (1 << (weight_bits - 1)) - 1
    s = W.abs().amax(dim=1).clamp(min=1e-12) / lim
    w8 = torch.round(W / s[:, None]).clamp(-lim, lim).to(torch.int8)
    k = (bits - 1) - math.floor(math.log2(float(s.max())))
    m = torch.round(s * 2.0 ** k).to(torch.int64)
    if int(m.max()) >= 1 << bits:
        k -= 1
        m = torch.round(s * 2.0 ** k).to(torch.int64)
    m = m.clamp(min=1)
    return w8, m, k


def quant_embedding_per_row(E: torch.Tensor):
    """E float [V, H] -> (e8 int8, m int64 [V], k int64 [V]) — per-row
    scale AND per-row shift (rows are per-token after lookup, so k may
    vary). (prepare.py:82-95)"""
    E = E.double()
    s = E.abs().amax(dim=1).clamp(min=1e-12) / 127.0
    e8 = torch.round(E / s[:, None]).clamp(-127, 127).to(torch.int8)
    k = 15 - torch.floor(torch.log2(s)).to(torch.int64)
    m = torch.round(s * torch.pow(2.0, k.double())).to(torch.int64)
    over = m >= (1 << 16)
    k = torch.where(over, k - 1, k)
    m = torch.where(over, torch.round(s * torch.pow(2.0, k.double())).to(torch.int64), m)
    return e8, m.clamp(min=1), k


def fit_dyadic_float_vec(s: torch.Tensor, bits: int = 16):
    """Vectorized fit_dyadic_float: s > 0 -> (m int64, k int64) per
    element, m in [2^(bits-1), 2^bits). (prepare.py:98-107)"""
    s = s.double()
    k = (bits - 1) - torch.floor(torch.log2(s)).to(torch.int64)
    m = torch.round(s * torch.pow(2.0, k.double())).to(torch.int64)
    over = m >= (1 << bits)
    k = torch.where(over, k - 1, k)
    m = torch.where(over, torch.round(s * torch.pow(2.0, k.double())).to(torch.int64), m)
    return m.clamp(min=1), k


def fit_gamma_dyadic(gamma: torch.Tensor, bits: int = 14):
    """Signed per-channel gamma -> (m int64, k int): m_c = round(gamma_c
    · 2^k), shared k sized so max|m| lands in [2^(bits-1), 2^bits).
    (prepare.py:109-121)"""
    g = gamma.double()
    amax = float(g.abs().max())
    assert amax > 0
    k = (bits - 1) - math.floor(math.log2(amax))
    m = torch.round(g * 2.0 ** k).to(torch.int64)
    if int(m.abs().max()) >= 1 << bits:
        k -= 1
        m = torch.round(g * 2.0 ** k).to(torch.int64)
    return m, k


# ---------------------------------------------------------------------------
# RoPE tables — the float builder (ops/rope.py:25; Rust consumes int16
# tables at ROPE_FRAC_BITS = 14, HF Qwen3 convention: inv_freq_j =
# theta^(-2j/hd) for j < hd/2, the angle vector tiled twice).
# ---------------------------------------------------------------------------

ROPE_FRAC_BITS = 14


def build_rope_tables(hd: int, max_pos: int, theta: float):
    j = torch.arange(0, hd // 2, dtype=torch.float64)
    inv_freq = theta ** (-2.0 * j / hd)
    pos = torch.arange(max_pos, dtype=torch.float64)
    ang = torch.cat([torch.outer(pos, inv_freq)] * 2, dim=-1)  # [T, hd]
    scale = float(1 << ROPE_FRAC_BITS)
    cos_t = torch.round(torch.cos(ang) * scale).to(torch.int16)
    sin_t = torch.round(torch.sin(ang) * scale).to(torch.int16)
    return cos_t, sin_t


# ---------------------------------------------------------------------------
# Calibration: static K/V ranges per head + the linear-input activation
# maxima — prepare.py:124-215 (a float pass over wikitext with forward
# hooks; post-QK-Norm and post-RoPE where applicable; gamma REMOVED from
# the hook captures via degamma, matching the runtime quantizer's bare
# x/rms view).
# ---------------------------------------------------------------------------


def calibrate_kv_ranges(model, tok, device, n_samples: int, seq_len: int,
                        rope_theta: float):
    from datasets import load_dataset

    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_kv = cfg.num_key_value_heads
    n_q = cfg.num_attention_heads
    group = n_q // n_kv
    hd = cfg.head_dim
    k_chan_max = torch.zeros(n_layers, n_kv, hd)
    q_chan_max = torch.zeros(n_layers, n_kv, hd)
    v_max = torch.zeros(n_layers, n_kv)

    # float RoPE angles (offline)
    j = torch.arange(0, hd // 2, dtype=torch.float64)
    inv_freq = rope_theta ** (-2.0 * j / hd)
    pos = torch.arange(seq_len, dtype=torch.float64)
    ang = torch.cat([torch.outer(pos, inv_freq)] * 2, dim=-1)  # [T, hd]
    cos_f, sin_f = torch.cos(ang).float(), torch.sin(ang).float()

    captured = {}

    def hook_factory(layer_idx, kind):
        def hook(module, inp, out):
            captured[(layer_idx, kind)] = out.detach()
        return hook

    hidden = cfg.hidden_size
    inter = cfg.intermediate_size
    attn_in_max = torch.zeros(n_layers, hidden)
    mlp_in_max = torch.zeros(n_layers, hidden)
    down_in_max = torch.zeros(n_layers, inter)
    o_in_max = torch.zeros(n_layers, n_q * hd)

    def pre_hook_factory(layer_idx, kind):
        def hook(module, inp):
            captured[(layer_idx, kind)] = inp[0].detach()
        return hook

    handles = []
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.k_norm.register_forward_hook(hook_factory(i, "k")))
        handles.append(layer.self_attn.q_norm.register_forward_hook(hook_factory(i, "q")))
        handles.append(layer.self_attn.v_proj.register_forward_hook(hook_factory(i, "v")))
        handles.append(layer.input_layernorm.register_forward_hook(hook_factory(i, "an")))
        handles.append(layer.post_attention_layernorm.register_forward_hook(hook_factory(i, "mn")))
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(pre_hook_factory(i, "dn")))
        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(pre_hook_factory(i, "on")))

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(ds["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]

    with torch.no_grad():
        for s in range(n_samples):
            chunk = ids[s * seq_len:(s + 1) * seq_len].unsqueeze(0).to(device)
            model(chunk)
            for i in range(len(model.model.layers)):
                def rope_it(x, n_heads):
                    # -> [B, T, n_heads, hd], rotated
                    if x.shape[1] == n_heads and x.shape[2] != n_heads:
                        x = x.transpose(1, 2)
                    T = x.shape[1]
                    c, si = cos_f[:T].to(x.dtype), sin_f[:T].to(x.dtype)
                    rot = torch.cat([-x[..., hd // 2:], x[..., : hd // 2]], dim=-1)
                    return x * c[None, :, None, :] + rot * si[None, :, None, :]

                kn = rope_it(captured[(i, "k")].float().cpu(), n_kv)
                k_chan_max[i] = torch.maximum(
                    k_chan_max[i], kn.abs().amax(dim=(0, 1)))          # [n_kv, hd]
                qn = rope_it(captured[(i, "q")].float().cpu(), n_q)
                qc = qn.abs().amax(dim=(0, 1))                          # [n_q, hd]
                qc = qc.view(n_kv, group, hd).amax(dim=1)               # GQA group max
                q_chan_max[i] = torch.maximum(q_chan_max[i], qc)
                v = captured[(i, "v")].float().cpu()
                v = v.reshape(v.shape[0], v.shape[1], n_kv, hd)
                v_max[i] = torch.maximum(v_max[i], v.abs().amax(dim=(0, 1, 3)))
                layer = model.model.layers[i]

                def degamma(x, g):
                    # the runtime quantizer sees x/rms WITHOUT gamma
                    # (gamma folds into the consumer weights)
                    s = torch.where(g >= 0, 1.0, -1.0)
                    return x / (s * g.abs().clamp(min=1e-4))

                g_a = layer.input_layernorm.weight.detach().float().cpu()
                g_m = layer.post_attention_layernorm.weight.detach().float().cpu()
                an = degamma(captured[(i, "an")].float().cpu(), g_a)
                mn = degamma(captured[(i, "mn")].float().cpu(), g_m)
                attn_in_max[i] = torch.maximum(attn_in_max[i], an.abs().amax(dim=(0, 1)))
                mlp_in_max[i] = torch.maximum(mlp_in_max[i], mn.abs().amax(dim=(0, 1)))
                dn = captured[(i, "dn")].float().cpu()
                down_in_max[i] = torch.maximum(down_in_max[i], dn.abs().amax(dim=(0, 1)))
                on = captured[(i, "on")].float().cpu()
                o_in_max[i] = torch.maximum(o_in_max[i], on.abs().amax(dim=(0, 1)))
    for h in handles:
        h.remove()
    return (k_chan_max, q_chan_max, v_max,
            attn_in_max, mlp_in_max, down_in_max, o_in_max)


# ---------------------------------------------------------------------------
# Main — prepare.py:217-380
# ---------------------------------------------------------------------------


def prepare(out_dir: str, calib_samples: int = 48, calib_len: int = 1024,
            kv_margin: float = 1.05, device: str = "cpu",
            smooth_o: bool = True, smooth_acts: bool = True,
            qk_alpha: float = 0.3):
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "Qwen/Qwen3-0.6B"
    os.makedirs(out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval().to(device)
    cfg = model.config

    # §4 architecture verification (fail loudly rather than mis-quantize)
    assert cfg.model_type == "qwen3", cfg.model_type
    assert cfg.tie_word_embeddings, "expected tied embeddings"
    n_layers = cfg.num_hidden_layers
    hidden = cfg.hidden_size
    n_q = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    hd = cfg.head_dim
    inter = cfg.intermediate_size
    vocab = cfg.vocab_size
    first = model.model.layers[0].self_attn
    assert hasattr(first, "q_norm") and hasattr(first, "k_norm"), "QK-Norm missing!"
    # §5 overflow assertions, computed from config (re-checked at Rust load)
    for K in (hidden, inter, n_q * hd, n_kv * hd):
        assert K * 127 * 127 < 2 ** 31, K
    assert cfg.max_position_embeddings * 128 * 127 < 2 ** 31  # probs·V accum

    rope_theta = float(getattr(cfg, "rope_theta",
                               cfg.rope_parameters["rope_theta"]))
    print(f"config ok: {n_layers}L hidden={hidden} heads={n_q}/{n_kv} hd={hd} "
          f"inter={inter} vocab={vocab} rope_theta={rope_theta}")

    print("calibrating static K/V ranges + QK channel stats...")
    (k_maxes, q_maxes, v_maxes,
     attn_in_max, mlp_in_max, down_in_max, o_in_max) = calibrate_kv_ranges(
        model, tok, device, calib_samples, calib_len, rope_theta)
    model.cpu()

    def act_smoothing(act_max: torch.Tensor, weights: list[torch.Tensor],
                      alpha: float = 0.5):
        """SmoothQuant-style analytic factors for one linear-input site.
        (prepare.py:246-266) Returns (inv_m int64 [C], inv_k int, s_used
        [C]): the runtime multiplies the activation by the dyadic inv
        (= 1/s); offline the exactly-matching s_used = 1/inv_dyadic folds
        into the consumer weight columns."""
        wmax = torch.stack([w.double().abs().amax(dim=0) for w in weights]).amax(dim=0)
        s = (act_max.double().clamp(min=1e-5).pow(alpha)
             / wmax.clamp(min=1e-5).pow(1 - alpha))
        s = s / s.median()          # normalize: only relative scale matters
        s = s.clamp(1 / 64.0, 64.0)
        inv = 1.0 / s
        inv_k = 13 - math.floor(math.log2(float(inv.max())))
        inv_m = torch.round(inv * 2.0 ** inv_k).to(torch.int64).clamp(min=1)
        s_used = 1.0 / (inv_m.double() * 2.0 ** -inv_k)
        return inv_m, inv_k, s_used

    def identity_smoothing(n: int):
        """No-op factors (the §9.2 '- activation smoothing' ablation)."""
        return (torch.ones(n, dtype=torch.int64), 0,
                torch.ones(n, dtype=torch.float64))

    tensors: dict[str, torch.Tensor] = {}
    scalars: dict = {
        "model_id": model_id,
        "n_layers": n_layers, "hidden": hidden, "n_q_heads": n_q,
        "n_kv_heads": n_kv, "head_dim": hd, "intermediate": inter,
        "vocab": vocab, "max_pos": int(cfg.max_position_embeddings),
        "rope_frac_bits": ROPE_FRAC_BITS,
        "eos_token_id": int(cfg.eos_token_id),
        # build parameters (documentation only; the runtime ignores them)
        "calib_samples": calib_samples, "calib_len": calib_len,
        "kv_margin": kv_margin, "qk_alpha": qk_alpha,
        "smooth_o": smooth_o, "smooth_acts": smooth_acts,
        "layers": [],
    }

    def add_linear(name: str, W: torch.Tensor,
                   fold_gamma: torch.Tensor | None = None):
        """W: [out, in] float. Stores transposed [in, out] for the GEMM.
        fold_gamma: per-INPUT-channel gamma folded into the columns."""
        Wf = W.double()
        if fold_gamma is not None:
            Wf = Wf * fold_gamma.double()[None, :]
        w8, m, k = quant_weight_per_channel(Wf)
        tensors[name + ".w8t"] = w8.t().contiguous()
        tensors[name + ".m"] = m
        return {"k": k, "min_m": int(m.min())}

    emb = model.model.embed_tokens.weight.detach()
    e8, em, ek = quant_embedding_per_row(emb)
    tensors["embed.e8"] = e8
    tensors["embed.m"] = em
    tensors["embed.k"] = ek

    # LM head = tied embedding with final-norm gamma folded into the
    # input columns
    final_gamma = model.model.norm.weight.detach()
    scalars["head"] = add_linear("head", emb, fold_gamma=final_gamma)

    cos_t, sin_t = build_rope_tables(hd, int(cfg.max_position_embeddings),
                                     rope_theta)
    tensors["rope.cos"] = cos_t
    tensors["rope.sin"] = sin_t

    sqrt_d = math.sqrt(hd)
    for i, layer in enumerate(model.model.layers):
        L: dict = {}
        attn, mlp = layer.self_attn, layer.mlp
        g_attn = layer.input_layernorm.weight.detach()
        g_mlp = layer.post_attention_layernorm.weight.detach()
        p = f"layers.{i}."
        # Activation smoothing (analytic SmoothQuant, alpha=0.5): the
        # runtime multiplies the quantizer input by dyadic 1/s; the
        # exactly-matching s folds into the (already gamma-folded)
        # consumer weight columns.
        Wq = attn.q_proj.weight.detach().double() * g_attn.double()[None, :]
        Wk = attn.k_proj.weight.detach().double() * g_attn.double()[None, :]
        Wv = attn.v_proj.weight.detach().double() * g_attn.double()[None, :]
        as_m, as_k, s_attn = (act_smoothing(attn_in_max[i], [Wq, Wk, Wv])
                              if smooth_acts else identity_smoothing(hidden))
        tensors[p + "as_m"] = as_m
        L["as_k"] = as_k
        Wg = mlp.gate_proj.weight.detach().double() * g_mlp.double()[None, :]
        Wu = mlp.up_proj.weight.detach().double() * g_mlp.double()[None, :]
        ms_m, ms_k, s_mlp = (act_smoothing(mlp_in_max[i], [Wg, Wu])
                             if smooth_acts else identity_smoothing(hidden))
        tensors[p + "ms_m"] = ms_m
        L["ms_k"] = ms_k
        Wd = mlp.down_proj.weight.detach().double()
        ds_m, ds_k, s_down = (act_smoothing(down_in_max[i], [Wd])
                              if smooth_acts else identity_smoothing(inter))
        tensors[p + "ds_m"] = ds_m
        L["ds_k"] = ds_k
        Wo = attn.o_proj.weight.detach().double()
        if smooth_o:
            os_m, os_k, s_o = act_smoothing(o_in_max[i], [Wo])
        else:
            os_m = torch.ones(n_q * hd, dtype=torch.int64)
            os_k, s_o = 0, torch.ones(n_q * hd, dtype=torch.float64)
        tensors[p + "os_m"] = os_m
        L["os_k"] = os_k

        L["q"] = add_linear(p + "q", Wq * s_attn[None, :])
        L["k"] = add_linear(p + "k", Wk * s_attn[None, :])
        L["v"] = add_linear(p + "v", Wv * s_attn[None, :])
        L["o"] = add_linear(p + "o", Wo * s_o[None, :])
        L["gate"] = add_linear(p + "gate", Wg * s_mlp[None, :])
        L["up"] = add_linear(p + "up", Wu * s_mlp[None, :])
        L["down"] = add_linear(p + "down", Wd * s_down[None, :])

        qg_m, qg_k = fit_gamma_dyadic(attn.q_norm.weight.detach())
        kg_m, kg_k = fit_gamma_dyadic(attn.k_norm.weight.detach())
        tensors[p + "q_norm.m"] = qg_m
        tensors[p + "k_norm.m"] = kg_m
        L["q_norm_k"] = qg_k
        L["k_norm_k"] = kg_k

        # ---- QK smoothing + static K/V int8 scales -----------------------
        # Per-channel factors c_d = maxK_d^a / maxQ_d^(1-a) (a = qk_alpha,
        # 0.3 in the canonical build; clamped to [2^-6, 2^6]) cancel
        # EXACTLY in the dot product: Q is multiplied by the dyadic c at
        # runtime (before its dynamic quant), K's static per-channel scale
        # is c·sK_h. Post-smoothing K ranges are balanced, so a per-head
        # base scale sK_h captures every channel well. (prepare.py:330-354)
        kc = k_maxes[i].double().clamp(min=1e-6)      # [n_kv, hd]
        qc = q_maxes[i].double().clamp(min=1e-6)      # [n_kv, hd]
        c = (kc.pow(qk_alpha) / qc.pow(1 - qk_alpha)).clamp(1 / 64.0, 64.0)
        # Q-side dyadic multiplier: shared shift per layer (the row scale
        # must stay uniform across channels); the value actually used is
        # m/2^ck.
        qs_k = 13 - math.floor(math.log2(float(c.max())))
        qs_m = torch.round(c * 2.0 ** qs_k).to(torch.int64).clamp(min=1)
        c_dyadic = qs_m.double() * 2.0 ** -qs_k       # exact values in use
        tensors[p + "qs_m"] = qs_m.reshape(-1)
        L["qs_k"] = qs_k

        smoothed_kmax = (kc / c_dyadic).amax(dim=1)   # [n_kv]
        s_kh = smoothed_kmax * kv_margin / 127.0      # per-head base scale
        # per-channel K cache scale = c_d · sK_h (16-bit dyadic each) —
        # the contract the Rust side documents on LayerWeights:
        # v(kq) = v(qs)·sqrt(hd)·v(ks) so the softmax consumes q·k/sqrt(hd)
        kq_m, kq_k = fit_dyadic_float_vec((c_dyadic * s_kh[:, None]).reshape(-1))
        tensors[p + "kq_m"] = kq_m
        tensors[p + "kq_k"] = kq_k
        # score scale per head: sK_h with 1/sqrt(hd) folded; V cache
        # scale per head: v_max · margin / 127
        ks_m, ks_k, vq_m, vq_k = [], [], [], []
        for h in range(n_kv):
            m2, k2 = fit_dyadic_float(float(s_kh[h]) / sqrt_d)
            s_v_raw = float(v_maxes[i, h]) * kv_margin / 127.0
            m3, k3 = fit_dyadic_float(s_v_raw)
            ks_m.append(m2); ks_k.append(k2)
            vq_m.append(m3); vq_k.append(k3)
        for nm, vals in [("ks_m", ks_m), ("ks_k", ks_k),
                         ("vq_m", vq_m), ("vq_k", vq_k)]:
            tensors[p + nm] = torch.tensor(vals, dtype=torch.int64)
        scalars["layers"].append(L)

    for name, t in tensors.items():
        assert t.dtype in (torch.int8, torch.int16, torch.int32, torch.int64), \
            f"float leak in artifact: {name} {t.dtype}"

    save_file(tensors, os.path.join(out_dir, "model.safetensors"))
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(scalars, f, indent=1)
    print(f"artifact written to {out_dir}")
    print("verify from Rust: IntRefModel::new(artifact::load_dir(dir)?.0)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", default="artifacts/qwen3-0.6b-int8")
    ap.add_argument("--calib-samples", type=int, default=48)
    ap.add_argument("--calib-len", type=int, default=1024)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--kv-margin", type=float, default=1.05)
    ap.add_argument("--no-smooth-o", action="store_true")
    ap.add_argument("--no-smooth-acts", action="store_true",
                    help="disable attn/mlp/down activation smoothing "
                         "(the §9.2 ablation; combine with --no-smooth-o)")
    ap.add_argument("--qk-alpha", type=float, default=0.3)
    args = ap.parse_args()
    prepare(args.out, args.calib_samples, args.calib_len,
            kv_margin=args.kv_margin, device=args.device,
            smooth_o=not args.no_smooth_o,
            smooth_acts=not args.no_smooth_acts,
            qk_alpha=args.qk_alpha)


if __name__ == "__main__":
    main()
