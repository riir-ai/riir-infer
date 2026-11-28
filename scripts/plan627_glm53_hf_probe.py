#!/usr/bin/env python3
"""Plan 627 Phase 0 — GLM-5.3-Flash arch probe from HF metadata (never the weights).

The `plan402_gguf_probe` pattern for safetensors/config models: the glm5_next
forward lane's assumptions (layer split, MLA geometry, the trained indexer,
MoE shape, mHC, MTP) are cheap to check from config.json and expensive to
discover after a ~650 GB pull. This probe reads ONLY small metadata files
(config.json + the HF model-info record); it never downloads weights and
writes nothing outside an optional cache dir.

Usage:
    py scripts/plan627_glm53_hf_probe.py [--repo zai-org/GLM-5.3-Flash] [--cache-dir DIR]

Exit 0 on a complete probe; exit 2 on a fetch/parse refusal (loud, named).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

REPO_DEFAULT = "zai-org/GLM-5.3-Flash"
HF_RAW = "https://huggingface.co/{repo}/resolve/main/{name}"
HF_API = "https://huggingface.co/api/models/{repo}"

GB = 1024.0 * 1024.0 * 1024.0
MIB = 1024.0 * 1024.0


def _reconfigure_streams() -> None:
    # The console-encoding defence (plan402_gguf_probe precedent, katgpt-rs
    # Issue 804): a box whose console is cp874 dies on the first non-ASCII
    # print with no verdict at all — backslashreplace keeps every ASCII
    # character exact and degrades exotic glyphs to visible escapes.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass  # detached / not a TextIOWrapper: best-effort, never fatal


def _cached_read(path: Path) -> str | None:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def _cached_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def fetch(repo: str, name: str, cache_dir: Path | None) -> str:
    """Fetch a small metadata file, honoring an optional cache dir."""
    cached = _cached_read(cache_dir / name.replace("/", "_")) if cache_dir else None
    if cached is not None:
        return cached
    url = HF_RAW.format(repo=repo, name=name)
    with urllib.request.urlopen(url, timeout=60) as resp:
        text = resp.read().decode("utf-8")
    if cache_dir is not None:
        _cached_write(cache_dir / name.replace("/", "_"), text)
    return text


def fetch_api(repo: str, cache_dir: Path | None) -> dict:
    cached = _cached_read(cache_dir / "_api.json") if cache_dir else None
    if cached is not None:
        data = json.loads(cached)
    else:
        url = HF_API.format(repo=repo)
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if cache_dir is not None:
            _cached_write(cache_dir / "_api.json", json.dumps(data))
    return data


def main() -> int:
    _reconfigure_streams()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=REPO_DEFAULT)
    ap.add_argument("--cache-dir", default=None,
                    help="optional cache dir for the fetched metadata JSONs")
    args = ap.parse_args()
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    try:
        cfg = json.loads(fetch(args.repo, "config.json", cache_dir))
        api = fetch_api(args.repo, cache_dir)
    except Exception as exc:  # loud refusal, never a silent partial probe
        print(f"PROBE REFUSED: {exc}")
        return 2

    t = cfg.get("text_config", cfg)
    print(f"repo: {args.repo}")
    print(f"pin sha: {api.get('sha')}  lastModified: {api.get('lastModified')}")
    print(f"arch: {cfg.get('architectures')}  model_type: {t.get('model_type')}")
    print()

    # --- layer map -----------------------------------------------------
    layer_types = t.get("layer_types") or []
    counts: dict[str, int] = {}
    for lt in layer_types:
        counts[lt] = counts.get(lt, 0) + 1
    full_layers = [i for i, lt in enumerate(layer_types)
                   if "sparse" in str(lt) or "full" in str(lt)]
    lac = t.get("linear_attn_config") or {}
    kda_layers = lac.get("kda_layers") or []
    print("LAYER MAP")
    print(f"  num_hidden_layers: {t.get('num_hidden_layers')}  types: {counts}")
    print(f"  full-attn (DSA/MLA) layers ({len(full_layers)}): {full_layers}")
    print(f"  linear-attn (KDA) layers ({len(kda_layers)}): {kda_layers}")
    print(f"  MTP draft layers: {t.get('num_nextn_predict_layers')}"
          f"  indexer shared into MTP iteration: {t.get('index_share_for_mtp_iteration')}")
    print()

    # --- DSA / MLA geometry ---------------------------------------------
    print("DSA/MLA GEOMETRY (the full-attn layers)")
    for k in ("hidden_size", "num_attention_heads", "num_key_value_heads",
              "q_lora_rank", "kv_lora_rank", "qk_nope_head_dim",
              "qk_rope_head_dim", "qk_head_dim", "v_head_dim", "mla_use_nope"):
        print(f"  {k}: {t.get(k)}")
    idx_keys = {k: t.get(k) for k in sorted(t.keys()) if k.startswith("index")}
    print(f"  indexer: {json.dumps(idx_keys)}")
    if not t.get("qk_rope_head_dim"):
        rope_note = "NONE on attention (qk_rope_head_dim == 0)"
    else:
        rope_note = f"rope head dim {t.get('qk_rope_head_dim')}"
    print(f"  rotary: {rope_note}; indexer_rope_interleave: {t.get('indexer_rope_interleave')}")
    print()

    # --- KDA geometry ----------------------------------------------------
    print("KDA (linear attention) GEOMETRY")
    for k in ("num_heads", "head_dim", "short_conv_kernel_size", "gate_lower_bound"):
        print(f"  {k}: {lac.get(k)}")
    print()

    # --- MoE ---------------------------------------------------------------
    print("MoE")
    for k in ("first_k_dense_replace", "n_routed_experts", "n_shared_experts",
              "num_experts_per_tok", "n_group", "topk_group", "topk_method",
              "scoring_func", "routed_scaling_factor", "moe_intermediate_size",
              "intermediate_size", "moe_router_dtype", "hidden_act",
              "swiglu_limit", "vocab_size"):
        print(f"  {k}: {t.get(k)}")
    print()

    # --- mHC -----------------------------------------------------------------
    print("mHC (hyper-connections)")
    for k in ("mhc", "hc_mult", "hc_eps", "hc_sinkhorn_iters"):
        print(f"  {k}: {t.get(k)}")
    print()

    # --- context / tokenizer --------------------------------------------------
    print("CONTEXT / TOKENIZER")
    print(f"  max_position_embeddings: {t.get('max_position_embeddings')}")
    print(f"  tie_word_embeddings: {t.get('tie_word_embeddings', cfg.get('tie_word_embeddings'))}")
    print(f"  vision tower present: {'vision_config' in cfg}"
          f"  (text lane defers multimodal per plan 627 Phase 1)")
    print()

    # --- storage + KV budget ---------------------------------------------------
    st = api.get("safetensors") or {}
    params = st.get("parameters") or {}
    total_params = st.get("total") or sum(params.values())
    used = api.get("usedStorage") or 0
    print("STORAGE (HF-reported)")
    for k, v in params.items():
        print(f"  {k}: {v:,}")
    print(f"  total params: {total_params:,}")
    if total_params:
        b_per_param = used / total_params
        note = ("~2.0 = bf16-class storage despite the fp8 quantization_config;"
                " verify per-tensor dtypes against the index at pull time"
                if b_per_param > 1.5 else "fp8-class storage as configured")
        print(f"  usedStorage: {used / GB:.1f} GB  ({b_per_param:.2f} B/param — {note})")
    q = cfg.get("quantization_config") or {}
    print(f"  quantization_config.quant_method: {q.get('quant_method')}"
          f"  (modules_to_not_convert entries: {len(q.get('modules_to_not_convert') or [])})")
    print()

    kv_rank = t.get("kv_lora_rank") or 0
    rope_kv = t.get("qk_rope_head_dim") or 0
    per_tok_per_layer_bf16 = (kv_rank + rope_kv) * 2
    print("KV BUDGET @ long context (DSA layers only; bf16)")
    for ctx in (131072, 1048576):
        per_layer = per_tok_per_layer_bf16 * ctx / GB
        total = per_layer * len(full_layers)
        print(f"  @{ctx}: {per_layer:.2f} GB/layer"
              f"  x{len(full_layers)} DSA layers = {total:.2f} GB")
    kda_state = (lac.get("num_heads") or 0) * (lac.get("head_dim") or 0) ** 2 * 4
    if kda_layers:
        print(f"  KDA recurrent state: {kda_state / MIB:.1f} MiB/layer/seq"
              f"  x{len(kda_layers)} = {kda_state * len(kda_layers) / GB:.2f} GB (per sequence)")
    kpool = t.get("index_kpool") or 1
    idx_dim = t.get("index_head_dim") or 0
    if kpool and idx_dim:
        for ctx in (131072, 1048576):
            pooled = ctx // kpool * idx_dim * 2
            print(f"  indexer KPool cache @{ctx}: {pooled / MIB:.1f} MiB/layer"
                  f"  x{len(full_layers)} = {pooled * len(full_layers) / GB:.2f} GB (bf16 upper bound)")
    print()

    print("PROBE COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
