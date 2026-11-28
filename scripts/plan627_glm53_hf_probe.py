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
    py scripts/plan627_glm53_hf_probe.py --shard-headers [--cache-dir DIR]

Exit 0 on a complete probe; exit 2 on a fetch/parse refusal (loud, named).
"""

from __future__ import annotations

import argparse
import json
import struct
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
        return json.loads(cached)
    url = HF_API.format(repo=repo)
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if cache_dir is not None:
        _cached_write(cache_dir / "_api.json", json.dumps(data))
    return data


def fetch_range(repo: str, name: str, start: int, end: int,
                cache_dir: Path | None) -> bytes:
    """Ranged GET of an LFS shard slice; refuses loud if Range is not honored
    (a CDN that answers 200 would otherwise stream the whole ~10 GB shard)."""
    cache_path = cache_dir / (name.replace("/", "_") + f".{start}.{end}") if cache_dir else None
    if cache_path is not None and cache_path.exists():
        return cache_path.read_bytes()
    url = HF_RAW.format(repo=repo, name=name)
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        if resp.status != 206:
            resp.close()
            raise RuntimeError(
                f"{name}: server answered {resp.status} to a Range request — "
                "range not honored; refusing instead of streaming a full shard")
        data = resp.read(end - start + 1 + 1)
    if len(data) != end - start + 1:
        raise RuntimeError(f"{name}: range read short ({len(data)} != {end - start + 1})")
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(data)
    return data


def safetensors_header(repo: str, name: str,
                       cache_dir: Path | None) -> tuple[dict, int]:
    """Read ONLY the safetensors header (8-byte LE length + JSON header) of a
    shard — never the tensor data. Returns (header, header_len)."""
    head = fetch_range(repo, name, 0, 7, cache_dir)
    (n,) = struct.unpack("<Q", head)
    if n <= 0 or n > 32 * 1024 * 1024:
        raise RuntimeError(f"{name}: implausible header length {n}")
    body = fetch_range(repo, name, 8, 7 + n, cache_dir)
    return json.loads(body.decode("utf-8")), n


def census_header(header: dict, header_len: int) -> dict:
    """Per-dtype {count, bytes} census over one shard's header."""
    census: dict[str, dict] = {}
    for tname, meta in header.items():
        if tname == "__metadata__":
            continue
        dt = meta.get("dtype", "?")
        lo, hi = meta.get("data_offsets", [0, 0])
        row = census.setdefault(dt, {"count": 0, "bytes": 0})
        row["count"] += 1
        row["bytes"] += max(0, hi - lo)
    total = sum(r["bytes"] for r in census.values())
    return {"census": census, "tensor_bytes": total, "header_len": header_len}


def probe_shard_headers(repo: str, cache_dir: Path | None) -> int:
    """Resolve the storage-dtype question from the index + sampled shard
    headers: which dtype do the weights actually STORE as (the 2.04 B/param
    anomaly vs the fp8 quantization_config)."""
    raw = fetch(repo, "model.safetensors.index.json", cache_dir)
    index = json.loads(raw)
    weight_map = index.get("weight_map") or {}
    if not weight_map:
        print("PROBE REFUSED: index carries no weight_map")
        return 2
    shards = sorted(set(weight_map.values()))
    idx_meta = index.get("metadata") or {}
    print("SAFETENSORS INDEX")
    print(f"  tensors: {len(weight_map)}  shards: {len(shards)}")
    if "Total_size" in idx_meta:
        ts = idx_meta["Total_size"]
        print(f"  index metadata Total_size: {ts:,} B = {ts / GB:.1f} GB"
              f"  (the data-segment ground truth; per-dtype proof below)")
    sample = [shards[0], shards[len(shards) // 2], shards[-1]]
    sample = sorted(dict.fromkeys(sample))
    measured_total = 0
    for shard in sample:
        header, hlen = safetensors_header(repo, shard, cache_dir)
        row = census_header(header, hlen)
        measured_total += row["tensor_bytes"]
        n_tensors = sum(r["count"] for r in row["census"].values())
        print(f"  {shard}: {n_tensors} tensors, data {row['tensor_bytes'] / GB:.2f} GB"
              f"  (header {hlen / 1024:.0f} KiB)")
        for dt, r in sorted(row["census"].items(), key=lambda kv: -kv[1]["bytes"]):
            print(f"      {dt}: {r['count']} tensors, {r['bytes'] / GB:.2f} GB")
        expert = [t for t, m in header.items()
                  if t != "__metadata__" and ".mlp." in t]
        if expert:
            dtypes = sorted({header[t]["dtype"] for t in expert})
            print(f"      expert (.mlp.) tensors: {len(expert)}, dtypes {dtypes}")
    if len(sample) == len(shards):
        est = measured_total
        tag = "(complete)"
    else:
        est = measured_total / len(sample) * len(shards)
        tag = f"(extrapolated from {len(sample)}/{len(shards)} sampled shards)"
    print(f"  sampled data total: {measured_total / GB:.1f} GB;"
          f" full-repo estimate {est / GB:.1f} GB {tag}")
    print()
    print("SHARD-HEADER PROBE COMPLETE")
    return 0


def main() -> int:
    _reconfigure_streams()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=REPO_DEFAULT)
    ap.add_argument("--cache-dir", default=None,
                    help="optional cache dir for the fetched metadata JSONs")
    ap.add_argument("--shard-headers", action="store_true",
                    help="resolve the storage-dtype question: fetch the "
                         "safetensors index + ranged header reads of sampled "
                         "shards (metadata only, never tensor data)")
    args = ap.parse_args()
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    if args.shard_headers:
        try:
            return probe_shard_headers(args.repo, cache_dir)
        except Exception as exc:
            print(f"PROBE REFUSED: {exc}")
            return 2

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
