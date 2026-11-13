#!/usr/bin/env python3
"""i1214 G3 — the eval token ids (one frozen read, shared by both lanes).

Writes `eval_ids.json` beside the calibrated artifact: the first
`--slice-tokens` tokens of the wikitext-2-raw-v1 TEST split (the standard
PPL split), tokenized with the Qwen3 tokenizer and concatenated — the
byte-identical id stream the Rust int reader and the Python bf16
baseline both consume, so ΔPPL is measured on ONE slice.

Stdout carries the blake3 digest of the id stream; the artifact digest
pins the slice. Run AFTER scripts/i1214_prepare.py:

    uv run --with torch,transformers,safetensors,datasets,blake3 python \
        scripts/i1214_eval_ids.py --artifact-dir artifacts/qwen3-0.6b-int8

(blake3 wheel is optional — the digest is also printed as a plain
byte-count + head/tail ids row for environments without it.)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--artifact-dir", default="artifacts/qwen3-0.6b-int8")
    ap.add_argument("--slice-tokens", type=int, default=32_768)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_id)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")

    # The detllm-style eval corpus: concatenate the raw test text, encode
    # once, slice. (The tokenizer adds no BOS by default for Qwen3 — the
    # ids stream is pure continuation text.)
    text = "".join(ds["text"])
    ids = tok(text, add_special_tokens=False)["input_ids"]
    slice_ids = ids[: args.slice_tokens]
    assert len(slice_ids) == args.slice_tokens, (
        f"test split too small: {len(slice_ids)} < {args.slice_tokens}"
    )

    raw = json.dumps(slice_ids, separators=(",", ":")).encode()
    md = hashlib.sha256(raw).hexdigest()

    out = {
        "model_id": args.model_id,
        "dataset": "Salesforce/wikitext",
        "config": "wikitext-2-raw-v1",
        "split": "test",
        "slice_tokens": args.slice_tokens,
        "chunk": args.chunk,
        "add_special_tokens": False,
        "sha256_ids_json": md,
        "ids": slice_ids,
    }
    path = os.path.join(args.artifact_dir, "eval_ids.json")
    os.makedirs(args.artifact_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))

    print(f"eval ids written: {path}")
    print(f"tokens={len(slice_ids)} chunk={args.chunk} sha256(ids_json)={md}")
    print(f"head_ids={slice_ids[:8]} tail_ids={slice_ids[-8:]}")


if __name__ == "__main__":
    main()
