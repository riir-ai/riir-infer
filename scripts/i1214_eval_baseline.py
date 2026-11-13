#!/usr/bin/env python3
"""i1214 G3 — the bf16/fp32 baseline PPL on the SAME frozen eval ids.

Reads `eval_ids.json` (scripts/i1214_eval_ids.py output), chunks it
identically to the Rust reader (chunk 1024, non-overlapping, position 0
of each chunk SKIPPED — predicting from nothing), and scores NLL with
the unquantized checkpoint in bf16 (the bar's reference) AND fp32 (the
calibration-precision reference). Prints both PPLs.

    uv run python scripts/i1214_eval_baseline.py \
        --artifact-dir artifacts/qwen3-0.6b-int8 [--device cpu]
"""

from __future__ import annotations

import argparse
import json
import math
import os

import torch


def ppl_for(model: torch.nn.Module, ids: list[int], chunk: int, device: str) -> float:
    model.eval().to(device)
    total = 0.0
    n_pred = 0
    with torch.no_grad():
        for ci in range(len(ids) // chunk):
            row = ids[ci * chunk : (ci + 1) * chunk]
            x = torch.tensor([row], dtype=torch.long, device=device)
            logits = model(x).logits[0]  # [chunk, vocab]
            logp = torch.log_softmax(logits.double(), dim=-1)
            # HF convention: logits[t] predicts ids[t+1] — score tokens
            # 1..chunk-1 with rows 0..chunk-2 (the last row predicts
            # past the chunk edge and is dropped; both lanes apply the
            # same rule)
            gold = torch.tensor(row[1:], dtype=torch.long, device=device)
            nll = -logp[torch.arange(0, chunk - 1, device=device), gold]
            total += float(nll.sum())
            n_pred += chunk - 1
    return math.exp(total / n_pred), total, n_pred


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--artifact-dir", default="artifacts/qwen3-0.6b-int8")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM

    meta = json.load(open(os.path.join(args.artifact_dir, "eval_ids.json")))
    ids: list[int] = meta["ids"]
    chunk: int = meta["chunk"]
    assert len(ids) % chunk == 0
    print(f"ids: n={len(ids)} chunk={chunk} sha256={meta['sha256_ids_json'][:16]}")

    for dtype, name in ((torch.bfloat16, "bf16"), (torch.float32, "fp32")):
        model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype)
        ppl, total, n_pred = ppl_for(model, ids, chunk, args.device)
        print(f"{name}: ppl={ppl:.6f} nll_sum={total:.6f} n_pred={n_pred}")
        del model


if __name__ == "__main__":
    main()
