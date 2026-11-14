#!/usr/bin/env python3
"""Plan 633 P1 stage 1 — background fetches (nemo + benchmark lane + datasets).
Runs detached; log at .raw/asr/p633_fetch.log"""
import sys, traceback
from huggingface_hub import hf_hub_download, snapshot_download

def step(name, fn):
    try:
        out = fn()
        print(f"[p633-fetch] {name} OK -> {out}", flush=True)
    except Exception as e:
        print(f"[p633-fetch] {name} FAIL: {e}", flush=True)
        traceback.print_exc()

R = "iapp/OpenThai2.0-ASR-Streaming"
step("nemo_bias", lambda: hf_hub_download(R, "openthai2.0-asr-streaming_bias-1.0.nemo",
     local_dir="/Users/katopz/git/riir-infer/.raw/asr/nemo"))
step("benchmark_lane", lambda: snapshot_download(R, repo_type="model",
     allow_patterns=["benchmark/*", "serve/*", "LICENSE", "README.md"],
     local_dir="/Users/katopz/git/riir-infer/.raw/asr/iapp"))
step("fleurs_th_test", lambda: snapshot_download("google/fleurs", repo_type="dataset",
     allow_patterns=["data/th_th/audio/test.tar.gz", "data/th_th/test.tsv"],
     local_dir="/Users/katopz/git/riir-infer/.raw/asr/fleurs"))
step("tvspeech", lambda: snapshot_download("typhoon-ai/TVSpeech", repo_type="dataset",
     local_dir="/Users/katopz/git/riir-infer/.raw/asr/tvspeech"))
print("[p633-fetch] ALL DONE", flush=True)
