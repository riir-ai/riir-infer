#!/usr/bin/env python3
"""Plan 633 P1.1 — NeMo PyTorch fp32 reference cell (the liquidcodec-comparable
denominator row). Uses iapp's own ChindaASR service class (serve/chinda_asr.py,
OpenMDW-1.1) — their exact streaming recipe: CacheAwareStreamingAudioBuffer,
att 560ms = [56,6], max_symbols_per_step=2, strip_lang_tags, prompt th-TH,
blank bias baked in the bias-1.0 checkpoint (delta 0). torch 1 thread.

NOT a timing cell until run on a quiet rig (the smoke rows below carry no load
provenance — the bench record quotes only quiet-rig rows)."""
import argparse, json, os, sys, time
import numpy as np

ASR = os.path.dirname(os.path.abspath(__file__))
NEMO_RAW = os.path.join(ASR, "nemo", "openthai2.0-asr-streaming_raw.nemo")
CELLS = os.path.join(ASR, "p633_cells.jsonl")
SEED = 633

def log(*a):
    print("[p633-p11]", *a, flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-clips", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(args.threads)
    sys.path.insert(0, os.path.join(ASR, "iapp", "serve"))
    os.environ.setdefault("CHINDA_NUM_THREADS", str(args.threads))
    import chinda_asr as ca

    if not os.path.exists(NEMO_RAW):
        sys.exit(f"missing {NEMO_RAW} — their loader refuses bias-baked files; fetch the raw nemo")
    asr = ca.ChindaASR(model_path=NEMO_RAW, device="cpu", num_threads=args.threads)
    asr.load()
    log(f"model loaded in {asr._replicas[0].load_s:.1f}s; threads={args.threads}")

    sys.path.insert(0, ASR)
    from p633_p1 import load_manifest, prep, set_cer
    import soundfile as sf

    for family in ("fleurs-th", "tvspeech"):
        man = load_manifest(family)
        clips = man["clips"][:args.max_clips] if args.max_clips else man["clips"]
        tot_audio = 0.0
        refs, hyps = [], []
        wall0 = time.perf_counter()
        for c in clips:
            wave, sr = sf.read(os.path.join(ASR, c["path"]), dtype="float32")
            if wave.ndim > 1:
                wave = wave[:, 0]
            dur = c["audio_s"]
            t0 = time.perf_counter()
            texts = asr.decode_pass([wave], "th-TH", 560, -1.0, batch_size=1)  # raw nemo + their standard delta
            dt = time.perf_counter() - t0
            hyps.append(texts[0])
            refs.append(c["ref"])
            tot_audio += dur
        wall = time.perf_counter() - wall0
        cer = set_cer(refs, hyps)
        row = {"cell": "P1.1", "precision": "fp32-nemo", "threads": args.threads,
               "family": family, "n": len(clips), "audio_s": round(tot_audio, 2),
               "wall_s": round(wall, 3), "rtfx_wall": round(tot_audio / wall, 2),
               "cer_set": round(cer, 2),
               "note": "iapp ChindaASR decode_pass recipe; RAW nemo + standard delta −1.0 (= their "
                       "benchmark posture, net bias identical to the bias-1.0 bake); dither as shipped; batch_size=1"}
        with open(CELLS, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        log(json.dumps(row, ensure_ascii=False))

if __name__ == "__main__":
    main()
