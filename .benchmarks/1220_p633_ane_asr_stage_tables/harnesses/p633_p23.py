#!/usr/bin/env python3
"""Plan 633 P2.3 + P2.5 — the ANE fp16 pipeline cells over the FULL 100-clip
slices (both families), quiet-rig gated (fail-closed preflight inside).

Per clip: mel frontend (CPU, NeMo preprocessor — the CPU-tail law) → ANE encoder
stream loop (first/step graphs + the CPU flush tail) → ANE dec+joint host loop.
Three walls measured per family: mel / encode(ANE incl. the flush tail) /
decode-tail(ANE). Rows:
  - cell P2.3-enc:  encoder-only RTFx (audio / encode wall)
  - cell P2.5-pipe: full-pipeline RTFx + CER set (card convention) — the
    G1/G2/G3 gate inputs (reference = the P1.1 re-verified rows 9.29/12.34).
The chunk machinery is chinda_asr's (the P1.1 recipe buffer), the decode tail is
p633_p24.AneDecodeTail (the parity-green host loop), the encoder graphs are the
P2.1 parity-green pair. Nothing here is a new conversion — measurement only.
"""
import argparse, json, os, sys, time
import numpy as np

ASR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ASR)
sys.path.insert(0, os.path.join(ASR, "iapp", "serve"))
PREFLIGHT = "/Users/katopz/git/riir-reflex/scripts/bench_preflight.sh"
CELLS = os.path.join(ASR, "p633_p23_cells.jsonl")
# the P1.1 reference (re-verified 2026-11-15 under the current env, n=100 quiet)
P11_REF_CER = {"fleurs-th": 9.29, "tvspeech": 12.34}
# the current-era CPU denominators (P1.2 fp, n=100 quiet 2026-11-15)
P12_CPU_RTFX = {"fleurs-th": 49.23, "tvspeech": 45.30}  # 4 threads, the best CPU cell


def log(*a):
    print("[p633-p23]", *a, flush=True)


def append_cell(row):
    with open(CELLS, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def preflight_or_die():
    import subprocess
    r = subprocess.run([PREFLIGHT], capture_output=True, text=True)
    out = r.stdout + r.stderr
    prov = next((l for l in out.splitlines() if l.startswith("PROVENANCE:")), "")
    if "✗" in out or "REFUSE" in out:
        log("preflight REFUSED — cell not run:\n", out.strip())
        raise SystemExit(3)
    return prov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-clips", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--family", default=None, help="run one family alone (default both)")
    ap.add_argument("--per-clip-log", action="store_true",
                    help="one progress line per clip (the slow-leg diagnostic)")
    args = ap.parse_args()
    prov = preflight_or_die()
    log("PROVENANCE:", prov)

    import torch
    import coremltools as ct
    import soundfile as sf
    import chinda_asr as ca
    from p633_p1 import load_manifest, tokens_to_text
    from p633_p24 import load_model, AneDecodeTail, ANE_DIR, NEMO_RAW

    torch.set_num_threads(args.threads)
    model = load_model(args.threads)

    # the reference-side text decoder (the P1.1 recipe's own tokenizer plumbing)
    asr = ca.ChindaASR(model_path=NEMO_RAW, device="cpu", num_threads=args.threads)
    asr.load()
    rep = asr._replicas[0]
    tmodel = rep.model
    cfg = tmodel.encoder.streaming_cfg

    enc_cml = {
        "first": ct.models.MLModel(os.path.join(ANE_DIR, "openthai_enc_first.mlpackage"),
                                   compute_units=ct.ComputeUnit.CPU_AND_NE),
        "step": ct.models.MLModel(os.path.join(ANE_DIR, "openthai_enc_step.mlpackage"),
                                  compute_units=ct.ComputeUnit.CPU_AND_NE),
    }
    enc_out = {}
    for tag in ("first", "step"):
        m = ct.models.MLModel(os.path.join(ANE_DIR, f"openthai_enc_{tag}.mlpackage"))
        enc_out[tag] = [o.name for o in m._spec.description.output]
    joint_paths = {t: os.path.join(ANE_DIR, f"openthai_{t}.mlpackage")
                   for t in ("dec_step", "joint_b1", "joint_b28", "joint_b56", "joint_b112")}
    tail = AneDecodeTail(model, joint_paths)
    onehot = np.zeros((1, 1, int(tmodel.num_prompts)), dtype=np.float32)
    onehot[0, 0, int(tmodel._inference_prompt_index)] = 1.0

    decode_fn = getattr(tmodel.decoding, "rnnt_decoder_predictions_tensor", None)

    for family in ([args.family] if args.family else ("fleurs-th", "tvspeech")):
        man = load_manifest(family)
        clips = man["clips"][: args.max_clips or None]
        n = len(clips)
        tot_audio = tot_mel = tot_enc = tot_dec = 0.0
        refs_n, hyps_n = [], []
        t_all = time.perf_counter()
        for ci, c in enumerate(clips):
            clip_t0 = time.perf_counter()
            clip_enc = clip_dec = 0.0
            wave, sr = sf.read(os.path.join(ASR, c["path"]), dtype="float32")
            if wave.ndim > 1:
                wave = wave[:, 0]
            with torch.inference_mode():
                buf = rep.new_buffer((56, 6))
                t0 = time.perf_counter()
                buf.append_audio(wave, stream_id=-1)
                tot_mel += time.perf_counter() - t0
                caches = None
                a_state, a_last = None, None
                toks = []
                for step_num, (chunk, lengths) in enumerate(iter(buf)):
                    if caches is None:
                        caches = tmodel.encoder.get_initial_cache_state(batch_size=1)
                    c1, c2, c3 = caches
                    ka = buf.is_buffer_empty()
                    if not ka:
                        tag = "first" if step_num == 0 else "step"
                        t0 = time.perf_counter()
                        # mel for this chunk happens inside the buffer iteration
                        # (chinda_asr machinery) — attributed to the mel wall below
                        ins = {
                            "mel": chunk.numpy().astype(np.float32),
                            "mel_len": lengths.numpy().astype(np.int32),
                            "cache_last_channel": c1.numpy().astype(np.float32),
                            "cache_last_time": c2.numpy().astype(np.float32),
                            "cache_last_channel_len": c3.numpy().astype(np.int32),
                            "prompt_onehot": onehot,
                        }
                        outs = enc_cml[tag].predict(ins)
                        tot_enc += time.perf_counter() - t0
                        clip_enc += time.perf_counter() - t0
                        vals = [outs[nm] for nm in enc_out[tag]]
                        enc_np = np.array(vals[0], dtype=np.float32).transpose(0, 2, 1)
                        enc_len = int(np.array(vals[1]).reshape(-1)[0])
                        caches = (torch.from_numpy(np.array(vals[2], dtype=np.float32)),
                                  torch.from_numpy(np.array(vals[3], dtype=np.float32)),
                                  torch.from_numpy(np.array(vals[4]).astype(np.int64)))
                    else:
                        # the CPU flush tail (the ANE steady graph never sees
                        # keep_all_outputs=True — the CPU-tail law); its mel +
                        # torch encoder wall lands in the mel/enc attributions
                        # via the buffers above — measured here as enc time
                        t0 = time.perf_counter()
                        enc_t, len_t, c1t, c2t, c3t = tmodel.encoder.cache_aware_stream_step(
                            processed_signal=chunk, processed_signal_length=lengths,
                            cache_last_channel=c1, cache_last_time=c2,
                            cache_last_channel_len=c3, keep_all_outputs=True,
                            drop_extra_pre_encoded=cfg.drop_extra_pre_encoded)
                        enc_np = tmodel._apply_prompt_to_encoded(enc_t).transpose(1, 2).numpy().astype(np.float32)
                        enc_len = int(len_t[0])
                        caches = (c1t, c2t, c3t)
                        tot_enc += time.perf_counter() - t0
                        clip_enc += time.perf_counter() - t0
                    t0 = time.perf_counter()
                    toks_step, a_state, a_last, _ = tail.decode(
                        enc_np, enc_len, state=a_state, last_token=a_last)
                    toks.extend(toks_step)
                    tot_dec += time.perf_counter() - t0
                    clip_dec += time.perf_counter() - t0
            hyp = tokens_to_text([t for t in toks if t is not None])
            clip_wall = time.perf_counter() - clip_t0
            if args.per_clip_log:
                log(f"{family} clip {ci+1}/{n} {os.path.basename(c['path'])} "
                    f"audio={c['audio_s']:.1f}s wall={clip_wall:.2f}s enc={clip_enc:.2f} dec={clip_dec:.2f}")
            refs_n.append(c["ref"])
            hyps_n.append(hyp)
            tot_audio += c["audio_s"]
            if (ci + 1) % 25 == 0:
                log(f"{family} {ci+1}/{n} audio={tot_audio:.0f}s enc={tot_enc:.1f}s "
                    f"dec={tot_dec:.1f}s mel={tot_mel:.1f}s")
        wall_all = time.perf_counter() - t_all

        # CER (the card convention, via chinda's own scorer if present)
        try:
            import editdistance  # the P1 lane's dependency
            def _cer(refs, hyps):
                edits = chars = 0
                for r, h in zip(refs, hyps):
                    r = r.replace(" ", "")
                    h = h.replace(" ", "")
                    edits += editdistance.eval(r, h)
                    chars += len(r)
                return 100.0 * edits / max(chars, 1)
            cer = _cer(refs_n, hyps_n)
        except Exception as e:
            log("cer scorer failed:", e)
            cer = None

        enc_rtfx = tot_audio / max(tot_enc, 1e-9)
        pipe_rtfx = tot_audio / max(wall_all, 1e-9)
        best_cpu = P12_CPU_RTFX[family]
        row_enc = {"cell": "P2.3-enc", "family": family, "n": n,
                   "audio_s": round(tot_audio, 2),
                   "mel_wall_s": round(tot_mel, 3), "enc_wall_s": round(tot_enc, 3),
                   "enc_rtfx": round(enc_rtfx, 2),
                   "provenance": prov, "quiet": True, "threads": args.threads}
        row_pipe = {"cell": "P2.5-pipe", "family": family, "n": n,
                    "audio_s": round(tot_audio, 2),
                    "mel_wall_s": round(tot_mel, 3), "enc_wall_s": round(tot_enc, 3),
                    "dec_wall_s": round(tot_dec, 3), "wall_s": round(wall_all, 3),
                    "enc_rtfx": round(enc_rtfx, 2),
                    "pipe_rtfx": round(pipe_rtfx, 2),
                    "cer_set": round(cer, 2) if cer is not None else None,
                    "p11_ref_cer": P11_REF_CER[family],
                    "cer_delta_vs_ref": round(cer - P11_REF_CER[family], 2) if cer is not None else None,
                    "best_cpu_rtfx": best_cpu,
                    "g2_ratio_vs_cpu": round(pipe_rtfx / best_cpu, 3),
                    "provenance": prov, "quiet": True, "threads": args.threads}
        append_cell(row_enc)
        append_cell(row_pipe)
        log(json.dumps(row_enc, ensure_ascii=False))
        log(json.dumps(row_pipe, ensure_ascii=False))


if __name__ == "__main__":
    main()
