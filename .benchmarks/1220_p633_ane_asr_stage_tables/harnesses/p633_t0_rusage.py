#!/usr/bin/env python3
"""Issue 1220 T0 — rusage CPU-seconds per audio-second + the CER comparability matrix.

Preregistered in .issues/1220_ane_warm_tier_adaptive_posture.md (T0):
  - Arms: onnx1t / onnx4t (P1.2 piecewise-dictation recipe) and ane (P2.5 streaming
    pipe) over the SAME 633 slice manifests, one arm per PROCESS (clean rusage).
  - Per clip: resource.getrusage(RUSAGE_SELF) user+sys delta, wall, audio_s; the
    hypothesis text is saved per arm so BOTH CER contracts score the SAME decode:
    prep  = p633_p1.set_cer (number-normalize + chinda-normalize + space-strip) —
            the contract behind the recorded 22.69 (ONNX) and 9.29 (P1.1) rows;
    raw   = space-strip only (p633_p23's inline scorer) — the contract behind the
            recorded 10.75 (ANE) row.
    The recorded 10.75-vs-22.69 gap spans THREE variables (scorer, decode recipe,
    export); this cell scores both arms under both contracts so the per-family
    quality column of the issue's ladder becomes like-for-like.
  - CPU-seconds is load-tolerant (the verdict's own words): the quiet-rig preflight
    still runs and is recorded verbatim, and --allow-load waives a load refusal
    EXPLICITLY (never silently) — waived rows carry load_waived=true + the
    provenance string.

Recipe identity: p633_p1.greedy_decode_long (ONNX arms) and the p633_p23 loop
(ANE arm) are mirrored VERBATIM; the imported modules are the lane's own files in
--asr-dir (sha256 logged), falling back to the preserved copies beside this script
only if the lane copies are gone.

Outputs (in --out-dir):
  p633_t0_cells.jsonl                     one summary row per (arm, family)
  p633_t0_clip_<arm>_<family>.jsonl       per-clip rusage/wall rows
  p633_t0_hyps_<arm>_<family>.json        refs + hyps (re-scorable forever)
"""
import argparse
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PREFLIGHT_DEFAULT = "/Users/katopz/git/riir-reflex/scripts/bench_preflight.sh"
NEEDED_LANE_MODULES = ("p633_p1.py", "p633_p24.py")


def log(*a):
    print("[p633-t0]", *a, flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rusage_now():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime, r.ru_stime


def run_preflight(args):
    """Record box state; a load refusal is honored unless --allow-load (T0's
    preregistered load tolerance — the waiver is LOUD, never silent)."""
    prov, refused = "PROVENANCE: preflight-missing", True
    if os.path.exists(args.preflight):
        r = subprocess.run([args.preflight], capture_output=True, text=True)
        out = r.stdout + r.stderr
        prov = next((l for l in out.splitlines() if l.startswith("PROVENANCE:")), prov)
        refused = ("✗" in out) or ("REFUSE" in out)
        if refused:
            log("preflight refusal recorded:", out.strip().splitlines()[-1:] or out.strip())
    else:
        log("preflight script missing:", args.preflight)
    if refused and not args.allow_load:
        log("preflight refused and --allow-load not passed — cell not run (exit 3)")
        raise SystemExit(3)
    return prov, refused and args.allow_load


def setup_asr(args):
    asr = os.path.abspath(args.asr_dir)
    if not os.path.isdir(asr):
        sys.exit(f"--asr-dir not a directory: {asr}")
    for mod in NEEDED_LANE_MODULES:
        dst = os.path.join(asr, mod)
        if not os.path.exists(dst):
            src = os.path.join(HERE, mod)
            if not os.path.exists(src):
                sys.exit(f"{mod} missing in BOTH the lane ({dst}) and the preserved set ({src})")
            shutil.copy2(src, dst)
            log(f"restored {mod} into the lane from the preserved set")
    for sub in ("iapp/serve", "iapp/benchmark", "fixtures", "onnx", "ane", "nemo"):
        if not os.path.isdir(os.path.join(asr, sub)):
            sys.exit(f"lane missing required subtree: {os.path.join(asr, sub)}")
    sys.path.insert(0, os.path.join(asr, "iapp", "serve"))
    sys.path.insert(0, asr)
    for mod in NEEDED_LANE_MODULES:
        log(f"{mod} sha256={sha256_file(os.path.join(asr, mod))[:16]}")
    return asr


def out_paths(out_dir, arm, family):
    base = os.path.join(out_dir, f"p633_t0_clip_{arm}_{family}.jsonl")
    hyps = os.path.join(out_dir, f"p633_t0_hyps_{arm}_{family}.json")
    return base, hyps


def score_both(refs, hyps, set_cer):
    """The two contracts on the SAME decode — the comparability matrix cells."""
    cer_prep = set_cer(refs, hyps)
    e = c = 0
    import editdistance
    for r, h in zip(refs, hyps):
        r, h = r.replace(" ", ""), (h or "").replace(" ", "")
        e += editdistance.eval(r, h)
        c += len(r)
    cer_raw = 100.0 * e / max(c, 1)
    return cer_prep, cer_raw


def write_summary(out_dir, row):
    with open(os.path.join(out_dir, "p633_t0_cells.jsonl"), "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log(json.dumps(row, ensure_ascii=False))


def arm_onnx(args, asr, arm, threads, prov, waived):
    from p633_p1 import (load_manifest, set_cer, tokens_to_text,
                         greedy_decode_long, load_onnx_sessions)
    import soundfile as sf
    enc_s, joint_s = load_onnx_sessions("fp", threads)
    out_dir = args.out_dir
    for family in args.families:
        man = load_manifest(family)
        clips = man["clips"][: args.max_clips or None]
        clip_path, hyps_path = out_paths(out_dir, arm, family)
        f_clip = open(clip_path, "w")
        refs, hyps = [], []
        cpu = wall = audio = 0.0
        ru0_u, ru0_s = rusage_now()
        t_all = time.perf_counter()
        for ci, c in enumerate(clips):
            wave, sr = sf.read(os.path.join(asr, c["path"]), dtype="float32", always_2d=False)
            if wave.ndim > 1:
                wave = wave[:, 0]
            u0, s0 = rusage_now()
            t0 = time.perf_counter()
            hyp_ids, _fwd = greedy_decode_long(enc_s, joint_s, wave, sr)
            dt_wall = time.perf_counter() - t0
            u1, s1 = rusage_now()
            dt_cpu = (u1 - u0) + (s1 - s0)
            hyp_text = tokens_to_text([i for i in hyp_ids if i is not None])
            refs.append(c["ref"])
            hyps.append(hyp_text)
            audio += c["audio_s"]
            cpu += dt_cpu
            wall += dt_wall
            f_clip.write(json.dumps({"i": ci, "path": c["path"],
                                     "audio_s": c["audio_s"],
                                     "wall_s": round(dt_wall, 4),
                                     "cpu_s": round(dt_cpu, 4),
                                     "user_s": round(u1 - u0, 4),
                                     "sys_s": round(s1 - s0, 4)},
                                    ensure_ascii=False) + "\n")
            if (ci + 1) % 25 == 0:
                log(f"{arm} {family} {ci+1}/{len(clips)} cpu={cpu:.1f}s wall={wall:.1f}s")
        wall_all = time.perf_counter() - t_all
        ru1_u, ru1_s = rusage_now()
        f_clip.close()
        cer_prep, cer_raw = score_both(refs, hyps, set_cer)
        with open(hyps_path, "w") as f:
            json.dump({"arm": arm, "family": family, "clips": [
                {"path": cp, "ref": r, "hyp": h} for cp, r, h in
                zip((c["path"] for c in clips), refs, hyps)]}, f, ensure_ascii=False)
        write_summary(out_dir, {
            "cell": "T0", "arm": arm, "family": family, "n": len(clips),
            "audio_s": round(audio, 2),
            "cpu_s": round(cpu, 3), "wall_s": round(wall_all, 3),
            "cpu_per_audio": round(cpu / max(audio, 1e-9), 4),
            "rtfx_wall": round(audio / max(wall_all, 1e-9), 2),
            "cpu_user_s": round(ru1_u - ru0_u, 3), "cpu_sys_s": round(ru1_s - ru0_s, 3),
            "cer_prep": round(cer_prep, 2), "cer_raw": round(cer_raw, 2),
            "decode_recipe": "piecewise-dictation (p633_p1.greedy_decode_long verbatim)",
            "threads": threads, "provenance": prov, "load_waived": waived,
            "rev": os.environ.get("P633_T0_REV", "unknown"),
            "note": "rusage = RUSAGE_SELF user+sys (all threads); audio_s from the slice manifest",
        })


def arm_ane(args, asr, prov, waived):
    import numpy as np
    import torch
    import coremltools as ct
    import soundfile as sf
    import chinda_asr as ca
    from p633_p1 import load_manifest, set_cer, tokens_to_text
    from p633_p24 import load_model, AneDecodeTail, ANE_DIR, NEMO_RAW

    torch.set_num_threads(args.threads)
    model = load_model(args.threads)
    asr_client = ca.ChindaASR(model_path=NEMO_RAW, device="cpu", num_threads=args.threads)
    asr_client.load()
    rep = asr_client._replicas[0]
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

    arm = "ane"
    out_dir = args.out_dir
    for family in args.families:
        man = load_manifest(family)
        clips = man["clips"][: args.max_clips or None]
        clip_path, hyps_path = out_paths(out_dir, arm, family)
        f_clip = open(clip_path, "w")
        refs, hyps = [], []
        cpu = audio = 0.0
        mel = enc = dec = 0.0
        ru0_u, ru0_s = rusage_now()
        t_all = time.perf_counter()
        for ci, c in enumerate(clips):
            u0, s0 = rusage_now()
            clip_t0 = time.perf_counter()
            wave, sr = sf.read(os.path.join(asr, c["path"]), dtype="float32")
            if wave.ndim > 1:
                wave = wave[:, 0]
            with torch.inference_mode():
                buf = rep.new_buffer((56, 6))
                t0 = time.perf_counter()
                buf.append_audio(wave, stream_id=-1)
                mel += time.perf_counter() - t0
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
                        ins = {
                            "mel": chunk.numpy().astype(np.float32),
                            "mel_len": lengths.numpy().astype(np.int32),
                            "cache_last_channel": c1.numpy().astype(np.float32),
                            "cache_last_time": c2.numpy().astype(np.float32),
                            "cache_last_channel_len": c3.numpy().astype(np.int32),
                            "prompt_onehot": onehot,
                        }
                        outs = enc_cml[tag].predict(ins)
                        enc += time.perf_counter() - t0
                        vals = [outs[nm] for nm in enc_out[tag]]
                        enc_np = np.array(vals[0], dtype=np.float32).transpose(0, 2, 1)
                        enc_len = int(np.array(vals[1]).reshape(-1)[0])
                        caches = (torch.from_numpy(np.array(vals[2], dtype=np.float32)),
                                  torch.from_numpy(np.array(vals[3], dtype=np.float32)),
                                  torch.from_numpy(np.array(vals[4]).astype(np.int64)))
                    else:
                        t0 = time.perf_counter()
                        enc_t, len_t, c1t, c2t, c3t = tmodel.encoder.cache_aware_stream_step(
                            processed_signal=chunk, processed_signal_length=lengths,
                            cache_last_channel=c1, cache_last_time=c2,
                            cache_last_channel_len=c3, keep_all_outputs=True,
                            drop_extra_pre_encoded=cfg.drop_extra_pre_encoded)
                        enc_np = tmodel._apply_prompt_to_encoded(
                            enc_t).transpose(1, 2).numpy().astype(np.float32)
                        enc_len = int(len_t[0])
                        caches = (c1t, c2t, c3t)
                        enc += time.perf_counter() - t0
                    t0 = time.perf_counter()
                    toks_step, a_state, a_last, _ = tail.decode(
                        enc_np, enc_len, state=a_state, last_token=a_last)
                    toks.extend(toks_step)
                    dec += time.perf_counter() - t0
            hyp = tokens_to_text([t for t in toks if t is not None])
            dt_wall = time.perf_counter() - clip_t0
            u1, s1 = rusage_now()
            dt_cpu = (u1 - u0) + (s1 - s0)
            refs.append(c["ref"])
            hyps.append(hyp)
            audio += c["audio_s"]
            cpu += dt_cpu
            f_clip.write(json.dumps({"i": ci, "path": c["path"],
                                     "audio_s": c["audio_s"],
                                     "wall_s": round(dt_wall, 4),
                                     "cpu_s": round(dt_cpu, 4),
                                     "user_s": round(u1 - u0, 4),
                                     "sys_s": round(s1 - s0, 4)},
                                    ensure_ascii=False) + "\n")
            if (ci + 1) % 25 == 0:
                log(f"{arm} {family} {ci+1}/{len(clips)} cpu={cpu:.1f}s "
                    f"enc={enc:.1f}s dec={dec:.1f}s mel={mel:.1f}s")
        wall_all = time.perf_counter() - t_all
        ru1_u, ru1_s = rusage_now()
        f_clip.close()
        cer_prep, cer_raw = score_both(refs, hyps, set_cer)
        with open(hyps_path, "w") as f:
            json.dump({"arm": arm, "family": family, "clips": [
                {"path": cp, "ref": r, "hyp": h} for cp, r, h in
                zip((c["path"] for c in clips), refs, hyps)]}, f, ensure_ascii=False)
        write_summary(out_dir, {
            "cell": "T0", "arm": arm, "family": family, "n": len(clips),
            "audio_s": round(audio, 2),
            "cpu_s": round(cpu, 3), "wall_s": round(wall_all, 3),
            "cpu_per_audio": round(cpu / max(audio, 1e-9), 4),
            "rtfx_wall": round(audio / max(wall_all, 1e-9), 2),
            "mel_wall_s": round(mel, 3), "enc_wall_s": round(enc, 3),
            "dec_wall_s": round(dec, 3),
            "cpu_user_s": round(ru1_u - ru0_u, 3), "cpu_sys_s": round(ru1_s - ru0_s, 3),
            "cer_prep": round(cer_prep, 2), "cer_raw": round(cer_raw, 2),
            "decode_recipe": "streaming-560ms (p633_p23 loop verbatim)",
            "threads": args.threads, "provenance": prov, "load_waived": waived,
            "rev": os.environ.get("P633_T0_REV", "unknown"),
            "note": "rusage = RUSAGE_SELF user+sys (all threads, host side only — "
                    "the ANE device's compute is not CPU time); audio_s from the manifest",
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=("onnx1t", "onnx4t", "ane"))
    ap.add_argument("--asr-dir", required=True)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--families", default="fleurs-th,tvspeech")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--max-clips", type=int, default=0)
    ap.add_argument("--preflight", default=PREFLIGHT_DEFAULT)
    ap.add_argument("--allow-load", action="store_true",
                    help="T0 is preregistered load-tolerant (rusage metric); waives a "
                         "load refusal EXPLICITLY and discloses it in every row")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    args.families = [f for f in args.families.split(",") if f]
    prov, waived = run_preflight(args)
    log("PROVENANCE:", prov, "| load_waived =", waived)
    asr = setup_asr(args)
    if args.arm == "ane":
        arm_ane(args, asr, prov, waived)
    else:
        arm_onnx(args, asr, args.arm, 1 if args.arm == "onnx1t" else 4, prov, waived)


if __name__ == "__main__":
    main()
