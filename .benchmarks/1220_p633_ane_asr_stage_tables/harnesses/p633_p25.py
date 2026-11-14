#!/usr/bin/env python3
"""Plan 633 P2.5 arm 2 — the ANE 8-bit palettized encoder conversion (the
FluidAudio/beshkenadze tier shape).

Post-conversion weight palettization of the P2.1 parity-green fp16 packages:
coremltools.optimize.coreml.palettize_weights with OpPalettizerConfig
(mode="kmeans", nbits=8) over BOTH encoder graphs. The dec/joint graphs keep
their fp16 forms (P2.4's parity-green tail; palettizing the LSTM/vocab is the
P2.4a-class follow-up if the tier sweep motivates it).

Modes: convert (fp16 → *_p8.mlpackage + compile) | census | p50 | parity |
all. parity = the P2.1 parity cell re-run against the p8 encoder graphs
(token identity vs the torch stream, 3 clips, load-tolerant). p50 = the
encoder-only per-call A/B vs the fp16 arm (fail-closed preflight inside).
Rows: p633_p25_parity.json / p633_p25_census.json / p633_p25_cells.jsonl.
"""
import argparse, json, os, subprocess, sys, time
import numpy as np

ASR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ASR)
sys.path.insert(0, os.path.join(ASR, "iapp", "serve"))
ANE_DIR = os.path.join(ASR, "ane")
PREFLIGHT = "/Users/katopz/git/riir-reflex/scripts/bench_preflight.sh"
PARITY = os.path.join(ASR, "p633_p25_parity.json")
CENSUS = os.path.join(ASR, "p633_p25_census.json")
CELLS = os.path.join(ASR, "p633_p25_cells.jsonl")


def log(*a):
    print("[p633-p25]", *a, flush=True)


def convert():
    """palettize_weights over the fp16 packages — the CPU-heavy kmeans pass."""
    import coremltools as ct
    from coremltools.optimize.coreml import (
        OpPalettizerConfig, OptimizationConfig, palettize_weights)
    for tag in ("first", "step"):
        src = os.path.join(ANE_DIR, f"openthai_enc_{tag}.mlpackage")
        dst = os.path.join(ANE_DIR, f"openthai_enc_{tag}_p8.mlpackage")
        if os.path.exists(dst):
            log("exists, skipping:", dst)
            continue
        t0 = time.time()
        m = ct.models.MLModel(src)
        cfg = OpPalettizerConfig(mode="kmeans", nbits=8, granularity="per_tensor",
                                 weight_threshold=2048)
        p8 = palettize_weights(m, config=OptimizationConfig(global_config=cfg))
        p8.save(dst)
        sz_src = sum(os.path.getsize(os.path.join(r, f))
                     for r, _, fs in os.walk(src) for f in fs)
        sz_dst = sum(os.path.getsize(os.path.join(r, f))
                     for r, _, fs in os.walk(dst) for f in fs)
        log(f"[{tag}] palettized in {time.time()-t0:.0f}s "
            f"{sz_src/1e6:.0f}MB → {sz_dst/1e6:.0f}MB ({sz_dst/max(sz_src,1)*100:.1f}%)")
        subprocess.run(["xcrun", "coremlcompiler", "compile", dst, ANE_DIR],
                       check=True, capture_output=True)


def census():
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan
    report = {}
    for tag in ("first", "step"):
        modelc = os.path.join(ANE_DIR, f"openthai_enc_{tag}_p8.mlmodelc")
        plan = MLComputePlan.load_from_path(modelc, compute_units=ct.ComputeUnit.CPU_AND_NE)
        prog = plan.model_structure.program
        ops = []
        for fn in prog.functions.values():
            stack = [fn.block]
            while stack:
                b = stack.pop()
                ops.extend(b.operations)
                for sub in getattr(b, "blocks", []) or []:
                    stack.append(sub)
        pref, costs = {}, []
        for op in ops:
            u = plan.get_compute_device_usage_for_mlprogram_operation(op)
            p = getattr(u, "preferred_compute_device", None)
            pn = type(p).__name__.replace("MLComputeDevice", "").replace("ML", "") if p else "unplaced"
            pref[pn] = pref.get(pn, 0) + 1
            try:
                c = plan.get_estimated_cost_for_mlprogram_operation(op)
                w = getattr(c, "weight", None)
                if w is not None:
                    costs.append((pn, float(w)))
            except Exception:
                pass
        placed = len(ops) - pref.get("unplaced", 0)
        ane = pref.get("NeuralEngineComputeDevice", 0) + pref.get("N", 0)
        ct_total = sum(w for _, w in costs)
        ct_ane = sum(w for d, w in costs if d == "NeuralEngineComputeDevice" or d == "N")
        report[tag] = {"n_ops": len(ops), "placed": placed, "pref_counts": pref,
                       "placed_ane_pct": round(100.0 * ane / placed, 2) if placed else None,
                       "cost_weighted_ane_pct": round(100.0 * ct_ane / ct_total, 2) if ct_total else None,
                       "n_cost_rows": len(costs)}
        log(f"[census {tag}] ops={len(ops)} pref={pref} "
            f"placed_ane={report[tag]['placed_ane_pct']} "
            f"cost_ane={report[tag]['cost_weighted_ane_pct']}")
    with open(CENSUS, "w") as f:
        json.dump(report, f, indent=2, default=str)


def parity(max_clips=3):
    """The P2.1 parity cell against the p8 encoder graphs — token identity vs
    the torch stream (the p8 fp16-drift gate at the palettized tier)."""
    import coremltools as ct
    import soundfile as sf
    import chinda_asr as ca
    from p633_p1 import load_manifest
    from p633_p24 import load_model, ANE_DIR, NEMO_RAW
    import torch

    asr = ca.ChindaASR(model_path=NEMO_RAW, device="cpu", num_threads=1)
    asr.load()
    rep = asr._replicas[0]
    tmodel = rep.model
    cfg = tmodel.encoder.streaming_cfg

    coreml = {
        "first": ct.models.MLModel(os.path.join(ANE_DIR, "openthai_enc_first_p8.mlpackage"),
                                   compute_units=ct.ComputeUnit.CPU_AND_NE),
        "step": ct.models.MLModel(os.path.join(ANE_DIR, "openthai_enc_step_p8.mlpackage"),
                                  compute_units=ct.ComputeUnit.CPU_AND_NE),
    }
    out_names = {}
    for tag in ("first", "step"):
        m = ct.models.MLModel(os.path.join(ANE_DIR, f"openthai_enc_{tag}_p8.mlpackage"))
        out_names[tag] = [o.name for o in m._spec.description.output]

    onehot = torch.zeros(1, 1, int(tmodel.num_prompts))
    onehot[0, 0, int(tmodel._inference_prompt_index)] = 1.0

    clips = []
    per_family = {"fleurs-th": 2, "tvspeech": 1}
    for family in ("fleurs-th", "tvspeech"):
        man = load_manifest(family)
        clips.extend(man["clips"][: per_family[family]])

    report = {"clips": []}
    for c in clips:
        wave, sr = sf.read(os.path.join(ASR, c["path"]), dtype="float32")
        if wave.ndim > 1:
            wave = wave[:, 0]
        with torch.inference_mode():
            buf = rep.new_buffer((56, 6))
            buf.append_audio(wave, stream_id=-1)
            caches = None
            hyps_ref = None
            tok_match = tok_total = 0
            emax_all, rms_all = 0.0, 0.0
            for step_num, (chunk, lengths) in enumerate(iter(buf)):
                if caches is None:
                    caches = tmodel.encoder.get_initial_cache_state(batch_size=1)
                c1, c2, c3 = caches
                ka = buf.is_buffer_empty()
                tag = "first" if step_num == 0 else "step"
                if not ka:
                    ins = {
                        "mel": chunk.numpy().astype(np.float32),
                        "mel_len": lengths.numpy().astype(np.int32),
                        "cache_last_channel": c1.numpy().astype(np.float32),
                        "cache_last_time": c2.numpy().astype(np.float32),
                        "cache_last_channel_len": c3.numpy().astype(np.int32),
                        "prompt_onehot": onehot.numpy().astype(np.float32),
                    }
                    outs = coreml[tag].predict(ins)
                    vals = [outs[n] for n in out_names[tag]]
                    enc_a = torch.from_numpy(np.array(vals[0], dtype=np.float32))
                    len_a = torch.from_numpy(np.array(vals[1]).astype(np.int64))
                    n1 = torch.from_numpy(np.array(vals[2], dtype=np.float32))
                    n2 = torch.from_numpy(np.array(vals[3], dtype=np.float32))
                    n3 = torch.from_numpy(np.array(vals[4]).astype(np.int64))
                    enc_t, len_t, c1t, c2t, c3t = tmodel.encoder.cache_aware_stream_step(
                        processed_signal=chunk, processed_signal_length=lengths,
                        cache_last_channel=c1, cache_last_time=c2,
                        cache_last_channel_len=c3, keep_all_outputs=False,
                        drop_extra_pre_encoded=0 if step_num == 0 else cfg.drop_extra_pre_encoded)
                    enc_t_post = tmodel._apply_prompt_to_encoded(enc_t)
                    emax_all = max(emax_all, float((enc_a.float() - enc_t_post).abs().max()))
                    rms_all = max(rms_all, float((enc_a.float() - enc_t_post).pow(2).mean().sqrt()
                                                 / max(float(enc_t_post.float().pow(2).mean().sqrt()), 1e-9)))
                    caches = (n1, n2, n3)
                else:
                    enc_t, len_t, c1t, c2t, c3t = tmodel.encoder.cache_aware_stream_step(
                        processed_signal=chunk, processed_signal_length=lengths,
                        cache_last_channel=c1, cache_last_time=c2,
                        cache_last_channel_len=c3, keep_all_outputs=True,
                        drop_extra_pre_encoded=cfg.drop_extra_pre_encoded)
                    enc_a = tmodel._apply_prompt_to_encoded(enc_t)
                    len_a = len_t
                    caches = (c1t, c2t, c3t)
                best_ref = tmodel.decoding.rnnt_decoder_predictions_tensor(
                    encoder_output=enc_a, encoded_lengths=len_a,
                    return_hypotheses=True, partial_hypotheses=hyps_ref)
                hyps_ref = best_ref
                # the p8 stream decoded by NeMo vs the fp16-era parity reference
                # is the token gate: compare against a TORCH-encoded decode
            # whole-clip torch reference decode
            with torch.inference_mode():
                buf2 = rep.new_buffer((56, 6))
                buf2.append_audio(wave, stream_id=-1)
                caches2 = None
                hyps_torch = None
                ref_tokens = []
                for step_num, (chunk, lengths) in enumerate(iter(buf2)):
                    if caches2 is None:
                        caches2 = tmodel.encoder.get_initial_cache_state(batch_size=1)
                    c1, c2, c3 = caches2
                    ka = buf2.is_buffer_empty()
                    enc_t, len_t, c1t, c2t, c3t = tmodel.encoder.cache_aware_stream_step(
                        processed_signal=chunk, processed_signal_length=lengths,
                        cache_last_channel=c1, cache_last_time=c2,
                        cache_last_channel_len=c3, keep_all_outputs=ka,
                        drop_extra_pre_encoded=0 if step_num == 0 else cfg.drop_extra_pre_encoded)
                    caches2 = (c1t, c2t, c3t)
                    enc_post = tmodel._apply_prompt_to_encoded(enc_t)
                    best = tmodel.decoding.rnnt_decoder_predictions_tensor(
                        encoder_output=enc_post, encoded_lengths=len_t,
                        return_hypotheses=True, partial_hypotheses=hyps_torch)
                    hyps_torch = best
                ref_tokens = hyps_torch[0].y_sequence.tolist()
            # the ANE-p8 stream's cumulative tokens were captured via hyps_ref threading
            ane_tokens = best_ref[0].y_sequence.tolist()
            tok_match = sum(int(a == b) for a, b in zip(ane_tokens, ref_tokens))
            tok_total = len(ref_tokens)
            row = {"clip": c["path"], "audio_s": c["audio_s"],
                   "ref_tokens": len(ref_tokens), "ane_tokens": len(ane_tokens),
                   "token_identity_pct": round(100.0 * tok_match / max(tok_total, 1), 3),
                   "exact_match": bool(ane_tokens == ref_tokens),
                   "enc_max_abs": round(emax_all, 4),
                   "enc_rel_rms_worst": round(rms_all, 6)}
            if not row["exact_match"]:
                for i, (a, b) in enumerate(zip(ane_tokens, ref_tokens)):
                    if a != b:
                        row["first_diverge_idx"] = i
                        break
            report["clips"].append(row)
            log(json.dumps({k: row[k] for k in ("clip", "token_identity_pct",
                                                "exact_match")}, ensure_ascii=False))
    with open(PARITY, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log("parity rows →", PARITY)
    return report


def p50():
    """Encoder-only per-call A/B: p8 vs fp16 at the steady shape (load 65)."""
    import coremltools as ct
    import subprocess
    r = subprocess.run([PREFLIGHT], capture_output=True, text=True)
    out = r.stdout + r.stderr
    prov = next((l for l in out.splitlines() if l.startswith("PROVENANCE:")), "")
    if "✗" in out or "REFUSE" in out:
        log("preflight REFUSED:\n", out.strip())
        raise SystemExit(3)
    log("PROVENANCE:", prov)
    rows = []
    ins = {
        "mel": np.random.randn(1, 128, 65).astype(np.float32) * 0.1,
        "mel_len": np.array([65], dtype=np.int32),
        "cache_last_channel": np.zeros((24, 1, 56, 1024), dtype=np.float32),
        "cache_last_time": np.zeros((24, 1, 1024, 8), dtype=np.float32),
        "cache_last_channel_len": np.array([56], dtype=np.int32),
        "prompt_onehot": np.zeros((1, 1, 128), dtype=np.float32),
    }
    ins["prompt_onehot"][0, 0, 32] = 1.0
    for label, pkg in (("fp16", "openthai_enc_step.mlpackage"),
                       ("p8", "openthai_enc_step_p8.mlpackage")):
        m = ct.models.MLModel(os.path.join(ANE_DIR, pkg),
                              compute_units=ct.ComputeUnit.CPU_AND_NE)
        for _ in range(20):
            m.predict(ins)
        ts = []
        for _ in range(200):
            t0 = time.perf_counter_ns()
            m.predict(ins)
            ts.append((time.perf_counter_ns() - t0) / 1e6)
        ts.sort()
        rows.append({"cell": "p50-enc-step", "arm": label,
                     "p50_ms": round(ts[len(ts)//2], 3),
                     "p95_ms": round(ts[int(len(ts)*0.95)], 3),
                     "provenance": prov, "quiet": True})
        log(label, rows[-1])
    with open(CELLS, "a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all",
                    choices=["all", "convert", "census", "parity", "p50"])
    args = ap.parse_args()
    os.makedirs(ANE_DIR, exist_ok=True)
    if args.mode in ("all", "convert"):
        convert()
    if args.mode in ("all", "census"):
        census()
    if args.mode in ("all", "parity"):
        parity()
    if args.mode in ("all", "p50") or args.mode == "p50":
        p50()


if __name__ == "__main__":
    main()
