#!/usr/bin/env python3
"""Plan 633 P2.1 — the ANE encoder conversion (beshkenadze functional shape).

`cache_aware_stream_step` with the three streaming caches as explicit inputs AND outputs
(no MLState), TWO fixed-shape graphs at att_context [56,6] (the 560 ms tier):
  - first  chunk: [1,128,49] mel, drop_extra_pre_encoded=0, cache_len 0..7
  - steady chunk: [1,128,65] mel = 9 pre-encode cache + 56 new, drop_extra_pre_encoded=2
(the runtime streaming_cfg is the shape authority: chunk_size [49,56], pre_encode_cache
[0,9], valid_out_len 7, last_channel_cache_size 56 — the plan's [1,128,121] sketch came
from the base-model note and is corrected here, recorded in the plan).
The prompt layer (PromptStreamingMixin._apply_prompt_to_encoded) is INSIDE the graph with
the host-built one-hot prompt vector [1,1,128] as an input (the host-gather cell; the
baked-const variant is the P2.3 measured comparison). The mel frontend stays CPU fp32
(the CPU-tail law). int inputs are int32 (CoreML-native; the int32 island precedent),
cast to int64 inside the wrapper.

Modes: convert | compile | census | parity | all. NOT a timing instrument — every latency
claim stays quiet-rig gated (P1.3/P2.3).
"""
import argparse, json, os, subprocess, sys, time
import numpy as np

ASR = os.path.dirname(os.path.abspath(__file__))
NEMO_RAW = os.path.join(ASR, "nemo", "openthai2.0-asr-streaming_raw.nemo")
ANE_DIR = os.path.join(ASR, "ane")
CELLS = os.path.join(ASR, "p633_cells.jsonl")
PARITY = os.path.join(ASR, "p633_p21_parity.json")
PROMPT = "th-TH"
ATT = [56, 6]

import torch

def log(*a):
    print("[p633-p21]", *a, flush=True)


def _install_numpy2_cast_compat():
    """coremltools 9 torch frontend + numpy>=2: ops._cast's const path calls
    dtype(np.array([v])), which numpy 2 refuses for any ndim>=1 array
    (ops.py:3048 — 'only 0-dimensional arrays can be converted'). The op's own
    contract (ops.py:3041) guarantees scalar-or-size-1 at that point, so .item()
    is exact. In-script monkeypatch, no env mutation; recorded in the plan."""
    import numpy as _np
    if _np.lib.NumpyVersion(_np.__version__) < "2.0.0":
        # numpy 1.x: the original ops._cast path is legal (dtype(size-1 array)
        # coerces); the patch is only needed for numpy>=2 strictness. Measured:
        # with numpy 1.26 the PATCHED path itself trips a downstream const-builder
        # bug, the original converts clean.
        return
    import coremltools.converters.mil.frontend.torch.ops as _ctops
    if getattr(_ctops, "_p633_np2_cast", False):
        return
    _orig = _ctops._cast

    def _cast_np2(context, node, dtype, dtype_name):
        inputs = _ctops._get_inputs(context, node, expected=1)
        x = inputs[0]
        if x.can_be_folded_to_const():
            val = getattr(x, "val", None)
            if isinstance(val, _np.ndarray) and not isinstance(val, _np.number):
                if val.ndim >= 1:
                    if val.size != 1:
                        raise ValueError("input to cast must be either a scalar or a length 1 tensor")
                    val = val.item()
                if not isinstance(val, dtype):
                    res = _ctops.mb.const(val=dtype(val), name=node.name)
                else:
                    res = x
                context.add(res, node.name)
                return
        return _orig(context, node, dtype, dtype_name)

    _ctops._cast = _cast_np2
    _ctops._p633_np2_cast = True
    log("numpy2 cast compat installed")


def load_model(threads=1):
    torch.set_num_threads(threads)
    try:
        from nemo.core import typecheck as nemo_tc
        nemo_tc.set_typecheck_enabled(False)
    except Exception as e:
        log("typecheck disable failed (continuing):", e)
    from nemo.collections.asr.models import ASRModel
    from nemo.utils import model_utils
    cfg = ASRModel.restore_from(restore_path=NEMO_RAW, return_config=True)
    cls = model_utils.import_class_by_path(cfg.target)
    model = cls.restore_from(restore_path=NEMO_RAW, map_location=torch.device("cpu"))
    model.eval()
    model.set_inference_prompt(PROMPT)
    model.encoder.set_default_att_context_size(list(ATT))
    return model


class EncoderStep(torch.nn.Module):
    """Functional cache-aware stream step + in-graph prompt layer.

    Outputs: encoded [B,1024,7] (post-prompt, what the decoder consumes),
    encoded_len [B], the three next caches, next cache_len."""

    def __init__(self, model, drop_extra: int):
        super().__init__()
        self.encoder = model.encoder
        self.prompt_kernel = model.prompt_kernel
        self.num_prompts = model.num_prompts
        self.drop_extra = drop_extra

    def forward(self, mel, mel_len, cache_last_channel, cache_last_time,
                cache_last_channel_len, prompt_onehot):
        enc, enc_len, c1, c2, c3 = self.encoder.cache_aware_stream_step(
            processed_signal=mel,
            processed_signal_length=mel_len.to(torch.int64),
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len.to(torch.int64),
            keep_all_outputs=False,
            drop_extra_pre_encoded=self.drop_extra,
        )
        # == PromptStreamingMixin._apply_prompt_to_encoded with the fixed-index
        # scatter hoisted to a [1,1,128] host-built input broadcast over time.
        enc = enc.transpose(1, 2)  # [B,D,T] -> [B,T,D]
        T = enc.shape[1]
        p = prompt_onehot.expand(-1, T, -1)
        enc = self.prompt_kernel(torch.cat([enc, p], dim=-1)).to(enc.dtype)
        return enc.transpose(1, 2), enc_len, c1, c2, c3


def shape_plan(model):
    cfg = model.encoder.streaming_cfg
    return {
        "att_context": list(model.encoder.att_context_size),
        "chunk_size": list(cfg.chunk_size),
        "shift_size": list(cfg.shift_size),
        "pre_encode_cache": list(cfg.pre_encode_cache_size),
        "drop_extra_pre_encoded": int(cfg.drop_extra_pre_encoded),
        "valid_out_len": int(cfg.valid_out_len),
        "last_channel_cache_size": int(cfg.last_channel_cache_size),
        "prompt_index": int(model._inference_prompt_index),
        "num_prompts": int(model.num_prompts),
        "prompt_kernel": repr(model.prompt_kernel),
        "self_attention_model": model.encoder.self_attention_model,
        "att_context_style": model.encoder.att_context_style,
        "n_layers": len(model.encoder.layers),
        "d_model": model.encoder.d_model,
    }


def example_inputs(model, first: bool):
    cfg = model.encoder.streaming_cfg
    T = cfg.chunk_size[0] + cfg.pre_encode_cache_size[0] if first \
        else cfg.chunk_size[1] + cfg.pre_encode_cache_size[1]
    ch = torch.zeros(24, 1, int(cfg.last_channel_cache_size), model.encoder.d_model)
    t = torch.zeros(24, 1, model.encoder.d_model, 8)
    ch_len = torch.tensor([0 if first else int(cfg.last_channel_cache_size)], dtype=torch.int32)
    onehot = torch.zeros(1, 1, int(model.num_prompts))
    onehot[0, 0, int(model._inference_prompt_index)] = 1.0
    return {
        "mel": torch.randn(1, model.encoder._feat_in, T),
        "mel_len": torch.tensor([T], dtype=torch.int32),
        "cache_last_channel": ch,
        "cache_last_time": t,
        "cache_last_channel_len": ch_len,
        "prompt_onehot": onehot,
    }, T


def convert_graph(model, first: bool):
    """ONNX-first conversion (deviation from beshkenadze's direct-torch route,
    recorded): torch 2.15's tracer keeps static conv-output sizes symbolic, and
    the CoreML torch frontend needs scalar consts in shape position
    (torch.ones(1, max_audio_length, ...) fails rank-mismatch). torch.onnx.export
    with NO dynamic axes folds every size computation into graph constants; the
    only runtime-symbolic value left is cache_last_channel_len, which reaches the
    graph exclusively through mask/offset tensor arithmetic (no int() sites), so
    the CoreML ONNX frontend converts it as the int32 island. The mel length is a
    constant per graph (49 first / 65 steady — uniform windows; the short final
    flush is the CPU tail, never this graph)."""
    import coremltools as ct
    _install_numpy2_cast_compat()
    drop = 0 if first else int(model.encoder.streaming_cfg.drop_extra_pre_encoded)
    tag = "first" if first else "step"
    wrap = EncoderStep(model, drop).eval()
    ex, T = example_inputs(model, first)
    onnx_path = os.path.join(ANE_DIR, f"openthai_enc_{tag}.onnx")
    names_in = ["mel", "mel_len", "cache_last_channel", "cache_last_time",
                "cache_last_channel_len", "prompt_onehot"]
    names_out = ["encoded", "encoded_len", "cache_last_channel_next",
                 "cache_last_time_next", "cache_last_channel_next_len"]
    args = tuple(ex[n] for n in names_in)
    t0 = time.time()
    with torch.inference_mode():
        traced = torch.jit.trace(wrap, args, check_trace=False)
        try:
            traced = torch.jit.freeze(traced)
            log(f"[{tag}] jit.freeze applied (static-shape constant folding)")
        except Exception as e:
            log(f"[{tag}] jit.freeze skipped: {e}")
    log(f"[{tag}] traced T={T} in {time.time()-t0:.1f}s")

    inputs = [
        ct.TensorType(name="mel", shape=(1, model.encoder._feat_in, T), dtype=np.float32),
        ct.TensorType(name="mel_len", shape=(1,), dtype=np.int32),
        ct.TensorType(name="cache_last_channel", shape=tuple(ex["cache_last_channel"].shape), dtype=np.float32),
        ct.TensorType(name="cache_last_time", shape=tuple(ex["cache_last_time"].shape), dtype=np.float32),
        ct.TensorType(name="cache_last_channel_len", shape=(1,), dtype=np.int32),
        ct.TensorType(name="prompt_onehot", shape=(1, 1, int(model.num_prompts)), dtype=np.float32),
    ]
    kw = dict(inputs=inputs, compute_precision=ct.precision.FLOAT16,
              compute_units=ct.ComputeUnit.CPU_AND_NE,
              minimum_deployment_target=ct.target.macOS14)
    # NOTE: no outputs= kwarg — passing ct.OutputType names here diverges from the
    # traced module's own output tuple and trips a ct const-builder bug (measured:
    # the same graph converts clean without it; the inhomogeneous-(3,)-const
    # failure only reproduces with the outputs list present). Output names are
    # read from the converted spec below instead.
    t0 = time.time()
    mlmodel = ct.convert(traced, **kw)
    out_names = [o.name for o in mlmodel._spec.description.output]
    in_names = [i.name for i in mlmodel._spec.description.input]
    log(f"[{tag}] converted in {time.time()-t0:.1f}s inputs={in_names} outputs={out_names}")
    path = os.path.join(ANE_DIR, f"openthai_enc_{tag}.mlpackage")
    mlmodel.save(path)
    return path, out_names


def compile_pkg(pkg_path):
    subprocess.run(["xcrun", "coremlcompiler", "compile", pkg_path, ANE_DIR],
                   check=True, capture_output=True)
    return pkg_path.replace(".mlpackage", ".mlmodelc")


# --------------------------------------------------------------------------- census
def census(modelc_paths):
    """P2.2 placement census — the instrument law measured here: MLComputePlan's
    default load uses the ALL plan and reports GPU preference for every placed op;
    the plan MUST be loaded at the posture it will run at (CPU_AND_NE) via the
    async load_from_path_on_device. Under that plan: every placed op prefers ANE,
    the unplaced remainder is the const/mask island (the beshkenadze precedent)."""
    import asyncio
    from coremltools.models.compute_plan import MLComputePlan
    import coremltools as ct
    report = {}
    for path in modelc_paths:
        tag = "first" if "first" in os.path.basename(path) else "step"
        plan = asyncio.run(MLComputePlan.load_from_path_on_device(
            path, compute_units=ct.ComputeUnit.CPU_AND_NE))
        prog = plan.model_structure.program
        ops = []
        for fn in prog.functions.values():
            stack = [fn.block]
            while stack:
                blk = stack.pop()
                ops.extend(blk.operations)
                for sub in getattr(blk, "blocks", []) or []:
                    stack.append(sub)
        pref_counts = {}
        op_counts = {}
        n_ops = len(ops)
        for op in ops:
            u = plan.get_compute_device_usage_for_mlprogram_operation(op)
            pref = getattr(u, "preferred_compute_device", None)
            pname = type(pref).__name__.replace("MLComputeDevice", "").replace("ML", "") if pref else "unplaced"
            pref_counts[pname] = pref_counts.get(pname, 0) + 1
            t = getattr(op, "operator_name", "?")
            op_counts[t] = op_counts.get(t, 0) + 1
        placed = n_ops - pref_counts.get("unplaced", 0)
        ane = pref_counts.get("NeuralEngineComputeDevice", 0) + pref_counts.get("N", 0)
        report[tag] = {
            "path": path, "n_ops": n_ops, "placed": placed,
            "pref_counts": pref_counts,
            "placed_ops_ane_pct": round(100.0 * ane / placed, 2) if placed else None,
            "op_counts_by_type": op_counts,
            "note": "plan loaded at CPU_AND_NE (async load_from_path_on_device); "
                    "unplaced = const/mask island ops without a preferred device; "
                    "cost-weighted axis pending get_estimated_cost (API returned None "
                    "at this ct version) — op-count-weighted reported alongside",
        }
        log(f"[census {tag}] ops={n_ops} pref={pref_counts} placed_ane_pct={report[tag]['placed_ops_ane_pct']}")
    with open(os.path.join(ASR, "p633_p21_census.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    return report


# --------------------------------------------------------------------------- parity
def parity(model, first_path, step_path, out_names, max_clips=3):
    import coremltools as ct
    import soundfile as sf
    sys.path.insert(0, ASR)
    sys.path.insert(0, os.path.join(ASR, "iapp", "serve"))
    from p633_p1 import load_manifest
    import chinda_asr as ca

    asr = ca.ChindaASR(model_path=NEMO_RAW, device="cpu", num_threads=1)
    asr.load()
    rep = asr._replicas[0]
    model = rep.model
    cfg = model.encoder.streaming_cfg

    coreml = {
        "first": ct.models.MLModel(first_path, compute_units=ct.ComputeUnit.CPU_AND_NE),
        "step": ct.models.MLModel(step_path, compute_units=ct.ComputeUnit.CPU_AND_NE),
    }

    def predict(tag, mel, mel_len, c1, c2, c3, ch_len, onehot):
        ins = {
            "mel": mel.numpy().astype(np.float32),
            "mel_len": mel_len.numpy().astype(np.int32),
            "cache_last_channel": c1.numpy().astype(np.float32),
            "cache_last_time": c2.numpy().astype(np.float32),
            "cache_last_channel_len": ch_len.numpy().astype(np.int32),
            "prompt_onehot": onehot.numpy().astype(np.float32),
        }
        outs = coreml[tag].predict(ins)
        vals = [outs[n] for n in out_names[tag]]
        enc = torch.from_numpy(np.array(vals[0], dtype=np.float32))
        enc_len = torch.from_numpy(np.array(vals[1]).astype(np.int64))
        n1 = torch.from_numpy(np.array(vals[2], dtype=np.float32))
        n2 = torch.from_numpy(np.array(vals[3], dtype=np.float32))
        n3 = torch.from_numpy(np.array(vals[4]).astype(np.int64))
        return enc, enc_len, n1, n2, n3

    onehot = torch.zeros(1, 1, int(model.num_prompts))
    onehot[0, 0, int(model._inference_prompt_index)] = 1.0

    clips = []
    per_family = {"fleurs-th": 2, "tvspeech": 1}
    for family in ("fleurs-th", "tvspeech"):
        man = load_manifest(family)
        clips.extend(man["clips"][: per_family[family]])

    report = {"clips": [], "shape_log": []}
    for c in clips:
        wave, sr = sf.read(os.path.join(ASR, c["path"]), dtype="float32")
        if wave.ndim > 1:
            wave = wave[:, 0]
        with torch.inference_mode():
            buf = rep.new_buffer((56, 6))
            buf.append_audio(wave, stream_id=-1)
            caches = None
            hyps_ref = pred_ref = None
            hyps_ane = None
            steps = []
            tok_match = tok_total = 0
            for step_num, (chunk, lengths) in enumerate(iter(buf)):
                if caches is None:
                    caches = model.encoder.get_initial_cache_state(batch_size=1)
                c1, c2, c3 = caches
                ka = buf.is_buffer_empty()
                drop = 0 if step_num == 0 else cfg.drop_extra_pre_encoded
                tag = "first" if step_num == 0 else "step"
                if step_num < 3:
                    report["shape_log"].append({
                        "clip": c["path"], "step": step_num, "tag": tag,
                        "chunk": list(chunk.shape), "len": lengths.tolist(),
                        "keep_all": bool(ka), "drop": drop,
                    })
                # ---- reference (torch, post-prompt via the model's own hook)
                (enc_t, len_t, c1t, c2t, c3t) = model.encoder.cache_aware_stream_step(
                    processed_signal=chunk, processed_signal_length=lengths,
                    cache_last_channel=c1, cache_last_time=c2, cache_last_channel_len=c3,
                    keep_all_outputs=ka, drop_extra_pre_encoded=drop)
                enc_t_post = model._apply_prompt_to_encoded(enc_t)
                best_t = model.decoding.rnnt_decoder_predictions_tensor(
                    encoder_output=enc_t_post, encoded_lengths=len_t,
                    return_hypotheses=True, partial_hypotheses=hyps_ref)
                hyps_ref = best_t
                tok_ref = best_t[0].y_sequence.tolist()
                # ---- ANE side
                if not ka:
                    enc_a, len_a, n1, n2, n3 = predict(
                        tag, chunk, lengths, c1, c2, c3,
                        c3, onehot)  # same ch_len input as torch side
                    emax = float((enc_a.float() - enc_t_post).abs().max())
                    rms_t = float(enc_t_post.float().pow(2).mean().sqrt())
                    rms_d = float((enc_a.float() - enc_t_post).pow(2).mean().sqrt())
                    cmax = max(float((n1 - c1t).abs().max()), float((n2 - c2t).abs().max()))
                    steps.append({
                        "step": step_num, "tag": tag,
                        "encoded_max_abs": round(emax, 6),
                        "encoded_rel_rms": round(rms_d / max(rms_t, 1e-9), 8),
                        "cache_max_abs": round(cmax, 6),
                        "len_match": bool(len_a.tolist() == len_t.tolist()),
                        "cache_len": n3.tolist(), "torch_cache_len": c3t.tolist(),
                    })
                    caches = (n1, n2, n3)
                    best_a = model.decoding.rnnt_decoder_predictions_tensor(
                        encoder_output=enc_a, encoded_lengths=len_a,
                        return_hypotheses=True, partial_hypotheses=hyps_ane)
                    hyps_ane = best_a
                    tok_a = best_a[0].y_sequence.tolist()
                else:
                    # final flush: CPU tail on both sides (the ANE steady graph never
                    # sees keep_all_outputs=True — the CPU-tail law)
                    enc_a, len_a = enc_t_post, len_t
                    caches = (c1t, c2t, c3t)
                    best_a = model.decoding.rnnt_decoder_predictions_tensor(
                        encoder_output=enc_a, encoded_lengths=len_a,
                        return_hypotheses=True, partial_hypotheses=hyps_ane)
                    hyps_ane = best_a
                    tok_a = best_a[0].y_sequence.tolist()
                    steps.append({"step": step_num, "tag": "flush-cpu", "note": "keep_all_outputs=True"})
                for a, b in zip(tok_a, tok_ref):
                    tok_total += 1
                    tok_match += int(a == b)
                if tok_a != tok_ref:
                    steps.append({"step": step_num, "tok_diverge": True,
                                  "ane_tail": tok_a[-8:], "ref_tail": tok_ref[-8:]})
            row = {
                "clip": c["path"], "family": c.get("family", "?"), "audio_s": c["audio_s"],
                "n_steps": len(steps), "steps": steps,
                "token_identity_pct": round(100.0 * tok_match / max(tok_total, 1), 3),
                "ref_cer": None,
            }
            report["clips"].append(row)
            log(json.dumps({k: row[k] for k in ("clip", "n_steps", "token_identity_pct")},
                           ensure_ascii=False))
    with open(PARITY, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log("parity rows →", PARITY)
    return report


# --------------------------------------------------------------------------- main
def p24_facts(model):
    """P2.4 prep — dump the decoder/joint structure + the exact stream-decode call
    shape the ANE decoder+joint wrapper must reproduce (facts only, no conversion)."""
    dec = model.decoder
    jn = model.joint
    facts = {
        "decoder_class": type(dec).__name__,
        "pred_hidden": int(dec.pred_hidden),
        "pred_rnn_layers": int(dec.pred_rnn_layers),
        "joint_class": type(jn).__name__,
        "joint_hidden": int(jn.joint_hidden),
        "joint_net": [type(m).__name__ for m in jn.joint_net],
        "vocab": int(jn.joint_net[-1].out_features) if hasattr(jn.joint_net[-1], "out_features") else None,
        "enc_hidden": int(jn.enc_hidden) if hasattr(jn, "enc_hidden") else None,
        "pred_hidden_joint": int(jn.pred_hidden) if hasattr(jn, "pred_hidden") else None,
        "project_encoder": repr(getattr(jn, "project_encoder", None)),
        "project_decoder": repr(getattr(jn, "project_decoder", None)),
        "prompt_kernel": repr(model.prompt_kernel),
    }
    log("p24 facts:", json.dumps(facts))
    # the stream-decall shape: one greedy step's calls (decoder LSTM step + joint)
    with torch.inference_mode():
        enc_frame = torch.randn(1, 1, model.encoder.d_model)
        enc_len = torch.tensor([1], dtype=torch.int64)
        print_t = getattr(model.decoding, "_SOS", None)
        log("decoding class:", type(model.decoding).__name__,
            "| has rnnt_decoder_predictions_tensor:",
            hasattr(model.decoding, "rnnt_decoder_predictions_tensor"))
        from nemo.collections.asr.parts.submodules.rnnt_greedy_decoding import GreedyRNNTInfer
        log("greedy infer class available")
    with open(os.path.join(ASR, "p633_p24_facts.json"), "w") as f:
        json.dump(facts, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all",
                    choices=["all", "convert", "compile", "census", "parity", "facts", "p24facts"])
    args = ap.parse_args()

    os.makedirs(ANE_DIR, exist_ok=True)
    if args.mode == "p24facts":
        model = load_model()
        p24_facts(model)
        return
    model = load_model()
    facts = shape_plan(model)
    log("facts:", json.dumps({k: v for k, v in facts.items() if k != "prompt_kernel"}))
    with open(os.path.join(ASR, "p633_p21_facts.json"), "w") as f:
        json.dump(facts, f, indent=2)

    first_pkg = os.path.join(ANE_DIR, "openthai_enc_first.mlpackage")
    step_pkg = os.path.join(ANE_DIR, "openthai_enc_step.mlpackage")
    first_c = first_pkg.replace(".mlpackage", ".mlmodelc")
    step_c = step_pkg.replace(".mlpackage", ".mlmodelc")

    if args.mode in ("all", "convert"):
        for first, pkg in ((True, first_pkg), (False, step_pkg)):
            if os.path.exists(pkg):
                log("exists, skipping convert:", pkg)
                continue
            convert_graph(model, first)
    out_names = {}
    if args.mode in ("all", "compile", "census", "parity", "all"):
        for pkg in (first_pkg, step_pkg):
            if os.path.exists(pkg) and not os.path.exists(pkg.replace(".mlpackage", ".mlmodelc")):
                compile_pkg(pkg)
        import coremltools as ct
        for tag, pkg in (("first", first_pkg), ("step", step_pkg)):
            m = ct.models.MLModel(pkg)
            out_names[tag] = [o.name for o in m._spec.description.output]
        log("output names:", json.dumps(out_names))

    if args.mode in ("all", "census"):
        census([p for p in (first_c, step_c) if os.path.exists(p)])
    if args.mode in ("all", "parity"):
        # ct's MLModel loader wants the .mlpackage (this macOS's coremlcompiler
        # emits a manifest-less .mlmodelc that the NATIVE API reads fine — that
        # bundle stays the host-integration artifact for riir-infer-ane)
        parity(model, first_pkg, step_pkg, out_names)


if __name__ == "__main__":
    main()
