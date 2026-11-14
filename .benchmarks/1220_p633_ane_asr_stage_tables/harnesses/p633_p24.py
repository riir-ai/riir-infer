#!/usr/bin/env python3
"""Plan 633 P2.4 — the decoder⊕joint ANE conversion (the ~27 M-param RNN-T tail).

TWO graph families, both fixed-shape, fp16, CPU_AND_NE (the P2.1 laws carry over:
numpy 1.26 pin — NO cast shim; jit.freeze only where needed; .mlpackage loads;
int inputs int32):
  - dec_step: the prediction LSTM ONE STEP — g_in [1,1,640] fp32 (host builds it:
    zeros for SOS, else the embedding row from the exported npz table), h/c
    [2,1,640] in AND out (no MLState — the explicit-cache law). Traced UNFROZEN
    (no conv → no static-size issue; freeze unnecessary, recorded).
  - joint_bF: joint_net(enc + pred) at FOUR fixed frame counts F ∈ {1,28,56,112} —
    F=1 is Posture A (per-step, tier-invariant), F>1 is the Posture B measurement
    vehicle (the SAME g broadcast across F frames prices the vocab matmul at batch
    F; it is NOT decode semantics — P0.2 Posture B). enc [1,F,1024] is the
    POST-prompt encoded frame (d_model 1024 confirmed by p633_p24_facts).
The host greedy loop mirrors NeMo GreedyRNNTInfer._greedy_decode EXACTLY (state
commits only on non-blank; symbols_added counts every inner iteration;
max_symbols from the model's own decoding cfg; no blank bias — that is iapp's
decode_pass, not the reference).

Modes: convert | compile | census | parity | p50 | all. parity = correctness
(load-tolerant). p50 = TIMING — runs bench_preflight.sh and REFUSES on ✗
(fail-closed quiet gate; PROVENANCE quoted in every row). Rows:
p633_p24_parity.json / p633_p24_census.json / p633_p24_cells.jsonl.
"""
import argparse, json, os, subprocess, sys, time
import numpy as np

ASR = os.path.dirname(os.path.abspath(__file__))
NEMO_BASE = os.environ.get("P633_NEMO_BASE") == "1"
if NEMO_BASE:
    # P2.6 — the SAME recipe on the base model (G5's cell): same class, same
    # dec/joint dims; the deltas are vocab 13,088 (the plan's 13,087 is the
    # card's off-by-one again) and the streaming tier (chunk [25,32],
    # att_context [56,3], valid_out_len 4). Outputs carry the base_ prefix so
    # the fine-tune artifacts stay untouched.
    NEMO_RAW = os.path.join(ASR, "nemo_base", "nemotron-3.5-asr-streaming-0.6b.nemo")
    TAG = "base_"
else:
    NEMO_RAW = os.path.join(ASR, "nemo", "openthai2.0-asr-streaming_raw.nemo")
    TAG = ""
ANE_DIR = os.path.join(ASR, "ane")
PREFLIGHT = "/Users/katopz/git/riir-reflex/scripts/bench_preflight.sh"
PARITY = os.path.join(ASR, f"p633_{TAG}p24_parity.json")
CENSUS = os.path.join(ASR, f"p633_{TAG}p24_census.json")
CELLS = os.path.join(ASR, f"p633_{TAG}p24_cells.jsonl")
EMBED_NPZ = os.path.join(ANE_DIR, f"openthai_{TAG}embed.npz")
BATCHES = (1, 28, 56, 112)
# P0.2 roofline denominators (AnePeaks::m5, vocab matmul 640×14989 fp16):
# runtime_ms at batch 1/28/56/112 — the Posture A/B predictions price the
# EXCESS over these, never compute_ms.
P02_RUNTIME_MS = {1: 0.3371, 28: 0.3404, 56: 0.3514, 112: 0.3881}

import torch

def log(*a):
    print("[p633-p24]", *a, flush=True)


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
    if NEMO_BASE:
        # the base keeps its OWN streaming posture (att [56,3], chunk [25,32]) —
        # never override; its prompt vocabulary is the card's, not th-TH
        try:
            model.set_inference_prompt("th")
        except Exception as e:
            log("base prompt set skipped:", e)
    else:
        model.set_inference_prompt("th-TH")
        model.encoder.set_default_att_context_size(list([56, 6]))
    return model


# ------------------------------------------------------------------ wrappers
class DecoderStep(torch.nn.Module):
    """One prediction-LSTM step: g_in [1,1,640] + state (h, c) [2,1,640] →
    g [1,1,640] + next state. The SOS/blank zero-vector and the token embedding
    are HOST-side (the embedding table rides the exported npz; the graph stays a
    pure LSTM step — the ANE-friendly shape)."""

    def __init__(self, dec):
        super().__init__()
        self.rnn = dec.prediction["dec_rnn"]

    def forward(self, g_in, h, c):
        g, (h2, c2) = self.rnn(g_in, (h, c))
        return g, h2, c2


class JointStep(torch.nn.Module):
    """joint_after_projection(project_encoder(f), project_prednet(g)) — the
    NeMo math exactly: joint_net(enc(f) + pred(g)) (rnnt.py:1696-1706)."""

    def __init__(self, jn, frames: int):
        super().__init__()
        self.enc = jn.enc
        self.pred = jn.pred
        self.joint_net = jn.joint_net
        self.frames = frames

    def forward(self, enc, pred):
        return self.joint_net(self.enc(enc) + self.pred(pred))


def joint_example_inputs(frames: int, d_model: int = 1024):
    return {
        "enc": torch.randn(1, frames, d_model),
        "pred": torch.randn(1, frames, 640),
    }


def convert_all(model):
    import coremltools as ct

    os.makedirs(ANE_DIR, exist_ok=True)
    # embedding table export (host-side lookup substrate)
    if not os.path.exists(EMBED_NPZ):
        emb = model.decoder.prediction["embed"].weight.detach().numpy().astype(np.float32)
        np.savez_compressed(EMBED_NPZ, embed=emb)
        log("embed table exported:", emb.shape, "→", os.path.basename(EMBED_NPZ))
    else:
        log("embed table exists:", EMBED_NPZ)

    jobs = [("dec_step", DecoderStep(model.decoder),
             {"g_in": torch.randn(1, 1, 640),
              "h": torch.zeros(2, 1, 640),
              "c": torch.zeros(2, 1, 640)},
             {"g_in": (1, 1, 640), "h": (2, 1, 640), "c": (2, 1, 640)}, False)]
    dm = int(model.encoder.d_model)
    for F in BATCHES:
        ex = joint_example_inputs(F, dm)
        jobs.append((f"joint_b{F}", JointStep(model.joint, F), ex,
                     {"enc": (1, F, dm), "pred": (1, F, 640)}, True))

    paths = {}
    for tag, wrap, ex, shapes, do_freeze in jobs:
        wrap = wrap.eval()  # explicit: the joint_net Dropout must trace as identity
        pkg = os.path.join(ANE_DIR, f"openthai_{TAG}{tag}.mlpackage")
        paths[tag] = pkg
        if os.path.exists(pkg):
            log("exists, skipping convert:", pkg)
            continue
        names_in = list(ex.keys())
        args = tuple(ex[n] for n in names_in)
        t0 = time.time()
        with torch.inference_mode():
            traced = torch.jit.trace(wrap, args, check_trace=False)
            if do_freeze:
                # joint only: freeze's constant folding is harmless here; the
                # decoder stays unfrozen (no conv → no static-size issue).
                try:
                    traced = torch.jit.freeze(traced)
                    log(f"[{tag}] jit.freeze applied")
                except Exception as e:
                    log(f"[{tag}] jit.freeze skipped: {e}")
        log(f"[{tag}] traced in {time.time()-t0:.1f}s")
        inputs = [ct.TensorType(name=n, shape=shapes[n], dtype=np.float32)
                  for n in names_in]
        kw = dict(inputs=inputs, compute_precision=ct.precision.FLOAT16,
                  compute_units=ct.ComputeUnit.CPU_AND_NE,
                  minimum_deployment_target=ct.target.macOS14)
        # no outputs= kwarg (the documented ct const-builder bug — P2.1 note)
        t0 = time.time()
        mlmodel = ct.convert(traced, **kw)
        in_names = [i.name for i in mlmodel._spec.description.input]
        out_names = [o.name for o in mlmodel._spec.description.output]
        log(f"[{tag}] converted in {time.time()-t0:.1f}s inputs={in_names} outputs={out_names}")
        mlmodel.save(pkg)
    return paths


def compile_pkg(pkg_path):
    subprocess.run(["xcrun", "coremlcompiler", "compile", pkg_path, ANE_DIR],
                   check=True, capture_output=True)
    return pkg_path.replace(".mlpackage", ".mlmodelc")


# ------------------------------------------------------------------ census
def census(paths):
    """P2.4 placement census at the posture it runs at — AND the P2.2 pending
    cost-weighted axis, CLOSED here: the SYNC MLComputePlan.load_from_path(path,
    compute_units=CPU_AND_NE) variant returns MLComputePlanCost(weight=...) per
    op (the async load_from_path_on_device returned None at ct 9 — the previous
    session's instrument finding, superseded)."""
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan
    report = {}
    for tag, pkg in paths.items():
        modelc = pkg.replace(".mlpackage", ".mlmodelc")
        if not os.path.exists(modelc):
            log("census skip (no .mlmodelc):", tag)
            continue
        plan = MLComputePlan.load_from_path(modelc, compute_units=ct.ComputeUnit.CPU_AND_NE)
        prog = plan.model_structure.program
        ops = []
        for fn in prog.functions.values():
            stack = [fn.block]
            while stack:
                blk = stack.pop()
                ops.extend(blk.operations)
                for sub in getattr(blk, "blocks", []) or []:
                    stack.append(sub)
        pref_counts, op_counts, costs = {}, {}, []
        for op in ops:
            u = plan.get_compute_device_usage_for_mlprogram_operation(op)
            pref = getattr(u, "preferred_compute_device", None)
            pname = type(pref).__name__.replace("MLComputeDevice", "").replace("ML", "") if pref else "unplaced"
            pref_counts[pname] = pref_counts.get(pname, 0) + 1
            t = getattr(op, "operator_name", "?")
            op_counts[t] = op_counts.get(t, 0) + 1
            try:
                c = plan.get_estimated_cost_for_mlprogram_operation(op)
                w = getattr(c, "weight", None)
                costs.append((pname, t, float(w)) if w is not None else None)
            except Exception:
                costs.append(None)
        placed = len(ops) - pref_counts.get("unplaced", 0)
        ane = pref_counts.get("NeuralEngineComputeDevice", 0) + pref_counts.get("N", 0)
        cost_rows = [r for r in costs if r]
        cost_total = sum(r[2] for r in cost_rows)
        cost_ane = sum(r[2] for r in cost_rows if r[0] == "NeuralEngineComputeDevice" or r[0] == "N")
        report[tag] = {
            "tag": tag, "n_ops": len(ops), "placed": placed,
            "pref_counts": pref_counts, "op_counts_by_type": op_counts,
            "placed_ops_ane_pct": round(100.0 * ane / placed, 2) if placed else None,
            "cost_weighted_ane_pct": round(100.0 * cost_ane / cost_total, 2) if cost_total else None,
            "n_cost_rows": len(cost_rows),
            "note": "plan loaded at CPU_AND_NE via the SYNC load_from_path "
                    "(the async on-device loader returned None costs at ct 9 — "
                    "superseded instrument finding); cost = MLComputePlanCost.weight",
        }
        log(f"[census {tag}] ops={len(ops)} pref={pref_counts} "
            f"placed_ane_pct={report[tag]['placed_ops_ane_pct']} "
            f"cost_ane_pct={report[tag]['cost_weighted_ane_pct']} "
            f"cost_rows={len(cost_rows)}/{len(ops)}")
    with open(CENSUS, "w") as f:
        json.dump(report, f, indent=2, default=str)
    return report


# ------------------------------------------------------------------ host loop
class AneDecodeTail:
    """The ANE decoder⊕joint greedy loop — NeMo _greedy_decode mirrored exactly:
    state commits ONLY on non-blank; symbols_added counts every inner iteration
    (the NeMo quirk); max_symbols from the model's decoding cfg; no blank bias
    (that is iapp's decode_pass, never the reference)."""

    def __init__(self, model, paths, n_threads=1):
        import coremltools as ct
        self.blank = model.decoder.blank_idx
        self.embed = np.load(EMBED_NPZ)["embed"]  # [14990, 640] fp32
        cfg = getattr(getattr(model, "decoding", None), "cfg", None)
        ms = getattr(getattr(cfg, "strategy", None), "max_symbols_per_step", None)
        self.max_symbols = int(ms) if ms else None
        log(f"decode tail: blank={self.blank} max_symbols={self.max_symbols}")
        self.dec = ct.models.MLModel(paths["dec_step"],
                                     compute_units=ct.ComputeUnit.CPU_AND_NE)
        self.joint1 = ct.models.MLModel(paths["joint_b1"],
                                        compute_units=ct.ComputeUnit.CPU_AND_NE)
        self.dec_out = [o.name for o in self.dec._spec.description.output]
        self.joint_out = [o.name for o in self.joint1._spec.description.output]
        self.joint_b = {F: ct.models.MLModel(paths[f"joint_b{F}"],
                      compute_units=ct.ComputeUnit.CPU_AND_NE) for F in BATCHES}

    def initial_state(self):
        z = np.zeros((2, 1, 640), dtype=np.float32)
        return z.copy(), z.copy()

    def _g_in(self, last_token):
        if last_token is None:
            return np.zeros((1, 1, 640), dtype=np.float32)
        return self.embed[last_token:last_token + 1, :].reshape(1, 1, 640)

    def pred_step(self, last_token, h, c):
        outs = self.dec.predict({"g_in": self._g_in(last_token),
                                 "h": h.astype(np.float32),
                                 "c": c.astype(np.float32)})
        g = np.array(outs[self.dec_out[0]], dtype=np.float32).reshape(1, 1, 640)
        h2 = np.array(outs[self.dec_out[1]], dtype=np.float32)
        c2 = np.array(outs[self.dec_out[2]], dtype=np.float32)
        return g, h2, c2

    def joint_step(self, enc_frame, g):
        # enc_frame [1,1,1024] fp32 → logits [14989] fp32
        outs = self.joint1.predict({"enc": enc_frame.astype(np.float32),
                                    "pred": g.astype(np.float32)})
        return np.array(outs[self.joint_out[0]], dtype=np.float32).reshape(-1)

    def decode(self, enc, enc_len, state=None, last_token=None):
        """enc [1,T,1024] fp32 (post-prompt, ANE-encoded), enc_len int.
        Returns (tokens, state, last_token, n_joint_calls)."""
        T = int(enc_len)
        h, c = state if state is not None else self.initial_state()
        tokens, n_calls = [], 0
        for t in range(T):
            f = enc[:, t:t + 1, :]
            not_blank, symbols = True, 0
            while not_blank and (self.max_symbols is None or symbols < self.max_symbols):
                g, h2, c2 = self.pred_step(last_token, h, c)
                logits = self.joint_step(f, g)
                n_calls += 1
                k = int(np.argmax(logits))
                if k == self.blank:
                    not_blank = False
                else:
                    tokens.append(k)
                    h, c = h2, c2
                    last_token = k
                symbols += 1
        return tokens, (h, c), last_token, n_calls


# ------------------------------------------------------------------ parity
def parity(model, enc_paths, max_clips=3):
    """Correctness only (load-tolerant): BOTH decoders consume the SAME
    ANE-encoded stream — ref = NeMo greedy (rnnt_decoder_predictions_tensor,
    threaded hyps, torch CPU), ANE = the host loop. Gate: token identity per
    clip. Numeric drift rows on the first clip (g / logits max-abs + rel-rms +
    argmax agreement per step)."""
    import coremltools as ct
    import soundfile as sf
    sys.path.insert(0, ASR)
    sys.path.insert(0, os.path.join(ASR, "iapp", "serve"))
    from p633_p1 import load_manifest
    import chinda_asr as ca

    asr = ca.ChindaASR(model_path=NEMO_RAW, device="cpu", num_threads=1)
    asr.load()
    rep = asr._replicas[0]
    tmodel = rep.model
    cfg = tmodel.encoder.streaming_cfg

    enc_cml = {
        "first": ct.models.MLModel(enc_paths["first"], compute_units=ct.ComputeUnit.CPU_AND_NE),
        "step": ct.models.MLModel(enc_paths["step"], compute_units=ct.ComputeUnit.CPU_AND_NE),
    }
    enc_out = {}
    for tag, pkg in enc_paths.items():
        m = ct.models.MLModel(pkg)
        enc_out[tag] = [o.name for o in m._spec.description.output]

    tail = AneDecodeTail(model, JOINT_PATHS)
    onehot = torch.zeros(1, 1, int(tmodel.num_prompts))
    onehot[0, 0, int(tmodel._inference_prompt_index)] = 1.0

    def enc_predict(tag, mel, mel_len, c1, c2, c3, ch_len):
        ins = {
            "mel": mel.numpy().astype(np.float32),
            "mel_len": mel_len.numpy().astype(np.int32),
            "cache_last_channel": c1.numpy().astype(np.float32),
            "cache_last_time": c2.numpy().astype(np.float32),
            "cache_last_channel_len": ch_len.numpy().astype(np.int32),
            "prompt_onehot": onehot.numpy().astype(np.float32),
        }
        outs = enc_cml[tag].predict(ins)
        vals = [outs[n] for n in enc_out[tag]]
        enc = torch.from_numpy(np.array(vals[0], dtype=np.float32))
        enc_len = torch.from_numpy(np.array(vals[1]).astype(np.int64))
        n1 = torch.from_numpy(np.array(vals[2], dtype=np.float32))
        n2 = torch.from_numpy(np.array(vals[3], dtype=np.float32))
        n3 = torch.from_numpy(np.array(vals[4]).astype(np.int64))
        return enc, enc_len, n1, n2, n3

    clips = []
    per_family = {"fleurs-th": 2, "tvspeech": 1}
    for family in ("fleurs-th", "tvspeech"):
        man = load_manifest(family)
        clips.extend(man["clips"][: per_family[family]])

    report = {"clips": [], "drift_first_clip": []}
    for ci, c in enumerate(clips):
        wave, sr = sf.read(os.path.join(ASR, c["path"]), dtype="float32")
        if wave.ndim > 1:
            wave = wave[:, 0]
        with torch.inference_mode():
            buf = rep.new_buffer((56, 6))
            buf.append_audio(wave, stream_id=-1)
            caches = None
            hyps_ref = None
            a_state, a_last = None, None
            ane_toks = []
            ref_final = None
            ref_chunks = 0
            joint_calls = 0
            drift = ci == 0
            for step_num, (chunk, lengths) in enumerate(iter(buf)):
                if caches is None:
                    caches = tmodel.encoder.get_initial_cache_state(batch_size=1)
                c1, c2, c3 = caches
                ka = buf.is_buffer_empty()
                drop = 0 if step_num == 0 else cfg.drop_extra_pre_encoded
                tag = "first" if step_num == 0 else "step"
                if not ka:
                    enc_a, len_a, n1, n2, n3 = enc_predict(tag, chunk, lengths, c1, c2, c3, c3)
                    caches = (n1, n2, n3)
                else:
                    # final flush: the CPU tail (the ANE steady graph never sees
                    # keep_all_outputs=True — the CPU-tail law)
                    enc_t, len_t, c1t, c2t, c3t = tmodel.encoder.cache_aware_stream_step(
                        processed_signal=chunk, processed_signal_length=lengths,
                        cache_last_channel=c1, cache_last_time=c2,
                        cache_last_channel_len=c3, keep_all_outputs=True,
                        drop_extra_pre_encoded=drop)
                    enc_a = tmodel._apply_prompt_to_encoded(enc_t)
                    len_a = len_t
                    caches = (c1t, c2t, c3t)
                # ---- reference decode (NeMo, threaded hyps) on THIS chunk.
                # STREAMING y_sequence is CUMULATIVE (whole transcript so far) —
                # the full transcript is the LAST chunk's sequence, never an
                # extend() across chunks (the prefix-repeat artifact). The
                # per-chunk NEW tokens are the prefix-strip of the cumulative.
                best_ref = tmodel.decoding.rnnt_decoder_predictions_tensor(
                    encoder_output=enc_a, encoded_lengths=len_a,
                    return_hypotheses=True, partial_hypotheses=hyps_ref)
                hyps_ref = best_ref
                ref_final = best_ref[0].y_sequence.tolist()
                ref_chunks += 1
                # ---- ANE decode on the SAME encoded chunk
                enc_np = enc_a.transpose(1, 2).numpy().astype(np.float32)  # [1,T,1024]
                state_in, last_in = a_state, a_last
                toks, a_state, a_last, nc = tail.decode(
                    enc_np, int(len_a[0]), state=a_state, last_token=a_last)
                joint_calls += nc
                ane_toks.extend(toks)
                # ---- numeric drift on the first clip's ANE steps (REPLAYS the
                # chunk's ENTRY state — captured above, never the post-decode one)
                if drift and not ka:
                    Tn = enc_np.shape[1]
                    hs, cs = state_in if state_in is not None else tail.initial_state()
                    lt = last_in
                    eq = tot = 0
                    gmax = gden = lmax = lden = 0.0
                    for t in range(min(Tn, 4)):
                        f = enc_np[:, t:t + 1, :]
                        f_t = torch.from_numpy(f)
                        not_blank, symbols = True, 0
                        while not_blank and symbols < (tail.max_symbols or 99):
                            g_an, h2, c2 = tail.pred_step(lt, hs, cs)
                            # torch twin for THIS step
                            if lt is None:
                                g_t, hid_t = tmodel.decoder.predict(None, (torch.from_numpy(hs), torch.from_numpy(cs)),
                                                                    add_sos=False, batch_size=1)
                            else:
                                y = torch.tensor([[lt]], dtype=torch.long)
                                g_t, hid_t = tmodel.decoder.predict(y, (torch.from_numpy(hs), torch.from_numpy(cs)),
                                                                    add_sos=False, batch_size=1)
                            log_t = tmodel.joint.joint(f_t, g_t)[0, 0, 0, :].numpy()
                            log_a = tail.joint_step(f, g_an)
                            gmax = max(gmax, float(np.abs(g_an - g_t.numpy()).max()))
                            gden = max(gden, float(np.abs(g_t.numpy()).max()))
                            lmax = max(lmax, float(np.abs(log_a - log_t).max()))
                            lden = max(lden, float(np.abs(log_t).max()))
                            k = int(np.argmax(log_a))
                            tot += 1
                            eq += int(k == int(np.argmax(log_t)))
                            if k == tail.blank:
                                not_blank = False
                            else:
                                hs, cs = h2, c2
                                lt = k
                            symbols += 1
                    report["drift_first_clip"].append({
                        "step": step_num, "tag": tag, "frames": Tn,
                        "g_max_abs": round(gmax, 6), "g_ref_scale": round(gden, 6),
                        "logit_max_abs": round(lmax, 4), "logit_ref_scale": round(lden, 4),
                        "argmax_agree": f"{eq}/{tot}",
                    })
            row = {
                "clip": c["path"], "family": c.get("family", "?"), "audio_s": c["audio_s"],
                "ref_chunks": ref_chunks, "ref_tokens": len(ref_final),
                "ane_tokens": len(ane_toks),
                "joint_calls": joint_calls,
                "token_identity_pct": round(100.0 * sum(int(a == b) for a, b in
                                         zip(ane_toks, ref_final)) / max(len(ref_final), 1), 3),
                "exact_match": bool(ane_toks == ref_final),
            }
            if ane_toks != ref_final:
                row["ref_tail"] = ref_final[-8:]
                row["ane_tail"] = ane_toks[-8:]
                # first divergence position (the honest diagnostics row)
                for i, (a, b) in enumerate(zip(ane_toks, ref_final)):
                    if a != b:
                        row["first_diverge_idx"] = i
                        row["ane_at_diverge"] = ane_toks[max(0, i - 4):i + 4]
                        row["ref_at_diverge"] = ref_final[max(0, i - 4):i + 4]
                        break
            report["clips"].append(row)
            log(json.dumps({k: row[k] for k in ("clip", "token_identity_pct",
                                                "exact_match", "joint_calls")},
                           ensure_ascii=False))
    with open(PARITY, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log("parity rows →", PARITY)
    return report


# ------------------------------------------------------------------ timing
def preflight_or_die():
    r = subprocess.run([PREFLIGHT], capture_output=True, text=True)
    out = r.stdout + r.stderr
    prov = ""
    for line in out.splitlines():
        if line.startswith("PROVENANCE:"):
            prov = line
    if "✗" in out or "REFUSE" in out:
        log("preflight REFUSED — timing cell not run:\n", out.strip())
        raise SystemExit(3)
    return prov


def p50_mode(model, paths):
    """Posture A per-call p50 (dec+joint pair) + the Posture B tier sweep.
    FAIL-CLOSED quiet gate inside; rows appended to p633_p24_cells.jsonl with
    the PROVENANCE line."""
    import coremltools as ct
    prov = preflight_or_die()
    log("PROVENANCE:", prov)
    rows = []

    dec = ct.models.MLModel(paths["dec_step"], compute_units=ct.ComputeUnit.CPU_AND_NE)
    dec_out = [o.name for o in dec._spec.description.output]
    j1 = ct.models.MLModel(paths["joint_b1"], compute_units=ct.ComputeUnit.CPU_AND_NE)
    j1_out = [o.name for o in j1._spec.description.output]
    joint_b = {F: (ct.models.MLModel(paths[f"joint_b{F}"],
                                     compute_units=ct.ComputeUnit.CPU_AND_NE),
                   [o.name for o in ct.models.MLModel(paths[f"joint_b{F}"])._spec.description.output])
               for F in BATCHES}

    emb = np.load(EMBED_NPZ)["embed"]
    g0 = emb[5:6, :].reshape(1, 1, 640).astype(np.float32)
    h0 = np.zeros((2, 1, 640), dtype=np.float32)
    c0 = np.zeros((2, 1, 640), dtype=np.float32)
    enc1 = np.random.randn(1, 1, 1024).astype(np.float32) * 0.5

    def timeit(fn, warm, iters):
        for _ in range(warm):
            fn()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter_ns()
            fn()
            ts.append((time.perf_counter_ns() - t0) / 1e6)
        ts.sort()
        n = len(ts)
        return {"p50_ms": round(ts[n // 2], 4),
                "p95_ms": round(ts[min(n - 1, int(n * 0.95))], 4),
                "tail_support_p95": n - min(n - 1, int(n * 0.95))}

    # ---- Posture A: per-iteration = 1 dec call + 1 joint call (the loop body)
    def pair_only():
        d = dec.predict({"g_in": g0, "h": h0, "c": c0})
        g = np.array(d[dec_out[0]], dtype=np.float32).reshape(1, 1, 640)
        j1.predict({"enc": enc1, "pred": g})
    r = timeit(pair_only, 300, 3000)
    rows.append({"cell": "p50A_pair_calls", "posture": "A", **r,
                 "p02_runtime_ms": P02_RUNTIME_MS[1],
                 "excess_vs_runtime": round(r["p50_ms"] / P02_RUNTIME_MS[1], 3),
                 "provenance": prov, "quiet": True})
    log("p50A pair:", r)

    def full_iter():
        d = dec.predict({"g_in": g0, "h": h0, "c": c0})
        g = np.array(d[dec_out[0]], dtype=np.float32).reshape(1, 1, 640)
        outs = j1.predict({"enc": enc1, "pred": g})
        logits = np.array(outs[j1_out[0]], dtype=np.float32).reshape(-1)
        return int(np.argmax(logits))
    r = timeit(full_iter, 300, 3000)
    rows.append({"cell": "p50A_full_iter", "posture": "A", **r,
                 "provenance": prov, "quiet": True,
                 "note": "loop body incl. host argmax + dict building — the honest "
                         "host-loop per-call cost"})
    log("p50A full iter:", r)

    # ---- Posture B: the joint at batch F (P0.2 measurement vehicle)
    for F in BATCHES:
        m, outs = joint_b[F]
        encF = np.random.randn(1, F, 1024).astype(np.float32) * 0.5
        predF = np.random.randn(1, F, 640).astype(np.float32)
        if F == 1:
            encF, predF = enc1, np.random.randn(1, 1, 640).astype(np.float32)

        def call_b():
            m.predict({"enc": encF, "pred": predF})
        r = timeit(call_b, 100, 1000)
        rows.append({"cell": f"p50B_joint_b{F}", "posture": "B", **r,
                     "p02_runtime_ms": P02_RUNTIME_MS[F],
                     "excess_vs_runtime": round(r["p50_ms"] / P02_RUNTIME_MS[F], 3),
                     "provenance": prov, "quiet": True})
        log(f"p50B b{F}:", r)

    with open(CELLS, "a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log("timing rows →", CELLS)
    return rows


def per_audio_s(model, paths):
    """One real fleurs clip through ANE encode + the ANE decode loop; decode
    wall / audio seconds (posture A realistic). Quiet-gated, same as p50."""
    import coremltools as ct
    import soundfile as sf
    sys.path.insert(0, ASR)
    sys.path.insert(0, os.path.join(ASR, "iapp", "serve"))
    from p633_p1 import load_manifest
    import chinda_asr as ca
    prov = preflight_or_die()
    log("PROVENANCE:", prov)

    asr = ca.ChindaASR(model_path=NEMO_RAW, device="cpu", num_threads=1)
    asr.load()
    rep = asr._replicas[0]
    tmodel = rep.model
    cfg = tmodel.encoder.streaming_cfg
    enc_paths = {t: os.path.join(ANE_DIR, f"openthai_enc_{t}.mlpackage")
                 for t in ("first", "step")}
    enc_cml = {
        "first": ct.models.MLModel(enc_paths["first"], compute_units=ct.ComputeUnit.CPU_AND_NE),
        "step": ct.models.MLModel(enc_paths["step"], compute_units=ct.ComputeUnit.CPU_AND_NE),
    }
    enc_out = {}
    for tag, pkg in enc_paths.items():
        m = ct.models.MLModel(pkg)
        enc_out[tag] = [o.name for o in m._spec.description.output]
    tail = AneDecodeTail(model, paths)
    onehot = np.zeros((1, 1, int(tmodel.num_prompts)), dtype=np.float32)
    onehot[0, 0, int(tmodel._inference_prompt_index)] = 1.0

    man = load_manifest("fleurs-th")
    c = man["clips"][0]
    # warm-up: three dummy calls per graph so the measured pass pays no first-
    # call compile/staging (ANE program cache priming); `tail` is built above
    z = np.zeros((2, 1, 640), dtype=np.float32)
    g0 = tail.embed[5:6, :].reshape(1, 1, 640)
    for _ in range(3):
        tail.pred_step(5, z, z)
        tail.joint_step(np.random.randn(1, 1, 1024).astype(np.float32) * 0.5, g0)
    for tag in ("first", "step"):
        d = 49 if tag == "first" else 65
        enc_cml[tag].predict({
            "mel": np.random.randn(1, 128, d).astype(np.float32) * 0.1,
            "mel_len": np.array([d], dtype=np.int32),
            "cache_last_channel": np.zeros((24, 1, 56, 1024), dtype=np.float32),
            "cache_last_time": np.zeros((24, 1, 1024, 8), dtype=np.float32),
            "cache_last_channel_len": np.array([0 if tag == "first" else 56], dtype=np.int32),
            "prompt_onehot": onehot,
        })
    wave, sr = sf.read(os.path.join(ASR, c["path"]), dtype="float32")
    if wave.ndim > 1:
        wave = wave[:, 0]
    with torch.inference_mode():
        buf = rep.new_buffer((56, 6))
        buf.append_audio(wave, stream_id=-1)
        caches = None
        a_state, a_last = None, None
        tokens, joint_calls = [], 0
        t_enc = t_dec = 0.0
        for step_num, (chunk, lengths) in enumerate(iter(buf)):
            if caches is None:
                caches = tmodel.encoder.get_initial_cache_state(batch_size=1)
            c1, c2, c3 = caches
            ka = buf.is_buffer_empty()
            if not ka:
                tag = "first" if step_num == 0 else "step"
                ins = {
                    "mel": chunk.numpy().astype(np.float32),
                    "mel_len": lengths.numpy().astype(np.int32),
                    "cache_last_channel": c1.numpy().astype(np.float32),
                    "cache_last_time": c2.numpy().astype(np.float32),
                    "cache_last_channel_len": c3.numpy().astype(np.int32),
                    "prompt_onehot": onehot,
                }
                t0 = time.perf_counter_ns()
                outs = enc_cml[tag].predict(ins)
                t_enc += (time.perf_counter_ns() - t0) / 1e6
                vals = [outs[n] for n in enc_out[tag]]
                enc_np = np.array(vals[0], dtype=np.float32).transpose(0, 2, 1)
                enc_len = int(np.array(vals[1]).reshape(-1)[0])
                caches = (torch.from_numpy(np.array(vals[2], dtype=np.float32)),
                          torch.from_numpy(np.array(vals[3], dtype=np.float32)),
                          torch.from_numpy(np.array(vals[4]).astype(np.int64)))
            else:
                enc_t, len_t, c1t, c2t, c3t = tmodel.encoder.cache_aware_stream_step(
                    processed_signal=chunk, processed_signal_length=lengths,
                    cache_last_channel=c1, cache_last_time=c2,
                    cache_last_channel_len=c3, keep_all_outputs=True,
                    drop_extra_pre_encoded=cfg.drop_extra_pre_encoded)
                enc_np = tmodel._apply_prompt_to_encoded(enc_t).transpose(1, 2).numpy().astype(np.float32)
                enc_len = int(len_t[0])
                caches = (c1t, c2t, c3t)
            t0 = time.perf_counter_ns()
            toks, a_state, a_last, nc = tail.decode(enc_np, enc_len,
                                                    state=a_state, last_token=a_last)
            t_dec += (time.perf_counter_ns() - t0) / 1e6
            tokens.extend(toks)
            joint_calls += nc
    audio_s = c["audio_s"]
    row = {"cell": "per_audio_s", "clip": c["path"], "audio_s": audio_s,
           "encode_ms": round(t_enc, 2), "decode_ms": round(t_dec, 2),
           "joint_calls": joint_calls,
           "decode_ms_per_audio_s": round(t_dec / audio_s, 2),
           "enc_rtfx_decode_tail": round(audio_s / (t_enc / 1000.0), 2),
           "provenance": prov, "quiet": True}
    log(json.dumps(row, ensure_ascii=False))
    with open(CELLS, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


# ------------------------------------------------------------------ main
JOINT_PATHS = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all",
                    choices=["all", "convert", "compile", "census", "parity",
                             "p50", "per_audio_s"])
    args = ap.parse_args()

    os.makedirs(ANE_DIR, exist_ok=True)
    model = load_model()

    if args.mode in ("all", "convert"):
        paths = convert_all(model)
    else:
        paths = {t: os.path.join(ANE_DIR, f"openthai_{TAG}{t}.mlpackage")
                 for t in ["dec_step"] + [f"joint_b{F}" for F in BATCHES]}
    JOINT_PATHS.update(paths)

    if args.mode in ("all", "compile"):
        for pkg in paths.values():
            if os.path.exists(pkg) and not os.path.exists(pkg.replace(".mlpackage", ".mlmodelc")):
                compile_pkg(pkg)
        log("compiled:", ", ".join(os.path.basename(p) for p in paths.values()))

    if args.mode in ("all", "census"):
        census(paths)

    if args.mode in ("all", "parity"):
        enc_paths = {t: os.path.join(ANE_DIR, f"openthai_enc_{t}.mlpackage")
                     for t in ("first", "step")}
        parity(model, enc_paths)

    if args.mode == "p50":
        p50_mode(model, paths)
    if args.mode == "per_audio_s":
        per_audio_s(model, paths)


if __name__ == "__main__":
    main()
