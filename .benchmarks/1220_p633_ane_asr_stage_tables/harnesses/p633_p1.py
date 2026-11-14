#!/usr/bin/env python3
"""Plan 633 P1 harness — CPU cells for OpenThai2.0-ASR (riir-infer plan 633).

Subcommands:
  manifests            build FLEURS-th + TVSpeech seeded slice manifests (blake3-digested)
  p12 [--threads N] [--precision fp|int8] [--max-clips N]   ONNX CPU EP cell (P1.2)
  p11 [--max-clips N]                                       NeMo PyTorch fp32 reference (P1.1)
  p13 [--max-clips N]                                       threading sweep 1/4/N on p12 fp

Decode recipe (iapp/OpenThai2.0-ASR-Streaming-ONNX README):
  mel: 128 log-mel bands, 25ms/10ms, pre-emph 0.97, no norm (NeMo preprocessor provides it)
  encoder: audio_signal [1,128,T] + length [1] + prompt_index [1] (32=th) -> encoded [1,T/8,1024]
  joint: encoder_outputs [1,1024,1] + targets [1,1] + target_length [1] + states [2,1,640]
         -> scores [1,14989] (blank = 14988) + next states; greedy, max_symbols_per_step=2,
         blank bias -1.0 (-2.5 for clips < 2s).

Cell results append to .raw/asr/p633_cells.jsonl (the bench record cites this file).
"""
import argparse, json, os, sys, time
import numpy as np

ASR = os.path.dirname(os.path.abspath(__file__))
ONNX_DIR = os.path.join(ASR, "onnx")
FIXTURES = os.path.join(ASR, "fixtures")
CELLS = os.path.join(ASR, "p633_cells.jsonl")
SEED = 633
SLICE_N = 100
PROMPT_TH = 32
BLANK_ID = 14988
MAX_SYMBOLS_PER_STEP = 2

def log(*a):
    print("[p633]", *a, flush=True)

def append_cell(row):
    with open(CELLS, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

# ---------------------------------------------------------------- manifests

def load_manifest(family):
    p = os.path.join(FIXTURES, f"{family}_slice.json")
    with open(p) as f:
        return json.load(f)

def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def audio_info(path):
    import soundfile as sf
    info = sf.info(path)
    return info.frames / info.samplerate, info.samplerate

def build_fleurs_manifest():
    """FLEURS-th test split: refs from iapp make_refs.py (validated vs their card CER)
    joined with our downloaded audio. Seeded 100-clip slice, seed 633."""
    import tarfile
    base = os.path.join(ASR, "fleurs", "data", "th_th")
    tsv = os.path.join(base, "test.tsv")
    tar = os.path.join(base, "audio", "test.tar.gz")
    refs_path = os.path.join(ASR, "refs", "fleurs.jsonl")
    if not (os.path.exists(tsv) and os.path.exists(tar) and os.path.exists(refs_path)):
        sys.exit("missing fleurs tsv/tar/refs (fetch + make_refs first)")
    audio_dir = os.path.join(base, "audio", "test")
    if not os.path.isdir(audio_dir) or not any(f.endswith(".wav") for f in os.listdir(audio_dir)):
        with tarfile.open(tar) as tf:
            tf.extractall(os.path.join(base, "audio"))
    refs = {}
    with open(refs_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            refs[str(r["id"])] = r
    log(f"fleurs refs: {len(refs)}; audio dir: {len(os.listdir(audio_dir))} files")
    # index audio files by stem AND full filename (ref ids carry .wav)
    audio = {}
    for f in os.listdir(audio_dir):
        stem, ext = os.path.splitext(f)
        if ext in (".wav", ".flac", ".mp3"):
            audio[stem] = os.path.join(audio_dir, f)
            audio[f] = os.path.join(audio_dir, f)
    matched_ids = [i for i in refs if i in audio]
    if len(matched_ids) != len(refs):
        sys.exit(f"fleurs audio join incomplete: {len(matched_ids)}/{len(refs)}")
    rng = np.random.default_rng(SEED)
    pick = sorted(rng.choice(len(matched_ids), size=SLICE_N, replace=False).tolist())
    slice_ids = [matched_ids[i] for i in pick]
    clips = []
    for cid in slice_ids:
        fn = audio[cid]
        dur, sr = audio_info(fn)
        clips.append({
            "id": cid, "path": os.path.relpath(fn, ASR),
            "audio_s": round(dur, 4), "sr": sr, "sha256": sha256_file(fn),
            "ref": refs[cid]["text"], "cluster": refs[cid].get("cluster", cid),
            "family": "fleurs-th",
        })
    man = {
        "family": "fleurs-th", "seed": SEED, "n": len(clips),
        "source": "google/fleurs th_th test (CC-BY-4.0); refs via iapp make_refs.py "
                  "(validated: their 560ms hypotheses score 8.47% on these refs = the card)",
        "split_rows": len(refs), "clips": clips,
    }
    with open(os.path.join(FIXTURES, "fleurs-th_slice.json"), "w") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    log(f"fleurs slice: {len(clips)} clips")

def build_tvspeech_manifest():
    """TVSpeech: refs from iapp make_refs.py joined with audio extracted from the
    test parquet (audio bytes embedded). Seeded 100-clip slice, seed 633."""
    import pyarrow.parquet as pq
    base = os.path.join(ASR, "tvspeech")
    parq = os.path.join(base, "data", "test-00000-of-00001.parquet")
    refs_path = os.path.join(ASR, "refs", "tvspeech.jsonl")
    if not (os.path.exists(parq) and os.path.exists(refs_path)):
        sys.exit("missing tvspeech parquet/refs")
    refs = {}
    with open(refs_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            refs[str(r["id"])] = r
    pf = pq.ParquetFile(parq)
    log(f"tvspeech parquet rows: {pf.metadata.num_rows}; refs: {len(refs)}")
    if pf.metadata.num_rows != len(refs):
        sys.exit("tvspeech row/refs count mismatch")
    rng = np.random.default_rng(SEED)
    pick = sorted(rng.choice(pf.metadata.num_rows, size=SLICE_N, replace=False).tolist())
    audio_dir = os.path.join(FIXTURES, "tvspeech_audio")
    os.makedirs(audio_dir, exist_ok=True)
    clips = []
    for batch_idx, batch in enumerate(pf.iter_batches(batch_size=1024)):
        rows = batch.to_pylist()
        for local_i, r in enumerate(rows):
            rid = str(batch_idx * 1024 + local_i)
            if rid not in {str(p) for p in pick}:
                continue
            a = r["audio"]
            ext = os.path.splitext(a.get("path") or "")[1] or ".wav"
            fn = os.path.join(audio_dir, f"{rid}{ext}")
            if not os.path.exists(fn):
                with open(fn, "wb") as f:
                    f.write(a["bytes"])
            dur, sr = audio_info(fn)
            clips.append({
                "id": rid, "path": os.path.relpath(fn, ASR),
                "audio_s": round(dur, 4), "sr": sr, "sha256": sha256_file(fn),
                "ref": refs[rid]["text"], "cluster": refs[rid].get("cluster", rid),
                "family": "tvspeech",
            })
    if len(clips) != SLICE_N:
        sys.exit(f"tvspeech slice incomplete: {len(clips)}/{SLICE_N}")
    man = {
        "family": "tvspeech", "seed": SEED, "n": len(clips),
        "source": "typhoon-ai/TVSpeech test (CC-BY); refs via iapp make_refs.py "
                  "(validated: their 560ms hypotheses score 12.41% on these refs = the card)",
        "split_rows": len(refs), "clips": clips,
    }
    with open(os.path.join(FIXTURES, "tvspeech_slice.json"), "w") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    log(f"tvspeech slice: {len(clips)} clips")

def cmd_manifests(_args):
    build_fleurs_manifest()
    build_tvspeech_manifest()

# ---------------------------------------------------------------- normalization + CER
# EXACTLY the card's chain: iapp benchmark/score.py prep() — chinda_norm.normalize(th)
# -> num_norm.normalize_numbers -> remove every space; CER = editdistance edits /
# reference characters over the WHOLE SET (their convention). Their modules are
# imported from the fetched iapp tree (OpenMDW-1.1, attribution in the bench record).

sys.path.insert(0, os.path.join(ASR, "iapp", "benchmark"))
from chinda_norm import normalize as _chinda_normalize  # noqa: E402
from num_norm import normalize_numbers as _num_normalize  # noqa: E402
import editdistance as _editdistance  # noqa: E402

def prep(t):
    return _num_normalize(_chinda_normalize(t or "", "th")).replace(" ", "")

def set_cer(refs, hyps):
    """Card-convention set-level CER: (total edits) / (total reference chars)."""
    tot_e = tot_r = 0
    for r, h in zip(refs, hyps):
        rn = prep(r)
        tot_e += _editdistance.eval(rn, prep(h))
        tot_r += len(rn)
    return 100.0 * tot_e / tot_r if tot_r else None

# ---------------------------------------------------------------- pause-split (verbatim port)
# iapp chinda_realtime.py Endpointer + split_at_pauses (OpenMDW-1.1) — the card's
# documented dictation posture: "Decode a long recording in pieces cut at pauses.
# Decoded as one long stream, the model drops words that follow a pause." Their
# rules: 20 ms frames, adaptive dB thresholds (10 s history), 3-frame speech start,
# 700 ms end-silence close, 30 s max utterance.

import collections as _collections
_FRAME = 320            # 20 ms @ 16 kHz
_PREROLL_S = 0.3
_START_FRAMES = 3
_HIST_FRAMES = 500      # 10 s of frame levels

class _Endpointer:
    def __init__(self):
        self.hist = _collections.deque(maxlen=_HIST_FRAMES)

    def is_speech(self, frame):
        db = 20 * np.log10(np.sqrt(np.mean(frame.astype(np.float64) ** 2) + 1e-10) + 1e-10)
        self.hist.append(db)
        h = np.fromiter(self.hist, dtype=np.float64)
        loud = np.percentile(h, 95) if len(h) >= 25 else max(db, -30.0)
        quiet = np.percentile(h, 10) if len(h) >= 25 else -90.0
        return db > max(loud - 30.0, quiet + 8.0, -70.0)

def split_at_pauses(audio, end_silence_ms=700, max_utterance_s=30.0):
    ep = _Endpointer()
    end_frames = max(5, int(end_silence_ms) // 20)
    max_utt = int(max_utterance_s * 16000)
    pre_max = int(_PREROLL_S * 16000 / _FRAME) + _START_FRAMES
    pre = run = 0
    start = 0
    open_ = False
    out = []
    for i in range(audio.shape[0] // _FRAME):
        n = (i + 1) * _FRAME
        sp = ep.is_speech(audio[i * _FRAME:n])
        if not open_:
            pre = min(pre + 1, pre_max)
            run = run + 1 if sp else 0
            if run >= _START_FRAMES:
                start, open_, run = max(0, n - _FRAME * pre), True, 0
            continue
        run = 0 if sp else run + 1
        length = n - start
        if run >= end_frames or (length >= max_utt and not sp) or length >= max_utt + 10 * 16000:
            out.append((start, n))
            open_, pre, run = False, 0, 0
    if open_:
        out.append((start, audio.shape[0]))
    return out

def greedy_decode_long(enc_s, joint_s, wave, sr, blank_bias=-1.0, short_bias=-2.5):
    """Pause-split then decode each piece; texts joined with one space. Timing covers
    the whole forward (splits + mel + encoder + joint loop) — the pipeline's cost."""
    t0 = time.perf_counter()
    if sr != 16000:
        import torchaudio
        w16 = torchaudio.functional.resample(torch.from_numpy(wave), sr, 16000).numpy()
    else:
        w16 = wave
    cuts = split_at_pauses(w16)
    if not cuts:
        cuts = [(0, w16.shape[0])]
    hyp_all = []
    for s, e in cuts:
        piece = w16[s:e]
        dur_piece = (e - s) / 16000.0
        mel = mel_features(piece, 16000)
        bias = short_bias if dur_piece < 2.0 else blank_bias
        ids, _ = greedy_decode(enc_s, joint_s, mel, bias)
        hyp_all.extend(ids)
        hyp_all.append(None)  # piece boundary -> space
    return hyp_all, time.perf_counter() - t0

# ---------------------------------------------------------------- ONNX cells (P1.2)

def load_onnx_sessions(precision, threads):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    provs = ["CPUExecutionProvider"]
    if precision == "fp":
        enc = os.path.join(ONNX_DIR, "encoder.onnx")
    else:
        enc = os.path.join(ONNX_DIR, "int8", "encoder.int8.onnx")
    joint = os.path.join(ONNX_DIR, "decoder_joint.onnx")
    enc_s = ort.InferenceSession(enc, sess_options=so, providers=provs)
    joint_s = ort.InferenceSession(joint, sess_options=so, providers=provs)
    return enc_s, joint_s

_PREPROC = None

def nemo_preprocessor():
    """The .nemo's own AudioToMelSpectrogramPreprocessor (dither 0 for determinism —
    the shipped cfg has 1e-5; the delta is recorded in the bench record)."""
    global _PREPROC
    if _PREPROC is None:
        from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
        _PREPROC = AudioToMelSpectrogramPreprocessor(sample_rate=16000, normalize="NA",
            window_size=0.025, window_stride=0.01, window="hann", features=128,
            n_fft=512, log=True, dither=0.0, pad_to=0)
        _PREPROC.eval()
    return _PREPROC

def mel_features(wave, sr):
    """Model-exact features: NeMo's own preprocessor (input parity with the P1.1
    NeMo cell — the runtime is the only variable between cells)."""
    import torch
    wf = torch.from_numpy(np.ascontiguousarray(wave)).float()
    if sr != 16000:
        import torchaudio
        wf = torchaudio.functional.resample(wf, sr, 16000)
    with torch.inference_mode():
        feats, _ = nemo_preprocessor()(input_signal=wf.unsqueeze(0),
                                       length=torch.tensor([wf.shape[0]]))
    return feats.numpy().astype(np.float32)  # [1,128,T]

def mel_features_pure(wave, sr):
    """Pure-torchaudio approximation (the production/ChindaBot frontend shape):
    preemph 0.97, torch.stft (n_fft 512, hop 160, win 400 periodic hann, center,
    reflect), power spec, librosa slaney-scale fbank, log(mel + 2**-24). Recorded
    parity vs NeMo's own preprocessor: body ≈ ≤0.2 (≤2.0 on loud-onset frame 0),
    NeMo's trailing feature-pad frame differs by construction."""
    import torch
    import librosa
    wf = torch.from_numpy(np.ascontiguousarray(wave)).float()
    if sr != 16000:
        import torchaudio
        wf = torchaudio.functional.resample(wf, sr, 16000)
    wf = wf.unsqueeze(0)  # [1, T]
    preemph = 0.97
    wf = torch.cat((wf[:, :1], wf[:, 1:] - preemph * wf[:, :-1]), dim=1)
    n_fft = 512; win_length = 400; hop = 160
    window = torch.hann_window(win_length, periodic=True)
    spec = torch.stft(wf, n_fft=n_fft, hop_length=hop, win_length=win_length,
                      window=window, center=True, pad_mode="reflect",
                      normalized=False, onesided=True, return_complex=True)
    spec = spec.abs().pow(2)  # [1, 257, T]
    fb = librosa.filters.mel(sr=16000, n_fft=n_fft, n_mels=128,
                             fmin=0.0, fmax=8000.0, norm="slaney")
    fb = torch.from_numpy(fb).float()
    mel = torch.matmul(fb, spec[0])  # [128, T]
    mel = torch.log(mel + 2 ** -24)
    return mel.unsqueeze(0).numpy().astype(np.float32)  # [1,128,T]

def greedy_decode(enc_s, joint_s, mel, blank_bias=-1.0):
    """Full-utterance encoder once, then the per-frame greedy RNN-T loop.
    Returns (hypothesis token ids, forward_wall_s)."""
    t0 = time.perf_counter()
    enc_in = {i.name: None for i in enc_s.get_inputs()}
    n_frames = mel.shape[2]
    enc_feed = {"audio_signal": mel, "length": np.array([n_frames], dtype=np.int64),
                "prompt_index": np.array([PROMPT_TH], dtype=np.int64)}
    for name in list(enc_feed):
        if name not in enc_in:
            enc_feed.pop(name)
    missing = [i.name for i in enc_s.get_inputs() if i.name not in enc_feed]
    if missing:
        sys.exit(f"encoder inputs unmatched: {missing} (have {list(enc_feed)})")
    encoded = enc_s.run(None, enc_feed)[0]  # [1, T', 1024]
    joint_in_names = [i.name for i in joint_s.get_inputs()]
    states = [np.zeros((2, 1, 640), dtype=np.float32) for _ in range(2)]  # LSTM h, c
    last_tok = BLANK_ID  # sos = blank (NeMo RNN-T convention; asserted vs chinda_asr.py)
    hyp = []
    for t in range(encoded.shape[1]):
        enc_t = np.ascontiguousarray(encoded[:, t:t + 1, :].transpose(0, 2, 1))  # [1,1024,1]
        y = last_tok
        emitted = 0
        while emitted < MAX_SYMBOLS_PER_STEP:
            feed = {"encoder_outputs": enc_t,
                    "targets": np.array([[y]], dtype=np.int32),
                    "target_length": np.array([1], dtype=np.int32),
                    "input_states_1": states[0], "input_states_2": states[1]}
            feed = {k: v for k, v in feed.items() if k in joint_in_names}
            outs = joint_s.run(None, feed)
            scores = outs[0]  # [1,1,14989]
            states = [outs[-2], outs[-1]]  # advance INSIDE the loop — states consume y
            scores = scores.reshape(-1).astype(np.float64)
            scores[BLANK_ID] += blank_bias
            a = int(np.argmax(scores))
            if a == BLANK_ID:
                break
            hyp.append(a)
            last_tok = a
            y = a  # the within-frame re-run must feed the token just emitted
            emitted += 1
    return hyp, time.perf_counter() - t0

def tokens_to_text(ids):
    toks = {}
    with open(os.path.join(ONNX_DIR, "tokens.txt"), encoding="utf-8") as f:
        for i, line in enumerate(f):
            piece = line.split("\t")[0].strip("\n")
            toks[i] = piece
    return "".join(toks.get(i, "") for i in ids).replace("▁", " ")

def run_cell(precision, threads, max_clips, families=("fleurs-th", "tvspeech")):
    enc_s, joint_s = load_onnx_sessions(precision, threads)
    for family in families:
        man = load_manifest(family)
        clips = man["clips"][:max_clips] if max_clips else man["clips"]
        tot_audio = tot_fwd = tot_mel = 0.0
        refs_n, hyps_n = [], []
        wall0 = time.perf_counter()
        for c in clips:
            path = os.path.join(ASR, c["path"])
            import soundfile as sf
            wave, sr = sf.read(path, dtype="float32", always_2d=False)
            if wave.ndim > 1:
                wave = wave[:, 0]
            dur, _ = audio_info(path)
            hyp, fwd = greedy_decode_long(enc_s, joint_s, wave, sr)
            tot_fwd += fwd
            hyp_text = tokens_to_text([i for i in hyp if i is not None])
            refs_n.append(c["ref"])
            hyps_n.append(hyp_text)
            tot_audio += dur
        wall_all = time.perf_counter() - wall0
        n = len(clips)
        c = set_cer(refs_n, hyps_n)
        row = {
            "cell": "P1.2", "precision": precision, "threads": threads, "family": family,
            "n": n, "audio_s": round(tot_audio, 2),
            "mel_wall_s": round(tot_mel, 3), "forward_wall_s": round(tot_fwd, 3),
            "wall_s": round(wall_all, 3),
            "rtfx_forward": round(tot_audio / tot_fwd, 2),
            "rtfx_wall": round(tot_audio / wall_all, 2),
            "cer_set": round(c, 2) if c is not None else None,
            "note": "piecewise (pause-split) decode per the card's dictation posture; wall = audio IO + "
                    "forward; forward = splits + mel + encoder+joint over all pieces; "
                    "cer_set = card convention (set-level edits / ref chars)",
        }
        append_cell(row)
        log(json.dumps(row, ensure_ascii=False))

def cmd_p12(args):
    run_cell(args.precision, args.threads, args.max_clips)

def cmd_p13(args):
    for th in (1, 4, os.cpu_count() or 30):
        run_cell("fp", th, args.max_clips)

# ---------------------------------------------------------------- frontend parity

def cmd_frontend_parity(args):
    """Informational: the pure-torchaudio production-shaped frontend vs NeMo's own
    preprocessor (dither 0 both). The ONNX cells feed NeMo's preprocessor directly,
    so this residual does not gate any cell — it documents the production gap."""
    import soundfile as sf, torch
    pre = nemo_preprocessor()
    man = load_manifest("fleurs-th")
    worst = 0.0
    for c in man["clips"][:args.max_clips or 3]:
        wave, sr = sf.read(os.path.join(ASR, c["path"]), dtype="float32")
        if wave.ndim > 1:
            wave = wave[:, 0]
        sig = torch.from_numpy(wave).unsqueeze(0)
        with torch.inference_mode():
            feats, _ = pre(input_signal=sig, length=torch.tensor([wave.shape[0]]))
        mine = torch.from_numpy(mel_features_pure(wave, sr))
        tmin = min(feats.shape[2], mine.shape[2])
        # NeMo's trailing feature-pad frame differs by construction — compare [0, T-1).
        t_cmp = tmin - 1
        d = (feats[:, :, :t_cmp] - mine[:, :, :t_cmp]).abs().max().item()
        worst = max(worst, d)
        log(f"clip {c['id']}: nemo T={feats.shape[2]} pure T={mine.shape[2]} body_max_abs_diff={d:.3e}")
    row = {"cell": "frontend-parity", "clips": args.max_clips or 3,
           "body_max_abs_diff": worst,
           "verdict": "informational-only (cells feed NeMo's preprocessor directly)"}
    append_cell(row)
    log(json.dumps(row))

# ---------------------------------------------------------------- NeMo reference (P1.1)

def cmd_p11(args):
    sys.exit("P1.1 lands after make_refs.py is read (streaming recipe mirror) + nemo download completes")

def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("manifests")
    p = sp.add_parser("frontend-parity")
    p.add_argument("--max-clips", type=int, default=3)
    p = sp.add_parser("p12")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--precision", default="fp", choices=["fp", "int8"])
    p.add_argument("--max-clips", type=int, default=0)
    p = sp.add_parser("p11")
    p.add_argument("--max-clips", type=int, default=0)
    p = sp.add_parser("p13")
    p.add_argument("--max-clips", type=int, default=0)
    args = ap.parse_args()
    {"manifests": cmd_manifests, "frontend-parity": cmd_frontend_parity,
     "p12": cmd_p12, "p11": cmd_p11, "p13": cmd_p13}[args.cmd](args)

if __name__ == "__main__":
    main()
