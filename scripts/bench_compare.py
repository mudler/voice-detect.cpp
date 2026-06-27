#!/usr/bin/env python3
"""Comparative latency benchmark: voice-detect.cpp (ggml) vs the reference stack.

For each speaker encoder (ECAPA, WeSpeaker, ERes2Net, CAM++) and the wav2vec2
analyze heads (emotion, age/gender) this measures end-to-end per-clip latency
two ways and prints a comparison table (ggml ms/clip | reference ms/clip |
speedup):

* OUR ENGINE  - the ``voicedetect-cli bench`` subcommand, which times the real
  ``embed_path`` / ``analyze_path_json`` C++ entry point (WAV decode + Kaldi
  FBank front end + encoder graph + L2-normalize, or the analyze head), warmup
  excluded.
* REFERENCE   - the SAME model via its native Python path on CPU:
    - ECAPA            -> SpeechBrain ``embedding_model`` (torch) on Kaldi FBank
    - WeSpeaker/ERes2Net/CAM++ -> onnxruntime CPU on Kaldi FBank
    - emotion          -> HF transformers ``superb/wav2vec2-base-superb-er``
    - age/gender       -> the audeering AgeGenderModel (transformers)
  warmup excluded, same clip, same N.

Methodology (honest): both sides run SINGLE-THREADED by default
(``--threads 1``; torch.set_num_threads(1); onnxruntime intra/inter-op = 1) for
an apples-to-apples core-for-core comparison. The reference loads the audio once
and times feature-extraction + forward per iteration; the ggml side re-decodes
the (tiny) WAV every call inside ``embed_path``, a sub-millisecond overhead on a
3 s clip. The ggml FBank uses a straightforward (non-FFT) mel filterbank, so part
of any ggml deficit on the speaker encoders is front-end, not the encoder graph.
Numbers are wall-clock means over N timed passes after a warmup pass; absolute
values are machine-specific (see the machine block in the emitted report).

Usage:
    .venv/bin/python scripts/bench_compare.py \
        --audio tests/fixtures/clip_a.wav --n 20 --threads 1 \
        --out benchmarks/RESULTS.md
    # add --device cuda (and a CUDA-enabled CLI build) to time the ggml side on GPU.

Exit codes: 0 = report written (per-model failures are recorded as "n/a", not
fatal). Models whose reference deps/weights are unavailable are clearly marked.
"""
import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone


# Reference-side thread count, mirrored from --threads so both engines are
# compared core-for-core at the SAME thread budget (set in main()).
REF_THREADS = 1


def _first_existing(paths):
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None


def discover_models(root):
    """Return the model registry, resolving ggml GGUF + reference paths under root."""
    m = lambda *p: os.path.join(root, *p)
    ecapa = os.environ.get("VD_BENCH_ECAPA_GGUF") or _first_existing([
        m("models", "ecapa_voxceleb_f32.gguf"),
        "/tmp/ecapa_f32.gguf",
        "/tmp/vd-publish/ecapa-tdnn-voxceleb.gguf",
    ])
    return [
        dict(name="ECAPA-TDNN", family="speaker", mode="embed",
             ggml=ecapa, ref="speechbrain",
             ref_model="speechbrain/spkrec-ecapa-voxceleb"),
        dict(name="WeSpeaker ResNet34", family="speaker", mode="embed",
             ggml=m("models", "wespeaker_resnet34_f32.gguf"), ref="onnx",
             onnx=m("models", "wespeaker_voxceleb_resnet34.onnx")),
        dict(name="ERes2Net", family="speaker", mode="embed",
             ggml=m("models", "eres2net_base_200k_f32.gguf"), ref="onnx",
             onnx=m("models", "3dspeaker_eres2net_base_200k.onnx")),
        dict(name="CAM++", family="speaker", mode="embed",
             ggml=m("models", "campplus_zh-cn_f32.gguf"), ref="onnx",
             onnx=m("models", "3dspeaker_campplus_zh-cn_16k.onnx")),
        dict(name="wav2vec2 emotion", family="analyze", mode="analyze",
             ggml=m("models", "wav2vec2_analyze_f32.gguf"), ref="hf_emotion",
             ref_model="superb/wav2vec2-base-superb-er"),
        dict(name="wav2vec2 age/gender", family="analyze", mode="analyze",
             ggml=m("models", "age_gender_audeering_f32.gguf"), ref="hf_age_gender",
             ref_model="audeering/wav2vec2-large-robust-24-ft-age-gender"),
    ]


# ---------------------------------------------------------------------------
# ggml side: drive the CLI bench subcommand.
# ---------------------------------------------------------------------------
def bench_ggml(cli, model, audio, mode, n, threads, device):
    if not model or not os.path.isfile(model):
        return None, "gguf missing"
    env = dict(os.environ)
    if device:
        env["VOICEDETECT_DEVICE"] = device  # "cpu" or a ggml device name e.g. "CUDA0"
    cmd = [cli, "bench", "--model", model, "--input", audio,
           "--mode", mode, "--n", str(n), "--threads", str(threads)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
    except Exception as e:  # noqa: BLE001
        return None, f"cli error: {e}"
    if out.returncode != 0:
        tail = (out.stderr or out.stdout).strip().splitlines()[-1:] or [""]
        return None, f"cli rc={out.returncode}: {tail[0]}"
    mobj = re.search(r"(?:embed|analyze):\s*([0-9.]+)\s*ms/clip", out.stdout)
    if not mobj:
        return None, "no timing line"
    return float(mobj.group(1)), None


# ---------------------------------------------------------------------------
# Reference side: native Python forward, CPU, single-thread, warmup excluded.
# ---------------------------------------------------------------------------
def _load_audio_16k(path):
    import numpy as np
    import soundfile as sf
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != 16000:
        n = int(round(len(audio) * 16000 / sr))
        audio = np.interp(np.linspace(0, len(audio), n, endpoint=False),
                          np.arange(len(audio)), audio).astype(np.float32)
    return audio


def _kaldi_fbank_cmn(audio):
    import torch
    import torchaudio.compliance.kaldi as kaldi
    wav = torch.from_numpy(audio).unsqueeze(0)
    fb = kaldi.fbank(wav, sample_frequency=16000, num_mel_bins=80,
                     frame_length=25.0, frame_shift=10.0, dither=0.0)
    return fb - fb.mean(dim=0, keepdim=True)  # CMN, [T, 80]


def _timeit(fwd, n):
    fwd()  # warmup (excluded)
    t0 = time.perf_counter()
    for _ in range(n):
        fwd()
    return (time.perf_counter() - t0) * 1000.0 / n


def bench_ref_speechbrain(spec, audio_path, n):
    import torch
    torch.set_num_threads(REF_THREADS)
    from speechbrain.inference.speaker import EncoderClassifier
    clf = EncoderClassifier.from_hparams(source=spec["ref_model"],
                                         savedir="./pretrained_models")
    clf.mods.eval()
    em = clf.mods.embedding_model
    audio = _load_audio_16k(audio_path)

    def fwd():
        feats = _kaldi_fbank_cmn(audio).unsqueeze(0)  # [1, T, 80]
        with torch.no_grad():
            emb = em(feats).squeeze()
        float(torch.linalg.vector_norm(emb))  # match L2-norm work
    return _timeit(fwd, n)


def bench_ref_onnx(spec, audio_path, n):
    import numpy as np
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = REF_THREADS
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(spec["onnx"], sess_options=so,
                                providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    audio = _load_audio_16k(audio_path)

    def fwd():
        feed = _kaldi_fbank_cmn(audio).numpy().astype(np.float32)[np.newaxis, :, :]
        out = sess.run(None, {in_name: feed})
        emb = np.asarray(out[0]).reshape(-1).astype(np.float32)
        emb / (np.linalg.norm(emb) or 1.0)
    return _timeit(fwd, n)


def bench_ref_hf_emotion(spec, audio_path, n):
    import torch
    torch.set_num_threads(REF_THREADS)
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    model = AutoModelForAudioClassification.from_pretrained(spec["ref_model"]).eval()
    fe = AutoFeatureExtractor.from_pretrained(spec["ref_model"])
    audio = _load_audio_16k(audio_path)

    def fwd():
        inputs = fe(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            model(inputs["input_values"])
    return _timeit(fwd, n)


def bench_ref_hf_age_gender(spec, audio_path, n):
    import torch
    import torch.nn as nn
    torch.set_num_threads(REF_THREADS)
    from transformers import (AutoConfig, AutoFeatureExtractor,
                              Wav2Vec2Model, Wav2Vec2PreTrainedModel)

    class ModelHead(nn.Module):
        def __init__(self, config, num_labels):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, num_labels)

        def forward(self, x):
            x = self.dropout(x)
            x = torch.tanh(self.dense(x))
            return self.out_proj(self.dropout(x))

    class AgeGenderModel(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.wav2vec2 = Wav2Vec2Model(config)
            self.age = ModelHead(config, 1)
            self.gender = ModelHead(config, 3)

        def forward(self, input_values):
            outputs = self.wav2vec2(input_values)
            pooled = torch.mean(outputs[0], dim=1)
            return self.age(pooled), self.gender(pooled)

    cfg = AutoConfig.from_pretrained(spec["ref_model"])
    fe = AutoFeatureExtractor.from_pretrained(spec["ref_model"])
    from huggingface_hub import hf_hub_download
    sd = None
    for fn in ("model.safetensors", "pytorch_model.bin"):
        try:
            path = hf_hub_download(spec["ref_model"], fn)
        except Exception:  # noqa: BLE001
            continue
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file
            sd = load_file(path)
        else:
            sd = torch.load(path, map_location="cpu")
        break
    if sd is None:
        raise FileNotFoundError("no weights for " + spec["ref_model"])
    model = AgeGenderModel(cfg)
    model.load_state_dict(sd, strict=True)
    model.eval()
    audio = _load_audio_16k(audio_path)

    def fwd():
        inputs = fe(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            model(inputs["input_values"])
    return _timeit(fwd, n)


REF_FN = {
    "speechbrain": bench_ref_speechbrain,
    "onnx": bench_ref_onnx,
    "hf_emotion": bench_ref_hf_emotion,
    "hf_age_gender": bench_ref_hf_age_gender,
}


def bench_ref(spec, audio_path, n):
    try:
        return REF_FN[spec["ref"]](spec, audio_path, n), None
    except SystemExit as e:  # gen_baseline-style dep guards
        return None, f"unavailable ({e})"
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
def machine_info(threads):
    import numpy as np
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    vers = {"numpy": np.__version__}
    for mod in ("torch", "onnxruntime", "transformers"):
        try:
            vers[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            vers[mod] = "n/a"
    return cpu, vers


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", default="tests/fixtures/clip_a.wav")
    ap.add_argument("--n", type=int, default=20, help="timed passes (warmup excluded)")
    ap.add_argument("--threads", type=int, default=1, help="ggml CLI threads")
    ap.add_argument("--cli", default=None, help="path to voicedetect-cli")
    ap.add_argument("--device", default="cpu",
                    help="ggml device for VOICEDETECT_DEVICE (cpu, or e.g. CUDA0)")
    ap.add_argument("--out", default="benchmarks/RESULTS.md")
    ap.add_argument("--append-gpu", action="store_true",
                    help="append GPU rows to --out instead of rewriting it (for gpu_verify.sh)")
    ap.add_argument("--ref-threads", type=int, default=None,
                    help="reference thread budget (defaults to --threads for a "
                         "matched core-for-core comparison)")
    args = ap.parse_args()

    global REF_THREADS
    REF_THREADS = args.ref_threads if args.ref_threads is not None else args.threads

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cli = args.cli or _first_existing([
        os.path.join(root, "build", "examples", "cli", "voicedetect-cli"),
        shutil.which("voicedetect-cli") or "",
    ])
    if not cli:
        print("ERROR: voicedetect-cli not found; build it or pass --cli", file=sys.stderr)
        return 2
    audio = args.audio if os.path.isfile(args.audio) else os.path.join(root, args.audio)

    models = discover_models(root)
    rows = []
    for spec in models:
        sys.stderr.write(f"[bench] {spec['name']} (ggml {args.device}) ... ")
        sys.stderr.flush()
        ggml_ms, gerr = bench_ggml(cli, spec["ggml"], audio, spec["mode"],
                                   args.n, args.threads, args.device)
        sys.stderr.write(f"{ggml_ms if ggml_ms else gerr}\n")
        ref_ms = ref_err = None
        if not args.append_gpu:  # reference is CPU-only; skip on GPU append pass
            sys.stderr.write(f"[bench] {spec['name']} (reference) ... ")
            sys.stderr.flush()
            ref_ms, ref_err = bench_ref(spec, audio, args.n)
            sys.stderr.write(f"{ref_ms if ref_ms else ref_err}\n")
        rows.append((spec, ggml_ms, gerr, ref_ms, ref_err))

    cpu, vers = machine_info(args.threads)
    label = f"GPU ({args.device})" if args.device != "cpu" else f"CPU ({args.threads} thread)"

    def fmt(v):
        return f"{v:.1f}" if isinstance(v, (int, float)) else "n/a"

    lines = []
    if args.device != "cpu":
        lines.append(f"\n### GPU rows - device `{args.device}` ({datetime.now(timezone.utc):%Y-%m-%d})\n")
        lines.append("| Model | ggml ms/clip (GPU) | reference ms/clip (CPU) | note |")
        lines.append("| ----- | ------------------ | ----------------------- | ---- |")
    else:
        lines.append(f"| Model | ggml ms/clip | reference ms/clip | speedup (ref/ggml) | note |")
        lines.append("| ----- | ------------ | ----------------- | ------------------ | ---- |")
    for spec, ggml_ms, gerr, ref_ms, ref_err in rows:
        speed = (f"{ref_ms / ggml_ms:.2f}x"
                 if isinstance(ggml_ms, (int, float)) and isinstance(ref_ms, (int, float))
                 else "n/a")
        note = "; ".join(x for x in (gerr, ref_err) if x) or "ok"
        if args.device != "cpu":
            lines.append(f"| {spec['name']} | {fmt(ggml_ms)} | {fmt(ref_ms)} | {note} |")
        else:
            lines.append(f"| {spec['name']} | {fmt(ggml_ms)} | {fmt(ref_ms)} | {speed} | {note} |")
    table = "\n".join(lines) + "\n"

    header = (
        "# voice-detect.cpp comparative benchmark\n\n"
        f"Generated: {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} "
        f"by `scripts/bench_compare.py` (N={args.n}, warmup excluded).\n\n"
        f"- Machine: {cpu}\n"
        f"- ggml side: `voicedetect-cli bench`, device `{args.device}`, "
        f"`--threads {args.threads}`\n"
        f"- Reference: CPU {REF_THREADS}-thread (torch.set_num_threads({REF_THREADS}) / onnxruntime "
        f"intra+inter-op=1)\n"
        f"- Versions: torch {vers['torch']}, onnxruntime {vers['onnxruntime']}, "
        f"transformers {vers['transformers']}, numpy {vers['numpy']}\n"
        f"- Clip: `{os.path.relpath(audio, root)}` (3 s, 16 kHz mono)\n\n"
        "speedup = reference / ggml (>1 means ggml is faster). Absolute numbers are "
        "machine-specific. The ggml FBank uses a radix-2 FFT power spectrum (parity "
        "exact vs the previous naive DFT), so the speaker-encoder deficit is the "
        "encoder graph (ggml f32 conv/mul_mat vs MLAS/torch), not the front end.\n\n"
        f"## Results - {label}\n\n"
    )

    out_path = args.out if os.path.isabs(args.out) else os.path.join(root, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if args.append_gpu and os.path.isfile(out_path):
        with open(out_path, "a") as f:
            f.write(table)
        sys.stderr.write(f"appended GPU rows to {out_path}\n")
    else:
        with open(out_path, "w") as f:
            f.write(header + table)
        sys.stderr.write(f"wrote {out_path}\n")
    print(header + table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
