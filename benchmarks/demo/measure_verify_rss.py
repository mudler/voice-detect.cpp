#!/usr/bin/env python3
"""Provenance for spec.json ram_ggml_mb / ram_onnx_mb: peak RSS of a real WeSpeaker
VERIFY, ggml vs the onnxruntime reference, captured by a parent /usr/bin/time -v.

The onnxruntime side here is the EXACT reference verify a user runs: soundfile to
read both wavs, torchaudio Kaldi-FBank + CMN for features (the same preprocessing
as the ggml path), onnxruntime CPU for the model, two embeddings + cosine. We
report the CPU-only torch wheel (the fair CPU-to-CPU comparison): ~334 MB, ~5.4x
the ggml binary. A default `pip install torch` pulls the CUDA wheel, whose runtime
libs inflate RSS even on a CPU verify (~690 MB, ~11x), but those CUDA libs are
unused on a CPU verify, so we report the conservative ~334 MB number. The ggml
binary is ~62 MB either way.

Usage (from the repo root), take the median of a few runs each:
    # ggml side - the lean C++ binary, mmap'd GGUF, zero Python
    /usr/bin/time -v build-cpu/examples/cli/voicedetect-cli verify \
        --model models/wespeaker_resnet34_f32.gguf \
        --a tests/fixtures/clip_a.wav --b tests/fixtures/clip_b_same.wav \
        --threads 8 --json 2>&1 | grep "Maximum resident"
    # onnxruntime reference side
    /usr/bin/time -v PY benchmarks/demo/measure_verify_rss.py 2>&1 | grep "Maximum resident"
"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]

import numpy as np
import soundfile as sf
import torch
import torchaudio.compliance.kaldi as kaldi
import onnxruntime as ort

THREADS = 8
torch.set_num_threads(THREADS)
onnx = str(ROOT / "models/wespeaker_voxceleb_resnet34.onnx")
so = ort.SessionOptions()
so.intra_op_num_threads = THREADS
so.inter_op_num_threads = 1
sess = ort.InferenceSession(onnx, sess_options=so, providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name


def emb(path):
    audio, _ = sf.read(str(path), always_2d=False)
    wav = torch.from_numpy(audio.astype("float32")).unsqueeze(0)
    fb = kaldi.fbank(wav, sample_frequency=16000, num_mel_bins=80,
                     frame_length=25.0, frame_shift=10.0, dither=0.0)
    feed = (fb - fb.mean(dim=0, keepdim=True)).numpy().astype("float32")[None, :, :]
    out = sess.run(None, {in_name: feed})
    e = np.asarray(out[0]).reshape(-1).astype("float32")
    return e / (np.linalg.norm(e) or 1.0)


if __name__ == "__main__":
    ea = emb(ROOT / "tests/fixtures/clip_a.wav")
    eb = emb(ROOT / "tests/fixtures/clip_b_same.wav")
    d = 1.0 - float(np.dot(ea, eb))
    print(f"onnxruntime reference verify: distance={d:.4f}")
