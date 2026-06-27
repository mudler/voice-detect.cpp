#!/usr/bin/env python3
"""Provenance for memory.png: print one model's reference peak RSS.

Runs exactly ONE reference forward (the same loader used by
scripts/bench_compare.py) for a given model key so a parent `/usr/bin/time -v`
captures the peak resident set size of the real Python reference path
(torch / onnxruntime + the model weights). The ggml side is measured the same
way directly on the `voicedetect-cli` binary.

Usage (from the repo root):
    PY=.venv/bin/python
    # reference side (torch / onnxruntime)
    /usr/bin/time -v $PY benchmarks/demo/measure_peak_rss.py wespeaker  2>&1 | grep "Maximum resident"
    # ggml side (the lean C++ binary, mmap'd GGUF)
    /usr/bin/time -v build-cpu/examples/cli/voicedetect-cli embed \
        --model models/wespeaker_resnet34_f32.gguf \
        --input tests/fixtures/clip_a.wav --threads 8 >/dev/null 2>&1

Keys: wespeaker ecapa eres2net campplus emotion agegender
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import bench_compare as bc

bc.REF_THREADS = 8  # the module global the reference fns read

AUDIO = os.path.join(ROOT, "tests/fixtures/clip_a.wav")
M = lambda *p: os.path.join(ROOT, "models", *p)

SPECS = {
    "ecapa":     dict(name="ECAPA-TDNN", ref="speechbrain",
                      ref_model="speechbrain/spkrec-ecapa-voxceleb"),
    "wespeaker": dict(name="WeSpeaker", ref="onnx",
                      onnx=M("wespeaker_voxceleb_resnet34.onnx")),
    "eres2net":  dict(name="ERes2Net", ref="onnx",
                      onnx=M("3dspeaker_eres2net_base_200k.onnx")),
    "campplus":  dict(name="CAM++", ref="onnx",
                      onnx=M("3dspeaker_campplus_zh-cn_16k.onnx")),
    "emotion":   dict(name="emotion", ref="hf_emotion",
                      ref_model="superb/wav2vec2-base-superb-er"),
    "agegender": dict(name="age-gender", ref="hf_age_gender",
                      ref_model="audeering/wav2vec2-large-robust-24-ft-age-gender"),
}

if __name__ == "__main__":
    spec = SPECS[sys.argv[1]]
    ms = bc.REF_FN[spec["ref"]](spec, AUDIO, 1)  # warmup + 1 timed forward
    print(f"{spec['name']} reference forward ~{ms:.1f} ms")
