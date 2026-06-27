#!/usr/bin/env python3
"""measure_voice - produce the HONEST 8-thread WeSpeaker numbers used by the
voice hero race, writing them to spec.json next to this file.

ggml side : voicedetect-cli bench --mode embed --threads 8 (min ms/embed)
onnx side : onnxruntime CPU, intra_op=8, Kaldi-FBank + CMN (same pre-proc as the
            ggml path), min ms/embed - mirrors scripts/bench_compare.bench_ref_onnx.

We interleave ggml and onnx rounds and take the MIN across rounds for each engine.
On a contended box the min approximates the uncontended latency (the run where the
scheduler gave the engine its full 8 cores), which is the fair core-for-core signal
- and it is honest: it is a real observed time, not an average inflated by the load
of an unrelated process. A verify computes TWO embeddings, so verify_ms = 2*embed_ms.
"""
import json, subprocess, re, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CLI = ROOT / "build-cpu/examples/cli/voicedetect-cli"
MODEL = ROOT / "models/wespeaker_resnet34_f32.gguf"
A = ROOT / "tests/fixtures/clip_a.wav"
B = ROOT / "tests/fixtures/clip_b_same.wav"
THREADS = 8
N = 40
ROUNDS = 6


def ggml_embed_min():
    out = subprocess.run([str(CLI), "bench", "--model", str(MODEL), "--input", str(A),
                          "--mode", "embed", "--n", str(N), "--threads", str(THREADS)],
                         capture_output=True, text=True, check=True)
    txt = out.stdout + out.stderr
    return float(re.search(r"min=([0-9.]+)", txt).group(1)), txt.strip()


def verify_verdict():
    out = subprocess.run([str(CLI), "verify", "--model", str(MODEL), "--a", str(A),
                          "--b", str(B), "--threads", str(THREADS), "--json"],
                         capture_output=True, text=True, check=True)
    txt = (out.stdout + out.stderr).strip()
    d = float(re.search(r"distance=([0-9.]+)", txt).group(1))
    thr = float(re.search(r"threshold=([0-9.]+)", txt).group(1))
    ver = "true" in re.search(r"verified=(\w+)", txt).group(1)
    return d, thr, ver, txt


def make_onnx_fwd():
    import numpy as np, soundfile as sf, torch
    import torchaudio.compliance.kaldi as kaldi
    import onnxruntime as ort
    torch.set_num_threads(THREADS)
    onnx = str(ROOT / "models/wespeaker_voxceleb_resnet34.onnx")
    so = ort.SessionOptions()
    so.intra_op_num_threads = THREADS
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(onnx, sess_options=so, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    audio, _ = sf.read(str(A), always_2d=False)
    audio = audio.astype("float32")
    wav = torch.from_numpy(audio).unsqueeze(0)
    feed = (lambda fb: (fb - fb.mean(dim=0, keepdim=True)).numpy().astype("float32")[None, :, :])(
        kaldi.fbank(wav, sample_frequency=16000, num_mel_bins=80,
                    frame_length=25.0, frame_shift=10.0, dither=0.0))

    def fwd():
        out = sess.run(None, {in_name: feed})
        emb = np.asarray(out[0]).reshape(-1).astype("float32")
        emb / (np.linalg.norm(emb) or 1.0)
    return fwd


def onnx_embed_min(fwd):
    fwd()
    best = 1e9
    for _ in range(N):
        t0 = time.perf_counter(); fwd(); best = min(best, (time.perf_counter() - t0) * 1000.0)
    return best


def main():
    fwd = make_onnx_fwd()
    g_best, o_best, graw = 1e9, 1e9, ""
    for _ in range(ROUNDS):
        g, graw = ggml_embed_min(); g_best = min(g_best, g)
        o_best = min(o_best, onnx_embed_min(fwd))
    d, thr, ver, vraw = verify_verdict()
    spec = {
        "ggml_embed_ms": round(g_best, 2),
        "onnx_embed_ms": round(o_best, 2),
        "ggml_verify_ms": round(2 * g_best, 2),
        "onnx_verify_ms": round(2 * o_best, 2),
        "ratio": round(o_best / g_best, 2),
        "distance": round(d, 3),
        "threshold": thr,
        "verified": ver,
        "verdict": "SAME SPEAKER" if ver else "DIFFERENT SPEAKER",
        "threads": THREADS,
        "n": N,
        "rounds": ROUNDS,
        "method": "min-of-rounds (uncontended-latency, interleaved A/B)",
        "ggml_raw": graw,
        "verify_raw": vraw,
    }
    (HERE / "spec.json").write_text(json.dumps(spec, indent=2))
    print(json.dumps(spec, indent=2))


if __name__ == "__main__":
    main()
