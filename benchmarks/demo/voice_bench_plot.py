#!/usr/bin/env python3
"""voice_bench_plot - honest WeSpeaker latency chart (dark theme), per-embed,
8 CPU threads, from the REAL measured numbers in spec.json.

The point of this chart (vs a winner-take-all race) is to show the FULL picture:
- end-to-end (wav -> embedding) the two engines are within ~12 %;
- onnxruntime's model inference ALONE is faster (10.6 ms vs ggml ~15 ms e2e);
- voice-detect.cpp reaches parity with a faster integrated FBank.
No inflated ratios. Lower is faster. Bars are the measured end-to-end medians; the
component split under onnxruntime is annotated (component medians do not sum to the
e2e median, so it is shown as context, not stacked).
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / "spec.json").read_text())
OUT = HERE.parent / "plots" / "voice_bench.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

BG = "#0d1117"
INK = "#d7dde5"
DIM = "#6e7681"
TEAL = "#3ec8e0"
SLATE = "#94a3b2"

g = SPEC["ggml_embed_ms"]
o = SPEC["onnx_embed_ms"]
o_model = SPEC["onnx_model_ms"]
o_fbank = SPEC["onnx_fbank_ms"]

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                     "axes.edgecolor": "#2a3340", "xtick.color": DIM, "ytick.color": INK})
fig, ax = plt.subplots(figsize=(11, 5.0), dpi=120)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

ax.barh(1, g, color=TEAL, height=0.46, zorder=3)
ax.text(g + 0.35, 1, f"{g:.1f} ms", va="center", color=INK, fontsize=13, fontweight="bold")
ax.barh(0, o, color=SLATE, height=0.46, zorder=3)
ax.text(o + 0.35, 0, f"{o:.1f} ms", va="center", color=INK, fontsize=13, fontweight="bold")

ax.set_yticks([1, 0])
ax.set_yticklabels(["voice-detect.cpp\n(ggml, AVX-512)", "onnxruntime\n(MLAS)"], fontsize=13)
ax.set_xlabel("end-to-end per-embed latency (ms)  -  lower is faster", color=DIM, fontsize=11)
ax.set_xlim(0, max(g, o) * 1.52)
ax.set_ylim(-0.7, 1.7)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(length=0)
ax.grid(axis="x", color="#222b35", zorder=0)

# honest context annotations
ax.text(g + 0.35, 1 - 0.30, "integrated FBank + model, one binary", va="center", color=TEAL, fontsize=10)
ax.text(o + 0.35, 0 - 0.30, f"model {o_model:.1f} ms  +  FBank {o_fbank:.1f} ms",
        va="center", color=SLATE, fontsize=10)  # FBank via torchaudio

ax.set_title("WeSpeaker ResNet34 speaker embedding  -  CPU, 8 threads",
             color=INK, fontsize=16, fontweight="bold", loc="left", pad=18)
fig.text(0.012, 0.95, "honest benchmark  -  AMD Ryzen 9 9950X3D, median, contended box",
         color=DIM, fontsize=10)

cap = ("End-to-end the two engines are within ~12% (ggml 15.0 ms vs onnxruntime 16.8 ms). onnxruntime's model "
       "inference alone (10.6 ms)\nis faster; voice-detect.cpp closes the gap with a faster integrated FBank. "
       "Embeddings are bit-exact (cosine parity = 1.000).\nNot a 1.5x win either way: comparable speed - but "
       "voice-detect.cpp ships as one static binary, no torch, no onnxruntime, no Python.")
fig.text(0.012, 0.02, cap, color=DIM, fontsize=10.5, va="bottom")

plt.subplots_adjust(left=0.20, right=0.97, top=0.84, bottom=0.32)
fig.savefig(OUT, facecolor=BG)
print("wrote", OUT)
