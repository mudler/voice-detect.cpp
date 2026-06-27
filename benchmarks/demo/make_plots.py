#!/usr/bin/env python3
"""Render the voice-detect.cpp comparative benchmark plots.

Three honest, reproducible figures comparing the voice-detect.cpp ggml engine
against the native reference stack (onnxruntime for the speaker encoders,
PyTorch / HF transformers for the wav2vec2 analyze heads), all on CPU at 8
threads on the same box.

    infer_speed.png  per-model inference latency (ms/clip), ggml vs reference
    memory.png       peak process RAM (RSS), ggml vs reference
    model_size.png   GGUF (f32) weight footprint on disk, per model

HOW THE NUMBERS WERE MEASURED (all real, no invented values):

* infer_speed: `scripts/bench_compare.py --threads 8` (warmup excluded, N=20),
  which times the real `voicedetect-cli bench` C++ entry point on the ggml side
  and the native Python forward (onnxruntime / torch) on the reference side, on
  the same clip `tests/fixtures/clip_a.wav` (3 s, 16 kHz mono).
  Machine: AMD Ryzen 9 9950X3D 16-core. Run: 2026-06-24.
  Absolute latency is machine-specific; the relative standing is the signal.

* memory: peak resident set size from `/usr/bin/time -v` on a single
  embed/analyze call, ggml side = the `voicedetect-cli` binary (mmap'd GGUF),
  reference side = the Python path (framework runtime + model weights), via the
  same loaders used by `scripts/bench_compare.py`.

* model_size: actual on-disk byte size of the shipped f32 GGUF files in
  `models/`, reported as decimal MB (bytes / 1e6).

Reproduce:
    ~/.local/bin/uv venv ~/recon-demos/venv
    ~/.local/bin/uv pip install --python ~/recon-demos/venv/bin/python pillow matplotlib numpy
    ~/recon-demos/venv/bin/python benchmarks/demo/make_plots.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

# --------------------------------------------------------------------------
# House palette (matches the LocalAI race demos: teal engine accent on #0d1117)
# --------------------------------------------------------------------------
BG      = "#0d1117"   # page background
PANEL   = "#11161d"   # axes face
GRID    = "#222a35"   # gridlines
INK     = "#e6edf3"   # primary text
DIM     = "#8b98a8"   # secondary text
TEAL    = "#3ec8e0"   # voice-detect.cpp (ggml)
SLATE   = "#94a3b2"   # reference (onnxruntime / torch)
GOLD    = "#ffcf56"   # win highlight
GREEN   = "#56d364"

GGML_LABEL = "voice-detect.cpp (ggml)"
REF_LABEL  = "reference (onnxruntime / torch)"

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": BG,
    "savefig.facecolor": BG,
    "axes.facecolor": PANEL,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": DIM,
    "ytick.color": DIM,
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.linewidth": 1.0,
})

# --------------------------------------------------------------------------
# Data (model order shared across the three plots)
# --------------------------------------------------------------------------
MODELS = [
    "WeSpeaker\nResNet34",
    "ECAPA-TDNN",
    "ERes2Net",
    "CAM++",
    "wav2vec2\nemotion",
    "wav2vec2\nage-gender",
]

# infer_speed @ 8 threads, ms/clip (lower is better) - bench_compare.py 2026-06-24
GGML_MS = [13.8,  57.3,  62.1,  34.2, 127.8, 205.4]
REF_MS  = [25.9,  37.8,  40.8,  27.2,  88.9, 242.2]

# peak RSS in MB (lower is better) - /usr/bin/time -v Maximum resident set size,
# median of 5 runs, KB/1024. Reference side = the CPU-only torch wheel (the fair
# CPU-to-CPU comparison; the default CUDA torch wheel inflates RSS with unused
# CUDA libs). Measured 2026-06-25.
GGML_MB = [ 61.1, 111.9,  78.4,  54.1, 470.7, 1330.8]
REF_MB  = [308.8, 481.3, 351.5, 314.1, 839.4, 2910.3]

# GGUF f32 weights on disk, decimal MB (bytes / 1e6) - ls models/
SIZE_MB = [26.53, 83.24, 39.52, 27.70, 378.30, 1270.20]


def _flat(label):
    return label.replace("\n", " ")


def footer(fig):
    fig.text(0.012, 0.012, "voice-detect.cpp", color=TEAL, fontsize=11,
             fontweight="bold", ha="left", va="bottom")
    fig.text(0.165, 0.012, "brought to you by the LocalAI team", color=DIM,
             fontsize=10, ha="left", va="bottom")
    fig.text(0.988, 0.012,
             "CPU @ 8 threads  .  AMD Ryzen 9 9950X3D  .  clip_a.wav 3s/16kHz  .  2026-06-24",
             color=DIM, fontsize=9, ha="right", va="bottom")


def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(length=0)


# --------------------------------------------------------------------------
# 1) inference speed
# --------------------------------------------------------------------------
def plot_speed():
    import numpy as np
    fig, ax = plt.subplots(figsize=(12.2, 6.6))
    x = np.arange(len(MODELS))
    w = 0.38
    b1 = ax.bar(x - w / 2, GGML_MS, w, label=GGML_LABEL, color=TEAL,
                edgecolor=BG, linewidth=0.6, zorder=3)
    b2 = ax.bar(x + w / 2, REF_MS, w, label=REF_LABEL, color=SLATE,
                edgecolor=BG, linewidth=0.6, zorder=3)

    for bars, vals in ((b1, GGML_MS), (b2, REF_MS)):
        for r, v in zip(bars, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + max(REF_MS) * 0.012,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=9.5,
                    color=INK)

    # Highlight the two models where ggml is faster (shorter bar = faster).
    for i in range(len(MODELS)):
        if GGML_MS[i] < REF_MS[i]:
            factor = REF_MS[i] / GGML_MS[i]
            top = max(GGML_MS[i], REF_MS[i])
            ax.text(x[i], top + max(REF_MS) * 0.075,
                    f"ggml {factor:.1f}x faster", ha="center", va="bottom",
                    fontsize=10.5, color=GOLD, fontweight="bold")
            ax.scatter([x[i] - w / 2], [GGML_MS[i] + max(REF_MS) * 0.012],
                       marker="*", s=90, color=GOLD, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=11)
    ax.set_ylabel("inference latency  (ms / clip, lower is better)", fontsize=12)
    ax.set_ylim(0, max(REF_MS) * 1.24)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax)

    fig.suptitle("Inference speed  .  voice-detect.cpp vs reference",
                 x=0.012, ha="left", fontsize=19, fontweight="bold", color=INK,
                 y=0.975)
    ax.set_title("same CPU, same 8 threads, same 3 s clip  .  warmup excluded, N=20",
                 loc="left", color=DIM, fontsize=12, pad=10)
    ax.legend(loc="upper left", frameon=False, fontsize=11.5,
              labelcolor=INK, bbox_to_anchor=(0.0, 1.0))
    footer(fig)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.83, bottom=0.115)
    p = os.path.join(OUT, "infer_speed.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


# --------------------------------------------------------------------------
# 2) memory
# --------------------------------------------------------------------------
def plot_memory():
    import numpy as np
    fig, ax = plt.subplots(figsize=(12.2, 6.6))
    y = np.arange(len(MODELS))[::-1]   # WeSpeaker on top
    h = 0.38
    ax.barh(y + h / 2, GGML_MB, h, label=GGML_LABEL, color=TEAL,
            edgecolor=BG, linewidth=0.6, zorder=3)
    ax.barh(y - h / 2, REF_MB, h, label=REF_LABEL, color=SLATE,
            edgecolor=BG, linewidth=0.6, zorder=3)

    def fmt_mb(v):
        return f"{v/1000:.2f} GB" if v >= 1000 else f"{v:.0f} MB"

    for yi, gv, rv in zip(y, GGML_MB, REF_MB):
        ax.text(gv * 1.06, yi + h / 2, fmt_mb(gv), va="center", ha="left",
                fontsize=9.5, color=TEAL)
        ax.text(rv * 1.06, yi - h / 2, fmt_mb(rv), va="center", ha="left",
                fontsize=9.5, color=SLATE)
        factor = rv / gv
        ax.text(gv * 0.92, yi + h / 2, f"{factor:.0f}x less", va="center",
                ha="right", fontsize=9.5, color=GOLD, fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlim(30, 9000)
    ax.set_yticks(y)
    ax.set_yticklabels([_flat(m) for m in MODELS], fontsize=11)
    ax.set_xlabel("peak process RAM  (MB, log scale, lower is better)", fontsize=12)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax)

    fig.suptitle("Peak memory  .  the ggml binary stays lean",
                 x=0.012, ha="left", fontsize=19, fontweight="bold", color=INK,
                 y=0.975)
    ax.set_title("max resident set size of one embed/analyze call  .  "
                 "ggml = mmap'd GGUF, no Python runtime  .  CPU-only torch reference",
                 loc="left", color=DIM, fontsize=12, pad=10)
    ax.legend(loc="upper right", frameon=False, fontsize=11.5, labelcolor=INK)
    footer(fig)
    fig.subplots_adjust(left=0.17, right=0.985, top=0.83, bottom=0.115)
    p = os.path.join(OUT, "memory.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


# --------------------------------------------------------------------------
# 3) model size
# --------------------------------------------------------------------------
def plot_size():
    import numpy as np
    fig, ax = plt.subplots(figsize=(12.2, 6.2))
    y = np.arange(len(MODELS))[::-1]
    ax.barh(y, SIZE_MB, 0.62, color=TEAL, edgecolor=BG, linewidth=0.6, zorder=3)

    def fmt(v):
        return f"{v/1000:.2f} GB" if v >= 1000 else f"{v:.1f} MB"

    for yi, v in zip(y, SIZE_MB):
        ax.text(v * 1.06, yi, fmt(v), va="center", ha="left", fontsize=11,
                color=INK, fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlim(15, 3000)
    ax.set_yticks(y)
    ax.set_yticklabels([_flat(m) for m in MODELS], fontsize=11)
    ax.set_xlabel("GGUF f32 weights on disk  (MB, log scale)", fontsize=12)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax)

    fig.suptitle("Model footprint  .  one GGUF file per model",
                 x=0.012, ha="left", fontsize=19, fontweight="bold", color=INK,
                 y=0.975)
    ax.set_title("self-contained f32 weights shipped in models/  .  "
                 "no framework, no graph runtime on disk",
                 loc="left", color=DIM, fontsize=12, pad=10)
    footer(fig)
    fig.subplots_adjust(left=0.17, right=0.985, top=0.83, bottom=0.12)
    p = os.path.join(OUT, "model_size.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


if __name__ == "__main__":
    for fn in (plot_speed, plot_memory, plot_size):
        print("wrote", fn())
