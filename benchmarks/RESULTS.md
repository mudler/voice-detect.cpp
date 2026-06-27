# voice-detect.cpp comparative benchmark

> For the durable campaign summary, final CPU standing vs MLAS/torch, and the documented path to true MLAS parity, see [`docs/cpu-optimization.md`](../docs/cpu-optimization.md).

Generated: 2026-06-22 14:47 UTC by `scripts/bench_compare.py` (N=20, warmup excluded).

- Machine: AMD Ryzen 9 9950X3D 16-Core Processor
- ggml side: `voicedetect-cli bench`, device `cpu`, `--threads 1`
- Reference: CPU 1-thread (torch.set_num_threads(1) / onnxruntime intra+inter-op=1)
- Versions: torch 2.11.0+cu130, onnxruntime 1.27.0, transformers 5.3.0, numpy 2.3.5
- Clip: `tests/fixtures/clip_a.wav` (3 s, 16 kHz mono)

speedup = reference / ggml (>1 means ggml is faster). Absolute numbers are machine-specific. The ggml FBank uses a radix-2 FFT power spectrum (parity exact vs the previous naive DFT), so the speaker-encoder deficit is the encoder graph (ggml f32 conv/mul_mat vs MLAS/torch), not the front end.

## Results - CPU (1 thread)

| Model | ggml ms/clip | reference ms/clip | speedup (ref/ggml) | note |
| ----- | ------------ | ----------------- | ------------------ | ---- |
| ECAPA-TDNN | 262.4 | 117.3 | 0.45x | ok |
| WeSpeaker ResNet34 | 216.6 | 49.2 | 0.23x | ok |
| ERes2Net | 254.3 | 54.3 | 0.21x | ok |
| CAM++ | 95.3 | 18.7 | 0.20x | ok |
| wav2vec2 emotion | 671.6 | 361.1 | 0.54x | ok |
| wav2vec2 age/gender | 1355.2 | 1022.4 | 0.75x | ok |
## CPU optimization #2: wav2vec2 analyze heads (flash attention + cached pos-conv)

Following the depth-anything.cpp recipe (profile host-side per-call work first, then
fuse attention). Two parity-safe wins on the wav2vec2 analyze heads, measured
matched-thermal (pristine HEAD vs optimized, same session, `--threads 1`):

| Model | ggml BEFORE ms | ggml AFTER ms | delta | what moved |
| ----- | -------------- | ------------- | ----- | ---------- |
| wav2vec2 emotion | 738.7 | ~645 | **-12.6%** | cached pos-conv (-20 ms) + flash attn (~-11 ms) |
| wav2vec2 age/gender | 1619.4 | ~1361 | **-16%** | cached pos-conv (-112 ms) + flash attn (~-30 ms) |

**Win A - cached positional-conv weight (the headline).** The weight-norm
reconstruction `w = g * v / ||v||` of the positional-conv kernel is a pure function
of the static model weights, yet it was rebuilt on the host on EVERY forward (a
scalar triple loop over K*IC*OC). Micro-benchmarked at **19.7 ms/call (emotion)**
and **112.2 ms/call (age/gender)** of pure per-call host overhead the GEMM threads
could not hide - exactly depth-anything's pos-embed-caching pattern. The analyzer
is constructed fresh per analyze call (`Model::analyze_path_json`), so the cache is
process-global keyed by the ModelLoader. Bit-exact (same reconstruction).

**Win B - fused flash attention (`ggml_flash_attn_ext`, F32 k/v, `GGML_PREC_F32`).**
Replaces the manual QK^T/softmax/xV path (which materialized a [T,T] scores matrix
and did 3 permute+cont copies per layer) with one streaming fused op. Small win
(attention is a minor FLOP share here vs the FFN matmuls; T~149), but it is BOTH a
latency win AND TIGHTER parity than the manual path: enc_layer_last max|d| drops
from 4.24e-5 -> 2.10e-5 (emotion) and 3.54e-5 -> 2.95e-5 (age/gender); all parity
gates stay green, dominant labels unchanged. `VD_ATTN=manual` restores the old path.

Speaker encoders are unchanged (no wav2vec2 path); their parity gates
(WeSpeaker/CAM++ encoder_out + embedding cosine = 1.000000) stay exact.

## Round 2 CPU optimization (stacked, vs prior baseline 0d9c1b3)

A second optimization round that stacks several parity-safe graph-level levers on
top of the pass-1/-2 baseline (`0d9c1b3`, the conv2d routing helper). Measured
contention-robust: back-to-back A/B (baseline binary vs optimized binary in the
same session), median-of-N, on a loaded 16-core box. **The absolute ms below are
load-specific - treat the deltas, not the raw numbers, as the signal.** Each lever
was re-gated after landing and held parity (cosine = 1.0, or within the model's
parity gate); the bit-exact ones are called out.

| Model | 1t before -> after | 1t delta | 8t before -> after | 8t delta | lever (parity) |
| ----- | ------------------ | -------- | ------------------ | -------- | -------------- |
| ERes2Net | 240 -> 221 ms | **-8.1%** | 120 -> 94 ms | **-21.6%** (to -35.7% isolated) | stride-1 1x1 conv -> `ggml_mul_mat` (skip im2col); parity-exact, **bit-identical** |
| CAM++ | 89 -> 63 ms | **-28.7%** | 48 -> 41 ms | **-14.4%** | channel-major dense layout + factored O(T') CAM context (was O(T'^2)); cosine = 1.0 |
| wav2vec2 age/gender | 1266 -> 1087 ms | **-14.1%** | 189 -> 173 ms | **-8.7%** | conv-encoder LayerNorm `ggml_cont` elimination (channel-major conv output); within gate |
| ECAPA-TDNN | - | - | (8t) | **-3.0%** | drop production-path intermediate captures (harness-only graph outputs); parity-exact |
| wav2vec2 emotion | - | flat | - | flat | no applicable lever this round; **no regression** |

The ERes2Net 1x1-conv lever is bit-identical (same f32 reduction over IC); the
8t win is **-21.6%** in the contention-robust back-to-back run and up to **-35.7%**
when measured isolated on a quiet box. CAM++ is the largest single-thread win
(-28.7%) from keeping the dense backbone channel-major (every 1x1 conv becomes one
`ggml_mul_mat`, no transpose churn) plus the O(T'^2) -> O(T' . n_seg) context
refactor. The wav2vec2 age/gender LayerNorm lever halves the per-conv-layer
`ggml_cont` work; emotion uses a different (GroupNorm) front end and is untouched,
so it is flat (and unregressed). ECAPA's small win is pure graph/readback overhead
removed from the production forward (no math change).

### AVX512 multi-ISA runtime dispatch: OPT-IN, default OFF (gated, unshipped)

`VOICEDETECT_CPU_ALL_VARIANTS` builds ggml's `GGML_CPU_ALL_VARIANTS` +
`GGML_BACKEND_DL` (one dlopen'd CPU backend per x86 micro-arch; the best the host
supports is picked at load). The dispatcher is **parity-exact** (same kernels,
wider SIMD: emotion enc_layer_last max|d| = 2.24e-5, all gates green). When it
selects the AVX512 variant (Zen4 / Sapphire Rapids) it is a clear win on the
GEMM-bound analyze heads but a clear loss on the bandwidth-bound 2D-conv encoder:

| Model (8 threads) | with AVX512 variant | note |
| ----------------- | ------------------- | ---- |
| wav2vec2 emotion | **-14.4%** | GEMM-bound, benefits from wider SIMD |
| wav2vec2 age/gender | **-19.2%** | GEMM-bound, benefits from wider SIMD |
| ERes2Net (2D-conv encoder) | **+21.9% (REGRESSION)** | bandwidth-bound; AVX512 sustained-clock downclock hurts the im2col conv |

Because one global ISA choice cannot satisfy both model families, the option stays
**default OFF** and is **gated / unshipped** pending two follow-ups: (1) per-model
ISA selection (route only the GEMM-bound heads to the AVX512 variant, keep the
conv encoders on AVX2/FMA), and (2) co-packaging the per-variant `.so` modules in
the distributed artifact. Neither is done in this round.

### Honest framing (Round 2)

These deltas are measured vs our OWN prior ggml baseline (`0d9c1b3`), NOT a fresh
head-to-head vs onnxruntime this round. The wins NARROW the gap to onnxruntime
substantially (sibling face-detect.cpp SCRFD detect ~1.7-2x, ERes2Net here
~1.3-1.6x at 8 threads relative to the pre-round ratio), but on the conv-bound
encoder models a residual gap remains intrinsic: ggml's im2col / Winograd conv on
small/mid feature maps vs onnxruntime's tuned conv kernels. wav2vec2 (GEMM-bound)
is the most likely to reach near-parity, especially on the gated AVX512 build.
ECAPA is floored by the single-threaded Kaldi FBank front end on the reference
side. No model regressed.

## Dead-end: direct 2D conv (`ggml_conv_2d_direct`) is SLOWER on this CPU

The depth-anything DPT-head win was K>1 `ggml_conv_2d_direct` (no im2col blow-up).
Ported to the speaker encoders' 2D convs (WeSpeaker ResNet34, ERes2Net, CAM++ FCM)
and A/B-measured, it is **~10-15% SLOWER** than im2col + llamafile-sgemm here:

| Model | im2col ms | direct ms |
| ----- | --------- | --------- |
| WeSpeaker ResNet34 | 226.7 | 260.7 |
| ERes2Net | 279.9 | 309.2 |
| CAM++ | 107.7 | 110.3 |

The feature maps are small (~[T,80] x <=256 ch) and tinyBLAS's blocked GEMM beats
ggml's basic direct-conv CPU kernel - the opposite of depth-anything's large
(504x336) DPT head. The default stays im2col on CPU and GPU; `VD_CONV2D=direct`
keeps the A/B path for other shapes/HW. (Winograd F(2x2,3x3) - depth-anything's
deeper win - was not pursued: it must beat llamafile's already-tuned sgemm, which
even plain direct conv loses to here, so the risk/effort is not justified by the
profile. Recorded for a future pass.)

## FFT FBank: before -> after (CPU, 1 thread)

The front end was a naive O(T*nfft^2) DFT that ate ~400 ms/clip (66-82% of every
speaker-encoder forward; profiled by separating the FBank host stage from the
encoder graph). Replacing it with a cached radix-2 FFT (`src/fbank.cpp`) dropped
the FBank stage to ~3 ms with **bit-identical** embeddings (FBank + embedding
parity gates stay cosine 1.0). The wav2vec2 analyze heads have no FBank, so they
are unchanged (the small deltas below are run-to-run noise).

| Model | ggml BEFORE | ggml AFTER | reference | speedup BEFORE | speedup AFTER |
| ----- | ----------- | ---------- | --------- | -------------- | ------------- |
| ECAPA-TDNN | 657.7 | 218.7 | 106.9 | 0.17x | 0.49x |
| WeSpeaker ResNet34 | 609.4 | 200.2 | 44.5 | 0.07x | 0.22x |
| ERes2Net | 636.4 | 223.1 | 45.3 | 0.07x | 0.20x |
| CAM++ | 487.3 | 83.1 | 18.6 | 0.04x | 0.22x |
| wav2vec2 emotion | 580.2 | 576.1 | 319.1 | 0.55x | 0.55x |
| wav2vec2 age/gender | 1238.2 | 1269.2 | 971.9 | 0.75x | 0.77x |

## Matched 8 threads (honest multi-thread)

`scripts/bench_compare.py --threads 8` runs BOTH sides at 8 threads. ggml absolute
latency drops a lot (ECAPA 219 -> 54 ms), but MLAS/torch parallelize at least as
well, so the relative ratio does not improve - multithreading is an absolute-latency
win, not a gap-closer. The engine default stays 1 thread (a LocalAI backend serving
concurrent requests must not oversubscribe); `--threads` is the tuning knob.

| Model | ggml@8 ms | reference@8 ms | speedup |
| ----- | --------- | -------------- | ------- |
| ECAPA-TDNN | 54.1 | 27.6 | 0.51x |
| WeSpeaker ResNet34 | 56.1 | 7.6 | 0.14x |
| ERes2Net | 82.9 | 16.3 | 0.20x |
| CAM++ | 30.5 | 12.1 | 0.40x |
| wav2vec2 emotion | 129.6 | 68.5 | 0.53x |
| wav2vec2 age/gender | 324.8 | 193.2 | 0.59x |

Note: a fresh `bench_compare.py` run rewrites the section above the
"CPU optimization #2" heading with the latest single-config table; the durable
records below are kept by hand after a rerun.

## Fair GPU comparison (GB10): ggml-GPU vs reference-GPU (SAME GPU)

Measured on the SAME NVIDIA GB10 (Grace-Blackwell, sm_121a, CUDA 13.0): torch-cuda
(torch 2.11.0+cu130) for the torch models, onnxruntime-gpu 1.24.0 (CUDAExecutionProvider)
for the ONNX models. Warmup 20, N=100, per-call median. speedup = reference-GPU /
ggml-GPU (>1 means ggml-GPU is faster).

| Model | ggml-GPU ms | reference-GPU ms | framework + provider | speedup (ref/ggml) |
| ----- | ----------- | ---------------- | -------------------- | ------------------ |
| ECAPA-TDNN | 8.92 | 11.49 | torch-cuda (SpeechBrain encoder) | **1.29x (ggml wins)** |
| WeSpeaker ResNet34 | 8.46 | 5.29 | onnxruntime-gpu CUDA EP | 0.63x (ref 1.6x faster) |
| ERes2Net | 10.06 | 6.34 | onnxruntime-gpu CUDA EP | 0.63x (ref 1.6x faster) |
| CAM++ | 15.74 | 11.59 | onnxruntime-gpu CUDA EP | 0.74x (ref 1.4x faster) |
| wav2vec2 emotion | 33.53 | 9.14 | torch-cuda (HF wav2vec2) | 0.27x (ref 3.7x faster) |
| wav2vec2 age/gender | 69.58 | 21.03 | torch-cuda (audeering) | 0.30x (ref 3.3x faster) |

On the SAME GB10 GPU, ggml-GPU beats the reference on 1 of 6 models (ECAPA-TDNN,
1.29x). On the other five the GPU reference is faster (1.4-1.6x on the ONNX speaker
encoders, 3.3-3.7x on the wav2vec2 transformers): cuDNN/cuBLAS (ORT) and torch's
fused CUDA kernels still out-run the ggml CUDA graph. Both reference frameworks DO
run on Blackwell sm_121a (no model was blocked).
