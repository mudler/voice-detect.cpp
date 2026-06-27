# Benchmarks

This documents the comparative latency harness for voice-detect.cpp: the ggml
engine vs the native Python/ONNX/torch reference each model came from. The
measured CPU results live in [../benchmarks/RESULTS.md](../benchmarks/RESULTS.md);
GPU rows are appended there by the CUDA-host script.

## What is measured

End-to-end per-clip latency (ms/clip, mean over N timed passes, one warmup pass
excluded) for every speaker encoder and the two wav2vec2 analyze heads:

| Model | ggml side | reference side |
| ----- | --------- | -------------- |
| ECAPA-TDNN | `voicedetect-cli bench` (embed) | SpeechBrain `embedding_model` (torch) on Kaldi FBank |
| WeSpeaker ResNet34 | `bench` (embed) | onnxruntime CPU on Kaldi FBank |
| ERes2Net | `bench` (embed) | onnxruntime CPU on Kaldi FBank |
| CAM++ | `bench` (embed) | onnxruntime CPU on Kaldi FBank |
| wav2vec2 emotion | `bench --mode analyze` | HF transformers `superb/wav2vec2-base-superb-er` |
| wav2vec2 age/gender | `bench --mode analyze` | audeering `AgeGenderModel` (transformers) |

The ggml side calls the real C++ entry point (`embed_path` /
`analyze_path_json`): WAV decode + Kaldi FBank front end + encoder/analyze graph
+ L2-normalize. The bench subcommand reports a machine-parseable
`embed: <ms> ms/clip over <N> passes` line that the harness scrapes.

## Running the CPU comparison

```bash
# build the CLI (CPU) first
cmake -B build -DVOICEDETECT_BUILD_TESTS=ON -DGGML_NATIVE=ON && cmake --build build -j

# reference venv (see README "Converting a model")
.venv/bin/python scripts/bench_compare.py \
    --audio tests/fixtures/clip_a.wav --n 20 --threads 1 \
    --out benchmarks/RESULTS.md
```

The ECAPA GGUF is resolved from `models/ecapa_voxceleb_f32.gguf`, then
`/tmp/ecapa_f32.gguf`, or `$VD_BENCH_ECAPA_GGUF`. All other GGUFs and the ONNX
reference files are expected under `models/`. A model whose reference deps or
weights are missing is recorded as `n/a` in the `note` column, never fatal.

## Methodology (honest)

- **Matched threads, both sides.** `--threads N` sets the ggml CLI thread count;
  the reference mirrors it (`torch.set_num_threads(N)`, onnxruntime
  `intra_op = N`) unless `--ref-threads M` overrides. Default is 1 thread on both
  sides: a core-for-core comparison, not throughput-at-max-cores.
- **Warmup excluded.** One untimed warmup pass, then N timed passes (default 20),
  wall-clock mean.
- **Front end.** The ggml FBank uses a radix-2 FFT power spectrum (cached twiddles
  + bit-reversal, double precision). It is bit-for-bit equal to the previous naive
  DFT within float rounding (the FBank/embedding parity gates stay cosine 1.0) but
  ~100x faster: on the 3 s clip the front end fell from ~400 ms to ~3 ms, so it is
  no longer a meaningful share of the per-clip time. The reference includes its
  own feature-extraction (Kaldi FBank / HF feature extractor) inside the timed
  loop.
- **Audio I/O.** The reference loads the clip once; the ggml side re-decodes the
  (tiny, 3 s) WAV every call inside `embed_path`, a sub-millisecond overhead.
- **`speedup = reference / ggml`** (>1 means ggml is faster). Absolute numbers
  are machine-specific; see the machine/version block at the top of
  `benchmarks/RESULTS.md`.

### Reading the current CPU numbers

The FFT front end (above) closed most of the speaker-encoder gap that was pure
front-end waste: single-thread ECAPA went 657 -> 228 ms, CAM++ 487 -> 81 ms
(2.9-6.0x), parity unchanged. The wav2vec2 analyze heads feed the raw waveform
straight to the conv stack (no FBank), so the FFT does not touch them.

What remains: the ggml f32 engine is still **slower** than the heavily
BLAS-optimized torch/onnxruntime reference (speedup < 1). After the FFT, the
per-clip time is dominated by the encoder graph itself - ggml's f32
`im2col + ggml_mul_mat` conv path (tinyBLAS/`GGML_LLAMAFILE` already on) vs
onnxruntime MLAS / torch, which is genuinely faster single-thread on CPU.
Multithreading is an absolute-latency win (ECAPA 228 -> ~54 ms at 8 threads) but
MLAS/torch parallelize at least as well, so it does **not** close the relative
ratio. The honest realistic CPU ceiling vs MLAS stays < 1x without a major
hand-written GEMM/conv effort. The value of the port is no-Python deployment, a
small `libvoicedetect.so`, GGUF quantization, and the same code path on any ggml
GPU backend. The remaining CPU lever is quantized (q8_0) weights to cut matmul
bandwidth (parity within the q8 tolerance, not bit-exact).

**Round 2 update (stacked graph-level levers, vs prior baseline `0d9c1b3`).** A
second optimization round narrowed the gap with several parity-safe levers
(numbers are contention-robust back-to-back A/B, median-of-N, on a loaded box;
deltas are the signal, absolute ms are load-specific). See
[../benchmarks/RESULTS.md](../benchmarks/RESULTS.md) for the full tables:
ERes2Net 1x1-conv -> `ggml_mul_mat` (1t -8.1%, 8t -21.6% to -35.7% isolated,
bit-identical); CAM++ channel-major layout + factored O(T') context (1t -28.7%,
8t -14.4%); wav2vec2 age/gender conv-LayerNorm `ggml_cont` elimination (1t -14.1%,
8t -8.7%); ECAPA drop of production-path intermediate captures (8t -3.0%);
emotion flat (no regression). An **opt-in (default OFF)** AVX512 multi-ISA
runtime dispatch (`VOICEDETECT_CPU_ALL_VARIANTS`) helps the GEMM-bound wav2vec2
heads (emotion 8t -14.4%, age/gender -19.2%) but **regresses** the
bandwidth-bound ERes2Net 2D-conv encoder **+21.9%** at 8 threads (AVX512
downclock), so it is gated/unshipped pending per-model ISA selection + variant
co-packaging. These deltas are vs our OWN prior ggml baseline, not a fresh
head-to-head vs onnxruntime; they narrow the gap (ERes2Net ~1.3-1.6x at 8t) but a
residual conv-kernel gap remains intrinsic, while the GEMM-bound wav2vec2 heads
are the most likely to reach near-parity (especially on the gated AVX512 build).
No model regressed.

## GPU verification + benchmark (CUDA host only)

`scripts/gpu_verify.sh` MUST run on a machine with an NVIDIA GPU (CUDA toolkit +
driver). It is intentionally **not** part of CI here, because the CI/dev box that
produced the CPU table has no GPU (`nvidia-smi`/`nvcc` absent). On a CUDA host it:

1. builds `build-cuda/` with `-DVOICEDETECT_GGML_CUDA=ON`,
2. regenerates the CPU reference goldens and runs the ctest parity gates with the
   engine forced onto the GPU (`VOICEDETECT_DEVICE=CUDA0`), confirming
   cosine `>= 0.9999` / max|d| `<= 1e-3` / identical verdict still hold on GPU,
3. runs the bench on the GPU and appends GPU rows to `benchmarks/RESULTS.md`.

```bash
# on a DGX / CUDA host:
./scripts/gpu_verify.sh
VD_GPU_DEVICE=CUDA0 N=50 ./scripts/gpu_verify.sh   # pick device / passes
SKIP_BUILD=1 ./scripts/gpu_verify.sh               # reuse build-cuda/
```

Device selection follows the engine's runtime backend pick in `src/backend.cpp`:
`VOICEDETECT_DEVICE=cpu` forces CPU, otherwise the value matches a ggml
backend-registry device by name (case-insensitive), e.g. `CUDA0`. The script
exports it for every parity gate and bench run.
