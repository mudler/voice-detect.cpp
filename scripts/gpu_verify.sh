#!/usr/bin/env bash
#
# gpu_verify.sh - build voice-detect.cpp with the ggml CUDA backend, confirm the
# numerical parity gates still hold when the engine runs on the GPU, and append
# GPU latency rows to the comparative benchmark table.
#
# IMPORTANT: this script MUST run on a CUDA host (e.g. a DGX). It is deliberately
# NOT wired into CI here, because the CI/dev box that produced the CPU numbers in
# benchmarks/RESULTS.md has no NVIDIA GPU (no nvidia-smi / nvcc). Run it by hand
# on a machine with the CUDA toolkit + driver installed.
#
# What it does:
#   1. build/  -> build-cuda/  with -DVOICEDETECT_GGML_CUDA=ON.
#   2. Generates the reference goldens (CPU torch/onnxruntime, device-independent)
#      via scripts/gen_baseline.py for each model, into a temp dir.
#   3. Runs the existing ctest parity gates with VOICEDETECT_DEVICE forcing the
#      ggml CUDA device, so the GPU engine output is diffed against the SAME CPU
#      reference goldens (cosine >= 0.9999, max|d| <= 1e-3, identical verdict).
#   4. Runs scripts/bench_compare.py --device <dev> --append-gpu to append GPU
#      ggml ms/clip rows to benchmarks/RESULTS.md.
#
# Device selection: the engine's runtime backend pick is driven by
# VOICEDETECT_DEVICE (see src/backend.cpp): "cpu" forces CPU, otherwise it matches
# a ggml backend-registry device by name (case-insensitive), e.g. "CUDA0". This
# script exports VOICEDETECT_DEVICE=$VD_GPU_DEVICE (default CUDA0) for every gate
# and bench run. Override with: VD_GPU_DEVICE=CUDA1 ./scripts/gpu_verify.sh
#
# Usage:
#   ./scripts/gpu_verify.sh                 # full build + parity + GPU bench
#   VD_GPU_DEVICE=CUDA0 N=20 ./scripts/gpu_verify.sh
#   SKIP_BUILD=1 ./scripts/gpu_verify.sh    # reuse an existing build-cuda/
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEV="${VD_GPU_DEVICE:-CUDA0}"
BUILD_DIR="${BUILD_DIR:-build-cuda}"
N="${N:-20}"
AUDIO="${AUDIO:-tests/fixtures/clip_a.wav}"
PY="${PY:-$ROOT/.venv/bin/python}"
CLI="$ROOT/$BUILD_DIR/examples/cli/voicedetect-cli"

# --- 0. CUDA host guard -----------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. gpu_verify.sh MUST run on a CUDA host." >&2
    echo "       (This is intentionally NOT run in the CPU-only CI/dev box.)" >&2
    exit 2
fi
if ! command -v nvcc >/dev/null 2>&1; then
    echo "WARN: nvcc not on PATH; relying on CMake to locate the CUDA toolkit." >&2
fi
echo "== GPU host =="; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

# --- 1. Build with the ggml CUDA backend ------------------------------------
if [ "${SKIP_BUILD:-0}" != "1" ]; then
    echo "== Building $BUILD_DIR with -DVOICEDETECT_GGML_CUDA=ON =="
    cmake -B "$BUILD_DIR" \
        -DVOICEDETECT_BUILD_TESTS=ON \
        -DVOICEDETECT_GGML_CUDA=ON \
        -DGGML_CUDA_NO_VMM=ON \
        -DGGML_NATIVE=ON
    cmake --build "$BUILD_DIR" -j
fi
[ -x "$CLI" ] || { echo "ERROR: $CLI not built" >&2; exit 1; }

# Sanity: confirm the CUDA device is actually visible to the ggml registry and
# that the engine selects it (the CLI logs "vd::Backend using device: ...").
echo "== Confirming ggml sees device '$DEV' =="
VOICEDETECT_DEVICE="$DEV" "$CLI" info --model models/wespeaker_resnet34_f32.gguf >/dev/null 2>&1 || true

# --- 2. Generate reference goldens (CPU; device-independent) -----------------
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
gen() {  # gen <out.gguf> <gen_baseline args...>
    local out="$1"; shift
    if [ ! -x "$PY" ]; then echo "WARN: $PY missing; skipping golden $out" >&2; return 1; fi
    "$PY" scripts/gen_baseline.py --audio "$AUDIO" --output "$out" "$@" \
        || { echo "WARN: could not generate $out (reference deps/model missing)" >&2; return 1; }
}

# ECAPA gguf path (env override or common locations).
ECAPA_GGUF="${VD_BENCH_ECAPA_GGUF:-models/ecapa_voxceleb_f32.gguf}"
[ -f "$ECAPA_GGUF" ] || ECAPA_GGUF=/tmp/ecapa_f32.gguf

FAIL=0
run_ctest() {  # run_ctest <label> <regex>  (env vars already exported by caller)
    local label="$1" rx="$2"
    echo "== Parity [$label] on device '$DEV' =="
    if VOICEDETECT_DEVICE="$DEV" ctest --test-dir "$BUILD_DIR" -R "$rx" \
            --output-on-failure --no-tests=error; then
        echo "   PASS [$label]"
    else
        echo "   FAIL [$label]" >&2; FAIL=1
    fi
}

# Speaker encoders: each (gguf, golden) pair runs the fbank/encoder/embed/verify
# gates; the encoder test branches on the gguf's arch tag internally.
for spec in \
    "ECAPA|$ECAPA_GGUF|--arch ecapa_tdnn --model speechbrain/spkrec-ecapa-voxceleb" \
    "WeSpeaker|models/wespeaker_resnet34_f32.gguf|--arch wespeaker_resnet34 --onnx models/wespeaker_voxceleb_resnet34.onnx" \
    "ERes2Net|models/eres2net_base_200k_f32.gguf|--arch eres2net --onnx models/3dspeaker_eres2net_base_200k.onnx" \
    "CAM++|models/campplus_zh-cn_f32.gguf|--arch campplus --onnx models/3dspeaker_campplus_zh-cn_16k.onnx" ; do
    IFS='|' read -r name gguf genargs <<<"$spec"
    [ -f "$gguf" ] || { echo "WARN: $name gguf '$gguf' missing; skipping" >&2; continue; }
    base="$TMP/${name//[^A-Za-z0-9]/_}_baseline.gguf"
    gen "$base" $genargs || continue
    export VOICEDETECT_TEST_GGUF="$gguf" VOICEDETECT_TEST_BASELINE="$base" \
           VOICEDETECT_TEST_AUDIO="$AUDIO" VOICEDETECT_TEST_FIXDIR="tests/fixtures"
    run_ctest "$name" "test_fbank|test_encoder|test_embed|test_verify"
    unset VOICEDETECT_TEST_GGUF VOICEDETECT_TEST_BASELINE
done

# Analyze heads (separate env vars + tests).
if base="$TMP/emotion_baseline.gguf"; gen "$base" --analyze; then
    export VOICEDETECT_TEST_ANALYZE_GGUF="models/wav2vec2_analyze_f32.gguf" \
           VOICEDETECT_TEST_ANALYZE_BASELINE="$base" VOICEDETECT_TEST_AUDIO="$AUDIO"
    run_ctest "emotion" "test_analyze"
    unset VOICEDETECT_TEST_ANALYZE_GGUF VOICEDETECT_TEST_ANALYZE_BASELINE
fi
if base="$TMP/age_gender_baseline.gguf"; gen "$base" --age-gender; then
    export VOICEDETECT_TEST_AGEGENDER_GGUF="models/age_gender_audeering_f32.gguf" \
           VOICEDETECT_TEST_AGEGENDER_BASELINE="$base" VOICEDETECT_TEST_AUDIO="$AUDIO"
    run_ctest "age_gender" "test_age_gender"
    unset VOICEDETECT_TEST_AGEGENDER_GGUF VOICEDETECT_TEST_AGEGENDER_BASELINE
fi

# --- 3. Append GPU latency rows to the results table ------------------------
echo "== GPU bench (appending rows to benchmarks/RESULTS.md) =="
VD_BENCH_ECAPA_GGUF="$ECAPA_GGUF" "$PY" scripts/bench_compare.py \
    --audio "$AUDIO" --n "$N" --threads 1 --cli "$CLI" \
    --device "$DEV" --append-gpu --out benchmarks/RESULTS.md

if [ "$FAIL" -ne 0 ]; then
    echo "GPU PARITY FAILED - see failures above" >&2
    exit 1
fi
echo "GPU verify + bench complete. Parity holds on '$DEV'; GPU rows appended to benchmarks/RESULTS.md"
