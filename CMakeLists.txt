cmake_minimum_required(VERSION 3.18)
project(voicedetect_cpp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(VOICEDETECT_BUILD_TESTS "Build ctest targets" OFF)
option(VOICEDETECT_BUILD_CLI   "Build voicedetect-cli" ON)
option(VOICEDETECT_SHARED      "Build libvoicedetect as shared" OFF)
option(VOICEDETECT_GGML_CUDA   "Forward GGML_CUDA" OFF)
option(VOICEDETECT_GGML_CUDNN  "Use cuDNN for conv2d on CUDA (implicit-GEMM, no im2col spill)" OFF)
option(VOICEDETECT_GGML_METAL  "Forward GGML_METAL" OFF)
option(VOICEDETECT_GGML_VULKAN "Forward GGML_VULKAN" OFF)
option(VOICEDETECT_GGML_HIP    "Forward GGML_HIP (ROCm)" OFF)

# Multi-ISA CPU backend with RUNTIME dispatch. ggml can build one CPU backend
# per x86 micro-arch (sse42/sandybridge/haswell/skylakex/icelake/zen4/...) as
# dlopen'd modules and pick the best one the host actually supports AT LOAD TIME.
# That lets a SINGLE portable binary use AVX512 on capable CPUs (e.g. Zen4 /
# Sapphire Rapids) without -march=native: GGML_NATIVE both downclocks AVX512
# parts under sustained wide-SIMD and bakes in the build host's ISA, which is
# wrong for a shipped binary. The GEMM-heavy wav2vec2 analyze models benefit
# most; the dispatcher is parity-exact (same kernels, wider SIMD lane width).
#
# Mechanics (ggml): GGML_CPU_ALL_VARIANTS requires GGML_BACKEND_DL, which in
# turn requires BUILD_SHARED_LIBS and is mutually exclusive with GGML_NATIVE.
# The CPU variants are emitted as libggml-cpu-<tag>.so MODULE libraries next to
# the executables; ggml_backend_load_all() (called from vd::Backend) discovers
# and registers the best supported one at startup.
#
# WHY DEFAULT OFF (GATE decision, not a free win). The dispatcher is numerically
# parity-exact, but selecting the AVX512 variant at runtime is NOT a uniform
# speedup. On wide-SIMD-capable CPUs the AVX512 cores DOWNCLOCK under sustained
# 512-bit work, and that hurts the bandwidth-bound encoders far more than the
# wider lanes help: the ERes2Net 2D-conv encoder measured a robust REGRESSION of
# +21.9% wall time at 8 threads when pinned to the AVX512 variant vs the AVX2
# (haswell) variant. The GEMM-heavy wav2vec2 analyze heads are the opposite case
# and do benefit. A blanket "always pick the widest ISA" policy therefore loses
# on the conv encoders, so this stays OPT-IN and the default static AVX2 build is
# unchanged.
#
# FOLLOW-UP REQUIRED to ship this enabled (NOT done here):
#   1. Per-model ISA gating: pin the bandwidth-bound 2D-conv encoders
#      (ERes2Net, and any conv-dominated speaker encoder) to the AVX2/haswell
#      variant at >=8 threads, and enable the AVX512/zen4/icelake variant only
#      for the GEMM-bound wav2vec2 analyze heads. ggml_backend_load_all() today
#      registers a SINGLE best CPU variant process-wide; per-model selection
#      needs either loading specific variant modules and binding each model's
#      Backend to the chosen one, or a thread-count-aware override.
#   2. Co-packaging the variant .so files: the libggml-cpu-<tag>.so MODULEs must
#      be shipped next to the consuming binary / inside the LocalAI backend OCI
#      image and found via get_executable_path(); the frozen C-ABI static build
#      ships none of these today.
# Until both land, leave this OFF.
option(VOICEDETECT_CPU_ALL_VARIANTS
       "Build all x86 CPU backend variants as dlopen modules + runtime-dispatch (portable AVX512)" OFF)

if(VOICEDETECT_CPU_ALL_VARIANTS)
  set(BUILD_SHARED_LIBS      ON  CACHE BOOL "" FORCE)  # required by GGML_BACKEND_DL
  set(GGML_BACKEND_DL        ON  CACHE BOOL "" FORCE)
  set(GGML_CPU_ALL_VARIANTS  ON  CACHE BOOL "" FORCE)
  # GGML_NATIVE is incompatible with GGML_BACKEND_DL (and is what downclocks /
  # hard-codes the build host's ISA). Force it OFF here so the guard below is a
  # no-op and the portable multi-variant build is produced.
  set(GGML_NATIVE            OFF CACHE BOOL "" FORCE)
  # Colocate every executable AND the variant MODULE .so in one bin dir so
  # ggml_backend_load_all()'s get_executable_path() search finds the variants
  # regardless of the test working directory.
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()

set(GGML_CUDA    ${VOICEDETECT_GGML_CUDA}   CACHE BOOL "" FORCE)
set(GGML_METAL   ${VOICEDETECT_GGML_METAL}  CACHE BOOL "" FORCE)
set(GGML_VULKAN  ${VOICEDETECT_GGML_VULKAN} CACHE BOOL "" FORCE)
set(GGML_HIP     ${VOICEDETECT_GGML_HIP}    CACHE BOOL "" FORCE)

# CUDA graphs: ggml defaults this OFF, but on CUDA builds capturing/replaying the
# compute graph is a small free speedup (never negative). Turn it on whenever we
# forward CUDA. The runtime kill-switch GGML_CUDA_DISABLE_GRAPHS=1 still works.
if(VOICEDETECT_GGML_CUDA)
  set(GGML_CUDA_GRAPHS ON CACHE BOOL "" FORCE)
endif()

# cuDNN conv path. When VOICEDETECT_GGML_CUDNN is on we (a) forward
# GGML_CUDA_USE_CUDNN so ggml-cuda compiles + links the cuDNN implicit-GEMM
# conv (src/ggml-cuda/cudnn-conv.cu) and (b) define VOICEDETECT_GGML_CUDNN for
# libvoicedetect so conv2d_auto emits GGML_OP_CONV_2D (ggml_conv_2d_direct) on
# CUDA instead of im2col+SGEMM, which ggml-cuda then routes to cuDNN. Requires
# CUDA. Runtime A/B: VD_CONV2D=im2col (never emit the op) / GGML_CUDA_USE_CUDNN=0
# (force the native fallback inside ggml-cuda).
if(VOICEDETECT_GGML_CUDNN)
  if(NOT VOICEDETECT_GGML_CUDA)
    message(FATAL_ERROR "VOICEDETECT_GGML_CUDNN requires VOICEDETECT_GGML_CUDA=ON")
  endif()
  set(GGML_CUDA_USE_CUDNN ON CACHE BOOL "" FORCE)
endif()

# Performance: -march=native (GGML_NATIVE) and tinyBLAS SGEMM (GGML_LLAMAFILE)
# give meaningful free speedups. Force them on unless the caller overrides.
if(NOT DEFINED GGML_NATIVE)
  set(GGML_NATIVE ON CACHE BOOL "ggml: optimize the build for the current system" FORCE)
endif()
if(NOT DEFINED GGML_LLAMAFILE)
  set(GGML_LLAMAFILE ON CACHE BOOL "ggml: use LLAMAFILE (tinyBLAS SGEMM)" FORCE)
endif()

# Apply any in-tree ggml patches before configuring the submodule so the patched
# source is compiled in. Idempotent: already-applied patches are silently
# skipped. voice-detect.cpp ships no patches today; the hook is kept so a future
# ggml fix can be dropped into third_party/ggml-patches/ without a CMake change.
find_program(BASH_EXECUTABLE bash)
if(BASH_EXECUTABLE AND EXISTS "${CMAKE_SOURCE_DIR}/scripts/apply_ggml_patches.sh"
   AND EXISTS "${CMAKE_SOURCE_DIR}/third_party/ggml-patches")
  execute_process(
    COMMAND "${BASH_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/scripts/apply_ggml_patches.sh"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    RESULT_VARIABLE _patch_result
    OUTPUT_VARIABLE _patch_output
    ERROR_VARIABLE  _patch_errors
  )
  if(_patch_output)
    message(STATUS "${_patch_output}")
  endif()
  if(_patch_result)
    message(WARNING "apply_ggml_patches.sh returned ${_patch_result}: ${_patch_errors}")
  endif()
endif()

add_subdirectory(third_party/ggml)

set(VOICEDETECT_SRC
    src/voicedetect.cpp
    src/voicedetect_capi.cpp
    src/model.cpp
    src/model_loader.cpp
    src/backend.cpp
    src/winograd.cpp
    src/directconv.cpp
    src/encoder.cpp
    src/encoder_resnet.cpp
    src/encoder_eres2net.cpp
    src/encoder_campplus.cpp
    src/analyze_w2v2.cpp
    src/fbank.cpp
    src/audio_io.cpp
    src/common.cpp)

if(VOICEDETECT_SHARED)
  add_library(voicedetect SHARED ${VOICEDETECT_SRC})
  # The C-API header marks symbols with plain extern "C", no __declspec(dllexport).
  # On MSVC that yields a DLL with no exported symbols (and no import .lib), which
  # is useless for dlopen/ctypes consumers. Auto-export every symbol so the DLL is
  # usable the same way the ELF/Mach-O shared libs are. No effect on Linux/macOS.
  set_target_properties(voicedetect PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
else()
  add_library(voicedetect STATIC ${VOICEDETECT_SRC})
endif()
target_include_directories(voicedetect PUBLIC include PRIVATE src ${CMAKE_SOURCE_DIR}/third_party)
target_compile_definitions(voicedetect PUBLIC $<$<CXX_COMPILER_ID:MSVC>:_USE_MATH_DEFINES>)
target_link_libraries(voicedetect PUBLIC ggml)
if(VOICEDETECT_GGML_CUDNN)
  target_compile_definitions(voicedetect PRIVATE VOICEDETECT_GGML_CUDNN)
endif()

# Winograd's hot GEMM uses AVX2 intrinsics. The production build sets
# GGML_NATIVE=OFF, so the library target is NOT compiled with -march=native;
# enable AVX2+FMA for just winograd.cpp when the host compiler supports them
# (the file falls back to a scalar path otherwise).
#
# AVX-512: winograd.cpp also carries a zmm winograd-domain GEMM microkernel. It is
# RUNTIME-CPUID-DISPATCHED, not gated on a global flag: the zmm code lives in
# functions tagged __attribute__((target("avx512f,avx512bw,avx512vl")))
# (GCC/Clang function multiversioning) and runs ONLY when __builtin_cpu_supports
# reports AVX-512 at run time; AVX2-only CPUs take the AVX2 microkernel. We
# deliberately DO NOT pass -mavx512f to the whole translation unit -- that would
# let the compiler hoist zmm into the always-run code and SIGILL on an AVX2-only
# host. So the default portable build is one binary that uses AVX-512 where
# present and AVX2 elsewhere, and ggml's CPU backend stays AVX2 (GGML_NATIVE=OFF).
# VOICEDETECT_WINO_AVX512 is set only when the compiler can emit such a
# per-function AVX-512 path without a global flag (probed below); otherwise the
# file compiles AVX2-only. Mirrors face-detect.cpp's CMake probe (commit 7ae5c4d).
include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)
check_cxx_compiler_flag("-mavx2" VOICEDETECT_HAS_AVX2)
if(VOICEDETECT_HAS_AVX2)
  set_source_files_properties(src/winograd.cpp src/directconv.cpp PROPERTIES
      COMPILE_OPTIONS "-mavx2;-mfma")
  # Probe function-multiversioning: can the compiler build an AVX-512 function via
  # a per-function target attribute (using the masked load/store the microkernel
  # needs) WITHOUT -mavx512f on the command line? GCC/Clang headers force-enable
  # the ISA for such functions, so the intrinsics are available even though the TU
  # is AVX2-only. Only winograd.cpp gets the AVX2 base flags here, never -mavx512f.
  set(CMAKE_REQUIRED_FLAGS "-mavx2 -mfma")
  check_cxx_source_compiles("
    #include <immintrin.h>
    __attribute__((target(\"avx512f,avx512bw,avx512vl\")))
    static void k(const float* u, const float* v, float* m, int oc) {
        __mmask16 mask = (__mmask16)((1u << (oc & 15)) - 1u);
        __m512 a = _mm512_maskz_loadu_ps(mask, u);
        a = _mm512_fmadd_ps(a, _mm512_set1_ps(v[0]), a);
        _mm512_mask_storeu_ps(m, mask, a);
        _mm512_storeu_ps(m, _mm512_loadu_ps(u));
    }
    int main() { float u[16]={0}, v[1]={0}, m[16]; k(u, v, m, 7); return (int)m[0]; }
  " VOICEDETECT_WINO_AVX512_DISPATCH)
  unset(CMAKE_REQUIRED_FLAGS)
  if(VOICEDETECT_WINO_AVX512_DISPATCH)
    set_source_files_properties(src/winograd.cpp PROPERTIES
        COMPILE_DEFINITIONS "VOICEDETECT_WINO_AVX512=1")
    # directconv.cpp carries the same runtime-CPUID-dispatched zmm path (function
    # multiversioning, no global -mavx512f); reuse the same probe result.
    set_source_files_properties(src/directconv.cpp PROPERTIES
        COMPILE_DEFINITIONS "VOICEDETECT_DCONV_AVX512=1")
  endif()
endif()

if(VOICEDETECT_BUILD_CLI)
  add_subdirectory(examples/cli)
endif()
if(VOICEDETECT_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
