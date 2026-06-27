# AGENTS.md

Durable reference for humans and agents maintaining voice-detect.cpp.

## AI-assisted contributions

This project follows the Linux kernel project's
[guidelines for AI coding assistants](https://docs.kernel.org/process/coding-assistants.html)
(the same policy LocalAI uses). Key rules for commits:

- **No `Signed-off-by` from AI.** Only a human submitter may sign off on the
  Developer Certificate of Origin.
- **No `Co-Authored-By: <AI>` trailers.** The human contributor owns the change.
- **Use an `Assisted-by:` trailer** to attribute AI involvement. Format:
  `Assisted-by: AGENT_NAME:MODEL_VERSION [TOOL1] [TOOL2]`
  (e.g. `Assisted-by: Claude:claude-opus-4-8 [Claude Code]`).
- The human submitter is responsible for reviewing, testing, and understanding
  every line of generated code.

## What this project is

voice-detect.cpp is a C++17/ggml inference engine for speaker recognition (voice
biometrics). A Python converter turns a speaker-embedding checkpoint (SpeechBrain
ECAPA-TDNN, or an ONNX WeSpeaker / 3D-Speaker encoder) into a metadata-driven
GGUF; the C++ model loader + encoder run the same computation natively with no
Python dependency at inference time.

Pipeline: decode/resample audio -> 16 kHz mono -> 80-dim Kaldi-compatible FBank
features -> speaker encoder -> L2-normalized embedding. Plus `verify` (cosine
distance vs threshold) and `analyze` (age/gender/emotion, phased last).

The public surface ships as a flat C-API (`include/voicedetect_capi.h` +
`libvoicedetect.so`) suitable for `dlopen`/FFI, and is the native replacement for
LocalAI's Python `speaker-recognition` backend.

## Performance invariants (do not regress)

These mirror parakeet.cpp's measured wins; keep them when implementing graphs:

- **Keep the persistent `ggml_gallocr`** in `src/backend.cpp`. Reusing one
  allocator across the many small per-utterance graphs (no per-call alloc/free)
  is the core throughput lever. The scheduler is used ONLY as a per-graph
  fallback when the active GPU backend lacks a kernel for some op.
- **Zero-copy weights.** `clone_weight` returns loader tensors directly so the
  same device buffer is reused every call; do not copy weights per call.

## Repository layout

```
include/             public C/C++ headers
                       voicedetect.h        , version + tiny C++ namespace vd layer
                       voicedetect_capi.h   , flat C-API for FFI / dlopen
src/                 libvoicedetect implementation
                       voicedetect.cpp      , version + thin vd::embed/verify wrappers
                       voicedetect_capi.cpp , flat C-API implementation
                       model.hpp/cpp        , load-once vd::Model orchestration (stub)
                       model_loader.hpp/cpp , GGUF -> VoiceDetectConfig + name->tensor
                       backend.hpp/cpp      , persistent ggml_backend_t + gallocr
                       fbank.hpp/cpp        , Kaldi-compatible FBank front end (stub)
                       audio_io.hpp/cpp     , dr_wav load + linear resample to 16k
                       common.hpp/cpp       , logging helpers
examples/cli/        voicedetect-cli binary (info, embed, verify, analyze, bench)
scripts/             Python tooling (reference-side only)
                       convert_voicedetect_to_gguf.py , checkpoint -> GGUF (--dtype f32|f16|q8_0)
                       gen_baseline.py                , reference intermediates -> baseline.gguf
                       apply_ggml_patches.sh          , optional in-tree ggml patches
                       requirements.txt               , speechbrain/torch/torchaudio/onnxruntime/gguf
tests/               ctest targets
                       parity.hpp           , header-only golden compare + cosine
                       test_smoke.cpp       , version + ABI (model-independent)
                       test_fbank.cpp       , FBank golden parity (RC-77 skip)
                       fixtures/            , small clips (not committed; see README)
                       python/check_convert.py , converter round-trip
docs/
  conversion.md     , GGUF schema reference
  parity.md         , coverage matrix + parity gates (cosine>=0.9999, max|d|<=1e-3, identical verdict)
  quantization.md   , quantization allowlist + policy
third_party/         vendored deps: ggml/ (submodule), dr_wav.h (single header)
models/              output dir for converted GGUFs (gitignored; MANIFEST.md tracked)
.github/workflows/   ci.yml, docker.yml, release.yml
```

## Build

```
cmake -B build -DVOICEDETECT_BUILD_TESTS=ON -DGGML_NATIVE=ON && cmake --build build -j
```

Use `-DGGML_NATIVE=OFF` when building for CI or portable binaries.

## Running tests

```
# Model-independent (run anywhere)
ctest --test-dir build --output-on-failure -LE model    # test_smoke

# Model/baseline-dependent (need venv + reference baseline; SKIP/77 without)
export VOICEDETECT_TEST_GGUF=/tmp/model.gguf
export VOICEDETECT_TEST_BASELINE=/tmp/baseline.gguf
export VOICEDETECT_TEST_AUDIO=tests/fixtures/clip.wav
ctest --test-dir build --output-on-failure
```

## C-API and LocalAI integration

`include/voicedetect_capi.h` defines the flat C-API. Build `libvoicedetect.so`
with `-DVOICEDETECT_SHARED=ON`. Verify exports with
`nm -D build-shared/libvoicedetect.so | grep voicedetect_capi`.

The LocalAI backend lives in the LocalAI repo and dlopens `libvoicedetect.so`.
These are the **frozen symbols** LocalAI depends on - do not remove or change any
signature without a coordinated bump on the LocalAI side AND on
`voicedetect_capi_abi_version`:

```
voicedetect_capi_abi_version
voicedetect_capi_load
voicedetect_capi_free
voicedetect_capi_last_error
voicedetect_capi_free_string
voicedetect_capi_free_vec
voicedetect_capi_embed_path     # WAV path  -> L2-normalized embedding (out_vec,out_dim)
voicedetect_capi_embed_pcm      # raw f32 PCM (+ sample_rate) -> embedding
voicedetect_capi_verify_paths   # two clips  -> cosine distance + verdict vs threshold
voicedetect_capi_analyze_path_json  # WAV path -> {age, gender, emotion} JSON
```

### ABI version bump rule

`voicedetect_capi_abi_version()` returns an integer (currently **1**) that
LocalAI checks for compatibility. **Bump it on any breaking change** to the
signatures or semantics of the frozen symbols above (the changelog block lives at
the top of `include/voicedetect_capi.h` AND `src/voicedetect_capi.cpp` - keep
them in sync). **Additive changes** (brand-new functions that do not alter
existing ones) are fine **without** a bump. Never let a C++ exception cross the C
boundary: every entry point catches and routes to the context's last-error
buffer.

## GGUF schema

See `docs/conversion.md`. Quick summary:

- `general.architecture = "voicedetect"`
- All metadata keys use the `voicedetect.*` prefix.
- **Tensor names are verbatim source `state_dict` / ONNX initializer names**, no
  remapping. The C++ loader maps `name -> ggml_tensor*` by exact string. Never
  remap tensor names at conversion time.

## ggml submodule

Tracked at `third_party/ggml`. No local patches today; the
`scripts/apply_ggml_patches.sh` hook only runs if `third_party/ggml-patches/`
exists. To bump: update the submodule SHA, rebuild, run the tests, fix any API
breakage in `src/model_loader.cpp` / `src/backend.cpp`.

## Implementation status (what the stubs leave to fill in)

- `src/fbank.cpp` - `FBank::compute` is a documented stub; implement the
  Kaldi-compatible front end (parity vs `torchaudio.compliance.kaldi`).
- `src/model.cpp` - `embed_16k` throws "not implemented"; build the encoder
  graphs (ECAPA first), pooling, embedding projection + L2-normalize.
- `analyze` heads - phased last; `voicedetect.analyze.present` gates them.
- `scripts/convert_voicedetect_to_gguf.py` - `load_state_dict` raises
  NotImplementedError; wire the SpeechBrain + ONNX loader branches.
- `scripts/gen_baseline.py` - reference forward + tensor capture skeleton.
- `tests/fixtures/` - add real clips (with provenance) once the engine runs.
