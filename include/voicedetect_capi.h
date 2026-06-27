#ifndef VOICEDETECT_CAPI_H
#define VOICEDETECT_CAPI_H

#ifdef __cplusplus
extern "C" {
#endif

// Flat C-API for voice-detect.cpp - designed for dlopen / cgo / purego (LocalAI).
//
// All functions are extern "C" and never let a C++ exception cross the
// boundary. A speaker-recognition model is loaded ONCE into an opaque
// `voicedetect_ctx` and reused across embed/verify/analyze calls. Returned
// strings are malloc'd UTF-8 owned by the caller and must be released with
// voicedetect_capi_free_string; returned float vectors with
// voicedetect_capi_free_vec.
//
// Pipeline: decode/resample audio -> 16 kHz mono -> 80-dim Kaldi-compatible
// FBank features -> speaker encoder -> L2-normalized embedding. `verify`
// compares two clips by cosine distance against a threshold; `analyze` runs the
// (phased) age/gender/emotion heads.

// Opaque speaker-recognition context (wraps a loaded model + last-error buffer).
typedef struct voicedetect_ctx voicedetect_ctx;

// ABI version of this header/implementation. Bump on any breaking change to the
// function signatures or semantics below. Additive changes (new functions) are
// fine without a bump; LocalAI checks this integer for compatibility.
//
// v1: initial flat C-API surface - load/free/last_error/free_string,
//     embed_path / embed_pcm (+ free_vec), verify_paths, analyze_path_json.
int voicedetect_capi_abi_version(void);

// Load a GGUF speaker-recognition model. Returns an owning context, or NULL on
// failure (bad/missing GGUF). The returned context must be released with
// voicedetect_capi_free.
voicedetect_ctx* voicedetect_capi_load(const char* gguf_path);

// Free a context obtained from voicedetect_capi_load. Safe on NULL.
void voicedetect_capi_free(voicedetect_ctx* ctx);

// Human-readable description of the last error on `ctx`, or "" if none. The
// returned pointer is owned by the context and valid until the next call on it
// (or until voicedetect_capi_free). Returns "" if `ctx` is NULL.
const char* voicedetect_capi_last_error(voicedetect_ctx* ctx);

// Free a string previously returned by voicedetect_capi_analyze_path_json (or
// any other char*-returning entry point). Safe on NULL.
void voicedetect_capi_free_string(char* s);

// Free a float vector previously returned via voicedetect_capi_embed_* out_vec.
// Safe on NULL.
void voicedetect_capi_free_vec(float* v);

// Compute the L2-normalized speaker embedding for a WAV file. On success returns
// 0 and sets `*out_vec` to a malloc'd array of `*out_dim` floats (release with
// voicedetect_capi_free_vec). On error returns nonzero, leaves `*out_vec` NULL /
// `*out_dim` 0, and sets the context's last error (see
// voicedetect_capi_last_error).
int voicedetect_capi_embed_path(voicedetect_ctx* ctx, const char* wav_path,
                                float** out_vec, int* out_dim);

// Like voicedetect_capi_embed_path but from in-memory mono float PCM
// (`pcm`, length `n_samples`). If `sample_rate != 16000` the audio is linearly
// resampled to 16 kHz first. Same ownership/error contract as embed_path.
int voicedetect_capi_embed_pcm(voicedetect_ctx* ctx, const float* pcm,
                               int n_samples, int sample_rate,
                               float** out_vec, int* out_dim);

// Verify whether two clips are the same speaker. Embeds `a` and `b`, computes
// the cosine distance (1 - cosine_similarity) between their L2-normalized
// embeddings, writes it to `*out_distance`, and sets `*out_verified` to 1 iff
// the distance is <= `threshold` (same speaker), else 0. On success returns 0;
// on error returns nonzero, leaves the out params unchanged, and sets the
// context's last error.
int voicedetect_capi_verify_paths(voicedetect_ctx* ctx, const char* a,
                                  const char* b, float threshold,
                                  float* out_distance, int* out_verified);

// Analyze a WAV file with the (phased) age/gender/emotion heads, returning a
// malloc'd UTF-8 JSON document (free with voicedetect_capi_free_string). The
// shape is:
//
//   {"age":42.0,
//    "gender":{"label":"female","female":0.88,"male":0.12},
//    "emotion":{"label":"neutral","scores":{"neutral":0.7, ...}}}
//
// On error returns NULL and sets the context's last error.
char* voicedetect_capi_analyze_path_json(voicedetect_ctx* ctx,
                                         const char* wav_path);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // VOICEDETECT_CAPI_H
