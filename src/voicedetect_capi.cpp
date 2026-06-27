#include "voicedetect_capi.h"
#include "model.hpp"      // vd::Model

#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <string>
#include <vector>

// ABI version. Bump on breaking changes to the signatures/semantics in
// include/voicedetect_capi.h. Additive changes (new functions) do not require a
// bump.
//
// v1: initial flat C-API - load/free/last_error/free_string, embed_path /
//     embed_pcm (+ free_vec), verify_paths, analyze_path_json.
#define VOICEDETECT_CAPI_ABI_VERSION 1

// The opaque context: a loaded model plus a buffer for the last error message.
struct voicedetect_ctx {
    std::unique_ptr<vd::Model> model;
    std::string last_error;
};

namespace {

// malloc a copy of `s` (NUL-terminated) so a C consumer frees it with free()
// (matching voicedetect_capi_free_string). Returns NULL on OOM.
char* dup_to_c(const std::string& s) {
    char* buf = static_cast<char*>(std::malloc(s.size() + 1));
    if (!buf) return nullptr;
    std::memcpy(buf, s.data(), s.size());
    buf[s.size()] = '\0';
    return buf;
}

// malloc a float array copy of `v`. Returns NULL on OOM or empty input.
float* dup_vec(const std::vector<float>& v) {
    if (v.empty()) return nullptr;
    float* buf = static_cast<float*>(std::malloc(v.size() * sizeof(float)));
    if (!buf) return nullptr;
    std::memcpy(buf, v.data(), v.size() * sizeof(float));
    return buf;
}

// Shared body for embed_path / embed_pcm: take an already-computed embedding and
// hand it to the caller via (out_vec, out_dim). Returns 0 on success.
int emit_embedding(voicedetect_ctx* ctx, const std::vector<float>& emb,
                   float** out_vec, int* out_dim) {
    float* buf = dup_vec(emb);
    if (!buf) { ctx->last_error = "out of memory"; return 1; }
    if (out_vec) *out_vec = buf; else { std::free(buf); }
    if (out_dim) *out_dim = (int)emb.size();
    ctx->last_error.clear();
    return 0;
}

} // namespace

extern "C" int voicedetect_capi_abi_version(void) {
    return VOICEDETECT_CAPI_ABI_VERSION;
}

extern "C" voicedetect_ctx* voicedetect_capi_load(const char* gguf_path) {
    if (!gguf_path) return nullptr;
    try {
        std::unique_ptr<vd::Model> model = vd::Model::load(gguf_path);
        if (!model) return nullptr;  // load failure (bad/missing GGUF)
        auto* ctx = new (std::nothrow) voicedetect_ctx();
        if (!ctx) return nullptr;
        ctx->model = std::move(model);
        return ctx;
    } catch (...) {
        // Never let an exception cross the boundary.
        return nullptr;
    }
}

extern "C" void voicedetect_capi_free(voicedetect_ctx* ctx) {
    delete ctx;  // safe on nullptr; ~unique_ptr releases the model.
}

extern "C" const char* voicedetect_capi_last_error(voicedetect_ctx* ctx) {
    if (!ctx) return "";
    return ctx->last_error.c_str();
}

extern "C" void voicedetect_capi_free_string(char* s) {
    std::free(s);
}

extern "C" void voicedetect_capi_free_vec(float* v) {
    std::free(v);
}

extern "C" int voicedetect_capi_embed_path(voicedetect_ctx* ctx, const char* wav_path,
                                           float** out_vec, int* out_dim) {
    if (out_vec) *out_vec = nullptr;
    if (out_dim) *out_dim = 0;
    if (!ctx) return 1;
    if (!ctx->model)  { ctx->last_error = "context has no loaded model"; return 1; }
    if (!wav_path)    { ctx->last_error = "wav_path is NULL"; return 1; }
    try {
        std::vector<float> emb = ctx->model->embed_path(wav_path);
        return emit_embedding(ctx, emb, out_vec, out_dim);
    } catch (const std::exception& e) {
        ctx->last_error = e.what();
        return 1;
    } catch (...) {
        ctx->last_error = "unknown error";
        return 1;
    }
}

extern "C" int voicedetect_capi_embed_pcm(voicedetect_ctx* ctx, const float* pcm,
                                          int n_samples, int sample_rate,
                                          float** out_vec, int* out_dim) {
    if (out_vec) *out_vec = nullptr;
    if (out_dim) *out_dim = 0;
    if (!ctx) return 1;
    if (!ctx->model)            { ctx->last_error = "context has no loaded model"; return 1; }
    if (!pcm || n_samples < 0)  { ctx->last_error = "invalid pcm buffer"; return 1; }
    try {
        std::vector<float> samples(pcm, pcm + n_samples);
        std::vector<float> emb = ctx->model->embed_pcm(samples, sample_rate);
        return emit_embedding(ctx, emb, out_vec, out_dim);
    } catch (const std::exception& e) {
        ctx->last_error = e.what();
        return 1;
    } catch (...) {
        ctx->last_error = "unknown error";
        return 1;
    }
}

extern "C" int voicedetect_capi_verify_paths(voicedetect_ctx* ctx, const char* a,
                                             const char* b, float threshold,
                                             float* out_distance, int* out_verified) {
    if (!ctx) return 1;
    if (!ctx->model) { ctx->last_error = "context has no loaded model"; return 1; }
    if (!a || !b)    { ctx->last_error = "clip path is NULL"; return 1; }
    try {
        float distance = 0.0f;
        bool  verified = false;
        ctx->model->verify_paths(a, b, threshold, distance, verified);
        if (out_distance) *out_distance = distance;
        if (out_verified) *out_verified = verified ? 1 : 0;
        ctx->last_error.clear();
        return 0;
    } catch (const std::exception& e) {
        ctx->last_error = e.what();
        return 1;
    } catch (...) {
        ctx->last_error = "unknown error";
        return 1;
    }
}

extern "C" char* voicedetect_capi_analyze_path_json(voicedetect_ctx* ctx,
                                                    const char* wav_path) {
    if (!ctx) return nullptr;
    if (!ctx->model) { ctx->last_error = "context has no loaded model"; return nullptr; }
    if (!wav_path)   { ctx->last_error = "wav_path is NULL"; return nullptr; }
    try {
        std::string json = ctx->model->analyze_path_json(wav_path);
        ctx->last_error.clear();
        char* out = dup_to_c(json);
        if (!out) { ctx->last_error = "out of memory"; return nullptr; }
        return out;
    } catch (const std::exception& e) {
        ctx->last_error = e.what();
        return nullptr;
    } catch (...) {
        ctx->last_error = "unknown error";
        return nullptr;
    }
}
