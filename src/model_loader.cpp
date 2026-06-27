#include "model_loader.hpp"
#include "common.hpp"
#include "backend.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "gguf.h"
#include <cstring>
#include <vector>
#include <utility>
namespace vd {

static uint32_t kv_u32(gguf_context* g, const char* k, uint32_t d=0){
    int64_t id = gguf_find_key(g,k); return id<0 ? d : (uint32_t)gguf_get_val_u32(g,id);
}
static float kv_f32(gguf_context* g, const char* k, float d=0){
    int64_t id = gguf_find_key(g,k); return id<0 ? d : gguf_get_val_f32(g,id);
}
static bool kv_bool(gguf_context* g, const char* k, bool d=false){
    int64_t id = gguf_find_key(g,k); return id<0 ? d : gguf_get_val_bool(g,id);
}
static std::string kv_str(gguf_context* g, const char* k, const char* d=""){
    int64_t id = gguf_find_key(g,k); return id<0 ? std::string(d) : std::string(gguf_get_val_str(g,id));
}
static std::vector<std::string> kv_str_arr(gguf_context* g, const char* k){
    std::vector<std::string> out;
    int64_t id = gguf_find_key(g,k);
    if(id>=0 && gguf_get_arr_type(g,id)==GGUF_TYPE_STRING){
        size_t n = gguf_get_arr_n(g,id);
        out.resize(n);
        for(size_t i=0;i<n;++i) out[i] = gguf_get_arr_str(g,id,i);
    }
    return out;
}
static std::vector<int32_t> kv_i32_arr(gguf_context* g, const char* k){
    std::vector<int32_t> out;
    int64_t id = gguf_find_key(g,k);
    if(id>=0 && gguf_get_arr_type(g,id)==GGUF_TYPE_INT32){
        size_t n = gguf_get_arr_n(g,id);
        const int32_t* d = (const int32_t*)gguf_get_arr_data(g,id);
        out.assign(d, d+n);
    }
    return out;
}

ModelLoader::~ModelLoader(){
    // Free the weight buffer BEFORE the ctxs. CPU path: a from_ptr buffer that
    // does NOT own its memory (ctx_ does). Device path: OWNS the device buffer.
    // ggml_backend_buffer_free handles both.
    if(weights_buf_){
        // Purge any persistent graph-cache entries that reference these weights
        // BEFORE the buffer is freed, so a later model reallocating this address
        // cannot false-hit a stale cached graph (multi-model hosting safety).
        invalidate_graph_cache_for_weights(weights_buf_);
        ggml_backend_buffer_free(weights_buf_);
    }
    if(device_ctx_) ggml_free(device_ctx_);
    if(gguf_) gguf_free(gguf_); if(ctx_) ggml_free(ctx_);
}

bool ModelLoader::realize_weights(ggml_backend_t backend){
    if(weights_buf_) return true;                       // idempotent
    if(!backend || !ctx_){ VD_LOG("realize_weights: null backend/ctx"); return false; }

    // Backend-agnostic CPU check: ggml_backend_is_cpu() lives in the ggml-cpu
    // module (a dlopen'd MODULE under GGML_BACKEND_DL, not linkable here), so
    // query the device type through the generic backend API instead.
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    const bool is_cpu = dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
    if (is_cpu) {
        // Fast path: borrow the host ctx memory directly (no copy). The GGUF is
        // loaded with no_alloc=false, so every tensor's data lives in one
        // contiguous ctx mem_buffer; wrap that exact memory as a CPU backend
        // buffer and point every tensor's ->buffer at it, so graphs can reference
        // the loader tensors DIRECTLY as leaves.
        void*  base = ggml_get_mem_buffer(ctx_);
        size_t size = ggml_get_mem_size(ctx_);
        weights_buf_ = ggml_backend_cpu_buffer_from_ptr(base, size);
        if(!weights_buf_){ VD_LOG("realize_weights: buffer_from_ptr failed"); return false; }
        for(auto& kv : tensors_) kv.second->buffer = weights_buf_;
        return true;
    }

    // Device path (CUDA/Metal/Vulkan/...): mirror every weight into a no_alloc
    // ctx, allocate THAT on the backend, upload each tensor's bytes from the host
    // source, and repoint the name->tensor map at the device tensors. ctx_ stays
    // alive as the host source.
    const size_t n = tensors_.size();
    struct ggml_init_params dp = {
        /*.mem_size  =*/ ggml_tensor_overhead() * (n + 8),
        /*.mem_buffer=*/ nullptr,
        /*.no_alloc  =*/ true,
    };
    device_ctx_ = ggml_init(dp);
    if(!device_ctx_){ VD_LOG("realize_weights: device ctx init failed"); return false; }

    std::vector<std::pair<ggml_tensor*, const void*>> ups; ups.reserve(n);
    std::unordered_map<std::string, ggml_tensor*> devmap; devmap.reserve(n);
    for (auto& kv : tensors_) {
        ggml_tensor* s = kv.second;
        ggml_tensor* d = ggml_new_tensor(device_ctx_, s->type, GGML_MAX_DIMS, s->ne);
        ggml_set_name(d, kv.first.c_str());
        devmap.emplace(kv.first, d);
        ups.emplace_back(d, s->data);
    }
    weights_buf_ = ggml_backend_alloc_ctx_tensors(device_ctx_, backend);
    if(!weights_buf_){ VD_LOG("realize_weights: alloc_ctx_tensors failed"); return false; }
    for (auto& pr : ups)
        ggml_backend_tensor_set(pr.first, pr.second, 0, ggml_nbytes(pr.first));
    tensors_.swap(devmap);
    return true;
}

bool ModelLoader::load(const std::string& path){
    struct gguf_init_params p{ /*no_alloc*/false, /*ctx*/&ctx_ };
    gguf_ = gguf_init_from_file(path.c_str(), p);
    if(!gguf_){ VD_LOG("gguf open failed: %s", path.c_str()); return false; }
    cfg_.arch          = kv_str(gguf_, "voicedetect.arch");
    cfg_.embedding_dim = kv_u32(gguf_, "voicedetect.embedding_dim");
    cfg_.l2_normalize  = kv_bool(gguf_, "voicedetect.l2_normalize", true);
    // FBank front end
    cfg_.sample_rate   = kv_u32(gguf_, "voicedetect.fbank.sample_rate", 16000);
    cfg_.n_mels        = kv_u32(gguf_, "voicedetect.fbank.n_mels", 80);
    cfg_.n_fft         = kv_u32(gguf_, "voicedetect.fbank.n_fft", 512);
    cfg_.win_length    = kv_u32(gguf_, "voicedetect.fbank.win_length", 400);
    cfg_.hop_length    = kv_u32(gguf_, "voicedetect.fbank.hop_length", 160);
    cfg_.preemph       = kv_f32(gguf_, "voicedetect.fbank.preemph", 0.97f);
    cfg_.fbank_low_freq  = kv_f32(gguf_, "voicedetect.fbank.low_freq", 20.0f);
    cfg_.fbank_high_freq = kv_f32(gguf_, "voicedetect.fbank.high_freq", 0.0f);
    cfg_.fbank_use_energy = kv_bool(gguf_, "voicedetect.fbank.use_energy", false);
    cfg_.fbank_cmn        = kv_bool(gguf_, "voicedetect.fbank.cmn", true);
    cfg_.fbank_window     = kv_str(gguf_, "voicedetect.fbank.window", "povey");
    // WeSpeaker ResNet34 block manifest (only present for that arch).
    if(cfg_.arch == "wespeaker_resnet34"){
        VoiceDetectConfig::ResNetConfig& r = cfg_.resnet;
        r.stem_weight  = kv_str(gguf_, "voicedetect.resnet.stem_weight");
        r.stem_bias    = kv_str(gguf_, "voicedetect.resnet.stem_bias");
        r.conv1_weight = kv_str_arr(gguf_, "voicedetect.resnet.conv1_weight");
        r.conv1_bias   = kv_str_arr(gguf_, "voicedetect.resnet.conv1_bias");
        r.conv2_weight = kv_str_arr(gguf_, "voicedetect.resnet.conv2_weight");
        r.conv2_bias   = kv_str_arr(gguf_, "voicedetect.resnet.conv2_bias");
        r.down_weight  = kv_str_arr(gguf_, "voicedetect.resnet.down_weight");
        r.down_bias    = kv_str_arr(gguf_, "voicedetect.resnet.down_bias");
        r.stride       = kv_i32_arr(gguf_, "voicedetect.resnet.stride");
        r.seg_weight   = kv_str(gguf_, "voicedetect.resnet.seg_weight");
        r.seg_bias     = kv_str(gguf_, "voicedetect.resnet.seg_bias");
        r.mean_vec     = kv_str(gguf_, "voicedetect.resnet.mean_vec");
        r.var_eps      = kv_f32(gguf_, "voicedetect.resnet.var_eps", 1e-8f);
    }
    // 3D-Speaker ERes2Net manifest (only present for that arch).
    if(cfg_.arch == "eres2net"){
        VoiceDetectConfig::ERes2NetConfig& e = cfg_.eres2net;
        e.conv_weight = kv_str_arr(gguf_, "voicedetect.eres2net.conv_weight");
        e.conv_bias   = kv_str_arr(gguf_, "voicedetect.eres2net.conv_bias");
        e.conv_stride = kv_i32_arr(gguf_, "voicedetect.eres2net.conv_stride");
        e.num_blocks  = kv_i32_arr(gguf_, "voicedetect.eres2net.num_blocks");
        e.m_channels  = kv_u32(gguf_, "voicedetect.eres2net.m_channels");
        e.scale       = kv_u32(gguf_, "voicedetect.eres2net.scale", 2);
        e.seg_weight  = kv_str(gguf_, "voicedetect.eres2net.seg_weight");
        e.seg_bias    = kv_str(gguf_, "voicedetect.eres2net.seg_bias");
        e.relu_clamp  = kv_f32(gguf_, "voicedetect.eres2net.relu_clamp", 20.0f);
        e.var_eps     = kv_f32(gguf_, "voicedetect.eres2net.var_eps", 1e-8f);
    }
    // 3D-Speaker CAM++ manifest (only present for that arch).
    if(cfg_.arch == "campplus"){
        VoiceDetectConfig::CamPPlusConfig& cp = cfg_.campplus;
        cp.conv_weight = kv_str_arr(gguf_, "voicedetect.campplus.conv_weight");
        cp.conv_bias   = kv_str_arr(gguf_, "voicedetect.campplus.conv_bias");
        cp.bn_prefix   = kv_str_arr(gguf_, "voicedetect.campplus.bn_prefix");
        cp.emb_bn_mean = kv_str(gguf_, "voicedetect.campplus.emb_bn_mean");
        cp.emb_bn_var  = kv_str(gguf_, "voicedetect.campplus.emb_bn_var");
        cp.bn_eps      = kv_f32(gguf_, "voicedetect.campplus.bn_eps", 1e-5f);
    }

    // Analyze heads (phased; absent -> present=false, engine skips analyze).
    cfg_.analyze_present = kv_bool(gguf_, "voicedetect.analyze.present", false);
    if(cfg_.analyze_present){
        cfg_.emotion_labels = kv_str_arr(gguf_, "voicedetect.analyze.emotion_labels");
        cfg_.gender_labels  = kv_str_arr(gguf_, "voicedetect.analyze.gender_labels");
    }
    // wav2vec2 analyze config (only present for the analyze GGUF). All dims/kernels/
    // strides live in KV so the C++ analyze graph carries no magic numbers. Shared
    // by the emotion (base) and age/gender (large-robust) analyze archs.
    if(cfg_.arch == "wav2vec2_emotion" || cfg_.arch == "wav2vec2_age_gender"){
        VoiceDetectConfig::W2V2Config& w = cfg_.w2v2;
        w.hidden_size     = kv_u32(gguf_, "voicedetect.w2v2.hidden_size");
        w.n_layers        = kv_u32(gguf_, "voicedetect.w2v2.n_layers");
        w.n_heads         = kv_u32(gguf_, "voicedetect.w2v2.n_heads");
        w.ff_dim          = kv_u32(gguf_, "voicedetect.w2v2.ff_dim");
        w.num_conv_layers = kv_u32(gguf_, "voicedetect.w2v2.num_conv_layers");
        w.conv_dims       = kv_i32_arr(gguf_, "voicedetect.w2v2.conv_dims");
        w.conv_kernels    = kv_i32_arr(gguf_, "voicedetect.w2v2.conv_kernels");
        w.conv_strides    = kv_i32_arr(gguf_, "voicedetect.w2v2.conv_strides");
        w.feat_extract_norm       = kv_str(gguf_, "voicedetect.w2v2.feat_extract_norm");
        w.conv_bias               = kv_bool(gguf_, "voicedetect.w2v2.conv_bias", false);
        w.do_normalize            = kv_bool(gguf_, "voicedetect.w2v2.do_normalize", false);
        w.feat_extract_activation = kv_str(gguf_, "voicedetect.w2v2.feat_extract_activation");
        w.hidden_act              = kv_str(gguf_, "voicedetect.w2v2.hidden_act");
        w.do_stable_layer_norm    = kv_bool(gguf_, "voicedetect.w2v2.do_stable_layer_norm", false);
        w.layer_norm_eps          = kv_f32(gguf_, "voicedetect.w2v2.layer_norm_eps", 1e-5f);
        w.num_conv_pos_embeddings = kv_u32(gguf_, "voicedetect.w2v2.num_conv_pos_embeddings");
        w.num_conv_pos_embedding_groups =
            kv_u32(gguf_, "voicedetect.w2v2.num_conv_pos_embedding_groups");
        w.pos_conv_weight_norm_dim = kv_u32(gguf_, "voicedetect.w2v2.pos_conv_weight_norm_dim", 2);
        w.use_weighted_layer_sum   = kv_bool(gguf_, "voicedetect.w2v2.use_weighted_layer_sum", false);
        w.classifier_proj_size     = kv_u32(gguf_, "voicedetect.w2v2.classifier_proj_size");
    }
    // tensors (verbatim names)
    const int64_t nt = gguf_get_n_tensors(gguf_);
    for(int64_t i=0;i<nt;++i){ const char* nm = gguf_get_tensor_name(gguf_,i);
        ggml_tensor* t = ggml_get_tensor(ctx_, nm); if(t) tensors_[nm]=t; }
    // The analyze GGUF carries no embedding head (no voicedetect.embedding_dim);
    // it is valid iff the analyze schema is present.
    return cfg_.embedding_dim>0 || cfg_.analyze_present;
}

ggml_tensor* ModelLoader::tensor(const std::string& n) const {
    auto it = tensors_.find(n); return it==tensors_.end()? nullptr : it->second;
}

} // namespace vd
