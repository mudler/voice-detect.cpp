#include "analyze_w2v2.hpp"
#include "backend.hpp"
#include "model_loader.hpp"
#include "ggml.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace vd {

// Strided Conv1d, NO bias, NO padding (the wav2vec2 feature encoder). Decomposed
// as F32 im2col + mul_mat ourselves: ggml_conv_1d() hardcodes an F16 im2col that
// asserts an F16 kernel, but the converter keeps these conv weights F32. This is
// the identical decomposition the speaker encoders use (see encoder.cpp), with a
// real stride and pad 0.
//   w ne = [K, IC, OC]; x ne = [L, IC]. Returns [OL, OC] with
//   OL = (L - K)/stride + 1. The [OL=time, OC=channel] output is exactly the
//   [L', IC'] layout the next conv layer consumes, so layers chain with no
//   transpose.
//
// `channels_first` swaps the mul_mat operand order so the SAME products land as
// [OC, OL] (channel fastest) instead of [OL, OC]. mul_mat(a,b) computes
// out[i,j] = sum_k a[k,i]*b[k,j], so mul_mat(w, im2col) is exactly the transpose
// of mul_mat(im2col, w) - identical values, channel on ne[0]. The
// feat_extract_norm == "layer" path uses this so its per-frame channel LayerNorm
// reduces ne[0] (ggml_norm's only reduction axis) directly, dropping the
// transpose-in that the time-major layout would otherwise require. Layout /
// reduction-axis only, so parity is unaffected.
static ggml_tensor* conv1d_strided(ggml_context* c, ggml_tensor* w,
                                   ggml_tensor* x, int stride,
                                   bool channels_first = false) {
    ggml_tensor* im2col = ggml_im2col(c, w, x, /*s0=*/stride, /*s1=*/0, /*p0=*/0,
                                      /*p1=*/0, /*d0=*/1, /*d1=*/0,
                                      /*is_2D=*/false, GGML_TYPE_F32);   // [IC*K, OL]
    ggml_tensor* col = ggml_reshape_2d(c, im2col, im2col->ne[0], im2col->ne[1]); // [IC*K, OL]
    ggml_tensor* ker = ggml_reshape_2d(c, w, w->ne[0] * w->ne[1], w->ne[2]);     // [IC*K, OC]
    if (channels_first)
        return ggml_mul_mat(c, ker, col);   // -> [OC, OL]
    return ggml_mul_mat(c, col, ker);       // -> [OL, OC]
}

// The 7-layer strided Conv1d feature encoder (see feature_encoder()/Task 18 for
// the full provenance). Consumes the raw-waveform input tensor `x` (ne = [L, 1])
// and returns `feat_extract_out` with ne = [T', conv_dim] (time outer, channel
// inner). Factored out so both feature_encoder() (Task 18 gate) and encode()
// (Task 19) share ONE conv graph - the math is byte-identical.
static ggml_tensor* build_conv_encoder(ggml_context* c, const ModelLoader& ml,
                                       const VoiceDetectConfig::W2V2Config& w,
                                       ggml_tensor* x) {
    const bool group_norm = (w.feat_extract_norm == "group");
    const bool layer_norm_conv = (w.feat_extract_norm == "layer");
    // HF builds conv0's GroupNorm as nn.GroupNorm(num_groups, num_channels) with
    // the default eps (it passes none); the checkpoint reports 1e-5. NOT
    // layer_norm_eps (that is the transformer LayerNorm's).
    const float gn_eps = 1e-5f;
    const float ln_eps = w.layer_norm_eps;

    // Wav2Vec2FeatureExtractor do_normalize: zero-mean/unit-var the raw waveform
    // before the conv stack. base/emotion has do_normalize == False (raw passthrough);
    // large-robust/age-gender has it True. x is [L,1] (ne0 == L, one channel), so
    // ggml_norm reduces ne0 = the whole sequence with biased var + eps 1e-7, exactly
    // (x - x.mean()) / sqrt(x.var() + 1e-7).
    if (w.do_normalize) {
        x = ggml_norm(c, x, 1e-7f);
    }
    for (uint32_t i = 0; i < w.num_conv_layers; ++i) {
        // Names are abbreviated by the converter (wav2vec2. -> w2v2.) to fit
        // ggml's 64-char limit; the rest is verbatim (see shorten_w2v2_name).
        const std::string p =
            "w2v2.feature_extractor.conv_layers." + std::to_string(i);
        ggml_tensor* cw = clone_weight(c, ml, (p + ".conv.weight").c_str()); // [K,IC,OC]
        // The "layer" path emits the conv channel-major ([OC,OL]) so its per-frame
        // channel LayerNorm runs over ne[0] in place; the "group" path keeps the
        // time-major [OL,OC] layout its per-channel-over-time GroupNorm expects.
        ggml_tensor* h = conv1d_strided(c, cw, x, w.conv_strides[i],
                                        /*channels_first=*/layer_norm_conv);

        // Conv1d bias (config.conv_bias): False for wav2vec2-base (emotion), True
        // for wav2vec2-large-robust (age/gender). Bias is [OC]; it broadcasts over
        // whichever axis is NOT the channel axis: ne[1]=OC for the time-major
        // [OL,OC] layout (reshape to [1,OC]), or ne[0]=OC for the channel-major
        // [OC,OL] layout (a bare [OC] tensor broadcasts over ne[1]=OL).
        if (w.conv_bias) {
            ggml_tensor* cb = clone_weight_opt(c, ml, (p + ".conv.bias").c_str());
            if (cb) h = layer_norm_conv
                ? ggml_add(c, h, cb)
                : ggml_add(c, h, ggml_reshape_2d(c, cb, 1, cb->ne[0]));
        }

        // feat_extract_norm == "group": ONLY conv layer 0 carries a GroupNorm
        // (Wav2Vec2GroupNormConvLayer) with num_groups == num_channels, i.e. each
        // channel is its own group -> per-channel mean/var over time. ggml_norm
        // normalizes over ne[0] (time) per channel ne[1], exactly that (biased var).
        if (group_norm && i == 0) {
            ggml_tensor* g = clone_weight(c, ml, (p + ".layer_norm.weight").c_str());
            ggml_tensor* b = clone_weight(c, ml, (p + ".layer_norm.bias").c_str());
            h = ggml_norm(c, h, gn_eps);
            h = ggml_mul(c, h, ggml_reshape_2d(c, g, 1, g->ne[0]));   // affine scale [1,C]
            h = ggml_add(c, h, ggml_reshape_2d(c, b, 1, b->ne[0]));   // affine shift [1,C]
        } else if (layer_norm_conv) {
            // feat_extract_norm == "layer" (Wav2Vec2LayerNormConvLayer, every conv
            // layer): a LayerNorm over the CHANNEL dim per time frame. PyTorch does
            // conv [C,T'] -> transpose [T',C] -> LayerNorm(C) -> transpose back. h is
            // already channel-major [OC=channel, OL=time], so ggml_norm reduces
            // ne[0]=OC (per frame) directly - no transpose-in. The single transpose
            // back to [OL,OC] is the only layout flip the next conv (time-major
            // im2col) requires, halving the two ggml_cont(transpose)s the old
            // transpose-in/transpose-out pair did per layer.
            ggml_tensor* g = clone_weight(c, ml, (p + ".layer_norm.weight").c_str());
            ggml_tensor* b = clone_weight(c, ml, (p + ".layer_norm.bias").c_str());
            // HF's Wav2Vec2LayerNormConvLayer uses a plain nn.LayerNorm with its
            // default eps (1e-5), a DISTINCT source from the transformer's
            // layer_norm_eps. They are assumed equal (both 1e-5 on this checkpoint),
            // so reusing ln_eps here preserves parity.
            h = ggml_norm(c, h, ln_eps);                             // reduce ne[0]=OC
            h = ggml_mul(c, h, g);                                   // g [OC] over ne[0]
            h = ggml_add(c, h, b);
            h = ggml_cont(c, ggml_transpose(c, h));                  // [OC,OL] -> [OL,OC]
        }

        // feat_extract_activation == "gelu" is HF's exact erf GELU (nn.GELU()),
        // NOT the tanh approximation (ggml_gelu).
        x = ggml_gelu_erf(c, h);                                      // [OL,OC]
    }
    return x;                                                        // [T', conv_dim]
}

static void reconstruct_pos_conv_weight(const ModelLoader& ml,
                                        const VoiceDetectConfig::W2V2Config& w,
                                        std::vector<float>& pos_w);

W2V2Analyzer::W2V2Analyzer(const ModelLoader& ml) : ml_(ml) {}

// Lazily reconstruct + cache the weight-normed positional-conv kernel. It is a
// pure function of the static model weights, so compute it once and reuse it on
// every forward (was ~20 ms emotion / ~112 ms age-gender of per-call host work).
// The cache is process-global keyed by the ModelLoader because the analyzer is
// constructed fresh on every analyze call (Model::analyze_path_json), so a
// per-instance cache would be wiped each time. Guarded for concurrent analyze
// calls on a shared model.
const std::vector<float>& W2V2Analyzer::pos_conv_weight() const {
    static std::mutex mtx;
    static std::unordered_map<const ModelLoader*, std::vector<float>> cache;
    std::lock_guard<std::mutex> lk(mtx);
    std::vector<float>& slot = cache[&ml_];
    if (slot.empty())
        reconstruct_pos_conv_weight(ml_, ml_.config().w2v2, slot);
    return slot;
}

std::vector<float> W2V2Analyzer::feature_encoder(const std::vector<float>& pcm,
                                                 int& out_T) const {
    const VoiceDetectConfig::W2V2Config& w = ml_.config().w2v2;

    std::vector<float> out;
    global_backend().compute([&](ggml_context* c) -> ggml_tensor* {
        // Raw waveform fed as a single-input-channel sequence [L, IC=1]. Whether
        // the HF Wav2Vec2FeatureExtractor zero-mean/unit-var normalizes the input
        // is do_normalize, read from KV and applied inside build_conv_encoder:
        // it is false for the group-norm/emotion path but true for the age/gender
        // (large-robust) checkpoint, so it is NOT unconditionally False here.
        const int64_t ne[2] = { (int64_t)pcm.size(), 1 };
        ggml_tensor* x = graph_input_tensor(c, GGML_TYPE_F32, 2, ne,
                                            pcm.data(), pcm.size() * sizeof(float));
        return build_conv_encoder(c, ml_, w, x);                     // [T', conv_dim]
    }, out);

    const int hidden = w.conv_dims.empty() ? (int)w.hidden_size : (int)w.conv_dims.back();
    out_T = hidden > 0 ? (int)(out.size() / (size_t)hidden) : 0;
    return out;
}

// ---------------------------------------------------------------------------
// Transformer-stack helpers. All hidden states are carried as ne = [H, T]
// (feature axis fastest), so ggml_mul_mat(W[in,out], x[in,T]) -> [out,T] and
// ggml_norm normalizes over the feature axis - the transformer convention.
// ---------------------------------------------------------------------------

// LayerNorm over the feature axis ne[0] (eps), then the per-feature affine
// weight/bias (1-D [H], broadcast across time). HF nn.LayerNorm uses the biased
// (population) variance, which is what ggml_norm computes.
static ggml_tensor* layer_norm(ggml_context* c, ggml_tensor* x,
                               ggml_tensor* g, ggml_tensor* b, float eps) {
    x = ggml_norm(c, x, eps);
    x = ggml_mul(c, x, g);   // g ne=[H] broadcasts over ne[1]=T
    x = ggml_add(c, x, b);
    return x;
}

// y = W x + b. W ne=[in,out], x ne=[in,T] -> [out,T]; b (1-D [out]) broadcasts.
static ggml_tensor* linear(ggml_context* c, ggml_tensor* W, ggml_tensor* b,
                           ggml_tensor* x) {
    ggml_tensor* y = ggml_mul_mat(c, W, x);   // [out, T]
    if (b) y = ggml_add(c, y, b);
    return y;
}

// Standard multi-head scaled-dot-product self-attention (Wav2Vec2Attention,
// eager). q is scaled by head_dim**-0.5 BEFORE the q.k^T product (HF folds the
// scale into the query, like whisper). No relative position (the position signal
// comes from the conv embedding) and no attention mask (single full-length clip).
// A/B for the QK^T/softmax/xV core, default "flash". The manual path
// materializes a [T,T] scores matrix and does several permute+cont copies per
// layer; on CPU those memory ops are not free (depth-anything measured the
// attention core at ~35% of its ViT backbone, dominated by the copies not the
// FLOPs). ggml_flash_attn_ext fuses scaled-QK^T + softmax + xV into one streaming
// op with no materialized scores. CPU fattn accepts F32 k/v, so parity stays
// tight with the manual F32 softmax. VD_ATTN=manual restores the old path.
static bool flash_attn_enabled() {
    static const bool flash = [] {
        const char* e = std::getenv("VD_ATTN");
        return !(e && std::strcmp(e, "manual") == 0);
    }();
    return flash;
}

static ggml_tensor* self_attention(ggml_context* c, const ModelLoader& ml,
                                   const std::string& p, ggml_tensor* h,
                                   int n_heads) {
    const int H  = (int)h->ne[0];
    const int T  = (int)h->ne[1];
    const int hd = H / n_heads;
    const float scaling = 1.0f / std::sqrt((float)hd);

    ggml_tensor* q = linear(c, clone_weight(c, ml, (p + ".q_proj.weight").c_str()),
                               clone_weight(c, ml, (p + ".q_proj.bias").c_str()), h);
    ggml_tensor* k = linear(c, clone_weight(c, ml, (p + ".k_proj.weight").c_str()),
                               clone_weight(c, ml, (p + ".k_proj.bias").c_str()), h);
    ggml_tensor* v = linear(c, clone_weight(c, ml, (p + ".v_proj.weight").c_str()),
                               clone_weight(c, ml, (p + ".v_proj.bias").c_str()), h);

    if (flash_attn_enabled()) {
        // q/k/v: [H,T] -> [hd, n_heads, T] -> permute to the flash layout
        // [hd, T, n_heads] (ne0=head_dim, ne1=tokens, ne2=heads). flash applies
        // the 1/sqrt(hd) scale internally, so q is NOT pre-scaled here.
        ggml_tensor* qf = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, q, hd, n_heads, T), 0, 2, 1, 3));
        ggml_tensor* kf = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, k, hd, n_heads, T), 0, 2, 1, 3));
        ggml_tensor* vf = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, v, hd, n_heads, T), 0, 2, 1, 3));
        ggml_tensor* o = ggml_flash_attn_ext(c, qf, kf, vf, nullptr, scaling, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(o, GGML_PREC_F32);  // match the manual F32 softmax
        ggml_tensor* out = ggml_reshape_2d(c, o, H, T);  // out is [hd, n_heads, T] -> [H, T]
        return linear(c, clone_weight(c, ml, (p + ".out_proj.weight").c_str()),
                         clone_weight(c, ml, (p + ".out_proj.bias").c_str()), out);
    }

    q = ggml_scale(c, q, scaling);

    // [H,T] -> [hd, n_heads, T]
    q = ggml_reshape_3d(c, q, hd, n_heads, T);
    k = ggml_reshape_3d(c, k, hd, n_heads, T);
    v = ggml_reshape_3d(c, v, hd, n_heads, T);

    // q,k -> [hd, T, n_heads]; v -> [T, hd, n_heads]
    q = ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3));
    k = ggml_cont(c, ggml_permute(c, k, 0, 2, 1, 3));
    v = ggml_cont(c, ggml_permute(c, v, 1, 2, 0, 3));

    ggml_tensor* kq = ggml_mul_mat(c, k, q);       // [T_k, T_q, n_heads]
    kq = ggml_soft_max(c, kq);                     // softmax over ne[0]=T_k
    ggml_tensor* kqv = ggml_mul_mat(c, v, kq);     // [hd, T_q, n_heads]

    // merge heads: [hd, T_q, n_heads] -> [hd, n_heads, T_q] -> [H, T_q]
    kqv = ggml_cont(c, ggml_permute(c, kqv, 0, 2, 1, 3));
    ggml_tensor* out = ggml_reshape_2d(c, kqv, H, T);
    return linear(c, clone_weight(c, ml, (p + ".out_proj.weight").c_str()),
                     clone_weight(c, ml, (p + ".out_proj.bias").c_str()), out);
}

// Positional conv embedding (Wav2Vec2PositionalConvEmbedding): a grouped Conv1d
// (kernel=128, groups=16, padding=64) whose weight is WEIGHT-NORMED, then a
// "same"-pad layer that drops the last time step (kernel is even), then GELU. The
// weight-norm reconstruction w = g * v / ||v|| (norm over every dim except the
// kernel axis, dim=2) is precomputed on the host and fed as `pos_w` with ne =
// [K, IC/groups, H]. Consumes proj ne=[H,T], returns the embedding ne=[H,T].
static ggml_tensor* positional_conv(ggml_context* c, const ModelLoader& ml,
                                    const VoiceDetectConfig::W2V2Config& w,
                                    ggml_tensor* proj,
                                    const std::vector<float>& pos_w) {
    const int H      = (int)proj->ne[0];
    const int T      = (int)proj->ne[1];
    const int K      = (int)w.num_conv_pos_embeddings;        // 128
    const int groups = (int)w.num_conv_pos_embedding_groups;  // 16
    const int pad    = K / 2;                                 // 64
    const int icg    = H / groups;                            // 48 (== ocg)

    const int64_t wne[3] = { K, icg, H };
    ggml_tensor* wfull = graph_input_tensor(c, GGML_TYPE_F32, 3, wne,
        pos_w.data(), pos_w.size() * sizeof(float));           // [K, icg, H]

    ggml_tensor* x = ggml_cont(c, ggml_transpose(c, proj));    // [T, H]

    // Grouped Conv1d: each output-channel group convolves only its own input
    // channels. Slice both the input channels and the weight output channels per
    // group, run a padded im2col conv, and concat along the channel axis.
    ggml_tensor* out = nullptr;
    for (int g = 0; g < groups; ++g) {
        ggml_tensor* xg = ggml_cont(c, ggml_view_2d(c, x, T, icg, x->nb[1],
                              (size_t)g * icg * x->nb[1]));     // [T, icg]
        ggml_tensor* wg = ggml_cont(c, ggml_view_3d(c, wfull, K, icg, icg,
                              wfull->nb[1], wfull->nb[2],
                              (size_t)g * icg * wfull->nb[2])); // [K, icg, ocg]
        ggml_tensor* im = ggml_im2col(c, wg, xg, /*s0=*/1, /*s1=*/0, /*p0=*/pad,
                                      /*p1=*/0, /*d0=*/1, /*d1=*/0,
                                      /*is_2D=*/false, GGML_TYPE_F32); // [icg*K, OL]
        ggml_tensor* yg = ggml_mul_mat(c,
            ggml_reshape_2d(c, im, im->ne[0], im->ne[1]),
            ggml_reshape_2d(c, wg, wg->ne[0] * wg->ne[1], wg->ne[2]));  // [OL, ocg]
        out = (g == 0) ? yg : ggml_concat(c, out, yg, /*dim=*/1);       // [OL, H]
    }

    ggml_tensor* bias = clone_weight(c, ml, "w2v2.encoder.pos_conv_embed.conv.bias");
    out = ggml_add(c, out, ggml_reshape_2d(c, bias, 1, H));    // [OL, H]

    // Wav2Vec2SamePadLayer: drop the last time step when the kernel is even.
    const int remove = (K % 2 == 0) ? 1 : 0;
    const int OL = (int)out->ne[0];
    out = ggml_cont(c, ggml_view_2d(c, out, OL - remove, H, out->nb[1], 0)); // [T, H]
    out = ggml_gelu_erf(c, out);                              // exact erf GELU
    return ggml_cont(c, ggml_transpose(c, out));              // [H, T]
}

// Weight-norm reconstruction of the positional conv kernel is a static weight
// transform, so do it once on the host (avoids an awkward in-graph reduction over
// every axis but the kernel). w[k,ic,oc] = g[k] * v[k,ic,oc] / ||v[k]||, the norm
// taken over ic,oc (every dim except the kernel axis dim=2). In ggml ne-order the
// kernel axis is ne[0], so the per-k norm reduces ne[1],ne[2]. Shared by encode()
// and emotion_logits() so the pos-conv embedding is byte-identical.
static void reconstruct_pos_conv_weight(const ModelLoader& ml,
                                        const VoiceDetectConfig::W2V2Config& w,
                                        std::vector<float>& pos_w) {
    std::vector<float> g_host, v_host;
    weight_to_host_f32(ml, "w2v2.encoder.pos_conv_embed.conv.weight_g", g_host);
    weight_to_host_f32(ml, "w2v2.encoder.pos_conv_embed.conv.weight_v", v_host);
    const int K   = (int)w.num_conv_pos_embeddings;            // 128
    const int OC  = (int)w.hidden_size;                        // 768
    const int icg = OC / (int)w.num_conv_pos_embedding_groups; // 48
    std::vector<double> sumsq(K, 0.0);
    for (int oc = 0; oc < OC; ++oc)
        for (int ic = 0; ic < icg; ++ic)
            for (int k = 0; k < K; ++k) {
                const double vv = v_host[(size_t)k + (size_t)ic * K +
                                         (size_t)oc * K * icg];
                sumsq[k] += vv * vv;
            }
    pos_w.resize(v_host.size());
    for (int k = 0; k < K; ++k) {
        const double norm = std::sqrt(sumsq[k]);
        const double s = (norm > 0.0) ? (double)g_host[k] / norm : 0.0;
        for (int oc = 0; oc < OC; ++oc)
            for (int ic = 0; ic < icg; ++ic) {
                const size_t idx = (size_t)k + (size_t)ic * K +
                                   (size_t)oc * K * icg;
                pos_w[idx] = (float)(v_host[idx] * s);
            }
    }
}

// Build the wav2vec2 encoder graph from the raw waveform and append every hidden
// state (the encoder-layernorm output `hidden_states[0]` + each of the n_layers
// layer outputs, i.e. n_layers + 1 states) to `states`, each ne=[H,T'] (feature
// axis fastest, the transformer layout). Returns the final hidden state
// (== states.back()). Shared by encode() (parity goldens) and emotion_logits()
// (the head's weighted-layer-sum consumes the whole stack). Must run inside a
// Backend::compute build lambda.
static ggml_tensor* build_transformer_states(
        ggml_context* c, const ModelLoader& ml,
        const VoiceDetectConfig::W2V2Config& w, const std::vector<float>& pcm,
        const std::vector<float>& pos_w, std::vector<ggml_tensor*>& states) {
    const float eps    = w.layer_norm_eps;
    const int n_layers = (int)w.n_layers;
    const int n_heads  = (int)w.n_heads;

    const int64_t ne[2] = { (int64_t)pcm.size(), 1 };
    ggml_tensor* x = graph_input_tensor(c, GGML_TYPE_F32, 2, ne,
                                        pcm.data(), pcm.size() * sizeof(float));
    ggml_tensor* feat = build_conv_encoder(c, ml, w, x);     // [T', conv_dim]

    // Transpose to the transformer layout [conv_dim, T'] (feature axis fastest).
    ggml_tensor* h = ggml_cont(c, ggml_transpose(c, feat));  // [conv_dim, T']

    // Feature projection: LayerNorm over the 512 conv dim -> Linear 512->hidden.
    h = layer_norm(c, h,
            clone_weight(c, ml, "w2v2.feature_projection.layer_norm.weight"),
            clone_weight(c, ml, "w2v2.feature_projection.layer_norm.bias"), eps);
    ggml_tensor* proj = linear(c,
            clone_weight(c, ml, "w2v2.feature_projection.projection.weight"),
            clone_weight(c, ml, "w2v2.feature_projection.projection.bias"), h);  // [H,T']

    // Positional conv embedding, added to the projected features.
    ggml_tensor* pos = positional_conv(c, ml, w, proj, pos_w);  // [H,T']
    ggml_tensor* hs = ggml_add(c, proj, pos);

    if (w.do_stable_layer_norm) {
        // Wav2Vec2EncoderStableLayerNorm (wav2vec2-large-robust, age/gender): NO
        // encoder LayerNorm before the layers; PRE-norm layers; ONE final encoder
        // LayerNorm AFTER all layers. HF's output_hidden_states records the input to
        // each layer, so states[0] = proj+pos (pre layer 0), states[i] = input to
        // layer i, and states[n_layers] = the final-LayerNorm'd last hidden state.
        states.push_back(hs);   // hidden_states[0]
        for (int i = 0; i < n_layers; ++i) {
            const std::string p = "w2v2.encoder.layers." + std::to_string(i);
            // attn sublayer: res + attn(layer_norm(hs)).
            ggml_tensor* res = hs;
            ggml_tensor* hn = layer_norm(c, hs,
                    clone_weight(c, ml, (p + ".layer_norm.weight").c_str()),
                    clone_weight(c, ml, (p + ".layer_norm.bias").c_str()), eps);
            ggml_tensor* a = self_attention(c, ml, p + ".attention", hn, n_heads);
            hs = ggml_add(c, res, a);
            // ff sublayer: hs + output_dense(gelu(intermediate_dense(final_layer_norm(hs)))).
            ggml_tensor* fn = layer_norm(c, hs,
                    clone_weight(c, ml, (p + ".final_layer_norm.weight").c_str()),
                    clone_weight(c, ml, (p + ".final_layer_norm.bias").c_str()), eps);
            ggml_tensor* ff = linear(c,
                    clone_weight(c, ml, (p + ".feed_forward.intermediate_dense.weight").c_str()),
                    clone_weight(c, ml, (p + ".feed_forward.intermediate_dense.bias").c_str()), fn);
            ff = ggml_gelu_erf(c, ff);
            ff = linear(c,
                    clone_weight(c, ml, (p + ".feed_forward.output_dense.weight").c_str()),
                    clone_weight(c, ml, (p + ".feed_forward.output_dense.bias").c_str()), ff);
            hs = ggml_add(c, hs, ff);
            if (i < n_layers - 1) states.push_back(hs);   // hidden_states[i+1]
        }
        // Final encoder LayerNorm -> last_hidden_state == hidden_states[n_layers].
        hs = layer_norm(c, hs,
                clone_weight(c, ml, "w2v2.encoder.layer_norm.weight"),
                clone_weight(c, ml, "w2v2.encoder.layer_norm.bias"), eps);
        states.push_back(hs);
        return hs;
    }

    // Post-norm path (do_stable_layer_norm == False, wav2vec2-base / emotion): the
    // encoder LayerNorm is applied to (proj + pos) BEFORE the layers, and each layer
    // is attn -> +residual -> layer_norm -> ff -> +residual -> final_layer_norm.
    hs = layer_norm(c, hs,
            clone_weight(c, ml, "w2v2.encoder.layer_norm.weight"),
            clone_weight(c, ml, "w2v2.encoder.layer_norm.bias"), eps);

    states.push_back(hs);   // hidden_states[0] (HF: pre-layer, post encoder LN)

    for (int i = 0; i < n_layers; ++i) {
        const std::string p = "w2v2.encoder.layers." + std::to_string(i);
        ggml_tensor* res = hs;
        ggml_tensor* a = self_attention(c, ml, p + ".attention", hs, n_heads);
        hs = ggml_add(c, res, a);
        hs = layer_norm(c, hs,
                clone_weight(c, ml, (p + ".layer_norm.weight").c_str()),
                clone_weight(c, ml, (p + ".layer_norm.bias").c_str()), eps);

        ggml_tensor* ff = linear(c,
                clone_weight(c, ml, (p + ".feed_forward.intermediate_dense.weight").c_str()),
                clone_weight(c, ml, (p + ".feed_forward.intermediate_dense.bias").c_str()), hs);
        ff = ggml_gelu_erf(c, ff);   // hidden_act == "gelu" -> exact erf GELU
        ff = linear(c,
                clone_weight(c, ml, (p + ".feed_forward.output_dense.weight").c_str()),
                clone_weight(c, ml, (p + ".feed_forward.output_dense.bias").c_str()), ff);
        hs = ggml_add(c, hs, ff);
        hs = layer_norm(c, hs,
                clone_weight(c, ml, (p + ".final_layer_norm.weight").c_str()),
                clone_weight(c, ml, (p + ".final_layer_norm.bias").c_str()), eps);

        states.push_back(hs);   // hidden_states[i+1] (AFTER encoder layer i)
    }
    return hs;
}

std::vector<float> W2V2Analyzer::encode(const std::vector<float>& pcm, int& out_T,
                                        std::vector<float>* dbg_layer0,
                                        std::vector<float>* dbg_layer_mid) const {
    const VoiceDetectConfig::W2V2Config& w = ml_.config().w2v2;
    const int n_layers = (int)w.n_layers;
    // enc_layer_mid is hidden_states[n_layers/2] (AFTER encoder layer n_layers/2-1).
    const int mid_idx  = n_layers / 2 - 1;

    const std::vector<float>& pos_w = pos_conv_weight();

    std::vector<float> out;
    global_backend().compute([&](ggml_context* c) -> ggml_tensor* {
        std::vector<ggml_tensor*> states;
        ggml_tensor* last = build_transformer_states(c, ml_, w, pcm, pos_w, states);

        // The hidden states are carried as ne=[H,T'] (feature axis fastest) for the
        // transformer math, but the goldens store [H,T'] hidden-major (numpy index
        // h*T'+t). A ggml [H,T'] tensor is flat h+t*H (time-major), so transpose to
        // ne=[T',H] (flat t+h*T') before readback to match the golden element order.
        auto to_golden = [&](ggml_tensor* t) { return ggml_cont(c, ggml_transpose(c, t)); };

        // states[i+1] is hidden_states[i+1] (AFTER layer i): enc_layer_0 == states[1],
        // enc_layer_mid == states[mid_idx+1] == hidden_states[n_layers/2].
        if (dbg_layer0)    capture_graph_output(to_golden(states[1]),           dbg_layer0);
        if (dbg_layer_mid) capture_graph_output(to_golden(states[mid_idx + 1]), dbg_layer_mid);
        return to_golden(last);   // ne=[T',H], golden enc_layer_last element order
    }, out);

    out_T = (w.hidden_size > 0) ? (int)(out.size() / (size_t)w.hidden_size) : 0;
    return out;
}

std::vector<float> W2V2Analyzer::emotion_logits(const std::vector<float>& pcm) const {
    const VoiceDetectConfig::W2V2Config& w = ml_.config().w2v2;

    const std::vector<float>& pos_w = pos_conv_weight();

    // Weighted-layer-sum weights: softmax(layer_weights) over the n_layers+1 stack.
    // The softmax is a tiny static reduction, so do it on the host and feed the
    // normalized scalars into the graph as per-state ggml_scale factors (avoids an
    // in-graph stack + broadcast). Max-shifted for numerical stability.
    std::vector<float> lw, lsw;
    weight_to_host_f32(ml_, "layer_weights", lw);
    if (w.use_weighted_layer_sum && !lw.empty()) {
        double mx = lw[0];
        for (float v : lw) if (v > mx) mx = v;
        double s = 0.0;
        lsw.resize(lw.size());
        for (size_t i = 0; i < lw.size(); ++i) { lsw[i] = (float)std::exp((double)lw[i] - mx); s += lsw[i]; }
        for (float& v : lsw) v = (float)(v / s);
    }

    std::vector<float> logits;
    global_backend().compute([&](ggml_context* c) -> ggml_tensor* {
        std::vector<ggml_tensor*> states;
        ggml_tensor* last = build_transformer_states(c, ml_, w, pcm, pos_w, states);

        // Weighted-layer-sum: sum_i softmax(layer_weights)[i] * hidden_states[i].
        // Without it (plain classifier) HF uses the final hidden state outputs[0].
        ggml_tensor* hs;   // [H,T']
        if (w.use_weighted_layer_sum && lsw.size() == states.size()) {
            ggml_tensor* acc = nullptr;
            for (size_t i = 0; i < states.size(); ++i) {
                ggml_tensor* term = ggml_scale(c, states[i], lsw[i]);
                acc = (i == 0) ? term : ggml_add(c, acc, term);
            }
            hs = acc;
        } else {
            hs = last;
        }

        // projector: Linear hidden -> classifier_proj_size, applied per time frame.
        ggml_tensor* proj = linear(c,
                clone_weight(c, ml_, "projector.weight"),
                clone_weight(c, ml_, "projector.bias"), hs);   // [P,T']

        // Mean-pool over time (HF pooled_output = hidden_states.mean(dim=time)):
        // transpose [P,T'] -> [T',P], ggml_mean reduces ne[0]=T' -> [1,P] -> [P].
        ggml_tensor* pooled = ggml_mean(c, ggml_cont(c, ggml_transpose(c, proj)));  // [1,P]
        pooled = ggml_reshape_1d(c, pooled, proj->ne[0]);                            // [P]

        // classifier: Linear classifier_proj_size -> num_emotions.
        return linear(c,
                clone_weight(c, ml_, "classifier.weight"),
                clone_weight(c, ml_, "classifier.bias"), pooled);   // [num_emotions]
    }, logits);

    return logits;
}

std::string W2V2Analyzer::analyze_json(const std::vector<float>& pcm) const {
    const std::vector<float> logits = emotion_logits(pcm);
    const std::vector<std::string>& labels = ml_.config().emotion_labels;

    // softmax(logits) -> probs, argmax -> dominant label.
    std::vector<float> probs(logits.size());
    int argmax = 0;
    if (!logits.empty()) {
        double mx = logits[0];
        for (float v : logits) if (v > mx) mx = v;
        double s = 0.0;
        for (size_t i = 0; i < logits.size(); ++i) { probs[i] = (float)std::exp((double)logits[i] - mx); s += probs[i]; }
        for (float& v : probs) v = (float)(v / s);
        for (size_t i = 1; i < logits.size(); ++i) if (logits[i] > logits[argmax]) argmax = (int)i;
    }
    const std::string dom =
        (!labels.empty() && argmax < (int)labels.size()) ? labels[argmax] : "";

    // Hand-rolled JSON (no new deps) matching the C-ABI shape in voicedetect_capi.h.
    // age/gender are emitted only when the GGUF carries those heads (this emotion
    // model does not); emotion is the default-on deliverable.
    auto fnum = [](double x) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.6g", x);
        return std::string(buf);
    };
    std::string js = "{\"emotion\":{\"label\":\"" + dom + "\",\"scores\":{";
    for (size_t i = 0; i < labels.size(); ++i) {
        if (i) js += ",";
        js += "\"" + labels[i] + "\":" + fnum(i < probs.size() ? probs[i] : 0.0f);
    }
    js += "}}}";
    return js;
}

std::vector<float> W2V2Analyzer::age_gender_raw(const std::vector<float>& pcm) const {
    const VoiceDetectConfig::W2V2Config& w = ml_.config().w2v2;

    const std::vector<float>& pos_w = pos_conv_weight();

    std::vector<float> out;
    global_backend().compute([&](ggml_context* c) -> ggml_tensor* {
        std::vector<ggml_tensor*> states;
        ggml_tensor* last = build_transformer_states(c, ml_, w, pcm, pos_w, states);  // [H,T']

        // AgeGenderModel: pooled = mean over time of the last hidden state, then two
        // ModelHeads. Mean-pool [H,T'] -> transpose [T',H] -> ggml_mean reduces
        // ne[0]=T' -> [1,H] -> [H].
        ggml_tensor* pooled = ggml_mean(c, ggml_cont(c, ggml_transpose(c, last)));  // [1,H]
        pooled = ggml_reshape_1d(c, pooled, last->ne[0]);                            // [H]

        // ModelHead: out_proj(tanh(dense(x))). dense/out_proj are plain Linears; the
        // dropout layers are eval-time identities.
        auto head = [&](const char* name) {
            ggml_tensor* h = linear(c,
                    clone_weight(c, ml_, (std::string(name) + ".dense.weight").c_str()),
                    clone_weight(c, ml_, (std::string(name) + ".dense.bias").c_str()), pooled);
            h = ggml_tanh(c, h);
            return linear(c,
                    clone_weight(c, ml_, (std::string(name) + ".out_proj.weight").c_str()),
                    clone_weight(c, ml_, (std::string(name) + ".out_proj.bias").c_str()), h);
        };
        ggml_tensor* age    = head("age");      // [1]
        ggml_tensor* gender = head("gender");   // [num_genders]
        // Concatenate so one backbone pass yields both heads: [age_raw, gender...].
        return ggml_concat(c, age, gender, /*dim=*/0);
    }, out);

    return out;
}

std::string W2V2Analyzer::analyze_age_gender_json(const std::vector<float>& pcm) const {
    const std::vector<float> raw = age_gender_raw(pcm);
    const std::vector<std::string>& labels = ml_.config().gender_labels;

    // raw = [age_raw, gender_logit_0, ...]. age years = age_raw * 100.
    const double age_years = raw.empty() ? 0.0 : (double)raw[0] * 100.0;
    std::vector<float> glogits(raw.begin() + (raw.empty() ? 0 : 1), raw.end());

    // softmax(gender_logits) -> probs, argmax -> dominant label.
    std::vector<float> probs(glogits.size());
    int argmax = 0;
    if (!glogits.empty()) {
        double mx = glogits[0];
        for (float v : glogits) if (v > mx) mx = v;
        double s = 0.0;
        for (size_t i = 0; i < glogits.size(); ++i) { probs[i] = (float)std::exp((double)glogits[i] - mx); s += probs[i]; }
        for (float& v : probs) v = (float)(v / s);
        for (size_t i = 1; i < glogits.size(); ++i) if (glogits[i] > glogits[argmax]) argmax = (int)i;
    }
    const std::string dom =
        (!labels.empty() && argmax < (int)labels.size()) ? labels[argmax] : "";

    auto fnum = [](double x) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.6g", x);
        return std::string(buf);
    };
    // C-ABI shape: {"age":<years>,"gender":{"label":..,"<label>":<prob>,...}}.
    std::string js = "{\"age\":" + fnum(age_years) +
                     ",\"gender\":{\"label\":\"" + dom + "\"";
    for (size_t i = 0; i < labels.size(); ++i)
        js += ",\"" + labels[i] + "\":" + fnum(i < probs.size() ? probs[i] : 0.0f);
    js += "}}";
    return js;
}

} // namespace vd
