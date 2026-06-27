#include "encoder.hpp"
#include "backend.hpp"
#include "model_loader.hpp"
#include "ggml.h"
#include <cmath>
#include <string>
#include <vector>

namespace vd {

CamPPlusEncoder::CamPPlusEncoder(const ModelLoader& ml) : ml_(ml) {}

using CCfg = VoiceDetectConfig::CamPPlusConfig;

// Pull the next conv (forward-execution order) from the flat manifest and apply
// it as a 1D Conv1d. The dense backbone is kept CHANNEL-MAJOR [C, T] (ne0=C,
// ne1=T) so every 1x1 conv (the bulk of CAM++: linear1, both CAM linears, the
// transit conv) collapses to a SINGLE ggml_mul_mat with no transpose churn:
// weight [IC, OC] times x [IC, T] lands directly as [OC, T]. The k>1 convs (the
// strided TDNN and each layer's dilated linear_local) still need im2col, which
// wants a TIME-MAJOR [W=T, IC] input; those two call sites pass a time-major
// tensor (the FCM output for the TDNN, the shared cont(transpose(cc)) for
// linear_local) and this helper emits channel-major [OC, OL] directly by
// ordering the mul_mat weight-first. The ONNX export folds the BatchNorm into
// some convs (so they carry a bias) and leaves others bias-free; the manifest
// records "" for the latter. ONNX weight [OC, IC, K] -> ggml ne = [K, IC, OC].
static ggml_tensor* next_conv1d(ggml_context* c, const ModelLoader& ml, const CCfg& cp,
                                int& idx, ggml_tensor* x, int stride, int pad, int dil) {
    ggml_tensor* w = clone_weight(c, ml, cp.conv_weight[idx].c_str());   // [K, IC, OC]
    ggml_tensor* b = cp.conv_bias[idx].empty()
                         ? nullptr : clone_weight(c, ml, cp.conv_bias[idx].c_str());
    ++idx;
    const int k = (int)w->ne[0];
    if (k == 1 && stride == 1 && pad == 0 && dil == 1) {
        // x CHANNEL-MAJOR [IC, T]; clean per-frame matmul -> [OC, T].
        ggml_tensor* w2 = ggml_reshape_2d(c, w, w->ne[1], w->ne[2]);     // [IC, OC]
        ggml_tensor* y  = ggml_mul_mat(c, w2, x);                        // [OC, T]
        if (b) y = ggml_add(c, y, ggml_reshape_2d(c, b, b->ne[0], 1));   // [OC,1] bcast over T
        return y;
    }
    // x TIME-MAJOR [W=T, IC]; im2col then emit channel-major [OC, OL].
    ggml_tensor* im = ggml_im2col(c, w, x, /*s0=*/stride, /*s1=*/0, /*p0=*/pad,
                                  /*p1=*/0, /*d0=*/dil, /*d1=*/0,
                                  /*is_2D=*/false, GGML_TYPE_F32);        // [IC*K, OL]
    ggml_tensor* y = ggml_mul_mat(c,
        ggml_reshape_2d(c, w, w->ne[0] * w->ne[1], w->ne[2]),            // [IC*K, OC]
        ggml_reshape_2d(c, im, im->ne[0], im->ne[1]));                   // [IC*K, OL]
    if (b) y = ggml_add(c, y, ggml_reshape_2d(c, b, b->ne[0], 1));       // [OC, OL]
    return y;
}

// Pull the next FCM conv (2D Conv2d over a [W=time, H=freq, C, N] tensor). torch
// stride is (freq, time); FCM only ever strides the frequency axis, so s_time is
// always 1 and s_freq carries the downsample. ONNX [OC, IC, KH, KW] -> ggml ne =
// [KW, KH, IC, OC] (the ggml_conv_2d kernel layout). Every FCM conv has its BN
// folded in, so it always carries a bias.
static ggml_tensor* next_conv2d(ggml_context* c, const ModelLoader& ml, const CCfg& cp,
                                int& idx, ggml_tensor* x, int s_freq, int pad) {
    ggml_tensor* w = clone_weight(c, ml, cp.conv_weight[idx].c_str());   // [KW, KH, IC, OC]
    ggml_tensor* b = clone_weight(c, ml, cp.conv_bias[idx].c_str());     // [OC]
    ++idx;
    ggml_tensor* y = conv2d_auto(c, w, x, /*s0=time*/1, /*s1=freq*/s_freq,
                                 pad, pad, 1, 1);                         // [OW, OH, OC, N]
    return ggml_add(c, y, ggml_reshape_4d(c, b, 1, 1, b->ne[0], 1));
}

// BatchNorm1d (eval) folded to a per-channel affine. PyTorch BN in eval applies
// y = gamma*(x-mu)/sqrt(var+eps) + beta with the running stats: a fixed scale +
// shift per channel. We precompute them on the host and apply broadcast constants
// over the time axis. `prefix` empty -> affine-free BN (gamma=1, beta=0); `prefix`
// reads <prefix>.{weight,bias,running_mean,running_var}. x ne = [T, C].
static ggml_tensor* bn1d(ggml_context* c, const ModelLoader& ml,
                         const std::string& prefix, ggml_tensor* x, float eps,
                         const std::string& mean_name, const std::string& var_name,
                         std::vector<std::vector<float>>& scratch) {
    std::vector<float> g, beta, mu, var;
    if (!prefix.empty()) {
        weight_to_host_f32(ml, (prefix + ".weight").c_str(), g);
        weight_to_host_f32(ml, (prefix + ".bias").c_str(), beta);
        weight_to_host_f32(ml, (prefix + ".running_mean").c_str(), mu);
        weight_to_host_f32(ml, (prefix + ".running_var").c_str(), var);
    } else {
        weight_to_host_f32(ml, mean_name.c_str(), mu);
        weight_to_host_f32(ml, var_name.c_str(), var);
        g.assign(mu.size(), 1.0f);     // affine-free: gamma=1, beta=0
        beta.assign(mu.size(), 0.0f);
    }
    const int C = (int)mu.size();
    scratch.emplace_back(C);
    scratch.emplace_back(C);
    std::vector<float>& scale = scratch[scratch.size() - 2];
    std::vector<float>& shift = scratch[scratch.size() - 1];
    for (int i = 0; i < C; ++i) {
        const float inv = g[i] / std::sqrt(var[i] + eps);
        scale[i] = inv;
        shift[i] = beta[i] - inv * mu[i];
    }
    const int64_t ne[2] = { 1, C };
    ggml_tensor* s  = graph_input_tensor(c, GGML_TYPE_F32, 2, ne,
                                         scale.data(), (size_t)C * sizeof(float));
    ggml_tensor* sh = graph_input_tensor(c, GGML_TYPE_F32, 2, ne,
                                         shift.data(), (size_t)C * sizeof(float));
    return ggml_add(c, ggml_mul(c, x, s), sh);  // [T, C]
}

// CHANNEL-MAJOR BatchNorm1d (eval): same per-channel affine as bn1d() but applied
// to a [C, T] (ne0=C, ne1=T) tensor, broadcasting the per-channel scale/shift over
// the time axis (ne1). All dense-backbone BN prefixes are affine (the affine-free
// embedding-head BN at the very end stays on the time-major bn1d path).
static ggml_tensor* bn1d_cm(ggml_context* c, const ModelLoader& ml,
                            const std::string& prefix, ggml_tensor* x, float eps,
                            std::vector<std::vector<float>>& scratch) {
    std::vector<float> g, beta, mu, var;
    weight_to_host_f32(ml, (prefix + ".weight").c_str(), g);
    weight_to_host_f32(ml, (prefix + ".bias").c_str(), beta);
    weight_to_host_f32(ml, (prefix + ".running_mean").c_str(), mu);
    weight_to_host_f32(ml, (prefix + ".running_var").c_str(), var);
    const int C = (int)mu.size();
    scratch.emplace_back(C);
    scratch.emplace_back(C);
    std::vector<float>& scale = scratch[scratch.size() - 2];
    std::vector<float>& shift = scratch[scratch.size() - 1];
    for (int i = 0; i < C; ++i) {
        const float inv = g[i] / std::sqrt(var[i] + eps);
        scale[i] = inv;
        shift[i] = beta[i] - inv * mu[i];
    }
    const int64_t ne[2] = { C, 1 };
    ggml_tensor* s  = graph_input_tensor(c, GGML_TYPE_F32, 2, ne,
                                         scale.data(), (size_t)C * sizeof(float));
    ggml_tensor* sh = graph_input_tensor(c, GGML_TYPE_F32, 2, ne,
                                         shift.data(), (size_t)C * sizeof(float));
    return ggml_add(c, ggml_mul(c, x, s), sh);  // [C, T]
}

// One FCM BasicResBlock: conv1 (3x3, stride on freq) -> ReLU -> conv2 (3x3) ->
// + shortcut (1x1 stride conv when stride!=1, else identity) -> ReLU. All BN is
// folded into the convs. `stride` is the frequency-axis stride.
static ggml_tensor* fcm_basicres(ggml_context* c, const ModelLoader& ml, const CCfg& cp,
                                 int& idx, ggml_tensor* x, int stride) {
    ggml_tensor* out = ggml_relu(c, next_conv2d(c, ml, cp, idx, x, stride, 1)); // conv1
    out = next_conv2d(c, ml, cp, idx, out, 1, 1);                               // conv2
    ggml_tensor* res = (stride != 1) ? next_conv2d(c, ml, cp, idx, x, stride, 0) // 1x1 shortcut
                                     : x;
    return ggml_relu(c, ggml_add(c, out, res));
}

std::vector<float> CamPPlusEncoder::forward_capture(const std::vector<float>& feats, int T,
                                                    std::vector<float>& pre_norm) const {
    const VoiceDetectConfig& cfg = ml_.config();
    const CCfg& cp = cfg.campplus;
    const int   nmel = (int)cfg.n_mels;
    const float eps  = cp.bn_eps;
    const std::string emb_mean = cp.emb_bn_mean;
    const std::string emb_var  = cp.emb_bn_var;

    // Backbone time length after the stride-2 TDNN (k=5, pad=2): T' = (T-1)/2 + 1.
    // Constant across the dense backbone (every later conv is "same"), so the CAM
    // context time-mixing matrix is built once below.
    const int Tp = (T - 1) / 2 + 1;

    // CAM context = c.mean(-1, keepdim) + seg100(c): both are linear over the time
    // axis. The dense Q[t', t] = 1/T' + (1/100 if same 100-frame segment) is rank
    // (1 + n_seg): Q = R . E with R[t, j] the per-frame pool weights and E[j, t']
    // the per-frame broadcast. Column 0 is the rank-1 global mean (1/T' everywhere,
    // broadcast to all t'); column 1+s is the s-th 100-frame count_include_pad
    // segment average (ONNX AveragePool ceil_mode=1, count_include_pad=1 -> the
    // partial last segment is still divided by 100), broadcast back to the frames
    // of that segment. Applying R then E is O(T' . n_seg) instead of the O(T'^2)
    // dense [T', T'] matmul, and is algebraically identical. n_seg ~ 2 for T' ~ 149.
    const int nseg  = (Tp + 99) / 100;
    const int ncols = 1 + nseg;
    std::vector<float> R((size_t)Tp * ncols, 0.0f);   // [T', ncols] pool weights
    std::vector<float> E((size_t)ncols * Tp, 0.0f);   // [ncols, T'] broadcast
    for (int t = 0; t < Tp; ++t) {
        R[(size_t)0 * Tp + t]             = 1.0f / (float)Tp;   // col 0: global mean
        R[(size_t)(1 + t / 100) * Tp + t] = 1.0f / 100.0f;      // col 1+s: seg pool
    }
    for (int tp = 0; tp < Tp; ++tp) {
        E[(size_t)tp * ncols + 0]             = 1.0f;           // global -> every frame
        E[(size_t)tp * ncols + (1 + tp / 100)] = 1.0f;          // seg -> its frames
    }

    // num_layers / dilation per CAMDenseTDNN block (fixed CAM++ topology).
    const int block_layers[3] = { 12, 24, 16 };
    const int block_dil[3]    = { 1, 2, 2 };

    std::vector<std::vector<float>> bn_scratch;
    bn_scratch.reserve(160);  // scale+shift per BN (56) plus stats/embedding

    std::vector<float> out;
    global_backend().compute([&](ggml_context* c) -> ggml_tensor* {
        // FBank features feat-major [n_mels, T] map DIRECTLY to a [W=T, H=freq,
        // IC=1, N=1] conv input: element (t, m) = feats[m*T + t]. This is the FCM's
        // [B, 1, F, T] input (after the forward's permute(0,2,1) + unsqueeze(1)).
        const int64_t ne[4] = { T, nmel, 1, 1 };
        ggml_tensor* x = graph_input_tensor(c, GGML_TYPE_F32, 4, ne,
                                            feats.data(), feats.size() * sizeof(float));

        int ci = 0;     // running conv index (225 total)
        int bi = 0;     // running affine-BN index (55 affine prefixes)

        // FCM front end: 3x3 stem (BN folded) -> ReLU; two BasicResBlock stages
        // (each: a stride-2 block then a stride-1 block) halving the freq axis
        // twice; a final 3x3 stride-2 conv -> ReLU. Freq 80 -> 10.
        //
        // The FCM is a genuine 2D 3x3-stride-1 conv stack (over [W=T, H=freq, C]
        // maps), so its stride-1 3x3 pad-1 convs are Winograd candidates just like
        // the other 2D encoders (WeSpeaker ResNet34, ERes2Net). Opt in for the FCM
        // build only: conv2d_auto's guard routes ONLY the stride-1 3x3 convs (the
        // conv1/conv2 of each BasicResBlock and the stem) through winograd_conv3x3,
        // while the freq-axis stride-2 downsample convs and the 1x1 shortcuts fall
        // through to im2col untouched. The dense TDNN backbone below is 1D
        // (next_conv1d, never conv2d_auto), so the scope has no effect there. The
        // FCM's channel counts are low (~32), where the direct conv's win over
        // Winograd evaporates: measured a statistical tie at 8t (~17.8 vs ~17.5 ms)
        // and only ~1.05x at 1t on a fraction-of-total stage. So the FCM is PINNED
        // to the AVX-512/AVX2 Winograd kernel here (the per-model fallback), unlike
        // WeSpeaker/ERes2Net which default to Direct. parity is the gate either way.
        ggml_tensor* h;
        {
            WinogradScope fcm_wino(true, Conv3x3Kernel::Winograd);
            h = ggml_relu(c, next_conv2d(c, ml_, cp, ci, x, 1, 1));  // conv1
            h = fcm_basicres(c, ml_, cp, ci, h, 2);   // layer1 block0 (freq 80->40)
            h = fcm_basicres(c, ml_, cp, ci, h, 1);   // layer1 block1
            h = fcm_basicres(c, ml_, cp, ci, h, 2);   // layer2 block0 (freq 40->20)
            h = fcm_basicres(c, ml_, cp, ci, h, 1);   // layer2 block1
            h = ggml_relu(c, next_conv2d(c, ml_, cp, ci, h, 2, 1));          // conv2 (freq 20->10)
        }

        // Reshape [W=T, H=10, C=32] -> [T, 320] (channel = c*10 + h, the torch
        // reshape(B, C*H, T) C-outer / freq-inner order falls out of the contiguous
        // ggml memory directly).
        h = ggml_reshape_2d(c, ggml_cont(c, h), T, h->ne[1] * h->ne[2]);     // [T, 320]

        // TDNN: Conv1d(320 -> 128, k=5, stride=2, pad=2) (BN folded) -> ReLU. Reads
        // the time-major FCM output via im2col and EMITS channel-major [128, T'] -
        // the layout the whole dense backbone now keeps.
        h = ggml_relu(c, next_conv1d(c, ml_, cp, ci, h, /*stride=*/2, /*pad=*/2, /*dil=*/1));

        const int64_t rne[2] = { Tp, ncols };
        ggml_tensor* Rt = graph_input_tensor(c, GGML_TYPE_F32, 2, rne,
                                             R.data(), R.size() * sizeof(float));
        const int64_t ene[2] = { ncols, Tp };
        ggml_tensor* Et = graph_input_tensor(c, GGML_TYPE_F32, 2, ene,
                                             E.data(), E.size() * sizeof(float));

        // Three CAMDenseTDNN blocks, each followed by a channel-halving transit.
        // All tensors stay CHANNEL-MAJOR [C, T'] here.
        for (int blk = 0; blk < 3; ++blk) {
            const int dil = block_dil[blk];
            for (int l = 0; l < block_layers[blk]; ++l) {
                // nonlinear1: BN -> ReLU; bn_function linear1: 1x1 conv (nonlinear2
                // BN folded in) -> nonlinear2 ReLU -> the CAMLayer input `c`.
                ggml_tensor* a = ggml_relu(c, bn1d_cm(c, ml_, cp.bn_prefix[bi++], h, eps,
                                                      bn_scratch));
                ggml_tensor* cc = ggml_relu(c,
                    next_conv1d(c, ml_, cp, ci, a, 1, 0, 1));               // [128, T']
                // Shared time-major view of cc for both time-mixing consumers: the
                // dilated linear_local (im2col wants [T', C]) and the CAM context.
                ggml_tensor* cct = ggml_cont(c, ggml_transpose(c, cc));     // [T', 128]
                // CAMLayer: y = linear_local(cc); m = sigmoid(linear2(relu(linear1(
                // context)))); out = y * m. context = mean + seg100, factored R . E:
                // pool over time (cct . R) then broadcast back (. E), both O(T').
                ggml_tensor* y = next_conv1d(c, ml_, cp, ci, cct, 1, dil, dil); // [32, T']
                ggml_tensor* pooled = ggml_mul_mat(c, Rt, cct);             // [ncols, 128]
                ggml_tensor* ctx = ggml_mul_mat(c, pooled, Et);             // [128, T']
                ctx = ggml_relu(c, next_conv1d(c, ml_, cp, ci, ctx, 1, 0, 1));// [64, T']
                ggml_tensor* m = ggml_sigmoid(c,
                    next_conv1d(c, ml_, cp, ci, ctx, 1, 0, 1));             // [32, T']
                ggml_tensor* o = ggml_mul(c, y, m);                          // [32, T']
                h = ggml_concat(c, h, o, /*dim=*/0);                        // dense cat (channel)
            }
            // Transit: nonlinear (BN -> ReLU) -> 1x1 conv (channel // 2). The LAST
            // transit folds the out_nonlinear BN into its conv (so it carries a
            // bias) and is followed by the out ReLU before stats pooling.
            ggml_tensor* t = ggml_relu(c, bn1d_cm(c, ml_, cp.bn_prefix[bi++], h, eps,
                                                  bn_scratch));
            t = next_conv1d(c, ml_, cp, ci, t, 1, 0, 1);
            h = (blk == 2) ? ggml_relu(c, t) : t;
        }

        // Back to TIME-MAJOR [T', C] for the (unchanged) stats pooling + dense head.
        h = ggml_cont(c, ggml_transpose(c, h));                             // [T', C]

        // Statistics pooling: per-channel mean + unbiased std over time -> [2C].
        // std = sqrt(var * T'/(T'-1)) (no eps; torch.std unbiased).
        ggml_tensor* mean = ggml_mean(c, h);                                // [1, C]
        ggml_tensor* dev  = ggml_sub(c, h, mean);
        ggml_tensor* var  = ggml_mean(c, ggml_sqr(c, dev));                 // biased [1, C]
        if (Tp > 1) var = ggml_scale(c, var, (float)Tp / (float)(Tp - 1));
        ggml_tensor* std  = ggml_sqrt(c, var);
        const int64_t C = mean->ne[1];
        ggml_tensor* mean_f = ggml_reshape_1d(c, ggml_cont(c, mean), C);
        ggml_tensor* std_f  = ggml_reshape_1d(c, ggml_cont(c, std), C);
        ggml_tensor* stats  = ggml_concat(c, mean_f, std_f, /*dim=*/0);     // [2C]

        // Dense: Conv1d(2C -> emb, 1x1, no bias) == a plain matmul over the stats.
        ggml_tensor* dw = clone_weight(c, ml_, cp.conv_weight[ci++].c_str()); // [1, 2C, emb]
        ggml_tensor* emb = ggml_mul_mat(c, ggml_reshape_2d(c, dw, dw->ne[1], dw->ne[2]),
                                        ggml_reshape_2d(c, stats, stats->ne[0], 1)); // [emb, 1]
        emb = ggml_reshape_2d(c, emb, 1, emb->ne[0]);                       // [1, emb] (T=1)

        // Final affine-free BatchNorm (the embedding head). pre-L2 encoder output.
        emb = bn1d(c, ml_, "", emb, eps, emb_mean, emb_var, bn_scratch);
        return ggml_reshape_1d(c, emb, emb->ne[1]);                         // [emb]
    }, out);

    pre_norm = out;  // raw ONNX embedding equivalent (encoder_out parity)

    // L2-normalize host-side so cosine == dot product, matching the golden.
    if (cfg.l2_normalize) {
        double n2 = 0.0;
        for (float v : out) n2 += (double)v * (double)v;
        const double norm = std::sqrt(n2);
        if (norm > 0.0)
            for (float& v : out) v = (float)((double)v / norm);
    }
    return out;
}

std::vector<float> CamPPlusEncoder::forward(const std::vector<float>& feats, int T) const {
    std::vector<float> pre_norm;
    return forward_capture(feats, T, pre_norm);
}

} // namespace vd
