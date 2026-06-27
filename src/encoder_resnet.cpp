#include "encoder.hpp"
#include "backend.hpp"
#include "directconv.hpp"
#include "model_loader.hpp"
#include "ggml.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace vd {

ResNet34Encoder::ResNet34Encoder(const ModelLoader& ml) : ml_(ml) {}

// Conv2d (ggml) + per-channel bias. The WeSpeaker ONNX folds BatchNorm into every
// Conv, so each conv carries weight + bias and there is no separate BN. ONNX
// weights are [OC, IC, KH, KW] -> ggml ne = [KW, KH, IC, OC], exactly the layout
// ggml_conv_2d wants. Input/output are [W=time, H=freq, C, N]. ONNX uses
// symmetric stride and pad 1 for the 3x3 convs, pad 0 for the 1x1 downsample.
static ggml_tensor* conv2d_bias(ggml_context* c, const ModelLoader& ml,
                                const std::string& w_name, const std::string& b_name,
                                ggml_tensor* x, int stride, int pad) {
    ggml_tensor* w = clone_weight(c, ml, w_name.c_str());          // [KW, KH, IC, OC]
    ggml_tensor* y = conv2d_auto(c, w, x, stride, stride, pad, pad, 1, 1);  // [OW,OH,OC,N]
    if (!b_name.empty()) {
        ggml_tensor* b = clone_weight(c, ml, b_name.c_str());      // [OC]
        y = ggml_add(c, y, ggml_reshape_4d(c, b, 1, 1, b->ne[0], 1));
    }
    return y;
}

// ResNet BasicBlock: conv1(3x3, stride) -> ReLU -> conv2(3x3, 1) -> +shortcut ->
// ReLU. The shortcut is a 1x1 conv (stride) when the block changes channels /
// downsamples, else the identity. The folded BN means conv1/conv2/downsample are
// plain conv+bias.
static ggml_tensor* basic_block(ggml_context* c, const ModelLoader& ml,
                                const VoiceDetectConfig::ResNetConfig& r, size_t i,
                                ggml_tensor* x) {
    const int stride = r.stride[i];
    ggml_tensor* h = conv2d_bias(c, ml, r.conv1_weight[i], r.conv1_bias[i], x, stride, 1);
    h = ggml_relu(c, h);
    h = conv2d_bias(c, ml, r.conv2_weight[i], r.conv2_bias[i], h, 1, 1);
    ggml_tensor* shortcut = x;
    if (!r.down_weight[i].empty())
        shortcut = conv2d_bias(c, ml, r.down_weight[i], r.down_bias[i], x, stride, 0);
    return ggml_relu(c, ggml_add(c, h, shortcut));
}

// Blocked-layout (nChw16c) BasicBlock: conv1(3x3,stride) -> bias -> ReLU ->
// conv2(3x3,1) -> bias -> +shortcut -> ReLU, all on the blocked buffer so the
// backbone never leaves nChw16c. Mirrors basic_block exactly, op-for-op; the
// only difference is the layout the ops run in (so it is parity-equivalent).
static ggml_tensor* basic_block_blocked(ggml_context* c, const ModelLoader& ml,
                                        const VoiceDetectConfig::ResNetConfig& r, size_t i,
                                        ggml_tensor* xb) {
    const int stride = r.stride[i];
    ggml_tensor* w1 = clone_weight(c, ml, r.conv1_weight[i].c_str());   // [3,3,IC,OC]
    ggml_tensor* b1 = clone_weight(c, ml, r.conv1_bias[i].c_str());     // [OC]
    // conv1: fuse bias + ReLU into the conv epilogue (one op, no extra passes).
    ggml_tensor* h = blocked_conv3x3(c, w1, xb, /*pad=*/1, stride, b1, /*relu=*/true);
    ggml_tensor* w2 = clone_weight(c, ml, r.conv2_weight[i].c_str());
    ggml_tensor* b2 = clone_weight(c, ml, r.conv2_bias[i].c_str());
    // conv2: fuse bias only (the ReLU follows the residual add).
    h = blocked_conv3x3(c, w2, h, /*pad=*/1, /*stride=*/1, b2, /*relu=*/false);
    ggml_tensor* shortcut = xb;
    if (!r.down_weight[i].empty()) {
        ggml_tensor* dw = clone_weight(c, ml, r.down_weight[i].c_str());  // [1,1,IC,OC]
        ggml_tensor* db = clone_weight(c, ml, r.down_bias[i].c_str());
        shortcut = blocked_conv1x1(c, dw, xb, stride, db, /*relu=*/false);  // fuse bias
    }
    return blocked_add_relu(c, h, shortcut);  // fused residual add + ReLU
}

// Number of leading BasicBlocks to run inside the blocked island.
// VD_BLOCKED_BACKBONE override: "0"/"off" forces the per-conv directconv path;
// "all"/"on" forces every block blocked (the whole backbone is ONE island, exactly
// 2 reorders); a positive integer runs that many leading blocks blocked (used for
// incremental island growth + re-gating). When the env is UNSET the default is the
// whole backbone IFF the AVX-512 blocked fast path is available at runtime (it beat
// the per-conv directconv: WeSpeaker ~64 vs ~67 ms @1t, ~9.0 vs ~11.1 ms @8t at
// cosine 1.0); on non-AVX512 hosts it defaults OFF so they keep the AVX2 directconv.
static int blocked_island_blocks(size_t n_blocks) {
    const char* e = std::getenv("VD_BLOCKED_BACKBONE");
    if (!e || !e[0]) return directconv_blocked_available() ? (int)n_blocks : 0;
    if (!std::strcmp(e, "0") || !std::strcmp(e, "off")) return 0;
    if (!std::strcmp(e, "all") || !std::strcmp(e, "on")) return (int)n_blocks;
    int k = std::atoi(e);
    if (k < 0) k = 0;
    if (k > (int)n_blocks) k = (int)n_blocks;
    return k;
}

std::vector<float> ResNet34Encoder::forward_capture(const std::vector<float>& feats, int T,
                                                    std::vector<float>& pre_norm) const {
    const VoiceDetectConfig& cfg = ml_.config();
    const VoiceDetectConfig::ResNetConfig& r = cfg.resnet;
    const int nmel = (int)cfg.n_mels;
    const float eps = r.var_eps;  // must outlive compute() (deferred graph-input copy)

    // Route this encoder's 3x3 stride-1 convs (the entire all-3x3 ResNet34
    // backbone) through the AVX2 Winograd kernel. Scoped so only WeSpeaker/ERes2Net
    // opt in; other 2D-conv callers (CAM++ FCM) keep the im2col path bit-identical.
    WinogradScope wino_scope(true);
    std::vector<float> out;
    global_backend().compute([&](ggml_context* c) -> ggml_tensor* {
        // FBank features feat-major [n_mels, T] (feats[m*T + t]) map DIRECTLY to a
        // [W=T, H=n_mels, IC=1, N=1] conv input: element (t, m) = feats[m*T + t],
        // i.e. spatial (time=t, freq=m). This is the ONNX [B,1,80,T] conv input
        // after its transpose+unsqueeze, with no extra reshuffle.
        const int64_t ne[4] = { T, nmel, 1, 1 };
        ggml_tensor* x = graph_input_tensor(c, GGML_TYPE_F32, 4, ne,
                                            feats.data(), feats.size() * sizeof(float));

        // Stem: 3x3 conv (stride 1, pad 1) -> ReLU.
        ggml_tensor* h = conv2d_bias(c, ml_, r.stem_weight, r.stem_bias, x, 1, 1);
        h = ggml_relu(c, h);

        // 4 ResNet stages of BasicBlocks (topology + stride from the manifest;
        // channel widths come from each conv weight's ne, no literals here).
        //
        // Blocked-island path (VD_BLOCKED_BACKBONE): reorder NCHW->nChw16c ONCE
        // here, run the leading `kb` blocks entirely in the blocked layout (no
        // per-conv reorder), reorder nChw16c->NCHW ONCE, then run any remaining
        // blocks on the per-conv directconv path. kb == n_blocks() makes the whole
        // backbone a single island with exactly 2 reorders (vs ~60 per-conv).
        const int kb = blocked_island_blocks(r.n_blocks());
        size_t i = 0;
        if (kb > 0) {
            ggml_tensor* xb = blocked_reorder_in(c, h);            // -> [16,W,H,CB]
            for (; i < (size_t)kb; ++i)
                xb = basic_block_blocked(c, ml_, r, i, xb);
            // true channel count after the last blocked block = its conv2 OC.
            const int outC = (int)clone_weight(c, ml_, r.conv2_weight[kb - 1].c_str())->ne[3];
            h = blocked_reorder_out(c, xb, outC);                  // -> [W',H',C,1]
        }
        for (; i < r.n_blocks(); ++i)
            h = basic_block(c, ml_, r, i, h);  // [W', H', 256, 1]

        // Temporal statistics pooling: per (channel, freq) mean + std over the
        // time axis (ne[0]=W'), preserving the H'=10 frequency resolution. ONNX
        // uses the UNBIASED variance (divide by N-1) plus eps inside the sqrt.
        const int N = (int)h->ne[0];                          // frames after pooling input
        ggml_tensor* mean = ggml_mean(c, h);                  // [1, H', C, 1]
        ggml_tensor* dev  = ggml_sub(c, h, mean);             // [W', H', C, 1] (mean bcast)
        ggml_tensor* var  = ggml_mean(c, ggml_sqr(c, dev));   // biased var [1, H', C, 1]
        if (N > 1) var = ggml_scale(c, var, (float)N / (float)(N - 1));  // -> unbiased
        const int64_t one[1] = { 1 };
        ggml_tensor* eps_t = graph_input_tensor(c, GGML_TYPE_F32, 1, one, &eps, sizeof(float));
        ggml_tensor* std_t = ggml_sqrt(c, ggml_add(c, var, eps_t));     // [1, H', C, 1]

        // Flatten each stat to [C*H'] in C-outer/H-inner order (the ONNX Flatten
        // axis=1 over [B, C, H']), then concat [mean ; std] -> [2*C*H'] = 5120.
        const int64_t HC = mean->ne[1] * mean->ne[2];
        ggml_tensor* mean_f = ggml_reshape_1d(c, ggml_cont(c, mean), HC);
        ggml_tensor* std_f  = ggml_reshape_1d(c, ggml_cont(c, std_t), HC);
        ggml_tensor* pooled = ggml_concat(c, mean_f, std_f, /*dim=*/0);  // [5120]

        // Embedding FC: seg_1 Gemm (transB) -> + bias -> - mean_vec. seg weight
        // ONNX [emb_dim, 5120] -> ggml ne = [5120, emb_dim], so mul_mat gives the
        // transposed-B product directly.
        ggml_tensor* W = clone_weight(c, ml_, r.seg_weight.c_str());     // [5120, emb_dim]
        ggml_tensor* emb = ggml_mul_mat(c, W, pooled);                  // [emb_dim, 1]
        ggml_tensor* sb = clone_weight(c, ml_, r.seg_bias.c_str());     // [emb_dim]
        emb = ggml_add(c, emb, sb);
        if (!r.mean_vec.empty()) {
            ggml_tensor* mv = clone_weight(c, ml_, r.mean_vec.c_str()); // [emb_dim]
            emb = ggml_sub(c, emb, mv);
        }
        return ggml_reshape_1d(c, emb, emb->ne[0]);                     // [emb_dim], pre-norm
    }, out);

    pre_norm = out;  // raw ONNX `embs` equivalent (encoder_out parity)

    // L2-normalize host-side so the embedding sits on the unit sphere (cosine ==
    // dot product), matching the golden's unit-normalized embedding.
    if (cfg.l2_normalize) {
        double n2 = 0.0;
        for (float v : out) n2 += (double)v * (double)v;
        const double norm = std::sqrt(n2);
        if (norm > 0.0)
            for (float& v : out) v = (float)((double)v / norm);
    }
    return out;
}

std::vector<float> ResNet34Encoder::forward(const std::vector<float>& feats, int T) const {
    std::vector<float> pre_norm;
    return forward_capture(feats, T, pre_norm);
}

} // namespace vd
