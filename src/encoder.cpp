#include "encoder.hpp"
#include "backend.hpp"
#include "model_loader.hpp"
#include "ggml.h"
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

namespace vd {

// ECAPA-TDNN BatchNorm1d epsilon (PyTorch default).
static constexpr float kBnEps = 1e-5f;

// SpeechBrain AttentiveStatisticsPooling variance floor (its `self.eps`). The
// weighted std is sqrt(clamp(weighted_var, kAspEps)); matching this exact value
// (and the clamp, not an additive eps) is parity-critical for near-constant
// channels where the variance underflows toward zero.
static constexpr float kAspEps = 1e-12f;

// Reflect-pad the time axis (ne[0]) of `x` ne = [T, C] by `pad` on each side,
// returning [T + 2*pad, C]. SpeechBrain's Conv1d defaults to padding_mode
// "reflect" (NOT zero padding); matching it is parity-critical at the edge
// frames. ggml has no reflect-pad op, so we build the reflected time index list
// on the host and gather with ggml_get_rows (which indexes ne[1]): transpose
// x to [C, T], gather [C, T+2*pad], transpose back. This works for any graph
// tensor, so every block reuses it.
//
// `idx_scratch` owns the host index buffers; they must outlive Backend::compute
// (graph_input_tensor registers a deferred copy). std::vector move preserves the
// inner buffer pointers across outer reallocation.
static ggml_tensor* reflect_pad_time(ggml_context* c, ggml_tensor* x, int pad,
                                     std::vector<std::vector<int32_t>>& idx_scratch) {
    if (pad <= 0) return x;
    const int T  = (int)x->ne[0];
    const int Tp = T + 2 * pad;
    idx_scratch.emplace_back(Tp);
    std::vector<int32_t>& idx = idx_scratch.back();
    for (int j = 0; j < Tp; ++j) {
        int t;
        if (j < pad)            t = pad - j;            // left reflect (no edge repeat)
        else if (j < pad + T)   t = j - pad;            // body
        else                    t = (T - 2) - (j - pad - T);  // right reflect
        idx[j] = t;
    }
    const int64_t ne[1] = { Tp };
    ggml_tensor* idx_t = graph_input_tensor(c, GGML_TYPE_I32, 1, ne,
                                            idx.data(), (size_t)Tp * sizeof(int32_t));
    ggml_tensor* xt = ggml_cont(c, ggml_transpose(c, x));  // [C, T]
    ggml_tensor* g  = ggml_get_rows(c, xt, idx_t);         // [C, Tp]
    return ggml_cont(c, ggml_transpose(c, g));             // [Tp, C]
}

// Conv1d with TDNN "same" + reflect padding (SpeechBrain semantics), stride 1.
// `w` ne = [kernel, in_ch, out_ch]; `x` ne = [T, in_ch]; bias `b` ne = [out_ch]
// (added broadcast per output channel). Returns ne = [T, out_ch].
//
// NOTE: ggml_conv_1d() hardcodes an F16 im2col that asserts an F16 kernel, but
// the converter keeps these conv weights F32. We reflect-pad, then run im2col +
// mul_mat in F32 ourselves (the same decomposition ggml_conv_1d uses, with
// pad=0 since padding is already applied) to keep full precision.
static ggml_tensor* conv1d_same(ggml_context* c, ggml_tensor* w, ggml_tensor* b,
                                ggml_tensor* x, int dilation,
                                std::vector<std::vector<int32_t>>& idx_scratch) {
    // Quantized 1x1 path: the converter stores allowlisted 1x1 Conv1d / linear
    // weights squeezed to a 2-D [IC, OC] matrix (ggml ne) and block-quantized
    // (q8_0/q4_0/f16) so they can ride ggml_mul_mat as src0 (the operand ggml
    // dequantizes on the fly). A 1x1 conv is a plain per-frame matmul, so there
    // is no im2col: x [T, IC] -> mul_mat(w[IC, OC], x^T[IC, T]) -> [OC, T] ->
    // transpose -> [T, OC]. (Reflect-pad/dilation are no-ops for k=1.) The F32
    // path below is byte-for-byte unchanged.
    if (w->type != GGML_TYPE_F32) {
        ggml_tensor* xt = ggml_cont(c, ggml_transpose(c, x));  // [IC, T]
        ggml_tensor* y  = ggml_mul_mat(c, w, xt);              // [OC, T]
        y = ggml_cont(c, ggml_transpose(c, y));               // [T, OC]
        if (b) y = ggml_add(c, y, ggml_reshape_2d(c, b, 1, b->ne[0]));
        return y;
    }
    const int k = (int)w->ne[0];
    const int pad = ((k - 1) / 2) * dilation;
    x = reflect_pad_time(c, x, pad, idx_scratch);  // [T + 2*pad, IC]
    ggml_tensor* im2col = ggml_im2col(c, w, x, /*s0=*/1, /*s1=*/0, /*p0=*/0,
                                      /*p1=*/0, /*d0=*/dilation, /*d1=*/0,
                                      /*is_2D=*/false, GGML_TYPE_F32);  // [IC*K, OL]
    ggml_tensor* im2 = ggml_reshape_2d(c, im2col, im2col->ne[0], im2col->ne[1]); // [IC*K, OL]
    ggml_tensor* w2  = ggml_reshape_2d(c, w, w->ne[0] * w->ne[1], w->ne[2]);     // [IC*K, OC]
    ggml_tensor* y;
    // PROFILE-DRIVEN (channel-major operand order): ggml_mul_mat is ~2x faster when
    // the STATIC weight is src0 and the activation is src1 (src1 is converted to the
    // vec_dot type each call; the smaller activation is far cheaper to convert than
    // the larger weight, and the result tiling is better). The original F32 path put
    // the activation (im2col) as src0 to land output [OL=T, OC] (time-major) for free.
    // Instead compute channel-major y'=[OC, OL] with the weight as src0, then reorder
    // back to [T, OC] at the boundary. mul_mat(A,B)^T == mul_mat(B,A) algebraically;
    // weight-as-src0 routes through the tinyBLAS (llamafile) sgemm, whose blocked
    // accumulation differs from the generic path only at f32 rounding (~4e-7 max|d|,
    // embedding cosine=1.000000 vs the golden - well inside the >=0.9999/1e-3 gate).
    // VD_MM_ACT_SRC0=1 restores the old (slow) order for A/B. Measured ~2x on the big
    // pointwise GEMMs (mfa 3072x3072, tdnn 1024x1024); tiny reorder cost on the
    // 128-ch res2net k=3 convs (the only k>1 convs, so no reorder trap).
    if (std::getenv("VD_MM_ACT_SRC0")) {
        y = ggml_mul_mat(c, im2, w2);                       // [OL=T, OC] (old slow order)
    } else {
        ggml_tensor* yt = ggml_mul_mat(c, w2, im2);         // [OC, OL] channel-major
        y = ggml_cont(c, ggml_transpose(c, yt));            // [OL=T, OC] boundary reorder
    }
    if (b) y = ggml_add(c, y, ggml_reshape_2d(c, b, 1, b->ne[0]));
    return y;  // [T, out_ch]
}

// BatchNorm1d in eval mode folded to a per-channel affine transform. PyTorch BN
// in eval applies y = gamma * (x - mu) / sqrt(var + eps) + beta with the running
// statistics, which is a fixed (input-independent) scale/shift per channel. We
// precompute scale = gamma / sqrt(var + eps) and shift = beta - scale * mu on the
// host, then apply them as broadcast graph constants. This avoids running
// sqrt/div in the graph (and avoids needing a host-backed eps scalar tensor under
// the backend's no_alloc build context). `x` ne = [T, C]; the constants are
// reshaped to [1, C] so ggml broadcasts them along the time axis.
//
// `scratch` owns the host-side scale/shift buffers; they must outlive the
// enclosing Backend::compute() call (graph_input_tensor registers a deferred
// copy of their bytes performed after graph allocation). std::vector move
// preserves the inner buffer pointers across outer reallocation, so the
// registered .data() pointers stay valid even as more blocks push entries.
static ggml_tensor* bn1d_eval(ggml_context* c, const ModelLoader& ml,
                              const std::string& p, ggml_tensor* x,
                              std::vector<std::vector<float>>& scratch) {
    std::vector<float> g, beta, mu, var;
    weight_to_host_f32(ml, (p + ".weight").c_str(), g);
    weight_to_host_f32(ml, (p + ".bias").c_str(), beta);
    weight_to_host_f32(ml, (p + ".running_mean").c_str(), mu);
    weight_to_host_f32(ml, (p + ".running_var").c_str(), var);
    const int C = (int)g.size();

    scratch.emplace_back(C);
    scratch.emplace_back(C);
    std::vector<float>& scale = scratch[scratch.size() - 2];
    std::vector<float>& shift = scratch[scratch.size() - 1];
    for (int i = 0; i < C; ++i) {
        const float inv = g[i] / std::sqrt(var[i] + kBnEps);
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

// TDNNBlock = Conv1d(k, dilation, "same") -> ReLU -> BatchNorm1d (the SpeechBrain
// frontend unit, also used inside Res2Net). `prefix` names the submodule, e.g.
// "blocks.1.tdnn1": its conv weights are `<prefix>.conv.conv.{weight,bias}` and
// its BN is `<prefix>.norm.norm.*`. Returns ne = [T, out_ch].
static ggml_tensor* tdnn_block(ggml_context* c, const ModelLoader& ml,
                               const std::string& prefix, ggml_tensor* x,
                               int dilation,
                               std::vector<std::vector<int32_t>>& idx_scratch,
                               std::vector<std::vector<float>>& bn_scratch) {
    ggml_tensor* w = clone_weight(c, ml, (prefix + ".conv.conv.weight").c_str());
    ggml_tensor* b = clone_weight(c, ml, (prefix + ".conv.conv.bias").c_str());
    ggml_tensor* h = conv1d_same(c, w, b, x, dilation, idx_scratch);
    h = ggml_relu(c, h);
    return bn1d_eval(c, ml, prefix + ".norm.norm", h, bn_scratch);
}

// Res2NetBlock (SpeechBrain): split the `scale` groups along the channel axis
// (ne[1]); group 0 is a passthrough, group i (>=1) is fed through TDNNBlock
// `blocks.{i-1}` after adding the PREVIOUS group's output (hierarchical residual,
// dim=1 chunk convention). The k=3, dilated convs are concatenated back along the
// channel axis. `prefix` = "blocks.1.res2net_block". Returns ne = [T, C].
static ggml_tensor* res2net_block(ggml_context* c, const ModelLoader& ml,
                                  const std::string& prefix, ggml_tensor* x,
                                  int scale, int dilation,
                                  std::vector<std::vector<int32_t>>& idx_scratch,
                                  std::vector<std::vector<float>>& bn_scratch) {
    const int64_t T = x->ne[0];
    const int64_t C = x->ne[1];
    const int64_t g = C / scale;  // channels per group
    ggml_tensor* out    = nullptr;
    ggml_tensor* y_prev = nullptr;
    for (int i = 0; i < scale; ++i) {
        // Group i occupies a contiguous [T, g] slab (channels are ne[1]-outer).
        ggml_tensor* x_i = ggml_cont(c,
            ggml_view_2d(c, x, T, g, x->nb[1], (size_t)i * g * x->nb[1]));
        ggml_tensor* y_i;
        if (i == 0) {
            y_i = x_i;  // passthrough
        } else {
            ggml_tensor* in = (i == 1) ? x_i : ggml_add(c, x_i, y_prev);
            y_i = tdnn_block(c, ml, prefix + ".blocks." + std::to_string(i - 1),
                             in, dilation, idx_scratch, bn_scratch);
        }
        out    = (i == 0) ? y_i : ggml_concat(c, out, y_i, /*dim=*/1);
        y_prev = y_i;
    }
    return out;  // [T, C]
}

// SEBlock (squeeze-excitation): pool the input over time, push it through a
// channel bottleneck (conv1: C->se_channels, ReLU; conv2: se_channels->C,
// sigmoid), and scale every time step of the input by the resulting per-channel
// gate. conv1/conv2 are plain 1x1 Conv1d (bias, NO BatchNorm). `prefix` =
// "blocks.1.se_block". Returns ne = [T, C].
static ggml_tensor* se_block(ggml_context* c, const ModelLoader& ml,
                             const std::string& prefix, ggml_tensor* x,
                             std::vector<std::vector<int32_t>>& idx_scratch) {
    ggml_tensor* s = ggml_mean(c, x);  // mean over time (ne[0]) -> [1, C]
    ggml_tensor* w1 = clone_weight(c, ml, (prefix + ".conv1.conv.weight").c_str());
    ggml_tensor* b1 = clone_weight(c, ml, (prefix + ".conv1.conv.bias").c_str());
    s = conv1d_same(c, w1, b1, s, /*dilation=*/1, idx_scratch);  // [1, se_ch]
    s = ggml_relu(c, s);
    ggml_tensor* w2 = clone_weight(c, ml, (prefix + ".conv2.conv.weight").c_str());
    ggml_tensor* b2 = clone_weight(c, ml, (prefix + ".conv2.conv.bias").c_str());
    s = conv1d_same(c, w2, b2, s, /*dilation=*/1, idx_scratch);  // [1, C]
    s = ggml_sigmoid(c, s);
    return ggml_mul(c, x, s);  // s broadcasts over time (ne[0]=1)
}

// SERes2NetBlock: tdnn1 -> res2net -> tdnn2 -> SE -> residual add of the block
// input. ECAPA's blocks all keep channels constant (in==out) so there is no
// projection shortcut: the residual is the raw block input. Dilation per block
// `idx` is `idx+1` (SpeechBrain uses 2,3,4 for blocks 1..3). Returns ne = [T, C].
static ggml_tensor* se_res2net_block(ggml_context* c, const ModelLoader& ml,
                                     int idx, ggml_tensor* x, int dilation,
                                     std::vector<std::vector<int32_t>>& idx_scratch,
                                     std::vector<std::vector<float>>& bn_scratch) {
    const std::string p = "blocks." + std::to_string(idx);
    ggml_tensor* residual = x;
    ggml_tensor* h = tdnn_block(c, ml, p + ".tdnn1", x, /*dilation=*/1,
                                idx_scratch, bn_scratch);
    // res2net_scale = 1 + (# of res2net sub-blocks); probe the verbatim manifest.
    int scale = 1;
    while (ml.tensor(p + ".res2net_block.blocks." + std::to_string(scale - 1) +
                     ".conv.conv.weight"))
        ++scale;
    h = res2net_block(c, ml, p + ".res2net_block", h, scale, dilation,
                      idx_scratch, bn_scratch);
    h = tdnn_block(c, ml, p + ".tdnn2", h, /*dilation=*/1, idx_scratch, bn_scratch);
    h = se_block(c, ml, p + ".se_block", h, idx_scratch);
    return ggml_add(c, h, residual);
}

// AttentiveStatisticsPooling (SpeechBrain, global_context=True). Input x ne=[T, C]
// is the MFA output (C=3072). SpeechBrain pools the variable-length time axis into
// a fixed [2C] statistics vector using a per-channel time attention:
//
//   1. Global time stats over uniform weights m=1/T (its `_compute_statistics`):
//        gmean = mean_t(x);  gstd = sqrt(clamp(mean_t((x-gmean)^2), eps)).
//   2. Global context: concat [x ; gmean ; gstd] over channels (3C=9216), each
//      global stat broadcast across all T frames -> the attention TDNN input.
//   3. Attention: `asp.tdnn` (Conv1d 1x1 -> ReLU -> BatchNorm1d) -> tanh ->
//      `asp.conv` (Conv1d 1x1 -> C logits) -> softmax over time (per channel).
//   4. Weighted stats with the attention weights a_t (sum_t a_t = 1 per channel):
//        mu = sum_t a_t*x_t;  sd = sqrt(clamp(sum_t a_t*(x_t-mu)^2, eps)).
//
// Returns concat(mu, sd) ne=[1, 2C]. This is the RAW ASP output: the separate
// `asp_bn` BatchNorm over 2C is a later stage (gen_baseline.py hooks `em.asp`,
// whose output precedes asp_bn), so it is intentionally NOT applied here.
static ggml_tensor* attentive_stat_pooling(ggml_context* c, const ModelLoader& ml,
                                          ggml_tensor* x,
                                          std::vector<std::vector<int32_t>>& idx_scratch,
                                          std::vector<std::vector<float>>& bn_scratch) {
    // Global time statistics (uniform weights -> ggml_mean is sum/T = the weighted
    // mean; the variance is mean_t of the squared deviation).
    ggml_tensor* gmean = ggml_mean(c, x);                                   // [1, C]
    ggml_tensor* gdev  = ggml_sub(c, x, gmean);                             // [T, C]
    ggml_tensor* gvar  = ggml_mean(c, ggml_sqr(c, gdev));                   // [1, C]
    ggml_tensor* gstd  = ggml_sqrt(c, ggml_clamp(c, gvar, kAspEps, 1e30f)); // [1, C]

    // Global context: broadcast the global stats across all T frames and stack
    // them onto x along the channel axis (ne[1]).
    ggml_tensor* mean_b = ggml_repeat(c, gmean, x);                         // [T, C]
    ggml_tensor* std_b  = ggml_repeat(c, gstd, x);                          // [T, C]
    ggml_tensor* ctx = ggml_concat(c, ggml_concat(c, x, mean_b, /*dim=*/1),
                                   std_b, /*dim=*/1);                       // [T, 3C]

    // Attention TDNN -> tanh -> 1x1 conv to per-channel time logits.
    ggml_tensor* a = tdnn_block(c, ml, "asp.tdnn", ctx, /*dilation=*/1,
                                idx_scratch, bn_scratch);                   // [T, att_ch]
    a = ggml_tanh(c, a);
    ggml_tensor* wc = clone_weight(c, ml, "asp.conv.conv.weight");
    ggml_tensor* bc = clone_weight(c, ml, "asp.conv.conv.bias");
    a = conv1d_same(c, wc, bc, a, /*dilation=*/1, idx_scratch);            // [T, C] logits
    a = ggml_soft_max(c, a);  // softmax over time (ne[0]=T) per channel    // [T, C]

    // Attention-weighted mean and std (the weights sum to 1 over time).
    ggml_tensor* mu  = ggml_sum_rows(c, ggml_mul(c, a, x));                 // [1, C]
    ggml_tensor* dev = ggml_sub(c, x, mu);                                 // [T, C]
    ggml_tensor* var = ggml_sum_rows(c, ggml_mul(c, a, ggml_sqr(c, dev))); // [1, C]
    ggml_tensor* sd  = ggml_sqrt(c, ggml_clamp(c, var, kAspEps, 1e30f));    // [1, C]
    return ggml_concat(c, mu, sd, /*dim=*/1);                              // [1, 2C]
}

EcapaEncoder::EcapaEncoder(const ModelLoader& ml) : ml_(ml) {}

std::vector<float> EcapaEncoder::run_forward(const std::vector<float>& feats, int T,
                                             EcapaCaptures& caps, bool capture) const {
    const int nmel = (int)ml_.config().n_mels;

    // Host-side scale/shift buffers for every BatchNorm in this graph. Reserve up
    // front so the buffers (referenced by deferred graph-input copies) are never
    // invalidated mid-build.
    std::vector<std::vector<float>> bn_scratch;
    bn_scratch.reserve(64);

    // Host index buffers for reflect-pad (one per conv); reserved so their data
    // pointers stay valid for the deferred graph-input copies.
    std::vector<std::vector<int32_t>> idx_scratch;
    idx_scratch.reserve(64);

    std::vector<float> out;
    global_backend().compute([&](ggml_context* c) -> ggml_tensor* {
        const int64_t ne[2] = { T, nmel };
        ggml_tensor* x = graph_input_tensor(c, GGML_TYPE_F32, 2, ne,
                                            feats.data(),
                                            feats.size() * sizeof(float));  // [T, nmel]

        // block0: TDNNBlock = Conv1d(k=5, dilation=1, same) -> ReLU -> BatchNorm1d.
        ggml_tensor* w = clone_weight(c, ml_, "blocks.0.conv.conv.weight");
        ggml_tensor* b = clone_weight(c, ml_, "blocks.0.conv.conv.bias");
        ggml_tensor* h = conv1d_same(c, w, b, x, /*dilation=*/1, idx_scratch);
        h = ggml_relu(c, h);
        h = bn1d_eval(c, ml_, "blocks.0.norm.norm", h, bn_scratch);
        if (capture) capture_graph_output(h, &caps.block0);

        // blocks 1..N-1: SE-Res2Net blocks. SpeechBrain's ECAPA-TDNN uses
        // dilation = block_index + 1 (2,3,4 for blocks 1,2,3). The block count is
        // derived from the manifest (presence of `blocks.i.tdnn1.*`) rather than a
        // literal 4, matching `voicedetect.ecapa.n_blocks`. Each block output is
        // collected for multi-layer feature aggregation below.
        std::vector<float>* block_caps[] = { nullptr, &caps.block1, &caps.block2,
                                             &caps.block3 };
        const int n_block_caps = (int)(sizeof(block_caps) / sizeof(block_caps[0]));
        std::vector<ggml_tensor*> block_outs;
        for (int i = 1;
             ml_.tensor("blocks." + std::to_string(i) + ".tdnn1.conv.conv.weight");
             ++i) {
            h = se_res2net_block(c, ml_, /*idx=*/i, h, /*dilation=*/i + 1,
                                 idx_scratch, bn_scratch);
            block_outs.push_back(h);
            if (capture && i < n_block_caps && block_caps[i])
                capture_graph_output(h, block_caps[i]);
        }

        // Multi-layer feature aggregation: concatenate the SE-Res2Net block
        // outputs (xl[1:] in SpeechBrain, NOT block0) along the channel axis
        // (ne[1]) -> 3*1024 = 3072 channels, then a 1x1 TDNNBlock (Conv1d -> ReLU
        // -> BatchNorm1d; `mfa.conv.conv.*` + `mfa.norm.norm.*`).
        ggml_tensor* cat = block_outs[0];
        for (size_t i = 1; i < block_outs.size(); ++i)
            cat = ggml_concat(c, cat, block_outs[i], /*dim=*/1);
        ggml_tensor* mfa = tdnn_block(c, ml_, "mfa", cat, /*dilation=*/1,
                                      idx_scratch, bn_scratch);
        if (capture) capture_graph_output(mfa, &caps.mfa);

        // Attentive statistics pooling: collapse the time axis into the [2*C]
        // mean+std statistics vector (the RAW ASP output, PRE asp_bn). pooled
        // ne = [1, 6144] (T=1, C=2*3072).
        ggml_tensor* pooled = attentive_stat_pooling(c, ml_, mfa, idx_scratch,
                                                     bn_scratch);
        if (capture) capture_graph_output(pooled, &caps.pooled);

        // Embedding head (SpeechBrain ECAPA order: ASP -> asp_bn -> fc):
        //   1. asp_bn: BatchNorm1d over the 6144 pooled vector (`asp_bn.norm.*`).
        //      Folded to the same per-channel affine as every other BN, broadcast
        //      over the single time step.
        //   2. fc: a 1x1 Conv1d projecting 6144 -> 192 (`fc.conv.{weight,bias}`;
        //      verbatim name, NOT `fc.conv.conv.*` - SpeechBrain's `fc` IS a
        //      Conv1d, so it has a single `.conv` submodule). Reuses conv1d_same
        //      with k=1 (pad=0, reflect-pad is a no-op), giving [1, 192].
        ggml_tensor* normed = bn1d_eval(c, ml_, "asp_bn.norm", pooled, bn_scratch);
        ggml_tensor* fw = clone_weight(c, ml_, "fc.conv.weight");
        ggml_tensor* fb = clone_weight(c, ml_, "fc.conv.bias");
        ggml_tensor* fc = conv1d_same(c, fw, fb, normed, /*dilation=*/1, idx_scratch);
        ggml_tensor* emb = ggml_reshape_1d(c, fc, fc->ne[1]);  // [192]
        // emb is the graph's return value, so it is always marked as an output
        // by Backend::compute; the explicit capture only mirrors it into
        // caps.pre_norm for the parity harness.
        if (capture) capture_graph_output(emb, &caps.pre_norm);
        return emb;  // PRE L2-norm; normalized on the host below.
    }, out);

    // L2-normalize host-side so the embedding sits on the unit sphere (cosine ==
    // dot product), matching the golden, which is the raw embedding_model output
    // unit-normalized. Done on the host (the graph output is the pre-norm fc
    // projection captured as caps.pre_norm) to avoid an in-graph sqrt/div over the
    // backend's no_alloc context.
    if (ml_.config().l2_normalize) {
        double n2 = 0.0;
        for (float v : out) n2 += (double)v * (double)v;
        const double norm = std::sqrt(n2);
        if (norm > 0.0)
            for (float& v : out) v = (float)((double)v / norm);
    }

    return out;
}

std::vector<float> EcapaEncoder::forward_capture(const std::vector<float>& feats, int T,
                                                 EcapaCaptures& caps) const {
    return run_forward(feats, T, caps, /*capture=*/true);
}

std::vector<float> EcapaEncoder::forward(const std::vector<float>& feats, int T) const {
    // Production path: skip the harness-only intermediate captures (block0-3,
    // mfa, pooled, pre_norm). Marking them as graph outputs forces a per-tensor
    // readback (~2.7ms) and blocks ggml op fusion; only the final embedding is
    // needed here.
    EcapaCaptures caps;
    return run_forward(feats, T, caps, /*capture=*/false);
}

} // namespace vd
