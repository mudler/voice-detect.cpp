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

ERes2NetEncoder::ERes2NetEncoder(const ModelLoader& ml) : ml_(ml) {}

using ECfg = VoiceDetectConfig::ERes2NetConfig;

// True channel count of a running activation, regardless of layout. NCHW keeps
// channels in ne[2]; the blocked nChw16c buffer keeps ceil(C/16) blocks in ne[3]
// and every ERes2Net split/concat channel count is a multiple of 16, so ne[3]*16
// is exact.
static inline int chans(ggml_tensor* x, bool blk) {
    return blk ? (int)(x->ne[3] * 16) : (int)x->ne[2];
}

// Pull the next conv in forward-EXECUTION order (the manifest is ordered to match
// the deterministic ERes2Net forward) and apply it. NCHW path: conv2d_auto with
// the per-conv stride and SAME-style padding ((K-1)/2 -> 0 for 1x1, 1 for 3x3).
// Blocked path (blk): the SAME conv on the nChw16c buffer via the blocked-island
// kernels (blocked_conv1x1 / blocked_conv3x3), with the per-channel bias FUSED
// into the conv epilogue (parity-equivalent to the NCHW post-add). The ONNX
// export folds BatchNorm into every conv, so each conv carries weight + bias; the
// BN-less downsample convs have an empty bias name. ONNX weights [OC,IC,KH,KW] ->
// ggml ne = [KW,KH,IC,OC]. Input/output [W=time, H=freq, C, N] (NCHW) or the
// blocked buffer [16, W, H, CB].
static ggml_tensor* next_conv(ggml_context* c, const ModelLoader& ml, const ECfg& e,
                              int& idx, ggml_tensor* x, bool blk) {
    ggml_tensor* w = clone_weight(c, ml, e.conv_weight[idx].c_str());      // [KW,KH,IC,OC]
    const int stride = e.conv_stride[idx];
    const int pad    = (int)(w->ne[0] - 1) / 2;
    ggml_tensor* b = e.conv_bias[idx].empty()
        ? nullptr : clone_weight(c, ml, e.conv_bias[idx].c_str());         // [OC] or none
    ++idx;
    if (blk) {
        // 1x1 vs 3x3 selected by kernel width; bias fused (no clamp here, the
        // caller applies hclamp, which is layout-agnostic).
        if (w->ne[0] == 1)
            return blocked_conv1x1(c, w, x, stride, b, /*relu=*/false);
        return blocked_conv3x3(c, w, x, pad, stride, b, /*relu=*/false);
    }
    ggml_tensor* y = conv2d_auto(c, w, x, stride, stride, pad, pad, 1, 1); // [OW,OH,OC,N]
    if (b) y = ggml_add(c, y, ggml_reshape_4d(c, b, 1, 1, b->ne[0], 1));
    return y;
}

// ERes2Net's in-block "ReLU" is a Hardtanh(0, clamp) (the model subclasses ReLU
// as Hardtanh(0, 20)); the stem uses a plain ReLU instead. ggml_clamp is a pure
// elementwise op, so it is layout-agnostic: it runs identically on NCHW and on
// the blocked nChw16c buffer (zero-padding lanes clamp(0)=0, invariant preserved).
static ggml_tensor* hclamp(ggml_context* c, ggml_tensor* x, float hi) {
    return ggml_clamp(c, x, 0.0f, hi);
}

// Channel slice [c0, c0+width) of an activation. NCHW: a strided view made
// contiguous so the downstream conv's im2col sees packed memory. Blocked: every
// ERes2Net split is 16-channel aligned (width and c0 are multiples of 16), so the
// slice is a CONTIGUOUS span of cb = width/16 blocks at cb0 = c0/16 - a plain
// ne[3] sub-view of the [16,W,H,CB] buffer, no copy (consecutive cb blocks are
// already contiguous with the H*W*16 element stride the blocked conv expects).
static ggml_tensor* chan_slice(ggml_context* c, ggml_tensor* x, int c0, int width, bool blk) {
    if (blk) {
        const int cb0 = c0 / 16, cbw = width / 16;
        return ggml_view_4d(c, x, x->ne[0], x->ne[1], x->ne[2], cbw,
                            x->nb[1], x->nb[2], x->nb[3], (size_t)cb0 * x->nb[3]);
    }
    ggml_tensor* v = ggml_view_4d(c, x, x->ne[0], x->ne[1], width, x->ne[3],
                                  x->nb[1], x->nb[2], x->nb[3],
                                  (size_t)c0 * x->nb[2]);
    return ggml_cont(c, v);
}

// Attentional Feature Fusion (AFF). Reference: x_att = 1 + tanh(local_att(cat(x,
// ds))); xo = x*x_att + ds*(2 - x_att). Reformulated constant-free (algebraically
// identical) as xo = (x + ds) + tanh(z) * (x - ds), where z = local_att(cat(x,ds))
// is conv1x1 -> SiLU -> conv1x1 (both folded conv+bias). The cat is along the
// channel axis (ggml dim 2).
// silu/tanh/add/sub/mul are all elementwise, hence layout-agnostic; only the
// channel concat axis differs (ggml dim 2 in NCHW, dim 3 = CB in the blocked
// buffer). All channel counts here are multiples of 16, so the blocked concat
// stacks whole cb blocks with no padding lanes in the middle.
static ggml_tensor* aff(ggml_context* c, const ModelLoader& ml, const ECfg& e,
                        int& idx, ggml_tensor* x, ggml_tensor* ds, bool blk) {
    const int cat_dim = blk ? 3 : 2;
    ggml_tensor* z = next_conv(c, ml, e, idx, ggml_concat(c, x, ds, cat_dim), blk); // local_att conv1
    z = ggml_silu(c, z);
    z = next_conv(c, ml, e, idx, z, blk);                                // local_att conv2
    ggml_tensor* att  = ggml_tanh(c, z);
    ggml_tensor* sum  = ggml_add(c, x, ds);
    ggml_tensor* diff = ggml_sub(c, x, ds);
    return ggml_add(c, sum, ggml_mul(c, att, diff));
}

// One Res2Net block (expansion 2). conv1 (1x1, block stride) -> clamp; split the
// result into `scale` chunks of `width` channels; for chunk i: i==0 takes the
// chunk as-is, else fuse the running POST-conv chunk with the next split (plain
// blocks ADD, the deeper "diff_AFF" blocks fuse via AFF), then conv (3x3) ->
// clamp; concat the per-chunk outputs; conv3 (1x1); + shortcut (1x1, block
// stride; identity when channels match and stride==1) -> clamp. `in_planes`
// tracks the running channel count across blocks so the shortcut presence is
// computed exactly as the reference _make_layer does.
static ggml_tensor* res2net_block(ggml_context* c, const ModelLoader& ml, const ECfg& e,
                                  int& idx, ggml_tensor* x, int planes, int stride,
                                  bool is_aff, int& in_planes, bool blk) {
    const float hi    = e.relu_clamp;
    const int   scale = (int)e.scale;
    const int   cat_dim = blk ? 3 : 2;
    ggml_tensor* out = hclamp(c, next_conv(c, ml, e, idx, x, blk), hi);  // conv1 + clamp
    const int width = chans(out, blk) / scale;
    ggml_tensor* sp    = nullptr;   // running chunk (POST-conv)
    ggml_tensor* accum = nullptr;
    for (int i = 0; i < scale; ++i) {
        ggml_tensor* chunk = chan_slice(c, out, i * width, width, blk);
        ggml_tensor* cur;
        if (i == 0)      cur = chunk;
        else if (is_aff) cur = aff(c, ml, e, idx, sp, chunk, blk);
        else             cur = ggml_add(c, sp, chunk);
        ggml_tensor* cv = hclamp(c, next_conv(c, ml, e, idx, cur, blk), hi);  // convs[i] + clamp
        sp = cv;
        accum = (i == 0) ? cv : ggml_concat(c, accum, cv, cat_dim);
    }
    out = next_conv(c, ml, e, idx, accum, blk);     // conv3 (1x1), clamp applied after add
    ggml_tensor* residual = x;
    if (stride != 1 || in_planes != 2 * planes)
        residual = next_conv(c, ml, e, idx, x, blk); // shortcut (1x1, stride)
    in_planes = 2 * planes;                          // expansion == 2
    return hclamp(c, ggml_add(c, out, residual), hi);
}

static ggml_tensor* res2net_layer(ggml_context* c, const ModelLoader& ml, const ECfg& e,
                                  int& idx, ggml_tensor* x, int planes, int n,
                                  int stride, bool is_aff, int& in_planes, bool blk) {
    for (int b = 0; b < n; ++b)
        x = res2net_block(c, ml, e, idx, x, planes, (b == 0 ? stride : 1), is_aff, in_planes, blk);
    return x;
}

std::vector<float> ERes2NetEncoder::forward_capture(const std::vector<float>& feats, int T,
                                                    std::vector<float>& pre_norm) const {
    const VoiceDetectConfig& cfg = ml_.config();
    const ECfg& e = cfg.eres2net;
    const int   nmel = (int)cfg.n_mels;
    const float eps  = e.var_eps;            // must outlive compute() (deferred input copy)
    const int   M    = (int)e.m_channels;
    const std::vector<int32_t> nb = e.num_blocks;

    // Route this encoder's 3x3 stride-1 convs (the stem + the Res2Net per-chunk
    // 3x3 convs; its 1x1s already take the mul_mat fast path in conv2d_auto)
    // through the AVX2 Winograd kernel. Scoped opt-in, same as WeSpeaker.
    WinogradScope wino_scope(true);

    // Blocked nChw16c island for the WHOLE Res2Net body: ONE reorder-in (after the
    // NCHW stem) and ONE reorder-out (before TSTP pooling), with every layer + the
    // cross-stage AFF running blocked so the convs never pay the per-conv
    // NCHW<->blocked transpose. The split/concat are 16-channel aligned (contiguous
    // cb spans) and clamp/silu/tanh/add/sub/mul are layout-agnostic, so this is
    // parity-EXACT vs the NCHW body (encoder_out + embedding cosine = 1.000000 on
    // both AVX-512 and AVX2).
    //
    // DEFAULT ON: route the whole Res2Net body through the blocked island. The
    // earlier "blocked regressed ~23% @1t" note only held on the GGML_NATIVE=ON
    // build, where ggml's mul_mat 1x1 fast path gets an AVX-512 tinyBLAS GEMM that
    // out-runs the blocked directconv. That build does NOT ship. On the portable
    // build LocalAI actually ships (GGML_NATIVE=OFF: AVX2-baseline ggml + the
    // custom kernels' own runtime AVX-512 dispatch) ggml's mul_mat is AVX2-only,
    // so the blocked nChw16c AVX-512 directconv island is a strict win at every
    // thread count: measured ~2.07x @1t / ~2.2x @4t / ~2.05x @8t on the shipped
    // build, and still 1.30x @1t / 1.62x @4t/8t on the pure-AVX2 fallback
    // (VOICEDETECT_DISABLE_AVX512=1, no AVX-512 anywhere). Parity is exact in every
    // config: encoder_out + embedding cosine = 1.000000 vs the golden, blocked-on
    // vs blocked-off effectively bit-identical (max|d| ~4e-7). No AVX2 guard is
    // needed; there is no tier where blocked loses. Set VD_ERES2NET_BLOCKED=0
    // (or "off") as the parity-verified A/B opt-out (matching VD_CONV2D /
    // VD_BLOCKED_BACKBONE).
    bool blk = true;
    if (const char* env = std::getenv("VD_ERES2NET_BLOCKED")) {
        if (!std::strcmp(env, "0") || !std::strcmp(env, "off")) blk = false;
        else if (!std::strcmp(env, "1") || !std::strcmp(env, "on")) blk = true;
    }
    std::vector<float> out;
    global_backend().compute([&](ggml_context* c) -> ggml_tensor* {
        // FBank features feat-major [n_mels, T] (feats[m*T + t]) map DIRECTLY to a
        // [W=T, H=n_mels(freq), IC=1, N=1] conv input: element (t, m) = feats[m*T
        // + t]. This is the ERes2Net [B,1,F,T] conv input (after its forward's
        // permute(0,2,1)+unsqueeze(1)), with no extra reshuffle.
        const int64_t ne[4] = { T, nmel, 1, 1 };
        ggml_tensor* x = graph_input_tensor(c, GGML_TYPE_F32, 4, ne,
                                            feats.data(), feats.size() * sizeof(float));

        int idx = 0;
        int in_planes = M;
        // Stem: 3x3 conv (stride 1, pad 1) -> plain ReLU. Kept in NCHW (IC=1);
        // the blocked body begins right after with ONE reorder-in.
        ggml_tensor* h = ggml_relu(c, next_conv(c, ml_, e, idx, x, /*blk=*/false));
        if (blk) h = blocked_reorder_in(c, h);   // NCHW -> [16, W, H, CB]

        // 4 Res2Net layers; the deeper two fuse scales with AFF (block_fuse).
        ggml_tensor* o1 = res2net_layer(c, ml_, e, idx, h,  M,     nb[0], 1, false, in_planes, blk);
        ggml_tensor* o2 = res2net_layer(c, ml_, e, idx, o1, M * 2, nb[1], 2, false, in_planes, blk);

        // Bottom-up cross-stage fusion: downsample the shallower feature and
        // AFF-merge it into the next layer's output, matching the reference
        // forward's interleaving (so the conv index stays in execution order).
        ggml_tensor* o1d  = next_conv(c, ml_, e, idx, o1, blk);   // layer1_downsample
        ggml_tensor* f12  = aff(c, ml_, e, idx, o2, o1d, blk);
        ggml_tensor* o3   = res2net_layer(c, ml_, e, idx, o2, M * 4, nb[2], 2, true, in_planes, blk);
        ggml_tensor* f12d = next_conv(c, ml_, e, idx, f12, blk);  // layer2_downsample
        ggml_tensor* f123 = aff(c, ml_, e, idx, o3, f12d, blk);
        ggml_tensor* o4   = res2net_layer(c, ml_, e, idx, o3, M * 8, nb[3], 2, true, in_planes, blk);
        ggml_tensor* f123d = next_conv(c, ml_, e, idx, f123, blk); // layer3_downsample
        ggml_tensor* f1234 = aff(c, ml_, e, idx, o4, f123d, blk);  // [W',H'=10,C=16M,1]
        // Reorder back to NCHW for TSTP pooling (mean/var over the time axis ne[0],
        // which is the 16-lane axis in the blocked buffer).
        if (blk) f1234 = blocked_reorder_out(c, f1234, chans(f1234, true));

        // TSTP (Temporal Statistics Pooling): per-(channel, freq) mean + unbiased
        // std over the time axis (ne[0]), preserving the H'=feat_dim/8 frequency
        // resolution. std = sqrt(unbiased_var + eps).
        const int N = (int)f1234->ne[0];
        ggml_tensor* mean = ggml_mean(c, f1234);                  // [1, H', C, 1]
        ggml_tensor* dev  = ggml_sub(c, f1234, mean);             // mean broadcast
        ggml_tensor* var  = ggml_mean(c, ggml_sqr(c, dev));       // biased var [1,H',C,1]
        if (N > 1) var = ggml_scale(c, var, (float)N / (float)(N - 1));
        const int64_t one[1] = { 1 };
        ggml_tensor* eps_t = graph_input_tensor(c, GGML_TYPE_F32, 1, one, &eps, sizeof(float));
        ggml_tensor* std_t = ggml_sqrt(c, ggml_add(c, var, eps_t));

        // Flatten each stat to [C*H'] in C-outer / H'(freq)-inner order (the
        // torch flatten(start_dim=1) over [B, C, H']), then concat [mean ; std].
        const int64_t HC = mean->ne[1] * mean->ne[2];
        ggml_tensor* mean_f = ggml_reshape_1d(c, ggml_cont(c, mean), HC);
        ggml_tensor* std_f  = ggml_reshape_1d(c, ggml_cont(c, std_t), HC);
        ggml_tensor* pooled = ggml_concat(c, mean_f, std_f, /*dim=*/0);  // [2*C*H']

        // Embedding FC: seg_1 Gemm (transB) + bias. seg_1 ONNX [emb_dim, 2*C*H']
        // -> ggml ne = [2*C*H', emb_dim], so mul_mat is the transposed-B product.
        ggml_tensor* W = clone_weight(c, ml_, e.seg_weight.c_str());     // [2*C*H', emb_dim]
        ggml_tensor* emb = ggml_mul_mat(c, W, pooled);                  // [emb_dim, 1]
        ggml_tensor* sb = clone_weight(c, ml_, e.seg_bias.c_str());     // [emb_dim]
        emb = ggml_add(c, emb, sb);
        return ggml_reshape_1d(c, emb, emb->ne[0]);                     // [emb_dim], pre-norm
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

std::vector<float> ERes2NetEncoder::forward(const std::vector<float>& feats, int T) const {
    std::vector<float> pre_norm;
    return forward_capture(feats, T, pre_norm);
}

} // namespace vd
