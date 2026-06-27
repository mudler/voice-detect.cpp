#pragma once
#include "ggml.h"

// Blocked-layout (nChw16c) AVX-512 register-tiled DIRECT 3x3 stride-1 convolution
// for the 2D speaker encoders, an A/B alternative to winograd_conv3x3. No im2col,
// no Winograd transform: streams the NCHW activation broadcast over a 16-OC weight
// vector (zmm), register-blocked over output-width, FMA over the IC*KH*KW window.
// Weights pre-packed [OC/16][KH][KW][IC][16] and cached at first use. Runtime
// CPUID dispatch (AVX-512 16c / AVX2 8c / scalar), no global -mavx512f.
//
// Layout (ggml ne, fastest dim first), identical contract to winograd_conv3x3:
//   x  : [W, H, IC, N]
//   w  : [3, 3, IC, OC]
//   out: [Wout, Hout, OC, N],  Wout = W + 2*pad - 2, Hout = H + 2*pad - 2
// Only KW==KH==3, stride==1, pad>=1, F32, N==1. Bias/ReLU applied by the caller
// (matching winograd / im2col); the f32 accumulation order differs from im2col so
// values are within fp32 tolerance, not bitwise identical.
namespace vd {

ggml_tensor* directconv_conv3x3(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x, int pad);

// ---- blocked-layout (nChw16c) island ops -----------------------------------
// Keep the whole ResNet backbone in the blocked buffer between ONE reorder-in
// (after the stem) and ONE reorder-out (before pooling), amortizing the per-conv
// NCHW<->blocked reorder tax over the entire backbone. The blocked buffer has ggml
// ne = [16, W, H, CB] (CB = ceil(C/16)); flat index ((cb*H+h)*W+w)*16 + l, channel
// c = cb*16 + l. Padding lanes (c >= C) are zero. The conv kh,kw,ic accumulation
// order matches directconv_conv3x3, so blocked vs NCHW direct conv is bit-identical
// (layout-only change); reorder is an exact permute.
//
//   x  (NCHW)   : [W, H, C, 1]
//   xb (blocked): [16, W, H, CB]
//   w 3x3       : [3, 3, IC, OC]   (ggml weight layout)
//   w 1x1       : [1, 1, IC, OC]
// True iff the blocked island's AVX-512 fast path will run at runtime. On
// non-AVX512 hosts the blocked ops fall back to scalar (slower than the per-conv
// directconv's AVX2 path), so callers default the island OFF unless this is true.
bool directconv_blocked_available();

ggml_tensor* blocked_reorder_in (ggml_context* ctx, ggml_tensor* x_nchw);          // -> [16,W,H,CB]
ggml_tensor* blocked_reorder_out(ggml_context* ctx, ggml_tensor* xb, int C);       // -> [W,H,C,1]
// Conv ops optionally FUSE per-channel bias add (+ ReLU) into the register tile,
// removing the separate blocked-bias/blocked-relu passes. bias==nullptr -> raw conv.
ggml_tensor* blocked_conv3x3    (ggml_context* ctx, ggml_tensor* w, ggml_tensor* xb, int pad, int stride,
                                 ggml_tensor* bias = nullptr, bool do_relu = false);
ggml_tensor* blocked_conv1x1    (ggml_context* ctx, ggml_tensor* w, ggml_tensor* xb, int stride,
                                 ggml_tensor* bias = nullptr, bool do_relu = false);
ggml_tensor* blocked_bias       (ggml_context* ctx, ggml_tensor* xb, ggml_tensor* bias);
ggml_tensor* blocked_relu       (ggml_context* ctx, ggml_tensor* xb);
ggml_tensor* blocked_add        (ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);
ggml_tensor* blocked_add_relu   (ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);  // max(0, a+b)

} // namespace vd
