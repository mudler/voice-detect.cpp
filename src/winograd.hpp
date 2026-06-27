#pragma once
#include "ggml.h"

// Winograd convolution for the 2D speaker encoders' large 3x3 stride-1 pad-1
// convs (WeSpeaker ResNet34's all-3x3 backbone over [T, n_mels] maps, plus
// ERes2Net's remaining 3x3 stride-1 convs).
//
// Implemented as a CPU custom op (ggml_custom_4d) with an AVX2 winograd-domain
// multiply (the production build sets GGML_NATIVE=OFF, so AVX-512 is unavailable
// and undesirable here; this file is arch-flagged -mavx2 -mfma in CMake, with a
// scalar fallback for non-AVX2 hosts). Ported verbatim (algorithm + inner
// kernel) from face-detect.cpp's parity-tested src/winograd.cpp, where it beat
// im2col+tinyBLAS on SCRFD's large 3x3 maps. The algorithm/inner kernel is
// selectable via the VOICEDETECT_WINO env var:
//   "f2"  : F(2x2,3x3), per-tile GEMV
//   "f2b" : F(2x2,3x3), blocked GEMM over a block of tiles  <-- default
//           (parity-identical to f2, reuses each U-row across the block)
//   "f4"  : F(4x4,3x3), 4x fewer mults vs direct, blocked GEMM. Less accurate
//           (1/6,1/24 fractions).
//
// Tensor layout (ggml ne, fastest dim first):
//   x : [W, H, IC, N]    input feature map (F32)
//   w : [3, 3, IC, OC]   filter (torch (OC,IC,KH,KW) reversed)  (F32)
//   out: [Wout, Hout, OC, N]  with Wout = W + 2*pad - 2, Hout = H + 2*pad - 2
//
// Only valid for KW==KH==3, stride==1, F32 inputs. `pad` is arbitrary (the
// speaker-encoder 3x3 convs always use pad=1 -> same-size output). Bias is NOT
// applied here; the caller adds it after via ggml_add (matching the
// direct/im2col conv path).
namespace vd {

ggml_tensor* winograd_conv3x3(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x, int pad);

} // namespace vd
