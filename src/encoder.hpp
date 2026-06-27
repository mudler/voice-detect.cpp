#pragma once
#include <vector>

namespace vd {

class ModelLoader;

// Intermediate activations captured during an ECAPA-TDNN encoder forward pass,
// for golden-parity gating (one capture per reference tensor dumped by
// scripts/gen_baseline.py). Each vector is channel-major / time-inner, i.e. the
// element for channel c at frame t lives at index c*T + t - the SAME memory
// layout ggml gives a [T, C] (ne[0]=T, ne[1]=C) tensor on readback, and the same
// layout the baseline's [C, T] numpy tensors land in once gguf reverses the
// shape. So caps.* compares element-for-element against the baseline with no
// transpose. Fields beyond block0 fill in as later encoder stages land.
struct EcapaCaptures {
    std::vector<float> block0, block1, block2, block3, mfa, pooled, pre_norm;
};

// ECAPA-TDNN speaker encoder graph (1024-channel SpeechBrain variant). Reads its
// weights verbatim from the ModelLoader (voicedetect.ecapa.* + the verbatim
// state_dict tensors) and runs on the process-global vd::Backend.
class EcapaEncoder {
public:
    explicit EcapaEncoder(const ModelLoader& ml);

    // feats: feat-major [n_mels, T] (feats[m*T + t]). Returns the L2-normalized
    // [embedding_dim] speaker embedding.
    std::vector<float> forward(const std::vector<float>& feats, int T) const;

    // Same as forward(), additionally filling `caps` with the per-stage
    // activations for parity gating.
    std::vector<float> forward_capture(const std::vector<float>& feats, int T,
                                       EcapaCaptures& caps) const;

private:
    // Shared graph builder for both forward() and forward_capture(). When
    // `capture` is true the 7 intermediate tensors (block0-3, mfa, pooled,
    // pre_norm) are marked as additional graph outputs and read back into
    // `caps` for golden-parity gating. The production forward() / C-ABI path
    // passes capture=false: those readbacks are harness-only and cost ~2.7ms
    // plus they block ggml op fusion, so they are skipped (the only required
    // output is the final embedding, which is the graph's return value).
    std::vector<float> run_forward(const std::vector<float>& feats, int T,
                                   EcapaCaptures& caps, bool capture) const;

    const ModelLoader& ml_;
};

// WeSpeaker ResNet34 speaker encoder (the ONNX-direct alt arch). A 2D-conv
// ResNet over the [n_mels, T] FBank features: a 3x3 conv stem, 4 stages of
// BasicBlocks ([3,4,6,3], channels [32,64,128,256], stride-2 + 1x1 downsample at
// each stage boundary), temporal statistics pooling (per-frequency mean + std
// over the time axis), and an embedding FC. The block topology is read from the
// loader's `voicedetect.resnet.*` manifest (the folded ONNX export keeps opaque
// integer tensor names); channel widths come from each weight's `ne`.
class ResNet34Encoder {
public:
    explicit ResNet34Encoder(const ModelLoader& ml);

    // feats: feat-major [n_mels, T] (feats[m*T + t]). Returns the L2-normalized
    // [embedding_dim] speaker embedding (256-d for WeSpeaker).
    std::vector<float> forward(const std::vector<float>& feats, int T) const;

    // Same as forward(), additionally filling `pre_norm` with the RAW (pre
    // L2-norm) embedding for the `encoder_out` parity gate.
    std::vector<float> forward_capture(const std::vector<float>& feats, int T,
                                       std::vector<float>& pre_norm) const;

private:
    const ModelLoader& ml_;
};

// 3D-Speaker ERes2Net speaker encoder (ONNX-direct alt arch). A 2D-conv ResNet
// over the [n_mels, T] FBank: a 3x3 conv stem (plain ReLU), then 4 layers of
// Res2Net blocks (scale-2 multi-scale hierarchical 3x3 convs; the deeper two
// layers fuse scales with attentional feature fusion - AFF), a bottom-up
// cross-stage fusion that AFF-merges each layer's output with a downsampled
// shallower feature, temporal statistics pooling (TSTP: per-(channel,freq) mean +
// unbiased std over time), and a seg_1 embedding FC. Block "ReLU" is Hardtanh(0,
// 20) (a clamp). The conv manifest (verbatim folded-conv names + per-conv stride,
// in forward-execution order) is read from `voicedetect.eres2net.*`; channel
// widths come from each weight's `ne`, padding from its kernel size.
class ERes2NetEncoder {
public:
    explicit ERes2NetEncoder(const ModelLoader& ml);

    // feats: feat-major [n_mels, T] (feats[m*T + t]). Returns the L2-normalized
    // [embedding_dim] speaker embedding (512-d for ERes2Net base).
    std::vector<float> forward(const std::vector<float>& feats, int T) const;

    // Same as forward(), additionally filling `pre_norm` with the RAW (pre
    // L2-norm) embedding for the `encoder_out` parity gate.
    std::vector<float> forward_capture(const std::vector<float>& feats, int T,
                                       std::vector<float>& pre_norm) const;

private:
    const ModelLoader& ml_;
};

// 3D-Speaker CAM++ speaker encoder (ONNX-direct alt arch). A 2D-conv FCM front end
// (3x3 stem + two BasicResBlock stages striding the frequency axis, reshaped to a
// [320, T] feature) feeds a 1D D-TDNN backbone: a strided TDNN, three densely
// connected CAMDenseTDNN blocks (num_layers [12, 24, 16], dilation [1, 2, 2],
// growth 32) each followed by a channel-halving transit, an out ReLU, statistics
// pooling (per-channel mean + unbiased std over time), a 1024->emb dense
// projection, and a final affine-free BatchNorm. Each dense layer applies a
// context-aware mask (CAM): out = linear_local(c) * sigmoid(linear2(relu(linear1(
// c.mean(-1) + seg100(c))))), where seg100 is a 100-frame count_include_pad
// segment average. The partly-BN-folded ONNX export keeps semantic names for the
// unfolded tensors and opaque ids for the folded convs, so the topology is read
// from the loader's `voicedetect.campplus.*` flat conv/BN manifests (execution
// order); channel widths come from each tensor's `ne`. Emits a 192-d embedding.
class CamPPlusEncoder {
public:
    explicit CamPPlusEncoder(const ModelLoader& ml);

    // feats: feat-major [n_mels, T] (feats[m*T + t]). Returns the L2-normalized
    // [embedding_dim] speaker embedding (192-d for CAM++).
    std::vector<float> forward(const std::vector<float>& feats, int T) const;

    // Same as forward(), additionally filling `pre_norm` with the RAW (pre
    // L2-norm) embedding for the `encoder_out` parity gate.
    std::vector<float> forward_capture(const std::vector<float>& feats, int T,
                                       std::vector<float>& pre_norm) const;

private:
    const ModelLoader& ml_;
};

} // namespace vd
