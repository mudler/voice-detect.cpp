#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
struct ggml_tensor;
struct ggml_context;
struct gguf_context;
struct ggml_backend_buffer;
typedef struct ggml_backend_buffer* ggml_backend_buffer_t;
struct ggml_backend;
typedef struct ggml_backend* ggml_backend_t;
namespace vd {

// Metadata-driven config for a speaker-recognition GGUF. All values live in GGUF
// KV under the `voicedetect.*` prefix; tensor names are kept VERBATIM from the
// source state_dict so the C++ engine is a 1:1 mapping (see docs/conversion.md).
struct VoiceDetectConfig {
    // Encoder family. One of:
    //   "ecapa_tdnn"          SpeechBrain ECAPA-TDNN (192-d, primary)
    //   "wespeaker_resnet34"  WeSpeaker ResNet34 (ONNX-direct)
    //   "eres2net"            3D-Speaker ERes2Net (ONNX-direct)
    //   "campplus"            3D-Speaker CAM++ (ONNX-direct)
    std::string arch;
    // Embedding head
    uint32_t embedding_dim = 0;     // 192 for ECAPA-TDNN; 256 for WeSpeaker/CAM++
    bool     l2_normalize  = true;  // L2-normalize the output embedding (cosine space)

    // FBank front end (Kaldi-compatible; shared by every encoder).
    uint32_t sample_rate = 16000;
    uint32_t n_mels      = 80;      // filterbank feature dim
    uint32_t n_fft       = 512;
    uint32_t win_length  = 400;     // 25 ms @ 16 kHz
    uint32_t hop_length  = 160;     // 10 ms @ 16 kHz
    float    preemph     = 0.97f;   // pre-emphasis coefficient
    float    fbank_low_freq  = 20.0f;
    float    fbank_high_freq = 0.0f;  // 0 = Nyquist
    bool     fbank_use_energy = false;
    bool     fbank_cmn        = true; // per-utterance cepstral mean normalization
    std::string fbank_window  = "povey";

    // WeSpeaker ResNet34 block manifest (arch == "wespeaker_resnet34"). The ONNX
    // export folds BatchNorm into the convs and names initializers with opaque
    // integers, so the block topology cannot be inferred from tensor names: the
    // converter records it here as parallel arrays (one entry per BasicBlock, in
    // forward order). Channel widths/stage depths are NOT stored; the C++ encoder
    // reads them from each tensor's `ne`. Empty in the non-ResNet archs.
    struct ResNetConfig {
        std::string stem_weight, stem_bias;     // 3x3 stem conv (+ folded BN bias)
        std::vector<std::string> conv1_weight, conv1_bias;  // per block: conv1
        std::vector<std::string> conv2_weight, conv2_bias;  // per block: conv2
        std::vector<std::string> down_weight, down_bias;    // 1x1 downsample or ""
        std::vector<int32_t>     stride;        // per block: spatial stride (1 or 2)
        std::string seg_weight, seg_bias;       // embedding FC (Gemm, 5120->emb_dim)
        std::string mean_vec;                   // post-FC mean centering ("" if none)
        float       var_eps = 1e-8f;            // std = sqrt(unbiased_var + var_eps)
        size_t n_blocks() const { return conv1_weight.size(); }
    };
    ResNetConfig resnet;

    // 3D-Speaker ERes2Net manifest (arch == "eres2net"). The folded ONNX export
    // names every BN-folded conv with an opaque integer, so the topology is
    // recovered as a flat conv list in forward-EXECUTION order (verbatim weight +
    // bias names, empty bias for the BN-less downsample convs, per-conv stride)
    // plus the fixed block topology. The C++ ERes2NetEncoder replays the
    // deterministic ERes2Net forward, pulling each conv by a running index;
    // channel widths come from each tensor's ne, padding from its kernel size.
    struct ERes2NetConfig {
        std::vector<std::string> conv_weight;   // 96 names, execution order
        std::vector<std::string> conv_bias;     // parallel; "" for downsample convs
        std::vector<int32_t>     conv_stride;    // parallel; spatial stride (1 or 2)
        std::vector<int32_t>     num_blocks;     // blocks per layer, e.g. [3,4,6,3]
        uint32_t m_channels = 0;                 // stem conv out channels (32 base)
        uint32_t scale       = 0;                // Res2Net multi-scale split (2)
        std::string seg_weight, seg_bias;        // embedding FC (seg_1 Gemm)
        float       relu_clamp = 20.0f;          // block ReLU = Hardtanh(0, clamp)
        float       var_eps    = 1e-8f;          // TSTP std = sqrt(unbiased_var + eps)
        size_t n_convs() const { return conv_weight.size(); }
    };
    ERes2NetConfig eres2net;

    // 3D-Speaker CAM++ manifest (arch == "campplus"). The partly-BN-folded ONNX
    // export names the folded convs with opaque ``onnx::Conv_*`` ids and keeps
    // semantic names for the rest, but emits every Conv and BatchNormalization in
    // forward-EXECUTION order, so the topology is recovered as two flat parallel
    // lists. The C++ CamPPlusEncoder replays the deterministic CAM++ forward (FCM
    // 2D front end + strided TDNN + 3 CAMDenseTDNN blocks [12,24,16] with
    // dilation [1,2,2] growth 32 + halving transits + stats pooling + 1024->emb
    // dense + affine-free BN), pulling each conv/BN by a running index; channel
    // widths come from each tensor's ne. Empty in the non-CAM++ archs.
    struct CamPPlusConfig {
        std::vector<std::string> conv_weight;   // 225 names, execution order
        std::vector<std::string> conv_bias;     // parallel; "" for the bias-free convs
        std::vector<std::string> bn_prefix;     // 55 affine BN prefixes, execution order
        std::string emb_bn_mean, emb_bn_var;    // final affine-free embedding BN stats
        float       bn_eps = 1e-5f;             // BatchNorm epsilon (PyTorch default)
        size_t n_convs() const { return conv_weight.size(); }
    };
    CamPPlusConfig campplus;

    // wav2vec2 analyze head config (arch == "wav2vec2_emotion" or
    // "wav2vec2_age_gender"). The analyze GGUF is a SEPARATE model from the speaker
    // encoders: a raw-16kHz-waveform wav2vec2 (7-layer strided Conv1d feature
    // encoder + transformer stack + a task head), NOT the mel-FBank speaker path.
    // Every dim/kernel/stride lives in KV under `voicedetect.w2v2.*` so the C++
    // analyze graph is fully metadata-driven. The emotion model (base) uses
    // feat_extract_norm "group" + post-norm layers; the age/gender model
    // (audeering wav2vec2-large-robust) uses feat_extract_norm "layer" + conv bias
    // + do_stable_layer_norm (pre-norm layers + a final encoder LayerNorm), all
    // selected from these KV. Weights are verbatim HF `wav2vec2.*` (-> `w2v2.`) plus
    // the head names (`projector.*/classifier.*` emotion, `age.*/gender.*` audeering).
    struct W2V2Config {
        uint32_t hidden_size = 0;        // transformer width (768 for base)
        uint32_t n_layers    = 0;        // transformer encoder layers (12)
        uint32_t n_heads     = 0;        // attention heads (12)
        uint32_t ff_dim      = 0;        // feed-forward intermediate (3072)
        uint32_t num_conv_layers = 0;    // feature-encoder Conv1d layers (7)
        std::vector<int32_t> conv_dims;      // out channels per conv (all 512)
        std::vector<int32_t> conv_kernels;   // [10,3,3,3,3,2,2]
        std::vector<int32_t> conv_strides;   // [5,2,2,2,2,2,2]
        std::string feat_extract_norm;       // "group" (only conv0 GroupNorm) | "layer"
                                             // (every conv layer LayerNorm'd, wav2vec2-large-robust)
        bool  conv_bias = false;             // feature-encoder Conv1d bias (True for large-robust)
        bool  do_normalize = false;          // FE zero-mean/unit-var the waveform (True for large-robust)
        std::string feat_extract_activation; // "gelu" (exact erf GELU)
        std::string hidden_act;              // transformer activation ("gelu")
        bool  do_stable_layer_norm = false;
        float layer_norm_eps = 1e-5f;
        uint32_t num_conv_pos_embeddings = 0;       // positional conv kernel (128)
        uint32_t num_conv_pos_embedding_groups = 0; // grouped positional conv (16)
        uint32_t pos_conv_weight_norm_dim = 2;      // weight_norm dim of the pos conv
        bool use_weighted_layer_sum = false;
        uint32_t classifier_proj_size = 0;
        size_t n_conv() const { return conv_kernels.size(); }
    };
    W2V2Config w2v2;

    // Analyze heads (age/gender/emotion). Phased last; present=false until the
    // wav2vec2 heads are wired, in which case the engine skips analyze entirely.
    bool analyze_present = false;
    std::vector<std::string> emotion_labels;  // e.g. ["neutral","happy",...]
    std::vector<std::string> gender_labels;   // e.g. ["female","male"]
};

class ModelLoader {
public:
    ModelLoader() = default;
    ~ModelLoader();
    // Load a GGUF. Returns false if the file is absent/unreadable or required
    // KV is missing - tolerant of a not-yet-existing file (no throw, no abort).
    bool load(const std::string& path);
    const VoiceDetectConfig& config() const { return cfg_; }
    ggml_tensor* tensor(const std::string& name) const; // nullptr if absent
    ggml_context* ggml_ctx() const { return ctx_; }

    // Give every weight tensor a backend buffer (ONCE), so graphs can reference
    // the loader's tensors DIRECTLY as leaves with zero per-call copying. CPU
    // path: wraps the contiguous ctx mem_buffer (zero-copy). Device path: mirrors
    // every weight into a no_alloc ctx, allocates on the backend, uploads, and
    // repoints the name->tensor map at the device tensors. Idempotent.
    bool realize_weights(ggml_backend_t backend);
    bool weights_realized() const { return weights_buf_ != nullptr; }
private:
    VoiceDetectConfig cfg_;
    gguf_context* gguf_ = nullptr;
    ggml_context* ctx_ = nullptr;
    ggml_backend_buffer_t weights_buf_ = nullptr;
    ggml_context* device_ctx_ = nullptr;  // no_alloc mirror ctx for device weights
    std::unordered_map<std::string, ggml_tensor*> tensors_;
};
} // namespace vd
