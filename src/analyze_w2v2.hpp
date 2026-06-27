#pragma once
#include <string>
#include <vector>

namespace vd {

class ModelLoader;

// wav2vec2 analyze engine (the SEPARATE emotion/analyze model, NOT a speaker
// encoder). wav2vec2 operates on the RAW 16 kHz waveform - there is no mel FBank
// front end. This class builds the wav2vec2-base CONV FEATURE ENCODER as a ggml
// graph: 7 strided Conv1d layers (kernels [10,3,3,3,3,2,2], strides
// [5,2,2,2,2,2,2], 512 channels, no bias). With feat_extract_norm == "group" ONLY
// conv layer 0 is followed by a GroupNorm (num_groups == num_channels, i.e.
// per-channel normalization over time); every conv is followed by the exact
// (erf) GELU. The kernels/strides/dims/norm/activation all come from the loader's
// `voicedetect.w2v2.*` KV (no magic numbers). Later tasks add the transformer
// stack (T19) and the emotion head + analyze JSON (T20).
class W2V2Analyzer {
public:
    explicit W2V2Analyzer(const ModelLoader& ml);

    // pcm16k: raw 16 kHz mono waveform in [-1, 1]. Runs the conv feature encoder
    // and returns the `feat_extract_out` features [hidden=conv_dim, T'] in
    // channel-major / time-inner layout (element for channel c at frame t lives at
    // index c*T' + t - the SAME flat layout the golden `feat_extract_out` numpy
    // tensor lands in). `out_T` receives T'.
    std::vector<float> feature_encoder(const std::vector<float>& pcm16k, int& out_T) const;

    // Run the FULL wav2vec2 encoder: conv feature encoder -> feature projection
    // (LayerNorm 512 + Linear 512->hidden) -> positional conv embedding
    // (grouped weight-normed Conv1d + GELU, added to the projected features) ->
    // encoder LayerNorm -> the `n_layers` post-norm transformer layers. Returns the
    // final hidden states `[hidden, T']` (hidden-major / time-inner: element for
    // dim h at frame t at index h*T' + t, matching the golden `enc_layer_*`). If
    // `dbg_layer0` / `dbg_layer_mid` are non-null they receive the per-layer hidden
    // states AFTER encoder layer 0 and after the middle layer (n_layers/2 - 1),
    // i.e. the golden `enc_layer_0` / `enc_layer_mid`. `out_T` receives T'.
    std::vector<float> encode(const std::vector<float>& pcm16k, int& out_T,
                              std::vector<float>* dbg_layer0 = nullptr,
                              std::vector<float>* dbg_layer_mid = nullptr) const;

    // Run the full wav2vec2 encoder + the emotion classification head
    // (superb/wav2vec2-base-superb-er, use_weighted_layer_sum=True): collect all
    // `n_layers + 1` hidden states (the encoder-layernorm output + every layer
    // output), take the weighted sum with softmax(layer_weights) over the stack,
    // run the `projector` Linear (hidden->classifier_proj_size), mean-pool over
    // time, then the `classifier` Linear (proj->num_emotions). Returns the raw
    // pre-softmax `emotion_logits` [num_emotions], matching the golden tensor.
    std::vector<float> emotion_logits(const std::vector<float>& pcm16k) const;

    // Run the full wav2vec2 encoder + the audeering age/gender head
    // (audeering/wav2vec2-large-robust-24-ft-age-gender, do_stable_layer_norm): take
    // the final hidden state (last encoder LayerNorm output), mean-pool over time,
    // then the two custom ModelHeads -> out_proj(tanh(dense(pooled))). Returns the
    // RAW head outputs concatenated as [age_raw, gender_logit_0, ...]: the age is a
    // single regression scalar (multiply by 100 for years) and the rest are the
    // num_genders gender logits (softmax for probabilities). Matches the golden
    // tensors age_raw + gender_logits.
    std::vector<float> age_gender_raw(const std::vector<float>& pcm16k) const;

    // Full analyze entry point: run emotion_logits(), softmax -> probabilities,
    // argmax -> dominant label, and serialize the C-ABI analyze JSON
    //   {"emotion":{"label":..,"scores":{<label>:<prob>,...}}}
    // (age/gender are added only when the GGUF carries those heads; the emotion
    // model does not). Mirrors the shape documented in voicedetect_capi.h.
    std::string analyze_json(const std::vector<float>& pcm16k) const;

    // Full age/gender analyze entry point (the audeering model): run
    // age_gender_raw(), scale the age to years (*100), softmax the gender logits,
    // argmax -> dominant gender label, and serialize the C-ABI analyze JSON
    //   {"age":<years>,"gender":{"label":..,"<label>":<prob>,...}}
    // Mirrors the shape documented in voicedetect_capi.h.
    std::string analyze_age_gender_json(const std::vector<float>& pcm16k) const;

private:
    const ModelLoader& ml_;

    // The weight-normed positional-conv kernel (w = g * v / ||v||) depends ONLY on
    // static model weights, yet it was rebuilt on the host on EVERY forward (a
    // scalar triple loop over K*IC*OC: ~20 ms emotion, ~112 ms age/gender per
    // call). Like depth-anything's pos-embed caching, it is built once and reused.
    // The analyzer is constructed fresh per analyze call (see Model::*), so the
    // cache cannot live on the instance; it is process-global keyed by the
    // ModelLoader (see analyze_w2v2.cpp), surviving across calls on the same model.
    const std::vector<float>& pos_conv_weight() const;
};

} // namespace vd
