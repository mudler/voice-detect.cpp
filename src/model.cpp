#include "model.hpp"
#include "audio_io.hpp"
#include "fbank.hpp"
#include "encoder.hpp"
#include "analyze_w2v2.hpp"
#include "backend.hpp"
#include "common.hpp"

#include <cmath>
#include <stdexcept>

namespace vd {

std::unique_ptr<Model> Model::load(const std::string& gguf_path) {
    auto m = std::unique_ptr<Model>(new Model());
    if (!m->loader_.load(gguf_path)) {
        return nullptr;  // bad/missing GGUF, or required KV absent
    }
    return m;
}

std::vector<float> Model::embed_16k(const std::vector<float>& pcm16k) const {
    // Stage 1: Kaldi-compatible FBank front end -> feat-major [n_mels, T].
    FBank fbank(loader_.config());
    int T = 0;
    std::vector<float> feats = fbank.compute(pcm16k, T);
    if (feats.empty() || T == 0) {
        throw std::runtime_error("voicedetect: FBank produced no frames");
    }

    // Stages 2-5 (speaker encoder forward + pooling + projection + L2-norm) are
    // owned by the arch-specific encoder. ECAPA-TDNN is the primary anchor; the
    // ONNX-direct encoders (wespeaker_resnet34 / eres2net / campplus) land later
    // and dispatch here on cfg.arch.
    const std::string& arch = loader_.config().arch;
    if (arch == "ecapa_tdnn") {
        return EcapaEncoder(loader_).forward(feats, T);
    }
    if (arch == "wespeaker_resnet34") {
        return ResNet34Encoder(loader_).forward(feats, T);
    }
    if (arch == "eres2net") {
        return ERes2NetEncoder(loader_).forward(feats, T);
    }
    if (arch == "campplus") {
        return CamPPlusEncoder(loader_).forward(feats, T);
    }
    throw std::runtime_error("voicedetect: unsupported arch '" + arch + "'");
}

std::vector<float> Model::embed_pcm(const std::vector<float>& pcm, int sample_rate) const {
    const std::vector<float> pcm16k =
        sample_rate == 16000 ? pcm : resample_linear(pcm, sample_rate, 16000);
    return embed_16k(pcm16k);
}

std::vector<float> Model::embed_path(const std::string& wav_path) const {
    Audio audio;
    if (!load_audio_16k_mono(wav_path, audio)) {
        throw std::runtime_error("voicedetect: failed to load audio: " + wav_path);
    }
    return embed_16k(audio.samples);
}

void Model::verify_paths(const std::string& a, const std::string& b, float threshold,
                         float& out_distance, bool& out_verified) const {
    const std::vector<float> ea = embed_path(a);
    const std::vector<float> eb = embed_path(b);
    if (ea.size() != eb.size() || ea.empty()) {
        throw std::runtime_error("voicedetect: embedding dimension mismatch in verify");
    }
    // Embeddings are L2-normalized, so cosine similarity is the dot product.
    double dot = 0.0;
    for (size_t i = 0; i < ea.size(); ++i) dot += (double)ea[i] * (double)eb[i];
    const float similarity = (float)dot;
    out_distance = 1.0f - similarity;          // cosine distance in [0, 2]
    out_verified = out_distance <= threshold;  // smaller distance => same speaker
}

std::string Model::analyze_path_json(const std::string& wav_path) const {
    if (!loader_.config().analyze_present) {
        throw std::runtime_error(
            "voicedetect: this model has no age/gender/emotion analyze heads");
    }
    const std::string& arch = loader_.config().arch;
    if (arch != "wav2vec2_emotion" && arch != "wav2vec2_age_gender") {
        throw std::runtime_error(
            "voicedetect: analyze head unsupported for arch '" + arch + "'");
    }
    Audio audio;
    if (!load_audio_16k_mono(wav_path, audio)) {
        throw std::runtime_error("voicedetect: failed to load audio: " + wav_path);
    }
    // Raw 16 kHz mono waveform -> wav2vec2 encoder + the arch's analyze head ->
    // analyze JSON. The emotion (base) and age/gender (large-robust) models are
    // SEPARATE analyze GGUFs, mirroring how the Python backend loads them apart.
    if (arch == "wav2vec2_age_gender") {
        return W2V2Analyzer(loader_).analyze_age_gender_json(audio.samples);
    }
    return W2V2Analyzer(loader_).analyze_json(audio.samples);
}

} // namespace vd
