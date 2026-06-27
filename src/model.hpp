#pragma once
#include "model_loader.hpp"

#include <memory>
#include <string>
#include <vector>

namespace vd {

// Load-once speaker-recognition context.
//
// Loads a GGUF model ONCE (owns the ModelLoader) and reuses it across many
// embed/verify/analyze calls. The component objects (FBank front end, encoder,
// pooling, analyze heads) are lightweight views over the ModelLoader, so they
// are constructed per call; the expensive part - parsing the GGUF and mapping
// every weight tensor - happens exactly once, in load(). This is what the flat
// C-API holds.
class Model {
public:
    // Loads the GGUF at `gguf_path`. Returns nullptr on failure (no throw).
    static std::unique_ptr<Model> load(const std::string& gguf_path);

    // Compute the L2-normalized speaker embedding from raw mono float PCM. If
    // `sample_rate != 16000` the audio is linearly resampled to 16 kHz first.
    // Throws std::runtime_error on failure (unimplemented stage, bad model, ...).
    std::vector<float> embed_pcm(const std::vector<float>& pcm, int sample_rate) const;

    // Embed a WAV file (loaded + resampled to 16 kHz mono). Throws on failure.
    std::vector<float> embed_path(const std::string& wav_path) const;

    // Cosine distance (1 - cosine_similarity) between two clips' L2-normalized
    // embeddings, and the verdict vs `threshold` (distance <= threshold => same
    // speaker). Throws std::runtime_error on failure.
    void verify_paths(const std::string& a, const std::string& b, float threshold,
                      float& out_distance, bool& out_verified) const;

    // Analyze age/gender/emotion -> JSON document (see voicedetect_capi.h).
    // Throws std::runtime_error if the model has no analyze heads or the stage
    // is not yet implemented.
    std::string analyze_path_json(const std::string& wav_path) const;

    int embedding_dim() const { return (int)loader_.config().embedding_dim; }
    const VoiceDetectConfig& config() const { return loader_.config(); }
    const ModelLoader& loader() const { return loader_; }

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

private:
    Model() = default;

    // Core orchestration: 16 kHz mono PCM -> L2-normalized embedding. Shared by
    // embed_pcm / embed_path / verify_paths.
    std::vector<float> embed_16k(const std::vector<float>& pcm16k) const;

    ModelLoader loader_;
};

} // namespace vd
