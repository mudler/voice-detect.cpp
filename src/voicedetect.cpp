#include "voicedetect.h"
#include "model.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#define VOICEDETECT_VERSION "0.0.1"

extern "C" const char* voicedetect_version(void) { return VOICEDETECT_VERSION; }

namespace vd {

// Thin convenience wrappers: load a model and run one call. For repeated work
// (and for the flat C-API / LocalAI) use vd::Model directly to avoid reloading
// the GGUF on every call.

std::vector<float> embed(const std::string& model_path, const std::string& wav_path) {
    std::unique_ptr<Model> model = Model::load(model_path);
    if (!model) {
        throw std::runtime_error("voicedetect: failed to load model: " + model_path);
    }
    return model->embed_path(wav_path);
}

float verify(const std::string& model_path, const std::string& a, const std::string& b) {
    std::unique_ptr<Model> model = Model::load(model_path);
    if (!model) {
        throw std::runtime_error("voicedetect: failed to load model: " + model_path);
    }
    float distance = 0.0f;
    bool  verified = false;
    model->verify_paths(a, b, /*threshold=*/0.0f, distance, verified);
    return distance;
}

} // namespace vd
