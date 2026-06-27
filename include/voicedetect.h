#ifndef VOICEDETECT_H
#define VOICEDETECT_H
#ifdef __cplusplus
extern "C" {
#endif
// Returns a static version string. Never null.
const char* voicedetect_version(void);
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <string>
#include <vector>
namespace vd {

// Convenience C++ layer over the load-once vd::Model. For repeated work (and for
// the flat C-API / LocalAI) use vd::Model directly to avoid reloading the GGUF
// on every call. These free functions reload the model each time.

// Compute the L2-normalized speaker embedding for a WAV file with the given
// GGUF model. Throws std::runtime_error on failure (model/audio load, etc.).
std::vector<float> embed(const std::string& model_path, const std::string& wav_path);

// Verify whether two clips are the same speaker: returns the cosine distance
// (1 - cosine_similarity) between the two L2-normalized embeddings. The caller
// compares it against a threshold (smaller = more similar). Throws
// std::runtime_error on failure.
float verify(const std::string& model_path, const std::string& a, const std::string& b);

} // namespace vd
#endif

#endif // VOICEDETECT_H
