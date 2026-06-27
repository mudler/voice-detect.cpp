#include "model.hpp"
#include "backend.hpp"
#include <cmath>
#include <cstdlib>

// End-to-end embed behavior check: raw 16 kHz PCM (a fixture WAV) -> FBank ->
// EcapaEncoder -> 192-d L2-normalized embedding, exercised through the public
// Model::embed_path path the C-ABI calls. Gates the shape (== embedding_dim) and
// the L2 norm (~= 1, the encoder normalizes), independent of any baseline.
//
// RC-77 skip convention: SKIPs (exit 77) when the model/audio env vars are unset
// or the GGUF fails to load, so CI without a checkpoint never breaks.
int main() {
    const char* gguf = std::getenv("VOICEDETECT_TEST_GGUF");
    const char* audio = std::getenv("VOICEDETECT_TEST_AUDIO");
    if (!gguf || !audio) return 77;
    auto m = vd::Model::load(gguf);
    if (!m) return 77;
    auto e = m->embed_path(audio);
    bool ok = ((int)e.size() == m->embedding_dim());
    double n = 0; for (float x : e) n += (double)x * x;
    ok = ok && std::abs(n - 1.0) < 1e-3;   // L2-normalized
    vd::shutdown_backend();
    return ok ? 0 : 1;
}
