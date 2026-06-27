#include "parity.hpp"
#include "fbank.hpp"
#include "audio_io.hpp"
#include "model_loader.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// Golden-parity test for the Kaldi-compatible FBank front end.
//
// Compares vd::FBank::compute against the reference "fbank" tensor dumped by
// scripts/gen_baseline.py (torchaudio.compliance.kaldi.fbank with the encoder's
// exact options). Gates: cosine >= 0.9999 and max-abs-diff <= 1e-3 (see
// docs/parity.md).
//
// RC-77 skip convention: this test SKIPS (exit 77) when the baseline GGUF env
// var is unset or unreadable, so CI without the reference venv never fails here.
// It will fail (exit 1) on a real numerical mismatch once FBank is implemented.
int main() {
    const char* baseline = std::getenv("VOICEDETECT_TEST_BASELINE");
    const char* model    = std::getenv("VOICEDETECT_TEST_GGUF");
    const char* audio    = std::getenv("VOICEDETECT_TEST_AUDIO");
    if (!baseline || !model || !audio) {
        std::fprintf(stderr,
            "test_fbank: set VOICEDETECT_TEST_BASELINE, VOICEDETECT_TEST_GGUF and "
            "VOICEDETECT_TEST_AUDIO to run; skipping\n");
        return 77;  // ctest SKIP
    }

    vd::ModelLoader ml;
    if (!ml.load(model)) {
        std::fprintf(stderr, "test_fbank: failed to load model %s; skipping\n", model);
        return 77;
    }

    vd::Audio in;
    if (!vd::load_audio_16k_mono(audio, in)) {
        std::fprintf(stderr, "test_fbank: failed to load audio %s; skipping\n", audio);
        return 77;
    }

    std::vector<float> ref;
    std::vector<int64_t> shape;
    if (!vdtest::load_baseline(baseline, "fbank", ref, shape)) {
        std::fprintf(stderr, "test_fbank: baseline missing 'fbank' tensor; skipping\n");
        return 77;
    }

    int T = 0;
    std::vector<float> got = vd::FBank(ml.config()).compute(in.samples, T);

    // Real numerical gate: feat-major [n_mels, T] vs the golden "fbank" (post-CMN)
    // tensor. Both the max-abs-diff bound and the cosine bound must hold.
    const bool   ok_abs = vdtest::compare(got, ref, "fbank", /*atol=*/1e-3f, /*rtol=*/0.0f);
    const double cs     = vdtest::cosine(got, ref);
    std::fprintf(stderr, "[fbank] cosine=%.6f\n", cs);
    return (ok_abs && cs >= 0.9999) ? 0 : 1;
}
