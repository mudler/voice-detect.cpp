// Isolated ERes2Net encoder-forward benchmark (mirror of bench_wespeaker).
// Times ONLY the encoder graph (FBank computed once, excluded). Threads via
// --threads. Conv routing honours the production env vars (VOICEDETECT_WINOGRAD,
// VD_CONV2D, VD_BLOCKED_BACKBONE, VOICEDETECT_DISABLE_AVX512).
#include "encoder.hpp"
#include "model_loader.hpp"
#include "fbank.hpp"
#include "audio_io.hpp"
#include "backend.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

int main(int argc, char** argv) {
    const char* gguf  = std::getenv("VOICEDETECT_TEST_GGUF");
    const char* audio = std::getenv("VOICEDETECT_TEST_AUDIO");
    int threads = 1, N = 20;
    for (int i = 1; i + 1 < argc; ++i) {
        if (!std::strcmp(argv[i], "--threads")) threads = std::atoi(argv[i + 1]);
        if (!std::strcmp(argv[i], "--n"))       N       = std::atoi(argv[i + 1]);
    }
    if (!gguf || !audio) { std::fprintf(stderr, "env unset\n"); return 77; }

    vd::set_num_threads(threads);
    vd::ModelLoader ml;
    if (!ml.load(gguf)) { std::fprintf(stderr, "load failed\n"); return 77; }
    vd::Audio in;
    if (!vd::load_audio_16k_mono(audio, in)) return 77;
    int T = 0;
    std::vector<float> feats = vd::FBank(ml.config()).compute(in.samples, T);
    if (feats.empty()) return 77;

    vd::ERes2NetEncoder enc(ml);
    std::vector<float> pre;
    for (int w = 0; w < 3; ++w) enc.forward_capture(feats, T, pre);

    std::vector<double> ms;
    ms.reserve(N);
    for (int i = 0; i < N; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        std::vector<float> e = enc.forward_capture(feats, T, pre);
        auto t1 = std::chrono::steady_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    double med = ms[ms.size() / 2];
    double best = ms.front();
    double sum = 0; for (double v : ms) sum += v;
    std::fprintf(stderr, "[bench eres2net] threads=%d T=%d N=%d  median=%.2f ms  best=%.2f ms  mean=%.2f ms\n",
                 threads, T, N, med, best, sum / N);
    vd::shutdown_backend();
    return 0;
}
