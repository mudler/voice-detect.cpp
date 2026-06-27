#include "voicedetect.h"
#include "model.hpp"
#include "model_loader.hpp"
#include "backend.hpp"   // vd::set_num_threads, vd::shutdown_backend

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

namespace {

// Minimal --flag <value> parser over argv. Returns the value for `flag`, or
// `dflt` if absent. Boolean flags (no value) are handled by has_flag().
std::string get_opt(int argc, char** argv, const char* flag, const std::string& dflt = "") {
    for (int i = 0; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
    return dflt;
}
bool has_flag(int argc, char** argv, const char* flag) {
    for (int i = 0; i < argc; ++i)
        if (std::strcmp(argv[i], flag) == 0) return true;
    return false;
}

void apply_threads(int argc, char** argv) {
    const std::string t = get_opt(argc, argv, "--threads");
    if (!t.empty()) vd::set_num_threads(std::atoi(t.c_str()));
}

int cmd_info(int argc, char** argv) {
    const std::string model = get_opt(argc, argv, "--model");
    if (model.empty()) { std::fprintf(stderr, "info: --model is required\n"); return 1; }
    vd::ModelLoader ml;
    if (!ml.load(model)) {
        std::fprintf(stderr, "voicedetect-cli: failed to load model %s\n", model.c_str());
        return 1;
    }
    const vd::VoiceDetectConfig& c = ml.config();
    std::printf("voice-detect.cpp %s\n", voicedetect_version());
    std::printf("model: %s\n", model.c_str());
    std::printf("  arch            : %s\n", c.arch.c_str());
    std::printf("  embedding_dim   : %u (l2_normalize=%s)\n",
                c.embedding_dim, c.l2_normalize ? "true" : "false");
    std::printf("  fbank           : n_mels=%u n_fft=%u win=%u hop=%u sr=%u\n",
                c.n_mels, c.n_fft, c.win_length, c.hop_length, c.sample_rate);
    std::printf("  fbank window    : %s (preemph=%.2f low=%.1f high=%.1f cmn=%s)\n",
                c.fbank_window.c_str(), c.preemph, c.fbank_low_freq, c.fbank_high_freq,
                c.fbank_cmn ? "true" : "false");
    std::printf("  analyze heads   : %s\n", c.analyze_present ? "present" : "absent");
    return 0;
}

int cmd_embed(int argc, char** argv) {
    apply_threads(argc, argv);
    const std::string model = get_opt(argc, argv, "--model");
    const std::string input = get_opt(argc, argv, "--input");
    const bool as_json = has_flag(argc, argv, "--json");
    if (model.empty() || input.empty()) {
        std::fprintf(stderr, "embed: --model and --input are required\n");
        return 1;
    }
    std::unique_ptr<vd::Model> m = vd::Model::load(model);
    if (!m) { std::fprintf(stderr, "voicedetect-cli: failed to load model %s\n", model.c_str()); return 1; }
    try {
        std::vector<float> emb = m->embed_path(input);
        if (as_json) {
            std::printf("{\"dim\":%zu,\"embedding\":[", emb.size());
            for (size_t i = 0; i < emb.size(); ++i) std::printf("%s%.6f", i ? "," : "", emb[i]);
            std::printf("]}\n");
        } else {
            for (size_t i = 0; i < emb.size(); ++i) std::printf("%s%.6f", i ? " " : "", emb[i]);
            std::printf("\n");
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "voicedetect-cli: %s\n", e.what());
        return 1;
    }
}

int cmd_verify(int argc, char** argv) {
    apply_threads(argc, argv);
    const std::string model = get_opt(argc, argv, "--model");
    const std::string a = get_opt(argc, argv, "--a");
    const std::string b = get_opt(argc, argv, "--b");
    const std::string th = get_opt(argc, argv, "--threshold", "0.25");
    if (model.empty() || a.empty() || b.empty()) {
        std::fprintf(stderr, "verify: --model, --a and --b are required\n");
        return 1;
    }
    std::unique_ptr<vd::Model> m = vd::Model::load(model);
    if (!m) { std::fprintf(stderr, "voicedetect-cli: failed to load model %s\n", model.c_str()); return 1; }
    try {
        float distance = 0.0f; bool verified = false;
        m->verify_paths(a, b, (float)std::atof(th.c_str()), distance, verified);
        std::printf("distance=%.6f threshold=%s verified=%s\n",
                    distance, th.c_str(), verified ? "true" : "false");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "voicedetect-cli: %s\n", e.what());
        return 1;
    }
}

int cmd_analyze(int argc, char** argv) {
    apply_threads(argc, argv);
    const std::string model = get_opt(argc, argv, "--model");
    const std::string input = get_opt(argc, argv, "--input");
    if (model.empty() || input.empty()) {
        std::fprintf(stderr, "analyze: --model and --input are required\n");
        return 1;
    }
    std::unique_ptr<vd::Model> m = vd::Model::load(model);
    if (!m) { std::fprintf(stderr, "voicedetect-cli: failed to load model %s\n", model.c_str()); return 1; }
    try {
        std::printf("%s\n", m->analyze_path_json(input).c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "voicedetect-cli: %s\n", e.what());
        return 1;
    }
}

int cmd_bench(int argc, char** argv) {
    apply_threads(argc, argv);
    const std::string model = get_opt(argc, argv, "--model");
    const std::string input = get_opt(argc, argv, "--input");
    // mode = "embed" (speaker encoders) or "analyze" (wav2vec2 analyze heads).
    const std::string mode = get_opt(argc, argv, "--mode", "embed");
    const std::string n_str = get_opt(argc, argv, "--n", "20");
    if (model.empty() || input.empty()) {
        std::fprintf(stderr, "bench: --model and --input are required\n");
        return 1;
    }
    const int N = std::max(1, std::atoi(n_str.c_str()));
    const bool analyze = (mode == "analyze");
    std::unique_ptr<vd::Model> m = vd::Model::load(model);
    if (!m) { std::fprintf(stderr, "voicedetect-cli: failed to load %s\n", model.c_str()); return 1; }
    try {
        auto once = [&]() {
            if (analyze) (void)m->analyze_path_json(input);
            else         (void)m->embed_path(input);
        };
        once();  // warmup (excluded from the timed loop)
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < N; ++i) once();
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / N;
        // Machine-parseable line consumed by scripts/bench_compare.py.
        std::printf("%s: %.2f ms/clip over %d passes\n", analyze ? "analyze" : "embed", ms, N);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "voicedetect-cli: %s\n", e.what());
        return 1;
    }
}

int usage() {
    std::fprintf(stderr,
        "usage:\n"
        "  voicedetect-cli info    --model <model.gguf>\n"
        "  voicedetect-cli embed   --model <model.gguf> --input <wav> [--threads N] [--json]\n"
        "  voicedetect-cli verify  --model <model.gguf> --a <wav> --b <wav> [--threshold T] [--threads N]\n"
        "  voicedetect-cli analyze --model <model.gguf> --input <wav> [--threads N]\n"
        "  voicedetect-cli bench   --model <model.gguf> --input <wav> [--mode embed|analyze] [--n N] [--threads N]\n");
    return 2;
}

} // namespace

int main(int argc, char** argv) {
    int rc;
    if (argc < 2)                                   { return usage(); }
    else if (std::strcmp(argv[1], "info")    == 0)  rc = cmd_info(argc - 1, argv + 1);
    else if (std::strcmp(argv[1], "embed")   == 0)  rc = cmd_embed(argc - 1, argv + 1);
    else if (std::strcmp(argv[1], "verify")  == 0)  rc = cmd_verify(argc - 1, argv + 1);
    else if (std::strcmp(argv[1], "analyze") == 0)  rc = cmd_analyze(argc - 1, argv + 1);
    else if (std::strcmp(argv[1], "bench")   == 0)  rc = cmd_bench(argc - 1, argv + 1);
    else                                            return usage();
    // Free the process-global backend before exit so GPU backends release device
    // memory while the driver is still alive.
    vd::shutdown_backend();
    return rc;
}
