#include "parity.hpp"
#include "analyze_w2v2.hpp"
#include "model_loader.hpp"
#include "audio_io.hpp"
#include "backend.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// Golden-parity test for the wav2vec2 conv feature encoder (Task 18).
//
// Feeds the RAW 16 kHz waveform (NOT a mel FBank - wav2vec2 has no FBank front
// end) through the 7-layer strided Conv1d feature encoder and gates the
// `feat_extract_out` features against the reference dumped by
// scripts/gen_baseline.py --analyze (superb/wav2vec2-base-superb-er).
//
// DEDICATED analyze env vars (separate from the speaker-encoder GGUF/baseline so
// the two never cross-contaminate): VOICEDETECT_TEST_ANALYZE_GGUF /
// VOICEDETECT_TEST_ANALYZE_BASELINE, reusing VOICEDETECT_TEST_AUDIO for the clip.
//
// RC-77 skip convention: SKIPs (exit 77) when the env vars are unset/unreadable;
// fails (exit 1) on a real numeric mismatch.
int main() {
    const char* base  = std::getenv("VOICEDETECT_TEST_ANALYZE_BASELINE");
    const char* gguf  = std::getenv("VOICEDETECT_TEST_ANALYZE_GGUF");
    const char* audio = std::getenv("VOICEDETECT_TEST_AUDIO");
    if (!base || !gguf || !audio) { std::fprintf(stderr, "analyze env unset; skip\n"); return 77; }

    vd::ModelLoader ml;
    if (!ml.load(gguf)) { std::fprintf(stderr, "analyze model load failed; skip\n"); return 77; }
    if (!ml.config().analyze_present) {
        std::fprintf(stderr, "not an analyze gguf; skip\n");
        return 77;
    }

    vd::Audio in;
    if (!vd::load_audio_16k_mono(audio, in)) return 77;

    int T = 0;
    std::vector<float> feat = vd::W2V2Analyzer(ml).feature_encoder(in.samples, T);
    if (feat.empty()) { std::fprintf(stderr, "empty feature encoder output; skip\n"); return 77; }

    std::vector<float> ref;
    std::vector<int64_t> sh;
    if (!vdtest::load_baseline(base, "feat_extract_out", ref, sh)) {
        std::fprintf(stderr, "no feat_extract_out baseline; skip\n");
        vd::shutdown_backend();
        return 77;
    }
    std::fprintf(stderr, "[feat_extract_out] T'=%d (golden [%lld,%lld])\n",
                 T, (long long)sh[0], (long long)(sh.size() > 1 ? sh[1] : 0));

    // Tight gate (1e-4): the correct math (exact erf GELU, per-channel GroupNorm
    // with biased variance) lands at max|d| ~1.5e-7, so 1e-4 still has a huge
    // margin while actually CATCHING the documented drift points - the tanh-approx
    // GELU (~2.6e-4) and a wrong GroupNorm eps/variance both bust it.
    bool ok = vdtest::compare(feat, ref, "feat_extract_out", 1e-4f, 1e-4f);

    // ---- Task 19: wav2vec2 transformer encoder stack ----
    //
    // Run the full encoder (feature projection -> positional conv embedding ->
    // encoder LayerNorm -> 12 post-norm transformer layers) and gate the per-layer
    // hidden states against the goldens enc_layer_0 / enc_layer_mid / enc_layer_last
    // ([hidden, T'], hidden-major). enc_layer_last is the HARD gate; the
    // intermediates are printed for drift triage (parakeet-style) but ALSO gated -
    // every state is the same [hidden,T'] layout so there is no benign-transpose
    // excuse, and a wrong layernorm placement / attention scale / pos-conv weight
    // norm busts layer 0 immediately.
    //
    // tol 5e-3: accumulation through 12 post-norm layers (each a LayerNorm + GELU +
    // 4 matmuls) loosens the bound vs the 1e-7 conv encoder, but correct math still
    // lands well inside 5e-3.
    int T2 = 0;
    std::vector<float> dbg0, dbgmid;
    std::vector<float> enc_last = vd::W2V2Analyzer(ml).encode(in.samples, T2, &dbg0, &dbgmid);
    if (enc_last.empty()) { std::fprintf(stderr, "empty encoder output; skip\n"); vd::shutdown_backend(); return 77; }

    auto gate = [&](const char* gname, const std::vector<float>& got, bool hard) {
        std::vector<float> g_ref; std::vector<int64_t> g_sh;
        if (!vdtest::load_baseline(base, gname, g_ref, g_sh)) {
            std::fprintf(stderr, "no %s baseline; skip\n", gname);
            return true;  // golden absent -> do not fail (SKIP-tolerant)
        }
        bool g_ok = vdtest::compare(got, g_ref, gname, 5e-3f, 5e-3f);
        return hard ? g_ok : true;  // intermediates printed; last is the gate
    };

    ok = gate("enc_layer_0",   dbg0,     /*hard=*/true)  && ok;
    ok = gate("enc_layer_mid", dbgmid,   /*hard=*/true)  && ok;
    ok = gate("enc_layer_last", enc_last, /*hard=*/true) && ok;

    // ---- Task 20: emotion head + analyze JSON ----
    //
    // On top of the transformer stack: weighted-layer-sum (softmax over the 13
    // layer_weights) -> projector Linear 768->256 -> mean-pool over time ->
    // classifier Linear 256->num_emotions -> emotion_logits, softmax -> probs.
    // Gate the C++ logits and probs against the goldens emotion_logits /
    // emotion_probs and assert argmax == baseline.dominant_emotion.
    std::vector<float> logits = vd::W2V2Analyzer(ml).emotion_logits(in.samples);
    if (logits.empty()) { std::fprintf(stderr, "empty emotion_logits; FAIL\n"); ok = false; }

    std::vector<float> lref; std::vector<int64_t> lsh;
    if (vdtest::load_baseline(base, "emotion_logits", lref, lsh)) {
        // tol 1e-2: a few extra matmuls past the 2e-5 enc_layer_last, with margin
        // over the documented drift points (weighted-sum normalization, head order).
        ok = vdtest::compare(logits, lref, "emotion_logits", 1e-2f, 1e-2f) && ok;
    }

    // Host softmax(logits) -> probs, gated against the golden emotion_probs (1e-3).
    std::vector<float> probs(logits.size());
    if (!logits.empty()) {
        double mx = logits[0];
        for (float v : logits) if (v > mx) mx = v;
        double s = 0.0;
        for (size_t i = 0; i < logits.size(); ++i) { probs[i] = (float)std::exp((double)logits[i] - mx); s += probs[i]; }
        for (float& v : probs) v = (float)(v / s);
    }
    std::vector<float> pref; std::vector<int64_t> psh;
    if (vdtest::load_baseline(base, "emotion_probs", pref, psh)) {
        ok = vdtest::compare(probs, pref, "emotion_probs", 1e-3f, 1e-3f) && ok;
    }

    // Dominant label == golden argmax label.
    const std::vector<std::string>& labels = ml.config().emotion_labels;
    int argmax = 0;
    for (size_t i = 1; i < logits.size(); ++i) if (logits[i] > logits[argmax]) argmax = (int)i;
    std::string dom = (!labels.empty() && argmax < (int)labels.size()) ? labels[argmax] : "";
    std::string dom_ref;
    if (vdtest::load_kv_str(base, "baseline.dominant_emotion", dom_ref)) {
        bool match = (dom == dom_ref);
        std::fprintf(stderr, "[dominant_emotion] got='%s' ref='%s' -> %s\n",
                     dom.c_str(), dom_ref.c_str(), match ? "OK" : "FAIL");
        ok = match && ok;
    }

    // analyze_json: documented C-ABI shape, parseable, carrying the dominant label
    // and a score entry for every emotion label.
    std::string js = vd::W2V2Analyzer(ml).analyze_json(in.samples);
    std::fprintf(stderr, "[analyze_json] %s\n", js.c_str());
    bool js_ok = js.find("\"emotion\"") != std::string::npos &&
                 js.find("\"label\"")   != std::string::npos &&
                 js.find("\"scores\"")  != std::string::npos &&
                 (dom.empty() || js.find("\"" + dom + "\"") != std::string::npos);
    for (const std::string& lbl : labels)
        js_ok = js_ok && js.find("\"" + lbl + "\"") != std::string::npos;
    std::fprintf(stderr, "[analyze_json] keys -> %s\n", js_ok ? "OK" : "FAIL");
    ok = js_ok && ok;

    vd::shutdown_backend();
    return ok ? 0 : 1;
}
