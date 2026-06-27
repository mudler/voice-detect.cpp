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

// Golden-parity test for the audeering wav2vec2-large-robust age/gender analyze
// head. This is a SEPARATE analyze model from the emotion one (a distinct GGUF),
// mirroring how the Python speaker-recognition backend loads them apart. It
// exercises the large-robust deltas vs the base emotion model: feat_extract_norm
// "layer" (every conv LayerNorm'd + conv bias) and do_stable_layer_norm (pre-norm
// transformer layers + a final encoder LayerNorm), 24 layers / hidden 1024.
//
// DEDICATED env vars (separate from the emotion analyze GGUF/baseline so the two
// never cross-contaminate): VOICEDETECT_TEST_AGEGENDER_GGUF /
// VOICEDETECT_TEST_AGEGENDER_BASELINE, reusing VOICEDETECT_TEST_AUDIO for the clip.
//
// RC-77 skip convention: SKIPs (exit 77) when the env vars are unset/unreadable;
// fails (exit 1) on a real numeric mismatch.
int main() {
    const char* base  = std::getenv("VOICEDETECT_TEST_AGEGENDER_BASELINE");
    const char* gguf  = std::getenv("VOICEDETECT_TEST_AGEGENDER_GGUF");
    const char* audio = std::getenv("VOICEDETECT_TEST_AUDIO");
    if (!base || !gguf || !audio) { std::fprintf(stderr, "age/gender env unset; skip\n"); return 77; }

    vd::ModelLoader ml;
    if (!ml.load(gguf)) { std::fprintf(stderr, "age/gender model load failed; skip\n"); return 77; }
    if (!ml.config().analyze_present || ml.config().arch != "wav2vec2_age_gender") {
        std::fprintf(stderr, "not an age/gender gguf; skip\n");
        return 77;
    }

    vd::Audio in;
    if (!vd::load_audio_16k_mono(audio, in)) return 77;

    // ---- conv feature encoder (layer-norm variant + conv bias) ----
    int T = 0;
    std::vector<float> feat = vd::W2V2Analyzer(ml).feature_encoder(in.samples, T);
    if (feat.empty()) { std::fprintf(stderr, "empty feature encoder output; skip\n"); return 77; }

    bool ok = true;
    std::vector<float> ref; std::vector<int64_t> sh;
    if (vdtest::load_baseline(base, "feat_extract_out", ref, sh)) {
        std::fprintf(stderr, "[feat_extract_out] T'=%d (golden [%lld,%lld])\n",
                     T, (long long)sh[0], (long long)(sh.size() > 1 ? sh[1] : 0));
        // Conv encoder is the exact erf-GELU + per-channel LayerNorm path; correct
        // math lands far inside 1e-3 while still catching a wrong norm/bias/GELU.
        ok = vdtest::compare(feat, ref, "feat_extract_out", 1e-3f, 1e-3f) && ok;
    }

    // ---- transformer backbone (stable / pre-norm layer norm) ----
    // encode() returns the final hidden state (the last encoder LayerNorm output),
    // == HF last_hidden_state == golden enc_layer_last. tol 5e-3: accumulation
    // through 24 pre-norm layers loosens the bound but correct math stays inside.
    int T2 = 0;
    std::vector<float> enc_last = vd::W2V2Analyzer(ml).encode(in.samples, T2);
    if (enc_last.empty()) { std::fprintf(stderr, "empty encoder output; skip\n"); vd::shutdown_backend(); return 77; }
    std::vector<float> eref; std::vector<int64_t> esh;
    if (vdtest::load_baseline(base, "enc_layer_last", eref, esh)) {
        ok = vdtest::compare(enc_last, eref, "enc_layer_last", 5e-3f, 5e-3f) && ok;
    }

    // ---- age/gender head ----
    // age_gender_raw() returns [age_raw, gender_logit_0, ...]: age_raw*100 = years,
    // softmax(gender_logits) = gender probs.
    std::vector<float> raw = vd::W2V2Analyzer(ml).age_gender_raw(in.samples);
    if (raw.empty()) { std::fprintf(stderr, "empty age/gender output; FAIL\n"); ok = false; }

    std::vector<float> age_got(raw.begin(), raw.begin() + (raw.empty() ? 0 : 1));
    std::vector<float> glogits_got(raw.begin() + (raw.empty() ? 0 : 1), raw.end());

    // age_raw gate (tol 1e-2 on the raw scalar -> ~1 year after *100, the spec bar).
    std::vector<float> aref; std::vector<int64_t> ash;
    if (vdtest::load_baseline(base, "age_raw", aref, ash)) {
        ok = vdtest::compare(age_got, aref, "age_raw", 1e-2f, 1e-2f) && ok;
        if (!aref.empty() && !age_got.empty())
            std::fprintf(stderr, "[age_years] got=%.3f ref=%.3f\n",
                         age_got[0] * 100.0f, aref[0] * 100.0f);
    }

    // gender logits gate (tol 1e-2, a few matmuls past enc_layer_last).
    std::vector<float> lref; std::vector<int64_t> lsh;
    if (vdtest::load_baseline(base, "gender_logits", lref, lsh)) {
        ok = vdtest::compare(glogits_got, lref, "gender_logits", 1e-2f, 1e-2f) && ok;
    }

    // gender probs (host softmax of the logits) gate (tol 1e-3).
    std::vector<float> gprobs(glogits_got.size());
    if (!glogits_got.empty()) {
        double mx = glogits_got[0];
        for (float v : glogits_got) if (v > mx) mx = v;
        double s = 0.0;
        for (size_t i = 0; i < glogits_got.size(); ++i) { gprobs[i] = (float)std::exp((double)glogits_got[i] - mx); s += gprobs[i]; }
        for (float& v : gprobs) v = (float)(v / s);
    }
    std::vector<float> pref; std::vector<int64_t> psh;
    if (vdtest::load_baseline(base, "gender_probs", pref, psh)) {
        ok = vdtest::compare(gprobs, pref, "gender_probs", 1e-3f, 1e-3f) && ok;
    }

    // Dominant gender label == golden argmax label.
    const std::vector<std::string>& labels = ml.config().gender_labels;
    int argmax = 0;
    for (size_t i = 1; i < glogits_got.size(); ++i) if (glogits_got[i] > glogits_got[argmax]) argmax = (int)i;
    std::string dom = (!labels.empty() && argmax < (int)labels.size()) ? labels[argmax] : "";
    std::string dom_ref;
    if (vdtest::load_kv_str(base, "baseline.dominant_gender", dom_ref)) {
        bool match = (dom == dom_ref);
        std::fprintf(stderr, "[dominant_gender] got='%s' ref='%s' -> %s\n",
                     dom.c_str(), dom_ref.c_str(), match ? "OK" : "FAIL");
        ok = match && ok;
    }

    // analyze_age_gender_json: documented C-ABI shape, parseable, carrying age,
    // the dominant gender label, and a score entry for every gender label.
    std::string js = vd::W2V2Analyzer(ml).analyze_age_gender_json(in.samples);
    std::fprintf(stderr, "[analyze_json] %s\n", js.c_str());
    bool js_ok = js.find("\"age\"")    != std::string::npos &&
                 js.find("\"gender\"") != std::string::npos &&
                 js.find("\"label\"")  != std::string::npos &&
                 (dom.empty() || js.find("\"" + dom + "\"") != std::string::npos);
    for (const std::string& lbl : labels)
        js_ok = js_ok && js.find("\"" + lbl + "\"") != std::string::npos;
    std::fprintf(stderr, "[analyze_json] keys -> %s\n", js_ok ? "OK" : "FAIL");
    ok = js_ok && ok;

    vd::shutdown_backend();
    return ok ? 0 : 1;
}
