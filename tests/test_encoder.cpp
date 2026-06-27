#include "parity.hpp"
#include "encoder.hpp"
#include "model_loader.hpp"
#include "fbank.hpp"
#include "audio_io.hpp"
#include "backend.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>

// Golden-parity test for the ECAPA-TDNN encoder block0 (TDNN frontend).
//
// Feeds the engine's real front end (Kaldi FBank + CMN) through the block0
// graph and gates the captured block0 activation against the reference
// "block0_out" tensor dumped by scripts/gen_baseline.py - which is captured by
// running embedding_model on the SAME Kaldi features, so this isolates the
// Conv1d + ReLU + BatchNorm1d math from any front-end framing difference.
//
// RC-77 skip convention: SKIPs (exit 77) when the baseline/model/audio env vars
// are unset or unreadable; fails (exit 1) on a real numeric mismatch.
int main() {
    const char* base  = std::getenv("VOICEDETECT_TEST_BASELINE");
    const char* gguf  = std::getenv("VOICEDETECT_TEST_GGUF");
    const char* audio = std::getenv("VOICEDETECT_TEST_AUDIO");
    if (!base || !gguf || !audio) { std::fprintf(stderr, "env unset; skip\n"); return 77; }

    vd::ModelLoader ml;
    if (!ml.load(gguf)) { std::fprintf(stderr, "model load failed; skip\n"); return 77; }

    vd::Audio in;
    if (!vd::load_audio_16k_mono(audio, in)) return 77;

    int T = 0;
    std::vector<float> feats = vd::FBank(ml.config()).compute(in.samples, T);
    if (feats.empty()) return 77;

    // WeSpeaker ResNet34 (ONNX-direct) parity gate. The baseline dumps the raw
    // ONNX `embs` (encoder_out) and its unit-normalized form (embedding). The
    // C++ embedding must match at cosine >= 0.9999 AND max|d| <= 1e-3 (the fp32
    // parity gate from the spec).
    if (ml.config().arch == "wespeaker_resnet34") {
        std::vector<float> pre_norm;
        std::vector<float> emb = vd::ResNet34Encoder(ml).forward_capture(feats, T, pre_norm);

        bool ok = true;
        std::vector<float> rc; std::vector<int64_t> sc;
        if (vdtest::load_baseline(base, "encoder_out", rc, sc)) {
            bool ok_enc = vdtest::compare(pre_norm, rc, "encoder_out", 1e-3f, 1e-3f);
            double ce = vdtest::cosine(pre_norm, rc);
            std::fprintf(stderr, "[encoder_out] cosine=%.6f\n", ce);
            ok = ok && ok_enc && ce >= 0.9999;
        }

        std::vector<float> re; std::vector<int64_t> se;
        if (!vdtest::load_baseline(base, "embedding", re, se)) {
            std::fprintf(stderr, "no embedding baseline; skip\n");
            vd::shutdown_backend();
            return 77;
        }
        bool ok_abs = vdtest::compare(emb, re, "embedding", 1e-3f, 0.0f);
        double cs = vdtest::cosine(emb, re);
        std::fprintf(stderr, "[embedding] cosine=%.6f\n", cs);
        ok = ok && ok_abs && cs >= 0.9999;

        vd::shutdown_backend();
        return ok ? 0 : 1;
    }

    // 3D-Speaker ERes2Net (ONNX-direct) parity gate. Same contract as WeSpeaker:
    // the baseline dumps the raw ONNX embedding (encoder_out) and its unit-norm
    // form (embedding); the C++ ERes2Net graph must match at cosine >= 0.9999 AND
    // max|d| <= 1e-3.
    if (ml.config().arch == "eres2net") {
        std::vector<float> pre_norm;
        std::vector<float> emb = vd::ERes2NetEncoder(ml).forward_capture(feats, T, pre_norm);

        bool ok = true;
        std::vector<float> rc; std::vector<int64_t> sc;
        if (vdtest::load_baseline(base, "encoder_out", rc, sc)) {
            bool ok_enc = vdtest::compare(pre_norm, rc, "encoder_out", 1e-3f, 1e-3f);
            double ce = vdtest::cosine(pre_norm, rc);
            std::fprintf(stderr, "[encoder_out] cosine=%.6f\n", ce);
            ok = ok && ok_enc && ce >= 0.9999;
        }

        std::vector<float> re; std::vector<int64_t> se;
        if (!vdtest::load_baseline(base, "embedding", re, se)) {
            std::fprintf(stderr, "no embedding baseline; skip\n");
            vd::shutdown_backend();
            return 77;
        }
        bool ok_abs = vdtest::compare(emb, re, "embedding", 1e-3f, 0.0f);
        double cs = vdtest::cosine(emb, re);
        std::fprintf(stderr, "[embedding] cosine=%.6f\n", cs);
        ok = ok && ok_abs && cs >= 0.9999;

        vd::shutdown_backend();
        return ok ? 0 : 1;
    }

    // 3D-Speaker CAM++ (ONNX-direct) parity gate. Same contract as the other
    // ONNX-direct archs: the baseline dumps the raw ONNX embedding (encoder_out,
    // 192-d) and its unit-norm form (embedding); the C++ CAM++ graph (FCM +
    // D-TDNN + CAM masking + stats pooling) must match at cosine >= 0.9999 AND
    // max|d| <= 1e-3.
    if (ml.config().arch == "campplus") {
        std::vector<float> pre_norm;
        std::vector<float> emb = vd::CamPPlusEncoder(ml).forward_capture(feats, T, pre_norm);

        bool ok = true;
        std::vector<float> rc; std::vector<int64_t> sc;
        if (vdtest::load_baseline(base, "encoder_out", rc, sc)) {
            bool ok_enc = vdtest::compare(pre_norm, rc, "encoder_out", 1e-3f, 1e-3f);
            double ce = vdtest::cosine(pre_norm, rc);
            std::fprintf(stderr, "[encoder_out] cosine=%.6f\n", ce);
            ok = ok && ok_enc && ce >= 0.9999;
        }

        std::vector<float> re; std::vector<int64_t> se;
        if (!vdtest::load_baseline(base, "embedding", re, se)) {
            std::fprintf(stderr, "no embedding baseline; skip\n");
            vd::shutdown_backend();
            return 77;
        }
        bool ok_abs = vdtest::compare(emb, re, "embedding", 1e-3f, 0.0f);
        double cs = vdtest::cosine(emb, re);
        std::fprintf(stderr, "[embedding] cosine=%.6f\n", cs);
        ok = ok && ok_abs && cs >= 0.9999;

        vd::shutdown_backend();
        return ok ? 0 : 1;
    }

    vd::EcapaCaptures caps;
    std::vector<float> emb = vd::EcapaEncoder(ml).forward_capture(feats, T, caps);

    std::vector<float> ref;
    std::vector<int64_t> sh;
    if (!vdtest::load_baseline(base, "block0_out", ref, sh)) return 77;

    bool ok = vdtest::compare(caps.block0, ref, "block0_out", 1e-3f, 1e-3f);

    std::vector<float> r1; std::vector<int64_t> s1;
    if (vdtest::load_baseline(base, "block1_out", r1, s1))
        ok = ok && vdtest::compare(caps.block1, r1, "block1_out", 5e-3f, 5e-3f);

    std::vector<float> r2; std::vector<int64_t> s2;
    if (vdtest::load_baseline(base, "block2_out", r2, s2))
        ok = ok && vdtest::compare(caps.block2, r2, "block2_out", 5e-3f, 5e-3f);

    std::vector<float> r3; std::vector<int64_t> s3;
    if (vdtest::load_baseline(base, "block3_out", r3, s3))
        ok = ok && vdtest::compare(caps.block3, r3, "block3_out", 5e-3f, 5e-3f);

    // MFA aggregates blocks 1,2,3 (3*1024=3072 channels) -> 1x1 TDNNBlock; the
    // deeper concat + conv accumulates more rounding, hence the looser tol.
    std::vector<float> rm; std::vector<int64_t> sm;
    if (vdtest::load_baseline(base, "mfa_out", rm, sm))
        ok = ok && vdtest::compare(caps.mfa, rm, "mfa_out", 1e-2f, 1e-2f);

    // Attentive statistics pooling: mfa [3072, T] -> pooled [6144] = concat of the
    // attention-weighted channel mean and std (SpeechBrain AttentiveStatisticsPooling
    // with global_context=True). Gated PRE asp_bn (the gen_baseline.py hook is on
    // em.asp, whose output is the raw ASP statistics; asp_bn is a later stage).
    std::vector<float> rp; std::vector<int64_t> sp;
    if (vdtest::load_baseline(base, "pooled", rp, sp))
        ok = ok && vdtest::compare(caps.pooled, rp, "pooled", 1e-2f, 1e-2f);

    // PRIMARY voice parity gate: asp_bn -> fc (6144->192) -> L2-normalize must
    // match the golden L2-normalized embedding at cosine >= 0.9999 AND
    // max|d| <= 1e-3 (the ECAPA fp32 parity gate from the spec).
    std::vector<float> re; std::vector<int64_t> se;
    if (vdtest::load_baseline(base, "embedding", re, se)) {
        bool ok_abs = vdtest::compare(emb, re, "embedding", 1e-3f, 0.0f);
        double cs = vdtest::cosine(emb, re);
        std::fprintf(stderr, "[embedding] cosine=%.6f\n", cs);
        ok = ok && ok_abs && cs >= 0.9999;
    }

    // Quantization parity gates (Task 13). When VOICEDETECT_TEST_GGUF_Q8 /
    // VOICEDETECT_TEST_GGUF_Q4 point at q8_0 / q4_0 GGUFs of the SAME checkpoint,
    // re-embed the same audio through the quantized weights and assert the
    // embedding stays near-lossless vs the F32 reference embedding (the golden
    // "embedding" baseline). q8_0 must clear cosine >= 0.999 AND max|d| <= 5e-3;
    // q4_0 is looser (cosine >= 0.997). These isolate the selective-quant
    // allowlist: a regression (quantizing a weight the engine reads raw, or one
    // the mul_mat path mishandles) surfaces as a parity miss here.
    auto quant_gate = [&](const char* env, double cos_min, float dmax) {
        const char* qg = std::getenv(env);
        if (!qg) return true;  // env unset -> skip this gate
        std::vector<float> re_q; std::vector<int64_t> se_q;
        if (!vdtest::load_baseline(base, "embedding", re_q, se_q)) return true;
        vd::ModelLoader mlq;
        if (!mlq.load(qg)) { std::fprintf(stderr, "[%s] load failed\n", env); return false; }
        int Tq = 0;
        std::vector<float> fq = vd::FBank(mlq.config()).compute(in.samples, Tq);
        if (fq.empty()) { std::fprintf(stderr, "[%s] fbank empty\n", env); return false; }
        std::vector<float> eq = vd::EcapaEncoder(mlq).forward(fq, Tq);
        bool gok = vdtest::compare(eq, re_q, env, dmax, 0.0f);
        double cq = vdtest::cosine(eq, re_q);
        std::fprintf(stderr, "[%s] cosine=%.6f (gate >= %.4f)\n", env, cq, cos_min);
        return gok && cq >= cos_min;
    };
    ok = quant_gate("VOICEDETECT_TEST_GGUF_Q8", 0.999, 5e-3f) && ok;
    ok = quant_gate("VOICEDETECT_TEST_GGUF_Q4", 0.997, 2e-2f) && ok;

    vd::shutdown_backend();
    return ok ? 0 : 1;
}
