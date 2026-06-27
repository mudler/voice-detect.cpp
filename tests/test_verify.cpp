#include "model.hpp"
#include "backend.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

// Verify-verdict parity gate: runs Model::verify_paths over the labeled pair
// set (tests/fixtures, see pairs.json) and asserts that, at the spec default
// cosine-distance threshold 0.25, the C++ verdict AND distance match the
// SpeechBrain reference decision for every pair. Convention (model.hpp /
// SpeechBrain): distance = 1 - cosine_similarity over L2-normalized embeddings,
// verified = distance <= threshold.
//
// PARITY, NOT LABEL: the reference SpeechBrain ECAPA model itself, run on these
// exact 3 s LibriVox clips, produces ref_dist=0.0654 for the same-speaker pair
// (clip_a/clip_b, reader lcw -> verified) and ref_dist=0.2016 for the
// different-speaker pair (clip_a/clip_c, reader bk). At the default threshold
// 0.25 the reference verdict for BOTH pairs is "verified" - i.e. the reference
// itself false-accepts the different-speaker clip at 0.25 (a property of the
// short fixtures + the strict default threshold, NOT of this port). So this gate
// compares the C++ verdict/distance to the *reference decision*, which is the
// "identical verdicts" the parity test exists to protect: if the embedding path
// drifts, the distance moves off the reference value and the gate goes RED.
// (Reference values reproduced via scripts/gen_baseline.py's Kaldi-FBank ->
// embedding_model path; see tests/fixtures/README.md for the clips.)
//
// RC-77 skip convention: SKIPs (exit 77) when the model/fixtures env vars are
// unset or the GGUF fails to load, so CI without a checkpoint never breaks.
int main() {
    const char* gguf = std::getenv("VOICEDETECT_TEST_GGUF");
    const char* dir = std::getenv("VOICEDETECT_TEST_FIXDIR");  // tests/fixtures
    if (!gguf || !dir) return 77;
    auto m = vd::Model::load(gguf);
    if (!m) return 77;
    std::string d = dir;
    const float threshold = 0.25f;   // spec default (CLI / C-ABI / model.hpp)
    const float dist_tol = 2e-3f;    // distance must track the reference value
    struct P { const char* a; const char* b; bool same; float ref_dist; bool ref_verdict; };
    P pairs[] = {
        // a              b                  label   reference cosine distance + verdict @0.25
        {"clip_a.wav", "clip_b_same.wav",   true,   0.0654f, true},
        {"clip_a.wav", "clip_c_diff.wav",   false,  0.2016f, true},
    };
    bool ok = true;
    for (auto& p : pairs) {
        float dist = 0; bool verdict = false;
        m->verify_paths(d + "/" + p.a, d + "/" + p.b, threshold, dist, verdict);
        bool dist_ok = std::fabs(dist - p.ref_dist) <= dist_tol;
        bool verdict_ok = (verdict == p.ref_verdict);
        std::fprintf(stderr,
                     "%s vs %s: dist=%.4f (ref=%.4f, d=%.1e) verdict=%d ref_verdict=%d label_same=%d -> dist_ok=%d verdict_ok=%d\n",
                     p.a, p.b, dist, p.ref_dist, std::fabs(dist - p.ref_dist),
                     verdict, p.ref_verdict, p.same, dist_ok, verdict_ok);
        ok = ok && dist_ok && verdict_ok;
    }
    vd::shutdown_backend();
    return ok ? 0 : 1;
}
