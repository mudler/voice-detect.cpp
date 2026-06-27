#include "model.hpp"
#include "backend.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Multi-model graph-cache safety smoke. The lever-1 persistent graph cache lives
// on the PROCESS-GLOBAL vd::Backend and keys entries by a fingerprint that folds
// each model's WEIGHT data pointers. Risk: free a ModelLoader, let a later model
// reallocate the same address, and a stale entry could false-hit and replay a
// cached graph whose leaves point at the freed weights (UB / wrong output).
//
// This proves the invalidate-on-free hardening (~ModelLoader ->
// invalidate_graph_cache_for_weights -> Backend::invalidate_weights_buffer):
//
//   Phase 0  reference: one isolated load + embed, then shutdown_backend() to
//            drop the global backend and its whole cache.
//   Phase 1  load A, embed (populates the cache), free A; assert the cache is
//            now EMPTY (A's entries were purged on free). With no entry left,
//            there is literally nothing for a later same-address model to
//            false-hit - the airtight safety guarantee.
//   Phase 2  load B (a fresh instance of the SAME gguf; the allocator very
//            likely hands back A's just-freed weights-buffer address). Its first
//            embed MUST be a cache MISS (not a stale hit) and MUST match the
//            reference bit-for-bit.
//   Phase 3  embed the same clip on B again: MUST be a cache HIT and bit-identical
//            (same-model reload keeps the hot cache - no perf regression).
//
// RC-77 skip when the model/audio env vars are unset.

static int fail(const char* msg) { std::fprintf(stderr, "cache-safety: %s\n", msg); return 1; }

int main() {
    const char* gguf  = std::getenv("VOICEDETECT_TEST_GGUF");
    const char* audio = std::getenv("VOICEDETECT_TEST_AUDIO");
    if (!gguf || !audio) return 77;

    // --- Phase 0: isolated reference embedding. -------------------------------
    std::vector<float> ref;
    {
        auto m = vd::Model::load(gguf);
        if (!m) return 77;
        ref = m->embed_path(audio);
    }
    vd::shutdown_backend();   // fresh global backend + empty cache for the test
    if (ref.empty()) return fail("reference embedding empty");

    // --- Phase 1: A loaded, embedded, freed -> cache must be purged. -----------
    {
        auto A = vd::Model::load(gguf);
        if (!A) return 77;
        (void)A->embed_path(audio);
        if (vd::global_backend().cache_size() == 0)
            return fail("expected A to populate the graph cache");
        A.reset();   // ~Model -> ~ModelLoader frees weights -> invalidate cache
        if (vd::global_backend().cache_size() != 0)
            return fail("cache NOT invalidated on model free (stale-hit risk)");
    }

    // --- Phase 2: B at the (likely) reused address must MISS and be correct. ---
    const uint64_t miss0 = vd::global_backend().cache_misses();
    auto B = vd::Model::load(gguf);
    if (!B) return 77;
    std::vector<float> eB = B->embed_path(audio);
    if (vd::global_backend().cache_misses() <= miss0)
        return fail("B's first embed was not a MISS (possible stale hit)");
    if (eB.size() != ref.size()) return fail("B embedding size mismatch");
    double dmax = 0.0;
    for (size_t i = 0; i < ref.size(); ++i)
        dmax = std::max(dmax, (double)std::fabs(eB[i] - ref[i]));
    if (dmax > 1e-6) {
        std::fprintf(stderr, "cache-safety: B diverged from reference, max|d|=%g\n", dmax);
        return 1;
    }

    // --- Phase 3: same-model reload still HITS (and is bit-identical). ---------
    const uint64_t hits0 = vd::global_backend().cache_hits();
    std::vector<float> eB2 = B->embed_path(audio);
    if (vd::global_backend().cache_hits() <= hits0)
        return fail("same-model reload did not HIT the cache (perf regression)");
    if (eB2.size() != eB.size()) return fail("HIT embedding size mismatch");
    for (size_t i = 0; i < eB.size(); ++i)
        if (eB2[i] != eB[i]) return fail("cache HIT output not bit-identical");

    vd::shutdown_backend();
    std::printf("cache-safety: PASS (purge-on-free; B MISS+correct; reload HIT bit-identical)\n");
    return 0;
}
