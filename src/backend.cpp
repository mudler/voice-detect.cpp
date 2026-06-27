#include "backend.hpp"
#include "common.hpp"
#include "model_loader.hpp"
#include "winograd.hpp"
#include "directconv.hpp"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace vd {

namespace {
// Number of graph nodes the metadata context must hold. The speaker encoder is
// the biggest single graph; this budget is generous for ECAPA/ResNet-class nets.
constexpr size_t kGraphSize = 16384;

struct PendingInput {
    ggml_tensor* tensor;
    const void*  host;
    size_t       nbytes;
};
struct PendingCapture {
    ggml_tensor*        tensor;
    std::vector<float>* dst;
};

// Structural fingerprint of a freshly-built graph. Two graphs that hash equal
// compute the identical function (same node ops, types, shapes, input/capture
// counts, output shape) and so are interchangeable for compute - this is the
// per-shape cache key. Built from the graph metadata only (cheap), it both keys
// the cache and guards against two DIFFERENT call sites (e.g. w2v2 encode vs a
// speaker encoder) colliding on the same audio length T.
uint64_t fingerprint_graph(ggml_cgraph* gf, ggml_tensor* output,
                           size_t n_inputs, size_t n_caps) {
    uint64_t h = 1469598103934665603ull;            // FNV-1a offset basis
    auto mix = [&](uint64_t v) { h ^= v; h *= 1099511628211ull; };
    mix((uint64_t)ggml_graph_n_nodes(gf));
    mix((uint64_t)n_inputs);
    mix((uint64_t)n_caps);
    if (output) {
        mix((uint64_t)output->type);
        for (int d = 0; d < GGML_MAX_DIMS; ++d) mix((uint64_t)output->ne[d]);
    }
    const int nn = ggml_graph_n_nodes(gf);
    for (int i = 0; i < nn; ++i) {
        ggml_tensor* t = ggml_graph_node(gf, i);
        mix((uint64_t)t->op);
        mix((uint64_t)t->type);
        for (int d = 0; d < GGML_MAX_DIMS; ++d) mix((uint64_t)t->ne[d]);
        // Fold in the op params. For GGML_OP_CUSTOM nodes these hold the compute
        // callback pointer + userdata (the DConvState carrying kind/pad/stride),
        // which op/type/ne alone do NOT distinguish: a blocked 3x3 and a blocked
        // 1x1 strided conv can build structurally identical graphs (same node
        // count, shapes, input/capture counts, output shape) while running
        // completely different kernels. Without this the per-shape cache
        // false-hits and replays the wrong custom op (e.g. conv1x1 replaying a
        // cached conv3x3), producing wrong output and out-of-bounds writes.
        // op_params is zero-initialised at tensor creation, so the unused tail is
        // deterministic across builds.
        for (int p = 0; p < (int)(GGML_MAX_OP_PARAMS / sizeof(int32_t)); ++p)
            mix((uint64_t)(uint32_t)t->op_params[p]);
        // Hash each source's data pointer. In the no_alloc build context only the
        // realized WEIGHT leaves carry a non-null ->data (their addresses are owned
        // by the ModelLoader and stable per model); inputs/intermediates are null.
        // Folding these in binds the fingerprint to THIS model's weight buffers, so
        // two distinct models that share an architecture + audio length T cannot
        // false-hit the same cached graph.
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            ggml_tensor* src = t->src[s];
            mix(src ? (uint64_t)(uintptr_t)src->data : 0ull);
        }
    }
    return h;
}

// Collect the distinct backend buffers backing this graph's WEIGHT leaves. Must
// be called on the freshly-built graph BEFORE gallocr allocation: at that point
// only realized weight leaves carry a non-null ->buffer (it points at the owning
// ModelLoader's weights buffer), while inputs/intermediates are still unallocated
// (null buffer). Recording these per cache entry is what lets
// invalidate_weights_buffer() purge an entry the moment its model is freed.
std::vector<ggml_backend_buffer_t> collect_weight_buffers(ggml_cgraph* gf) {
    std::vector<ggml_backend_buffer_t> bufs;
    const int nn = ggml_graph_n_nodes(gf);
    for (int i = 0; i < nn; ++i) {
        ggml_tensor* t = ggml_graph_node(gf, i);
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            ggml_tensor* src = t->src[s];
            if (!src || !src->buffer) continue;
            if (std::find(bufs.begin(), bufs.end(), src->buffer) == bufs.end())
                bufs.push_back(src->buffer);
        }
    }
    return bufs;
}
} // namespace

struct Backend::Impl {
    ggml_backend_t       backend     = nullptr;  // primary device (GPU or CPU)
    ggml_backend_t       cpu_backend = nullptr;  // fallback backend (GPU path only)
    ggml_backend_sched_t sched       = nullptr;  // GPU path: schedules over {backend, cpu_backend}
    bool                 use_sched   = false;

    // Persistent per-shape graph cache (parakeet.cpp pattern). Each entry owns its
    // ggml_context (graph node STRUCTURE + metadata), the built ggml_cgraph, and
    // its OWN gallocr (a dedicated device buffer, so entries never alias each
    // other's tensor storage). The graph object AND its tensor data pointers stay
    // STABLE across calls, so on CUDA the captured CUDA graph replays through
    // cudaGraphExecUpdate instead of churning destroy + reinstantiate every call.
    struct GraphEntry {
        uint64_t                  fp     = 0;
        ggml_context*             ctx    = nullptr;
        ggml_cgraph*              gf     = nullptr;
        ggml_gallocr_t            galloc = nullptr;
        ggml_tensor*              output = nullptr;
        std::vector<ggml_tensor*> inputs;   // stable input tensors (build order)
        std::vector<ggml_tensor*> caps;     // stable capture tensors (build order)
        // Distinct backend buffers backing this graph's WEIGHT leaves (each is a
        // ModelLoader's weights buffer). invalidate_weights_buffer() drops the
        // entry the instant any of these is freed, so a freed-then-reused address
        // can never alias into a stale replay (multi-model hosting safety).
        std::vector<ggml_backend_buffer_t> weight_bufs;
        uint64_t                  lru    = 0;
    };
    std::vector<GraphEntry> cache;
    uint64_t                lru_clock = 0;
    uint64_t                hits      = 0;   // cache hit / miss counters (test introspection)
    uint64_t                misses    = 0;
    size_t                  cache_cap = 8;   // 0 == caching disabled (per-call)

    // Scratch context reused for the per-call graph build. The build lambda runs
    // exactly ONCE per compute() (the conv/bn helpers populate caller-owned scratch
    // vectors mid-build, so a second build pass would double them) - we always
    // build here to harvest the fresh input host pointers and the structural
    // fingerprint. On a cache MISS this scratch is PROMOTED to a cache entry and a
    // fresh scratch is allocated next call; on a HIT it is ggml_reset and reused.
    ggml_context*           scratch = nullptr;

    std::vector<PendingInput>   pending;
    std::vector<PendingCapture> captures;
};

// Thread-local pointer to the Backend whose compute() build lambda is currently
// executing, so add_graph_input() can route registrations without threading the
// Backend through every build lambda. compute is not re-entrant on a thread.
static thread_local Backend* t_active = nullptr;

// Backend-agnostic helpers used in place of the CPU-backend-specific entry
// points (ggml_backend_cpu_init / _is_cpu / _set_n_threads). Those live in the
// ggml-cpu translation unit, which is a dlopen'd MODULE in a multi-variant
// (GGML_BACKEND_DL) build and therefore NOT linkable from here. The generic
// device/registry API below works identically in both the single static build
// and the runtime-dispatch build.

// Discover and register every available backend module once. In a
// GGML_BACKEND_DL build this loads the per-micro-arch CPU variants
// (libggml-cpu-<tag>.so) and registers the best one the host supports, giving
// runtime AVX512 selection. In a statically-linked build the CPU backend is
// already registered and this is a harmless no-op.
static void ensure_backends_loaded() {
    static std::once_flag once;
    std::call_once(once, [] { ggml_backend_load_all(); });
}

static bool backend_is_cpu(ggml_backend_t b) {
    if (!b) return false;
    ggml_backend_dev_t dev = ggml_backend_get_device(b);
    return dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
}

// Set the thread count via the backend registry's exported proc address. The
// CPU backend (any variant) advertises "ggml_backend_set_n_threads"; non-CPU
// backends return null and are left untouched.
static void backend_set_n_threads(ggml_backend_t b, int n_threads) {
    if (!b) return;
    ggml_backend_dev_t dev = ggml_backend_get_device(b);
    if (!dev) return;
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (!reg) return;
    auto set_fn = (ggml_backend_set_n_threads_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (set_fn) set_fn(b, n_threads);
}

// Initialise a CPU backend through the registry (picks the best loaded variant).
static ggml_backend_t backend_cpu_init() {
    return ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
}

Backend::Backend(int n_threads) : impl_(new Impl()) {
    ensure_backends_loaded();

    // Persistent-graph cache capacity (number of distinct input shapes / call
    // sites kept hot). 0 disables caching (one-shot per-call build, the legacy
    // behaviour) as an A/B kill-switch.
    if (const char* env = std::getenv("VOICEDETECT_GRAPH_CACHE")) {
        const long v = std::atol(env);
        impl_->cache_cap = v >= 0 ? (size_t)v : impl_->cache_cap;
    }

    // Optional override via VOICEDETECT_DEVICE:
    //   - "cpu"           forces the CPU backend.
    //   - a device name   selects that registry device by name (case-insensitive).
    //   - unset           auto-pick the first GPU / integrated-GPU device.
    const char* force = std::getenv("VOICEDETECT_DEVICE");
    const std::string want = force ? force : "";
    const bool force_cpu = want == "cpu" || want == "CPU";

    auto iequals = [](const std::string& a, const std::string& b) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i)
            if (std::tolower((unsigned char)a[i]) != std::tolower((unsigned char)b[i]))
                return false;
        return true;
    };

    if (!force_cpu) {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            const auto type = ggml_backend_dev_type(dev);
            const char* name = ggml_backend_dev_name(dev);

            bool selected;
            if (!want.empty()) {
                selected = name && iequals(want, name);
            } else {
                selected = type == GGML_BACKEND_DEVICE_TYPE_GPU ||
                           type == GGML_BACKEND_DEVICE_TYPE_IGPU;
            }
            if (!selected) continue;

            impl_->backend = ggml_backend_dev_init(dev, nullptr);
            if (impl_->backend) {
                device_name_ = name ? name : "";
                impl_->use_sched = type != GGML_BACKEND_DEVICE_TYPE_CPU;
                VD_LOG("vd::Backend using device: %s", device_name_.c_str());
                break;
            }
        }
        if (!want.empty() && !impl_->backend)
            VD_LOG("vd::Backend: VOICEDETECT_DEVICE=%s not found; falling back to CPU",
                   want.c_str());
    }
    if (!impl_->backend) {              // CPU fallback (or CPU-only build)
        impl_->backend = backend_cpu_init();
        device_name_ = "cpu";
    }
    if (!impl_->backend) {
        VD_LOG("backend init returned null");
        return;
    }
    if (impl_->use_sched) {
        impl_->cpu_backend = backend_cpu_init();
        if (!impl_->cpu_backend) {
            VD_LOG("vd::Backend: CPU fallback init failed; disabling sched");
            impl_->use_sched = false;
        }
    }
    set_n_threads(n_threads);
}

Backend::~Backend() {
    if (impl_) {
        for (Impl::GraphEntry& e : impl_->cache) {
            if (e.galloc) ggml_gallocr_free(e.galloc);
            if (e.ctx)    ggml_free(e.ctx);
        }
        impl_->cache.clear();
        if (impl_->scratch)     ggml_free(impl_->scratch);
        if (impl_->sched)       ggml_backend_sched_free(impl_->sched);
        if (impl_->cpu_backend) ggml_backend_free(impl_->cpu_backend);
        if (impl_->backend)     ggml_backend_free(impl_->backend);
        delete impl_;
        impl_ = nullptr;
    }
}

void Backend::set_n_threads(int n_threads) {
    n_threads_ = n_threads > 0 ? n_threads : 1;
    if (impl_ && impl_->backend && backend_is_cpu(impl_->backend)) {
        backend_set_n_threads(impl_->backend, n_threads_);
    }
    if (impl_ && impl_->cpu_backend) {
        backend_set_n_threads(impl_->cpu_backend, n_threads_);
    }
}

ggml_backend_t Backend::handle() const {
    return impl_ ? impl_->backend : nullptr;
}

void Backend::invalidate_weights_buffer(ggml_backend_buffer_t buf) {
    if (!impl_ || !buf) return;
    for (size_t i = 0; i < impl_->cache.size();) {
        Impl::GraphEntry& e = impl_->cache[i];
        const bool refs = std::find(e.weight_bufs.begin(), e.weight_bufs.end(), buf)
                          != e.weight_bufs.end();
        if (refs) {
            if (e.galloc) ggml_gallocr_free(e.galloc);
            if (e.ctx)    ggml_free(e.ctx);
            impl_->cache.erase(impl_->cache.begin() + (long)i);
        } else {
            ++i;
        }
    }
}

size_t   Backend::cache_size()   const { return impl_ ? impl_->cache.size() : 0; }
uint64_t Backend::cache_hits()   const { return impl_ ? impl_->hits : 0; }
uint64_t Backend::cache_misses() const { return impl_ ? impl_->misses : 0; }

void Backend::register_input(ggml_tensor* t, const void* host, size_t nbytes) {
    impl_->pending.push_back({t, host, nbytes});
}

void Backend::register_capture(ggml_tensor* t, std::vector<float>* dst) {
    impl_->captures.push_back({t, dst});
}

bool Backend::compute(const std::function<ggml_tensor*(ggml_context*)>& build,
                      std::vector<float>& out) {
    if (!impl_ || !impl_->backend) {
        VD_LOG("Backend::compute called on an uninitialised backend");
        return false;
    }

    const struct ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() * kGraphSize +
                            ggml_graph_overhead_custom(kGraphSize, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };

    // --- Build the graph ONCE per call into the scratch context. ----------------
    // This harvests the fresh input host pointers (the audio differs every call)
    // and the structural fingerprint. ggml_reset reuses the scratch buffer on a
    // hit (no malloc churn); a fresh ctx is allocated only after a miss promoted
    // the previous scratch into a cache entry.
    if (!impl_->scratch) {
        impl_->scratch = ggml_init(params);
        if (!impl_->scratch) {
            VD_LOG("Backend::compute: ggml_init failed");
            return false;
        }
    } else {
        ggml_reset(impl_->scratch);
    }
    ggml_context* ctx = impl_->scratch;

    impl_->pending.clear();
    impl_->captures.clear();
    Backend* prev_active = t_active;
    t_active = this;
    struct ggml_tensor* output = build(ctx);
    t_active = prev_active;

    if (!output) {
        VD_LOG("Backend::compute: build() returned null output tensor");
        impl_->pending.clear();
        impl_->captures.clear();
        return false;
    }
    ggml_set_output(output);
    for (const PendingCapture& pc : impl_->captures) ggml_set_output(pc.tensor);

    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx, kGraphSize, false);
    for (const PendingCapture& pc : impl_->captures)
        ggml_build_forward_expand(gf, pc.tensor);
    ggml_build_forward_expand(gf, output);

    bool need_sched = false;
    if (impl_->use_sched) {
        const int n_nodes = ggml_graph_n_nodes(gf);
        for (int i = 0; i < n_nodes; ++i) {
            if (!ggml_backend_supports_op(impl_->backend, ggml_graph_node(gf, i))) {
                need_sched = true;
                break;
            }
        }
    }

    // === Sched path: a node is unsupported on the primary device, so the graph
    // is scheduled across {primary, cpu}. Kept per-call (no persistent caching)
    // exactly as before; the scratch context is reused (reset) next call. =========
    if (need_sched) {
        if (!impl_->sched) {
            ggml_backend_t backs[2] = { impl_->backend, impl_->cpu_backend };
            impl_->sched = ggml_backend_sched_new(
                backs, /*bufts=*/nullptr, /*n_backends=*/2,
                /*graph_size=*/kGraphSize, /*parallel=*/false, /*op_offload=*/true);
            if (!impl_->sched) {
                VD_LOG("Backend::compute: ggml_backend_sched_new failed");
                impl_->pending.clear();
                impl_->captures.clear();
                return false;
            }
        }
        ggml_backend_sched_reset(impl_->sched);
        if (!ggml_backend_sched_alloc_graph(impl_->sched, gf)) {
            VD_LOG("Backend::compute: ggml_backend_sched_alloc_graph failed");
            impl_->pending.clear();
            impl_->captures.clear();
            return false;
        }
        for (const PendingInput& pi : impl_->pending)
            ggml_backend_tensor_set(pi.tensor, pi.host, 0, pi.nbytes);
        impl_->pending.clear();

        enum ggml_status status = ggml_backend_sched_graph_compute(impl_->sched, gf);
        if (status != GGML_STATUS_SUCCESS) {
            VD_LOG("Backend::compute: sched compute failed (status=%d)", (int)status);
            impl_->captures.clear();
            return false;
        }
        for (const PendingCapture& pc : impl_->captures) {
            size_t cn = (size_t)ggml_nelements(pc.tensor);
            pc.dst->resize(cn);
            ggml_backend_tensor_get(pc.tensor, pc.dst->data(), 0, cn * sizeof(float));
        }
        impl_->captures.clear();
        size_t n = (size_t)ggml_nelements(output);
        out.resize(n);
        ggml_backend_tensor_get(output, out.data(), 0, n * sizeof(float));
        return true;
    }

    // === gallocr path: persistent per-shape cached graph. =======================
    const uint64_t fp = fingerprint_graph(gf, output,
                                          impl_->pending.size(), impl_->captures.size());

    Impl::GraphEntry* hit = nullptr;
    for (Impl::GraphEntry& e : impl_->cache) {
        if (e.fp == fp) { hit = &e; break; }
    }

    if (hit) {
        ++impl_->hits;
        // Re-set ONLY the input tensor DATA into the cached (stable-address) input
        // tensors, then replay the cached graph. The cgraph object and every node
        // data pointer are unchanged, so the CUDA graph updates in place
        // (cudaGraphExecUpdate) and replays - no destroy + reinstantiate.
        GGML_ASSERT(hit->inputs.size() == impl_->pending.size() &&
                    "graph cache hit: input count mismatch");
        for (size_t i = 0; i < impl_->pending.size(); ++i) {
            ggml_backend_tensor_set(hit->inputs[i], impl_->pending[i].host, 0,
                                    impl_->pending[i].nbytes);
        }
        impl_->pending.clear();

        enum ggml_status status = ggml_backend_graph_compute(impl_->backend, hit->gf);
        if (status != GGML_STATUS_SUCCESS) {
            VD_LOG("Backend::compute: cached graph_compute failed (status=%d)",
                   (int)status);
            impl_->captures.clear();
            return false;
        }

        GGML_ASSERT(hit->caps.size() == impl_->captures.size() &&
                    "graph cache hit: capture count mismatch");
        for (size_t i = 0; i < impl_->captures.size(); ++i) {
            ggml_tensor* src = hit->caps[i];
            size_t cn = (size_t)ggml_nelements(src);
            impl_->captures[i].dst->resize(cn);
            ggml_backend_tensor_get(src, impl_->captures[i].dst->data(), 0,
                                    cn * sizeof(float));
        }
        impl_->captures.clear();

        size_t n = (size_t)ggml_nelements(hit->output);
        out.resize(n);
        ggml_backend_tensor_get(hit->output, out.data(), 0, n * sizeof(float));
        hit->lru = ++impl_->lru_clock;
        return true;
    }

    // --- MISS: allocate a dedicated gallocr, run, and (unless caching is disabled)
    // promote this scratch graph into a new cache entry. ------------------------
    ++impl_->misses;
    // Snapshot the weight buffers NOW (pre-alloc): only weight leaves are buffered
    // at this point, so this captures exactly the ModelLoader weights this graph
    // depends on (inputs/intermediates get their buffer from gallocr below).
    std::vector<ggml_backend_buffer_t> weight_bufs = collect_weight_buffers(gf);
    ggml_gallocr_t galloc =
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(impl_->backend));
    if (!galloc) {
        VD_LOG("Backend::compute: ggml_gallocr_new failed");
        impl_->pending.clear();
        impl_->captures.clear();
        return false;
    }
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        VD_LOG("Backend::compute: ggml_gallocr_alloc_graph failed");
        ggml_gallocr_free(galloc);
        impl_->pending.clear();
        impl_->captures.clear();
        return false;
    }

    for (const PendingInput& pi : impl_->pending) {
        ggml_backend_tensor_set(pi.tensor, pi.host, 0, pi.nbytes);
    }

    enum ggml_status status = ggml_backend_graph_compute(impl_->backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        VD_LOG("Backend::compute: ggml_backend_graph_compute failed (status=%d)",
               (int)status);
        ggml_gallocr_free(galloc);
        impl_->pending.clear();
        impl_->captures.clear();
        return false;
    }

    for (const PendingCapture& pc : impl_->captures) {
        size_t cn = (size_t)ggml_nelements(pc.tensor);
        pc.dst->resize(cn);
        ggml_backend_tensor_get(pc.tensor, pc.dst->data(), 0, cn * sizeof(float));
    }

    size_t n = (size_t)ggml_nelements(output);
    out.resize(n);
    ggml_backend_tensor_get(output, out.data(), 0, n * sizeof(float));

    if (impl_->cache_cap == 0) {
        // Caching disabled: free this graph's resources, mirroring the legacy
        // per-call build/free. The scratch ctx is reused (reset) next call.
        ggml_gallocr_free(galloc);
        impl_->pending.clear();
        impl_->captures.clear();
        return true;
    }

    Impl::GraphEntry e;
    e.fp     = fp;
    e.ctx    = impl_->scratch;   // the scratch ctx now OWNS this graph
    e.gf     = gf;
    e.galloc = galloc;
    e.output = output;
    e.inputs.reserve(impl_->pending.size());
    for (const PendingInput& pi : impl_->pending) e.inputs.push_back(pi.tensor);
    e.caps.reserve(impl_->captures.size());
    for (const PendingCapture& pc : impl_->captures) e.caps.push_back(pc.tensor);
    e.weight_bufs = std::move(weight_bufs);
    e.lru    = ++impl_->lru_clock;
    impl_->pending.clear();
    impl_->captures.clear();

    // The scratch ctx is now owned by the cache entry; force a fresh one next call.
    impl_->scratch = nullptr;
    impl_->cache.push_back(std::move(e));

    // Evict the least-recently-used entries beyond the capacity cap. The entry we
    // just appended carries the highest lru, so it is never the victim.
    while (impl_->cache.size() > impl_->cache_cap) {
        size_t victim = 0;
        for (size_t i = 1; i < impl_->cache.size(); ++i)
            if (impl_->cache[i].lru < impl_->cache[victim].lru) victim = i;
        Impl::GraphEntry& v = impl_->cache[victim];
        if (v.galloc) ggml_gallocr_free(v.galloc);
        if (v.ctx)    ggml_free(v.ctx);
        impl_->cache.erase(impl_->cache.begin() + (long)victim);
    }
    return true;
}

void add_graph_input(ggml_tensor* t, const void* host, size_t nbytes) {
    GGML_ASSERT(t_active != nullptr &&
                "add_graph_input called outside a Backend::compute build lambda");
    ggml_set_input(t);
    t_active->register_input(t, host, nbytes);
}

ggml_tensor* graph_input_tensor(ggml_context* ctx, int type, int n_dims,
                                const int64_t* ne, const void* host,
                                size_t nbytes) {
    ggml_tensor* t = ggml_new_tensor(ctx, (ggml_type)type, n_dims, ne);
    add_graph_input(t, host, nbytes);
    return t;
}

void capture_graph_output(ggml_tensor* t, std::vector<float>* dst) {
    GGML_ASSERT(t_active != nullptr &&
                "capture_graph_output called outside a Backend::compute build lambda");
    t_active->register_capture(t, dst);
}

// ---------------------------------------------------------------------------
// Process-global backend + thread-count override.
// ---------------------------------------------------------------------------
static Backend* g_backend = nullptr;
static int      g_num_threads = 0;  // 0 == unset

void set_num_threads(int n) { g_num_threads = n > 0 ? n : 0; }
int  num_threads()          { return g_num_threads; }

// Default CPU thread count for the process-global backend when no explicit
// override is given. The previous default of 1 left voice-detect running
// single-threaded inside a LocalAI host - a real perf loss, since these
// conv/matmul forwards are compute-bound and scale near-linearly to a handful
// of threads. Cap at 8: beyond that there is no gain for these graphs and a
// cross-CCD/SMT regression risk. VOICEDETECT_THREADS lets a multi-tenant host
// tune the cap to its box, and vd::set_num_threads()/CLI --threads still win.
static int default_n_threads() {
    if (const char* env = std::getenv("VOICEDETECT_THREADS")) {
        const int v = std::atoi(env);
        if (v > 0) return v;
    }
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 4;
    return (int)std::min(hw, 8u);
}

Backend& global_backend() {
    if (!g_backend) {
        int nt = g_num_threads > 0 ? g_num_threads : default_n_threads();
        g_backend = new Backend(nt);
    }
    return *g_backend;
}

void shutdown_backend() {
    delete g_backend;
    g_backend = nullptr;
}

void invalidate_graph_cache_for_weights(ggml_backend_buffer_t buf) {
    // Do NOT lazily create the global backend here: a ModelLoader destroyed after
    // shutdown_backend() (or before any compute) has nothing to invalidate.
    if (g_backend) g_backend->invalidate_weights_buffer(buf);
}

// Per-encoder opt-in flag for Winograd routing (see WinogradScope). thread_local
// because the routing decision is taken at graph-BUILD time on the calling
// thread; the custom op then runs on the backend worker threads.
namespace {
thread_local bool g_winograd_route = false;
// Per-encoder choice of the optimized 3x3 kernel (default Direct). Read at build time.
thread_local Conv3x3Kernel g_conv3x3_kernel = Conv3x3Kernel::Direct;
}

static bool winograd_routing_enabled() { return g_winograd_route; }
static Conv3x3Kernel conv3x3_kernel() { return g_conv3x3_kernel; }

WinogradScope::WinogradScope(bool on, Conv3x3Kernel kernel)
    : prev_(g_winograd_route), prev_kernel_(g_conv3x3_kernel) {
    g_winograd_route = on;
    g_conv3x3_kernel = kernel;
}
WinogradScope::~WinogradScope() {
    g_winograd_route = prev_;
    g_conv3x3_kernel = prev_kernel_;
}

ggml_tensor* conv2d_auto(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x,
                         int s0, int s1, int p0, int p1, int d0, int d1) {
    // 1x1 stride-1 fast path: a 1x1 conv is a pure per-pixel GEMM over the input
    // channels - out[w,h,oc] = sum_ic x[w,h,ic] * W[ic,oc]. ggml_conv_2d would run
    // im2col (which, for a 1x1 stride-1 conv, only TRANSPOSES x from [W,H,IC,N] to
    // [IC, W*H, N]) and then a mul_mat. We skip the generic im2col op entirely and
    // feed the transposed activations straight to ggml_mul_mat: identical math (the
    // same f32 reduction over IC), so this is parity-EXACT, while dropping a graph
    // node and the im2col kernel's per-call overhead. ~23% of ERes2Net compute is
    // these 1x1 im2col copies (also hit by WeSpeaker's 1x1 downsample shortcut and
    // CAM++ FCM's 1x1 shortcut), so the hook stays shared.
    //
    // Guards: only the genuinely-1x1, stride-1, unpadded, undilated, single-batch,
    // f32-weight case. Strided 1x1 downsamples are NOT routed here - their im2col
    // already performs the optimal strided gather, and a plain mul_mat over the full
    // grid would silently drop the subsampling (wrong result); they stay on
    // ggml_conv_2d below. N>1 falls through too (mul_mat batches via src1, but here
    // the weight is src0; all shipped speaker encoders run N=1).
    if (w->ne[0] == 1 && w->ne[1] == 1 && s0 == 1 && s1 == 1 &&
        p0 == 0 && p1 == 0 && d0 == 1 && d1 == 1 &&
        x->ne[3] == 1 && w->type == GGML_TYPE_F32) {
        const int64_t WH = x->ne[0] * x->ne[1];   // pixels per channel
        const int64_t IC = x->ne[2];
        const int64_t OC = w->ne[3];
        // x [W,H,IC,1] -> [W*H, IC] -> transpose -> packed [IC, W*H].
        ggml_tensor* x2 = ggml_reshape_2d(ctx, x, WH, IC);
        ggml_tensor* xt = ggml_cont(ctx, ggml_transpose(ctx, x2));   // [IC, W*H]
        ggml_tensor* w2 = ggml_reshape_2d(ctx, w, IC, OC);           // [IC, OC]
        // mul_mat(src0=xt[IC,W*H], src1=w2[IC,OC]) -> [W*H, OC]; reduction over IC
        // matches ggml_conv_2d's im2col+mul_mat exactly.
        ggml_tensor* y = ggml_mul_mat(ctx, xt, w2);                  // [W*H, OC]
        return ggml_reshape_4d(ctx, y, x->ne[0], x->ne[1], OC, 1);   // [W,H,OC,1]
    }

#ifdef VOICEDETECT_GGML_CUDNN
    // CUDA + cuDNN: emit GGML_OP_CONV_2D (ggml_conv_2d_direct) instead of
    // ggml_conv_2d's im2col + cuBLAS/cutlass SGEMM. ggml-cuda routes that op to
    // a cuDNN implicit-GEMM (cudnn-conv.cu): the convolution is streamed from the
    // NCHW activations with NO im2col global-memory spill (that one kernel was
    // ~62% of WeSpeaker's GPU time, nsys). ggml's conv tensors are already in
    // cuDNN NCHW order, so consecutive convs stay NCHW with zero transpose tax.
    // Covers the all-3x3 backbone (stride 1 and the stride-2 stage entries) plus
    // the 1x1 stride-2 downsample shortcuts that fell through to im2col above.
    // A/B: VD_CONV2D=im2col forces the legacy path; VD_CONV2D=cudnn is the
    // explicit opt-in (default when built with cuDNN). The kernel-level kill
    // switch GGML_CUDA_USE_CUDNN=0 makes ggml-cuda fall back to the native conv.
    {
        bool use_cudnn = !backend_is_cpu(global_backend().handle());
        if (const char* mode = std::getenv("VD_CONV2D")) {
            if      (!std::strcmp(mode, "im2col")) use_cudnn = false;
            else if (!std::strcmp(mode, "cudnn"))  use_cudnn = use_cudnn;
        }
        if (use_cudnn)
            return ggml_conv_2d_direct(ctx, w, x, s0, s1, p0, p1, d0, d1);
    }
#endif

    // AVX2 Winograd F(2x2,3x3) for the large 3x3 stride-1 pad-1 convs of the 2D
    // speaker encoders: WeSpeaker ResNet34 (an all-3x3 backbone over [T, n_mels]
    // maps - our heaviest encoder) and ERes2Net's remaining 3x3 stride-1 convs
    // (its 1x1s already take the mul_mat fast path above). Winograd does ~2.25x
    // fewer multiplies than im2col+sgemm; the custom CPU op (winograd.cpp) is the
    // same kernel that beat im2col+tinyBLAS on face-detect.cpp's SCRFD 3x3 maps.
    //
    // CPU-only: on CUDA the im2col + mul_mat (cuBLAS-class) path wins, so the GPU
    // graph keeps ggml_conv_2d (mirrors face-detect.cpp's device-aware gate). The
    // bias add + ReLU are applied by the caller (conv2d_bias), unchanged - Winograd
    // emits neither, exactly like ggml_conv_2d here.
    //
    // Guards: CPU backend, genuinely 3x3 (KW==KH==3), full (non-grouped, IC==x IC),
    // stride 1, symmetric pad, undilated, single batch, f32 weight. The winograd
    // op handles arbitrary pad; speaker-encoder 3x3 convs are always pad 1 (same
    // size). Winograd's f32 winograd-domain accumulation order differs from the
    // im2col GEMM, so values are within fp32 tolerance, not bitwise identical - the
    // encoder parity gate (cosine >= 0.9999) covers this.
    // Routed ONLY for the encoders that opt in via WinogradScope (WeSpeaker
    // ResNet34, ERes2Net) so the whole all-3x3 backbone is covered - not just the
    // large stem - while non-opted callers (e.g. CAM++'s 2D FCM, a parity control)
    // stay bit-identical on im2col. A/B override via VOICEDETECT_WINOGRAD: "on"
    // (force all eligible CPU convs, ignoring the scope) | "off" (force disable).
    bool wino_route = winograd_routing_enabled();
    if (const char* wmode = std::getenv("VOICEDETECT_WINOGRAD")) {
        if      (!std::strcmp(wmode, "off")) wino_route = false;
        else if (!std::strcmp(wmode, "on"))  wino_route = true;
    }
    const bool wino = wino_route &&
        backend_is_cpu(global_backend().handle()) &&
        w->ne[0] == 3 && w->ne[1] == 3 && w->ne[2] == x->ne[2] &&
        s0 == 1 && s1 == 1 && p0 == p1 && d0 == 1 && d1 == 1 &&
        x->ne[3] == 1 && w->type == GGML_TYPE_F32;
    // Hand-tuned MLAS-class blocked DIRECT conv (nChw16c, AVX-512 register-tiled),
    // the DEFAULT for the WinogradScope-opted 3x3-s1 pad>=1 convs. The GEMM-free
    // direct kernel pays no input/output transform, only the per-conv blocked-output
    // reorder (the SPIKE tax), and beat AVX-512 Winograd on WeSpeaker (1.29x@1t,
    // 1.55x@8t) and ERes2Net at cosine 1.0. Each encoder pins its measured-best
    // kernel via WinogradScope's 2nd arg (default Direct); Winograd stays reachable
    // for models where it wins. Same CPU-only / f32 / single-batch guards as
    // Winograd; pad>=1 required (the direct kernel's interior/edge column split
    // assumes it - sub-1 pad falls back to Winograd). Env A/B overrides ignore the
    // per-model pin: VD_CONV2D=directblocked forces Direct, VD_CONV2D=winograd forces
    // Winograd. Parity is within fp32 tolerance (different FMA order), covered by the
    // encoder cosine >= 0.999 gate.
    Conv3x3Kernel kernel = conv3x3_kernel();
    if (const char* mode = std::getenv("VD_CONV2D")) {
        if      (!std::strcmp(mode, "directblocked")) kernel = Conv3x3Kernel::Direct;
        else if (!std::strcmp(mode, "winograd"))      kernel = Conv3x3Kernel::Winograd;
    }
    if (wino && kernel == Conv3x3Kernel::Direct && p0 >= 1)
        return directconv_conv3x3(ctx, w, x, p0);

    if (wino)
        return winograd_conv3x3(ctx, w, x, p0);

    // MEASURED on this CPU (AMD Ryzen 9 9950X3D, GGML_LLAMAFILE on): for the
    // speaker-encoder 2D convs ggml_conv_2d_direct is ~10-15% SLOWER than
    // im2col + llamafile-sgemm (WeSpeaker 227->261, ERes2Net 280->309, CAM++
    // 108->110 ms/clip). The feature maps here are small (~[T,80] x <=256 ch)
    // and tinyBLAS's blocked GEMM beats ggml's basic direct-conv CPU kernel - the
    // opposite of depth-anything's large DPT head. So the default stays im2col on
    // BOTH CPU and GPU. VD_CONV2D=direct keeps the A/B path for other shapes/HW.
    bool direct = false;
    if (const char* mode = std::getenv("VD_CONV2D")) {
        if      (!std::strcmp(mode, "direct")) direct = (w->ne[0] > 1 || w->ne[1] > 1);
        else if (!std::strcmp(mode, "im2col")) direct = false;
    }
    if (direct)
        return ggml_conv_2d_direct(ctx, w, x, s0, s1, p0, p1, d0, d1);
    return ggml_conv_2d(ctx, w, x, s0, s1, p0, p1, d0, d1);
}

void ensure_weights_realized(const ModelLoader& ml) {
    if (ml.weights_realized()) return;
    ModelLoader& mut = const_cast<ModelLoader&>(ml);
    mut.realize_weights(global_backend().handle());
}

ggml_tensor* clone_weight(ggml_context* /*ctx*/, const ModelLoader& ml,
                          const char* name) {
    ensure_weights_realized(ml);
    ggml_tensor* src = ml.tensor(name);
    assert(src && "missing tensor");
    return src;
}

ggml_tensor* clone_weight_opt(ggml_context* ctx, const ModelLoader& ml,
                              const char* name) {
    if (!ml.tensor(name)) return nullptr;
    return clone_weight(ctx, ml, name);
}

void weight_to_host_f32(const ModelLoader& ml, const char* name, std::vector<float>& out) {
    ensure_weights_realized(ml);
    ggml_tensor* t = ml.tensor(name);
    GGML_ASSERT(t && "weight_to_host_f32: missing tensor");
    GGML_ASSERT(t->type == GGML_TYPE_F32 && "weight_to_host_f32: tensor not f32");
    out.resize((size_t)ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
}

} // namespace vd
