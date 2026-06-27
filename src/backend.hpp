#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_backend;
typedef struct ggml_backend* ggml_backend_t;
struct ggml_backend_buffer;
typedef struct ggml_backend_buffer* ggml_backend_buffer_t;

namespace vd {

class ModelLoader;

// Persistent compute backend + reusable graph allocator.
//
// The speaker-encoder inference path builds many small ggml graphs. A naive
// per-call ggml_init -> build -> compute -> ggml_free spends a large fraction of
// wall time on allocator churn. Following parakeet.cpp's design, this keeps a
// persistent ggml_backend_t (GPU if available, else CPU) plus a persistent
// ggml_gallocr_t reused across every graph computation.
//
// Backend::compute() builds the graph in a no_alloc=true context (metadata
// only), allocates it via the persistent gallocr, pushes the host input data
// AFTER alloc, runs it, and reads the output back.
//
// CORRECTNESS-CRITICAL ordering: with no_alloc=true a tensor's ->data is NULL
// until ggml_gallocr_alloc_graph. Input tensor data MUST be written AFTER alloc
// (via add_graph_input registering a deferred copy), never inline in the build
// lambda.
class Backend {
public:
    explicit Backend(int n_threads);
    ~Backend();

    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;

    void set_n_threads(int n_threads);
    int  n_threads() const { return n_threads_; }
    const char* device_name() const { return device_name_.c_str(); }

    // The underlying ggml backend. Exposed so the loader can give its weight
    // tensors a backend buffer over the SAME backend graphs run on.
    ggml_backend_t handle() const;

    // --- Multi-model graph-cache safety (see backend.cpp). ----------------------
    // Drop every persistent graph-cache entry that references `buf` (a freed
    // ModelLoader's weights buffer). Called from ~ModelLoader BEFORE the buffer is
    // released, so a later model reallocating the same address cannot false-hit a
    // stale cached graph built over the now-freed weights. Entries belonging to
    // other (still-live) models are left untouched, so a concurrently-hosted model
    // keeps its hot cache.
    void invalidate_weights_buffer(ggml_backend_buffer_t buf);

    // Cache introspection (exercised by the cache-safety smoke test).
    size_t   cache_size()   const;
    uint64_t cache_hits()   const;
    uint64_t cache_misses() const;

    // Build + run a single graph on the persistent backend/gallocr and copy the
    // output tensor's f32 contents into `out`. Returns true on success.
    bool compute(const std::function<ggml_tensor*(ggml_context*)>& build,
                 std::vector<float>& out);

    // Backend-internal hooks used by add_graph_input()/capture_graph_output().
    void register_input(ggml_tensor* t, const void* host, size_t nbytes);
    void register_capture(ggml_tensor* t, std::vector<float>* dst);

private:
    struct Impl;
    Impl* impl_;
    int   n_threads_ = 1;
    std::string device_name_ = "cpu";
};

// Register a host-backed graph input for the active Backend::compute build
// phase. Marks `t` as a ggml graph input and records that `nbytes` from `host`
// must be copied into `t` AFTER the graph is allocated. Must be called from
// inside a build lambda passed to Backend::compute. `host` must stay valid until
// Backend::compute returns.
void add_graph_input(ggml_tensor* t, const void* host, size_t nbytes);

// Create a graph input tensor of the given type and ne[] inside `ctx`, mark it
// as a graph input, and register a host->device copy of `nbytes` from `host` to
// be performed after the gallocr allocates it. Returns the new tensor.
ggml_tensor* graph_input_tensor(ggml_context* ctx, int type, int n_dims,
                                const int64_t* ne, const void* host,
                                size_t nbytes);

// Capture an intermediate graph tensor for readback after Backend::compute.
// `*dst` is resized to the tensor's element count and filled with its f32
// contents once the graph has run. Must be called from inside a build lambda.
void capture_graph_output(ggml_tensor* t, std::vector<float>* dst);

// Reference a weight tensor from the loader DIRECTLY as a graph leaf (ZERO
// per-call copy). Realizes the loader's weights once (lazily). Returns nullptr
// iff `name` is absent (clone_weight_opt); asserts present otherwise
// (clone_weight). `ctx` is unused (kept for call-site compatibility).
ggml_tensor* clone_weight(ggml_context* ctx, const ModelLoader& ml, const char* name);
ggml_tensor* clone_weight_opt(ggml_context* ctx, const ModelLoader& ml, const char* name);

// Ensure the loader's weights have a backend buffer (zero-copy) on the
// process-global Backend. Idempotent.
void ensure_weights_realized(const ModelLoader& ml);

// Copy a weight tensor's f32 contents into `out` on the host, regardless of
// whether the weight lives in CPU or device memory. For host-side computation
// that needs raw floats (preprocessing) - NOT for graph leaves (use clone_weight).
void weight_to_host_f32(const ModelLoader& ml, const char* name, std::vector<float>& out);

// Route a 2D convolution. The default is ggml_conv_2d (im2col + llamafile-sgemm),
// MEASURED fastest for the speaker-encoder conv shapes on this CPU - the native
// direct conv (ggml_conv_2d_direct) is ~10-15% slower here (see
// benchmarks/RESULTS.md; the opposite of depth-anything's large DPT head). The
// VD_CONV2D=direct|im2col env override keeps the A/B path for other shapes/HW.
// w:[KW,KH,IC,OC] x:[W,H,IC,N] -> [W_out,H_out,OC,N]. Bias is NOT added here.
ggml_tensor* conv2d_auto(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x,
                         int s0, int s1, int p0, int p1, int d0, int d1);

// Which optimized 3x3 stride-1 CPU kernel a WinogradScope routes its eligible
// convs through. Direct = the hand-tuned MLAS-class blocked nChw16c direct conv
// (directconv.cpp), the DEFAULT - it beat AVX-512 Winograd on WeSpeaker/ERes2Net.
// Winograd = the AVX2/AVX-512 F(2x2,3x3) kernel (winograd.cpp), kept reachable as
// the per-model fallback for shapes where it still wins (and via VD_CONV2D=winograd).
enum class Conv3x3Kernel { Direct, Winograd };

// Per-encoder opt-in for the optimized 3x3 stride-1 routing inside conv2d_auto.
// The 2D speaker encoders enable it around their graph build: WeSpeaker ResNet34
// (an all-3x3 backbone, our heaviest encoder), ERes2Net's remaining 3x3 stride-1
// convs, and CAM++'s FCM 3x3 stack. Each picks its MEASURED-best kernel via the
// second arg (default Direct). Every other caller keeps the im2col path.
// Overrides: VD_CONV2D=winograd forces Winograd, VD_CONV2D=directblocked forces
// Direct (both ignore the per-model pin); VOICEDETECT_WINOGRAD=on|off still
// force-toggles the opt-in globally for A/B benchmarking. RAII; restores the
// prior values (so nested/re-entrant builds compose correctly). The flags are read
// at graph-BUILD time (the calling thread), so they are thread_local.
class WinogradScope {
public:
    explicit WinogradScope(bool on, Conv3x3Kernel kernel = Conv3x3Kernel::Direct);
    ~WinogradScope();
    WinogradScope(const WinogradScope&) = delete;
    WinogradScope& operator=(const WinogradScope&) = delete;
private:
    bool prev_;
    Conv3x3Kernel prev_kernel_;
};

// Process-global persistent Backend (created lazily on first use). Exposed so
// the weight-realization path can give the loader's tensors a backend buffer on
// the SAME backend that graphs run on.
Backend& global_backend();

// Free the process-global backend. Call once at program exit (after all model
// objects are destroyed) so GPU backends release device memory while the driver
// is still alive. A later global_backend() call recreates it.
void shutdown_backend();

// Invalidate any persistent graph-cache entries on the process-global Backend
// that reference `buf`. Safe during teardown: it does NOT create the global
// Backend if none exists yet, so a ModelLoader destroyed after shutdown_backend()
// (or before any compute ran) is a harmless no-op. Called by ~ModelLoader.
void invalidate_graph_cache_for_weights(ggml_backend_buffer_t buf);

// Process-global override for the ggml compute thread count (the `--threads N`
// switch). 0 == unset (honor per-call values).
void set_num_threads(int n);
int  num_threads();

} // namespace vd
