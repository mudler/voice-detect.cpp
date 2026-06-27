#include "directconv.hpp"
#include <algorithm>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <chrono>

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

// Blocked-layout (nChw16c) AVX-512 register-tiled DIRECT convolution microkernel
// for the WeSpeaker ResNet34 3x3 stride-1 pad-1 backbone (the Winograd-routed
// set). NO im2col, NO Winograd transform: it streams the NCHW activation
// (broadcast over the 16-OC weight vector) and FMA-accumulates a register tile of
// output-width pixels x 16 output channels directly over the IC*KH*KW window. The
// 16 OC sit in zmm lanes; one aligned zmm weight load is reused across the whole
// output-width strip.
//
// Ship-safety mirrors winograd.cpp EXACTLY: the zmm code carries
// __attribute__((target("avx512f,avx512bw,avx512vl"))) (function multiversioning)
// so it is emitted from an AVX2-only TU (default portable build, GGML_NATIVE=OFF,
// -mavx2 -mfma, NO global -mavx512f); it runs ONLY when __builtin_cpu_supports
// reports AVX-512 at run time, else the AVX2 (ymm, 8c) fallback runs -- so an
// AVX2-only CPU never executes a zmm instruction. A global -mavx512f build
// (GGML_NATIVE=ON) defines __AVX512F__ and the attribute is dropped.
#if defined(__AVX512F__)
#  define VD_DCONV_HAVE_AVX512 1
#  define VD_DCONV_AVX512_TARGET   /* ISA already global; no per-fn attribute */
#elif defined(VOICEDETECT_DCONV_AVX512) && defined(__AVX2__)
#  define VD_DCONV_HAVE_AVX512 1
#  define VD_DCONV_AVX512_TARGET __attribute__((target("avx512f,avx512bw,avx512vl")))
#endif

namespace vd {
namespace {


#if defined(VD_DCONV_HAVE_AVX512)
static bool dconv_use_avx512() {
    static const bool use512 = [] {
        const char* off = std::getenv("VOICEDETECT_DISABLE_AVX512");
        const bool disabled = off && off[0] != '\0' && off[0] != '0';
        const bool supported = __builtin_cpu_supports("avx512f")
                            && __builtin_cpu_supports("avx512bw")
                            && __builtin_cpu_supports("avx512vl");
        const bool sel = supported && !disabled;
        if (std::getenv("VOICEDETECT_DCONV_VERBOSE"))
            std::fprintf(stderr,
                         "[directconv] microkernel: %s (avx512 supported=%d, "
                         "disabled=%d)\n", sel ? "AVX-512(16c)" : "AVX2(8c)",
                         (int)supported, (int)disabled);
        return sel;
    }();
    return use512;
}
#endif

// Output-width register-tile width. OWB acc strips (each = LANE output channels
// for one output pixel) + 1 weight register must fit the vector register file.
// AVX-512: 32 zmm -> OWB up to ~28; start 14 (15 zmm live). AVX2: 16 ymm.
#ifndef VD_DCONV_OWB
#define VD_DCONV_OWB 14
#endif
constexpr int OWB = VD_DCONV_OWB;

// Column-tile width for the 2-OC-block kernel. 2*OWB2 acc + 2 weight + 1 bcast
// must fit 32 zmm; OWB2=14 -> 28+3 = 31.
#ifndef VD_DCONV_OWB2
#define VD_DCONV_OWB2 12
#endif
constexpr int OWB2 = VD_DCONV_OWB2;

// ------------------------------------------------------------------------
// Persistent per-op state. Caches the blocked-packed weights (computed once from
// w->data). LANE = 16 (AVX-512) or 8 (AVX2/scalar). Weights packed
// [OCB][KH][KW][IC][LANE] so the kh,kw,ic inner loop walks them contiguously and
// each (ocb,kh,kw,ic) yields one aligned LANE-wide load of LANE output channels.
// ------------------------------------------------------------------------
struct DConvState {
    int W = 0, H = 0, IC = 0, OC = 0, N = 0, pad = 0;
    int stride = 1;         // spatial stride (1 or 2); used by the blocked path
    int kind = 0;           // 0 = 3x3, 1 = 1x1 (blocked downsample)
    int lane = 16;          // OC block width
    int OCB = 0;            // OC / lane
    const void* wdata = nullptr;
    bool has_bias = false;      // fused bias add (bias is src[2])
    bool do_relu = false;       // fused ReLU after bias
    std::vector<float> Wpack;   // 3x3: [OCB*KH*KW*IC*lane]; 1x1: [OCB*IC*lane]
    std::vector<float> Bpack;   // [OCB*lane] bias broadcast pack (if has_bias)
    std::once_flag once;
};

// ---- env-gated per-shape @1t profiler (VD_DCONV_PROF=1) --------------------
// Accurate only at nth==1 (the whole node runs in one call). Accumulates wall
// time + FLOPs per conv shape and dumps a per-stage GFLOP/s table at exit.
struct ProfEntry { double secs = 0; double flops = 0; long calls = 0;
                   int W=0,H=0,IC=0,OC=0,Wout=0,Hout=0,stride=0,kind=0; };
static std::mutex g_prof_mtx;
static std::unordered_map<uint64_t, ProfEntry> g_prof;
static bool prof_on() {
    static const bool on = [] {
        const char* e = std::getenv("VD_DCONV_PROF");
        return e && e[0] && e[0] != '0';
    }();
    return on;
}
static void prof_dump() {
    std::lock_guard<std::mutex> lk(g_prof_mtx);
    std::fprintf(stderr, "\n[directconv prof] per-shape @1t (kind 0=3x3 1=1x1)\n");
    std::fprintf(stderr, "  %4s %4s %4s %4s %5s %5s %3s %2s %8s %8s %9s\n",
                 "W","H","IC","OC","Wout","Hout","st","k","calls","ms/call","GFLOP/s");
    for (auto& kv : g_prof) {
        ProfEntry& p = kv.second;
        double ms = p.calls ? (p.secs / p.calls) * 1e3 : 0;
        double gf = p.secs > 0 ? (p.flops / p.secs) / 1e9 : 0;
        std::fprintf(stderr, "  %4d %4d %4d %4d %5d %5d %3d %2d %8ld %8.3f %9.1f\n",
                     p.W,p.H,p.IC,p.OC,p.Wout,p.Hout,p.stride,p.kind,p.calls,ms,gf);
    }
}
struct ProfTimer {
    DConvState* st; int Wout, Hout, kind; bool active;
    std::chrono::steady_clock::time_point t0;
    ProfTimer(DConvState* s, int wo, int ho, int knd, int nth)
        : st(s), Wout(wo), Hout(ho), kind(knd), active(prof_on() && nth == 1) {
        if (active) t0 = std::chrono::steady_clock::now();
    }
    ~ProfTimer() {
        if (!active) return;
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        // FLOPs: kind0 3x3 = 2*OC*IC*9*Wout*Hout; kind1 1x1 = 2*OC*IC*Wout*Hout.
        double taps = (kind == 0) ? 9.0 : 1.0;
        double fl = 2.0 * st->OC * st->IC * taps * Wout * Hout;
        uint64_t key = ((uint64_t)(uintptr_t)st->wdata) ^ ((uint64_t)Wout << 1) ^ ((uint64_t)Hout << 17);
        std::lock_guard<std::mutex> lk(g_prof_mtx);
        static std::once_flag once; std::call_once(once, [] { std::atexit(prof_dump); });
        ProfEntry& p = g_prof[key];
        p.secs += dt; p.flops += fl; p.calls += 1;
        p.W=st->W; p.H=st->H; p.IC=st->IC; p.OC=st->OC;
        p.Wout=Wout; p.Hout=Hout; p.stride=st->stride; p.kind=kind;
    }
};

// Pack per-channel bias into a [OCB*lane] broadcast buffer (pad lanes -> 0).
static void build_pack_bias(DConvState* st, const float* b) {
    const int OC = st->OC, lane = st->lane, OCB = st->OCB;
    st->Bpack.assign((size_t)OCB * lane, 0.0f);
    for (int ocb = 0; ocb < OCB; ++ocb)
        for (int l = 0; l < lane; ++l) {
            const int oc = ocb * lane + l;
            st->Bpack[(size_t)ocb * lane + l] = (oc < OC) ? b[oc] : 0.0f;
        }
}

static void build_pack(DConvState* st, const float* w) {
    const int IC = st->IC, OC = st->OC, lane = st->lane, OCB = st->OCB;
    // ggml w ne = [KW=3, KH=3, IC, OC]: w[((oc*IC+ic)*3 + kh)*3 + kw].
    st->Wpack.assign((size_t)OCB * 9 * IC * lane, 0.0f);
    for (int ocb = 0; ocb < OCB; ++ocb)
        for (int kh = 0; kh < 3; ++kh)
            for (int kw = 0; kw < 3; ++kw)
                for (int ic = 0; ic < IC; ++ic) {
                    float* dst = st->Wpack.data() +
                        ((((size_t)ocb * 9 + kh * 3 + kw) * IC) + ic) * lane;
                    for (int l = 0; l < lane; ++l) {
                        const int oc = ocb * lane + l;
                        dst[l] = (oc < OC)
                            ? w[(((size_t)oc * IC + ic) * 3 + kh) * 3 + kw]
                            : 0.0f;
                    }
                }
}

// 1x1 weight pack [OCB][IC][lane]: ggml w ne = [1, 1, IC, OC], so
// w[oc*IC + ic]. dst[(ocb*IC + ic)*lane + l] = w[(ocb*lane+l)*IC + ic].
static void build_pack_1x1(DConvState* st, const float* w) {
    const int IC = st->IC, OC = st->OC, lane = st->lane, OCB = st->OCB;
    st->Wpack.assign((size_t)OCB * IC * lane, 0.0f);
    for (int ocb = 0; ocb < OCB; ++ocb)
        for (int ic = 0; ic < IC; ++ic) {
            float* dst = st->Wpack.data() + ((size_t)ocb * IC + ic) * lane;
            for (int l = 0; l < lane; ++l) {
                const int oc = ocb * lane + l;
                dst[l] = (oc < OC) ? w[(size_t)oc * IC + ic] : 0.0f;
            }
        }
}

// ========================================================================
// AVX-512 microkernel: one output row (ocb, ih) -> blocked row buffer
// outrow[W*16] (outrow[ow*16 + lane]). Interior columns [pad, W-1-pad] run the
// no-branch register-tiled path; the edge columns run a masked-guard path.
// ========================================================================
#if defined(VD_DCONV_HAVE_AVX512)
// One interior strip of CW output columns. CW is compile-time so the j loops
// FULLY UNROLL: no per-FMA branch, weight loaded once per ic and reused across
// the strip via embedded-broadcast FMA (vfmadd231ps {1to16}).
template<int CW>
static inline VD_DCONV_AVX512_TARGET void dconv_strip_avx512(
        const float* x, const float* wocb, float* outrow,
        int W, int H, int IC, int ih, int pad, int ow0) {
    const size_t HW = (size_t)H * W;
    __m512 acc[CW];
    for (int j = 0; j < CW; ++j) acc[j] = _mm512_setzero_ps();
    for (int kh = 0; kh < 3; ++kh) {
        const int iy = ih + kh - pad;
        if (iy < 0 || iy >= H) continue;
        for (int kw = 0; kw < 3; ++kw) {
            const float* wk = wocb + ((size_t)kh * 3 + kw) * IC * 16;
            const float* in = x + (size_t)iy * W + (ow0 + kw - pad);
            for (int ic = 0; ic < IC; ++ic) {
                const __m512 wv = _mm512_loadu_ps(wk + (size_t)ic * 16);
                const float* inc = in + (size_t)ic * HW;
                for (int j = 0; j < CW; ++j)
                    acc[j] = _mm512_fmadd_ps(_mm512_set1_ps(inc[j]), wv, acc[j]);
            }
        }
    }
    for (int j = 0; j < CW; ++j)
        _mm512_store_ps(outrow + (size_t)(ow0 + j) * 16, acc[j]);
}

// Two-OC-block strip: each input broadcast feeds TWO weight vectors (the two OC
// blocks ocb0/ocb1), halving the broadcast-load pressure per FMA -> pushes the
// kernel from load-bound toward FMA-bound. 2*CW acc + 2 weight + 1 bcast regs.
template<int CW>
static inline VD_DCONV_AVX512_TARGET void dconv_strip2_avx512(
        const float* x, const float* wocb0, const float* wocb1,
        float* outrow0, float* outrow1,
        int W, int H, int IC, int ih, int pad, int ow0) {
    const size_t HW = (size_t)H * W;
    __m512 a0[CW], a1[CW];
    for (int j = 0; j < CW; ++j) { a0[j] = _mm512_setzero_ps(); a1[j] = _mm512_setzero_ps(); }
    for (int kh = 0; kh < 3; ++kh) {
        const int iy = ih + kh - pad;
        if (iy < 0 || iy >= H) continue;
        for (int kw = 0; kw < 3; ++kw) {
            const size_t off = ((size_t)kh * 3 + kw) * IC * 16;
            const float* wk0 = wocb0 + off;
            const float* wk1 = wocb1 + off;
            const float* in = x + (size_t)iy * W + (ow0 + kw - pad);
            for (int ic = 0; ic < IC; ++ic) {
                const __m512 w0 = _mm512_loadu_ps(wk0 + (size_t)ic * 16);
                const __m512 w1 = _mm512_loadu_ps(wk1 + (size_t)ic * 16);
                const float* inc = in + (size_t)ic * HW;
                for (int j = 0; j < CW; ++j) {
                    const __m512 b = _mm512_set1_ps(inc[j]);
                    a0[j] = _mm512_fmadd_ps(b, w0, a0[j]);
                    a1[j] = _mm512_fmadd_ps(b, w1, a1[j]);
                }
            }
        }
    }
    for (int j = 0; j < CW; ++j) {
        _mm512_store_ps(outrow0 + (size_t)(ow0 + j) * 16, a0[j]);
        _mm512_store_ps(outrow1 + (size_t)(ow0 + j) * 16, a1[j]);
    }
}

// One edge column for a single OC block (full bounds checks).
static inline VD_DCONV_AVX512_TARGET void dconv_edge_avx512(
        const float* x, const float* wocb, float* outrow,
        int W, int H, int IC, int ih, int pad, int ow) {
    const size_t HW = (size_t)H * W;
    __m512 acc = _mm512_setzero_ps();
    for (int kh = 0; kh < 3; ++kh) {
        const int iy = ih + kh - pad;
        if (iy < 0 || iy >= H) continue;
        for (int kw = 0; kw < 3; ++kw) {
            const int ix = ow + kw - pad;
            if (ix < 0 || ix >= W) continue;
            const float* wk = wocb + ((size_t)kh * 3 + kw) * IC * 16;
            const float* in = x + (size_t)iy * W + ix;
            for (int ic = 0; ic < IC; ++ic)
                acc = _mm512_fmadd_ps(_mm512_set1_ps(in[(size_t)ic * HW]),
                                      _mm512_loadu_ps(wk + (size_t)ic * 16), acc);
        }
    }
    _mm512_store_ps(outrow + (size_t)ow * 16, acc);
}

// Interior strips for one OC block, fully-unrolled fixed widths + remainder.
static inline VD_DCONV_AVX512_TARGET void dconv_interior1_avx512(
        const float* x, const float* wocb, float* outrow,
        int W, int H, int IC, int ih, int pad) {
    const int hi = W - 1 - pad;
    int ow0 = pad;
    for (; ow0 + OWB - 1 <= hi; ow0 += OWB)
        dconv_strip_avx512<OWB>(x, wocb, outrow, W, H, IC, ih, pad, ow0);
    int rem = hi - ow0 + 1;
    while (rem >= 8) { dconv_strip_avx512<8>(x, wocb, outrow, W, H, IC, ih, pad, ow0); ow0 += 8; rem -= 8; }
    while (rem >= 4) { dconv_strip_avx512<4>(x, wocb, outrow, W, H, IC, ih, pad, ow0); ow0 += 4; rem -= 4; }
    while (rem >= 2) { dconv_strip_avx512<2>(x, wocb, outrow, W, H, IC, ih, pad, ow0); ow0 += 2; rem -= 2; }
    while (rem >= 1) { dconv_strip_avx512<1>(x, wocb, outrow, W, H, IC, ih, pad, ow0); ow0 += 1; rem -= 1; }
}

// Two-OC-block output row: edges per block, interior via the shared-broadcast
// 2-block strips.
static inline VD_DCONV_AVX512_TARGET void dconv_row2_avx512(
        const float* x, const float* Wpack, float* outrow0, float* outrow1,
        int W, int H, int IC, int ocb0, int ocb1, int ih, int pad) {
    const float* w0 = Wpack + (size_t)ocb0 * 9 * IC * 16;
    const float* w1 = Wpack + (size_t)ocb1 * 9 * IC * 16;
    for (int e = 0; e < 2 * pad; ++e) {
        const int ow = (e < pad) ? e : (W - pad + (e - pad));
        if (ow < 0 || ow >= W) continue;
        if (e >= pad && ow < pad) continue;
        dconv_edge_avx512(x, w0, outrow0, W, H, IC, ih, pad, ow);
        dconv_edge_avx512(x, w1, outrow1, W, H, IC, ih, pad, ow);
    }
    const int hi = W - 1 - pad;
    int ow0 = pad;
    for (; ow0 + OWB2 - 1 <= hi; ow0 += OWB2)
        dconv_strip2_avx512<OWB2>(x, w0, w1, outrow0, outrow1, W, H, IC, ih, pad, ow0);
    int rem = hi - ow0 + 1;
    while (rem >= 8) { dconv_strip2_avx512<8>(x, w0, w1, outrow0, outrow1, W, H, IC, ih, pad, ow0); ow0 += 8; rem -= 8; }
    while (rem >= 4) { dconv_strip2_avx512<4>(x, w0, w1, outrow0, outrow1, W, H, IC, ih, pad, ow0); ow0 += 4; rem -= 4; }
    while (rem >= 2) { dconv_strip2_avx512<2>(x, w0, w1, outrow0, outrow1, W, H, IC, ih, pad, ow0); ow0 += 2; rem -= 2; }
    while (rem >= 1) { dconv_strip2_avx512<1>(x, w0, w1, outrow0, outrow1, W, H, IC, ih, pad, ow0); ow0 += 1; rem -= 1; }
}

static inline VD_DCONV_AVX512_TARGET void dconv_row_avx512(
        const float* x, const float* Wpack, float* outrow,
        int W, int H, int IC, int ocb, int ih, int pad) {
    const size_t HW = (size_t)H * W;
    const float* wocb = Wpack + (size_t)ocb * 9 * IC * 16;

    // Edge columns (ow in [0,pad) and [W-pad,W)): full bounds checks. Two ranges
    // walked as one loop so the zmm intrinsics stay in this target-tagged body
    // (a lambda would not inherit the function-multiversioning target).
    for (int e = 0; e < 2 * pad; ++e) {
        const int ow = (e < pad) ? e : (W - pad + (e - pad));
        if (ow < 0 || ow >= W) continue;
        if (e >= pad && ow < pad) continue;   // overlap when W < 2*pad
        __m512 acc = _mm512_setzero_ps();
        for (int kh = 0; kh < 3; ++kh) {
            const int iy = ih + kh - pad;
            if (iy < 0 || iy >= H) continue;
            for (int kw = 0; kw < 3; ++kw) {
                const int ix = ow + kw - pad;
                if (ix < 0 || ix >= W) continue;
                const float* wk = wocb + ((size_t)kh * 3 + kw) * IC * 16;
                const float* in = x + (size_t)iy * W + ix;
                for (int ic = 0; ic < IC; ++ic)
                    acc = _mm512_fmadd_ps(_mm512_set1_ps(in[(size_t)ic * HW]),
                                          _mm512_loadu_ps(wk + (size_t)ic * 16), acc);
            }
        }
        _mm512_store_ps(outrow + (size_t)ow * 16, acc);
    }

    // Interior columns ow in [pad, W-1-pad]: ix = ow + kw - pad in [0, W-1] always
    // (holds for pad >= 1, which every WeSpeaker 3x3 conv uses). Bulk strips run
    // the fully-unrolled fixed-OWB kernel; the last partial strip is split into
    // smaller fixed-width kernels (8/4/2/1) so every column stays branchless.
    const int hi = W - 1 - pad;             // last interior column
    int ow0 = pad;
    for (; ow0 + OWB - 1 <= hi; ow0 += OWB)
        dconv_strip_avx512<OWB>(x, wocb, outrow, W, H, IC, ih, pad, ow0);
    int rem = hi - ow0 + 1;
    while (rem >= 8) { dconv_strip_avx512<8>(x, wocb, outrow, W, H, IC, ih, pad, ow0); ow0 += 8; rem -= 8; }
    while (rem >= 4) { dconv_strip_avx512<4>(x, wocb, outrow, W, H, IC, ih, pad, ow0); ow0 += 4; rem -= 4; }
    while (rem >= 2) { dconv_strip_avx512<2>(x, wocb, outrow, W, H, IC, ih, pad, ow0); ow0 += 2; rem -= 2; }
    while (rem >= 1) { dconv_strip_avx512<1>(x, wocb, outrow, W, H, IC, ih, pad, ow0); ow0 += 1; rem -= 1; }
}
#endif

// ========================================================================
// AVX2 microkernel: LANE=8 ymm. Same structure.
// ========================================================================
#if defined(__AVX2__)
static inline void dconv_row_avx2(
        const float* x, const float* Wpack, float* outrow,
        int W, int H, int IC, int ocb, int ih, int pad) {
    const size_t HW = (size_t)H * W;
    const float* wocb = Wpack + (size_t)ocb * 9 * IC * 8;
    auto edge_col = [&](int ow) {
        __m256 acc = _mm256_setzero_ps();
        for (int kh = 0; kh < 3; ++kh) {
            const int iy = ih + kh - pad;
            if (iy < 0 || iy >= H) continue;
            for (int kw = 0; kw < 3; ++kw) {
                const int ix = ow + kw - pad;
                if (ix < 0 || ix >= W) continue;
                const float* wk = wocb + ((size_t)kh * 3 + kw) * IC * 8;
                const float* in = x + (size_t)iy * W + ix;
                for (int ic = 0; ic < IC; ++ic)
                    acc = _mm256_fmadd_ps(_mm256_set1_ps(in[ic * HW]),
                                          _mm256_loadu_ps(wk + (size_t)ic * 8), acc);
            }
        }
        _mm256_store_ps(outrow + (size_t)ow * 8, acc);
    };
    for (int ow = 0; ow < pad && ow < W; ++ow) edge_col(ow);
    for (int ow = std::max(W - pad, pad); ow < W; ++ow) edge_col(ow);
    for (int ow0 = pad; ow0 <= W - 1 - pad; ow0 += OWB) {
        const int owc = std::min(OWB, (W - pad) - ow0);
        __m256 acc[OWB];
        for (int j = 0; j < owc; ++j) acc[j] = _mm256_setzero_ps();
        for (int kh = 0; kh < 3; ++kh) {
            const int iy = ih + kh - pad;
            if (iy < 0 || iy >= H) continue;
            for (int kw = 0; kw < 3; ++kw) {
                const float* wk = wocb + ((size_t)kh * 3 + kw) * IC * 8;
                const float* in = x + (size_t)iy * W + (ow0 + kw - pad);
                for (int ic = 0; ic < IC; ++ic) {
                    const __m256 wv = _mm256_loadu_ps(wk + (size_t)ic * 8);
                    const float* inc = in + (size_t)ic * HW;
                    for (int j = 0; j < owc; ++j)
                        acc[j] = _mm256_fmadd_ps(_mm256_set1_ps(inc[j]), wv, acc[j]);
                }
            }
        }
        for (int j = 0; j < owc; ++j)
            _mm256_store_ps(outrow + (size_t)(ow0 + j) * 8, acc[j]);
    }
}
#endif

// Scalar fallback (non-x86): LANE-wide blocked row, plain loops.
static void dconv_row_scalar(const float* x, const float* Wpack, float* outrow,
                             int W, int H, int IC, int lane, int ocb, int ih, int pad) {
    const size_t HW = (size_t)H * W;
    const float* wocb = Wpack + (size_t)ocb * 9 * IC * lane;
    for (int ow = 0; ow < W; ++ow) {
        float* o = outrow + (size_t)ow * lane;
        for (int l = 0; l < lane; ++l) o[l] = 0.0f;
        for (int kh = 0; kh < 3; ++kh) {
            const int iy = ih + kh - pad;
            if (iy < 0 || iy >= H) continue;
            for (int kw = 0; kw < 3; ++kw) {
                const int ix = ow + kw - pad;
                if (ix < 0 || ix >= W) continue;
                const float* wk = wocb + ((size_t)kh * 3 + kw) * IC * lane;
                const float* in = x + (size_t)iy * W + ix;
                for (int ic = 0; ic < IC; ++ic) {
                    const float s = in[(size_t)ic * HW];
                    const float* wv = wk + (size_t)ic * lane;
                    for (int l = 0; l < lane; ++l) o[l] += s * wv[l];
                }
            }
        }
    }
}

// Transpose one blocked output row [W*lane] -> NCHW dst planes:
// dst[(ocb*lane + l)*HW + ih*W + ow] = outrow[ow*lane + l].
static inline void scatter_row(float* y, const float* outrow,
                               int W, size_t HW, int lane, int ocb, int ih, int OC) {
    const size_t base = ih * (size_t)W;
    for (int l = 0; l < lane; ++l) {
        const int oc = ocb * lane + l;
        if (oc >= OC) break;
        float* yc = y + (size_t)oc * HW + base;
        for (int ow = 0; ow < W; ++ow) yc[ow] = outrow[(size_t)ow * lane + l];
    }
}

static void dconv_compute(ggml_tensor* dst, int ith, int nth, void* userdata) {
    DConvState* st = (DConvState*)userdata;
    const ggml_tensor* xt = dst->src[0];
    const ggml_tensor* wt = dst->src[1];
    const float* x = (const float*)xt->data;
    float* y = (float*)dst->data;

    std::call_once(st->once, [&] { build_pack(st, (const float*)wt->data); });

    ProfTimer _pt(st, (int)dst->ne[0], (int)dst->ne[1], /*kind=*/0, nth);

    const int W = st->W, H = st->H, IC = st->IC, OC = st->OC;
    const int pad = st->pad, lane = st->lane, OCB = st->OCB;
    const float* Wpack = st->Wpack.data();
    const size_t HW = (size_t)H * W;

    // Per-thread blocked output-row scratch (aligned for vector stores). Two rows
    // so the AVX-512 path can emit a pair of OC blocks per pass.
    std::vector<float> rowbuf((size_t)W * lane * 2 + 32);
    float* outrow0 = (float*)(((uintptr_t)rowbuf.data() + 63) & ~(uintptr_t)63);
    float* outrow1 = outrow0 + (size_t)W * lane;

#if defined(VD_DCONV_HAVE_AVX512)
    if (lane == 16) {
        // Work items = (OC-block PAIR, ih). Each pass shares one input broadcast
        // across two OC blocks. Odd trailing block falls back to the 1-block row.
        const int npairs = (OCB + 1) / 2;
        const int64_t total = (int64_t)npairs * H;
        const int64_t beg = total * ith / nth, end = total * (ith + 1) / nth;
        for (int64_t idx = beg; idx < end; ++idx) {
            const int p  = (int)(idx / H);
            const int ih = (int)(idx % H);
            const int ocb0 = 2 * p, ocb1 = 2 * p + 1;
            if (ocb1 < OCB) {
                dconv_row2_avx512(x, Wpack, outrow0, outrow1, W, H, IC, ocb0, ocb1, ih, pad);
                scatter_row(y, outrow0, W, HW, 16, ocb0, ih, OC);
                scatter_row(y, outrow1, W, HW, 16, ocb1, ih, OC);
            } else {
                dconv_row_avx512(x, Wpack, outrow0, W, H, IC, ocb0, ih, pad);
                scatter_row(y, outrow0, W, HW, 16, ocb0, ih, OC);
            }
        }
        return;
    }
#endif
    // AVX2 / scalar: single OC block per (ocb, ih) work item.
    const int64_t total = (int64_t)OCB * H;
    const int64_t beg = total * ith / nth, end = total * (ith + 1) / nth;
    for (int64_t idx = beg; idx < end; ++idx) {
        const int ocb = (int)(idx / H);
        const int ih  = (int)(idx % H);
#if defined(__AVX2__)
        if (lane == 8) { dconv_row_avx2(x, Wpack, outrow0, W, H, IC, ocb, ih, pad); }
        else
#endif
        { dconv_row_scalar(x, Wpack, outrow0, W, H, IC, lane, ocb, ih, pad); }
        scatter_row(y, outrow0, W, HW, lane, ocb, ih, OC);
    }
}

// ========================================================================
// BLOCKED-LAYOUT (nChw16c) ISLAND: ops that consume AND produce the blocked
// buffer ne = [16, W, H, CB] (flat index ((cb*H + h)*W + w)*16 + l, channel
// c = cb*16 + l). They let the whole ResNet backbone stay blocked between ONE
// reorder-in (after the stem) and ONE reorder-out (before pooling), instead of
// the per-conv NCHW<->blocked round trip the directconv path pays on every layer.
// Channel widths here (32/64/128/256) are all multiples of 16, so CB = C/16 is
// exact and there are no padding lanes for WeSpeaker.
// ========================================================================
constexpr int BLK = 16;   // nChw16c block width (the island is AVX-512 sized)

// --- blocked 3x3 conv: one output column (ocb, oh, ow), all bounds checked.
// Reads blocked input, writes the 16-OC tile straight into the blocked output;
// kh,kw,ic accumulation order matches dconv_*  so the result is bit-identical to
// the NCHW direct conv (layout-only change).
#if defined(VD_DCONV_HAVE_AVX512)
// Conv epilogue: optional fused bias add (bvec = 16 bias floats for this OC block,
// or nullptr) + optional ReLU, then store. Fusing bias/ReLU into the conv's own
// register tile removes the separate full-buffer blocked-bias / blocked-relu
// passes (the master path applies them as cheap NCHW built-ins; in the blocked
// island they were standalone scalar passes, the dominant remaining overhead).
static inline VD_DCONV_AVX512_TARGET void bemit_avx512(
        float* p, __m512 acc, const float* bvec, bool relu) {
    if (bvec) acc = _mm512_add_ps(acc, _mm512_loadu_ps(bvec));
    if (relu) acc = _mm512_max_ps(acc, _mm512_setzero_ps());
    _mm512_storeu_ps(p, acc);
}

static inline VD_DCONV_AVX512_TARGET void bconv3x3_col_avx512(
        const float* xb, const float* wocb, float* orow,
        int W, int H, int IC, int oh, int ow, int pad, int stride,
        const float* bvec = nullptr, bool relu = false) {
    __m512 acc = _mm512_setzero_ps();
    for (int kh = 0; kh < 3; ++kh) {
        const int iy = oh * stride + kh - pad;
        if (iy < 0 || iy >= H) continue;
        for (int kw = 0; kw < 3; ++kw) {
            const int ix = ow * stride + kw - pad;
            if (ix < 0 || ix >= W) continue;
            const float* wk = wocb + ((size_t)kh * 3 + kw) * IC * BLK;
            const float* col = xb + (((size_t)iy * W) + ix) * BLK;  // (cb=0,l=0) base
            for (int ic = 0; ic < IC; ++ic) {
                const int cb = ic >> 4, l = ic & 15;
                const float s = col[(size_t)cb * H * W * BLK + l];
                acc = _mm512_fmadd_ps(_mm512_set1_ps(s),
                                      _mm512_loadu_ps(wk + (size_t)ic * BLK), acc);
            }
        }
    }
    bemit_avx512(orow + (size_t)ow * BLK, acc, bvec, relu);
}

// stride-1 interior register-tiled strip (CW output columns), fully unrolled.
// Blocked input column stride is BLK floats (vs 1 for NCHW). ix = ow + kw - pad.
template<int CW>
static inline VD_DCONV_AVX512_TARGET void bconv3x3_strip_s1_avx512(
        const float* xb, const float* wocb, float* orow,
        int W, int H, int IC, int ih, int pad, int ow0,
        const float* bvec = nullptr, bool relu = false) {
    __m512 acc[CW];
    for (int j = 0; j < CW; ++j) acc[j] = _mm512_setzero_ps();
    for (int kh = 0; kh < 3; ++kh) {
        const int iy = ih + kh - pad;
        if (iy < 0 || iy >= H) continue;
        for (int kw = 0; kw < 3; ++kw) {
            const float* wk = wocb + ((size_t)kh * 3 + kw) * IC * BLK;
            const int ix0 = ow0 + kw - pad;
            const float* base = xb + (((size_t)iy * W) + ix0) * BLK;  // cb=0,l=0
            for (int ic = 0; ic < IC; ++ic) {
                const int cb = ic >> 4, l = ic & 15;
                const float* inc = base + (size_t)cb * H * W * BLK + l;
                const __m512 wv = _mm512_loadu_ps(wk + (size_t)ic * BLK);
                for (int j = 0; j < CW; ++j)
                    acc[j] = _mm512_fmadd_ps(_mm512_set1_ps(inc[(size_t)j * BLK]), wv, acc[j]);
            }
        }
    }
    for (int j = 0; j < CW; ++j)
        bemit_avx512(orow + (size_t)(ow0 + j) * BLK, acc[j], bvec, relu);
}

// stride-1 interior 2-OC-block strip: each (strided, expensive) blocked-input
// broadcast feeds TWO OC-block weight vectors, halving the broadcast-load count
// per FMA (the master directconv's winning lever, here aimed at the blocked
// layout's width-strided input reads). Writes both OC blocks' columns.
template<int CW>
static inline VD_DCONV_AVX512_TARGET void bconv3x3_strip2_s1_avx512(
        const float* xb, const float* wocb0, const float* wocb1,
        float* orow0, float* orow1,
        int W, int H, int IC, int ih, int pad, int ow0,
        const float* bvec0 = nullptr, const float* bvec1 = nullptr, bool relu = false) {
    __m512 a0[CW], a1[CW];
    for (int j = 0; j < CW; ++j) { a0[j] = _mm512_setzero_ps(); a1[j] = _mm512_setzero_ps(); }
    for (int kh = 0; kh < 3; ++kh) {
        const int iy = ih + kh - pad;
        if (iy < 0 || iy >= H) continue;
        for (int kw = 0; kw < 3; ++kw) {
            const size_t off = ((size_t)kh * 3 + kw) * IC * BLK;
            const float* wk0 = wocb0 + off;
            const float* wk1 = wocb1 + off;
            const int ix0 = ow0 + kw - pad;
            const float* base = xb + (((size_t)iy * W) + ix0) * BLK;
            for (int ic = 0; ic < IC; ++ic) {
                const int cb = ic >> 4, l = ic & 15;
                const float* inc = base + (size_t)cb * H * W * BLK + l;
                const __m512 w0 = _mm512_loadu_ps(wk0 + (size_t)ic * BLK);
                const __m512 w1 = _mm512_loadu_ps(wk1 + (size_t)ic * BLK);
                for (int j = 0; j < CW; ++j) {
                    const __m512 b = _mm512_set1_ps(inc[(size_t)j * BLK]);
                    a0[j] = _mm512_fmadd_ps(b, w0, a0[j]);
                    a1[j] = _mm512_fmadd_ps(b, w1, a1[j]);
                }
            }
        }
    }
    for (int j = 0; j < CW; ++j) {
        bemit_avx512(orow0 + (size_t)(ow0 + j) * BLK, a0[j], bvec0, relu);
        bemit_avx512(orow1 + (size_t)(ow0 + j) * BLK, a1[j], bvec1, relu);
    }
}

// stride-1 2-OC-block output-row driver: edges per block (single-OC col kernel),
// interior via the shared-broadcast 2-block strips.
static inline VD_DCONV_AVX512_TARGET void bconv3x3_row2_s1_avx512(
        const float* xb, const float* Wpack, float* y,
        int W, int H, int IC, int Wout, int Hout, int ocb0, int ocb1, int oh, int pad,
        const float* Bpack, bool relu) {
    const float* w0 = Wpack + (size_t)ocb0 * 9 * IC * BLK;
    const float* w1 = Wpack + (size_t)ocb1 * 9 * IC * BLK;
    const float* b0 = Bpack ? Bpack + (size_t)ocb0 * BLK : nullptr;
    const float* b1 = Bpack ? Bpack + (size_t)ocb1 * BLK : nullptr;
    float* orow0 = y + (((size_t)ocb0 * Hout + oh) * Wout) * BLK;
    float* orow1 = y + (((size_t)ocb1 * Hout + oh) * Wout) * BLK;
    int ow = 0;
    for (; ow < pad && ow < Wout; ++ow) {
        bconv3x3_col_avx512(xb, w0, orow0, W, H, IC, oh, ow, pad, 1, b0, relu);
        bconv3x3_col_avx512(xb, w1, orow1, W, H, IC, oh, ow, pad, 1, b1, relu);
    }
    const int hi = Wout - 1 - pad;
    int ow0 = std::max(pad, ow);
    for (; ow0 + OWB2 - 1 <= hi; ow0 += OWB2)
        bconv3x3_strip2_s1_avx512<OWB2>(xb, w0, w1, orow0, orow1, W, H, IC, oh, pad, ow0, b0, b1, relu);
    int rem = hi - ow0 + 1;
    while (rem >= 8) { bconv3x3_strip2_s1_avx512<8>(xb, w0, w1, orow0, orow1, W, H, IC, oh, pad, ow0, b0, b1, relu); ow0 += 8; rem -= 8; }
    while (rem >= 4) { bconv3x3_strip2_s1_avx512<4>(xb, w0, w1, orow0, orow1, W, H, IC, oh, pad, ow0, b0, b1, relu); ow0 += 4; rem -= 4; }
    while (rem >= 2) { bconv3x3_strip2_s1_avx512<2>(xb, w0, w1, orow0, orow1, W, H, IC, oh, pad, ow0, b0, b1, relu); ow0 += 2; rem -= 2; }
    while (rem >= 1) { bconv3x3_strip2_s1_avx512<1>(xb, w0, w1, orow0, orow1, W, H, IC, oh, pad, ow0, b0, b1, relu); ow0 += 1; rem -= 1; }
    for (int owe = std::max(Wout - pad, ow0); owe < Wout; ++owe) {
        bconv3x3_col_avx512(xb, w0, orow0, W, H, IC, oh, owe, pad, 1, b0, relu);
        bconv3x3_col_avx512(xb, w1, orow1, W, H, IC, oh, owe, pad, 1, b1, relu);
    }
}

// ---- STRIDED 3x3 register-tiled strip (stride>=2) -------------------------
// PROFILE-DRIVEN: the stride-1 width-tiled kernel is already FMA-bound near peak
// at every WeSpeaker spatial (270-308 GFLOP/s even at the small 38x10 stage-4
// map). The collapse is the stride-2 DOWNSAMPLE conv1 of stages 2/3/4, which had
// NO register-tiled kernel and fell through to the per-column single-accumulator
// path at ~42 GFLOP/s (1/7 peak) - dragging each stage's aggregate to the
// 203/219/175 seen at the encoder level. For a strided conv consecutive output
// columns map to input columns ix = ow*stride + kw - pad, i.e. a CONSTANT input
// column step of stride*BLK floats; so the exact same OC-in-lanes width tile
// fills, just walking the input with a stride*BLK step. kh,kw,ic order is
// identical to bconv3x3_col_avx512, so the result is bit-identical (parity-safe).
// Two OC blocks share each (strided) input broadcast, as in the stride-1 path.
template<int CW>
static inline VD_DCONV_AVX512_TARGET void bconv3x3_strip2_sN_avx512(
        const float* xb, const float* wocb0, const float* wocb1,
        float* orow0, float* orow1,
        int W, int H, int IC, int oh, int pad, int stride, int ow0,
        const float* bvec0, const float* bvec1, bool relu) {
    __m512 a0[CW], a1[CW];
    for (int j = 0; j < CW; ++j) { a0[j] = _mm512_setzero_ps(); a1[j] = _mm512_setzero_ps(); }
    const size_t jstep = (size_t)stride * BLK;
    for (int kh = 0; kh < 3; ++kh) {
        const int iy = oh * stride + kh - pad;
        if (iy < 0 || iy >= H) continue;
        for (int kw = 0; kw < 3; ++kw) {
            const size_t off = ((size_t)kh * 3 + kw) * IC * BLK;
            const float* wk0 = wocb0 + off;
            const float* wk1 = wocb1 + off;
            const int ix0 = ow0 * stride + kw - pad;   // interior: all CW taps in-bounds
            const float* base = xb + (((size_t)iy * W) + ix0) * BLK;
            for (int ic = 0; ic < IC; ++ic) {
                const int cb = ic >> 4, l = ic & 15;
                const float* inc = base + (size_t)cb * H * W * BLK + l;
                const __m512 w0 = _mm512_loadu_ps(wk0 + (size_t)ic * BLK);
                const __m512 w1 = _mm512_loadu_ps(wk1 + (size_t)ic * BLK);
                for (int j = 0; j < CW; ++j) {
                    const __m512 b = _mm512_set1_ps(inc[(size_t)j * jstep]);
                    a0[j] = _mm512_fmadd_ps(b, w0, a0[j]);
                    a1[j] = _mm512_fmadd_ps(b, w1, a1[j]);
                }
            }
        }
    }
    for (int j = 0; j < CW; ++j) {
        bemit_avx512(orow0 + (size_t)(ow0 + j) * BLK, a0[j], bvec0, relu);
        bemit_avx512(orow1 + (size_t)(ow0 + j) * BLK, a1[j], bvec1, relu);
    }
}

// strided 2-OC-block output-row driver: low/high edge columns via the bounds-
// checked single-OC col kernel, the interior via register-tiled strided strips.
// Interior output cols are those where every kw tap lands in [0,W-1]:
// ow in [ceil(pad/stride), floor((W-1-(KW-1)+pad)/stride)].
static inline VD_DCONV_AVX512_TARGET void bconv3x3_row2_sN_avx512(
        const float* xb, const float* Wpack, float* y,
        int W, int H, int IC, int Wout, int Hout, int ocb0, int ocb1, int oh,
        int pad, int stride, const float* Bpack, bool relu) {
    const float* w0 = Wpack + (size_t)ocb0 * 9 * IC * BLK;
    const float* w1 = Wpack + (size_t)ocb1 * 9 * IC * BLK;
    const float* b0 = Bpack ? Bpack + (size_t)ocb0 * BLK : nullptr;
    const float* b1 = Bpack ? Bpack + (size_t)ocb1 * BLK : nullptr;
    float* orow0 = y + (((size_t)ocb0 * Hout + oh) * Wout) * BLK;
    float* orow1 = y + (((size_t)ocb1 * Hout + oh) * Wout) * BLK;
    int ow_lo = (pad + stride - 1) / stride;            // ceil(pad/stride)
    int ow_hi = (W - 3 + pad) / stride;                 // last fully-interior col
    if (ow_hi > Wout - 1) ow_hi = Wout - 1;
    if (ow_lo > Wout) ow_lo = Wout;
    for (int ow = 0; ow < ow_lo; ++ow) {                // low edge columns
        bconv3x3_col_avx512(xb, w0, orow0, W, H, IC, oh, ow, pad, stride, b0, relu);
        bconv3x3_col_avx512(xb, w1, orow1, W, H, IC, oh, ow, pad, stride, b1, relu);
    }
    int ow0 = ow_lo;
    for (; ow0 + OWB2 - 1 <= ow_hi; ow0 += OWB2)
        bconv3x3_strip2_sN_avx512<OWB2>(xb, w0, w1, orow0, orow1, W, H, IC, oh, pad, stride, ow0, b0, b1, relu);
    int rem = ow_hi - ow0 + 1;
    while (rem >= 8) { bconv3x3_strip2_sN_avx512<8>(xb, w0, w1, orow0, orow1, W, H, IC, oh, pad, stride, ow0, b0, b1, relu); ow0 += 8; rem -= 8; }
    while (rem >= 4) { bconv3x3_strip2_sN_avx512<4>(xb, w0, w1, orow0, orow1, W, H, IC, oh, pad, stride, ow0, b0, b1, relu); ow0 += 4; rem -= 4; }
    while (rem >= 2) { bconv3x3_strip2_sN_avx512<2>(xb, w0, w1, orow0, orow1, W, H, IC, oh, pad, stride, ow0, b0, b1, relu); ow0 += 2; rem -= 2; }
    while (rem >= 1) { bconv3x3_strip2_sN_avx512<1>(xb, w0, w1, orow0, orow1, W, H, IC, oh, pad, stride, ow0, b0, b1, relu); ow0 += 1; rem -= 1; }
    for (int ow = (ow_hi + 1 > ow_lo ? ow_hi + 1 : ow_lo); ow < Wout; ++ow) {  // high edge
        bconv3x3_col_avx512(xb, w0, orow0, W, H, IC, oh, ow, pad, stride, b0, relu);
        bconv3x3_col_avx512(xb, w1, orow1, W, H, IC, oh, ow, pad, stride, b1, relu);
    }
}

// blocked 1x1 strided conv: one output column. pad 0, single tap.
static inline VD_DCONV_AVX512_TARGET void bconv1x1_col_avx512(
        const float* xb, const float* wocb, float* orow,
        int W, int H, int IC, int oh, int ow, int stride,
        const float* bvec = nullptr, bool relu = false) {
    const int iy = oh * stride, ix = ow * stride;
    const float* col = xb + (((size_t)iy * W) + ix) * BLK;
    __m512 acc = _mm512_setzero_ps();
    for (int ic = 0; ic < IC; ++ic) {
        const int cb = ic >> 4, l = ic & 15;
        acc = _mm512_fmadd_ps(_mm512_set1_ps(col[(size_t)cb * H * W * BLK + l]),
                              _mm512_loadu_ps(wocb + (size_t)ic * BLK), acc);
    }
    bemit_avx512(orow + (size_t)ow * BLK, acc, bvec, relu);
}

// blocked 1x1 strided conv: register-tiled strip of CW output columns for TWO
// OC blocks. pad 0, single tap. The collapsed-to-one-tap analogue of
// bconv3x3_strip2_sN_avx512: per ic, load both OC-block weight vectors once and
// reuse each (strided) input broadcast across the CW columns AND both OC blocks
// (2*CW FMAs per broadcast) - vs the per-column kernel's 1 FMA per broadcast +
// weight load (the load-bound ~40 GFLOP/s the downsample shortcuts ran at). pad-0
// 1x1 has no out-of-bounds taps (ix = ow*stride is always in [0,W-1]), so the
// whole row tiles with no edge-column split. Input column step = stride*BLK.
template<int CW>
static inline VD_DCONV_AVX512_TARGET void bconv1x1_strip2_avx512(
        const float* xb, const float* wocb0, const float* wocb1,
        float* orow0, float* orow1,
        int W, int H, int IC, int oh, int stride, int ow0,
        const float* bvec0, const float* bvec1, bool relu) {
    __m512 a0[CW], a1[CW];
    for (int j = 0; j < CW; ++j) { a0[j] = _mm512_setzero_ps(); a1[j] = _mm512_setzero_ps(); }
    const size_t jstep = (size_t)stride * BLK;
    const int iy = oh * stride;
    const float* base = xb + (((size_t)iy * W) + (size_t)ow0 * stride) * BLK;
    for (int ic = 0; ic < IC; ++ic) {
        const int cb = ic >> 4, l = ic & 15;
        const float* inc = base + (size_t)cb * H * W * BLK + l;
        const __m512 w0 = _mm512_loadu_ps(wocb0 + (size_t)ic * BLK);
        const __m512 w1 = _mm512_loadu_ps(wocb1 + (size_t)ic * BLK);
        for (int j = 0; j < CW; ++j) {
            const __m512 b = _mm512_set1_ps(inc[(size_t)j * jstep]);
            a0[j] = _mm512_fmadd_ps(b, w0, a0[j]);
            a1[j] = _mm512_fmadd_ps(b, w1, a1[j]);
        }
    }
    for (int j = 0; j < CW; ++j) {
        bemit_avx512(orow0 + (size_t)(ow0 + j) * BLK, a0[j], bvec0, relu);
        bemit_avx512(orow1 + (size_t)(ow0 + j) * BLK, a1[j], bvec1, relu);
    }
}

// 1x1 strided 2-OC-block output-row driver. pad-0 1x1 has no edge columns, so the
// full row tiles via register-tiled strips; the partial tail steps down through
// fixed CW widths (each fully unrolled).
static inline VD_DCONV_AVX512_TARGET void bconv1x1_row2_avx512(
        const float* xb, const float* Wpack, float* y,
        int W, int H, int IC, int Wout, int Hout, int ocb0, int ocb1, int oh,
        int stride, const float* Bpack, bool relu) {
    const float* w0 = Wpack + (size_t)ocb0 * IC * BLK;
    const float* w1 = Wpack + (size_t)ocb1 * IC * BLK;
    const float* b0 = Bpack ? Bpack + (size_t)ocb0 * BLK : nullptr;
    const float* b1 = Bpack ? Bpack + (size_t)ocb1 * BLK : nullptr;
    float* orow0 = y + (((size_t)ocb0 * Hout + oh) * Wout) * BLK;
    float* orow1 = y + (((size_t)ocb1 * Hout + oh) * Wout) * BLK;
    int ow0 = 0;
    for (; ow0 + OWB2 - 1 < Wout; ow0 += OWB2)
        bconv1x1_strip2_avx512<OWB2>(xb, w0, w1, orow0, orow1, W, H, IC, oh, stride, ow0, b0, b1, relu);
    int rem = Wout - ow0;
    while (rem >= 8) { bconv1x1_strip2_avx512<8>(xb, w0, w1, orow0, orow1, W, H, IC, oh, stride, ow0, b0, b1, relu); ow0 += 8; rem -= 8; }
    while (rem >= 4) { bconv1x1_strip2_avx512<4>(xb, w0, w1, orow0, orow1, W, H, IC, oh, stride, ow0, b0, b1, relu); ow0 += 4; rem -= 4; }
    while (rem >= 2) { bconv1x1_strip2_avx512<2>(xb, w0, w1, orow0, orow1, W, H, IC, oh, stride, ow0, b0, b1, relu); ow0 += 2; rem -= 2; }
    while (rem >= 1) { bconv1x1_strip2_avx512<1>(xb, w0, w1, orow0, orow1, W, H, IC, oh, stride, ow0, b0, b1, relu); ow0 += 1; rem -= 1; }
}
#endif

// ========================================================================
// AVX2 blocked-island microkernel (nChw16c kept; the 16-OC tile is held as a
// PAIR of ymm registers - lo = OC 0..7, hi = OC 8..15). Layout, weight pack,
// reorder, state and the kh,kw,ic accumulation order are IDENTICAL to the
// AVX-512 path, so an AVX2 host that runs the blocked island is bit-identical
// to the AVX-512 island (same FMA order, just half the lane width per FMA) and
// to the scalar fallback. Runtime-selected when AVX-512 is absent/disabled.
// This is what lets an AVX2-only host run the whole ResNet/Res2Net backbone in
// nChw16c (two reorders) instead of paying the per-conv NCHW transpose.
// ========================================================================
#if defined(__AVX2__)
// AVX2 register-tile column count: 2*CW acc (lo+hi) + 2 weight + 1 bcast must
// fit the 16 ymm registers. CW=6 -> 12+3 = 15 live.
#ifndef VD_DCONV_OWB_A2
#define VD_DCONV_OWB_A2 6
#endif
constexpr int OWB_A2 = VD_DCONV_OWB_A2;

static inline void bemit_avx2(float* p, __m256 lo, __m256 hi,
                              const float* bvec, bool relu) {
    if (bvec) {
        lo = _mm256_add_ps(lo, _mm256_loadu_ps(bvec));
        hi = _mm256_add_ps(hi, _mm256_loadu_ps(bvec + 8));
    }
    if (relu) {
        const __m256 z = _mm256_setzero_ps();
        lo = _mm256_max_ps(lo, z);
        hi = _mm256_max_ps(hi, z);
    }
    _mm256_storeu_ps(p,     lo);
    _mm256_storeu_ps(p + 8, hi);
}

// 3x3 single output column (any stride; full bounds checks). 16-OC tile = 2 ymm.
static inline void bconv3x3_col_avx2(
        const float* xb, const float* wocb, float* orow,
        int W, int H, int IC, int oh, int ow, int pad, int stride,
        const float* bvec = nullptr, bool relu = false) {
    __m256 lo = _mm256_setzero_ps(), hi = _mm256_setzero_ps();
    for (int kh = 0; kh < 3; ++kh) {
        const int iy = oh * stride + kh - pad;
        if (iy < 0 || iy >= H) continue;
        for (int kw = 0; kw < 3; ++kw) {
            const int ix = ow * stride + kw - pad;
            if (ix < 0 || ix >= W) continue;
            const float* wk = wocb + ((size_t)kh * 3 + kw) * IC * BLK;
            const float* col = xb + (((size_t)iy * W) + ix) * BLK;
            for (int ic = 0; ic < IC; ++ic) {
                const int cb = ic >> 4, l = ic & 15;
                const __m256 b = _mm256_set1_ps(col[(size_t)cb * H * W * BLK + l]);
                lo = _mm256_fmadd_ps(b, _mm256_loadu_ps(wk + (size_t)ic * BLK),     lo);
                hi = _mm256_fmadd_ps(b, _mm256_loadu_ps(wk + (size_t)ic * BLK + 8), hi);
            }
        }
    }
    bemit_avx2(orow + (size_t)ow * BLK, lo, hi, bvec, relu);
}

// stride-1 interior register-tiled strip (CW output columns), fully unrolled.
template<int CW>
static inline void bconv3x3_strip_s1_avx2(
        const float* xb, const float* wocb, float* orow,
        int W, int H, int IC, int ih, int pad, int ow0,
        const float* bvec = nullptr, bool relu = false) {
    __m256 lo[CW], hi[CW];
    for (int j = 0; j < CW; ++j) { lo[j] = _mm256_setzero_ps(); hi[j] = _mm256_setzero_ps(); }
    for (int kh = 0; kh < 3; ++kh) {
        const int iy = ih + kh - pad;
        if (iy < 0 || iy >= H) continue;
        for (int kw = 0; kw < 3; ++kw) {
            const float* wk = wocb + ((size_t)kh * 3 + kw) * IC * BLK;
            const int ix0 = ow0 + kw - pad;
            const float* base = xb + (((size_t)iy * W) + ix0) * BLK;
            for (int ic = 0; ic < IC; ++ic) {
                const int cb = ic >> 4, l = ic & 15;
                const float* inc = base + (size_t)cb * H * W * BLK + l;
                const __m256 wlo = _mm256_loadu_ps(wk + (size_t)ic * BLK);
                const __m256 whi = _mm256_loadu_ps(wk + (size_t)ic * BLK + 8);
                for (int j = 0; j < CW; ++j) {
                    const __m256 b = _mm256_set1_ps(inc[(size_t)j * BLK]);
                    lo[j] = _mm256_fmadd_ps(b, wlo, lo[j]);
                    hi[j] = _mm256_fmadd_ps(b, whi, hi[j]);
                }
            }
        }
    }
    for (int j = 0; j < CW; ++j)
        bemit_avx2(orow + (size_t)(ow0 + j) * BLK, lo[j], hi[j], bvec, relu);
}

// stride-1 single-OC-block output-row driver (edges + register-tiled interior).
static inline void bconv3x3_row_s1_avx2(
        const float* xb, const float* wocb, float* orow,
        int W, int H, int IC, int Wout, int oh, int pad,
        const float* bvec, bool relu) {
    int ow = 0;
    for (; ow < pad && ow < Wout; ++ow)
        bconv3x3_col_avx2(xb, wocb, orow, W, H, IC, oh, ow, pad, 1, bvec, relu);
    const int hi = Wout - 1 - pad;
    int ow0 = std::max(pad, ow);
    for (; ow0 + OWB_A2 - 1 <= hi; ow0 += OWB_A2)
        bconv3x3_strip_s1_avx2<OWB_A2>(xb, wocb, orow, W, H, IC, oh, pad, ow0, bvec, relu);
    int rem = hi - ow0 + 1;
    while (rem >= 4) { bconv3x3_strip_s1_avx2<4>(xb, wocb, orow, W, H, IC, oh, pad, ow0, bvec, relu); ow0 += 4; rem -= 4; }
    while (rem >= 2) { bconv3x3_strip_s1_avx2<2>(xb, wocb, orow, W, H, IC, oh, pad, ow0, bvec, relu); ow0 += 2; rem -= 2; }
    while (rem >= 1) { bconv3x3_strip_s1_avx2<1>(xb, wocb, orow, W, H, IC, oh, pad, ow0, bvec, relu); ow0 += 1; rem -= 1; }
    for (int owe = std::max(Wout - pad, ow0); owe < Wout; ++owe)
        bconv3x3_col_avx2(xb, wocb, orow, W, H, IC, oh, owe, pad, 1, bvec, relu);
}

// STRIDED (stride>=2) interior register-tiled strip - same width tile as the
// stride-1 AVX2 strip, walking the input with a stride*BLK column step. Mirrors
// the AVX-512 strided strip so an AVX2-only host recovers the stride-2 downsample
// convs too (kh,kw,ic order identical -> bit-identical to the AVX-512 island).
template<int CW>
static inline void bconv3x3_strip_sN_avx2(
        const float* xb, const float* wocb, float* orow,
        int W, int H, int IC, int oh, int pad, int stride, int ow0,
        const float* bvec = nullptr, bool relu = false) {
    __m256 lo[CW], hi[CW];
    for (int j = 0; j < CW; ++j) { lo[j] = _mm256_setzero_ps(); hi[j] = _mm256_setzero_ps(); }
    const size_t jstep = (size_t)stride * BLK;
    for (int kh = 0; kh < 3; ++kh) {
        const int iy = oh * stride + kh - pad;
        if (iy < 0 || iy >= H) continue;
        for (int kw = 0; kw < 3; ++kw) {
            const float* wk = wocb + ((size_t)kh * 3 + kw) * IC * BLK;
            const int ix0 = ow0 * stride + kw - pad;
            const float* base = xb + (((size_t)iy * W) + ix0) * BLK;
            for (int ic = 0; ic < IC; ++ic) {
                const int cb = ic >> 4, l = ic & 15;
                const float* inc = base + (size_t)cb * H * W * BLK + l;
                const __m256 wlo = _mm256_loadu_ps(wk + (size_t)ic * BLK);
                const __m256 whi = _mm256_loadu_ps(wk + (size_t)ic * BLK + 8);
                for (int j = 0; j < CW; ++j) {
                    const __m256 b = _mm256_set1_ps(inc[(size_t)j * jstep]);
                    lo[j] = _mm256_fmadd_ps(b, wlo, lo[j]);
                    hi[j] = _mm256_fmadd_ps(b, whi, hi[j]);
                }
            }
        }
    }
    for (int j = 0; j < CW; ++j)
        bemit_avx2(orow + (size_t)(ow0 + j) * BLK, lo[j], hi[j], bvec, relu);
}

// strided single-OC-block row driver (edges + register-tiled strided interior).
static inline void bconv3x3_row_sN_avx2(
        const float* xb, const float* wocb, float* orow,
        int W, int H, int IC, int Wout, int oh, int pad, int stride,
        const float* bvec, bool relu) {
    int ow_lo = (pad + stride - 1) / stride;            // ceil(pad/stride)
    int ow_hi = (W - 3 + pad) / stride;                 // last fully-interior col
    if (ow_hi > Wout - 1) ow_hi = Wout - 1;
    if (ow_lo > Wout) ow_lo = Wout;
    for (int ow = 0; ow < ow_lo; ++ow)
        bconv3x3_col_avx2(xb, wocb, orow, W, H, IC, oh, ow, pad, stride, bvec, relu);
    int ow0 = ow_lo;
    for (; ow0 + OWB_A2 - 1 <= ow_hi; ow0 += OWB_A2)
        bconv3x3_strip_sN_avx2<OWB_A2>(xb, wocb, orow, W, H, IC, oh, pad, stride, ow0, bvec, relu);
    int rem = ow_hi - ow0 + 1;
    while (rem >= 4) { bconv3x3_strip_sN_avx2<4>(xb, wocb, orow, W, H, IC, oh, pad, stride, ow0, bvec, relu); ow0 += 4; rem -= 4; }
    while (rem >= 2) { bconv3x3_strip_sN_avx2<2>(xb, wocb, orow, W, H, IC, oh, pad, stride, ow0, bvec, relu); ow0 += 2; rem -= 2; }
    while (rem >= 1) { bconv3x3_strip_sN_avx2<1>(xb, wocb, orow, W, H, IC, oh, pad, stride, ow0, bvec, relu); ow0 += 1; rem -= 1; }
    for (int ow = (ow_hi + 1 > ow_lo ? ow_hi + 1 : ow_lo); ow < Wout; ++ow)
        bconv3x3_col_avx2(xb, wocb, orow, W, H, IC, oh, ow, pad, stride, bvec, relu);
}

// blocked 1x1 strided conv: one output column. 16-OC tile = 2 ymm.
static inline void bconv1x1_col_avx2(
        const float* xb, const float* wocb, float* orow,
        int W, int H, int IC, int oh, int ow, int stride,
        const float* bvec = nullptr, bool relu = false) {
    const int iy = oh * stride, ix = ow * stride;
    const float* col = xb + (((size_t)iy * W) + ix) * BLK;
    __m256 lo = _mm256_setzero_ps(), hi = _mm256_setzero_ps();
    for (int ic = 0; ic < IC; ++ic) {
        const int cb = ic >> 4, l = ic & 15;
        const __m256 b = _mm256_set1_ps(col[(size_t)cb * H * W * BLK + l]);
        lo = _mm256_fmadd_ps(b, _mm256_loadu_ps(wocb + (size_t)ic * BLK),     lo);
        hi = _mm256_fmadd_ps(b, _mm256_loadu_ps(wocb + (size_t)ic * BLK + 8), hi);
    }
    bemit_avx2(orow + (size_t)ow * BLK, lo, hi, bvec, relu);
}
#endif  // __AVX2__

// scalar fallback column for the 3x3 (works for any stride; lane-wide blocked).
static inline void bconv3x3_col_scalar(
        const float* xb, const float* wocb, float* orow,
        int W, int H, int IC, int oh, int ow, int pad, int stride, int lane,
        const float* bvec = nullptr, bool relu = false) {
    float* o = orow + (size_t)ow * lane;
    for (int l = 0; l < lane; ++l) o[l] = 0.0f;
    for (int kh = 0; kh < 3; ++kh) {
        const int iy = oh * stride + kh - pad;
        if (iy < 0 || iy >= H) continue;
        for (int kw = 0; kw < 3; ++kw) {
            const int ix = ow * stride + kw - pad;
            if (ix < 0 || ix >= W) continue;
            const float* wk = wocb + ((size_t)kh * 3 + kw) * IC * lane;
            const float* col = xb + (((size_t)iy * W) + ix) * lane;
            for (int ic = 0; ic < IC; ++ic) {
                const int cb = ic / lane, ll = ic % lane;
                const float s = col[(size_t)cb * H * W * lane + ll];
                const float* wv = wk + (size_t)ic * lane;
                for (int l = 0; l < lane; ++l) o[l] += s * wv[l];
            }
        }
    }
    for (int l = 0; l < lane; ++l) {
        if (bvec) o[l] += bvec[l];
        if (relu && o[l] < 0.0f) o[l] = 0.0f;
    }
}
static inline void bconv1x1_col_scalar(
        const float* xb, const float* wocb, float* orow,
        int W, int H, int IC, int oh, int ow, int stride, int lane,
        const float* bvec = nullptr, bool relu = false) {
    const int iy = oh * stride, ix = ow * stride;
    const float* col = xb + (((size_t)iy * W) + ix) * lane;
    float* o = orow + (size_t)ow * lane;
    for (int l = 0; l < lane; ++l) o[l] = 0.0f;
    for (int ic = 0; ic < IC; ++ic) {
        const int cb = ic / lane, ll = ic % lane;
        const float s = col[(size_t)cb * H * W * lane + ll];
        const float* wv = wocb + (size_t)ic * lane;
        for (int l = 0; l < lane; ++l) o[l] += s * wv[l];
    }
    for (int l = 0; l < lane; ++l) {
        if (bvec) o[l] += bvec[l];
        if (relu && o[l] < 0.0f) o[l] = 0.0f;
    }
}

// Blocked 3x3 conv compute: blocked in -> blocked out. dst ne = [16,Wout,Hout,OCB].
static void bconv3x3_compute(ggml_tensor* dst, int ith, int nth, void* userdata) {
    DConvState* st = (DConvState*)userdata;
    const ggml_tensor* xt = dst->src[0];   // blocked input [16,W,H,CB]
    const ggml_tensor* wt = dst->src[1];
    const float* xb = (const float*)xt->data;
    float* y = (float*)dst->data;
    std::call_once(st->once, [&] {
        build_pack(st, (const float*)wt->data);
        if (st->has_bias) build_pack_bias(st, (const float*)dst->src[2]->data);
    });

    const int W = st->W, H = st->H, IC = st->IC, OC = st->OC;
    const int pad = st->pad, stride = st->stride, lane = st->lane, OCB = st->OCB;
    const int Wout = (int)dst->ne[1], Hout = (int)dst->ne[2];
    const float* Wpack = st->Wpack.data();
    const float* Bpack = st->has_bias ? st->Bpack.data() : nullptr;
    const bool relu = st->do_relu;
    (void)OC;
    ProfTimer _pt(st, Wout, Hout, /*kind=*/0, nth);

#if defined(VD_DCONV_HAVE_AVX512)
    if (lane == 16 && stride == 1 && dconv_use_avx512()) {
        // Work items = (OC-block PAIR, oh): each pass shares one (width-strided)
        // blocked-input broadcast across two OC blocks. Odd trailing block falls
        // back to the single-OC strip path.
        const int npairs = (OCB + 1) / 2;
        const int64_t total = (int64_t)npairs * Hout;
        const int64_t beg = total * ith / nth, end = total * (ith + 1) / nth;
        for (int64_t idx = beg; idx < end; ++idx) {
            const int p  = (int)(idx / Hout);
            const int oh = (int)(idx % Hout);
            const int ocb0 = 2 * p, ocb1 = 2 * p + 1;
            if (ocb1 < OCB) {
                bconv3x3_row2_s1_avx512(xb, Wpack, y, W, H, IC, Wout, Hout, ocb0, ocb1, oh, pad, Bpack, relu);
            } else {
                float* orow = y + (((size_t)ocb0 * Hout + oh) * Wout) * lane;
                const float* wocb = Wpack + (size_t)ocb0 * 9 * IC * lane;
                const float* bvec = Bpack ? Bpack + (size_t)ocb0 * lane : nullptr;
                int ow = 0;
                for (; ow < pad && ow < Wout; ++ow)
                    bconv3x3_col_avx512(xb, wocb, orow, W, H, IC, oh, ow, pad, 1, bvec, relu);
                const int hi = Wout - 1 - pad;
                int ow0 = std::max(pad, ow);
                for (; ow0 + OWB - 1 <= hi; ow0 += OWB)
                    bconv3x3_strip_s1_avx512<OWB>(xb, wocb, orow, W, H, IC, oh, pad, ow0, bvec, relu);
                int rem = hi - ow0 + 1;
                while (rem >= 8) { bconv3x3_strip_s1_avx512<8>(xb, wocb, orow, W, H, IC, oh, pad, ow0, bvec, relu); ow0 += 8; rem -= 8; }
                while (rem >= 4) { bconv3x3_strip_s1_avx512<4>(xb, wocb, orow, W, H, IC, oh, pad, ow0, bvec, relu); ow0 += 4; rem -= 4; }
                while (rem >= 2) { bconv3x3_strip_s1_avx512<2>(xb, wocb, orow, W, H, IC, oh, pad, ow0, bvec, relu); ow0 += 2; rem -= 2; }
                while (rem >= 1) { bconv3x3_strip_s1_avx512<1>(xb, wocb, orow, W, H, IC, oh, pad, ow0, bvec, relu); ow0 += 1; rem -= 1; }
                for (int owe = std::max(Wout - pad, ow0); owe < Wout; ++owe)
                    bconv3x3_col_avx512(xb, wocb, orow, W, H, IC, oh, owe, pad, 1, bvec, relu);
            }
        }
        return;
    }
    if (lane == 16 && stride > 1 && dconv_use_avx512()) {
        // STRIDED (downsample) 3x3: register-tiled, 2 OC blocks per (strided)
        // input broadcast - same OC-in-lanes width tile as stride-1, walking the
        // input with a stride*BLK column step. Replaces the old per-column path
        // (~42 GFLOP/s) that had no tiling. Odd trailing OC block (rare; never
        // for WeSpeaker's even 64/128/256) falls back to the per-column kernel.
        const int npairs = (OCB + 1) / 2;
        const int64_t total = (int64_t)npairs * Hout;
        const int64_t beg = total * ith / nth, end = total * (ith + 1) / nth;
        for (int64_t idx = beg; idx < end; ++idx) {
            const int p  = (int)(idx / Hout);
            const int oh = (int)(idx % Hout);
            const int ocb0 = 2 * p, ocb1 = 2 * p + 1;
            if (ocb1 < OCB) {
                bconv3x3_row2_sN_avx512(xb, Wpack, y, W, H, IC, Wout, Hout, ocb0, ocb1, oh, pad, stride, Bpack, relu);
            } else {
                float* orow = y + (((size_t)ocb0 * Hout + oh) * Wout) * lane;
                const float* wocb = Wpack + (size_t)ocb0 * 9 * IC * lane;
                const float* bvec = Bpack ? Bpack + (size_t)ocb0 * lane : nullptr;
                for (int ow = 0; ow < Wout; ++ow)
                    bconv3x3_col_avx512(xb, wocb, orow, W, H, IC, oh, ow, pad, stride, bvec, relu);
            }
        }
        return;
    }
#endif
#if defined(__AVX2__)
    if (lane == 16 && stride == 1) {
        // AVX2 nChw16c stride-1: single-OC-block (16 ch = 2 ymm) register-tiled
        // rows. Work items = (ocb, oh). Same kh,kw,ic order as the zmm path.
        const int64_t total = (int64_t)OCB * Hout;
        const int64_t beg = total * ith / nth, end = total * (ith + 1) / nth;
        for (int64_t idx = beg; idx < end; ++idx) {
            const int ocb = (int)(idx / Hout);
            const int oh  = (int)(idx % Hout);
            float* orow = y + (((size_t)ocb * Hout + oh) * Wout) * lane;
            const float* wocb = Wpack + (size_t)ocb * 9 * IC * lane;
            const float* bvec = Bpack ? Bpack + (size_t)ocb * lane : nullptr;
            bconv3x3_row_s1_avx2(xb, wocb, orow, W, H, IC, Wout, oh, pad, bvec, relu);
        }
        return;
    }
    if (lane == 16 && stride > 1) {
        // AVX2 strided (downsample) 3x3: register-tiled rows (replaces the old
        // per-column AVX2 path). Same width tile, stride*BLK input step.
        const int64_t total = (int64_t)OCB * Hout;
        const int64_t beg = total * ith / nth, end = total * (ith + 1) / nth;
        for (int64_t idx = beg; idx < end; ++idx) {
            const int ocb = (int)(idx / Hout);
            const int oh  = (int)(idx % Hout);
            float* orow = y + (((size_t)ocb * Hout + oh) * Wout) * lane;
            const float* wocb = Wpack + (size_t)ocb * 9 * IC * lane;
            const float* bvec = Bpack ? Bpack + (size_t)ocb * lane : nullptr;
            bconv3x3_row_sN_avx2(xb, wocb, orow, W, H, IC, Wout, oh, pad, stride, bvec, relu);
        }
        return;
    }
#endif
    const int64_t total = (int64_t)OCB * Hout;
    const int64_t beg = total * ith / nth, end = total * (ith + 1) / nth;
    for (int64_t idx = beg; idx < end; ++idx) {
        const int ocb = (int)(idx / Hout);
        const int oh  = (int)(idx % Hout);
        float* orow = y + (((size_t)ocb * Hout + oh) * Wout) * lane;
        const float* wocb = Wpack + (size_t)ocb * 9 * IC * lane;
        const float* bvec = Bpack ? Bpack + (size_t)ocb * lane : nullptr;
#if defined(VD_DCONV_HAVE_AVX512)
        if (lane == 16 && dconv_use_avx512()) {   // stride > 1
            for (int ow = 0; ow < Wout; ++ow)
                bconv3x3_col_avx512(xb, wocb, orow, W, H, IC, oh, ow, pad, stride, bvec, relu);
            continue;
        }
#endif
#if defined(__AVX2__)
        if (lane == 16) {   // stride > 1, AVX2
            for (int ow = 0; ow < Wout; ++ow)
                bconv3x3_col_avx2(xb, wocb, orow, W, H, IC, oh, ow, pad, stride, bvec, relu);
            continue;
        }
#endif
        for (int ow = 0; ow < Wout; ++ow)
            bconv3x3_col_scalar(xb, wocb, orow, W, H, IC, oh, ow, pad, stride, lane, bvec, relu);
    }
}

// Blocked 1x1 strided conv compute. dst ne = [16,Wout,Hout,OCB].
static void bconv1x1_compute(ggml_tensor* dst, int ith, int nth, void* userdata) {
    DConvState* st = (DConvState*)userdata;
    const ggml_tensor* xt = dst->src[0];
    const ggml_tensor* wt = dst->src[1];
    const float* xb = (const float*)xt->data;
    float* y = (float*)dst->data;
    std::call_once(st->once, [&] {
        build_pack_1x1(st, (const float*)wt->data);
        if (st->has_bias) build_pack_bias(st, (const float*)dst->src[2]->data);
    });

    const int W = st->W, H = st->H, IC = st->IC;
    const int stride = st->stride, lane = st->lane, OCB = st->OCB;
    const int Wout = (int)dst->ne[1], Hout = (int)dst->ne[2];
    const float* Wpack = st->Wpack.data();
    const float* Bpack = st->has_bias ? st->Bpack.data() : nullptr;
    const bool relu = st->do_relu;
    ProfTimer _pt(st, Wout, Hout, /*kind=*/1, nth);

    // A/B: VD_DCONV_NOTILE1X1=1 forces the legacy per-column 1x1 path (the
    // ~40-GFLOP/s baseline) for same-session before/after measurement.
    static const bool no_tile_1x1 = [] {
        const char* e = std::getenv("VD_DCONV_NOTILE1X1");
        return e && e[0] && e[0] != '0';
    }();
#if defined(VD_DCONV_HAVE_AVX512)
    if (lane == 16 && dconv_use_avx512() && !no_tile_1x1) {
        // Register-tiled 2-OC-block path (mirrors the strided-3x3 win): work items
        // = (OC-block PAIR, oh); each strided input broadcast feeds CW columns x 2
        // OC blocks. Odd trailing block (rare; never for WeSpeaker's even widths)
        // falls back to the per-column kernel.
        const int npairs = (OCB + 1) / 2;
        const int64_t total = (int64_t)npairs * Hout;
        const int64_t beg = total * ith / nth, end = total * (ith + 1) / nth;
        for (int64_t idx = beg; idx < end; ++idx) {
            const int p  = (int)(idx / Hout);
            const int oh = (int)(idx % Hout);
            const int ocb0 = 2 * p, ocb1 = 2 * p + 1;
            if (ocb1 < OCB) {
                bconv1x1_row2_avx512(xb, Wpack, y, W, H, IC, Wout, Hout, ocb0, ocb1, oh, stride, Bpack, relu);
            } else {
                float* orow = y + (((size_t)ocb0 * Hout + oh) * Wout) * lane;
                const float* wocb = Wpack + (size_t)ocb0 * IC * lane;
                const float* bvec = Bpack ? Bpack + (size_t)ocb0 * lane : nullptr;
                for (int ow = 0; ow < Wout; ++ow)
                    bconv1x1_col_avx512(xb, wocb, orow, W, H, IC, oh, ow, stride, bvec, relu);
            }
        }
        return;
    }
#endif
    const int64_t total = (int64_t)OCB * Hout;
    const int64_t beg = total * ith / nth, end = total * (ith + 1) / nth;
    for (int64_t idx = beg; idx < end; ++idx) {
        const int ocb = (int)(idx / Hout);
        const int oh  = (int)(idx % Hout);
        float* orow = y + (((size_t)ocb * Hout + oh) * Wout) * lane;
        const float* wocb = Wpack + (size_t)ocb * IC * lane;
        const float* bvec = Bpack ? Bpack + (size_t)ocb * lane : nullptr;
        for (int ow = 0; ow < Wout; ++ow) {
#if defined(__AVX2__)
            if (lane == 16) { bconv1x1_col_avx2(xb, wocb, orow, W, H, IC, oh, ow, stride, bvec, relu); continue; }
#endif
            bconv1x1_col_scalar(xb, wocb, orow, W, H, IC, oh, ow, stride, lane, bvec, relu);
        }
    }
}

// reorder NCHW [W,H,C,1] -> blocked [16,W,H,CB] (zero-pads channels C..CB*16-1).
static void reorder_in_compute(ggml_tensor* dst, int ith, int nth, void*) {
    const ggml_tensor* xt = dst->src[0];
    const float* x = (const float*)xt->data;
    float* y = (float*)dst->data;
    const int W = (int)xt->ne[0], H = (int)xt->ne[1], C = (int)xt->ne[2];
    const int CB = (int)dst->ne[3];
    const size_t HW = (size_t)H * W;
    const int64_t total = (int64_t)CB * H;
    const int64_t beg = total * ith / nth, end = total * (ith + 1) / nth;
    for (int64_t idx = beg; idx < end; ++idx) {
        const int cb = (int)(idx / H), h = (int)(idx % H);
        for (int w = 0; w < W; ++w) {
            float* o = y + (((size_t)cb * H + h) * W + w) * BLK;
            for (int l = 0; l < BLK; ++l) {
                const int c = cb * BLK + l;
                o[l] = (c < C) ? x[(size_t)c * HW + (size_t)h * W + w] : 0.0f;
            }
        }
    }
}

// reorder blocked [16,W,H,CB] -> NCHW [W,H,C,1] (drops padding lanes).
static void reorder_out_compute(ggml_tensor* dst, int ith, int nth, void*) {
    const ggml_tensor* xt = dst->src[0];
    const float* xb = (const float*)xt->data;
    float* y = (float*)dst->data;
    const int W = (int)dst->ne[0], H = (int)dst->ne[1], C = (int)dst->ne[2];
    const size_t HW = (size_t)H * W;
    const int64_t beg = (int64_t)C * ith / nth, end = (int64_t)C * (ith + 1) / nth;
    for (int64_t c = beg; c < end; ++c) {
        const int cb = (int)c >> 4, l = (int)c & 15;
        float* yc = y + (size_t)c * HW;
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w)
                yc[(size_t)h * W + w] = xb[(((size_t)cb * H + h) * W + w) * BLK + l];
    }
}

// blocked per-channel bias add (channel c = cb*16+l; skip padding lanes c>=C).
static void bbias_compute(ggml_tensor* dst, int ith, int nth, void*) {
    const ggml_tensor* xt = dst->src[0];
    const ggml_tensor* bt = dst->src[1];
    const float* xb = (const float*)xt->data;
    const float* b = (const float*)bt->data;
    float* y = (float*)dst->data;
    const int W = (int)dst->ne[1], H = (int)dst->ne[2], CB = (int)dst->ne[3];
    const int C = (int)bt->ne[0];
    const size_t HW = (size_t)H * W;
    const int64_t beg = (int64_t)CB * ith / nth, end = (int64_t)CB * (ith + 1) / nth;
    for (int64_t cb = beg; cb < end; ++cb) {
        for (size_t p = 0; p < HW; ++p) {
            float* o = y + ((size_t)cb * HW + p) * BLK;
            const float* in = xb + ((size_t)cb * HW + p) * BLK;
            for (int l = 0; l < BLK; ++l) {
                const int c = (int)cb * BLK + l;
                o[l] = in[l] + ((c < C) ? b[c] : 0.0f);
            }
        }
    }
}

// blocked ReLU (whole buffer, no channel awareness needed).
static void brelu_compute(ggml_tensor* dst, int ith, int nth, void*) {
    const ggml_tensor* xt = dst->src[0];
    const float* x = (const float*)xt->data;
    float* y = (float*)dst->data;
    const int64_t n = ggml_nelements(dst);
    const int64_t beg = n * ith / nth, end = n * (ith + 1) / nth;
    for (int64_t i = beg; i < end; ++i) y[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

// blocked elementwise add (residual): dst = src0 + src1.
static void badd_compute(ggml_tensor* dst, int ith, int nth, void*) {
    const float* a = (const float*)dst->src[0]->data;
    const float* b = (const float*)dst->src[1]->data;
    float* y = (float*)dst->data;
    const int64_t n = ggml_nelements(dst);
    const int64_t beg = n * ith / nth, end = n * (ith + 1) / nth;
    for (int64_t i = beg; i < end; ++i) y[i] = a[i] + b[i];
}

// fused residual add + ReLU (the BasicBlock tail): dst = max(0, src0 + src1).
// One pass instead of a blocked_add then a blocked_relu.
static void badd_relu_compute(ggml_tensor* dst, int ith, int nth, void*) {
    const float* a = (const float*)dst->src[0]->data;
    const float* b = (const float*)dst->src[1]->data;
    float* y = (float*)dst->data;
    const int64_t n = ggml_nelements(dst);
    const int64_t beg = n * ith / nth, end = n * (ith + 1) / nth;
    for (int64_t i = beg; i < end; ++i) { float v = a[i] + b[i]; y[i] = v > 0.0f ? v : 0.0f; }
}

struct StateKey {
    const void* wdata; int W, H, IC, OC, N, pad, stride, kind, flags;
    bool operator==(const StateKey& o) const {
        return wdata == o.wdata && W == o.W && H == o.H && IC == o.IC &&
               OC == o.OC && N == o.N && pad == o.pad &&
               stride == o.stride && kind == o.kind && flags == o.flags;
    }
};
struct StateKeyHash {
    size_t operator()(const StateKey& k) const {
        size_t h = (size_t)k.wdata;
        auto mix = [&h](int v) { h = h * 1000003u + (size_t)(uint32_t)v; };
        mix(k.W); mix(k.H); mix(k.IC); mix(k.OC); mix(k.N); mix(k.pad);
        mix(k.stride); mix(k.kind); mix(k.flags);
        return h;
    }
};
static std::mutex g_states_mtx;
static std::unordered_map<StateKey, DConvState*, StateKeyHash> g_states;

static int select_lane() {
#if defined(VD_DCONV_HAVE_AVX512)
    if (dconv_use_avx512()) return 16;
#endif
#if defined(__AVX2__)
    return 8;
#else
    return 8;   // scalar still uses an 8-wide pack
#endif
}

static DConvState* get_state(ggml_tensor* w, ggml_tensor* x, int pad) {
    const int W = (int)x->ne[0], H = (int)x->ne[1], IC = (int)x->ne[2], N = (int)x->ne[3];
    const int OC = (int)w->ne[3];
    StateKey key{ w->data, W, H, IC, OC, N, pad, 1, 0, 0 };
    std::lock_guard<std::mutex> lk(g_states_mtx);
    auto it = g_states.find(key);
    if (it != g_states.end()) return it->second;
    DConvState* st = new DConvState();
    st->W = W; st->H = H; st->IC = IC; st->OC = OC; st->N = N; st->pad = pad;
    st->lane = select_lane();
    st->OCB = (OC + st->lane - 1) / st->lane;
    st->wdata = w->data;
    g_states[key] = st;
    return st;
}

// Blocked-island state: input is the blocked buffer [16,W,H,CB] so W,H come from
// ne[1],ne[2]; IC is the TRUE input channel count (the weight's ne[2]); lane is
// ALWAYS 16 (nChw16c). kind 0 = 3x3, 1 = 1x1. flags encodes fused bias/relu so a
// reused weight with a different epilogue gets its own packed state.
static DConvState* get_state_blocked(ggml_tensor* w, ggml_tensor* xb,
                                     int pad, int stride, int kind,
                                     bool has_bias, bool do_relu) {
    const int W = (int)xb->ne[1], H = (int)xb->ne[2];
    const int IC = (int)w->ne[2], OC = (int)w->ne[3], N = 1;
    const int flags = (has_bias ? 1 : 0) | (do_relu ? 2 : 0);
    StateKey key{ w->data, W, H, IC, OC, N, pad, stride, kind, flags };
    std::lock_guard<std::mutex> lk(g_states_mtx);
    auto it = g_states.find(key);
    if (it != g_states.end()) return it->second;
    DConvState* st = new DConvState();
    st->W = W; st->H = H; st->IC = IC; st->OC = OC; st->N = N;
    st->pad = pad; st->stride = stride; st->kind = kind;
    st->has_bias = has_bias; st->do_relu = do_relu;
    st->lane = BLK;
    st->OCB = (OC + BLK - 1) / BLK;
    st->wdata = w->data;
    g_states[key] = st;
    return st;
}

} // namespace

ggml_tensor* directconv_conv3x3(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x, int pad) {
    const int OC = (int)w->ne[3];
    const int N  = (int)x->ne[3];
    const int Wout = (int)x->ne[0] + 2 * pad - 2;
    const int Hout = (int)x->ne[1] + 2 * pad - 2;
    DConvState* st = get_state(w, x, pad);
    ggml_tensor* args[2] = { x, w };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, Wout, Hout, OC, N,
                          args, 2, dconv_compute, GGML_N_TASKS_MAX, st);
}

// ---- blocked-island public ops --------------------------------------------

bool directconv_blocked_available() {
    // The blocked island now has BOTH a zmm (nChw16c, AVX-512) and a ymm
    // (nChw16c held as 2 ymm, AVX2) vectorized fast path. Either one beats the
    // per-conv directconv on WeSpeaker: blocked-AVX2 measured ~2.2x faster @1t
    // (105 vs 232 ms) and ~2.3x @8t (14.6 vs 33.5 ms) vs the AVX2 per-conv
    // directconv, at cosine 1.000000 (bit-identical FMA order). So enable the
    // blocked default whenever EITHER ISA path will run at runtime; only a
    // pure-scalar host (no AVX2) keeps the per-conv directconv to avoid a scalar
    // blocked regression.
#if defined(VD_DCONV_HAVE_AVX512)
    if (dconv_use_avx512()) return true;
#endif
#if defined(__AVX2__)
    return true;
#else
    return false;
#endif
}

ggml_tensor* blocked_reorder_in(ggml_context* ctx, ggml_tensor* x_nchw) {
    const int W = (int)x_nchw->ne[0], H = (int)x_nchw->ne[1], C = (int)x_nchw->ne[2];
    const int CB = (C + BLK - 1) / BLK;
    ggml_tensor* args[1] = { x_nchw };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, BLK, W, H, CB,
                          args, 1, reorder_in_compute, GGML_N_TASKS_MAX, nullptr);
}

ggml_tensor* blocked_reorder_out(ggml_context* ctx, ggml_tensor* xb, int C) {
    const int W = (int)xb->ne[1], H = (int)xb->ne[2];
    ggml_tensor* args[1] = { xb };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, W, H, C, 1,
                          args, 1, reorder_out_compute, GGML_N_TASKS_MAX, nullptr);
}

ggml_tensor* blocked_conv3x3(ggml_context* ctx, ggml_tensor* w, ggml_tensor* xb,
                             int pad, int stride, ggml_tensor* bias, bool do_relu) {
    const int W = (int)xb->ne[1], H = (int)xb->ne[2];
    const int OC = (int)w->ne[3];
    const int Wout = (W + 2 * pad - 3) / stride + 1;
    const int Hout = (H + 2 * pad - 3) / stride + 1;
    const int OCB = (OC + BLK - 1) / BLK;
    DConvState* st = get_state_blocked(w, xb, pad, stride, /*kind=*/0,
                                       bias != nullptr, do_relu);
    ggml_tensor* args[3] = { xb, w, bias };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, BLK, Wout, Hout, OCB,
                          args, bias ? 3 : 2, bconv3x3_compute, GGML_N_TASKS_MAX, st);
}

ggml_tensor* blocked_conv1x1(ggml_context* ctx, ggml_tensor* w, ggml_tensor* xb,
                             int stride, ggml_tensor* bias, bool do_relu) {
    const int W = (int)xb->ne[1], H = (int)xb->ne[2];
    const int OC = (int)w->ne[3];
    const int Wout = (W - 1) / stride + 1;
    const int Hout = (H - 1) / stride + 1;
    const int OCB = (OC + BLK - 1) / BLK;
    DConvState* st = get_state_blocked(w, xb, /*pad=*/0, stride, /*kind=*/1,
                                       bias != nullptr, do_relu);
    ggml_tensor* args[3] = { xb, w, bias };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, BLK, Wout, Hout, OCB,
                          args, bias ? 3 : 2, bconv1x1_compute, GGML_N_TASKS_MAX, st);
}

ggml_tensor* blocked_bias(ggml_context* ctx, ggml_tensor* xb, ggml_tensor* bias) {
    ggml_tensor* args[2] = { xb, bias };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, xb->ne[0], xb->ne[1], xb->ne[2], xb->ne[3],
                          args, 2, bbias_compute, GGML_N_TASKS_MAX, nullptr);
}

ggml_tensor* blocked_relu(ggml_context* ctx, ggml_tensor* xb) {
    ggml_tensor* args[1] = { xb };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, xb->ne[0], xb->ne[1], xb->ne[2], xb->ne[3],
                          args, 1, brelu_compute, GGML_N_TASKS_MAX, nullptr);
}

ggml_tensor* blocked_add(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b) {
    ggml_tensor* args[2] = { a, b };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, a->ne[0], a->ne[1], a->ne[2], a->ne[3],
                          args, 2, badd_compute, GGML_N_TASKS_MAX, nullptr);
}

ggml_tensor* blocked_add_relu(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b) {
    ggml_tensor* args[2] = { a, b };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, a->ne[0], a->ne[1], a->ne[2], a->ne[3],
                          args, 2, badd_relu_compute, GGML_N_TASKS_MAX, nullptr);
}

} // namespace vd
