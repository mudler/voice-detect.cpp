#include "winograd.hpp"
#include <algorithm>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

// AVX-512 winograd-domain GEMM microkernel availability + runtime dispatch.
//
// voice-detect.cpp's winograd GEMM carries an AVX-512 zmm path. It ships in the
// default portable build (GGML_NATIVE=OFF, winograd.cpp compiled -mavx2 -mfma,
// NO global -mavx512f) as a RUNTIME-CPUID-DISPATCHED path: the zmm functions
// carry __attribute__((target("avx512f,avx512bw,avx512vl"))) (GCC/Clang function
// multiversioning), so the compiler emits their zmm code even though this TU is
// AVX2-only (VOICEDETECT_WINO_AVX512 is defined by a CMake probe). wino_gemm_block
// / wino_gemv call them ONLY when __builtin_cpu_supports reports AVX-512 at run
// time, so AVX2-only CPUs never execute a zmm instruction -- no SIGILL, one
// portable binary. A global -mavx512f build (e.g. GGML_NATIVE=ON on an AVX-512
// host) defines __AVX512F__ instead; then the ISA is already on and the same
// kernel is compiled without the per-function attribute and still runtime-gated.
// Mirrors face-detect.cpp/src/winograd.cpp (commit 7ae5c4d).
#if defined(__AVX512F__)
#  define VD_WINO_HAVE_AVX512 1
#  define VD_WINO_AVX512_TARGET   /* ISA already global; no per-fn attribute */
#elif defined(VOICEDETECT_WINO_AVX512) && defined(__AVX2__)
#  define VD_WINO_HAVE_AVX512 1
#  define VD_WINO_AVX512_TARGET __attribute__((target("avx512f,avx512bw,avx512vl")))
#endif

namespace vd {
namespace {

#if defined(VD_WINO_HAVE_AVX512)
// Cached decision: run the AVX-512 GEMM microkernel? True only when the running
// CPU advertises avx512f+bw+vl (so the zmm code never executes on an AVX2-only
// host) AND the VOICEDETECT_DISABLE_AVX512 test hook is unset (forcing it lets us
// exercise + parity-check the AVX2 fallback on an AVX-512 box). __builtin_cpu_*
// reads the CPUID feature bits the C runtime fills in before main; checked once.
// VOICEDETECT_WINO_VERBOSE prints the one-time selection for ship-safety proofs.
static bool wino_use_avx512() {
    static const bool use512 = [] {
        const char* off = std::getenv("VOICEDETECT_DISABLE_AVX512");
        const bool disabled = off && off[0] != '\0' && off[0] != '0';
        const bool supported = __builtin_cpu_supports("avx512f")
                            && __builtin_cpu_supports("avx512bw")
                            && __builtin_cpu_supports("avx512vl");
        const bool sel = supported && !disabled;
        if (std::getenv("VOICEDETECT_WINO_VERBOSE"))
            std::fprintf(stderr,
                         "[winograd] GEMM microkernel: %s (avx512 supported=%d, "
                         "disabled=%d)\n", sel ? "AVX-512" : "AVX2",
                         (int)supported, (int)disabled);
        return sel;
    }();
    return use512;
}
#endif

// ========================================================================
// Mode selection (VOICEDETECT_WINO env): which Winograd algorithm + kernel.
//   "f2"  : F(2x2,3x3), per-tile GEMV
//   "f2b" : F(2x2,3x3), blocked GEMM over a block of tiles  <-- auto default
//           (parity-identical to f2; reuses each U row across TB tiles)
//   "f4"  : F(4x4,3x3), blocked GEMM (4x fewer mults vs direct; less accurate)
// Parity: f2/f2b are exact (halves+ints, max|d|~1e-5). f4 uses 1/6,1/24
// fractions so it is less accurate -- gated by the suite.
// Ported from depth-anything.cpp/src/winograd.cpp; the SIMD inner kernels are
// re-targeted to AVX2 (GGML_NATIVE=OFF here -> no AVX-512).
// ========================================================================
enum class Mode { F2, F2B, F4 };

static Mode parse_mode() {
    const char* m = std::getenv("VOICEDETECT_WINO");
    if (m) {
        if (!std::strcmp(m, "f2"))  return Mode::F2;
        if (!std::strcmp(m, "f2b")) return Mode::F2B;
        if (!std::strcmp(m, "f4"))  return Mode::F4;
    }
    return Mode::F2B;   // auto default (fastest parity-exact variant)
}

// Tile-block width for the blocked GEMM microkernel (number of tiles batched
// per winograd-domain multiply). 8 keeps the AVX2 accumulators in ymm registers
// while amortizing each U-row load across 8 tiles.
constexpr int TB = 8;

// ------------------------------------------------------------------------
// F(2x2,3x3) transforms (exact: halves + integers).
//   B^T = [[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]]
//   G   = [[1,0,0],[.5,.5,.5],[.5,-.5,.5],[0,0,1]]
//   A^T = [[1,1,1,0],[0,1,-1,-1]]
// ------------------------------------------------------------------------
struct F2Policy {
    static constexpr int IT = 4, OT = 2, P = 16;   // input tile / output tile / positions

    // U = G g G^T, 3x3 -> 4x4 (u[16]).
    static void filt(const float g[9], float u[16]) {
        float Gg[4][3];
        for (int j = 0; j < 3; ++j) {
            float c0 = g[0*3 + j], c1 = g[1*3 + j], c2 = g[2*3 + j];
            Gg[0][j] = c0;
            Gg[1][j] = 0.5f * (c0 + c1 + c2);
            Gg[2][j] = 0.5f * (c0 - c1 + c2);
            Gg[3][j] = c2;
        }
        for (int i = 0; i < 4; ++i) {
            float c0 = Gg[i][0], c1 = Gg[i][1], c2 = Gg[i][2];
            u[i*4 + 0] = c0;
            u[i*4 + 1] = 0.5f * (c0 + c1 + c2);
            u[i*4 + 2] = 0.5f * (c0 - c1 + c2);
            u[i*4 + 3] = c2;
        }
    }
    // V = B^T d B, 4x4 -> 4x4 (v[16]).
    static void inp(const float d[16], float v[16]) {
        float m[16];
        for (int j = 0; j < 4; ++j) {
            float r0 = d[0*4 + j], r1 = d[1*4 + j], r2 = d[2*4 + j], r3 = d[3*4 + j];
            m[0*4 + j] = r0 - r2;
            m[1*4 + j] = r1 + r2;
            m[2*4 + j] = r2 - r1;
            m[3*4 + j] = r1 - r3;
        }
        for (int i = 0; i < 4; ++i) {
            float c0 = m[i*4 + 0], c1 = m[i*4 + 1], c2 = m[i*4 + 2], c3 = m[i*4 + 3];
            v[i*4 + 0] = c0 - c2;
            v[i*4 + 1] = c1 + c2;
            v[i*4 + 2] = c2 - c1;
            v[i*4 + 3] = c1 - c3;
        }
    }
    // Y = A^T m A, 4x4 -> 2x2 (y[4]).
    static void outp(const float m[16], float y[4]) {
        float p[8];
        for (int j = 0; j < 4; ++j) {
            float r0 = m[0*4 + j], r1 = m[1*4 + j], r2 = m[2*4 + j], r3 = m[3*4 + j];
            p[0*4 + j] = r0 + r1 + r2;
            p[1*4 + j] = r1 - r2 - r3;
        }
        for (int i = 0; i < 2; ++i) {
            float c0 = p[i*4 + 0], c1 = p[i*4 + 1], c2 = p[i*4 + 2], c3 = p[i*4 + 3];
            y[i*2 + 0] = c0 + c1 + c2;
            y[i*2 + 1] = c1 - c2 - c3;
        }
    }
};

// ------------------------------------------------------------------------
// F(4x4,3x3) transforms (Lavin & Gray). 6x6 input tile -> 4x4 output, 36
// winograd positions. Uses 1/6 and 1/24, so float32 accuracy is lower than F2.
// ------------------------------------------------------------------------
struct F4Policy {
    static constexpr int IT = 6, OT = 4, P = 36;

    static inline void Brow(const float x[6], float r[6]) {
        r[0] = 4.0f*x[0] - 5.0f*x[2] + x[4];
        r[1] = -4.0f*x[1] - 4.0f*x[2] + x[3] + x[4];
        r[2] = 4.0f*x[1] - 4.0f*x[2] - x[3] + x[4];
        r[3] = -2.0f*x[1] - x[2] + 2.0f*x[3] + x[4];
        r[4] = 2.0f*x[1] - x[2] - 2.0f*x[3] + x[4];
        r[5] = 4.0f*x[1] - 5.0f*x[3] + x[5];
    }
    static inline void Grow(const float y[3], float u[6]) {
        const float a = y[0], b = y[1], c = y[2];
        u[0] = 0.25f * a;
        u[1] = -(a + b + c) * (1.0f/6.0f);
        u[2] = (-a + b - c) * (1.0f/6.0f);
        u[3] = a*(1.0f/24.0f) + b*(1.0f/12.0f) + c*(1.0f/6.0f);
        u[4] = a*(1.0f/24.0f) - b*(1.0f/12.0f) + c*(1.0f/6.0f);
        u[5] = c;
    }
    static inline void Arow(const float m[6], float o[4]) {
        o[0] = m[0] + m[1] + m[2] + m[3] + m[4];
        o[1] = m[1] - m[2] + 2.0f*m[3] - 2.0f*m[4];
        o[2] = m[1] + m[2] + 4.0f*m[3] + 4.0f*m[4];
        o[3] = m[1] - m[2] + 8.0f*m[3] - 8.0f*m[4] + m[5];
    }

    // U = G g G^T, 3x3 -> 6x6 (u[36]).
    static void filt(const float g[9], float u[36]) {
        float Gg[6][3];
        for (int j = 0; j < 3; ++j) {
            float col[3] = { g[0*3 + j], g[1*3 + j], g[2*3 + j] };
            float out[6]; Grow(col, out);
            for (int i = 0; i < 6; ++i) Gg[i][j] = out[i];
        }
        for (int i = 0; i < 6; ++i) {
            float out[6]; Grow(Gg[i], out);
            for (int k = 0; k < 6; ++k) u[i*6 + k] = out[k];
        }
    }
    // V = B^T d B, 6x6 -> 6x6 (v[36]).
    static void inp(const float d[36], float v[36]) {
        float m[36];
        for (int j = 0; j < 6; ++j) {
            float col[6] = { d[0*6+j], d[1*6+j], d[2*6+j], d[3*6+j], d[4*6+j], d[5*6+j] };
            float out[6]; Brow(col, out);
            for (int i = 0; i < 6; ++i) m[i*6 + j] = out[i];
        }
        for (int i = 0; i < 6; ++i) {
            float out[6]; Brow(m + i*6, out);
            for (int k = 0; k < 6; ++k) v[i*6 + k] = out[k];
        }
    }
    // Y = A^T m A, 6x6 -> 4x4 (y[16]).
    static void outp(const float m[36], float y[16]) {
        float p[24];
        for (int j = 0; j < 6; ++j) {
            float col[6] = { m[0*6+j], m[1*6+j], m[2*6+j], m[3*6+j], m[4*6+j], m[5*6+j] };
            float out[4]; Arow(col, out);
            for (int i = 0; i < 4; ++i) p[i*6 + j] = out[i];
        }
        for (int i = 0; i < 4; ++i) {
            float out[4]; Arow(p + i*6, out);
            for (int k = 0; k < 4; ++k) y[i*4 + k] = out[k];
        }
    }
};

// ------------------------------------------------------------------------
// Persistent per-op state: caches the filter transform U (computed once from
// w->data; reused across forwards). Scratch (V,M) is per-thread.
// ------------------------------------------------------------------------
struct WinogradState {
    Mode mode = Mode::F2B;
    int W = 0, H = 0, IC = 0, OC = 0, N = 0, pad = 0;
    int Wout = 0, Hout = 0, tilesX = 0, tilesY = 0;
    const void* wdata = nullptr;
    // U layout: U[pos*IC*OC + ic*OC + oc], pos in 0..P-1. OC innermost so the
    // winograd-domain multiply vectorizes over OC.
    std::vector<float> U;
    std::once_flag once;
};

template<class Pol>
static void build_U(WinogradState* st, const float* w) {
    const int IC = st->IC, OC = st->OC;
    st->U.assign((size_t)Pol::P * IC * OC, 0.0f);
    float u[Pol::P];
    for (int oc = 0; oc < OC; ++oc) {
        for (int ic = 0; ic < IC; ++ic) {
            const float* g = w + ((size_t)oc * IC + ic) * 9;
            Pol::filt(g, u);
            for (int pos = 0; pos < Pol::P; ++pos)
                st->U[(size_t)pos * IC * OC + (size_t)ic * OC + oc] = u[pos];
        }
    }
}

// ------------------------------------------------------------------------
// Per-tile GEMV (mode "f2"): M[oc] = sum_ic U[ic,oc]*V[ic].
// ------------------------------------------------------------------------
#if defined(VD_WINO_HAVE_AVX512)
// AVX-512 per-tile GEMV microkernel (mode "f2"). VD_WINO_AVX512_TARGET confines
// the zmm ISA to this function (see the dispatch note at the top); only reached
// when wino_use_avx512() is true. Same ascending-ic FMA chain as the AVX2 path,
// so it is bit-identical per OC lane.
static inline VD_WINO_AVX512_TARGET void wino_gemv_avx512(const float* Upos, const float* Vpos, float* out, int IC, int OC) {
    int oc = 0;
    for (; oc + 16 <= OC; oc += 16) {
        __m512 acc = _mm512_setzero_ps();
        const float* up = Upos + oc;
        for (int ic = 0; ic < IC; ++ic)
            acc = _mm512_fmadd_ps(_mm512_loadu_ps(up + (size_t)ic * OC),
                                  _mm512_set1_ps(Vpos[ic]), acc);
        _mm512_storeu_ps(out + oc, acc);
    }
    if (oc < OC) {
        const int rem = OC - oc;
        const __mmask16 mask = (__mmask16)((1u << rem) - 1u);
        __m512 acc = _mm512_setzero_ps();
        const float* up = Upos + oc;
        for (int ic = 0; ic < IC; ++ic)
            acc = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(mask, up + (size_t)ic * OC),
                                  _mm512_set1_ps(Vpos[ic]), acc);
        _mm512_mask_storeu_ps(out + oc, mask, acc);
    }
}
#endif

static inline void wino_gemv(const float* Upos, const float* Vpos, float* out, int IC, int OC) {
#if defined(VD_WINO_HAVE_AVX512)
    // Runtime-dispatched: AVX-512 zmm microkernel only when the CPU advertises
    // AVX-512, else the AVX2 microkernel below -- one binary, no SIGILL on
    // AVX2-only hosts.
    if (wino_use_avx512()) { wino_gemv_avx512(Upos, Vpos, out, IC, OC); return; }
#endif
#if defined(__AVX2__)
    int oc = 0;
    for (; oc + 8 <= OC; oc += 8) {
        __m256 acc = _mm256_setzero_ps();
        const float* up = Upos + oc;
        for (int ic = 0; ic < IC; ++ic)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(up + (size_t)ic * OC),
                                  _mm256_set1_ps(Vpos[ic]), acc);
        _mm256_storeu_ps(out + oc, acc);
    }
    for (; oc < OC; ++oc) {           // scalar tail for OC % 8
        float acc = 0.0f;
        const float* up = Upos + oc;
        for (int ic = 0; ic < IC; ++ic) acc += up[(size_t)ic * OC] * Vpos[ic];
        out[oc] = acc;
    }
#else
    for (int oc = 0; oc < OC; ++oc) out[oc] = 0.0f;
    for (int ic = 0; ic < IC; ++ic) {
        const float vv = Vpos[ic];
        const float* up = Upos + (size_t)ic * OC;
        for (int oc = 0; oc < OC; ++oc) out[oc] += up[oc] * vv;
    }
#endif
}

// ------------------------------------------------------------------------
// Blocked GEMM microkernel for one winograd position:
//   M[t][oc] = sum_ic U[ic][oc] * V[ic][t],  t in [0,TBcur), oc in [0,OC).
// U: [IC][OC] row-major (OC innermost). V: [IC][TB] row-major. M: [TB][OC].
// Each loaded U-row is reused across all TBcur tiles -> far better arithmetic
// intensity than the per-tile GEMV.
// ------------------------------------------------------------------------
#if defined(VD_WINO_HAVE_AVX512)
// AVX-512 blocked winograd-domain GEMM microkernel (modes "f2b"/"f4", the hot
// default path). VD_WINO_AVX512_TARGET confines the zmm ISA to this function (see
// the dispatch note at the top); only reached when wino_use_avx512() is true.
// Same ascending-ic FMA chain as the AVX2 path, so it is bit-identical per OC
// lane.
static inline VD_WINO_AVX512_TARGET void wino_gemm_block_avx512(const float* U, const float* V, float* M,
                                                                int IC, int OC, int TBcur) {
    int oc = 0;
    for (; oc + 16 <= OC; oc += 16) {
        __m512 acc[TB];
        for (int t = 0; t < TB; ++t) acc[t] = _mm512_setzero_ps();
        const float* up = U + oc;
        for (int ic = 0; ic < IC; ++ic) {
            const __m512 u = _mm512_loadu_ps(up + (size_t)ic * OC);
            const float* vp = V + (size_t)ic * TB;
            for (int t = 0; t < TBcur; ++t)
                acc[t] = _mm512_fmadd_ps(u, _mm512_set1_ps(vp[t]), acc[t]);
        }
        for (int t = 0; t < TBcur; ++t)
            _mm512_storeu_ps(M + (size_t)t * OC + oc, acc[t]);
    }
    if (oc < OC) {
        const int rem = OC - oc;
        const __mmask16 mask = (__mmask16)((1u << rem) - 1u);
        __m512 acc[TB];
        for (int t = 0; t < TB; ++t) acc[t] = _mm512_setzero_ps();
        const float* up = U + oc;
        for (int ic = 0; ic < IC; ++ic) {
            const __m512 u = _mm512_maskz_loadu_ps(mask, up + (size_t)ic * OC);
            const float* vp = V + (size_t)ic * TB;
            for (int t = 0; t < TBcur; ++t)
                acc[t] = _mm512_fmadd_ps(u, _mm512_set1_ps(vp[t]), acc[t]);
        }
        for (int t = 0; t < TBcur; ++t)
            _mm512_mask_storeu_ps(M + (size_t)t * OC + oc, mask, acc[t]);
    }
}
#endif

static inline void wino_gemm_block(const float* U, const float* V, float* M,
                                   int IC, int OC, int TBcur) {
#if defined(VD_WINO_HAVE_AVX512)
    // Runtime-dispatched: AVX-512 zmm microkernel only when the CPU advertises
    // AVX-512, else the AVX2 microkernel below -- one binary, no SIGILL on
    // AVX2-only hosts.
    if (wino_use_avx512()) { wino_gemm_block_avx512(U, V, M, IC, OC, TBcur); return; }
#endif
#if defined(__AVX2__)
    int oc = 0;
    for (; oc + 8 <= OC; oc += 8) {
        __m256 acc[TB];
        for (int t = 0; t < TB; ++t) acc[t] = _mm256_setzero_ps();
        const float* up = U + oc;
        for (int ic = 0; ic < IC; ++ic) {
            const __m256 u = _mm256_loadu_ps(up + (size_t)ic * OC);
            const float* vp = V + (size_t)ic * TB;
            for (int t = 0; t < TBcur; ++t)
                acc[t] = _mm256_fmadd_ps(u, _mm256_set1_ps(vp[t]), acc[t]);
        }
        for (int t = 0; t < TBcur; ++t)
            _mm256_storeu_ps(M + (size_t)t * OC + oc, acc[t]);
    }
    if (oc < OC) {                    // scalar tail for OC % 8 (same accum order)
        for (int t = 0; t < TBcur; ++t)
            for (int o = oc; o < OC; ++o) M[(size_t)t * OC + o] = 0.0f;
        for (int ic = 0; ic < IC; ++ic) {
            const float* up = U + (size_t)ic * OC;
            const float* vp = V + (size_t)ic * TB;
            for (int t = 0; t < TBcur; ++t) {
                const float vv = vp[t];
                float* mt = M + (size_t)t * OC;
                for (int o = oc; o < OC; ++o) mt[o] += up[o] * vv;
            }
        }
    }
#else
    for (int t = 0; t < TBcur; ++t)
        for (int oc = 0; oc < OC; ++oc) M[(size_t)t * OC + oc] = 0.0f;
    for (int ic = 0; ic < IC; ++ic) {
        const float* up = U + (size_t)ic * OC;
        const float* vp = V + (size_t)ic * TB;
        for (int t = 0; t < TBcur; ++t) {
            const float vv = vp[t];
            float* mt = M + (size_t)t * OC;
            for (int oc = 0; oc < OC; ++oc) mt[oc] += up[oc] * vv;
        }
    }
#endif
}

// ------------------------------------------------------------------------
// Original F(2x2) per-tile path (mode "f2").
// ------------------------------------------------------------------------
static void compute_f2_gemv(WinogradState* st, ggml_tensor* dst, int ith, int nth) {
    const ggml_tensor* xt = dst->src[0];
    const float* x = (const float*)xt->data;
    float* y = (float*)dst->data;

    const int W = st->W, H = st->H, IC = st->IC, OC = st->OC, pad = st->pad;
    const int Wout = st->Wout, Hout = st->Hout;
    const int tilesX = st->tilesX, tilesY = st->tilesY;
    const float* U = st->U.data();

    const int ntiles = tilesX * tilesY;
    const int64_t total = (int64_t)st->N * ntiles;
    const int64_t beg = total * ith / nth;
    const int64_t end = total * (ith + 1) / nth;

    std::vector<float> Vbuf((size_t)16 * IC);
    std::vector<float> Mbuf((size_t)16 * OC);
    float dpatch[16], vpatch[16], mpatch[16], ypatch[4];

    for (int64_t idx = beg; idx < end; ++idx) {
        const int n  = (int)(idx / ntiles);
        const int t  = (int)(idx % ntiles);
        const int ty = t / tilesX, tx = t % tilesX;
        const int oy0 = ty * 2, ox0 = tx * 2;
        const int iy0 = oy0 - pad, ix0 = ox0 - pad;
        const float* xn = x + (size_t)n * IC * H * W;

        for (int ic = 0; ic < IC; ++ic) {
            const float* xc = xn + (size_t)ic * H * W;
            for (int i = 0; i < 4; ++i) {
                const int yy = iy0 + i;
                const bool yok = (yy >= 0 && yy < H);
                const float* row = yok ? (xc + (size_t)yy * W) : nullptr;
                for (int j = 0; j < 4; ++j) {
                    const int xx = ix0 + j;
                    dpatch[i*4 + j] = (yok && xx >= 0 && xx < W) ? row[xx] : 0.0f;
                }
            }
            F2Policy::inp(dpatch, vpatch);
            for (int pos = 0; pos < 16; ++pos) Vbuf[(size_t)pos * IC + ic] = vpatch[pos];
        }
        for (int pos = 0; pos < 16; ++pos)
            wino_gemv(U + (size_t)pos * IC * OC, Vbuf.data() + (size_t)pos * IC,
                      Mbuf.data() + (size_t)pos * OC, IC, OC);

        float* yn = y + (size_t)n * OC * Hout * Wout;
        for (int oc = 0; oc < OC; ++oc) {
            for (int pos = 0; pos < 16; ++pos) mpatch[pos] = Mbuf[(size_t)pos * OC + oc];
            F2Policy::outp(mpatch, ypatch);
            float* yc = yn + (size_t)oc * Hout * Wout;
            for (int i = 0; i < 2; ++i) {
                const int oy = oy0 + i; if (oy >= Hout) continue;
                for (int j = 0; j < 2; ++j) {
                    const int ox = ox0 + j; if (ox >= Wout) continue;
                    yc[(size_t)oy * Wout + ox] = ypatch[i*2 + j];
                }
            }
        }
    }
}

// ------------------------------------------------------------------------
// Blocked path (modes "f2b" / "f4"): batch TB tiles per winograd-domain GEMM.
// ------------------------------------------------------------------------
template<class Pol>
static void compute_blocked(WinogradState* st, ggml_tensor* dst, int ith, int nth) {
    constexpr int IT = Pol::IT, OT = Pol::OT, P = Pol::P;
    const ggml_tensor* xt = dst->src[0];
    const float* x = (const float*)xt->data;
    float* y = (float*)dst->data;

    const int W = st->W, H = st->H, IC = st->IC, OC = st->OC, pad = st->pad;
    const int Wout = st->Wout, Hout = st->Hout;
    const int tilesX = st->tilesX, tilesY = st->tilesY;
    const float* U = st->U.data();

    const int ntiles = tilesX * tilesY;
    const int64_t total = (int64_t)st->N * ntiles;
    const int64_t nblocks = (total + TB - 1) / TB;
    const int64_t bbeg = nblocks * ith / nth;
    const int64_t bend = nblocks * (ith + 1) / nth;

    std::vector<float> Vblk((size_t)P * IC * TB);   // V[pos][ic][t]
    std::vector<float> Mblk((size_t)P * TB * OC);   // M[pos][t][oc]
    float dpatch[IT*IT], vpatch[P], mpatch[P], ypatch[OT*OT];

    for (int64_t b = bbeg; b < bend; ++b) {
        const int64_t t0 = b * TB;
        const int TBcur = (int)std::min<int64_t>(TB, total - t0);

        // 1. Input transform for each tile in the block -> Vblk[pos][ic][t].
        for (int tb = 0; tb < TBcur; ++tb) {
            const int64_t idx = t0 + tb;
            const int n  = (int)(idx / ntiles);
            const int t  = (int)(idx % ntiles);
            const int ty = t / tilesX, tx = t % tilesX;
            const int iy0 = ty * OT - pad, ix0 = tx * OT - pad;
            const float* xn = x + (size_t)n * IC * H * W;
            for (int ic = 0; ic < IC; ++ic) {
                const float* xc = xn + (size_t)ic * H * W;
                for (int i = 0; i < IT; ++i) {
                    const int yy = iy0 + i;
                    const bool yok = (yy >= 0 && yy < H);
                    const float* row = yok ? (xc + (size_t)yy * W) : nullptr;
                    for (int j = 0; j < IT; ++j) {
                        const int xx = ix0 + j;
                        dpatch[i*IT + j] = (yok && xx >= 0 && xx < W) ? row[xx] : 0.0f;
                    }
                }
                Pol::inp(dpatch, vpatch);
                float* vbase = Vblk.data() + (size_t)ic * TB + tb;
                for (int pos = 0; pos < P; ++pos)
                    vbase[(size_t)pos * IC * TB] = vpatch[pos];
            }
        }

        // 2. Winograd-domain blocked GEMM per position.
        for (int pos = 0; pos < P; ++pos)
            wino_gemm_block(U + (size_t)pos * IC * OC,
                            Vblk.data() + (size_t)pos * IC * TB,
                            Mblk.data() + (size_t)pos * TB * OC, IC, OC, TBcur);

        // 3. Output transform per tile per oc -> scatter OTxOT into dst.
        for (int tb = 0; tb < TBcur; ++tb) {
            const int64_t idx = t0 + tb;
            const int n  = (int)(idx / ntiles);
            const int t  = (int)(idx % ntiles);
            const int ty = t / tilesX, tx = t % tilesX;
            const int oy0 = ty * OT, ox0 = tx * OT;
            float* yn = y + (size_t)n * OC * Hout * Wout;
            for (int oc = 0; oc < OC; ++oc) {
                const float* mbase = Mblk.data() + (size_t)tb * OC + oc;
                for (int pos = 0; pos < P; ++pos)
                    mpatch[pos] = mbase[(size_t)pos * TB * OC];
                Pol::outp(mpatch, ypatch);
                float* yc = yn + (size_t)oc * Hout * Wout;
                for (int i = 0; i < OT; ++i) {
                    const int oy = oy0 + i; if (oy >= Hout) continue;
                    for (int j = 0; j < OT; ++j) {
                        const int ox = ox0 + j; if (ox >= Wout) continue;
                        yc[(size_t)oy * Wout + ox] = ypatch[i*OT + j];
                    }
                }
            }
        }
    }
}

static void winograd_compute(ggml_tensor* dst, int ith, int nth, void* userdata) {
    WinogradState* st = (WinogradState*)userdata;
    const ggml_tensor* wt = dst->src[1];
    const float* w = (const float*)wt->data;

    switch (st->mode) {
        case Mode::F4:
            std::call_once(st->once, [&]{ build_U<F4Policy>(st, w); });
            compute_blocked<F4Policy>(st, dst, ith, nth);
            break;
        case Mode::F2B:
            std::call_once(st->once, [&]{ build_U<F2Policy>(st, w); });
            compute_blocked<F2Policy>(st, dst, ith, nth);
            break;
        case Mode::F2:
        default:
            std::call_once(st->once, [&]{ build_U<F2Policy>(st, w); });
            compute_f2_gemv(st, dst, ith, nth);
            break;
    }
}

// ------------------------------------------------------------------------
// Keyed cache of op states (U transformed once per (filter,shape,mode)).
// ------------------------------------------------------------------------
struct StateKey {
    const void* wdata; int W, H, IC, OC, N, pad; int mode;
    bool operator==(const StateKey& o) const {
        return wdata == o.wdata && W == o.W && H == o.H && IC == o.IC &&
               OC == o.OC && N == o.N && pad == o.pad && mode == o.mode;
    }
};
struct StateKeyHash {
    size_t operator()(const StateKey& k) const {
        size_t h = (size_t)k.wdata;
        auto mix = [&h](int v) { h = h * 1000003u + (size_t)(uint32_t)v; };
        mix(k.W); mix(k.H); mix(k.IC); mix(k.OC); mix(k.N); mix(k.pad); mix(k.mode);
        return h;
    }
};

static std::mutex g_states_mtx;
static std::unordered_map<StateKey, WinogradState*, StateKeyHash> g_states;

static int out_tile(Mode m) { return m == Mode::F4 ? 4 : 2; }

static WinogradState* get_state(ggml_tensor* w, ggml_tensor* x, int pad, Mode mode) {
    const int W = (int)x->ne[0], H = (int)x->ne[1], IC = (int)x->ne[2], N = (int)x->ne[3];
    const int OC = (int)w->ne[3];
    StateKey key{ w->data, W, H, IC, OC, N, pad, (int)mode };
    std::lock_guard<std::mutex> lk(g_states_mtx);
    auto it = g_states.find(key);
    if (it != g_states.end()) return it->second;
    WinogradState* st = new WinogradState();
    st->mode = mode;
    st->W = W; st->H = H; st->IC = IC; st->OC = OC; st->N = N; st->pad = pad;
    st->Wout = W + 2 * pad - 2;
    st->Hout = H + 2 * pad - 2;
    const int OT = out_tile(mode);
    st->tilesX = (st->Wout + OT - 1) / OT;
    st->tilesY = (st->Hout + OT - 1) / OT;
    st->wdata = w->data;
    g_states[key] = st;
    return st;
}

} // namespace

ggml_tensor* winograd_conv3x3(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x, int pad) {
    const int OC = (int)w->ne[3];
    const int N  = (int)x->ne[3];
    const int Wout = (int)x->ne[0] + 2 * pad - 2;
    const int Hout = (int)x->ne[1] + 2 * pad - 2;
    const Mode mode = parse_mode();
    WinogradState* st = get_state(w, x, pad, mode);
    ggml_tensor* args[2] = { x, w };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, Wout, Hout, OC, N,
                          args, 2, winograd_compute, GGML_N_TASKS_MAX, st);
}

} // namespace vd
