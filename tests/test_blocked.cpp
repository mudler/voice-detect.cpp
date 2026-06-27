// Unit-parity tests for the blocked-layout (nChw16c) island ops in
// directconv.cpp. Each op is checked against a hand-written NCHW reference on
// random data (no model needed). A blocked op is a layout-only change, so the
// reorder round-trip is bit-exact and the blocked 3x3/1x1 conv must match the
// NCHW direct accumulation to fp32 tolerance (same kh,kw,ic order). Gate: any
// mismatch beyond 1e-4 max|d| fails (exit 1); all-pass exits 0.
#include "directconv.hpp"
#include "backend.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>

using namespace vd;

static std::mt19937 g_rng(1234);
static std::vector<float> randv(size_t n) {
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (auto& x : v) x = d(g_rng);
    return v;
}

static double maxabs(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return 1e30;
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i)
        m = std::max(m, std::fabs((double)a[i] - (double)b[i]));
    return m;
}

static int g_fail = 0;
static void check(const char* name, double d, double tol) {
    bool ok = d <= tol;
    std::fprintf(stderr, "[%s] max|d|=%.3e tol=%.1e -> %s\n", name, d, tol, ok ? "OK" : "FAIL");
    if (!ok) g_fail = 1;
}

// helper: graph input from a host vector with explicit ne[]
static ggml_tensor* in(ggml_context* c, const int64_t* ne, int nd, const std::vector<float>& v) {
    return graph_input_tensor(c, GGML_TYPE_F32, nd, ne, v.data(), v.size() * sizeof(float));
}

int main() {
    set_num_threads(4);

    // ---- 1. reorder round-trip: NCHW -> blocked -> NCHW is bit-exact ----
    {
        const int W = 13, H = 7, C = 64;
        std::vector<float> x = randv((size_t)W * H * C);
        const int64_t ne[4] = { W, H, C, 1 };
        std::vector<float> rt;
        global_backend().compute([&](ggml_context* c) -> ggml_tensor* {
            ggml_tensor* t = in(c, ne, 4, x);
            ggml_tensor* xb = blocked_reorder_in(c, t);
            return blocked_reorder_out(c, xb, C);
        }, rt);
        check("reorder_roundtrip", maxabs(rt, x), 0.0);  // exact permute
    }

    // ---- 2. blocked 3x3 conv (stride 1, pad 1) vs NCHW reference ----
    auto ref_conv3x3 = [](const std::vector<float>& x, int W, int H, int IC,
                          const std::vector<float>& w, int OC, int pad, int stride,
                          int& Wout, int& Hout) {
        Wout = (W + 2 * pad - 3) / stride + 1;
        Hout = (H + 2 * pad - 3) / stride + 1;
        std::vector<float> y((size_t)Wout * Hout * OC, 0.0f);
        const size_t HW = (size_t)H * W;
        for (int oc = 0; oc < OC; ++oc)
            for (int oh = 0; oh < Hout; ++oh)
                for (int ow = 0; ow < Wout; ++ow) {
                    float acc = 0;
                    for (int kh = 0; kh < 3; ++kh) {
                        int iy = oh * stride + kh - pad;
                        if (iy < 0 || iy >= H) continue;
                        for (int kw = 0; kw < 3; ++kw) {
                            int ix = ow * stride + kw - pad;
                            if (ix < 0 || ix >= W) continue;
                            for (int ic = 0; ic < IC; ++ic)
                                acc += x[(size_t)ic * HW + (size_t)iy * W + ix] *
                                       w[(((size_t)oc * IC + ic) * 3 + kh) * 3 + kw];
                        }
                    }
                    y[((size_t)oc * Hout + oh) * Wout + ow] = acc;
                }
        return y;
    };

    auto run_conv3x3 = [&](int W, int H, int IC, int OC, int pad, int stride, const char* tag) {
        std::vector<float> x = randv((size_t)W * H * IC);
        std::vector<float> w = randv((size_t)3 * 3 * IC * OC);
        int Wout, Hout;
        std::vector<float> ref = ref_conv3x3(x, W, H, IC, w, OC, pad, stride, Wout, Hout);
        const int64_t xne[4] = { W, H, IC, 1 };
        const int64_t wne[4] = { 3, 3, IC, OC };
        std::vector<float> got;
        global_backend().compute([&](ggml_context* c) -> ggml_tensor* {
            ggml_tensor* xt = in(c, xne, 4, x);
            ggml_tensor* wt = in(c, wne, 4, w);
            ggml_tensor* xb = blocked_reorder_in(c, xt);
            ggml_tensor* yb = blocked_conv3x3(c, wt, xb, pad, stride);
            return blocked_reorder_out(c, yb, OC);
        }, got);
        check(tag, maxabs(got, ref), 1e-4);
    };
    run_conv3x3(20, 11, 32, 64, 1, 1, "conv3x3_s1");
    run_conv3x3(20, 11, 64, 128, 1, 2, "conv3x3_s2");
    run_conv3x3(7, 5, 16, 16, 1, 1, "conv3x3_small");

    // fused conv3x3 + bias + ReLU (the conv1 epilogue) vs NCHW reference
    auto run_conv3x3_fused = [&](int W, int H, int IC, int OC, int stride, const char* tag) {
        std::vector<float> x = randv((size_t)W * H * IC);
        std::vector<float> w = randv((size_t)3 * 3 * IC * OC);
        std::vector<float> bias = randv(OC);
        int Wout, Hout;
        std::vector<float> ref = ref_conv3x3(x, W, H, IC, w, OC, 1, stride, Wout, Hout);
        for (int oc = 0; oc < OC; ++oc)
            for (int p = 0; p < Wout * Hout; ++p) {
                float v = ref[(size_t)oc * Wout * Hout + p] + bias[oc];
                ref[(size_t)oc * Wout * Hout + p] = v > 0 ? v : 0;  // bias + ReLU
            }
        const int64_t xne[4] = { W, H, IC, 1 };
        const int64_t wne[4] = { 3, 3, IC, OC };
        const int64_t bne[1] = { OC };
        std::vector<float> got;
        global_backend().compute([&](ggml_context* c) -> ggml_tensor* {
            ggml_tensor* xt = in(c, xne, 4, x);
            ggml_tensor* wt = in(c, wne, 4, w);
            ggml_tensor* bt = in(c, bne, 1, bias);
            ggml_tensor* xb = blocked_reorder_in(c, xt);
            ggml_tensor* yb = blocked_conv3x3(c, wt, xb, 1, stride, bt, /*relu=*/true);
            return blocked_reorder_out(c, yb, OC);
        }, got);
        check(tag, maxabs(got, ref), 1e-4);
    };
    run_conv3x3_fused(20, 11, 32, 64, 1, "conv3x3_bias_relu_s1");
    run_conv3x3_fused(20, 11, 64, 128, 2, "conv3x3_bias_relu_s2");

    // ---- 3. blocked 1x1 strided conv vs NCHW reference ----
    auto run_conv1x1 = [&](int W, int H, int IC, int OC, int stride, const char* tag) {
        std::vector<float> x = randv((size_t)W * H * IC);
        std::vector<float> w = randv((size_t)IC * OC);
        int Wout = (W - 1) / stride + 1, Hout = (H - 1) / stride + 1;
        std::vector<float> ref((size_t)Wout * Hout * OC, 0.0f);
        const size_t HW = (size_t)H * W;
        for (int oc = 0; oc < OC; ++oc)
            for (int oh = 0; oh < Hout; ++oh)
                for (int ow = 0; ow < Wout; ++ow) {
                    float acc = 0;
                    for (int ic = 0; ic < IC; ++ic)
                        acc += x[(size_t)ic * HW + (size_t)(oh * stride) * W + ow * stride] *
                               w[(size_t)oc * IC + ic];
                    ref[((size_t)oc * Hout + oh) * Wout + ow] = acc;
                }
        const int64_t xne[4] = { W, H, IC, 1 };
        const int64_t wne[4] = { 1, 1, IC, OC };
        std::vector<float> got;
        global_backend().compute([&](ggml_context* c) -> ggml_tensor* {
            ggml_tensor* xt = in(c, xne, 4, x);
            ggml_tensor* wt = in(c, wne, 4, w);
            ggml_tensor* xb = blocked_reorder_in(c, xt);
            ggml_tensor* yb = blocked_conv1x1(c, wt, xb, stride);
            return blocked_reorder_out(c, yb, OC);
        }, got);
        check(tag, maxabs(got, ref), 1e-4);
    };
    run_conv1x1(20, 11, 64, 128, 2, "conv1x1_s2");
    run_conv1x1(15, 9, 32, 64, 2, "conv1x1_s2b");

    // ---- 4. blocked bias / relu / add vs NCHW reference ----
    {
        const int W = 13, H = 7, C = 64;
        std::vector<float> x = randv((size_t)W * H * C);
        std::vector<float> b = randv(C);
        const size_t HW = (size_t)H * W;
        std::vector<float> ref_bias(x.size()), ref_relu(x.size());
        for (int c = 0; c < C; ++c)
            for (size_t p = 0; p < HW; ++p) {
                float v = x[(size_t)c * HW + p] + b[c];
                ref_bias[(size_t)c * HW + p] = v;
                ref_relu[(size_t)c * HW + p] = v > 0 ? v : 0;
            }
        const int64_t xne[4] = { W, H, C, 1 };
        const int64_t bne[1] = { C };
        std::vector<float> got_bias, got_relu;
        global_backend().compute([&](ggml_context* c) -> ggml_tensor* {
            ggml_tensor* xt = in(c, xne, 4, x);
            ggml_tensor* bt = in(c, bne, 1, b);
            ggml_tensor* xb = blocked_reorder_in(c, xt);
            ggml_tensor* yb = blocked_bias(c, xb, bt);
            return blocked_reorder_out(c, yb, C);
        }, got_bias);
        check("blocked_bias", maxabs(got_bias, ref_bias), 1e-5);
        global_backend().compute([&](ggml_context* c) -> ggml_tensor* {
            ggml_tensor* xt = in(c, xne, 4, x);
            ggml_tensor* bt = in(c, bne, 1, b);
            ggml_tensor* xb = blocked_reorder_in(c, xt);
            ggml_tensor* yb = blocked_relu(c, blocked_bias(c, xb, bt));
            return blocked_reorder_out(c, yb, C);
        }, got_relu);
        check("blocked_bias_relu", maxabs(got_relu, ref_relu), 1e-5);

        // residual add
        std::vector<float> x2 = randv((size_t)W * H * C);
        std::vector<float> ref_add(x.size());
        for (size_t i = 0; i < x.size(); ++i) ref_add[i] = x[i] + x2[i];
        const int64_t x2ne[4] = { W, H, C, 1 };
        std::vector<float> got_add;
        global_backend().compute([&](ggml_context* c) -> ggml_tensor* {
            ggml_tensor* a = blocked_reorder_in(c, in(c, xne, 4, x));
            ggml_tensor* bb = blocked_reorder_in(c, in(c, x2ne, 4, x2));
            return blocked_reorder_out(c, blocked_add(c, a, bb), C);
        }, got_add);
        check("blocked_add", maxabs(got_add, ref_add), 1e-5);
    }

    shutdown_backend();
    std::fprintf(stderr, "test_blocked: %s\n", g_fail ? "FAIL" : "ALL OK");
    return g_fail;
}
