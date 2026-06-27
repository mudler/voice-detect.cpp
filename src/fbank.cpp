#include "fbank.hpp"
#include "common.hpp"
#include <cmath>
#include <vector>
#include <limits>

// Kaldi-compatible log-Mel filterbank front end.
//
// Matches torchaudio.compliance.kaldi.fbank with the options the SpeechBrain /
// WeSpeaker / 3D-Speaker reference path uses (povey window, pre-emphasis 0.97,
// dither 0, remove_dc_offset, snip_edges, use_power, use_log_fbank), then applies
// per-utterance cepstral mean normalization (CMN). Host-side C++ (no ggml graph):
// the gate in tests/test_fbank.cpp diffs this against the golden "fbank" tensor.
//
// Why scaling does not matter here: pre-emphasis, DC removal and windowing are
// linear; the power spectrum scales by a constant; log turns that into an additive
// per-bin offset that the per-utterance CMN subtracts away. So whether the PCM is
// in [-1,1] or [-32768,32767] the post-CMN features are identical - which is what
// the golden captures.

namespace vd {
namespace {
constexpr double kPI = 3.14159265358979323846;

inline double mel(double f) { return 1127.0 * std::log(1.0 + f / 700.0); }

inline bool is_pow2(int n) { return n > 0 && (n & (n - 1)) == 0; }

// Radix-2 Cooley-Tukey FFT plan for a fixed transform size N (power of two).
// Geometry-only: the bit-reversal permutation and the twiddle table depend on N
// alone, not on the audio, so the plan is built once and reused for every frame
// (and across calls, via the thread-local cache below). Double precision keeps
// the power spectrum bit-for-bit aligned with the naive DFT parity reference.
struct FFTPlan {
    int n = 0;
    std::vector<int>    rev;   // bit-reversal permutation
    std::vector<double> wr;    // twiddle real, per stage concatenated
    std::vector<double> wi;    // twiddle imag

    void init(int N) {
        n = N;
        rev.assign(N, 0);
        int logN = 0;
        while ((1 << logN) < N) ++logN;
        for (int i = 0; i < N; ++i) {
            int r = 0;
            for (int b = 0; b < logN; ++b)
                if (i & (1 << b)) r |= 1 << (logN - 1 - b);
            rev[i] = r;
        }
        // Twiddles for each stage: for half-size m=len/2, w_j = exp(-2*pi*i*j/len).
        wr.clear();
        wi.clear();
        for (int len = 2; len <= N; len <<= 1) {
            const int m = len >> 1;
            for (int j = 0; j < m; ++j) {
                const double ang = -2.0 * kPI * j / len;
                wr.push_back(std::cos(ang));
                wi.push_back(std::sin(ang));
            }
        }
    }

    // In-place iterative FFT on interleaved-by-array re/im buffers (length n).
    void run(std::vector<double>& re, std::vector<double>& im) const {
        for (int i = 0; i < n; ++i) {
            const int j = rev[i];
            if (j > i) { std::swap(re[i], re[j]); std::swap(im[i], im[j]); }
        }
        int off = 0;  // running offset into the twiddle table
        for (int len = 2; len <= n; len <<= 1) {
            const int m = len >> 1;
            for (int s = 0; s < n; s += len) {
                for (int j = 0; j < m; ++j) {
                    const double cwr = wr[off + j], cwi = wi[off + j];
                    const int a = s + j, b = s + j + m;
                    const double tr = cwr * re[b] - cwi * im[b];
                    const double ti = cwr * im[b] + cwi * re[b];
                    re[b] = re[a] - tr; im[b] = im[a] - ti;
                    re[a] += tr;        im[a] += ti;
                }
            }
            off += m;
        }
    }
};

// Real power spectrum |X|^2 for bins k=0..n_fft/2, matching
// torch.fft.rfft(x).abs().pow(2). Uses a cached radix-2 FFT for power-of-two
// n_fft (the only sizes the speaker encoders use); falls back to the naive DFT
// otherwise so any future non-pow2 config stays correct.
void powspec(const std::vector<float>& frame, int n_fft, std::vector<double>& out) {
    const int half = n_fft / 2;
    out.assign(half + 1, 0.0);

    if (!is_pow2(n_fft)) {
        for (int k = 0; k <= half; ++k) {
            double re = 0.0, im = 0.0;
            for (int n = 0; n < (int)frame.size(); ++n) {
                double a = -2.0 * kPI * k * n / n_fft;
                re += frame[n] * std::cos(a);
                im += frame[n] * std::sin(a);
            }
            out[k] = re * re + im * im;
        }
        return;
    }

    static thread_local FFTPlan plan;
    static thread_local std::vector<double> re, im;
    if (plan.n != n_fft) plan.init(n_fft);
    re.assign(n_fft, 0.0);
    im.assign(n_fft, 0.0);
    const int fn = (int)frame.size();
    for (int i = 0; i < fn && i < n_fft; ++i) re[i] = frame[i];
    plan.run(re, im);
    for (int k = 0; k <= half; ++k) out[k] = re[k] * re[k] + im[k] * im[k];
}
} // namespace

std::vector<float> FBank::compute(const std::vector<float>& pcm16k, int& out_T) const {
    out_T = 0;
    const int win  = (int)cfg_.win_length, hop = (int)cfg_.hop_length;
    const int nfft = (int)cfg_.n_fft, nmel = (int)cfg_.n_mels;
    const int sr   = (int)cfg_.sample_rate;
    if (win <= 0 || hop <= 0 || nfft < win || nmel <= 0) return {};
    if ((int)pcm16k.size() < win) return {};

    // snip_edges=true framing (drop the trailing partial frame).
    const int T = 1 + ((int)pcm16k.size() - win) / hop;

    // POVEY window: (0.5 - 0.5*cos(2*pi*i/(N-1)))^0.85.
    std::vector<double> window(win);
    for (int i = 0; i < win; ++i)
        window[i] = std::pow(0.5 - 0.5 * std::cos(2.0 * kPI * i / (win - 1)), 0.85);

    // Kaldi triangular mel filterbank over [low, high] (high==0 -> Nyquist).
    // Bin frequency f(k) = k*sr/nfft; the Nyquist bin (k=nfft/2) lands on the last
    // filter's right edge and gets zero weight, matching torchaudio's zero pad.
    const int    half = nfft / 2;
    const double lo   = cfg_.fbank_low_freq;
    const double hi   = cfg_.fbank_high_freq > 0.0 ? cfg_.fbank_high_freq : sr / 2.0;
    const double mlo  = mel(lo), mhi = mel(hi);
    const double mstep = (mhi - mlo) / (nmel + 1);
    std::vector<std::vector<double>> fb(nmel, std::vector<double>(half + 1, 0.0));
    for (int m = 0; m < nmel; ++m) {
        const double l = mlo + m * mstep;
        const double c = mlo + (m + 1) * mstep;
        const double r = mlo + (m + 2) * mstep;
        for (int k = 0; k <= half; ++k) {
            const double f  = (double)k * sr / nfft;
            const double mf = mel(f);
            double w = 0.0;
            if (mf > l && mf <= c)      w = (mf - l) / (c - l);
            else if (mf > c && mf < r)  w = (r - mf) / (r - c);
            fb[m][k] = w;
        }
    }

    std::vector<float>  feats((size_t)nmel * T, 0.0f); // feat-major [n_mels, T]
    std::vector<float>  frame(nfft), raw(win);
    std::vector<double> ps;
    const double eps = std::numeric_limits<float>::epsilon();

    for (int t = 0; t < T; ++t) {
        const int off = t * hop;
        for (int i = 0; i < win; ++i) raw[i] = pcm16k[off + i];

        // remove_dc_offset: subtract the per-frame mean (before pre-emphasis).
        double mean = 0.0;
        for (int i = 0; i < win; ++i) mean += raw[i];
        mean /= win;
        for (int i = 0; i < win; ++i) raw[i] -= (float)mean;

        // Pre-emphasis with replicate padding: y[i]=x[i]-a*x[i-1], y[0]=x[0]-a*x[0].
        // Backward in-place keeps x[i-1] intact.
        for (int i = win - 1; i > 0; --i) raw[i] -= cfg_.preemph * raw[i - 1];
        raw[0] -= cfg_.preemph * raw[0];

        // Window then zero-pad to n_fft.
        for (int i = 0; i < nfft; ++i)
            frame[i] = (i < win) ? (float)(raw[i] * window[i]) : 0.0f;

        powspec(frame, nfft, ps);

        for (int m = 0; m < nmel; ++m) {
            double e = 0.0;
            for (int k = 0; k <= half; ++k) e += fb[m][k] * ps[k];
            feats[(size_t)m * T + t] = (float)std::log(e > eps ? e : eps);
        }
    }

    // Per-utterance CMN: subtract each mel bin's mean over time.
    if (cfg_.fbank_cmn) {
        for (int m = 0; m < nmel; ++m) {
            double s = 0.0;
            for (int t = 0; t < T; ++t) s += feats[(size_t)m * T + t];
            const float mu = (float)(s / T);
            for (int t = 0; t < T; ++t) feats[(size_t)m * T + t] -= mu;
        }
    }

    out_T = T;
    return feats;
}

} // namespace vd
