#pragma once
#include <vector>
#include "model_loader.hpp"   // vd::VoiceDetectConfig

namespace vd {

// Kaldi-compatible log-Mel filterbank (FBank) frontend.
//
// This is the shared front end for EVERY supported speaker encoder (ECAPA-TDNN,
// WeSpeaker ResNet34, 3D-Speaker ERes2Net, CAM++). It turns 16 kHz mono float
// PCM into an [n_mels, T] (default 80-dim) feature matrix that the encoder
// consumes.
//
// PARITY-CRITICAL: the features MUST match torchaudio.compliance.kaldi.fbank
// (which SpeechBrain and the WeSpeaker/3D-Speaker ONNX exports use) bit-for-bit
// within tolerance - same pre-emphasis (0.97), POVEY window, dithering OFF,
// snip_edges semantics, mel filterbank (low_freq/high_freq), log floor, and the
// per-utterance mean normalization (CMN) the reference applies before the
// encoder. Any deviation shows up as a verification-threshold drift, so this
// stage gets its own golden-parity test (tests/test_fbank.cpp) against a
// scripts/gen_baseline.py dump.
class FBank {
public:
    // Construct from the model config (n_mels, n_fft, win/hop, low/high freq,
    // window type, pre-emphasis, energy floor, CMN flag - all GGUF-driven).
    explicit FBank(const VoiceDetectConfig& cfg) : cfg_(cfg) {}

    // Compute feat-major log-Mel features [n_mels, T] (out[m*T + t]) from 16 kHz
    // mono PCM. Returns an empty vector on failure; sets `out_T` to the number
    // of frames T.
    std::vector<float> compute(const std::vector<float>& pcm16k, int& out_T) const;

    int n_mels() const { return (int)cfg_.n_mels; }

private:
    const VoiceDetectConfig& cfg_;
};

} // namespace vd
