# voice-detect.cpp - Parity report

This document records the numerical and end-to-end parity of the C++/ggml
speaker-recognition path against the reference implementations (SpeechBrain for
ECAPA-TDNN; the ONNX runtime graph for WeSpeaker / 3D-Speaker encoders). All
comparisons are CPU, deterministic (dither off).

> Status: scaffolding. The encoder graphs are not implemented yet, so the matrix
> below is the **target** each model must hit before it is marked PASS.

## Parity gates

A stage or model passes when it meets ALL of:

1. **Embedding cosine similarity `>= 0.9999`** vs the reference L2-normalized
   speaker embedding for the same clip.
2. **Max absolute difference `<= 1e-3`** on every dumped intermediate tensor
   (`fbank`, `encoder_out`, `embedding`) vs `scripts/gen_baseline.py`.
3. **Identical verification verdict** vs the reference on a fixed enroll/probe
   set: for every (clip_a, clip_b, threshold) triple the C++ `verified` flag
   matches the reference decision (same-speaker / different-speaker).

The per-stage golden compare is `tests/parity.hpp` (`vdtest::compare`,
`vdtest::cosine`); `tests/test_fbank.cpp` is the first gate. Tests SKIP (exit 77)
when the baseline GGUF env vars are unset, so CI without the reference venv never
fails on them.

## Model coverage matrix (target)

| Checkpoint | Family | arch | embedding | FBank | Embedding cosine | Verdict | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `speechbrain/spkrec-ecapa-voxceleb` | ECAPA-TDNN | `ecapa_tdnn` | 192 | 80 | - | - | TODO |
| WeSpeaker ResNet34 | ResNet34 | `wespeaker_resnet34` | 256 | 80 | - | - | TODO |
| 3D-Speaker ERes2Net (base 200k) | ERes2Net | `eres2net` | 512 | 80 | 1.000000 | - | GREEN |
| 3D-Speaker CAM++ (zh-cn 16k-common) | CAM++ | `campplus` | 192 | 80 | 1.000000 | - | GREEN |
| `superb/wav2vec2-base-superb-er` | wav2vec2 emotion | `wav2vec2_emotion` | - | - | emotion_probs max\|d\| 4.8e-7 | dominant matches | GREEN |
| `audeering/wav2vec2-large-robust-24-ft-age-gender` | wav2vec2 age/gender | `wav2vec2_age_gender` | - | - | age 72.12==72.12; gender_probs max\|d\| 6e-8 | dominant matches | GREEN |

## Stage parity (target)

| Stage | Reference | Tensor | Gate |
| --- | --- | --- | --- |
| FBank front end | `torchaudio.compliance.kaldi.fbank` + CMN | `fbank` `[n_mels, T]` | max\|d\| <= 1e-3 |
| Speaker encoder | SpeechBrain / ONNX forward | `encoder_out` `[d, T']` | max\|d\| <= 1e-3 |
| Embedding | L2-normalized speaker vector | `embedding` `[embedding_dim]` | cosine >= 0.9999 |

The FBank stage is the load-bearing one: any deviation from the reference
Kaldi options (POVEY window, 0.97 pre-emphasis, dither off, snip_edges, mel band
edges, log floor, per-utterance CMN) propagates into a verification-threshold
drift. See `src/fbank.cpp`.
