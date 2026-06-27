# Test fixtures

Small audio clips used by the stage-parity and verification tests live here.
They exist **solely as deterministic parity fixtures** for the golden-baseline
dumper (`tests/python/gen_baseline.py`) and the end-to-end verification test
(`test_verify.cpp`). They are not training/eval data and carry no warranty about
acoustic quality.

## Clips

| File | Speaker | Source poem (LibriVox reader id) |
|------|---------|----------------------------------|
| `clip_a.wav`      | A | "Christmas Night" (reader `lcw`) |
| `clip_b_same.wav` | A | "God in a Heart" (reader `lcw`)  |
| `clip_c_diff.wav` | B | "All Golden Summer Day" (reader `bk`) |

Speaker labels: `clip_a.wav` and `clip_b_same.wav` are the **same speaker**
(LibriVox reader `lcw`); `clip_c_diff.wav` is a **different speaker** (LibriVox
reader `bk`). The labeled pairs consumed by the verification test are in
`pairs.json`.

## Provenance

- **Source corpus:** LibriVox - *Short Poetry Collection 272*, hosted on the
  Internet Archive.
- **Item URL:** https://archive.org/details/spc272_2602_librivox
- **Track download URLs:**
  - `clip_a.wav`      &larr; https://archive.org/download/spc272_2602_librivox/spc272_christmasnight_lcw.mp3
  - `clip_b_same.wav` &larr; https://archive.org/download/spc272_2602_librivox/spc272_godinaheart_lcw.mp3
  - `clip_c_diff.wav` &larr; https://archive.org/download/spc272_2602_librivox/spc272_allgoldensummerday_bk.mp3
- **License:** Public Domain Mark 1.0 (https://creativecommons.org/publicdomain/mark/1.0/).
  All LibriVox recordings are dedicated to the public domain. The reader-id
  suffix in each filename (`_lcw`, `_bk`) identifies the volunteer narrator and
  is what lets us pick a same-speaker pair vs a different-speaker clip.

## Transform

Each clip is the first 3 s of speech starting 5 s into the source track (to skip
the spoken title/announcement), resampled to 16 kHz mono 16-bit PCM:

```bash
ffmpeg -ss 5 -i spc272_christmasnight_lcw.mp3    -ac 1 -ar 16000 -t 3 -sample_fmt s16 clip_a.wav
ffmpeg -ss 5 -i spc272_godinaheart_lcw.mp3       -ac 1 -ar 16000 -t 3 -sample_fmt s16 clip_b_same.wav
ffmpeg -ss 5 -i spc272_allgoldensummerday_bk.mp3 -ac 1 -ar 16000 -t 3 -sample_fmt s16 clip_c_diff.wav
```

`file tests/fixtures/*.wav` should report:
`RIFF (little-endian) data, WAVE audio, Microsoft PCM, 16 bit, mono 16000 Hz`.
