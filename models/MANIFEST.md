# Model manifest

Converted GGUFs are written here by `scripts/convert_voicedetect_to_gguf.py` and
are gitignored (only this file is tracked). This manifest tracks the expected
published set of speaker-recognition GGUFs.

| Model | arch | embedding | Source | dtypes | Status |
| --- | --- | --- | --- | --- | --- |
| `spkrec-ecapa-voxceleb` | `ecapa_tdnn` | 192 | [speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) | f32, f16, q8_0 | TODO |
| `wespeaker-resnet34` | `wespeaker_resnet34` | 256 | WeSpeaker | f32, f16, q8_0 | TODO |
| `eres2net` | `eres2net` | 192 | 3D-Speaker | f32, f16, q8_0 | TODO |
| `campplus` | `campplus` | 192 | 3D-Speaker | f32, f16, q8_0 | TODO |

Model weights are governed by each source project's original license - check the
model card before redistributing.

## Tensor-name manifests

`ecapa_tdnn_manifest.txt` is a committed text artifact listing every verbatim
SpeechBrain `embedding_model` state_dict tensor name and its torch shape (one
line per tensor, `NAME<TAB>d0,d1,d2`). It is the source of truth that the later
graph/converter tasks read tensor names from. Regenerate it in a reference venv:

```bash
python3 -m venv --system-site-packages .venv
. .venv/bin/activate && pip install -r scripts/requirements.txt
python3 scripts/trace_ecapa.py            # writes models/ecapa_tdnn_manifest.txt
```

Notes:

- The `mean_var_norm` normalizer for `spkrec-ecapa-voxceleb` is an
  `InputNormalization` with `norm_type=sentence` and an empty state_dict: it
  computes per-utterance statistics at runtime and contributes no persistent
  tensors, so the manifest is the 231 `embedding_model` tensors only.
- When the manifest is absent (a CI checkout without the artifact),
  `tests/python/check_manifest.py` exits 77 (ctest skip) instead of failing.
