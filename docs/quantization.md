# Quantization policy

The converter (`scripts/convert_voicedetect_to_gguf.py`) supports `--dtype f32`
(default), `f16`, `q8_0`, and `q4_0`. The C++ engine only tolerates a non-F32
weight when that weight is fed **directly** into `ggml_mul_mat` as `src0` (the
operand ggml dequantizes on the fly inside the compute graph).

Every other weight is read by hand-rolled C++ as a raw `float*` (FBank
filterbank/window, batch-norm running stats, biases), or is too small / wrong
shape to block-quantize. Those MUST stay F32 or the engine produces a wrong
embedding.

## Why only the 1x1 convs

The ECAPA conv weights are stored kernel-first (ggml `ne = [K, IC, OC]`), and the
F32 conv path (`conv1d_same` in `src/encoder.cpp`) feeds them to `ggml_mul_mat`
as **`src1`** after an `ggml_im2col`, which ggml requires to be F32. So a weight
is quantizable only when it is a **1x1 Conv1d / linear projection** (`kernel ==
1`): for `K == 1` the im2col is a no-op and a 1x1 conv is a plain per-frame
matmul, so the converter stores the weight squeezed to a 2-D `[IC, OC]` matrix
(the trivial kernel axis dropped, contraction dim `IC` on `ne[0]`) and the engine
reads a non-F32 leaf as a `ggml_mul_mat` `src0` (`y = (W . x^T)^T`). The
larger-kernel convs (block0 `k=5`, the dilated res2net `k=3`) go through the real
im2col path and stay F32 - their leading `ne[0]` is the kernel size (1/3/5),
never block-aligned, and they are tiny anyway.

`q8_0`/`q4_0` block (size 32) along the contraction dim `IC`, so an allowlisted
1x1 weight additionally needs `IC % 32 == 0` and both `IC, OC >= 32`.

## Allowlist (ECAPA-TDNN)

`quant_plan` quantizes the 1x1 conv/linear weights above, **minus** the
parity-sensitive exclusions in `_QUANT_EXCLUDE`. Measured on
`speechbrain/spkrec-ecapa-voxceleb` (clips a/b/c), quantizing the excluded
weights pushes q8_0 past `max|d| <= 5e-3` and/or drops q4_0 below `cosine >=
0.997`:

| Excluded (kept F32) | Why |
|---|---|
| `fc.*` | final 6144->192 projection, shapes the embedding directly |
| `asp.tdnn` / `asp.conv` | attention path - error amplified by the softmax |
| `*.se_block.*` | squeeze-excitation gate (sigmoid); dominant `max|d|` contributor |
| `*.tdnn1` / `*.tdnn2` | per-block channel mixers inside the residual path |

What remains quantized is the **multi-layer feature-aggregation conv** (`mfa`, the
single largest weight at `3072x3072`), which is near-lossless at q8_0 **and**
q4_0. The selective-quant sweep (max `|d|` of the 192-d embedding vs the F32
reference; `mfa` quantized = the shipped allowlist):

| Quantized weights | q8_0 cosine | q8_0 max\|d\| | q4_0 cosine | q4_0 max\|d\| |
|---|---|---|---|---|
| all 1x1 (no exclusions) | 0.99941 | 7.5e-3 | 0.96645 | 5.4e-2 |
| + exclude fc, asp | 0.99945 | 7.0e-3 | 0.97142 | 5.3e-2 |
| + exclude se | 0.99962 | 5.7e-3 | 0.98429 | 3.8e-2 |
| + exclude tdnn1 | 0.99996 | 2.1e-3 | 0.99440 | 2.1e-2 |
| **+ exclude tdnn2 (mfa only, shipped)** | **0.999992** | **7.4e-4** | **0.99863** | **1.04e-2** |

## Measured parity (shipped allowlist)

Size on disk and embedding parity vs the F32 baseline, `spkrec-ecapa-voxceleb`:

| dtype | size | vs F32 | embedding cosine (a/b/c) | embedding max\|d\| |
|---|---|---|---|---|
| f32  | 79.4 MB | 1.00x | 1.000000 | 3.7e-7 |
| q8_0 | 52.9 MB | 0.67x | 0.999992 / 0.999994 / 0.999993 | 7.4e-4 |
| q4_0 | 48.4 MB | 0.61x | 0.998630 / 0.998729 / 0.998839 | 1.16e-2 |

Both clear the `test_encoder` quant gates: q8_0 `cosine >= 0.999` AND `max|d| <=
5e-3`; q4_0 `cosine >= 0.997`. `max|d|` degrades monotonically f32 < q8_0 < q4_0.

### Verification-verdict note (q4_0)

q4_0 preserves the speaker-verification **verdict** (the same-/different-speaker
decision at the 0.25 default threshold matches the reference for both fixture
pairs), but its pairwise cosine **distance** can drift up to ~2.8e-3 from the F32
value - just past the strict `2e-3` distance-parity tolerance in `test_verify`.
So q4_0 is suitable for embedding/identification and verdicts; the exact
distance-parity guarantee is an f32/q8_0 property. q8_0 distances track the
reference to `<= 2e-4`.

## K-quants

For K-quants (`q4_k`, `q5_k`, `q6_k`), re-quantize an F32 GGUF with the CLI once
a `quantize` subcommand is added (`voicedetect-cli quantize <in> <out> <type>`).
