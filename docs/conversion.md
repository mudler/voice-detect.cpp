# GGUF conversion schema

`scripts/convert_voicedetect_to_gguf.py` turns a speaker-recognition checkpoint
(SpeechBrain ECAPA-TDNN, or an ONNX WeSpeaker / 3D-Speaker encoder) into a single
**GGUF** file consumed by the C++ `ModelLoader`.

The design rule is **fully metadata-driven**: every configuration value lives in
GGUF KV, and **tensor names are kept verbatim from the source `state_dict`** (no
renaming). This makes the C++ port a 1:1 mapping of the reference model - new
checkpoints/variants require no C++ changes, only new KV/tensors.

```
.venv/bin/python scripts/convert_voicedetect_to_gguf.py \
    --model speechbrain/spkrec-ecapa-voxceleb \
    --output model.gguf
```

The acceptance check is `tests/python/check_convert.py` (registered as the
`check_convert` ctest, label `model`; skips with exit 77 if `gguf` or the
checkpoint are unavailable).

## GGUF header

- `general.architecture` = `"voicedetect"` (the GGUFWriter arch; what the reader keys on).
- `general.name` = the `--model` argument (HF id or path).

## KV metadata (`voicedetect.*`)

| Key | Type | Meaning |
| --- | --- | --- |
| `voicedetect.arch` | string | `ecapa_tdnn` \| `wespeaker_resnet34` \| `eres2net` \| `campplus` |
| `voicedetect.embedding_dim` | u32 | speaker embedding size (192 for ECAPA) |
| `voicedetect.l2_normalize` | bool | L2-normalize the output embedding (cosine space) |
| `voicedetect.fbank.sample_rate` | u32 | 16000 |
| `voicedetect.fbank.n_mels` | u32 | filterbank feature dim (80) |
| `voicedetect.fbank.n_fft` | u32 | FFT size (512) |
| `voicedetect.fbank.win_length` | u32 | window samples (400 = 25 ms) |
| `voicedetect.fbank.hop_length` | u32 | hop samples (160 = 10 ms) |
| `voicedetect.fbank.preemph` | f32 | pre-emphasis coefficient (0.97) |
| `voicedetect.fbank.low_freq` / `.high_freq` | f32 | mel band edges (20 / 0=Nyquist) |
| `voicedetect.fbank.use_energy` | bool | append a 0th energy coefficient (false) |
| `voicedetect.fbank.cmn` | bool | per-utterance cepstral mean normalization (true) |
| `voicedetect.fbank.window` | string | window type (`povey`) |
| `voicedetect.analyze.present` | bool | the age/gender/emotion heads are present |
| `voicedetect.analyze.kind` | string | analyze head kind (`emotion` or `age_gender`) |
| `voicedetect.analyze.emotion_labels` | [string] | emotion class labels (emotion analyze head) |
| `voicedetect.analyze.gender_labels` | [string] | gender class labels (age/gender analyze head) |
| `voicedetect.analyze.gender_labels` | [string] | gender class labels (analyze head) |

### WeSpeaker ResNet34 block manifest (`arch == wespeaker_resnet34`)

The WeSpeaker export is an ONNX graph with **folded BatchNorm** (each Conv carries
its weight + bias) and **opaque integer initializer names**, so the C++ engine
cannot infer the block topology from tensor names. The converter walks the graph
and records it as parallel arrays (one entry per ResNet BasicBlock, forward order);
channel widths / stage depths are **not** stored: the engine reads them from each
weight's `ne`.

| Key | Type | Meaning |
| --- | --- | --- |
| `voicedetect.resnet.stem_weight` / `.stem_bias` | string | 3x3 stem conv (folded BN) |
| `voicedetect.resnet.conv1_weight` / `.conv1_bias` | [string] | per-block conv1 names |
| `voicedetect.resnet.conv2_weight` / `.conv2_bias` | [string] | per-block conv2 names |
| `voicedetect.resnet.down_weight` / `.down_bias` | [string] | per-block 1x1 downsample (`""` = identity) |
| `voicedetect.resnet.stride` | [i32] | per-block spatial stride (1 or 2) |
| `voicedetect.resnet.seg_weight` / `.seg_bias` | string | embedding FC (Gemm, 5120 -> emb_dim) |
| `voicedetect.resnet.mean_vec` | string | post-FC mean centering (`""` if none) |
| `voicedetect.resnet.var_eps` | f32 | std = sqrt(unbiased_var + eps) in stats pooling |

The C++ FBank front end is shared (Kaldi povey + per-utterance CMN); the ONNX
input is rank-3 `[1, frames, n_mels]`, so the engine feeds the FBank features
directly with no transpose beyond the conv layout.

### 3D-Speaker ERes2Net manifest (`arch == eres2net`)

ERes2Net (3D-Speaker) is a 2D ResNet over the FBank with Res2Net multi-scale
blocks, attentional feature fusion (AFF), and a bottom-up cross-stage fusion. Its
ONNX export also has **folded BatchNorm** and **opaque conv names**. Crucially,
`torch.onnx` emits Conv nodes in forward-**execution** order, so the converter
records a single flat conv list in that order; the C++ `ERes2NetEncoder` replays
the deterministic ERes2Net forward and pulls each conv by a running index. The
block topology (`num_blocks=[3,4,6,3]`, `scale=2`) is the fixed ERes2Net shape;
`m_channels` and the embedding dim are read from tensor shapes (base/std/large
differ only in those). The base 200k model feeds an 80-dim FBank and emits a
512-d embedding. ERes2Net's `feature_normalize_type=global-mean` is exactly the
per-utterance CMN the shared front end already applies.

| Key | Type | Meaning |
| --- | --- | --- |
| `voicedetect.eres2net.conv_weight` / `.conv_bias` | [string] | flat conv names, execution order (`""` bias for the 3 BN-less downsample convs) |
| `voicedetect.eres2net.conv_stride` | [i32] | per-conv spatial stride (1 or 2) |
| `voicedetect.eres2net.num_blocks` | [i32] | blocks per layer (`[3,4,6,3]`) |
| `voicedetect.eres2net.m_channels` | u32 | stem conv out channels (32 base) |
| `voicedetect.eres2net.scale` | u32 | Res2Net multi-scale split (2) |
| `voicedetect.eres2net.seg_weight` / `.seg_bias` | string | embedding FC (`seg_1` Gemm) |
| `voicedetect.eres2net.relu_clamp` | f32 | in-block ReLU = Hardtanh(0, clamp); stem uses plain ReLU |
| `voicedetect.eres2net.var_eps` | f32 | TSTP std = sqrt(unbiased_var + eps) |

### 3D-Speaker CAM++ manifest (`arch == campplus`)

CAM++ (3D-Speaker) is a 2D-conv FCM front end (3x3 stem + two BasicResBlock stages
striding the frequency axis, reshaped to a `[320, T]` feature) feeding a 1D D-TDNN
backbone: a strided TDNN, three densely connected `CAMDenseTDNNBlock`s (num_layers
`[12, 24, 16]`, dilation `[1, 2, 2]`, growth 32) each followed by a channel-halving
transit, an out ReLU, statistics pooling (per-channel mean + unbiased std over
time), a 1024->192 dense projection, and a final **affine-free** BatchNorm. Each
dense layer applies a context-aware mask (CAM): `out = linear_local(c) *
sigmoid(linear2(relu(linear1(c.mean(-1) + seg100(c)))))`, where `seg100` is a
100-frame **count_include_pad** segment average (ONNX `AveragePool` ceil_mode=1,
count_include_pad=1: the partial last segment is still divided by 100, NOT the
valid count). The export folds every BatchNorm that immediately follows a
single-consumer Conv into that Conv (renamed `onnx::Conv_*`, gaining a bias) and
keeps semantic names for the rest; both Conv (225) and BatchNormalization (56)
nodes are emitted in forward-**execution** order, so the converter records two flat
parallel lists and the C++ `CamPPlusEncoder` replays the fixed forward, pulling
each conv/BN by a running index. The block topology is hardcoded and asserted via
the 225/56 counts. CAM++'s `feature_normalize_type=global-mean` is exactly the
per-utterance CMN the shared front end already applies.

| Key | Type | Meaning |
| --- | --- | --- |
| `voicedetect.campplus.conv_weight` / `.conv_bias` | [string] | flat conv names, execution order (`""` bias for the BN-folded-free convs) |
| `voicedetect.campplus.bn_prefix` | [string] | the 55 affine BatchNorm prefixes (execution order); each has `.weight/.bias/.running_mean/.running_var` |
| `voicedetect.campplus.emb_bn_mean` / `.emb_bn_var` | string | the final affine-free embedding BatchNorm running stats |
| `voicedetect.campplus.bn_eps` | f32 | BatchNorm epsilon (PyTorch default 1e-5) |

### wav2vec2 emotion analyze head (`arch == wav2vec2_emotion`, `--analyze`)

The analyze head is a **separate model** from the speaker encoders and gets its
own analyze GGUF (default `superb/wav2vec2-base-superb-er`, Apache-2.0; convert
with `--analyze`). It is the wav2vec2-base architecture: a raw-waveform conv
feature extractor (7 Conv1d layers, kernel `[10,3,3,3,3,2,2]`, stride
`[5,2,2,2,2,2,2]`, `feat_extract_norm=group` so only conv layer 0 has a norm),
feature projection (LayerNorm + 512->768 Linear), a positional conv embedding
(grouped Conv1d, weight-norm parametrized on the kernel axis `dim=2` via
`...parametrizations.weight.original0/original1`), `encoder.layer_norm`, then 12
post-norm transformer layers (self-attention + GELU feed-forward). The
classification head uses `use_weighted_layer_sum`: a softmax over `layer_weights`
(13 values) weights all hidden states, then `projector` (768->256), mean-pool over
time, and `classifier` (256->N emotions). All weights are lifted from the HF
`state_dict` (no mel FBank, no L2-normalized embedding: the analyze path writes
neither the FBank KV nor `voicedetect.embedding_dim`).

Names are stored **near-verbatim**, abbreviated by one deterministic transform so
every name fits ggml's 64-char `GGML_MAX_NAME` (the reader rejects names `>= 64`,
which a fully verbatim wav2vec2 state_dict overflows, e.g. the 70-char
`...pos_conv_embed.conv.parametrizations.weight.original0`): the `wav2vec2.` module
prefix becomes `w2v2.`, and the weight-norm parametrization buffers become the
canonical `weight_g` / `weight_v` (`original0` -> `weight_g`, `original1` ->
`weight_v`). Everything after the prefix is verbatim, so the C++ analyze graph
applies the SAME transform to map a module path to a tensor (e.g.
`w2v2.feature_extractor.conv_layers.0.conv.weight`).

| Key | Type | Meaning |
| --- | --- | --- |
| `voicedetect.analyze.present` | bool | analyze head present (always true here) |
| `voicedetect.analyze.emotion_labels` | [string] | emotion class labels in id order |
| `voicedetect.analyze.num_emotions` | u32 | number of emotion classes |
| `voicedetect.analyze.pooling` | string | classifier pooling over time (`mean`) |
| `voicedetect.w2v2.hidden_size` / `.n_layers` / `.n_heads` / `.ff_dim` | u32 | transformer dims |
| `voicedetect.w2v2.num_conv_layers` | u32 | conv feature-extractor layer count |
| `voicedetect.w2v2.conv_dims` / `.conv_kernels` / `.conv_strides` | [i32] | per-conv channels / kernel / stride |
| `voicedetect.w2v2.feat_extract_norm` | string | `group` (norm only on conv layer 0) or `layer` (every conv) |
| `voicedetect.w2v2.conv_bias` | bool | feature-encoder Conv1d bias (true for large-robust) |
| `voicedetect.w2v2.do_normalize` | bool | zero-mean/unit-var the input waveform (true for large-robust) |
| `voicedetect.w2v2.feat_extract_activation` / `.hidden_act` | string | activation (`gelu`) |
| `voicedetect.w2v2.do_stable_layer_norm` | bool | pre-norm (true) vs post-norm (false) transformer |
| `voicedetect.w2v2.layer_norm_eps` | f32 | LayerNorm epsilon |
| `voicedetect.w2v2.num_conv_pos_embeddings` / `.num_conv_pos_embedding_groups` | u32 | positional conv kernel / groups |
| `voicedetect.w2v2.pos_conv_weight_norm_dim` | u32 | weight-norm dim of the positional conv (`2`) |
| `voicedetect.w2v2.use_weighted_layer_sum` | bool | weight all hidden states by `softmax(layer_weights)` |
| `voicedetect.w2v2.classifier_proj_size` | u32 | projector output dim |

The `voicedetect.analyze.*` keys are emitted **only** for a model that bundles
the wav2vec2 emotion head; offline embedding models omit them and the C++ loader
sets `analyze_present=false`.

### wav2vec2 age/gender analyze head (`arch == wav2vec2_age_gender`, `--age-gender`)

A **second, separate** analyze model (default
`audeering/wav2vec2-large-robust-24-ft-age-gender`, **CC-BY-NC-SA-4.0,
non-commercial**; convert with `--age-gender`). It reuses the same wav2vec2 conv
encoder + transformer infrastructure as the emotion head, driven entirely by the
shared `voicedetect.w2v2.*` KV, but exercises the **large-robust** deltas:

- backbone dims 1024 / 24 layers / 16 heads / ff 4096 (vs base 768 / 12 / 12 / 3072);
- `feat_extract_norm = layer` so **every** conv layer is LayerNorm'd over channels,
  and `conv_bias = true` (the base/emotion convs have neither);
- `do_normalize = true`: the input waveform is zero-mean/unit-var normalized
  (`(x - mean) / sqrt(var + 1e-7)`) before the conv stack;
- `do_stable_layer_norm = true`: **pre-norm** transformer layers (LayerNorm before
  each sublayer) with **no** encoder LayerNorm before the stack and a single final
  encoder LayerNorm after it.

The custom audeering head is **not** an HF `SequenceClassification` head (which
`AutoModel` mangles): it is two `ModelHead`s lifted verbatim - `age.{dense,out_proj}`
(1024->1024->1) and `gender.{dense,out_proj}` (1024->1024->3) - each `out_proj(tanh(
dense(x)))` over the time-mean-pooled last hidden state. The age output is a single
regression scalar scaled `* 100` to years; the gender logits softmax to
female/male/child probabilities. KV: `voicedetect.analyze.kind = age_gender`,
`voicedetect.analyze.gender_labels` (id order) and `voicedetect.analyze.num_genders`.

## Tensors

Tensor names are **verbatim** source `state_dict` / ONNX initializer names. The
C++ `ModelLoader` maps `name -> ggml_tensor*` by exact string; never remap names
at conversion time.

## Baseline intermediates

`scripts/gen_baseline.py` dumps reference stage tensors (`fbank`, `encoder_out`,
`embedding`) into a separate `baseline.gguf` consumed by the parity tests. See
[parity.md](parity.md).

For the wav2vec2 emotion analyze head (`--analyze`) it dumps the analyze
intermediates instead: `feat_extract_out` `[conv_dim, T']` (conv feature-extractor
output), `enc_layer_0` / `enc_layer_mid` / `enc_layer_last` `[hidden, T']`
(transformer hidden states after layer 0, the middle layer, and the final layer),
and `emotion_logits` / `emotion_probs` `[num_emotions]`, plus KV
`baseline.emotion_labels` and `baseline.dominant_emotion`. These are the goldens
for the C++ analyze parity gates (T18-20).

For the wav2vec2 age/gender head (`--age-gender`) it dumps `feat_extract_out`
`[conv_dim, T']`, `enc_layer_last` `[hidden, T']` (the final encoder-LayerNorm
output / last hidden state), `age_raw` `[1]` (the raw regression scalar, age years
`= * 100`), and `gender_logits` / `gender_probs` `[num_genders]`, plus KV
`baseline.gender_labels`, `baseline.dominant_gender` and `baseline.age_years`.
Consumed by the `test_age_gender` parity gate.
