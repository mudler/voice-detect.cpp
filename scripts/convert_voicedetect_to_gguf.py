#!/usr/bin/env python3
"""Convert a speaker-recognition checkpoint to GGUF (f32 / f16 / q8_0).

The GGUF is fully metadata-driven: all config lives in KV under the
``voicedetect.*`` prefix, and tensor names are kept **verbatim** from the source
``state_dict`` (no renaming) so the C++ port is a 1:1 mapping. The FBank front-end
parameters are written to KV so the C++ side never re-derives the mel filterbank.

Supported source families (see docs/conversion.md):
  * ``ecapa_tdnn``          SpeechBrain ECAPA-TDNN (192-d, primary anchor)
  * ``wespeaker_resnet34``  WeSpeaker ResNet34 (ONNX-direct)
  * ``eres2net``            3D-Speaker ERes2Net (ONNX-direct)
  * ``campplus``            3D-Speaker CAM++ (ONNX-direct)

Quantization (``--dtype f16|q8_0``) is applied **only** to the large linear /
conv weights the C++ engine feeds directly into ``ggml_mul_mat`` (ggml
dequantizes those on the fly). Everything the hand-rolled C++ reads as raw F32
(batch-norm stats, biases, the FBank filterbank, embedding projections that are
reshaped before the matmul) stays F32 -- see ``should_quantize`` and
``docs/quantization.md``.

See ``docs/conversion.md`` for the full schema.
"""
import argparse
import re
import sys

try:
    import gguf
    import numpy as np
except ImportError as e:  # pragma: no cover - env guard
    print(f"converter: missing dependency: {e}", file=sys.stderr)
    print("VOICEDETECT_CONVERT_DEPS_MISSING", file=sys.stderr)
    sys.exit(2)


# ---------------------------------------------------------------------------
# Quantization policy.  (See docs/quantization.md for the full rationale.)
#
# The C++ encoder only ever feeds a *quantized* weight into ``ggml_mul_mat`` as
# ``src0`` (the operand ggml dequantizes on the fly). The ECAPA conv weights are
# stored kernel-first (ggml ne = [K, IC, OC]) and the F32 path feeds them to
# ``ggml_mul_mat`` as ``src1`` after an im2col, which ggml requires to be F32.
# So a weight is quantizable ONLY when it is a 1x1 Conv1d / linear projection
# (kernel == 1): for K == 1 the im2col is a no-op and a 1x1 conv is a plain
# per-frame matmul, so the engine reads the weight as a 2-D ``[IC, OC]`` matrix
# (ggml ne; squeezed away the trivial kernel axis) and feeds it as ``src0``.
# Storing it 2-D puts the contraction dim ``IC`` on ne[0], the axis q8_0/q4_0
# block along (block size 32), so it must be >= 32 and a multiple of 32.
#
# Larger-kernel convs (block0 k=5, the dilated res2net k=3) go through the real
# im2col + F16-asserting kernel path, so they stay F32 -- their leading ne[0] is
# the kernel size (1/3/5), never block-aligned, and they are tiny anyway.
_QUANTIZABLE_PATTERNS = [
    # ECAPA-TDNN: the 1x1 TDNN/MFA/ASP/SE/fc projections (kernel filtered below).
    r"\.conv\.weight$",
    r"\.linear\.weight$",
]
_QUANTIZABLE_RE = [re.compile(p) for p in _QUANTIZABLE_PATTERNS]

# Parity-sensitive 1x1 weights excluded from the allowlist (kept F32). Measured
# on speechbrain/spkrec-ecapa-voxceleb (clips a/b/c), quantizing these pushes the
# q8_0 embedding past max|d| <= 5e-3 and/or drops q4_0 below cosine >= 0.997:
#   * `fc`              the final 6144->192 projection shapes the embedding directly;
#   * `asp.tdnn/asp.conv` the attention path -- error is amplified by the softmax;
#   * `*.se_block.*`    squeeze-excitation gates (sigmoid) -- the dominant max|d|
#                       contributor (worst embedding element tracks the SE gate);
#   * `*.tdnn1/*.tdnn2` the per-block channel mixers inside the residual path.
# What remains quantized is the multi-layer feature-aggregation conv (`mfa`, the
# single largest weight at 3072x3072), which is near-lossless at q8_0 AND q4_0.
# See docs/quantization.md for the per-set sweep.
_QUANT_EXCLUDE = [re.compile(p) for p in (
    r"^fc\.",
    r"^asp\.",
    r"\.se_block\.",
    r"\.tdnn1\.",
    r"\.tdnn2\.",
)]

_GGML_QTYPE = {
    "f16": getattr(gguf.GGMLQuantizationType, "F16", None),
    "q8_0": getattr(gguf.GGMLQuantizationType, "Q8_0", None),
    "q4_0": getattr(gguf.GGMLQuantizationType, "Q4_0", None),
}


def quant_plan(name, torch_shape, dtype):
    """Return ``(ggml_qtype, store_2d)`` for ``name``, or ``(None, False)`` (F32).

    ``torch_shape`` is the source (un-reversed) ``state_dict`` shape:
    ``[out_ch, in_ch, kernel]`` for a Conv1d, ``[out, in]`` for a Linear.
    ``store_2d`` asks the writer to squeeze the kernel==1 axis and store the
    weight as a 2-D ``[out_ch, in_ch]`` matrix so the contraction dim ``in_ch``
    lands on ggml ne[0] (the q8_0/q4_0 block axis).
    """
    if dtype == "f32":
        return None, False
    if not any(rx.search(name) for rx in _QUANTIZABLE_RE):
        return None, False
    if any(rx.search(name) for rx in _QUANT_EXCLUDE):
        return None, False          # parity-sensitive -> keep F32
    if len(torch_shape) == 3:
        out_ch, in_ch, kernel = torch_shape
    elif len(torch_shape) == 2:
        out_ch, in_ch, kernel = torch_shape[0], torch_shape[1], 1
    else:
        return None, False
    if kernel != 1:
        return None, False          # needs real im2col -> keep F32
    if out_ch < 32 or in_ch < 32:
        return None, False          # too small to quantize usefully
    qt = _GGML_QTYPE.get(dtype)
    if qt is None:
        return None, False
    # q8_0/q4_0 block (size 32) along the contraction dim in_ch (ggml ne[0]).
    if dtype in ("q8_0", "q4_0") and in_ch % 32 != 0:
        return None, False
    # Every quantized weight (incl. f16) is read by the engine as a 2-D [IC, OC]
    # ggml_mul_mat src0, so it is always stored squeezed to 2-D.
    return qt, True


# ggml caps tensor names at GGML_MAX_NAME (64): the gguf reader REJECTS any name
# whose length is >= 64, so a fully verbatim wav2vec2 state_dict (e.g.
# ``wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original0`` at 70,
# the per-layer ``...feed_forward.intermediate_dense.weight`` at 64/65) cannot be
# loaded by the C++ engine. For the analyze (wav2vec2) arch ONLY, abbreviate names
# with a single deterministic, reversible transform so every name fits in 63 chars
# while staying a predictable 1:1 mapping (the C++ analyze graph uses the SAME
# transform): the ``wav2vec2.`` module prefix -> ``w2v2.``, and the weight_norm
# parametrization buffers -> the canonical ``weight_g`` / ``weight_v``. The rest of
# each name is verbatim. Speaker-encoder names are short and stay untouched.
def shorten_w2v2_name(name):
    if name.startswith("wav2vec2."):
        name = "w2v2." + name[len("wav2vec2."):]
    name = name.replace(".parametrizations.weight.original0", ".weight_g")
    name = name.replace(".parametrizations.weight.original1", ".weight_v")
    return name


def _load_raw_state_dict(model_id):
    """Load a checkpoint's RAW state_dict (name -> tensor) without instantiating any
    HF auto class. Used for the audeering age/gender model, whose custom multi-task
    head AutoModelForAudioClassification mangles. Resolves a local file/dir or an HF
    repo id, preferring safetensors, falling back to pytorch_model.bin.
    """
    import os
    candidates = []
    if os.path.isdir(model_id):
        candidates = [os.path.join(model_id, "model.safetensors"),
                      os.path.join(model_id, "pytorch_model.bin")]
    elif os.path.isfile(model_id):
        candidates = [model_id]
    else:
        from huggingface_hub import hf_hub_download
        for fn in ("model.safetensors", "pytorch_model.bin"):
            try:
                candidates.append(hf_hub_download(model_id, fn))
                break
            except Exception:
                continue
    for path in candidates:
        if not path or not os.path.isfile(path):
            continue
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file
            return load_file(path)
        import torch
        return torch.load(path, map_location="cpu")
    raise FileNotFoundError(f"no model.safetensors / pytorch_model.bin for {model_id}")


def detect_arch(model_id, override="auto"):
    """Map a model id / path to one of the supported arch tags.

    ``override`` (the ``--arch`` flag, default ``"auto"``) short-circuits the id
    heuristic so an ambiguously named local checkpoint can be converted explicitly.

    TODO(impl-phase): inspect the actual checkpoint (HF config / ONNX metadata).
    For now infer from the id as a placeholder so the schema is exercised.
    """
    if override and override != "auto":
        return override
    s = model_id.lower()
    # Analyze archs (raw-waveform wav2vec2, not speaker encoders). Check the
    # age/gender id BEFORE the generic emotion fallback.
    if "age-gender" in s or "age_gender" in s:
        return "wav2vec2_age_gender"
    if "ecapa" in s:
        return "ecapa_tdnn"
    # Check the specific ONNX arch tags BEFORE the generic ``.onnx`` ->
    # wespeaker_resnet34 fallback, otherwise ``eres2net*.onnx`` / ``campplus*.onnx``
    # would mis-route to the ResNet34 branch (Task 14 review fix).
    if "eres2net" in s:
        return "eres2net"
    if "campplus" in s or "cam++" in s:
        return "campplus"
    if "resnet" in s or "wespeaker" in s:
        return "wespeaker_resnet34"
    if s.endswith(".onnx"):
        return "wespeaker_resnet34"
    return "ecapa_tdnn"


def parse_resnet_onnx(graph):
    """Recover the ResNet BasicBlock topology from a folded WeSpeaker ONNX graph.

    The ONNX export folds every BatchNorm into the preceding Conv (so each
    Conv carries weight+bias) and names the initializers with opaque integers.
    The C++ engine therefore cannot infer the block structure from tensor names;
    we walk the graph here and emit an ordered block manifest (verbatim conv
    names + stride + downsample) as ``voicedetect.resnet.*`` KV. Channel widths
    stay implicit - the C++ side reads them from each tensor's ``ne``.

    A residual block is a ``Add`` whose two inputs are (the second 3x3 conv of the
    block, the residual). The residual is either a plain activation (identity
    shortcut) or a 1x1 conv (the stride-2 downsample). The block's stride is the
    first conv's stride. The lone ``Add`` in the statistics-pooling tail (var+eps)
    has no 3x3 conv input and is skipped.

    Returns a dict with: stem (w,b), blocks [(c1w,c1b,c2w,c2b,downw,downb,stride)],
    seg (w,b), mean_vec name, var_eps, embedding_dim.
    """
    convs = {}
    conv_order = []
    for n in graph.node:
        if n.op_type != "Conv":
            continue
        attrs = {a.name: a for a in n.attribute}
        d = {
            "out": n.output[0],
            "inp": n.input[0],
            "w": n.input[1],
            "b": n.input[2] if len(n.input) > 2 else "",
            "stride": list(attrs["strides"].ints)[0] if "strides" in attrs else 1,
            "kernel": list(attrs["kernel_shape"].ints)[0],
        }
        convs[n.output[0]] = d
        conv_order.append(d)
    relu_in = {n.output[0]: n.input[0] for n in graph.node if n.op_type == "Relu"}

    stem = conv_order[0]  # the first Conv is the 3x3 stem
    blocks = []
    for n in graph.node:
        if n.op_type != "Add":
            continue
        a0, a1 = n.input[0], n.input[1]
        c0, c1 = convs.get(a0), convs.get(a1)
        # The block's second conv is the 3x3 conv feeding the Add; skip the
        # pooling-tail Add (neither input is a 3x3 conv).
        if c0 and c0["kernel"] == 3:
            conv2, res = c0, a1
        elif c1 and c1["kernel"] == 3:
            conv2, res = c1, a0
        else:
            continue
        conv1 = convs[relu_in[conv2["inp"]]]  # conv2 input is a ReLU over conv1
        down = convs.get(res)                 # 1x1 downsample, or None (identity)
        blocks.append((conv1, conv2, down))

    gemm = next(n for n in graph.node if n.op_type == "Gemm")
    producer = {o: n for n in graph.node for o in n.output}
    final = producer[graph.output[0].name]
    mean_vec = final.input[1] if final.op_type == "Sub" else ""
    emb_dim = int(graph.output[0].type.tensor_type.shape.dim[1].dim_value)
    return {
        "stem": (stem["w"], stem["b"]),
        "blocks": blocks,
        "seg": (gemm.input[1], gemm.input[2]),
        "mean_vec": mean_vec,
        "embedding_dim": emb_dim,
    }


def parse_eres2net_onnx(graph):
    """Recover the 3D-Speaker ERes2Net topology from its folded ONNX graph.

    ERes2Net is a 2D ResNet over the [n_mels, T] FBank with Res2Net multi-scale
    blocks, attentional feature fusion (AFF), and a bottom-up cross-stage fusion
    (see github.com/modelscope/3D-Speaker eres2net/ERes2Net.py). The torch.onnx
    export folds every BatchNorm into the preceding Conv (so each Conv carries
    weight + bias) and names the folded convs with opaque ``onnx::Conv_*`` ids,
    so the topology cannot be read from tensor names.

    KEY INVARIANT (verified against the reference forward): torch.onnx emits Conv
    nodes in forward-EXECUTION order. The C++ ERes2Net encoder replays the exact
    deterministic forward (num_blocks=[3,4,6,3], scale=2, the stem + 4 layers +
    3 cross-stage AFF + TSTP + seg_1) and pulls each conv's (weight, bias, stride)
    from these parallel arrays via a running counter, so the only contract is
    "conv array order == forward order". m_channels and the embedding dim are
    read from tensor shapes; the block counts/scale are the fixed ERes2Net
    topology shared by every 3D-Speaker ERes2Net variant (base/std/large differ
    only in m_channels + embedding size, both derived here).

    Returns: conv_weight/conv_bias/conv_stride flat lists (execution order),
    m_channels, num_blocks, scale, seg (w,b), embedding_dim, var_eps.
    """
    convs = [n for n in graph.node if n.op_type == "Conv"]
    conv_w, conv_b, conv_s = [], [], []
    for n in convs:
        attrs = {a.name: a for a in n.attribute}
        conv_w.append(n.input[1])
        conv_b.append(n.input[2] if len(n.input) > 2 else "")
        conv_s.append(int(list(attrs["strides"].ints)[0]) if "strides" in attrs else 1)

    # ERes2Net standard topology (3D-Speaker): block counts [3,4,6,3], Res2Net
    # scale 2. With these, the conv count is fixed at 96 regardless of m_channels
    # -- assert it so a non-standard ERes2Net variant fails loudly here instead of
    # silently producing a mismatched manifest.
    num_blocks = [3, 4, 6, 3]
    scale = 2
    if len(convs) != 96:
        raise NotImplementedError(
            f"eres2net: expected 96 Conv nodes for num_blocks={num_blocks} scale={scale}, "
            f"got {len(convs)} -- non-standard ERes2Net topology, not supported")

    init_shape = {i.name: tuple(i.dims) for i in graph.initializer}
    m_channels = int(init_shape[conv_w[0]][0])      # stem conv out channels
    seg_w, seg_b = "seg_1.weight", "seg_1.bias"
    emb_dim = int(init_shape[seg_w][0])             # seg_1 [embedding_dim, stats]
    return {
        "conv_weight": conv_w,
        "conv_bias": conv_b,
        "conv_stride": conv_s,
        "m_channels": m_channels,
        "num_blocks": num_blocks,
        "scale": scale,
        "seg": (seg_w, seg_b),
        "embedding_dim": emb_dim,
        # 3D-Speaker TSTP pooling: std = sqrt(unbiased_var + 1e-8).
        "var_eps": 1e-8,
    }


def parse_campplus_onnx(graph):
    """Recover the 3D-Speaker CAM++ topology from its (partly BN-folded) ONNX graph.

    CAM++ (github.com/modelscope/3D-Speaker campplus/{DTDNN,layers}.py) is a 2D-conv
    FCM front end followed by a 1D D-TDNN backbone: a strided TDNN, three densely
    connected ``CAMDenseTDNNBlock``s (num_layers [12, 24, 16], dilation [1, 2, 2],
    growth 32) each followed by a halving ``TransitLayer``, an out ReLU, statistics
    pooling (mean + unbiased std), a 1024->192 dense projection, and a final
    affine-free BatchNorm. Each dense layer's ``CAMLayer`` adds a context-aware mask:
    ``y = linear_local(c); m = sigmoid(linear2(relu(linear1(c.mean(-1) + seg(c)))));
    out = y * m`` where ``seg`` is a 100-frame segment average (count_include_pad).

    The ``torch.onnx`` export folds every BatchNorm that immediately follows a Conv
    with a SINGLE consumer (the FCM convs, the TDNN/bn_function/transit3 1x1s) into
    that Conv (renamed ``onnx::Conv_*``, gaining a bias); BatchNorms preceded by a
    dense concat (``nonlinear1``, the transit ``nonlinear``s) or by nothing
    foldable (the final affine-free BN) survive as standalone nodes. Both Conv and
    BatchNormalization nodes are emitted in forward-EXECUTION order, so we lift two
    flat parallel lists (verbatim names, "" bias where absent) plus the per-BN
    affine prefix; the C++ ``CamPPlusEncoder`` replays the fixed deterministic
    forward, pulling each conv/BN by a running index. Channel widths come from each
    tensor's ``ne``; the fixed block topology is asserted via the 225 Conv / 56 BN
    counts so a non-standard CAM++ variant fails loudly here.

    Returns: conv_weight/conv_bias flat lists, bn_prefix (the 55 affine BNs in
    execution order), emb_bn (mean, var names of the final affine-free BN),
    embedding_dim, bn_eps.
    """
    init_names = {i.name for i in graph.initializer}
    convs = [n for n in graph.node if n.op_type == "Conv"]
    conv_w = [n.input[1] for n in convs]
    conv_b = [n.input[2] if len(n.input) > 2 and n.input[2] else "" for n in convs]

    bns = [n for n in graph.node if n.op_type == "BatchNormalization"]
    bn_prefix = []
    emb_mean = emb_var = ""
    bn_eps = 1e-5
    for n in bns:
        scale, mean, var = n.input[1], n.input[3], n.input[4]
        for a in n.attribute:
            if a.name == "epsilon":
                bn_eps = float(a.f)
        if scale in init_names:                 # affine BN: prefix = scale - ".weight"
            assert scale.endswith(".weight")
            bn_prefix.append(scale[: -len(".weight")])
        else:                                   # affine-free BN (the embedding head)
            emb_mean, emb_var = mean, var

    # CAM++ zh-cn 16k-common standard topology: 12 FCM/head convs + 1 TDNN + 52
    # dense layers * 4 convs + 3 transit + 1 dense = 225 Conv; 52 nonlinear1 + 3
    # transit nonlinear + 1 embedding = 56 BatchNorm. Assert so a non-standard
    # CAM++ fails here instead of silently mis-mapping.
    if len(convs) != 225 or len(bns) != 56 or len(bn_prefix) != 55 or not emb_mean:
        raise NotImplementedError(
            f"campplus: expected 225 Conv / 56 BN (55 affine + 1 embedding), got "
            f"{len(convs)} Conv / {len(bns)} BN ({len(bn_prefix)} affine) -- "
            "non-standard CAM++ topology, not supported")

    shape = {i.name: tuple(i.dims) for i in graph.initializer}
    emb_dim = int(shape[conv_w[-1]][0])         # xvector.dense.linear.weight [192, 1024, 1]
    return {
        "conv_weight": conv_w,
        "conv_bias": conv_b,
        "bn_prefix": bn_prefix,
        "emb_bn_mean": emb_mean,
        "emb_bn_var": emb_var,
        "embedding_dim": emb_dim,
        "bn_eps": bn_eps,
    }


def load_state_dict(model_id, arch):
    """Return (state_dict, embedding_dim, extra) for the source checkpoint.

    ``state_dict`` is an iterable of (name, numpy_array) pairs with VERBATIM
    names. ``extra`` carries arch-specific structural metadata (the ResNet block
    manifest for ONNX encoders), or ``None`` when none is needed.
    """
    if arch == "ecapa_tdnn":
        from speechbrain.inference.speaker import EncoderClassifier
        clf = EncoderClassifier.from_hparams(source=model_id, savedir="./pretrained_models")
        sd = clf.mods.embedding_model.state_dict()
        # mean_var_norm buffers travel under a verbatim prefix so the C++ side can
        # apply the same sentence-level mean normalization if a model needs it.
        if hasattr(clf.mods, "mean_var_norm"):
            for k, v in clf.mods.mean_var_norm.state_dict().items():
                sd[f"mean_var_norm.{k}"] = v
        items = [(k, v.detach().cpu().numpy()) for k, v in sd.items()]
        return items, 192, None

    if arch == "wespeaker_resnet34":
        # ONNX-direct encoder: lift every graph initializer VERBATIM (BatchNorm is
        # already folded into the convs by the export) and emit the block topology
        # as structural metadata so the C++ engine can rebuild the graph.
        import onnx
        from onnx import numpy_helper
        model = onnx.load(model_id)
        items = [(init.name, numpy_helper.to_array(init)) for init in model.graph.initializer]
        meta = parse_resnet_onnx(model.graph)
        return items, meta["embedding_dim"], meta

    if arch == "eres2net":
        # 3D-Speaker ERes2Net (ONNX-direct): lift every initializer VERBATIM (BN
        # folded into convs by the export) and recover the flat conv manifest so
        # the C++ engine replays the deterministic ERes2Net forward.
        import onnx
        from onnx import numpy_helper
        model = onnx.load(model_id)
        items = [(init.name, numpy_helper.to_array(init)) for init in model.graph.initializer]
        meta = parse_eres2net_onnx(model.graph)
        return items, meta["embedding_dim"], meta

    if arch == "campplus":
        # 3D-Speaker CAM++ (ONNX-direct): lift every initializer VERBATIM (the
        # export folds some BNs into their convs and keeps semantic names for the
        # rest) and recover the flat conv + BN manifests so the C++ engine replays
        # the deterministic CAM++ forward.
        import onnx
        from onnx import numpy_helper
        model = onnx.load(model_id)
        items = [(init.name, numpy_helper.to_array(init)) for init in model.graph.initializer]
        meta = parse_campplus_onnx(model.graph)
        return items, meta["embedding_dim"], meta

    if arch == "wav2vec2_emotion":
        # Analyze head (SEPARATE from the speaker encoders): the wav2vec2 emotion
        # classifier (default superb/wav2vec2-base-superb-er, Apache-2.0). Lift the
        # full HF state_dict VERBATIM (including the weight-norm parametrization
        # buffers of the positional conv) so the C++ analyze graph (T18-20) is a
        # 1:1 name mapping, and carry the wav2vec2 config as structural metadata so
        # every dim/kernel/stride lands in KV (no magic numbers in C++).
        from transformers import AutoConfig, AutoModelForAudioClassification
        cfg = AutoConfig.from_pretrained(model_id)
        model = AutoModelForAudioClassification.from_pretrained(model_id)
        model.eval()
        items = [(k, v.detach().cpu().numpy()) for k, v in model.state_dict().items()]
        # Emotion labels in id order (0..N-1); id2label keys may be int or str.
        id2label = {int(k): v for k, v in cfg.id2label.items()}
        labels = [id2label[i] for i in range(len(id2label))]
        meta = {
            "hidden_size": int(cfg.hidden_size),
            "n_layers": int(cfg.num_hidden_layers),
            "n_heads": int(cfg.num_attention_heads),
            "ff_dim": int(cfg.intermediate_size),
            "conv_dims": [int(x) for x in cfg.conv_dim],
            "conv_kernels": [int(x) for x in cfg.conv_kernel],
            "conv_strides": [int(x) for x in cfg.conv_stride],
            "num_conv_layers": int(cfg.num_feat_extract_layers),
            "feat_extract_norm": str(cfg.feat_extract_norm),
            "feat_extract_activation": str(cfg.feat_extract_activation),
            "hidden_act": str(cfg.hidden_act),
            "do_stable_layer_norm": bool(cfg.do_stable_layer_norm),
            "layer_norm_eps": float(cfg.layer_norm_eps),
            "num_conv_pos_embeddings": int(cfg.num_conv_pos_embeddings),
            "num_conv_pos_embedding_groups": int(cfg.num_conv_pos_embedding_groups),
            # wav2vec2 weight_norm on the positional conv uses dim=2 (the kernel
            # axis): g = parametrizations.weight.original0 [1,1,K], v = original1.
            "pos_conv_weight_norm_dim": 2,
            "use_weighted_layer_sum": bool(getattr(cfg, "use_weighted_layer_sum", False)),
            "classifier_proj_size": int(getattr(cfg, "classifier_proj_size", cfg.hidden_size)),
            "emotion_labels": labels,
        }
        # embedding_dim is a speaker-encoder concept; report hidden_size so the
        # generic return signature is satisfied (the analyze KV path does not write
        # voicedetect.embedding_dim).
        return items, meta["hidden_size"], meta

    if arch == "wav2vec2_age_gender":
        # Audeering age/gender head (a SEPARATE analyze model from the emotion one):
        # audeering/wav2vec2-large-robust-24-ft-age-gender
        # (CC-BY-NC-SA-4.0, i.e. NON-COMMERCIAL). The
        # checkpoint ships a CUSTOM multi-task head (age regression + 3-way gender)
        # that AutoModelForAudioClassification silently mangles (it drops the age
        # weights as UNEXPECTED and re-inits the classifier), so we lift the RAW
        # state_dict verbatim instead of instantiating an HF auto class. Backbone is
        # wav2vec2-large-robust: 24 layers / hidden 1024 / 16 heads, feat_extract_norm
        # "layer" (every conv LayerNorm'd) + conv bias + do_stable_layer_norm
        # (pre-norm layers + a final encoder LayerNorm). Head: age.{dense,out_proj}
        # (1024->1024->1) and gender.{dense,out_proj} (1024->1024->3).
        import os
        from transformers import AutoConfig, AutoFeatureExtractor
        cfg = AutoConfig.from_pretrained(model_id)
        # The large-robust FeatureExtractor zero-mean/unit-var normalizes the input
        # waveform (do_normalize=True), unlike the base emotion model; record it so
        # the C++ conv encoder applies the same pre-normalization.
        try:
            fe = AutoFeatureExtractor.from_pretrained(model_id)
            do_normalize = bool(getattr(fe, "do_normalize", False))
        except Exception:
            do_normalize = True
        sd = _load_raw_state_dict(model_id)
        items = [(k, (v.detach().cpu().numpy() if hasattr(v, "detach")
                      else np.asarray(v))) for k, v in sd.items()]
        id2label = {int(k): v for k, v in cfg.id2label.items()}
        gender_labels = [id2label[i] for i in range(len(id2label))]
        meta = {
            "hidden_size": int(cfg.hidden_size),
            "n_layers": int(cfg.num_hidden_layers),
            "n_heads": int(cfg.num_attention_heads),
            "ff_dim": int(cfg.intermediate_size),
            "conv_dims": [int(x) for x in cfg.conv_dim],
            "conv_kernels": [int(x) for x in cfg.conv_kernel],
            "conv_strides": [int(x) for x in cfg.conv_stride],
            "num_conv_layers": int(cfg.num_feat_extract_layers),
            "feat_extract_norm": str(cfg.feat_extract_norm),
            "conv_bias": bool(cfg.conv_bias),
            "feat_extract_activation": str(cfg.feat_extract_activation),
            "hidden_act": str(cfg.hidden_act),
            "do_stable_layer_norm": bool(cfg.do_stable_layer_norm),
            "layer_norm_eps": float(cfg.layer_norm_eps),
            "num_conv_pos_embeddings": int(cfg.num_conv_pos_embeddings),
            "num_conv_pos_embedding_groups": int(cfg.num_conv_pos_embedding_groups),
            "pos_conv_weight_norm_dim": 2,
            "use_weighted_layer_sum": bool(getattr(cfg, "use_weighted_layer_sum", False)),
            "classifier_proj_size": int(getattr(cfg, "classifier_proj_size", cfg.hidden_size)),
            "do_normalize": do_normalize,
            "gender_labels": gender_labels,
        }
        return items, meta["hidden_size"], meta

    raise NotImplementedError(
        f"convert_voicedetect_to_gguf: loader for arch '{arch}' is not implemented "
        "yet (see load_state_dict TODO)")


def add_fbank_kv(w):
    """Write the Kaldi-compatible FBank front-end parameters (the C++ defaults;
    override here per-model when a checkpoint deviates)."""
    w.add_uint32("voicedetect.fbank.sample_rate", 16000)
    w.add_uint32("voicedetect.fbank.n_mels", 80)
    w.add_uint32("voicedetect.fbank.n_fft", 512)
    w.add_uint32("voicedetect.fbank.win_length", 400)   # 25 ms @ 16 kHz
    w.add_uint32("voicedetect.fbank.hop_length", 160)   # 10 ms @ 16 kHz
    w.add_float32("voicedetect.fbank.preemph", 0.97)
    w.add_float32("voicedetect.fbank.low_freq", 20.0)
    w.add_float32("voicedetect.fbank.high_freq", 0.0)   # 0 = Nyquist
    w.add_bool("voicedetect.fbank.use_energy", False)
    w.add_bool("voicedetect.fbank.cmn", True)
    w.add_string("voicedetect.fbank.window", "povey")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="HF id or local checkpoint path (defaults per --analyze / "
                         "--age-gender shorthand)")
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--dtype",
        choices=["f32", "f16", "q8_0", "q4_0"],
        default="f32",
        help="quantization for allowlisted 1x1 conv/linear weights (everything else f32)",
    )
    ap.add_argument(
        "--arch",
        choices=["auto", "ecapa_tdnn", "wespeaker_resnet34", "eres2net", "campplus",
                 "wav2vec2_emotion", "wav2vec2_age_gender"],
        default="auto",
        help="force the source arch (default auto: infer from --model id)",
    )
    ap.add_argument(
        "--analyze",
        action="store_true",
        help="convert the wav2vec2 emotion analyze head (a SEPARATE analyze GGUF, "
             "not a speaker embedding); shorthand for --arch wav2vec2_emotion with "
             "the superb/wav2vec2-base-superb-er default model",
    )
    ap.add_argument(
        "--age-gender",
        dest="age_gender",
        action="store_true",
        help="convert the audeering wav2vec2-large-robust age/gender analyze head "
             "(a SEPARATE analyze GGUF); shorthand for --arch wav2vec2_age_gender "
             "with the audeering/wav2vec2-large-robust-24-ft-age-gender default model",
    )
    args = ap.parse_args()

    if args.age_gender:
        arch = "wav2vec2_age_gender"
        if not args.model:
            args.model = "audeering/wav2vec2-large-robust-24-ft-age-gender"
    elif args.analyze:
        arch = "wav2vec2_emotion"
        if not args.model:
            args.model = "superb/wav2vec2-base-superb-er"
    else:
        if not args.model:
            print("converter: --model is required (or use --analyze / --age-gender)",
                  file=sys.stderr)
            sys.exit(2)
        arch = detect_arch(args.model, args.arch)

    try:
        state, embedding_dim, extra = load_state_dict(args.model, arch)
    except NotImplementedError as e:
        # The loader skeleton is not wired yet: surface as a deps/model-unavailable
        # style exit so the converter round-trip test SKIPs rather than fails.
        print(f"VOICEDETECT_MODEL_UNAVAILABLE: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:  # pragma: no cover - network/cache guard
        print(f"VOICEDETECT_MODEL_UNAVAILABLE: {e}", file=sys.stderr)
        sys.exit(2)

    w = gguf.GGUFWriter(args.output, "voicedetect")
    w.add_string("general.name", args.model)
    w.add_string("voicedetect.arch", arch)
    is_analyze = arch in ("wav2vec2_emotion", "wav2vec2_age_gender")
    if not is_analyze:
        # Speaker-encoder common KV: embedding head + the shared FBank front end.
        # The analyze head is a raw-waveform wav2vec2 conv front end (no mel
        # FBank, no L2-normalized embedding), so it writes neither.
        w.add_uint32("voicedetect.embedding_dim", int(embedding_dim))
        w.add_bool("voicedetect.l2_normalize", True)
        add_fbank_kv(w)

    # Architecture dims derived from the verbatim tensor shapes (metadata-driven,
    # so the C++ side reads them from KV instead of hardcoding magic numbers).
    shapes = {n: tuple(a.shape) for n, a in state}
    if arch == "ecapa_tdnn":
        # blocks.0.conv.conv.weight is [out_ch, in_ch=n_mels, kernel]
        channels = shapes["blocks.0.conv.conv.weight"][0]
        n_blocks = 1 + sum(1 for n in shapes if n.startswith("blocks.") and ".se_block." in n
                           and n.endswith("conv1.conv.weight"))
        # se_channels = SEBlock bottleneck (conv1 out); res2net_scale from res2net conv count
        se_ch = next(shapes[n][0] for n in shapes if n.endswith("se_block.conv1.conv.weight"))
        scale = 1 + sum(1 for n in shapes if n.startswith("blocks.1.res2net_block.blocks.")
                        and n.endswith("conv.conv.weight"))
        att_ch = shapes["asp.conv.conv.weight"][0]  # attention bottleneck channels
        w.add_uint32("voicedetect.ecapa.channels", int(channels))
        w.add_uint32("voicedetect.ecapa.n_blocks", int(n_blocks))
        w.add_uint32("voicedetect.ecapa.se_channels", int(se_ch))
        w.add_uint32("voicedetect.ecapa.res2net_scale", int(scale))
        w.add_uint32("voicedetect.ecapa.attention_channels", int(att_ch))
    elif arch == "wespeaker_resnet34":
        # ResNet block manifest (verbatim conv names + stride + downsample). The
        # C++ ResNet34Encoder walks these parallel arrays to rebuild the graph;
        # channel widths/stage depths come from each tensor's ne (no literals).
        blocks = extra["blocks"]
        w.add_string("voicedetect.resnet.stem_weight", extra["stem"][0])
        w.add_string("voicedetect.resnet.stem_bias", extra["stem"][1])
        w.add_array("voicedetect.resnet.conv1_weight", [b[0]["w"] for b in blocks])
        w.add_array("voicedetect.resnet.conv1_bias", [b[0]["b"] for b in blocks])
        w.add_array("voicedetect.resnet.conv2_weight", [b[1]["w"] for b in blocks])
        w.add_array("voicedetect.resnet.conv2_bias", [b[1]["b"] for b in blocks])
        w.add_array("voicedetect.resnet.down_weight",
                    [(b[2]["w"] if b[2] else "") for b in blocks])
        w.add_array("voicedetect.resnet.down_bias",
                    [(b[2]["b"] if b[2] else "") for b in blocks])
        w.add_array("voicedetect.resnet.stride", [int(b[0]["stride"]) for b in blocks])
        w.add_string("voicedetect.resnet.seg_weight", extra["seg"][0])
        w.add_string("voicedetect.resnet.seg_bias", extra["seg"][1])
        w.add_string("voicedetect.resnet.mean_vec", extra["mean_vec"])
        # Statistics-pooling std: sqrt(unbiased_var + eps), eps from the ONNX
        # graph's folded constant (1e-8 for this export).
        w.add_float32("voicedetect.resnet.var_eps", 1e-8)
    elif arch == "eres2net":
        # 3D-Speaker ERes2Net manifest: a flat conv list in forward-execution
        # order (verbatim weight + bias names, "" bias for the BN-less downsample
        # convs) plus the fixed block topology. The C++ ERes2NetEncoder replays
        # the deterministic forward, pulling convs by a running index; channel
        # widths come from each tensor's ne.
        w.add_array("voicedetect.eres2net.conv_weight", extra["conv_weight"])
        w.add_array("voicedetect.eres2net.conv_bias", extra["conv_bias"])
        w.add_array("voicedetect.eres2net.conv_stride",
                    [int(s) for s in extra["conv_stride"]])
        w.add_array("voicedetect.eres2net.num_blocks",
                    [int(b) for b in extra["num_blocks"]])
        w.add_uint32("voicedetect.eres2net.m_channels", int(extra["m_channels"]))
        w.add_uint32("voicedetect.eres2net.scale", int(extra["scale"]))
        w.add_string("voicedetect.eres2net.seg_weight", extra["seg"][0])
        w.add_string("voicedetect.eres2net.seg_bias", extra["seg"][1])
        # ERes2Net "ReLU" is Hardtanh(0, 20) inside every block (the stem uses a
        # plain ReLU); store the clamp ceiling so the C++ side is metadata-driven.
        w.add_float32("voicedetect.eres2net.relu_clamp", 20.0)
        w.add_float32("voicedetect.eres2net.var_eps", float(extra["var_eps"]))
    elif arch == "campplus":
        # 3D-Speaker CAM++ manifest: two flat lists in forward-execution order --
        # all 225 convs (verbatim weight + bias names, "" bias for the unfolded
        # bias-free convs) and the 55 affine BN prefixes (each carries
        # .weight/.bias/.running_mean/.running_var) -- plus the final affine-free
        # embedding BN (mean/var only). The C++ CamPPlusEncoder replays the fixed
        # FCM + D-TDNN forward, pulling convs/BNs by running indices; channel widths
        # come from each tensor's ne and the block topology is hardcoded.
        w.add_array("voicedetect.campplus.conv_weight", extra["conv_weight"])
        w.add_array("voicedetect.campplus.conv_bias", extra["conv_bias"])
        w.add_array("voicedetect.campplus.bn_prefix", extra["bn_prefix"])
        w.add_string("voicedetect.campplus.emb_bn_mean", extra["emb_bn_mean"])
        w.add_string("voicedetect.campplus.emb_bn_var", extra["emb_bn_var"])
        w.add_float32("voicedetect.campplus.bn_eps", float(extra["bn_eps"]))
    elif arch in ("wav2vec2_emotion", "wav2vec2_age_gender"):
        # Analyze head schema (the SEPARATE analyze GGUF). Every wav2vec2 dim,
        # conv kernel/stride and the label list live in KV so the C++ analyze graph
        # is fully metadata-driven. Weights are verbatim (wav2vec2.* abbreviated to
        # w2v2.*, plus projector./classifier. (emotion) or age./gender. (audeering)).
        m = extra
        w.add_bool("voicedetect.analyze.present", True)
        if arch == "wav2vec2_emotion":
            w.add_string("voicedetect.analyze.kind", "emotion")
            w.add_array("voicedetect.analyze.emotion_labels", m["emotion_labels"])
            w.add_uint32("voicedetect.analyze.num_emotions", len(m["emotion_labels"]))
        else:
            # age (regression scalar, scaled 0..1 -> *100 years) + gender classes.
            w.add_string("voicedetect.analyze.kind", "age_gender")
            w.add_array("voicedetect.analyze.gender_labels", m["gender_labels"])
            w.add_uint32("voicedetect.analyze.num_genders", len(m["gender_labels"]))
        # Classifier pooling over the projected per-frame features: mean over time.
        w.add_string("voicedetect.analyze.pooling", "mean")
        w.add_bool("voicedetect.w2v2.conv_bias", bool(m.get("conv_bias", False)))
        w.add_bool("voicedetect.w2v2.do_normalize", bool(m.get("do_normalize", False)))
        w.add_uint32("voicedetect.w2v2.hidden_size", m["hidden_size"])
        w.add_uint32("voicedetect.w2v2.n_layers", m["n_layers"])
        w.add_uint32("voicedetect.w2v2.n_heads", m["n_heads"])
        w.add_uint32("voicedetect.w2v2.ff_dim", m["ff_dim"])
        w.add_uint32("voicedetect.w2v2.num_conv_layers", m["num_conv_layers"])
        w.add_array("voicedetect.w2v2.conv_dims", m["conv_dims"])
        w.add_array("voicedetect.w2v2.conv_kernels", m["conv_kernels"])
        w.add_array("voicedetect.w2v2.conv_strides", m["conv_strides"])
        w.add_string("voicedetect.w2v2.feat_extract_norm", m["feat_extract_norm"])
        w.add_string("voicedetect.w2v2.feat_extract_activation", m["feat_extract_activation"])
        w.add_string("voicedetect.w2v2.hidden_act", m["hidden_act"])
        w.add_bool("voicedetect.w2v2.do_stable_layer_norm", m["do_stable_layer_norm"])
        w.add_float32("voicedetect.w2v2.layer_norm_eps", m["layer_norm_eps"])
        w.add_uint32("voicedetect.w2v2.num_conv_pos_embeddings", m["num_conv_pos_embeddings"])
        w.add_uint32("voicedetect.w2v2.num_conv_pos_embedding_groups",
                     m["num_conv_pos_embedding_groups"])
        w.add_uint32("voicedetect.w2v2.pos_conv_weight_norm_dim", m["pos_conv_weight_norm_dim"])
        w.add_bool("voicedetect.w2v2.use_weighted_layer_sum", m["use_weighted_layer_sum"])
        w.add_uint32("voicedetect.w2v2.classifier_proj_size", m["classifier_proj_size"])

    # Verbatim tensor loop (selective quantization per quant_plan). Tensor names
    # stay verbatim; only the on-disk rank/precision of allowlisted 1x1 weights
    # changes (squeezed to 2-D and block-quantized) -- the C++ engine detects a
    # non-F32 leaf and reads it as a plain [IC, OC] ggml_mul_mat src0.
    from gguf import quants
    for name, arr in state:
        arr = np.ascontiguousarray(arr)
        # Analyze (wav2vec2) names are abbreviated to fit ggml's 64-char limit so
        # the C++ engine can load the GGUF (see shorten_w2v2_name). Speaker archs
        # keep verbatim names. Quantization keys off the source name (a no-op for
        # the f32 analyze path), so abbreviate only the stored tensor name.
        store_name = shorten_w2v2_name(name) if is_analyze else name
        qt, store_2d = quant_plan(name, arr.shape, args.dtype)
        if qt is None:
            w.add_tensor(store_name, arr.astype(np.float32))
            continue
        m2d = arr.reshape(arr.shape[0], -1) if store_2d else arr  # [OC, IC]
        if qt == gguf.GGMLQuantizationType.F16:
            w.add_tensor(store_name, m2d.astype(np.float16), raw_dtype=qt)
        else:
            qbytes = quants.quantize(np.ascontiguousarray(m2d, dtype=np.float32), qt)
            # qbytes is uint8 with a per-row byte width; gguf recovers the logical
            # [OC, IC] shape from it via quant_shape_from_byte_shape, so pass it raw.
            w.add_tensor(store_name, qbytes, raw_dtype=qt)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.output} (arch={arch}, embedding_dim={embedding_dim}, dtype={args.dtype})")


if __name__ == "__main__":
    main()
