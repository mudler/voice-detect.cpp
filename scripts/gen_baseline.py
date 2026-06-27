#!/usr/bin/env python3
"""Dump reference intermediate tensors to ``baseline.gguf`` for C++ parity.

Validates each C++ stage (FBank -> encoder -> pooling -> embedding) by diffing
against the exact tensors this script captures from the SpeechBrain ECAPA-TDNN
reference. Correctness and determinism are paramount: dithering is OFF, the
forward runs under ``torch.no_grad()`` in ``eval()``.

Stored tensors (squeezed; f32). Axis order documented in docs/conversion.md:

* ``fbank_raw``   ``[n_mels, T]``    torchaudio.compliance.kaldi.fbank output,
                                     pre per-utterance CMN.
* ``fbank``       ``[n_mels, T]``    after per-utterance cepstral mean
                                     normalization (CMN), the front end the
                                     C++ FBank+CMN stage must match.
* ``block{0..3}_out`` ``[C, T]``     ECAPA SE-Res2Net block outputs (channel-major).
* ``mfa_out``     ``[3072, T]``      multi-layer feature aggregation output.
* ``pooled``      ``[6144]``         attentive statistics pooling output
                                     (mean+std over 3072 channels).
* ``embedding``   ``[192]``          final L2-normalized speaker embedding.

Front-end note (PARITY-CRITICAL): every encoder golden here is captured by
running ``embedding_model`` directly on the *Kaldi* CMN FBank above - the SAME
front end the C++ engine uses - NOT via ``encode_batch``. ``encode_batch`` would
run SpeechBrain's internal ``compute_features`` (a different STFT/framing on a
different value scale, e.g. T=301 vs the Kaldi T=298), producing block/mfa/pooled
tensors the kaldi-fed C++ encoder can never reproduce. Feeding the Kaldi features
straight into ``embedding_model.forward([B, T, n_mels])`` keeps fbank ->
block0..3 -> mfa -> pooled -> embedding a single self-consistent chain.

Embedding note: the captured ``embedding`` is the raw ``embedding_model`` output
(before the classifier path's ``mean_var_norm_emb``, which is verification-only
and NOT part of the engine's embedding), unit-L2-normalized so the golden sits on
the same scale the C++ embedding gate (max-abs-diff) compares against. Cosine
similarity is invariant to that scaling.

These are the golden values for tests/test_fbank.cpp, the encoder component
tests, and the embedding parity gate (see docs/parity.md).

Exit codes: 0 = wrote baseline, 2 = deps/model unavailable.
"""
import argparse
import sys

try:
    import gguf
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio  # noqa: F401  (registers the kaldi compliance ops)
    import torchaudio.compliance.kaldi as kaldi
except ImportError as e:  # pragma: no cover - env guard
    print(f"VOICEDETECT_CONVERT_DEPS_MISSING: {e}", file=sys.stderr)
    sys.exit(2)


def _sq(a):
    return np.ascontiguousarray(np.squeeze(a).astype(np.float32))


def _load_audio_16k(path):
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != 16000:
        n = int(round(len(audio) * 16000 / sr))
        audio = np.interp(np.linspace(0, len(audio), n, endpoint=False),
                          np.arange(len(audio)), audio).astype(np.float32)
    return audio


def _kaldi_fbank_cmn(audio):
    """Kaldi FBank (povey window, preemph 0.97, dither 0) + per-utterance CMN -
    the exact shared C++ front end. Returns (fb_cmn [T,80], fbank_raw [80,T])."""
    wav = torch.from_numpy(audio).unsqueeze(0)  # [1, T]
    fb = kaldi.fbank(wav, sample_frequency=16000, num_mel_bins=80,
                     frame_length=25.0, frame_shift=10.0, dither=0.0)  # [T, 80]
    fbank_raw = fb.numpy().T                                  # [80, T]
    fb_cmn = fb - fb.mean(dim=0, keepdim=True)                # CMN, [T, 80]
    return fb_cmn, fbank_raw


def gen_ecapa(args, w):
    from speechbrain.inference.speaker import EncoderClassifier
    try:
        clf = EncoderClassifier.from_hparams(source=args.model, savedir="./pretrained_models")
    except Exception as e:
        print(f"VOICEDETECT_MODEL_UNAVAILABLE: {e}", file=sys.stderr)
        sys.exit(2)
    clf.mods.eval()

    audio = _load_audio_16k(args.audio)
    fb_cmn, fbank_raw = _kaldi_fbank_cmn(audio)
    fbank = fb_cmn.numpy().T                                  # [80, T]

    cap = {}

    def hook(name):
        def f(m, i, o):
            t = o[0] if isinstance(o, (tuple, list)) else o
            cap[name] = t.detach().cpu().float().numpy()
        return f

    em = clf.mods.embedding_model
    handles = [
        em.blocks[0].register_forward_hook(hook("block0_out")),
        em.blocks[1].register_forward_hook(hook("block1_out")),
        em.blocks[2].register_forward_hook(hook("block2_out")),
        em.blocks[3].register_forward_hook(hook("block3_out")),
        em.mfa.register_forward_hook(hook("mfa_out")),
        em.asp.register_forward_hook(hook("pooled")),
    ]
    # Front-end consistency (PARITY-CRITICAL): the C++ engine's front end is the
    # Kaldi FBank above, NOT SpeechBrain's internal compute_features (which uses a
    # different STFT/framing and yields a different T, e.g. 301 vs 298, on a
    # totally different value scale). Capturing the per-block goldens from
    # encode_batch() would dump tensors the kaldi-fed C++ encoder can never
    # reproduce. So we run embedding_model directly on the SAME kaldi CMN features
    # the C++ feeds: embedding_model.forward takes [B, T, n_mels] and transposes
    # internally. This keeps block0..3 / mfa / pooled / embedding all consistent
    # with the engine's real front end.
    feats = fb_cmn.unsqueeze(0)  # [1, T, n_mels]
    with torch.no_grad():
        emb = em(feats).squeeze().detach().cpu().float().numpy()  # [192]
    for h in handles:
        h.remove()

    # L2-normalize to match what the C++ embedding stage emits (see module docstring).
    norm = float(np.linalg.norm(emb))
    if norm > 0:
        emb = emb / norm

    w.add_tensor("fbank", _sq(fbank))
    w.add_tensor("fbank_raw", _sq(fbank_raw))
    for k in ["block0_out", "block1_out", "block2_out", "block3_out", "mfa_out", "pooled"]:
        if k in cap:
            w.add_tensor(k, _sq(cap[k]))
    w.add_tensor("embedding", _sq(emb))


def gen_wespeaker(args, w):
    """ONNX-direct golden (WeSpeaker ResNet34 / 3D-Speaker ERes2Net / CAM++). Runs the
    ONNX session on the SAME Kaldi FBank+CMN features the C++ engine feeds
    (rank-3 [1, frames, n_mels], the OnnxDirectEngine convention) and dumps:

    * ``encoder_out`` the RAW ONNX embedding output (pre L2-norm; 256-d for
                      WeSpeaker ResNet34, 512-d for ERes2Net base).
    * ``embedding``   the unit-L2-normalized embedding the C++ gate compares
                      against (cosine + max|d| on the unit sphere).
    """
    try:
        import onnxruntime as ort
    except ImportError as e:
        print(f"VOICEDETECT_CONVERT_DEPS_MISSING: {e}", file=sys.stderr)
        sys.exit(2)
    onnx_path = args.onnx or args.model
    try:
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"VOICEDETECT_MODEL_UNAVAILABLE: {e}", file=sys.stderr)
        sys.exit(2)

    audio = _load_audio_16k(args.audio)
    fb_cmn, fbank_raw = _kaldi_fbank_cmn(audio)
    fbank = fb_cmn.numpy().T                                  # [80, T]

    feed = fb_cmn.numpy().astype(np.float32)[np.newaxis, :, :]  # [1, T, 80]
    in_name = sess.get_inputs()[0].name
    out = sess.run(None, {in_name: feed})
    embs = np.asarray(out[0]).reshape(-1).astype(np.float32)   # [256] raw

    emb = embs.copy()
    norm = float(np.linalg.norm(emb))
    if norm > 0:
        emb = emb / norm

    w.add_tensor("fbank", _sq(fbank))
    w.add_tensor("fbank_raw", _sq(fbank_raw))
    w.add_tensor("encoder_out", _sq(embs))
    w.add_tensor("embedding", _sq(emb))


def gen_wav2vec2_emotion(args, w):
    """Dump the wav2vec2 emotion analyze goldens (a SEPARATE analyze model, not a
    speaker encoder). Runs ``superb/wav2vec2-base-superb-er`` deterministically on
    the fixture clip and stores the intermediates the C++ analyze graph (T18-20)
    parity-gates against. The front end is the raw 16 kHz waveform fed through the
    HF ``Wav2Vec2FeatureExtractor`` (``do_normalize=False`` for this checkpoint, so
    the waveform passes through unchanged) - NOT the mel FBank the speaker encoders
    use.

    Stored tensors (squeezed; f32). Axis order:

    * ``feat_extract_out`` ``[conv_dim, T']`` the conv feature-extractor output
                                              (HF ``[B, C, T']`` squeezed), the input
                                              to the feature projection.
    * ``enc_layer_0``  ``[hidden, T']``  transformer hidden state AFTER encoder
                                         layer 0 (``hidden_states[1]``), transposed
                                         to hidden-major.
    * ``enc_layer_mid````[hidden, T']``  AFTER the middle encoder layer
                                         (``hidden_states[n_layers // 2]``).
    * ``enc_layer_last````[hidden, T']`` AFTER the final encoder layer
                                         (``hidden_states[n_layers]``).
    * ``emotion_logits````[num_emotions]`` the classifier logits (pre-softmax).
    * ``emotion_probs`` ``[num_emotions]`` softmax(logits), for reference.

    KV ``baseline.emotion_labels`` carries the id-ordered label list and
    ``baseline.dominant_emotion`` the argmax label.
    """
    try:
        from transformers import (AutoConfig, AutoFeatureExtractor,
                                  AutoModelForAudioClassification)
    except ImportError as e:
        print(f"VOICEDETECT_CONVERT_DEPS_MISSING: {e}", file=sys.stderr)
        sys.exit(2)
    try:
        cfg = AutoConfig.from_pretrained(args.model)
        model = AutoModelForAudioClassification.from_pretrained(args.model)
        fe = AutoFeatureExtractor.from_pretrained(args.model)
    except Exception as e:
        print(f"VOICEDETECT_MODEL_UNAVAILABLE: {e}", file=sys.stderr)
        sys.exit(2)
    model.eval()

    audio = _load_audio_16k(args.audio)  # raw 16 kHz mono float32
    inputs = fe(audio, sampling_rate=16000, return_tensors="pt")

    cap = {}
    h = model.wav2vec2.feature_extractor.register_forward_hook(
        lambda mod, i, o: cap.__setitem__("fe", o.detach().cpu().float().numpy()))
    with torch.no_grad():
        out = model(inputs["input_values"], output_hidden_states=True)
    h.remove()

    hs = out.hidden_states                 # 13 states: [proj_out, layer0..layer11]
    n_layers = int(cfg.num_hidden_layers)
    feat_extract_out = np.squeeze(cap["fe"])              # [C, T']
    # hidden_states[k] is [B, T', H]; store hidden-major [H, T'].

    def _enc(idx):
        return np.squeeze(hs[idx].detach().cpu().float().numpy()).T   # [H, T']

    logits = np.squeeze(out.logits.detach().cpu().float().numpy())    # [num_emotions]
    probs = np.squeeze(torch.softmax(out.logits, dim=-1).detach().cpu().float().numpy())

    id2label = {int(k): v for k, v in cfg.id2label.items()}
    labels = [id2label[i] for i in range(len(id2label))]

    w.add_tensor("feat_extract_out", _sq(feat_extract_out))
    w.add_tensor("enc_layer_0", _sq(_enc(1)))
    w.add_tensor("enc_layer_mid", _sq(_enc(n_layers // 2)))
    w.add_tensor("enc_layer_last", _sq(_enc(n_layers)))
    w.add_tensor("emotion_logits", _sq(logits))
    w.add_tensor("emotion_probs", _sq(probs))
    w.add_array("baseline.emotion_labels", labels)
    w.add_string("baseline.dominant_emotion", labels[int(np.argmax(logits))])


def gen_wav2vec2_age_gender(args, w):
    """Dump the audeering age/gender analyze goldens (a SEPARATE analyze model from
    the emotion one). Runs ``audeering/wav2vec2-large-robust-24-ft-age-gender``
    deterministically on the fixture clip. The checkpoint ships a CUSTOM multi-task
    head that ``AutoModelForAudioClassification`` mangles, so we instantiate the
    official ``AgeGenderModel`` (wav2vec2-large-robust backbone + two ``ModelHead``s:
    age regression 1024->1024->1, gender 1024->1024->3) and load the raw weights.

    Stored tensors (squeezed; f32). Axis order:

    * ``feat_extract_out`` ``[conv_dim, T']`` conv feature-extractor output.
    * ``enc_layer_last`` ``[hidden, T']`` the last hidden state (final encoder
                                          LayerNorm output), hidden-major.
    * ``age_raw`` ``[1]``   the raw age regression scalar (age years == *100).
    * ``gender_logits`` ``[num_genders]`` the pre-softmax gender logits.
    * ``gender_probs``  ``[num_genders]`` softmax(gender_logits), for reference.

    KV ``baseline.gender_labels`` carries the id-ordered label list,
    ``baseline.dominant_gender`` the argmax label, and ``baseline.age_years`` the age.
    """
    try:
        import torch.nn as nn
        from transformers import (AutoConfig, AutoFeatureExtractor,
                                  Wav2Vec2Model, Wav2Vec2PreTrainedModel)
    except ImportError as e:
        print(f"VOICEDETECT_CONVERT_DEPS_MISSING: {e}", file=sys.stderr)
        sys.exit(2)

    # The official audeering recipe (verbatim from the model card): a Wav2Vec2 model
    # with two regression/classification heads. AutoModel mangles this, so we build
    # it explicitly and load the published weights into it.
    class ModelHead(nn.Module):
        def __init__(self, config, num_labels):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, num_labels)

        def forward(self, x):
            x = self.dropout(x)
            x = self.dense(x)
            x = torch.tanh(x)
            x = self.dropout(x)
            return self.out_proj(x)

    class AgeGenderModel(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.wav2vec2 = Wav2Vec2Model(config)
            self.age = ModelHead(config, 1)
            self.gender = ModelHead(config, 3)
            # No init_weights(): we load the full published state_dict below, and
            # transformers 5.x weight-tying (all_tied_weights_keys) trips on this
            # custom subclass. Skipping it is safe since every weight is overwritten.

        def forward(self, input_values, output_hidden_states=False):
            outputs = self.wav2vec2(input_values, output_hidden_states=output_hidden_states)
            pooled = torch.mean(outputs[0], dim=1)
            return outputs, pooled, self.age(pooled), self.gender(pooled)

    try:
        cfg = AutoConfig.from_pretrained(args.model)
        fe = AutoFeatureExtractor.from_pretrained(args.model)
        # Build from config and load the raw published weights directly:
        # AgeGenderModel.from_pretrained trips transformers 5.x weight-tying on this
        # custom subclass, and AutoModel mangles the head, so neither auto path works.
        import os
        sd = None
        if os.path.isdir(args.model):
            cands = [os.path.join(args.model, "model.safetensors"),
                     os.path.join(args.model, "pytorch_model.bin")]
        elif os.path.isfile(args.model):
            cands = [args.model]
        else:
            from huggingface_hub import hf_hub_download
            cands = []
            for fn in ("model.safetensors", "pytorch_model.bin"):
                try:
                    cands.append(hf_hub_download(args.model, fn)); break
                except Exception:
                    continue
        for path in cands:
            if not path or not os.path.isfile(path):
                continue
            if path.endswith(".safetensors"):
                from safetensors.torch import load_file
                sd = load_file(path)
            else:
                sd = torch.load(path, map_location="cpu")
            break
        if sd is None:
            raise FileNotFoundError(f"no weights for {args.model}")
        model = AgeGenderModel(cfg)
        model.load_state_dict(sd, strict=True)
    except Exception as e:
        print(f"VOICEDETECT_MODEL_UNAVAILABLE: {e}", file=sys.stderr)
        sys.exit(2)
    model.eval()

    audio = _load_audio_16k(args.audio)  # raw 16 kHz mono float32
    inputs = fe(audio, sampling_rate=16000, return_tensors="pt")

    cap = {}
    h = model.wav2vec2.feature_extractor.register_forward_hook(
        lambda mod, i, o: cap.__setitem__("fe", o.detach().cpu().float().numpy()))
    with torch.no_grad():
        outputs, pooled, logits_age, logits_gender = model(
            inputs["input_values"], output_hidden_states=True)
    h.remove()

    n_layers = int(cfg.num_hidden_layers)
    hs = outputs.hidden_states                  # n_layers+1 states
    feat_extract_out = np.squeeze(cap["fe"])    # [C, T']
    enc_last = np.squeeze(hs[n_layers].detach().cpu().float().numpy()).T  # [H, T']

    age_raw = np.squeeze(logits_age.detach().cpu().float().numpy()).reshape(-1)   # [1]
    gender_logits = np.squeeze(logits_gender.detach().cpu().float().numpy())      # [3]
    gender_probs = np.squeeze(
        torch.softmax(logits_gender, dim=-1).detach().cpu().float().numpy())

    id2label = {int(k): v for k, v in cfg.id2label.items()}
    labels = [id2label[i] for i in range(len(id2label))]

    w.add_tensor("feat_extract_out", _sq(feat_extract_out))
    w.add_tensor("enc_layer_last", _sq(enc_last))
    w.add_tensor("age_raw", _sq(age_raw))
    w.add_tensor("gender_logits", _sq(gender_logits))
    w.add_tensor("gender_probs", _sq(gender_probs))
    w.add_array("baseline.gender_labels", labels)
    w.add_string("baseline.dominant_gender", labels[int(np.argmax(gender_logits))])
    w.add_float32("baseline.age_years", float(age_raw[0]) * 100.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="HF id, local checkpoint, or ONNX path "
                         "(defaults per --arch / --analyze)")
    ap.add_argument("--audio", required=True, help="16 kHz mono WAV")
    ap.add_argument("--output", required=True)
    ap.add_argument("--arch",
                    choices=["ecapa_tdnn", "wespeaker_resnet34", "eres2net", "campplus",
                             "wav2vec2_emotion", "wav2vec2_age_gender"],
                    default="ecapa_tdnn", help="reference architecture")
    ap.add_argument("--analyze", action="store_true",
                    help="dump the wav2vec2 emotion analyze goldens (shorthand for "
                         "--arch wav2vec2_emotion, default model "
                         "superb/wav2vec2-base-superb-er)")
    ap.add_argument("--age-gender", dest="age_gender", action="store_true",
                    help="dump the audeering age/gender analyze goldens (shorthand "
                         "for --arch wav2vec2_age_gender, default model "
                         "audeering/wav2vec2-large-robust-24-ft-age-gender)")
    ap.add_argument("--onnx", default=None,
                    help="ONNX model path (ONNX-direct archs; defaults to --model)")
    args = ap.parse_args()

    if args.age_gender:
        args.arch = "wav2vec2_age_gender"
    elif args.analyze:
        args.arch = "wav2vec2_emotion"
    if args.model is None:
        args.model = {
            "wav2vec2_age_gender": "audeering/wav2vec2-large-robust-24-ft-age-gender",
            "wav2vec2_emotion": "superb/wav2vec2-base-superb-er",
        }.get(args.arch, "speechbrain/spkrec-ecapa-voxceleb")

    w = gguf.GGUFWriter(args.output, "voicedetect-baseline")
    w.add_string("baseline.audio", args.audio)
    # The ONNX-direct archs (WeSpeaker ResNet34, 3D-Speaker ERes2Net, CAM++) share one
    # generic golden dumper: it runs onnxruntime on the SAME Kaldi FBank+CMN
    # features the C++ engine feeds and stores encoder_out (raw embs) + embedding
    # (unit-L2-normalized). ERes2Net's "global-mean" feature normalization is
    # exactly the per-utterance CMN this path already applies.
    if args.arch == "wav2vec2_age_gender":
        gen_wav2vec2_age_gender(args, w)
    elif args.arch == "wav2vec2_emotion":
        gen_wav2vec2_emotion(args, w)
    elif args.arch in ("wespeaker_resnet34", "eres2net", "campplus"):
        gen_wespeaker(args, w)
    else:
        gen_ecapa(args, w)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
