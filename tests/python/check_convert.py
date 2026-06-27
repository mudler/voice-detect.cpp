#!/usr/bin/env python3
"""Acceptance check for the voice-model -> GGUF converter.

Default (speaker) mode runs ``scripts/convert_voicedetect_to_gguf.py`` on an
anchor ECAPA checkpoint, re-opens the produced GGUF with ``gguf.GGUFReader``, and
asserts the metadata-driven schema (arch KV + embedding_dim + FBank params +
verbatim tensor names).

``--analyze`` mode instead converts the wav2vec2 emotion checkpoint
(``superb/wav2vec2-base-superb-er``) via the converter's ``--analyze`` path and
asserts the analyze schema: ``voicedetect.analyze.present == True``, a non-empty
``voicedetect.analyze.emotion_labels`` list, the wav2vec2 dim KV, and at least one
verbatim ``wav2vec2.*`` weight. This is a SEPARATE model from the speaker encoder
(its own analyze GGUF), so it gets its own ctest gate and skips independently.

Exit codes (ctest convention): 0 = pass, 77 = skip (deps/model absent), 1 = fail.
"""
import os
import subprocess
import sys
import tempfile

# Skip cleanly if the GGUF reader itself is unavailable in this environment.
try:
    import gguf
except ImportError:
    print("check_convert: 'gguf' not installed; skipping", file=sys.stderr)
    sys.exit(77)

ANALYZE = "--analyze" in sys.argv[1:]
AGE_GENDER = "--age-gender" in sys.argv[1:]

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
conv = os.path.join(root, "scripts", "convert_voicedetect_to_gguf.py")


def run_converter(argv, out):
    r = subprocess.run([sys.executable, conv, *argv, "--output", out],
                       capture_output=True, text=True)
    print(r.stdout, end="")
    print(r.stderr, end="", file=sys.stderr)
    # The converter exits 2 (and prints a marker) when its deps are not installed,
    # or when the checkpoint cannot be obtained (no network, HF 403, etc.). CI
    # without the reference venv or without model access must skip, not fail.
    if (r.returncode == 2
            or "VOICEDETECT_CONVERT_DEPS_MISSING" in r.stderr
            or "VOICEDETECT_MODEL_UNAVAILABLE" in r.stderr):
        print("check_convert: converter dependencies or model unavailable; skipping",
              file=sys.stderr)
        sys.exit(77)
    if r.returncode != 0:
        print("check_convert: converter failed", file=sys.stderr)
        sys.exit(1)


def check_analyze():
    model = os.environ.get("VOICEDETECT_TEST_ANALYZE_MODEL",
                           "superb/wav2vec2-base-superb-er")
    out = os.path.join(tempfile.gettempdir(), "vd_check_analyze.gguf")
    run_converter(["--analyze", "--model", model], out)

    reader = gguf.GGUFReader(out)
    kv = {f.name: f for f in reader.fields.values()}
    names = {t.name for t in reader.tensors}

    assert "voicedetect.arch" in kv, "missing voicedetect.arch"
    present = kv.get("voicedetect.analyze.present")
    assert present is not None, "missing voicedetect.analyze.present"
    assert bool(present.parts[present.data[0]][0]), "voicedetect.analyze.present is not True"

    labels = kv.get("voicedetect.analyze.emotion_labels")
    assert labels is not None and len(labels.data) > 0, "emotion_labels missing/empty"

    for k in ["voicedetect.w2v2.hidden_size", "voicedetect.w2v2.n_layers",
              "voicedetect.w2v2.n_heads", "voicedetect.w2v2.ff_dim",
              "voicedetect.w2v2.conv_dims", "voicedetect.w2v2.conv_kernels",
              "voicedetect.w2v2.conv_strides"]:
        assert k in kv, f"missing {k}"

    # Analyze names are abbreviated (wav2vec2. -> w2v2.) so they fit ggml's 64-char
    # GGML_MAX_NAME and the C++ engine can load the GGUF (see shorten_w2v2_name in
    # the converter). The rest of each name stays verbatim.
    assert any(n.startswith("w2v2.") for n in names), "w2v2.* weight missing"
    assert "w2v2.encoder.layers.0.attention.q_proj.weight" in names, \
        "transformer weight missing"
    # Every stored tensor name must be loadable by ggml (< GGML_MAX_NAME == 64).
    too_long = [n for n in names if len(n) >= 64]
    assert not too_long, f"tensor names exceed ggml's 64-char limit: {too_long}"
    print("check_convert (analyze) OK:", len(names), "tensors")


def check_age_gender():
    model = os.environ.get("VOICEDETECT_TEST_AGEGENDER_MODEL",
                           "audeering/wav2vec2-large-robust-24-ft-age-gender")
    out = os.path.join(tempfile.gettempdir(), "vd_check_age_gender.gguf")
    run_converter(["--age-gender", "--model", model], out)

    reader = gguf.GGUFReader(out)
    kv = {f.name: f for f in reader.fields.values()}
    names = {t.name for t in reader.tensors}

    arch = kv.get("voicedetect.arch")
    assert arch is not None, "missing voicedetect.arch"
    present = kv.get("voicedetect.analyze.present")
    assert present is not None, "missing voicedetect.analyze.present"
    assert bool(present.parts[present.data[0]][0]), "voicedetect.analyze.present is not True"

    labels = kv.get("voicedetect.analyze.gender_labels")
    assert labels is not None and len(labels.data) > 0, "gender_labels missing/empty"

    # The large-robust deltas vs the base emotion model must be recorded in KV so the
    # C++ analyze graph selects the layer-norm conv encoder + stable (pre-norm) layers
    # + input normalization.
    for k in ["voicedetect.w2v2.hidden_size", "voicedetect.w2v2.n_layers",
              "voicedetect.w2v2.conv_bias", "voicedetect.w2v2.do_normalize",
              "voicedetect.w2v2.feat_extract_norm", "voicedetect.w2v2.do_stable_layer_norm"]:
        assert k in kv, f"missing {k}"

    # The custom audeering heads (age + gender ModelHeads) must be present verbatim.
    for n in ["age.dense.weight", "age.out_proj.weight",
              "gender.dense.weight", "gender.out_proj.weight"]:
        assert n in names, f"age/gender head weight missing: {n}"
    assert any(n.startswith("w2v2.") for n in names), "w2v2.* weight missing"
    too_long = [n for n in names if len(n) >= 64]
    assert not too_long, f"tensor names exceed ggml's 64-char limit: {too_long}"
    print("check_convert (age/gender) OK:", len(names), "tensors")


def check_speaker():
    # Default anchor: SpeechBrain ECAPA-TDNN (192-d) speaker embedding model.
    model = os.environ.get("VOICEDETECT_TEST_MODEL", "speechbrain/spkrec-ecapa-voxceleb")
    out = os.path.join(tempfile.gettempdir(), "vd_check.gguf")
    run_converter(["--model", model], out)

    reader = gguf.GGUFReader(out)
    kv = {f.name: f for f in reader.fields.values()}

    assert "general.architecture" in kv, "missing general.architecture"
    assert "voicedetect.arch" in kv, "missing voicedetect.arch"
    assert "voicedetect.embedding_dim" in kv, "missing voicedetect.embedding_dim"

    names = {t.name for t in reader.tensors}
    assert names, "no tensors exported"

    ecapa_keys = ["voicedetect.ecapa.channels", "voicedetect.ecapa.n_blocks",
                  "voicedetect.ecapa.se_channels", "voicedetect.ecapa.res2net_scale",
                  "voicedetect.ecapa.attention_channels"]
    for k in ecapa_keys:
        assert k in kv, f"missing {k}"
    assert any(n == "blocks.0.conv.conv.weight" for n in names), "verbatim ECAPA name missing"
    assert any(n == "fc.conv.weight" for n in names), "fc weight missing"

    print("check_convert OK:", len(names), "tensors")


if AGE_GENDER:
    check_age_gender()
elif ANALYZE:
    check_analyze()
else:
    check_speaker()
sys.exit(0)
