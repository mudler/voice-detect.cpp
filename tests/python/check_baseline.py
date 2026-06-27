import os, subprocess, sys, tempfile
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import gguf
except ImportError:
    print("check_baseline: gguf missing; skip", file=sys.stderr); sys.exit(77)

ANALYZE = "--analyze" in sys.argv[1:]
AGE_GENDER = "--age-gender" in sys.argv[1:]
out = os.path.join(tempfile.gettempdir(),
                   "vd_baseline_age_gender.gguf" if AGE_GENDER else
                   "vd_baseline_analyze.gguf" if ANALYZE else "vd_baseline.gguf")
audio = os.path.join(root, "tests", "fixtures", "clip_a.wav")
if not os.path.exists(audio):
    print("check_baseline: fixture absent; skip", file=sys.stderr); sys.exit(77)

if AGE_GENDER:
    # Age/gender (audeering wav2vec2-large-robust) goldens: a SEPARATE analyze model
    # from the emotion one; gen_baseline.py defaults the model via --age-gender.
    argv = ["--age-gender", "--audio", audio, "--output", out]
    need = ["feat_extract_out", "enc_layer_last", "age_raw",
            "gender_logits", "gender_probs"]
elif ANALYZE:
    # Analyze (wav2vec2 emotion) goldens: a SEPARATE model from the speaker
    # encoder; gen_baseline.py defaults the model via --analyze.
    argv = ["--analyze", "--audio", audio, "--output", out]
    need = ["feat_extract_out", "enc_layer_0", "enc_layer_mid", "enc_layer_last",
            "emotion_logits", "emotion_probs"]
else:
    argv = ["--model", "speechbrain/spkrec-ecapa-voxceleb", "--audio", audio, "--output", out]
    need = ["fbank", "block0_out", "mfa_out", "pooled", "embedding"]

r = subprocess.run([sys.executable, os.path.join(root, "scripts", "gen_baseline.py"), *argv],
                   capture_output=True, text=True)
if r.returncode == 2 or "VOICEDETECT_MODEL_UNAVAILABLE" in r.stderr or "VOICEDETECT_CONVERT_DEPS_MISSING" in r.stderr:
    print("check_baseline: deps/model unavailable; skip", file=sys.stderr); sys.exit(77)
assert r.returncode == 0, r.stderr
rd = gguf.GGUFReader(out)
names = {t.name for t in rd.tensors}
for n in need:
    assert n in names, f"baseline missing {n}"
print("check_baseline OK:", sorted(names)); sys.exit(0)
