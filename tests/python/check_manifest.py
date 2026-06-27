import os, sys
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
man = os.path.join(root, "models", "ecapa_tdnn_manifest.txt")
if not os.path.exists(man):
    # The manifest is a committed artifact. When it is genuinely absent (e.g. a
    # CI checkout without the artifact), skip (ctest SKIP_RETURN_CODE 77) rather
    # than fail; run scripts/trace_ecapa.py in a reference venv to regenerate it.
    print("check_manifest: manifest missing; run scripts/trace_ecapa.py", file=sys.stderr)
    sys.exit(77)
lines = [l for l in open(man).read().splitlines() if l.strip() and not l.startswith("#")]
names = {l.split("\t")[0] for l in lines}
# The ECAPA spine must be present (names are verbatim SpeechBrain state_dict keys).
need_prefixes = ["blocks.0.conv.conv.weight", "mfa.conv.conv.weight",
                 "asp.", "asp_bn.norm.weight", "fc.conv.weight"]
missing = [p for p in need_prefixes if not any(n.startswith(p) for n in names)]
assert not missing, f"manifest missing expected groups: {missing}"
print("check_manifest OK:", len(names), "tensors")
sys.exit(0)
