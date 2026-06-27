#!/usr/bin/env python3
"""Trace a SpeechBrain ECAPA-TDNN checkpoint: print every state_dict tensor
name + torch shape (verbatim, used as the GGUF tensor names) and the
mean_var_norm normalizer buffers. Writes models/ecapa_tdnn_manifest.txt."""
import argparse, os, sys
try:
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
except ImportError as e:
    print(f"VOICEDETECT_CONVERT_DEPS_MISSING: {e}", file=sys.stderr); sys.exit(2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="speechbrain/spkrec-ecapa-voxceleb")
    ap.add_argument("--output", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "ecapa_tdnn_manifest.txt"))
    args = ap.parse_args()
    try:
        clf = EncoderClassifier.from_hparams(source=args.model, savedir="./pretrained_models")
    except Exception as e:
        print(f"VOICEDETECT_MODEL_UNAVAILABLE: {e}", file=sys.stderr); sys.exit(2)
    rows = []
    emb = clf.mods.embedding_model.state_dict()
    for k in sorted(emb):
        rows.append((k, tuple(emb[k].shape)))
    norm = clf.mods.mean_var_norm.state_dict() if hasattr(clf.mods, "mean_var_norm") else {}
    for k in sorted(norm):
        rows.append((f"mean_var_norm.{k}", tuple(norm[k].shape)))
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write("# verbatim SpeechBrain ECAPA state_dict tensor names + torch shapes\n")
        for k, s in rows:
            f.write(f"{k}\t{','.join(str(x) for x in s)}\n")
    for k, s in rows:
        print(f"{k}\t{s}")
    print(f"wrote {args.output} ({len(rows)} tensors)", file=sys.stderr)

if __name__ == "__main__":
    main()
