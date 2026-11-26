"""Fit downloadable browser heads against the exact pinned INT8 ONNX encoder."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

from scripts.cabinet_final import HEADS, assign_groups, estimator, load_images, read_dss, score_set, splits
from scripts.probe_browser_cabinet import LOCAL_VIEWS, MEAN, SIDE, STD, VIEWS, sampled, summarize


def sample(image, inset):
    # Average a fixed 4×4 sample grid in each cell identically to the JS implementation.
    a = np.asarray(image)
    h, w = a.shape[:2]
    offsets = (np.arange(SIDE * 4) + .5) / (SIDE * 4)
    u = inset + offsets * (1 - 2 * inset)
    grid = a[np.minimum((u * h).astype(int), h - 1)[:, None], np.minimum((u * w).astype(int), w - 1)]
    rgb = grid.reshape(SIDE, 4, SIDE, 4, 3).mean(axis=(1, 3), dtype=np.float32)
    return ((rgb / 255. - MEAN) / STD).transpose(2, 0, 1)[None].copy()


def vector(session, image):
    parts = []
    for inset in [0, .15]:
        tokens = session.run(None, {"pixel_values": sample(image, inset)})[0][0]
        for segment in (tokens[0], tokens[1:].mean(axis=0)):
            parts.append(segment / max(1e-12, np.linalg.norm(segment)))
    return np.concatenate(parts)


def local_average_vector(session, image):
    pixels = np.asarray(image)
    parts = []
    for name in LOCAL_VIEWS:
        encoded, _ = sampled(pixels, VIEWS[name])
        tokens = session.run(None, {"pixel_values": encoded})[0][0]
        features, _ = summarize(tokens)
        parts.append(features)
    return np.mean(parts, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--recipe", choices=["baseline", "local-average"], default="baseline")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    frames = read_dss(args.source)
    _, hashes, metadata, _ = load_images(frames)
    reviewed = json.loads(Path("scripts/cabinet_groups.json").read_text())
    assign_groups(frames, hashes, metadata, reviewed)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(args.onnx), options, providers=["CPUExecutionProvider"])
    rows = [f for f in frames if f.truth is not None]
    extract = local_average_vector if args.recipe == "local-average" else vector
    xs = []
    for f in rows:
        with Image.open(f.path) as im:
            xs.append(extract(session, ImageOps.exif_transpose(im).convert("RGB")))
    x = np.stack(xs)
    y = np.asarray([f.truth for f in rows])
    g = np.asarray([f.group for f in rows])
    oof = np.empty_like(y,dtype=float)
    for train, test in splits(np.array([f.stratum for f in rows]),g,5,42):
        for j in range(3):
            oof[test,j] = estimator(x.shape[1]).fit(x[train],y[train,j]).predict_proba(x[test])[:,1]
    data = {"format": 1, "heads": {}, "metrics": score_set(y,oof), "labeled":len(y),"groups":len(set(g))}
    if args.recipe == "local-average":
        data.update(format=2, recipe="local-average", views=[VIEWS[name] for name in LOCAL_VIEWS])
    for j, head in enumerate(HEADS):
        scaler, clf = estimator(x.shape[1]).fit(x,y[:,j]).steps
        weights = clf[1].coef_[0] / scaler[1].scale_
        bias = clf[1].intercept_[0] - weights @ scaler[1].mean_
        data["heads"][head] = {"weights":weights.round(8).tolist(),"bias":float(bias)}
    out = args.out or Path("browser-ml/heads-v2.json" if args.recipe == "local-average" else "browser-ml/heads.json")
    out.write_text(json.dumps(data,separators=(",",":")) + "\n", encoding="utf-8")
    print(json.dumps(data["metrics"],indent=2))


if __name__ == "__main__":
    main()
