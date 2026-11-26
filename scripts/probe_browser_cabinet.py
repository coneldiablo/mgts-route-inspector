"""Evaluate browser-compatible cabinet features without exporting source photographs."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.cabinet_final import HEADS, assign_groups, load_images, read_dss, score_set, splits
SIDE = 224
MEAN = np.asarray([.485, .456, .406], dtype=np.float32)
STD = np.asarray([.229, .224, .225], dtype=np.float32)

VIEWS = {
    "whole": (0., 0., 1., 1.),
    "center": (.15, .15, .85, .85),
    "core": (.27, .27, .73, .73),
    "left": (.05, .15, .55, .85),
    "right": (.45, .15, .95, .85),
    "top": (.15, .05, .85, .55),
    "bottom": (.15, .45, .85, .95),
}
LOCAL_VIEWS = ("left", "right", "top", "bottom")
SEEDS = (42, 117, 203, 389)
CHOICES = (
    ("base", "standard", 8),
    ("whole", "standard", 8),
    ("center", "standard", 8),
    ("core", "standard", 8),
    ("scales", "standard", 8),
    ("scale_average", "standard", 8),
    ("local", "standard", 8),
    ("local_average", "standard", 8),
    ("regions", "standard", 8),
    ("center_edges", "standard", 8),
    ("texture", "standard", 8),
    ("material", "standard", 8),
    ("base_texture", "standard", 8),
    ("base_material", "standard", 8),
    ("base", "standard", 2),
    ("base", "standard", 32),
    ("base", "unit", 3),
    ("base", "unit", 12),
)
BASELINE = CHOICES[0]


def sampled(image, box):
    height, width = image.shape[:2]
    left, top, right, bottom = box
    offsets = (np.arange(SIDE * 4) + .5) / (SIDE * 4)
    x = np.minimum(((left + offsets * (right - left)) * width).astype(int), width - 1)
    y = np.minimum(((top + offsets * (bottom - top)) * height).astype(int), height - 1)
    grid = image[y[:, None], x]
    rgb = grid.reshape(SIDE, 4, SIDE, 4, 3).mean(axis=(1, 3), dtype=np.float32) / 255.
    encoded = ((rgb - MEAN) / STD).transpose(2, 0, 1)[None].copy()
    return encoded, rgb


def unit(vector):
    return vector / max(1e-12, np.linalg.norm(vector))


def summarize(tokens):
    patches = tokens[1:].reshape(16, 16, -1)
    return np.concatenate([unit(tokens[0]), unit(patches.mean(axis=(0, 1)))]), patches


def material_features(rgb):
    r, g, b = rgb.transpose(2, 0, 1)
    hi = np.maximum(np.maximum(r, g), b)
    lo = np.minimum(np.minimum(r, g), b)
    sat = hi - lo
    gray = .299 * r + .587 * g + .114 * b
    dx = np.abs(gray[:, 1:] - gray[:, :-1])
    dy = np.abs(gray[1:, :] - gray[:-1, :])
    flat_grad = np.maximum(np.pad(dx, ((0, 0), (0, 1))), np.pad(dy, ((0, 1), (0, 0))))
    values = []
    for v in (0, 56, 112, 168):
        for h in (0, 56, 112, 168):
            patch = np.s_[v:v+56, h:h+56]
            vectors = (gray[patch], sat[patch], (r - b)[patch], flat_grad[patch])
            for local in vectors:
                values.extend((np.mean(local), np.std(local)))
            values.append(np.mean(((r > g * 1.08) & (g > b * 1.08) & (sat > .06))[patch]))
    return np.asarray(values, dtype=np.float32)


def extract_frame(session, image):
    vectors, tokens = {}, {}
    for name, view in VIEWS.items():
        encoded, rgb = sampled(image, view)
        vectors[name], tokens[name] = summarize(session.run(None, {"pixel_values": encoded})[0][0])
        if name == "center":
            material = material_features(rgb)

    w, c, k = (vectors[name] for name in ("whole", "center", "core"))
    global_vectors = (w, c, k)
    parts = [vectors[name] for name in LOCAL_VIEWS]
    base = np.concatenate([w, c])
    spatial, edges, texture = [], [], []
    for name in ("whole", "center"):
        patches = tokens[name]
        spatial.extend(unit(patches[y:y+8, x:x+8].mean(axis=(0, 1))) for y in (0, 8) for x in (0, 8))
        edges.extend(unit(patches[:, x:x+4].mean(axis=(0, 1))) for x in (0, 12))
        edges.extend(unit(patches[y:y+4, :].mean(axis=(0, 1))) for y in (0, 12))
        texture.append(unit(patches.std(axis=(0, 1))))
    texture = np.concatenate(texture)
    return {
        "base": base, "whole": w, "center": c, "core": k,
        "scales": np.concatenate(global_vectors),
        "scale_average": np.mean(global_vectors, axis=0),
        "local": np.concatenate(parts),
        "local_average": np.mean(parts, axis=0),
        "regions": np.concatenate(spatial),
        "center_edges": np.concatenate([c, *edges]),
        "texture": texture,
        "material": material,
        "base_texture": np.concatenate([base, texture]),
        "base_material": np.concatenate([base, material]),
    }


def model(spec, dimension):
    _, normalization, strength = spec
    if normalization == "standard":
        c = strength / dimension
    else:
        c = strength
    classifier = LogisticRegression(C=c, class_weight="balanced", max_iter=2000, solver="liblinear", random_state=42)
    return make_pipeline(StandardScaler(), classifier) if normalization == "standard" else classifier


def choice_name(spec):
    return f"{spec[0]}__{spec[1]}_{spec[2]}"


def predict(spec, matrices, y, train, test):
    x = matrices[spec[0]]
    if np.unique(y[train]).size < 2:
        return np.full(len(test), (y[train].sum() + 1) / (len(train) + 2), dtype=float)
    return model(spec, x.shape[1]).fit(x[train], y[train]).predict_proba(x[test])[:, 1]


def oof_for(spec, partitions, matrices, y):
    values = np.full(y.shape[0], np.nan)
    for train, test in partitions:
        values[test] = predict(spec, matrices, y, train, test)
    if not np.isfinite(values).all():
        raise ValueError("Incomplete out-of-fold predictions")
    return values


def rank(matrices, labels, partitions):
    return sorted(CHOICES, key=lambda spec: f1_score(
        labels, oof_for(spec, partitions, matrices, labels) >= .5,
        average="macro", zero_division=0,
    ), reverse=True)


def test(matrices, y, strata, groups):
    reports, rows = {}, {}
    for seed in SEEDS:
        outer = splits(strata, groups, 5, seed)
        scores = {choice_name(spec): np.column_stack([
            oof_for(spec, outer, matrices, y[:, j]) for j in range(len(HEADS))
        ]) for spec in CHOICES}
        scores["nested-top3"] = np.full(y.shape, np.nan)
        scores["nested-top1"] = np.full(y.shape, np.nan)
        for train, target in outer:
            if set(groups[train]) & set(groups[target]):
                raise ValueError("Split mixes linked photos")
            inner = splits(strata[train], groups[train], 3, 43)
            features = {code: x[train] for code, x in matrices.items()}
            for j, label in enumerate(y.T):
                chosen = rank(features, label[train], inner)[:3]
                versions = [predict(spec, matrices, label, train, target) for spec in chosen]
                scores["nested-top1"][target, j] = versions[0]
                scores["nested-top3"][target, j] = np.mean(versions, axis=0)
        reports[seed] = {name: score_set(y, values) for name, values in scores.items()}
        rows[seed] = scores
        print("seed", seed, sorted(
            [(name, metric["defect_macro_f1"], metric["object_macro_f1"]) for name, metric in reports[seed].items()],
            key=lambda row: row[1], reverse=True)[:9], flush=True)
    return reports, rows


def prepare(sources, onnx, cache):
    frames = read_dss(sources)
    _, hashes, metadata, _ = load_images(frames)
    reviewed = json.loads(Path("scripts/cabinet_groups.json").read_text())
    assign_groups(frames, hashes, metadata, reviewed)
    indices = [i for i, frame in enumerate(frames) if frame.truth is not None]
    y = np.asarray([frames[i].truth for i in indices])
    strata = np.asarray([frames[i].stratum for i in indices])
    groups = np.asarray([frames[i].group for i in indices])
    if cache and cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            matrices = dict(saved)
    else:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 3
        session = ort.InferenceSession(str(onnx), options, providers=["CPUExecutionProvider"])
        extracted = []
        for position, i in enumerate(indices, 1):
            with Image.open(frames[i].path) as raw:
                image = np.asarray(ImageOps.exif_transpose(raw).convert("RGB"))
            extracted.append(extract_frame(session, image))
            if position % 10 == 0:
                print("encoded", position, "of", len(indices), flush=True)
        matrices = {name: np.stack([frame[name] for frame in extracted]) for name in extracted[0]}
        if cache:
            np.savez_compressed(cache, **matrices)
    return matrices, y, strata, groups, indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--onnx", type=Path, default=Path("browser-ml/runtime/dino-int8.onnx"))
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    matrices, y, strata, groups, indices = prepare(args.source, args.onnx, args.cache)
    reports, _ = test(matrices, y, strata, groups)
    data = {
        "images": len(indices), "groups": len(set(groups)), "seeds": SEEDS, "reports": reports,
        "seconds": round(time.monotonic() - started, 2),
    }
    args.report.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
