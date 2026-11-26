"""Paired cabinet recognition ablation with pixel-only Grounding DINO crops.

Run from the project root with both original --source directories, in delivery
order. A pinned public detector is downloaded unless --detector points to a local
snapshot. The output includes private contact sheets and image embeddings; only
the protocol, aggregate benchmark, and trained type-head.npz belong in a release.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from inspector.cabinet import SiglipEmbedder
from inspector.cabinet_roi import CabinetGrounder, MARGIN, MIN_AREA, MIN_SCORE, MODEL, PROMPT, REVISION
from inspector.features import Dinov2Embedder
from scripts.cabinet_final import assign_groups, load_images, read_dss, score_set, splits

SEEDS = (42, 17, 23, 71, 113)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def classifier(dimension):
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=8.0 / dimension, class_weight="balanced", solver="liblinear", random_state=42, max_iter=2000,
    ))


def save_type_head(path, x, target):
    head = classifier(x.shape[1]).fit(x, target)
    scaler, logreg = (step for _, step in head.steps)
    weights = logreg.coef_[0] / scaler.scale_
    bias = logreg.intercept_[0] - scaler.mean_ @ weights
    np.savez_compressed(path, weights=weights, bias=bias)


def fit_oof(x, y, strata, groups, seed):
    predictions = np.zeros(y.shape, dtype=float)
    for train, test in splits(strata, groups, 5, seed):
        if set(groups[train]) & set(groups[test]):
            raise AssertionError("Visual group leakage")
        for j in range(y.shape[1]):
            head = classifier(x.shape[1])
            head.fit(x[train], y[train, j])
            predictions[test, j] = head.predict_proba(x[test])[:, 1]
    metrics = score_set(y, predictions)
    dual = (y[:, 1:] == 1).all(axis=1)
    metrics["both_found"] = int((predictions[dual, 1:] >= 0.5).all(axis=1).sum())
    return predictions, metrics


def compare_with_v02(control_rows, candidate, y, expected_ids, expected_folds):
    by_id = {r["image_id"]: r for r in control_rows}
    ordered = [by_id[key] for key in expected_ids]
    if len(ordered) != len(control_rows) or not np.array_equal([r["fold"] for r in ordered], expected_folds):
        raise ValueError("The archived predictions must use identical held-out images and folds.")
    truth = [[r["truth"][name] for name in ("telecom", "mechanical", "coating")] for r in ordered]
    if not np.array_equal(truth, y):
        raise ValueError("Ground truth does not match the archived run.")
    control = np.array([[
        r["object_type_score"] if r["object_type"] == "telecom_cabinet" else 1-r["object_type_score"],
        r["scores"]["mechanical"], r["scores"]["coating"],
    ] for r in ordered])
    metrics = {}
    dual = (y[:, 1:] == 1).all(axis=1)
    for name, scores in (("v02", control), ("v02_and_frame_roi", (control + candidate) / 2)):
        metrics[name] = score_set(y, scores)
        metrics[name]["both_found"] = int((scores[dual, 1:] >= .50).all(axis=1).sum())
    return metrics


def save_contact_sheet(images, regions, directory):
    sheets = [Image.new("RGB", (4 * 180, 4 * 150), "white") for _ in range((len(images) + 15) // 16)]
    for i, (image, region) in enumerate(zip(images, regions, strict=True)):
        preview = ImageOps.contain(image, (178, 129)).copy()
        if region.bbox:
            x1, y1, x2, y2 = region.bbox
            draw = ImageDraw.Draw(preview)
            draw.rectangle((x1 * preview.width, y1 * preview.height, x2 * preview.width, y2 * preview.height),
                           outline="lime", width=2)
        x, y = (i % 4) * 180, (i % 16 // 4) * 150
        sheet = sheets[i // 16]
        sheet.paste(preview, (x, y))
        ImageDraw.Draw(sheet).text((x, y + 130), f"DSS-{i + 1:03}: " + (f"{region.score:.2f}" if region.score is not None else "full"), fill="black")
    for i, sheet in enumerate(sheets):
        sheet.save(directory / f"private-review-{i + 1}.jpg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=Path, required=True)
    parser.add_argument("--detector", type=Path)
    parser.add_argument("--control-oof", type=Path, help="Optional frozen v0.2, seed-42 held-out predictions")
    parser.add_argument("--out", type=Path, default=Path("runs/cabinet-roi-grounding"))
    parser.add_argument("--reviewed-groups", type=Path, default=Path("scripts/cabinet_groups.json"))
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    write(args.out / "protocol.json", dict(
        source=MODEL, revision=REVISION, prompt=PROMPT, minimum_score=MIN_SCORE,
        minimum_area=MIN_AREA, context_margin=MARGIN, input_side=[560, 900],
        seeds=SEEDS, folds=5, threshold=0.50, candidate="full plus padded ROI, frozen DINOv2-S and SigLIP2",
        control="full frame, same encoders and regularized logistic regression, shrinkage 8/d",
        fallback="whole image when no eligible grounding",
    ))
    frames = read_dss(args.source)
    images, hashes, metadata, _ = load_images(frames)
    assign_groups(frames, hashes, metadata, json.loads(args.reviewed_groups.read_text()))
    model = CabinetGrounder(args.detector)
    start = time.monotonic()
    regions = []
    for i, im in enumerate(images, 1):
        regions.append(model.locate(im))
        if i == 1 or i % 10 == 0:
            print(f"grounded {i}/{len(images)} in {time.monotonic()-start:.1f}s", flush=True)
    del model
    gc.collect()
    crops = [r.view(im) for r, im in zip(regions, images, strict=True)]
    save_contact_sheet(images, regions, args.out)
    write(args.out / "private-regions.json", [{"id":f"DSS-{i + 1:03}", **asdict(region)}
                                              for i, region in enumerate(regions)])
    features = {}
    for name, encoder_cls in [("dino", Dinov2Embedder), ("siglip", SiglipEmbedder)]:
        encoder = encoder_cls()
        features["full_" + name] = encoder.embed(images)
        features["roi_" + name] = encoder.embed(crops)
        del encoder
        gc.collect()
    features["frame"] = np.concatenate([features["full_dino"], features["full_siglip"]], axis=1)
    features["roi"] = np.concatenate([features["roi_dino"], features["roi_siglip"]], axis=1)
    features["frame_roi"] = np.concatenate([features["frame"], features["roi"]], axis=1)
    np.savez_compressed(args.out / "private-features.npz", **features)
    idx = [i for i, f in enumerate(frames) if f.truth is not None]
    y = np.array([frames[i].truth for i in idx])
    save_type_head(args.out / "type-head.npz", features["frame_roi"][idx], y[:, 0])
    strata = np.array([frames[i].stratum for i in idx])
    groups = np.array([frames[i].group for i in idx])
    report = {"images":len(images), "labeled":len(idx), "groups":len(set(groups)),
              "grounded":sum(r.bbox is not None for r in regions), "seeds":{}}
    logits = {}
    for seed in SEEDS:
        fold = {}
        for method in ("frame", "frame_roi"):
            scores, fold[method] = fit_oof(features[method][idx], y, strata, groups, seed)
            logits[f"{seed}_{method}"] = scores
        report["seeds"][seed] = fold
    if args.control_oof:
        outer = splits(strata, groups, 5, 42)
        fold_for = np.empty(len(idx), dtype=int)
        for fold, (_, test) in enumerate(outer):
            fold_for[test] = fold
        report["paired_v02_seed42"] = compare_with_v02(
            json.loads(args.control_oof.read_text()), logits["42_frame_roi"], y,
            [f"DSS-{i + 1:03}" for i in idx], fold_for,
        )
    np.savez_compressed(args.out / "oof.npz", **logits)
    write(args.out / "benchmark.json", report)
    print([(seed, [(name, round(scores["defect_macro_f1"],4)) for name, scores in methods.items()])
           for seed, methods in report["seeds"].items()], flush=True)


if __name__ == "__main__":
    main()
