"""Run the predeclared, grouped comparison against the cabinet v0.2 control."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import f1_score

from inspector import cabinet
from inspector.cabinet import CabinetFeatures, HEADS, MODEL_FORMAT, cabinet_result, head_probabilities
from inspector.config import quality_config
from inspector.quality import assess_quality
from scripts.cabinet_final import (
    VARIANTS, assign_groups, estimator, load_images, read_dss, score_set, signal_metrics, splits, write_json,
)

SEEDS = (42, 17, 23, 71, 113)
RECIPES = ("control", "base-expanded", "family-balanced")
VARIANTS_BASE = (*VARIANTS, "base", "base_multiscale", "base_siglip")
FAMILIES = (
    ("classical", "mobile"),
    ("dino", "multiscale", "hybrid"),
    ("siglip", "siglip_multi", "siglip_dino"),
    ("base", "base_multiscale", "base_siglip"),
)
C_MULTIPLIERS = (8.0, 2.0, 24.0)


def features_for(images, hashes: np.ndarray, cache: Path | None, device: str,
                 variants: tuple[str, ...] = VARIANTS_BASE) -> dict[str, np.ndarray]:
    stamp = hashlib.sha256()
    stamp.update(Path(cabinet.__file__).read_bytes())
    stamp.update(json.dumps([device, variants, hashes.tolist()]).encode())
    fingerprint = stamp.hexdigest()
    if cache and cache.is_file():
        with np.load(cache, allow_pickle=False) as stored:
            if stored["fingerprint"].item() == fingerprint:
                return {key: stored[key] for key in variants}
    features = CabinetFeatures(device).extract(images, set(variants))
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(cache.name + ".tmp")
        with tmp.open("wb") as output:
            np.savez_compressed(output, fingerprint=fingerprint, **features)
        os.replace(tmp, cache)
    return features


def rank_learners(features, y, groups, partitions, multipliers=C_MULTIPLIERS):
    scores = {}
    for name, x in features.items():
        for multiplier in multipliers:
            truth, probabilities = [], []
            for train, val in partitions:
                if set(groups[train]) & set(groups[val]):
                    raise RuntimeError("Связанные снимки пересекли внутренние разбиения.")
                fitted = estimator(x.shape[1], multiplier).fit(x[train], y[train])
                truth.extend(y[val])
                probabilities.extend(fitted.predict_proba(x[val])[:, 1])
            scores[name, multiplier] = f1_score(truth, np.asarray(probabilities) >= .5,
                                               average="macro", zero_division=0)
    return scores


def select_learners(scores: dict, recipe: str) -> list[tuple[str, float]]:
    if recipe == "family-balanced":
        winners = [
            max(((name, c) for name in family for c in C_MULTIPLIERS), key=scores.__getitem__)
            for family in FAMILIES
        ]
        return sorted(winners, key=scores.__getitem__, reverse=True)[:3]
    variants = VARIANTS if recipe == "control" else VARIANTS_BASE
    return sorted(((name, 8.0) for name in variants), key=scores.__getitem__, reverse=True)[:3]


def fitted_heads(features, truth, train, ranked, recipe):
    return {
        head: [
            {"variant": name, "multiplier": c,
             "estimator": estimator(features[name].shape[1], c).fit(features[name][train], truth[train, i])}
            for name, c in select_learners(ranked[i], recipe)
        ]
        for i, head in enumerate(HEADS)
    }


def report_scores(truth: np.ndarray, scores: np.ndarray) -> dict:
    metrics = score_set(truth, scores)
    metrics["dual"] = signal_metrics(truth[:, 1:].all(axis=1), (scores[:, 1:] >= .5).all(axis=1))
    return metrics


def evaluate(features, y, groups, strata, seed, recipes=RECIPES, multipliers=C_MULTIPLIERS):
    results = {recipe: np.empty(y.shape) for recipe in recipes}
    fold_ids = np.full(len(y), -1, dtype=int)
    for fold, (train, test) in enumerate(splits(strata, groups, n=5, seed=seed)):
        if set(groups[train]) & set(groups[test]) or (fold_ids[test] >= 0).any():
            raise RuntimeError("Некорректные внешние разбиения для OOF-оценки.")
        fold_ids[test] = fold
        inner = [(train[a], train[b]) for a, b in splits(strata[train], groups[train], n=3, seed=seed + 1)]
        ranked = [rank_learners(features, y[:, i], groups, inner, multipliers) for i in range(y.shape[1])]
        subset = {name: array[test] for name, array in features.items()}
        for recipe in recipes:
            heads = fitted_heads(features, y, train, ranked, recipe)
            for i, head in enumerate(HEADS):
                results[recipe][test, i] = head_probabilities(heads[head], subset)
    if (fold_ids < 0).any():
        raise RuntimeError("Внешняя оценка потеряла один или несколько снимков.")
    return results, fold_ids


def promoted_recipe(summary):
    reference = summary["control"]
    eligible = []
    for recipe, candidate in summary.items():
        if recipe == "control":
            continue
        improvements = [candidate[seed]["defect_macro_f1"] - reference[seed]["defect_macro_f1"] for seed in SEEDS]
        candidate_fp = np.mean([candidate[seed]["any_damage"]["fp"] for seed in SEEDS])
        reference_fp = np.mean([reference[seed]["any_damage"]["fp"] for seed in SEEDS])
        primary_any_gain = candidate[42]["any_damage"]["f1"] - reference[42]["any_damage"]["f1"]
        qualifies = (np.mean(improvements) >= .01 and sum(gain > 0 for gain in improvements) >= 3
                     and improvements[0] >= -.005 and candidate_fp <= reference_fp + 1
                     and primary_any_gain >= -.02)
        if qualifies:
            eligible.append((float(np.mean(improvements)), improvements[0], -candidate_fp, recipe))
    return max(eligible)[-1] if eligible else "control"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, type=Path)
    parser.add_argument("--reviewed-groups", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--private-feature-cache", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--control-only", action="store_true", help="Reproduce v0.2 without DINOv2-Base")
    args = parser.parse_args()
    args.out = args.out or Path("runs/final-cabinet-control" if args.control_only else "runs/final-cabinet-experiment")
    args.artifact = args.artifact or Path(
        "artifacts/final-cabinet-control" if args.control_only else "artifacts/final-cabinet-v0.3",
    )
    recipes = ("control",) if args.control_only else RECIPES
    variants = VARIANTS if args.control_only else VARIANTS_BASE
    multipliers = (8.0,) if args.control_only else C_MULTIPLIERS
    args.out.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    frames = read_dss(args.source)
    images, hashes, metadata, _ = load_images(frames)
    assign_groups(frames, hashes, metadata, json.loads(args.reviewed_groups.read_text(encoding="utf-8")))
    labeled = np.array([i for i, frame in enumerate(frames) if frame.truth is not None])
    y = np.array([frames[i].truth for i in labeled])
    groups = np.array([frames[i].group for i in labeled])
    strata = np.array([frames[i].stratum for i in labeled])
    print(f"Готовы к извлечению признаков {len(frames)} кадров, {len(set(groups))} визуальных групп.", flush=True)
    all_features = features_for(images, metadata[:, 0], args.private_feature_cache, args.device, variants)
    features = {name: values[labeled] for name, values in all_features.items()}
    print(f"Замороженные признаки готовы за {time.monotonic() - started:.1f} с.", flush=True)

    metrics: dict = {recipe: {} for recipe in recipes}
    by_seed = {}
    for seed in SEEDS:
        scores, folds = evaluate(features, y, groups, strata, seed, recipes, multipliers)
        for recipe in recipes:
            metrics[recipe][seed] = report_scores(y, scores[recipe])
        by_seed[seed] = [
            {
                "image_id": f"DSS-{image + 1:03d}", "group": int(groups[i]), "fold": int(folds[i]),
                "truth": dict(zip(HEADS, map(int, y[i]), strict=True)),
                "scores": {
                    recipe: {head: float(scores[recipe][i, j]) for j, head in enumerate(HEADS)}
                    for recipe in recipes
                },
            }
            for i, image in enumerate(labeled)
        ]
        print(seed, {recipe: round(metrics[recipe][seed]["defect_macro_f1"], 4) for recipe in recipes}, flush=True)

    recipe = promoted_recipe(metrics)
    primary = metrics[recipe][42]
    write_json(args.out / "comparison.json", {
        "seeds": SEEDS, "groups": len(set(groups)), "labeled": len(labeled), "total": len(frames),
        "selected": recipe, "primary": primary, "metrics": metrics,
        "mean_defect_macro_f1": {
            name: float(np.mean([metrics[name][seed]["defect_macro_f1"] for seed in SEEDS])) for name in recipes
        },
    })
    write_json(args.out / "oof_comparison.json", by_seed)
    outer = splits(strata, groups, n=5, seed=42)
    ranked = [rank_learners(features, y[:, i], groups, outer, multipliers) for i in range(y.shape[1])]
    heads = fitted_heads(features, y, np.arange(len(y)), ranked, recipe)
    model = {"format": MODEL_FORMAT, "heads": heads, "recipe": recipe}
    args.artifact.mkdir(parents=True, exist_ok=True)
    tmp = args.artifact / "model.joblib.tmp"
    joblib.dump(model, tmp)
    os.replace(tmp, args.artifact / "model.joblib")
    scores = {head: head_probabilities(heads[head], all_features) for head in HEADS}
    quality = quality_config("lowres")
    write_json(args.out / "predictions.json", [
        {"image_id": f"DSS-{i + 1:03d}", **cabinet_result(
            {head: scores[head][i] for head in HEADS}, assess_quality(image, bbox=None, cfg=quality),
        )}
        for i, image in enumerate(images)
    ])
    selected_rows = []
    for row in by_seed[42]:
        values = row["scores"][recipe]
        predicted = cabinet_result(values, assess_quality(
            images[int(row["image_id"].split("-")[-1]) - 1], bbox=None, cfg=quality,
        ))
        selected_rows.append({**{key: row[key] for key in ("image_id", "truth", "group", "fold")}, **predicted})
    write_json(args.out / "oof_predictions.json", selected_rows)
    print(f"Выбран вариант {recipe}. Артефакт: {args.artifact / 'model.joblib'}.", flush=True)


if __name__ == "__main__":
    main()
