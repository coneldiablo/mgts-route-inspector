"""Pair the optional grounded type head with the existing v0.2 damage heads."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

from inspector.cabinet import HEADS, cabinet_result, head_probabilities
from inspector.cabinet_roi import REVISION
from inspector.config import quality_config
from inspector.quality import assess_quality
from scripts.cabinet_compare import SEEDS, report_scores
from scripts.cabinet_final import VARIANTS, assign_groups, estimator, load_images, read_dss, splits, write_json


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, type=Path)
    parser.add_argument("--reviewed-groups", type=Path, required=True)
    parser.add_argument("--reference-features", type=Path, required=True)
    parser.add_argument("--roi-features", type=Path, required=True)
    parser.add_argument("--roi-regions", type=Path, required=True)
    parser.add_argument("--control", default=Path("runs/final-cabinet-experiment/oof_comparison.json"), type=Path)
    parser.add_argument("--control-model", default=Path("artifacts/final-cabinet-v0.2/model.joblib"), type=Path)
    parser.add_argument("--out", default=Path("runs/final-cabinet-v0.3-roi"), type=Path)
    parser.add_argument("--artifact", default=Path("artifacts/final-cabinet-v0.3-roi"), type=Path)
    args = parser.parse_args()

    frames = read_dss(args.source)
    images, hashes, metadata, _ = load_images(frames)
    assign_groups(frames, hashes, metadata, read_json(args.reviewed_groups))
    labeled = [i for i, frame in enumerate(frames) if frame.truth is not None]
    y = np.asarray([frames[i].truth for i in labeled])
    groups = np.asarray([frames[i].group for i in labeled])
    strata = np.asarray([frames[i].stratum for i in labeled])
    all_ids = [f"DSS-{i + 1:03d}" for i in range(len(frames))]
    regions = read_json(args.roi_regions)
    if [region["id"] for region in regions] != all_ids:
        raise ValueError("Порядок снимков для областей объекта отличается.")
    use_roi = np.array([region["bbox"] is not None for region in regions])
    found = use_roi[labeled]
    with np.load(args.reference_features, allow_pickle=False) as cached, np.load(args.roi_features, allow_pickle=False) as roi:
        frame_features = {name: cached[name] for name in VARIANTS}
        frame_pair = np.concatenate((frame_features["dino"], frame_features["siglip"]), axis=1)
        joined_all = roi["frame_roi"]
        if joined_all.shape != (len(frames), 2 * frame_pair.shape[1]) or not np.allclose(
            frame_pair, joined_all[:, :frame_pair.shape[1]], atol=1e-6, rtol=1e-4,
        ):
            raise ValueError("Эмбеддинги полного кадра и ROI относятся к разным входам или версиям.")
    joined = joined_all[labeled]
    control = read_json(args.control)
    args.out.mkdir(parents=True, exist_ok=True)
    quality_config_for_run = quality_config("lowres")
    qualities = [assess_quality(image, bbox=None, cfg=quality_config_for_run) for image in images]
    all_rows = {}
    metrics = {}
    gains = []
    for seed in SEEDS:
        rows = control[str(seed)]
        refs = [all_ids[i] for i in labeled]
        if [row["image_id"] for row in rows] != refs or [row["group"] for row in rows] != groups.tolist():
            raise ValueError("Снимки или группы контроля не совпадают.")
        if [[row["truth"][head] for head in HEADS] for row in rows] != y.tolist():
            raise ValueError("Разметка не совпадает с контролем.")
        raw_type = np.empty(len(y))
        base = np.asarray([[row["scores"]["control"][head] for head in HEADS] for row in rows])
        for fold, (train, test) in enumerate(splits(strata, groups, n=5, seed=seed)):
            if any(rows[i]["fold"] != fold for i in test) or set(groups[train]) & set(groups[test]):
                raise ValueError("Нарушено парное внешнее разбиение.")
            head = estimator(joined.shape[1]).fit(joined[train], y[train, 0])
            raw_type[test] = head.predict_proba(joined[test])[:, 1]
        combined = base.copy()
        combined[found, 0] = raw_type[found]
        old_correct = int(((base[:, 0] >= .5) == y[:, 0]).sum())
        new_correct = int(((combined[:, 0] >= .5) == y[:, 0]).sum())
        gains.append(new_correct - old_correct)
        actions = Counter()
        result_rows = []
        for i, row in enumerate(rows):
            old = dict(zip(HEADS, base[i], strict=True))
            quality = qualities[labeled[i]]
            output = cabinet_result(old, quality, reported_type_score=float(raw_type[i]) if found[i] else None)
            if output["action"] != cabinet_result(old, quality)["action"]:
                raise RuntimeError("Уточнение типа изменило существующую политику действий.")
            source = "grounded_roi" if found[i] else "full_frame"
            actions[output["action"], int(y[i, 1:].any())] += 1
            result_rows.append({**{key: row[key] for key in ("image_id", "truth", "group", "fold")},
                                **output, "object_type_source": source})
        all_rows[seed] = result_rows
        metrics[seed] = {
            "control_correct": old_correct, "candidate_correct": new_correct,
            "control": report_scores(y, base), "candidate": report_scores(y, combined),
            "oof_actions": {action: {"damage": actions[action, 1], "normal": actions[action, 0]}
                            for action in ("repair", "inspect", "no_action")},
        }
        print(seed, f"type: {old_correct} → {new_correct} of {len(y)}", flush=True)

    eligible = gains[0] > 0 and sum(g > 0 for g in gains) >= 4 and all(g >= 0 for g in gains)
    write_json(args.out / "comparison.json", {
        "seeds": SEEDS, "labeled": len(y), "total": len(frames), "groups": len(set(groups)),
        "grounded_labeled": int(found.sum()), "grounded_total": int(use_roi.sum()),
        "fallback_type_source": "v0.2 full_frame", "selected": "type_roi" if eligible else "control",
        "metrics": metrics,
    })
    write_json(args.out / "oof_by_seed.json", all_rows)
    write_json(args.out / "oof_predictions.json", all_rows[42])
    if not eligible:
        print("Критерий улучшения типа не выполнен.")
        return

    artifact = joblib.load(args.control_model)
    head = estimator(joined.shape[1]).fit(joined, y[:, 0])
    artifact["type_roi"] = {"estimator": head, "grounding_revision": REVISION}
    args.artifact.mkdir(parents=True, exist_ok=True)
    temporary = args.artifact / "model.joblib.tmp"
    joblib.dump(artifact, temporary)
    os.replace(temporary, args.artifact / "model.joblib")
    scores = {name: head_probabilities(specs, frame_features) for name, specs in artifact["heads"].items()}
    type_scores = head.predict_proba(joined_all)[:, 1]
    write_json(args.out / "predictions.json", [
        {
            "image_id": name,
            **cabinet_result({head: scores[head][i] for head in HEADS}, qualities[i],
                             reported_type_score=float(type_scores[i]) if use_roi[i] else None),
            "object_type_source": "grounded_roi" if use_roi[i] else "full_frame",
        }
        for i, name in enumerate(all_ids)
    ])
    print(f"Модель с уточнением типа: {args.artifact / 'model.joblib'}", flush=True)


if __name__ == "__main__":
    main()
