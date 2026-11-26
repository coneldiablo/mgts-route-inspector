"""Train an Ultralytics YOLO manhole locator on independent boxed frames."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from inspector.detector import box_iou
from inspector.manifest import Record, data_fingerprint, image_sha256, load_manifest
from inspector.quality import open_image
from scripts.benchmark_transfer import _check_disjoint, _named_paths
from scripts.train_locator import _checked_boxes, _frame_boxes, _metrics, _validation_split

MAX_DETECTIONS = 20


def _background_records(file: Path | None) -> list[Record]:
    if file is None:
        return []
    paths = [Path(line.strip()).resolve() for line in file.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not paths or any(not path.is_file() for path in paths):
        raise ValueError(f"Список {file} пуст или содержит отсутствующие изображения.")
    hashes = [image_sha256(path) for path in paths]
    if len(set(hashes)) != len(hashes):
        raise ValueError(f"В {file} обнаружены повторные изображения.")
    return [Record(f"background-{digest}", path, f"background-{digest}", label="no_target", asset_id_origin="image")
            for path, digest in zip(paths, hashes, strict=True)]


def _write_split(dataset: Path, name: str, records: list[Record], backgrounds: list[Record]) -> None:
    image_dir = dataset / "images" / name
    label_dir = dataset / "labels" / name
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    frames = _frame_boxes(records) + [(record.image_path, []) for record in backgrounds]
    for i, (path, boxes) in enumerate(frames):
        with open_image(path) as photo:
            width, height = photo.size
            _checked_boxes(path, boxes, width, height)
            photo.save(image_dir / f"{i:05d}.png")
        normalized = []
        for x1, y1, x2, y2 in boxes:
            normalized.append(((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                               (x2 - x1) / width, (y2 - y1) / height))
        (label_dir / f"{i:05d}.txt").write_text(
            "".join("0 " + " ".join(f"{value:.8f}" for value in row) + "\n" for row in normalized),
            encoding="ascii",
        )


def _predict_rows(model, frames, *, imgsz: int, batch: int, device: str) -> list[dict]:
    rows = []
    for start in range(0, len(frames), batch):
        part = frames[start:start + batch]
        images = [open_image(path) for path, _ in part]
        try:
            predictions = model.predict(images, imgsz=imgsz, conf=0.01, max_det=MAX_DETECTIONS,
                                        device=device, verbose=False)
            for image, (path, boxes), prediction in zip(images, part, predictions, strict=True):
                targets = _checked_boxes(path, boxes, image.width, image.height)
                detections = []
                for values, conf in zip(prediction.boxes.xyxy.cpu().tolist(),
                                        prediction.boxes.conf.cpu().tolist(), strict=True):
                    bbox = tuple(values)
                    detections.append({"box": bbox, "score": conf,
                                       "iou": max((box_iou(bbox, target) for target in targets), default=0.0)})
                detections.sort(key=lambda item: -item["score"])
                first = detections[0] if detections else None
                rows.append({
                    "image": path.name, "score": first["score"] if first else None,
                    "predicted_box": first["box"] if first else None, "target_boxes": targets,
                    "top1_iou": first["iou"] if first else 0.0, "detections": detections,
                    "full_frame_iou": max((box_iou((0, 0, image.width, image.height), target)
                                           for target in targets), default=0.0),
                })
        finally:
            for image in images:
                image.close()
    return rows


def _strict_stats(rows: list[dict], threshold: float) -> dict:
    accepted = correct = missing = ambiguous = 0
    for row in rows:
        detections = [d for d in row["detections"] if d["score"] >= threshold]
        if len(detections) == 1:
            accepted += 1
            correct += detections[0]["iou"] >= 0.5
        elif not detections:
            missing += 1
        else:
            ambiguous += 1
    return {"accepted": accepted, "iou50": correct, "missing": missing, "ambiguous": ambiguous,
            "f1": 2 * correct / (len(rows) + accepted)}


def _threshold(rows: list[dict], negatives: list[dict]) -> tuple[float, list[dict], str]:
    grid = [0.01, 0.025, *[index / 20 for index in range(1, 20)]]
    if not negatives:
        scores = [(threshold, _strict_stats(rows, threshold)) for threshold in grid]
        selected = max(scores, key=lambda item: (item[1]["f1"], item[1]["iou50"], item[0]))[0]
        return selected, [], "Maximize F1 of one-box crops at IoU≥0.5 on training validation only."
    curve = []
    for threshold in grid:
        pos = _metrics(rows, threshold)
        false_flags = sum(row["score"] is not None and row["score"] >= threshold for row in negatives)
        valid = pos["n_top1_iou50"]
        predicted = pos["n_returned"] + false_flags
        curve.append({"threshold": threshold, "localized": valid, "positive_frames": len(rows),
                      "wrong_positive_boxes": pos["n_returned"] - valid,
                      "false_detection_frames": false_flags, "negative_frames": len(negatives),
                      "negative_fpr": false_flags / len(negatives), "recall": valid / len(rows),
                      "precision": valid / predicted if predicted else None,
                      "f1": 2 * valid / (predicted + len(rows))})
    allowed = [point for point in curve if point["negative_fpr"] <= 0.10 and point["localized"]]
    if allowed:
        best = max(allowed, key=lambda point: (point["localized"], point["f1"], point["threshold"]))
        rule = "Maximize top-box recall at IoU≥0.5 with ≤10% false flags on internal backgrounds."
    else:
        best = max(curve, key=lambda point: (point["f1"], -point["false_detection_frames"], point["threshold"]))
        rule = "No usable cutoff met the internal 10% background budget; maximize top-box F1."
    return best["threshold"], curve, rule


def _write_report(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Обучение YOLO на размеченных рамках колодцев.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--holdout", action="append", default=[], metavar="NAME=DATA")
    parser.add_argument("--distractors", type=Path, help="Манифест кадров без целевого объекта")
    parser.add_argument("--train-backgrounds", type=Path, help="Список путей к фоновым кадрам для обучения")
    parser.add_argument("--val-backgrounds", type=Path, help="Независимый список путей к фонам для выбора порога")
    parser.add_argument("--weights", type=Path, required=True, help="Локальные исходные веса, например YOLO26n")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    if not args.weights.is_file() or args.out.exists():
        parser.error("Требуются локальный файл весов и новый путь отчёта.")
    if min(args.epochs, args.batch_size) < 1 or args.imgsz < 32 or args.imgsz % 32:
        parser.error("Укажите положительное число эпох/батч и размер стороны, кратный 32.")
    if (args.train_backgrounds is None) != (args.val_backgrounds is None):
        parser.error("Передайте оба списка фоновых кадров.")

    training = load_manifest(args.train)
    backgrounds_train = _background_records(args.train_backgrounds)
    backgrounds_val = _background_records(args.val_backgrounds)
    cohort_paths = _named_paths(args.holdout)
    cohorts = {name: load_manifest(path) for name, path in cohort_paths.items()}
    negatives = load_manifest(args.distractors) if args.distractors else []
    checks = dict(cohorts)
    if backgrounds_train:
        if {"train_backgrounds", "val_backgrounds"}.intersection(checks):
            parser.error("Имена train_backgrounds и val_backgrounds зарезервированы.")
        checks.update({"train_backgrounds": backgrounds_train, "val_backgrounds": backgrounds_val})
    if negatives:
        if "distractors" in checks or any(record.bbox is not None for record in negatives):
            parser.error("Для кадров без объекта используйте отдельное имя и манифест без рамок.")
        checks["distractors"] = negatives
    _check_disjoint(training, checks)
    train, validation = _validation_split(training, args.seed)
    out = args.out.resolve()
    dataset = out / "dataset"
    _write_split(dataset, "train", train, backgrounds_train)
    _write_split(dataset, "val", validation, backgrounds_val)
    dataset_yaml = dataset / "dataset.yaml"
    _write_report(dataset_yaml, {"path": str(dataset), "train": "images/train", "val": "images/val",
                                 "names": ["manhole"]})

    os.environ.update({"YOLO_CONFIG_DIR": str(out / "ultralytics-config"), "YOLO_OFFLINE": "true",
                       "YOLO_AUTOINSTALL": "false", "WANDB_MODE": "disabled"})
    from ultralytics import YOLO, __version__, settings

    integrations = ("sync", "clearml", "comet", "dvc", "mlflow", "neptune", "raytune", "tensorboard", "wandb", "hub")
    settings.update({key: False for key in integrations if key in settings})
    starting_weights = args.weights.resolve()
    detector = YOLO(str(starting_weights))
    detector.train(data=str(dataset_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch_size,
                   device=args.device, seed=args.seed, workers=0, amp=False, plots=False,
                   project=str(out), name="training", exist_ok=False, cache=False, optimizer="AdamW",
                   lr0=0.001, lrf=0.1, degrees=5.0, mosaic=0.5, close_mosaic=min(5, args.epochs),
                   patience=args.epochs)
    checkpoint = out / "best.pt"
    shutil.copyfile(detector.trainer.best, checkpoint)
    frozen = YOLO(str(checkpoint))
    inference = {"imgsz": args.imgsz, "batch": args.batch_size, "device": args.device}
    val_rows = _predict_rows(frozen, _frame_boxes(validation), **inference)
    val_negative_rows = _predict_rows(frozen, [(record.image_path, []) for record in backgrounds_val], **inference)
    threshold, threshold_curve, threshold_rule = _threshold(val_rows, val_negative_rows)
    report = {
        "architecture": starting_weights.stem, "ultralytics": __version__,
        "pretrained_sha256": image_sha256(starting_weights), "checkpoint_sha256": image_sha256(checkpoint),
        "training_fingerprint": data_fingerprint(training), "train_frames": len(_frame_boxes(train)),
        "train_backgrounds": len(backgrounds_train), "val_backgrounds": len(backgrounds_val),
        "train_background_fingerprint": data_fingerprint(backgrounds_train) if backgrounds_train else None,
        "val_background_fingerprint": data_fingerprint(backgrounds_val) if backgrounds_val else None,
        "validation_ids": [record.image_id for record in validation], "imgsz": args.imgsz,
        "max_det": MAX_DETECTIONS, "score_threshold": threshold,
        "threshold_rule": threshold_rule, "threshold_curve": threshold_curve,
        "validation": _metrics(val_rows, threshold), "validation_strict": _strict_stats(val_rows, threshold),
        "holdouts": {},
    }
    report_path = out / "report.json"
    _write_report(report_path, report)
    _write_report(out / "locator.json", {"format": "mgts-yolo-locator-v1", "weights": checkpoint.name,
                                         "image_size": args.imgsz, "score_threshold": threshold,
                                         "model": starting_weights.stem.replace("yolo", "YOLO", 1)})
    print(f"Артефакт зафиксирован: {checkpoint}. Порог {threshold:.3f}.", flush=True)

    for name, records in cohorts.items():
        rows = _predict_rows(frozen, _frame_boxes(records), **inference)
        metrics = _metrics(rows, threshold)
        report["holdouts"][name] = {"fingerprint": data_fingerprint(records), "metrics": metrics,
                                     "strict": _strict_stats(rows, threshold), "rows": rows}
        print(f"{name}. IoU≥0.5 в {metrics['n_top1_iou50']}/{metrics['n_frames']} кадрах.", flush=True)
    if negatives:
        rows = _predict_rows(frozen, [(record.image_path, []) for record in negatives], **inference)
        false_detections = sum(row["score"] is not None and row["score"] >= threshold for row in rows)
        report["distractors"] = {"fingerprint": data_fingerprint(negatives), "n_frames": len(rows),
                                  "false_detection_frames": false_detections, "rows": rows}
        print(f"Кадры без объекта. Ложные детекции {false_detections}/{len(rows)}.", flush=True)
    _write_report(report_path, report)


if __name__ == "__main__":
    main()
