"""Fit a two-class TorchVision SSDLite manhole locator from boxed manifests."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import ImageEnhance, ImageOps
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor

from inspector.detector import ARCHITECTURE, LOCATOR_FORMAT, Box, ManholeLocator, box_iou, build_locator_model
from inspector.manifest import Record, data_fingerprint, load_manifest
from inspector.quality import open_image
from scripts.benchmark_transfer import _check_disjoint, _named_paths


def _frame_boxes(records: list[Record]) -> list[tuple[Path, list[Box]]]:
    frames: dict[Path, list[Box]] = {}
    for record in records:
        if record.bbox is None:
            raise ValueError(f"В {record.image_id} отсутствует рамка колодца.")
        frames.setdefault(record.image_path, []).append(record.bbox)
    return list(frames.items())


def _checked_boxes(path: Path, boxes: list[Box], width: int, height: int) -> list[Box]:
    for x1, y1, x2, y2 in boxes:
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"Рамка вне кадра {path}: {(x1, y1, x2, y2)} для размера {width}×{height}.")
    return boxes


class BoxedFrames:
    def __init__(self, records: list[Record]):
        self.frames = _frame_boxes(records)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int):
        path, boxes = self.frames[index]
        image = open_image(path)
        boxes = _checked_boxes(path, boxes, image.width, image.height)
        if random.random() < 0.5:
            image = ImageOps.mirror(image)
            boxes = [(image.width - x2, y1, image.width - x1, y2) for x1, y1, x2, y2 in boxes]
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
        return to_tensor(image), {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.ones(len(boxes), dtype=torch.int64),
        }


def _collate(batch):
    return tuple(zip(*batch, strict=True))


def _validation_split(records: list[Record], seed: int) -> tuple[list[Record], list[Record]]:
    # Group rows from the same source asset or frame if it has several boxes.
    # Different frames without a supplied physical ID use their own path.
    groups = [r.asset_id if r.asset_id_origin == "provided" else str(r.image_path) for r in records]
    if len(set(groups)) < 8:
        raise ValueError("Для обучения детектора нужны как минимум восемь независимых групп.")
    train, validation = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(records, groups=groups))
    return [records[i] for i in train], [records[i] for i in validation]


def _predict_rows(model, records: list[Record], batch_size: int, device: str) -> list[dict]:
    locator = ManholeLocator(model, score_threshold=0.01, device=device)
    frames = _frame_boxes(records)
    rows = []
    for start in range(0, len(frames), batch_size):
        batch = frames[start:start + batch_size]
        images = [open_image(path) for path, _ in batch]
        detections_by_image = locator.locate_batch(images)
        for img, (path, boxes), detections in zip(images, batch, detections_by_image, strict=True):
            boxes = _checked_boxes(path, boxes, img.width, img.height)
            box, score = detections[0] if detections else (None, None)
            rows.append({
                "image": path.name, "score": score, "predicted_box": box, "target_boxes": boxes,
                "top1_iou": max((box_iou(box, actual) for actual in boxes), default=0.0) if box is not None else 0.0,
                "full_frame_iou": max(box_iou((0, 0, img.width, img.height), actual) for actual in boxes),
                "detections": [{"box": b, "score": s, "iou": max(box_iou(b, actual) for actual in boxes)} for b, s in detections],
            })
        for image in images:
            image.close()
    return rows


def _metrics(rows: list[dict], threshold: float) -> dict:
    selected = [r for r in rows if r["score"] is not None and r["score"] >= threshold]
    n = len(rows)
    hits = sum(r["top1_iou"] >= 0.5 for r in selected)
    any_hits = sum(any(d["iou"] >= 0.5 and d["score"] >= threshold for d in r["detections"]) for r in rows)
    return {
        "n_frames": n, "n_returned": len(selected), "n_top1_iou50": hits, "n_any_iou50": any_hits,
        "coverage": len(selected) / n,
        "top1_iou50_recall": hits / n,
        "top1_iou50_precision": hits / len(selected) if selected else None,
        "any_iou50_recall": any_hits / n,
        "mean_top1_iou_returned": sum(r["top1_iou"] for r in selected) / len(selected) if selected else None,
        "full_frame_iou50_recall": sum(r["full_frame_iou"] >= 0.5 for r in rows) / n,
    }


def _threshold(rows: list[dict]) -> float:
    thresholds = [0.01, 0.025, 0.05, 0.1, *[i / 20 for i in range(3, 20)]]
    scores = [(threshold, _metrics(rows, threshold)) for threshold in thresholds]
    best_recall_count = max(m["n_top1_iou50"] for _, m in scores)
    suitable = [(value, m) for value, m in scores if (m["top1_iou50_precision"] or 0.0) >= 0.90
                and m["n_top1_iou50"] >= 0.9 * best_recall_count]
    if suitable:
        return max(suitable, key=lambda item: (item[1]["n_top1_iou50"], item[0]))[0]
    # F1 based on valid returned regions. A missed region and an imprecise
    # crop both count against a candidate operating threshold.
    return max(scores, key=lambda item: (
        2 * item[1]["n_top1_iou50"] / (item[1]["n_frames"] + item[1]["n_returned"]), item[0],
    ))[0]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Обучение SSDLite по размеченным рамкам колодцев.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--holdout", action="append", default=[], metavar="NAME=DATA")
    parser.add_argument("--out", type=Path, required=True, help="Новый каталог для модели и отчёта")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-epochs", type=int, default=0, help="Дополнительные эпохи с последним блоком MobileNet")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.fine_tune_epochs < 0 or args.batch_size < 1:
        parser.error("Необходимо хотя бы по одной эпохе и изображению в батче.")
    if args.out.exists():
        parser.error("Каталог для модели уже существует. Выберите новый путь.")

    training = load_manifest(args.train)
    holdout_paths = _named_paths(args.holdout)
    holdouts = {name: load_manifest(path) for name, path in holdout_paths.items()}
    _check_disjoint(training, holdouts)
    train, validation = _validation_split(training, args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = DataLoader(BoxedFrames(train), batch_size=args.batch_size, shuffle=True, collate_fn=_collate)
    model = build_locator_model(pretrained=True).to(args.device)
    optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.0015, weight_decay=0.0001)
    best_state = None
    best_validation = None
    best_score = (-1, -1.0)
    best_epoch = 0
    start = time.monotonic()
    for epoch in range(1, args.epochs + args.fine_tune_epochs + 1):
        if epoch == args.epochs + 1:
            model.backbone.features[-1].requires_grad_(True)
            model.backbone.extra.requires_grad_(True)
            optim = torch.optim.AdamW([
                {"params": model.head.parameters(), "lr": 0.0005},
                {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": 0.0002},
            ], weight_decay=0.0001)
        model.train()
        # Keep pretrained batch-normalization statistics frozen, including in
        # the 1x1 feature maps where the final batch can contain one frame.
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()
        losses = []
        for images, targets in data:
            images = [image.to(args.device) for image in images]
            targets = [{key: tensor.to(args.device) for key, tensor in target.items()} for target in targets]
            optim.zero_grad(set_to_none=True)
            loss = sum(model(images, targets).values())
            value = float(loss.detach())
            if not math.isfinite(value):
                raise ValueError(f"Некорректный loss на эпохе {epoch}.")
            loss.backward()
            optim.step()
            losses.append(value)
        validation_rows = _predict_rows(model, validation, args.batch_size, args.device)
        metrics = _metrics(validation_rows, 0.01)
        score = (metrics["n_top1_iou50"], sum(r["top1_iou"] for r in validation_rows))
        print(f"Эпоха {epoch}. Loss {np.mean(losses):.3f}. Рамки IoU≥0.5 на валидации "
              f"{score[0]}/{len(validation_rows)}. {time.monotonic() - start:.1f} с", flush=True)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_validation = validation_rows
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})

    assert best_state is not None and best_validation is not None
    threshold = _threshold(best_validation)
    model.load_state_dict(best_state)
    model.to("cpu")
    report = {
        "architecture": ARCHITECTURE, "initial_weights": "TorchVision SSDLite320 COCO_V1",
        "training_fingerprint": data_fingerprint(training), "training_n_frames": len(data.dataset),
        "best_epoch": best_epoch, "frozen_epochs": args.epochs, "fine_tune_epochs": args.fine_tune_epochs,
        "score_threshold": threshold, "validation": _metrics(best_validation, threshold),
        "threshold_rule": "Train validation. Prefer precision >= 0.90 if it keeps >= 90% of all correct crops; otherwise maximize crop F1.",
        "validation_ids": [r.image_id for r in validation], "holdouts": {},
    }
    for name, records in holdouts.items():
        rows = _predict_rows(model, records, args.batch_size, "cpu")
        report["holdouts"][name] = {"fingerprint": data_fingerprint(records), "metrics": _metrics(rows, threshold), "rows": rows}
    args.out.mkdir(parents=True)
    torch.save({"format": LOCATOR_FORMAT, "architecture": ARCHITECTURE, "score_threshold": threshold,
                "state_dict": best_state}, args.out / "locator.pt")
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Валидация обучения. Рамки IoU≥0.5 {report['validation']['n_top1_iou50']}/"
          f"{report['validation']['n_frames']}. Порог {threshold:.3f}.", flush=True)
    for name, result in report["holdouts"].items():
        m = result["metrics"]
        print(f"{name}. Рамки IoU≥0.5 {m['n_top1_iou50']}/{m['n_frames']}. "
              f"Детекции {m['n_returned']}/{m['n_frames']}. Порог {threshold:.3f}.", flush=True)


if __name__ == "__main__":
    main()
