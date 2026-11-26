"""Train and validate the image-only DSS cabinet model from the supplied XLSX files."""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from inspector.cabinet import CabinetFeatures, CabinetInspector, HEADS, MODEL_FORMAT, cabinet_result
from inspector.config import quality_config
from inspector.manifest import image_sha256
from inspector.quality import assess_quality

VARIANTS = ("classical", "dino", "multiscale", "hybrid", "mobile", "siglip", "siglip_multi", "siglip_dino")
ENSEMBLE_SIZE = 3
EXTENSIONS = {".jpg", ".jpeg", ".png"}
DUAL = re.compile(r"\bс\s+повреждени\w*\s+лкп\b", re.I)
SIGNALS = {"telecom": "ТКШ", "mechanical": "Механическое повреждение", "coating": "Дефект ЛКП"}


@dataclass
class Frame:
    path: Path
    truth: tuple[int, int, int] | None
    stratum: str = ""
    group: int = -1


def read_dss(sources: list[Path]) -> list[Frame]:
    from openpyxl import load_workbook

    frames = []
    for root in sources:
        sheets = sorted(root.glob("*/*.xlsx"))
        if not sheets:
            raise ValueError(f"Таблицы XLSX не найдены в {root}")
        indexed: set[Path] = set()
        for path in sheets:
            folder = path.parent
            name = folder.name.casefold()
            telecom = int("ткш" in name)
            if not telecom and "рщ" not in name:
                raise ValueError(f"Неизвестный тип объекта в {folder.name}")
            mechanical = int(name.startswith("мп"))
            coating = int(name.startswith("неудовлетворительное") and "лкп" in name)
            if not (mechanical or coating or name.startswith("удовлетворительное")):
                raise ValueError(f"Неизвестное состояние в {folder.name}")
            state = "mechanical" if mechanical else "coating" if coating else "normal"
            photos = {image.stem.casefold(): image for image in sorted(folder.iterdir())
                      if image.is_file() and image.suffix.casefold() in EXTENSIONS}
            book = load_workbook(path, read_only=True, data_only=True)
            try:
                rows = book.active.iter_rows(values_only=True)
                header = next(rows)
                names = [str(item).strip().casefold() for item in header]
                file_index, status_index = names.index("имя файла"), names.index("статус")
                for row in rows:
                    key = str(row[file_index]).strip().casefold()
                    if key == "none":
                        continue
                    key = Path(key).stem if Path(key).suffix.casefold() in EXTENSIONS else key.removesuffix(".0")
                    image = photos.get(key)
                    if image is None or image in indexed:
                        raise ValueError(f"Файл отсутствует или встречается повторно в {path.name}: {key}")
                    indexed.add(image)
                    status = str(row[status_index]).strip()
                    labels = (telecom, mechanical, int(coating or (mechanical and bool(DUAL.search(status)))))
                    frames.append(Frame(image, labels, f"{telecom}-{state}"))
            finally:
                book.close()
        frames.extend(Frame(photo, None) for photo in sorted(root.rglob("*"))
                      if photo.is_file() and photo.suffix.casefold() in EXTENSIONS and photo not in indexed)
    return frames


def load_images(frames: list[Frame]) -> tuple[list[Image.Image], list[int], np.ndarray, int]:
    images, hashes, digests, moments = [], [], [], []
    gps_count = 0
    for frame in frames:
        digests.append(image_sha256(frame.path))
        with Image.open(frame.path) as raw:
            exif = raw.getexif()
            gps_count += int(34853 in exif and bool(exif.get_ifd(34853)))
            datetime = (exif.get_ifd(34665).get(36867) if 34665 in exif else None) or exif.get(306)
            moments.append(datetime[:16] if isinstance(datetime, str) and datetime.startswith(("19", "20")) else None)
            image = ImageOps.exif_transpose(raw).convert("RGB")
            image.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
        pixels = np.asarray(image.convert("L").resize((9, 8)), dtype=int)
        signature = int.from_bytes(np.packbits(pixels[:, :-1] > pixels[:, 1:]).tobytes(), "big")
        hashes.append(signature)
        images.append(image)
    return images, hashes, np.asarray(list(zip(digests, moments, strict=True)), dtype=object), gps_count


def assign_groups(frames: list[Frame], hashes: list[int], metadata: np.ndarray,
                  reviewed_groups: list[list[str]] | None = None) -> None:
    parent = list(range(len(frames)))

    def representative(i: int) -> int:
        while parent[i] != i:
            i = parent[i]
        return i

    def join(i: int, j: int) -> None:
        a, b = representative(i), representative(j)
        parent[max(a, b)] = min(a, b)

    reviewed = [set(names) for names in reviewed_groups or []]
    for i, first in enumerate(frames):
        a = first.path.stem
        for j in range(i):
            b = frames[j].path.stem
            sequenced = (
                len(a) == len(b) == 8 and a.isdecimal() and b.isdecimal()
                and not a.startswith(("19", "20")) and not b.startswith(("19", "20"))
                and abs(int(a) - int(b)) <= 4
            )
            same_time = bool(metadata[i, 1] and metadata[i, 1] == metadata[j, 1])
            nearby = a == b or (hashes[i] ^ hashes[j]).bit_count() <= 10
            manual = any({a, b} <= group or {f"DSS-{i + 1:03d}", f"DSS-{j + 1:03d}"} <= group
                         for group in reviewed)
            if sequenced or same_time or nearby or manual or metadata[i, 0] == metadata[j, 0]:
                join(i, j)
    for i, frame in enumerate(frames):
        frame.group = representative(i)


def splits(strata: np.ndarray, group_ids: np.ndarray, n: int, seed: int):
    splitter = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(strata)), strata, group_ids))


def estimator(dimension: int):
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=8.0 / dimension, class_weight="balanced", max_iter=2000, solver="liblinear",
        random_state=42,
    ))


def predict_head(x: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray) -> np.ndarray:
    return estimator(x.shape[1]).fit(x[train], y[train]).predict_proba(x[test])[:, 1]


def variant_score(x: np.ndarray, y: np.ndarray, partitions) -> float:
    truth, scores = [], []
    for train, test in partitions:
        truth.extend(y[test])
        scores.extend(predict_head(x, y, train, test))
    return float(f1_score(truth, np.asarray(scores) >= .5, average="macro", zero_division=0))


def choose(variants: dict[str, np.ndarray], y: np.ndarray, partitions) -> list[str]:
    return sorted(VARIANTS, key=lambda name: variant_score(variants[name], y, partitions), reverse=True)[:ENSEMBLE_SIZE]


def signal_metrics(truth: np.ndarray, estim: np.ndarray) -> dict:
    tp = int(np.sum(truth & estim))
    fp = int(np.sum(~truth & estim))
    fn = int(np.sum(truth & ~estim))
    tn = int(np.sum(~truth & ~estim))
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, estim, average="binary", zero_division=0,
    )
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=float(precision), recall=float(recall), f1=float(f1),
                accuracy=float(accuracy_score(truth, estim)))


def score_set(y: np.ndarray, scores: np.ndarray) -> dict:
    predicted = scores >= .50
    truth = y.astype(bool)
    by_head = {name: signal_metrics(truth[:, j], predicted[:, j]) for j, name in enumerate(HEADS)}
    by_head["any_damage"] = signal_metrics(truth[:, 1:].any(axis=1), predicted[:, 1:].any(axis=1))
    by_head["object_macro_f1"] = float(f1_score(truth[:, 0], predicted[:, 0], average="macro"))
    by_head["defect_macro_f1"] = float(np.mean([by_head[head]["f1"] for head in HEADS[1:]]))
    by_head["defect_exact_match"] = float((truth[:, 1:] == predicted[:, 1:]).all(axis=1).mean())
    return by_head


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_training(args) -> None:
    started = time.monotonic()
    frames = read_dss(args.source)
    images, hashes, metadata, gps_count = load_images(frames)
    reviewed = json.loads(args.reviewed_groups.read_text()) if args.reviewed_groups else []
    assign_groups(frames, hashes, metadata, reviewed)
    print(f"DSS: {len(frames)} images opened, {gps_count} EXIF GPS entries. Extracting features.", flush=True)

    labeled = np.array([i for i, frame in enumerate(frames) if frame.truth is not None])
    y = np.array([frames[i].truth for i in labeled])
    groups = np.array([frames[i].group for i in labeled])
    stratification = np.array([frames[i].stratum for i in labeled])
    features = CabinetFeatures(args.device).extract(images, set(VARIANTS))
    main = {name: values[labeled] for name, values in features.items()}
    base = main["classical"]
    print(f"Features ready after {time.monotonic() - started:.1f}s. Nested grouped evaluation.", flush=True)

    oof = np.zeros(y.shape)
    classical_oof = np.zeros(y.shape)
    fold_for = {}
    outer = splits(stratification, groups, n=5, seed=42)
    for fold, (train, test) in enumerate(outer):
        if set(groups[train]) & set(groups[test]):
            raise RuntimeError("Нарушение разделения связанных фото.")
        for i in test:
            fold_for[labeled[i]] = fold
        nested = [(train[a], train[b]) for a, b in splits(stratification[train], groups[train], n=3, seed=43)]
        for j in range(y.shape[1]):
            selected = choose(main, y[:, j], nested)
            oof[test, j] = np.mean([predict_head(main[name], y[:, j], train, test) for name in selected], axis=0)
            classical_oof[test, j] = predict_head(base, y[:, j], train, test)
    scored = score_set(y, oof)
    baseline = score_set(y, classical_oof)
    artifact = {"format": MODEL_FORMAT, "heads": {}}
    for j, head in enumerate(HEADS):
        variant = choose(main, y[:, j], outer)
        artifact["heads"][head] = [
            {"variant": name, "estimator": estimator(main[name].shape[1]).fit(main[name], y[:, j])}
            for name in variant
        ]

    args.out.mkdir(parents=True, exist_ok=True)
    args.artifact.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.artifact / "model.joblib")
    predictions = {
        name: np.mean([spec["estimator"].predict_proba(features[spec["variant"]])[:, 1] for spec in specs], axis=0)
        for name, specs in artifact["heads"].items()
    }
    rows, oof_rows = [], []
    cfg = quality_config("lowres")
    for i, (frame, image) in enumerate(zip(frames, images, strict=True)):
        scores = {name: values[i] for name, values in predictions.items()}
        quality = assess_quality(image, bbox=None, cfg=cfg)
        frame_id = f"DSS-{i + 1:03d}"
        rows.append({"image_id": frame_id, "source_file": str(frame.path), **cabinet_result(scores, quality)})
    for k, i in enumerate(labeled):
        oof_rows.append({
            "image_id": rows[i]["image_id"], "fold": fold_for[i], "group": int(groups[k]),
            "truth": {head: int(y[k][j]) for j, head in enumerate(HEADS)},
            **cabinet_result(dict(zip(HEADS, oof[k], strict=True)), assess_quality(images[i], bbox=None, cfg=cfg)),
        })
    action_counts = Counter(row["action"] for row in rows)
    policy = {
        action: {
            "damage": sum(row["action"] == action and (row["truth"]["mechanical"] or row["truth"]["coating"])
                          for row in oof_rows),
            "normal": sum(row["action"] == action and not (row["truth"]["mechanical"] or row["truth"]["coating"])
                          for row in oof_rows),
        }
        for action in ("repair", "inspect", "no_action")
    }
    report = {
        "labeled": int(len(labeled)), "unlabeled": len(frames) - len(labeled), "total": len(frames),
        "dual": int(np.sum((y[:, 1:] == 1).all(axis=1))),
        "groups": int(len(set(groups))), "folds": len(outer), "gps_exif": gps_count,
        "label_counts": {name: int(y[:, i].sum()) for i, name in enumerate(HEADS)},
        "model": {
            "name": "DINOv2 × SigLIP 2 · три независимые головы",
            "validation": "5 внешних групповых OOF-фолдов, три лучшие модели каждой головы по 3 внутренним фолдам",
            "macroF1": scored["defect_macro_f1"],
            "signalMetrics": [{"code": name, "label": SIGNALS[name], **scored[name]} for name in HEADS],
            "features": {name: [spec["variant"] for spec in specs] for name, specs in artifact["heads"].items()},
        },
        "metrics": scored,
        "baseline": baseline,
        "policy_oof": policy,
        "prediction_counts": dict(action_counts),
        "seconds": round(time.monotonic() - started, 2),
    }
    write_json(args.out / "public_metrics.json", report)
    write_json(args.out / "predictions.json", rows)
    write_json(args.out / "oof_predictions.json", oof_rows)
    fields = ["image_id", "object_type", "object_type_score", "mechanical", "coating", "action", "reason"]
    with (args.out / "inspection_queue.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row["scores"].get(field, row.get(field)) for field in fields})
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--source", action="append", required=True, type=Path)
    train.add_argument("--out", default=Path("runs/final-cabinet"), type=Path)
    train.add_argument("--artifact", default=Path("artifacts/final-cabinet"), type=Path)
    train.add_argument("--reviewed-groups", type=Path)
    train.add_argument("--device", default="cpu")
    predict = commands.add_parser("predict")
    predict.add_argument("image", type=Path)
    predict.add_argument("--model", default=Path("artifacts/final-cabinet/model.joblib"), type=Path)
    args = parser.parse_args()
    if args.command == "train":
        run_training(args)
    else:
        print(json.dumps(CabinetInspector.load(args.model).predict_path(args.image), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
