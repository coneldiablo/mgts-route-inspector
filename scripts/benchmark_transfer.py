"""Compare frozen inspector artifacts on disjoint, labelled photo cohorts.

Run from the project root with ``python -m scripts.benchmark_transfer``. The
benchmark verifies the supplied training manifest against each artifact and
never selects a threshold on any of the evaluation cohorts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from inspector.classifier import InspectorModel, load_model
from inspector.config import AppConfig, PUBLIC_SOURCE, QualityConfig, quality_config
from inspector.manifest import Record, data_fingerprint, image_sha256, load_manifest
from inspector.pipeline import predict_records
from inspector.quality import open_image


def _rate(count: int, total: int) -> float | None:
    return round(count / total, 6) if total else None


def _rgb_digest(record: Record) -> str:
    with open_image(record.image_path) as image:
        return hashlib.sha256(f"{image.size}:".encode() + image.tobytes()).hexdigest()


def _check_disjoint(train: list[Record], cohorts: dict[str, list[Record]]) -> None:
    seen_assets = {r.asset_key for r in train}
    seen_images = {r.image_id for r in train}
    seen_bytes = {image_sha256(r.image_path) for r in train}
    seen_rgb = {_rgb_digest(r) for r in train}
    for name, records in cohorts.items():
        assets = {r.asset_key for r in records}
        images = {r.image_id for r in records}
        raw = {image_sha256(r.image_path) for r in records}
        rgb = {_rgb_digest(r) for r in records}
        overlaps = {
            "asset_key": len(seen_assets & assets),
            "image_id": len(seen_images & images),
            "bytes": len(seen_bytes & raw),
            "decoded_rgb": len(seen_rgb & rgb),
        }
        if any(overlaps.values()):
            raise ValueError(f"Выборка {name} пересекается с предыдущими данными: {overlaps}")
        seen_assets.update(assets)
        seen_images.update(images)
        seen_bytes.update(raw)
        seen_rgb.update(rgb)


def _archive_families(root: Path | None) -> dict[str, str]:
    if root is None:
        return {}
    families: dict[str, str] = {}
    for filename in ("train.zip", "test.zip"):
        with ZipFile(root / filename) as archive:
            for member in archive.namelist():
                path = Path(member)
                if path.suffix.lower() != ".jpg":
                    continue
                family = "void" if path.parent.name == "void" else path.stem.split(" (")[0]
                digest = hashlib.sha256(archive.read(member)).hexdigest()
                families.setdefault(digest, family)
    return families


def _cohort_groups(records: list[Record], families: dict[str, str]) -> dict[str, str]:
    return {
        r.image_id: families.get(image_sha256(r.image_path), r.source)
        if r.source == PUBLIC_SOURCE and families else r.source
        for r in records
    }


def _exclude_known_pairs(
    path: Path | None, train: list[Record], cohorts: dict[str, list[Record]],
) -> dict[str, dict[str, list[str]]]:
    excluded: dict[str, dict[str, list[str]]] = {name: {} for name in cohorts}
    if path is None:
        return excluded
    train_ids = {r.image_id for r in train}
    cohort_for_id = {r.image_id: name for name, records in cohorts.items() for r in records}
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not {"query", "train"}.issubset(reader.fieldnames or []):
            raise ValueError(f"В {path} нужны столбцы query и train.")
        for row in reader:
            name = cohort_for_id.get(row["query"])
            if row["train"] in train_ids and name is not None:
                matches = excluded[name].setdefault(row["query"], [])
                if row["train"] not in matches:
                    matches.append(row["train"])
    return excluded


def _benchmark_rows(
    name: str, cohort: str, records: list[Record], groups: dict[str, str], model: InspectorModel,
) -> list[dict]:
    threshold = float(model.threshold)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError(f"Порог артефакта {name} вне диапазона [0, 1].")
    settings = model.meta.get("quality_config")
    quality = QualityConfig(**settings) if settings else quality_config(model.meta.get("quality_profile", "default"))
    unusable = model.meta.get("unusable_label", "unusable")
    labelled = [r for r in records if r.label and r.label != unusable]
    if not labelled:
        raise ValueError(f"В {cohort} отсутствуют пригодные для оценки метки.")
    seen_labels = {r.label for r in labelled}
    if model.mode != "anomaly" and not seen_labels.issubset(model.classes):
        raise ValueError(f"Выборка {cohort} содержит неизвестные модели {name} классы: {sorted(seen_labels - set(model.classes))}")
    observations = predict_records(
        labelled, model, AppConfig(quality=quality), threshold, model.meta.get("crop_mode", "auto"), verbose=False,
    )
    return [
        {
            "model": name, "cohort": cohort, "image_id": o.record.image_id, "asset_id": o.record.asset_id,
            "source": o.record.source, "group": groups[o.record.image_id], "condition": o.record.condition,
            "label": o.record.label, "threshold": threshold, "status": o.status,
            "score": float(o.prediction.p_finding) if o.quality.ok else None,
            "raw_label": o.prediction.pred_label if o.quality.ok else None,
            "decision": (model.normal_label if o.status == "normal" else o.sign if o.status == "finding" else None),
            "refusal": " | ".join(o.quality.reasons),
        }
        for o in observations
    ]


def summarize(rows: list[dict], normal_label: str) -> dict:
    positives = [r for r in rows if r["label"] != normal_label]
    negatives = [r for r in rows if r["label"] == normal_label]
    tp = sum(r["status"] == "finding" for r in positives)
    fp = sum(r["status"] == "finding" for r in negatives)
    statuses = Counter(r["status"] for r in rows)
    refused_positive = sum(r["status"] == "rephoto" for r in positives)
    refused_negative = statuses["rephoto"] - refused_positive
    support = Counter(r["label"] for r in rows)
    predicted = Counter(r["decision"] for r in rows if r["decision"] is not None)
    correct = Counter(r["label"] for r in rows if r["label"] == r["decision"])
    per_label = {
        label: {"support": n, "predicted": predicted[label], "correct": correct[label],
                "precision": _rate(correct[label], predicted[label]), "recall": _rate(correct[label], n),
                "f1": _rate(2 * correct[label], n + predicted[label])}
        for label, n in sorted(support.items())
    }
    return {
        "n": len(rows), "positive": len(positives), "negative": len(negatives),
        "tp": tp, "fp": fp,
        "fn_usable": len(positives) - refused_positive - tp,
        "tn_usable": len(negatives) - refused_negative - fp,
        "refused_positive": refused_positive, "refused_negative": refused_negative,
        "statuses": dict(statuses), "coverage": _rate(len(rows) - statuses["rephoto"], len(rows)),
        "precision": _rate(tp, tp + fp), "recall_usable": _rate(tp, len(positives) - refused_positive),
        "recall_e2e": _rate(tp, len(positives)),
        "f1_e2e": _rate(2 * tp, len(positives) + tp + fp) if positives else None,
        "fpr_usable": _rate(fp, len(negatives) - refused_negative), "fpr_e2e": _rate(fp, len(negatives)),
        "review_rate": _rate(len(rows) - statuses["normal"], len(rows)),
        "label_accuracy": _rate(sum(correct.values()), len(rows)),
        "label_macro_f1": round(sum(x["f1"] for x in per_label.values()) / len(per_label), 6) if len(per_label) > 1 else None,
        "per_label": per_label,
    }


def _render(report: dict) -> str:
    def pct(value):
        return "—" if value is None else f"{value:.2%}"

    label_note = (" Разметка `manhole/void` относится к видимости крышки."
                  if PUBLIC_SOURCE in report["train"]["sources"] else "")
    lines = ["# Сравнение на отложенных кадрах", "",
             f"Пороги взяты из сохранённых моделей.{label_note} "
             "Прочерки означают, что метрика не определена для имеющихся классов.", "",
             "| Модель | Когорта | Кадров | Порог | TP | FP | Отказы | Precision | Recall со всеми кадрами | FPR со всеми кадрами | Все запросы мастеру | F1 по типам |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for result in report["results"]:
        m = result["metrics"]
        lines.append(f"| {result['model']} | {result['cohort']} | {m['n']} | {result['threshold']:.4g} | "
                     f"{m['tp']} | {m['fp']} | {m['statuses'].get('rephoto', 0)} | {pct(m['precision'])} | "
                     f"{pct(m['recall_e2e'])} | {pct(m['fpr_e2e'])} | {pct(m['review_rate'])} | {pct(m['label_macro_f1'])} |")
    lines.extend(["", "Запросы мастеру включают находки, слабые сигналы и запросы повторного кадра. "
                  "Перед оценкой проверены пересечения по ID, исходному файлу и декодированному изображению.", ""])
    for name, cohort in report["cohorts"].items():
        if cohort["excluded_similar"]:
            lines.append(f"- {name}. Вероятные повторы обучающих сцен вне оценки — "
                         f"{', '.join(cohort['excluded_similar'])}.")
    if report["public_archive_groups"]:
        lines.extend(["", "## Группы исходных файлов", "",
                      "Группы восстановлены по именам файлов в архиве. Для подтверждения переноса между маршрутами "
                      "понадобятся идентификаторы маршрутов.", "",
                      "| Модель | Когорта | Группа файлов | Кадров | FPR | Recall |",
                      "|---|---|---|---:|---:|---:|"])
        for result in report["results"]:
            for group, m in result["groups"].items():
                lines.append(f"| {result['model']} | {result['cohort']} | {group} | {m['n']} | "
                             f"{pct(m['fpr_e2e'])} | {pct(m['recall_e2e'])} |")
    return "\n".join(lines) + "\n"


def _named_paths(items: list[str]) -> dict[str, Path]:
    paths = {}
    for item in items:
        name, sep, path = item.partition("=")
        if not sep or not name or not path or name in paths:
            raise ValueError(f"Укажите уникальное имя и путь в формате NAME=PATH. Получено {item!r}.")
        paths[name] = Path(path)
    return paths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Сравнение замороженных моделей на отложенных кадрах.")
    parser.add_argument("--model", action="append", required=True, metavar="NAME=ARTIFACT")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--holdout", action="append", required=True, metavar="NAME=DATA")
    parser.add_argument("--similar-pairs", type=Path)
    parser.add_argument("--public-archives", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.out.exists():
            raise ValueError("Каталог отчёта уже существует. Выберите новый путь.")
        models = _named_paths(args.model)
        paths = _named_paths(args.holdout)
        train = load_manifest(args.train)
        cohorts = {name: load_manifest(path) for name, path in paths.items()}
        _check_disjoint(train, cohorts)
        excluded = _exclude_known_pairs(args.similar_pairs, train, cohorts)
        families = _archive_families(args.public_archives)
        groups = {name: _cohort_groups(records, families) for name, records in cohorts.items()}
        train_hash = data_fingerprint(train)
        report = {"train": {"path": str(args.train), "fingerprint": train_hash, "n": len(train),
                            "sources": dict(Counter(r.source for r in train))}, "cohorts": {
            name: {"path": str(paths[name]), "fingerprint": data_fingerprint(records),
                   "n_manifest": len(records), "excluded_similar": excluded[name]}
            for name, records in cohorts.items()
        }, "public_archive_groups": bool(families), "results": []}
        all_rows = []
        for name, artifact in models.items():
            model = load_model(artifact)
            if model.meta.get("data_fingerprint") != train_hash:
                raise ValueError(f"Обучающий манифест модели {name} отличается от переданного через --train.")
            if float(model.meta.get("threshold", math.nan)) != model.threshold:
                raise ValueError(f"В артефакте {name} сохранены разные пороги.")
            for cohort, records in cohorts.items():
                start = time.monotonic()
                kept = [rec for rec in records if rec.image_id not in excluded[cohort]]
                rows = _benchmark_rows(name, cohort, kept, groups[cohort], model)
                slices: dict[str, list[dict]] = {}
                for row in rows:
                    slices.setdefault(row["group"], []).append(row)
                report["results"].append({
                    "model": name, "artifact": str(artifact), "backbone": model.meta.get("backbone"),
                    "threshold": model.threshold, "threshold_reason": model.meta.get("threshold_reason"),
                    "task": model.meta.get("task"), "cohort": cohort, "normal_label": model.normal_label,
                    "seconds": round(time.monotonic() - start, 3), "metrics": summarize(rows, model.normal_label),
                    "groups": {group: summarize(part, model.normal_label) for group, part in sorted(slices.items())},
                })
                all_rows.extend(rows)
        args.out.mkdir(parents=True)
        (args.out / "benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        (args.out / "benchmark.md").write_text(_render(report), encoding="utf-8")
        with (args.out / "predictions.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
        print(args.out / "benchmark.md")
    except (OSError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
