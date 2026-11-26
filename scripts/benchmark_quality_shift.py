"""Measure a fixed model under reproducible image-quality corruptions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageEnhance, ImageFilter

from inspector.classifier import load_model
from inspector.manifest import Record, data_fingerprint, load_manifest
from inspector.quality import open_image
from scripts.benchmark_transfer import _benchmark_rows, _check_disjoint, _named_paths, summarize

EXPOSURE_FACTOR = 0.45
BLUR_RADIUS = 2.0
RESIZE_FACTOR = 0.50
SCENARIOS = {
    "clean": "без изменения",
    "dark": f"яркость ×{EXPOSURE_FACTOR:g}",
    "blur": f"гауссово размытие, радиус {BLUR_RADIUS:g} px",
    "downscale": f"размер кадра ×{RESIZE_FACTOR:g}",
}


def _variants(records: list[Record], scenario: str, directory: Path) -> list[Record]:
    if scenario == "clean":
        return records

    directory.mkdir()
    transformed = []
    for index, record in enumerate(records):
        image = open_image(record.image_path)
        bbox = record.bbox
        if scenario == "dark":
            image = ImageEnhance.Brightness(image).enhance(EXPOSURE_FACTOR)
        elif scenario == "blur":
            image = image.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))
        elif scenario == "downscale":
            width, height = image.size
            size = (max(1, round(width * RESIZE_FACTOR)), max(1, round(height * RESIZE_FACTOR)))
            image = image.resize(size, Image.Resampling.LANCZOS)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                sx, sy = size[0] / width, size[1] / height
                bbox = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
        else:
            raise ValueError(f"Неизвестное изменение изображения {scenario!r}.")

        path = directory / f"{index:05}.png"
        image.save(path)
        transformed.append(replace(record, image_path=path, bbox=bbox))
    return transformed


def _markdown(report: dict) -> str:
    def percent(value: float | None) -> str:
        return "—" if value is None else f"{value:.1%}"

    lines = [
        "# Устойчивость к ухудшению изображения", "",
        "Яркость, размытие и уменьшение применены к полным отложенным кадрам. "
        "Затем штатный конвейер проверил качество и выполнил кадрирование. "
        "Координаты рамок масштабированы вместе с кадрами. Пороги зафиксированы при обучении.", "",
        "Ухудшения имитированы на тех же фотографиях. Результаты оценивают устойчивость "
        "к указанным преобразованиям. Ночные маршруты и реальная съёмка в движении требуют самостоятельной выборки.", "",
        "Accuracy и recall включают кадры, оставленные без решения. FP — находки на кадрах нормы. "
        "Слабый сигнал и запрос пересъёмки указаны отдельно.", "",
        "| Модель | Выборка | Условие | N | TP | FP | Слабый сигнал | Пересъёмка | Accuracy | Recall | FPR |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["results"]:
        m = row["metrics"]
        lines.append(
            f"| {row['model']} | {row['cohort']} | {SCENARIOS[row['scenario']]} | {m['n']} | "
            f"{m['tp']} | {m['fp']} | {m['statuses'].get('uncertain', 0)} | {m['statuses'].get('rephoto', 0)} | "
            f"{percent(m['label_accuracy'])} | {percent(m['recall_e2e'])} | {percent(m['fpr_e2e'])} |"
        )
    lines.extend(["", "`predictions.csv` содержит решения по кадрам и причины запроса пересъёмки.", ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Проверка устойчивости замороженных моделей к тёмным и нерезким кадрам.")
    parser.add_argument("--model", required=True, action="append", metavar="NAME=ARTIFACT")
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--holdout", required=True, action="append", metavar="NAME=DATA")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        if args.out.exists():
            raise ValueError("Каталог отчёта уже существует. Выберите новый путь.")
        train = load_manifest(args.train)
        cohort_paths = _named_paths(args.holdout)
        cohorts = {name: load_manifest(path) for name, path in cohort_paths.items()}
        _check_disjoint(train, cohorts)
        fingerprint = data_fingerprint(train)

        models = {name: load_model(path) for name, path in _named_paths(args.model).items()}
        for name, model in models.items():
            if model.meta.get("data_fingerprint") != fingerprint:
                raise ValueError(f"Переданная обучающая выборка не соответствует модели {name}.")
            if float(model.meta.get("threshold", math.nan)) != model.threshold:
                raise ValueError(f"В артефакте {name} сохранены разные пороги.")

        report = {
            "train": {"path": str(args.train), "fingerprint": fingerprint, "n": len(train)},
            "cohorts": {name: {"path": str(cohort_paths[name]), "n": len(rows),
                               "fingerprint": data_fingerprint(rows)} for name, rows in cohorts.items()},
            "scenarios": SCENARIOS,
            "simulated": True,
            "results": [],
        }
        all_rows = []
        with TemporaryDirectory(prefix="inspector-quality-shift-") as directory:
            root = Path(directory)
            for scenario in SCENARIOS:
                for index, (cohort, records) in enumerate(cohorts.items()):
                    changed = _variants(records, scenario, root / f"{index}-{scenario}")
                    groups = {rec.image_id: rec.source for rec in changed}
                    for name, model in models.items():
                        rows = _benchmark_rows(name, cohort, changed, groups, model)
                        all_rows.extend({"scenario": scenario, **row} for row in rows)
                        reasons = Counter(reason for row in rows if row["status"] == "rephoto"
                                          for reason in row["refusal"].split(" | ") if reason)
                        report["results"].append({
                            "model": name, "backbone": model.meta.get("backbone"), "cohort": cohort,
                            "scenario": scenario, "threshold": model.threshold,
                            "metrics": summarize(rows, model.normal_label), "refusal_reasons": dict(reasons),
                        })

        args.out.mkdir(parents=True)
        (args.out / "benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        (args.out / "benchmark.md").write_text(_markdown(report), encoding="utf-8")
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
