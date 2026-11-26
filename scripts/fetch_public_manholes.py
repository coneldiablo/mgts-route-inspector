"""Fetch a public real-photo manhole dataset and build manifests.

Source: https://huggingface.co/datasets/delima87/manhole_covers_dataset
Uploader-declared license: Apache-2.0.
Classes: `manhole` -> normal reference, `void` -> unconfirmed_cover (review framing).
The dataset is image-level, 224x224 and was collected from the web; it is not MGTS
field data. Keep SOURCE.md next to the images and verify redistribution rights.

The author's `test/void` files duplicate `train/void`, so the official test split
cannot be used for recall. Instead:
  * a stratified holdout is taken from the train split (both classes, no hash overlap);
  * the official test `manhole` photos are kept as an extra normal-only check.

Usage:
    python scripts/fetch_public_manholes.py \
        --out data/real --holdout-out data/real_holdout --official-out data/real_official_test
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
import tempfile
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

BASE = "https://huggingface.co/datasets/delima87/manhole_covers_dataset/resolve/main/manhole_covers_dataset"
SPLITS = {
    "train": f"{BASE}/train.zip",
    "test": f"{BASE}/test.zip",
}
ARCHIVE_SHA256 = {
    "train": "9f5be52f90a51808d8153b7fd561c2bcef73fc67981528ae40b5bf013214b737",
    "test": "e440ab51882ed3b69527897152a489c8d56d8dc562e4a7543eaa1f9733854eea",
}
LABEL_MAP = {"manhole": "normal", "void": "unconfirmed_cover"}
SOURCE = "public:delima87/manhole_covers_dataset"
LICENSE = "Apache-2.0 (uploader-declared)"
FIELDS = [
    "image", "image_id", "asset_id", "label", "object_type", "captured_at",
    "lat", "lon", "condition", "criticality", "source", "license", "dataset_split", "source_label",
]


def _download(url: str, path: Path) -> None:
    if path.is_file() and _sha256(path) == ARCHIVE_SHA256[path.stem]:
        print(f"  уже скачано: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  скачиваю {url}")
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        pending = Path(tmp) / path.name
        with urllib.request.urlopen(url, timeout=120) as response, pending.open("wb") as fh:
            shutil.copyfileobj(response, fh)
        if _sha256(pending) != ARCHIVE_SHA256[path.stem]:
            raise ValueError(f"Контрольная сумма архива {path.name} изменилась.")
        pending.replace(path)
    print(f"  {path} — {path.stat().st_size / 1024 / 1024:.1f} МБ")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract(zip_paths: list[Path], work_dir: Path) -> Path:
    work_dir.mkdir(parents=True, exist_ok=False)
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(work_dir)
    return work_dir


def _collect(split_dir: Path, split: str, seen: set[str]) -> list[dict]:
    """Collect unique files, skipping exact duplicates already seen."""
    rows: list[dict] = []
    for folder, label in LABEL_MAP.items():
        for src in sorted((split_dir / folder).glob("*.jpg")):
            digest = _sha256(src)
            if digest in seen:
                continue
            seen.add(digest)
            rows.append(
                {
                    "label": label,
                    "source_label": folder,
                    "object_type": "well",
                    "captured_at": "",
                    "lat": "",
                    "lon": "",
                    "condition": "unknown",
                    "criticality": "0.5",
                    "source": SOURCE,
                    "license": LICENSE,
                    "origin": split,
                    "_src": src,
                }
            )
    return rows


def _assign_ids(rows: list[dict], prefix: str, split: str) -> list[dict]:
    for index, row in enumerate(rows, start=1):
        asset = f"{prefix}-{index:05d}"
        row["image"] = f"images/{asset}.jpg"
        row["image_id"] = asset
        row["asset_id"] = asset
        row["dataset_split"] = split
    return rows


def _stratified_split(rows: list[dict], fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Hold out a share of every class, deterministically and without hash overlap."""
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)
    rng = random.Random(seed)
    train: list[dict] = []
    holdout: list[dict] = []
    for label in sorted(by_label):
        items = by_label[label][:]
        rng.shuffle(items)
        if len(items) < 2:
            raise ValueError(f"Недостаточно фото класса {label} для разделения")
        n_hold = min(len(items) - 1, max(1, round(len(items) * fraction)))
        holdout.extend(items[:n_hold])
        train.extend(items[n_hold:])
    rng.shuffle(train)
    rng.shuffle(holdout)
    return train, holdout


def _write_split(rows: list[dict], out_dir: Path, split: str, title: str) -> None:
    images = out_dir / "images"
    images.mkdir(parents=True)
    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            shutil.copyfile(row["_src"], images / Path(row["image"]).name)
            writer.writerow({k: row[k] for k in FIELDS})
    positives = sum(r["label"] != "normal" for r in rows)
    (out_dir / "SOURCE.md").write_text(
        _source_md(title, len(rows), positives, split), encoding="utf-8"
    )
    print(
        f"  {split}: {len(rows)} фото, normal={len(rows) - positives}, "
        f"void={positives} -> {out_dir}"
    )


def _source_md(title: str, n_images: int, n_positive: int, split: str) -> str:
    return f"""# Источник данных: {title}

- Датасет: `delima87/manhole_covers_dataset`
- URL: https://huggingface.co/datasets/delima87/manhole_covers_dataset
- Лицензия по заявлению автора: **Apache-2.0**
- Исходные классы сохранены в source_label. Метка manhole соответствует классу normal в демонстрации. Метка void соответствует unconfirmed_cover и означает запрос на проверку кадра. Состояние крышки в наборе не размечено.
- Размер исходных изображений: 224×224; это не оригиналы МГТС, а веб-подборка.
- Разметка уровневая (на изображение), без bbox; возможны слабые/ошибочные метки.
- Эта часть: `{split}`, {n_images} фото, из них положительных {n_positive}.
- Отложенная часть собрана стратифицированно из train; официальный test/void
  дублирует train/void, поэтому для recall он не годится.
- Изображения не являются данными МГТС и не подтверждают качество на реальных маршрутах.
- Перед публикацией или коммерческим использованием проверьте права на исходные фотографии.

Скрипт загрузки: `scripts/fetch_public_manholes.py`.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Скачать публичный датасет колодцев и собрать manifest")
    parser.add_argument("--out", default="data/real", help="обучающая часть")
    parser.add_argument("--holdout-out", default="data/real_holdout", help="отложенная часть")
    parser.add_argument("--official-out", default="data/real_official_test", help="официальный test (только normal)")
    parser.add_argument("--cache", default="data/_downloads/manhole_covers", help="кэш zip-файлов")
    parser.add_argument("--holdout-fraction", type=float, default=0.12, help="доля отложенной части каждого класса")
    parser.add_argument("--seed", type=int, default=7, help="детерминированное перемешивание")
    args = parser.parse_args(argv)

    if not 0 < args.holdout_fraction < 1:
        parser.error("--holdout-fraction должна находиться между 0 и 1")
    outputs = [Path(p).resolve() for p in (args.out, args.holdout_out, args.official_out)]
    paths = [*outputs, Path(args.cache).resolve()]
    if any(a == b or a in b.parents or b in a.parents for i, a in enumerate(paths) for b in paths[i + 1:]):
        parser.error("Для выборок и кэша нужны отдельные каталоги без вложения")
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    zips = {}
    for split, url in SPLITS.items():
        path = cache / f"{split}.zip"
        _download(url, path)
        zips[split] = path

    with tempfile.TemporaryDirectory(dir=cache) as scratch:
        staging = Path(scratch)
        work = staging / "extracted"
        _extract([zips["train"], zips["test"]], work)

        seen: set[str] = set()
        train_pool = _collect(work / "train", "train", seen)
        official_test = _collect(work / "test", "test", seen)

        train_rows, holdout_rows = _stratified_split(train_pool, args.holdout_fraction, args.seed)
        _assign_ids(train_rows, "MH", "train")
        _assign_ids(holdout_rows, "MHH", "holdout")
        _assign_ids(official_test, "MHT", "official_test")


        planned = [(train_rows, outputs[0], "train", "обучающая часть"),
                   (holdout_rows, outputs[1], "holdout", "отложенная часть"),
                   (official_test, outputs[2], "official_test", "официальный test")]
        legacy_dirs = set()
        for rows, dest, split, title in planned:
            if dest.exists() and any(dest.iterdir()):
                manifest = dest / "manifest.csv"
                existing = []
                if manifest.is_file():
                    with manifest.open(newline="", encoding="utf-8") as fh:
                        existing = list(csv.DictReader(fh))
                expected = [{k: row[k] for k in FIELDS} for row in rows]
                old = [{k: ("open_hatch" if k == "label" and v == "unconfirmed_cover" else v)
                        for k, v in item.items() if k != "source_label"} for item in expected]
                if existing == old:
                    legacy_dirs.add(dest)
                if (existing != expected and dest not in legacy_dirs or not (dest / "SOURCE.md").is_file()
                        or any(not (dest / row["image"]).is_file()
                               or _sha256(dest / row["image"]) != _sha256(row["_src"]) for row in rows)):
                    raise ValueError(f"Каталог {dest} занят другой версией данных. Выберите новую папку.")
        for rows, dest, split, title in planned:
            if dest in legacy_dirs:
                with tempfile.TemporaryDirectory(dir=dest) as tmp:
                    manifest = Path(tmp) / "manifest.csv"
                    with manifest.open("w", newline="", encoding="utf-8") as fh:
                        writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
                        writer.writeheader()
                        writer.writerows(rows)
                    source = Path(tmp) / "SOURCE.md"
                    source.write_text(_source_md(title, len(rows), sum(r["label"] != "normal" for r in rows), split), encoding="utf-8")
                    source.replace(dest / source.name)
                    manifest.replace(dest / manifest.name)
                print(f"  {split}: обновлены метки проверенных фото в {dest}")
                continue
            if dest.exists() and any(dest.iterdir()):
                print(f"  {split}: {len(rows)} фото уже проверены в {dest}")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
                pending = Path(tmp) / "split"
                _write_split(rows, pending, split, title)
                pending.rename(dest)
    print("Готово. Данные синтетикой не являются; см. SOURCE.md в каждой папке.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
