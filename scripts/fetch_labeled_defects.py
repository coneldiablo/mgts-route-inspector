"""Build source-separated photo benchmarks with manual open/closed box labels."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import shutil
import subprocess
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from PIL import Image, ImageOps


SOURCES = {
    "cube": (
        "cubeai/manhole-for-yolov8",
        "9e083e9cd3ee3c4ca64cfa8fc37079058bba87696424a033dbafca1bdbc7ca7f",
        "CC0 (Kaggle uploader declaration)",
    ),
    "deep": (
        "deeppratap/manhole-detection",
        "7f145af3f6a7825ce62a5023f341f02312d39f128b11d525ddd7c67194d27cbf",
        "CC BY 4.0 (original Roboflow export)",
    ),
}
ROBOFLOW_SOURCE = "https://universe.roboflow.com/school-kvcpg/manholes-16igm/dataset/1"
LABELS = {0: "normal", 1: "open_hatch"}
MAX_HASH_DISTANCE = 12
VERIFIED_HOLDOUT = {
    "cube": {
        "close105.jpg", "close106.jpg", "close117.jpg", "close153.jpg", "close156.jpg", "close159.jpg",
        "close169.jpg", "close20.jpg", "close201.jpg", "close21.jpg", "close26.jpg", "close37.jpg",
        "close39.jpg", "close40.jpg", "close54.jpg", "close76.jpg", "close8.jpg",
        "open125.jpg", "open131.jpg", "open140.jpg", "open141.jpg", "open169.jpg", "open193.jpg",
        "open3.jpg", "open45.jpg", "open50.jpg", "open60.jpg", "open71.jpg", "open87.jpg", "open94.jpg", "open95.jpg",
    },
    "deep": {
        "0001_jpg", "0008_jpg", "0012_jpg", "0018_jpg", "5205b40a5f88173d98735501441458f_jpg",
        "939a16af3258bf749afa1245e2fef857_jpeg_jpg", "1_20220707100532_jpg", "Drain-and-Cover-Lifting-mary-Blog-082016_jpg",
        "IMG_2034_jpg",
        "dangerous-opened-manhole-hole-cover-opened-sewerage-hatch-traffic-background-"
        "dangerous-opened-manhole-hole-cover-opened-162167819_jpg",
        "manhole-20130425_4634_jpg",
        "open-hatch-broken-manhole-cover-danger-zone-open-hatch-broken-manhole-plastic-cover-218797501_jpg",
        "walkway-opened-sewers-hatch-21404592_jpg",
    },
}
VERIFIED_CUBE_TRAIN = {
    'close1.jpg', 'close101.jpg', 'close102.jpg', 'close104.jpg', 'close107.jpg', 'close108.jpg',
    'close110.jpg', 'close112.jpg', 'close113.jpg', 'close118.jpg', 'close119.jpg', 'close124.jpg',
    'close13.jpg', 'close131.jpg', 'close133.jpg', 'close134.jpg', 'close136.jpg', 'close142.jpg',
    'close147.jpg', 'close15.jpg', 'close150.jpg', 'close151.jpg', 'close154.jpg', 'close155.jpg',
    'close158.jpg', 'close16.jpg', 'close163.jpg', 'close165.jpg', 'close167.jpg', 'close168.jpg',
    'close172.jpg', 'close174.jpg', 'close176.jpg', 'close177.jpg', 'close178.jpg', 'close180.jpg',
    'close182.jpg', 'close186.jpg', 'close188.jpg', 'close189.jpg', 'close190.jpg', 'close191.jpg',
    'close194.jpg', 'close195.jpg', 'close197.jpg', 'close27.jpg', 'close3.jpg', 'close36.jpg',
    'close44.jpg', 'close46.jpg', 'close50.jpg', 'close52.jpg', 'close53.jpg', 'close56.jpg',
    'close57.jpg', 'close58.jpg', 'close6.jpg', 'close61.jpg', 'close63.jpg', 'close66.jpg', 'close7.jpg',
    'close73.jpg', 'close77.jpg', 'close80.jpg', 'close84.jpg', 'close87.jpg', 'close90.jpg',
    'close92.jpg', 'close95.jpg',
    'open0.jpg', 'open1.jpg', 'open104.jpg', 'open105.jpg', 'open108.jpg', 'open110.jpg', 'open116.jpg',
    'open12.jpg', 'open120.jpg', 'open121.jpg', 'open130.jpg', 'open133.jpg', 'open134.jpg',
    'open137.jpg', 'open14.jpg', 'open144.jpg', 'open145.jpg', 'open148.jpg', 'open15.jpg', 'open171.jpg',
    'open173.jpg', 'open177.jpg', 'open179.jpg', 'open184.jpg', 'open19.jpg', 'open190.jpg',
    'open191.jpg', 'open198.jpg', 'open2.jpg', 'open20.jpg', 'open202.jpg', 'open203.jpg', 'open204.jpg',
    'open25.jpg', 'open26.jpg', 'open30.jpg', 'open31.jpg', 'open37.jpg', 'open44.jpg', 'open47.jpg',
    'open48.jpg', 'open53.jpg', 'open55.jpg', 'open57.jpg', 'open58.jpg', 'open59.jpg', 'open61.jpg',
    'open62.jpg', 'open70.jpg', 'open76.jpg', 'open78.jpg', 'open79.jpg', 'open8.jpg', 'open85.jpg',
    'open88.jpg', 'open89.jpg', 'open9.jpg', 'open91.jpg', 'open92.jpg',
}


@dataclass(frozen=True)
class Photo:
    source: str
    split: str
    name: str
    original: str
    raw: bytes
    digest: str
    fingerprints: tuple[int, int]
    label: str
    bbox: tuple[int, int, int, int]


def _image_hash(image: Image.Image) -> int:
    """128 difference bits, split among horizontal and vertical gradients."""
    gray = image.convert("L")
    horizontal = gray.resize((9, 8), Image.Resampling.LANCZOS).tobytes()
    vertical = gray.resize((8, 9), Image.Resampling.LANCZOS).tobytes()
    result = 0
    for y in range(8):
        for x in range(8):
            i = y * 9 + x
            j = y * 8 + x
            result = (result << 1) | (horizontal[i] > horizontal[i + 1])
            result = (result << 1) | (vertical[j] > vertical[j + 8])
    return result


def _read_archive(path: Path, source: str, report: Counter) -> list[Photo]:
    photos = []
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        for name in sorted(names):
            entry = PurePosixPath(name)
            if len(entry.parts) != 3 or entry.parts[0] not in {"train", "valid", "test"}:
                continue
            split, folder, filename = entry.parts
            if folder != "images" or entry.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            report["images"] += 1
            label_path = f"{split}/labels/{entry.stem}.txt"
            if label_path not in names:
                raise ValueError(f"Missing label file: {name}")
            annotations = [line.split() for line in archive.read(label_path).decode().splitlines() if line.strip()]
            if len(annotations) != 1:
                report["not_single_object"] += 1
                continue
            class_id, *corners = annotations[0]
            if len(corners) != 4 or int(class_id) not in LABELS:
                raise ValueError(f"Unexpected annotation in {label_path}")
            x, y, w, h = map(float, corners)
            if not all(0 <= value <= 1 for value in (x, y, w, h)) or not (w > 0 and h > 0):
                raise ValueError(f"Invalid normalized bounds in {label_path}")
            raw = archive.read(entry.as_posix())
            with Image.open(io.BytesIO(raw)) as image:
                rgb = image.convert("RGB")
                iw, ih = rgb.size
                bbox = (
                    max(0, math.floor((x - w / 2) * iw)), max(0, math.floor((y - h / 2) * ih)),
                    min(iw, math.ceil((x + w / 2) * iw)), min(ih, math.ceil((y + h / 2) * ih)),
                )
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    raise ValueError(f"Empty clipped bounds in {label_path}")
                digest = hashlib.sha256(f"{iw}x{ih}:".encode() + rgb.tobytes()).hexdigest()
                fingerprints = _image_hash(rgb), _image_hash(ImageOps.mirror(rgb))
            original = entry.stem.split(".rf.", 1)[0]
            photos.append(Photo(source, split, filename, original, raw, digest, fingerprints, LABELS[int(class_id)], bbox))
    return photos


def _matched(one: Photo, other: Photo) -> bool:
    same_stem = one.source == other.source and one.original == other.original
    same_pixels = one.digest == other.digest
    return same_stem or same_pixels or any(
        (candidate ^ reference).bit_count() <= MAX_HASH_DISTANCE
        for candidate in one.fingerprints for reference in other.fingerprints
    )


def _groups(photos: list[Photo]) -> list[list[Photo]]:
    roots = list(range(len(photos)))

    def root(index: int) -> int:
        while roots[index] != index:
            roots[index] = roots[roots[index]]
            index = roots[index]
        return index

    for i, photo in enumerate(photos):
        for j, other in enumerate(photos[:i]):
            if _matched(photo, other):
                roots[root(j)] = root(i)
    groups = defaultdict(list)
    for i, photo in enumerate(photos):
        groups[root(i)].append(photo)
    return list(groups.values())


def _publish(out: Path, photos: list[Photo], rejected: Counter) -> None:
    splits = defaultdict(list)
    for group in _groups(photos):
        if len({p.label for p in group}) != 1:
            rejected["label_conflict"] += len(group)
            continue
        primary = [p for p in group if p.source == "cube"]
        candidates = primary or group
        chosen = min(candidates, key=lambda p: ({"test": 0, "valid": 1, "train": 2}[p.split], p.name))
        split = ("cube/train" if chosen.split == "train" else "cube/holdout") if primary else "deep/holdout"
        splits[split].append(chosen)
        rejected["duplicate_variant"] += len(group) - 1
        if primary:
            rejected["external_close_to_primary"] += len(group) - len(primary)

    expected = {"cube/train", "cube/holdout", "deep/holdout"}
    if splits.keys() != expected or any(set(p.label for p in rows) != set(LABELS.values()) for rows in splits.values()):
        raise ValueError("Every split must contain both classes")
    for split, rows in sorted(splits.items()):
        directory = out / split
        (directory / "images").mkdir(parents=True)
        columns = ["image", "image_id", "asset_id", "id_origin", "label", "object_type", "source", "source_split",
                   "source_filename", "license", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
        with (directory / "manifest.csv").open("w", encoding="utf-8", newline="") as manifest:
            writer = csv.writer(manifest)
            writer.writerow(columns)
            for p in sorted(rows, key=lambda row: row.name):
                identifier = f"{p.source}-{p.digest[:16]}"
                image = f"images/{identifier}{Path(p.name).suffix.lower()}"
                (directory / image).write_bytes(p.raw)
                dataset, _, license_name = SOURCES[p.source]
                writer.writerow([image, identifier, identifier, "image", p.label, "manhole", f"public:{dataset}",
                                 p.split, p.name, license_name, *p.bbox])

        notes = (
            f"# {split}\n\n"
            f"Source: https://www.kaggle.com/datasets/{SOURCES[rows[0].source][0]}\n\n"
            + (f"Original archive and dataset attribution: {ROBOFLOW_SOURCE}\n\n" if rows[0].source == "deep" else "")
            + f"License recorded at source: {SOURCES[rows[0].source][2]}.\n\n"
            "The supervised target is visible closure status. `normal` means the publisher labeled the cover closed; "
            "it is not a comprehensive condition inspection. `open_hatch` is the source's open label. Retained images "
            "have exactly one annotated object and original source pixels. Boxes use source annotations.\n\n"
            "The authors do not provide physical asset or route IDs. Grouping approximates scene uniqueness using "
            "source stems, decoded-pixel hashes and perceptual comparisons including horizontal flips. Related images are "
            "represented once. Cube holds out official validation/test groups; Deep is a separate-source check after "
            "screening against the full Cube input. Different scenes or assets are not guaranteed. The sources may "
            "include close-ups and internet imagery of unknown collection origin. No coordinates are generated.\n"
        )
        (directory / "SOURCE.md").write_text(notes, encoding="utf-8")

    report = {split: dict(Counter(p.label for p in rows)) for split, rows in sorted(splits.items())}
    (out / "summary.json").write_text(json.dumps({"splits": report, "excluded_after_grouping": rejected}, indent=2) + "\n")
    print(json.dumps({"splits": report, "excluded_after_grouping": rejected}, indent=2))


def _publish_verified(out: Path) -> None:
    report = {}
    selections = {(source, "holdout"): selected for source, selected in VERIFIED_HOLDOUT.items()}
    selections["cube", "train"] = VERIFIED_CUBE_TRAIN
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        verified = Path(tmp) / "verified"
        for (source, split), selected in selections.items():
            source_dir = out / source / split
            with (source_dir / "manifest.csv").open(encoding="utf-8", newline="") as manifest:
                reader = csv.DictReader(manifest)
                columns = list(reader.fieldnames or ())
                rows = [row for row in reader if row["source_filename"].split(".rf.", 1)[0] in selected]
            if len(rows) != len(selected):
                raise ValueError(f"Curated source mismatch for {source}/{split}, expected {len(selected)}, found {len(rows)}")
            target = verified / source / split
            (target / "images").mkdir(parents=True)
            with (target / "manifest.csv").open("w", encoding="utf-8", newline="") as manifest:
                writer = csv.DictWriter(manifest, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
            for row in rows:
                image = Path(row["image"])
                shutil.copyfile(source_dir / image, target / image)
            notes = (source_dir / "SOURCE.md").read_text(encoding="utf-8")
            notes += (
                "\nA manual whole-frame review retained visibly open or visibly closed access holes with usable source boxes. "
                "Renders, cartoons, tunnel interiors, drains, unclear views and clear repetitions of the same view sequence "
                "were omitted. Several holes are open during planned work. Closure labels alone do not establish urgent damage. "
                "Curation changes the class distribution and the holdout size limits metric precision. "
                "This is an offline benchmark on third-party material, with no verified municipal asset or route IDs.\n"
            )
            (target / "SOURCE.md").write_text(notes, encoding="utf-8")
            report[f"{source}/{split}"] = dict(Counter(row["label"] for row in rows))
        destination = out / "verified"
        for source, split in selections:
            existing = destination / source / split / "manifest.csv"
            staged = verified / source / split / "manifest.csv"
            if existing.exists() and existing.read_bytes() != staged.read_bytes():
                raise ValueError(f"Refusing to overwrite curated split: {existing}")
        for source, split in selections:
            target = destination / source / split
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                (verified / source / split).replace(target)
        summary = verified / "summary.json"
        summary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        summary.replace(destination / "summary.json")
    print(json.dumps({"verified_splits": report}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/ml-bench"))
    parser.add_argument("--archives", type=Path, help="Directory containing verified cube.zip and deep.zip")
    parser.add_argument("--verified-only", action="store_true", help="Publish the manual subsets from existing manifests")
    args = parser.parse_args()
    if args.verified_only:
        _publish_verified(args.out)
        return
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"Output must be empty: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as tmp:
        temp = Path(tmp)
        photos = []
        counts = Counter()
        for name, (source, sha256, _) in SOURCES.items():
            path = (args.archives or temp) / f"{name}.zip"
            if args.archives is None:
                subprocess.run([
                    "curl", "-fLsS", "--retry", "1", "--max-time", "120", "--max-filesize", "30000000",
                    f"https://www.kaggle.com/api/v1/datasets/download/{source}", "-o", str(path),
                ], check=True)
            with path.open("rb") as file:
                if hashlib.file_digest(file, "sha256").hexdigest() != sha256:
                    raise ValueError(f"Unexpected archive hash: {path}")
            photos.extend(_read_archive(path, name, counts))
        print(json.dumps({"source_images": counts}, indent=2))
        ready = temp / "export"
        _publish(ready, photos, Counter())
        _publish_verified(ready)
        ready.replace(args.out)


if __name__ == "__main__":
    main()
