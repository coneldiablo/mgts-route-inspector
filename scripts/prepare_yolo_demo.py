"""Prepare an inference pack from the curated public holdouts."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from inspector.manifest import load_manifest


def restore_distractors(directory: Path, public_holdout: Path) -> None:
    with (directory / "manifest.csv").open(encoding="utf-8", newline="") as manifest:
        filenames = [Path(row["image"]).name for row in csv.DictReader(manifest)]
    images = directory / "images"
    images.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        shutil.copyfile(public_holdout / "images" / name, images / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verified", type=Path, default=Path("data/ml-bench/verified"))
    parser.add_argument("--out", type=Path, default=Path("data/ml-demo-raw"))
    parser.add_argument("--restore-distractors", type=Path, metavar="MANIFEST_DIR")
    parser.add_argument("--public-holdout", type=Path, default=Path("data/real_holdout"))
    args = parser.parse_args()
    if args.restore_distractors is not None:
        restore_distractors(args.restore_distractors, args.public_holdout)
        return
    cohorts = {
        name: load_manifest(args.verified / name / "holdout") for name in ("cube", "deep")
    }
    image_dir = args.out / "images"
    image_dir.mkdir(parents=True, exist_ok=False)
    fields = ["image", "image_id", "asset_id", "id_origin", "label", "object_type", "source"]
    with (args.out / "manifest.csv").open("x", encoding="utf-8", newline="") as manifest:
        writer = csv.DictWriter(manifest, fieldnames=fields)
        writer.writeheader()
        for name, records in cohorts.items():
            for record in records:
                target = image_dir / record.image_path.name
                shutil.copyfile(record.image_path, target)
                writer.writerow({
                    "image": target.relative_to(args.out).as_posix(),
                    "image_id": record.image_id,
                    "asset_id": record.asset_id,
                    "id_origin": "image",
                    "label": record.label,
                    "object_type": "well",
                    "source": f"public:verified/{name}",
                })


if __name__ == "__main__":
    main()
