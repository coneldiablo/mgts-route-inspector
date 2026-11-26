"""Reproduce the frozen browser OOD experiment and emit aggregate measurements only."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

from scripts.cabinet_final import assign_groups, load_images, read_dss, splits
from scripts.train_browser_cabinet import local_average_vector

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = 0.6208919882774353
SEEDS = (42, 17, 23, 71, 113)
PUBLIC_SETS = {
    "parcel_lockers": (121421582, 127614311, 120625723, 84692371, 92034548, 127736182, 181225747, 78756712,
                       132375413, 110473150, 110492857, 115658680, 77062619, 115653536, 175644994, 111108485,
                       111114714, 110665410, 110471485, 135879287, 164575571),
    "rail_signal_cabinets": (48589460, 57266188, 171387746, 44450953, 67352878, 128823943, 195955377,
                             77540776, 165683362, 4044888, 74392148),
}
RAIL_CROPS = {
    48589460: (.52, 0, 1, 1), 57266188: (.29, .16, .88, .92), 44450953: (.33, .10, .96, .96),
    67352878: (.28, .02, 1, 1), 195955377: (0, .19, .55, .94), 4044888: (.30, .08, .93, .98),
}
HTTP = ("curl", "--fail", "--location", "--silent", "--show-error", "--retry", "2", "--max-time", "25",
        "--user-agent", "MGTSPrototypeResearch/1.0 (public media evaluation)")


def download_commons(folder: Path) -> dict[int, Path]:
    """Fetch public thumbnails and check the authors' stated Creative Commons licenses."""
    page_ids = [page_id for values in PUBLIC_SETS.values() for page_id in values]
    query = urlencode({"action": "query", "pageids": "|".join(map(str, page_ids)), "prop": "imageinfo",
                       "iiprop": "url|extmetadata", "iiurlwidth": 512, "format": "json"})
    response = subprocess.check_output([*HTTP, f"https://commons.wikimedia.org/w/api.php?{query}"])
    pages = json.loads(response)["query"]["pages"]

    def download(page_id: int) -> tuple[int, Path]:
        info = pages[str(page_id)]["imageinfo"][0]
        license_name = info["extmetadata"]["LicenseShortName"]["value"]
        if not license_name.startswith(("CC BY", "CC0")):
            raise ValueError(f"Unexpected Wikimedia license on page {page_id}: {license_name}")
        source = folder / f"{page_id}.download"
        url = info["thumburl"].partition("?")[0]
        subprocess.run([*HTTP, url, "--output", str(source)], check=True)
        target = folder / f"{page_id}.jpg"
        with Image.open(source) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
            image.thumbnail((640, 640))
            image.save(target, quality=95)
        source.unlink()
        return page_id, target

    with ThreadPoolExecutor(max_workers=7) as pool:
        return dict(pool.map(download, page_ids))


def make_rail_crops(commons: dict[int, Path]) -> list[Path]:
    crops = []
    for page_id, bounds in RAIL_CROPS.items():
        target = commons[page_id].with_name(f"{page_id}-crop.jpg")
        with Image.open(commons[page_id]) as source:
            width, height = source.size
            box = tuple(round(x * size) for x, size in zip(bounds, (width, height, width, height)))
            source.crop(box).save(target, quality=95)
        crops.append(target)
    return crops


def unit(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True).clip(1e-12)


def encode(session: ort.InferenceSession, files: list[Path]) -> np.ndarray:
    features = []
    for file in files:
        with Image.open(file) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
            features.append(local_average_vector(session, image))
    return unit(np.stack(features))


def centroids(features: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, list[int]]:
    means = np.stack([unit(features[groups == key].mean(axis=0)) for key in np.unique(groups)])
    labels = KMeans(n_clusters=6, random_state=42, n_init=10).fit_predict(means)
    while True:
        keys, counts = np.unique(labels, return_counts=True)
        if counts.min() >= 5 or len(keys) == 1:
            break
        smallest = keys[np.argmin(counts)]
        merged = unit(means[labels == smallest].mean(axis=0))
        remaining = [key for key in keys if key != smallest]
        prototypes = np.stack([unit(means[labels == key].mean(axis=0)) for key in remaining])
        labels[labels == smallest] = remaining[np.argmax(prototypes @ merged)]
    return np.stack([unit(means[labels == key].mean(axis=0)) for key in keys]), counts.tolist()


def similarities(features: np.ndarray, references: np.ndarray) -> np.ndarray:
    return (features @ references.T).max(axis=1)


def tally(scores: np.ndarray) -> dict:
    return {"flagged": int((scores < THRESHOLD).sum()), "total": len(scores), "min": float(scores.min())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", type=Path, required=True, help="Locally authorized DSS directories")
    parser.add_argument("--manhole-holdout", type=Path, required=True, help="Public data directory with manifest.csv")
    parser.add_argument("--onnx", type=Path, default=ROOT / "browser-ml" / "runtime" / "dino-int8.onnx")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frames = read_dss(args.sources)
    _, hashes, metadata, _ = load_images(frames)
    reviewed = json.loads((ROOT / "scripts" / "cabinet_groups.json").read_text(encoding="utf-8"))
    assign_groups(frames, hashes, metadata, reviewed)
    rows = [frame for frame in frames if frame.truth is not None]
    groups = np.array([frame.group for frame in rows])
    strata = np.array([frame.stratum for frame in rows])
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.onnx), sess_options=opts, providers=["CPUExecutionProvider"])

    with tempfile.TemporaryDirectory(prefix="cabinet-ood-public-") as directory, ThreadPoolExecutor(max_workers=1) as pool:
        public_download = pool.submit(download_commons, Path(directory))
        features = encode(session, [frame.path for frame in rows])
        centers, sizes = centroids(features, groups)
        references = {"frozen_nearest": features, "aggregated_centroids": centers}
        measurements = {name: {"id_oof": {}, "ood": {}} for name in references}
        primary_scores = {}
        for seed in SEEDS:
            scores = {name: np.empty(len(rows)) for name in references}
            for train, test in splits(strata, groups, 5, seed):
                assert not set(groups[train]) & set(groups[test])
                fold = {"frozen_nearest": features[train], "aggregated_centroids": centroids(features[train], groups[train])[0]}
                for name, prototypes in fold.items():
                    scores[name][test] = similarities(features[test], prototypes)
            for name, values in scores.items():
                measurements[name]["id_oof"][str(seed)] = tally(values)
                if seed == SEEDS[0]:
                    primary_scores[name] = values

        commons = public_download.result()
        public_cases = {key: [commons[page_id] for page_id in members] for key, members in PUBLIC_SETS.items()}
        public_cases["rail_framing_control"] = make_rail_crops(commons)
        with (args.manhole_holdout / "manifest.csv").open(encoding="utf-8", newline="") as manifest:
            public_cases["manhole_positives"] = [args.manhole_holdout / row["image"] for row in csv.DictReader(manifest)
                                                  if row["source_label"] == "manhole"]
        for case, files in public_cases.items():
            public_features = encode(session, files)
            for name, prototypes in references.items():
                scores = similarities(public_features, prototypes)
                auc = roc_auc_score(np.r_[np.ones(len(rows)), np.zeros(len(files))], np.r_[primary_scores[name], scores])
                measurements[name]["ood"][case] = {**tally(scores), "max": float(scores.max()), "auroc": float(auc)}

    output = {"status": "research_not_deployed", "threshold": THRESHOLD, "measurements": measurements,
              "cluster_sizes": sizes}
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
