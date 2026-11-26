"""Repeat the inference-only checks on the public inspector artifacts.

Run from the project root: python -m scripts.audit_frozen_model --output FILE.csv
The CSV contains a row per model, split, photograph and viewing condition.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from scipy.signal import correlate

from inspector.classifier import classification_image, load_model
from inspector.config import QualityConfig
from inspector.features import TorchEmbedder
from inspector.manifest import data_fingerprint, image_sha256, load_manifest
from inspector.quality import assess_quality, open_image


ARTIFACT = Path("artifacts/public")
CACHE = Path("artifacts/torch-cache")
CHECKPOINT = CACHE / "hub/checkpoints/mobilenet_v3_small-047dcff4.pth"


def archive_families(split):
    archive = Path("data/_downloads/manhole_covers") / ("test.zip" if split == "real_official_test" else "train.zip")
    families = {}
    with ZipFile(archive) as z:
        for name in z.namelist():
            path = Path(name)
            if path.suffix.lower() == ".jpg":
                family = "void" if path.parent.name == "void" else path.stem.split(" (")[0]
                families[hashlib.sha256(z.read(name)).hexdigest()] = family
    return families


def jpeg75(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=75, subsampling=2)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


CONDITIONS = {
    "original": lambda im: im,
    "mirror": ImageOps.mirror,
    "jpeg75": jpeg75,
    "brightness70": lambda im: ImageEnhance.Brightness(im).enhance(0.70),
    "blur1": lambda im: im.filter(ImageFilter.GaussianBlur(radius=1)),
    "resize160": lambda im: im.resize((160, 160), Image.Resampling.BILINEAR),
}


def audit_condition(records, images, families, name, model, condition):
    cfg = QualityConfig(**model.meta["quality_config"])
    rows = []
    for start in range(0, len(records), 16):
        batch = records[start:start + 16]
        views = [CONDITIONS[condition](im) for im in images[start:start + 16]]
        quality = [assess_quality(im, r.bbox, cfg) for r, im in zip(batch, views, strict=True)]
        inputs = [classification_image(im, r, model.meta["crop_mode"])
                  for im, r, q in zip(views, batch, quality, strict=True) if q.ok]
        predictions = model.predict_images(inputs) if inputs else []
        assert len(predictions) == len(inputs)
        predictions = iter(predictions)
        for r, q in zip(batch, quality, strict=True):
            score = next(predictions).p_finding if q.ok else None
            if score is None:
                status = "rephoto"
            elif score >= model.threshold:
                status = "finding"
            elif score >= max(0, model.threshold - cfg.uncertain_band):
                status = "uncertain"
            else:
                status = "normal"
            rows.append(dict(model=name, condition=condition, image_id=r.image_id,
                             archive_family=families[r.image_id], label=r.label, threshold=model.threshold,
                             score=score, status=status, sharpness=q.sharpness, contrast=q.contrast,
                             brightness=q.brightness, refusal="|".join(q.reasons)))
    return rows


def scene_features(encoder, records):
    vectors, grey = [], []
    for start in range(0, len(records), 32):
        batch = [open_image(r.image_path) for r in records[start:start + 32]]
        vectors.append(encoder.embed(batch))
        grey.extend(np.asarray(im.convert("L").resize((64, 64), Image.Resampling.LANCZOS), dtype=float)
                    for im in batch)
    vectors = np.concatenate(vectors)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-10)
    return vectors, grey


def patch_norms(image, side):
    # Norm of each mean-centred square patch, via sums over rectangular windows.
    def sums(values):
        table = np.pad(values.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        return table[side:, side:] - table[:-side, side:] - table[side:, :-side] + table[:-side, :-side]

    return np.sqrt(np.maximum(sums(image * image) - sums(image) ** 2 / side ** 2, 0))


def find_scene_candidates(train, checks):
    encoder = TorchEmbedder(device="cpu")
    train_vectors, train_grey = scene_features(encoder, train)
    norms = [patch_norms(image, 48) for image in train_grey]
    pairs = []
    for split, records in checks.items():
        vectors, grey = scene_features(encoder, records)
        similarities = vectors @ train_vectors.T
        for i, record in enumerate(records):
            centre = grey[i][8:-8, 8:-8]
            centre = centre - centre.mean()
            if centre.std() < 8:
                continue
            for j in np.argsort(similarities[i])[-5:]:
                dots = correlate(train_grey[j], centre, mode="valid", method="fft")
                denom = np.maximum(norms[j] * np.linalg.norm(centre), 1e-8)
                scores = np.where(norms[j] / 48 >= 8, dots / denom, 0)
                y, x = np.unravel_index(scores.argmax(), scores.shape)
                if scores[y, x] >= .95:
                    pairs.append(dict(split=split, query=record.image_id, train=train[j].image_id,
                                      corr=scores[y, x], dx=int(x) - 8, dy=int(y) - 8,
                                      similarity=similarities[i, j]))
    columns = ["split", "query", "train", "corr", "dx", "dy", "similarity"]
    return pd.DataFrame(pairs, columns=columns).sort_values("corr", ascending=False)


def main():
    parser = argparse.ArgumentParser(description="Аудит сохранённой публичной модели при фиксированном пороге")
    parser.add_argument("--output", type=Path, required=True, help="новый CSV для результатов аудита")
    parser.add_argument("--pairs", type=Path, help="новый CSV для пар визуально похожих сцен")
    args = parser.parse_args()
    outputs = [args.output, args.pairs] if args.pairs else [args.output]
    if len({p.resolve() for p in outputs}) != len(outputs):
        parser.error("Для разных результатов нужны разные имена")
    if any(p.suffix.lower() != ".csv" or p.exists() for p in outputs):
        parser.error("Выберите новое имя файла CSV")
    if not CHECKPOINT.is_file() or image_sha256(CHECKPOINT) != "047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f":
        parser.error("Локальные веса MobileNetV3 не совпали с проверенной версией")
    os.environ["TORCH_HOME"] = str(CACHE.resolve())
    monitored = [CHECKPOINT, *sorted(p for p in ARTIFACT.rglob("*") if p.is_file())]
    signatures = {p: image_sha256(p) for p in monitored}
    models = {name: load_model(path) for name, path in (("hybrid", ARTIFACT), ("baseline", ARTIFACT / "baseline"))}
    train = load_manifest("data/real")
    fingerprint = data_fingerprint(train)
    for name, model in models.items():
        path = ARTIFACT if name == "hybrid" else ARTIFACT / "baseline"
        assert model.mode == "supervised" and model.meta == json.loads((path / "meta.json").read_text())
        assert model.meta["data_fingerprint"] == fingerprint and model.meta["threshold"] == model.threshold
    seen_hashes = {image_sha256(r.image_path) for r in train}
    seen_ids = {r.asset_id for r in train}
    rows, checks = [], {}
    for split in ("real_holdout", "real_official_test"):
        records = load_manifest(Path("data") / split)
        checks[split] = records
        archive = archive_families(split)
        hashes = {r.image_id: image_sha256(r.image_path) for r in records}
        ids = {r.asset_id for r in records}
        assert not seen_ids.intersection(ids) and not seen_hashes.intersection(hashes.values())
        seen_ids.update(ids)
        seen_hashes.update(hashes.values())
        families = {k: archive[v] for k, v in hashes.items()}
        images = [open_image(r.image_path) for r in records]
        for name, model in models.items():
            conditions = CONDITIONS if name == "hybrid" else ("original",)
            for condition in conditions:
                observations = audit_condition(records, images, families, name, model, condition)
                rows.extend(dict(split=split, **row) for row in observations)
                print(split, name, condition, len(observations), flush=True)
    pairs = find_scene_candidates(train, checks) if args.pairs else None
    assert signatures == {p: image_sha256(p) for p in monitored}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False, mode="x")
    if pairs is not None:
        args.pairs.parent.mkdir(parents=True, exist_ok=True)
        pairs.to_csv(args.pairs, index=False, mode="x")


if __name__ == "__main__":
    main()
