"""Bootstrap the verified public backbones and promote the evaluated local release."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import ssl
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

from inspector.cabinet import DEFAULT_CABINET_MODEL, ROI_RELEASE, SIGLIP_MODEL, SIGLIP_REVISION, SIGLIP_SHA256
from inspector.cabinet_roi import MODEL as GROUNDING_MODEL
from inspector.cabinet_roi import REVISION as GROUNDING_REVISION
from inspector.cabinet_roi import WEIGHTS_SHA256 as GROUNDING_SHA256
from inspector.features import DINOV2_MODEL, DINOV2_REVISION, DINOV2_SHA256
from inspector.manifest import image_sha256

GROUNDING_FILES = {
    "added_tokens.json": "909e96cb32d92ce728a01bc99850cbba26196d74115c17ebeb019275412588f2",
    "config.json": "eec82c5ab66e16df12a9a212e68ac011779927c2536cf9078658e35d85f0c67a",
    "model.safetensors": GROUNDING_SHA256,
    "preprocessor_config.json": "8454179ba95e2ad22947835aad7b45862a601fc0055ab88bf1ee70892d3aea60",
    "special_tokens_map.json": "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
    "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    "tokenizer_config.json": "d40ab645b68211910b9170d22433d43186a6ec8ee6fd10ba170524b25bf4fb56",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}
MOBILENET_FILE = "mobilenet_v3_small-047dcff4.pth"
MOBILENET_SHA256 = "047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f"
CSV_FIELDS = ("image_id", "object_type", "object_type_score", "mechanical", "coating", "action", "reason",
              "object_type_source")


def fetch_verified(url: str, target: Path, expected: str, *, verify_only: bool = False) -> None:
    if target.is_file() and image_sha256(target) == expected:
        print(f"Проверено: {target}", flush=True)
        return
    if verify_only:
        raise ValueError(f"Файл отсутствует или не совпала SHA-256: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Загрузка {target.name} → {target.parent}", flush=True)
    digest = hashlib.sha256()
    with NamedTemporaryFile(dir=target.parent, prefix=target.name + ".", delete=False) as output:
        temporary = Path(output.name)
    try:
        tls = ssl.create_default_context()
        try:
            import certifi
        except ImportError:
            pass
        else:
            tls.load_verify_locations(cafile=certifi.where())
        with urlopen(url, timeout=45, context=tls) as source, temporary.open("wb") as output:
            while chunk := source.read(1 << 20):
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"Не совпала SHA-256 загруженного файла: {target.name}")
        os.replace(temporary, target)
        print(f"Проверено: {target}", flush=True)
    finally:
        temporary.unlink(missing_ok=True)


def _hf_files(model, revision, files, directory, verify_only):
    hub = f"https://huggingface.co/{model}/resolve/{revision}"
    for name, digest in files.items():
        fetch_verified(f"{hub}/{name}", directory / name, digest, verify_only=verify_only)


def prepare_weights(*, detector_only: bool = False, without_hr: bool = False,
                    verify_only: bool = False) -> None:
    grounder = Path(os.getenv("INSPECTOR_CABINET_ROI_DIR") or "artifacts/cabinet-roi-grounding/detector").expanduser()
    _hf_files(GROUNDING_MODEL, GROUNDING_REVISION, GROUNDING_FILES, grounder, verify_only)
    if detector_only:
        return
    dino = Path(os.getenv("INSPECTOR_DINOV2_DIR") or "artifacts/dinov2-small").expanduser()
    siglip = Path(os.getenv("INSPECTOR_SIGLIP_DIR") or "artifacts/final-backbones/siglip2").expanduser()
    _hf_files(DINOV2_MODEL, DINOV2_REVISION, DINOV2_SHA256, dino, verify_only)
    _hf_files(SIGLIP_MODEL, SIGLIP_REVISION, {"model.safetensors": SIGLIP_SHA256}, siglip, verify_only)
    import torch

    cache = Path(torch.hub.get_dir()) / "checkpoints"
    fetch_verified(f"https://download.pytorch.org/models/{MOBILENET_FILE}", cache / MOBILENET_FILE,
                   MOBILENET_SHA256, verify_only=verify_only)
    if not without_hr:
        from inspector.cabinet_hr import MODEL, REVISION, SHA256

        directory = Path(os.getenv("INSPECTOR_SIGLIP_HR_DIR") or "artifacts/cabinet-hr-siglip").expanduser()
        _hf_files(MODEL, REVISION, {"model.safetensors": SHA256}, directory, verify_only)


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _publish_json(path: Path, data) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _same_ids(before, after):
    return [r["image_id"] for r in before] == [r["image_id"] for r in after]


def promote_roi(control: Path, revision: Path, model: Path, out: Path, target: Path = DEFAULT_CABINET_MODEL) -> None:
    import joblib

    artifact = joblib.load(model)
    if "type_roi" not in artifact:
        raise ValueError("Артефакт не содержит проверенной головы ROI.")
    report = _read(control / "public_metrics.json")
    paired = _read(revision / "comparison.json")
    if paired["selected"] != "type_roi":
        raise ValueError("Парное сравнение не разрешило переключение на ROI.")
    primary = paired["metrics"]["42"]
    old_oof = _read(control / "oof_predictions.json")
    new_oof = _read(revision / "oof_predictions.json")
    old_predictions = _read(control / "predictions.json")
    new_predictions = _read(revision / "predictions.json")
    keys = ("scores", "action")
    if not _same_ids(old_oof, new_oof) or not _same_ids(old_predictions, new_predictions):
        raise ValueError("Идентификаторы контроля и кандидата не совпадают.")
    if any(old[key] != new[key] for before, after in [(old_oof, new_oof), (old_predictions, new_predictions)]
           for old, new in zip(before, after, strict=True) for key in keys):
        raise ValueError("ROI изменил вероятности дефектов или политику действий.")
    rows = [old | new | {"model_release": ROI_RELEASE}
            for old, new in zip(old_predictions, new_predictions, strict=True)]
    new_oof = [row | {"model_release": ROI_RELEASE} for row in new_oof]
    score = primary["candidate"]
    report["model"].update(
        name="DINOv2 × SigLIP 2 · тип по области шкафа", release=ROI_RELEASE,
        validation="5 внешних групповых OOF-фолдов; фиксированная голова ROI только для типа",
        type_classifier={
            "method": "grounded_roi_or_full_frame", "control_correct": primary["control_correct"],
            "candidate_correct": primary["candidate_correct"], "grounded_labeled": paired["grounded_labeled"],
            "grounded_total": paired["grounded_total"], "fallback": "v0.2 full_frame",
        },
    )
    report["model"]["signalMetrics"] = [
        row | score[row["code"]] for row in report["model"]["signalMetrics"]
    ]
    report["metrics"] = score
    report["policy_oof"] = primary["oof_actions"]
    report["prediction_counts"] = dict(Counter(row["action"] for row in rows))
    report["type_source_counts"] = dict(Counter(row["object_type_source"] for row in rows))

    out.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    _publish_json(out / "predictions.json", rows)
    _publish_json(out / "oof_predictions.json", new_oof)
    temporary_queue = out / "inspection_queue.csv.tmp"
    with temporary_queue.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row["scores"].get(field, row.get(field)) for field in CSV_FIELDS})
    os.replace(temporary_queue, out / "inspection_queue.csv")
    temporary_model = target.with_name(target.name + ".tmp")
    shutil.copyfile(model, temporary_model)
    os.replace(temporary_model, target)
    _publish_json(out / "public_metrics.json", report)
    print(f"Активная модель {ROI_RELEASE}: {target}", flush=True)
