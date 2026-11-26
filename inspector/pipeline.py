"""End-to-end run: images + manifest -> observations -> unique object cards."""
from __future__ import annotations

import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .classifier import InspectorModel, Prediction, classification_image, load_model
from .config import AppConfig, QualityConfig, quality_config, PUBLIC_REVIEW_LABEL
from .location import (
    DEFAULT_LOCATION_RADIUS_M,
    Location,
    haversine_m,
    load_registry,
    resolve_location,
    validate_location_radius,
)
from .manifest import Record, data_fingerprint, image_sha256, load_manifest
from .quality import Quality, assess_quality, open_image, save_jpeg

if TYPE_CHECKING:
    from PIL import Image

    from .detector import Locator
    from .vertex_vision import VertexVision

STATUS_LABELS = {
    "finding": "Требует проверки",
    "uncertain": "Слабый сигнал",
    "rephoto": "Нужен другой кадр",
    "normal": "Без замечания",
}


@dataclass
class Observation:
    record: Record
    quality: Quality
    prediction: Prediction
    status: str
    sign: str
    image_file: str | None = None
    crop_file: str | None = None
    image_digest: str | None = None
    location: Location | None = None
    localization: dict[str, Any] | None = None
    decision_threshold: float | None = None
    cloud_review: dict[str, Any] | None = None


def _safe_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text)[:120] or "image"


def _status_for(p: float, threshold: float, quality: Quality, band: float, *, defect_only: bool = False) -> str:
    if not quality.ok:
        return "rephoto"
    if p >= threshold:
        return "finding"
    if defect_only or p >= max(0.0, threshold - band):
        return "uncertain"
    return "normal"


def _review_observation(
    obs: Observation, image: Image.Image, reply: dict[str, Any], reviewer_name: str,
    model: InspectorModel, cfg: AppConfig, threshold: float, crop_mode: str,
) -> Observation:
    cloud = {
        "model": reviewer_name,
        **{key: reply[key] for key in ("status", "seconds", "prediction", "evidence", "error") if key in reply},
        "confirmation": "unavailable" if reply.get("error") else "advisory",
    }
    obs.cloud_review = cloud
    bbox = reply.get("bbox")
    if bbox is None:
        return obs
    candidate = replace(obs.record, bbox=bbox)
    quality = assess_quality(image, bbox, cfg.quality)
    if not quality.ok:
        return obs
    prediction = model.predict_images([classification_image(image, candidate, crop_mode)])[0]
    status = _status_for(prediction.p_finding, threshold, quality, cfg.quality.uncertain_band)
    crop_label = prediction.sign if status == "finding" else status
    original_label = obs.sign if obs.status == "finding" else obs.status
    if crop_label != reply["prediction"] or original_label not in {"uncertain", crop_label}:
        cloud["confirmation"] = "local_disagreement"
        return obs
    cloud["confirmation"] = "confirmed_bbox"
    return replace(obs, record=candidate, quality=quality, prediction=prediction, status=status,
                   sign=crop_label, decision_threshold=None,
                   localization={"source": "vertex_confirmed", "bbox": list(bbox), "box_count": 1})


def predict_records(
    records: list[Record],
    model: InspectorModel,
    cfg: AppConfig,
    threshold: float,
    crop_mode: str = "auto",
    verbose: bool = True,
    locator: Locator | None = None,
    single_target: bool = False,
    full_frame_model: InspectorModel | None = None,
    cloud_reviewer: VertexVision | None = None,
    cloud_limit: int = 8,
) -> list[Observation]:
    observations: list[Observation] = []
    full_quality = cfg.quality
    cloud_remaining = cloud_limit
    if cloud_reviewer is not None and (
        locator is None or not single_target or full_frame_model is None or crop_mode == "full"
        or model.mode != "supervised" or set(model.classes) != {"normal", "open_hatch"}
        or cloud_limit < 1
    ):
        raise ValueError("Для облачной перепроверки нужны локализатор, модель кропов люка, "
                         "резервная модель полного кадра, режим одного объекта и положительный лимит.")
    if full_frame_model is not None:
        if locator is None or not single_target:
            raise ValueError("Модель полного кадра требует локализатор и режим одного целевого люка.")
        if (full_frame_model.meta.get("crop_mode") != "full" or full_frame_model.mode != model.mode
                or set(full_frame_model.classes) != set(model.classes) or full_frame_model.normal_label != model.normal_label):
            raise ValueError("Запасную модель нужно обучить на полных кадрах с теми же классами и задачей.")
        stored = full_frame_model.meta.get("quality_config")
        if stored:
            full_quality = QualityConfig(**stored)
    started = time.monotonic()
    for start in range(0, len(records), 16):
        images = []
        full_images = []
        batch = records[start:start + 16]
        opened = []
        for record in batch:
            try:
                opened.append(open_image(record.image_path))
            except OSError as exc:
                raise ValueError(f"Не удалось прочитать изображение {record.image_id}: {record.image_path}") from exc
        detections = iter(locator.locate_batch([
            image for record, image in zip(batch, opened, strict=True) if record.bbox is None
        ])) if locator is not None else None
        prepared = []
        for record, image in zip(batch, opened, strict=True):
            localization = None
            use_full = False
            if detections is not None:
                if record.bbox is not None:
                    localization = {"source": "provided"}
                else:
                    boxes = next(detections)
                    if boxes and (len(boxes) == 1 or single_target):
                        bbox, score = boxes[0]
                        record = replace(record, bbox=bbox)
                        localization = {"source": "detected", "score": round(score, 4), "bbox": list(bbox),
                                        "box_count": len(boxes)}
                    else:
                        localization = {"source": "ambiguous" if boxes else "not_detected", "box_count": len(boxes)}
                        if not boxes and full_frame_model is not None:
                            use_full = True
                            localization["source"] = "full_frame"
            quality = assess_quality(image, record.bbox, full_quality if use_full else cfg.quality)
            if detections is not None and record.bbox is None and not use_full:
                reason = ("несколько люков без однозначной привязки" if localization["box_count"] else
                          "локализатор не нашёл люк")
                quality = replace(quality, ok=False, score=0.0, reasons=[*quality.reasons, reason])
            prepared.append((record, localization, quality, use_full))
            if quality.ok:
                if use_full:
                    full_images.append(image)
                else:
                    images.append(classification_image(image, record, crop_mode))
        predictions = iter(model.predict_images(images))
        full_predictions = iter(full_frame_model.predict_images(full_images)) if full_images else iter(())
        review_queue = []
        for (record, localization, quality, use_full), image in zip(prepared, opened, strict=True):
            prediction = (next(full_predictions if use_full else predictions) if quality.ok else
                          Prediction(0, 0, {}, "unusable", "unusable"))
            effective_threshold = full_frame_model.threshold if use_full else threshold
            band = full_quality.uncertain_band if use_full else cfg.quality.uncertain_band
            status = _status_for(prediction.p_finding, effective_threshold, quality, band,
                                 defect_only=model.mode == "defect_type")
            sign = "normal" if status == "normal" else prediction.sign
            observation = Observation(record, quality, prediction, status, sign, localization=localization,
                                      decision_threshold=effective_threshold if use_full else None)
            if (cloud_reviewer is not None and cloud_remaining and quality.ok
                    and (use_full or status == "uncertain")):
                review_queue.append((len(observations), image))
                cloud_remaining -= 1
            observations.append(observation)
        if cloud_reviewer is not None and review_queue:
            with ThreadPoolExecutor(max_workers=min(3, len(review_queue))) as pool:
                futures = []
                for index, image in review_queue:
                    record = observations[index].record
                    full_frame = record.bbox is None
                    source = image if full_frame else classification_image(image, record, crop_mode)
                    futures.append(pool.submit(cloud_reviewer.inspect, source, full_frame=full_frame))
                for (index, image), future in zip(review_queue, futures, strict=True):
                    observations[index] = _review_observation(
                        observations[index], image, future.result(), cloud_reviewer.model, model, cfg, threshold, crop_mode,
                    )
    if verbose:
        print(f"Проверка качества и инференс: {len(observations)} кадров, {time.monotonic() - started:.1f} с")
    return observations


def _bbox_pct(bbox, width: int, height: int) -> list[float] | None:
    if not bbox:
        return None
    x1, y1, x2, y2 = bbox
    return [
        round(max(0.0, x1 / width) * 100, 2),
        round(max(0.0, y1 / height) * 100, 2),
        round(min(1.0, x2 / width) * 100, 2),
        round(min(1.0, y2 / height) * 100, 2),
    ]


def _why(obs: Observation, cfg: AppConfig) -> str:
    q = obs.quality
    parts = ([f"оценка модели {obs.prediction.p_finding:.2f}", "кадр пригоден для оценки"] if q.ok
             else ["кадр непригоден: " + ", ".join(q.reasons)])
    if q.area_fraction is not None:
        parts.append(f"объект занимает {q.area_fraction:.0%} кадра")
    if obs.prediction.distance is not None:
        parts.append(f"отклонение от нормы {obs.prediction.distance:.3f}")
    if obs.localization and obs.localization["source"] == "detected":
        parts.append(f"рамку построил детектор, оценка {obs.localization['score']:.2f}")
    if obs.localization and obs.localization["source"] == "full_frame":
        parts.append("состояние оценено по всему кадру, рамка отсутствует")
    if obs.localization and obs.localization["source"] == "vertex_confirmed":
        parts.append("рамку предложил Gemini, состояние подтверждено локальной моделью по кропу")
    return "; ".join(parts)


def _frame(obs: Observation, cfg: AppConfig) -> dict[str, Any]:
    action = cfg.resolve_sign(obs.sign)["action"] if obs.status == "finding" else None
    return {
        "image_id": obs.record.image_id,
        "captured_at": obs.record.captured_at,
        "p_finding": round(obs.prediction.p_finding, 4) if obs.quality.ok else None,
        "pred_label": obs.prediction.pred_label,
        "sign": obs.sign,
        "status": obs.status,
        "status_label": STATUS_LABELS.get(obs.status, obs.status),
        "quality_ok": obs.quality.ok,
        "quality_score": obs.quality.score,
        "quality_reasons": obs.quality.reasons,
        "condition": obs.record.condition,
        "image_file": obs.image_file,
        "crop_file": obs.crop_file,
        "action": action,
        **({"localization": obs.localization} if obs.localization else {}),
        **({"cloud_review": obs.cloud_review} if obs.cloud_review else {}),
        "location": _observation_location(obs).payload(),
    }


def _observation_location(obs: Observation) -> Location:
    return obs.location if obs.location is not None else resolve_location(obs.record)


def _card_location(best: Observation, history: list[Observation], radius_m: float) -> tuple[Location, str]:
    selected = _observation_location(best)
    if best.record.asset_id_origin != "provided":
        return selected, best.record.image_id

    # A second angle may supply a point for the same ID. A conflicting point or
    # type needs review before any location for that asset is distributed.
    located = [(o, _observation_location(o)) for o in history]
    for obs, location in located:
        if location.status == "ambiguous":
            return location, obs.record.image_id

    locations_with_points = [(o, location) for o, location in located if location.point is not None]
    if not locations_with_points:
        return selected, best.record.image_id

    anchor = next(
        ((o, location) for o, location in locations_with_points if o is best),
        locations_with_points[0],
    )
    obs, location = anchor
    assert location.point is not None
    for other, known in locations_with_points:
        if known.point is not None and haversine_m(location.point, known.point) > radius_m:
            return Location("ambiguous", "object_coordinates", camera_point=selected.camera_point,
                            reason="object_coordinates_conflict"), other.record.image_id
    return replace(location, camera_point=selected.camera_point), obs.record.image_id


def build_cards(
    observations: list[Observation],
    cfg: AppConfig,
    threshold: float,
    include_normal: bool | None = None,
    run_id: str = "run",
    location_radius_m: float = DEFAULT_LOCATION_RADIUS_M,
) -> list[dict[str, Any]]:
    include_normal = cfg.include_normal_cards if include_normal is None else include_normal
    by_asset: dict[tuple[str, str], list[Observation]] = {}
    for obs in observations:
        by_asset.setdefault(obs.record.asset_key, []).append(obs)

    cards: list[dict[str, Any]] = []
    for obs_list in by_asset.values():
        findings = [o for o in obs_list if o.status in {"finding", "uncertain"}]
        if findings:
            groups: dict[str, list[Observation]] = {}
            for obs in findings:
                groups.setdefault(obs.sign, []).append(obs)
            for sign, group in groups.items():
                best = max(group, key=lambda o: (o.prediction.p_finding, o.quality.score))
                cards.append(_make_card(best, group, obs_list, cfg, threshold, run_id, "finding_group", location_radius_m))
            continue
        normal = [o for o in obs_list if o.status == "normal"]
        if normal:
            if include_normal:
                best = max(normal, key=lambda o: (o.quality.score, o.prediction.p_normal))
                cards.append(_make_card(best, normal, obs_list, cfg, threshold, run_id, "normal", location_radius_m))
            continue
        best = max(obs_list, key=lambda o: o.quality.score)
        cards.append(_make_card(best, obs_list, obs_list, cfg, threshold, run_id, "rephoto", location_radius_m))

    def sort_key(card: dict[str, Any]):
        order = {"finding": 0, "uncertain": 1, "rephoto": 2, "normal": 3}
        return (order.get(card["status"], 9), -card["priority"], card["asset_id"])

    cards.sort(key=sort_key)
    for i, card in enumerate(cards, start=1):
        card["queue_position"] = i
    return cards


def _make_card(
    best: Observation,
    group: list[Observation],
    all_obs: list[Observation],
    cfg: AppConfig,
    threshold: float,
    run_id: str,
    kind: str,
    location_radius_m: float,
) -> dict[str, Any]:
    sign_info = cfg.resolve_sign(best.sign)
    status = "rephoto" if kind == "rephoto" else ("normal" if kind == "normal" else best.status)
    if kind == "rephoto":
        action = "Запросить повторный кадр: " + ", ".join(best.quality.reasons) + "."
        title = "Недостаточно данных для оценки"
        short = "Нужен другой кадр"
    elif kind == "normal":
        action = "Плановый осмотр; действий по проверяемому признаку не требуется."
        title = "Признак не выявлен"
        short = "Без замечания"
    elif best.status == "uncertain":
        action = (
            "Сигнал слабее порога. Проверить вручную или запросить кадр крупнее; "
            "автоматическое замечание не создаётся без подтверждения."
        )
        title = sign_info["title"]
        short = sign_info["short"]
    else:
        action = sign_info["action"]
        title = sign_info["title"]
        short = sign_info["short"]

    if kind == "normal" and best.prediction.sign == PUBLIC_REVIEW_LABEL:
        title = "Модель распознала крышку на снимке"
        short = "Крышка в кадре"
        action = "Кадр можно передать мастеру для оценки состояния крышки."
    candidate_sign = best.prediction.sign
    candidate_info = cfg.resolve_sign(candidate_sign)
    severity = float(sign_info.get("severity", 0.5))
    priority = best.prediction.p_finding * severity + 0.1 * best.record.criticality
    if kind == "rephoto":
        priority = 0.35
    elif kind == "normal":
        priority = 0.0
    elif best.status == "uncertain":
        priority *= 0.7

    # Every visit of the asset is kept as history; the best usable frame is the basis.
    history = sorted(all_obs, key=lambda o: (o.record.captured_at or "", o.record.image_id))
    frames = [_frame(o, cfg) for o in history]
    location, evidence_frame_id = _card_location(best, history, location_radius_m)
    location_payload = location.payload()
    if evidence_frame_id != best.record.image_id:
        location_payload["evidence_image_id"] = evidence_frame_id
    duplicates = max(0, len([o for o in group if o.status in {"finding", "uncertain"}]) - 1)
    image_id = best.record.image_id
    card_seed = f"{run_id}|{best.record.asset_id}|{best.sign}|{kind}"
    if best.record.extra.get("id_origin") == "image":
        card_seed = json.dumps([run_id, image_id, best.record.asset_id_origin, best.sign, kind])
    card_id = hashlib.sha1(card_seed.encode()).hexdigest()[:12]
    return {
        "card_id": card_id,
        "run_id": run_id,
        "kind": kind,
        "status": status,
        "status_label": STATUS_LABELS.get(status, status),
        "asset_id": best.record.asset_id,
        "object_type": best.record.object_type,
        "lat": location.lat,
        "lon": location.lon,
        "coordinates": location.coordinates,
        "location": location_payload,
        "captured_at": best.record.captured_at,
        "finding_type": best.sign if status != "normal" else "normal",
        "finding_short": short,
        "finding_title": title,
        "action": action,
        "candidate_sign": candidate_sign,
        "candidate_short": candidate_info.get("short", candidate_sign),
        "candidate_title": candidate_info.get("title", candidate_sign),
        "candidate_action": candidate_info.get("action"),
        "candidate_severity": float(candidate_info.get("severity", 0.5)),
        "priority": round(float(priority), 4),
        "p_finding": round(best.prediction.p_finding, 4) if best.quality.ok else None,
        "threshold": best.decision_threshold if best.decision_threshold is not None else threshold,
        "p_normal": round(best.prediction.p_normal, 4) if best.quality.ok else None,
        "model_score_note": "Оценка модели, не вероятность аварии",
        "quality": best.quality.to_dict(),
        "why": _why(best, cfg),
        "image_id": image_id,
        "image_file": best.image_file,
        "crop_file": best.crop_file,
        "bbox_pct": _bbox_pct(best.record.bbox, best.quality.width, best.quality.height),
        **({"localization": best.localization} if best.localization else {}),
        **({"cloud_review": best.cloud_review} if best.cloud_review else {}),
        "frames_count": len(frames),
        "duplicates_merged": duplicates,
        "frames": frames,
        "review_status": "new",
        "review_note": "",
        "model_version": None,
        "condition": best.record.condition,
        "criticality": best.record.criticality,
    }


def _copy_media(observations: list[Observation], out_dir: Path, max_side: int = 1400, crop_mode: str = "auto") -> None:
    images_dir = out_dir / "images"
    crops_dir = out_dir / "crops"
    for d in (images_dir, crops_dir):
        d.mkdir(parents=True, exist_ok=True)
    for obs in observations:
        record = obs.record
        obs.image_digest = image_sha256(record.image_path)
        img = open_image(record.image_path)
        stem = _safe_name(record.image_id) + "-" + hashlib.sha256(record.image_id.encode()).hexdigest()[:12]
        fname = stem + ".jpg"
        save_jpeg(img, images_dir / fname, max_side=max_side)
        obs.image_file = f"images/{fname}"
        crop = classification_image(img, record, crop_mode)
        crop_fname = stem + "__crop.jpg"
        save_jpeg(crop, crops_dir / crop_fname, max_side=900)
        obs.crop_file = f"crops/{crop_fname}"


def analyze(
    data_dir: str | Path,
    artifact_dir: str | Path,
    out_dir: str | Path,
    cfg: AppConfig | None = None,
    threshold: float | None = None,
    crop_mode: str | None = None,
    include_normal: bool | None = True,
    quality_profile: str | None = None,
    verbose: bool = True,
    registry_path: str | Path | None = None,
    location_radius_m: float = DEFAULT_LOCATION_RADIUS_M,
    locator_path: str | Path | None = None,
    single_target: bool = False,
    full_frame_artifact: str | Path | None = None,
    cloud_review_model: str | None = None,
    cloud_limit: int = 8,
) -> dict[str, Any]:
    validate_location_radius(location_radius_m)
    registry = load_registry(registry_path) if registry_path is not None else None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(artifact_dir)
    full_frame_model = load_model(full_frame_artifact) if full_frame_artifact is not None else None
    locator = None
    if locator_path is not None:
        from .detector import load_locator

        locator = load_locator(locator_path)
    threshold = float(threshold if threshold is not None else model.threshold)
    crop_mode = crop_mode or model.meta.get("crop_mode", "auto")
    profile = quality_profile or model.meta.get("quality_profile", "default")
    if cfg is None:
        stored = model.meta.get("quality_config")
        cfg = AppConfig(quality=QualityConfig(**stored) if stored and quality_profile is None else quality_config(profile))
    elif quality_profile is not None:
        cfg = replace(cfg, quality=quality_config(quality_profile))
    records = load_manifest(data_dir)
    if any(out_dir.resolve() in record.image_path.parents for record in records):
        raise ValueError("Каталог вывода содержит исходные изображения. Выберите отдельный каталог.")
    t0 = time.time()
    cloud_reviewer = None
    if cloud_review_model is not None:
        from .vertex_vision import VertexVision

        cloud_reviewer = VertexVision(cloud_review_model)
    observations = predict_records(records, model, cfg, threshold, crop_mode, verbose=verbose,
                                   locator=locator, single_target=single_target, full_frame_model=full_frame_model,
                                   cloud_reviewer=cloud_reviewer, cloud_limit=cloud_limit)
    for obs in observations:
        obs.location = resolve_location(obs.record, registry, location_radius_m)
    if verbose:
        print(f"Копирование кадров и кропов в {out_dir}…")
    _copy_media(observations, out_dir, crop_mode=crop_mode)
    cards = build_cards(observations, cfg, threshold, include_normal, run_id=out_dir.name,
                        location_radius_m=location_radius_m)
    evidence = {o.record.image_id: o.image_digest for o in observations}
    # Keep decisions only when the reviewed evidence, settings and model match.
    for card in cards:
        reviewed = {k: v for k, v in card.items() if k != "queue_position"}
        if registry is None and all(frame["location"]["camera_lat"] is None for frame in card["frames"]):
            # Provenance for historical coordinates is derived metadata. Keep IDs
            # for existing decisions on identical unregistered batches.
            reviewed.pop("location")
            reviewed["frames"] = [{k: v for k, v in frame.items() if k != "location"} for frame in card["frames"]]
        identity = [reviewed, model.meta.get("created_at"), cfg.quality.to_dict(), crop_mode,
                    [evidence[frame["image_id"]] for frame in card["frames"]]]
        if full_frame_model is not None and any(
            frame.get("localization", {}).get("source") == "full_frame" for frame in card["frames"]
        ):
            identity.append(full_frame_model.meta.get("created_at"))
        card["card_id"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
        card["model_version"] = model.meta.get("version")

    obs_rows = []
    for o in observations:
        location = _observation_location(o)
        payload = location.payload()
        obs_rows.append(
            {
                "image_id": o.record.image_id,
                "asset_id": o.record.asset_id,
                "captured_at": o.record.captured_at,
                "lat": location.lat,
                "lon": location.lon,
                "reported_object_lat": o.record.lat,
                "reported_object_lon": o.record.lon,
                "label": o.record.label,
                "source_status": o.record.extra.get("source_status"),
                "route_id": o.record.extra.get("route_id"),
                "visit_id": o.record.extra.get("visit_id"),
                "location_status": location.status,
                "location_source": location.source,
                "location_reason": location.reason,
                "matched_asset_id": location.matched_asset_id,
                "camera_lat": payload["camera_lat"],
                "camera_lon": payload["camera_lon"],
                "location_candidates": json.dumps(payload["candidates"], ensure_ascii=False),
                "condition": o.record.condition,
                "status": o.status,
                "sign": o.sign,
                "pred_label": o.prediction.pred_label,
                "p_finding": round(o.prediction.p_finding, 4) if o.quality.ok else None,
                "threshold": o.decision_threshold if o.decision_threshold is not None else threshold,
                "quality_ok": o.quality.ok,
                "quality_reasons": "|".join(o.quality.reasons),
                "sharpness": o.quality.sharpness,
                "brightness": o.quality.brightness,
                "area_fraction": o.quality.area_fraction,
                "image_file": o.image_file,
                **({"localization_source": o.localization["source"],
                    "localization_score": o.localization.get("score")} if o.localization else {}),
                **({"cloud_result": o.cloud_review["confirmation"],
                    "cloud_prediction": o.cloud_review["prediction"],
                    "cloud_status": o.cloud_review["status"]} if o.cloud_review else {}),
            }
        )
    pd.DataFrame(obs_rows).to_csv(out_dir / "observations.csv", index=False)

    status_counts = pd.Series([o.status for o in observations]).value_counts().to_dict()
    reasons: dict[str, int] = {}
    for o in observations:
        for reason in o.quality.reasons:
            key = re.sub(r"\s*\(.*?\)", "", reason)
            reasons[key] = reasons.get(key, 0) + 1
    signs: dict[str, int] = {}
    for card in cards:
        if card["status"] in {"finding", "uncertain"}:
            signs[card["finding_short"]] = signs.get(card["finding_short"], 0) + 1

    synthetic = any(r.source == "synthetic" for r in records) or (
        Path(data_dir) / "manifest.csv"
    ).read_text(encoding="utf-8", errors="ignore").find("synthetic") >= 0
    summary = {
        "run_id": out_dir.name,
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "model": {
            "backbone": model.meta.get("backbone"),
            "task": model.meta.get("task"),
            "version": model.meta.get("version"),
            "mode": model.mode,
            "classes": model.classes,
            "trained_at": model.meta.get("created_at"),
            "data_fingerprint": model.meta.get("data_fingerprint"),
        },
        "threshold": threshold,
        "n_images": len(records),
        "n_processed": len(observations),
        "n_assets": len({r.asset_key for r in records}),
        "n_assets_with_provided_id": len({r.asset_id for r in records if r.asset_id_origin == "provided"}),
        "n_frames_without_provided_id": sum(r.asset_id_origin != "provided" for r in records),
        "n_findings": int(sum(1 for c in cards if c["status"] == "finding")),
        "n_uncertain": int(sum(1 for c in cards if c["status"] == "uncertain")),
        "n_rephoto": int(sum(1 for c in cards if c["status"] == "rephoto")),
        "n_normal_cards": int(sum(1 for c in cards if c["status"] == "normal")),
        "n_cards": len(cards),
        "n_queue": int(sum(1 for c in cards if c["status"] != "normal")),
        "observation_status": status_counts,
        "location_status_counts": pd.Series([c["location"]["status"] for c in cards]).value_counts().to_dict(),
        "location_radius_m": location_radius_m,
        "quality_reasons": reasons,
        "signs": signs,
        "coverage_decisive_share": round(
            (status_counts.get("finding", 0) + status_counts.get("normal", 0)) / max(1, len(observations)), 4
        ),
        "refusal_share": round(status_counts.get("rephoto", 0) / max(1, len(observations)), 4),
        "synthetic": bool(synthetic),
        "source_counts": pd.Series([r.source for r in records]).value_counts().to_dict(),
        "label_counts": pd.Series([r.label or "none" for r in records]).value_counts().to_dict(),
        "quality_profile": profile,
        "seconds": round(time.time() - t0, 2),
        "seconds_per_image": round((time.time() - t0) / max(1, len(observations)), 3),
        "quality_config": cfg.quality.to_dict(),
        "data_fingerprint": data_fingerprint(records),
    }
    if locator is not None:
        summary["locator"] = {"backend": getattr(locator, "backend", None),
                              "threshold": locator.score_threshold,
                              "single_target": single_target,
                              "full_frame_fallback": full_frame_model is not None,
                              "counts": pd.Series([o.localization["source"] for o in observations]).value_counts().to_dict()}
    if cloud_reviewer is not None:
        results = [o.cloud_review["confirmation"] for o in observations if o.cloud_review]
        summary["cloud_review"] = {"model": cloud_reviewer.model, "requests": len(results),
                                   "counts": pd.Series(results, dtype=str).value_counts().to_dict()}
    (out_dir / "cards.json").write_text(json.dumps(cards, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        print(
            f"Обработано {summary['n_processed']} кадров за {summary['seconds']} с. "
            f"Сформировано {summary['n_cards']} карточек ({summary['n_findings']} требуют проверки, "
            f"{summary['n_rephoto']} требуют нового кадра)."
        )
        print(f"Объектов с ID — {summary['n_assets_with_provided_id']}, "
              f"кадров без ID — {summary['n_frames_without_provided_id']}.")
        print(f"Карточки: {out_dir / 'cards.json'}")
    return summary


def inspect_single(
    image_path: str | Path,
    asset_id: str,
    model: InspectorModel,
    cfg: AppConfig,
    threshold: float | None = None,
    out_dir: str | Path | None = None,
    locator: Locator | None = None,
    single_target: bool = False,
    full_frame_model: InspectorModel | None = None,
    cloud_reviewer: VertexVision | None = None,
) -> dict[str, Any]:
    """Analyze one new frame for the live-demo upload path."""
    threshold = float(threshold if threshold is not None else model.threshold)
    record = Record(
        image_id=Path(image_path).stem,
        image_path=Path(image_path),
        asset_id=asset_id,
        object_type="well",
        captured_at=pd.Timestamp.now().isoformat(timespec="seconds"),
        source="upload",
        asset_id_origin="image",
    )
    obs = predict_records([record], model, cfg, threshold, "auto", verbose=False,
                          locator=locator, single_target=single_target, full_frame_model=full_frame_model,
                          cloud_reviewer=cloud_reviewer, cloud_limit=1)[0]
    if out_dir is not None:
        _copy_media([obs], Path(out_dir), max_side=1400)
    card = build_cards([obs], cfg, threshold, include_normal=True, run_id="upload")[0]
    card["run_id"] = "upload"
    card["model_version"] = model.meta.get("version")
    return card
