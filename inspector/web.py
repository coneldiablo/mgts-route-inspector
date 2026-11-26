"""FastAPI app for the master's queue and upload demo."""
from __future__ import annotations

import io
from collections import Counter
import json
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from .classifier import load_model, validate_artifact_task
from .config import AppConfig, QualityConfig, quality_config, PUBLIC_SOURCE
from .pipeline import inspect_single
from .store import ReviewStore, card_location_status, export_csv, export_geojson, export_json, location_candidate, merge_reviews

PACKAGE_DIR = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


class ReviewInput(BaseModel):
    decision: str
    note: str = ""


class LocationInput(BaseModel):
    asset_id: str


def _read_cards(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "cards.json"
    if not path.exists():
        raise HTTPException(404, f"В {run_dir} нет cards.json. Сначала выполните run.")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "summary.json"
    if not path.exists():
        raise HTTPException(404, f"В {run_dir} нет summary.json.")
    return json.loads(path.read_text(encoding="utf-8"))


def create_app(
    run_dir: str | Path,
    artifact_dir: str | Path | None = None,
    config: AppConfig | None = None,
    quality_profile: str | None = None,
    locator_path: str | Path | None = None,
    single_target: bool = False,
    full_frame_artifact: str | Path | None = None,
    cloud_review_model: str | None = None,
) -> FastAPI:
    run_dir = Path(run_dir).resolve()
    artifact_dir = Path(artifact_dir).resolve() if artifact_dir else None
    locator_path = Path(locator_path).resolve() if locator_path else None
    if locator_path and not locator_path.is_file():
        raise FileNotFoundError(f"Не найден артефакт локализации {locator_path}")
    fallback_path = Path(full_frame_artifact).resolve() if full_frame_artifact else None
    if fallback_path and not (fallback_path / "model.joblib").is_file():
        raise FileNotFoundError(f"Модель полного кадра не найдена в {fallback_path}")
    if fallback_path and (locator_path is None or not single_target):
        raise ValueError("Запасной классификатор требует --locator и --single-target.")
    cloud_reviewer = None
    if cloud_review_model is not None:
        if artifact_dir is None or fallback_path is None:
            raise ValueError("Для облачной перепроверки нужны --artifact и --full-frame-artifact.")
        from .vertex_vision import VertexVision

        cloud_reviewer = VertexVision(cloud_review_model)
    cfg = config or AppConfig()
    run_summary = _read_summary(run_dir)
    if PUBLIC_SOURCE in run_summary.get("source_counts", {}) and "open_hatch" in run_summary.get("model", {}).get("classes", []):
        raise ValueError("Этот публичный прогон использует старую трактовку void. Выполните python -m inspector demo.")
    meta = {}
    if artifact_dir and (artifact_dir / "meta.json").is_file():
        meta = json.loads((artifact_dir / "meta.json").read_text(encoding="utf-8"))
        validate_artifact_task(artifact_dir, meta)
        trained_at = run_summary.get("model", {}).get("trained_at")
        if trained_at and trained_at != meta.get("created_at"):
            raise ValueError("Указанная модель отличается от модели сохранённого прогона.")
    stored_quality = run_summary.get("quality_config") or meta.get("quality_config")
    if quality_profile is not None:
        cfg = replace(cfg, quality=quality_config(quality_profile))
    elif stored_quality:
        cfg = replace(cfg, quality=QualityConfig(**stored_quality))
    elif meta.get("quality_profile"):
        cfg = replace(cfg, quality=quality_config(meta["quality_profile"]))
    reviews = ReviewStore(run_dir)

    app = FastAPI(title="Маршрутный осмотр МГТС", docs_url="/api/docs")
    app.state.model = None
    app.state.locator = None
    app.state.full_frame_model = None
    app.state.cfg = cfg

    app.mount("/static", StaticFiles(directory=PACKAGE_DIR / "static"), name="static")

    def reviewed_cards(items: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        cards = _read_cards(run_dir) if items is None else items
        return merge_reviews(cards, reviews.all(), reviews.location_assignments())

    def require_card(card_id: str) -> dict[str, Any]:
        for item in _read_cards(run_dir):
            if item["card_id"] == card_id:
                return item
        raise HTTPException(404, "Карточка не найдена")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return FileResponse(PACKAGE_DIR / "templates" / "index.html")

    @app.get("/api/health")
    def health():
        return {"ok": True, "run": run_dir.name}

    @app.get("/api/summary")
    def summary():
        data = _read_summary(run_dir)
        cards = reviewed_cards()
        data["review_counts"] = dict(Counter(c["review_status"] for c in cards if c.get("review_updated_at")))
        data["location_status_counts"] = dict(Counter(card_location_status(card) for card in cards))
        eval_path = artifact_dir / "eval.json" if artifact_dir else None
        if eval_path and eval_path.exists():
            data["eval"] = json.loads(eval_path.read_text(encoding="utf-8"))
        holdout_path = artifact_dir / "eval_holdout.json" if artifact_dir else None
        if holdout_path and holdout_path.exists():
            data["holdout_eval"] = json.loads(holdout_path.read_text(encoding="utf-8"))
        if cloud_reviewer is not None:
            data["upload_cloud_review"] = cloud_reviewer.model
        data["config"] = cfg.to_dict()
        return data

    @app.get("/api/cards")
    def cards():
        return reviewed_cards()

    @app.get("/api/cards/{card_id}")
    def card(card_id: str):
        return reviewed_cards([require_card(card_id)])[0]

    @app.post("/api/cards/{card_id}/decision")
    def decision(card_id: str, body: ReviewInput):
        require_card(card_id)
        action = body.decision.strip()
        note = body.note.strip()
        try:
            saved = reviews.upsert(card_id, action, note)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"ok": True, "review": saved}

    @app.post("/api/cards/{card_id}/location")
    def assign_location(card_id: str, body: LocationInput):
        item = require_card(card_id)
        asset_id = body.asset_id.strip()
        if location_candidate(item, asset_id) is None:
            raise HTTPException(422, "Выберите объект из списка кандидатов карточки")
        reviews.assign_location(card_id, asset_id)
        return {"ok": True, "card": reviewed_cards([item])[0]}

    @app.get("/media/{rel_path:path}")
    def media(rel_path: str):
        candidate = (run_dir / rel_path).resolve()
        if candidate != run_dir and run_dir not in candidate.parents:
            raise HTTPException(403, "Путь вне каталога прогона")
        parts = candidate.relative_to(run_dir).parts
        media_dir = len(parts) == 2 or (len(parts) == 4 and parts[0] == "uploads")
        if not (media_dir and parts[-2] in {"images", "crops"} and candidate.suffix == ".jpg"):
            raise HTTPException(404, "Файл не найден")
        if not candidate.is_file():
            raise HTTPException(404, "Файл не найден")
        return FileResponse(candidate)

    @app.get("/api/export.csv")
    def export_csv_endpoint():
        return PlainTextResponse(
            "\ufeff" + export_csv(reviewed_cards()),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="mgts_inspection_queue.csv"'},
        )

    @app.get("/api/export.json")
    def export_json_endpoint():
        return PlainTextResponse(
            export_json(reviewed_cards()),
            media_type="application/json; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="mgts_inspection_queue.json"'},
        )

    @app.get("/api/export.geojson")
    def export_geojson_endpoint():
        return JSONResponse(
            export_geojson(reviewed_cards()),
            media_type="application/geo+json",
            headers={"Content-Disposition": 'attachment; filename="mgts_inspection_queue.geojson"'},
        )

    @app.get("/api/model")
    def model_info():
        if not artifact_dir or not (artifact_dir / "meta.json").exists():
            raise HTTPException(404, "Артефакт модели не найден")
        return json.loads((artifact_dir / "meta.json").read_text(encoding="utf-8"))

    def _get_model():
        if app.state.model is None:
            if not artifact_dir or not (artifact_dir / "model.joblib").exists():
                raise HTTPException(503, "Модель не подключена. Укажите --artifact.")
            app.state.model = load_model(artifact_dir)
        return app.state.model

    def _get_locator():
        if locator_path is not None and app.state.locator is None:
            from .detector import load_locator

            app.state.locator = load_locator(locator_path)
        return app.state.locator

    def _get_full_frame_model():
        if fallback_path is not None and app.state.full_frame_model is None:
            app.state.full_frame_model = load_model(fallback_path)
        return app.state.full_frame_model

    @app.post("/api/analyze")
    async def analyze_upload(file: UploadFile = File(...), asset_id: str = Form("UPLOAD-001")):
        """Live path used on the final: a new frame goes through to a card."""
        raw = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Файл больше 20 МБ")
        try:
            with Image.open(io.BytesIO(raw)) as image:
                if image.width * image.height > 24_000_000:
                    raise HTTPException(413, "Разрешение превышает 24 мегапикселя")
                image.load()
        except Image.DecompressionBombError as exc:
            raise HTTPException(413, "Слишком большое изображение") from exc
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise HTTPException(400, f"Не удалось прочитать изображение: {exc}") from exc
        model = _get_model()
        upload_root = run_dir / "uploads"
        token = uuid.uuid4().hex[:10]
        upload_dir = upload_root / token
        upload_dir.mkdir(parents=True, exist_ok=True)
        safe_asset = "".join(c for c in asset_id if c.isalnum() or c in "-_")[:40] or "UPLOAD-001"
        path = upload_dir / f"{safe_asset}.original"
        path.write_bytes(raw)
        try:
            result = await run_in_threadpool(
                inspect_single, path, asset_id=safe_asset, model=model, cfg=cfg,
                threshold=run_summary.get("threshold", model.threshold), out_dir=upload_dir,
                locator=_get_locator(), single_target=single_target, full_frame_model=_get_full_frame_model(),
                cloud_reviewer=cloud_reviewer,
            )
        finally:
            path.unlink(missing_ok=True)
        prefix = f"uploads/{token}/"
        result["image_file"] = prefix + result["image_file"]
        result["crop_file"] = prefix + result["crop_file"]
        for frame in result.get("frames", []):
            if frame.get("image_file"):
                frame["image_file"] = prefix + frame["image_file"]
            if frame.get("crop_file"):
                frame["crop_file"] = prefix + frame["crop_file"]
        return {"ok": True, "card": result}

    return app
