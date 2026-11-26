"""Isolated stateless jury endpoint for a single JPEG cabinet photograph."""

from __future__ import annotations

import asyncio
import io
import math
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError

from .cabinet_vertex import CLOUD_WEIGHT, MODEL as VERTEX_MODEL, CabinetArgon
from .vertex_vision import VertexVision

MAX_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 16_000_000
MAX_IN_FLIGHT = 2
MAX_CLOUD_ATTEMPTS = 80  # One full 62-frame competition set plus individual jury uploads.
MIN_CLOUD_INTERVAL = 2.0
CLOUD_WAIT_TIMEOUT = 35.0
UPLOAD_TIMEOUT = 15
DEFECTS = ("mechanical", "coating")
OBJECT_TYPES = frozenset({"telecom_cabinet", "distribution_board"})
DEFAULT_MODEL = Path("artifacts/final-cabinet/model.joblib")


class _ProjectArgon(CabinetArgon):
    def __init__(self):
        VertexVision.__init__(self, model=VERTEX_MODEL, project=os.environ.get("MGTS_VERTEX_PROJECT"))


async def _read_jpeg(request: Request) -> bytes:
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "image/jpeg":
        raise HTTPException(415, "Ожидается JPEG.")
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            length = int(content_length)
        except ValueError:
            raise HTTPException(400, "Некорректный размер запроса.") from None
        if length < 0:
            raise HTTPException(400, "Некорректный размер запроса.")
        if length > MAX_BYTES:
            raise HTTPException(413, "JPEG должен занимать не более 5 МиБ.")
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > MAX_BYTES:
            raise HTTPException(413, "JPEG должен занимать не более 5 МиБ.")
        body.extend(chunk)
    return bytes(body)


def _decode_jpeg(body: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(body), formats=("JPEG",)) as original:
            if original.width * original.height > MAX_PIXELS:
                raise HTTPException(413, "JPEG должен содержать не более 16 мегапикселей.")
            pixels = ImageOps.exif_transpose(original).convert("RGB")
            pixels.info.clear()
            return pixels
    except Image.DecompressionBombError:
        raise HTTPException(413, "JPEG должен содержать не более 16 мегапикселей.") from None
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(422, "Не удалось прочитать JPEG.") from None


def _utc_day():
    return datetime.now(timezone.utc).date()


def _score(value) -> float:
    if type(value) not in {int, float} or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Score outside [0, 1].")
    return float(value)


def _defect_scores(result: dict) -> dict[str, float]:
    return {name: _score(result["scores"][name]) for name in DEFECTS}


class _Analysis:
    def __init__(self, path, predictor, cloud_client, cloud_factory, cloud_budget):
        self.path = path
        self.predictor = predictor
        self.cloud_client = cloud_client
        self.cloud_factory = cloud_factory
        self.cloud_budget = cloud_budget
        self.remaining = cloud_budget
        self.cloud_day = _utc_day()
        self.last_cloud_start = -math.inf
        self.local_lock = threading.Lock()
        self.cloud_lock = threading.Lock()

    def local(self, image):
        with self.local_lock:
            if self.predictor is None:
                from .cabinet import CabinetInspector

                self.predictor = CabinetInspector.load(self.path)
            return self.predictor.predict_image(image)

    def cloud(self, image):
        deadline = time.monotonic() + CLOUD_WAIT_TIMEOUT
        if not self.cloud_lock.acquire(timeout=CLOUD_WAIT_TIMEOUT):
            return {"model": VERTEX_MODEL, "status": "cloud_wait_timeout", "scores": None}
        try:
            day = _utc_day()
            if day > self.cloud_day:
                self.cloud_day = day
                self.remaining = self.cloud_budget
            if self.remaining <= 0:
                return {"model": VERTEX_MODEL, "status": "budget_exhausted", "scores": None}
            now = time.monotonic()
            next_start = self.last_cloud_start + MIN_CLOUD_INTERVAL
            if next_start > deadline:
                return {"model": VERTEX_MODEL, "status": "cloud_wait_timeout", "scores": None}
            delay = next_start - now
            if delay > 0:
                time.sleep(delay)
            self.remaining -= 1
            self.last_cloud_start = time.monotonic()
            try:
                if self.cloud_client is None:
                    self.cloud_client = self.cloud_factory()
                reply = self.cloud_client.inspect(image)
                model = reply.get("model_version") or reply.get("model") or VERTEX_MODEL
                status = reply["status"]
                if status == "ok":
                    if reply.get("readable") is not True:
                        return {"model": model, "status": "unreadable", "scores": None}
                    return {"model": model, "status": status, "scores": _defect_scores(reply)}
                return {"model": model, "status": status, "scores": None}
            except Exception:
                return {"model": VERTEX_MODEL, "status": "unavailable", "scores": None}
        finally:
            self.cloud_lock.release()

    def analyze(self, body: bytes) -> dict:
        image = _decode_jpeg(body)
        try:
            result = self.local(image)
            local = _defect_scores(result)
            release = result["model_release"]
            object_type = result["object_type"]
            if object_type not in OBJECT_TYPES or not isinstance(release, str) or not release:
                raise ValueError("Invalid cabinet result.")
            object_score = _score(result["object_type_score"])
        except Exception:
            raise HTTPException(503, "Локальная модель временно недоступна.") from None
        cloud = self.cloud(image)
        weight = CLOUD_WEIGHT if cloud["status"] == "ok" else 0.0
        scores = {name: (1 - weight) * local[name] + weight * cloud["scores"][name] for name in DEFECTS} if weight else local
        return {
            "mode": "hybrid" if weight else "server_local",
            "model_version": release,
            "object_type": object_type,
            "object_type_score": object_score,
            "scores": scores,
            "cloud_status": cloud["status"],
            "analysis": {
                "local": {"model": release, "scores": local},
                "cloud": cloud,
                "weights": {"local": 1 - weight, "cloud": weight},
            },
        }


def create_app(*, predictor=None, cloud_client=None, cloud_factory=_ProjectArgon, cloud_budget=MAX_CLOUD_ATTEMPTS):
    model_path = Path(os.environ.get("MGTS_CABINET_MODEL", DEFAULT_MODEL))
    analysis = _Analysis(model_path, predictor, cloud_client, cloud_factory, cloud_budget)
    in_flight = threading.BoundedSemaphore(MAX_IN_FLIGHT)
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://coneldiablo.github.io"],
        allow_origin_regex=r"https?://(?:localhost|127\.0\.0\.1|\[::1\])(?::\d{1,5})?",
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "ngrok-skip-browser-warning"],
    )

    @app.get("/healthz")
    def healthz():
        if analysis.predictor is None and not model_path.is_file():
            raise HTTPException(503, "Локальный артефакт отсутствует.")
        return {"status": "ok", "service": "mgts-jury"}

    @app.post("/analyze")
    async def analyze(request: Request):
        if not in_flight.acquire(blocking=False):
            raise HTTPException(429, "Сервер занят. Повторите запрос чуть позже.",
                                headers={"Retry-After": "3"})
        try:
            try:
                body = await asyncio.wait_for(_read_jpeg(request), timeout=UPLOAD_TIMEOUT)
            except asyncio.TimeoutError:
                raise HTTPException(408, "Время загрузки JPEG истекло.") from None
            return await run_in_threadpool(analysis.analyze, body)
        finally:
            in_flight.release()

    return app
