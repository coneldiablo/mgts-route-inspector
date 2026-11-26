"""Local cabinet inspection desk. SQLite stores each finding and its supporting images."""
from __future__ import annotations

import csv
import io
import json
import math
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

TASKS = {
    "mechanical": ("Механическое повреждение", "Проверить дверь, крепления и закрывание", "Снять дверь спереди и крупно её петли и кромку"),
    "coating": ("Повреждение покрытия", "Осмотреть и восстановить повреждённое покрытие", "Снять поверхность с повреждением крупно при рассеянном свете"),
}
AUTOMATIC_RETAKE_INSTRUCTIONS = {
    "mechanical": "Снять петли и крепления крупным планом.",
    "coating": "Снять участок покрытия крупным планом при рассеянном свете.",
}
MAX_BYTES = 20 * 1024 * 1024
SCHEMA = """
CREATE TABLE IF NOT EXISTS cards(id TEXT PRIMARY KEY, created TEXT DEFAULT CURRENT_TIMESTAMP,
    prediction TEXT NOT NULL, lat REAL, lon REAL);
CREATE TABLE IF NOT EXISTS photos(id TEXT PRIMARY KEY, card TEXT NOT NULL REFERENCES cards(id),
    purpose TEXT NOT NULL, scope TEXT NOT NULL, data BLOB NOT NULL, created TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS findings(card TEXT NOT NULL REFERENCES cards(id), code TEXT NOT NULL,
    state TEXT NOT NULL, evidence TEXT REFERENCES photos(id), PRIMARY KEY(card,code));
CREATE TABLE IF NOT EXISTS requests(id TEXT PRIMARY KEY, card TEXT NOT NULL REFERENCES cards(id),
    scope TEXT NOT NULL, instruction TEXT NOT NULL, evidence TEXT REFERENCES photos(id),
    origin TEXT NOT NULL DEFAULT 'manual' CHECK(origin IN ('manual', 'automatic')),
    source_photo_id TEXT REFERENCES photos(id));
"""


class Action(BaseModel):
    action: str
    photo_id: str | None = None


class Retake(BaseModel):
    scope: str
    instruction: str = Field(default="", max_length=400)


@contextmanager
def connection(path):
    db = sqlite3.connect(path, timeout=15)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    try:
        with db:
            yield db
    finally:
        db.close()


async def read_photo(upload):
    raw = await upload.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "Допустимый размер фотографии — до 20 МБ.")
    try:
        with Image.open(io.BytesIO(raw)) as opened:
            if opened.format not in {"JPEG", "PNG", "WEBP"} or opened.width * opened.height > 40_000_000:
                raise HTTPException(422, "Загрузите JPEG, PNG или WebP до 40 мегапикселей.")
            img = ImageOps.exif_transpose(opened).convert("RGB")
            img.info.clear()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(422, "Не удалось прочитать изображение.")
    # Inference receives the original pixels. The evidence copy omits EXIF.
    thumb = img.copy()
    thumb.thumbnail((1800, 1800))
    encoded = io.BytesIO()
    thumb.save(encoded, "JPEG", quality=88)
    return img, encoded.getvalue()


def require_card(db, card):
    row = db.execute("SELECT * FROM cards WHERE id=?", (card,)).fetchone()
    if row is None:
        raise HTTPException(404, "Осмотр не найден.")
    return row


def retake_instruction(scope: str) -> str:
    return TASKS[scope][2] if scope in TASKS else "Снять шкаф целиком, прямо и при хорошем освещении."


def automatic_retake_scope(result: dict) -> str | None:
    if (result.get("quality") or {}).get("ok") is False:
        return "object"

    scores = result.get("scores") or {}
    selected = None
    nearest = math.inf
    # TASKS orders mechanical first so it wins ties, including decimal pairs such as 0.55 and 0.45.
    for scope in TASKS:
        score = scores.get(scope)
        if not isinstance(score, (int, float)) or not 0.40 <= score <= 0.60:
            continue
        distance = abs(score - 0.5)
        if distance < nearest and not math.isclose(distance, nearest, rel_tol=0.0, abs_tol=1e-12):
            selected, nearest = scope, distance
    return selected


def details(db, card):
    row = require_card(db, card)
    findings = []
    for item in db.execute("SELECT code,state,evidence FROM findings WHERE card=? ORDER BY rowid", (card,)):
        title, work, retake = TASKS[item["code"]]
        findings.append(dict(item) | dict(title=title, work=work, retake=retake))
    photos = [dict(p) for p in db.execute("SELECT id,purpose,scope,created FROM photos WHERE card=? ORDER BY rowid", (card,))]
    requests = [dict(r) for r in db.execute(
        "SELECT id,scope,instruction,evidence,origin,source_photo_id FROM requests WHERE card=? ORDER BY rowid",
        (card,),
    )]
    closed = any(f["state"] == "accepted" for f in findings) and all(f["state"] in {"accepted", "dismissed"} for f in findings) and all(r["evidence"] for r in requests)
    return dict(id=card, created=row["created"], lat=row["lat"], lon=row["lon"], prediction=json.loads(row["prediction"]),
                findings=findings, photos=photos, requests=requests, closed=bool(closed))


def create_app(db_path=None, predictor=None):
    state_dir = Path(os.environ.get("MGTS_FIELD_DIR", "runs/field"))
    if db_path is None:
        state_dir.mkdir(parents=True, exist_ok=True)
        db_path = state_dir / "inspections.sqlite3"
    with connection(db_path) as db:
        db.executescript(SCHEMA)
        db.execute("BEGIN IMMEDIATE")
        columns = {row["name"] for row in db.execute("PRAGMA table_info(requests)")}
        if "origin" not in columns:
            db.execute("ALTER TABLE requests ADD COLUMN origin TEXT NOT NULL DEFAULT 'manual' "
                       "CHECK(origin IN ('manual', 'automatic'))")
        if "source_photo_id" not in columns:
            db.execute("ALTER TABLE requests ADD COLUMN source_photo_id TEXT REFERENCES photos(id)")
    model_lock = threading.Lock()

    def predict(image):
        nonlocal predictor
        with model_lock:
            if predictor is None:
                from .cabinet import CabinetInspector
                path = Path(os.environ.get("MGTS_CABINET_MODEL", "artifacts/final-cabinet/model.joblib"))
                if not path.is_file():
                    raise HTTPException(503, "Установите модель по инструкции docs/FINAL.md.")
                predictor = CabinetInspector.load(path)
            return predictor.predict_image(image)

    app = FastAPI(title="МГТС · Осмотр шкафов")

    @app.get("/")
    @app.get("/field")
    def home():
        return FileResponse(Path(__file__).parent / "templates" / "field.html")

    @app.get("/api/inspections")
    def queue():
        with connection(db_path) as db:
            items = [details(db, r["id"]) for r in db.execute("SELECT id FROM cards ORDER BY rowid DESC")]
        priority = {"repair": 0, "inspect": 1, "no_action": 2}
        return sorted(items, key=lambda c: (c["closed"], priority.get(c["prediction"]["action"], 1)))

    @app.post("/api/inspections", status_code=201)
    async def inspect(image: UploadFile = File(...), lat: float | None = Form(None), lon: float | None = Form(None)):
        if (lat is None) != (lon is None) or (lat is not None and not (math.isfinite(lat) and -90 <= lat <= 90 and math.isfinite(lon) and -180 <= lon <= 180)):
            raise HTTPException(422, "Укажите обе координаты объекта в градусах WGS84.")
        pixels, evidence = await read_photo(image)
        result = await run_in_threadpool(predict, pixels)
        scope = automatic_retake_scope(result)
        card = uuid.uuid4().hex
        photo = uuid.uuid4().hex
        with connection(db_path) as db:
            db.execute("INSERT INTO cards(id,prediction,lat,lon) VALUES(?,?,?,?)", (card, json.dumps(result, ensure_ascii=False), lat, lon))
            db.execute("INSERT INTO photos(id,card,purpose,scope,data) VALUES(?,?,?,?,?)", (photo,card,"initial","object",evidence))
            for finding in result["findings"]:
                code = finding["code"]
                if code in TASKS:
                    db.execute("INSERT OR IGNORE INTO findings(card,code,state) VALUES(?,?,'proposed')", (card, code))
            if scope is not None:
                instruction = AUTOMATIC_RETAKE_INSTRUCTIONS.get(scope) or retake_instruction(scope)
                db.execute(
                    "INSERT INTO requests(id,card,scope,instruction,origin,source_photo_id) VALUES(?,?,?,?,?,?)",
                    (uuid.uuid4().hex, card, scope, instruction, "automatic", photo),
                )
            return details(db, card)

    @app.get("/api/inspections/{card}")
    def get_card(card: str):
        with connection(db_path) as db:
            return details(db, card)

    @app.post("/api/inspections/{card}/findings/{code}")
    def update(card: str, code: str, command: Action):
        if code not in TASKS or command.action not in {"confirm", "dismiss", "accept"}:
            raise HTTPException(422, "Неизвестное замечание или действие.")
        with connection(db_path) as db:
            db.execute("BEGIN IMMEDIATE")
            require_card(db, card)
            item = db.execute("SELECT state,evidence FROM findings WHERE card=? AND code=?", (card,code)).fetchone()
            current = item["state"] if item else None
            if command.action == "accept":
                photo = db.execute("SELECT id FROM photos WHERE card=? AND id=? AND purpose='after' AND scope IN (?, 'object')", (card,command.photo_id,code)).fetchone()
                if not photo or current not in {"confirmed","accepted"}:
                    raise HTTPException(409, "Подтвердите работу и приложите фото результата для этого замечания.")
                if current == "accepted" and item["evidence"] != command.photo_id:
                    raise HTTPException(409, "Работа уже принята с другим снимком.")
                db.execute("UPDATE findings SET state='accepted',evidence=? WHERE card=? AND code=?", (command.photo_id,card,code))
            else:
                if current == "accepted" or (current is None and command.action == "dismiss"):
                    raise HTTPException(409, "Действие недоступно в текущем состоянии.")
                state = "confirmed" if command.action == "confirm" else "dismissed"
                db.execute("INSERT INTO findings(card,code,state) VALUES(?,?,?) ON CONFLICT(card,code) DO UPDATE SET state=excluded.state", (card,code,state))
            return details(db, card)

    @app.post("/api/inspections/{card}/retakes",status_code=201)
    def request(card: str, payload: Retake):
        if payload.scope not in {*TASKS, "object"}:
            raise HTTPException(422, "Неизвестная зона.")
        instruction = payload.instruction.strip() or retake_instruction(payload.scope)
        with connection(db_path) as db:
            db.execute("BEGIN IMMEDIATE")
            require_card(db, card)
            prior = db.execute("SELECT id FROM requests WHERE card=? AND scope=? AND instruction=? AND evidence IS NULL", (card,payload.scope,instruction)).fetchone()
            if not prior:
                db.execute("INSERT INTO requests(id,card,scope,instruction) VALUES(?,?,?,?)", (uuid.uuid4().hex,card,payload.scope,instruction))
            return details(db, card)

    @app.post("/api/inspections/{card}/photos",status_code=201)
    async def attach(card: str, image: UploadFile = File(...), purpose: str = Form(...), scope: str = Form("object"), request_id: str = Form("")):
        if purpose not in {"retake","after"} or scope not in {*TASKS, "object"}:
            raise HTTPException(422, "Проверьте назначение фотографии.")
        _, data = await read_photo(image)
        photo_id = uuid.uuid4().hex
        with connection(db_path) as db:
            db.execute("BEGIN IMMEDIATE")
            require_card(db, card)
            if db.execute("SELECT 1 FROM photos WHERE data=? LIMIT 1", (data,)).fetchone():
                raise HTTPException(409, "Этот снимок уже приложен к осмотру. Выберите новый кадр.")
            if purpose == "retake":
                req = db.execute("SELECT * FROM requests WHERE card=? AND id=?", (card,request_id)).fetchone()
                if not req or req["evidence"]:
                    raise HTTPException(409, "Выберите открытый запрос пересъёмки для этого объекта.")
                scope = req["scope"]
            elif not db.execute("SELECT 1 FROM findings WHERE card=? AND state='confirmed' AND (code=? OR ?='object')", (card,scope,scope)).fetchone():
                raise HTTPException(409, "Сначала подтвердите работу по выбранному замечанию.")
            db.execute("INSERT INTO photos(id,card,purpose,scope,data) VALUES(?,?,?,?,?)", (photo_id,card,purpose,scope,data))
            if purpose == "retake":
                db.execute("UPDATE requests SET evidence=? WHERE id=?", (photo_id,request_id))
            return details(db, card)

    @app.get("/api/inspections/{card}/photos/{photo}")
    def photograph(card: str, photo: str):
        with connection(db_path) as db:
            row = db.execute("SELECT data FROM photos WHERE card=? AND id=?", (card,photo)).fetchone()
            if not row:
                raise HTTPException(404, "Фото не найдено.")
            return Response(row["data"], media_type="image/jpeg", headers={"Cache-Control":"private, no-store"})

    @app.get("/api/export.csv")
    def export():
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(["Осмотр","Объект","Широта","Долгота","Рекомендация","Замечание","Решение мастера","ID фото приёмки"])
        with connection(db_path) as db:
            for row in db.execute("SELECT c.*,f.code,f.state,f.evidence FROM cards c LEFT JOIN findings f ON c.id=f.card ORDER BY c.rowid, f.rowid"):
                prediction = json.loads(row["prediction"])
                writer.writerow([row["id"], prediction["object_type"],row["lat"],row["lon"],prediction["action"],row["code"],row["state"],row["evidence"]])
        return Response("\ufeff"+out.getvalue(),media_type="text/csv; charset=utf-8",headers={"Content-Disposition":"attachment; filename=mgts-inspections.csv"})
    return app
