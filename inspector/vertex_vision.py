"""Vertex vision using application default credentials and structured outputs."""

from __future__ import annotations

import base64
import io
import json
import re
import time
from typing import Any

from PIL import Image

LABELS = ("open_hatch", "normal", "uncertain")
PROMPT = (
    "Classify the visible condition of the manhole centered in this cropped surface photograph. "
    'Use "open_hatch" if the cover is absent or shifted enough that the opening is visibly exposed. '
    'Use "normal" if the opening is visibly covered, even if the cover is dark. '
    'Use "uncertain" when the target or its opening cannot be distinguished, '
    "or for a lookalike such as a pothole, drain grate, or interior photograph. "
    "Report one short piece of visible evidence in Russian. An uncovered manhole may be undergoing routine work."
)


def request_body(image: Image.Image, *, full_frame: bool = False, model: str = "gemini-3.5-flash") -> dict:
    image = image.convert("RGB")
    image.thumbnail((1024, 1024))
    with io.BytesIO() as jpeg:
        image.save(jpeg, format="JPEG", quality=90)
        encoded = base64.b64encode(jpeg.getvalue()).decode("ascii")
    prompt = PROMPT
    fields = {
        "label": {"type": "STRING", "enum": list(LABELS)},
        "visible_evidence": {"type": "STRING"},
    }
    if full_frame:
        prompt = PROMPT.replace("centered in this cropped surface photograph", "visible on the surface in this full photograph")
        prompt += (
            " Identify the visible rim or cover on the ground. "
            "Return its tight box as [y_min, x_min, y_max, x_max] with integer coordinates normalized to 0–1000. "
            "Return uncertain and an empty array if there is no clearly identifiable single target."
        )
        fields["box_yxyx"] = {"type": "ARRAY", "items": {"type": "INTEGER"}}
    gemini_4 = model.startswith("gemini-4-")
    generation = {
        "maxOutputTokens": 1024 if gemini_4 else 256,
        "thinkingConfig": {"thinkingLevel": "LOW" if gemini_4 else "MINIMAL"},
        "responseMimeType": "application/json",
        "responseSchema": {"type": "OBJECT", "required": list(fields), "properties": fields},
    }
    if not gemini_4:
        generation["temperature"] = 0.0
    return {
        "contents": [{"role": "user", "parts": [
            {"text": prompt},
            {"inlineData": {"mimeType": "image/jpeg", "data": encoded}},
        ]}],
        "generationConfig": generation,
    }


def decode_response(document: dict) -> dict:
    parts = document["candidates"][0]["content"]["parts"]
    result = json.loads("".join(part.get("text", "") for part in parts if not part.get("thought")))
    if not isinstance(result, dict) or result.get("label") not in LABELS or not isinstance(result.get("visible_evidence"), str):
        raise ValueError("Ответ Vertex не соответствует схеме результата.")
    reply = {"prediction": result["label"], "evidence": result["visible_evidence"][:500],
             "usage": document.get("usageMetadata", {})}
    if "box_yxyx" in result:
        reply["box_yxyx"] = result["box_yxyx"]
    return reply


def _image_box(values: list, size: tuple[int, int]) -> tuple[float, float, float, float]:
    if not isinstance(values, list) or len(values) != 4 or any(type(v) is not int for v in values):
        raise ValueError("Неверные координаты рамки Vertex.")
    y1, x1, y2, x2 = values
    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        raise ValueError("Рамка Vertex выходит за границы допустимой области.")
    width, height = size
    return (x1 * width / 1000, y1 * height / 1000, x2 * width / 1000, y2 * height / 1000)


class VertexVision:
    def __init__(self, model: str = "gemini-4-argon", project: str | None = None):
        if not re.fullmatch(r"gemini-[a-z0-9.\-]+", model):
            raise ValueError("Нужен идентификатор модели Gemini из Google Cloud.")
        try:
            from google.auth import default
            from google.auth.exceptions import GoogleAuthError
            from google.auth.transport.requests import AuthorizedSession, Request
        except ImportError as exc:
            raise ValueError("Для Vertex установите дополнение проекта .[vertex].") from exc
        try:
            credentials, default_project = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            credentials.refresh(Request())
        except GoogleAuthError as exc:
            raise ValueError(f"Доступ через ADC не получен ({type(exc).__name__}).") from exc
        project = project or default_project or getattr(credentials, "quota_project_id", None)
        if not project or not re.fullmatch(r"[a-zA-Z0-9-]+", project):
            raise ValueError("Не определён проект Google Cloud для ADC.")
        resource = f"projects/{project}/locations/global/publishers/google/models/{model}:generateContent"
        version = "v1beta1" if model.startswith("gemini-4-") else "v1"
        self._url = f"https://aiplatform.googleapis.com/{version}/{resource}"
        self._credentials = credentials
        self._session_type = AuthorizedSession
        self.model = model

    def inspect(self, image: Image.Image, *, full_frame: bool = False) -> dict[str, Any]:
        from google.auth.exceptions import GoogleAuthError
        from requests.exceptions import RequestException

        start = time.monotonic()
        payload = request_body(image, full_frame=full_frame, model=self.model)
        try:
            with self._session_type(self._credentials) as session:
                response = session.post(self._url, json=payload, timeout=(8, 55))
        except (GoogleAuthError, RequestException) as exc:
            return {"status": 0, "seconds": round(time.monotonic() - start, 3),
                    "prediction": "uncertain", "error": type(exc).__name__}
        row = {"status": response.status_code, "seconds": round(time.monotonic() - start, 3)}
        if response.ok:
            try:
                doc = response.json()
                result = decode_response(doc)
                if full_frame and result["prediction"] != "uncertain":
                    result["bbox"] = _image_box(result["box_yxyx"], image.size)
                return {**row, **result}
            except (KeyError, IndexError, TypeError, ValueError):
                return {**row, "prediction": "uncertain", "error": "invalid_response"}
        try:
            error = response.json().get("error", {}).get("status")
        except (AttributeError, ValueError):
            error = None
        return {**row, "prediction": "uncertain", "error": error or "http_error"}
