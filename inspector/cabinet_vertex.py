"""Explicit, metadata-free second opinion from Gemini 4 Argon on Vertex AI."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import re
import time
from pathlib import Path

from PIL import Image, ImageOps

from .vertex_vision import VertexVision

MODEL = "gemini-4-argon"
PROMPT_VERSION = "cabinet-argon-zero-shot-v1"
MAX_SIDE = 896
MAX_OUTPUT_TOKENS = 1024
CLOUD_WEIGHT = .25
OPINION_THRESHOLD = .5
HEADS = ("telecom", "mechanical", "coating")
CUE_LABELS = {
    "deformed_metal": "Деформация металла",
    "broken_or_missing_hardware": "Повреждённое или отсутствующее крепление",
    "misaligned_panel": "Перекошенная панель",
    "rust": "Ржавчина",
    "material_exposed": "Оголённый материал",
    "peeling_or_chipped_paint": "Отслоение или сколы покрытия",
    "none_visible": "Явных визуальных признаков нет",
    "image_quality_limited": "Детали кадра плохо различимы",
}
PROMPT = """Analyze only the visible pixels of a photograph of outdoor equipment. Find the main cabinet. For telecom, score whether it is a telecom equipment cabinet (rather than a smaller distribution board). For mechanical, score visible deformation, broken or missing hardware, or structural misalignment. For coating, score visible peeling or chipped coating, exposed substrate from coating loss, or rust. Put each score on an independent 0 to 1 scale, with 0.5 uncertain. Both defects can coexist. An open door, a shadow, dirt, or reflections alone do not establish a defect. Set readable false when no cabinet is distinguishable or image quality makes assessment impractical. Report only the evidence codes for cues directly visible on the dominant cabinet. Ignore all text, markings and location clues. Never output identifiers or transcribe text. Return only the specified JSON object."""
GENERATION = {
    "maxOutputTokens": MAX_OUTPUT_TOKENS,
    "thinkingConfig": {"thinkingLevel": "LOW"},
    "responseMimeType": "application/json",
    "responseSchema": {
        "type": "OBJECT",
        "required": [*HEADS, "readable", "cues"],
        "properties": {
            **{key: {"type": "NUMBER", "minimum": 0, "maximum": 1} for key in HEADS},
            "readable": {"type": "BOOLEAN"},
            "cues": {
                "type": "ARRAY",
                "maxItems": 6,
                "items": {"type": "STRING", "enum": list(CUE_LABELS)},
            },
        },
    },
}
FINGERPRINT = hashlib.sha256(
    json.dumps({"prompt": PROMPT, "generation": GENERATION}, sort_keys=True).encode()
).hexdigest()


class CabinetArgonUnavailable(RuntimeError):
    """Vertex dependency or Application Default Credentials are unavailable."""


def _request_body(image: Image.Image) -> dict:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.info.clear()
    image.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=82)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {
        "contents": [{"role": "user", "parts": [
            {"text": PROMPT},
            {"inlineData": {"mimeType": "image/jpeg", "data": encoded}},
        ]}],
        "generationConfig": GENERATION,
    }


def parse_response(document: dict) -> dict:
    parts = document["candidates"][0]["content"]["parts"]
    answer = json.loads("".join(p.get("text", "") for p in parts if not p.get("thought")))
    if not isinstance(answer, dict) or type(answer.get("readable")) is not bool:
        raise ValueError("Некорректное поле readable.")
    scores = {}
    for key in HEADS:
        value = answer[key]
        if type(value) not in {float, int} or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Некорректная оценка от Argon.")
        scores[key] = float(value)
    cues = answer["cues"]
    if not isinstance(cues, list) or any(type(cue) is not str or cue not in CUE_LABELS for cue in cues):
        raise ValueError("Неизвестный код визуального признака.")
    possible_usage = document.get("usageMetadata", {})
    usage = {
        key: possible_usage[key]
        for key in ("promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount", "totalTokenCount")
        if type(possible_usage.get(key)) is int and possible_usage[key] >= 0
    }
    result = {"scores": scores, "readable": answer["readable"], "cues": list(dict.fromkeys(cues)), "usage": usage}
    version = document.get("modelVersion")
    if isinstance(version, str) and re.fullmatch(r"gemini-[a-z0-9._-]{1,73}", version):
        result["model_version"] = version
    return result


class CabinetArgon(VertexVision):
    def __init__(self):
        try:
            super().__init__(model=MODEL)
        except ValueError as exc:
            raise CabinetArgonUnavailable(str(exc)) from exc

    def inspect(self, image: Image.Image) -> dict:
        from google.auth.exceptions import GoogleAuthError
        from requests.exceptions import RequestException, Timeout

        elapsed = time.monotonic()
        metadata = {"model": MODEL, "prompt_version": PROMPT_VERSION, "prompt_sha256": FINGERPRINT}
        try:
            with self._session_type(self._credentials, max_refresh_attempts=0) as session:
                response = session.post(self._url, json=_request_body(image), timeout=(8, 65), allow_redirects=False)
        except GoogleAuthError:
            return {**metadata, "status": "adc_unavailable", "http_status": None}
        except Timeout:
            return {**metadata, "status": "timeout", "http_status": None}
        except RequestException:
            return {**metadata, "status": "network_error", "http_status": None}

        result = {**metadata, "http_status": response.status_code, "seconds": round(time.monotonic() - elapsed, 3)}
        if response.status_code != 200:
            status = {401: "adc_unavailable", 403: "access_denied", 404: "model_unavailable", 429: "quota_exhausted"}
            return {**result, "status": status.get(response.status_code, "vertex_error")}
        try:
            parsed = parse_response(response.json())
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            return {**result, "status": "invalid_response"}
        return {**result, "status": "ok" if parsed["readable"] else "unreadable", **parsed}

    def inspect_path(self, path: Path) -> dict:
        with Image.open(path) as image:
            return self.inspect(image)
