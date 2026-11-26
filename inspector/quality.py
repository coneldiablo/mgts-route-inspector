"""Image loading and frame-quality checks that run before any model decision."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from .config import QualityConfig


def open_image(path: str | Path) -> Image.Image:
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def crop_bbox(img: Image.Image, bbox: tuple[float, float, float, float], margin: float = 0.08) -> Image.Image:
    w, h = img.size
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, x1 - bw * margin)
    y1 = max(0, y1 - bh * margin)
    x2 = min(w, x2 + bw * margin)
    y2 = min(h, y2 + bh * margin)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return img
    return img.crop((int(x1), int(y1), int(x2), int(y2)))


@dataclass
class Quality:
    ok: bool
    score: float
    sharpness: float
    brightness: float
    contrast: float
    area_fraction: float | None
    width: int
    height: int
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _laplacian_variance(gray: np.ndarray) -> float:
    a = gray.astype(np.float32)
    if min(a.shape) < 3:
        return 0.0
    lap = (
        -4.0 * a[1:-1, 1:-1]
        + a[:-2, 1:-1]
        + a[2:, 1:-1]
        + a[1:-1, :-2]
        + a[1:-1, 2:]
    )
    return float(lap.var())


def assess_quality(
    img: Image.Image,
    bbox: tuple[float, float, float, float] | None,
    cfg: QualityConfig | None = None,
) -> Quality:
    cfg = cfg or QualityConfig()
    width, height = img.size
    reasons: list[str] = []

    # Compute sharpness on a fixed-size copy so the threshold is comparable.
    scale = 512.0 / max(width, height)
    small = img.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.Resampling.BILINEAR,
    ) if max(width, height) > 512 else img
    gray = np.asarray(small.convert("L"), dtype=np.float32)
    sharpness = _laplacian_variance(gray)
    brightness = float(gray.mean())
    contrast = float(gray.std())

    area_fraction = None
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        visible_width = max(0.0, min(width, x2) - max(0, x1))
        visible_height = max(0.0, min(height, y2) - max(0, y1))
        area_fraction = visible_width * visible_height / (width * height)

    if width < cfg.min_width or height < cfg.min_height:
        reasons.append(f"низкое разрешение ({width}×{height})")
    if sharpness < cfg.min_sharpness:
        reasons.append("кадр нерезкий")
    if brightness < cfg.min_brightness:
        reasons.append("слишком темно")
    if brightness > cfg.max_brightness:
        reasons.append("пересвет")
    if contrast < cfg.min_contrast:
        reasons.append("низкий контраст")
    if area_fraction is not None and area_fraction < cfg.min_area_fraction:
        reasons.append("объект занимает слишком малую часть кадра")

    ok = not reasons
    sharp_score = min(1.0, sharpness / max(cfg.min_sharpness * 2.0, 1e-6))
    bright_score = max(0.0, 1.0 - abs(brightness - 128.0) / 128.0)
    contrast_score = min(1.0, contrast / 45.0)
    score = 0.55 * sharp_score + 0.25 * bright_score + 0.20 * contrast_score
    if not ok:
        score *= 0.45

    return Quality(
        ok=ok,
        score=round(float(score), 4),
        sharpness=round(sharpness, 2),
        brightness=round(brightness, 2),
        contrast=round(contrast, 2),
        area_fraction=None if area_fraction is None else round(float(area_fraction), 4),
        width=width,
        height=height,
        reasons=reasons,
    )


def save_jpeg(img: Image.Image, path: str | Path, max_side: int = 1600, quality: int = 88) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = img.copy()
    if max(out.size) > max_side:
        ratio = max_side / max(out.size)
        out = out.resize((max(1, int(out.width * ratio)), max(1, int(out.height * ratio))), Image.Resampling.LANCZOS)
    out.convert("RGB").save(path, "JPEG", quality=quality, optimize=True)
