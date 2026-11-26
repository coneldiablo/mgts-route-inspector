"""Manifest parsing: one row is one image tied to one object visit."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config import PUBLIC_SOURCE, PUBLIC_REVIEW_LABEL


@dataclass
class Record:
    image_id: str
    image_path: Path
    asset_id: str
    object_type: str = "well"
    captured_at: str | None = None
    lat: float | None = None
    lon: float | None = None
    label: str | None = None
    condition: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    criticality: float = 0.5
    source: str = "unknown"
    row_index: int = 0
    extra: dict[str, str] = field(default_factory=dict)
    asset_id_origin: str = "provided"
    object_type_explicit: bool = False

    @property
    def asset_key(self) -> tuple[str, str]:
        identifier = self.image_id if self.asset_id_origin == "image" else self.asset_id
        return self.asset_id_origin, identifier

    @property
    def display_coords(self) -> str:
        if self.lat is None or self.lon is None:
            return "координаты не переданы"
        return f"{self.lat:.5f}, {self.lon:.5f}"


def _clean(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    return text or None


def _float(value) -> float | None:
    text = _clean(value)
    if text is None:
        return None
    try:
        number = float(text.replace(",", "."))
        return number if math.isfinite(number) else None
    except ValueError:
        return None


def _bbox(row: pd.Series) -> tuple[float, float, float, float] | None:
    keys = ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")
    if not all(k in row and _float(row[k]) is not None for k in keys):
        return None
    x1, y1, x2, y2 = (_float(row[k]) for k in keys)
    if x2 <= x1 or y2 <= y1:  # type: ignore[operator]
        return None
    return (x1, y1, x2, y2)  # type: ignore[return-value]


def _fallback_asset_id(image_id: str) -> str:
    """Use a filename convention such as KKS-001__frame2 before giving up."""
    for sep in ("__", "_frame", "-f"):
        if sep in image_id:
            return image_id.split(sep)[0]
    return image_id


def load_manifest(data_dir: str | Path, filename: str = "manifest.csv") -> list[Record]:
    data_dir = Path(data_dir)
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден {path}. Нужен manifest.csv с колонкой image (путь относительно {data_dir})."
        )
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "image" not in df.columns:
        raise ValueError("В manifest.csv нет обязательной колонки image")
    records: list[Record] = []
    seen_ids: set[str] = set()
    for idx, row in df.iterrows():
        image_rel = _clean(row["image"])
        if image_rel is None:
            raise ValueError(f"В строке {idx + 2} не указан путь image.")
        image_path = (data_dir / image_rel).resolve()
        image_id = _clean(row.get("image_id")) or Path(image_rel).stem
        if image_id in seen_ids:
            raise ValueError(f"Повторяющийся image_id в строке {idx + 2}: {image_id}")
        seen_ids.add(image_id)
        criticality = _float(row.get("criticality"))
        asset_raw = _clean(row.get("asset_id"))
        source = _clean(row.get("source")) or "unknown"
        id_hint = _clean(row.get("id_origin"))
        if id_hint == "image" or source == PUBLIC_SOURCE:
            asset_id = image_id if id_hint == "image" else (asset_raw or _fallback_asset_id(image_id))
            asset_id_origin = "image"
        else:
            asset_id = asset_raw or _fallback_asset_id(image_id)
            asset_id_origin = "provided" if asset_raw and id_hint in {None, "provided"} else "fallback"
        object_type = _clean(row.get("object_type"))
        label = _clean(row.get("label"))
        if label == "open_hatch" and source == PUBLIC_SOURCE:
            label = PUBLIC_REVIEW_LABEL
        records.append(
            Record(
                image_id=image_id,
                image_path=image_path,
                asset_id=asset_id,
                object_type=object_type or "well",
                captured_at=_clean(row.get("captured_at")),
                lat=_float(row.get("lat")),
                lon=_float(row.get("lon")),
                label=label,
                condition=_clean(row.get("condition")),
                bbox=_bbox(row),
                criticality=0.5 if criticality is None else criticality,
                source=source,
                row_index=int(idx),
                extra={k: str(row[k]) for k in df.columns if k not in {
                    "image", "image_id", "asset_id", "object_type", "captured_at",
                    "lat", "lon", "label", "condition", "criticality", "source",
                    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
                } and _clean(row[k]) is not None},
                asset_id_origin=asset_id_origin,
                object_type_explicit=object_type is not None,
            )
        )
    if not records:
        raise ValueError("manifest.csv не содержит ни одной строки с изображением")
    missing = [r.image_path for r in records if not r.image_path.is_file()]
    if missing:
        preview = "\n".join(str(p) for p in missing[:5])
        raise FileNotFoundError(f"Не найдено {len(missing)} изображений, например:\n{preview}")
    return records


def image_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_fingerprint(records: list[Record], legacy: bool = False) -> str:
    digest = hashlib.sha256()
    for r in sorted(records, key=lambda row: row.image_id):
        if legacy:
            st = r.image_path.stat()
            digest.update(f"{r.image_id}|{r.asset_id}|{st.st_size}|{int(st.st_mtime)}".encode())
        else:
            fields = [r.image_id, r.asset_id, r.object_type, r.label, r.bbox, r.condition, r.source,
                      r.criticality, r.captured_at, r.lat, r.lon, r.extra, image_sha256(r.image_path)]
            digest.update(json.dumps(fields, sort_keys=True).encode() + b"\n")
    return digest.hexdigest()[:16] if legacy else "v2-" + digest.hexdigest()


def to_frame(records: list[Record]) -> pd.DataFrame:
    rows = []
    for r in records:
        row = {
            "image_id": r.image_id,
            "image": str(r.image_path),
            "asset_id": r.asset_id,
            "object_type": r.object_type,
            "captured_at": r.captured_at,
            "lat": r.lat,
            "lon": r.lon,
            "label": r.label,
            "condition": r.condition,
            "criticality": r.criticality,
            "source": r.source,
        }
        if r.bbox:
            row.update(dict(zip(("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"), r.bbox)))
        rows.append(row)
    return pd.DataFrame(rows)
