"""Locate an observation without treating the camera's position as the asset's."""
from __future__ import annotations

import csv
import math
from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .manifest import Record


EARTH_RADIUS_M = 6_371_008.8
DEFAULT_LOCATION_RADIUS_M = 30.0
Point = tuple[float, float]


def coordinate_pair(lat: object, lon: object) -> Point | None:
    if lat is None or lon is None or isinstance(lat, bool) or isinstance(lon, bool):
        return None
    try:
        a = float(str(lat).strip().replace(",", "."))
        b = float(str(lon).strip().replace(",", "."))
    except ValueError:
        return None
    return (a, b) if -90 <= a <= 90 and -180 <= b <= 180 else None


def validate_location_radius(radius_m: float) -> None:
    if not math.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("Радиус поиска по реестру должен быть положительным числом в метрах.")


def haversine_m(start: Point, end: Point) -> float:
    lat1, lon1 = map(math.radians, start)
    lat2, lon2 = map(math.radians, end)
    chord = math.sin((lat2 - lat1) / 2) ** 2
    chord += math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(min(1.0, chord)))


@dataclass(frozen=True)
class RegistryAsset:
    asset_id: str
    lat: float
    lon: float
    object_type: str
    address: str = ""

    @property
    def point(self) -> Point:
        return self.lat, self.lon


@dataclass(frozen=True)
class Candidate:
    asset_id: str
    lat: float
    lon: float
    object_type: str
    distance_m: float | None
    address: str


def _candidate(asset: RegistryAsset, distance_m: float | None) -> Candidate:
    distance = None if distance_m is None else round(distance_m, 1)
    return Candidate(asset.asset_id, asset.lat, asset.lon, asset.object_type, distance, asset.address)


def _normalize_type(value: str) -> str:
    normalized = value.strip().casefold().replace("ё", "е").replace("_", " ")
    if normalized in {"well", "manhole", "люк", "колодец", "кабельный колодец", "ккс"}:
        return "well"
    if normalized in {"pole", "опора", "столб", "опора двойного назначения", "одн"}:
        return "pole"
    if normalized in {"unknown", "неизвестно", "не указан", "объект", "другое"}:
        return ""
    return normalized


class AssetRegistry:
    def __init__(self, assets: list[RegistryAsset]):
        self.by_id = {asset.asset_id: asset for asset in assets}
        self._by_latitude = sorted(assets, key=lambda asset: asset.lat)
        self._latitudes = [asset.lat for asset in self._by_latitude]

    def nearby(self, point: Point, radius_m: float, object_type: str) -> tuple[Candidate, ...]:
        latitude_delta = math.degrees(radius_m / EARTH_RADIUS_M)
        start = bisect_left(self._latitudes, point[0] - latitude_delta)
        stop = bisect_right(self._latitudes, point[0] + latitude_delta)
        matches = []
        for asset in self._by_latitude[start:stop]:
            asset_type = _normalize_type(asset.object_type)
            if object_type and asset_type and object_type != asset_type:
                continue
            distance = haversine_m(point, asset.point)
            if distance <= radius_m:
                matches.append((distance, asset.asset_id, asset))
        matches.sort(key=lambda item: item[:2])
        return tuple(_candidate(asset, distance) for distance, _, asset in matches)


def load_registry(path: str | Path) -> AssetRegistry:
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        header = file.readline()
        try:
            dialect = csv.Sniffer().sniff(header, delimiters=",;\t")
        except csv.Error as exc:
            raise ValueError("Ожидается CSV-реестр с колонками asset_id, lat, lon, object_type.") from exc
        file.seek(0)
        rows = csv.DictReader(file, dialect=dialect)
        columns = rows.fieldnames or []
        required = {"asset_id", "lat", "lon", "object_type"}
        if not required.issubset(columns):
            raise ValueError("В реестре нужны колонки asset_id, lat, lon, object_type.")
        if any(columns.count(column) != 1 for column in required):
            raise ValueError("В реестре повторяется одна из колонок asset_id, lat, lon, object_type.")
        assets: list[RegistryAsset] = []
        seen: set[str] = set()
        for row in rows:
            asset_id = (row["asset_id"] or "").strip()
            coords = coordinate_pair(row["lat"], row["lon"])
            if not asset_id or coords is None:
                raise ValueError(f"Строка {rows.line_num} реестра: укажите ID и координаты объекта WGS84.")
            if asset_id in seen:
                raise ValueError(f"Строка {rows.line_num} реестра: повторяющийся ID {asset_id}.")
            seen.add(asset_id)
            assets.append(RegistryAsset(asset_id, *coords, (row["object_type"] or "").strip(),
                                        (row.get("address") or "").strip()))
    return AssetRegistry(assets)


@dataclass(frozen=True)
class Location:
    status: str
    source: str
    point: Point | None = None
    camera_point: Point | None = None
    matched_asset_id: str | None = None
    address: str | None = None
    reason: str | None = None
    candidates: tuple[Candidate, ...] = field(default_factory=tuple)

    @property
    def lat(self) -> float | None:
        return self.point[0] if self.point is not None else None

    @property
    def lon(self) -> float | None:
        return self.point[1] if self.point is not None else None

    @property
    def coordinates(self) -> str:
        if self.point is None:
            return "координаты не переданы"
        return f"{self.point[0]:.5f}, {self.point[1]:.5f}"

    def payload(self) -> dict:
        return {
            "status": self.status,
            "source": self.source,
            "reason": self.reason,
            "matched_asset_id": self.matched_asset_id,
            "candidates": [asdict(candidate) for candidate in self.candidates],
            "address": self.address,
            "camera_lat": self.camera_point[0] if self.camera_point else None,
            "camera_lon": self.camera_point[1] if self.camera_point else None,
        }


def resolve_location(
    record: Record,
    registry: AssetRegistry | None = None,
    radius_m: float = DEFAULT_LOCATION_RADIUS_M,
) -> Location:
    object_point = coordinate_pair(record.lat, record.lon)
    camera_point = coordinate_pair(record.extra.get("camera_lat"), record.extra.get("camera_lon"))
    object_type = _normalize_type(record.object_type) if record.object_type_explicit else ""
    official_id = record.asset_id if record.asset_id_origin == "provided" else None
    asset = registry.by_id.get(official_id) if registry is not None and official_id else None
    if asset:
        distance = haversine_m(object_point, asset.point) if object_point is not None else None
        registry_type = _normalize_type(asset.object_type)
        reason = None
        if distance is not None and distance > radius_m:
            reason = "registry_coordinates_conflict"
        elif object_type and registry_type and object_type != registry_type:
            reason = "registry_type_conflict"
        if reason:
            return Location("ambiguous", "registry", camera_point=camera_point, reason=reason,
                            candidates=(_candidate(asset, distance),))
        return Location("confirmed", "registry", asset.point, camera_point, asset.asset_id,
                        address=asset.address or None)

    if object_point is not None:
        return Location("reported", "object_coordinates", object_point, camera_point,
                        reason="asset_id_not_in_registry" if official_id and registry is not None else None)
    if official_id:
        return Location("reported", "asset_id", camera_point=camera_point,
                        reason="asset_id_not_in_registry" if registry is not None else None)
    if camera_point is None:
        return Location("unlocated", "none")
    if registry is None:
        return Location("unlocated", "camera_coordinates", camera_point=camera_point, reason="registry_required")
    candidates = registry.nearby(camera_point, radius_m, object_type)
    if not candidates:
        return Location("unlocated", "camera_coordinates", camera_point=camera_point, reason="no_nearby_asset")
    return Location("candidate" if len(candidates) == 1 else "ambiguous", "camera_coordinates",
                    camera_point=camera_point, candidates=candidates)
