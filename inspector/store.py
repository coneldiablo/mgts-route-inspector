"""Review decisions stored locally; no external service and no hidden state."""
from __future__ import annotations

import csv
import io
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VALID_DECISIONS = {"confirmed", "rejected", "rephoto_requested", "new"}
MATCHABLE_LOCATIONS = {"candidate", "ambiguous"}


class ReviewStore:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / "reviews.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reviews (
                    card_id TEXT PRIMARY KEY,
                    decision TEXT NOT NULL,
                    note TEXT DEFAULT '',
                    reviewer TEXT DEFAULT 'master',
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS location_reviews (
                    card_id TEXT PRIMARY KEY,
                    asset_id TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def _conn(self):
        return sqlite3.connect(self.path)

    def upsert(self, card_id: str, decision: str, note: str = "", reviewer: str = "master") -> dict[str, Any]:
        if decision not in VALID_DECISIONS:
            raise ValueError(f"Недопустимое решение: {decision}")
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO reviews(card_id, decision, note, reviewer, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(card_id) DO UPDATE SET
                    decision=excluded.decision,
                    note=excluded.note,
                    reviewer=excluded.reviewer,
                    updated_at=excluded.updated_at
                """,
                (card_id, decision, note, reviewer, now),
            )
        return {"card_id": card_id, "decision": decision, "note": note, "reviewer": reviewer, "updated_at": now}

    def all(self) -> dict[str, dict[str, Any]]:
        with self._conn() as conn:
            cur = conn.execute("SELECT card_id, decision, note, reviewer, updated_at FROM reviews")
            return {
                row[0]: {
                    "decision": row[1],
                    "note": row[2],
                    "reviewer": row[3],
                    "updated_at": row[4],
                }
                for row in cur.fetchall()
            }

    def assign_location(self, card_id: str, asset_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO location_reviews(card_id, asset_id, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(card_id) DO UPDATE SET
                    asset_id=excluded.asset_id,
                    updated_at=excluded.updated_at
                """,
                (card_id, asset_id, now),
            )

    def location_assignments(self) -> dict[str, dict[str, str]]:
        with self._conn() as conn:
            rows = conn.execute("SELECT card_id, asset_id, updated_at FROM location_reviews")
            return {
                card_id: {"asset_id": asset_id, "updated_at": updated_at}
                for card_id, asset_id, updated_at in rows
            }

    def count_by_decision(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for value in self.all().values():
            counts[value["decision"]] = counts.get(value["decision"], 0) + 1
        return counts


def object_point(card: dict[str, Any]) -> tuple[float, float] | None:
    lat, lon = card.get("lat"), card.get("lon")
    if not all(isinstance(n, (int, float)) and not isinstance(n, bool) for n in (lat, lon)):
        return None
    if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return lat, lon


def card_location_status(card: dict[str, Any]) -> str:
    status = (card.get("location") or {}).get("status")
    if status:
        return status
    return "reported" if object_point(card) else "unlocated"


def location_candidate(card: dict[str, Any], asset_id: str) -> dict[str, Any] | None:
    location = card.get("location") or {}
    if location.get("status") not in MATCHABLE_LOCATIONS:
        return None
    return next(
        (item for item in location.get("candidates", [])
         if item.get("asset_id") == asset_id and object_point(item) is not None),
        None,
    )


def merge_reviews(
    cards: list[dict[str, Any]],
    reviews: dict[str, dict[str, Any]],
    location_assignments: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    out = []
    for card in cards:
        item = dict(card)
        review = reviews.get(card["card_id"])
        if review:
            item["review_status"] = review["decision"]
            item["review_note"] = review.get("note", "")
            item["review_updated_at"] = review.get("updated_at")
        assignment = (location_assignments or {}).get(card["card_id"])
        if assignment and (candidate := location_candidate(card, assignment["asset_id"])):
            lat, lon = candidate["lat"], candidate["lon"]
            item["lat"], item["lon"] = lat, lon
            item["coordinates"] = f"{lat:.5f}, {lon:.5f}"
            item["location"] = {
                **card["location"],
                "status": "confirmed",
                "source": "manual",
                "matched_asset_id": candidate["asset_id"],
                "address": candidate.get("address"),
                "confirmed_at": assignment["updated_at"],
            }
        out.append(item)
    return out


def export_csv(cards: list[dict[str, Any]]) -> str:
    fields = [
        "queue_position", "card_id", "asset_id", "object_type", "status", "finding_short",
        "p_finding", "threshold", "coordinates", "captured_at", "image_id",
        "action", "why", "review_status", "review_note", "duplicates_merged",
        "model_version", "lat", "lon", "location_status", "location_source", "matched_asset_id",
        "address", "location_confirmed_at",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for card in cards:
        location = card.get("location") or {}
        writer.writerow({
            **card,
            "location_status": card_location_status(card),
            "location_source": location.get("source") or ("object_coordinates" if object_point(card) else "none"),
            "matched_asset_id": location.get("matched_asset_id"),
            "address": location.get("address"),
            "location_confirmed_at": location.get("confirmed_at"),
        })
    return buffer.getvalue()


def export_json(cards: list[dict[str, Any]]) -> str:
    return json.dumps(cards, ensure_ascii=False, indent=2)


def export_geojson(cards: list[dict[str, Any]]) -> dict[str, Any]:
    features = []
    for card in cards:
        status = card_location_status(card)
        point = object_point(card)
        if point is None or status not in {"confirmed", "reported"}:
            continue
        lat, lon = point
        location = card.get("location") or {}
        properties = {key: card.get(key) for key in (
            "card_id", "asset_id", "object_type", "status", "finding_short", "action", "review_status", "captured_at",
        )}
        properties.update(
            location_status=status,
            location_source=location.get("source") or "object_coordinates",
            matched_asset_id=location.get("matched_asset_id"),
            address=location.get("address"),
        )
        features.append({
            "type": "Feature",
            "id": card["card_id"],
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": properties,
        })
    return {"type": "FeatureCollection", "features": features}
