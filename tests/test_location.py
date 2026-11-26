from __future__ import annotations

import csv
from dataclasses import replace

import pytest

from inspector.classifier import Prediction
from inspector.config import AppConfig
from inspector.location import haversine_m, load_registry, resolve_location
from inspector.manifest import Record, load_manifest
from inspector.pipeline import Observation, build_cards
from inspector.quality import Quality


@pytest.fixture
def registry(tmp_path):
    path = tmp_path / "registry.csv"
    path.write_text(
        "asset_id;lat;lon;object_type;address\n"
        "001;55,75;37,6202;well;Колодец у дома 1\n"
        "02;55,75;37,6203;pole;Опора у дома 1\n"
        "03;55,82;37,62;well;Колодец у дома 2\n",
        encoding="utf-8-sig",
    )
    return load_registry(path)


def _camera_record(tmp_path, **overrides):
    record = Record("a", tmp_path / "photo.jpg", "a", asset_id_origin="image",
                    extra={"camera_lat": "55.75", "camera_lon": "37.62"})
    return replace(record, **overrides)


def _observe(record, quality_score=1.0):
    quality = Quality(True, quality_score, 50, 128, 20, None, 640, 480, [])
    return Observation(record, quality, Prediction(.9, .1, {}, "open_hatch", "open_hatch"),
                       "finding", "open_hatch")


def test_registry_id_wins_over_camera_position(tmp_path, registry):
    record = _camera_record(tmp_path, asset_id="03", asset_id_origin="provided")
    location = resolve_location(record, registry)
    assert (location.status, location.source, location.matched_asset_id) == ("confirmed", "registry", "03")
    assert location.point == (55.82, 37.62)
    assert location.camera_point == (55.75, 37.62)
    assert location.address == "Колодец у дома 2"


def test_camera_uses_distance_and_only_explicit_type(tmp_path, registry):
    assert haversine_m((0.0, 179.9999), (0.0, -179.9999)) == pytest.approx(22.24, abs=0.05)
    record = _camera_record(tmp_path, asset_id="001")
    location = resolve_location(record, registry, radius_m=25)
    assert (location.status, location.source) == ("ambiguous", "camera_coordinates")
    assert location.point is None and location.matched_asset_id is None
    assert [entry.asset_id for entry in location.candidates] == ["001", "02"]
    assert location.candidates[0].distance_m == pytest.approx(12.5, abs=.1)

    explicit_type = replace(record, object_type="Опора", object_type_explicit=True)
    location = resolve_location(explicit_type, registry, radius_m=25)
    assert location.status == "candidate" and location.point is None
    assert [entry.asset_id for entry in location.candidates] == ["02"]

    outside = resolve_location(record, registry, radius_m=4)
    assert (outside.status, outside.reason, outside.candidates) == ("unlocated", "no_nearby_asset", ())


def test_reported_object_point_and_registry_conflict(tmp_path, registry):
    object_point = _camera_record(tmp_path, lat=55.81, lon=37.62)
    location = resolve_location(object_point, registry)
    assert (location.status, location.source, location.point) == ("reported", "object_coordinates", (55.81, 37.62))

    reported_id = replace(object_point, asset_id="001", asset_id_origin="provided")
    conflict = resolve_location(reported_id, registry)
    assert (conflict.status, conflict.source, conflict.reason) == ("ambiguous", "registry", "registry_coordinates_conflict")
    assert conflict.point is None and conflict.matched_asset_id is None
    assert [entry.asset_id for entry in conflict.candidates] == ["001"]


def test_camera_metadata_and_unknown_coordinates_do_not_publish_asset_point(tmp_path, registry):
    record = _camera_record(tmp_path)
    camera_only = resolve_location(record)
    assert (camera_only.status, camera_only.source, camera_only.reason) == (
        "unlocated", "camera_coordinates", "registry_required",
    )
    assert camera_only.point is None

    invalid = resolve_location(replace(record, lat=90.5, lon=37.62, extra={"camera_lat": "NaN", "camera_lon": "37.62"}),
                               registry)
    assert (invalid.status, invalid.source, invalid.point) == ("unlocated", "none", None)

    unmapped_id = replace(record, asset_id="not_in_registry", asset_id_origin="provided")
    reported = resolve_location(unmapped_id, registry)
    assert (reported.status, reported.source, reported.reason) == (
        "reported", "asset_id", "asset_id_not_in_registry",
    )
    assert reported.point is None and reported.candidates == ()


def test_intake_image_ids_do_not_merge_with_official_ids_or_each_other(tmp_path, registry):
    image = tmp_path / "photo.jpg"
    image.touch()
    with (tmp_path / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows([
            ["image", "image_id", "asset_id", "id_origin", "object_type"],
            [image.name, "verified", "001", "provided", "well"],
            [image.name, "001", "", "image", ""],
            [image.name, "001__a", "", "image", ""],
            [image.name, "001__b", "", "image", ""],
        ])
    records = load_manifest(tmp_path)
    observations = [_observe(record) for record in records]
    for observation in observations:
        observation.location = resolve_location(observation.record, registry)
    cards = build_cards(observations, AppConfig(), .5)
    assert len(cards) == 4 and len({card["card_id"] for card in cards}) == 4
    assert {card["image_id"] for card in cards if card["location"]["status"] == "confirmed"} == {"verified"}


def test_multiple_views_keep_location_evidence_and_escalate_coordinate_conflicts(tmp_path, registry):
    best = _observe(Record("best", tmp_path / "photo.jpg", "001"))
    located = _observe(Record("angle", tmp_path / "photo.jpg", "001", lat=55.75, lon=37.6202), .8)
    card, = build_cards([best, located], AppConfig(), .5)
    assert card["image_id"] == "best" and card["location"]["evidence_image_id"] == "angle"
    assert card["lat"] == 55.75 and card["lon"] == 37.6202

    best.location = resolve_location(best.record, registry)
    located.record = replace(located.record, lat=55.81)
    located.location = resolve_location(located.record, registry)
    card, = build_cards([best, located], AppConfig(), .5)
    assert card["location"]["reason"] == "registry_coordinates_conflict"
    assert (card["lat"], card["lon"], card["frames_count"]) == (None, None, 2)


def test_registry_invalid_or_duplicate_id_fails_early(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("asset_id,lat,lon,object_type\n01,55,37,well\n01,55,37,well\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Строка 3.*повторяющийся ID 01"):
        load_registry(path)
    path.write_text("asset_id,lat,lon,object_type\n01,0,360,well\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Строка 2.*WGS84"):
        load_registry(path)
