import csv
import io
import json

from fastapi.testclient import TestClient

from inspector.web import create_app


def test_location_review_is_independent_and_reaches_exports(tmp_path):
    cards = [
        {
            "card_id": "camera-detection", "asset_id": "frame-001", "status": "finding",
            "review_status": "new", "lat": None, "lon": None,
            "location": {
                "status": "ambiguous", "source": "camera_coordinates",
                "camera_lat": 55.754, "camera_lon": 37.62,
                "candidates": [
                    {"asset_id": "W-17", "lat": 55.75403, "lon": 37.62018, "distance_m": 12.0},
                    {"asset_id": "W-18", "lat": 55.75411, "lon": 37.62021, "distance_m": 18.0},
                ],
            },
        },
        {"card_id": "legacy", "asset_id": "known", "status": "finding", "lat": 55.7, "lon": 37.6},
    ]
    (tmp_path / "cards.json").write_text(json.dumps(cards), encoding="utf-8")
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    with TestClient(create_app(tmp_path)) as client:
        features = client.get("/api/export.geojson").json()["features"]
        assert [feature["id"] for feature in features] == ["legacy"]
        assert client.post("/api/cards/camera-detection/location", json={"asset_id": "W-99"}).status_code == 422
        assert client.post("/api/cards/legacy/location", json={"asset_id": "W-17"}).status_code == 422

        response = client.post("/api/cards/camera-detection/location", json={"asset_id": "W-18"})
        assert response.status_code == 200
        card = response.json()["card"]
        assert card["location"]["source"] == "manual"
        assert card["location"]["matched_asset_id"] == "W-18"
        assert card["location"]["camera_lat"] == 55.754
        assert card["review_status"] == "new"
        assert card["asset_id"] == "frame-001"
        assert client.get("/api/summary").json()["location_status_counts"] == {"reported": 1, "confirmed": 1}
        assert client.post("/api/cards/camera-detection/decision", json={"decision": "confirmed"}).status_code == 200

    with TestClient(create_app(tmp_path)) as client:
        details = client.get("/api/cards/camera-detection").json()
        assert details["review_status"] == details["location"]["status"] == "confirmed"
        geojson = client.get("/api/export.geojson")
        assert geojson.headers["content-type"].startswith("application/geo+json")
        feature = geojson.json()["features"][0]
        assert feature["geometry"]["coordinates"] == [37.62021, 55.75411]
        assert feature["properties"]["matched_asset_id"] == "W-18"
        rows = list(csv.DictReader(io.StringIO(client.get("/api/export.csv").text.lstrip("\ufeff"))))
        assert rows[0]["location_source"] == "manual"
        assert rows[0]["matched_asset_id"] == "W-18"
        assert client.get("/api/export.json").json()[0]["location"]["confirmed_at"]
